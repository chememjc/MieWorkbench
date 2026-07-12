# =============================================================================
# analysis_imaging.py — exit-pupil / chief-ray stage + field-based imaging
# metrics (distortion, telecentricity, tangential/sagittal best focus,
# PSF-peak Strehl). Numpy-only, float64 — no torch, no Qt, no h5py.
#
# Everything here consumes ONLY arrays already exported to rays_full.npz
# (run_trace.write_rays_full: pos/dir/opl/lam/power/source_id/birth_pos per
# detector) plus, at the post_process level, the per-(source, detector)
# scalar powers from case.json["detected"]. No new trace-time physics.
#
# Pupil model (v2 — the future.md "exit-pupil/chief-ray search stage"):
# each field point (= source, the fan convention core.wizards.
# design_field_fan builds) contributes one power-weighted CENTROID RAY
# (mean landing point, mean incoming direction) at the detector; the exit
# pupil center E is the least-squares intersection point of those rays'
# backward extensions. The chief ray of a field bundle is the traced ray
# whose line passes closest to E (refined by re-centroiding the closest
# few percent), the reference sphere is centered on the chief landing
# point with radius |E - chief landing|, and pupil coordinates are ray
# intersections with the plane through E perpendicular to the ON-AXIS
# bundle's centroid direction.
#
# Honest limits:
#   * >= 2 field points (sources) with distinct chief directions are
#     needed to locate E; a single bundle (or an image-side telecentric
#     system, where every chief ray is parallel and E is at infinity)
#     degenerates — exit_pupil_center() then returns (None, reason) and
#     callers fall back to the source-referenced pupil (analysis.py's v1
#     model) / the bundle centroid chief, with the reason reported.
#   * The OPD generalizes analysis.opd_from_rays' r0 form to a reference
#     sphere centered on the CHIEF landing point: W_i = opl_i +
#     n_amb*|hit_i - C_chief| - (chief value). For a sphere centered at C
#     the sphere radius contributes the same constant n_amb*R to every
#     ray, so referencing to C is identical to referencing to the sphere
#     — exact in the same near-image approximation opd_from_rays makes
#     (each ray propagated STRAIGHT to C from its landing point).
#   * best_focus_scan treats post-detector propagation as straight lines
#     from the last traced segment (pos + t*dir) — valid because the
#     detector is the final surface in these scenes.
# =============================================================================
import math

import numpy as np


# ---------------------------------------------------------------------------
# field bundles
# ---------------------------------------------------------------------------
_GROUP_KEYS = ("pos", "dir", "opl", "lam", "power", "birth_pos")


def field_groups(cols, min_rays=1):
    """Group one detector's exported ray columns by source_id (= field
    point). `cols` is the per-detector dict post_process.render_ray_analysis
    builds from rays_full.npz (needs pos/dir/opl/lam/power/source_id;
    birth_pos rides along when present). Returns {source_id: {col: array}}
    for the sources with at least `min_rays` rays, ascending source_id."""
    sid = np.asarray(cols["source_id"]).astype(int)
    out = {}
    for s in sorted(set(sid.tolist())):
        m = sid == s
        if int(m.sum()) < min_rays:
            continue
        out[s] = {k: np.asarray(cols[k])[m] for k in _GROUP_KEYS
                  if k in cols}
    return out


def centroid_ray(group):
    """Power-weighted centroid ray of one field bundle: (mean landing
    point on the detector (3,), mean incoming unit direction (3,))."""
    pw = np.asarray(group["power"], dtype=np.float64)
    wsum = float(np.sum(pw)) or 1.0
    c = (pw[:, None] * group["pos"]).sum(axis=0) / wsum
    d = (pw[:, None] * group["dir"]).sum(axis=0) / wsum
    nd = float(np.linalg.norm(d))
    if nd <= 0:
        raise ValueError("field bundle has a zero mean direction "
                         "(isotropic arrival?) — no centroid ray")
    return c, d / nd


# ---------------------------------------------------------------------------
# exit pupil
# ---------------------------------------------------------------------------
def exit_pupil_center(centroid_rays, cond_tol=1e-6):
    """Least-squares exit-pupil center E from the field bundles' centroid
    rays: the point minimizing sum_s dist^2(E, line(C_s, d_s)) over field
    points, where each line is the landing point extended (backward)
    along the incoming direction. Closed form: the 3x3 normal equations
    sum_s (I - d d^T) E = sum_s (I - d d^T) C.

    centroid_rays: iterable of (point (3,), unit_dir (3,)) as returned by
    centroid_ray().

    Returns (E (3,), None) on success, (None, reason_str) when the solve
    is degenerate: fewer than 2 rays, or (nearly) parallel chief rays
    (rank-deficient normal matrix — image-side telecentric, exit pupil at
    infinity). Callers fall back to the source-referenced pupil and report
    the reason."""
    rays = list(centroid_rays)
    if len(rays) < 2:
        return None, ("need >= 2 field bundles to triangulate the exit "
                      "pupil (got %d) — falling back to the source-"
                      "referenced pupil" % len(rays))
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d in rays:
        p = np.asarray(p, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ p
    w = np.linalg.eigvalsh(A)
    if w[0] <= cond_tol * max(w[-1], 1e-300):
        return None, ("field chief rays are (nearly) parallel — exit "
                      "pupil at infinity (image-side telecentric?); "
                      "falling back to the source-referenced pupil")
    return np.linalg.solve(A, b), None


def _line_point_dist(E, pos, dirs):
    """Distance from point E to each ray line pos_i + t*dir_i (t signed —
    the backward extension counts)."""
    d = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    v = E[None, :] - pos
    along = np.sum(v * d, axis=-1)
    perp = v - along[:, None] * d
    return np.linalg.norm(perp, axis=-1)


def chief_ray(group, E, normal, frac=0.05, min_refine=10):
    """Chief ray of one field bundle: the traced ray whose (backward-
    extended) line passes closest to the exit-pupil center E, refined once
    by power-weighted re-centroiding over the closest `frac` of rays by
    that distance. E=None (degenerate pupil solve) falls back to the
    bundle's centroid ray, flagged method='centroid'.

    Returns dict:
      landing (3,)   chief landing point on the detector = image height
                     vector origin (h_s = |landing - axis landing|)
      dir (3,)       chief incoming unit direction
      cra_deg        chief-ray angle to the detector normal, degrees in
                     [0, 90] (0 = telecentric)
      opl            chief optical path length at the landing point (same
                     power-weighted subset average as landing/dir)
      method         'exit_pupil' | 'centroid'
      n_used         rays in the refinement subset
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    pos = np.asarray(group["pos"], dtype=np.float64)
    dirs = np.asarray(group["dir"], dtype=np.float64)
    opl = np.asarray(group["opl"], dtype=np.float64)
    pw = np.asarray(group["power"], dtype=np.float64)
    if E is None:
        c, d = centroid_ray(group)
        o = float(np.sum(pw * opl) / (float(np.sum(pw)) or 1.0))
        method, n_used = "centroid", len(pos)
    else:
        E = np.asarray(E, dtype=np.float64)
        dist = _line_point_dist(E, pos, dirs)
        k = max(min_refine, int(math.ceil(frac * len(pos))))
        k = min(k, len(pos))
        idx = np.argsort(dist)[:k]
        w = pw[idx]
        wsum = float(np.sum(w)) or 1.0
        c = (w[:, None] * pos[idx]).sum(axis=0) / wsum
        d = (w[:, None] * dirs[idx]).sum(axis=0) / wsum
        d = d / np.linalg.norm(d)
        o = float(np.sum(w * opl[idx]) / wsum)
        method, n_used = "exit_pupil", int(k)
    cosang = abs(float(np.dot(d, normal)))
    cra = math.degrees(math.acos(min(1.0, max(-1.0, cosang))))
    return {"landing": c, "dir": d, "cra_deg": cra, "opl": o,
            "method": method, "n_used": n_used}


# ---------------------------------------------------------------------------
# pupil plane + coordinates
# ---------------------------------------------------------------------------
def _basis_perp(axis, prefer):
    """Unit vector = the component of `prefer` perpendicular to `axis`
    (None when degenerate)."""
    w = np.asarray(prefer, dtype=np.float64)
    w = w - float(np.dot(w, axis)) * axis
    n = float(np.linalg.norm(w))
    if n < 1e-9:
        return None
    return w / n


def pupil_plane(E, axis_group, prefer_u=(0.0, 0.0, 1.0),
                fallback_u=(0.0, 1.0, 0.0)):
    """Pupil sampling plane: through E, perpendicular to the ON-AXIS field
    bundle's centroid direction. The normalization radius is the on-axis
    bundle's RMS in-plane radius * sqrt(2) (a uniform disc of radius R has
    RMS radius R/sqrt(2), so this estimates the marginal-ray edge).

    Returns dict {origin, normal, uhat, vhat, radius_m} or raises
    ValueError when the on-axis bundle cannot span a pupil (all rays
    parallel to the plane, or zero radius)."""
    E = np.asarray(E, dtype=np.float64)
    _, axis_dir = centroid_ray(axis_group)
    u = _basis_perp(axis_dir, prefer_u)
    if u is None:
        u = _basis_perp(axis_dir, fallback_u)
    if u is None:
        raise ValueError("degenerate pupil basis (axis direction parallel "
                         "to both preferred transverse references)")
    v = np.cross(axis_dir, u)
    plane = {"origin": E, "normal": axis_dir, "uhat": u, "vhat": v,
             "radius_m": 1.0}
    xy, ok = pupil_coords(axis_group, plane, normalized=False)
    if not np.any(ok):
        raise ValueError("no on-axis ray intersects the pupil plane")
    r2 = np.sum(xy[ok] ** 2, axis=-1)
    pw = np.asarray(axis_group["power"], dtype=np.float64)[ok]
    wsum = float(np.sum(pw)) or 1.0
    radius = math.sqrt(float(np.sum(pw * r2)) / wsum) * math.sqrt(2.0)
    if radius <= 0:
        raise ValueError("on-axis pupil footprint has zero radius")
    plane["radius_m"] = radius
    return plane


def pupil_coords(group, plane, normalized=True):
    """Intersect each ray's (backward-extended) line with the pupil plane;
    return (pupil_xy (N,2), valid (N,) bool). Coordinates are relative to
    the pupil center E in the plane's (uhat, vhat) basis, divided by the
    normalization radius when `normalized` (so the on-axis bundle spans
    ~ the unit disc). Rays parallel to the plane are masked invalid."""
    O = plane["origin"]
    n = plane["normal"]
    pos = np.asarray(group["pos"], dtype=np.float64)
    dirs = np.asarray(group["dir"], dtype=np.float64)
    denom = dirs @ n
    ok = np.abs(denom) > 1e-12
    t = np.zeros(len(pos))
    t[ok] = ((O - pos[ok]) @ n) / denom[ok]
    X = pos + t[:, None] * dirs
    rel = X - O[None, :]
    xy = np.stack([rel @ plane["uhat"], rel @ plane["vhat"]], axis=1)
    if normalized:
        xy = xy / plane["radius_m"]
    return xy, ok


# ---------------------------------------------------------------------------
# OPD at the exit pupil + PSF-peak Strehl
# ---------------------------------------------------------------------------
def opd_exit_pupil(group, chief, E=None, n_ambient=1.0):
    """Chief-referenced OPD (metres) per ray against the reference sphere
    centered at the chief landing point C with radius |E - C| (see module
    header: the radius contributes an identical constant to every ray, so
    the r0 form of analysis.opd_from_rays generalizes directly with
    r0 = C and the chief's own OPL as the reference value):

        W_i = opl_i + n_ambient * |hit_i - C| - opl_chief

    `E` is accepted for API symmetry/documentation (it fixes the sphere)
    but does not enter the difference."""
    C = np.asarray(chief["landing"], dtype=np.float64)
    pos = np.asarray(group["pos"], dtype=np.float64)
    opl = np.asarray(group["opl"], dtype=np.float64)
    total = opl + n_ambient * np.linalg.norm(pos - C, axis=-1)
    return total - float(chief["opl"])


def strehl_psf_peak(pupil_xy, opd, amplitudes, lam):
    """Amplitude-weighted PSF-peak-ratio Strehl estimate,

        S = |sum_j a_j exp(i 2 pi W_j / lam_j)|^2 / (sum_j a_j)^2,

    i.e. the on-axis coherent sum over the pupil samples relative to the
    aberration-free (W == 0) sum — each traced ray is one pupil sample, so
    the ray density already carries the pupil apodization and no explicit
    area weights are needed. Pass piston/tip/tilt-REMOVED opd (tilt merely
    displaces the PSF peak; leaving it in would wrongly count it as a
    Strehl loss). `pupil_xy` is accepted for API symmetry (the samples'
    pupil locations) but the on-axis sum needs only W and a. `lam` may be
    scalar or per-ray."""
    opd = np.asarray(opd, dtype=np.float64)
    a = np.asarray(amplitudes, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    denom = float(np.sum(a)) ** 2
    if denom <= 0:
        return 0.0
    phase = 2.0 * np.pi * opd / lam
    field = np.sum(a * np.exp(1j * phase))
    return float(np.abs(field) ** 2 / denom)


# ---------------------------------------------------------------------------
# tangential / sagittal best focus (field curvature, astigmatism)
# ---------------------------------------------------------------------------
def best_focus_scan(group, chief, normal, z_range=None,
                    yhat=(0.0, 1.0, 0.0)):
    """Tangential/sagittal best-focus positions along the chief ray.

    Rays propagate as straight lines past the detector (pos + t*dir, the
    last traced segment). The bundle is split by pupil azimuth about the
    chief: the tangential (meridional) direction u_T is the detector
    normal's component perpendicular to the chief direction (the plane
    containing chief ray and detector normal); for the on-axis case
    (chief parallel to the normal) u_T falls back to `yhat`'s transverse
    component (the y-plane). A ray joins the T family when its transverse
    direction leans more along u_T than along u_S = chief x u_T, else S.

    For each family the power-weighted RMS transverse spread of the
    family's own component (T rays' u_T spread, S rays' u_S spread) is an
    exact quadratic in the signed defocus z along the chief direction
    (transverse position is affine in z), so the minimum is closed-form —
    `z_range` (z_min_m, z_max_m), when given, only clamps the result.

    Returns dict {z_t_m, z_s_m, astig_m (= z_t - z_s), rms_t_m, rms_s_m,
    n_t, n_s}; a family with < 3 usable rays yields NaN entries."""
    c_dir = np.asarray(chief["dir"], dtype=np.float64)
    c_dir = c_dir / np.linalg.norm(c_dir)
    C = np.asarray(chief["landing"], dtype=np.float64)
    u_t = _basis_perp(c_dir, np.asarray(normal, dtype=np.float64))
    if u_t is None:
        u_t = _basis_perp(c_dir, np.asarray(yhat, dtype=np.float64))
    if u_t is None:
        raise ValueError("cannot build a meridional axis (chief parallel "
                         "to both the detector normal and yhat)")
    u_s = np.cross(c_dir, u_t)

    pos = np.asarray(group["pos"], dtype=np.float64)
    dirs = np.asarray(group["dir"], dtype=np.float64)
    pw = np.asarray(group["power"], dtype=np.float64)
    denom = dirs @ c_dir
    ok = np.abs(denom) > 1e-9
    pos, dirs, pw, denom = pos[ok], dirs[ok], pw[ok], denom[ok]

    a = pos - C[None, :]
    a_par = a @ c_dir
    g = dirs / denom[:, None]                     # d(point)/dz along chief
    # transverse position at defocus z:  x(z) = A + z * B  (per component)
    gt, gs = g @ u_t, g @ u_s
    At = a @ u_t - a_par * gt
    As = a @ u_s - a_par * gs
    tan_family = np.abs(gt) >= np.abs(gs)

    def _closed_form(m, A, B):
        if int(m.sum()) < 3:
            return float("nan"), float("nan"), int(m.sum())
        w = pw[m]
        wsum = float(np.sum(w)) or 1.0
        Ac = A[m] - float(np.sum(w * A[m])) / wsum
        Bc = B[m] - float(np.sum(w * B[m])) / wsum
        bb = float(np.sum(w * Bc * Bc))
        if bb <= 0:
            return float("nan"), float("nan"), int(m.sum())
        z = -float(np.sum(w * Ac * Bc)) / bb
        if z_range is not None:
            z = min(max(z, float(z_range[0])), float(z_range[1]))
        rms = math.sqrt(max(0.0, float(
            np.sum(w * (Ac + z * Bc) ** 2)) / wsum))
        return z, rms, int(m.sum())

    z_t, rms_t, n_t = _closed_form(tan_family, At, gt)
    z_s, rms_s, n_s = _closed_form(~tan_family, As, gs)
    return {"z_t_m": z_t, "z_s_m": z_s, "astig_m": z_t - z_s,
            "rms_t_m": rms_t, "rms_s_m": rms_s, "n_t": n_t, "n_s": n_s}


# ---------------------------------------------------------------------------
# field angles + f_eff calibration + distortion polynomial (pure math the
# post renderers share; nothing here reads files)
# ---------------------------------------------------------------------------
def source_directions_from_model(model):
    """{source label: emission unit direction} from a model.json dict: the
    emit face's canonical normal (plane faces) or fingerprint normal_hint
    (curved emitters), SIGN-agnostic (callers compare directions through
    |dot|, so the toward-origin emission sign policy does not matter).
    Sources whose emit face carries no usable normal are omitted."""
    out = {}
    for b in model.get("bodies", []):
        src = b.get("source")
        if not isinstance(src, dict):
            continue
        fid = src.get("emit_face")
        d = None
        for f in b.get("faces", []):
            if f.get("id") != fid:
                continue
            surf = f.get("surface") or {}
            if surf.get("type") == "plane" and surf.get("normal"):
                d = surf["normal"]
            elif (f.get("fingerprint") or {}).get("normal_hint"):
                d = f["fingerprint"]["normal_hint"]
            break
        if d is None:
            continue
        d = np.asarray(d, dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n > 0:
            out[b.get("label", b.get("name", ""))] = d / n
    return out


def field_angle_annotations_from_model(model):
    """{source label: field_angle_deg} for sources whose extracted record
    carries the (optional) `field_angle_deg` annotation — the
    core.wizards.design_field_fan body property, once extract_geometry
    echoes it into the source dict. Empty until that echo lands (the
    direction-derived fallback below covers un-annotated scenes)."""
    out = {}
    for b in model.get("bodies", []):
        src = b.get("source")
        if isinstance(src, dict) and src.get("field_angle_deg") is not None:
            out[b.get("label", b.get("name", ""))] = float(
                src["field_angle_deg"])
    return out


def pick_axis_source(sids, labels, annotations, source_dirs, det_normal,
                     chiefs=None):
    """The on-axis field point among source ids `sids`, by priority:
      1. smallest |field_angle_deg| annotation (when every sid has one);
      2. source emission direction most parallel to the detector normal
         (|dot|, from source_dirs = source_directions_from_model);
      3. chief/centroid incoming direction most parallel to the detector
         normal (post-optics — a last resort, wrong for telecentric
         images, hence the priority order).
    labels maps sid -> source label (case.json['sources'] order)."""
    sids = list(sids)
    ann = {s: annotations.get(labels.get(s)) for s in sids}
    if all(a is not None for a in ann.values()) and sids:
        return min(sids, key=lambda s: abs(ann[s]))
    n = np.asarray(det_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    dirs = {s: source_dirs.get(labels.get(s)) for s in sids}
    if all(d is not None for d in dirs.values()) and sids:
        return max(sids, key=lambda s: abs(float(np.dot(dirs[s], n))))
    if chiefs:
        return max(sids,
                   key=lambda s: abs(float(np.dot(chiefs[s]["dir"], n))))
    return sids[0]


def field_angles_deg(sids, axis_sid, labels, annotations, source_dirs):
    """{sid: field angle theta_s in degrees} — the annotation when
    present, else the angle between the source's emission direction and
    the AXIS source's (|dot|; sign-agnostic, distortion/vignetting are
    radial). Sources with neither yield None."""
    out = {}
    d0 = source_dirs.get(labels.get(axis_sid))
    for s in sids:
        a = annotations.get(labels.get(s))
        if a is not None:
            out[s] = abs(float(a))
            continue
        d = source_dirs.get(labels.get(s))
        if d is None or d0 is None:
            out[s] = None
            continue
        dot = min(1.0, abs(float(np.dot(d, d0))))
        out[s] = math.degrees(math.acos(dot))
    return out


def grid_distortion(theta_deg, h_m):
    """Radial distortion vs. a paraxial reference from measured field
    points: theta_deg/h_m are matched arrays of NON-axis field angles
    (deg, > 0) and real image heights |chief_s - chief_axis| (m).

    f_eff calibration (standard practice, documented limitation): the
    reference focal length is fit through the SMALLEST-angle field point,
    f_eff = h_min / tan(theta_min) — the innermost point is treated as
    distortion-free, so reported distortion is relative to it (any true
    distortion already present at theta_min biases f_eff by the same
    factor; keep the innermost field angle small).

    Returns dict {f_eff_m, rows: [(theta_deg, h_m, h_ref_m,
    distortion_pct)...] sorted by theta, poly: (r_max_m, [k2, k4...])}
    where distortion_pct(h_ref) ~= 100 * sum_j k_j (h_ref/r_max)^(2j).
    Needs >= 2 field points for a nonzero polynomial (with exactly one
    the single point IS the calibration and distortion is identically 0)."""
    th = np.asarray(theta_deg, dtype=np.float64)
    h = np.asarray(h_m, dtype=np.float64)
    if len(th) == 0 or len(th) != len(h):
        raise ValueError("grid_distortion needs matched non-empty "
                         "theta/h arrays")
    order = np.argsort(th)
    th, h = th[order], h[order]
    t = np.tan(np.radians(th))
    if t[0] <= 0 or h[0] <= 0:
        raise ValueError("field angles/heights must be positive "
                         "(axis point excluded)")
    f_eff = float(h[0] / t[0])
    h_ref = f_eff * t
    dist_pct = 100.0 * (h - h_ref) / h_ref
    rows = [(float(a), float(b), float(c), float(d))
            for a, b, c, d in zip(th, h, h_ref, dist_pct)]
    r_max = float(np.max(h_ref))
    x = h_ref / r_max
    n_terms = min(2, max(1, len(th) - 1))
    ks = [0.0] * n_terms
    if len(th) >= 2:
        # LSQ fit dist_pct/100 = sum_j k_j x^(2j) over the field points
        A = np.stack([x ** (2 * (j + 1)) for j in range(n_terms)], axis=1)
        sol, *_ = np.linalg.lstsq(A, dist_pct / 100.0, rcond=None)
        ks = [float(v) for v in sol]
    return {"f_eff_m": f_eff, "rows": rows, "poly": (r_max, ks)}


def distortion_map_radius(r, poly):
    """Apply the fitted radial distortion polynomial: r -> r * (1 + D(r)),
    D(r) = sum_j k_j (r/r_max)^(2j). Vectorized; used to deform the
    synthetic square grid in the distortion visual."""
    r_max, ks = poly
    r = np.asarray(r, dtype=np.float64)
    x = r / r_max
    D = np.zeros_like(r)
    for j, k in enumerate(ks, start=1):
        D += k * x ** (2 * j)
    return r * (1.0 + D)
