# =============================================================================
# sources.py — light-source sampling.
#
# Emitting face: the face named in the contract's source.emit_face (default:
# closest-to-origin, chosen at extract time; CLI-overridable upstream).
# Sampling is stratified-jittered in the face's canonical UV, rejected
# against the trim, so per-sample area weights dA are uniform-in-area over
# the face (needed as the coherent-gather quadrature weight).
#
# Emission direction ("light emitted toward the origin only"):
#   * planar face  -> collimated along the face normal, sign chosen so the
#     beam heads toward the origin hemisphere (per-face).
#   * curved face  -> local surface normal per sample, sign chosen per
#     sample toward the origin; a mixed-sign face triggers a loud warning
#     and the against-origin samples are dropped with their power credited
#     to the 'emission_clipped' audit bucket.
#
# Wavelengths (nm params from the contract):
#   * neither lambdamin nor lambdamax  -> monochromatic at lambdac
#   * both        -> asymmetric Gaussian: sigma- = lambdac - lambdamin,
#                    sigma+ = lambdamax - lambdac (each side a half-normal,
#                    side chosen with probability sigma/(sigma-+sigma+))
#   * exactly one -> uniform on [lambdac - w, lambdac + w] where w is the
#                    given half-width (spec: "uniformly distributed around
#                    the center wavelength")
# Sampling is STRATIFIED: n_lambda equal-probability quantiles, one
# deterministic lambda per stratum, equal power weights. Rays are assigned
# a stratum id so the gather keeps per-stratum coherent accumulators
# (different optical frequencies never interfere stationarily).
#
# Phase reference: opl = 0 on the emitting surface — the surface IS the
# source wavefront (plane wave for a flat laser, sphere wave for the
# divergent laser). 'coherent' sources get zero initial phase; incoherent
# sources get a uniform random phase per ray (fringe visibility ~ 0).
#
# Polarization (contract source.polarization, parsed dict from
# common.parse_polarization_spec; absent -> unpolarized):
#   * unpolarized -> TWO mutually-incoherent orthogonal populations
#     (pol_stratum 0/1), rays alternating between them; the gather keeps
#     per-(source, lam, pol) accumulators so the populations never
#     interfere. This is exact for polarizer/retarder chains (Malus etc.),
#     unlike the old equal-split single Jones vector (which was really
#     45-degree linear light).
#   * linear:<deg> / circular:left|right / elliptical:<psi>:<chi> -> one
#     population with the exact Jones vector.
# Angle reference frame per ray: e_ref = global +z projected transverse
# to the emission direction (fallback +y when emitting along z);
# e_perp = dir x e_ref. s_hat is set to e_ref so (Es, Ep) IS the Jones
# vector in that frame. linear:<deg> rotates from e_ref toward e_perp.
# Circular handedness: 'right' means the E-vector rotates clockwise as
# seen by an observer facing the ONCOMING beam (optics/Hecht convention);
# with the field convention Re[E exp(-i w t)] and p_hat = dir x s_hat
# that is Jones (1, +i)/sqrt(2) in the (e_ref, e_perp) basis.
# Elliptical (psi, chi): standard orientation/ellipticity angles,
# E = (cos psi cos chi - i sin psi sin chi,
#      sin psi cos chi + i cos psi sin chi).
# =============================================================================
import numpy as np

from .rays import RayBatch


def n_pol_strata(src):
    """Number of mutually-incoherent polarization populations a source
    emits: 2 for unpolarized (the default), 1 for any explicit state."""
    pol = src.get("polarization") or {"kind": "unpolarized"}
    return 2 if pol.get("kind", "unpolarized") == "unpolarized" else 1


def _pol_reference_frame(dirs):
    """Per-ray transverse reference frame: e_ref = z projected transverse
    (fallback y when |z x dir| ~ 0), e_perp = dir x e_ref."""
    z = np.array([0.0, 0.0, 1.0])
    y = np.array([0.0, 1.0, 0.0])
    ref = z - np.sum(dirs * z, axis=-1, keepdims=True) * dirs
    nrm = np.linalg.norm(ref, axis=-1)
    fallback = nrm < 1e-9
    if np.any(fallback):
        alt = y - np.sum(dirs[fallback] * y, axis=-1, keepdims=True) \
            * dirs[fallback]
        ref[fallback] = alt
        nrm = np.linalg.norm(ref, axis=-1)
    e_ref = ref / nrm[:, None]
    e_perp = np.cross(dirs, e_ref)
    return e_ref, e_perp


def jones_for(pol, pol_stratum):
    """Unit Jones vector (Es, Ep) complex pair in the (e_ref, e_perp)
    basis for a polarization dict + stratum index. |Es|^2+|Ep|^2 = 1."""
    kind = (pol or {"kind": "unpolarized"}).get("kind", "unpolarized")
    if kind == "unpolarized":
        # two orthogonal fully-polarized populations of equal power
        return (1.0 + 0j, 0j) if pol_stratum == 0 else (0j, 1.0 + 0j)
    if kind == "linear":
        th = np.deg2rad(pol["angle_deg"])
        return (np.cos(th) + 0j, np.sin(th) + 0j)
    if kind == "circular":
        # 'right' = clockwise facing the oncoming beam (module header)
        s = 1.0 if pol["handedness"] == "right" else -1.0
        return (1.0 / np.sqrt(2) + 0j, s * 1j / np.sqrt(2))
    if kind == "elliptical":
        psi = np.deg2rad(pol["psi_deg"])
        chi = np.deg2rad(pol["chi_deg"])
        return (np.cos(psi) * np.cos(chi) - 1j * np.sin(psi) * np.sin(chi),
                np.sin(psi) * np.cos(chi) + 1j * np.cos(psi) * np.sin(chi))
    raise ValueError("unknown polarization kind %r" % kind)


def wavelength_strata(src, n_lambda):
    """Deterministic per-stratum wavelengths [m] (equal probability each)."""
    lam_c = src["lambdac_nm"]
    lam_lo = src.get("lambdamin_nm")
    lam_hi = src.get("lambdamax_nm")
    q = (np.arange(n_lambda) + 0.5) / n_lambda      # stratum centers in CDF
    if lam_lo is None and lam_hi is None:
        return np.full(1, lam_c * 1e-9)             # monochromatic: 1 stratum
    if lam_lo is not None and lam_hi is not None:
        sig_m = lam_c - lam_lo
        sig_p = lam_hi - lam_c
        if sig_m < 0 or sig_p < 0:
            raise ValueError("source %r: lambdamin/lambdamax must bracket "
                             "lambdac" % src)
        # two half-normals glued at lambda_c with weights sig-/sig+
        from scipy.stats import norm
        w_m = sig_m / (sig_m + sig_p)
        lam = np.empty(n_lambda)
        left = q < w_m
        # left side: q in [0,w_m) -> half-normal below lambda_c
        qq = q[left] / max(w_m, 1e-300)
        lam[left] = lam_c - np.abs(norm.ppf(0.5 + 0.5 * (1 - qq))) * sig_m
        qq = (q[~left] - w_m) / max(1 - w_m, 1e-300)
        lam[~left] = lam_c + np.abs(norm.ppf(0.5 + 0.5 * qq)) * sig_p
        return lam * 1e-9
    # exactly one bound: symmetric uniform around lambda_c
    w = (lam_c - lam_lo) if lam_lo is not None else (lam_hi - lam_c)
    return (lam_c - w + 2.0 * w * q) * 1e-9


def sample_source(scene, body, src, source_id, n_rays, n_lambda, rng,
                  ledger=None, differentials=False):
    """Sample a RayBatch for one source. Power split equally across rays;
    each ray belongs to one wavelength stratum. differentials=True
    allocates Igehy ray differentials (wavefront patch h = sqrt(A/N)
    along the transverse frame; curvature from the emit surface's shape
    operator) for --ray-differentials dA tracking."""
    face = scene.emit_faces.get(body.index)
    if face is None:
        raise ValueError("source %s has no emit face built — extractor/"
                         "scene mismatch" % body.label)
    surf = face.surface
    lam_strata = wavelength_strata(src, n_lambda)
    n_strata = len(lam_strata)
    power_W = src["power_mW"] * 1e-3
    coherent = bool(src.get("coherent", False))

    pts, normals = _sample_face_points(face, n_rays, rng)
    n = len(pts)

    # direction: toward-origin sign policy
    to_origin = -pts                                  # origin - point
    flat = surf.__class__.__name__ == "Plane"
    if flat:
        n0 = normals[0]
        sign = 1.0 if np.dot(n0, np.mean(to_origin, axis=0)) >= 0 else -1.0
        dirs = np.tile(sign * n0, (n, 1))
        clipped = np.zeros(n, dtype=bool)
    else:
        dots = np.sum(normals * to_origin, axis=-1)
        sign = np.where(dots >= 0.0, 1.0, -1.0)
        # per-sample flip would fold the wavefront: emit only the samples
        # whose natural normal faces the origin; drop (and account) others
        frac_neg = np.mean(sign < 0)
        if 0.0 < frac_neg < 1.0:
            import warnings
            warnings.warn(
                "source %s: emitting face normals straddle the origin "
                "direction (%.1f%% clipped) — emission clipped to the "
                "origin-facing side" % (body.label, 100 * frac_neg))
        if frac_neg == 1.0:
            dirs = -normals
            clipped = np.zeros(n, dtype=bool)
        else:
            dirs = normals
            clipped = sign < 0
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    keep = ~clipped
    pts, dirs = pts[keep], dirs[keep]
    n_kept = len(pts)
    if n_kept == 0:
        raise ValueError("source %s: all emission samples clipped — "
                         "check geometry orientation" % body.label)

    p_ray = power_W / n                               # per-sample power
    if ledger is not None:
        # emitted = the FULL source power; clipped samples immediately
        # balance into their bucket so closure holds
        ledger.emit(np.full(n, source_id), np.full(n, p_ray))
        if np.any(clipped):
            ledger.credit("emission_clipped",
                          np.full(int(np.sum(clipped)), source_id),
                          np.full(int(np.sum(clipped)), p_ray))

    pol = src.get("polarization") or {"kind": "unpolarized"}
    n_pol = n_pol_strata(src)

    batch = RayBatch(n_kept)
    batch.pos[:] = pts
    batch.dir[:] = dirs
    idx = np.arange(n_kept)
    batch.lam[:] = lam_strata[idx % n_strata]
    batch.lam_stratum[:] = idx % n_strata
    # interleave so every (lam, pol) combination is uniformly filled
    batch.pol_stratum[:] = (idx // n_strata) % n_pol
    batch.source_id[:] = source_id
    batch.coherent[:] = coherent
    batch.birth_power[:] = p_ray
    # polarization basis: s_hat = the global-z-referenced transverse frame
    # (module header) so (Es, Ep) IS the Jones vector in that frame
    e_ref, _ = _pol_reference_frame(dirs)
    batch.s_hat[:] = e_ref
    amp = np.sqrt(p_ray)
    if coherent:
        phase = np.ones(n_kept, dtype=np.complex128)
    else:
        phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=n_kept))
    for ps in range(n_pol):
        js, jp = jones_for(pol, ps)
        sel = batch.pol_stratum == ps
        batch.Es[sel] = amp * js * phase[sel]
        batch.Ep[sel] = amp * jp * phase[sel]

    if differentials:
        from .differentials import init_flat, init_curved
        e_perp = np.cross(dirs, e_ref)
        h = np.sqrt((face.area_m2 or 1e-6) / n)
        if flat:
            dPdx, dDdx, dPdy, dDdy = init_flat(dirs, e_ref, e_perp, h)
        else:
            S = surf.normal_derivative(pts)
            n_can = surf.normal(pts)
            sign = np.sign(np.sum(dirs * n_can, axis=-1))
            dPdx, dDdx, dPdy, dDdy = init_curved(dirs, e_ref, e_perp, h,
                                                 S, sign)
        batch.alloc_differentials()
        batch.dPdx[:] = dPdx
        batch.dDdx[:] = dDdx
        batch.dPdy[:] = dPdy
        batch.dDdy[:] = dDdy
    return batch


def _emit_face_from_record(scene, body, src):
    raise ValueError(
        "source %s: emit_face %r is not in the scene's face table — "
        "extractor/scene mismatch" % (body.label, src["emit_face"]))


def _sample_face_points(face, n_rays, rng):
    """Stratified-jittered area sampling of an analytic face.

    Strategy: rejection-sample in a UV bounding box using the trim test,
    with area weights corrected by the local surface metric (first
    fundamental form). For the surfaces used here the metric is:
      plane: 1;  sphere: R^2 cos(v);  cylinder: R;  cone: |v| tan-ish;
    handled by importance-correcting the v coordinate analytically for
    sphere (sample sin(v) uniformly) and uniformly otherwise.
    Returns (points (M,3), normals (M,3) canonical).
    """
    surf = face.surface
    cls = surf.__class__.__name__
    # UV bounds from the trim polygon loops
    if face.trim.mode == "untrimmed":
        if cls == "Sphere":
            u_lo, u_hi = -np.pi, np.pi
            v_lo, v_hi = -np.pi / 2, np.pi / 2
        else:
            raise NotImplementedError(
                "untrimmed emitting face of type %s" % cls)
    elif face.trim.mode == "band":
        u_lo, u_hi = -np.pi, np.pi
        v_lo, v_hi = face.trim.v_band
    else:
        allu = np.concatenate([lp[:, 0] for lp in face.trim.loops])
        allv = np.concatenate([lp[:, 1] for lp in face.trim.loops])
        u_lo, u_hi = float(allu.min()), float(allu.max())
        v_lo, v_hi = float(allv.min()), float(allv.max())

    pts = np.empty((0, 3))
    target = n_rays
    tries = 0
    while len(pts) < target and tries < 60:
        m = int((target - len(pts)) * 1.8) + 16
        u = rng.uniform(u_lo, u_hi, size=m)
        if cls == "Sphere":
            # uniform in area: sample sin(v) uniformly
            sv = rng.uniform(np.sin(v_lo), np.sin(v_hi), size=m)
            v = np.arcsin(sv)
        else:
            v = rng.uniform(v_lo, v_hi, size=m)
        cand = _uv_to_xyz(surf, u, v)
        # containment evaluated through the same to_uv convention the trim
        # polygon itself was built with
        ok = face.trim.contains(surf.to_uv(cand))
        pts = np.concatenate([pts, cand[ok]], axis=0)
        tries += 1
    if len(pts) < target:
        raise RuntimeError(
            "source face %s: area sampling failed to converge "
            "(%d/%d after %d rounds) — trim geometry suspect"
            % (face.id, len(pts), target, tries))
    pts = pts[:target]
    normals = surf.normal(pts)
    return pts, normals


def _wrap(surf, u):
    return (u + np.pi) % (2 * np.pi) - np.pi


def _uv_to_xyz(surf, u, v):
    cls = surf.__class__.__name__
    if cls == "Plane":
        return surf.origin + u[:, None] * surf.t1 + v[:, None] * surf.t2
    if cls == "Sphere":
        cu_, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        return (surf.c
                + surf.r * (cv * cu_)[:, None] * surf.t1
                + surf.r * (cv * su)[:, None] * surf.t2
                + surf.r * sv[:, None] * surf.axis)
    if cls == "Cylinder":
        cu_, su = np.cos(u), np.sin(u)
        return (surf.o
                + surf.r * cu_[:, None] * surf.t1
                + surf.r * su[:, None] * surf.t2
                + v[:, None] * surf.a)
    raise NotImplementedError("emitting face of type %s" % cls)


def _rings_uv(loop_uv, c_uv, pattern):
    """rings:dr=<mm>:nper=<N>[:nrings=<K>] -> centroid + concentric rings,
    every dr mm, nper rays per ring, out to the trim rim (or K rings)."""
    r_rim = float(np.max(np.linalg.norm(loop_uv - c_uv, axis=-1)))
    dr = pattern["dr_mm"] * 1e-3
    n_rings = pattern["nrings"]
    if n_rings is None:
        n_rings = int(np.floor(r_rim / dr + 1e-9))
    uv = [c_uv]
    for k in range(1, n_rings + 1):
        theta = 2.0 * np.pi * np.arange(pattern["nper"]) / pattern["nper"]
        ring = c_uv + (k * dr) * np.stack(
            [np.cos(theta), np.sin(theta)], axis=-1)
        uv.append(ring)
    return np.concatenate([np.atleast_2d(p) for p in uv], axis=0)


def _fan_uv(loop_uv, c_uv, pattern):
    """fan[:n=<K>] (default K=5) -> centroid, then up to 4 cardinal rays
    along the face's local +y/-y/+x/-x directions at 95% of the trim's
    AXIS-ALIGNED extent in that direction (not the corner-to-corner rim
    radius rings uses) so the cardinal points land inside non-circular
    (e.g. square) apertures instead of past their corners. Any rays beyond
    the 4 cardinals fill the largest inscribed circle (95% of the smallest
    of the four axial extents) evenly spaced, offset by 45 deg so they
    don't coincide with the cardinal directions.
    """
    n = pattern["n"]
    u_lo, v_lo = loop_uv.min(axis=0)
    u_hi, v_hi = loop_uv.max(axis=0)
    ext_px = float(u_hi - c_uv[0])
    ext_mx = float(c_uv[0] - u_lo)
    ext_py = float(v_hi - c_uv[1])
    ext_my = float(c_uv[1] - v_lo)

    cardinals = [((0.0, 1.0), ext_py),    # +y (top)
                 ((0.0, -1.0), ext_my),   # -y (bottom)
                 ((1.0, 0.0), ext_px),    # +x (right)
                 ((-1.0, 0.0), ext_mx)]   # -x (left)

    uv = [c_uv]
    n_cardinal = min(max(n - 1, 0), 4)
    for (du, dv), ext in cardinals[:n_cardinal]:
        uv.append(c_uv + 0.95 * ext * np.array([du, dv]))

    extra = n - 1 - n_cardinal
    if extra > 0:
        r_fill = 0.95 * min(ext_px, ext_mx, ext_py, ext_my)
        theta = 2.0 * np.pi * np.arange(extra) / extra + np.pi / 4.0
        ring = c_uv + r_fill * np.stack(
            [np.cos(theta), np.sin(theta)], axis=-1)
        uv.append(ring)
    return np.concatenate([np.atleast_2d(p) for p in uv], axis=0)


def _pattern_uv_points(kind, loop_uv, c_uv, pattern):
    if kind == "rings":
        return _rings_uv(loop_uv, c_uv, pattern)
    if kind == "fan":
        return _fan_uv(loop_uv, c_uv, pattern)
    raise ValueError("sample_viz_pattern: unknown pattern kind %r"
                     % (kind,))


def _sphere_pattern_points(surf, loop_xyz, kind, pattern):
    """Pattern points on a spherical cap (divergent-laser emit face):
    generate the 2D pattern in the rim's best-fit plane, then lift each
    point onto the sphere along the plane normal, choosing the
    intersection on the cap side (nearer the cap apex o + R*Ŵ, W = rim
    centroid - sphere center)."""
    c3 = loop_xyz.mean(axis=0)
    M = loop_xyz - c3
    _, _, vt = np.linalg.svd(M, full_matrices=False)
    e1, e2, n_pl = vt[0], vt[1], vt[2]
    loop2 = np.stack([M @ e1, M @ e2], axis=-1)
    uv = _pattern_uv_points(kind, loop2, loop2.mean(axis=0), pattern)
    pts_plane = c3 + uv[:, :1] * e1 + uv[:, 1:] * e2

    centre, radius = surf.c, surf.r
    w = c3 - centre
    wn = np.linalg.norm(w)
    apex = centre + radius * (w / wn if wn > 1e-12 else n_pl)
    d = pts_plane - centre
    b = d @ n_pl
    c = np.sum(d * d, axis=-1) - radius * radius
    disc = b * b - c
    ok = disc >= 0.0
    pts_plane, b, disc = pts_plane[ok], b[ok], disc[ok]
    root = np.sqrt(disc)
    cand1 = pts_plane + (-b + root)[:, None] * n_pl
    cand2 = pts_plane + (-b - root)[:, None] * n_pl
    d1 = np.linalg.norm(cand1 - apex, axis=-1)
    d2 = np.linalg.norm(cand2 - apex, axis=-1)
    return np.where((d1 <= d2)[:, None], cand1, cand2)


def sample_viz_pattern(scene, body, src, source_id, pattern, n_lambda):
    """Deterministic viz-overlay ray positions: one central ray plus either
    concentric rings or a small cardinal fan (pattern from
    common.parse_viz_pattern_spec: {"kind": "rings", "dr_mm", "nper",
    "nrings"} or {"kind": "fan", "n"}).

    VISUAL HELPER ONLY: callers trace the returned batch in a separate
    viz-only pass (throwaway ledger, no detector grids), so these rays
    can never affect flux, detector images, or the energy audit.

    Planar emit faces get the pattern directly in their metric uv space;
    SPHERICAL caps (divergent lasers) get it in the rim's best-fit plane
    lifted onto the cap, with per-point normal directions (a diverging
    fan, matching sample_source's curved-face emission). Other surface
    types return None with a warning so the caller falls back to the
    default random viz rays.
    """
    face = scene.emit_faces.get(body.index)
    if face is None:
        raise ValueError("source %s has no emit face built" % body.label)
    surf = face.surface
    cls = surf.__class__.__name__
    if cls not in ("Plane", "Sphere"):
        import warnings
        warnings.warn("source %s: --viz-pattern needs a planar or "
                      "spherical emit face (got %s); falling back to "
                      "default viz rays" % (body.label, cls))
        return None

    if face.trim.mode == "untrimmed" or not getattr(face.trim, "loops", None):
        raise ValueError("source %s: emitting face has no trim loops"
                         % body.label)
    kind = pattern["kind"]

    if cls == "Plane":
        # uv == metres for a Plane: t1/t2 are orthonormal
        loop_uv = np.concatenate([np.asarray(lp) for lp in face.trim.loops])
        c_uv = loop_uv.mean(axis=0)
        uv = _pattern_uv_points(kind, loop_uv, c_uv, pattern)
        inside = face.trim.contains(uv)
        uv = uv[inside]
        if len(uv) == 0:
            raise ValueError("source %s: viz pattern produced no rays "
                             "inside the emit face (pattern too large for "
                             "the aperture?)" % body.label)
        pts = _uv_to_xyz(surf, uv[:, 0], uv[:, 1])
        n = len(pts)
        # same toward-origin direction policy as sample_source's flat branch
        n0 = surf.normal(pts)[0]
        sign = 1.0 if np.dot(n0, -np.mean(pts, axis=0)) >= 0 else -1.0
        dirs = np.tile(sign * n0, (n, 1))
    else:
        loop_uv = [np.asarray(lp) for lp in face.trim.loops]
        loop_xyz = np.concatenate(
            [_uv_to_xyz(surf, lp[:, 0], lp[:, 1]) for lp in loop_uv])
        pts = _sphere_pattern_points(surf, loop_xyz, kind, pattern)
        if len(pts) == 0:
            raise ValueError("source %s: viz pattern produced no rays on "
                             "the emitting cap" % body.label)
        n = len(pts)
        # per-point normals, origin-facing side (sample_source's curved
        # policy: flip wholesale when the natural normal faces away)
        normals = surf.normal(pts)
        dots = np.sum(normals * (-pts), axis=-1)
        if np.all(dots < 0):
            normals = -normals
        dirs = normals
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    lam_strata = wavelength_strata(src, n_lambda)
    n_strata = len(lam_strata)
    pol = src.get("polarization") or {"kind": "unpolarized"}
    power_W = src["power_mW"] * 1e-3
    p_ray = power_W / n

    batch = RayBatch(n)
    batch.pos[:] = pts
    batch.dir[:] = dirs
    idx = np.arange(n)
    batch.lam[:] = lam_strata[idx % n_strata]
    batch.lam_stratum[:] = idx % n_strata
    batch.pol_stratum[:] = 0
    batch.source_id[:] = source_id
    batch.coherent[:] = bool(src.get("coherent", False))
    batch.birth_power[:] = p_ray
    e_ref, _ = _pol_reference_frame(dirs)
    batch.s_hat[:] = e_ref
    js, jp = jones_for(pol, 0)
    amp = np.sqrt(p_ray)
    batch.Es[:] = amp * js
    batch.Ep[:] = amp * jp
    return batch
