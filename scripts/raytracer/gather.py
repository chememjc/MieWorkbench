# =============================================================================
# gather.py — coherent Huygens–Fresnel "final gather".
#
# Every coherent ray that reaches a detector is a wavelet sample taken at
# its LAST interaction point (segment start). The complex field at each
# detector pixel is the Rayleigh–Sommerfeld-I style sum
#
#   E(p) = sum_i  E_i sqrt(dA_i) * K_i * exp(i k (opl_i + n_amb r_ip)) / r_ip
#          * (1 / i lambda)
#
# with obliquity K = clamp(0.5 (cos(theta_prop) + cos(theta_det)), 0, 1),
# where theta_prop is between the sample's geometric ray direction and the
# sample->pixel direction (no backward radiation) and theta_det is between
# sample->pixel and the detector normal. Interference (double slit, focal
# Airy structure) emerges from this sum because the geometric phase
# k*opl_i differs between paths.
#
# NORMALIZATION (documented model choice, see the plan): exact wavefront
# patch areas dA_i are not tracked through refraction (that needs ray
# differentials — future.md). Samples carry the SOURCE-referenced patch
# area A_source/N_rays, and the finished per-(source, stratum) intensity
# map is renormalized so its integral equals the geometrically detected
# power. Fringe geometry is exact; the raw-vs-normalized factor is
# reported as `norm_factor` — values far from O(1) indicate the sampling
# or the dA assumption is breaking and must be investigated.
#
# PRECISION: r and the total phase are computed in float64 and reduced
# mod 2pi BEFORE any float32 trig. Path lengths are ~1e5–1e6 waves; float32
# phase would inject O(1 rad) errors and destroy fringes. Accumulators are
# complex64 (post-reduction magnitudes are O(1)).
#
# SAMPLING GATE: with REGULAR sample grids an undersampled wavefront
# aliases into plausible fake fringes; with the random-jittered sampling
# used by sources.py, undersampling instead shows up as an incoherent
# speckle pedestal of relative power ~ 1/M_eff, where
# M_eff = (sum|a|)^2 / sum|a|^2 is the effective sample count. The hard
# gate is therefore on M_eff (default >= 1000, pedestal <= 0.1%); the
# worst-case neighbor phase step is still computed and reported as a
# diagnostic (it flags when samples stop resolving the wavefront even
# before the pedestal matters). Sources MUST jitter their sampling — a
# regular grid would silently re-enable coherent aliasing.
# =============================================================================
import numpy as np

C_AMBIENT_N = 1.000272          # default ambient index for the free flight


class GatherError(RuntimeError):
    pass


def effective_samples(E3):
    """M_eff = (sum|a|)^2 / sum|a|^2 — the incoherent speckle pedestal in
    the rendered intensity is ~ 1/M_eff of the coherent peak."""
    a = np.sqrt(np.sum(np.abs(E3) ** 2, axis=-1))
    s2 = float(np.sum(a ** 2))
    if s2 <= 0:
        return 0.0
    return float(np.sum(a)) ** 2 / s2


def check_sampling(pos, direction, det_grid, lam, max_step=np.pi / 2):
    """DIAGNOSTIC phase-step estimate (not the gate — see module header).

    Worst-case sample-to-sample phase step ~ k * delta * sin(theta_max):
    delta = typical nearest-neighbor sample spacing (from the sample cloud
    footprint), theta_max = largest angle between a sample's ray direction
    and the direction from that sample to any detector corner.
    """
    m = len(pos)
    if m < 4:
        raise GatherError("gather needs >= 4 samples, got %d" % m)
    # footprint area from the two principal extents of the sample cloud
    rel = pos - pos.mean(axis=0)
    cov = rel.T @ rel / m
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    a = 4.0 * np.sqrt(np.maximum(evals[0], 1e-30))   # ~full extents
    b = 4.0 * np.sqrt(np.maximum(evals[1], 1e-30))
    if b < 1e-12:                                    # quasi-1D cloud (slit)
        delta = a / m
    else:
        delta = np.sqrt(a * b / m)
    # theta_max over detector corners (coarse but conservative: use the
    # extreme pixel centers)
    corners = np.array([
        det_grid.pixel_centers[0, 0], det_grid.pixel_centers[0, -1],
        det_grid.pixel_centers[-1, 0], det_grid.pixel_centers[-1, -1]])
    sin_max = 0.0
    for c in corners:
        v = c[None, :] - pos
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        cosang = np.clip(np.sum(v * direction, axis=-1), -1.0, 1.0)
        sin_max = max(sin_max, float(np.max(np.sqrt(1 - cosang ** 2))))
    k = 2.0 * np.pi / float(np.min(lam))
    return k * delta * max(sin_max, 1e-9)


# ---------------------------------------------------------------------------
def accumulate_numpy(pos, E3, lam, opl, det_grid, dir_prop,
                     pixel_chunk=16384):
    """CPU backend on the detector's pixel-center grid. Returns (H, W)
    complex64 (Ex, Ey) in the detector frame."""
    Ex, Ey = points_numpy(pos, E3, lam, opl,
                          det_grid.pixel_centers.reshape(-1, 3),
                          det_grid.xhat, det_grid.yhat, det_grid.normal,
                          dir_prop, pixel_chunk)
    return (Ex.reshape(det_grid.H, det_grid.W),
            Ey.reshape(det_grid.H, det_grid.W))


def points_numpy(pos, E3, lam, opl, points, xhat, yhat, nrm, dir_prop,
                 pixel_chunk=16384, with_pedestal=False, occ=None):
    """Field at arbitrary points (Q,3) — used for both the pixel grid and
    sub-pixel refinement. E3: (M,3) complex field vectors (Jones expanded
    to 3D, sqrt(dA) folded in). with_pedestal also returns the exact MC
    noise expectation sum_i w_i^2 |a_i|^2 per point.

    occ (optional gather-occlusion, see _build_occlusion): a tuple
    (mask_ts, tile_of_point) with mask_ts a (n_tiles, M) bool array (True =
    sample visible to that tile) and tile_of_point (Q,) mapping each point
    to a detector tile. Blocked (sample, point) pairs get obliquity K = 0,
    zeroing both the field and the pedestal contribution. occ=None is a
    no-op with numerics bit-identical to the un-occluded kernel."""
    npix = points.shape[0]
    k = 2.0 * np.pi / float(lam[0])
    n_amb = C_AMBIENT_N
    Ex_flat = np.zeros(npix, dtype=np.complex64)
    Ey_flat = np.zeros(npix, dtype=np.complex64)
    ped_flat = np.zeros(npix, dtype=np.float64) if with_pedestal else None
    Exs = (E3 @ xhat.astype(np.complex128))
    Eys = (E3 @ yhat.astype(np.complex128))
    amp2 = (np.abs(Exs) ** 2 + np.abs(Eys) ** 2).astype(np.float64)
    inv_lam2 = 1.0 / float(lam[0]) ** 2
    inv_ilam = -1j / float(lam[0])       # 1/(i lambda) = -i/lambda

    # dot-product formulation: never materialize the (Q, M, 3) displacement
    # tensor. (P - pos_i).d_i = P @ d_i - pos_i.d_i; |P - pos_i|^2 =
    # |P|^2 - 2 P.pos_i + |pos_i|^2 — all (Q, M) via matmuls.
    pos_dot_dir = np.sum(pos * dir_prop, axis=-1)        # (M,)
    pos_dot_n = pos @ nrm                                # (M,)
    pos_sq = np.sum(pos * pos, axis=-1)                  # (M,)
    for lo in range(0, npix, pixel_chunk):
        hi = min(lo + pixel_chunk, npix)
        P = points[lo:hi]                                # (Q,3)
        r2 = (np.sum(P * P, axis=-1)[:, None]
              - 2.0 * (P @ pos.T) + pos_sq[None, :])
        r = np.sqrt(np.maximum(r2, 1e-18))
        inv_r = 1.0 / r
        rhat_dot_dir = ((P @ dir_prop.T) - pos_dot_dir[None, :]) * inv_r
        cos_det = np.abs((P @ nrm)[:, None] - pos_dot_n[None, :]) * inv_r
        K = np.clip(0.5 * (rhat_dot_dir + cos_det), 0.0, 1.0)
        K[rhat_dot_dir <= 0.0] = 0.0                     # no back-radiation
        if occ is not None:                              # gather occlusion
            mask_ts, tile_of_point = occ
            K = K * mask_ts[tile_of_point[lo:hi]]        # (Q,M) 0/1
        phase = k * (opl[None, :] + n_amb * r)
        phase = np.mod(phase, 2.0 * np.pi)               # reduce in f64
        prop = (np.cos(phase).astype(np.float32)
                + 1j * np.sin(phase).astype(np.float32))
        Kir = K * inv_r
        if with_pedestal:
            ped_flat[lo:hi] = (Kir ** 2 @ amp2) * inv_lam2
        w = Kir.astype(np.float32) * prop                # (Q,M) c64
        Ex_flat[lo:hi] = (w @ Exs.astype(np.complex64)) * inv_ilam
        Ey_flat[lo:hi] = (w @ Eys.astype(np.complex64)) * inv_ilam
    if with_pedestal:
        return Ex_flat, Ey_flat, ped_flat
    return Ex_flat, Ey_flat


def accumulate_torch(pos, E3, lam, opl, det_grid, dir_prop,
                     pixel_chunk=16384, sample_chunk=8192):
    """CUDA backend on the detector's pixel-center grid."""
    Ex, Ey = points_torch(pos, E3, lam, opl,
                          det_grid.pixel_centers.reshape(-1, 3),
                          det_grid.xhat, det_grid.yhat, det_grid.normal,
                          dir_prop, pixel_chunk, sample_chunk)
    return (Ex.reshape(det_grid.H, det_grid.W),
            Ey.reshape(det_grid.H, det_grid.W))


def points_torch(pos, E3, lam, opl, points, xhat_np, yhat_np, nrm_np,
                 dir_prop, pixel_chunk=16384, sample_chunk=8192,
                 with_pedestal=False, occ=None):
    """CUDA field evaluation at arbitrary points: float64 r/phase,
    float32 accumulation. with_pedestal also returns sum w^2 |a|^2.

    occ: same (mask_ts, tile_of_point) as points_numpy. The mask is built
    once in numpy (see _build_occlusion) and merely uploaded here, so the
    two backends apply the IDENTICAL occlusion mask by construction."""
    import torch
    dev = torch.device("cuda")
    pix = torch.from_numpy(np.ascontiguousarray(points)).to(
        dev, torch.float64)
    npix = pix.shape[0]
    k = 2.0 * np.pi / float(lam[0])
    n_amb = C_AMBIENT_N

    pos_t = torch.from_numpy(pos).to(dev, torch.float64)
    dirp = torch.from_numpy(dir_prop).to(dev, torch.float64)
    opl_t = torch.from_numpy(opl).to(dev, torch.float64)
    nrm = torch.from_numpy(nrm_np).to(dev, torch.float64)
    Exs = torch.from_numpy((E3 @ xhat_np).astype(np.complex64)).to(dev)
    Eys = torch.from_numpy((E3 @ yhat_np).astype(np.complex64)).to(dev)
    amp2 = (Exs.abs() ** 2 + Eys.abs() ** 2).to(torch.float64)
    inv_lam2 = 1.0 / float(lam[0]) ** 2
    mask_ts_t = tile_t = None
    if occ is not None:
        mask_ts_np, tile_of_point = occ
        mask_ts_t = torch.from_numpy(
            np.ascontiguousarray(mask_ts_np)).to(dev, torch.float32)
        tile_t = torch.from_numpy(
            np.ascontiguousarray(tile_of_point)).to(dev, torch.long)

    Ex = torch.zeros(npix, dtype=torch.complex64, device=dev)
    Ey = torch.zeros(npix, dtype=torch.complex64, device=dev)
    ped = torch.zeros(npix, dtype=torch.float64, device=dev) \
        if with_pedestal else None
    M = pos_t.shape[0]
    # dot-product formulation (see accumulate_numpy): only (Q, M) 2-D
    # intermediates, ~6 live fp64 arrays of pixel_chunk x sample_chunk
    pos_dot_dir = (pos_t * dirp).sum(-1)                 # (M,)
    pos_dot_n = pos_t @ nrm                              # (M,)
    pos_sq = (pos_t * pos_t).sum(-1)                     # (M,)
    for plo in range(0, npix, pixel_chunk):
        phi_ = min(plo + pixel_chunk, npix)
        P = pix[plo:phi_]
        P_sq = (P * P).sum(-1)
        P_dot_n = P @ nrm
        for slo in range(0, M, sample_chunk):
            shi = min(slo + sample_chunk, M)
            # keep peak live tensors ~5 x (Q, S) fp64; free eagerly
            r = P_sq[:, None] - 2.0 * (P @ pos_t[slo:shi].T)
            r += pos_sq[None, slo:shi]
            r.clamp_(min=1e-18).sqrt_()
            inv_r = 1.0 / r
            cprop = ((P @ dirp[slo:shi].T)
                     - pos_dot_dir[None, slo:shi]) * inv_r
            K = torch.abs(P_dot_n[:, None] - pos_dot_n[None, slo:shi])
            K *= inv_r
            K += cprop
            K *= 0.5
            K.clamp_(0.0, 1.0)
            K.masked_fill_(cprop <= 0.0, 0.0)
            del cprop
            if occ is not None:                          # gather occlusion
                # slice the SAMPLE columns first: indexing rows first
                # materializes a (pixel_chunk, M_total) intermediate that
                # OOMs at large sample counts (caught by the all-flags
                # integration test)
                K *= mask_ts_t[:, slo:shi][tile_t[plo:phi_]]
            phase = torch.remainder(
                k * (opl_t[slo:shi][None, :] + n_amb * r),
                2.0 * np.pi)
            del r
            K *= inv_r
            del inv_r
            if with_pedestal:
                ped[plo:phi_] += (K ** 2 @ amp2[slo:shi]) * inv_lam2
            w = K.to(torch.float32) * torch.polar(
                torch.ones_like(phase, dtype=torch.float32),
                phase.to(torch.float32))
            del K, phase
            Ex[plo:phi_] += w @ Exs[slo:shi]
            Ey[plo:phi_] += w @ Eys[slo:shi]
            del w
    inv_ilam = complex(0.0, -1.0 / float(lam[0]))
    if with_pedestal:
        return ((Ex * inv_ilam).cpu().numpy(),
                (Ey * inv_ilam).cpu().numpy(),
                ped.cpu().numpy())
    return ((Ex * inv_ilam).cpu().numpy(),
            (Ey * inv_ilam).cpu().numpy())


# ===========================================================================
# OPTIONAL GATHER OCCLUSION  (render_coherent(..., occlusion=...))
#
# The free-space Rayleigh-Sommerfeld sum above propagates EVERY sample to
# EVERY pixel along a straight line — it never checks whether a scene body
# sits in that line of sight (README §6.2 item 3). With `occlusion` supplied
# we ray-cast each (sample, detector-tile) segment against the occluder faces
# and zero the obliquity K of blocked pairs. Two levels keep it affordable:
#
#   Level 1 (prefilter, _face_active): a PROVABLY CONSERVATIVE axis-aligned
#     bounding-box test. Every shadow segment lies inside the convex hull of
#     (sample points) u (detector pixel centres); any real face/segment
#     intersection point lies in BOTH the face AABB and that hull AABB, so if
#     those two AABBs are disjoint the face cannot occlude anything and is
#     dropped. (The alternative "cast the 8x8 bbox-corner extreme rays and
#     drop the face if none hit" is NOT conservative — a small occluder can
#     block only interior segments while missing every corner ray — so it is
#     deliberately not used.) When a face's world AABB cannot be bounded
#     cheaply (non-planar, or a band/untrimmed trim) the face is KEPT
#     (unsure -> conservative); Level 2 then simply finds no blocking.
#
#   Level 2 (_build_occlusion): split the detector into `tile` x `tile`
#     pixel tiles; for every (sample, tile-centre) pair cast one shadow ray
#     (origin = sample, aim at the tile centre, range = the segment length)
#     through each surviving face via face.intersect (AnalyticFace / MeshFace
#     share this (t, hit) interface with a t_eps self-hit guard). A hit with
#     t in (t_eps, dist - t_eps) marks that (sample, tile) blocked. The
#     boolean mask (n_tiles, M) is built ONCE in numpy and handed to both
#     kernels, so the torch and numpy images apply the identical mask.
#
# Documented physics caveats (intentional model choices):
#   * TILE QUANTIZATION: a shadow edge is resolved only to `tile` pixels
#     (default 16). Sharp shadow geometry needs the tracer proper, or a
#     smaller tile (occlusion["tile"], down to 1 = per-pixel shadow rays).
#   * OPAQUE OCCLUDERS: ANY face blocks fully — even a clear glass lens face
#     shadows as if opaque. This is deliberate and conservative: the tracer's
#     refracted field already re-anchors as fresh gather samples at the lens
#     EXIT, so also letting the pre-lens samples shine straight through would
#     double-count that path. Removing that double-count is the whole point.
#   * SELF-OCCLUSION: a sample's own last interaction face would ideally be
#     excluded, but the t_eps guard already makes self-hits rare;
#     occlusion["exclude_last"] (per-sample last-face id, or None) is accepted
#     and honoured if given but the default None path relies on t_eps.
#
# MEMORY: the mask is n_tiles x M bool = ceil(H/tile)*ceil(W/tile) * M bytes
# (e.g. 512^2 detector, tile 16, M=1e5 -> 1024*1e5 = ~100 MB). The shadow
# cast itself touches the same n_tiles*M ray count, chunked to _SHADOW_CHUNK
# rays per face.intersect call. Cost scales with n_active_faces * M * n_tiles.
# ===========================================================================
_SHADOW_CHUNK = 1 << 20            # shadow rays per face.intersect() batch
_OCC_T_EPS = 1e-7                  # near/far segment guard (matches face t_eps)


def _face_world_aabb(face):
    """Cheap conservative world-space AABB of an occluder face, or None when
    it cannot be bounded cheaply (caller then keeps the face). Only planar,
    polygon-trimmed AnalyticFaces are bounded here — their trim loops are the
    exact 3-D boundary; curved / band / untrimmed faces return None."""
    surf = getattr(face, "surface", None)
    trim = getattr(face, "trim", None)
    if surf is None or trim is None:
        return None
    if surf.__class__.__name__ != "Plane":
        return None
    if getattr(trim, "mode", None) != "polygon" or not getattr(trim, "loops",
                                                               None):
        return None
    pts = [surf.origin + lp[:, 0:1] * surf.t1 + lp[:, 1:2] * surf.t2
           for lp in trim.loops]
    allp = np.concatenate(pts)
    return allp.min(axis=0), allp.max(axis=0)


def _face_active(face, seg_lo, seg_hi):
    """Level-1 prefilter: True if the face MIGHT occlude (conservative).
    seg_lo/seg_hi bound the convex hull of all shadow segments."""
    bb = _face_world_aabb(face)
    if bb is None:
        return True                        # unbounded -> keep (conservative)
    f_lo, f_hi = bb
    # AABB overlap is a necessary condition for the face to meet any segment
    return bool(np.all(f_hi >= seg_lo - 1e-12)
                and np.all(f_lo <= seg_hi + 1e-12))


def _tile_layout(det_grid, tile):
    """Return (tile_centers (n_tiles,3), tile_of_grid_point (H*W,), Tw)."""
    H, W = det_grid.H, det_grid.W
    Th = int(np.ceil(H / tile))
    Tw = int(np.ceil(W / tile))
    rc = np.minimum((np.arange(Th) * tile + tile // 2), H - 1)
    cc = np.minimum((np.arange(Tw) * tile + tile // 2), W - 1)
    centers = det_grid.pixel_centers[np.repeat(rc, Tw),
                                     np.tile(cc, Th)]           # (n_tiles,3)
    rows = np.repeat(np.arange(H), W)
    cols = np.tile(np.arange(W), H)
    tile_of_grid_point = (rows // tile) * Tw + (cols // tile)
    return centers, tile_of_grid_point.astype(np.intp), Tw


def _build_occlusion(det_grid, pos, occlusion):
    """Build the (n_tiles, M) visibility mask for one sample population.

    Returns (mask_ts bool, tile_of_grid_point, tile, Tw, diag). mask_ts is
    True where a sample is visible to a tile. diag reports the prefilter and
    the blocked-pair fraction. Faces belonging to this detector grid never
    occlude their own screen."""
    faces = occlusion.get("faces") or []
    tile = int(occlusion.get("tile", 16))
    exclude_last = occlusion.get("exclude_last")
    M = len(pos)
    centers, tile_of_grid_point, Tw = _tile_layout(det_grid, tile)
    n_tiles = centers.shape[0]
    mask = np.ones((n_tiles, M), dtype=bool)             # True = visible

    # convex-hull AABB of all shadow segments (sample points u tile centres)
    seg_lo = np.minimum(pos.min(axis=0), centers.min(axis=0))
    seg_hi = np.maximum(pos.max(axis=0), centers.max(axis=0))

    det_face = getattr(det_grid, "face", None)
    det_id = getattr(det_face, "id", None)
    n_tested = 0
    n_active = 0
    for face in faces:
        if face is det_face or (det_id is not None
                                and getattr(face, "id", None) == det_id):
            continue                                     # never self-occlude
        n_tested += 1
        if not _face_active(face, seg_lo, seg_hi):
            continue
        n_active += 1
        blocked = _shadow_face(face, pos, centers, exclude_last)
        mask &= ~blocked                                 # (n_tiles, M)
    frac = float(1.0 - mask.mean()) if mask.size else 0.0
    diag = {"n_faces_tested": int(n_tested),
            "n_faces_active": int(n_active),
            "frac_pairs_blocked": frac}
    return mask, tile_of_grid_point, tile, Tw, diag


def _shadow_face(face, pos, centers, exclude_last):
    """(n_tiles, M) bool: True where face blocks the sample->tile segment."""
    M = len(pos)
    n_tiles = centers.shape[0]
    n_pairs = M * n_tiles
    face_id = getattr(face, "id", None)
    blocked = np.zeros((M, n_tiles), dtype=bool)          # (sample, tile)
    flat = blocked.reshape(-1)
    for lo in range(0, n_pairs, _SHADOW_CHUNK):
        hi = min(lo + _SHADOW_CHUNK, n_pairs)
        idx = np.arange(lo, hi)
        si = idx // n_tiles
        ti = idx % n_tiles
        o = pos[si]
        seg = centers[ti] - o
        dist = np.sqrt(np.sum(seg * seg, axis=-1))
        safe = dist > _OCC_T_EPS
        d = seg / np.where(safe[:, None], dist[:, None], 1.0)
        t, hit = face.intersect(o, d)
        blk = hit & safe & (t > _OCC_T_EPS) & (t < dist - _OCC_T_EPS)
        if exclude_last is not None and face_id is not None:
            blk &= (exclude_last[si] != face_id)          # skip self-hits
        flat[lo:hi] = blk
    return blocked.T                                      # (n_tiles, M)


# ---------------------------------------------------------------------------
def render_coherent(det_grid, sample_area_m2, backend="auto",
                    enforce_gate=True, min_eff_samples=1000.0,
                    occlusion=None, save_fields=False):
    """Render every (source, stratum) sample set on a detector into the
    detector's incoherent spectral cube. Returns diagnostics per key.

    Gate: M_eff >= min_eff_samples (speckle pedestal <= 1/min_eff_samples
    of the coherent peak). Raise rays if it trips.

    occlusion: None (default; zero overhead, numerics bit-identical to the
    un-occluded render) or a dict
        {"faces": [face objects to test],       # scene faces minus detectors
         "exclude_last": ndarray|None,          # per-sample last-face id
         "tile": int}                           # shadow tile size, default 16
    See the OPTIONAL GATHER OCCLUSION block above for the two-level scheme
    and the documented caveats."""
    merged = det_grid.merged_samples()
    diags = {}
    if not merged:
        return diags
    use_torch = False
    if backend in ("auto", "torch"):
        try:
            import torch
            use_torch = torch.cuda.is_available()
        except ImportError:
            use_torch = False
        if backend == "torch" and not use_torch:
            raise GatherError("backend=torch requested but CUDA/torch "
                              "unavailable")

    for key, s in merged.items():
        # ---- split populations ------------------------------------------
        # smooth (non-scattered) samples represent a piecewise-smooth
        # wavefront randomly sampled: E[|sum|^2] = |signal|^2 + sum|a|^2,
        # and the second term is MC noise -> subtract its exact
        # expectation (computed by the same kernel). Scattered samples
        # (particles/roughness) have physically random phases: their
        # sum|a|^2 pedestal IS the speckle intensity -> keep it.
        scat = s.get("scattered")
        if scat is None:
            scat = np.zeros(len(s["pos"]), dtype=bool)
        p_hat = np.cross(s["dir"], s["s_hat"])
        # per-sample wavefront patch areas from --ray-differentials where
        # tracked; NaN/invalid rows (grating orders, scattered lobes,
        # birefringent splits) fall back to the source-referenced area
        dA_fallback = sample_area_m2.get(key, 1.0)
        dA_arr = s.get("dA")
        if dA_arr is not None and np.any(np.isfinite(dA_arr)):
            good = np.isfinite(dA_arr) & (dA_arr > 0)
            dA = np.where(good, dA_arr, dA_fallback)
            n_diff_dA = int(np.sum(good))
        else:
            dA = np.full(len(s["pos"]), dA_fallback)
            n_diff_dA = 0
        E3_all = (s["Es"][:, None] * s["s_hat"]
                  + s["Ep"][:, None] * p_hat) * np.sqrt(dA)[:, None]
        m_eff = effective_samples(E3_all)
        step = check_sampling(s["pos"], s["dir"], det_grid, s["lam"])
        if enforce_gate and m_eff < min_eff_samples:
            raise GatherError(
                "gather undersampled on %s for source/stratum %s: "
                "effective samples M_eff=%.0f < %.0f (speckle pedestal "
                "%.2e of peak). Increase --rays by ~%.0fx."
                % (det_grid.label, key, m_eff, min_eff_samples,
                   1.0 / max(m_eff, 1e-9),
                   min_eff_samples / max(m_eff, 1e-9)))

        # gather occlusion: build the (n_tiles, M) visibility mask ONCE for
        # all samples of this key; the population loop slices its columns.
        occ_ctx = None
        occ_diag = None
        if occlusion is not None:
            mask_full, tile_of_grid, tile, Tw, occ_diag = _build_occlusion(
                det_grid, s["pos"], occlusion)
            occ_ctx = (mask_full, tile_of_grid, tile, Tw)

        inten = np.zeros((det_grid.H, det_grid.W))
        fields = None
        if save_fields:
            fields = (np.zeros((det_grid.H, det_grid.W),
                               dtype=np.complex128),
                      np.zeros((det_grid.H, det_grid.W),
                               dtype=np.complex128))
        pop_diag = {}
        for pop_name, sel, unbiased in (("smooth", ~scat, True),
                                        ("speckle", scat, False)):
            if not np.any(sel):
                continue
            occ_pop = None
            if occ_ctx is not None:
                mask_full, tile_of_grid, tile, Tw = occ_ctx
                occ_pop = (mask_full[:, sel], tile_of_grid, tile, Tw)
            ii, nd = _render_population(
                det_grid, s["pos"][sel], s["dir"][sel], s["lam"][sel],
                s["opl"][sel], E3_all[sel], s["power"][sel],
                use_torch, unbiased, occ=occ_pop)
            if save_fields:
                # plain (biased-pedestal-free is meaningless for a field)
                # complex sum over ALL samples of the population, scaled
                # by sqrt of the same normalization factor as intensity
                pfn = points_torch if use_torch else points_numpy
                occ_f = None
                if occ_ctx is not None:
                    occ_f = (occ_ctx[0][:, sel], occ_ctx[1])
                Exf, Eyf = pfn(
                    s["pos"][sel], E3_all[sel], s["lam"][sel],
                    s["opl"][sel], det_grid.pixel_centers.reshape(-1, 3),
                    det_grid.xhat, det_grid.yhat, det_grid.normal,
                    s["dir"][sel], occ=occ_f)
            # normalize this population to ITS geometric arrival power.
            # The cross-estimator integral is unbiased, so the factor is
            # a meaningful O(1) diagnostic of the dA approximation; a
            # factor far from 1 means sampling/dA assumptions broke.
            p_pop = float(np.sum(s["power"][sel]))
            raw = float(ii.sum())
            factor = p_pop / raw if raw > 0 else 0.0
            inten += ii * factor
            if save_fields:
                fields[0][:] += np.sqrt(factor) \
                    * Exf.reshape(det_grid.H, det_grid.W)
                fields[1][:] += np.sqrt(factor) \
                    * Eyf.reshape(det_grid.H, det_grid.W)
            nd.update({"power_W": p_pop, "raw_integral": raw,
                       "norm_factor_applied": factor,
                       # dimensionless dA-quality diagnostic: the raw
                       # integral is a per-pixel intensity SUM, so the
                       # applied factor inherently carries a pixel-area
                       # scale. factor/pixel_area ~ O(1) means the sample
                       # dA model is consistent (with --ray-differentials
                       # it is, except near caustics where differential
                       # patch areas legitimately collapse and the
                       # renormalization restores total power).
                       "norm_factor_dimensionless":
                           factor / (det_grid.pixel_m ** 2),
                       "n_samples": int(np.sum(sel))})
            pop_diag[pop_name] = nd
        # DO NOT clip: negative pixels are zero-mean estimator noise and
        # the stored map must stay unbiased (sums/spectra/profiles then
        # average the noise away). Clipping happens only at PNG-render
        # time. The noise floor is estimated from the negative tail
        # (zero-mean symmetric): sigma ~ std of the negative pixels.
        inten[~det_grid.mask] = 0.0
        neg = inten[inten < 0]
        noise_floor = float(np.sqrt(np.mean(neg ** 2))) if len(neg) \
            else 0.0

        b = det_grid.lam_bin(np.array([s["lam"][0]]))[0]
        det_grid.inc[b] += inten
        if save_fields:
            # per-key complex field maps, consumed by save_detectors into
            # the .h5 'fields/<s>_<l>_<p>/{Ex,Ey}' layout (post_process
            # renders Stokes maps from them)
            if not hasattr(det_grid, "fields"):
                det_grid.fields = {}
            det_grid.fields[key] = fields
        diags[key] = {
            "n_samples": int(len(s["pos"])),
            "effective_samples": float(m_eff),
            "lambda_nm": float(s["lam"][0] / 1e-9),
            "phase_step_rad": float(step),
            "detected_geometric_W":
                det_grid.detected_geometric.get(key, 0.0),
            "noise_floor_W_per_px": noise_floor,
            "n_differential_dA": n_diff_dA,
            "populations": pop_diag,
            "backend": "torch" if use_torch else "numpy",
        }
        if occ_diag is not None:
            diags[key]["occlusion"] = occ_diag
    return diags


N_CROSS_GROUPS = 4


def _cross_intensity(pos, dirp, lam, opl, E3, pts, det_grid, use_torch,
                     unbiased, rng_seed=0, occ=None):
    """Intensity at points via the G-group cross-estimator.

    unbiased=True (smooth wavefronts): I = (|sum_g E_g|^2 -
    sum_g |E_g|^2) / (1 - 1/G) — the MC speckle noise has zero mean and
    MAY GO NEGATIVE per point; the spatial integral is unbiased, so no
    clipping bias enters the power bookkeeping (clip for display only,
    downstream).
    unbiased=False (physical speckle): plain |sum E|^2 — the pedestal is
    real intensity.

    occ (gather occlusion): (mask_ts (n_tiles, M), tile_of_point (Q,)) for
    the samples/points of THIS call, or None. Column subsetting by the
    cross-estimator groups is applied here."""
    pfn = points_torch if use_torch else points_numpy
    m = len(pos)
    if not unbiased or m < 4 * N_CROSS_GROUPS:
        Ex, Ey = pfn(pos, E3, lam, opl, pts, det_grid.xhat,
                     det_grid.yhat, det_grid.normal, dirp, occ=occ)
        return (np.abs(Ex) ** 2 + np.abs(Ey) ** 2).astype(np.float64)
    G = N_CROSS_GROUPS
    groups = np.random.default_rng(rng_seed).integers(0, G, m)
    Ex_tot = np.zeros(pts.shape[0], dtype=np.complex64)
    Ey_tot = np.zeros(pts.shape[0], dtype=np.complex64)
    sum_sq = np.zeros(pts.shape[0], dtype=np.float64)
    for g in range(G):
        sel = groups == g
        if not np.any(sel):
            continue
        occ_g = None if occ is None else (occ[0][:, sel], occ[1])
        Ex, Ey = pfn(pos[sel], E3[sel], lam[sel], opl[sel], pts,
                     det_grid.xhat, det_grid.yhat, det_grid.normal,
                     dirp[sel], occ=occ_g)
        Ex_tot += Ex
        Ey_tot += Ey
        sum_sq += (np.abs(Ex) ** 2 + np.abs(Ey) ** 2).astype(np.float64)
    tot = (np.abs(Ex_tot) ** 2 + np.abs(Ey_tot) ** 2).astype(np.float64)
    return (tot - sum_sq) / (1.0 - 1.0 / G)


def _render_population(det_grid, pos, dirp, lam, opl, E3, power,
                       use_torch, unbiased, occ=None):
    """Render one sample population to a (H, W) intensity map (unnorm.,
    possibly with negative noise pixels when unbiased). Applies sub-pixel
    hot-spot refinement.

    occ (gather occlusion): (mask_ts (n_tiles, M), tile_of_grid_point,
    tile, Tw) for THIS population's samples, or None. The grid render maps
    each pixel via tile_of_grid_point; sub-pixel refinement maps each hot
    pixel to its tile so the whole sub-grid inherits the pixel's shadow."""
    grid_pts = det_grid.pixel_centers.reshape(-1, 3)
    mask_ts = tile = Tw = None
    grid_occ = None
    if occ is not None:
        mask_ts, tile_of_grid, tile, Tw = occ
        grid_occ = (mask_ts, tile_of_grid)
    inten = _cross_intensity(pos, dirp, lam, opl, E3, grid_pts, det_grid,
                             use_torch, unbiased, occ=grid_occ)
    inten = inten.reshape(det_grid.H, det_grid.W)

    # sub-pixel refinement of geometrically hot pixels: a sub-pixel focal
    # spot falls between pixel-center samples and its power would be
    # silently lost otherwise (a physical pixel integrates intensity
    # over its area)
    n_hot = 0
    s_sub = 0
    hot = _hot_pixels(det_grid, pos, dirp, power)
    if hot is not None:
        hy, hx = hot
        n_hot = len(hy)
        s_sub = _subgrid_factor(det_grid, dirp, lam)
        if n_hot and s_sub > 1:
            pts = _subpixel_points(det_grid, hy, hx, s_sub)
            sub_occ = None
            if occ is not None:
                tile_of_hot = (hy // tile) * Tw + (hx // tile)
                sub_occ = (mask_ts,
                           np.repeat(tile_of_hot,
                                     s_sub * s_sub).astype(np.intp))
            sub_i = _cross_intensity(pos, dirp, lam, opl, E3, pts,
                                     det_grid, use_torch, unbiased,
                                     occ=sub_occ)
            inten[hy, hx] = sub_i.reshape(
                n_hot, s_sub * s_sub).mean(axis=1)
    return inten, {"refined_pixels": int(n_hot), "subgrid": int(s_sub)}


def _hot_pixels(det_grid, pos, dirp, power, frac=5e-4, cap=768):
    """Pixels geometrically receiving concentrated power: project each
    sample along its ray to the detector plane, histogram power, return
    (rows, cols) of pixels above `frac` of the total (top `cap` by
    power)."""
    nrm = det_grid.normal
    plane_off = float(det_grid.pixel_centers[0, 0] @ nrm)
    denom = dirp @ nrm
    ok = np.abs(denom) > 1e-9
    if not np.any(ok):
        return None
    t = (plane_off - pos[ok] @ nrm) / denom[ok]
    land = pos[ok] + t[:, None] * dirp[ok]
    fx, fy = det_grid.to_grid(land)
    xi = np.floor(fx).astype(int)
    yi = np.floor(fy).astype(int)
    inb = ((xi >= 0) & (xi < det_grid.W) & (yi >= 0)
           & (yi < det_grid.H) & (t > 0))
    if not np.any(inb):
        return None
    pmap = np.zeros((det_grid.H, det_grid.W))
    np.add.at(pmap, (yi[inb], xi[inb]), power[ok][inb])
    total = pmap.sum()
    if total <= 0:
        return None
    hy, hx = np.where(pmap > frac * total)
    if len(hy) > cap:
        order = np.argsort(pmap[hy, hx])[::-1][:cap]
        hy, hx = hy[order], hx[order]
    return (hy, hx) if len(hy) else None


def _subgrid_factor(det_grid, dirp, lam, s_max=24):
    """Sub-grid points per pixel side: pitch <= lambda / (4 sin theta),
    theta = 95th-percentile angle between arriving rays and the normal."""
    cosang = np.abs(dirp @ det_grid.normal)
    sin95 = float(np.percentile(np.sqrt(np.clip(1 - cosang ** 2, 0, 1)),
                                95))
    if sin95 < 1e-6:
        return 1
    delta = float(lam[0]) / (4.0 * sin95)
    return int(np.clip(np.ceil(det_grid.pixel_m / delta), 1, s_max))


def _subpixel_points(det_grid, hy, hx, s):
    """(n_hot * s^2, 3) world points on an s x s jitter-free sub-grid."""
    offs = (np.arange(s) + 0.5) / s          # in pixel units
    oy, ox = np.meshgrid(offs, offs, indexing="ij")
    base = det_grid.pixel_centers[hy, hx]    # (n,3) pixel centers
    # pixel centers are at +0.5 px; shift to pixel corner then add offsets
    corner = (base - 0.5 * det_grid.pixel_m * det_grid.xhat
              - 0.5 * det_grid.pixel_m * det_grid.yhat)
    pts = (corner[:, None, None, :]
           + (ox * det_grid.pixel_m)[None, :, :, None]
           * det_grid.xhat[None, None, None, :]
           + (oy * det_grid.pixel_m)[None, :, :, None]
           * det_grid.yhat[None, None, None, :])
    return pts.reshape(-1, 3)
