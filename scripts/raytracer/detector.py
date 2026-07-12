# =============================================================================
# detector.py — detector pixel grids and accumulation.
#
# Grid: each detector face gets a deterministic in-plane basis:
#   xhat = normalized projection of the global axis most orthogonal to the
#          face normal; yhat = n x xhat.
# The pixel rectangle is the trim polygon's bbox in that frame; pixels
# outside the trim are masked. The basis/origin/pitch are recorded so
# post-processing and case.json can reproduce the mapping exactly.
#
# Accumulation:
#   * incoherent power (all incoherent-source rays, and continuum-scattered
#     rays): direct bilinear splat of ray power into (spectral_bin, H, W).
#   * coherent gather samples: rays from coherent sources arriving at the
#     detector are recorded as Huygens wavelet samples (position/dir/Jones/
#     opl/lambda) and rendered to a complex field per (source, lambda
#     stratum) by gather.py; per-stratum intensities then add incoherently
#     into the spectral cube.
#
# Normalization (documented model choice): the coherent image from gather.py
# is renormalized so its integrated power equals the geometric
# detected power for that (source, detector) pair — fringe structure comes
# from phase, absolute scale from energy conservation. The applied factor is
# reported; drift far from 1 flags a sampling problem (see gather.py).
# =============================================================================
import numpy as np

C_LIGHT_MPS = 299792458.0

# canonical order of the four selectable time products (pulsed-optics P4)
TIME_PRODUCTS = ("pulse", "spectrogram", "streak", "cube")

# FWHM = _FWHM_SIGMA * sigma for a Gaussian
_FWHM_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def resolve_time_products(args, scene):
    """The tuple of active time products for this run (subset of
    TIME_PRODUCTS, canonical order), or () when time products are off.

    Rules (locked design, pulsed-optics P4):
      * --time-products given: exactly what it parsed to (cli_specs.
        parse_time_products; 'none' parses to the empty tuple).
      * flag absent + any PULSED source (src['pulse'] with a duration):
        auto-default to ('pulse', 'spectrogram').
      * flag absent, no pulsed source (every pre-existing CW scene): ().

    Shared by run_trace (drives track_time + detector recording) and
    cengine.detect_features (any active product is an unported feature ->
    Python engine), so the two can never disagree."""
    tp = getattr(args, "time_products", None)
    if tp is not None:
        return tuple(tp)
    for _, src in scene.sources:
        pulse = src.get("pulse") or {}
        if pulse.get("duration_s"):
            return ("pulse", "spectrogram")
    return ()


def _splat_records(acc, t, sigma, power, aux, t_lo, dt):
    """Deposit every arrival record into `acc` (n_aux, n_t) IN PLACE and
    return the total EXCLUDED power (records whose kernel has zero
    in-window support).

    Each record deposits a discretely-normalized kernel along the time
    axis at row aux[i]: a Gaussian of std sigma[i] evaluated at the bin
    centers (the analytic envelope), or a single-bin delta (histogram
    mode / a CW record / any kernel with sigma < dt/6, whose +-3 sigma
    support fits inside one bin — discretely it IS a delta, and
    evaluating it would underflow: see below). NORMALIZATION IS THE
    ENERGY-CONSERVATION INVARIANT: each record's weights over the bins it
    actually lands in (including kernels clipped by the window edge) are
    rescaled to sum to exactly power[i], so sum(acc) == summed in-window
    record power to float64 rounding.

    Vectorized in chunks (records can be 1e7): callers pre-sort by sigma
    so each chunk's bin-window width (±3 sigma of the chunk max) matches
    its own kernels; the per-chunk work array is (m, 2K+1) and m is
    shrunk to keep it bounded. No Python loop over records."""
    n_t = acc.shape[1]
    n = len(t)
    flat = acc.reshape(-1)
    excluded = 0.0
    budget = 16 * 1024 * 1024          # work-array elements per chunk
    i = 0
    while i < n:
        m = min(131072, n - i)
        smax = float(sigma[i + m - 1])          # sorted: chunk max sigma
        halfK = int(np.ceil(3.0 * smax / dt)) if smax > 0.0 else 0
        if halfK > 0 and m * (2 * halfK + 1) > budget:
            m = min(max(1024, budget // (2 * halfK + 1)), n - i)
            smax = float(sigma[i + m - 1])
            halfK = int(np.ceil(3.0 * smax / dt)) if smax > 0.0 else 0
        sl = slice(i, i + m)
        tc, pc, ac, sc = t[sl], power[sl], aux[sl], sigma[sl]
        pos = (tc - t_lo) / dt                  # fractional bin coordinate
        # sigma < dt/6 -> delta: a fs-scale kernel binned over a ns-scale
        # window (auto window spanning several path lengths) evaluates
        # exp(-0.5*(offset/sigma)^2) at offsets of tens of sigma — rs goes
        # subnormal, pc/rs overflows to inf and 0*inf NaNs the neighbor
        # bins. At the dt/6 boundary the worst-case bin-centre offset is
        # 3 sigma (exp(-4.5) ~ 0.011), a safe 12x margin from underflow.
        z = sc < dt / 6.0
        if np.any(z):                           # delta records: one bin
            # right-edge-INCLUSIVE like np.histogram: a record at exactly
            # t_hi (the un-padded histogram auto window's own maximum)
            # lands in the last bin instead of being excluded
            ok = (pos[z] >= 0.0) & (pos[z] <= n_t)
            j = np.clip(np.floor(pos[z]).astype(np.int64), 0, n_t - 1)
            excluded += float(np.sum(pc[z][~ok]))
            np.add.at(flat, ac[z][ok] * n_t + j[ok],
                      pc[z][ok].astype(flat.dtype, copy=False))
        g = ~z
        if np.any(g):                           # Gaussian records
            jc = np.floor(pos[g]).astype(np.int64)
            jj = jc[:, None] + np.arange(-halfK, halfK + 1)[None, :]
            t_cent = t_lo + (jj + 0.5) * dt
            w = np.exp(-0.5 * ((t_cent - tc[g][:, None])
                               / sc[g][:, None]) ** 2)
            valid = (jj >= 0) & (jj < n_t)
            w = np.where(valid, w, 0.0)
            rs = w.sum(axis=1)
            # floor guards mixed-width chunks (halfK from the chunk-max
            # sigma can reach a narrow record tens of ITS sigma out): a
            # kernel with only subnormal in-window support books as
            # excluded instead of dividing by a subnormal. In-window
            # Gaussians (sigma >= dt/6 here) have rs >= exp(-4.5).
            okr = rs > 1e-12
            excluded += float(np.sum(pc[g][~okr]))
            scale = np.zeros_like(rs)
            scale[okr] = pc[g][okr] / rs[okr]
            w *= scale[:, None]
            msk = valid & okr[:, None]
            np.add.at(flat, (ac[g][:, None] * n_t + jj)[msk],
                      w[msk].astype(flat.dtype, copy=False))
        i += m
    return excluded


class _IncoherentAccumMixin:
    """Shared incoherent-power splat + per-key detected tally.

    A host grid must provide: to_grid(points) -> (fx, fy) fractional pixel
    coords, self.W/self.H (pixel counts), self.inc (spectral_bins, H, W),
    self.spectral_bins, self.lam_lo/self.lam_hi, and the tally dicts
    self.detected_incoherent / self.detected_incoherent_n. The planar and
    curved grids share EXACTLY this math (bilinear splat + unique-key power
    sum) so the two detector families book detected power identically."""

    def _init_time(self, time_rec):
        """Time-product state (pulsed-optics P4). time_rec: None (off — the
        pre-existing zero-overhead default) or {'envelope': 'analytic' |
        'histogram'} to buffer compact arrival records in both deposit
        paths for finalize_time."""
        self.time_record = time_rec is not None
        self.time_envelope = (time_rec or {}).get("envelope", "analytic")
        self.time_records = []      # chunked-append: one dict per deposit
        self.time_data = {}         # finalize_time: dataset name -> array
        self.time_attrs = {}        # finalize_time: .h5 attrs

    def lam_bin(self, lam):
        b = ((lam - self.lam_lo) / max(self.lam_hi - self.lam_lo, 1e-30)
             * self.spectral_bins).astype(int)
        return np.clip(b, 0, self.spectral_bins - 1)

    def _record_time_arrivals(self, fx, fy, lam, power, source_id,
                              lam_stratum, gopl, gdd):
        """Time-product arrival records (pulsed-optics P4): appended by BOTH
        deposit paths when this grid was built with time recording on AND
        the trace tracked time (gopl is not None). Compact chunked-append
        columns (one dict of arrays per deposit call), sized for 1e7+
        records: t f64, power f64 (f64 so the discrete energy-conservation
        invariant holds to 1e-12 against the detected tallies), fx/fy/lam
        f32, source_id/lam_stratum i16, gdd f32 (analytic envelope only).
        fx/fy are the raw fractional pixel coords to_grid returns (pixel j
        spans [j, j+1)); binning happens once, in finalize_time."""
        n = len(power)
        if n == 0:
            return
        rec = {
            "t": np.asarray(gopl, dtype=np.float64) / C_LIGHT_MPS,
            "fx": np.asarray(fx, dtype=np.float32),
            "fy": np.asarray(fy, dtype=np.float32),
            "lam": np.asarray(lam, dtype=np.float32),
            "power": np.asarray(power, dtype=np.float64).copy(),
            "source_id": np.broadcast_to(
                np.asarray(source_id), (n,)).astype(np.int16),
            "lam_stratum": np.broadcast_to(
                np.asarray(lam_stratum), (n,)).astype(np.int16),
        }
        if self.time_envelope == "analytic":
            rec["gdd"] = (np.asarray(gdd, dtype=np.float32) if gdd is not None
                          else np.zeros(n, dtype=np.float32))
        self.time_records.append(rec)

    def deposit_incoherent(self, points, power, lam,
                          source_id=None, lam_stratum=None, pol_stratum=None,
                          gopl=None, gdd=None):
        """Bilinear splat of ray power [W] at surface points (self.inc splat
        math is UNCHANGED between planar and curved — only to_grid differs).
        source_id/lam_stratum/pol_stratum are optional (None preserves the
        pre-existing call signature) per-ray keys used only to accumulate
        detected_incoherent[(s, l, p)] += power_sum, mirroring
        add_gather_samples' detected_geometric tally so the two populations
        combine under the same key shape. gopl/gdd (track_time only, also
        optional): per-ray group path [m] / accumulated GDD [s^2] at the
        hit, consumed ONLY by the time-product arrival recording — the
        splat and every tally are untouched by their presence."""
        fx, fy = self.to_grid(points)
        if self.time_record and gopl is not None:
            self._record_time_arrivals(fx, fy, lam, power, source_id,
                                       lam_stratum, gopl, gdd)
        fx = fx - 0.5
        fy = fy - 0.5
        x0 = np.floor(fx).astype(int)
        y0 = np.floor(fy).astype(int)
        wx = fx - x0
        wy = fy - y0
        b = self.lam_bin(lam)
        for dx, dy, w in ((0, 0, (1 - wx) * (1 - wy)),
                          (1, 0, wx * (1 - wy)),
                          (0, 1, (1 - wx) * wy),
                          (1, 1, wx * wy)):
            xi = x0 + dx
            yi = y0 + dy
            ok = (xi >= 0) & (xi < self.W) & (yi >= 0) & (yi < self.H)
            np.add.at(self.inc, (b[ok], yi[ok], xi[ok]),
                      power[ok] * w[ok])
        if source_id is None or len(power) == 0:
            return
        keys = np.stack([np.asarray(source_id), np.asarray(lam_stratum),
                         np.asarray(pol_stratum)], axis=1)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, power)
        counts = np.zeros(len(uniq), dtype=np.int64)
        np.add.at(counts, inv, 1)
        for row, p, n in zip(uniq, sums, counts):
            key = (int(row[0]), int(row[1]), int(row[2]))
            self.detected_incoherent[key] = (
                self.detected_incoherent.get(key, 0.0) + float(p))
            self.detected_incoherent_n[key] = (
                self.detected_incoherent_n.get(key, 0) + int(n))

    # ------------------------------------------------------------------
    # Time products (pulsed-optics P4)
    # ------------------------------------------------------------------
    def merged_time_records(self):
        """Concatenated arrival-record columns, or None when nothing was
        recorded."""
        if not self.time_records:
            return None
        return {k: np.concatenate([r[k] for r in self.time_records])
                for k in self.time_records[0]}

    def finalize_time(self, cfg):
        """Bin the buffered arrival records into the SELECTED time products
        (cfg from run_trace.build_time_cfg: products/bins/window/envelope/
        cube_res + the per-source pulse duration and per-(source, stratum)
        angular bandwidth tables). Populates self.time_data (dataset name
        -> array, consumed by run_trace.save_detectors) and self.time_attrs.

        Products (all stored as arrival-power DENSITIES [W/s] along t, so
        sum(product) * dt == summed in-window record power [W]):
          time_profile            (n_t,)            float64
          time_profile_by_source  (n_sources, n_t)  float64  (with 'pulse')
          time_spectrogram        (spectral_bins, n_t)  float64
          time_streak             (n_t, W)          float64
          time_cube               (n_t, H', W')     float32 (H' = min(H,
                                  cube_res); per-axis binning factor in
                                  the attrs)

        Envelope: 'analytic' (default) — each record deposits a Gaussian
        of FWHM sqrt(tau0^2 + (2 sqrt(2 ln2) |gdd| domega_stratum)^2)
        where tau0 is the record's SOURCE pulse duration (0 for CW — a
        pure delta, histogram-like) and domega_stratum its wavelength
        stratum's angular bandwidth (sources.stratum_domega, mapped via
        (source_id, lam_stratum)); 'histogram' — a plain weighted
        histogram of t. NOTE fringe-resolved timing is out of scope:
        coherent records carry geometric power at geometric arrival time.

        Window: explicit cfg['window'] (seconds), else the exact record
        [t_min, t_max] padded by 3x the widest kernel FWHM. Per-record
        discrete kernel normalization (see _splat_records) makes every
        product conserve energy EXACTLY over the in-window bins, window
        clipping included; fully-out-of-window (or non-finite-t) record
        power is reported in time_excluded_W."""
        products = tuple(cfg.get("products") or ())
        if not products or not self.time_record:
            return
        n_t = int(cfg["bins"])
        envelope = cfg.get("envelope") or "analytic"
        n_src = max(int(cfg.get("n_sources", 1)), 1)
        recs = self.merged_time_records() or {
            "t": np.zeros(0), "fx": np.zeros(0, np.float32),
            "fy": np.zeros(0, np.float32), "lam": np.zeros(0, np.float32),
            "power": np.zeros(0), "source_id": np.zeros(0, np.int16),
            "lam_stratum": np.zeros(0, np.int16)}
        n_all = len(recs["t"])
        finite = np.isfinite(recs["t"])
        excl_nonfinite = float(np.sum(recs["power"][~finite]))
        if not np.all(finite):
            recs = {k: v[finite] for k, v in recs.items()}
        t = recs["t"]
        power = recs["power"]
        sid = np.clip(recs["source_id"].astype(np.int64), 0, n_src - 1)

        # per-record kernel FWHM
        if envelope == "analytic" and len(t):
            tau0 = np.asarray(cfg["tau0_by_source"], dtype=np.float64)
            dom = np.asarray(cfg["domega"], dtype=np.float64)
            k_i = np.clip(recs["lam_stratum"].astype(np.int64), 0,
                          dom.shape[1] - 1)
            gdd = recs.get("gdd")
            gdd = np.abs(np.asarray(gdd, dtype=np.float64)) \
                if gdd is not None else np.zeros(len(t))
            fwhm = np.sqrt(tau0[sid] ** 2
                           + (_FWHM_SIGMA * gdd * dom[sid, k_i]) ** 2)
        else:
            fwhm = np.zeros(len(t))
        sigma = fwhm / _FWHM_SIGMA

        window = cfg.get("window")
        if window is not None:
            t_lo, t_hi = float(window[0]), float(window[1])
        elif len(t):
            pad = 3.0 * float(fwhm.max()) if len(fwhm) else 0.0
            t_lo = float(t.min()) - pad
            t_hi = float(t.max()) + pad
        else:
            t_lo, t_hi = 0.0, 1e-9
        if not (t_hi > t_lo):
            # a single delta arrival under histogram mode: open a minimal
            # window so dt stays finite
            eps = max(1e-12, abs(t_lo) * 1e-9)
            t_lo, t_hi = t_lo - eps, t_hi + eps
        dt = (t_hi - t_lo) / n_t

        # power-weighted arrival-time percentiles (diagnostic attrs)
        if len(t) and float(power.sum()) > 0.0:
            order = np.argsort(t)
            cw = np.cumsum(power[order])
            cw = cw / cw[-1]
            t_p001 = float(np.interp(1e-3, cw, t[order]))
            t_p999 = float(np.interp(1.0 - 1e-3, cw, t[order]))
        else:
            t_p001 = t_p999 = t_lo

        # sort by kernel width once; every product splats the same sorted
        # records with a different row index (aux)
        srt = np.argsort(sigma, kind="stable")
        t_s, p_s, sig_s = t[srt], power[srt], sigma[srt]
        fx_s = recs["fx"][srt].astype(np.float64)
        fy_s = recs["fy"][srt].astype(np.float64)
        lam_s = recs["lam"][srt].astype(np.float64)
        sid_s = sid[srt]

        data = {}
        excl_win = None
        if "pulse" in products:
            acc = np.zeros((n_src, n_t))
            excl_win = _splat_records(acc, t_s, sig_s, p_s, sid_s, t_lo, dt)
            data["time_profile"] = acc.sum(axis=0) / dt
            data["time_profile_by_source"] = acc / dt
        if "spectrogram" in products:
            acc = np.zeros((self.spectral_bins, n_t))
            e = _splat_records(acc, t_s, sig_s, p_s,
                               self.lam_bin(lam_s), t_lo, dt)
            excl_win = e if excl_win is None else excl_win
            data["time_spectrogram"] = acc / dt
        if "streak" in products:
            xp = np.clip(np.floor(fx_s).astype(np.int64), 0, self.W - 1)
            acc = np.zeros((self.W, n_t))
            e = _splat_records(acc, t_s, sig_s, p_s, xp, t_lo, dt)
            excl_win = e if excl_win is None else excl_win
            data["time_streak"] = np.ascontiguousarray(acc.T) / dt
        cube_res = int(cfg.get("cube_res") or 256)
        if "cube" in products:
            H2, W2 = min(self.H, cube_res), min(self.W, cube_res)
            yp = np.clip(np.floor(fy_s * (H2 / self.H)).astype(np.int64),
                         0, H2 - 1)
            xp = np.clip(np.floor(fx_s * (W2 / self.W)).astype(np.int64),
                         0, W2 - 1)
            acc = np.zeros((H2 * W2, n_t), dtype=np.float32)
            e = _splat_records(acc, t_s, sig_s, p_s, yp * W2 + xp, t_lo, dt)
            excl_win = e if excl_win is None else excl_win
            data["time_cube"] = np.ascontiguousarray(np.transpose(
                acc.reshape(H2, W2, n_t), (2, 0, 1))) / np.float32(dt)
        excl_win = excl_win or 0.0

        self.time_data = data
        self.time_attrs = {
            "t_lo_s": float(t_lo), "t_hi_s": float(t_hi),
            "time_bins": int(n_t), "time_dt_s": float(dt),
            "time_envelope": str(envelope),
            "time_products": ",".join(products),
            "time_cube_res": int(cube_res),
            "t_p001_s": t_p001, "t_p999_s": t_p999,
            "time_n_records": int(n_all),
            "time_total_W": float(np.sum(p_s)) - float(excl_win),
            "time_excluded_W": float(excl_win) + excl_nonfinite,
            "time_window_explicit": bool(window is not None),
        }
        if "cube" in products:
            self.time_attrs.update({
                "time_cube_H": int(min(self.H, cube_res)),
                "time_cube_W": int(min(self.W, cube_res)),
                "time_cube_bin_y": float(self.H / min(self.H, cube_res)),
                "time_cube_bin_x": float(self.W / min(self.W, cube_res)),
            })


class DetectorGrid(_IncoherentAccumMixin):
    def __init__(self, face, resolution, spectral_bins, lam_range,
                 label="", time_rec=None):
        surf = face.surface
        if surf.__class__.__name__ != "Plane":
            raise NotImplementedError(
                "detector face %s is not planar — v1 supports planar "
                "detector screens only (see future.md)" % face.id)
        self.face = face
        self.label = label or face.id
        n = surf.n
        ax = np.zeros(3)
        ax[int(np.argmin(np.abs(n)))] = 1.0
        x = ax - np.dot(ax, n) * n
        self.xhat = x / np.linalg.norm(x)
        self.yhat = np.cross(n, self.xhat)
        self.normal = n

        # extent from the trim polygon in this frame
        allpts = np.concatenate(
            [np.asarray(lp) for lp in face.trim.loops]) \
            if face.trim.mode == "polygon" else None
        if allpts is None:
            raise NotImplementedError(
                "detector face %s trim did not resolve to a polygon" % face.id)
        # trim loops are stored in the face's canonical plane UV (t1,t2 of
        # the Plane); convert loop UV -> world -> grid frame
        world = [surf.origin + lp[:, 0:1] * surf.t1 + lp[:, 1:2] * surf.t2
                 for lp in face.trim.loops]
        pts2 = [np.stack([w @ self.xhat, w @ self.yhat], axis=-1)
                for w in world]
        allp = np.concatenate(pts2)
        self.x_lo, self.y_lo = allp.min(axis=0)
        self.x_hi, self.y_hi = allp.max(axis=0)
        span_x = self.x_hi - self.x_lo
        span_y = self.y_hi - self.y_lo
        span = max(span_x, span_y)
        # square pixels; grid covers the bbox with resolution along the
        # longer side
        self.pixel_m = span / resolution
        self.W = max(8, int(np.ceil(span_x / self.pixel_m)))
        self.H = max(8, int(np.ceil(span_y / self.pixel_m)))
        self.spectral_bins = spectral_bins
        self.lam_lo, self.lam_hi = lam_range
        self.inc = np.zeros((spectral_bins, self.H, self.W))
        # gather samples per (source_id, lam_stratum): lists of arrays
        self.samples = {}
        self.detected_geometric = {}       # (source_id) -> W
        # incoherent per-(source_id, lam_stratum, pol_stratum) detected
        # power tally, kept in lockstep with detected_geometric above (same
        # key shape) so post-processing can merge the two without special
        # casing which population a key came from.
        self.detected_incoherent = {}
        self.detected_incoherent_n = {}     # (source_id, lam, pol) -> ray count
        # --export-rays: per-detector-event landing records (list of dicts
        # of per-ray arrays), populated by Tracer._export_records when the
        # trace config has export_rays on. Empty otherwise (zero overhead).
        self.ray_records = []
        self._init_time(time_rec)

        # trim mask in pixel space
        xs = self.x_lo + (np.arange(self.W) + 0.5) * self.pixel_m
        ys = self.y_lo + (np.arange(self.H) + 0.5) * self.pixel_m
        gx, gy = np.meshgrid(xs, ys)
        pts_world = (surf.origin[None, :]
                     + gx.reshape(-1, 1) * self.xhat
                     + gy.reshape(-1, 1) * self.yhat)
        # shift from plane origin: grid frame coords are absolute
        # projections; rebuild world points properly:
        pts_world = (gx.reshape(-1, 1) * self.xhat
                     + gy.reshape(-1, 1) * self.yhat)
        # add the component of the plane origin normal to (xhat,yhat) span
        n_comp = surf.origin - (surf.origin @ self.xhat) * self.xhat \
            - (surf.origin @ self.yhat) * self.yhat
        pts_world = pts_world + n_comp
        self.pixel_centers = pts_world.reshape(self.H, self.W, 3)
        uv = surf.to_uv(pts_world)
        self.mask = face.trim.contains(uv).reshape(self.H, self.W)

    # ------------------------------------------------------------------
    def to_grid(self, points):
        """World points on the plane -> fractional pixel coords (x, y)."""
        gx = points @ self.xhat - self.x_lo
        gy = points @ self.yhat - self.y_lo
        return gx / self.pixel_m, gy / self.pixel_m

    def add_gather_samples(self, source_id, lam_stratum, pol_stratum,
                           pos, direction, Es, Ep, s_hat, lam, opl, power,
                           scattered, dA=None, pos_hit=None, gopl=None,
                           gdd=None):
        """Record coherent Huygens samples (ray states at the segment start
        that reached this detector). Keyed per (source, wavelength stratum,
        polarization stratum): different strata never interfere.

        dA: optional per-sample wavefront patch areas [m^2] from ray-
        differential tracking (--ray-differentials); NaN entries fall back
        to the source-referenced sample_area in the gather.

        pos_hit/gopl/gdd (track_time only, all optional): the GEOMETRIC
        hit point on the detector and the group path [m] / accumulated GDD
        [s^2] advanced to it. Consumed ONLY by the time-product arrival
        recording — coherent rays contribute their GEOMETRIC power at
        their geometric arrival time (fringe-resolved timing is out of
        scope: the Huygens gather reconstructs stationary interference,
        and its phase math is untouched by these kwargs — the sample dicts
        below are byte-identical with or without them)."""
        key = (int(source_id), int(lam_stratum), int(pol_stratum))
        if self.time_record and gopl is not None and pos_hit is not None:
            fx, fy = self.to_grid(pos_hit)
            self._record_time_arrivals(fx, fy, lam, power, key[0], key[1],
                                       gopl, gdd)
        rec = self.samples.setdefault(key, [])
        if dA is None:
            dA = np.full(len(lam), np.nan)
        rec.append({"pos": pos, "dir": direction, "Es": Es, "Ep": Ep,
                    "s_hat": s_hat, "lam": lam, "opl": opl,
                    "power": power, "scattered": scattered, "dA": dA})
        self.detected_geometric[key] = (
            self.detected_geometric.get(key, 0.0) + float(np.sum(power)))

    def merged_samples(self):
        """{(source, stratum): dict of concatenated arrays}"""
        out = {}
        for key, recs in self.samples.items():
            out[key] = {k: np.concatenate([r[k] for r in recs])
                        if np.ndim(recs[0][k]) else np.array(
                            [r[k] for r in recs])
                        for k in recs[0]}
        return out


class CurvedDetectorGrid(_IncoherentAccumMixin):
    """Detector pixel grid over a trimmed Sphere or Cylinder face.

    Incoherent path only (Phase 10). Pixels are a regular grid in the
    surface's canonical (u, v) parameterization (see surfaces.to_uv):

      * Sphere:   u = azimuth [rad], v = latitude [rad].
                  per-pixel metric area = R^2 * cos(v) * du * dv.
      * Cylinder: u = azimuth [rad], v = axial coordinate [m].
                  per-pixel metric area = R * du * dv (constant).

    Power is splatted (bilinear) into self.inc exactly like the planar
    DetectorGrid — the ONLY difference is to_grid maps world hits through
    surf.to_uv instead of an in-plane projection, so detected-power tallies
    and the energy-audit booking are byte-for-byte the same interface the
    tracer's _detector_event drives. post_process divides self.inc by the
    per-pixel area map (self.pixel_area_map) to get irradiance, so the
    total detected POWER is grid-geometry-independent.

    Coherent Huygens gather is NOT supported on curved screens: the planar
    gather kernel assumes a flat aperture. add_gather_samples raises; use
    coherent=false sources on a curved detector.
    """

    def __init__(self, face, resolution, spectral_bins, lam_range,
                 label="", time_rec=None):
        surf = face.surface
        stype = surf.__class__.__name__
        if stype not in ("Sphere", "Cylinder"):
            raise NotImplementedError(
                "CurvedDetectorGrid supports Sphere/Cylinder faces only "
                "(got %s on face %s)" % (stype, face.id))
        self.face = face
        self.label = label or face.id
        self.surface = surf
        self.surface_type = "sphere" if stype == "Sphere" else "cylinder"
        self.is_sphere = stype == "Sphere"
        self.radius = float(surf.r)
        self.periodic_u = bool(getattr(surf, "periodic_u", False))

        # trimmed (u, v) parameter range: the trim loops are already stored
        # in this surface's canonical uv (unwrapped u for periodic faces),
        # so their bbox is the face's uv extent.
        allp = np.concatenate([np.asarray(lp) for lp in face.trim.loops])
        self.u_lo, self.v_lo = (float(x) for x in allp.min(axis=0))
        self.u_hi, self.v_hi = (float(x) for x in allp.max(axis=0))
        span_u = self.u_hi - self.u_lo
        span_v = self.v_hi - self.v_lo

        # metric spans -> square-ish pixels: arc length across u is R*du for
        # both classes; across v it is R*dv (sphere latitude) or dv (cylinder
        # axial). resolution counts pixels along the longer metric side.
        arc_u = self.radius * span_u
        arc_v = self.radius * span_v if self.is_sphere else span_v
        span = max(arc_u, arc_v, 1e-30)
        self.pixel_m = span / resolution
        self.W = max(8, int(np.ceil(arc_u / self.pixel_m)))
        self.H = max(8, int(np.ceil(arc_v / self.pixel_m)))
        self.du = span_u / self.W          # parameter step per pixel (u)
        self.dv = span_v / self.H          # parameter step per pixel (v)

        self.spectral_bins = spectral_bins
        self.lam_lo, self.lam_hi = lam_range
        self.inc = np.zeros((spectral_bins, self.H, self.W))

        # pixel-center uv, world points, containment mask, and TRUE metric
        # per-pixel area map.
        us = self.u_lo + (np.arange(self.W) + 0.5) * self.du
        vs = self.v_lo + (np.arange(self.H) + 0.5) * self.dv
        gu, gv = np.meshgrid(us, vs)                       # (H, W)
        uvc = np.stack([gu.reshape(-1), gv.reshape(-1)], axis=-1)
        self.mask = face.trim.contains(uvc).reshape(self.H, self.W)
        self.pixel_centers = surf.uv_to_xyz(
            gu.reshape(-1), gv.reshape(-1)).reshape(self.H, self.W, 3)
        if self.is_sphere:
            area = self.radius ** 2 * np.cos(gv) * self.du * self.dv
        else:
            area = np.full((self.H, self.W), self.radius * self.du * self.dv)
        # masked pixels collect nothing; guard the divide in post with a
        # positive area everywhere the mask is False (inc is 0 there anyway).
        self.pixel_area_map = np.where(self.mask, area, 0.0)

        # per-(source, lam, pol) tallies, same key shape / semantics as the
        # planar grid (detected_geometric stays empty: no coherent path).
        self.samples = {}
        self.detected_geometric = {}
        self.detected_incoherent = {}
        self.detected_incoherent_n = {}
        self.ray_records = []
        self._init_time(time_rec)

        # nominal in-plane frame at the arc center so --export-rays' meta
        # (xhat/yhat/normal/x_lo/y_lo) stays populated; NOT used by the splat.
        u_c = 0.5 * (self.u_lo + self.u_hi)
        v_c = 0.5 * (self.v_lo + self.v_hi)
        center = surf.uv_to_xyz(np.array([u_c]), np.array([v_c]))[0]
        self.normal = face.normal_out_of_solid(center[None])[0]
        du_dir = surf.uv_to_xyz(np.array([u_c + 1e-4]),
                                np.array([v_c]))[0] - center
        nrm = np.linalg.norm(du_dir)
        self.xhat = du_dir / nrm if nrm > 0 else np.array([1.0, 0.0, 0.0])
        self.yhat = np.cross(self.normal, self.xhat)
        yn = np.linalg.norm(self.yhat)
        self.yhat = self.yhat / yn if yn > 0 else np.array([0.0, 1.0, 0.0])
        self.x_lo, self.y_lo = self.u_lo, self.v_lo

    def to_grid(self, points):
        """World hits -> fractional pixel coords (fu along W, fv along H) via
        the surface's canonical uv. Periodic-u hits are unwrapped into the
        face's u-range so bilinear neighbours near the arc edges land right;
        out-of-grid neighbours are dropped by the splat's own range mask."""
        uv = self.surface.to_uv(points)
        u = uv[..., 0]
        v = uv[..., 1]
        if self.periodic_u:
            u = self.u_lo + np.mod(u - self.u_lo, 2.0 * np.pi)
        fu = (u - self.u_lo) / self.du
        fv = (v - self.v_lo) / self.dv
        return fu, fv

    def add_gather_samples(self, *args, **kwargs):
        raise NotImplementedError(
            "coherent gather on curved detectors unsupported — use "
            "coherent=false sources on this detector")

    def merged_samples(self):
        # no coherent population is ever recorded (add_gather_samples raises),
        # so gather.render_coherent short-circuits on the empty dict.
        return {}


# ---------------------------------------------------------------------------
# Spectral cube -> color rendering (CIE 1931 -> sRGB)
# ---------------------------------------------------------------------------
# Compact CIE 1931 2-deg color matching functions, 380-780 nm @ 5 nm
# (Wyszecki & Stiles); linear interpolation between entries.
_CIE_LAM = np.arange(380.0, 785.0, 5.0)
_CIE_X = np.array([
    0.0014, 0.0022, 0.0042, 0.0077, 0.0143, 0.0232, 0.0435, 0.0776, 0.1344,
    0.2148, 0.2839, 0.3285, 0.3483, 0.3481, 0.3362, 0.3187, 0.2908, 0.2511,
    0.1954, 0.1421, 0.0956, 0.0580, 0.0320, 0.0147, 0.0049, 0.0024, 0.0093,
    0.0291, 0.0633, 0.1096, 0.1655, 0.2257, 0.2904, 0.3597, 0.4334, 0.5121,
    0.5945, 0.6784, 0.7621, 0.8425, 0.9163, 0.9786, 1.0263, 1.0567, 1.0622,
    1.0456, 1.0026, 0.9384, 0.8544, 0.7514, 0.6424, 0.5419, 0.4479, 0.3608,
    0.2835, 0.2187, 0.1649, 0.1212, 0.0874, 0.0636, 0.0468, 0.0329, 0.0227,
    0.0158, 0.0114, 0.0081, 0.0058, 0.0041, 0.0029, 0.0020, 0.0014, 0.0010,
    0.0007, 0.0005, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000])
_CIE_Y = np.array([
    0.0000, 0.0001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0012, 0.0022, 0.0040,
    0.0073, 0.0116, 0.0168, 0.0230, 0.0298, 0.0380, 0.0480, 0.0600, 0.0739,
    0.0910, 0.1126, 0.1390, 0.1693, 0.2080, 0.2586, 0.3230, 0.4073, 0.5030,
    0.6082, 0.7100, 0.7932, 0.8620, 0.9149, 0.9540, 0.9803, 0.9950, 1.0000,
    0.9950, 0.9786, 0.9520, 0.9154, 0.8700, 0.8163, 0.7570, 0.6949, 0.6310,
    0.5668, 0.5030, 0.4412, 0.3810, 0.3210, 0.2650, 0.2170, 0.1750, 0.1382,
    0.1070, 0.0816, 0.0610, 0.0446, 0.0320, 0.0232, 0.0170, 0.0119, 0.0082,
    0.0057, 0.0041, 0.0029, 0.0021, 0.0015, 0.0010, 0.0007, 0.0005, 0.0004,
    0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000])
_CIE_Z = np.array([
    0.0065, 0.0105, 0.0201, 0.0362, 0.0679, 0.1102, 0.2074, 0.3713, 0.6456,
    1.0391, 1.3856, 1.6230, 1.7471, 1.7826, 1.7721, 1.7441, 1.6692, 1.5281,
    1.2876, 1.0419, 0.8130, 0.6162, 0.4652, 0.3533, 0.2720, 0.2123, 0.1582,
    0.1117, 0.0782, 0.0573, 0.0422, 0.0298, 0.0203, 0.0134, 0.0087, 0.0057,
    0.0039, 0.0027, 0.0021, 0.0018, 0.0017, 0.0014, 0.0011, 0.0010, 0.0008,
    0.0006, 0.0003, 0.0002, 0.0002, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])

_XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                         [-0.9689, 1.8758, 0.0415],
                         [0.0557, -0.2040, 1.0570]])


def cie_xyz_weights(lam_nm):
    """(3,) CIE xyz weights for wavelength(s) in nm (vectorized)."""
    x = np.interp(lam_nm, _CIE_LAM, _CIE_X, left=0.0, right=0.0)
    y = np.interp(lam_nm, _CIE_LAM, _CIE_Y, left=0.0, right=0.0)
    z = np.interp(lam_nm, _CIE_LAM, _CIE_Z, left=0.0, right=0.0)
    return np.stack([x, y, z], axis=-1)


def spectral_cube_to_srgb(cube, lam_lo, lam_hi, gamma=True,
                          percentile=99.5):
    """(bins, H, W) power cube -> (H, W, 3) sRGB in [0,1].

    Brightness is normalized to the given percentile of the luminance map
    (so a hot focal spot does not black out the rest of the image); the
    linear irradiance data itself is saved separately in HDF5.
    """
    bins = cube.shape[0]
    lam_c = lam_lo + (np.arange(bins) + 0.5) * (lam_hi - lam_lo) / bins
    w = cie_xyz_weights(lam_c / 1e-9 if lam_c.max() < 1e-3 else lam_c)
    XYZ = np.tensordot(w, cube, axes=(0, 0))          # (3, H, W)
    rgb = np.tensordot(_XYZ_TO_SRGB, XYZ, axes=(1, 0))
    rgb = np.clip(rgb, 0.0, None)
    lum = XYZ[1]
    scale = np.percentile(lum[lum > 0], percentile) if np.any(lum > 0) \
        else 1.0
    if scale <= 0:
        scale = 1.0
    rgb = np.clip(rgb / scale, 0.0, 1.0)
    if gamma:
        rgb = np.where(rgb <= 0.0031308, 12.92 * rgb,
                       1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.moveaxis(rgb, 0, -1)


_LUMINOUS_EFFICACY = 683.002    # lm/W, CIE photopic V(lambda) peak (555 nm)


def spectral_cube_to_lux(cube, lam_lo, lam_hi, pixel_area_m2):
    """(bins, H, W) power cube [W] -> (lux_map (H, W) [lx], luminous_flux
    [lm]). Photometric weighting uses the y-bar (photopic V(lambda)) row
    of cie_xyz_weights, which already handles the m-vs-nm lam_lo/lam_hi
    heuristic the same way spectral_cube_to_srgb does."""
    bins = cube.shape[0]
    lam_c = lam_lo + (np.arange(bins) + 0.5) * (lam_hi - lam_lo) / bins
    v = cie_xyz_weights(lam_c / 1e-9 if lam_c.max() < 1e-3 else lam_c)[:, 1]
    weighted = np.tensordot(v, cube, axes=(0, 0))        # (H, W), watts
    luminous_flux_lm = _LUMINOUS_EFFICACY * float(weighted.sum())
    lux_map = _LUMINOUS_EFFICACY * weighted / pixel_area_m2
    return lux_map, luminous_flux_lm


# CODATA 2018 exact SI defining constants
_Q_E = 1.602176634e-19          # elementary charge [C]
_H_PLANCK = 6.62607015e-34      # Planck constant [J s]
_C_LIGHT = 299792458.0          # speed of light [m/s]


def spectral_cube_to_photocurrent(cube, lam_lo, lam_hi, qe_lam_um, qe_vals):
    """(bins, ...) power cube [W] weighted by a detector quantum-efficiency
    curve -> (photocurrent_A, qe_weighted_power_W, coverage_frac).

    lam_lo/lam_hi are the cube's spectral extent in METRES (h5 attrs); the
    QE curve is given as (qe_lam_um [um], qe_vals [fraction]) from
    optprops.load_detectors. Per bin-center wavelength lambda_c:

      QE(lambda_c)  via np.interp with left=0/right=0 -- a cube extending
                    past the tabulated range simply contributes 0 there
                    (NOT interp_hard: a display stage must never crash),
      R(lambda)     = QE * q * lambda / (h*c)  responsivity [A/W],
      photocurrent  = sum_bins R(lambda_c) * P_bin  [A],
      qe_weighted_power = sum_bins QE(lambda_c) * P_bin  [W],
      coverage_frac = (power in bins whose center is inside the QE table)
                      / total power  (0 if total power is 0).

    Power per bin is summed over all trailing (pixel) axes, so a (bins,H,W)
    cube and a (bins,) spectrum give the same scalars."""
    cube = np.asarray(cube, dtype=np.float64)
    bins = cube.shape[0]
    lam_c_m = lam_lo + (np.arange(bins) + 0.5) * (lam_hi - lam_lo) / bins
    lam_c_um = lam_c_m * 1e6
    qe_lam_um = np.asarray(qe_lam_um, dtype=np.float64)
    qe_vals = np.asarray(qe_vals, dtype=np.float64)
    qe_c = np.interp(lam_c_um, qe_lam_um, qe_vals, left=0.0, right=0.0)
    resp = qe_c * _Q_E * lam_c_m / (_H_PLANCK * _C_LIGHT)     # [A/W] per bin
    p_bin = cube.reshape(bins, -1).sum(axis=1)                # [W] per bin
    photocurrent_A = float(np.dot(resp, p_bin))
    qe_weighted_power_W = float(np.dot(qe_c, p_bin))
    total_W = float(p_bin.sum())
    inside = (lam_c_um >= qe_lam_um[0]) & (lam_c_um <= qe_lam_um[-1])
    coverage_frac = float(p_bin[inside].sum() / total_W) if total_W != 0.0 \
        else 0.0
    return photocurrent_A, qe_weighted_power_W, coverage_frac
