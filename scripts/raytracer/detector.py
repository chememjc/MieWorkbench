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


class DetectorGrid:
    def __init__(self, face, resolution, spectral_bins, lam_range,
                 label=""):
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

    def lam_bin(self, lam):
        b = ((lam - self.lam_lo) / max(self.lam_hi - self.lam_lo, 1e-30)
             * self.spectral_bins).astype(int)
        return np.clip(b, 0, self.spectral_bins - 1)

    def deposit_incoherent(self, points, power, lam,
                          source_id=None, lam_stratum=None, pol_stratum=None):
        """Bilinear splat of ray power [W] at plane points (self.inc splat
        math is UNCHANGED). source_id/lam_stratum/pol_stratum are optional
        (None preserves the pre-existing call signature) per-ray keys used
        only to accumulate detected_incoherent[(s, l, p)] += power_sum,
        mirroring add_gather_samples' detected_geometric tally so the two
        populations combine under the same key shape."""
        fx, fy = self.to_grid(points)
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

    def add_gather_samples(self, source_id, lam_stratum, pol_stratum,
                           pos, direction, Es, Ep, s_hat, lam, opl, power,
                           scattered, dA=None):
        """Record coherent Huygens samples (ray states at the segment start
        that reached this detector). Keyed per (source, wavelength stratum,
        polarization stratum): different strata never interfere.

        dA: optional per-sample wavefront patch areas [m^2] from ray-
        differential tracking (--ray-differentials); NaN entries fall back
        to the source-referenced sample_area in the gather."""
        key = (int(source_id), int(lam_stratum), int(pol_stratum))
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
