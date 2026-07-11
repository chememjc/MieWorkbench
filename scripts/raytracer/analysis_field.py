# =============================================================================
# analysis_field.py -- Phase 2 (lowhanging.md #5, tier A): PSF / FFT-MTF /
# encircled-energy math for a coherent detector's complex field maps.
#
# Numpy-only, float64 (complex128 for fields), pure array-in/array-out --
# no h5py, no tracer imports. The consumer (a later post_process.py wiring
# step) reads the coherent per-key field pair from a detector .h5's
# 'fields/<s>_<l>_<p>/{Ex,Ey}' groups -- each an (H,W) complex128 dataset,
# same convention save_detectors() writes for render_stokes_maps -- and the
# per-key pixel pitch from the file's 'pixel_m' attr (metres/pixel, square
# pixels). This module only ever sees the resulting (H,W) arrays + pixel_m.
#
# PSF: incoherent irradiance |Ex|^2 + |Ey|^2 of ONE coherent gather key (a
# single (source, lambda-stratum, pol-stratum) -- summing keys first would
# incoherently blur an otherwise-coherent spot, so that choice is left to
# the caller).
#
# MTF: the modulus of the (DC-normalized) 2-D FFT of the PSF -- the
# autocorrelation theorem makes this exactly the (incoherent) optical
# transfer function's magnitude, standard diffraction-MTF metrology.
# Frequencies are reported in cycles/mm (the conventional lens-design
# unit); pixel_m is metres/pixel so cycles/metre = fftfreq(N, d=pixel_m)
# and cycles/mm = that * 1e-3.
#
# Encircled/ensquared energy: direct per-pixel radial (Euclidean) or
# per-pixel half-width (Chebyshev/L-infinity, i.e. a square aperture) sort
# + cumulative sum -- no binning, so no extra pixelization error beyond
# the PSF's own sampling (matters for a first-Airy-zero EE check near 84%
# where a coarse radial bin would smear the step).
# =============================================================================
import numpy as np


def psf_from_fields(Ex, Ey):
    """(H,W) complex Ex,Ey (a coherent gather key's field maps) -> (H,W)
    float64 irradiance |Ex|^2 + |Ey|^2. Trivial by construction, but kept
    as the one canonical place callers compute a PSF from fields so the
    convention (sum of BOTH polarization components) stays consistent."""
    Ex = np.asarray(Ex)
    Ey = np.asarray(Ey)
    return (np.abs(Ex) ** 2 + np.abs(Ey) ** 2).astype(np.float64)


def normalize_psf(psf):
    """(H,W) irradiance -> (psf/peak, peak). peak is the raw max value
    (0 -> returns the input unchanged with peak=0.0, no division by zero)."""
    psf = np.asarray(psf, dtype=np.float64)
    peak = float(psf.max()) if psf.size else 0.0
    if peak <= 0:
        return psf.copy(), peak
    return psf / peak, peak


def _power_centroid(img, pixel):
    """(H,W) non-negative-weighted image -> (cx, cy) power centroid in
    PHYSICAL units (same units as `pixel`), measured from the array's
    pixel-center coordinate origin (col 0 / row 0 pixel center = 0,0).
    Falls back to the geometric center if the image sums to <= 0."""
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape
    total = float(img.sum())
    xs = np.arange(W) * pixel
    ys = np.arange(H) * pixel
    if total <= 0:
        return float(xs.mean()), float(ys.mean())
    cx = float(np.sum(xs[np.newaxis, :] * img) / total)
    cy = float(np.sum(ys[:, np.newaxis] * img) / total)
    return cx, cy


def _radial_grid(img, center, pixel):
    """(H,W) img -> (XX, YY, R) physical-unit coordinate grids relative to
    `center` (cx, cy) (power centroid if center is None), pixel-center
    convention matching _power_centroid."""
    H, W = img.shape
    if center is None:
        cx, cy = _power_centroid(img, pixel)
    else:
        cx, cy = center
    xs = np.arange(W) * pixel - cx
    ys = np.arange(H) * pixel - cy
    XX, YY = np.meshgrid(xs, ys)
    R = np.sqrt(XX ** 2 + YY ** 2)
    return XX, YY, R


def radial_profile(img, center=None, pixel=1.0, nbins=None):
    """(H,W) img -> (r_centers, mean_profile), the azimuthal mean of `img`
    binned by radius from `center` (default: the power centroid, robust
    for an off-center spot -- geometric-center binning would smear an
    off-axis PSF across radius bins). `pixel` converts pixel index
    spacing to physical units (metres, mm, whatever the caller uses
    consistently). `nbins` defaults to roughly one bin per pixel of
    radius (min(H,W)//2), i.e. close to the native radial resolution.
    Empty bins (no pixel falls in them) are dropped from the output."""
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape
    _, _, R = _radial_grid(img, center, pixel)
    if nbins is None:
        nbins = max(int(min(H, W) // 2), 1)
    rmax = float(R.max())
    if rmax <= 0:
        return np.zeros(0), np.zeros(0)
    edges = np.linspace(0.0, rmax, nbins + 1)
    r_flat = R.ravel()
    img_flat = img.ravel()
    bin_idx = np.clip(np.digitize(r_flat, edges) - 1, 0, nbins - 1)
    sums = np.bincount(bin_idx, weights=img_flat, minlength=nbins)
    counts = np.bincount(bin_idx, minlength=nbins)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    valid = counts > 0
    mean_profile = np.zeros(nbins)
    mean_profile[valid] = sums[valid] / counts[valid]
    return r_centers[valid], mean_profile[valid]


def mtf2d(psf, pixel_m):
    """(H,W) real PSF + pixel pitch (metres) -> dict:
      mtf          (H,W) float64, |FFT2(psf)| fftshifted and DC-normalized
                   (mtf[DC] == 1.0)
      fx_cy_mm     (W,) full (signed) frequency axis, cycles/mm
      fy_cy_mm     (H,) full (signed) frequency axis, cycles/mm
      tangential   (W,) mtf's DC ROW (the x-frequency slice at fy=0)
      sagittal     (H,) mtf's DC COLUMN (the y-frequency slice at fx=0)
      freq_cy_mm   the positive-frequency half of fx_cy_mm (DC to Nyquist)
                   -- indexes the TANGENTIAL half-slice
      freq_y_cy_mm the positive-frequency half of fy_cy_mm -- indexes the
                   SAGITTAL half-slice (== freq_cy_mm only when H == W; a
                   non-square detector grid has different half-lengths,
                   which used to crash the MTF plotter)
    DC-normalization divides by the DC bin (psf.sum(), i.e. total energy)
    rather than assuming it is exactly 1, so an un-normalized PSF works.
    """
    psf = np.asarray(psf, dtype=np.float64)
    H, W = psf.shape
    F = np.fft.fftshift(np.fft.fft2(psf))
    mag = np.abs(F)
    dc = mag[H // 2, W // 2]
    mtf = mag / dc if dc > 0 else mag
    fx_cy_m = np.fft.fftshift(np.fft.fftfreq(W, d=pixel_m))
    fy_cy_m = np.fft.fftshift(np.fft.fftfreq(H, d=pixel_m))
    fx_cy_mm = fx_cy_m * 1e-3
    fy_cy_mm = fy_cy_m * 1e-3
    tangential = mtf[H // 2, :]
    sagittal = mtf[:, W // 2]
    freq_cy_mm = fx_cy_mm[W // 2:]
    freq_y_cy_mm = fy_cy_mm[H // 2:]
    return {
        "mtf": mtf,
        "fx_cy_mm": fx_cy_mm,
        "fy_cy_mm": fy_cy_mm,
        "tangential": tangential,
        "sagittal": sagittal,
        "freq_cy_mm": freq_cy_mm,
        "freq_y_cy_mm": freq_y_cy_mm,
    }


def mtf50(freq, mtf_slice):
    """(freq, mtf_slice) 1-D arrays (freq ascending, mtf_slice presumed to
    start >= 0.5 at/near DC and fall off) -> the frequency of the first
    downward crossing of 0.5, linearly interpolated between the bracketing
    samples. Returns nan if mtf_slice never drops to/below 0.5 (or is
    already below 0.5 at freq[0], or has fewer than 2 samples)."""
    freq = np.asarray(freq, dtype=np.float64)
    mtf_slice = np.asarray(mtf_slice, dtype=np.float64)
    if freq.size < 2 or mtf_slice[0] < 0.5:
        return float("nan")
    below = np.nonzero(mtf_slice <= 0.5)[0]
    if below.size == 0:
        return float("nan")
    i1 = int(below[0])
    if i1 == 0:
        return float(freq[0])
    i0 = i1 - 1
    m0, m1 = mtf_slice[i0], mtf_slice[i1]
    f0, f1 = freq[i0], freq[i1]
    if m0 == m1:
        return float(f0)
    frac = (m0 - 0.5) / (m0 - m1)
    return float(f0 + frac * (f1 - f0))


def encircled_energy(psf, center=None, pixel=1.0):
    """(H,W) psf -> (radii, ee_fraction), the cumulative fraction of total
    energy inside radius r from `center` (power centroid if None), sorted
    ascending by the EXACT per-pixel radius (no radial binning -- avoids
    smearing a sharp encircled-energy step, e.g. the first Airy zero).
    ee_fraction is normalized so it -> 1.0 at the largest radius (which is
    psf.sum() total, or 0-filled if the psf sums to <= 0)."""
    psf = np.asarray(psf, dtype=np.float64)
    _, _, R = _radial_grid(psf, center, pixel)
    order = np.argsort(R.ravel())
    r_sorted = R.ravel()[order]
    w_sorted = psf.ravel()[order]
    total = float(w_sorted.sum())
    cum = np.cumsum(w_sorted)
    ee = cum / total if total > 0 else np.zeros_like(cum)
    return r_sorted, ee


def ensquared_energy(psf, center=None, pixel=1.0):
    """(H,W) psf -> (half_widths, ee_fraction), the ensquared-energy
    analog of encircled_energy: a "radius" here is the Chebyshev
    (L-infinity) half-width max(|dx|, |dy|) from `center`, i.e. energy
    inside a centered SQUARE of that half-width rather than a circle."""
    psf = np.asarray(psf, dtype=np.float64)
    XX, YY, _ = _radial_grid(psf, center, pixel)
    half = np.maximum(np.abs(XX), np.abs(YY))
    order = np.argsort(half.ravel())
    h_sorted = half.ravel()[order]
    w_sorted = psf.ravel()[order]
    total = float(w_sorted.sum())
    cum = np.cumsum(w_sorted)
    ee = cum / total if total > 0 else np.zeros_like(cum)
    return h_sorted, ee


def ee_radius(radii, ee, frac):
    """(radii, ee) from encircled_energy/ensquared_energy + a target
    fraction (e.g. 0.5/0.8/0.9) -> the interpolated radius at which the
    cumulative curve first reaches `frac` (np.interp against the
    monotonically non-decreasing `ee` curve; clamps to the end radii
    outside [ee.min(), ee.max()], matching np.interp's default behavior)."""
    radii = np.asarray(radii, dtype=np.float64)
    ee = np.asarray(ee, dtype=np.float64)
    return float(np.interp(frac, ee, radii))
