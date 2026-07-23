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


def log_annular_power(power_img, pixel, center, n_rings, r_min, r_max):
    """(H,W) PER-PIXEL POWER image [W] (NOT irradiance -- caller multiplies
    by pixel area first, e.g. render_detector's already-computed
    irradiance*pixel_area) -> (edges, ring_power, power_inside_rmin,
    power_outside_rmax):

      edges              (n_rings+1,) log-spaced radii from r_min to r_max,
                         same physical units as `pixel` (mm by convention
                         in this codebase's detector-plane code, but the
                         function itself is unit-agnostic)
      ring_power         (n_rings,) exact SUM (not the mean radial_profile
                         reports -- these are already per-pixel Watts) of
                         power_img over pixels whose radius from `center`
                         falls in ring i's half-open interval
                         [edges[i], edges[i+1]) (the LAST ring is closed on
                         the right, r <= r_max, so no boundary pixel at
                         exactly r_max is dropped)
      power_inside_rmin  sum of pixels with r < r_min (the log scale can't
                         start at r=0, so this bucket + ring_power +
                         power_outside_rmax is what makes the partition
                         exhaustive)
      power_outside_rmax sum of pixels with r > r_max

    `center` follows _radial_grid's convention: (cx, cy) in physical units,
    or None for the power centroid. The four returned quantities are an
    EXACT partition of power_img (every pixel counted exactly once), so
    ring_power.sum() + power_inside_rmin + power_outside_rmax ==
    power_img.sum() to float64 roundoff (~1e-12 relative) regardless of
    n_rings/r_min/r_max -- the "ring power closure" invariant callers are
    expected to report alongside the per-ring table.

    r_min must be > 0 (log spacing has no zero) and r_max > r_min."""
    power_img = np.asarray(power_img, dtype=np.float64)
    r_min = float(r_min)
    r_max = float(r_max)
    n_rings = int(n_rings)
    if r_min <= 0:
        raise ValueError("r_min must be > 0 for log spacing (got %g)" % r_min)
    if r_max <= r_min:
        raise ValueError(
            "r_max must be > r_min (got %g <= %g)" % (r_max, r_min))
    if n_rings < 1:
        raise ValueError("n_rings must be >= 1 (got %d)" % n_rings)
    _, _, R = _radial_grid(power_img, center, pixel)
    edges = np.geomspace(r_min, r_max, n_rings + 1)
    r_flat = R.ravel()
    p_flat = power_img.ravel()

    power_inside_rmin = float(p_flat[r_flat < r_min].sum())
    power_outside_rmax = float(p_flat[r_flat > r_max].sum())
    in_range = (r_flat >= r_min) & (r_flat <= r_max)
    ring_power = np.zeros(n_rings, dtype=np.float64)
    if np.any(in_range):
        # digitize: edges[i-1] <= r < edges[i] -> bin i; r == edges[-1]
        # (exactly r_max) digitizes one PAST the last bin, so clip it back
        # into ring n_rings-1 (the "last ring closed on the right" rule) --
        # harmless for everything else since in_range already excludes
        # r < r_min (-> bin 0-1 = -1, would clip to 0 but is masked out)
        # and r > r_max (masked out too).
        idx = np.clip(np.digitize(r_flat[in_range], edges) - 1, 0,
                      n_rings - 1)
        ring_power = np.bincount(idx, weights=p_flat[in_range],
                                 minlength=n_rings).astype(np.float64)
    return edges, ring_power, power_inside_rmin, power_outside_rmax


def ee_radius(radii, ee, frac):
    """(radii, ee) from encircled_energy/ensquared_energy + a target
    fraction (e.g. 0.5/0.8/0.9) -> the interpolated radius at which the
    cumulative curve first reaches `frac` (np.interp against the
    monotonically non-decreasing `ee` curve; clamps to the end radii
    outside [ee.min(), ee.max()], matching np.interp's default behavior)."""
    radii = np.asarray(radii, dtype=np.float64)
    ee = np.asarray(ee, dtype=np.float64)
    return float(np.interp(frac, ee, radii))


# =============================================================================
# Image simulation (imaging-analysis round): coherent / incoherent /
# partially-coherent imaging of a REAL intensity object through the
# system's coherent AMPLITUDE PSF h (complex, (H,W), CENTERED: kernel
# origin at index (H//2, W//2), i.e. where np.fft.fftshift puts DC --
# np.fft.ifftshift moves that pixel back to [0,0] for both parities of
# H/W). In the pipeline h comes from a coherent point/collimated run's
# saved detector field (post_process.render_image_sim); everything here
# is pure array-in/array-out Fourier optics on a shared grid:
#
#   coherent:   U = obj_amp (circ-conv) h,  I = |U|^2      (obj_amp = sqrt
#               of the intensity object -- amplitudes convolve, Goodman
#               ch. 6: a coherent system is linear in FIELD)
#   incoherent: I = obj (circ-conv) |h|^2 / sum(|h|^2)     (intensities
#               convolve; equivalently IFT(FT(obj) . OTF) with OTF the
#               autocorrelation of the pupil -- hence the classic factor
#               of 2: incoherent cutoff = 2x the coherent ATF cutoff)
#   partial:    Abbe source integration -- each illumination source point
#               s (a tilted plane wave) yields the coherent sub-image
#               |IFT(FT(obj_amp) . P(f + s))|^2 (the object spectrum
#               G(f - s) shifted into the pupil P equals, by the shift
#               theorem + the modulus, shifting P by +s); the observed
#               image is the s-integral over the effective source, here a
#               uniform disc of radius sigma * (pupil support radius) in
#               pupil-frequency pixels -- sigma is the standard partial-
#               coherence factor NA_cond/NA_obj. sigma=0 -> exactly the
#               coherent image; sigma >~ 2 -> effectively incoherent.
#
# All convolutions are CIRCULAR (plain FFT products, no zero padding) --
# correct for the periodic detector grid these fields live on and exactly
# what the oracle tests pin; callers with bright content at the frame
# edge should pad first. Grids must match: obj.shape == amp_psf.shape.
# =============================================================================
def coherent_transfer(amp_psf):
    """(H,W) complex CENTERED amplitude PSF h -> the centered coherent
    amplitude transfer function (the system pupil, up to scaling):
    P = fftshift(FFT2(ifftshift(h))). ifftshift moves the kernel origin
    (H//2, W//2) to [0,0] so P carries no origin-offset phase ramp, and
    the final fftshift centers DC at (H//2, W//2) again. Inverse of
    h = fftshift(IFFT2(ifftshift(P))) to machine precision."""
    h = np.asarray(amp_psf, dtype=np.complex128)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(h)))


def pupil_radius_px(P, threshold=0.05):
    """(H,W) CENTERED pupil/ATF -> its support radius in frequency pixels
    from the DC pixel (H//2, W//2): the largest pixel radius where
    |P| > threshold * max|P| (default 5% of peak -- exact for a hard-edged
    pupil, a sane support definition for an apodized/aberrated one).
    Returns 0.0 for an all-zero P."""
    P = np.asarray(P)
    mag = np.abs(P)
    peak = float(mag.max()) if mag.size else 0.0
    if peak <= 0:
        return 0.0
    H, W = mag.shape
    ys, xs = np.indices((H, W))
    r = np.hypot(ys - H // 2, xs - W // 2)
    return float(r[mag > threshold * peak].max())


def _convolve_centered(arr, kernel_centered):
    """(H,W) arr circularly convolved with a CENTERED (origin at
    (H//2, W//2)) kernel via plain FFT products; ifftshift moves the
    kernel origin to [0,0] so the output is not translated. Complex out."""
    A = np.fft.fft2(arr)
    K = np.fft.fft2(np.fft.ifftshift(np.asarray(kernel_centered)))
    return np.fft.ifft2(A * K)


def _check_image_shapes(obj, kern, what):
    if obj.shape != kern.shape:
        raise ValueError(
            "object %s and %s %s must share one grid (resample the "
            "object first)" % (obj.shape, what, kern.shape))


def image_coherent(obj, amp_psf):
    """Coherent image of a REAL intensity object through the centered
    complex amplitude PSF `amp_psf`: object amplitude = sqrt(clip(obj,0))
    (an intensity pattern's field is its square root, taken real/in-phase
    -- a self-luminous phase structure is not modeled), circularly
    convolved IN AMPLITUDE with h, returned as intensity |U|^2 (real,
    >= 0). No normalization is applied beyond h's own scale: a unit
    delta object returns exactly |h|^2."""
    obj = np.asarray(obj, dtype=np.float64)
    h = np.asarray(amp_psf, dtype=np.complex128)
    _check_image_shapes(obj, h, "amplitude PSF")
    obj_amp = np.sqrt(np.clip(obj, 0.0, None))
    U = _convolve_centered(obj_amp, h)
    return (np.abs(U) ** 2).astype(np.float64)


def image_incoherent(obj, psf):
    """Incoherent image of a REAL intensity object through the centered
    REAL intensity PSF `psf` (= |h|^2): intensities convolve. The PSF is
    normalized to unit sum first (a unit-power kernel: the image of a
    uniform object is that same uniform level, and a binary object can
    never overshoot 1 -- the physical contrast with image_coherent's
    edge ringing). Returns real, clipped >= 0 (FFT roundoff can leave
    ~1e-16-level negatives)."""
    obj = np.asarray(obj, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)
    _check_image_shapes(obj, psf, "PSF")
    total = float(psf.sum())
    if total <= 0:
        raise ValueError("intensity PSF sums to %g (need > 0)" % total)
    img = np.real(_convolve_centered(obj, psf / total))
    return np.clip(img, 0.0, None)


def image_partial(obj, amp_psf, sigma, n_src=150):
    """Partially-coherent image of a REAL intensity object via the Abbe
    source-integration method (see the block comment above): coherent
    sub-images |IFT(FT(obj_amp) . roll(P, s))|^2 are averaged (uniform
    weights) over illumination source points s on a filled disc of
    radius sigma * pupil_radius_px(P) in pupil-frequency PIXELS.

    Source sampling: every integer frequency-pixel offset inside the
    disc, thinned to a regular sublattice (smallest integer stride,
    always keeping the on-axis point s=0 and the lattice's symmetry)
    when the disc holds more than `n_src` points -- so the count is
    <= n_src (default 150) and grows to at most that as sigma grows.
    sigma=0 (or a disc smaller than one frequency pixel) returns
    image_coherent(obj, amp_psf) EXACTLY; large sigma approaches
    image_incoherent(obj, |amp_psf|^2). Pupil shifts use np.roll, so
    keep (1 + sigma) * pupil radius < min(H,W)/2 or the shifted pupil
    wraps around the frequency window (documented aliasing limit)."""
    obj = np.asarray(obj, dtype=np.float64)
    h = np.asarray(amp_psf, dtype=np.complex128)
    _check_image_shapes(obj, h, "amplitude PSF")
    if sigma < 0:
        raise ValueError("sigma must be >= 0 (got %g)" % sigma)
    if n_src < 1:
        raise ValueError("n_src must be >= 1 (got %d)" % n_src)
    P = coherent_transfer(h)
    r_src = float(sigma) * pupil_radius_px(P)
    if r_src < 0.5:
        # the source disc holds only the on-axis point: exactly coherent
        return image_coherent(obj, amp_psf)

    # integer source-point lattice inside the disc; thin by the smallest
    # stride whose sublattice count fits n_src (stride multiples keep the
    # on-axis point and the +/- symmetry)
    stride = 1
    while True:
        kmax = int(r_src // stride)
        ks = np.arange(-kmax, kmax + 1) * stride
        KY, KX = np.meshgrid(ks, ks, indexing="ij")
        keep = (KY ** 2 + KX ** 2) <= r_src ** 2 + 1e-9
        if int(keep.sum()) <= n_src:
            break
        stride += 1
    offsets = np.stack([KY[keep], KX[keep]], axis=1)

    obj_amp = np.sqrt(np.clip(obj, 0.0, None))
    G = np.fft.fft2(obj_amp)
    out = np.zeros(obj.shape, dtype=np.float64)
    for dy, dx in offsets:
        Ps = np.roll(P, (int(dy), int(dx)), axis=(0, 1))
        U = np.fft.ifft2(G * np.fft.ifftshift(Ps))
        out += np.abs(U) ** 2
    return out / len(offsets)
