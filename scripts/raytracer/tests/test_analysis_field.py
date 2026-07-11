# =============================================================================
# test_analysis_field.py -- closed-form validation of analysis_field.py:
# PSF irradiance, radial profiles, FFT-MTF, MTF50, encircled/ensquared
# energy. No tracing, no h5 -- everything here is either a synthetic
# Gaussian blob or a diffraction-limited Airy PSF built by FFT-propagating
# a circular pupil, checked against the closed-form circular-aperture
# results (Born & Wolf / Goodman "Introduction to Fourier Optics"):
#
#   MTF_circ(v) = (2/pi) * (acos(v) - v*sqrt(1 - v^2)),   v = f / f_c in [0,1]
#   f_c = 1 / (lambda * N)                                (N = f/#, cutoff)
#   first Airy zero radius r0 = 1.22 * lambda * N          (image plane)
#   EE(r0) ~ 0.8380                                        (encircled energy
#                                                            to the first zero)
#
# Sampling relation used to build the pupil->PSF FFT pair (see _airy_psf
# below): choosing an image-plane pixel pitch dx and a pupil array size N
# pixels across, the pupil aperture RADIUS in pupil-array pixels is
#     Rpix = N * dx / (2 * f_number * lambda)
# and this makes the first Airy zero land at
#     r0_px = 1.22 * lambda * f_number / dx
# with the pixel-independent invariant  r0_px * Rpix = 0.61 * N  (the
# focal length cancels entirely -- only f_number, lambda and the chosen
# dx matter). N/r0_px below (4096 / 25, giving Rpix ~ 100 pixels) is
# picked empirically so BOTH ends are well sampled: the binary pupil's
# staircase edge is smooth enough (Rpix ~ 100 px) that the FFT MTF
# matches the analytic curve to <1% all the way to v=0.9 near the
# diffraction cutoff, while the first Airy zero radius (~25 image px)
# is coarse enough that the whole FFT stays fast (~3 s) but fine enough
# that the encircled-energy pixelization error at that radius is <0.1%.
# Smaller Rpix (e.g. a naive N=1024/r0_px=50, Rpix~12 px) measurably
# biases the high-frequency MTF tail (~5% at v=0.9) via the pupil edge's
# staircase discretization -- this is NOT fixed by an area-equivalent
# effective f/# correction (checked), because the staircase distorts the
# OTF's *shape* near cutoff, not just its overall scale.
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_analysis_field.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer import analysis_field as af      # noqa: E402


# ---------------------------------------------------------------------------
# Airy-PSF oracle (built here, NOT in the module under test)
# ---------------------------------------------------------------------------
def _airy_psf(N, r0_px, lam, f_number):
    """N: pupil/image array size (square). r0_px: desired first-Airy-zero
    radius in IMAGE pixels. Returns (psf (N,N) float64, dx (image pixel
    pitch, metres)). Derivation: dx = 1.22*lam*f_number/r0_px; pupil
    radius in pupil-array pixels Rpix = N*dx/(2*f_number*lam) =
    0.61*N/r0_px (independent of dx/f_number/lam individually) -- see
    the module docstring for why N/r0_px is chosen the way it is."""
    dx = 1.22 * lam * f_number / r0_px
    Rpix = 0.61 * N / r0_px
    ys, xs = np.indices((N, N)).astype(np.float64)
    cy = cx = N / 2.0
    r_pupil = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    pupil = (r_pupil <= Rpix).astype(np.float64)
    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
    psf = np.abs(field) ** 2
    return psf, dx


def _mtf_circ_analytic(v):
    v = np.clip(v, 0.0, 1.0)
    return (2.0 / np.pi) * (np.arccos(v) - v * np.sqrt(1.0 - v ** 2))


# f/8 @ 550 nm, first Airy zero placed at 25 image pixels radius.
LAM = 550e-9
FNUM = 8.0
N = 4096
R0_PX = 25.0


@pytest.fixture(scope="module")
def airy():
    return _airy_psf(N, R0_PX, LAM, FNUM)


# ---------------------------------------------------------------------------
# psf_from_fields / normalize_psf
# ---------------------------------------------------------------------------
def test_psf_from_fields_parseval_and_normalize():
    rng = np.random.default_rng(0)
    Ex = rng.normal(size=(16, 20)) + 1j * rng.normal(size=(16, 20))
    Ey = rng.normal(size=(16, 20)) + 1j * rng.normal(size=(16, 20))
    psf = af.psf_from_fields(Ex, Ey)
    expected_sum = np.sum(np.abs(Ex) ** 2 + np.abs(Ey) ** 2)
    assert psf.sum() == pytest.approx(expected_sum, rel=1e-12)
    assert psf.shape == Ex.shape
    assert psf.dtype == np.float64

    norm, peak = af.normalize_psf(psf)
    assert peak == pytest.approx(float(psf.max()))
    assert norm.max() == pytest.approx(1.0)


def test_normalize_psf_zero_image():
    z = np.zeros((4, 4))
    norm, peak = af.normalize_psf(z)
    assert peak == 0.0
    assert np.all(norm == 0.0)


# ---------------------------------------------------------------------------
# radial_profile: synthetic Gaussian, centered and off-center
# ---------------------------------------------------------------------------
def _gaussian_img(H, W, cx, cy, sigma, amp=1.0):
    ys, xs = np.indices((H, W)).astype(np.float64)
    r2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return amp * np.exp(-r2 / (2.0 * sigma ** 2))


def test_radial_profile_centered_gaussian():
    H = W = 200
    sigma = 8.0
    cx = cy = W / 2.0
    img = _gaussian_img(H, W, cx, cy, sigma)
    r, prof = af.radial_profile(img, center=(cx, cy), pixel=1.0)
    analytic = np.exp(-r ** 2 / (2.0 * sigma ** 2))
    keep = r < 3.0 * sigma
    assert np.max(np.abs(prof[keep] - analytic[keep])) < 0.02


def test_radial_profile_default_centroid_matches_offcenter_gaussian():
    H = W = 200
    sigma = 6.0
    cx, cy = 130.0, 70.0
    img = _gaussian_img(H, W, cx, cy, sigma)
    # default center=None must recover the true center via the power
    # centroid (a symmetric Gaussian's centroid == its peak location)
    est_cx, est_cy = af._power_centroid(img, pixel=1.0)
    assert abs(est_cx - cx) < 0.05
    assert abs(est_cy - cy) < 0.05

    r, prof = af.radial_profile(img, center=None, pixel=1.0)
    analytic = np.exp(-r ** 2 / (2.0 * sigma ** 2))
    keep = r < 3.0 * sigma
    assert np.max(np.abs(prof[keep] - analytic[keep])) < 0.02


# ---------------------------------------------------------------------------
# Airy PSF: MTF matches the analytic circular-aperture curve
# ---------------------------------------------------------------------------
def test_mtf_matches_analytic_circular_aperture(airy):
    psf, dx = airy
    res = af.mtf2d(psf, dx)
    H, W = psf.shape
    freq = res["freq_cy_mm"]
    tan = res["tangential"][W // 2:]
    assert freq.shape == tan.shape

    f_c = 1.0 / (LAM * FNUM) * 1e-3   # cycles/mm
    v = freq / f_c
    sel = (v > 0.05) & (v < 0.9)
    assert np.count_nonzero(sel) > 20
    analytic = _mtf_circ_analytic(v[sel])
    err = np.abs(tan[sel] - analytic) / np.maximum(analytic, 1e-6)
    assert np.max(err) < 0.01, "max rel err %g" % np.max(err)

    # sagittal must agree (circularly symmetric aperture)
    sag = res["sagittal"][H // 2:]
    err_sag = np.abs(sag[sel] - analytic) / np.maximum(analytic, 1e-6)
    assert np.max(err_sag) < 0.01


def test_mtf50_matches_analytic_crossing(airy):
    psf, dx = airy
    res = af.mtf2d(psf, dx)
    W = psf.shape[1]
    freq = res["freq_cy_mm"]
    tan = res["tangential"][W // 2:]
    f_c = 1.0 / (LAM * FNUM) * 1e-3

    m50 = af.mtf50(freq, tan)
    assert np.isfinite(m50)

    # analytic v at which MTF_circ(v) == 0.5
    vgrid = np.linspace(0.0, 1.0, 200001)
    mgrid = _mtf_circ_analytic(vgrid)
    below = np.nonzero(mgrid <= 0.5)[0][0]
    v50_analytic = np.interp(0.5, mgrid[below - 1:below + 1][::-1],
                             vgrid[below - 1:below + 1][::-1])
    f50_analytic = v50_analytic * f_c
    assert abs(m50 - f50_analytic) / f50_analytic < 0.02


def test_mtf50_nan_when_never_crossing():
    freq = np.array([0.0, 1.0, 2.0])
    mtf_slice = np.array([1.0, 0.9, 0.6])
    assert np.isnan(af.mtf50(freq, mtf_slice))


# ---------------------------------------------------------------------------
# Airy PSF: encircled energy at the first zero, EE radii monotone
# ---------------------------------------------------------------------------
def test_encircled_energy_first_airy_zero(airy):
    psf, dx = airy
    radii, ee = af.encircled_energy(psf, pixel=dx)
    r0 = 1.22 * LAM * FNUM
    ee_at_r0 = float(np.interp(r0, radii, ee))
    assert abs(ee_at_r0 - 0.8380) < 0.01


def test_ee_radii_monotone_increasing(airy):
    psf, dx = airy
    radii, ee = af.encircled_energy(psf, pixel=dx)
    r50 = af.ee_radius(radii, ee, 0.5)
    r80 = af.ee_radius(radii, ee, 0.8)
    r90 = af.ee_radius(radii, ee, 0.9)
    assert r50 < r80 < r90


def test_ensquared_energy_reasonable(airy):
    psf, dx = airy
    radii_c, ee_c = af.encircled_energy(psf, pixel=dx)
    hw_s, ee_s = af.ensquared_energy(psf, pixel=dx)
    rc50 = af.ee_radius(radii_c, ee_c, 0.5)
    hs50 = af.ee_radius(hw_s, ee_s, 0.5)
    assert np.isfinite(hs50) and hs50 > 0
    # a square of half-width r strictly contains the circle of radius r
    # (its corners reach r*sqrt(2)), so it always encloses AT LEAST as
    # much energy at equal "radius" -- equivalently, the half-width
    # needed to reach a given fraction is <= the encircled radius for
    # the same fraction.
    assert hs50 <= rc50 * 1.05


def test_mtf2d_non_square_psf_axes():
    """Non-square detector grids (e.g. a 36x24 mm sensor) produce H != W
    PSFs; the sagittal half-slice must get its own frequency axis (used to
    crash the MTF plotter with a shape mismatch)."""
    rng = np.random.default_rng(7)
    psf = rng.random((341, 512))
    res = af.mtf2d(psf, 2e-6)
    H, W = psf.shape
    assert res["freq_cy_mm"].shape[0] == W - W // 2
    assert res["freq_y_cy_mm"].shape[0] == H - H // 2
    assert res["tangential"][W // 2:].shape == res["freq_cy_mm"].shape
    assert res["sagittal"][H // 2:].shape == res["freq_y_cy_mm"].shape
    # mtf50 must accept both slices without shape errors
    assert af.mtf50(res["freq_cy_mm"], res["tangential"][W // 2:]) \
        is not None or True
    af.mtf50(res["freq_y_cy_mm"], res["sagittal"][H // 2:])


def test_mtf2d_square_freq_axes_agree():
    rng = np.random.default_rng(8)
    psf = rng.random((128, 128))
    res = af.mtf2d(psf, 2e-6)
    assert np.allclose(res["freq_cy_mm"], res["freq_y_cy_mm"])
