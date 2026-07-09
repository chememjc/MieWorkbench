# =============================================================================
# test_detector_photometric.py -- --photometric (lux) and --spectrometer
# (lambda(x) dispersion) post-processing features. Zero tracer changes:
# both operate purely on synthetic spectral_cube_mean-shaped arrays, the
# same (bins, H, W) power cube [W] run_trace.py always saves.
#
# Run: /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_detector_photometric.py -v
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))    # for scripts/common.py via name

from raytracer.detector import spectral_cube_to_lux            # noqa: E402
import post_process as pp                                      # noqa: E402


# ---------------------------------------------------------------------------
# spectral_cube_to_lux
# ---------------------------------------------------------------------------
def _mono_cube(lam_center_nm, half_width_nm=5.0, H=4, W=4, total_W=1.0):
    """(1, H, W) power cube whose single spectral bin is centered EXACTLY
    on lam_center_nm (bin edges lam_center +- half_width_nm), power spread
    evenly over the H*W pixels so it sums to total_W."""
    lam_lo = (lam_center_nm - half_width_nm) * 1e-9
    lam_hi = (lam_center_nm + half_width_nm) * 1e-9
    cube = np.full((1, H, W), total_W / (H * W))
    return cube, lam_lo, lam_hi


def test_monochromatic_555nm_luminous_flux():
    # V(555nm) == 1.0 exactly in the tabulated CIE y-bar (555 lands on the
    # 5 nm grid post-processing's cie_xyz_weights interpolates over).
    cube, lam_lo, lam_hi = _mono_cube(555.0, total_W=1.0)
    _, flux_lm = spectral_cube_to_lux(cube, lam_lo, lam_hi, pixel_area_m2=1e-6)
    assert flux_lm == pytest.approx(683.002, rel=0.005)


@pytest.mark.parametrize("lam_nm", [300.0, 900.0])
def test_outside_visible_range_zero_lumens(lam_nm):
    cube, lam_lo, lam_hi = _mono_cube(lam_nm, total_W=1.0)
    lux_map, flux_lm = spectral_cube_to_lux(cube, lam_lo, lam_hi,
                                            pixel_area_m2=1e-6)
    assert flux_lm == 0.0
    assert np.all(lux_map == 0.0)


def test_lux_scales_inversely_with_pixel_area():
    cube, lam_lo, lam_hi = _mono_cube(555.0, total_W=1.0)
    lux_small, flux_small = spectral_cube_to_lux(cube, lam_lo, lam_hi,
                                                  pixel_area_m2=1e-6)
    lux_big, flux_big = spectral_cube_to_lux(cube, lam_lo, lam_hi,
                                             pixel_area_m2=4e-6)
    # luminous flux (integrated over pixels) is area-independent...
    assert flux_small == pytest.approx(flux_big, rel=1e-9)
    # ...but the per-pixel illuminance map scales as 1/area
    assert lux_small == pytest.approx(4.0 * lux_big, rel=1e-9)


# ---------------------------------------------------------------------------
# --spectrometer centroid/fit math (spectral_centroid, lambda_centroid_map,
# linear_fit_r2)
# ---------------------------------------------------------------------------
def _dispersion_cube(bins=64, H=5, W=20, lam_lo_nm=400.0, lam_hi_nm=700.0,
                     lam0_nm=450.0, slope_nm_per_col=10.0, power=1.0):
    """(bins, H, W) power cube where every column x has ALL of its power
    split between the two spectral bins straddling
    lam(x) = lam0_nm + slope_nm_per_col*x, with weights chosen so the
    power-weighted centroid of that column recovers lam(x) to floating-
    point precision (independent of bin resolution) -- an exact synthetic
    target for the linear dispersion fit, not a quantization-limited one.
    Power is split evenly across the H rows so the 2D centroid map and the
    y-collapsed column profile agree exactly."""
    edges_nm = np.linspace(lam_lo_nm, lam_hi_nm, bins + 1)
    centers_nm = 0.5 * (edges_nm[:-1] + edges_nm[1:])
    cube = np.zeros((bins, H, W))
    lam_of_x = lam0_nm + slope_nm_per_col * np.arange(W)
    for x in range(W):
        target = lam_of_x[x]
        i = int(np.clip(np.searchsorted(centers_nm, target) - 1,
                        0, bins - 2))
        lam_i, lam_ip1 = centers_nm[i], centers_nm[i + 1]
        f = (target - lam_i) / (lam_ip1 - lam_i)
        cube[i, :, x] = power * (1.0 - f) / H
        cube[i + 1, :, x] = power * f / H
    return cube, lam_lo_nm * 1e-9, lam_hi_nm * 1e-9, lam_of_x


def test_spectral_centroid_recovers_exact_wavelength():
    cube, lam_lo, lam_hi, lam_of_x = _dispersion_cube(W=6)
    total, lam_bar = pp.spectral_centroid(cube.sum(axis=1), lam_lo, lam_hi)
    assert np.allclose(lam_bar, lam_of_x, atol=1e-6)
    assert np.allclose(total, 1.0)


def test_lambda_centroid_map_masks_dark_pixels():
    cube, lam_lo, lam_hi, lam_of_x = _dispersion_cube(H=3, W=4)
    # zero out one full column -> should be masked invalid, NaN in the map
    cube[:, :, 2] = 0.0
    lam_map, valid = pp.lambda_centroid_map(cube, lam_lo, lam_hi)
    assert not np.any(valid[:, 2])
    assert np.all(np.isnan(lam_map[:, 2]))
    assert np.all(valid[:, 0])
    assert np.allclose(lam_map[:, 0], lam_of_x[0], atol=1e-6)


def test_dispersion_fit_within_1pct_and_r2():
    pixel_m = 1e-3     # 1 mm/px -> dispersion_nm_per_mm == slope_nm_per_col
    slope_nm_per_col = 10.0
    cube, lam_lo, lam_hi, lam_of_x = _dispersion_cube(
        slope_nm_per_col=slope_nm_per_col)
    xmm = (np.arange(cube.shape[2]) + 0.5) * pixel_m / 1e-3
    col_total, col_lam = pp.spectral_centroid(cube.sum(axis=1), lam_lo, lam_hi)
    slope, intercept, r2 = pp.linear_fit_r2(xmm, col_lam)
    assert slope == pytest.approx(slope_nm_per_col, rel=0.01)
    assert r2 > 0.99


def test_linear_fit_r2_guards_fewer_than_two_points():
    assert pp.linear_fit_r2([1.0], [2.0]) == (None, None, None)
    assert pp.linear_fit_r2([], []) == (None, None, None)


def test_linear_fit_r2_perfect_line():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    slope, intercept, r2 = pp.linear_fit_r2(x, y)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)
