# =============================================================================
# test_instrument_products.py -- samples-instruments round, two new post-
# stage products (both fast/synthetic, no tracing, no case directory needed
# except for the absorbance --reference-case tests, which build a MINIMAL
# one by hand, mirroring test_detector_instrument.py's _write_synthetic_h5
# fixture style):
#
#   A. diode_array instrument class (post_process.render_diode_array) --
#      a physical linear-array readout binned onto REAL pixel geometry,
#      reusing _pick_dispersion_axis/_axis_wavelength_fit (the SAME
#      centroid-fit machinery _render_spectrometer's lambda(x) plot uses)
#      to auto-detect the dispersion axis instead of assuming x.
#   B. --ring-profile log-annular readout (analysis_field.log_annular_power
#      + post_process.parse_ring_spec/render_ring_profile) and
#      --reference-case absorbance (post_process.render_absorbance).
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_instrument_products.py -v
# =============================================================================
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

from raytracer.analysis_field import log_annular_power  # noqa: E402
from raytracer.optprops import load_instruments          # noqa: E402
import post_process as pp                                 # noqa: E402

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0


# =============================================================================
# A. analysis_field.log_annular_power
# =============================================================================
def _thin_ring_image(n=101, r_lo=5.0, r_hi=6.0, value=1.0):
    """(n,n) image with `value` on every pixel whose INTEGER-grid radius
    (center = geometric center) falls in [r_lo, r_hi), else 0."""
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img = np.zeros((n, n))
    img[(r >= r_lo) & (r < r_hi)] = value
    return img, cx, cy


def test_log_annular_power_peaks_at_expected_ring():
    img, cx, cy = _thin_ring_image(r_lo=5.0, r_hi=6.0)
    edges, ring_power, inside, outside = log_annular_power(
        img, 1.0, (cx, cy), 8, 1.0, 20.0)
    # the illuminated annulus (r in [5,6)) must land entirely in ONE ring,
    # and that ring must bracket [5,6)
    nz = np.nonzero(ring_power)[0]
    assert len(nz) == 1
    i = nz[0]
    assert edges[i] <= 5.0 and edges[i + 1] >= 6.0
    assert ring_power[i] == pytest.approx(img.sum())
    assert inside == 0.0
    assert outside == 0.0


def test_log_annular_power_closure_exact():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64))
    edges, ring_power, inside, outside = log_annular_power(
        img, 0.7, None, 12, 0.3, 25.0)
    total = float(img.sum())
    residual = ring_power.sum() + inside + outside - total
    assert abs(residual) <= 1e-12 * max(abs(total), 1.0)


def test_log_annular_power_closure_exact_with_explicit_center():
    rng = np.random.default_rng(1)
    img = rng.random((50, 73))
    edges, ring_power, inside, outside = log_annular_power(
        img, 1.3, (12.0, 30.0), 20, 0.5, 40.0)
    total = float(img.sum())
    assert abs(ring_power.sum() + inside + outside - total) \
        <= 1e-12 * max(abs(total), 1.0)


def test_log_annular_power_log_spacing():
    img = np.ones((40, 40))
    edges, _, _, _ = log_annular_power(img, 1.0, None, 10, 0.5, 50.0)
    assert edges[0] == pytest.approx(0.5)
    assert edges[-1] == pytest.approx(50.0)
    ratios = edges[1:] / edges[:-1]
    assert np.allclose(ratios, ratios[0])           # constant ratio -> log
    assert ratios[0] == pytest.approx((50.0 / 0.5) ** (1.0 / 10))


def test_log_annular_power_bad_params_raise():
    img = np.ones((10, 10))
    with pytest.raises(ValueError, match="r_min"):
        log_annular_power(img, 1.0, None, 4, 0.0, 10.0)
    with pytest.raises(ValueError, match="r_max"):
        log_annular_power(img, 1.0, None, 4, 5.0, 5.0)
    with pytest.raises(ValueError, match="n_rings"):
        log_annular_power(img, 1.0, None, 0, 1.0, 10.0)


# =============================================================================
# B. parse_ring_spec
# =============================================================================
def test_parse_ring_spec_defaults_center_none():
    spec = pp.parse_ring_spec("n=32:rmin_mm=0.05:rmax_mm=10")
    assert spec == {"n": 32, "rmin_mm": 0.05, "rmax_mm": 10.0,
                    "center": None}


def test_parse_ring_spec_chief_is_none_too():
    spec = pp.parse_ring_spec("n=8:rmin_mm=1:rmax_mm=2:center=chief")
    assert spec["center"] is None


def test_parse_ring_spec_peak():
    spec = pp.parse_ring_spec("n=8:rmin_mm=1:rmax_mm=2:center=peak")
    assert spec["center"] == "peak"


def test_parse_ring_spec_xy_center():
    spec = pp.parse_ring_spec("n=8:rmin_mm=1:rmax_mm=2:center=1.5,-2.25")
    assert spec["center"] == (1.5, -2.25)


@pytest.mark.parametrize("bad", [
    "", "rmin_mm=1:rmax_mm=2", "n=0:rmin_mm=1:rmax_mm=2",
    "n=8:rmin_mm=0:rmax_mm=2", "n=8:rmin_mm=2:rmax_mm=2",
    "n=8:rmin_mm=1:rmax_mm=2:center=bogus", "n=abc:rmin_mm=1:rmax_mm=2",
    "n=8:rmin_mm=1:rmax_mm=2:frobnicate=1",
])
def test_parse_ring_spec_rejects_bad_specs(bad):
    with pytest.raises(ValueError):
        pp.parse_ring_spec(bad)


# =============================================================================
# render_ring_profile: 'X,Y' center matches the grid xhat/yhat/x_lo/y_lo
# convention, and a full end-to-end smoke test via a synthetic h5.
# =============================================================================
def _write_ring_h5(path, label, n=41, pixel_m=1e-4):
    """A single bright pixel off-center, all other pixels zero -- a clean
    target for both the 'peak' and 'X,Y' center-resolution paths."""
    bins = 1
    cube = np.zeros((bins, n, n))
    iy, ix = 25, 30
    cube[0, iy, ix] = 1e-6
    x_lo, y_lo = -n * pixel_m / 2, -n * pixel_m / 2
    with h5py.File(path, "w") as h:
        h["spectral_cube_mean"] = cube
        h["mask"] = np.ones((n, n), dtype=bool)
        h.attrs.update({
            "label": label, "H": n, "W": n, "pixel_m": pixel_m,
            "lam_lo_m": 500e-9, "lam_hi_m": 600e-9,
            "xhat": [0.0, 1.0, 0.0], "yhat": [0.0, 0.0, 1.0],
            "normal": [1.0, 0.0, 0.0], "x_lo": x_lo, "y_lo": y_lo,
            "seeds": 1,
        })
    return iy, ix, x_lo, y_lo


def test_render_ring_profile_peak_center_finds_bright_pixel(tmp_path):
    h5path = tmp_path / "det.h5"
    iy, ix, x_lo, y_lo = _write_ring_h5(h5path, "Det")
    adir = tmp_path / "analysis"
    ring_spec = {"n": 6, "rmin_mm": 0.01, "rmax_mm": 5.0, "center": "peak"}
    report = {"detectors": {"Det": {}}}
    pp.render_ring_profile(h5path, adir, ring_spec, report)
    block = report["detectors"]["Det"]["rings"]
    # all power is exactly AT the peak pixel (r=0 from it) -> the whole
    # image's power must show up as "inside_rmin" (r < rmin), none in any
    # ring and none outside
    assert block["power_inside_rmin_W"] == pytest.approx(1e-6)
    assert block["power_outside_rmax_W"] == pytest.approx(0.0)
    assert abs(block["closure_residual_W"]) <= 1e-12
    assert (adir / "rings_Det.csv").exists()
    assert (adir / "rings_Det.png").exists()


def test_render_ring_profile_xy_center_matches_grid_convention(tmp_path):
    h5path = tmp_path / "det.h5"
    iy, ix, x_lo, y_lo = _write_ring_h5(h5path, "Det")
    pixel_m = 1e-4
    # the bright pixel's own 'mm in the detector grid frame' coordinate
    # (pos@xhat*1e3 convention): x_lo + ix*pixel_m, in mm
    x_mm = (x_lo + ix * pixel_m) * 1e3
    y_mm = (y_lo + iy * pixel_m) * 1e3
    adir = tmp_path / "analysis"
    ring_spec = {"n": 6, "rmin_mm": 0.001, "rmax_mm": 5.0,
                "center": (x_mm, y_mm)}
    report = {"detectors": {"Det": {}}}
    pp.render_ring_profile(h5path, adir, ring_spec, report)
    block = report["detectors"]["Det"]["rings"]
    # centering exactly on the bright pixel -> same all-inside-rmin result
    # as the 'peak' test above
    assert block["power_inside_rmin_W"] == pytest.approx(1e-6, rel=1e-6)
    assert block["power_outside_rmax_W"] == pytest.approx(0.0)


# =============================================================================
# diode_array instrument class
# =============================================================================
def _dispersed_line_cube(lam0_nm=600.0, bins=40, H=10, W=200,
                        pixel_m=20e-6, line_width_nm=2.0,
                        col_width_nm=3.0, peak_power_w=2e-15):
    """Synthetic detector-plane cube: a spectrograph's dispersed image of a
    single spectral line at lam0_nm. Dispersion is linear in x (500nm at
    x=0 to 700nm at x=W*pixel_m); the illuminated band is rows 3:7 and a
    narrow (col_width_nm sigma, IN WAVELENGTH UNITS mapped through the
    linear dispersion) Gaussian spot in x centered wherever lambda(x) ==
    lam0_nm -- wide enough to span several simulation columns (so the
    dispersion-axis centroid fit has >= 2 valid columns), narrow enough to
    give a clean, well-localized peak pixel."""
    lam_lo, lam_hi = 500e-9, 700e-9
    lam_c_nm = (lam_lo + (np.arange(bins) + 0.5)
               * (lam_hi - lam_lo) / bins) / 1e-9
    x_centers_mm = (np.arange(W) + 0.5) * pixel_m / 1e-3
    total_span_mm = W * pixel_m / 1e-3
    slope_true = (700.0 - 500.0) / total_span_mm
    lam_of_x = 500.0 + slope_true * x_centers_mm

    cube = np.zeros((bins, H, W))
    for j in range(W):
        weights = np.exp(-0.5 * ((lam_c_nm - lam_of_x[j])
                                 / line_width_nm) ** 2)
        s = weights.sum()
        if s > 0:
            weights = weights / s
        col_power = np.exp(-0.5 * ((lam_of_x[j] - lam0_nm)
                                   / col_width_nm) ** 2) * peak_power_w
        cube[:, 3:7, j] = weights[:, None] * col_power / 4.0
    return cube, lam_lo, lam_hi, lam0_nm, slope_true


def _diode_entry(**overrides):
    entry = dict(pixel_pitch_um=100.0, pixel_height_um=200.0, n_px=16,
                lam_um=np.array([0.5, 0.6, 0.7]), qe=np.array([0.5, 0.5, 0.5]),
                full_well_e=1e12, read_noise_e=0.0, bit_depth=16,
                adc_gain_e_per_dn=1.0, integration_time_s_default=1.0,
                stray_light_floor=0.0)
    entry.update(overrides)
    return entry


def test_diode_array_peaks_at_expected_pixel():
    cube, lam_lo, lam_hi, lam0_nm, slope_true = _dispersed_line_cube()
    pixel_m = 20e-6
    entry = _diode_entry()
    report = {"detectors": {"Det": {"instrument": {}}}}
    lam_px, counts = pp.render_diode_array(
        cube, lam_lo, lam_hi, pixel_m, entry, "ideal", None, None,
        "safe", "Det", report)
    assert report["detectors"]["Det"]["instrument"]["dispersion_axis"] == "x"
    peak_px = int(np.argmax(counts))
    # within about one array-pixel's own wavelength span of the true line
    pitch_mm = entry["pixel_pitch_um"] * 1e-3
    tol_nm = abs(slope_true) * pitch_mm * 1.5
    assert abs(lam_px[peak_px] - lam0_nm) <= tol_nm
    assert counts[peak_px] > 0


def test_diode_array_qe_weighting_flat_vs_stepped_differ():
    # odd n_px + a wider spectral line (relative to one array pixel's own
    # ~5 nm/px span) so the illuminated band lands as ONE clean, non-tied
    # peak pixel with well-populated neighbors on both sides -- an EVEN
    # n_px centers the array on a pixel BOUNDARY (the two middle pixels
    # tie), which makes "the peak" ambiguous for this symmetry check.
    cube, lam_lo, lam_hi, lam0_nm, _ = _dispersed_line_cube(
        peak_power_w=2e-17, col_width_nm=8.0)
    pixel_m = 20e-6

    def run(qe_vals):
        entry = _diode_entry(qe=np.array(qe_vals), full_well_e=1e12,
                             adc_gain_e_per_dn=1.0, n_px=15)
        report = {"detectors": {"Det": {"instrument": {}}}}
        return pp.render_diode_array(cube, lam_lo, lam_hi, pixel_m, entry,
                                     "ideal", None, None, "safe", "Det",
                                     report)

    lam_px, flat_counts = run([0.5, 0.5, 0.5])       # flat QE
    _, step_counts = run([0.2, 0.5, 0.8])            # QE increases with lambda
    peak = int(np.argmax(flat_counts))
    lo, hi = peak - 1, peak + 1
    assert flat_counts[lo] > 0 and flat_counts[hi] > 0
    # flat QE: the (symmetric-in-wavelength) line reads roughly symmetric
    # counts either side of the peak (not exact -- the Gaussian is sampled
    # onto a discrete pixel grid, so some skew from the peak's exact
    # sub-pixel offset is expected)
    flat_ratio = flat_counts[lo] / flat_counts[hi]
    assert flat_ratio == pytest.approx(1.0, abs=0.15)
    # stepped (increasing) QE must skew the SAME two side-pixels toward the
    # higher-wavelength (higher-QE) side -- i.e. the ratio moves down, away
    # from the flat-QE ratio, by more than sampling noise alone
    step_ratio = step_counts[lo] / step_counts[hi]
    assert step_ratio < flat_ratio - 0.02


def test_diode_array_ideal_mode_is_deterministic_no_seed():
    cube, lam_lo, lam_hi, *_ = _dispersed_line_cube()
    pixel_m = 20e-6
    entry = _diode_entry()
    r1 = {"detectors": {"Det": {"instrument": {}}}}
    r2 = {"detectors": {"Det": {"instrument": {}}}}
    _, c1 = pp.render_diode_array(cube, lam_lo, lam_hi, pixel_m, entry,
                                  "ideal", None, None, "safe", "Det", r1)
    _, c2 = pp.render_diode_array(cube, lam_lo, lam_hi, pixel_m, entry,
                                  "ideal", None, None, "safe", "Det", r2)
    assert np.array_equal(c1, c2)
    assert "seed" not in r1["detectors"]["Det"]["instrument"]
    assert r1["detectors"]["Det"]["instrument"] \
        == r2["detectors"]["Det"]["instrument"]


def test_diode_array_full_mode_seed_reproducible():
    cube, lam_lo, lam_hi, *_ = _dispersed_line_cube(peak_power_w=5e-16)
    pixel_m = 20e-6
    entry = _diode_entry(read_noise_e=5.0)
    seed = 4242
    r1 = {"detectors": {"Det": {"instrument": {}}}}
    r2 = {"detectors": {"Det": {"instrument": {}}}}
    _, c1 = pp.render_diode_array(cube, lam_lo, lam_hi, pixel_m, entry,
                                  "full", seed, None, "safe", "Det", r1)
    _, c2 = pp.render_diode_array(cube, lam_lo, lam_hi, pixel_m, entry,
                                  "full", seed, None, "safe", "Det", r2)
    assert np.array_equal(c1, c2)
    assert r1["detectors"]["Det"]["instrument"]["seed"] == seed
    assert r1["detectors"]["Det"]["instrument"] \
        == r2["detectors"]["Det"]["instrument"]


def test_diode_array_ideal_counts_invert_to_power_within_quantization():
    """Single-array-pixel bench-comparison gate, mirroring
    test_camera_ideal_counts_invert_to_power_within_quantization exactly,
    but through render_diode_array's own binning path (n_px=1, W=H=1)."""
    lam = 550e-9
    QE = 0.5
    t_int = 1.0
    electrons_target = 1.0e5
    P_true = electrons_target * H_PLANCK * C_LIGHT / (QE * lam * t_int)
    cube = np.array([[[P_true]]])
    lam_lo, lam_hi = lam - 1e-9, lam + 1e-9
    pixel_m = 10e-6
    entry = dict(pixel_pitch_um=10.0, pixel_height_um=10.0, n_px=1,
                lam_um=np.array([0.4, 0.7]), qe=np.array([0.5, 0.5]),
                full_well_e=1e12, read_noise_e=0.0, bit_depth=24,
                adc_gain_e_per_dn=50.0, integration_time_s_default=t_int,
                stray_light_floor=0.0)
    report = {"detectors": {"Det": {"instrument": {}}}}
    _, counts = pp.render_diode_array(cube, lam_lo, lam_hi, pixel_m, entry,
                                      "ideal", None, None, "safe", "Det",
                                      report)
    gain = entry["adc_gain_e_per_dn"]
    electrons_recovered = float(counts[0]) * gain
    P_recovered = electrons_recovered * H_PLANCK * C_LIGHT / (QE * lam * t_int)
    tol_W = 0.5 * gain * H_PLANCK * C_LIGHT / (QE * lam * t_int)
    assert abs(P_recovered - P_true) <= tol_W * 1.0001
    assert report["detectors"]["Det"]["instrument"]["saturation_fraction"] \
        == 0.0


def test_diode_array_saturates_at_full_well_and_quantizes_exactly():
    lam = 550e-9
    cube = np.array([[[1.0]]])            # deliberately huge, 1 W
    lam_lo, lam_hi = lam - 1e-9, lam + 1e-9
    entry = dict(pixel_pitch_um=10.0, pixel_height_um=10.0, n_px=1,
                lam_um=np.array([0.4, 0.7]), qe=np.array([0.5, 0.5]),
                full_well_e=1000.0, read_noise_e=0.0, bit_depth=16,
                adc_gain_e_per_dn=1.0, integration_time_s_default=1.0,
                stray_light_floor=0.0)
    pixel_m = 10e-6
    report = {"detectors": {"Det": {"instrument": {}}}}
    _, counts = pp.render_diode_array(cube, lam_lo, lam_hi, pixel_m, entry,
                                      "ideal", None, None, "safe", "Det",
                                      report)
    block = report["detectors"]["Det"]["instrument"]
    assert block["saturated"] is True
    assert block["saturation_fraction"] == pytest.approx(1.0)
    assert float(counts[0]) == pytest.approx(1000.0)   # DN == full_well/gain


# =============================================================================
# tcd1304_array registry row
# =============================================================================
def test_tcd1304_array_registry_row_loads():
    reg = load_instruments()
    assert "tcd1304_array" in reg
    row = reg["tcd1304_array"]
    assert row["class"] == "diode_array"
    assert row["pixel_pitch_um"] == pytest.approx(8.0)
    assert row["pixel_height_um"] == pytest.approx(200.0)
    assert row["n_px"] == 3648
    assert row["full_well_e"] == pytest.approx(100000.0)
    assert row["read_noise_e"] == pytest.approx(25.0)
    assert row["bit_depth"] == 12
    assert row["stray_light_floor"] == pytest.approx(5e-4)
    assert row["reference"]
    # qe table loaded, in (0, 1]
    assert np.all(row["qe"] > 0) and np.all(row["qe"] <= 1)
    assert row["lam_um"].min() < 0.45 and row["lam_um"].max() > 1.0


# =============================================================================
# --reference-case absorbance (post_process.render_absorbance)
# =============================================================================
def _write_case_h5(case_dir, label, cube, lam_lo, lam_hi, H=1, W=1,
                   pixel_m=5e-5):
    (case_dir / "detectors").mkdir(parents=True, exist_ok=True)
    with h5py.File(case_dir / "detectors" / (label + ".h5"), "w") as h:
        h["spectral_cube_mean"] = cube
        h["mask"] = np.ones((H, W), dtype=bool)
        h.attrs.update({
            "label": label, "H": H, "W": W, "pixel_m": pixel_m,
            "lam_lo_m": lam_lo, "lam_hi_m": lam_hi,
            "xhat": [0.0, 1.0, 0.0], "yhat": [0.0, 0.0, 1.0],
            "normal": [1.0, 0.0, 0.0], "x_lo": 0.0, "y_lo": 0.0,
            "seeds": 1,
        })


def _spectrometer_entry(**overrides):
    entry = dict(lam_lo_nm=400.0, lam_hi_nm=800.0, resolution_fwhm_nm=60.0,
                slit_um=25.0, stray_light_floor=1e-3,
                lam_um=np.array([0.4, 0.5, 0.6, 0.7, 0.8]),
                qe=np.array([0.3, 0.4, 0.4, 0.35, 0.2]))
    entry.update(overrides)
    return entry


def _build_absorbance_fixture(tmp_path, c_transmit, resolution_fwhm_nm=60.0):
    """A reference ('blank') case + a report/instrument_products pair
    standing in for the current ('sample') case, related by a known
    constant transmittance c_transmit: sample_cube = c_transmit * ref_cube
    everywhere -> A(lambda) must come back EXACTLY -log10(c_transmit)
    (QE/resolution/floor are identical for both, all cancel in the ratio).
    """
    bins = 5
    lam_lo, lam_hi = 400e-9, 800e-9
    pw_ref = np.array([0.1, 0.3, 1.0, 0.4, 0.2]) * 1e-9
    cube_ref = pw_ref.reshape(bins, 1, 1)
    cube_sample = c_transmit * cube_ref

    ref_dir = tmp_path / "ref"
    _write_case_h5(ref_dir, "Det", cube_ref, lam_lo, lam_hi)
    with open(ref_dir / "case.json", "w") as fh:
        json.dump({"seed": 0, "status": "completed"}, fh)

    entry = _spectrometer_entry(resolution_fwhm_nm=resolution_fwhm_nm)
    report = {"detectors": {"Det": {"instrument": {}, "resolution": [1, 1]}}}
    x_arr, y_arr = pp.render_instrument_spectrometer(
        cube_sample, lam_lo, lam_hi, entry, "ideal", None, None, "Det",
        "Det", report)
    instrument_products = {"Det": ("row_test", "ideal", "spectrometer",
                                   x_arr, y_arr)}
    instrument_registry = {"row_test": entry}
    return ref_dir, instrument_products, instrument_registry, report


def test_absorbance_exact_constant_transmittance(tmp_path):
    c = 0.37
    ref_dir, products, registry, report = _build_absorbance_fixture(
        tmp_path, c)
    outdir = tmp_path / "instrument"
    pp.render_absorbance(ref_dir, products, registry, outdir, report)
    block = report["detectors"]["Det"]["absorbance"]
    assert block["n_masked_px"] == 0
    assert block["peak_A"] == pytest.approx(-np.log10(c), abs=1e-9)

    with open(outdir / "absorbance_Det.csv") as fh:
        rows = list(csv.reader(fh))
    A_vals = [float(r[3]) for r in rows[1:]]
    assert np.allclose(A_vals, -np.log10(c), atol=1e-8)


def test_absorbance_masks_below_stray_floor(tmp_path):
    # a SHARP instrument resolution leaves most of the continuous output
    # grid far from any of the 5 discrete input bins at/near the floor --
    # those samples must be masked (I0 <= stray_light_floor * peak)
    c = 0.5
    ref_dir, products, registry, report = _build_absorbance_fixture(
        tmp_path, c, resolution_fwhm_nm=2.0)
    outdir = tmp_path / "instrument"
    pp.render_absorbance(ref_dir, products, registry, outdir, report)
    block = report["detectors"]["Det"]["absorbance"]
    assert block["n_masked_px"] > 0
    # wherever it WASN'T masked, the ratio still comes back exact
    assert block["peak_A"] == pytest.approx(-np.log10(c), abs=1e-9)


def test_absorbance_hard_errors_on_row_mismatch(tmp_path):
    ref_dir, products, registry, report = _build_absorbance_fixture(
        tmp_path, 0.5)
    with open(ref_dir / "report.json", "w") as fh:
        json.dump({"detectors": {"Det": {"instrument": {"row":
                                                        "other_row"}}}}, fh)
    with pytest.raises(SystemExit, match="instrument row"):
        pp.render_absorbance(ref_dir, products, registry,
                             tmp_path / "instrument", report)


def test_absorbance_hard_errors_on_grid_mismatch(tmp_path):
    ref_dir, products, registry, report = _build_absorbance_fixture(
        tmp_path, 0.5)
    report["detectors"]["Det"]["resolution"] = [2, 2]   # sample says 2x2
    with pytest.raises(SystemExit, match="pixel grid"):
        pp.render_absorbance(ref_dir, products, registry,
                             tmp_path / "instrument", report)


def test_absorbance_missing_reference_detector_is_silent_skip(tmp_path, capsys):
    ref_dir, products, registry, report = _build_absorbance_fixture(
        tmp_path, 0.5)
    products["OtherDet"] = products.pop("Det")  # no OtherDet.h5 in ref_dir
    report["detectors"]["OtherDet"] = report["detectors"].pop("Det")
    pp.render_absorbance(ref_dir, products, registry,
                        tmp_path / "instrument", report)
    assert "absorbance" not in report["detectors"]["OtherDet"]
    assert "no detector" in capsys.readouterr().out
