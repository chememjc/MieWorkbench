# =============================================================================
# test_detector_instrument.py -- virtual instrument layer (engine3.md P2.5
# §9): detector.spectral_cube_to_electrons is the pure camera-response
# primitive (mirrors spectral_cube_to_photocurrent); post_process.py's
# render_instrument_* functions add saturation/quantization (both modes)
# and the stochastic noise chain (full mode only). No tracer or case
# directory needed -- synthetic cubes/entries throughout, plus one
# synthetic-h5 end-to-end check of render_detector's wiring.
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_detector_instrument.py -v
# =============================================================================
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import stats

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

from raytracer.detector import spectral_cube_to_electrons  # noqa: E402
import post_process as pp                                   # noqa: E402

# CODATA exact SI (must match detector.py)
Q = 1.602176634e-19
H = 6.62607015e-34
C = 299792458.0


# ---------------------------------------------------------------------------
# spectral_cube_to_electrons -- pure response primitive
# ---------------------------------------------------------------------------
def test_electrons_flat_qe_hand_computed():
    lam = 550e-9
    P = 2.0e-7                                # W
    cube = np.array([[[P]]])                  # (1 bin, 1x1 pixel)
    lam_lo, lam_hi = lam - 1e-9, lam + 1e-9
    qe_lam_um = np.array([0.4, 0.7])
    qe_vals = np.array([0.5, 0.5])
    t_int = 2.0
    e_img = spectral_cube_to_electrons(cube, lam_lo, lam_hi, qe_lam_um,
                                       qe_vals, t_int, pixel_area_ratio=1.0)
    expect = 0.5 * P * t_int * lam / (H * C)
    assert e_img.shape == (1, 1)
    assert float(e_img[0, 0]) == pytest.approx(expect, rel=1e-9)


def test_electrons_pixel_area_ratio_scales_linearly():
    lam = 550e-9
    cube = np.array([[[1.0]]])
    lam_lo, lam_hi = lam - 1e-9, lam + 1e-9
    qe_lam_um, qe_vals = np.array([0.4, 0.7]), np.array([0.6, 0.6])
    base = spectral_cube_to_electrons(cube, lam_lo, lam_hi, qe_lam_um,
                                      qe_vals, 1.0, pixel_area_ratio=1.0)
    scaled = spectral_cube_to_electrons(cube, lam_lo, lam_hi, qe_lam_um,
                                        qe_vals, 1.0, pixel_area_ratio=0.25)
    assert float(scaled[0, 0]) == pytest.approx(0.25 * float(base[0, 0]))


def test_electrons_outside_qe_table_contributes_zero():
    bins = 2
    lam_lo, lam_hi = 500e-9, 1300e-9      # bin centers 700nm, 1100nm
    cube = np.zeros((bins, 1, 1))
    cube[0, 0, 0] = 1.0
    cube[1, 0, 0] = 1.0
    qe_lam_um, qe_vals = np.array([0.60, 0.90]), np.array([0.5, 0.5])
    e_img = spectral_cube_to_electrons(cube, lam_lo, lam_hi, qe_lam_um,
                                       qe_vals, 1.0)
    # only the 700 nm bin (inside [600,900]nm) contributes
    expect = 0.5 * 1.0 * 1.0 * 700e-9 / (H * C)
    assert float(e_img[0, 0]) == pytest.approx(expect, rel=1e-6)


def test_electrons_spectrum_and_cube_agree():
    qe_lam_um, qe_vals = np.array([0.40, 0.80]), np.array([0.7, 0.9])
    spectrum = np.array([0.2, 0.5, 0.3])
    cube = np.zeros((3, 2, 2))
    for b in range(3):
        cube[b] = spectrum[b] / 4.0
    args = (400e-9, 800e-9, qe_lam_um, qe_vals, 1.0)
    total_cube = spectral_cube_to_electrons(cube, *args).sum()
    total_spec = spectral_cube_to_electrons(spectrum, *args)
    assert float(total_cube) == pytest.approx(float(total_spec), rel=1e-9)


# ---------------------------------------------------------------------------
# label -> owning body ownership (mirrors detector_qe_curve_for_label)
# ---------------------------------------------------------------------------
def test_instrument_ownership_prefix_match():
    bodies = {"Cam": "camera_generic:ideal"}
    assert pp.detector_instrument_for_label(
        "Cam.Pad.Face3", bodies) == "camera_generic:ideal"
    assert pp.detector_instrument_for_label("Cam", bodies) \
        == "camera_generic:ideal"
    assert pp.detector_instrument_for_label("Other.X.Face1", bodies) is None


def test_instrument_ownership_longest_prefix_wins():
    bodies = {"Det": "powermeter_generic", "Detector": "camera_generic"}
    assert pp.detector_instrument_for_label(
        "Detector.Pad.Face1", bodies) == "camera_generic"
    assert pp.detector_instrument_for_label(
        "Det.Pad.Face1", bodies) == "powermeter_generic"


# ---------------------------------------------------------------------------
# parse_instrument_spec
# ---------------------------------------------------------------------------
def test_parse_instrument_spec_default_mode_is_full():
    assert pp.parse_instrument_spec("camera_generic") \
        == ("camera_generic", "full")


def test_parse_instrument_spec_explicit_mode():
    assert pp.parse_instrument_spec("camera_generic:ideal") \
        == ("camera_generic", "ideal")
    assert pp.parse_instrument_spec("camera_generic:FULL") \
        == ("camera_generic", "full")


def test_parse_instrument_spec_bad_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        pp.parse_instrument_spec("camera_generic:fast")


def test_parse_instrument_spec_empty_rejected():
    with pytest.raises(ValueError):
        pp.parse_instrument_spec("")
    with pytest.raises(ValueError):
        pp.parse_instrument_spec(":ideal")


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------
def test_instrument_seed_deterministic_and_distinct():
    s1 = pp._instrument_seed(7, "camera_generic", "Det.Pad.Face1")
    s2 = pp._instrument_seed(7, "camera_generic", "Det.Pad.Face1")
    assert s1 == s2
    assert s1 == pp._instrument_seed(7, "camera_generic", "Det.Pad.Face1")
    # any of the three inputs changing changes the seed
    assert s1 != pp._instrument_seed(8, "camera_generic", "Det.Pad.Face1")
    assert s1 != pp._instrument_seed(7, "powermeter_generic",
                                     "Det.Pad.Face1")
    assert s1 != pp._instrument_seed(7, "camera_generic", "Other.Face1")
    assert isinstance(s1, int) and 0 <= s1 < 2 ** 63


# ---------------------------------------------------------------------------
# camera: ideal-mode counts -> W inversion (the P2.5 bench-comparison gate)
# ---------------------------------------------------------------------------
def _camera_entry(**overrides):
    entry = dict(
        pixel_pitch_um=10.0, integration_time_s_default=1.0,
        lam_um=np.array([0.4, 0.7]), qe=np.array([0.5, 0.5]),
        full_well_e=1e12, read_noise_e=0.0, dark_current_e_per_s=0.0,
        bit_depth=24, adc_gain_e_per_dn=50.0,
    )
    entry.update(overrides)
    return entry


def test_camera_ideal_counts_invert_to_power_within_quantization():
    lam = 550e-9
    QE = 0.5
    entry = _camera_entry()
    t_int = entry["integration_time_s_default"]
    # pick a single-pixel power that lands well inside both the full-well
    # and bit-depth ranges (a real camera pixel sees ~1e4-1e5 e-, not the
    # ~1e11 e-/s a bare milliwatt would dump on one pixel)
    electrons_target = 1.0e5
    P_true = electrons_target * H * C / (QE * lam * t_int)
    cube = np.array([[[P_true]]])
    lam_lo, lam_hi = lam - 1e-9, lam + 1e-9
    pixel_area = (entry["pixel_pitch_um"] * 1e-6) ** 2   # ratio == 1
    report = {"detectors": {"Det": {"instrument": {}}}}
    pp.render_instrument_camera(cube, lam_lo, lam_hi, pixel_area, entry,
                                "ideal", None, None, "safe", "Det", report)
    block = report["detectors"]["Det"]["instrument"]
    assert "seed" not in block            # ideal mode draws no rng
    counts = block["mean_counts"]         # single pixel: mean == the value

    QE = 0.5
    gain = entry["adc_gain_e_per_dn"]
    t_int = entry["integration_time_s_default"]
    electrons_recovered = counts * gain
    P_recovered = electrons_recovered * H * C / (QE * lam * t_int)
    tol_W = 0.5 * gain * H * C / (QE * lam * t_int)   # half-DN quantization
    assert abs(P_recovered - P_true) <= tol_W * 1.0001
    assert block["saturation_fraction"] == 0.0


def test_camera_ideal_saturates_at_full_well():
    lam = 550e-9
    cube = np.array([[[1.0]]])            # 1 W -- deliberately huge
    lam_lo, lam_hi = lam - 1e-9, lam + 1e-9
    entry = _camera_entry(full_well_e=1000.0, adc_gain_e_per_dn=1.0,
                          bit_depth=16)
    pixel_area = (entry["pixel_pitch_um"] * 1e-6) ** 2
    report = {"detectors": {"Det": {"instrument": {}}}}
    pp.render_instrument_camera(cube, lam_lo, lam_hi, pixel_area, entry,
                                "ideal", None, None, "safe", "Det", report)
    block = report["detectors"]["Det"]["instrument"]
    assert block["saturation_fraction"] == 1.0
    assert block["mean_counts"] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# camera: full-mode noise statistics (shot + read) chi-square check
# ---------------------------------------------------------------------------
def test_camera_full_mode_noise_matches_shot_plus_read_variance(tmp_path):
    lam = 550e-9
    QE = 1.0
    t_int = 1.0
    signal_e_target = 5000.0
    P = signal_e_target * H * C / (QE * lam * t_int)
    n = 100                                # 100x100 = 10000 "iid" pixels
    cube = np.full((1, n, n), P)
    lam_lo, lam_hi = lam - 1e-9, lam + 1e-9
    entry = _camera_entry(
        pixel_pitch_um=10.0, integration_time_s_default=t_int,
        lam_um=np.array([0.4, 0.7]), qe=np.array([QE, QE]),
        full_well_e=1e9,                  # never saturate
        read_noise_e=5.0, dark_current_e_per_s=200.0,
        bit_depth=24, adc_gain_e_per_dn=1.0)   # counts ~= electrons
    pixel_area = (entry["pixel_pitch_um"] * 1e-6) ** 2
    report = {"detectors": {"Det": {"instrument": {}}}}
    seed = pp._instrument_seed(1, "camera_generic", "Det")
    pp.render_instrument_camera(cube, lam_lo, lam_hi, pixel_area, entry,
                                "full", seed, tmp_path, "safe", "Det",
                                report)
    counts = np.load(tmp_path / "instr_safe_camera_full_counts.npy")
    assert counts.shape == (n, n)

    dark_mean = entry["dark_current_e_per_s"] * t_int
    theory_var = signal_e_target + dark_mean + entry["read_noise_e"] ** 2
    sample_var = float(np.var(counts.astype(np.float64), ddof=1))

    N = counts.size
    chi2 = (N - 1) * sample_var / theory_var
    lo = stats.chi2.ppf(0.005, df=N - 1)
    hi = stats.chi2.ppf(0.995, df=N - 1)
    assert lo <= chi2 <= hi, (
        "sample variance %.3g vs theory %.3g (chi2=%.1f not in [%.1f,%.1f])"
        % (sample_var, theory_var, chi2, lo, hi))

    theory_mean = signal_e_target + dark_mean
    assert counts.mean() == pytest.approx(theory_mean, rel=0.02)
    assert "seed" in report["detectors"]["Det"]["instrument"]
    assert report["detectors"]["Det"]["instrument"]["seed"] == seed


def test_camera_full_mode_seed_is_reproducible(tmp_path):
    lam = 550e-9
    cube = np.full((1, 8, 8), 2.0e-7)
    lam_lo, lam_hi = lam - 1e-9, lam + 1e-9
    entry = _camera_entry(read_noise_e=3.0, dark_current_e_per_s=50.0)
    pixel_area = (entry["pixel_pitch_um"] * 1e-6) ** 2
    seed = 12345
    out1, out2 = tmp_path / "a", tmp_path / "b"
    out1.mkdir()
    out2.mkdir()
    r1 = {"detectors": {"Det": {"instrument": {}}}}
    r2 = {"detectors": {"Det": {"instrument": {}}}}
    pp.render_instrument_camera(cube, lam_lo, lam_hi, pixel_area, entry,
                                "full", seed, out1, "safe", "Det", r1)
    pp.render_instrument_camera(cube, lam_lo, lam_hi, pixel_area, entry,
                                "full", seed, out2, "safe", "Det", r2)
    c1 = np.load(out1 / "instr_safe_camera_full_counts.npy")
    c2 = np.load(out2 / "instr_safe_camera_full_counts.npy")
    assert np.array_equal(c1, c2)
    assert r1["detectors"]["Det"]["instrument"] \
        == r2["detectors"]["Det"]["instrument"]


# ---------------------------------------------------------------------------
# powermeter
# ---------------------------------------------------------------------------
def test_powermeter_ideal_flat_responsivity_recovers_power():
    lam = 800e-9
    P_true = 1.234e-3
    cube = np.array([[P_true]])           # (1 bin,) spectrum-shaped
    lam_lo, lam_hi = lam - 1e-9, lam + 1e-9
    entry = dict(resp_table=None, flat_responsivity_a_w=0.5,
                aperture_mm=9.5, nep_w_per_sqrthz=1e-13, bandwidth_hz=10.0,
                display_digits=4)
    report = {"detectors": {"Det": {"instrument": {}}}}
    pp.render_instrument_powermeter(cube, lam_lo, lam_hi, entry, "ideal",
                                    None, report, "Det")
    block = report["detectors"]["Det"]["instrument"]
    assert block["power_reported_W"] == pytest.approx(P_true, rel=1e-9)
    assert "seed" not in block
    assert block["lam_ref_nm"] == pytest.approx(800.0, abs=1.0)


def test_powermeter_ideal_table_responsivity_monochromatic_exact():
    lam = 700e-9
    P_true = 5.0e-6
    cube = np.array([[P_true]])
    lam_lo, lam_hi = lam - 1e-9, lam + 1e-9
    entry = dict(
        resp_table={"lam_um": np.array([0.6, 0.7, 0.8]),
                   "responsivity_a_w": np.array([0.3, 0.45, 0.5])},
        flat_responsivity_a_w=None, aperture_mm=9.5,
        nep_w_per_sqrthz=1e-13, bandwidth_hz=10.0, display_digits=4)
    report = {"detectors": {"Det": {"instrument": {}}}}
    pp.render_instrument_powermeter(cube, lam_lo, lam_hi, entry, "ideal",
                                    None, report, "Det")
    # monochromatic at exactly a table node (700nm -> R=0.45): the
    # photocurrent/R_ref inversion is exact regardless of table shape
    assert report["detectors"]["Det"]["instrument"]["power_reported_W"] \
        == pytest.approx(P_true, rel=1e-9)


def test_powermeter_full_mode_noise_matches_nep_bandwidth():
    lam = 800e-9
    P_true = 1e-6
    lam_lo, lam_hi = lam - 1e-9, lam + 1e-9
    entry = dict(resp_table=None, flat_responsivity_a_w=0.4,
                aperture_mm=9.5, nep_w_per_sqrthz=2e-9, bandwidth_hz=100.0,
                display_digits=6)
    theory_sigma = entry["nep_w_per_sqrthz"] * np.sqrt(entry["bandwidth_hz"])
    readings = []
    for i in range(400):
        cube = np.array([[P_true]])
        seed = pp._instrument_seed(1, "powermeter_generic", "Det%d" % i)
        report = {"detectors": {"Det%d" % i: {"instrument": {}}}}
        pp.render_instrument_powermeter(cube, lam_lo, lam_hi, entry, "full",
                                        seed, report, "Det%d" % i)
        readings.append(
            report["detectors"]["Det%d" % i]["instrument"]["power_reported_W"])
    readings = np.array(readings)
    sample_var = float(np.var(readings - P_true, ddof=1))
    theory_var = theory_sigma ** 2
    N = len(readings)
    chi2 = (N - 1) * sample_var / theory_var
    lo = stats.chi2.ppf(0.005, df=N - 1)
    hi = stats.chi2.ppf(0.995, df=N - 1)
    assert lo <= chi2 <= hi, (
        "sample var %.3g vs theory %.3g (chi2=%.1f not in [%.1f,%.1f])"
        % (sample_var, theory_var, chi2, lo, hi))


# ---------------------------------------------------------------------------
# spectrometer
# ---------------------------------------------------------------------------
def _spectrometer_entry(**overrides):
    entry = dict(lam_lo_nm=400.0, lam_hi_nm=800.0, resolution_fwhm_nm=2.0,
                slit_um=25.0, stray_light_floor=1e-3,
                lam_um=np.array([0.4, 0.5, 0.6, 0.7, 0.8]),
                qe=np.array([0.3, 0.4, 0.4, 0.35, 0.2]))
    entry.update(overrides)
    return entry


def test_spectrometer_ideal_stray_light_floor_present_away_from_peak():
    bins = 5
    lam_lo, lam_hi = 400e-9, 800e-9
    pw = np.zeros(bins)
    pw[2] = 1.0                            # sharp line at bin-center 600nm
    cube = pw.reshape(bins, 1, 1)
    entry = _spectrometer_entry()
    report = {"detectors": {"Det": {"instrument": {}}}}
    pp.render_instrument_spectrometer(cube, lam_lo, lam_hi, entry, "ideal",
                                      None, None, "safe", "Det", report)
    block = report["detectors"]["Det"]["instrument"]
    assert block["peak_power_W"] > 0
    assert "seed" not in block
    assert block["resolution_fwhm_nm"] == 2.0


def test_spectrometer_full_mode_seed_reproducible():
    bins = 5
    lam_lo, lam_hi = 400e-9, 800e-9
    pw = np.array([0.1, 0.3, 1.0, 0.4, 0.05])
    cube = pw.reshape(bins, 1, 1)
    entry = _spectrometer_entry()
    seed = 999
    r1 = {"detectors": {"Det": {"instrument": {}}}}
    r2 = {"detectors": {"Det": {"instrument": {}}}}
    pp.render_instrument_spectrometer(cube, lam_lo, lam_hi, entry, "full",
                                      seed, None, "safe", "Det", r1)
    pp.render_instrument_spectrometer(cube, lam_lo, lam_hi, entry, "full",
                                      seed, None, "safe", "Det", r2)
    assert r1["detectors"]["Det"]["instrument"] \
        == r2["detectors"]["Det"]["instrument"]
    assert r1["detectors"]["Det"]["instrument"]["seed"] == seed


# ---------------------------------------------------------------------------
# render_instrument dispatcher: unassigned / unknown-row / bad-spec paths
# ---------------------------------------------------------------------------
def test_render_instrument_noop_when_unassigned():
    report = {"detectors": {"Det": {}}}
    pp.render_instrument(np.zeros((1, 1, 1)), 400e-9, 800e-9, 1e-8, None,
                         None, "safe", "Det", report, {}, {}, 0)
    assert "instrument" not in report["detectors"]["Det"]


def test_render_instrument_unknown_row_notes_and_skips(capsys):
    report = {"detectors": {"Det": {}}}
    pp.render_instrument(np.zeros((1, 1, 1)), 400e-9, 800e-9, 1e-8, None,
                         None, "safe", "Det", report,
                         {"Det": "not_a_real_row"}, {}, 0)
    assert "instrument" not in report["detectors"]["Det"]
    assert "unknown instrument row" in capsys.readouterr().out


def test_render_instrument_bad_spec_notes_and_skips(capsys):
    report = {"detectors": {"Det": {}}}
    pp.render_instrument(np.zeros((1, 1, 1)), 400e-9, 800e-9, 1e-8, None,
                         None, "safe", "Det", report,
                         {"Det": "camera_generic:bogus"}, {}, 0)
    assert "instrument" not in report["detectors"]["Det"]
    assert "bad instrument spec" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# end-to-end: render_detector wires a synthetic h5 into report.json
# ---------------------------------------------------------------------------
def _write_synthetic_h5(path, label, H=4, W=4, bins=3):
    lam_lo, lam_hi = 500e-9, 600e-9
    cube = np.full((bins, H, W), 1e-9)
    with h5py.File(path, "w") as h:
        h["spectral_cube_mean"] = cube
        h["mask"] = np.ones((H, W), dtype=bool)
        h.attrs.update({
            "label": label, "H": H, "W": W, "pixel_m": 5e-5,
            "lam_lo_m": lam_lo, "lam_hi_m": lam_hi,
            "xhat": [0.0, 1.0, 0.0], "yhat": [0.0, 0.0, 1.0],
            "normal": [1.0, 0.0, 0.0],
            "x_lo": -W * 5e-5 / 2, "y_lo": -H * 5e-5 / 2,
            "seeds": 1,
        })


def test_render_detector_populates_instrument_report_block(tmp_path):
    from raytracer.optprops import load_instruments
    label = "Cam.Pad.Face1"
    h5path = tmp_path / "det.h5"
    _write_synthetic_h5(h5path, label)
    img, spec, instr = (tmp_path / d for d in ("images", "spectra",
                                               "instrument"))
    for d in (img, spec, instr):
        d.mkdir()
    registry = load_instruments()          # shipped camera_generic row
    report = {"detectors": {}}
    pp.render_detector(
        h5path, img, spec, report,
        instrument_bodies={"Cam": "camera_generic:ideal"},
        instrument_registry=registry, outdir_instr=instr, case_seed=42)
    block = report["detectors"][label]["instrument"]
    assert block["row"] == "camera_generic"
    assert block["class"] == "camera"
    assert block["mode"] == "ideal"
    assert "seed" not in block
    assert (instr / "instr_Cam_Pad_Face1_camera_ideal.png").exists()
    assert (instr / "instr_Cam_Pad_Face1_camera_ideal_counts.npy").exists()
