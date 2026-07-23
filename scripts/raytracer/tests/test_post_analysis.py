# =============================================================================
# test_post_analysis.py -- render_field_analysis / render_wavefront
# (post_process.py) validated against SYNTHETIC case directories: no
# tracing, no FreeCAD, no run_trace.py -- everything here builds the exact
# on-disk artifacts those two stages consume directly:
#
#   (a) a detector .h5 with a hand-built 'fields/<key>/{Ex,Ey}' Gaussian
#       field pair (the layout run_trace.save_detectors writes for
#       --save-fields; see post_process._iter_field_keys) -> checks
#       render_field_analysis's PNGs/CSV/report block.
#   (b) a hand-built rays_full.npz (the layout run_trace.write_rays_full
#       writes for --export-rays) with an EXACTLY known injected Zernike
#       defocus (Z4) baked into the 'opl' column and identical landing/
#       birth positions so the OPD is analytically predictable -> checks
#       render_wavefront recovers the injected coefficient and Strehl.
#   (c) both are silent no-ops (no crash, no analysis/ dir) when their
#       inputs are absent.
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_post_analysis.py -q
# =============================================================================
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import post_process                                        # noqa: E402
from raytracer import analysis as an                        # noqa: E402


# ---------------------------------------------------------------------------
# (a) render_field_analysis -- synthetic Gaussian field pair
# ---------------------------------------------------------------------------
def _gaussian_field(H, W, sigma_px, amp):
    ys, xs = np.indices((H, W)).astype(np.float64)
    cy, cx = H / 2.0, W / 2.0
    r2 = (xs - cx) ** 2 + (ys - cy) ** 2
    Ex = (amp * np.exp(-r2 / (2.0 * sigma_px ** 2))).astype(np.complex128)
    Ey = np.zeros((H, W), dtype=np.complex128)
    return Ex, Ey


def _write_fields_h5(path, label="Det.Pad.Face1", H=64, W=64,
                     pixel_m=2e-6):
    """A minimal detector .h5 carrying only what render_field_analysis
    reads: top-level attrs (label, pixel_m) + two 'fields/<key>/{Ex,Ey}'
    groups -- key '0_0_0' is the DOMINANT (higher-peak) key, '0_0_1' a
    weaker/broader one, so the dominant-key selection logic is exercised."""
    Ex0, Ey0 = _gaussian_field(H, W, sigma_px=6.0, amp=1.0)
    Ex1, Ey1 = _gaussian_field(H, W, sigma_px=10.0, amp=0.4)
    with h5py.File(path, "w") as h:
        h["fields/0_0_0/Ex"] = Ex0
        h["fields/0_0_0/Ey"] = Ey0
        h["fields/0_0_1/Ex"] = Ex1
        h["fields/0_0_1/Ey"] = Ey1
        h.attrs["label"] = label
        h.attrs["pixel_m"] = pixel_m
        h.attrs["H"] = H
        h.attrs["W"] = W


def test_render_field_analysis_pngs_report_and_csv(tmp_path):
    label = "Det.Pad.Face1"
    h5path = tmp_path / "det.h5"
    _write_fields_h5(h5path, label=label)
    adir = tmp_path / "analysis"
    report = {"detectors": {}}
    csv_emitter = post_process.CsvEmitter(tmp_path / "data")

    post_process.render_field_analysis(h5path, adir, report, csv_emitter)

    safe = label.replace(".", "_")
    assert (adir / ("psf_%s.png" % safe)).exists()
    assert (adir / ("mtf_%s.png" % safe)).exists()
    assert (adir / ("ee_%s.png" % safe)).exists()

    block = report["detectors"][label]["analysis"]
    for k in ("psf_peak_W_m2", "mtf50_tan_cy_mm", "mtf50_sag_cy_mm",
              "ee_r50_um", "ee_r80_um", "ee_r90_um"):
        assert k in block
    # three panel rows: the two physical keys + the incoherent-sum 'all' row
    assert set(block["keys"]) == {"0_0_0", "0_0_1", "all"}

    # dominant-power key (higher amplitude) drives the headline scalars
    assert block["psf_peak_W_m2"] == pytest.approx(
        block["keys"]["0_0_0"]["psf_peak_W_m2"])
    assert block["psf_peak_W_m2"] != pytest.approx(
        block["keys"]["0_0_1"]["psf_peak_W_m2"])

    # EE radii monotone for every rendered key
    for name, m in block["keys"].items():
        r50, r80, r90 = m["ee_r50_um"], m["ee_r80_um"], m["ee_r90_um"]
        assert r50 < r80 < r90, name
        assert np.isfinite(m["mtf50_tan_cy_mm"])
        assert np.isfinite(m["mtf50_sag_cy_mm"])

    # CSVs: radial PSF profile / MTF slices / EE curve per key (no 2-D maps)
    data = tmp_path / "data"
    for key in ("0_0_0", "0_0_1", "all"):
        assert (data / ("psf_radial_%s_%s.csv" % (safe, key))).exists()
        assert (data / ("mtf_slices_%s_%s.csv" % (safe, key))).exists()
        assert (data / ("ee_%s_%s.csv" % (safe, key))).exists()


def test_render_field_analysis_no_csv_without_emitter(tmp_path):
    h5path = tmp_path / "det.h5"
    _write_fields_h5(h5path)
    adir = tmp_path / "analysis"
    report = {"detectors": {}}
    post_process.render_field_analysis(h5path, adir, report, None)
    assert not (tmp_path / "data").exists()
    assert (adir / "psf_Det_Pad_Face1.png").exists()


# ---------------------------------------------------------------------------
# (b) render_wavefront -- synthetic rays_full.npz with an EXACT injected
# Zernike defocus (see module docstring for the hit==ref trick that makes
# the OPD analytically equal to the injected Zernike sum, up to a piston
# shift that the fit absorbs without touching the recovered defocus term).
# ---------------------------------------------------------------------------
def _write_synthetic_rays_full(case_dir, label="Det.Pad.Face1", lam_m=633e-9,
                               truth_j4_waves=0.30):
    safe = label.replace(".", "_")
    # deterministic, ORIGIN-SYMMETRIC pupil grid: a regular (x,y) mesh over
    # [-1,1]^2 clipped to the unit disc. Odd sample counts + symmetric
    # endpoints guarantee (i) the power-weighted centroid is EXACTLY zero
    # and (ii) the point (x=1,y=0) sits exactly on the unit circle, so
    # _pupil_xy's max-radius normalization returns px,py == x,y exactly.
    xs = np.linspace(-1.0, 1.0, 41)
    X, Y = np.meshgrid(xs, xs)
    keep = (X ** 2 + Y ** 2) <= 1.0 + 1e-12
    x = X[keep].ravel()
    y = Y[keep].ravel()
    n = len(x)
    assert n > 200

    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    truth = np.zeros(15)
    truth[3] = truth_j4_waves * lam_m       # Noll j=4, defocus
    opl = an.zernike_basis(15, rho, theta) @ truth   # metres

    r_pupil_m = 0.005                        # arbitrary emitting-aperture
    birth_pos = np.stack([r_pupil_m * x, r_pupil_m * y,
                         np.zeros(n)], axis=1)
    # identical landing point for every ray (hit == ref by construction,
    # see module docstring) -- the whole injected OPD then comes from
    # 'opl' alone, with zero ambient-distance contribution.
    ref_point = np.array([0.0, 0.0, 0.05])
    pos = np.tile(ref_point, (n, 1))

    payload = {
        "%s/pos" % safe: pos,
        "%s/opl" % safe: opl,
        "%s/lam" % safe: np.full(n, lam_m),
        "%s/source_id" % safe: np.zeros(n, dtype=np.int16),
        "%s/lam_stratum" % safe: np.zeros(n, dtype=np.int16),
        "%s/pol_stratum" % safe: np.zeros(n, dtype=np.int16),
        "%s/power" % safe: np.ones(n),
        "%s/coherent" % safe: np.ones(n, dtype=bool),
        "%s/birth_pos" % safe: birth_pos,
    }
    meta = {"seed": 42, "model": "synthetic", "max_reflections": 6,
           "export_rays_max": 2000000,
           "detectors": {safe: {
               "label": label, "xhat": [1.0, 0.0, 0.0],
               "yhat": [0.0, 1.0, 0.0], "normal": [0.0, 0.0, 1.0],
               "x_lo": 0.0, "y_lo": 0.0, "n_total": n, "n_kept": n,
               "kept_fraction": 1.0}}}
    payload["meta"] = np.array(json.dumps(meta))
    np.savez(case_dir / "rays_full.npz", **payload)
    return truth


def test_render_wavefront_recovers_injected_defocus(tmp_path):
    label = "Det.Pad.Face1"
    lam_m = 633e-9
    truth_j4 = 0.30
    truth = _write_synthetic_rays_full(tmp_path, label=label, lam_m=lam_m,
                                       truth_j4_waves=truth_j4)
    report = {"detectors": {}}
    csv_emitter = post_process.CsvEmitter(tmp_path / "data")

    post_process.render_wavefront(tmp_path, report, csv_emitter)

    safe = label.replace(".", "_")
    assert (tmp_path / "analysis" / ("wavefront_%s.png" % safe)).exists()
    zpath = tmp_path / "data" / ("zernike_%s.csv" % safe)
    assert zpath.exists()

    block = report["detectors"][label]["wavefront"]
    assert block["n_rays"] > 200
    assert len(block["keys"]) == 1
    row = block["keys"][0]
    assert row["source_id"] == 0 and row["lam_stratum"] == 0 \
        and row["pol_stratum"] == 0

    # Z4 (defocus) recovered within 2% of the injected value; rms_waves is
    # entirely defocus (piston/tip/tilt are fit but excluded from rms).
    assert abs(row["rms_waves"] - truth_j4) / truth_j4 < 0.02
    assert abs(block["rms_waves"] - truth_j4) / truth_j4 < 0.02

    # Strehl matches strehl_marechal() of the SAME residual RMS directly.
    expected_strehl = an.strehl_marechal(truth_j4 * lam_m, lam_m)
    assert abs(block["strehl_marechal"] - expected_strehl) < 1e-6

    # zernike_<label>.csv: noll_j==4 coefficient (in waves) matches truth
    import csv
    with open(zpath) as fh:
        rows = list(csv.DictReader(fh))
    j4_rows = [r for r in rows if int(r["noll_j"]) == 4]
    assert len(j4_rows) == 1
    assert abs(float(j4_rows[0]["coeff_waves"]) - truth_j4) / truth_j4 < 0.02
    assert j4_rows[0]["name"] == "defocus"
    assert int(j4_rows[0]["fringe_j"]) > 0


def test_render_wavefront_wavefront_point_override(tmp_path):
    """--wavefront-point overrides the default power-weighted landing
    centroid; since every synthetic ray already lands at the same point,
    an override at that SAME point (in mm) must reproduce the identical
    fit (hit == ref either way)."""
    label = "Det.Pad.Face1"
    lam_m = 633e-9
    _write_synthetic_rays_full(tmp_path, label=label, lam_m=lam_m,
                               truth_j4_waves=0.2)
    report = {"detectors": {}}
    # ref_point was [0, 0, 0.05] -- in the (xhat, yhat) = (x, y) frame that
    # is (0mm, 0mm) (the z=0.05 component lies along 'normal', restored
    # from the ray's own position by render_wavefront).
    post_process.render_wavefront(tmp_path, report, None,
                                  wavefront_point=(0.0, 0.0))
    block = report["detectors"][label]["wavefront"]
    assert abs(block["rms_waves"] - 0.2) / 0.2 < 0.02


# ---------------------------------------------------------------------------
# (c) silent no-ops when inputs are absent
# ---------------------------------------------------------------------------
def test_render_field_analysis_noop_without_fields_group(tmp_path):
    h5path = tmp_path / "det.h5"
    with h5py.File(h5path, "w") as h:
        h["mask"] = np.ones((4, 4), dtype=bool)
        h.attrs["label"] = "Det.Pad.Face1"
        h.attrs["pixel_m"] = 1e-6
    adir = tmp_path / "analysis"
    report = {"detectors": {}}
    post_process.render_field_analysis(h5path, adir, report, None)
    assert not adir.exists()
    assert report["detectors"] == {}


def test_render_wavefront_noop_without_npz(tmp_path):
    report = {"detectors": {}}
    post_process.render_wavefront(tmp_path, report, None)
    assert not (tmp_path / "analysis").exists()
    assert report["detectors"] == {}


def test_render_wavefront_noop_below_min_rays(tmp_path):
    # only 50 rays (< MIN_WAVEFRONT_RAYS) -> no-op, no analysis/ dir
    label = "Det.Pad.Face1"
    safe = label.replace(".", "_")
    n = 50
    rng = np.random.default_rng(0)
    bp = rng.normal(size=(n, 3)) * 1e-3
    bp[:, 2] = 0.0
    pos = np.tile(np.array([0.0, 0.0, 0.05]), (n, 1))
    payload = {
        "%s/pos" % safe: pos,
        "%s/opl" % safe: np.zeros(n),
        "%s/lam" % safe: np.full(n, 633e-9),
        "%s/source_id" % safe: np.zeros(n, dtype=np.int16),
        "%s/lam_stratum" % safe: np.zeros(n, dtype=np.int16),
        "%s/pol_stratum" % safe: np.zeros(n, dtype=np.int16),
        "%s/power" % safe: np.ones(n),
        "%s/coherent" % safe: np.ones(n, dtype=bool),
        "%s/birth_pos" % safe: bp,
    }
    meta = {"detectors": {safe: {
        "label": label, "xhat": [1.0, 0.0, 0.0], "yhat": [0.0, 1.0, 0.0],
        "normal": [0.0, 0.0, 1.0], "x_lo": 0.0, "y_lo": 0.0,
        "n_total": n, "n_kept": n, "kept_fraction": 1.0}}}
    payload["meta"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "rays_full.npz", **payload)

    report = {"detectors": {}}
    post_process.render_wavefront(tmp_path, report, None)
    assert not (tmp_path / "analysis").exists()
    assert report["detectors"] == {}
