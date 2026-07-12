# =============================================================================
# test_gdd_budget.py — pulsed-optics Phase P5: the per-element material-
# dispersion budget (--gdd-budget / free byproduct of any time product).
#
# Covers:
#   * the budget's GDD column equals gdd_per_length(material, lam_ref) x
#     the table's own mean path at 1e-3 relative (THE table-consistency
#     gate), n_g matches the material's group index, totals sum the rows,
#     and the mean path is the physical slab thickness (power-weighted;
#     double-bounce echoes push it high by ~2 R^2).
#   * the budget appears WITHOUT --gdd-budget on a pulsed scene (group
#     delay already tracked for the auto time products).
#   * traced pulse FWHM matches the budget's own tau_out prediction at 2%
#     (THE locked broadening gate) — fused silica at 400 nm.
#   * --gdd-budget on a CW scene forces group-delay tracking (block
#     present, engine=python reason=gdd_budget) without any time_*
#     datasets appearing in the detector .h5.
#   * tau_out is exactly tau0*sqrt(1+(4 ln2 phi2/tau0^2)^2) over the
#     block's own phi2.
#   * run_pipeline forwards --gdd-budget verbatim.
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_gdd_budget.py -q
# =============================================================================
import json
import sys
from pathlib import Path

import numpy as np
import h5py
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "raytracer" / "tests"))

import common                                              # noqa: E402
import cli_specs                                            # noqa: E402
import run_trace                                            # noqa: E402
import run_pipeline                                         # noqa: E402
import scenehelpers as sh                                  # noqa: E402
from raytracer import cengine                              # noqa: E402
from raytracer.materials import gdd_per_length             # noqa: E402
from raytracer.optprops import load_optical_properties     # noqa: E402
from post_process import _fwhm_interp                      # noqa: E402

C = 299792458.0


def _run(tmp_path, bodies, extra, name="case", rays=3000, nlambda=3):
    model = sh.make_model(bodies)
    common.validate_model(model)
    mj = tmp_path / (name + "_model.json")
    common.write_json(mj, model)
    case = tmp_path / name
    rc = run_trace.main([
        "--model-json", str(mj), "--case-dir", str(case),
        "--rays", str(rays), "--nlambda", str(nlambda),
        "--resolution", "48", "--workers", "1"] + extra)
    assert rc == 0
    return case


def _case_json(case):
    return json.loads((case / "case.json").read_text())


def _two_slab_pulsed():
    return [sh.source_body(x=-0.02, half=0.001, power_mW=1.0,
                           lambdac_nm=633.0, lambdamin_nm=630.0,
                           lambdamax_nm=636.0, pulse_duration_ps=0.5),
            sh.slab_body("SlabA", "bk7", 0.0, 0.004, half=0.01),
            sh.slab_body("SlabB", "fused_silica", 0.006, 0.011, half=0.01),
            sh.detector_body(x=0.03, half=0.02)]


# --------------------------------------------------------------------------- #
# 1. table consistency: GDD == gdd_per_length * the table's own mean path
# --------------------------------------------------------------------------- #
def test_budget_rows_consistent_and_free_on_pulsed_scene(tmp_path):
    # NO --gdd-budget: the pulsed auto time products already track group
    # delay, so the budget must come for free
    case = _run(tmp_path, _two_slab_pulsed(), [])
    block = _case_json(case).get("gdd_budget")
    assert block, "pulsed scene must emit the budget without the flag"
    assert block["lambda_ref_nm"] == pytest.approx(633.0)
    rows = {r["label"]: r for r in block["rows"]}
    assert set(rows) == {"SlabA", "SlabB"}
    matdb = load_optical_properties().matdb
    lam = block["lambda_ref_nm"] * 1e-9
    for label, matname, L_phys in (("SlabA", "bk7", 0.004),
                                   ("SlabB", "fused_silica", 0.005)):
        r = rows[label]
        assert r["material"] == matname
        mat = matdb.get(matname)
        L = r["L_bar_mm"] * 1e-3
        # power-weighted mean path ~ physical thickness (a hair BELOW it:
        # flux_in books the full incident power, front-face Fresnel
        # reflection included, while only the transmitted share tallies
        # bulk path; the double-bounce echo pushes the other way)
        assert L == pytest.approx(L_phys, rel=0.01)
        # THE 1e-3 consistency gate: the table's own numbers must agree
        assert r["gdd_fs2"] == pytest.approx(
            float(gdd_per_length(mat, lam)) * L * 1e30, rel=1e-3)
        assert r["n_g"] == pytest.approx(float(mat.n_group(lam)), rel=1e-6)
        assert r["gd_fs"] == pytest.approx(
            r["n_g"] * L / C * 1e15, rel=1e-9)
    for k in ("gd_fs", "gdd_fs2", "tod_fs3"):
        assert block["total"][k] == pytest.approx(
            sum(r[k] for r in block["rows"]), rel=1e-12)


# --------------------------------------------------------------------------- #
# 2. tau_out is exactly the Gaussian-broadening formula over the block's phi2
# --------------------------------------------------------------------------- #
def test_pulse_annotation_formula(tmp_path):
    case = _run(tmp_path, _two_slab_pulsed(), [])
    block = _case_json(case)["gdd_budget"]
    assert len(block["pulses"]) == 1
    p = block["pulses"][0]
    assert p["tau0_fs"] == pytest.approx(500.0)
    assert p["lambda_c_nm"] == pytest.approx(633.0)
    want = p["tau0_fs"] * np.sqrt(
        1.0 + (4.0 * np.log(2.0) * p["phi2_fs2"] / p["tau0_fs"] ** 2) ** 2)
    assert p["tau_out_fs"] == pytest.approx(want, rel=1e-12)
    # the reference source IS the pulsed source, so phi2 must equal the
    # table total
    assert p["phi2_fs2"] == pytest.approx(block["total"]["gdd_fs2"],
                                          rel=1e-12)


# --------------------------------------------------------------------------- #
# 3. THE locked gate: traced FWHM matches the budget's tau_out at 2%
# --------------------------------------------------------------------------- #
def test_traced_broadening_matches_budget_within_2pct(tmp_path):
    tau0 = 100e-15
    lam0 = 400.0                                    # nm (strong GDD)
    L = 0.02
    sig_w = 4.0 * np.log(2.0) / tau0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sig_lam_nm = (lam0 * 1e-9) ** 2 * sig_w / (2.0 * np.pi * C) / 1e-9
    silica = load_optical_properties().matdb.get("fused_silica")
    n_g = float(silica.n_group(lam0 * 1e-9))
    t_c = (0.03 + n_g * L) / C
    spec = "%.12g,%.12g" % ((t_c - 1e-12) / 1e-9, (t_c + 1e-12) / 1e-9)
    case = _run(tmp_path, [
        sh.source_body(x=-0.02, half=0.001, power_mW=1.0,
                       lambdac_nm=lam0,
                       lambdamin_nm=lam0 - sig_lam_nm,
                       lambdamax_nm=lam0 + sig_lam_nm,
                       pulse_duration_ps=tau0 / 1e-12),
        sh.slab_body("Slab", "fused_silica", 0.0, L, half=0.01),
        sh.detector_body(x=0.03, half=0.02),
    ], ["--time-products", "pulse", "--time-bins", "1024",
        "--time-window", spec], rays=10000, nlambda=17)
    block = _case_json(case)["gdd_budget"]
    tau_out = block["pulses"][0]["tau_out_fs"] * 1e-15
    with h5py.File(case / "detectors" / "Det_Pad_Face1.h5") as h:
        prof = h["time_profile"][...]
        t_lo = float(h.attrs["t_lo_s"])
        dt = float(h.attrs["time_dt_s"])
    tc = t_lo + (np.arange(len(prof)) + 0.5) * dt
    fwhm, _, _ = _fwhm_interp(tc, prof)
    assert fwhm is not None
    assert abs(fwhm - tau_out) / tau_out < 0.02, \
        "traced %.4g fs vs budget tau_out %.4g fs" % (fwhm / 1e-15,
                                                      tau_out / 1e-15)


# --------------------------------------------------------------------------- #
# 4. --gdd-budget on a CW scene: budget without time products
# --------------------------------------------------------------------------- #
def test_flag_on_cw_scene_budget_only(tmp_path):
    bodies = [sh.source_body(x=-0.02, half=0.001, power_mW=1.0,
                             lambdac_nm=633.0),
              sh.slab_body("Slab", "bk7", 0.0, 0.004, half=0.01),
              sh.detector_body(x=0.03, half=0.02)]
    case = _run(tmp_path, bodies, ["--gdd-budget"])
    cj = _case_json(case)
    assert "gdd_budget" in cj
    assert "time_products" not in cj
    assert cj["engine"] == "python"
    if cengine.binary_path() is not None:
        assert "gdd_budget" in cj["engine_reason"]
    with h5py.File(case / "detectors" / "Det_Pad_Face1.h5") as h:
        assert not any(k.startswith("time_") for k in h.keys())
    # and without the flag the CW scene emits no budget at all
    case2 = _run(tmp_path, bodies, [], name="off")
    assert "gdd_budget" not in _case_json(case2)


# --------------------------------------------------------------------------- #
# 5. run_pipeline forwards the flag
# --------------------------------------------------------------------------- #
def test_run_pipeline_forwards_gdd_budget():
    p = cli_specs.build_parser("pipeline")
    args = p.parse_args(["--models", "x.FCStd", "--preset", "quick",
                         "--gdd-budget"])
    cmd = run_pipeline.trace_cmd("x", Path("/tmp/case"), args)
    assert "--gdd-budget" in cmd
    args = p.parse_args(["--models", "x.FCStd", "--preset", "quick"])
    assert "--gdd-budget" not in run_pipeline.trace_cmd(
        "x", Path("/tmp/case"), args)
