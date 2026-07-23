# =============================================================================
# test_sequential_backend.py -- P4b engine-side routing + DLS machinery
# (engine3.md Sec 5/8). Runs under "$MIEWB_OPTICS_PYTHON" (NO Optiland,
# NO FreeCAD): it exercises the operand->backend routing table, the
# single-sourced merit residual form, the damped-least-squares driver on an
# analytic bowl, and the sequential-backend guards in optimize.py/tolerance.py
# -- all with an injected evaluate_fn, exactly as the existing
# test_optimize.py does. The Optiland bridge itself is tested under env/ in
# mieworkbench/tests/test_optiland_sequential.py (the pinned-interpreter split).
# =============================================================================
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import optimize   # noqa: E402
import tolerance  # noqa: E402


# --------------------------------------------------------------------------
# operand -> backend routing table (the ONE authority)
# --------------------------------------------------------------------------
def test_operand_backend_routing():
    assert optimize.operand_backend("spot_rms") == "sequential"
    assert optimize.operand_backend("focus") == "sequential"
    assert optimize.operand_backend("encircled_energy") == "sequential"
    assert optimize.operand_backend("mtf50") == "mc"
    assert optimize.operand_backend("detected_power") == "mc"
    # a raw flattened report.json key is MC-produced
    assert optimize.operand_backend("Body.Face5.total_power_W") == "mc"


def test_mc_only_operands_filter():
    ops = [{"operand": "spot_rms"}, {"operand": "detected_power"},
           {"operand": "focus"}, {"operand": "Body.x"}]
    assert sorted(optimize.mc_only_operands(ops)) == ["Body.x", "detected_power"]


# --------------------------------------------------------------------------
# single-sourced merit: residual vector <-> scalar score
# --------------------------------------------------------------------------
def _out(spot_um, det="D"):
    return {"backend_used": "sequential",
            "detectors": {det: {"spot": [{"rms_radius_um": spot_um,
                                          "rms_pw_radius_um": spot_um,
                                          "n_rays": 100}]}}}


def test_residuals_sum_of_squares_equals_score():
    """sum(residual_i^2) == score() for DLS-supported operands, so the two
    optimizer paths minimize the SAME merit."""
    mf = optimize.MeritFunction([
        {"operand": "spot_rms", "detector": None, "target": 2.0, "weight": 3.0},
    ])
    out = _out(5.0)
    merit, _, _ = mf.score(out)
    resid, _, missing = mf.residuals(out)
    assert not missing
    assert math.isclose(sum(r * r for r in resid), merit, rel_tol=1e-12)
    # explicit value: w*(v-t)^2 = 3*(5-2)^2 = 27
    assert math.isclose(merit, 27.0, rel_tol=1e-12)


def test_residuals_penalized_are_finite_sentinels():
    mf = optimize.MeritFunction([
        {"operand": "spot_rms", "detector": "MISSING", "target": 0.0,
         "weight": 1.0}])
    resid, rows, missing = mf.residuals({"detectors": {}})
    assert missing
    assert all(math.isfinite(r) for r in resid)
    assert len(resid) == 1


def test_dls_rejects_pure_maximize():
    mf = optimize.MeritFunction([
        {"operand": "detected_power", "detector": None, "target": 0.0,
         "weight": 1.0}])
    ok, reason = mf.dls_supported()
    assert not ok and "maximize" in reason


# --------------------------------------------------------------------------
# the DLS driver converges on an analytic bowl (no trace, injected merit)
# --------------------------------------------------------------------------
def test_dls_converges_on_analytic_bowl():
    """spot RMS = 100*|x - 3| (bowl minimized at x=3); DLS over the residual
    sqrt(w)*(v-0) must drive x to 3."""
    target_x = 3.0

    def evaluate_fn(params):
        return _out(100.0 * abs(params["x"] - target_x))

    eng = optimize.OptimizationEngine(
        [{"name": "x", "start": 0.0, "lo": -5.0, "hi": 8.0}],
        [{"operand": "spot_rms", "detector": None, "target": 0.0,
          "weight": 1.0}],
        evaluate_fn, algorithm="dls", budget=40, tol=1e-9, progress=False)
    best = eng.run()
    assert abs(best["params"]["x"] - target_x) < 1e-2, best
    assert best["merit"] < 1.0


def test_simplex_alias_is_nelder_mead():
    """--algorithm simplex still runs the derivative-free path (unchanged)."""
    def evaluate_fn(params):
        return _out(100.0 * abs(params["x"] - 2.0))

    eng = optimize.OptimizationEngine(
        [{"name": "x", "start": 0.0, "lo": -5.0, "hi": 8.0}],
        [{"operand": "spot_rms", "detector": None, "target": 0.0,
          "weight": 1.0}],
        evaluate_fn, algorithm="simplex", budget=60, tol=1e-9, progress=False)
    best = eng.run()
    assert abs(best["params"]["x"] - 2.0) < 5e-2, best


# --------------------------------------------------------------------------
# CLI guards: sequential backend rejects MC-only operands
# --------------------------------------------------------------------------
def test_optimize_cli_rejects_mc_operand_on_sequential():
    with pytest.raises(SystemExit):
        optimize.main(["--model", "x.FCStd", "--var", "R:25:20:30",
                       "--operand", "detected_power:0:1",
                       "--eval-backend", "sequential"])


def test_tolerance_cli_rejects_mc_operand_on_sequential():
    with pytest.raises(SystemExit):
        tolerance.main(["--model", "x.FCStd", "--tolerance", "R:25:normal:1",
                        "--operand", "mtf50:0:1", "--eval-backend",
                        "sequential"])


# --------------------------------------------------------------------------
# fast_eval accepts the sequential backend and exposes its plumbing
# --------------------------------------------------------------------------
def test_fast_eval_accepts_sequential_backend():
    import fast_eval
    model = "auto_designed_lens.FCStd"    # resolves via basemodels/
    # bad backend still rejected
    with pytest.raises(ValueError):
        fast_eval.Evaluator(model, backend="bogus")
    # sequential is a recognized backend with its own helpers + config.
    ev = fast_eval.Evaluator(model, backend="sequential")
    assert ev.backend == "sequential"
    assert ev.model_stop is True and ev.seq_n_rays > 0
    assert hasattr(ev, "_evaluate_sequential")
    assert hasattr(ev, "_finish_sequential")
    assert issubclass(fast_eval.SequentialUnsupported, RuntimeError)
