# =============================================================================
# test_optimize.py — the merit-function optimizer's unit tests, against a
# MOCK evaluator (a known analytic bowl), so the loop's convergence is
# pinned WITHOUT a slow trace: both algorithms must find the bowl minimum
# within tolerance, failed evaluations must be penalized (never fatal),
# and the merit/operand math is tested against synthetic report blocks.
# =============================================================================
import json
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import cli_specs                                     # noqa: E402
import optimize                                      # noqa: E402
from optimize import (MeritFunction, OptimizationEngine,  # noqa: E402
                      OptimizeError, PENALTY, operand_needs, operand_value)

BOWL_MIN = {"x": 3.0, "y": -1.0}
VARIABLES = [{"name": "x", "start": 8.0, "lo": -10.0, "hi": 10.0},
             {"name": "y", "start": 6.0, "lo": -10.0, "hi": 10.0}]
RAW_OPERAND = [{"operand": "Det.bowl", "detector": None,
                "target": 0.0, "weight": 1.0}]


def bowl_evaluator(params):
    """Analytic bowl (x-3)^2 + (y+1)^2 exposed as a raw merit key."""
    v = (params["x"] - 3.0) ** 2 + (params["y"] + 1.0) ** 2
    return {"detectors": {}, "merits": {"Det.bowl": v},
            "backend_used": "mock"}


# ---------------------------------------------------------------------------
# spec parsing (cli_specs owns the grammar)
# ---------------------------------------------------------------------------
def test_parse_var_spec():
    v = cli_specs.parse_var_spec("dim.lenspos:-4:-8:8")
    assert v == {"name": "dim.lenspos", "start": -4.0, "lo": -8.0,
                 "hi": 8.0}
    for bad in ("lenspos:0:5:-5",      # lo >= hi
                "lenspos:9:-5:5",      # start outside bounds
                "lenspos:0:1",         # wrong arity
                ":0:-1:1"):            # empty name
        with pytest.raises(Exception):
            cli_specs.parse_var_spec(bad)


def test_parse_operand_spec():
    o = cli_specs.parse_operand_spec("spot_rms@Body.Pad.Face5:0:2.5")
    assert o == {"operand": "spot_rms", "detector": "Body.Pad.Face5",
                 "target": 0.0, "weight": 2.5}
    o = cli_specs.parse_operand_spec("detected_power:0:1")
    assert o["operand"] == "detected_power" and o["detector"] is None
    # raw merit keys (contain a '.') pass through
    o = cli_specs.parse_operand_spec("Body003.Pad001.Face5.total_power_W:1e-3:1")
    assert o["operand"] == "Body003.Pad001.Face5.total_power_W"
    assert o["target"] == 1e-3
    with pytest.raises(Exception):
        cli_specs.parse_operand_spec("not_an_operand:0:1")


def test_operand_needs():
    assert operand_needs([{"operand": "spot_rms"}]) == {"export_rays"}
    assert operand_needs([{"operand": "focus"}]) == {"export_rays"}
    assert operand_needs([{"operand": "mtf50"},
                          {"operand": "encircled_energy"}]) == {
        "save_fields"}
    assert operand_needs([{"operand": "detected_power"}]) == set()


# ---------------------------------------------------------------------------
# operand extraction / merit math against synthetic report blocks
# ---------------------------------------------------------------------------
def _out(detectors=None, merits=None):
    return {"detectors": detectors or {}, "merits": merits or {}}


def test_spot_rms_combines_rows_n_weighted():
    out = _out(detectors={"D1.Pad.Face3": {"spot": [
        {"rms_radius_um": 10.0, "n_rays": 100},
        {"rms_radius_um": 20.0, "n_rays": 300},
    ]}})
    v, note = operand_value(out, {"operand": "spot_rms", "detector": None,
                                  "target": 0.0, "weight": 1.0})
    assert note is None
    expected = math.sqrt((100 * 10.0 ** 2 + 300 * 20.0 ** 2) / 400.0)
    assert v == pytest.approx(expected)


def test_operand_detector_qualifier_and_power_sum():
    dets = {"A.Pad.Face1": {"total_power_W": 1.0},
            "B.Pad.Face2": {"total_power_W": 3.0}}
    spec = {"operand": "detected_power", "detector": None,
            "target": 0.0, "weight": 1.0}
    v, _ = operand_value(_out(dets), spec)
    assert v == 4.0                       # unqualified sums detectors
    spec["detector"] = "B.Pad.Face2"      # exact label
    v, _ = operand_value(_out(dets), spec)
    assert v == 3.0
    spec["detector"] = "Face1"            # dotted-suffix match
    v, _ = operand_value(_out(dets), spec)
    assert v == 1.0
    spec["detector"] = "Face9"            # no match -> None + note
    v, note = operand_value(_out(dets), spec)
    assert v is None and "no detector matches" in note


def test_analysis_operands_read_field_block():
    dets = {"D.Pad.Face3": {"analysis": {"ee_r80_um": 12.5,
                                         "mtf50_tan_cy_mm": 88.0}}}
    v, _ = operand_value(_out(dets), {"operand": "encircled_energy",
                                      "detector": None, "target": 0.0,
                                      "weight": 1.0})
    assert v == 12.5
    v, _ = operand_value(_out(dets), {"operand": "mtf50", "detector": None,
                                      "target": 0.0, "weight": 1.0})
    assert v == 88.0


def test_merit_directions_and_targets():
    dets = {"D.Pad.Face3": {"total_power_W": 2.0,
                            "spot": [{"rms_radius_um": 5.0, "n_rays": 10}]}}
    # minimize spot toward 0: w*(v-t)^2
    m = MeritFunction([{"operand": "spot_rms", "detector": None,
                        "target": 0.0, "weight": 2.0}])
    merit, rows, missing = m.score(_out(dets))
    assert not missing and merit == pytest.approx(2.0 * 25.0)
    # pure maximize (target 0): -w*v
    m = MeritFunction([{"operand": "detected_power", "detector": None,
                        "target": 0.0, "weight": 3.0}])
    merit, _, _ = m.score(_out(dets))
    assert merit == pytest.approx(-6.0)
    # maximize toward a nonzero target: w*(v-t)^2 residual
    m = MeritFunction([{"operand": "detected_power", "detector": None,
                        "target": 5.0, "weight": 1.0}])
    merit, _, _ = m.score(_out(dets))
    assert merit == pytest.approx(9.0)


def test_merit_missing_operand_is_penalized():
    m = MeritFunction([{"operand": "spot_rms", "detector": None,
                        "target": 0.0, "weight": 1.0}])
    merit, rows, missing = m.score(_out(detectors={"D": {}}))
    assert merit == PENALTY and missing
    assert rows[0]["value"] is None


# ---------------------------------------------------------------------------
# THE CONVERGENCE ORACLES (mock bowl, both algorithms)
# ---------------------------------------------------------------------------
def test_local_converges_on_bowl():
    eng = OptimizationEngine(VARIABLES, RAW_OPERAND, bowl_evaluator,
                             algorithm="local", budget=200, tol=1e-12,
                             progress=False)
    best = eng.run()
    assert eng.n_evals <= 200
    assert best["params"]["x"] == pytest.approx(BOWL_MIN["x"], abs=1e-3)
    assert best["params"]["y"] == pytest.approx(BOWL_MIN["y"], abs=1e-3)
    assert best["merit"] < 1e-6
    # history is per-eval and starts at the START design
    assert len(eng.history) == eng.n_evals
    assert eng.history[0]["params"] == {"x": 8.0, "y": 6.0}
    # best-so-far is monotone non-increasing
    bests = [e["best_merit"] for e in eng.history]
    assert all(b2 <= b1 for b1, b2 in zip(bests, bests[1:]))


def test_global_cma_converges_on_bowl():
    eng = OptimizationEngine(VARIABLES, RAW_OPERAND, bowl_evaluator,
                             algorithm="global", budget=300, seed=42,
                             progress=False)
    best = eng.run()
    assert eng.n_evals <= 300
    assert best["params"]["x"] == pytest.approx(BOWL_MIN["x"], abs=0.05)
    assert best["params"]["y"] == pytest.approx(BOWL_MIN["y"], abs=0.05)
    assert best["merit"] < 1e-2


def test_global_cma_one_dimensional():
    """CMA must also work on a single variable (the auto_designed_lens
    case is 1-D)."""
    var = [{"name": "x", "start": 9.0, "lo": -10.0, "hi": 10.0}]

    def f(params):
        return {"detectors": {},
                "merits": {"Det.bowl": (params["x"] - 3.0) ** 2}}

    eng = OptimizationEngine(var, RAW_OPERAND, f, algorithm="global",
                             budget=120, seed=42, progress=False)
    best = eng.run()
    assert best["params"]["x"] == pytest.approx(3.0, abs=0.05)


def test_failed_evals_are_penalized_not_fatal():
    """An evaluator that raises over part of the domain (a trace failure
    region straddling the search path) must not crash the loop; the best
    must land in the valid region. Note the penalty is FLAT — a start
    deep inside the failure region gives the optimizer no gradient, so
    the start sits just inside the valid side of the boundary."""
    calls = {"n": 0, "failed": 0}

    def flaky(params):
        calls["n"] += 1
        if params["x"] < 0:
            calls["failed"] += 1
            raise RuntimeError("simulated trace failure")
        return bowl_evaluator(params)

    variables = [{"name": "x", "start": 0.5, "lo": -10.0, "hi": 10.0},
                 {"name": "y", "start": 0.0, "lo": -10.0, "hi": 10.0}]
    eng = OptimizationEngine(variables, RAW_OPERAND, flaky,
                             algorithm="global", budget=250, seed=7,
                             progress=False)
    best = eng.run()
    assert calls["failed"] > 0, "the failure region was never probed"
    assert best["params"]["x"] >= 0
    assert best["merit"] < 1.0
    penalized = [e for e in eng.history if e["penalized"]]
    assert penalized and all(e["merit"] == PENALTY for e in penalized)
    assert all("simulated trace failure" in e["note"] for e in penalized)


def test_budget_is_respected():
    eng = OptimizationEngine(VARIABLES, RAW_OPERAND, bowl_evaluator,
                             algorithm="local", budget=10, progress=False)
    eng.run()
    assert eng.n_evals <= 10


def test_bounds_are_respected():
    seen = []

    def spy(params):
        seen.append(dict(params))
        return bowl_evaluator(params)

    variables = [{"name": "x", "start": 4.9, "lo": 4.0, "hi": 5.0},
                 {"name": "y", "start": -1.0, "lo": -2.0, "hi": 0.0}]
    eng = OptimizationEngine(variables, RAW_OPERAND, spy,
                             algorithm="global", budget=150, seed=3,
                             progress=False)
    best = eng.run()
    for p in seen:
        assert 4.0 <= p["x"] <= 5.0 and -2.0 <= p["y"] <= 0.0
    # constrained minimum: x pinned to its lower bound, y free at -1
    assert best["params"]["x"] == pytest.approx(4.0, abs=0.1)
    assert best["params"]["y"] == pytest.approx(-1.0, abs=0.2)


def test_progress_events_emitted(monkeypatch, capsys):
    """MIEWB_PROGRESS=1 -> one '@MIEWB' line per evaluation with stage
    'optimize' and frac = evals/budget."""
    import common
    monkeypatch.setenv("MIEWB_PROGRESS", "1")
    eng = OptimizationEngine(VARIABLES, RAW_OPERAND, bowl_evaluator,
                             algorithm="local", budget=15)
    eng.run()
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith(common.PROGRESS_PREFIX)]
    assert len(lines) == eng.n_evals
    events = [common.parse_progress_line(ln) for ln in lines]
    assert all(ev["stage"] == "optimize" for ev in events)
    assert events[0]["eval"] == 1
    assert events[-1]["frac"] == pytest.approx(eng.n_evals / 15.0)
    assert "best" in events[-1] and "params" in events[-1]


def test_all_penalized_raises_optimize_error():
    def always_fail(params):
        raise RuntimeError("nope")

    eng = OptimizationEngine(VARIABLES, RAW_OPERAND, always_fail,
                             algorithm="local", budget=5, progress=False)
    with pytest.raises(OptimizeError, match="penalized"):
        eng.run()


def test_config_file_merging(tmp_path):
    """--config JSON fills defaults; explicit CLI flags win."""
    cfg = {"algorithm": "global", "budget": 77,
           "var": ["lenspos:0:-5:5"], "operand": ["spot_rms:0:1"]}
    cfg_path = tmp_path / "opt.json"
    cfg_path.write_text(json.dumps(cfg))
    parser = cli_specs.build_parser("optimize")
    args = parser.parse_args(["--model", "example.FCStd",
                              "--config", str(cfg_path),
                              "--budget", "12"])
    optimize._apply_config_file(parser, args)
    assert args.algorithm == "global"      # from config
    assert args.budget == 12               # CLI wins
    assert args.var == [{"name": "lenspos", "start": 0.0, "lo": -5.0,
                         "hi": 5.0}]
    assert args.operand[0]["operand"] == "spot_rms"
