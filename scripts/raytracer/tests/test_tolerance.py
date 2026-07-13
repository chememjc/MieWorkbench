# =============================================================================
# test_tolerance.py — the tolerancing engine's unit tests, against MOCK
# evaluators (analytic merits), so the sensitivity math and Monte-Carlo
# statistics are pinned WITHOUT a slow trace: the sensitivity ranking must
# match the known analytic gradient ordering, the yield fraction must match
# the analytic probability for a known distribution+threshold within MC
# error, failed evaluations must be penalized (never fatal), and the
# nested focus compensator must recover the merit per draw.
# =============================================================================
import json
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import cli_specs                                     # noqa: E402
import tolerance                                     # noqa: E402
from tolerance import (PENALTY, ToleranceEngine,     # noqa: E402
                       ToleranceError)

RAW_OPERAND = [{"operand": "Det.m", "detector": None,
                "target": 0.0, "weight": 1.0}]


def _tol(name, nominal, dist="normal", band=0.5):
    return {"name": name, "nominal": nominal, "dist": dist, "band": band}


def _merit_fn(f):
    """Wrap an analytic NON-NEGATIVE params->scalar into a fast_eval-
    shaped output such that the ENGINE'S SCORE equals f exactly:
    MeritFunction squares a raw merit key toward its target
    (weight*(v-0)^2), so the mock exposes sqrt(f) as the raw value."""
    def evaluate(params):
        v = f(params)
        assert v >= 0.0, "mock merits must be non-negative"
        return {"detectors": {}, "merits": {"Det.m": math.sqrt(v)},
                "backend_used": "mock"}
    return evaluate


# ---------------------------------------------------------------------------
# spec parsing (cli_specs owns the grammar)
# ---------------------------------------------------------------------------
def test_parse_tolerance_spec():
    t = cli_specs.parse_tolerance_spec("dim.lenspos:4:normal:0.25")
    assert t == {"name": "dim.lenspos", "nominal": 4.0, "dist": "normal",
                 "band": 0.25}
    t = cli_specs.parse_tolerance_spec("lensdy:0:UNIFORM:1.5")
    assert t["dist"] == "uniform" and t["band"] == 1.5
    for bad in ("lenspos:0:normal",          # wrong arity
                "lenspos:0:lognormal:1",     # unknown distribution
                "lenspos:0:normal:0",        # band <= 0
                "lenspos:0:normal:-1",
                ":0:normal:1",               # empty name
                "lenspos:x:normal:1"):       # non-numeric
        with pytest.raises(Exception):
            cli_specs.parse_tolerance_spec(bad)


def test_parse_compensator_spec():
    c = cli_specs.parse_compensator_spec("detpos:40:60")
    assert c == {"name": "detpos", "start": 50.0, "lo": 40.0, "hi": 60.0}
    c = cli_specs.parse_compensator_spec("detpos:50.236:40:60")
    assert c["start"] == 50.236
    for bad in ("detpos:60:40",              # lo >= hi
                "detpos:70:40:60",           # start outside bounds
                "detpos:40",                 # wrong arity
                ":40:60"):                   # empty name
        with pytest.raises(Exception):
            cli_specs.parse_compensator_spec(bad)


# ---------------------------------------------------------------------------
# engine configuration errors
# ---------------------------------------------------------------------------
def test_engine_rejects_bad_config():
    ev = _merit_fn(lambda p: 0.0)
    with pytest.raises(ToleranceError, match="at least one tolerance"):
        ToleranceEngine([], RAW_OPERAND, ev)
    with pytest.raises(ToleranceError, match="duplicate"):
        ToleranceEngine([_tol("a", 0), _tol("a", 1)], RAW_OPERAND, ev)
    with pytest.raises(ToleranceError, match="DIFFERENT variable"):
        ToleranceEngine([_tol("a", 0)], RAW_OPERAND, ev,
                        compensator={"name": "a", "start": 0.0,
                                     "lo": -1.0, "hi": 1.0})


# ---------------------------------------------------------------------------
# sensitivity: finite differences against a known analytic gradient
# ---------------------------------------------------------------------------
def test_sensitivity_ranking_matches_analytic_gradient_order():
    """M = 4a^2 + b^2 + 0.25c^2 at nominal (1,1,1): gradients (8, 2, 0.5)
    -> the ranked table must come out a > b > c, and the central
    differences of a quadratic are EXACT."""
    ev = _merit_fn(lambda p: (4.0 * p["a"] ** 2 + p["b"] ** 2
                              + 0.25 * p["c"] ** 2))
    tols = [_tol("b", 1.0), _tol("c", 1.0), _tol("a", 1.0)]  # scrambled
    eng = ToleranceEngine(tols, RAW_OPERAND, ev, draws=0, progress=False)
    rows = eng.run_sensitivity()
    assert [r["name"] for r in rows] == ["a", "b", "c"]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    # central difference of k*x^2 at x0 is exactly 2*k*x0
    by_name = {r["name"]: r for r in rows}
    assert by_name["a"]["derivative"] == pytest.approx(8.0)
    assert by_name["b"]["derivative"] == pytest.approx(2.0)
    assert by_name["c"]["derivative"] == pytest.approx(0.5)
    # impact = max over +/-band of |M - M0|; for k*x^2, band 0.5 at x0=1:
    # k*((1.5)^2 - 1) = 1.25k
    assert by_name["a"]["impact"] == pytest.approx(4.0 * 1.25)
    assert by_name["b"]["impact"] == pytest.approx(1.25)
    assert by_name["c"]["impact"] == pytest.approx(0.25 * 1.25)
    # 1 nominal + 2 evals per parameter
    assert eng.n_evals == 1 + 2 * 3


def test_sensitivity_at_merit_minimum_ranks_by_impact():
    """Toleranced AT the design's merit minimum (defocus at best focus,
    the standard case): the derivative vanishes but the band still costs
    merit — the impact ranking must still put the steep bowl first."""
    ev = _merit_fn(lambda p: 100.0 * p["x"] ** 2 + 1.0 * p["y"] ** 2)
    eng = ToleranceEngine([_tol("y", 0.0), _tol("x", 0.0)], RAW_OPERAND,
                          ev, progress=False)
    rows = eng.run_sensitivity()
    by_name = {r["name"]: r for r in rows}
    assert by_name["x"]["derivative"] == pytest.approx(0.0, abs=1e-9)
    assert by_name["x"]["impact"] == pytest.approx(100.0 * 0.25)
    assert [r["name"] for r in rows] == ["x", "y"]


def test_sensitivity_delta_scales_the_probe():
    ev = _merit_fn(lambda p: p["x"] ** 2)
    eng = ToleranceEngine([_tol("x", 0.0, band=1.0)], RAW_OPERAND, ev,
                          sens_delta=0.5, progress=False)
    rows = eng.run_sensitivity()
    assert rows[0]["delta"] == 0.5
    assert rows[0]["impact"] == pytest.approx(0.25)


def test_sensitivity_penalized_parameter_ranks_last():
    def f(p):
        if p["bad"] != 0.0:
            raise RuntimeError("simulated trace failure")
        return 3.0 * p["good"] ** 2

    eng = ToleranceEngine([_tol("bad", 0.0), _tol("good", 1.0)],
                          RAW_OPERAND, _merit_fn(f), progress=False)
    rows = eng.run_sensitivity()
    assert [r["name"] for r in rows] == ["good", "bad"]
    assert rows[1]["penalized"] and "simulated" in rows[1]["note"]
    assert rows[1]["derivative"] is None


# ---------------------------------------------------------------------------
# Monte-Carlo yield against analytic probabilities
# ---------------------------------------------------------------------------
N_MC = 4000     # MC error ~ sqrt(p(1-p)/N) <= 0.008; gates at ~4 sigma


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def test_mc_yield_matches_normal_cdf():
    """merit = x^2 with x ~ normal(0, 1): P(merit <= t) =
    P(|x| <= sqrt(t)) = 2*Phi(sqrt(t)) - 1 — the yield fraction must
    match the analytic probability within MC error."""
    ev = _merit_fn(lambda p: p["x"] ** 2)
    for t in (0.25, 1.0):
        eng = ToleranceEngine([_tol("x", 0.0, "normal", 1.0)],
                              RAW_OPERAND, ev, draws=N_MC, threshold=t,
                              mc_seed=123, progress=False)
        mc = eng.run_monte_carlo()
        expected = 2.0 * _phi(math.sqrt(t)) - 1.0
        assert mc["yield_fraction"] == pytest.approx(expected, abs=0.03)
        assert mc["n_pass"] == round(mc["yield_fraction"] * N_MC)


def test_mc_yield_matches_uniform_probability():
    """merit = x^2 with x ~ uniform(-2, +2): P(merit <= 1) =
    P(|x| <= 1) = 1/2 exactly, and every draw must stay inside the
    band."""
    seen = []

    def f(p):
        seen.append(p["x"])
        return p["x"] ** 2

    eng = ToleranceEngine([_tol("x", 0.0, "uniform", 2.0)], RAW_OPERAND,
                          _merit_fn(f), draws=N_MC, threshold=1.0,
                          mc_seed=7, progress=False)
    mc = eng.run_monte_carlo()
    assert mc["yield_fraction"] == pytest.approx(0.5, abs=0.03)
    draws = seen[1:]         # seen[0] is the nominal evaluation
    assert len(draws) == N_MC
    assert all(-2.0 <= x <= 2.0 for x in draws)


def test_mc_yield_drops_as_band_widens():
    """merit = x^2, threshold fixed: a wider tolerance band must lose
    yield (P(|x| <= 1) falls with sigma)."""
    ev = _merit_fn(lambda p: p["x"] ** 2)
    yields = []
    for band in (0.5, 1.0, 2.0):
        eng = ToleranceEngine([_tol("x", 0.0, "normal", band)],
                              RAW_OPERAND, ev, draws=N_MC, threshold=1.0,
                              mc_seed=99, progress=False)
        mc = eng.run_monte_carlo()
        # analytic: P(x^2 <= 1) = 2*Phi(1/band) - 1
        expected = 2.0 * _phi(1.0 / band) - 1.0
        assert mc["yield_fraction"] == pytest.approx(expected, abs=0.03)
        yields.append(mc["yield_fraction"])
    assert yields[0] > yields[1] > yields[2]


def test_mc_no_threshold_reports_distribution_only():
    """merit = x^2, x ~ N(0,1) -> chi^2_1: mean 1, std sqrt(2)."""
    ev = _merit_fn(lambda p: p["x"] ** 2)
    eng = ToleranceEngine([_tol("x", 0.0, "normal", 1.0)], RAW_OPERAND,
                          ev, draws=200, mc_seed=5, progress=False)
    mc = eng.run_monte_carlo()
    assert mc["yield_fraction"] is None and mc["n_pass"] is None
    assert mc["stats"]["n"] == 200
    assert mc["stats"]["mean"] == pytest.approx(1.0, abs=0.35)
    assert mc["stats"]["std"] == pytest.approx(math.sqrt(2.0), abs=0.5)
    assert mc["stats"]["p10"] < mc["stats"]["p50"] < mc["stats"]["p90"]


def test_mc_histogram_partitions_the_draws():
    ev = _merit_fn(lambda p: p["x"] ** 2)
    eng = ToleranceEngine([_tol("x", 0.0, "uniform", 1.0)], RAW_OPERAND,
                          ev, draws=500, hist_bins=16, mc_seed=11,
                          progress=False)
    mc = eng.run_monte_carlo()
    h = mc["histogram"]
    assert len(h["bin_edges"]) == 17 and len(h["counts"]) == 16
    assert sum(h["counts"]) == 500
    assert all(a < b for a, b in zip(h["bin_edges"], h["bin_edges"][1:]))
    assert h["bin_edges"][0] == pytest.approx(mc["stats"]["min"])
    assert h["bin_edges"][-1] == pytest.approx(mc["stats"]["max"])


def test_mc_failed_draws_penalized_never_fatal():
    """An evaluator that raises over part of the domain: penalized draws
    count against yield, and are excluded from stats/histogram."""
    def f(p):
        if p["x"] < 0.0:
            raise RuntimeError("simulated trace failure")
        return p["x"]

    eng = ToleranceEngine([_tol("x", 0.5, "normal", 1.0)], RAW_OPERAND,
                          _merit_fn(f), draws=400, threshold=1e9,
                          mc_seed=21, progress=False)
    mc = eng.run_monte_carlo()
    assert 0 < mc["n_penalized"] < 400
    # threshold is huge: every non-penalized draw passes, no penalized one
    assert mc["n_pass"] == 400 - mc["n_penalized"]
    assert mc["stats"]["n"] == 400 - mc["n_penalized"]
    assert sum(mc["histogram"]["counts"]) == mc["stats"]["n"]
    bad = [e for e in mc["detail"] if e["penalized"]]
    assert all(e["merit"] == PENALTY and not e["passed"] for e in bad)
    assert all("simulated trace failure" in e["note"] for e in bad)


def test_mc_draws_are_seed_deterministic():
    ev = _merit_fn(lambda p: p["x"] ** 2)
    def run(seed):
        eng = ToleranceEngine([_tol("x", 0.0, "normal", 1.0)],
                              RAW_OPERAND, ev, draws=50, mc_seed=seed,
                              progress=False)
        return [e["params"]["x"] for e in eng.run_monte_carlo()["detail"]]
    assert run(42) == run(42)
    assert run(42) != run(43)


def test_nominal_failure_is_fatal():
    def f(p):
        raise RuntimeError("nope")

    eng = ToleranceEngine([_tol("x", 0.0)], RAW_OPERAND, _merit_fn(f),
                          progress=False)
    with pytest.raises(ToleranceError, match="NOMINAL"):
        eng.run_nominal()


# ---------------------------------------------------------------------------
# the focus compensator (nested optimize engine per draw)
# ---------------------------------------------------------------------------
def test_compensator_recovers_yield():
    """merit = (x - c)^2: x is toleranced wide, c is the compensator.
    Uncompensated (c stuck at 0) most draws blow the threshold;
    compensated, the nested optimizer drives c -> x and every draw
    passes. This is the yield-recovery contract the demo oracle rechecks
    on a real lens."""
    ev = _merit_fn(lambda p: (p["x"] - p["c"]) ** 2)
    tols = [_tol("x", 0.0, "normal", 2.0)]
    threshold = 0.25

    eng_raw = ToleranceEngine(tols, RAW_OPERAND,
                              _merit_fn(lambda p: p["x"] ** 2),
                              draws=200, threshold=threshold, mc_seed=31,
                              progress=False)
    y_raw = eng_raw.run_monte_carlo()["yield_fraction"]
    # analytic: P(x^2 <= 0.25) = 2*Phi(0.5/2) - 1 ~ 0.197
    assert y_raw == pytest.approx(0.197, abs=0.06)

    eng_comp = ToleranceEngine(
        tols, RAW_OPERAND, ev, draws=40, threshold=threshold, mc_seed=31,
        compensator={"name": "c", "start": 0.0, "lo": -10.0, "hi": 10.0},
        comp_budget=30, progress=False)
    mc = eng_comp.run_monte_carlo()
    assert mc["yield_fraction"] > y_raw
    assert mc["yield_fraction"] >= 0.9
    # per-draw records carry the recovered compensator value ~ x
    for e in mc["detail"]:
        comp = e["compensator"]
        assert comp["name"] == "c" and comp["evals"] <= 30
        assert comp["value"] == pytest.approx(e["params"]["x"], abs=0.5)


def test_compensator_start_used_when_not_optimizing():
    """Nominal/sensitivity evaluations hold the compensator at START."""
    seen = []

    def f(p):
        seen.append(dict(p))
        return p["x"] ** 2

    eng = ToleranceEngine([_tol("x", 1.0)], RAW_OPERAND, _merit_fn(f),
                          compensator={"name": "c", "start": 7.5,
                                       "lo": 0.0, "hi": 10.0},
                          progress=False)
    eng.run_sensitivity()
    assert seen and all(p["c"] == 7.5 for p in seen)


def test_compensation_failure_is_penalized_not_fatal():
    calls = {"n": 0}

    def f(p):
        calls["n"] += 1
        raise RuntimeError("dead region")

    # nominal must pass, draws must fail: fail only when x != nominal
    def g(p):
        if p["x"] != 0.0:
            raise RuntimeError("dead region")
        return 0.0

    eng = ToleranceEngine([_tol("x", 0.0)], RAW_OPERAND, _merit_fn(g),
                          draws=5, threshold=1.0,
                          compensator={"name": "c", "start": 0.0,
                                       "lo": -1.0, "hi": 1.0},
                          comp_budget=4, progress=False)
    mc = eng.run_monte_carlo()
    assert mc["n_penalized"] == 5 and mc["yield_fraction"] == 0.0
    assert all("compensation failed" in e["note"] for e in mc["detail"])


# ---------------------------------------------------------------------------
# progress + config plumbing
# ---------------------------------------------------------------------------
def test_progress_events_contract(monkeypatch, capsys):
    """MIEWB_PROGRESS=1 -> stage-'tolerance' events: per-sensitivity
    parameter, one sensitivity_done carrying the compact ranked table
    (what the GUI bar chart renders), and per-draw events with
    frac = draw/draws (what the GUI histogram consumes)."""
    import common
    monkeypatch.setenv("MIEWB_PROGRESS", "1")
    ev = _merit_fn(lambda p: 4.0 * p["a"] ** 2 + p["b"] ** 2)
    eng = ToleranceEngine([_tol("a", 1.0), _tol("b", 1.0)], RAW_OPERAND,
                          ev, draws=10, threshold=100.0, mc_seed=1)
    eng.run_sensitivity()
    eng.run_monte_carlo()
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith(common.PROGRESS_PREFIX)]
    events = [common.parse_progress_line(ln) for ln in lines]
    assert all(e["stage"] == "tolerance" for e in events)

    sens = [e for e in events if e.get("phase") == "sensitivity"]
    assert [e["param"] for e in sens] == ["a", "b"]
    done = [e for e in events if e.get("phase") == "sensitivity_done"]
    assert len(done) == 1
    table = done[0]["sensitivity"]
    assert [r["name"] for r in table] == ["a", "b"]
    assert table[0]["rank"] == 1 and table[0]["impact"] > table[1]["impact"]

    draws = [e for e in events if e.get("phase") == "mc"]
    assert len(draws) == 10
    for e in draws:
        assert e["frac"] == pytest.approx(e["draw"] / 10.0)
        assert "merit" in e and "merit_yield" in e and "params" in e
    assert draws[-1]["merit_yield"] == pytest.approx(1.0)  # threshold 100


def test_config_file_merging(tmp_path):
    """--config JSON fills defaults; explicit CLI flags win; spec strings
    are parsed."""
    cfg = {"draws": 77, "mc_seed": 9,
           "tolerance": ["lenspos:0:normal:0.5"],
           "operand": ["spot_rms:0:1"],
           "compensator": "detpos:40:60"}
    cfg_path = tmp_path / "tol.json"
    cfg_path.write_text(json.dumps(cfg))
    parser = cli_specs.build_parser("tolerance")
    args = parser.parse_args(["--model", "example.FCStd",
                              "--config", str(cfg_path),
                              "--draws", "12"])
    tolerance._apply_config_file(parser, args)
    assert args.draws == 12                # CLI wins
    assert args.mc_seed == 9               # from config
    assert args.tolerance == [{"name": "lenspos", "nominal": 0.0,
                               "dist": "normal", "band": 0.5}]
    assert args.compensator["name"] == "detpos"
    assert args.operand[0]["operand"] == "spot_rms"
