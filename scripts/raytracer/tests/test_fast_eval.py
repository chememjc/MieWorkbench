# =============================================================================
# test_fast_eval.py — the fast evaluator's PARITY ORACLE + fault injection.
#
# The whole point of scripts/fast_eval.py's worker backend (persistent
# FreeCAD + in-place extraction + fingerprint face cache) is that it is
# INDISTINGUISHABLE from the reference full pipeline (permute_model.py ->
# extract_geometry.py -> run_trace.py -> post_process.py). This module
# pins that claim:
#
#   * parity: for several distinct param dicts on example.FCStd, the
#     worker backend's extracted model.json equals the full backend's
#     (numeric comparison — bit-parity is NOT attainable even full-vs-full
#     because OCC's recompute is ULP-nondeterministic run to run, measured
#     at ~1e-16 relative / ~1e-18 m absolute on this machine; the oracle
#     therefore gates at atol 1e-12 / rtol 1e-9, six orders of magnitude
#     above the noise and far below any real geometry change), and the
#     same-seed merits match.
#   * fault injection: SIGKILLing the persistent worker between evals and
#     mid-conversation both recover (relaunch + re-open + replay params)
#     and the next evaluate() still returns the correct merit.
#   * determinism/caching: identical params twice -> identical merits,
#     with the face cache hitting every body; a changed param invalidates
#     (only) the affected bodies.
#   * fallback: when the worker path is unavailable, evaluate() falls
#     back to the full path for that eval instead of raising.
#
# Cost: one module-scoped "session" runs everything once (~4-6 min: the
# full backend pays 2 FreeCAD AppImage launches per eval and each -c
# launch executes its script twice); the tests then assert on the
# collected artifacts. Ray counts are tiny (5k rays, 128 px) — the merits
# only need to be DETERMINISTIC, not converged.
# =============================================================================
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import common                                    # noqa: E402
import fast_eval                                 # noqa: E402
from fast_eval import Evaluator, EvalError       # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(common.FREECAD_APPIMAGE),
    reason="FreeCAD AppImage not available (%s)" % common.FREECAD_APPIMAGE)

MODEL = "example.FCStd"
PARAM_NAMES = ["lenspos", "sphered"]
EVALS = [
    {"lenspos": -3.0, "sphered": 24.0},
    {"lenspos": 0.0, "sphered": 30.0},
    {"lenspos": 4.0, "sphered": 27.0},
]
FIDELITY = dict(rays=5000, resolution=128, nlambda=3, spectral_bins=8,
                seeds=1, seed0=42)

# numeric parity gates (see header): geometry floats live in metres/m^2/m^3
GEOM_ATOL = 1e-12
GEOM_RTOL = 1e-9
# mesh_area_m2 comes from the OCC tessellator, which is chaotically
# sensitive to ULP-level BRep perturbations (triangulation choices can
# flip); physics on analytic faces never reads it, so it gets its own gate
MESH_AREA_RTOL = 1e-3
MERIT_RTOL = 1e-6
MERIT_ATOL = 1e-12


# ---------------------------------------------------------------------------
# the one expensive session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def session():
    root = Path(tempfile.mkdtemp(
        prefix="fasteval-test-",
        dir=str(common.PROJECT_DIR / "var")))
    s = {"root": root, "full": {}, "worker": {}, "t_full": [],
         "t_worker": []}
    try:
        # ---- reference: the full backend --------------------------------
        with Evaluator(MODEL, params=PARAM_NAMES, backend="full",
                       workdir=root / "full", **FIDELITY) as ev_full:
            for pd in EVALS:
                t0 = time.monotonic()
                out = ev_full.evaluate(pd)
                s["t_full"].append(time.monotonic() - t0)
                s["full"][out["variant"]] = out

        # ---- fast path: the worker backend ------------------------------
        ev = Evaluator(MODEL, params=PARAM_NAMES, backend="worker",
                       workdir=root / "worker", **FIDELITY)
        with ev:
            for pd in EVALS:
                t0 = time.monotonic()
                out = ev.evaluate(pd)
                s["t_worker"].append(time.monotonic() - t0)
                s["worker"][out["variant"]] = out

            # determinism + cache: repeat the last params verbatim
            s["repeat"] = ev.evaluate(EVALS[-1])
            # cache invalidation: move one param only
            changed = dict(EVALS[-1], lenspos=EVALS[-1]["lenspos"] + 1.0)
            s["changed"] = ev.evaluate(changed)

            # ---- fault injection 1: SIGKILL between evals ----------------
            pid = ev.worker_pid()
            assert pid, "worker should be alive here"
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
            s["after_kill"] = ev.evaluate(EVALS[0])

            # ---- fault injection 2: SIGKILL mid-conversation -------------
            # (no _ensure_worker in between: the raw request must detect
            # the death, relaunch, re-open the doc, replay the cumulative
            # params, and then succeed)
            pid2 = ev.worker_pid()
            assert pid2 and pid2 != pid
            os.kill(pid2, signal.SIGKILL)
            time.sleep(0.5)
            structure = ev._client.request("get_structure",
                                           {"doc": ev._doc})
            s["recoveries_after_midkill"] = ev._client.n_recoveries
            s["structure_after_midkill"] = structure
            s["applied_after_midkill"] = dict(ev._applied)

            # ---- fallback: worker path unavailable -> full path ----------
            ev._teardown_worker()
            real_start = fast_eval._WorkerClient.start

            def broken_start(self):
                raise fast_eval.FcDead("simulated: worker cannot launch")

            fast_eval._WorkerClient.start = broken_start
            try:
                s["fallback"] = ev.evaluate(EVALS[1])
            finally:
                fast_eval._WorkerClient.start = real_start

        yield s
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------
def _num_close(a, b, rtol, atol):
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def _assert_tree_equal(a, b, path, failures):
    """Structural equality with numeric tolerance (see gates above)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            failures.append("%s: key sets differ: %s vs %s"
                            % (path, sorted(a), sorted(b)))
            return
        for k in a:
            _assert_tree_equal(a[k], b[k], "%s.%s" % (path, k), failures)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            failures.append("%s: list lengths differ: %d vs %d"
                            % (path, len(a), len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_tree_equal(x, y, "%s[%d]" % (path, i), failures)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            failures.append("%s: %r != %r" % (path, a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        rtol = (MESH_AREA_RTOL if path.endswith("mesh_area_m2")
                else GEOM_RTOL)
        if not _num_close(float(a), float(b), rtol, GEOM_ATOL):
            failures.append("%s: %.17g != %.17g (|d|=%.3g)"
                            % (path, a, b, abs(a - b)))
    else:
        if a != b:
            failures.append("%s: %r != %r" % (path, a, b))


def _load_model(out):
    with open(out["model_json"]) as fh:
        return json.load(fh)


def _merit_diff(m_a, m_b, rtol=MERIT_RTOL, atol=MERIT_ATOL):
    assert set(m_a) == set(m_b), (
        "merit key sets differ: only-in-a=%s only-in-b=%s"
        % (sorted(set(m_a) - set(m_b)), sorted(set(m_b) - set(m_a))))
    bad = {k: (m_a[k], m_b[k]) for k in m_a
           if not _num_close(m_a[k], m_b[k], rtol, atol)}
    return bad


# ---------------------------------------------------------------------------
# THE PARITY ORACLE
# ---------------------------------------------------------------------------
def test_parity_geometry(session):
    """worker-extracted model.json == full-extracted model.json (numeric
    gate; source_fcstd is provenance — the worker never writes a variant
    file — and is excluded)."""
    assert set(session["worker"]) == set(session["full"])
    for variant in session["full"]:
        a = _load_model(session["worker"][variant])
        b = _load_model(session["full"][variant])
        a.pop("source_fcstd"), b.pop("source_fcstd")
        failures = []
        _assert_tree_equal(a, b, variant, failures)
        assert not failures, (
            "geometry parity failed for %s (%d diffs):\n%s"
            % (variant, len(failures), "\n".join(failures[:20])))


def test_parity_merits(session):
    """Same params, same seed -> the worker backend's merit scalars equal
    the reference full backend's."""
    for variant in session["full"]:
        m_w = session["worker"][variant]["merits"]
        m_f = session["full"][variant]["merits"]
        assert m_w, "no merits extracted for %s" % variant
        bad = _merit_diff(m_w, m_f)
        assert not bad, ("merit parity failed for %s: %s" % (variant, bad))
        # and the closure ledger held on both paths
        assert session["worker"][variant]["closure_ok"]
        assert session["full"][variant]["closure_ok"]


def test_parity_used_the_fast_path(session):
    """The parity evals above must actually have exercised the worker path
    (a silent fallback to 'full' would make the oracle vacuous)."""
    for variant, out in session["worker"].items():
        assert out["backend_used"] == "worker", (variant, out["backend_used"])


# ---------------------------------------------------------------------------
# determinism + fingerprint cache
# ---------------------------------------------------------------------------
def test_determinism_same_params_identical_merits(session):
    variant = session["repeat"]["variant"]
    first = session["worker"][variant]["merits"]
    again = session["repeat"]["merits"]
    assert first == again, (
        "identical params must reproduce identical merits (the face cache "
        "replays the unchanged geometry bit-for-bit within one worker "
        "session); diffs: %s" % _merit_diff(first, again, 0.0, 0.0))


def test_cache_hits_on_unchanged_body(session):
    cache = session["repeat"]["cache"]
    assert cache["misses"] == [], (
        "an unchanged re-eval must hit the cache for every body: %s" % cache)
    assert len(cache["hits"]) >= 5   # example.FCStd has 7 traced bodies


def test_cache_invalidated_on_changed_body(session):
    cache = session["changed"]["cache"]
    assert cache["misses"], (
        "changing lenspos must invalidate at least the moved body: %s"
        % cache)
    assert cache["hits"], (
        "changing lenspos must NOT invalidate every body — the "
        "fingerprint cache would be useless: %s" % cache)
    assert not (set(cache["hits"]) & set(cache["misses"]))
    # the changed merit must actually differ from the unchanged one
    # (guards against a cache so aggressive it returns stale geometry)
    assert session["changed"]["merits"] != session["repeat"]["merits"]


# ---------------------------------------------------------------------------
# fault injection / recovery
# ---------------------------------------------------------------------------
def test_recovery_after_kill_between_evals(session):
    out = session["after_kill"]
    assert out["backend_used"] == "worker", (
        "the eval after a SIGKILL must run on the RELAUNCHED worker, "
        "not silently fall back (got %r)" % out["backend_used"])
    # ... and produce the correct merit (vs the full-backend reference)
    ref = session["full"][out["variant"]]["merits"]
    bad = _merit_diff(out["merits"], ref)
    assert not bad, ("post-recovery merits differ from the reference: %s"
                     % bad)


def test_recovery_mid_conversation_replays_state(session):
    assert session["recoveries_after_midkill"] >= 1, (
        "the raw request after a mid-conversation SIGKILL must have gone "
        "through FcClient relaunch-and-replay")
    # the replayed document must carry the cumulative parameter state:
    # the spreadsheet cells hold exactly what apply_params last wrote
    structure = session["structure_after_midkill"]
    applied = session["applied_after_midkill"]
    dim = next(sh for sh in structure["sheets"] if sh["label"] == "dim")
    for name, value in applied.items():
        raw = dim["aliases"][name]["raw"]
        assert raw == "=%.10g mm" % value, (
            "param %s not replayed after recovery: cell raw is %r, "
            "expected %r" % (name, raw, "=%.10g mm" % value))


def test_fallback_to_full_when_worker_unavailable(session):
    out = session["fallback"]
    assert out["backend_used"] == "full-fallback"
    assert "simulated" in out["fallback_reason"]
    ref = session["full"][out["variant"]]["merits"]
    bad = _merit_diff(out["merits"], ref)
    assert not bad, ("fallback merits differ from the reference: %s" % bad)


# ---------------------------------------------------------------------------
# the point of the exercise
# ---------------------------------------------------------------------------
def test_worker_is_faster_than_full(session):
    t_full = sorted(session["t_full"])[len(session["t_full"]) // 2]
    # exclude the first worker eval (it pays the one-time worker launch)
    warm = session["t_worker"][1:]
    t_worker = sum(warm) / len(warm)
    print("\nfast_eval timing: full=%.1fs/eval (median), "
          "worker=%.1fs/eval (warm mean) -> speedup %.1fx"
          % (t_full, t_worker, t_full / t_worker))
    assert t_worker < t_full, (
        "the fast path (%.1fs) must beat the full pipeline (%.1fs)"
        % (t_worker, t_full))


def test_param_vector_is_locked(session):
    """Partial/mismatched param dicts are rejected loudly (a partial
    vector would silently inherit the previous eval's leftovers on the
    worker document and diverge from the full backend)."""
    root = session["root"]
    ev = Evaluator(MODEL, params=PARAM_NAMES, backend="full",
                   workdir=root / "lockcheck", **FIDELITY)
    with pytest.raises(ValueError, match="locked parameter set"):
        ev.evaluate({"lenspos": 1.0})
    with pytest.raises(ValueError, match="non-empty"):
        ev.evaluate({})
