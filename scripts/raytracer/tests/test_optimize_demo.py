# =============================================================================
# test_optimize_demo.py — the auto_designed_lens PHYSICS ORACLE for
# scripts/optimize.py, end to end through the real CLI (optics-env python
# -> fast_eval worker backend -> FreeCAD -> trace -> post -> report.json).
#
# The scene (basemodels/auto_designed_lens.FCStd, built by
# make_test_scenes.py): the lens_pcx BK7 singlet (EFL 48.536, BFL 45.236
# at 633nm) whose body Placement.Base.x is expression-bound to the 'dim'
# sheet's 'lenspos' alias; a collimated Ø10 source; a detector fixed 4mm
# PAST the lenspos=0 focal plane. Collimated input means the focus
# translates 1:1 with the lens, so the spot-minimizing position is
# lenspos = +4.0mm exactly (paraxial). The optimizer starts DEFOCUSED at
# lenspos=-6 (10mm of defocus -> ~800um power-weighted spot RMS) and must
# drive it near +4 (~300um, the singlet's spherical-aberration floor).
#
# Gates: >=2x spot-RMS reduction (measured 2.7x), found lenspos within
# 1mm of the paraxial focus (the RMS-vs-defocus bowl is ~10um/0.25mm flat
# near the bottom at this ray budget), and the '@MIEWB' stage-"optimize"
# progress contract the GUI plot consumes.
#
# Cost: ONE optimization run (~1-2 min: worker launch + ~12 quick
# incoherent evals at 5k rays/128px + the final coherent re-eval),
# module-scoped; the tests assert on the collected report.
# =============================================================================
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import common                                    # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(common.FREECAD_APPIMAGE),
    reason="FreeCAD AppImage not available (%s)" % common.FREECAD_APPIMAGE)

MODEL = "auto_designed_lens.FCStd"
START_LENSPOS = -6.0
TRUE_FOCUS_LENSPOS = 4.0        # paraxial: detector 4mm past lenspos=0 focus
BUDGET = 16


@pytest.fixture(scope="module")
def session():
    model = common.BASEMODELS_DIR / MODEL
    assert model.exists(), (
        "%s missing — build it with: FreeCAD.AppImage -c "
        "scripts/make_test_scenes.py -- --scene auto_designed_lens "
        "--outdir basemodels" % model)
    root = Path(tempfile.mkdtemp(prefix="optdemo-test-",
                                 dir=str(common.PROJECT_DIR / "var")))
    out_dir = root / "case"
    try:
        env = dict(os.environ, MIEWB_PROGRESS="1")
        proc = subprocess.run(
            [common.OPTICS_PYTHON, str(SCRIPTS / "optimize.py"),
             "--model", str(model),
             "--var", "lenspos:%g:-8:8" % START_LENSPOS,
             "--operand", "spot_rms:0:1",
             "--algorithm", "local", "--budget", str(BUDGET),
             "--rays", "5000", "--resolution", "128", "--nlambda", "3",
             "--out", str(out_dir), "--workdir", str(root / "fasteval")],
            env=env, capture_output=True, text=True, timeout=1200)
        assert proc.returncode == 0, (
            "optimize.py failed (exit %d):\n%s\n%s"
            % (proc.returncode, proc.stdout[-4000:], proc.stderr[-2000:]))
        with open(out_dir / "report.json") as fh:
            report = json.load(fh)
        yield {"report": report, "stdout": proc.stdout,
               "out_dir": out_dir}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _rms_um(entry):
    """spot_rms:0:1 merit is weight*(rms-0)^2 -> rms [um]."""
    return math.sqrt(entry["merit"])


def test_optimizer_reduces_spot_rms_at_least_2x(session):
    report = session["report"]
    history = report["history"]
    assert history[0]["params"]["lenspos"] == pytest.approx(START_LENSPOS)
    rms_start = _rms_um(history[0])
    rms_best = _rms_um(report["best"])
    assert rms_best > 0
    assert rms_start / rms_best >= 2.0, (
        "expected >=2x spot-RMS reduction, got %.2fx (start %.1f um -> "
        "best %.1f um)" % (rms_start / rms_best, rms_start, rms_best))
    # sanity scale: the singlet's aberration floor is a few hundred um,
    # the defocused start high hundreds
    assert 100.0 < rms_best < 500.0
    assert rms_start > 600.0


def test_optimizer_finds_the_focus(session):
    best = session["report"]["best"]
    found = best["params"]["lenspos"]
    assert abs(found - TRUE_FOCUS_LENSPOS) <= 1.0, (
        "optimized lenspos %.3f is not near the true focus %.1f"
        % (found, TRUE_FOCUS_LENSPOS))


def test_report_contract(session):
    report = session["report"]
    assert report["status"] == "completed"
    assert report["algorithm"] == "local"
    assert 1 <= report["n_evals"] <= BUDGET
    assert len(report["history"]) == report["n_evals"]
    # per-eval records carry params + merit + operand rows
    for e in report["history"]:
        assert set(e["params"]) == {"lenspos"}
        assert -8.0 <= e["params"]["lenspos"] <= 8.0
        assert e["operands"][0]["operand"] == "spot_rms"
        assert e["backend_used"] in ("worker", "full", "full-fallback")
    # best-so-far is monotone non-increasing
    bests = [e["best_merit"] for e in report["history"]]
    assert all(b2 <= b1 for b1, b2 in zip(bests, bests[1:]))


def test_final_coherent_reevaluation(session):
    """The best design is re-evaluated with coherence as authored; this
    scene's source is authored coherent=False, so the number must agree
    with the inner loop's (up to OCC recompute noise on re-extract)."""
    fc = session["report"].get("final_coherent")
    assert fc is not None and "merit" in fc, fc
    assert fc["merit"] == pytest.approx(session["report"]["best"]["merit"],
                                        rel=1e-3)


def test_progress_stream_contract(session):
    """The '@MIEWB' lines the GUI convergence plot consumes: stage
    'optimize', one event per eval with eval/merit/best/params extras,
    frac = evals/budget, and a final status=completed event."""
    events = [common.parse_progress_line(ln)
              for ln in session["stdout"].splitlines()
              if ln.startswith(common.PROGRESS_PREFIX)]
    events = [e for e in events if e]
    assert events and all(e["stage"] == "optimize" for e in events)
    per_eval = [e for e in events if "eval" in e]
    assert len(per_eval) == session["report"]["n_evals"]
    for e in per_eval:
        assert e["frac"] == pytest.approx(e["eval"] / BUDGET)
        assert "merit" in e and "best" in e and "params" in e
    assert events[-1]["status"] == "completed"
