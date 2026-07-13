# =============================================================================
# test_tolerance_demo.py — the tolerance_lens PHYSICS ORACLE for
# scripts/tolerance.py, end to end through the real CLI (optics-env python
# -> fast_eval worker backend -> FreeCAD -> trace -> post -> report.json).
#
# The scene (basemodels/tolerance_lens.FCStd, built by make_test_scenes.py):
# the auto_designed_lens BK7 singlet (EFL 48.536 at 633nm) with THREE
# spreadsheet-driven degrees of freedom — dim.lenspos (lens axial
# position), dim.lensdy (lens transverse decenter) and dim.detpos
# (detector axial position, nominally AT the lenspos=0 focal plane
# x=50.236). Collimated Ø10 input, coherent=False (geometric spot). The
# nominal design is IN FOCUS: spot RMS sits at the singlet's
# spherical-aberration floor (~300um power-weighted).
#
# Physics the oracle pins (three CLI runs, one module-scoped fixture):
#
#   A  sensitivity + narrow MC   lenspos/lensdy both normal(0, 1mm).
#      Axial position errors defocus 1:1 (collimated input) and cost
#      merit QUADRATICALLY; equal-size decenter errors mostly TRANSLATE
#      the spot (RMS about the centroid is first-order insensitive), so
#      the ranked sensitivity table must put lenspos FIRST by a clear
#      factor (measured 6188 vs 1347 merit impact at +/-1mm). At the
#      1mm-band fabrication quality every draw stays near the floor:
#      yield 1.0 against the shared threshold.
#
#   B  wide MC                   lenspos uniform(+/-4mm), no compensator.
#      Pure defocus draws ride the RMS-vs-defocus bowl; at +/-4mm several
#      draws blow the threshold: the yield DROPS below the narrow run's
#      (measured 4/6 = 0.667) while staying a plausible fraction in
#      (0, 1) — wider tolerances -> lower yield.
#
#   C  wide MC + compensator     run B plus --compensator detpos:40:62.
#      Refocusing the detector is the classic focus compensator: the
#      nested per-draw optimization must drive detpos to track the
#      perturbed focus (detpos ~= 50.236 + lenspos) and recover every
#      draw's merit to ~ the nominal floor: yield returns to 1.0 >
#      run B's.
#
# Cost: three tolerance.py runs sharing a fast_eval workdir (~4-5 min
# total: 11 + 7 + ~40 quick incoherent evals at 5k rays/128px), module-
# scoped; the tests assert on the collected reports.
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

MODEL = "tolerance_lens.FCStd"
FOCAL_DETPOS = 50.236        # detector nominal = the lenspos=0 focal plane
THRESHOLD = 110000.0         # merit = rms^2 [um^2]: rms <= ~332um passes
DRAWS = 6
SEED = 42
NARROW_BAND = 1.0            # run A: normal sigma [mm]
WIDE_BAND = 4.0              # runs B/C: uniform half-width [mm]


def _run(model, out_dir, workdir, extra):
    env = dict(os.environ, MIEWB_PROGRESS="1")
    proc = subprocess.run(
        [common.OPTICS_PYTHON, str(SCRIPTS / "tolerance.py"),
         "--model", str(model),
         "--operand", "spot_rms:0:1",
         "--draws", str(DRAWS), "--merit-threshold", str(THRESHOLD),
         "--mc-seed", str(SEED),
         "--rays", "5000", "--resolution", "128", "--nlambda", "3",
         "--out", str(out_dir), "--workdir", str(workdir)] + extra,
        env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, (
        "tolerance.py failed (exit %d):\n%s\n%s"
        % (proc.returncode, proc.stdout[-4000:], proc.stderr[-2000:]))
    with open(out_dir / "report.json") as fh:
        report = json.load(fh)
    return report, proc.stdout


@pytest.fixture(scope="module")
def session():
    model = common.BASEMODELS_DIR / MODEL
    assert model.exists(), (
        "%s missing — build it with: FreeCAD.AppImage -c "
        "scripts/make_test_scenes.py -- --scene tolerance_lens "
        "--outdir basemodels" % model)
    root = Path(tempfile.mkdtemp(prefix="toldemo-test-",
                                 dir=str(common.PROJECT_DIR / "var")))
    try:
        workdir = root / "fasteval"
        narrow, narrow_stdout = _run(
            model, root / "narrow", workdir,
            ["--tolerance", "lenspos:0:normal:%g" % NARROW_BAND,
             "--tolerance", "lensdy:0:normal:%g" % NARROW_BAND])
        wide, _ = _run(
            model, root / "wide", workdir,
            ["--tolerance", "lenspos:0:uniform:%g" % WIDE_BAND,
             "--skip-sensitivity"])
        comp, _ = _run(
            model, root / "comp", workdir,
            ["--tolerance", "lenspos:0:uniform:%g" % WIDE_BAND,
             "--skip-sensitivity",
             "--compensator", "detpos:%g:40:62" % FOCAL_DETPOS,
             "--comp-budget", "8"])
        yield {"narrow": narrow, "wide": wide, "comp": comp,
               "narrow_stdout": narrow_stdout}
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# nominal design
# ---------------------------------------------------------------------------
def test_nominal_design_is_at_the_aberration_floor(session):
    """In-focus singlet: nominal power-weighted spot RMS ~ 300um (the
    spherical-aberration floor), identical across the three runs."""
    for tag in ("narrow", "wide", "comp"):
        nominal = session[tag]["nominal"]
        assert not nominal["penalized"]
        rms = math.sqrt(nominal["merit"])
        assert 250.0 < rms < 350.0, "%s: nominal rms %.1f um" % (tag, rms)


# ---------------------------------------------------------------------------
# (a) sensitivity ranking is physically sensible
# ---------------------------------------------------------------------------
def test_sensitivity_ranks_defocus_over_decenter(session):
    """Axial (defocus) errors dominate equal-size decenter errors on the
    spot-RMS merit: lenspos must rank first by a clear factor."""
    rows = session["narrow"]["sensitivity"]
    assert [r["name"] for r in rows] == ["lenspos", "lensdy"]
    assert rows[0]["rank"] == 1 and not rows[0]["penalized"]
    assert not rows[1]["penalized"]
    assert rows[0]["impact"] >= 2.0 * rows[1]["impact"], (
        "expected lenspos to dominate: impacts %.1f vs %.1f"
        % (rows[0]["impact"], rows[1]["impact"]))
    # the finite differences actually ran: both probes evaluated
    for row in rows:
        assert row["merit_plus"] is not None
        assert row["merit_minus"] is not None
        assert row["derivative"] is not None
        assert math.isfinite(row["derivative"])


# ---------------------------------------------------------------------------
# (b) Monte-Carlo yield: plausible, and DROPS as the band widens
# ---------------------------------------------------------------------------
def test_yield_drops_as_the_tolerance_band_widens(session):
    y_narrow = session["narrow"]["mc"]["yield_fraction"]
    y_wide = session["wide"]["mc"]["yield_fraction"]
    assert y_narrow >= 0.8, "narrow band should almost always pass"
    assert 0.0 < y_wide < 1.0, (
        "wide band should straddle the threshold (got %g)" % y_wide)
    assert y_wide <= y_narrow - 0.2, (
        "widening the tolerance band must cost yield: narrow %.3f -> "
        "wide %.3f" % (y_narrow, y_wide))
    # the failing wide draws are the LARGE defocus ones (bowl physics)
    detail = session["wide"]["mc"]["detail"]
    failed = [abs(e["params"]["lenspos"]) for e in detail
              if not e["passed"]]
    passed = [abs(e["params"]["lenspos"]) for e in detail if e["passed"]]
    assert failed and passed
    assert min(failed) > min(passed)


# ---------------------------------------------------------------------------
# (c) the focus compensator recovers the yield
# ---------------------------------------------------------------------------
def test_compensator_recovers_yield(session):
    y_wide = session["wide"]["mc"]["yield_fraction"]
    y_comp = session["comp"]["mc"]["yield_fraction"]
    assert y_comp > y_wide
    assert y_comp >= 0.99, (
        "refocusing the detector should recover every draw (got %.3f)"
        % y_comp)
    # compensated merits collapse back to ~ the nominal floor
    nominal = session["comp"]["nominal"]["merit"]
    stats = session["comp"]["mc"]["stats"]
    assert stats["p90"] <= 1.10 * nominal


def test_compensator_tracks_the_perturbed_focus(session):
    """Collimated input: the focus translates 1:1 with the lens, so the
    recovered detector position must be ~ FOCAL_DETPOS + lenspos."""
    for e in session["comp"]["mc"]["detail"]:
        comp = e["compensator"]
        assert comp["name"] == "detpos" and comp["evals"] <= 8
        ideal = FOCAL_DETPOS + e["params"]["lenspos"]
        assert abs(comp["value"] - ideal) <= 1.5, (
            "draw %d: detpos %.3f vs ideal %.3f (lenspos %.3f)"
            % (e["draw"], comp["value"], ideal, e["params"]["lenspos"]))


# ---------------------------------------------------------------------------
# report + progress contracts (what the GUI consumes)
# ---------------------------------------------------------------------------
def test_report_contract(session):
    for tag in ("narrow", "wide", "comp"):
        report = session[tag]
        assert report["status"] == "completed"
        mc = report["mc"]
        assert mc["draws"] == DRAWS and len(mc["detail"]) == DRAWS
        assert mc["n_penalized"] == 0
        assert mc["threshold"] == THRESHOLD
        assert mc["n_pass"] == round(mc["yield_fraction"] * DRAWS)
        h = mc["histogram"]
        assert sum(h["counts"]) == DRAWS
        assert len(h["bin_edges"]) == len(h["counts"]) + 1
        stats = mc["stats"]
        assert stats["n"] == DRAWS
        assert stats["min"] <= stats["p50"] <= stats["max"]
    assert session["comp"]["mc"]["compensated"] is True
    assert session["wide"]["mc"]["compensated"] is False


def test_progress_stream_contract(session):
    """The '@MIEWB' lines the GUI consumes: stage 'tolerance', one
    sensitivity_done event carrying the compact ranked table, per-draw
    events with frac = draw/draws + merit/yield extras, and a final
    status=completed event."""
    events = [common.parse_progress_line(ln)
              for ln in session["narrow_stdout"].splitlines()
              if ln.startswith(common.PROGRESS_PREFIX)]
    events = [e for e in events if e]
    assert events and all(e["stage"] == "tolerance" for e in events)
    done = [e for e in events if e.get("phase") == "sensitivity_done"]
    assert len(done) == 1
    assert [r["name"] for r in done[0]["sensitivity"]] == ["lenspos",
                                                           "lensdy"]
    draws = [e for e in events if e.get("phase") == "mc"]
    assert len(draws) == DRAWS
    for e in draws:
        assert e["frac"] == pytest.approx(e["draw"] / DRAWS)
        assert "merit" in e and "merit_yield" in e and "params" in e
    assert events[-1]["status"] == "completed"
    assert events[-1]["merit_yield"] == pytest.approx(
        session["narrow"]["mc"]["yield_fraction"])
