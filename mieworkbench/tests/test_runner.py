"""RunController tests: build_args() round-trips through the real
cli_specs pipeline parser, and two REAL runs of scripts/run_pipeline.py
(--print-only and a --dry-run trace) prove the GUI -> QProcess -> @MIEWB
progress loop end to end."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

import cli_specs  # noqa: E402  (stdlib-only)

from mieworkbench.core.runner import RunController  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MODEL = REPO_ROOT / "example.FCStd"


def test_build_args_skips_defaults():
    args = RunController.build_args(
        {"rays": None, "seeds": None, "keep_going": False, "var": []})
    assert args == []


def test_build_args_round_trips_through_parser():
    cfg = {
        "rays": 2e5, "resolution": 256, "seeds": 3, "backend": "numpy",
        "dry_run": True, "keep_going": True,
        "source_face": ["Body1.Pad.Face1", "Body2.Pad.Face2"],
        "var": ["lenspos", "x"], "min": [1.0, -2.0], "max": [5.0, 2.0],
        "n": [2, 1],
    }
    args = RunController.build_args(cfg)

    parser = cli_specs.build_parser("pipeline")
    ns = parser.parse_args(["--models", "dummy.FCStd"] + args)

    assert ns.rays == cfg["rays"]
    assert ns.resolution == cfg["resolution"]
    assert ns.seeds == cfg["seeds"]
    assert ns.backend == cfg["backend"]
    assert ns.dry_run == cfg["dry_run"]
    assert ns.keep_going == cfg["keep_going"]
    assert ns.source_face == cfg["source_face"]
    assert ns.var == cfg["var"]
    assert ns.min == cfg["min"]
    assert ns.max == cfg["max"]
    assert ns.n == cfg["n"]


def test_build_args_never_emits_excluded_dests():
    args = RunController.build_args(
        {"models": ["x.FCStd"], "print_only": True, "help": True})
    assert args == []


def test_real_print_only_run(qtbot):
    assert EXAMPLE_MODEL.exists(), "example.FCStd fixture model is missing"
    runner = RunController()
    lines = []
    runner.line.connect(lines.append)

    with qtbot.waitSignal(runner.finished, timeout=60000) as blocker:
        started = runner.start(str(EXAMPLE_MODEL),
                               ["--print-only", "--preset", "quick"])
        assert started

    assert blocker.args == [0]
    assert any("print-only" in line for line in lines)


def test_real_dry_run_reports_trace_progress(qtbot):
    assert EXAMPLE_MODEL.exists(), "example.FCStd fixture model is missing"
    runner = RunController()
    progress_events = []
    runner.progress.connect(progress_events.append)

    with qtbot.waitSignal(runner.finished, timeout=120000) as blocker:
        started = runner.start(
            str(EXAMPLE_MODEL),
            ["--dry-run", "--preset", "quick", "--tag", "guishell"],
            steps="trace")
        assert started

    assert blocker.args == [0]
    assert any(ev.get("stage") == "trace" for ev in progress_events)
