"""OptimizePane + OptimizeController tests: table -> spec-string -> argv
round-trip through the real cli_specs 'optimize' parser, the live
convergence plot fed by synthetic progress events, run-state gating, and
one REAL QProcess run of a stub script proving the '@MIEWB' -> progress()
-> plot loop end to end (the stub stands in for optics-env optimize.py so
the GUI suite stays fast and FreeCAD-free)."""

import os
import sys
import textwrap

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

import cli_specs  # noqa: E402  (stdlib-only)

from mieworkbench.core.optimize_controller import (  # noqa: E402
    OptimizeController)
from mieworkbench.panes.optimize_pane import (  # noqa: E402
    OptimizePane, PENALTY_FLOOR)


# ---------------------------------------------------------------------------
# pane: tables -> config -> spec strings
# ---------------------------------------------------------------------------
def test_pane_default_state(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    assert pane.var_table.rowCount() == 0
    # one default operand row (spot_rms) is seeded
    assert pane.operand_table.rowCount() == 1
    cfg = pane.config()
    assert cfg["var"] == []
    assert cfg["operand"] == ["spot_rms:0:1"]
    assert cfg["algorithm"] == "local"
    assert cfg["budget"] == 40
    assert not pane.stop_btn.isEnabled()
    assert pane.run_btn.isEnabled()


def test_pane_config_round_trips_through_parser(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.add_variable("lenspos", -6.0, -8.0, 8.0)
    pane.add_variable("dim.sphered", 25.0, 20.0, 30.0)
    pane.add_operand("detected_power", "Body.Pad.Face5", 0.0, 2.0)
    pane.algorithm_combo.setCurrentText("global")
    pane.budget_spin.setValue(55)
    pane.rays_spin.setValue(5000)
    pane.final_coherent_check.setChecked(False)

    cfg = pane.config()
    args = OptimizeController.build_args(cfg)
    parser = cli_specs.build_parser("optimize")
    ns = parser.parse_args(["--model", "dummy.FCStd"] + args)

    assert ns.var == [
        {"name": "lenspos", "start": -6.0, "lo": -8.0, "hi": 8.0},
        {"name": "dim.sphered", "start": 25.0, "lo": 20.0, "hi": 30.0}]
    assert ns.operand[0]["operand"] == "spot_rms"
    assert ns.operand[1] == {"operand": "detected_power",
                             "detector": "Body.Pad.Face5",
                             "target": 0.0, "weight": 2.0}
    assert ns.algorithm == "global"
    assert ns.budget == 55
    assert ns.rays == 5000.0
    assert ns.no_final_coherent is True


def test_pane_config_skips_blank_rows_and_defaults(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.add_variable("", 0, -1, 1)       # blank name -> skipped
    cfg = pane.config()
    assert cfg["var"] == []
    args = OptimizeController.build_args(cfg)
    # defaults (algorithm local, budget 40, tol 1e-3, preset quick,
    # backend worker, final coherent on) must not be emitted
    for flag in ("--algorithm", "--budget", "--tol", "--preset",
                 "--eval-backend", "--no-final-coherent", "--rays"):
        assert flag not in args, args


def test_pane_bad_number_raises_with_row_named(qtbot):
    import pytest
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.add_variable("lenspos", 0, -1, 1)
    pane.var_table.item(0, 1).setText("not-a-number")
    with pytest.raises(ValueError, match="lenspos"):
        pane.config()


# ---------------------------------------------------------------------------
# pane: progress -> plot / best readout / run-state
# ---------------------------------------------------------------------------
def _event(i, merit, best, params=None, budget=10):
    return {"stage": "optimize", "frac": i / budget, "status": "running",
            "msg": "", "eval": i, "budget": budget, "merit": merit,
            "best": best, "params": params or {"lenspos": float(i)},
            "best_params": params or {"lenspos": float(i)}}


def test_progress_feeds_plot_and_best_label(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.on_started()
    assert pane.stop_btn.isEnabled() and not pane.run_btn.isEnabled()

    pane.on_progress(_event(1, 100.0, 100.0))
    pane.on_progress(_event(2, 40.0, 40.0,
                            params={"lenspos": 4.25}))
    # non-optimize stages and non-eval events are ignored
    pane.on_progress({"stage": "trace", "frac": 0.5, "eval": 9,
                      "merit": 1.0, "best": 1.0})
    pane.on_progress({"stage": "optimize", "frac": 1.0,
                      "status": "completed", "msg": "done"})
    assert pane.plot.point_count() == 2
    assert "40" in pane.best_label.text()
    assert "lenspos=4.25" in pane.best_label.text()

    pane.on_finished(0)
    assert pane.run_btn.isEnabled() and not pane.stop_btn.isEnabled()

    # a new run clears the plot
    pane.on_started()
    assert pane.plot.point_count() == 0
    pane.on_finished(3)
    assert "exit 3" in pane.best_label.text()


def test_penalized_evals_excluded_from_plot_scaling(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.on_progress(_event(1, 25.0, 25.0))
    pane.on_progress(_event(2, 1e9, 25.0))     # PENALTY sentinel
    pane.on_progress(_event(3, 16.0, 16.0))
    assert pane.plot.point_count() == 3
    assert pane.plot.plotted_merits() == [25.0, 16.0]
    assert all(m < PENALTY_FLOOR for m in pane.plot.plotted_merits())


# ---------------------------------------------------------------------------
# controller: argv construction + a real stub-process run
# ---------------------------------------------------------------------------
def test_build_args_appends_repeatable_flags():
    args = OptimizeController.build_args(
        {"var": ["a:0:-1:1", "b:2:0:4"], "operand": ["spot_rms:0:1"],
         "algorithm": "global", "budget": 20})
    assert args.count("--var") == 2
    assert args[args.index("--var") + 1] == "a:0:-1:1"
    assert "--operand" in args and "--algorithm" in args
    assert args[args.index("--budget") + 1] == "20"


def test_build_args_never_emits_model_or_help():
    args = OptimizeController.build_args(
        {"model": "x.FCStd", "help": True, "var": []})
    assert args == []


STUB = textwrap.dedent("""\
    import json, sys, time
    for i in (1, 2, 3):
        ev = {"ev": "progress", "stage": "optimize", "frac": i / 3.0,
              "msg": "eval %d/3" % i, "status": "running", "eval": i,
              "budget": 3, "merit": 100.0 / i, "best": 100.0 / i,
              "params": {"lenspos": float(i)},
              "best_params": {"lenspos": float(i)}}
        print("@MIEWB " + json.dumps(ev), flush=True)
    print("[optimize] best merit 33.3 at {'lenspos': 3.0}", flush=True)
    sys.exit(0)
""")


def test_controller_runs_stub_and_feeds_pane(qtbot, tmp_path):
    """End to end through a REAL QProcess: stub optimize.py emits three
    '@MIEWB' events + one log line; the controller must classify them
    (progress vs line) and the pane's plot must fill."""
    stub = tmp_path / "stub_optimize.py"
    stub.write_text(STUB)
    ctl = OptimizeController(python=sys.executable, script=str(stub))
    pane = OptimizePane()
    qtbot.addWidget(pane)

    lines, events = [], []
    ctl.line.connect(lines.append)
    ctl.progress.connect(events.append)
    ctl.progress.connect(pane.on_progress)

    with qtbot.waitSignal(ctl.finished, timeout=30000) as blocker:
        # the stub ignores its args; --model is still passed positionally
        assert ctl.start("dummy.FCStd", ["--var", "lenspos:0:-1:1",
                                         "--operand", "spot_rms:0:1"])
        assert ctl.is_running()
    assert blocker.args == [0]
    assert not ctl.is_running()

    assert len(events) == 3
    assert all(e["stage"] == "optimize" for e in events)
    assert pane.plot.point_count() == 3
    assert any(ln.startswith("[optimize]") for ln in lines)
    # a second start is allowed once finished
    with qtbot.waitSignal(ctl.finished, timeout=30000):
        assert ctl.start("dummy.FCStd", [])


def test_controller_refuses_concurrent_start(qtbot, tmp_path):
    stub = tmp_path / "slow_stub.py"
    stub.write_text("import time\ntime.sleep(30)\n")
    ctl = OptimizeController(python=sys.executable, script=str(stub))
    try:
        assert ctl.start("dummy.FCStd", [])
        assert ctl.is_running()
        assert not ctl.start("dummy.FCStd", [])   # refused, no-op
    finally:
        with qtbot.waitSignal(ctl.finished, timeout=10000):
            ctl.stop()
        assert not ctl.is_running()


def test_console_classifies_optimize_lines():
    from mieworkbench.panes.console import (STAGE_CHOICES, classify_stage)
    assert "optimize" in STAGE_CHOICES
    assert classify_stage("[optimize] best merit 1.0") == "optimize"
    assert classify_stage("[trace] seed 1/3") == "trace"
