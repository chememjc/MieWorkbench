"""TolerancePane + ToleranceController tests: table -> spec-string -> argv
round-trip through the real cli_specs 'tolerance' parser, the live yield
histogram + sensitivity bar chart fed by synthetic progress events,
run-state gating, and one REAL QProcess run of a stub script proving the
'@MIEWB' -> progress() -> plots loop end to end (the stub stands in for
optics-env tolerance.py so the GUI suite stays fast and FreeCAD-free)."""

import os
import sys
import textwrap

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

import cli_specs  # noqa: E402  (stdlib-only)

from mieworkbench.core.tolerance_controller import (  # noqa: E402
    ToleranceController)
from mieworkbench.panes.tolerance_pane import (  # noqa: E402
    PENALTY_FLOOR, TolerancePane)


# ---------------------------------------------------------------------------
# pane: tables -> config -> spec strings
# ---------------------------------------------------------------------------
def test_pane_default_state(qtbot):
    pane = TolerancePane()
    qtbot.addWidget(pane)
    assert pane.tol_table.rowCount() == 0
    # one default operand row (spot_rms) is seeded
    assert pane.operand_table.rowCount() == 1
    cfg = pane.config()
    assert cfg["tolerance"] == []
    assert cfg["operand"] == ["spot_rms:0:1"]
    assert cfg["draws"] == 50
    assert "merit_threshold" not in cfg    # checkbox off by default
    assert "compensator" not in cfg        # group unchecked by default
    assert not pane.stop_btn.isEnabled()
    assert pane.run_btn.isEnabled()


def test_pane_config_round_trips_through_parser(qtbot):
    pane = TolerancePane()
    qtbot.addWidget(pane)
    pane.add_tolerance("lenspos", 0.0, "normal", 1.0)
    pane.add_tolerance("dim.lensdy", 0.0, "uniform", 4.0)
    pane.add_operand("detected_power", "Body.Pad.Face5", 0.0, 2.0)
    pane.draws_spin.setValue(12)
    pane.seed_spin.setValue(7)
    pane.threshold_check.setChecked(True)
    pane.threshold_spin.setValue(110000.0)
    pane.comp_group.setChecked(True)
    pane.comp_var_edit.setText("detpos")
    pane.comp_start_spin.setValue(50.236)
    pane.comp_lo_spin.setValue(40.0)
    pane.comp_hi_spin.setValue(62.0)
    pane.comp_budget_spin.setValue(8)
    pane.skip_sens_check.setChecked(True)
    pane.rays_spin.setValue(5000)

    cfg = pane.config()
    args = ToleranceController.build_args(cfg)
    parser = cli_specs.build_parser("tolerance")
    ns = parser.parse_args(["--model", "dummy.FCStd"] + args)

    assert ns.tolerance == [
        {"name": "lenspos", "nominal": 0.0, "dist": "normal",
         "band": 1.0},
        {"name": "dim.lensdy", "nominal": 0.0, "dist": "uniform",
         "band": 4.0}]
    assert ns.operand[0]["operand"] == "spot_rms"
    assert ns.operand[1] == {"operand": "detected_power",
                             "detector": "Body.Pad.Face5",
                             "target": 0.0, "weight": 2.0}
    assert ns.draws == 12 and ns.mc_seed == 7
    assert ns.merit_threshold == 110000.0
    assert ns.compensator == {"name": "detpos", "start": 50.236,
                              "lo": 40.0, "hi": 62.0}
    assert ns.comp_budget == 8
    assert ns.skip_sensitivity is True
    assert ns.rays == 5000.0


def test_pane_config_skips_blank_rows_and_defaults(qtbot):
    pane = TolerancePane()
    qtbot.addWidget(pane)
    pane.add_tolerance("", 0, "normal", 1)      # blank name -> skipped
    # compensator enabled but no variable named -> omitted
    pane.comp_group.setChecked(True)
    cfg = pane.config()
    assert cfg["tolerance"] == []
    assert "compensator" not in cfg
    args = ToleranceController.build_args(cfg)
    # defaults (draws 50, seed 42, sens-delta 1, hist-bins 20, preset
    # quick, backend worker, no threshold/compensator/rays) never emitted
    for flag in ("--draws", "--mc-seed", "--sens-delta", "--hist-bins",
                 "--preset", "--eval-backend", "--merit-threshold",
                 "--compensator", "--comp-budget", "--rays",
                 "--skip-sensitivity"):
        assert flag not in args, args


def test_pane_bad_cells_raise_with_row_named(qtbot):
    import pytest
    pane = TolerancePane()
    qtbot.addWidget(pane)
    pane.add_tolerance("lenspos", 0, "normal", 1)
    pane.tol_table.item(0, 3).setText("wide")
    with pytest.raises(ValueError, match="lenspos"):
        pane.config()
    pane.tol_table.item(0, 3).setText("1.5")
    pane.comp_group.setChecked(True)
    pane.comp_var_edit.setText("detpos")
    pane.comp_lo_spin.setValue(5.0)
    pane.comp_hi_spin.setValue(-5.0)            # lo >= hi
    with pytest.raises(ValueError, match="detpos"):
        pane.config()


# ---------------------------------------------------------------------------
# pane: progress -> plots / status / run-state
# ---------------------------------------------------------------------------
def _draw_event(i, merit, merit_yield=None, draws=10):
    return {"stage": "tolerance", "frac": i / draws, "status": "running",
            "msg": "", "phase": "mc", "draw": i, "draws": draws,
            "merit": merit, "penalized": merit >= PENALTY_FLOOR,
            "merit_yield": merit_yield, "params": {"lenspos": float(i)}}


def _sens_done_event(rows):
    return {"stage": "tolerance", "frac": None, "status": "running",
            "msg": "", "phase": "sensitivity_done", "sensitivity": rows}


def test_progress_feeds_plots_and_status(qtbot):
    pane = TolerancePane()
    qtbot.addWidget(pane)
    pane.on_started()
    assert pane.stop_btn.isEnabled() and not pane.run_btn.isEnabled()

    rows = [{"name": "lenspos", "rank": 1, "impact": 6188.0,
             "derivative": -561.5},
            {"name": "lensdy", "rank": 2, "impact": 1346.6,
             "derivative": 397.4}]
    pane.on_progress(_sens_done_event(rows))
    assert [r["name"] for r in pane.sens_plot.rows()] == ["lenspos",
                                                          "lensdy"]

    pane.on_progress(_draw_event(1, 90000.0, 1.0))
    pane.on_progress(_draw_event(2, 150000.0, 0.5))
    # non-tolerance stages and non-draw events are ignored
    pane.on_progress({"stage": "optimize", "eval": 9, "merit": 1.0,
                      "best": 1.0})
    pane.on_progress({"stage": "tolerance", "frac": None, "msg": "x",
                      "status": "running", "phase": "sensitivity",
                      "param": "lenspos"})
    assert pane.hist_plot.merit_count() == 2
    assert "yield 0.500" in pane.status_label.text()
    assert "2/10" in pane.status_label.text()

    pane.on_progress({"stage": "tolerance", "frac": 1.0, "msg": "done",
                      "status": "completed", "n_evals": 17,
                      "merit_yield": 0.5})
    assert "Done" in pane.status_label.text()
    assert "17 evals" in pane.status_label.text()

    pane.on_finished(0)
    assert pane.run_btn.isEnabled() and not pane.stop_btn.isEnabled()

    # a new run clears both plots
    pane.on_started()
    assert pane.hist_plot.merit_count() == 0
    assert pane.sens_plot.rows() == []
    pane.on_finished(3)
    assert "exit 3" in pane.status_label.text()


def test_penalized_draws_excluded_from_histogram_binning(qtbot):
    pane = TolerancePane()
    qtbot.addWidget(pane)
    pane.on_progress(_draw_event(1, 90000.0))
    pane.on_progress(_draw_event(2, 1e9))       # PENALTY sentinel
    pane.on_progress(_draw_event(3, 110000.0))
    assert pane.hist_plot.merit_count() == 3
    assert pane.hist_plot.plotted_merits() == [90000.0, 110000.0]
    edges, counts = pane.hist_plot.bins()
    assert sum(counts) == 2
    assert len(edges) == len(counts) + 1
    assert edges[0] <= 90000.0 and edges[-1] >= 110000.0


def test_histogram_bins_partition_identical_merits(qtbot):
    """All draws identical must not zero-divide the binning."""
    pane = TolerancePane()
    qtbot.addWidget(pane)
    for i in (1, 2, 3):
        pane.on_progress(_draw_event(i, 5.0))
    edges, counts = pane.hist_plot.bins()
    assert sum(counts) == 3


# ---------------------------------------------------------------------------
# controller: argv construction + a real stub-process run
# ---------------------------------------------------------------------------
def test_build_args_appends_repeatable_flags():
    args = ToleranceController.build_args(
        {"tolerance": ["a:0:normal:1", "b:2:uniform:4"],
         "operand": ["spot_rms:0:1"], "draws": 20,
         "compensator": "c:0:-1:1", "comp_budget": 8})
    assert args.count("--tolerance") == 2
    assert args[args.index("--tolerance") + 1] == "a:0:normal:1"
    assert "--operand" in args
    assert args[args.index("--draws") + 1] == "20"
    assert args[args.index("--compensator") + 1] == "c:0:-1:1"
    assert args[args.index("--comp-budget") + 1] == "8"


def test_build_args_never_emits_model_or_help():
    args = ToleranceController.build_args(
        {"model": "x.FCStd", "help": True, "tolerance": []})
    assert args == []


STUB = textwrap.dedent("""\
    import json, sys
    sens = {"ev": "progress", "stage": "tolerance", "frac": None,
            "msg": "ranking", "status": "running",
            "phase": "sensitivity_done",
            "sensitivity": [{"name": "lenspos", "rank": 1,
                             "impact": 6188.0, "derivative": -561.5}]}
    print("@MIEWB " + json.dumps(sens), flush=True)
    for i in (1, 2, 3):
        ev = {"ev": "progress", "stage": "tolerance", "frac": i / 3.0,
              "msg": "draw %d/3" % i, "status": "running", "phase": "mc",
              "draw": i, "draws": 3, "merit": 90000.0 + 1000.0 * i,
              "penalized": False, "merit_yield": 1.0,
              "params": {"lenspos": float(i)}}
        print("@MIEWB " + json.dumps(ev), flush=True)
    print("[tolerance] yield: 3/3 = 1.000 (merit <= 110000)", flush=True)
    sys.exit(0)
""")


def test_controller_runs_stub_and_feeds_pane(qtbot, tmp_path):
    """End to end through a REAL QProcess: stub tolerance.py emits one
    sensitivity_done + three draw events + one log line; the controller
    must classify them (progress vs line) and the pane's plots must
    fill."""
    stub = tmp_path / "stub_tolerance.py"
    stub.write_text(STUB)
    ctl = ToleranceController(python=sys.executable, script=str(stub))
    pane = TolerancePane()
    qtbot.addWidget(pane)

    lines, events = [], []
    ctl.line.connect(lines.append)
    ctl.progress.connect(events.append)
    ctl.progress.connect(pane.on_progress)

    with qtbot.waitSignal(ctl.finished, timeout=30000) as blocker:
        # the stub ignores its args; --model is still passed positionally
        assert ctl.start("dummy.FCStd",
                         ["--tolerance", "lenspos:0:normal:1",
                          "--operand", "spot_rms:0:1"])
        assert ctl.is_running()
    assert blocker.args == [0]
    assert not ctl.is_running()

    assert len(events) == 4
    assert all(e["stage"] == "tolerance" for e in events)
    assert pane.hist_plot.merit_count() == 3
    assert [r["name"] for r in pane.sens_plot.rows()] == ["lenspos"]
    assert any(ln.startswith("[tolerance]") for ln in lines)
    # a second start is allowed once finished
    with qtbot.waitSignal(ctl.finished, timeout=30000):
        assert ctl.start("dummy.FCStd", [])


def test_controller_refuses_concurrent_start(qtbot, tmp_path):
    stub = tmp_path / "slow_stub.py"
    stub.write_text("import time\ntime.sleep(30)\n")
    ctl = ToleranceController(python=sys.executable, script=str(stub))
    try:
        assert ctl.start("dummy.FCStd", [])
        assert ctl.is_running()
        assert not ctl.start("dummy.FCStd", [])   # refused, no-op
    finally:
        with qtbot.waitSignal(ctl.finished, timeout=10000):
            ctl.stop()
        assert not ctl.is_running()


def test_console_classifies_tolerance_lines():
    from mieworkbench.panes.console import (STAGE_CHOICES, classify_stage)
    assert "tolerance" in STAGE_CHOICES
    assert classify_stage("[tolerance] yield: 4/6 = 0.667") == "tolerance"
    assert classify_stage("[optimize] best merit 1.0") == "optimize"
