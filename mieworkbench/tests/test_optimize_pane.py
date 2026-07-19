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
from mieworkbench.core.variables import VarRow  # noqa: E402
from mieworkbench.panes.optimize_pane import (  # noqa: E402
    OptimizePane, PENALTY_FLOOR, variable_bounds)


def _varrows():
    """A synthetic parse_sheet() result: one variable with real sweep
    bounds, one whose sheet has no __min/__max meta (parse_sheet echoes
    vmin == vmax == value), and one zero-valued unbounded variable."""
    return {
        "lenspos": VarRow(name="lenspos", value_raw="-6", value=-6.0,
                          vmin=-8.0, vmax=8.0, nstep=5, enabled=True,
                          row=1),
        "gap": VarRow(name="gap", value_raw="5", value=5.0,
                      vmin=5.0, vmax=5.0, nstep=0, enabled=True, row=2),
        "tiltz": VarRow(name="tiltz", value_raw="0", value=0.0,
                        vmin=0.0, vmax=0.0, nstep=0, enabled=True, row=3),
    }


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
    pane.set_variables(_varrows())            # lenspos is a miewb_vars global
    pane.add_variable("lenspos", -6.0, -8.0, 8.0)
    pane.add_variable("dim.sphered", 25.0, 20.0, 30.0)
    pane.add_variable("barealias", 25.0, 20.0, 30.0)   # not a global
    pane.add_operand("detected_power", "Body.Pad.Face5", 0.0, 2.0)
    pane.algorithm_combo.setCurrentText("global")
    pane.budget_spin.setValue(55)
    pane.rays_spin.setValue(5000)
    pane.final_coherent_check.setChecked(False)

    cfg = pane.config()
    args = OptimizeController.build_args(cfg)
    parser = cli_specs.build_parser("optimize")
    ns = parser.parse_args(["--model", "dummy.FCStd"] + args)

    # a name that IS a miewb_vars global is emitted sheet-qualified;
    # 'sheetlabel.alias' passes through untouched; a bare name NOT in the
    # varrows stays bare (a literal dim-sheet alias)
    assert ns.var == [
        {"name": "miewb_vars.lenspos", "start": -6.0, "lo": -8.0, "hi": 8.0},
        {"name": "dim.sphered", "start": 25.0, "lo": 20.0, "hi": 30.0},
        {"name": "barealias", "start": 25.0, "lo": 20.0, "hi": 30.0}]
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


# ---------------------------------------------------------------------------
# pane: config() -> apply_config() -> config() persistence round-trip
# ---------------------------------------------------------------------------
def test_apply_config_round_trips(qtbot):
    """A rich config (multiple variables incl. a sheet-qualified global
    and a bare dim-sheet alias, a non-default operand, and every scalar
    setting away from its default) survives a fresh pane's
    config() -> apply_config() -> config() cycle unchanged -- the
    persistence contract Project.set/get_optimize_config relies on."""
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.set_variables(_varrows())
    pane.add_variable("lenspos", -6.0, -8.0, 8.0)      # miewb_vars global
    pane.add_variable("barealias", 25.0, 20.0, 30.0)   # bare dim alias
    pane.add_operand("detected_power", "Body.Pad.Face5", 0.0, 2.0)
    pane.algorithm_combo.setCurrentText("global")
    pane.budget_spin.setValue(55)
    pane.tol_spin.setValue(1e-5)
    pane.preset_combo.setCurrentText("detailed")
    pane.backend_combo.setCurrentText("full")
    pane.rays_spin.setValue(5000)
    pane.final_coherent_check.setChecked(False)
    cfg1 = pane.config()

    fresh = OptimizePane()
    qtbot.addWidget(fresh)
    fresh.apply_config(cfg1)
    cfg2 = fresh.config()
    assert cfg2 == cfg1


def test_apply_config_default_state_round_trips(qtbot):
    """The as-constructed default config (empty vars, the seeded default
    operand row, every setting at its default) also round-trips."""
    pane = OptimizePane()
    qtbot.addWidget(pane)
    cfg1 = pane.config()

    fresh = OptimizePane()
    qtbot.addWidget(fresh)
    fresh.apply_config(cfg1)
    assert fresh.config() == cfg1


def test_apply_config_none_and_empty_are_noop(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.add_variable("lenspos", -6.0, -8.0, 8.0)
    before = pane.config()
    pane.apply_config(None)
    pane.apply_config({})
    assert pane.config() == before


def test_apply_config_skips_unparseable_specs(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.apply_config({"var": ["not:a:valid"], "operand": ["also bad"]})
    assert pane.config()["var"] == []
    assert pane.config()["operand"] == []


# ---------------------------------------------------------------------------
# pane: miewb_vars name dropdowns + auto-fill
# ---------------------------------------------------------------------------
def test_variable_bounds_fallbacks():
    rows = _varrows()
    assert variable_bounds(rows["lenspos"]) == (-6.0, -8.0, 8.0)
    # unspecified bounds (vmin == vmax == value) -> value ± 10 %
    assert variable_bounds(rows["gap"]) == (5.0, 4.5, 5.5)
    # zero value -> ± 0.1
    assert variable_bounds(rows["tiltz"]) == (0.0, -0.1, 0.1)


def test_set_variables_populates_name_combos(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.add_variable("custom", 1.0, 0.0, 2.0)   # typed before any scene
    pane.set_variables(_varrows())
    combo = pane.var_table.cellWidget(0, 0)
    assert [combo.itemText(i) for i in range(combo.count())] \
        == ["lenspos", "gap", "tiltz"]
    # the existing row's typed name and explicit values survive
    assert combo.currentText() == "custom"
    assert pane.var_table.item(0, 1).text() == "1"
    # back to no scene: combo empties but stays editable free-text
    pane.set_variables({})
    assert combo.count() == 0
    assert combo.isEditable()
    assert combo.currentText() == "custom"
    assert pane.variables() == ["custom:1:0:2"]


def test_pick_variable_autofills_start_and_bounds(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.set_variables(_varrows())
    row = pane.add_variable()          # blank -> adopts the first variable
    combo = pane.var_table.cellWidget(row, 0)
    assert combo.currentText() == "lenspos"
    # sheet has real __min/__max: start = value, bounds = vmin/vmax
    assert pane.var_table.item(row, 1).text() == "-6"
    assert pane.var_table.item(row, 2).text() == "-8"
    assert pane.var_table.item(row, 3).text() == "8"

    combo.setCurrentText("gap")        # no bounds meta -> value ± 10 %
    assert pane.var_table.item(row, 1).text() == "5"
    assert pane.var_table.item(row, 2).text() == "4.5"
    assert pane.var_table.item(row, 3).text() == "5.5"

    combo.setCurrentText("tiltz")      # zero value -> ± 0.1
    assert pane.var_table.item(row, 1).text() == "0"
    assert pane.var_table.item(row, 2).text() == "-0.1"
    assert pane.var_table.item(row, 3).text() == "0.1"

    # the assembled spec strings read the combo, not a table item; a
    # miewb_vars global is emitted sheet-qualified
    assert pane.variables() == ["miewb_vars.tiltz:0:-0.1:0.1"]

    # typing a non-variable name leaves the cells alone; a 'sheet.alias'
    # name passes through unqualified
    combo.setCurrentText("dim.ct")
    assert pane.var_table.item(row, 1).text() == "0"
    assert pane.variables() == ["dim.ct:0:-0.1:0.1"]


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


# ---------------------------------------------------------------------------
# qualification contract: pane specs resolve through the REAL split_var
# ---------------------------------------------------------------------------
def _real_split_var():
    """The genuine permute_model.split_var (the backend resolver at
    permute_model.py). permute_model imports FreeCAD at module load, so
    stub it with a MagicMock — split_var itself never touches FreeCAD, so
    we exercise the real code path and can never drift from it."""
    import types  # noqa: F401
    from unittest import mock
    scripts_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    with mock.patch.dict(sys.modules, {"FreeCAD": mock.MagicMock()}):
        import importlib
        permute_model = importlib.import_module("permute_model")
    return permute_model.split_var


def test_global_variable_resolves_to_miewb_vars_sheet(qtbot):
    """A pane seeded from a miewb_vars-only variable set must emit specs
    that permute_model.split_var routes to sheet 'miewb_vars', NOT the
    per-element 'dim' sheet (the bare-name bug)."""
    split_var = _real_split_var()
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.set_variables(_varrows())
    pane.add_variable("lenspos")           # a global
    pane.add_variable("dim.sphered", 1.0, 0.0, 2.0)   # a dim-sheet alias
    specs = pane.variables()

    # spec string is "name:start:lo:hi"; the name is everything before the
    # FIRST ':' — and split_var reads its sheet from the name
    def sheet_of(spec):
        name = spec.split(":", 1)[0]
        return split_var(name)[0]

    assert sheet_of(specs[0]) == "miewb_vars"       # global -> miewb_vars
    assert sheet_of(specs[1]) == "dim"              # 'dim.sphered' -> dim


# ---------------------------------------------------------------------------
# failure surfacing: banner captures the first backend error line
# ---------------------------------------------------------------------------
def test_failure_banner_surfaces_first_error(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.on_started()
    assert not pane.error_banner.isVisibleTo(pane)

    # simulate the process output: a real PermuteError line + all-penalized
    # evaluations (merit == PENALTY sentinel), then a non-zero exit
    pane.on_line("[trace] extracting geometry")
    pane.on_line("permute_model.py: alias 'z' not found on spreadsheet "
                 "'dim' (available aliases: ct, r_front)")
    pane.on_line("PermuteError: variant 0 failed")
    pane.on_progress(_event(1, 1e9, 1e9))
    pane.on_progress(_event(2, 1e9, 1e9))
    pane.on_finished(1)

    assert pane.error_banner.isVisibleTo(pane)
    txt = pane.error_banner.text()
    assert "alias 'z' not found on spreadsheet 'dim'" in txt
    assert "2 evaluations penalized" in txt
    assert "miewb_vars.<name>" in txt          # the qualification hint
    # only the FIRST matching error line is latched
    assert "PermiteError" not in txt

    # a clean re-run clears the banner
    pane.on_started()
    assert not pane.error_banner.isVisibleTo(pane)
    pane.on_progress(_event(1, 5.0, 5.0))
    pane.on_finished(0)
    assert not pane.error_banner.isVisibleTo(pane)


# ---------------------------------------------------------------------------
# WP3: convergence-plot data inspection + Apply optimum
# ---------------------------------------------------------------------------
def test_convergence_plot_stores_params_history_and_var_names(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.on_progress(_event(1, 5.0, 5.0, params={"z": 10.0}))
    pane.on_progress(_event(2, 2.0, 2.0, params={"z": 12.0, "gap": 3.0}))
    hist = pane.plot.history()
    assert [h["eval"] for h in hist] == [1, 2]
    assert hist[1]["params"] == {"z": 12.0, "gap": 3.0}
    # insertion-ordered union of param keys across points
    assert pane.plot.var_names() == ["z", "gap"]


def test_convergence_plot_tooltip_text(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.on_progress(_event(1, 5.0, 5.0, params={"z": 10.0}))
    pane.on_progress(_event(2, 2.0, 2.0, params={"z": 12.0}))
    tip = pane.plot._tooltip_for(1)      # the better (rank 1) point
    head = tip.splitlines()[0]
    assert head.startswith("eval 2")
    assert "rank 1" in head
    assert "z = 12" in tip


def test_convergence_plot_show_data_dialog_contents(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.on_progress(_event(1, 5.0, 5.0, params={"z": 10.0}))
    pane.plot._show_data_dialog()
    dlg = pane.plot._data_dialog
    assert dlg is not None
    assert dlg.table.horizontalHeaderItem(0).text() == "eval#"
    assert dlg.table.rowCount() == 1


def test_convergence_plot_data_dialog_csv_export(qtbot, tmp_path):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.on_progress(_event(1, 5.0, 5.0, params={"z": 10.0}))
    pane.plot._show_data_dialog()
    out = tmp_path / "conv.csv"
    pane.plot._data_dialog.export_to(str(out))
    assert out.exists() and out.read_text().splitlines()[0].startswith("eval#")


def test_apply_button_state_machine(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    assert not pane.apply_btn.isEnabled()

    # started -> disabled + best cleared
    pane.on_started()
    assert not pane.apply_btn.isEnabled()
    # progress alone does not enable (only on_finished decides)
    pane.on_progress(_event(1, 5.0, 5.0, params={"miewb_vars.gap": 3.0}))
    assert not pane.apply_btn.isEnabled()
    # clean finish with a real best -> enabled
    pane.on_finished(0)
    assert pane.apply_btn.isEnabled()
    assert pane.best_merit() == 5.0
    assert pane.best_params() == {"miewb_vars.gap": 3.0}

    # a penalized-only run -> no usable best -> disabled
    pane.on_started()
    pane.on_progress(_event(2, 1e9, 1e9, params={"miewb_vars.gap": 1.0}))
    pane.on_finished(0)
    assert not pane.apply_btn.isEnabled()

    # a clean run that exits non-zero -> disabled
    pane.on_started()
    pane.on_progress(_event(1, 2.0, 2.0, params={"miewb_vars.gap": 4.0}))
    pane.on_finished(1)
    assert not pane.apply_btn.isEnabled()

    # a clean run then reset_best() (scene load) -> disabled + cleared
    pane.on_started()
    pane.on_progress(_event(1, 2.0, 2.0, params={"miewb_vars.gap": 4.0}))
    pane.on_finished(0)
    assert pane.apply_btn.isEnabled()
    pane.reset_best()
    assert not pane.apply_btn.isEnabled()
    assert pane.best_params() == {}


def test_set_start_values_rewrites_matching_rows(qtbot):
    pane = OptimizePane()
    qtbot.addWidget(pane)
    pane.set_variables(_varrows())
    pane.add_variable("gap", start=5.0, lo=0.0, hi=10.0)
    pane.add_variable("lenspos", start=-6.0, lo=-8.0, hi=8.0)
    # keys are the miewb_vars-qualified spec names the optimizer reports
    pane.set_start_values({"miewb_vars.gap": 7.5})
    specs = {s.split(":")[0]: s for s in pane.config()["var"]}
    assert specs["miewb_vars.gap"].startswith("miewb_vars.gap:7.5:")
    assert specs["miewb_vars.lenspos"].startswith("miewb_vars.lenspos:-6:")
