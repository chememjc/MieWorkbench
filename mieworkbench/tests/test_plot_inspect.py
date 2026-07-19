"""panes/plot_inspect.py: the pure ranking / hit-test / tooltip / table
builders (Qt-free), plus the non-modal DataTableDialog (contents + CSV
export to a tmp path through the monkeypatchable file-dialog seam) and
the dialog-free context-menu builder."""

import csv
import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.panes import plot_inspect as pi  # noqa: E402


# ---------------------------------------------------------------------------
# dense_ranks
# ---------------------------------------------------------------------------
def test_dense_ranks_ties_and_penalized():
    # ascending merit -> rank 1 best; ties share a rank; the next
    # distinct value takes the next integer (1,2,2,3)
    assert pi.dense_ranks([10.0, 20.0, 20.0, 30.0]) == [1, 2, 2, 3]


def test_dense_ranks_unordered_input_keeps_input_order():
    assert pi.dense_ranks([30.0, 10.0, 20.0, 10.0]) == [3, 1, 2, 1]


def test_dense_ranks_penalized_and_none_unranked():
    ranks = pi.dense_ranks([5.0, 1e8, 2.0, None, 5.0])
    assert ranks == [2, None, 1, None, 2]


# ---------------------------------------------------------------------------
# nearest_point_index
# ---------------------------------------------------------------------------
def test_nearest_point_index_hits_closest():
    pts = [(0.0, 0.0), (100.0, 100.0), (10.0, 10.0)]
    assert pi.nearest_point_index(pts, (11.0, 9.0)) == 2


def test_nearest_point_index_miss_beyond_radius():
    pts = [(0.0, 0.0), (100.0, 100.0)]
    assert pi.nearest_point_index(pts, (50.0, 50.0), max_dist_px=12) is None


def test_nearest_point_index_empty():
    assert pi.nearest_point_index([], (0.0, 0.0)) is None


# ---------------------------------------------------------------------------
# format_point_tooltip
# ---------------------------------------------------------------------------
def test_format_point_tooltip_full():
    txt = pi.format_point_tooltip("eval 12", {"z": 94.5, "efl": 200},
                                  0.0034, 2)
    lines = txt.splitlines()
    assert lines[0] == "eval 12 — merit 0.0034 (rank 2)"
    assert lines[1] == "z = 94.5"
    assert lines[2] == "efl = 200"


def test_format_point_tooltip_no_merit_no_rank():
    txt = pi.format_point_tooltip("draw 3", {}, None, None)
    assert txt == "draw 3"


def test_format_point_tooltip_preserves_param_order():
    txt = pi.format_point_tooltip("x", {"b": 1, "a": 2}, None, None)
    assert txt.splitlines()[1:] == ["b = 1", "a = 2"]


# ---------------------------------------------------------------------------
# history_table
# ---------------------------------------------------------------------------
def _opt_entries():
    return [
        {"eval": 1, "params": {"z": 10.0}, "merit": 5.0},
        {"eval": 2, "params": {"z": 12.0, "gap": 3.0}, "merit": 2.0},
        {"eval": 3, "params": {"gap": 4.0}, "merit": 1e8},   # penalized
    ]


def test_history_table_optimize_var_union_and_ranks():
    headers, rows = pi.history_table(_opt_entries(), "optimize")
    # var columns are the insertion-ordered union of param keys
    assert headers == ["eval#", "z", "gap", "merit", "rank"]
    assert rows[0] == ["1", "10", "", "5", "2"]
    assert rows[1] == ["2", "12", "3", "2", "1"]
    # penalized eval has no rank and a blank for the missing var
    assert rows[2] == ["3", "", "4", "1e+08", ""]


def test_history_table_mc_pass_fail_column():
    entries = [
        {"draw": 1, "params": {"z": 1.0}, "merit": 5.0, "passed": True},
        {"draw": 2, "params": {"z": 2.0}, "merit": 9.0, "passed": False},
        {"draw": 3, "params": {"z": 3.0}, "merit": 7.0, "passed": None},
    ]
    headers, rows = pi.history_table(entries, "mc")
    assert headers == ["draw#", "z", "merit", "pass/fail", "rank"]
    assert rows[0][-2] == "pass"
    assert rows[1][-2] == "fail"
    assert rows[2][-2] == ""


def test_history_table_sensitivity():
    entries = [
        {"name": "lenspos", "impact": 0.9, "derivative": -0.1, "rank": 1},
        {"name": "gap", "impact": None, "derivative": 0.0, "rank": 2},
    ]
    headers, rows = pi.history_table(entries, "sensitivity")
    assert headers == ["name", "impact", "derivative", "rank"]
    assert rows[0] == ["lenspos", "0.9", "-0.1", "1"]
    assert rows[1] == ["gap", "", "0", "2"]


# ---------------------------------------------------------------------------
# DataTableDialog + build_plot_menu (Qt shell)
# ---------------------------------------------------------------------------
def test_data_table_dialog_contents_and_csv_export(qtbot, tmp_path):
    dlg = pi.DataTableDialog(_opt_entries(), "optimize", "Opt history")
    qtbot.addWidget(dlg)
    assert dlg.windowTitle() == "Opt history"
    assert dlg.table.columnCount() == 5
    assert dlg.table.rowCount() == 3
    assert dlg.table.item(1, 0).text() == "2"

    out = tmp_path / "hist.csv"
    dlg.export_to(str(out))
    with open(out, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["eval#", "z", "gap", "merit", "rank"]
    assert rows[2] == ["2", "12", "3", "2", "1"]


def test_data_table_dialog_export_button_uses_seam(qtbot, tmp_path,
                                                   monkeypatch):
    dlg = pi.DataTableDialog(_opt_entries(), "optimize")
    qtbot.addWidget(dlg)
    out = tmp_path / "via_button.csv"
    monkeypatch.setattr(pi, "_choose_csv_path",
                        lambda parent, default: str(out))
    dlg.export_btn.click()
    assert out.exists()


def test_data_table_dialog_export_button_cancelled(qtbot, monkeypatch):
    dlg = pi.DataTableDialog(_opt_entries(), "optimize")
    qtbot.addWidget(dlg)
    monkeypatch.setattr(pi, "_choose_csv_path", lambda parent, default: None)
    dlg.export_btn.click()   # no path chosen -> no write, no raise


def test_build_plot_menu_actions_fire(qtbot):
    fired = []
    menu = pi.build_plot_menu([("Show data…", lambda: fired.append("a"))])
    qtbot.addWidget(menu)
    actions = menu.actions()
    assert [a.text() for a in actions] == ["Show data…"]
    actions[0].trigger()
    assert fired == ["a"]
