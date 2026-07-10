"""Offscreen tests for the right-click "save image" / "export data as
CSV" / "export table as CSV" actions added to the Results and Compare
panes (see CLAUDE.md's GUI round notes). Covers:

  - results.resolve_data_csv: the index.csv / same-basename pairing
    resolution shared by every image display.
  - results.build_image_context_menu: dialog-free menu construction
    (action texts, enabled state) plus invoking the actions with the
    save-dialog seam (_choose_save_path) monkeypatched.
  - widgets.table_export.export_table_csv: atomic CSV write from a
    QTableWidget.
  - The table right-click "Export CSV..." wiring on ResultsPane
    (summary/power) and ComparePane (metrics_table).

No test ever calls QMenu.exec()/QFileDialog -- menus are built and
inspected directly, and the one dialog seam (_choose_save_path) is
monkeypatched to a fixed path, per CLAUDE.md's "never show an
unguarded modal in a pane code path" trap.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QWidget  # noqa: E402

from mieworkbench.panes import results  # noqa: E402
from mieworkbench.panes.compare_pane import ComparePane  # noqa: E402
from mieworkbench.panes.results import ResultsPane  # noqa: E402
from mieworkbench.tests.test_results_problems_panels import (  # noqa: E402
    make_fake_case)
from mieworkbench.widgets.table_export import export_table_csv  # noqa: E402

# same tiny valid PNG literal the other results/compare tests use
_PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00"
        b"\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\r"
        b"IDATx\x9cc\xfc\xff\xff?\x00\x05\xfe\x02\xfe\xa75\x81\x84\x00"
        b"\x00\x00\x00IEND\xaeB`\x82")


def _write_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG)


def _write_index_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["file", "entity", "chart", "units",
                            "provenance", "image"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -- resolve_data_csv ---------------------------------------------------------

def test_resolve_data_csv_index_hit(tmp_path):
    case = tmp_path / "case1"
    _write_png(case / "images" / "foo.png")
    (case / "data").mkdir(parents=True)
    (case / "data" / "foo_data.csv").write_text("x,y\n1,2\n")
    _write_index_csv(case / "data" / "index.csv", [{
        "file": "data/foo_data.csv", "entity": "Screen", "chart": "line",
        "units": "W", "provenance": "post_process", "image": "images/foo.png",
    }])

    hit = results.resolve_data_csv(str(case / "images" / "foo.png"))
    assert hit == os.path.normpath(str(case / "data" / "foo_data.csv"))


def test_resolve_data_csv_basename_fallback(tmp_path):
    case = tmp_path / "case2"
    _write_png(case / "images" / "bar.png")
    (case / "data").mkdir(parents=True)
    (case / "data" / "bar.csv").write_text("x,y\n3,4\n")
    # deliberately no index.csv here

    hit = results.resolve_data_csv(str(case / "images" / "bar.png"))
    assert hit == os.path.normpath(str(case / "data" / "bar.csv"))


def test_resolve_data_csv_no_match(tmp_path):
    case = tmp_path / "case3"
    _write_png(case / "images" / "baz.png")
    # no data/ dir at all
    assert results.resolve_data_csv(str(case / "images" / "baz.png")) is None

    # data/ dir exists but nothing pairs with baz.png
    (case / "data").mkdir(parents=True)
    (case / "data" / "unrelated.csv").write_text("a,b\n1,2\n")
    assert results.resolve_data_csv(str(case / "images" / "baz.png")) is None


# -- build_image_context_menu --------------------------------------------------

def test_build_image_context_menu_no_pairing(qtbot, tmp_path):
    png = tmp_path / "img.png"
    _write_png(png)
    parent = QWidget()
    qtbot.addWidget(parent)

    menu = results.build_image_context_menu(str(png), parent)
    actions = menu.actions()
    assert [a.text() for a in actions] == [
        "Save image as…", "Export data as CSV…"]
    assert actions[0].isEnabled()
    assert not actions[1].isEnabled()


def test_build_image_context_menu_with_pairing(qtbot, tmp_path):
    case = tmp_path / "case"
    png = case / "images" / "foo.png"
    _write_png(png)
    (case / "data").mkdir(parents=True)
    (case / "data" / "foo.csv").write_text("a,b\n1,2\n")

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = results.build_image_context_menu(str(png), parent)
    export_act = menu.actions()[1]
    assert export_act.isEnabled()


def test_save_image_action_copies_bytes(qtbot, tmp_path, monkeypatch):
    src = tmp_path / "src.png"
    _write_png(src)
    dest = tmp_path / "out" / "saved.png"
    monkeypatch.setattr(results, "_choose_save_path",
                        lambda parent, default_name: str(dest))

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = results.build_image_context_menu(str(src), parent)
    save_act = menu.actions()[0]
    assert save_act.text() == "Save image as…"
    save_act.trigger()

    assert dest.read_bytes() == src.read_bytes()


def test_export_csv_action_copies_bytes(qtbot, tmp_path, monkeypatch):
    case = tmp_path / "case"
    png = case / "images" / "foo.png"
    _write_png(png)
    (case / "data").mkdir(parents=True)
    src_csv = case / "data" / "foo.csv"
    src_csv.write_text("a,b\n1,2\n")
    dest = tmp_path / "exported" / "foo_data.csv"
    monkeypatch.setattr(results, "_choose_save_path",
                        lambda parent, default_name: str(dest))

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = results.build_image_context_menu(str(png), parent)
    export_act = menu.actions()[1]
    assert export_act.isEnabled()
    export_act.trigger()

    assert dest.read_text() == src_csv.read_text()


def test_status_callback_invoked_after_save(qtbot, tmp_path, monkeypatch):
    src = tmp_path / "src.png"
    _write_png(src)
    dest = tmp_path / "saved.png"
    monkeypatch.setattr(results, "_choose_save_path",
                        lambda parent, default_name: str(dest))
    messages = []

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = results.build_image_context_menu(str(src), parent, messages.append)
    menu.actions()[0].trigger()

    assert len(messages) == 1
    assert str(dest) in messages[0]


# -- export_table_csv -----------------------------------------------------------

def test_export_table_csv_roundtrip_atomic(qtbot, tmp_path):
    table = QTableWidget(2, 2)
    qtbot.addWidget(table)
    table.setHorizontalHeaderLabels(["A", "B"])
    table.setItem(0, 0, QTableWidgetItem("1"))
    table.setItem(0, 1, QTableWidgetItem("2"))
    table.setItem(1, 0, QTableWidgetItem("3"))
    # (1, 1) deliberately left unset -> should export as ""

    path = tmp_path / "out.csv"
    export_table_csv(table, path)

    assert not os.path.exists(str(path) + ".tmp")
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["A", "B"]
    assert rows[1] == ["1", "2"]
    assert rows[2] == ["3", ""]


# -- table right-click "Export CSV..." wiring ----------------------------------

def test_results_summary_table_export_menu(qtbot, tmp_path, monkeypatch):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = make_fake_case(tmp_path)
    pane.load_case(str(case))
    assert pane.summary.rowCount() > 0

    dest = tmp_path / "summary_out.csv"
    monkeypatch.setattr(results, "_choose_save_path",
                        lambda parent, default_name: str(dest))

    menu = pane._build_table_export_menu(pane.summary, "summary.csv")
    act = menu.actions()[0]
    assert act.text() == "Export CSV…"
    act.trigger()

    with open(dest, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["Detector", "Power [mW]", "Peak [W/m²]",
                       "Pixel [µm]", "Visibility"]
    assert len(rows) == 1 + pane.summary.rowCount()


def test_results_power_table_export_menu(qtbot, tmp_path, monkeypatch):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = make_fake_case(tmp_path)
    pane.load_case(str(case))

    dest = tmp_path / "power_out.csv"
    monkeypatch.setattr(results, "_choose_save_path",
                        lambda parent, default_name: str(dest))

    menu = pane._build_table_export_menu(pane.power, "power.csv")
    menu.actions()[0].trigger()
    assert dest.exists()


def test_results_table_context_menu_policy(qtbot):
    from PySide6.QtCore import Qt
    pane = ResultsPane()
    qtbot.addWidget(pane)
    assert pane.summary.contextMenuPolicy() == Qt.CustomContextMenu
    assert pane.power.contextMenuPolicy() == Qt.CustomContextMenu


def test_compare_metrics_table_export_menu(qtbot, tmp_path, monkeypatch):
    from mieworkbench.tests.test_compare_pane import make_fake_summary

    pane = ComparePane()
    qtbot.addWidget(pane)
    out_dir = tmp_path / "cmp_out"
    make_fake_summary(out_dir)
    assert pane.load_summary(out_dir)
    assert pane.metrics_table.rowCount() > 0

    dest = tmp_path / "metrics_out.csv"
    monkeypatch.setattr(results, "_choose_save_path",
                        lambda parent, default_name: str(dest))

    menu = pane._build_table_export_menu(pane.metrics_table, "metrics.csv")
    act = menu.actions()[0]
    assert act.text() == "Export CSV…"
    act.trigger()

    with open(dest, newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 1 + pane.metrics_table.rowCount()


def test_compare_gallery_image_menu_no_exec(qtbot, tmp_path):
    """Compare's galleries reuse results._Gallery/_ClickableLabel
    wholesale -- confirm a thumbnail there gets the same context menu
    without ever calling .exec()."""
    from mieworkbench.tests.test_compare_pane import make_fake_summary

    pane = ComparePane()
    qtbot.addWidget(pane)
    out_dir = tmp_path / "cmp_out"
    make_fake_summary(out_dir)
    assert pane.load_summary(out_dir)

    gallery = pane.images_gallery
    assert gallery._paths
    # find the actual _ClickableLabel widget in the grid to exercise its
    # dialog-free menu builder directly
    found = None
    for i in range(gallery._grid.count()):
        cell = gallery._grid.itemAt(i).widget()
        if cell is None:
            continue
        for child in cell.findChildren(results._ClickableLabel):
            found = child
            break
        if found:
            break
    assert found is not None
    menu = found.build_context_menu()
    assert [a.text() for a in menu.actions()] == [
        "Save image as…", "Export data as CSV…"]
