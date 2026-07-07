"""Offscreen tests for the results pane, problems pane, transform panel,
and paraview launcher helpers."""

import json
import os
import sys

import pytest

from PySide6.QtCore import Qt

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
REAL_CASE = os.path.join(REPO, "results", "example", "quick-phase0")

from mieworkbench.core import paraview_launcher  # noqa: E402
from mieworkbench.panes.problems import ProblemsPane  # noqa: E402
from mieworkbench.panes.results import ResultsPane  # noqa: E402
from mieworkbench.panes.transform_panel import (  # noqa: E402
    ReferencePointPicker, TransformPanel)


def make_fake_case(tmp_path, status="completed"):
    case = tmp_path / "case"
    (case / "images").mkdir(parents=True)
    (case / "case.json").write_text(json.dumps({"status": status}))
    (case / "report.json").write_text(json.dumps({
        "closure_ok": True,
        "detectors": {"Screen.Face5": {"total_power_W": 0.0054,
                                       "peak_irradiance_W_m2": 12.3,
                                       "pixel_um": 58.6,
                                       "resolution": [512, 512]}}}))
    # tiny valid png
    png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00"
           b"\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\r"
           b"IDATx\x9cc\xfc\xff\xff?\x00\x05\xfe\x02\xfe\xa75\x81\x84\x00"
           b"\x00\x00\x00IEND\xaeB`\x82")
    (case / "images" / "det_x.png").write_bytes(png)
    return case


def test_results_pane_fake_case(qtbot, tmp_path):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = make_fake_case(tmp_path)
    pane.load_case(str(case))
    assert "completed" in pane.title.text()
    assert pane.summary.rowCount() == 1
    assert pane.summary.item(0, 1).text() == "5.4"
    assert "OK" in pane.audit.text()
    assert not pane.pv_btn.isEnabled()      # no viz/*.vtp in the fake


@pytest.mark.skipif(not os.path.isdir(REAL_CASE),
                    reason="phase-0 example case not present")
def test_results_pane_real_case(qtbot):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    pane.load_case(REAL_CASE)
    assert pane.summary.rowCount() >= 3     # example has 3 detectors
    assert pane.pv_btn.isEnabled()          # viz/rays.vtp exists
    files = paraview_launcher.viz_files(REAL_CASE)
    assert any(f.endswith("rays.vtp") for f in files)


def test_results_pane_monitor_reads_progress(qtbot, tmp_path):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = make_fake_case(tmp_path, status="estimated")
    (case / "progress.json").write_text(json.dumps(
        {"ev": "progress", "stage": "trace", "frac": 0.4,
         "msg": "seed 2/4", "status": "running"}))
    pane.load_case(str(case), monitor=True)
    assert "trace" in pane.title.text() and "40%" in pane.title.text()
    pane.stop_monitoring()


def test_problems_pane_canned_scene(qtbot):
    pane = ProblemsPane()
    qtbot.addWidget(pane)

    class FakeProject:
        structure = {"bodies": [
            {"name": "L", "label": "L", "tip": "Pad", "solid_closed": True,
             "placement": {"pos_mm": [0, 0, 0], "quat": [0, 0, 0, 1]},
             "properties": {"material": {"type": "t", "group": "Base",
                                         "value": "unobtanium"}}}]}
        def is_open(self):
            return True
    pane.project = FakeProject()
    blocked = []
    pane.validationChanged.connect(blocked.append)
    findings = pane.run_checks()
    assert pane.listw.count() == len(findings) > 0
    assert blocked == [True]      # unknown material + no source/detector
    assert "error" in pane.summary.text()


def test_problems_pane_deep_check_clean_scene(qtbot, monkeypatch):
    """Deep check on a clean scene appends an INFO Finding."""
    pane = ProblemsPane()
    qtbot.addWidget(pane)

    class FakeProject:
        structure = {"bodies": [
            {"name": "Laser", "label": "Laser", "tip": "Pad",
             "solid_closed": True, "placement": {"pos_mm": [0, 0, 0],
                                                  "quat": [0, 0, 0, 1]},
             "properties": {"power": {"type": "t", "group": "Base",
                                     "value": 5.0},
                           "lambdac": {"type": "t", "group": "Base",
                                      "value": 633.0}}},
            {"name": "Lens", "label": "Lens", "tip": "Pad",
             "solid_closed": True, "placement": {"pos_mm": [0, 0, 0],
                                                  "quat": [0, 0, 0, 1]},
             "properties": {"material": {"type": "t", "group": "Base",
                                        "value": "bk7"}}},
            {"name": "Screen", "label": "Screen", "tip": "Pad",
             "solid_closed": True, "placement": {"pos_mm": [0, 0, 0],
                                                  "quat": [0, 0, 0, 1]},
             "properties": {"material": {"type": "t", "group": "Base",
                                        "value": "detector"}}}]}
        doc = "fake_doc"
        class FakeFc:
            def request(self, op, kwargs):
                # Simulate successful check with no problems
                return {"invalid": [], "open_solids": [], "overlaps": [],
                        "face_pairs_checked": 42}
        fc = FakeFc()
        def is_open(self):
            return True
    pane.project = FakeProject()
    blocked = []
    pane.validationChanged.connect(blocked.append)
    findings = pane.run_checks(deep=True)
    # Should have runtime estimate info + deep check passed info
    info_findings = [f for f in findings if f.severity == "info"]
    assert any("Deep check passed" in f.message for f in info_findings), \
        [f.message for f in info_findings]
    assert "Deep check:" in pane.summary.text()
    # No errors
    assert "0 error(s)" in pane.summary.text()


def test_problems_pane_deep_check_worker_error(qtbot, monkeypatch):
    """Deep check with worker exception surfaces the error text."""
    pane = ProblemsPane()
    qtbot.addWidget(pane)

    class FakeProject:
        structure = {"bodies": [
            {"name": "Laser", "label": "Laser", "tip": "Pad",
             "solid_closed": True, "placement": {"pos_mm": [0, 0, 0],
                                                  "quat": [0, 0, 0, 1]},
             "properties": {"power": {"type": "t", "group": "Base",
                                     "value": 5.0},
                           "lambdac": {"type": "t", "group": "Base",
                                      "value": 633.0}}},
            {"name": "Screen", "label": "Screen", "tip": "Pad",
             "solid_closed": True, "placement": {"pos_mm": [0, 0, 0],
                                                  "quat": [0, 0, 0, 1]},
             "properties": {"material": {"type": "t", "group": "Base",
                                        "value": "detector"}}}]}
        doc = "fake_doc"
        class FakeFc:
            def request(self, op, kwargs):
                raise RuntimeError("FreeCAD worker timeout: connection lost")
        fc = FakeFc()
        def is_open(self):
            return True
    pane.project = FakeProject()
    findings = pane.run_checks(deep=True)
    # Should have warning with the error text
    warnings = [f for f in findings if f.severity == "warning"]
    assert any("connection lost" in f.message for f in warnings), \
        [f.message for f in warnings]
    assert "Deep check:" in pane.summary.text()


def test_problems_pane_deep_check_buttons_enabled_after(qtbot):
    """Deep check re-enables both buttons after completion."""
    pane = ProblemsPane()
    qtbot.addWidget(pane)

    class FakeProject:
        structure = {"bodies": [
            {"name": "Laser", "label": "Laser", "tip": "Pad",
             "solid_closed": True, "placement": {"pos_mm": [0, 0, 0],
                                                  "quat": [0, 0, 0, 1]},
             "properties": {"power": {"type": "t", "group": "Base",
                                     "value": 5.0},
                           "lambdac": {"type": "t", "group": "Base",
                                      "value": 633.0}}},
            {"name": "Screen", "label": "Screen", "tip": "Pad",
             "solid_closed": True, "placement": {"pos_mm": [0, 0, 0],
                                                  "quat": [0, 0, 0, 1]},
             "properties": {"material": {"type": "t", "group": "Base",
                                        "value": "detector"}}}]}
        doc = "fake_doc"
        class FakeFc:
            def request(self, op, kwargs):
                return {"invalid": [], "open_solids": [], "overlaps": [],
                        "face_pairs_checked": 0}
        fc = FakeFc()
        def is_open(self):
            return True
    pane.project = FakeProject()
    # Both buttons should be enabled before check
    assert pane.btn.isEnabled()
    assert pane.deep.isEnabled()
    pane.run_checks(deep=True)
    # Both buttons should be re-enabled after check
    assert pane.btn.isEnabled()
    assert pane.deep.isEnabled()


def test_paraview_find_and_viz_files(tmp_path):
    # sibling-of-pvpython discovery is machine-specific; just exercise
    # the no-files path
    assert paraview_launcher.viz_files(tmp_path) == []
    ok, msg = paraview_launcher.launch(tmp_path)
    assert not ok


def test_transform_panel_offscreen(qtbot, monkeypatch):
    from mieworkbench.panes import transform_panel as tp
    shown = []
    monkeypatch.setattr(tp.QMessageBox, "information",
                        lambda *a, **k: shown.append(a))
    monkeypatch.setattr(tp.QMessageBox, "warning",
                        lambda *a, **k: shown.append(a))
    panel = TransformPanel()
    qtbot.addWidget(panel)
    assert "No element" in panel.target.text()
    # apply with no project/selection: message box path, no exception
    panel._apply_translate()
    assert shown


def test_reference_picker_specs(qtbot):
    p = ReferencePointPicker("About:")
    qtbot.addWidget(p)
    assert p.spec() == {"kind": "origin"}
    p.kind.setCurrentIndex(1)     # fixed point
    p.point.setText("1, 2, 3")
    assert p.spec() == {"kind": "fixed", "point_mm": [1.0, 2.0, 3.0]}
    p.kind.setCurrentIndex(3)     # com
    spec = p.spec()
    assert spec["kind"] == "com"


def test_results_gallery_lightbox_click(qtbot, tmp_path):
    """Test that clicking a thumbnail opens the lightbox dialog."""
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = make_fake_case(tmp_path)
    pane.load_case(str(case))

    # Get the images gallery
    gallery = pane.galleries["images"]
    assert gallery._paths  # ensure images were loaded

    # Simulate clicking a thumbnail by calling the handler
    first_path = gallery._paths[0]
    gallery._thumbnail_clicked(first_path)

    # Verify lightbox was created and shown
    assert gallery._lightbox is not None
    assert gallery._lightbox.isVisible()
    assert os.path.basename(first_path) in gallery._lightbox.windowTitle()


def test_results_gallery_lightbox_arrow_keys(qtbot, tmp_path):
    """Test that arrow keys cycle through images in lightbox."""
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = make_fake_case(tmp_path)
    # Create multiple test images (remove the one from make_fake_case first)
    import shutil
    shutil.rmtree(case / "images")
    (case / "images").mkdir(parents=True)
    png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00"
           b"\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\r"
           b"IDATx\x9cc\xfc\xff\xff?\x00\x05\xfe\x02\xfe\xa75\x81\x84\x00"
           b"\x00\x00\x00IEND\xaeB`\x82")
    (case / "images" / "det_a.png").write_bytes(png)
    (case / "images" / "det_b.png").write_bytes(png)
    (case / "images" / "det_c.png").write_bytes(png)

    pane.load_case(str(case))
    gallery = pane.galleries["images"]
    assert len(gallery._paths) == 3

    # Open lightbox on first image
    first_path = gallery._paths[0]
    gallery._thumbnail_clicked(first_path)
    lightbox = gallery._lightbox

    # Initial index should be 0
    assert lightbox.current_index == 0
    initial_title = lightbox.windowTitle()

    # Press Right arrow
    from PySide6.QtGui import QKeyEvent
    right_event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Right, Qt.NoModifier)
    lightbox.keyPressEvent(right_event)
    assert lightbox.current_index == 1
    assert lightbox.windowTitle() != initial_title

    # Press Right arrow again
    lightbox.keyPressEvent(right_event)
    assert lightbox.current_index == 2

    # Press Right arrow (should wrap around)
    lightbox.keyPressEvent(right_event)
    assert lightbox.current_index == 0

    # Press Left arrow
    left_event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Left, Qt.NoModifier)
    lightbox.keyPressEvent(left_event)
    assert lightbox.current_index == 2


def test_results_gallery_lightbox_esc_close(qtbot, tmp_path):
    """Test that Esc key closes the lightbox."""
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = make_fake_case(tmp_path)
    pane.load_case(str(case))

    gallery = pane.galleries["images"]
    first_path = gallery._paths[0]
    gallery._thumbnail_clicked(first_path)
    lightbox = gallery._lightbox

    assert lightbox.isVisible()

    # Simulate Esc key press
    from PySide6.QtGui import QKeyEvent
    esc_event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    lightbox.keyPressEvent(esc_event)

    assert not lightbox.isVisible()


def test_results_gallery_lightbox_refresh_doesnt_crash(qtbot, tmp_path):
    """Test that gallery refresh with open lightbox doesn't crash."""
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = make_fake_case(tmp_path)
    pane.load_case(str(case))

    gallery = pane.galleries["images"]
    first_path = gallery._paths[0]
    gallery._thumbnail_clicked(first_path)
    lightbox = gallery._lightbox

    assert lightbox.isVisible()

    # Add more images and refresh
    png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00"
           b"\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\r"
           b"IDATx\x9cc\xfc\xff\xff?\x00\x05\xfe\x02\xfe\xa75\x81\x84\x00"
           b"\x00\x00\x00IEND\xaeB`\x82")
    (case / "images" / "det_extra.png").write_bytes(png)

    # Refresh gallery
    pane.refresh()

    # Lightbox should still be visible and not crash
    assert lightbox.isVisible()
    # The lightbox should have updated its paths and clamped the index
    assert len(lightbox.paths) == 2
    assert lightbox.current_index < len(lightbox.paths)


def test_results_power_tab_from_report(qtbot, tmp_path):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = make_fake_case(tmp_path)
    report = json.loads((case / "report.json").read_text())
    report["elements"] = {
        "Lens": {"power_in_W": 0.001, "power_out_W": 0.0008,
                 "absorbed_W": 0.0002, "detected_W": 0.0},
        "Screen": {"power_in_W": 0.0008, "power_out_W": 0.00026,
                   "absorbed_W": 0.0, "detected_W": 0.00054},
    }
    (case / "report.json").write_text(json.dumps(report))
    pane.load_case(str(case))
    assert pane.power.rowCount() == 2
    row = {pane.power.item(r, 0).text(): r
           for r in range(pane.power.rowCount())}
    assert pane.power.item(row["Lens"], 1).text() == "1"        # 1 mW in
    assert pane.power.item(row["Lens"], 3).text() == "0.2"      # absorbed
    assert pane.power.item(row["Screen"], 4).text() == "0.54"   # detected


def test_results_power_tab_falls_back_to_audit(qtbot, tmp_path):
    """Old cases: report.json has no 'elements' but audit.json exists —
    the pane mines it with common.element_power_table."""
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = make_fake_case(tmp_path)
    (case / "audit.json").write_text(json.dumps({
        "gate": 1e-3,
        "per_seed": [{
            "sources": {}, "closure_gate": 1e-3, "closure_ok": True,
            "by_body_W": {"Lens": 0.0002},
            "by_surface_W": {"Det.Pad.Face1": 0.00054},
            "element_flux_W": {"Lens": {"in_W": 0.001, "out_W": 0.0008}},
            "detected_W": {"Det.Pad.Face1": 0.00054},
        }]}))
    pane.load_case(str(case))
    labels = {pane.power.item(r, 0).text()
              for r in range(pane.power.rowCount())}
    assert labels == {"Lens", "Det"}
    row = {pane.power.item(r, 0).text(): r
           for r in range(pane.power.rowCount())}
    assert pane.power.item(row["Det"], 4).text() == "0.54"
    assert pane.power.item(row["Lens"], 3).text() == "0.2"
