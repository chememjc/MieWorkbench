"""Offscreen tests for the results pane, problems pane, transform panel,
and paraview launcher helpers."""

import json
import os
import sys

import pytest

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
