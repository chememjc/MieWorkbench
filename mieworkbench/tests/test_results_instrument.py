"""Offscreen tests for the Results pane's "Instrument" tab (engine3.md
P2.5 §9 virtual instrument layer): report.json's optional per-detector
'instrument' block + <case>/instrument/*.png auto-glob gallery. Old cases
carry neither -- the tab must load with zero rows/images, never raise.
Mirrors test_results_analysis_sources.py's Analysis-tab test shape.
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.panes import results  # noqa: E402
from mieworkbench.panes.results import ResultsPane  # noqa: E402
from mieworkbench.tests.test_results_problems_panels import (  # noqa: E402
    make_fake_case)

# same tiny valid PNG literal the other results/compare tests use
_PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00"
        b"\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\r"
        b"IDATx\x9cc\xfc\xff\xff?\x00\x05\xfe\x02\xfe\xa75\x81\x84\x00"
        b"\x00\x00\x00IEND\xaeB`\x82")


def _make_case_with_instrument(tmp_path):
    case = make_fake_case(tmp_path)
    report = json.loads((case / "report.json").read_text())
    report["detectors"]["Screen.Face5"]["instrument"] = {
        "row": "camera_generic", "class": "camera", "mode": "full",
        "seed": 123456789,
        "integration_time_s": 0.01,
        "saturation_fraction": 0.02,
        "mean_counts": 812.4,
        "max_counts": 4095.0,
        "snr_estimate": 118.7,
    }
    (case / "report.json").write_text(json.dumps(report))
    idir = case / "instrument"
    idir.mkdir(parents=True)
    (idir / "instr_Screen_Face5_camera_full.png").write_bytes(_PNG)
    (idir / "instr_Screen_Face5_camera_full_counts.npy").write_bytes(b"\x00")
    return case


def test_instrument_tab_metrics_and_gallery(qtbot, tmp_path):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = _make_case_with_instrument(tmp_path)
    pane.load_case(str(case))

    # gallery picked up instrument/*.png (the .npy is not a gallery image)
    idir = case / "instrument"
    assert pane.galleries["instrument"]._paths == [
        str(idir / "instr_Screen_Face5_camera_full.png")]

    rows = {(pane.instrument_metrics.item(r, 0).text(),
            pane.instrument_metrics.item(r, 1).text()):
           pane.instrument_metrics.item(r, 2).text()
           for r in range(pane.instrument_metrics.rowCount())}
    assert rows[("Screen.Face5", "row")] == "camera_generic"
    assert rows[("Screen.Face5", "class")] == "camera"
    assert rows[("Screen.Face5", "mode")] == "full"
    assert rows[("Screen.Face5", "seed")] == "123456789"
    assert rows[("Screen.Face5", "saturation_fraction")] == "0.02"
    assert rows[("Screen.Face5", "mean_counts")] == "812.4"
    assert rows[("Screen.Face5", "max_counts")] == "4095"
    assert rows[("Screen.Face5", "snr_estimate")] == "118.7"

    labels = [pane.tabs.tabText(i) for i in range(pane.tabs.count())]
    assert "Instrument" in labels


def test_instrument_tab_empty_when_absent(qtbot, tmp_path):
    """Old cases (no 'instrument' key at all, or a detector never assigned
    an instrument) load with zero rows and no crash."""
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = make_fake_case(tmp_path)
    pane.load_case(str(case))
    assert pane.instrument_metrics.rowCount() == 0
    assert pane.galleries["instrument"]._paths == []


def test_instrument_tab_cleared_by_clear_case(qtbot, tmp_path):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = _make_case_with_instrument(tmp_path)
    pane.load_case(str(case))
    assert pane.instrument_metrics.rowCount() > 0
    assert pane.galleries["instrument"]._paths

    pane.clear_case()
    assert pane.instrument_metrics.rowCount() == 0
    assert pane.galleries["instrument"]._grid.count() == 0
    assert pane.galleries["instrument"]._paths == []


def test_instrument_metrics_export_csv(qtbot, tmp_path, monkeypatch):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = _make_case_with_instrument(tmp_path)
    pane.load_case(str(case))

    dest = tmp_path / "instrument_out.csv"
    monkeypatch.setattr(results, "_choose_save_path",
                        lambda parent, default_name: str(dest))
    menu = pane._build_table_export_menu(
        pane.instrument_metrics, "instrument_metrics.csv")
    menu.actions()[0].trigger()

    with open(dest, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["Detector", "Metric", "Value"]
    assert len(rows) == 1 + pane.instrument_metrics.rowCount()


def test_instrument_tab_multiple_detectors_and_classes(qtbot, tmp_path):
    """A powermeter block (no counts/saturation fields) coexists cleanly
    with a camera block on a different detector -- the flattener is
    schema-agnostic per detector, matching how a real bench-comparison
    scene tags two different detector bodies with two different
    instrument classes."""
    case = make_fake_case(tmp_path)
    report = json.loads((case / "report.json").read_text())
    report["detectors"]["Screen.Face5"]["instrument"] = {
        "row": "camera_generic", "class": "camera", "mode": "ideal",
        "mean_counts": 500.0,
    }
    report["detectors"]["Meter.Face1"] = {
        "total_power_W": 1.0e-3, "peak_irradiance_W_m2": 0.0,
        "pixel_um": 0.0, "resolution": [1, 1],
        "instrument": {
            "row": "powermeter_generic", "class": "powermeter",
            "mode": "ideal", "power_reported_W": 9.87e-4,
            "power_reported_display": "0.000987",
        },
    }
    (case / "report.json").write_text(json.dumps(report))
    pane = ResultsPane()
    qtbot.addWidget(pane)
    pane.load_case(str(case))

    rows = {(pane.instrument_metrics.item(r, 0).text(),
            pane.instrument_metrics.item(r, 1).text()):
           pane.instrument_metrics.item(r, 2).text()
           for r in range(pane.instrument_metrics.rowCount())}
    assert rows[("Screen.Face5", "class")] == "camera"
    assert rows[("Screen.Face5", "mean_counts")] == "500"
    assert rows[("Meter.Face1", "class")] == "powermeter"
    assert rows[("Meter.Face1", "power_reported_display")] == "0.000987"
    # ideal mode: no 'seed' key on either detector
    assert ("Screen.Face5", "seed") not in rows
    assert ("Meter.Face1", "seed") not in rows
