"""Offscreen tests for the Results pane's "Analysis" and "Sources" tabs
(report.json's optional per-detector 'analysis' block and 'per_source'
list). Old cases carry neither -- both tabs must load with zero rows,
never raise. See CLAUDE.md's GUI round notes for the report.json shape
and the shared save/export-CSV seams reused here.
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


def _make_case_with_analysis_and_sources(tmp_path):
    case = make_fake_case(tmp_path)
    report = json.loads((case / "report.json").read_text())
    report["detectors"]["Screen.Face5"]["analysis"] = {
        "strehl": 0.93,
        "rms_waves": 0.045,
        "mtf50": {"tan": 42.1, "sag": 39.8},
        "ee": {"r50_um": 3.2, "r80_um": 6.7, "r90_um": 9.1},
        "spot_rms": {
            "Laser@633": {"rms_waves": 0.021},
            "Laser@532": {"rms_waves": 0.030},
        },
    }
    report["detectors"]["Screen.Face5"]["per_source"] = [
        {"source": "Laser", "lam_stratum": "633nm", "pol_stratum": "s",
         "coherent_W": 0.002, "incoherent_W": 0.0001},
        {"source": "Laser", "lam_stratum": "532nm", "pol_stratum": "p",
         "coherent_W": 0.0015, "incoherent_W": 0.0},
    ]
    (case / "report.json").write_text(json.dumps(report))
    (case / "analysis").mkdir(parents=True)
    (case / "analysis" / "psf_Screen.png").write_bytes(_PNG)
    return case


def test_analysis_tab_metrics_and_gallery(qtbot, tmp_path):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = _make_case_with_analysis_and_sources(tmp_path)
    pane.load_case(str(case))

    # gallery picked up the analysis/*.png
    assert pane.galleries["analysis"]._paths == [
        str(case / "analysis" / "psf_Screen.png")]

    # metrics table flattened every scalar leaf of the analysis block
    rows = {(pane.analysis_metrics.item(r, 0).text(),
            pane.analysis_metrics.item(r, 1).text()):
           pane.analysis_metrics.item(r, 2).text()
           for r in range(pane.analysis_metrics.rowCount())}
    assert rows[("Screen.Face5", "strehl")] == "0.93"
    assert rows[("Screen.Face5", "rms_waves")] == "0.045"
    assert rows[("Screen.Face5", "mtf50.tan")] == "42.1"
    assert rows[("Screen.Face5", "mtf50.sag")] == "39.8"
    assert rows[("Screen.Face5", "ee.r50_um")] == "3.2"
    assert rows[("Screen.Face5", "ee.r80_um")] == "6.7"
    assert rows[("Screen.Face5", "ee.r90_um")] == "9.1"
    assert rows[("Screen.Face5", "spot_rms.Laser@633.rms_waves")] == "0.021"
    assert rows[("Screen.Face5", "spot_rms.Laser@532.rms_waves")] == "0.03"


def test_analysis_tab_empty_when_absent(qtbot, tmp_path):
    """Old cases (no 'analysis' key at all) load with zero rows and no
    crash -- the block is entirely optional."""
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = make_fake_case(tmp_path)
    pane.load_case(str(case))
    assert pane.analysis_metrics.rowCount() == 0
    assert pane.galleries["analysis"]._paths == []


def test_sources_tab_from_report(qtbot, tmp_path):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = _make_case_with_analysis_and_sources(tmp_path)
    pane.load_case(str(case))

    assert pane.sources.rowCount() == 2
    row0 = [pane.sources.item(0, c).text() for c in range(7)]
    assert row0[0] == "Screen.Face5"
    assert row0[1] == "Laser"
    assert row0[2] == "633nm"
    assert row0[3] == "s"
    assert row0[4] == "2"          # 0.002 W -> 2 mW coherent
    assert row0[5] == "0.1"        # 0.0001 W -> 0.1 mW incoherent
    assert row0[6] == "2.1"        # total mW

    row1 = [pane.sources.item(1, c).text() for c in range(7)]
    assert row1[1] == "Laser"
    assert row1[2] == "532nm"
    assert row1[4] == "1.5"
    assert row1[5] == "0"
    assert row1[6] == "1.5"


def test_sources_tab_empty_when_absent(qtbot, tmp_path):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = make_fake_case(tmp_path)
    pane.load_case(str(case))
    assert pane.sources.rowCount() == 0


def test_analysis_and_sources_cleared_by_clear_case(qtbot, tmp_path):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = _make_case_with_analysis_and_sources(tmp_path)
    pane.load_case(str(case))
    assert pane.analysis_metrics.rowCount() > 0
    assert pane.sources.rowCount() > 0
    assert pane.galleries["analysis"]._paths

    pane.clear_case()
    assert pane.analysis_metrics.rowCount() == 0
    assert pane.sources.rowCount() == 0
    assert pane.galleries["analysis"]._grid.count() == 0
    assert pane.galleries["analysis"]._paths == []


def test_analysis_metrics_export_csv(qtbot, tmp_path, monkeypatch):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = _make_case_with_analysis_and_sources(tmp_path)
    pane.load_case(str(case))

    dest = tmp_path / "analysis_out.csv"
    monkeypatch.setattr(results, "_choose_save_path",
                        lambda parent, default_name: str(dest))
    menu = pane._build_table_export_menu(
        pane.analysis_metrics, "analysis_metrics.csv")
    menu.actions()[0].trigger()

    with open(dest, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["Detector", "Metric", "Value"]
    assert len(rows) == 1 + pane.analysis_metrics.rowCount()


def test_sources_export_csv(qtbot, tmp_path, monkeypatch):
    pane = ResultsPane()
    qtbot.addWidget(pane)
    case = _make_case_with_analysis_and_sources(tmp_path)
    pane.load_case(str(case))

    dest = tmp_path / "sources_out.csv"
    monkeypatch.setattr(results, "_choose_save_path",
                        lambda parent, default_name: str(dest))
    menu = pane._build_table_export_menu(pane.sources, "sources.csv")
    menu.actions()[0].trigger()

    with open(dest, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["Detector", "Source", "λ stratum", "Pol stratum",
                       "Coherent [mW]", "Incoherent [mW]", "Total [mW]"]
    assert len(rows) == 1 + pane.sources.rowCount()
