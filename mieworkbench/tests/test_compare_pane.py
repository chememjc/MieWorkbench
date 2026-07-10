"""Offscreen tests for ComparePane: synthesize a fake summary.json + tiny
PNGs (no real compare_sweep.py run — that's covered, gated on the optics
env, by test_compare_sweep_backend.py) and drive the pane's public API.

QProcess launches are captured by monkeypatching the pane's
_start_process() seam (see compare_pane.py's docstring) rather than the
real QProcess machinery — compare_sweep.py needs the optics env, which
this fast offline suite must not depend on. _on_finished() is dialog-free
by design so tests can call it directly to exercise the summary-load path
after faking a successful run.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

import common  # noqa: E402  (stdlib-only shared contract hub)

from mieworkbench.panes.compare_pane import ComparePane  # noqa: E402

# same tiny valid PNG literal test_results_problems_panels.py uses
_PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00"
        b"\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\r"
        b"IDATx\x9cc\xfc\xff\xff?\x00\x05\xfe\x02\xfe\xa75\x81\x84\x00"
        b"\x00\x00\x00IEND\xaeB`\x82")


def _write_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG)


def make_fake_summary(out_dir):
    """A 2-variant, 1-detector manifest-mode summary.json + the PNGs it
    references, matching compare_sweep.py's documented schema."""
    out_dir.mkdir(parents=True, exist_ok=True)
    detectors = {
        "D": {"total_power_W": 1e-6, "peak_irradiance_W_m2": 400.0,
             "profile_visibility": 0.9, "centroid_x_mm": 0.025,
             "centroid_y_mm": 0.025, "rms_spot_radius_mm": 0.0},
    }
    summary = {
        "mode": "product", "model": "modelA", "case": "quick",
        "ref": "modelA-gap10", "order": ["miewb_vars.gap"],
        "variables_varying": ["miewb_vars.gap"],
        "variants": [
            {"stem": "modelA-gap10", "values": {"miewb_vars.gap": 10.0},
             "case_dir": "/x/modelA-gap10/quick", "detectors": detectors},
            {"stem": "modelA-gap20", "values": {"miewb_vars.gap": 20.0},
             "case_dir": "/x/modelA-gap20/quick", "detectors": detectors},
        ],
        "plots": ["plot_total_power_W_D_vs_miewb_vars_gap.png"],
        "gallery": {"D": [
            {"stem": "modelA-gap10", "values_label": "gap=10",
             "image": "gallery/modelA-gap10_D.png"},
            {"stem": "modelA-gap20", "values_label": "gap=20",
             "image": "gallery/modelA-gap20_D.png"},
        ]},
        "diffs": {"D": [
            {"stem": "modelA-gap20", "values_label": "gap=20",
             "image": "diff/modelA-gap20_D.png"},
        ]},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary))
    _write_png(out_dir / "plot_total_power_W_D_vs_miewb_vars_gap.png")
    _write_png(out_dir / "gallery" / "modelA-gap10_D.png")
    _write_png(out_dir / "gallery" / "modelA-gap20_D.png")
    _write_png(out_dir / "diff" / "modelA-gap20_D.png")
    return summary


# ---------------------------------------------------------------------------
# empty state
# ---------------------------------------------------------------------------
def test_empty_state_before_any_summary(qtbot):
    pane = ComparePane()
    qtbot.addWidget(pane)
    assert pane.stack.currentWidget() is pane.empty_label
    assert "Run a sweep" in pane.empty_label.text()
    assert not pane.compare_cases_btn.isEnabled()


# ---------------------------------------------------------------------------
# load_summary populates every tab
# ---------------------------------------------------------------------------
def test_load_summary_populates_tabs(qtbot, tmp_path):
    out_dir = tmp_path / "out"
    make_fake_summary(out_dir)

    pane = ComparePane()
    qtbot.addWidget(pane)
    ok = pane.load_summary(out_dir)
    assert ok
    assert pane.stack.currentWidget() is pane.tabs

    # Metrics tab: 2 variants x 1 detector = 2 rows
    assert pane.metrics_table.rowCount() == 2
    assert pane.metrics_gallery._paths      # the metric-vs-var plot thumb

    # Images tab: single detector "D", both variant renders shown
    assert [pane.images_detector_combo.itemText(i)
           for i in range(pane.images_detector_combo.count())] == ["D"]
    assert len(pane.images_gallery._paths) == 2

    # Difference tab: ref combo defaults to summary["ref"], diff gallery
    # has only the non-ref variant
    assert pane.ref_combo.currentText() == "modelA-gap10"
    assert [pane.diff_detector_combo.itemText(i)
           for i in range(pane.diff_detector_combo.count())] == ["D"]
    assert len(pane.diff_gallery._paths) == 1

    # Scrub tab: slider spans variant count - 1, detector combo populated
    assert pane.scrub_slider.maximum() == 1
    assert [pane.scrub_detector_combo.itemText(i)
           for i in range(pane.scrub_detector_combo.count())] == ["D"]
    assert "modelA-gap10" in pane.scrub_caption.text()


def test_load_summary_missing_file_is_a_noop(qtbot, tmp_path):
    pane = ComparePane()
    qtbot.addWidget(pane)
    ok = pane.load_summary(tmp_path / "nowhere")
    assert not ok
    assert pane.stack.currentWidget() is pane.empty_label


# ---------------------------------------------------------------------------
# add_case flow
# ---------------------------------------------------------------------------
def test_add_case_enables_button_and_builds_argv(qtbot, tmp_path, monkeypatch):
    pane = ComparePane()
    qtbot.addWidget(pane)
    assert not pane.compare_cases_btn.isEnabled()

    case_a = tmp_path / "results" / "foo" / "quick"
    case_b = tmp_path / "results" / "bar" / "normal"
    pane.add_case(str(case_a))
    assert pane.compare_cases_btn.isEnabled()
    assert "1" in pane.compare_cases_btn.text()
    pane.add_case(str(case_b))
    assert "2" in pane.compare_cases_btn.text()

    captured = {}

    def fake_start(argv):
        captured["argv"] = argv
        return True

    monkeypatch.setattr(pane, "_start_process", fake_start)
    qtbot.mouseClick(pane.compare_cases_btn,
                     __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.LeftButton)

    argv = captured["argv"]
    assert "--cases" in argv
    i = argv.index("--cases")
    assert argv[i + 1:i + 3] == [str(case_a), str(case_b)]
    assert "--out" in argv


# ---------------------------------------------------------------------------
# run_compare() argv construction (manifest mode)
# ---------------------------------------------------------------------------
def test_run_compare_manifest_default_out_and_argv(qtbot, tmp_path,
                                                    monkeypatch):
    manifest_path = tmp_path / "results" / "modelA" / "sweep-quick.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(
        {"model": "modelA", "case": "quick", "mode": "product",
        "order": [], "variants": []}))

    pane = ComparePane()
    qtbot.addWidget(pane)

    captured = {}

    def fake_start(argv):
        captured["argv"] = argv
        return True

    monkeypatch.setattr(pane, "_start_process", fake_start)
    ok = pane.run_compare(manifest_path=str(manifest_path))
    assert ok

    argv = captured["argv"]
    assert argv[1].endswith("compare_sweep.py")
    assert "--manifest" in argv
    assert argv[argv.index("--manifest") + 1] == str(manifest_path)
    expected_out = str(common.RESULTS_DIR / "comparisons"
                       / "sweep_modelA_quick")
    assert argv[argv.index("--out") + 1] == expected_out


def test_run_compare_refused_while_running(qtbot, monkeypatch):
    pane = ComparePane()
    qtbot.addWidget(pane)
    monkeypatch.setattr(pane, "is_running", lambda: True)
    monkeypatch.setattr(pane, "_start_process",
                        lambda argv: pytest.fail("should not be called"))
    ok = pane.run_compare(case_dirs=["/x/a/quick"])
    assert not ok


# ---------------------------------------------------------------------------
# _on_finished() dialog-free success path
# ---------------------------------------------------------------------------
def test_on_finished_success_loads_summary(qtbot, tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    make_fake_summary(out_dir)

    pane = ComparePane()
    qtbot.addWidget(pane)
    monkeypatch.setattr(pane, "_start_process", lambda argv: True)

    finished_events = []
    pane.compareFinished.connect(finished_events.append)

    ok = pane.run_compare(case_dirs=["/x/a/quick"], out_dir=str(out_dir))
    assert ok
    assert pane.stack.currentWidget() is pane.empty_label   # not yet loaded

    pane._on_finished(0)
    assert finished_events == [True]
    assert pane.stack.currentWidget() is pane.tabs
    assert pane.metrics_table.rowCount() == 2


def test_on_finished_failure_keeps_empty_state(qtbot, tmp_path, monkeypatch):
    pane = ComparePane()
    qtbot.addWidget(pane)
    monkeypatch.setattr(pane, "_start_process", lambda argv: True)

    finished_events = []
    pane.compareFinished.connect(finished_events.append)

    pane.run_compare(case_dirs=["/x/a/quick"], out_dir=str(tmp_path / "out"))
    pane._on_finished(1)
    assert finished_events == [False]
    assert pane.stack.currentWidget() is pane.empty_label


# ---------------------------------------------------------------------------
# Difference tab ref selector re-runs with the new --ref
# ---------------------------------------------------------------------------
def test_ref_combo_change_triggers_rerun_with_new_ref(qtbot, tmp_path,
                                                      monkeypatch):
    out_dir = tmp_path / "out"
    make_fake_summary(out_dir)

    manifest_path = tmp_path / "sweep-quick.manifest.json"
    manifest_path.write_text(json.dumps(
        {"model": "modelA", "case": "quick", "mode": "product",
        "order": [], "variants": []}))

    pane = ComparePane()
    qtbot.addWidget(pane)
    monkeypatch.setattr(pane, "_start_process", lambda argv: True)
    pane.run_compare(manifest_path=str(manifest_path), out_dir=str(out_dir))
    pane.load_summary(out_dir)   # populates ref_combo with "modelA-gap10"
    assert pane.ref_combo.currentText() == "modelA-gap10"

    captured = {}

    def fake_start(argv):
        captured["argv"] = argv
        return True

    monkeypatch.setattr(pane, "_start_process", fake_start)
    pane.ref_combo.setCurrentText("modelA-gap20")

    argv = captured["argv"]
    assert "--ref" in argv
    assert argv[argv.index("--ref") + 1] == "modelA-gap20"
    assert "--manifest" in argv
    assert argv[argv.index("--manifest") + 1] == str(manifest_path)
    assert argv[argv.index("--out") + 1] == str(out_dir)
