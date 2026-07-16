"""GUI surface for the P1 chunked-run contract (checkpoint/resume/extend).

Four layers, fastest first:
  - core.checkpointinfo: pure-Python resume_state/extend_state/spr/preset-
    tag/config-building tests against hand-built case dirs (no Qt, no
    subprocess).
  - RunController.start_resume/start_extend: argv construction, verified
    by monkeypatching self.start() to record its call (no subprocess).
  - RunDialog's extend mode: widget tests exactly like test_rundialog.py's
    pure-widget layer (construct directly, drive the spin box/buttons,
    never call .exec()).
  - MainWindow._on_resume_run/_on_extend_run wiring: monkeypatch the
    runner + RunDialog.exec, following test_rundialog.py's MainWindow-
    wiring idiom.
  - ONE real end-to-end smoke test: a module-scope fixture traces a tiny
    doubleslit case for real (subprocess, C engine, --no-gather-gate) to
    get a completed checkpoint fast, then drives RunController.start_extend
    for real (qtbot.waitSignal) and checks checkpoint.json's target_rays
    doubled. Skipped without the built C engine + extracted
    geometry/doubleslit (this worktree may lack geometry/ -- see
    CLAUDE.md's pinned-interpreter table; copy it from the main checkout
    as ad hoc test setup, never commit it).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

import common                                               # noqa: E402
# this worktree's cengine/build/ is gitignored and not built per-worktree;
# fall back to the shared machine build (CLAUDE.md's pinned-tool
# convention) unless the environment already names one.
os.environ.setdefault(
    "MIEWB_CENGINE", "/home3/raytracegui/cengine/build/miewb-trace")
from raytracer import cengine                               # noqa: E402
from PySide6.QtCore import Qt                                # noqa: E402
from PySide6.QtWidgets import QDialog, QDialogButtonBox      # noqa: E402

from mieworkbench.core import checkpointinfo                # noqa: E402
from mieworkbench.core.runner import RunController           # noqa: E402
from mieworkbench.mainwindow import MainWindow                # noqa: E402
from mieworkbench.panes.rundialog import RunDialog            # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
MODEL_JSON = REPO_ROOT / "geometry" / "doubleslit" / "model.json"
FCSTD = REPO_ROOT / "doubleslit.FCStd"


# ===========================================================================
# 1. core.checkpointinfo -- pure Python, hand-built case dirs
# ===========================================================================
def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _tracing_checkpoint(case_dir, target_rays=8000, n_chunks=2):
    _write_json(case_dir / "cengine" / "checkpoint.json", {
        "schema_version": 1, "status": "tracing",
        "target_rays": target_rays, "seeds": 1, "seed0": 42,
        "chunks": [{"seed": 42, "lo": i * 2000, "hi": (i + 1) * 2000}
                  for i in range(n_chunks)],
        "extensions": []})


def _completed_case(case_dir, target_rays=8000, n_samples_per_key=1500):
    _write_json(case_dir / "cengine" / "checkpoint.json", {
        "schema_version": 1, "status": "completed",
        "target_rays": target_rays, "seeds": 1, "seed0": 42,
        "chunks": [{"seed": 42, "lo": 0, "hi": target_rays}],
        "extensions": []})
    _write_json(case_dir / "case.json", {
        "status": "completed",
        "options": {"rays": target_rays, "resolution": 64, "nlambda": 3,
                   "spectral_bins": 4, "seeds": 1, "backend": "auto"},
        "gather": {"seed42": {"Screen.Face5": {
            "0/0/0": {"n_samples": n_samples_per_key},
            "0/0/1": {"n_samples": n_samples_per_key}}}}})


def test_resume_state_none_for_missing_checkpoint(tmp_path):
    assert checkpointinfo.resume_state(tmp_path / "nope") is None


def test_resume_state_dead_tracing_case(tmp_path):
    case = tmp_path / "quick"
    _tracing_checkpoint(case, target_rays=8000, n_chunks=2)
    state = checkpointinfo.resume_state(case)
    assert state == {"target_rays": 8000, "n_chunks": 2}


def test_resume_state_none_when_status_not_tracing(tmp_path):
    case = tmp_path / "quick"
    _completed_case(case)
    assert checkpointinfo.resume_state(case) is None


def test_resume_state_none_when_case_is_live(tmp_path):
    case = tmp_path / "quick"
    _tracing_checkpoint(case)
    common.acquire_case_lock(case)
    try:
        assert checkpointinfo.resume_state(case) is None
    finally:
        common.release_case_lock(case)


def test_extend_state_none_for_missing_checkpoint(tmp_path):
    assert checkpointinfo.extend_state(tmp_path / "nope") is None


def test_extend_state_none_when_checkpoint_not_completed(tmp_path):
    case = tmp_path / "quick"
    _tracing_checkpoint(case)
    assert checkpointinfo.extend_state(case) is None


def test_extend_state_measures_spr_from_gather_diagnostics(tmp_path):
    case = tmp_path / "quick"
    _completed_case(case, target_rays=8000, n_samples_per_key=1500)
    state = checkpointinfo.extend_state(case)
    assert state["current_rays"] == 8000
    # 3000 total surviving samples (2 keys x 1500) / 8000 rays (1 seed)
    assert state["spr"] == pytest.approx(3000 / 8000)
    assert state["m_eff_proxy"] == pytest.approx(state["spr"] * 8000)


def test_measured_spr_falls_back_to_default_with_no_samples():
    assert checkpointinfo.measured_spr({"gather": {}}, 1000) \
        == common.DEFAULT_SPR
    assert checkpointinfo.measured_spr(None, 1000) == common.DEFAULT_SPR


def test_resolve_preset_tag_splits_case_name(tmp_path):
    assert checkpointinfo.resolve_preset_tag(tmp_path / "quick") \
        == ("quick", None)
    assert checkpointinfo.resolve_preset_tag(tmp_path / "normal-foo") \
        == ("normal", "foo")
    assert checkpointinfo.resolve_preset_tag(tmp_path / "detailed-a-b") \
        == ("detailed", "a-b")
    # unrecognized name -> best-effort fallback, never a crash
    assert checkpointinfo.resolve_preset_tag(tmp_path / "weirdname") \
        == ("quick", None)


def test_build_resume_config_pins_checkpoint_target_rays(tmp_path):
    case = tmp_path / "normal-foo"
    _tracing_checkpoint(case, target_rays=12000, n_chunks=1)
    config = checkpointinfo.build_resume_config(case)
    assert config["resume"] is True
    assert config["engine"] == "c"
    assert config["rays"] == 12000.0
    assert config["preset"] == "normal"
    assert config["tag"] == "foo"


def test_build_resume_config_carries_case_options(tmp_path):
    case = tmp_path / "quick"
    _completed_case(case, target_rays=8000)
    config = checkpointinfo.build_resume_config(case)
    assert config["resolution"] == 64
    assert config["nlambda"] == 3
    assert config["spectral_bins"] == 4
    assert config["seeds"] == 1


def test_build_extend_config_sets_extend_and_drops_resume(tmp_path):
    case = tmp_path / "quick"
    _completed_case(case, target_rays=8000)
    config = checkpointinfo.build_extend_config(case, 16000)
    assert "resume" not in config
    assert config["extend"] == 16000.0
    assert config["engine"] == "c"


# ===========================================================================
# 2. RunController.start_resume/start_extend -- argv construction only
# ===========================================================================
def test_start_resume_builds_argv_and_calls_start(tmp_path, monkeypatch):
    case = tmp_path / "quick-tag1"
    _tracing_checkpoint(case, target_rays=9000, n_chunks=1)

    calls = []
    ctl = RunController()
    monkeypatch.setattr(
        ctl, "start",
        lambda model_path, extra_args=None, steps=None, extra_env=None:
            calls.append((model_path, extra_args, steps, extra_env))
            or True)

    ok = ctl.start_resume("model.FCStd", case)
    assert ok is True
    assert len(calls) == 1
    model_path, extra_args, steps, extra_env = calls[0]
    assert model_path == "model.FCStd"
    assert steps == "trace,post,viz"
    assert "--resume" in extra_args
    assert "--rays" in extra_args
    assert extra_args[extra_args.index("--rays") + 1] == "9000.0"
    assert "--tag" in extra_args
    assert extra_args[extra_args.index("--tag") + 1] == "tag1"


def test_start_extend_builds_argv_and_calls_start(tmp_path, monkeypatch):
    case = tmp_path / "quick"
    _completed_case(case, target_rays=8000)

    calls = []
    ctl = RunController()
    monkeypatch.setattr(
        ctl, "start",
        lambda model_path, extra_args=None, steps=None, extra_env=None:
            calls.append((model_path, extra_args, steps, extra_env))
            or True)

    ok = ctl.start_extend("model.FCStd", case, 16000)
    assert ok is True
    model_path, extra_args, steps, extra_env = calls[0]
    assert steps == "trace,post,viz"
    assert "--resume" not in extra_args
    assert "--extend" in extra_args
    assert extra_args[extra_args.index("--extend") + 1] == "16000.0"


def test_start_extend_honors_steps_override(tmp_path, monkeypatch):
    case = tmp_path / "quick"
    _completed_case(case, target_rays=8000)
    calls = []
    ctl = RunController()
    monkeypatch.setattr(
        ctl, "start",
        lambda model_path, extra_args=None, steps=None, extra_env=None:
            calls.append(steps) or True)
    ctl.start_extend("model.FCStd", case, 16000, steps="trace")
    assert calls == ["trace"]


def test_start_resume_single_flight_refused_when_running(tmp_path,
                                                          monkeypatch):
    case = tmp_path / "quick"
    _tracing_checkpoint(case)
    ctl = RunController()
    monkeypatch.setattr(ctl, "is_running", lambda: True)
    assert ctl.start_resume("model.FCStd", case) is False


# ===========================================================================
# 3. RunDialog extend mode -- pure widget tests
# ===========================================================================
EXTEND_CTX = {
    "current_rays": 8000,
    "spr": 0.375,
    "resolution": 64,
    "nlambda": 3,
    "backend": "c",
    "n_coherent_sources": 1,
    "model_stem": "doubleslit",
}
EXTEND_RUN_PARAMS = {"resolution": 64, "nlambda": 3, "backend": "c",
                     "model_stem": "doubleslit"}
EXTEND_BASE_ESTIMATE = common.estimate(8000, 64, 3, 1, "c",
                                       model_stem="doubleslit")


def _all_label_texts(dialog):
    from PySide6.QtWidgets import QLabel
    return [w.text() for w in dialog.findChildren(QLabel)]


def test_extend_mode_title_and_banner(qtbot):
    dialog = RunDialog(EXTEND_RUN_PARAMS, EXTEND_BASE_ESTIMATE,
                       extend_ctx=EXTEND_CTX)
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Extend Run"
    texts = " ".join(_all_label_texts(dialog))
    assert "Extending a COMPLETED run" in texts
    assert "8000" in texts


def test_extend_mode_default_new_rays_is_2x_current(qtbot):
    dialog = RunDialog(EXTEND_RUN_PARAMS, EXTEND_BASE_ESTIMATE,
                       extend_ctx=EXTEND_CTX)
    qtbot.addWidget(dialog)
    assert dialog.rays_spin.value() == 16000.0
    assert dialog.extend_target_rays() == 16000


def test_extend_mode_presets_update_spin_value(qtbot):
    from PySide6.QtWidgets import QPushButton
    dialog = RunDialog(EXTEND_RUN_PARAMS, EXTEND_BASE_ESTIMATE,
                       extend_ctx=EXTEND_CTX)
    qtbot.addWidget(dialog)
    presets = {b.text(): b for b in dialog.findChildren(QPushButton)
              if b.text() in ("x2", "x5", "x10")}
    assert set(presets) == {"x2", "x5", "x10"}
    qtbot.mouseClick(presets["x10"], Qt.LeftButton)
    assert dialog.rays_spin.value() == 80000.0
    assert dialog.extend_target_rays() == 80000


def test_extend_mode_projection_updates_on_spin_change(qtbot):
    dialog = RunDialog(EXTEND_RUN_PARAMS, EXTEND_BASE_ESTIMATE,
                       extend_ctx=EXTEND_CTX)
    qtbot.addWidget(dialog)
    before = dialog.meff_label.text()
    dialog.rays_spin.setValue(80000.0)
    after = dialog.meff_label.text()
    assert before != after
    assert "x lower" in dialog.pedestal_label.text()


def test_extend_mode_has_extend_button_and_no_skip_checkbox(qtbot):
    dialog = RunDialog(EXTEND_RUN_PARAMS, EXTEND_BASE_ESTIMATE,
                       extend_ctx=EXTEND_CTX)
    qtbot.addWidget(dialog)
    assert dialog.skip_checkbox is None
    buttons = dialog.findChild(QDialogButtonBox)
    ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_btn.text() == "Extend"
    assert buttons.button(QDialogButtonBox.StandardButton.Cancel) \
        is not None


def test_extend_target_rays_none_outside_extend_mode(qtbot):
    from mieworkbench.tests.test_rundialog import FAKE_PARAMS, FAKE_ESTIMATE
    dialog = RunDialog(FAKE_PARAMS, FAKE_ESTIMATE, calibrated=True)
    qtbot.addWidget(dialog)
    assert dialog.extend_target_rays() is None


# ===========================================================================
# 4. MainWindow wiring: _on_resume_run / _on_extend_run
# ===========================================================================
def test_on_resume_run_noop_without_case_dir(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._on_resume_run(None)     # must not raise


def test_on_resume_run_needs_open_model(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.model_path is None
    case = tmp_path / "quick"
    _tracing_checkpoint(case)

    started = []
    window.runner.start_resume = lambda *a, **k: started.append(True)
    window._on_resume_run(str(case))
    assert started == []


def test_on_resume_run_launches_with_case_dir(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.model_path = str(FCSTD) if FCSTD.exists() else "dummy.FCStd"
    case = tmp_path / "quick"
    _tracing_checkpoint(case)

    calls = []
    monkeypatch.setattr(
        window.runner, "start_resume",
        lambda model_path, cd, extra_env=None:
            calls.append((model_path, str(cd))) or True)

    window._on_resume_run(str(case))
    assert calls == [(window.model_path, str(case))]
    assert window._resume_extend_case_dir == str(case)


def test_on_resume_run_warns_when_already_running(qtbot, tmp_path,
                                                   monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.model_path = "dummy.FCStd"
    case = tmp_path / "quick"
    _tracing_checkpoint(case)
    monkeypatch.setattr(window.runner, "is_running", lambda: True)

    started = []
    monkeypatch.setattr(window.runner, "start_resume",
                        lambda *a, **k: started.append(True))
    window._on_resume_run(str(case))
    assert started == []


def test_on_extend_run_noop_when_not_extendable(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window.model_path = "dummy.FCStd"
    case = tmp_path / "quick"      # no checkpoint at all
    window._on_extend_run(str(case))    # must not raise, no dialog shown


def test_on_extend_run_noop_when_window_hidden(qtbot, tmp_path):
    """isVisible-guarded like _confirm_run_dialog: an offscreen (hidden)
    window never exec's the modal -- there's no user to pick a new ray
    count, so it just declines instead of hanging."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.model_path = "dummy.FCStd"
    assert window.isVisible() is False
    case = tmp_path / "quick"
    _completed_case(case, target_rays=8000)
    window._on_extend_run(str(case))
    assert not hasattr(window, "_last_extend_dialog")


def test_on_extend_run_launches_on_accept(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "isVisible", lambda: True)
    window.model_path = "dummy.FCStd"
    case = tmp_path / "quick"
    _completed_case(case, target_rays=8000)

    def fake_exec(self):
        self.rays_spin.setValue(16000.0)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(RunDialog, "exec", fake_exec)

    calls = []
    monkeypatch.setattr(
        window.runner, "start_extend",
        lambda model_path, cd, new_rays, extra_env=None:
            calls.append((model_path, str(cd), new_rays)) or True)

    window._on_extend_run(str(case))
    assert calls == [(window.model_path, str(case), 16000)]
    assert window._last_extend_dialog.extend_ctx["current_rays"] == 8000
    assert window._resume_extend_case_dir == str(case)


def test_on_extend_run_cancel_does_not_launch(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "isVisible", lambda: True)
    window.model_path = "dummy.FCStd"
    case = tmp_path / "quick"
    _completed_case(case, target_rays=8000)

    monkeypatch.setattr(RunDialog, "exec",
                        lambda self: QDialog.DialogCode.Rejected)
    started = []
    monkeypatch.setattr(window.runner, "start_extend",
                        lambda *a, **k: started.append(True))
    window._on_extend_run(str(case))
    assert started == []


def test_on_extend_run_warns_when_already_running(qtbot, tmp_path,
                                                   monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.model_path = "dummy.FCStd"
    case = tmp_path / "quick"
    _completed_case(case, target_rays=8000)
    monkeypatch.setattr(window.runner, "is_running", lambda: True)

    execs = []
    monkeypatch.setattr(RunDialog, "exec", lambda self: execs.append(1))
    window._on_extend_run(str(case))
    assert execs == []     # never even builds the dialog


# ===========================================================================
# 5. ONE real end-to-end smoke test (subprocess setup + real controller)
# ===========================================================================
pytestmark_real = pytest.mark.skipif(
    cengine.binary_path() is None or not MODEL_JSON.exists()
    or not FCSTD.exists(),
    reason="needs the built C engine + extracted geometry/doubleslit + "
           "doubleslit.FCStd")


@pytest.fixture(scope="module")
def real_checkpointed_case(tmp_path_factory):
    """Trace a REAL, tiny (fast) doubleslit case straight through
    run_trace.py (bypassing run_pipeline -- --no-gather-gate isn't a
    pipeline-level flag, and 30000 rays already clears the coherent
    gather's M_eff>=1000 gate honestly, so this is not a shortcut around
    real physics, just a cheap starting point). Module-scoped: every real-
    subprocess test in this file reuses the one completed+checkpointed
    case instead of re-tracing."""
    if cengine.binary_path() is None or not MODEL_JSON.exists() \
            or not FCSTD.exists():
        pytest.skip("needs the built C engine + extracted "
                    "geometry/doubleslit + doubleslit.FCStd")
    results_root = tmp_path_factory.mktemp("ckpt_gui_results")
    case_dir = results_root / "doubleslit" / "quick-guiext"
    cmd = [common.OPTICS_PYTHON, str(SCRIPTS / "run_trace.py"),
           "--model-json", str(MODEL_JSON), "--case-dir", str(case_dir),
           "--rays", "30000", "--resolution", "64", "--nlambda", "3",
           "--spectral-bins", "4", "--engine", "c", "--seeds", "1"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, (
        "setup trace failed (%d):\n%s\n%s"
        % (p.returncode, p.stdout[-3000:], p.stderr[-3000:]))
    ckpt = json.loads((case_dir / "cengine" / "checkpoint.json").read_text())
    assert ckpt["status"] == "completed" and ckpt["target_rays"] == 30000
    return {"results_root": results_root, "case_dir": case_dir}


@pytestmark_real
def test_real_extend_flow_doubles_rays_in_checkpoint(
        qtbot, real_checkpointed_case):
    """Drives the REAL flow: checkpointinfo reads the real checkpoint,
    RunController.start_extend launches a REAL run_pipeline.py subprocess
    with --extend, and the checkpoint on disk shows target_rays doubled
    (and gains one more chunk/extension entry) once it finishes -- steps
    limited to "trace" alone (skip post/viz) to stay well under budget;
    the P1 chunked-run contract itself (checkpoint/merge/gather) is
    exercised in full by scripts/raytracer/tests/test_checkpoint_extend.py,
    this test's job is only the GUI wiring on top of it."""
    case_dir = real_checkpointed_case["case_dir"]
    results_root = real_checkpointed_case["results_root"]

    state = checkpointinfo.extend_state(case_dir)
    assert state is not None
    assert state["current_rays"] == 30000
    new_rays = state["current_rays"] * 2

    ctl = RunController()
    with qtbot.waitSignal(ctl.finished, timeout=60000) as blocker:
        started = ctl.start_extend(str(FCSTD), case_dir, new_rays,
                                   extra_env={"MIEWB_RESULTS_DIR":
                                              str(results_root)},
                                   steps="trace")
        assert started

    assert blocker.args == [0], "extend run_pipeline exited nonzero"
    ckpt = json.loads((case_dir / "cengine" / "checkpoint.json").read_text())
    assert ckpt["status"] == "completed"
    assert ckpt["target_rays"] == new_rays
    assert ckpt["extensions"][-1] == {"from": 30000, "to": new_rays}
