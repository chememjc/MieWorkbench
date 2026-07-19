"""WP2 tests: PreviewConfigWidget spec<->fields roundtrip, Project
preview-config persistence (FreeCAD-gated, mirrors the optimize/tolerance
config test in test_save_flows.py), MainWindow._preview_pattern_spec
resolution order, self-sufficient bead-animation enable (auto-launches a
preview when there is nothing to animate), and the manual "Live ray
preview…" dialog (prefill + bare-integer shorthand)."""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

from mieworkbench.core.project import Project, ProjectError  # noqa: E402
from mieworkbench.mainwindow import MainWindow, QInputDialog  # noqa: E402
from mieworkbench.widgets.preview_config import (  # noqa: E402
    PreviewConfigWidget, fields_from_spec, spec_from_fields,
)


# ---------------------------------------------------------------------------
# PreviewConfigWidget / pure helpers
# ---------------------------------------------------------------------------
def test_spec_from_fields_fan():
    assert spec_from_fields("fan", n=7) == "fan:n=7"


def test_spec_from_fields_rings_omits_auto_nrings():
    assert spec_from_fields("rings", dr_mm=1.0, nper=12, nrings=0) \
        == "rings:dr=1:nper=12"


def test_spec_from_fields_rings_with_nrings():
    assert spec_from_fields("rings", dr_mm=0.5, nper=8, nrings=4) \
        == "rings:dr=0.5:nper=8:nrings=4"


def test_fields_from_spec_roundtrip_fan():
    assert fields_from_spec("fan:n=9") == {"kind": "fan", "n": 9}


def test_fields_from_spec_roundtrip_rings():
    assert fields_from_spec("rings:dr=0.5:nper=8:nrings=4") == {
        "kind": "rings", "dr_mm": 0.5, "nper": 8, "nrings": 4}


def test_fields_from_spec_rings_no_nrings_defaults_zero():
    assert fields_from_spec("rings:dr=1:nper=12")["nrings"] == 0


def test_fields_from_spec_invalid_raises():
    with pytest.raises(ValueError):
        fields_from_spec("bogus:pattern")


def test_widget_spec_roundtrip_fan(qtbot):
    w = PreviewConfigWidget()
    qtbot.addWidget(w)
    w.set_spec("fan:n=42")
    assert w.spec() == "fan:n=42"
    assert w.kind_combo.currentData() == "fan"
    assert w.fan_n_spin.value() == 42


def test_widget_spec_roundtrip_rings(qtbot):
    w = PreviewConfigWidget()
    qtbot.addWidget(w)
    w.set_spec("rings:dr=2:nper=6:nrings=3")
    assert w.spec() == "rings:dr=2:nper=6:nrings=3"
    assert w.kind_combo.currentData() == "rings"
    assert w.rings_dr_spin.value() == 2
    assert w.rings_nper_spin.value() == 6
    assert w.rings_nrings_spin.value() == 3


def test_widget_default_spec_is_valid_fan(qtbot):
    w = PreviewConfigWidget()
    qtbot.addWidget(w)
    assert w.spec() == "fan:n=5"


def test_widget_set_spec_invalid_raises(qtbot):
    w = PreviewConfigWidget()
    qtbot.addWidget(w)
    with pytest.raises(ValueError):
        w.set_spec("not:a:pattern")


# ---------------------------------------------------------------------------
# Project.get_preview_config / set_preview_config
# ---------------------------------------------------------------------------
@pytest.mark.freecad
def test_project_preview_config_persists_across_save_reopen(tmp_path):
    project = Project()
    try:
        path = str(tmp_path / "scene.FCStd")
        project.new_document(path)
        assert project.variables_sheet() is None
        assert project.get_preview_config() is None

        cfg = {"spec": "fan:n=8"}
        project.set_preview_config(cfg)
        assert project.variables_sheet() is not None
        assert project.get_preview_config() == cfg

        # no-op when unchanged
        undo_text_before = project.undo_stack.undo_text()
        project.set_preview_config(cfg)
        assert project.undo_stack.undo_text() == undo_text_before

        # undo restores the prior (unset) state
        assert project.undo()
        assert project.get_preview_config() is None
        assert project.redo()
        assert project.get_preview_config() == cfg

        # undo restores the PRIOR value, not just "unset"
        cfg2 = {"spec": "rings:dr=1:nper=12:nrings=4"}
        project.set_preview_config(cfg2)
        assert project.get_preview_config() == cfg2
        assert project.undo()
        assert project.get_preview_config() == cfg

        # travels through a real save + reopen
        project.save()
        project.close()
        project.open_fcstd(path)
        assert project.get_preview_config() == cfg
    finally:
        project.shutdown()


@pytest.mark.freecad
def test_project_set_preview_config_invalid_spec_raises_untouched(tmp_path):
    project = Project()
    try:
        path = str(tmp_path / "scene2.FCStd")
        project.new_document(path)
        with pytest.raises(ProjectError):
            project.set_preview_config({"spec": "not:a:real:pattern"})
        # nothing was written -- no sheet, no config
        assert project.get_preview_config() is None
    finally:
        project.shutdown()


# ---------------------------------------------------------------------------
# MainWindow._preview_pattern_spec resolution order
# ---------------------------------------------------------------------------
@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_preview_pattern_spec_default_fallback(window):
    window.settings.set("preview_pattern_spec", "")
    assert window._preview_pattern_spec() == "fan:n=5"


def test_preview_pattern_spec_uses_settings_when_no_project(window):
    window.settings.set("preview_pattern_spec", "fan:n=11")
    assert window._preview_pattern_spec() == "fan:n=11"


def test_preview_pattern_spec_prefers_project_over_settings(window,
                                                             monkeypatch):
    window.settings.set("preview_pattern_spec", "fan:n=11")
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.project, "get_preview_config",
                        lambda: {"spec": "rings:dr=1:nper=9"})
    assert window._preview_pattern_spec() == "rings:dr=1:nper=9"


def test_preview_pattern_spec_falls_back_on_invalid_project_spec(
        window, monkeypatch):
    window.settings.set("preview_pattern_spec", "fan:n=11")
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.project, "get_preview_config",
                        lambda: {"spec": "garbage"})
    assert window._preview_pattern_spec() == "fan:n=11"


def test_preview_pattern_spec_falls_back_on_invalid_settings_spec(
        window, monkeypatch):
    window.settings.set("preview_pattern_spec", "garbage")
    monkeypatch.setattr(window.project, "is_open", lambda: False)
    assert window._preview_pattern_spec() == "fan:n=5"


# ---------------------------------------------------------------------------
# Self-sufficient bead-animation enable
# ---------------------------------------------------------------------------
def test_anim_enable_starts_preview_with_configured_pattern(window,
                                                             monkeypatch):
    calls = []
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.runner, "is_running", lambda: False)
    monkeypatch.setattr(window.raypreview, "is_running", lambda: False)
    monkeypatch.setattr(window, "_resolve_preview_cfg",
                        lambda: {"spec": "rings:dr=2:nper=6",
                                 "engine": "sequential"})

    def fake_start(project, workspace, pattern=None, engine=None,
                   only_bodies=None, optical_properties=None):
        calls.append((pattern, engine))
        return True

    monkeypatch.setattr(window.raypreview, "start", fake_start)

    # no segments loaded (fresh window) -> needs_preview is True
    assert not window.anim_controller.has_segments()
    window._on_anim_enabled_toggled(True)

    assert calls == [("rings:dr=2:nper=6", "sequential")]


def test_anim_enable_does_not_double_start_when_busy(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.runner, "is_running", lambda: False)
    monkeypatch.setattr(window.raypreview, "is_running", lambda: True)
    monkeypatch.setattr(window.raypreview, "start",
                        lambda *a, **kw: calls.append(1) or True)

    window._on_anim_enabled_toggled(True)

    assert calls == []   # already-running preview blocks a second start


def test_anim_enable_noop_when_segments_already_fresh(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window.anim_controller, "has_segments",
                        lambda: True)
    monkeypatch.setattr(window.scene3d.view, "overlay_is_stale",
                        lambda: False)
    monkeypatch.setattr(window.raypreview, "start",
                        lambda *a, **kw: calls.append(1) or True)

    window._on_anim_enabled_toggled(True)

    assert calls == []   # nothing to (re)generate


# ---------------------------------------------------------------------------
# Manual "Live ray preview…" -> Preview Configuration dialog
# ---------------------------------------------------------------------------
_DISPLAY_KEYS = ("ray_dimming_mode", "ray_dimming_floor",
                 "ray_dimming_range_db", "anim_enabled", "anim_bead_size",
                 "anim_speed_mm_s", "anim_fps", "anim_ray_cap",
                 "anim_bead_opacity_mode", "anim_bead_opacity_db",
                 "preview_pattern_spec", "preview_engine_mode")


def _drive_preview_dialog(window, monkeypatch, edit_fn=None):
    """Open the (never exec'd -- the window is hidden) dialog via
    _on_ray_preview, optionally edit it, accept, and return the
    raypreview.start calls. Saves/restores every key the accept path
    persists."""
    saved = {k: window.settings._qs.value(k, None) for k in _DISPLAY_KEYS}
    calls = []
    monkeypatch.setattr(window.raypreview, "start",
                        lambda *a, pattern=None, engine=None, **kw:
                            calls.append((pattern, engine)) or True)
    # the accept path persists a changed cfg into the project; these
    # tests fake is_open on a workerless Project, so stub the write
    monkeypatch.setattr(window.project, "set_preview_config",
                        lambda cfg: None)
    try:
        window._on_ray_preview()
        dialog = window._last_preview_dialog
        if edit_fn is not None:
            edit_fn(dialog)
        dialog.accept()      # fires accepted -> _apply_preview_dialog
    finally:
        for k, v in saved.items():
            if v is None:
                window.settings._qs.remove(k)
            else:
                window.settings._qs.setValue(k, v)
        window.settings._qs.sync()
    return dialog, calls


def test_manual_dialog_prefilled_with_resolved_cfg(window, monkeypatch):
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.raypreview, "is_running", lambda: False)
    monkeypatch.setattr(window, "_resolve_preview_cfg",
                        lambda: {"spec": "rings:dr=1:nper=8",
                                 "engine": "sequential"})

    dialog, calls = _drive_preview_dialog(window, monkeypatch)

    assert dialog.values()["spec"] == "rings:dr=1:nper=8"
    assert dialog.values()["engine"] == "sequential"
    assert calls == [("rings:dr=1:nper=8", "sequential")]


def test_manual_dialog_accepts_bare_integer(window, monkeypatch):
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.raypreview, "is_running", lambda: False)

    dialog, calls = _drive_preview_dialog(
        window, monkeypatch,
        edit_fn=lambda d: d._on_spec_text_edited("8"))

    assert dialog.values()["spec"] == "fan:n=8"
    assert [c[0] for c in calls] == ["fan:n=8"]


def test_manual_dialog_invalid_text_keeps_last_valid_spec(window,
                                                          monkeypatch):
    """Garbage in the Advanced row shows the inline error and leaves the
    pattern fields (= the accepted spec) at their last valid state --
    the dialog can never launch an invalid pattern."""
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.raypreview, "is_running", lambda: False)
    monkeypatch.setattr(window, "_resolve_preview_cfg",
                        lambda: {"spec": "fan:n=7", "engine": "full"})

    def edit(dialog):
        dialog._on_spec_text_edited("not a pattern")
        assert dialog.spec_error_label.text()

    dialog, calls = _drive_preview_dialog(window, monkeypatch,
                                          edit_fn=edit)

    assert [c[0] for c in calls] == ["fan:n=7"]


def test_manual_dialog_engine_reaches_start(window, monkeypatch):
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.raypreview, "is_running", lambda: False)
    monkeypatch.setattr(window, "_resolve_preview_cfg",
                        lambda: {"spec": "fan:n=5",
                                 "engine": "sequential"})

    def edit(dialog):
        idx = dialog.engine_combo.findData("full")
        dialog.engine_combo.setCurrentIndex(idx)

    _dialog, calls = _drive_preview_dialog(window, monkeypatch,
                                           edit_fn=edit)

    assert calls == [("fan:n=5", "full")]


# ---------------------------------------------------------------------------
# {"spec", "engine"} resolution (project -> QSettings -> defaults)
# ---------------------------------------------------------------------------
def test_resolve_cfg_defaults_engine_full(window, monkeypatch):
    """No stored engine anywhere resolves to "full" (reflections visible
    out of the box -- the owner default), incl. for old {"spec"} dicts."""
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.project, "get_preview_config",
                        lambda: {"spec": "fan:n=9"})   # old-style dict
    window.settings.set("preview_engine_mode", "")
    assert window._resolve_preview_cfg() == {"spec": "fan:n=9",
                                             "engine": "full"}


def test_resolve_cfg_project_engine_wins(window, monkeypatch):
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.project, "get_preview_config",
                        lambda: {"spec": "fan:n=9",
                                 "engine": "sequential"})
    assert window._resolve_preview_cfg()["engine"] == "sequential"


def test_resolve_cfg_settings_engine_fallback(window, monkeypatch):
    monkeypatch.setattr(window.project, "is_open", lambda: False)
    saved = window.settings._qs.value("preview_engine_mode", None)
    try:
        window.settings.set("preview_engine_mode", "sequential")
        assert window._resolve_preview_cfg()["engine"] == "sequential"
        window.settings.set("preview_engine_mode", "bogus")
        assert window._resolve_preview_cfg()["engine"] == "full"
    finally:
        if saved is None:
            window.settings._qs.remove("preview_engine_mode")
        else:
            window.settings._qs.setValue("preview_engine_mode", saved)
        window.settings._qs.sync()


def test_set_preview_config_rejects_bad_engine(window, monkeypatch):
    from mieworkbench.core.project import ProjectError
    with pytest.raises(ProjectError):
        window.project.set_preview_config({"spec": "fan:n=5",
                                           "engine": "warp"})
