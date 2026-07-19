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
    monkeypatch.setattr(window, "_preview_pattern_spec",
                        lambda: "rings:dr=2:nper=6")

    def fake_start(project, workspace, pattern=None, only_bodies=None,
                   optical_properties=None):
        calls.append(pattern)
        return True

    monkeypatch.setattr(window.raypreview, "start", fake_start)

    # no segments loaded (fresh window) -> needs_preview is True
    assert not window.anim_controller.has_segments()
    window._on_anim_enabled_toggled(True)

    assert calls == ["rings:dr=2:nper=6"]


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
# Manual "Live ray preview…" dialog
# ---------------------------------------------------------------------------
def test_manual_dialog_prefilled_with_resolved_spec(window, monkeypatch):
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.raypreview, "is_running", lambda: False)
    monkeypatch.setattr(window, "_preview_pattern_spec",
                        lambda: "rings:dr=1:nper=8")

    seen = {}

    def fake_get_text(*args, **kwargs):
        seen["text"] = kwargs.get("text")
        return kwargs.get("text"), True

    monkeypatch.setattr(QInputDialog, "getText", fake_get_text)

    calls = []
    monkeypatch.setattr(window.raypreview, "start",
                        lambda *a, pattern=None, **kw:
                            calls.append(pattern) or True)

    window._on_ray_preview()

    assert seen["text"] == "rings:dr=1:nper=8"
    assert calls == ["rings:dr=1:nper=8"]


def test_manual_dialog_accepts_bare_integer(window, monkeypatch):
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.raypreview, "is_running", lambda: False)
    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *a, **kw: ("8", True))

    calls = []
    monkeypatch.setattr(window.raypreview, "start",
                        lambda *a, pattern=None, **kw:
                            calls.append(pattern) or True)

    window._on_ray_preview()

    assert calls == ["fan:n=8"]


def test_manual_dialog_invalid_spec_shows_status_no_start(window,
                                                           monkeypatch):
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.raypreview, "is_running", lambda: False)
    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *a, **kw: ("not a pattern", True))

    calls = []
    monkeypatch.setattr(window.raypreview, "start",
                        lambda *a, **kw: calls.append(1) or True)

    window._on_ray_preview()

    assert calls == []
