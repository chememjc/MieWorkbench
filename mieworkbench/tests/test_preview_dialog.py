"""PreviewConfigDialog unit tests -- constructed directly, never
exec'd (guarded-modal contract: offscreen runs must not block)."""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

from mieworkbench.panes.previewdialog import (  # noqa: E402
    DEFAULT_VALUES, PreviewConfigDialog,
)

VALUES = {
    "spec": "rings:dr=2:nper=8:nrings=3",
    "engine": "sequential",
    "dim_mode": "log",
    "dim_floor": 5.0,
    "dim_range_db": 40.0,
    "anim_enabled": True,
    "anim_bead_size": 2.5,
    "anim_speed_mm_s": 7.0,
    "anim_fps": 30,
    "anim_ray_cap": 100,
    "anim_bead_opacity_mode": "power",
    "anim_bead_opacity_db": 25.0,
}


def test_values_round_trip_every_section(qtbot):
    dialog = PreviewConfigDialog(dict(VALUES))
    qtbot.addWidget(dialog)
    assert dialog.values() == VALUES


def test_defaults_engine_is_full(qtbot):
    dialog = PreviewConfigDialog()
    qtbot.addWidget(dialog)
    v = dialog.values()
    assert v["engine"] == "full"
    assert v["spec"] == DEFAULT_VALUES["spec"]


def test_invalid_ctor_spec_falls_back_to_default(qtbot):
    dialog = PreviewConfigDialog({"spec": "not a pattern"})
    qtbot.addWidget(dialog)
    assert dialog.values()["spec"] == DEFAULT_VALUES["spec"]


# -- Advanced-row sync -------------------------------------------------------
def test_widget_edit_updates_advanced_text(qtbot):
    dialog = PreviewConfigDialog({"spec": "fan:n=5"})
    qtbot.addWidget(dialog)
    dialog.pattern_widget.fan_n_spin.setValue(9)
    assert dialog.spec_edit.text() == "fan:n=9"
    assert dialog.spec_error_label.text() == ""


def test_advanced_text_updates_widget(qtbot):
    dialog = PreviewConfigDialog({"spec": "fan:n=5"})
    qtbot.addWidget(dialog)
    dialog._on_spec_text_edited("rings:dr=2:nper=8")
    assert dialog.pattern_widget.kind_combo.currentData() == "rings"
    assert dialog.pattern_widget.rings_dr_spin.value() == 2.0
    assert dialog.pattern_widget.rings_nper_spin.value() == 8
    assert dialog.values()["spec"] == "rings:dr=2:nper=8"


def test_advanced_text_bare_integer_shorthand(qtbot):
    dialog = PreviewConfigDialog({"spec": "fan:n=5"})
    qtbot.addWidget(dialog)
    dialog._on_spec_text_edited("7")
    assert dialog.values()["spec"] == "fan:n=7"
    assert dialog.pattern_widget.fan_n_spin.value() == 7


def test_advanced_text_garbage_keeps_widget_valid(qtbot):
    dialog = PreviewConfigDialog({"spec": "fan:n=5"})
    qtbot.addWidget(dialog)
    dialog._on_spec_text_edited("total garbage")
    assert dialog.spec_error_label.text()          # inline error shown
    assert dialog.values()["spec"] == "fan:n=5"    # widget untouched
    # a subsequent valid edit clears the error
    dialog._on_spec_text_edited("fan:n=6")
    assert dialog.spec_error_label.text() == ""
    assert dialog.values()["spec"] == "fan:n=6"


# -- Log-range preset <-> custom --------------------------------------------
@pytest.mark.parametrize("db,is_preset", [(30.0, True), (40.0, True),
                                          (60.0, True), (37.0, False)])
def test_range_preset_custom_mapping(qtbot, db, is_preset):
    dialog = PreviewConfigDialog({"dim_mode": "log", "dim_range_db": db})
    qtbot.addWidget(dialog)
    if is_preset:
        assert dialog.range_combo.currentData() == db
        assert not dialog.range_spin.isEnabled()
    else:
        assert dialog.range_combo.currentData() is None   # Custom…
        assert dialog.range_spin.isEnabled()
    assert dialog.values()["dim_range_db"] == db


def test_range_row_enabled_only_in_log_mode(qtbot):
    dialog = PreviewConfigDialog({"dim_mode": "linear",
                                  "dim_range_db": 30.0})
    qtbot.addWidget(dialog)
    assert not dialog.range_combo.isEnabled()
    dialog.dim_mode_combo.setCurrentIndex(
        dialog.dim_mode_combo.findData("log"))
    assert dialog.range_combo.isEnabled()


def test_range_preset_switch_updates_value(qtbot):
    dialog = PreviewConfigDialog({"dim_mode": "log",
                                  "dim_range_db": 30.0})
    qtbot.addWidget(dialog)
    dialog.range_combo.setCurrentIndex(dialog.range_combo.findData(60.0))
    assert dialog.values()["dim_range_db"] == 60.0
    # Custom… re-enables the spin at the last value
    dialog.range_combo.setCurrentIndex(dialog.range_combo.findData(None))
    assert dialog.range_spin.isEnabled()
    dialog.range_spin.setValue(47.0)
    assert dialog.values()["dim_range_db"] == 47.0


# -- Engine auto-log ---------------------------------------------------------
def test_full_trace_with_dim_off_auto_selects_log(qtbot):
    dialog = PreviewConfigDialog({"engine": "sequential",
                                  "dim_mode": "off"})
    qtbot.addWidget(dialog)
    dialog.engine_combo.setCurrentIndex(
        dialog.engine_combo.findData("full"))
    assert dialog.values()["dim_mode"] == "log"


def test_full_trace_leaves_explicit_dim_mode_alone(qtbot):
    dialog = PreviewConfigDialog({"engine": "sequential",
                                  "dim_mode": "linear"})
    qtbot.addWidget(dialog)
    dialog.engine_combo.setCurrentIndex(
        dialog.engine_combo.findData("full"))
    assert dialog.values()["dim_mode"] == "linear"


def test_back_to_sequential_keeps_log(qtbot):
    dialog = PreviewConfigDialog({"engine": "sequential",
                                  "dim_mode": "off"})
    qtbot.addWidget(dialog)
    dialog.engine_combo.setCurrentIndex(
        dialog.engine_combo.findData("full"))
    dialog.engine_combo.setCurrentIndex(
        dialog.engine_combo.findData("sequential"))
    assert dialog.values()["dim_mode"] == "log"   # no reverse action
