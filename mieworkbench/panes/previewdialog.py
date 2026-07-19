"""PreviewConfigDialog - the all-in-one ray-preview configuration dialog.

Replaces the old raw QInputDialog --viz-pattern prompt AND consolidates
every preview-related editor into one surface (owner decision):
  - Ray pattern: the existing widgets.preview_config.PreviewConfigWidget
    (Fan / Rings combo + validated spin boxes);
  - Engine: "Sequential (fast, no reflections)" vs "Full trace (shows
    reflections)" -- full trace forces the real Monte-Carlo preview
    subprocess (Fresnel ghost children, 6-bounce engine cap, weak-ray
    power floor) instead of the on-axis sequential fast path;
  - Overlay display: ray-extinction mode incl. the new Logarithmic (dB)
    curve with a preset-or-custom dynamic range, plus the opacity floor;
  - Bead animation: the tracer-bead keys that used to live in the
    Settings "Defaults" tab (same widgets/ranges/tooltips);
  - Advanced: the composed --viz-pattern spec string, editable, kept in
    bidirectional sync with the pattern widget (bare integer means
    fan:n=<int>, matching the old prompt's shorthand).

Auto-behavior (owner decision): switching the engine to Full trace while
extinction is Off selects Logarithmic -- Fresnel ghosts are 20-40 dB
below the beam and indistinguishable at uniform opacity; an explicitly
chosen Linear/Perceptual mode is left alone.

The dialog is dumb state: the constructor takes a plain values dict and
values() returns the same shape; the caller (MainWindow) owns
persistence, live-apply, and preview launch.

Guarded-modal contract (CLAUDE.md: "Never show an unguarded modal in a
pane code path" / "guard on isVisible()"): this module never calls
.exec() itself -- the caller (MainWindow) owns that, guarded on
isVisible(). Tests construct a PreviewConfigDialog directly and drive
its controls without ever invoking exec(), so offscreen test runs can
never block on it.
"""

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QSpinBox,
    QVBoxLayout,
)

from ..widgets.preview_config import PreviewConfigWidget

# (data, combo label) rows, in combo order.
_ENGINES = [
    ("sequential", "Sequential (fast, no reflections)"),
    ("full", "Full trace (shows reflections)"),
]
_DIM_MODES = [
    ("off", "Off"),
    ("linear", "Linear (P/P₀)"),
    ("sqrt", "Perceptual (√(P/P₀))"),
    ("log", "Logarithmic (dB)"),
]
# Log dynamic-range presets: (dB value, combo label). None = Custom.
_RANGE_PRESETS = [
    (30.0, "30 dB — matches bead default; secondary uncoated "
           "ghosts faint"),
    (40.0, "40 dB — secondary ghosts ~30% opacity; coated ghost "
           "pairs visible"),
    (60.0, "60 dB — nearly everything visible"),
    (None, "Custom…"),
]

DEFAULT_VALUES = {
    "spec": "fan:n=5",
    "engine": "full",
    "dim_mode": "off",
    "dim_floor": 0.0,
    "dim_range_db": 30.0,
    "anim_enabled": False,
    "anim_bead_size": 1.0,
    "anim_speed_mm_s": 2.0,
    "anim_fps": 15,
    "anim_ray_cap": 300,
    "anim_bead_opacity_mode": "off",
    "anim_bead_opacity_db": 30.0,
}


def _combo_select_data(combo, data, fallback_index=0):
    idx = combo.findData(data)
    combo.setCurrentIndex(idx if idx >= 0 else fallback_index)


class PreviewConfigDialog(QDialog):
    """All preview settings in one accept-only dialog.

    values: dict with the DEFAULT_VALUES keys (missing keys fall back to
        the defaults; an invalid spec falls back to the default spec so
        the dialog always opens in a valid state).
    values() returns the same shape; the spec always comes from the
        pattern widget, so an accepted dialog can never yield an invalid
        spec regardless of what was typed in the Advanced row.
    """

    def __init__(self, values=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview Configuration")
        v = dict(DEFAULT_VALUES)
        v.update(values or {})
        self._syncing_spec = False

        layout = QVBoxLayout(self)

        # -- Ray pattern -----------------------------------------------------
        pattern_box = QGroupBox("Ray pattern")
        pattern_lay = QVBoxLayout(pattern_box)
        self.pattern_widget = PreviewConfigWidget()
        try:
            self.pattern_widget.set_spec(v["spec"])
        except ValueError:
            self.pattern_widget.set_spec(DEFAULT_VALUES["spec"])
        pattern_lay.addWidget(self.pattern_widget)
        layout.addWidget(pattern_box)

        # -- Engine ----------------------------------------------------------
        engine_box = QGroupBox("Trace engine")
        engine_form = QFormLayout(engine_box)
        self.engine_combo = QComboBox()
        for data, label in _ENGINES:
            self.engine_combo.addItem(label, data)
        self.engine_combo.setToolTip(
            "Sequential: in-process fast path, primary transmitted chain "
            "only (exact bead timing, no Fresnel ghosts).\n"
            "Full trace: the real Monte-Carlo engine in a subprocess -- "
            "reflection/ghost children up to the 6-bounce engine cap with "
            "the standard weak-ray power floor; slower, and every "
            "auto-preview after an edit pays that cost too.")
        _combo_select_data(self.engine_combo, v["engine"])
        engine_form.addRow("Engine:", self.engine_combo)
        layout.addWidget(engine_box)

        # -- Overlay display -------------------------------------------------
        display_box = QGroupBox("Overlay display")
        display_form = QFormLayout(display_box)

        self.dim_mode_combo = QComboBox()
        for data, label in _DIM_MODES:
            self.dim_mode_combo.addItem(label, data)
        self.dim_mode_combo.setToolTip(
            "Ray extinction: fade ray segments by remaining power "
            "(same setting as the toolbar combo / View menu). "
            "Logarithmic maps opacity to dB below the source over the "
            "dynamic range below -- the mode that keeps weak Fresnel "
            "reflections visible.")
        _combo_select_data(self.dim_mode_combo, v["dim_mode"])
        display_form.addRow("Ray extinction:", self.dim_mode_combo)

        self.range_combo = QComboBox()
        for db, label in _RANGE_PRESETS:
            self.range_combo.addItem(label, db)
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(1.0, 120.0)
        self.range_spin.setDecimals(0)
        self.range_spin.setSuffix(" dB")
        self.range_spin.setToolTip(
            "Custom dynamic range: a segment R dB below the source "
            "renders at opacity 1 - R/range (uncoated-glass ghosts sit "
            "~14 dB down per reflection, coated ones ~19 dB)")
        range_row = QHBoxLayout()
        range_row.addWidget(self.range_combo, 1)
        range_row.addWidget(self.range_spin)
        self._set_range_db(float(v["dim_range_db"]))
        self.range_combo.currentIndexChanged.connect(
            self._on_range_preset_changed)
        display_form.addRow("Log dynamic range:", range_row)

        self.dim_floor_spin = QDoubleSpinBox()
        self.dim_floor_spin.setRange(0.0, 100.0)
        self.dim_floor_spin.setSuffix(" %")
        self.dim_floor_spin.setValue(float(v["dim_floor"]))
        self.dim_floor_spin.setToolTip(
            "Minimum segment opacity under extinction (0 = fade fully)")
        display_form.addRow("Extinction floor:", self.dim_floor_spin)
        layout.addWidget(display_box)

        # -- Bead animation (same widgets/ranges as the old Settings
        #    "Defaults" tab -- these edit the identical live keys) ------------
        anim_box = QGroupBox("Tracer-bead animation")
        anim_form = QFormLayout(anim_box)

        self.anim_enabled_check = QCheckBox("Show tracer beads")
        self.anim_enabled_check.setChecked(bool(v["anim_enabled"]))
        anim_form.addRow("Animation:", self.anim_enabled_check)

        self.anim_size_spin = QDoubleSpinBox()
        self.anim_size_spin.setRange(0.05, 50.0)
        self.anim_size_spin.setSuffix(" mm")
        self.anim_size_spin.setValue(float(v["anim_bead_size"]))
        anim_form.addRow("Bead size:", self.anim_size_spin)

        self.anim_speed_spin = QDoubleSpinBox()
        self.anim_speed_spin.setRange(0.01, 1000.0)
        self.anim_speed_spin.setSuffix(" mm/s")
        self.anim_speed_spin.setValue(float(v["anim_speed_mm_s"]))
        self.anim_speed_spin.setToolTip(
            "mm of ray path per real second for a vacuum bead "
            "(glass beads run slower by 1/n)")
        anim_form.addRow("Bead speed:", self.anim_speed_spin)

        self.anim_fps_spin = QSpinBox()
        self.anim_fps_spin.setRange(1, 120)
        self.anim_fps_spin.setValue(int(v["anim_fps"]))
        anim_form.addRow("Animation FPS:", self.anim_fps_spin)

        self.anim_cap_spin = QSpinBox()
        self.anim_cap_spin.setRange(0, 100000)
        self.anim_cap_spin.setValue(int(v["anim_ray_cap"]))
        self.anim_cap_spin.setToolTip(
            "Max beads DRAWN simultaneously per source (0 = unlimited); "
            "an honest render cap on busy run overlays, not a trace cap")
        anim_form.addRow("Bead cap / source:", self.anim_cap_spin)

        self.anim_opacity_combo = QComboBox()
        self.anim_opacity_combo.addItem("Opaque", "off")
        self.anim_opacity_combo.addItem("By power", "power")
        self.anim_opacity_combo.setToolTip(
            "Fade tracer beads by optical power (leading-wavefront beads "
            "stay solid); Opaque = always-solid beads")
        _combo_select_data(self.anim_opacity_combo,
                           v["anim_bead_opacity_mode"])
        anim_form.addRow("Bead opacity:", self.anim_opacity_combo)

        self.anim_opacity_db_spin = QDoubleSpinBox()
        self.anim_opacity_db_spin.setRange(10.0, 60.0)
        self.anim_opacity_db_spin.setDecimals(0)
        self.anim_opacity_db_spin.setSuffix(" dB")
        self.anim_opacity_db_spin.setValue(float(v["anim_bead_opacity_db"]))
        self.anim_opacity_db_spin.setToolTip(
            "Dynamic range of the power-to-opacity map (power mode only)")
        anim_form.addRow("Opacity range:", self.anim_opacity_db_spin)
        layout.addWidget(anim_box)

        # -- Advanced: the composed spec string, editable ----------------------
        adv_box = QGroupBox("Advanced")
        adv_form = QFormLayout(adv_box)
        self.spec_edit = QLineEdit(self.pattern_widget.spec())
        self.spec_edit.setToolTip(
            "The --viz-pattern spec the fields above compose "
            "('fan:n=K' or 'rings:dr=D:nper=N[:nrings=K]'; a bare "
            "integer means fan:n=<int>). Editing it updates the fields.")
        adv_form.addRow("Pattern spec:", self.spec_edit)
        self.spec_error_label = QLabel("")
        self.spec_error_label.setStyleSheet("color: #e74c3c;")
        self.spec_error_label.setWordWrap(True)
        adv_form.addRow(self.spec_error_label)
        layout.addWidget(adv_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # wiring: spec sync, log-range enablement, engine auto-log
        self.pattern_widget.specChanged.connect(self._on_widget_spec)
        self.spec_edit.textEdited.connect(self._on_spec_text_edited)
        self.dim_mode_combo.currentIndexChanged.connect(
            self._update_range_enabled)
        self.engine_combo.currentIndexChanged.connect(
            self._on_engine_changed)
        self._update_range_enabled()

    # -- Advanced-row sync ------------------------------------------------------
    def _on_widget_spec(self, spec):
        if self._syncing_spec:
            return
        self._syncing_spec = True
        try:
            self.spec_edit.setText(spec)
            self.spec_error_label.setText("")
        finally:
            self._syncing_spec = False

    def _on_spec_text_edited(self, text):
        if self._syncing_spec:
            return
        spec = text.strip()
        try:
            spec = "fan:n=%d" % int(spec)
        except ValueError:
            pass   # not a bare integer -- use the typed spec as-is
        self._syncing_spec = True
        try:
            self.pattern_widget.set_spec(spec)
            self.spec_error_label.setText("")
        except ValueError as exc:
            # fields keep their last valid state; OK stays safe because
            # values() reads the widget, never this text
            self.spec_error_label.setText("Invalid pattern: %s" % exc)
        finally:
            self._syncing_spec = False

    # -- Log-range preset/custom ------------------------------------------------
    def _set_range_db(self, db):
        """Select the matching preset, else Custom + spin box."""
        for i, (preset, _label) in enumerate(_RANGE_PRESETS):
            if preset is not None and abs(preset - db) < 1e-9:
                self.range_combo.setCurrentIndex(i)
                self.range_spin.setValue(preset)
                self.range_spin.setEnabled(False)
                return
        self.range_combo.setCurrentIndex(len(_RANGE_PRESETS) - 1)
        self.range_spin.setValue(db)
        self.range_spin.setEnabled(True)

    def _on_range_preset_changed(self, _index):
        preset = self.range_combo.currentData()
        if preset is None:
            self.range_spin.setEnabled(self.dim_mode_combo.currentData()
                                       == "log")
        else:
            self.range_spin.setValue(preset)
            self.range_spin.setEnabled(False)

    def _update_range_enabled(self, *_args):
        is_log = self.dim_mode_combo.currentData() == "log"
        self.range_combo.setEnabled(is_log)
        self.range_spin.setEnabled(
            is_log and self.range_combo.currentData() is None)

    # -- Engine auto-log --------------------------------------------------------
    def _on_engine_changed(self, _index):
        """Full trace with extinction Off would render 20-40 dB-down
        ghosts at full opacity -- flip Off to Logarithmic so the mode
        the user just asked for is actually legible. An explicit
        Linear/Perceptual choice is left alone; no reverse action."""
        if (self.engine_combo.currentData() == "full"
                and self.dim_mode_combo.currentData() == "off"):
            _combo_select_data(self.dim_mode_combo, "log")

    # -- API --------------------------------------------------------------------
    def values(self):
        return {
            "spec": self.pattern_widget.spec(),
            "engine": self.engine_combo.currentData(),
            "dim_mode": self.dim_mode_combo.currentData(),
            "dim_floor": self.dim_floor_spin.value(),
            "dim_range_db": self.range_spin.value(),
            "anim_enabled": self.anim_enabled_check.isChecked(),
            "anim_bead_size": self.anim_size_spin.value(),
            "anim_speed_mm_s": self.anim_speed_spin.value(),
            "anim_fps": self.anim_fps_spin.value(),
            "anim_ray_cap": self.anim_cap_spin.value(),
            "anim_bead_opacity_mode": self.anim_opacity_combo.currentData(),
            "anim_bead_opacity_db": self.anim_opacity_db_spin.value(),
        }
