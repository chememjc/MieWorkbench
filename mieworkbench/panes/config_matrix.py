"""ConfigMatrix - the graphical configuration matrix for run_pipeline.py.

Introspects cli_specs.build_parser("pipeline") and builds one widget per
CLI option, grouped exactly as the parser's own argument groups (so this
form can never silently drift from the real CLI: a new --option added to
cli_specs.py shows up here automatically on next launch). --help,
--models (handled elsewhere - the open model), and --print-only (a
run-time flag, not a configuration value) are never rendered.

Widget-per-action-kind rules:
  store_true              -> QCheckBox
  choices                 -> QComboBox (a blank first entry stands in for
                              "unset" when the action's own default is
                              None, e.g. --backend/--rough-fresnel; when
                              the default is a real value, e.g. --preset,
                              the combo starts on that value instead)
  append                  -> one-line QLineEdit, semicolon-separated
                              values, split into repeated flags by
                              RunController.build_args()
  type=int (non-append)   -> QSpinBox, 0..1e9, with 0 as the "unset -
                              fall back to preset/default" sentinel
                              (specialValueText shows what that fallback
                              currently is)
  type=float (non-append) -> QLineEdit + QDoubleValidator; empty = unset
  plain str (non-append)  -> QLineEdit; empty = unset

values() only returns entries that differ from the parser default (the
"unset" states above all map back to "equal to default", i.e. omitted),
so to_args() never forwards a flag the pipeline would have picked anyway.
"""

import json
import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import common     # noqa: E402  (stdlib-only shared contract hub)
import cli_specs   # noqa: E402  (stdlib-only; single source of truth for CLIs)

from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

EXCLUDED_DESTS = ("help", "models", "print_only", "preset")


def _fmt_num(value):
    return ("%g" % value) if isinstance(value, float) else str(value)


class ConfigMatrix(QWidget):
    estimateRequested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parser = cli_specs.build_parser("pipeline")
        self.widgets = {}     # dest -> widget
        self.actions = {}     # dest -> argparse.Action

        self._preset_action = self._find_action("preset")

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(sorted(common.PRESETS))
        self.preset_combo.setCurrentText(self._preset_action.default)
        self.preset_combo.setToolTip(
            self._preset_action.help
            or "Fidelity preset (fills rays/resolution/nlambda/"
               "spectral-bins/viz-rays defaults)")
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)

        self.estimate_button = QPushButton("Estimate runtime")
        self.estimate_button.setToolTip(
            "Estimate wall-clock runtime and accumulator memory for the "
            "current rays/resolution/nlambda/backend settings")
        self.estimate_button.clicked.connect(
            lambda: self.estimateRequested.emit(self.estimate_params()))

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Preset:"))
        top_row.addWidget(self.preset_combo)
        top_row.addStretch(1)
        top_row.addWidget(self.estimate_button)

        groups_layout = QVBoxLayout()
        for group in self._parser._action_groups:
            actions = [a for a in group._group_actions
                      if a.dest not in EXCLUDED_DESTS and a.option_strings]
            if not actions:
                continue
            box = QGroupBox(group.title or "options")
            form = QFormLayout()
            for action in actions:
                widget = self._make_widget(action)
                self.widgets[action.dest] = widget
                self.actions[action.dest] = action
                form.addRow(action.option_strings[-1], widget)
            box.setLayout(form)
            groups_layout.addWidget(box)
        groups_layout.addStretch(1)

        scroll_body = QWidget()
        scroll_body.setLayout(groups_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_body)

        outer = QVBoxLayout(self)
        outer.addLayout(top_row)
        outer.addWidget(scroll)

        self._on_preset_changed(self.preset_combo.currentText())

    # -- construction helpers -----------------------------------------------
    def _find_action(self, dest):
        for action in self._parser._actions:
            if action.dest == dest:
                return action
        raise KeyError("no such pipeline option: %r" % dest)

    def _make_widget(self, action):
        kind = type(action).__name__
        if kind == "_StoreTrueAction":
            widget = QCheckBox()
            widget.setChecked(False)
        elif action.choices:
            widget = QComboBox()
            items = [str(c) for c in action.choices]
            if action.default is None:
                widget.addItem("")
                widget.addItems(items)
                widget.setCurrentText("")
            else:
                widget.addItems(items)
                widget.setCurrentText(str(action.default))
        elif kind == "_AppendAction":
            widget = QLineEdit()
            widget.setPlaceholderText("value1;value2;...")
        elif action.type is int:
            widget = QSpinBox()
            widget.setRange(0, 10 ** 9)
            widget.setSpecialValueText("(default)")
            widget.setValue(0)
        elif action.type is float:
            widget = QLineEdit()
            widget.setValidator(QDoubleValidator())
        else:
            widget = QLineEdit()
            if action.default:
                widget.setPlaceholderText(str(action.default))
        widget.setToolTip(action.help or action.option_strings[-1])
        return widget

    # -- preset -> placeholder wiring (never overwrites entered values) ------
    _PRESET_SENSITIVE_INT = ("resolution", "nlambda", "spectral_bins")

    def _on_preset_changed(self, preset_name):
        preset_vals = common.PRESETS.get(preset_name)
        if preset_vals is None:
            return
        rays_widget = self.widgets.get("rays")
        if rays_widget is not None:
            rays_widget.setPlaceholderText(_fmt_num(preset_vals["rays"]))
        for dest in self._PRESET_SENSITIVE_INT:
            widget = self.widgets.get(dest)
            if widget is not None:
                widget.setSpecialValueText(
                    "(preset: %s)" % _fmt_num(preset_vals[dest]))

    # -- values <-> widgets ---------------------------------------------------
    def values(self):
        """{dest: value} for every widget whose value differs from the
        pipeline parser's own default (plus "preset" if changed)."""
        out = {}
        current_preset = self.preset_combo.currentText()
        if current_preset and current_preset != self._preset_action.default:
            out["preset"] = current_preset

        for dest, widget in self.widgets.items():
            action = self.actions[dest]
            kind = type(action).__name__
            if isinstance(widget, QCheckBox):
                if widget.isChecked():
                    out[dest] = True
            elif kind == "_AppendAction":
                items = [seg.strip() for seg in widget.text().split(";")
                        if seg.strip()]
                if items:
                    if action.type is float:
                        out[dest] = [float(x) for x in items]
                    elif action.type is int:
                        out[dest] = [int(x) for x in items]
                    else:
                        out[dest] = items
            elif isinstance(widget, QComboBox):
                text = widget.currentText()
                if not text:
                    continue
                if action.default is not None and text == str(action.default):
                    continue
                out[dest] = text
            elif isinstance(widget, QSpinBox):
                value = widget.value()
                if value != 0:
                    out[dest] = value
            elif isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if not text:
                    continue
                out[dest] = float(text) if action.type is float else text
        return out

    def set_values(self, values):
        """Apply {dest: value} (as produced by values()) to the widgets.
        Destinations not present in `values` are left untouched."""
        if "preset" in values:
            self.preset_combo.setCurrentText(str(values["preset"]))
        for dest, widget in self.widgets.items():
            if dest not in values:
                continue
            value = values[dest]
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QLineEdit):
                action = self.actions[dest]
                if type(action).__name__ == "_AppendAction":
                    widget.setText(";".join(str(v) for v in value))
                else:
                    widget.setText(str(value))

    def reset_to_defaults(self):
        """Session boundary (File > Open/New/Close): put every widget back
        to its parser default so the previous project's run config can't
        leak into the next one. Mirrors _make_widget's initial state; a
        .MieWB open re-applies its own simparams.json right after."""
        self.preset_combo.setCurrentText(self._preset_action.default)
        for dest, widget in self.widgets.items():
            action = self.actions[dest]
            if isinstance(widget, QCheckBox):
                widget.setChecked(False)
            elif isinstance(widget, QComboBox):
                widget.setCurrentText("" if action.default is None
                                      else str(action.default))
            elif isinstance(widget, QSpinBox):
                widget.setValue(0)
            elif isinstance(widget, QLineEdit):
                widget.clear()

    # -- args / json ------------------------------------------------------------
    def to_args(self):
        from mieworkbench.core.runner import RunController
        return RunController.build_args(self.values())

    def to_json(self):
        return json.dumps(self.values(), indent=1, sort_keys=True)

    def from_json(self, text):
        self.set_values(json.loads(text))

    # -- estimate -----------------------------------------------------------------
    def estimate_params(self):
        """Resolve current widget values (falling back to the active
        preset) into kwargs for common.estimate()."""
        preset = self.preset_combo.currentText() or self._preset_action.default
        preset_vals = common.PRESETS[preset]
        vals = self.values()
        return {
            "rays": vals.get("rays", preset_vals["rays"]),
            "resolution": vals.get("resolution", preset_vals["resolution"]),
            "nlambda": vals.get("nlambda", preset_vals["nlambda"]),
            "backend": vals.get("backend") or common.DEFAULTS["backend"],
            "n_coherent_sources": 1,
            "n_detectors": 1,
            "save_fields": bool(vals.get("save_fields", False)),
            "n_pol_strata": 1,
        }
