"""Settings - persistent tool-path / directory configuration for MieWorkbench.

Wraps QSettings("CurtisAnalytical", "MieWorkbench") so the GUI remembers
where the pinned interpreters and data directories live across sessions,
without ever hard-coding a machine-specific path in this module. The
*defaults* (what a fresh install shows before the user changes anything)
come straight from scripts/common.py - the same stdlib-only module the
pipeline scripts themselves use to resolve MIEWB_FREECAD / MIEWB_OPTICS_
PYTHON / MIEWB_PVPYTHON / MIEWB_GEOMETRY_DIR / MIEWB_RESULTS_DIR /
MIEWB_OPTPROPS_DIR - so "default" already reflects any of those env vars
that happened to be set when the GUI process started.

RunController reads env_overrides() and layers it on top of the current
process environment before launching run_pipeline.py, so a value the user
changed in the Settings dialog reaches every pipeline subprocess without
the user having to export anything by hand.
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import common  # noqa: E402  (stdlib-only shared contract hub)

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QTabWidget,
    QVBoxLayout, QWidget, QFileDialog, QDialogButtonBox,
)

ORG_NAME = "CurtisAnalytical"
APP_NAME = "MieWorkbench"

# settings key -> (env var name, common.py default, "file" | "dir", label)
FIELDS = [
    ("freecad", "MIEWB_FREECAD", common.FREECAD_APPIMAGE, "file",
     "FreeCAD AppImage"),
    ("optics_python", "MIEWB_OPTICS_PYTHON", common.OPTICS_PYTHON, "file",
     "Optics env python"),
    ("pvpython", "MIEWB_PVPYTHON", common.PVPYTHON, "file", "pvpython"),
    ("geometry_dir", "MIEWB_GEOMETRY_DIR", common.GEOMETRY_DIR, "dir",
     "Geometry cache dir"),
    ("results_dir", "MIEWB_RESULTS_DIR", common.RESULTS_DIR, "dir",
     "Results dir"),
    ("optprops_dir", "MIEWB_OPTPROPS_DIR", common.OPTPROPS_DIR, "dir",
     "Optical properties dir"),
]


class Settings:
    """Thin wrapper over QSettings; all values are plain strings (paths)."""

    def __init__(self):
        self._qs = QSettings(ORG_NAME, APP_NAME)

    # -- defaults -------------------------------------------------------------
    @staticmethod
    def default(key):
        for k, _env, default, _kind, _label in FIELDS:
            if k == key:
                return str(default)
        raise KeyError("unknown settings key %r" % key)

    # -- generic get/set --------------------------------------------------------
    def get(self, key, default=None):
        stored = self._qs.value(key, None)
        if stored is not None and stored != "":
            return stored
        if default is not None:
            return default
        try:
            return self.default(key)
        except KeyError:
            return None

    def set(self, key, value):
        self._qs.setValue(key, value)
        self._qs.sync()

    # -- boolean UI preferences (view toggles etc.) ----------------------------
    # stored as "true"/"false" strings -- this wrapper is strings-only by
    # design (see FIELDS), and QSettings round-trips booleans
    # platform-dependently, so normalize explicitly.
    def get_bool(self, key, default):
        stored = self._qs.value(key, None)
        if stored is None or stored == "":
            return bool(default)
        return str(stored).strip().lower() in ("true", "1", "yes")

    def set_bool(self, key, value):
        self._qs.setValue(key, "true" if value else "false")
        self._qs.sync()

    # -- convenience accessors used by RunController / other panes ------------
    def freecad(self):
        return self.get("freecad")

    def optics_python(self):
        return self.get("optics_python")

    def pvpython(self):
        return self.get("pvpython")

    def geometry_dir(self):
        return self.get("geometry_dir")

    def results_dir(self):
        return self.get("results_dir")

    def optprops_dir(self):
        return self.get("optprops_dir")

    # -- autodetect / diagnostics ---------------------------------------------
    def autodetect(self):
        """Return {key: {"path": str, "exists": bool}} for every field."""
        report = {}
        for key, _env, _default, _kind, _label in FIELDS:
            path = self.get(key)
            report[key] = {"path": path, "exists": os.path.exists(path)}
        return report

    def env_overrides(self):
        """{"MIEWB_FREECAD": "...", ...} for values that differ from the
        common.py default - what RunController layers onto a subprocess
        environment."""
        out = {}
        for key, env_var, default, _kind, _label in FIELDS:
            value = self.get(key)
            if str(value) != str(default):
                out[env_var] = str(value)
        return out


# ---------------------------------------------------------------------------
# SettingsDialog - a QTabWidget: "Tool Paths" (path fields + Browse +
# autodetect status per row) and "Defaults" (ray extinction + tracer-bead
# animation -- the SAME live QSettings keys the View menu/toolbars edit,
# pushed into the open session on OK). Reachable from File > Settings…
# ---------------------------------------------------------------------------
class SettingsDialog(QDialog):
    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MieWorkbench Settings")
        self.setToolTip("Configure pinned tool paths and data directories")
        self.settings = settings or Settings()

        self._edits = {}
        self._status_labels = {}

        form = QFormLayout()
        for key, env_var, default, kind, label in FIELDS:
            edit = QLineEdit(str(self.settings.get(key)))
            edit.setToolTip(
                "%s (env override: %s; default: %s)" % (label, env_var,
                                                         default))
            browse = QPushButton("Browse…")
            browse.setToolTip("Choose the %s for %s" % (
                "file" if kind == "file" else "directory", label))
            status = QLabel()
            status.setToolTip("Whether this path currently exists on disk")

            def make_browse_handler(edit=edit, kind=kind, label=label):
                def handler():
                    if kind == "file":
                        path, _ = QFileDialog.getOpenFileName(
                            self, "Choose %s" % label, edit.text())
                    else:
                        path = QFileDialog.getExistingDirectory(
                            self, "Choose %s" % label, edit.text())
                    if path:
                        edit.setText(path)
                        self._refresh_status()
                return handler

            browse.clicked.connect(make_browse_handler())
            edit.textChanged.connect(self._refresh_status)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(edit, 1)
            row_layout.addWidget(browse)
            row_layout.addWidget(status)
            form.addRow(label + ":", row)

            self._edits[key] = edit
            self._status_labels[key] = status

        self._refresh_status()

        paths_page = QWidget()
        paths_page.setLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        tabs = QTabWidget()
        tabs.addTab(paths_page, "Tool Paths")
        tabs.addTab(self._build_defaults_page(), "Defaults")

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    # -- Defaults tab (extinction + animation live keys) -----------------------
    def _num(self, key, default):
        try:
            return float(self.settings.get(key, str(default)) or default)
        except (TypeError, ValueError):
            return float(default)

    def _build_defaults_page(self):
        page = QWidget()
        form = QFormLayout(page)

        self.dim_mode_combo = QComboBox()
        for label in ("Off", "Linear", "Perceptual"):
            self.dim_mode_combo.addItem(label)
        modes = ("off", "linear", "sqrt")
        current = self.settings.get("ray_dimming_mode", "off")
        self.dim_mode_combo.setCurrentIndex(
            modes.index(current) if current in modes else 0)
        self.dim_mode_combo.setToolTip(
            "Ray extinction: fade ray segments by remaining power "
            "(same setting as the toolbar combo / View menu)")
        form.addRow("Ray extinction:", self.dim_mode_combo)

        self.dim_floor_spin = QDoubleSpinBox()
        self.dim_floor_spin.setRange(0.0, 100.0)
        self.dim_floor_spin.setSuffix(" %")
        self.dim_floor_spin.setValue(self._num("ray_dimming_floor", 0.0))
        self.dim_floor_spin.setToolTip(
            "Minimum segment opacity under extinction (0 = fade fully)")
        form.addRow("Extinction floor:", self.dim_floor_spin)

        self.anim_enabled_check = QCheckBox("Show tracer beads")
        self.anim_enabled_check.setChecked(
            self.settings.get_bool("anim_enabled", False))
        form.addRow("Animation:", self.anim_enabled_check)

        self.anim_size_spin = QDoubleSpinBox()
        self.anim_size_spin.setRange(0.05, 50.0)
        self.anim_size_spin.setSuffix(" mm")
        self.anim_size_spin.setValue(self._num("anim_bead_size", 1.0))
        form.addRow("Bead size:", self.anim_size_spin)

        self.anim_speed_spin = QDoubleSpinBox()
        self.anim_speed_spin.setRange(0.01, 1000.0)
        self.anim_speed_spin.setSuffix(" mm/s")
        self.anim_speed_spin.setValue(self._num("anim_speed_mm_s", 2.0))
        self.anim_speed_spin.setToolTip(
            "mm of ray path per real second for a vacuum bead "
            "(glass beads run slower by 1/n)")
        form.addRow("Bead speed:", self.anim_speed_spin)

        self.anim_fps_spin = QSpinBox()
        self.anim_fps_spin.setRange(1, 120)
        self.anim_fps_spin.setValue(int(self._num("anim_fps", 15)))
        form.addRow("Animation FPS:", self.anim_fps_spin)

        self.anim_cap_spin = QSpinBox()
        self.anim_cap_spin.setRange(0, 100000)
        self.anim_cap_spin.setValue(int(self._num("anim_ray_cap", 300)))
        self.anim_cap_spin.setToolTip(
            "Max beads DRAWN simultaneously per source (0 = unlimited); "
            "an honest render cap on busy run overlays, not a trace cap")
        form.addRow("Bead cap / source:", self.anim_cap_spin)

        note = QLabel(
            "These are the persisted values every session starts from — "
            "the toolbar and View-menu controls edit the same settings.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        form.addRow(note)
        return page

    def _refresh_status(self):
        for key, _env, _default, _kind, _label in FIELDS:
            path = self._edits[key].text()
            ok = os.path.exists(path)
            label = self._status_labels[key]
            label.setText("found" if ok else "missing")
            label.setStyleSheet(
                "color: #2ecc71;" if ok else "color: #e74c3c;")

    def _on_accept(self):
        for key, _env, _default, _kind, _label in FIELDS:
            self.settings.set(key, self._edits[key].text())

        # Defaults tab -> the same live keys the toolbars/menus edit
        modes = ("off", "linear", "sqrt")
        mode = modes[self.dim_mode_combo.currentIndex()]
        self.settings.set("ray_dimming_mode", mode)
        self.settings.set("ray_dimming_floor",
                          str(self.dim_floor_spin.value()))
        self.settings.set_bool("anim_enabled",
                               self.anim_enabled_check.isChecked())
        self.settings.set("anim_bead_size",
                          str(self.anim_size_spin.value()))
        self.settings.set("anim_speed_mm_s",
                          str(self.anim_speed_spin.value()))
        self.settings.set("anim_fps", str(self.anim_fps_spin.value()))
        self.settings.set("anim_ray_cap", str(self.anim_cap_spin.value()))

        # push into the open session (parent is the MainWindow at the
        # only call site; guarded so the dialog stays usable standalone)
        parent = self.parent()
        if parent is not None and hasattr(parent, "_on_ray_dimming_mode"):
            parent._on_ray_dimming_mode(mode)
            parent._set_ray_dimming_floor(self.dim_floor_spin.value())
        if parent is not None and hasattr(parent, "anim_controller"):
            parent.anim_controller.apply_settings(
                bead_size_mm=self.anim_size_spin.value(),
                speed_mm_s=self.anim_speed_spin.value(),
                fps=self.anim_fps_spin.value(),
                ray_cap=self.anim_cap_spin.value())
            # route enabled through the shared action so menu/toolbar
            # check-state follows
            parent.anim_enable_action.setChecked(
                self.anim_enabled_check.isChecked())
            # sync the toolbar spin/combo editors
            parent.anim_size_spin.setValue(self.anim_size_spin.value())
            parent.anim_speed_spin.setValue(self.anim_speed_spin.value())
            parent.anim_fps_combo.setCurrentText(
                str(self.anim_fps_spin.value()))
        self.accept()
