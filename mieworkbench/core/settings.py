"""Settings - tool-path / directory configuration for MieWorkbench.

Machine paths live in ONE place: <repo>/miewb.env (created by
scripts/setup_env.sh, parsed by scripts/common.py). This module's path
fields read and WRITE that file directly - a change in the Settings
dialog is immediately visible to CLI runs and vice versa. QSettings
("CurtisAnalytical", "MieWorkbench") now stores only non-path UI
preferences (booleans, animation keys, the RunDialog session skip...).
Legacy QSettings-stored paths are migrated into miewb.env once (values
equal to the old baked defaults are discarded - the old dialog stored
untouched defaults too).

Resolution per path field: exported MIEWB_* env var (locked in the
dialog) > live miewb.env entry (re-read per call, so dialog and
controllers never go stale) > repo-derived default for directories /
None for tools (unconfigured or configured absent).

RunController (and the optimize/tolerance controllers) read
env_overrides() and layer it onto the subprocess environment before
launching run_pipeline.py, so every child resolves exactly what the
dialog shows even if the shell exported something stale.
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import common  # noqa: E402  (stdlib-only shared contract hub)

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QTabWidget,
    QVBoxLayout, QWidget, QFileDialog, QDialogButtonBox,
)

ORG_NAME = "CurtisAnalytical"
APP_NAME = "MieWorkbench"

# settings key -> (env var name, "file" | "dir", label)
FIELDS = [
    ("freecad", "MIEWB_FREECAD", "file", "FreeCAD AppImage"),
    ("optics_python", "MIEWB_OPTICS_PYTHON", "file", "Optics env python"),
    ("pvpython", "MIEWB_PVPYTHON", "file", "pvpython"),
    ("geometry_dir", "MIEWB_GEOMETRY_DIR", "dir", "Geometry cache dir"),
    ("results_dir", "MIEWB_RESULTS_DIR", "dir", "Results dir"),
    ("optprops_dir", "MIEWB_OPTPROPS_DIR", "dir", "Optical properties dir"),
]

_FIELD_BY_KEY = {f[0]: f for f in FIELDS}

# Repo-derived directory defaults (tools have NO default any more).
_DIR_DEFAULTS = {
    "geometry_dir": str(common.PROJECT_DIR / "geometry"),
    "results_dir": str(common.PROJECT_DIR / "results"),
    "optprops_dir": str(common.PROJECT_DIR / "opticalproperties"),
}

# The pre-miewb.env baked defaults, FROZEN for the one-time QSettings
# migration only: the old dialog stored every field on OK (including
# untouched defaults), so a stored value equal to one of these means
# "not user intent" and must NOT be migrated onto a fresh miewb.env.
# This is the single sanctioned absolute-path literal block left in the GUI.
_LEGACY_DEFAULTS = {
    "freecad": ["/home3/freecad/FreeCAD.AppImage"],
    "optics_python": ["/home3/optics/env/bin/python"],
    "pvpython": ["/home3/paraview/ParaView-6.1.1-MPI-Linux-Python3.12"
                 "-x86_64/bin/pvpython"],
    "geometry_dir": [_DIR_DEFAULTS["geometry_dir"]],
    "results_dir": [_DIR_DEFAULTS["results_dir"]],
    "optprops_dir": [_DIR_DEFAULTS["optprops_dir"]],
}

_MIGRATION_FLAG = "paths_migrated_to_miewb_env"


class Settings:
    """Path fields resolve through miewb.env; everything else QSettings."""

    def __init__(self, env_file=None):
        self._qs = QSettings(ORG_NAME, APP_NAME)
        self._env_file = str(env_file) if env_file else str(common.ENV_FILE)
        self._migrate_qsettings_paths()

    @property
    def env_file(self):
        return self._env_file

    # -- defaults -------------------------------------------------------------
    @staticmethod
    def default(key):
        """Repo-derived default for directory fields; '' for tools (no
        baked default any more - miewb.env is the source of truth)."""
        if key in _DIR_DEFAULTS:
            return _DIR_DEFAULTS[key]
        if key in _FIELD_BY_KEY:
            return ""
        raise KeyError("unknown settings key %r" % key)

    # -- path-field resolution --------------------------------------------------
    def env_locked(self, key):
        """True when an exported env var overrides this field (the dialog
        shows it read-only; unset the variable to edit here)."""
        field = _FIELD_BY_KEY.get(key)
        return field is not None and field[1] in os.environ

    def _resolve_path_field(self, key):
        """env var > live miewb.env > dir default / None (tools).
        '' anywhere = configured absent -> None."""
        env_var = _FIELD_BY_KEY[key][1]
        if env_var in os.environ:
            return os.environ[env_var] or None
        try:
            cfg = common.load_env_file(self._env_file)
        except ValueError:
            cfg = {}
        if env_var in cfg:
            return cfg[env_var] or None
        return _DIR_DEFAULTS.get(key)

    # -- generic get/set --------------------------------------------------------
    def get(self, key, default=None):
        if key in _FIELD_BY_KEY:
            resolved = self._resolve_path_field(key)
            if resolved is not None:
                return resolved
            return default
        stored = self._qs.value(key, None)
        if stored is not None and stored != "":
            return stored
        return default

    def set(self, key, value):
        if key in _FIELD_BY_KEY:
            common.update_env_file(
                {_FIELD_BY_KEY[key][1]: "" if value is None else str(value)},
                self._env_file)
            return
        self._qs.setValue(key, value)
        self._qs.sync()

    # -- one-time QSettings -> miewb.env migration ------------------------------
    def _migrate_qsettings_paths(self):
        if self._qs.value(_MIGRATION_FLAG, None):
            return
        try:
            cfg = common.load_env_file(self._env_file)
        except ValueError:
            cfg = {}
        moved = {}
        for key, env_var, _kind, _label in FIELDS:
            stored = self._qs.value(key, None)
            if stored in (None, ""):
                continue
            if str(stored) in _LEGACY_DEFAULTS[key]:
                continue  # stored-but-untouched old default, not user intent
            if env_var not in cfg:  # miewb.env / setup_env.sh wins otherwise
                moved[env_var] = str(stored)
        if moved:
            common.update_env_file(moved, self._env_file)
        for key, _env, _kind, _label in FIELDS:
            self._qs.remove(key)
        self._qs.setValue(_MIGRATION_FLAG, "true")
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
        """Return {key: {"path": str|None, "exists": bool}} per field."""
        report = {}
        for key, _env, _kind, _label in FIELDS:
            path = self.get(key)
            report[key] = {"path": path,
                           "exists": bool(path) and os.path.exists(path)}
        return report

    def env_overrides(self):
        """{"MIEWB_FREECAD": "...", ...} - the RESOLVED value of every
        path field (a None tool becomes "" = configured absent), layered
        onto subprocess environments by the run/optimize/tolerance
        controllers so children agree with the dialog even if the shell
        exported something stale."""
        out = {}
        for key, env_var, _kind, _label in FIELDS:
            value = self.get(key)
            out[env_var] = "" if value is None else str(value)
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
        self._initial = {}

        form = QFormLayout()
        header = QLabel(
            "Paths are stored in <b>%s</b> (single source of truth, "
            "shared with CLI runs; created by scripts/setup_env.sh). "
            "Leave a field empty for a tool this machine doesn't have."
            % self.settings.env_file)
        header.setWordWrap(True)
        form.addRow(header)
        for key, env_var, kind, label in FIELDS:
            value = self.settings.get(key)
            self._initial[key] = "" if value is None else str(value)
            edit = QLineEdit(self._initial[key])
            if self.settings.env_locked(key):
                edit.setEnabled(False)
                edit.setToolTip(
                    "%s is overridden by the exported %s environment "
                    "variable — unset it to edit here" % (label, env_var))
            else:
                edit.setToolTip("%s (miewb.env key: %s)" % (label, env_var))
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

    # -- Defaults tab (pointer to the Preview Configuration dialog) ------------
    def _build_defaults_page(self):
        """Pointer page: the extinction + tracer-bead defaults moved into
        the consolidated Preview Configuration dialog (MainWindow's
        _open_preview_dialog). The button is hasattr-guarded so this
        dialog stays constructible standalone (tests, no MainWindow)."""
        page = QWidget()
        form = QFormLayout(page)
        note = QLabel(
            "Ray extinction and tracer-bead animation defaults now live "
            "in the Preview Configuration dialog (Rays toolbar button ▸ "
            "\"Live ray preview…\", or Simulation Settings ▸ Ray "
            "Preview). The toolbar and View-menu controls edit the same "
            "settings.")
        note.setWordWrap(True)
        form.addRow(note)
        parent = self.parent()
        self.preview_config_button = QPushButton(
            "Open Preview Configuration…")
        self.preview_config_button.setEnabled(
            parent is not None and hasattr(parent, "_open_preview_dialog"))
        if self.preview_config_button.isEnabled():
            self.preview_config_button.clicked.connect(
                lambda: parent._open_preview_dialog(launch=False))
        form.addRow(self.preview_config_button)
        return page

    def _refresh_status(self):
        for key, _env, _kind, _label in FIELDS:
            path = self._edits[key].text().strip()
            label = self._status_labels[key]
            if not path:
                label.setText("absent (configured)")
                label.setStyleSheet("color: #f39c12;")
                continue
            ok = os.path.exists(path)
            label.setText("found" if ok else "missing")
            label.setStyleSheet(
                "color: #2ecc71;" if ok else "color: #e74c3c;")

    def _on_accept(self):
        updates = {}
        for key, env_var, _kind, _label in FIELDS:
            if self.settings.env_locked(key):
                continue
            text = self._edits[key].text().strip()
            if text != self._initial[key]:
                updates[env_var] = text
        if updates:
            common.update_env_file(updates, self.settings.env_file)
        self.accept()
