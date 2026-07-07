"""MainWindow - Zemax-inspired multi-pane shell for MieWorkbench.

Fully integrated: the central VTK optical-train view, single-element
inspector with face picking, element property editor, transform panel,
library (primitives + property libraries), console + stage chips, results
viewer (with ParaView handoff and live monitor mode), problems pane, and
the .FCStd / .MieWB / .MieSim open/save/run flows.

File-format flows
-----------------
open .FCStd   -> live FreeCAD session on that file (in place).
open .MieWB   -> explode into var/work/<name>-<hash>/, open the workspace
                 model, load simparams into the config matrix; runs point
                 MIEWB_GEOMETRY_DIR/MIEWB_RESULTS_DIR into the workspace
                 and Save repacks the archive.
open .MieSim  -> view results (monitor mode if the case is live); the
                 user can open the embedded workbench for editing/rerun,
                 in which case a successful run REPLACES the .MieSim.
"""

import hashlib
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "scripts")))
import common  # noqa: E402  (stdlib-only shared contract hub)
import miewb_tool  # noqa: E402  (stdlib-only archive engine)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QDockWidget, QFileDialog, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QStyle, QVBoxLayout,
    QWidget,
)

from .core.librarymgr import LibraryManager
from .core.project import Project, ProjectError
from .core.runner import RunController
from .core.settings import Settings, SettingsDialog
from .panes.config_matrix import ConfigMatrix
from .panes.console import ConsolePane
from .panes.element_editor import ElementEditorPane
from .panes.inspector3d import InspectorPane
from .panes.library import LibraryPane
from .panes.problems import ProblemsPane
from .panes.prop_editor import PropEditorPane
from .panes.results import ResultsPane
from .panes.scene3d import Scene3DPane
from .panes.transform_panel import TransformPanel
from .panes.wizard_dialog import ElementWizardDialog

STAGE_ORDER = ["extract", "trace", "post", "viz"]

_CHIP_COLORS = {
    "running": "#3b82f6",
    "completed": "#22c55e",
    "estimated": "#22c55e",
    "failed": "#ef4444",
}
_CHIP_DEFAULT = "#6b7280"

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MieWorkbench")
        self.resize(1500, 950)

        self.model_path = None          # the .FCStd the pipeline runs on
        self.opened_path = None         # what the user opened (any format)
        self.workspace = None           # exploded .MieWB/.MieSim dir
        self.miewb_path = None          # archive to repack on Save
        self.miesim_out = None          # .MieSim to update after a rerun
        self._has_validation_errors = False

        self.settings = Settings()
        self.project = Project(self.settings)
        self.runner = RunController(self.settings, self)
        self.config_matrix = ConfigMatrix()
        self.config_matrix.estimateRequested.connect(self._show_estimate)
        self.library_manager = LibraryManager(
            os.path.join(REPO, "opticalproperties"),
            os.path.join(REPO, "primitives"))
        self._prop_editor_window = None

        self._build_central()
        self.stage_chips = {}
        self._build_docks()
        self._build_menus()
        self._build_toolbar()
        self.statusBar().showMessage("Ready")
        self._wire_runner()
        self._wire_panes()
        self._update_window_title()

    # -- central --------------------------------------------------------------
    def _build_central(self):
        self.scene3d = Scene3DPane()
        self.scene3d.setObjectName("scene3d_host")
        self.setCentralWidget(self.scene3d)

    # -- docks ------------------------------------------------------------------
    def _build_docks(self):
        self.inspector = InspectorPane()
        self.inspector.setObjectName("inspector_host")
        self.inspector_dock = self._add_dock(
            "Element Inspector", "inspector_dock", self.inspector,
            Qt.DockWidgetArea.LeftDockWidgetArea)

        self.element_editor = ElementEditorPane()
        self.element_editor.setObjectName("element_editor_host")
        self.element_editor_dock = self._add_dock(
            "Element Properties", "element_editor_dock",
            self.element_editor, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.splitDockWidget(self.inspector_dock, self.element_editor_dock,
                             Qt.Orientation.Vertical)

        self.transform_panel = TransformPanel()
        self.transform_panel.setObjectName("transform_host")
        self.transform_dock = self._add_dock(
            "Position / Orientation", "transform_dock",
            self.transform_panel, Qt.DockWidgetArea.RightDockWidgetArea)

        self.library = LibraryPane(self.library_manager)
        self.library.setObjectName("library_host")
        self.library_dock = self._add_dock(
            "Library", "library_dock", self.library,
            Qt.DockWidgetArea.RightDockWidgetArea)
        self.splitDockWidget(self.transform_dock, self.library_dock,
                             Qt.Orientation.Vertical)

        self.console_dock = self._add_dock(
            "Console", "console_dock", self._build_bottom_widget(),
            Qt.DockWidgetArea.BottomDockWidgetArea)

        self.results = ResultsPane(self.settings)
        self.results.setObjectName("results_host")
        self.results_dock = self._add_dock(
            "Results", "results_dock", self.results,
            Qt.DockWidgetArea.BottomDockWidgetArea)

        self.problems = ProblemsPane()
        self.problems.setObjectName("problems_host")
        self.problems_dock = self._add_dock(
            "Problems", "problems_dock", self.problems,
            Qt.DockWidgetArea.BottomDockWidgetArea)

        self.tabifyDockWidget(self.console_dock, self.results_dock)
        self.tabifyDockWidget(self.console_dock, self.problems_dock)
        self.console_dock.raise_()
        self.resizeDocks([self.console_dock], [230],
                         Qt.Orientation.Vertical)

    def _add_dock(self, title, object_name, widget, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setWidget(widget)
        dock.toggleViewAction().setToolTip("Show/hide the %s pane" % title)
        self.addDockWidget(area, dock)
        return dock

    def _build_bottom_widget(self):
        container = QWidget()
        container.setObjectName("console_host")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)

        strip = QHBoxLayout()
        for stage in STAGE_ORDER:
            chip = QLabel(stage)
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setToolTip("%s stage status (blue=running, green=done, "
                            "red=failed)" % stage)
            chip.setStyleSheet(self._chip_style(_CHIP_DEFAULT))
            self.stage_chips[stage] = chip
            strip.addWidget(chip)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setToolTip("Overall pipeline progress")
        strip.addWidget(self.progress_bar, 1)

        self.console = ConsolePane()

        layout.addLayout(strip)
        layout.addWidget(self.console)
        return container

    @staticmethod
    def _chip_style(color):
        return ("QLabel { background-color: %s; color: white; "
                "border-radius: 8px; padding: 2px 10px; }" % color)

    # -- menus / toolbar --------------------------------------------------------
    def _build_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        act = file_menu.addAction("&Open…")
        act.setToolTip("Open an optical model or archive "
                       "(.FCStd / .MieWB / .MieSim)")
        act.triggered.connect(self._on_open)

        act = file_menu.addAction("Open &Results / Case…")
        act.setToolTip("View a results case directory (live cases open "
                       "read-only in monitor mode)")
        act.triggered.connect(self._on_open_case)

        self.save_action = file_menu.addAction("&Save")
        self.save_action.setToolTip("Save the model (.FCStd), repacking "
                                    "the .MieWB archive in workspace mode")
        self.save_action.triggered.connect(self._on_save)
        self.save_action.setEnabled(False)

        self.save_as_action = file_menu.addAction("Save &As…")
        self.save_as_action.setToolTip(
            "Save as a bare .FCStd model or a self-contained .MieWB "
            "workbench archive")
        self.save_as_action.triggered.connect(self._on_save_as)
        self.save_as_action.setEnabled(False)

        act = file_menu.addAction("&Export Run Script…")
        act.setToolTip("Pack a .MieWB and write a standalone shell script "
                       "that runs it headlessly (e.g. on a remote server) "
                       "producing a .MieSim")
        act.triggered.connect(self._on_export_script)

        file_menu.addSeparator()
        act = file_menu.addAction("&Settings…")
        act.setToolTip("Configure tool paths and data directories")
        act.triggered.connect(self._on_settings)

        file_menu.addSeparator()
        act = file_menu.addAction("&Quit")
        act.setToolTip("Exit MieWorkbench")
        act.triggered.connect(self.close)

        sim_menu = menubar.addMenu("&Simulation")
        self.run_action = sim_menu.addAction("&Run Pipeline…")
        self.run_action.setToolTip(
            "Configure and run the extract/trace/post/viz pipeline")
        self.run_action.triggered.connect(self._on_run_pipeline_dialog)

        self.estimate_action = sim_menu.addAction("&Estimate Runtime")
        self.estimate_action.setToolTip(
            "Estimate wall-clock runtime and memory for the current "
            "configuration without running anything")
        self.estimate_action.triggered.connect(self._on_estimate)

        self.dry_run_action = sim_menu.addAction("&Dry Run")
        self.dry_run_action.setToolTip(
            "Run the pipeline with --dry-run (trace estimates only, no "
            "post/viz)")
        self.dry_run_action.triggered.connect(self._on_dry_run)

        self.validate_action = sim_menu.addAction("&Validate Scene")
        self.validate_action.setToolTip(
            "Check the scene for missing information and likely errors")
        self.validate_action.triggered.connect(self._on_validate)

        self.stop_action = sim_menu.addAction("&Stop")
        self.stop_action.setToolTip("Stop the running pipeline")
        self.stop_action.triggered.connect(self.runner.stop)

        tools_menu = menubar.addMenu("&Tools")
        act = tools_menu.addAction("&Property Library Editor…")
        act.setToolTip("View/edit/import optical property definitions "
                       "(materials, coatings, polarizers, filters, "
                       "gratings, birefringence)")
        act.triggered.connect(lambda: self._open_prop_editor())

        act = tools_menu.addAction("Open in &ParaView")
        act.setToolTip("Launch interactive ParaView on the loaded case's "
                       ".vtp data")
        act.triggered.connect(self.results._open_paraview)

        view_menu = menubar.addMenu("&View")
        for dock in (self.inspector_dock, self.element_editor_dock,
                     self.transform_dock, self.library_dock,
                     self.console_dock, self.results_dock,
                     self.problems_dock):
            view_menu.addAction(dock.toggleViewAction())

        help_menu = menubar.addMenu("&Help")
        act = help_menu.addAction("&About")
        act.setToolTip("About MieWorkbench")
        act.triggered.connect(self._on_about)

    def _build_toolbar(self):
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("main_toolbar")
        style = self.style()

        open_tb = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
            "Open")
        open_tb.setToolTip("Open a model or archive")
        open_tb.triggered.connect(self._on_open)

        save_tb = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "Save")
        save_tb.setToolTip("Save the model / repack the workbench")
        save_tb.triggered.connect(self._on_save)

        run_tb = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Run")
        run_tb.setToolTip("Run Pipeline… (configure and start)")
        run_tb.triggered.connect(self._on_run_pipeline_dialog)

        stop_tb = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaStop), "Stop")
        stop_tb.setToolTip("Stop the running pipeline")
        stop_tb.triggered.connect(self.runner.stop)

        estimate_tb = toolbar.addAction(
            style.standardIcon(
                QStyle.StandardPixmap.SP_MessageBoxInformation),
            "Estimate")
        estimate_tb.setToolTip("Estimate runtime for the current "
                               "configuration")
        estimate_tb.triggered.connect(self._on_estimate)

    # -- pane wiring ---------------------------------------------------------------
    def _wire_panes(self):
        self.scene3d.set_project(self.project)
        self.element_editor.set_project(self.project)
        self.transform_panel.set_project(self.project)
        self.problems.set_project(self.project)
        self.problems.set_sources(
            lambda: (self.library_manager.project_lib
                     or self.library_manager.system_lib).load(),
            lambda: self.config_matrix.values())

        self.scene3d.selectionChanged.connect(self._on_scene_selection)
        self.inspector.faceSelectionChanged.connect(
            self.element_editor.set_face_selection)
        self.problems.selectBodyRequested.connect(
            lambda body: self._on_scene_selection(body, set()))
        self.problems.validationChanged.connect(
            self._on_validation_changed)

        self.library.addElementRequested.connect(self._on_add_element)
        self.library.openEditorRequested.connect(
            lambda cat, lib: self._open_prop_editor(cat, lib))

        self.project.sceneLoaded.connect(self._on_scene_loaded)
        self.project.dirtyChanged.connect(
            lambda dirty: self._update_window_title())

    def _on_scene_loaded(self):
        self.save_action.setEnabled(True)
        self.save_as_action.setEnabled(True)
        self._update_window_title()

    def _on_scene_selection(self, body_name, faces):
        if not body_name:
            return
        self.inspector.set_body(self.project, body_name)
        self.element_editor.set_face_selection(body_name, set(faces))
        self.transform_panel.set_body(body_name)

    def _on_validation_changed(self, has_errors):
        self._has_validation_errors = has_errors

    def _on_validate(self):
        self.problems_dock.raise_()
        self.problems.run_checks()

    # -- element addition (library + wizard) ------------------------------------------
    def _on_add_element(self, info, label):
        if not self.project.is_open():
            QMessageBox.information(
                self, "MieWorkbench",
                "Open or create a model first (File → Open…).")
            return
        matdb = None
        try:
            matdb = (self.library_manager.project_lib
                     or self.library_manager.system_lib).load().matdb
        except Exception:
            pass
        dialog = ElementWizardDialog(info, label, matdb=matdb, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        label = dialog.element_label()
        if not label:
            return
        try:
            self.project.import_primitive(info["path"], label)
            changed = dialog.changed_params()
            if changed:
                units = {a: s.get("unit", "")
                         for a, s in info.get("params", {}).items()}
                for alias, value in changed.items():
                    unit = units.get(alias, "")
                    raw = ("=%.10g %s" % (value, unit)).strip() \
                        if unit else "%.10g" % value
                    self.project.set_spreadsheet("dim_%s" % label,
                                                 alias, raw)
                self.project.rebuild_primitive(label)
        except Exception as exc:
            QMessageBox.warning(self, "Add element failed", str(exc))
            return
        self.statusBar().showMessage("Added %s" % label, 5000)

    def _open_prop_editor(self, category=None, which_library=None):
        if self._prop_editor_window is None:
            self._prop_editor_window = PropEditorPane(self.library_manager)
            self._prop_editor_window.setWindowTitle(
                "MieWorkbench — Property Library Editor")
            self._prop_editor_window.resize(1000, 620)
        self._prop_editor_window.show()
        self._prop_editor_window.raise_()

    # -- runner wiring -----------------------------------------------------------
    def _wire_runner(self):
        self.runner.line.connect(self.console.append_line)
        self.runner.progress.connect(self._on_progress)
        self.runner.started.connect(self._on_started)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_error)

    def _on_started(self):
        for chip in self.stage_chips.values():
            chip.setStyleSheet(self._chip_style(_CHIP_DEFAULT))
        self.progress_bar.setValue(0)
        self.console_dock.raise_()
        self.statusBar().showMessage("Pipeline started")

    def _on_progress(self, event):
        stage = event.get("stage")
        msg = event.get("msg", "")
        frac = event.get("frac")
        status = event.get("status", "running")
        self.statusBar().showMessage("[%s] %s" % (stage, msg))
        if stage == "pipeline" and frac is not None:
            self.progress_bar.setValue(int(round(frac * 100)))
        elif stage in self.stage_chips:
            color = _CHIP_COLORS.get(status, _CHIP_DEFAULT)
            self.stage_chips[stage].setStyleSheet(self._chip_style(color))

    def _on_finished(self, exit_code):
        if exit_code == 0:
            self.statusBar().showMessage("Pipeline finished", 5000)
            self._after_successful_run()
        else:
            self.statusBar().showMessage(
                "Pipeline exited with code %d (see console; exit 4 means "
                "the case is locked by a live run)" % exit_code, 8000)

    def _after_successful_run(self):
        case_dir = self._current_case_dir()
        if case_dir and os.path.isdir(case_dir):
            self.results.load_case(case_dir)
            self.results_dock.raise_()
        if self.workspace and self.miewb_path \
                and common.read_case_status(
                    os.path.join(case_dir or "", "case.json")) \
                == "completed":
            out = self.miesim_out or os.path.splitext(
                self.miewb_path)[0] + ".MieSim"
            try:
                # refresh the embedded workbench, then pack results
                miewb_tool.pack_miewb(
                    self.model_path, self.miewb_path,
                    optprops_dir=self._workspace_optprops(),
                    simparams=self.config_matrix.values())
                miewb_tool.pack_miesim(
                    self.workspace, out, self.miewb_path,
                    model_stem=os.path.splitext(
                        os.path.basename(self.model_path))[0],
                    case=os.path.basename(case_dir))
                self.statusBar().showMessage("Updated %s" % out, 8000)
            except Exception as exc:
                self.statusBar().showMessage(
                    "MieSim packing failed: %s" % exc, 8000)

    def _current_case_dir(self):
        if not self.model_path:
            return None
        stem = os.path.splitext(os.path.basename(self.model_path))[0]
        values = self.config_matrix.values()
        case = common.case_name(values.get("preset", "quick"),
                                values.get("tag"))
        if self.workspace:
            return os.path.join(self.workspace, "results", stem, case)
        return str(common.case_dir(stem, case))

    def _on_error(self, message):
        self.statusBar().showMessage("Pipeline error: %s" % message, 5000)

    # -- open flows -----------------------------------------------------------
    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open optical model", "",
            "Optical models (*.FCStd *.MieWB *.MieSim);;All files (*)")
        if path:
            self.open_model(path)

    def open_model(self, path):
        kind = miewb_tool.sniff(path)
        try:
            if kind == "FCStd":
                self._open_fcstd(path)
            elif kind == "MieWB":
                self._open_miewb(path)
            elif kind == "MieSim":
                self._open_miesim(path)
            else:
                QMessageBox.warning(
                    self, "MieWorkbench",
                    "%s is not a recognized .FCStd / .MieWB / .MieSim "
                    "file." % os.path.basename(path))
                return
        except (ProjectError, miewb_tool.MieFormatError, OSError) as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self.opened_path = path
        self._update_window_title()

    def _workspace_dir(self, path):
        digest = hashlib.sha1(
            os.path.abspath(path).encode("utf-8")).hexdigest()[:8]
        stem = os.path.splitext(os.path.basename(path))[0]
        ws = os.path.join(REPO, "var", "work", "%s-%s" % (stem, digest))
        os.makedirs(ws, exist_ok=True)
        return ws

    def _workspace_optprops(self):
        if self.workspace:
            d = os.path.join(self.workspace, "opticalproperties")
            if os.path.isdir(d):
                return d
        return os.path.join(REPO, "opticalproperties")

    def _open_fcstd(self, path):
        self.workspace = None
        self.miewb_path = None
        self.miesim_out = None
        self.project.open_fcstd(path)
        self.model_path = path
        self.library_manager.set_project_root(None)

    def _open_miewb(self, path, miesim_out=None):
        ws = self._workspace_dir(path)
        manifest = miewb_tool.unpack(path, ws)
        stem = manifest.get("model_stem") or "model"
        named = os.path.join(ws, "%s.FCStd" % stem)
        if not os.path.exists(named):
            import shutil
            shutil.copy2(os.path.join(ws, manifest.get("fcstd",
                                                       "model.FCStd")),
                         named)
        self.project.open_fcstd(named)
        self.model_path = named
        self.workspace = ws
        self.miewb_path = path
        self.miesim_out = miesim_out
        self.library_manager.set_project_root(ws)
        simparams_path = os.path.join(ws, "simparams.json")
        if os.path.exists(simparams_path):
            try:
                with open(simparams_path) as fh:
                    self.config_matrix.set_values(json.load(fh))
            except Exception:
                pass
        self.statusBar().showMessage(
            "Workbench opened in workspace %s" % ws, 8000)

    def _open_miesim(self, path):
        manifest = miewb_tool.read_manifest(path)
        ws = self._workspace_dir(path)
        miewb_tool.unpack(path, ws)
        case_dir = os.path.join(ws, "results", manifest["model"],
                                manifest["case"])
        live = (common.lock_info(case_dir) is not None
                and not common.lock_is_stale(case_dir))
        self.results.load_case(case_dir, monitor=live)
        self.results_dock.raise_()
        if live:
            self.statusBar().showMessage(
                "This case is RUNNING — opened read-only in monitor mode")
            return
        answer = QMessageBox.question(
            self, "Open results",
            "View results only, or open the embedded workbench for "
            "editing?\n\nA successful rerun will UPDATE this .MieSim.",
            QMessageBox.StandardButton.Open
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Open:
            wb = os.path.join(ws, "input.MieWB")
            self._open_miewb(wb, miesim_out=os.path.abspath(path))

    def _on_open_case(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open results case directory",
            str(common.RESULTS_DIR))
        if not path:
            return
        live = (common.lock_info(path) is not None
                and not common.lock_is_stale(path))
        self.results.load_case(path, monitor=live)
        self.results_dock.raise_()
        if live:
            self.statusBar().showMessage(
                "Case is RUNNING — monitor mode (read-only)")

    # -- save flows ----------------------------------------------------------------
    def _on_save(self):
        if not self.project.is_open():
            return
        try:
            self.project.save()
            if self.miewb_path:
                miewb_tool.pack_miewb(
                    self.model_path, self.miewb_path,
                    optprops_dir=self._workspace_optprops(),
                    simparams=self.config_matrix.values())
                self.statusBar().showMessage(
                    "Saved and repacked %s" % self.miewb_path, 5000)
            else:
                self.statusBar().showMessage(
                    "Saved %s" % self.model_path, 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _on_save_as(self):
        if not self.project.is_open():
            return
        path, selected = QFileDialog.getSaveFileName(
            self, "Save As", "",
            "Workbench archive (*.MieWB);;FreeCAD model (*.FCStd)")
        if not path:
            return
        try:
            if path.lower().endswith(".miewb") or "MieWB" in selected:
                self.project.save()
                miewb_tool.pack_miewb(
                    self.model_path, path,
                    optprops_dir=self._workspace_optprops(),
                    simparams=self.config_matrix.values())
            else:
                self.project.save_as(path)
                self.model_path = path
                self.workspace = None
                self.miewb_path = None
            self.statusBar().showMessage("Saved %s" % path, 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
        self._update_window_title()

    def _update_window_title(self):
        name = os.path.basename(self.opened_path or self.model_path or "")
        dirty = "*" if (self.project.is_open()
                        and self.project.is_dirty()) else ""
        if name:
            self.setWindowTitle("MieWorkbench — %s%s" % (name, dirty))
        else:
            self.setWindowTitle("MieWorkbench")

    # -- export ---------------------------------------------------------------------
    def _on_export_script(self):
        if not self._require_model():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Run Script", "", "Shell scripts (*.sh)")
        if not path:
            return
        wb_path = os.path.splitext(path)[0] + ".MieWB"
        sim_path = os.path.splitext(path)[0] + ".MieSim"
        try:
            if self.project.is_open():
                self.project.save()
            miewb_tool.pack_miewb(
                self.model_path, wb_path,
                optprops_dir=self._workspace_optprops(),
                simparams=self.config_matrix.values())
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        tool = os.path.join(REPO, "scripts", "miewb_tool.py")
        lines = [
            "#!/bin/sh",
            "# MieWorkbench headless run script — needs a repo clone; "
            "override tool",
            "# paths via MIEWB_FREECAD / MIEWB_OPTICS_PYTHON / "
            "MIEWB_PVPYTHON if they",
            "# live elsewhere on this machine.",
            "set -e",
            "python3 %s run %s -o %s" % (
                shlex.quote(tool), shlex.quote(os.path.abspath(wb_path)),
                shlex.quote(os.path.abspath(sim_path))),
        ]
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass
        self.statusBar().showMessage(
            "Wrote %s (+ %s)" % (path, os.path.basename(wb_path)), 8000)

    def _on_settings(self):
        dialog = SettingsDialog(self.settings, self)
        dialog.exec()

    def _on_about(self):
        QMessageBox.about(
            self, "About MieWorkbench",
            "MieWorkbench\nA PySide6 workbench for the FreeCAD-driven "
            "Mie optical ray tracer:\nscene editing, element wizards, "
            "property libraries, validation,\nsimulation orchestration "
            "and results analysis.")

    # -- Simulation actions -----------------------------------------------------
    def _require_model(self):
        if not self.model_path:
            QMessageBox.warning(
                self, "No model open",
                "Open a model (.FCStd or .MieWB) before running the "
                "pipeline.")
            return False
        return True

    def _preflight(self):
        """Save + validate before launching; errors block, warnings ask."""
        if not self._require_model():
            return None
        if self.project.is_open() and self.project.is_dirty():
            try:
                self.project.save()
            except Exception as exc:
                QMessageBox.critical(self, "Save failed", str(exc))
                return None
        findings = self.problems.run_checks()
        from .core import validation as _v
        n_err = sum(1 for f in findings if f.severity == _v.ERROR)
        n_warn = sum(1 for f in findings if f.severity == _v.WARNING)
        if n_err:
            self.problems_dock.raise_()
            QMessageBox.critical(
                self, "Validation failed",
                "%d error(s) must be fixed before running — see the "
                "Problems pane." % n_err)
            return None
        if n_warn:
            answer = QMessageBox.question(
                self, "Validation warnings",
                "%d warning(s) found (see Problems pane). Run anyway?"
                % n_warn)
            if answer != QMessageBox.StandardButton.Yes:
                return None
        args = self.config_matrix.to_args()
        if self.workspace:
            args += ["--optical-properties", self._workspace_optprops()]
        return args

    def _run_env(self):
        if not self.workspace:
            return None
        return {"MIEWB_GEOMETRY_DIR": os.path.join(self.workspace,
                                                   "geometry"),
                "MIEWB_RESULTS_DIR": os.path.join(self.workspace,
                                                  "results")}

    def _on_run_pipeline_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Run Pipeline")
        layout = QVBoxLayout(dialog)
        layout.addWidget(self.config_matrix)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Run")
        layout.addWidget(buttons)

        def on_run():
            args = self._preflight()
            if args is None:
                return
            if self.runner.start(self.model_path, args,
                                 extra_env=self._run_env()):
                dialog.accept()
            else:
                QMessageBox.warning(
                    self, "Pipeline already running",
                    "A pipeline run is already in progress.")

        buttons.accepted.connect(on_run)
        buttons.rejected.connect(dialog.reject)
        dialog.resize(720, 640)
        dialog.exec()

    def _on_estimate(self):
        self._show_estimate(self.config_matrix.estimate_params())

    def _show_estimate(self, params):
        result = common.estimate(
            params["rays"], params["resolution"], params["nlambda"],
            params["n_coherent_sources"], params["backend"],
            n_detectors=params["n_detectors"],
            save_fields=params["save_fields"],
            n_pol_strata=params["n_pol_strata"])
        message = (
            "Trace:  %s\n"
            "Gather: %s\n"
            "Total:  %s\n"
            "Accumulator memory: %.3f GB"
            % (common.fmt_duration(result["trace_s"]),
               common.fmt_duration(result["gather_s"]),
               common.fmt_duration(result["total_s"]),
               result["accumulator_GB"]))
        QMessageBox.information(self, "Runtime Estimate", message)

    def _on_dry_run(self):
        args = self._preflight()
        if args is None:
            return
        if "--dry-run" not in args:
            args = args + ["--dry-run"]
        if not self.runner.start(self.model_path, args,
                                 extra_env=self._run_env()):
            QMessageBox.warning(
                self, "Pipeline already running",
                "A pipeline run is already in progress.")

    # -- shutdown -------------------------------------------------------------------
    def closeEvent(self, event):
        # only prompt an interactive user; a hidden window (tests, teardown
        # during app shutdown) must never block on a modal dialog
        if self.isVisible() and self.project.is_open() \
                and self.project.is_dirty():
            answer = QMessageBox.question(
                self, "Unsaved changes",
                "The model has unsaved changes. Save before exiting?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Save:
                self._on_save()
        try:
            self.project.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
