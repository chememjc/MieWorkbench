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
import shutil
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "scripts")))
import common  # noqa: E402  (stdlib-only shared contract hub)
import miewb_tool  # noqa: E402  (stdlib-only archive engine)

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QDockWidget, QFileDialog, QHBoxLayout,
    QInputDialog, QLabel, QMainWindow, QMenu, QMessageBox, QProgressBar,
    QStyle, QToolButton, QVBoxLayout, QWidget,
)

from .core import paraview_launcher
from .core.librarymgr import LibraryManager
from .core.project import Project, ProjectError
from .core.raypreview import RayPreviewController
from .core.runner import RunController
from .core.selection import SelectionModel
from .core.settings import Settings, SettingsDialog
from .core.transforms import Operation, element_bounds
from .panes.config_matrix import ConfigMatrix
from .panes.console import ConsolePane
from .panes.element_editor import ElementEditorPane
from .panes.inspector3d import InspectorPane
from .panes.library import LibraryPane
from .panes.outliner import OutlinerPane
from .panes.problems import ProblemsPane
from .panes.prop_editor import PropEditorPane
from .panes.results import ResultsPane
from .panes.scene3d import Scene3DPane
from .panes.transform_panel import TransformPanel
from .panes.element_wizard import TypeChooserDialog
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


def _aabb_overlap(a, b):
    """Axis-aligned boxes ([lo3], [hi3]) intersect (touching counts)."""
    return all(a[0][k] <= b[1][k] and b[0][k] <= a[1][k] for k in range(3))


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
        self._clipboard_element = None  # element label copied for paste

        self.settings = Settings()
        self.project = Project(self.settings)
        self.selection = SelectionModel(self)
        self.raypreview = RayPreviewController(self)
        self._preview_target = "scene"   # or "inspector"
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
        self.outliner = OutlinerPane()
        self.outliner.setObjectName("outliner_host")
        self.outliner_dock = self._add_dock(
            "Scene Elements", "outliner_dock", self.outliner,
            Qt.DockWidgetArea.LeftDockWidgetArea)

        self.inspector = InspectorPane()
        self.inspector.setObjectName("inspector_host")
        self.inspector_dock = self._add_dock(
            "Element Inspector", "inspector_dock", self.inspector,
            Qt.DockWidgetArea.LeftDockWidgetArea)
        self.splitDockWidget(self.outliner_dock, self.inspector_dock,
                             Qt.Orientation.Vertical)

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
        self.new_action = file_menu.addAction("&New…")
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_action.setToolTip(
            "Create a new simulation from scratch "
            "(.MieWB workbench archive, or a bare .FCStd model)")
        self.new_action.triggered.connect(self._on_new)

        self.open_action = file_menu.addAction("&Open…")
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.setToolTip("Open an optical model or archive "
                                    "(.FCStd / .MieWB / .MieSim)")
        self.open_action.triggered.connect(self._on_open)

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

        edit_menu = menubar.addMenu("&Edit")
        self.undo_action = edit_menu.addAction("&Undo")
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self._on_undo)
        self.redo_action = edit_menu.addAction("&Redo")
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self._on_redo)
        stack = self.project.undo_stack
        stack.canUndoChanged.connect(self._on_can_undo_changed)
        stack.canRedoChanged.connect(self._on_can_redo_changed)
        stack.error.connect(self._on_undo_error)

        edit_menu.addSeparator()
        self.add_element_action = edit_menu.addAction("&Add Element…")
        self.add_element_action.setToolTip(
            "Add an element from the primitive library")
        self.add_element_action.triggered.connect(self._on_add_element_action)

        # deliberately NOT Ctrl+C/Ctrl+V/Del: window-level shortcuts would
        # steal those keys from every text field; the outliner handles Del
        # itself when it has focus
        self.copy_action = edit_menu.addAction("&Copy Element")
        self.copy_action.setShortcut("Ctrl+Shift+C")
        self.copy_action.setEnabled(False)
        self.copy_action.triggered.connect(lambda: self._on_copy_element())

        self.paste_action = edit_menu.addAction("&Paste Element")
        self.paste_action.setShortcut("Ctrl+Shift+V")
        self.paste_action.setEnabled(False)
        self.paste_action.triggered.connect(self._on_paste_element)

        self.delete_action = edit_menu.addAction("&Delete Element")
        self.delete_action.setEnabled(False)
        self.delete_action.triggered.connect(
            lambda: self._on_delete_element())

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
        for dock in (self.outliner_dock, self.inspector_dock,
                     self.element_editor_dock, self.transform_dock,
                     self.library_dock, self.console_dock,
                     self.results_dock, self.problems_dock):
            view_menu.addAction(dock.toggleViewAction())

        help_menu = menubar.addMenu("&Help")
        act = help_menu.addAction("&About")
        act.setToolTip("About MieWorkbench")
        act.triggered.connect(self._on_about)

    def _build_toolbar(self):
        """Grouped main toolbar: file | undo/redo | element ops |
        run/stop/estimate | validate | view."""
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("main_toolbar")
        style = self.style()

        def icon(pixmap):
            return style.standardIcon(pixmap)

        self.new_action.setIcon(icon(QStyle.StandardPixmap.SP_FileIcon))
        toolbar.addAction(self.new_action)
        self.open_action.setIcon(
            icon(QStyle.StandardPixmap.SP_DialogOpenButton))
        toolbar.addAction(self.open_action)
        self.save_action.setIcon(
            icon(QStyle.StandardPixmap.SP_DialogSaveButton))
        toolbar.addAction(self.save_action)

        toolbar.addSeparator()
        self.undo_action.setIcon(icon(QStyle.StandardPixmap.SP_ArrowBack))
        toolbar.addAction(self.undo_action)
        self.redo_action.setIcon(
            icon(QStyle.StandardPixmap.SP_ArrowForward))
        toolbar.addAction(self.redo_action)

        toolbar.addSeparator()
        self.add_element_action.setIcon(
            icon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        toolbar.addAction(self.add_element_action)
        self.copy_action.setIcon(
            icon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        toolbar.addAction(self.copy_action)
        self.paste_action.setIcon(
            icon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        toolbar.addAction(self.paste_action)
        self.delete_action.setIcon(
            icon(QStyle.StandardPixmap.SP_TrashIcon))
        toolbar.addAction(self.delete_action)

        toolbar.addSeparator()
        self.run_action.setIcon(icon(QStyle.StandardPixmap.SP_MediaPlay))
        toolbar.addAction(self.run_action)
        self.stop_action.setIcon(icon(QStyle.StandardPixmap.SP_MediaStop))
        toolbar.addAction(self.stop_action)
        self.estimate_action.setIcon(
            icon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        toolbar.addAction(self.estimate_action)

        toolbar.addSeparator()
        self.validate_action.setIcon(
            icon(QStyle.StandardPixmap.SP_DialogApplyButton))
        toolbar.addAction(self.validate_action)

        toolbar.addSeparator()
        rays_btn = QToolButton()
        rays_btn.setText("Rays")
        rays_btn.setToolTip("Ray display: toggle the overlay, reload the "
                            "last run's rays, or trace a live preview fan")
        rays_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        rays_menu = QMenu(rays_btn)
        act = rays_menu.addAction("Show/hide overlay")
        act.triggered.connect(self.scene3d.rays_button.toggle)
        act = rays_menu.addAction("Load last run's rays")
        act.triggered.connect(self._load_case_rays)
        act = rays_menu.addAction("Live ray preview…")
        act.setToolTip("Trace a small deterministic fan from each source "
                       "(center + top/bottom/left/right of the emit face) "
                       "through the current scene")
        act.triggered.connect(self._on_ray_preview)
        rays_btn.setMenu(rays_menu)
        toolbar.addWidget(rays_btn)

        fit_tb = toolbar.addAction(
            icon(QStyle.StandardPixmap.SP_BrowserReload), "Fit view")
        fit_tb.setToolTip("Reset the 3D camera to frame the whole scene")
        fit_tb.triggered.connect(self.scene3d.view.fit_camera)

    # -- ray displays ---------------------------------------------------------------
    def _case_rays_vtp(self):
        case = getattr(self.results, "case_dir", None)
        if not case:
            return None
        for f in paraview_launcher.viz_files(case):
            if f.endswith("rays.vtp"):
                return f
        return None

    def _load_case_rays(self):
        path = self._case_rays_vtp()
        if path is None:
            self.statusBar().showMessage(
                "No traced rays available — run the pipeline (or use "
                "Live ray preview…)", 6000)
            return
        self.scene3d.load_rays_vtp(path)
        self.scene3d.set_rays_stale(False)
        self.statusBar().showMessage("Loaded ray overlay: %s" % path, 5000)

    def _load_case_rays_quiet(self):
        path = self._case_rays_vtp()
        if path is not None:
            self.scene3d.load_rays_vtp(path)
            self.scene3d.set_rays_stale(False)

    def _preview_workspace(self):
        if self.workspace:
            return self.workspace
        digest = hashlib.sha1(
            (self.model_path or "unsaved").encode("utf-8")).hexdigest()[:8]
        ws = os.path.join(REPO, "var", "work", "preview-%s" % digest)
        os.makedirs(ws, exist_ok=True)
        return ws

    def _on_ray_preview(self, only_bodies=None, target="scene"):
        if not self.project.is_open():
            self.statusBar().showMessage("Open or create a model first",
                                         5000)
            return
        if self.raypreview.is_running():
            self.statusBar().showMessage("Ray preview already running…",
                                         4000)
            return
        n, ok = QInputDialog.getInt(
            self, "Live ray preview",
            "Rays per source (center + edge midpoints, then rim fill):",
            5, 1, 999)
        if not ok:
            return
        self._preview_target = target
        started = self.raypreview.start(
            self.project, self._preview_workspace(),
            pattern="fan:n=%d" % n, only_bodies=only_bodies,
            optical_properties=self._workspace_optprops())
        if started:
            self.statusBar().showMessage(
                "Tracing %d preview ray(s) per source…" % n)

    def _on_scene_rays_requested(self):
        """Rays toggle checked with no overlay: load the last run's rays
        if a case is loaded, else offer the live preview."""
        path = self._case_rays_vtp()
        if path is not None:
            self.scene3d.load_rays_vtp(path)
            self.scene3d.set_rays_stale(False)
            self.statusBar().showMessage("Loaded ray overlay from the "
                                         "last run", 5000)
            return
        self._on_ray_preview()

    def _on_inspector_rays_requested(self):
        body = self.inspector._body_name
        if body is None:
            return
        try:
            only = self.project.element_bodies(
                self.project.element_group(body))
        except ProjectError:
            only = [body]
        self._on_ray_preview(only_bodies=only, target="inspector")

    def _on_preview_finished(self, vtp_path):
        if self._preview_target == "inspector":
            self.inspector.load_rays_vtp(vtp_path)
        else:
            self.scene3d.load_rays_vtp(vtp_path)
            self.scene3d.set_rays_stale(False)
        self.statusBar().showMessage("Ray preview ready", 5000)

    def _on_preview_failed(self, message):
        self.console.append_line("[preview] " + message)
        self.console_dock.raise_()
        self.statusBar().showMessage("Ray preview failed — see Console",
                                     8000)

    def _on_geometry_changed(self, *_args):
        self.scene3d.set_rays_stale(True)

    # -- undo/redo -----------------------------------------------------------------
    def _on_undo(self):
        try:
            self.project.undo()
        except Exception as exc:
            QMessageBox.warning(self, "Undo failed", str(exc))

    def _on_redo(self):
        try:
            self.project.redo()
        except Exception as exc:
            QMessageBox.warning(self, "Redo failed", str(exc))

    def _on_can_undo_changed(self, enabled, text):
        self.undo_action.setEnabled(enabled)
        self.undo_action.setText("&Undo %s" % text if text else "&Undo")

    def _on_can_redo_changed(self, enabled, text):
        self.redo_action.setEnabled(enabled)
        self.redo_action.setText("&Redo %s" % text if text else "&Redo")

    def _on_undo_error(self, message):
        # mid-stack failure: the stack was cleared to avoid divergence
        self.statusBar().showMessage("Undo history reset: %s" % message,
                                     10000)
        if self.isVisible():   # modal dialogs hang offscreen teardown
            QMessageBox.warning(self, "Undo history reset", message)

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

        # all selection flows through the shared SelectionModel: 3D picks,
        # outliner rows and problems-pane jumps stay in sync
        self.outliner.set_project(self.project)
        self.scene3d.selectionChanged.connect(
            lambda body, faces: self.selection.select(body, faces,
                                                      origin="scene3d"))
        self.outliner.selectBodyRequested.connect(
            lambda body: self.selection.select(body, (), origin="outliner"))
        self.problems.selectBodyRequested.connect(
            lambda body: self.selection.select(body, ()))
        self.selection.changed.connect(self._on_selection_changed)

        self.outliner.customizeRequested.connect(self._on_customize_element)
        self.outliner.deleteRequested.connect(self._on_delete_element)
        self.outliner.copyRequested.connect(self._on_copy_element)
        self.outliner.pasteRequested.connect(self._on_paste_element)

        self.inspector.faceSelectionChanged.connect(
            self.element_editor.set_face_selection)
        # the editor's assignment LIST is an alternative face-picking
        # surface: rows chosen there highlight in the inspector's 3D view
        self.element_editor.facesPicked.connect(
            lambda _body, faces: self.inspector.set_selected_faces(faces))
        # every value dropdown reflects the ACTIVE property library (the
        # project's embedded one when a .MieWB is open, else the system's)
        self.element_editor.set_prop_library(
            lambda: (self.library_manager.project_lib
                     or self.library_manager.system_lib))
        # right-click in the inspector's 3D view pops the same Active
        # Properties menu as the editor's assignment table
        self.inspector.contextMenuRequested.connect(
            self._on_inspector_context_menu)
        self.problems.validationChanged.connect(
            self._on_validation_changed)

        self.library.addElementRequested.connect(self._on_add_element)
        self.library.openEditorRequested.connect(
            lambda cat, lib: self._open_prop_editor(cat, lib))

        self.project.sceneLoaded.connect(self._on_scene_loaded)
        self.project.dirtyChanged.connect(
            lambda dirty: self._update_window_title())

        # ray displays: previews land in the requesting view; geometry
        # edits mark any loaded overlay stale
        self.raypreview.finished.connect(self._on_preview_finished)
        self.raypreview.failed.connect(self._on_preview_failed)
        self.inspector.raysPreviewRequested.connect(
            self._on_inspector_rays_requested)
        self.scene3d.raysPreviewRequested.connect(
            self._on_scene_rays_requested)
        self.project.bodiesReshaped.connect(self._on_geometry_changed)
        self.project.bodiesMoved.connect(self._on_geometry_changed)

    def _on_scene_loaded(self):
        self.save_action.setEnabled(True)
        self.save_as_action.setEnabled(True)
        self._update_window_title()

    def _on_selection_changed(self, body_name, faces, origin):
        enable = bool(body_name)
        self.copy_action.setEnabled(enable)
        self.delete_action.setEnabled(enable)
        if not body_name:
            return
        self.inspector.set_body(self.project, body_name)
        self.element_editor.set_face_selection(body_name, set(faces))
        self.transform_panel.set_body(body_name)
        if origin != "outliner":
            self.outliner.set_selected_body(body_name)
        if origin != "scene3d" and hasattr(self.scene3d, "select_body"):
            self.scene3d.select_body(body_name)

    # kept for tests/back-compat: route an explicit (body, faces) pair
    # through the shared selection model
    def _on_scene_selection(self, body_name, faces):
        self.selection.select(body_name, faces)

    def _on_inspector_context_menu(self, global_pos):
        menu = self.element_editor.build_active_properties_menu(self)
        if menu is not None:
            menu.exec(global_pos)

    def _on_validation_changed(self, has_errors):
        self._has_validation_errors = has_errors

    def _on_validate(self):
        self.problems_dock.raise_()
        self.problems.run_checks()

    # -- element addition (library + wizard) ------------------------------------------
    def _load_matdb(self):
        try:
            return (self.library_manager.project_lib
                    or self.library_manager.system_lib).load().matdb
        except Exception:
            return None

    def _registry_names(self, category):
        lib = (self.library_manager.project_lib
               or self.library_manager.system_lib)
        return [row.get("name", "") for row in lib.registry_rows(category)
                if row.get("name")]

    def _apply_wizard_output(self, dialog, info, label):
        """Write the wizard's changed parameters + device properties for
        an (already imported) element. Runs inside the caller's macro."""
        changed = dialog.changed_params()
        if changed:
            units = {a: s.get("unit", "")
                     for a, s in info.get("params", {}).items()}
            values = {}
            for alias, value in changed.items():
                unit = units.get(alias, "")
                values[alias] = ("=%.10g %s" % (value, unit)).strip() \
                    if unit else "%.10g" % value
            self.project.set_element_parameters(
                "dim_%s" % label, values, rebuild_group=label)
        props = dialog.changed_props()
        if props:
            names = self.project.element_bodies(label)
            for name, value in props.items():
                # apply to the member body that carries the property
                # (single-body elements: just that body); fall back to
                # the first member for newly introduced props
                target = names[0]
                for member in names:
                    if name in (self.project.body(member)
                                .get("properties") or {}):
                        target = member
                        break
                self.project.set_property(target, name, value)

    def _on_add_element(self, info, label):
        if not self.project.is_open():
            QMessageBox.information(
                self, "MieWorkbench",
                "Open or create a model first (File → Open…).")
            return
        dialog = ElementWizardDialog(
            info, label, matdb=self._load_matdb(),
            registry_names=self._registry_names, parent=self,
            show_preview=True)
        previewed = [False]
        last_applied = [None]

        def snapshot():
            return (dialog.changed_params(), dialog.changed_props())

        def do_preview():
            new_label = dialog.element_label()
            if not new_label:
                return
            try:
                if not previewed[0]:
                    self.project.begin_macro("Add %s" % new_label)
                    self.project.import_primitive(info["path"], new_label)
                    previewed[0] = True
                    dialog.label_edit.setEnabled(False)
                elif snapshot() == last_applied[0]:
                    return
                self._apply_wizard_output(dialog, info, new_label)
                last_applied[0] = snapshot()
                self.statusBar().showMessage(
                    "Previewing %s — Cancel removes it" % new_label, 4000)
            except Exception as exc:
                QMessageBox.warning(dialog, "Preview failed", str(exc))
        dialog.previewRequested.connect(do_preview)

        accepted = dialog.exec() == QDialog.Accepted
        label = dialog.element_label()
        if not accepted:
            if previewed[0]:
                self.project.abort_macro()   # rolls the preview back
            return
        if not label:
            return
        try:
            if not previewed[0]:
                self.project.begin_macro("Add %s" % label)
                self.project.import_primitive(info["path"], label)
            if snapshot() != last_applied[0]:
                self._apply_wizard_output(dialog, info, label)
        except Exception as exc:
            self.project.abort_macro()
            QMessageBox.warning(self, "Add element failed", str(exc))
            return
        self.project.end_macro()
        self.statusBar().showMessage("Added %s" % label, 5000)

    def _on_add_element_action(self):
        """Toolbar/menu 'Add element': start the add flow for the library's
        current primitive, else the type-first wizard (what → which →
        configure)."""
        if hasattr(self.library, "start_add_current"):
            if self.library.start_add_current():
                return
        if not self.project.is_open():
            QMessageBox.information(
                self, "MieWorkbench",
                "Open or create a model first (File → New… / Open…).")
            return
        chooser = TypeChooserDialog(
            self.library_manager.primitives_list(), parent=self)
        if chooser.exec() != QDialog.Accepted:
            return
        info = chooser.chosen_info()
        if info is None:
            return
        from .panes.library import default_label
        used = {b["label"] for b in self.project.structure.get("bodies", [])}
        self._on_add_element(info, default_label(info["kind"], used))

    def _on_customize_element(self, body_name):
        """Outliner double-click: reopen the wizard on a primitive-built
        element (prefilled), else focus the property editors."""
        self.selection.select(body_name, ())
        try:
            body = self.project.body(body_name)
        except ProjectError:
            return
        props = body.get("properties", {}) or {}
        kind = props.get("miewb_primitive", {}).get("value")
        group = props.get("miewb_group", {}).get("value")
        info = None
        if kind:
            for candidate in self.library_manager.primitives_list():
                if candidate.get("kind") == kind:
                    info = candidate
                    break
        if info is None or not group:
            self.element_editor_dock.raise_()   # hand-authored: editor
            return
        sheet = self.project.sheet_for_body(body_name)
        sheet_values = {}
        if sheet is not None:
            from .panes.element_editor import parse_sheet_raw
            for alias, entry in (sheet.get("aliases") or {}).items():
                try:
                    sheet_values[alias] = parse_sheet_raw(
                        entry.get("raw", ""))["number"]
                except ValueError:
                    pass
        prop_values = {}
        for member in self.project.element_bodies(group):
            for name, entry in (self.project.body(member)
                                .get("properties") or {}).items():
                if not name.startswith("miewb_") \
                        and name not in prop_values:
                    prop_values[name] = entry.get("value")
        dialog = ElementWizardDialog.for_element(
            info, group, sheet_values=sheet_values,
            prop_values=prop_values, matdb=self._load_matdb(),
            registry_names=self._registry_names, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        self.project.begin_macro("Customize %s" % group)
        try:
            # diff against the CURRENT values, not the primitive defaults
            units = {a: s.get("unit", "")
                     for a, s in info.get("params", {}).items()}
            values = {}
            for alias, value in dialog.params().items():
                if alias in sheet_values and \
                        abs(value - sheet_values[alias]) <= 1e-12:
                    continue
                unit = units.get(alias, "")
                values[alias] = ("=%.10g %s" % (value, unit)).strip() \
                    if unit else "%.10g" % value
            if values:
                self.project.set_element_parameters(
                    "dim_%s" % group, values, rebuild_group=group,
                    text="Customize %s" % group)
            for name, value in dialog.props().items():
                if name in prop_values and (
                        prop_values[name] == value
                        or str(prop_values[name]) == str(value)):
                    continue
                names = self.project.element_bodies(group)
                target = names[0]
                for member in names:
                    if name in (self.project.body(member)
                                .get("properties") or {}):
                        target = member
                        break
                self.project.set_property(target, name, value)
        except Exception as exc:
            self.project.abort_macro()
            QMessageBox.warning(self, "Customize failed", str(exc))
            return
        self.project.end_macro()
        self.statusBar().showMessage("Updated %s" % group, 5000)

    def _selected_element(self):
        if self.selection.body is None:
            return None
        try:
            return self.project.element_group(self.selection.body)
        except ProjectError:
            return None

    def _on_copy_element(self, element=None):
        element = element or self._selected_element()
        if not element:
            return
        self._clipboard_element = element
        self.paste_action.setEnabled(True)
        self.statusBar().showMessage(
            "Copied %s — paste with Ctrl+Shift+V" % element, 5000)

    def _on_delete_element(self, element=None):
        element = element or self._selected_element()
        if not element or not self.project.is_open():
            return
        try:
            self.project.delete_element(element)
        except Exception as exc:
            QMessageBox.warning(self, "Delete failed", str(exc))
            return
        self.selection.clear()
        self.statusBar().showMessage(
            "Deleted %s (Ctrl+Z to undo)" % element, 5000)

    def _unique_element_label(self, base):
        used = set()
        for b in self.project.structure.get("bodies", []):
            used.add(b["label"])
            group = b["properties"].get("miewb_group", {}).get("value")
            if group:
                used.add(group)
        n = 1
        while True:
            candidate = "%s_copy%s" % (base, "" if n == 1 else n)
            if candidate not in used:
                return candidate
            n += 1

    def _on_paste_element(self):
        src = self._clipboard_element
        if not src or not self.project.is_open():
            return
        try:
            src_names = self.project.element_bodies(src)
        except ProjectError:
            self.statusBar().showMessage(
                "Copied element %s no longer exists" % src, 5000)
            return
        new_label = self._unique_element_label(src)
        bodies = self.project.structure.get("bodies", [])
        bounds = element_bounds(bodies, self.project.body_states, src_names)
        # offset the copy +x past the source (and past anything else
        # occupying that spot) so it never lands invisibly coincident
        if bounds is not None:
            lo, hi = bounds
            step = max(1.2 * (hi[0] - lo[0]), 5.0)
        else:
            step = 10.0
        others = [element_bounds(bodies, self.project.body_states,
                                 [b["name"]])
                  for b in bodies if b["name"] not in src_names]
        others = [o for o in others if o is not None]
        shift = step
        for _ in range(20):
            if bounds is None:
                break
            lo, hi = bounds
            cand = ([lo[0] + shift, lo[1], lo[2]],
                    [hi[0] + shift, hi[1], hi[2]])
            if not any(_aabb_overlap(cand, o) for o in others):
                break
            shift += step
        self.project.begin_macro("Paste %s" % new_label)
        try:
            self.project.duplicate_element(src, new_label)
            for name in self.project.element_bodies(new_label):
                self.project.apply_operation(
                    name, Operation("translate",
                                    {"vector_mm": [shift, 0.0, 0.0]}))
        except Exception as exc:
            self.project.abort_macro()
            QMessageBox.warning(self, "Paste failed", str(exc))
            return
        self.project.end_macro()
        names = self.project.element_bodies(new_label)
        if names:
            self.selection.select(names[0], ())
        self.statusBar().showMessage("Pasted %s" % new_label, 5000)

    def _open_prop_editor(self, category=None, which_library=None):
        if self._prop_editor_window is None:
            self._prop_editor_window = PropEditorPane(self.library_manager)
            self._prop_editor_window.setWindowTitle(
                "MieWorkbench — Property Library Editor")
            self._prop_editor_window.resize(1000, 620)
        if category:
            self._prop_editor_window.show_category(
                category, which_library or "system")
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
            self._load_case_rays_quiet()
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

    def _on_new(self):
        path, selected = QFileDialog.getSaveFileName(
            self, "New Simulation", "",
            "Workbench archive (*.MieWB);;FreeCAD model (*.FCStd)")
        if not path:
            return
        if not path.lower().endswith((".miewb", ".fcstd")):
            path += ".FCStd" if "FCStd" in selected else ".MieWB"
        try:
            if path.lower().endswith(".miewb"):
                self._new_miewb(path)
            else:
                self._new_fcstd(path)
        except (ProjectError, OSError) as exc:
            QMessageBox.critical(self, "New simulation failed", str(exc))
            return
        self.opened_path = path
        self._update_window_title()
        self.library_dock.raise_()
        self.statusBar().showMessage(
            "New simulation created — add elements from the Library "
            "(Edit → Add Element…)", 8000)

    def _new_fcstd(self, path):
        self.workspace = None
        self.miewb_path = None
        self.miesim_out = None
        self.project.new_document(path)
        self.project.stash_root = None
        self.model_path = path
        self.library_manager.set_project_root(None)

    def _new_miewb(self, path):
        """A fresh workbench: workspace + empty model + the system optical
        property library as the project library, packed immediately so
        the .MieWB exists on disk from the start."""
        ws = self._workspace_dir(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        model = os.path.join(ws, "%s.FCStd" % stem)
        self.project.new_document(model)
        self.project.stash_root = os.path.join(ws, "undo")
        self.model_path = model
        self.workspace = ws
        self.miewb_path = path
        self.miesim_out = None
        optprops = os.path.join(ws, "opticalproperties")
        if not os.path.isdir(optprops):
            shutil.copytree(os.path.join(REPO, "opticalproperties"),
                            optprops)
        self.library_manager.set_project_root(ws)
        miewb_tool.pack_miewb(model, path, optprops_dir=optprops,
                              simparams=self.config_matrix.values())

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
        self.project.stash_root = None
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
        self.project.stash_root = os.path.join(ws, "undo")
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
