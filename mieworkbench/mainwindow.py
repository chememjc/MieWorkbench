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
import tempfile

import numpy as np

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "scripts")))
import common  # noqa: E402  (stdlib-only shared contract hub)
import miewb_tool  # noqa: E402  (stdlib-only archive engine)

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QActionGroup, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMenu,
    QMessageBox, QProgressBar, QSpinBox, QStyle, QTabWidget, QToolButton,
    QVBoxLayout, QWidget,
)

from .core import checkpointinfo
from .core import paraview_launcher
from .core import variables
from .core.beadanim import (AnimationController, format_sim_time,
                            precompute_segments)
from .core.librarymgr import LibraryManager
from .core.optimize_controller import OptimizeController
from .core.tolerance_controller import ToleranceController
from .core.previewscheduler import PreviewScheduler
from .core.project import Project, ProjectError
from .core.raypreview import RayPreviewController
from .core.runner import RunController
from .core.selection import SelectionModel
from .core.settings import Settings, SettingsDialog
from .core.transforms import Operation, element_bounds
from .core.train import EXCLUDE_PROP
from .panes.compare_pane import ComparePane
from .panes.config_matrix import ConfigMatrix
from .panes.console import ConsolePane
from .panes.element_editor import ElementEditorPane
from .panes.inspector3d import InspectorPane
from .panes.library import LibraryPane
from .panes.optimize_pane import OptimizePane
from .panes.tolerance_pane import TolerancePane
from .panes.outliner import OutlinerPane
from .panes.problems import ProblemsPane
from .panes.prop_editor import PropEditorPane
from .panes.py_console import PyConsolePane
from .panes.results import ResultsPane
from .panes.rundialog import RunDialog
from .panes.scene3d import Scene3DPane
from .panes.train_editor import TrainEditorPane
from .panes.transform_panel import TransformPanel
from .panes.element_wizard import TypeChooserDialog
from .panes.wizard_dialog import (ElementWizardDialog, FieldFanDialog,
                                  ZoomPairDialog)
from .widgets.preview_config import PreviewConfigWidget
from .widgets.style import checked_toolbutton_stylesheet

try:
    # a parallel round authors this pane; the optical-train wiring degrades
    # gracefully (no Variables dock, config-matrix sweep fields still work)
    # until it lands.
    from .panes.variables_pane import VariablesPane
except Exception:   # pragma: no cover - only while the pane is unwritten
    VariablesPane = None

import train_solver  # noqa: E402  (stdlib-only; shared chain math)

TrainError = train_solver.TrainError

STAGE_ORDER = ["extract", "trace", "post", "viz", "optimize",
               "tolerance"]

_CHIP_COLORS = {
    "running": "#3b82f6",
    "completed": "#22c55e",
    "estimated": "#22c55e",
    "failed": "#ef4444",
}
_CHIP_DEFAULT = "#6b7280"

# ray extinction modes, in the toolbar combo's index order
_RAY_DIM_MODES = ("off", "linear", "sqrt")

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# Help menu: one action per docs/guide page, grouped into submenus that
# mirror docs/guide/README.md's index tables. Paths are relative to
# docs/guide/ (resolved against REPO at menu-build time); keep this list
# and that index in sync by hand -- there is no code generator for either.
DOCS_GUIDE_DIR = os.path.join(REPO, "docs", "guide")
DOCS_GUIDE_PAGES = {
    "GUI": [
        ("3D Viewport", "viewport-3d.md"),
        ("Outliner", "outliner.md"),
        ("Element Inspector", "inspector.md"),
        ("Element Editor", "element-editor.md"),
        ("Position / Orientation", "transform.md"),
        ("Optical Train Editor", "train-editor.md"),
        ("Variables", "variables.md"),
        ("Compare", "compare.md"),
        ("Optimize", "optimize.md"),
        ("Tolerance", "tolerance.md"),
        ("Results", "results.md"),
        ("Library Browser", "library-browser.md"),
        ("Property Library Editor", "property-library-editor.md"),
        ("Run & Validate", "run-and-validate.md"),
        ("Animation", "animation.md"),
        ("Console and Problems", "console-and-problems.md"),
    ],
    "System": [
        ("Pipeline CLI", "pipeline-cli.md"),
        ("File Formats", "file-formats.md"),
        ("Headless / Remote", "headless-remote.md"),
        ("Authoring", "authoring.md"),
        ("Demo Gallery", "demo-gallery.md"),
    ],
    "Walkthroughs": [
        ("Walkthroughs (index)", "walkthroughs/README.md"),
        ("camera_triplet", "walkthroughs/camera-triplet.md"),
        ("schmidt_cassegrain", "walkthroughs/schmidt-cassegrain.md"),
        ("double_gauss", "walkthroughs/double-gauss.md"),
        ("fiber_coupling_doublet", "walkthroughs/fiber-coupling-doublet.md"),
    ],
}


def _aabb_overlap(a, b):
    """Axis-aligned boxes ([lo3], [hi3]) intersect (touching counts)."""
    return all(a[0][k] <= b[1][k] and b[0][k] <= a[1][k] for k in range(3))


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MieWorkbench")
        self.resize(1500, 950)
        # window-level chrome only -- widget-level stylesheets (stage
        # chips, etc.) are more specific and always win over this
        self.setStyleSheet(
            checked_toolbutton_stylesheet(self.palette()))

        self.model_path = None          # the .FCStd the pipeline runs on
        self.opened_path = None         # what the user opened (any format)
        self.workspace = None           # exploded .MieWB/.MieSim dir
        self.miewb_path = None          # archive to repack on Save
        self.miesim_out = None          # .MieSim to update after a rerun
        self._has_validation_errors = False
        self._clipboard_element = None  # element label copied for paste
        self._pending_manifest = None   # sweep manifest awaiting a Compare
        self._train_refresh_pending = False   # 0-ms coalescing guard
        # P1 chunked-run contract: the case_dir a Resume/Extend launch
        # targets, so _after_successful_run() reloads THAT case even when
        # it doesn't match the current ConfigMatrix preset/tag (see
        # _on_resume_run/_on_extend_run).
        self._resume_extend_case_dir = None

        self.settings = Settings()
        # RunDialog "don't ask again this session": stored via the same
        # QSettings-backed Settings wrapper every other toggle uses, but
        # explicitly reset here on every launch -- the owner requirement
        # is "always ask per run", so a skip must never survive a
        # restart; only checking the box DURING this session suppresses
        # the dialog for its remaining runs.
        self.settings.set_bool("run_dialog_skip_session", False)
        self.project = Project(self.settings)
        self.selection = SelectionModel(self)
        self.raypreview = RayPreviewController(self)
        self._preview_target = "scene"   # or "inspector"
        self.preview_scheduler = PreviewScheduler(parent=self)
        self.runner = RunController(self.settings, self)
        self.optimizer_ctl = OptimizeController(self.settings, self)
        self.tolerance_ctl = ToleranceController(self.settings, self)
        self.config_matrix = ConfigMatrix()
        self.config_matrix.estimateRequested.connect(self._show_estimate)
        # persistent Ray Preview tab of the Simulation Settings dialog
        # (WP2) -- one instance, reparented into the dialog on demand,
        # same pattern as config_matrix above.
        self.preview_config = PreviewConfigWidget()
        self.library_manager = LibraryManager(
            os.path.join(REPO, "opticalproperties"),
            os.path.join(REPO, "primitives"))
        self._prop_editor_window = None

        self._build_central()
        self.stage_chips = {}
        self._build_docks()
        self._init_animation()
        self._build_menus()
        self._build_toolbar()
        self._build_animation_toolbar()
        self._rebuild_folds_menu()      # initial (disabled/no-fold) state
        self.statusBar().showMessage("Ready")
        self._wire_runner()
        self._wire_panes()
        self._update_window_title()

    # -- central --------------------------------------------------------------
    def _build_central(self):
        """The central graphics area is a bottom-tabbed QTabWidget: the
        VTK 3D view plus the three big analysis surfaces (Optimize,
        Tolerance, Results) that used to crowd the bottom dock bar."""
        self.scene3d = Scene3DPane()
        self.scene3d.setObjectName("scene3d_host")

        self.optimize_pane = OptimizePane()
        self.optimize_pane.setObjectName("optimize_host")

        self.tolerance_pane = TolerancePane()
        self.tolerance_pane.setObjectName("tolerance_host")

        self.results = ResultsPane(self.settings)
        self.results.setObjectName("results_host")

        self.central_tabs = QTabWidget()
        self.central_tabs.setObjectName("central_tabs")
        self.central_tabs.setTabPosition(QTabWidget.TabPosition.South)
        self.central_tabs.setDocumentMode(True)
        self.central_tabs.addTab(self.scene3d, "3D View")
        self.central_tabs.addTab(self.optimize_pane, "Optimize")
        self.central_tabs.addTab(self.tolerance_pane, "Tolerance")
        self.central_tabs.addTab(self.results, "Results")
        self.setCentralWidget(self.central_tabs)

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

        self.py_console = PyConsolePane()
        self.py_console.set_context(project=self.project, window=self,
                                    runner=self.runner, np=np)
        self.py_console_dock = self._add_dock(
            "Python", "py_console_dock", self.py_console,
            Qt.DockWidgetArea.BottomDockWidgetArea)

        self.problems = ProblemsPane()
        self.problems.setObjectName("problems_host")
        self.problems_dock = self._add_dock(
            "Problems", "problems_dock", self.problems,
            Qt.DockWidgetArea.BottomDockWidgetArea)

        self.tabifyDockWidget(self.console_dock, self.py_console_dock)
        self.tabifyDockWidget(self.console_dock, self.problems_dock)
        self.console_dock.raise_()
        self.resizeDocks([self.console_dock], [230],
                         Qt.Orientation.Vertical)

        self._build_train_docks()

    def _build_train_docks(self):
        """Optical-train feature docks: an LDE-style train editor tabbed
        with the outliner, the sweep-Variables pane tabbed near the
        library, and the sweep-Compare pane tabbed at the bottom (hidden
        until a sweep completes or the user adds cases)."""
        self.train_editor = TrainEditorPane(self.project, self.selection)
        self.train_editor.setObjectName("train_editor_host")
        self.train_editor_dock = self._add_dock(
            "Optical Train", "train_editor_dock", self.train_editor,
            Qt.DockWidgetArea.LeftDockWidgetArea)
        self.tabifyDockWidget(self.outliner_dock, self.train_editor_dock)
        self.outliner_dock.raise_()

        if VariablesPane is not None:
            self.variables_pane = VariablesPane(self.project)
            self.variables_pane.setObjectName("variables_host")
            self.variables_dock = self._add_dock(
                "Variables", "variables_dock", self.variables_pane,
                Qt.DockWidgetArea.RightDockWidgetArea)
            self.tabifyDockWidget(self.library_dock, self.variables_dock)
        else:
            self.variables_pane = None
            self.variables_dock = None

        self.compare_pane = ComparePane(settings=self.settings)
        self.compare_pane.setObjectName("compare_host")
        self.compare_dock = self._add_dock(
            "Compare", "compare_dock", self.compare_pane,
            Qt.DockWidgetArea.BottomDockWidgetArea)
        self.tabifyDockWidget(self.console_dock, self.compare_dock)
        self.compare_dock.hide()        # revealed on first sweep/compare
        self.console_dock.raise_()

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

        self.revert_action = file_menu.addAction("&Revert to Saved")
        self.revert_action.setToolTip(
            "Discard all unsaved changes and restore the model to its "
            "last saved state on disk")
        self.revert_action.triggered.connect(self._on_revert)
        self.revert_action.setEnabled(False)

        self.close_action = file_menu.addAction("&Close")
        self.close_action.setShortcut(QKeySequence.StandardKey.Close)
        self.close_action.setToolTip(
            "Close the current model (prompts if there are unsaved "
            "changes)")
        self.close_action.triggered.connect(self._on_close_model)
        self.close_action.setEnabled(False)

        self.export_fcstd_action = file_menu.addAction("Export &FCStd…")
        self.export_fcstd_action.setToolTip(
            "Write a standalone .FCStd copy of the current document "
            "(fold states, placements and train metadata as they stand); "
            "the live document is untouched")
        self.export_fcstd_action.triggered.connect(self._on_export_fcstd)
        self.export_fcstd_action.setEnabled(False)

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

        edit_menu.addSeparator()
        # Esc is otherwise unbound at the window level (the 3D view's own Esc
        # cancels an in-progress axis drag via a VTK observer, not a QAction)
        self.clear_selection_action = edit_menu.addAction("Clear &Selection")
        self.clear_selection_action.setShortcut("Esc")
        self.clear_selection_action.setToolTip("Clear the current selection")
        self.clear_selection_action.setEnabled(False)
        self.clear_selection_action.triggered.connect(
            lambda: self.selection.clear(origin="clear_action"))

        sim_menu = menubar.addMenu("&Simulation")
        self.settings_action = sim_menu.addAction("Simulation &Settings…")
        self.settings_action.setToolTip(
            "View/edit the simulation settings (rays, resolution, engine, "
            "…) without running. OK saves them into the open .MieWB so "
            "they travel with the project.")
        self.settings_action.triggered.connect(
            self._on_simulation_settings_dialog)

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

        self.optimize_action = sim_menu.addAction("&Optimize…")
        self.optimize_action.setToolTip(
            "Open the merit-function optimizer (drive design variables "
            "to minimize spot RMS / maximize detected power, with a live "
            "convergence plot)")
        self.optimize_action.triggered.connect(self._on_show_optimize)

        self.tolerance_action = sim_menu.addAction("&Tolerance…")
        self.tolerance_action.setToolTip(
            "Open the tolerancing study pane (rank which tolerances "
            "dominate the merit, Monte-Carlo the as-built yield, and "
            "recover it with a focus compensator)")
        self.tolerance_action.triggered.connect(self._on_show_tolerance)

        self.stop_action = sim_menu.addAction("&Stop")
        self.stop_action.setToolTip("Stop the running pipeline")
        self.stop_action.triggered.connect(self.runner.stop)

        # per-fold "Folds" menu: one checkable action per fold element
        # (checked = folded), rebuilt from the live TrainModel whenever
        # the train indicators refresh. Exposed both here and as a
        # toolbar dropdown (_build_toolbar) sharing this SAME QMenu
        # instance so the two stay in sync for free.
        sim_menu.addSeparator()
        self.folds_menu = QMenu("&Folds", self)
        sim_menu.addMenu(self.folds_menu)

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

        act = tools_menu.addAction("&Zoom-pair Calculator…")
        act.setToolTip("Two-group zoom relationship: BFL(z)/EFL(z)/total "
                       "track for a front+rear focal-length pair, with a "
                       "copyable train-grammar expression string")
        act.triggered.connect(self._open_zoom_pair_calculator)

        act = tools_menu.addAction("&Field-angle Fan…")
        act.setToolTip("Insert N field-point sources aimed at a pivot "
                       "(one per field angle) — the source layout the "
                       "imaging products (distortion, vignetting, field "
                       "curves, telecentricity) analyze")
        act.triggered.connect(self._open_field_fan_wizard)

        view_menu = menubar.addMenu("&View")
        dock_toggles = [self.outliner_dock, self.train_editor_dock,
                        self.inspector_dock, self.element_editor_dock,
                        self.transform_dock, self.library_dock,
                        self.console_dock, self.py_console_dock,
                        self.problems_dock, self.compare_dock]
        if self.variables_dock is not None:
            dock_toggles.insert(6, self.variables_dock)
        for dock in dock_toggles:
            view_menu.addAction(dock.toggleViewAction())

        view_menu.addSeparator()
        self.face_indicators_action = view_menu.addAction(
            "Face Orientation &Indicators")
        self.face_indicators_action.setCheckable(True)
        self.face_indicators_action.setToolTip(
            "Show orientation glyphs on faces: red half-disc = source "
            "emit / detector face, blue dot = optic +x face, green = "
            "aperture +x face (visual only, never traced)")
        show_indicators = self.settings.get_bool("show_face_indicators",
                                                 True)
        self.face_indicators_action.setChecked(show_indicators)
        self._apply_face_indicators(show_indicators)
        self.face_indicators_action.toggled.connect(
            self._on_face_indicators_toggled)

        self.auto_preview_action = view_menu.addAction(
            "Auto-update Ray &Preview")
        self.auto_preview_action.setCheckable(True)
        self.auto_preview_action.setToolTip(
            "Re-trace the live preview fan automatically ~1 s after any "
            "optics-affecting edit (stale rays grey out immediately)")
        auto_preview = self.settings.get_bool("auto_preview_rays", True)
        self.auto_preview_action.setChecked(auto_preview)
        self.preview_scheduler.set_enabled(auto_preview)
        self.auto_preview_action.toggled.connect(
            self._on_auto_preview_toggled)

        # Ray dimming: submenu kept as an attribute on self, NEVER
        # retrieved back via QAction.menu() (ownership would transfer to
        # Python and the GC would delete the C++ menu).
        self.ray_dimming_menu = view_menu.addMenu("Ray &Dimming")
        self.ray_dimming_menu.setToolTipsVisible(True)
        group = QActionGroup(self)
        group.setExclusive(True)

        def dim_action(label, mode, tip):
            act = self.ray_dimming_menu.addAction(label)
            act.setCheckable(True)
            act.setToolTip(tip)
            act.triggered.connect(
                lambda checked=False, m=mode: self._on_ray_dimming_mode(m))
            group.addAction(act)
            return act

        self.ray_dim_off_action = dim_action(
            "&Off", "off",
            "Rays render fully opaque regardless of remaining power")
        self.ray_dim_linear_action = dim_action(
            "&Linear (opacity = P/P₀)", "linear",
            "Fade each segment linearly with its remaining power relative "
            "to the ray's power at the source; splits/reflections dim "
            "consistently (applies to the live preview and loaded run "
            "overlays)")
        self.ray_dim_sqrt_action = dim_action(
            "&Perceptual (opacity = √(P/P₀))", "sqrt",
            "Square-root curve: compensates the eye's nonlinearity so a "
            "50/50 split looks half as bright instead of nearly gone "
            "after a few bounces")
        self.ray_dimming_menu.addSeparator()
        self.ray_dim_floor_action = self.ray_dimming_menu.addAction(
            "&Minimum Opacity…")
        self.ray_dim_floor_action.setToolTip(
            "Floor the dimmed opacity at a percentage so heavily "
            "attenuated rays stay faintly traceable (0 = fade fully to "
            "invisible)")
        self.ray_dim_floor_action.triggered.connect(
            self._on_ray_dimming_floor)

        self._ray_dim_mode = self.settings.get("ray_dimming_mode", "off")
        if self._ray_dim_mode not in ("off", "linear", "sqrt"):
            self._ray_dim_mode = "off"
        try:
            self._ray_dim_floor = float(
                self.settings.get("ray_dimming_floor", "0") or 0)
        except (TypeError, ValueError):
            self._ray_dim_floor = 0.0
        {"off": self.ray_dim_off_action,
         "linear": self.ray_dim_linear_action,
         "sqrt": self.ray_dim_sqrt_action}[self._ray_dim_mode].setChecked(
            True)
        self._apply_ray_dimming()

        # tracer-bead animation on/off: ONE checkable action shared by
        # this menu and the Animation toolbar (Qt syncs both natively)
        self.anim_enable_action = view_menu.addAction(
            "Tracer Bead &Animation")
        self.anim_enable_action.setCheckable(True)
        self.anim_enable_action.setToolTip(
            "Animate spheres ('tracer beads') riding each ray at the "
            "physical speed c/n — beads slow down inside glass, splits "
            "spawn both children the instant the parent arrives. "
            "Controlled from the Animation toolbar; beads are hidden "
            "when off.")
        self.anim_enable_action.setChecked(self.anim_controller.enabled)
        self.anim_enable_action.toggled.connect(
            self._on_anim_enabled_toggled)

        # kept as self.help_menu, NEVER re-fetched via QAction.menu() (see
        # _build_help_menu's docstring / CLAUDE.md's PySide6 trap)
        self.help_menu = menubar.addMenu("&Help")
        help_menu = self.help_menu
        self._build_help_menu(help_menu)
        help_menu.addSeparator()
        act = help_menu.addAction("&About")
        act.setToolTip("About MieWorkbench")
        act.triggered.connect(self._on_about)

    def _build_help_menu(self, help_menu):
        """Per-feature guide entries: one action per docs/guide/*.md page,
        grouped into GUI/System/Walkthroughs submenus mirroring
        docs/guide/README.md's index, plus "Open Documentation Folder".
        Degrades gracefully (disabled items, no crash) when docs/guide/
        is missing -- e.g. a stripped-down deployment that ships only the
        compiled app. Submenu references are kept on help_menu itself
        (help_menu.doc_submenus), never re-fetched via QAction.menu():
        PySide6 hands ownership of a NEW wrapper to Python on that call,
        so the GC would delete the underlying C++ QMenu out from under a
        later access (CLAUDE.md's PySide6 trap)."""
        have_guide = os.path.isdir(DOCS_GUIDE_DIR)
        help_menu.doc_submenus = {}
        for group, pages in DOCS_GUIDE_PAGES.items():
            submenu = help_menu.addMenu(group)
            help_menu.doc_submenus[group] = submenu
            for label, relpath in pages:
                path = os.path.join(DOCS_GUIDE_DIR, relpath)
                act = submenu.addAction(label)
                exists = have_guide and os.path.isfile(path)
                act.setEnabled(exists)
                act.setToolTip(path if exists else
                               "%s (not found)" % path)
                if exists:
                    act.triggered.connect(
                        lambda _c=False, p=path: self._open_doc_page(p))

        help_menu.addSeparator()
        self.open_docs_folder_action = help_menu.addAction(
            "Open Documentation &Folder")
        self.open_docs_folder_action.setToolTip(DOCS_GUIDE_DIR)
        self.open_docs_folder_action.setEnabled(have_guide)
        if have_guide:
            self.open_docs_folder_action.triggered.connect(
                lambda: self._open_doc_page(DOCS_GUIDE_DIR))

    @staticmethod
    def _open_doc_page(path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

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
        self.folds_toolbutton = QToolButton()
        self.folds_toolbutton.setText("Folds ▾")
        self.folds_toolbutton.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup)
        self.folds_toolbutton.setMenu(self.folds_menu)
        toolbar.addWidget(self.folds_toolbutton)

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

        # the SAME checkable QAction lives in the View menu and here --
        # Qt keeps the two representations in sync natively. _build_menus()
        # (which creates face_indicators_action) always runs before
        # _build_toolbar() -- see __init__.
        self.face_indicators_action.setIconText("Face marks")
        toolbar.addAction(self.face_indicators_action)

        ext_label = QLabel(" Extinction: ")
        ext_label.setToolTip("Ray extinction (attenuation dimming): fade "
                             "ray segments by remaining power")
        toolbar.addWidget(ext_label)
        self.ray_dim_combo = QComboBox()
        for label in ("Off", "Linear", "Perceptual"):
            self.ray_dim_combo.addItem(label)
        self.ray_dim_combo.setToolTip(
            "Ray extinction: Off = full opacity; Linear = opacity is each "
            "segment's remaining power relative to its ray's power at the "
            "source (P/P₀); Perceptual = √(P/P₀). Same setting as "
            "View ▸ Ray Dimming; applies live to loaded rays. Dims ray "
            "LINES only -- animation bead opacity is a separate control "
            "on the Animation toolbar.")
        self.ray_dim_combo.setCurrentIndex(
            _RAY_DIM_MODES.index(self._ray_dim_mode))
        self.ray_dim_combo.currentIndexChanged.connect(
            self._on_ray_dim_combo)
        toolbar.addWidget(self.ray_dim_combo)

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
        self._warn_if_dim_data_missing()

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
        text, ok = QInputDialog.getText(
            self, "Live ray preview",
            "Pattern (--viz-pattern spec, e.g. 'fan:n=5' or "
            "'rings:dr=1:nper=12[:nrings=K]'; a bare integer means "
            "fan:n=<int>):",
            text=self._preview_pattern_spec())
        if not ok:
            return
        spec = text.strip()
        try:
            spec = "fan:n=%d" % int(spec)
        except ValueError:
            pass   # not a bare integer -- use the typed spec as-is
        try:
            common.parse_viz_pattern_spec(spec)
        except ValueError as exc:
            self.statusBar().showMessage(
                "Invalid ray-preview pattern: %s" % exc, 6000)
            return
        self._preview_target = target
        started = self.raypreview.start(
            self.project, self._preview_workspace(),
            pattern=spec, only_bodies=only_bodies,
            optical_properties=self._workspace_optprops())
        if started:
            self.statusBar().showMessage(
                "Tracing preview rays (%s)…" % spec)

    def _preview_pattern_spec(self):
        """Resolve the ray-preview pattern spec: the open project's
        stored spec (Project.get_preview_config, re-validated -- a
        hand-edited or stale document must never surface a broken
        pattern) -> the last spec used this install (QSettings) ->
        the "fan:n=5" default. Never raises."""
        if self.project.is_open():
            try:
                cfg = self.project.get_preview_config()
            except Exception:
                cfg = None
            spec = (cfg or {}).get("spec") if cfg else None
            if spec:
                try:
                    common.parse_viz_pattern_spec(spec)
                    return spec
                except ValueError:
                    pass
        stored = self.settings.get("preview_pattern_spec", None)
        if stored:
            try:
                common.parse_viz_pattern_spec(stored)
                return stored
            except ValueError:
                pass
        return "fan:n=5"

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

    def _on_preview_finished(self, vtp_path, engine="engine fan"):
        if self._preview_target == "inspector":
            self.inspector.load_rays_vtp(vtp_path)
        else:
            self.scene3d.load_rays_vtp(vtp_path)
            self.scene3d.set_rays_stale(False)
            # a whole-scene fan passes through the inspected element too:
            # refresh a checked inspector overlay rather than leaving it
            # greyed-out stale
            if self.inspector.rays_button.isChecked():
                self.inspector.load_rays_vtp(vtp_path)
        # engine hint (P4b preview unification): which trace engine produced
        # this overlay -- "sequential (exact)" (Optiland, deterministic, no
        # MC noise) vs "engine fan" (the general Python-engine viz trace,
        # the fallback for scenes outside the sequential bridge's scope).
        self.statusBar().showMessage("Ray preview ready — %s" % engine, 5000)
        self._warn_if_dim_data_missing()

    def _on_preview_failed(self, message):
        self.console.append_line("[preview] " + message)
        self.console_dock.raise_()
        self.statusBar().showMessage("Ray preview failed — see Console",
                                     8000)

    def _on_geometry_changed(self, *_args):
        self.scene3d.set_rays_stale(True)

    # -- auto preview ------------------------------------------------------------
    def _on_optics_changed(self):
        """Anything trace-relevant changed: grey the loaded ray overlays
        immediately and (if enabled) schedule a debounced auto preview."""
        self.scene3d.set_rays_stale(True)
        self.inspector.set_rays_stale(True)
        self.preview_scheduler.notify_change()

    def _start_scene_preview(self):
        """Guarded launch of a whole-scene preview (the resolved
        pattern spec, see _preview_pattern_spec) into the main 3D
        view. Shared by the debounced auto-preview scheduler and the
        bead-animation enable handler -- one implementation, both
        callers. Returns True if a preview was actually launched."""
        if not self.project.is_open():
            return False
        if self.runner.is_running():
            return False   # never compete with a real pipeline run; its
                            # own rays load when it completes
        if self.raypreview.is_running():
            # a preview is already in flight; queue one more behind it
            self.preview_scheduler.notify_busy(True)
            self.preview_scheduler.notify_change()
            return False
        self._preview_target = "scene"
        started = self.raypreview.start(
            self.project, self._preview_workspace(),
            pattern=self._preview_pattern_spec(),
            optical_properties=self._workspace_optprops())
        if started:
            self.preview_scheduler.notify_busy(True)
            self.statusBar().showMessage("Auto-updating ray preview…",
                                         3000)
        return started

    def _on_auto_preview_wanted(self):
        self._start_scene_preview()

    def _on_auto_preview_toggled(self, checked):
        self.preview_scheduler.set_enabled(checked)
        self.settings.set_bool("auto_preview_rays", checked)

    # -- ray dimming -----------------------------------------------------------
    def _apply_ray_dimming(self):
        self.scene3d.view.set_ray_dimming(self._ray_dim_mode,
                                          self._ray_dim_floor)
        self.inspector.view.set_ray_dimming(self._ray_dim_mode,
                                            self._ray_dim_floor)
        self._warn_if_dim_data_missing()

    def _warn_if_dim_data_missing(self):
        if (self.scene3d.view.ray_dimming_data_missing()
                or self.inspector.view.ray_dimming_data_missing()):
            self.statusBar().showMessage(
                "Loaded rays predate per-segment power data (rel_power) — "
                "extinction has no effect until you re-run or re-preview.",
                6000)

    def _set_ray_dim_ui(self, mode):
        """Sync BOTH extinction editors (View-menu radio group + toolbar
        combo) to `mode` with signals blocked, so neither re-triggers the
        handler. The combo may not exist yet during _build_menus."""
        action = {"off": self.ray_dim_off_action,
                  "linear": self.ray_dim_linear_action,
                  "sqrt": self.ray_dim_sqrt_action}[mode]
        for act in (self.ray_dim_off_action, self.ray_dim_linear_action,
                    self.ray_dim_sqrt_action):
            act.blockSignals(True)
        action.setChecked(True)
        for act in (self.ray_dim_off_action, self.ray_dim_linear_action,
                    self.ray_dim_sqrt_action):
            act.blockSignals(False)
        combo = getattr(self, "ray_dim_combo", None)
        if combo is not None:
            combo.blockSignals(True)
            combo.setCurrentIndex(_RAY_DIM_MODES.index(mode))
            combo.blockSignals(False)

    def _on_ray_dim_combo(self, index):
        self._on_ray_dimming_mode(_RAY_DIM_MODES[index])

    def _on_ray_dimming_mode(self, mode):
        self._ray_dim_mode = mode
        self._set_ray_dim_ui(mode)
        self._apply_ray_dimming()
        self.settings.set("ray_dimming_mode", mode)

    def _set_ray_dimming_floor(self, floor_pct):
        """Dialog-free setter (the QInputDialog path calls this; tests
        call it directly -- modal dialogs hang offscreen runs)."""
        self._ray_dim_floor = max(0.0, min(100.0, float(floor_pct)))
        self._apply_ray_dimming()
        self.settings.set("ray_dimming_floor", str(self._ray_dim_floor))

    def _on_ray_dimming_floor(self):
        value, ok = QInputDialog.getDouble(
            self, "Ray Dimming Minimum Opacity",
            "Minimum segment opacity (% of fully opaque):",
            self._ray_dim_floor, 0.0, 100.0, 1)
        if ok:
            self._set_ray_dimming_floor(value)

    # -- face indicators -------------------------------------------------------
    def _apply_face_indicators(self, visible):
        self.scene3d.view.set_face_indicators_visible(visible)
        self.inspector.view.set_face_indicators_visible(visible)

    def _on_face_indicators_toggled(self, checked):
        self._apply_face_indicators(checked)
        self.settings.set_bool("show_face_indicators", checked)

    # -- tracer-bead animation ---------------------------------------------------
    def _anim_setting(self, key, default):
        try:
            return float(self.settings.get(key, str(default)) or default)
        except (TypeError, ValueError):
            return default

    def _init_animation(self):
        """Controller + persisted defaults; the scene3d view owns the
        BeadLayer (beads live in the main 3D view only)."""
        view = self.scene3d.view
        self.anim_controller = AnimationController(
            layer=view.beads, render=view._render, parent=self)
        self.anim_controller.apply_settings(
            bead_size_mm=self._anim_setting("anim_bead_size", 1.0),
            speed_mm_s=self._anim_setting("anim_speed_mm_s", 2.0),
            fps=int(self._anim_setting("anim_fps", 15)),
            ray_cap=int(self._anim_setting("anim_ray_cap", 300)),
            bead_opacity_mode=self.settings.get("anim_bead_opacity_mode",
                                                "off"),
            bead_opacity_range_db=self._anim_setting("anim_bead_opacity_db",
                                                     30.0),
            enabled=self.settings.get_bool("anim_enabled", False))
        view.overlayChanged.connect(self._on_scene_overlay_changed)
        self.anim_controller.frameAdvanced.connect(self._on_anim_frame)
        self.anim_controller.availabilityChanged.connect(
            lambda _avail: self._update_anim_transport_enabled())

    def _build_animation_toolbar(self):
        tb = self.addToolBar("Animation")
        tb.setObjectName("animation_toolbar")
        style = self.style()

        # the SAME checkable QAction lives in the View menu and here --
        # Qt keeps the two representations in sync natively
        tb.addAction(self.anim_enable_action)
        tb.addSeparator()

        self.anim_play_action = tb.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Play")
        self.anim_play_action.setToolTip(
            "Play the tracer beads (loops until Stop)")
        self.anim_play_action.triggered.connect(self.anim_controller.play)
        self.anim_pause_action = tb.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaPause),
            "Pause")
        self.anim_pause_action.setToolTip("Pause the beads in place")
        self.anim_pause_action.triggered.connect(self.anim_controller.pause)
        self.anim_stop_action = tb.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaStop), "Stop")
        self.anim_stop_action.setToolTip(
            "Stop and rewind: beads return to the sources at t = 0")
        self.anim_stop_action.triggered.connect(self.anim_controller.stop)
        self.anim_step_action = tb.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward),
            "Step")
        self.anim_step_action.setToolTip(
            "Single-step one frame (speed ÷ fps mm of vacuum path)")
        self.anim_step_action.triggered.connect(self.anim_controller.step)
        tb.addSeparator()

        tb.addWidget(QLabel(" Bead "))
        self.anim_size_spin = QDoubleSpinBox()
        self.anim_size_spin.setRange(0.05, 50.0)
        self.anim_size_spin.setDecimals(2)
        self.anim_size_spin.setSingleStep(0.25)
        self.anim_size_spin.setSuffix(" mm")
        self.anim_size_spin.setValue(self.anim_controller.bead_size_mm)
        self.anim_size_spin.setToolTip("Tracer bead diameter-ish sphere "
                                       "radius in scene millimetres")
        self.anim_size_spin.valueChanged.connect(self._on_anim_size)
        tb.addWidget(self.anim_size_spin)

        tb.addWidget(QLabel(" Speed "))
        self.anim_speed_spin = QDoubleSpinBox()
        self.anim_speed_spin.setRange(0.01, 1000.0)
        self.anim_speed_spin.setDecimals(2)
        self.anim_speed_spin.setSuffix(" mm/s")
        self.anim_speed_spin.setValue(self.anim_controller.speed_mm_s)
        self.anim_speed_spin.setToolTip(
            "Playback speed: mm of ray path per real second for a bead "
            "in vacuum -- beads inside glass move slower by 1/n")
        self.anim_speed_spin.valueChanged.connect(self._on_anim_speed)
        tb.addWidget(self.anim_speed_spin)

        tb.addWidget(QLabel(" FPS "))
        self.anim_fps_combo = QComboBox()
        for fps in (5, 10, 15, 24, 30):
            self.anim_fps_combo.addItem(str(fps))
        self.anim_fps_combo.setCurrentText(str(self.anim_controller.fps))
        self.anim_fps_combo.setToolTip("Animation frames per second")
        self.anim_fps_combo.currentTextChanged.connect(self._on_anim_fps)
        tb.addWidget(self.anim_fps_combo)

        tb.addWidget(QLabel(" Cap "))
        self.anim_ray_cap_spin = QSpinBox()
        self.anim_ray_cap_spin.setRange(1, 100000)
        self.anim_ray_cap_spin.setSingleStep(50)
        self.anim_ray_cap_spin.setValue(
            int(self._anim_setting("anim_ray_cap", 300)))
        self.anim_ray_cap_spin.setToolTip(
            "Max animated rays per source")
        self.anim_ray_cap_spin.valueChanged.connect(self._on_anim_ray_cap)
        tb.addWidget(self.anim_ray_cap_spin)

        tb.addSeparator()
        tb.addWidget(QLabel(" Bead opacity "))
        self.anim_opacity_combo = QComboBox()
        self.anim_opacity_combo.addItem("Opaque", "off")
        self.anim_opacity_combo.addItem("By power", "power")
        cur_mode = self.anim_controller.bead_opacity_mode
        self.anim_opacity_combo.setCurrentIndex(1 if cur_mode == "power"
                                                else 0)
        self.anim_opacity_combo.setToolTip(
            "Bead opacity: Opaque (default) or fade beads by their optical "
            "power over a log-dB range, leading-wavefront beads stay solid")
        self.anim_opacity_combo.currentIndexChanged.connect(
            self._on_anim_opacity_mode)
        tb.addWidget(self.anim_opacity_combo)

        self.anim_opacity_db_spin = QDoubleSpinBox()
        self.anim_opacity_db_spin.setRange(10.0, 60.0)
        self.anim_opacity_db_spin.setDecimals(0)
        self.anim_opacity_db_spin.setSingleStep(5.0)
        self.anim_opacity_db_spin.setSuffix(" dB")
        self.anim_opacity_db_spin.setValue(
            self.anim_controller.bead_opacity_range_db)
        self.anim_opacity_db_spin.setToolTip(
            "Dynamic range of the power-to-opacity map: a bead at "
            "Pmax/10^(dB/10) fades to the faint floor")
        self.anim_opacity_db_spin.setEnabled(cur_mode == "power")
        self.anim_opacity_db_spin.valueChanged.connect(self._on_anim_opacity_db)
        tb.addWidget(self.anim_opacity_db_spin)

        tb.addSeparator()
        self.anim_readout = QLabel("")
        self.anim_readout.setToolTip(
            "Simulation clock (auto unit) and the vacuum-equivalent "
            "optical path c·t travelled so far")
        tb.addWidget(self.anim_readout)
        self._on_anim_frame(0.0, 0.0)
        self._update_anim_transport_enabled()

    def _on_anim_enabled_toggled(self, checked):
        self.anim_controller.apply_settings(enabled=checked)
        self.settings.set_bool("anim_enabled", checked)
        self._update_anim_transport_enabled()
        if not checked:
            return
        # self-sufficient enable: if there is nothing to animate yet (no
        # segments, or the loaded overlay is stale) generate one instead
        # of just complaining -- beads park paused at t=0 when the fresh
        # segments land (_on_scene_overlay_changed), never auto-play.
        needs_preview = (not self.anim_controller.has_segments()
                         or self.scene3d.view.overlay_is_stale())
        if (needs_preview and self.project.is_open()
                and not self.runner.is_running()
                and not self.raypreview.is_running()):
            self._start_scene_preview()
            self.statusBar().showMessage("Generating ray preview…", 4000)
        else:
            # genuinely blocked (no project, a real run/preview already
            # in flight, ...) -- fall back to the informational warning
            self._warn_if_anim_data_missing()

    def _on_anim_size(self, value):
        self.anim_controller.apply_settings(bead_size_mm=value)
        self.settings.set("anim_bead_size", str(value))

    def _on_anim_speed(self, value):
        self.anim_controller.apply_settings(speed_mm_s=value)
        self.settings.set("anim_speed_mm_s", str(value))

    def _on_anim_fps(self, text):
        try:
            self.anim_controller.apply_settings(fps=int(text))
            self.settings.set("anim_fps", text)
        except ValueError:
            pass

    def _on_anim_ray_cap(self, value):
        self.anim_controller.apply_settings(ray_cap=value)
        self.settings.set("anim_ray_cap", str(value))

    def _on_anim_opacity_mode(self, _index):
        mode = self.anim_opacity_combo.currentData() or "off"
        self.anim_controller.apply_settings(bead_opacity_mode=mode)
        self.settings.set("anim_bead_opacity_mode", mode)
        self.anim_opacity_db_spin.setEnabled(mode == "power")
        if (mode == "power" and self.anim_controller.has_segments()
                and not self.anim_controller.power_available()):
            self.statusBar().showMessage(
                "Loaded rays predate per-segment power — beads stay opaque "
                "until you re-run or re-preview.", 6000)

    def _on_anim_opacity_db(self, value):
        self.anim_controller.apply_settings(bead_opacity_range_db=value)
        self.settings.set("anim_bead_opacity_db", str(value))

    def _on_anim_frame(self, clock_s, path_mm):
        readout = getattr(self, "anim_readout", None)
        if readout is not None:
            readout.setText(" t = %s   path = %.2f mm "
                            % (format_sim_time(clock_s), path_mm))

    def _update_anim_transport_enabled(self):
        play = getattr(self, "anim_play_action", None)
        if play is None:
            return                       # toolbar not built yet
        avail = (self.anim_controller.has_segments()
                 and self.anim_controller.enabled)
        for act in (self.anim_play_action, self.anim_pause_action,
                    self.anim_stop_action, self.anim_step_action):
            act.setEnabled(avail)

    def _on_scene_overlay_changed(self):
        view = self.scene3d.view
        if view.overlay_is_stale() or view._rays_polydata is None:
            self.anim_controller.set_segments(None)
            return
        self.anim_controller.set_segments(
            precompute_segments(view._rays_polydata))
        if self.anim_controller.enabled:
            self._warn_if_anim_data_missing()

    def _warn_if_anim_data_missing(self):
        view = self.scene3d.view
        if (view._rays_polydata is not None
                and not view.overlay_is_stale()
                and not self.anim_controller.has_segments()):
            self.statusBar().showMessage(
                "Loaded rays predate timing data (opl) — the bead "
                "animation is unavailable until you re-run or "
                "re-preview.", 6000)

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
        self.transform_panel.set_scene_view(self.scene3d.view)
        self.problems.set_project(self.project)
        self.problems.set_sources(
            lambda: (self.library_manager.project_lib
                     or self.library_manager.system_lib).load(),
            lambda: self.config_matrix.values())

        # all selection flows through the shared SelectionModel: 3D picks,
        # outliner rows and problems-pane jumps stay in sync. A bare-body
        # origin that names a member of a multi-body group is EXPANDED to a
        # whole-element selection by the dispatcher; the two explicit sub-
        # selection origins (outliner child rows, inspector member lists)
        # are exempt.
        self.outliner.set_project(self.project)
        self.scene3d.selectionChanged.connect(
            lambda body, faces: self.selection.select(body, faces,
                                                      origin="scene3d"))
        self.outliner.selectElementRequested.connect(
            self._on_outliner_element_selected)
        self.outliner.selectBodyRequested.connect(
            lambda body: self.selection.select(body, (),
                                               origin="outliner_child"))
        self.problems.selectBodyRequested.connect(
            lambda body: self.selection.select(body, (), origin="problems"))
        self.inspector.memberSelected.connect(
            lambda body: self.selection.select(body, (),
                                               origin="inspector_member"))
        self.element_editor.memberSelected.connect(
            lambda body: self.selection.select(body, (),
                                               origin="inspector_member"))
        self.selection.changed.connect(self._on_selection_changed)
        # Clear-selection lives on the 3D-view button row too
        self.scene3d.add_toolbar_action(self.clear_selection_action)

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

        # optical-train editor: arming "pick reference in 3D" routes the
        # scene view's one-shot pick back into the pane (mirrors the
        # transform panel's snap-to-face pick path).
        self.train_editor.pickReferenceRequested.connect(
            self._on_train_pick_reference)
        # train indicators (excluded bodies, chain links, outliner badges)
        # recompute after any load/property/move, coalesced to one pass.
        self.project.sceneLoaded.connect(self._schedule_train_refresh)
        self.project.propertiesChanged.connect(self._schedule_train_refresh)
        self.project.bodiesMoved.connect(self._schedule_train_refresh)
        if self.variables_pane is not None:
            self.project.sceneLoaded.connect(
                lambda: self.variables_pane.refresh())

        # Optimize/Tolerance name dropdowns track the miewb_vars sheet
        # (same Project signals the Variables dock refreshes on)
        self.project.sceneLoaded.connect(self._refresh_pane_variables)
        self.project.propertiesChanged.connect(
            self._refresh_pane_variables)
        # Pre-populate the panes from any config stashed on the scene --
        # scene-open ONLY (not propertiesChanged: that fires on every
        # unrelated property write, which would stomp in-progress
        # unsaved pane edits). Runs after _refresh_pane_variables so the
        # name combos are already seeded when apply_config rebuilds rows.
        self.project.sceneLoaded.connect(self._load_pane_configs)

        # a finished sweep hands its manifest to the Compare pane
        self.runner.finished.connect(self._maybe_run_compare)

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

        # auto ray-preview: any optics-affecting edit greys the overlays
        # and (debounced) re-traces the preview fan
        self.project.opticsChanged.connect(self._on_optics_changed)
        self.preview_scheduler.previewWanted.connect(
            self._on_auto_preview_wanted)
        self.raypreview.finished.connect(
            lambda _path: self.preview_scheduler.notify_run_finished())
        self.raypreview.failed.connect(
            lambda _msg: self.preview_scheduler.notify_run_failed())

    def _refresh_pane_variables(self, *_args):
        """Feed the Optimize/Tolerance panes' variable-name dropdowns
        from the scene's miewb_vars sheet (empty dict when no scene or
        no sheet — the combos stay editable free-text)."""
        sheet = (self.project.variables_sheet()
                 if self.project.is_open() else None)
        try:
            varrows = variables.parse_sheet(sheet) if sheet else {}
        except Exception:
            varrows = {}
        self.optimize_pane.set_variables(varrows)
        self.tolerance_pane.set_variables(varrows)

    def _load_pane_configs(self, *_args):
        """Pre-populate the Optimize/Tolerance panes from any config
        stashed on the miewb_vars sheet (Project.set_optimize_config/
        set_tolerance_config) -- so a saved scene reopens with the panes
        already configured, even though the run itself was not repeated.
        Best-effort: a missing/stale/unparseable config must never block
        opening the scene."""
        if not self.project.is_open():
            return
        try:
            opt_cfg = self.project.get_optimize_config()
        except Exception:
            opt_cfg = None
        if opt_cfg:
            try:
                self.optimize_pane.apply_config(opt_cfg)
            except Exception:
                pass
        try:
            tol_cfg = self.project.get_tolerance_config()
        except Exception:
            tol_cfg = None
        if tol_cfg:
            try:
                self.tolerance_pane.apply_config(tol_cfg)
            except Exception:
                pass
        try:
            self.preview_config.set_spec(self._preview_pattern_spec())
        except Exception:
            pass

    def _on_scene_loaded(self):
        has_doc = self.project.is_open()
        self.save_action.setEnabled(has_doc)
        self.save_as_action.setEnabled(has_doc)
        self.export_fcstd_action.setEnabled(has_doc)
        self.revert_action.setEnabled(has_doc)
        self.close_action.setEnabled(has_doc)
        self._update_window_title()

    # origins that are EXPLICIT sub-selections (a single member body) and so
    # must NOT be expanded into a whole-element selection
    _NO_EXPAND_ORIGINS = {"outliner_child", "inspector_member"}

    def _on_outliner_element_selected(self, element, primary):
        """A top-level outliner row: select the whole element (expand to its
        member bodies; a single-body element stays a plain body select)."""
        try:
            bodies = self.project.element_bodies(element)
        except Exception:
            bodies = [primary] if primary else []
        if len(bodies) > 1:
            self.selection.select_element(element, bodies, origin="outliner",
                                          primary=primary)
        else:
            self.selection.select(primary or element, (), origin="outliner")

    def _on_selection_changed(self, body_name, faces, origin):
        # Expand a bare-body 3D/problems/programmatic pick into its whole
        # element (union highlight, element panes). The element==None guard
        # stops the re-entrant select_element from re-expanding (loop break);
        # explicit sub-selection origins skip expansion entirely.
        if (body_name and origin not in self._NO_EXPAND_ORIGINS
                and self.selection.element is None):
            try:
                element = self.project.element_group(body_name)
                bodies = self.project.element_bodies(element)
            except Exception:
                element, bodies = None, [body_name]
            if len(bodies) > 1:
                self.selection.select_element(
                    element, bodies, faces=faces, origin=origin,
                    primary=body_name)
                return   # select_element re-emits; this pass stops here

        element = self.selection.element
        bodies = list(self.selection.bodies)
        primary = self.selection.body
        faces = self.selection.faces
        single = bool(primary) and not self.selection.is_element()

        # 3D highlight. Driven even for a scene3d-originated pick: the view
        # pre-highlighted only the clicked body, but an expanded element
        # needs the whole-member UNION (select_body/select_element dedupe
        # the redundant single-body case, so no double render).
        if single:
            self.scene3d.select_body(primary)
        else:
            self.scene3d.select_element(bodies)

        # inspector + element editor: single body -> face-selection surface;
        # multi-body element OR empty -> blank + count hint + member list
        if single:
            self.inspector.set_body(self.project, primary)
            self.element_editor.set_face_selection(primary, set(faces))
        else:
            self.inspector.set_element(self.project, element, bodies)
            self.element_editor.set_element(element, bodies)

        # transform panel operates on the primary (group moves are safe via
        # Project._flush_placement/miewb_group); None -> neutral
        self.transform_panel.set_body(primary)

        # outliner echo (skip when the outliner initiated the selection)
        if origin not in ("outliner", "outliner_child"):
            self.outliner.set_selected_body(primary)

        self._update_selection_actions()

    def _update_selection_actions(self):
        """Single authority for selection-dependent action state: copy/
        delete/clear + the transform panel's operation buttons, all keyed
        off whether anything is selected."""
        has = bool(self.selection.body)
        self.copy_action.setEnabled(has)
        self.delete_action.setEnabled(has)
        self.clear_selection_action.setEnabled(has)
        self.transform_panel.set_operations_enabled(has)

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

    # -- optical-train indicators ---------------------------------------------
    def _on_train_pick_reference(self, _element):
        """Arm the scene view's one-shot pick and route the result back
        into the train editor (mirrors transform_panel's snap-face pick)."""
        self.scene3d.view.pick_face_once(
            self.train_editor.on_reference_picked)

    def _schedule_train_refresh(self, *_args):
        """Coalesce a burst of project signals into one indicator pass."""
        if self._train_refresh_pending:
            return
        self._train_refresh_pending = True
        QTimer.singleShot(0, self._refresh_train_indicators)

    def _refresh_train_indicators(self):
        """Push exclusion ghosting + chain-link overlay to the 3D view and
        train badges to the outliner from the current TrainModel."""
        self._train_refresh_pending = False
        view = self.scene3d.view
        if not self.project.is_open():
            view.set_excluded_bodies(set())
            view.set_chain_links([])
            self.outliner.set_train_info({})
            self._rebuild_folds_menu()
            return
        structure = self.project.structure or {}

        # excluded bodies: an element is excluded if ANY of its bodies
        # carries a truthy miewb_exclude (unfolded fold mirrors); ghost the
        # whole element.
        excl_elements = set()
        for b in structure.get("bodies", []):
            entry = (b.get("properties") or {}).get(EXCLUDE_PROP)
            if entry and entry.get("value"):
                try:
                    excl_elements.add(self.project.element_group(b["name"]))
                except ProjectError:
                    pass
        excluded = set()
        for el in excl_elements:
            try:
                excluded.update(self.project.element_bodies(el))
            except ProjectError:
                pass
        view.set_excluded_bodies(excluded)

        tm = self.project.train()
        records = tm.records()
        view.set_chain_links(self._chain_links(tm, records))

        # outliner badges: chained/fold/folded/excluded + a validation
        # problem string matched to an element by substring of its label.
        try:
            problems = tm.validate()
        except Exception:
            problems = []
        info = {}
        for el in tm.element_labels():
            rec = records[el]
            problem = next((msg for _sev, msg in problems if el in msg),
                           None)
            info[el] = {
                "chained": rec.get("mode") == "chained",
                "fold": bool(rec.get("fold")),
                "folded": bool(rec.get("folded", True)),
                "excluded": el in excl_elements,
                "problem": problem,
            }
        self.outliner.set_train_info(info)
        self._rebuild_folds_menu(tm)

    def _rebuild_folds_menu(self, tm=None):
        """Repopulate the shared Folds menu (Simulation menu + toolbar
        dropdown) from the current TrainModel: one checkable action per
        fold element (checked = folded, "(excluded)" suffix when
        unfolded), then Unfold/Refold all. Signals are blocked while
        clearing/re-adding actions so the rebuild can't re-enter
        set_fold_state via a stray toggled() during teardown."""
        menu = self.folds_menu
        menu.blockSignals(True)
        try:
            menu.clear()
            tm = tm if tm is not None else self.project.train()
            records = tm.records()
            folds = sorted(tm.folds())
            for element in folds:
                folded = bool(records[element].get("folded", True))
                text = element if folded else "%s (excluded)" % element
                act = menu.addAction(text)
                act.setCheckable(True)
                act.setChecked(folded)
                act.triggered.connect(
                    lambda checked, el=element:
                        self._on_toggle_fold(el, checked))
            menu.addSeparator()
            act = menu.addAction("Unfold all")
            act.triggered.connect(lambda: self.project.set_folds_all(False))
            act = menu.addAction("Refold all")
            act.triggered.connect(lambda: self.project.set_folds_all(True))
        finally:
            menu.blockSignals(False)
        has_folds = bool(folds)
        self.folds_toolbutton.setEnabled(has_folds)
        self.folds_toolbutton.setToolTip(
            "Fold/unfold individual segments" if has_folds
            else "No fold elements in this scene")

    def _on_toggle_fold(self, element, checked):
        try:
            self.project.set_fold_state(element, checked)
        except ProjectError as e:
            self.statusBar().showMessage(str(e))

    def _chain_links(self, tm, records):
        """[{from, to, kind}] mm-world links from each chained element's
        parent exit-port frame origin to the element's entry point."""
        if not tm.has_train():
            return []
        try:
            solved = tm.solve(self.project.train_variables())
        except TrainError:
            return []
        frames = solved.get("frames", {})
        links = []
        for el in tm.chained_elements():
            rec = records[el]
            ref = rec.get("ref")
            if not ref or ref not in frames:
                continue
            port = rec.get("port") or train_solver._default_port(
                records.get(ref, {}))
            frame = frames.get(ref, {}).get(port)
            if not frame:
                continue
            state = self.project.body_states.get(tm.primary_body_name(el))
            entry_local = (rec.get("local") or {}).get("entry")
            if state is None or entry_local is None:
                continue
            entry_world = train_solver.transform_point(
                state.current.to_dict(), entry_local)
            links.append({
                "from": [float(v) for v in frame["origin"]],
                "to": [float(v) for v in entry_world],
                "kind": "fold" if rec.get("fold") else "chain",
            })
        return links

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
        self._maybe_chain_new_element(label)

    def _maybe_chain_new_element(self, label):
        """User decision: a newly added element chains to the currently
        selected element, or (if nothing is selected but a train exists) to
        the last element in solve order, at a default 10 mm gap. A SEPARATE
        undo step from the import (import_primitive pushes its own macro).
        No dialog/checkbox: this always fires when a chain target exists —
        the user can Undo it independently, or Anchor the element in the
        train editor."""
        try:
            tm = self.project.train()
            if label not in tm.element_labels():
                return
            ref = self._selected_element()
            if ref == label:
                ref = None
            if not ref:
                if not tm.has_train():
                    return          # first element / no train: leave anchored
                try:
                    order = train_solver.sort_chain(tm.records())
                except TrainError:
                    order = tm.element_labels()
                ref = next((el for el in reversed(order) if el != label),
                           None)
            if not ref or ref == label:
                return
            self.project.set_chain(
                label, {"ref": ref, "distance": 10.0},
                text="Chain %s to %s" % (label, ref))
            self.statusBar().showMessage(
                "Chained %s to %s (Ctrl+Z to unchain)" % (label, ref), 5000)
        except (ProjectError, TrainError) as exc:
            self.statusBar().showMessage("Auto-chain skipped: %s" % exc, 5000)

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
        if self.selection.element is not None:
            return self.selection.element
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
            if len(names) > 1:
                self.selection.select_element(new_label, names,
                                              origin="paste")
            else:
                self.selection.select(names[0], (), origin="paste")
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

    def _open_zoom_pair_calculator(self):
        # non-modal like the prop editor; a fresh instance each time is
        # cheap (no state to preserve) and avoids a stale singleton
        # holding onto a closed C++ widget.
        dlg = ZoomPairDialog(self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.show()
        dlg.raise_()

    # -- field-angle fan wizard --------------------------------------------------
    def _source_primitive_infos(self):
        """{kind: info} for the library's Sources-category primitives."""
        return {info["kind"]: info
                for info in self.library_manager.primitives_list()
                if info.get("category") == "Sources"
                and info.get("params")}

    def _open_field_fan_wizard(self):
        if not self.project.is_open():
            QMessageBox.information(
                self, "MieWorkbench",
                "Open or create a model first (File → New… / Open…).")
            return
        infos = self._source_primitive_infos()
        kinds = [(k, i.get("label", k)) for k, i in sorted(infos.items())]
        # laser_collimated first: the canonical field-fan source
        kinds.sort(key=lambda kv: (kv[0] != "laser_collimated", kv[0]))
        dlg = FieldFanDialog(source_kinds=kinds or None, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        design = dlg.result()
        if design is None:
            return
        info = infos.get(design["source_kind"])
        if info is None:
            QMessageBox.warning(self, "Field-angle fan",
                                "Primitive %r not found in the library"
                                % design["source_kind"])
            return
        try:
            self._insert_field_fan(design, info, dlg.aperture_mm())
        except Exception as exc:
            self.project.abort_macro()
            QMessageBox.warning(self, "Field-angle fan failed", str(exc))
            return
        if design.get("note"):
            self.statusBar().showMessage("Field fan: %s" % design["note"],
                                         8000)
        else:
            self.statusBar().showMessage(
                "Added %d field-fan source(s)" % len(design["sources"]),
                5000)

    def _insert_field_fan(self, design, info, aperture_mm=None):
        """Import the fan's N source primitives as ONE undo macro: per
        source import_primitive -> optional diameter edit -> the
        field_angle_deg property (recorded for the imaging products; the
        extractor echo is pending, post_process falls back to emit-face
        directions meanwhile) -> anchored set_placement aiming at the
        pivot."""
        from .panes.library import default_label
        used = {b["label"] for b in self.project.structure.get("bodies",
                                                               [])}
        self.project.begin_macro("Add field fan")
        for entry in design["sources"]:
            base = "Field_%s" % entry["name_suffix"]
            label = base if base not in used else default_label(base, used)
            used.add(label)
            self.project.import_primitive(info["path"], label)
            if aperture_mm is not None and "diameter" in (info.get("params")
                                                          or {}):
                self.project.set_element_parameters(
                    "dim_%s" % label,
                    {"diameter": "=%.10g mm" % aperture_mm},
                    rebuild_group=label)
            names = self.project.element_bodies(label)
            self.project.set_property(names[0], "field_angle_deg",
                                      float(entry["angle_deg"]))
            body = self.project.body(label)["name"]
            self.project.apply_operation(body, Operation(
                "set_placement", {"pos_mm": list(entry["pos_mm"]),
                                  "quat": list(entry["quat"])}))
        self.project.end_macro()

    # -- runner wiring -----------------------------------------------------------
    def _wire_runner(self):
        self.runner.line.connect(self.console.append_line)
        self.runner.progress.connect(self._on_progress)
        self.runner.started.connect(self._on_started)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_error)
        # P1 chunked-run contract affordances (Results pane header buttons)
        self.results.resumeRequested.connect(self._on_resume_run)
        self.results.extendRequested.connect(self._on_extend_run)
        self._wire_optimizer()

    def _wire_optimizer(self):
        """Optimizer controller <-> pane: progress feeds BOTH the pane's
        convergence plot and the shared stage-chip/status handler (the
        'optimize' chip is in STAGE_ORDER)."""
        self.optimizer_ctl.line.connect(self.console.append_line)
        self.optimizer_ctl.line.connect(self.optimize_pane.on_line)
        self.optimizer_ctl.progress.connect(self.optimize_pane.on_progress)
        self.optimizer_ctl.progress.connect(self._on_progress)
        self.optimizer_ctl.started.connect(self.optimize_pane.on_started)
        self.optimizer_ctl.finished.connect(self._on_optimize_finished)
        self.optimizer_ctl.error.connect(self._on_error)
        self.optimize_pane.runRequested.connect(self._on_run_optimize)
        self.optimize_pane.stopRequested.connect(self.optimizer_ctl.stop)
        self.optimize_pane.applyRequested.connect(self._on_apply_optimum)
        # a fresh scene has no optimum yet -> disable Apply
        self.project.sceneLoaded.connect(self.optimize_pane.reset_best)
        self._wire_tolerance()

    def _wire_tolerance(self):
        """Tolerance controller <-> pane: progress feeds BOTH the pane's
        result plots and the shared stage-chip/status handler (the
        'tolerance' chip is in STAGE_ORDER)."""
        self.tolerance_ctl.line.connect(self.console.append_line)
        self.tolerance_ctl.line.connect(self.tolerance_pane.on_line)
        self.tolerance_ctl.progress.connect(
            self.tolerance_pane.on_progress)
        self.tolerance_ctl.progress.connect(self._on_progress)
        self.tolerance_ctl.started.connect(self.tolerance_pane.on_started)
        self.tolerance_ctl.finished.connect(self._on_tolerance_finished)
        self.tolerance_ctl.error.connect(self._on_error)
        self.tolerance_pane.runRequested.connect(self._on_run_tolerance)
        self.tolerance_pane.stopRequested.connect(self.tolerance_ctl.stop)

    def _on_show_optimize(self):
        self.central_tabs.setCurrentWidget(self.optimize_pane)

    def _on_show_tolerance(self):
        self.central_tabs.setCurrentWidget(self.tolerance_pane)

    def _on_run_tolerance(self):
        """Tolerance pane Run button: launch scripts/tolerance.py on the
        open model under the optics-env python. Dialog-free (offscreen-
        test discipline): problems land in the status bar."""
        if not self.model_path:
            self.statusBar().showMessage(
                "Open a model before tolerancing", 8000)
            return False
        if self.tolerance_ctl.is_running():
            self.statusBar().showMessage(
                "A tolerance study is already running", 8000)
            return False
        try:
            config = self.tolerance_pane.config()
        except ValueError as exc:
            self.statusBar().showMessage("Tolerance: %s" % exc, 8000)
            return False
        if not config["tolerance"]:
            self.statusBar().showMessage(
                "Tolerance: add at least one tolerance row", 8000)
            return False
        if not config["operand"]:
            self.statusBar().showMessage(
                "Tolerance: add at least one operand row", 8000)
            return False
        if self.project.is_open():
            # a run implies persistence: the scene reopens with this
            # config pre-populated even if the run is never repeated
            try:
                self.project.set_tolerance_config(config)
            except Exception:
                pass
        args = ToleranceController.build_args(config)
        if not self.tolerance_ctl.start(self.model_path, args,
                                        extra_env=self._run_env()):
            self.statusBar().showMessage(
                "Could not start the tolerance study", 8000)
            return False
        self.stage_chips["tolerance"].setStyleSheet(
            self._chip_style(_CHIP_COLORS["running"]))
        self.statusBar().showMessage("Tolerance study started")
        return True

    def _on_tolerance_finished(self, exit_code):
        self.tolerance_pane.on_finished(exit_code)
        if exit_code == 0:
            self.statusBar().showMessage("Tolerance study finished", 5000)
        else:
            self.statusBar().showMessage(
                "Tolerance study exited with code %d (see console)"
                % exit_code, 8000)
            self.stage_chips["tolerance"].setStyleSheet(
                self._chip_style(_CHIP_COLORS["failed"]))

    def _on_run_optimize(self):
        """Optimize pane Run button: launch scripts/optimize.py on the
        open model under the optics-env python. Dialog-free (offscreen-
        test discipline): problems land in the status bar."""
        if not self.model_path:
            self.statusBar().showMessage(
                "Open a model before optimizing", 8000)
            return False
        if self.optimizer_ctl.is_running():
            self.statusBar().showMessage(
                "An optimization is already running", 8000)
            return False
        try:
            config = self.optimize_pane.config()
        except ValueError as exc:
            self.statusBar().showMessage("Optimize: %s" % exc, 8000)
            return False
        if not config["var"]:
            self.statusBar().showMessage(
                "Optimize: add at least one variable row", 8000)
            return False
        if not config["operand"]:
            self.statusBar().showMessage(
                "Optimize: add at least one operand row", 8000)
            return False
        if self.project.is_open():
            # a run implies persistence: the scene reopens with this
            # config pre-populated even if the run is never repeated
            try:
                self.project.set_optimize_config(config)
            except Exception:
                pass
        args = OptimizeController.build_args(config)
        if not self.optimizer_ctl.start(self.model_path, args,
                                        extra_env=self._run_env()):
            self.statusBar().showMessage(
                "Could not start the optimizer", 8000)
            return False
        self.stage_chips["optimize"].setStyleSheet(
            self._chip_style(_CHIP_COLORS["running"]))
        self.statusBar().showMessage("Optimization started")
        return True

    def _on_apply_optimum(self):
        """Apply optimum: write the optimizer's best-found parameters back
        into the scene (one undo step). Dialog-free — problems land in the
        status bar (offscreen-test discipline)."""
        if not self.project.is_open():
            self.statusBar().showMessage(
                "Open a model before applying an optimum", 8000)
            return False
        if self.optimizer_ctl.is_running():
            self.statusBar().showMessage(
                "Wait for the optimization to finish before applying", 8000)
            return False
        params = self.optimize_pane.best_params()
        if not params:
            self.statusBar().showMessage("No optimum to apply yet", 8000)
            return False
        try:
            self.project.apply_parameter_values(params)
        except Exception as exc:
            self.statusBar().showMessage("Apply optimum: %s" % exc, 8000)
            return False
        self.optimize_pane.set_start_values(params)
        summary = "  ".join("%s=%.6g" % (k, v)
                            for k, v in sorted(params.items()))
        self.statusBar().showMessage("Applied optimum:  %s" % summary, 8000)
        return True

    def _on_optimize_finished(self, exit_code):
        self.optimize_pane.on_finished(exit_code)
        if exit_code == 0:
            self.statusBar().showMessage("Optimization finished", 5000)
        else:
            self.statusBar().showMessage(
                "Optimization exited with code %d (see console)"
                % exit_code, 8000)
            self.stage_chips["optimize"].setStyleSheet(
                self._chip_style(_CHIP_COLORS["failed"]))

    def _reset_run_indicators(self):
        for chip in self.stage_chips.values():
            chip.setStyleSheet(self._chip_style(_CHIP_DEFAULT))
        self.progress_bar.setValue(0)

    def _on_started(self):
        self._reset_run_indicators()
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
        # a Resume/Extend launch targets an EXISTING case_dir that may not
        # match the current ConfigMatrix preset/tag (_current_case_dir's
        # normal derivation) -- prefer it when set (see _on_resume_run/
        # _on_extend_run), one-shot.
        case_dir = self._resume_extend_case_dir or self._current_case_dir()
        self._resume_extend_case_dir = None
        if case_dir and os.path.isdir(case_dir):
            self.results.load_case(case_dir)
            self.central_tabs.setCurrentWidget(self.results)
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
                    simparams=self.config_matrix.values(),
                    prescription=self._prescription_or_none())
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

    # -- P1 chunked-run contract: resume/extend --------------------------------
    def _already_running_notice(self):
        """Shared "a pipeline run is already in progress" notice -- a
        QMessageBox when there's a user to click it, the status bar
        otherwise (CLAUDE.md: never show an unguarded modal in a pane
        code path; a hidden window in an offscreen test must never block
        on .exec())."""
        if self.isVisible():
            QMessageBox.warning(
                self, "Pipeline already running",
                "A pipeline run is already in progress.")
        else:
            self.statusBar().showMessage(
                "A pipeline run is already in progress.", 8000)

    def _on_resume_run(self, case_dir):
        """Results pane "Resume run" button: relaunch the pipeline
        (trace,post,viz -- extract is skipped, the case's geometry is
        untouched) with --resume against THIS case's own checkpoint. Same
        dialog-free, status-bar-only error reporting as the rest of the
        runner wiring (no unguarded modal -- CLAUDE.md)."""
        if not case_dir:
            return
        if self.runner.is_running():
            self._already_running_notice()
            return
        if not self.model_path:
            self.statusBar().showMessage(
                "Open the model this case belongs to before resuming",
                8000)
            return
        self._resume_extend_case_dir = str(case_dir)
        if not self.runner.start_resume(self.model_path, case_dir,
                                        extra_env=self._run_env()):
            self._resume_extend_case_dir = None
            self._already_running_notice()
            return
        self.statusBar().showMessage("Resuming interrupted run…")

    def _on_extend_run(self, case_dir):
        """Results pane "Extend run..." button: build the extend context
        (current rays/measured spr from checkpointinfo.extend_state) and
        show RunDialog in extend mode; on accept, launch --extend to the
        chosen new total. The dialog itself is isVisible-guarded exactly
        like _confirm_run_dialog -- an "Extend" action needs the user to
        pick a new ray count, so a hidden window (offscreen tests, no
        user to ask) just declines rather than exec'ing a modal nobody
        can close."""
        if not case_dir:
            return
        if self.runner.is_running():
            self._already_running_notice()
            return
        if not self.model_path:
            self.statusBar().showMessage(
                "Open the model this case belongs to before extending",
                8000)
            return
        state = checkpointinfo.extend_state(case_dir)
        if state is None:
            self.statusBar().showMessage(
                "This case is not an extendable completed C-engine run",
                8000)
            return
        if not self.isVisible():
            return
        case_json = checkpointinfo.read_case(case_dir) or {}
        opts = case_json.get("options") or {}
        n_coh = 1 if (case_json.get("gather") or {}) else 0
        # case_dir is results/<model_stem>/<case>/ -- model_stem is the
        # PARENT directory name (matches common.estimate()'s model_stem
        # calibration key, "trace_rps_<eng>:<model_stem>" / "spr:<...>").
        model_stem = os.path.basename(
            os.path.dirname(str(case_dir).rstrip("/")))
        ctx = {
            "current_rays": state["current_rays"],
            "spr": state["spr"],
            "resolution": opts.get("resolution", 512),
            "nlambda": opts.get("nlambda", 5),
            "backend": "c",
            "n_coherent_sources": n_coh,
            "save_fields": bool(opts.get("save_fields")),
            "model_stem": model_stem,
        }
        run_params = {
            "resolution": ctx["resolution"],
            "nlambda": ctx["nlambda"],
            "backend": "c (chunked)",
            "model_stem": model_stem,
        }
        base_estimate = common.estimate(
            ctx["current_rays"], ctx["resolution"], ctx["nlambda"],
            ctx["n_coherent_sources"], "c", save_fields=ctx["save_fields"],
            model_stem=model_stem)
        calibrated = common.estimate_is_calibrated(
            "c", model_stem=model_stem)
        dialog = RunDialog(run_params, base_estimate, calibrated=calibrated,
                           extend_ctx=ctx, parent=self)
        self._last_extend_dialog = dialog     # test hook
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_rays = dialog.extend_target_rays()
        self._resume_extend_case_dir = str(case_dir)
        if not self.runner.start_extend(self.model_path, case_dir, new_rays,
                                        extra_env=self._run_env()):
            self._resume_extend_case_dir = None
            self._already_running_notice()
            return
        self.statusBar().showMessage(
            "Extending run to %d rays…" % new_rays)

    # -- open flows -----------------------------------------------------------
    def _maybe_save_changes(self, verb="continuing"):
        """Prompt Save / Discard / Cancel when the open model has unsaved
        changes. Returns False when the user cancels (the caller must
        abort). Hidden windows (offscreen tests, teardown) never block on
        the modal -- unsaved changes are then treated as discarded, same
        policy as closeEvent."""
        if not (self.project.is_open() and self.project.is_dirty()):
            return True
        if not self.isVisible():
            return True
        answer = QMessageBox.question(
            self, "Unsaved changes",
            "The model has unsaved changes. Save before %s?" % verb,
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            self._on_save()
            if self.project.is_dirty():   # save failed and was reported
                return False
        return True

    def _reset_session_views(self):
        """Clear every view artifact tied to the outgoing session: ray
        overlays (the old rays used to survive into a newly opened model
        and wreck the render), the preview chain/scheduler, the shared
        selection, the loaded results case (full widget wipe), the run
        indicators/console, a still-running pipeline, and the run config
        (a .MieWB re-applies its own simparams right after this)."""
        if self.raypreview.is_running():
            self.raypreview.cancel()
        if self.runner.is_running():
            self.runner.stop()
        if self.optimizer_ctl.is_running():
            self.optimizer_ctl.stop()
        if self.tolerance_ctl.is_running():
            self.tolerance_ctl.stop()
        self.preview_scheduler.reset()
        self.scene3d.clear_rays()
        self.scene3d.set_rays_stale(False)
        self.inspector.clear_rays()
        self.inspector.set_rays_stale(False)
        self.selection.clear()
        self.inspector.set_body(self.project, None)
        self.element_editor.set_face_selection(None, set())
        self.results.clear_case()
        self._reset_run_indicators()
        self.console.clear()
        self.config_matrix.reset_to_defaults()
        self._pending_manifest = None
        self.compare_pane.clear()
        self.compare_dock.hide()

    def _clear_session_paths(self):
        self.model_path = None
        self.opened_path = None
        self.workspace = None
        self.miewb_path = None
        self.miesim_out = None
        self.library_manager.set_project_root(None)

    def _on_close_model(self):
        if not self.project.is_open():
            return
        if not self._maybe_save_changes("closing"):
            return
        self._reset_session_views()
        self.project.close()          # emits sceneLoaded -> views empty
        self._clear_session_paths()
        self._update_window_title()
        self.statusBar().showMessage("Model closed", 4000)

    def _on_revert(self):
        if not self.project.is_open():
            return
        if self.isVisible():
            answer = QMessageBox.question(
                self, "Revert to Saved",
                "Discard ALL unsaved changes and restore %s to its last "
                "saved state?" % os.path.basename(self.model_path or ""),
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._reset_session_views()
        try:
            self.project.revert()
        except ProjectError as exc:
            QMessageBox.critical(self, "Revert failed", str(exc))
            return
        self._update_window_title()
        self.statusBar().showMessage("Reverted to last saved state", 5000)

    def _on_open(self):
        if not self._maybe_save_changes("opening another model"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open optical model", "",
            "Optical models (*.FCStd *.MieWB *.MieSim);;All files (*)")
        if path:
            self.open_model(path)

    def _on_new(self):
        if not self._maybe_save_changes("creating a new simulation"):
            return
        path, selected = QFileDialog.getSaveFileName(
            self, "New Simulation", "",
            "Workbench archive (*.MieWB);;FreeCAD model (*.FCStd)")
        if not path:
            return
        if not path.lower().endswith((".miewb", ".fcstd")):
            path += ".FCStd" if "FCStd" in selected else ".MieWB"
        self._reset_session_views()
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
                              simparams=self.config_matrix.values(),
                              prescription=self._prescription_or_none())

    def open_model(self, path):
        # the outgoing session's ray overlays/selection/results must never
        # leak into the incoming one (old rays used to stay on screen and
        # break the new scene's render)
        self._reset_session_views()
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

    def _prescription_or_none(self):
        """The prescription doc to embed when packing a .MieWB (engine3 Sec
        3, P5), or None. Never raises -- a prescription is an optional,
        additive member, so any failure just packs a prescription-free
        workbench exactly as before."""
        try:
            if self.project is None:
                return None
            return self.project.build_prescription()
        except Exception:
            return None

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
        self.central_tabs.setCurrentWidget(self.results)
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
        self.central_tabs.setCurrentWidget(self.results)
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
                    simparams=self.config_matrix.values(),
                    prescription=self._prescription_or_none())
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
                    simparams=self.config_matrix.values(),
                    prescription=self._prescription_or_none())
                # retarget the session: the existing workspace stays the
                # live unpacked session; File->Save now repacks into the
                # new archive
                self.miewb_path = path
                self.opened_path = path
                self.miesim_out = None
            else:
                self.project.save_as(path)
                self.model_path = path
                self.opened_path = path
                self.workspace = None
                self.miewb_path = None
                self.miesim_out = None
                self.library_manager.set_project_root(None)
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
        tmpdir = None
        try:
            src = self.model_path
            if self.project.is_open():
                # pack the CURRENT state without silently saving the
                # original: export a copy and pack that
                tmpdir = tempfile.mkdtemp(prefix="miewb-export-")
                src = os.path.join(tmpdir,
                                   os.path.basename(self.model_path))
                self.project.export_fcstd(src)
            miewb_tool.pack_miewb(
                src, wb_path,
                optprops_dir=self._workspace_optprops(),
                simparams=self.config_matrix.values(),
                prescription=self._prescription_or_none())
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
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

    def _export_fcstd(self, path):
        """Dialog-free core (tests call this directly): write a standalone
        .FCStd copy of the current document via the worker. Returns the
        written path."""
        return self.project.export_fcstd(path)

    def _on_export_fcstd(self):
        if not self.project.is_open():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export FCStd", "", "FreeCAD model (*.FCStd)")
        if not path:
            return
        if not path.lower().endswith(".fcstd"):
            path += ".FCStd"
        try:
            self._export_fcstd(path)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage("Exported %s" % path, 6000)

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

    # class-level default for the dialog-free test seam: hidden windows
    # (offscreen tests) resolve the Save&Run prompt to this instead of a
    # modal; tests flip it to False to exercise the cancel path.
    _save_before_run_hook = True

    def _confirm_save_before_run(self):
        """Run gate: the pipeline traces the on-disk model, so unsaved
        changes must be saved first. Never saves silently — a visible
        window prompts Save&Run/Cancel. Returns True to proceed."""
        if not (self.project.is_open() and self.project.is_dirty()):
            return True
        if self.isVisible():
            answer = QMessageBox.question(
                self, "Unsaved changes",
                "The model has unsaved changes, and the simulation runs "
                "on the last saved file.\n\nSave and run?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save)
            if answer != QMessageBox.StandardButton.Save:
                return False
        elif not self._save_before_run_hook:
            return False
        self._on_save()   # reports its own failures; dirty stays set then
        return not self.project.is_dirty()

    def _preflight(self):
        """Validate before launching; errors block, warnings ask. Unsaved
        changes gate through _confirm_save_before_run (never a silent
        save)."""
        if not self._require_model():
            return None
        if not self._confirm_save_before_run():
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

    def _merged_run_config(self):
        """config_matrix.values() with the Variables pane's enabled sweep
        superseding the raw config-matrix var/min/max/n fields."""
        config = self.config_matrix.values()
        vp = self.variables_pane
        if vp is not None and vp.has_enabled_sweep():
            varnames, mins, maxs, ns = vp.sweep_spec()
            config["var"] = list(varnames)
            config["min"] = list(mins)
            config["max"] = list(maxs)
            config["n"] = list(ns)
            config["sweep_mode"] = vp.sweep_mode
        return config

    def _model_stem(self):
        if not self.model_path:
            return None
        return os.path.splitext(os.path.basename(self.model_path))[0]

    def _run_estimate(self, params=None):
        """(params, estimate_dict, calibrated) for the current
        config-matrix settings -- the ONE place that calls
        common.estimate()/estimate_is_calibrated(), so the Estimate
        button (info-only RunDialog) and the pre-run RunDialog can never
        show different numbers. params defaults to
        config_matrix.estimate_params(); model_stem is filled in from the
        open model when the caller didn't already set one (so per-scene
        calibration -- trace_rps_<eng>:<stem> / spr:<stem> -- applies)."""
        params = dict(params or self.config_matrix.estimate_params())
        params.setdefault("model_stem", self._model_stem())
        model_stem = params.get("model_stem")
        result = common.estimate(
            params["rays"], params["resolution"], params["nlambda"],
            params["n_coherent_sources"], params["backend"],
            n_detectors=params["n_detectors"],
            save_fields=params["save_fields"],
            n_pol_strata=params["n_pol_strata"], model_stem=model_stem)
        calibrated = common.estimate_is_calibrated(
            params["backend"], model_stem=model_stem)
        return params, result, calibrated

    def _single_run_estimate_s(self):
        _, result, _ = self._run_estimate()
        return result["total_s"]

    @staticmethod
    def _config_run_count(config):
        varnames = config.get("var") or []
        if not varnames:
            return 1
        mins = config.get("min") or []
        maxs = config.get("max") or []
        ns = config.get("n") or []
        if len({len(varnames), len(mins), len(maxs), len(ns)}) != 1:
            return 1     # malformed; write_sweep_manifest will raise
        try:
            value_lists = [common.sweep_values(float(a), float(b), int(c))
                           for a, b, c in zip(mins, maxs, ns)]
            combos = common.sweep_combos(
                value_lists, config.get("sweep_mode") or "product")
        except Exception:
            return 1
        return len(combos)

    def _sweep_summary(self, config):
        runs = self._config_run_count(config)
        per_run_s = self._single_run_estimate_s()
        total_s = runs * per_run_s
        return {
            "runs": runs,
            "per_run_s": per_run_s,
            "total_s": total_s,
            "text": "%d run%s x %s = %s" % (
                runs, "" if runs == 1 else "s",
                common.fmt_duration(per_run_s),
                common.fmt_duration(total_s)),
        }

    def _confirm_run_dialog(self):
        """P1 per-run accuracy-vs-time confirmation (owner requirement:
        "always ask per run"). isVisible-guarded like _confirm_sweep --
        a hidden window (offscreen tests) always proceeds without
        blocking on the modal. Honors the in-session "don't ask again"
        skip (settings key run_dialog_skip_session, reset every launch
        in __init__ -- see the comment there)."""
        if self.settings.get_bool("run_dialog_skip_session", False):
            return True
        if not self.isVisible():
            return True
        params, result, calibrated = self._run_estimate()
        dialog = RunDialog(params, result, calibrated=calibrated,
                           info_only=False, parent=self)
        self._last_run_dialog = dialog     # test hook
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        if dialog.skip_requested():
            self.settings.set_bool("run_dialog_skip_session", True)
        return True

    def _confirm_sweep(self, summary):
        """Dialog-free (tests call directly): confirm a multi-variant
        launch. Hidden windows default to True so offscreen tests never
        block on the modal."""
        if not self.isVisible():
            return True
        answer = QMessageBox.question(
            self, "Confirm sweep",
            "This launches %d simulation runs.\n\n%s\n\nProceed?"
            % (summary["runs"], summary["text"]))
        return answer == QMessageBox.StandardButton.Yes

    def _run_pipeline(self, dry_run=False):
        """Save + validate + merge the sweep, confirm the run (accuracy-
        vs-time RunDialog, then a multi-variant sweep-count confirm if
        applicable), write the sweep manifest, then launch. Dialog-free
        except those two confirmations (both isVisible-guarded, and the
        RunDialog is skipped outright on dry_run -- it never actually
        traces anything). Returns True on launch."""
        if self._preflight() is None:     # save + validate (warns on error)
            return False
        if not dry_run and not self._confirm_run_dialog():
            return False
        config = self._merged_run_config()
        args = RunController.build_args(config)
        if self.workspace:
            args += ["--optical-properties", self._workspace_optprops()]
        if dry_run and "--dry-run" not in args:
            args += ["--dry-run"]

        env = self._run_env()
        results_root = env.get("MIEWB_RESULTS_DIR") if env else None
        self._pending_manifest = None
        if self._config_run_count(config) > 1:
            if not self._confirm_sweep(self._sweep_summary(config)):
                return False
            # a dry run produces no post/viz, so there is nothing to
            # Compare — confirm the launch but skip the manifest/handoff.
            if not dry_run:
                try:
                    manifest = RunController.write_sweep_manifest(
                        self.model_path, config, results_root=results_root)
                    self._pending_manifest = (
                        str(manifest) if manifest else None)
                except Exception as exc:
                    self.console.append_line(
                        "[sweep] manifest write failed: %s" % exc)

        if not self.runner.start(self.model_path, args, extra_env=env):
            QMessageBox.warning(
                self, "Pipeline already running",
                "A pipeline run is already in progress.")
            return False
        return True

    def _maybe_run_compare(self, exit_code):
        """After a successful multi-variant run, hand the stashed sweep
        manifest to the Compare pane and reveal its dock."""
        manifest = self._pending_manifest
        self._pending_manifest = None
        if exit_code != 0 or not manifest:
            return
        if self.compare_pane.run_compare(manifest_path=manifest):
            self.compare_dock.show()
            self.compare_dock.raise_()

    def _config_matrix_dialog(self, title, ok_label):
        """Shared modal wrapper around the (single, shared) ConfigMatrix
        widget — used by both Run Pipeline… and Simulation Settings…, so
        the two views can never show different values."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        layout.addWidget(self.config_matrix)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(ok_label)
        layout.addWidget(buttons)
        dialog.resize(720, 640)
        return dialog, buttons

    def persist_simparams(self):
        """Write the current simulation settings into the open project:
        the workspace's simparams.json AND the .MieWB archive (repacked
        with the last-saved model, exactly like File->Save does). No-op
        for bare .FCStd sessions (there is no project archive to carry
        them). Best-effort: failures land in the status bar, never a
        modal (offscreen-test discipline). Returns True when persisted."""
        if not self.miewb_path:
            return False
        try:
            values = self.config_matrix.values()
            if self.workspace:
                simparams_path = os.path.join(self.workspace,
                                              "simparams.json")
                with open(simparams_path, "w") as fh:
                    json.dump(values, fh, indent=1)
            miewb_tool.pack_miewb(
                self.model_path, self.miewb_path,
                optprops_dir=self._workspace_optprops(),
                simparams=values,
                prescription=self._prescription_or_none())
            self.statusBar().showMessage(
                "Simulation settings saved into %s" % self.miewb_path,
                5000)
            return True
        except Exception as exc:
            self.statusBar().showMessage(
                "Could not save simulation settings: %s" % exc, 8000)
            return False

    def _on_simulation_settings_dialog(self):
        """Simulation menu > Simulation Settings…: view/edit the settings
        WITHOUT running. Tabbed: "Simulation" is the shared ConfigMatrix
        widget (reparenting/OK/apply semantics UNCHANGED from before —
        deliberately NOT routed through _config_matrix_dialog, which
        Run Pipeline… still uses standalone); "Ray Preview" is the
        persistent per-document preview-pattern editor (WP2). OK persists
        simparams into the open .MieWB exactly as before, and (if the
        pattern changed) the preview spec into the project + QSettings;
        Cancel leaves both widgets' state as-is (shared instances, so
        edits made before Cancel remain visible next time but are not
        written anywhere)."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Simulation Settings")
        layout = QVBoxLayout(dialog)

        tabs = QTabWidget()
        tabs.addTab(self.config_matrix, "Simulation")
        tabs.addTab(self.preview_config, "Ray Preview")
        layout.addWidget(tabs)

        original_spec = self._preview_pattern_spec()
        try:
            self.preview_config.set_spec(original_spec)
        except ValueError:
            pass

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        layout.addWidget(buttons)
        dialog.resize(720, 640)

        def on_ok():
            if self.miewb_path:
                self.persist_simparams()
            else:
                self.statusBar().showMessage(
                    "Settings applied for this session (open/save a "
                    ".MieWB to store them with the project)", 8000)
            spec = self.preview_config.spec()
            if spec != original_spec and self.project.is_open():
                try:
                    self.project.set_preview_config({"spec": spec})
                except ProjectError as exc:
                    self.statusBar().showMessage(
                        "Could not save ray-preview pattern: %s" % exc,
                        8000)
            self.settings.set("preview_pattern_spec", spec)
            dialog.accept()

        buttons.accepted.connect(on_ok)
        buttons.rejected.connect(dialog.reject)
        dialog.exec()

    def _on_run_pipeline_dialog(self):
        dialog, buttons = self._config_matrix_dialog("Run Pipeline", "Run")

        def on_run():
            if self._run_pipeline():
                # a run's settings should stick with the project too —
                # persist them so the .MieWB always reflects what was run
                if self.miewb_path:
                    self.persist_simparams()
                dialog.accept()

        buttons.accepted.connect(on_run)
        buttons.rejected.connect(dialog.reject)
        dialog.exec()

    def _on_estimate(self):
        self._show_estimate(self.config_matrix.estimate_params())

    def _show_estimate(self, params):
        """The ConfigMatrix "Estimate runtime" button: an info-only
        RunDialog (Close button only, no Run/Cancel/skip checkbox) built
        from the SAME common.estimate() call _confirm_run_dialog uses, so
        the two entry points can never disagree. isVisible-guarded so
        offscreen tests can call this directly without blocking."""
        params, result, calibrated = self._run_estimate(params)
        dialog = RunDialog(params, result, calibrated=calibrated,
                           info_only=True, parent=self)
        self._last_estimate_dialog = dialog     # test hook
        if self.isVisible():
            dialog.exec()
        return dialog

    def _on_dry_run(self):
        self._run_pipeline(dry_run=True)

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
        self.shutdown_resources()
        try:
            self.project.shutdown()
        except Exception:
            pass
        super().closeEvent(event)

    def shutdown_resources(self):
        """Idempotent native-resource teardown: every QTimer, secondary
        top-level window and VTK interactor must be released here or the
        interpreter hangs after app.exec() returns (also reached via
        aboutToQuit for quit paths that bypass closeEvent)."""
        if getattr(self, "_resources_shut_down", False):
            return
        self._resources_shut_down = True
        try:
            if self.raypreview.is_running():
                self.raypreview.cancel()
        except Exception:
            pass
        try:
            if self.optimizer_ctl.is_running():
                self.optimizer_ctl.stop()
        except Exception:
            pass
        try:
            if self.tolerance_ctl.is_running():
                self.tolerance_ctl.stop()
        except Exception:
            pass
        try:
            self.preview_scheduler.reset()
        except Exception:
            pass
        try:
            self.anim_controller.stop()
        except Exception:
            pass
        try:
            self.results.stop_monitoring()
        except Exception:
            pass
        if self._prop_editor_window is not None:
            try:
                self._prop_editor_window.close()
                self._prop_editor_window.deleteLater()
            except Exception:
                pass
            self._prop_editor_window = None
        for pane in (self.scene3d, self.inspector):
            try:
                pane.shutdown()
            except Exception:
                pass
