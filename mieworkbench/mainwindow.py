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
from PySide6.QtGui import QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMenu,
    QMessageBox, QProgressBar, QStyle, QToolButton, QVBoxLayout, QWidget,
)

from .core import paraview_launcher
from .core.beadanim import (AnimationController, format_sim_time,
                            precompute_segments)
from .core.librarymgr import LibraryManager
from .core.previewscheduler import PreviewScheduler
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

# ray extinction modes, in the toolbar combo's index order
_RAY_DIM_MODES = ("off", "linear", "sqrt")

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
        self.preview_scheduler = PreviewScheduler(parent=self)
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
        self._init_animation()
        self._build_menus()
        self._build_toolbar()
        self._build_animation_toolbar()
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
            "View ▸ Ray Dimming; applies live to loaded rays.")
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
            # a whole-scene fan passes through the inspected element too:
            # refresh a checked inspector overlay rather than leaving it
            # greyed-out stale
            if self.inspector.rays_button.isChecked():
                self.inspector.load_rays_vtp(vtp_path)
        self.statusBar().showMessage("Ray preview ready", 5000)
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

    def _on_auto_preview_wanted(self):
        if not self.project.is_open():
            return
        if self.runner.is_running():
            return   # never compete with a real pipeline run; its own
                     # rays load when it completes
        if self.raypreview.is_running():
            # a manual preview is in flight; queue one more behind it
            self.preview_scheduler.notify_busy(True)
            self.preview_scheduler.notify_change()
            return
        self._preview_target = "scene"
        started = self.raypreview.start(
            self.project, self._preview_workspace(),
            pattern="fan:n=5",
            optical_properties=self._workspace_optprops())
        if started:
            self.preview_scheduler.notify_busy(True)
            self.statusBar().showMessage("Auto-updating ray preview…",
                                         3000)

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
        if checked:
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

        # auto ray-preview: any optics-affecting edit greys the overlays
        # and (debounced) re-traces the preview fan
        self.project.opticsChanged.connect(self._on_optics_changed)
        self.preview_scheduler.previewWanted.connect(
            self._on_auto_preview_wanted)
        self.raypreview.finished.connect(
            lambda _path: self.preview_scheduler.notify_run_finished())
        self.raypreview.failed.connect(
            lambda _msg: self.preview_scheduler.notify_run_failed())

    def _on_scene_loaded(self):
        has_doc = self.project.is_open()
        self.save_action.setEnabled(has_doc)
        self.save_as_action.setEnabled(has_doc)
        self.revert_action.setEnabled(has_doc)
        self.close_action.setEnabled(has_doc)
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
                              simparams=self.config_matrix.values())

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
