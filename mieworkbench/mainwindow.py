"""MainWindow - Zemax-inspired multi-pane dock shell for MieWorkbench.

Phase-1 shell only: every pane except the console is a labeled placeholder
(objectName set so a later phase can swap the placeholder for the real
widget without touching this file's layout code). The parts that ARE real
already are the ones this phase is actually about: the console pane, the
run-status strip (stage chips + overall progress bar), the graphical
configuration matrix, and RunController wiring - i.e. the whole
GUI -> QProcess -> @MIEWB progress loop end to end.

Layout (Qt dock areas):
    left,  top    -> Element Inspector   (inspector_host)
    left,  bottom -> Element Properties  (element_editor_host)
    right, top    -> Position/Orientation(transform_host)
    right, bottom -> Library             (library_host)
    bottom        -> stage chips + progress bar + ConsolePane
    central       -> Optical Train 3D View placeholder (scene3d_host)
"""

import os
import shlex
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "scripts")))
import common  # noqa: E402  (stdlib-only shared contract hub)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QDockWidget, QFileDialog, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QStyle, QVBoxLayout,
    QWidget,
)

from .core.runner import RunController, RUN_PIPELINE_SCRIPT
from .core.settings import Settings, SettingsDialog
from .panes.config_matrix import ConfigMatrix
from .panes.console import ConsolePane

STAGE_ORDER = ["extract", "trace", "post", "viz"]

_CHIP_COLORS = {
    "running": "#3b82f6",
    "completed": "#22c55e",
    "estimated": "#22c55e",
    "failed": "#ef4444",
}
_CHIP_DEFAULT = "#6b7280"


def _placeholder(title, object_name):
    """A simple dark, centered QLabel used for every not-yet-built pane."""
    widget = QWidget()
    widget.setObjectName(object_name)
    layout = QVBoxLayout(widget)
    label = QLabel(title)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet("color: #9ca3af; background-color: #111827;")
    label.setToolTip("%s - placeholder, wired up in a later phase" % title)
    layout.addWidget(label)
    widget.setStyleSheet("background-color: #111827;")
    return widget


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MieWorkbench")
        self.resize(1400, 900)

        self.model_path = None
        self.settings = Settings()
        self.runner = RunController(self.settings, self)
        self.config_matrix = ConfigMatrix()
        self.config_matrix.estimateRequested.connect(self._show_estimate)

        self._build_central()
        self.stage_chips = {}
        self._build_docks()
        self._build_menus()
        self._build_toolbar()
        self.statusBar().showMessage("Ready")
        self._wire_runner()
        self._update_window_title()

    # -- central --------------------------------------------------------------
    def _build_central(self):
        central = _placeholder("Optical Train 3D View", "scene3d_host")
        self.setCentralWidget(central)

    # -- docks ------------------------------------------------------------------
    def _build_docks(self):
        self.inspector_dock = self._add_dock(
            "Element Inspector", "inspector_dock",
            _placeholder("Element Inspector", "inspector_host"),
            Qt.DockWidgetArea.LeftDockWidgetArea)
        self.element_editor_dock = self._add_dock(
            "Element Properties", "element_editor_dock",
            _placeholder("Element Properties", "element_editor_host"),
            Qt.DockWidgetArea.LeftDockWidgetArea)
        self.splitDockWidget(self.inspector_dock, self.element_editor_dock,
                             Qt.Orientation.Vertical)

        self.transform_dock = self._add_dock(
            "Position / Orientation", "transform_dock",
            _placeholder("Position / Orientation", "transform_host"),
            Qt.DockWidgetArea.RightDockWidgetArea)
        self.library_dock = self._add_dock(
            "Library", "library_dock",
            _placeholder("Library", "library_host"),
            Qt.DockWidgetArea.RightDockWidgetArea)
        self.splitDockWidget(self.transform_dock, self.library_dock,
                             Qt.Orientation.Vertical)

        self.console_dock = self._add_dock(
            "Console", "console_dock", self._build_bottom_widget(),
            Qt.DockWidgetArea.BottomDockWidgetArea)
        self.resizeDocks([self.console_dock], [220], Qt.Orientation.Vertical)

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
        act.setToolTip("Open an optical model (.FCStd/.MieWB/.MieSim)")
        act.triggered.connect(self._on_open)

        act = file_menu.addAction("&Save")
        act.setToolTip("Save the open model (not yet implemented)")
        act.setEnabled(False)

        act = file_menu.addAction("Save &As…")
        act.setToolTip("Save the open model under a new name (not yet "
                       "implemented)")
        act.setEnabled(False)

        act = file_menu.addAction("&Export Run Script…")
        act.setToolTip("Write the current configuration as a standalone "
                       "shell script that reruns this pipeline command")
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

        self.stop_action = sim_menu.addAction("&Stop")
        self.stop_action.setToolTip("Stop the running pipeline")
        self.stop_action.triggered.connect(self.runner.stop)

        view_menu = menubar.addMenu("&View")
        for dock in (self.inspector_dock, self.element_editor_dock,
                    self.transform_dock, self.library_dock,
                    self.console_dock):
            view_menu.addAction(dock.toggleViewAction())

        help_menu = menubar.addMenu("&Help")
        act = help_menu.addAction("&About")
        act.setToolTip("About MieWorkbench")
        act.triggered.connect(self._on_about)

    def _build_toolbar(self):
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("main_toolbar")
        style = self.style()

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
        else:
            self.statusBar().showMessage(
                "Pipeline exited with code %d" % exit_code, 5000)

    def _on_error(self, message):
        self.statusBar().showMessage("Pipeline error: %s" % message, 5000)

    # -- File actions ---------------------------------------------------------
    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open optical model", "",
            "Optical models (*.FCStd *.MieWB *.MieSim)")
        if path:
            self.open_model(path)

    def open_model(self, path):
        """Store the model path for later use; real loading (geometry,
        FreeCAD session) arrives in a later phase."""
        self.model_path = path
        self._update_window_title()

    def _update_window_title(self):
        if self.model_path:
            self.setWindowTitle(
                "MieWorkbench — %s" % os.path.basename(self.model_path))
        else:
            self.setWindowTitle("MieWorkbench")

    def _on_export_script(self):
        if not self._require_model():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Run Script", "", "Shell scripts (*.sh)")
        if not path:
            return
        args = self.config_matrix.to_args()
        cmd = ["python3", str(RUN_PIPELINE_SCRIPT), "--models",
              self.model_path] + args
        with open(path, "w") as fh:
            fh.write("#!/bin/sh\n")
            fh.write(" ".join(shlex.quote(part) for part in cmd) + "\n")
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass
        self.statusBar().showMessage("Wrote run script to %s" % path, 5000)

    def _on_settings(self):
        dialog = SettingsDialog(self.settings, self)
        dialog.exec()

    def _on_about(self):
        QMessageBox.about(
            self, "About MieWorkbench",
            "MieWorkbench\nA PySide6 GUI shell for the optical "
            "ray-tracing pipeline.")

    # -- Simulation actions -----------------------------------------------------
    def _require_model(self):
        if not self.model_path:
            QMessageBox.warning(
                self, "No model open",
                "Open a model (.FCStd) before running the pipeline.")
            return False
        return True

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
            if not self._require_model():
                return
            args = self.config_matrix.to_args()
            if self.runner.start(self.model_path, args):
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
        if not self._require_model():
            return
        args = self.config_matrix.to_args()
        if "--dry-run" not in args:
            args = args + ["--dry-run"]
        if not self.runner.start(self.model_path, args):
            QMessageBox.warning(
                self, "Pipeline already running",
                "A pipeline run is already in progress.")
