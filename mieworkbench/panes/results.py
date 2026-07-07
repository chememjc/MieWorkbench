"""Results pane: browse a completed (or in-progress) case.

Works on a case directory (results/<model>/<case>/ inside a workspace or
repo tree). Shows report.json headline numbers, the audit outcome, and
thumbnail galleries of images/, spectra/, plots/, viz/; "Open in
ParaView" hands the .vtp data to interactive ParaView.

MONITOR MODE (read-only view of a RUNNING case): when the case is locked
by a live process, a QTimer polls progress.json + the images as they
appear; editing/rerun affordances are the main window's job to disable -
this pane only ever reads."""

import json
import os

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import sys
_SCRIPTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import common  # noqa: E402

from ..core import paraview_launcher

_THUMB_W = 320


class _Gallery(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._host = QWidget()
        self._grid = QGridLayout(self._host)
        self.setWidget(self._host)
        self._shown = {}

    def show_images(self, paths):
        changed = False
        for i, path in enumerate(paths):
            mtime = os.path.getmtime(path) if os.path.exists(path) else 0
            if self._shown.get(path) == mtime:
                continue
            changed = True
        if not changed and len(paths) == len(self._shown):
            return
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._shown = {}
        for i, path in enumerate(sorted(paths)):
            box = QVBoxLayout()
            label = QLabel()
            pm = QPixmap(path)
            if not pm.isNull():
                label.setPixmap(pm.scaledToWidth(
                    _THUMB_W, Qt.SmoothTransformation))
            label.setToolTip(path)
            cap = QLabel(os.path.basename(path))
            cap.setStyleSheet("color: gray; font-size: 10px;")
            cell = QWidget()
            box.addWidget(label)
            box.addWidget(cap)
            cell.setLayout(box)
            self._grid.addWidget(cell, i // 3, i % 3)
            self._shown[path] = os.path.getmtime(path) \
                if os.path.exists(path) else 0


class ResultsPane(QWidget):
    statusChanged = Signal(str)     # case status string for the title bar

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.case_dir = None
        self._monitor = QTimer(self)
        self._monitor.setInterval(1000)
        self._monitor.timeout.connect(self.refresh)

        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        self.title = QLabel("No results loaded")
        self.title.setStyleSheet("font-weight: bold;")
        head.addWidget(self.title, 1)
        self.pv_btn = QPushButton("Open in ParaView")
        self.pv_btn.setToolTip("Launch interactive ParaView on this "
                               "case's rays/detector .vtp data")
        self.pv_btn.clicked.connect(self._open_paraview)
        self.pv_btn.setEnabled(False)
        head.addWidget(self.pv_btn)
        lay.addLayout(head)

        self.tabs = QTabWidget()
        self.summary = QTableWidget(0, 5)
        self.summary.setHorizontalHeaderLabels(
            ["Detector", "Power [mW]", "Peak [W/m²]", "Pixel [µm]",
             "Visibility"])
        self.summary.horizontalHeader().setStretchLastSection(True)
        self.summary.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabs.addTab(self.summary, "Summary")
        self.galleries = {}
        for name in ("images", "spectra", "plots", "viz"):
            g = _Gallery()
            self.galleries[name] = g
            self.tabs.addTab(g, name.capitalize())
        lay.addWidget(self.tabs)
        self.audit = QLabel("")
        lay.addWidget(self.audit)

    # -- loading -------------------------------------------------------------
    def load_case(self, case_dir, monitor=False):
        self.case_dir = str(case_dir)
        self._monitor.stop()
        if monitor:
            self._monitor.start()
        self.refresh()

    def stop_monitoring(self):
        self._monitor.stop()

    def refresh(self):
        if not self.case_dir or not os.path.isdir(self.case_dir):
            return
        status = common.read_case_status(
            os.path.join(self.case_dir, "case.json"))
        progress = ""
        ppath = os.path.join(self.case_dir, "progress.json")
        if self._monitor.isActive() and os.path.exists(ppath):
            try:
                with open(ppath) as fh:
                    ev = json.load(fh)
                frac = ev.get("frac")
                progress = "  [%s %s%s]" % (
                    ev.get("stage", "?"),
                    "" if frac is None else "%.0f%% " % (100 * frac),
                    ev.get("msg", ""))
            except (OSError, ValueError):
                pass
        base = os.path.basename(self.case_dir.rstrip("/"))
        self.title.setText("%s — %s%s" % (base, status, progress))
        self.statusChanged.emit(status)

        report_path = os.path.join(self.case_dir, "report.json")
        self.summary.setRowCount(0)
        if os.path.exists(report_path):
            try:
                with open(report_path) as fh:
                    report = json.load(fh)
            except (OSError, ValueError):
                report = {}
            dets = report.get("detectors", {})
            self.summary.setRowCount(len(dets))
            for row, (label, d) in enumerate(sorted(dets.items())):
                vals = [label,
                        "%.4g" % (d.get("total_power_W", 0.0) * 1e3),
                        "%.4g" % d.get("peak_irradiance_W_m2", 0.0),
                        "%.3g" % d.get("pixel_um", 0.0),
                        "%.3g" % d.get("profile_visibility", 0.0)
                        if "profile_visibility" in d else "—"]
                for col, val in enumerate(vals):
                    self.summary.setItem(row, col,
                                         QTableWidgetItem(str(val)))
            closure = report.get("closure_ok")
            self.audit.setText(
                "energy closure: %s"
                % ("OK ✓" if closure else
                   ("FAILED ✗" if closure is not None else "n/a")))

        from glob import glob
        for name, gallery in self.galleries.items():
            gallery.show_images(
                glob(os.path.join(self.case_dir, name, "*.png")))
        self.pv_btn.setEnabled(
            bool(paraview_launcher.viz_files(self.case_dir)))

    def _open_paraview(self):
        ok, msg = paraview_launcher.launch(self.case_dir, self.settings)
        self.statusChanged.emit(("ParaView: " if ok else
                                 "ParaView failed: ") + msg)
