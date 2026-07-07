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
    QDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
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


class _ClickableLabel(QLabel):
    """QLabel that calls a handler when clicked."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self._on_click = None

    def set_click_handler(self, callback):
        """Set callback to invoke on click; callback receives path."""
        self._on_click = callback

    def mousePressEvent(self, event):
        if self._on_click:
            self._on_click(self.path)
        super().mousePressEvent(event)


class _LightboxDialog(QDialog):
    """Non-modal dialog for viewing gallery images full-size with arrow-key cycling."""

    def __init__(self, paths, initial_index, parent=None):
        super().__init__(parent)
        self.paths = paths
        self.current_index = max(0, min(initial_index, len(paths) - 1)) \
            if paths else 0
        self._original_pixmap = None

        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        layout = QVBoxLayout(self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.image_label)
        layout.addWidget(self.scroll)
        self.setLayout(layout)

        self._load_image_at(self.current_index)
        self._scale_to_fit()

    def _load_image_at(self, index):
        """Load and display image at the given index."""
        if index < 0 or index >= len(self.paths):
            return
        self.current_index = index
        path = self.paths[self.current_index]
        pm = QPixmap(path)
        self.setWindowTitle(os.path.basename(path))
        if not pm.isNull():
            self._original_pixmap = pm
            # Scale if necessary
            display_pm = self._scale_pixmap(pm)
            self.image_label.setPixmap(display_pm)

    def _scale_pixmap(self, pixmap):
        """Scale pixmap to fit screen if larger than 85% of available space."""
        if pixmap.isNull():
            return pixmap

        screen = self.screen()
        if not screen:
            return pixmap

        geom = screen.availableGeometry()
        max_width = int(geom.width() * 0.85)
        max_height = int(geom.height() * 0.85)

        w = pixmap.width()
        h = pixmap.height()

        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            scaled_pm = pixmap.scaledToWidth(
                int(w * scale), Qt.SmoothTransformation)
            return scaled_pm
        return pixmap

    def _scale_to_fit(self):
        """Resize window to fit the displayed image."""
        if not self._original_pixmap or self._original_pixmap.isNull():
            return

        screen = self.screen()
        if not screen:
            return

        geom = screen.availableGeometry()
        max_width = int(geom.width() * 0.85)
        max_height = int(geom.height() * 0.85)

        w = self._original_pixmap.width()
        h = self._original_pixmap.height()

        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            self.resize(int(w * scale) + 20, int(h * scale) + 20)
        else:
            self.resize(w + 20, h + 20)

    def set_paths(self, paths):
        """Update the list of paths (called when gallery refreshes)."""
        old_path = self.paths[self.current_index] \
            if self.paths else None
        self.paths = paths

        if not self.paths:
            return

        if old_path and old_path in self.paths:
            self.current_index = self.paths.index(old_path)
        else:
            self.current_index = max(0, min(self.current_index,
                                            len(self.paths) - 1))

        self._load_image_at(self.current_index)

    def keyPressEvent(self, event):
        """Handle Esc, Left/Right arrow keys."""
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Left and self.paths:
            next_index = (self.current_index - 1) % len(self.paths)
            self._load_image_at(next_index)
        elif event.key() == Qt.Key_Right and self.paths:
            next_index = (self.current_index + 1) % len(self.paths)
            self._load_image_at(next_index)
        else:
            super().keyPressEvent(event)


class _Gallery(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._host = QWidget()
        self._grid = QGridLayout(self._host)
        self.setWidget(self._host)
        self._shown = {}
        self._paths = []  # sorted list of current paths
        self._lightbox = None  # reference to open lightbox dialog

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
        self._paths = sorted(paths)
        for i, path in enumerate(self._paths):
            box = QVBoxLayout()
            label = _ClickableLabel(path)
            label.set_click_handler(self._thumbnail_clicked)
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

        # Update lightbox if it's open and visible
        if self._lightbox is not None and self._lightbox.isVisible():
            self._lightbox.set_paths(self._paths)

    def _thumbnail_clicked(self, path):
        """Open lightbox for the clicked image."""
        if path not in self._paths:
            return
        index = self._paths.index(path)
        self._lightbox = _LightboxDialog(self._paths, index, parent=self)
        self._lightbox.show()


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

        # per-element energy accounting from the trace ledger
        self.power = QTableWidget(0, 5)
        self.power.setHorizontalHeaderLabels(
            ["Element", "In [mW]", "Out [mW]", "Absorbed [mW]",
             "Detected [mW]"])
        self.power.horizontalHeader().setStretchLastSection(True)
        self.power.setEditTriggers(QTableWidget.NoEditTriggers)
        self.power.setToolTip(
            "Boundary power flux per element (seed-averaged): In = power "
            "arriving from outside, Out = power leaving again, Absorbed = "
            "losses inside/at the element, Detected = power recorded by "
            "detector faces. In − Out ≈ Absorbed; small shortfalls are "
            "rays truncated by the generation/power caps.")
        self.tabs.addTab(self.power, "Power")

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
            self._populate_power(report.get("elements")
                                 or self._elements_from_audit())

        from glob import glob
        for name, gallery in self.galleries.items():
            gallery.show_images(
                glob(os.path.join(self.case_dir, name, "*.png")))
        self.pv_btn.setEnabled(
            bool(paraview_launcher.viz_files(self.case_dir)))

    def _populate_power(self, elements):
        elements = elements or {}
        self.power.setRowCount(len(elements))
        for row, (label, e) in enumerate(sorted(elements.items())):
            vals = [label,
                    "%.4g" % (e.get("power_in_W", 0.0) * 1e3),
                    "%.4g" % (e.get("power_out_W", 0.0) * 1e3),
                    "%.4g" % (e.get("absorbed_W", 0.0) * 1e3),
                    "%.4g" % (e.get("detected_W", 0.0) * 1e3)]
            for col, val in enumerate(vals):
                self.power.setItem(row, col, QTableWidgetItem(val))

    def _elements_from_audit(self):
        """Cases post-processed before report.json carried 'elements':
        mine audit.json directly with the same aggregation (stdlib-only
        common.element_power_table, shared with post_process)."""
        path = os.path.join(self.case_dir, "audit.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as fh:
                audit = json.load(fh)
            # detected rows are keyed by body NAME (face ids); map them to
            # labels via the model.json recorded in case.json, when present
            name_to_label = {}
            try:
                with open(os.path.join(self.case_dir, "case.json")) as fh:
                    mj = json.load(fh).get("options", {}).get("model_json")
                if mj and os.path.exists(mj):
                    with open(mj) as fh:
                        model = json.load(fh)
                    name_to_label = {b["name"]: b.get("label", b["name"])
                                     for b in model.get("bodies", [])}
            except Exception:
                pass
            return common.element_power_table(audit, name_to_label)
        except Exception:
            return {}

    def _open_paraview(self):
        ok, msg = paraview_launcher.launch(self.case_dir, self.settings)
        self.statusChanged.emit(("ParaView: " if ok else
                                 "ParaView failed: ") + msg)
