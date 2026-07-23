"""Results pane: browse a completed (or in-progress) case.

Works on a case directory (results/<model>/<case>/ inside a workspace or
repo tree). Shows report.json headline numbers, the audit outcome, and
thumbnail galleries of images/, spectra/, plots/, viz/, imaging/ (the
--image-sim products) plus the Time partition; "Open in ParaView" hands
the .vtp data to interactive ParaView.

MONITOR MODE (read-only view of a RUNNING case): when the case is locked
by a live process, a QTimer polls progress.json + the images as they
appear; editing/rerun affordances are the main window's job to disable -
this pane only ever reads.

P1 chunked-run contract affordances (header buttons, next to "Open in
ParaView"): "Resume run" appears for a DEAD case whose
cengine/checkpoint.json says status='tracing' (interrupted mid-trace,
lock free -- core.checkpointinfo.resume_state); "Extend run..." appears
for a COMPLETED C-engine case with a checkpoint
(core.checkpointinfo.extend_state). Both just emit a case_dir signal --
this pane never launches a subprocess itself (that's RunController's job,
wired up by MainWindow, matching the "this pane only ever reads" rule
above: it reads checkpoint state, but the actual resume/extend LAUNCH is
the main window's)."""

import csv
import json
import os
import shutil

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMenu,
    QPushButton, QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

import sys
_SCRIPTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import common  # noqa: E402

from ..core import checkpointinfo, paraview_launcher
from ..widgets.table_export import export_table_csv

_THUMB_W = 320


# -- data-export context menu (shared by every image display: gallery ------
# thumbnails, the lightbox, and Compare's galleries which reuse this module's
# classes wholesale) --------------------------------------------------------

def resolve_data_csv(png_path):
    """Find the data CSV paired with a displayed PNG, per the pipeline's
    data/index.csv contract (see the object-placer round's chain-API
    docs / CLAUDE.md): walk up from the PNG's own directory looking for
    a sibling ``data/`` directory (the first one found, going upward, is
    treated as authoritative -- this is "the dir that contains data/",
    the case dir for a results/<model>/<case> tree or a compare out-dir
    alike). If that data/ has an index.csv, parse it (stdlib csv) and
    match the ``image`` column against the PNG by path-relative-to-that-
    dir or by basename; the matching row's ``file`` column gives the
    CSV. Failing that, fall back to a same-basename CSV directly in that
    data/ dir. Returns an absolute path (str) or None -- never raises on
    a malformed/missing index.csv, since a missing pairing is just "no
    CSV export available", not an error."""
    png_path = os.path.abspath(str(png_path))
    basename = os.path.basename(png_path)
    stem, _ext = os.path.splitext(basename)

    d = os.path.dirname(png_path)
    prev = None
    while d and d != prev:
        data_dir = os.path.join(d, "data")
        if os.path.isdir(data_dir):
            index_path = os.path.join(data_dir, "index.csv")
            if os.path.exists(index_path):
                try:
                    with open(index_path, newline="") as fh:
                        for row in csv.DictReader(fh):
                            image = (row.get("image") or "").strip()
                            if not image:
                                continue
                            image_abs = os.path.normpath(
                                os.path.join(d, image))
                            if (image_abs == os.path.normpath(png_path)
                                    or os.path.basename(image) == basename):
                                csv_rel = (row.get("file") or "").strip()
                                if csv_rel:
                                    return os.path.normpath(
                                        os.path.join(d, csv_rel))
                except (OSError, csv.Error):
                    pass
            fallback = os.path.join(data_dir, stem + ".csv")
            if os.path.exists(fallback):
                return fallback
            # A data/ dir exists at this level (the case-dir boundary) but
            # nothing paired -- don't keep climbing past it.
            return None
        prev = d
        d = os.path.dirname(d)
    return None


def _choose_save_path(parent, default_name):
    """Dialog seam: the ONE place a save QFileDialog is shown. Tests
    monkeypatch this module function so offscreen runs never open a real
    modal (CLAUDE.md: never show an unguarded modal in a pane code
    path)."""
    path, _filt = QFileDialog.getSaveFileName(parent, "Save as…",
                                              default_name)
    return path or None


def _copy_to_chosen_path(parent, src_path, status_cb=None):
    """Prompt (via the dialog seam) and copy src_path there verbatim --
    a file copy, never a pixmap re-encode, so a saved PNG is
    bit-identical to the source."""
    dest = _choose_save_path(parent, os.path.basename(src_path))
    if not dest:
        return
    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    shutil.copyfile(src_path, dest)
    if status_cb:
        status_cb("Saved %s" % dest)


def build_image_context_menu(png_path, parent, status_cb=None):
    """Build (don't show) the right-click menu for a displayed image:
    "Save image as…" (always enabled, copies the PNG) and "Export data
    as CSV…" (enabled only when resolve_data_csv finds a pairing).
    Dialog-free to build -- tests call this directly and assert action
    texts/enabled states without ever invoking .exec()."""
    menu = QMenu(parent)
    save_act = menu.addAction("Save image as…")
    save_act.triggered.connect(
        lambda: _copy_to_chosen_path(parent, png_path, status_cb))
    csv_path = resolve_data_csv(png_path)
    export_act = menu.addAction("Export data as CSV…")
    export_act.setEnabled(csv_path is not None)
    if csv_path is not None:
        export_act.triggered.connect(
            lambda: _copy_to_chosen_path(parent, csv_path, status_cb))
    return menu


class _ClickableLabel(QLabel):
    """QLabel that calls a handler when clicked and offers a right-click
    save-image/export-data menu (build_image_context_menu)."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self._on_click = None
        self._status_cb = None

    def set_click_handler(self, callback):
        """Set callback to invoke on click; callback receives path."""
        self._on_click = callback

    def set_status_callback(self, callback):
        """callback(msg) is invoked with a short "Saved <path>" string
        after a successful save/export -- status-bar feedback, never a
        modal."""
        self._status_cb = callback

    def mousePressEvent(self, event):
        if self._on_click:
            self._on_click(self.path)
        super().mousePressEvent(event)

    def build_context_menu(self):
        """Dialog-free: build (don't show) the right-click menu -- the
        seam tests exercise directly."""
        return build_image_context_menu(self.path, self, self._status_cb)

    def contextMenuEvent(self, event):
        self.build_context_menu().exec(event.globalPos())


class _LightboxDialog(QDialog):
    """Non-modal dialog for viewing gallery images full-size with arrow-key cycling."""

    def __init__(self, paths, initial_index, parent=None,
                status_callback=None):
        super().__init__(parent)
        self.paths = paths
        self.current_index = max(0, min(initial_index, len(paths) - 1)) \
            if paths else 0
        self._original_pixmap = None
        self._status_cb = status_callback

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

    def build_context_menu(self):
        """Dialog-free: build (don't show) the right-click menu for the
        currently-displayed image, or None if there's nothing shown."""
        if not self.paths:
            return None
        path = self.paths[self.current_index]
        return build_image_context_menu(path, self, self._status_cb)

    def contextMenuEvent(self, event):
        menu = self.build_context_menu()
        if menu is not None:
            menu.exec(event.globalPos())


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
        self._status_cb = None

    def set_status_callback(self, callback):
        """Wire in a callable(msg) invoked after a thumbnail/lightbox
        save-image or export-CSV action completes (e.g.
        ResultsPane.statusChanged.emit or ComparePane's log strip) --
        status-bar feedback only, never a modal."""
        self._status_cb = callback

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
            label.set_status_callback(self._status_cb)
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
        self._lightbox = _LightboxDialog(self._paths, index, parent=self,
                                         status_callback=self._status_cb)
        self._lightbox.show()

    def clear(self):
        """Session boundary: drop every thumbnail and close an open
        lightbox (isVisible-guarded -- offscreen teardown safety)."""
        if self._lightbox is not None:
            if self._lightbox.isVisible():
                self._lightbox.close()
            self._lightbox = None
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._shown = {}
        self._paths = []


class ResultsPane(QWidget):
    statusChanged = Signal(str)     # case status string for the title bar
    resumeRequested = Signal(str)   # case_dir -- "Resume run" clicked
    extendRequested = Signal(str)   # case_dir -- "Extend run..." clicked

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
        # (clear_case() below resets to exactly this state)
        self.title.setStyleSheet("font-weight: bold;")
        head.addWidget(self.title, 1)
        self.resume_btn = QPushButton("Resume run")
        self.resume_btn.setToolTip(
            "Resume this interrupted C-engine trace from its last "
            "checkpoint (P1 chunked-run contract)")
        self.resume_btn.clicked.connect(
            lambda: self.resumeRequested.emit(self.case_dir))
        self.resume_btn.setEnabled(False)
        self.resume_btn.hide()
        head.addWidget(self.resume_btn)
        self.extend_btn = QPushButton("Extend run…")
        self.extend_btn.setToolTip(
            "Additively raise this completed run's ray count (P1 "
            "chunked-run contract)")
        self.extend_btn.clicked.connect(
            lambda: self.extendRequested.emit(self.case_dir))
        self.extend_btn.setEnabled(False)
        self.extend_btn.hide()
        head.addWidget(self.extend_btn)
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
        self.summary.setContextMenuPolicy(Qt.CustomContextMenu)
        self.summary.customContextMenuRequested.connect(
            self._on_summary_context_menu)
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
        self.power.setContextMenuPolicy(Qt.CustomContextMenu)
        self.power.customContextMenuRequested.connect(
            self._on_power_context_menu)
        self.tabs.addTab(self.power, "Power")

        self.galleries = {}

        # per-detector analysis metrics (Strehl/RMS/MTF50/EE/spot RMS --
        # flattened from report.json's optional 'analysis' block) above
        # a thumbnail gallery of results/<case>/analysis/*.png
        self.analysis_metrics = QTableWidget(0, 3)
        self.analysis_metrics.setHorizontalHeaderLabels(
            ["Detector", "Metric", "Value"])
        self.analysis_metrics.horizontalHeader().setStretchLastSection(True)
        self.analysis_metrics.setEditTriggers(QTableWidget.NoEditTriggers)
        self.analysis_metrics.setMaximumHeight(160)
        self.analysis_metrics.setContextMenuPolicy(Qt.CustomContextMenu)
        self.analysis_metrics.customContextMenuRequested.connect(
            self._on_analysis_context_menu)

        analysis_tab = QWidget()
        analysis_lay = QVBoxLayout(analysis_tab)
        analysis_lay.setContentsMargins(0, 0, 0, 0)
        analysis_lay.addWidget(self.analysis_metrics)
        analysis_gallery = _Gallery()
        analysis_gallery.set_status_callback(self.statusChanged.emit)
        self.galleries["analysis"] = analysis_gallery
        analysis_lay.addWidget(analysis_gallery, 1)
        self.tabs.addTab(analysis_tab, "Analysis")

        # per-source power breakdown (report.json's optional 'per_source'
        # list on each detector block)
        self.sources = QTableWidget(0, 7)
        self.sources.setHorizontalHeaderLabels(
            ["Detector", "Source", "λ stratum", "Pol stratum",
             "Coherent [mW]", "Incoherent [mW]", "Total [mW]"])
        self.sources.horizontalHeader().setStretchLastSection(True)
        self.sources.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sources.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sources.customContextMenuRequested.connect(
            self._on_sources_context_menu)
        self.tabs.addTab(self.sources, "Sources")

        for name in ("images", "spectra", "plots", "viz"):
            g = _Gallery()
            g.set_status_callback(self.statusChanged.emit)
            self.galleries[name] = g
            self.tabs.addTab(g, name.capitalize())
        # time-domain products get their own category (pulsed-optics P11):
        # det_*_time_*.png + gdd_budget.png would otherwise crowd Images
        # (four extra PNGs per detector on a pulsed run). The tab is empty
        # (and harmless) on every CW case.
        time_gallery = _Gallery()
        time_gallery.set_status_callback(self.statusChanged.emit)
        self.galleries["time"] = time_gallery
        self.tabs.addTab(time_gallery, "Time")
        # dynamic light scattering products (samples-instruments round):
        # dls_correlate.py writes <case>/dls/correlogram.png +
        # gamma_vs_q2.png; refresh()'s auto-glob loop picks the directory up
        # by gallery name, so this tab is empty (and harmless) on every case
        # run without run_dls.py + dls_correlate.py.
        dls_gallery = _Gallery()
        dls_gallery.set_status_callback(self.statusChanged.emit)
        self.galleries["dls"] = dls_gallery
        self.tabs.addTab(dls_gallery, "DLS")
        # image-simulation products (imaging-analysis round): --image-sim
        # writes <case>/imaging/image_sim_*.png; refresh()'s auto-glob
        # loop picks the directory up by gallery name, so this tab is
        # empty (and harmless) on every case run without the flag.
        imaging_gallery = _Gallery()
        imaging_gallery.set_status_callback(self.statusChanged.emit)
        self.galleries["imaging"] = imaging_gallery
        self.tabs.addTab(imaging_gallery, "Imaging")

        # virtual instrument layer (engine3.md P2.5 §9): flattened
        # report.json detectors.<label>.instrument block (row/class/mode/
        # seed/saturation/counts/SNR for a camera; power reading for a
        # power meter; resolution/floor for a spectrometer) above a
        # thumbnail gallery of <case>/instrument/*.png (camera counts
        # images -- the spectrometer's own instrument_*.png lands in the
        # existing Spectra tab, see post_process.render_instrument_
        # spectrometer, so it is not duplicated here).
        self.instrument_metrics = QTableWidget(0, 3)
        self.instrument_metrics.setHorizontalHeaderLabels(
            ["Detector", "Metric", "Value"])
        self.instrument_metrics.horizontalHeader().setStretchLastSection(
            True)
        self.instrument_metrics.setEditTriggers(QTableWidget.NoEditTriggers)
        self.instrument_metrics.setMaximumHeight(160)
        self.instrument_metrics.setContextMenuPolicy(Qt.CustomContextMenu)
        self.instrument_metrics.customContextMenuRequested.connect(
            self._on_instrument_context_menu)

        instrument_tab = QWidget()
        instrument_lay = QVBoxLayout(instrument_tab)
        instrument_lay.setContentsMargins(0, 0, 0, 0)
        instrument_lay.addWidget(self.instrument_metrics)
        instrument_gallery = _Gallery()
        instrument_gallery.set_status_callback(self.statusChanged.emit)
        self.galleries["instrument"] = instrument_gallery
        instrument_lay.addWidget(instrument_gallery, 1)
        self.tabs.addTab(instrument_tab, "Instrument")

        lay.addWidget(self.tabs)
        self.audit = QLabel("")
        lay.addWidget(self.audit)
        # pulse/GDD readout (pulsed-optics P11): one compact line fed from
        # report.json's gdd_budget block + the detectors' time_products
        # FWHMs; empty (hidden) for CW cases without --gdd-budget
        self.pulse_summary = QLabel("")
        self.pulse_summary.setWordWrap(True)
        self.pulse_summary.setTextInteractionFlags(
            Qt.TextSelectableByMouse)
        lay.addWidget(self.pulse_summary)
        self.pulse_summary.hide()

    # -- table CSV export (right-click on summary/power) ----------------------
    def _build_table_export_menu(self, table, default_name):
        """Dialog-free: build (don't show) a table's right-click Export
        CSV menu -- tests call this directly and assert on it."""
        menu = QMenu(self)
        act = menu.addAction("Export CSV…")
        act.triggered.connect(lambda: self._export_table(table,
                                                          default_name))
        return menu

    def _export_table(self, table, default_name):
        dest = _choose_save_path(self, default_name)
        if not dest:
            return
        export_table_csv(table, dest)
        self.statusChanged.emit("Saved %s" % dest)

    def _on_summary_context_menu(self, pos):
        menu = self._build_table_export_menu(self.summary, "summary.csv")
        menu.exec(self.summary.viewport().mapToGlobal(pos))

    def _on_power_context_menu(self, pos):
        menu = self._build_table_export_menu(self.power, "power.csv")
        menu.exec(self.power.viewport().mapToGlobal(pos))

    def _on_analysis_context_menu(self, pos):
        menu = self._build_table_export_menu(
            self.analysis_metrics, "analysis_metrics.csv")
        menu.exec(self.analysis_metrics.viewport().mapToGlobal(pos))

    def _on_sources_context_menu(self, pos):
        menu = self._build_table_export_menu(self.sources, "sources.csv")
        menu.exec(self.sources.viewport().mapToGlobal(pos))

    def _on_instrument_context_menu(self, pos):
        menu = self._build_table_export_menu(
            self.instrument_metrics, "instrument_metrics.csv")
        menu.exec(self.instrument_metrics.viewport().mapToGlobal(pos))

    # -- loading -------------------------------------------------------------
    def load_case(self, case_dir, monitor=False):
        self.case_dir = str(case_dir)
        self._monitor.stop()
        if monitor:
            self._monitor.start()
        self.refresh()

    def stop_monitoring(self):
        self._monitor.stop()

    def clear_case(self):
        """Session boundary (File -> Close / opening a different model):
        stop live monitoring, forget the loaded case AND wipe every
        results widget -- forgetting only the pointer left the previous
        model's summary/power tables, galleries and audit line on screen
        after File > Open."""
        self._monitor.stop()
        self.case_dir = None
        self.title.setText("No results loaded")
        self.summary.setRowCount(0)
        self.power.setRowCount(0)
        self.analysis_metrics.setRowCount(0)
        self.sources.setRowCount(0)
        self.instrument_metrics.setRowCount(0)
        for gallery in self.galleries.values():
            gallery.clear()
        self.audit.setText("")
        self.pulse_summary.setText("")
        self.pulse_summary.hide()
        self.pv_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.resume_btn.hide()
        self.extend_btn.setEnabled(False)
        self.extend_btn.hide()
        self.statusChanged.emit("")

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

        resume_state = checkpointinfo.resume_state(self.case_dir)
        self.resume_btn.setVisible(resume_state is not None)
        self.resume_btn.setEnabled(resume_state is not None)
        if resume_state is not None:
            self.resume_btn.setToolTip(
                "Resume interrupted trace (%d chunk(s) done toward %d "
                "rays) from checkpoint"
                % (resume_state["n_chunks"], resume_state["target_rays"]))
        extend_state = checkpointinfo.extend_state(self.case_dir)
        self.extend_btn.setVisible(extend_state is not None)
        self.extend_btn.setEnabled(extend_state is not None)
        if extend_state is not None:
            self.extend_btn.setToolTip(
                "Additively extend past %d rays (current M_eff proxy "
                "~%.3g)" % (extend_state["current_rays"],
                            extend_state["m_eff_proxy"]))

        report_path = os.path.join(self.case_dir, "report.json")
        self.summary.setRowCount(0)
        self.analysis_metrics.setRowCount(0)
        self.sources.setRowCount(0)
        self.instrument_metrics.setRowCount(0)
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
            self._populate_analysis_metrics(dets)
            self._populate_sources(dets)
            self._populate_instrument(dets)
            self._populate_pulse_summary(report)

        from glob import glob
        for name, gallery in self.galleries.items():
            if name == "time":
                continue   # partitioned out of images/ below
            paths = glob(os.path.join(self.case_dir, name, "*.png"))
            if name == "images":
                time_paths = [p for p in paths if self._is_time_image(p)]
                paths = [p for p in paths if p not in set(time_paths)]
                self.galleries["time"].show_images(time_paths)
            gallery.show_images(paths)
        self.pv_btn.setEnabled(
            bool(paraview_launcher.viz_files(self.case_dir)))

    @staticmethod
    def _is_time_image(path):
        """images/ PNGs that belong on the Time tab: the four per-detector
        time products and the dispersion-budget table."""
        base = os.path.basename(path)
        if base == "gdd_budget.png":
            return True
        return base.startswith("det_") and ("_time_" in base)

    @staticmethod
    def _fmt_fs(fs):
        """fs -> compact string, promoting to ps/ns when large."""
        for scale, unit in ((1e6, "ns"), (1e3, "ps")):
            if abs(fs) >= scale:
                return "%.4g %s" % (fs / scale, unit)
        return "%.4g fs" % fs

    def _populate_pulse_summary(self, report):
        """One line: per pulsed source tau0 -> predicted tau_out + total
        material GDD (report.json gdd_budget, written by any --gdd-budget
        or time-product run), then each detector's measured time-profile
        FWHM (detectors.*.time_products.fwhm_s). Hidden when neither
        block exists (every pre-existing CW case)."""
        bits = []
        budget = report.get("gdd_budget") or {}
        for p in budget.get("pulses", []):
            bits.append("%s: τ₀ %s → %s predicted (φ₂ %.4g fs²)"
                        % (p.get("source", "?"),
                           self._fmt_fs(p.get("tau0_fs", 0.0)),
                           self._fmt_fs(p.get("tau_out_fs", 0.0)),
                           p.get("phi2_fs2", 0.0)))
        if budget and not budget.get("pulses"):
            bits.append("total material GDD %.4g fs²"
                        % budget.get("total", {}).get("gdd_fs2", 0.0))
        for label, d in sorted((report.get("detectors") or {}).items()):
            tp = d.get("time_products") or {}
            if "fwhm_s" in tp:
                bits.append("%s: measured FWHM %s"
                            % (label, self._fmt_fs(tp["fwhm_s"] * 1e15)))
        if bits:
            self.pulse_summary.setText("pulse/GDD:  " + "  ·  ".join(bits))
            self.pulse_summary.show()
        else:
            self.pulse_summary.setText("")
            self.pulse_summary.hide()

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

    @staticmethod
    def _flatten_scalars(prefix, obj):
        """Yield (dotted.path, value) for every scalar leaf under obj
        (dicts recurse with dotted keys, lists index with [i]) -- the
        report.json 'analysis' block's exact shape (PSF/MTF/EE/spot per
        detector, keyed however the analysis stage groups by
        source/lambda) isn't pinned by contract here, so this stays
        schema-agnostic: only scalar (str/bool/int/float) leaves become
        rows, containers are walked, None is skipped."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = "%s.%s" % (prefix, k) if prefix else str(k)
                for pair in ResultsPane._flatten_scalars(key, v):
                    yield pair
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                key = "%s[%d]" % (prefix, i) if prefix else "[%d]" % i
                for pair in ResultsPane._flatten_scalars(key, v):
                    yield pair
        elif isinstance(obj, bool) or isinstance(obj, (int, float, str)):
            yield prefix, obj
        # None / anything else: not a displayable scalar, skip

    def _populate_analysis_metrics(self, dets):
        """Flatten each detector's optional 'analysis' block (Strehl,
        RMS waves, MTF50 tan/sag, EE r50/r80/r90, per-(source,lambda)
        spot RMS, ...) into (Detector, Metric, Value) rows. A detector
        with no 'analysis' block (or an old case with none at all)
        contributes zero rows -- old cases load unchanged."""
        rows = []
        for label, d in sorted((dets or {}).items()):
            analysis = d.get("analysis")
            if not isinstance(analysis, dict):
                continue
            for metric, value in self._flatten_scalars("", analysis):
                if isinstance(value, float):
                    value = "%.4g" % value
                rows.append((label, metric, value))
        self.analysis_metrics.setRowCount(len(rows))
        for row, (det, metric, value) in enumerate(rows):
            for col, val in enumerate((det, metric, value)):
                self.analysis_metrics.setItem(
                    row, col, QTableWidgetItem(str(val)))

    def _populate_instrument(self, dets):
        """Flatten each detector's optional 'instrument' block (engine3.md
        P2.5 §9: row/class/mode/seed + per-class fields -- saturation_
        fraction/mean_counts/max_counts/snr_estimate for a camera,
        power_reported_W/lam_ref_nm for a power meter, peak_power_W/
        resolution_fwhm_nm for a spectrometer) into (Detector, Metric,
        Value) rows, same schema-agnostic flattening as
        _populate_analysis_metrics. A detector with no 'instrument' block
        (the common case -- nothing assigned, or an old case) contributes
        zero rows."""
        rows = []
        for label, d in sorted((dets or {}).items()):
            instrument = d.get("instrument")
            if not isinstance(instrument, dict):
                continue
            for metric, value in self._flatten_scalars("", instrument):
                if isinstance(value, float):
                    value = "%.4g" % value
                rows.append((label, metric, value))
        self.instrument_metrics.setRowCount(len(rows))
        for row, (det, metric, value) in enumerate(rows):
            for col, val in enumerate((det, metric, value)):
                self.instrument_metrics.setItem(
                    row, col, QTableWidgetItem(str(val)))

    def _populate_sources(self, dets):
        """Flatten each detector's optional 'per_source' list (rows of
        {source, lam_stratum, pol_stratum, coherent_W, incoherent_W})
        into the Sources tab; absent/malformed -> zero rows (old cases
        load unchanged)."""
        rows = []
        for label, d in sorted((dets or {}).items()):
            per_source = d.get("per_source")
            if not isinstance(per_source, list):
                continue
            for entry in per_source:
                if not isinstance(entry, dict):
                    continue
                coherent_w = entry.get("coherent_W", 0.0) or 0.0
                incoherent_w = entry.get("incoherent_W", 0.0) or 0.0
                rows.append((
                    label,
                    entry.get("source", ""),
                    entry.get("lam_stratum", ""),
                    entry.get("pol_stratum", ""),
                    "%.4g" % (coherent_w * 1e3),
                    "%.4g" % (incoherent_w * 1e3),
                    "%.4g" % ((coherent_w + incoherent_w) * 1e3),
                ))
        self.sources.setRowCount(len(rows))
        for row, vals in enumerate(rows):
            for col, val in enumerate(vals):
                self.sources.setItem(row, col, QTableWidgetItem(str(val)))

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
