"""ComparePane: sweep-comparison viewer/driver (Phase E).

Wraps scripts/compare_sweep.py — the optics-env script that reads a
sweep manifest (or a plain list of case dirs) and writes metrics.csv,
metric-vs-variable plots, a gallery, difference maps, and summary.json
into an output directory. This pane never imports numpy/h5py/matplotlib
itself (the GUI venv only ships numpy/scipy/h5py, no matplotlib) — it
only ever SHELLS OUT to the optics-env python via QProcess and then reads
back summary.json + the PNGs compare_sweep.py already rendered, mirroring
core/runner.py's and core/raypreview.py's QProcess plumbing.

Public API:
  load_summary(out_dir)      -- read out_dir/summary.json and populate.
  run_compare(manifest_path=None, case_dirs=None, ref=None, out_dir=None,
              python=None)   -- build argv, launch compare_sweep.py.
  add_case(case_dir)         -- dialog-free "Add case…" (append + enable
                                the "Compare N cases" button).

Testability seam: run_compare() calls self._start_process(argv) to launch
the QProcess; tests can monkeypatch that one method to capture argv
without spawning a real subprocess, then call the dialog-free
_on_finished(exit_code) directly to exercise the summary-load path (the
same pattern core/raypreview.py's tests use with real stub executables,
just one level more unit-y since compare_sweep.py needs the optics env).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                "scripts")))
import common  # noqa: E402  (stdlib-only shared contract hub)

from PySide6.QtCore import Qt, QProcess, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QSlider, QStackedWidget, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from .results import _Gallery

REPO_DIR = Path(__file__).resolve().parent.parent.parent
COMPARE_SWEEP_SCRIPT = REPO_DIR / "scripts" / "compare_sweep.py"

METRIC_COLUMNS = ["total_power_W", "peak_irradiance_W_m2",
                  "profile_visibility", "centroid_x_mm", "centroid_y_mm",
                  "rms_spot_radius_mm"]

EMPTY_MSG = "Run a sweep to compare results, or add finished cases"


def default_optics_python():
    """Same resolution order runner.py/raypreview.py use: env override,
    then common.py's machine-pinned default."""
    return os.environ.get("MIEWB_OPTICS_PYTHON", common.OPTICS_PYTHON)


def _default_out_manifest(manifest_path):
    """Mirror compare_sweep.py's default_out_manifest() so the GUI knows
    where summary.json will land without re-implementing the script."""
    data = json.loads(Path(manifest_path).read_text())
    return str(common.RESULTS_DIR / "comparisons"
              / ("sweep_%s_%s" % (data.get("model"), data.get("case"))))


def _case_label(case_dir):
    p = Path(case_dir)
    return "%s/%s" % (p.parent.name, p.name)


def _default_out_cases(case_dirs):
    """Mirror compare_sweep.py's default_out_cases()."""
    names = [_case_label(c) for c in case_dirs]
    joined = "_vs_".join(n.replace("/", "_") for n in names)
    if len(joined) > 120:
        joined = "%s_vs_%d_more" % (names[0].replace("/", "_"), len(names) - 1)
    return str(common.RESULTS_DIR / "comparisons" / joined)


class ComparePane(QWidget):
    compareFinished = Signal(bool)     # True on success

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._proc = None
        self._case_list = []
        self._summary = None
        self._out_dir = None
        # remembers the inputs of the last run_compare() call so the
        # Difference tab's ref selector can relaunch with a new --ref
        self._last_manifest_path = None
        self._last_case_dirs = None
        self._last_python = None

        outer = QVBoxLayout(self)
        head = QHBoxLayout()
        self.add_case_btn = QPushButton("Add case…")
        self.add_case_btn.clicked.connect(self._add_case_clicked)
        head.addWidget(self.add_case_btn)
        self.compare_cases_btn = QPushButton("Compare cases")
        self.compare_cases_btn.setEnabled(False)
        self.compare_cases_btn.clicked.connect(
            lambda: self.run_compare(case_dirs=list(self._case_list)))
        head.addWidget(self.compare_cases_btn)
        head.addStretch(1)
        outer.addLayout(head)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)

        self.empty_label = QLabel(EMPTY_MSG)
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.stack.addWidget(self.empty_label)

        self.tabs = QTabWidget()
        self._build_metrics_tab()
        self._build_images_tab()
        self._build_diff_tab()
        self._build_scrub_tab()
        self.stack.addWidget(self.tabs)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setFixedHeight(80)
        outer.addWidget(self.log)

        self.stack.setCurrentWidget(self.empty_label)

    # -- tab construction -------------------------------------------------------
    def _build_metrics_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.metrics_table = QTableWidget(
            0, 3 + len(METRIC_COLUMNS))
        self.metrics_table.setHorizontalHeaderLabels(
            ["Variant", "Detector", "Values"] + METRIC_COLUMNS)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.metrics_table)
        self.metrics_gallery = _Gallery()
        lay.addWidget(self.metrics_gallery)
        self.tabs.addTab(w, "Metrics")

    def _build_images_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        head = QHBoxLayout()
        head.addWidget(QLabel("Detector:"))
        self.images_detector_combo = QComboBox()
        self.images_detector_combo.currentTextChanged.connect(
            self._refresh_images_tab)
        head.addWidget(self.images_detector_combo)
        head.addStretch(1)
        lay.addLayout(head)
        self.images_gallery = _Gallery()
        lay.addWidget(self.images_gallery)
        self.tabs.addTab(w, "Images")

    def _build_diff_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        head = QHBoxLayout()
        head.addWidget(QLabel("Reference:"))
        self.ref_combo = QComboBox()
        self.ref_combo.currentTextChanged.connect(self._on_ref_changed)
        head.addWidget(self.ref_combo)
        head.addWidget(QLabel("Detector:"))
        self.diff_detector_combo = QComboBox()
        self.diff_detector_combo.currentTextChanged.connect(
            self._refresh_diff_tab)
        head.addWidget(self.diff_detector_combo)
        head.addStretch(1)
        lay.addLayout(head)
        self.diff_gallery = _Gallery()
        lay.addWidget(self.diff_gallery)
        self.tabs.addTab(w, "Difference")

    def _build_scrub_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        head = QHBoxLayout()
        head.addWidget(QLabel("Detector:"))
        self.scrub_detector_combo = QComboBox()
        self.scrub_detector_combo.currentTextChanged.connect(
            self._refresh_scrub_image)
        head.addWidget(self.scrub_detector_combo)
        head.addStretch(1)
        lay.addLayout(head)
        self.scrub_slider = QSlider(Qt.Horizontal)
        self.scrub_slider.setMinimum(0)
        self.scrub_slider.setMaximum(0)
        self.scrub_slider.valueChanged.connect(self._refresh_scrub_image)
        lay.addWidget(self.scrub_slider)
        self.scrub_caption = QLabel("")
        self.scrub_caption.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.scrub_caption)
        self.scrub_image = QLabel()
        self.scrub_image.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.scrub_image, 1)
        self.tabs.addTab(w, "Scrub")

    # -- "Add case…" flow --------------------------------------------------------
    def _add_case_clicked(self):
        case_dir = QFileDialog.getExistingDirectory(
            self, "Add finished case directory")
        if case_dir:
            self.add_case(case_dir)

    def add_case(self, case_dir):
        """Dialog-free: append case_dir (a results/<model>/<case> dir) to
        the pending comparison list and enable the compare button."""
        case_dir = str(case_dir)
        if case_dir not in self._case_list:
            self._case_list.append(case_dir)
        n = len(self._case_list)
        self.compare_cases_btn.setText(
            "Compare %d case%s" % (n, "" if n == 1 else "s"))
        self.compare_cases_btn.setEnabled(n >= 1)
        return list(self._case_list)

    # -- running compare_sweep.py -------------------------------------------------
    def is_running(self):
        return (self._proc is not None
                and self._proc.state() != QProcess.NotRunning)

    def _resolve_python(self):
        if self.settings is not None:
            try:
                return self.settings.optics_python()
            except Exception:
                pass
        return default_optics_python()

    def _build_argv(self, manifest_path, case_dirs, ref, out_dir, python):
        py = python or self._resolve_python()
        argv = [py, str(COMPARE_SWEEP_SCRIPT)]
        if manifest_path:
            argv += ["--manifest", str(manifest_path)]
        else:
            argv += ["--cases"] + [str(c) for c in case_dirs]
        argv += ["--out", str(out_dir)]
        if ref:
            argv += ["--ref", str(ref)]
        return argv

    def run_compare(self, manifest_path=None, case_dirs=None, ref=None,
                    out_dir=None, python=None):
        """Launch compare_sweep.py (manifest OR case_dirs mode). Returns
        True if launched, False if refused (already running, or neither
        manifest_path nor case_dirs given)."""
        if not manifest_path and not case_dirs:
            self._append_log("run_compare: need manifest_path or case_dirs")
            return False
        if self.is_running():
            self._append_log("run_compare: a comparison is already running")
            return False

        if out_dir is None:
            out_dir = (_default_out_manifest(manifest_path) if manifest_path
                      else _default_out_cases(case_dirs))

        self._last_manifest_path = manifest_path
        self._last_case_dirs = list(case_dirs) if case_dirs else None
        self._last_python = python
        self._pending_out_dir = str(out_dir)

        argv = self._build_argv(manifest_path, case_dirs, ref, out_dir,
                                python)
        self._append_log("running: %s" % " ".join(argv))
        ok = self._start_process(argv)
        if not ok:
            self._append_log("failed to start compare_sweep.py")
        return ok

    def _start_process(self, argv):
        """Actual QProcess launch — factored out as a seam so tests can
        monkeypatch it to capture argv without spawning a real
        subprocess (compare_sweep.py needs the optics env, which a fast
        offline GUI test shouldn't depend on)."""
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.setProgram(argv[0])
        proc.setArguments(argv[1:])
        proc.readyReadStandardOutput.connect(self._on_ready_read)
        proc.finished.connect(self._on_finished)
        self._proc = proc
        proc.start()
        if not proc.waitForStarted(5000):
            self._proc = None
            return False
        return True

    def _on_ready_read(self):
        proc = self.sender()
        if proc is None:
            return
        while proc.canReadLine():
            raw = bytes(proc.readLine()).decode("utf-8", errors="replace")
            self._append_log(raw.rstrip("\r\n"))

    def _on_finished(self, exit_code, _exit_status=QProcess.NormalExit):
        """Dialog-free on purpose (tests call this directly after
        monkeypatching _start_process)."""
        self._proc = None
        if exit_code == 0:
            self._append_log("compare_sweep.py finished OK")
            ok = self.load_summary(self._pending_out_dir)
            self.compareFinished.emit(ok)
        else:
            self._append_log("compare_sweep.py FAILED (exit %d)"
                             % exit_code)
            self.compareFinished.emit(False)

    def _append_log(self, line):
        self.log.appendPlainText(line)

    # -- loading summary.json ----------------------------------------------------
    def load_summary(self, out_dir):
        out_dir = Path(out_dir)
        path = out_dir / "summary.json"
        if not path.exists():
            self._append_log("no summary.json in %s" % out_dir)
            return False
        try:
            with open(path) as fh:
                summary = json.load(fh)
        except (OSError, ValueError) as exc:
            self._append_log("bad summary.json in %s: %s" % (out_dir, exc))
            return False
        self._summary = summary
        self._out_dir = out_dir
        self._populate(summary)
        self.stack.setCurrentWidget(self.tabs)
        return True

    def _abs(self, rel_path):
        return str(self._out_dir / rel_path)

    def _all_detectors(self, summary):
        labels = set()
        for v in summary.get("variants", []):
            labels |= set(v.get("detectors", {}))
        labels |= set(summary.get("gallery", {}))
        labels |= set(summary.get("diffs", {}))
        return sorted(labels)

    def _values_label(self, variant, order):
        values = variant.get("values") or {}
        if not values:
            return variant.get("stem", "")
        keys = [k for k in (order or values.keys()) if k in values]
        return ", ".join("%s=%g" % (k.rsplit(".", 1)[-1], values[k])
                         for k in keys)

    def _populate(self, summary):
        self._populate_metrics(summary)
        self._populate_images(summary)
        self._populate_diff(summary)
        self._populate_scrub(summary)

    def _populate_metrics(self, summary):
        order = summary.get("order") or []
        rows = []
        for v in summary.get("variants", []):
            vlabel = self._values_label(v, order)
            dets = v.get("detectors") or {}
            if not dets:
                rows.append((v.get("stem", ""), "-", vlabel, {}))
                continue
            for label, metrics in sorted(dets.items()):
                rows.append((v.get("stem", ""), label, vlabel, metrics))
        self.metrics_table.setRowCount(len(rows))
        for r, (stem, label, vlabel, metrics) in enumerate(rows):
            cols = [stem, label, vlabel] + [
                ("%.4g" % metrics[m]) if metrics.get(m) is not None else "—"
                for m in METRIC_COLUMNS]
            for c, val in enumerate(cols):
                self.metrics_table.setItem(r, c, QTableWidgetItem(str(val)))
        plot_paths = [self._abs(p) for p in summary.get("plots", [])]
        self.metrics_gallery.show_images(plot_paths)

    def _populate_images(self, summary):
        gallery = summary.get("gallery", {})
        self.images_detector_combo.blockSignals(True)
        self.images_detector_combo.clear()
        self.images_detector_combo.addItems(sorted(gallery.keys()))
        self.images_detector_combo.blockSignals(False)
        self._refresh_images_tab()

    def _refresh_images_tab(self):
        if not self._summary:
            self.images_gallery.show_images([])
            return
        gallery = self._summary.get("gallery", {})
        label = self.images_detector_combo.currentText()
        entries = gallery.get(label, [])
        self.images_gallery.show_images(
            [self._abs(e["image"]) for e in entries])

    def _populate_diff(self, summary):
        stems = [v.get("stem", "") for v in summary.get("variants", [])]
        self.ref_combo.blockSignals(True)
        self.ref_combo.clear()
        self.ref_combo.addItems(stems)
        ref = summary.get("ref")
        if ref in stems:
            self.ref_combo.setCurrentText(ref)
        self.ref_combo.blockSignals(False)

        diffs = summary.get("diffs", {})
        self.diff_detector_combo.blockSignals(True)
        self.diff_detector_combo.clear()
        self.diff_detector_combo.addItems(sorted(diffs.keys()))
        self.diff_detector_combo.blockSignals(False)
        self._refresh_diff_tab()

    def _refresh_diff_tab(self):
        if not self._summary:
            self.diff_gallery.show_images([])
            return
        diffs = self._summary.get("diffs", {})
        label = self.diff_detector_combo.currentText()
        entries = diffs.get(label, [])
        self.diff_gallery.show_images(
            [self._abs(e["image"]) for e in entries])

    def _on_ref_changed(self, new_ref):
        if not new_ref or self._summary is None:
            return
        if new_ref == self._summary.get("ref"):
            return
        self.run_compare(manifest_path=self._last_manifest_path,
                         case_dirs=self._last_case_dirs, ref=new_ref,
                         out_dir=self._out_dir, python=self._last_python)

    def _populate_scrub(self, summary):
        variants = summary.get("variants", [])
        self.scrub_slider.blockSignals(True)
        self.scrub_slider.setMinimum(0)
        self.scrub_slider.setMaximum(max(0, len(variants) - 1))
        self.scrub_slider.setValue(0)
        self.scrub_slider.blockSignals(False)

        labels = self._all_detectors(summary)
        self.scrub_detector_combo.blockSignals(True)
        self.scrub_detector_combo.clear()
        self.scrub_detector_combo.addItems(labels)
        self.scrub_detector_combo.blockSignals(False)
        self._refresh_scrub_image()

    def _refresh_scrub_image(self):
        if not self._summary:
            self.scrub_image.clear()
            self.scrub_caption.setText("")
            return
        variants = self._summary.get("variants", [])
        if not variants:
            self.scrub_image.clear()
            self.scrub_caption.setText("")
            return
        idx = min(self.scrub_slider.value(), len(variants) - 1)
        variant = variants[idx]
        label = self.scrub_detector_combo.currentText()
        order = self._summary.get("order") or []
        vlabel = self._values_label(variant, order)
        self.scrub_caption.setText("%s — %s" % (variant.get("stem", ""),
                                                vlabel))
        gallery = self._summary.get("gallery", {}).get(label, [])
        entry = next((e for e in gallery
                     if e.get("stem") == variant.get("stem")), None)
        if entry is None:
            self.scrub_image.clear()
            return
        pm = QPixmap(self._abs(entry["image"]))
        if not pm.isNull():
            self.scrub_image.setPixmap(pm.scaledToWidth(
                480, Qt.SmoothTransformation))

    # -- session reset -----------------------------------------------------------
    def clear(self):
        self._summary = None
        self._out_dir = None
        self._case_list = []
        self.compare_cases_btn.setText("Compare cases")
        self.compare_cases_btn.setEnabled(False)
        self.metrics_table.setRowCount(0)
        self.metrics_gallery.clear()
        self.images_gallery.clear()
        self.diff_gallery.clear()
        self.images_detector_combo.clear()
        self.diff_detector_combo.clear()
        self.ref_combo.clear()
        self.scrub_detector_combo.clear()
        self.scrub_slider.setMaximum(0)
        self.scrub_image.clear()
        self.scrub_caption.setText("")
        self.log.clear()
        self.stack.setCurrentWidget(self.empty_label)
