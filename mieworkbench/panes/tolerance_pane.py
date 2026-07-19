"""TolerancePane - the sensitivity + Monte-Carlo tolerancing dock (F2).

Full GUI parity with scripts/tolerance.py: a tolerance table
(name/nominal/distribution/band), an operand table (the same merit
grammar as the optimizer), a focus-compensator picker, the Monte-Carlo
settings (draws/seed/merit-threshold/&c), Run/Stop, and a LIVE result
view — a yield histogram fed by the per-draw '@MIEWB' events and a
sensitivity bar chart fed by the phase="sensitivity_done" event
(ToleranceController parses both).

Plotting mirrors panes/optimize_pane.py: PySide6.QtCharts when
importable, else a dependency-free QPainter widget — same data API
either way (the data lives on the wrapper, so tests never care which
backend drew the pixels). Penalized evaluations (merit >= 1e8 —
optimize.PENALTY for failed evals) are excluded from the histogram
binning so one bad draw cannot flatten the distribution.

Wiring (mainwindow.py): runRequested/stopRequested out; on_progress /
on_started / on_finished in. The pane owns NO QProcess — that is
core/tolerance_controller.py's job, mirroring the OptimizePane/
OptimizeController split.
"""

import argparse
import math
import sys
from os.path import dirname, join, normpath

sys.path.insert(0, normpath(join(dirname(__file__), "..", "..",
                                 "scripts")))
import cli_specs  # noqa: E402  (stdlib-only; specs + parsers)
import common     # noqa: E402  (stdlib-only; PRESETS)

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QToolTip, QVBoxLayout, QWidget,
)

from ..core.variables import qualify_var_name
from .optimize_pane import (NAME_COMBO_TOOLTIP, match_error_line,
                            variable_bounds, _make_error_banner)
from .plot_inspect import (DataTableDialog, build_plot_menu,
                           format_point_tooltip)

try:
    from PySide6.QtCharts import (QBarCategoryAxis, QBarSeries, QBarSet,
                                  QChart, QChartView, QLineSeries,
                                  QValueAxis)
    HAVE_QTCHARTS = True
except ImportError:            # pragma: no cover - GUI venv ships QtCharts
    HAVE_QTCHARTS = False

# merits at/above this are penalty sentinels (optimize.PENALTY) — the
# histogram counts them separately, never bins them
PENALTY_FLOOR = 1e8


# =============================================================================
# result plots (QtCharts when available, QPainter fallback)
# =============================================================================
class _BarChartBase(QWidget):
    """Shared QtCharts/QPainter scaffolding: subclasses provide
    categories() + values() and the base rebuilds a one-set bar series
    (or repaints the fallback canvas)."""

    y_title = "value"

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if HAVE_QTCHARTS:
            self._chart = QChart()
            self._chart.setTitle(title)
            self._chart.legend().setVisible(False)
            self._view = QChartView(self._chart)
            self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
            layout.addWidget(self._view)
        else:
            self._canvas = _PainterBars(self, title)
            layout.addWidget(self._canvas)

    # subclasses: -> (list[str], list[float])
    def _bars(self):
        raise NotImplementedError

    def _connect_bar_hover(self, bar_set):    # pragma: no cover - QtCharts
        """Hook: subclasses connect bar_set.hovered here (default no-op).
        Called once per rebuild, since QBarSet is recreated each time."""

    def _rebuild(self):
        if not HAVE_QTCHARTS:
            self._canvas.update()
            return
        cats, vals = self._bars()
        self._chart.removeAllSeries()
        for ax in list(self._chart.axes()):
            self._chart.removeAxis(ax)
        if not vals:
            return
        bar_set = QBarSet("")
        for v in vals:
            bar_set.append(float(v))
        self._connect_bar_hover(bar_set)
        series = QBarSeries()
        series.append(bar_set)
        self._chart.addSeries(series)
        ax_x = QBarCategoryAxis()
        ax_x.append(cats)
        ax_y = QValueAxis()
        ax_y.setTitleText(self.y_title)
        hi = max(vals)
        ax_y.setRange(0.0, hi * 1.1 if hi > 0 else 1.0)
        self._chart.addAxis(ax_x, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(ax_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(ax_x)
        series.attachAxis(ax_y)


class _PainterBars(QWidget):
    """Dependency-free fallback: labelled vertical bars."""

    def __init__(self, plot, title):
        super().__init__(plot)
        self._plot = plot
        self._title = title
        self.setMinimumHeight(140)

    def paintEvent(self, _event):    # pragma: no cover - fallback backend
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1e1e1e"))
        p.setPen(QPen(QColor("#9ca3af")))
        p.drawText(6, 14, self._title)
        margin = 24
        area = QRectF(margin, 20, self.width() - margin - 8,
                      self.height() - margin - 24)
        p.setPen(QPen(QColor("#6b7280")))
        p.drawRect(area)
        cats, vals = self._plot._bars()
        if not vals:
            p.drawText(area, Qt.AlignmentFlag.AlignCenter, "no data yet")
            return
        hi = max(vals) or 1.0
        n = len(vals)
        slot = area.width() / n
        p.setPen(Qt.PenStyle.NoPen)
        for i, v in enumerate(vals):
            h = area.height() * (v / (hi * 1.1))
            bar = QRectF(area.left() + i * slot + slot * 0.15,
                         area.bottom() - h, slot * 0.7, h)
            p.fillRect(bar, QColor("#22d3ee"))
        p.setPen(QPen(QColor("#9ca3af")))
        for i, c in enumerate(cats):
            p.drawText(QRectF(area.left() + i * slot, area.bottom() + 2,
                              slot, 16),
                       Qt.AlignmentFlag.AlignHCenter, str(c))
        p.drawText(4, int(area.top()) + 10, "%.3g" % hi)


class SensitivityBarPlot(_BarChartBase):
    """Ranked merit-impact bars, one per tolerance parameter. Fed from
    the phase='sensitivity_done' progress event's compact table."""

    y_title = "merit impact"

    def __init__(self, parent=None):
        super().__init__("Sensitivity (merit impact over the band)",
                         parent)
        self._rows = []
        self._data_dialog = None
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    def set_rows(self, rows):
        """rows: [{"name","rank","impact","derivative"}] (ranked;
        impact None = penalized parameter, drawn as zero)."""
        self._rows = [dict(r) for r in rows]
        self._rebuild()

    def clear(self):
        self._rows = []
        self._rebuild()

    def rows(self):
        return [dict(r) for r in self._rows]

    def _bars(self):
        cats = [r["name"] for r in self._rows]
        vals = [float(r["impact"]) if r.get("impact") is not None else 0.0
                for r in self._rows]
        return cats, vals

    # -- hover / context menu ------------------------------------------------------
    def _connect_bar_hover(self, bar_set):    # pragma: no cover - QtCharts
        bar_set.hovered.connect(self._on_bar_hovered)

    def _on_bar_hovered(self, status, index):   # pragma: no cover - QtCharts
        if not status or index < 0 or index >= len(self._rows):
            QToolTip.hideText()
            return
        r = self._rows[index]
        QToolTip.showText(QCursor.pos(), format_point_tooltip(
            r.get("name", ""), {"derivative": r.get("derivative")},
            r.get("impact"), r.get("rank")))

    def _on_context_menu(self, pos):
        menu = build_plot_menu([("Show data…", self._show_data_dialog)],
                               self)
        menu.exec(self.mapToGlobal(pos))

    def _show_data_dialog(self):
        self._data_dialog = DataTableDialog(
            self.rows(), "sensitivity", "Sensitivity", self)
        self._data_dialog.show()
        self._data_dialog.raise_()


class MeritDistributionPlot(QWidget):
    """Live Monte-Carlo merit distribution: a frequency polygon (a line
    through the histogram bin centres) plus a cumulative-distribution
    curve on a RIGHT-hand 0..1 axis (the "Both" view). add_merit() per
    draw; bins are recomputed over the collected non-penalized merits
    (penalty sentinels are counted, never binned).

    The public data API is preserved from the former YieldHistogram
    (add_merit / merit_count / plotted_merits / penalized_count / bins)
    so existing tests keep passing; add_merit now also records the draw's
    params + pass/fail for hover + the "Show data…" dialog. QtCharts when
    importable, else a dependency-free QPainter canvas — same API either
    way."""

    MAX_BINS = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self._merits = []
        self._params = []
        self._passed = []
        self._data_dialog = None
        self.setMinimumHeight(150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if HAVE_QTCHARTS:
            self._chart = QChart()
            self._chart.setTitle("Monte-Carlo merit distribution")
            self._chart.legend().setVisible(True)
            self._freq_series = QLineSeries()
            self._freq_series.setName("draws")
            self._cdf_series = QLineSeries()
            self._cdf_series.setName("cumulative")
            self._chart.addSeries(self._freq_series)
            self._chart.addSeries(self._cdf_series)
            self._ax_x = QValueAxis()
            self._ax_x.setTitleText("merit")     # fixes the missing label
            self._ax_y = QValueAxis()
            self._ax_y.setTitleText("draws")
            self._ax_cdf = QValueAxis()
            self._ax_cdf.setTitleText("cumulative")
            self._ax_cdf.setRange(0.0, 1.0)
            self._chart.addAxis(self._ax_x, Qt.AlignmentFlag.AlignBottom)
            self._chart.addAxis(self._ax_y, Qt.AlignmentFlag.AlignLeft)
            self._chart.addAxis(self._ax_cdf, Qt.AlignmentFlag.AlignRight)
            self._freq_series.attachAxis(self._ax_x)
            self._freq_series.attachAxis(self._ax_y)
            self._cdf_series.attachAxis(self._ax_x)
            self._cdf_series.attachAxis(self._ax_cdf)
            self._freq_series.hovered.connect(self._on_hovered)
            self._cdf_series.hovered.connect(self._on_hovered)
            self._view = QChartView(self._chart)
            self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
            layout.addWidget(self._view)
        else:
            self._canvas = _PainterDistribution(self)
            layout.addWidget(self._canvas)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    # -- data API (preserved) ------------------------------------------------------
    def add_merit(self, merit, params=None, passed=None):
        self._merits.append(float(merit))
        self._params.append(dict(params) if params else {})
        self._passed.append(passed)
        self._rebuild()

    def clear(self):
        self._merits, self._params, self._passed = [], [], []
        self._rebuild()

    def merit_count(self):
        return len(self._merits)

    def plotted_merits(self):
        """The merits that participate in the binning (penalty
        sentinels excluded)."""
        return [m for m in self._merits if m < PENALTY_FLOOR]

    def penalized_count(self):
        """How many recorded draws were penalty sentinels (merit >=
        PENALTY_FLOOR) — counted, never binned."""
        return sum(1 for m in self._merits if m >= PENALTY_FLOOR)

    def bins(self):
        """(edges, counts) over the plotted merits (edges has
        len(counts)+1 entries; both empty when no data)."""
        ok = self.plotted_merits()
        if not ok:
            return [], []
        lo, hi = min(ok), max(ok)
        if hi <= lo:
            pad = abs(lo) * 1e-9 + 1e-12
            lo, hi = lo - pad, hi + pad
        nb = min(self.MAX_BINS, max(4, int(math.ceil(math.sqrt(len(ok))))))
        width = (hi - lo) / nb
        edges = [lo + k * width for k in range(nb)] + [hi]
        counts = [0] * nb
        for m in ok:
            counts[min(int((m - lo) / width), nb - 1)] += 1
        return edges, counts

    def cdf_points(self):
        """[(merit, cumulative_fraction)] over the plotted merits sorted
        ascending — monotone, ending at 1.0 (empty when no data)."""
        ok = sorted(self.plotted_merits())
        n = len(ok)
        return [(m, (i + 1) / n) for i, m in enumerate(ok)] if n else []

    def history(self):
        """Per-draw records [{"draw","merit","params","passed"}] in
        arrival order (draw# is the 1-based arrival index)."""
        return [{"draw": i + 1, "merit": m, "params": dict(p),
                 "passed": pa}
                for i, (m, p, pa) in enumerate(
                    zip(self._merits, self._params, self._passed))]

    def var_names(self):
        seen = []
        for params in self._params:
            for key in params:
                if key not in seen:
                    seen.append(key)
        return seen

    # -- rendering -----------------------------------------------------------------
    def _rebuild(self):
        if not HAVE_QTCHARTS:
            self._canvas.update()
            return
        edges, counts = self.bins()
        self._freq_series.clear()
        self._cdf_series.clear()
        if counts:
            centres = [0.5 * (a + b) for a, b in zip(edges, edges[1:])]
            for c, y in zip(centres, counts):
                self._freq_series.append(c, float(y))
            self._ax_x.setRange(edges[0], edges[-1])
            top = max(counts)
            self._ax_y.setRange(0.0, top * 1.1 if top > 0 else 1.0)
        for m, frac in self.cdf_points():
            self._cdf_series.append(m, frac)

    # -- hover / context menu ------------------------------------------------------
    def _on_hovered(self, point, state):    # pragma: no cover - QtCharts
        if not state:
            QToolTip.hideText()
            return
        QToolTip.showText(QCursor.pos(), "merit %.4g" % point.x())

    def _on_context_menu(self, pos):
        menu = build_plot_menu([("Show data…", self._show_data_dialog)],
                               self)
        menu.exec(self.mapToGlobal(pos))

    def _show_data_dialog(self):
        self._data_dialog = DataTableDialog(
            self.history(), "mc", "Monte-Carlo draws", self)
        self._data_dialog.show()
        self._data_dialog.raise_()


class _PainterDistribution(QWidget):
    """Dependency-free fallback: frequency polygon + CDF + x tick labels."""

    def __init__(self, plot):
        super().__init__(plot)
        self._plot = plot
        self.setMinimumHeight(150)

    def paintEvent(self, _event):    # pragma: no cover - fallback backend
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1e1e1e"))
        p.setPen(QPen(QColor("#9ca3af")))
        p.drawText(6, 14, "Monte-Carlo merit distribution")
        margin = 32
        area = QRectF(margin, 20, self.width() - 2 * margin,
                      self.height() - margin - 24)
        p.setPen(QPen(QColor("#6b7280")))
        p.drawRect(area)
        edges, counts = self._plot.bins()
        if not counts:
            p.drawText(area, Qt.AlignmentFlag.AlignCenter, "no data yet")
            return
        lo, hi = edges[0], edges[-1]
        span = (hi - lo) or 1.0
        top = max(counts) or 1

        def x_of(m):
            return area.left() + area.width() * ((m - lo) / span)

        # frequency polygon through bin centres
        centres = [0.5 * (a + b) for a, b in zip(edges, edges[1:])]
        p.setPen(QPen(QColor("#22d3ee"), 2))
        prev = None
        for c, y in zip(centres, counts):
            cur = QPointF(x_of(c),
                          area.bottom() - area.height() * (y / (top * 1.1)))
            if prev is not None:
                p.drawLine(prev, cur)
            prev = cur
        # CDF on the same x, full-height 0..1
        p.setPen(QPen(QColor("#e879f9"), 2))
        prev = None
        for m, frac in self._plot.cdf_points():
            cur = QPointF(x_of(m), area.bottom() - area.height() * frac)
            if prev is not None:
                p.drawLine(prev, cur)
            prev = cur
        # x tick labels at the edges
        p.setPen(QPen(QColor("#9ca3af")))
        p.drawText(QRectF(area.left(), area.bottom() + 2, 60, 14),
                   Qt.AlignmentFlag.AlignLeft, "%.3g" % lo)
        p.drawText(QRectF(area.right() - 60, area.bottom() + 2, 60, 14),
                   Qt.AlignmentFlag.AlignRight, "%.3g" % hi)
        p.drawText(QRectF(area.center().x() - 20, area.bottom() + 2, 40, 14),
                   Qt.AlignmentFlag.AlignHCenter, "merit")


# =============================================================================
# the pane
# =============================================================================
TOL_HEADERS = ["Variable", "Nominal", "Distribution", "Band"]
OPERAND_HEADERS = ["Operand", "Detector (optional)", "Target", "Weight"]


class TolerancePane(QWidget):
    runRequested = Signal()
    stopRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._varrows = {}     # {name: core.variables.VarRow} from the scene
        self._first_error = None   # first backend error line this run
        self._mc_threshold = None  # captured at on_started for pass/fail
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.error_banner = _make_error_banner()
        layout.addWidget(self.error_banner)

        tables_row = QHBoxLayout()
        tables_row.addWidget(self._build_tolerance_group(), 1)
        tables_row.addWidget(self._build_operand_group(), 1)
        layout.addLayout(tables_row)

        mid_row = QHBoxLayout()
        mid_row.addWidget(self._build_compensator_group(), 1)
        mid_row.addLayout(self._build_settings_grid(), 2)
        layout.addLayout(mid_row)

        layout.addLayout(self._build_run_row())

        plots_row = QHBoxLayout()
        self.sens_plot = SensitivityBarPlot()
        self.hist_plot = MeritDistributionPlot()
        plots_row.addWidget(self.sens_plot, 1)
        plots_row.addWidget(self.hist_plot, 1)
        layout.addLayout(plots_row, 1)

    # -- construction ------------------------------------------------------------
    def _build_tolerance_group(self):
        group = QGroupBox("Tolerances")
        v = QVBoxLayout(group)
        self.tol_table = QTableWidget(0, len(TOL_HEADERS))
        self.tol_table.setHorizontalHeaderLabels(TOL_HEADERS)
        self.tol_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.tol_table.setToolTip(
            "Tolerance parameters: spreadsheet cell aliases (bare 'alias' "
            "on the dim sheet, 'sheetlabel.alias', or a global picked "
            "from the dropdown = emitted miewb_vars.<name>), the nominal "
            "value, the perturbation distribution and its band (1-sigma "
            "for normal, half-width for uniform) in mm")
        v.addWidget(self.tol_table)
        row = QHBoxLayout()
        self.tol_add_btn = QPushButton("Add")
        self.tol_add_btn.setToolTip("Add a tolerance row")
        self.tol_add_btn.clicked.connect(lambda: self.add_tolerance())
        self.tol_del_btn = QPushButton("Remove")
        self.tol_del_btn.setToolTip("Remove the selected tolerance row")
        self.tol_del_btn.clicked.connect(
            lambda: self._remove_current(self.tol_table))
        row.addWidget(self.tol_add_btn)
        row.addWidget(self.tol_del_btn)
        row.addStretch(1)
        v.addLayout(row)
        return group

    def _build_operand_group(self):
        group = QGroupBox("Merit operands")
        v = QVBoxLayout(group)
        self.operand_table = QTableWidget(0, len(OPERAND_HEADERS))
        self.operand_table.setHorizontalHeaderLabels(OPERAND_HEADERS)
        self.operand_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.operand_table.setToolTip(
            "Merit operands (same grammar as the optimizer): spot_rms/"
            "focus and encircled_energy are minimized, detected_power "
            "and mtf50 maximized; a raw flattened report.json merit key "
            "(contains a '.') is minimized toward its target.")
        v.addWidget(self.operand_table)
        row = QHBoxLayout()
        self.operand_add_btn = QPushButton("Add")
        self.operand_add_btn.setToolTip("Add an operand row")
        self.operand_add_btn.clicked.connect(lambda: self.add_operand())
        self.operand_del_btn = QPushButton("Remove")
        self.operand_del_btn.setToolTip("Remove the selected operand row")
        self.operand_del_btn.clicked.connect(
            lambda: self._remove_current(self.operand_table))
        row.addWidget(self.operand_add_btn)
        row.addWidget(self.operand_del_btn)
        row.addStretch(1)
        v.addLayout(row)
        # a sensible default merit
        self.add_operand("spot_rms", "", 0.0, 1.0)
        return group

    def _build_compensator_group(self):
        self.comp_group = QGroupBox("Focus compensator")
        self.comp_group.setCheckable(True)
        self.comp_group.setChecked(False)
        self.comp_group.setToolTip(
            "Optimize one variable (e.g. the detector/focus position) "
            "per Monte-Carlo draw BEFORE recording its merit — the "
            "as-built refocus an assembly line would do")
        grid = QGridLayout(self.comp_group)
        grid.addWidget(QLabel("Variable:"), 0, 0)
        self.comp_var_edit = QLineEdit()
        self.comp_var_edit.setPlaceholderText("e.g. detpos")
        self.comp_var_edit.setToolTip(
            "Compensator spreadsheet cell alias (must differ from every "
            "tolerance parameter). Bare name = a `dim`-sheet alias; a "
            "global miewb_vars variable is emitted miewb_vars.<name> "
            "automatically; 'sheetlabel.alias' passes through as typed.")
        grid.addWidget(self.comp_var_edit, 0, 1, 1, 3)

        grid.addWidget(QLabel("Start:"), 1, 0)
        self.comp_start_spin = self._dspin(0.0)
        self.comp_start_spin.setToolTip(
            "Resting value when not compensating (and the nested "
            "optimizer's start)")
        grid.addWidget(self.comp_start_spin, 1, 1)
        grid.addWidget(QLabel("Lo:"), 1, 2)
        self.comp_lo_spin = self._dspin(-10.0)
        grid.addWidget(self.comp_lo_spin, 1, 3)
        grid.addWidget(QLabel("Hi:"), 1, 4)
        self.comp_hi_spin = self._dspin(10.0)
        grid.addWidget(self.comp_hi_spin, 1, 5)

        grid.addWidget(QLabel("Budget/draw:"), 2, 0)
        self.comp_budget_spin = QSpinBox()
        self.comp_budget_spin.setRange(1, 10000)
        self.comp_budget_spin.setValue(10)
        self.comp_budget_spin.setToolTip(
            "Merit evaluations per draw for the nested compensator "
            "optimization")
        grid.addWidget(self.comp_budget_spin, 2, 1)
        return self.comp_group

    @staticmethod
    def _dspin(value):
        s = QDoubleSpinBox()
        s.setDecimals(6)
        s.setRange(-1e9, 1e9)
        s.setValue(value)
        return s

    def _build_settings_grid(self):
        grid = QGridLayout()
        grid.addWidget(QLabel("Draws:"), 0, 0)
        self.draws_spin = QSpinBox()
        self.draws_spin.setRange(0, 1000000)
        self.draws_spin.setValue(50)
        self.draws_spin.setToolTip(
            "Monte-Carlo perturbation draws (0 = sensitivity only)")
        grid.addWidget(self.draws_spin, 0, 1)

        grid.addWidget(QLabel("Seed:"), 0, 2)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(42)
        self.seed_spin.setToolTip("RNG seed for the perturbation draws")
        grid.addWidget(self.seed_spin, 0, 3)

        self.threshold_check = QCheckBox("Yield threshold:")
        self.threshold_check.setToolTip(
            "A draw PASSES when its merit <= this; yield = passes/draws")
        grid.addWidget(self.threshold_check, 0, 4)
        self.threshold_spin = self._dspin(0.0)
        self.threshold_spin.setEnabled(False)
        self.threshold_check.toggled.connect(
            self.threshold_spin.setEnabled)
        grid.addWidget(self.threshold_spin, 0, 5)

        grid.addWidget(QLabel("Sens. delta:"), 1, 0)
        self.sens_delta_spin = QDoubleSpinBox()
        self.sens_delta_spin.setDecimals(4)
        self.sens_delta_spin.setRange(1e-4, 100.0)
        self.sens_delta_spin.setValue(1.0)
        self.sens_delta_spin.setToolTip(
            "Finite-difference step as a fraction of each band")
        grid.addWidget(self.sens_delta_spin, 1, 1)

        self.skip_sens_check = QCheckBox("Skip sensitivity")
        self.skip_sens_check.setToolTip(
            "Skip the finite-difference sensitivity table (Monte-Carlo "
            "only)")
        grid.addWidget(self.skip_sens_check, 1, 2, 1, 2)

        grid.addWidget(QLabel("Hist bins:"), 1, 4)
        self.hist_bins_spin = QSpinBox()
        self.hist_bins_spin.setRange(1, 1000)
        self.hist_bins_spin.setValue(20)
        self.hist_bins_spin.setToolTip(
            "Merit-histogram bin count in the report")
        grid.addWidget(self.hist_bins_spin, 1, 5)

        grid.addWidget(QLabel("Preset:"), 2, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(sorted(common.PRESETS))
        self.preset_combo.setCurrentText("quick")
        self.preset_combo.setToolTip(
            "fast_eval fidelity preset per evaluation")
        grid.addWidget(self.preset_combo, 2, 1)

        grid.addWidget(QLabel("Rays/eval:"), 2, 2)
        self.rays_spin = QDoubleSpinBox()
        self.rays_spin.setDecimals(0)
        self.rays_spin.setRange(0, 1e9)
        self.rays_spin.setValue(0)
        self.rays_spin.setSpecialValueText("preset")
        self.rays_spin.setToolTip(
            "Primary rays per source per evaluation (0 = preset default)")
        grid.addWidget(self.rays_spin, 2, 3)

        grid.addWidget(QLabel("Backend:"), 2, 4)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["worker", "full"])
        self.backend_combo.setToolTip(
            "fast_eval backend: worker = persistent FreeCAD (fast), "
            "full = fresh pipeline per eval (reference)")
        grid.addWidget(self.backend_combo, 2, 5)
        grid.setColumnStretch(6, 1)
        return grid

    def _build_run_row(self):
        row = QHBoxLayout()
        self.run_btn = QPushButton("Run Tolerance Study")
        self.run_btn.setToolTip(
            "Launch scripts/tolerance.py on the open model")
        self.run_btn.clicked.connect(self.runRequested.emit)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop the running tolerance study")
        self.stop_btn.clicked.connect(self.stopRequested.emit)
        self.status_label = QLabel("No tolerance study run yet")
        self.status_label.setToolTip(
            "Draws completed, latest merit and running yield")
        row.addWidget(self.run_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.status_label, 1)
        return row

    # -- scene variables (miewb_vars dropdown + auto-fill) ------------------------
    def set_variables(self, varrows):
        """Publish the scene's miewb_vars variables ({name: VarRow} from
        core.variables.parse_sheet) into every row's name dropdown. The
        combos stay editable so aliases outside miewb_vars (dim-sheet
        cells like 'dim.ct') can still be typed — an empty dict simply
        leaves a plain editable combo."""
        self._varrows = dict(varrows or {})
        for row in range(self.tol_table.rowCount()):
            combo = self.tol_table.cellWidget(row, 0)
            if combo is not None:
                self._repopulate_name_combo(combo)

    def _repopulate_name_combo(self, combo):
        """Swap the combo's item list for the current variable names,
        preserving whatever name the row already shows (signals blocked:
        repopulation must never trigger an auto-fill)."""
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(list(self._varrows))
        combo.setCurrentText(current)
        combo.blockSignals(False)

    def _make_name_combo(self, name):
        combo = QComboBox()
        combo.setEditable(True)      # names outside miewb_vars are typed
        combo.addItems(list(self._varrows))
        combo.setToolTip(NAME_COMBO_TOOLTIP)
        if name:      # blank keeps Qt's default: the first scene variable
            combo.setCurrentText(name)
        return combo

    def _combo_row(self, table, combo):
        for row in range(table.rowCount()):
            if table.cellWidget(row, 0) is combo:
                return row
        return None

    def _nominal_band(self, varrow):
        """(nominal, band) auto-fill: nominal = the sheet's current
        value; band = half the __min/__max span when real, else 10 % of
        |value| (0.1 for a zero value) — variable_bounds' fallback."""
        value, lo, hi = variable_bounds(varrow)
        return value, (hi - lo) / 2.0

    def _on_name_chosen(self, combo, text):
        """A row's name-combo changed: when the name is a known
        miewb_vars variable, auto-fill that row's nominal and band."""
        varrow = self._varrows.get(text)
        if varrow is None:
            return
        row = self._combo_row(self.tol_table, combo)
        if row is None:
            return
        nominal, band = self._nominal_band(varrow)
        for col, val in ((1, nominal), (3, band)):
            self.tol_table.setItem(row, col, QTableWidgetItem("%g" % val))

    @staticmethod
    def _row_name(table, row):
        combo = table.cellWidget(row, 0)
        if combo is not None:
            return combo.currentText().strip()
        item = table.item(row, 0)
        return (item.text() if item is not None else "").strip()

    # -- table helpers -------------------------------------------------------------
    @staticmethod
    def _remove_current(table):
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)

    def add_tolerance(self, name="", nominal=None, dist="normal",
                      band=None):
        """Insert a tolerance row. Explicit nominal/band are honored;
        any left as None auto-fill from the (known) named miewb_vars
        variable, else fall back to 0 / 0.1. A blank name adopts the
        combo's default choice (the first scene variable)."""
        table = self.tol_table
        row = table.rowCount()
        table.insertRow(row)
        name_combo = self._make_name_combo(name)
        table.setCellWidget(row, 0, name_combo)
        if not name:
            name = name_combo.currentText()   # first scene variable
        fill = (0.0, 0.1)
        if name in self._varrows:
            fill = self._nominal_band(self._varrows[name])
        nominal = fill[0] if nominal is None else nominal
        band = fill[1] if band is None else band
        table.setItem(row, 1, QTableWidgetItem("%g" % nominal))
        combo = QComboBox()
        combo.addItems(list(cli_specs.TOLERANCE_DISTS))
        combo.setCurrentText(dist)
        combo.setToolTip("normal(nominal, band) or "
                         "uniform(nominal +/- band)")
        table.setCellWidget(row, 2, combo)
        table.setItem(row, 3, QTableWidgetItem("%g" % band))
        # connect AFTER seeding the cells so explicit values always win
        name_combo.currentTextChanged.connect(
            lambda text, c=name_combo: self._on_name_chosen(c, text))
        return row

    def add_operand(self, operand="spot_rms", detector="", target=0.0,
                    weight=1.0):
        table = self.operand_table
        row = table.rowCount()
        table.insertRow(row)
        combo = QComboBox()
        combo.setEditable(True)      # raw merit keys are typed in
        combo.addItems(list(cli_specs.OPTIMIZE_OPERANDS))
        combo.setCurrentText(operand)
        combo.setToolTip("Named operand, or type a raw flattened "
                         "report.json merit key (contains a '.')")
        table.setCellWidget(row, 0, combo)
        table.setItem(row, 1, QTableWidgetItem(detector))
        table.setItem(row, 2, QTableWidgetItem("%g" % target))
        table.setItem(row, 3, QTableWidgetItem("%g" % weight))
        return row

    def _cell_text(self, table, row, col):
        item = table.item(row, col)
        return (item.text() if item is not None else "").strip()

    # -- config assembly -------------------------------------------------------------
    def tolerances(self):
        """Table rows -> 'name:nominal:dist:band' spec strings (rows with
        an empty name are skipped; non-numeric cells raise ValueError
        with the row named)."""
        out = []
        for row in range(self.tol_table.rowCount()):
            name = self._row_name(self.tol_table, row)
            if not name:
                continue
            nums = []
            for col, what in ((1, "nominal"), (3, "band")):
                text = self._cell_text(self.tol_table, row, col)
                try:
                    nums.append(float(text))
                except ValueError:
                    raise ValueError(
                        "tolerance row %d (%s): %s %r is not a number"
                        % (row + 1, name, what, text))
            combo = self.tol_table.cellWidget(row, 2)
            dist = combo.currentText() if combo else "normal"
            out.append("%s:%s:%s:%s" % (
                qualify_var_name(name, self._varrows), "%g" % nums[0],
                dist, "%g" % nums[1]))
        return out

    def operands(self):
        """Table rows -> 'operand[@detector]:target:weight' specs."""
        out = []
        for row in range(self.operand_table.rowCount()):
            combo = self.operand_table.cellWidget(row, 0)
            operand = combo.currentText().strip() if combo else ""
            if not operand:
                continue
            detector = self._cell_text(self.operand_table, row, 1)
            head = "%s@%s" % (operand, detector) if detector else operand
            nums = []
            for col, what in ((2, "target"), (3, "weight")):
                text = self._cell_text(self.operand_table, row, col)
                try:
                    nums.append(float(text))
                except ValueError:
                    raise ValueError(
                        "operand row %d (%s): %s %r is not a number"
                        % (row + 1, operand, what, text))
            out.append("%s:%s:%s" % (head, "%g" % nums[0], "%g" % nums[1]))
        return out

    def compensator(self):
        """'var:start:lo:hi' spec string, or None when disabled/empty.
        Raises ValueError on inconsistent bounds (dialog-free: the shell
        surfaces it in the status bar)."""
        if not self.comp_group.isChecked():
            return None
        name = self.comp_var_edit.text().strip()
        if not name:
            return None
        start = self.comp_start_spin.value()
        lo, hi = self.comp_lo_spin.value(), self.comp_hi_spin.value()
        if not lo < hi:
            raise ValueError("compensator %s: Lo must be < Hi" % name)
        if not lo <= start <= hi:
            raise ValueError(
                "compensator %s: Start %g outside [%g, %g]"
                % (name, start, lo, hi))
        return "%s:%s:%s:%s" % (qualify_var_name(name, self._varrows),
                                "%g" % start, "%g" % lo, "%g" % hi)

    def config(self):
        """The controller/build_args config dict (cli_specs 'tolerance'
        dests). Raises ValueError on malformed table cells."""
        cfg = {
            "tolerance": self.tolerances(),
            "operand": self.operands(),
            "draws": self.draws_spin.value(),
            "mc_seed": self.seed_spin.value(),
            "sens_delta": self.sens_delta_spin.value(),
            "skip_sensitivity": self.skip_sens_check.isChecked(),
            "hist_bins": self.hist_bins_spin.value(),
            "preset": self.preset_combo.currentText(),
            "eval_backend": self.backend_combo.currentText(),
        }
        if self.threshold_check.isChecked():
            cfg["merit_threshold"] = float(self.threshold_spin.value())
        comp = self.compensator()
        if comp is not None:
            cfg["compensator"] = comp
            cfg["comp_budget"] = self.comp_budget_spin.value()
        if self.rays_spin.value() > 0:
            cfg["rays"] = float(self.rays_spin.value())
        return cfg

    def apply_config(self, cfg):
        """Rebuild the tolerance/operand tables, compensator group, and
        settings from a config() dict (as persisted by
        Project.set_tolerance_config / returned by get_tolerance_config).
        Spec strings that fail to parse are skipped defensively -- a
        hand-edited or stale document must never block opening the
        scene. Round-trips: config() -> apply_config() -> config()
        reproduces the same dict."""
        if not cfg:
            return
        self.tol_table.setRowCount(0)
        for spec in cfg.get("tolerance") or []:
            try:
                t = cli_specs.parse_tolerance_spec(spec)
            except argparse.ArgumentTypeError:
                continue
            self.add_tolerance(t["name"], t["nominal"], t["dist"],
                               t["band"])
        self.operand_table.setRowCount(0)
        for spec in cfg.get("operand") or []:
            try:
                o = cli_specs.parse_operand_spec(spec)
            except argparse.ArgumentTypeError:
                continue
            self.add_operand(o["operand"], o["detector"] or "",
                             o["target"], o["weight"])
        if "draws" in cfg:
            self.draws_spin.setValue(int(cfg["draws"]))
        if "mc_seed" in cfg:
            self.seed_spin.setValue(int(cfg["mc_seed"]))
        if "sens_delta" in cfg:
            self.sens_delta_spin.setValue(float(cfg["sens_delta"]))
        self.skip_sens_check.setChecked(bool(cfg.get("skip_sensitivity",
                                                     False)))
        if "hist_bins" in cfg:
            self.hist_bins_spin.setValue(int(cfg["hist_bins"]))
        if "preset" in cfg:
            self.preset_combo.setCurrentText(str(cfg["preset"]))
        if "eval_backend" in cfg:
            self.backend_combo.setCurrentText(str(cfg["eval_backend"]))
        has_threshold = "merit_threshold" in cfg
        self.threshold_check.setChecked(has_threshold)
        self.threshold_spin.setEnabled(has_threshold)
        if has_threshold:
            self.threshold_spin.setValue(float(cfg["merit_threshold"]))
        comp_spec = cfg.get("compensator")
        if comp_spec:
            try:
                c = cli_specs.parse_compensator_spec(comp_spec)
            except argparse.ArgumentTypeError:
                c = None
        else:
            c = None
        self.comp_group.setChecked(c is not None)
        if c is not None:
            self.comp_var_edit.setText(c["name"])
            self.comp_start_spin.setValue(c["start"])
            self.comp_lo_spin.setValue(c["lo"])
            self.comp_hi_spin.setValue(c["hi"])
            if "comp_budget" in cfg:
                self.comp_budget_spin.setValue(int(cfg["comp_budget"]))
        self.rays_spin.setValue(float(cfg.get("rays") or 0.0))

    # -- run-state / progress slots ----------------------------------------------------
    def on_started(self):
        self.sens_plot.clear()
        self.hist_plot.clear()
        self._first_error = None
        # capture the run's yield threshold so per-draw pass/fail matches
        # tolerance.py exactly (the @MIEWB mc event carries no threshold)
        self._mc_threshold = (float(self.threshold_spin.value())
                              if self.threshold_check.isChecked() else None)
        self.error_banner.setVisible(False)
        self.error_banner.clear()
        self.status_label.setText("Tolerancing…")
        self.set_running(True)

    def on_line(self, text):
        """A raw stdout/stderr line from the tolerance process (wired
        alongside the console feed). Latches the FIRST substantive error
        line so on_finished can surface it on the banner."""
        if self._first_error is None:
            hit = match_error_line(text)
            if hit is not None:
                self._first_error = hit.strip()

    def on_progress(self, event):
        if event.get("stage") != "tolerance":
            return
        phase = event.get("phase")
        if phase == "sensitivity_done":
            self.sens_plot.set_rows(event.get("sensitivity") or [])
            return
        if phase == "mc" and event.get("merit") is not None:
            merit = event["merit"]
            # pass/fail mirrors tolerance.py: threshold set AND not
            # penalized AND merit <= threshold (else None when no threshold)
            if self._mc_threshold is None:
                passed = None
            else:
                passed = (not event.get("penalized")
                          and merit <= self._mc_threshold)
            self.hist_plot.add_merit(merit, params=event.get("params"),
                                     passed=passed)
            y = event.get("merit_yield")
            self.status_label.setText(
                "Draw %s/%s   merit %.6g%s"
                % (event.get("draw"), event.get("draws"), event["merit"],
                   "" if y is None else "   yield %.3f" % y))
            return
        if event.get("status") == "completed":
            y = event.get("merit_yield")
            n = event.get("n_evals")
            parts = ["Done"]
            if n is not None:
                parts.append("%s evals" % n)
            if y is not None:
                parts.append("yield %.3f" % y)
            self.status_label.setText("   ".join(parts))

    def on_finished(self, exit_code):
        self.set_running(False)
        n_pen = self.hist_plot.penalized_count()
        n_ok = len(self.hist_plot.plotted_merits())
        if exit_code != 0:
            self.status_label.setText(
                "%s (exit %d — see console)"
                % (self.status_label.text(), exit_code))
        # surface a failure banner when the run failed OR every draw was
        # penalized (no usable merit distribution)
        if exit_code != 0 or (n_pen and n_ok == 0):
            self._show_error_banner(exit_code, n_pen)
        else:
            self.error_banner.setVisible(False)

    def _show_error_banner(self, exit_code, n_pen):
        parts = []
        if self._first_error:
            parts.append(self._first_error)
        elif exit_code != 0:
            parts.append("Tolerance study failed (exit %d)." % exit_code)
        else:
            parts.append("No draw produced a usable merit.")
        if n_pen:
            parts.append(
                "%d draw%s penalized (failed / no usable merit)."
                % (n_pen, "" if n_pen == 1 else "s"))
        if self._first_error and "not found on spreadsheet" in self._first_error:
            parts.append(
                "Hint: a global variable must be addressed miewb_vars.<name>; "
                "a bare name resolves against the per-element `dim` sheet.")
        parts.append("See the Console tab for the full log.")
        self.error_banner.setText("  ".join(parts))
        self.error_banner.setVisible(True)

    def set_running(self, running):
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for w in (self.tol_table, self.operand_table, self.comp_group,
                  self.draws_spin, self.seed_spin, self.threshold_check,
                  self.threshold_spin, self.sens_delta_spin,
                  self.skip_sens_check, self.hist_bins_spin,
                  self.preset_combo, self.rays_spin, self.backend_combo,
                  self.tol_add_btn, self.tol_del_btn,
                  self.operand_add_btn, self.operand_del_btn):
            w.setEnabled(not running)
        if not running:
            # re-arm the threshold spin's enabled state (blanket-disable
            # above would otherwise leave it live with the box unchecked)
            self.threshold_spin.setEnabled(
                self.threshold_check.isChecked())
