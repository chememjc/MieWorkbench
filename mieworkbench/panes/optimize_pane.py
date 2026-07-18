"""OptimizePane - the merit-function optimizer dock (feature F1).

Full GUI parity with scripts/optimize.py: a variable table
(name/start/lo/hi), an operand table (operand/detector/target/weight), the
algorithm/budget/tolerance + fidelity settings, Run/Stop, a best-so-far
readout, and a LIVE CONVERGENCE PLOT fed by the '@MIEWB' progress events
OptimizeController parses (stage "optimize", one event per evaluation
carrying eval/budget/merit/best/params extras).

Plotting: the GUI venv ships PySide6.QtCharts, which is used when
importable; otherwise a dependency-free QPainter line plot takes over
(same ConvergencePlot API either way, so the pane and its tests never
care which backend drew the pixels). Penalized evaluations (merit >=
1e8 — optimize.py's PENALTY for failed/incomplete evals) are excluded
from axis scaling so one bad candidate cannot flatten the whole plot.

Wiring (mainwindow.py): runRequested/stopRequested out; on_progress /
on_started / on_finished in. The pane owns NO QProcess — that is
core/optimize_controller.py's job, mirroring the ConsolePane/RunController
split.
"""

import argparse
import re
import sys
from os.path import dirname, join, normpath

sys.path.insert(0, normpath(join(dirname(__file__), "..", "..",
                                 "scripts")))
import cli_specs  # noqa: E402  (stdlib-only; OPTIMIZE_OPERANDS + parsers)
import common     # noqa: E402  (stdlib-only; PRESETS)

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core.variables import qualify_var_name  # noqa: E402

# Backend error lines worth surfacing on the pane's failure banner. The
# PermuteError "alias '<x>' not found on spreadsheet 'dim'" (the classic
# unqualified-variable failure) is the first-class case; the rest catch
# generic tracebacks/exceptions so a run that dies for any reason still
# gets its first substantive line pulled up out of the console.
_ERROR_LINE_RE = re.compile(
    r"alias '.*?' not found on spreadsheet"
    r"|not found on spreadsheet"
    r"|PermuteError"
    r"|Traceback \(most recent call last\)"
    r"|\bError\b|\bERROR\b|\bException\b|\bFAILED\b")

# tooltip shared by the variable name combos of both panes — spells out
# how permute_model.split_var resolves each name form
NAME_COMBO_TOOLTIP = (
    "Design variable name. Resolution (permute_model.split_var):\n"
    "  bare 'alias'          -> the per-element `dim` sheet\n"
    "  'sheetlabel.alias'    -> that named sheet (e.g. dim_Lens1.ct)\n"
    "  'miewb_vars.<name>'   -> a global variable\n"
    "Names listed in this dropdown are miewb_vars globals; they are "
    "emitted sheet-qualified (miewb_vars.<name>) automatically. Type "
    "any other spreadsheet cell alias for a per-element parameter.")

try:
    from PySide6.QtCharts import (QChart, QChartView, QLineSeries,
                                  QScatterSeries, QValueAxis)
    HAVE_QTCHARTS = True
except ImportError:            # pragma: no cover - GUI venv ships QtCharts
    HAVE_QTCHARTS = False

# merits at/above this are penalty sentinels (optimize.PENALTY) — plotted
# data excludes them from axis scaling
PENALTY_FLOOR = 1e8


def variable_bounds(varrow):
    """(value, lo, hi) auto-fill numbers for one miewb_vars variable
    (a core.variables.VarRow).

    Uses the sheet's __min/__max sweep bounds when they are real
    (vmin < vmax). parse_sheet writes vmin == vmax == value when the
    sheet carries no __min/__max meta, so a degenerate band means
    "unspecified": fall back to value ± 10 % (± 0.1 when the value is 0,
    so a zero-valued variable still gets a usable band)."""
    try:
        value = float(varrow.value)
    except (TypeError, ValueError):
        value = 0.0
    try:
        lo, hi = float(varrow.vmin), float(varrow.vmax)
    except (TypeError, ValueError):
        lo = hi = value
    if not lo < hi:
        delta = abs(value) * 0.1 or 0.1
        lo, hi = value - delta, value + delta
    return value, lo, hi


def match_error_line(text):
    """The line `text` verbatim if it looks like a substantive backend
    error worth surfacing (matches _ERROR_LINE_RE), else None. Shared by
    both panes' on_line() so the pattern lives in one place."""
    return text if _ERROR_LINE_RE.search(text or "") else None


def _make_error_banner():
    """A styled, word-wrapped, hidden-by-default QLabel used by both
    panes to surface the first backend error prominently (NOT a modal —
    offscreen-test discipline, see CLAUDE.md)."""
    banner = QLabel()
    banner.setWordWrap(True)
    banner.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    banner.setStyleSheet(
        "QLabel { background: #7f1d1d; color: #fee2e2; border: 1px solid "
        "#f87171; border-radius: 4px; padding: 6px 8px; }")
    banner.setVisible(False)
    return banner


# =============================================================================
# Convergence plot (QtCharts when available, QPainter fallback)
# =============================================================================
class ConvergencePlot(QWidget):
    """Per-eval merit points + a best-so-far line. add_point()/clear();
    the data lives on this wrapper so tests are backend-agnostic."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._evals = []
        self._merits = []
        self._bests = []
        self.setMinimumHeight(160)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if HAVE_QTCHARTS:
            self._merit_series = QScatterSeries()
            self._merit_series.setName("merit")
            self._merit_series.setMarkerSize(6.0)
            self._best_series = QLineSeries()
            self._best_series.setName("best so far")
            self._chart = QChart()
            self._chart.addSeries(self._merit_series)
            self._chart.addSeries(self._best_series)
            self._chart.legend().setVisible(True)
            self._ax_x = QValueAxis()
            self._ax_x.setTitleText("evaluation")
            self._ax_x.setLabelFormat("%d")
            self._ax_y = QValueAxis()
            self._ax_y.setTitleText("merit")
            self._chart.addAxis(self._ax_x, Qt.AlignmentFlag.AlignBottom)
            self._chart.addAxis(self._ax_y, Qt.AlignmentFlag.AlignLeft)
            for s in (self._merit_series, self._best_series):
                s.attachAxis(self._ax_x)
                s.attachAxis(self._ax_y)
            view = QChartView(self._chart)
            view.setRenderHint(QPainter.RenderHint.Antialiasing)
            layout.addWidget(view)
        else:
            self._canvas = _PainterPlot(self)
            layout.addWidget(self._canvas)

    # -- data API ----------------------------------------------------------------
    def add_point(self, eval_i, merit, best):
        self._evals.append(int(eval_i))
        self._merits.append(float(merit))
        self._bests.append(None if best is None else float(best))
        if HAVE_QTCHARTS:
            if merit < PENALTY_FLOOR:
                self._merit_series.append(QPointF(eval_i, merit))
            if best is not None and best < PENALTY_FLOOR:
                self._best_series.append(QPointF(eval_i, best))
            self._rescale()
        else:
            self._canvas.update()

    def clear(self):
        self._evals, self._merits, self._bests = [], [], []
        if HAVE_QTCHARTS:
            self._merit_series.clear()
            self._best_series.clear()
        else:
            self._canvas.update()

    def point_count(self):
        return len(self._evals)

    def plotted_merits(self):
        """The merits that participate in the plot/axis scaling (penalty
        sentinels excluded)."""
        return [m for m in self._merits if m < PENALTY_FLOOR]

    def penalized_count(self):
        """How many recorded evaluations were penalty sentinels (merit >=
        PENALTY_FLOOR) — kept out of the merit series so autoscale still
        works, surfaced instead as a count."""
        return sum(1 for m in self._merits if m >= PENALTY_FLOOR)

    # -- QtCharts axis upkeep ------------------------------------------------------
    def _rescale(self):
        if not self._evals:
            return
        ys = self.plotted_merits()
        ys += [b for b in self._bests if b is not None
               and b < PENALTY_FLOOR]
        self._ax_x.setRange(0.0, max(self._evals) + 1.0)
        if ys:
            lo, hi = min(ys), max(ys)
            pad = (hi - lo) * 0.08 or (abs(hi) * 0.1 + 1e-12)
            self._ax_y.setRange(lo - pad, hi + pad)


class _PainterPlot(QWidget):
    """Dependency-free fallback: axes + merit dots + best-so-far line."""

    def __init__(self, plot):
        super().__init__(plot)
        self._plot = plot
        self.setMinimumHeight(150)

    def paintEvent(self, _event):    # pragma: no cover - fallback backend
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1e1e1e"))
        margin = 28
        area = QRectF(margin, 8, self.width() - margin - 8,
                      self.height() - margin - 8)
        p.setPen(QPen(QColor("#6b7280")))
        p.drawRect(area)
        evals = self._plot._evals
        merits = self._plot._merits
        bests = self._plot._bests
        pts = [(e, m) for e, m in zip(evals, merits) if m < PENALTY_FLOOR]
        line = [(e, b) for e, b in zip(evals, bests)
                if b is not None and b < PENALTY_FLOOR]
        ys = [m for _, m in pts] + [b for _, b in line]
        if not ys:
            p.drawText(area, Qt.AlignmentFlag.AlignCenter,
                       "no evaluations yet")
            return
        x_hi = max(evals) + 1.0
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.08 or (abs(hi) * 0.1 + 1e-12)
        lo, hi = lo - pad, hi + pad

        def to_xy(e, y):
            fx = area.left() + area.width() * (e / x_hi)
            fy = area.bottom() - area.height() * ((y - lo) / (hi - lo))
            return QPointF(fx, fy)

        p.setPen(QPen(QColor("#22d3ee"), 2))
        prev = None
        for e, b in line:
            cur = to_xy(e, b)
            if prev is not None:
                p.drawLine(prev, cur)
            prev = cur
        p.setPen(QPen(QColor("#e879f9"), 5))
        for e, m in pts:
            p.drawPoint(to_xy(e, m))
        p.setPen(QPen(QColor("#9ca3af")))
        p.drawText(4, int(area.top()) + 10, "%.3g" % hi)
        p.drawText(4, int(area.bottom()), "%.3g" % lo)


# =============================================================================
# the pane
# =============================================================================
VAR_HEADERS = ["Variable", "Start", "Lo", "Hi"]
OPERAND_HEADERS = ["Operand", "Detector (optional)", "Target", "Weight"]


class OptimizePane(QWidget):
    runRequested = Signal()
    stopRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._varrows = {}     # {name: core.variables.VarRow} from the scene
        self._first_error = None   # first backend error line this run
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.error_banner = _make_error_banner()
        layout.addWidget(self.error_banner)

        tables_row = QHBoxLayout()
        tables_row.addWidget(self._build_var_group(), 1)
        tables_row.addWidget(self._build_operand_group(), 1)
        layout.addLayout(tables_row)

        layout.addLayout(self._build_settings_row())
        layout.addLayout(self._build_run_row())

        self.plot = ConvergencePlot()
        layout.addWidget(self.plot, 1)

    # -- construction ------------------------------------------------------------
    def _build_var_group(self):
        group = QGroupBox("Optimization variables")
        v = QVBoxLayout(group)
        self.var_table = QTableWidget(0, len(VAR_HEADERS))
        self.var_table.setHorizontalHeaderLabels(VAR_HEADERS)
        self.var_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.var_table.setToolTip(
            "Design variables: spreadsheet cell aliases (bare 'alias' on "
            "the dim sheet, 'sheetlabel.alias', or a global picked from "
            "the dropdown = emitted miewb_vars.<name>), with the start "
            "value and bounds in mm")
        v.addWidget(self.var_table)
        row = QHBoxLayout()
        self.var_add_btn = QPushButton("Add")
        self.var_add_btn.setToolTip("Add a variable row")
        self.var_add_btn.clicked.connect(lambda: self.add_variable())
        self.var_del_btn = QPushButton("Remove")
        self.var_del_btn.setToolTip("Remove the selected variable row")
        self.var_del_btn.clicked.connect(
            lambda: self._remove_current(self.var_table))
        row.addWidget(self.var_add_btn)
        row.addWidget(self.var_del_btn)
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
            "Merit operands: spot_rms/focus and encircled_energy are "
            "minimized, detected_power and mtf50 maximized; a raw "
            "flattened report.json merit key (contains a '.') is "
            "minimized toward its target. Weight scales each term.")
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

    def _build_settings_row(self):
        grid = QGridLayout()
        grid.addWidget(QLabel("Algorithm:"), 0, 0)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["local", "global"])
        self.algorithm_combo.setToolTip(
            "local = scipy Nelder-Mead within the bounds; global = "
            "nevergrad CMA-ES")
        grid.addWidget(self.algorithm_combo, 0, 1)

        grid.addWidget(QLabel("Budget:"), 0, 2)
        self.budget_spin = QSpinBox()
        self.budget_spin.setRange(1, 1000000)
        self.budget_spin.setValue(40)
        self.budget_spin.setToolTip("Maximum merit evaluations")
        grid.addWidget(self.budget_spin, 0, 3)

        grid.addWidget(QLabel("Tolerance:"), 0, 4)
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setDecimals(9)
        self.tol_spin.setRange(0.0, 1e6)
        self.tol_spin.setValue(1e-3)
        self.tol_spin.setToolTip(
            "Local-algorithm merit convergence tolerance (scipy fatol)")
        grid.addWidget(self.tol_spin, 0, 5)

        grid.addWidget(QLabel("Preset:"), 1, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(sorted(common.PRESETS))
        self.preset_combo.setCurrentText("quick")
        self.preset_combo.setToolTip(
            "fast_eval fidelity preset per evaluation")
        grid.addWidget(self.preset_combo, 1, 1)

        grid.addWidget(QLabel("Rays/eval:"), 1, 2)
        self.rays_spin = QDoubleSpinBox()
        self.rays_spin.setDecimals(0)
        self.rays_spin.setRange(0, 1e9)
        self.rays_spin.setValue(0)
        self.rays_spin.setSpecialValueText("preset")
        self.rays_spin.setToolTip(
            "Primary rays per source per evaluation (0 = preset default)")
        grid.addWidget(self.rays_spin, 1, 3)

        grid.addWidget(QLabel("Backend:"), 1, 4)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["worker", "full"])
        self.backend_combo.setToolTip(
            "fast_eval backend: worker = persistent FreeCAD (fast), "
            "full = fresh pipeline per eval (reference)")
        grid.addWidget(self.backend_combo, 1, 5)

        self.final_coherent_check = QCheckBox("Final coherent re-eval")
        self.final_coherent_check.setChecked(True)
        self.final_coherent_check.setToolTip(
            "Re-evaluate the best design once with source coherence as "
            "authored (the inner loop always runs incoherent) for a "
            "faithful final number")
        grid.addWidget(self.final_coherent_check, 1, 6)
        grid.setColumnStretch(7, 1)
        return grid

    def _build_run_row(self):
        row = QHBoxLayout()
        self.run_btn = QPushButton("Run Optimization")
        self.run_btn.setToolTip(
            "Launch scripts/optimize.py on the open model")
        self.run_btn.clicked.connect(self.runRequested.emit)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop the running optimization")
        self.stop_btn.clicked.connect(self.stopRequested.emit)
        self.best_label = QLabel("No optimization run yet")
        self.best_label.setToolTip("Best merit and parameters so far")
        row.addWidget(self.run_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.best_label, 1)
        return row

    # -- scene variables (miewb_vars dropdown + auto-fill) ------------------------
    def set_variables(self, varrows):
        """Publish the scene's miewb_vars variables ({name: VarRow} from
        core.variables.parse_sheet) into every row's name dropdown. The
        combos stay editable so aliases outside miewb_vars (dim-sheet
        cells like 'dim.ct') can still be typed — an empty dict simply
        leaves a plain editable combo."""
        self._varrows = dict(varrows or {})
        for row in range(self.var_table.rowCount()):
            combo = self.var_table.cellWidget(row, 0)
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

    def _on_name_chosen(self, combo, text):
        """A row's name-combo changed: when the name is a known
        miewb_vars variable, auto-fill that row's start/lo/hi (start =
        the sheet's current value; bounds = __min/__max when real, else
        value ± 10 %)."""
        varrow = self._varrows.get(text)
        if varrow is None:
            return
        row = self._combo_row(self.var_table, combo)
        if row is None:
            return
        start, lo, hi = variable_bounds(varrow)
        for col, val in ((1, start), (2, lo), (3, hi)):
            self.var_table.setItem(row, col, QTableWidgetItem("%g" % val))

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

    def add_variable(self, name="", start=None, lo=None, hi=None):
        """Insert a variable row. Explicit start/lo/hi are honored; any
        left as None auto-fill from the (known) named miewb_vars variable
        via variable_bounds, else fall back to 0 / -1 / 1. A blank name
        adopts the combo's default choice (the first scene variable)."""
        table = self.var_table
        row = table.rowCount()
        table.insertRow(row)
        combo = self._make_name_combo(name)
        table.setCellWidget(row, 0, combo)
        if not name:
            name = combo.currentText()   # first scene variable, if any
        fill = (0.0, -1.0, 1.0)
        if name in self._varrows:
            fill = variable_bounds(self._varrows[name])
        for col, (explicit, auto) in enumerate(zip((start, lo, hi), fill),
                                               start=1):
            val = auto if explicit is None else explicit
            table.setItem(row, col, QTableWidgetItem("%g" % val))
        # connect AFTER seeding the cells so explicit values always win
        combo.currentTextChanged.connect(
            lambda text, c=combo: self._on_name_chosen(c, text))
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
    def variables(self):
        """Table rows -> 'name:start:lo:hi' spec strings (rows with an
        empty name are skipped; non-numeric cells raise ValueError with
        the row named)."""
        out = []
        for row in range(self.var_table.rowCount()):
            name = self._row_name(self.var_table, row)
            if not name:
                continue
            nums = []
            for col in (1, 2, 3):
                text = self._cell_text(self.var_table, row, col)
                try:
                    nums.append(float(text))
                except ValueError:
                    raise ValueError(
                        "variable row %d (%s): %r is not a number"
                        % (row + 1, name, text))
            out.append("%s:%s:%s:%s" % (
                qualify_var_name(name, self._varrows), "%g" % nums[0],
                "%g" % nums[1], "%g" % nums[2]))
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

    def config(self):
        """The controller/build_args config dict (cli_specs 'optimize'
        dests). Raises ValueError on malformed table cells."""
        cfg = {
            "var": self.variables(),
            "operand": self.operands(),
            "algorithm": self.algorithm_combo.currentText(),
            "budget": self.budget_spin.value(),
            "tol": self.tol_spin.value(),
            "preset": self.preset_combo.currentText(),
            "eval_backend": self.backend_combo.currentText(),
            "no_final_coherent": not self.final_coherent_check.isChecked(),
        }
        if self.rays_spin.value() > 0:
            cfg["rays"] = float(self.rays_spin.value())
        return cfg

    def apply_config(self, cfg):
        """Rebuild the variable/operand tables and settings from a
        config() dict (as persisted by Project.set_optimize_config /
        returned by get_optimize_config). Spec strings that fail to
        parse are skipped defensively -- a hand-edited or stale document
        must never block opening the scene. Round-trips: config() ->
        apply_config() -> config() reproduces the same dict."""
        if not cfg:
            return
        self.var_table.setRowCount(0)
        for spec in cfg.get("var") or []:
            try:
                v = cli_specs.parse_var_spec(spec)
            except argparse.ArgumentTypeError:
                continue
            self.add_variable(v["name"], v["start"], v["lo"], v["hi"])
        self.operand_table.setRowCount(0)
        for spec in cfg.get("operand") or []:
            try:
                o = cli_specs.parse_operand_spec(spec)
            except argparse.ArgumentTypeError:
                continue
            self.add_operand(o["operand"], o["detector"] or "",
                             o["target"], o["weight"])
        if "algorithm" in cfg:
            self.algorithm_combo.setCurrentText(str(cfg["algorithm"]))
        if "budget" in cfg:
            self.budget_spin.setValue(int(cfg["budget"]))
        if "tol" in cfg:
            self.tol_spin.setValue(float(cfg["tol"]))
        if "preset" in cfg:
            self.preset_combo.setCurrentText(str(cfg["preset"]))
        if "eval_backend" in cfg:
            self.backend_combo.setCurrentText(str(cfg["eval_backend"]))
        self.final_coherent_check.setChecked(
            not cfg.get("no_final_coherent", False))
        self.rays_spin.setValue(float(cfg.get("rays") or 0.0))

    # -- run-state / progress slots ----------------------------------------------------
    def on_started(self):
        self.plot.clear()
        self._first_error = None
        self.error_banner.setVisible(False)
        self.error_banner.clear()
        self.best_label.setText("Optimizing…")
        self.set_running(True)

    def on_line(self, text):
        """A raw stdout/stderr line from the optimizer process (wired
        alongside the console feed). Latches the FIRST substantive error
        line so on_finished can surface it on the banner even though the
        merit stream itself only carries penalty sentinels."""
        if self._first_error is None:
            hit = match_error_line(text)
            if hit is not None:
                self._first_error = hit.strip()

    def on_progress(self, event):
        if event.get("stage") != "optimize" or "eval" not in event:
            return
        merit = event.get("merit")
        best = event.get("best")
        if merit is None:
            return
        self.plot.add_point(event["eval"], merit, best)
        if best is not None:
            params = event.get("best_params") or {}
            ptxt = "  ".join("%s=%.6g" % (k, v)
                             for k, v in sorted(params.items()))
            self.best_label.setText(
                "Best merit %.6g after %s eval(s)   %s"
                % (best, event["eval"], ptxt))

    def on_finished(self, exit_code):
        self.set_running(False)
        n_pen = self.plot.penalized_count()
        n_ok = len(self.plot.plotted_merits())
        if exit_code != 0:
            self.best_label.setText(
                "%s (exit %d — see console)"
                % (self.best_label.text(), exit_code))
        # surface a failure banner when the run failed OR produced no
        # usable merit (every evaluation penalized) — the silent-empty-
        # graph case the bare-name bug used to hit
        if exit_code != 0 or (n_pen and n_ok == 0):
            self._show_error_banner(exit_code, n_pen)
        else:
            self.error_banner.setVisible(False)

    def _show_error_banner(self, exit_code, n_pen):
        parts = []
        if self._first_error:
            parts.append(self._first_error)
        elif exit_code != 0:
            parts.append("Optimization failed (exit %d)." % exit_code)
        else:
            parts.append("No evaluation produced a usable merit.")
        if n_pen:
            parts.append(
                "%d evaluation%s penalized (failed / no usable merit)."
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
        for w in (self.var_table, self.operand_table,
                  self.algorithm_combo, self.budget_spin, self.tol_spin,
                  self.preset_combo, self.rays_spin, self.backend_combo,
                  self.final_coherent_check, self.var_add_btn,
                  self.var_del_btn, self.operand_add_btn,
                  self.operand_del_btn):
            w.setEnabled(not running)
