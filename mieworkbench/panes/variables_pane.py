"""VariablesPane - a dock-pane editor over the miewb_vars sweep-variables
sheet (see mieworkbench.core.variables for the sheet contract and
train_solver/permute_model semantics this must never drift from).

Idioms mirror panes/train_editor.py (see docs/UI_TESTING.md and that
module's docstring):
  * every mutation goes through the Project API
    (ensure_variables_sheet/apply_variable_cells) - never a raw
    set_cell/set_property call from this module;
  * the Value column shows the STORED expression verbatim, appending the
    evaluated value in parentheses for DISPLAY only ("gap*2  (= 50.0)");
    the EditRole is the bare expression;
  * every mutating action is reachable through a dialog-free method
    (add_variable / remove_variable / commit_field / set_sweep_enabled)
    so the offscreen test suite can drive it without a real dialog;
  * errors surface in a bottom status label, never a modal.
"""

import re
from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QHBoxLayout, QInputDialog, QLabel,
    QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from ..core import variables as V
import train_solver  # noqa: E402  (scripts/ already on sys.path via the
                      # core.variables import above)

# columns
COL_NAME = 0
COL_VALUE = 1
COL_MIN = 2
COL_MAX = 3
COL_STEPS = 4
COL_SWEEP = 5
COL_COMMENT = 6
_HEADERS = ["Name", "Value", "Min", "Max", "Steps", "Sweep", "Comment"]

_RED = QColor("#c0392b")

# clearing a row means blanking every column of its spreadsheet row
_ROW_COLUMN_LETTERS = "ABCDEF"


def _fmt(val):
    """Compact numeric display: keep a trailing .0 on whole numbers, else
    a %g-style rendering (mirrors train_editor._fmt)."""
    try:
        val = float(val)
    except (TypeError, ValueError):
        return str(val)
    if val == int(val):
        return "%.1f" % val
    return "%g" % val


def _value_display(row):
    """DISPLAY text for a VarRow's value cell: the bare expression for a
    plain number, else "<expr>  (= <value>)"."""
    try:
        float(row.value_raw)
        return row.value_raw
    except (TypeError, ValueError):
        pass
    if row.value is None:
        return row.value_raw
    return "%s  (= %s)" % (row.value_raw, _fmt(row.value))


class _DualItem(QTableWidgetItem):
    """A cell with distinct DisplayRole and EditRole strings (a plain
    QTableWidgetItem collapses the two): the Value column shows
    'gap*2  (= 50.0)' but edits as the bare 'gap*2'."""

    def __init__(self, edit_val, disp_val=None):
        super().__init__(disp_val if disp_val is not None else edit_val)
        self._edit = edit_val

    def data(self, role):
        if role == Qt.EditRole:
            return self._edit
        return super().data(role)

    def setData(self, role, value):
        if role == Qt.EditRole:
            self._edit = value       # capture a committed edit
        super().setData(role, value)  # still emits itemChanged


class VariablesPane(QWidget):
    """Sweep-variables editor dock.

    Public API the mainwindow (or a test) drives:
      * VariablesPane(project, parent=None)
      * add_variable(name, value="0", vmin=None, vmax=None, nstep=None,
        enabled=False, comment="") -> bool
      * remove_variable(name) -> bool
      * commit_field(name, field, text) -> bool, field one of
        "value"/"vmin"/"vmax"/"nstep"/"comment"
      * set_sweep_enabled(name, enabled) -> bool
      * refresh()
      * sweep_mode property ("product"/"zip"; GUI run-config only, never
        written to the sheet - the mainwindow persists it into simparams)
      * sweep_spec() / run_count() / estimate_text(single_run_s) /
        has_enabled_sweep()
      * item_for(name, column) - the QTableWidgetItem for a variable's
        row (None if unknown), for test assertions
    """

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self._project = project
        self._updating = False        # populating the table (block echoes)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(self._build_toolbar())

        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(_HEADERS)
        # Value column accepts expressions over the other variables --
        # advertise the one authoritative grammar string.
        self.table.horizontalHeaderItem(1).setToolTip(
            "Value, or an expression over the other variables.\n\n%s"
            % train_solver.EXPR_HELP)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self.table)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        self._connect_project()
        self.refresh()

    # -- wiring ---------------------------------------------------------------
    def _connect_project(self):
        p = self._project
        if p is None:
            return
        p.sceneLoaded.connect(self.refresh)
        p.propertiesChanged.connect(self.refresh)

    def _build_toolbar(self):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        self.btn_add = QToolButton()
        self.btn_add.setText("Add variable")
        self.btn_add.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_add.clicked.connect(self._on_add_clicked)
        row.addWidget(self.btn_add)

        self.btn_remove = QToolButton()
        self.btn_remove.setText("Remove")
        self.btn_remove.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_remove.clicked.connect(self._on_remove_clicked)
        row.addWidget(self.btn_remove)

        row.addWidget(QLabel("Sweep mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Product (all combinations)", "product")
        self._mode_combo.addItem("Zip (paired)", "zip")
        row.addWidget(self._mode_combo)

        row.addStretch(1)
        return row

    # -- sweep_mode: GUI run-config only, never written to the sheet ----------
    @property
    def sweep_mode(self):
        return self._mode_combo.currentData()

    @sweep_mode.setter
    def sweep_mode(self, mode):
        i = self._mode_combo.findData(str(mode))
        if i >= 0:
            self._mode_combo.setCurrentIndex(i)

    # -- data access ------------------------------------------------------------
    def _current_rows(self):
        sheet = self._project.variables_sheet() if self._project else None
        return V.parse_sheet(sheet)

    def sweep_spec(self):
        return V.sweep_spec(self._current_rows())

    def run_count(self):
        return V.run_count(self._current_rows(), self.sweep_mode)

    def estimate_text(self, single_run_s):
        return V.estimate_sweep(self._current_rows(), self.sweep_mode,
                                single_run_s)["text"]

    def has_enabled_sweep(self):
        return any(r.enabled for r in self._current_rows().values())

    # -- status / errors --------------------------------------------------------
    def _set_error(self, msg):
        self.status.setStyleSheet("color: #c0392b;")
        self.status.setText(str(msg))

    def _set_info(self, msg=""):
        self.status.setStyleSheet("color: gray;")
        self.status.setText(str(msg))

    # -- table (re)population ----------------------------------------------------
    def refresh(self, *_args):
        if self._project is None:
            return
        self._updating = True
        try:
            self.table.setRowCount(0)
            rows = self._current_rows()
            names = sorted(rows, key=lambda n: rows[n].row)
            errors = V.check_cycles(rows)
            bad = {}                  # name -> message
            for msg in errors:
                tokens = set(re.split(r"\W+", msg))
                for name in names:
                    if name in tokens:
                        bad.setdefault(name, msg)
            self.table.setRowCount(len(names))
            for i, name in enumerate(names):
                row = rows[name]
                message = bad.get(name)
                if message is None and row.value is None:
                    message = "%r could not be evaluated" % name
                self._populate_row(i, row, message)
            self.table.resizeColumnsToContents()
        finally:
            self._updating = False

    def _populate_row(self, i, row, invalid_message):
        name_item = QTableWidgetItem(row.name)
        name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.table.setItem(i, COL_NAME, name_item)

        self.table.setItem(i, COL_VALUE,
                           _DualItem(row.value_raw, _value_display(row)))
        self.table.setItem(i, COL_MIN, QTableWidgetItem(_fmt(row.vmin)))
        self.table.setItem(i, COL_MAX, QTableWidgetItem(_fmt(row.vmax)))
        self.table.setItem(i, COL_STEPS, QTableWidgetItem(str(row.nstep)))

        sweep_item = QTableWidgetItem()
        sweep_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
                            | Qt.ItemIsSelectable)
        sweep_item.setCheckState(Qt.Checked if row.enabled else Qt.Unchecked)
        self.table.setItem(i, COL_SWEEP, sweep_item)

        # column A (comment) is never aliased, so the worker echo never
        # carries its content back (see Project._cell_preimage) - blank on
        # every refresh; still write-able (see commit_field "comment").
        self.table.setItem(i, COL_COMMENT, QTableWidgetItem(""))

        if invalid_message:
            for col in range(self.table.columnCount()):
                item = self.table.item(i, col)
                item.setForeground(QBrush(_RED))
                item.setToolTip(invalid_message)

    def _row_index(self, name):
        for i in range(self.table.rowCount()):
            it = self.table.item(i, COL_NAME)
            if it is not None and it.text() == name:
                return i
        return None

    def item_for(self, name, column):
        i = self._row_index(name)
        return None if i is None else self.table.item(i, column)

    def _mark_invalid(self, name, message):
        i = self._row_index(name)
        if i is None:
            return
        self._updating = True
        try:
            for col in range(self.table.columnCount()):
                item = self.table.item(i, col)
                if item is not None:
                    item.setForeground(QBrush(_RED))
                    item.setToolTip(message)
        finally:
            self._updating = False

    # -- mutations ----------------------------------------------------------------
    def add_variable(self, name, value="0", vmin=None, vmax=None, nstep=None,
                      enabled=False, comment=""):
        name = str(name)
        err = V.validate_name(name)
        if err:
            self._set_error(err)
            return False
        rows = self._current_rows()
        if name in rows:
            self._set_error("variable %r already exists" % name)
            return False
        if vmin is None or vmax is None:
            try:
                numeric = train_solver.eval_expr(
                    value, self._project.train_variables())
            except Exception:
                numeric = 0.0
            if vmin is None:
                vmin = numeric
            if vmax is None:
                vmax = numeric
        if nstep is None:
            nstep = 0
        row = V.next_free_row(self._project.variables_sheet())
        plan = V.cell_plan(name, row=row, value=value, vmin=vmin, vmax=vmax,
                           nstep=nstep, enabled=bool(enabled),
                           comment=comment)
        try:
            self._project.apply_variable_cells(
                plan, text="Add variable %s" % name)
        except Exception as exc:
            self._set_error(str(exc))
            return False
        self._set_info()
        self.refresh()
        return True

    def remove_variable(self, name):
        name = str(name)
        rows = self._current_rows()
        row = rows.get(name)
        if row is None:
            self._set_error("no such variable %r" % name)
            return False
        plan = [{"cell": "%s%d" % (col, row.row), "raw": ""}
                for col in _ROW_COLUMN_LETTERS]
        try:
            self._project.apply_variable_cells(
                plan, text="Remove variable %s" % name)
        except Exception as exc:
            self._set_error(str(exc))
            return False
        self._set_info()
        self.refresh()
        return True

    def commit_field(self, name, field, text):
        """Commit one edited cell for variable `name`. `field` is one of
        "value"/"vmin"/"vmax"/"nstep"/"comment". Validates before
        touching the Project API; an invalid value sets the error state
        (red row + status message) and writes nothing."""
        name = str(name)
        rows = self._current_rows()
        row = rows.get(name)
        if row is None:
            self._set_error("no such variable %r" % name)
            return False
        text = "" if text is None else str(text)

        if field == "value":
            stripped = text.strip()
            if not stripped:
                msg = "value must not be empty"
                self._set_error(msg)
                self._mark_invalid(name, msg)
                return False
            candidate = dict(rows)
            candidate[name] = replace(row, value_raw=stripped)
            errors = V.check_cycles(candidate)
            if errors:
                self._set_error(errors[0])
                self._mark_invalid(name, errors[0])
                return False
            plan = V.cell_plan(name, row=row.row, value=stripped)
        elif field in ("vmin", "vmax"):
            try:
                val = float(text)
            except (TypeError, ValueError):
                msg = "%s must be numeric: %r" % (field, text)
                self._set_error(msg)
                self._mark_invalid(name, msg)
                return False
            plan = V.cell_plan(name, row=row.row, **{field: val})
        elif field == "nstep":
            stripped = text.strip()
            try:
                n = 0 if not stripped else int(float(stripped))
            except (TypeError, ValueError):
                msg = "steps must be an integer: %r" % text
                self._set_error(msg)
                self._mark_invalid(name, msg)
                return False
            plan = V.cell_plan(name, row=row.row, nstep=n)
        elif field == "comment":
            plan = V.cell_plan(name, row=row.row, comment=text)
        else:
            raise ValueError("unknown field %r" % field)

        try:
            self._project.apply_variable_cells(
                plan, text="Edit %s of %s" % (field, name))
        except Exception as exc:
            self._set_error(str(exc))
            self.refresh()
            return False
        self._set_info()
        self.refresh()
        return True

    def set_sweep_enabled(self, name, enabled):
        name = str(name)
        rows = self._current_rows()
        row = rows.get(name)
        if row is None:
            self._set_error("no such variable %r" % name)
            return False
        plan = V.cell_plan(name, row=row.row, enabled=bool(enabled))
        try:
            self._project.apply_variable_cells(
                plan, text="Toggle sweep for %s" % name)
        except Exception as exc:
            self._set_error(str(exc))
            self.refresh()
            return False
        self._set_info()
        self.refresh()
        return True

    # -- toolbar handlers (dialog-fronted; tests drive the methods above) ------
    def _on_add_clicked(self):
        name, ok = QInputDialog.getText(self, "Add variable", "Name:")
        if ok and name.strip():
            self.add_variable(name.strip())

    def _on_remove_clicked(self):
        row = self.table.currentRow()
        if row < 0:
            self._set_error("Select a variable to remove first")
            return
        item = self.table.item(row, COL_NAME)
        if item is None:
            return
        self.remove_variable(item.text())

    # -- item change routing -----------------------------------------------------
    def _on_item_changed(self, item):
        if self._updating:
            return
        col = item.column()
        name_item = self.table.item(item.row(), COL_NAME)
        if name_item is None:
            return
        name = name_item.text()
        if col == COL_VALUE:
            self.commit_field(name, "value", item.data(Qt.EditRole))
        elif col == COL_MIN:
            self.commit_field(name, "vmin", item.text())
        elif col == COL_MAX:
            self.commit_field(name, "vmax", item.text())
        elif col == COL_STEPS:
            self.commit_field(name, "nstep", item.text())
        elif col == COL_SWEEP:
            self.set_sweep_enabled(name, item.checkState() == Qt.Checked)
        elif col == COL_COMMENT:
            self.commit_field(name, "comment", item.text())
