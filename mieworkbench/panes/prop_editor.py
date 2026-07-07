"""prop_editor.py -- PropEditorPane: the "editor mode" for the optical
property library (materials / coatings / polarizers / filters / gratings /
birefringence).

One QTabWidget, one tab per registry. Each tab (_CategoryEditor) shows the
registry's rows in a QTableWidget (columns = the registry csv's own header,
so a new column added to a registry shows up here automatically); a library
selector combo at the top of the pane switches every tab between the
system library (core.librarymgr.LibraryManager.system_lib) and the project
library (.project_lib, disabled when no project root is set).

Cells are read-only until "Edit" is toggled on. Commit path: gather the
table's current text back into row dicts, refuse if any 'reference'
(citation) cell is empty, atomically rewrite the registry csv (tmp +
os.replace), then validate the result through PropLibrary.validate() (the
real load_optical_properties loader) via core.proplib.Transaction /
validate_and_commit -- on failure every touched file is rolled back and the
loader's own error message is surfaced.

Selecting a row that references a spectral table (nk_file / table /
table_csv column) plots it with PySide6.QtCharts (one QLineSeries per value
column, first column on the x-axis) -- no matplotlib anywhere in the GUI
process.

"Import table..." maps an arbitrary external csv's columns onto the
category's required table schema (TABLE_SCHEMA below); the mapping itself
is applied by the free function apply_column_mapping() so it is unit
testable without going through the (modal) mapping dialog.
"""
import csv
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from PySide6.QtCharts import QChart, QChartView, QLineSeries
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from ..core.proplib import CATEGORY_INFO, LibraryWriteError, Transaction, \
    validate_and_commit

REQUIRED_COLUMN = "reference"
MISSING_REFERENCE_COLOR = QColor(120, 30, 30)

# Registry -> (tab label, table-file schema for Import-table / plotting).
# Schema is None for categories whose rows don't reference a spectral table
# (birefringence rows reference other materials.csv rows, not a file).
CATEGORY_TABS = (
    ("materials", "Materials", ("wavelength_nm", "n", "k")),
    ("coatings", "Coatings", ("wavelength_nm", "Rs", "Rp", "Ts", "Tp")),
    ("polarizers", "Polarizers",
     ("wavelength_nm", "T_parallel", "T_perpendicular")),
    ("filters", "Filters", ("wavelength_nm", "transmittance_internal")),
    ("gratings", "Gratings", ("wavelength_nm", "order", "eta_s", "eta_p")),
    ("uniaxial", "Birefringence", None),
)
TABLE_SCHEMA = {cat: schema for cat, _, schema in CATEGORY_TABS}


class PropEditorError(RuntimeError):
    pass


def apply_column_mapping(src_headers, src_rows, mapping, required_cols):
    """Pure mapping-application function (kept free of Qt so it is testable
    without a modal dialog).

    src_headers: column names of the imported csv.
    src_rows: list of dict rows (csv.DictReader output) from that csv.
    mapping: {dest_col: src_col} for every entry in required_cols.
    required_cols: the destination schema's column names, in order.

    -> (dest_headers, dest_rows) where dest_rows is a list of tuples of
    floats, ready to write out with csv.writer. Raises ValueError naming
    the problem (missing mapping / unknown source column / non-numeric
    cell)."""
    missing = [c for c in required_cols if c not in mapping]
    if missing:
        raise ValueError("no source column mapped for required column(s) %s"
                         % missing)
    for dest, src in mapping.items():
        if src not in src_headers:
            raise ValueError("mapped source column %r not found in the "
                             "imported file" % src)
    dest_headers = list(required_cols)
    dest_rows = []
    for i, src_row in enumerate(src_rows):
        try:
            dest_rows.append(tuple(float(src_row[mapping[c]])
                                   for c in dest_headers))
        except (TypeError, ValueError) as exc:
            raise ValueError("imported row %d: %s" % (i + 2, exc))
    return dest_headers, dest_rows


def _atomic_write_registry(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    import os
    os.replace(tmp, path)


def _find_row(rows, name):
    name_l = (name or "").strip().lower()
    for row in rows:
        if (row.get("name") or "").strip().lower() == name_l:
            return row
    return None


class _CategoryEditor(QWidget):
    """One registry tab: table + edit/add/delete/import controls + a
    QtCharts plot of the selected row's spectral table, if any."""

    rowCommitted = Signal()

    def __init__(self, category, manager, parent=None):
        super().__init__(parent)
        self.category = category
        self.manager = manager
        self.which_library = "system"
        self._fieldnames = []
        self._edit_mode = False

        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        self.edit_toggle = QToolButton()
        self.edit_toggle.setText("Edit")
        self.edit_toggle.setCheckable(True)
        self.edit_toggle.toggled.connect(self.set_edit_mode)

        self.add_button = QPushButton("Add row")
        self.add_button.clicked.connect(self.add_row)
        self.delete_button = QPushButton("Delete row")
        self.delete_button.clicked.connect(self._delete_selected_row)
        self.commit_button = QPushButton("Save changes")
        self.commit_button.clicked.connect(self._on_commit_clicked)
        self.import_button = QPushButton("Import table...")
        self.import_button.clicked.connect(self._on_import_clicked)
        self.import_button.setEnabled(TABLE_SCHEMA[category] is not None)
        self.copy_to_system_button = QPushButton("Copy to system library")
        self.copy_to_system_button.clicked.connect(
            self._on_copy_to_system_clicked)
        self.copy_to_system_button.setEnabled(False)

        toolbar = QHBoxLayout()
        for w in (self.edit_toggle, self.add_button, self.delete_button,
                 self.commit_button, self.import_button,
                 self.copy_to_system_button):
            toolbar.addWidget(w)
        toolbar.addStretch(1)

        self.chart_view = QChartView(QChart())
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setMinimumHeight(200)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self.table, stretch=2)
        layout.addWidget(self.chart_view, stretch=1)

        self.reload()

    # -- library selection -------------------------------------------------
    def current_lib(self):
        if self.which_library == "project":
            return self.manager.project_lib
        return self.manager.system_lib

    def set_library(self, which):
        self.which_library = which
        self.copy_to_system_button.setEnabled(which == "project")
        self.reload()

    # -- loading / rendering -------------------------------------------------
    def reload(self):
        lib = self.current_lib()
        if lib is None:
            self._fieldnames = []
            rows = []
        else:
            path = lib.registry_path(self.category)
            if path.exists():
                self._fieldnames = lib.registry_fieldnames(self.category)
            rows = lib.registry_rows(self.category)
        self._populate_table(rows)

    def _populate_table(self, rows):
        self.table.blockSignals(True)
        self.table.setColumnCount(len(self._fieldnames))
        self.table.setHorizontalHeaderLabels(self._fieldnames)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, col in enumerate(self._fieldnames):
                value = row.get(col, "") or ""
                item = QTableWidgetItem(value)
                if col == REQUIRED_COLUMN and not value.strip():
                    item.setBackground(MISSING_REFERENCE_COLOR)
                    item.setToolTip(
                        "reference (citation) is required -- every "
                        "registry row must name its data source before it "
                        "can be saved")
                if not self._edit_mode:
                    item.setFlags(item.flags()
                                 & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, item)
        self.table.blockSignals(False)
        self.chart_view.setChart(QChart())

    def set_edit_mode(self, on):
        self._edit_mode = bool(on)
        rows = self.rows_from_table()
        self._populate_table(rows)

    def row_count(self):
        return self.table.rowCount()

    def rows_from_table(self):
        rows = []
        for r in range(self.table.rowCount()):
            row = {}
            for c, col in enumerate(self._fieldnames):
                item = self.table.item(r, c)
                row[col] = item.text() if item is not None else ""
            rows.append(row)
        return rows

    def selected_row_dict(self):
        rows = self.table.selectionModel().selectedRows() \
            if self.table.selectionModel() else []
        if not rows:
            return None
        return self.rows_from_table()[rows[0].row()]

    # -- add / delete --------------------------------------------------------
    def add_row(self):
        rows = self.rows_from_table()
        rows.append({c: "" for c in self._fieldnames})
        self._edit_mode = True
        self.edit_toggle.setChecked(True)
        self._populate_table(rows)

    def _delete_selected_row(self):
        rows = self.table.selectionModel().selectedRows() \
            if self.table.selectionModel() else []
        if not rows:
            return
        self.delete_row(rows[0].row())

    def delete_row(self, index):
        rows = self.rows_from_table()
        del rows[index]
        self._populate_table(rows)

    # -- commit --------------------------------------------------------------
    def commit(self):
        """Write the table's current content to the registry csv, validate
        through the real loader, roll back + raise on failure. Raises
        PropEditorError for a blocked commit (empty reference cell) without
        touching any file."""
        lib = self.current_lib()
        if lib is None:
            raise PropEditorError("no project library to save to")
        rows = self.rows_from_table()
        blank_refs = [row.get("name", "<unnamed>") for row in rows
                     if not (row.get(REQUIRED_COLUMN) or "").strip()]
        if blank_refs:
            raise PropEditorError(
                "cannot save: row(s) %s are missing a 'reference' "
                "citation" % ", ".join(repr(n) for n in blank_refs))
        path = lib.registry_path(self.category)
        txn = Transaction()
        txn.track(path)
        try:
            _atomic_write_registry(path, self._fieldnames, rows)
            validate_and_commit(lib, txn)
        except LibraryWriteError:
            raise
        except Exception:
            txn.rollback()
            raise
        self.reload()
        self.rowCommitted.emit()
        return True

    def _on_commit_clicked(self):
        try:
            self.commit()
        except (PropEditorError, LibraryWriteError) as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    # -- plotting --------------------------------------------------------------
    def _on_selection_changed(self):
        row = self.selected_row_dict()
        self._update_plot(row)

    def _table_filename_for_row(self, row):
        info = CATEGORY_INFO[self.category]
        if row is None or not info["file_dir"]:
            return None
        for col in info["file_cols"]:
            fname = (row.get(col) or "").strip()
            if fname:
                return fname
        return None

    def _update_plot(self, row):
        fname = self._table_filename_for_row(row)
        lib = self.current_lib()
        if not fname or lib is None:
            self.chart_view.setChart(QChart())
            return
        try:
            headers, rows = lib.table_data(self.category, fname)
        except Exception:
            self.chart_view.setChart(QChart())
            return
        self._render_chart(headers, rows, row.get("name", ""))

    def _render_chart(self, headers, rows, title):
        chart = QChart()
        chart.setTitle(title)
        for i, col in enumerate(headers[1:], start=1):
            series = QLineSeries()
            series.setName(col)
            for r in rows:
                series.append(r[0], r[i])
            chart.addSeries(series)
        chart.createDefaultAxes()
        self.chart_view.setChart(chart)

    # -- import table -----------------------------------------------------
    def import_table_file(self, src_path, column_mapping, row_name):
        """Read src_path, apply column_mapping onto this category's table
        schema, write tables/<row_name>.mietab, point the row named
        row_name at it and commit the whole table (registry + table file)
        together, then validate. `row_name` is looked up in the LIVE table
        (rows_from_table()), not the on-disk registry, so a row just
        created with add_row() but not yet saved (e.g. a tabulated
        material, whose nk_file column can't validate on its own before a
        table exists) can be completed and saved in one step. Returns the
        list of paths written; raises PropEditorError / LibraryWriteError
        (already rolled back) on failure. This is the non-modal function
        the "Import table..." button drives -- it takes no Qt dialogs."""
        schema = TABLE_SCHEMA[self.category]
        if schema is None:
            raise PropEditorError(
                "%s rows do not reference a spectral table" % self.category)
        with open(src_path, newline="") as fh:
            reader = csv.DictReader(fh)
            src_headers = list(reader.fieldnames or [])
            src_rows = list(reader)
        dest_headers, dest_rows = apply_column_mapping(
            src_headers, src_rows, column_mapping, schema)

        lib = self.current_lib()
        if lib is None:
            raise PropEditorError("no library selected")
        rows = self.rows_from_table()
        target = _find_row(rows, row_name)
        if target is None:
            raise PropEditorError(
                "row %r must already exist (use Add row first) before "
                "importing its table" % row_name)
        blank_refs = [r.get("name", "<unnamed>") for r in rows
                     if not (r.get(REQUIRED_COLUMN) or "").strip()]
        if blank_refs:
            raise PropEditorError(
                "cannot save: row(s) %s are missing a 'reference' "
                "citation" % ", ".join(repr(n) for n in blank_refs))

        info = CATEGORY_INFO[self.category]
        tables_dir = lib.root / info["file_dir"]
        tables_dir.mkdir(parents=True, exist_ok=True)
        dest_path = tables_dir / ("%s.mietab" % row_name)

        txn = Transaction()
        txn.track(dest_path)
        reg_path = lib.registry_path(self.category)
        txn.track(reg_path)
        try:
            with open(dest_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(dest_headers)
                writer.writerows(dest_rows)
            target[info["file_cols"][0]] = dest_path.name
            _atomic_write_registry(reg_path, self._fieldnames, rows)
            validate_and_commit(lib, txn)
        except Exception:
            txn.rollback()
            raise
        self.reload()
        return [dest_path, reg_path]

    def _on_import_clicked(self):
        src, _ = QFileDialog.getOpenFileName(
            self, "Import table csv", "", "CSV files (*.csv)")
        if not src:
            return
        row = self.selected_row_dict()
        row_name = row.get("name") if row else None
        if not row_name:
            QMessageBox.warning(self, "Import table",
                                "Select the row to attach the table to "
                                "first (Add row, if it's new).")
            return
        with open(src, newline="") as fh:
            src_headers = list(csv.DictReader(fh).fieldnames or [])
        dialog = ColumnMappingDialog(src_headers, TABLE_SCHEMA[self.category],
                                     self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.import_table_file(src, dialog.mapping(), row_name)
        except (PropEditorError, LibraryWriteError, ValueError) as exc:
            QMessageBox.warning(self, "Import failed", str(exc))

    # -- promote to system --------------------------------------------------
    def promote_row(self, name, force=False):
        """Non-modal promote-to-system call: returns the list of paths
        written on success, or a conflict dict {'system_row', 'project_row'}
        if a differing row already exists in the system library and
        force=False. Raises LibraryWriteError (already rolled back) if the
        resulting system library fails validation."""
        result = self.manager.promote_to_system(self.category, name,
                                                force=force)
        if not isinstance(result, dict):
            self.reload()
        return result

    def _on_copy_to_system_clicked(self):
        row = self.selected_row_dict()
        if not row or not row.get("name"):
            return
        try:
            result = self.promote_row(row["name"])
        except LibraryWriteError as exc:
            QMessageBox.warning(self, "Copy to system library failed",
                                str(exc))
            return
        if isinstance(result, dict):
            box = QMessageBox(self)
            box.setWindowTitle("Conflict with system library")
            box.setText(
                "%r already exists in the system library with different "
                "content. Overwrite it with the project version?"
                % row["name"])
            box.setStandardButtons(QMessageBox.StandardButton.Yes
                                  | QMessageBox.StandardButton.Cancel)
            if box.exec() == QMessageBox.StandardButton.Yes:
                try:
                    self.promote_row(row["name"], force=True)
                except LibraryWriteError as exc:
                    QMessageBox.warning(self, "Copy to system library failed",
                                        str(exc))


class ColumnMappingDialog(QDialog):
    """source-column -> required-schema-column combo boxes. Modal UI only
    -- the actual mapping/application logic lives in apply_column_mapping()
    so tests never need to exec() this."""

    def __init__(self, src_headers, required_cols, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Map imported columns")
        self._combos = {}
        form = QFormLayout()
        for dest in required_cols:
            combo = QComboBox()
            combo.addItems(src_headers)
            guess = next((h for h in src_headers
                         if h.lower() == dest.lower()), None)
            if guess:
                combo.setCurrentText(guess)
            self._combos[dest] = combo
            form.addRow(dest, combo)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def mapping(self):
        return {dest: combo.currentText()
               for dest, combo in self._combos.items()}


class PropEditorPane(QWidget):
    """The dockable editor: a library selector + one tab per registry."""

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager

        self.library_combo = QComboBox()
        self.library_combo.addItem("System library", "system")
        self.library_combo.addItem("Project library", "project")
        if manager.project_lib is None:
            self.library_combo.model().item(1).setEnabled(False)
        self.library_combo.currentIndexChanged.connect(
            self._on_library_changed)

        top = QHBoxLayout()
        top.addWidget(QLabel("Library:"))
        top.addWidget(self.library_combo)
        top.addStretch(1)

        self.tabs = QTabWidget()
        self._editors = {}
        for category, label, _schema in CATEGORY_TABS:
            editor = _CategoryEditor(category, manager)
            self._editors[category] = editor
            self.tabs.addTab(editor, label)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.tabs)

    def editor(self, category):
        return self._editors[category]

    def refresh_project_availability(self):
        enabled = self.manager.project_lib is not None
        self.library_combo.model().item(1).setEnabled(enabled)

    def show_category(self, category, which_library):
        """Public entry point for the host window (e.g. routing
        LibraryPane.openEditorRequested(category, which_library)): switch
        the library combo to system/project and select the matching
        category tab."""
        index = self.library_combo.findData(which_library)
        if index != -1:
            self.library_combo.setCurrentIndex(index)
        self.tabs.setCurrentWidget(self._editors[category])

    def _on_library_changed(self, index):
        which = self.library_combo.itemData(index)
        for editor in self._editors.values():
            editor.set_library(which)
