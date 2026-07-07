"""library.py -- LibraryPane: the "Library" dock (mainwindow.py's
library_dock placeholder is swapped for this in a later phase).

Three tabs:
  Elements         - primitives/*.FCStd (core.librarymgr.LibraryManager.
                     primitives_list()), grouped by category. Double-click
                     or "Add to scene" emits addElementRequested(info,
                     label); the default label (kind + running number) is
                     computed by the free function default_label() so it
                     is testable without going through the (modal) label
                     dialog.
  Project library  - per-category row counts from the project PropLibrary
                     (raw registry reads -- a project library may be
                     legitimately incomplete/invalid mid-edit, so this
                     never calls PropLibrary.categories(), which requires
                     a full, validated load).
  System library   - same, for the system PropLibrary.

Both summary tabs' "Open in editor" button and double-clicking a row emit
openEditorRequested(category, which_library) for the host window to route
to PropEditorPane.show_category(category, which_library).
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QTabWidget, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ..core.proplib import CATEGORIES


def default_label(kind, used_labels):
    """kind + running number: 'lens_pcx' if unused yet, else 'lens_pcx_2',
    'lens_pcx_3', ... . Pure function so the "Add to scene" default-label
    logic is testable without the modal LabelDialog."""
    used = set(used_labels)
    if kind not in used:
        return kind
    n = 2
    while ("%s_%d" % (kind, n)) in used:
        n += 1
    return "%s_%d" % (kind, n)


class LabelDialog(QDialog):
    """Modal-only wrapper around a single label text field; the default
    value comes from default_label()."""

    def __init__(self, default_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add element to scene")
        self._edit = QLineEdit(default_text)
        form = QFormLayout()
        form.addRow("Label:", self._edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def label(self):
        return self._edit.text().strip()


class _LibrarySummary(QWidget):
    """Per-category row counts for one PropLibrary (system or project),
    read straight off the registry csvs (no validation, since a project
    library may be an intentionally partial work in progress). Rows show
    just "<category> (<count>)" -- the full entry-name listing was too
    noisy; drill into names via "Open in editor" / double-click, which
    both route to the same PropEditorPane tab."""

    openEditorRequested = Signal(str, str)   # category, which_library

    def __init__(self, which_library, manager, parent=None):
        super().__init__(parent)
        self.which_library = which_library
        self.manager = manager

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(
            self._on_item_double_clicked)
        self.open_button = QPushButton("Open in editor")
        self.open_button.clicked.connect(self._on_open_clicked)

        layout = QVBoxLayout(self)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.open_button)

        self.refresh()

    def _lib(self):
        if self.which_library == "project":
            return self.manager.project_lib
        return self.manager.system_lib

    def refresh(self):
        self.list_widget.clear()
        lib = self._lib()
        if lib is None:
            item = QListWidgetItem("(no project library set)")
            item.setData(Qt.ItemDataRole.UserRole, None)
            self.list_widget.addItem(item)
            self.open_button.setEnabled(False)
            return
        self.open_button.setEnabled(True)
        for category in CATEGORIES:
            rows = lib.registry_rows(category)
            item = QListWidgetItem("%s (%d)" % (category, len(rows)))
            item.setData(Qt.ItemDataRole.UserRole, category)
            self.list_widget.addItem(item)

    def _open_for_item(self, item):
        category = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not category:
            category = CATEGORIES[0]
        self.openEditorRequested.emit(category, self.which_library)

    def _on_open_clicked(self):
        self._open_for_item(self.list_widget.currentItem())

    def _on_item_double_clicked(self, item):
        self._open_for_item(item)


class LibraryPane(QWidget):
    addElementRequested = Signal(dict, str)   # primitive_info, label
    openEditorRequested = Signal(str, str)    # category, which_library

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self._used_labels = set()
        self._primitives = []

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)

        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setAlignment(Qt.AlignmentFlag.AlignTop
                                  | Qt.AlignmentFlag.AlignLeft)

        self.add_button = QPushButton("Add to scene")
        self.add_button.clicked.connect(self._on_add_clicked)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)

        elements_tab = QWidget()
        top_row = QHBoxLayout()
        top_row.addWidget(self.refresh_button)
        top_row.addStretch(1)
        top_row.addWidget(self.add_button)
        el_layout = QVBoxLayout(elements_tab)
        el_layout.addLayout(top_row)
        el_layout.addWidget(self.tree, stretch=2)
        el_layout.addWidget(self.details, stretch=1)

        self.project_summary = _LibrarySummary("project", manager)
        self.system_summary = _LibrarySummary("system", manager)
        self.project_summary.openEditorRequested.connect(
            self.openEditorRequested)
        self.system_summary.openEditorRequested.connect(
            self.openEditorRequested)

        self.tabs = QTabWidget()
        self.tabs.addTab(elements_tab, "Elements")
        self.tabs.addTab(self.project_summary, "Project library")
        self.tabs.addTab(self.system_summary, "System library")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

        self.refresh()

    # -- elements tab -------------------------------------------------------
    def refresh(self):
        """Rescan primitives/*.FCStd (user-dropped files must appear) and
        refresh both library summaries."""
        self._primitives = self.manager.primitives_list()
        self.tree.clear()
        groups = {}
        for info in self._primitives:
            groups.setdefault(info["category"], []).append(info)
        for category in sorted(groups):
            group_item = QTreeWidgetItem([category])
            group_item.setFlags(group_item.flags()
                                & ~Qt.ItemFlag.ItemIsSelectable)
            self.tree.addTopLevelItem(group_item)
            for info in sorted(groups[category], key=lambda i: i["label"]):
                child = QTreeWidgetItem([info["label"]])
                child.setToolTip(0, info.get("tooltip", ""))
                child.setData(0, Qt.ItemDataRole.UserRole, info)
                group_item.addChild(child)
        self.tree.expandAll()
        self.project_summary.refresh()
        self.system_summary.refresh()

    def primitive_count(self):
        return len(self._primitives)

    def _selected_primitive(self):
        items = self.tree.selectedItems()
        if not items:
            return None
        return items[0].data(0, Qt.ItemDataRole.UserRole)

    def _on_selection_changed(self):
        info = self._selected_primitive()
        if info is None:
            self.details.setText("")
            return
        lines = [info["label"], info.get("tooltip", "")]
        params = info.get("params") or {}
        if params:
            lines.append("")
            for alias, spec in params.items():
                lines.append("%s = %s %s -- %s"
                             % (alias, spec.get("default"),
                                spec.get("unit", ""), spec.get("help", "")))
        self.details.setText("\n".join(str(l) for l in lines))

    def _on_item_double_clicked(self, item, _column):
        info = item.data(0, Qt.ItemDataRole.UserRole)
        if info is not None:
            self._start_add_flow(info)

    def _on_add_clicked(self):
        info = self._selected_primitive()
        if info is not None:
            self._start_add_flow(info)

    def start_add_current(self):
        """Toolbar 'Add element' entry point: start the add flow for the
        currently selected primitive. Returns False when nothing usable
        is selected (caller raises the pane for browsing instead)."""
        self.tabs.setCurrentIndex(0)
        info = self._selected_primitive()
        if info is None:
            return False
        self._start_add_flow(info)
        return True

    def _start_add_flow(self, info):
        label = default_label(info["kind"], self._used_labels)
        dialog = LabelDialog(label, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dialog.label() or label
        self.request_add_element(info, chosen)

    def request_add_element(self, info, label):
        """Non-modal entry point used directly by tests (and by
        _start_add_flow after the label dialog is accepted): emits
        addElementRequested(info, label)."""
        self._used_labels.add(label)
        self.addElementRequested.emit(info, label)
