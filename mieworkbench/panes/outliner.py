"""OutlinerPane - a list/tree of every element in the scene, so elements
can be selected, deleted, copied and pasted by NAME instead of hunting
for them in the 3D view.

Rows are ELEMENTS: bodies sharing a miewb_group collapse into one
top-level row (label = the group) with the member bodies as children;
ungrouped bodies are one row each. Columns: element / role (source,
detector, optic, ignored) / primitive kind (miewb_primitive tag, empty
for hand-authored bodies).

Signals (the main window owns the actual behavior):
    selectBodyRequested(str)   click or keyboard move -> select in the app
    customizeRequested(str)    double-click -> open the editor/wizard
    deleteRequested(str)       Del key / context menu (element group)
    copyRequested(str)         context menu (element group)
    pasteRequested()           context menu
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QMenu, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)


def role_for_body(body):
    """Same classification the 3D view colors by (kept dependency-free of
    the VTK widget so the outliner imports cheaply)."""
    props = body.get("properties", {}) or {}
    if "power" in props and "lambdac" in props:
        return "source"
    material = props.get("material", {}).get("value")
    if material == "detector":
        return "detector"
    if not material or material == "none":
        return "ignored"
    return "optic"


class OutlinerPane(QWidget):
    selectBodyRequested = Signal(str)
    customizeRequested = Signal(str)
    deleteRequested = Signal(str)
    copyRequested = Signal(str)
    pasteRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project = None
        self._updating = False

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Element", "Role", "Kind"])
        self.tree.setRootIsDecorated(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.currentItemChanged.connect(self._on_current_changed)
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)

    # -- Project wiring -----------------------------------------------------
    def set_project(self, project):
        if self._project is not None:
            for sig in (self._project.sceneLoaded,
                        self._project.bodiesReshaped,
                        self._project.propertiesChanged):
                try:
                    sig.disconnect(self._refresh)
                except (RuntimeError, TypeError):
                    pass
        self._project = project
        if project is not None:
            project.sceneLoaded.connect(self._refresh)
            project.bodiesReshaped.connect(self._refresh)
            project.propertiesChanged.connect(self._refresh)
        self._refresh()

    def _refresh(self, *_args):
        self._updating = True
        try:
            selected = self.selected_body()
            self.tree.clear()
            structure = getattr(self._project, "structure", None) or {}
            groups = {}          # group label -> [body dict]
            singles = []
            for body in structure.get("bodies", []):
                group = (body.get("properties", {})
                         .get("miewb_group", {}).get("value"))
                if group:
                    groups.setdefault(group, []).append(body)
                else:
                    singles.append(body)
            for group in sorted(groups):
                members = groups[group]
                kind = (members[0].get("properties", {})
                        .get("miewb_primitive", {}).get("value", ""))
                roles = sorted({role_for_body(b) for b in members})
                top = QTreeWidgetItem([group, "/".join(roles), kind])
                top.setData(0, Qt.UserRole, members[0]["name"])
                top.setData(0, Qt.UserRole + 1, group)
                self.tree.addTopLevelItem(top)
                if len(members) > 1:
                    for b in members:
                        child = QTreeWidgetItem(
                            [b["label"], role_for_body(b), ""])
                        child.setData(0, Qt.UserRole, b["name"])
                        child.setData(0, Qt.UserRole + 1, group)
                        top.addChild(child)
            for body in sorted(singles, key=lambda b: b["label"]):
                item = QTreeWidgetItem([body["label"],
                                        role_for_body(body), ""])
                item.setData(0, Qt.UserRole, body["name"])
                item.setData(0, Qt.UserRole + 1, body["label"])
                self.tree.addTopLevelItem(item)
            self.tree.expandAll()
            for c in range(3):
                self.tree.resizeColumnToContents(c)
            if selected:
                self.set_selected_body(selected)
        finally:
            self._updating = False

    # -- selection ------------------------------------------------------------
    def selected_body(self):
        item = self.tree.currentItem()
        return item.data(0, Qt.UserRole) if item is not None else None

    def selected_element(self):
        """The element identity (miewb_group or label) of the current row."""
        item = self.tree.currentItem()
        return item.data(0, Qt.UserRole + 1) if item is not None else None

    def set_selected_body(self, body_name):
        """Programmatic highlight (from the shared SelectionModel); does
        not re-emit selectBodyRequested."""
        self._updating = True
        try:
            for item in self._walk():
                if item.data(0, Qt.UserRole) == body_name:
                    self.tree.setCurrentItem(item)
                    return
            self.tree.setCurrentItem(None)
        finally:
            self._updating = False

    def _walk(self):
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            yield top
            for j in range(top.childCount()):
                yield top.child(j)

    def _on_current_changed(self, current, _previous):
        if self._updating or current is None:
            return
        body = current.data(0, Qt.UserRole)
        if body:
            self.selectBodyRequested.emit(body)

    def _on_double_clicked(self, item, _column):
        body = item.data(0, Qt.UserRole)
        if body:
            self.customizeRequested.emit(body)

    # -- context menu / keys ----------------------------------------------------
    def _on_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        menu = QMenu(self)
        if item is not None:
            element = item.data(0, Qt.UserRole + 1)
            act = menu.addAction("Copy")
            act.triggered.connect(
                lambda: self.copyRequested.emit(element))
            act = menu.addAction("Delete")
            act.triggered.connect(
                lambda: self.deleteRequested.emit(element))
            menu.addSeparator()
        act = menu.addAction("Paste")
        act.triggered.connect(self.pasteRequested.emit)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            element = self.selected_element()
            if element:
                self.deleteRequested.emit(element)
                return
        super().keyPressEvent(event)
