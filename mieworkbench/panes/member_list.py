"""MemberListWidget - the clickable member-body list shared by the
Element Inspector and the Element Properties editor (WP4).

When the current selection is a multi-body element (or nothing is
selected) neither pane can show a single body's face-level surface, so
both fall back to a neutral state: a count hint plus this list of the
element's member bodies. Clicking a member is an explicit SUB-selection
of that one body (memberChosen carries the body NAME) — the host wires it
into the shared SelectionModel with a distinct origin so the dispatcher
treats it as sub-selection, not element expansion.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget,
)


class MemberListWidget(QWidget):
    memberChosen = Signal(str)   # body name of the clicked member

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: gray;")
        self.list = QListWidget()
        self.list.setToolTip(
            "Member bodies of this element — click one to inspect/edit it "
            "on its own")
        self.list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.hint)
        layout.addWidget(self.list)

    def set_members(self, hint, members):
        """`members`: ordered list of (body_name, display_label) pairs."""
        self.hint.setText(hint or "")
        self.list.clear()
        for name, label in members or ():
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.list.addItem(item)

    def _on_item_clicked(self, item):
        name = item.data(Qt.ItemDataRole.UserRole)
        if name:
            self.memberChosen.emit(name)
