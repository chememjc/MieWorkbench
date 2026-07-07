"""SelectionModel - the single source of truth for "which element/faces
are selected", shared by the 3D view, the outliner, the inspector, the
element editor and the transform panel.

Selection used to live implicitly in Scene3DPane (3D clicks only); panes
now subscribe to this model instead, so list-based selection (outliner,
problems pane) and 3D picking stay in sync. `origin` names the pane that
initiated the change - subscribers that would echo the selection back
(the 3D view re-highlighting, the outliner re-selecting its row) skip
updates they originated, breaking feedback loops without blockSignals
gymnastics.
"""

from PySide6.QtCore import QObject, Signal


class SelectionModel(QObject):
    changed = Signal(str, object, str)   # body_name ('' = none), faces, origin

    def __init__(self, parent=None):
        super().__init__(parent)
        self.body = None
        self.faces = set()

    def select(self, body_name, faces=(), origin=""):
        body_name = body_name or None
        faces = set(faces or ())
        if body_name == self.body and faces == self.faces:
            return
        self.body = body_name
        self.faces = faces
        self.changed.emit(body_name or "", set(faces), origin)

    def clear(self, origin=""):
        self.select(None, (), origin)
