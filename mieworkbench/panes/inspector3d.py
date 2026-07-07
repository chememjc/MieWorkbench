"""InspectorPane - the single-element 3D view: shows only the currently
selected body (centered, camera fit), and is the PRIMARY face-selection
surface (Scene3DPane's picking exists too, but this pane is where users
are expected to build up a face selection for the element editor).

set_body(project, body_name) tears down and rebuilds the view around just
that body; it also (re)connects to the Project so a reshape/move/property
edit of the body currently shown refreshes automatically, without the
caller having to call set_body() again by hand.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ..widgets.facepicker import pick_to_selection, select_all
from ..widgets.vtkview import VtkSceneView


class InspectorPane(QWidget):
    faceSelectionChanged = Signal(str, set)   # body_name, {face_id, ...}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project = None
        self._body_name = None
        self._face_ids = []
        self._selection = set()

        self.view = VtkSceneView(self)
        self.view.facePicked.connect(self._on_face_picked)

        self.select_all_button = QPushButton("Select all faces")
        self.select_all_button.setToolTip(
            "Select every face of the current body")
        self.select_all_button.clicked.connect(self._select_all)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Clear the face selection")
        self.clear_button.clicked.connect(self.clear_selection)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.select_all_button)
        toolbar.addWidget(self.clear_button)
        toolbar.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self.view)

    # -- body / project wiring -----------------------------------------------
    def set_body(self, project, body_name):
        if self._project is not project:
            self._disconnect_project()
            self._project = project
            self._connect_project()
        self._body_name = body_name
        self._rebuild()

    def _connect_project(self):
        if self._project is None:
            return
        self._project.bodiesReshaped.connect(self._on_bodies_reshaped)
        self._project.bodiesMoved.connect(self._on_bodies_moved)
        self._project.propertiesChanged.connect(self._on_properties_changed)

    def _disconnect_project(self):
        if self._project is None:
            return
        for sig, slot in (
                (self._project.bodiesReshaped, self._on_bodies_reshaped),
                (self._project.bodiesMoved, self._on_bodies_moved),
                (self._project.propertiesChanged,
                 self._on_properties_changed)):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def _rebuild(self):
        if self._project is None or self._body_name is None:
            self.view.load_bodies({}, {"bodies": []})
            self._face_ids = []
            self.clear_selection()
            return
        body = self._project.body(self._body_name)
        faces = self._project.faces.get(self._body_name, {"faces": []})
        structure = {"bodies": [body],
                    "sheets": self._project.sheets()}
        self._face_ids = [f["id"] for f in faces.get("faces", [])]
        self.view.load_bodies({self._body_name: faces}, structure)
        self.view.fit_camera()
        self.clear_selection()

    def _on_bodies_reshaped(self, names):
        if self._body_name in (names or []):
            self._rebuild()

    def _on_bodies_moved(self, placements):
        placement = (placements or {}).get(self._body_name)
        if placement is not None:
            self.view.update_placement(self._body_name, placement)

    def _on_properties_changed(self, body_hint):
        # properties don't change geometry; nothing to redraw here, but
        # kept as a hook (and connected) so future property-driven visual
        # cues (e.g. role recoloring after a material edit) have a place
        # to live without changing this pane's public surface.
        pass

    # -- selection ------------------------------------------------------------
    def _on_face_picked(self, body_name, face_id, additive):
        if body_name != self._body_name:
            return
        self._selection = pick_to_selection(
            self._selection, face_id, additive, all_faces=self._face_ids)
        self._apply_selection()

    def _select_all(self):
        self._selection = select_all(self._face_ids)
        self._apply_selection()

    def clear_selection(self):
        self._selection = set()
        self._apply_selection()

    def _apply_selection(self):
        self.view.set_selection(self._selection)
        self.faceSelectionChanged.emit(self._body_name, set(self._selection))

    def selection(self):
        return self._body_name, set(self._selection)
