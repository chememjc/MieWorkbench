"""Scene3DPane - the optical-train 3D view: every body in the current
Project, in one shared VtkSceneView, plus a small toolbar (fit camera,
axis-aligned views, and a rays-overlay toggle for results/viz/rays.vtp).

Wiring to Project (see core/project.py's signal docs):
    sceneLoaded     -> load_bodies(project.faces, project.structure)
    bodiesReshaped  -> reload_bodies(...) for just the reshaped bodies
    bodiesMoved     -> update_placement(name, placement) per moved body

Face/body selection: a click resolves to (body_name, face_id) via the
view's facePicked signal. Clicking a different body always replaces the
selection with that one face; clicking within the same body follows
widgets.facepicker.pick_to_selection's plain/Ctrl-click semantics. Either
way the pane re-highlights the picked face(s) and emits
selectionChanged(body_name, set_of_face_ids).
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ..widgets.facepicker import pick_to_selection
from ..widgets.vtkview import VtkSceneView

_VIEW_BUTTONS = [("+X", "+x"), ("-X", "-x"), ("+Y", "+y"), ("+Z", "+z")]


class Scene3DPane(QWidget):
    selectionChanged = Signal(str, set)   # body_name, {face_id, ...}
    raysPreviewRequested = Signal()       # Rays checked with nothing loaded

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project = None
        self._selected_body = None
        self._selected_faces = set()
        self._rays_path = None
        self._rays_visible = True
        self._rays_stale = False
        self._rays_tooltip = (
            "Show/hide the loaded ray overlay (results/viz/rays.vtp)")

        self.view = VtkSceneView(self)
        self.view.facePicked.connect(self._on_face_picked)

        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Reset the camera to frame the whole "
                                   "scene")
        self.fit_button.clicked.connect(self.view.fit_camera)

        self._view_buttons = {}
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.fit_button)
        for label, axis in _VIEW_BUTTONS:
            btn = QPushButton(label)
            btn.setToolTip("View along the %s axis" % axis)
            btn.clicked.connect(lambda _checked=False, a=axis:
                               self.view.view_along(a))
            toolbar.addWidget(btn)
            self._view_buttons[axis] = btn

        self.rays_button = QPushButton("Rays")
        self.rays_button.setCheckable(True)
        self.rays_button.setChecked(True)
        self.rays_button.setToolTip(self._rays_tooltip)
        self.rays_button.toggled.connect(self._on_rays_toggled)
        toolbar.addStretch(1)
        toolbar.addWidget(self.rays_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self.view)

    # -- Project wiring -----------------------------------------------------
    def set_project(self, project):
        if self._project is not None:
            for sig, slot in self._project_connections():
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        self._project = project
        if project is not None:
            for sig, slot in self._project_connections():
                sig.connect(slot)
            if getattr(project, "structure", None):
                self._on_scene_loaded()

    def _project_connections(self):
        p = self._project
        return [
            (p.sceneLoaded, self._on_scene_loaded),
            (p.bodiesReshaped, self._on_bodies_reshaped),
            (p.bodiesMoved, self._on_bodies_moved),
        ]

    def _on_scene_loaded(self):
        self.view.load_bodies(self._project.faces, self._project.structure)
        self._clear_selection()

    def _on_bodies_reshaped(self, names):
        self.view.reload_bodies(self._project.faces, self._project.structure,
                                only=names)
        if self._selected_body in (names or []):
            self._clear_selection()

    def _on_bodies_moved(self, placements):
        for name, placement in (placements or {}).items():
            if placement is not None:
                self.view.update_placement(name, placement)

    # -- selection ------------------------------------------------------------
    def _on_face_picked(self, body_name, face_id, additive):
        if body_name != self._selected_body:
            self._selected_body = body_name
            self._selected_faces = {face_id}
        else:
            self._selected_faces = pick_to_selection(
                self._selected_faces, face_id, additive)
        self.view.set_selection(self._selected_faces)
        self.selectionChanged.emit(self._selected_body,
                                   set(self._selected_faces))

    def _clear_selection(self):
        self._selected_body = None
        self._selected_faces = set()
        self.view.clear_highlights()

    def select_body(self, body_name):
        """Programmatic selection (outliner/problems-pane driven):
        highlight EVERY face of the body so the element is obvious in the
        train. Does not re-emit selectionChanged (the shared selection
        model is the caller)."""
        if body_name == self._selected_body and not self._selected_faces:
            return
        self._selected_body = body_name
        self._selected_faces = set()
        faces = (self._project.faces.get(body_name, {}).get("faces", [])
                 if self._project is not None else [])
        self.view.set_selection({f["id"] for f in faces})

    def selection(self):
        return self._selected_body, set(self._selected_faces)

    # -- rays overlay ---------------------------------------------------------
    def load_rays_vtp(self, path):
        self._rays_path = path
        self.view.load_vtp_overlay(path)
        self.view.set_overlay_visible(self._rays_visible)
        self.set_rays_stale(False)

    def remove_rays(self):
        self._rays_path = None
        self.view.remove_overlay()
        self.set_rays_stale(False)

    # clear_rays: same as remove_rays, kept as the name-symmetric API with
    # InspectorPane.clear_rays() (the mainwindow orchestrator wires both
    # panes the same way and shouldn't have to remember two verbs).
    clear_rays = remove_rays

    def set_rays_stale(self, stale):
        """Grey out the rays button/tooltip after a geometry edit (the
        mainwindow calls this on bodiesReshaped/bodiesMoved for a project
        that has a loaded ray overlay) — the overlay itself is left in
        place (still the last-known rays), just visually flagged as
        possibly out of date until a fresh preview/trace reloads it."""
        self._rays_stale = bool(stale)
        if self._rays_stale:
            self.rays_button.setText("Rays (stale)")
            self.rays_button.setStyleSheet("color: gray;")
            self.rays_button.setToolTip(
                self._rays_tooltip + " -- STALE: the scene changed since "
                "these rays were generated")
        else:
            self.rays_button.setText("Rays")
            self.rays_button.setStyleSheet("")
            self.rays_button.setToolTip(self._rays_tooltip)

    def _on_rays_toggled(self, checked):
        self._rays_visible = bool(checked)
        self.view.set_overlay_visible(self._rays_visible)
        if checked and self._rays_path is None:
            # nothing to show yet: ask the host to produce rays (last run's
            # overlay or a live preview) instead of silently doing nothing
            self.raysPreviewRequested.emit()
