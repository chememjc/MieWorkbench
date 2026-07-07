"""VtkSceneView - the shared VTK render-window widget for MieWorkbench's 3D
panes (Scene3DPane's optical-train view and InspectorPane's single-element
view each own one instance).

Units/placement convention (mirrors core.transforms): body-local face STLs
are BODY-LOCAL METRES (see core/project.py's Project.faces); Project
placements are mm + a FreeCAD-order quaternion (x, y, z, w). The scene
renders in METRES (so later ray/detector .vtp overlays -- also emitted in
metres by the raytracer -- line up without a rescale). Each body's actors
share one vtkTransform built by placement_to_vtk_transform(): translation
is pos_mm * 1e-3, rotation is core.transforms.quat_to_matrix(quat) (the
exact same convention the transform engine uses, reused here rather than
re-derived so the two can never drift apart).

Role coloring (see role_for_body/_ROLE_STYLE): a body is a "source" when
its custom properties include both 'power' and 'lambdac' (the extractor's
source-tagging convention, see scripts/extract_geometry.py's header); a
"detector" when its 'material' property equals 'detector'; everything else
is an "optic". Sources render red-ish and fully lit (boosted ambient/
diffuse standing in for "emissive" -- VTK's fixed-function pipeline has no
true emission term); detectors gray-blue translucent; optics glassy light
blue translucent. A selected face is highlighted orange with edges shown,
overriding the body's role color on just that face's actor (one actor per
face makes this a simple property swap, no shaders needed).

Offscreen safety: building this widget (actors/mappers/readers/the
orientation marker) never touches the GPU -- only vtkRenderWindow-level
Initialize()/Render() do real OpenGL work, and those crash immediately
under Qt's "offscreen" platform plugin (BadWindow). is_offscreen() below
gates every call site so the whole widget (and the panes built on it)
stays constructible -- and functional, short of an actual repaint -- in
pytest's headless environment.
"""

import os

import numpy as np

# vtkmodules.qt needs the Qt binding named BEFORE QVTKRenderWindowInteractor
# is imported.
import vtkmodules.qt
vtkmodules.qt.PyQtImpl = "PySide6"

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkPolyDataMapper, vtkRenderer,
)
# Registers the OpenGL2 factory overrides (mapper/actor GL implementations)
# used by vtkRenderingCore classes above; import for side effects only.
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..core.transforms import quat_to_matrix
from .facepicker import FacePicker

# -- role -> (RGB 0..1, opacity) -------------------------------------------
_ROLE_STYLE = {
    "source":   ((0.85, 0.20, 0.16), 1.00),
    "detector": ((0.55, 0.62, 0.72), 0.55),
    "optic":    ((0.58, 0.80, 0.96), 0.45),
}
_SELECTED_COLOR = (1.00, 0.55, 0.00)
_SELECTED_EDGE_COLOR = (0.10, 0.10, 0.10)

_AXIS_DIRECTIONS = {
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}


def is_offscreen():
    """True when Qt is running the headless 'offscreen' platform plugin
    (what the test suite sets via QT_QPA_PLATFORM=offscreen) -- the signal
    to skip every GPU-touching VTK call (Initialize/Render/EnabledOn)."""
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        return True
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            return app.platformName().lower() == "offscreen"
    except Exception:
        pass
    return False


def placement_to_vtk_transform(placement):
    """{"pos_mm": [x,y,z], "quat": [x,y,z,w]} -> vtkTransform mapping
    body-local metres to world-frame metres: rotation from the quaternion
    (core.transforms' x,y,z,w convention), translation pos_mm * 1e-3."""
    pos_mm = np.asarray(placement.get("pos_mm", [0.0, 0.0, 0.0]), dtype=float)
    quat = placement.get("quat", [0.0, 0.0, 0.0, 1.0])
    rot = quat_to_matrix(quat)
    pos_m = pos_mm * 1e-3

    matrix = vtkMatrix4x4()
    matrix.Identity()
    for i in range(3):
        for j in range(3):
            matrix.SetElement(i, j, float(rot[i, j]))
        matrix.SetElement(i, 3, float(pos_m[i]))

    transform = vtkTransform()
    transform.SetMatrix(matrix)
    return transform


def role_for_body(body):
    """'source' | 'detector' | 'optic' from a Project structure body dict's
    custom properties (see module docstring)."""
    props = body.get("properties") or {}
    if "power" in props and "lambdac" in props:
        return "source"
    material = (props.get("material") or {}).get("value")
    if isinstance(material, str) and material.strip().lower() == "detector":
        return "detector"
    return "optic"


class VtkSceneView(QWidget):
    """A trackball-camera VTK render window with per-face actors grouped
    under per-body transforms, an orientation-axes overlay, and click-to-
    pick face selection (see widgets/facepicker.py)."""

    facePicked = Signal(str, str, bool)   # body_name, face_id, additive

    def __init__(self, parent=None):
        super().__init__(parent)

        self.interactor = QVTKRenderWindowInteractor(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.interactor)

        self.renderer = vtkRenderer()
        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground(0.09, 0.10, 0.13)     # dark
        self.renderer.SetBackground2(0.18, 0.20, 0.26)
        self.interactor.GetRenderWindow().AddRenderer(self.renderer)

        self._style = vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self._style)

        self._axes_actor = vtkAxesActor()
        self._axes_widget = vtkOrientationMarkerWidget()
        self._axes_widget.SetOrientationMarker(self._axes_actor)
        self._axes_widget.SetInteractor(self.interactor)
        self._axes_widget.SetViewport(0.0, 0.0, 0.20, 0.20)
        if not is_offscreen():
            self._axes_widget.EnabledOn()
            self._axes_widget.InteractiveOff()

        # bookkeeping
        self._structure = None
        self._faces_dict = None
        self._body_transforms = {}     # body_name -> vtkTransform
        self._body_actors = {}         # body_name -> [actor, ...]
        self._actor_face_map = {}      # actor -> (body_name, face_id)
        self._actor_base_style = {}    # actor -> (color, opacity)
        self._face_actor_map = {}      # face_id -> actor
        self._selection = set()
        self._rays_actor = None

        self.picker = FacePicker(
            self.interactor, self.renderer, self._actor_face_map,
            self._on_picked)

        if not is_offscreen():
            self.interactor.Initialize()

    # -- picking --------------------------------------------------------
    def _on_picked(self, body_name, face_id, additive):
        if body_name is None or face_id is None:
            return
        self.facePicked.emit(body_name, face_id, bool(additive))

    # -- scene construction ----------------------------------------------
    def load_bodies(self, faces_dict, structure):
        """(Re)build every actor from scratch for the whole structure."""
        self._clear_scene()
        self._structure = structure
        self._faces_dict = faces_dict
        for body in (structure or {}).get("bodies", []):
            self._build_body_actors(body, faces_dict)
        self.fit_camera()

    def reload_bodies(self, faces_dict, structure, only=None):
        """Re-create actors for `only` (default: every body) -- used after
        a reshape (bodiesReshaped) where STLs actually changed on disk."""
        self._structure = structure
        self._faces_dict = faces_dict
        body_by_name = {b["name"]: b for b in (structure or {}).get(
            "bodies", [])}
        names = list(only) if only is not None else list(body_by_name)
        for name in names:
            self._remove_body_actors(name)
            body = body_by_name.get(name)
            if body is not None:
                self._build_body_actors(body, faces_dict)
        self._render()

    def update_placement(self, body_name, placement_dict):
        """Move a body's shared transform in place -- no STL re-read."""
        transform = self._body_transforms.get(body_name)
        if transform is None:
            return
        moved = placement_to_vtk_transform(placement_dict)
        transform.SetMatrix(moved.GetMatrix())
        self._render()

    # -- internals ----------------------------------------------------------
    def _build_body_actors(self, body, faces_dict):
        name = body["name"]
        transform = placement_to_vtk_transform(body.get("placement", {}))
        self._body_transforms[name] = transform
        role = role_for_body(body)
        color, opacity = _ROLE_STYLE[role]

        face_entries = (faces_dict.get(name) or {}).get("faces", [])
        actors = []
        for f in face_entries:
            reader = vtkSTLReader()
            reader.SetFileName(f["stl"])
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.SetUserTransform(transform)
            prop = actor.GetProperty()
            prop.SetColor(*color)
            prop.SetOpacity(opacity)
            if role == "source":
                prop.SetAmbient(0.55)
                prop.SetDiffuse(0.65)

            face_id = f["id"]
            self._actor_face_map[actor] = (name, face_id)
            self._actor_base_style[actor] = (color, opacity)
            self._face_actor_map[face_id] = actor
            actors.append(actor)
            self.renderer.AddActor(actor)

        self._body_actors[name] = actors
        if self._selection:
            self.set_selection(self._selection)

    def _remove_body_actors(self, body_name):
        for actor in self._body_actors.pop(body_name, []):
            self.renderer.RemoveActor(actor)
            self._actor_face_map.pop(actor, None)
            self._actor_base_style.pop(actor, None)
        for face_id in [fid for fid, a in self._face_actor_map.items()
                        if a not in self._actor_face_map]:
            self._face_actor_map.pop(face_id, None)
        self._body_transforms.pop(body_name, None)

    def _clear_scene(self):
        for name in list(self._body_actors):
            self._remove_body_actors(name)
        self._selection = set()

    # -- selection / highlighting -----------------------------------------
    def set_selection(self, face_ids):
        self._selection = set(face_ids or [])
        for actor, (_, face_id) in self._actor_face_map.items():
            self._apply_face_style(actor, face_id in self._selection)
        self._render()

    def clear_highlights(self):
        self.set_selection(set())

    def _apply_face_style(self, actor, selected):
        prop = actor.GetProperty()
        if selected:
            prop.SetColor(*_SELECTED_COLOR)
            prop.SetOpacity(1.0)
            prop.EdgeVisibilityOn()
            prop.SetEdgeColor(*_SELECTED_EDGE_COLOR)
            prop.SetLineWidth(2.0)
        else:
            color, opacity = self._actor_base_style.get(
                actor, (_SELECTED_COLOR, 1.0))
            prop.SetColor(*color)
            prop.SetOpacity(opacity)
            prop.EdgeVisibilityOff()

    # -- camera -------------------------------------------------------------
    def fit_camera(self):
        self.renderer.ResetCamera()
        self._render()

    def view_along(self, axis):
        direction = _AXIS_DIRECTIONS.get(axis)
        if direction is None:
            raise ValueError("unknown view axis %r" % axis)
        camera = self.renderer.GetActiveCamera()
        fp = camera.GetFocalPoint()
        dist = camera.GetDistance() or 1.0
        camera.SetPosition(fp[0] + direction[0] * dist,
                           fp[1] + direction[1] * dist,
                           fp[2] + direction[2] * dist)
        camera.SetViewUp(*(0.0, 1.0, 0.0) if axis in ("+z", "-z")
                         else (0.0, 0.0, 1.0))
        self.renderer.ResetCameraClippingRange()
        self._render()

    # -- ray/result overlays --------------------------------------------------
    def load_vtp_overlay(self, path):
        """Read a .vtp polydata file (e.g. results/viz/rays.vtp); colors by
        cell scalar array 'rgb' if present, else a uniform yellow. Returns
        the new vtkActor (also tracked internally so remove_overlay() can
        take it back out)."""
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        polydata = reader.GetOutput()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)

        cell_data = polydata.GetCellData() if polydata is not None else None
        rgb_array = cell_data.GetArray("rgb") if cell_data is not None else None
        if rgb_array is not None:
            mapper.SetScalarModeToUseCellData()
            mapper.SelectColorArray("rgb")
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()
        else:
            mapper.ScalarVisibilityOff()
            actor.GetProperty().SetColor(1.0, 0.9, 0.2)
        actor.GetProperty().SetLineWidth(1.5)

        self.remove_overlay()
        self._rays_actor = actor
        self.renderer.AddActor(actor)
        self._render()
        return actor

    def remove_overlay(self):
        if self._rays_actor is not None:
            self.renderer.RemoveActor(self._rays_actor)
            self._rays_actor = None
            self._render()

    def set_overlay_visible(self, visible):
        if self._rays_actor is not None:
            self._rays_actor.SetVisibility(bool(visible))
            self._render()

    # -- rendering ------------------------------------------------------------
    def _render(self):
        if is_offscreen():
            return
        self.interactor.GetRenderWindow().Render()
