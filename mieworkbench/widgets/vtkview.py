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

import math
import os

import numpy as np

# vtkmodules.qt needs the Qt binding named BEFORE QVTKRenderWindowInteractor
# is imported.
import vtkmodules.qt
vtkmodules.qt.PyQtImpl = "PySide6"

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkActor2D, vtkCoordinate, vtkPolyDataMapper,
    vtkPolyDataMapper2D, vtkRenderer, vtkTextActor,
)
# Registers the OpenGL2 factory overrides (mapper/actor GL implementations)
# used by vtkRenderingCore classes above; import for side effects only.
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..core.transforms import quat_to_matrix
from .faceindicators import FaceIndicatorLayer
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


# ---------------------------------------------------------------------------
# scale bar -- pure math, no VTK objects touched, so the test suite can
# exercise it offscreen with plain floats (see VtkSceneView._update_scale_bar
# for the thin VTK-facing wrapper around these).
# ---------------------------------------------------------------------------
_NICE_MULTIPLIERS = (1.0, 2.0, 5.0)     # the 1-2-5 decade sequence
_SCALEBAR_MARGIN_X = 0.03    # normalized-viewport gap from the right edge
_SCALEBAR_Y = 0.06           # normalized-viewport height of the bar off the bottom
_SCALEBAR_TICK_HALF = 0.01   # normalized-viewport half-height of the end ticks
_SCALEBAR_LABEL_GAP = 0.025  # normalized-viewport gap between bar and label


def px_per_metre_parallel(parallel_scale_m, viewport_height_px):
    """Pixels-per-metre for an orthographic (parallel-projection) camera.
    vtkCamera's ParallelScale is, by VTK convention, half the world-space
    height visible in the viewport, so px/m = height_px / (2 * scale)."""
    if parallel_scale_m <= 0 or viewport_height_px <= 0:
        return 0.0
    return viewport_height_px / (2.0 * parallel_scale_m)


def px_per_metre_perspective(distance_m, view_angle_deg, viewport_height_px):
    """Pixels-per-metre for a perspective camera: the world-space height
    visible at the focal plane is 2 * distance * tan(view_angle / 2)
    (similar triangles from the camera to the focal point)."""
    if distance_m <= 0 or viewport_height_px <= 0:
        return 0.0
    half_angle_rad = math.radians(view_angle_deg) / 2.0
    world_height_m = 2.0 * distance_m * math.tan(half_angle_rad)
    if world_height_m <= 0:
        return 0.0
    return viewport_height_px / world_height_m


def nice_bar_length(px_per_m, viewport_px, frac_lo=0.2, frac_hi=0.3):
    """Snap the ideal bar length -- one that would occupy roughly the
    midpoint of [frac_lo, frac_hi] of the viewport width -- to the nearest
    1-2-5 decade sequence (1, 2, 5, 10, 20, 50, ... mm/um/m -- whatever
    scale the metres happen to land in), returning the result in metres.
    Falls back to the closest-to-band candidate if no exact 1-2-5 multiple
    lands inside the band (can happen at extreme zoom). Returns 0.0 for a
    degenerate camera/viewport (guards the log10 below)."""
    if px_per_m <= 0 or viewport_px <= 0:
        return 0.0
    target_m = ((frac_lo + frac_hi) / 2.0) * viewport_px / px_per_m
    if target_m <= 0:
        return 0.0
    exponent = int(math.floor(math.log10(target_m)))

    best_length, best_gap = None, None
    for e in (exponent - 1, exponent, exponent + 1):
        for mult in _NICE_MULTIPLIERS:
            length_m = mult * (10.0 ** e)
            frac = length_m * px_per_m / viewport_px
            if frac_lo <= frac <= frac_hi:
                return length_m
            gap = (frac_lo - frac) if frac < frac_lo else (frac - frac_hi)
            if best_gap is None or gap < best_gap:
                best_gap, best_length = gap, length_m
    return best_length


def _format_nice_number(value):
    """Render a 1-2-5-sequence-derived number without float noise or a
    pointless trailing ".0" -- 500.0000000001 -> "500", 0.5 -> "0.5"."""
    rounded = round(value, 6)
    if rounded == int(rounded):
        return str(int(rounded))
    return ("%.3f" % rounded).rstrip("0").rstrip(".")


def format_bar_label(length_m):
    """Physical length (metres, the scene's internal unit) -> a display
    string in the GUI's mm convention, falling back to micrometres below
    2.5 mm so small bars don't read as "0.5 mm" -- e.g. "500 µm",
    "5 mm", "20 mm"."""
    length_mm = length_m * 1e3
    if length_mm < 2.5:
        return "%s µm" % _format_nice_number(length_mm * 1e3)
    return "%s mm" % _format_nice_number(length_mm)


class VtkSceneView(QWidget):
    """A trackball-camera VTK render window with per-face actors grouped
    under per-body transforms, an orientation-axes overlay, and click-to-
    pick face selection (see widgets/facepicker.py)."""

    # body_name, face_id, mode -- mode is a facepicker.PICK_MODES string
    # from real clicks; typed `object` so legacy tests/callers emitting the
    # old boolean `additive` flag still work (receivers normalize via
    # facepicker.normalize_pick_mode / pick_to_selection).
    facePicked = Signal(str, str, object)
    contextRequested = Signal(int, int)   # VTK event coords (origin
                                          # bottom-left); see
                                          # enable_context_menu()

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

        # scale bar (2D overlay, bottom-right, unobtrusive) -- the actors
        # themselves touch the GPU no more than any other actor does (only
        # Initialize()/Render() do real OpenGL work), but vtkTextActor logs
        # a noisy "Failed getting the TextRenderer instance" the moment
        # it's constructed under the offscreen platform plugin, so -- like
        # the axes widget above -- the whole thing is built only when live;
        # offscreen it just stays absent (see set_scale_bar_visible/
        # _update_scale_bar, both no-ops when these are None).
        self._scale_bar_visible = True
        self._scalebar_points = None
        self._scalebar_line_actor = None
        self._scalebar_text_actor = None
        if not is_offscreen():
            self._build_scale_bar()
            # Recompute on every render -- covers interactive camera moves
            # (dolly/pan/rotate all end in a Render()) as well as the
            # explicit fit_camera()/view_along()/load_bodies() call sites,
            # with no need to hook each of those separately.
            self.renderer.AddObserver("StartEvent", self._on_render_start)
            self._update_scale_bar()

        # face-orientation indicator glyphs (see widgets/faceindicators.py);
        # actor construction is GPU-free, so this is offscreen-safe
        self._indicators = FaceIndicatorLayer(self.renderer)

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
        self._rays_polydata = None
        self._overlay_stale = False

        self.picker = FacePicker(
            self.interactor, self.renderer, self._actor_face_map,
            self._on_picked)

        if not is_offscreen():
            self.interactor.Initialize()

    # -- picking --------------------------------------------------------
    def _on_picked(self, body_name, face_id, mode):
        if body_name is None or face_id is None:
            return
        self.facePicked.emit(body_name, face_id, mode)

    def enable_context_menu(self):
        """Opt in to right-click context-menu requests: right-button
        presses emit contextRequested(x, y) in VTK event coordinates
        instead of starting the trackball style's right-drag dolly (which
        a popup menu would leave stuck mid-drag -- see facepicker.py).
        Scroll-wheel zoom is unaffected. Views that never call this keep
        the stock right-drag zoom."""
        self.picker.enable_context(
            lambda x, y: self.contextRequested.emit(int(x), int(y)))

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
        self._indicators.rebuild_body(body, face_entries, transform, role)
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
        self._indicators.remove_body(body_name)

    def set_face_indicators_visible(self, visible):
        """Menu-toggle hook: show/hide every face-orientation glyph (in
        this view); newly built glyphs inherit the state."""
        self._indicators.set_visible(visible)
        self._render()

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

    # -- scale bar ------------------------------------------------------------
    def _build_scale_bar(self):
        """Create the (initially zero-length, repositioned on the first
        _update_scale_bar()) scale-bar actors: a 2-point polyline for the
        bar itself plus two short end ticks, all one polydata driven by a
        Normalized-Viewport vtkCoordinate so its raw point coordinates are
        plain 0..1 viewport fractions instead of pixels -- and a
        vtkTextActor for the length label, positioned the same way."""
        points = vtkPoints()
        for _ in range(6):     # 0/1=bar ends, 2/3=left tick, 4/5=right tick
            points.InsertNextPoint(0.0, 0.0, 0.0)
        lines = vtkCellArray()
        for a, b in ((0, 1), (2, 3), (4, 5)):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(a)
            lines.InsertCellPoint(b)
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        self._scalebar_points = points

        coord = vtkCoordinate()
        coord.SetCoordinateSystemToNormalizedViewport()
        mapper = vtkPolyDataMapper2D()
        mapper.SetInputData(polydata)
        mapper.SetTransformCoordinate(coord)

        line_actor = vtkActor2D()
        line_actor.SetMapper(mapper)
        line_actor.GetProperty().SetColor(0.88, 0.88, 0.88)
        line_actor.GetProperty().SetLineWidth(1.5)
        self._scalebar_line_actor = line_actor
        self.renderer.AddViewProp(line_actor)

        text_actor = vtkTextActor()
        text_actor.GetTextProperty().SetFontSize(12)
        text_actor.GetTextProperty().SetColor(0.88, 0.88, 0.88)
        text_actor.GetTextProperty().SetJustificationToCentered()
        text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self._scalebar_text_actor = text_actor
        self.renderer.AddViewProp(text_actor)

    def _on_render_start(self, *_vtk_args):
        # vtkObject observer callback signature is (caller, event_name);
        # unused, but VTK always passes them.
        self._update_scale_bar()

    def _update_scale_bar(self):
        """Recompute the bar's snapped length (nice_bar_length) and the
        on-screen placement of its ticks/label from the current camera --
        called on every render (see the StartEvent observer in __init__)
        so it always fits regardless of what changed the camera."""
        if self._scalebar_line_actor is None:
            return
        width_px, height_px = self.interactor.GetRenderWindow().GetSize()
        if width_px <= 0 or height_px <= 0:
            return

        camera = self.renderer.GetActiveCamera()
        if camera.GetParallelProjection():
            ppm = px_per_metre_parallel(camera.GetParallelScale(), height_px)
        else:
            ppm = px_per_metre_perspective(
                camera.GetDistance(), camera.GetViewAngle(), height_px)

        length_m = nice_bar_length(ppm, width_px)
        if length_m <= 0:
            self._scalebar_line_actor.SetVisibility(False)
            self._scalebar_text_actor.SetVisibility(False)
            return

        width_frac = length_m * ppm / width_px
        right_x = 1.0 - _SCALEBAR_MARGIN_X
        left_x = right_x - width_frac
        centre_x = (left_x + right_x) / 2.0
        y = _SCALEBAR_Y

        points = self._scalebar_points
        points.SetPoint(0, left_x, y, 0.0)
        points.SetPoint(1, right_x, y, 0.0)
        points.SetPoint(2, left_x, y - _SCALEBAR_TICK_HALF, 0.0)
        points.SetPoint(3, left_x, y + _SCALEBAR_TICK_HALF, 0.0)
        points.SetPoint(4, right_x, y - _SCALEBAR_TICK_HALF, 0.0)
        points.SetPoint(5, right_x, y + _SCALEBAR_TICK_HALF, 0.0)
        points.Modified()

        self._scalebar_text_actor.SetInput(format_bar_label(length_m))
        self._scalebar_text_actor.SetPosition(centre_x, y + _SCALEBAR_LABEL_GAP)

        self._apply_scale_bar_visibility()

    def _apply_scale_bar_visibility(self):
        if self._scalebar_line_actor is None:
            return
        self._scalebar_line_actor.SetVisibility(self._scale_bar_visible)
        self._scalebar_text_actor.SetVisibility(self._scale_bar_visible)

    def set_scale_bar_visible(self, visible):
        """Public toggle -- visible by default (see __init__). A no-op
        offscreen, where the actors were never built."""
        self._scale_bar_visible = bool(visible)
        self._apply_scale_bar_visibility()
        self._render()

    # -- ray/result overlays --------------------------------------------------
    @staticmethod
    def _apply_overlay_coloring(actor, polydata):
        """Wavelength coloring: color by the per-cell 'rgb' array
        (written by raytracer.vtkexport.write_vtp_polylines from each ray
        segment's wavelength) when present, else a uniform yellow."""
        mapper = actor.GetMapper()
        cell_data = polydata.GetCellData() if polydata is not None else None
        rgb_array = (cell_data.GetArray("rgb")
                     if cell_data is not None else None)
        if rgb_array is not None:
            mapper.SetScalarModeToUseCellData()
            mapper.SelectColorArray("rgb")
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()
        else:
            mapper.ScalarVisibilityOff()
            actor.GetProperty().SetColor(1.0, 0.9, 0.2)

    def load_vtp_overlay(self, path):
        """Read a .vtp polydata file (e.g. results/viz/rays.vtp); colors by
        cell scalar array 'rgb' if present, else a uniform yellow. Returns
        the new vtkActor (also tracked internally so remove_overlay() can
        take it back out). A freshly loaded overlay is never stale."""
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        polydata = reader.GetOutput()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        self._apply_overlay_coloring(actor, polydata)
        actor.GetProperty().SetLineWidth(1.5)

        self.remove_overlay()
        self._rays_actor = actor
        self._rays_polydata = polydata
        self._overlay_stale = False
        self.renderer.AddActor(actor)
        self._render()
        return actor

    def remove_overlay(self):
        if self._rays_actor is not None:
            self.renderer.RemoveActor(self._rays_actor)
            self._rays_actor = None
            self._rays_polydata = None
            self._overlay_stale = False
            self._render()

    def set_overlay_stale(self, stale):
        """Grey the ray overlay itself (not just a button label) while the
        scene has changed since the rays were generated: uniform grey at
        low opacity. Un-staling restores the wavelength coloring."""
        stale = bool(stale)
        if stale == self._overlay_stale and self._rays_actor is None:
            return
        self._overlay_stale = stale
        if self._rays_actor is None:
            return
        prop = self._rays_actor.GetProperty()
        if stale:
            self._rays_actor.GetMapper().ScalarVisibilityOff()
            prop.SetColor(0.45, 0.45, 0.45)
            prop.SetOpacity(0.35)
        else:
            prop.SetOpacity(1.0)
            self._apply_overlay_coloring(self._rays_actor,
                                         self._rays_polydata)
        self._render()

    def overlay_is_stale(self):
        return self._overlay_stale

    def set_overlay_visible(self, visible):
        if self._rays_actor is not None:
            self._rays_actor.SetVisibility(bool(visible))
            self._render()

    # -- rendering ------------------------------------------------------------
    def _render(self):
        if is_offscreen():
            return
        self.interactor.GetRenderWindow().Render()
