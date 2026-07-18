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
blue translucent -- EXCEPT an absorbing aperture-stop plate (see
body_style), which renders opaque dark instead of glassy blue. A selected
face is highlighted solid orange, overriding the body's role color on just
that face's actor (one actor per face makes this a simple property swap,
no shaders needed).

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
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData, vtkGlyph3D, vtkPolyDataNormals,
)
from vtkmodules.vtkFiltersSources import vtkSphereSource
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

# -- opaque override for absorbing aperture stops (WP5), see body_style ----
# iris/iris_bladed/pinhole/slit discs carry a body-level 'absorbance'
# derived from their 'blackness' sheet param (scripts/primitivelib.py); at
# blackness>=0.95 the classic glassy-blue "optic" look reads wrong -- these
# are opaque blackened metal, not a transmissive element.
_ABSORBER_STYLE = ((0.12, 0.12, 0.13), 1.00)
_ABSORBER_THRESHOLD = 0.5

# -- ghosted (train-excluded) style, see set_excluded_bodies ---------------
_GHOST_OPACITY = 0.25
_GHOST_GRAY = (0.55, 0.55, 0.55)

# -- chain-link overlay, see set_chain_links --------------------------------
_CHAIN_LINK_COLOR = (0.35, 0.65, 0.95)    # cool blue
_FOLD_LINK_COLOR = (0.95, 0.60, 0.20)     # orange
_LINK_OPACITY = 0.55
_LINK_LINE_WIDTH = 1.5
# VTK 9.6's OpenGL2 backend still honors classic per-actor line stippling
# through vtkProperty (verified on this build: SetLineStipplePattern /
# SetLineStippleRepeatFactor are both present) -- no need for a geometric
# dash-segment fallback.
_LINK_STIPPLE_PATTERN = 0xF0F0            # 4-on/4-off, doubled by the factor
_LINK_STIPPLE_REPEAT = 2


def _ghost_color(color):
    """Blend a role color halfway toward neutral grey -- the "ghosted"
    look applied to train-excluded elements (set_excluded_bodies)."""
    return tuple(0.5 * c + 0.5 * g for c, g in zip(color, _GHOST_GRAY))

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


def force_vtk_init():
    """MIEWB_FORCE_VTK_INIT=1 forces interactor.Initialize() even offscreen
    -- solely so a headless test can reproduce the real Initialize()-then-
    teardown path the exit-hang fix targets (an Initialize()d-but-never-
    Finalize()d QVTKRenderWindowInteractor hangs the interpreter after
    app.exec() returns). It gates ONLY the Initialize() call, not the
    GPU/text-renderer work (axes widget, scale bar) that genuinely crashes
    or spams under the offscreen platform plugin."""
    return os.environ.get("MIEWB_FORCE_VTK_INIT") == "1"


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


def body_style(body):
    """(RGB 0..1, opacity) render style for a body: the opaque
    _ABSORBER_STYLE override for absorbing aperture-stop plates (iris,
    iris_bladed, pinhole, slit -- see primitivelib.py), else the plain
    role-based _ROLE_STYLE. Pure function so re-tessellation
    (_build_body_actors) and the tests can exercise the rule without VTK.

    The override applies only when ALL hold:
      - role_for_body(body) == 'optic' (sources/detectors never darken)
      - body property 'absorbance' parses as a float >= _ABSORBER_THRESHOLD
        (non-numeric/absent -> not an absorber, e.g. edge_blackened lenses,
        which only carry a bool 'edge_blackened' prop -- the per-face
        absorbance they derive is extract-time-only, never a body prop)
      - body property 'material' is not (case/whitespace-insensitively)
        'air' (excludes the aperture's own clear-opening 'plug' body)
      - body property 'mirror' is absent (a reflective element must stay
        its normal role color even if it also happens to carry a high
        absorbance -- this is a user-facing requirement, not inferred from
        current primitives: no shipped kind currently stamps both)
    """
    role = role_for_body(body)
    if role != "optic":
        return _ROLE_STYLE[role]
    props = body.get("properties") or {}
    if "mirror" in props:
        return _ROLE_STYLE[role]
    material = (props.get("material") or {}).get("value")
    if isinstance(material, str) and material.strip().lower() == "air":
        return _ROLE_STYLE[role]
    try:
        absorbance = float((props.get("absorbance") or {}).get("value"))
    except (TypeError, ValueError):
        return _ROLE_STYLE[role]
    if absorbance >= _ABSORBER_THRESHOLD:
        return _ABSORBER_STYLE
    return _ROLE_STYLE[role]


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


class BeadLayer:
    """Tracer-bead glyphs (owned by VtkSceneView, driven per frame by
    core/beadanim.AnimationController): one sphere per active ray
    segment, positioned by the animation clock, colored by the segment's
    wavelength rgb. Opaque by default (ForceOpaque) -- attenuation dimming
    applies to the ray LINES, never to the beads, so a bead stays findable
    even on a nearly-faded ray. The opt-in "power" bead-opacity mode
    passes a per-bead alpha to update_beads: the scalar array becomes a
    4-component RGBA uchar (DirectScalars routes the 4th component to the
    translucent pass) and ForceOpaque is lifted for that actor; alpha=None
    restores the opaque default. Construction is GPU-free (offscreen-safe,
    same rationale as FaceIndicatorLayer)."""

    def __init__(self, renderer):
        self.renderer = renderer
        self._points = vtkPoints()
        self._rgb = vtkUnsignedCharArray()
        self._rgb.SetNumberOfComponents(3)
        self._rgb.SetName("rgb")
        self._poly = vtkPolyData()
        self._poly.SetPoints(self._points)
        self._poly.GetPointData().SetScalars(self._rgb)

        self._sphere = vtkSphereSource()
        self._sphere.SetThetaResolution(12)
        self._sphere.SetPhiResolution(12)
        self._sphere.SetRadius(1e-3)          # 1 mm default, in metres

        glyph = vtkGlyph3D()
        glyph.SetInputData(self._poly)
        glyph.SetSourceConnection(self._sphere.GetOutputPort())
        glyph.SetColorModeToColorByScalar()
        glyph.SetScaleModeToDataScalingOff()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetColorModeToDirectScalars()
        mapper.ScalarVisibilityOn()
        self.actor = vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetOpacity(1.0)
        self.actor.ForceOpaqueOn()
        self.actor.SetVisibility(False)       # off until animation enabled
        self.renderer.AddActor(self.actor)

    def update_beads(self, points_m, rgb, alpha=None):
        """points_m (M,3) float metres; rgb (M,3) uint8. `alpha` (M,) float
        in [0,1] switches the beads to a 4-component RGBA scalar (opt-in
        "power" opacity mode); alpha=None keeps the opaque 3-component path
        bit-identical to the default."""
        from vtkmodules.util.numpy_support import numpy_to_vtk
        pts = np.asarray(points_m, dtype=float).reshape(-1, 3)
        rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        self._points.Reset()
        for i in range(len(pts)):
            self._points.InsertNextPoint(*(float(c) for c in pts[i]))
        self._points.Modified()

        if alpha is None:
            scalars = np.ascontiguousarray(rgb)
            self.actor.GetProperty().SetOpacity(1.0)
            self.actor.ForceOpaqueOn()
        else:
            a = np.clip(np.round(255.0 * np.asarray(alpha, dtype=float)),
                        0, 255).astype(np.uint8)
            rgba = np.empty((len(rgb), 4), dtype=np.uint8)
            rgba[:, :3] = rgb
            rgba[:, 3] = a
            scalars = np.ascontiguousarray(rgba)
            self.actor.ForceOpaqueOff()
        self._rgb = numpy_to_vtk(scalars, deep=1)   # keep a ref alive
        self._rgb.SetName("rgb")
        self._poly.GetPointData().SetScalars(self._rgb)
        self._poly.Modified()

    def set_radius_m(self, radius_m):
        self._sphere.SetRadius(max(1e-6, float(radius_m)))

    def set_visible(self, visible):
        self.actor.SetVisibility(bool(visible))

    def clear(self):
        self.update_beads(np.zeros((0, 3)), np.zeros((0, 3), np.uint8))

    def bead_count(self):
        return self._points.GetNumberOfPoints()


class VtkSceneView(QWidget):
    """A trackball-camera VTK render window with per-face actors grouped
    under per-body transforms, an orientation-axes overlay, and click-to-
    pick face selection (see widgets/facepicker.py)."""

    # body_name, face_id, mode -- mode is a facepicker.PICK_MODES string
    # from real clicks; typed `object` so legacy tests/callers emitting the
    # old boolean `additive` flag still work (receivers normalize via
    # facepicker.normalize_pick_mode / pick_to_selection).
    facePicked = Signal(str, str, object)
    overlayChanged = Signal()             # ray overlay loaded / removed /
                                          # stale-flag flipped (the bead-
                                          # animation controller re-reads
                                          # the polydata on this)
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
        self._render_start_obs = None    # renderer StartEvent observer id
        if not is_offscreen():
            self._build_scale_bar()
            # Recompute on every render -- covers interactive camera moves
            # (dolly/pan/rotate all end in a Render()) as well as the
            # explicit fit_camera()/view_along()/load_bodies() call sites,
            # with no need to hook each of those separately.
            self._render_start_obs = self.renderer.AddObserver(
                "StartEvent", self._on_render_start)
            self._update_scale_bar()

        # face-orientation indicator glyphs (see widgets/faceindicators.py);
        # actor construction is GPU-free, so this is offscreen-safe
        self._indicators = FaceIndicatorLayer(self.renderer)

        # tracer-bead animation glyphs (driven by core/beadanim); also
        # GPU-free to build, invisible until the animation is enabled
        self.beads = BeadLayer(self.renderer)

        # bookkeeping
        self._structure = None
        self._faces_dict = None
        self._body_transforms = {}     # body_name -> vtkTransform
        self._body_actors = {}         # body_name -> [actor, ...]
        self._actor_face_map = {}      # actor -> (body_name, face_id)
        self._actor_base_style = {}    # actor -> (color, opacity)
        self._face_actor_map = {}      # face_id -> actor
        self._selection = set()
        self._excluded_bodies = set()  # body_name set -> ghosted (see
                                       # set_excluded_bodies)
        self._chain_links_actor = None       # set_chain_links overlay actor
        self._chain_links_polydata = None
        self._rays_actor = None
        self._rays_polydata = None
        self._overlay_stale = False
        self._dim_mode = "off"         # 'off' | 'linear' | 'sqrt'
        self._dim_floor = 0.0          # minimum opacity, percent 0-100

        self.picker = FacePicker(
            self.interactor, self.renderer, self._actor_face_map,
            self._on_picked)

        # one-shot pick + axis-drag state (see pick_face_once/begin_axis_drag)
        self._once_pick_cb = None
        self._axis_drag = None

        self._shutdown_done = False
        if not is_offscreen() or force_vtk_init():
            self.interactor.Initialize()

    # -- teardown --------------------------------------------------------
    def shutdown(self):
        """Idempotent teardown of the native VTK resources this view owns,
        so the interpreter can exit cleanly to the shell. An
        Initialize()d-but-never-Finalize()d QVTKRenderWindowInteractor
        hangs the process at teardown after app.exec() returns; this
        detaches the orientation-marker widget, drops every observer the
        view (and its picker) registered, then Finalize()s the render
        window and closes the interactor. Every step is guarded: offscreen
        the GPU-touching resources were never created, and shutdown() may
        be called twice (closeEvent + an explicit host call)."""
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True

        # end any in-progress modal axis drag (removes its interactor
        # observers and restores trackball control)
        try:
            self._end_axis_drag()
        except Exception:
            pass

        # disable + detach the orientation-marker widget while it still has
        # a live interactor (EnabledOff first, then drop the back-reference
        # so the widget no longer pins the interactor/render window)
        widget = getattr(self, "_axes_widget", None)
        if widget is not None:
            try:
                widget.EnabledOff()
            except Exception:
                pass
            try:
                widget.SetInteractor(None)
            except Exception:
                pass
            self._axes_widget = None

        # remove the per-render scale-bar observer registered on the renderer
        obs = getattr(self, "_render_start_obs", None)
        renderer = getattr(self, "renderer", None)
        if obs is not None and renderer is not None:
            try:
                renderer.RemoveObserver(obs)
            except Exception:
                pass
            self._render_start_obs = None

        # drop the face picker's interactor observers (left/right button)
        picker = getattr(self, "picker", None)
        if picker is not None:
            try:
                picker.detach()
            except Exception:
                pass

        # the actual fix: Finalize() the render window (releases the GL
        # context / event loop hook) then close the interactor
        interactor = getattr(self, "interactor", None)
        if interactor is not None:
            try:
                render_window = interactor.GetRenderWindow()
                if render_window is not None:
                    render_window.Finalize()
            except Exception:
                pass
            try:
                interactor.close()
            except Exception:
                pass

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    # -- picking --------------------------------------------------------
    def _on_picked(self, body_name, face_id, mode):
        # a one-shot pick (armed by pick_face_once) consumes the NEXT click
        # without touching the persistent selection; a miss disarms too.
        if self._once_pick_cb is not None:
            cb = self._once_pick_cb
            self._once_pick_cb = None
            cb(body_name, face_id)
            return
        if body_name is None or face_id is None:
            return
        self.facePicked.emit(body_name, face_id, mode)

    def pick_face_once(self, callback):
        """Arm a one-shot face pick: the next click calls
        callback(body_name, face_id) instead of changing the selection
        (both None on an empty-space miss, which also disarms). Passing
        None cancels a pending arm."""
        self._once_pick_cb = callback

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

    # -- drag along a fixed axis -------------------------------------------
    def _display_ray(self, display_xy):
        """World-space (origin, direction) of the viewing ray through a
        display pixel (origin bottom-left, VTK convention)."""
        ren = self.renderer
        x, y = display_xy

        def world_at(z):
            ren.SetDisplayPoint(float(x), float(y), float(z))
            ren.DisplayToWorld()
            w = ren.GetWorldPoint()
            return np.array(w[:3]) / (w[3] if abs(w[3]) > 1e-12 else 1.0)

        near = world_at(0.0)
        far = world_at(1.0)
        d = far - near
        n = np.linalg.norm(d)
        return near, (d / n if n > 1e-12 else d)

    def _drag_to_axis(self, display_xy):
        """Point on the drag axis closest to the viewing ray through
        `display_xy` (line-line nearest point). Pure geometry given the
        current camera; exercised directly by tests."""
        if self._axis_drag is None:
            return None
        a0, ad = self._axis_drag["point"], self._axis_drag["dir"]
        r0, rd = self._display_ray(display_xy)
        a = float(np.dot(rd, rd))
        b = float(np.dot(rd, ad))
        c = float(np.dot(ad, ad))
        w0 = r0 - a0
        d = float(np.dot(rd, w0))
        e = float(np.dot(ad, w0))
        denom = a * c - b * b
        s = (e / c) if abs(denom) < 1e-12 else (a * e - b * d) / denom
        return a0 + s * ad

    def begin_axis_drag(self, axis_point, axis_dir, on_move, on_commit,
                        on_abort):
        """Enter a modal drag-along-axis mode: mouse motion reports the
        nearest axis point via on_move(world_point), a left click commits
        via on_commit(world_point), Esc aborts via on_abort(). Trackball
        camera control is suppressed for the duration (high-priority
        observers that abort the event) and fully restored on exit. No-op
        offscreen -- tests drive _drag_to_axis() directly."""
        self._axis_drag = {
            "point": np.asarray(axis_point, float),
            "dir": np.asarray(axis_dir, float)
            / max(np.linalg.norm(axis_dir), 1e-12),
            "on_move": on_move, "on_commit": on_commit,
            "on_abort": on_abort, "obs": []}
        if is_offscreen():
            return
        it = self.interactor
        self._axis_drag["obs"] = [
            it.AddObserver("MouseMoveEvent", self._axis_drag_move, 20.0),
            it.AddObserver("LeftButtonPressEvent",
                           self._axis_drag_click, 20.0),
            it.AddObserver("KeyPressEvent", self._axis_drag_key, 20.0)]

    def _end_axis_drag(self):
        drag = self._axis_drag
        self._axis_drag = None
        if drag is not None:
            for oid in drag.get("obs", []):
                self.interactor.RemoveObserver(oid)

    def _axis_drag_move(self, obj, event):
        obj.SetAbortFlag(1)                       # keep the trackball out
        pt = self._drag_to_axis(self.interactor.GetEventPosition())
        if pt is not None and self._axis_drag is not None:
            self._axis_drag["on_move"](pt)

    def _axis_drag_click(self, obj, event):
        obj.SetAbortFlag(1)
        pt = self._drag_to_axis(self.interactor.GetEventPosition())
        cb = self._axis_drag["on_commit"] if self._axis_drag else None
        self._end_axis_drag()
        if cb is not None and pt is not None:
            cb(pt)

    def _axis_drag_key(self, obj, event):
        if self.interactor.GetKeySym() not in ("Escape", "Return", "KP_Enter"):
            return
        obj.SetAbortFlag(1)
        cb = self._axis_drag["on_abort"] if self._axis_drag else None
        self._end_axis_drag()
        if cb is not None:
            cb()

    # -- internals ----------------------------------------------------------
    def _build_body_actors(self, body, faces_dict):
        name = body["name"]
        transform = placement_to_vtk_transform(body.get("placement", {}))
        self._body_transforms[name] = transform
        role = role_for_body(body)
        color, opacity = body_style(body)

        face_entries = (faces_dict.get(name) or {}).get("faces", [])
        actors = []
        for f in face_entries:
            reader = vtkSTLReader()
            reader.SetFileName(f["stl"])
            clean = vtkCleanPolyData()   # STL has unshared vertices; merge points first
            clean.SetInputConnection(reader.GetOutputPort())
            normals = vtkPolyDataNormals()
            normals.SetInputConnection(clean.GetOutputPort())
            normals.SplittingOff()       # per-face STLs: each actor is one smooth
                                          # face, no sharp edges within an actor
            normals.ComputePointNormalsOn()
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())

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
        # Re-apply per-actor style (selection highlight AND/OR ghosting)
        # unconditionally -- freshly (re)built actors (bodiesReshaped ->
        # reload_bodies -> _build_body_actors) must pick up whatever
        # exclusion/selection state is already live, not just the base
        # role color set above.
        for actor in actors:
            _, face_id = self._actor_face_map[actor]
            self._apply_face_style(actor, face_id in self._selection)

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
        # a full reload means all-new world geometry: stale chain-link
        # endpoints from the previous document would be meaningless (and
        # possibly reference a since-vanished element), so drop them too
        # -- the caller re-supplies fresh links after the new scene loads.
        self.set_chain_links([])

    # -- selection / highlighting -----------------------------------------
    def set_selection(self, face_ids):
        self._selection = set(face_ids or [])
        for actor, (_, face_id) in self._actor_face_map.items():
            self._apply_face_style(actor, face_id in self._selection)
        self._render()

    def clear_highlights(self):
        self.set_selection(set())

    def _apply_face_style(self, actor, selected):
        """Compose the three independent style layers for one face actor:
        base role color -> ghosted (train-excluded) dim/grey -> selection
        highlight. A ghosted body stays ghosted even when selected (dim +
        grey), just gains the selection color, so the two states never
        fight over which color wins."""
        body_name, _ = self._actor_face_map.get(actor, (None, None))
        ghosted = body_name in self._excluded_bodies
        prop = actor.GetProperty()
        color, opacity = self._actor_base_style.get(
            actor, (_SELECTED_COLOR, 1.0))
        if ghosted:
            color, opacity = _ghost_color(color), _GHOST_OPACITY
            prop.SetSpecular(0.0)
        if selected and not ghosted:
            color, opacity = _SELECTED_COLOR, 1.0
        prop.SetColor(*color)
        prop.SetOpacity(opacity)

    # -- train indicators: ghosted exclusion + chain-link overlay ---------
    def set_excluded_bodies(self, names):
        """Ghost every body in `names` (dim opacity ~0.25, grey tint, no
        specular pop) and restore normal role styling to any body that
        left the set. Composes with selection highlighting (see
        _apply_face_style) and is stored as instance state so it survives
        a bodiesReshaped rebuild -- _build_body_actors re-applies it to
        every freshly built actor."""
        self._excluded_bodies = set(names or [])
        for actor, (_, face_id) in self._actor_face_map.items():
            self._apply_face_style(actor, face_id in self._selection)
        self._render()

    def set_chain_links(self, links):
        """Dotted polyline overlay for the optical-train chain/fold
        linkage: links = [{"from": [x,y,z], "to": [x,y,z], "kind":
        "chain"|"fold"}] in mm world coordinates (scaled to the scene's
        metre convention here). "chain" links render cool blue, "fold"
        links orange; set_chain_links([]) clears the overlay. Lives in
        its own PickableOff() actor so FacePicker's vtkCellPicker never
        resolves a click onto a linkage line (see widgets/facepicker.py --
        a miss there is only guaranteed for actors the picker can't even
        select). The caller passes fresh geometry after every move; there
        is no per-body bookkeeping to maintain here."""
        if self._chain_links_actor is not None:
            self.renderer.RemoveActor(self._chain_links_actor)
            self._chain_links_actor = None
            self._chain_links_polydata = None
        if not links:
            self._render()
            return

        points = vtkPoints()
        cells = vtkCellArray()
        colors = vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("rgb")
        for link in links:
            a = np.asarray(link["from"], dtype=float) * 1e-3
            b = np.asarray(link["to"], dtype=float) * 1e-3
            i0 = points.InsertNextPoint(*(float(c) for c in a))
            i1 = points.InsertNextPoint(*(float(c) for c in b))
            cells.InsertNextCell(2)
            cells.InsertCellPoint(i0)
            cells.InsertCellPoint(i1)
            kind = link.get("kind", "chain")
            rgb = _FOLD_LINK_COLOR if kind == "fold" else _CHAIN_LINK_COLOR
            colors.InsertNextTuple3(*(int(round(255 * c)) for c in rgb))

        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(cells)
        polydata.GetCellData().SetScalars(colors)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarModeToUseCellData()
        mapper.ScalarVisibilityOn()

        actor = vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetLineWidth(_LINK_LINE_WIDTH)
        prop.SetOpacity(_LINK_OPACITY)
        prop.SetLighting(False)
        prop.SetLineStipplePattern(_LINK_STIPPLE_PATTERN)
        prop.SetLineStippleRepeatFactor(_LINK_STIPPLE_REPEAT)
        actor.PickableOff()

        self._chain_links_actor = actor
        self._chain_links_polydata = polydata
        self.renderer.AddActor(actor)
        self._render()

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
    def _compose_dim_rgba(self, polydata):
        """Build (or refresh) the per-cell 'rgba_dim' uchar array on
        `polydata`: the 'rgb' wavelength triple plus alpha from
        'rel_power' (each segment's power relative to its ray's power at
        the source) through the current dimming curve and floor. Returns
        True when the array was (re)built, False when the polydata lacks
        the inputs (legacy rays.vtp predating rel_power)."""
        from vtkmodules.util.numpy_support import (numpy_to_vtk,
                                                   vtk_to_numpy)
        cell_data = polydata.GetCellData()
        rgb_array = cell_data.GetArray("rgb")
        rel_array = cell_data.GetArray("rel_power")
        if rgb_array is None or rel_array is None:
            return False
        rgb = vtk_to_numpy(rgb_array).reshape(-1, 3)
        rel = np.clip(vtk_to_numpy(rel_array), 0.0, 1.0)
        a = np.sqrt(rel) if self._dim_mode == "sqrt" else rel
        a = np.maximum(a, self._dim_floor / 100.0)
        rgba = np.empty((rgb.shape[0], 4), dtype=np.uint8)
        rgba[:, :3] = rgb
        rgba[:, 3] = np.clip(np.round(255.0 * a), 0, 255).astype(np.uint8)
        vtk_rgba = numpy_to_vtk(rgba, deep=1)
        vtk_rgba.SetName("rgba_dim")
        cell_data.AddArray(vtk_rgba)   # replaces any previous instance
        return True

    def _apply_overlay_coloring(self, actor, polydata):
        """Wavelength coloring: color by the per-cell 'rgb' array
        (written by raytracer.vtkexport.write_vtp_polylines from each ray
        segment's wavelength) when present, else a uniform yellow. With
        ray dimming on (set_ray_dimming) and a rel_power-carrying file,
        color by a composed 4-component 'rgba_dim' array instead --
        DirectScalars treats the 4th uchar component as per-cell alpha
        and VTK routes the actor through the translucent pass by itself.

        Mode is UseCellFIELDData, not UseCellData: the latter only ever
        colors by the ACTIVE cell scalars, so a rays.vtp whose 'rgb' array
        wasn't marked active (every file written before the vtkexport
        Scalars= fix) silently rendered flat white. Field-data mode honors
        SelectColorArray by NAME, which works for old and new files."""
        mapper = actor.GetMapper()
        cell_data = polydata.GetCellData() if polydata is not None else None
        rgb_array = (cell_data.GetArray("rgb")
                     if cell_data is not None else None)
        if rgb_array is not None:
            color_array = "rgb"
            if self._dim_mode != "off" and self._compose_dim_rgba(polydata):
                color_array = "rgba_dim"
            mapper.SetScalarModeToUseCellFieldData()
            mapper.SelectColorArray(color_array)
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()
        else:
            mapper.ScalarVisibilityOff()
            actor.GetProperty().SetColor(1.0, 0.9, 0.2)

    def ray_dimming_data_missing(self):
        """True when dimming is requested but the loaded overlay lacks the
        rel_power cell array (a rays.vtp from before the feature) -- the
        extinction setting is silently inert on such data, and the shell
        surfaces a status-bar hint instead of leaving the user guessing."""
        if self._dim_mode == "off" or self._rays_polydata is None:
            return False
        cell_data = self._rays_polydata.GetCellData()
        return (cell_data.GetArray("rgb") is not None
                and cell_data.GetArray("rel_power") is None)

    def set_ray_dimming(self, mode, floor_pct=0.0):
        """Attenuation dimming for the ray overlay. mode: 'off' | 'linear'
        (opacity = P/P_birth) | 'sqrt' (perceptual, sqrt of that);
        floor_pct: minimum opacity in percent. Applies immediately to a
        loaded overlay and to every overlay loaded later; a stale (greyed)
        overlay just stores the state -- un-staling re-runs
        _apply_overlay_coloring, which reads it."""
        self._dim_mode = str(mode)
        self._dim_floor = float(floor_pct)
        if self._rays_actor is not None and not self._overlay_stale:
            self._apply_overlay_coloring(self._rays_actor,
                                         self._rays_polydata)
            self._render()

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
        self.overlayChanged.emit()
        return actor

    def remove_overlay(self):
        if self._rays_actor is not None:
            self.renderer.RemoveActor(self._rays_actor)
            self._rays_actor = None
            self._rays_polydata = None
            self._overlay_stale = False
            self._render()
            self.overlayChanged.emit()

    def set_overlay_stale(self, stale):
        """Grey the ray overlay itself (not just a button label) while the
        scene has changed since the rays were generated: uniform grey at
        low opacity. Un-staling restores the wavelength coloring."""
        stale = bool(stale)
        if stale == self._overlay_stale and self._rays_actor is None:
            return
        changed = stale != self._overlay_stale
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
        if changed:
            self.overlayChanged.emit()

    def overlay_is_stale(self):
        return self._overlay_stale

    def set_overlay_visible(self, visible):
        if self._rays_actor is not None:
            self._rays_actor.SetVisibility(bool(visible))
            self._render()

    # -- rendering ------------------------------------------------------------
    def _render(self):
        # Once shutdown() has Finalize()d the render window a Render() call
        # dereferences freed native resources and segfaults. A late
        # sceneLoaded can still reach the view during teardown: closeEvent
        # shuts the panes down BEFORE project.shutdown() closes the
        # document, whose sceneLoaded then drives load_bodies -> _render on
        # the dead window. Respect _shutdown_done exactly like is_offscreen.
        if is_offscreen() or getattr(self, "_shutdown_done", False):
            return
        self.interactor.GetRenderWindow().Render()
