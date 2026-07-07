"""Face-orientation indicator glyphs for VtkSceneView.

Purely visual GUI chrome answering "which way is this element facing":
- red HALF-DISC on a source body's emission face (skipped for
  single-face closed bodies, i.e. spherical emitters with no meaningful
  emit direction) and on a detector body's detection face -- both located
  by the same closest-centroid-to-origin heuristic the extractor uses
  (core.facemaps.active_face_index);
- green DISC on the body-local +x face of aperture primitives
  (slit/iris/pinhole);
- blue DISC on the body-local +x face of every other traced optic (any
  body with a real material tag).

Never written to the FCStd and never read by the extractor -- the glyphs
use the tessellation-time centroid_m/normal_hint face metadata that
already flows through GeomCache into Project.faces, so placing them costs
no FreeCAD round-trip. normal_hint is FreeCAD's (sign-corrected)
normalAt(), NOT the physics contract's orientation_outward -- fine for a
visual cue, do not reuse this module for anything physical.

Two halves, same pattern as facepicker/vtkview's scale bar:
* pure numpy/py functions (classification, +x face rule, glyph point
  geometry) -- unit-tested directly with plain floats;
* FaceIndicatorLayer -- the VTK actor bookkeeping, owned by VtkSceneView
  (one per view). Actors are constructed GPU-free (safe offscreen), share
  the body's vtkTransform (placement moves follow for free) and are
  PickableOff so they can never steal a vtkCellPicker hit from the face
  underneath.
"""

import math
from collections import namedtuple

import numpy as np

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

from ..core.facemaps import active_face_index

INDICATOR_RED = (0.90, 0.15, 0.15)     # source emit / detector face
INDICATOR_BLUE = (0.15, 0.35, 0.95)    # generic optic local-+x face
INDICATOR_GREEN = (0.10, 0.80, 0.30)   # aperture (slit/iris/pinhole) +x

_APERTURE_KINDS = ("slit", "iris", "pinhole")

# glyph: "half_disc" (flat edge = orientation cue) | "disc"
IndicatorSpec = namedtuple("IndicatorSpec", "face_id glyph color")

_LIFT_M = 1e-5          # 10 µm off the face along its normal (z-fighting)
_RADIUS_FRACTION = 0.18
_RADIUS_FLOOR_M = 3e-4  # 0.3 mm -- tiny faces still get a visible dot
_SEGMENTS = 24


# ---------------------------------------------------------------------------
# pure half
# ---------------------------------------------------------------------------
def plus_x_face(faces_meta):
    """The face id with the most-positive body-local centroid x; ties
    broken by the largest +x normal_hint component. None when no face
    carries a centroid."""
    best = None   # (cx, nx, face_id)
    for f in faces_meta or []:
        c = f.get("centroid_m")
        if c is None:
            continue
        n = f.get("normal_hint") or (0.0, 0.0, 0.0)
        key = (float(c[0]), float(n[0]))
        if best is None or key > best[0]:
            best = (key, f["id"])
    return best[1] if best is not None else None


def indicator_radius(face_area_m2):
    """Glyph radius scaled to the face (metres)."""
    try:
        area = max(float(face_area_m2 or 0.0), 0.0)
    except (TypeError, ValueError):
        area = 0.0
    return max(_RADIUS_FRACTION * math.sqrt(area), _RADIUS_FLOOR_M)


def _face_by_index(faces_meta, index):
    for f in faces_meta or []:
        if f["id"].endswith(".Face%d" % index):
            return f["id"]
    return None


def classify_indicators(body, faces_meta, role):
    """[IndicatorSpec, ...] for one body. `role` is vtkview's
    role_for_body(body) result (passed in to avoid a circular import)."""
    props = body.get("properties") or {}
    faces_meta = faces_meta or []
    if role in ("source", "detector"):
        if role == "source" and len(faces_meta) <= 1:
            return []   # spherical emitter: no meaningful emit direction
        idx = active_face_index(props, faces_meta)
        if idx is None:
            return []
        face_id = _face_by_index(faces_meta, idx)
        if face_id is None:
            return []
        return [IndicatorSpec(face_id, "half_disc", INDICATOR_RED)]

    material = (props.get("material") or {}).get("value")
    if not isinstance(material, str) or not material.strip() \
            or material.strip().lower() == "none":
        return []   # untraced body: no indicator

    face_id = plus_x_face(faces_meta)
    if face_id is None:
        return []
    kind = (props.get("miewb_primitive") or {}).get("value") or ""
    if any(str(kind).startswith(k) for k in _APERTURE_KINDS):
        return [IndicatorSpec(face_id, "disc", INDICATOR_GREEN)]
    return [IndicatorSpec(face_id, "disc", INDICATOR_BLUE)]


def rotation_to_normal(normal):
    """3x3 rotation matrix mapping local +z to `normal` (Rodrigues); the
    remaining in-plane orientation is fixed deterministically by aligning
    local +x with the face-plane projection of body-local +y (falling
    back to +x when the normal is (anti)parallel to y)."""
    z = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(z)
    z = z / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
    ref = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(ref, z))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    x = ref - np.dot(ref, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def glyph_points(centroid_m, normal_hint, radius_m, glyph,
                 lift_m=_LIFT_M, segments=_SEGMENTS):
    """(points (N,3) ndarray in body-local metres, triangle index
    triples) for a disc or half-disc triangle fan centered on the face,
    lifted `lift_m` along the normal to kill z-fighting. The half-disc's
    flat edge runs along the local +x axis of rotation_to_normal()."""
    rot = rotation_to_normal(normal_hint)
    center = np.asarray(centroid_m, dtype=float) + \
        lift_m * rot[:, 2]
    sweep = math.pi if glyph == "half_disc" else 2.0 * math.pi
    n_arc = max(int(segments), 3) + 1
    points = [center]
    for i in range(n_arc):
        ang = sweep * i / (n_arc - 1)
        local = np.array([radius_m * math.cos(ang),
                          radius_m * math.sin(ang), 0.0])
        points.append(center + rot @ local)
    # the arc's last point duplicates its first for a full disc (0 and
    # 2*pi), so the plain fan below already closes it; for a half-disc
    # the fan stops at the flat edge, which is exactly the wanted shape
    tris = [(0, i, i + 1) for i in range(1, n_arc)]
    return np.array(points), tris


# ---------------------------------------------------------------------------
# VTK half
# ---------------------------------------------------------------------------
def _build_glyph_actor(spec, face_meta, transform):
    pts, tris = glyph_points(face_meta.get("centroid_m", (0.0, 0.0, 0.0)),
                             face_meta.get("normal_hint", (0.0, 0.0, 1.0)),
                             indicator_radius(face_meta.get("area_m2")),
                             spec.glyph)
    vpoints = vtkPoints()
    for p in pts:
        vpoints.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    cells = vtkCellArray()
    for a, b, c in tris:
        cells.InsertNextCell(3)
        cells.InsertCellPoint(a)
        cells.InsertCellPoint(b)
        cells.InsertCellPoint(c)
    polydata = vtkPolyData()
    polydata.SetPoints(vpoints)
    polydata.SetPolys(cells)

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.SetUserTransform(transform)
    actor.PickableOff()          # never steal a face pick
    prop = actor.GetProperty()
    prop.SetColor(*spec.color)
    prop.SetAmbient(1.0)         # flat, unlit UI chrome
    prop.SetDiffuse(0.0)
    prop.SetSpecular(0.0)
    prop.LightingOff()
    return actor


class FaceIndicatorLayer:
    """Per-view indicator actor bookkeeping (owned by VtkSceneView)."""

    def __init__(self, renderer):
        self.renderer = renderer
        self._body_actors = {}    # body_name -> [actor, ...]
        self._visible = True

    def rebuild_body(self, body, faces_meta, transform, role):
        name = body["name"]
        self.remove_body(name)
        faces_by_id = {f["id"]: f for f in faces_meta or []}
        actors = []
        for spec in classify_indicators(body, faces_meta, role):
            face_meta = faces_by_id.get(spec.face_id)
            if face_meta is None or face_meta.get("centroid_m") is None:
                continue
            actor = _build_glyph_actor(spec, face_meta, transform)
            actor.SetVisibility(self._visible)
            self.renderer.AddActor(actor)
            actors.append(actor)
        if actors:
            self._body_actors[name] = actors

    def remove_body(self, body_name):
        for actor in self._body_actors.pop(body_name, []):
            self.renderer.RemoveActor(actor)

    def clear(self):
        for name in list(self._body_actors):
            self.remove_body(name)

    def set_visible(self, visible):
        self._visible = bool(visible)
        for actors in self._body_actors.values():
            for actor in actors:
                actor.SetVisibility(self._visible)

    def actor_count(self):
        return sum(len(a) for a in self._body_actors.values())
