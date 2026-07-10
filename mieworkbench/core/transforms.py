"""Transform / reference-point engine for MieWorkbench positioning.

Pure numpy, no Qt, no FreeCAD - fully unit-testable. All lengths in mm
(FreeCAD native units for Placements); angles in degrees at the API
surface, radians internally.

Model
-----
A body's pose is a `Placement` (position + unit quaternion), exactly
FreeCAD's model. Positioning operations build a 4x4 world-frame matrix M
and PRE-multiply it onto the placement: P' = M @ P. That makes operations
composable and repeatable ("apply again" re-applies the same M), and an
operation history is just a list of matrices.

Reference points (the "about" of a rotation / the anchor of a relative
translation) resolve to a world-frame point from:
  - the origin, or any fixed user point;
  - a body (itself or another): its center of mass, its optical center,
    or a point at parameter t along the line through a chosen face
    (face centroid, along the face normal).
The optical center convention: the axis is the normal line through the
body's largest planar-or-spherical face centroid; the optical center is
the closest point on that line to the body's bbox center (falls back to
the bbox center itself for bodies with no usable face). This puts the
point on the optical axis at mid-body, which matches the lens-stack
intuition without needing lens-specific knowledge.

Body geometry arrives from two sources (see BodyState.from_worker):
  - the FreeCAD worker's get_structure (global-frame CoM/bbox at the
    placement it had when queried - the "structure placement");
  - the tessellation metadata (face centroids/normals in BODY-LOCAL
    metres).
Because the GUI moves bodies without round-tripping FreeCAD, BodyState
keeps the structure placement AND the current placement, converts the
structure-frame quantities to body-local once, and re-derives world
coordinates from the current placement on demand.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Quaternions (x, y, z, w) - FreeCAD's Rotation.Q order
# ---------------------------------------------------------------------------
def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < _EPS:
        raise ValueError("zero quaternion")
    return q / n


def quat_to_matrix(q):
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R):
    """3x3 rotation matrix -> quaternion (x,y,z,w), Shepperd's method."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
        w = (R[2, 1] - R[1, 2]) / s
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
        w = (R[0, 2] - R[2, 0]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
        w = (R[1, 0] - R[0, 1]) / s
    return quat_normalize([x, y, z, w])


def axis_angle_quat(axis, angle_deg):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < _EPS:
        raise ValueError("zero rotation axis")
    axis = axis / n
    half = np.radians(float(angle_deg)) / 2.0
    return np.array([*(axis * np.sin(half)), np.cos(half)])


def quat_to_axis_angle(q):
    """Quaternion (x,y,z,w) -> (unit axis, angle_deg). The identity
    rotation returns the +z axis and 0 degrees (any axis is valid)."""
    x, y, z, w = quat_normalize(q)
    w = float(np.clip(w, -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < _EPS:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return np.array([x, y, z]) / s, float(np.degrees(angle))


def quat_from_euler(rx_deg, ry_deg, rz_deg):
    """Intrinsic X-Y-Z Tait-Bryan angles (degrees) -> quaternion.
    R = Rx(rx) @ Ry(ry) @ Rz(rz): rotate about X, then the new Y, then
    the new Z."""
    R = (quat_to_matrix(axis_angle_quat([1, 0, 0], rx_deg))
         @ quat_to_matrix(axis_angle_quat([0, 1, 0], ry_deg))
         @ quat_to_matrix(axis_angle_quat([0, 0, 1], rz_deg)))
    return matrix_to_quat(R)


def euler_from_quat(q):
    """Quaternion -> intrinsic X-Y-Z Tait-Bryan angles (degrees),
    inverse of quat_from_euler. At the gimbal-lock pole (ry = +/-90 deg)
    the X/Z split is arbitrary; rz is pinned to 0 and rx carries the
    combined rotation (documented caveat - absolute entry stays exact)."""
    R = quat_to_matrix(q)
    sy = float(np.clip(R[0, 2], -1.0, 1.0))
    ry = np.arcsin(sy)
    if abs(np.cos(ry)) > 1e-9:
        rx = np.arctan2(-R[1, 2], R[2, 2])
        rz = np.arctan2(-R[0, 1], R[0, 0])
    else:                                   # gimbal lock: rx + rz coupled
        rx = np.arctan2(R[2, 1], R[1, 1])
        rz = 0.0
    return np.degrees([rx, ry, rz])


def align_rotation(from_axis, to_axis, keep_hemisphere=True):
    """Minimal rotation (quaternion) taking unit `from_axis` onto
    `to_axis` (Rodrigues / half-vector construction). When
    `keep_hemisphere` and the axes point into opposite hemispheres
    (dot < 0), align to -to_axis instead, so an already-placed element
    only tilts by the acute angle rather than flipping ~180 degrees."""
    a = np.asarray(from_axis, float)
    b = np.asarray(to_axis, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < _EPS or nb < _EPS:
        raise ValueError("zero alignment axis")
    a = a / na
    b = b / nb
    if keep_hemisphere and np.dot(a, b) < 0.0:
        b = -b
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if d > 1.0 - _EPS:
        return np.array([0.0, 0.0, 0.0, 1.0])          # already aligned
    if d < -1.0 + _EPS:                                 # antiparallel
        # any axis perpendicular to a works; pick a stable one
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < _EPS:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        return axis_angle_quat(perp, 180.0)
    axis = np.cross(a, b)
    return quat_normalize([*axis, 1.0 + d])             # half-vector form


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------
@dataclass
class Placement:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))   # mm
    quat: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))    # x,y,z,w

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float).reshape(3)
        self.quat = quat_normalize(self.quat)

    @classmethod
    def from_dict(cls, d):
        """From the FreeCAD worker's {"pos_mm": [...], "quat": [...]}."""
        return cls(np.array(d["pos_mm"], dtype=float),
                   np.array(d["quat"], dtype=float))

    def to_dict(self):
        return {"pos_mm": [float(v) for v in self.pos],
                "quat": [float(v) for v in self.quat]}

    def matrix(self):
        M = np.eye(4)
        M[:3, :3] = quat_to_matrix(self.quat)
        M[:3, 3] = self.pos
        return M

    @classmethod
    def from_matrix(cls, M):
        M = np.asarray(M, dtype=float)
        return cls(M[:3, 3].copy(), matrix_to_quat(M[:3, :3]))

    def transform_point(self, p_local):
        return quat_to_matrix(self.quat) @ np.asarray(p_local, float) \
            + self.pos

    def transform_vector(self, v_local):
        return quat_to_matrix(self.quat) @ np.asarray(v_local, float)

    def inverse_point(self, p_world):
        return quat_to_matrix(self.quat).T @ (
            np.asarray(p_world, float) - self.pos)

    def inverse_vector(self, v_world):
        return quat_to_matrix(self.quat).T @ np.asarray(v_world, float)


# ---------------------------------------------------------------------------
# World-frame operations (all return a 4x4 matrix M; apply as P' = M @ P)
# ---------------------------------------------------------------------------
def translate_matrix(v_mm):
    M = np.eye(4)
    M[:3, 3] = np.asarray(v_mm, dtype=float).reshape(3)
    return M


def rotate_matrix(axis, angle_deg, about_mm=(0.0, 0.0, 0.0)):
    """Rotation about an arbitrary world point: T(p) @ R @ T(-p)."""
    p = np.asarray(about_mm, dtype=float).reshape(3)
    R = np.eye(4)
    R[:3, :3] = quat_to_matrix(axis_angle_quat(axis, angle_deg))
    return translate_matrix(p) @ R @ translate_matrix(-p)


def apply_world(M, placement):
    """P' = M @ P (world-frame pre-multiplication)."""
    return Placement.from_matrix(np.asarray(M, float) @ placement.matrix())


def reflect_matrix(point, normal):
    """Householder reflection about the plane through `point` with
    `normal` (4x4, det = -1). Delegates to the shared stdlib train
    solver so the GUI and the headless permute path share ONE
    implementation. IMPROPER: reflect points/directions/planes with it,
    never a Placement (use train_solver.fold_rotation for those)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import train_solver
    return np.array(train_solver.reflect_matrix(
        [float(v) for v in point], [float(v) for v in normal]))


def project_point_on_axis(point, axis_point, axis_dir):
    """Foot of the perpendicular from `point` onto the line through
    `axis_point` with direction `axis_dir`, plus the signed distance
    `t_along` from `axis_point` to the foot (mm). `axis_dir` need not be
    unit."""
    p = np.asarray(point, float)
    a = np.asarray(axis_point, float)
    d = np.asarray(axis_dir, float)
    n = np.linalg.norm(d)
    if n < _EPS:
        raise ValueError("zero axis direction")
    d = d / n
    t = float(np.dot(p - a, d))
    return a + t * d, t


# ---------------------------------------------------------------------------
# Body state & reference-point resolution
# ---------------------------------------------------------------------------
@dataclass
class FaceInfo:
    id: str
    centroid_local_mm: np.ndarray     # body-local, mm
    normal_local: Optional[np.ndarray]
    area_m2: float


@dataclass
class BodyState:
    name: str
    label: str
    current: Placement                       # live GUI placement
    com_local_mm: np.ndarray                 # body-local center of mass
    bbox_center_local_mm: np.ndarray         # body-local bbox center
    faces: List[FaceInfo] = field(default_factory=list)

    @classmethod
    def from_worker(cls, body_dict, face_meta=None):
        """Build from the worker's get_structure body dict (+ optional
        tessellation face list with centroid_m / normal_hint, body-local
        METRES). CoM/bbox in the structure dict are world-frame at the
        structure placement; convert to body-local here so later GUI-side
        moves stay consistent."""
        struct_pl = Placement.from_dict(body_dict["placement"])
        com_world = np.array(body_dict["center_of_mass_mm"] or [0, 0, 0],
                             dtype=float)
        bb = body_dict["bbox_mm"]
        bbox_center_world = np.array(
            [(bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2])
        faces = []
        for f in face_meta or []:
            normal = f.get("normal_hint")
            faces.append(FaceInfo(
                id=f["id"],
                centroid_local_mm=np.array(f["centroid_m"], float) * 1000.0,
                normal_local=(None if normal is None
                              else np.asarray(normal, float)),
                area_m2=float(f.get("area_m2", 0.0))))
        return cls(name=body_dict["name"], label=body_dict["label"],
                   current=struct_pl,
                   com_local_mm=struct_pl.inverse_point(com_world),
                   bbox_center_local_mm=struct_pl.inverse_point(
                       bbox_center_world),
                   faces=faces)

    # -- world-frame views at the CURRENT placement ------------------------
    def com_world(self):
        return self.current.transform_point(self.com_local_mm)

    def bbox_center_world(self):
        return self.current.transform_point(self.bbox_center_local_mm)

    def face(self, face_id):
        for f in self.faces:
            if f.id == face_id:
                return f
        raise KeyError("body %s has no face %r" % (self.name, face_id))

    def face_centroid_world(self, face_id):
        return self.current.transform_point(
            self.face(face_id).centroid_local_mm)

    def face_normal_world(self, face_id):
        n = self.face(face_id).normal_local
        if n is None:
            raise ValueError("face %s has no usable normal" % face_id)
        return self.current.transform_vector(n)

    def _axis_face(self):
        """Largest face with a normal - the optical-axis convention."""
        usable = [f for f in self.faces if f.normal_local is not None]
        if not usable:
            return None
        return max(usable, key=lambda f: f.area_m2)

    def optical_center_world(self):
        """Closest point to the bbox center on the largest face's normal
        line; bbox center if no usable face exists."""
        f = self._axis_face()
        c = self.bbox_center_world()
        if f is None:
            return c
        p0 = self.current.transform_point(f.centroid_local_mm)
        n = self.current.transform_vector(f.normal_local)
        n = n / max(np.linalg.norm(n), _EPS)
        return p0 + np.dot(c - p0, n) * n

    def optical_axis_world(self):
        """Unit vector of the optical-axis convention (None if no face)."""
        f = self._axis_face()
        if f is None:
            return None
        n = self.current.transform_vector(f.normal_local)
        return n / max(np.linalg.norm(n), _EPS)

    def face_line_point_world(self, face_id, t_mm):
        """centroid + t * normal, world frame, t in mm."""
        p0 = self.face_centroid_world(face_id)
        n = self.face_normal_world(face_id)
        n = n / max(np.linalg.norm(n), _EPS)
        return p0 + float(t_mm) * n


# reference spec kinds -> required params
#   {"kind": "origin"}
#   {"kind": "fixed", "point_mm": [x, y, z]}
#   {"kind": "com",            "body": name}
#   {"kind": "optical_center", "body": name}
#   {"kind": "bbox_center",    "body": name}
#   {"kind": "face_point",     "body": name, "face": face_id, "t_mm": t}
class ReferenceResolver:
    def __init__(self, bodies: Dict[str, BodyState]):
        self.bodies = bodies

    def _body(self, name):
        try:
            return self.bodies[name]
        except KeyError:
            # allow label lookup for UI convenience
            matches = [b for b in self.bodies.values() if b.label == name]
            if len(matches) == 1:
                return matches[0]
            raise KeyError("unknown body %r" % name)

    def resolve_point(self, spec) -> np.ndarray:
        kind = spec.get("kind", "origin")
        if kind == "origin":
            return np.zeros(3)
        if kind == "fixed":
            return np.asarray(spec["point_mm"], dtype=float).reshape(3)
        body = self._body(spec["body"])
        if kind == "com":
            return body.com_world()
        if kind == "optical_center":
            return body.optical_center_world()
        if kind == "bbox_center":
            return body.bbox_center_world()
        if kind == "face_point":
            return body.face_line_point_world(spec["face"],
                                              spec.get("t_mm", 0.0))
        raise ValueError("unknown reference kind %r" % kind)

    def resolve_axis(self, spec) -> np.ndarray:
        """Axis specs: {"kind":"global","axis":"x|y|z"} |
        {"kind":"vector","vector":[..]} | {"kind":"face_normal","body":..,
        "face":..} | {"kind":"optical_axis","body":..} |
        {"kind":"two_points","a":<point spec>,"b":<point spec>}."""
        kind = spec.get("kind", "global")
        if kind == "global":
            unit = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
            return np.array(unit[spec.get("axis", "z")], dtype=float)
        if kind == "vector":
            v = np.asarray(spec["vector"], dtype=float)
            n = np.linalg.norm(v)
            if n < _EPS:
                raise ValueError("zero axis vector")
            return v / n
        if kind == "face_normal":
            n = self._body(spec["body"]).face_normal_world(spec["face"])
            return n / max(np.linalg.norm(n), _EPS)
        if kind == "optical_axis":
            n = self._body(spec["body"]).optical_axis_world()
            if n is None:
                raise ValueError("body %r has no optical axis"
                                 % spec["body"])
            return n
        if kind == "two_points":
            a = self.resolve_point(spec["a"])
            b = self.resolve_point(spec["b"])
            v = b - a
            n = np.linalg.norm(v)
            if n < _EPS:
                raise ValueError("axis endpoints coincide")
            return v / n
        raise ValueError("unknown axis kind %r" % kind)


# ---------------------------------------------------------------------------
# Operations (UI-level, serializable, repeatable)
# ---------------------------------------------------------------------------
@dataclass
class Operation:
    """A serializable positioning operation.

    kind "translate": params {"vector_mm": [x,y,z]}  (world frame), or
                      {"toward": <point spec>, "distance_mm": d, "from":
                       <point spec>} - translate along from->toward.
    kind "rotate":    params {"axis": <axis spec>, "angle_deg": a,
                      "about": <point spec>}.
    kind "set_placement": params {"pos_mm": [x,y,z], "quat": [x,y,z,w]} -
                      ABSOLUTE world placement (replaces the current pose,
                      not composed onto it). For direct field entry.
    """
    kind: str
    params: dict

    def matrix(self, resolver: ReferenceResolver) -> np.ndarray:
        if self.kind == "translate":
            if "vector_mm" in self.params:
                return translate_matrix(self.params["vector_mm"])
            src = resolver.resolve_point(self.params["from"])
            dst = resolver.resolve_point(self.params["toward"])
            v = dst - src
            n = np.linalg.norm(v)
            if n < _EPS:
                return np.eye(4)
            d = self.params.get("distance_mm")
            v = v if d is None else v / n * float(d)
            return translate_matrix(v)
        if self.kind == "rotate":
            axis = resolver.resolve_axis(self.params["axis"])
            about = resolver.resolve_point(
                self.params.get("about", {"kind": "origin"}))
            return rotate_matrix(axis, self.params["angle_deg"], about)
        if self.kind == "set_placement":
            return Placement(self.params["pos_mm"],
                             self.params["quat"]).matrix()
        raise ValueError("unknown operation kind %r" % self.kind)

    def apply(self, resolver: ReferenceResolver, body: BodyState):
        """Apply to a body's CURRENT placement; returns the new Placement
        (caller stores it / pushes undo). The matrix is resolved at apply
        time, so 'apply again' after other moves uses live references.
        `set_placement` is absolute - it ignores the current pose."""
        if self.kind == "set_placement":
            body.current = Placement(
                np.asarray(self.params["pos_mm"], float),
                np.asarray(self.params["quat"], float))
            return body.current
        M = self.matrix(resolver)
        body.current = apply_world(M, body.current)
        return body.current


def snap_to_axis_ops(body_state, target_point, target_axis,
                     keep_hemisphere=True):
    """Two Operations that snap `body_state` onto the optical axis line
    through `target_point` with direction `target_axis`:

      1. rotate the body's optical axis onto `target_axis`, about the
         body's own optical center (so the center stays fixed);
      2. translate the optical center onto the axis LINE by its
         perpendicular offset, preserving the along-axis coordinate.

    Both operations carry WORLD-frame constants captured now (a fixed
    about-point and a world vector), so applying them in sequence is
    deterministic and undoes cleanly. The body is not otherwise moved
    along the beam - the caller supplies any along-axis offset as a
    further translate."""
    a_hat = np.asarray(target_axis, float)
    n = np.linalg.norm(a_hat)
    if n < _EPS:
        raise ValueError("zero target axis")
    a_hat = a_hat / n
    c = body_state.optical_center_world()
    src_axis = body_state.optical_axis_world()
    ops = []
    if src_axis is not None:
        q = align_rotation(src_axis, a_hat, keep_hemisphere=keep_hemisphere)
        axis, angle_deg = quat_to_axis_angle(q)
        if abs(angle_deg) > _EPS:
            ops.append(Operation("rotate", {
                "axis": {"kind": "vector", "vector": [float(v) for v in axis]},
                "angle_deg": float(angle_deg),
                "about": {"kind": "fixed",
                          "point_mm": [float(v) for v in c]}}))
    to_target = np.asarray(target_point, float) - c
    delta = to_target - np.dot(to_target, a_hat) * a_hat   # perp component
    if np.linalg.norm(delta) > _EPS:
        ops.append(Operation("translate",
                             {"vector_mm": [float(v) for v in delta]}))
    return ops


def element_bounds(bodies, body_states, names):
    """World AABB (mm) of the named bodies as ([xmin,ymin,zmin],
    [xmax,ymax,zmax]), or None if nothing had a bbox.

    The worker's bbox_mm is world-space AS OF the last structure fetch;
    pure GUI-side moves only update BodyState, so the box is corrected by
    the body's (current - fetched) translation. Rotation drift is ignored
    - this feeds paste-offset placement heuristics, not physics."""
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    found = False
    for b in bodies:
        if b["name"] not in names:
            continue
        bb = b.get("bbox_mm")
        if not bb:
            continue
        delta = [0.0, 0.0, 0.0]
        state = (body_states or {}).get(b["name"])
        if state is not None:
            cur = state.current.to_dict()["pos_mm"]
            orig = (b.get("placement") or {}).get("pos_mm", cur)
            delta = [c - o for c, o in zip(cur, orig)]
        for k in range(3):
            lo[k] = min(lo[k], bb[k] + delta[k])
            hi[k] = max(hi[k], bb[3 + k] + delta[k])
        found = True
    return ([*lo], [*hi]) if found else None
