"""Optical-train chain solver — the ONE implementation shared by the GUI
(mieworkbench.core.train) and the headless variant permuter
(permute_model.py via train_fcstd.py).

PURE STDLIB by contract: FreeCAD's embedded python has no numpy, and this
module must produce bit-identical placements in both interpreters. All
math is plain-float lists; lengths in mm, angles in degrees at the API
surface. Quaternions are (x, y, z, w) — FreeCAD's Rotation.Q order — and
placements are the worker dict {"pos_mm": [x,y,z], "quat": [x,y,z,w]}.
World-frame 4x4 matrices PRE-multiply placements (P' = M @ P), matching
mieworkbench.core.transforms.

Model
-----
An element is either ANCHORED (its world placement is authored directly)
or CHAINED: derived from an edge {reference element, exit port, distance,
decenter, tilt, ...}. Chains form a TREE (beamsplitters have several exit
ports); the solve walks it in topological order.

A beam PORT FRAME is {"origin", "dir", "up"}: origin on the beam axis
(mm, world), dir the unit propagation direction, up a unit transverse
reference. The right-handed transverse basis is u = up x dir, v = up
(so u x v = dir); decenter_x/decenter_y are measured along (u, v).

Distance is VERTEX-TO-VERTEX along the beam: from the parent port origin
(the parent's exit vertex, or the beam/mirror-plane intersection for
reflect ports) to this element's ENTRY vertex, measured along `dir`.

A chained element's orientation is built RELATIVE TO the parent beam
frame (its local optical axis maps onto `dir`, its local up onto `up`,
then tilts apply about the beam axes). Because of that, re-solving after
a fold-state change rotates the whole downstream arm rigidly — folds
need no special downstream handling.

Port propagation:
  * transmissive ports ("out"/"transmit") never redirect the train: the
    outgoing dir/up equal the incoming ones and the origin is the
    incoming beam line intersected with the exit-vertex plane;
  * "reflect" ports mirror the frame about the element's ACTUAL placed
    reflect plane (dir and up reflect; u is re-derived to keep the triad
    right-handed — a raw reflection is improper, det = -1);
  * "deviate" ports (prisms, gratings used in a bent train) rotate the
    frame by an expression-driven deviation/azimuth about the port
    origin.
  * an UNFOLDED fold propagates the incoming frame unchanged from the
    same origin (pass-through), so downstream distances keep their
    meaning and re-folding is a pure re-solve.

Expressions
-----------
Every numeric edge field accepts `2*gap + 5`-style expressions over the
global variables (miewb_vars). Grammar: numbers, variable names, + - * /,
unary +/- and parentheses, the constant `pi`, and a whitelisted set of
math functions: sin cos tan asin acos atan atan2 sqrt abs radians
degrees, plus radian-argument variants sinr cosr tanr asinr acosr atanr
atan2r. TRIG IS DEGREES-NATIVE by project convention (the tilt / fold
deviation / azimuth fields are all in degrees): `sin(30)` == 0.5 and
asin/acos/atan/atan2 RETURN degrees; the `*r` variants take/return
radians like Python's math module. A user variable named `pi` (or any
function name) shadows the builtin — variables win. Nothing else: no
other calls, no attributes, no keyword arguments. NOTE: dim-sheet cell
expressions (`=<<miewb_vars>>.x * 1mm`) go through FreeCAD's own
expression engine, NOT this grammar — the two diverge deliberately.
Variables may reference each other; resolve_variables() evaluates them in
dependency order and reports circular references by naming the full
cycle path.

Anchored elements normally carry a literal world placement, but they may
ALSO be expression-driven: rec["pose_expr"] holds a {field: expr} subset
of POSE_EXPR_FIELDS (pos_x/pos_y/pos_z world position components and
rot_rx/rot_ry/rot_rz world Euler angles). place_anchored() bakes those
against the same variables, so a goniometer detector at `R*cos(theta)`,
`R*sin(theta)` re-solves when the variables change. Pose expressions are
valid on anchored elements ONLY (a chained element with one is an error);
solve_chain returns such baked placements alongside the chained ones.
"""

import ast
import math

EPS = 1e-12

DEG = math.pi / 180.0

TRANSMIT_PORTS = ("out", "transmit")
ROT_ORDERS = ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx")

# Anchored-pose expression fields (see place_anchored). pos_x/y/z override
# the corresponding literal world position component; rot_rx/ry/rz drive the
# world orientation as an intrinsic Euler triad (rec["rot_order"], default
# "xyz" — the SAME euler_matrix3 convention the chain tilt fields use).
POSE_POS_FIELDS = ("pos_x", "pos_y", "pos_z")
POSE_ROT_FIELDS = ("rot_rx", "rot_ry", "rot_rz")
POSE_EXPR_FIELDS = POSE_POS_FIELDS + POSE_ROT_FIELDS


class TrainError(ValueError):
    """Any solver-level failure (bad expression, cycle, missing ref)."""


class ExprError(TrainError):
    """Expression parse/evaluate failure."""


class CycleError(TrainError):
    """Circular reference; .path holds the offending chain."""

    def __init__(self, message, path):
        super().__init__(message)
        self.path = list(path)


# ---------------------------------------------------------------------------
# Expression evaluation (+ - * / and whitelisted functions over variables)
# ---------------------------------------------------------------------------
_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}

# Degrees-native trig (project convention: every angle field in the chain
# recipe is degrees); the `*r` variants are the plain-radian math calls.
_ALLOWED_FUNCS = {
    "sin": lambda x: math.sin(x * DEG),
    "cos": lambda x: math.cos(x * DEG),
    "tan": lambda x: math.tan(x * DEG),
    "asin": lambda x: math.asin(x) / DEG,
    "acos": lambda x: math.acos(x) / DEG,
    "atan": lambda x: math.atan(x) / DEG,
    "atan2": lambda y, x: math.atan2(y, x) / DEG,
    "sinr": math.sin,
    "cosr": math.cos,
    "tanr": math.tan,
    "asinr": math.asin,
    "acosr": math.acos,
    "atanr": math.atan,
    "atan2r": math.atan2,
    "sqrt": math.sqrt,
    "abs": abs,
    "radians": math.radians,
    "degrees": math.degrees,
}
_FUNC_ARITY = {"atan2": 2, "atan2r": 2}   # everything else takes 1
_CONSTANTS = {"pi": math.pi}

# The ONE grammar description every GUI tooltip / error surface reuses.
EXPR_HELP = (
    "Expressions: numbers, variables, + - * / ( ), constant pi, and "
    "functions sin cos tan asin acos atan atan2 sqrt abs radians degrees "
    "— trig is in DEGREES (sin(30)=0.5; asin returns degrees); "
    "sinr/cosr/tanr/asinr/acosr/atanr/atan2r take radians instead."
)


def _eval_node(node, variables, expr):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables, expr)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return float(node.value)
        raise ExprError("non-numeric constant %r in %r"
                        % (node.value, expr))
    if isinstance(node, ast.Name):
        # Variables first: a user variable shadows the builtin constants.
        if node.id in variables:
            return float(variables[node.id])
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ExprError("unknown variable %r in %r (defined variables: "
                        "%s; constants: %s)"
                        % (node.id, expr,
                           ", ".join(sorted(variables)) or "none",
                           ", ".join(sorted(_CONSTANTS))))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) \
                or node.func.id not in _ALLOWED_FUNCS:
            fname = (node.func.id if isinstance(node.func, ast.Name)
                     else type(node.func).__name__)
            raise ExprError("function %r not allowed in %r (allowed: %s)"
                            % (fname, expr,
                               " ".join(sorted(_ALLOWED_FUNCS))))
        if node.keywords:
            raise ExprError("keyword arguments not allowed in %r" % expr)
        fname = node.func.id
        arity = _FUNC_ARITY.get(fname, 1)
        if len(node.args) != arity:
            raise ExprError("%s() takes exactly %d argument%s in %r"
                            % (fname, arity, "s" if arity != 1 else "",
                               expr))
        args = [_eval_node(a, variables, expr) for a in node.args]
        try:
            return float(_ALLOWED_FUNCS[fname](*args))
        except (ValueError, OverflowError, ZeroDivisionError) as e:
            # Wrap so GUI panes (which only catch TrainError) never see a
            # raw ValueError from e.g. sqrt(-1) or asin(2).
            raise ExprError("%s() domain error in %r: %s"
                            % (fname, expr, e))
    if isinstance(node, ast.BinOp):
        for op_type, fn in _ALLOWED_BINOPS.items():
            if isinstance(node.op, op_type):
                a = _eval_node(node.left, variables, expr)
                b = _eval_node(node.right, variables, expr)
                if isinstance(node.op, ast.Div) and b == 0.0:
                    raise ExprError("division by zero in %r" % expr)
                return fn(a, b)
        raise ExprError("operator %s not allowed in %r (only + - * /)"
                        % (type(node.op).__name__, expr))
    if isinstance(node, ast.UnaryOp):
        v = _eval_node(node.operand, variables, expr)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise ExprError("unary %s not allowed in %r"
                        % (type(node.op).__name__, expr))
    raise ExprError("%s not allowed in %r (numbers, variables, + - * / "
                    "and whitelisted functions only)"
                    % (type(node).__name__, expr))


def eval_expr(expr, variables=None):
    """Evaluate `expr` (a number, or an expression over variable names —
    see EXPR_HELP for the grammar) to a float. `variables`: {name: float}.
    Raises ExprError."""
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return float(expr)
    text = str(expr).strip()
    if not text:
        raise ExprError("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ExprError("cannot parse %r: %s" % (text, e))
    return float(_eval_node(tree, variables or {}, text))


def expr_names(expr):
    """The set of variable names an expression references ([] for plain
    numbers). Function names in call position are NOT references (else
    `sin(gap)` would report a phantom `sin` dependency); `pi` IS included
    when used — harmless to dependency resolution (eval falls back to the
    constant) and correct when a user defines their own `pi` variable.
    Raises ExprError on unparseable input."""
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return set()
    text = str(expr).strip()
    if not text:
        raise ExprError("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ExprError("cannot parse %r: %s" % (text, e))
    func_names = {id(n.func) for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and id(n) not in func_names}


def resolve_variables(raw):
    """{name: raw value-or-expression} -> {name: float}, evaluating in
    dependency order. Detects circular references explicitly and raises
    CycleError naming the full path (e.g. "focus -> gap -> focus")."""
    values = {}
    state = {}                       # name -> "visiting" | "done"
    stack = []

    def visit(name):
        if state.get(name) == "done":
            return
        if state.get(name) == "visiting":
            cycle = stack[stack.index(name):] + [name]
            raise CycleError("circular variable reference: %s"
                             % " -> ".join(cycle), cycle)
        state[name] = "visiting"
        stack.append(name)
        try:
            for dep in sorted(expr_names(raw[name])):
                if dep in raw:
                    visit(dep)
                # unknown names fall through to eval_expr's error below
            values[name] = eval_expr(raw[name], values)
        finally:
            stack.pop()
        state[name] = "done"

    for name in sorted(raw):
        visit(name)
    return values


# ---------------------------------------------------------------------------
# Vector / quaternion / matrix math (plain lists; ports of
# mieworkbench.core.transforms so both sides agree bit-for-bit)
# ---------------------------------------------------------------------------
def vadd(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vsub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vscale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def vnorm(a):
    return math.sqrt(vdot(a, a))


def vunit(a, what="vector"):
    n = vnorm(a)
    if n < EPS:
        raise TrainError("zero %s" % what)
    return vscale(a, 1.0 / n)


def reflect_dir(d, normal):
    """Reflect direction `d` about a plane with unit `normal`."""
    n = vunit(normal, "mirror normal")
    return vsub(d, vscale(n, 2.0 * vdot(d, n)))


def quat_normalize(q):
    n = math.sqrt(sum(c * c for c in q))
    if n < EPS:
        raise TrainError("zero quaternion")
    return [c / n for c in q]


def quat_to_matrix3(q):
    x, y, z, w = quat_normalize(q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def matrix3_to_quat(R):
    """3x3 rotation -> quaternion (x,y,z,w); Shepperd's method, exactly
    as transforms.matrix_to_quat."""
    t = R[0][0] + R[1][1] + R[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2][1] - R[1][2]) / s
        y = (R[0][2] - R[2][0]) / s
        z = (R[1][0] - R[0][1]) / s
    elif R[0][0] >= R[1][1] and R[0][0] >= R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        x = 0.25 * s
        y = (R[0][1] + R[1][0]) / s
        z = (R[0][2] + R[2][0]) / s
        w = (R[2][1] - R[1][2]) / s
    elif R[1][1] >= R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        x = (R[0][1] + R[1][0]) / s
        y = 0.25 * s
        z = (R[1][2] + R[2][1]) / s
        w = (R[0][2] - R[2][0]) / s
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        x = (R[0][2] + R[2][0]) / s
        y = (R[1][2] + R[2][1]) / s
        z = 0.25 * s
        w = (R[1][0] - R[0][1]) / s
    return quat_normalize([x, y, z, w])


def axis_angle_matrix3(axis, angle_deg):
    """Rodrigues rotation matrix about unit-normalized `axis`."""
    u = vunit(axis, "rotation axis")
    c = math.cos(angle_deg * DEG)
    s = math.sin(angle_deg * DEG)
    x, y, z = u
    C = 1.0 - c
    return [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]


def mat3_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def mat3_vec(A, v):
    return [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]


def mat3_T(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def mat4_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def mat4_identity():
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def translate_matrix(v):
    M = mat4_identity()
    M[0][3], M[1][3], M[2][3] = float(v[0]), float(v[1]), float(v[2])
    return M


def rotate_matrix(axis, angle_deg, about=(0.0, 0.0, 0.0)):
    """Proper rotation about an arbitrary world point: T(p) R T(-p)."""
    R3 = axis_angle_matrix3(axis, angle_deg)
    M = mat4_identity()
    for i in range(3):
        for j in range(3):
            M[i][j] = R3[i][j]
    p = [float(c) for c in about]
    Rp = mat3_vec(R3, p)
    for i in range(3):
        M[i][3] = p[i] - Rp[i]
    return M


def reflect_matrix(point, normal):
    """Householder reflection about the plane through `point` with
    `normal`: M = I - 2nn^T (4x4, affine). det = -1 — an IMPROPER
    transform: valid for reflecting points, directions and planes, but
    NEVER apply it to a placement (quaternions represent proper rotations
    only; use fold_rotation for placements)."""
    n = vunit(normal, "mirror normal")
    M = mat4_identity()
    for i in range(3):
        for j in range(3):
            M[i][j] -= 2.0 * n[i] * n[j]
    d = 2.0 * vdot(point, n)
    for i in range(3):
        M[i][3] = d * n[i]
    return M


def placement_matrix(pl):
    """{"pos_mm","quat"} -> 4x4."""
    R = quat_to_matrix3(pl["quat"])
    M = mat4_identity()
    for i in range(3):
        for j in range(3):
            M[i][j] = R[i][j]
        M[i][3] = float(pl["pos_mm"][i])
    return M


def matrix_placement(M):
    """4x4 -> {"pos_mm","quat"} (rotation part must be proper)."""
    R = [row[:3] for row in M[:3]]
    det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
           - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
           + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    if det < 0.0:
        raise TrainError("improper transform (det<0) cannot become a "
                         "placement — use fold_rotation, not "
                         "reflect_matrix, on placements")
    return {"pos_mm": [M[0][3], M[1][3], M[2][3]],
            "quat": matrix3_to_quat(R)}


def apply_to_placement(M, pl):
    """P' = M @ P."""
    return matrix_placement(mat4_mul(M, placement_matrix(pl)))


def transform_point(pl, p_local):
    R = quat_to_matrix3(pl["quat"])
    return vadd(mat3_vec(R, list(p_local)), list(pl["pos_mm"]))


def transform_vector(pl, v_local):
    return mat3_vec(quat_to_matrix3(pl["quat"]), list(v_local))


# ---------------------------------------------------------------------------
# Euler tilt matrices (intrinsic, about the BEAM frame axes)
# ---------------------------------------------------------------------------
_AXIS_VEC = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0],
             "z": [0.0, 0.0, 1.0]}


def euler_matrix3(order, rx_deg, ry_deg, rz_deg):
    """Intrinsic Euler/Tait-Bryan rotation: for order "xyz" this is
    Rx @ Ry @ Rz (rotate about X, then the NEW Y, then the NEW Z) —
    matching transforms.quat_from_euler for the default order."""
    if order not in ROT_ORDERS:
        raise TrainError("rot order %r not one of %s"
                         % (order, "/".join(ROT_ORDERS)))
    angle = {"x": rx_deg, "y": ry_deg, "z": rz_deg}
    R = None
    for ax in order:
        Ra = axis_angle_matrix3(_AXIS_VEC[ax], angle[ax])
        R = Ra if R is None else mat3_mul(R, Ra)
    return R


def _clamp(v):
    return max(-1.0, min(1.0, v))


def euler_from_matrix3(R, order):
    """Inverse of euler_matrix3 (degrees). Formulas ported from the
    well-known intrinsic-order extraction set (three.js
    Euler.setFromRotationMatrix); at the gimbal pole the first angle
    carries the coupled rotation and the last is pinned to 0."""
    m11, m12, m13 = R[0]
    m21, m22, m23 = R[1]
    m31, m32, m33 = R[2]
    if order == "xyz":
        y = math.asin(_clamp(m13))
        if abs(m13) < 1.0 - 1e-9:
            x = math.atan2(-m23, m33)
            z = math.atan2(-m12, m11)
        else:
            x = math.atan2(m32, m22)
            z = 0.0
    elif order == "yxz":
        x = math.asin(-_clamp(m23))
        if abs(m23) < 1.0 - 1e-9:
            y = math.atan2(m13, m33)
            z = math.atan2(m21, m22)
        else:
            y = math.atan2(-m31, m11)
            z = 0.0
    elif order == "zxy":
        x = math.asin(_clamp(m32))
        if abs(m32) < 1.0 - 1e-9:
            y = math.atan2(-m31, m33)
            z = math.atan2(-m12, m22)
        else:
            y = 0.0
            z = math.atan2(m21, m11)
    elif order == "zyx":
        y = math.asin(-_clamp(m31))
        if abs(m31) < 1.0 - 1e-9:
            x = math.atan2(m32, m33)
            z = math.atan2(m21, m11)
        else:
            x = 0.0
            z = math.atan2(-m12, m22)
    elif order == "yzx":
        z = math.asin(_clamp(m21))
        if abs(m21) < 1.0 - 1e-9:
            x = math.atan2(-m23, m22)
            y = math.atan2(-m31, m11)
        else:
            x = 0.0
            y = math.atan2(m13, m33)
    elif order == "xzy":
        z = math.asin(-_clamp(m12))
        if abs(m12) < 1.0 - 1e-9:
            x = math.atan2(m32, m22)
            y = math.atan2(m13, m11)
        else:
            x = math.atan2(-m23, m33)
            y = 0.0
    else:
        raise TrainError("rot order %r not one of %s"
                         % (order, "/".join(ROT_ORDERS)))
    out = {"x": x / DEG, "y": y / DEG, "z": z / DEG}
    return out["x"], out["y"], out["z"]


# ---------------------------------------------------------------------------
# Beam port frames
# ---------------------------------------------------------------------------
def stable_up(direction):
    """A deterministic unit `up` perpendicular to `direction`: world +z
    projected out, falling back to +y for near-vertical beams."""
    d = vunit(direction, "beam direction")
    for ref in ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]):
        up = vsub(ref, vscale(d, vdot(ref, d)))
        n = vnorm(up)
        if n > 1e-6:
            return vscale(up, 1.0 / n)
    raise TrainError("cannot derive an up vector")     # unreachable


def frame_basis(frame):
    """(u, v, d): right-handed transverse basis of a port frame with
    v = up, d = dir, u = up x dir (u x v = d)."""
    d = vunit(frame["dir"], "frame dir")
    v = vunit(frame["up"], "frame up")
    v = vunit(vsub(v, vscale(d, vdot(v, d))), "frame up")  # re-orthogonalize
    u = vcross(v, d)
    return u, v, d


def make_frame(origin, direction, up=None):
    d = vunit(direction, "beam direction")
    if up is None:
        v = stable_up(d)
    else:
        v = vsub(list(up), vscale(d, vdot(list(up), d)))
        n = vnorm(v)
        v = stable_up(d) if n < 1e-9 else vscale(v, 1.0 / n)
    return {"origin": [float(c) for c in origin],
            "dir": d, "up": v}


def fold_rotation(incoming_dir, mirror_point, mirror_normal):
    """The PROPER rigid transform that folds a downstream train about a
    mirror plane: a rotation by the deviation angle about the fold line
    (axis = d_in x d_out through `mirror_point`). Returns (4x4 matrix,
    reflected unit direction). At normal incidence (d_out = -d_in) the
    fold axis is degenerate; any axis in the mirror plane works and a
    stable one is chosen."""
    d_in = vunit(incoming_dir, "incoming direction")
    d_out = vunit(reflect_dir(d_in, mirror_normal), "reflected direction")
    axis = vcross(d_in, d_out)
    if vnorm(axis) < 1e-12:
        if vdot(d_in, d_out) > 0.0:                    # grazing no-op
            return mat4_identity(), d_out
        n = vunit(mirror_normal, "mirror normal")
        axis = vcross(d_in, n)                          # normal incidence
        if vnorm(axis) < 1e-12:
            axis = stable_up(d_in)
        return rotate_matrix(axis, 180.0, mirror_point), d_out
    angle = math.degrees(math.acos(_clamp(vdot(d_in, d_out))))
    return rotate_matrix(axis, angle, mirror_point), d_out


# ---------------------------------------------------------------------------
# Element records
# ---------------------------------------------------------------------------
# An element RECORD is a plain dict (built by core/train.py from body
# properties, or by train_fcstd.py from the FreeCAD document):
#   {
#     "label": str,                     element identity (miewb_group)
#     "mode": "anchored"|"chained",
#     "ref": str, "port": str,          chained only
#     "distance": expr, "decenter_x": expr, "decenter_y": expr,
#     "tilt_rx": expr, "tilt_ry": expr, "tilt_rz": expr,
#     "rot_order": "xyz", "pos_rot_order": "pos_first"|"rot_first",
#     "flip": bool,                     beam-side surface = local exit
#     "pivot": "entrance"|"center"|"exit"|"x,y,z",
#     "fold": bool, "folded": bool,
#     "fold_deviation": expr, "fold_azimuth": expr,   fold/deviate ports
#     "local": {                        element-LOCAL port geometry (mm)
#        "entry": [3], "exit": [3],     vertices on the local optical axis
#        "axis": [3],                   local beam direction entry->exit
#        "up": [3],                     local transverse reference
#        "reflect_plane": {"point": [3], "normal": [3]} | None,
#     },
#   }
_DEF_LOCAL = {"entry": [0.0, 0.0, 0.0], "exit": [0.0, 0.0, 0.0],
              "axis": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0],
              "reflect_plane": None}


def _local(rec):
    loc = dict(_DEF_LOCAL)
    loc.update(rec.get("local") or {})
    if "entry" not in (rec.get("local") or {}) and "exit" in (
            rec.get("local") or {}):
        loc["entry"] = loc["exit"]
    if "exit" not in (rec.get("local") or {}) and "entry" in (
            rec.get("local") or {}):
        loc["exit"] = loc["entry"]
    if rec.get("flip"):
        # flipped element: its (former) exit surface faces the beam. Swap
        # the port vertices and reverse the local axis; _beam_R then
        # orients the body 180 deg about `up` (deterministic), and the
        # chain distance still measures to the actual beam-side vertex.
        loc = dict(loc, entry=list(loc["exit"]), exit=list(loc["entry"]),
                   axis=[-c for c in loc["axis"]])
    return loc


def _pivot_local(rec, loc):
    pv = rec.get("pivot") or "entrance"
    if pv == "entrance":
        return list(loc["entry"])
    if pv == "exit":
        return list(loc["exit"])
    if pv == "center":
        return vscale(vadd(loc["entry"], loc["exit"]), 0.5)
    try:
        parts = [float(x) for x in str(pv).split(",")]
        if len(parts) != 3:
            raise ValueError
    except ValueError:
        raise TrainError("element %r: pivot %r is not entrance/center/"
                         "exit or 'x,y,z'" % (rec.get("label"), pv))
    return parts


def _edge_value(rec, key, variables, default=0.0):
    raw = rec.get(key)
    if raw in (None, ""):
        return float(default)
    try:
        return eval_expr(raw, variables)
    except ExprError as e:
        raise ExprError("element %r, field %s: %s"
                        % (rec.get("label"), key, e))


# ---------------------------------------------------------------------------
# Anchored pose expressions
# ---------------------------------------------------------------------------
def has_pose_expr(rec):
    """True when an element record carries any non-empty anchored-pose
    expression (rec["pose_expr"], a {field: expr} subset of
    POSE_EXPR_FIELDS)."""
    pe = rec.get("pose_expr") or {}
    return any(pe.get(k) not in (None, "") for k in POSE_EXPR_FIELDS)


def _pose_value(rec, field, raw, variables):
    try:
        return eval_expr(raw, variables)
    except ExprError as e:
        raise ExprError("element %r, pose field %s: %s"
                        % (rec.get("label"), field, e))


def place_anchored(rec, base, variables):
    """Bake an ANCHORED element's world placement from its pose
    expressions (rec["pose_expr"]) over the resolved `variables`. Returns
    `base` unchanged when the element carries no pose expression (today's
    literal behaviour — byte-identical).

    Semantics (kept deliberately simple + deterministic so both the GUI
    and the headless permute path produce bit-identical results):
      * pos_x/pos_y/pos_z — per-component: a component WITH an expression
        is evaluated; a component WITHOUT one keeps `base`'s literal
        position value.
      * rot_rx/rot_ry/rot_rz — if ANY rotation field is present, the WHOLE
        orientation is rebuilt from the three angles (a missing angle
        defaults to 0) as an intrinsic Euler triad in rec["rot_order"]
        (default "xyz"); if NO rotation field is present, `base`'s
        rotation is kept. Pose angles are WORLD-frame degrees (the tilt /
        chain convention — DEGREES-native).

    Pose expressions are ONLY valid on anchored elements: a chained
    element with a pose expression is a hard error (the two placement
    mechanisms are mutually exclusive)."""
    if not has_pose_expr(rec):
        return base
    if rec.get("mode") == "chained":
        raise TrainError(
            "element %r has anchored-pose expression(s) but is chained; "
            "pose expressions are valid on anchored elements only "
            "(unchain it, or clear the pose expressions)"
            % rec.get("label"))
    pe = rec.get("pose_expr") or {}
    if base is not None:
        pos = [float(c) for c in base["pos_mm"]]
        quat = [float(c) for c in base["quat"]]
    else:
        pos = [0.0, 0.0, 0.0]
        quat = [0.0, 0.0, 0.0, 1.0]
    for i, field in enumerate(POSE_POS_FIELDS):
        raw = pe.get(field)
        if raw not in (None, ""):
            pos[i] = _pose_value(rec, field, raw, variables)
    if any(pe.get(f) not in (None, "") for f in POSE_ROT_FIELDS):
        order = rec.get("rot_order") or "xyz"
        angs = []
        for field in POSE_ROT_FIELDS:
            raw = pe.get(field)
            angs.append(_pose_value(rec, field, raw, variables)
                        if raw not in (None, "") else 0.0)
        quat = matrix3_to_quat(euler_matrix3(order, *angs))
    return {"pos_mm": pos, "quat": quat}


# ---------------------------------------------------------------------------
# Chain topology
# ---------------------------------------------------------------------------
def sort_chain(records):
    """Topologically order element records (anchored roots first).
    `records`: {label: rec}. Raises CycleError naming the loop, or
    TrainError for a dangling reference."""
    order = []
    state = {}
    stack = []

    def visit(label):
        if state.get(label) == "done":
            return
        if state.get(label) == "visiting":
            cycle = stack[stack.index(label):] + [label]
            raise CycleError("circular train reference: %s"
                             % " -> ".join(cycle), cycle)
        state[label] = "visiting"
        stack.append(label)
        rec = records[label]
        if rec.get("mode") == "chained":
            ref = rec.get("ref")
            if not ref:
                raise TrainError("element %r is chained but has no "
                                 "reference" % label)
            if ref not in records:
                raise TrainError("element %r chains to unknown element %r"
                                 % (label, ref))
            visit(ref)
        stack.pop()
        state[label] = "done"
        order.append(label)

    for label in sorted(records):
        visit(label)
    return order


def downstream_of(records, label):
    """Labels of every element whose chain reference path leads through
    `label` (children, grandchildren, ... in topological order)."""
    children = {}
    for name, rec in records.items():
        if rec.get("mode") == "chained" and rec.get("ref"):
            children.setdefault(rec["ref"], []).append(name)
    out = []
    queue = sorted(children.get(label, []))
    while queue:
        cur = queue.pop(0)
        out.append(cur)
        queue.extend(sorted(children.get(cur, [])))
    return out


# ---------------------------------------------------------------------------
# Placement construction / inversion
# ---------------------------------------------------------------------------
def _beam_R(frame, loc):
    """World rotation aligning the element-local frame (axis, up) with
    the beam frame: columns of W are (u, v, d), columns of L are the
    local transverse triad; R0 = W @ L^T."""
    u, v, d = frame_basis(frame)
    la = vunit(loc["axis"], "local axis")
    lv = vsub(list(loc["up"]), vscale(la, vdot(list(loc["up"]), la)))
    n = vnorm(lv)
    lv = stable_up(la) if n < 1e-9 else vscale(lv, 1.0 / n)
    lu = vcross(lv, la)
    W = [[u[i], v[i], d[i]] for i in range(3)]
    L = [[lu[i], lv[i], la[i]] for i in range(3)]
    return mat3_mul(W, mat3_T(L))


def place_chained(parent_frame, rec, variables):
    """Solve one chained element's placement from its parent exit port
    frame. Returns {"pos_mm","quat"}."""
    loc = _local(rec)
    u, v, d = frame_basis(parent_frame)
    dist = _edge_value(rec, "distance", variables)
    dx = _edge_value(rec, "decenter_x", variables)
    dy = _edge_value(rec, "decenter_y", variables)
    rx = _edge_value(rec, "tilt_rx", variables)
    ry = _edge_value(rec, "tilt_ry", variables)
    rz = _edge_value(rec, "tilt_rz", variables)
    order = rec.get("rot_order") or "xyz"
    pos_rot = rec.get("pos_rot_order") or "pos_first"
    if pos_rot not in ("pos_first", "rot_first"):
        raise TrainError("element %r: pos_rot_order %r not pos_first/"
                         "rot_first" % (rec.get("label"), pos_rot))

    o = list(parent_frame["origin"])
    p_nominal = vadd(o, vscale(d, dist))            # on-axis entry target
    decenter = vadd(vscale(u, dx), vscale(v, dy))

    R0 = _beam_R(parent_frame, loc)                 # untilted orientation
    # tilt about the beam-frame axes (u=x, v=y, d=z), intrinsic order
    E = euler_matrix3(order, rx, ry, rz)
    W = [[u[i], v[i], d[i]] for i in range(3)]
    T_world = mat3_mul(mat3_mul(W, E), mat3_T(W))

    if pos_rot == "pos_first":
        # decenter along the incoming axes, then tilt about the pivot
        p_entry = vadd(p_nominal, decenter)
        pos0 = vsub(p_entry, mat3_vec(R0, loc["entry"]))
        pl0 = {"pos_mm": pos0, "quat": matrix3_to_quat(R0)}
        pivot_w = transform_point(pl0, _pivot_local(rec, loc))
        M = _rot3_about(T_world, pivot_w)
        return apply_to_placement(M, pl0)
    # rot_first: tilt about the pivot at the nominal (undecentered)
    # entry, then decenter along the TILTED transverse axes
    pos0 = vsub(p_nominal, mat3_vec(R0, loc["entry"]))
    pl0 = {"pos_mm": pos0, "quat": matrix3_to_quat(R0)}
    pivot_w = transform_point(pl0, _pivot_local(rec, loc))
    M = _rot3_about(T_world, pivot_w)
    pl1 = apply_to_placement(M, pl0)
    u2 = mat3_vec(T_world, u)
    v2 = mat3_vec(T_world, v)
    shift = vadd(vscale(u2, dx), vscale(v2, dy))
    return apply_to_placement(translate_matrix(shift), pl1)


def _rot3_about(R3, point):
    M = mat4_identity()
    for i in range(3):
        for j in range(3):
            M[i][j] = R3[i][j]
    Rp = mat3_vec(R3, list(point))
    for i in range(3):
        M[i][3] = point[i] - Rp[i]
    return M


def derive_edge(parent_frame, placement, rec):
    """Inverse of place_chained for the pos_first case: given an
    element's world placement and its parent port frame, recover
    {distance, decenter_x, decenter_y, tilt_rx, tilt_ry, tilt_rz} as
    floats (used to sync chain fields after a spatial drag). Tilt angles
    come back in the record's rot_order."""
    loc = _local(rec)
    u, v, d = frame_basis(parent_frame)
    entry_w = transform_point(placement, loc["entry"])
    rel = vsub(entry_w, list(parent_frame["origin"]))
    out = {
        "distance": vdot(rel, d),
        "decenter_x": vdot(rel, u),
        "decenter_y": vdot(rel, v),
    }
    R0 = _beam_R(parent_frame, loc)
    R = quat_to_matrix3(placement["quat"])
    T_world = mat3_mul(R, mat3_T(R0))
    W = [[u[i], v[i], d[i]] for i in range(3)]
    E = mat3_mul(mat3_mul(mat3_T(W), T_world), W)
    rx, ry, rz = euler_from_matrix3(E, rec.get("rot_order") or "xyz")
    out["tilt_rx"], out["tilt_ry"], out["tilt_rz"] = rx, ry, rz
    return out


# ---------------------------------------------------------------------------
# Port propagation
# ---------------------------------------------------------------------------
def _line_plane(origin, direction, plane_point, plane_normal):
    """Intersection of a line with a plane; falls back to the foot of
    `plane_point` on the line when they are parallel."""
    denom = vdot(direction, plane_normal)
    if abs(denom) < 1e-9:
        t = vdot(vsub(plane_point, origin), direction)
    else:
        t = vdot(vsub(plane_point, origin), plane_normal) / denom
    return vadd(origin, vscale(direction, t))


def exit_frames(rec, placement, incoming_frame, variables=None):
    """Port frames leaving an element, given its placed pose and the
    incoming beam frame. Returns {port_name: frame}.

    Ports:
      "out"/"transmit" — pass-through: dir/up = incoming; origin =
          incoming line intersected with the exit-vertex plane (the
          plane through the world exit vertex, normal = element axis).
      "reflect"        — mirrored about the element's placed reflect
          plane (requires local.reflect_plane); origin = incoming line
          intersected with that plane. Right-handedness restored by
          construction (u re-derived from up x dir).
      "deviate"        — incoming frame rotated by fold_deviation about
          the azimuth-selected transverse axis, at the entry vertex.
    An element with no incoming frame (anchored root, e.g. a source)
    gets a self frame: origin = exit vertex, dir = its own placed axis.

    When the record is a FOLD and rec["folded"] is falsy, "reflect" and
    "deviate" become pass-through frames at the SAME origin — the
    unfolded train continues straight and downstream distances keep
    their meaning.
    """
    loc = _local(rec)
    axis_w = vunit(transform_vector(placement, loc["axis"]), "element axis")
    up_w = transform_vector(placement, loc["up"])
    exit_w = transform_point(placement, loc["exit"])
    entry_w = transform_point(placement, loc["entry"])

    if incoming_frame is None:
        incoming_frame = make_frame(entry_w, axis_w, up_w)
    d_in = list(incoming_frame["dir"])
    up_in = list(incoming_frame["up"])
    o_in = list(incoming_frame["origin"])

    frames = {}
    # pass-through
    out_origin = _line_plane(o_in, d_in, exit_w, axis_w)
    frames["out"] = make_frame(out_origin, d_in, up_in)
    frames["transmit"] = frames["out"]

    unfolded = bool(rec.get("fold")) and not rec.get("folded", True)

    rp = loc.get("reflect_plane")
    if rp:
        pt_w = transform_point(placement, rp["point"])
        n_w = vunit(transform_vector(placement, rp["normal"]),
                    "reflect normal")
        hit = _line_plane(o_in, d_in, pt_w, n_w)
        if unfolded:
            frames["reflect"] = make_frame(hit, d_in, up_in)
        else:
            d_out = reflect_dir(d_in, n_w)
            up_out = reflect_dir(up_in, n_w)
            frames["reflect"] = make_frame(hit, d_out, up_out)

    # a "deviate" port exists for any element with an explicit
    # fold_deviation (non-specular redirects: gratings, prisms) and for
    # plane-less folds. An explicit deviation coexists with (and, for
    # chaining defaults, beats) the specular reflect port — a grating's
    # diffracted beam is NOT the mirror reflection.
    has_dev = rec.get("fold_deviation") not in (None, "")
    if has_dev or (rec.get("fold") and not rp):
        origin = frames["reflect"]["origin"] if rp else entry_w
        dev = _edge_value(rec, "fold_deviation", variables or {}, 90.0)
        az = _edge_value(rec, "fold_azimuth", variables or {}, 0.0)
        if unfolded or abs(dev) < EPS:
            frames["deviate"] = make_frame(origin, d_in, up_in)
        else:
            u, v, d = frame_basis(incoming_frame)
            axis = mat3_vec(axis_angle_matrix3(d, az), u)
            R = axis_angle_matrix3(axis, dev)
            frames["deviate"] = make_frame(
                origin, mat3_vec(R, d), mat3_vec(R, v))
    return frames


def _default_port(rec):
    loc = _local(rec)
    if rec.get("fold_deviation") not in (None, ""):
        return "deviate"
    rp = loc.get("reflect_plane")
    if rec.get("fold"):
        return "reflect" if rp else "deviate"
    if rp and list(loc["entry"]) == list(loc["exit"]):
        # a pure mirror (coincident ports + a reflective surface):
        # chaining "downstream" of it means the reflected beam — the
        # pass-through port of an opaque mirror is physically meaningless
        # as a default (beamsplitters have entry != exit and keep "out")
        return "reflect"
    return "out"


# ---------------------------------------------------------------------------
# Full solve
# ---------------------------------------------------------------------------
def solve_chain(records, anchors, variables):
    """Solve every chained element's placement.

    records:   {label: element record}
    anchors:   {label: {"pos_mm","quat"}} — current placements; REQUIRED
               for anchored records, used as the incoming-frame seed.
               (Chained records' entries are ignored/overwritten.)
    variables: {name: float} — already resolved (see resolve_variables).

    Returns {"placements": {label: placement}   (chained elements, plus
                                                  anchored elements that
                                                  carry a pose expression),
             "frames":     {label: {port: frame}},
             "order":      [labels in solve order]}.
    """
    order = sort_chain(records)
    placements = {}
    incoming = {}
    all_frames = {}
    solved = {}

    for label in order:
        rec = records[label]
        if rec.get("mode") == "chained":
            if has_pose_expr(rec):
                raise TrainError(
                    "element %r is chained but carries anchored-pose "
                    "expression(s) %s; pose expressions are valid on "
                    "anchored elements only (unchain it, or clear the pose "
                    "expressions)"
                    % (label, sorted(k for k in POSE_EXPR_FIELDS
                                     if (rec.get("pose_expr")
                                         or {}).get(k) not in (None, ""))))
            ref = rec["ref"]
            port = rec.get("port") or _default_port(records[ref])
            try:
                parent_frame = all_frames[ref][port]
            except KeyError:
                raise TrainError(
                    "element %r chains to port %r of %r, which has ports: "
                    "%s" % (label, port, ref,
                            ", ".join(sorted(all_frames.get(ref, {})))))
            pl = place_chained(parent_frame, rec, variables)
            placements[label] = pl
            incoming[label] = parent_frame
        else:
            base = anchors.get(label)
            if has_pose_expr(rec):
                # anchored, but its world pose is expression-driven: bake it
                # (this placement seeds the frame AND is flushed like a
                # chained one, so it can be swept). place_anchored keeps
                # unexpressed components from `base` and may be given None
                # only when every pose component is expressed.
                pl = place_anchored(rec, base, variables)
                placements[label] = pl
            elif base is None:
                raise TrainError("anchored element %r has no known "
                                 "placement" % label)
            else:
                pl = base
            incoming[label] = None
        solved[label] = pl
        all_frames[label] = exit_frames(rec, pl, incoming[label], variables)
    return {"placements": placements, "frames": all_frames, "order": order}
