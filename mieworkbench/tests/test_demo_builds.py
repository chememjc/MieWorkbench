"""Phase F: assemble the demos/ gallery's chain topology THROUGH THE
PANES against a scripted worker (no FreeCAD, no real fc_server) -- a
regression net over the exact positions scripts/make_demos.py's ten
demo_* functions encode, doubling as a UX shakedown of the affordances a
real user has for building one of these systems: TrainEditorPane's public
API (pick-reference / commit_field / set_edge_details / toggle_fold) and
VariablesPane.add_variable (the variables dock landed mid-round; this
suite drives its real pane method).

Every demo test:
  1. builds an EMPTY scripted Project (make_empty_scene(), below -- NOT
     train_test_support.make_scene(), which pre-seeds a canned SRC/L1/L2/
     FM/DET scene for the OTHER train test files);
  2. imports each element via project.import_primitive (the scripted
     worker's import_primitive now synthesizes REAL primitivelib port
     geometry -- see the additive change in train_test_support.py -- so
     a lens's ct/R_front etc. actually move its ports, and
     set_element_parameters's rebuild keeps them live);
  3. creates the demo's variables through VariablesPane.add_variable
     (name/value/min/max/nstep, exactly the d.variable rows);
  4. chains elements through TrainEditorPane's public methods wherever
     one exists (chain_element(), below), falling back to a single
     supplemental project.set_chain() call for edge fields the pane
     cannot reach at all (port / flip / fold / fold_deviation /
     fold_azimuth) -- every such fallback is a friction point logged in
     demos/UXNOTES_ROUND2.md;
  5. asserts final primary-body positions against make_demos.py's
     d.expect() values (lifted from scripts/make_demos.py by inspection;
     the pure math helpers -- rot_z/unit/ang/deviate_params/n_glass/
     paraxial_image_x/solve_achromat -- are imported from that module
     directly so this suite can never silently drift from its numbers);
  6. exercises ONE undo at the end (nudge the last element's distance,
     undo, confirm the position is restored).

newtonian additionally toggles the Diagonal fold off (Eyepiece
straightens onto the primary axis) and back on (exact restore);
michelson has no fold element, so no fold-toggle assertion applies there.

Run: QT_QPA_PLATFORM=offscreen env/bin/python -m pytest \
         mieworkbench/tests/test_demo_builds.py -q
"""

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

from mieworkbench.core.geomcache import GeomCache             # noqa: E402
from mieworkbench.core.project import Project                  # noqa: E402
from mieworkbench.core.selection import SelectionModel         # noqa: E402
from mieworkbench.core.transforms import Operation              # noqa: E402
from mieworkbench.panes.train_editor import TrainEditorPane     # noqa: E402
from mieworkbench.panes.variables_pane import VariablesPane     # noqa: E402

from mieworkbench.tests.train_test_support import (              # noqa: E402
    TrainFakeWorker, pos_of,
)

# Pure math only: rot_z/unit/ang/deviate_params/n_glass/paraxial_image_x/
# solve_achromat. NEVER Demo/d.chain/d.variable/d.add -- those wrap the
# same project.set_chain/apply_variable_cells/apply_operation calls this
# suite drives directly through the panes, and reusing them would defeat
# the whole point of the shakedown.
import make_demos as md   # noqa: E402
import primitivelib        # noqa: E402  (metadata only, no FreeCAD)


# ---------------------------------------------------------------------------
# Scene / pane scaffolding
# ---------------------------------------------------------------------------
def make_empty_scene():
    """A scripted-worker Project with NO bodies/sheets -- every element in
    every demo test is assembled from nothing, the way a user opens
    File -> New and starts adding primitives (train_test_support.
    make_scene() pre-seeds a canned SRC/L1/L2/FM/DET scene for the OTHER
    train test files and is not reused here)."""
    structure = {"doc": "scene", "label": "scene",
                "file": "/nowhere/scene.FCStd", "bodies": [], "sheets": []}
    project = Project()
    fake = TrainFakeWorker(structure)
    project._fc = fake
    project._cache = GeomCache(fake, cache_root=tempfile.mkdtemp(
        prefix="miewb_demo_test_"))
    project.doc = "scene"
    project.fcstd_path = "/nowhere/scene.FCStd"
    project.structure = fake.request("get_structure", {"doc": "scene"})
    fake.ops.clear()
    return project, fake


def make_panes(qtbot, project):
    """The two docks a user builds a demo with: the train editor and the
    variables pane (which landed mid-round -- this suite drove
    project.apply_variable_cells + core.variables.cell_plan directly
    until it did; see demos/UXNOTES_ROUND2.md #7)."""
    sel = SelectionModel()
    pane = TrainEditorPane(project, sel)
    qtbot.addWidget(pane)
    vpane = VariablesPane(project)
    qtbot.addWidget(vpane)
    return pane, vpane, sel


def add_variable(vpane, name, value, vmin=None, vmax=None, nstep=None,
                 enabled=False, comment=""):
    """VariablesPane.add_variable, asserting success (the pane reports
    failures through its status label instead of raising)."""
    ok = vpane.add_variable(name, value=value, vmin=vmin, vmax=vmax,
                            nstep=nstep, enabled=enabled, comment=comment)
    assert ok is True, vpane.status.text()
    return name


def import_element(project, kind, label, params=None):
    """project.import_primitive + (if params given) set_element_parameters
    -- the same two-call sequence scripts/make_demos.py's Demo._import
    issues to bake a demo's chosen dims (ct, R_front, aperture, ...) into
    the element before it's chained. There is no 'insert primitive' pane
    in this round to wrap this further."""
    project.import_primitive(str(kind) + ".FCStd", label)
    if params:
        spec = primitivelib.PRIMITIVES[kind]["params"]
        values = {}
        for alias, value in params.items():
            if isinstance(value, str):
                values[alias] = "=%s" % value
            else:
                values[alias] = primitivelib.sheet_raw(
                    float(value), spec[alias]["unit"])
        project.set_element_parameters("dim_%s" % label, values,
                                       rebuild_group=label)
    return label


def anchor_at(project, label, pos, rot_deg=0.0, quat=None):
    """Place an anchored (unchained) element at an absolute world pose --
    every demo's source. The train editor has NO affordance for this at
    all (by design: it's the Position/Orientation Absolute panel's job,
    docs/UI_TESTING.md manual-checklist item #12) -- but there's also no
    cross-link from the train editor to that other panel for a freshly
    imported element (UXNOTES_ROUND2.md #8), so this goes straight
    through project.apply_operation, exactly like scripts/make_demos.py's
    Demo.add()."""
    q = list(quat) if quat is not None else md.rot_z(rot_deg)
    if list(pos) != [0.0, 0.0, 0.0] or q != [0.0, 0.0, 0.0, 1.0]:
        body = project.body(label)["name"]
        project.apply_operation(body, Operation(
            "set_placement", {"pos_mm": list(pos), "quat": q}))


def _fold_tilt(deviation_deg):
    """Same tilt_ry formula scripts/make_demos.py's Demo.fold_mirror uses
    for a deviation/azimuth-specified fold mirror: tilt_ry =
    -(180 - deviation) / 2, rot_order 'zyx', tilt_rz = azimuth."""
    return -(180.0 - float(deviation_deg)) / 2.0


def chain_element(pane, project, label, ref, distance, port=None,
                  decenter_x=None, decenter_y=None,
                  tilt_rx=None, tilt_ry=None, tilt_rz=None,
                  rot_order=None, pos_rot_order="pos_first",
                  pivot="entrance", raw_edge=None):
    """Chain `label` onto `ref` through the TrainEditorPane's OWN public
    methods wherever one exists:
      * reference: 'Pick reference in 3D' (begin_pick_reference /
        on_reference_picked) -- deterministic, unlike set_mode(
        "chained")'s ref-inference (previously stored ref, else 'the
        element directly above it in the tree'), which only works by
        luck once a scene has more than one branch or an alphabetically-
        surprising import order (UXNOTES_ROUND2.md #1);
      * distance / decenter_x / decenter_y / tilt_rx / tilt_ry / tilt_rz:
        commit_field, one tree cell at a time (make_demos.py's d.chain
        sets every edge field in ONE call -- UXNOTES_ROUND2.md #2);
      * rot_order / pos_rot_order / pivot: the Edge details dialog's
        dialog-free core, set_edge_details -- also all-or-nothing
        (UXNOTES_ROUND2.md #6).
    then a SINGLE supplemental project.set_chain call for whatever the
    pane cannot reach at all: port (#3), flip (#4), fold/folded (#5),
    fold_deviation/fold_azimuth (#5)."""
    ref_body = project.body(ref)["name"]
    pane.begin_pick_reference(label)
    pane.on_reference_picked(ref_body)
    assert project.train().records()[label].get("ref") == ref, \
        pane.status.text()
    assert pane.commit_field(label, "distance", str(distance)) is True, \
        pane.status.text()
    for field, value in (("decenter_x", decenter_x),
                        ("decenter_y", decenter_y),
                        ("tilt_rx", tilt_rx),
                        ("tilt_ry", tilt_ry),
                        ("tilt_rz", tilt_rz)):
        if value is not None:
            assert pane.commit_field(label, field, str(value)) is True, \
                pane.status.text()
    if rot_order is not None:
        assert pane.set_edge_details(label, rot_order, pos_rot_order,
                                     pivot) is True, pane.status.text()
    edge = dict(raw_edge or {})
    if port is not None:
        edge["port"] = port
    if edge:
        project.set_chain(label, edge, text="Edge fields of %s" % label)


def assert_pos(project, label, expected, tol=1e-6):
    body = project.train().primary_body_name(label)
    cur = pos_of(project, body)
    err = max(abs(c - e) for c, e in zip(cur, expected))
    assert err <= tol, "%s: expected %s, got %s (err %.3g mm)" % (
        label, list(expected), cur, err)


def exercise_one_undo(pane, project, label):
    """Step 6 of the Phase F spec: one undo at the end, on the last chain
    edit, restoring the moved element -- nudge `label`'s distance by a
    fixed offset, confirm it moved, undo, confirm it's back."""
    body = project.train().primary_body_name(label)
    before = pos_of(project, body)
    rec = project.train().records()[label]
    cur = rec.get("distance") or "0"
    assert pane.commit_field(label, "distance", "(%s) + 5" % cur) is True, \
        pane.status.text()
    nudged = pos_of(project, body)
    assert not np.allclose(nudged, before, atol=1e-9)
    project.undo()
    restored = pos_of(project, body)
    assert np.allclose(restored, before, atol=1e-6)


# ---------------------------------------------------------------------------
# beam_expander
# ---------------------------------------------------------------------------
def test_beam_expander_builds_through_panes(qtbot):
    lam = 650.0
    n = md.n_glass("bk7", lam)
    f1, f2 = 50.0, 150.0
    r1, r2 = (n - 1.0) * f1, (n - 1.0) * f2
    ct1, ct2 = 3.0, 5.0

    project, _fake = make_empty_scene()
    pane, vpane, _sel = make_panes(qtbot, project)

    add_variable(vpane, "sep", f1 + f2, 150.0, 300.0, 6,
                comment="lens separation, outer vertex to outer vertex, mm")
    add_variable(vpane, "screen_dist", 40.0, 10.0, 150.0, 6,
                comment="L2 outer vertex to screen, mm")

    import_element(project, "laser_collimated", "Laser",
                  params={"diameter": 3.0})
    anchor_at(project, "Laser", (-30, 0, 0))

    import_element(project, "lens_pcx", "L1",
                  params={"R_front": r1, "aperture": 12.0, "ct": ct1})
    chain_element(pane, project, "L1", "Laser", 30.0)

    import_element(project, "lens_pcx", "L2",
                  params={"R_front": r2, "aperture": 30.0, "ct": ct2})
    chain_element(pane, project, "L2", "L1", "sep - %.10g" % (ct1 + ct2),
                 raw_edge={"flip": True})

    import_element(project, "detector_plane", "Screen",
                  params={"width": 40.0})
    chain_element(pane, project, "Screen", "L2", "screen_dist")

    assert_pos(project, "L1", (0, 0, 0))
    assert_pos(project, "L2", (f1 + f2, 0, 0))
    assert_pos(project, "Screen", (f1 + f2 + 40.0, 0, 0))

    exercise_one_undo(pane, project, "Screen")


# ---------------------------------------------------------------------------
# newtonian / dobsonian (shared topology)
# ---------------------------------------------------------------------------
def _build_newtonian_like(qtbot, rfl, ap, L, mirror_t, diag_w, eye_w):
    project, _fake = make_empty_scene()
    pane, vpane, _sel = make_panes(qtbot, project)

    add_variable(vpane, "eye_dist", L, 150.0, 300.0, 5,
                comment="diagonal center to eyepiece plane, mm")

    import_element(project, "laser_collimated", "Star",
                  params={"diameter": ap * 0.98})
    anchor_at(project, "Star", (-rfl - 60.0, 0, 0))

    import_element(project, "mirror_parabolic", "Primary",
                  params={"rfl": rfl, "aperture": ap, "thickness": mirror_t})
    chain_element(pane, project, "Primary", "Star", rfl + 60.0)

    import_element(project, "mirror_flat", "Diagonal",
                  params={"width": diag_w, "thickness": 5.0,
                          "round_flag": 1})
    chain_element(pane, project, "Diagonal", "Primary",
                 "%.10g - eye_dist" % rfl,
                 tilt_rz=180.0, tilt_ry=_fold_tilt(90.0), rot_order="zyx",
                 raw_edge={"fold": True, "folded": True})

    import_element(project, "detector_plane", "Eyepiece",
                  params={"width": eye_w})
    chain_element(pane, project, "Eyepiece", "Diagonal", "eye_dist")

    xd = -(rfl - L)
    assert_pos(project, "Primary", (0, 0, 0))
    assert_pos(project, "Diagonal", (xd, 0, 0))
    assert_pos(project, "Eyepiece", (xd, L, 0))
    return project, pane


def test_newtonian_builds_through_panes(qtbot):
    project, pane = _build_newtonian_like(
        qtbot, rfl=900.0, ap=150.0, L=210.0, mirror_t=15.0, diag_w=52.0,
        eye_w=20.0)
    exercise_one_undo(pane, project, "Eyepiece")

    # newtonian-specific: unfold the Diagonal -> Eyepiece re-collinearizes
    # onto the straight-through primary axis; refold -> exact restore.
    before = pos_of(project, project.train().primary_body_name("Eyepiece"))
    assert pane.toggle_fold("Diagonal", False) is True
    straightened = pos_of(
        project, project.train().primary_body_name("Eyepiece"))
    assert not np.allclose(straightened, before, atol=1e-6)
    assert abs(straightened[1]) < 1e-6      # off the +y arm, back on-axis

    assert pane.toggle_fold("Diagonal", True) is True
    restored = pos_of(project, project.train().primary_body_name("Eyepiece"))
    assert np.allclose(restored, before, atol=1e-9)


def test_dobsonian_builds_through_panes(qtbot):
    project, pane = _build_newtonian_like(
        qtbot, rfl=1000.0, ap=200.0, L=250.0, mirror_t=18.0, diag_w=72.0,
        eye_w=25.0)
    exercise_one_undo(pane, project, "Eyepiece")


# ---------------------------------------------------------------------------
# michelson
# ---------------------------------------------------------------------------
def test_michelson_builds_through_panes(qtbot):
    tilt_deg = math.degrees(633e-9 / (2.0 * 0.002))
    bs_exit = 3.0 * math.sqrt(2.0)

    project, _fake = make_empty_scene()
    pane, vpane, _sel = make_panes(qtbot, project)

    add_variable(vpane, "arm1", 60.0, 40.0, 100.0, 6,
                comment="BS center to M1 (transmit arm), mm")
    add_variable(vpane, "arm2", 60.0, 40.0, 100.0, 6,
                comment="BS center to M2 (reflect arm), mm")
    add_variable(vpane, "m1_tilt", "%.10g" % tilt_deg, 0.0, 0.05, 10,
                comment="M1 fringe tilt, deg (~5 fringes across 10mm)")
    add_variable(vpane, "screen_arm", 60.0,
                comment="BS center to screen (output arm), mm")

    import_element(project, "laser_collimated", "Laser",
                  params={"diameter": 8.0})
    anchor_at(project, "Laser", (-60, 0, 0))

    import_element(project, "bs_plate", "BS",
                  params={"width": 30.0, "thickness": 3.0, "round_flag": 1,
                          "wedge_deg": 0.0})
    chain_element(pane, project, "BS", "Laser", 60.0, tilt_ry=45.0)

    import_element(project, "mirror_flat", "M1",
                  params={"width": 15.0, "round_flag": 1})
    chain_element(pane, project, "M1", "BS", "arm1 - %.10g" % bs_exit,
                 port="transmit", tilt_ry="m1_tilt")

    import_element(project, "mirror_flat", "M2",
                  params={"width": 15.0, "round_flag": 1})
    chain_element(pane, project, "M2", "BS", "arm2", port="reflect")

    import_element(project, "detector_plane", "Screen",
                  params={"width": 12.0, "round_flag": 0})
    chain_element(pane, project, "Screen", "M2", "arm2 + screen_arm",
                 port="reflect")

    assert_pos(project, "BS", (0, 0, 0))
    assert_pos(project, "M1", (60, 0, 0))
    assert_pos(project, "M2", (0, -60, 0))
    assert_pos(project, "Screen", (0, 60, 0))

    exercise_one_undo(pane, project, "Screen")


# ---------------------------------------------------------------------------
# prism_spectrometer
# ---------------------------------------------------------------------------
def test_prism_spectrometer_builds_through_panes(qtbot):
    glass = "sf5"
    A = 60.0
    n550 = md.n_glass(glass, 550.0)
    dmin = 2.0 * math.degrees(math.asin(
        n550 * math.sin(math.radians(A / 2.0)))) - A
    inc = (A + dmin) / 2.0
    rotation = 30.0 - inc
    dev_dir = md.unit(-dmin)
    y0 = 6.0
    Rc = 25.0 / math.sqrt(3.0)
    verts = [(Rc * math.cos(math.radians(a + rotation)),
              Rc * math.sin(math.radians(a + rotation)))
             for a in (90.0, 210.0, 330.0)]

    def _outward(pf, qf):
        ex, ey = qf[0] - pf[0], qf[1] - pf[1]
        nx, ny = ey, -ex
        mx = (pf[0] + qf[0]) / 2.0 - (
            verts[0][0] + verts[1][0] + verts[2][0]) / 3.0
        my = (pf[1] + qf[1]) / 2.0 - (
            verts[0][1] + verts[1][1] + verts[2][1]) / 3.0
        if nx * mx + ny * my < 0:
            nx, ny = -nx, -ny
        norm = math.hypot(nx, ny)
        return nx / norm, ny / norm

    def _refract(d_in, nrm, n1, n2):
        dx, dy = d_in
        nx, ny = nrm
        if dx * nx + dy * ny > 0:
            nx, ny = -nx, -ny
        cosi = -(dx * nx + dy * ny)
        eta = n1 / n2
        s2 = eta * eta * (1.0 - cosi * cosi)
        f = eta * cosi - math.sqrt(1.0 - s2)
        return (eta * dx + f * nx, eta * dy + f * ny)

    def _hit(p, dvec, a_pt, b_pt):
        ex, ey = b_pt[0] - a_pt[0], b_pt[1] - a_pt[1]
        det = dvec[0] * (-ey) - dvec[1] * (-ex)
        t = ((a_pt[0] - p[0]) * (-ey) - (a_pt[1] - p[1]) * (-ex)) / det
        return (p[0] + t * dvec[0], p[1] + t * dvec[1])

    n_in = _outward(verts[0], verts[1])
    p_ent = _hit((-60.0, y0), (1.0, 0.0), verts[0], verts[1])
    d_in = _refract((1.0, 0.0), n_in, 1.0, n550)
    p_exit = _hit(p_ent, d_in, verts[2], verts[0])
    lens_c = (p_exit[0] + 40.0 * dev_dir[0], p_exit[1] + 40.0 * dev_dir[1])
    det_c = (p_exit[0] + 141.0 * dev_dir[0], p_exit[1] + 141.0 * dev_dir[1])

    project, _fake = make_empty_scene()
    pane, vpane, _sel = make_panes(qtbot, project)

    add_variable(vpane, "cam_dist", 40.0, 20.0, 80.0, 6,
                comment="prism exit point to camera lens, mm")
    add_variable(vpane, "det_dist", 97.0, 50.0, 150.0, 6,
                comment="camera back vertex to detector, mm")

    import_element(project, "source_broadband", "Source",
                  params={"diameter": 8.0})
    anchor_at(project, "Source", (-60, y0, 0))

    import_element(project, "prism", "Prism",
                  params={"side": 25.0, "height": 25.0, "rotation": rotation})
    chain_element(pane, project, "Prism", "Source", 60.0, decenter_x=-y0,
                 raw_edge={"fold_deviation": "%.10g" % (-dmin),
                          "fold_azimuth": 90.0})

    u_dev = (-dev_dir[1], dev_dir[0])
    delta = (lens_c[0] - 0.0, lens_c[1] - 0.0)
    along = delta[0] * dev_dir[0] + delta[1] * dev_dir[1]
    across = delta[0] * u_dev[0] + delta[1] * u_dev[1]
    n_lens = md.n_glass("bk7", 550.0)

    import_element(project, "lens_pcx", "Camera",
                  params={"R_front": (n_lens - 1.0) * 100.0,
                          "aperture": 22.0, "ct": 4.0})
    chain_element(pane, project, "Camera", "Prism",
                 "%.10g + cam_dist" % (along - 40.0),
                 decenter_x="%.10g" % across)

    import_element(project, "detector_plane", "Screen",
                  params={"width": 25.0})
    chain_element(pane, project, "Screen", "Camera", "det_dist",
                 decenter_x="%.10g" % across)

    assert_pos(project, "Prism", (0, 0, 0))
    assert_pos(project, "Camera", (lens_c[0], lens_c[1], 0))
    assert_pos(project, "Screen", (det_c[0], det_c[1], 0))

    exercise_one_undo(pane, project, "Screen")


# ---------------------------------------------------------------------------
# czerny_turner
# ---------------------------------------------------------------------------
def test_czerny_turner_builds_through_panes(qtbot):
    theta_i, theta_d = 6.127, 25.896
    off = 34.0
    u = md.unit(180.0 - theta_i)
    v = md.unit(180.0 + theta_d)
    C = (100.0 * u[0], 100.0 * u[1])
    M2 = (100.0 * v[0], 100.0 * v[1])
    w = md.unit(md.ang(u) + off)
    S = (C[0] + 100.0 * w[0], C[1] + 100.0 * w[1])
    m = md.unit(md.ang((-v[0], -v[1])) - off)
    D = (M2[0] + 100.0 * m[0], M2[1] + 100.0 * m[1])

    project, _fake = make_empty_scene()
    pane, vpane, _sel = make_panes(qtbot, project)

    add_variable(vpane, "arm_coll", 100.0, 80.0, 150.0, 5,
                comment="slit to collimator mirror, mm")
    add_variable(vpane, "arm_cam", 100.0, 80.0, 150.0, 5,
                comment="grating to camera mirror, mm")
    add_variable(vpane, "det_arm", 100.0, 80.0, 150.0, 5,
                comment="camera mirror to detector, mm")

    import_element(project, "laser_divergent", "SlitSource",
                  params={"diameter": 2.0, "roc": 8.0, "length": 8.0})
    anchor_at(project, "SlitSource",
             (S[0] + 8.0 * w[0], S[1] + 8.0 * w[1], 0),
             rot_deg=md.ang((-w[0], -w[1])))

    import_element(project, "slit", "Slit",
                  params={"width": 20.0, "height": 20.0, "slit_width": 0.3,
                          "slit_height": 8.0})
    chain_element(pane, project, "Slit", "SlitSource", 8.0)

    dev_c, _az_c_unused, az_c = md.deviate_params(
        (-w[0], -w[1]), (-u[0], -u[1]))
    import_element(project, "mirror_concave", "Collimator",
                  params={"R": 200.0, "aperture": 40.0, "ct": 6.0})
    chain_element(pane, project, "Collimator", "Slit", "arm_coll",
                 tilt_rz=az_c, tilt_ry=_fold_tilt(dev_c), rot_order="zyx",
                 raw_edge={"fold": True, "folded": True})

    dev_g, gaz_g, _mir_g_unused = md.deviate_params(
        (-u[0], -u[1]), (v[0], v[1]))
    import_element(project, "grating_plate", "Grating")
    chain_element(pane, project, "Grating", "Collimator", "arm_coll",
                 tilt_ry="%.10g" % (-md.ang((-u[0], -u[1]))),
                 raw_edge={"fold": True, "folded": True,
                          "fold_deviation": "%.10g" % dev_g,
                          "fold_azimuth": gaz_g})

    dev_m, _az_m_unused, az_m = md.deviate_params((v[0], v[1]), (m[0], m[1]))
    import_element(project, "mirror_concave", "CameraMirror",
                  params={"R": 200.0, "aperture": 40.0, "ct": 6.0})
    chain_element(pane, project, "CameraMirror", "Grating", "arm_cam",
                 tilt_rz=az_m, tilt_ry=_fold_tilt(dev_m), rot_order="zyx",
                 raw_edge={"fold": True, "folded": True})

    import_element(project, "detector_plane", "Screen",
                  params={"width": 30.0})
    chain_element(pane, project, "Screen", "CameraMirror", "det_arm")

    assert_pos(project, "Slit", (S[0], S[1], 0))
    assert_pos(project, "Collimator", (C[0], C[1], 0))
    assert_pos(project, "Grating", (0, 0, 0))
    assert_pos(project, "CameraMirror", (M2[0], M2[1], 0))
    assert_pos(project, "Screen", (D[0], D[1], 0))

    exercise_one_undo(pane, project, "Screen")


# ---------------------------------------------------------------------------
# camera_triplet
# ---------------------------------------------------------------------------
def test_camera_triplet_builds_through_panes(qtbot):
    lam = 550.0
    L1 = {"R_front": 20.115, "R_back": 269.375, "ct": 3.010, "ap": 15.0}
    L2 = {"R_front": 23.577, "R_back": 20.065, "ct": 0.502, "ap": 10.0}
    L3 = {"R_front": 117.632, "R_back": 19.012, "ct": 2.960, "ap": 12.5}
    air12, air23 = 5.016, 5.418
    x1 = 0.0
    x2 = x1 + L1["ct"] + air12
    x3 = x2 + L2["ct"] + air23
    surfaces = [
        (x1, 20.115, None, "bk7"), (x1 + L1["ct"], -269.375, "bk7", None),
        (x2, -23.577, None, "sf5"), (x2 + L2["ct"], 20.065, "sf5", None),
        (x3, 117.632, None, "bk7"), (x3 + L3["ct"], -19.012, "bk7", None),
    ]
    x_img = md.paraxial_image_x(surfaces, float("-inf"), lam)

    project, _fake = make_empty_scene()
    pane, vpane, _sel = make_panes(qtbot, project)

    add_variable(vpane, "air12", air12, 3.0, 8.0, 5,
                comment="L1 back vertex to L2 front vertex, mm")
    add_variable(vpane, "air23", air23, 3.0, 8.0, 5,
                comment="L2 back vertex to L3 front vertex, mm")
    add_variable(vpane, "stop_d", 6.94, 3.0, 12.0, 6,
                comment="iris opening diameter, mm (~f/5.6 at 6.94)")

    import_element(project, "laser_collimated", "Scene",
                  params={"diameter": 14.0})
    anchor_at(project, "Scene", (-40, 0, 0))

    import_element(project, "lens_dcx", "L1",
                  params={"R_front": L1["R_front"], "R_back": L1["R_back"],
                          "ct": L1["ct"], "aperture": L1["ap"]})
    chain_element(pane, project, "L1", "Scene", 40.0)

    import_element(project, "lens_dcv", "L2",
                  params={"R_front": L2["R_front"], "R_back": L2["R_back"],
                          "ct": L2["ct"], "aperture": L2["ap"]})
    chain_element(pane, project, "L2", "L1", "air12")

    import_element(project, "iris", "Stop",
                  params={"outer_diameter": 18.0, "thickness": 0.4,
                          "hole_diameter": "<<miewb_vars>>.stop_d * 1mm"})
    chain_element(pane, project, "Stop", "L2", 0.95)

    import_element(project, "lens_dcx", "L3",
                  params={"R_front": L3["R_front"], "R_back": L3["R_back"],
                          "ct": L3["ct"], "aperture": L3["ap"]})
    chain_element(pane, project, "L3", "Stop", "air23 - 0.95")

    import_element(project, "detector_plane", "Sensor",
                  params={"width": 36.0, "height": 24.0, "round_flag": 0})
    chain_element(pane, project, "Sensor", "L3",
                 "%.10g" % (x_img - (x3 + L3["ct"])))

    assert_pos(project, "L1", (x1, 0, 0))
    assert_pos(project, "L2", (x2, 0, 0))
    assert_pos(project, "Stop", (x2 + L2["ct"] + 0.95, 0, 0))
    assert_pos(project, "L3", (x3, 0, 0))
    assert_pos(project, "Sensor", (x_img, 0, 0))

    exercise_one_undo(pane, project, "Sensor")


# ---------------------------------------------------------------------------
# microscope_objective
# ---------------------------------------------------------------------------
def test_microscope_objective_builds_through_panes(qtbot):
    lam = 550.0
    a1 = md.solve_achromat(25.0)
    a2 = md.solve_achromat(50.0)
    x1, x2 = 0.0, 10.0
    x_obj = -22.0

    def ach_surfaces(x0, a):
        return [
            (x0, a["R_front"], None, "bk7"),
            (x0 + a["ct_crown"], a["R_iface"], "bk7", "sf5"),
            (x0 + a["ct_crown"] + a["ct_flint"], a["R_back"], "sf5", None),
        ]
    surfaces = ach_surfaces(x1, a1) + ach_surfaces(x2, a2)
    x_img = md.paraxial_image_x(surfaces, x_obj, lam)

    def ach_params(a, aperture):
        out = {k: a[k] for k in ("R_front", "R_iface", "R_back",
                                 "ct_crown", "ct_flint")}
        out["aperture"] = aperture
        return out

    exit1 = primitivelib.port_frames(
        "lens_achromat", ach_params(a1, 10.0))["exit"][0]
    exit2 = primitivelib.port_frames(
        "lens_achromat", ach_params(a2, 12.0))["exit"][0]

    project, _fake = make_empty_scene()
    pane, vpane, _sel = make_panes(qtbot, project)

    add_variable(vpane, "obj_dist", -x_obj, 15.0, 35.0, 5,
                comment="object plane to front achromat vertex, mm")
    add_variable(vpane, "ach_gap", x2 - x1, 5.0, 20.0, 5,
                comment="achromat front-vertex spacing, mm")

    import_element(project, "laser_divergent", "Object",
                  params={"diameter": 2.0, "roc": 5.0, "length": 6.0})
    anchor_at(project, "Object", (x_obj, 0, 0))

    import_element(project, "lens_achromat", "Front",
                  params=ach_params(a1, 10.0))
    chain_element(pane, project, "Front", "Object", "obj_dist")

    import_element(project, "lens_achromat", "Rear",
                  params=ach_params(a2, 12.0))
    chain_element(pane, project, "Rear", "Front",
                 "ach_gap - %.10g" % exit1)

    import_element(project, "detector_plane", "Image", params={"width": 30.0})
    chain_element(pane, project, "Image", "Rear",
                 "%.10g" % (x_img - (x2 + exit2)))

    assert_pos(project, "Front", (x1, 0, 0))
    assert_pos(project, "Rear", (x2, 0, 0))
    assert_pos(project, "Image", (x_img, 0, 0))

    exercise_one_undo(pane, project, "Image")


# ---------------------------------------------------------------------------
# fiber_coupler
# ---------------------------------------------------------------------------
def test_fiber_coupler_builds_through_panes(qtbot):
    lam = 650.0
    n = md.n_glass("bk7", lam)
    R = 1.0
    bfl = R * (2.0 - n) / (2.0 * (n - 1.0))
    x_fiber = 2.0 * R + bfl

    project, _fake = make_empty_scene()
    pane, vpane, _sel = make_panes(qtbot, project)

    add_variable(vpane, "exit_gap", 0.6, 0.1, 3.0, 6,
                comment="fiber exit face to detector, mm")

    import_element(project, "laser_collimated", "Laser",
                  params={"diameter": 0.6, "length": 5.0})
    anchor_at(project, "Laser", (-6, 0, 0))

    import_element(project, "lens_ball", "Ball", params={"diameter": 2.0 * R})
    chain_element(pane, project, "Ball", "Laser", 6.0)

    import_element(project, "fiber_optic", "Fiber", params={"length": 75.0})
    chain_element(pane, project, "Fiber", "Ball", "%.10g" % bfl)

    import_element(project, "detector_plane", "Exit",
                  params={"width": 2.0, "round_flag": 0})
    chain_element(pane, project, "Exit", "Fiber", "exit_gap")

    assert_pos(project, "Ball", (0, 0, 0))
    assert_pos(project, "Fiber", (x_fiber, 0, 0))
    assert_pos(project, "Exit", (x_fiber + 75.0 + 0.6, 0, 0))

    exercise_one_undo(pane, project, "Exit")


# ---------------------------------------------------------------------------
# schmidt_cassegrain
# ---------------------------------------------------------------------------
def test_schmidt_cassegrain_builds_through_panes(qtbot):
    # the hand-authored quartic corrector plate (added post-assembly under
    # the FreeCAD AppImage by scripts/tools/add_schmidt_corrector.py) is
    # outside the fcclient op path / Project scope this suite exercises;
    # its 5mm + 320mm offset is already baked into x_primary below,
    # matching scripts/make_demos.py's demo_schmidt_cassegrain exactly.
    x_primary = 5.0 + 320.0
    x_secondary = x_primary - 312.62
    x_focus = x_primary + 150.0

    project, _fake = make_empty_scene()
    pane, vpane, _sel = make_panes(qtbot, project)

    add_variable(vpane, "sct_sep", 312.62, 280.0, 340.0, 6,
                comment="primary vertex to secondary vertex, mm")
    add_variable(vpane, "back_focus", 150.0, 100.0, 200.0, 5,
                comment="primary vertex plane to focal plane, mm")

    import_element(project, "laser_collimated", "Star",
                  params={"diameter": 198.0})
    anchor_at(project, "Star", (-60, 0, 0))

    import_element(project, "mirror_annular", "Primary",
                  params={"R": 812.8, "aperture": 203.2,
                          "hole_diameter": 60.0, "ct": 18.0})
    chain_element(pane, project, "Primary", "Star", x_primary + 60.0)

    import_element(project, "mirror_convex", "Secondary",
                  params={"R": 231.07, "aperture": 66.0, "ct": 6.0})
    chain_element(pane, project, "Secondary", "Primary", "sct_sep",
                 port="reflect")

    import_element(project, "detector_plane", "Focus", params={"width": 15.0})
    chain_element(pane, project, "Focus", "Secondary",
                 "sct_sep + back_focus", port="reflect")

    assert_pos(project, "Primary", (x_primary, 0, 0))
    assert_pos(project, "Secondary", (x_secondary, 0, 0))
    assert_pos(project, "Focus", (x_focus, 0, 0))

    exercise_one_undo(pane, project, "Focus")
