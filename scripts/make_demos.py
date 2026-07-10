#!/usr/bin/env python3
# =============================================================================
# make_demos.py — build the demos/ gallery THROUGH THE GUI'S OWN OP PATH.
#
# Interpreter: the GUI venv (env/bin/python) — each scene is assembled by
# a real mieworkbench.core.project.Project session over the persistent
# fc_server worker, positioned through the OPTICAL-TRAIN chain API
# (set_chain / fold_mirror / global variables in the miewb_vars sheet)
# exactly as interactive editing would. Every demo self-checks its built
# placements (Demo.expect) and validates its train before save; the
# committed demos/baselines/ + scripts/run_demo_equivalence.py gate the
# rebuild against the pre-train gallery. The UX shakedowns live in
# demos/UXNOTES.md (absolute-pose era) and demos/UXNOTES_ROUND2.md
# (train era).
#
#   env/bin/python scripts/make_demos.py [--demo NAME|all] [--outdir demos]
#                                        [--no-pack] [--list]
#
# Each demo produces demos/<name>.FCStd plus (default) a packed
# demos/<name>.MieWB embedding the property library and a quick-preset
# simparams.json (with per-demo overrides such as max_reflections).
# Smoke-run any of them with:
#   python3 scripts/miewb_tool.py run demos/<name>.MieWB -o /tmp/<name>.MieSim
#
# The schmidt_cassegrain corrector plate is the one hand-authored body
# (quartic Schmidt profile — the catalog asphere is conic-only): after the
# fcclient assembly, scripts/tools/add_schmidt_corrector.py is run under
# the FreeCAD AppImage to add it to the saved .FCStd.
#
# Prescription sources (full citations in demos/README.md): Cooke triplet
# (MathWorks Cooke design, uniformly rescaled to 50 mm EFL), C8-style SCT
# (Suiter's ZEMAX table + the classic 0.866-zone corrector), N-BK7/F2/SF
# Sellmeier data (Schott catalog / refractiveindex.info), ball-lens BFL
# (standard formula, Edmund Optics app notes), fiber core (Fleming 1984 /
# Malitson 1965 interpolation, see opticalproperties/materials.miemat).
# =============================================================================
import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import common  # noqa: E402
import miewb_tool  # noqa: E402
from mieworkbench.core.fcclient import FcClient  # noqa: E402
from mieworkbench.core.wizards import solve_achromat  # noqa: E402
import primitivelib  # noqa: E402  (metadata only, no FreeCAD)

PRIMDIR = REPO / "primitives"
FREECAD = common.FREECAD_APPIMAGE


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------
def rot_z(deg):
    """Quaternion [x,y,z,w] for a rotation about the world z axis."""
    half = math.radians(deg) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _expr(value):
    """Chain-field value -> stored string (numbers canonicalized,
    variable expressions verbatim)."""
    if isinstance(value, str):
        return value
    return "%.17g" % float(value)


def unit(deg):
    """Unit vector in the x-y plane at `deg` from +x."""
    return (math.cos(math.radians(deg)), math.sin(math.radians(deg)))


def ang(vec):
    return math.degrees(math.atan2(vec[1], vec[0]))


def deviate_params(d_in, d_out):
    """(deviation_deg, deviate_azimuth_deg, mirror_azimuth_deg) for a
    planar beam turn from direction `d_in` (x-y tuple) into `d_out`.

    Two azimuth conventions coexist in the chain model (both validated
    by the solver tests):
      * deviate ports rotate the frame about the incoming u axis spun by
        `fold_azimuth` about the beam — in-plane turns need axis +/-z,
        i.e. deviate azimuth +/-90 (sign from the turn handedness);
      * fold MIRRORS take azimuth = the transverse direction the beam
        folds TOWARD (0 = +u, 90 = +v, 180 = -u), from which
        Demo.fold_mirror derives the plate tilt.
    """
    di = (d_in[0], d_in[1])
    do = (d_out[0], d_out[1])
    ni = math.hypot(*di)
    no = math.hypot(*do)
    di = (di[0] / ni, di[1] / ni)
    do = (do[0] / no, do[1] / no)
    dot = max(-1.0, min(1.0, di[0] * do[0] + di[1] * do[1]))
    deviation = math.degrees(math.acos(dot))
    cross_z = di[0] * do[1] - di[1] * do[0]
    # for planar scenes (up = +z): u = z x d = left of travel; a
    # counterclockwise turn (+cross) folds toward +u
    deviate_az = 90.0 if cross_z >= 0.0 else -90.0
    mirror_az = 0.0 if cross_z >= 0.0 else 180.0
    return deviation, deviate_az, mirror_az


def _sellmeier_cache():
    rows = {}
    with open(REPO / "opticalproperties" / "materials.miemat",
              newline="") as fh:
        for row in csv.DictReader(fh):
            if row["model"] == "sellmeier":
                rows[row["name"]] = [float(row["p%d" % i])
                                     for i in range(1, 7)]
    return rows


_SELLMEIER = None


def n_glass(name, lam_nm):
    """Refractive index from the shipped materials registry (Sellmeier)."""
    global _SELLMEIER
    if _SELLMEIER is None:
        _SELLMEIER = _sellmeier_cache()
    b1, b2, b3, c1, c2, c3 = _SELLMEIER[name]
    l2 = (lam_nm * 1e-3) ** 2
    return math.sqrt(1.0 + b1 * l2 / (l2 - c1) + b2 * l2 / (l2 - c2)
                     + b3 * l2 / (l2 - c3))


def paraxial_image_x(surfaces, x_obj, lam_nm):
    """Paraxial image plane x for an on-axis object point at x_obj.

    surfaces: [(x_vertex_mm, R_mm_or_None_for_flat, glass_before_or_None,
    glass_after_or_None), ...] in beam order (+x); None glass = air.
    Sign convention: R > 0 means center of curvature at +x of the vertex.
    """
    def idx(glass):
        return 1.0 if glass is None else n_glass(glass, lam_nm)

    if x_obj is None or not math.isfinite(x_obj):
        y, u, x = 1.0, 0.0, surfaces[0][0]     # collimated bundle
    else:
        y, u, x = 0.0, 0.05, x_obj             # axial object point
    for xv, R, g1, g2 in surfaces:
        y += u * (xv - x)
        x = xv
        n1, n2 = idx(g1), idx(g2)
        if R is not None:
            u = (n1 * u - y * (n2 - n1) / R) / n2
        else:
            u = n1 * u / n2
    if abs(u) < 1e-12:
        raise ValueError("paraxial bundle did not converge")
    return x - y / u


# ---------------------------------------------------------------------------
# Demo assembly wrapper around the GUI op set
# ---------------------------------------------------------------------------
class Demo:
    """Assembles a demo scene through the SAME session object the GUI
    uses (mieworkbench.core.project.Project over the persistent worker):
    import_primitive -> set_element_parameters -> set_chain/set_property
    -> save. Elements are positioned through the OPTICAL-TRAIN chain API
    wherever a beam relationship exists (`chain`/`fold_mirror`), with
    `add` kept for anchored elements (and for make_library_tests.py,
    which reuses this class unchanged)."""

    def __init__(self, fc, fcstd_path):
        from mieworkbench.core.project import Project
        self.fc = fc
        self.path = str(fcstd_path)
        self.project = Project()
        self.project._fc = fc              # share the persistent worker
        self.project.new_document(self.path)
        self.doc = self.project.doc
        self.notes = []
        self._var_row = 0
        self.detector_pins = []   # [(body_label, beam_dir)] see pin_detector
        self.grating_pins = []    # [(body_label, beam_dir, value)]

    def note(self, text):
        self.notes.append(text)

    # -- global variables ----------------------------------------------------
    def variable(self, name, value, vmin=None, vmax=None, nstep=None,
                 enabled=False, comment=""):
        """Add a row to the miewb_vars sheet: value cell aliased `name`
        (unitless), sweep metadata in the __min/__max/__n/__on columns.
        `value` may be an expression over other variables ("gap*2")."""
        if self._var_row == 0:
            self.fc.request("create_sheet",
                            {"doc": self.doc, "label": "miewb_vars"})
        self._var_row += 1
        r = self._var_row
        cells = [("A%d" % r, str(comment or name), None),
                 ("B%d" % r, "=%s" % value, name)]
        if vmin is not None:
            cells.append(("C%d" % r, "=%.10g" % vmin, "%s__min" % name))
        if vmax is not None:
            cells.append(("D%d" % r, "=%.10g" % vmax, "%s__max" % name))
        if nstep is not None:
            cells.append(("E%d" % r, "=%d" % nstep, "%s__n" % name))
        cells.append(("F%d" % r, "=%d" % (1 if enabled else 0),
                      "%s__on" % name))
        for cell, raw, alias in cells:
            req = {"doc": self.doc, "sheet": "miewb_vars",
                   "cell": cell, "raw": raw}
            if alias:
                req["alias"] = alias
            self.fc.request("set_cell", req)
        self.project._refetch_structure()
        return name

    # -- element creation ------------------------------------------------------
    def _import(self, kind, label, params=None, props=None):
        self.project.import_primitive(
            str(PRIMDIR / (kind + ".FCStd")), label)
        if params:
            spec = primitivelib.PRIMITIVES[kind]["params"]
            values = {}
            for alias, value in params.items():
                if isinstance(value, str):
                    # variable-driven dim: FreeCAD expression over the
                    # globals sheet, e.g. "<<miewb_vars>>.stop_d"
                    values[alias] = "=%s" % value
                else:
                    values[alias] = primitivelib.sheet_raw(
                        float(value), spec[alias]["unit"])
            self.project.set_element_parameters(
                "dim_%s" % label, values, rebuild_group=label)
        for key, value in (props or {}).items():
            body, prop = key if isinstance(key, tuple) else (label, key)
            self.project.set_property(body, prop, value)
        return label

    def add(self, kind, label, pos=(0.0, 0.0, 0.0), rot_deg=None, quat=None,
            params=None, props=None):
        """Import a primitive ANCHORED at an absolute pose (the pre-train
        idiom; still right for sources and one-off placements).
        rot_deg: rotation about world z; quat wins if given."""
        from mieworkbench.core.transforms import Operation
        self._import(kind, label, params, props)
        q = quat if quat is not None else rot_z(rot_deg or 0.0)
        if list(pos) != [0.0, 0.0, 0.0] or q != [0.0, 0.0, 0.0, 1.0]:
            body = self.project.body(label)["name"]
            self.project.apply_operation(body, Operation(
                "set_placement", {"pos_mm": list(pos), "quat": list(q)}))
        return label

    def chain(self, kind, label, ref, distance, port=None, params=None,
              props=None, **edge):
        """Import a primitive and CHAIN it `distance` mm down-beam of
        `ref` (vertex-to-vertex along the beam; expressions over the
        miewb_vars globals welcome). Extra edge fields pass through:
        decenter_x/y, tilt_rx/ry/rz, flip, fold, folded, fold_deviation,
        fold_azimuth, rot_order, pivot..."""
        self._import(kind, label, params, props)
        full = {"ref": ref, "distance": _expr(distance)}
        if port:
            full["port"] = port
        for k, v in edge.items():
            full[k] = _expr(v) if not isinstance(v, bool) else v
        self.project.set_chain(label, full, text="Chain %s" % label)
        return label

    def fold_mirror(self, label, ref, distance, deviation=90.0,
                    azimuth=0.0, kind="mirror_flat", port=None,
                    params=None, props=None):
        """A chained FOLD mirror: deviation/azimuth-driven orientation
        (rot_order zyx, tilt_ry = -(180-deviation)/2 — the same math as
        Project.insert_fold_mirror), unfoldable from the GUI."""
        tilt = "-(180 - (%s)) / 2" % _expr(deviation) \
            if isinstance(deviation, str) \
            else "%.10g" % (-(180.0 - float(deviation)) / 2.0)
        return self.chain(kind, label, ref, distance, port=port,
                          params=params, props=props,
                          fold=True, folded=True, rot_order="zyx",
                          tilt_rz=azimuth, tilt_ry=tilt)

    def body_labels(self, group):
        st = self.fc.request("get_structure", {"doc": self.doc})
        return [b["label"] for b in st["bodies"]
                if (b.get("properties", {}).get("miewb_group", {})
                    .get("value")) == group]

    def expect(self, label, pos, tol=1e-6):
        """Self-check: assert the element's primary body sits at `pos`
        (mm). Catches chain-arithmetic slips at BUILD time instead of at
        the equivalence gate."""
        body = self.project.train().primary_body_name(label)
        cur = self.project.body_states[body].current.to_dict()["pos_mm"]
        err = max(abs(c - p) for c, p in zip(cur, pos))
        if err > tol:
            raise AssertionError(
                "%s: expected %s, built %s (err %.3g mm)"
                % (label, list(pos), cur, err))

    def pin_detector(self, label, beam_dir):
        """Record that `label`'s recording face must be pinned in
        simparams: the extractor's closest-to-origin auto-pick selects a
        thin EDGE face on rotated off-axis detectors. FaceN numbering is
        NOT stable across rebuilds or even save/reload, so the id is
        resolved AFTER save by batch-extracting the shipped file (see
        resolve_detector_pins) and picking the plane face whose OUTWARD
        normal opposes the beam arrival direction `beam_dir`."""
        self.detector_pins.append((label, tuple(beam_dir)))

    def pin_grating(self, label, beam_dir, value):
        """Like pin_detector, for a grating face: the shipped plate's
        'Face1=...' property points at whatever face happens to be #1
        after this document's save/reload, so the demo passes an explicit
        --grating CLI override (which takes precedence) on the face
        resolved post-save from the beam direction."""
        self.grating_pins.append((label, tuple(beam_dir), value))

    def save(self):
        # run the chain validator first: a demo must never ship with a
        # cycle, dangling ref or unresolvable expression
        problems = self.project.train().validate()
        errors = [msg for sev, msg in problems if sev == "error"]
        if errors:
            raise AssertionError("train validation failed:\n  %s"
                                 % "\n  ".join(errors))
        self.project.save()


# ---------------------------------------------------------------------------
# the demos
# ---------------------------------------------------------------------------
def demo_beam_expander(d):
    """3x Keplerian beam expander: two BK7 plano-convex lenses (f=50 +
    f=150 at 650 nm), spacing f1+f2, convex sides outward. Train-built:
    everything chains down the beam; `sep` and `screen_dist` are live
    variables."""
    lam = 650.0
    n = n_glass("bk7", lam)
    f1, f2 = 50.0, 150.0
    r1, r2 = (n - 1.0) * f1, (n - 1.0) * f2
    ct1, ct2 = 3.0, 5.0
    d.variable("sep", f1 + f2, 150.0, 300.0, 6,
               comment="lens separation, outer vertex to outer vertex "
                       "(f1+f2 for a collimated output), mm")
    d.variable("screen_dist", 40.0, 10.0, 150.0, 6,
               comment="L2 outer vertex to screen, mm")
    d.add("laser_collimated", "Laser", pos=(-30, 0, 0),
          params={"diameter": 3.0},
          props={"lambdac": lam, "coherent": False})
    d.chain("lens_pcx", "L1", "Laser", 30.0,
            params={"R_front": r1, "aperture": 12.0, "ct": ct1})
    # convex side toward the expanded output: the flip edge field (the
    # old build hand-rolled a 180-deg rotation). Vertex-to-vertex gap =
    # sep minus both center thicknesses.
    d.chain("lens_pcx", "L2", "L1", "sep - %.10g" % (ct1 + ct2),
            flip=True,
            params={"R_front": r2, "aperture": 30.0, "ct": ct2})
    d.chain("detector_plane", "Screen", "L2", "screen_dist",
            params={"width": 40.0})
    d.expect("L1", (0, 0, 0))
    d.expect("L2", (f1 + f2, 0, 0))
    d.expect("Screen", (f1 + f2 + 40.0, 0, 0))
    d.note("beam_expander: linear chain; 'flip' replaced the hand-rolled "
           "180-deg rotation of the old build")
    return {"preset": "quick"}


def _newtonian_like(d, rfl, ap, L, mirror_t, diag_w, eye_w):
    """Shared Newtonian topology: star -> parabolic primary (retro
    reflect) -> flat diagonal (unfoldable 90-deg fold) -> eyepiece."""
    d.variable("eye_dist", L, 150.0, 300.0, 5,
               comment="diagonal center to eyepiece plane, mm")
    d.add("laser_collimated", "Star", pos=(-rfl - 60.0, 0, 0),
          params={"diameter": ap * 0.98},
          props={"lambdac": 550.0, "lambdamin": 450.0,
                 "lambdamax": 650.0, "coherent": False})
    d.chain("mirror_parabolic", "Primary", "Star", rfl + 60.0,
            params={"rfl": rfl, "aperture": ap, "thickness": mirror_t})
    # the converging cone travels -x after the primary; fold into +y.
    # Round flat: at 45 deg a circle presents a foreshortened
    # cone_diam/cos45 aperture (the D-shape's chord would clip the cone).
    # Beam frame after the retro: u = -y, so 'toward +y' = azimuth 180.
    d.fold_mirror("Diagonal", "Primary", "%.10g - eye_dist" % rfl,
                  azimuth=180.0,
                  params={"width": diag_w, "thickness": 5.0,
                          "round_flag": 1})
    d.chain("detector_plane", "Eyepiece", "Diagonal", "eye_dist",
            params={"width": eye_w})
    xd = -(rfl - L)
    d.expect("Primary", (0, 0, 0))
    d.expect("Diagonal", (xd, 0, 0))
    d.expect("Eyepiece", (xd, L, 0))
    d.pin_detector("Eyepiece", (0.0, 1.0, 0.0))


def demo_newtonian(d):
    """150 mm f/6 Newtonian: parabolic primary (rfl=900), 45-deg diagonal
    210 mm before focus, detector at the folded focal plane. The diagonal
    is a proper FOLD element: unfold it from the train editor and the
    eyepiece re-collinearizes onto the straight-through axis."""
    _newtonian_like(d, rfl=900.0, ap=150.0, L=210.0, mirror_t=15.0,
                    diag_w=52.0, eye_w=20.0)
    d.note("newtonian: the diagonal chains as a fold (deviation 90, "
           "azimuth 180) — no more hand-computed -135 deg quaternion")
    return {"preset": "quick"}


def demo_dobsonian(d):
    """200 mm f/5 Dobsonian: optically a Newtonian (the mount is not an
    optical element); bigger, faster prescription than the newtonian demo."""
    _newtonian_like(d, rfl=1000.0, ap=200.0, L=250.0, mirror_t=18.0,
                    diag_w=72.0, eye_w=25.0)
    return {"preset": "quick"}


def demo_michelson(d):
    """Michelson interferometer on the train model: laser -> 50:50 plate
    BS (both arms chained off its transmit/reflect ports) -> retro
    mirrors -> screen chained down the recombined return. M1 carries the
    fringe tilt as the live variable `m1_tilt` (lambda/(2 theta) sets the
    pitch); arm lengths are variables too."""
    tilt_deg = math.degrees(633e-9 / (2.0 * 0.002))   # 5 fringes / 10 mm
    # the transmit-port origin sits where the beam exits the tilted
    # plate's BACK face: thickness/cos(45) = 3*sqrt(2) past the center
    bs_exit = 3.0 * math.sqrt(2.0)
    d.variable("arm1", 60.0, 40.0, 100.0, 6,
               comment="BS center to M1 (transmit arm), mm")
    d.variable("arm2", 60.0, 40.0, 100.0, 6,
               comment="BS center to M2 (reflect arm), mm")
    d.variable("m1_tilt", "%.10g" % tilt_deg, 0.0, 0.05, 10,
               comment="M1 fringe tilt, deg (0.00907 deg ~ 5 fringes "
                       "across the 10 mm screen at 633 nm)")
    d.variable("screen_arm", 60.0,
               comment="BS center to screen (output arm), mm")
    d.add("laser_collimated", "Laser", pos=(-60, 0, 0),
          params={"diameter": 8.0},
          props={"lambdac": 633.0})          # coherent=True (default)
    # PLATE beamsplitter, not the cube: the cube's cemented 5 um gap sits
    # at 45 deg to the internal beams and bleeds ~37% of the power into
    # seam loss; the plate's coated front face splits cleanly (wedge 0 so
    # the two arms stay parallel). tilt_ry=45 puts the coated-face normal
    # at 225 deg: +x input reflects to -y.
    d.chain("bs_plate", "BS", "Laser", 60.0, tilt_ry=45.0,
            params={"width": 30.0, "thickness": 3.0, "round_flag": 1,
                    "wedge_deg": 0.0})
    # fringe tilt lives on M1 (end of the transmit arm — nothing chains
    # off its return, so the tilt perturbs no downstream frame; the OLD
    # build tilted M2 instead, which is optically equivalent)
    d.chain("mirror_flat", "M1", "BS", "arm1 - %.10g" % bs_exit,
            port="transmit", tilt_ry="m1_tilt",
            params={"width": 15.0, "round_flag": 1})
    d.chain("mirror_flat", "M2", "BS", "arm2", port="reflect",
            params={"width": 15.0, "round_flag": 1})
    # the recombined output: M2's retro return passes back through the
    # BS and lands on the screen at +y — chained straight down M2's
    # reflect port, total path arm2 + screen_arm from the M2 surface
    d.chain("detector_plane", "Screen", "M2", "arm2 + screen_arm",
            port="reflect",
            params={"width": 12.0, "round_flag": 0})
    d.expect("BS", (0, 0, 0))
    d.expect("M1", (60, 0, 0))
    d.expect("M2", (0, -60, 0))
    d.expect("Screen", (0, 60, 0))
    d.note("michelson: both interferometer arms hang off the BS ports; "
           "the fringe tilt is a plain variable (m1_tilt) instead of a "
           "hand-computed quaternion")
    d.pin_detector("Screen", (0.0, 1.0, 0.0))
    return {"preset": "quick"}


def demo_prism_spectrometer(d):
    """25 mm equilateral BK7-ish (F2-class) prism at minimum deviation for
    550 nm; broadband beam disperses ~2.3 deg over 450-650 nm onto a
    detector via a f=100 focusing lens (honest ~4 mm spectrum)."""
    # sf5 is the shipped dense flint (F2 itself is not in the registry)
    glass = "sf5"
    A = 60.0
    n550 = n_glass(glass, 550.0)
    dmin = 2.0 * math.degrees(math.asin(n550 * math.sin(math.radians(
        A / 2.0)))) - A
    # prism 'rotation' param turns the geometry itself; entrance-face
    # outward normal sits at 150 deg + rotation, so incidence
    # (A + Dmin)/2 against the +x beam needs:
    inc = (A + dmin) / 2.0
    rotation = 30.0 - inc
    dev_dir = unit(-dmin)          # deviated (toward the base, -y)
    # trace the CENTRAL ray (at the source height y0) through the two
    # faces to anchor the camera arm at the real exit point -- the beam
    # walks several mm down the exit face, so aiming from the prism
    # centroid misses (found by the first 0 mW smoke run). y0 lifts the
    # whole 8 mm bundle onto the entrance face (its lower vertex sits
    # just below the axis).
    y0 = 6.0
    Rc = 25.0 / math.sqrt(3.0)
    verts = [(Rc * math.cos(math.radians(a + rotation)),
              Rc * math.sin(math.radians(a + rotation)))
             for a in (90.0, 210.0, 330.0)]

    def _outward(pf, qf):
        ex, ey = qf[0] - pf[0], qf[1] - pf[1]
        nx, ny = ey, -ex
        mx = (pf[0] + qf[0]) / 2.0 - (verts[0][0] + verts[1][0]
                                      + verts[2][0]) / 3.0
        my = (pf[1] + qf[1]) / 2.0 - (verts[0][1] + verts[1][1]
                                      + verts[2][1]) / 3.0
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
    n_out = _outward(verts[2], verts[0])
    p_ent = _hit((-60.0, y0), (1.0, 0.0), verts[0], verts[1])
    d_in = _refract((1.0, 0.0), n_in, 1.0, n550)
    p_exit = _hit(p_ent, d_in, verts[2], verts[0])
    lens_c = (p_exit[0] + 40.0 * dev_dir[0], p_exit[1] + 40.0 * dev_dir[1])
    det_c = (p_exit[0] + 141.0 * dev_dir[0], p_exit[1] + 141.0 * dev_dir[1])

    d.variable("cam_dist", 40.0, 20.0, 80.0, 6,
               comment="prism exit point to camera lens, mm (along the "
                       "deviated 550 nm beam)")
    d.variable("det_dist", 97.0, 50.0, 150.0, 6,
               comment="camera back vertex to detector, mm")
    d.add("source_broadband", "Source", pos=(-60, y0, 0),
          params={"diameter": 8.0},
          props={"lambdac": 550.0, "lambdamin": 450.0, "lambdamax": 650.0})
    # the prism chains as a 'deviate' element: its port turns the train
    # by -dmin in the layout plane (deviate azimuth 90 = in-plane fold
    # axis +z, negative deviation = clockwise/toward -y). decenter_x
    # drops the prism centroid 6 mm below the beam line (the source is
    # lifted so the bundle lands on the entrance face).
    d.chain("prism", "Prism", "Source", 60.0, decenter_x=-y0,
            fold_deviation="%.10g" % (-dmin), fold_azimuth=90.0,
            params={"side": 25.0, "height": 25.0, "rotation": rotation},
            props={"material": glass})
    # the deviate port anchors at the prism CENTROID (the catalog prism's
    # documented port approximation), but the real 550 nm beam exits the
    # glass at p_exit after walking down the exit face — the offsets
    # below re-anchor the camera arm onto the traced beam (same trace the
    # old build used to aim the camera).
    dev = deviate_params((1.0, 0.0), dev_dir)
    u_dev = (-dev_dir[1], dev_dir[0])           # z x d = left of travel
    delta = (lens_c[0] - 0.0, lens_c[1] - 0.0)  # from the centroid origin
    along = delta[0] * dev_dir[0] + delta[1] * dev_dir[1]
    across = delta[0] * u_dev[0] + delta[1] * u_dev[1]
    n_lens = n_glass("bk7", 550.0)
    d.chain("lens_pcx", "Camera", "Prism", "%.10g + cam_dist" % (along
                                                                 - 40.0),
            decenter_x="%.10g" % across,
            params={"R_front": (n_lens - 1.0) * 100.0, "aperture": 22.0,
                    "ct": 4.0})
    # the camera's pass-through port continues the PORT line (centroid-
    # anchored), not the decentered true beam — carry the same offset
    d.chain("detector_plane", "Screen", "Camera", "det_dist",
            decenter_x="%.10g" % across,
            params={"width": 25.0})
    del dev
    d.expect("Prism", (0, 0, 0))
    d.expect("Camera", (lens_c[0], lens_c[1], 0))
    d.expect("Screen", (det_c[0], det_c[1], 0))
    d.note("prism_spectrometer: the prism is a deviate-port chain element "
           "(deviation = -Dmin); the camera arm still needed the offline "
           "central-ray trace to correct the centroid-port approximation "
           "for the beam's walk down the exit face")
    d.pin_detector("Screen", (dev_dir[0], dev_dir[1], 0.0))
    return {"preset": "quick"}


def demo_czerny_turner(d):
    """Crossed Czerny-Turner: divergent broadband slit source, R=200
    collimator, 600 g/mm grating (first order), R=200 camera mirror,
    400-700 nm across ~25 mm of detector."""
    theta_i, theta_d = 6.127, 25.896      # incidence / 550 nm diffraction
    off = 34.0                            # off-axis mirror working angle
    G = (0.0, 0.0)
    u = unit(180.0 - theta_i)             # grating -> collimator (-x side)
    v = unit(180.0 + theta_d)             # grating -> camera mirror
    C = (100.0 * u[0], 100.0 * u[1])
    M2 = (100.0 * v[0], 100.0 * v[1])
    w = unit(ang(u) + off)                # collimator -> slit direction
    S = (C[0] + 100.0 * w[0], C[1] + 100.0 * w[1])
    # mirror normals bisect in/out directions (mirror law n ~ d_out - d_in)
    n_coll = (-u[0] - (-w[0]), -u[1] - (-w[1]))     # w - u
    m = unit(ang((-v[0], -v[1])) - off)   # camera mirror -> detector
    n_cam = (m[0] - v[0], m[1] - v[1])
    D = (M2[0] + 100.0 * m[0], M2[1] + 100.0 * m[1])

    d.variable("arm_coll", 100.0, 80.0, 150.0, 5,
               comment="slit to collimator mirror, mm (its focal length)")
    d.variable("arm_cam", 100.0, 80.0, 150.0, 5,
               comment="grating to camera mirror, mm")
    d.variable("det_arm", 100.0, 80.0, 150.0, 5,
               comment="camera mirror to detector, mm (its focal length)")
    d.add("laser_divergent", "SlitSource",
          pos=(S[0] + 8.0 * w[0], S[1] + 8.0 * w[1], 0),
          rot_deg=ang((-w[0], -w[1])),
          params={"diameter": 2.0, "roc": 8.0, "length": 8.0},
          props={"lambdac": 550.0, "lambdamin": 400.0, "lambdamax": 700.0,
                 "coherent": False})
    # the whole spectrometer chains off the slit: every fold below is
    # deviation/azimuth data instead of the old hand-computed bisector
    # normals ('point this element at that one', finally)
    d.chain("slit", "Slit", "SlitSource", 8.0,
            params={"width": 20.0, "height": 20.0, "slit_width": 0.3,
                    "slit_height": 8.0})
    dev_c, _, az_c = deviate_params((-w[0], -w[1]), (-u[0], -u[1]))
    d.fold_mirror("Collimator", "Slit", "arm_coll", kind="mirror_concave",
                  deviation=dev_c, azimuth=az_c,
                  params={"R": 200.0, "aperture": 40.0, "ct": 6.0})
    # mirror=1.0 makes the diffraction REFLECTIVE (grating.apply_to_batch
    # switches on the body's mirror property >= 0.5, not the material) --
    # the catalog plate is a bk7 TRANSMISSION grating by default. The
    # grating face is pinned post-save via a --grating CLI override
    # (FaceN numbering is unstable across save/reload); the explicit
    # 0,1,0 periodicity vector keeps the orders in the x-y layout plane.
    # As a chain element the grating is a DEVIATE port (diffraction is
    # not specular: theta_i != theta_d), tilted so the plate normal sits
    # theta_i off the incoming beam.
    dev_g, gaz_g, _ = deviate_params((-u[0], -u[1]), (v[0], v[1]))
    d.chain("grating_plate", "Grating", "Collimator", "arm_coll",
            fold=True, folded=True,
            fold_deviation="%.10g" % dev_g, fold_azimuth=gaz_g,
            tilt_ry="%.10g" % (-ang((-u[0], -u[1]))),
            props={"material": "aluminum", "mirror": 1.0})
    d.pin_grating("Grating", (-u[0], -u[1], 0.0),
                  "600:0,1,0:orders=-1..1")
    dev_m, _, az_m = deviate_params((v[0], v[1]), (m[0], m[1]))
    d.fold_mirror("CameraMirror", "Grating", "arm_cam",
                  kind="mirror_concave", deviation=dev_m, azimuth=az_m,
                  params={"R": 200.0, "aperture": 40.0, "ct": 6.0})
    d.chain("detector_plane", "Screen", "CameraMirror", "det_arm",
            params={"width": 30.0})
    d.expect("Slit", (S[0], S[1], 0))
    d.expect("Collimator", (C[0], C[1], 0))
    d.expect("Grating", (0, 0, 0))
    d.expect("CameraMirror", (M2[0], M2[1], 0))
    d.expect("Screen", (D[0], D[1], 0))
    d.note("czerny_turner: the four aim angles are now deviation/azimuth "
           "chain data (deviate_params) — zero hand-placed quaternions; "
           "the arms are live variables")
    d.pin_detector("Screen", (m[0], m[1], 0.0))
    return {"preset": "quick"}


def demo_camera_triplet(d):
    """Cooke triplet, ~50 mm EFL (MathWorks design uniformly rescaled),
    iris stopped to ~f/5.6, full-frame 36x24 mm sensor at the paraxially
    computed focus (using the shipped bk7/sf5 glasses)."""
    lam = 550.0
    # scaled prescription: (R1, R2, ct, aperture-diam) per element +
    # air gaps; flint element uses sf5 (the shipped dense flint)
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
    x_img = paraxial_image_x(surfaces, float("-inf"), lam)

    d.variable("air12", air12, 3.0, 8.0, 5,
               comment="L1 back vertex to L2 front vertex, mm")
    d.variable("air23", air23, 3.0, 8.0, 5,
               comment="L2 back vertex to L3 front vertex, mm")
    d.variable("stop_d", 6.94, 3.0, 12.0, 6,
               comment="iris opening diameter, mm (~f/5.6 at 6.94)")
    d.add("laser_collimated", "Scene", pos=(-40, 0, 0),
          params={"diameter": 14.0},
          props={"lambdac": lam, "lambdamin": 450.0, "lambdamax": 650.0,
                 "coherent": False})
    d.chain("lens_dcx", "L1", "Scene", 40.0,
            params={"R_front": L1["R_front"], "R_back": L1["R_back"],
                    "ct": L1["ct"], "aperture": L1["ap"]})
    d.chain("lens_dcv", "L2", "L1", "air12",
            params={"R_front": L2["R_front"], "R_back": L2["R_back"],
                    "ct": L2["ct"], "aperture": L2["ap"]},
            props={"material": "sf5"})
    # the stop sits just behind the flint element; its concave back
    # surface bulges 0.64 mm past the vertex at this aperture, so leave
    # ~1 mm of axial clearance or the solids overlap. The opening is
    # variable-driven THROUGH the dim sheet (a FreeCAD expression over
    # the miewb_vars global), so sweeping stop_d rebuilds the iris.
    d.chain("iris", "Stop", "L2", 0.95,
            params={"outer_diameter": 18.0, "thickness": 0.4,
                    "hole_diameter": "<<miewb_vars>>.stop_d * 1mm"})
    d.chain("lens_dcx", "L3", "Stop", "air23 - 0.95",
            params={"R_front": L3["R_front"], "R_back": L3["R_back"],
                    "ct": L3["ct"], "aperture": L3["ap"]})
    d.chain("detector_plane", "Sensor", "L3",
            "%.10g" % (x_img - (x3 + L3["ct"])),
            params={"width": 36.0, "height": 24.0, "round_flag": 0})
    d.expect("L1", (x1, 0, 0))
    d.expect("L2", (x2, 0, 0))
    d.expect("Stop", (x2 + L2["ct"] + 0.95, 0, 0))
    d.expect("L3", (x3, 0, 0))
    d.expect("Sensor", (x_img, 0, 0))
    d.note("camera_triplet: air gaps and the iris opening are live "
           "variables (the stop drives its dim sheet through a FreeCAD "
           "expression); the paraxial focus still came from the offline "
           "solve (%.2f mm)" % x_img)
    return {"preset": "quick"}


def demo_microscope_objective(d):
    """Simplified Lister-type 10x objective: two air-spaced achromats
    (f=20 front, f=40 rear, 10 mm apart), point source at the object
    plane, image on a detector at the paraxial conjugate (~10x)."""
    lam = 550.0
    # f=25/f=50 pair (Lister's 2:1 ratio): the f=20 design's scaled
    # interface radius (8.8 mm) leaves no aperture margin for the NA-0.25
    # cone, and its crown/flint local geometry self-overlaps at 10 mm
    a1 = solve_achromat(25.0)
    a2 = solve_achromat(50.0)
    x1, x2 = 0.0, 10.0
    x_obj = -22.0

    def ach_surfaces(x0, a):
        return [
            (x0, a["R_front"], None, "bk7"),
            (x0 + a["ct_crown"], a["R_iface"], "bk7", "sf5"),
            (x0 + a["ct_crown"] + a["ct_flint"], a["R_back"], "sf5", None),
        ]
    surfaces = ach_surfaces(x1, a1) + ach_surfaces(x2, a2)
    x_img = paraxial_image_x(surfaces, x_obj, lam)

    d.add("laser_divergent", "Object", pos=(x_obj, 0, 0),
          params={"diameter": 2.0, "roc": 5.0, "length": 6.0},
          props={"lambdac": lam, "lambdamin": 450.0, "lambdamax": 650.0,
                 "coherent": False})
    # aperture must stay under the scaled R_iface (|R|=8.8 mm at f=20)
    # or the meridian arcs cannot close -- 10 mm covers the NA 0.25 cone
    def ach_params(a, aperture):
        out = {k: a[k] for k in ("R_front", "R_iface", "R_back",
                                 "ct_crown", "ct_flint")}
        out["aperture"] = aperture
        return out
    # exit vertices come from the same port formulas the solver uses
    exit1 = primitivelib.port_frames("lens_achromat",
                                     ach_params(a1, 10.0))["exit"][0]
    exit2 = primitivelib.port_frames("lens_achromat",
                                     ach_params(a2, 12.0))["exit"][0]
    d.variable("obj_dist", -x_obj, 15.0, 35.0, 5,
               comment="object plane to front achromat vertex, mm")
    d.variable("ach_gap", x2 - x1, 5.0, 20.0, 5,
               comment="achromat front-vertex spacing, mm")
    d.chain("lens_achromat", "Front", "Object", "obj_dist",
            params=ach_params(a1, 10.0))
    d.chain("lens_achromat", "Rear", "Front",
            "ach_gap - %.10g" % exit1, params=ach_params(a2, 12.0))
    d.chain("detector_plane", "Image", "Rear",
            "%.10g" % (x_img - (x2 + exit2)), params={"width": 30.0})
    d.expect("Front", (x1, 0, 0))
    d.expect("Rear", (x2, 0, 0))
    d.expect("Image", (x_img, 0, 0))
    d.note("microscope_objective: solve_achromat covered the lens design "
           "and the chain uses its exact exit vertices; the image plane "
           "still came from the offline paraxial solve (%.1f mm)" % x_img)
    return {"preset": "quick"}


def demo_fiber_coupler(d):
    """Ball-lens fiber coupler: 650 nm laser -> 2 mm BK7 ball (BFL 0.47 mm)
    -> 75 mm of 200 um / 0.22 NA step-index fiber (TIR guiding) ->
    detector at the exit face."""
    lam = 650.0
    n = n_glass("bk7", lam)
    R = 1.0
    bfl = R * (2.0 - n) / (2.0 * (n - 1.0))
    x_fiber = 2.0 * R + bfl
    d.variable("exit_gap", 0.6, 0.1, 3.0, 6,
               comment="fiber exit face to detector, mm")
    d.add("laser_collimated", "Laser", pos=(-6, 0, 0),
          params={"diameter": 0.6, "length": 5.0},
          props={"lambdac": lam, "coherent": False})
    d.chain("lens_ball", "Ball", "Laser", 6.0,
            params={"diameter": 2.0 * R})
    d.chain("fiber_optic", "Fiber", "Ball", "%.10g" % bfl,
            params={"length": 75.0})
    d.chain("detector_plane", "Exit", "Fiber", "exit_gap",
            params={"width": 2.0, "round_flag": 0})
    d.expect("Ball", (0, 0, 0))
    d.expect("Fiber", (x_fiber, 0, 0))
    d.expect("Exit", (x_fiber + 75.0 + 0.6, 0, 0))
    d.note("fiber_coupler: the ball-to-fiber gap IS the BFL, now visible "
           "as a chain distance instead of a baked coordinate")
    return {"preset": "quick", "max_reflections": 200}


def demo_schmidt_cassegrain(d):
    """C8-style 203 mm f/10 Schmidt-Cassegrain (Suiter's table): quartic
    corrector (hand-authored, added post-assembly), perforated spherical
    primary (R=812.8), spherical secondary (R=231.07), focus 150 mm behind
    the primary vertex."""
    # corrector (added post-assembly) occupies x = 0..5; Suiter's spacings
    # measure 320 mm of air from its BACK face to the primary vertex
    x_primary = 5.0 + 320.0
    x_secondary = x_primary - 312.62
    x_focus = x_primary + 150.0
    d.variable("sct_sep", 312.62, 280.0, 340.0, 6,
               comment="primary vertex to secondary vertex, mm")
    d.variable("back_focus", 150.0, 100.0, 200.0, 5,
               comment="primary vertex plane to focal plane, mm")
    d.add("laser_collimated", "Star", pos=(-60, 0, 0),
          params={"diameter": 198.0},
          props={"lambdac": 550.0, "lambdamin": 450.0,
                 "lambdamax": 650.0, "coherent": False})
    d.chain("mirror_annular", "Primary", "Star", x_primary + 60.0,
            params={"R": 812.8, "aperture": 203.2, "hole_diameter": 60.0,
                    "ct": 18.0})
    # the cone reflected off the primary travels -x; the chained
    # secondary auto-faces the returning beam (the old build needed an
    # explicit 180-deg flip), and the beam bounces back +x through the
    # primary's perforation to the focal plane behind it
    d.chain("mirror_convex", "Secondary", "Primary", "sct_sep",
            port="reflect",
            params={"R": 231.07, "aperture": 66.0, "ct": 6.0})
    d.chain("detector_plane", "Focus", "Secondary",
            "sct_sep + back_focus", port="reflect",
            params={"width": 15.0})
    d.expect("Primary", (x_primary, 0, 0))
    d.expect("Secondary", (x_secondary, 0, 0))
    d.expect("Focus", (x_focus, 0, 0))
    d.note("schmidt_cassegrain: the Cassegrain return path chains off "
           "reflect ports (back focus is a live variable); the corrector "
           "still needs the hand-authored FreeCAD pass (quartic asphere)")
    return {"preset": "quick", "corrector": True}


def demo_michelson_folded(d):
    """Michelson with a Z-FOLDED transmit arm: two extra 45-deg fold
    mirrors compact the 60 mm arm into a 45 x 15 mm dogleg with the SAME
    optical path length. Both folds are proper fold elements — unfold
    them in the train editor and the arm re-collinearizes into the plain
    michelson layout exactly. The fold mirrors are bare aluminum (real
    Fresnel losses, ~0.9 reflectance x 4 extra bounces round-trip); the
    `ideal_folds` variable switches them to perfect mirrors (mirror=1)
    for an efficiency A/B."""
    tilt_deg = math.degrees(633e-9 / (2.0 * 0.002))
    bs_exit = 3.0 * math.sqrt(2.0)
    d.variable("arm1", 60.0, 40.0, 100.0, 6,
               comment="BS center to M1, TOTAL optical path (matches the "
                       "unfolded michelson), mm")
    d.variable("fold_in", 20.0, 10.0, 30.0, 4,
               comment="BS center to first fold mirror, mm")
    d.variable("fold_up", 15.0, 8.0, 25.0, 4,
               comment="dogleg height (FoldA to FoldB), mm")
    d.variable("arm2", 60.0, 40.0, 100.0, 6,
               comment="BS center to M2 (reflect arm), mm")
    d.variable("m1_tilt", "%.10g" % tilt_deg, 0.0, 0.05, 10,
               comment="M1 fringe tilt, deg")
    d.variable("screen_arm", 60.0,
               comment="BS center to screen (output arm), mm")
    d.variable("ideal_folds", 0.0, 0.0, 1.0, 1,
               comment="0 = bare-aluminum fold mirrors (honest ~10% loss "
                       "per bounce), 1 = perfect fold mirrors")
    d.add("laser_collimated", "Laser", pos=(-60, 0, 0),
          params={"diameter": 8.0},
          props={"lambdac": 633.0})
    d.chain("bs_plate", "BS", "Laser", 60.0, tilt_ry=45.0,
            params={"width": 30.0, "thickness": 3.0, "round_flag": 1,
                    "wedge_deg": 0.0})
    # the Z-fold: +x -> +y (azimuth 0 = toward +u = +y here) -> back to
    # +x (incoming +y has u = -x, so azimuth 180). Bare aluminum, with
    # the perfect-mirror switch expression-driven off ideal_folds.
    fold_props = {"material": "aluminum", "mirror": 0.0,
                  "miewb_expr_mirror": "ideal_folds"}
    d.fold_mirror("FoldA", "BS", "fold_in - %.10g" % bs_exit,
                  port="transmit", azimuth=0.0,
                  params={"width": 15.0, "round_flag": 1},
                  props=fold_props)
    d.fold_mirror("FoldB", "FoldA", "fold_up", azimuth=180.0,
                  params={"width": 15.0, "round_flag": 1},
                  props=fold_props)
    d.chain("mirror_flat", "M1", "FoldB", "arm1 - fold_in - fold_up",
            tilt_ry="m1_tilt",
            params={"width": 15.0, "round_flag": 1})
    d.chain("mirror_flat", "M2", "BS", "arm2", port="reflect",
            params={"width": 15.0, "round_flag": 1})
    d.chain("detector_plane", "Screen", "M2", "arm2 + screen_arm",
            port="reflect",
            params={"width": 12.0, "round_flag": 0})
    d.expect("BS", (0, 0, 0))
    d.expect("FoldA", (20, 0, 0))
    d.expect("FoldB", (20, 15, 0))
    d.expect("M1", (45, 15, 0))
    d.expect("M2", (0, -60, 0))
    d.expect("Screen", (0, 60, 0))
    d.note("michelson_folded: unfolding BOTH folds reproduces the plain "
           "michelson layout (M1 back at x=60) — the folded arm keeps "
           "the optical path by construction")
    d.pin_detector("Screen", (0.0, 1.0, 0.0))
    return {"preset": "quick"}


def demo_ktp_walkoff(d):
    """Biaxial walk-off bench: 633 nm narrow unpolarized beam through a 15 mm
    KTP plate whose X principal axis is at 45 deg in the layout plane (Y
    principal = out-of-plane) — the maximum-walk-off geometry. The in-plane
    sheet walks off ~0.85 mm in z while the out-of-plane (n_y) sheet goes
    straight: two spots on the screen."""
    d.add("laser_collimated", "Laser", pos=(-10, 0, 0),
          params={"diameter": 0.3, "length": 4.0},
          props={"lambdac": 633.0, "coherent": False,
                 "polarization": "unpolarized"})
    d.chain("window", "KTP", "Laser", 10.0,
            params={"width": 12.0, "thickness": 15.0, "round_flag": 0},
            props={"material": "ktp",
                   "crystal_axis": "0.70711,0,0.70711",
                   "crystal_axis2": "0,1,0"})
    d.chain("detector_plane", "Screen", "KTP", 5.0,
            params={"width": 8.0, "round_flag": 0})
    d.expect("KTP", (0, 0, 0))
    d.expect("Screen", (20, 0, 0))
    d.note("ktp_walkoff: biaxial two-sheet walk-off (KTP), crystal frame set "
           "with crystal_axis + crystal_axis2")
    return {"preset": "quick"}


def demo_gaussian_bench(d):
    """Gaussian-beam propagation bench: a 50 um-waist (M2=1.0) 633 nm source
    (beam_waist + m2 props, incoherent beam mode) expands ~5x over 62 mm (5
    Rayleigh ranges) of empty air onto a screen."""
    d.add("laser_collimated", "Laser", pos=(-2, 0, 0),
          params={"diameter": 2.0, "length": 4.0},
          props={"lambdac": 633.0, "coherent": False,
                 "beam_waist": 0.05, "m2": 1.0})
    d.chain("detector_plane", "Screen", "Laser", 62.0,
            params={"width": 5.0, "round_flag": 1})
    d.expect("Screen", (60, 0, 0))
    d.note("gaussian_bench: Gaussian source beam mode (beam_waist=50um, "
           "M2=1.0) expands 5x over 5 Rayleigh ranges")
    return {"preset": "quick"}


def demo_ghost_doublet(d):
    """Fresnel-ghost bench: two uncoated N-BK7 windows (4 mm thick, 8 mm
    vertex spacing) in a collimated 633 nm incoherent beam. The screen
    records the direct beam plus the natural double-bounce Fresnel ghosts
    (each ~R^2 of the direct beam). Run with --ghost-analysis to enumerate
    the ghost paths."""
    d.add("laser_collimated", "Laser", pos=(-20, 0, 0),
          params={"diameter": 6.0, "length": 8.0},
          props={"lambdac": 633.0, "coherent": False})
    d.chain("window", "Glass1", "Laser", 20.0,
            params={"width": 20.0, "thickness": 4.0, "round_flag": 0})
    d.chain("window", "Glass2", "Glass1", 4.0,
            params={"width": 20.0, "thickness": 4.0, "round_flag": 0})
    d.chain("detector_plane", "Screen", "Glass2", 18.0,
            params={"width": 20.0, "round_flag": 0})
    d.expect("Glass1", (0, 0, 0))
    d.expect("Glass2", (8, 0, 0))
    d.expect("Screen", (30, 0, 0))
    d.note("ghost_doublet: two uncoated BK7 windows -> natural Fresnel "
           "ghosts (view with --ghost-analysis)")
    return {"preset": "quick", "ghost_analysis": True}


def demo_scatter_plate(d):
    """Measured-scatter bench: a BK7 window at 45 deg with an ABg scatter
    finish (scatter=polished_bk7_glass); the collimated 633 nm beam's Fresnel
    reflection folds to +y where a screen catches the specular spot plus the
    diffuse scatter lobe. (Anchored poses: the reflected arm is a hard 90 deg
    turn off a plain window face.)"""
    d.add("laser_collimated", "Laser", pos=(-30, 0, 0),
          params={"diameter": 6.0, "length": 8.0},
          props={"lambdac": 633.0, "coherent": False})
    d.add("window", "Window", pos=(0, 0, 0), quat=rot_z(-45.0),
          params={"width": 24.0, "thickness": 3.0, "round_flag": 0},
          props={"material": "bk7", "scatter": "polished_bk7_glass"})
    d.add("detector_plane", "DetRefl", pos=(0, 30, 0), quat=rot_z(90.0),
          params={"width": 30.0, "round_flag": 0})
    d.note("scatter_plate: ABg measured scatter (polished_bk7_glass) on a "
           "45 deg window; the reflected arm catches specular + scatter")
    return {"preset": "quick"}


def demo_curved_focal(d):
    """Curved-detector bench: a collimated 633 nm beam is focused by a BK7
    plano-convex lens (f~48.5 mm) onto a CYLINDRICAL detector (a lens_cyl
    body tagged material=detector, axis along z) whose curved face sits at
    the focus — a curved focal-surface screen. coherent=False (geometric
    focus)."""
    d.add("laser_collimated", "Laser", pos=(-30, 0, 0),
          params={"diameter": 10.0, "length": 8.0},
          props={"lambdac": 633.0, "coherent": False})
    d.chain("lens_pcx", "Lens", "Laser", 30.0,
            params={"R_front": 25.0, "ct": 5.0, "aperture": 20.0})
    # focus ~= back vertex (x=5) + BFL(45.24) = 50.24; a convex cylindrical
    # screen (lens_cyl material=detector) anchored with its front vertex on
    # the focus (lens_cyl can't chain — no port frame — so it is anchored).
    d.add("lens_cyl", "CurvedDet", pos=(50.24, 0, 0),
          params={"R": 20.0, "ct": 3.0, "aperture": 12.0, "height": 12.0},
          props={"material": "detector"})
    d.expect("Lens", (0, 0, 0))
    d.note("curved_focal: PCX lens focuses onto a cylindrical (curved) "
           "detector face at the focus")
    return {"preset": "quick"}


DEMOS = {
    "beam_expander": demo_beam_expander,
    "ktp_walkoff": demo_ktp_walkoff,
    "gaussian_bench": demo_gaussian_bench,
    "ghost_doublet": demo_ghost_doublet,
    "scatter_plate": demo_scatter_plate,
    "curved_focal": demo_curved_focal,
    "newtonian": demo_newtonian,
    "dobsonian": demo_dobsonian,
    "michelson": demo_michelson,
    "michelson_folded": demo_michelson_folded,
    "prism_spectrometer": demo_prism_spectrometer,
    "czerny_turner": demo_czerny_turner,
    "camera_triplet": demo_camera_triplet,
    "microscope_objective": demo_microscope_objective,
    "fiber_coupler": demo_fiber_coupler,
    "schmidt_cassegrain": demo_schmidt_cassegrain,
}


def _resolve_front_face(model, label, beam_dir):
    """The plane face of `label` whose OUTWARD normal opposes the beam
    arrival direction (unique: the far cap points along the beam, edges
    are perpendicular)."""
    body = next(b for b in model["bodies"]
                if b.get("label", b["name"]) == label or b["name"] == label)
    best, best_dot = None, -2.0
    for f in body["faces"]:
        surf = f.get("surface", {})
        if surf.get("type") != "plane":
            continue
        n = surf.get("normal") or [0.0, 0.0, 0.0]
        if not f.get("orientation_outward", True):
            n = [-c for c in n]
        dot = -sum(a * b for a, b in zip(n, beam_dir))
        if dot > best_dot:
            best, best_dot = f["id"], dot
    if best is None or best_dot < 0.7:
        raise RuntimeError("could not resolve front face for %s "
                           "(best dot %.2f)" % (label, best_dot))
    return best


def resolve_detector_pins(fcstd_path, pins):
    """Batch-extract the SAVED scene and resolve each pinned detector's
    recording face: the plane face whose outward normal opposes the
    demo's known beam arrival direction (unique -- the opposite cap
    points along the beam and the edges are perpendicular). Extraction
    of the same saved file is exactly what run_pipeline does, so the
    FaceN ids match the trace's numbering."""
    import json
    outdir = Path(tempfile.mkdtemp(prefix="miewb-demo-extract-"))
    try:
        subprocess.run(
            [str(FREECAD), "-c",
             str(REPO / "scripts" / "extract_geometry.py"), "--",
             "--model", str(fcstd_path), "--outdir", str(outdir)],
            stdin=subprocess.DEVNULL, check=True, capture_output=True,
            text=True)
        model_path = outdir / Path(fcstd_path).stem / "model.json"
        model = json.loads(model_path.read_text())
        detector_pins, grating_pins = pins
        det = [_resolve_front_face(model, label, beam_dir)
               for label, beam_dir in detector_pins]
        grat = ["%s:%s" % (_resolve_front_face(model, label, beam_dir),
                           value)
                for label, beam_dir, value in grating_pins]
        return det, grat
    finally:
        shutil.rmtree(str(outdir), ignore_errors=True)


def add_corrector(fcstd_path):
    """Run the FreeCAD helper that adds the hand-authored Schmidt
    corrector body to the saved scene."""
    script = REPO / "scripts" / "tools" / "add_schmidt_corrector.py"
    result = subprocess.run(
        [str(FREECAD), "-c", str(script), "--", "--model", str(fcstd_path)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True)
    ok = "CORRECTOR OK" in (result.stdout + result.stderr)
    if not ok:
        raise RuntimeError("add_schmidt_corrector failed:\n%s\n%s"
                           % (result.stdout[-2000:], result.stderr[-500:]))


def build_demo(name, outdir, pack=True):
    outdir.mkdir(parents=True, exist_ok=True)
    fcstd = outdir / ("%s.FCStd" % name)
    if fcstd.exists():
        fcstd.unlink()
    fc = FcClient()
    try:
        demo = Demo(fc, fcstd)
        simparams = DEMOS[name](demo)
        demo.save()
    finally:
        fc.shutdown()
    wants_corrector = simparams.pop("corrector", False)
    if wants_corrector:
        add_corrector(fcstd)
    if demo.detector_pins or demo.grating_pins:
        det, grat = resolve_detector_pins(
            fcstd, (demo.detector_pins, demo.grating_pins))
        if det:
            simparams["detector_face"] = det
        if grat:
            simparams["grating"] = grat
    if pack:
        miewb_tool.pack_miewb(fcstd, outdir / ("%s.MieWB" % name),
                              simparams=simparams)
    print("[demo] %-22s -> %s%s" % (name, fcstd.name,
                                    " (+MieWB)" if pack else ""),
          flush=True)
    return demo.notes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", default="all",
                   help="demo name or 'all' (see --list)")
    p.add_argument("--outdir", default=str(REPO / "demos"))
    p.add_argument("--no-pack", action="store_true",
                   help="skip the .MieWB packing step")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        for name in sorted(DEMOS):
            print(name)
        return 0
    names = sorted(DEMOS) if args.demo == "all" else [args.demo]
    all_notes = []
    for name in names:
        if name not in DEMOS:
            p.error("unknown demo %r" % name)
        all_notes += build_demo(name, Path(args.outdir),
                                pack=not args.no_pack)
    if all_notes:
        print("\nUX notes gathered while building:")
        for n in all_notes:
            print("  - " + n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
