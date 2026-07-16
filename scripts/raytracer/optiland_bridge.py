# =============================================================================
# optiland_bridge.py -- convert a SIMPLE, sequential-expressible MieWorkbench
# scene (a geometry/<stem>/model.json, the SAME data the trace engine reads --
# NOT FreeCAD) into an Optiland sequential `Optic`, for the P4a parity ORACLE
# (engine3.md Section 5 / Section 15 P4a: "an arbiter exists before a second
# physics truth does").
#
# Interpreter: env/bin/python  (the project GUI venv, which is where Optiland
# is installed -- see INSTALL.md's "Optional: the Optiland parity oracle"
# section). This module is NOT importable under /home3/optics/env (the engine
# env), which deliberately never gets Optiland; the engine itself never imports
# this module (guarded by the task contract). It DOES import raytracer.optprops
# (numpy-only, importable under both envs) to resolve glass indices from the
# SAME materials registry the engine uses, so the two physics truths share one
# material library.
#
# SCOPE (oracle-minimal, on purpose -- this is a geometric arbiter, not a
# general importer):
#   * rotationally symmetric, on-axis, refractive elements: sphere / plane
#     optical faces, one common axis (+x world = +z Optiland);
#   * homogeneous glasses (n resolved from the registry at the test
#     wavelength, installed as an Optiland IdealMaterial -- dispersion within
#     a single monochromatic geometric oracle is irrelevant);
#   * air gaps between elements; a flat image (detector) plane.
# Explicitly OUT of scope (raises BridgeUnsupported, never a silent wrong
# answer): tilts/decenters, conics/aspheres/Forbes, cylinders as optical
# surfaces (barrel rims are skipped, not refracted), gratings, mirrors/folds,
# birefringence, diffractive/coating physics. The aperture stop is DROPPED as
# a refracting surface (an air-filled iris does not bend rays); vignetting is
# whatever the Monte-Carlo engine already applied -- the oracle traces the
# exact pupil positions the MC run recorded, so the stop's only job (clipping)
# is inherited, not re-derived.
#
# UNIT CONTRACT (pinned in ONE place, test_optiland_oracle.py::
# test_unit_contract): model.json is SI metres; Optiland is millimetres.
# MM_PER_M is the single conversion constant; every length crossing the bridge
# is multiplied by it exactly once.
# =============================================================================
import json
import warnings
from pathlib import Path

import numpy as np

# The ONE length conversion: model.json metres -> Optiland millimetres.
MM_PER_M = 1000.0

# absolute metre pad when checking an axial vertex lies inside a body bbox.
_AXIS_TOL_M = 1e-9


class BridgeUnsupported(Exception):
    """The scene uses a feature outside the oracle bridge's sequential scope."""


# --------------------------------------------------------------------------
# model.json -> a plain, engine-independent sequential description
# --------------------------------------------------------------------------
class OpticalSurface:
    """One refracting surface on the common axis, in SI metres (world +x)."""

    def __init__(self, vertex_x_m, radius_m, is_flat, body_label,
                 material_after, is_entry):
        self.vertex_x_m = vertex_x_m          # axial intercept, world metres
        self.radius_m = radius_m              # signed: +ve if centre is at +x
        self.is_flat = is_flat                # plane -> infinite radius
        self.body_label = body_label
        self.material_after = material_after   # glass name or "air" downstream
        self.is_entry = is_entry              # entry (glass side) vs exit (air)


class SequentialSystem:
    """The bridge's engine-neutral intermediate form."""

    def __init__(self, surfaces, image_z_m, wavelength_nm, ambient,
                 stem, det_label):
        self.surfaces = surfaces              # z-ordered list of OpticalSurface
        self.image_z_m = image_z_m            # detector plane, world metres
        self.wavelength_nm = wavelength_nm
        self.ambient = ambient                # ambient material name
        self.stem = stem
        self.det_label = det_label


def _face_optical_kind(face):
    """'sphere' | 'plane' | None (None = not an axial refracting face)."""
    t = (face.get("surface") or {}).get("type")
    return t if t in ("sphere", "plane") else None


def _sphere_vertex_and_radius(face, bbox):
    """Axial vertex (world x) and SIGNED radius (Optiland convention: +ve if
    the centre of curvature is downstream, at greater x). The sphere crosses
    the axis at center_x +/- |R|; the physical vertex is the intercept nearest
    the face centroid (robust for menisci where both intercepts are finite)."""
    surf = face["surface"]
    cx = surf["center"][0]
    R = abs(surf["radius"])
    cand = (cx - R, cx + R)
    fcx = face["fingerprint"]["centroid"][0]        # the surface patch sits here
    vertex = min(cand, key=lambda v: abs(v - fcx))
    lo, hi = bbox["min"][0] - _AXIS_TOL_M, bbox["max"][0] + _AXIS_TOL_M
    if not (lo <= vertex <= hi):
        raise BridgeUnsupported(
            "sphere vertex %.6g m outside body bbox [%.6g, %.6g] -- off-axis "
            "or non-sequential surface" % (vertex, lo, hi))
    signed_R = R if cx > vertex else -R
    return vertex, signed_R


def _plane_vertex(face):
    """Plane axial vertex = its origin x (the plane must be normal to the
    axis; verified here)."""
    surf = face["surface"]
    n = np.asarray(surf["normal"], float)
    if abs(abs(n[0]) - 1.0) > 1e-6:
        raise BridgeUnsupported(
            "plane normal %s is not along the optical axis (+x)" % (n.tolist(),))
    return surf["origin"][0]


def _is_glass(name):
    """A body material that could be a refracting LENS element -- named, and
    neither ambient air nor vacuum/none (an air-filled iris plug is not a
    lens)."""
    return bool(name) and str(name).lower() not in ("air", "none", "vacuum")


# absorptive-index threshold above which a body is treated as opaque (a stop /
# iris / mirror substrate), NOT a refracting lens. bk7/sf5 have k==0; aluminium
# and other metals have k of order 1-10, so this cleanly separates the two.
_OPAQUE_K = 1e-6


def _medium_nk(name, wavelength_nm, optprops_dir=None):
    """(n, k) of ANY named medium at the wavelength, from the engine's OWN
    registry -- INCLUDING air (n=1.000272 at 633 nm, the real ambient the C
    engine traces in). Only vacuum/none/None (and names absent from the
    registry) fall back to (1.0, 0.0)."""
    if not name or str(name).lower() in ("none", "vacuum"):
        return 1.0, 0.0
    from raytracer import optprops                  # engine-env-safe (numpy)
    root = optprops_dir or (Path(__file__).resolve().parents[2]
                            / "opticalproperties")
    mdb = optprops.load_optical_properties(str(root)).matdb
    try:
        nc = mdb.get(name).n_complex(wavelength_nm * 1e-9)
    except Exception:
        return 1.0, 0.0
    return float(nc.real), abs(float(nc.imag))


def _is_transparent_dielectric(name, wavelength_nm, optprops_dir=None):
    """A refracting-lens material: a lens-candidate glass (not air/vacuum) that
    is non-absorbing at the test wavelength (so an aluminium iris annulus,
    with large k, is excluded)."""
    if not _is_glass(name):
        return False
    return _medium_nk(name, wavelength_nm, optprops_dir)[1] <= _OPAQUE_K


def load_sequential_system(geometry_dir, wavelength_nm=None):
    """model.json -> SequentialSystem. wavelength_nm defaults to the scene
    source's lambdac. Raises BridgeUnsupported on out-of-scope features."""
    geometry_dir = Path(geometry_dir)
    with open(geometry_dir / "model.json") as fh:
        model = json.load(fh)
    ambient = model.get("ambient_material", "air")

    optics, detectors, sources = [], [], []
    for b in model["bodies"]:
        role = b.get("role")
        if role == "detector":
            detectors.append(b)
        elif role == "source":
            sources.append(b)
        elif role == "optic":
            optics.append(b)

    if len(sources) != 1:
        raise BridgeUnsupported("oracle bridge needs exactly one source, got %d"
                                % len(sources))
    if len(detectors) != 1:
        raise BridgeUnsupported("oracle bridge needs exactly one detector, got "
                                "%d" % len(detectors))
    src = sources[0]["source"]
    if src.get("coherent", False):
        warnings.warn("source is coherent; oracle geometry is unaffected but "
                      "the MC side should run coherent=false")
    if wavelength_nm is None:
        wavelength_nm = float(src["lambdac_nm"])

    det = detectors[0]
    det_face_id = det["detector"]["face"]
    det_z = None
    for f in det["faces"]:
        if f["id"] == det_face_id:
            det_z = _plane_vertex(f)
    if det_z is None:
        raise BridgeUnsupported("detector primary face %s not found" % det_face_id)

    surfaces = []
    for b in optics:
        mat = b.get("material")
        if not _is_transparent_dielectric(mat, wavelength_nm):
            # air plug / iris fill / opaque aperture (aluminium stop) / mirror
            # substrate -> not a refracting surface the sequential beam passes
            continue
        bbox = b["bbox_m"]
        faces = []
        for f in b["faces"]:
            kind = _face_optical_kind(f)
            if kind is None:
                continue          # cylinder barrel rim etc. -- not an axial face
            if kind == "sphere":
                vx, R = _sphere_vertex_and_radius(f, bbox)
                faces.append((vx, R, False))
            else:
                faces.append((_plane_vertex(f), None, True))
        if len(faces) != 2:
            raise BridgeUnsupported(
                "optic %r has %d axial refracting faces (need exactly 2 for a "
                "sequential singlet element)" % (b["label"], len(faces)))
        faces.sort(key=lambda t: t[0])        # entry (small x) then exit
        (v0, R0, flat0), (v1, R1, flat1) = faces
        surfaces.append(OpticalSurface(v0, R0, flat0, b["label"], mat, True))
        surfaces.append(OpticalSurface(v1, R1, flat1, b["label"], ambient, False))

    surfaces.sort(key=lambda s: s.vertex_x_m)
    if not surfaces:
        raise BridgeUnsupported("no refracting glass surfaces found")
    return SequentialSystem(surfaces, det_z, wavelength_nm, ambient,
                            geometry_dir.name, det["label"])


# --------------------------------------------------------------------------
# SequentialSystem -> Optiland Optic
# --------------------------------------------------------------------------
def resolve_index(material_name, wavelength_nm, optprops_dir=None):
    """Real refractive index of ANY medium (glass OR ambient air) from the
    MieWorkbench materials registry at the test wavelength -- the same registry,
    and the same real-air ambient, the engine uses."""
    return _medium_nk(material_name, wavelength_nm, optprops_dir)[0]


def build_optic(system, epd_mm, optprops_dir=None):
    """SequentialSystem -> configured Optiland Optic (object at infinity,
    on-axis field, monochromatic at the system wavelength). epd_mm sets the
    entrance-pupil diameter used to normalize pupil coordinates."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # 0.6.0->0.7 API deprecations
        from optiland.optic import Optic
        from optiland.materials import IdealMaterial

        # Ambient index: the engine traces in the scene's real ambient (the
        # registry's "air" is n=1.000272 at 633 nm, NOT vacuum). Using it here
        # instead of Optiland's default n=1.0 "air" string is what closes the
        # ~4e-4-in-index / few-um-in-landing gap against the C engine -- an
        # ADJUDICATED modeling match, not a fudge (see the test docstring).
        n_amb = resolve_index(system.ambient, system.wavelength_nm, optprops_dir)
        amb_material = IdealMaterial(n_amb)

        opt = Optic()
        # object at infinity; surface-0 material = the medium up to surface 1
        opt.add_surface(index=0, thickness=np.inf, material=amb_material)
        n_surf = len(system.surfaces)
        for i, s in enumerate(system.surfaces):
            radius = np.inf if s.is_flat else s.radius_m * MM_PER_M
            nxt_z = (system.surfaces[i + 1].vertex_x_m if i + 1 < n_surf
                     else system.image_z_m)
            thickness = (nxt_z - s.vertex_x_m) * MM_PER_M
            if _is_glass(s.material_after):
                material = IdealMaterial(resolve_index(
                    s.material_after, system.wavelength_nm, optprops_dir))
            else:
                material = amb_material
            opt.add_surface(index=i + 1, radius=radius, thickness=thickness,
                            material=material, is_stop=(i == 0))
        opt.add_surface(index=n_surf + 1)                   # image plane
        opt.set_aperture(aperture_type="EPD", value=epd_mm)
        opt.set_field_type(field_type="angle")
        opt.add_field(y=0.0)
        opt.add_wavelength(system.wavelength_nm * 1e-3, is_primary=True)  # um
    return opt


def trace_collimated_fan(opt, pupil_radii_mm, epd_mm, wavelength_um):
    """Trace on-axis, axis-parallel rays at the given entrance-pupil RADII
    (mm). Returns (r_land_mm, radial_slope) at the image surface, aligned to
    pupil_radii_mm. radial_slope = d(r)/d(z) of each ray leaving the image
    plane (for analytic best-focus). Sign of r follows +Py."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Py = np.asarray(pupil_radii_mm, float) / (epd_mm / 2.0)
        Px = np.zeros_like(Py)
        opt.trace_generic(Hx=0.0, Hy=0.0, Px=Px, Py=Py, wavelength=wavelength_um)
        sg = opt.surface_group
        r = np.asarray(sg.y[-1], float)               # image-plane height (mm)
        M = np.asarray(sg.M[-1], float)               # dy direction component
        N = np.asarray(sg.N[-1], float)               # dz (axial) component
    return r, M / N


def trace_pupil_world(opt, pupil_y_m, pupil_z_m, epd_mm, wavelength_um,
                      image_z_m):
    """Trace on-axis, axis-parallel rays entering at the given WORLD transverse
    pupil positions (metres; world axis is +x). Returns image-plane (pos, dir)
    as (N,3) arrays in the WORLD frame (metres / unit dir), directly
    comparable to the MC export's `pos`/`dir`. Frame map (right-handed):
    Optiland (x_o, y_o, z_o=axis) <-> world (y, z, x); pupil Px<-world y,
    Py<-world z (verified by the oracle's per-ray residual: the alternate map
    is off by ~2x the spot radius)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        half = epd_mm / 2.0
        Px = np.asarray(pupil_y_m, float) * MM_PER_M / half
        Py = np.asarray(pupil_z_m, float) * MM_PER_M / half
        opt.trace_generic(Hx=0.0, Hy=0.0, Px=Px, Py=Py, wavelength=wavelength_um)
        sg = opt.surface_group
        xo = np.asarray(sg.x[-1], float) / MM_PER_M   # -> world y (m)
        yo = np.asarray(sg.y[-1], float) / MM_PER_M   # -> world z (m)
        L = np.asarray(sg.L[-1], float)               # -> world dir y
        M = np.asarray(sg.M[-1], float)               # -> world dir z
        N = np.asarray(sg.N[-1], float)               # -> world dir x (axis)
    n = len(xo)
    pos = np.column_stack([np.full(n, image_z_m), xo, yo])
    dirn = np.column_stack([N, L, M])
    dirn /= np.linalg.norm(dirn, axis=1, keepdims=True)
    return pos, dirn


def best_focus_shift_mm(r_mm, slope_per_z):
    """Axial shift s (mm, +downstream) from a reference plane MINIMIZING the
    transverse RMS of a rotationally symmetric bundle. r(s) = r + s*slope;
    minimize mean(r(s)^2) -> s = -sum(r*slope)/sum(slope^2). Returns (s, ok):
    ok=False when the bundle is ~collimated (slope ~ 0, focus at infinity)."""
    r = np.asarray(r_mm, float)
    v = np.asarray(slope_per_z, float)
    denom = float(np.sum(v * v))
    if denom < 1e-18 or float(np.sqrt(np.mean(v * v))) < 1e-9:
        return 0.0, False
    return -float(np.sum(r * v)) / denom, True
