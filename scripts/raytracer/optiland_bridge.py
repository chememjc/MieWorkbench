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
    """One optical surface on the common axis, in SI metres (world +x).

    The oracle constructs sphere/plane refracting surfaces with the original
    six positional args; P4b adds keyword fields (defaulting to the oracle's
    behaviour) so ONE class carries conics/aspheres (surface_kind='asphere'
    with conic k + SI even-polynomial coeffs + validity r_max), reflective
    surfaces (is_mirror) and hard aperture stops (is_stop with a physical
    clear semi_diameter). Unset keywords reproduce the P4a sphere/plane
    refracting surface exactly, so the parity oracle is untouched."""

    def __init__(self, vertex_x_m, radius_m, is_flat, body_label,
                 material_after, is_entry, surface_kind=None, conic=0.0,
                 coeffs_si=None, r_max_m=None, is_mirror=False,
                 is_stop=False, semi_diameter_m=None):
        self.vertex_x_m = vertex_x_m          # axial intercept, world metres
        self.radius_m = radius_m              # signed: +ve if centre is at +x
        self.is_flat = is_flat                # plane -> infinite radius
        self.body_label = body_label
        self.material_after = material_after   # glass name or "air" downstream
        self.is_entry = is_entry              # entry (glass side) vs exit (air)
        # sphere/plane keep surface_kind None (== the oracle's implicit kind).
        self.surface_kind = surface_kind      # 'asphere' | 'conic' | 'stop' | None
        self.conic = float(conic)             # conic constant k (0 for a sphere)
        self.coeffs_si = list(coeffs_si or [])  # even-poly A4,A6,... SI (m^-(3+2i))
        self.r_max_m = r_max_m                # asphere validity radius, metres
        self.is_mirror = bool(is_mirror)      # reflective surface (material=mirror)
        self.is_stop = bool(is_stop)          # hard aperture stop (physical clip)
        self.semi_diameter_m = semi_diameter_m  # clear semi-aperture, metres


class SequentialSystem:
    """The bridge's engine-neutral intermediate form."""

    def __init__(self, surfaces, image_z_m, wavelength_nm, ambient,
                 stem, det_label, source_semidiameter_m=None,
                 stop=None, n_mirrors=0):
        self.surfaces = surfaces              # z-ordered list of OpticalSurface
        self.image_z_m = image_z_m            # detector plane, world metres
        self.wavelength_nm = wavelength_nm
        self.ambient = ambient                # ambient material name
        self.stem = stem
        self.det_label = det_label
        # entrance-beam semi-diameter from the source body (the collimated
        # object-space pupil radius the MC engine fills), metres.
        self.source_semidiameter_m = source_semidiameter_m
        # {'z_m':..., 'semidiameter_m':..., 'body_label':...} when a real
        # aperture stop was modelled (model_stop=True); None otherwise.
        self.stop = stop
        self.n_mirrors = int(n_mirrors)       # reflective surfaces in the train


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


def _asphere_vertex_and_params(face, bbox):
    """Asphere/conic axial vertex (world x) and SIGNED base radius plus its
    conic constant, SI even-polynomial coeffs and validity r_max (all metres).
    Sign of R follows the sphere convention (+ve when the base centre of
    curvature is downstream, at greater x). The extractor emits the vertex on
    the revolution axis and an axis along +/-x for an on-axis element; a
    non-axial asphere axis is a hard BridgeUnsupported (off-axis segment)."""
    surf = face["surface"]
    axis = np.asarray(surf["axis"], float)
    if abs(abs(axis[0]) - 1.0) > 1e-6:
        raise BridgeUnsupported(
            "asphere axis %s is not along the optical axis (+x) -- off-axis "
            "or tilted asphere segment" % (axis.tolist(),))
    vertex = float(surf["vertex"][0])
    lo, hi = bbox["min"][0] - _AXIS_TOL_M, bbox["max"][0] + _AXIS_TOL_M
    if not (lo <= vertex <= hi):
        raise BridgeUnsupported(
            "asphere vertex %.6g m outside body bbox [%.6g, %.6g]"
            % (vertex, lo, hi))
    R = float(surf["R"])
    # Signed R (Optiland): +ve when the base sphere's centre (vertex + R*axis,
    # canonical +axis component) lies downstream of the vertex. The extractor's
    # canonical axis points into the sag half-space; the physical centre sits
    # at vertex + R * (+x-ward axis). Reduce to the same rule the sphere path
    # uses by projecting the axis onto +x.
    signed_R = R if axis[0] > 0 else -R
    k = float(surf.get("k", 0.0))
    coeffs = list(surf.get("coeffs", []) or [])
    r_max = surf.get("r_max")
    r_max = float(r_max) if r_max is not None else None
    return vertex, signed_R, k, coeffs, r_max


def _optical_faces(body):
    """The body's axial refracting/reflecting faces as
    (vertex_x_m, kind, payload) tuples, sorted by vertex x. kind is
    'sphere'|'asphere'|'plane'; payload carries the per-kind geometry.
    Raises BridgeUnsupported on a qforbes optical face (Optiland 0.6.0 has
    forbes_qbfs/forbes_q2d geometries, but the MieWorkbench Q-bfs/Q-con
    normalisation -- sigma_inv cos-factor, r/r_max scaling, Jacobi(0,4)
    recursion -- is not verified to match Optiland's convention; a validated
    Forbes mapping is a later tranche)."""
    bbox = body["bbox_m"]
    faces = []
    for f in body["faces"]:
        t = (f.get("surface") or {}).get("type")
        if t == "qforbes":
            raise BridgeUnsupported(
                "body %r carries a Q-Forbes optical face; the MieWorkbench "
                "Q-bfs/Q-con normalisation is not yet verified against "
                "Optiland's forbes geometries (later tranche)"
                % body.get("label"))
        if t == "sphere":
            vx, R = _sphere_vertex_and_radius(f, bbox)
            faces.append((vx, "sphere", {"radius_m": R}))
        elif t == "asphere":
            vx, R, k, coeffs, r_max = _asphere_vertex_and_params(f, bbox)
            faces.append((vx, "asphere",
                          {"radius_m": R, "conic": k, "coeffs_si": coeffs,
                           "r_max_m": r_max}))
        elif t == "plane":
            faces.append((_plane_vertex(f), "plane", {}))
        # cylinder / cone / mesh barrels are not axial optical faces -> skip
    faces.sort(key=lambda t: t[0])
    return faces


def _is_glass(name):
    """A body material that could be a refracting LENS element -- named, and
    neither ambient air nor vacuum/none (an air-filled iris plug is not a
    lens)."""
    return bool(name) and str(name).lower() not in ("air", "none", "vacuum")


# absorptive-index threshold above which a body is treated as opaque (a stop /
# iris / mirror substrate), NOT a refracting lens. bk7/sf5 have k==0; aluminium
# and other metals have k of order 1-10, so this cleanly separates the two.
_OPAQUE_K = 1e-6


# The materials registry load is ~1 s; cache the matdb per root so a merit
# loop (many build_optic calls) resolves indices in microseconds. P4b turns
# this bridge from an oracle (built once) into an evaluator (built per eval).
_MATDB_CACHE = {}


def _matdb(optprops_dir=None):
    root = str(optprops_dir or (Path(__file__).resolve().parents[2]
                                / "opticalproperties"))
    mdb = _MATDB_CACHE.get(root)
    if mdb is None:
        from raytracer import optprops              # engine-env-safe (numpy)
        mdb = optprops.load_optical_properties(root).matdb
        _MATDB_CACHE[root] = mdb
    return mdb


def _medium_nk(name, wavelength_nm, optprops_dir=None):
    """(n, k) of ANY named medium at the wavelength, from the engine's OWN
    registry -- INCLUDING air (n=1.000272 at 633 nm, the real ambient the C
    engine traces in). Only vacuum/none/None (and names absent from the
    registry) fall back to (1.0, 0.0)."""
    if not name or str(name).lower() in ("none", "vacuum"):
        return 1.0, 0.0
    try:
        nc = _matdb(optprops_dir).get(name).n_complex(wavelength_nm * 1e-9)
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


def _transverse_semidiameter_m(bbox):
    """The larger transverse (y,z) half-extent of a body bbox, metres -- the
    clear beam radius of a round element (y==z half-extents)."""
    hy = 0.5 * (bbox["max"][1] - bbox["min"][1])
    hz = 0.5 * (bbox["max"][2] - bbox["min"][2])
    return max(hy, hz)


def _stop_from_opaque_body(body):
    """Detect an aperture-stop annulus: an opaque body with a central bore
    (an inner cylinder face) and NO curved optical face. Returns
    {'z_m','semidiameter_m','body_label'} (clear semi-diameter = the smallest
    cylinder radius = the bore) or None if the body is not a plain stop."""
    cyl_radii = []
    has_curved_optic = False
    planes_x = []
    for f in body["faces"]:
        s = f.get("surface") or {}
        t = s.get("type")
        if t == "cylinder":
            cyl_radii.append(float(s["radius"]))
        elif t in ("sphere", "asphere", "qforbes"):
            has_curved_optic = True
        elif t == "plane":
            planes_x.append(float(s["origin"][0]))
    if has_curved_optic or not cyl_radii:
        return None
    bore = min(cyl_radii)
    bbox = body["bbox_m"]
    z_m = 0.5 * (bbox["min"][0] + bbox["max"][0])   # axial centre of the annulus
    return {"z_m": z_m, "semidiameter_m": bore, "body_label": body["label"]}


def load_sequential_system(geometry_dir, wavelength_nm=None, model_stop=False):
    """model.json -> SequentialSystem. wavelength_nm defaults to the scene
    source's lambdac. Raises BridgeUnsupported on out-of-scope features.

    P4b scope beyond the P4a oracle: rotationally-symmetric conics/aspheres
    (mapped to Optiland even-asphere/standard geometries), fold-free on-axis
    reflective surfaces (one reflection; multi-mirror double-pass ordering
    needs the train recipe and stays unsupported), and -- when model_stop is
    True -- a real aperture stop from an opaque iris annulus (its clear bore
    becomes an Optiland float-by-stop aperture). The oracle path
    (model_stop=False) still DROPS the stop, exactly as P4a documented, so the
    parity suite is unchanged. Chained/tilted/folded trains stay unsupported:
    every off-axis surface (plane normal / sphere vertex / asphere axis) is a
    hard BridgeUnsupported, never a silent wrong answer."""
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
    source_semidiameter_m = _transverse_semidiameter_m(sources[0]["bbox_m"])

    det = detectors[0]
    det_face_id = det["detector"]["face"]
    det_z = None
    for f in det["faces"]:
        if f["id"] == det_face_id:
            det_z = _plane_vertex(f)
    if det_z is None:
        raise BridgeUnsupported("detector primary face %s not found" % det_face_id)

    surfaces = []
    stop = None
    n_mirrors = 0
    for b in optics:
        mat = b.get("material")
        transparent = _is_transparent_dielectric(mat, wavelength_nm)
        faces = _optical_faces(b)
        curved = [f for f in faces if f[1] in ("sphere", "asphere")]

        if not transparent:
            # opaque body: a curved metal face is a MIRROR; a bored flat
            # annulus is an aperture stop; a plain air plug is inert.
            if curved:
                if len(curved) != 1:
                    raise BridgeUnsupported(
                        "reflective body %r has %d curved faces (need exactly "
                        "one mirror surface)" % (b["label"], len(curved)))
                vx, kind, pl = curved[0]
                surfaces.append(_mirror_surface(vx, kind, pl, b["label"],
                                                ambient))
                n_mirrors += 1
                continue
            # a stop is an OPAQUE (absorbing metal) bored annulus; an air plug
            # or vacuum fill (k~0) is inert and never a stop.
            is_opaque = _medium_nk(mat, wavelength_nm)[1] > _OPAQUE_K
            if model_stop and is_opaque:
                cand = _stop_from_opaque_body(b)
                if cand is not None:
                    if stop is not None:
                        raise BridgeUnsupported(
                            "scene has more than one aperture stop; the "
                            "sequential bridge models a single stop")
                    stop = cand
            continue        # air plug / iris fill / dropped stop -> no surface

        # transparent dielectric element (refracting singlet: exactly 2 faces)
        if len(faces) != 2:
            raise BridgeUnsupported(
                "optic %r has %d axial refracting faces (need exactly 2 for a "
                "sequential singlet element)" % (b["label"], len(faces)))
        (v0, k0, p0), (v1, k1, p1) = faces
        surfaces.append(_refracting_surface(v0, k0, p0, b["label"], mat, True))
        surfaces.append(_refracting_surface(v1, k1, p1, b["label"], ambient,
                                            False))

    if n_mirrors > 1:
        raise BridgeUnsupported(
            "scene has %d reflective surfaces; multi-mirror double-pass "
            "sequential ordering needs the train recipe (later tranche)"
            % n_mirrors)

    surfaces.sort(key=lambda s: s.vertex_x_m)
    if stop is not None:
        surfaces.append(OpticalSurface(
            stop["z_m"], None, True, stop["body_label"], ambient, False,
            surface_kind="stop", is_stop=True,
            semi_diameter_m=stop["semidiameter_m"]))
        surfaces.sort(key=lambda s: s.vertex_x_m)
    if not surfaces:
        raise BridgeUnsupported("no refracting glass surfaces found")
    return SequentialSystem(surfaces, det_z, wavelength_nm, ambient,
                            geometry_dir.name, det["label"],
                            source_semidiameter_m=source_semidiameter_m,
                            stop=stop, n_mirrors=n_mirrors)


def _refracting_surface(vertex_x, kind, payload, body_label, material_after,
                        is_entry):
    """Build an OpticalSurface for a refracting sphere/plane/asphere face."""
    if kind == "plane":
        return OpticalSurface(vertex_x, None, True, body_label,
                              material_after, is_entry, surface_kind="plane")
    if kind == "sphere":
        return OpticalSurface(vertex_x, payload["radius_m"], False, body_label,
                              material_after, is_entry, surface_kind="sphere")
    # asphere / conic
    kk = "conic" if not payload["coeffs_si"] else "asphere"
    return OpticalSurface(
        vertex_x, payload["radius_m"], False, body_label, material_after,
        is_entry, surface_kind=kk, conic=payload["conic"],
        coeffs_si=payload["coeffs_si"], r_max_m=payload["r_max_m"])


def _mirror_surface(vertex_x, kind, payload, body_label, ambient):
    """Build a reflective OpticalSurface (material after reflection = ambient;
    the ray reverses but stays in the ambient medium)."""
    if kind == "sphere":
        return OpticalSurface(vertex_x, payload["radius_m"], False, body_label,
                              ambient, False, surface_kind="sphere",
                              is_mirror=True)
    kk = "conic" if not payload["coeffs_si"] else "asphere"
    return OpticalSurface(
        vertex_x, payload["radius_m"], False, body_label, ambient, False,
        surface_kind=kk, conic=payload["conic"],
        coeffs_si=payload["coeffs_si"], r_max_m=payload["r_max_m"],
        is_mirror=True)


# --------------------------------------------------------------------------
# SequentialSystem -> Optiland Optic
# --------------------------------------------------------------------------
def resolve_index(material_name, wavelength_nm, optprops_dir=None):
    """Real refractive index of ANY medium (glass OR ambient air) from the
    MieWorkbench materials registry at the test wavelength -- the same registry,
    and the same real-air ambient, the engine uses."""
    return _medium_nk(material_name, wavelength_nm, optprops_dir)[0]


def _coeffs_si_to_mm(coeffs_si):
    """Even-asphere coeffs SI (A_m multiplies r_m^(4+2i), units m^-(3+2i)) ->
    Optiland even-asphere coefficient list in millimetres. Sag_mm = sag_m *
    MM_PER_M and r_mm = r_m * MM_PER_M give A_mm = A_m * MM_PER_M^(1-(4+2i)).
    Optiland's EvenAsphere indexes coefficients[i] onto r^(2(i+1)) (its [0] is
    the r^2 term), so a leading 0.0 aligns MieWorkbench's A4,A6,... (which
    start at r^4) with Optiland's r^4,r^6,... slots."""
    out = [0.0]
    for i, a in enumerate(coeffs_si):
        m = 4 + 2 * i
        out.append(float(a) * (MM_PER_M ** (1 - m)))
    return out


def _add_optical_surface(opt, index, s, radius, thickness, material,
                         is_stop, mirror_material):
    """Add one surface, dispatching on its kind (standard sphere/plane/conic
    vs even-asphere). Reflective surfaces use the shared mirror material.
    Physical aperture stops carry a RadialAperture so vignetting is real."""
    mat = mirror_material if s.is_mirror else material
    kwargs = dict(index=index, radius=radius, thickness=thickness,
                  material=mat, is_stop=is_stop)
    if s.is_stop and s.semi_diameter_m is not None:
        from optiland.physical_apertures import RadialAperture
        kwargs["aperture"] = RadialAperture(r_max=s.semi_diameter_m * MM_PER_M)
    if s.surface_kind == "asphere" and s.coeffs_si:
        opt.add_surface(surface_type="even_asphere", conic=s.conic,
                        coefficients=_coeffs_si_to_mm(s.coeffs_si), **kwargs)
    elif s.surface_kind == "conic" or (s.conic and not s.coeffs_si):
        opt.add_surface(surface_type="standard", conic=s.conic, **kwargs)
    else:
        opt.add_surface(surface_type="standard", **kwargs)


def build_optic(system, epd_mm=None, optprops_dir=None):
    """SequentialSystem -> configured Optiland Optic (object at infinity,
    on-axis field, monochromatic at the system wavelength).

    Aperture semantics (the entrance-pupil / stop upgrade for P4b):
      * a modelled aperture stop (system.stop, from model_stop=True) becomes
        an Optiland float-by-stop aperture -- Optiland derives the entrance
        pupil by paraxial-tracing the stop's clear bore, exactly the stop the
        MC engine's iris clips to;
      * else if epd_mm is given (the P4a oracle path) the EPD is set verbatim
        and surface 1 is the stop -- UNCHANGED, so the parity oracle is
        untouched;
      * else the entrance pupil is the source's own clear aperture
        (2 * source_semidiameter), the collimated object-space beam the MC
        run fills.
    After a reflective surface the propagation reverses, so downstream signed
    thicknesses flip -- handled for a single on-axis mirror (multi-mirror
    scenes are rejected upstream in load_sequential_system)."""
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
        mirror_material = "mirror"      # Optiland's reflective-material spec

        opt = Optic()
        # object at infinity; surface-0 material = the medium up to surface 1
        opt.add_surface(index=0, thickness=np.inf, material=amb_material)
        n_surf = len(system.surfaces)
        has_stop = system.stop is not None
        direction = 1.0        # +1 downstream; flips to -1 after a mirror
        for i, s in enumerate(system.surfaces):
            radius = np.inf if s.is_flat else s.radius_m * MM_PER_M
            nxt_z = (system.surfaces[i + 1].vertex_x_m if i + 1 < n_surf
                     else system.image_z_m)
            thickness = direction * (nxt_z - s.vertex_x_m) * MM_PER_M
            if _is_glass(s.material_after):
                material = IdealMaterial(resolve_index(
                    s.material_after, system.wavelength_nm, optprops_dir))
            else:
                material = amb_material
            # the system stop: the modelled hard stop if present, else the
            # first surface (the oracle / source-aperture convention).
            is_stop = s.is_stop if has_stop else (i == 0)
            _add_optical_surface(opt, i + 1, s, radius, thickness, material,
                                 is_stop, mirror_material)
            if s.is_mirror:
                direction = -direction
        opt.add_surface(index=n_surf + 1)                   # image plane

        if has_stop:
            opt.set_aperture(aperture_type="float_by_stop_size",
                             value=2.0 * system.stop["semidiameter_m"] * MM_PER_M)
        elif epd_mm is not None:
            opt.set_aperture(aperture_type="EPD", value=epd_mm)
        else:
            if not system.source_semidiameter_m:
                raise BridgeUnsupported(
                    "no aperture stop, no epd_mm and no source aperture -- "
                    "cannot define the entrance pupil")
            opt.set_aperture(aperture_type="EPD",
                             value=2.0 * system.source_semidiameter_m * MM_PER_M)
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


def trace_pupil_world_path(opt, pupil_y_m, pupil_z_m, epd_mm, wavelength_um,
                           source_pos_m):
    """P4b preview-unification (engine3.md Sec.5/Sec.15 P4b: "preview
    unified"): like trace_pupil_world, but returns the FULL per-surface
    world-frame polyline each ray follows -- source plane -> every optical
    surface intercept (sag-corrected, not the paraxial vertex) -> the image
    plane -- for the GUI's ray-overlay visualization, not just the final
    landing point.

    source_pos_m: (N,3) world-frame ray origins (the real emit-face points,
    e.g. from raytracer.sources.sample_viz_pattern) -- used verbatim as each
    path's first vertex. Valid because the object bundle is COLLIMATED and
    on-axis (the caller's job to verify): a straight line in a homogeneous
    medium has the SAME transverse (y, z) position at the source plane, the
    entrance pupil, and any other plane before the first optical surface,
    so pupil_y_m/pupil_z_m (used to build the normalized Optiland Px/Py,
    exactly trace_pupil_world's convention) may be taken directly from
    source_pos_m's own y/z columns.

    Returns (n_rays, n_surfaces+1, 3) world-frame points (n_surfaces = the
    system's optical surfaces, +1 for the image plane; the source point is
    prepended, giving n_surfaces+2 vertices total -- n_surfaces+1 SEGMENTS
    per ray) and a boolean (n_rays,) validity mask: False for any ray whose
    path contains a non-finite vertex (TIR'd/vignetted/missed a surface --
    the physical stop's RadialAperture clips these exactly as the real
    iris would).

    Frame map is trace_pupil_world's (verified there): Optiland
    (x_o, y_o, z_o=axis) <-> world (y, z, x=axial). Unlike trace_pupil_world
    (which only reads the LAST surface, index -1), this reads sg.x/y/z's
    full per-surface stack and skips index 0 (Optiland's object surface --
    a bookkeeping placeholder at infinity, never a real vertex).

    Rays whose normalized pupil coordinate falls OUTSIDE the unit disc
    (beyond the entrance pupil this `opt` was built with -- e.g. a fan
    sampled over the source body's full clear aperture when a downstream
    iris models a materially smaller stop) are never handed to
    `trace_generic` (which hard-rejects |P|>1 with a ValueError): they are
    real vignetting, reported as `valid=False` exactly like a ray that
    TIR'd or missed a surface -- a physical clip, not an error."""
    src = np.asarray(source_pos_m, float)
    n_rays = src.shape[0]
    half = epd_mm / 2.0
    Px_full = np.asarray(pupil_y_m, float) * MM_PER_M / half
    Py_full = np.asarray(pupil_z_m, float) * MM_PER_M / half
    in_pupil = (Px_full * Px_full + Py_full * Py_full) <= 1.0
    idx = np.where(in_pupil)[0]
    valid = np.zeros(n_rays, dtype=bool)
    if idx.size == 0:
        # every ray fell outside this system's entrance pupil -- nothing to
        # trace; shape (n_rays, 1, 3) is a degenerate (single source-point,
        # no segments) path so callers' shape expectations still hold.
        return src[:, None, :], valid

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Px, Py = Px_full[idx], Py_full[idx]
        opt.trace_generic(Hx=0.0, Hy=0.0, Px=Px, Py=Py, wavelength=wavelength_um)
        sg = opt.surface_group
        # sg.x/y/z stack every RECORDED surface (object..image); the object
        # surface (index 0) carries no physical vertex -- drop it.
        world_y = np.asarray(sg.x[1:], float) / MM_PER_M     # (n_surf, n_in)
        world_z = np.asarray(sg.y[1:], float) / MM_PER_M
        world_x = np.asarray(sg.z[1:], float) / MM_PER_M

    n_surf = world_x.shape[0]
    traced = np.stack([world_x, world_y, world_z], axis=-1)  # (n_surf,n_in,3)
    traced = np.transpose(traced, (1, 0, 2))                 # (n_in,n_surf,3)

    out = np.full((n_rays, n_surf + 1, 3), np.nan)
    out[:, 0, :] = src
    out[idx, 1:, :] = traced
    valid[idx] = np.all(np.isfinite(traced.reshape(idx.size, -1)), axis=1)
    return out, valid


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


# --------------------------------------------------------------------------
# P4b sequential MERIT evaluation -- deterministic, noise-free operands on
# the Optiland trace (spot RMS, best-focus shift, geometric encircled-energy).
# This is the "evaluate operands DIRECTLY in Optiland" path (engine3.md Sec 8);
# the operand->backend routing lives in scripts/optimize.py's catalog.
# --------------------------------------------------------------------------
def sample_unit_disc(n, seed=0):
    """n area-uniform samples of the unit pupil disc as normalized (Px, Py) in
    [-1,1]. Uniform random (r=sqrt(U)) reproduces the MC engine's own uniform
    aperture sampling distribution, so a spot RMS computed over this bundle is
    directly comparable to the MC report's (the arbiter's shared-sampling
    principle, optiland_oracle_support.py)."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(n))
    th = 2.0 * np.pi * rng.random(n)
    return r * np.cos(th), r * np.sin(th)


def _trace_image_xy_mm(opt, wavelength_um, Px, Py):
    """Trace the on-axis (field 0) pupil bundle; return image-surface (x,y) in
    mm and the axial direction cosine N (for best-focus slopes), NaN-cleaned to
    the rays that actually reached the image (vignetted rays are dropped so a
    physical stop's clip is honoured)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt.trace_generic(Hx=0.0, Hy=0.0, Px=np.asarray(Px, float),
                          Py=np.asarray(Py, float), wavelength=wavelength_um)
        sg = opt.surface_group
        x = np.asarray(sg.x[-1], float)
        y = np.asarray(sg.y[-1], float)
    valid = np.isfinite(x) & np.isfinite(y)
    return x, y, valid


def spot_rms_mm(x_mm, y_mm):
    """RMS radial spot size (mm) about the transverse centroid."""
    x = np.asarray(x_mm, float)
    y = np.asarray(y_mm, float)
    xc, yc = x.mean(), y.mean()
    return float(np.sqrt(np.mean((x - xc) ** 2 + (y - yc) ** 2)))


def encircled_radius_mm(x_mm, y_mm, frac=0.8):
    """Geometric encircled-energy radius (mm): the radius about the centroid
    containing `frac` of the (equal-weight) rays. A GEOMETRIC proxy for the
    diffraction ee_r80 -- honest as a ray-density measure, NOT a PSF integral
    (documented in the operand routing table)."""
    x = np.asarray(x_mm, float)
    y = np.asarray(y_mm, float)
    r = np.sqrt((x - x.mean()) ** 2 + (y - y.mean()) ** 2)
    r.sort()
    if r.size == 0:
        return float("nan")
    k = min(r.size - 1, max(0, int(np.ceil(frac * r.size)) - 1))
    return float(r[k])


def evaluate_geometry(system, n_rays=4096, seed=0, ee_frac=0.8,
                      optprops_dir=None):
    """Deterministic sequential merit block for ONE detector: builds the
    Optiland optic, traces an area-uniform on-axis pupil bundle, and returns
    spot RMS (m), the best-focus axial shift from the detector (m, two-plane
    convention-free) and a geometric encircled-energy radius (m). Purely
    geometric -- microseconds-class, noise-free, differentiable. Raises
    BridgeUnsupported (via build_optic/load) for out-of-scope scenes."""
    lam_um = system.wavelength_nm * 1e-3
    Px, Py = sample_unit_disc(n_rays, seed=seed)
    opt = build_optic(system, optprops_dir=optprops_dir)
    x0, y0, v0 = _trace_image_xy_mm(opt, lam_um, Px, Py)

    # best focus from POSITIONS at two image planes dz apart (the oracle's
    # adjudicated convention-free route; Optiland's reported image direction
    # cosines carry an ambient-index slope artifact).
    dz_mm = 1.0
    sys2 = SequentialSystem(system.surfaces, system.image_z_m + dz_mm / MM_PER_M,
                            system.wavelength_nm, system.ambient, system.stem,
                            system.det_label,
                            source_semidiameter_m=system.source_semidiameter_m,
                            stop=system.stop, n_mirrors=system.n_mirrors)
    opt2 = build_optic(sys2, optprops_dir=optprops_dir)
    x1, y1, v1 = _trace_image_xy_mm(opt2, lam_um, Px, Py)

    valid = v0 & v1
    x0, y0, x1, y1 = x0[valid], y0[valid], x1[valid], y1[valid]
    if x0.size < 8:
        raise BridgeUnsupported(
            "sequential trace produced < 8 unvignetted rays (%d) -- the pupil "
            "is fully clipped or the scene is degenerate" % int(x0.size))

    spot_m = spot_rms_mm(x0, y0) / MM_PER_M
    ee_m = encircled_radius_mm(x0, y0, frac=ee_frac) / MM_PER_M

    vx = (x1 - x0) / dz_mm
    vy = (y1 - y0) / dz_mm
    denom = float(np.sum(vx * vx + vy * vy))
    if denom < 1e-18 or float(np.sqrt(np.mean(vx * vx + vy * vy))) < 1e-9:
        focus_shift_m, afocal = 0.0, True     # collimated output (afocal)
    else:
        s_mm = -float(np.sum(x0 * vx + y0 * vy)) / denom
        focus_shift_m, afocal = s_mm / MM_PER_M, False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # Optiland paraxial np-scalar
        f2_mm = float(opt.paraxial.f2())
        epd_mm = float(opt.paraxial.EPD())
    return {
        "detector": system.det_label,
        "wavelength_nm": system.wavelength_nm,
        "n_rays": int(x0.size),
        "spot_rms_m": spot_m,
        "focus_shift_m": focus_shift_m,
        "afocal": afocal,
        "ee_radius_m": ee_m,
        "ee_frac": ee_frac,
        "paraxial_f2_mm": f2_mm,
        "epd_mm": epd_mm,
    }
