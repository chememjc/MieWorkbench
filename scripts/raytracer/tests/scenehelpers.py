# =============================================================================
# scenehelpers.py — build synthetic model.json dicts (boxes on the x-axis)
# so physics tests can drive the REAL Scene/Tracer pipeline without FreeCAD.
#
# Geometry convention: everything on the x-axis, beam travels +x from a
# square emitting plane at x_src < 0 toward a detector plane at x_det > 0.
# Boxes are axis-aligned: [x0, x1] x [-half, half]^2 with 6 plane faces,
# canonical normals OUTWARD (orientation_outward = True).
# =============================================================================
import numpy as np


def _rect_face(fid, origin, normal, corners, area):
    return {
        "id": fid,
        "surface": {"type": "plane", "origin": list(origin),
                    "normal": list(normal)},
        "orientation_outward": True,
        "area_m2": float(area),
        "fingerprint": {},
        "mesh_stl": "",
        "trim_polylines_xyz": [[list(c) for c in corners]],
    }


def box_faces(name, x0, x1, half):
    """6 outward-normal plane faces of [x0,x1] x [-half,half]^2."""
    h = half
    side_area = (x1 - x0) * 2 * h
    end_area = (2 * h) ** 2
    return [
        _rect_face("%s.Pad.Face1" % name, [x0, 0, 0], [-1, 0, 0],
                   [[x0, -h, -h], [x0, h, -h], [x0, h, h], [x0, -h, h]],
                   end_area),
        _rect_face("%s.Pad.Face2" % name, [x1, 0, 0], [1, 0, 0],
                   [[x1, -h, -h], [x1, h, -h], [x1, h, h], [x1, -h, h]],
                   end_area),
        _rect_face("%s.Pad.Face3" % name, [0, -h, 0], [0, -1, 0],
                   [[x0, -h, -h], [x1, -h, -h], [x1, -h, h], [x0, -h, h]],
                   side_area),
        _rect_face("%s.Pad.Face4" % name, [0, h, 0], [0, 1, 0],
                   [[x0, h, -h], [x1, h, -h], [x1, h, h], [x0, h, h]],
                   side_area),
        _rect_face("%s.Pad.Face5" % name, [0, 0, -h], [0, 0, -1],
                   [[x0, -h, -h], [x1, -h, -h], [x1, h, -h], [x0, h, -h]],
                   side_area),
        _rect_face("%s.Pad.Face6" % name, [0, 0, h], [0, 0, 1],
                   [[x0, -h, h], [x1, -h, h], [x1, h, h], [x0, h, h]],
                   side_area),
    ]


def source_body(name="Src", x=-0.02, half=0.001, power_mW=1.0,
                lambdac_nm=633.0, coherent=False, polarization=None,
                lambdamin_nm=None, lambdamax_nm=None, apodization=None,
                beam_waist_mm=None, m2=1.0, spectrum=None,
                pulse_energy_uJ=None, pulse_duration_ps=None,
                rep_rate_hz=None):
    """Source with a single square emitting plane at x (normal +x; the
    toward-origin policy sends rays along +x). apodization: already-parsed
    dict (common.parse_apodization_spec). beam_waist_mm: sets source.beam
    {waist_mm, m2} — half MUST be large enough that the waist's Gaussian
    tail doesn't need excessive rejection-sampling tries against the
    emitting face's physical aperture. pulse_energy_uJ/pulse_duration_ps/
    rep_rate_hz: pulsed-optics Phase P3 raw properties (raytracer.scene.
    _parse_pulse_source does the XOR/derivation) — pass power_mW=0.0 for
    a pulse_energy-only source (extract_geometry's "unset power" sentinel;
    None also works here since this helper builds the dict directly, but
    0.0 matches what a real extracted model.json contains)."""
    face = _rect_face("%s.Pad.Face1" % name, [x, 0, 0], [1, 0, 0],
                      [[x, -half, -half], [x, half, -half],
                       [x, half, half], [x, -half, half]],
                      (2 * half) ** 2)
    src = {"power_mW": power_mW, "lambdac_nm": lambdac_nm,
           "emit_face": face["id"], "coherent": coherent}
    if lambdamin_nm is not None:
        src["lambdamin_nm"] = lambdamin_nm
    if lambdamax_nm is not None:
        src["lambdamax_nm"] = lambdamax_nm
    if polarization is not None:
        src["polarization"] = polarization
    if apodization is not None:
        src["apodization"] = apodization
    if beam_waist_mm is not None:
        src["beam"] = {"waist_mm": beam_waist_mm, "m2": m2}
    if spectrum is not None:
        src["spectrum"] = spectrum
    if pulse_energy_uJ is not None:
        src["pulse_energy_uJ"] = pulse_energy_uJ
    if pulse_duration_ps is not None:
        src["pulse_duration_ps"] = pulse_duration_ps
    if rep_rate_hz is not None:
        src["rep_rate_hz"] = rep_rate_hz
    return {"name": name, "label": name, "role": "source",
            "source": src, "faces": [face]}


def slab_body(name, material, x0, x1, half=0.01, **extra):
    """Optic slab [x0,x1] with 6 analytic plane faces. extra: polarizer,
    polarizer_axis, filter, crystal_axis, coating, roughness_faces,
    grating, mirror, absorbance ... (contract keys, already-parsed forms)."""
    body = {"name": name, "label": name, "role": "optic",
            "material": material, "faces": box_faces(name, x0, x1, half)}
    body.update(extra)
    return body


def detector_body(name="Det", x=0.03, half=0.01):
    face = _rect_face("%s.Pad.Face1" % name, [x, 0, 0], [-1, 0, 0],
                      [[x, -half, -half], [x, half, -half],
                       [x, half, half], [x, -half, half]],
                      (2 * half) ** 2)
    return {"name": name, "label": name, "role": "detector",
            "detector": {"face": face["id"]}, "faces": [face]}


def make_model(bodies):
    return {"schema_version": 2, "source_fcstd": "synthetic",
            "spreadsheet": {}, "ambient_material": "air",
            "validation": {}, "bodies": bodies}


def trace_scene(model, rays=20000, n_lambda=1, seed=3, power_floor=1e-12,
                resolution=256, optprops=None, max_reflections=6):
    """Build Scene, run the tracer, return (result, grids, scene)."""
    import common
    from raytracer.scene import Scene
    from raytracer.sources import sample_source
    from raytracer.tracer import Tracer, TraceConfig
    from raytracer.detector import DetectorGrid
    from raytracer.optprops import load_optical_properties

    common.validate_model(model)
    if optprops is None:
        optprops = load_optical_properties()
    scene = Scene(model, optprops.matdb, optprops.coatings,
                  optprops=optprops)
    lam_lo = min(s["lambdac_nm"] for _, s in scene.sources) - 100.0
    lam_hi = max(s["lambdac_nm"] for _, s in scene.sources) + 100.0
    for _, s in scene.sources:
        if s.get("lambdamin_nm"):
            lam_lo = min(lam_lo, s["lambdamin_nm"] - 50.0)
        if s.get("lambdamax_nm"):
            lam_hi = max(lam_hi, s["lambdamax_nm"] + 50.0)
        lam_tab = s.get("_spectrum_lam_nm")     # tabulated spectrum: cover it
        if lam_tab is not None:
            lam_lo = min(lam_lo, float(np.min(lam_tab)) - 20.0)
            lam_hi = max(lam_hi, float(np.max(lam_tab)) + 20.0)
    grids = {fid: DetectorGrid(scene.faces[fid], resolution, 16,
                               (lam_lo * 1e-9, lam_hi * 1e-9),
                               label=scene.faces[fid].id)
             for fid in scene.detector_faces}
    cfg = TraceConfig(rays=rays, n_lambda=n_lambda, seed=seed,
                      power_floor=power_floor,
                      max_reflections=max_reflections)
    tracer = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(seed)
    batches = [sample_source(scene, scene.bodies[b], s, i, cfg.rays,
                             cfg.n_lambda, rng, ledger=tracer.ledger)
               for i, (b, s) in enumerate(scene.sources)]
    result = tracer.run(batches)
    return result, grids, scene
