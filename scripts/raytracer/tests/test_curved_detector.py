# =============================================================================
# test_curved_detector.py — Phase 10: curved (Sphere/Cylinder) detector grids,
# incoherent path. Exercises the REAL Scene/Tracer pipeline (no FreeCAD) with
# analytic sphere/cylinder detector faces built here.
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_curved_detector.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import common                                              # noqa: E402
from raytracer.scene import Scene                          # noqa: E402
from raytracer.sources import sample_source                # noqa: E402
from raytracer.tracer import Tracer, TraceConfig           # noqa: E402
from raytracer.detector import DetectorGrid, CurvedDetectorGrid  # noqa: E402
from raytracer.surfaces import Sphere, Cylinder            # noqa: E402
from raytracer.optprops import load_optical_properties     # noqa: E402
from . import scenehelpers as sh                           # noqa: E402


# --------------------------------------------------------------------------
# scene builders (curved detector faces + a small-sphere isotropic source)
# --------------------------------------------------------------------------
def _face_dict(fid, spec, corners_uv, surf, area):
    corners = surf.uv_to_xyz(np.array([c[0] for c in corners_uv]),
                             np.array([c[1] for c in corners_uv]))
    return {"id": fid, "surface": spec, "orientation_outward": True,
            "area_m2": float(area), "fingerprint": {}, "mesh_stl": "",
            "trim_polylines_xyz": [[list(c) for c in corners]]}


def cylinder_detector(name, cx, R, phi, hz):
    """Concave cylindrical screen: axis +z through (cx,0,0), radius R, arc of
    half-angle phi centered on u=-pi/2 (facing -x, toward the beam), axial
    half-height hz. u=azimuth, v=axial."""
    origin, axis = [cx, 0.0, 0.0], [0.0, 0.0, 1.0]
    surf = Cylinder(origin, axis, R)
    u0 = -np.pi / 2
    u_lo, u_hi, v_lo, v_hi = u0 - phi, u0 + phi, -hz, hz
    corners = [(u_lo, v_lo), (u_hi, v_lo), (u_hi, v_hi), (u_lo, v_hi)]
    area = R * (u_hi - u_lo) * (v_hi - v_lo)
    fid = "%s.Pad.Face1" % name
    spec = {"type": "cylinder", "origin": origin, "axis": axis, "radius": R}
    face = _face_dict(fid, spec, corners, surf, area)
    return {"name": name, "label": name, "role": "detector",
            "detector": {"face": fid}, "faces": [face]}


def sphere_detector(name, C, R, phi_u, phi_v):
    """Spherical-cap screen centered on (u=-pi/2, v=0) (facing -x). u=azimuth,
    v=latitude. A curvilinear uv-rectangle patch."""
    surf = Sphere(C, R)
    u0, v0 = -np.pi / 2, 0.0
    u_lo, u_hi, v_lo, v_hi = u0 - phi_u, u0 + phi_u, v0 - phi_v, v0 + phi_v
    corners = [(u_lo, v_lo), (u_hi, v_lo), (u_hi, v_hi), (u_lo, v_hi)]
    area = R ** 2 * (u_hi - u_lo) * (np.sin(v_hi) - np.sin(v_lo))
    fid = "%s.Pad.Face1" % name
    spec = {"type": "sphere", "center": list(C), "radius": R}
    face = _face_dict(fid, spec, corners, surf, area)
    return {"name": name, "label": name, "role": "detector",
            "detector": {"face": fid}, "faces": [face]}


def sphere_point_source(name, C, r_src, phi_u, phi_v, power_mW=1.0,
                        lambdac_nm=633.0, coherent=False):
    """Isotropic point source: a small spherical cap centered at C, concentric
    with a sphere detector at C. Uniform-area sampling over the cap emits rays
    radially from ~C into the matching uv-rectangle cone (surf.normal points
    outward = radial), so it is an exact point source at the detector center."""
    surf = Sphere(C, r_src)
    u0, v0 = -np.pi / 2, 0.0
    u_lo, u_hi, v_lo, v_hi = u0 - phi_u, u0 + phi_u, v0 - phi_v, v0 + phi_v
    corners = [(u_lo, v_lo), (u_hi, v_lo), (u_hi, v_hi), (u_lo, v_hi)]
    area = r_src ** 2 * (u_hi - u_lo) * (np.sin(v_hi) - np.sin(v_lo))
    fid = "%s.Pad.Face1" % name
    spec = {"type": "sphere", "center": list(C), "radius": r_src}
    face = _face_dict(fid, spec, corners, surf, area)
    src = {"power_mW": power_mW, "lambdac_nm": lambdac_nm,
           "emit_face": fid, "coherent": coherent}
    return {"name": name, "label": name, "role": "source",
            "source": src, "faces": [face]}


def _grid_for(face, resolution, bins, lam_range):
    stype = face.surface.__class__.__name__
    cls = CurvedDetectorGrid if stype in ("Sphere", "Cylinder") \
        else DetectorGrid
    return cls(face, resolution, bins, lam_range, label=face.id)


def run(model, rays=200000, seed=3, resolution=64, bins=8,
        lam_range=(500e-9, 760e-9), max_reflections=6):
    """Drive Scene + Tracer with the surface-type detector dispatch."""
    common.validate_model(model)
    optprops = load_optical_properties()
    scene = Scene(model, optprops.matdb, optprops.coatings, optprops=optprops)
    grids = {fid: _grid_for(scene.faces[fid], resolution, bins, lam_range)
             for fid in scene.detector_faces}
    cfg = TraceConfig(rays=rays, n_lambda=1, seed=seed, power_floor=1e-14,
                      max_reflections=max_reflections)
    tracer = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(seed)
    batches = [sample_source(scene, scene.bodies[b], s, i, cfg.rays,
                             cfg.n_lambda, rng, ledger=tracer.ledger)
               for i, (b, s) in enumerate(scene.sources)]
    result = tracer.run(batches)
    return result, grids, scene


def _det(grids):
    return next(iter(grids.values()))


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_power_invariance_planar_vs_cylinder():
    """A collimated incoherent beam onto a planar vs a cylindrical detector
    subtending the same beam detects the same total power; closure holds."""
    src = sh.source_body("Src", x=-0.02, half=0.004, power_mW=1.0,
                         lambdac_nm=633.0)
    plan = sh.make_model([src, sh.detector_body("Det", x=0.02, half=0.012)])
    cyl = sh.make_model([src, cylinder_detector("Det", cx=0.03, R=0.02,
                                                phi=0.6, hz=0.009)])

    r_p, g_p, _ = run(plan)
    r_c, g_c, _ = run(cyl)

    assert isinstance(_det(g_p), DetectorGrid)
    assert isinstance(_det(g_c), CurvedDetectorGrid)

    P_p = float(_det(g_p).inc.sum())
    P_c = float(_det(g_c).inc.sum())
    # both catch (essentially) the whole 1 mW beam
    assert P_p == pytest.approx(1e-3, rel=0.02)
    assert P_c == pytest.approx(P_p, rel=0.02)

    for rep in (r_p.ledger.report(r_p.source_names),
                r_c.ledger.report(r_c.source_names)):
        assert rep["closure_ok"]

    # deliverable 4: the grid's per-key detected tally == the ledger booking
    tally = sum(_det(g_c).detected_incoherent.values())
    booked = sum(r_c.ledger.detected.values())
    assert tally == pytest.approx(booked, rel=1e-9)


def test_cylinder_obliquity_cosine():
    """Collimated beam on a wide cylindrical arc: detector irradiance follows
    cos(theta) = cos(offset-from-arc-center) across the arc."""
    R, phi = 0.02, 1.2
    src = sh.source_body("Src", x=-0.03, half=0.018, power_mW=1.0,
                         lambdac_nm=633.0)
    model = sh.make_model([src, cylinder_detector("Det", cx=0.03, R=R,
                                                  phi=phi, hz=0.02)])
    _, grids, _ = run(model, rays=400000, resolution=64)
    det = _det(grids)
    area = np.where(det.pixel_area_map > 0, det.pixel_area_map, np.inf)
    irr = det.inc.sum(axis=0) / area                    # (H, W) W/m^2

    # per-column mean irradiance over the central axial band (|z| small)
    zc = det.v_lo + (np.arange(det.H) + 0.5) * det.dv
    band = np.abs(zc) < 0.010
    col_irr = irr[band].mean(axis=0)                    # (W,)
    u_c = det.u_lo + (np.arange(det.W) + 0.5) * det.du
    offset = u_c + np.pi / 2                            # 0 at arc center
    cos_t = np.cos(offset)

    lit = col_irr > 0.25 * col_irr.max()               # illuminated columns
    assert lit.sum() > 20
    ratio = col_irr[lit] / cos_t[lit]
    # cos(theta) sweeps a real range across the lit arc (not near-flat)
    assert cos_t[lit].min() < 0.6
    # irradiance / cos(theta) is constant across the arc within MC noise
    assert ratio.std() / ratio.mean() < 0.08


def test_sphere_solid_angle_flat_irradiance():
    """Isotropic point source at a sphere-cap detector's center: per-pixel
    POWER tracks pixel area, so irradiance is flat across the cap."""
    C = (0.05, 0.0, 0.0)
    R_det, r_src, phi = 0.03, 0.003, 0.32
    det_b = sphere_detector("Det", C, R_det, phi, phi)
    src_b = sphere_point_source("Src", C, r_src, phi, phi, power_mW=1.0)
    model = sh.make_model([src_b, det_b])
    _, grids, _ = run(model, rays=500000, resolution=44)
    det = _det(grids)
    assert isinstance(det, CurvedDetectorGrid)

    power = det.inc.sum(axis=0)                          # (H, W) W per pixel
    area = det.pixel_area_map
    lit = (area > 0) & (power > 0.2 * power[area > 0].max())
    assert lit.sum() > 200

    # per-pixel power is proportional to pixel area (flat irradiance): bin by
    # latitude v (where the sphere area element varies most) and check the
    # band irradiance is constant.
    vc = det.v_lo + (np.arange(det.H) + 0.5) * det.dv
    vgrid = np.broadcast_to(vc[:, None], power.shape)
    bands = np.linspace(det.v_lo + 0.02, det.v_hi - 0.02, 6)
    band_irr = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        m = lit & (vgrid >= lo) & (vgrid < hi)
        if m.sum() > 30:
            band_irr.append(power[m].sum() / area[m].sum())
    band_irr = np.array(band_irr)
    assert len(band_irr) >= 4
    assert band_irr.std() / band_irr.mean() < 0.06

    # area element really does vary across the cap (cos(v)) — otherwise the
    # test would be trivially satisfied by a constant-area grid
    assert area[lit].max() / area[lit].min() > 1.03


def test_coherent_source_curved_detector_raises():
    """A coherent source hitting a curved detector raises the clear guard."""
    src = sh.source_body("Src", x=-0.02, half=0.004, power_mW=1.0,
                         lambdac_nm=633.0, coherent=True)
    model = sh.make_model([src, cylinder_detector("Det", cx=0.03, R=0.02,
                                                  phi=0.6, hz=0.009)])
    with pytest.raises(NotImplementedError, match="coherent gather on curved"):
        run(model, rays=2000)


def test_planar_h5_backward_compatible(tmp_path):
    """Planar detector .h5 files gain NO new attrs/datasets (byte-compatible):
    exactly the pre-Phase-10 attr set, no pixel_area_map, no surface_type."""
    import h5py
    import run_trace

    src = sh.source_body("Src", x=-0.02, half=0.004, power_mW=1.0)
    model = sh.make_model([src, sh.detector_body("Det", x=0.02, half=0.012)])
    _, grids, _ = run(model, rays=20000, resolution=32)
    assert isinstance(_det(grids), DetectorGrid)

    run_trace.save_detectors(tmp_path, [grids], seeds=1)
    (h5file,) = list((tmp_path / "detectors").glob("*.h5"))
    with h5py.File(h5file) as h:
        attrs = set(h.attrs.keys())
        datasets = set(h.keys())
    expected_attrs = {"label", "H", "W", "pixel_m", "lam_lo_m", "lam_hi_m",
                      "xhat", "yhat", "normal", "x_lo", "y_lo", "seeds"}
    assert attrs == expected_attrs
    assert "pixel_area_map" not in datasets
    assert "surface_type" not in attrs


def test_curved_h5_has_area_map(tmp_path):
    """Curved detector .h5 carries surface_type + a per-pixel area map."""
    import h5py
    import run_trace

    src = sh.source_body("Src", x=-0.02, half=0.004, power_mW=1.0)
    model = sh.make_model([src, cylinder_detector("Det", cx=0.03, R=0.02,
                                                  phi=0.6, hz=0.009)])
    _, grids, _ = run(model, rays=20000, resolution=32)
    run_trace.save_detectors(tmp_path, [grids], seeds=1)
    (h5file,) = list((tmp_path / "detectors").glob("*.h5"))
    with h5py.File(h5file) as h:
        assert h.attrs["surface_type"] == "cylinder"
        amap = h["pixel_area_map"][...]
        det = _det(grids)
        assert amap.shape == (det.H, det.W)
        # cylinder area element R*du*dv is constant on lit pixels
        lit = amap > 0
        assert np.allclose(amap[lit], det.radius * det.du * det.dv)
