# =============================================================================
# test_polarizer_filter.py — polarizer elements, bulk spectral filters,
# tabulated coatings, and uniaxial birefringence THROUGH THE REAL TRACER
# (synthetic box scenes on the x-axis, no FreeCAD needed).
#
# Quantitative pins:
#   * Malus's law: detected(theta)/detected(0) = cos^2(theta) (ratio form —
#     Fresnel factors cancel exactly for an ideal polarizer)
#   * crossed polarizers: T ~ T_perp; rejected power lands in the
#     polarizer_absorbed ledger bucket; closure < 1e-3 everywhere
#   * bulk filter: detected ratio = T_int^(d/d_ref) (Beer-Lambert scaling)
#   * calcite slab: o-spot straight, e-spot displaced by d*tan(rho),
#     polarization-selectable via the source polarization
#   * circular-polarizer Jones stage: linear in -> circular out with the
#     documented handedness
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from raytracer.tests import scenehelpers as sh                # noqa: E402
from raytracer.optprops import load_optical_properties        # noqa: E402


@pytest.fixture(scope="module")
def props(tmp_path_factory):
    """Real materials + a synthetic polarizer/filter registry with exact
    numbers (tests must not depend on vendor-digitized curves)."""
    import shutil
    root = tmp_path_factory.mktemp("optprops") / "opticalproperties"
    real = SCRIPTS.parent / "opticalproperties"
    shutil.copytree(real, root)
    # Overwrite the copied registries IN PLACE (new self-describing names,
    # matching what copytree brought over from the renamed repo library) --
    # writing to the legacy .csv name instead would leave the copied
    # .miepol/.miefilt file sitting alongside a stray new .csv one, and the
    # loader's fallback prefers the (untouched, real-library) .miepol/
    # .miefilt file over this synthetic one.
    (root / "polarizer" / "polarizers.miepol").write_text(
        'name,type,table_csv,retardance_waves,reference\n'
        'ideal_lp,linear,ideal_lp.csv,,"synthetic ideal"\n'
        'qcirc_r,circular_right,ideal_lp.csv,0.25,"synthetic"\n'
        'qcirc_l,circular_left,ideal_lp.csv,0.25,"synthetic"\n')
    (root / "polarizer" / "tables" / "ideal_lp.csv").write_text(
        "wavelength_nm,T_parallel,T_perpendicular\n"
        "300,1.0,1e-6\n1200,1.0,1e-6\n")
    (root / "filter" / "filters.miefilt").write_text(
        'name,table_csv,ref_thickness_mm,reference\n'
        'f_test,f_test.csv,2.0,"synthetic"\n')
    (root / "filter" / "tables" / "f_test.csv").write_text(
        "wavelength_nm,transmittance_internal\n"
        "300,0.9\n1200,0.9\n")
    return load_optical_properties(root=root)


def _detected(result, grids):
    (fid, det), = grids.items()
    return float(det.inc.sum()), det


def _closure_ok(result):
    rep = result.ledger.report(result.source_names)
    for name, s in rep["sources"].items():
        assert s["closure_error"] < 1e-3, (name, s["closure_error"])
    return rep


# ---------------------------------------------------------------------------
# Malus's law + crossed polarizers
# ---------------------------------------------------------------------------
def _malus_run(props, polarization, axis=(0.0, 0.0, 1.0)):
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False,
                       polarization=polarization),
        sh.slab_body("Pol", "pmma", 0.000, 0.002, half=0.01,
                     polarizer="ideal_lp", polarizer_axis=list(axis)),
        sh.detector_body(x=0.02, half=0.01),
    ])
    result, grids, _ = sh.trace_scene(model, rays=8000, optprops=props)
    rep = _closure_ok(result)
    p, _ = _detected(result, grids)
    return p, rep


def test_malus_law(props):
    """detected(theta)/detected(0) = cos^2 theta; axis = z, so linear:theta
    is at angle theta to the transmission axis."""
    p0, _ = _malus_run(props, {"kind": "linear", "angle_deg": 0.0})
    p30, _ = _malus_run(props, {"kind": "linear", "angle_deg": 30.0})
    p60, _ = _malus_run(props, {"kind": "linear", "angle_deg": 60.0})
    assert p30 / p0 == pytest.approx(np.cos(np.deg2rad(30)) ** 2, rel=0.01)
    assert p60 / p0 == pytest.approx(np.cos(np.deg2rad(60)) ** 2, rel=0.01)


def test_crossed_polarizer_floor_and_bucket(props):
    """linear:90 against an axis-z polarizer: transmission at the T_perp
    floor and the rejected power in polarizer_absorbed."""
    p0, rep0 = _malus_run(props, {"kind": "linear", "angle_deg": 0.0})
    p90, rep90 = _malus_run(props, {"kind": "linear", "angle_deg": 90.0})
    assert p90 / p0 == pytest.approx(1e-6, rel=1.0)   # ER floor, factor 2
    src = next(iter(rep90["sources"].values()))
    # essentially all the power that entered the film was rejected
    assert src["polarizer_absorbed"] > 0.85 * src["emitted_W"]
    src0 = next(iter(rep0["sources"].values()))
    assert src0["polarizer_absorbed"] < 1e-5 * src0["emitted_W"]


def test_unpolarized_half_transmission(props):
    """Unpolarized in -> (T_par + T_perp)/2 ~ 1/2 relative to aligned."""
    p0, _ = _malus_run(props, {"kind": "linear", "angle_deg": 0.0})
    pu, _ = _malus_run(props, None)     # default unpolarized
    assert pu / p0 == pytest.approx(0.5, rel=0.02)


# ---------------------------------------------------------------------------
# bulk spectral filter: Beer-Lambert thickness scaling
# ---------------------------------------------------------------------------
def _filter_run(props, thickness_m, with_filter):
    extra = {"filter": "f_test"} if with_filter else {}
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False),
        sh.slab_body("Filt", "bk7", 0.0, thickness_m, half=0.01, **extra),
        sh.detector_body(x=0.02, half=0.01),
    ])
    result, grids, _ = sh.trace_scene(model, rays=6000, optprops=props)
    _closure_ok(result)
    return _detected(result, grids)[0]


def test_filter_beer_lambert_scaling(props):
    """f_test: T_int = 0.9 at d_ref = 2 mm. A slab at d_ref transmits
    exactly 0.9x the no-filter power (identical geometry cancels Fresnel);
    at 2*d_ref it transmits 0.9^2."""
    base2 = _filter_run(props, 0.002, with_filter=False)
    filt2 = _filter_run(props, 0.002, with_filter=True)
    assert filt2 / base2 == pytest.approx(0.9, rel=5e-3)
    base4 = _filter_run(props, 0.004, with_filter=False)
    filt4 = _filter_run(props, 0.004, with_filter=True)
    assert filt4 / base4 == pytest.approx(0.81, rel=5e-3)


# ---------------------------------------------------------------------------
# calcite slab: double refraction through the real tracer
# ---------------------------------------------------------------------------
CALCITE_T = 0.010                      # slab thickness [m]
# c-axis at 45 deg in the x-z plane -> e-ray walks off in +/-z by
# t*tan(rho); o-ray goes straight. rho(45deg, 590nm) = 6.23 deg.
C_AXIS = [np.sqrt(0.5), 0.0, np.sqrt(0.5)]


def _calcite_run(props, polarization, lambdac=590.0):
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False, half=2e-4,
                       lambdac_nm=lambdac, polarization=polarization),
        sh.slab_body("Cal", "calcite", 0.0, CALCITE_T, half=0.01,
                     crystal_axis=C_AXIS),
        sh.detector_body(x=CALCITE_T + 0.001, half=0.01),
    ])
    result, grids, scene = sh.trace_scene(model, rays=20000,
                                          optprops=props, resolution=512)
    _closure_ok(result)
    (fid, det), = grids.items()
    return det


def _spot_positions(det):
    """z-positions [m] of intensity maxima along the detector's grid axis
    that maps to global z (read xhat/yhat — never assume)."""
    img = det.inc.sum(axis=0)
    z_is_x = abs(det.xhat[2]) > abs(det.yhat[2])
    prof = img.sum(axis=0) if z_is_x else img.sum(axis=1)
    n = len(prof)
    if z_is_x:
        coord = det.x_lo + (np.arange(n) + 0.5) * det.pixel_m
        sgn = np.sign(det.xhat[2])
    else:
        coord = det.y_lo + (np.arange(n) + 0.5) * det.pixel_m
        sgn = np.sign(det.yhat[2])
    # spots = contiguous runs above 20% of max -> intensity centroids
    # (a finite-width source makes each spot a noisy plateau, not a
    # single-pixel peak)
    above = prof > 0.2 * prof.max()
    spots = []
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        w = prof[i:j]
        spots.append(float(np.sum(coord[i:j] * w) / np.sum(w)))
        i = j
    return sorted(sgn * z for z in spots)


def test_calcite_double_spot_separation(props):
    """Unpolarized: two spots separated by t*tan(6.23deg) = 1.09 mm."""
    det = _calcite_run(props, None)
    spots = _spot_positions(det)
    assert len(spots) == 2, spots
    sep = abs(spots[1] - spots[0])
    expect = CALCITE_T * np.tan(np.deg2rad(6.23))
    assert sep == pytest.approx(expect, rel=0.02), (sep, expect)


def test_calcite_polarization_selects_spot(props):
    """Pure o-polarized light (along e_o = y here) -> one straight spot;
    pure e-polarized (in the (k,c) plane, along z) -> one displaced spot."""
    # e_ref = z; the o eigenvector for k=+x, c in x-z plane is +-y, i.e.
    # linear:90; the e eigenvector is along z, i.e. linear:0.
    det_o = _calcite_run(props, {"kind": "linear", "angle_deg": 90.0})
    det_e = _calcite_run(props, {"kind": "linear", "angle_deg": 0.0})
    spots_o = _spot_positions(det_o)
    spots_e = _spot_positions(det_e)
    assert len(spots_o) == 1 and abs(spots_o[0]) < 3e-4, spots_o
    expect = CALCITE_T * np.tan(np.deg2rad(6.23))
    assert len(spots_e) == 1, spots_e
    assert abs(abs(spots_e[0]) - expect) < 0.02 * expect + 1e-4, \
        (spots_e, expect)


# ---------------------------------------------------------------------------
# conical-point runtime guard (engine3.md Sec 7.2): c-axis along the beam
# puts every primary ray's incident k exactly in the optic-axis degeneracy
# cone (a flat unapodized source is collimated -- see scenehelpers' header
# comment), so this is the cheapest possible trigger. Must NOT perturb the
# ordinary 45deg-axis calcite walk-off tests above (no physics change, only
# a counted read of the existing degeneracy test).
# ---------------------------------------------------------------------------
def test_conical_point_guard_axis_along_beam(props):
    n_rays = 2000
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False, half=2e-4,
                       lambdac_nm=590.0),
        sh.slab_body("Cal", "calcite", 0.0, CALCITE_T, half=0.01,
                     crystal_axis=[1.0, 0.0, 0.0]),   # || beam direction
        sh.detector_body(x=CALCITE_T + 0.001, half=0.01),
    ])
    with pytest.warns(UserWarning, match="optic-axis degeneracy of Cal"):
        result, grids, scene = sh.trace_scene(model, rays=n_rays,
                                              optprops=props)
    assert result.conical_guard.get("Cal") == n_rays
    _closure_ok(result)


def test_calcite_walkoff_unaffected_by_guard_instrumentation(props):
    """The guard is read-only counting; the 45deg-axis walk-off separation
    (test_calcite_double_spot_separation) must reproduce bit-for-bit and
    raise NO conical-point warning (|k x c| = sin(45deg), nowhere near the
    1e-9 degeneracy cone)."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        det = _calcite_run(props, None)
    assert not any("optic-axis degeneracy" in str(w.message) for w in caught), \
        [str(w.message) for w in caught]
    spots = _spot_positions(det)
    assert len(spots) == 2, spots
    sep = abs(spots[1] - spots[0])
    expect = CALCITE_T * np.tan(np.deg2rad(6.23))
    assert sep == pytest.approx(expect, rel=0.02), (sep, expect)


# ---------------------------------------------------------------------------
# circular-polarizer Jones stage (unit level, real _apply_polarizer)
# ---------------------------------------------------------------------------
def test_circular_polarizer_output_handedness(props):
    from types import SimpleNamespace
    from raytracer.tracer import Tracer
    from raytracer.rays import RayBatch

    entry = props.polarizers["qcirc_r"]
    fake_scene = SimpleNamespace(polarizers={"qcirc_r": entry,
                                             "qcirc_l":
                                             props.polarizers["qcirc_l"]})
    fake_self = SimpleNamespace(scene=fake_scene)
    for name, expect_sign in (("qcirc_r", +1.0), ("qcirc_l", -1.0)):
        body = SimpleNamespace(polarizer=name,
                               polarizer_axis=np.array([0.0, 0.0, 1.0]),
                               label="CP")
        b = RayBatch(4)
        b.dir[:] = [1.0, 0.0, 0.0]
        b.s_hat[:] = [0.0, 0.0, 1.0]        # e_ref frame
        b.lam[:] = 532e-9
        b.Es[:] = 1.0                        # linear along the axis
        b.Ep[:] = 0.0
        Tracer._apply_polarizer(fake_self, body, b,
                                np.ones(4, dtype=bool))
        # output must be circular: |Es| == |Ep|, +-90deg relative phase
        assert np.allclose(np.abs(b.Es), np.abs(b.Ep), rtol=1e-9)
        rel = np.angle(b.Ep / b.Es)
        assert np.allclose(rel, expect_sign * np.pi / 2, atol=1e-9), \
            (name, rel)
        # ideal stage: no power lost in the retarder, T_par = 1
        assert np.allclose(np.abs(b.Es) ** 2 + np.abs(b.Ep) ** 2, 1.0,
                           rtol=1e-6)
