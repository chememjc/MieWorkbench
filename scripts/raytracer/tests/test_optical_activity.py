# =============================================================================
# test_optical_activity.py — scene-level natural optical activity (gyrotropic
# uniaxial crystals) end-to-end through the real Scene/Tracer pipeline.
#
# Physics: alpha-quartz along its optic axis is gyrotropic; a linear input's
# polarization plane rotates by rho*d with rho = 21.77 deg/mm @ 589.3 nm
# (uniaxial.miebrf registry datum; Kaminsky Rep. Prog. Phys. 63, 1575 (2000)).
# Between crossed analyzers the transmission is sin^2(rho*d), between parallel
# analyzers cos^2(rho*d). A non-gyrotropic uniaxial (calcite) along its axis is
# degenerate (n_o == n_e) and does NOT rotate -> crossed extinction.
#
# The tracer routes near-axis gyrotropic rays through the isotropic n_o path
# (single child, full Jones) and rotates the plane in Tracer.step()
# (_apply_optical_activity); the o/e split machinery is bypassed there because
# the o/e eigenbasis is degenerate/arbitrary on the axis. See tracer.py.
# =============================================================================
import sys
import math
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))              # tests/
import scenehelpers as sh                                    # noqa: E402

RHO = 21.77          # deg/mm, alpha-quartz rotatory power @589.3 nm (registry)
LAM = 589.3          # nm (the sodium-D reference line the datum is quoted at)


def _run(material, analyzer_axis, thick_mm=2.0, crystal_axis=(1.0, 0.0, 0.0),
         rays=40000, seed=3):
    """Source (unpolarized) -> z-axis input polarizer -> crystal slab (optic
    axis along the beam) -> analyzer -> detector. Returns (detected_W,
    max_closure_error). Boxes are on the x-axis in metres."""
    x0 = 0.003
    x1 = x0 + thick_mm * 1e-3
    model = sh.make_model([
        sh.source_body("Src", x=-0.02, half=0.002, power_mW=1.0,
                       lambdac_nm=LAM, coherent=False),
        sh.slab_body("InPol", "air", 0.000, 0.001, half=0.01,
                     polarizer="ideal_linear", polarizer_axis=[0.0, 0.0, 1.0]),
        sh.slab_body("Xtal", material, x0, x1, half=0.01,
                     crystal_axis=list(crystal_axis)),
        sh.slab_body("Ana", "air", x1 + 0.001, x1 + 0.002, half=0.01,
                     polarizer="ideal_linear",
                     polarizer_axis=list(analyzer_axis)),
        sh.detector_body("Det", x=x1 + 0.01, half=0.01),
    ])
    result, grids, _ = sh.trace_scene(model, rays=rays, n_lambda=1, seed=seed,
                                      power_floor=1e-14)
    (fid, det), = grids.items()
    rep = result.ledger.report(result.source_names)
    cl = max(s["closure_error"] for s in rep["sources"].values())
    return float(det.inc.sum()), cl


def _crossed_fraction(material, thick_mm=2.0, **kw):
    """crossed / (crossed + parallel) detected power — the fraction of the
    post-input-polarizer light that a crossed analyzer passes. Equals
    sin^2(rho*d) for optical rotation."""
    par, cl_p = _run(material, [0.0, 0.0, 1.0], thick_mm=thick_mm, **kw)   # z
    cross, cl_x = _run(material, [0.0, 1.0, 0.0], thick_mm=thick_mm, **kw)  # y
    assert cl_p < 1e-3 and cl_x < 1e-3, (cl_p, cl_x)   # unitary => closes
    return cross / (cross + par)


def test_quartz_crossed_analyzer_sin2():
    """2 mm z-cut quartz, crossed analyzer: sin^2(rho*d) = sin^2(43.54 deg)."""
    frac = _crossed_fraction("quartz", thick_mm=2.0)
    expect = math.sin(math.radians(RHO * 2.0)) ** 2
    assert frac == pytest.approx(expect, rel=0.02), (frac, expect)


def test_quartz_parallel_analyzer_cos2():
    """Parallel analyzer sees the complement cos^2(rho*d)."""
    par_frac = 1.0 - _crossed_fraction("quartz", thick_mm=2.0)
    expect = math.cos(math.radians(RHO * 2.0)) ** 2
    assert par_frac == pytest.approx(expect, rel=0.02), (par_frac, expect)


def test_quartz_rotation_scales_with_thickness():
    """Rotation is rho*d: halving the slab halves the angle. 1 mm ->
    sin^2(21.77 deg), distinct from the 2 mm value -> proves genuine optical
    rotation (linear in path), not a fixed polarization scramble."""
    frac1 = _crossed_fraction("quartz", thick_mm=1.0)
    expect1 = math.sin(math.radians(RHO * 1.0)) ** 2
    assert frac1 == pytest.approx(expect1, rel=0.05), (frac1, expect1)


def test_calcite_along_axis_no_rotation():
    """Regression: a non-gyrotropic uniaxial (calcite) along its optic axis
    is index-degenerate and does NOT rotate the plane -> crossed extinction.
    Confirms gyration is applied ONLY to bodies carrying registry gyration
    data, never to plain birefringent crystals."""
    frac = _crossed_fraction("calcite", thick_mm=2.0)
    assert frac < 0.02, frac        # crossed analyzer extinguishes (no rotation)
