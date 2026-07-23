# =============================================================================
# test_gaussian_beam.py — Phase 6 (lowhanging.md #8): Gaussian-beam source
# mode (source.beam {waist_mm, m2}) and transverse apodization
# (source.apodization {kind:'gaussian', w0_mm, order}).
#
#   * beam mode: waist propagation oracle w(z) = w0*sqrt(1+(z/zR)^2),
#     zR = pi*w0^2/lambda, measured from a detector's second moment in
#     air with no optics (direct-deposit incoherent path); M2 scaling of
#     the far-field divergence.
#   * apodization: exact energy conservation (renormalized by construction),
#     near-field radial power CDF matches the Gaussian aperture profile,
#     and a super-Gaussian (order=3) core is flatter than a plain
#     Gaussian (order=1) at the same w0.
#
# All traces use collimated (non-beam) or beam-mode sources with NO optics
# in between the source and detector, so the incoherent direct-deposit path
# is exercised — no coherent-gather dA subtlety is in play here (see the
# scoped-limitation note in raytracer/sources.py for that).
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest scripts/raytracer/tests/test_gaussian_beam.py -q
# =============================================================================
import numpy as np
import pytest

import common
from raytracer.scene import Scene
from raytracer.sources import sample_source

from . import scenehelpers as sh

LAM_NM = 633.0
LAM_M = LAM_NM * 1e-9


def _radial_moments(det):
    """(gx, gy, irr): pixel-center grid coords (m) + summed power map."""
    xs = det.x_lo + (np.arange(det.W) + 0.5) * det.pixel_m
    ys = det.y_lo + (np.arange(det.H) + 0.5) * det.pixel_m
    gx, gy = np.meshgrid(xs, ys)
    irr = det.inc.sum(axis=0)
    return gx, gy, irr


def _second_moment_width(det):
    """2*sqrt(Var_x + Var_y) — the w s.t. a radially-symmetric profile
    I(r) ~ exp(-2 r^2/w^2) has per-axis variance w^2/4 each."""
    gx, gy, irr = _radial_moments(det)
    total = irr.sum()
    cx = (gx * irr).sum() / total
    cy = (gy * irr).sum() / total
    varx = ((gx - cx) ** 2 * irr).sum() / total
    vary = ((gy - cy) ** 2 * irr).sum() / total
    return float(np.sqrt(2.0 * (varx + vary)))


def _trace_beam_to_z(w0_mm, z_m, m2=1.0, rays=200000, seed=1,
                     src_x=-0.02, half_mm=1.0, det_half_mm=1.0,
                     resolution=400):
    src = sh.source_body(x=src_x, half=half_mm * 1e-3, power_mW=1.0,
                         lambdac_nm=LAM_NM, coherent=False,
                         beam_waist_mm=w0_mm, m2=m2)
    det = sh.detector_body(x=src_x + z_m, half=det_half_mm * 1e-3)
    model = sh.make_model([src, det])
    result, grids, scene = sh.trace_scene(model, rays=rays, n_lambda=1,
                                          seed=seed, resolution=resolution)
    det_grid = list(grids.values())[0]
    return result, det_grid


# ---------------------------------------------------------------------------
# beam mode: waist propagation oracle
# ---------------------------------------------------------------------------
def test_beam_waist_propagation_near_and_far():
    w0_mm = 0.05                                       # 50 um waist
    w0_m = w0_mm * 1e-3
    zR = np.pi * w0_m ** 2 / LAM_M

    z_near = 0.3 * zR                                  # z << zR regime
    z_far = 5.0 * zR                                    # z >> zR regime

    _, det_near = _trace_beam_to_z(w0_mm, z_near, det_half_mm=0.3)
    w_meas_near = _second_moment_width(det_near)
    w_expect_near = w0_m * np.sqrt(1.0 + (z_near / zR) ** 2)
    assert abs(w_meas_near - w_expect_near) / w_expect_near < 0.08, (
        w_meas_near, w_expect_near)

    _, det_far = _trace_beam_to_z(w0_mm, z_far, det_half_mm=2.0)
    w_meas_far = _second_moment_width(det_far)
    w_expect_far = w0_m * np.sqrt(1.0 + (z_far / zR) ** 2)
    assert abs(w_meas_far - w_expect_far) / w_expect_far < 0.08, (
        w_meas_far, w_expect_far)

    # sanity: far field is much wider than near field (divergence is real)
    assert w_meas_far > 3.0 * w_meas_near


def test_beam_m2_doubles_far_field_divergence():
    w0_mm = 0.05
    w0_m = w0_mm * 1e-3
    zR = np.pi * w0_m ** 2 / LAM_M
    z_far = 5.0 * zR

    _, det_m1 = _trace_beam_to_z(w0_mm, z_far, m2=1.0, det_half_mm=2.0,
                                 seed=2)
    _, det_m2 = _trace_beam_to_z(w0_mm, z_far, m2=2.0, det_half_mm=3.0,
                                 seed=3)
    w1 = _second_moment_width(det_m1)
    w2 = _second_moment_width(det_m2)
    # at z=5*zR the linear (divergence) term dominates w(z), so w scales
    # ~linearly with m2 (residual w0 contribution keeps the ratio a hair
    # under 2 - see the m2-doubling derivation in sources.py's beam block)
    ratio = w2 / w1
    assert 1.8 < ratio < 2.15, (w1, w2, ratio)


def test_beam_and_apodization_combined_energy_exact():
    # a source can carry BOTH source.beam and source.apodization at once
    # (e.g. an apodized real laser); this exercises the two features'
    # separately-scoped w0 variables in the same sample_source call and
    # re-checks the exact-power-conservation invariant with both active.
    apod = common.parse_apodization_spec("gaussian:w0=0.2:order=2")
    src = sh.source_body(x=-0.02, half=1e-3, power_mW=3.0,
                         lambdac_nm=LAM_NM, coherent=False,
                         beam_waist_mm=0.05, m2=1.5, apodization=apod)
    det = sh.detector_body(x=0.01, half=2e-3)
    model = sh.make_model([src, det])
    result, grids, scene = sh.trace_scene(model, rays=50000, n_lambda=1,
                                          seed=7, resolution=200)
    power_W = 3.0e-3
    err = abs(result.ledger.emitted[0] - power_W) / power_W
    assert err < 1e-12, err
    assert np.all(result.ledger.closure() < 1e-3)


# ---------------------------------------------------------------------------
# apodization
# ---------------------------------------------------------------------------
def _trace_apodized(w0_mm, order, rays=200000, seed=5, half_mm=2.0,
                    det_x=-0.019, det_half_mm=2.0, resolution=400):
    apod = common.parse_apodization_spec("gaussian:w0=%g:order=%d"
                                         % (w0_mm, order))
    src = sh.source_body(x=-0.02, half=half_mm * 1e-3, power_mW=2.5,
                         lambdac_nm=LAM_NM, coherent=False,
                         apodization=apod)
    det = sh.detector_body(x=det_x, half=det_half_mm * 1e-3)
    model = sh.make_model([src, det])
    result, grids, scene = sh.trace_scene(model, rays=rays, n_lambda=1,
                                          seed=seed, resolution=resolution)
    return result, list(grids.values())[0]


def test_apodization_energy_exact():
    result, det = _trace_apodized(0.3, order=1)
    power_W = 2.5e-3
    err = abs(result.ledger.emitted[0] - power_W) / power_W
    assert err < 1e-12, err
    # closure gate (physics invariant): the full ledger still balances
    assert np.all(result.ledger.closure() < 1e-3)


def test_apodization_gaussian_profile_matches_theory():
    w0_mm = 0.3
    w0_m = w0_mm * 1e-3
    _, det = _trace_apodized(w0_mm, order=1)
    gx, gy, irr = _radial_moments(det)
    r = np.sqrt(gx ** 2 + gy ** 2)
    total = irr.sum()

    for R_over_w0, tol in ((0.5, 0.04), (1.0, 0.04), (1.5, 0.03)):
        R = R_over_w0 * w0_m
        frac_meas = irr[r < R].sum() / total
        frac_theory = 1.0 - np.exp(-2.0 * R_over_w0 ** 2)
        assert abs(frac_meas - frac_theory) < tol, (
            R_over_w0, frac_meas, frac_theory)


def test_apodization_super_gaussian_flatter_core():
    w0_mm = 0.3
    w0_m = w0_mm * 1e-3
    # apodization only reweights POWER, not position (sampling stays
    # uniform over the whole physical aperture) — a wide aperture wastes
    # almost all rays outside the profile's core, starving the small
    # center/shell query bins below of samples; keep the aperture close
    # to w0 and raise the ray budget so both bins get enough rays.
    _, det1 = _trace_apodized(w0_mm, order=1, seed=11, rays=800000,
                              half_mm=0.6, det_half_mm=0.7)
    _, det3 = _trace_apodized(w0_mm, order=3, seed=12, rays=800000,
                              half_mm=0.6, det_half_mm=0.7)

    def _core_ratio(det):
        gx, gy, irr = _radial_moments(det)
        r = np.sqrt(gx ** 2 + gy ** 2)
        center = irr[r < 0.1 * w0_m].sum() / max(np.sum(r < 0.1 * w0_m), 1)
        shell_lo, shell_hi = 0.25 * w0_m, 0.35 * w0_m
        shell = (r >= shell_lo) & (r < shell_hi)
        shell_val = irr[shell].sum() / max(np.sum(shell), 1)
        return shell_val / center      # 1.0 = perfectly flat core

    ratio1 = _core_ratio(det1)
    ratio3 = _core_ratio(det3)
    # order=1 (plain Gaussian) falls off measurably by r=0.3*w0; order=3
    # (super-Gaussian) stays much closer to flat -> ratio3 much closer to 1
    assert ratio3 > ratio1, (ratio1, ratio3)
    assert ratio3 > 0.9, ratio3
    assert ratio1 < 0.9, ratio1


# ---------------------------------------------------------------------------
# contract: beam_waist requires a planar emitting face
# ---------------------------------------------------------------------------
class _FakeBody:
    def __init__(self, index, label):
        self.index = index
        self.label = label


class _FakeScene:
    def __init__(self, emit_faces):
        self.emit_faces = emit_faces


def test_beam_waist_rejects_nonplanar_face():
    from raytracer.surfaces import Sphere, AnalyticFace

    r = 0.001
    face = AnalyticFace("Src.Synth.Face1", Sphere([-0.02, 0, 0], r), [],
                        True, 0, 0, area_m2=4.0 * np.pi * r ** 2)
    scene = _FakeScene({0: face})
    body = _FakeBody(0, "Src")
    src = {"power_mW": 1.0, "lambdac_nm": LAM_NM, "coherent": False,
           "emit_face": face.id, "beam": {"waist_mm": 0.05, "m2": 1.0}}
    rng = np.random.default_rng(0)
    with pytest.raises(NotImplementedError):
        sample_source(scene, body, src, 0, 1000, 1, rng)
