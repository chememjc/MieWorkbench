# =============================================================================
# test_figure_error.py -- Zernike surface figure error (engine3 Sec 11 / P8).
#
#   * analysis.zernike_cart_grad + surfaces.PerturbedSurface sag/gradient vs
#     finite differences (1e-8);
#   * the three physics GATES: (a) a lambda/4 PV Z4 defocus on a flat mirror
#     shifts the reflected focus by the analytic amount; (b) Z6 astigmatism
#     produces two orthogonal, opposite-sign line foci (the textbook astigmatic
#     line images); (c) the Marechal Strehl matches the direct PSF-peak Strehl
#     for small RMS (< lambda/14) within 10%;
#   * the opticalproperties/figure registry loader;
#   * the cengine routing token (figure_error is Python-only).
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#         scripts/raytracer/tests/test_figure_error.py -q
# =============================================================================
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

from raytracer import surfaces, analysis, optprops, cengine   # noqa: E402

FIG_CSV = SCRIPTS.parent / "opticalproperties" / "figure" / "figures.miefig"
LAM = 633e-9
R_NORM = 12.5e-3


def _reflect(d, n):
    return d - 2.0 * np.sum(d * n, axis=-1, keepdims=True) * n


# --------------------------------------------------------------------------- #
# derivatives vs finite differences
# --------------------------------------------------------------------------- #
def test_zernike_cart_grad_vs_finite_difference():
    rng = np.random.default_rng(0)
    uv = rng.uniform(-0.75, 0.75, size=(400, 2))
    u, v = uv[:, 0], uv[:, 1]
    h = 1e-7

    def Z(j, uu, vv):
        return analysis.zernike(j, np.hypot(uu, vv), np.arctan2(vv, uu))

    worst = 0.0
    for j in range(2, 16):
        gu, gv = analysis.zernike_cart_grad(j, u, v)
        fdu = (Z(j, u + h, v) - Z(j, u - h, v)) / (2 * h)
        fdv = (Z(j, u, v + h) - Z(j, u, v - h)) / (2 * h)
        worst = max(worst, np.abs(gu - fdu).max(), np.abs(gv - fdv).max())
    assert worst < 1e-6, worst


def test_perturbed_surface_sag_gradient_vs_finite_difference():
    base = surfaces.Plane([0, 0, 0], [1, 0, 0])
    coeffs = {4: 45.7e-9, 5: 100e-9, 6: 30e-9, 11: 20e-9}
    ps = surfaces.PerturbedSurface(base, coeffs, r_norm=R_NORM)
    rng = np.random.default_rng(1)
    pts = np.column_stack([np.zeros(500),
                           rng.uniform(-8e-3, 8e-3, 500),
                           rng.uniform(-8e-3, 8e-3, 500)])
    s, gu, gv = ps._sag_and_grad(pts)
    h = 1e-8
    fdu = (ps.sag(pts + h * ps.t1) - ps.sag(pts - h * ps.t1)) / (2 * h)
    fdv = (ps.sag(pts + h * ps.t2) - ps.sag(pts - h * ps.t2)) / (2 * h)
    assert np.abs(gu - fdu).max() < 1e-8
    assert np.abs(gv - fdv).max() < 1e-8


def test_perturbed_surface_intersect_lands_on_displaced_surface():
    base = surfaces.Plane([0, 0, 0], [1, 0, 0])
    ps = surfaces.PerturbedSurface(base, {4: 45.7e-9, 6: 80e-9}, r_norm=R_NORM)
    rng = np.random.default_rng(2)
    ys = rng.uniform(-8e-3, 8e-3, 50)
    o = np.column_stack([np.full(ys.size, 0.02), ys, np.zeros(ys.size)])
    d = np.tile([-1.0, 0.0, 0.0], (ys.size, 1)).astype(float)
    t, valid = ps.intersect(o, d)
    p = o + t[:, 0:1] * d
    # the hit's axial coordinate equals the figure sag there (surface displaced
    # by delta along the +x base normal)
    assert np.abs(p[:, 0] - ps.sag(p)).max() < 1e-12


# --------------------------------------------------------------------------- #
# GATE (a): lambda/4 PV defocus (Z4) shifts the reflected focus analytically
# --------------------------------------------------------------------------- #
def test_gate_a_defocus_focus_shift():
    base = surfaces.Plane([0, 0, 0], [1, 0, 0])
    c4 = (LAM / 4.0) / (2.0 * math.sqrt(3.0))     # RMS coeff of lambda/4 PV
    ps = surfaces.PerturbedSurface(base, {4: c4}, r_norm=R_NORM)
    ys = np.linspace(-8e-3, 8e-3, 41)
    ys = ys[np.abs(ys) > 1e-4]
    o = np.column_stack([np.full(ys.size, -0.05), ys, np.zeros(ys.size)])
    d = np.tile([1.0, 0.0, 0.0], (ys.size, 1)).astype(float)
    t, _ = ps.intersect(o, d)
    p = o + t[:, 0:1] * d
    dout = _reflect(d, ps.normal(p))
    s = -p[:, 1] / dout[:, 1]                       # cross y=0
    x_focus = np.abs(p[:, 0] + s * dout[:, 0])
    # sag ~ a r^2 with a = 2 sqrt(3) c4 / r_norm^2 (Z4 quadratic term);
    # a flat mirror + that curvature reflects a collimated beam to f = 1/(4a)
    a = 2.0 * math.sqrt(3.0) * c4 / R_NORM ** 2
    f_analytic = 1.0 / (4.0 * a)
    assert np.std(x_focus) / np.mean(x_focus) < 1e-6      # a real point focus
    assert abs(np.mean(x_focus) - f_analytic) / f_analytic < 1e-3


# --------------------------------------------------------------------------- #
# GATE (b): astigmatism (Z6) -> two orthogonal, opposite-sign line foci
# --------------------------------------------------------------------------- #
def test_gate_b_astigmatism_line_foci():
    base = surfaces.Plane([0, 0, 0], [1, 0, 0])
    c6 = 300e-9
    ps = surfaces.PerturbedSurface(base, {6: c6}, r_norm=R_NORM)
    g = np.linspace(-8e-3, 8e-3, 25)
    g = g[np.abs(g) > 1e-4]
    Y, Z = np.meshgrid(g, g)
    Y = Y.ravel(); Z = Z.ravel()
    o = np.column_stack([np.full(Y.size, -0.05), Y, Z])
    d = np.tile([1.0, 0.0, 0.0], (Y.size, 1)).astype(float)
    t, _ = ps.intersect(o, d)
    p = o + t[:, 0:1] * d
    dout = _reflect(d, ps.normal(p))
    with np.errstate(divide="ignore", invalid="ignore"):
        fy = p[:, 0] - p[:, 1] / dout[:, 1] * dout[:, 0]     # meridional focus
        fz = p[:, 0] - p[:, 2] / dout[:, 2] * dout[:, 0]     # sagittal focus
    fy = np.nanmedian(fy)
    fz = np.nanmedian(fz)
    # the two line foci sit on opposite sides (one converging, one diverging),
    # equal in magnitude for balanced Z6
    assert fy * fz < 0.0
    assert abs(abs(fy) - abs(fz)) / abs(fy) < 1e-6
    # magnitude vs analytic: sag = A(y^2 - z^2), A = sqrt(6) c6 / r_norm^2
    A = math.sqrt(6.0) * c6 / R_NORM ** 2
    assert abs(abs(fy) - 1.0 / (4.0 * A)) / (1.0 / (4.0 * A)) < 1e-3


# --------------------------------------------------------------------------- #
# GATE (c): Marechal Strehl matches the direct PSF-peak Strehl (small RMS)
# --------------------------------------------------------------------------- #
def test_gate_c_strehl_marechal_matches_direct():
    base = surfaces.Plane([0, 0, 0], [1, 0, 0])
    # SURFACE RMS lambda/32 -> reflected WAVEFRONT RMS ~ lambda/16 (< lambda/14,
    # inside the Marechal small-aberration regime the gate probes)
    rms = LAM / 32.0
    ps = surfaces.PerturbedSurface(base, {11: rms}, r_norm=R_NORM)
    g = np.linspace(-R_NORM * 0.999, R_NORM * 0.999, 80)
    Y, Z = np.meshgrid(g, g)
    m = np.hypot(Y, Z).ravel() <= R_NORM
    pts = np.column_stack([np.zeros(int(m.sum())), Y.ravel()[m], Z.ravel()[m]])
    W = 2.0 * ps.sag(pts)                            # reflected wavefront = 2*sag
    sigma = float(np.sqrt(np.mean((W - W.mean()) ** 2)))
    assert sigma / LAM < 1.0 / 14.0
    strehl_mar = analysis.strehl_marechal(sigma, LAM)
    k = 2.0 * np.pi / LAM
    strehl_direct = float(abs(np.mean(np.exp(1j * k * W))) ** 2)
    assert abs(strehl_mar - strehl_direct) / strehl_direct < 0.10


# --------------------------------------------------------------------------- #
# registry loader
# --------------------------------------------------------------------------- #
def test_figures_registry_loads_and_validates():
    figs = optprops.load_figures(csv_path=FIG_CSV)
    assert "fig_lambda4_defocus_633" in figs
    entry = figs["fig_lambda4_defocus_633"]
    assert entry["coeffs"] == {4: pytest.approx(45.7e-9)}
    assert entry["r_norm_m"] == pytest.approx(12.5e-3)
    assert entry["reference"]
    # every entry: Noll j>=2, positive r_norm, at least one coeff
    for name, e in figs.items():
        assert e["r_norm_m"] > 0
        assert e["coeffs"]
        assert all(j >= 2 for j in e["coeffs"]), name


def test_figures_registry_rejects_piston(tmp_path):
    bad = tmp_path / "bad.miefig"
    bad.write_text("name,coeffs,r_norm_mm,reference\n"
                   "bad,1:100,12.5,x\n")
    with pytest.raises(Exception):
        optprops.load_figures(csv_path=bad)


# --------------------------------------------------------------------------- #
# engine routing: figure_error is Python-only
# --------------------------------------------------------------------------- #
def test_figure_error_token_is_unported():
    assert "figure_error" not in cengine.PORTED
    assert "surface:perturbedsurface" not in cengine.PORTED


def test_detect_features_emits_figure_error_for_perturbed_surface():
    base = surfaces.Plane([0, 0, 0], [1, 0, 0])
    ps = surfaces.PerturbedSurface(base, {4: 45.7e-9}, r_norm=R_NORM)
    face = types.SimpleNamespace(surface=ps)
    scene = types.SimpleNamespace(
        faces=[face], sources=[], bodies=[], gratings={}, roughness={},
        scatter={}, face_coatings={}, extra_detector_faces=[],
        detector_faces={}, matdb=None, temperature_c=None, ambient=None)
    args = types.SimpleNamespace(
        biref_approx=False, particles=None, particle_threshold=None,
        ray_differentials=False, export_rays=False, ghost_analysis=False,
        pol_transport=False, viz_pattern=None, save_fields=False,
        gdd_budget=False, rough_fresnel=None, importance_scatter=False,
        time_products=(), engine="auto")
    feats = cengine.detect_features(args, scene)
    assert "figure_error" in feats
    assert "surface:perturbedsurface" in feats
    # and such a scene can never route to C
    assert feats - cengine.PORTED
