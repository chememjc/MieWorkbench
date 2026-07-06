# =============================================================================
# test_grating_roughness.py — closed-form validation of grating.py
# (vector grating equation + lamellar efficiency model) and roughness.py
# (TIS specular factor + Beckmann microfacet sampling).
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_grating_roughness.py -v
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer import grating as gr                     # noqa: E402
from raytracer import roughness as rg                   # noqa: E402
from raytracer import fresnel as fr                      # noqa: E402


# ---------------------------------------------------------------------------
# grating.order_directions
# ---------------------------------------------------------------------------
def test_grating_order_angles():
    # Normal incidence, 600 l/mm transmission grating in air, 633nm.
    N = 1
    d = np.array([[0.0, 0.0, -1.0]])
    n_hat = np.array([[0.0, 0.0, 1.0]])
    g_hat = np.array([[1.0, 0.0, 0.0]])
    lam = np.array([633e-9])
    n1 = np.array([1.0])
    n2 = np.array([1.0])
    orders = [-2, -1, 0, 1, 2]

    out = gr.order_directions(d, n_hat, g_hat, 600.0, lam, orders, n1, n2)
    d_period = 1e-3 / 600.0
    for m in orders:
        dirs_t, prop_t, dirs_r, prop_r = out[m]
        s_m = m * lam[0] / d_period
        if abs(s_m) <= 1.0:
            assert prop_t[0]
            theta = np.arcsin(s_m)
            # direction measured from -n_hat (forward, transmitted axis)
            got_theta = np.arcsin(np.clip(dirs_t[0, 0], -1, 1))
            assert abs(got_theta - theta) < 1e-9
        else:
            assert not prop_t[0]

    # 1200 l/mm: m = +-2 must be evanescent (|m*lam/d| > 1)
    out2 = gr.order_directions(d, n_hat, g_hat, 1200.0, lam, [-2, -1, 0, 1, 2],
                                n1, n2)
    d_period2 = 1e-3 / 1200.0
    for m in (-2, 2):
        s_m = m * lam[0] / d_period2
        assert abs(s_m) > 1.0
        _, prop_t, _, _ = out2[m]
        assert not prop_t[0]
    for m in (-1, 0, 1):
        _, prop_t, _, _ = out2[m]
        assert prop_t[0]


def test_grating_m0_is_snell():
    rng = np.random.default_rng(1)
    N = 20
    theta_i = np.deg2rad(30.0)
    d = np.tile([np.sin(theta_i), 0.0, -np.cos(theta_i)], (N, 1))
    n_hat = np.tile([0.0, 0.0, 1.0], (N, 1))
    # arbitrary tangential groove direction (grating eqn at m=0 must not
    # depend on it at all)
    g_hat = np.zeros((N, 3))
    g_hat[:, 0] = rng.normal(size=N)
    g_hat[:, 1] = rng.normal(size=N)
    g_hat /= np.linalg.norm(g_hat, axis=-1, keepdims=True)
    lam = np.full(N, 633e-9)
    n1 = np.full(N, 1.0)
    n2 = np.full(N, 1.5)

    out = gr.order_directions(d, n_hat, g_hat, 600.0, lam, [0], n1, n2)
    dirs_t, prop_t, dirs_r, prop_r = out[0]
    assert np.all(prop_t) and np.all(prop_r)

    cos_i = -np.sum(d * n_hat, axis=-1)
    expect_t = fr.refract_dir(d, n_hat, cos_i, n1, n2)
    expect_r = fr.reflect_dir(d, n_hat)
    assert np.max(np.abs(dirs_t - expect_t)) < 1e-12
    assert np.max(np.abs(dirs_r - expect_r)) < 1e-12


def test_grating_oblique_conical():
    # Conical mount: incident direction has a component ALONG the groove
    # lines (perpendicular to g_hat, in-plane). That component must be
    # identical across every diffraction order.
    n_hat = np.array([[0.0, 0.0, 1.0]])
    g_hat = np.array([[1.0, 0.0, 0.0]])          # periodicity along x
    h_hat = np.cross(n_hat, g_hat)               # groove-line direction (y)
    h_hat /= np.linalg.norm(h_hat, axis=-1, keepdims=True)

    theta = np.deg2rad(20.0)     # tilt off-normal
    conical = np.deg2rad(35.0)   # rotation within tangent plane
    d = (-np.cos(theta) * n_hat
         + np.sin(theta) * (np.cos(conical) * g_hat
                             + np.sin(conical) * h_hat))
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    lam = np.array([532e-9])
    n1 = np.array([1.0])
    n2 = np.array([1.5])
    orders = [-2, -1, 0, 1, 2]
    out = gr.order_directions(d, n_hat, g_hat, 300.0, lam, orders, n1, n2)

    h_comp_t = []
    h_comp_r = []
    for m in orders:
        dirs_t, prop_t, dirs_r, prop_r = out[m]
        h_comp_t.append(np.sum(dirs_t * h_hat, axis=-1)[0])
        h_comp_r.append(np.sum(dirs_r * h_hat, axis=-1)[0])
    h_comp_t = np.array(h_comp_t)
    h_comp_r = np.array(h_comp_r)
    assert np.max(np.abs(h_comp_t - h_comp_t[0])) < 1e-12
    assert np.max(np.abs(h_comp_r - h_comp_r[0])) < 1e-12
    # and reflected along-groove component must equal the incident one
    # (same medium both sides of the reflection)
    d_h = np.sum(d * h_hat, axis=-1)[0]
    assert abs(h_comp_r[0] - d_h) < 1e-12


# ---------------------------------------------------------------------------
# grating.lamellar_efficiencies
# ---------------------------------------------------------------------------
def test_grating_efficiencies():
    orders = list(range(-3, 4))
    eta, total = gr.lamellar_efficiencies(orders, duty=0.5)
    assert abs(eta[1] - 4.0 / np.pi ** 2) < 1e-12
    assert abs(eta[-1] - 4.0 / np.pi ** 2) < 1e-12
    assert eta[0] < 1e-15
    for m in (-2, 2):
        assert eta[m] < 1e-15
    assert abs(eta[3] - 4.0 / (9.0 * np.pi ** 2)) < 1e-12
    assert total < 1.0
    assert abs(total - sum(eta.values())) < 1e-15

    # explicit efficiencies pass-through
    explicit = [0.05, 0.1, 0.05, 0.6, 0.05, 0.1, 0.05]
    eta2, total2 = gr.lamellar_efficiencies(orders, efficiencies=explicit)
    assert eta2 == dict(zip(orders, explicit))
    assert abs(total2 - sum(explicit)) < 1e-15

    # explicit efficiencies summing > 1 must raise
    bad = [0.5] * len(orders)
    with pytest.raises(ValueError):
        gr.lamellar_efficiencies(orders, efficiencies=bad)


def test_energy_partition():
    rng = np.random.default_rng(42)
    orders = list(range(-5, 6))
    for _ in range(50):
        duty = rng.uniform(0.001, 0.999)
        eta, total = gr.lamellar_efficiencies(orders, duty=duty)
        assert all(v >= -1e-15 for v in eta.values())
        assert total <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# roughness.specular_power_factor
# ---------------------------------------------------------------------------
def test_roughness_tis():
    cos_i = np.linspace(0.1, 1.0, 20)
    lam = np.full(20, 633e-9)

    A0 = rg.specular_power_factor(0.0, cos_i, lam)
    assert np.allclose(A0, 1.0)

    sigmas = np.array([0.0, 1e-9, 5e-9, 20e-9, 50e-9])
    for cosv in (0.2, 0.5, 1.0):
        for lamv in (400e-9, 633e-9, 1000e-9):
            A = rg.specular_power_factor(sigmas, cosv, lamv)
            expect = np.exp(-(4 * np.pi * sigmas * cosv / lamv) ** 2)
            assert np.max(np.abs(A - expect)) < 1e-15
            # monotone decreasing in sigma
            assert np.all(np.diff(A) <= 1e-15)


def test_slope_from_sigma_lcorr():
    sigma = 20e-9
    lcorr = 1e-6
    slope = rg.slope_from_sigma_lcorr(sigma, lcorr)
    assert abs(slope - np.sqrt(2.0) * sigma / lcorr) < 1e-18


# ---------------------------------------------------------------------------
# roughness.beckmann_sample
# ---------------------------------------------------------------------------
def test_beckmann_stats():
    rng = np.random.default_rng(7)
    n_hat = np.array([[0.0, 0.0, 1.0]])
    sigma_slope = 0.1
    k = 100_000
    m = rg.beckmann_sample(n_hat, sigma_slope, rng, k)

    assert m.shape == (1, k, 3)
    norms = np.linalg.norm(m, axis=-1)
    assert np.max(np.abs(norms - 1.0)) < 1e-10

    cos_theta = np.sum(m[0] * n_hat[0], axis=-1)
    assert np.all(cos_theta > np.cos(np.deg2rad(89.0)) - 1e-9)

    mean_vec = m[0].mean(axis=0)
    mean_dir = mean_vec / np.linalg.norm(mean_vec)
    assert np.dot(mean_dir, n_hat[0]) > 0.99

    tan_theta = np.sqrt(np.maximum(1.0 - cos_theta ** 2, 0.0)) / cos_theta
    rms_tan = np.sqrt(np.mean(tan_theta ** 2))
    assert abs(rms_tan - sigma_slope) / sigma_slope < 0.03


def test_beckmann_reflect_energy():
    rng = np.random.default_rng(3)
    n_hat = np.array([[0.0, 0.0, 1.0]])
    d = np.array([[0.3, 0.0, -np.sqrt(1 - 0.3 ** 2)]])
    k = 100
    sigma_slope = 0.02   # small -> tightly clustered around specular

    mfacets = rg.beckmann_sample(n_hat, sigma_slope, rng, k)[0]      # (k,3)
    d_rep = np.tile(d, (k, 1))
    r = fr.reflect_dir(d_rep, mfacets)

    norms = np.linalg.norm(r, axis=-1)
    assert np.max(np.abs(norms - 1.0)) < 1e-10

    specular = fr.reflect_dir(d, n_hat)[0]
    cos_to_spec = r @ specular
    # small-sigma microfacets: reflected rays cluster tightly around the
    # macroscopic specular direction
    assert np.mean(cos_to_spec) > 0.99
    assert np.min(cos_to_spec) > 0.9


# ---------------------------------------------------------------------------
# grating.groove_vector
# ---------------------------------------------------------------------------
def test_groove_vector_plane():
    from raytracer.surfaces import Plane

    p = Plane([0, 0, 0], [0, 0, 1])
    N = 5
    n_hat_sample = np.tile(p.n, (N, 1))

    gu = gr.groove_vector(p, "u", n_hat_sample)
    assert np.allclose(gu, np.tile(p.t1, (N, 1)), atol=1e-12)
    gv = gr.groove_vector(p, "v", n_hat_sample)
    assert np.allclose(gv, np.tile(p.t2, (N, 1)), atol=1e-12)

    gxyz = gr.groove_vector(p, "1,1,0", n_hat_sample)
    assert np.allclose(np.linalg.norm(gxyz, axis=-1), 1.0)
    assert np.max(np.abs(np.sum(gxyz * n_hat_sample, axis=-1))) < 1e-12

    with pytest.raises(ValueError):
        gr.groove_vector(p, "0,0,1", n_hat_sample)   # parallel to normal
