# =============================================================================
# test_birefringence.py — closed-form validation of birefringence.py:
# uniaxial double refraction (n_e(theta), eigenbasis, refract_in/out, walk-off).
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_birefringence.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer import fresnel as fr            # noqa: E402
from raytracer import birefringence as bi      # noqa: E402

# calcite @ 590 nm (negative uniaxial), quartz (positive uniaxial)
CAL_NO, CAL_NE = 1.658, 1.486
QTZ_NO, QTZ_NE = 1.5443, 1.5534


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# phase index
# ---------------------------------------------------------------------------
def test_n_e_theta_endpoints_and_monotonic():
    # cos = 1 -> n_o ; cos = 0 -> n_e ; strictly monotonic between
    assert abs(bi.n_e_theta(1.0, CAL_NO, CAL_NE) - CAL_NO) < 1e-12
    assert abs(bi.n_e_theta(0.0, CAL_NO, CAL_NE) - CAL_NE) < 1e-12
    assert abs(bi.n_e_theta(-1.0, CAL_NO, CAL_NE) - CAL_NO) < 1e-12
    cos = np.cos(np.linspace(0.0, np.pi / 2, 200))
    n = bi.n_e_theta(cos, CAL_NO, CAL_NE)
    # calcite n_o > n_e, so n decreases as theta grows (cos decreases)
    assert np.all(np.diff(n) < 0)                       # strictly monotone
    assert n.max() <= CAL_NO + 1e-12 and n.min() >= CAL_NE - 1e-12


def test_n_e_theta_dispersion_arrays():
    cos = np.array([1.0, 0.0, 0.5])
    no = np.array([1.66, 1.658, 1.655])
    ne = np.array([1.49, 1.486, 1.484])
    n = bi.n_e_theta(cos, no, ne)
    assert abs(n[0] - no[0]) < 1e-12
    assert abs(n[1] - ne[1]) < 1e-12


# ---------------------------------------------------------------------------
# eigenbasis
# ---------------------------------------------------------------------------
def test_eigenbasis_orthonormal_and_degenerate():
    rng = np.random.default_rng(3)
    k = _unit(rng.normal(size=(200, 3)))
    c = _unit(rng.normal(size=(200, 3)))
    eo, ee = bi.eigenbasis(k, c)
    for v in (eo, ee):
        assert np.max(np.abs(np.linalg.norm(v, axis=-1) - 1)) < 1e-12
    assert np.max(np.abs(np.sum(eo * k, axis=-1))) < 1e-12   # eo _|_ k
    assert np.max(np.abs(np.sum(ee * k, axis=-1))) < 1e-12   # ee _|_ k
    assert np.max(np.abs(np.sum(eo * ee, axis=-1))) < 1e-12  # eo _|_ ee
    # eo in-plane-normal: eo _|_ c (perpendicular to the (k,c) plane)
    assert np.max(np.abs(np.sum(eo * c, axis=-1))) < 1e-12
    # degenerate k || c: deterministic, finite, orthonormal to k
    kc = np.tile([0.0, 0.0, 1.0], (2, 1))
    cc = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    eo2, ee2 = bi.eigenbasis(kc, cc)
    assert np.all(np.isfinite(eo2)) and np.all(np.isfinite(ee2))
    assert np.max(np.abs(np.linalg.norm(eo2, axis=-1) - 1)) < 1e-12
    assert np.max(np.abs(np.sum(eo2 * kc, axis=-1))) < 1e-12
    eo2b, _ = bi.eigenbasis(kc, cc)
    assert np.array_equal(eo2, eo2b)                     # deterministic


# ---------------------------------------------------------------------------
# isotropic limit n_e == n_o reduces to fresnel
# ---------------------------------------------------------------------------
def test_isotropic_limit_matches_fresnel():
    rng = np.random.default_rng(7)
    N = 300
    d = _unit(rng.normal(size=(N, 3)))
    nh = _unit(rng.normal(size=(N, 3)))
    # force cos_i > 0 (n_hat against the ray)
    flip = np.sum(d * nh, axis=-1) > 0
    nh[flip] *= -1.0
    c = _unit(rng.normal(size=(N, 3)))
    n1 = 1.0
    ncrys = 1.5                                          # n_e == n_o
    cos_i = -np.sum(d * nh, axis=-1)
    ref = fr.refract_dir(d, nh, cos_i, np.full(N, n1), np.full(N, ncrys))
    out = bi.refract_in(d, nh, c, n1, ncrys, ncrys)
    assert np.max(np.abs(out["k_o"] - ref)) < 1e-12
    assert np.max(np.abs(out["k_e"] - ref)) < 1e-12     # e == o in iso limit
    assert np.max(np.abs(out["s_e"] - out["k_e"])) < 1e-12   # rho = 0
    assert np.max(np.abs(out["n_phase_e"] - ncrys)) < 1e-12
    assert np.max(np.abs(out["n_ray_e"] - ncrys)) < 1e-12
    assert not out["tir_o"].any() and not out["tir_e"].any()
    assert np.max(np.abs(bi.walkoff_angle(
        np.sum(out["k_e"] * c, axis=-1), ncrys, ncrys))) < 1e-12


# ---------------------------------------------------------------------------
# walk-off magnitude and sign
# ---------------------------------------------------------------------------
def test_calcite_walkoff_45deg():
    rho = bi.walkoff_angle(np.cos(np.deg2rad(45.0)), CAL_NO, CAL_NE)
    assert abs(np.rad2deg(rho) - 6.23) < 0.05          # Hecht / Saleh-Teich
    assert rho > 0                                     # negative uniaxial sign


def test_quartz_positive_uniaxial_sign():
    cos45 = np.cos(np.deg2rad(45.0))
    rho_q = bi.walkoff_angle(cos45, QTZ_NO, QTZ_NE)
    rho_c = bi.walkoff_angle(cos45, CAL_NO, CAL_NE)
    assert rho_q * rho_c < 0                            # OPPOSITE sign
    theta = np.arccos(cos45)
    expect = np.arctan((QTZ_NO ** 2 / QTZ_NE ** 2) * np.tan(theta)) - theta
    assert abs(rho_q - expect) < 1e-2 * abs(expect)     # within 1%
    assert 0.28 <= abs(np.rad2deg(rho_q)) <= 0.6


# ---------------------------------------------------------------------------
# normal-surface consistency and tangential continuity
# ---------------------------------------------------------------------------
def _random_entry(rng, N):
    d = _unit(rng.normal(size=(N, 3)))
    nh = _unit(rng.normal(size=(N, 3)))
    flip = np.sum(d * nh, axis=-1) > 0
    nh[flip] *= -1.0
    c = _unit(rng.normal(size=(N, 3)))
    return d, nh, c


def test_normal_surface_residual():
    # K_e = n_phase_e * k_e must lie on (K.c)^2/n_o^2 + |Kperp|^2/n_e^2 = 1
    rng = np.random.default_rng(11)
    d, nh, c = _random_entry(rng, 400)
    out = bi.refract_in(d, nh, c, 1.0, CAL_NO, CAL_NE)
    ok = ~out["tir_e"]
    K = out["n_phase_e"][:, None] * out["k_e"]
    Kc = np.sum(K * c, axis=-1)
    Kp2 = np.sum(K * K, axis=-1) - Kc ** 2
    res = Kc ** 2 / CAL_NO ** 2 + Kp2 / CAL_NE ** 2 - 1.0
    assert np.max(np.abs(res[ok])) < 1e-12
    # and |K_e| equals the phase-index formula from its own angle
    n_from_angle = bi.n_e_theta(np.sum(out["k_e"] * c, axis=-1),
                                CAL_NO, CAL_NE)
    assert np.max(np.abs(out["n_phase_e"][ok] - n_from_angle[ok])) < 1e-12


def test_tangential_wavevector_continuity():
    rng = np.random.default_rng(13)
    d, nh, c = _random_entry(rng, 400)
    n1 = 1.0
    out = bi.refract_in(d, nh, c, n1, CAL_NO, CAL_NE)
    ok = ~out["tir_e"]
    # (n1 d_in - n_phase_e k_e) must have zero tangential component
    diff = n1 * d - out["n_phase_e"][:, None] * out["k_e"]
    tang = diff - np.sum(diff * nh, axis=-1)[:, None] * nh
    assert np.max(np.linalg.norm(tang[ok], axis=-1)) < 1e-12
    # ordinary too
    diff_o = n1 * d - CAL_NO * out["k_o"]
    tang_o = diff_o - np.sum(diff_o * nh, axis=-1)[:, None] * nh
    assert np.max(np.linalg.norm(tang_o[~out["tir_o"]], axis=-1)) < 1e-12


# ---------------------------------------------------------------------------
# refract_out reduces to fresnel for the ordinary mode
# ---------------------------------------------------------------------------
def test_refract_out_ordinary_matches_fresnel():
    rng = np.random.default_rng(17)
    N = 300
    k = _unit(rng.normal(size=(N, 3)))
    nh = _unit(rng.normal(size=(N, 3)))
    flip = np.sum(k * nh, axis=-1) > 0
    nh[flip] *= -1.0
    c = _unit(rng.normal(size=(N, 3)))
    n_o, n2 = CAL_NO, 1.0
    cos_i = -np.sum(k * nh, axis=-1)
    ref = fr.refract_dir(k, nh, cos_i, np.full(N, n_o), np.full(N, n2))
    tir_ref = fr.is_tir(cos_i, np.full(N, n_o), np.full(N, n2))
    d_out, tir = bi.refract_out(k, np.zeros(N, bool), nh, c,
                                n_o, CAL_NE, n2)
    ok = ~tir_ref
    assert np.array_equal(tir, tir_ref)
    assert np.max(np.abs(d_out[ok] - ref[ok])) < 1e-12


# ---------------------------------------------------------------------------
# beam-displacer: normal incidence, c at 45deg -> e-ray walk-off, k straight
# ---------------------------------------------------------------------------
def test_beam_displacer_normal_incidence():
    d = np.array([[0.0, 0.0, -1.0]])
    nh = np.array([[0.0, 0.0, 1.0]])                    # against the ray
    c = _unit(np.array([[np.sin(np.deg2rad(45)), 0.0,
                         np.cos(np.deg2rad(45))]]))
    out = bi.refract_in(d, nh, c, 1.0, CAL_NO, CAL_NE)
    # both wavevectors travel straight down (normal incidence, k_t = 0)
    assert np.allclose(out["k_o"][0], [0, 0, -1], atol=1e-12)
    assert np.allclose(out["k_e"][0], [0, 0, -1], atol=1e-12)
    # e-RAY walks off by rho ~ 6.23deg; lateral / axial = tan(rho)
    s_e = out["s_e"][0]
    lateral = np.hypot(s_e[0], s_e[1])
    rho = np.arctan2(lateral, -s_e[2])
    assert abs(np.rad2deg(rho) - 6.23) < 0.05
    # walk-off toward the c-axis in-plane projection (+x here)
    assert s_e[0] > 0
    # slab of thickness h: displacement = h * tan(rho) within 0.5%
    h = 10.0
    disp = h * lateral / (-s_e[2])
    assert abs(disp - h * np.tan(np.deg2rad(6.23))) < 5e-3 * h


# ---------------------------------------------------------------------------
# Wollaston-relevant: normal incidence, c IN the interface plane (theta = 90)
# ---------------------------------------------------------------------------
def test_wollaston_c_in_interface_plane():
    # normal incidence + c _|_ n_hat  ->  k _|_ c (theta = 90deg): both waves
    # travel straight, the e phase index is exactly n_e, walk-off vanishes at
    # this principal direction (the o/e SPLIT in a Wollaston comes from the
    # n_o != n_e index difference at the internal wedge, not walk-off here).
    d = np.array([[0.0, 0.0, -1.0]])
    nh = np.array([[0.0, 0.0, 1.0]])
    c = np.array([[1.0, 0.0, 0.0]])                     # in the interface plane
    out = bi.refract_in(d, nh, c, 1.0, CAL_NO, CAL_NE)
    assert np.allclose(out["k_o"][0], [0, 0, -1], atol=1e-12)
    assert np.allclose(out["k_e"][0], [0, 0, -1], atol=1e-12)
    assert abs(out["n_phase_e"][0] - CAL_NE) < 1e-12    # theta = 90 -> n_e
    assert abs(out["n_phase_o"][0] - CAL_NO) < 1e-12
    assert np.allclose(out["s_e"][0], [0, 0, -1], atol=1e-12)   # rho = 0
    assert abs(out["n_ray_e"][0] - CAL_NE) < 1e-12


# ---------------------------------------------------------------------------
# plane-parallel slab round trip: exit directions parallel to entry
# ---------------------------------------------------------------------------
def test_slab_roundtrip_parallel_and_displaced():
    # slab faces _|_ z, front at z = 0, back at z = -h. Oblique entry from air.
    h = 8.0
    theta_i = np.deg2rad(25.0)
    d = np.array([[np.sin(theta_i), 0.0, -np.cos(theta_i)]])
    nh_front = np.array([[0.0, 0.0, 1.0]])              # against entry ray
    nh_back = np.array([[0.0, 0.0, 1.0]])               # against internal ray
    c = _unit(np.array([[0.3, 0.5, 0.8]]))              # arbitrary c-axis
    n_air = 1.0
    out = bi.refract_in(d, nh_front, c, n_air, CAL_NO, CAL_NE)
    assert not out["tir_o"].any() and not out["tir_e"].any()

    # ordinary: wavevector == ray, exit direction parallel to entry (1e-12)
    d_o, tir_o = bi.refract_out(out["k_o"], np.zeros(1, bool),
                                nh_back, c, CAL_NO, CAL_NE, n_air)
    assert not tir_o.any()
    assert np.max(np.abs(d_o[0] - d[0])) < 1e-12

    # extraordinary: exit direction parallel to entry (1e-9)
    d_e, tir_e = bi.refract_out(out["k_e"], np.ones(1, bool),
                                nh_back, c, CAL_NO, CAL_NE, n_air)
    assert not tir_e.any()
    assert np.max(np.abs(d_e[0] - d[0])) < 1e-9

    # lateral displacement between the o and e exit points is nonzero: the
    # o-ray travels along k_o, the e-ray along s_e (walked off)
    to = -h / out["k_o"][0, 2]
    te = -h / out["s_e"][0, 2]
    p_o = to * out["k_o"][0]
    p_e = te * out["s_e"][0]
    lateral = np.hypot(*(p_e[:2] - p_o[:2]))
    assert lateral > 1e-3


# ---------------------------------------------------------------------------
# TIR of an internal e-ray at the crystal -> air exit
# ---------------------------------------------------------------------------
def test_internal_e_ray_tir():
    c = _unit(np.array([[0.2, 0.3, 0.9], [0.2, 0.3, 0.9]]))
    nh = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    # steep internal wavevector (60deg to normal) -> beyond critical for air;
    # shallow (10deg) -> transmits
    for ang, expect_tir in ((60.0, True), (10.0, False)):
        a = np.deg2rad(ang)
        k = np.array([[np.sin(a), 0.0, -np.cos(a)]])
        d_out, tir = bi.refract_out(k, np.ones(1, bool),
                                    nh[:1], c[:1], CAL_NO, CAL_NE, 1.0)
        assert tir[0] == expect_tir
        if not expect_tir:
            assert np.all(np.isfinite(d_out))
            assert abs(np.linalg.norm(d_out[0]) - 1.0) < 1e-12


def test_ray_k_roundtrip():
    """k_from_ray inverts ray_from_k exactly for random wavevectors, for
    both calcite (negative) and quartz (positive) indices."""
    from raytracer.birefringence import ray_from_k, k_from_ray, n_e_theta
    rng = np.random.default_rng(11)
    k = rng.normal(size=(300, 3))
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    c = np.array([0.36, -0.48, 0.8])
    for n_o, n_e in ((1.658, 1.486), (1.5443, 1.5534)):
        s, n_phase, n_ray = ray_from_k(k, c, n_o, n_e)
        k2 = k_from_ray(s, c, n_o, n_e)
        assert np.max(np.linalg.norm(k2 - k, axis=-1)) < 1e-12
        # n_ray = n_phase*cos(rho) <= n_phase, equality iff no walk-off
        assert np.all(n_ray <= n_phase + 1e-15)
        cos_kc = np.abs(k @ c)
        expect = n_e_theta(k @ c, n_o, n_e)
        assert np.allclose(n_phase, expect, atol=1e-12)
