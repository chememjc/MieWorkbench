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


# ===========================================================================
# EXACT Lekner-1991 interface amplitudes (uniaxial_interface_in/_out).
# P6: replaces the effective-index approximation with the exact boundary-
# value-problem solution. Oracles below are the P6 acceptance gates.
# ===========================================================================
QTZ_NO, QTZ_NE = 1.5443, 1.5534   # quartz (positive uniaxial); calcite above


def _interface_geom(thetas_deg, alpha_deg, phi_deg):
    """Interface normal +z; plane of incidence = x-z; incident ray travels
    into the interface (d_z < 0). Optic axis at polar alpha from the normal,
    azimuth phi from the plane of incidence (+x)."""
    th = np.deg2rad(np.atleast_1d(thetas_deg).astype(float))
    n = th.shape[0]
    nh = np.tile([0, 0, 1.0], (n, 1))
    d = np.stack([np.sin(th), np.zeros(n), -np.cos(th)], axis=1)
    al, ph = np.deg2rad(alpha_deg), np.deg2rad(phi_deg)
    c = np.tile([np.sin(al) * np.cos(ph), np.sin(al) * np.sin(ph),
                 np.cos(al)], (n, 1))
    return d, nh, c


def test_uniaxial_interface_isotropic_reduction():
    """ORACLE 1: n_o = n_e = n -> the exact amplitudes reduce to Fresnel to
    1e-12 over 200 angles. Reflection Jones equals rs/rp with zero cross
    terms; per-input transmitted power equals the Fresnel Ts/Tp."""
    N = 200
    thetas = np.linspace(0.1, 80, N)
    d, nh, c = _interface_geom(thetas, 37.0, 41.0)
    n1 = np.full(N, 1.0)
    ncr = np.full(N, 1.55)
    amp = bi.uniaxial_interface_in(d, nh, c, n1, ncr, ncr)
    cos_i = -np.sum(d * nh, axis=1)
    rs, rp, ts, tp, ct = fr.fresnel_coeffs(cos_i, n1.astype(complex),
                                           ncr.astype(complex))
    Rs, Rp, Ts, Tp = fr.power_coeffs(rs, rp, ts, tp, cos_i, ct,
                                     n1.astype(complex), ncr.astype(complex))
    assert np.max(np.abs(amp["rss"] - rs)) < 1e-12
    assert np.max(np.abs(amp["rpp"] - rp)) < 1e-12
    assert np.max(np.abs(amp["rsp"])) < 1e-12
    assert np.max(np.abs(amp["rps"])) < 1e-12
    Ts_bvp = np.abs(amp["tos"]) ** 2 + np.abs(amp["tes"]) ** 2
    Tp_bvp = np.abs(amp["top"]) ** 2 + np.abs(amp["tep"]) ** 2
    assert np.max(np.abs(Ts_bvp - Ts)) < 1e-12
    assert np.max(np.abs(Tp_bvp - Tp)) < 1e-12


@pytest.mark.parametrize("no,ne", [(CAL_NO, CAL_NE), (QTZ_NO, QTZ_NE)])
def test_uniaxial_interface_energy_closure(no, ne):
    """ORACLE 3: Poynting-flux closure. The flux-normalized 4x4 scattering
    matrix is UNITARY across a dense (theta, alpha, phi) grid for calcite
    (negative) AND quartz (positive) uniaxials: sum of |amplitude|^2 == 1
    per input to 1e-10, and the two input columns are orthogonal.
    Also pins the BRANCH CHOICE: the transmitted-e Poynting flux stays > 0
    (a wrong ellipsoid root would flip its sign and break closure)."""
    worst_norm = 0.0
    worst_orth = 0.0
    for alpha in np.linspace(0, 90, 10):
        for phi in np.linspace(0, 90, 10):
            thetas = np.linspace(0.01, 80, 40)
            d, nh, c = _interface_geom(thetas, alpha, phi)
            n1 = np.full(len(thetas), 1.0)
            a = bi.uniaxial_interface_in(d, nh, c, n1,
                                         np.full(len(thetas), no),
                                         np.full(len(thetas), ne))
            col_s = (np.abs(a["rss"]) ** 2 + np.abs(a["rps"]) ** 2
                     + np.abs(a["tos"]) ** 2 + np.abs(a["tes"]) ** 2)
            col_p = (np.abs(a["rsp"]) ** 2 + np.abs(a["rpp"]) ** 2
                     + np.abs(a["top"]) ** 2 + np.abs(a["tep"]) ** 2)
            orth = (np.conj(a["rss"]) * a["rsp"] + np.conj(a["rps"]) * a["rpp"]
                    + np.conj(a["tos"]) * a["top"]
                    + np.conj(a["tes"]) * a["tep"])
            worst_norm = max(worst_norm, np.max(np.abs(col_s - 1)),
                             np.max(np.abs(col_p - 1)))
            worst_orth = max(worst_orth, np.max(np.abs(orth)))
    assert worst_norm < 1e-10, worst_norm
    assert worst_orth < 1e-10, worst_orth


@pytest.mark.parametrize("no,ne", [(CAL_NO, CAL_NE), (QTZ_NO, QTZ_NE)])
def test_uniaxial_azimuth_parity(no, ne):
    """ORACLE 2 (cross-check vs Lekner, JOSA A 40, 722 (2023)): with the
    optic axis IN the interface plane, r_sp and r_ps are ODD in the axis
    azimuth phi and vanish identically at phi = 0 and 90 deg; the magnitude
    peaks near 45 deg."""
    theta, alpha = 50.0, 90.0        # axis in the surface (Lekner geometry)
    for phi in (0.0, 90.0):
        d, nh, c = _interface_geom([theta], alpha, phi)
        a = bi.uniaxial_interface_in(d, nh, c, np.array([1.0]),
                                     np.array([no]), np.array([ne]))
        assert abs(a["rsp"][0]) < 1e-14
        assert abs(a["rps"][0]) < 1e-14
    for phi in (10.0, 22.5, 30.0, 45.0, 60.0):
        dp, nhp, cp = _interface_geom([theta], alpha, phi)
        dm, nhm, cm = _interface_geom([theta], alpha, -phi)
        ap = bi.uniaxial_interface_in(dp, nhp, cp, np.array([1.0]),
                                      np.array([no]), np.array([ne]))
        am = bi.uniaxial_interface_in(dm, nhm, cm, np.array([1.0]),
                                      np.array([no]), np.array([ne]))
        assert abs(ap["rsp"][0] + am["rsp"][0]) < 1e-14   # odd in phi
        assert abs(ap["rps"][0] + am["rps"][0]) < 1e-14
    grid = np.linspace(0, 90, 19)
    mags = []
    for phi in grid:
        dp, nhp, cp = _interface_geom([theta], alpha, phi)
        ap = bi.uniaxial_interface_in(dp, nhp, cp, np.array([1.0]),
                                      np.array([no]), np.array([ne]))
        mags.append(abs(ap["rsp"][0]))
    assert 40.0 <= grid[int(np.argmax(mags))] <= 55.0


def test_uniaxial_normal_incidence_axis_in_plane():
    """ORACLE 4: at normal incidence with the optic axis in the interface
    plane, the two eigen-reflectivities equal the isotropic Fresnel values
    with n_o and n_e respectively (textbook), to 1e-12, with no cross
    coupling."""
    d, nh, c = _interface_geom([0.0], 90.0, 0.0)   # axis along +x, in plane
    a = bi.uniaxial_interface_in(d, nh, c, np.array([1.0]),
                                 np.array([CAL_NO]), np.array([CAL_NE]))
    r_o = abs((1 - CAL_NO) / (1 + CAL_NO))
    r_e = abs((1 - CAL_NE) / (1 + CAL_NE))
    got = sorted([abs(a["rss"][0]), abs(a["rpp"][0])])
    assert abs(got[0] - r_e) < 1e-12 and abs(got[1] - r_o) < 1e-12
    assert abs(a["rsp"][0]) < 1e-12 and abs(a["rps"][0]) < 1e-12


def test_uniaxial_exit_isotropic_reduction():
    """EXIT ORACLE 1: n_o = n_e -> the exit transmitted power reduces to
    Fresnel. The incident (isotropic) o-mode's E projects onto s/p; the
    exact T_total equals a_s^2*Ts + a_p^2*Tp."""
    N = 100
    thetas = np.linspace(0.1, 30, N)          # sub-critical crystal->air
    th = np.deg2rad(thetas)
    nh = np.tile([0, 0, 1.0], (N, 1))
    kh = np.stack([np.sin(th), 0 * th, -np.cos(th)], axis=1)
    c = np.tile([0.3, 0.5, 0.8], (N, 1))
    ncr, n2 = np.full(N, 1.55), np.full(N, 1.0)
    cos_k = -np.sum(kh * nh, axis=1)
    rs, rp, ts, tp, ct = fr.fresnel_coeffs(cos_k, ncr.astype(complex),
                                           n2.astype(complex))
    _, _, Ts, Tp = fr.power_coeffs(rs, rp, ts, tp, cos_k, ct,
                                   ncr.astype(complex), n2.astype(complex))
    out = bi.uniaxial_interface_out(kh, np.zeros(N, bool), nh, c, ncr, ncr, n2)
    eo, _ = bi.eigenbasis(kh, c)
    s_new, p_new = fr.pol_basis(kh, nh)
    a_s = np.sum(eo * s_new, axis=1)
    a_p = np.sum(eo * p_new, axis=1)
    assert np.max(np.abs(out["T_total"] - (a_s ** 2 * Ts + a_p ** 2 * Tp))) \
        < 1e-12


@pytest.mark.parametrize("no,ne", [(CAL_NO, CAL_NE), (QTZ_NO, QTZ_NE)])
def test_uniaxial_exit_transmittance_physical(no, ne):
    """EXIT: the exact transmitted power fraction is in [0, 1] and full-field
    tangential-E/H continuity (energy conservation) holds; reflectance is
    R = 1 - T_total (the crystal-side incident/reflected interference means
    the self-flux split is not clean -- documented in birefringence.py)."""
    for alpha in np.linspace(0, 90, 7):
        for phi in np.linspace(0, 90, 7):
            thetas = np.linspace(0.01, 25, 20)   # sub-critical internal
            th = np.deg2rad(thetas)
            nh = np.tile([0, 0, 1.0], (20, 1))
            kh = np.stack([np.sin(th), 0 * th, -np.cos(th)], axis=1)
            al, ph = np.deg2rad(alpha), np.deg2rad(phi)
            cc = np.tile([np.sin(al) * np.cos(ph), np.sin(al) * np.sin(ph),
                          np.cos(al)], (20, 1))
            for is_e in (False, True):
                out = bi.uniaxial_interface_out(kh, np.full(20, is_e), nh, cc,
                                                np.full(20, no),
                                                np.full(20, ne),
                                                np.full(20, 1.0))
                ok = ~out["tir"]
                assert np.all(out["T_total"][ok] <= 1.0 + 1e-12)
                assert np.all(out["T_total"][ok] >= -1e-12)


def test_uniaxial_approx_error_finding():
    """ORACLE 6 (a FINDING, not a hard gate): tabulate the reflected-Jones
    error |exact - effective-index| for calcite with the optic axis at 45deg
    polar and phi in {0, 22.5, 45} deg over theta 0..80 deg. engine3 Sec 7.4
    predicts it is smallest at phi=0/90 and largest near phi=45; this pins
    the SHAPE of that prediction. Representative table (s-input reflected
    Jones L2 error), computed by scripts .../scratch val_biref.py:

        phi     theta=10   30       50       70       80
        0.0     0          ~1e-16   ~1e-16   ~1e-16   ~1e-16   (exact == approx)
        22.5    1.6e-3     3.7e-3   4.4e-3   3.8e-3   3.1e-3
        45.0    3.8e-3     8.7e-3   1.07e-2  9.4e-3   7.0e-3
    """
    no, ne = CAL_NO, CAL_NE
    errs = {}
    for phi in (0.0, 22.5, 45.0):
        row = []
        for theta in (10, 30, 50, 70, 80):
            d, nh, c = _interface_geom([theta], 45.0, phi)
            n1 = np.array([1.0])
            res = bi.refract_in(d, nh, c, n1, np.array([no]), np.array([ne]))
            a = bi.uniaxial_interface_in(d, nh, c, n1, np.array([no]),
                                         np.array([ne]), res=res)
            cos_i = -np.sum(d * nh, axis=1)
            s_new, p_new = fr.pol_basis(d, nh)
            eo_i, ee_i = bi.eigenbasis(d, c)
            cs_o = np.sum(eo_i * s_new, axis=-1)
            sn_o = np.sum(eo_i * p_new, axis=-1)
            rs_o, rp_o, _, _, _ = fr.fresnel_coeffs(cos_i, n1.astype(complex),
                                                    np.array([no], complex))
            npe = res["n_phase_e"].astype(complex)
            rs_e, rp_e, _, _, _ = fr.fresnel_coeffs(cos_i, n1.astype(complex),
                                                    npe)
            Eo_i, Ee_i = cs_o, -sn_o                 # channel proj, s-input
            Es_ap = Eo_i * cs_o * rs_o - Ee_i * sn_o * rs_e
            Ep_ap = Eo_i * sn_o * rp_o + Ee_i * cs_o * rp_e
            err = np.hypot(abs(Es_ap[0] - a["rss"][0]),
                           abs(Ep_ap[0] - a["rps"][0]))
            row.append(err)
        errs[phi] = np.array(row)
    # SHAPE assertions (engine3 Sec 7.4 prediction)
    assert np.max(errs[0.0]) < 1e-12                       # phi=0 -> exact
    assert np.max(errs[45.0]) > np.max(errs[22.5]) > 5e-4  # grows toward 45
    assert 1e-3 < np.max(errs[45.0]) < 5e-2                # O(1%) at 45deg
