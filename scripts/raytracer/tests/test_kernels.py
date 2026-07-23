# =============================================================================
# test_kernels.py — closed-form validation of the physics kernels:
# fresnel.py (Snell/Fresnel/TIR/polarization), thinfilm.py (TMM),
# surfaces.py (analytic intersections + trim containment).
# Run: /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_kernels.py -v
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer import fresnel as fr                     # noqa: E402
from raytracer import thinfilm as tf                    # noqa: E402
from raytracer.surfaces import (Plane, Sphere, Cylinder, Cone, Torus,
                                TrimPolygon, AnalyticFace)  # noqa: E402


# ---------------------------------------------------------------------------
# Fresnel
# ---------------------------------------------------------------------------
def test_snell_refraction_angles():
    n1, n2 = 1.0, 1.5
    theta_i = np.linspace(0.01, 1.4, 50)
    d = np.stack([np.sin(theta_i), -np.cos(theta_i),
                  np.zeros_like(theta_i)], axis=-1)
    n_hat = np.tile([0.0, 1.0, 0.0], (50, 1))          # into incident medium
    cos_i = -np.sum(d * n_hat, axis=-1)
    t = fr.refract_dir(d, n_hat, cos_i, np.full(50, n1), np.full(50, n2))
    theta_t = np.arccos(np.clip(-t @ np.array([0.0, 1.0, 0.0]), -1, 1))
    expect = np.arcsin(n1 / n2 * np.sin(theta_i))
    assert np.max(np.abs(theta_t - expect)) < 1e-12


def test_fresnel_energy_conservation():
    n1 = np.full(200, 1.0)
    n2 = np.full(200, 1.7)
    cos_i = np.cos(np.linspace(0.0, np.pi / 2 * 0.999, 200))
    rs, rp, ts, tp, ct = fr.fresnel_coeffs(cos_i, n1, n2)
    Rs, Rp, Ts, Tp = fr.power_coeffs(rs, rp, ts, tp, cos_i, ct, n1, n2)
    assert np.max(np.abs(Rs + Ts - 1.0)) < 1e-12
    assert np.max(np.abs(Rp + Tp - 1.0)) < 1e-12


def test_brewster():
    n1, n2 = 1.0, 1.5
    theta_b = np.arctan(n2 / n1)
    cos_i = np.array([np.cos(theta_b)])
    rs, rp, ts, tp, ct = fr.fresnel_coeffs(cos_i, np.array([n1]),
                                           np.array([n2]))
    assert abs(rp[0]) ** 2 < 1e-10
    # normal incidence sanity: R = ((n2-n1)/(n2+n1))^2 = 0.04
    rs0, _, _, _, _ = fr.fresnel_coeffs(np.array([1.0]), np.array([n1]),
                                        np.array([n2]))
    assert abs(abs(rs0[0]) ** 2 - 0.04) < 1e-12


def test_tir_magnitude_and_rhomb_phase():
    # glass -> air, past the critical angle
    n1v, n2v = 1.51, 1.0
    theta = np.deg2rad(54.6)                     # Fresnel-rhomb angle
    cos_i = np.array([np.cos(theta)])
    rs, rp, ts, tp, ct = fr.fresnel_coeffs(cos_i, np.array([n1v]),
                                           np.array([n2v]))
    assert abs(abs(rs[0]) - 1.0) < 1e-12
    assert abs(abs(rp[0]) - 1.0) < 1e-12
    # relative phase delta_p - delta_s obeys the rhomb formula
    # tan(D/2) = cos(t) sqrt(sin^2 t - n^2) / sin^2 t,  n = n2/n1
    n = n2v / n1v
    st, cth = np.sin(theta), np.cos(theta)
    expect = 2.0 * np.arctan(cth * np.sqrt(st ** 2 - n ** 2) / st ** 2)
    got = np.angle(rp[0]) - np.angle(rs[0])
    got = (got + np.pi) % (2 * np.pi) - np.pi
    # sign convention may flip overall phase; compare magnitudes
    assert abs(abs(got) - abs(expect)) < 1e-10 or \
        abs(abs(abs(got) - np.pi) - abs(expect)) < 1e-10


def test_metal_branch_absorbing():
    # aluminum-ish at 633 nm: n2 = 1.4 + 7.6i, from air
    cos_i = np.cos(np.linspace(0, 1.55, 40))
    n1 = np.ones(40)
    n2 = np.full(40, 1.4 + 7.6j)
    rs, rp, ts, tp, ct = fr.fresnel_coeffs(cos_i, n1, n2)
    # decaying wave into the metal
    assert np.all(np.imag(n2 * ct) >= 0)
    R_avg = 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)
    assert np.all(R_avg > 0.7) and np.all(R_avg <= 1.0)
    # normal-incidence closed form: R = |n2-1|^2/|n2+1|^2
    r0 = abs((1.4 + 7.6j - 1) / (1.4 + 7.6j + 1)) ** 2
    assert abs(R_avg[0] - r0) < 1e-10


def test_reflect_dir_and_pol_basis_unitary():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(100, 3))
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    n = rng.normal(size=(100, 3))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    r = fr.reflect_dir(d, n)
    assert np.max(np.abs(np.linalg.norm(r, axis=-1) - 1)) < 1e-12
    # angle of incidence == angle of reflection: r.n = -(d.n)
    assert np.max(np.abs(np.sum(d * n, axis=-1) + np.sum(r * n, axis=-1))) \
        < 1e-12
    s, p = fr.pol_basis(d, n)
    for v in (s, p):
        assert np.max(np.abs(np.linalg.norm(v, axis=-1) - 1)) < 1e-12
        assert np.max(np.abs(np.sum(v * d, axis=-1))) < 1e-12
    # rotation to a second random basis conserves |E|^2
    s2, p2 = fr.pol_basis(d, rng.normal(size=(100, 3)))
    Es = rng.normal(size=100) + 1j * rng.normal(size=100)
    Ep = rng.normal(size=100) + 1j * rng.normal(size=100)
    Es2, Ep2 = fr.rotate_jones(Es, Ep, s, p, s2, p2)
    assert np.max(np.abs((np.abs(Es2) ** 2 + np.abs(Ep2) ** 2)
                         - (np.abs(Es) ** 2 + np.abs(Ep) ** 2))) < 1e-10


# ---------------------------------------------------------------------------
# Thin films (TMM)
# ---------------------------------------------------------------------------
def _mgf2_qw_R(n_f, n_s):
    return ((n_s - n_f ** 2) / (n_s + n_f ** 2)) ** 2


def test_tmm_zero_layers_is_bare_fresnel():
    lam = np.full(30, 550e-9)
    cos_i = np.cos(np.linspace(0, 1.5, 30))
    n1 = np.ones(30)
    n2 = np.full(30, 1.52 + 0j)
    rs0, rp0, ts0, tp0, ct = fr.fresnel_coeffs(cos_i, n1, n2)
    rs, rp, ts, tp, etas = tf.tmm_coeffs(lam, cos_i, n1, n2, [], [])
    assert np.max(np.abs(rs - rs0)) < 1e-12
    assert np.max(np.abs(rp - rp0)) < 1e-12
    Rs, Rp, Ts, Tp = tf.tmm_power(rs, rp, ts, tp, etas)
    Rs0, Rp0, Ts0, Tp0 = fr.power_coeffs(rs0, rp0, ts0, tp0, cos_i, ct,
                                         n1, n2)
    assert np.max(np.abs(Ts - Ts0)) < 1e-12
    assert np.max(np.abs(Tp - Tp0)) < 1e-12


def test_tmm_quarterwave_mgf2_on_bk7():
    n_f, n_s = 1.3777, 1.5185          # MgF2, BK7 at 550 nm
    lam0 = 550e-9
    d = lam0 / (4 * n_f)
    lam = np.array([lam0])
    cos_i = np.array([1.0])
    rs, rp, ts, tp, etas = tf.tmm_coeffs(
        lam, cos_i, np.array([1.0]), np.array([n_s + 0j]),
        [np.array([n_f + 0j])], [d])
    R = abs(rs[0]) ** 2
    expect = _mgf2_qw_R(n_f, n_s)      # ~1.26e-2
    assert abs(R - expect) < 1e-6
    assert abs(R - 0.0126) < 1e-3
    # and it beats the bare interface (~4.2%)
    bare = ((n_s - 1) / (n_s + 1)) ** 2
    assert R < bare / 3


def test_tmm_halfwave_absentee():
    # half-wave layer at design wavelength: R equals the bare substrate
    n_f, n_s = 1.38, 1.52
    lam0 = 550e-9
    d = lam0 / (2 * n_f)
    rs, rp, ts, tp, etas = tf.tmm_coeffs(
        np.array([lam0]), np.array([1.0]), np.array([1.0]),
        np.array([n_s + 0j]), [np.array([n_f + 0j])], [d])
    bare = ((n_s - 1) / (n_s + 1)) ** 2
    assert abs(abs(rs[0]) ** 2 - bare) < 1e-10


def test_tmm_energy_conservation_oblique():
    lam = np.full(50, 633e-9)
    cos_i = np.cos(np.linspace(0, 1.5, 50))
    n1 = np.ones(50)
    n2 = np.full(50, 1.52 + 0j)
    layers_n = [np.full(50, 1.38 + 0j), np.full(50, 2.35 + 0j)]
    layers_d = [100e-9, 60e-9]
    rs, rp, ts, tp, etas = tf.tmm_coeffs(lam, cos_i, n1, n2,
                                         layers_n, layers_d)
    Rs, Rp, Ts, Tp = tf.tmm_power(rs, rp, ts, tp, etas)
    assert np.max(np.abs(Rs + Ts - 1)) < 1e-10
    assert np.max(np.abs(Rp + Tp - 1)) < 1e-10


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------
def _rays_toward(target, origins):
    d = target - origins
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def test_sphere_intersection_exact():
    s = Sphere([0.0, 0.0, 0.0], 0.015)
    o = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.014, 0.0]])
    d = np.tile([1.0, 0.0, 0.0], (2, 1))
    t, valid = s.intersect(o, d)
    assert valid[0].all()
    assert abs(t[0, 0] - (1 - 0.015)) < 1e-12
    assert abs(t[0, 1] - (1 + 0.015)) < 1e-12
    # chord at y=0.014: x = sqrt(r^2 - y^2)
    x = np.sqrt(0.015 ** 2 - 0.014 ** 2)
    assert abs(t[1, 0] - (1 - x)) < 1e-12
    n = s.normal(np.array([[0.015, 0.0, 0.0]]))
    assert np.allclose(n, [[1.0, 0.0, 0.0]], atol=1e-15)


def test_cylinder_cone_intersection():
    c = Cylinder([0, 0, 0], [0, 0, 1], 0.01)
    o = np.array([[-1.0, 0.0, 0.5]])
    d = np.array([[1.0, 0.0, 0.0]])
    t, valid = c.intersect(o, d)
    assert valid[0].all() and abs(t[0, 0] - 0.99) < 1e-12
    k = Cone([0, 0, 0], [0, 0, 1], np.deg2rad(30))
    o = np.array([[-1.0, 0.0, 0.1]])
    t, valid = k.intersect(o, d)
    # at height z=0.1, cone radius = 0.1*tan(30deg)
    r = 0.1 * np.tan(np.deg2rad(30))
    hit = o[0] + t[0, valid[0]][0] * d[0]
    assert abs(np.hypot(hit[0], hit[1]) - r) < 1e-10


def test_torus_vs_nproots():
    rng = np.random.default_rng(1)
    tor = Torus([0.01, -0.02, 0.005], [0.0, 0.3, 1.0], 0.05, 0.012)
    o = rng.normal(scale=0.2, size=(64, 3))
    d = rng.normal(size=(64, 3))
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    t, valid = tor.intersect(o, d)
    # brute-force reference roots via np.roots on the same quartic built
    # from first principles in the torus frame
    M = np.stack([tor.t1, tor.t2, tor.a], axis=0)
    for i in range(64):
        ol = (o[i] - tor.c) @ M.T
        dl = d[i] @ M.T
        G = ol @ ol + tor.R ** 2 - tor.r ** 2
        a4 = 1.0
        a3 = 4 * (ol @ dl)
        a2 = 4 * (ol @ dl) ** 2 + 2 * G - 4 * tor.R ** 2 * (1 - dl[2] ** 2)
        a1 = 4 * (ol @ dl) * G - 8 * tor.R ** 2 * (
            (ol @ dl) - ol[2] * dl[2])
        a0 = G ** 2 - 4 * tor.R ** 2 * (ol[0] ** 2 + ol[1] ** 2)
        ref = np.roots([a4, a3, a2, a1, a0])
        ref = np.sort(ref[np.abs(ref.imag) < 1e-7].real)
        got = np.sort(t[i][valid[i]])
        assert len(got) == len(ref), "root count mismatch ray %d" % i
        if len(ref):
            assert np.max(np.abs(got - ref)) < 1e-7
    # normal points from tube center to surface point
    p = tor.c + tor.t1 * (tor.R + tor.r)
    n = tor.normal(p[None, :])
    assert np.allclose(n[0], tor.t1, atol=1e-12)


def test_trimmed_spherical_cap():
    # cap of a R=0.1 sphere centered at origin, cut at z = 0.08 (cap around
    # +z pole... but UV poles are on z — use a cap around +x instead to
    # exercise the generic path): trim circle at x = 0.08.
    R, xcut = 0.1, 0.08
    s = Sphere([0, 0, 0], R)
    rho = np.sqrt(R ** 2 - xcut ** 2)
    ang = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    wire = np.stack([np.full_like(ang, xcut), rho * np.cos(ang),
                     rho * np.sin(ang)], axis=-1)
    face = AnalyticFace("B.F.Face1", s, [wire.tolist()], True, 0, 0)
    # ray hitting the cap center: valid; ray hitting the back: rejected
    o = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.09],
                  [-1.0, 0.001, 0.0]])
    d = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    t, hit = face.intersect(o, d)
    assert hit[0] and abs(t[0] - (1 - R)) < 1e-12   # front apex, on cap
    assert not hit[1]                               # misses cap band
    # third ray travels +x from inside-ish; first sphere hit is at
    # x=-sqrt(...) (not on cap) but second hit x=+ is on the cap
    assert hit[2] and abs(t[2] - (1 + np.sqrt(R**2 - 0.001**2))) < 1e-10


def test_full_sphere_untrimmed():
    # A complete sphere (GlassSphere case): FreeCAD's trim wire is a
    # degenerate seam meridian. With face_area = 4 pi R^2 the trim must
    # resolve to 'untrimmed' and accept every surface point.
    R = 0.015
    s = Sphere([0, 0, 0], R)
    # seam meridian through the poles (degenerate wire, doubled path)
    tt = np.linspace(-np.pi / 2, np.pi / 2, 64)
    meridian = np.stack([R * np.cos(tt), np.zeros_like(tt),
                         R * np.sin(tt)], axis=-1)
    wire = np.concatenate([meridian, meridian[::-1]], axis=0)
    tp = TrimPolygon(s, [wire.tolist()], face_area=4 * np.pi * R ** 2)
    assert tp.mode == "untrimmed"
    uv = s.to_uv(np.array([[R, 0, 0], [0, R, 0], [0, 0, R],
                           [-R, 0, 0]]))
    assert tp.contains(uv).all()


def test_sphere_polar_cap_band():
    # cap around the +z UV pole, cut at z = 0.6 R: the wire circles the
    # axis (winding 1); area matching must pick the NORTH cap, not the
    # degenerate zone between the wire's own v extremes.
    R = 0.1
    zcut = 0.06
    s = Sphere([0, 0, 0], R)
    rho = np.sqrt(R ** 2 - zcut ** 2)
    ang = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    wire = np.stack([rho * np.cos(ang), rho * np.sin(ang),
                     np.full_like(ang, zcut)], axis=-1)
    cap_area = 2 * np.pi * R * (R - zcut)      # spherical cap area
    tp = TrimPolygon(s, [wire.tolist()], face_area=cap_area)
    assert tp.mode == "band"
    inside = s.to_uv(np.array([[0, 0, R]]))            # pole point
    rim = s.to_uv(np.array([[rho, 0, zcut]]))
    below = s.to_uv(np.array([[R, 0, 0], [0, 0, -R]]))  # equator, S pole
    assert tp.contains(inside).all()
    assert tp.contains(rim).all()
    assert not tp.contains(below).any()


def test_plane_face_trim():
    p = Plane([0, 0, 0], [0, 0, 1])
    sq = [[-0.01, -0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0],
          [-0.01, 0.01, 0]]
    face = AnalyticFace("B.F.Face2", p, [sq], True, 0, 0)
    o = np.array([[0.0, 0.0, 1.0], [0.02, 0.0, 1.0]])
    d = np.tile([0.0, 0.0, -1.0], (2, 1))
    t, hit = face.intersect(o, d)
    assert hit[0] and abs(t[0] - 1.0) < 1e-15
    assert not hit[1]
    # hole: same square with an inner hole square — center now misses
    hole = [[-0.002, -0.002, 0], [0.002, -0.002, 0], [0.002, 0.002, 0],
            [-0.002, 0.002, 0]]
    face2 = AnalyticFace("B.F.Face3", p, [sq, hole], True, 0, 0)
    t2, hit2 = face2.intersect(o[:1], d[:1])
    assert not hit2[0]


def test_face_exclude_and_eps():
    R = 0.01
    s = Sphere([0, 0, 0], R)
    tt = np.linspace(-np.pi / 2, np.pi / 2, 64)
    meridian = np.stack([R * np.cos(tt), np.zeros_like(tt),
                         R * np.sin(tt)], axis=-1)
    wire = np.concatenate([meridian, meridian[::-1]], axis=0)
    face = AnalyticFace("B.F.Face4", s, [wire.tolist()], True, 0, 0,
                        area_m2=4 * np.pi * R ** 2)
    o = np.array([[-0.01, 0.0, 0.0]])          # ON the surface
    d = np.array([[1.0, 0.0, 0.0]])
    t, hit = face.intersect(o, d)
    # eps guard must skip the t~0 self-hit and find the far side at 0.02
    assert hit[0] and abs(t[0] - 0.02) < 1e-12
    t2, hit2 = face.intersect(o, d, exclude_mask=np.array([True]))
    assert not hit2[0]


def test_cos_theta_t_weakly_absorbing_incident_propagating_branch():
    """Regression (samples-instruments round): a WEAKLY ABSORBING incident
    medium (water, k~1e-8) into an exactly lossless glass used to flip the
    propagating root — Im(1-s2) picks up numerical-dust negativity from
    n1's absorption, the principal sqrt has Im(ct) < 0 by ~1e-13, and the
    unconditional Im(n2*ct) >= 0 decay rule negated ct. The near-cancelling
    denominator then gave |rs| ~ 15 (R ~ 230 instead of 0.004), and a
    nested-cylinder chord loop amplified every bounce — the
    vial_cylindrical 1e16 closure explosion. The branch rule now applies
    the radiation condition Re(n2*ct) >= 0 in the effectively-propagating
    regime and the decay condition only for genuinely evanescent roots."""
    n1 = np.array([1.3321009 + 1.4655e-08j])   # water @633 (tabulated k)
    n2 = np.array([1.52 + 0.0j])               # lossless soda-lime glass
    ci = np.array([0.99998373])
    ct = fr.cos_theta_t(ci, n1, n2)
    assert np.real(ct[0]) > 0.99                     # propagating, forward
    rs, rp, ts, tp, ct2 = fr.fresnel_coeffs(ci, n1, n2)
    assert np.abs(rs[0]) < 1.0 and np.abs(rp[0]) < 1.0
    Rs, Rp, Ts, Tp = fr.power_coeffs(rs, rp, ts, tp, ci, ct2, n1, n2)
    # normal-incidence water->glass: R ~ 0.44%, R+T = 1
    assert Rs[0] == pytest.approx(0.0044, abs=0.001)
    assert Rs[0] + Ts[0] == pytest.approx(1.0, abs=1e-6)
    # genuine TIR branch untouched: glass->air past the critical angle
    n1g = np.array([1.52 + 0.0j])
    n2a = np.array([1.0 + 0.0j])
    ci_tir = np.array([0.5])                    # 60 deg > critical 41.1 deg
    ct_tir = fr.cos_theta_t(ci_tir, n1g, n2a)
    assert abs(np.real(ct_tir[0])) < 1e-12      # evanescent: pure imaginary
    assert np.imag(n2a[0] * ct_tir[0]) > 0      # decays into medium 2
    rs_t, _, _, _, _ = fr.fresnel_coeffs(ci_tir, n1g, n2a)
    assert np.abs(rs_t[0]) == pytest.approx(1.0, abs=1e-12)   # |r|=1 at TIR
