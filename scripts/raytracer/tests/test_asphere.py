# =============================================================================
# test_asphere.py — validation of surfaces.Asphere (conic + polynomial optical
# asphere) and the analytic normal_derivative shape operators on every
# primitive. Run:
#   /home3/optics/env/bin/python -m pytest raytracer/tests/test_asphere.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer.surfaces import (Plane, Sphere, Cylinder, Cone, Torus,   # noqa
                                Asphere, AnalyticFace)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _asphere_f(asp, o, d):
    """Independent scalar oracle: f(t) = axial(t) - sag(radial(t))."""
    def f(t):
        p = o + t * d
        rel = p - asp.v
        h = rel @ asp.a
        w = rel - h * asp.a
        r = np.linalg.norm(w)
        z, ok = asp._sag(np.array([r]))
        return h - float(z[0])
    return f


def _nearest(roots, valid, t_eps=1e-7):
    t = np.where(valid & (roots > t_eps), roots, np.inf)
    return t.min(axis=1)


# ---------------------------------------------------------------------------
# 1. coeffs=[], k=0  ==  the near cap of a sphere
# ---------------------------------------------------------------------------
def test_asphere_reduces_to_sphere():
    R, r_max = 0.05, 0.02
    axis = np.array([0.0, 0.0, 1.0])
    vertex = np.zeros(3)
    asp = Asphere(vertex, axis, R, 0.0, [], r_max)
    sph = Sphere(vertex + R * axis, R)          # tangent at the vertex

    rng = np.random.default_rng(0)
    o = np.array([0, 0, -0.1]) + rng.normal(scale=0.01, size=(200, 3))
    tgt = rng.normal(scale=0.015, size=(200, 3))
    tgt[:, 2] = 0.002
    d = tgt - o
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    ta, va = asp.intersect(o, d)
    ts, vs = sph.intersect(o, d)

    for i in range(200):
        aroots = np.sort(ta[i][va[i] & (ta[i] > 1e-7)])
        # sphere hits restricted to the NEAR cap (z < R) within the disc
        cap = []
        for k in range(2):
            if not vs[i, k] or ts[i, k] <= 1e-7:
                continue
            p = o[i] + ts[i, k] * d[i]
            r = np.hypot(p[0], p[1])
            if r <= r_max + 1e-12 and p[2] < R:
                cap.append(ts[i, k])
        cap = np.sort(cap)
        assert len(aroots) == len(cap), "root-count mismatch ray %d" % i
        if len(cap):
            assert np.max(np.abs(aroots - cap)) < 1e-10


# ---------------------------------------------------------------------------
# 2. parabola k=-1: closed-form ray-paraboloid intersection
# ---------------------------------------------------------------------------
def test_parabola_closed_form():
    R = 0.04
    asp = Asphere([0, 0, 0], [0, 0, 1.0], R, -1.0, [], 0.03)
    # sag = r^2 / (2R) exactly for k=-1
    # axial ray
    o = np.array([[0.0, 0.0, -0.1]])
    d = np.array([[0.0, 0.0, 1.0]])
    t, v = asp.intersect(o, d)
    assert abs(_nearest(t, v)[0] - 0.1) < 1e-12

    # slanted ray, analytic quadratic z = (x^2+y^2)/(2R)
    o = np.array([0.005, 0.0, -0.05])
    d = np.array([0.3, 0.1, 1.0])
    d = d / np.linalg.norm(d)
    a = (d[0] ** 2 + d[1] ** 2) / (2 * R)
    b = (o[0] * d[0] + o[1] * d[1]) / R - d[2]
    c = (o[0] ** 2 + o[1] ** 2) / (2 * R) - o[2]
    roots = np.roots([a, b, c])
    roots = np.sort(roots[np.abs(roots.imag) < 1e-12].real)
    roots = roots[roots > 1e-7]
    tt, vv = asp.intersect(o[None], d[None])
    got = _nearest(tt, vv)[0]
    assert abs(got - roots[0]) < 1e-12


# ---------------------------------------------------------------------------
# 3. polynomial term: hit matches an independent brentq root
# ---------------------------------------------------------------------------
def test_polynomial_term_vs_brentq():
    asp = Asphere([0, 0, 0], [0, 0, 1.0], 0.05, -0.7,
                  [1.5e3, -8.0e6], 0.015)
    rng = np.random.default_rng(4)
    o = np.array([0.002, -0.001, -0.08]) + rng.normal(scale=0.002, size=3)
    d = np.array([0.05, -0.03, 1.0])
    d = d / np.linalg.norm(d)
    tt, vv = asp.intersect(o[None], d[None])
    got = _nearest(tt, vv)[0]
    assert np.isfinite(got)
    f = _asphere_f(asp, o, d)
    # bracket around the numeric hit
    oracle = brentq(f, got - 1e-4, got + 1e-4, xtol=1e-15, rtol=1e-15)
    assert abs(got - oracle) < 1e-10


# ---------------------------------------------------------------------------
# 4. normal: finite-difference of the sag surface vs analytic
# ---------------------------------------------------------------------------
def test_asphere_normal_finite_difference():
    asp = Asphere([0.01, -0.02, 0.0], [0.0, 0.0, 1.0], 0.06, -0.4,
                  [2e3, 5e6], 0.015)
    rng = np.random.default_rng(5)
    r = rng.uniform(0.001, 0.013, 60)
    ph = rng.uniform(0, 2 * np.pi, 60)

    def surf(r, ph):
        w = r[:, None] * (np.cos(ph)[:, None] * asp.t1
                          + np.sin(ph)[:, None] * asp.t2)
        z, _ = asp._sag(r)
        return asp.v[None] + w + z[:, None] * asp.a

    p = surf(r, ph)
    eps = 1e-7
    dr = (surf(r + eps, ph) - surf(r - eps, ph)) / (2 * eps)
    dphi = (surf(r, ph + eps) - surf(r, ph - eps)) / (2 * eps)
    nfd = np.cross(dphi, dr)                      # orient toward +axis
    nfd /= np.linalg.norm(nfd, axis=-1, keepdims=True)
    nfd *= np.sign(nfd @ asp.a)[:, None]
    nan = asp.normal(p)
    assert np.max(np.linalg.norm(nan - nfd, axis=-1)) < 1e-7


# ---------------------------------------------------------------------------
# 5. normal_derivative: central FD vs analytic, every primitive
# ---------------------------------------------------------------------------
def _fd_shape_operator(surf, p, eps=1e-7):
    J = surf.normal_derivative(p)
    worst = 0.0
    for k in range(3):
        e = np.zeros((len(p), 3))
        e[:, k] = 1.0
        dn = (surf.normal(p + eps * e) - surf.normal(p - eps * e)) / (2 * eps)
        Jd = np.einsum("nij,nj->ni", J, e)
        num = np.linalg.norm(dn - Jd, axis=-1)
        den = np.linalg.norm(dn, axis=-1) + 1e-9
        worst = max(worst, float((num / den).max()))
    return worst


def test_normal_derivative_all_primitives():
    rng = np.random.default_rng(7)

    # plane
    pl = Plane([0.1, 0.2, -0.1], [0.2, -0.3, 1.0])
    p = rng.normal(scale=0.1, size=(20, 3))
    assert np.all(pl.normal_derivative(p) == 0.0)

    # sphere
    c = np.array([0.1, -0.2, 0.3])
    s = Sphere(c, 0.05)
    th = rng.uniform(0.2, 2.9, 40)
    ph = rng.uniform(0, 2 * np.pi, 40)
    p = c + 0.05 * np.stack([np.sin(th) * np.cos(ph),
                             np.sin(th) * np.sin(ph), np.cos(th)], -1)
    assert _fd_shape_operator(s, p) < 1e-5

    # cylinder
    cy = Cylinder([0.01, 0, 0], [0.2, 0.3, 1.0], 0.03)
    ph = rng.uniform(0, 2 * np.pi, 40)
    zz = rng.uniform(-0.1, 0.1, 40)
    p = (cy.o + cy.r * (np.cos(ph)[:, None] * cy.t1
                        + np.sin(ph)[:, None] * cy.t2) + zz[:, None] * cy.a)
    assert _fd_shape_operator(cy, p) < 1e-5

    # cone
    co = Cone([0, 0, 0], [0.1, 0.2, 1.0], np.deg2rad(25))
    hh = rng.uniform(0.02, 0.2, 40)
    ph = rng.uniform(0, 2 * np.pi, 40)
    rho = hh * np.tan(co.ha)
    p = (co.apex + hh[:, None] * co.a
         + rho[:, None] * (np.cos(ph)[:, None] * co.t1
                           + np.sin(ph)[:, None] * co.t2))
    assert _fd_shape_operator(co, p) < 1e-5

    # torus
    to = Torus([0.01, -0.02, 0.005], [0.0, 0.3, 1.0], 0.05, 0.012)
    au = rng.uniform(0, 2 * np.pi, 40)
    tu = rng.uniform(0, 2 * np.pi, 40)
    shat = np.cos(au)[:, None] * to.t1 + np.sin(au)[:, None] * to.t2
    ring = to.c + to.R * shat
    p = ring + to.r * (np.cos(tu)[:, None] * shat + np.sin(tu)[:, None] * to.a)
    assert _fd_shape_operator(to, p) < 1e-5

    # asphere
    asp = Asphere([0, 0, 0], [0, 0, 1.0], 0.05, -0.5, [1e3, 2e6], 0.015)
    r = rng.uniform(0.001, 0.014, 40)
    ph = rng.uniform(0, 2 * np.pi, 40)
    w = r[:, None] * np.stack([np.cos(ph), np.sin(ph), np.zeros_like(ph)], -1)
    z, _ = asp._sag(r)
    p = w + z[:, None] * np.array([0, 0, 1.0])
    assert _fd_shape_operator(asp, p) < 1e-5


# ---------------------------------------------------------------------------
# 6. miss cases: outside r_max, tangent, non-convergence -> clean miss
# ---------------------------------------------------------------------------
def test_asphere_miss_cases_no_nan():
    asp = Asphere([0, 0, 0], [0, 0, 1.0], 0.05, 0.0, [], 0.01)
    # ray parallel to axis but outside the r_max disc
    o = np.array([[0.05, 0.0, -0.1]])
    d = np.array([[0.0, 0.0, 1.0]])
    t, v = asp.intersect(o, d)
    assert not v.any()
    assert not np.isnan(t[v]).any()

    # ray in the far field pointing away — no forward hit
    o2 = np.array([[0.0, 0.0, 0.1]])
    d2 = np.array([[0.0, 0.0, 1.0]])
    t2, v2 = asp.intersect(o2, d2)
    assert _nearest(t2, v2)[0] == np.inf

    # grazing / tangent-ish ray skimming the rim plane, well outside disc
    o3 = np.array([[0.2, 0.0, 0.0]])
    d3 = np.array([[0.0, 1.0, 0.0]])
    t3, v3 = asp.intersect(o3, d3)
    assert not v3.any()
    assert np.all(np.isfinite(t3[v3]))          # vacuously true, no NaN leaks


# ---------------------------------------------------------------------------
# 7. trimmed asphere cap contains/excludes correctly
# ---------------------------------------------------------------------------
def test_asphere_trimmed_cap():
    R, r_trim = 0.05, 0.01
    asp = Asphere([0, 0, 0], [0, 0, 1.0], R, 0.0, [], 0.02)
    ang = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    z_rim, _ = asp._sag(np.full_like(ang, r_trim))
    wire = np.stack([r_trim * np.cos(ang), r_trim * np.sin(ang), z_rim],
                    axis=-1)
    face = AnalyticFace("B.F.Face1", asp, [wire.tolist()], True, 0, 0)

    # ray down the axis: hits the vertex (inside the trim) -> valid
    o = np.array([[0.0, 0.0, -0.1], [0.008, 0.0, -0.1], [0.015, 0.0, -0.1]])
    d = np.tile([0.0, 0.0, 1.0], (3, 1))
    t, hit = face.intersect(o, d)
    assert hit[0]                                # centre, inside r_trim
    assert hit[1]                                # r=0.008 < r_trim
    assert not hit[2]                            # r=0.015 > r_trim, trimmed
