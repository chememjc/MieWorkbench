# =============================================================================
# test_ray_differentials.py — finite-difference validation of the Igehy ray
# differentials in raytracer/differentials.py.
#
# Every interaction formula is checked against a central-difference oracle: a
# central ray plus four offset rays (+/-h in each abstract wavefront parameter)
# are traced through the SAME transfer -> surface.intersect -> reflect/refract
# path the tracer uses, and the numerical dP_hit / dD are compared against the
# analytic module outputs. Focal tests confirm the wavefront patch collapses at
# the mirror focus (R/2) and the single-surface refraction focus (n2 R/(n2-n1)).
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest raytracer/tests/test_ray_differentials.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer import differentials as rd            # noqa: E402
from raytracer import fresnel as fr                  # noqa: E402
from raytracer.surfaces import Sphere, Cylinder, Asphere  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def perp_basis(D):
    """Two orthonormal vectors perpendicular to each unit direction in D."""
    D = _unit(D)
    a = np.zeros_like(D)
    a[np.arange(len(D)), np.argmin(np.abs(D), axis=-1)] = 1.0
    e1 = _unit(np.cross(a, D))
    e2 = np.cross(D, e1)
    return e1, e2


def nearest_hit(surface, o, d, far=False):
    """Smallest (or, far=True, largest) positive intersection parameter."""
    tcand, valid = surface.intersect(o, d)
    tt = np.where(valid & (tcand > 1e-9), tcand, np.nan)
    if far:
        tt = np.where(np.isnan(tt), -np.inf, tt)
        t = np.max(tt, axis=1)
    else:
        tt = np.where(np.isnan(tt), np.inf, tt)
        t = np.min(tt, axis=1)
    return t, o + t[:, None] * d


def sign_and_nhat(surface, p, d):
    """Return (sign_flip, n_hat) with n_hat = sign*canonical opposing the ray.

    This is exactly the sign correction the tracer applies before handing S to
    reflect()/refract(): S_used = sign_flip[:,None,None]*normal_derivative(p).
    """
    n_can = surface.normal(p)
    s = -np.sign(np.sum(d * n_can, axis=-1))
    s = np.where(s == 0.0, 1.0, s)
    return s, s[:, None] * n_can


def trace_interaction(surface, o, d, kind, n1=1.0, n2=1.5, far=False):
    """Trace rays to the surface and return (t, hit, n_hat, D_out)."""
    d = _unit(d)
    t, p = nearest_hit(surface, o, d, far=far)
    s, n_hat = sign_and_nhat(surface, p, d)
    cos_i = -np.sum(d * n_hat, axis=-1)
    if kind == "reflect":
        d_out = fr.reflect_dir(d, n_hat)
    else:
        d_out = fr.refract_dir(d, n_hat, cos_i,
                               np.full(len(d), n1), np.full(len(d), n2))
    return t, p, s, n_hat, d_out


def rel_err(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)


# ---------------------------------------------------------------------------
# free-space laws
# ---------------------------------------------------------------------------
def test_spherical_wave_r_squared():
    # Point source: dP = 0, dD spans the transverse plane. After propagating a
    # distance r the patch area must grow exactly as r^2.
    D = _unit(np.array([[0.3, -0.7, 0.5]]))
    e1, e2 = perp_basis(D)
    dPdx = np.zeros((1, 3))
    dPdy = np.zeros((1, 3))
    dDdx, dDdy = e1, e2                      # already perpendicular to D
    radii = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    areas = []
    for r in radii:
        dPx = rd.transfer(dPdx, dDdx, D, r)
        dPy = rd.transfer(dPdy, dDdy, D, r)
        areas.append(rd.patch_area(dPx, dPy, D)[0])
    areas = np.array(areas)
    ratio = areas / radii ** 2
    assert np.max(np.abs(ratio / ratio[0] - 1.0)) < 1e-10


def test_collimated_area_invariant():
    # Flat source: dD = 0, so the patch area is constant under free transfer.
    D = _unit(np.array([[0.1, 0.2, 1.0]]))
    e1, e2 = perp_basis(D)
    dPdx, dDdx, dPdy, dDdy = rd.init_flat(D, e1, e2, 0.003)
    a0 = rd.patch_area(dPdx, dPdy, D)[0]
    for t in (0.0, 0.05, 0.5, 5.0):
        dPx = rd.transfer(dPdx, dDdx, D, t)
        dPy = rd.transfer(dPdy, dDdy, D, t)
        assert abs(rd.patch_area(dPx, dPy, D)[0] - a0) < 1e-14


def test_init_curved_matches_direction_derivative():
    # On a curved emitter dD = sign * S @ dP; verify against a finite-difference
    # of the emitted direction (= sign * canonical normal) across the face.
    R = 0.02
    s = Sphere([0.0, 0.0, 0.0], R)
    p0 = np.array([[0.0, 0.0, R]])                 # emit point (north pole)
    e1 = np.array([[1.0, 0.0, 0.0]])
    e2 = np.array([[0.0, 1.0, 0.0]])
    sign = np.array([1.0])                         # emit outward (= canonical)
    S = s.normal_derivative(p0)                    # RAW canonical shape operator
    h = 1.0                                         # unit footprint per param
    dPdx, dDdx, dPdy, dDdy = rd.init_curved(
        s.normal(p0), e1, e2, h, S, sign)
    # FD along the position derivative dPdx: emitted dir = sign*canonical normal
    ds = 1e-6
    emit = lambda pt: sign[:, None] * s.normal(pt)   # normal is scale-free
    fd = (emit(p0 + ds * dPdx) - emit(p0 - ds * dPdx)) / (2 * ds)
    assert rel_err(dDdx, fd) < 1e-6


# ---------------------------------------------------------------------------
# finite-difference validation of transfer_to_surface + reflect/refract
# ---------------------------------------------------------------------------
def _fd_case(surface, O0, D0, kind, far=False, n1=1.0, n2=1.5, h=1e-7,
             seed=0):
    rng = np.random.default_rng(seed)
    D0 = _unit(D0)
    # random, well-conditioned input differentials; dD perpendicular to D0.
    dPdx = rng.normal(size=(1, 3))
    dPdy = rng.normal(size=(1, 3))
    dDdx = rng.normal(size=(1, 3))
    dDdy = rng.normal(size=(1, 3))
    dDdx -= np.sum(dDdx * D0, axis=-1, keepdims=True) * D0
    dDdy -= np.sum(dDdy * D0, axis=-1, keepdims=True) * D0

    # analytic path (central ray)
    t0, p0, s0, n_hat0, dout0 = trace_interaction(
        surface, O0, D0, kind, n1, n2, far=far)
    assert np.isfinite(t0[0]), "central ray missed the surface"
    S0 = s0[:, None, None] * surface.normal_derivative(p0)
    dPx = rd.transfer_to_surface(dPdx, dDdx, D0, t0, n_hat0)
    dPy = rd.transfer_to_surface(dPdy, dDdy, D0, t0, n_hat0)
    if kind == "reflect":
        _, dDx = rd.reflect(dPx, dDdx, D0, n_hat0, S0)
        _, dDy = rd.reflect(dPy, dDdy, D0, n_hat0, S0)
    else:
        eta = n1 / n2
        _, dDx = rd.refract(dPx, dDdx, D0, n_hat0, S0, eta, dout0)
        _, dDy = rd.refract(dPy, dDdy, D0, n_hat0, S0, eta, dout0)

    # finite-difference oracle: 4 offset rays
    def offset(dP, dD, step):
        Oo = O0 + step * dP
        Do = _unit(D0 + step * dD)
        _, po, _, _, douto = trace_interaction(
            surface, Oo, Do, kind, n1, n2, far=far)
        return po, douto

    pxp, dxp = offset(dPdx, dDdx, +h)
    pxm, dxm = offset(dPdx, dDdx, -h)
    pyp, dyp = offset(dPdy, dDdy, +h)
    pym, dym = offset(dPdy, dDdy, -h)
    dPx_fd = (pxp - pxm) / (2 * h)
    dPy_fd = (pyp - pym) / (2 * h)
    dDx_fd = (dxp - dxm) / (2 * h)
    dDy_fd = (dyp - dym) / (2 * h)

    assert rel_err(dPx, dPx_fd) < 1e-5, "dP_hit x"
    assert rel_err(dPy, dPy_fd) < 1e-5, "dP_hit y"
    assert rel_err(dDx, dDx_fd) < 1e-5, "dD out x"
    assert rel_err(dDy, dDy_fd) < 1e-5, "dD out y"


@pytest.mark.parametrize("kind", ["reflect", "refract"])
def test_fd_sphere(kind):
    s = Sphere([0.01, -0.02, 0.03], 0.05)
    O0 = np.array([[0.30, 0.12, 0.22]])
    D0 = np.array([[-1.0, -0.45, -0.6]])          # oblique, hits the front cap
    _fd_case(s, O0, D0, kind, seed=1)


@pytest.mark.parametrize("kind", ["reflect", "refract"])
def test_fd_cylinder(kind):
    c = Cylinder([0.0, 0.0, 0.0], [0.15, 0.25, 1.0], 0.04)
    O0 = np.array([[0.25, 0.10, 0.05]])
    D0 = np.array([[-1.0, -0.35, 0.1]])
    _fd_case(c, O0, D0, kind, seed=2)


@pytest.mark.parametrize("kind", ["reflect", "refract"])
def test_fd_asphere(kind):
    a = Asphere([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.03, -0.5,
                [5.0e2], 0.02)
    O0 = np.array([[0.006, -0.004, 0.20]])
    D0 = np.array([[0.02, -0.015, -1.0]])         # comes down onto the cap
    _fd_case(a, O0, D0, kind, seed=3)


# ---------------------------------------------------------------------------
# focal behaviour
# ---------------------------------------------------------------------------
def test_mirror_focus_R_over_2():
    # Concave spherical mirror: parallel rays converge to f = R/2 from vertex.
    # Rays start at the center and strike the inner (concave) wall at the +z
    # vertex, so the nearest hit is the concave surface.
    R = 0.05
    s = Sphere([0.0, 0.0, 0.0], R)
    D0 = np.array([[0.0, 0.0, 1.0]])
    O0 = np.array([[0.0, 0.0, 0.0]])               # center -> hits vertex (0,0,R)
    e1 = np.array([[1.0, 0.0, 0.0]])
    e2 = np.array([[0.0, 1.0, 0.0]])
    h = 1e-4 * R
    dPdx, dDdx, dPdy, dDdy = rd.init_flat(D0, e1, e2, h)

    t0, p0 = nearest_hit(s, O0, D0)
    sflip, n_hat0 = sign_and_nhat(s, p0, D0)
    S0 = sflip[:, None, None] * s.normal_derivative(p0)
    dPx = rd.transfer_to_surface(dPdx, dDdx, D0, t0, n_hat0)
    dPy = rd.transfer_to_surface(dPdy, dDdy, D0, t0, n_hat0)
    d_out = fr.reflect_dir(D0, n_hat0)
    _, dDx = rd.reflect(dPx, dDdx, D0, n_hat0, S0)
    _, dDy = rd.reflect(dPy, dDdy, D0, n_hat0, S0)

    ts = np.linspace(0.2 * R, 0.8 * R, 4001)
    dA = np.array([rd.patch_area(dPx + tt * dDx, dPy + tt * dDy, d_out)[0]
                   for tt in ts])
    t_min = ts[np.argmin(dA)]
    assert abs(t_min - R / 2) < 0.01 * R


def test_single_surface_refraction_focus():
    # Single spherical refracting interface n1=1 -> n2=1.5, R=25 mm: paraxial
    # focus at n2 R/(n2-n1) = 75 mm from the vertex along the refracted ray.
    R = 0.025
    n1, n2 = 1.0, 1.5
    # Convex interface: center of curvature AHEAD of the ray so the surface
    # converges. Vertex at the origin, center at (0,0,R); rays enter from below.
    s = Sphere([0.0, 0.0, R], R)
    D0 = np.array([[0.0, 0.0, 1.0]])
    O0 = np.array([[0.0, 0.0, -0.05]])             # outside, hits vertex (0,0,0)
    e1 = np.array([[1.0, 0.0, 0.0]])
    e2 = np.array([[0.0, 1.0, 0.0]])
    h = 1e-4 * R
    dPdx, dDdx, dPdy, dDdy = rd.init_flat(D0, e1, e2, h)

    t0, p0 = nearest_hit(s, O0, D0)
    sflip, n_hat0 = sign_and_nhat(s, p0, D0)
    cos_i = -np.sum(D0 * n_hat0, axis=-1)
    d_out = fr.refract_dir(D0, n_hat0, cos_i, np.array([n1]), np.array([n2]))
    S0 = sflip[:, None, None] * s.normal_derivative(p0)
    dPx = rd.transfer_to_surface(dPdx, dDdx, D0, t0, n_hat0)
    dPy = rd.transfer_to_surface(dPdy, dDdy, D0, t0, n_hat0)
    eta = n1 / n2
    _, dDx = rd.refract(dPx, dDdx, D0, n_hat0, S0, eta, d_out)
    _, dDy = rd.refract(dPy, dDdy, D0, n_hat0, S0, eta, d_out)

    f = n2 * R / (n2 - n1)                          # 0.075 m from vertex
    ts = np.linspace(0.6 * f, 1.4 * f, 8001)
    dA = np.array([rd.patch_area(dPx + tt * dDx, dPy + tt * dDy, d_out)[0]
                   for tt in ts])
    t_min = ts[np.argmin(dA)]
    assert abs(t_min - f) < 0.01 * f


# ---------------------------------------------------------------------------
# NaN propagation on TIR / grazing
# ---------------------------------------------------------------------------
def test_tir_produces_nan_no_exception():
    from raytracer.surfaces import Plane
    n1, n2 = 1.5, 1.0                              # dense -> rare: TIR possible
    pl = Plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    n_hat = np.array([[0.0, 0.0, 1.0]])            # into incident medium
    theta = np.deg2rad(60.0)                       # past critical (~41.8 deg)
    D = np.array([[np.sin(theta), 0.0, -np.cos(theta)]])
    cos_i = -np.sum(D * n_hat, axis=-1)
    d_out = fr.refract_dir(D, n_hat, cos_i, np.array([n1]), np.array([n2]))
    S = pl.normal_derivative(np.zeros((1, 3)))     # flat -> zero shape operator
    dP_hit = np.array([[0.001, 0.002, 0.0]])
    dD = np.array([[0.0, 0.001, 0.0]])
    _, dD_ref = rd.refract(dP_hit, dD, D, n_hat, S, n1 / n2, d_out)
    assert np.all(np.isnan(dD_ref))
    # patch_area with a NaN differential yields NaN, no exception raised
    dA = rd.patch_area(dP_hit, dD_ref, D)
    assert np.isnan(dA[0])
