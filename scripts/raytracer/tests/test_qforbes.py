# =============================================================================
# test_qforbes.py -- validation of surfaces.QForbes (ISO 10110-12 Forbes
# Q-type asphere: base conic + Qbfs/Qcon orthonormal departure, engine3.md
# Sec 7.6) that does NOT need prysm -- intersect/normal self-consistency vs
# finite differences, round-trip through the model.json surface schema, and
# a reduces-to-Asphere sanity check. The 1e-12 agreement against prysm's own
# Qbfs/Qcon lives in mieworkbench/tests/test_qforbes_prysm_oracle.py (run
# under env/bin/python, which has prysm; "$MIEWB_OPTICS_PYTHON"'s env does not). Run:
#   "$MIEWB_OPTICS_PYTHON" -m pytest raytracer/tests/test_qforbes.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer.surfaces import (Asphere, QForbes, AnalyticFace,   # noqa
                                make_surface)
import common                                                     # noqa


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _qforbes_f(qf, o, d):
    """Independent scalar oracle: f(t) = axial(t) - sag(radial(t))."""
    def f(t):
        p = o + t * d
        rel = p - qf.v
        h = rel @ qf.a
        w = rel - h * qf.a
        r = np.linalg.norm(w)
        z, ok = qf._sag(np.array([r]))
        return h - float(z[0])
    return f


def _nearest(roots, valid, t_eps=1e-7):
    t = np.where(valid & (roots > t_eps), roots, np.inf)
    return t.min(axis=1)


KINDS = ["qbfs", "qcon"]


# ---------------------------------------------------------------------------
# 1. coeffs=[] reduces exactly to the base conic (matches Asphere coeffs=[])
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_qforbes_no_departure_matches_asphere_conic(kind):
    R, k, r_max = 0.05, -0.6, 0.02
    axis = np.array([0.0, 0.0, 1.0])
    vertex = np.zeros(3)
    qf = QForbes(vertex, axis, R, k, [], r_max, kind=kind)
    asp = Asphere(vertex, axis, R, k, [], r_max)     # same bare conic

    rng = np.random.default_rng(1)
    o = np.array([0, 0, -0.1]) + rng.normal(scale=0.01, size=(200, 3))
    tgt = rng.normal(scale=0.015, size=(200, 3))
    tgt[:, 2] = 0.002
    d = tgt - o
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    tq, vq = qf.intersect(o, d)
    ta, va = asp.intersect(o, d)
    for i in range(200):
        rq = np.sort(tq[i][vq[i] & (tq[i] > 1e-7)])
        ra = np.sort(ta[i][va[i] & (ta[i] > 1e-7)])
        assert len(rq) == len(ra), "root-count mismatch ray %d (%s)" % (i, kind)
        if len(rq):
            assert np.max(np.abs(rq - ra)) < 1e-9


# ---------------------------------------------------------------------------
# 2. departure term: hit matches an independent brentq root
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_qforbes_intersect_vs_brentq(kind):
    qf = QForbes([0, 0, 0], [0, 0, 1.0], 0.05, -0.7,
                [1.2e-3, -6e-4, 2.5e-4, -8e-5], 0.015, kind=kind)
    rng = np.random.default_rng(4)
    o = np.array([0.002, -0.001, -0.08]) + rng.normal(scale=0.002, size=3)
    d = np.array([0.05, -0.03, 1.0])
    d = d / np.linalg.norm(d)
    tt, vv = qf.intersect(o[None], d[None])
    got = _nearest(tt, vv)[0]
    assert np.isfinite(got)
    f = _qforbes_f(qf, o, d)
    oracle = brentq(f, got - 1e-4, got + 1e-4, xtol=1e-15, rtol=1e-15)
    assert abs(got - oracle) < 1e-9


# ---------------------------------------------------------------------------
# 3. normal: finite-difference of the sag surface vs analytic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_qforbes_normal_finite_difference(kind):
    qf = QForbes([0.01, -0.02, 0.0], [0.0, 0.0, 1.0], 0.06, -0.4,
                [8e-4, 3e-4, -1e-4], 0.015, kind=kind)
    rng = np.random.default_rng(5)
    r = rng.uniform(0.001, 0.013, 60)
    ph = rng.uniform(0, 2 * np.pi, 60)

    def surf(r, ph):
        w = r[:, None] * (np.cos(ph)[:, None] * qf.t1
                          + np.sin(ph)[:, None] * qf.t2)
        z, _ = qf._sag(r)
        return qf.v[None] + w + z[:, None] * qf.a

    p = surf(r, ph)
    eps = 1e-7
    dr = (surf(r + eps, ph) - surf(r - eps, ph)) / (2 * eps)
    dphi = (surf(r, ph + eps) - surf(r, ph - eps)) / (2 * eps)
    nfd = np.cross(dphi, dr)
    nfd /= np.linalg.norm(nfd, axis=-1, keepdims=True)
    nfd *= np.sign(nfd @ qf.a)[:, None]
    nan = qf.normal(p)
    assert np.max(np.linalg.norm(nan - nfd, axis=-1)) < 1e-6


# ---------------------------------------------------------------------------
# 4. normal_derivative (shape operator): central FD vs analytic
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


@pytest.mark.parametrize("kind", KINDS)
def test_qforbes_normal_derivative_finite_difference(kind):
    qf = QForbes([0, 0, 0], [0, 0, 1.0], 0.05, -0.5,
                [7e-4, -2e-4, 5e-5], 0.015, kind=kind)
    rng = np.random.default_rng(7)
    r = rng.uniform(0.001, 0.014, 40)
    ph = rng.uniform(0, 2 * np.pi, 40)
    w = r[:, None] * np.stack([np.cos(ph), np.sin(ph), np.zeros_like(ph)], -1)
    z, _ = qf._sag(r)
    p = w + z[:, None] * np.array([0, 0, 1.0])
    assert _fd_shape_operator(qf, p) < 1e-5


# ---------------------------------------------------------------------------
# 5. miss cases: outside r_max, tangent, non-convergence -> clean miss
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_qforbes_miss_cases_no_nan(kind):
    qf = QForbes([0, 0, 0], [0, 0, 1.0], 0.05, 0.0, [3e-4], 0.01, kind=kind)
    o = np.array([[0.05, 0.0, -0.1]])
    d = np.array([[0.0, 0.0, 1.0]])
    t, v = qf.intersect(o, d)
    assert not v.any()
    assert not np.isnan(t[v]).any()

    o2 = np.array([[0.0, 0.0, 0.1]])
    d2 = np.array([[0.0, 0.0, 1.0]])
    t2, v2 = qf.intersect(o2, d2)
    assert _nearest(t2, v2)[0] == np.inf

    o3 = np.array([[0.2, 0.0, 0.0]])
    d3 = np.array([[0.0, 1.0, 0.0]])
    t3, v3 = qf.intersect(o3, d3)
    assert not v3.any()
    assert np.all(np.isfinite(t3[v3]))


# ---------------------------------------------------------------------------
# 6. trimmed cap containment (AnalyticFace, generic to any surface exposing
#    to_uv/intersect) -- same pattern as test_asphere_trimmed_cap
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_qforbes_trimmed_cap(kind):
    R, r_trim = 0.05, 0.01
    qf = QForbes([0, 0, 0], [0, 0, 1.0], R, 0.0, [4e-4, -1e-4], 0.02, kind=kind)
    ang = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    z_rim, _ = qf._sag(np.full_like(ang, r_trim))
    wire = np.stack([r_trim * np.cos(ang), r_trim * np.sin(ang), z_rim],
                    axis=-1)
    face = AnalyticFace("B.F.Face1", qf, [wire.tolist()], True, 0, 0)

    o = np.array([[0.0, 0.0, -0.1], [0.008, 0.0, -0.1], [0.015, 0.0, -0.1]])
    d = np.tile([0.0, 0.0, 1.0], (3, 1))
    t, hit = face.intersect(o, d)
    assert hit[0]
    assert hit[1]
    assert not hit[2]


# ---------------------------------------------------------------------------
# 7. round-trip through the model.json surface schema (common.py + make_surface)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_qforbes_model_json_roundtrip(kind):
    spec = {
        "type": "qforbes", "kind": kind,
        "vertex": [0.001, -0.002, 0.0], "axis": [0.0, 0.0, 1.0],
        "R": 0.045, "k": -0.55,
        "coeffs": [6e-4, -2e-4, 9e-5, -3e-5], "r_max": 0.013,
    }
    # schema gate: same function extract_geometry-produced model.json faces
    # go through (common._check_surface_params via validate_model's face
    # walk); must accept this dict without raising.
    common._check_surface_params("qforbes", spec, "test_qforbes_roundtrip")

    qf_a = make_surface(spec)
    assert isinstance(qf_a, QForbes)
    qf_b = QForbes(spec["vertex"], spec["axis"], spec["R"], spec["k"],
                   spec["coeffs"], spec["r_max"], kind=kind)

    r = np.linspace(1e-6, spec["r_max"] * 0.99, 50)
    za, _ = qf_a._sag(r)
    zb, _ = qf_b._sag(r)
    assert np.array_equal(za, zb)

    # a bad kind must be rejected by both the schema check and the class
    bad = dict(spec, kind="not-a-kind")
    with pytest.raises(common.ContractError):
        common._check_surface_params("qforbes", bad, "ctx")
    with pytest.raises(ValueError):
        make_surface(bad)
