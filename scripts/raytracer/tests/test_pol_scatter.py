# =============================================================================
# test_pol_scatter.py — polarized explicit-Mie azimuth sampling (WS-F3a).
#
# Pins the physics of the phi (azimuth) law introduced in particles.py:
#   dI(phi) ∝ |S1|^2 |E_perp(phi)|^2 + |S2|^2 |E_par(phi)|^2
#           = a + b cos(2 phi) + c sin(2 phi)
# with, for the incident field projected onto the (t1v, t2v) frame,
#   a = (s1+s2)/2 (|A|^2+|B|^2), b = (s2-s1)/2 (|A|^2-|B|^2),
#   c = (s2-s1) Re(A conj B),   s1=|S1|^2, s2=|S2|^2.
#
# Run: /home3/optics/env/bin/python -m pytest \
#      scripts/raytracer/tests/test_pol_scatter.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from raytracer.mie import MieEvaluator, LogNormalDistribution        # noqa
from raytracer.particles import ParticleCloud, _sample_phi_polarized  # noqa
from raytracer.rays import RayBatch                                  # noqa
from raytracer.audit import PowerLedger                              # noqa


class _ConstMat:
    def __init__(self, n, k=0.0, density=1000.0):
        self._n = n
        self._k = k
        self.density = density

    def n_complex(self, lam):
        lam = np.atleast_1d(lam)
        return np.full(lam.shape, self._n + 1j * self._k,
                       dtype=np.complex128)


def _r_for_x(x, lam=633e-9, n_host=1.0):
    return x * lam / (2 * np.pi * n_host)


# ---------------------------------------------------------------------------
# analytic helpers mirroring the particles.py derivation
# ---------------------------------------------------------------------------
def _abc(S1, S2, A, B):
    s1 = np.abs(S1) ** 2
    s2 = np.abs(S2) ** 2
    A2 = np.abs(A) ** 2
    B2 = np.abs(B) ** 2
    a = 0.5 * (s1 + s2) * (A2 + B2)
    b = 0.5 * (s2 - s1) * (A2 - B2)
    c = (s2 - s1) * np.real(A * np.conj(B))
    return a, b, c


# ===========================================================================
# 1. theta-marginal invariance: integrating dI(phi) over phi is proportional
#    to (|S1|^2 + |S2|^2), i.e. the unpolarized phase function — for ANY
#    incident Jones state. Both the exact analytic identity (1e-12) and a
#    numeric quadrature (1e-6) are checked.
# ===========================================================================
def test_theta_marginal_invariance_analytic():
    rng = np.random.default_rng(0)
    # random complex S1, S2 and random incident field (A, B) in the frame
    for _ in range(200):
        S1 = rng.normal() + 1j * rng.normal()
        S2 = rng.normal() + 1j * rng.normal()
        A = rng.normal() + 1j * rng.normal()
        B = rng.normal() + 1j * rng.normal()
        a, b, c = _abc(S1, S2, A, B)
        # ∫_0^2pi (a + b cos2phi + c sin2phi) dphi = 2 pi a
        integral = 2 * np.pi * a
        s1s2 = np.abs(S1) ** 2 + np.abs(S2) ** 2
        Pin = np.abs(A) ** 2 + np.abs(B) ** 2
        assert integral == pytest.approx(np.pi * s1s2 * Pin, rel=1e-12)


def test_theta_marginal_invariance_numeric():
    rng = np.random.default_rng(1)
    phi = np.linspace(0.0, 2 * np.pi, 20001)
    S1 = rng.normal() + 1j * rng.normal()
    S2 = rng.normal() + 1j * rng.normal()
    # two very different polarization states -> same phi-integral (up to the
    # shared (s1+s2)*Pin factor). Compare integrals with matched Pin.
    for (A, B) in [(1 + 0j, 0 + 0j),          # linear along t1
                   (0 + 0j, 1 + 0j),          # linear along t2
                   (1 + 0j, 1j)]:             # circular
        a, b, c = _abc(S1, S2, A, B)
        dI = a + b * np.cos(2 * phi) + c * np.sin(2 * phi)
        integral = np.trapezoid(dI, phi)
        Pin = np.abs(A) ** 2 + np.abs(B) ** 2
        s1s2 = np.abs(S1) ** 2 + np.abs(S2) ** 2
        assert integral == pytest.approx(np.pi * s1s2 * Pin, rel=1e-6)
        assert np.all(dI >= -1e-12)   # pdf is non-negative


# ===========================================================================
# 2. chi-square: linear-polarized input, x=2 sphere, small enough to be
#    dipole-like — scattering is suppressed at phi where the field lies in
#    the scattering plane. Histogram sampled phi at a fixed theta bin and
#    test against the analytic a + b cos2phi + c sin2phi law.
# ===========================================================================
def _chi2_pvalue(counts, expected):
    from scipy.stats import chi2 as _chi2dist
    stat = np.sum((counts - expected) ** 2 / expected)
    dof = len(counts) - 1
    return 1.0 - _chi2dist.cdf(stat, dof)


def test_chi2_linear_polarized():
    ev = MieEvaluator(_ConstMat(1.5), _ConstMat(1.0, 0.0, 1.204))
    x = 2.0
    lam = 633e-9
    r = _r_for_x(x, lam)
    # fixed polar angle: theta = 90 deg (mu = 0). At small x (dipole),
    # |S2(90)| << |S1(90)| so the parallel channel is suppressed.
    mu0 = 0.0
    S1, S2 = ev.amplitudes(r, lam, np.array([mu0]))
    S1, S2 = complex(S1[0]), complex(S2[0])

    # linear input purely along t1 in-frame: A = 1, B = 0 (Ep = 0 case with
    # s_hat aligned to t1). This is the "Ep=0" linear-polarized state.
    A, B = 1.0 + 0j, 0.0 + 0j
    a, b, c = _abc(S1, S2, A, B)
    assert abs(c) < 1e-12                      # no sin2phi for B=0
    # dipole suppression: with |S2|<|S1|, b<0 -> minima near phi=0, pi
    assert np.abs(S2) < np.abs(S1)
    assert b < 0

    rng = np.random.default_rng(42)
    N = 200_000
    phi = _sample_phi_polarized(np.full(N, a), np.full(N, b),
                                np.full(N, c), rng)
    nb = 32
    counts, edges = np.histogram(phi, bins=nb, range=(0, 2 * np.pi))
    cen = 0.5 * (edges[1:] + edges[:-1])
    w = a + b * np.cos(2 * cen) + c * np.sin(2 * cen)
    expected = w / w.sum() * N
    p = _chi2_pvalue(counts, expected)
    assert p > 1e-3, p

    # qualitative minimum location: the pdf minimum is where b cos2phi is
    # most negative -> cos2phi = +1 -> phi = 0 or pi (since b < 0).
    kmin = np.argmin(counts)
    ang = cen[kmin] % np.pi
    assert min(ang, np.pi - ang) < np.pi / 8, cen[kmin]


# ===========================================================================
# 3. circular input: b = c = 0 -> phi uniform. chi-square vs uniform.
# ===========================================================================
def test_chi2_circular_uniform():
    ev = MieEvaluator(_ConstMat(1.5), _ConstMat(1.0, 0.0, 1.204))
    r = _r_for_x(2.0)
    S1, S2 = ev.amplitudes(r, 633e-9, np.array([0.3]))
    S1, S2 = complex(S1[0]), complex(S2[0])
    # circular: A = 1, B = i  => |A|=|B|, Re(A conj B) = Re(1 * (-i)) = 0
    A, B = 1.0 + 0j, 1j
    a, b, c = _abc(S1, S2, A, B)
    assert abs(b) < 1e-12 and abs(c) < 1e-12

    rng = np.random.default_rng(7)
    N = 200_000
    phi = _sample_phi_polarized(np.full(N, a), np.full(N, b),
                                np.full(N, c), rng)
    counts, _ = np.histogram(phi, bins=32, range=(0, 2 * np.pi))
    expected = np.full(32, N / 32.0)
    assert _chi2_pvalue(counts, expected) > 1e-3


# ===========================================================================
# 4. energy: per-event child power == albedo * P_in EXACTLY (1e-12) for
#    random incident Jones states, with pol_scatter on AND off.
# ===========================================================================
def _make_cloud(pol_scatter):
    class _S:
        pass

    class _DB:
        def __init__(self):
            self.m = {"water": _ConstMat(1.33, 0.0, 998.0)}

        def get(self, n):
            return self.m[n]

    scene = _S()
    scene.matdb = _DB()
    scene.ambient = _ConstMat(1.0, 0.0, 1.204)
    scene.bodies = []
    spec = {"box_corner_m": [0.0, -2e-3, -2e-3],
            "box_size_m": [4e-3, 4e-3, 4e-3],
            "material": "water", "phi": 0.45,
            "median_um": 50.0, "gsd": 1.0}
    return ParticleCloud(spec, scene, threshold=1e6, seed=7,
                         lam_list=[633e-9], pol_scatter=pol_scatter)


def _random_input_batch(m, rng):
    batch = RayBatch(m)
    batch.pos[:] = np.stack([np.full(m, -1e-3),
                             rng.uniform(-1.5e-3, 1.5e-3, m),
                             rng.uniform(-1.5e-3, 1.5e-3, m)], axis=-1)
    batch.dir[:] = [1.0, 0.0, 0.0]
    batch.s_hat[:] = [0.0, 0.0, 1.0]
    # random elliptical Jones state per ray, normalized so total power = 1
    Es = rng.normal(size=m) + 1j * rng.normal(size=m)
    Ep = rng.normal(size=m) + 1j * rng.normal(size=m)
    norm = np.sqrt((np.abs(Es) ** 2 + np.abs(Ep) ** 2).sum())
    batch.Es[:] = Es / norm
    batch.Ep[:] = Ep / norm
    batch.lam[:] = 633e-9
    batch.coherent[:] = True
    batch.birth_power[:] = np.abs(batch.Es) ** 2 + np.abs(batch.Ep) ** 2
    return batch


@pytest.mark.parametrize("pol", [True, False])
def test_per_event_energy_exact(pol):
    cloud = _make_cloud(pol)
    ex = cloud.explicit

    class _T:
        ledger = PowerLedger(1)
    tr = _T()

    rng = np.random.default_rng(3)
    batch = _random_input_batch(5000, rng)
    # birth_power was set = |Es|^2+|Ep|^2 (the incident power) per ray and is
    # copied unchanged by RayBatch.select, so it records each collided ray's
    # incident power. albedo = 1 here (k=0), so per-event child power must
    # equal that incident power exactly.
    _, _, _, child = cloud.intercept(tr, batch, np.full(len(batch), 0.1),
                                     np.zeros(len(batch), dtype=np.int32))
    assert child is not None and len(child) > 20
    alb = cloud.explicit.albedo_p
    assert np.allclose(alb, 1.0, atol=1e-12)     # lossless -> albedo == 1
    assert np.allclose(child.power, child.birth_power, rtol=0, atol=1e-12)


# ===========================================================================
# 5. legacy flag: pol_scatter=False reproduces uniform azimuth for ANY input
#    state (chi-square vs uniform even for linear-polarized input).
# ===========================================================================
def test_legacy_uniform_azimuth():
    # sampling helper is only invoked when pol_scatter=True; the legacy path
    # calls rng.uniform directly. Drive a full collision batch with a linear-
    # polarized input under pol_scatter=False and confirm the resulting
    # azimuths (recovered from d_out) are uniform.
    cloud = _make_cloud(pol_scatter=False)

    class _T:
        ledger = PowerLedger(1)
    tr = _T()

    m = 8000
    batch = RayBatch(m)
    rng = np.random.default_rng(11)
    batch.pos[:] = np.stack([np.full(m, -1e-3),
                             rng.uniform(-1.5e-3, 1.5e-3, m),
                             rng.uniform(-1.5e-3, 1.5e-3, m)], axis=-1)
    batch.dir[:] = [1.0, 0.0, 0.0]
    batch.s_hat[:] = [0.0, 0.0, 1.0]
    batch.Es[:] = np.sqrt(1.0 / m)       # fully s-polarized (linear)
    batch.Ep[:] = 0.0
    batch.lam[:] = 633e-9
    batch.coherent[:] = True
    batch.birth_power[:] = 1.0 / m
    _, _, _, child = cloud.intercept(tr, batch, np.full(m, 0.1),
                                     np.zeros(m, dtype=np.int32))
    assert child is not None and len(child) > 50
    # recover azimuth of d_out about the incident +x axis: phi = atan2(z, y)
    d = child.dir
    phi = np.arctan2(d[:, 2], d[:, 1]) % (2 * np.pi)
    counts, _ = np.histogram(phi, bins=24, range=(0, 2 * np.pi))
    expected = np.full(24, len(phi) / 24.0)
    assert _chi2_pvalue(counts, expected) > 1e-3


# ===========================================================================
# 6. cross-check the sampler against a direct pdf histogram for an arbitrary
#    (a, b, c) with |b|, |c| < a (guaranteeing non-negativity).
# ===========================================================================
def test_sampler_matches_pdf_general():
    rng = np.random.default_rng(99)
    a, b, c = 1.0, 0.4, -0.3
    N = 300_000
    phi = _sample_phi_polarized(np.full(N, a), np.full(N, b),
                                np.full(N, c), rng)
    counts, edges = np.histogram(phi, bins=48, range=(0, 2 * np.pi))
    cen = 0.5 * (edges[1:] + edges[:-1])
    w = a + b * np.cos(2 * cen) + c * np.sin(2 * cen)
    expected = w / w.sum() * N
    assert _chi2_pvalue(counts, expected) > 1e-3
