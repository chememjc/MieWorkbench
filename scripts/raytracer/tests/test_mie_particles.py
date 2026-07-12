# =============================================================================
# test_mie_particles.py — Mie tables vs canonical values (Wiscombe MIEV0
# test cases, Rayleigh limit), phase-function normalization/sampling,
# phi -> number density, and the continuum-medium energy split.
# Run: /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_mie_particles.py -v
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from raytracer.mie import (MieEvaluator, LogNormalDistribution,   # noqa
                           number_density, EnsembleTables)
from raytracer.rays import RayBatch                               # noqa
from raytracer.audit import PowerLedger                           # noqa


class _ConstMat:
    def __init__(self, n, k=0.0, density=1000.0):
        self._n = n
        self._k = k
        self.density = density

    def n_complex(self, lam):
        lam = np.atleast_1d(lam)
        return np.full(lam.shape, self._n + 1j * self._k,
                       dtype=np.complex128)


def _evaluator(n_p=1.5, k_p=0.0):
    return MieEvaluator(_ConstMat(n_p, k_p), _ConstMat(1.0, 0.0, 1.204))


def _r_for_x(x, lam=633e-9, n_host=1.0):
    return x * lam / (2 * np.pi * n_host)


def test_qext_wiscombe_m15():
    """Wiscombe MIEV0 canonical: m=1.5 (non-absorbing), x=10 and x=100."""
    ev = _evaluator(1.5)
    qext, qsca, g = ev.efficiencies([_r_for_x(10.0)], [633e-9])
    assert qext[0] == pytest.approx(2.8820, abs=2e-3)
    assert qsca[0] == pytest.approx(qext[0], rel=1e-12)   # non-absorbing
    qext, qsca, g = ev.efficiencies([_r_for_x(100.0)], [633e-9])
    assert qext[0] == pytest.approx(2.0944, abs=2e-3)


def test_qext_extinction_paradox():
    """x -> large: Qext -> 2."""
    ev = _evaluator(1.33)
    qext, _, _ = ev.efficiencies([_r_for_x(2000.0)], [633e-9])
    assert qext[0] == pytest.approx(2.0, abs=0.05)


def test_rayleigh_limit():
    """x << 1: Qsca = (8/3) x^4 |(m^2-1)/(m^2+2)|^2."""
    m = 1.33
    x = 0.01
    ev = _evaluator(m)
    _, qsca, _ = ev.efficiencies([_r_for_x(x)], [633e-9])
    expect = (8.0 / 3.0) * x ** 4 * abs((m ** 2 - 1) / (m ** 2 + 2)) ** 2
    assert qsca[0] == pytest.approx(expect, rel=0.01)


def test_absorbing_particle_albedo():
    ev = _evaluator(1.5, 0.1)
    qext, qsca, _ = ev.efficiencies([_r_for_x(5.0)], [633e-9])
    assert qsca[0] < qext[0]              # absorption present
    assert qsca[0] / qext[0] > 0.3        # but not black


def test_phase_function_normalized_and_forward_peaked():
    ev = _evaluator(1.33)
    mu, p, cdf = ev.phase_function(_r_for_x(20.0), 633e-9)
    integral = 2 * np.pi * np.trapezoid(p, mu)
    assert integral == pytest.approx(1.0, rel=1e-6)
    # large sphere: strongly forward peaked
    assert p[-1] > 100 * p[len(p) // 2]
    assert np.all(np.diff(cdf) >= 0) and cdf[0] == 0 and cdf[-1] == 1


def test_phase_sampling_matches_pdf():
    ev = _evaluator(1.33)
    r = _r_for_x(3.0)
    rng = np.random.default_rng(0)
    mu_s = ev.sample_scatter_mu(r, 633e-9, rng, 200000)
    mu, p, cdf = ev.phase_function(r, 633e-9)
    hist, edges = np.histogram(mu_s, bins=64, range=(-1, 1), density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    p_ref = np.interp(centers, mu, p) * 2 * np.pi   # density over mu
    ok = p_ref > p_ref.max() * 0.02
    rel = np.abs(hist[ok] - p_ref[ok]) / p_ref[ok]
    assert np.median(rel) < 0.05


def test_number_density_monodisperse():
    """gsd=1: N = f_v / V_p exactly; hand-checked volume fraction."""
    dist = LogNormalDistribution(median_r=5e-6, gsd=1.0)
    phi = 1e-4
    rho_p, rho_h = 1000.0, 1.204
    N, f_v = number_density(phi, rho_p, rho_h, dist)
    f_v_hand = (phi / rho_p) / (phi / rho_p + (1 - phi) / rho_h)
    assert f_v == pytest.approx(f_v_hand, rel=1e-12)
    Vp = 4 / 3 * np.pi * (5e-6) ** 3
    assert N == pytest.approx(f_v / Vp, rel=1e-12)


def test_lognormal_mean_volume_vs_mc():
    dist = LogNormalDistribution(median_r=2e-6, gsd=1.6,
                                 r_min=1e-12, r_max=1.0)
    rng = np.random.default_rng(1)
    r = np.exp(rng.normal(dist.mu, dist.sigma, 400000))
    mc = np.mean(4 / 3 * np.pi * r ** 3)
    assert dist.mean_volume() == pytest.approx(mc, rel=0.02)


def test_ensemble_mu_ext_monodisperse():
    """Monodisperse ensemble: mu_ext = N pi r^2 Qext exactly."""
    ev = _evaluator(1.33)
    r = 5e-6
    dist = LogNormalDistribution(median_r=r, gsd=1.0)
    N = 1e10
    tab = EnsembleTables(ev, dist, N, [633e-9])
    qext, _, _ = ev.efficiencies([r], [633e-9])
    assert tab.mu_ext(633e-9) == pytest.approx(
        N * np.pi * r ** 2 * qext[0], rel=1e-9)
    assert tab.albedo(633e-9) == pytest.approx(1.0, rel=1e-9)


def test_continuum_energy_split():
    """Parent Beer-Lambert decay + child power = collided fraction."""
    from raytracer.particles import ParticleCloud

    class _FakeScene:
        pass

    class _FakeDB:
        def __init__(self):
            self.mats = {"water": _ConstMat(1.33, 0.0, 998.0)}

        def get(self, name):
            return self.mats[name]

    scene = _FakeScene()
    scene.matdb = _FakeDB()
    scene.ambient = _ConstMat(1.0, 0.0, 1.204)
    scene.bodies = []

    # NOTE phi is a MASS fraction vs the ambient AIR: phi=0.02 is only
    # f_v ~ 2.4e-5 by volume -> tau ~ 0.07 over the 10 mm box
    spec = {"box_corner_m": [0.0, -5e-3, -5e-3],
            "box_size_m": [10e-3, 10e-3, 10e-3],
            "material": "water", "phi": 2e-2,
            "median_um": 10.0, "gsd": 1.0}
    cloud = ParticleCloud(spec, scene, threshold=1.0,   # force continuum
                          seed=3, lam_list=[633e-9])
    assert cloud.mode == "continuum"

    class _FakeTracer:
        ledger = PowerLedger(1)
    tr = _FakeTracer()

    n = 20000
    batch = RayBatch(n)
    batch.pos[:] = [-5e-3, 0.0, 0.0]
    batch.dir[:] = [1.0, 0.0, 0.0]
    batch.s_hat[:] = [0.0, 0.0, 1.0]
    batch.Es[:] = np.sqrt(0.5 / n)
    batch.Ep[:] = np.sqrt(0.5 / n)
    batch.lam[:] = 633e-9
    batch.coherent[:] = True
    batch.birth_power[:] = 1.0 / n

    t = np.full(n, 0.1)          # pretend surface 100 mm downstream
    fid = np.zeros(n, dtype=np.int32)
    p0 = batch.power.sum()
    t2, fid2, batch2, child = cloud.intercept(tr, batch, t, fid)
    tau = cloud.tables.mu_ext(633e-9) * 10e-3        # box depth 10 mm
    # ballistic transmission
    p_after = batch2.power.sum()
    assert p_after == pytest.approx(p0 * np.exp(-tau), rel=1e-9)
    # scattered child carries the collided power (albedo = 1 here)
    assert child is not None
    assert child.power.sum() == pytest.approx(p0 * (1 - np.exp(-tau)),
                                              rel=1e-9)
    assert not child.coherent.any()
    # scatter points inside the box
    assert np.all(child.pos[:, 0] >= 0.0 - 1e-12)
    assert np.all(child.pos[:, 0] <= 10e-3 + 1e-12)
    # tau should be meaningful for the test to matter
    assert 0.02 < tau < 5.0, tau


def test_explicit_realization():
    """Explicit mode: placement is non-overlapping and inside the box;
    collisions scatter with energy conserved (albedo=1, lossless)."""
    from raytracer.particles import ParticleCloud

    class _FakeScene:
        pass

    class _FakeDB:
        def __init__(self):
            self.mats = {"water": _ConstMat(1.33, 0.0, 998.0)}

        def get(self, name):
            return self.mats[name]

    scene = _FakeScene()
    scene.matdb = _FakeDB()
    scene.ambient = _ConstMat(1.0, 0.0, 1.204)
    scene.bodies = []

    # 50-um monodisperse droplets; note phi is MASS fraction vs air, so
    # even a modest droplet count needs phi ~ 0.45 (water is ~800x denser
    # than air) — this yields ~1000 particles / f_v ~ 1e-3 here
    spec = {"box_corner_m": [0.0, -2e-3, -2e-3],
            "box_size_m": [4e-3, 4e-3, 4e-3],
            "material": "water", "phi": 0.45,
            "median_um": 50.0, "gsd": 1.0}
    cloud = ParticleCloud(spec, scene, threshold=1e6, seed=7,
                          lam_list=[633e-9])
    assert cloud.mode == "explicit"
    ex = cloud.explicit
    n = len(ex.radii)
    assert 10 < n < 20000
    # inside box
    assert np.all(ex.centers >= np.array(spec["box_corner_m"]) - 1e-12)
    assert np.all(ex.centers <= np.array(spec["box_corner_m"])
                  + np.array(spec["box_size_m"]) + 1e-12)
    # non-overlapping
    from scipy.spatial import cKDTree
    tree = cKDTree(ex.centers)
    pairs = tree.query_pairs(2 * ex.radii.max())
    for i, j in pairs:
        d = np.linalg.norm(ex.centers[i] - ex.centers[j])
        assert d >= ex.radii[i] + ex.radii[j] - 1e-12

    class _FakeTracer:
        ledger = PowerLedger(1)
    tr = _FakeTracer()

    m = 5000
    batch = RayBatch(m)
    rng = np.random.default_rng(0)
    batch.pos[:] = np.stack([np.full(m, -1e-3),
                             rng.uniform(-1.5e-3, 1.5e-3, m),
                             rng.uniform(-1.5e-3, 1.5e-3, m)], axis=-1)
    batch.dir[:] = [1.0, 0.0, 0.0]
    batch.s_hat[:] = [0.0, 0.0, 1.0]
    batch.Es[:] = np.sqrt(0.5 / m)
    batch.Ep[:] = np.sqrt(0.5 / m)
    batch.lam[:] = 633e-9
    batch.coherent[:] = True
    batch.birth_power[:] = 1.0 / m

    t = np.full(m, 0.1)
    fid = np.zeros(m, dtype=np.int32)
    p0 = batch.power.sum()
    t2, fid2, batch2, child = cloud.intercept(tr, batch, t, fid)
    assert child is not None and len(child) > 20   # collisions happened
    # scattered rays marked + coherent preserved (frozen realization)
    assert child.scattered.all()
    assert child.coherent.all()
    # energy: survivors + scattered == input (albedo 1, k=0)
    p_after = batch2.power.sum() + child.power.sum()
    assert p_after == pytest.approx(p0, rel=1e-9)
    # scattered off forward direction on average
    assert np.mean(child.dir[:, 0]) < 0.9999


# ---------------------------------------------------------------------------
# tau= (target optical depth) knob — P10 item 2
# ---------------------------------------------------------------------------
class _FakeScene:
    pass


class _FakeDB:
    def __init__(self):
        self.mats = {"water": _ConstMat(1.33, 0.0, 998.0)}

    def get(self, name):
        return self.mats[name]


def _fake_scene():
    scene = _FakeScene()
    scene.matdb = _FakeDB()
    scene.ambient = _ConstMat(1.0, 0.0, 1.204)
    scene.bodies = []
    return scene


def test_resolve_tau_phi_reproduces_target_tau():
    """A synthetic tau= spec must resolve to a phi whose ensemble mu_ext
    reproduces the requested tau (within 1%), along the box's first
    (along-beam) dimension."""
    from raytracer.particles import ParticleCloud

    scene = _fake_scene()
    box_len_m = 40e-3
    spec = {"box_corner_m": [0.0, -20e-3, -20e-3],
            "box_size_m": [box_len_m, 40e-3, 40e-3],
            "material": "water", "tau": 1.14,
            "median_um": 2.0, "gsd": 1.6}
    cloud = ParticleCloud(spec, scene, threshold=1.0,   # force continuum
                          seed=1, lam_list=[532e-9])
    assert cloud.tau_resolved is not None
    assert cloud.tau_resolved["tau_target"] == pytest.approx(1.14)
    assert cloud.spec["phi"] is not None and 0 < cloud.spec["phi"] < 1
    tau_actual = cloud.tables.mu_ext(532e-9) * box_len_m
    assert tau_actual == pytest.approx(1.14, rel=1e-2)
    # echoed into diagnostics() (-> case.json)
    diag = cloud.diagnostics()
    assert diag["tau_resolved"]["resolved_phi"] == pytest.approx(
        cloud.spec["phi"])
    assert diag["phi"] == pytest.approx(cloud.spec["phi"])


def test_particles_phi_tau_mutually_exclusive():
    """common.parse_particles_spec rejects phi+tau together, and requires
    at least one of them."""
    import common  # scripts/ root, already on sys.path via SCRIPTS insert

    with pytest.raises(ValueError, match="mutually exclusive"):
        common.parse_particles_spec(
            "box=0,0,0:1,1,1;material=water;phi=1e-4;tau=1.0;median_um=5")
    with pytest.raises(ValueError):
        common.parse_particles_spec(
            "box=0,0,0:1,1,1;material=water;median_um=5")
    spec = common.parse_particles_spec(
        "box=0,0,0:1,1,1;material=water;tau=0.5;median_um=5")
    assert spec["tau"] == pytest.approx(0.5)
    assert spec["phi"] is None
