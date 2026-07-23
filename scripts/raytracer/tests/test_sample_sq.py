# =============================================================================
# test_sample_sq.py — inter-particle structure factor S(q) in the continuum
# ensemble (mie.EnsembleTables) and explicit paracrystal LATTICE realizations
# (particles.ExplicitRealization). Companion to test_mie_particles.py (whose
# sq=None path stays byte-identical).
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_sample_sq.py -q
# =============================================================================
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from raytracer.mie import (MieEvaluator, LogNormalDistribution,       # noqa
                           EnsembleTables)
from raytracer.rays import RayBatch                                   # noqa
from raytracer.audit import PowerLedger                              # noqa
from raytracer.tests.test_mie_particles import (                     # noqa
    _ConstMat, _FakeDB, _FakeSphereScene, _FakeBody, _sample_row)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _FakeBoxScene:
    """Scene whose 'body 0' interior is its AABB — exercises the lattice /
    contains_points path with a box region (for the fcc-density check)."""

    def __init__(self, lo, hi):
        self.matdb = _FakeDB()
        self.matdb.mats["polystyrene"] = _ConstMat(1.59, 0.0, 1050.0)
        self.ambient = _ConstMat(1.0, 0.0, 1.204)
        self.bodies = []
        self._lo = np.asarray(lo, dtype=float)
        self._hi = np.asarray(hi, dtype=float)

    def point_inside_body(self, pts, body_index):
        assert body_index == 0
        pts = np.asarray(pts)
        return np.all((pts >= self._lo) & (pts <= self._hi), axis=-1)


def _run_continuum(row, seed=1, n=8000, seg=4e-3):
    """Fire a batch of rays inside a water-filled sample body and return
    (medium, survivors, child, p0)."""
    from raytracer.particles import BodyParticleMedium

    scene = _FakeSphereScene([0.0, 0.0, 0.0], 5e-3)
    body = _FakeBody(0, "Liquid", "water", [-5e-3] * 3, [5e-3] * 3)
    med = BodyParticleMedium("s", row, body, scene, seed=seed,
                             lam_list=[633e-9])
    assert med.mode == "continuum"

    class _T:
        ledger = PowerLedger(1)
    batch = RayBatch(n)
    batch.pos[:] = [0.0, 0.0, 0.0]
    batch.dir[:] = [1.0, 0.0, 0.0]
    batch.s_hat[:] = [0.0, 0.0, 1.0]
    batch.Es[:] = np.sqrt(0.5 / n)
    batch.Ep[:] = np.sqrt(0.5 / n)
    batch.lam[:] = 633e-9
    batch.birth_power[:] = 1.0 / n
    batch.push_medium(np.ones(n, dtype=bool), np.zeros(n, dtype=np.int64))
    t = np.full(n, seg)
    fid = np.zeros(n, dtype=np.int32)
    p0 = batch.power.sum()
    tr = _T()
    _, _, b2, child = med.intercept(tr, batch, t, fid)
    return med, b2, child, p0, tr


# ---------------------------------------------------------------------------
# S(q) in the continuum ensemble
# ---------------------------------------------------------------------------
def test_sq_py_suppresses_forward_scattering():
    """A Percus-Yevick hard-sphere S(q) (phi_hs=0.3, S<<1 at low q) must:
      * draw FEWER small-angle (forward, mu near 1) directions than sq=None,
      * lower mu_ext (mu_sca scaled by <S>_p < 1),
      * keep albedo <= 1, and
      * still split energy exactly (parent + children + absorbed == input)."""
    row_none = _sample_row(mode="continuum", median_um=1.0, gsd=1.3)
    row_py = _sample_row(mode="continuum", median_um=1.0, gsd=1.3,
                         sq_model="py", sq_params={"phi_hs": 0.3})

    med0, b0, c0, p0_0, _ = _run_continuum(row_none, seed=1)
    med1, b1, c1, p0_1, tr1 = _run_continuum(row_py, seed=1)

    # forward scattering suppressed: fewer children near the +x (mu -> 1)
    fwd_none = np.mean(c0.dir[:, 0] > 0.9)
    fwd_py = np.mean(c1.dir[:, 0] > 0.9)
    assert fwd_py < fwd_none - 0.02
    assert np.mean(c1.dir[:, 0]) < np.mean(c0.dir[:, 0])

    # mu_ext reduced by the structure factor
    assert med1.tables.mu_ext(633e-9) < med0.tables.mu_ext(633e-9)
    assert med1.tables._by_lam[633e-9]["sq_mean_S"] < 1.0

    # albedo stays physical
    assert med1.tables.albedo(633e-9) <= 1.0 + 1e-12

    # energy split exact: parent (Beer-Lambert) + children == input
    # (albedo 1 here since k=0, so the absorbed bucket is 0)
    assert (b1.power.sum() + c1.power.sum()) == pytest.approx(p0_1, rel=1e-9)


def test_sq_constant_table_scales_mu_sca_exactly():
    """<S>_p of a CONSTANT S(q)=0.5 is exactly 0.5 (weighted mean of a
    constant), so mu_sca' == 0.5*mu_sca to machine precision — checked on
    near-isotropic Rayleigh particles so nothing is angular-resolution
    limited."""
    ev = MieEvaluator(_ConstMat(1.59, 0.0, 1050.0),
                      _ConstMat(1.33, 0.0, 998.0))
    dist = LogNormalDistribution(median_r=0.01e-6, gsd=1.0)   # x << 1
    tab_none = EnsembleTables(ev, dist, 1e14, [0.633e-6])
    q_tab = np.linspace(0.0, 60.0, 50)
    s_tab = np.full(50, 0.5)
    tab = EnsembleTables(
        ev, dist, 1e14, [0.633e-6],
        sq=("table", {"table_data": {"q_per_um": q_tab, "s": s_tab}}, {}))
    assert tab._by_lam[0.633e-6]["sq_mean_S"] == pytest.approx(0.5, abs=1e-6)
    ms0 = tab_none._by_lam[0.633e-6]["mu_sca"]
    ms1 = tab._by_lam[0.633e-6]["mu_sca"]
    assert ms1 == pytest.approx(0.5 * ms0, rel=1e-6)
    # mu_abs unchanged (k=0 here -> both 0), mu_ext = mu_abs + mu_sca'
    assert tab._by_lam[0.633e-6]["mu_ext"] == pytest.approx(ms1, rel=1e-9)


def test_sq_q_grid_units_backscatter_max():
    """The internal q grid must top out at q_max = 4*pi*n_host/lambda_um
    (backscatter). Water host n=1.33 at lambda=0.55 um -> ~30.4 1/um. A
    metre/micrometre unit slip would land at 3.04e7 or 3.04e-5 instead."""
    ev = MieEvaluator(_ConstMat(1.59, 0.0, 1050.0),
                      _ConstMat(1.33, 0.0, 998.0))
    dist = LogNormalDistribution(median_r=0.5e-6, gsd=1.0)

    def _S(q):                       # prebuilt callable form
        return np.ones_like(np.asarray(q, dtype=float))

    tab = EnsembleTables(ev, dist, 1e12, [0.55e-6], sq=_S)
    qmax = tab._by_lam[0.55e-6]["sq_qmax"]
    assert qmax == pytest.approx(4.0 * np.pi * 1.33 / 0.55, rel=1e-9)
    assert 25.0 < qmax < 35.0     # sanity band (catches a gross unit slip)


def test_body_medium_sq_row_no_longer_raises_and_diagnostics():
    """A sample row with an S(q) model builds without raising, and its
    diagnostics carry sq_model + the mu_sca scale factor (-> case.json)."""
    from raytracer.particles import BodyParticleMedium

    scene = _FakeSphereScene([0.0, 0.0, 0.0], 5e-3)
    body = _FakeBody(0, "Liquid", "water", [-5e-3] * 3, [5e-3] * 3)
    row = _sample_row(mode="continuum", median_um=1.0, gsd=1.3,
                      sq_model="py", sq_params={"phi_hs": 0.25})
    med = BodyParticleMedium("s", row, body, scene, seed=2, lam_list=[633e-9])
    d = med.diagnostics()
    assert d["sq_model"] == "py"
    assert 0.0 < d["mu_sca_scale"] < 1.0
    assert "633.0" in d["sq_mean_S"]


def test_sq_none_row_leaves_tables_untouched():
    """A sq_model='none' registry row must produce sq=None tables (no S(q)
    machinery) — byte-identical to the CLI path."""
    from raytracer.particles import BodyParticleMedium

    scene = _FakeSphereScene([0.0, 0.0, 0.0], 5e-3)
    body = _FakeBody(0, "Liquid", "water", [-5e-3] * 3, [5e-3] * 3)
    med = BodyParticleMedium("s", _sample_row(mode="continuum"), body,
                             scene, seed=2, lam_list=[633e-9])
    assert med.tables._sq is None
    assert "sq_model" not in med.diagnostics()


# ---------------------------------------------------------------------------
# explicit paracrystal lattice realizations
# ---------------------------------------------------------------------------
def _lattice_row(**over):
    row = _sample_row(mode="explicit", median_um=0.3, gsd=1.0,
                      sq_model="paracrystal",
                      sq_params={"lattice": "fcc", "a_um": 0.45, "g": 0.05})
    row.update(over)
    return row


def test_lattice_fcc_count_and_jitter_stats():
    """fcc sites at a=0.45 um in a 5 um box: the generated site count
    matches the fcc number density 4/a^3 * V within edge effects, and the
    per-axis jitter std ~ g*a."""
    lo = [-2.5e-6] * 3
    hi = [2.5e-6] * 3
    scene = _FakeBoxScene(lo, hi)
    body = _FakeBody(0, "Xtal", "water", lo, hi)

    from raytracer.particles import BodyParticleMedium
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # overlap warning is expected
        med = BodyParticleMedium("s", _lattice_row(), body, scene, seed=0,
                                 lam_list=[633e-9])
    ex = med.explicit
    assert ex.lattice_info is not None
    a_um, V_um3 = 0.45, 5.0 ** 3
    expect = 4.0 / a_um ** 3 * V_um3
    n_gen = ex.lattice_info["n_sites_generated"]
    assert abs(n_gen / expect - 1.0) < 0.15         # edge effects only
    assert ex.lattice_info["n_sites"] == len(ex.centers)
    assert ex.lattice_info["n_sites"] <= n_gen      # clipping only removes

    # jitter statistics: offset of kept sites from their base lattice point
    off = ex.centers - ex._lattice_base
    assert np.std(off) / 1e-6 == pytest.approx(0.05 * a_um, rel=0.1)

    # diagnostics echo the lattice + the overridden count
    d = med.diagnostics()
    assert d["lattice"]["lattice"] == "fcc"
    assert d["lattice"]["n_sites"] == len(ex.centers)
    assert "phi_count" in d["lattice"]


def test_lattice_centers_inside_sphere_body():
    """Body-bound lattice: every kept jittered center lies inside the real
    spherical body interior (contains_points clip), not just its AABB."""
    R = 1.5e-6
    scene = _FakeSphereScene([0.0, 0.0, 0.0], R)
    body = _FakeBody(0, "Bead", "water", [-R] * 3, [R] * 3)

    from raytracer.particles import BodyParticleMedium
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = BodyParticleMedium("s", _lattice_row(), body, scene, seed=3,
                                 lam_list=[633e-9])
    ex = med.explicit
    assert len(ex.centers) > 20
    r = np.linalg.norm(ex.centers, axis=-1)
    assert np.all(r < R)                             # strictly inside sphere


def test_lattice_demo_scale_base_packing_no_overlap():
    """The sane demo-scale case (0.3 um spheres, a=0.45 um fcc): the base
    (unjittered) lattice packing has NO overlapping spheres — fcc nearest-
    neighbour a/sqrt(2)=0.318 um exceeds the 0.30 um sphere diameter. (The
    small g=0.05 jitter can nudge a few pairs into contact at this tight
    margin; that is warned, not rejected — lattice packing is the author's
    choice, verified below.)"""
    from scipy.spatial import cKDTree

    lo = [-2.5e-6] * 3
    hi = [2.5e-6] * 3
    scene = _FakeBoxScene(lo, hi)
    body = _FakeBody(0, "Xtal", "water", lo, hi)

    from raytracer.particles import BodyParticleMedium
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = BodyParticleMedium("s", _lattice_row(), body, scene, seed=0,
                                 lam_list=[633e-9])
    ex = med.explicit
    r_max = float(ex.radii.max())
    base = ex._lattice_base
    tree = cKDTree(base)
    nn, _ = tree.query(base, k=2)
    min_nn = float(nn[:, 1].min())
    assert min_nn >= 2.0 * r_max - 1e-15            # base lattice fits
    # and the overlap check ran and recorded a pair count (advisory)
    assert "n_overlapping_pairs" in ex.lattice_info


def test_lattice_overlap_warns_when_too_dense():
    """A deliberately too-dense lattice (a small enough that the base fcc
    NN < sphere diameter) warns once rather than raising."""
    lo = [-1.5e-6] * 3
    hi = [1.5e-6] * 3
    scene = _FakeBoxScene(lo, hi)
    body = _FakeBody(0, "Xtal", "water", lo, hi)
    # 0.5 um spheres on a 0.4 um fcc cell: NN 0.283 um < 0.5 um diameter
    row = _lattice_row(median_um=0.5,
                       sq_params={"lattice": "fcc", "a_um": 0.4, "g": 0.02})

    from raytracer.particles import BodyParticleMedium
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        med = BodyParticleMedium("s", row, body, scene, seed=0,
                                 lam_list=[633e-9])
    assert any("overlapping" in str(w.message) for w in rec)
    assert med.explicit.lattice_info["n_overlapping_pairs"] > 0


def test_lattice_continuum_paracrystal_also_gets_sq():
    """Both modes of a paracrystal row work: in CONTINUUM mode the same
    row still routes S(q) through the ensemble tables (peaks -> a mu_sca
    scale != 1)."""
    from raytracer.particles import BodyParticleMedium

    scene = _FakeSphereScene([0.0, 0.0, 0.0], 5e-3)
    body = _FakeBody(0, "Liquid", "water", [-5e-3] * 3, [5e-3] * 3)
    row = _sample_row(mode="continuum", median_um=0.3, gsd=1.0,
                      sq_model="paracrystal",
                      sq_params={"lattice": "fcc", "a_um": 0.45, "g": 0.3})
    med = BodyParticleMedium("s", row, body, scene, seed=1, lam_list=[633e-9])
    assert med.mode == "continuum"
    assert med.tables._sq is not None
    assert med.diagnostics()["sq_model"] == "paracrystal"
