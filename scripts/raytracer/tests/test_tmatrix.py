# =============================================================================
# test_tmatrix.py — TMatrixEvaluator (spheroid T-matrix optics) vs the Mie
# limit, physical sanity (prolate/oblate/sphere), disk-cache behavior, and
# the soft-import error message. Mirrors the structure/conventions of
# test_mie_particles.py.
#
# Run: /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_tmatrix.py -q
#      /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_tmatrix.py -q -m "not slow"
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from raytracer import tmatrix as tm_mod                          # noqa
from raytracer.tmatrix import TMatrixEvaluator, make_evaluator    # noqa
from raytracer.mie import MieEvaluator                            # noqa

pytmatrix = pytest.importorskip(
    "pytmatrix", reason="pytmatrix not installed in this interpreter")


class _ConstMat:
    """Minimal duck-typed material: constant (n, k) + density, matching
    the interface raytracer.materials.Material exposes to mie.py/tmatrix.py
    (n_complex(lam) vectorized, .density)."""

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


def _tm(aspect_ratio, cache_dir, n_p=1.5, k_p=0.0, n_mu=41,
        n_alpha=5, n_beta=10):
    return TMatrixEvaluator(
        _ConstMat(n_p, k_p), _ConstMat(1.0, 0.0, 1.204),
        aspect_ratio=aspect_ratio, cache_dir=str(cache_dir),
        n_mu=n_mu, n_alpha=n_alpha, n_beta=n_beta)


def _mie(n_p=1.5, k_p=0.0):
    return MieEvaluator(_ConstMat(n_p, k_p), _ConstMat(1.0, 0.0, 1.204))


# ---------------------------------------------------------------------------
# FAST smoke test (tiny grid, no @pytest.mark.slow): aspect_ratio==1 falls
# back to pytmatrix's single-orientation solver (no orientation-averaging
# cost), so this is quick despite exercising the full evaluator.
# ---------------------------------------------------------------------------
def test_smoke_sphere_matches_mie_qext(tmp_path):
    x = 5.0
    lam = 633e-9
    r = _r_for_x(x, lam)
    ev_tm = _tm(1.0, tmp_path, n_mu=361)
    ev_mie = _mie(1.5, 0.0)

    qext_tm, qsca_tm, g_tm = ev_tm.efficiencies([r], [lam])
    qext_mie, qsca_mie, g_mie = ev_mie.efficiencies([r], [lam])

    assert qext_tm[0] == pytest.approx(qext_mie[0], rel=1e-3)
    assert qsca_tm[0] == pytest.approx(qsca_mie[0], rel=1e-3)
    assert g_tm[0] == pytest.approx(g_mie[0], abs=2e-3)


def test_smoke_sphere_s1_s2_ratio_matches_mie(tmp_path):
    """S1/S2 Mie-limit agreement (aspect=1): the two libraries carry
    different overall complex normalizations/phase gauges (see the module
    docstring in tmatrix.py), so we compare the physically meaningful
    quantity — the magnitude ratio |S1|/|S2| at each scattering angle,
    which both mie.py and particles.py are the only things ever derived
    from S1/S2 downstream."""
    x = 5.0
    lam = 633e-9
    r = _r_for_x(x, lam)
    ev_tm = _tm(1.0, tmp_path, n_mu=361)
    ev_mie = _mie(1.5, 0.0)

    mus = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    S1_tm, S2_tm = ev_tm.amplitudes(r, lam, mus)
    S1_mie, S2_mie = ev_mie.amplitudes(r, lam, mus)

    ratio_tm = np.abs(S1_tm) / np.abs(S2_tm)
    ratio_mie = np.abs(S1_mie) / np.abs(S2_mie)
    assert ratio_tm == pytest.approx(ratio_mie, rel=2e-2)
    # forward degeneracy S1(0)==S2(0) for a sphere, in EITHER convention
    assert abs(S1_tm[0]) == pytest.approx(abs(S2_tm[0]), rel=1e-6)


# ---------------------------------------------------------------------------
# Fast-integration Qsca/g cross-checked against pytmatrix's own (slow)
# dblquad-based reference functions, at a SINGLE (r, lam) point (aspect
# != 1, so this exercises the real orientation-averaging path we actually
# use in production, unlike the tests above).
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_fast_qsca_g_matches_slow_dblquad_reference(tmp_path):
    from pytmatrix.tmatrix import Scatterer
    from pytmatrix import orientation, scatter

    r = 1.0e-6
    lam = 1.0e-6
    aspect = 1.3
    ev = _tm(aspect, tmp_path, n_p=1.5, k_p=0.02, n_mu=181)
    qext, qsca, g = ev.efficiencies([r], [lam])

    n_h = 1.0
    m = complex(1.5, 0.02) / n_h
    sca = Scatterer(radius=r, wavelength=lam / n_h, m=m, axis_ratio=aspect,
                     suppress_warning=True)
    sca.orient = orientation.orient_averaged_fixed
    sca.or_pdf = orientation.uniform_pdf()
    sca.n_alpha = 5
    sca.n_beta = 10
    sca.set_geometry((90.0, 90.0, 0.0, 0.0, 0.0, 0.0))
    qsca_ref = scatter.sca_xsect(sca) / (np.pi * r ** 2)
    g_ref = scatter.asym(sca)

    assert qsca[0] == pytest.approx(qsca_ref, rel=5e-3)
    assert g[0] == pytest.approx(g_ref, abs=5e-3)


# ---------------------------------------------------------------------------
# Phase function: mirrors test_mie_particles.test_phase_function_normalized
# ---------------------------------------------------------------------------
def test_phase_function_normalized(tmp_path):
    r = _r_for_x(8.0)
    ev = _tm(1.4, tmp_path, n_mu=181)
    mu, p, cdf = ev.phase_function(r, 633e-9)
    integral = 2 * np.pi * np.trapezoid(p, mu)
    assert integral == pytest.approx(1.0, rel=1e-6)
    assert np.all(np.diff(cdf) >= 0) and cdf[0] == 0 and cdf[-1] == 1


# ---------------------------------------------------------------------------
# Physical sanity: prolate vs oblate vs sphere differ measurably at fixed
# (volume-equivalent) radius.
# ---------------------------------------------------------------------------
def test_prolate_oblate_sphere_differ_at_fixed_volume(tmp_path):
    r = _r_for_x(6.0)
    lam = 633e-9
    ev_sphere = _tm(1.0, tmp_path, n_mu=181)
    ev_oblate = _tm(1.8, tmp_path, n_mu=181)
    ev_prolate = _tm(1.0 / 1.8, tmp_path, n_mu=181)

    qe_s, qs_s, g_s = ev_sphere.efficiencies([r], [lam])
    qe_o, qs_o, g_o = ev_oblate.efficiencies([r], [lam])
    qe_p, qs_p, g_p = ev_prolate.efficiencies([r], [lam])

    # measurable difference: > 1% relative in at least one of Qext/g
    assert (abs(qe_o[0] - qe_s[0]) / qe_s[0] > 0.01
            or abs(g_o[0] - g_s[0]) > 0.01)
    assert (abs(qe_p[0] - qe_s[0]) / qe_s[0] > 0.01
            or abs(g_p[0] - g_s[0]) > 0.01)
    assert (abs(qe_o[0] - qe_p[0]) / qe_s[0] > 0.01
            or abs(g_o[0] - g_p[0]) > 0.01)
    # energy sanity: Qabs = Qext - Qsca >= 0 (non-absorbing here -> ~= 0,
    # up to the residual numerical mismatch between the single-point
    # optical-theorem Qext and the trapz-integrated Qsca — a couple % of
    # slack for a lossless material; the tight, absorbing-material check
    # is test_absorbing_spheroid_energy_sanity below)
    for qe, qs in ((qe_s, qs_s), (qe_o, qs_o), (qe_p, qs_p)):
        assert qe[0] - qs[0] > -0.02 * qe[0]


@pytest.mark.slow
def test_absorbing_spheroid_energy_sanity(tmp_path):
    """Qsca <= Qext (Qabs >= 0) for a lossy spheroid, via the FAST
    integration path (not pytmatrix's own dblquad) — sanity-checks that
    the shortcut doesn't silently break energy bookkeeping off-sphere."""
    r = _r_for_x(4.0)
    ev = _tm(1.5, tmp_path, n_p=1.5, k_p=0.1, n_mu=91)
    qext, qsca, g = ev.efficiencies([r], [633e-9])
    assert qsca[0] < qext[0]
    assert qsca[0] / qext[0] > 0.2   # absorbing but not black


# ---------------------------------------------------------------------------
# Broader Mie-limit coverage across x ~ 1-10 and two wavelengths (slow: many
# solves, even though each is individually fast at aspect=1).
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_aspect_one_matches_mie_over_x_range(tmp_path):
    ev_tm = _tm(1.0, tmp_path, n_mu=361)
    ev_mie = _mie(1.5, 0.0)
    for lam in (488e-9, 633e-9):
        for x in (1.0, 2.0, 5.0, 10.0):
            r = _r_for_x(x, lam)
            qext_tm, qsca_tm, _ = ev_tm.efficiencies([r], [lam])
            qext_mie, qsca_mie, _ = ev_mie.efficiencies([r], [lam])
            assert qext_tm[0] == pytest.approx(qext_mie[0], rel=1e-3)
            assert qsca_tm[0] == pytest.approx(qsca_mie[0], rel=1e-3)


# ---------------------------------------------------------------------------
# Cache: a second evaluator instance (same params, same cache_dir) must
# hit the disk cache with ZERO pytmatrix solves.
# ---------------------------------------------------------------------------
def test_disk_cache_hit_avoids_recompute(tmp_path, monkeypatch):
    r = _r_for_x(5.0)
    lam = 633e-9
    ev1 = _tm(1.3, tmp_path, n_mu=41)
    qext1, qsca1, g1 = ev1.efficiencies([r], [lam])
    S1_1, S2_1 = ev1.amplitudes(r, lam, np.array([1.0, 0.0, -1.0]))

    calls = {"n": 0}
    real_solve_uncached = TMatrixEvaluator._solve_uncached

    def counting_solve_uncached(self, r_, lam_):
        calls["n"] += 1
        return real_solve_uncached(self, r_, lam_)

    monkeypatch.setattr(TMatrixEvaluator, "_solve_uncached",
                        counting_solve_uncached)

    ev2 = _tm(1.3, tmp_path, n_mu=41)   # fresh instance, empty memory cache
    qext2, qsca2, g2 = ev2.efficiencies([r], [lam])
    S1_2, S2_2 = ev2.amplitudes(r, lam, np.array([1.0, 0.0, -1.0]))

    assert calls["n"] == 0, "disk cache miss: _solve_uncached was called"
    assert qext2[0] == pytest.approx(qext1[0], rel=1e-12)
    assert qsca2[0] == pytest.approx(qsca1[0], rel=1e-12)
    assert g2[0] == pytest.approx(g1[0], rel=1e-12)
    np.testing.assert_allclose(S1_2, S1_1)
    np.testing.assert_allclose(S2_2, S2_1)


def test_disk_cache_key_distinguishes_aspect_ratio(tmp_path):
    """Different aspect_ratio -> different cache entries (not accidentally
    shared), even at identical (r, lam) and material."""
    r = _r_for_x(5.0)
    lam = 633e-9
    ev_a = _tm(1.3, tmp_path, n_mu=41)
    ev_b = _tm(1.6, tmp_path, n_mu=41)
    qa, _, _ = ev_a.efficiencies([r], [lam])
    qb, _, _ = ev_b.efficiencies([r], [lam])
    assert qa[0] != pytest.approx(qb[0], rel=1e-6)


# ---------------------------------------------------------------------------
# Soft-import error message
# ---------------------------------------------------------------------------
def test_soft_import_error_message(tmp_path, monkeypatch):
    monkeypatch.setattr(tm_mod, "_HAVE_PYTMATRIX", False)
    monkeypatch.setattr(tm_mod, "_PYTMATRIX_IMPORT_ERROR",
                        ImportError("no module named pytmatrix"))
    with pytest.raises(ImportError) as excinfo:
        TMatrixEvaluator(_ConstMat(1.5), _ConstMat(1.0, 0.0, 1.204),
                        aspect_ratio=1.3, cache_dir=str(tmp_path))
    msg = str(excinfo.value)
    assert "pytmatrix" in msg
    assert "pip install" in msg
    assert "INSTALL.md" in msg


def test_make_evaluator_factory(tmp_path):
    mat_p = _ConstMat(1.5)
    mat_h = _ConstMat(1.0, 0.0, 1.204)
    ev_sphere = make_evaluator(mat_p, mat_h, "sphere")
    assert isinstance(ev_sphere, MieEvaluator)
    assert not isinstance(ev_sphere, TMatrixEvaluator)

    ev_spheroid = make_evaluator(mat_p, mat_h, "spheroid", aspect_ratio=1.4,
                                 cache_dir=str(tmp_path))
    assert isinstance(ev_spheroid, TMatrixEvaluator)

    with pytest.raises(ValueError):
        make_evaluator(mat_p, mat_h, "spheroid")   # missing aspect_ratio
    with pytest.raises(ValueError):
        make_evaluator(mat_p, mat_h, "cube")       # unknown shape
