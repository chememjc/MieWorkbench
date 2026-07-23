# =============================================================================
# test_analysis.py — Zernike/Strehl wavefront-analysis oracles:
# Noll bookkeeping, orthonormality, exact coefficient recovery, OPD-from-rays
# geometry, Maréchal Strehl.
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_analysis.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer import analysis as an           # noqa: E402


def _disc_samples(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    rho = np.sqrt(rng.uniform(0.0, 1.0, n))     # uniform over the disc
    theta = rng.uniform(-np.pi, np.pi, n)
    return rho, theta


# ---------------------------------------------------------------------------
# indexing conventions
# ---------------------------------------------------------------------------
def test_noll_to_nm_first_terms():
    want = {1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0), 5: (2, -2),
            6: (2, 2), 7: (3, -1), 8: (3, 1), 9: (3, -3), 10: (3, 3),
            11: (4, 0)}
    for j, nm in want.items():
        assert an.noll_to_nm(j) == nm, j


def test_fringe_indices_of_the_classics():
    # piston 1, tilt 2/3, defocus 4, astig 5/6, coma 7/8, spherical 9
    assert an.fringe_index(0, 0) == 1
    assert an.fringe_index(2, 0) == 4
    assert an.fringe_index(4, 0) == 9


# ---------------------------------------------------------------------------
# orthonormality (Noll normalization: <Zi Zj> = delta_ij over the disc)
# ---------------------------------------------------------------------------
def test_zernike_orthonormality_monte_carlo():
    rho, theta = _disc_samples(n=400000, seed=3)
    B = an.zernike_basis(11, rho, theta)
    G = (B.T @ B) / len(rho)
    assert np.max(np.abs(G - np.eye(11))) < 2e-2   # MC tolerance
    # unit RMS on the diagonal to the same tolerance
    assert np.max(np.abs(np.diag(G) - 1.0)) < 2e-2


# ---------------------------------------------------------------------------
# coefficient recovery
# ---------------------------------------------------------------------------
def test_fit_recovers_injected_defocus_and_coma():
    rho, theta = _disc_samples()
    lam = 633e-9
    truth = np.zeros(15)
    truth[3] = 0.25 * lam     # Z4 defocus (Noll)
    truth[7] = -0.10 * lam    # Z8 x-coma
    truth[1] = 0.05 * lam     # tilt (must not leak into rms_wavefront)
    opd = an.zernike_basis(15, rho, theta) @ truth
    out = an.fit_zernike(rho, theta, opd, jmax=15)
    assert np.max(np.abs(out["coeffs"] - truth)) < 1e-3 * lam
    want_rms = np.sqrt(truth[3] ** 2 + truth[7] ** 2)
    assert abs(out["rms_wavefront"] - want_rms) < 1e-3 * lam
    assert out["residual_rms"] < 1e-9 * lam


def test_fit_with_noise_and_weights_is_stable():
    rho, theta = _disc_samples(n=5000, seed=7)
    rng = np.random.default_rng(11)
    lam = 633e-9
    truth = np.zeros(11)
    truth[3] = 0.2 * lam
    opd = an.zernike_basis(11, rho, theta) @ truth \
        + rng.normal(0.0, 0.01 * lam, len(rho))
    w = rng.uniform(0.5, 1.5, len(rho))
    out = an.fit_zernike(rho, theta, opd, jmax=11, weights=w)
    assert abs(out["coeffs"][3] - truth[3]) < 5e-3 * lam
    assert out["residual_rms"] < 0.02 * lam


# ---------------------------------------------------------------------------
# OPD-from-rays geometry
# ---------------------------------------------------------------------------
def test_opd_from_rays_perfect_focus_is_flat():
    # rays from a fan that all pass through the reference point with equal
    # OPL-to-focus must give zero OPD
    rng = np.random.default_rng(5)
    n = 500
    pupil = rng.uniform(-1, 1, (n, 2))
    pupil = pupil[np.linalg.norm(pupil, axis=1) <= 1.0]
    ref = np.array([0.0, 0.0, 0.1])
    # each ray "lands" somewhere on a sphere of radius R around the focus,
    # having spent opl = L0 - R to get there (perfect spherical wave)
    R = 0.007
    dirs = rng.normal(size=(len(pupil), 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    hits = ref + R * dirs
    L0 = 0.25
    opl = np.full(len(pupil), L0 - R)
    opd, rho, theta = an.opd_from_rays(pupil, hits, opl, ref)
    assert np.max(np.abs(opd)) < 1e-15 + 1e-12 * L0
    assert rho.max() <= 1.0 + 1e-12


def test_opd_from_rays_defocus_shows_quadratic():
    # collimated bundle hitting a flat detector: referencing the wavefront
    # to a point AT the detector plane centre gives W ~ sqrt(x^2+d^2)-d,
    # the classic defocus quadratic in pupil radius
    n = 41
    x = np.linspace(-1.0, 1.0, n)
    pupil = np.stack([x, np.zeros(n)], axis=1)
    a = 0.005                                   # semi-aperture, metres
    hits = np.stack([a * x, np.zeros(n), np.zeros(n)], axis=1)
    opl = np.zeros(n)                           # collimated, equal paths
    d = 0.050
    ref = np.array([0.0, 0.0, d])
    opd, rho, theta = an.opd_from_rays(pupil, hits, opl, ref)
    want = np.sqrt((a * x) ** 2 + d ** 2) - d
    assert np.max(np.abs(opd - want)) < 1e-12


def test_strehl_marechal_values():
    lam = 633e-9
    assert an.strehl_marechal(0.0, lam) == 1.0
    # lambda/14 RMS -> ~0.82 (the classic diffraction-limited criterion)
    s = an.strehl_marechal(lam / 14.0, lam)
    assert 0.80 < s < 0.83
