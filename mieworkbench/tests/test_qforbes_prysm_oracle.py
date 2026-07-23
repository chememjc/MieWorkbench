# =============================================================================
# test_qforbes_prysm_oracle.py -- QForbes (raytracer.surfaces.QForbes) sag and
# first-radial-derivative vs prysm's own Qbfs/Qcon Forbes-polynomial surfaces,
# to 1e-12. Run under env/bin/python (the project GUI venv, which has
# numpy/scipy AND prysm installed -- see INSTALL.md's "Optional: prysm
# oracle" section); NOT under "$MIEWB_OPTICS_PYTHON"'s env, which is never modified by
# this round, and does not have prysm.
#
# prysm dependency: installed via
#   env/bin/pip install \
#     "git+https://github.com/brandondube/prysm@f8d72fb66f1c1e5858abdd3f4685805ef319d97b"
# PINNED SHA: f8d72fb66f1c1e5858abdd3f4685805ef319d97b ("x/raytracing: code
# cleanup", 2026-06-14). This is NOT the literal tip of prysm's master
# (26a4209a63b7a254c8c1276c4ac0eeff3ef8369a, "x/raytracing: refactor,
# improve code reuse esp. in tests", 2026-07-12) -- that commit (and the one
# before it, eb52449) ship a `prysm.x.raytracing` package whose own
# `__init__.py` imports modules that were never committed
# (`_first_order.py`, `_namespaces.py` respectively), so `import
# prysm.x.raytracing.sags` (needed for the base-conic + Forbes-departure
# composition oracle below) raises ModuleNotFoundError at both of those
# revisions. f8d72fb is the newest commit confirmed (this session,
# 2026-07-16) to import cleanly; `prysm.polynomials.qpoly` (the Forbes
# recurrence itself) is unaffected either way -- it has not changed
# meaningfully since well before this pin.
#
# Reference formulas exercised here (prysm's, verbatim):
#   * prysm.x.raytracing.sags.Q2d_and_der  -- full Qbfs surface (base conic
#     + departure, WITH the 1/sigma "cos factor" composition) for the m=0
#     azimuthal order, which is exactly Qbfs.
#   * prysm.x.raytracing.sags.conic_sag_and_normal + polynomials.qpoly.
#     compute_z_zprime_Qcon -- Qcon has no ready-made "full surface" helper
#     in prysm's raytracing module (Q2d_and_der's m=0 path is hardwired to
#     Qbfs), so the Qcon oracle is assembled from prysm's own base-conic
#     sag/normal (conic_sag_and_normal, giving sigma = n_hat_z exactly) and
#     its own Qcon departure sum (compute_z_zprime_Qcon), combined with the
#     SAME product-rule composition QForbes uses (z = base + (1/sigma) *
#     departure). This also cross-checks QForbes's sigma_inv = sqrt(1 +
#     zc'(r)^2) identity against prysm's independently-computed normal
#     vector, since sigma_inv here comes from prysm's n_hat, not from our
#     own conic derivative.
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("prysm")

from raytracer.surfaces import QForbes                        # noqa: E402
from prysm.x.raytracing.sags import (                          # noqa: E402
    Q2d_and_der, conic_sag_and_normal)
from prysm.polynomials.qpoly import compute_z_zprime_Qcon      # noqa: E402

TOL = 1e-12


def _qbfs_oracle(R, k, coeffs, r_max, r):
    """Full Qbfs surface sag + dz/dr from prysm's Q2d_and_der (m=0 ==
    Qbfs; base conic + 1/sigma-normalized departure, all inside prysm)."""
    c = 1.0 / R
    t = np.zeros_like(r)
    z, zprimer, _ = Q2d_and_der(list(coeffs), [], [], r, t, r_max, c, k)
    return z, zprimer


def _qcon_oracle(R, k, coeffs, r_max, r):
    """Full Qcon surface sag + dz/dr assembled from prysm's OWN base-conic
    sag/normal (conic_sag_and_normal) and Qcon departure sum
    (compute_z_zprime_Qcon), composed with the same product rule QForbes
    uses. prysm has no ready-made Qcon 'full surface' raytracing helper."""
    c = 1.0 / R
    t = np.zeros_like(r)
    base_sag, n_hat = conic_sag_and_normal(c, k, r, t)
    sigma_inv = 1.0 / n_hat[..., 2]
    u = r / r_max
    S, Sprime_du = compute_z_zprime_Qcon(list(coeffs), u, u * u)

    h = 1e-7   # absolute metres; empirically ~1e-15-level FD error here --
               # NOT scaled by r_max (a relative step this small suffers
               # catastrophic cancellation in the eps/h roundoff term)
    _, n_hat_p = conic_sag_and_normal(c, k, r + h, t)
    _, n_hat_m = conic_sag_and_normal(c, k, r - h, t)
    dsigma_inv_dr = (1.0 / n_hat_p[..., 2] - 1.0 / n_hat_m[..., 2]) / (2 * h)

    z = base_sag + sigma_inv * S
    zprimer = (-n_hat[..., 0] / n_hat[..., 2]) \
        + dsigma_inv_dr * S + sigma_inv * (Sprime_du / r_max)
    return z, zprimer


@pytest.mark.parametrize("R,k,coeffs,r_max", [
    (0.05, -0.5, [1e-6, -2e-7, 3e-8], 0.015),
    (0.03, 0.0, [2e-6], 0.01),
    (-0.08, -1.0, [], 0.02),
    (0.04, 1.5, [5e-7, 1e-7, -4e-8, 2e-9], 0.012),
])
def test_qbfs_sag_and_first_derivative_vs_prysm(R, k, coeffs, r_max):
    qf = QForbes([0, 0, 0], [0, 0, 1.0], R, k, coeffs, r_max, kind="qbfs")
    r = np.linspace(1e-6, r_max * 0.999, 400)
    z, valid = qf._sag(r)
    zp = qf._sag_p(r)
    assert np.all(valid)

    z_o, zp_o = _qbfs_oracle(R, k, coeffs, r_max, r)
    assert np.max(np.abs(z - z_o)) < TOL, np.max(np.abs(z - z_o))
    assert np.max(np.abs(zp - zp_o)) < TOL, np.max(np.abs(zp - zp_o))


@pytest.mark.parametrize("R,k,coeffs,r_max", [
    (0.05, -0.5, [1e-6, -2e-7, 3e-8, 5e-9], 0.015),
    (0.03, 0.0, [2e-6], 0.01),
    (-0.08, -2.0, [], 0.02),
    (0.04, 2.5, [4e-7, -1e-7, 3e-8, -6e-9], 0.012),
])
def test_qcon_sag_and_first_derivative_vs_prysm(R, k, coeffs, r_max):
    qf = QForbes([0, 0, 0], [0, 0, 1.0], R, k, coeffs, r_max, kind="qcon")
    r = np.linspace(1e-6, r_max * 0.999, 400)
    z, valid = qf._sag(r)
    zp = qf._sag_p(r)
    assert np.all(valid)

    z_o, zp_o = _qcon_oracle(R, k, coeffs, r_max, r)
    assert np.max(np.abs(z - z_o)) < TOL, np.max(np.abs(z - z_o))
    assert np.max(np.abs(zp - zp_o)) < TOL, np.max(np.abs(zp - zp_o))
