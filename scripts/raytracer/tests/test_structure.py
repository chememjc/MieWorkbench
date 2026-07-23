# =============================================================================
# test_structure.py — validation of structure.py: S(q) models (PY hard
# sphere, Baxter sticky sphere, Teixeira fractal, powder paracrystal,
# tabulated) + the sq_evaluate registry dispatcher.
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_structure.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gamma as G

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer import structure as st          # noqa: E402


# ---------------------------------------------------------------------------
# Percus-Yevick hard sphere
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phi", [0.05, 0.2, 0.4])
def test_py_s0_compressibility_limit(phi):
    r = 0.1
    s0 = st.sq_percus_yevick(np.array([0.0]), r, phi)[0]
    target = (1 - phi) ** 4 / (1 + 2 * phi) ** 2
    assert abs(s0 - target) < 1e-10


def test_py_large_q_to_one():
    # spec: S(q->inf) -> 1 within 1e-3 at q*r = 100 (r the HS radius)
    r = 0.1
    q = np.array([100.0 / r])
    s = st.sq_percus_yevick(q, r, 0.3)[0]
    assert abs(s - 1.0) < 1e-3


def test_py_first_peak_phi049():
    r = 0.1
    A = np.linspace(0.05, 30, 20000)
    q = A / (2 * r)
    s = st.sq_percus_yevick(q, r, 0.49)
    imax = np.argmax(s)
    peak_height = s[imax]
    peak_A = A[imax]
    assert 2.5 <= peak_height <= 3.2
    assert 6.5 <= peak_A <= 7.3


@pytest.mark.parametrize("phi", [0.05, 0.2, 0.4, 0.49])
def test_py_positive_everywhere(phi):
    r = 0.1
    A = np.linspace(1e-4, 40, 5000)
    q = A / (2 * r)
    s = st.sq_percus_yevick(q, r, phi)
    assert np.all(s > 0)


def test_py_no_nan_at_q_zero():
    s = st.sq_percus_yevick(np.array([0.0]), 0.1, 0.3)
    assert np.all(np.isfinite(s))


def test_py_smooth_across_series_threshold():
    # continuity of S(q) across the internal small-A series/exact switch
    r = 0.1
    A = np.array([0.5 * st._A_SMALL, 0.9 * st._A_SMALL,
                  1.1 * st._A_SMALL, 2.0 * st._A_SMALL])
    q = A / (2 * r)
    s = st.sq_percus_yevick(q, r, 0.3)
    assert np.max(np.abs(np.diff(s))) < 1e-3


# ---------------------------------------------------------------------------
# Baxter sticky hard sphere
# ---------------------------------------------------------------------------
def test_baxter_tau_infinity_matches_py():
    r, phi = 0.1, 0.3
    A = np.linspace(0.01, 25, 3000)
    q = A / (2 * r)
    s_baxter = st.sq_baxter(q, r, phi, 1e6)
    s_py = st.sq_percus_yevick(q, r, phi)
    assert np.max(np.abs(s_baxter - s_py)) < 1e-6


def test_baxter_moderate_stickiness_enhances_s0():
    r, phi, tau = 0.1, 0.1, 0.15
    s0_baxter = st.sq_baxter(np.array([1e-8 / (2 * r)]), r, phi, tau)[0]
    s0_py = st.sq_percus_yevick(np.array([1e-8 / (2 * r)]), r, phi)[0]
    assert s0_baxter > s0_py


def test_baxter_unphysical_combo_raises():
    with pytest.raises(ValueError):
        st.sq_baxter(np.array([1.0]), 0.1, 0.2, 0.05)


def test_baxter_positive_everywhere_valid_combo():
    r, phi, tau = 0.1, 0.1, 0.15
    A = np.linspace(1e-3, 40, 5000)
    q = A / (2 * r)
    s = st.sq_baxter(q, r, phi, tau)
    assert np.all(s > 0)


# --- exact-solution cross-checks against SasView (independent transcription)
def _sasview_sticky_iq(q, radius_effective, volfraction, perturb, stickiness):
    """Independent, standalone transcription of SasView's
    sasmodels/models/stickyhardsphere.c Iq() (the exact Baxter/Menon-Manohar-
    Rao PY sticky-hard-sphere S(q)), written straight from the published C
    source and sharing NO code with structure.py. Scalar q; q and
    radius_effective in any consistent inverse-length units. Returns -1.0 on
    the same unphysical branches the C code does."""
    import math
    onemineps = 1.0 - perturb
    eta = volfraction / onemineps ** 3
    sig = 2.0 * radius_effective
    aa = sig / onemineps
    etam1 = 1.0 - eta
    etam1sq = etam1 * etam1
    qa = eta / 6.0
    qb = stickiness + eta / etam1
    qc = (1.0 + eta / 2.0) / etam1sq
    radic = qb * qb - 2.0 * qa * qc
    if radic < 0:
        return -1.0
    radic = math.sqrt(radic)
    lam = (qb - radic) / qa
    lam2 = (qb + radic) / qa
    if lam2 < lam:
        lam = lam2
    test = 1.0 + 2.0 * eta
    mu = lam * eta * etam1
    if mu > test:
        return -1.0
    alpha = (1.0 + 2.0 * eta - mu) / etam1sq
    beta = (mu - 3.0 * eta) / (2.0 * etam1sq)
    kk = q * aa
    k2 = kk * kk
    k3 = kk * k2
    ds = math.sin(kk)
    dc = math.cos(kk)
    aq1 = ((ds - kk * dc) * alpha) / k3
    aq2 = (beta * (1.0 - dc)) / k2
    aq3 = (lam * ds) / (12.0 * kk)
    aq = 1.0 + 12.0 * eta * (aq1 + aq2 - aq3)
    bq1 = alpha * (0.5 / kk - ds / k2 + (1.0 - dc) / k3)
    bq2 = beta * (1.0 / kk - ds / k2)
    bq3 = (lam / 12.0) * ((1.0 - dc) / kk)
    bq = 12.0 * eta * (bq1 + bq2 - bq3)
    return 1.0 / (aq * aq + bq * bq)


def test_baxter_sasview_published_reference_values():
    # SasView's own model test block (sasmodels/models/stickyhardsphere.py):
    #   {radius_effective=50, volfraction=0.1, perturb=0.05, stickiness=0.2}
    #   q = [0.001, 0.003]  ->  Iq = [1.09718, 1.087830]
    # First confirm the independent transcription reproduces SasView's
    # published numbers (validates the algorithm we implemented).
    got = [_sasview_sticky_iq(q, 50.0, 0.1, 0.05, 0.2)
           for q in (0.001, 0.003)]
    assert got[0] == pytest.approx(1.09718, abs=2e-5)
    assert got[1] == pytest.approx(1.087830, abs=2e-5)


def test_baxter_matches_sasview_at_perturb_zero():
    # Our sq_baxter is the delta-shell (perturb -> 0) limit; it must equal
    # the full independent SasView transcription at perturb=0, across a
    # spread of (q, r, phi, tau). This is the exact-formula spot check.
    cases = [
        # (r_um, phi, tau, A=q*sigma probe points)
        (0.1, 0.1, 0.15, [0.2, 1.0, 3.0, 6.5, 12.0, 25.0]),
        (0.1, 0.2, 0.30, [0.1, 0.5, 2.0, 7.0, 15.0]),
        (0.2, 0.3, 0.50, [0.3, 1.5, 5.0, 10.0, 20.0]),
        (0.05, 0.35, 1.0, [0.05, 2.0, 6.0, 14.0]),
    ]
    for r, phi, tau, As in cases:
        for A in As:
            q = A / (2 * r)
            ours = st.sq_baxter(np.array([q]), r, phi, tau)[0]
            ref = _sasview_sticky_iq(q, r, phi, 0.0, tau)
            assert ref > 0
            assert abs(ours - ref) < 1e-9 * max(1.0, ref), \
                "mismatch r=%g phi=%g tau=%g A=%g: %g vs %g" \
                % (r, phi, tau, A, ours, ref)


def test_baxter_s0_matches_full_compressibility():
    # phi=0.1, tau=0.15 is exactly where the OLD fuller-formula attempt went
    # NEGATIVE; the exact solution must be positive, finite, and equal to the
    # closed-form Baxter compressibility S(0) = 1/A(0)^2, with A(0) assembled
    # here by an algebraically-independent reduction
    #   A(0) = 1 + (4 eta - eta^2 - eta*mu)/(1-eta)^2 - eta*lambda .
    r, phi, tau = 0.1, 0.1, 0.15
    eta = phi
    lam = st._baxter_lambda(phi, tau)
    mu = lam * eta * (1.0 - eta)
    a0 = 1.0 + (4.0 * eta - eta ** 2 - eta * mu) / (1.0 - eta) ** 2 - eta * lam
    s0_compress = 1.0 / a0 ** 2
    s0 = st.sq_baxter(np.array([0.0]), r, phi, tau)[0]
    assert np.isfinite(s0) and s0 > 0.0
    assert abs(s0 - s0_compress) < 1e-12
    # and, sanity, it lands in the physically-sensible enhanced-but-finite
    # range (positive, and > the PY value at the same phi)
    s0_py = st.sq_percus_yevick(np.array([0.0]), r, phi)[0]
    assert 0.0 < s0_py < s0 < 5.0


def test_baxter_s0_monotonic_in_tau():
    # stickier (smaller tau) => larger S(0); S(0) must increase monotonically
    # as tau decreases toward the sticky limit.
    r, phi = 0.1, 0.2
    taus = [100.0, 5.0, 1.2, 0.6, 0.3, 0.16]  # decreasing = stickier
    s0 = [st.sq_baxter(np.array([0.0]), r, phi, t)[0] for t in taus]
    assert all(v > 0.0 for v in s0)
    assert all(s0[i] < s0[i + 1] for i in range(len(s0) - 1))
    # the loosest (tau=100) should be within a hair of the PY value
    s0_py = st.sq_percus_yevick(np.array([0.0]), r, phi)[0]
    assert abs(s0[0] - s0_py) < 1e-2 * s0_py


# ---------------------------------------------------------------------------
# Teixeira fractal aggregate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("df", [1.8, 2.5])
def test_fractal_loglog_slope(df):
    # um; wide scale separation (1/xi=1e-3 << q << 1/r0=20) so the window
    # is deep in the pure-power-law fractal regime, not the crossovers.
    xi, r0 = 1000.0, 0.05
    q = np.logspace(np.log10(0.1), np.log10(2.0), 300)
    s = st.sq_fractal(q, xi, df, r0)
    slope = np.polyfit(np.log(q), np.log(s - 1.0), 1)[0]
    assert abs(slope - (-df)) < 0.05


@pytest.mark.parametrize("df", [1.8, 2.5])
def test_fractal_s0_analytic_limit(df):
    xi, r0 = 50.0, 2.0
    limit = 1.0 + df * G(df) * (xi / r0) ** df
    s0 = st.sq_fractal(np.array([0.0]), xi, df, r0)[0]
    assert abs(s0 - limit) / limit < 1e-9
    # and the raw formula converges to the same limit as q -> 0
    s_tiny = st.sq_fractal(np.array([1e-7 / xi]), xi, df, r0)[0]
    assert abs(s_tiny - limit) / limit < 1e-3


def test_fractal_large_q_to_one():
    xi, r0, df = 10.0, 0.05, 2.3
    s = st.sq_fractal(np.array([500.0]), xi, df, r0)[0]
    assert abs(s - 1.0) < 1e-2


def test_fractal_no_nan_at_q_zero():
    s = st.sq_fractal(np.array([0.0]), 20.0, 2.0, 0.5)
    assert np.all(np.isfinite(s))


# ---------------------------------------------------------------------------
# powder paracrystal
# ---------------------------------------------------------------------------
def _first_peak(q, s, lo, hi):
    mask = (q >= lo) & (q <= hi)
    idx = np.argmax(s[mask])
    return q[mask][idx], s[mask][idx]


def test_paracrystal_fcc_first_peak_position():
    a, g = 0.45, 0.05
    q1 = 2 * np.pi / a
    q_expect = q1 * np.sqrt(3)          # (111)
    q = np.linspace(0.5 * q_expect, 1.3 * q_expect, 20000)
    s = st.sq_paracrystal(q, "fcc", a, g)
    qpk, _ = _first_peak(q, s, 0.9 * q_expect, 1.1 * q_expect)
    assert abs(qpk - q_expect) < 0.02 * q_expect


def test_paracrystal_bcc_first_peak_position():
    a, g = 0.45, 0.05
    q1 = 2 * np.pi / a
    q_expect = q1 * np.sqrt(2)          # (110)
    q = np.linspace(0.5 * q_expect, 1.3 * q_expect, 20000)
    s = st.sq_paracrystal(q, "bcc", a, g)
    qpk, _ = _first_peak(q, s, 0.9 * q_expect, 1.1 * q_expect)
    assert abs(qpk - q_expect) < 0.02 * q_expect


def test_paracrystal_sc_first_peak_position():
    a, g = 0.45, 0.05
    q1 = 2 * np.pi / a
    q_expect = q1                        # (100)
    q = np.linspace(0.5 * q_expect, 1.3 * q_expect, 20000)
    s = st.sq_paracrystal(q, "sc", a, g)
    qpk, _ = _first_peak(q, s, 0.9 * q_expect, 1.1 * q_expect)
    assert abs(qpk - q_expect) < 0.02 * q_expect


def test_paracrystal_fcc_forbidden_100_absent():
    a, g = 0.45, 0.05
    q1 = 2 * np.pi / a
    q_111 = q1 * np.sqrt(3)
    q_100 = q1 * 1.0                     # forbidden for fcc
    q = np.linspace(0.5 * q_111, 1.3 * q_111, 40000)
    s = st.sq_paracrystal(q, "fcc", a, g)
    peak_111 = s.max()
    local_100 = s[np.argmin(np.abs(q - q_100))]
    assert peak_111 / local_100 > 5.0


def test_paracrystal_large_g_broad_liquid_like():
    a, g = 0.45, 0.4
    q1 = 2 * np.pi / a
    q = np.linspace(0.5, 15 * q1, 5000)
    s = st.sq_paracrystal(q, "fcc", a, g)
    assert np.all(s > 0)
    assert s.max() < 3.0
    assert abs(s[-1] - 1.0) < 0.1


@pytest.mark.parametrize("lattice", ["fcc", "bcc", "sc"])
def test_paracrystal_positive_everywhere(lattice):
    a, g = 0.45, 0.1
    q = np.linspace(1e-3, 10 * (2 * np.pi / a), 4000)
    s = st.sq_paracrystal(q, lattice, a, g)
    assert np.all(s > 0)


def test_paracrystal_invalid_lattice_raises():
    with pytest.raises(ValueError):
        st.sq_paracrystal(np.array([1.0]), "hcp", 0.45, 0.1)


# ---------------------------------------------------------------------------
# tabulated S(q)
# ---------------------------------------------------------------------------
def test_table_interpolation_and_clamping():
    q_tab = np.array([0.0, 1.0, 2.0, 3.0])
    s_tab = np.array([0.5, 1.5, 1.0, 1.0])
    q = np.array([-1.0, 0.5, 1.5, 3.0, 10.0])
    s = st.sq_table(q, q_tab, s_tab)
    assert s[0] == pytest.approx(0.5)     # clamped below range
    assert s[1] == pytest.approx(1.0)     # midpoint interp 0.5,1.5 -> 1.0
    assert s[2] == pytest.approx(1.25)    # midpoint interp 1.5,1.0 -> 1.25
    assert s[3] == pytest.approx(1.0)
    assert s[4] == pytest.approx(1.0)     # clamped above range


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------
def test_evaluate_none():
    q = np.array([0.5, 1.0, 2.0])
    s = st.sq_evaluate("none", {}, q)
    assert np.allclose(s, 1.0)


def test_evaluate_py_uses_context_defaults():
    q = np.linspace(0.1, 10, 50)
    context = {"phi_v": 0.3, "r_mean_um": 0.1}
    s_ctx = st.sq_evaluate("py", {}, q, context)
    s_direct = st.sq_percus_yevick(q, 0.1, 0.3)
    assert np.allclose(s_ctx, s_direct)


def test_evaluate_py_params_override_context():
    q = np.linspace(0.1, 10, 50)
    params = {"phi_hs": 0.2, "r_hs_um": 0.2}
    context = {"phi_v": 0.3, "r_mean_um": 0.1}
    s_ctx = st.sq_evaluate("py", params, q, context)
    s_direct = st.sq_percus_yevick(q, 0.2, 0.2)
    assert np.allclose(s_ctx, s_direct)


def test_evaluate_py_missing_defaults_raises():
    q = np.array([1.0])
    with pytest.raises(ValueError):
        st.sq_evaluate("py", {}, q, {})


def test_evaluate_baxter():
    q = np.linspace(0.1, 10, 50)
    params = {"tau_stick": 1e6}
    context = {"phi_v": 0.3, "r_mean_um": 0.1}
    s_ctx = st.sq_evaluate("baxter", params, q, context)
    s_direct = st.sq_baxter(q, 0.1, 0.3, 1e6)
    assert np.allclose(s_ctx, s_direct)


def test_evaluate_fractal_uses_context_r0():
    q = np.linspace(0.1, 10, 50)
    params = {"xi_um": 20.0, "df": 2.0}
    context = {"r_mean_um": 0.3}
    s_ctx = st.sq_evaluate("fractal", params, q, context)
    s_direct = st.sq_fractal(q, 20.0, 2.0, 0.3)
    assert np.allclose(s_ctx, s_direct)


def test_evaluate_paracrystal():
    q = np.linspace(0.1, 30, 50)
    params = {"lattice": "bcc", "a_um": 0.45, "g": 0.1}
    s_ctx = st.sq_evaluate("paracrystal", params, q)
    s_direct = st.sq_paracrystal(q, "bcc", 0.45, 0.1)
    assert np.allclose(s_ctx, s_direct)


def test_evaluate_table():
    q = np.array([0.5, 1.5])
    td = {"q_per_um": np.array([0.0, 1.0, 2.0]),
          "s": np.array([1.0, 2.0, 1.0])}
    params = {"table_data": td}
    s_ctx = st.sq_evaluate("table", params, q)
    s_direct = st.sq_table(q, td["q_per_um"], td["s"])
    assert np.allclose(s_ctx, s_direct)


def test_evaluate_unknown_model_raises():
    with pytest.raises(ValueError):
        st.sq_evaluate("bogus", {}, np.array([1.0]))
