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
