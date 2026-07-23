# =============================================================================
# test_dispersion.py -- dispersion-derivative API (group index, GDD, TOD)
#
# Oracle values are literature-pinned (RP Photonics / refractiveindex.info,
# both derived from the same Sellmeier rows the library ships):
#   fused silica @ 800 nm:  n_g = 1.46714,  GVD = +36.16 fs^2/mm
#   N-BK7      @ 800 nm:                    GVD = +44.65 fs^2/mm
#   N-BK7      @ 1064 nm:  n_g = 1.52065,   GVD = +22.37 fs^2/mm
#   fused silica zero-dispersion crossing ~1.27 um (GVD < 0 at 1550 nm)
#   fused silica TOD @ 800 nm ~ +27.5 fs^3/mm
#
# Run with:
#   "$MIEWB_OPTICS_PYTHON" -m pytest scripts/raytracer/tests/test_dispersion.py -v
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.raytracer.materials import (
    Material, MaterialDB, gdd_per_length, tod_per_length, C_LIGHT_M_S,
)
from scripts.raytracer.birefringence import n_e_theta, n_group_e_theta
from scripts.raytracer.optprops import load_optical_properties

FS2_PER_MM = 1e27          # s^2/m -> fs^2/mm
FS3_PER_MM = 1e42          # s^3/m -> fs^3/mm


@pytest.fixture(scope="module")
def db():
    return MaterialDB.load()


@pytest.fixture(scope="module")
def props():
    return load_optical_properties()


# ---------------------------------------------------------------------------
# group index vs literature
# ---------------------------------------------------------------------------
def test_ng_fused_silica_800(db):
    ng = db.get("fused_silica").n_group(800e-9)
    assert abs(ng - 1.467145) < 1e-4


def test_ng_bk7_1064(db):
    ng = db.get("bk7").n_group(1064e-9)
    assert abs(ng - 1.520645) < 1e-4


def test_ng_exceeds_phase_index_in_normal_dispersion(db):
    # dn/dlam < 0 (normal dispersion) => n_g > n
    for name, lam in (("fused_silica", 800e-9), ("bk7", 550e-9),
                      ("caf2", 800e-9)):
        m = db.get(name)
        assert m.n_group(lam) > np.real(m.n_complex(lam))
        assert m.dn_dlam(lam) < 0.0


# ---------------------------------------------------------------------------
# GVD / GDD-per-length vs literature
# ---------------------------------------------------------------------------
def test_gvd_fused_silica_800(db):
    gvd = gdd_per_length(db.get("fused_silica"), 800e-9) * FS2_PER_MM
    assert abs(gvd - 36.16) / 36.16 < 0.01


def test_gvd_bk7_800(db):
    gvd = gdd_per_length(db.get("bk7"), 800e-9) * FS2_PER_MM
    assert abs(gvd - 44.65) / 44.65 < 0.01


def test_gvd_bk7_1064(db):
    gvd = gdd_per_length(db.get("bk7"), 1064e-9) * FS2_PER_MM
    assert abs(gvd - 22.37) / 22.37 < 0.01


def test_fused_silica_anomalous_at_1550(db):
    # zero-dispersion wavelength ~1.27 um: GVD must flip sign by 1550 nm
    m = db.get("fused_silica")
    assert gdd_per_length(m, 1064e-9) > 0.0
    assert gdd_per_length(m, 1550e-9) < 0.0


def test_tod_fused_silica_800(db):
    tod = tod_per_length(db.get("fused_silica"), 800e-9) * FS3_PER_MM
    assert abs(tod - 27.5) / 27.5 < 0.10


# ---------------------------------------------------------------------------
# model coverage / cross-checks
# ---------------------------------------------------------------------------
def test_constant_model_zero_derivative():
    m = Material("test_const", "glass", "constant",
                 (1.5, 0.0, 0.0, 0.0, 0.0, 0.0), 2500.0)
    assert m.dn_dlam(633e-9) == 0.0
    assert m.n_group(633e-9) == pytest.approx(1.5)
    assert m.d2n_dlam2(633e-9) == 0.0


def test_cauchy_analytic_vs_numeric():
    # dn/dlam analytic must match a central difference of n itself
    m = Material("test_cauchy", "glass", "cauchy",
                 (1.5, 0.004, 3e-5, 0.0, 0.0, 0.0), 2500.0)
    lam = 600e-9
    h = lam * 1e-4
    numeric = (np.real(m.n_complex(lam + h))
               - np.real(m.n_complex(lam - h))) / (2 * h)
    assert m.dn_dlam(lam) == pytest.approx(numeric, rel=1e-6)


def test_sellmeier_analytic_vs_numeric(db):
    m = db.get("fused_silica")
    lam = 800e-9
    h = lam * 1e-4
    numeric = (np.real(m.n_complex(lam + h))
               - np.real(m.n_complex(lam - h))) / (2 * h)
    assert m.dn_dlam(lam) == pytest.approx(numeric, rel=1e-6)


def test_tabulated_matches_analytic(db):
    # sample the fused-silica Sellmeier onto a fine table; the tabulated
    # path's knot-gradient derivative must track the analytic one
    src = db.get("fused_silica")
    lam_um = np.linspace(0.4, 1.6, 1201)          # 1 nm spacing
    tab = Material("test_tab", "glass", "tabulated",
                   (float("nan"),) * 6, 2203.0, nk_file="synthetic")
    tab.nk_lambda_um = lam_um
    tab.nk_n = np.real(src.n_complex(lam_um * 1e-6))
    tab.nk_k = np.zeros_like(lam_um)

    lam = 800e-9
    assert tab.dn_dlam(lam) == pytest.approx(src.dn_dlam(lam), rel=1e-3)
    assert tab.n_group(lam) == pytest.approx(src.n_group(lam), abs=2e-5)
    # second derivative from a linear-interp table is knot-scale approximate
    assert (gdd_per_length(tab, lam)
            == pytest.approx(gdd_per_length(src, lam), rel=0.05))


def test_tabulated_stencil_clamps_at_table_edge(db):
    src = db.get("fused_silica")
    lam_um = np.linspace(0.7, 0.9, 201)
    tab = Material("test_tab_edge", "glass", "tabulated",
                   (float("nan"),) * 6, 2203.0, nk_file="synthetic")
    tab.nk_lambda_um = lam_um
    tab.nk_n = np.real(src.n_complex(lam_um * 1e-6))
    tab.nk_k = np.zeros_like(lam_um)
    # at the very edge the stencil must clamp inside, not raise
    val = tab.d2n_dlam2(0.7005e-6)
    assert np.isfinite(val)


def test_vectorized_shapes(db):
    m = db.get("bk7")
    lam = np.array([500e-9, 800e-9, 1064e-9])
    for fn in (m.dn_dlam, m.n_group, m.d2n_dlam2, m.d3n_dlam3):
        out = fn(lam)
        assert np.shape(out) == (3,)
    assert isinstance(m.dn_dlam(800e-9), float)
    assert isinstance(m.n_group(800e-9), float)


# ---------------------------------------------------------------------------
# birefringent directional group index
# ---------------------------------------------------------------------------
def test_n_group_e_theta_limits(props):
    mat_o, mat_e = props.matdb.get_uniaxial("calcite")
    lam = 590e-9
    # along the axis (cos=1) the e-wave IS the o-wave: group index of o
    ng_axis = n_group_e_theta(np.array([1.0]), mat_o, mat_e, lam)
    assert float(ng_axis[0]) == pytest.approx(mat_o.n_group(lam), abs=1e-6)
    # perpendicular (cos=0): pure e group index
    ng_perp = n_group_e_theta(np.array([0.0]), mat_o, mat_e, lam)
    assert float(ng_perp[0]) == pytest.approx(mat_e.n_group(lam), abs=1e-6)
    # in between: strictly between the two (calcite: n_e < n_o)
    ng_45 = float(n_group_e_theta(np.array([np.cos(np.pi / 4)]),
                                  mat_o, mat_e, lam)[0])
    lo, hi = sorted((mat_e.n_group(lam), mat_o.n_group(lam)))
    assert lo < ng_45 < hi


def test_n_group_e_theta_exceeds_phase(props):
    # normal dispersion: directional group index > directional phase index
    mat_o, mat_e = props.matdb.get_uniaxial("calcite")
    lam = 590e-9
    cos_kc = np.array([np.cos(np.pi / 3)])
    n_phase = n_e_theta(cos_kc,
                        np.real(mat_o.n_complex(lam)),
                        np.real(mat_e.n_complex(lam)))
    n_grp = n_group_e_theta(cos_kc, mat_o, mat_e, lam)
    assert float(n_grp[0]) > float(n_phase[0])
