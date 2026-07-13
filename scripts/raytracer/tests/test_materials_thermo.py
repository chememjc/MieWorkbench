# =============================================================================
# test_materials_thermo.py -- Schott power-series dispersion model, thermo-optic
# n(lambda, T) (Schott TIE-19), and the relaxed Sellmeier-C validator.
#
#   /home3/optics/env/bin/python -m pytest \
#       scripts/raytracer/tests/test_materials_thermo.py -v
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))            # scenehelpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # scripts/

from scripts.raytracer.materials import (
    Material, MaterialDB, MaterialError, DEFAULT_T_REF_C,
)

# Standard Schott N-BK7 TIE-19 thermo-optic constants (D0,D1,D2,E0,E1,lam_tk um)
NBK7_THERMO = (1.86e-6, 1.31e-8, -1.37e-11, 4.34e-7, 6.27e-10, 0.170)
# Schott N-BK7 Sellmeier-1 coefficients (B1,B2,B3,C1,C2,C3)
NBK7_SELLMEIER = (1.03961212, 0.231792344, 1.01046945,
                  0.00600069867, 0.0200179144, 103.560653)

MAT_HEADER = ("name,class,model,p1,p2,p3,p4,p5,p6,nk_file,density_kg_m3,"
              "transmission_um_min,transmission_um_max,notes,reference,"
              "thermo_d0,thermo_d1,thermo_d2,thermo_e0,thermo_e1,"
              "thermo_lambda_tk,thermo_t_ref_c\n")


def _write_db(tmp_path, rows):
    p = tmp_path / "materials.miemat"
    p.write_text(MAT_HEADER + "".join(r + "\n" for r in rows))
    return MaterialDB.load(csv_path=p, nk_dir=tmp_path / "nk")


# ---------------------------------------------------------------------------
# Schott power-series model
# ---------------------------------------------------------------------------
def test_schott_model_matches_sellmeier_glass():
    """A Schott power-series fit of N-BK7 (from its Sellmeier form) reproduces
    the index across the visible to a few 1e-4."""
    sell = Material("bk7s", "glass", "sellmeier", NBK7_SELLMEIER, 2510,
                    reference="test")
    lams = np.linspace(0.45, 0.65, 9) * 1e-6
    # Fit n^2 = a0 + a1 l^2 + a2 l^-2 + a3 l^-4 + a4 l^-6 + a5 l^-8 to sellmeier
    lu = lams * 1e6
    n2 = np.real(sell.n_complex(lams)) ** 2
    A = np.vstack([np.ones_like(lu), lu**2, lu**-2, lu**-4, lu**-6, lu**-8]).T
    coeffs, *_ = np.linalg.lstsq(A, n2, rcond=None)
    schott = Material("bk7schott", "glass", "schott", list(coeffs), 2510,
                      reference="test")
    n_schott = np.real(schott.n_complex(lams))
    n_sell = np.real(sell.n_complex(lams))
    assert np.max(np.abs(n_schott - n_sell)) < 5e-4
    # derivative is finite and negative (normal dispersion: n falls with lambda)
    assert schott.dn_dlam(0.5876e-6) < 0


# ---------------------------------------------------------------------------
# Thermo-optic n(lambda, T) -- Schott TIE-19
# ---------------------------------------------------------------------------
def _bk7_thermo():
    return Material("bk7t", "glass", "sellmeier", NBK7_SELLMEIER, 2510,
                    reference="Schott N-BK7 TIE-19", thermo=NBK7_THERMO,
                    t_ref_c=20.0)


def test_thermo_self_consistency():
    """Finite-difference dn/dT matches the analytic Schott dn_abs/dT(0)."""
    m = _bk7_thermo()
    lam = 587.6e-9
    n0 = m.n_complex(lam, T=20.0).real
    fd = (m.n_complex(lam, T=20.5).real - m.n_complex(lam, T=19.5).real) / 1.0
    D0, D1, D2, E0, E1, ltk = NBK7_THERMO
    l2 = 0.5876 ** 2
    analytic = (n0**2 - 1) / (2 * n0) * (D0 + E0 / (l2 - ltk**2))
    assert fd == pytest.approx(analytic, rel=1e-3)


def test_thermo_nbk7_published_magnitude():
    """Physics oracle: N-BK7 dn_abs/dT at 587.6 nm / 20 C is ~1.4e-6 /K
    (positive; crown glass), matching the standard TIE-19 constants."""
    m = _bk7_thermo()
    lam = 587.6e-9
    dndt = m.n_complex(lam, T=21.0).real - m.n_complex(lam, T=20.0).real
    assert dndt > 0                        # heating raises the index
    assert dndt == pytest.approx(1.39e-6, abs=2e-7)


def test_thermo_none_and_absent_leave_n_unchanged():
    m = _bk7_thermo()
    lam = 587.6e-9
    base = m.n_complex(lam).real
    assert m.n_complex(lam, T=None).real == base          # T not requested
    assert m.n_complex(lam, T=20.0).real == base          # T == t_ref
    plain = Material("bk7p", "glass", "sellmeier", NBK7_SELLMEIER, 2510,
                     reference="test")
    assert not plain.has_thermo
    assert plain.n_complex(lam, T=80.0).real == plain.n_complex(lam).real


def test_thermo_array_input():
    m = _bk7_thermo()
    lams = np.array([480e-9, 587.6e-9, 656e-9])
    out = m.n_complex(lams, T=60.0)
    assert out.shape == (3,)
    assert np.all(out.real > m.n_complex(lams).real)      # all shifted up


# ---------------------------------------------------------------------------
# Loader: thermo columns + relaxed Sellmeier-C + backward compat
# ---------------------------------------------------------------------------
def test_loader_reads_thermo_columns(tmp_path):
    B = NBK7_SELLMEIER
    row = ("bk7,glass,sellmeier,%g,%g,%g,%g,%g,%g,,2510,0.35,2.5,,ref,"
           "%g,%g,%g,%g,%g,%g,20" % (B + NBK7_THERMO))
    db = _write_db(tmp_path, [row])
    m = db.get("bk7")
    assert m.has_thermo
    assert m.t_ref_c == 20.0
    assert m.n_complex(587.6e-9, T=21.0).real > m.n_complex(587.6e-9).real


def test_partial_thermo_is_error(tmp_path):
    B = NBK7_SELLMEIER
    # only D0 set -> partial model must hard-error
    row = ("bad,glass,sellmeier,%g,%g,%g,%g,%g,%g,,2510,,,,ref,"
           "1.86e-6,,,,,," % B)
    with pytest.raises(MaterialError, match="partial thermo"):
        _write_db(tmp_path, [row])


def test_relaxed_sellmeier_allows_nonpositive_C(tmp_path):
    # a Sellmeier row with a zero and a negative C now loads (mathematically
    # well-behaved; previously hard-rejected)
    row = "weird,glass,sellmeier,1.0,0.2,0.9,0.0,-0.01,100.0,,2500,,,,ref"
    db = _write_db(tmp_path, [row])
    n = db.get("weird").n_complex(550e-9)
    assert np.isfinite(n.real)


def test_schott_model_via_loader(tmp_path):
    row = "sch,glass,schott,2.30,0.008,0.008,0.0,0.0,0.0,,2500,,,,ref"
    db = _write_db(tmp_path, [row])
    assert np.isfinite(db.get("sch").n_complex(587.6e-9).real)


def test_shipped_db_backward_compat():
    """The shipped 168-material DB still loads and no existing row has a
    thermo model yet (added later by the AGF importer)."""
    db = MaterialDB.load()
    assert len(db) == 168
    assert not db.get("bk7").has_thermo
    assert db.get("bk7").t_ref_c == DEFAULT_T_REF_C


# ---------------------------------------------------------------------------
# End-to-end: temperature threads through the Scene and reaches medium_index,
# and cengine routes a real thermo-optic shift to the Python engine.
# ---------------------------------------------------------------------------
def _thermo_db(tmp_path):
    p = tmp_path / "materials.miemat"
    B = NBK7_SELLMEIER
    T = NBK7_THERMO
    p.write_text(MAT_HEADER
                 + "air,gas,constant,1.0,0.0,,,,,,1.2,,,,ref\n"
                 + ("bk7,glass,sellmeier,%g,%g,%g,%g,%g,%g,,2510,0.35,2.5,,ref,"
                    "%g,%g,%g,%g,%g,%g,20\n" % (B + T)))
    return MaterialDB.load(csv_path=p, nk_dir=tmp_path / "nk")


def _glass_body_index(scene):
    for i, b in enumerate(scene.bodies):
        if b.material == "bk7":
            return i
    raise AssertionError("no bk7 body in scene")


def test_scene_global_temperature_shifts_medium_index(tmp_path):
    import scenehelpers as sh
    from raytracer.scene import Scene
    matdb = _thermo_db(tmp_path)
    model = sh.make_model([sh.source_body(), sh.slab_body("glass", "bk7", -0.01, 0.01), sh.detector_body()])
    lam = 587.6e-9
    cold = Scene(model, matdb, {}, optprops=None)                 # no temp
    hot = Scene(model, matdb, {}, optprops=None, temperature_c=80.0)
    gi = _glass_body_index(cold)
    n_ref = cold.medium_index(gi, lam).real
    n_hot = hot.medium_index(_glass_body_index(hot), lam).real
    assert n_hot > n_ref                                          # heated -> up
    # matches the material's own thermo shift
    expect = matdb.get("bk7").n_complex(lam, T=80.0).real
    assert n_hot == pytest.approx(expect, rel=1e-12)


def test_per_body_temperature_overrides_scene(tmp_path):
    import scenehelpers as sh
    from raytracer.scene import Scene
    matdb = _thermo_db(tmp_path)
    # body carries its own 40 C; scene-global is 80 C -> body wins
    model = sh.make_model([sh.source_body(),
                           sh.slab_body("glass", "bk7", -0.01, 0.01,
                                        temperature=40.0),
                           sh.detector_body()])
    scene = Scene(model, matdb, {}, optprops=None, temperature_c=80.0)
    gi = _glass_body_index(scene)
    got = scene.medium_index(gi, 587.6e-9).real
    expect = matdb.get("bk7").n_complex(587.6e-9, T=40.0).real
    assert got == pytest.approx(expect, rel=1e-12)


def test_cengine_routes_temperature_to_python(tmp_path):
    import types
    import scenehelpers as sh
    from raytracer.scene import Scene
    from raytracer import cengine
    matdb = _thermo_db(tmp_path)
    model = sh.make_model([sh.source_body(), sh.slab_body("glass", "bk7", -0.01, 0.01), sh.detector_body()])
    scene = Scene(model, matdb, {}, optprops=None, temperature_c=80.0)
    import cli_specs
    tp = cli_specs.build_parser("trace")
    args = types.SimpleNamespace(**{a.dest: a.default for a in tp._actions})
    feats = cengine.detect_features(args, scene)
    assert "temperature" in feats
    assert "temperature" not in cengine.PORTED
    # a scene with no temperature set does NOT emit the token
    plain = Scene(model, matdb, {}, optprops=None)
    assert "temperature" not in cengine.detect_features(args, plain)
