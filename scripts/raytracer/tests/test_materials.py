# =============================================================================
# test_materials.py -- pytest suite for scripts/raytracer/materials.py
#
# Run with:
#   /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_materials.py -v
# =============================================================================
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.raytracer.materials import (
    MaterialDB, MaterialError, load_coatings, DEFAULT_MATERIALS_CSV,
    DEFAULT_NK_DIR, DEFAULT_COATINGS_CSV,
)


@pytest.fixture(scope="module")
def db():
    return MaterialDB.load()


# ---------------------------------------------------------------------------
# Dispersion spot-checks against literature/refractiveindex.info values
# ---------------------------------------------------------------------------
def test_bk7_nd(db):
    n = db.get("bk7").n_complex(587.6e-9)
    assert n.real == pytest.approx(1.5168, abs=1e-4)
    assert n.imag == pytest.approx(0.0, abs=1e-12)


def test_fused_silica(db):
    n = db.get("fused_silica").n_complex(587.6e-9)
    assert n.real == pytest.approx(1.4585, abs=2e-4)


def test_mgf2(db):
    n = db.get("mgf2").n_complex(550e-9)
    assert n.real == pytest.approx(1.3777, abs=1e-3)


def test_water(db):
    n = db.get("water").n_complex(589e-9)
    assert n.real == pytest.approx(1.333, abs=1e-3)


def test_sapphire(db):
    n = db.get("sapphire_o").n_complex(589e-9)
    assert n.real == pytest.approx(1.768, abs=2e-3)


def test_polystyrene(db):
    n = db.get("polystyrene").n_complex(589e-9)
    assert n.real == pytest.approx(1.5916, abs=2e-3)


def test_aluminum_visible(db):
    """Cross-check against the raw Rakic 1995 table (linear interp), the
    same source nk_data/aluminum.csv was built from, at ~650nm. The real
    Rakic data show a genuine interband-transition feature near 800nm, so
    n(650nm) is close to ~1.55-1.6 (not the idealized ~1.47 sometimes
    quoted from smoothed Drude fits) -- verified directly against the
    tabulated source below."""
    with open(DEFAULT_NK_DIR / "aluminum.mienk", newline="") as fh:
        rows = list(csv.DictReader(fh))
    lam_nm = np.array([float(r["wavelength_nm"]) for r in rows])
    n_tab = np.array([float(r["n"]) for r in rows])
    k_tab = np.array([float(r["k"]) for r in rows])
    n_expected = np.interp(650.0, lam_nm, n_tab)
    k_expected = np.interp(650.0, lam_nm, k_tab)

    n = db.get("aluminum").n_complex(650e-9)
    assert isinstance(n, complex)
    assert n.real == pytest.approx(n_expected, rel=0.05)
    assert n.imag == pytest.approx(k_expected, rel=0.05)
    assert n.imag > 1.0  # aluminum is strongly absorbing in the visible


# ---------------------------------------------------------------------------
# Loader / validation behaviour
# ---------------------------------------------------------------------------
def test_tabulated_out_of_range_raises(db):
    mat = db.get("aluminum")
    with pytest.raises(ValueError):
        mat.n_complex(50e-9)  # far below the tabulated 207-1240nm range
    with pytest.raises(ValueError):
        mat.n_complex(5000e-9)  # far above


def test_unknown_material_lists_available(db):
    with pytest.raises(KeyError) as excinfo:
        db.get("unobtainium")
    msg = str(excinfo.value)
    assert "unobtainium" in msg
    assert "bk7" in msg.lower()


def test_vectorized(db):
    mat = db.get("bk7")
    lam = np.array([486.1e-9, 587.6e-9, 656.3e-9])
    n = mat.n_complex(lam)
    assert isinstance(n, np.ndarray)
    assert n.dtype == np.complex128
    assert n.shape == lam.shape
    # dispersion: n should decrease with increasing wavelength (normal
    # dispersion) over the visible range for BK7
    assert n[0].real > n[1].real > n[2].real


def test_coating_quarterwave(db):
    coatings = load_coatings(db=db)
    assert coatings["MgF2"]["kind"] == "tmm"
    layers = coatings["MgF2"]["layers"]
    assert len(layers) == 1
    mat_name, spec = layers[0]
    assert mat_name == "mgf2"
    assert isinstance(spec, tuple) and spec[0] == "qw"
    lam0_m = spec[1]
    assert lam0_m == pytest.approx(550e-9, rel=1e-6)
    n0 = db.get("mgf2").n_complex(lam0_m).real
    thickness_m = lam0_m / (4.0 * n0)
    expected = 550e-9 / (4.0 * 1.3777)
    assert thickness_m == pytest.approx(expected, rel=0.01)


def test_malformed_csv_rejected(tmp_path):
    bad_csv = tmp_path / "materials_bad.csv"
    bad_csv.write_text(
        "name,class,model,p1,p2,p3,p4,p5,p6,nk_file,density_kg_m3,"
        "transmission_um_min,transmission_um_max,notes,reference\n"
        "bogus,not_a_class,constant,1.5,0.0,,,,,,1000,,,,\"some ref\"\n"
    )
    with pytest.raises(ValueError):
        MaterialDB.load(csv_path=bad_csv, nk_dir=DEFAULT_NK_DIR)


def test_malformed_csv_bad_model(tmp_path):
    bad_csv = tmp_path / "materials_bad2.csv"
    bad_csv.write_text(
        "name,class,model,p1,p2,p3,p4,p5,p6,nk_file,density_kg_m3,"
        "transmission_um_min,transmission_um_max,notes,reference\n"
        "bogus,glass,not_a_model,1.5,0.0,,,,,,1000,,,,\"some ref\"\n"
    )
    with pytest.raises(ValueError):
        MaterialDB.load(csv_path=bad_csv, nk_dir=DEFAULT_NK_DIR)


def test_sellmeier_negative_c_now_accepted(tmp_path):
    # A negative Sellmeier C is mathematically well-behaved (no real pole), so
    # the loader now accepts it (relaxed from the old C>0 hard-reject to allow
    # genuine catalog fits). It must still load and evaluate to a finite index.
    ok_csv = tmp_path / "materials_negc.csv"
    ok_csv.write_text(
        "name,class,model,p1,p2,p3,p4,p5,p6,nk_file,density_kg_m3,"
        "transmission_um_min,transmission_um_max,notes,reference\n"
        "negc,glass,sellmeier,1.0,0.2,0.9,0.006,-0.01,103.5,,2500,,,,\"some ref\"\n"
    )
    db = MaterialDB.load(csv_path=ok_csv, nk_dir=DEFAULT_NK_DIR)
    assert np.isfinite(db.get("negc").n_complex(550e-9).real)


def test_sellmeier_nonfinite_c_rejected(tmp_path):
    # ... but a non-finite (blank/NaN) required Sellmeier parameter is still an
    # error (a required parameter, not a relaxed constraint).
    bad_csv = tmp_path / "materials_bad3.csv"
    bad_csv.write_text(
        "name,class,model,p1,p2,p3,p4,p5,p6,nk_file,density_kg_m3,"
        "transmission_um_min,transmission_um_max,notes,reference\n"
        "badsell,glass,sellmeier,1.0,0.0,0.0,0.006,,103.5,,2500,,,,\"some ref\"\n"
    )
    with pytest.raises(ValueError):
        MaterialDB.load(csv_path=bad_csv, nk_dir=DEFAULT_NK_DIR)


def test_missing_reference_rejected(tmp_path):
    bad_csv = tmp_path / "materials_bad4.csv"
    bad_csv.write_text(
        "name,class,model,p1,p2,p3,p4,p5,p6,nk_file,density_kg_m3,"
        "transmission_um_min,transmission_um_max,notes,reference\n"
        "noref,glass,constant,1.5,0.0,,,,,,1000,,,,\n"
    )
    with pytest.raises(ValueError):
        MaterialDB.load(csv_path=bad_csv, nk_dir=DEFAULT_NK_DIR)


def test_coating_unknown_material_rejected(tmp_path, db):
    bad_csv = tmp_path / "coatings_bad.csv"
    bad_csv.write_text(
        'name,layers,reference\n'
        'BadCoat,unobtainium:100.0,"bad reference"\n'
    )
    with pytest.raises(MaterialError):
        load_coatings(csv_path=bad_csv, db=db)


def test_detector_and_vacuum_sentinels(db):
    n_det = db.get("detector").n_complex(550e-9)
    assert n_det.real == pytest.approx(1.0)
    assert n_det.imag == pytest.approx(0.0)
    n_vac = db.get("vacuum").n_complex(550e-9)
    assert n_vac.real == pytest.approx(1.0)


def test_water_absorption_shoulder(db):
    """Sanity check the water k table: overall NIR absorption rise, with
    the weak first-overtone shoulder around 975nm sitting above the
    500nm baseline."""
    mat = db.get("water")
    k500 = mat.n_complex(500e-9).imag
    k900 = mat.n_complex(900e-9).imag
    assert k900 > k500


def test_aluminum_k_rises_into_ir(db):
    mat = db.get("aluminum")
    k_vis = mat.n_complex(500e-9).imag
    k_ir = mat.n_complex(1200e-9).imag
    assert k_ir > k_vis


def test_default_paths_exist():
    assert DEFAULT_MATERIALS_CSV.exists()
    assert DEFAULT_COATINGS_CSV.exists()
    assert DEFAULT_NK_DIR.is_dir()


def test_used_names_and_case_insensitive_get(db):
    names = db.used_names()
    assert "bk7" in [n.lower() for n in names]
    assert db.get("BK7") is db.get("bk7")
