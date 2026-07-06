# =============================================================================
# test_opticalproperties.py — loaders for the opticalproperties/ library
# (optprops.py) + the shared per-face / polarization spec parsers in
# common.py. Synthetic tmp_path fixtures; independent of the shipped data.
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import common  # noqa: E402
from raytracer.materials import (MaterialDB, MaterialError,  # noqa: E402
                                 load_coatings)
from raytracer import optprops  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures: a minimal synthetic opticalproperties tree
# ---------------------------------------------------------------------------
MAT_HEADER = ("name,class,model,p1,p2,p3,p4,p5,p6,nk_file,density_kg_m3,"
              "transmission_um_min,transmission_um_max,notes,reference\n")


@pytest.fixture
def optroot(tmp_path):
    root = tmp_path / "opticalproperties"
    for sub in ("nk", "birefringence", "polarizer/tables", "filter/tables",
                "coating/tables", "grating/tables"):
        (root / sub).mkdir(parents=True)
    (root / "materials.csv").write_text(
        MAT_HEADER
        + 'air,gas,constant,1.000272,0,,,,,,1.2,,,,"Ciddor 1996"\n'
        + 'cal_o,glass,constant,1.658,0,,,,,,2711,,,,"test o"\n'
        + 'cal_e,glass,constant,1.486,0,,,,,,2711,,,,"test e"\n')
    (root / "coating" / "coatings.csv").write_text(
        'name,layers,table,aoi_deg,reference\n'
        'tab45,,tab45.csv,45,"test table coating"\n')
    (root / "coating" / "tables" / "tab45.csv").write_text(
        "wavelength_nm,Rs,Rp,Ts,Tp\n"
        "400,0.98,0.05,0.01,0.94\n"
        "700,0.99,0.04,0.005,0.95\n")
    (root / "birefringence" / "uniaxial.csv").write_text(
        'name,n_o_material,n_e_material,reference,notes\n'
        'calcite,cal_o,cal_e,"Ghosh 1999","negative uniaxial"\n')
    (root / "polarizer" / "polarizers.csv").write_text(
        'name,type,table_csv,retardance_waves,reference\n'
        'lp_test,linear,lp_test.csv,,"synthetic"\n'
        'cp_test,circular_left,lp_test.csv,0.25,"synthetic"\n')
    (root / "polarizer" / "tables" / "lp_test.csv").write_text(
        "wavelength_nm,T_parallel,T_perpendicular\n"
        "400,0.80,1e-4\n"
        "700,0.85,2e-4\n")
    (root / "filter" / "filters.csv").write_text(
        'name,table_csv,ref_thickness_mm,reference\n'
        'f_test,f_test.csv,2.0,"synthetic"\n')
    (root / "filter" / "tables" / "f_test.csv").write_text(
        "wavelength_nm,transmittance_internal\n"
        "400,1e-4\n"
        "550,0.90\n"
        "700,1e-4\n")
    (root / "grating" / "gratings.csv").write_text(
        'name,model,lines_per_mm,params,table_csv,reference\n'
        'vbg,bragg_kogelnik,1800,thickness_um=3000;dn=0.0005,,"OptiGrate"\n'
        'dmn,dammann,100,"transitions=0.03863,0.39084",,"Dammann 1971"\n'
        'tbl,table,600,,tbl.csv,"synthetic"\n')
    (root / "grating" / "tables" / "tbl.csv").write_text(
        "wavelength_nm,order,eta_s,eta_p\n"
        "400,0,0.10,0.20\n700,0,0.15,0.25\n"
        "400,-1,0.60,0.50\n700,-1,0.70,0.55\n")
    return root


@pytest.fixture
def props(optroot):
    return optprops.load_optical_properties(root=optroot)


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------
def test_full_tree_loads(props):
    assert len(props.matdb) == 3
    assert set(props.coatings) == {"tab45"}
    assert set(props.polarizers) == {"lp_test", "cp_test"}
    assert set(props.filters) == {"f_test"}
    assert set(props.gratings) == {"vbg", "dmn", "tbl"}
    assert set(props.uniaxial) == {"calcite"}


def test_uniaxial_attached_to_db(props):
    db = props.matdb
    assert db.is_birefringent("calcite")
    assert db.is_birefringent("CALCITE")   # case-insensitive like get()
    assert not db.is_birefringent("air")
    mo, me = db.get_uniaxial("calcite")
    assert mo.n_complex(550e-9).real == pytest.approx(1.658)
    assert me.n_complex(550e-9).real == pytest.approx(1.486)


def test_table_coating_shape(props):
    spec = props.coatings["tab45"]
    assert spec["kind"] == "table"
    assert spec["aoi_deg"] == 45.0
    assert spec["Rs"].shape == spec["lam_um"].shape
    # interpolation inside range
    rs = optprops.interp_hard(0.55, spec["lam_um"], spec["Rs"], "t")
    assert 0.98 <= float(rs) <= 0.99


def test_polarizer_extinction_ratio(props):
    p = props.polarizers["lp_test"]
    er = p["T_par"] / p["T_perp"]
    assert er[0] == pytest.approx(8000.0)
    assert p["retardance_waves"] == 0.25   # default
    assert props.polarizers["cp_test"]["type"] == "circular_left"


def test_filter_alpha_roundtrip(props):
    f = props.filters["f_test"]
    # T(d_ref) must round-trip through alpha to 1e-9
    T = np.exp(-f["alpha_per_m"] * f["ref_thickness_m"])
    assert T[1] == pytest.approx(0.90, rel=1e-9)
    assert np.all(f["alpha_per_m"] >= 0)


def test_grating_registry(props):
    vbg = props.gratings["vbg"]
    assert vbg["model"] == "bragg_kogelnik"
    assert vbg["params"]["thickness_um"] == 3000.0
    dmn = props.gratings["dmn"]
    assert dmn["params"]["transitions"] == [0.03863, 0.39084]
    tbl = props.gratings["tbl"]["table"]
    assert set(tbl) == {0, -1}
    assert tbl[-1]["eta_s"][0] == 0.60


def test_optional_categories_absent(optroot):
    # a trimmed library without polarizer/filter/grating csvs still loads
    for f in ("polarizer/polarizers.csv", "filter/filters.csv",
              "grating/gratings.csv", "birefringence/uniaxial.csv"):
        (optroot / f).unlink()
    props = optprops.load_optical_properties(root=optroot)
    assert props.polarizers == {} and props.filters == {}
    assert props.gratings == {} and props.uniaxial == {}
    assert not props.matdb.is_birefringent("calcite")


# ---------------------------------------------------------------------------
# hard-validation failures
# ---------------------------------------------------------------------------
def test_interp_hard_rejects_outside_range():
    with pytest.raises(MaterialError):
        optprops.interp_hard(0.3, np.array([0.4, 0.7]),
                             np.array([1.0, 2.0]), "ctx")


def test_coating_table_energy_violation(optroot):
    (optroot / "coating" / "tables" / "tab45.csv").write_text(
        "wavelength_nm,Rs,Rp,Ts,Tp\n400,0.9,0.05,0.2,0.94\n700,0.9,0.04,0.2,0.95\n")
    db = MaterialDB.load(csv_path=optroot / "materials.csv",
                         nk_dir=optroot / "nk")
    with pytest.raises(MaterialError, match="R\\+T<=1"):
        load_coatings(csv_path=optroot / "coating" / "coatings.csv", db=db)


def test_coating_both_layers_and_table_rejected(optroot):
    (optroot / "coating" / "coatings.csv").write_text(
        'name,layers,table,aoi_deg,reference\n'
        'bad,air:100.0,tab45.csv,45,"ref"\n')
    db = MaterialDB.load(csv_path=optroot / "materials.csv",
                         nk_dir=optroot / "nk")
    with pytest.raises(MaterialError, match="exactly one"):
        load_coatings(csv_path=optroot / "coating" / "coatings.csv", db=db)


def test_polarizer_bad_type_rejected(optroot):
    (optroot / "polarizer" / "polarizers.csv").write_text(
        'name,type,table_csv,retardance_waves,reference\n'
        'bad,diagonal,lp_test.csv,,"ref"\n')
    with pytest.raises(MaterialError, match="type"):
        optprops.load_polarizers(optroot / "polarizer" / "polarizers.csv")


def test_polarizer_swapped_axes_rejected(optroot):
    (optroot / "polarizer" / "tables" / "lp_test.csv").write_text(
        "wavelength_nm,T_parallel,T_perpendicular\n"
        "400,1e-4,0.80\n700,2e-4,0.85\n")
    with pytest.raises(MaterialError, match="T_perpendicular"):
        optprops.load_polarizers(optroot / "polarizer" / "polarizers.csv")


def test_filter_zero_transmittance_rejected(optroot):
    (optroot / "filter" / "tables" / "f_test.csv").write_text(
        "wavelength_nm,transmittance_internal\n400,0.0\n700,0.9\n")
    with pytest.raises(MaterialError, match="floor"):
        optprops.load_filters(optroot / "filter" / "filters.csv")


def test_grating_table_over_unity_rejected(optroot):
    (optroot / "grating" / "tables" / "tbl.csv").write_text(
        "wavelength_nm,order,eta_s,eta_p\n"
        "400,0,0.6,0.2\n700,0,0.6,0.25\n"
        "400,-1,0.6,0.5\n700,-1,0.7,0.55\n")
    with pytest.raises(MaterialError, match="exceed 1"):
        optprops.load_gratings(optroot / "grating" / "gratings.csv")


def test_uniaxial_unknown_material_rejected(optroot):
    (optroot / "birefringence" / "uniaxial.csv").write_text(
        'name,n_o_material,n_e_material,reference,notes\n'
        'quartz,quartz_o,quartz_e,"Ghosh",""\n')
    db = MaterialDB.load(csv_path=optroot / "materials.csv",
                         nk_dir=optroot / "nk")
    with pytest.raises(MaterialError, match="unknown material"):
        optprops.load_uniaxial(optroot / "birefringence" / "uniaxial.csv",
                               db=db)


def test_missing_reference_rejected(optroot):
    (optroot / "filter" / "filters.csv").write_text(
        'name,table_csv,ref_thickness_mm,reference\n'
        'noref,f_test.csv,2.0,\n')
    with pytest.raises(MaterialError, match="reference"):
        optprops.load_filters(optroot / "filter" / "filters.csv")


# ---------------------------------------------------------------------------
# common.py spec parsers (shared FreeCAD-property / CLI grammar)
# ---------------------------------------------------------------------------
def test_facemap_all_and_perface():
    fm = common.parse_facemap_spec("MgF2")
    assert fm == {common.FACEMAP_ALL: "MgF2"}
    fm = common.parse_facemap_spec("Face3=MgF2;Face5=pbs", body="B",
                                   feature="Pad")
    assert fm == {"B.Pad.Face3": "MgF2", "B.Pad.Face5": "pbs"}
    # trace stage sees fully-qualified ids without context
    fm = common.parse_facemap_spec("B.Pad.Face3=MgF2")
    assert fm == {"B.Pad.Face3": "MgF2"}


def test_facemap_value_with_equals_kept_whole():
    # roughness value contains ':lcorr=' — must survive as one value
    fm = common.parse_facemap_spec("Face1=200:lcorr=5", body="B",
                                   feature="Pad")
    assert fm == {"B.Pad.Face1": "200:lcorr=5"}
    assert common.parse_rough_value(fm["B.Pad.Face1"]) == {
        "sigma_nm": 200.0, "lcorr_um": 5.0}


def test_facemap_rejects_mixed_and_bare_without_context():
    with pytest.raises(ValueError):
        common.parse_facemap_spec("MgF2;Face3=x", body="B", feature="P")
    with pytest.raises(ValueError):
        common.parse_facemap_spec("Face3=MgF2")   # no context


def test_polarization_specs():
    assert common.parse_polarization_spec("unpolarized") == {
        "kind": "unpolarized"}
    lin = common.parse_polarization_spec("linear:30")
    assert lin == {"kind": "linear", "angle_deg": 30.0}
    circ = common.parse_polarization_spec("Circular:LEFT")
    assert circ == {"kind": "circular", "handedness": "left"}
    ell = common.parse_polarization_spec("elliptical:30:-15")
    assert ell["chi_deg"] == -15.0
    for bad in ("linear", "circular:up", "elliptical:0:80", ""):
        with pytest.raises(ValueError):
            common.parse_polarization_spec(bad)


def test_axis_spec_normalized():
    ax = common.parse_axis_spec("0,0,2")
    assert ax == [0.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        common.parse_axis_spec("0,0,0")
    with pytest.raises(ValueError):
        common.parse_axis_spec("1,2")


def test_grating_registry_value():
    g = common.parse_grating_value("@vbg_1800:orders=0..1")
    assert g["registry"] == "vbg_1800" and g["orders"] == (0, 1)
    g = common.parse_grating_value("600:v:eff=0.1,0.8,0.1,0.0,0.0")
    assert g["lines_per_mm"] == 600.0 and g["registry"] is None
    with pytest.raises(ValueError):
        common.parse_grating_value("@")


def test_validate_model_v2_additions():
    """schema v2 body keys pass validation; malformed ones are rejected."""
    model = {
        "schema_version": 2, "source_fcstd": "x.FCStd", "spreadsheet": {},
        "ambient_material": "air", "validation": {},
        "bodies": [
            {"name": "Src", "label": "Src", "role": "source",
             "source": {"power_mW": 1.0, "lambdac_nm": 633.0,
                        "emit_face": "Src.Pad.Face1", "coherent": True,
                        "polarization": {"kind": "linear",
                                         "angle_deg": 30.0}},
             "faces": []},
            {"name": "Pol", "label": "Pol", "role": "optic",
             "material": "pmma", "polarizer": "lp_test",
             "polarizer_axis": [0.0, 0.0, 1.0],
             "crystal_axis": [1.0, 0.0, 0.0],
             "coating": {"Pol.Pad.Face3": "MgF2"},
             "roughness_faces": {"Pol.Pad.Face1": "200:lcorr=5"},
             "grating": {"Pol.Pad.Face2": "600:v"},
             "faces": [_plane_face("Pol.Pad.Face1")]},
            {"name": "Det", "label": "Det", "role": "detector",
             "detector": {"face": "Det.Pad.Face1"},
             "faces": [_plane_face("Det.Pad.Face1")]},
        ],
    }
    common.validate_model(model)
    # non-unit axis rejected
    model["bodies"][1]["crystal_axis"] = [1.0, 1.0, 0.0]
    with pytest.raises(common.ContractError):
        common.validate_model(model)
    model["bodies"][1]["crystal_axis"] = [1.0, 0.0, 0.0]
    # grating applied to every face rejected
    model["bodies"][1]["grating"] = {common.FACEMAP_ALL: "600:v"}
    with pytest.raises(common.ContractError):
        common.validate_model(model)
    model["bodies"][1]["grating"] = None
    # bad polarization kind rejected
    model["bodies"][0]["source"]["polarization"] = {"kind": "diagonal"}
    with pytest.raises(common.ContractError):
        common.validate_model(model)


def test_validate_model_asphere_surface():
    face = {
        "id": "L.Pad.Face1",
        "surface": {"type": "asphere", "vertex": [0, 0, 0],
                    "axis": [1, 0, 0], "R": -0.025, "k": -0.6,
                    "coeffs": [1.2e-6], "r_max": 0.012},
        "orientation_outward": True, "area_m2": 1e-4, "fingerprint": {},
        "mesh_stl": "", "trim_polylines_xyz": [[[0, 0, 0], [0, 1, 0],
                                                [0, 0, 1]]],
    }
    model = {
        "schema_version": 2, "source_fcstd": "x.FCStd", "spreadsheet": {},
        "ambient_material": "air", "validation": {},
        "bodies": [
            {"name": "S", "label": "S", "role": "source",
             "source": {"power_mW": 1.0, "lambdac_nm": 633.0,
                        "emit_face": "S.Pad.Face1", "coherent": True},
             "faces": []},
            {"name": "L", "label": "L", "role": "optic", "material": "bk7",
             "faces": [face]},
            {"name": "D", "label": "D", "role": "detector",
             "detector": {"face": "D.Pad.Face1"},
             "faces": [_plane_face("D.Pad.Face1")]},
        ],
    }
    common.validate_model(model)
    face["surface"]["R"] = 0.0     # signed but nonzero
    with pytest.raises(common.ContractError):
        common.validate_model(model)
    face["surface"]["R"] = -0.025
    face["surface"]["r_max"] = -1.0
    with pytest.raises(common.ContractError):
        common.validate_model(model)


def _plane_face(fid):
    return {
        "id": fid,
        "surface": {"type": "plane", "origin": [0, 0, 0],
                    "normal": [1, 0, 0]},
        "orientation_outward": True, "area_m2": 1e-4, "fingerprint": {},
        "mesh_stl": "",
        "trim_polylines_xyz": [[[0, 0, 0], [0, 1, 0], [0, 0, 1]]],
    }
