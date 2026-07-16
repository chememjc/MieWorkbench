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


INSTRUMENT_HEADER = (
    "name,class,reference,notes,pixel_pitch_um,width_px,height_px,"
    "fill_factor,qe_table,full_well_e,read_noise_e,dark_current_e_per_s,"
    "bit_depth,adc_gain_e_per_dn,integration_time_s_default,"
    "responsivity_table,flat_responsivity_a_w,aperture_mm,"
    "nep_w_per_sqrthz,bandwidth_hz,display_digits,lam_lo_nm,lam_hi_nm,"
    "resolution_fwhm_nm,slit_um,stray_light_floor,detector_qe_table,"
    "analyzer_states,extinction_ratio,retarder_error_deg,opd_sampling_um,"
    "reference_arm_model,shg_crystal,delay_range_fs\n")


@pytest.fixture
def optroot(tmp_path):
    root = tmp_path / "opticalproperties"
    for sub in ("nk", "birefringence", "polarizer/tables", "filter/tables",
                "coating/tables", "grating/tables", "detector/tables",
                "emission/tables", "instrument/tables"):
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
    (root / "detector" / "detectors.csv").write_text(
        'name,table_csv,reference,notes\n'
        'det_test,det_test.csv,"synthetic","Si test QE"\n')
    (root / "detector" / "tables" / "det_test.csv").write_text(
        "wavelength_nm,qe\n"
        "400,0.60\n"
        "700,0.85\n")
    (root / "emission" / "emitters.csv").write_text(
        'name,kind,table_csv,reference,notes\n'
        'led_test,continuous,led_test.csv,"synthetic","test SPD"\n')
    (root / "emission" / "tables" / "led_test.csv").write_text(
        "wavelength_nm,relative_power\n"
        "400,0.0\n"
        "500,10.0\n"
        "600,5.0\n"
        "700,0.0\n")
    (root / "instrument" / "instruments.csv").write_text(
        INSTRUMENT_HEADER
        + "cam_test,camera,synthetic,test cam,5.0,640,480,1.0,cam_test.csv,"
          "20000,3.0,50,12,4.0,0.02,,,,,,,,,,,,,,,,,,,\n"
        + "pm_test,powermeter,synthetic,test pm,,,,,,,,,,,,pm_test.csv,,"
          "9.5,1e-13,10,4,,,,,,,,,,,,,\n"
        + "spec_test,spectrometer,synthetic,test spec,,,,,,,,,,,,,,,,,,"
          "400,900,1.0,25,0.001,spec_test.csv,,,,,,,\n")
    (root / "instrument" / "tables" / "cam_test.csv").write_text(
        "wavelength_nm,qe\n400,0.40\n700,0.60\n")
    (root / "instrument" / "tables" / "pm_test.csv").write_text(
        "wavelength_nm,responsivity_a_w\n400,0.2\n700,0.5\n")
    (root / "instrument" / "tables" / "spec_test.csv").write_text(
        "wavelength_nm,qe\n400,0.3\n900,0.1\n")
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
    assert set(props.detectors) == {"det_test"}
    assert set(props.emission) == {"led_test"}
    assert set(props.instruments) == {"cam_test", "pm_test", "spec_test"}


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


def test_detector_registry(props):
    d = props.detectors["det_test"]
    assert d["lam_um"].tolist() == pytest.approx([0.4, 0.7])   # nm -> um
    assert d["qe"].tolist() == [0.60, 0.85]
    assert d["reference"] == "synthetic"
    assert d["notes"] == "Si test QE"
    # inside-range interpolation is linear
    qe_mid = optprops.interp_hard(0.55, d["lam_um"], d["qe"], "det")
    assert float(qe_mid) == pytest.approx(0.725)


def test_detector_qe_over_unity_rejected(optroot):
    (optroot / "detector" / "tables" / "det_test.csv").write_text(
        "wavelength_nm,qe\n400,0.6\n700,1.2\n")
    with pytest.raises(MaterialError, match="qe must be <= 1"):
        optprops.load_detectors(optroot / "detector" / "detectors.csv")


def test_detector_qe_zero_rejected(optroot):
    (optroot / "detector" / "tables" / "det_test.csv").write_text(
        "wavelength_nm,qe\n400,0.0\n700,0.85\n")
    with pytest.raises(MaterialError, match="qe must be > 0"):
        optprops.load_detectors(optroot / "detector" / "detectors.csv")


def test_detector_missing_reference_rejected(optroot):
    (optroot / "detector" / "detectors.csv").write_text(
        'name,table_csv,reference,notes\n'
        'noref,det_test.csv,,"x"\n')
    with pytest.raises(MaterialError, match="reference"):
        optprops.load_detectors(optroot / "detector" / "detectors.csv")


def test_emission_registry(props):
    e = props.emission["led_test"]
    assert e["kind"] == "continuous"
    assert e["lam_um"].tolist() == pytest.approx([0.4, 0.5, 0.6, 0.7])  # nm->um
    assert e["lam_nm"].tolist() == pytest.approx([400.0, 500.0, 600.0, 700.0])
    assert e["relative_power"].tolist() == [0.0, 10.0, 5.0, 0.0]
    assert e["reference"] == "synthetic"
    assert e["notes"] == "test SPD"


def test_emission_missing_reference_rejected(optroot):
    (optroot / "emission" / "emitters.csv").write_text(
        'name,kind,table_csv,reference,notes\n'
        'noref,continuous,led_test.csv,,"x"\n')
    with pytest.raises(MaterialError, match="reference"):
        optprops.load_emission(optroot / "emission" / "emitters.csv")


def test_emission_negative_power_rejected(optroot):
    (optroot / "emission" / "tables" / "led_test.csv").write_text(
        "wavelength_nm,relative_power\n400,1.0\n500,-2.0\n700,1.0\n")
    with pytest.raises(MaterialError, match="relative_power must be >= 0"):
        optprops.load_emission(optroot / "emission" / "emitters.csv")


def test_emission_zero_integral_rejected(optroot):
    (optroot / "emission" / "tables" / "led_test.csv").write_text(
        "wavelength_nm,relative_power\n400,0.0\n700,0.0\n")
    with pytest.raises(MaterialError, match="integrates to <= 0"):
        optprops.load_emission(optroot / "emission" / "emitters.csv")


def test_emission_unknown_kind_rejected(optroot):
    (optroot / "emission" / "emitters.csv").write_text(
        'name,kind,table_csv,reference,notes\n'
        'bb,blackbody,led_test.csv,"ref","x"\n')
    with pytest.raises(MaterialError, match="needs engine support"):
        optprops.load_emission(optroot / "emission" / "emitters.csv")


def test_emission_too_few_rows_rejected(optroot):
    (optroot / "emission" / "tables" / "led_test.csv").write_text(
        "wavelength_nm,relative_power\n500,1.0\n")
    with pytest.raises(MaterialError, match="fewer than 2 rows"):
        optprops.load_emission(optroot / "emission" / "emitters.csv")


# ---------------------------------------------------------------------------
# instrument/instruments.csv -- virtual instrument layer (P2.5, engine3.md
# §9). See optprops.load_instruments docstring for the full per-class field
# list; these tests exercise the loader's own validation branches, not the
# post_process render_instrument_* pipeline (covered by
# test_detector_instrument.py).
# ---------------------------------------------------------------------------
def test_instrument_registry_camera(props):
    cam = props.instruments["cam_test"]
    assert cam["class"] == "camera"
    assert cam["pixel_pitch_um"] == 5.0
    assert cam["width_px"] == 640 and cam["height_px"] == 480
    assert cam["fill_factor"] == 1.0
    assert cam["lam_um"].tolist() == pytest.approx([0.4, 0.7])
    assert cam["qe"].tolist() == [0.40, 0.60]
    assert cam["full_well_e"] == 20000.0
    assert cam["bit_depth"] == 12
    assert cam["reference"] == "synthetic"


def test_instrument_registry_powermeter(props):
    pm = props.instruments["pm_test"]
    assert pm["class"] == "powermeter"
    assert pm["resp_table"]["lam_um"].tolist() == pytest.approx([0.4, 0.7])
    assert pm["resp_table"]["responsivity_a_w"].tolist() == [0.2, 0.5]
    assert pm["flat_responsivity_a_w"] is None
    assert pm["aperture_mm"] == 9.5
    assert pm["display_digits"] == 4


def test_instrument_registry_spectrometer(props):
    spec = props.instruments["spec_test"]
    assert spec["class"] == "spectrometer"
    assert spec["lam_lo_nm"] == 400.0 and spec["lam_hi_nm"] == 900.0
    assert spec["resolution_fwhm_nm"] == 1.0
    assert spec["stray_light_floor"] == 0.001
    assert spec["lam_um"].tolist() == pytest.approx([0.4, 0.9])


def test_instrument_unknown_class_rejected(optroot):
    (optroot / "instrument" / "instruments.csv").write_text(
        INSTRUMENT_HEADER
        + "bad,telescope,synthetic,x," + "," * 29 + "\n")
    with pytest.raises(MaterialError, match="class"):
        optprops.load_instruments(optroot / "instrument" / "instruments.csv")


def test_instrument_camera_qe_over_unity_rejected(optroot):
    (optroot / "instrument" / "tables" / "cam_test.csv").write_text(
        "wavelength_nm,qe\n400,0.4\n700,1.5\n")
    with pytest.raises(MaterialError, match="qe_table qe"):
        optprops.load_instruments(optroot / "instrument" / "instruments.csv")


def test_instrument_powermeter_needs_exactly_one_responsivity(optroot):
    # both responsivity_table AND flat_responsivity_a_w set -> rejected
    (optroot / "instrument" / "instruments.csv").write_text(
        INSTRUMENT_HEADER
        + ("pm_bad,powermeter,synthetic,x," + "," * 11
           + "pm_test.csv,0.4,9.5,1e-13,10,4," + "," * 12 + "\n"))
    with pytest.raises(MaterialError, match="exactly one"):
        optprops.load_instruments(optroot / "instrument" / "instruments.csv")
    # neither set -> also rejected
    (optroot / "instrument" / "instruments.csv").write_text(
        INSTRUMENT_HEADER
        + ("pm_bad,powermeter,synthetic,x," + "," * 11
           + ",,9.5,1e-13,10,4," + "," * 12 + "\n"))
    with pytest.raises(MaterialError, match="exactly one"):
        optprops.load_instruments(optroot / "instrument" / "instruments.csv")


def test_instrument_spectrometer_bad_range_rejected(optroot):
    (optroot / "instrument" / "instruments.csv").write_text(
        INSTRUMENT_HEADER
        + ("spec_bad,spectrometer,synthetic,x," + "," * 17
           + "900,400,1.0,25,0.001,spec_test.csv," + "," * 7 + "\n"))
    with pytest.raises(MaterialError, match="lam_hi_nm must be > lam_lo_nm"):
        optprops.load_instruments(optroot / "instrument" / "instruments.csv")


def test_instrument_missing_reference_rejected(optroot):
    (optroot / "instrument" / "instruments.csv").write_text(
        "name,class,reference,notes\ncam_test,camera,,x\n")
    with pytest.raises(MaterialError, match="reference"):
        optprops.load_instruments(optroot / "instrument" / "instruments.csv")


def test_instrument_placeholder_class_validates_when_present(optroot):
    # PLACEHOLDER_INSTRUMENT_CLASSES ship with no rows, but the schema is
    # hard-validated the moment a row IS authored -- e.g. polarimeter needs
    # analyzer_states >= 2.
    (optroot / "instrument" / "instruments.csv").write_text(
        INSTRUMENT_HEADER
        + ("pol_test,polarimeter,synthetic,x," + "," * 23
           + "4,1000,0.1,,,,\n"))
    props = optprops.load_instruments(
        optroot / "instrument" / "instruments.csv")
    pol = props["pol_test"]
    assert pol["analyzer_states"] == 4
    assert pol["extinction_ratio"] == 1000.0
    assert pol["retarder_error_deg"] == 0.1


def test_instrument_placeholder_class_analyzer_states_floor(optroot):
    (optroot / "instrument" / "instruments.csv").write_text(
        INSTRUMENT_HEADER
        + ("pol_bad,polarimeter,synthetic,x," + "," * 23
           + "1,1000,0.1,,,,\n"))
    with pytest.raises(MaterialError, match="analyzer_states must be >= 2"):
        optprops.load_instruments(optroot / "instrument" / "instruments.csv")


def test_instrument_classes_constant_includes_placeholders():
    assert set(optprops.PLACEHOLDER_INSTRUMENT_CLASSES) == {
        "polarimeter", "wavefront_sensor", "autocorrelator"}
    assert set(optprops.PLACEHOLDER_INSTRUMENT_CLASSES) < set(
        optprops.INSTRUMENT_CLASSES)


# ---------------------------------------------------------------------------
# shipped opticalproperties/instrument/ -- generic camera/powermeter/
# spectrometer rows (P2.5), datasheet-sourced starter profiles.
# ---------------------------------------------------------------------------
def test_shipped_instrument_generic_rows_load(shipped_props):
    assert set(shipped_props.instruments) == {
        "camera_generic", "powermeter_generic", "spectrometer_generic"}
    cam = shipped_props.instruments["camera_generic"]
    assert cam["class"] == "camera"
    assert cam["width_px"] == 2448 and cam["height_px"] == 2048
    assert 0 < cam["fill_factor"] <= 1.0
    assert np.all((cam["qe"] > 0) & (cam["qe"] <= 1))
    assert "IMX264" in cam["reference"]

    pm = shipped_props.instruments["powermeter_generic"]
    assert pm["class"] == "powermeter"
    assert pm["resp_table"] is not None
    assert pm["flat_responsivity_a_w"] is None
    assert "Thorlabs" in pm["reference"]

    spec = shipped_props.instruments["spectrometer_generic"]
    assert spec["class"] == "spectrometer"
    assert spec["lam_lo_nm"] < spec["lam_hi_nm"]
    assert "USB4000" in spec["reference"]


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
# shipped opticalproperties/ library -- catalog primitives generated by
# scripts/tools/gen_registry_rows.py (bs ratio family, pellicles, reflective
# and absorptive ND, shortpass/notch). Loads the REAL repo tree (no root=
# override), same as run_trace's default.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def shipped_props():
    return optprops.load_optical_properties()


def test_shipped_full_tree_loads(shipped_props):
    # the whole shipped library (materials/coatings/polarizers/filters/
    # gratings/uniaxial) hard-validates clean, including every row this
    # generator added.
    assert len(shipped_props.matdb) > 0
    for name in ("bs_3070_vis_45", "bs_7030_vis_45", "bs_4060_vis_45",
                 "bs_6040_vis_45", "bs_9010_vis_45", "bs_1090_vis_45",
                 "pellicle_4555_45", "pellicle_uncoated_45",
                 "nd_refl_od03", "nd_refl_od06", "nd_refl_od10",
                 "nd_refl_od20", "nd_refl_od30"):
        assert name in shipped_props.coatings
    for name in ["nd_od%02d" % int(round(od * 10)) for od in
                 (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.3, 2.0, 3.0, 4.0)] \
            + ["shortpass_600", "notch_633_25"]:
        assert name in shipped_props.filters


def test_shipped_hamamatsu_s1223_loads(shipped_props):
    d = shipped_props.detectors["hamamatsu_s1223"]
    assert d["lam_um"].tolist() == pytest.approx([0.660, 0.780, 0.830, 0.960])
    assert d["qe"].tolist() == [0.845, 0.826, 0.807, 0.775]
    assert np.all((d["qe"] > 0) & (d["qe"] <= 1))
    assert "Hamamatsu S1223" in d["reference"]


def test_shipped_led_white_2733k_loads(shipped_props):
    e = shipped_props.emission["led_white_2733k"]
    assert e["kind"] == "continuous"
    assert e["lam_nm"].min() == pytest.approx(400.0)
    assert e["lam_nm"].max() == pytest.approx(700.0)
    assert np.all(e["relative_power"] >= 0)
    assert np.trapezoid(e["relative_power"], e["lam_um"]) > 0
    assert "CIE 015:2018" in e["reference"]
    # power-weighted mean lambda matches the primitive's baked lambdac (584.6)
    lam = e["lam_nm"]
    p = e["relative_power"]
    mean_lam = np.trapezoid(lam * p, lam) / np.trapezoid(p, lam)
    assert mean_lam == pytest.approx(584.6, abs=0.1)


def test_nd_od10_filter_thickness_scaling(shipped_props):
    f = shipped_props.filters["nd_od10"]
    T_at_ref = np.exp(-f["alpha_per_m"] * f["ref_thickness_m"])
    assert f["ref_thickness_m"] == pytest.approx(2.0e-3)
    assert np.all(T_at_ref == pytest.approx(0.1, abs=1e-6))
    # doubling the thickness (2mm -> 4mm) squares the transmittance
    T_at_4mm = np.exp(-f["alpha_per_m"] * (2 * f["ref_thickness_m"]))
    assert np.all(T_at_4mm == pytest.approx(0.01, abs=1e-6))


def test_bs_3070_ratio_and_total_match_5050(shipped_props):
    bs3070 = shipped_props.coatings["bs_3070_vis_45"]
    bs5050 = shipped_props.coatings["bs_5050_vis_45"]
    assert bs3070["kind"] == "table" and bs3070["aoi_deg"] == 45.0
    Ravg = (bs3070["Rs"] + bs3070["Rp"]) / 2.0
    Tavg = (bs3070["Ts"] + bs3070["Tp"]) / 2.0
    ratio = Ravg / Tavg
    assert np.all(ratio == pytest.approx(30.0 / 70.0, rel=1e-2))
    total_3070 = Ravg + Tavg
    total_5050 = (bs5050["Rs"] + bs5050["Rp"]) / 2.0 \
        + (bs5050["Ts"] + bs5050["Tp"]) / 2.0
    # same wavelength grid for both tables -> compare row-for-row
    assert bs3070["lam_um"] == pytest.approx(bs5050["lam_um"])
    assert total_3070 == pytest.approx(total_5050, abs=1e-6)


def test_pellicle_and_notch_rows_load(shipped_props):
    pel = shipped_props.coatings["pellicle_4555_45"]
    assert pel["kind"] == "table"
    assert np.all(pel["Rs"] == pytest.approx(0.4455))
    assert np.all(pel["Ts"] == pytest.approx(0.5445))
    unc = shipped_props.coatings["pellicle_uncoated_45"]
    assert np.all(unc["Rs"] < 0.1) and np.all(unc["Ts"] > 0.85)

    notch = shipped_props.filters["notch_633_25"]
    T_center = optprops.interp_hard(0.633, notch["lam_um"],
                                    np.exp(-notch["alpha_per_m"]
                                           * notch["ref_thickness_m"]),
                                    "notch")
    assert float(T_center) == pytest.approx(1e-4, abs=1e-6)
    T_outside = optprops.interp_hard(0.5, notch["lam_um"],
                                     np.exp(-notch["alpha_per_m"]
                                            * notch["ref_thickness_m"]),
                                     "notch")
    assert float(T_outside) == pytest.approx(0.9, abs=1e-3)

    sp = shipped_props.filters["shortpass_600"]
    assert sp["lam_um"].min() >= 0.4


def test_library_expansion_counts(shipped_props):
    # Pins the shipped library's total row counts after
    # scripts/tools/merge_library_data.py merged library_data/ into the
    # live registries (library.md Sec.7 / library_data/README.md).
    # UPDATE THESE NUMBERS when the library grows (new rows merged or a
    # new gen_registry_rows.py-style generator adds more) -- don't just
    # bump them to make a failure go away, verify the new count is right.
    #
    # materials was 168, not the library.md plan's 169: the BAK4 glass row
    # was dropped as a duplicate of N-BAK4 (library_data/README.md), which
    # the plan's total didn't subtract. It grew to 847 when
    # scripts/tools/import_agf.py appended the Schott+Ohara Zemax AGF glass
    # catalogs (679 new rows; see library_data/agf/README.md).
    assert len(shipped_props.matdb) == 847
    # P2: +1 for bs_5050_vis_45_ph (phase-carrying table-coating demo row)
    assert len(shipped_props.coatings) == 39
    assert len(shipped_props.filters) == 56
    assert len(shipped_props.polarizers) == 17
    assert len(shipped_props.gratings) == 8
    assert len(shipped_props.uniaxial) == 13


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
    model["bodies"][0]["source"]["polarization"] = {"kind": "linear",
                                                     "angle_deg": 30.0}
    # optional detector.qe_curve accepts a string ...
    model["bodies"][2]["detector"]["qe_curve"] = "hamamatsu_s1223"
    common.validate_model(model)
    # ... and rejects a non-string
    model["bodies"][2]["detector"]["qe_curve"] = 42
    with pytest.raises(common.ContractError):
        common.validate_model(model)
    model["bodies"][2]["detector"]["qe_curve"] = "hamamatsu_s1223"
    # optional detector.instrument (P2.5 virtual instrument layer) accepts
    # a string ('row' or 'row:mode') ...
    model["bodies"][2]["detector"]["instrument"] = "camera_generic:ideal"
    common.validate_model(model)
    # ... and rejects a non-string
    model["bodies"][2]["detector"]["instrument"] = 42
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
