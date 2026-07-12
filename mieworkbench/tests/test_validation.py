"""Validation-framework tests: canned structures + typo corpus.

Every malformed input must produce a Finding with a message — never an
exception escaping the validator."""

import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

from mieworkbench.core import validation  # noqa: E402
from raytracer.optprops import load_optical_properties  # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
OPTPROPS = load_optical_properties(os.path.join(REPO, "opticalproperties"))


def body(name, props, closed=True):
    return {"name": name, "label": name, "tip": "Pad",
            "solid_closed": closed, "face_count": 6,
            "placement": {"pos_mm": [0, 0, 0], "quat": [0, 0, 0, 1]},
            "properties": {k: {"type": "t", "group": "Base", "value": v}
                           for k, v in props.items()}}


def good_structure():
    return {"bodies": [
        body("Laser", {"power": 5.0, "lambdac": 633.0, "coherent": True}),
        body("Lens", {"material": "bk7"}),
        body("Screen", {"material": "detector"}),
    ]}


def sheet(element_label, aliases):
    """A structure['sheets'] entry for element_label's dim sheet ('dim_X'),
    aliases: {alias: value}."""
    return {"name": "dim_%s" % element_label, "label": "dim_%s" % element_label,
           "aliases": {k: {"cell": "A1", "raw": str(v), "value": v,
                          "unit": "mm"} for k, v in aliases.items()}}


def run(structure, config=None, optprops=OPTPROPS):
    return validation.Validator(structure, optprops, config).validate()


def messages(findings, severity=None):
    return [f.message for f in findings
            if severity is None or f.severity == severity]


def test_good_scene_has_no_errors():
    findings = run(good_structure())
    assert not validation.has_errors(findings)
    # but it still reports the runtime estimate as info
    assert any("estimated runtime" in m for m in messages(findings))


def test_missing_source_and_detector():
    findings = run({"bodies": [body("Lens", {"material": "bk7"})]})
    errs = messages(findings, validation.ERROR)
    assert any("no light source" in m for m in errs)
    assert any("no detector" in m for m in errs)


def test_unknown_material():
    s = good_structure()
    s["bodies"][1] = body("Lens", {"material": "unobtanium"})
    errs = messages(run(s), validation.ERROR)
    assert any("unknown material 'unobtanium'" in m for m in errs)


def test_detector_face_out_of_range_is_an_error():
    s = good_structure()
    # Screen has face_count 6; Face9 doesn't exist
    s["bodies"][2] = body("Screen",
                          {"material": "detector", "detector_face": "Face9"})
    errs = messages(run(s), validation.ERROR)
    assert any("detector_face names Face9" in m for m in errs)


def test_detector_face_valid_index_is_clean():
    s = good_structure()
    s["bodies"][2] = body("Screen",
                          {"material": "detector", "detector_face": "Face3"})
    assert not validation.has_errors(run(s))


def test_detector_face_full_id_valid_is_clean():
    s = good_structure()
    s["bodies"][2] = body(
        "Screen",
        {"material": "detector", "detector_face": "Screen.Pad.Face3"})
    assert not validation.has_errors(run(s))


def test_detector_face_bad_syntax_is_an_error():
    s = good_structure()
    s["bodies"][2] = body("Screen",
                          {"material": "detector", "detector_face": "top"})
    errs = messages(run(s), validation.ERROR)
    assert any("bad detector_face" in m for m in errs)


def test_detector_face_on_non_detector_warns():
    s = good_structure()
    s["bodies"][1] = body("Lens",
                         {"material": "bk7", "detector_face": "Face1"})
    warns = messages(run(s), validation.WARNING)
    assert any("only" in m and "detector" in m for m in warns)


def test_uniaxial_material_is_known():
    s = good_structure()
    s["bodies"][1] = body("Xtal", {"material": "calcite",
                                   "crystal_axis": "0,0,1"})
    assert not validation.has_errors(run(s))


def test_biaxial_material_without_crystal_axis2_is_an_error():
    s = good_structure()
    s["bodies"][1] = body("Xtal", {"material": "ktp",
                                   "crystal_axis": "1,0,0"})
    errs = messages(run(s), validation.ERROR)
    assert any("biaxial" in m and "crystal_axis2" in m for m in errs)


def test_biaxial_material_with_crystal_axis2_is_clean():
    s = good_structure()
    s["bodies"][1] = body("Xtal", {"material": "ktp",
                                   "crystal_axis": "1,0,0",
                                   "crystal_axis2": "0,1,0"})
    assert not validation.has_errors(run(s))


def test_scatter_and_roughness_same_face_clash():
    s = good_structure()
    s["bodies"][1] = body("Lens", {"material": "bk7",
                                   "scatter": "polished_bk7_glass",
                                   "roughness": "50"})
    errs = messages(run(s), validation.ERROR)
    assert any("scatter and roughness" in m for m in errs)


def test_scatter_and_diffuser_same_face_clash():
    s = good_structure()
    s["bodies"][1] = body("Lens", {"material": "bk7",
                                   "scatter": "polished_bk7_glass",
                                   "diffuser": "grit:120"})
    errs = messages(run(s), validation.ERROR)
    assert any("scatter and diffuser" in m for m in errs)


def test_scatter_on_different_faces_from_roughness_is_clean():
    s = good_structure()
    s["bodies"][1] = body("Lens", {
        "material": "bk7",
        "scatter": "Face1=polished_bk7_glass",
        "roughness": "Face2=50"})
    assert not validation.has_errors(run(s))


def test_unknown_scatter_entry():
    s = good_structure()
    s["bodies"][1] = body("Lens", {"material": "bk7",
                                   "scatter": "NoSuchScatter"})
    errs = messages(run(s), validation.ERROR)
    assert any("unknown scatter entry 'NoSuchScatter'" in m for m in errs)


def test_bad_apodization_spec_is_an_error():
    s = good_structure()
    s["bodies"][0]["properties"]["apodization"] = {
        "type": "t", "group": "Base", "value": "gaussian:order=1"}   # no w0
    errs = messages(run(s), validation.ERROR)
    assert any("bad apodization spec" in m for m in errs)


def test_good_apodization_spec_is_clean():
    s = good_structure()
    s["bodies"][0]["properties"]["apodization"] = {
        "type": "t", "group": "Base", "value": "gaussian:w0=2:order=2"}
    assert not validation.has_errors(run(s))


@pytest.mark.parametrize("value", [-1.0, 0.0, "wide"])
def test_bad_beam_waist_is_an_error(value):
    s = good_structure()
    s["bodies"][0]["properties"]["beam_waist"] = {
        "type": "t", "group": "Base", "value": value}
    errs = messages(run(s), validation.ERROR)
    assert any("beam_waist must be a number > 0" in m for m in errs)


@pytest.mark.parametrize("value", [0.5, "blurry"])
def test_bad_m2_is_an_error(value):
    s = good_structure()
    s["bodies"][0]["properties"]["m2"] = {
        "type": "t", "group": "Base", "value": value}
    errs = messages(run(s), validation.ERROR)
    assert any("m2 must be a number >= 1.0" in m for m in errs)


def test_good_beam_waist_and_m2_are_clean():
    s = good_structure()
    s["bodies"][0]["properties"]["beam_waist"] = {
        "type": "t", "group": "Base", "value": 1.5}
    s["bodies"][0]["properties"]["m2"] = {
        "type": "t", "group": "Base", "value": 1.2}
    assert not validation.has_errors(run(s))


def test_pulse_power_and_energy_both_set_is_an_error():
    s = good_structure()
    s["bodies"][0]["properties"]["pulse_energy"] = {
        "type": "t", "group": "Base", "value": 10.0}
    errs = messages(run(s), validation.ERROR)
    assert any("both power" in m and "pulse_energy" in m for m in errs)


def test_pulse_energy_without_rep_rate_is_an_error():
    s = good_structure()
    # pulse_energy-only source: no 'power' property at all
    s["bodies"][0] = body("Laser", {"lambdac": 633.0, "pulse_energy": 10.0})
    errs = messages(run(s), validation.ERROR)
    assert any("pulse_energy needs rep_rate" in m for m in errs)


def test_pulse_energy_and_rep_rate_is_clean():
    s = good_structure()
    s["bodies"][0] = body("Laser", {"lambdac": 633.0, "pulse_energy": 10.0,
                                    "rep_rate": 1000.0})
    assert not validation.has_errors(run(s))


def test_pulse_props_on_non_source_body_warns():
    s = good_structure()
    s["bodies"][1] = body("Lens", {"material": "bk7", "rep_rate": 1000.0})
    warns = messages(run(s), validation.WARNING)
    assert any("only meaningful on source bodies" in m for m in warns)


@pytest.mark.parametrize("prop,value", [
    ("pulse_energy", -1.0), ("pulse_duration", 0.0), ("rep_rate", -5.0)])
def test_bad_pulse_value_is_an_error(prop, value):
    s = good_structure()
    s["bodies"][0]["properties"][prop] = {
        "type": "t", "group": "Base", "value": value}
    errs = messages(run(s), validation.ERROR)
    assert any("%s must be > 0" % prop in m for m in errs)


def test_unknown_coating_and_bad_facemap():
    s = good_structure()
    s["bodies"][1]["properties"]["coating"] = {
        "type": "t", "group": "Base", "value": "NoSuchCoating"}
    errs = messages(run(s), validation.ERROR)
    assert any("unknown coating 'NoSuchCoating'" in m for m in errs)

    s["bodies"][1]["properties"]["coating"]["value"] = "Face3="
    errs = messages(run(s), validation.ERROR)
    assert any("bad coating spec" in m for m in errs)


def test_unknown_nonlinear_entry_is_an_error():
    s = good_structure()
    s["bodies"][1] = body("Lens", {"material": "bk7",
                                   "nonlinear": "NoSuchNlo"})
    errs = messages(run(s), validation.ERROR)
    assert any("unknown nonlinear entry 'NoSuchNlo'" in m for m in errs)


def test_pockels_row_is_clean():
    s = good_structure()
    s["bodies"][1] = body("Xtal", {"material": "linbo3",
                                   "crystal_axis": "0,1,0",
                                   "nonlinear": "linbo3_eo",
                                   "pockels_voltage": 500.0,
                                   "pockels_gap": 1.0})
    assert not validation.has_errors(run(s))


def test_pockels_voltage_without_pockels_row_warns():
    s = good_structure()
    s["bodies"][1] = body("Lens", {"material": "bk7",
                                   "pockels_voltage": 500.0})
    warns = messages(run(s), validation.WARNING)
    assert any("no effect" in m for m in warns)


def test_pockels_voltage_with_chi2_row_warns():
    s = good_structure()
    s["bodies"][1] = body("Xtal", {"material": "bbo",
                                   "crystal_axis": "0,1,0",
                                   "nonlinear": "bbo_shg_800_type1",
                                   "pockels_voltage": 500.0})
    warns = messages(run(s), validation.WARNING)
    assert any("no effect" in m for m in warns)


def test_pockels_voltage_on_source_warns():
    s = good_structure()
    s["bodies"][0]["properties"]["pockels_voltage"] = {
        "type": "t", "group": "Base", "value": 500.0}
    warns = messages(run(s), validation.WARNING)
    assert any("pockels_voltage is only meaningful" in m for m in warns)


def test_saturable_inline_spec_is_clean():
    s = good_structure()
    s["bodies"][1] = body("Sat", {"material": "air",
                                  "saturable": "sat:I_sat=1e3:T0=0.5"})
    assert not validation.has_errors(run(s))


def test_saturable_bad_inline_spec_is_an_error():
    s = good_structure()
    s["bodies"][1] = body("Sat", {"material": "air",
                                  "saturable": "sat:I_sat=1e3"})
    errs = messages(run(s), validation.ERROR)
    assert any("bad saturable spec" in m for m in errs)


def test_saturable_unknown_registry_entry_is_an_error():
    s = good_structure()
    s["bodies"][1] = body("Sat", {"material": "air",
                                  "saturable": "@NoSuchSaturable"})
    errs = messages(run(s), validation.ERROR)
    assert any("unknown saturable absorber registry entry" in m
              for m in errs)


def test_saturable_on_source_warns():
    s = good_structure()
    s["bodies"][0]["properties"]["saturable"] = {
        "type": "t", "group": "Base", "value": "sat:I_sat=1e3:T0=0.5"}
    warns = messages(run(s), validation.WARNING)
    assert any("saturable is only meaningful" in m for m in warns)


def test_kerr_n2_inline_spec_is_clean():
    s = good_structure()
    s["bodies"][1] = body("Kerr", {"material": "air", "kerr_n2": "n2:3e-20"})
    assert not validation.has_errors(run(s))


def test_kerr_n2_unknown_registry_entry_is_an_error():
    s = good_structure()
    s["bodies"][1] = body("Kerr", {"material": "air",
                                   "kerr_n2": "@NoSuchN2"})
    errs = messages(run(s), validation.ERROR)
    assert any("unknown Kerr n2 registry entry" in m for m in errs)


def test_tpa_beta_on_source_warns():
    s = good_structure()
    s["bodies"][0]["properties"]["tpa_beta"] = {
        "type": "t", "group": "Base", "value": 50.0}
    warns = messages(run(s), validation.WARNING)
    assert any("tpa_beta is only meaningful" in m for m in warns)


def test_open_solid_flagged():
    s = good_structure()
    s["bodies"][1] = body("Lens", {"material": "bk7"}, closed=False)
    errs = messages(run(s), validation.ERROR)
    assert any("not a closed solid" in m for m in errs)


TYPO_CORPUS = [
    ("polarization", "linear"),          # missing angle
    ("polarization", "circular:up"),
    ("polarization", "stokes"),
    ("polarizer_axis", "1,2"),
    ("polarizer_axis", "a,b,c"),
    ("crystal_axis", "0,0,0"),
    ("crystal_axis2", "0,0,0"),
    ("mirror", "1.5"),
    ("mirror", "shiny"),
    ("absorbance", "-0.2"),
    ("roughness", "Face3="),
    ("scatter", "Face3="),
    ("apodization", "gaussian:w0=-1"),
    ("apodization", "square:w0=1"),
    ("beam_waist", "wide"),
    ("m2", "blurry"),
    ("surface_override", "Face1=a;Face1=b"),
    ("grating", "Face2=0:v"),            # lines/mm must be > 0
    ("grating", "Face2=@"),
    ("saturable", "sat:I_sat=1e3"),      # missing T0
    ("saturable", "sat:I_sat=1e3:T0=1.5"),
    ("kerr_n2", "n2:0"),                  # must be non-zero
]


@pytest.mark.parametrize("prop,value", TYPO_CORPUS)
def test_typo_corpus_yields_findings_not_crashes(prop, value):
    s = good_structure()
    s["bodies"][1]["properties"][prop] = {
        "type": "t", "group": "Base", "value": value}
    findings = run(s)     # must not raise
    assert validation.has_errors(findings), (prop, value)


def test_filter_out_of_band():
    s = good_structure()
    # bp_550_40's table is narrow; a broadband source far outside it must
    # trip the coverage check
    s["bodies"][0] = body("Laser", {"power": 5.0, "lambdac": 1500.0,
                                    "coherent": True})
    s["bodies"][1]["properties"]["filter"] = {
        "type": "t", "group": "Base", "value": "bp_550_40"}
    errs = messages(run(s), validation.ERROR)
    assert any("filter 'bp_550_40'" in m and "hard-error" in m
               for m in errs)


def test_sampling_gate_warning():
    findings = run(good_structure(),
                   config={"rays": 500, "nlambda": 5,
                           "min_eff_samples": 1000})
    warns = messages(findings, validation.WARNING)
    assert any("below the gather gate" in m for m in warns)


def test_gather_preflight_warns_on_clipped_aperture():
    """A Ø6 mm coherent beam through a Ø0.2 mm pinhole (the UXNOTES_ROUND3
    #18 example) at 'quick' ray budgets must warn ahead of a failed
    trace, naming the ray multiplier needed."""
    s = good_structure()
    s["bodies"][0]["properties"]["diameter"] = {
        "type": "t", "group": "Base", "value": 6.0}
    s["bodies"][1]["properties"]["miewb_primitive"] = {
        "type": "t", "group": "Base", "value": "pinhole"}
    s["sheets"] = [sheet("Lens", {"hole_diameter": 0.2}),
                  sheet("Laser", {"diameter": 6.0})]
    findings = run(s, config={"rays": 1e5, "nlambda": 5})
    warns = messages(findings, validation.WARNING)
    assert any("gather" in m and "Ø0.2" in m for m in warns)
    assert not validation.has_errors(findings)   # WARNING only, never fails


def test_gather_preflight_clean_when_aperture_clears_beam():
    """A generous aperture relative to the beam must not warn."""
    s = good_structure()
    s["bodies"][0]["properties"]["diameter"] = {
        "type": "t", "group": "Base", "value": 1.0}
    s["sheets"] = [sheet("Lens", {"aperture": 25.0}),
                  sheet("Laser", {"diameter": 1.0})]
    findings = run(s, config={"rays": 1e5, "nlambda": 5})
    warns = messages(findings, validation.WARNING)
    assert not any("gather" in m and "aperture" in m for m in warns)


def test_gather_preflight_skips_incoherent_sources():
    """The preflight only concerns the coherent gather; an incoherent
    source clipped the same way must not warn."""
    s = good_structure()
    s["bodies"][0]["properties"]["coherent"] = {
        "type": "t", "group": "Base", "value": False}
    s["bodies"][0]["properties"]["diameter"] = {
        "type": "t", "group": "Base", "value": 6.0}
    s["sheets"] = [sheet("Lens", {"hole_diameter": 0.2}),
                  sheet("Laser", {"diameter": 6.0})]
    findings = run(s, config={"rays": 1e5, "nlambda": 5})
    assert not any(f.check == "gather-preflight" for f in findings)


def test_no_library_degrades_gracefully():
    findings = run(good_structure(), optprops=None)
    warns = messages(findings, validation.WARNING)
    assert any("name checks skipped" in m for m in warns)
    assert not validation.has_errors(findings)


@pytest.mark.freecad
def test_real_example_structure_validates():
    from mieworkbench.core.fcclient import FcClient
    with FcClient() as fc:
        st = fc.open_document(os.path.join(REPO, "example.FCStd"))
        findings = validation.Validator(st, OPTPROPS,
                                        {"rays": 1e5, "nlambda": 5}) \
            .validate()
        fc.close(st["doc"])
    assert not validation.has_errors(findings), \
        [f.message for f in findings if f.severity == validation.ERROR]


def test_deep_checks_clean_scene_has_info_finding():
    """Deep check on a clean scene (mocked) returns INFO Finding."""
    class MockFc:
        def request(self, op, kwargs):
            # Simulate successful check with no problems
            return {"invalid": [], "open_solids": [], "overlaps": [],
                    "face_pairs_checked": 42}
    class MockProject:
        doc = "fake_doc"
        fc = MockFc()
        structure = {"bodies": [
            {"name": "L1", "label": "L1", "tip": "Pad"},
            {"name": "L2", "label": "L2", "tip": "Pad"},
            {"name": "D", "label": "D", "tip": "Pad"},
        ]}
    findings = validation.run_deep_checks(MockProject())
    # Should return a single INFO Finding
    assert len(findings) == 1
    assert findings[0].severity == validation.INFO
    assert "Deep check passed" in findings[0].message
    assert "3 bodies recomputed" in findings[0].message


def test_deep_checks_with_errors():
    """Deep check returns ERRORs and WARNINGs for problems found."""
    class MockFc:
        def request(self, op, kwargs):
            return {
                "invalid": ["BadBody"],
                "open_solids": ["OpenSolid"],
                "overlaps": [{"a": "Part1", "b": "Part2", "volume_mm3": 0.5}]
            }
    class MockProject:
        doc = "fake_doc"
        fc = MockFc()
        structure = {"bodies": [
            {"name": "BadBody", "label": "BadBody"},
            {"name": "OpenSolid", "label": "OpenSolid"},
            {"name": "Part1", "label": "Part1"},
            {"name": "Part2", "label": "Part2"},
        ]}
    findings = validation.run_deep_checks(MockProject())
    # Should return errors and warning, no info
    errors = [f for f in findings if f.severity == validation.ERROR]
    warnings = [f for f in findings if f.severity == validation.WARNING]
    infos = [f for f in findings if f.severity == validation.INFO]
    assert len(errors) == 2
    assert len(warnings) == 1
    assert len(infos) == 0
    assert any("failed to recompute" in f.message for f in errors)
    assert any("not a closed solid" in f.message for f in errors)
    assert any("overlap" in f.message for f in warnings)


# ---------------------------------------------------------------------------
# placement traps (source at origin; rotated unpinned detector)
# ---------------------------------------------------------------------------
def _traps(findings):
    return [f for f in findings if f.check == "placement-traps"]


def test_source_at_origin_warns():
    st = good_structure()
    findings = run(st)
    warns = _traps(findings)
    assert any("origin" in f.message for f in warns)


def test_source_off_origin_is_clean():
    st = good_structure()
    st["bodies"][0]["placement"]["pos_mm"] = [-40.0, 0.0, 0.0]
    assert not any("origin" in f.message for f in _traps(run(st)))


def test_rotated_detector_without_pin_warns():
    import math
    st = good_structure()
    st["bodies"][0]["placement"]["pos_mm"] = [-40.0, 0.0, 0.0]
    q = [0.0, 0.0, math.sin(math.radians(22.5)),
         math.cos(math.radians(22.5))]           # 45 deg about z
    st["bodies"][2]["placement"] = {"pos_mm": [50.0, 50.0, 0.0], "quat": q}
    warns = _traps(run(st))
    assert any("detector_face" in f.message for f in warns)


def test_rotated_detector_with_pin_is_clean():
    import math
    st = good_structure()
    st["bodies"][0]["placement"]["pos_mm"] = [-40.0, 0.0, 0.0]
    q = [0.0, 0.0, math.sin(math.radians(22.5)),
         math.cos(math.radians(22.5))]
    st["bodies"][2]["placement"] = {"pos_mm": [50.0, 50.0, 0.0], "quat": q}
    st["bodies"][2]["properties"]["detector_face"] = {
        "type": "t", "group": "Base", "value": "Face3"}
    assert not any("detector_face pin" in f.message or
                   "EDGE face" in f.message for f in _traps(run(st)))


def test_unrotated_detector_is_clean():
    st = good_structure()
    st["bodies"][0]["placement"]["pos_mm"] = [-40.0, 0.0, 0.0]
    assert _traps(run(st)) == []
