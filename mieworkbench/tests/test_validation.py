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


def test_uniaxial_material_is_known():
    s = good_structure()
    s["bodies"][1] = body("Xtal", {"material": "calcite",
                                   "crystal_axis": "0,0,1"})
    assert not validation.has_errors(run(s))


def test_unknown_coating_and_bad_facemap():
    s = good_structure()
    s["bodies"][1]["properties"]["coating"] = {
        "type": "t", "group": "Base", "value": "NoSuchCoating"}
    errs = messages(run(s), validation.ERROR)
    assert any("unknown coating 'NoSuchCoating'" in m for m in errs)

    s["bodies"][1]["properties"]["coating"]["value"] = "Face3="
    errs = messages(run(s), validation.ERROR)
    assert any("bad coating spec" in m for m in errs)


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
    ("mirror", "1.5"),
    ("mirror", "shiny"),
    ("absorbance", "-0.2"),
    ("roughness", "Face3="),
    ("surface_override", "Face1=a;Face1=b"),
    ("grating", "Face2=0:v"),            # lines/mm must be > 0
    ("grating", "Face2=@"),
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
