# =============================================================================
# test_save_fields_detectors.py — --save-fields-detectors LABEL[,LABEL...]
# (design-usability round): restricts --save-fields' complex Ex/Ey
# field-map writes to a named subset of detectors instead of every
# detector. Unknown label is a hard error (never a silent no-op).
#
# Two independent (coherent source, detector) pairs offset in y so they
# never physically interact — lets one scene exercise two detectors with
# genuinely different --save-fields-detectors outcomes.
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_save_fields_detectors.py -q
# =============================================================================
import copy
import json
import sys
from pathlib import Path

import h5py
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "raytracer" / "tests"))

import common                                              # noqa: E402
import cli_specs                                            # noqa: E402
import run_trace                                            # noqa: E402
import scenehelpers as sh                                  # noqa: E402


def _shift_y(body, dy):
    """Deep-copy `body` with every face's geometry translated by dy along
    y — lets two source/detector pairs coexist in one scene without
    physically interacting (scenehelpers' bodies are all built on the
    x-axis at y=0)."""
    body = copy.deepcopy(body)
    for face in body["faces"]:
        face["surface"]["origin"][1] += dy
        for poly in face["trim_polylines_xyz"]:
            for pt in poly:
                pt[1] += dy
    return body


def _two_pair_model():
    pol = {"kind": "linear", "angle_deg": 0.0}
    srcA = sh.source_body(name="SrcA", coherent=True, power_mW=1.0,
                          half=0.004, polarization=pol, beam_waist_mm=0.05)
    detA = sh.detector_body(name="DetA", x=0.03, half=0.02)
    srcB = _shift_y(sh.source_body(name="SrcB", coherent=True, power_mW=1.0,
                                   half=0.004, polarization=pol,
                                   beam_waist_mm=0.05), 1.0)
    detB = _shift_y(sh.detector_body(name="DetB", x=0.03, half=0.02), 1.0)
    return sh.make_model([srcA, detA, srcB, detB])


def _write_model(tmp_path):
    model = _two_pair_model()
    common.validate_model(model)
    mj = tmp_path / "model.json"
    common.write_json(mj, model)
    return mj


# ---------------------------------------------------------------------------
# Unit tests: resolve_save_fields_detectors (no trace, fast)
# ---------------------------------------------------------------------------
class _Args:
    def __init__(self, spec):
        self.save_fields_detectors = spec


def test_resolve_none_when_flag_absent():
    got = run_trace.resolve_save_fields_detectors(
        _Args(None), ["DetA.Pad.Face1", "DetB.Pad.Face1"])
    assert got is None


def test_resolve_parses_comma_separated_and_strips_whitespace():
    got = run_trace.resolve_save_fields_detectors(
        _Args(" DetA.Pad.Face1 , DetB.Pad.Face1"),
        ["DetA.Pad.Face1", "DetB.Pad.Face1", "DetC.Pad.Face1"])
    assert got == {"DetA.Pad.Face1", "DetB.Pad.Face1"}


def test_resolve_unknown_label_hard_errors_naming_available():
    with pytest.raises(SystemExit) as excinfo:
        run_trace.resolve_save_fields_detectors(
            _Args("Bogus"), ["DetA.Pad.Face1", "DetB.Pad.Face1"])
    msg = str(excinfo.value)
    assert "Bogus" in msg
    assert "DetA.Pad.Face1" in msg and "DetB.Pad.Face1" in msg


# ---------------------------------------------------------------------------
# End-to-end: only the named detector's .h5 gets a fields/ group
# ---------------------------------------------------------------------------
def test_save_fields_detectors_subset_e2e(tmp_path):
    mj = _write_model(tmp_path)
    case = tmp_path / "case"
    rc = run_trace.main([
        "--model-json", str(mj), "--case-dir", str(case),
        "--rays", "20000", "--nlambda", "1", "--resolution", "24",
        "--seed0", "42", "--no-gather-gate",
        "--save-fields", "--save-fields-detectors", "DetA.Pad.Face1",
    ])
    assert rc == 0

    with h5py.File(case / "detectors" / "DetA_Pad_Face1.h5") as h:
        assert "fields" in h, "named detector should have a fields/ group"
        assert len(h["fields"].keys()) > 0

    with h5py.File(case / "detectors" / "DetB_Pad_Face1.h5") as h:
        assert "fields" not in h, \
            "un-named detector must NOT get a fields/ group"

    cj = json.loads((case / "case.json").read_text())
    assert cj["options"]["save_fields_detectors"] == "DetA.Pad.Face1"
    # forced off the C engine when a subset is requested (see run_trace.py)
    assert cj["engine"] == "python"


def test_save_fields_bare_still_saves_every_detector(tmp_path):
    """Regression guard: --save-fields' pre-existing (no subset) behavior
    is unchanged by the new flag's addition."""
    mj = _write_model(tmp_path)
    case = tmp_path / "case"
    rc = run_trace.main([
        "--model-json", str(mj), "--case-dir", str(case),
        "--rays", "20000", "--nlambda", "1", "--resolution", "24",
        "--seed0", "42", "--no-gather-gate", "--save-fields",
    ])
    assert rc == 0
    for label in ("DetA_Pad_Face1", "DetB_Pad_Face1"):
        with h5py.File(case / "detectors" / (label + ".h5")) as h:
            assert "fields" in h, label


def test_save_fields_detectors_unknown_label_hard_errors_e2e(tmp_path):
    mj = _write_model(tmp_path)
    case = tmp_path / "case"
    with pytest.raises(SystemExit) as excinfo:
        run_trace.main([
            "--model-json", str(mj), "--case-dir", str(case),
            "--rays", "2000", "--nlambda", "1", "--resolution", "16",
            "--save-fields", "--save-fields-detectors", "NoSuchDetector",
        ])
    msg = str(excinfo.value)
    assert "NoSuchDetector" in msg
    assert "DetA.Pad.Face1" in msg and "DetB.Pad.Face1" in msg
    # fails BEFORE any trace/gather work — no detector output written
    assert not (case / "detectors").exists()


def test_cli_specs_trace_parser_accepts_flag():
    args = cli_specs.build_parser("trace").parse_args([
        "--model-json", "m.json", "--case-dir", "c",
        "--save-fields", "--save-fields-detectors", "DetA,DetB"])
    assert args.save_fields_detectors == "DetA,DetB"
    assert args.save_fields is True
