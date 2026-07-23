# =============================================================================
# test_prescription.py -- the prescription-primary data model (engine3 Sec 3,
# P5). Pure-python coverage of the loader, the builder emission and the
# .MieWB round-trip; plus a FreeCAD-gated end-to-end cross-check (extractor
# verify + emit-from-prescription, and the deliberate-mismatch hard error).
#
#   "$MIEWB_OPTICS_PYTHON" -m pytest raytracer/tests/test_prescription.py -q
#   MIEWB_RUN_FREECAD=1 ... (adds the extractor cross-check)
# =============================================================================
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import common  # noqa: E402
import primitivelib as pl  # noqa: E402  (metadata + pure emission; no FreeCAD)
import miewb_tool  # noqa: E402
from raytracer import prescription as pr  # noqa: E402

RUN_FREECAD = os.environ.get("MIEWB_RUN_FREECAD") == "1"
freecad_only = pytest.mark.skipif(
    not RUN_FREECAD or not common.FREECAD_APPIMAGE,
    reason="set MIEWB_RUN_FREECAD=1 to run the FreeCAD extractor cross-check")
FREECAD_APPIMAGE = common.FREECAD_APPIMAGE
PROBE = Path(__file__).resolve().parent / "_prescription_fc_probe.py"

COVERED = sorted(pl.prescription_kinds())


def _defaults(kind):
    return {a: s["default"] for a, s in pl.PRIMITIVES[kind]["params"].items()}


# ---------------------------------------------------------------------------
# 1. Loader: schema round-trip + validation
# ---------------------------------------------------------------------------
def test_loader_roundtrip_stable():
    entry = pl.build_prescription_entry("lens_pcx", _defaults("lens_pcx"))
    doc = pr.new_document({"Lens1": entry})
    text = pr.dumps(doc)
    assert pr.loads(text) == doc
    # stable/byte-identical re-dump
    assert pr.dumps(pr.loads(text)) == text


def test_loader_rejects_bad_schema_version():
    doc = pr.new_document({"L": pl.build_prescription_entry(
        "lens_ball", _defaults("lens_ball"))})
    doc["schema_version"] = 2
    with pytest.raises(pr.PrescriptionError):
        pr.validate(doc)


def test_loader_rejects_bad_surface():
    doc = pr.new_document({"L": pl.build_prescription_entry(
        "lens_pcx", _defaults("lens_pcx"))})
    doc["elements"]["L"]["surfaces"][0]["radius"] = -1.0
    with pytest.raises(pr.PrescriptionError):
        pr.validate(doc)


def test_loader_rejects_mesh_surface():
    doc = pr.new_document({"L": pl.build_prescription_entry(
        "lens_ball", _defaults("lens_ball"))})
    doc["elements"]["L"]["surfaces"][0] = {"role": "x", "type": "mesh"}
    with pytest.raises(pr.PrescriptionError):
        pr.validate(doc)


def test_save_load_file(tmp_path):
    doc = pr.new_document({"L": pl.build_prescription_entry(
        "lens_dcx", _defaults("lens_dcx"))})
    p = tmp_path / "x.prescription.json"
    pr.save(p, doc)
    assert pr.load(p) == doc


def test_sidecar_path():
    assert pr.sidecar_path("/a/b/lens_pcx.FCStd").name \
        == "lens_pcx.prescription.json"


# ---------------------------------------------------------------------------
# 2. Builder emission: every covered kind produces a valid entry whose
#    surfaces validate against the shared model.json surface schema.
# ---------------------------------------------------------------------------
def test_every_covered_kind_emits_valid_entry():
    for kind in COVERED:
        entry = pl.build_prescription_entry(kind, _defaults(kind))
        assert entry is not None, kind
        assert entry["kind"] == kind
        assert entry["surfaces"], kind
        # validates through the loader (which reuses common._check_surface_params)
        pr.new_document({kind: entry})
        # every surface geometry key is contract-valid SI
        for s in entry["surfaces"]:
            common._check_surface_params(s["type"], s, "%s/%s" % (kind, s))


def test_uncovered_kind_returns_none():
    assert pl.build_prescription_entry("prism", {"foo": 1.0}) is None
    assert pl.build_prescription_entry("mirror_flat", {}) is None


def test_pcx_sphere_matches_geometry():
    # front vertex at local 0, signed R=25 mm -> centre at (0.025,0,0), r=0.025
    e = pl.build_prescription_entry("lens_pcx", _defaults("lens_pcx"))
    front = next(s for s in e["surfaces"] if s["role"] == "front")
    assert front["type"] == "sphere"
    assert front["center"] == pytest.approx([0.025, 0.0, 0.0])
    assert front["radius"] == pytest.approx(0.025)
    edge = next(s for s in e["surfaces"] if s["role"] == "edge")
    assert edge["type"] == "cylinder"
    assert edge["radius"] == pytest.approx(0.010)   # aperture 20 mm / 2


def test_dcx_back_sphere_center():
    # back vertex at ct=6 mm, signed R2=-40 -> centre at (6-40) mm = -0.034 m
    e = pl.build_prescription_entry("lens_dcx", _defaults("lens_dcx"))
    back = next(s for s in e["surfaces"] if s["role"] == "back")
    assert back["center"] == pytest.approx([-0.034, 0.0, 0.0])
    assert back["radius"] == pytest.approx(0.040)


def test_asphere_coeff_si_conversion():
    # A4 = 6.586562e-06 mm^-3 -> SI m^-3 is x 1000^3 = x 1e9
    e = pl.build_prescription_entry("lens_asphere", _defaults("lens_asphere"))
    front = next(s for s in e["surfaces"] if s["type"] == "asphere")
    assert front["R"] == pytest.approx(0.0206033)
    assert front["k"] == pytest.approx(-1.0)
    assert front["coeffs"][0] == pytest.approx(6.586562e-06 * 1e9)


def test_mirror_parabolic_declares_positive_R():
    # builder declares R = 2*rfl (positive), k = -1
    e = pl.build_prescription_entry("mirror_parabolic",
                                    _defaults("mirror_parabolic"))
    asp = next(s for s in e["surfaces"] if s["type"] == "asphere")
    assert asp["R"] == pytest.approx(0.1)     # 2 * 50 mm
    assert asp["k"] == pytest.approx(-1.0)
    assert asp["coeffs"] == []


# ---------------------------------------------------------------------------
# 3. .MieWB round-trip (miewb_tool pack/unpack)
# ---------------------------------------------------------------------------
def _tiny_fcstd(path):
    # a minimal but valid zip -- miewb_tool stores model.FCStd verbatim and
    # never parses it at pack time, so any bytes suffice for the round-trip.
    import zipfile
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Document.xml", "<Document/>")


def test_miewb_roundtrips_prescription(tmp_path):
    fcstd = tmp_path / "m.FCStd"
    _tiny_fcstd(fcstd)
    doc = pr.new_document({"Lens1": pl.build_prescription_entry(
        "lens_pcx", _defaults("lens_pcx"))})
    out = tmp_path / "m.MieWB"
    miewb_tool.pack_miewb(fcstd, out,
                          optprops_dir=common.OPTPROPS_DIR,
                          prescription=doc)
    import zipfile
    with zipfile.ZipFile(out) as zf:
        assert "prescription.json" in zf.namelist()
        assert miewb_tool.sniff(out) == "MieWB"
    dest = tmp_path / "unp"
    miewb_tool.unpack(out, dest)
    assert pr.load(dest / "prescription.json") == doc


def test_miewb_without_prescription_omits_member(tmp_path):
    fcstd = tmp_path / "m.FCStd"
    _tiny_fcstd(fcstd)
    out = tmp_path / "m.MieWB"
    miewb_tool.pack_miewb(fcstd, out, optprops_dir=common.OPTPROPS_DIR)
    import zipfile
    with zipfile.ZipFile(out) as zf:
        assert "prescription.json" not in zf.namelist()   # backward compat


# ---------------------------------------------------------------------------
# 4. FreeCAD-gated extractor cross-check (build -> extract -> verify + emit;
#    the deliberate-mismatch hard error).
# ---------------------------------------------------------------------------
@freecad_only
def test_extractor_crosscheck_and_drift_error():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "result.json"
        proc = subprocess.run(
            [FREECAD_APPIMAGE, "-c", str(PROBE), "--", "--out", str(out)],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=600)
        assert out.exists(), "probe produced no result:\n%s" % proc.stdout
        res = json.loads(out.read_text())

    base, presc = res["base"], res["presc"]
    # same faces, same types
    assert set(base) == set(presc)
    for fid in base:
        assert base[fid]["type"] == presc[fid]["type"]

    # the front cap is a sphere emitted from the prescription; it agrees with
    # the native-OCC extraction to floating-point precision (the placement in
    # the probe is a real translate + 13 deg rotation).
    front = next(f for f, s in presc.items() if s["type"] == "sphere")
    b, p = base[front], presc[front]
    for i in range(3):
        assert abs(b["center"][i] - p["center"][i]) < 1e-12
    assert abs(b["radius"] - p["radius"]) < 1e-12

    # deliberate +5 um radius mismatch -> hard error naming the drift
    assert res["drift_raised"] is True
    assert "drift" in res["drift_msg"].lower() \
        or "prescription" in res["drift_msg"].lower()
