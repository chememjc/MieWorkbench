"""Real-FreeCAD integration tests (marked 'freecad', env-gated).

Run with:  MIEWB_RUN_FREECAD=1 env/bin/python -m pytest \
               mieworkbench/tests/test_fcserver_integration.py -q

The round-trip oracle: edit a model through the worker, save a copy, run the
REAL extract_geometry on both, and diff the model.json contracts. This pins
the worker's write path to the same semantics the pipeline consumes.
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.fcclient import FcClient  # noqa: E402
from mieworkbench.core.geomcache import GeomCache  # noqa: E402

pytestmark = pytest.mark.freecad

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXAMPLE = os.path.join(REPO, "example.FCStd")
LENS_DCX = os.path.join(REPO, "basemodels", "lens_dcx.FCStd")


def extract_model_json(fcstd_path, out_dir):
    """Run the real extract_geometry.py on one model, return its model.json."""
    appimage = os.environ.get("MIEWB_FREECAD",
                              "/home3/freecad/FreeCAD.AppImage")
    script = os.path.join(REPO, "scripts", "extract_geometry.py")
    subprocess.run(
        [appimage, "-c", script, "--", "--models", fcstd_path,
         "--outdir", str(out_dir)],
        stdin=subprocess.DEVNULL, check=True, capture_output=True, text=True)
    stem = os.path.splitext(os.path.basename(fcstd_path))[0]
    with open(os.path.join(str(out_dir), stem, "model.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def fc():
    with FcClient() as client:
        yield client


def test_noop_save_roundtrip_preserves_contract(fc, tmp_path_factory):
    """Open + save_copy with NO edits must extract to an identical contract
    (modulo source path)."""
    tmp = tmp_path_factory.mktemp("noop")
    st = fc.open_document(LENS_DCX)
    copy_path = str(tmp / "lens_dcx_copy.FCStd")
    fc.request("save_copy", {"doc": st["doc"], "path": copy_path})
    fc.close(st["doc"])

    orig = extract_model_json(LENS_DCX, tmp / "g_orig")
    copy = extract_model_json(copy_path, tmp / "g_copy")
    for m, stem in ((orig, "lens_dcx"), (copy, "lens_dcx_copy")):
        m.pop("source_fcstd", None)
        m.pop("extracted_note", None)
        # warning texts embed the model stem; normalize before comparing
        val = m.get("validation", {})
        val["warnings"] = [w.replace(stem + ":", "<stem>:")
                           for w in val.get("warnings", [])]
    assert orig == copy


def test_spreadsheet_edit_lands_in_extract(fc, tmp_path_factory):
    """Editing 'lensth' 2mm -> 3mm through the worker must change the lens
    geometry that extract_geometry reports (volume grows)."""
    tmp = tmp_path_factory.mktemp("edit")
    st = fc.open_document(EXAMPLE)
    doc = st["doc"]
    r = fc.request("set_spreadsheet", {"doc": doc, "sheet": "dim",
                                       "alias": "lensth", "raw": "=3 mm"})
    assert r["changed_bodies"], "lensth edit must reshape the lens body"
    edited_path = str(tmp / "example_lensth3.FCStd")
    fc.request("save_copy", {"doc": doc, "path": edited_path})
    # restore in-memory state for other tests (copy already on disk)
    fc.request("set_spreadsheet", {"doc": doc, "sheet": "dim",
                                   "alias": "lensth", "raw": "=2 mm"})
    fc.close(doc)

    orig = extract_model_json(EXAMPLE, tmp / "g_orig")
    edit = extract_model_json(edited_path, tmp / "g_edit")

    def lens_volume(model):
        for body in model["bodies"]:
            if body["label"] == "Lens":
                return body["volume_m3"]
        raise AssertionError("no Lens body found")

    assert edit["spreadsheet"]["lensth"]["si"] == pytest.approx(0.003)
    assert lens_volume(edit) > lens_volume(orig)


def test_property_edit_lands_in_extract(fc, tmp_path_factory):
    """Setting a coating tag through the worker must surface in model.json."""
    tmp = tmp_path_factory.mktemp("prop")
    st = fc.open_document(LENS_DCX)
    doc = st["doc"]
    fc.request("set_property", {"doc": doc, "body": "Lens",
                                "name": "coating", "value": "MgF2"})
    edited_path = str(tmp / "lens_dcx_coated.FCStd")
    fc.request("save_copy", {"doc": doc, "path": edited_path})
    fc.request("remove_property", {"doc": doc, "body": "Lens",
                                   "name": "coating"})
    fc.close(doc)

    model = extract_model_json(edited_path, tmp / "g")
    lens = [b for b in model["bodies"] if b["label"] == "Lens"][0]
    # extract normalizes coating specs to a facemap; whole-body form is
    # {"__all__": name}
    assert lens.get("coating") == {"__all__": "MgF2"}


def test_placement_edit_moves_extracted_geometry(fc, tmp_path_factory):
    """A +5mm x placement set through the worker must shift the extracted
    face centroids by 5mm (placement is applied at extraction)."""
    tmp = tmp_path_factory.mktemp("plc")
    st = fc.open_document(LENS_DCX)
    doc = st["doc"]
    fc.request("set_placement", {"doc": doc, "body": "Lens",
                                 "pos_mm": [5.0, 0.0, 0.0],
                                 "quat": [0.0, 0.0, 0.0, 1.0]})
    edited_path = str(tmp / "lens_dcx_moved.FCStd")
    fc.request("save_copy", {"doc": doc, "path": edited_path})
    fc.close(doc)

    orig = extract_model_json(LENS_DCX, tmp / "g_orig")
    moved = extract_model_json(edited_path, tmp / "g_moved")

    def face1_centroid_x(model):
        lens = [b for b in model["bodies"] if b["label"] == "Lens"][0]
        return lens["faces"][0]["fingerprint"]["centroid"][0]

    dx = face1_centroid_x(moved) - face1_centroid_x(orig)
    assert dx == pytest.approx(0.005, abs=1e-9)  # metres


def test_geomcache_full_stack(fc, tmp_path_factory):
    """GeomCache against the real worker: tessellate, hit, selective refresh."""
    tmp = tmp_path_factory.mktemp("cache")
    st = fc.open_document(EXAMPLE)
    doc = st["doc"]
    cache = GeomCache(fc, cache_root=str(tmp))
    r1 = cache.faces_for(doc, EXAMPLE)
    assert sum(len(v["faces"]) for v in r1.values()) >= 20
    for v in r1.values():
        for f in v["faces"]:
            assert os.path.getsize(f["stl"]) > 84
    r2 = cache.faces_for(doc, EXAMPLE)          # pure hit
    assert {k: v["shape_key"] for k, v in r1.items()} == \
           {k: v["shape_key"] for k, v in r2.items()}
    fc.close(doc)
