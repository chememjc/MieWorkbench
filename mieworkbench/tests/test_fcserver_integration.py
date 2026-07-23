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
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

from mieworkbench.core.fcclient import FcClient  # noqa: E402
from mieworkbench.core.geomcache import GeomCache  # noqa: E402
import common  # noqa: E402

pytestmark = pytest.mark.freecad

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXAMPLE = os.path.join(REPO, "example.FCStd")
LENS_DCX = os.path.join(REPO, "basemodels", "lens_dcx.FCStd")


def extract_model_json(fcstd_path, out_dir):
    """Run the real extract_geometry.py on one model, return its model.json."""
    if not common.FREECAD_APPIMAGE:
        pytest.skip("MIEWB_FREECAD not configured")
    appimage = common.FREECAD_APPIMAGE
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


def test_detector_face_property_lands_in_extract(fc, tmp_path_factory):
    """Setting `detector_face` on a detector body through the worker must
    make extract_geometry bake it into the detector dict as the recorded
    face with autodetected=False (in place of the closest-to-origin
    auto-pick)."""
    tmp = tmp_path_factory.mktemp("detface")
    orig = extract_model_json(EXAMPLE, tmp / "g_orig")
    dets = [b for b in orig["bodies"] if b.get("role") == "detector"]
    assert dets, "example.FCStd should have at least one detector body"
    det = dets[0]
    label = det["label"]
    face_ids = [f["id"] for f in det["faces"]]
    auto_id = det["detector"]["face"]
    assert det["detector"].get("autodetected") is True
    # pick a face DIFFERENT from the auto-pick when the body has more than
    # one face, so the override is observably distinct
    target_id = next((fid for fid in face_ids if fid != auto_id), auto_id)
    target_faceN = target_id.rsplit(".", 1)[-1]   # bare 'FaceN'

    st = fc.open_document(EXAMPLE)
    doc = st["doc"]
    fc.request("set_property", {"doc": doc, "body": label,
                                "name": "detector_face",
                                "value": target_faceN})
    edited_path = str(tmp / "example_detface.FCStd")
    fc.request("save_copy", {"doc": doc, "path": edited_path})
    fc.request("remove_property", {"doc": doc, "body": label,
                                   "name": "detector_face"})
    fc.close(doc)

    model = extract_model_json(edited_path, tmp / "g_edit")
    edited = [b for b in model["bodies"] if b["label"] == label][0]
    assert edited["detector"]["autodetected"] is False
    assert edited["detector"]["face"] == target_id


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


PRIM_PCX = os.path.join(REPO, "primitives", "lens_pcx.FCStd")
PRIM_ACHROMAT = os.path.join(REPO, "primitives", "lens_achromat.FCStd")


def _bodies_by_label(structure):
    return {b["label"]: b for b in structure["bodies"]}


def _element_snapshot(fc, doc, group):
    """Comparable snapshot of an element: label -> (placement, contract
    props); sheet aliases of dim_<group>."""
    st = fc.request("get_structure", {"doc": doc})
    bodies = {}
    for b in st["bodies"]:
        if b["properties"].get("miewb_group", {}).get("value") == group \
                or b["label"] == group:
            props = {k: v["value"] for k, v in b["properties"].items()}
            bodies[b["label"]] = (b["placement"], props)
    sheets = {s["label"]: {a: e["raw"] for a, e in s["aliases"].items()}
              for s in st["sheets"] if s["label"] == "dim_%s" % group}
    return {"bodies": bodies, "sheets": sheets}


def test_duplicate_delete_restore_element(fc, tmp_path_factory):
    """The A2 op trio end-to-end on a multi-body primitive (achromat):
    duplicate under a new group, delete-with-stash, restore verbatim."""
    tmp = tmp_path_factory.mktemp("elops")
    doc_path = str(tmp / "scene.FCStd")
    st = fc.request("new_document", {"path": doc_path})
    doc = st["doc"]

    r = fc.request("import_primitive", {"doc": doc, "path": PRIM_ACHROMAT,
                                        "label": "doublet"})
    assert len(r["bodies"]) == 2

    # move the element so restore has a non-identity placement to preserve
    fc.request("set_placement", {"doc": doc, "body": "doublet",
                                 "pos_mm": [1.0, 2.0, 3.0],
                                 "quat": [0.0, 0.0, 0.0, 1.0]})
    # and a user prop that must survive the stash round-trip
    member = r["bodies"][0]["name"]
    fc.request("set_property", {"doc": doc, "body": member,
                                "name": "roughness", "value": "50"})

    # -- duplicate ---------------------------------------------------------
    d = fc.request("duplicate_element", {"doc": doc, "element": "doublet",
                                         "new_label": "doublet2"})
    assert len(d["bodies"]) == 2
    labels = {b["label"] for b in d["bodies"]}
    assert all(l.startswith("doublet2_") for l in labels)
    for b in d["bodies"]:
        assert b["properties"]["miewb_group"]["value"] == "doublet2"
    st2 = fc.request("get_structure", {"doc": doc})
    assert "dim_doublet2" in {s["label"] for s in st2["sheets"]}
    assert len(st2["bodies"]) == 4

    # -- delete with stash --------------------------------------------------
    before = _element_snapshot(fc, doc, "doublet")
    stash = str(tmp / "stash_doublet.FCStd")
    dl = fc.request("delete_element", {"doc": doc, "element": "doublet",
                                       "stash_path": stash})
    assert len(dl["deleted"]) == 2
    assert os.path.isfile(stash)
    st3 = fc.request("get_structure", {"doc": doc})
    assert len(st3["bodies"]) == 2          # only the duplicate remains
    assert "dim_doublet" not in {s["label"] for s in st3["sheets"]}

    # -- restore -------------------------------------------------------------
    rs = fc.request("import_bodies", {"doc": doc, "path": stash})
    assert len(rs["bodies"]) == 2
    after = _element_snapshot(fc, doc, "doublet")
    assert after == before                  # labels/placements/props verbatim
    fc.close(doc)


def test_delete_single_ungrouped_body(fc, tmp_path_factory):
    """delete_element on a plain body (no miewb_group) removes just it."""
    tmp = tmp_path_factory.mktemp("delone")
    st = fc.open_document(LENS_DCX)
    doc = st["doc"]
    n_before = len(st["bodies"])
    dl = fc.request("delete_element", {"doc": doc, "element": "Lens"})
    assert len(dl["deleted"]) == 1
    st2 = fc.request("get_structure", {"doc": doc})
    assert len(st2["bodies"]) == n_before - 1
    fc.close(doc)


def test_duplicate_refuses_existing_label(fc, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dupdup")
    doc_path = str(tmp / "scene.FCStd")
    st = fc.request("new_document", {"path": doc_path})
    doc = st["doc"]
    fc.request("import_primitive", {"doc": doc, "path": PRIM_PCX,
                                    "label": "L1"})
    from mieworkbench.core.fcclient import FcError
    with pytest.raises(FcError):
        fc.request("duplicate_element", {"doc": doc, "element": "L1",
                                         "new_label": "L1"})
    fc.close(doc)
