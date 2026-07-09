"""Real-FreeCAD integration tests for the three train-authoring worker ops
(set_property with a `group` param, create_sheet, set_cell) added to
scripts/fcserver/fcops.py.

Run with:  MIEWB_RUN_FREECAD=1 QT_QPA_PLATFORM=offscreen env/bin/python \
               -m pytest mieworkbench/tests/test_fcserver_train_ops.py -q

Same gated-integration pattern as test_fcserver_integration.py: a real
FreeCAD AppImage worker under FcClient, skipped unless MIEWB_RUN_FREECAD=1.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.fcclient import FcClient, FcError  # noqa: E402

pytestmark = pytest.mark.freecad

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
LENS_DCX = os.path.join(REPO, "basemodels", "lens_dcx.FCStd")
PRIM_PCX = os.path.join(REPO, "primitives", "lens_pcx.FCStd")


@pytest.fixture(scope="module")
def fc():
    with FcClient() as client:
        yield client


def _bodies_by_label(structure):
    return {b["label"]: b for b in structure["bodies"]}


def _sheets_by_label(structure):
    return {s["label"]: s for s in structure["sheets"]}


# ---------------------------------------------------------------------------
# set_property group handling
# ---------------------------------------------------------------------------

def test_set_property_group_create_update_migrate(fc, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("prop_group")
    st = fc.open_document(LENS_DCX)
    doc = st["doc"]

    # -- create in a non-default group -------------------------------------
    r = fc.request("set_property", {"doc": doc, "body": "Lens",
                                    "name": "train_tag", "value": "v1",
                                    "group": "MieTrain"})
    st2 = fc.request("get_structure", {"doc": doc})
    lens = _bodies_by_label(st2)["Lens"]
    prop = lens["properties"]["train_tag"]
    assert prop["group"] == "MieTrain"
    assert prop["value"] == "v1"

    # -- set again, SAME group: value updates, group unchanged --------------
    fc.request("set_property", {"doc": doc, "body": "Lens",
                                "name": "train_tag", "value": "v2",
                                "group": "MieTrain"})
    st3 = fc.request("get_structure", {"doc": doc})
    lens3 = _bodies_by_label(st3)["Lens"]
    prop3 = lens3["properties"]["train_tag"]
    assert prop3["group"] == "MieTrain"
    assert prop3["value"] == "v2"

    # -- set again, DIFFERENT group: migrates ---------------------------------
    fc.request("set_property", {"doc": doc, "body": "Lens",
                                "name": "train_tag", "value": "v3",
                                "group": "Base"})
    st4 = fc.request("get_structure", {"doc": doc})
    lens4 = _bodies_by_label(st4)["Lens"]
    prop4 = lens4["properties"]["train_tag"]
    assert prop4["group"] == "Base"
    assert prop4["value"] == "v3"

    fc.request("remove_property", {"doc": doc, "body": "Lens",
                                   "name": "train_tag"})
    fc.close(doc)


def test_set_property_default_group_is_base(fc, tmp_path_factory):
    """Omitting `group` preserves the pre-existing "Base" default."""
    st = fc.open_document(LENS_DCX)
    doc = st["doc"]
    fc.request("set_property", {"doc": doc, "body": "Lens",
                                "name": "notes", "value": "hi"})
    st2 = fc.request("get_structure", {"doc": doc})
    lens = _bodies_by_label(st2)["Lens"]
    assert lens["properties"]["notes"]["group"] == "Base"
    fc.request("remove_property", {"doc": doc, "body": "Lens", "name": "notes"})
    fc.close(doc)


# ---------------------------------------------------------------------------
# create_sheet
# ---------------------------------------------------------------------------

def test_create_sheet_idempotent(fc, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sheet")
    doc_path = str(tmp / "scene.FCStd")
    st = fc.request("new_document", {"path": doc_path})
    doc = st["doc"]

    r1 = fc.request("create_sheet", {"doc": doc, "label": "miewb_vars"})
    assert r1["sheet"]["label"] == "miewb_vars"
    st1 = fc.request("get_structure", {"doc": doc})
    assert "miewb_vars" in _sheets_by_label(st1)
    n_sheets_before = len(st1["sheets"])
    internal_name = r1["sheet"]["name"]

    # second call: idempotent, no duplicate sheet created
    r2 = fc.request("create_sheet", {"doc": doc, "label": "miewb_vars"})
    assert r2["sheet"]["name"] == internal_name
    st2 = fc.request("get_structure", {"doc": doc})
    assert len(st2["sheets"]) == n_sheets_before

    fc.close(doc)


# ---------------------------------------------------------------------------
# set_cell
# ---------------------------------------------------------------------------

def test_set_cell_alias_and_expression(fc, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cell")
    doc_path = str(tmp / "scene.FCStd")
    st = fc.request("new_document", {"path": doc_path})
    doc = st["doc"]
    fc.request("create_sheet", {"doc": doc, "label": "miewb_vars"})

    fc.request("set_cell", {"doc": doc, "sheet": "miewb_vars",
                            "cell": "B1", "raw": "=10", "alias": "gap"})
    st1 = fc.request("get_structure", {"doc": doc})
    sheet1 = _sheets_by_label(st1)["miewb_vars"]
    assert sheet1["aliases"]["gap"]["value"] == pytest.approx(10.0)

    # expression referencing the first alias
    fc.request("set_cell", {"doc": doc, "sheet": "miewb_vars",
                            "cell": "B2", "raw": "=gap*2", "alias": "arm"})
    st2 = fc.request("get_structure", {"doc": doc})
    sheet2 = _sheets_by_label(st2)["miewb_vars"]
    assert sheet2["aliases"]["arm"]["value"] == pytest.approx(20.0)

    fc.close(doc)


def test_set_cell_alias_validation_rejects_bad_names(fc, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cell_bad")
    doc_path = str(tmp / "scene.FCStd")
    st = fc.request("new_document", {"path": doc_path})
    doc = st["doc"]
    fc.request("create_sheet", {"doc": doc, "label": "miewb_vars"})

    # looks like a cell address
    with pytest.raises(FcError) as exc1:
        fc.request("set_cell", {"doc": doc, "sheet": "miewb_vars",
                                "cell": "B3", "raw": "=1", "alias": "R1"})
    assert "cell address" in str(exc1.value)

    # not a valid identifier (leading digit)
    with pytest.raises(FcError) as exc2:
        fc.request("set_cell", {"doc": doc, "sheet": "miewb_vars",
                                "cell": "B4", "raw": "=1", "alias": "2bad"})
    assert "identifier" in str(exc2.value)

    fc.close(doc)


def test_set_cell_clear_removes_content_and_alias(fc, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cell_clear")
    doc_path = str(tmp / "scene.FCStd")
    st = fc.request("new_document", {"path": doc_path})
    doc = st["doc"]
    fc.request("create_sheet", {"doc": doc, "label": "miewb_vars"})
    fc.request("set_cell", {"doc": doc, "sheet": "miewb_vars",
                            "cell": "B1", "raw": "=10", "alias": "gap"})
    st1 = fc.request("get_structure", {"doc": doc})
    assert "gap" in _sheets_by_label(st1)["miewb_vars"]["aliases"]

    fc.request("set_cell", {"doc": doc, "sheet": "miewb_vars",
                            "cell": "B1", "raw": ""})
    st2 = fc.request("get_structure", {"doc": doc})
    assert "gap" not in _sheets_by_label(st2)["miewb_vars"]["aliases"]

    fc.close(doc)


# ---------------------------------------------------------------------------
# save + reopen round-trip
# ---------------------------------------------------------------------------

def test_save_reopen_preserves_group_and_aliases(fc, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("roundtrip")
    doc_path = str(tmp / "scene.FCStd")
    st = fc.request("new_document", {"path": doc_path})
    doc = st["doc"]

    fc.request("import_primitive", {"doc": doc, "path": PRIM_PCX,
                                    "label": "L1"})
    fc.request("set_property", {"doc": doc, "body": "L1",
                                "name": "train_tag", "value": "roundtrip",
                                "group": "MieTrain"})
    fc.request("create_sheet", {"doc": doc, "label": "miewb_vars"})
    fc.request("set_cell", {"doc": doc, "sheet": "miewb_vars",
                            "cell": "B1", "raw": "=42", "alias": "gap"})

    copy_path = str(tmp / "scene_copy.FCStd")
    fc.request("save_copy", {"doc": doc, "path": copy_path})
    fc.close(doc)

    st2 = fc.open_document(copy_path)
    doc2 = st2["doc"]
    lens = _bodies_by_label(st2)["L1"]
    prop = lens["properties"]["train_tag"]
    assert prop["group"] == "MieTrain"
    assert prop["value"] == "roundtrip"

    sheet = _sheets_by_label(st2)["miewb_vars"]
    assert sheet["aliases"]["gap"]["value"] == pytest.approx(42.0)

    fc.close(doc2)
