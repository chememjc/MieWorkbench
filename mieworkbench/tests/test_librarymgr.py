"""LibraryManager unit tests. NEVER touches the real opticalproperties/ or
primitives/ trees: the system side is always a tmp copy of the real
library (promote_to_system tests) or the real tree opened read-only
(ensure_project_item / self-contained tests only ever WRITE under the
project side, a tmp_path)."""
import csv
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

from mieworkbench.core.librarymgr import (LibraryError,           # noqa: E402
                                          LibraryManager,
                                          used_names_from_structure)
from mieworkbench.core.proplib import LibraryWriteError            # noqa: E402
from raytracer.optprops import load_optical_properties             # noqa: E402

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "opticalproperties"))
PRIMITIVES_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "primitives"))


def _manager(tmp_path, project=True):
    return LibraryManager(REPO_ROOT, PRIMITIVES_ROOT,
                          project_root=(tmp_path / "project") if project
                          else None)


def _system_copy_manager(tmp_path):
    sys_copy = tmp_path / "system_opticalproperties"
    shutil.copytree(REPO_ROOT, sys_copy)
    return LibraryManager(sys_copy, PRIMITIVES_ROOT,
                          project_root=tmp_path / "project")


# ---------------------------------------------------------------------------
# ensure_project_item
# ---------------------------------------------------------------------------
def test_ensure_project_item_copies_material_and_nk_table(tmp_path):
    mgr = _manager(tmp_path)
    written = mgr.ensure_project_item("materials", "aluminum")

    proj_registry = mgr.project_lib.registry_path("materials")
    assert proj_registry.exists()
    assert proj_registry in written

    proj_nk = mgr.project_lib.root / "nk" / "aluminum.mienk"
    assert proj_nk.exists()
    assert proj_nk in written

    rows = mgr.project_lib.registry_rows("materials")
    assert len(rows) == 1
    assert rows[0]["name"] == "aluminum"
    assert rows[0]["reference"].strip()


def test_ensure_project_item_is_idempotent(tmp_path):
    mgr = _manager(tmp_path)
    mgr.ensure_project_item("materials", "aluminum")
    written_again = mgr.ensure_project_item("materials", "aluminum")
    assert written_again == []
    assert len(mgr.project_lib.registry_rows("materials")) == 1


def test_ensure_project_item_unknown_name_raises(tmp_path):
    mgr = _manager(tmp_path)
    with pytest.raises(LibraryError):
        mgr.ensure_project_item("materials", "not_a_real_material")


def test_ensure_project_item_requires_project_root():
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    with pytest.raises(LibraryError):
        mgr.ensure_project_item("materials", "aluminum")


# ---------------------------------------------------------------------------
# used_names_from_structure
# ---------------------------------------------------------------------------
def test_used_names_from_structure_facemap_and_registry_grating():
    structure = {"bodies": [
        {"name": "Body", "properties": {
            "material": {"value": "bk7"},
            "coating": {"value": "Face3=MgF2;Face5=pbs_visible_45"},
        }},
        {"name": "Body001", "properties": {
            "material": {"value": "aluminum"},
            "polarizer": {"value": "ideal_linear"},
            "filter": {"value": "bp_550_40"},
            "grating": {"value": "Face1=@vbg_1800:orders=0..1"},
        }},
        {"name": "Body002", "properties": {
            "coating": {"value": "Al_mirror_bare"},   # all-faces form
        }},
    ]}
    used = used_names_from_structure(structure)
    assert used["materials"] == {"bk7", "aluminum"}
    assert used["coatings"] == {"MgF2", "pbs_visible_45", "Al_mirror_bare"}
    assert used["polarizers"] == {"ideal_linear"}
    assert used["filters"] == {"bp_550_40"}
    assert used["gratings"] == {"vbg_1800"}


def test_used_names_from_structure_empty_structure():
    assert used_names_from_structure({"bodies": []}) == {
        "materials": set(), "coatings": set(), "polarizers": set(),
        "filters": set(), "gratings": set()}
    assert used_names_from_structure(None)["materials"] == set()


# ---------------------------------------------------------------------------
# ensure_project_library_selfcontained -- the acid test
# ---------------------------------------------------------------------------
def test_selfcontained_project_library_loads_with_real_loader(tmp_path):
    mgr = _manager(tmp_path)
    # 'MgF2' is a single-layer TMM coating ("mgf2:qw@550" in the real
    # registry) -- its layer material 'mgf2' must be pulled in even though
    # it never appears directly in `used`.
    used = {
        "materials": {"bk7"},
        "coatings": {"MgF2", "pbs_visible_45"},
        "polarizers": {"ideal_linear"},
        "filters": set(),
        "gratings": {"vbg_1800"},
    }
    written = mgr.ensure_project_library_selfcontained(used)
    assert written   # something was actually copied

    proj_props_dir = mgr.project_lib.root
    props = load_optical_properties(proj_props_dir)   # must not raise

    assert set(props.matdb) >= {"bk7", "mgf2"}
    assert set(props.coatings) == {"MgF2", "pbs_visible_45"}
    assert set(props.polarizers) == {"ideal_linear"}
    assert set(props.gratings) == {"vbg_1800"}
    assert props.filters == {}


def test_selfcontained_pulls_in_uniaxial_o_and_e_materials(tmp_path):
    mgr = _manager(tmp_path)
    used = {"materials": set(), "coatings": set(), "polarizers": set(),
           "filters": set(), "gratings": set(), "uniaxial": {"quartz"}}
    mgr.ensure_project_library_selfcontained(used)

    props = load_optical_properties(mgr.project_lib.root)
    assert set(props.matdb) >= {"quartz_o", "quartz_e"}
    assert "quartz" in props.uniaxial
    assert props.matdb.is_birefringent("quartz")


def test_selfcontained_creates_empty_but_valid_library_when_nothing_used(
        tmp_path):
    mgr = _manager(tmp_path)
    written = mgr.ensure_project_library_selfcontained({})
    assert written == []
    # materials.miemat / coatings.miecoat must exist (header-only) since
    # the loader hard-requires both, even when unused.
    assert mgr.project_lib.registry_path("materials").exists()
    assert mgr.project_lib.registry_path("coatings").exists()
    props = load_optical_properties(mgr.project_lib.root)
    assert len(props.matdb) == 0
    assert props.coatings == {}


def test_selfcontained_unknown_name_raises_before_writing_anything(tmp_path):
    mgr = _manager(tmp_path)
    with pytest.raises(LibraryError):
        mgr.ensure_project_library_selfcontained(
            {"coatings": {"not_a_real_coating"}})
    assert not mgr.project_lib.registry_path("coatings").exists()


def test_selfcontained_rolls_back_when_final_validation_fails(tmp_path):
    mgr = _system_copy_manager(tmp_path)
    # Break the system copy: delete a polarizer's table file. Copying the
    # registry ROW doesn't check the file exists (ensure_project_item only
    # copies what it finds), so the project polarizers.miepol gets created,
    # but the resulting project library can't load -- exercising a real
    # rollback of files this call itself created.
    row = next(r for r in mgr.system_lib.registry_rows("polarizers")
              if r["name"] == "ideal_linear")
    table_path = mgr.system_lib.table_path("polarizers", row["table_csv"])
    table_path.unlink()

    # a pre-existing, unrelated file from an earlier call that must survive
    mgr.ensure_project_item("materials", "gold")
    marker = mgr.project_lib.root / "nk" / "gold.mienk"
    assert marker.exists()
    before = marker.read_bytes()

    with pytest.raises(LibraryWriteError):
        mgr.ensure_project_library_selfcontained(
            {"polarizers": {"ideal_linear"}})

    # newly-created-by-this-call file is rolled back away entirely
    assert not mgr.project_lib.registry_path("polarizers").exists()
    # unrelated pre-existing file is untouched
    assert marker.exists()
    assert marker.read_bytes() == before


# ---------------------------------------------------------------------------
# promote_to_system -- always against a tmp COPY of the system library
# ---------------------------------------------------------------------------
def test_promote_to_system_happy_path_new_item(tmp_path):
    mgr = _system_copy_manager(tmp_path)
    mgr.ensure_project_item("materials", "aluminum")

    # remove aluminum from the system copy so promote adds it back fresh
    rows = [r for r in mgr.system_lib.registry_rows("materials")
           if r["name"] != "aluminum"]
    fieldnames = mgr.system_lib.registry_fieldnames("materials")
    path = mgr.system_lib.registry_path("materials")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (mgr.system_lib.root / "nk" / "aluminum.mienk").unlink()
    mgr.system_lib.reload()

    written = mgr.promote_to_system("materials", "aluminum")
    assert not isinstance(written, dict)
    assert any(p.name == "aluminum.mienk" for p in written)
    ok, _ = mgr.system_lib.validate()
    assert ok
    assert "aluminum" in mgr.system_lib.material_names()


def test_promote_to_system_conflict_without_force(tmp_path):
    mgr = _system_copy_manager(tmp_path)
    mgr.ensure_project_item("materials", "aluminum")

    rows = mgr.project_lib.registry_rows("materials")
    for r in rows:
        if r["name"] == "aluminum":
            r["notes"] = r["notes"] + " (edited in project)"
    fieldnames = mgr.system_lib.registry_fieldnames("materials")
    with open(mgr.project_lib.registry_path("materials"), "w",
             newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = mgr.promote_to_system("materials", "aluminum")
    assert isinstance(result, dict)
    assert result["system_row"]["name"] == "aluminum"
    assert result["project_row"]["notes"] != result["system_row"]["notes"]
    # nothing was written: the system copy is untouched
    assert "(edited in project)" not in \
        mgr.system_lib.registry_path("materials").read_text()


def test_promote_to_system_conflict_force_overwrites(tmp_path):
    mgr = _system_copy_manager(tmp_path)
    mgr.ensure_project_item("materials", "aluminum")
    rows = mgr.project_lib.registry_rows("materials")
    for r in rows:
        if r["name"] == "aluminum":
            r["notes"] = r["notes"] + " (edited in project)"
    fieldnames = mgr.system_lib.registry_fieldnames("materials")
    with open(mgr.project_lib.registry_path("materials"), "w",
             newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    written = mgr.promote_to_system("materials", "aluminum", force=True)
    assert not isinstance(written, dict)
    ok, _ = mgr.system_lib.validate()
    assert ok
    assert "(edited in project)" in \
        mgr.system_lib.registry_path("materials").read_text()


def test_promote_to_system_no_such_project_row_raises(tmp_path):
    mgr = _system_copy_manager(tmp_path)
    with pytest.raises(LibraryError):
        mgr.promote_to_system("materials", "aluminum")


# ---------------------------------------------------------------------------
# primitives_list
# ---------------------------------------------------------------------------
def test_primitives_list_from_real_tree():
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    items = mgr.primitives_list()
    assert len(items) >= 20
    kinds = {i["kind"] for i in items}
    assert "lens_pcx" in kinds
    assert "pbs_cube" in kinds
    lens = next(i for i in items if i["kind"] == "lens_pcx")
    assert lens["category"] == "Lenses"
    assert "R_front" in lens["params"]
    assert lens["params"]["R_front"]["default"] == 25.0


def test_primitives_list_falls_back_for_user_dropped_fcstd(tmp_path):
    prims_dir = tmp_path / "primitives"
    shutil.copytree(PRIMITIVES_ROOT, prims_dir)
    # a user drops a raw .FCStd with no .meta.json sidecar and no entry in
    # primitivelib.PRIMITIVES
    (prims_dir / "my_custom_widget.FCStd").write_bytes(b"not a real fcstd")

    mgr = LibraryManager(REPO_ROOT, prims_dir)
    items = mgr.primitives_list()
    custom = next(i for i in items if i["kind"] == "my_custom_widget")
    assert custom["category"] == "User"
    assert custom["label"] == "my_custom_widget"
    assert custom["params"] == {}
