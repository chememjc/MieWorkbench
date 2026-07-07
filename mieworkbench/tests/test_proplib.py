"""PropLibrary unit tests. Reads the real system opticalproperties/ root
read-only; the one corruption test works on a tmp copy only -- the real
library under version control is never touched."""
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.proplib import PropLibrary  # noqa: E402

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "opticalproperties"))

EXPECTED_COUNTS = {
    "materials": 24, "coatings": 10, "polarizers": 5, "filters": 3,
    "gratings": 3, "uniaxial": 3,
}


def _system_lib():
    return PropLibrary(REPO_ROOT)


def test_categories_counts():
    lib = _system_lib()
    cats = lib.categories()
    assert set(cats) == set(EXPECTED_COUNTS)
    for name, expected in EXPECTED_COUNTS.items():
        assert len(cats[name]) == expected, name


def test_material_names_matches_categories():
    lib = _system_lib()
    assert lib.material_names() == lib.categories()["materials"]
    assert "bk7" in lib.material_names()
    assert "aluminum" in lib.material_names()


def test_registry_rows_columns():
    lib = _system_lib()
    rows = lib.registry_rows("materials")
    assert len(rows) == EXPECTED_COUNTS["materials"]
    expected_cols = {"name", "class", "model", "p1", "p2", "p3", "p4", "p5",
                     "p6", "nk_file", "density_kg_m3",
                     "transmission_um_min", "transmission_um_max", "notes",
                     "reference"}
    assert expected_cols <= set(rows[0].keys())
    assert all(row["reference"].strip() for row in rows)

    coating_rows = lib.registry_rows("coatings")
    assert len(coating_rows) == EXPECTED_COUNTS["coatings"]
    assert {"name", "layers", "table", "aoi_deg", "reference"} \
        <= set(coating_rows[0].keys())


def test_registry_fieldnames():
    lib = _system_lib()
    assert lib.registry_fieldnames("materials")[0] == "name"
    assert "table_csv" in lib.registry_fieldnames("polarizers")


def test_registry_path_resolves_existing_file():
    lib = _system_lib()
    path = lib.registry_path("materials")
    assert path.exists()
    assert path.name == "materials.miemat"


def test_table_path_and_table_data_increasing_wavelengths():
    lib = _system_lib()
    path = lib.table_path("materials", "aluminum.mienk")
    assert path.exists()

    headers, rows = lib.table_data("materials", "aluminum.mienk")
    assert headers == ["wavelength_nm", "n", "k"]
    assert len(rows) >= 2
    wavelengths = [r[0] for r in rows]
    assert wavelengths == sorted(wavelengths)
    assert wavelengths[0] < wavelengths[-1]


def test_table_data_for_a_coating_table():
    lib = _system_lib()
    row = next(r for r in lib.registry_rows("coatings")
              if r["name"] == "pbs_visible_45")
    headers, rows = lib.table_data("coatings", row["table"])
    assert headers[0] == "wavelength_nm"
    assert {"Rs", "Rp", "Ts", "Tp"} <= set(headers)
    wavelengths = [r[0] for r in rows]
    assert wavelengths == sorted(wavelengths)


def test_validate_ok_on_real_library():
    lib = _system_lib()
    ok, msg = lib.validate()
    assert ok is True
    assert msg == ""


def test_validate_reports_loader_error_on_corrupt_table(tmp_path):
    corrupt_root = tmp_path / "opticalproperties"
    shutil.copytree(REPO_ROOT, corrupt_root)
    lib = PropLibrary(corrupt_root)

    ok, _ = lib.validate()
    assert ok is True   # sanity: the copy starts out valid

    nk_path = corrupt_root / "nk" / "aluminum.mienk"
    with open(nk_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)
    # break strictly-increasing wavelength_nm by swapping two data rows
    rows[0], rows[1] = rows[1], rows[0]
    with open(nk_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok2, msg2 = lib.validate()
    assert ok2 is False
    assert "not strictly increasing" in msg2 or "increasing" in msg2


def test_reload_invalidates_cache(tmp_path):
    lib_root = tmp_path / "opticalproperties"
    shutil.copytree(REPO_ROOT, lib_root)
    lib = PropLibrary(lib_root)
    first = lib.load()
    assert lib.load() is first     # cached

    lib.reload()
    second = lib.load()
    assert second is not first     # reloaded fresh
