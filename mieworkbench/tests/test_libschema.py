"""libschema.py tests.

(a) is the drift-proofing contract: it loads the LIVE system
opticalproperties/ registries through the real PropLibrary (the same
object prop_editor.py uses) and asserts every column that ships today has
a COLUMN_SCHEMA entry with a non-empty description -- this is what keeps
libschema.py honest as registries gain columns over time. The rest are
plain unit tests of the pure-data lookup/validation helpers."""
import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core import libschema                        # noqa: E402
from mieworkbench.core.proplib import CATEGORY_INFO, PropLibrary  # noqa: E402

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "opticalproperties"))


# ---------------------------------------------------------------------------
# (a) drift-proofing: every live registry column is documented
# ---------------------------------------------------------------------------
def test_every_live_registry_column_is_documented():
    lib = PropLibrary(REPO_ROOT)
    checked_any = False
    for category in CATEGORY_INFO:
        fieldnames = lib.registry_fieldnames(category)
        assert fieldnames, "%s registry has no columns" % category
        schema = libschema.COLUMN_SCHEMA.get(category)
        assert schema is not None, \
            "libschema.COLUMN_SCHEMA has no entry for category %r " \
            "(known to core.proplib.CATEGORY_INFO)" % category
        for col in fieldnames:
            info = schema.get(col)
            assert info is not None, \
                "libschema.COLUMN_SCHEMA[%r] is missing column %r " \
                "(live in %s)" % (category, col, category)
            assert info.description.strip(), \
                "libschema.COLUMN_SCHEMA[%r][%r] has an empty description" \
                % (category, col)
            checked_any = True
    assert checked_any


def test_no_stray_schema_columns_for_documented_categories():
    """A COLUMN_SCHEMA entry naming a column that no longer exists in the
    live registry is itself a drift bug (stale documentation) -- catch it
    the same way as a missing one."""
    lib = PropLibrary(REPO_ROOT)
    for category in CATEGORY_INFO:
        fieldnames = set(lib.registry_fieldnames(category))
        schema_cols = set(libschema.COLUMN_SCHEMA.get(category, {}))
        stray = schema_cols - fieldnames
        assert not stray, \
            "libschema.COLUMN_SCHEMA[%r] documents column(s) %s that do " \
            "not exist in the live registry" % (category, sorted(stray))


# ---------------------------------------------------------------------------
# lookup / status_text / tooltip_text
# ---------------------------------------------------------------------------
def test_lookup_known_column():
    info = libschema.lookup("materials", "density_kg_m3")
    assert info is not None
    assert info.units == "kg/m^3"
    assert "density" in info.description.lower()


def test_lookup_unknown_category_or_column_returns_none():
    assert libschema.lookup("not_a_category", "name") is None
    assert libschema.lookup("materials", "not_a_column") is None


def test_status_text_known_column_has_name_description_units_format():
    text = libschema.status_text("materials", "density_kg_m3")
    assert text.startswith("density_kg_m3")
    assert "kg/m^3" in text
    assert "density" in text.lower()


def test_status_text_unknown_column_is_empty():
    assert libschema.status_text("materials", "not_a_column") == ""


def test_tooltip_text_unknown_column_has_fallback_text():
    text = libschema.tooltip_text("materials", "not_a_column")
    assert "no schema entry" in text.lower()


def test_tooltip_text_known_column_mentions_units_and_format():
    text = libschema.tooltip_text("coatings", "aoi_deg")
    assert "Units: deg" in text
    assert "Format:" in text


def test_table_schema_lookup():
    info = libschema.lookup_table("materials", "n")
    assert info is not None
    info2 = libschema.lookup_table("materials", "not_a_table_col")
    assert info2 is None


# ---------------------------------------------------------------------------
# validate_cell (advisory)
# ---------------------------------------------------------------------------
def test_validate_cell_blank_is_always_ok():
    ok, msg = libschema.validate_cell("materials", "density_kg_m3", "")
    assert ok and msg == ""
    ok, msg = libschema.validate_cell("materials", "density_kg_m3", "   ")
    assert ok and msg == ""


def test_validate_cell_undocumented_column_is_always_ok():
    ok, msg = libschema.validate_cell("materials", "not_a_column", "garbage")
    assert ok and msg == ""


def test_validate_cell_float_gt_rejects_non_numeric():
    ok, msg = libschema.validate_cell("filters", "ref_thickness_mm", "abc")
    assert not ok
    assert "float" in msg


def test_validate_cell_float_gt_rejects_non_positive():
    ok, msg = libschema.validate_cell("filters", "ref_thickness_mm", "-1.0")
    assert not ok
    assert ">" in msg


def test_validate_cell_float_gt_accepts_valid():
    ok, msg = libschema.validate_cell("filters", "ref_thickness_mm", "3.0")
    assert ok and msg == ""


def test_validate_cell_enum_rejects_unknown_value():
    ok, msg = libschema.validate_cell("materials", "class", "bogus_class")
    assert not ok
    assert "gas" in msg


def test_validate_cell_enum_accepts_valid_value():
    ok, msg = libschema.validate_cell("materials", "class", "glass")
    assert ok and msg == ""


def test_table_wavelength_column_has_plausible_range_validator():
    # the shared _WAVELENGTH_NM table-column validator plausibility-gates
    # to [100, 20000] nm (used by lookup_table -- the per-row spectral
    # table columns, not the registry columns validate_cell checks).
    info = libschema.lookup_table("materials", "wavelength_nm")
    lo, hi = info.validator["range"]
    assert lo == 100.0 and hi == 20000.0


def test_validate_cell_ge_rejects_negative():
    ok, msg = libschema.validate_cell("materials", "density_kg_m3", "-1")
    assert not ok
    assert ">=" in msg


def test_validate_cell_int_validator():
    ok, msg = libschema.validate_cell("diffusers", "grit", "120")
    assert ok and msg == ""
    ok, msg = libschema.validate_cell("diffusers", "grit", "-5")
    assert not ok
    ok, msg = libschema.validate_cell("diffusers", "grit", "not_int")
    assert not ok
