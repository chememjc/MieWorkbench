"""Pure (no FreeCAD, no Qt) tests for mieworkbench.core.variables — the
GUI-side view over the miewb_vars sweep-variables sheet — plus the
shared common.sweep_combos combination generator and run_pipeline's
variant_output_names(), which must always agree with permute_model.py's
naming for both sweep modes.

Run: QT_QPA_PLATFORM=offscreen env/bin/python -m pytest \
         mieworkbench/tests/test_variables.py -q
"""

import itertools
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

from mieworkbench.core import variables as V  # noqa: E402

import common          # noqa: E402  (stdlib-only shared contract hub)
import run_pipeline    # noqa: E402  (system-python3 orchestrator script)


# ---------------------------------------------------------------------------
# validate_name
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "gap", "abc12", "_priv", "focal_length", "x", "gap2"
])
def test_validate_name_accepts(name):
    assert V.validate_name(name) is None


@pytest.mark.parametrize("name", [
    "R1", "A2", "ab12",       # cell-address-shaped
    "gap__min", "gap__on",    # meta-suffix collisions
    "has__double",            # contains reserved "__"
    "1abc",                   # doesn't start with letter/underscore
    "my var",                 # invalid characters
    "",                       # empty
    None,                     # not a string at all
])
def test_validate_name_rejects(name):
    assert V.validate_name(name) is not None


# ---------------------------------------------------------------------------
# parse_sheet round-trip
# ---------------------------------------------------------------------------
def _cell(cell, raw, value):
    return {"cell": cell, "raw": raw, "value": value, "unit": None}


def synthetic_echo():
    """A miewb_vars sheet echo with two variables:
    gap (row 1, full sweep meta) and half = gap/2 + 3 (row 2, no meta ->
    parse_sheet defaults min/max to the value and n to 0, enabled True)."""
    return {
        "name": "Spreadsheet", "label": "miewb_vars",
        "aliases": {
            "gap": _cell("B1", "=18", 18.0),
            "gap__min": _cell("C1", "=5", 5.0),
            "gap__max": _cell("D1", "=30", 30.0),
            "gap__n": _cell("E1", "=3", 3.0),
            "gap__on": _cell("F1", "=1", 1.0),
            "half": _cell("B2", "=gap/2 + 3", 12.0),
        },
    }


def test_parse_sheet_round_trip():
    rows = V.parse_sheet(synthetic_echo())
    assert set(rows) == {"gap", "half"}

    gap = rows["gap"]
    assert gap.value_raw == "18"
    assert gap.value == 18.0
    assert (gap.vmin, gap.vmax, gap.nstep) == (5.0, 30.0, 3)
    assert gap.enabled is True
    assert gap.row == 1

    half = rows["half"]
    assert half.value_raw == "gap/2 + 3"
    assert half.value == 12.0
    # no meta cells -> min/max default to the value, n=0, enabled=True
    assert (half.vmin, half.vmax, half.nstep) == (12.0, 12.0, 0)
    assert half.enabled is True
    assert half.row == 2


def test_parse_sheet_disabled_row():
    echo = synthetic_echo()
    echo["aliases"]["gap__on"] = _cell("F1", "=0", 0.0)
    rows = V.parse_sheet(echo)
    assert rows["gap"].enabled is False


def test_next_free_row():
    assert V.next_free_row(synthetic_echo()) == 3
    assert V.next_free_row({"aliases": {}}) == 1
    assert V.next_free_row(None) == 1


# ---------------------------------------------------------------------------
# cell_plan
# ---------------------------------------------------------------------------
def test_cell_plan_requires_row():
    with pytest.raises(ValueError):
        V.cell_plan("gap", value=18.0)


def test_cell_plan_new_row_all_fields():
    plan = V.cell_plan("gap", row=3, value=18.0, vmin=5.0, vmax=30.0,
                       nstep=3, enabled=True, comment="lens gap")
    by_cell = {p["cell"]: p for p in plan}
    assert by_cell["A3"]["raw"] == "lens gap"
    assert "alias" not in by_cell["A3"]
    assert by_cell["B3"] == {"cell": "B3", "raw": "=18", "alias": "gap"}
    assert by_cell["C3"] == {"cell": "C3", "raw": "=5", "alias": "gap__min"}
    assert by_cell["D3"] == {"cell": "D3", "raw": "=30", "alias": "gap__max"}
    assert by_cell["E3"] == {"cell": "E3", "raw": "=3", "alias": "gap__n"}
    assert by_cell["F3"] == {"cell": "F3", "raw": "=1", "alias": "gap__on"}


def test_cell_plan_partial_update_existing_row():
    # only touch the value; an existing row's meta cells are untouched
    plan = V.cell_plan("gap", row=1, value="gap_old*2")
    assert plan == [{"cell": "B1", "raw": "=gap_old*2", "alias": "gap"}]


def test_cell_plan_disabled():
    plan = V.cell_plan("gap", row=1, enabled=False)
    assert plan == [{"cell": "F1", "raw": "=0", "alias": "gap__on"}]


def test_cell_plan_bad_name_raises():
    with pytest.raises(ValueError):
        V.cell_plan("R1", row=1, value=1.0)


# ---------------------------------------------------------------------------
# sweep_spec / run_count / estimate_sweep
# ---------------------------------------------------------------------------
def three_var_rows():
    return {
        "b_var": V.VarRow("b_var", "1", 1.0, 0.0, 2.0, 2, True, 1),
        "a_var": V.VarRow("a_var", "2", 2.0, 0.0, 4.0, 1, True, 2),
        "z_off": V.VarRow("z_off", "3", 3.0, 0.0, 6.0, 1, False, 3),
    }


def test_sweep_spec_ordering_and_filtering():
    rows = three_var_rows()
    varnames, mins, maxs, ns = V.sweep_spec(rows)
    # sorted by name, "z_off" excluded (disabled)
    assert varnames == ["miewb_vars.a_var", "miewb_vars.b_var"]
    assert mins == [0.0, 0.0]
    assert maxs == [4.0, 2.0]
    assert ns == [1, 2]


def test_sweep_spec_empty():
    assert V.sweep_spec({}) == ([], [], [], [])


def test_run_count_product_vs_zip():
    rows = {
        "a": V.VarRow("a", "0", 0.0, 0.0, 1.0, 1, True, 1),   # 2 values
        "b": V.VarRow("b", "0", 0.0, 0.0, 1.0, 3, True, 2),   # 4 values
    }
    assert V.run_count(rows, "product") == 2 * 4

    rows_equal = {
        "a": V.VarRow("a", "0", 0.0, 0.0, 1.0, 2, True, 1),   # 3 values
        "b": V.VarRow("b", "0", 0.0, 0.0, 2.0, 2, True, 2),   # 3 values
    }
    assert V.run_count(rows_equal, "zip") == 3

    # length-1 broadcast: n=0 -> single value, rides along a longer sweep
    rows_broadcast = {
        "a": V.VarRow("a", "0", 0.0, 0.0, 1.0, 2, True, 1),   # 3 values
        "b": V.VarRow("b", "5", 5.0, 5.0, 5.0, 0, True, 2),   # 1 value
    }
    assert V.run_count(rows_broadcast, "zip") == 3

    with pytest.raises(ValueError):
        V.run_count(rows, "zip")   # 2 vs 4, no broadcast -> mismatch


def test_run_count_no_enabled_vars():
    assert V.run_count({}, "product") == 1


def test_estimate_sweep():
    rows = {"a": V.VarRow("a", "0", 0.0, 0.0, 1.0, 1, True, 1)}   # 2 values
    est = V.estimate_sweep(rows, "product", single_run_estimate_s=45.0)
    assert est["runs"] == 2
    assert est["per_run_s"] == 45.0
    assert est["total_s"] == 90.0
    assert est["text"] == "2 runs x %s = %s" % (
        common.fmt_duration(45.0), common.fmt_duration(90.0))


# ---------------------------------------------------------------------------
# check_cycles
# ---------------------------------------------------------------------------
def test_check_cycles_clean():
    rows = {
        "gap": V.VarRow("gap", "18", 18.0, 18.0, 18.0, 0, True, 1),
        "half": V.VarRow("half", "gap/2 + 3", 12.0, 12.0, 12.0, 0, True, 2),
    }
    assert V.check_cycles(rows) == []


def test_check_cycles_surfaces_a_b_a():
    rows = {
        "a": V.VarRow("a", "b + 1", None, 0.0, 0.0, 0, True, 1),
        "b": V.VarRow("b", "a + 1", None, 0.0, 0.0, 0, True, 2),
    }
    errors = V.check_cycles(rows)
    assert len(errors) == 1
    assert "a -> b -> a" in errors[0] or "b -> a -> b" in errors[0]


# ---------------------------------------------------------------------------
# common.sweep_combos
# ---------------------------------------------------------------------------
def test_sweep_combos_product_matches_itertools():
    lists = [[1, 2, 3], ["x", "y"]]
    assert common.sweep_combos(lists, mode="product") == \
        list(itertools.product(*lists))


def test_sweep_combos_zip_equal_length():
    lists = [[1, 2, 3], [10, 20, 30]]
    assert common.sweep_combos(lists, mode="zip") == \
        [(1, 10), (2, 20), (3, 30)]


def test_sweep_combos_zip_broadcast_length_one():
    lists = [[1, 2, 3], [99]]
    assert common.sweep_combos(lists, mode="zip") == \
        [(1, 99), (2, 99), (3, 99)]


def test_sweep_combos_zip_mismatch_raises():
    with pytest.raises(ValueError):
        common.sweep_combos([[1, 2, 3], [1, 2]], mode="zip")


def test_sweep_combos_unknown_mode_raises():
    with pytest.raises(ValueError):
        common.sweep_combos([[1, 2]], mode="bogus")


def test_sweep_combos_empty_value_lists():
    assert common.sweep_combos([], mode="product") == [()]
    assert common.sweep_combos([], mode="zip") == [()]


# ---------------------------------------------------------------------------
# run_pipeline.variant_output_names <-> permute_model naming agreement
# ---------------------------------------------------------------------------
def _independent_names(stem, varspecs, mode):
    """Reimplementation independent of variant_output_names' internals,
    chaining common.variant_name over common.sweep_combos directly — the
    two must agree by construction since both ultimately call the same
    common.sweep_values/sweep_combos/variant_name."""
    value_lists = [common.sweep_values(vmin, vmax, n)
                   for (_, vmin, vmax, n) in varspecs]
    names = [v[0] for v in varspecs]
    out = []
    for combo in common.sweep_combos(value_lists, mode=mode):
        out_name = stem
        for var, value in zip(names, combo):
            out_name = common.variant_name(out_name, var, value)
        out.append(out_name)
    return out


@pytest.mark.parametrize("mode", ["product", "zip"])
def test_variant_output_names_matches_independent_chain(mode):
    stem = "example"
    varspecs = [("lenspos", -5.0, 5.0, 2), ("sphered", 20.0, 40.0, 2)]
    got = run_pipeline.variant_output_names(stem, varspecs, mode)
    want = _independent_names(stem, varspecs, mode)
    assert got == want
    assert len(got) == (9 if mode == "product" else 3)


def test_variant_output_names_default_mode_is_product():
    stem = "example"
    varspecs = [("lenspos", -5.0, 5.0, 1), ("sphered", 20.0, 40.0, 1)]
    assert run_pipeline.variant_output_names(stem, varspecs) == \
        run_pipeline.variant_output_names(stem, varspecs, "product")


# ---------------------------------------------------------------------------
# miewb_tool.simparams_to_args is generic: {"sweep_mode": "zip"} needs no
# special-casing to become --sweep-mode zip
# ---------------------------------------------------------------------------
def test_simparams_sweep_mode_passthrough():
    import miewb_tool
    args = miewb_tool.simparams_to_args({"sweep_mode": "zip"})
    assert args == ["--sweep-mode", "zip"]
