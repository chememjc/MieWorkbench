"""core/facemaps.py tests -- pure per-face assignment arithmetic, checked
against scripts/common.py's parse_facemap_spec as the oracle throughout.
No Qt anywhere in this file."""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

import common  # noqa: E402  (stdlib-only shared contract hub)
import pytest  # noqa: E402

from mieworkbench.core.facemaps import (  # noqa: E402
    FACEMAP_PROPERTIES, Assignment, assignment_is_invalid,
    assignments_for_body, face_label, filter_assignments, menu_model,
    merge_facemap, remove_faces, sorted_face_ids, value_check_state,
)

F1 = "Body.Pad.Face1"
F2 = "Body.Pad.Face2"
F3 = "Body.Pad.Face3"
F5 = "Body.Pad.Face5"


def _props(**kwargs):
    """{name: raw} -> the Project body-dict property form."""
    return {name: {"value": raw} for name, raw in kwargs.items()}


# ---------------------------------------------------------------------------
# merge_facemap (moved from element_editor; oracle: parse_facemap_spec)
# ---------------------------------------------------------------------------
def test_merge_facemap_adds_a_face_alongside_an_existing_one():
    raw = merge_facemap("Face5=X", "Body", "Pad", [F3, F5], {F3}, "MgF2")
    parsed = common.parse_facemap_spec(raw, body="Body", feature="Pad")
    assert parsed == {F3: "MgF2", F5: "X"}


def test_merge_facemap_assigning_every_face_collapses_to_bare_value():
    raw = merge_facemap(None, "Body", "Pad", [F3, F5], {F3, F5}, "MgF2")
    assert raw == "MgF2"
    parsed = common.parse_facemap_spec(raw, body="Body", feature="Pad")
    assert parsed == {common.FACEMAP_ALL: "MgF2"}


def test_merge_facemap_expands_existing_all_form_before_overriding():
    raw = merge_facemap("MgF2", "Body", "Pad", [F3, F5], {F3}, "SiO2")
    parsed = common.parse_facemap_spec(raw, body="Body", feature="Pad")
    assert parsed == {F3: "SiO2", F5: "MgF2"}
    # not collapsed -- the two faces disagree
    assert common.FACEMAP_ALL not in parsed


def test_merge_facemap_reassigning_all_faces_to_same_value_recollapses():
    raw = merge_facemap("Face3=A;Face5=B", "Body", "Pad", [F3, F5],
                        {F3, F5}, "Z")
    assert raw == "Z"


def test_merge_facemap_single_face_body_is_bare_all_form():
    # selecting the ONLY face of a body is selecting "every face" -- the
    # per-face table's oracle re-parse should agree it's the whole-body
    # shorthand, not a Face1=... entry.
    raw = merge_facemap(None, "Lens", "Revolution",
                        ["Lens.Revolution.Face1"],
                        {"Lens.Revolution.Face1"}, "SiO2")
    assert raw == "SiO2"


# ---------------------------------------------------------------------------
# remove_faces
# ---------------------------------------------------------------------------
def test_remove_faces_drops_one_entry():
    raw = remove_faces("Face3=A;Face5=B", "Body", "Pad", [F3, F5], {F3})
    parsed = common.parse_facemap_spec(raw, body="Body", feature="Pad")
    assert parsed == {F5: "B"}


def test_remove_last_face_returns_none():
    assert remove_faces("Face3=A", "Body", "Pad", [F3, F5], {F3}) is None
    assert remove_faces("A", "Body", "Pad", [F3], {F3}) is None
    assert remove_faces(None, "Body", "Pad", [F3], {F3}) is None


def test_remove_faces_expands_whole_body_form_first():
    raw = remove_faces("MgF2", "Body", "Pad", [F3, F5], {F5})
    parsed = common.parse_facemap_spec(raw, body="Body", feature="Pad")
    assert parsed == {F3: "MgF2"}


def test_remove_faces_recollapse_when_survivors_agree():
    raw = remove_faces("Face1=A;Face2=A;Face3=B", "Body", "Pad",
                       [F1, F2], {F3})
    # F3 named in the string but not on the body anymore; survivors cover
    # every real face with one value -> bare form
    assert raw == "A"


def test_remove_faces_ignores_faces_not_in_the_map():
    raw = remove_faces("Face3=A;Face5=B", "Body", "Pad", [F1, F3, F5], {F1})
    parsed = common.parse_facemap_spec(raw, body="Body", feature="Pad")
    assert parsed == {F3: "A", F5: "B"}


# ---------------------------------------------------------------------------
# assignments_for_body
# ---------------------------------------------------------------------------
def test_assignments_group_by_property_and_value():
    props = _props(coating="Face1=MgF2;Face2=MgF2;Face3=SiO2",
                   roughness="50")
    out = assignments_for_body(props, "Body", "Pad", [F1, F2, F3])
    assert out == [
        Assignment("coating", "MgF2", frozenset({F1, F2}), False),
        Assignment("coating", "SiO2", frozenset({F3}), False),
        Assignment("roughness", "50", frozenset({F1, F2, F3}), True),
    ]


def test_assignments_whole_body_expands_all_faces():
    out = assignments_for_body(_props(coating="MgF2"), "Body", "Pad",
                               [F1, F2])
    (a,) = out
    assert a.whole_body and a.face_ids == frozenset({F1, F2})


def test_assignments_skip_missing_and_empty():
    assert assignments_for_body({}, "Body", "Pad", [F1]) == []
    assert assignments_for_body(_props(coating=""), "Body", "Pad",
                                [F1]) == []


def test_unparseable_raw_becomes_invalid_row():
    out = assignments_for_body(_props(coating="Face1=MgF2;Face1=SiO2"),
                               "Body", "Pad", [F1, F2])
    (a,) = out
    assert assignment_is_invalid(a)
    assert a.prop == "coating" and a.value == "Face1=MgF2;Face1=SiO2"


def test_assignments_follow_facemap_property_order():
    props = _props(surface_override="Face1=asphere:R=10;k=0;r_max=5",
                   coating="MgF2")
    out = assignments_for_body(props, "Body", "Pad", [F1])
    assert [a.prop for a in out] == ["coating", "surface_override"]


# ---------------------------------------------------------------------------
# scatter (ABg/BSDF registry) -- a facemap property alongside coating/
# roughness/diffuser, added for the biaxial/apodization/scatter round
# ---------------------------------------------------------------------------
def test_scatter_is_a_facemap_property():
    assert "scatter" in FACEMAP_PROPERTIES
    # grouped with the other alternative-surface-model properties, ahead
    # of grating/surface_override
    assert FACEMAP_PROPERTIES.index("scatter") \
        > FACEMAP_PROPERTIES.index("diffuser")
    assert FACEMAP_PROPERTIES.index("scatter") \
        < FACEMAP_PROPERTIES.index("grating")


def test_scatter_assignment_grouping_and_merge():
    props = _props(scatter="Face1=polished_bk7_glass;"
                          "Face2=polished_bk7_glass;Face3=diamond_turned")
    out = assignments_for_body(props, "Body", "Pad", [F1, F2, F3])
    assert out == [
        Assignment("scatter", "diamond_turned", frozenset({F3}), False),
        Assignment("scatter", "polished_bk7_glass",
                   frozenset({F1, F2}), False),
    ]
    raw = merge_facemap("Face1=polished_bk7_glass", "Body", "Pad",
                        [F1, F5], {F5}, "polished_bk7_glass")
    assert raw == "polished_bk7_glass"   # collapses like any other facemap


def test_scatter_whole_body_form():
    out = assignments_for_body(_props(scatter="polished_bk7_glass"),
                               "Body", "Pad", [F1, F2])
    (a,) = out
    assert a.prop == "scatter" and a.whole_body
    assert a.face_ids == frozenset({F1, F2})


# ---------------------------------------------------------------------------
# filter_assignments / value_check_state
# ---------------------------------------------------------------------------
def _canned_assignments():
    return [
        Assignment("coating", "MgF2", frozenset({F1, F2}), False),
        Assignment("coating", "SiO2", frozenset({F3}), False),
        Assignment("roughness", "50", frozenset({F1, F2, F3}), True),
    ]


def test_filter_no_selection_returns_everything():
    a = _canned_assignments()
    assert filter_assignments(a, set()) == a
    assert filter_assignments(a, None) == a


def test_filter_keeps_touching_and_whole_body_rows():
    a = _canned_assignments()
    out = filter_assignments(a, {F3})
    assert [x.value for x in out] == ["SiO2", "50"]


def test_value_check_state_against_selection():
    a = _canned_assignments()
    assert value_check_state(a, "coating", "MgF2", {F1, F2}) == "all"
    assert value_check_state(a, "coating", "MgF2", {F1, F3}) == "some"
    assert value_check_state(a, "coating", "MgF2", {F3}) == "none"
    assert value_check_state(a, "roughness", "50", {F1, F2, F3}) == "all"


def test_value_check_state_whole_body_when_no_selection():
    a = _canned_assignments()
    assert value_check_state(a, "roughness", "50", set(),
                             all_face_ids=[F1, F2, F3]) == "all"
    assert value_check_state(a, "coating", "MgF2", set(),
                             all_face_ids=[F1, F2, F3]) == "some"
    assert value_check_state(a, "coating", "Au", set(),
                             all_face_ids=[F1, F2, F3]) == "none"
    # no selection AND no known faces -> nothing to be checked against
    assert value_check_state(a, "coating", "MgF2", set()) == "none"


# ---------------------------------------------------------------------------
# menu_model
# ---------------------------------------------------------------------------
def test_menu_model_lists_current_values_first_then_registry():
    a = _canned_assignments()
    model = menu_model(a, {F1}, {"coating": ["Au", "MgF2"]}, [F1, F2, F3])
    coating = next(m for m in model if m["prop"] == "coating")
    values = [i["value"] for i in coating["items"]]
    # current values (MgF2, SiO2) lead; registry extras follow, deduped
    assert values == ["MgF2", "SiO2", "Au"]
    by_value = {i["value"]: i for i in coating["items"]}
    assert by_value["MgF2"]["checked"] is True      # on all of {F1}
    assert by_value["SiO2"]["checked"] is False
    assert by_value["SiO2"]["partial"] is False
    assert by_value["Au"] == {"value": "Au", "checked": False,
                              "partial": False}


def test_menu_model_partial_state_and_every_property_present():
    a = _canned_assignments()
    model = menu_model(a, {F1, F3}, {}, [F1, F2, F3])
    assert [m["prop"] for m in model] == [
        "coating", "roughness", "diffuser", "scatter", "grating",
        "surface_override"]
    coating = model[0]
    by_value = {i["value"]: i for i in coating["items"]}
    assert by_value["MgF2"]["partial"] is True      # F1 yes, F3 no
    assert by_value["MgF2"]["checked"] is False
    assert all(m["custom"] for m in model)


def test_menu_model_skips_invalid_rows_values():
    a = [Assignment("coating", "Face1=x;Face1=y", frozenset(), False)]
    model = menu_model(a, set(), {}, [F1])
    coating = model[0]
    assert coating["items"] == []


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def test_face_label_and_sort():
    assert face_label("Body.Pad.Face12") == "Face12"
    assert sorted_face_ids([F5, F1, F3]) == [F1, F3, F5]


def test_element_editor_reexports_survive():
    # legacy import surface: panes.element_editor re-exports the pure API
    from mieworkbench.panes.element_editor import (  # noqa: F401
        FACEMAP_PROPERTIES, active_face_index, merge_facemap as m2,
        validate_facemap_value,
    )
    assert m2 is merge_facemap
    assert "diffuser" in FACEMAP_PROPERTIES
    assert validate_facemap_value("Face1=MgF2", "B", "Pad", 3) is None
    with pytest.raises(TypeError):
        merge_facemap()  # keeps its signature (smoke)
