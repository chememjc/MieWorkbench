"""Pure-logic tests for widgets.facepicker's selection-set arithmetic.
No VTK/Qt involved -- the vtkCellPicker wiring itself (FacePicker) needs a
real render window to do anything, so it's only exercised by
@pytest.mark.needs_gl tests elsewhere (skipped offscreen).
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

from mieworkbench.widgets.facepicker import (  # noqa: E402
    pick_to_selection, select_all,
)


def test_plain_click_replaces_selection():
    assert pick_to_selection({"Face1", "Face2"}, "Face3", False) == {"Face3"}


def test_plain_click_on_already_selected_face_keeps_just_that_face():
    assert pick_to_selection({"Face1"}, "Face1", False) == {"Face1"}


def test_ctrl_click_adds_when_absent():
    result = pick_to_selection({"Face1"}, "Face2", True)
    assert result == {"Face1", "Face2"}


def test_ctrl_click_removes_when_present():
    result = pick_to_selection({"Face1", "Face2"}, "Face2", True)
    assert result == {"Face1"}


def test_ctrl_click_from_empty_selection_adds():
    assert pick_to_selection(set(), "Face1", True) == {"Face1"}


def test_plain_miss_clears_selection():
    assert pick_to_selection({"Face1", "Face2"}, None, False) == set()


def test_additive_miss_preserves_selection():
    assert pick_to_selection({"Face1"}, None, True) == {"Face1"}


def test_current_selection_none_treated_as_empty():
    assert pick_to_selection(None, "Face1", False) == {"Face1"}
    assert pick_to_selection(None, "Face1", True) == {"Face1"}


def test_all_faces_membership_is_validated_when_given():
    with pytest.raises(ValueError):
        pick_to_selection(set(), "FaceX", False,
                          all_faces=["Face1", "Face2"])
    # no error when omitted, or when the face IS in all_faces
    assert pick_to_selection(set(), "Face1", False,
                             all_faces=["Face1", "Face2"]) == {"Face1"}


def test_select_all():
    assert select_all(["Face1", "Face2", "Face3"]) == {
        "Face1", "Face2", "Face3"}


def test_select_all_empty():
    assert select_all(None) == set()
    assert select_all([]) == set()
