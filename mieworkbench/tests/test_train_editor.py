"""TrainEditorPane tests (offscreen): tree shape, dialog-free edits routed
through the Project train API, expression display vs EditRole, invalid-
expression rejection, fold toggling, mode round-trips, two-way selection
sync with origin echo-suppression, 3D reference re-chaining, fold-mirror
insertion, and problem highlighting. Driven with the scripted TrainFake
worker (no FreeCAD, no dialogs). See docs/UI_TESTING.md."""

import os
import sys

import numpy as np
import pytest
from PySide6.QtCore import Qt

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.selection import SelectionModel          # noqa: E402
from mieworkbench.core.train import TRAIN_GROUP                 # noqa: E402
from mieworkbench.panes.train_editor import (                   # noqa: E402
    TrainEditorPane, COL_ELEMENT, COL_MODE, COL_REF, COL_PORT, COL_DIST,
    COL_DECX, COL_DECY, COL_TILTX, COL_TILTY, COL_TILTZ, COL_FOLD, COL_FLIP,
    ROLE_ELEMENT, ORIGIN, _RED,
)
from mieworkbench.tests.train_test_support import make_scene, pos_of  # noqa: E402


def make_pane(qtbot, project):
    sel = SelectionModel()
    pane = TrainEditorPane(project, sel)
    qtbot.addWidget(pane)
    return pane, sel


def _make_source(project, label):
    """Tag an element's primary body as a source so it sorts first."""
    props = project.body(label)["properties"]
    props["power"] = {"type": "App::PropertyFloat", "group": "Base",
                      "value": 1.0}
    props["lambdac"] = {"type": "App::PropertyFloat", "group": "Base",
                        "value": 550.0}


# ---------------------------------------------------------------------------
# Tree shape
# ---------------------------------------------------------------------------
def test_tree_nesting_and_port_group_rows(qtbot):
    project, _ = make_scene()
    _make_source(project, "SRC")
    # main train SRC -> L1 -> FM, then FM splits: DET on reflect, L2 on out
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("FM", {"ref": "L1", "distance": "20"})
    project.set_chain("DET", {"ref": "FM", "port": "reflect",
                              "distance": "15"})
    project.set_chain("L2", {"ref": "FM", "port": "out", "distance": "5"})
    pane, _sel = make_pane(qtbot, project)

    # single anchored root: the source, and it is the only top-level row
    assert pane.tree.topLevelItemCount() == 1
    src = pane.tree.topLevelItem(0)
    assert src.data(COL_ELEMENT, ROLE_ELEMENT) == "SRC"

    # L1 nested under SRC, FM nested under L1 (single-child: no port row)
    l1 = pane.item_for_element("L1")
    fm = pane.item_for_element("FM")
    assert l1.parent() is src
    assert fm.parent() is l1

    # FM has TWO children on different ports -> two intermediate port rows
    assert fm.childCount() == 2
    port_rows = [fm.child(i) for i in range(2)]
    assert all(r.data(COL_ELEMENT, ROLE_ELEMENT) is None for r in port_rows)
    labels = sorted(r.text(COL_ELEMENT) for r in port_rows)
    assert any("reflect" in x for x in labels)
    assert any("out" in x for x in labels)

    # DET hangs under the reflect port row, L2 under the out port row
    det = pane.item_for_element("DET")
    l2 = pane.item_for_element("L2")
    assert "reflect" in det.parent().text(COL_ELEMENT)
    assert "out" in l2.parent().text(COL_ELEMENT)


# ---------------------------------------------------------------------------
# Editing a distance cell
# ---------------------------------------------------------------------------
def test_commit_distance_moves_body_and_one_undo(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    pane, _sel = make_pane(qtbot, project)
    assert np.allclose(pos_of(project, "L1"), [17, 0, 0])

    assert pane.commit_field("L1", "distance", "30") is True
    # SRC exit x=5, +30 -> entry target 35, entry local -2 -> pos 37
    assert np.allclose(pos_of(project, "L1"), [37, 0, 0])
    # display refreshed
    assert pane.item_for_element("L1").data(COL_DIST, Qt.EditRole) == "30"

    project.undo()
    assert np.allclose(pos_of(project, "L1"), [17, 0, 0])


def test_expression_display_vs_editrole(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "gap*2"})  # gap=25
    pane, _sel = make_pane(qtbot, project)
    item = pane.item_for_element("L1")
    assert item.data(COL_DIST, Qt.DisplayRole) == "gap*2  (= 50.0)"
    assert item.data(COL_DIST, Qt.EditRole) == "gap*2"


def test_invalid_expression_does_not_mutate(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    pane, _sel = make_pane(qtbot, project)
    before = pos_of(project, "L1")

    assert pane.commit_field("L1", "distance", "2*+") is False
    assert "Invalid" in pane.status.text()
    assert np.allclose(pos_of(project, "L1"), before)   # unchanged


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
def _fold_scene():
    project, fake = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("FM", {"ref": "L1", "distance": "20", "fold": True,
                             "folded": True, "tilt_ry": "-45"})
    project.set_chain("DET", {"ref": "FM", "port": "reflect",
                              "distance": "15"})
    return project, fake


def test_fold_toggle_straightens_downstream_and_greys(qtbot):
    project, _ = _fold_scene()
    pane, _sel = make_pane(qtbot, project)
    # folded: DET is on the reflected +y arm
    assert np.allclose(pos_of(project, "DET"), [39, 15, 0], atol=1e-9)
    fm = pane.item_for_element("FM")
    assert fm.checkState(COL_FOLD) == Qt.Checked

    assert pane.toggle_fold("FM", False) is True
    # DET re-collinearized onto +x, same along-beam distance
    assert np.allclose(pos_of(project, "DET"), [54, 0, 0], atol=1e-9)
    # the unfolded fold row renders greyed/italic
    fm = pane.item_for_element("FM")
    assert fm.checkState(COL_FOLD) == Qt.Unchecked
    assert fm.font(COL_ELEMENT).italic()


# ---------------------------------------------------------------------------
# Mode round trip
# ---------------------------------------------------------------------------
def test_mode_anchored_chained_round_trip(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    pane, _sel = make_pane(qtbot, project)
    pos = pos_of(project, "L1")

    assert pane.set_mode("L1", "anchored") is True
    assert not project.train().is_chained("L1")
    assert np.allclose(pos_of(project, "L1"), pos)

    assert pane.set_mode("L1", "chained") is True     # reuses stored ref
    tm = project.train()
    assert tm.is_chained("L1")
    assert tm.records()["L1"]["ref"] == "SRC"
    assert np.allclose(pos_of(project, "L1"), pos)


# ---------------------------------------------------------------------------
# Selection sync (both directions, origin echo-suppression)
# ---------------------------------------------------------------------------
def test_selection_sync_both_ways(qtbot):
    project, _ = make_scene()
    pane, sel = make_pane(qtbot, project)
    events = []
    sel.changed.connect(lambda b, f, o: events.append((b, o)))

    # row -> selection (origin train_editor)
    pane.tree.setCurrentItem(pane.item_for_element("SRC"))
    assert sel.body == "SRC"
    assert events[-1] == ("SRC", ORIGIN)

    # selection from elsewhere -> row highlighted, no echo back
    events.clear()
    sel.select("L1", origin="outliner")
    cur = pane.tree.currentItem()
    assert cur.data(COL_ELEMENT, ROLE_ELEMENT) == "L1"
    # exactly one event (the outliner-originated one); no train_editor echo
    assert events == [("L1", "outliner")]


# ---------------------------------------------------------------------------
# Pick reference in 3D
# ---------------------------------------------------------------------------
def test_pick_reference_rechains_and_refuses_descendant(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "20"})
    pane, _sel = make_pane(qtbot, project)

    # refuse chaining L1 to its own descendant L2
    pane.begin_pick_reference("L1")
    pane.on_reference_picked("L2", None)
    assert project.body("L1")["properties"]["miewb_train_ref"]["value"] \
        == "SRC"
    assert "descendant" in pane.status.text()

    # legitimately re-chain L2 onto SRC
    pane.begin_pick_reference("L2")
    pane.on_reference_picked("SRC", None)
    assert project.train().records()["L2"]["ref"] == "SRC"


def test_begin_pick_emits_request(qtbot):
    project, _ = make_scene()
    pane, _sel = make_pane(qtbot, project)
    seen = []
    pane.pickReferenceRequested.connect(seen.append)
    pane.begin_pick_reference("L1")
    assert seen == ["L1"]


# ---------------------------------------------------------------------------
# Insert fold mirror
# ---------------------------------------------------------------------------
def test_insert_fold_inserts_and_reanchors(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "20"})
    pane, _sel = make_pane(qtbot, project)
    before = pos_of(project, "L2")

    label = pane.insert_fold("L1", 5.0, 90.0, 0.0)
    assert label == "Fold1"
    assert "Fold1" in project.train().element_labels()

    # L2 re-anchored onto the mirror's reflected arm
    props = project.body("L2")["properties"]
    assert props["miewb_train_ref"]["value"] == "Fold1"
    assert props["miewb_train_port"]["value"] == "reflect"
    assert not np.allclose(pos_of(project, "L2"), before)

    # the new fold row carries a checked "folded" box
    fold_item = pane.item_for_element("Fold1")
    assert fold_item is not None
    assert fold_item.checkState(COL_FOLD) == Qt.Checked


# ---------------------------------------------------------------------------
# Edge details
# ---------------------------------------------------------------------------
def test_set_edge_details_writes_fields(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    pane, _sel = make_pane(qtbot, project)
    assert pane.set_edge_details("L1", "zyx", "rot_first", "center") is True
    props = project.body("L1")["properties"]
    assert props["miewb_train_rot_order"]["value"] == "zyx"
    assert props["miewb_train_pos_rot_order"]["value"] == "rot_first"
    assert props["miewb_train_pivot"]["value"] == "center"


# ---------------------------------------------------------------------------
# Validation highlighting
# ---------------------------------------------------------------------------
def test_cycle_renders_rows_red_with_tooltip(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    pane, _sel = make_pane(qtbot, project)
    # forge a cycle behind the API's back (as a hand-edited file could)
    project.body("SRC")["properties"]["miewb_train_mode"] = {
        "type": "App::PropertyString", "group": TRAIN_GROUP,
        "value": "chained"}
    project.body("SRC")["properties"]["miewb_train_ref"] = {
        "type": "App::PropertyString", "group": TRAIN_GROUP, "value": "L1"}
    pane.rebuild()

    src = pane.item_for_element("SRC")
    l1 = pane.item_for_element("L1")
    assert src is not None and l1 is not None
    assert src.foreground(COL_ELEMENT).color() == _RED
    assert l1.foreground(COL_ELEMENT).color() == _RED
    assert "circular" in src.toolTip(COL_ELEMENT)


# ---------------------------------------------------------------------------
# Phase G (round-2 UX refinements): port selector
# ---------------------------------------------------------------------------
def test_available_ports_reflects_ref_geometry(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("FM", {"ref": "L1", "distance": "20", "fold": True,
                             "folded": True, "tilt_ry": "-45"})
    pane, _sel = make_pane(qtbot, project)
    records = project.train().records()

    # a plain lens (no reflect plane, no fold): only the transmissive ports
    assert TrainEditorPane._available_ports(records["L1"]) == \
        ["out", "transmit"]
    # a fold mirror with a reflect plane: transmissive + reflect
    assert TrainEditorPane._available_ports(records["FM"]) == \
        ["out", "transmit", "reflect"]


def test_available_ports_deviate_without_reflect_plane(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    # L1 has no reflect_plane; marking it fold with no plane -> deviate-only
    project.set_chain("L1", {"fold": True, "folded": True})
    records = project.train().records()
    assert TrainEditorPane._available_ports(records["L1"]) == \
        ["out", "transmit", "deviate"]


def test_commit_port_writes_chosen_port(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("FM", {"ref": "L1", "distance": "20", "fold": True,
                             "folded": True, "tilt_ry": "-45"})
    pane, _sel = make_pane(qtbot, project)

    pane.begin_pick_reference("DET")
    pane.on_reference_picked("FM")
    assert pane.commit_field("DET", "distance", "15") is True
    assert pane.commit_port("DET", "reflect") is True
    assert project.train().records()["DET"]["port"] == "reflect"
    # matches the position a raw set_chain(port="reflect") would produce
    # (see test_fold_toggle_straightens_downstream_and_greys's fold scene)
    assert np.allclose(pos_of(project, "DET"), [39, 15, 0], atol=1e-6)


def test_commit_port_rejects_unavailable_port(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "10"})
    pane, _sel = make_pane(qtbot, project)

    # L1 is a plain lens: "reflect" is not one of its available ports
    assert pane.commit_port("L2", "reflect") is False
    assert "L2:" in pane.status.text()
    assert project.train().records()["L2"].get("port") in (None, "")


def test_port_combo_delegate_choices_for_chained_row(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("FM", {"ref": "L1", "distance": "20", "fold": True,
                             "folded": True, "tilt_ry": "-45"})
    project.set_chain("DET", {"ref": "FM", "port": "reflect",
                              "distance": "15"})
    pane, _sel = make_pane(qtbot, project)

    assert pane._port_choices("DET") == ["out", "transmit", "reflect"]
    # anchored / unchained rows offer nothing (delegate stays read-only)
    assert pane._port_choices("SRC") == []


# ---------------------------------------------------------------------------
# Phase G: make-foldable (identity vs state)
# ---------------------------------------------------------------------------
def test_fold_checkbox_checkable_for_any_chained_element(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    pane, _sel = make_pane(qtbot, project)
    item = pane.item_for_element("L1")
    # a plain (non-fold) chained element: checkbox present, unchecked
    assert item.checkState(COL_FOLD) == Qt.Unchecked


def test_mark_fold_sets_identity_and_folded(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("FM", {"ref": "L1", "distance": "20"})  # not a fold yet
    pane, _sel = make_pane(qtbot, project)

    assert pane.mark_fold("FM", True) is True
    rec = project.train().records()["FM"]
    assert rec["fold"] is True
    assert rec["folded"] is True
    item = pane.item_for_element("FM")
    assert item.checkState(COL_FOLD) == Qt.Checked

    # unmark: clears identity, leaves `folded` untouched
    assert pane.mark_fold("FM", False) is True
    rec = project.train().records()["FM"]
    assert rec["fold"] is False


def test_mark_fold_refuses_on_anchored_element(qtbot):
    project, _ = make_scene()
    pane, _sel = make_pane(qtbot, project)
    assert pane.mark_fold("SRC", True) is False
    assert "SRC:" in pane.status.text()


def test_checking_fold_box_on_plain_chained_row_marks_it_foldable(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    pane, _sel = make_pane(qtbot, project)
    item = pane.item_for_element("L1")

    item.setCheckState(COL_FOLD, Qt.Checked)
    # the itemChanged-triggered mutation is deferred a tick (see
    # TrainEditorPane._defer's docstring: rebuilding the tree
    # synchronously from inside the edited item's own setData call
    # segfaults -- Qt's item-view machinery is still unwinding above us)
    qtbot.wait(10)

    rec = project.train().records()["L1"]
    assert rec["fold"] is True
    assert rec["folded"] is True


# ---------------------------------------------------------------------------
# Phase G: flip
# ---------------------------------------------------------------------------
def test_commit_flip_writes_flip_and_checkbox_reflects_it(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    pane, _sel = make_pane(qtbot, project)

    assert pane.commit_flip("L1", True) is True
    assert project.train().records()["L1"]["flip"] is True
    item = pane.item_for_element("L1")
    assert item.checkState(COL_FLIP) == Qt.Checked

    assert pane.commit_flip("L1", False) is True
    assert project.train().records()["L1"]["flip"] is False


def test_flip_checkbox_wiring_via_item_changed(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    pane, _sel = make_pane(qtbot, project)
    item = pane.item_for_element("L1")

    item.setCheckState(COL_FLIP, Qt.Checked)
    qtbot.wait(10)     # deferred mutation -- see TrainEditorPane._defer

    assert project.train().records()["L1"]["flip"] is True


# ---------------------------------------------------------------------------
# Phase G: partial edge details + deviate-port fields
# ---------------------------------------------------------------------------
def test_set_edge_details_partial_update_leaves_others_alone(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    pane, _sel = make_pane(qtbot, project)

    assert pane.set_edge_details("L1", pivot="center") is True
    props = project.body("L1")["properties"]
    assert props["miewb_train_pivot"]["value"] == "center"
    assert "miewb_train_rot_order" not in props
    assert "miewb_train_pos_rot_order" not in props

    assert pane.set_edge_details(
        "L1", fold_deviation="30", fold_azimuth="45") is True
    props = project.body("L1")["properties"]
    assert props["miewb_train_fold_deviation"]["value"] == "30"
    assert props["miewb_train_fold_azimuth"]["value"] == "45"
    assert props["miewb_train_pivot"]["value"] == "center"   # untouched


def test_set_edge_details_no_args_is_a_noop(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    pane, _sel = make_pane(qtbot, project)
    assert pane.set_edge_details("L1") is True
    assert "miewb_train_pivot" not in project.body("L1")["properties"]


# ---------------------------------------------------------------------------
# Phase G: reliable chain-to-selected via selection history
# ---------------------------------------------------------------------------
def test_chain_selected_uses_selection_history_over_the_solve_order_guess(
        qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("FM", {"ref": "L1", "distance": "20", "fold": True,
                             "folded": True, "tilt_ry": "-45"})
    pane, sel = make_pane(qtbot, project)

    # without any history, the solve-order fallback would pick L2 (the
    # last root in solve order), NOT FM -- exactly the old mis-chain trap
    assert pane._last_train_element("DET") == "L2"

    # click FM (e.g. via the outliner/3D view), then the element to chain
    sel.select("FM", origin="outliner")
    pane.tree.setCurrentItem(pane.item_for_element("DET"))

    pane.chain_selected()
    assert project.train().records()["DET"]["ref"] == "FM"


def test_chain_selected_falls_back_to_solve_order_without_history(qtbot):
    project, _ = make_scene()
    pane, _sel = make_pane(qtbot, project)
    # everything anchored: sort_chain's order is [DET, FM, L1, L2, SRC]
    # (alphabetical roots); the last one that isn't L2 itself is SRC
    assert pane._last_train_element("L2") == "SRC"

    pane.tree.setCurrentItem(pane.item_for_element("L2"))
    pane.chain_selected()
    # no prior distinct selection: falls back to the last element in
    # solve order (never alphabetical tree POSITION, which would have
    # picked whatever sits directly above L2 in the tree)
    assert project.train().records()["L2"]["ref"] == "SRC"


def test_chain_selected_refuses_when_everything_is_downstream(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "20"})
    project.set_chain("FM", {"ref": "L1", "distance": "5", "fold": True,
                             "folded": True})
    project.set_chain("DET", {"ref": "FM", "port": "reflect",
                              "distance": "10"})
    pane, _sel = make_pane(qtbot, project)

    pane.tree.setCurrentItem(pane.item_for_element("SRC"))
    pane.chain_selected()
    assert pane.status.text().startswith("SRC:")
    assert not project.train().is_chained("SRC")


def test_chain_selected_no_selection_refuses(qtbot):
    project, _ = make_scene()
    pane, _sel = make_pane(qtbot, project)
    pane.chain_selected()
    assert "Select an element" in pane.status.text()


# ---------------------------------------------------------------------------
# Phase G: status/error ergonomics
# ---------------------------------------------------------------------------
def test_error_status_prefixed_with_element_and_scrolls(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    pane, _sel = make_pane(qtbot, project)

    assert pane.commit_field("L1", "distance", "2*+") is False
    assert pane.status.text().startswith("L1:")
    assert "Invalid" in pane.status.text()
    # scrollToItem must not raise even offscreen with an unrealized viewport
    assert pane.item_for_element("L1") is not None


def test_descendant_refusal_still_prefixed(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "20"})
    pane, _sel = make_pane(qtbot, project)

    pane.begin_pick_reference("L1")
    pane.on_reference_picked("L2", None)
    assert pane.status.text().startswith("L1:")
    assert "descendant" in pane.status.text()


# ---------------------------------------------------------------------------
# Phase G: header tooltips (beam-frame documentation)
# ---------------------------------------------------------------------------
def test_header_tooltips_document_beam_frame(qtbot):
    project, _ = make_scene()
    pane, _sel = make_pane(qtbot, project)
    header = pane.tree.headerItem()
    for col in (COL_DECX, COL_DECY, COL_TILTX, COL_TILTY, COL_TILTZ):
        tip = header.toolTip(col)
        assert "beam" in tip.lower()
    # X/Y/Z naming is kept (matches the stored field names); only tooltip'd
    assert pane.tree.headerItem().text(COL_DECX) == "Dec X"
    assert pane.tree.headerItem().text(COL_TILTZ) == "Tilt Z"


# ---------------------------------------------------------------------------
# Phase G: anchored cross-navigation
# ---------------------------------------------------------------------------
def test_context_menu_offers_set_absolute_pose_for_anchored_and_emits(qtbot):
    project, _ = make_scene()
    pane, _sel = make_pane(qtbot, project)

    menu = pane._build_context_menu("SRC")     # SRC is an anchored root
    texts = [a.text() for a in menu.actions()]
    assert "Set absolute pose…" in texts

    seen = []
    pane.editAnchorRequested.connect(seen.append)
    act = next(a for a in menu.actions() if a.text() == "Set absolute pose…")
    act.trigger()
    assert seen == ["SRC"]


def test_context_menu_offers_mark_fold_and_port_submenu_for_chained(qtbot):
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("FM", {"ref": "L1", "distance": "20"})   # plain, no fold
    pane, _sel = make_pane(qtbot, project)

    menu = pane._build_context_menu("FM")
    texts = [a.text() for a in menu.actions()]
    assert "Mark as fold mirror" in texts
    assert any(t.startswith("Chain onto port") for t in texts)
    # a chained row never offers "Set absolute pose..." (that's the
    # anchored-only cross-navigation entry)
    assert "Set absolute pose…" not in texts

    act = next(a for a in menu.actions() if a.text() == "Mark as fold mirror")
    act.trigger()
    assert project.train().records()["FM"]["fold"] is True

    menu2 = pane._build_context_menu("FM")
    texts2 = [a.text() for a in menu2.actions()]
    assert "Unmark fold" in texts2
    assert "Mark as fold mirror" not in texts2
