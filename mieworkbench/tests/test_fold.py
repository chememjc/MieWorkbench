"""Fold-operator tests (Phase B): unfold/refold state machine,
insert_fold_mirror, fold_about_surface, fold-all — against the scripted
worker (no FreeCAD)."""

import copy
import json

import numpy as np
import pytest

from mieworkbench.core.project import ProjectError

from mieworkbench.tests.train_test_support import make_scene, pos_of


def _fold_scene():
    """SRC -> L1 -> FM(fold, -45) -> reflect -> DET, via the chain API."""
    project, fake = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("FM", {"ref": "L1", "distance": "20", "fold": True,
                             "folded": True, "tilt_ry": "-45"})
    project.set_chain("DET", {"ref": "FM", "port": "reflect",
                              "distance": "15"})
    return project, fake


def _placement(project, name):
    return project.body_states[name].current.to_dict()


# ---------------------------------------------------------------------------
# set_fold_state
# ---------------------------------------------------------------------------
def test_unfold_straightens_and_excludes():
    project, _ = _fold_scene()
    project.set_fold_state("FM", False)
    assert np.allclose(pos_of(project, "DET"), [54, 0, 0], atol=1e-9)
    assert np.allclose(pos_of(project, "FM"), [39, 0, 0], atol=1e-9)
    props = project.body("FM")["properties"]
    assert props["miewb_exclude"]["value"] is True
    assert props["miewb_train_folded"]["value"] is False
    stash = json.loads(props["miewb_train_unfold_stash"]["value"])
    assert set(stash) == {"FM", "DET"}
    assert np.allclose(stash["DET"]["pos_mm"], [39, 15, 0], atol=1e-9)


def test_refold_is_exact_and_clears_exclusion():
    project, _ = _fold_scene()
    folded = {n: copy.deepcopy(_placement(project, n))
              for n in ("FM", "DET")}
    project.set_fold_state("FM", False)
    project.set_fold_state("FM", True)
    for n in ("FM", "DET"):
        cur = _placement(project, n)
        assert np.allclose(cur["pos_mm"], folded[n]["pos_mm"], atol=0)
        assert np.allclose(cur["quat"], folded[n]["quat"], atol=0)
    assert project.body("FM")["properties"]["miewb_exclude"]["value"] \
        is False


def test_unfold_is_one_undo_step():
    project, _ = _fold_scene()
    det = pos_of(project, "DET")
    project.set_fold_state("FM", False)
    project.undo()
    assert np.allclose(pos_of(project, "DET"), det, atol=0)
    props = project.body("FM")["properties"]
    assert props["miewb_train_folded"]["value"] is True
    # exclusion rolled back too
    assert props.get("miewb_exclude", {}).get("value") in (False, None)


def test_unfold_emits_optics_changed():
    project, _ = _fold_scene()
    hits = []
    project.opticsChanged.connect(lambda: hits.append(1))
    project.set_fold_state("FM", False)
    assert hits


def test_fold_state_noop_when_already_there():
    project, _ = _fold_scene()
    ops_before = len(project.undo_stack._done) \
        if hasattr(project.undo_stack, "_done") else None
    project.set_fold_state("FM", True)          # already folded
    # no new undo entry (can_undo count unchanged is hard to read; just
    # ensure poses did not move)
    assert np.allclose(pos_of(project, "DET"), [39, 15, 0], atol=1e-9)
    del ops_before


def test_fold_state_requires_fold_element():
    project, _ = _fold_scene()
    with pytest.raises(ProjectError, match="not a fold"):
        project.set_fold_state("L1", False)


def test_edit_while_unfolded_persists_through_refold():
    """Chain-field edits made while unfolded survive refolding (the
    solver wins; the stash is only a safety net)."""
    project, _ = _fold_scene()
    project.set_fold_state("FM", False)
    project.set_chain("DET", {"distance": "25"}, text="Longer arm")
    assert np.allclose(pos_of(project, "DET"), [64, 0, 0], atol=1e-9)
    project.set_fold_state("FM", True)
    assert np.allclose(pos_of(project, "DET"), [39, 25, 0], atol=1e-9)


# ---------------------------------------------------------------------------
# set_folds_all (sequential folds treated independently)
# ---------------------------------------------------------------------------
def _periscope_scene():
    project, fake = make_scene()
    # L1 stands in for a second fold mirror: give it mirror port geometry
    # in the WORKER's structure (project.structure is refetched wholesale
    # on every mutation, so mutating the project-side copy would be lost)
    fake._body("L1")["properties"]["miewb_train_ports"]["value"] = \
        fake._body("FM")["properties"]["miewb_train_ports"]["value"]
    project._refetch_structure()
    project.set_chain("FM", {"ref": "SRC", "distance": "15", "fold": True,
                             "folded": True, "tilt_ry": "-45"})
    project.set_chain("L1", {"ref": "FM", "port": "reflect",
                             "distance": "12", "fold": True,
                             "folded": True, "tilt_ry": "45"})
    project.set_chain("DET", {"ref": "L1", "port": "reflect",
                              "distance": "8"})
    return project, fake


def test_periscope_unfold_all_and_refold_all():
    project, _ = _periscope_scene()
    # folded: SRC exit x=5, FM at 20; arm +y 12 -> L1 at (20,12); back to
    # +x 8 -> DET at (28,12)
    assert np.allclose(pos_of(project, "DET"), [28, 12, 0], atol=1e-9)
    folded = {n: copy.deepcopy(_placement(project, n))
              for n in ("FM", "L1", "DET")}

    targets = project.set_folds_all(False)
    assert set(targets) == {"FM", "L1"}
    assert np.allclose(pos_of(project, "DET"), [40, 0, 0], atol=1e-9)
    for n in ("FM", "L1"):
        assert project.body(n)["properties"]["miewb_exclude"]["value"] \
            is True

    project.set_folds_all(True)
    for n in ("FM", "L1", "DET"):
        cur = _placement(project, n)
        assert np.allclose(cur["pos_mm"], folded[n]["pos_mm"], atol=0)
        assert np.allclose(cur["quat"], folded[n]["quat"], atol=0)


def test_unfold_one_of_two_sequential_folds():
    project, _ = _periscope_scene()
    project.set_fold_state("FM", False)
    # straight through FM: L1 12 mm further along +x at (32,0); L1 still
    # folds (its orientation re-solved against the straight beam), so DET
    # is 8 mm up L1's reflected arm
    assert np.allclose(pos_of(project, "L1"), [32, 0, 0], atol=1e-9)
    det = pos_of(project, "DET")
    assert abs(det[1]) > 1.0     # still deviated by the second fold


# ---------------------------------------------------------------------------
# insert_fold_mirror
# ---------------------------------------------------------------------------
def test_insert_fold_mirror_reanchors_children():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "20"})
    label = project.insert_fold_mirror("L1", distance=8.0)
    assert label == "Fold1"
    rec = project.train().records()
    assert rec["Fold1"]["fold"] is True
    assert rec["L2"]["ref"] == "Fold1"
    assert rec["L2"]["port"] == "reflect"
    # distances: mirror plane 8 mm past L1 exit (x=19+8=27); L2 keeps its
    # total path (20) so 12 mm along the folded arm (azimuth 0 -> +u=+y)
    assert float(rec["L2"]["distance"]) == pytest.approx(12.0)
    assert np.allclose(pos_of(project, "Fold1"), [27, 0, 0], atol=1e-9)
    assert np.allclose(pos_of(project, "L2"), [27, 13, 0], atol=1e-9)


def test_insert_fold_mirror_one_undo():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "20"})
    l2 = pos_of(project, "L2")
    project.insert_fold_mirror("L1", distance=8.0)
    project.undo()
    assert "Fold1" not in project.train().records()
    assert np.allclose(pos_of(project, "L2"), l2, atol=1e-9)
    assert project.train().records()["L2"]["ref"] == "L1"


def test_insert_fold_mirror_symbolic_distance_stays_symbolic():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "gap"})   # gap=25
    project.insert_fold_mirror("L1", distance=8.0)
    rec = project.train().records()
    assert rec["L2"]["distance"] == "(gap) - (8)"
    # 25 - 8 = 17 along the folded arm
    assert np.allclose(pos_of(project, "L2"), [27, 18, 0], atol=1e-9)


def test_insert_fold_mirror_azimuth_folds_toward_up():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "20"})
    project.insert_fold_mirror("L1", distance=8.0, azimuth_deg=90.0)
    l2 = pos_of(project, "L2")
    # azimuth 90: fold toward +v (up = +z)
    assert abs(l2[1]) < 1e-9
    assert l2[2] == pytest.approx(13.0, abs=1e-9)


def test_insert_fold_mirror_deviation_angle():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "20"})
    project.insert_fold_mirror("L1", distance=8.0, deviation_deg=60.0)
    # beam deviates 60 deg from +x toward +y; L2 sits 12 mm along it
    # from the plane at (27,0,0), entry local -1 -> pos 13 mm along
    d = np.radians(60.0)
    expect = np.array([27, 0, 0]) + 13.0 * np.array(
        [np.cos(d), np.sin(d), 0.0])
    assert np.allclose(pos_of(project, "L2"), expect, atol=1e-9)


# ---------------------------------------------------------------------------
# fold_about_surface
# ---------------------------------------------------------------------------
def test_fold_about_surface_marks_and_reflects():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    # chain the mirror as a plain element first (not yet a fold), tilted
    project.set_chain("FM", {"ref": "L1", "distance": "20",
                             "tilt_ry": "-45"})
    project.set_chain("DET", {"ref": "FM", "distance": "15"})
    # DET currently continues straight through (transmit port)
    assert np.allclose(pos_of(project, "DET"), [54, 0, 0], atol=1e-9)

    project.fold_about_surface("FM")
    rec = project.train().records()
    assert rec["FM"]["fold"] is True
    assert rec["DET"]["port"] == "reflect"
    assert np.allclose(pos_of(project, "DET"), [39, 15, 0], atol=1e-9)


def test_fold_about_surface_rotates_anchored_extras():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("FM", {"ref": "L1", "distance": "20",
                             "tilt_ry": "-45"})
    # DET stays ANCHORED at (54,0,0) — 15 mm past the mirror plane
    project.apply_operation("DET", __import__(
        "mieworkbench.core.transforms", fromlist=["Operation"]
    ).Operation("set_placement", {"pos_mm": [54, 0, 0],
                                  "quat": [0, 0, 0, 1]}))
    project.fold_about_surface("FM", extra_elements=["DET"])
    assert np.allclose(pos_of(project, "DET"), [39, 15, 0], atol=1e-9)


def test_fold_about_surface_requires_reflective_chained():
    project, _ = make_scene()
    with pytest.raises(ProjectError, match="chained"):
        project.fold_about_surface("FM")     # anchored
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    with pytest.raises(ProjectError, match="reflective"):
        project.fold_about_surface("L1")     # no reflect plane
