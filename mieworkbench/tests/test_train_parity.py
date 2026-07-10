"""The GUI <-> permute parity oracle (marked 'freecad', env-gated).

THE drift-killer test for the optical-train feature: a chained scene is
built through the Project API against the real worker, saved, and then
re-baked headlessly by permute_model.py (which re-solves the train via
train_fcstd/train_solver per variant). The placements written by the two
paths must agree to 1e-9 — the GUI's live solve and the headless variant
solve are the SAME solver fed the same inputs, and this test keeps it
that way.

Run: MIEWB_RUN_FREECAD=1 QT_QPA_PLATFORM=offscreen env/bin/python \
         -m pytest mieworkbench/tests/test_train_parity.py -q
"""

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.project import Project  # noqa: E402

pytestmark = pytest.mark.freecad

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRIMITIVES = os.path.join(REPO, "primitives")
APPIMAGE = os.environ.get("MIEWB_FREECAD", "/home3/freecad/FreeCAD.AppImage")


def run_permute(model_path, outdir, var, vmin, vmax, n):
    script = os.path.join(REPO, "scripts", "permute_model.py")
    proc = subprocess.run(
        [APPIMAGE, "-c", script, "--", "--model", str(model_path),
         "--var", var, "--min", str(vmin), "--max", str(vmax),
         "--n", str(n), "--outdir", str(outdir)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True)
    assert proc.returncode == 0, \
        "permute failed:\n%s\n%s" % (proc.stdout[-2000:], proc.stderr[-800:])
    return proc


@pytest.fixture()
def project(tmp_path):
    p = Project()
    p.new_document(str(tmp_path / "parity.FCStd"))
    yield p
    p.shutdown()


def build_chained_scene(project, gap=18.0):
    """laser -> lens_pcx -> fold mirror -> detector, chained with a
    variable-driven distance and a fold."""
    fc, doc = project.fc, project.doc
    # variables sheet through the new worker ops
    fc.request("create_sheet", {"doc": doc, "label": "miewb_vars"})
    fc.request("set_cell", {"doc": doc, "sheet": "miewb_vars",
                            "cell": "B1", "raw": "=%g" % gap,
                            "alias": "gap"})
    fc.request("set_cell", {"doc": doc, "sheet": "miewb_vars",
                            "cell": "B2", "raw": "=gap/2 + 3",
                            "alias": "half"})
    project._refetch_structure()

    for kind, label in (("laser_collimated", "SRC"),
                        ("lens_pcx", "L1"),
                        ("mirror_flat", "FM"),
                        ("detector_plane", "DET")):
        project.import_primitive(
            os.path.join(PRIMITIVES, "%s.FCStd" % kind), label)

    project.set_chain("L1", {"ref": "SRC", "distance": "gap*2"})
    project.set_chain("FM", {"ref": "L1", "distance": "half",
                             "fold": True, "folded": True,
                             "tilt_ry": "-45"})
    project.set_chain("DET", {"ref": "FM", "port": "reflect",
                              "distance": "25", "decenter_y": "0.5",
                              "tilt_rx": "2"})


def placements_from_worker(fc, path):
    structure = fc.open_document(str(path))
    try:
        return {b["label"]: b["placement"] for b in structure["bodies"]}
    finally:
        fc.close(structure["doc"])


def assert_placements_close(got, want, labels, atol=1e-9):
    for label in labels:
        g, w = got[label], want[label]
        assert np.allclose(g["pos_mm"], w["pos_mm"], atol=atol), \
            "%s pos: %s != %s" % (label, g["pos_mm"], w["pos_mm"])
        q1, q2 = np.array(g["quat"]), np.array(w["quat"])
        d = min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2))
        assert d < atol, "%s quat: %s != %s" % (label, g["quat"], w["quat"])


def test_parity_same_value_roundtrip(project, tmp_path):
    """Same variable value: permute's re-bake must reproduce the GUI's
    placements exactly (identical solver, identical inputs)."""
    build_chained_scene(project, gap=18.0)
    gui = {el: st.to_dict()
           for el, st in ((b["label"],
                           project.body_states[b["name"]].current)
                          for b in project.structure["bodies"])}
    project.save()

    outdir = tmp_path / "variants"
    run_permute(project.fcstd_path, outdir, "miewb_vars.gap", 18.0, 18.0, 0)
    variant = outdir / "parity-miewb_vars_gap18.FCStd"
    assert variant.exists(), sorted(os.listdir(outdir))

    got = placements_from_worker(project.fc, variant)
    assert_placements_close(got, gui, ["SRC", "L1", "FM", "DET"])


def test_parity_swept_value_matches_gui_prediction(project, tmp_path):
    """Different variable value: the variant's placements must equal
    what the GUI solver predicts for that value."""
    build_chained_scene(project, gap=18.0)
    project.save()

    # GUI-side prediction at gap=30 (resolve the dependent 'half' too)
    tm = project.train()
    import train_solver
    variables = train_solver.resolve_variables(
        {"gap": "30", "half": "gap/2 + 3"})
    predicted = tm.solve(variables)["placements"]

    outdir = tmp_path / "variants"
    run_permute(project.fcstd_path, outdir, "miewb_vars.gap", 30.0, 30.0, 0)
    variant = outdir / "parity-miewb_vars_gap30.FCStd"
    assert variant.exists(), sorted(os.listdir(outdir))

    got = placements_from_worker(project.fc, variant)
    assert_placements_close(got, predicted, ["L1", "FM", "DET"])
    # sanity: the sweep actually moved things vs the authored gap
    gui_now = {b["label"]:
               project.body_states[b["name"]].current.to_dict()
               for b in project.structure["bodies"]}
    assert not np.allclose(got["L1"]["pos_mm"], gui_now["L1"]["pos_mm"])


def extract_model_json(fcstd_path, out_dir):
    """Run the real extract_geometry.py on one model, return model.json."""
    import json
    script = os.path.join(REPO, "scripts", "extract_geometry.py")
    proc = subprocess.run(
        [APPIMAGE, "-c", script, "--", "--models", str(fcstd_path),
         "--outdir", str(out_dir)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-500:]
    stem = os.path.splitext(os.path.basename(str(fcstd_path)))[0]
    with open(os.path.join(str(out_dir), stem, "model.json")) as fh:
        return json.load(fh)


def test_unfolded_mirror_excluded_from_extraction(project, tmp_path):
    """miewb_exclude (set by unfold) removes the fold mirror from the
    physics contract entirely; refolding restores it."""
    build_chained_scene(project, gap=18.0)
    project.set_fold_state("FM", False)
    project.save()
    model = extract_model_json(project.fcstd_path, tmp_path / "ex_unfolded")
    labels = {b.get("label") for b in model.get("bodies", [])}
    assert "FM" not in labels
    assert {"SRC", "L1", "DET"} <= labels

    project.set_fold_state("FM", True)
    project.save()
    model = extract_model_json(project.fcstd_path, tmp_path / "ex_folded")
    labels = {b.get("label") for b in model.get("bodies", [])}
    assert "FM" in labels


def test_parity_unfolded_state_survives_permute(project, tmp_path):
    """Unfold in the GUI, save; the permuted variant must keep the train
    unfolded (fold state is document state, not run state)."""
    build_chained_scene(project, gap=18.0)
    project.set_chain("FM", {"folded": False}, text="Unfold FM")
    gui = {b["label"]: project.body_states[b["name"]].current.to_dict()
           for b in project.structure["bodies"]}
    project.save()

    outdir = tmp_path / "variants"
    run_permute(project.fcstd_path, outdir, "miewb_vars.gap", 18.0, 18.0, 0)
    variant = outdir / "parity-miewb_vars_gap18.FCStd"
    got = placements_from_worker(project.fc, variant)
    assert_placements_close(got, gui, ["SRC", "L1", "FM", "DET"])
    # and the detector really is on the straight-through axis: the fold
    # was a -45 about v, so folded DET sat off the x axis; unfolded it
    # must be back on it (y ~ decenter only)
    assert abs(got["DET"]["pos_mm"][1] - gui["DET"]["pos_mm"][1]) < 1e-9
