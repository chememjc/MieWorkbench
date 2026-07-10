"""Real-worker demo-build tier (marked 'freecad', env-gated): build the
representative demos through the REAL fc_server via the chain API and
exercise fold round-trips on the real document. The scripted-worker
twin (test_demo_builds.py) covers all eleven demos per-pane; this tier
proves the same flows against actual FreeCAD geometry.

Run: MIEWB_RUN_FREECAD=1 QT_QPA_PLATFORM=offscreen env/bin/python \
         -m pytest mieworkbench/tests/test_demo_builds_freecad.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

pytestmark = pytest.mark.freecad


def _build(name, tmp_path):
    import make_demos
    from mieworkbench.core.fcclient import FcClient
    fc = FcClient()
    demo = make_demos.Demo(fc, tmp_path / ("%s.FCStd" % name))
    simparams = make_demos.DEMOS[name](demo)
    return fc, demo, simparams


def _pos(demo, label):
    body = demo.project.train().primary_body_name(label)
    return demo.project.body_states[body].current.to_dict()["pos_mm"]


def test_newtonian_real_build_and_fold_roundtrip(tmp_path):
    fc, demo, _ = _build("newtonian", tmp_path)
    try:
        # expect() self-checks already ran inside the demo function; now
        # exercise the fold round-trip against the REAL document
        eye_folded = list(_pos(demo, "Eyepiece"))
        demo.project.set_fold_state("Diagonal", False)
        eye_flat = _pos(demo, "Eyepiece")
        assert abs(eye_flat[1]) < 1e-9              # straightened onto -x
        assert np.allclose(eye_flat, [-900.0, 0, 0], atol=1e-6)
        props = demo.project.body(
            demo.project.train().primary_body_name("Diagonal"))["properties"]
        assert props["miewb_exclude"]["value"] is True
        demo.project.set_fold_state("Diagonal", True)
        assert np.allclose(_pos(demo, "Eyepiece"), eye_folded, atol=0)
        demo.save()
    finally:
        fc.shutdown()


def test_michelson_folded_real_build_unfolds_to_michelson(tmp_path):
    fc, demo, _ = _build("michelson_folded", tmp_path)
    try:
        m1_folded = list(_pos(demo, "M1"))
        assert np.allclose(m1_folded, [45, 15, 0], atol=1e-9)
        demo.project.set_folds_all(False)
        # both folds open: M1 back at the plain-michelson position
        assert np.allclose(_pos(demo, "M1"), [60, 0, 0], atol=1e-9)
        demo.project.set_folds_all(True)
        assert np.allclose(_pos(demo, "M1"), m1_folded, atol=0)
        # variable ripple on the real worker: shrink the dogleg height
        demo.project.apply_variable_cells(
            [{"cell": "B3", "raw": "=10", "alias": "fold_up"}],
            text="Shrink dogleg")
        assert np.allclose(_pos(demo, "FoldB"), [20, 10, 0], atol=1e-9)
        assert np.allclose(_pos(demo, "M1"), [50, 10, 0], atol=1e-9)
        demo.project.undo()
        assert np.allclose(_pos(demo, "M1"), m1_folded, atol=1e-9)
        demo.save()
    finally:
        fc.shutdown()


def test_czerny_turner_real_build_validates(tmp_path):
    fc, demo, _ = _build("czerny_turner", tmp_path)
    try:
        problems = demo.project.train().validate()
        assert not [m for s, m in problems if s == "error"]
        demo.save()
    finally:
        fc.shutdown()
