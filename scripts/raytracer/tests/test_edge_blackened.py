# =============================================================================
# test_edge_blackened.py -- edge blackening on lens primitives (engine3 Sec 11
# / P8): the cylindrical lens barrel absorbs stray/ghost light while the
# refracting surfaces stay clear. Per-face absorbance driven by the
# edge_blackened body property, resolved by the extractor to the barrel
# (cylinder) faces (immune to FaceN renumbering: identified by surface TYPE).
#
# FreeCAD + optics env, MIEWB_RUN_FREECAD-gated + slow. Authors two identical
# diverging-lens scenes (clear vs blackened rim), extracts, traces both, and
# checks the blackened rim absorbs the edge-incident light the clear rim
# transmits.
#
# Run: MIEWB_RUN_FREECAD=1 /home3/optics/env/bin/python -m pytest \
#         scripts/raytracer/tests/test_edge_blackened.py -q -m ''
# =============================================================================
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

RUN_FREECAD = os.environ.get("MIEWB_RUN_FREECAD") == "1"
FREECAD = os.environ.get("MIEWB_FREECAD", "/home3/freecad/FreeCAD.AppImage")
OPTPROPS = str(SCRIPTS.parent / "opticalproperties")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not RUN_FREECAD,
                       reason="set MIEWB_RUN_FREECAD=1 for the FreeCAD e2e"),
]


def _trace(geo, tag):
    import common
    from raytracer.scene import Scene
    from raytracer.sources import sample_source
    from raytracer.tracer import Tracer, TraceConfig
    from raytracer.detector import DetectorGrid
    from raytracer.optprops import load_optical_properties
    opt = load_optical_properties(root=OPTPROPS)
    model = common.load_model(geo / tag / "model.json")
    common.validate_model(model)
    sc = Scene(model, opt.matdb, opt.coatings, optprops=opt,
               geometry_dir=str(geo / tag))
    fid = list(sc.detector_faces)[0]
    grids = {fid: DetectorGrid(sc.faces[fid], 200, 4, (600e-9, 660e-9),
                               label=sc.faces[fid].id)}
    cfg = TraceConfig(rays=60000, n_lambda=1, seed=3, power_floor=1e-12)
    tr = Tracer(sc, cfg, grids)
    rng = np.random.default_rng(3)
    b, s = sc.sources[0]
    batch = sample_source(sc, sc.bodies[b], s, 0, cfg.rays, 1, rng,
                          ledger=tr.ledger)
    res = tr.run([batch])
    rep = res.ledger.report([sc.bodies[b].label])
    src = list(rep["sources"].values())[0]
    return sc, src, float(grids[fid].inc.sum())


def test_edge_blackening_absorbs_the_barrel_ghost(tmp_path):
    probe = Path(__file__).resolve().parent / "_edge_blackened_fc_probe.py"
    subprocess.run([FREECAD, "-c", str(probe), "--", "--outdir", str(tmp_path)],
                   stdin=subprocess.DEVNULL, check=True, timeout=600,
                   capture_output=True)
    geo = tmp_path / "geo"
    subprocess.run(
        [FREECAD, "-c", str(SCRIPTS / "extract_geometry.py"), "--",
         "--models", str(tmp_path / "edge2_0.FCStd"),
         str(tmp_path / "edge2_1.FCStd"), "--outdir", str(geo)],
        stdin=subprocess.DEVNULL, check=True, timeout=600, capture_output=True)

    sc0, r0, d0 = _trace(geo, "edge2_0")
    sc1, r1, d1 = _trace(geo, "edge2_1")

    # mechanism: only the blackened lens gets a per-face absorbance override,
    # and it targets exactly the cylindrical barrel (the whole-body absorbance
    # stays 0 -- the refracting caps are NOT dimmed)
    assert len(sc0.face_absorbance) == 0
    assert len(sc1.face_absorbance) >= 1
    for fid, a in sc1.face_absorbance.items():
        assert a == pytest.approx(1.0)
        assert type(sc1.faces[fid].surface).__name__ == "Cylinder"
    lens1 = next(b for b in sc1.bodies if b.label == "Lens")
    assert lens1.absorbance == 0.0

    # physics: the blackened rim absorbs the edge-incident ghost light the
    # clear rim transmits -> surface absorption appears, detected light drops
    assert r1["absorbed_surface"] > r0["absorbed_surface"] + 1e-8
    assert d1 <= d0
    # both scenes still close energy
    assert r0["closure_error"] < 1e-3
    assert r1["closure_error"] < 1e-3
