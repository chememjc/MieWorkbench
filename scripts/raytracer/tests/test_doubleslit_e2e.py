# =============================================================================
# test_doubleslit_e2e.py — THE end-to-end wave-optics validation: a real
# FreeCAD scene (doubleslit.FCStd authored by make_test_scenes.py, extracted
# by extract_geometry.py) traced through the full engine must produce
# Young fringes with pitch lambda*L/d and high visibility on the screen.
#
# Requires geometry/doubleslit/model.json (run make_test_scenes.py +
# extract_geometry.py first); skipped otherwise. Takes ~1-3 min.
# Run: /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_doubleslit_e2e.py -v
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import common                                            # noqa: E402
from raytracer.materials import MaterialDB, load_coatings  # noqa: E402
from raytracer.scene import Scene                        # noqa: E402
from raytracer.sources import sample_source              # noqa: E402
from raytracer.tracer import Tracer, TraceConfig         # noqa: E402
from raytracer.detector import DetectorGrid              # noqa: E402
from raytracer import gather                             # noqa: E402

MODEL_JSON = SCRIPTS.parent / "geometry" / "doubleslit" / "model.json"

pytestmark = pytest.mark.skipif(
    not MODEL_JSON.exists(),
    reason="author + extract doubleslit.FCStd first (make_test_scenes.py)")

# scene constants as authored by make_test_scenes.py
LAM = 633e-9
D_SEP = 0.5e-3
SLIT_W = 0.1e-3
L_PLATE_TO_SCREEN = 0.099        # filler exit x=1 mm -> screen x=100 mm


def test_double_slit_end_to_end():
    model = common.load_model(MODEL_JSON)
    db = MaterialDB.load()
    scene = Scene(model, db, load_coatings(db=db))
    grids = {fid: DetectorGrid(scene.faces[fid], 1024, 4,
                               (600e-9, 660e-9),
                               label=scene.faces[fid].id)
             for fid in scene.detector_faces}
    cfg = TraceConfig(rays=150000, n_lambda=1, seed=3)
    tracer = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(3)
    bidx, src = scene.sources[0]
    batch = sample_source(scene, scene.bodies[bidx], src, 0, cfg.rays, 1,
                          rng, ledger=tracer.ledger)
    tracer.run([batch])
    det = list(grids.values())[0]
    area = scene.emit_faces[bidx].area_m2 / cfg.rays
    gather.render_coherent(det, {(0, 0): area}, backend="auto",
                           min_eff_samples=500)
    irr = det.inc.sum(axis=0)
    px = det.pixel_m

    # fringes vary along GLOBAL y: find which grid axis that is from the
    # recorded basis (never assume — the basis tie-break is arbitrary)
    y_along_x = abs(det.xhat[1]) > abs(det.yhat[1])
    if y_along_x:
        band = irr[det.H // 2 - 100:det.H // 2 + 100, :].mean(axis=0)
    else:
        band = irr[:, det.W // 2 - 100:det.W // 2 + 100].mean(axis=1)

    mid = len(band) // 2
    nh = int(0.55e-3 / px)
    prof = band[mid - nh:mid + nh]
    y = (np.arange(len(prof)) - len(prof) / 2) * px

    # correlation with the analytic Fraunhofer pattern
    ana = (np.cos(np.pi * D_SEP * y / (LAM * L_PLATE_TO_SCREEN)) ** 2
           * np.sinc(SLIT_W * y / (LAM * L_PLATE_TO_SCREEN)) ** 2)
    corr = np.corrcoef(np.maximum(prof, 0), ana)[0, 1]
    assert corr > 0.75, "fringe pattern does not match Fraunhofer (r=%.3f)" \
        % corr

    # pitch from the central fringe minima
    sm = np.convolve(np.maximum(prof, 0), np.ones(3) / 3, mode="same")
    mins = [i for i in range(2, len(sm) - 2)
            if sm[i] < sm[i - 1] and sm[i] <= sm[i + 1]
            and sm[i] < 0.5 * sm.max()]
    pos = np.array(mins) * px
    sp = np.diff(pos)
    sp = sp[(sp > 0.6 * LAM * L_PLATE_TO_SCREEN / D_SEP)
            & (sp < 1.6 * LAM * L_PLATE_TO_SCREEN / D_SEP)]
    assert len(sp) >= 2, "not enough fringe minima found"
    pitch = float(np.median(sp))
    expect = LAM * L_PLATE_TO_SCREEN / D_SEP
    # pixel quantization at 11.7 um/px limits pitch accuracy to ~1 px
    assert pitch == pytest.approx(expect, abs=1.5 * px), (pitch, expect)

    # visibility over the central fringes
    seg = np.maximum(
        prof[len(prof) // 2 - int(1.5 * expect / px):
             len(prof) // 2 + int(1.5 * expect / px)], 0)
    V = (seg.max() - seg.min()) / (seg.max() + seg.min())
    assert V > 0.85, "fringe visibility %.3f too low" % V

    # energy closure of the whole trace
    rep = tracer.ledger.report([scene.bodies[bidx].label])
    err = list(rep["sources"].values())[0]["closure_error"]
    assert err < 1e-3, err
