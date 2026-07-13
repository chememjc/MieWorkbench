# =============================================================================
# test_thermal_defocus_demo.py -- end-to-end "working demo" for the thermo-optic
# temperature feature: trace a REAL bk7 double-convex lens (geometry/lens_dcx)
# at two operating temperatures and show that heating the glass raises its index
# and measurably defocuses the geometric spot on the detector -- while a same-
# temperature control is bit-identical. Complements the unit dn/dT oracle
# (test_materials_thermo.py) by proving temperature propagates all the way
# through the tracer to the detected image.
#
#   /home3/optics/env/bin/python -m pytest \
#       scripts/raytracer/tests/test_thermal_defocus_demo.py -v
# =============================================================================
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import common  # noqa: E402
from raytracer.scene import Scene  # noqa: E402
from raytracer.sources import sample_source  # noqa: E402
from raytracer.tracer import Tracer, TraceConfig  # noqa: E402
from raytracer.detector import DetectorGrid  # noqa: E402
from raytracer.optprops import load_optical_properties  # noqa: E402

MODEL = _ROOT / "geometry" / "lens_dcx" / "model.json"


def _trace_at_temperature(T, rays=50000, res=128, seed=3):
    """Trace geometry/lens_dcx (bk7 lens) at operating temperature T (deg C).
    Returns (detector image HxW, rms spot radius px, lens index at 633nm)."""
    model = json.loads(MODEL.read_text())
    common.validate_model(model)
    opt = load_optical_properties()
    scene = Scene(model, opt.matdb, opt.coatings, optprops=opt, temperature_c=T)
    lo = min(s["lambdac_nm"] for _, s in scene.sources) - 100.0
    hi = max(s["lambdac_nm"] for _, s in scene.sources) + 100.0
    grids = {fid: DetectorGrid(scene.faces[fid], res, 16, (lo * 1e-9, hi * 1e-9),
                               label=scene.faces[fid].id)
             for fid in scene.detector_faces}
    cfg = TraceConfig(rays=rays, n_lambda=1, seed=seed, power_floor=1e-12,
                      max_reflections=6)
    tr = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(seed)
    batches = [sample_source(scene, scene.bodies[b], s, i, cfg.rays,
                             cfg.n_lambda, rng, ledger=tr.ledger)
               for i, (b, s) in enumerate(scene.sources)]
    tr.run(batches)
    img = list(grids.values())[0].inc.sum(axis=0)
    H, W = img.shape
    ys, xs = np.mgrid[0:H, 0:W]
    p = img.sum()
    cy, cx = (ys * img).sum() / p, (xs * img).sum() / p
    rms = float(np.sqrt((((ys - cy) ** 2 + (xs - cx) ** 2) * img).sum() / p))
    lens_i = next(i for i, b in enumerate(scene.bodies) if b.material == "bk7")
    n = float(scene.medium_index(lens_i, 633e-9).real)
    return img, rms, n


@pytest.mark.skipif(not MODEL.exists(), reason="lens_dcx geometry not extracted")
def test_thermal_defocus_end_to_end():
    img_cold, rms_cold, n_cold = _trace_at_temperature(20.0)
    img_ctrl, _, _ = _trace_at_temperature(20.0)          # same-T control
    img_hot, rms_hot, n_hot = _trace_at_temperature(400.0)

    # 1. temperature reached the trace: bk7 index rose with T (dn/dT > 0)
    assert n_hot > n_cold
    assert (n_hot - n_cold) == pytest.approx(1.08e-3, rel=0.2)

    # 2. the same-temperature run is deterministic / bit-identical
    assert np.array_equal(img_cold, img_ctrl)

    # 3. heating measurably changed the detected image (not a no-op)
    assert np.abs(img_hot - img_cold).sum() > 1e-6

    # 4. correct physics: higher index -> shorter focal length -> the detector
    #    (fixed) sits past the new focus -> the geometric spot grows (defocus)
    assert rms_hot > rms_cold

    # 5. no gross energy leak across the temperature change: the detected
    #    power shifts only slightly (higher n raises the surfaces' Fresnel
    #    reflectance -> a real ~1e-4 change, not a spurious loss).
    assert img_hot.sum() == pytest.approx(img_cold.sum(), rel=1e-2)
