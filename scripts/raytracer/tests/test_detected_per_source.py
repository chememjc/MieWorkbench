# =============================================================================
# test_detected_per_source.py — per-(source, lam_stratum, pol_stratum)
# detected-power retention added to DetectorGrid (detector.py's
# detected_incoherent alongside the pre-existing detected_geometric) and
# surfaced through run_trace.py's case.json["detected"] block.
#
# Scene: two sources (one coherent, one not) aimed directly at one detector
# with nothing in between, so every emitted watt lands on the detector and
# the ledger's detected_W is exactly the sum of both populations' tallies.
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest scripts/raytracer/tests/test_detected_per_source.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import common                                            # noqa: E402
from raytracer.scene import Scene                        # noqa: E402
from raytracer.sources import sample_source               # noqa: E402
from raytracer.tracer import Tracer, TraceConfig          # noqa: E402
from raytracer.detector import DetectorGrid                # noqa: E402
from raytracer import gather                              # noqa: E402
from raytracer.optprops import load_optical_properties    # noqa: E402
from raytracer.tests import scenehelpers as sh             # noqa: E402


def _two_source_scene():
    src_coh = sh.source_body("SrcCoh", x=-0.02, half=1e-3,
                             lambdac_nm=633.0, coherent=True,
                             polarization={"kind": "linear",
                                          "angle_deg": 0.0})
    src_inc = sh.source_body("SrcInc", x=-0.02, half=1e-3,
                             lambdac_nm=633.0, coherent=False,
                             polarization={"kind": "linear",
                                          "angle_deg": 90.0})
    det = sh.detector_body("Det", x=0.03, half=0.02)
    return sh.make_model([src_coh, src_inc, det])


def _build_and_run(rays=8000, n_lambda=1, seed=11):
    model = _two_source_scene()
    common.validate_model(model)
    optprops = load_optical_properties()
    scene = Scene(model, optprops.matdb, optprops.coatings, optprops=optprops)
    grids = {fid: DetectorGrid(scene.faces[fid], 128, 4,
                               (600e-9, 660e-9), label=scene.faces[fid].id)
             for fid in scene.detector_faces}
    cfg = TraceConfig(rays=rays, n_lambda=n_lambda, seed=seed,
                      power_floor=1e-9)
    tracer = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(seed)
    batches = [sample_source(scene, scene.bodies[b], s, i, cfg.rays,
                             cfg.n_lambda, rng, ledger=tracer.ledger)
               for i, (b, s) in enumerate(scene.sources)]
    result = tracer.run(batches)
    det = list(grids.values())[0]
    return result, det, scene


def test_detected_incoherent_and_geometric_keys_present():
    result, det, scene = _build_and_run()
    # source 0 (coherent) populates detected_geometric; source 1
    # (incoherent) populates detected_incoherent -- disjoint source_id sets.
    assert det.detected_geometric, "no coherent detected-power keys"
    assert det.detected_incoherent, "no incoherent detected-power keys"
    coh_sources = {k[0] for k in det.detected_geometric}
    inc_sources = {k[0] for k in det.detected_incoherent}
    assert coh_sources == {0}
    assert inc_sources == {1}
    # n_samples tracked per incoherent key too
    assert all(n > 0 for n in det.detected_incoherent_n.values())


def test_coherent_keys_match_gather_diags():
    result, det, scene = _build_and_run()
    n_lambda = 1
    sample_area = {}
    for sid, (bidx, src) in enumerate(scene.sources):
        area = scene.emit_faces[bidx].area_m2 or 1e-6
        from raytracer.sources import wavelength_strata, n_pol_strata
        n_strata = len(wavelength_strata(src, n_lambda))
        n_pol = n_pol_strata(src)
        rays_per_key = max(8000 / (n_strata * n_pol), 1)
        for st in range(n_strata):
            for ps in range(n_pol):
                sample_area[(sid, st, ps)] = area / rays_per_key
    diags = gather.render_coherent(det, sample_area, backend="numpy",
                                   enforce_gate=False)
    assert diags, "no coherent gather keys rendered"
    for key, d in diags.items():
        assert d["detected_geometric_W"] == det.detected_geometric[key]


def test_per_source_sum_equals_ledger_detected_total():
    result, det, scene = _build_and_run()
    ledger_total = result.ledger.detected[det.label]
    per_source_total = (sum(det.detected_geometric.values())
                        + sum(det.detected_incoherent.values()))
    assert per_source_total == pytest.approx(ledger_total, rel=1e-12)
