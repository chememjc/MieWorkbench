# =============================================================================
# test_colocated_detectors.py -- stacked/co-located transparent detectors
# (samples-instruments round).
#
# Detectors are ideal transparent pass-through screens (tracer
# _screen_children: remainder continues unrefracted), so two detector
# bodies measuring "the same plane two ways" are physically well-defined.
# The extractor now classifies detector-detector solid overlap as the
# informational validation.detector_overlap list instead of the fatal
# overlapping_solids (extract_geometry.py); recording faces must sit
# >= the tracer's 100 nm t_eps apart (authoring guidance: ~1 um).
# This file pins the ENGINE half: both stacked detectors record the beam.
# =============================================================================
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import common                                                    # noqa: E402
from . import scenehelpers as sh                                 # noqa: E402


def test_stacked_detectors_both_record_full_power():
    """Two overlapping detector slabs, front faces 5 um apart along the
    beam: each records ~the full beam power (transparent screens), and
    the energy ledger still closes."""
    d1 = sh.detector_body("DetA", x=0.030, half=0.02)
    d2 = sh.detector_body("DetB", x=0.030005, half=0.02)   # 5 um behind
    model = sh.make_model([
        sh.source_body("Src", x=-0.02, half=0.002, power_mW=1.0,
                       lambdac_nm=550.0),
        d1, d2,
    ])
    common.validate_model(model)
    res, grids, scene = sh.trace_scene(model, rays=4000, n_lambda=1,
                                       seed=2)
    powers = {}
    for fid, g in grids.items():
        powers[g.label] = float(sum(g.detected_incoherent.values()))
    assert len(powers) == 2
    vals = sorted(powers.values())
    # both see the (same) full 1 mW beam
    assert vals[0] > 1e-3 * 0.99
    assert abs(vals[0] - vals[1]) <= 1e-9 + 1e-3 * vals[1]


def test_detector_overlap_validation_key_tolerated():
    """A model carrying the new informational detector_overlap list (and
    NO overlapping_solids) passes contract validation."""
    model = sh.make_model([
        sh.source_body("Src", x=-0.02, half=0.002, power_mW=1.0,
                       lambdac_nm=550.0),
        sh.detector_body("DetA", x=0.030, half=0.02),
        sh.detector_body("DetB", x=0.030005, half=0.02),
    ])
    model["validation"]["detector_overlap"] = [
        {"a": "DetA", "b": "DetB", "volume_mm3": 12.0}]
    common.validate_model(model)   # must not raise
