# =============================================================================
# test_element_flux.py — per-element boundary-flux tallies (in_W/out_W) and
# the separate detected_W ledger. These are DIAGNOSTIC tallies layered onto
# the trace loop with zero RNG consumption; the closure gate and every
# pinned physics invariant must be untouched (the rest of the suite is the
# no-change guard).
# =============================================================================
import numpy as np

from .scenehelpers import (
    detector_body, make_model, slab_body, source_body, trace_scene,
)


def _flux(rep, label):
    return rep["element_flux_W"][label]


def test_absorbing_window_in_minus_out_equals_absorbed():
    """source -> absorbing window -> detector: the window's boundary flux
    must satisfy in - out == power absorbed at/in the window (surface +
    bulk), to closure-level tolerance (truncation floor)."""
    model = make_model([
        source_body(power_mW=1.0, coherent=False),
        slab_body("Window", "bk7", 0.0, 0.005, absorbance=0.3),
        detector_body(x=0.03),
    ])
    result, grids, scene = trace_scene(model, rays=20000)
    rep = result.ledger.report([n for n, _ in scene.sources])
    assert rep["closure_ok"]

    fx = _flux(rep, "Window")
    emitted = sum(s["emitted_W"] for s in rep["sources"].values())
    # every emitted ray hits the (much larger) window face
    assert fx["in_W"] >= 0.999 * emitted
    absorbed = (rep["by_body_W"].get("Window", 0.0)
                + rep["by_surface_W"].get("Window", 0.0))
    assert absorbed > 0.2 * emitted        # absorbance=0.3 bites
    balance = fx["in_W"] - fx["out_W"] - absorbed
    assert abs(balance) <= 1e-3 * emitted


def test_detected_ledger_matches_detector_diagnostic():
    model = make_model([
        source_body(power_mW=1.0, coherent=False),
        slab_body("Window", "bk7", 0.0, 0.005),
        detector_body(name="Det", x=0.03),
    ])
    result, grids, scene = trace_scene(model, rays=20000)
    rep = result.ledger.report([n for n, _ in scene.sources])
    assert rep["closure_ok"]
    det = rep["detected_W"]
    # detector tallies are keyed by the detector FACE id (DetectorGrid
    # label convention)
    assert set(det) == {"Det.Pad.Face1"}
    detected = det["Det.Pad.Face1"]
    # detected power can't exceed what left the window toward it
    assert 0.0 < detected <= _flux(rep, "Window")["out_W"] * (1 + 1e-9)
    # by_surface keeps its historical mixed meaning; the detector entry
    # there must equal the dedicated detected tally
    assert np.isclose(rep["by_surface_W"]["Det.Pad.Face1"], detected)


def test_chain_flux_flows_downstream():
    """Two windows in series: what leaves W1 forward is what arrives at
    W2 (collimated beam, oversized apertures)."""
    model = make_model([
        source_body(power_mW=1.0, coherent=False),
        slab_body("W1", "bk7", 0.0, 0.003),
        slab_body("W2", "bk7", 0.01, 0.013),
        detector_body(x=0.03),
    ])
    result, grids, scene = trace_scene(model, rays=20000)
    rep = result.ledger.report([n for n, _ in scene.sources])
    assert rep["closure_ok"]
    w1, w2 = _flux(rep, "W1"), _flux(rep, "W2")
    # W1's out includes the front-face Fresnel back-reflection (leaves
    # toward the source), so out >= what W2 receives; the forward share
    # dominates for near-normal incidence on bk7
    assert w2["in_W"] <= w1["out_W"] * (1 + 1e-9)
    assert w2["in_W"] >= 0.85 * w1["out_W"]


def test_mirror_reflects_everything_back_out():
    model = make_model([
        source_body(power_mW=1.0, coherent=False),
        slab_body("Mirror", "bk7", 0.0, 0.003, mirror=1.0),
        detector_body(x=0.03),
    ])
    result, grids, scene = trace_scene(model, rays=5000)
    rep = result.ledger.report([n for n, _ in scene.sources])
    assert rep["closure_ok"]
    fx = _flux(rep, "Mirror")
    # a perfect mirror: everything that arrives leaves again
    assert np.isclose(fx["out_W"], fx["in_W"], rtol=1e-9)
    assert (rep["by_body_W"].get("Mirror", 0.0)
            + rep["by_surface_W"].get("Mirror", 0.0)) <= 1e-12
