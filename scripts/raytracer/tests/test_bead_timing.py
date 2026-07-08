# =============================================================================
# test_bead_timing.py — the opl0/opl1 viz columns (tracer-bead animation).
#
# Gates:
#   1. opl1 − opl0 = n·L on a traversed leg (index-correct segment
#      duration, t = opl/c).
#   2. Time continuity at a split: a child's first-segment opl0 equals
#      its parent's opl1 at the interface (beads hand off seamlessly;
#      reflected + transmitted children spawn at the same instant).
#   3. Escaped rays get a synthetic opl1 = opl0 + n·0.25 matching the
#      drawn 0.25 m stub — WITHOUT mutating batch.opl (the coherence
#      path): every recorded window is strictly positive.
#
# Run: /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_bead_timing.py -v
# =============================================================================
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

from raytracer.vtkexport import write_vtp_polylines      # noqa: E402

from .scenehelpers import (make_model, source_body, slab_body,  # noqa: E402
                           detector_body, trace_scene)


X_SRC, X_SLAB0, X_SLAB1 = -0.02, -0.005, 0.005
N_BK7_633 = 1.5151       # BK7 phase index near 633 nm (loose tolerance)


def _traced_viz():
    model = make_model([
        source_body("Src", x=X_SRC, half=0.001, power_mW=1.0,
                    lambdac_nm=633.0),
        slab_body("Slab", "BK7", X_SLAB0, X_SLAB1),
        detector_body("Det", x=0.03),
    ])
    result, _grids, _scene = trace_scene(model, rays=2000, seed=7)
    return result.viz.as_array()


def test_opl_window_is_index_times_length():
    viz = _traced_viz()
    assert viz.shape[1] == 13
    opl0, opl1 = viz[:, 11], viz[:, 12]
    assert np.all(opl1 > opl0)           # every window strictly positive

    length = np.linalg.norm(viz[:, 6:9] - viz[:, 3:6], axis=1)
    n_seg = (opl1 - opl0) / length

    # source -> slab front: ambient AIR leg (n ≈ 1.000272 at 633 nm --
    # the registry's air is real air, not vacuum)
    at_source = np.isclose(viz[:, 3], X_SRC, atol=1e-9)
    assert np.allclose(n_seg[at_source], 1.0003, atol=5e-4)
    assert np.allclose(opl0[at_source], 0.0, atol=1e-12)   # t = 0 at source

    # inside the slab (start at front face, moving +x): n = n_BK7
    inside = (np.isclose(viz[:, 3], X_SLAB0, atol=1e-9)
              & (viz[:, 6] > viz[:, 3]) & ~at_source)
    assert np.any(inside)
    assert np.allclose(n_seg[inside], N_BK7_633, atol=5e-3)


def test_child_opl_continuous_at_split():
    viz = _traced_viz()
    opl0, opl1 = viz[:, 11], viz[:, 12]
    # the air leg reaching the slab front ends with opl1 = 0.015 m
    # (15 mm at n=1); every child segment starting there (reflected AND
    # transmitted) must begin its window at exactly that value
    at_source = np.isclose(viz[:, 3], X_SRC, atol=1e-9)
    arrive = at_source & np.isclose(viz[:, 6], X_SLAB0, atol=1e-9)
    assert np.any(arrive)
    t_split = np.unique(np.round(opl1[arrive], 12))
    assert len(t_split) == 1
    children = np.isclose(viz[:, 3], X_SLAB0, atol=1e-9) & ~at_source
    gen1 = children & np.isclose(opl0, t_split[0], atol=1e-12)
    assert np.any(gen1)
    # both directions spawn at the split instant: one child continues +x
    # (transmitted), one goes back -x (reflected)
    assert np.any(viz[gen1, 6] > X_SLAB0)
    assert np.any(viz[gen1, 6] < X_SLAB0)


def test_escape_stub_gets_synthetic_window():
    # source only -- every ray escapes immediately past the detector-less
    # void; without the synthetic opl1 these stubs would have zero
    # duration and beads would never move in preview scenes
    model = make_model([
        source_body("Src", x=X_SRC, half=0.001, power_mW=1.0,
                    lambdac_nm=633.0),
        detector_body("Det", x=0.03),
    ])
    result, _grids, _scene = trace_scene(model, rays=500, seed=3)
    viz = result.viz.as_array()
    opl0, opl1 = viz[:, 11], viz[:, 12]
    assert np.all(opl1 > opl0)
    # the escaped stub is drawn 0.25 m long in ambient air: window
    # 0.25·n_air (registry air n ≈ 1.000272, not vacuum)
    escaped = ~np.isclose(viz[:, 6], 0.03, atol=1e-9)
    assert np.any(escaped)
    assert np.allclose((opl1 - opl0)[escaped], 0.25, rtol=5e-4)


def test_vtp_export_opl_cell_arrays(tmp_path):
    rows = np.zeros((2, 13))
    rows[:, 1] = 633e-9
    rows[:, 2] = 1e-3
    rows[:, 3:9] = np.arange(6) * 1e-3
    rows[:, 10] = 1.0
    rows[:, 11] = [0.0, 1e-3]
    rows[:, 12] = [1e-3, 2e-3]
    out = tmp_path / "rays.vtp"
    write_vtp_polylines(out, rows)
    text = out.read_text()
    assert 'Name="opl0"' in text and 'Name="opl1"' in text
    assert 'Scalars="rgb"' in text
    # legacy 11-col input still accepted, no opl arrays
    out11 = tmp_path / "rays11.vtp"
    write_vtp_polylines(out11, rows[:, :11])
    assert "opl0" not in out11.read_text()
