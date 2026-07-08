"""beadanim tests: pure-numpy segment math (precompute/lerp/cap) and the
AnimationController transport state machine, driven by direct _tick()
calls (the QTimer never runs offscreen -- docs/UI_TESTING.md)."""

import math
import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vtkmodules.vtkIOXML import vtkXMLPolyDataReader  # noqa: E402

from mieworkbench.core.beadanim import (  # noqa: E402
    AnimationController, C_M_S, active_positions, format_sim_time,
    precompute_segments,
)
from mieworkbench.tests.vtk_test_support import write_simple_vtp  # noqa: E402


def _load_polydata(path):
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def _segments(tmp_path, opl, rel=None):
    path = tmp_path / "rays.vtp"
    write_simple_vtp(path, with_rgb=True,
                     rel_power=rel or [1.0] * len(opl), opl=opl)
    return precompute_segments(_load_polydata(path))


# ---------------------------------------------------------------------------
# precompute_segments
# ---------------------------------------------------------------------------
def test_precompute_returns_none_without_opl(tmp_path):
    path = tmp_path / "legacy.vtp"
    write_simple_vtp(path, with_rgb=True, rel_power=[1.0])
    assert precompute_segments(_load_polydata(path)) is None
    assert precompute_segments(None) is None


def test_precompute_extracts_windows_and_geometry(tmp_path):
    seg = _segments(tmp_path, opl=[(0.0, 1.0), (1.0, 2.5)])
    assert seg is not None
    assert np.allclose(seg.t0 * C_M_S, [0.0, 1.0])
    assert np.allclose(seg.t1 * C_M_S, [1.0, 2.5])
    assert seg.t_max == pytest.approx(2.5 / C_M_S)
    # fixture cells run (0,i,0) -> (1,i,0)
    assert np.allclose(seg.p0[0], [0, 0, 0])
    assert np.allclose(seg.p1[1], [1, 1, 0])


def test_precompute_drops_zero_duration_cells(tmp_path):
    seg = _segments(tmp_path, opl=[(0.0, 1.0), (1.0, 1.0)])
    assert len(seg.t0) == 1


# ---------------------------------------------------------------------------
# active_positions
# ---------------------------------------------------------------------------
def test_lerp_positions_and_boundaries(tmp_path):
    seg = _segments(tmp_path, opl=[(0.0, 1.0), (1.0, 2.0)])
    c = C_M_S

    pts, rgb = active_positions(seg, 0.0)          # t=0: bead at source
    assert len(pts) == 1
    assert np.allclose(pts[0], [0, 0, 0])
    assert tuple(rgb[0]) == (255, 255, 0)

    pts, _ = active_positions(seg, 0.5 / c)        # mid first segment
    assert np.allclose(pts[0], [0.5, 0, 0])

    pts, _ = active_positions(seg, 1.0 / c)        # handoff instant:
    assert len(pts) == 1                           # first ended (t<t1),
    assert np.allclose(pts[0], [0.0, 1, 0])        # second starts at f=0

    pts, _ = active_positions(seg, 3.0 / c)        # beyond t_max: empty
    assert len(pts) == 0


def test_glass_bead_lags_by_index(tmp_path):
    """Two same-geometry segments; one 'in glass' has an n=1.5 window.
    At equal clock the glass bead has covered 1/1.5 of the distance."""
    seg = _segments(tmp_path, opl=[(0.0, 1.0), (0.0, 1.5)])
    pts, _ = active_positions(seg, 0.6 / C_M_S)
    assert np.allclose(pts[0][0], 0.6)             # vacuum bead
    assert np.allclose(pts[1][0], 0.4)             # glass bead: 0.6/1.5


def test_ray_cap_bounds_drawn_beads(tmp_path):
    seg = _segments(tmp_path, opl=[(0.0, 1.0)] * 5)
    pts, _ = active_positions(seg, 0.5 / C_M_S)
    assert len(pts) == 5
    pts, _ = active_positions(seg, 0.5 / C_M_S, ray_cap=2)
    assert len(pts) == 2


# ---------------------------------------------------------------------------
# AnimationController (no layer; direct ticks)
# ---------------------------------------------------------------------------
def _controller(tmp_path, qtbot=None):
    ctrl = AnimationController()
    ctrl.apply_settings(enabled=True)
    ctrl.set_segments(_segments(tmp_path, opl=[(0.0, 0.01)]))
    return ctrl


def test_transport_state_machine(tmp_path, qtbot):
    ctrl = _controller(tmp_path)
    assert ctrl.state == "stopped" and ctrl.clock == 0.0

    ctrl.play()
    assert ctrl.state == "playing"
    ctrl.pause()
    assert ctrl.state == "paused"

    ctrl.step()                      # advances exactly one frame
    expected = (ctrl.speed_mm_s / ctrl.fps) / 1000.0 / C_M_S
    assert ctrl.clock == pytest.approx(expected)

    ctrl.stop()
    assert ctrl.state == "stopped" and ctrl.clock == 0.0


def test_clock_wraps_at_t_max(tmp_path, qtbot):
    ctrl = _controller(tmp_path)
    ctrl.apply_settings(speed_mm_s=20.0, fps=1)    # 20 mm per frame
    ctrl.play()
    ctrl._tick()                                   # 20 mm > 10 mm window
    assert ctrl.clock == 0.0                       # looped

    frames = []
    ctrl.frameAdvanced.connect(lambda t, mm: frames.append((t, mm)))
    ctrl.apply_settings(speed_mm_s=2.0, fps=15)
    ctrl._tick()
    assert frames and frames[-1][1] == pytest.approx(2.0 / 15.0)


def test_disabled_or_missing_segments_blocks_transport(tmp_path, qtbot):
    ctrl = AnimationController()
    ctrl.apply_settings(enabled=True)
    ctrl.set_segments(None)                        # legacy overlay
    assert not ctrl.has_segments()
    ctrl.play()
    assert ctrl.state == "stopped"                 # play refused

    ctrl.set_segments(_segments(tmp_path, opl=[(0.0, 0.01)]))
    ctrl.apply_settings(enabled=False)             # animation off
    ctrl.step()
    assert ctrl.clock == 0.0


def test_availability_signal(tmp_path, qtbot):
    ctrl = AnimationController()
    seen = []
    ctrl.availabilityChanged.connect(seen.append)
    ctrl.set_segments(_segments(tmp_path, opl=[(0.0, 0.01)]))
    ctrl.set_segments(None)
    assert seen == [True, False]


# ---------------------------------------------------------------------------
# time formatting
# ---------------------------------------------------------------------------
def test_format_sim_time_auto_units():
    assert format_sim_time(0.0).endswith("fs")
    assert format_sim_time(5e-15) == "5.0 fs"
    assert format_sim_time(3.3e-12).endswith("ps")
    assert format_sim_time(2e-9).endswith("ns")
    assert format_sim_time(4e-6).endswith("µs")
    assert format_sim_time(1.5e-3).endswith("ms")
