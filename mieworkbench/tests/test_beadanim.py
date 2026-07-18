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
    ALPHA_FLOOR, AnimationController, C_M_S, SegmentSet, _power_alpha,
    active_positions, compute_leading_flags, format_sim_time,
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

    pts, rgb, alpha = active_positions(seg, 0.0)   # t=0: bead at source
    assert len(pts) == 1
    assert np.allclose(pts[0], [0, 0, 0])
    assert tuple(rgb[0]) == (255, 255, 0)
    assert alpha is None                           # "off" mode: no alpha

    pts, _, _ = active_positions(seg, 0.5 / c)     # mid first segment
    assert np.allclose(pts[0], [0.5, 0, 0])

    pts, _, _ = active_positions(seg, 1.0 / c)     # handoff instant:
    assert len(pts) == 1                           # first ended (t<t1),
    assert np.allclose(pts[0], [0.0, 1, 0])        # second starts at f=0

    pts, _, _ = active_positions(seg, 3.0 / c)     # beyond t_max: empty
    assert len(pts) == 0


def test_glass_bead_lags_by_index(tmp_path):
    """Two same-geometry segments; one 'in glass' has an n=1.5 window.
    At equal clock the glass bead has covered 1/1.5 of the distance."""
    seg = _segments(tmp_path, opl=[(0.0, 1.0), (0.0, 1.5)])
    pts, _, _ = active_positions(seg, 0.6 / C_M_S)
    assert np.allclose(pts[0][0], 0.6)             # vacuum bead
    assert np.allclose(pts[1][0], 0.4)             # glass bead: 0.6/1.5


def test_ray_cap_bounds_drawn_beads(tmp_path):
    seg = _segments(tmp_path, opl=[(0.0, 1.0)] * 5)
    pts, _, _ = active_positions(seg, 0.5 / C_M_S)
    assert len(pts) == 5
    pts, _, _ = active_positions(seg, 0.5 / C_M_S, ray_cap=2)
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
# bead opacity: leading-wavefront lineage reconstruction
# ---------------------------------------------------------------------------
def _lineage(segments):
    """segments: list of (p0, p1, opl0, opl1, power); return leading flags
    in the same order (compute_leading_flags is order-independent)."""
    p0 = np.array([s[0] for s in segments], dtype=float)
    p1 = np.array([s[1] for s in segments], dtype=float)
    opl0 = np.array([s[2] for s in segments], dtype=float)
    opl1 = np.array([s[3] for s in segments], dtype=float)
    power = np.array([s[4] for s in segments], dtype=float)
    return compute_leading_flags(p0, p1, opl0, opl1, power)


def test_leading_root_and_5050_split_both_arms():
    # root -> 50/50 split: both children stay leading
    lead = _lineage([
        ((0, 0, 0), (1, 0, 0), 0.0, 1.0, 1.0),     # root
        ((1, 0, 0), (2, 1, 0), 1.0, 2.0, 0.5),     # arm A
        ((1, 0, 0), (2, -1, 0), 1.0, 2.0, 0.5),    # arm B
    ])
    assert list(lead) == [True, True, True]


def test_leading_9010_split_only_bright_arm():
    lead = _lineage([
        ((0, 0, 0), (1, 0, 0), 0.0, 1.0, 1.0),     # root
        ((1, 0, 0), (2, 1, 0), 1.0, 2.0, 0.9),     # bright
        ((1, 0, 0), (2, -1, 0), 1.0, 2.0, 0.1),    # ghost
    ])
    assert list(lead) == [True, True, False]


def test_leading_second_generation_split_inherits():
    # root -> (leading arm) -> 50/50 grandchildren both leading
    lead = _lineage([
        ((0, 0, 0), (1, 0, 0), 0.0, 1.0, 1.0),     # root
        ((1, 0, 0), (2, 1, 0), 1.0, 2.0, 0.5),     # leading arm
        ((2, 1, 0), (3, 2, 0), 2.0, 3.0, 0.25),    # grandchild A
        ((2, 1, 0), (3, 0, 0), 2.0, 3.0, 0.25),    # grandchild B
    ])
    assert list(lead) == [True, True, True, True]


def test_leading_ghost_children_never_leading():
    # a non-leading ghost's single child does NOT become leading
    lead = _lineage([
        ((0, 0, 0), (1, 0, 0), 0.0, 1.0, 1.0),     # root
        ((1, 0, 0), (2, 1, 0), 1.0, 2.0, 0.9),     # bright (leading)
        ((1, 0, 0), (2, -1, 0), 1.0, 2.0, 0.1),    # ghost (not leading)
        ((2, -1, 0), (3, -2, 0), 2.0, 3.0, 0.1),   # ghost's child
    ])
    assert list(lead) == [True, True, False, False]


def test_leading_single_continuation_inherits():
    # a straight ray (root then one continuation) stays leading throughout
    lead = _lineage([
        ((0, 0, 0), (1, 0, 0), 0.0, 1.0, 1.0),
        ((1, 0, 0), (2, 0, 0), 1.0, 2.0, 1.0),
    ])
    assert list(lead) == [True, True]


# ---------------------------------------------------------------------------
# bead opacity: log-dB alpha map
# ---------------------------------------------------------------------------
def test_power_alpha_map():
    power = np.array([1.0, 10.0, 100.0, 1000.0])   # Pmax = 1000
    leading = np.zeros(4, dtype=bool)
    alpha = _power_alpha(power, leading, range_db=30.0)
    assert alpha[3] == pytest.approx(1.0)          # P == Pmax
    assert alpha[2] == pytest.approx(1.0 - 10.0 / 30.0)   # -10 dB
    assert alpha[1] == pytest.approx(1.0 - 20.0 / 30.0)   # -20 dB
    assert alpha[0] == pytest.approx(ALPHA_FLOOR)         # -30 dB -> floor


def test_power_alpha_floor_and_nonpositive():
    power = np.array([1.0, 1e-9, 0.0, -5.0])       # Pmax = 1
    alpha = _power_alpha(power, np.zeros(4, bool), range_db=30.0)
    assert alpha[0] == pytest.approx(1.0)
    assert np.all(alpha[1:] == pytest.approx(ALPHA_FLOOR))  # deep/zero/neg


def test_power_alpha_leading_pinned_to_one():
    power = np.array([1000.0, 1e-6])               # second bead is minuscule
    leading = np.array([False, True])
    alpha = _power_alpha(power, leading, range_db=30.0)
    assert alpha[1] == pytest.approx(1.0)          # leading overrides floor


# ---------------------------------------------------------------------------
# bead opacity: active_positions integration (power mode, cap, fallback)
# ---------------------------------------------------------------------------
def _power_segset(p0, p1, opl0, opl1, power, leading, source_id=None):
    p0 = np.asarray(p0, float)
    n = len(p0)
    t0 = np.asarray(opl0, float) / C_M_S
    t1 = np.asarray(opl1, float) / C_M_S
    return SegmentSet(
        p0=p0, p1=np.asarray(p1, float), t0=t0, t1=t1,
        rgb=np.full((n, 3), 200, np.uint8),
        source_id=(np.zeros(n, np.int64) if source_id is None
                   else np.asarray(source_id, np.int64)),
        t_max=float(t1.max()),
        power=np.asarray(power, float),
        leading=np.asarray(leading, bool))


def test_off_mode_ignores_power_and_returns_none_alpha():
    seg = _power_segset(
        p0=[(0, i, 0) for i in range(4)], p1=[(1, i, 0) for i in range(4)],
        opl0=[0.0] * 4, opl1=[1.0] * 4, power=[1, 2, 3, 4],
        leading=[True, False, False, True])
    pts, rgb, alpha = active_positions(seg, 0.5 / C_M_S, ray_cap=2)
    assert alpha is None                           # off mode: no alpha
    assert len(pts) == 2 and len(rgb) == 2         # first-N cap preserved


def test_power_mode_alpha_and_cap_keeps_brightest_and_leading():
    # four active beads, cap=2: leading (idx0) always kept, plus brightest
    # non-leading (idx3, power 4). idx1/idx2 dropped.
    seg = _power_segset(
        p0=[(0, i, 0) for i in range(4)], p1=[(1, i, 0) for i in range(4)],
        opl0=[0.0] * 4, opl1=[1.0] * 4, power=[1, 2, 3, 4],
        leading=[True, False, False, False])
    pts, rgb, alpha = active_positions(
        seg, 0.5 / C_M_S, ray_cap=2, opacity_mode="power")
    assert len(pts) == 2 and alpha is not None
    # leading bead alpha pinned to 1; brightest bead is Pmax -> alpha 1
    assert np.allclose(np.sort(alpha), [1.0, 1.0])


def test_power_mode_cap_always_keeps_all_leading():
    # three leading beads, cap=1: all leading kept despite the cap
    seg = _power_segset(
        p0=[(0, i, 0) for i in range(3)], p1=[(1, i, 0) for i in range(3)],
        opl0=[0.0] * 3, opl1=[1.0] * 3, power=[1, 2, 3],
        leading=[True, True, True])
    pts, _, alpha = active_positions(
        seg, 0.5 / C_M_S, ray_cap=1, opacity_mode="power")
    assert len(pts) == 3
    assert np.allclose(alpha, 1.0)


def test_power_mode_falls_back_on_legacy_segments(tmp_path):
    # legacy overlay (write_simple_vtp writes no `power` array)
    seg = _segments(tmp_path, opl=[(0.0, 1.0), (0.0, 1.0)])
    assert seg.power is None and seg.leading is None
    pts, _, alpha = active_positions(
        seg, 0.5 / C_M_S, opacity_mode="power")
    assert alpha is None and len(pts) == 2         # graceful opaque fallback


def test_precompute_leaves_power_none_without_power_array(tmp_path):
    seg = _segments(tmp_path, opl=[(0.0, 1.0)])
    assert seg is not None
    assert seg.power is None and seg.leading is None


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
