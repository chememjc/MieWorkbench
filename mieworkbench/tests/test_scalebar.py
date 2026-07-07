"""Adaptive scale-bar tests: the pure snap/format/px-per-metre math (no VTK
objects touched -- runs anywhere, no GPU/offscreen concerns) plus a smoke
test that VtkSceneView constructs cleanly offscreen with the scale-bar
actors absent (see vtkview.py's module docstring / is_offscreen() for why
GPU-touching work is gated the same way the orientation-axes widget is).
"""

import math
import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

from mieworkbench.widgets.vtkview import (  # noqa: E402
    VtkSceneView, format_bar_label, nice_bar_length, px_per_metre_parallel,
    px_per_metre_perspective,
)
from mieworkbench.tests.vtk_test_support import make_two_body_scene  # noqa: E402


# ---------------------------------------------------------------------------
# px_per_metre_parallel / px_per_metre_perspective
# ---------------------------------------------------------------------------
def test_px_per_metre_parallel_basic():
    # ParallelScale is half the world-space viewport height (VTK
    # convention): scale=1m, height=1000px -> 500 px/m.
    assert px_per_metre_parallel(1.0, 1000) == pytest.approx(500.0)


def test_px_per_metre_parallel_degenerate_inputs_are_zero():
    assert px_per_metre_parallel(0.0, 1000) == 0.0
    assert px_per_metre_parallel(-1.0, 1000) == 0.0
    assert px_per_metre_parallel(1.0, 0) == 0.0


def test_px_per_metre_perspective_known_angle():
    # pick a view_angle whose half-angle has a clean tangent:
    # half_angle = atan(0.5) -> full angle in degrees.
    half_angle_deg = math.degrees(math.atan(0.5))
    view_angle_deg = 2.0 * half_angle_deg
    # world_height = 2 * distance * tan(half_angle) = 2*1*0.5 = 1m
    ppm = px_per_metre_perspective(1.0, view_angle_deg, 1000)
    assert ppm == pytest.approx(1000.0)


def test_px_per_metre_perspective_degenerate_inputs_are_zero():
    assert px_per_metre_perspective(0.0, 30.0, 1000) == 0.0
    assert px_per_metre_perspective(-1.0, 30.0, 1000) == 0.0
    assert px_per_metre_perspective(1.0, 30.0, 0) == 0.0


def test_px_per_metre_perspective_larger_distance_means_fewer_px_per_metre():
    near = px_per_metre_perspective(1.0, 30.0, 1000)
    far = px_per_metre_perspective(10.0, 30.0, 1000)
    assert far < near


# ---------------------------------------------------------------------------
# nice_bar_length
# ---------------------------------------------------------------------------
def test_nice_bar_length_snaps_into_1_2_5_sequence():
    # px_per_m and viewport chosen so the "ideal" (25%-of-viewport) length
    # is right around 1mm; the snapped result must be one of 1-2-5 * 10^n.
    px_per_m = 200_000.0   # 200 px/mm
    viewport_px = 1000
    length_m = nice_bar_length(px_per_m, viewport_px)
    assert length_m > 0
    mantissa = length_m / (10.0 ** math.floor(math.log10(length_m)))
    # mantissa should be close to 1, 2, or 5 (allow float slop)
    assert any(abs(mantissa - m) < 1e-6 for m in (1.0, 2.0, 5.0))


def test_nice_bar_length_lands_in_requested_fraction_band_when_possible():
    # A "friendly" px_per_m/viewport combo where an exact 1-2-5 candidate
    # falls inside [0.2, 0.3] of the viewport width.
    px_per_m = 100_000.0    # 100 px/mm
    viewport_px = 1000      # 1000 px wide
    length_m = nice_bar_length(px_per_m, viewport_px, frac_lo=0.2, frac_hi=0.3)
    frac = length_m * px_per_m / viewport_px
    assert 0.2 - 1e-9 <= frac <= 0.3 + 1e-9


def test_nice_bar_length_respects_custom_frac_bounds():
    px_per_m = 100_000.0
    viewport_px = 1000
    length_wide = nice_bar_length(px_per_m, viewport_px, frac_lo=0.4, frac_hi=0.5)
    length_narrow = nice_bar_length(px_per_m, viewport_px, frac_lo=0.05, frac_hi=0.1)
    assert length_wide > length_narrow


def test_nice_bar_length_degenerate_inputs_are_zero():
    assert nice_bar_length(0.0, 1000) == 0.0
    assert nice_bar_length(-5.0, 1000) == 0.0
    assert nice_bar_length(100.0, 0) == 0.0
    assert nice_bar_length(100.0, -1) == 0.0


def test_nice_bar_length_grows_as_camera_zooms_out():
    # Zooming out means fewer px/m -- each pixel spans more world-space, so
    # to keep occupying the same fraction of the viewport the snapped bar
    # must represent a LARGER physical length.
    viewport_px = 1000
    near = nice_bar_length(500_000.0, viewport_px)   # zoomed in: 500 px/mm
    far = nice_bar_length(500.0, viewport_px)        # zoomed way out
    assert far > near


# ---------------------------------------------------------------------------
# format_bar_label
# ---------------------------------------------------------------------------
def test_format_bar_label_micrometres_below_threshold():
    assert format_bar_label(500e-6) == "500 µm"
    assert format_bar_label(1e-6) == "1 µm"
    assert format_bar_label(0.001) == "1000 µm"   # 1mm < 2.5mm -> um


def test_format_bar_label_millimetres_at_and_above_threshold():
    assert format_bar_label(2.5e-3) == "2.5 mm"
    assert format_bar_label(5e-3) == "5 mm"
    assert format_bar_label(0.02) == "20 mm"
    assert format_bar_label(0.05) == "50 mm"


def test_format_bar_label_no_trailing_float_noise():
    # 5e-4 m == 0.5mm -> 500 um, must not render as "500.00000001 µm"
    label = format_bar_label(5e-4)
    assert label == "500 µm"
    label2 = format_bar_label(2e-2)
    assert label2 == "20 mm"


# ---------------------------------------------------------------------------
# offscreen smoke test
# ---------------------------------------------------------------------------
def test_scale_bar_absent_offscreen_and_toggle_does_not_crash(qtbot, tmp_path):
    view = VtkSceneView()
    qtbot.addWidget(view)

    # Gated exactly like the orientation-axes widget: never built offscreen.
    assert view._scalebar_line_actor is None
    assert view._scalebar_text_actor is None
    assert view._scale_bar_visible is True   # default-visible toggle state

    # Toggling, loading a scene, and moving the camera must not crash even
    # though the actors don't exist offscreen.
    view.set_scale_bar_visible(False)
    assert view._scale_bar_visible is False
    view.set_scale_bar_visible(True)
    assert view._scale_bar_visible is True

    structure, faces = make_two_body_scene(tmp_path)
    view.load_bodies(faces, structure)
    view.fit_camera()
    view.view_along("+x")

    assert view._scalebar_line_actor is None
    assert view._scalebar_text_actor is None
