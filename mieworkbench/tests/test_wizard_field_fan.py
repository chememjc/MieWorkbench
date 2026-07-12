"""Field-angle fan wizard (pulsed-optics P9): core.wizards.design_field_fan
math (arc/plane spacing, aim-at-pivot orientation, overlap auto-grow, the
demo_curved_focal_surface literal-compatibility contract) plus an
offscreen smoke test of the FieldFanDialog page (following
test_wizard_dialog_zoom_pair.py — no .exec() anywhere, no unguarded
modals)."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core import wizards                        # noqa: E402
from mieworkbench.panes.wizard_dialog import FieldFanDialog  # noqa: E402


# ---------------------------------------------------------------------------
# design_field_fan math
# ---------------------------------------------------------------------------
def _rot_z(deg):
    half = math.radians(deg) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def test_arc_spacing_positions_and_aim():
    R = 80.0
    fan = wizards.design_field_fan([0.0, 10.0, -10.0],
                                   pivot_mm=(5.0, -2.0, 1.0),
                                   radius_mm=R, aperture_mm=4.0)
    assert fan["spacing"] == "arc"
    assert fan["radius_mm"] == R and fan["note"] is None
    for s in fan["sources"]:
        # arc spacing: every source exactly R from the pivot
        d2 = math.dist(s["pos_mm"], fan["pivot_mm"])
        assert d2 == pytest.approx(R, abs=1e-9)
        # aimed at the pivot: pos + R*dir == pivot
        for i in range(3):
            assert s["pos_mm"][i] + R * s["dir"][i] == \
                pytest.approx(fan["pivot_mm"][i], abs=1e-9)
        # the field_angle_deg prop rides along
        assert s["props"]["field_angle_deg"] == s["angle_deg"]


def test_n_theta_max_form_spans_symmetric_angles():
    fan = wizards.design_field_fan(5, theta_max_deg=20.0)
    assert [s["angle_deg"] for s in fan["sources"]] == \
        [-20.0, -10.0, 0.0, 10.0, 20.0]
    single = wizards.design_field_fan(1)
    assert [s["angle_deg"] for s in single["sources"]] == [0.0]


def test_plane_spacing_reproduces_demo_curved_focal_surface_literals():
    """The make_demos.demo_curved_focal_surface hand fan, pre-wizard:
    Axis (-40, 0, 0), FieldP (-40, -40*tan16, 0) rot +16, FieldM
    mirrored — design_field_fan(spacing='plane') must reproduce those
    literals EXACTLY (the demo now calls it; the committed placement
    baselines depend on it)."""
    theta = 16.0
    y_off = 40.0 * math.tan(math.radians(theta))
    fan = wizards.design_field_fan([0.0, theta, -theta],
                                   pivot_mm=(0.0, 0.0, 0.0),
                                   radius_mm=40.0, spacing="plane",
                                   aperture_mm=8.0)
    expected = [((-40.0, 0.0, 0.0), _rot_z(0.0)),
                ((-40.0, -y_off, 0.0), _rot_z(theta)),
                ((-40.0, y_off, 0.0), _rot_z(-theta))]
    assert fan["note"] is None
    for s, (pos, quat) in zip(fan["sources"], expected):
        assert max(abs(a - b) for a, b in zip(s["pos_mm"], pos)) < 1e-12
        assert max(abs(a - b) for a, b in zip(s["quat"], quat)) < 1e-12


def test_xz_plane_fan_aims_at_pivot():
    fan = wizards.design_field_fan([0.0, 12.0], plane="xz",
                                   radius_mm=50.0, aperture_mm=2.0)
    s = fan["sources"][1]
    assert s["pos_mm"][1] == pytest.approx(0.0, abs=1e-12)   # y untouched
    assert s["dir"][2] == pytest.approx(
        math.sin(math.radians(12.0)), abs=1e-12)
    for i in range(3):
        assert s["pos_mm"][i] + 50.0 * s["dir"][i] == \
            pytest.approx(0.0, abs=1e-9)


def test_overlap_guard_auto_grows_radius_with_note():
    # 5 sources over +/-10 deg on R=20 with 8 mm bodies cannot fit
    fan = wizards.design_field_fan(5, theta_max_deg=10.0, radius_mm=20.0,
                                   aperture_mm=8.0)
    assert fan["radius_mm"] > fan["radius_requested_mm"]
    assert fan["note"] is not None and "auto-grown" in fan["note"]
    # at the grown radius every adjacent pair clears the bounding diameter
    pts = [s["pos_mm"] for s in sorted(fan["sources"],
                                       key=lambda s: s["angle_deg"])]
    for a, b in zip(pts, pts[1:]):
        assert math.dist(a, b) >= 8.0


def test_default_aperture_from_primitive_metadata():
    # laser_collimated ships a 10 mm 'diameter' default; the guard uses it
    fan = wizards.design_field_fan(2, theta_max_deg=5.0, radius_mm=500.0)
    assert fan["source_diameter_mm"] == pytest.approx(10.0)


def test_field_fan_input_validation():
    with pytest.raises(ValueError):
        wizards.design_field_fan(3)                 # N form needs theta_max
    with pytest.raises(ValueError):
        wizards.design_field_fan([0.0, 0.0])        # duplicate angles
    with pytest.raises(ValueError):
        wizards.design_field_fan([0.0, 95.0])       # |theta| >= 90
    with pytest.raises(ValueError):
        wizards.design_field_fan([])                # empty list
    with pytest.raises(ValueError):
        wizards.design_field_fan(2, theta_max_deg=5.0, radius_mm=0.0)
    with pytest.raises(ValueError):
        wizards.design_field_fan(2, theta_max_deg=5.0, plane="yz")
    with pytest.raises(ValueError):
        wizards.design_field_fan(2, theta_max_deg=5.0, spacing="grid")


# ---------------------------------------------------------------------------
# FieldFanDialog smoke (offscreen)
# ---------------------------------------------------------------------------
def test_field_fan_dialog_computes_and_fills_summary(qtbot):
    dlg = FieldFanDialog(source_kinds=[("laser_collimated",
                                        "Collimated laser")])
    qtbot.addWidget(dlg)
    dlg.n_spin.setValue(3)
    dlg.theta_edit.setText("16")
    dlg.pivot_edit.setText("0, 0, 0")
    dlg.radius_edit.setText("40")
    dlg.aperture_edit.setText("8")
    assert dlg._compute() is True

    expected = wizards.design_field_fan(
        3, theta_max_deg=16.0, pivot_mm=(0.0, 0.0, 0.0), radius_mm=40.0,
        source_kind="laser_collimated", aperture_mm=8.0)
    got = dlg.result()
    assert got is not None
    assert [s["angle_deg"] for s in got["sources"]] == \
        [s["angle_deg"] for s in expected["sources"]]
    assert got["sources"][0]["pos_mm"] == \
        pytest.approx(expected["sources"][0]["pos_mm"])
    assert dlg.aperture_mm() == 8.0
    assert "3 source(s)" in dlg.result_label.text()


def test_field_fan_dialog_explicit_angle_list_wins(qtbot):
    dlg = FieldFanDialog()
    qtbot.addWidget(dlg)
    dlg.angles_edit.setText("0, 8, 16")
    dlg.radius_edit.setText("100")
    assert dlg._compute() is True
    assert [s["angle_deg"] for s in dlg.result()["sources"]] == \
        [0.0, 8.0, 16.0]


def test_field_fan_dialog_bad_input_shows_message_not_crash(qtbot):
    dlg = FieldFanDialog()
    qtbot.addWidget(dlg)
    dlg.pivot_edit.setText("1, 2")          # needs three coordinates
    assert dlg._compute() is False
    assert dlg.result() is None
    assert dlg.result_label.text()          # explanatory text, not blank
    # accept path never raises / never opens a modal on bad input
    dlg._accept_with_compute()
    assert dlg.result() is None
