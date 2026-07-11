"""core/opticalvalues menu-model oracle tests (pure, no Qt)."""

import math

import pytest

from mieworkbench.core import opticalvalues as ov
from mieworkbench.core import paraxial
from mieworkbench.tests.test_paraxial import (
    _body, _sheet, _tm, _fprop, _sprop, PCX, idx, N_BK7)


def _two_lens():
    bodies = [
        _body("SRC", kind="laser_collimated",
              props={"power": _fprop(5.0), "lambdac": _fprop(633.0)}),
        _body("L1", kind="lens_pcx", material="bk7",
              chain={"mode": "chained", "ref": "SRC", "distance": "30"}),
        _body("L2", kind="lens_pcx", material="bk7",
              chain={"mode": "chained", "ref": "L1", "distance": "40"}),
    ]
    sheets = [_sheet("SRC", {"diameter": 6.0}), _sheet("L1", PCX),
              _sheet("L2", PCX)]
    return _tm(bodies, sheets)


def test_distance_field_offers_prev_bfl_efl_afocal():
    tm = _two_lens()
    entries = ov.value_menu_model(tm, "L2", "distance", index_fn=idx)
    by_var = {e["suggest_var"]: e for e in entries}
    card = paraxial.element_cardinals("lens_pcx", PCX, idx, 633.0)
    assert by_var["bfl_l1"]["value"] == pytest.approx(card["bfl"],
                                                      rel=1e-12)
    assert by_var["efl_l1"]["value"] == pytest.approx(card["efl"],
                                                      rel=1e-12)
    # afocal spacing: rear focus of L1 coincides with front focus of L2
    assert by_var["afocal_l1"]["value"] == pytest.approx(
        card["bfl"] - card["ffl"], rel=1e-12)
    assert all(e["kind"] == "length" for e in entries)


def test_image_distance_after_ref_matches_truncated_system():
    tm = _two_lens()
    entries = ov.value_menu_model(tm, "L2", "distance", index_fn=idx)
    by_var = {e["suggest_var"]: e for e in entries}
    sub = paraxial.system_summary(tm, index_fn=idx, through_element="L1")
    assert by_var["img_l1"]["value"] == pytest.approx(
        sub["image_distance_mm"], rel=1e-12)
    # single upstream lens: image distance == its BFL
    assert by_var["img_l1"]["value"] == pytest.approx(
        by_var["bfl_l1"]["value"], rel=1e-12)


def test_system_entries_present():
    tm = _two_lens()
    entries = ov.value_menu_model(tm, "L2", "distance", index_fn=idx)
    groups = {e["group"] for e in entries}
    assert "System" in groups


def test_angle_field_on_lens_ref_offers_nothing_optical():
    tm = _two_lens()
    entries = ov.value_menu_model(tm, "L2", "tilt_rx", index_fn=idx)
    # lens ref: no grating/prism entries; no length entries leak in
    assert all(e["kind"] == "angle" for e in entries)
    assert entries == []


def test_non_optical_field_returns_empty():
    tm = _two_lens()
    assert ov.value_menu_model(tm, "L2", "rot_order", index_fn=idx) == []
    assert ov.field_kind("mode") is None


def test_sheet_alias_kinds():
    assert ov.field_kind("R_front", is_sheet_alias=True) == "length"
    assert ov.field_kind("ct", is_sheet_alias=True) == "length"
    assert ov.field_kind("aperture", is_sheet_alias=True) == "length"
    assert ov.field_kind("round_flag", is_sheet_alias=True) is None
    assert ov.field_kind("rotation", is_sheet_alias=True) is None


def test_grating_orders():
    bodies = [
        _body("SRC", kind="laser_collimated",
              props={"power": _fprop(5.0), "lambdac": _fprop(633.0)}),
        _body("G", kind="grating_plate", material="bk7",
              chain={"mode": "chained", "ref": "SRC", "distance": "30"},
              props={"grating": _sprop("Face2=600:v:orders=-1..1")}),
        _body("D", kind="detector_plane", material="detector",
              chain={"mode": "chained", "ref": "G", "distance": "50",
                     "fold_deviation": "20"}),
    ]
    sheets = [_sheet("SRC", {"diameter": 6.0}),
              _sheet("G", {"width": 25.0, "thickness": 3.0}),
              _sheet("D", {"width": 30.0, "thickness": 1.0})]
    tm = _tm(bodies, sheets)
    entries = ov.value_menu_model(tm, "D", "fold_deviation", index_fn=idx)
    grat = [e for e in entries if e["group"].startswith("Grating")]
    assert len(grat) == 2  # orders -1 and +1 (0 skipped)
    expected = math.degrees(math.asin(633e-9 * 600e3))
    vals = sorted(e["value"] for e in grat)
    assert vals[1] == pytest.approx(expected, rel=1e-9)
    assert vals[0] == pytest.approx(-expected, rel=1e-9)


def test_prism_min_deviation():
    bodies = [
        _body("SRC", kind="laser_collimated",
              props={"power": _fprop(5.0), "lambdac": _fprop(633.0)}),
        _body("P", kind="prism", material="bk7",
              chain={"mode": "chained", "ref": "SRC", "distance": "30"}),
        _body("D", kind="detector_plane", material="detector",
              chain={"mode": "chained", "ref": "P", "distance": "50"}),
    ]
    sheets = [_sheet("SRC", {"diameter": 6.0}),
              _sheet("P", {"side": 25.0, "height": 25.0, "rotation": 0.0}),
              _sheet("D", {"width": 30.0, "thickness": 1.0})]
    tm = _tm(bodies, sheets)
    entries = ov.value_menu_model(tm, "D", "fold_deviation", index_fn=idx)
    prism = [e for e in entries if e["group"].startswith("Prism")]
    assert len(prism) == 1
    # BK7 (n = 1.51680) equilateral: delta_min = 2 asin(n sin 30) - 60
    expected = 2 * math.degrees(math.asin(N_BK7 * 0.5)) - 60.0
    assert prism[0]["value"] == pytest.approx(expected, rel=1e-9)
    assert expected == pytest.approx(38.65, abs=0.05)


def test_anchored_element_gets_only_system_entries():
    bodies = [
        _body("SRC", kind="laser_collimated",
              props={"power": _fprop(5.0), "lambdac": _fprop(633.0)}),
        _body("L1", kind="lens_pcx", material="bk7",
              chain={"mode": "chained", "ref": "SRC", "distance": "30"}),
        _body("FREE", kind="lens_pcx", material="bk7"),
    ]
    sheets = [_sheet("SRC", {"diameter": 6.0}), _sheet("L1", PCX),
              _sheet("FREE", PCX)]
    tm = _tm(bodies, sheets)
    entries = ov.value_menu_model(tm, "FREE", "distance", index_fn=idx)
    assert entries and all(e["group"] == "System" for e in entries)
