# =============================================================================
# test_import_zemax_coating.py — scripts/tools/import_zemax_coating.py
# (Zemax TABLE coating file -> opticalproperties/coating .mietab + registry
# row converter, P2).
#
# The TABLE/ANGL/WAVE syntax exercised here is corroborated by the Zemax
# community/support knowledge base (accessed 2026-07-16):
#   - community.zemax.com "Beam splitter coating as function of angle"
#     (thread 1261): "TABLE <coating name> / ANGL <angle> / WAVE <wave> Rs
#     Rp Ts Tp Ars Arp Ats Atp ... both angles and wavelengths must be
#     listed in ascending order ... wavelength must be expressed in
#     micrometers ... [phase columns] are optional [omitted = no phase
#     change]."
#   - support.zemax.com "How to model a dichroic beam splitter" (the
#     knowledge-base article the above thread cites as authoritative),
#     which gives the worked example reproduced verbatim in
#     test_parse_documented_swp_example below:
#       TABLE SWP
#       ANGL 45
#       WAVE 0.400 0.0 0.0 1.0 1.0
#       WAVE 0.525 1.0 1.0 0.0 0.0
#     (a shortwave-pass filter at 45 deg AOI: fully transmissive at 400nm,
#     fully reflective at 525nm, no phase columns.)
# =============================================================================
import csv
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
TOOLS = SCRIPTS / "tools"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TOOLS))

import import_zemax_coating as izc                              # noqa: E402
from raytracer.materials import _load_coating_table              # noqa: E402


# ---------------------------------------------------------------------------
# parser: the documented example, verbatim
# ---------------------------------------------------------------------------
SWP_EXAMPLE = (
    "TABLE SWP\n"
    "ANGL 45\n"
    "WAVE 0.400 0.0 0.0 1.0 1.0\n"
    "WAVE 0.525 1.0 1.0 0.0 0.0\n"
)


def test_parse_documented_swp_example():
    name, blocks = izc.parse_zemax_table(SWP_EXAMPLE, ctx="swp")
    assert name == "SWP"
    assert len(blocks) == 1
    b = blocks[0]
    assert b["aoi_deg"] == pytest.approx(45.0)
    assert b["has_phase"] is False
    assert b["lam_um"] == pytest.approx([0.400, 0.525])
    assert b["Rs"] == pytest.approx([0.0, 1.0])
    assert b["Rp"] == pytest.approx([0.0, 1.0])
    assert b["Ts"] == pytest.approx([1.0, 0.0])
    assert b["Tp"] == pytest.approx([1.0, 0.0])


def test_select_block_single_block_no_aoi_needed():
    name, blocks = izc.parse_zemax_table(SWP_EXAMPLE, ctx="swp")
    b = izc.select_block(blocks, None)
    assert b is blocks[0]
    b2 = izc.select_block(blocks, 45.0)
    assert b2 is blocks[0]
    with pytest.raises(izc.ZemaxCoatingError, match="no ANGL"):
        izc.select_block(blocks, 30.0)


# ---------------------------------------------------------------------------
# parser: phase-carrying block (9-number WAVE rows)
# ---------------------------------------------------------------------------
PHASE_EXAMPLE = (
    "! comment lines and blank lines are ignored\n"
    "\n"
    "TABLE bs_test_45\n"
    "ANGL 45\n"
    "WAVE 0.400 0.50 0.48 0.48 0.50 128.14 -52.44 -141.86 -142.44\n"
    "WAVE 0.700 0.50 0.48 0.48 0.50 -171.58 8.60 -81.58 -81.40\n"
)


def test_parse_phase_carrying_block():
    name, blocks = izc.parse_zemax_table(PHASE_EXAMPLE, ctx="bs")
    assert name == "bs_test_45"
    b = blocks[0]
    assert b["has_phase"] is True
    assert b["Ars"] == pytest.approx([128.14, -171.58])
    assert b["Atp"] == pytest.approx([-142.44, -81.40])


def test_multi_angle_file_requires_explicit_aoi():
    text = (
        "TABLE multi\n"
        "ANGL 0\n"
        "WAVE 0.400 0.1 0.1 0.9 0.9\n"
        "WAVE 0.700 0.1 0.1 0.9 0.9\n"
        "ANGL 45\n"
        "WAVE 0.400 0.5 0.4 0.5 0.6\n"
        "WAVE 0.700 0.5 0.4 0.5 0.6\n"
    )
    name, blocks = izc.parse_zemax_table(text, ctx="multi")
    assert len(blocks) == 2
    with pytest.raises(izc.ZemaxCoatingError, match="ANGL blocks"):
        izc.select_block(blocks, None)
    b45 = izc.select_block(blocks, 45.0)
    assert b45["aoi_deg"] == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# malformed input -> hard errors (never silent misparse)
# ---------------------------------------------------------------------------
def test_angl_out_of_order_is_an_error():
    text = ("TABLE bad\nANGL 45\nWAVE 0.4 0.5 0.5 0.5 0.5\n"
            "ANGL 30\nWAVE 0.4 0.5 0.5 0.5 0.5\n")
    with pytest.raises(izc.ZemaxCoatingError, match="ascending"):
        izc.parse_zemax_table(text, ctx="bad")


def test_wave_out_of_order_within_block_is_an_error():
    text = ("TABLE bad\nANGL 45\nWAVE 0.7 0.5 0.5 0.5 0.5\n"
            "WAVE 0.4 0.5 0.5 0.5 0.5\n")
    with pytest.raises(izc.ZemaxCoatingError, match="ascending"):
        izc.parse_zemax_table(text, ctx="bad")


def test_wave_wrong_column_count_is_an_error():
    text = "TABLE bad\nANGL 45\nWAVE 0.4 0.5 0.5 0.5\n"
    with pytest.raises(izc.ZemaxCoatingError, match="5 numbers"):
        izc.parse_zemax_table(text, ctx="bad")


def test_wave_mixed_phase_and_no_phase_rows_is_an_error():
    text = ("TABLE bad\nANGL 45\n"
            "WAVE 0.4 0.5 0.5 0.5 0.5 0 0 0 0\n"
            "WAVE 0.7 0.5 0.5 0.5 0.5\n")
    with pytest.raises(izc.ZemaxCoatingError, match="mix phase"):
        izc.parse_zemax_table(text, ctx="bad")


def test_missing_table_header_is_an_error():
    with pytest.raises(izc.ZemaxCoatingError, match="TABLE"):
        izc.parse_zemax_table("ANGL 45\nWAVE 0.4 0.5 0.5 0.5 0.5\n", ctx="bad")


# ---------------------------------------------------------------------------
# .mietab writer round-trips through materials.py's REAL loader
# ---------------------------------------------------------------------------
def test_write_mietab_round_trips_through_the_real_loader(tmp_path):
    name, blocks = izc.parse_zemax_table(PHASE_EXAMPLE, ctx="bs")
    b = izc.select_block(blocks, None)
    out_path = tmp_path / "bs_test_45.mietab"
    izc.write_mietab(b, out_path)

    loaded = _load_coating_table(out_path, "roundtrip")
    assert loaded["phase_valid"] is True
    assert loaded["lam_um"] == pytest.approx([0.400, 0.700])
    assert loaded["Rs"] == pytest.approx([0.50, 0.50])
    assert loaded["ats_deg"] == pytest.approx([-141.86, -81.58])
    assert loaded["atp_deg"] == pytest.approx([-142.44, -81.40])


def test_write_mietab_no_phase_columns_when_source_has_none(tmp_path):
    name, blocks = izc.parse_zemax_table(SWP_EXAMPLE, ctx="swp")
    b = izc.select_block(blocks, None)
    out_path = tmp_path / "swp.mietab"
    izc.write_mietab(b, out_path)
    with open(out_path, newline="") as fh:
        fieldnames = csv.DictReader(fh).fieldnames
    assert fieldnames == ["wavelength_nm", "Rs", "Rp", "Ts", "Tp"]

    loaded = _load_coating_table(out_path, "roundtrip")
    assert loaded["phase_valid"] is False


# ---------------------------------------------------------------------------
# CLI end-to-end: parse -> write table -> upsert registry row (idempotent)
# ---------------------------------------------------------------------------
def test_cli_main_writes_table_and_upserts_registry(tmp_path):
    src = tmp_path / "bs_test_45.dat"
    src.write_text(PHASE_EXAMPLE)
    registry = tmp_path / "coatings.miecoat"
    registry.write_text("name,layers,table,aoi_deg,reference\r\n")

    rc = izc.main([str(src), "--reference", "unit test fixture",
                  "--out-table", str(tmp_path / "bs_test_45.mietab"),
                  "--merge-into", str(registry)])
    assert rc == 0

    with open(registry, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["name"] == "bs_test_45"
    assert rows[0]["table"] == "bs_test_45.mietab"
    assert rows[0]["aoi_deg"] == "45"
    assert rows[0]["reference"] == "unit test fixture"

    loaded = _load_coating_table(tmp_path / "bs_test_45.mietab", "cli")
    assert loaded["phase_valid"] is True

    # re-running with a different reference UPDATES the row in place
    # (idempotent by-name upsert, same convention as gen_registry_rows.py)
    rc2 = izc.main([str(src), "--reference", "updated citation",
                    "--out-table", str(tmp_path / "bs_test_45.mietab"),
                    "--merge-into", str(registry)])
    assert rc2 == 0
    with open(registry, newline="") as fh:
        rows2 = list(csv.DictReader(fh))
    assert len(rows2) == 1
    assert rows2[0]["reference"] == "updated citation"


def test_cli_main_requires_reference_unless_dry_run(tmp_path, capsys):
    src = tmp_path / "swp.dat"
    src.write_text(SWP_EXAMPLE)
    rc = izc.main([str(src)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--reference is required" in err

    rc_dry = izc.main([str(src), "--dry-run"])
    assert rc_dry == 0
