# =============================================================================
# test_primitivelib.py -- pytest suite for scripts/primitivelib.py
# (diameter/width param rename, round/rect shape support, legacy-alias
# migration in read_params).
#
# Two tiers:
#   * pure-python checks of the PRIMITIVES registry / LEGACY_ALIASES table --
#     these import primitivelib without FreeCAD (metadata-only import path)
#     and run under the plain engine pytest invocation.
#   * FreeCAD-gated checks (round_flag geometry + rebuild_element + legacy
#     read_params fallback) -- these shell out to the real FreeCAD AppImage
#     via _primitivelib_fc_probe.py, same pattern as
#     mieworkbench/tests/test_fcserver_integration.py's extract_model_json.
#     Gated behind MIEWB_RUN_FREECAD=1 (same convention used project-wide)
#     since they need /home3/freecad/FreeCAD.AppImage.
#
# Run:
#   /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_primitivelib.py -q
#   MIEWB_RUN_FREECAD=1 /home3/optics/env/bin/python -m pytest \
#       scripts/raytracer/tests/test_primitivelib.py -q   # + FreeCAD checks
# =============================================================================
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import primitivelib as pl  # noqa: E402  (metadata-only import; no FreeCAD)

RUN_FREECAD = os.environ.get("MIEWB_RUN_FREECAD") == "1"
freecad_only = pytest.mark.skipif(
    not RUN_FREECAD,
    reason="set MIEWB_RUN_FREECAD=1 to run FreeCAD-backed primitivelib tests")

FREECAD_APPIMAGE = os.environ.get("MIEWB_FREECAD", "/home3/freecad/FreeCAD.AppImage")
PROBE_SCRIPT = Path(__file__).resolve().parent / "_primitivelib_fc_probe.py"

# -- kinds affected by each rename (per the requirements) -------------------
DIAMETER_KINDS = ["laser_collimated", "laser_divergent", "source_broadband"]
WIDTH_KINDS = ["detector_plane", "mirror_flat", "window", "polarizer_plate",
               "waveplate", "filter_plate", "grating_plate"]
ROUND_FLAG_DEFAULTS = {
    "laser_collimated": 1, "laser_divergent": 1, "source_broadband": 1,
    "detector_plane": 0, "mirror_flat": 0, "window": 1,
    "polarizer_plate": 1, "waveplate": 1, "filter_plate": 1,
    "grating_plate": 0,
}
NEW_DEFAULTS = {
    ("laser_collimated", "diameter"): 10.0,
    ("laser_divergent", "diameter"): 10.0,
    ("source_broadband", "diameter"): 10.0,
    ("detector_plane", "width"): 30.0,
    ("mirror_flat", "width"): 25.0,
    ("window", "width"): 25.0,
    ("polarizer_plate", "width"): 20.0,
    ("waveplate", "width"): 16.0,
    ("filter_plate", "width"): 25.0,
    ("grating_plate", "width"): 25.0,
}


# ---------------------------------------------------------------------------
# Pure-python: registry renames
# ---------------------------------------------------------------------------
def test_diameter_kinds_renamed_from_radius():
    for kind in DIAMETER_KINDS:
        params = pl.PRIMITIVES[kind]["params"]
        assert "radius" not in params, kind
        assert "diameter" in params, kind


def test_width_kinds_renamed_from_half():
    for kind in WIDTH_KINDS:
        params = pl.PRIMITIVES[kind]["params"]
        assert "half" not in params, kind
        assert "width" in params, kind


def test_all_renamed_kinds_have_round_flag():
    for kind in DIAMETER_KINDS + WIDTH_KINDS:
        assert "round_flag" in pl.PRIMITIVES[kind]["params"], kind


@pytest.mark.parametrize("kind,expected", sorted(ROUND_FLAG_DEFAULTS.items()))
def test_round_flag_default_per_kind(kind, expected):
    assert pl.PRIMITIVES[kind]["params"]["round_flag"]["default"] == expected


@pytest.mark.parametrize("kind_alias,expected", sorted(NEW_DEFAULTS.items()))
def test_renamed_value_doubled_from_legacy_default(kind_alias, expected):
    kind, alias = kind_alias
    assert pl.PRIMITIVES[kind]["params"][alias]["default"] == expected


def test_lens_curvature_and_aperture_params_unchanged():
    # curvature radii are physics and 'aperture' is already diametral --
    # neither should be touched by this rename.
    lens_pcx = pl.PRIMITIVES["lens_pcx"]["params"]
    assert set(lens_pcx) == {"R_front", "ct", "aperture"}
    assert lens_pcx["aperture"]["default"] == 20.0
    assert lens_pcx["R_front"]["default"] == 25.0
    assert "aperture" in pl.PRIMITIVES["lens_dcx"]["params"]


def test_no_param_alias_looks_like_a_cell_address():
    cell_re = re.compile(r"^[A-Za-z]{1,2}[0-9]{1,4}$")
    for kind, spec in pl.PRIMITIVES.items():
        for alias in spec["params"]:
            assert not cell_re.match(alias), "%s.%s looks like a cell " \
                "address" % (kind, alias)


def test_legacy_aliases_table():
    assert pl.LEGACY_ALIASES["diameter"] == ("radius", 2.0)
    assert pl.LEGACY_ALIASES["width"] == ("half", 2.0)


def test_shared_plate_builder_signature():
    # _build_plate is pure python (no FreeCAD needed to introspect it); the
    # builder-mapping consistency (every WIDTH_KINDS primitive routed
    # through it) is exercised for real in the FreeCAD-gated rebuild
    # round-trip below (rebuild_element calls builders()[kind] internally).
    import inspect
    sig = inspect.signature(pl._build_plate)
    assert list(sig.parameters) == \
        ["doc", "group", "width_mm", "thickness_mm", "round_flag", "name"]


def test_diameter_and_width_help_documents_round_rect():
    for kind in DIAMETER_KINDS:
        help_text = pl.PRIMITIVES[kind]["params"]["diameter"]["help"]
        assert "circular" in help_text and "rectangular" in help_text
    for kind in WIDTH_KINDS:
        help_text = pl.PRIMITIVES[kind]["params"]["width"]["help"]
        assert "circular" in help_text and "rectangular" in help_text


# ---------------------------------------------------------------------------
# FreeCAD-gated: real geometry + rebuild_element round-trip
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fc_probe_result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("primlib_fc_probe")
    out = tmp / "probe.json"
    subprocess.run(
        [FREECAD_APPIMAGE, "-c", str(PROBE_SCRIPT), "--", "--out", str(out)],
        stdin=subprocess.DEVNULL, check=True, capture_output=True, text=True)
    with open(out) as fh:
        return json.load(fh)


@freecad_only
def test_read_params_legacy_fallback(fc_probe_result):
    """A pre-rename sheet carrying only 'radius' (no 'diameter', no
    'round_flag') must read back as diameter = radius * 2, round_flag =
    its spec default."""
    legacy = fc_probe_result["legacy_fallback"]
    assert legacy["diameter"] == pytest.approx(10.0)   # radius=5 -> 10.0
    assert legacy["round_flag"] == 1                    # spec default
    assert legacy["length"] == pytest.approx(12.0)       # alias present as-is


@freecad_only
def test_window_round_vs_rect_face_counts(fc_probe_result):
    """primitives/window.FCStd (round_flag=1 default) is a cylinder (3
    faces); flipping round_flag to 0 and calling rebuild_element must turn
    it into a box (6 faces)."""
    rt = fc_probe_result["rebuild_roundtrip"]
    assert rt["round_faces"] == 3
    assert rt["rect_faces"] == 6


@freecad_only
def test_rebuild_element_preserves_label_placement_props(fc_probe_result):
    rt = fc_probe_result["rebuild_roundtrip"]
    assert rt["label"] == "MyWindowLabel"
    assert rt["placement_base"] == pytest.approx([1.0, 2.0, 3.0])
    assert rt["filter_prop"] == "probe_marker"
