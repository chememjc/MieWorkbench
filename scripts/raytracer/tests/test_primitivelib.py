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


# ---------------------------------------------------------------------------
# New catalog primitives (v2-feature-round batch 1+2): plate-likes, prisms,
# mirrors, apertures. Pure-python structural checks first, then FreeCAD-gated
# build/rebuild + geometry invariants.
# ---------------------------------------------------------------------------
NEW_PLATE_KINDS = [
    "bs_plate", "pbs_plate", "dichroic_plate", "pellicle", "nd_filter",
    "nd_reflective", "filter_bandpass", "filter_longpass", "filter_shortpass",
    "filter_notch", "window_wedged", "diffuser_plate",
]
NEW_PRISM_MIRROR_APERTURE_KINDS = [
    "prism_right_angle", "prism_wedge", "prism_dove", "prism_penta",
    "prism_rhomboid", "mirror_concave", "mirror_convex", "mirror_d_shaped",
    "iris", "pinhole", "slit", "retro_corner_cube",
]
BATCH3_KINDS = [
    "bs_cube", "anamorphic_pair", "polarizer_glan_taylor", "mirror_parabolic",
]
BATCHC_KINDS = [
    "fiber_optic", "mirror_annular",
]
NEW_KINDS = (NEW_PLATE_KINDS + NEW_PRISM_MIRROR_APERTURE_KINDS
             + BATCH3_KINDS + BATCHC_KINDS)
APERTURE_KINDS = ("iris", "pinhole", "slit")
# every non-aperture kind that builds two bodies (vs. the single-body norm)
TWO_BODY_KINDS = APERTURE_KINDS + (
    "bs_cube", "anamorphic_pair", "polarizer_glan_taylor", "fiber_optic")


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_new_kind_registered_with_category_and_props(kind):
    spec = pl.PRIMITIVES[kind]
    assert spec["category"]
    assert spec["label"]
    assert spec["tooltip"]
    assert spec["params"]
    assert "props" in spec


@pytest.mark.parametrize("kind", APERTURE_KINDS)
def test_aperture_kinds_have_blackness_and_derived_absorbance(kind):
    params = pl.PRIMITIVES[kind]["params"]
    assert "blackness" in params
    assert 0.95 <= params["blackness"]["default"] <= 1.0
    assert "absorbance" in pl.PRIMITIVES[kind].get("derived_props", ())


def test_no_new_kind_defines_a_round_flag_where_the_spec_says_always_round():
    # pellicle, prism_wedge, retro_corner_cube are always-round/no-shape-
    # toggle primitives per the authoring spec -- they use 'diameter' or
    # 'aperture', not 'width'+'round_flag'.
    for kind in ("pellicle", "prism_wedge", "retro_corner_cube"):
        assert "round_flag" not in pl.PRIMITIVES[kind]["params"], kind


def test_derived_props_excluded_from_rebuild_baseline_for_apertures():
    for kind in APERTURE_KINDS:
        assert pl.PRIMITIVES[kind]["derived_props"] == ("absorbance",)
    # asphere-backed primitives: the surface_override string is derived
    # from the sheet params by the builder; preserving the pre-rebuild
    # value would trip the extractor's <1 um verification (found by the
    # newtonian demo: a rebuilt rfl=900 primary kept its rfl=50 override)
    for kind in ("lens_asphere", "mirror_parabolic"):
        assert pl.PRIMITIVES[kind]["derived_props"] == \
            ("surface_override",), kind
    # the cube splitters' coating string names a face index of the
    # freshly built plate -- re-derived every rebuild
    for kind in ("bs_cube", "pbs_cube"):
        assert pl.PRIMITIVES[kind]["derived_props"] == ("coating",), kind
    exempt = set(APERTURE_KINDS) | {"lens_asphere", "mirror_parabolic",
                                    "bs_cube", "pbs_cube"}
    for kind in set(pl.PRIMITIVES) - exempt:
        assert not pl.PRIMITIVES[kind].get("derived_props"), kind


@freecad_only
def test_new_kinds_build_and_rebuild_preserve_label_and_placement(
        fc_probe_result):
    per_kind = fc_probe_result["new_kinds_build_rebuild"]
    for kind in NEW_KINDS:
        info = per_kind[kind]
        expected_n = 2 if kind in TWO_BODY_KINDS else 1
        assert info["n_before"] == expected_n, kind
        assert info["n_after"] == expected_n, kind
        assert info["label_ok"], kind
        assert info["placement_ok"], kind


@freecad_only
@pytest.mark.parametrize("kind", APERTURE_KINDS)
def test_aperture_disc_absorbance_tracks_blackness(fc_probe_result, kind):
    info = fc_probe_result["apertures"][kind]
    assert info["initial"]["n_bodies"] == 2
    assert info["initial"]["plug_material"] == "air"
    assert info["initial"]["disc_absorbance"] == \
        pytest.approx(info["initial"]["blackness_param"])
    # after rebuild with blackness changed to 0.5, the disc's absorbance
    # prop must track the NEW value (derived_props keeps rebuild_element's
    # generic extra-prop preservation from clobbering it with the stale
    # pre-rebuild absorbance).
    after = info["after_blackness_rebuild"]
    assert after["n_bodies"] == 2
    assert after["plug_material"] == "air"
    assert after["disc_absorbance"] == pytest.approx(0.5)


@freecad_only
def test_retro_corner_cube_has_three_mutually_perpendicular_back_faces(
        fc_probe_result):
    cc = fc_probe_result["corner_cube"]
    assert cc["n_faces"] == 4
    # of the C(4,2)=6 pairs, exactly 3 involve the mutually-perpendicular
    # trihedral (each of the 3 back faces is perpendicular to the other 2);
    # the remaining 3 pairs are back-face-vs-entrance-face (not perpendicular).
    assert cc["n_perp_pairs"] == 3


# ---------------------------------------------------------------------------
# Batch 3 (final): beamsplitter cube, anamorphic prism pair, Glan-Taylor
# polarizer, on-axis parabolic mirror (descoped from an off-axis OAP -- see
# the module docstring in primitivelib._build_mirror_parabolic).
# ---------------------------------------------------------------------------
@freecad_only
def test_bs_cube_nested_plate_with_coating_on_the_plate(fc_probe_result):
    """Nested-plate design (glass-glass split interface): the old
    two-prism + 5 um air gap TIR'd the transmitted arm at 45 deg (past
    BK7's critical angle) and lost ~1/3 of the power to seam loss."""
    info = fc_probe_result["batch3_geometry"]["bs_cube"]
    assert info["n_bodies"] == 2
    assert info["plate_inside"] is True
    assert info["coating_cube"] is None
    assert info["coating_plate"] is not None
    assert "bs_5050_vis_45" in info["coating_plate"]


@freecad_only
def test_anamorphic_pair_two_nonoverlapping_bodies_identity_placement(
        fc_probe_result):
    info = fc_probe_result["batch3_geometry"]["anamorphic_pair"]
    assert info["n_bodies"] == 2
    assert info["gap_mm"] > 0.0     # no overlap between the two prisms
    assert info["materials"] == ["bk7", "bk7"]
    # offsets are baked into each prism's local geometry, not a Placement
    # transform (the achromat convention, unlike pbs_cube's shifted body)
    assert all(info["placements_identity"])


@freecad_only
def test_glan_taylor_two_calcite_bodies_with_crystal_axis_and_gap(
        fc_probe_result):
    info = fc_probe_result["batch3_geometry"]["polarizer_glan_taylor"]
    assert info["n_bodies"] == 2
    assert info["gap_mm"] == pytest.approx(0.005, abs=1e-6)
    assert info["materials"] == ["calcite", "calcite"]
    assert info["crystal_axes"] == ["0,0,1", "0,0,1"]


# ---------------------------------------------------------------------------
# Batch C (demo-gallery round): step-index fiber, annular concave mirror,
# rectangular detector_plane. Physics validated end-to-end by the demos/
# gallery smoke runs (fiber TIR closure, annular-mirror focus fraction);
# here: registry structure + the material row's NA against the cladding.
# ---------------------------------------------------------------------------
def test_fiber_optic_registry():
    spec = pl.PRIMITIVES["fiber_optic"]
    assert spec["category"] == "Fiber Optics"
    assert set(spec["params"]) == {"core_diameter", "clad_diameter",
                                   "length", "gap"}
    # the 5 um optical-contact modeling gap convention
    assert spec["params"]["gap"]["default"] == pytest.approx(0.005)
    assert spec["params"]["clad_diameter"]["default"] > \
        spec["params"]["core_diameter"]["default"]


def test_mirror_annular_registry():
    spec = pl.PRIMITIVES["mirror_annular"]
    assert set(spec["params"]) == {"R", "aperture", "hole_diameter", "ct"}
    assert spec["params"]["hole_diameter"]["default"] < \
        spec["params"]["aperture"]["default"]
    assert spec["props"]["material"] == "aluminum"


def test_detector_plane_height_param_defaults_to_square():
    params = pl.PRIMITIVES["detector_plane"]["params"]
    assert params["height"]["default"] == 0.0   # 0 = legacy square shape


def test_fiber_core_material_gives_na_022_vs_fused_silica():
    import numpy as np
    from raytracer.optprops import load_optical_properties
    props = load_optical_properties()
    lam = np.array([650e-9])
    n_core = props.matdb.get("fiber_core_na22").n_complex(lam).real[0]
    n_clad = props.matdb.get("fused_silica").n_complex(lam).real[0]
    assert (n_core ** 2 - n_clad ** 2) ** 0.5 == pytest.approx(0.22,
                                                               abs=2e-3)


def test_mirror_parabolic_k_minus_one_not_user_tunable():
    # a true parabola: k is baked in, not exposed as a spec param (unlike
    # lens_asphere, where k is a free conic-constant knob)
    params = pl.PRIMITIVES["mirror_parabolic"]["params"]
    assert set(params) == {"rfl", "aperture", "thickness"}


# ---------------------------------------------------------------------------
# mirror_parabolic: extraction sanity (surface_override=asphere verified,
# reflecting face present in model.json) + an engine-level geometric focus
# check. The FCStd scene is built by the FreeCAD probe
# (probe_build_mirror_parabolic_scene); extraction itself needs a SECOND
# FreeCAD subprocess (extract_geometry.py, same AppImage) since the probe
# script can't import extract_geometry.py directly (no __main__ guard --
# it calls main()/os._exit() at import time, same trap as make_test_scenes).
# The engine-level physics (ray injection + tracer stepping) then runs in
# THIS (optics-env) interpreter -- no FreeCAD needed for that half.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mirror_parabolic_model_json(fc_probe_result, tmp_path_factory):
    if not RUN_FREECAD:
        pytest.skip("set MIEWB_RUN_FREECAD=1 to run FreeCAD-backed "
                    "primitivelib tests")
    fcstd = fc_probe_result["mirror_parabolic_scene"]["path"]
    outdir = tmp_path_factory.mktemp("mirror_parabolic_geom")
    extract_script = SCRIPTS_DIR / "extract_geometry.py"
    subprocess.run(
        [FREECAD_APPIMAGE, "-c", str(extract_script), "--",
         "--models", fcstd, "--outdir", str(outdir), "--strict"],
        stdin=subprocess.DEVNULL, check=True, capture_output=True, text=True)
    stem = Path(fcstd).stem
    model_path = outdir / stem / "model.json"
    assert model_path.exists()
    with open(model_path) as fh:
        return json.load(fh)


@freecad_only
def test_mirror_parabolic_extraction_verifies_asphere_override(
        mirror_parabolic_model_json):
    mirror = [b for b in mirror_parabolic_model_json["bodies"]
             if b["name"] == "Mirror"][0]
    asphere_faces = [f for f in mirror["faces"]
                     if f["surface"]["type"] == "asphere"]
    assert len(asphere_faces) == 1
    surf = asphere_faces[0]["surface"]
    assert surf["k"] == pytest.approx(-1.0)
    assert surf["R"] == pytest.approx(0.1, rel=1e-6)   # 2*rfl = 2*0.050 m


@freecad_only
def test_mirror_parabolic_geometric_focus(mirror_parabolic_model_json):
    """A coherent=false collimated bundle parallel to the mirror axis must
    converge (geometrically -- no diffraction) at the paraxial focus
    x=-rfl from the vertex: an exact parabola has zero on-axis spherical
    aberration at ANY aperture, so essentially all reflected rays should
    cross the focal plane within a tiny fraction of the aperture radius.
    Manual ray injection + tracer.step() harvesting (the
    test_integration.py::test_traced_focus_matches_lensmaker pattern) is
    used instead of a physical detector plane: a co-axial on-axis screen
    between the source and the mirror would intercept the OUTGOING
    collimated beam before it ever reaches the mirror (self-shadowing)."""
    import numpy as np

    import common
    from raytracer.scene import Scene
    from raytracer.tracer import Tracer, TraceConfig
    from raytracer.rays import RayBatch
    from raytracer.optprops import load_optical_properties

    model = mirror_parabolic_model_json
    optprops = load_optical_properties()
    scene = Scene(model, optprops.matdb, optprops.coatings,
                 optprops=optprops)

    rfl_m = 0.050
    aperture_m = 0.025
    x_focus = -rfl_m

    rng = np.random.default_rng(0)
    m = 400
    r = (aperture_m / 2.0 * 0.9) * np.sqrt(rng.uniform(0.0, 1.0, m))
    th = rng.uniform(0.0, 2 * np.pi, m)
    y, z = r * np.cos(th), r * np.sin(th)

    batch = RayBatch(m)
    batch.pos[:] = np.stack([np.full(m, -0.20), y, z], axis=-1)
    batch.dir[:] = np.tile([1.0, 0.0, 0.0], (m, 1))
    batch.s_hat[:] = np.tile([0.0, 0.0, 1.0], (m, 1))
    batch.Es[:] = 1.0
    batch.Ep[:] = 1.0
    batch.lam[:] = 633e-9
    batch.birth_power[:] = batch.power

    cfg = TraceConfig(rays=m, n_lambda=1, seed=1, power_floor=1e-6)
    tracer = Tracer(scene, cfg, {})     # no screens: pure geometry

    queue = [batch]
    hit_y, hit_z = [], []
    for _ in range(6):
        if not queue:
            break
        children = tracer.step(queue.pop())
        if children is None or len(children) == 0:
            continue
        c = children
        sel = c.dir[:, 0] < -0.9    # reflected bundle now heads back -x
        if np.any(sel):
            p, d = c.pos[sel], c.dir[sel]
            tstar = (x_focus - p[:, 0]) / d[:, 0]
            hit_y.extend((p[:, 1] + tstar * d[:, 1]).tolist())
            hit_z.extend((p[:, 2] + tstar * d[:, 2]).tolist())
        queue.append(children)

    hit_y, hit_z = np.array(hit_y), np.array(hit_z)
    assert len(hit_y) >= m * 0.9, "lost too many rays off the mirror"
    radius = np.hypot(hit_y, hit_z)
    frac_concentrated = float(np.mean(radius < 2e-3))    # 2 mm
    assert frac_concentrated > 0.80, frac_concentrated


# ---------------------------------------------------------------------------
# Workstream D: monochromatic-LED source presets (library_data/
# emission_led_monochromatic.csv). lambdamin/lambdamax = cwl -+ FWHM/2.3548
# so the existing Gaussian source (sources.py:19-22, half-normal each side)
# reproduces the LED's datasheet FWHM.
# ---------------------------------------------------------------------------
# cwl_nm, fwhm_nm straight from the CSV (led_type -> "led_" + led_type).
LED_KINDS_CSV = {
    "led_deep_red_660": (660, 20),
    "led_red_630": (625, 20),
    "led_amber_590": (590, 20),
    "led_green_525": (527, 30),
    "led_blue_470": (472, 20),
    "led_royal_blue_450": (452, 20),
    "led_uv_365": (365, 9.0),
    "led_uv_385": (385, 11),
}


@pytest.mark.parametrize("kind", sorted(LED_KINDS_CSV))
def test_led_kind_registered_sources_category(kind):
    spec = pl.PRIMITIVES[kind]
    assert spec["category"] == "Sources"
    assert spec["label"]
    assert spec["tooltip"]
    assert spec["params"]
    assert kind in pl.builders() if pl._HAVE_FREECAD else True


@pytest.mark.parametrize("kind", sorted(LED_KINDS_CSV))
def test_led_kind_lambda_ordering(kind):
    props = pl.PRIMITIVES[kind]["props"]
    assert props["lambdamin"] < props["lambdac"] < props["lambdamax"]
    assert props["coherent"] is False
    assert props["power"] == pytest.approx(5.0)


@pytest.mark.parametrize("kind,cwl_fwhm", sorted(LED_KINDS_CSV.items()))
def test_led_kind_lambdac_matches_csv_cwl(kind, cwl_fwhm):
    cwl, _fwhm = cwl_fwhm
    props = pl.PRIMITIVES[kind]["props"]
    assert props["lambdac"] == pytest.approx(cwl)


@pytest.mark.parametrize("kind,cwl_fwhm", sorted(LED_KINDS_CSV.items()))
def test_led_kind_bounds_reproduce_csv_fwhm(kind, cwl_fwhm):
    _cwl, fwhm = cwl_fwhm
    props = pl.PRIMITIVES[kind]["props"]
    recovered_fwhm = (props["lambdamax"] - props["lambdamin"]) * 2.3548 / 2.0
    assert recovered_fwhm == pytest.approx(fwhm, rel=0.005)


def test_led_kinds_share_source_broadband_params():
    ref_params = pl.PRIMITIVES["source_broadband"]["params"]
    for kind in LED_KINDS_CSV:
        params = pl.PRIMITIVES[kind]["params"]
        assert set(params) == set(ref_params)
        for alias, spec in ref_params.items():
            assert params[alias]["default"] == spec["default"], (kind, alias)
            assert params[alias]["unit"] == spec["unit"], (kind, alias)


def test_led_kinds_use_source_broadband_builder():
    if not pl._HAVE_FREECAD:
        pytest.skip("builders() needs FreeCAD")
    b = pl.builders()
    for kind in LED_KINDS_CSV:
        assert b[kind] is b["source_broadband"], kind


@pytest.mark.parametrize("kind", sorted(LED_KINDS_CSV))
def test_led_kind_meta_json_matches_registry(kind):
    meta_path = REPO_ROOT / "primitives" / (kind + ".meta.json")
    assert meta_path.exists(), meta_path
    with open(meta_path) as fh:
        meta = json.load(fh)
    assert meta["kind"] == kind
    assert meta["category"] == "Sources"
    assert meta["props"] == pl.PRIMITIVES[kind]["props"]
