"""Tests for primitivelib.port_frames -- element-local beam-port geometry.

Runs under the GUI venv (plain python, no FreeCAD): port_frames is a pure
function over the PRIMITIVES dim-sheet parameters. Every formula was derived
from the corresponding _build_* function and cross-checked against the shipped
primitives/*.FCStd bounding boxes / face normals (see the module docstring).
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import primitivelib  # noqa: E402


# Kinds that port_frames implements (the demo set + cheap extras). The complex
# reflective-fold kinds are deliberately KeyError (bbox fallback in the caller).
IMPLEMENTED = [
    "laser_collimated", "laser_divergent", "source_broadband",
    "led_deep_red_660", "led_uv_365",
    "detector_plane",
    "lens_pcx", "lens_dcx", "lens_dcv", "lens_pcv", "lens_meniscus",
    "lens_ball", "lens_rod", "lens_asphere", "lens_achromat", "axicon",
    "mirror_flat", "mirror_concave", "mirror_convex", "mirror_parabolic",
    "mirror_annular", "mirror_d_shaped",
    "bs_plate", "pbs_plate", "dichroic_plate", "nd_reflective", "pellicle",
    "bs_cube", "pbs_cube",
    "prism", "grating_plate",
    "iris", "slit", "pinhole",
    "fiber_optic",
    "window", "filter_plate", "polarizer_plate", "waveplate", "nd_filter",
    "filter_bandpass", "filter_longpass", "filter_shortpass", "filter_notch",
    "diffuser_plate", "window_wedged", "prism_wedge",
    # samples-instruments round: cuvettes/vial/vat (nested pairs, beam
    # along +x -- x=0 is the outer glass front, x=path_length+2*wall or
    # x=diameter the outer glass back) + the bare air sample_region + the
    # four lamp/image sources.
    "cuvette_square", "cuvette_capillary", "flow_cell",
    "vial_cylindrical", "vat_cylindrical", "sample_region",
    "tungsten_halogen", "d2_lamp", "hg_calibration", "source_image",
]

REFLECTIVE = {
    "mirror_flat", "mirror_concave", "mirror_convex", "mirror_parabolic",
    "mirror_annular", "mirror_d_shaped",
    "bs_plate", "pbs_plate", "dichroic_plate", "nd_reflective", "pellicle",
    "bs_cube", "pbs_cube", "grating_plate",
}


def _norm(v):
    return math.sqrt(sum(c * c for c in v))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_all_implemented_kinds_present():
    for kind in IMPLEMENTED:
        assert kind in primitivelib.PRIMITIVES, kind
        pf = primitivelib.port_frames(kind, {})
        assert set(pf) == {"entry", "exit", "axis", "up", "reflect_plane"}


@pytest.mark.parametrize("kind", IMPLEMENTED)
def test_port_invariants(kind):
    pf = primitivelib.port_frames(kind, {})
    entry, exit_, axis, up = pf["entry"], pf["exit"], pf["axis"], pf["up"]

    # all finite
    for name, vec in (("entry", entry), ("exit", exit_),
                      ("axis", axis), ("up", up)):
        assert len(vec) == 3, name
        for c in vec:
            assert math.isfinite(c), (kind, name, c)

    # axis and up: unit and orthogonal
    assert abs(_norm(axis) - 1.0) < 1e-12, kind
    assert abs(_norm(up) - 1.0) < 1e-12, kind
    assert abs(_dot(axis, up)) < 1e-12, kind

    # entry and exit lie on the local axis line through entry, along axis
    d = [exit_[i] - entry[i] for i in range(3)]
    dn = _norm(d)
    if dn > 1e-12:
        dhat = [c / dn for c in d]
        # exit is forward along axis (never behind entry)
        assert abs(abs(_dot(dhat, axis)) - 1.0) < 1e-9, kind
        assert _dot(d, axis) >= -1e-9, kind
    # both vertices are on the y=z=0 axis for these +x-authored primitives
    assert abs(entry[1]) < 1e-12 and abs(entry[2]) < 1e-12, kind
    assert abs(exit_[1]) < 1e-12 and abs(exit_[2]) < 1e-12, kind

    rp = pf["reflect_plane"]
    if kind in REFLECTIVE:
        assert rp is not None, kind
        assert abs(_norm(rp["normal"]) - 1.0) < 1e-12, kind
        # reflect plane point lies on the incoming beam axis (y=z=0 locally)
        assert abs(rp["point"][1]) < 1e-9 and abs(rp["point"][2]) < 1e-9, kind
        # normal points back toward the entry side (into the incoming +x beam)
        assert _dot(rp["normal"], axis) < 0.0, kind
    else:
        assert rp is None, kind


# --- exact vertex-to-vertex thickness formulas -----------------------------

def test_lens_pcx_ct():
    pf = primitivelib.port_frames("lens_pcx", {"R_front": 25.0, "ct": 5.0,
                                               "aperture": 20.0})
    assert pf["entry"] == [0.0, 0.0, 0.0]
    assert pf["exit"] == [5.0, 0.0, 0.0]
    # front convex vertex is the sag apex on-axis at x=0; back flat at ct=5
    assert abs(_norm([pf["exit"][i] - pf["entry"][i]
                      for i in range(3)]) - 5.0) < 1e-12


@pytest.mark.parametrize("kind,ct", [
    ("lens_dcx", 6.0), ("lens_pcv", 3.0), ("lens_dcv", 3.0),
    ("lens_meniscus", 4.0), ("lens_asphere", 6.0),
])
def test_ct_lenses_vertex_to_vertex(kind, ct):
    pf = primitivelib.port_frames(kind, {"ct": ct})
    assert abs(pf["exit"][0] - ct) < 1e-12, kind
    assert pf["entry"][0] == 0.0


def test_lens_ball_diameter():
    pf = primitivelib.port_frames("lens_ball", {"diameter": 8.0})
    assert pf["exit"][0] == 8.0            # sphere spans x=0..2R


def test_lens_achromat_stack():
    pf = primitivelib.port_frames("lens_achromat", {"ct_crown": 6.0,
                                                    "gap": 0.005,
                                                    "ct_flint": 3.0})
    assert abs(pf["exit"][0] - 9.005) < 1e-12


def test_fiber_length():
    pf = primitivelib.port_frames("fiber_optic", {"length": 75.0})
    assert pf["exit"][0] == 75.0


def test_axicon_axial_extent():
    pf = primitivelib.port_frames("axicon", {"base_angle": 10.0,
                                             "aperture": 22.0})
    expect = 11.0 * math.tan(math.radians(10.0))
    assert abs(pf["exit"][0] - expect) < 1e-12


# --- hand-computed reflect-plane cross-checks ------------------------------

def test_mirror_flat_reflect_front():
    # _build_plate front (-x) face at x=0 is the front-surface metal reflector.
    pf = primitivelib.port_frames("mirror_flat", {"thickness": 3.0,
                                                  "width": 25.0})
    assert pf["entry"] == [0.0, 0.0, 0.0]
    assert pf["exit"] == [0.0, 0.0, 0.0]        # entry == exit at the surface
    assert pf["reflect_plane"]["point"] == [0.0, 0.0, 0.0]
    assert pf["reflect_plane"]["normal"] == [-1.0, 0.0, 0.0]


def test_bs_plate_reflect_front_transmit_back():
    pf = primitivelib.port_frames("bs_plate", {"thickness": 3.0})
    assert pf["entry"] == [0.0, 0.0, 0.0]       # coated front face
    assert pf["exit"] == [3.0, 0.0, 0.0]        # transmit exits back face
    assert pf["reflect_plane"]["point"] == [0.0, 0.0, 0.0]
    assert pf["reflect_plane"]["normal"] == [-1.0, 0.0, 0.0]


def test_mirror_parabolic_vertex_and_tangent():
    # revolved sag bulges toward -x with the on-axis vertex at x=0; the
    # tangent-plane normal at the vertex is the axis direction, -x.
    pf = primitivelib.port_frames("mirror_parabolic", {"rfl": 50.0,
                                                       "aperture": 25.0,
                                                       "thickness": 10.0})
    assert pf["entry"] == [0.0, 0.0, 0.0]
    assert pf["exit"] == [0.0, 0.0, 0.0]
    assert pf["reflect_plane"]["point"] == [0.0, 0.0, 0.0]
    assert pf["reflect_plane"]["normal"] == [-1.0, 0.0, 0.0]


def test_bs_cube_diagonal_split():
    pf = primitivelib.port_frames("bs_cube", {"cube": 25.0, "height": 25.0})
    assert pf["entry"] == [0.0, 0.0, 0.0]
    assert pf["exit"] == [25.0, 0.0, 0.0]       # transmit through the cube
    rp = pf["reflect_plane"]
    assert rp["point"] == [12.5, 0.0, 0.0]      # cube center on the axis
    s = math.sqrt(0.5)
    assert abs(rp["normal"][0] + s) < 1e-12
    assert abs(rp["normal"][1] + s) < 1e-12
    assert abs(rp["normal"][2]) < 1e-12


def test_grating_plate_reflect_front():
    pf = primitivelib.port_frames("grating_plate", {"thickness": 3.0})
    assert pf["reflect_plane"]["point"] == [0.0, 0.0, 0.0]
    assert pf["reflect_plane"]["normal"] == [-1.0, 0.0, 0.0]
    assert pf["exit"] == [3.0, 0.0, 0.0]


@pytest.mark.parametrize("kind", [
    "cuvette_square", "cuvette_capillary", "flow_cell",
])
def test_samples_rect_cell_exit_is_path_length_plus_two_wall(kind):
    pf = primitivelib.port_frames(kind, {"path_length": 4.0, "wall": 1.5})
    assert pf["entry"] == [0.0, 0.0, 0.0]
    assert pf["exit"] == [7.0, 0.0, 0.0]      # 4.0 + 2*1.5
    assert pf["reflect_plane"] is None


@pytest.mark.parametrize("kind", ["vial_cylindrical", "vat_cylindrical"])
def test_samples_cylindrical_cell_exit_is_diameter(kind):
    pf = primitivelib.port_frames(kind, {"diameter": 12.0})
    assert pf["entry"] == [0.0, 0.0, 0.0]
    assert pf["exit"] == [12.0, 0.0, 0.0]
    assert pf["reflect_plane"] is None


def test_sample_region_exit_is_width():
    pf = primitivelib.port_frames("sample_region", {"width": 20.0})
    assert pf["entry"] == [0.0, 0.0, 0.0]
    assert pf["exit"] == [20.0, 0.0, 0.0]
    assert pf["reflect_plane"] is None


@pytest.mark.parametrize("kind", [
    "tungsten_halogen", "d2_lamp", "hg_calibration", "source_image",
])
def test_samples_lamp_sources_are_points(kind):
    pf = primitivelib.port_frames(kind, {})
    assert pf["entry"] == [0.0, 0.0, 0.0], kind
    assert pf["exit"] == [0.0, 0.0, 0.0], kind
    assert pf["reflect_plane"] is None, kind


def test_source_and_detector_are_points():
    for kind in ("laser_collimated", "source_broadband", "detector_plane"):
        pf = primitivelib.port_frames(kind, {})
        assert pf["entry"] == [0.0, 0.0, 0.0], kind
        assert pf["exit"] == [0.0, 0.0, 0.0], kind
        assert pf["reflect_plane"] is None, kind


def test_defaults_used_for_missing_params():
    # ct defaults to 5.0 for lens_pcx; passing nothing must still work.
    assert primitivelib.port_frames("lens_pcx", {})["exit"][0] == 5.0
    assert primitivelib.port_frames("lens_pcx", None)["exit"][0] == 5.0


def test_legacy_alias_tolerated():
    # diameter <- radius*2 (LEGACY_ALIASES); a legacy sheet key still resolves.
    pf = primitivelib.port_frames("lens_ball", {"radius": 4.0})
    assert pf["exit"][0] == 8.0


def test_keyerror_for_unknown_kind():
    with pytest.raises(KeyError):
        primitivelib.port_frames("not_a_primitive", {})


@pytest.mark.parametrize("kind", [
    "lens_cyl", "lens_fresnel", "retro_corner_cube", "prism_right_angle",
    "prism_dove", "prism_penta", "prism_rhomboid", "anamorphic_pair",
    "polarizer_glan_taylor",
])
def test_keyerror_for_unported_kinds(kind):
    # these exist in PRIMITIVES but have no closed-form port -> caller falls
    # back to a bbox heuristic.
    assert kind in primitivelib.PRIMITIVES
    with pytest.raises(KeyError):
        primitivelib.port_frames(kind, {})
