"""core/paraxial oracle tests — the ABCD engine must agree with the
wizards.py thick-lens formulas (the project's pinned lens oracle), the
classic two-lens/mirror closed forms, and the f/# = f/D identity."""

import math

import pytest

from mieworkbench.core import paraxial, wizards
from mieworkbench.core.train import TrainModel

N_BK7 = 1.51680        # fixed test index — index_fn below is a constant
N_SF5 = 1.67271


def idx(material, lam_nm):
    return {"bk7": N_BK7, "sf5": N_SF5}.get(material, N_BK7)


# ---------------------------------------------------------------------------
# per-element cardinals vs the wizards thick-lens oracle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind,params,R1,R2", [
    ("lens_pcx", {"R_front": 25.0, "ct": 5.0, "aperture": 20.0},
     25.0, None),
    ("lens_dcx", {"R_front": 40.0, "R_back": 40.0, "ct": 6.0,
                  "aperture": 20.0}, 40.0, -40.0),
    ("lens_pcv", {"R_back": 25.0, "ct": 3.0, "aperture": 20.0},
     None, 25.0),
    ("lens_dcv", {"R_front": 40.0, "R_back": 40.0, "ct": 3.0,
                  "aperture": 20.0}, -40.0, 40.0),
    ("lens_meniscus", {"R_front": 20.0, "R_back": 40.0, "ct": 4.0,
                       "aperture": 18.0}, 20.0, 40.0),
])
def test_element_cardinals_match_wizards(kind, params, R1, R2):
    card = paraxial.element_cardinals(kind, params, idx, 587.6)
    ct = params["ct"]
    assert card["efl"] == pytest.approx(
        wizards.thick_lens_efl(R1, R2, N_BK7, ct), rel=1e-12)
    assert card["bfl"] == pytest.approx(
        wizards.thick_lens_bfl(R1, R2, N_BK7, ct), rel=1e-12)


def test_ball_lens_cardinals():
    D = 8.0
    card = paraxial.element_cardinals("lens_ball", {"diameter": D},
                                      idx, 587.6)
    efl_oracle = N_BK7 * D / (4.0 * (N_BK7 - 1.0))
    assert card["efl"] == pytest.approx(efl_oracle, rel=1e-12)
    # focus measured from the back surface = EFL - D/2 (H2 at center)
    assert card["bfl"] == pytest.approx(efl_oracle - D / 2.0, rel=1e-12)


def test_asphere_uses_vertex_radius():
    card = paraxial.element_cardinals(
        "lens_asphere", {"R": 20.6033, "k": -2.29547, "ct": 6.0,
                         "aperture": 20.0}, idx, 587.6)
    assert card["efl"] == pytest.approx(
        wizards.thick_lens_efl(20.6033, None, N_BK7, 6.0), rel=1e-12)


def test_cylinder_flagged():
    card = paraxial.element_cardinals(
        "lens_cyl", {"R": 25.0, "ct": 5.0, "aperture": 20.0,
                     "height": 20.0}, idx, 587.6)
    assert card["cylindrical"] is True
    assert card["efl"] == pytest.approx(
        wizards.thick_lens_efl(25.0, None, N_BK7, 5.0), rel=1e-12)


def test_mirrors():
    assert paraxial.element_cardinals(
        "mirror_concave", {"R": 100.0, "aperture": 25.0, "ct": 4.0},
        idx, 587.6)["efl"] == pytest.approx(50.0)
    assert paraxial.element_cardinals(
        "mirror_convex", {"R": 100.0, "aperture": 25.0, "ct": 4.0},
        idx, 587.6)["efl"] == pytest.approx(-50.0)
    assert paraxial.element_cardinals(
        "mirror_parabolic", {"rfl": 50.0, "aperture": 25.0,
                             "thickness": 10.0},
        idx, 587.6)["efl"] == pytest.approx(50.0)
    flat = paraxial.element_cardinals(
        "mirror_flat", {"width": 25.0, "thickness": 3.0, "round_flag": 0},
        idx, 587.6)
    assert flat["afocal"] is True and flat["mirror"] is True


def test_achromat_close_to_design_focal_length():
    params = {"R_front": 31.0, "R_iface": -21.956, "R_back": -64.497,
              "ct_crown": 6.0, "ct_flint": 3.0, "gap": 0.005,
              "aperture": 18.0}
    card = paraxial.element_cardinals("lens_achromat", params, idx, 587.6)
    # shipped BK7/SF5 "f=50" design: the exact paraxial EFL of the
    # prescription is 51.03 mm (50 is the nominal label) — pin within 3%
    # of nominal and tightly against the exact value
    assert card["efl"] == pytest.approx(50.0, rel=0.03)
    assert card["efl"] == pytest.approx(51.03, rel=0.002)


def test_achromat_is_achromatic():
    """The whole point of the doublet: EFL at the F and C lines must agree
    far more tightly than a singlet's (real matdb dispersion check lives
    in the e2e suite; here the constant-index stand-ins differ per glass,
    so evaluate with a tiny two-line dispersion model instead)."""
    params = {"R_front": 31.0, "R_iface": -21.956, "R_back": -64.497,
              "ct_crown": 6.0, "ct_flint": 3.0, "gap": 0.005,
              "aperture": 18.0}

    def idx_disp(mat, lam):
        # two-point linear dispersion through (F, C) for BK7 / SF5
        pts = {"bk7": (1.52238, 1.51432), "sf5": (1.68876, 1.66661)}
        nf, nc = pts.get(mat, pts["bk7"])
        t = (lam - 486.1) / (656.3 - 486.1)
        return nf + (nc - nf) * t

    f_F = paraxial.element_cardinals("lens_achromat", params, idx_disp,
                                     486.1)["efl"]
    f_C = paraxial.element_cardinals("lens_achromat", params, idx_disp,
                                     656.3)["efl"]
    assert abs(f_F - f_C) / f_F < 2e-3


def test_fresnel_thin_lens():
    card = paraxial.element_cardinals(
        "lens_fresnel", {"aperture": 24.0, "f_design": 50.0,
                         "n_design": 1.51508, "n_facets": 12.0,
                         "back": 2.0}, idx, 587.6)
    assert card["efl"] == pytest.approx(50.0) and card["approximate"]


def test_slab_window_image_shift():
    M, t, meta = paraxial.element_matrix(
        "window", {"width": 25.0, "thickness": 3.0, "round_flag": 1},
        idx, 587.6, material="bk7")
    # slab: A=1, B=t/n, C=0, D=1
    assert M == pytest.approx((1.0, 3.0 / N_BK7, 0.0, 1.0))


def test_unknown_kind_is_passthrough():
    M, t, meta = paraxial.element_matrix("prism", {"side": 25.0}, idx, 587.6)
    assert M == paraxial.IDENT and meta["passthrough"]


# ---------------------------------------------------------------------------
# train fixtures
# ---------------------------------------------------------------------------
def _sprop(v):
    return {"type": "App::PropertyString", "group": "Base", "value": v}


def _fprop(v):
    return {"type": "App::PropertyFloat", "group": "Base", "value": v}


def _body(name, kind=None, material=None, chain=None, props=None):
    p = {"miewb_group": _sprop(name)}
    if kind:
        p["miewb_primitive"] = _sprop(kind)
    if material:
        p["material"] = _sprop(material)
    for field, val in (chain or {}).items():
        from mieworkbench.core.train import FIELD_PROPS
        p[FIELD_PROPS[field]] = _sprop(str(val)) if not isinstance(val, bool) \
            else {"type": "App::PropertyBool", "group": "MieTrain",
                  "value": val}
    p.update(props or {})
    return {"name": name, "label": name, "tip": "%s_pad" % name,
            "placement": {"pos_mm": [0, 0, 0], "quat": [0, 0, 0, 1]},
            "properties": p}


def _sheet(element, aliases):
    return {"name": "dim", "label": "dim_%s" % element,
            "aliases": {a: {"cell": "B1", "raw": "=%s" % v, "value": v,
                            "unit": "mm"} for a, v in aliases.items()}}


def _tm(bodies, sheets):
    return TrainModel({"bodies": bodies, "sheets": sheets}, {})


PCX = {"R_front": 25.0, "ct": 5.0, "aperture": 20.0}


def _two_lens_model(spacing=40.0, ap2=20.0):
    bodies = [
        _body("SRC", kind="laser_collimated",
              props={"power": _fprop(5.0), "lambdac": _fprop(633.0)}),
        _body("L1", kind="lens_pcx", material="bk7",
              chain={"mode": "chained", "ref": "SRC", "distance": "30"}),
        _body("L2", kind="lens_pcx", material="bk7",
              chain={"mode": "chained", "ref": "L1",
                     "distance": str(spacing)}),
        _body("DET", kind="detector_plane", material="detector",
              chain={"mode": "chained", "ref": "L2", "distance": "80"}),
    ]
    sheets = [
        _sheet("SRC", {"diameter": 6.0, "length": 20.0}),
        _sheet("L1", PCX),
        _sheet("L2", dict(PCX, aperture=ap2)),
        _sheet("DET", {"width": 30.0, "height": 0.0, "thickness": 1.0,
                       "round_flag": 0}),
    ]
    return _tm(bodies, sheets)


def test_design_wavelength_from_source():
    tm = _two_lens_model()
    assert paraxial.design_wavelength_nm(tm) == pytest.approx(633.0)


def test_design_wavelength_fallback():
    tm = _tm([_body("L1", kind="lens_pcx", material="bk7")],
             [_sheet("L1", PCX)])
    assert paraxial.design_wavelength_nm(tm) == pytest.approx(
        paraxial.DESIGN_LAMBDA_FALLBACK_NM)


def test_chain_path_order_spacing_flags():
    tm = _two_lens_model()
    path, warnings = paraxial.chain_path(tm)
    assert [p["element"] for p in path] == ["SRC", "L1", "L2", "DET"]
    assert path[0]["is_source"] and path[3]["is_detector"]
    assert path[1]["spacing_mm"] == pytest.approx(30.0)
    assert path[2]["spacing_mm"] == pytest.approx(40.0)
    assert path[1]["aperture_mm"] == pytest.approx(20.0)
    assert warnings == []


def test_chain_path_expression_distance_and_tilt_warning():
    bodies = [
        _body("SRC", kind="laser_collimated",
              props={"power": _fprop(5.0), "lambdac": _fprop(633.0)}),
        _body("L1", kind="lens_pcx", material="bk7",
              chain={"mode": "chained", "ref": "SRC",
                     "distance": "gap*2", "tilt_rx": "1.5"}),
    ]
    sheets = [_sheet("SRC", {"diameter": 6.0}), _sheet("L1", PCX)]
    tm = _tm(bodies, sheets)
    path, warnings = paraxial.chain_path(tm, {"gap": 25.0})
    assert path[1]["spacing_mm"] == pytest.approx(50.0)
    assert any("tilt_rx" in w for w in warnings)


def test_system_single_element_matches_element_cardinals():
    tm = _two_lens_model()
    # truncate: only SRC + L1 by rebuilding a single-lens model
    bodies = [
        _body("SRC", kind="laser_collimated",
              props={"power": _fprop(5.0), "lambdac": _fprop(633.0)}),
        _body("L1", kind="lens_pcx", material="bk7",
              chain={"mode": "chained", "ref": "SRC", "distance": "30"}),
    ]
    tm1 = _tm(bodies, [_sheet("SRC", {"diameter": 6.0}), _sheet("L1", PCX)])
    s = paraxial.system_summary(tm1, index_fn=idx)
    card = paraxial.element_cardinals("lens_pcx", PCX, idx, 633.0)
    assert s["efl"] == pytest.approx(card["efl"], rel=1e-12)
    assert s["bfl"] == pytest.approx(card["bfl"], rel=1e-12)
    assert s["image_distance_mm"] == pytest.approx(card["bfl"], rel=1e-12)


def test_system_two_lens_efl_matches_principal_plane_formula():
    spacing = 40.0
    tm = _two_lens_model(spacing=spacing)
    s = paraxial.system_summary(tm, index_fn=idx)
    card = paraxial.element_cardinals("lens_pcx", PCX, idx, 633.0)
    f = card["efl"]
    # separation between H2 of L1 and H1 of L2
    d_pp = (spacing - card["pp2_mm"]) + card["pp1_mm"]
    f_sys = f * f / (f + f - d_pp)
    assert s["efl"] == pytest.approx(f_sys, rel=1e-9)


def test_system_fno_equals_f_over_d():
    tm_bodies = [
        _body("SRC", kind="laser_collimated",
              props={"power": _fprop(5.0), "lambdac": _fprop(633.0)}),
        _body("L1", kind="lens_pcx", material="bk7",
              chain={"mode": "chained", "ref": "SRC", "distance": "30"}),
    ]
    tm = _tm(tm_bodies, [_sheet("SRC", {"diameter": 6.0}),
                         _sheet("L1", PCX)])
    s = paraxial.system_summary(tm, index_fn=idx)
    # collimated input, lens is its own stop: f/# = EFL / aperture
    assert s["limiting_element"] == "L1"
    assert s["fno_working"] == pytest.approx(s["efl"] / PCX["aperture"],
                                             rel=1e-9)
    assert s["na"] == pytest.approx(PCX["aperture"] / (2 * s["efl"]),
                                    rel=1e-9)


def test_system_iris_becomes_limiting_stop():
    bodies = [
        _body("SRC", kind="laser_collimated",
              props={"power": _fprop(5.0), "lambdac": _fprop(633.0)}),
        _body("L1", kind="lens_pcx", material="bk7",
              chain={"mode": "chained", "ref": "SRC", "distance": "30"}),
        _body("STOP", kind="iris",
              chain={"mode": "chained", "ref": "L1", "distance": "5"}),
    ]
    sheets = [_sheet("SRC", {"diameter": 6.0}), _sheet("L1", PCX),
              _sheet("STOP", {"diameter": 8.0, "thickness": 1.0})]
    tm = _tm(bodies, sheets)
    s = paraxial.system_summary(tm, index_fn=idx)
    assert s["limiting_element"] == "STOP"
    # marginal through the stop: NA = (stop/2) * (h_lens/h_stop) / ...
    # simply: f/# grows vs the lens-limited case
    assert s["fno_working"] > s["efl"] / PCX["aperture"]


def test_finite_conjugate_thin_lens_2f():
    # near-thin lens: 2f -> 2f imaging at m = -1
    params = {"R_front": 25.0, "ct": 0.01, "aperture": 20.0}
    bodies = [
        _body("SRC", kind="laser_divergent",
              props={"power": _fprop(5.0), "lambdac": _fprop(633.0)}),
        _body("L1", kind="lens_pcx", material="bk7",
              chain={"mode": "chained", "ref": "SRC", "distance": "30"}),
    ]
    tm = _tm(bodies, [_sheet("SRC", {"diameter": 6.0}),
                      _sheet("L1", params)])
    card = paraxial.element_cardinals("lens_pcx", params, idx, 633.0)
    f = card["efl"]
    s = paraxial.system_summary(tm, index_fn=idx, object_distance_mm=2 * f)
    assert s["image_distance_mm"] == pytest.approx(2 * f, rel=1e-3)
    assert s["magnification"] == pytest.approx(-1.0, abs=2e-3)


def test_system_stops_at_detector_and_reports_gap():
    tm = _two_lens_model()
    s = paraxial.system_summary(tm, index_fn=idx)
    assert s["n_optical_elements"] == 2
    assert s.get("detector_gap_mm") == pytest.approx(80.0)


def test_mirror_in_train_contributes_power():
    bodies = [
        _body("SRC", kind="laser_collimated",
              props={"power": _fprop(5.0), "lambdac": _fprop(633.0)}),
        _body("M1", kind="mirror_concave", material="aluminum",
              chain={"mode": "chained", "ref": "SRC", "distance": "50"}),
    ]
    sheets = [_sheet("SRC", {"diameter": 6.0}),
              _sheet("M1", {"R": 200.0, "aperture": 25.0, "ct": 4.0})]
    tm = _tm(bodies, sheets)
    s = paraxial.system_summary(tm, index_fn=idx)
    assert s["efl"] == pytest.approx(100.0, rel=1e-12)
    assert s["image_distance_mm"] == pytest.approx(100.0, rel=1e-12)
