"""Wizard math validated against the physics constants pinned in
make_test_scenes.SCENES (the original project's oracle values)."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

from mieworkbench.core import wizards  # noqa: E402
import make_test_scenes  # noqa: E402  (SCENES importable without FreeCAD)

SCENES = make_test_scenes.SCENES
N_633 = 1.51508     # bk7 @ 633nm, as pinned in SCENES
N_D = 1.51680       # bk7 @ d-line


# -- forward formula vs the SCENES oracle -------------------------------------
@pytest.mark.parametrize("scene", ["lens_pcx", "lens_dcx", "lens_pcv",
                                   "lens_dcv", "lens_sphere_control"])
def test_thick_lens_efl_matches_scenes(scene):
    s = SCENES[scene]
    efl = wizards.thick_lens_efl(s["R1_mm"], s["R2_mm"],
                                 s["n_633"], s["thickness_mm"])
    assert efl == pytest.approx(s["expected_efl_mm"], abs=0.01)


def test_bfl_matches_pcx_scene():
    s = SCENES["lens_pcx"]
    bfl = wizards.thick_lens_bfl(s["R1_mm"], s["R2_mm"], s["n_633"],
                                 s["thickness_mm"])
    assert bfl == pytest.approx(s["expected_bfl_mm"], abs=0.01)


def test_ball_lens_matches_scene():
    s = SCENES["lens_ball"]
    out = wizards.solve_ball(s["expected_efl_mm"], s["n_d"])
    assert out["diameter"] == pytest.approx(s["diameter_mm"], abs=0.01)
    assert out["bfl"] == pytest.approx(s["expected_bfl_mm"], abs=0.01)


# -- inverse solvers round-trip through the forward formula --------------------
@pytest.mark.parametrize("form,f,d", [
    ("pcx", 50.0, 5.0), ("pcx", 100.0, 6.0),
    ("pcv", -50.0, 3.0),
    ("dcx", 40.0, 6.0), ("dcx", 200.0, 4.0),
    ("dcv", -40.0, 3.0),
])
def test_solver_roundtrip_exact(form, f, d):
    solver = {"pcx": wizards.solve_pcx, "pcv": wizards.solve_pcv,
              "dcx": wizards.solve_equiconvex,
              "dcv": wizards.solve_equiconcave}[form]
    out = solver(f, N_633, d)
    assert out["efl"] == pytest.approx(f, rel=1e-9)


def test_pcx_50mm_radius_value():
    out = wizards.solve_pcx(50.0, N_633, 5.0)
    assert out["R_front"] == pytest.approx(0.51508 * 50.0, rel=1e-12)
    assert out["R_back"] is None


def test_best_form_shape_factor_and_focal():
    out = wizards.solve_best_form(50.0, N_633, 4.0)
    q = 2 * (N_633 ** 2 - 1) / (N_633 + 2)
    assert out["shape_factor"] == pytest.approx(q)
    # thin-lens design: exact EFL lands within ~2% for a 4mm lens
    assert out["efl"] == pytest.approx(50.0, rel=0.02)


def test_meniscus_solver_exact():
    out = wizards.solve_meniscus(60.0, N_633, 4.0, R_front=20.0)
    assert out["efl"] == pytest.approx(60.0, rel=1e-9)
    # both surfaces curve the same way -> same sign
    assert out["R_back"] is not None and out["R_back"] > 0


def test_asphere_conic():
    out = wizards.solve_asphere(40.0, N_633, 6.0)
    # matches the SCENES lens_asphere full-lens-corrected design
    assert out["k"] == pytest.approx(SCENES["lens_asphere"]["conic_k"],
                                     abs=0.001)
    assert out["A4_mm3"] == pytest.approx(
        SCENES["lens_asphere"]["asphere_A4_mm"], rel=1e-9)
    # A4 scale transfer: half the focal length -> 8x the coefficient
    out20 = wizards.solve_asphere(20.0, N_633, 3.0)
    assert out20["A4_mm3"] == pytest.approx(8 * out["A4_mm3"], rel=1e-9)


def test_achromat_scaling():
    out = wizards.solve_achromat(100.0)
    assert out["R_front"] == pytest.approx(62.0)
    assert out["ct_crown"] == pytest.approx(12.0)
    out50 = wizards.solve_achromat(50.0)
    assert out50["R_front"] == pytest.approx(
        SCENES["lens_achromat"]["R1_mm"])


def test_impossible_designs_raise():
    with pytest.raises(ValueError):
        wizards.solve_pcx(-50.0, N_633, 5.0)
    with pytest.raises(ValueError):
        wizards.solve_equiconvex(2.0, N_633, 6.0)   # too thick for f
    with pytest.raises(ValueError):
        wizards.solve_ball(-5.0, N_633)
    with pytest.raises(ValueError):
        wizards.design_lens("klingon", 50.0)


def test_design_lens_maps_to_primitive_params():
    out = wizards.design_lens("dcx", 40.0, n=N_633, ct_mm=6.0)
    assert out["primitive"] == "lens_dcx"
    p = out["params"]
    # primitive stores back-radius MAGNITUDE (R_back alias > 0)
    assert p["R_back"] > 0
    assert p["R_front"] == pytest.approx(p["R_back"])
    efl = wizards.thick_lens_efl(p["R_front"], -p["R_back"], N_633, 6.0)
    assert efl == pytest.approx(40.0, rel=1e-9)
    # matches the shipped lens_dcx scene design (R=40, ct=6 -> f=39.845)
    scene = SCENES["lens_dcx"]
    assert wizards.thick_lens_efl(scene["R1_mm"], scene["R2_mm"], N_633,
                                  6.0) == pytest.approx(39.845, abs=0.01)


def test_design_lens_with_real_matdb():
    from raytracer.materials import MaterialDB
    db = MaterialDB.load()
    out = wizards.design_lens("pcx", 50.0, matdb=db, material="bk7",
                              lam_nm=633.0)
    assert out["design"]["n"] == pytest.approx(N_633, abs=2e-4)
    out2 = wizards.design_lens("ball", 5.870, matdb=db, material="bk7")
    assert out2["params"]["diameter"] == pytest.approx(8.0, abs=0.05)


# ---------------------------------------------------------------------------
# Waveplate thickness solver (primitivelib 'waveplate' kind's default quartz
# retarder) -- oracle: at 633nm zero-order half-wave quartz thickness is
# computed FROM the real birefringence registry (not a hardcoded constant),
# and must reproduce exactly 0.5 waves of retardance by construction.
# ---------------------------------------------------------------------------
def _real_optprops_matdb():
    from raytracer.optprops import load_optical_properties
    return load_optical_properties().matdb


def test_waveplate_half_wave_quartz_633nm_oracle():
    db = _real_optprops_matdb()
    out = wizards.waveplate_thickness("half", 633.0, matdb=db)
    n_o, n_e = wizards.birefringence_at(db, "quartz", 633.0)
    assert out["waves"] == pytest.approx(0.5, rel=1e-12)
    # zero-order half-wave quartz at 633nm is ~35 um (Ghosh 1999 Sellmeier
    # coefficients for quartz_o/quartz_e, per opticalproperties/materials.miemat)
    assert out["thickness"] * 1000.0 == pytest.approx(35.0, abs=1.0)
    assert out["n_o"] == pytest.approx(n_o)
    assert out["n_e"] == pytest.approx(n_e)
    assert out["n_e"] > out["n_o"]     # quartz is positive uniaxial


def test_waveplate_quarter_wave_is_half_of_half_wave_thickness():
    db = _real_optprops_matdb()
    half = wizards.waveplate_thickness("half", 633.0, matdb=db)
    quarter = wizards.waveplate_thickness("quarter", 633.0, matdb=db)
    assert quarter["waves"] == pytest.approx(0.25, rel=1e-12)
    assert quarter["thickness"] == pytest.approx(half["thickness"] / 2.0,
                                                 rel=1e-9)


def test_waveplate_order_adds_whole_waves():
    db = _real_optprops_matdb()
    zero_order = wizards.waveplate_thickness("half", 633.0, order=0, matdb=db)
    first_order = wizards.waveplate_thickness("half", 633.0, order=1, matdb=db)
    assert first_order["waves"] == pytest.approx(1.5, rel=1e-12)
    assert first_order["thickness"] == pytest.approx(
        3.0 * zero_order["thickness"], rel=1e-9)


def test_waveplate_default_matdb_lazy_load_matches_explicit():
    db = _real_optprops_matdb()
    explicit = wizards.waveplate_thickness("half", 633.0, matdb=db)
    default = wizards.waveplate_thickness("half", 633.0)   # no matdb given
    assert default["thickness"] == pytest.approx(explicit["thickness"])


def test_waveplate_bad_kind_and_order_raise():
    with pytest.raises(ValueError):
        wizards.waveplate_thickness("eighth", 633.0)
    with pytest.raises(ValueError):
        wizards.waveplate_thickness("half", 633.0, order=-1)
    with pytest.raises(ValueError):
        wizards.waveplate_thickness("half", 633.0, order=0.5)


########################################################################
# Gaussian-beam source extras (beam_waist/m2/apodization) -- lowhanging
# round: wizard fields for the new source props, writing body properties
# through the same changed_props()/set_property() path as power/lambdac/
# polarization.
########################################################################
def _laser_info():
    import json
    repo = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                         "..", ".."))
    meta_path = os.path.join(repo, "primitives", "laser_collimated.meta.json")
    with open(meta_path) as fh:
        info = json.load(fh)
    info["path"] = os.path.join(repo, "primitives", "laser_collimated.FCStd")
    return info


def test_property_rows_for_source_includes_beam_fields():
    from mieworkbench.panes.element_wizard import property_rows_for
    rows = property_rows_for(_laser_info())
    by_name = {name: (kind, default) for name, kind, default, _tip in rows}
    assert by_name["beam_waist"] == ("float_optional", None)
    assert by_name["m2"] == ("float", 1.0)
    assert by_name["apodization"] == ("apod", None)


def test_apodization_editor_round_trip(qtbot):
    from mieworkbench.panes.element_wizard import ApodizationEditor
    ed = ApodizationEditor(None)
    qtbot.addWidget(ed)
    assert ed.spec() is None
    assert ed.kind_combo.currentText() == "none"
    assert not ed.w0_edit.isEnabled()

    ed.kind_combo.setCurrentText("gaussian")
    assert ed.w0_edit.isEnabled()
    ed.w0_edit.setText("2")
    assert ed.spec() == "gaussian:w0=2"
    ed.order_edit.setText("3")
    assert ed.spec() == "gaussian:w0=2:order=3"

    ed2 = ApodizationEditor("gaussian:w0=1.5:order=2")
    qtbot.addWidget(ed2)
    assert ed2.kind_combo.currentText() == "gaussian"
    assert ed2.w0_edit.text() == "1.5"
    assert ed2.order_edit.text() == "2"
    assert ed2.spec() == "gaussian:w0=1.5:order=2"

    ed2.kind_combo.setCurrentText("none")
    assert ed2.spec() is None
    assert not ed2.w0_edit.isEnabled()


def test_properties_form_beam_waist_blank_omits_from_values(qtbot):
    from mieworkbench.panes.element_wizard import (
        PropertiesFormWidget, property_rows_for,
    )
    rows = property_rows_for(_laser_info())
    form = PropertiesFormWidget(rows)
    qtbot.addWidget(form)

    # defaults: waist blank/off -> beam_waist and apodization absent, m2
    # gated off (no waist set yet)
    values = form.values()
    assert "beam_waist" not in values
    assert "apodization" not in values
    _kind, m2_editor, _default = form._editors["m2"]
    assert not m2_editor.isEnabled()

    # setting a waist both writes the value AND enables m2
    _kind, bw_editor, _default = form._editors["beam_waist"]
    bw_editor.setText("1.5")
    assert m2_editor.isEnabled()
    assert form.values()["beam_waist"] == pytest.approx(1.5)

    # clearing it again turns m2 back off and drops the key
    bw_editor.setText("")
    assert not m2_editor.isEnabled()
    assert "beam_waist" not in form.values()


def test_wizard_dialog_beam_and_apodization_round_trip(qtbot, tmp_path):
    """The wizard writes beam_waist/m2/apodization through the SAME
    changed_props() path power/lambdac/polarization already use (verified
    by feeding the result into a FakeProject and checking its call
    audit -- see vtk_test_support.FakeProject)."""
    from mieworkbench.panes.wizard_dialog import ElementWizardDialog
    from mieworkbench.tests.vtk_test_support import (
        FakeProject, make_two_body_scene,
    )

    dlg = ElementWizardDialog(_laser_info(), "laser1")
    qtbot.addWidget(dlg)

    # nothing changed yet -- beam extras are off by default
    changed = dlg.changed_props()
    assert "beam_waist" not in changed
    assert "m2" not in changed
    assert "apodization" not in changed

    _kind, bw_editor, _default = dlg.props_form._editors["beam_waist"]
    bw_editor.setText("2")
    _kind, m2_editor, _default = dlg.props_form._editors["m2"]
    m2_editor.setText("1.2")
    _kind, apod_editor, _default = dlg.props_form._editors["apodization"]
    apod_editor.kind_combo.setCurrentText("gaussian")
    apod_editor.w0_edit.setText("2")

    changed = dlg.changed_props()
    assert changed["beam_waist"] == pytest.approx(2.0)
    assert changed["m2"] == pytest.approx(1.2)
    assert changed["apodization"] == "gaussian:w0=2"

    # feed it through the same set_property mechanism _apply_wizard_output
    # uses in mainwindow.py, and confirm it lands on the body + the call
    # audit sees it (same shape as every other wizard-written property)
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    for name, value in changed.items():
        project.set_property("Lens", name, value)
    calls_by_name = {c[2]: c[3] for c in project.calls
                    if c[0] == "set_property"}
    assert calls_by_name["beam_waist"] == pytest.approx(2.0)
    assert calls_by_name["apodization"] == "gaussian:w0=2"
    assert project.body("Lens")["properties"]["apodization"]["value"] \
        == "gaussian:w0=2"


def test_waveplate_designer_fills_thickness(qtbot):
    """The waveplate wizard's retardance designer writes the solved
    quartz thickness into the parameter table."""
    import json
    from mieworkbench.panes.wizard_dialog import ElementWizardDialog
    repo = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                         "..", ".."))
    meta_path = os.path.join(repo, "primitives", "waveplate.meta.json")
    with open(meta_path) as fh:
        info = json.load(fh)
    info["path"] = os.path.join(repo, "primitives", "waveplate.FCStd")
    dlg = ElementWizardDialog(info, "wp1")
    qtbot.addWidget(dlg)
    dlg.wp_lambda.setText("633")
    dlg.wp_order.setText("0")
    dlg.wp_kind.setCurrentIndex(0)      # half-wave
    dlg._compute_waveplate()
    from mieworkbench.core.wizards import waveplate_thickness
    expected = waveplate_thickness("half", 633.0)["thickness"]
    assert dlg.params()["thickness"] == pytest.approx(expected, rel=1e-4)
    assert "mm" in dlg.wp_out.text()
