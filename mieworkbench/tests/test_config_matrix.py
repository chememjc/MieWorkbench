"""ConfigMatrix unit tests: widget-kind introspection, values()/to_args()
non-default filtering, JSON round-trip, and preset-driven placeholders."""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

import cli_specs  # noqa: E402  (stdlib-only)
from PySide6.QtWidgets import QCheckBox, QComboBox, QSpinBox  # noqa: E402

from mieworkbench.panes.config_matrix import ConfigMatrix  # noqa: E402


def test_known_widget_kinds(qtbot):
    matrix = ConfigMatrix()
    qtbot.addWidget(matrix)

    assert isinstance(matrix.widgets["seeds"], QSpinBox)

    backend = matrix.widgets["backend"]
    assert isinstance(backend, QComboBox)
    items = {backend.itemText(i) for i in range(backend.count())}
    assert {"auto", "torch", "numpy"} <= items

    assert isinstance(matrix.widgets["dry_run"], QCheckBox)


def test_values_default_is_empty(qtbot):
    matrix = ConfigMatrix()
    qtbot.addWidget(matrix)
    assert matrix.values() == {}
    assert matrix.to_args() == []


def test_values_to_args_only_nondefault(qtbot):
    matrix = ConfigMatrix()
    qtbot.addWidget(matrix)

    matrix.widgets["seeds"].setValue(3)
    matrix.widgets["backend"].setCurrentText("numpy")
    matrix.widgets["dry_run"].setChecked(True)
    matrix.widgets["source_face"].setText(
        "Body1.Pad.Face1;Body2.Pad.Face2")

    args = matrix.to_args()

    assert args.count("--seeds") == 1
    assert args[args.index("--seeds") + 1] == "3"
    assert args[args.index("--backend") + 1] == "numpy"
    assert "--dry-run" in args
    sf_positions = [i for i, a in enumerate(args) if a == "--source-face"]
    assert [args[i + 1] for i in sf_positions] == [
        "Body1.Pad.Face1", "Body2.Pad.Face2"]

    # untouched fields never appear
    assert "--rays" not in args
    assert "--resolution" not in args
    assert "--keep-going" not in args

    # and the real pipeline parser reproduces exactly what was set
    parser = cli_specs.build_parser("pipeline")
    ns = parser.parse_args(["--models", "x.FCStd"] + args)
    assert ns.seeds == 3
    assert ns.backend == "numpy"
    assert ns.dry_run is True
    assert ns.source_face == ["Body1.Pad.Face1", "Body2.Pad.Face2"]


def test_json_round_trip(qtbot):
    matrix = ConfigMatrix()
    qtbot.addWidget(matrix)

    matrix.widgets["rays"].setText("200000")
    matrix.widgets["resolution"].setValue(256)
    matrix.widgets["grating"].setText("Body.Obj.Face1:600:v")
    matrix.preset_combo.setCurrentText("detailed")

    values_before = matrix.values()
    text = matrix.to_json()

    matrix2 = ConfigMatrix()
    qtbot.addWidget(matrix2)
    matrix2.from_json(text)

    assert matrix2.values() == values_before


def test_preset_change_updates_placeholder_not_values(qtbot):
    matrix = ConfigMatrix()
    qtbot.addWidget(matrix)

    rays_widget = matrix.widgets["rays"]
    resolution_widget = matrix.widgets["resolution"]
    assert rays_widget.text() == ""
    assert resolution_widget.value() == 0
    initial_placeholder = rays_widget.placeholderText()

    matrix.preset_combo.setCurrentText("detailed")

    # value still unset - preset only touched the placeholder
    assert rays_widget.text() == ""
    assert resolution_widget.value() == 0
    assert rays_widget.placeholderText() != initial_placeholder
    assert "rays" not in matrix.values()
    assert "resolution" not in matrix.values()

def test_dim_rays_options_render_and_round_trip(qtbot):
    """The attenuation-dimming pipeline options are reachable from the GUI
    run config: a mode combo and a floor line edit, omitted at defaults,
    forwarded as pipeline argv when set."""
    from PySide6.QtWidgets import QLineEdit

    matrix = ConfigMatrix()
    qtbot.addWidget(matrix)

    dim = matrix.widgets["dim_rays"]
    assert isinstance(dim, QComboBox)
    items = {dim.itemText(i) for i in range(dim.count())}
    assert {"off", "linear", "sqrt"} <= items
    assert dim.currentText() == "off"
    assert isinstance(matrix.widgets["dim_rays_floor"], QLineEdit)

    # defaults stay out of simparams/argv
    assert "dim_rays" not in matrix.values()
    assert "dim_rays_floor" not in matrix.values()

    dim.setCurrentText("linear")
    matrix.widgets["dim_rays_floor"].setText("5")
    values = matrix.values()
    assert values["dim_rays"] == "linear"
    assert values["dim_rays_floor"] == 5.0
    args = matrix.to_args()
    assert "--dim-rays" in args
    assert args[args.index("--dim-rays") + 1] == "linear"
    assert "--dim-rays-floor" in args

def test_reset_to_defaults_round_trip(qtbot):
    """A reset widget must be indistinguishable from a fresh one — the
    session-boundary guarantee that a previous project's run config
    can't leak into the next."""
    matrix = ConfigMatrix()
    qtbot.addWidget(matrix)
    matrix.preset_combo.setCurrentText("detailed")
    matrix.widgets["seeds"].setValue(7)
    matrix.widgets["backend"].setCurrentText("numpy")
    matrix.widgets["dry_run"].setChecked(True)
    matrix.widgets["rays"].setText("123456")
    matrix.widgets["source_face"].setText("Body.Pad.Face1")
    assert matrix.values() != {}

    matrix.reset_to_defaults()
    assert matrix.values() == {}
    assert matrix.to_args() == []
    assert matrix.preset_combo.currentText() == "quick"


def test_save_fields_top_row_checkbox(qtbot):
    """--save-fields gets a dedicated, always-visible top-row checkbox next
    to Preset (not the generic physics-options group it would otherwise
    fall into) but must still round-trip through the standard widgets/
    values()/set_values()/reset_to_defaults()/estimate_params() machinery
    exactly like every other flag."""
    matrix = ConfigMatrix()
    qtbot.addWidget(matrix)

    assert isinstance(matrix.save_fields_check, QCheckBox)
    assert matrix.widgets["save_fields"] is matrix.save_fields_check
    assert not matrix.save_fields_check.isChecked()
    assert "save_fields" not in matrix.values()
    assert matrix.estimate_params()["save_fields"] is False

    matrix.save_fields_check.setChecked(True)
    assert matrix.values()["save_fields"] is True
    assert matrix.estimate_params()["save_fields"] is True
    args = matrix.to_args()
    assert "--save-fields" in args

    text = matrix.to_json()
    matrix2 = ConfigMatrix()
    qtbot.addWidget(matrix2)
    matrix2.from_json(text)
    assert matrix2.save_fields_check.isChecked()

    matrix.reset_to_defaults()
    assert not matrix.save_fields_check.isChecked()
    assert "save_fields" not in matrix.values()


def test_product_flags_render_as_checkbox_rows(qtbot):
    # pulsed-optics P11: --time-products / --imaging-products get one
    # checkbox per product (cli_specs.PRODUCT_FLAG_CHOICES) instead of a
    # free-text line; values()/set_values round-trip the comma string and
    # to_args forwards it verbatim
    from mieworkbench.panes.config_matrix import ProductChecks
    matrix = ConfigMatrix()
    qtbot.addWidget(matrix)
    for dest, (products, allow_none) in \
            cli_specs.PRODUCT_FLAG_CHOICES.items():
        w = matrix.widgets[dest]
        assert isinstance(w, ProductChecks)
        assert set(w.checks) == set(products)
        assert (w.none_check is not None) == allow_none
    tp = matrix.widgets["time_products"]
    tp.checks["cube"].setChecked(True)
    tp.checks["pulse"].setChecked(True)
    vals = matrix.values()
    assert vals["time_products"] == "pulse,cube"   # canonical order
    args = matrix.to_args()
    i = args.index("--time-products")
    assert args[i + 1] == "pulse,cube"
    # the real pipeline parser accepts what the widget produced
    ns = cli_specs.build_parser("pipeline").parse_args(
        ["--models", "x.FCStd"] + args)
    assert ns.time_products == ("pulse", "cube")
    # round-trip through set_values
    matrix.reset_to_defaults()
    assert matrix.values() == {}
    matrix.set_values(vals)
    assert matrix.values()["time_products"] == "pulse,cube"
    # 'none' is exclusive: it unchecks and disables the product boxes
    tp.none_check.setChecked(True)
    assert matrix.values()["time_products"] == "none"
    assert not tp.checks["pulse"].isEnabled()
    tp.none_check.setChecked(False)
    assert tp.checks["pulse"].isEnabled()
