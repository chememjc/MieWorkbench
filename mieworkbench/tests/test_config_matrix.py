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
