"""GUI-surface tests for the thermo-optic temperature feature:
- the scene-global --temperature flag auto-surfaces in the Run config matrix
  (proving the cli_specs -> ConfigMatrix auto-wiring), and
- the per-body 'temperature' override is a first-class element-editor property.
So neither half of the feature is CLI-only.
"""
import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

from mieworkbench.panes.config_matrix import ConfigMatrix  # noqa: E402
from mieworkbench.panes import element_editor as ee  # noqa: E402


# --- scene-global --temperature auto-surfaces in the Run config matrix -------
def test_temperature_auto_surfaces_in_config_matrix(qtbot):
    matrix = ConfigMatrix()
    qtbot.addWidget(matrix)
    assert "temperature" in matrix.widgets, \
        "--temperature did not auto-surface as a Run-config widget"


def test_temperature_round_trips_to_args(qtbot):
    matrix = ConfigMatrix()
    qtbot.addWidget(matrix)
    w = matrix.widgets["temperature"]
    # float args render as a QLineEdit in ConfigMatrix; set text + read back
    w.setText("55")
    args = matrix.to_args()
    assert "--temperature" in args
    i = args.index("--temperature")
    assert float(args[i + 1]) == 55.0
    # default (blank) must not emit the flag
    w.setText("")
    assert "--temperature" not in matrix.to_args()


# --- per-body 'temperature' override is a real element-editor property -------
def test_temperature_is_a_numeric_contract_property():
    assert "temperature" in ee.CONTRACT_PROPERTIES
    assert "temperature" in ee.NUMERIC_PROPERTIES
    assert ee.PROPERTY_DEFAULTS["temperature"] == 20.0
    # a tooltip exists so the pane never shows a bare/undocumented row
    assert "temperature" in ee.TOOLTIPS
    assert "dn/dT" in ee.TOOLTIPS["temperature"]
