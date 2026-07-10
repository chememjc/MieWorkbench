"""core/units.py -- the display-unit map must stay complete: every
contract property the element editor can add needs an explicit entry
(None = deliberately unitless), so a newly added contract property can't
silently ship without a unit decision."""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.units import (  # noqa: E402
    PROPERTY_UNITS, format_power_mw, label_with_unit,
)
from mieworkbench.panes.element_editor import (  # noqa: E402
    CONTRACT_PROPERTIES,
)


def test_every_contract_property_has_a_unit_decision():
    missing = [p for p in CONTRACT_PROPERTIES if p not in PROPERTY_UNITS]
    assert not missing, "no unit decision for: %s" % missing


def test_label_with_unit():
    assert label_with_unit("power") == "power [mW]"
    assert label_with_unit("lambdac") == "lambdac [nm]"
    assert label_with_unit("material") == "material"      # unitless
    assert label_with_unit("not_a_prop") == "not_a_prop"  # unknown passes


def test_new_biaxial_apodization_scatter_props_have_unit_decisions():
    """Lowhanging round: beam_waist/m2/apodization (Gaussian-beam source
    extras), crystal_axis2 (biaxial Y principal axis), and scatter
    (ABg/BSDF per-face registry) all need explicit unit decisions, same
    as every other contract property."""
    assert label_with_unit("beam_waist") == "beam_waist [mm]"
    assert label_with_unit("m2") == "m2"                  # unitless
    assert label_with_unit("apodization") == "apodization"  # spec string
    assert label_with_unit("crystal_axis2") == "crystal_axis2 [x,y,z]"
    assert label_with_unit("scatter") == "scatter"        # registry name


def test_format_power_mw():
    assert format_power_mw(0.005) == "5"
    assert format_power_mw(1.23e-6) == "0.00123"
