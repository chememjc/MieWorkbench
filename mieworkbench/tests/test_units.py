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


def test_format_power_mw():
    assert format_power_mw(0.005) == "5"
    assert format_power_mw(1.23e-6) == "0.00123"
