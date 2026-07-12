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


def test_new_pulsed_source_props_have_unit_decisions():
    """Pulsed-optics Phase P3: pulse_energy/pulse_duration/rep_rate (source
    body properties) need explicit unit decisions, same as every other
    contract property."""
    assert label_with_unit("pulse_energy") == "pulse_energy [µJ]"
    assert label_with_unit("pulse_duration") == "pulse_duration [ps]"
    assert label_with_unit("rep_rate") == "rep_rate [Hz]"


def test_format_power_mw():
    assert format_power_mw(0.005) == "5"
    assert format_power_mw(1.23e-6) == "0.00123"


def test_new_nlo_props_have_unit_decisions():
    """Pulsed-optics Phase P8: Pockels cell (nonlinear/pockels_voltage/
    pockels_gap) + saturable absorber (saturable) + two-photon absorption
    (tpa_beta) + Kerr thin lens (kerr_n2) need explicit unit decisions,
    same as every other contract property. These are not (yet) wired
    into element_editor.CONTRACT_PROPERTIES (same anticipatory-unit-entry
    pattern P3 used for pulse_energy/pulse_duration/rep_rate before those
    landed in the editor) -- test_every_contract_property_has_a_unit_
    decision only asserts the OTHER direction (every CONTRACT_PROPERTIES
    entry has a unit), so this test pins the new keys directly."""
    assert label_with_unit("nonlinear") == "nonlinear"          # registry name
    assert label_with_unit("pockels_voltage") == "pockels_voltage [V]"
    assert label_with_unit("pockels_gap") == "pockels_gap [mm]"
    assert label_with_unit("saturable") == "saturable"          # spec string
    assert label_with_unit("tpa_beta") == "tpa_beta [cm/GW]"
    assert label_with_unit("kerr_n2") == "kerr_n2"               # spec string
