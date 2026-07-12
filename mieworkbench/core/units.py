"""Display units for the optical body-tagging contract and GUI panels.

Single source of truth for what unit each contract property is expressed
in (docs/RAYTRACER.md §5 is the authority; these strings are display-only
and never converted -- the engine reads the raw values). Properties whose
value is a name, spec string, or flag map to None and render without a
unit suffix.
"""

# Every property in element_editor.CONTRACT_PROPERTIES must have an entry
# here (pinned by tests/test_units.py); None = unitless/spec-valued.
PROPERTY_UNITS = {
    "material": None,             # registry name / 'detector' / 'none'
    "power": "mW",
    "lambdac": "nm",
    "lambdamin": "nm",
    "lambdamax": "nm",
    "coherent": None,             # bool
    "spectrum": None,             # emission-registry name (tabulated SPD)
    "polarization": None,         # spec string
    "coating": None,              # registry name or per-face map
    "roughness": "nm RMS",
    "diffuser": None,             # spec string (grit:/slope:/@registry)
    "filter": None,               # registry name
    "polarizer": None,            # registry name
    "polarizer_axis": "x,y,z",    # body-local axis triple
    "crystal_axis": "x,y,z",
    "crystal_axis2": "x,y,z",     # biaxial Y principal axis (X = crystal_axis)
    "grating": None,              # per-face map spec
    "surface_override": None,     # per-face analytic-surface spec
    "mirror": "0–1",         # reflected fraction
    "absorbance": "0–1",     # absorbed fraction
    "qe_curve": None,             # detector QE-curve registry name
    "beam_waist": "mm",           # Gaussian beam waist w0 (source-only)
    "m2": None,                    # beam quality factor M^2 (unitless, >=1)
    "apodization": None,          # spec string ('gaussian:w0=<mm>[:order=<n>]')
    "scatter": None,               # ABg/BSDF registry name or per-face map
    "detector_face": None,         # FaceN pin for the detector screen face
    "pulse_energy": "µJ",          # per-pulse energy (source-only)
    "pulse_duration": "ps",        # pulse FWHM duration (source-only)
    "rep_rate": "Hz",              # pulse repetition rate (source-only)
    # Pulsed-optics Phase P8: Pockels cell / saturable absorber / TPA / Kerr.
    "nonlinear": None,              # nonlinear.mienlo registry name (pockels/chi2)
    "pockels_voltage": "V",         # Pockels cell applied voltage
    "pockels_gap": "mm",            # transverse-geometry electrode gap d (E=V/d)
    "saturable": None,              # registry name or inline 'sat:I_sat=..:T0=..'
    "tpa_beta": "cm/GW",            # two-photon-absorption coefficient
    "kerr_n2": None,                # registry name or inline 'n2:<m2/W>'
}

# Geometry/transform panels (values live in FreeCAD sheets / placements).
LENGTH_UNIT = "mm"
ANGLE_UNIT = "deg"


def label_with_unit(name):
    """'power' -> 'power [mW]'; unitless names pass through unchanged."""
    unit = PROPERTY_UNITS.get(name)
    return "%s [%s]" % (name, unit) if unit else name


def format_power_mw(power_w):
    """Watts -> display string in mW (results/power-table formatting)."""
    return "%.6g" % (float(power_w) * 1e3)
