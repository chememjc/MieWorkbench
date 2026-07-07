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
    "polarization": None,         # spec string
    "coating": None,              # registry name or per-face map
    "roughness": "nm RMS",
    "diffuser": None,             # spec string (grit:/slope:/@registry)
    "filter": None,               # registry name
    "polarizer": None,            # registry name
    "polarizer_axis": "x,y,z",    # body-local axis triple
    "crystal_axis": "x,y,z",
    "grating": None,              # per-face map spec
    "surface_override": None,     # per-face analytic-surface spec
    "mirror": "0–1",         # reflected fraction
    "absorbance": "0–1",     # absorbed fraction
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
