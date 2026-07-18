"""libschema.py -- per-column documentation + advisory validation for the
optical property library registries edited by panes/prop_editor.py.

This is a PURE-DATA module (no Qt, no I/O): COLUMN_SCHEMA describes every
column of every registry PropLibrary/core.proplib.CATEGORY_INFO exposes
(materials, coatings, polarizers, filters, gratings, uniaxial, diffusers,
detectors, emission); TABLE_COLUMN_SCHEMA does the same for the per-row
spectral TABLE files those registries reference (the tuples in
panes/prop_editor.py's TABLE_SCHEMA/CATEGORY_TABS).

Content is distilled from three sources that must agree (and the
drift-proofing test in mieworkbench/tests/test_libschema.py enforces it
against the live registries so this module can't silently go stale):
  - docs/RAYTRACER.md Sec.7 (the human-facing authoring contract)
  - scripts/raytracer/optprops.py / materials.py (the loaders -- the
    actual hard-validation rules; ANY conflict between the docs and the
    loader code defers to the loader, since that's what actually runs)
  - the live opticalproperties/ registry CSVs (to confirm every column
    that ships today is documented, including ones the prose docs don't
    call out by name, e.g. the P9 gyration_* / k_x,y,z optional columns)

NOT covered here (out of PropLibrary's CATEGORY_INFO -- proplib.py has no
registry_rel/file_dir entry for them, so the editor pane cannot open them
today): birefringence/biaxial.mibiax, figure/figures.miefig,
nonlinear/nonlinear.mienlo, scatter/bsdf.miebsdf,
instrument/instruments.mieinst. If any of those categories is ever wired
into CATEGORY_INFO, extend COLUMN_SCHEMA (and the drift test will start
enforcing it automatically).

Validation is advisory only (Sec.4 of the round brief this module was
written for): the loaders in optprops.py/materials.py are the one hard
gate, run at commit time. A `validator` spec here just tells the GUI what
"looks wrong" well before that -- the same rule may be re-checked (more
strictly) by the real loader on Save.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ColumnInfo:
    """One column's documentation + optional advisory-validator spec.

    `validator`, if set, is a dict with a `kind` key:
      {"kind": "float", "gt": 0.0}         -- must parse as float, > gt
      {"kind": "float", "ge": 0.0}         -- must parse as float, >= ge
      {"kind": "float"}                    -- must parse as float
      {"kind": "float", "range": (lo, hi)} -- must parse as float AND fall
                                               in [lo, hi] (a *plausible*
                                               physical range, e.g. a
                                               wavelength in nm -- not a
                                               hard registry rule)
      {"kind": "int", "gt": 0}             -- must parse as int, > gt
      {"kind": "enum", "values": (...)}    -- must be one of `values`
                                               (case-sensitive, matches the
                                               loader's own comparison)
    Every validator only fires on a NON-BLANK cell -- optional columns are
    never flagged for being empty (only the reference/citation column has
    that separate, pre-existing required-ness rule in prop_editor.py).
    """
    description: str
    units: str = ""
    format: str = ""
    validator: Optional[dict] = None


# ---------------------------------------------------------------------------
# shared column text (columns that mean the same thing in every registry)
# ---------------------------------------------------------------------------
def _name(what):
    return ColumnInfo(
        "Unique row identifier; how bodies/other rows reference this %s "
        "(e.g. a body's `material` property, or a coating's `layers` "
        "field)." % what,
        units="", format="unique string, case-insensitive lookup")


_REFERENCE = ColumnInfo(
    "Citation for where this row's data came from (datasheet, paper, "
    "catalog). Required on every row in every registry -- the loader "
    "hard-errors on a blank reference; this is the one column the editor "
    "already refuses to save empty.",
    units="", format="citation string (non-empty)")

_NOTES = ColumnInfo(
    "Free-text annotation -- spot-check results, caveats, provenance "
    "detail that doesn't fit `reference`. Never validated by the loader.",
    units="", format="free text (optional)")

_TABLE_CSV = lambda subdir: ColumnInfo(     # noqa: E731
    "Filename of the per-row spectral table in `%s/` (new-extension "
    ".mietab preferred, legacy .csv sibling still resolved). The table's "
    "own required columns are documented in TABLE_COLUMN_SCHEMA for this "
    "category." % subdir,
    units="", format="filename (e.g. 'name.mietab')")


# ---------------------------------------------------------------------------
# materials.miemat
# ---------------------------------------------------------------------------
_MATERIALS = {
    "name": _name("material"),
    "class": ColumnInfo(
        "Organizational category. Not cross-validated against any "
        "electrical/physical role -- documentation only.",
        format="enum: gas | glass | liquid | polymer | metal | oxide | "
               "film | special",
        validator={"kind": "enum",
                   "values": ("gas", "glass", "liquid", "polymer", "metal",
                              "oxide", "film", "special")}),
    "model": ColumnInfo(
        "Dispersion model used to evaluate n(lambda) [+ k(lambda)]. "
        "Determines which of p1..p6 / nk_file are consulted.",
        format="enum: sellmeier | schott | cauchy | constant | tabulated",
        validator={"kind": "enum",
                   "values": ("sellmeier", "schott", "cauchy", "constant",
                              "tabulated")}),
    "p1": ColumnInfo(
        "Model parameter 1. sellmeier: B1. schott: a0. cauchy: A. "
        "constant: n.",
        format="float (meaning depends on `model`)",
        validator={"kind": "float"}),
    "p2": ColumnInfo(
        "Model parameter 2. sellmeier: B2. schott: a1. cauchy: B "
        "(coefficient of 1/lambda_um^2). constant: k (imaginary index, "
        "0 if blank).",
        format="float (meaning depends on `model`)",
        validator={"kind": "float"}),
    "p3": ColumnInfo(
        "Model parameter 3. sellmeier: B3. schott: a2 (coefficient of "
        "1/lambda_um^2). cauchy: C (coefficient of 1/lambda_um^4).",
        format="float (meaning depends on `model`)",
        validator={"kind": "float"}),
    "p4": ColumnInfo(
        "Model parameter 4. sellmeier: C1 (must be finite; a negative or "
        "zero value is a legitimate fit, not an error). schott: a3 "
        "(coefficient of 1/lambda_um^4).",
        format="float, finite (meaning depends on `model`)",
        validator={"kind": "float"}),
    "p5": ColumnInfo(
        "Model parameter 5. sellmeier: C2 (finite). schott: a4 "
        "(coefficient of 1/lambda_um^6).",
        format="float, finite (meaning depends on `model`)",
        validator={"kind": "float"}),
    "p6": ColumnInfo(
        "Model parameter 6. sellmeier: C3 (finite). schott: a5 "
        "(coefficient of 1/lambda_um^8).",
        format="float, finite (meaning depends on `model`)",
        validator={"kind": "float"}),
    "nk_file": ColumnInfo(
        "Filename of a tabulated n,k spectrum in `nk/` (new-extension "
        ".mienk preferred, legacy .csv fallback). REQUIRED when "
        "model=tabulated; ignored otherwise.",
        format="filename (e.g. 'gold.mienk') -- required iff model=tabulated"),
    "density_kg_m3": ColumnInfo(
        "Mass density, used by particle-cloud mass-fraction bookkeeping. "
        "Must be > 0 for every material except `vacuum`/`detector`, which "
        "may be exactly 0 (sentinel).",
        units="kg/m^3", format="float > 0 (0 allowed only for vacuum/detector)",
        validator={"kind": "float", "ge": 0.0}),
    "transmission_um_min": ColumnInfo(
        "Advisory lower bound of the material's usable transmission "
        "window for a PARAMETRIC model (sellmeier/schott/cauchy/constant); "
        "evaluating n_complex() below this warns but does not raise. "
        "Not applied to a tabulated nk_file, whose own range is a hard "
        "limit instead.",
        units="um", format="float, optional; must be < transmission_um_max",
        validator={"kind": "float", "gt": 0.0}),
    "transmission_um_max": ColumnInfo(
        "Advisory upper bound of the material's usable transmission "
        "window (see transmission_um_min).",
        units="um", format="float, optional; must be > transmission_um_min",
        validator={"kind": "float", "gt": 0.0}),
    "notes": _NOTES,
    "reference": _REFERENCE,
    "thermo_d0": ColumnInfo(
        "Schott TIE-19 thermo-optic coefficient D0 (linear dn/dT term). "
        "Optional -- absent means no thermal index shift is applied even "
        "if a run sets --temperature.",
        format="float, optional (Schott TIE-19)", validator={"kind": "float"}),
    "thermo_d1": ColumnInfo(
        "Schott TIE-19 thermo-optic coefficient D1 (quadratic dT^2 term).",
        format="float, optional (Schott TIE-19)", validator={"kind": "float"}),
    "thermo_d2": ColumnInfo(
        "Schott TIE-19 thermo-optic coefficient D2 (cubic dT^3 term).",
        format="float, optional (Schott TIE-19)", validator={"kind": "float"}),
    "thermo_e0": ColumnInfo(
        "Schott TIE-19 thermo-optic coefficient E0 (dispersive dT term, "
        "divides by lambda_um^2 - lambda_tk^2).",
        format="float, optional (Schott TIE-19)", validator={"kind": "float"}),
    "thermo_e1": ColumnInfo(
        "Schott TIE-19 thermo-optic coefficient E1 (dispersive dT^2 term).",
        format="float, optional (Schott TIE-19)", validator={"kind": "float"}),
    "thermo_lambda_tk": ColumnInfo(
        "Schott TIE-19 'TK' reference wavelength used in the E0/E1 "
        "dispersive denominator (lambda_um^2 - lambda_tk^2).",
        units="um", format="float, optional (Schott TIE-19)",
        validator={"kind": "float", "gt": 0.0}),
    "thermo_t_ref_c": ColumnInfo(
        "Reference temperature the thermo_* coefficients are measured "
        "relative to (dT = T - t_ref). Defaults to 20 C if blank.",
        units="degC", format="float, optional (default 20)",
        validator={"kind": "float"}),
}

# ---------------------------------------------------------------------------
# coating/coatings.miecoat
# ---------------------------------------------------------------------------
_COATINGS = {
    "name": _name("coating"),
    "layers": ColumnInfo(
        "TMM (thin-film characteristic-matrix) layer stack, incident side "
        "toward the substrate. Exactly one of `layers`/`table` must be "
        "set per row.",
        format="';'-separated 'material:thickness_spec' terms, e.g. "
               "'mgf2:qw@550' or 'ta2o5:100.0;sio2:150.0' -- thickness_spec "
               "is a literal nm thickness or 'qw@<lam0_nm>' (dispersive "
               "quarter-wave at that design wavelength)"),
    "table": _TABLE_CSV("coating/tables"),
    "aoi_deg": ColumnInfo(
        "Angle of incidence the measured `table` was taken at. Only "
        "meaningful when `table` is set (ignored for `layers` rows, which "
        "compute R/T at the ray's actual AOI via TMM).",
        units="deg", format="float, optional (default 0 if table set)",
        validator={"kind": "float", "range": (0.0, 90.0)}),
    "reference": _REFERENCE,
}

# ---------------------------------------------------------------------------
# polarizer/polarizers.miepol
# ---------------------------------------------------------------------------
_POLARIZERS = {
    "name": _name("polarizer"),
    "type": ColumnInfo(
        "Polarizer kind -- controls how the Jones diattenuator/retarder "
        "stage is built from the T_parallel/T_perpendicular table.",
        format="enum: linear | circular_left | circular_right",
        validator={"kind": "enum",
                   "values": ("linear", "circular_left", "circular_right")}),
    "table_csv": _TABLE_CSV("polarizer/tables"),
    "retardance_waves": ColumnInfo(
        "Waveplate retardance bundled with a circular polarizer (a real "
        "circular polarizer is a linear sheet + quarter-wave film). "
        "Defaults to 0.25 (quarter-wave) if blank.",
        units="waves", format="float, optional (default 0.25)",
        validator={"kind": "float", "gt": 0.0}),
    "reference": _REFERENCE,
}

# ---------------------------------------------------------------------------
# filter/filters.miefilt
# ---------------------------------------------------------------------------
_FILTERS = {
    "name": _name("filter"),
    "table_csv": _TABLE_CSV("filter/tables"),
    "ref_thickness_mm": ColumnInfo(
        "Physical thickness the tabulated transmittance_internal curve "
        "was measured at; a body's actual path length is scaled against "
        "this via Beer-Lambert (tau(d2) = tau(d1)^(d2/d1)).",
        units="mm", format="float > 0",
        validator={"kind": "float", "gt": 0.0}),
    "reference": _REFERENCE,
}

# ---------------------------------------------------------------------------
# grating/gratings.miegrat
# ---------------------------------------------------------------------------
_GRATINGS = {
    "name": _name("grating"),
    "model": ColumnInfo(
        "Diffraction model. Determines which `params` keys are required "
        "and whether `table_csv` is consulted.",
        format="enum: lamellar | bragg_kogelnik | dammann | table",
        validator={"kind": "enum",
                   "values": ("lamellar", "bragg_kogelnik", "dammann",
                              "table")}),
    "lines_per_mm": ColumnInfo(
        "Groove/fringe frequency.",
        units="lines/mm", format="float > 0",
        validator={"kind": "float", "gt": 0.0}),
    "params": ColumnInfo(
        "Model-specific parameters. bragg_kogelnik REQUIRES "
        "thickness_um and dn (slant_deg optional, default 0). dammann "
        "REQUIRES transitions=x1,x2,... (strictly increasing, each in "
        "(0,1)). Ignored for model=table/lamellar.",
        format="';'-separated 'key=value' pairs, value is a float or a "
               "comma-list of floats, e.g. "
               "'thickness_um=3000;dn=0.0005;slant_deg=0' or "
               "'transitions=0.03863,0.39084'"),
    "table_csv": ColumnInfo(
        "Filename of a per-order efficiency/amplitude table in "
        "grating/tables/. REQUIRED when model=table; ignored otherwise. "
        "Two on-disk schemas exist (distinguished by the table file's own "
        "first line): legacy v1 (wavelength_nm,order,eta_s,eta_p, real "
        "efficiencies) and v2 RCWA (marked '# mietab grating v2 "
        "[side=...]', complex amp_s/amp_p vs wavelength/theta/phi).",
        format="filename -- required iff model=table"),
    "reference": _REFERENCE,
}

# ---------------------------------------------------------------------------
# birefringence/uniaxial.miebrf
# ---------------------------------------------------------------------------
_UNIAXIAL = {
    "name": _name("uniaxial crystal"),
    "n_o_material": ColumnInfo(
        "materials.miemat row supplying the ordinary-ray index n_o.",
        format="materials.miemat row name (must exist)"),
    "n_e_material": ColumnInfo(
        "materials.miemat row supplying the extraordinary-ray index n_e.",
        format="materials.miemat row name (must exist)"),
    "reference": _REFERENCE,
    "notes": _NOTES,
    "gyration_deg_per_mm": ColumnInfo(
        "Optional natural optical activity (rotatory power) at "
        "gyration_ref_nm, feeding the Berreman gyrotropic term. Absent = "
        "non-gyrotropic. If set, gyration_ref_nm AND gyration_reference "
        "must also be set.",
        units="deg/mm", format="float, optional",
        validator={"kind": "float"}),
    "gyration_ref_nm": ColumnInfo(
        "Reference wavelength the gyration_deg_per_mm measurement was "
        "taken at. Required iff gyration_deg_per_mm is set.",
        units="nm", format="float > 0, required iff gyration_deg_per_mm set",
        validator={"kind": "float", "gt": 0.0}),
    "gyration_reference": ColumnInfo(
        "Citation for the gyration measurement specifically (separate "
        "from the row's main `reference`). Required iff "
        "gyration_deg_per_mm is set.",
        format="citation string, required iff gyration_deg_per_mm set"),
}

# ---------------------------------------------------------------------------
# diffuser/diffusers.miedif
# ---------------------------------------------------------------------------
_DIFFUSERS = {
    "name": _name("diffuser"),
    "grit": ColumnInfo(
        "Catalog ground-glass grit number, mapped to an RMS microfacet "
        "slope via roughness.slope_for_grit (log-log interpolated DG-"
        "series calibration: 120,220,600,1500). At least one of "
        "`grit`/`slope_rms` is required; if both are given they must "
        "agree with the mapping within 20%.",
        format="int > 0, e.g. 120 | 220 | 600 | 1500",
        validator={"kind": "int", "gt": 0}),
    "slope_rms": ColumnInfo(
        "RMS microfacet slope directly (dimensionless small-angle "
        "Beckmann slope). At least one of `grit`/`slope_rms` is required.",
        format="float in (0, 1), optional if `grit` given",
        validator={"kind": "float", "range": (0.0, 1.0)}),
    "reference": _REFERENCE,
}

# ---------------------------------------------------------------------------
# detector/detectors.miedet
# ---------------------------------------------------------------------------
_DETECTORS = {
    "name": _name("detector QE curve"),
    "table_csv": _TABLE_CSV("detector/tables"),
    "reference": _REFERENCE,
    "notes": _NOTES,
}

# ---------------------------------------------------------------------------
# emission/emitters.miesrc
# ---------------------------------------------------------------------------
_EMISSION = {
    "name": _name("emission spectrum"),
    "kind": ColumnInfo(
        "Spectrum shape family. Only 'continuous' (piecewise-linear PDF "
        "table) has engine support today -- 'blackbody'/'line' rows are "
        "staged but rejected at load with a needs-engine-support error.",
        format="enum: continuous (blackbody/line staged, not yet loadable)",
        validator={"kind": "enum", "values": ("continuous",)}),
    "table_csv": _TABLE_CSV("emission/tables"),
    "reference": _REFERENCE,
    "notes": _NOTES,
}

# ---------------------------------------------------------------------------
# category -> {column: ColumnInfo}
# ---------------------------------------------------------------------------
COLUMN_SCHEMA = {
    "materials": _MATERIALS,
    "coatings": _COATINGS,
    "polarizers": _POLARIZERS,
    "filters": _FILTERS,
    "gratings": _GRATINGS,
    "uniaxial": _UNIAXIAL,
    "diffusers": _DIFFUSERS,
    "detectors": _DETECTORS,
    "emission": _EMISSION,
}


# ---------------------------------------------------------------------------
# TABLE_COLUMN_SCHEMA: per-row spectral TABLE file columns (the file named
# by a registry row's table/table_csv/nk_file column), mirroring
# panes/prop_editor.py's TABLE_SCHEMA tuples. `uniaxial` has no table (its
# rows reference other materials.miemat rows, not a file) and is omitted.
# ---------------------------------------------------------------------------
_WAVELENGTH_NM = ColumnInfo(
    "Sample wavelength. Every table is required to be strictly "
    "increasing in this column; interpolation never extrapolates outside "
    "the tabulated range (a hard error, not a silent clamp).",
    units="nm", format="float > 0, strictly increasing down the table",
    validator={"kind": "float", "range": (100.0, 20000.0)})

TABLE_COLUMN_SCHEMA = {
    "materials": {
        "wavelength_nm": _WAVELENGTH_NM,
        "n": ColumnInfo("Real refractive index at this wavelength.",
                        format="float", validator={"kind": "float"}),
        "k": ColumnInfo("Extinction coefficient (imaginary index) at this "
                        "wavelength; interpolated log-linearly when the "
                        "whole column is positive, else linearly.",
                        format="float >= 0", validator={"kind": "float", "ge": 0.0}),
    },
    "coatings": {
        "wavelength_nm": _WAVELENGTH_NM,
        "Rs": ColumnInfo("s-polarized power reflectance.", units="fraction",
                         format="float in [0,1]",
                         validator={"kind": "float", "range": (0.0, 1.0)}),
        "Rp": ColumnInfo("p-polarized power reflectance.", units="fraction",
                         format="float in [0,1]",
                         validator={"kind": "float", "range": (0.0, 1.0)}),
        "Ts": ColumnInfo("s-polarized power transmittance. Rs+Ts<=1 per "
                         "row (remainder is absorption).",
                         units="fraction", format="float in [0,1]",
                         validator={"kind": "float", "range": (0.0, 1.0)}),
        "Tp": ColumnInfo("p-polarized power transmittance. Rp+Tp<=1 per "
                         "row.", units="fraction", format="float in [0,1]",
                         validator={"kind": "float", "range": (0.0, 1.0)}),
    },
    "polarizers": {
        "wavelength_nm": _WAVELENGTH_NM,
        "T_parallel": ColumnInfo(
            "Power transmission fraction for light polarized PARALLEL to "
            "the transmission axis. Must exceed T_perpendicular at every "
            "wavelength.", units="fraction", format="float in (0,1]",
            validator={"kind": "float", "range": (0.0, 1.0)}),
        "T_perpendicular": ColumnInfo(
            "Power transmission fraction for light polarized "
            "PERPENDICULAR to the transmission axis (extinction ratio = "
            "T_parallel/T_perpendicular).", units="fraction",
            format="float in (0,1], must be < T_parallel",
            validator={"kind": "float", "range": (0.0, 1.0)}),
    },
    "filters": {
        "wavelength_nm": _WAVELENGTH_NM,
        "transmittance_internal": ColumnInfo(
            "Internal (bulk, surface-reflection-excluded) transmittance "
            "at ref_thickness_mm. Must be > 0 -- a stopband uses a small "
            "floor like 1e-6, never an exact 0.",
            units="fraction", format="float in (0,1]",
            validator={"kind": "float", "range": (0.0, 1.0)}),
    },
    "gratings": {
        "wavelength_nm": _WAVELENGTH_NM,
        "order": ColumnInfo(
            "Diffraction order this row's efficiency applies to.",
            format="int", validator={"kind": "int", "gt": -1000}),
        "eta_s": ColumnInfo(
            "s-polarized diffraction efficiency for this order/"
            "wavelength. Per-wavelength/polarization order efficiencies "
            "must sum to <= 1.", units="fraction", format="float in [0,1]",
            validator={"kind": "float", "range": (0.0, 1.0)}),
        "eta_p": ColumnInfo(
            "p-polarized diffraction efficiency for this order/"
            "wavelength.", units="fraction", format="float in [0,1]",
            validator={"kind": "float", "range": (0.0, 1.0)}),
    },
    "detectors": {
        "wavelength_nm": _WAVELENGTH_NM,
        "qe": ColumnInfo(
            "Fractional quantum efficiency at this wavelength. Zero-"
            "filled (not extrapolated) outside the table's own range.",
            units="fraction", format="float in (0,1]",
            validator={"kind": "float", "range": (0.0, 1.0)}),
    },
    "emission": {
        "wavelength_nm": _WAVELENGTH_NM,
        "relative_power": ColumnInfo(
            "Relative spectral power density (arbitrary units -- only "
            "the SHAPE matters; normalized to a PDF internally). Must be "
            ">= 0 everywhere and integrate to > 0.",
            units="arbitrary", format="float >= 0",
            validator={"kind": "float", "ge": 0.0}),
    },
}


# ---------------------------------------------------------------------------
# lookup helpers
# ---------------------------------------------------------------------------
def lookup(category, column):
    """ColumnInfo for `column` in the registry `category`, or None if
    either is unknown / undocumented (an unrecognized column degrades
    gracefully -- registries intentionally drift ahead of this module
    sometimes; see the module docstring)."""
    return COLUMN_SCHEMA.get(category, {}).get(column)


def lookup_table(category, column):
    """ColumnInfo for `column` in the per-row spectral TABLE file of
    registry `category` (see TABLE_COLUMN_SCHEMA), or None."""
    return TABLE_COLUMN_SCHEMA.get(category, {}).get(column)


def status_text(category, column):
    """One-line 'name -- description [units] (format)' string for a
    status bar / tooltip, or '' if there's no schema entry for this
    (category, column)."""
    info = lookup(category, column)
    if info is None:
        return ""
    parts = [column, "--", info.description]
    if info.units:
        parts.append("[%s]" % info.units)
    if info.format:
        parts.append("(%s)" % info.format)
    return " ".join(parts)


def tooltip_text(category, column):
    """Multi-line rich-text tooltip body for a column header, or a plain
    'no schema entry' fallback string for an undocumented column (dynamic
    registry columns must degrade gracefully, never crash the pane)."""
    info = lookup(category, column)
    if info is None:
        return "%s -- no schema entry (undocumented / registry-added " \
               "column)" % column
    lines = [info.description]
    if info.units:
        lines.append("Units: %s" % info.units)
    if info.format:
        lines.append("Format: %s" % info.format)
    return "\n".join(lines)


def validate_cell(category, column, text):
    """Advisory validation for one cell's raw text against its column's
    validator spec. -> (ok, message). Always ok=True (no message) for a
    blank cell, an undocumented column, or a column with no validator --
    this function NEVER blocks a save, it only informs the GUI whether to
    flag a cell for a human to double check."""
    info = lookup(category, column)
    if info is None or info.validator is None:
        return True, ""
    text = (text or "").strip()
    if not text:
        return True, ""
    spec = info.validator
    kind = spec.get("kind")
    if kind in ("float", "int"):
        try:
            value = float(text) if kind == "float" else int(text)
        except (TypeError, ValueError):
            return False, "expected %s%s" % (
                kind, (" (%s)" % info.format) if info.format else "")
        gt = spec.get("gt")
        if gt is not None and not value > gt:
            return False, "must be > %g (got %g)" % (gt, value)
        ge = spec.get("ge")
        if ge is not None and not value >= ge:
            return False, "must be >= %g (got %g)" % (ge, value)
        rng = spec.get("range")
        if rng is not None and not (rng[0] <= value <= rng[1]):
            return False, "implausible value %g (expected roughly [%g, %g]%s)" % (
                value, rng[0], rng[1],
                " %s" % info.units if info.units else "")
        return True, ""
    if kind == "enum":
        values = spec.get("values", ())
        if text not in values:
            return False, "expected one of: %s" % ", ".join(values)
        return True, ""
    return True, ""
