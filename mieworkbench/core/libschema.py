"""libschema.py -- per-column documentation + advisory validation for the
optical property library registries edited by panes/prop_editor.py.

This is a PURE-DATA module (no Qt, no I/O): COLUMN_SCHEMA describes every
column of every registry PropLibrary/core.proplib.CATEGORY_INFO exposes
(materials, coatings, polarizers, filters, gratings, uniaxial, diffusers,
detectors, emission, biaxial, figures, nonlinear, scatter, instruments);
TABLE_COLUMN_SCHEMA does the same for the per-row spectral TABLE files
those registries reference (the tuples in panes/prop_editor.py's
TABLE_SCHEMA/CATEGORY_TABS).

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

Two of these registries have a shape the others don't:
  - nonlinear/nonlinear.mienlo is a WIDE, KIND-DISCRIMINATED table: one
    'kind' column selects which of the remaining columns apply (a
    chi2_tensor row's d_il_pm_V is meaningless on a saturable row, and
    vice versa) -- each column's ColumnInfo says which kind(s) it belongs
    to. The registry csv also allows full-line '#' comments (documenting
    the d_il_pm_V/r_coeffs_pm_V packing grammar in the file header);
    core.proplib.CATEGORY_INFO["nonlinear"]["comment_prefix"] = "#" is
    what tells PropLibrary.registry_rows()/registry_fieldnames() to strip
    those lines before csv.DictReader, and prop_editor's
    _atomic_write_registry re-prepends them on save.
  - instrument/instruments.mieinst is WIDE and CLASS-discriminated the
    same way ('class' picks camera/powermeter/spectrometer/polarimeter/
    wavefront_sensor/autocorrelator); three PLACEHOLDER classes
    (polarimeter/wavefront_sensor/autocorrelator) have hard-validated
    column schemas in optprops.py but no shipped rows yet -- documented
    here anyway since load_instruments will enforce them the moment a row
    of that class is authored.

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
        "Spectrum shape family (samples-instruments round). 'continuous' "
        "= piecewise-linear PDF table (table_csv); 'blackbody' = analytic "
        "Planck synthesized to a dense table at load (params column: "
        "temp_k:3000;lam_lo_nm:350;lam_hi_nm:2500 — no table_csv); "
        "'lines' = discrete emission lines (lines column "
        "'nm:intensity;...', optional params linewidth_nm; no table_csv).",
        format="enum: continuous | blackbody | lines",
        validator={"kind": "enum",
                   "values": ("continuous", "blackbody", "lines")}),
    "table_csv": _TABLE_CSV("emission/tables"),
    "params": ColumnInfo(
        "':'-keyed ';'-separated per-kind parameters: blackbody temp_k/"
        "lam_lo_nm/lam_hi_nm (required); lines linewidth_nm (optional, "
        "floors at 1e-3 nm). Blank for continuous rows.",
        format="key:value;key:value (blank ok)"),
    "lines": ColumnInfo(
        "kind=lines only: the discrete line list, "
        "'wavelength_nm:relative_intensity;...' (e.g. Hg pen-lamp "
        "'253.65:1500;435.83:400;546.07:1000').",
        format="nm:intensity;... (blank unless kind=lines)"),
    "reference": _REFERENCE,
    "notes": _NOTES,
}

# ---------------------------------------------------------------------------
# birefringence/biaxial.mibiax
# ---------------------------------------------------------------------------
_BIAXIAL = {
    "name": _name("biaxial crystal"),
    "n_x_material": ColumnInfo(
        "materials.miemat row supplying the principal-axis index n_x (by "
        "convention the smallest of the three principal indices for a "
        "positive biaxial crystal -- not enforced by the loader).",
        format="materials.miemat row name (must exist)"),
    "n_y_material": ColumnInfo(
        "materials.miemat row supplying the principal-axis index n_y.",
        format="materials.miemat row name (must exist)"),
    "n_z_material": ColumnInfo(
        "materials.miemat row supplying the principal-axis index n_z.",
        format="materials.miemat row name (must exist)"),
    "reference": _REFERENCE,
    "notes": _NOTES,
}

# ---------------------------------------------------------------------------
# figure/figures.miefig
# ---------------------------------------------------------------------------
_FIGURES = {
    "name": _name("surface figure-error set"),
    "coeffs": ColumnInfo(
        "Zernike (Noll indexing) surface-figure-error coefficient set, "
        "applied at scene build as a raytracer.surfaces.PerturbedSurface "
        "sag perturbation over the transverse pupil. Each term's rms_nm is "
        "the SURFACE sag RMS at that Noll index (a mirror's WAVEFRONT "
        "error is 2x this and falls out of the tracer's OPL naturally). "
        "j=1 (piston) is rejected as a meaningless constant offset; "
        "duplicate j in one cell is rejected.",
        units="nm (per term)",
        format="';'-separated 'j:rms_nm' terms, e.g. '5:100;6:100' -- j is "
               "a Noll index int >= 2, rms_nm a float"),
    "r_norm_mm": ColumnInfo(
        "Pupil (clear-aperture) radius the coeffs are normalized to -- "
        "the Zernike terms are evaluated over rho = r/r_norm_mm in [0,1].",
        units="mm", format="float > 0",
        validator={"kind": "float", "gt": 0.0}),
    "reference": _REFERENCE,
    "notes": ColumnInfo(
        "Free-text annotation -- here often used to name which Zernike "
        "term(s) the set represents (e.g. 'Z4 only', 'Z9+Z10 trefoil'). "
        "Never validated by the loader.",
        format="free text (optional)"),
}

# ---------------------------------------------------------------------------
# nonlinear/nonlinear.mienlo -- chi(2)/chi(3)/EO/saturable-absorber registry
# WIDE, kind-discriminated: 'kind' selects which of the remaining columns
# are meaningful on a given row (the others stay blank). Full-line '#'
# comments are allowed in the csv itself (see module docstring / the file's
# own header block for the d_il_pm_V / r_coeffs_pm_V packing grammar this
# mirrors).
# ---------------------------------------------------------------------------
_NONLINEAR = {
    "kind": ColumnInfo(
        "Row-type discriminator -- selects which of this row's other "
        "columns are read (the rest stay blank).",
        format="enum: chi2_tensor | chi2_process | pockels | n2 | saturable",
        validator={"kind": "enum",
                   "values": ("chi2_tensor", "chi2_process", "pockels",
                              "n2", "saturable")}),
    "name": _name("nonlinear-optical row"),
    "crystal": ColumnInfo(
        "Host crystal for kind=chi2_tensor/chi2_process/pockels rows. "
        "Must name a birefringence/uniaxial.miebrf or "
        "birefringence/biaxial.mibiax row (cross-checked when the loader "
        "is given both registry handles, i.e. via load_optical_properties; "
        "skipped when nonlinear.mienlo is loaded standalone).",
        format="uniaxial.miebrf / biaxial.mibiax row name -- required iff "
               "kind in {chi2_tensor, chi2_process, pockels}"),
    "point_group": ColumnInfo(
        "Crystal point group symbol (e.g. '3m', 'mm2', '-42m'), documents "
        "which entries of the 3x6 d-matrix the point-group symmetry "
        "forces to zero/equal -- informational, not cross-validated "
        "against a lookup table.",
        format="point-group string -- required iff kind=chi2_tensor"),
    "d_il_pm_V": ColumnInfo(
        "Full 3x6 contracted (Voigt) second-order nonlinear-susceptibility "
        "d-matrix in the crystal principal frame, row-major.",
        units="pm/V",
        format="three '|'-separated rows i=1..3 (polarization component), "
               "each exactly six ';'-separated floats l=1..6 (Voigt "
               "11->1,22->2,33->3,23/32->4,13/31->5,12/21->6): "
               "'d11;d12;d13;d14;d15;d16|d21;...;d26|d31;...;d36' -- "
               "required iff kind=chi2_tensor"),
    "kleinman": ColumnInfo(
        "Whether Kleinman symmetry (d_il fully permutation-symmetric, "
        "valid far from resonance) was assumed when reducing the "
        "independent tensor components -- documents the d_il_pm_V "
        "provenance, not re-derived by the loader.",
        format="literal 'true' or 'false' -- required iff kind=chi2_tensor",
        validator={"kind": "enum", "values": ("true", "false")}),
    "lam_ref_nm": ColumnInfo(
        "Reference wavelength the row's coefficient (d_il_pm_V for "
        "chi2_tensor, n2_m2_W for n2) was measured/quoted at.",
        units="nm",
        format="float > 0 -- required iff kind in {chi2_tensor, n2}",
        validator={"kind": "float", "gt": 0.0}),
    "process": ColumnInfo(
        "Specific SHG phase-matching process this chi2_process row "
        "characterizes with a single effective coefficient (as opposed to "
        "the full tensor in a chi2_tensor row).",
        format="enum: shg_type1 | shg_type2 -- required iff kind=chi2_process",
        validator={"kind": "enum", "values": ("shg_type1", "shg_type2")}),
    "lam_pump_nm": ColumnInfo(
        "Fundamental (pump) wavelength this SHG process row is "
        "phase-matched at.",
        units="nm",
        format="float > 0 -- required iff kind=chi2_process",
        validator={"kind": "float", "gt": 0.0}),
    "theta_deg": ColumnInfo(
        "Phase-matching polar angle theta (crystal-cut angle) for this "
        "SHG process.",
        units="deg", format="float -- required iff kind=chi2_process",
        validator={"kind": "float"}),
    "phi_deg": ColumnInfo(
        "Phase-matching azimuthal angle phi for this SHG process.",
        units="deg", format="float -- required iff kind=chi2_process",
        validator={"kind": "float"}),
    "d_eff_pm_V": ColumnInfo(
        "Single effective nonlinear coefficient for this cut/process "
        "(projection of the full d-matrix onto the phase-matched "
        "geometry) -- must be non-zero.",
        units="pm/V",
        format="float, non-zero -- required iff kind=chi2_process",
        validator={"kind": "float"}),
    "r_coeffs_pm_V": ColumnInfo(
        "Named linear electro-optic (Pockels) coefficients used by the "
        "row's `geometry`.",
        units="pm/V",
        format="';'-separated 'rNN=value' pairs, e.g. 'r33=30.8;r13=8.6' "
               "-- required iff kind=pockels"),
    "geometry": ColumnInfo(
        "Pockels-cell electrode/field geometry the r_coeffs_pm_V values "
        "apply to (selects which raytracer.nlo shifted-index formula "
        "consumes them).",
        format="enum: longitudinal | transverse -- required iff kind=pockels",
        validator={"kind": "enum", "values": ("longitudinal", "transverse")}),
    "material": ColumnInfo(
        "materials.miemat row this Kerr (n2) coefficient applies to. "
        "Resolved LAZILY by the Kerr consumer at use time (not checked "
        "against materials.miemat by this loader), so a staged n2 row may "
        "precede its materials.miemat index row.",
        format="materials.miemat row name -- required iff kind=n2"),
    "n2_m2_W": ColumnInfo(
        "Nonlinear (Kerr) refractive index n2, must be non-zero.",
        units="m^2/W",
        format="float, non-zero -- required iff kind=n2",
        validator={"kind": "float"}),
    "I_sat_W_cm2": ColumnInfo(
        "Saturation intensity of a saturable absorber (SESAM etc.) -- the "
        "intensity at which the bulk/device absorption drops by half in "
        "the alpha(I)=alpha0/(1+I/I_sat) model.",
        units="W/cm^2",
        format="float > 0 -- required iff kind=saturable",
        validator={"kind": "float", "gt": 0.0}),
    "T0": ColumnInfo(
        "Unsaturated (low-intensity) transmission OR reflectance "
        "fraction, depending on whether the device is transmissive (bulk "
        "absorber) or a mirror (SESAM); alpha0_per_mm's blank/set state "
        "documents which. Must be in (0, 1].",
        units="fraction",
        format="float in (0,1] -- required iff kind=saturable",
        validator={"kind": "float", "range": (0.0, 1.0)}),
    "tau_recovery_s": ColumnInfo(
        "Absorber recovery (relaxation) time constant.",
        units="s",
        format="float >= 0 -- required iff kind=saturable",
        validator={"kind": "float", "ge": 0.0}),
    "alpha0_per_mm": ColumnInfo(
        "OPTIONAL bulk unsaturated absorption coefficient per millimetre, "
        "consumed by raytracer.nlo.saturable_alpha0_per_m for the bulk "
        "alpha(I) = alpha0/(1+I/I_sat) hook. Leave blank for a MIRROR/"
        "reflectance device row (T0 = 1-A, not a bulk transmission) -- "
        "blank falls back to reading T0 itself as a per-mm transmission "
        "(alpha0 = -ln(T0)/mm), which would misrepresent a device spec as "
        "a bulk coefficient if filled in for a mirror row.",
        units="1/mm",
        format="float > 0, optional (saturable rows only)",
        validator={"kind": "float", "gt": 0.0}),
    "reference": _REFERENCE,
    "notes": _NOTES,
}

# ---------------------------------------------------------------------------
# scatter/bsdf.miebsdf -- ABg measured-scatter surfaces
# ---------------------------------------------------------------------------
_SCATTER = {
    "name": _name("BSDF scatter surface"),
    "model": ColumnInfo(
        "Scatter model family. Only 'abg' has engine support today.",
        format="enum: abg", validator={"kind": "enum", "values": ("abg",)}),
    "A": ColumnInfo(
        "ABg model reflected-side amplitude parameter: "
        "BSDF(u) = A/(B + u^g), u = |beta - beta0| the direction-cosine "
        "offset from specular.",
        format="float > 0", validator={"kind": "float", "gt": 0.0}),
    "B": ColumnInfo(
        "ABg model reflected-side B parameter (denominator offset -- "
        "controls the shoulder/knee of the scatter lobe).",
        format="float > 0", validator={"kind": "float", "gt": 0.0}),
    "g": ColumnInfo(
        "ABg model reflected-side rolloff exponent. g=2 gives a "
        "closed-form radial CDF (the common case).",
        format="float > 0", validator={"kind": "float", "gt": 0.0}),
    "tis_cap": ColumnInfo(
        "Optional ceiling on the reflected-side total integrated scatter "
        "(TIS) used by the tracer's specular/scattered power split -- pins "
        "an ABg fit that would otherwise over-integrate to a plausible "
        "measured total-scatter fraction.",
        units="fraction", format="float in (0,1], optional",
        validator={"kind": "float", "range": (0.0, 1.0)}),
    "btdf": ColumnInfo(
        "Enables the OPTIONAL transmitted-side (BTDF) scatter split about "
        "the refracted direction, using the btdf_* ABg triple below. "
        "Falsey/blank behaves exactly as a reflected-side-only row.",
        format="boolean: 1|true|yes|on (enabled) or blank|0|false|no|off "
               "(disabled)",
        validator={"kind": "enum",
                   "values": ("1", "true", "yes", "on", "0", "false", "no",
                              "off", "")}),
    "btdf_A": ColumnInfo(
        "BTDF-side A parameter (see `A`); defaults to the reflected-side "
        "A when blank. Only read when `btdf` is truthy.",
        format="float > 0, optional (defaults to `A`)",
        validator={"kind": "float", "gt": 0.0}),
    "btdf_B": ColumnInfo(
        "BTDF-side B parameter (see `B`); defaults to the reflected-side "
        "B when blank. Only read when `btdf` is truthy.",
        format="float > 0, optional (defaults to `B`)",
        validator={"kind": "float", "gt": 0.0}),
    "btdf_g": ColumnInfo(
        "BTDF-side rolloff exponent (see `g`); defaults to the "
        "reflected-side g when blank. Only read when `btdf` is truthy.",
        format="float > 0, optional (defaults to `g`)",
        validator={"kind": "float", "gt": 0.0}),
    "btdf_tis_cap": ColumnInfo(
        "Optional TIS ceiling for the transmitted-side split (see "
        "`tis_cap`). Only read when `btdf` is truthy.",
        units="fraction", format="float in (0,1], optional",
        validator={"kind": "float", "range": (0.0, 1.0)}),
    "reference": _REFERENCE,
    "notes": _NOTES,
}

# ---------------------------------------------------------------------------
# instrument/instruments.mieinst -- virtual instrument layer (post-process
# response model over an ideal detector plane). WIDE, class-discriminated:
# 'class' selects which of the remaining columns are meaningful on a given
# row. polarimeter/wavefront_sensor/autocorrelator are PLACEHOLDER classes
# (schema defined + hard-validated, no shipped rows yet -- engine3.md Sec.9).
# ---------------------------------------------------------------------------
_INSTRUMENTS = {
    "name": _name("instrument profile"),
    "class": ColumnInfo(
        "Instrument-model discriminator -- selects which of this row's "
        "other columns are read (the rest stay blank) and which "
        "post_process.render_instrument dispatcher handles it.",
        format="enum: camera | powermeter | spectrometer | polarimeter | "
               "wavefront_sensor | autocorrelator (the last three are "
               "PLACEHOLDER classes with a validated schema but no shipped "
               "rows yet)",
        validator={"kind": "enum",
                   "values": ("camera", "powermeter", "spectrometer",
                              "polarimeter", "wavefront_sensor",
                              "autocorrelator")}),
    "pixel_pitch_um": ColumnInfo(
        "Camera pixel pitch (assumed square).",
        units="um", format="float > 0 -- required iff class=camera",
        validator={"kind": "float", "gt": 0.0}),
    "width_px": ColumnInfo(
        "Camera sensor width.",
        units="px", format="int > 0 -- required iff class=camera",
        validator={"kind": "int", "gt": 0}),
    "height_px": ColumnInfo(
        "Camera sensor height.",
        units="px", format="int > 0 -- required iff class=camera",
        validator={"kind": "int", "gt": 0}),
    "fill_factor": ColumnInfo(
        "Camera pixel fill factor (active photosensitive area fraction).",
        units="fraction",
        format="float in (0,1] -- required iff class=camera",
        validator={"kind": "float", "range": (0.0, 1.0)}),
    "qe_table": ColumnInfo(
        "Filename of the camera's per-wavelength quantum-efficiency table "
        "(column 'qe', fractional, in instrument/tables/).",
        format="filename (e.g. 'camera_generic_qe.mietab') -- required "
               "iff class=camera"),
    "full_well_e": ColumnInfo(
        "Camera pixel full-well capacity (saturation clipping level).",
        units="electrons",
        format="float > 0 -- required iff class=camera",
        validator={"kind": "float", "gt": 0.0}),
    "read_noise_e": ColumnInfo(
        "Camera read noise (RMS electrons added per readout, 'full' mode "
        "only).",
        units="electrons rms",
        format="float >= 0 -- required iff class=camera",
        validator={"kind": "float", "ge": 0.0}),
    "dark_current_e_per_s": ColumnInfo(
        "Camera dark current.",
        units="electrons/s",
        format="float >= 0 -- required iff class=camera",
        validator={"kind": "float", "ge": 0.0}),
    "bit_depth": ColumnInfo(
        "Camera ADC bit depth (quantization applied to the counts image).",
        units="bits", format="int > 0 -- required iff class=camera",
        validator={"kind": "int", "gt": 0}),
    "adc_gain_e_per_dn": ColumnInfo(
        "Camera ADC gain (electrons per digital number/count).",
        units="electrons/DN",
        format="float > 0 -- required iff class=camera",
        validator={"kind": "float", "gt": 0.0}),
    "integration_time_s_default": ColumnInfo(
        "Camera default exposure/integration time used when a run doesn't "
        "override it.",
        units="s", format="float > 0 -- required iff class=camera",
        validator={"kind": "float", "gt": 0.0}),
    "responsivity_table": ColumnInfo(
        "Filename of the powermeter's per-wavelength responsivity table "
        "(column 'responsivity_a_w', in instrument/tables/). EXACTLY ONE "
        "of responsivity_table / flat_responsivity_a_w must be set on a "
        "powermeter row (the other left blank).",
        format="filename, optional (mutually exclusive with "
               "flat_responsivity_a_w) -- class=powermeter only"),
    "flat_responsivity_a_w": ColumnInfo(
        "Flat (wavelength-independent) responsivity, an alternative to a "
        "full responsivity_table. EXACTLY ONE of the two must be set on a "
        "powermeter row.",
        units="A/W",
        format="float > 0, optional (mutually exclusive with "
               "responsivity_table) -- class=powermeter only",
        validator={"kind": "float", "gt": 0.0}),
    "aperture_mm": ColumnInfo(
        "Powermeter sensor active aperture diameter.",
        units="mm", format="float > 0 -- required iff class=powermeter",
        validator={"kind": "float", "gt": 0.0}),
    "nep_w_per_sqrthz": ColumnInfo(
        "Powermeter noise-equivalent power spectral density.",
        units="W/sqrt(Hz)",
        format="float > 0 -- required iff class=powermeter",
        validator={"kind": "float", "gt": 0.0}),
    "bandwidth_hz": ColumnInfo(
        "Powermeter detection bandwidth (with nep_w_per_sqrthz, sets the "
        "noise floor in 'full' mode).",
        units="Hz", format="float > 0 -- required iff class=powermeter",
        validator={"kind": "float", "gt": 0.0}),
    "display_digits": ColumnInfo(
        "Powermeter display resolution -- the reported power is "
        "significant-figure rounded to this many digits.",
        format="int >= 1 -- required iff class=powermeter",
        validator={"kind": "int", "gt": 0}),
    "lam_lo_nm": ColumnInfo(
        "Spectrometer lower wavelength bound of the reported range; must "
        "be < lam_hi_nm.",
        units="nm",
        format="float > 0 -- required iff class=spectrometer",
        validator={"kind": "float", "gt": 0.0}),
    "lam_hi_nm": ColumnInfo(
        "Spectrometer upper wavelength bound; must be > lam_lo_nm.",
        units="nm",
        format="float > 0 -- required iff class=spectrometer",
        validator={"kind": "float", "gt": 0.0}),
    "resolution_fwhm_nm": ColumnInfo(
        "Spectrometer spectral resolution (Gaussian convolution FWHM "
        "applied to the reported spectrum).",
        units="nm",
        format="float > 0 -- required iff class=spectrometer",
        validator={"kind": "float", "gt": 0.0}),
    "slit_um": ColumnInfo(
        "Spectrometer entrance slit width.",
        units="um", format="float > 0 -- required iff class=spectrometer",
        validator={"kind": "float", "gt": 0.0}),
    "stray_light_floor": ColumnInfo(
        "Spectrometer stray-light floor added to the reported spectrum.",
        units="fraction",
        format="float in [0,1) -- required iff class=spectrometer",
        validator={"kind": "float", "range": (0.0, 1.0)}),
    "detector_qe_table": ColumnInfo(
        "Filename of the spectrometer's internal detector "
        "quantum-efficiency table (column 'qe', in instrument/tables/).",
        format="filename (e.g. 'spectrometer_generic_qe.mietab') -- "
               "required iff class=spectrometer"),
    "analyzer_states": ColumnInfo(
        "Polarimeter number of distinct analyzer states sampled per "
        "measurement (PLACEHOLDER class -- see module note).",
        format="int >= 2 -- required iff class=polarimeter",
        validator={"kind": "int", "gt": 1}),
    "extinction_ratio": ColumnInfo(
        "Polarimeter analyzer extinction ratio (PLACEHOLDER class).",
        format="float > 0 -- required iff class=polarimeter",
        validator={"kind": "float", "gt": 0.0}),
    "retarder_error_deg": ColumnInfo(
        "Polarimeter retarder retardance error (PLACEHOLDER class).",
        units="deg",
        format="float >= 0 -- required iff class=polarimeter",
        validator={"kind": "float", "ge": 0.0}),
    "opd_sampling_um": ColumnInfo(
        "Wavefront sensor optical-path-difference sampling step "
        "(PLACEHOLDER class).",
        units="um",
        format="float > 0 -- required iff class=wavefront_sensor",
        validator={"kind": "float", "gt": 0.0}),
    "reference_arm_model": ColumnInfo(
        "Wavefront sensor reference-arm model identifier (PLACEHOLDER "
        "class) -- a non-empty descriptive string, not cross-validated "
        "against a lookup table.",
        format="non-empty string -- required iff class=wavefront_sensor"),
    "shg_crystal": ColumnInfo(
        "Autocorrelator SHG crystal name (PLACEHOLDER class) -- "
        "informational only; NOT cross-checked against birefringence/"
        "nonlinear registries (unlike nonlinear.mienlo's own `crystal` "
        "column).",
        format="non-empty string -- required iff class=autocorrelator"),
    "delay_range_fs": ColumnInfo(
        "Autocorrelator scan delay-line range (PLACEHOLDER class).",
        units="fs",
        format="float > 0 -- required iff class=autocorrelator",
        validator={"kind": "float", "gt": 0.0}),
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
    "biaxial": _BIAXIAL,
    "figures": _FIGURES,
    "nonlinear": _NONLINEAR,
    "scatter": _SCATTER,
    "instruments": _INSTRUMENTS,
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
    # instruments has NO CATEGORY_TABS import schema (its table shape
    # varies by row class -- see prop_editor.CATEGORY_TABS' comment), but
    # its two live per-row table shapes are documented here anyway since
    # the chart still reads/plots them directly by filename.
    "instruments": {
        "wavelength_nm": _WAVELENGTH_NM,
        "qe": ColumnInfo(
            "Fractional quantum efficiency at this wavelength (camera "
            "qe_table / spectrometer detector_qe_table rows).",
            units="fraction", format="float in (0,1]",
            validator={"kind": "float", "range": (0.0, 1.0)}),
        "responsivity_a_w": ColumnInfo(
            "Photodiode responsivity at this wavelength (powermeter "
            "responsivity_table rows).",
            units="A/W", format="float > 0",
            validator={"kind": "float", "gt": 0.0}),
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
