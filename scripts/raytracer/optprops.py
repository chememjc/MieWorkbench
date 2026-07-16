#!/usr/bin/env python3
# =============================================================================
# optprops.py -- loaders for the opticalproperties/ component library.
#
# Directory layout (README §7):
#   opticalproperties/
#     materials.csv                  MaterialDB (materials.py)
#     nk/                            tabulated n,k spectra (materials.py)
#     birefringence/uniaxial.csv     crystal -> (o-ray, e-ray) material rows
#     polarizer/polarizers.csv       registry -> tables/<name>.csv
#     filter/filters.csv             registry -> tables/<name>.csv
#     coating/coatings.csv           TMM stacks + measured tables (materials.py)
#     grating/gratings.csv           registry (kogelnik/dammann/table models)
#     nonlinear/nonlinear.mienlo     chi2 tensors/processes + Pockels + Kerr
#                                    n2 + saturable absorbers (nlo.py math)
#
# Every registry hard-validates its referenced table files at load time and
# requires a non-empty `reference` (citation) column — same policy as
# materials.csv. Table interpolation NEVER extrapolates: interp_hard()
# raises MaterialError outside the tabulated range, matching the nk-table
# convention in materials.py.
#
# load_optical_properties(root, db) is the one-call entry used by
# run_trace; the individual load_* functions are importable for tests.
# =============================================================================
import csv
from pathlib import Path

import numpy as np

from .materials import (MaterialDB, MaterialError, load_coatings,
                        DEFAULT_OPTPROPS_DIR, resolve_prop_path,
                        _swap_suffix)

DEFAULT_BIREFRINGENCE_CSV = DEFAULT_OPTPROPS_DIR / "birefringence" / "uniaxial.miebrf"
DEFAULT_BIAXIAL_CSV = DEFAULT_OPTPROPS_DIR / "birefringence" / "biaxial.mibiax"
DEFAULT_POLARIZERS_CSV = DEFAULT_OPTPROPS_DIR / "polarizer" / "polarizers.miepol"
DEFAULT_FILTERS_CSV = DEFAULT_OPTPROPS_DIR / "filter" / "filters.miefilt"
DEFAULT_GRATINGS_CSV = DEFAULT_OPTPROPS_DIR / "grating" / "gratings.miegrat"
DEFAULT_DETECTORS_CSV = DEFAULT_OPTPROPS_DIR / "detector" / "detectors.miedet"
DEFAULT_EMISSION_CSV = DEFAULT_OPTPROPS_DIR / "emission" / "emitters.miesrc"
DEFAULT_NONLINEAR_CSV = DEFAULT_OPTPROPS_DIR / "nonlinear" / "nonlinear.mienlo"

POLARIZER_TYPES = ("linear", "circular_left", "circular_right")
GRATING_MODELS = ("lamellar", "bragg_kogelnik", "dammann", "table")
# Emitter kinds with engine support THIS round. 'blackbody' (analytic Planck)
# and 'line' (discrete point-mass lines) are staged in library_data/ but need
# their own source models — rejected here rather than silently mis-sampled.
EMISSION_KINDS = ("continuous",)
# nonlinear/nonlinear.mienlo row kinds (pulsed-optics round; the math lives
# in raytracer/nlo.py, the tracer-side SHG event is a later phase).
NLO_KINDS = ("chi2_tensor", "chi2_process", "pockels", "n2", "saturable")
NLO_PROCESSES = ("shg_type1", "shg_type2")
NLO_GEOMETRIES = ("longitudinal", "transverse")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def interp_hard(lam_um, lam_tab, val_tab, ctx):
    """Linear interpolation with a hard error outside the tabulated range
    (no extrapolation — matches materials.py nk-table policy)."""
    lam_um = np.asarray(lam_um, dtype=np.float64)
    lo, hi = lam_tab[0], lam_tab[-1]
    bad = (lam_um < lo) | (lam_um > hi)
    if np.any(bad):
        first = float(np.atleast_1d(lam_um)[np.atleast_1d(bad)][0])
        raise MaterialError(
            "%s: wavelength %.6g um outside tabulated range [%.6g, %.6g] um "
            "-- no extrapolation" % (ctx, first, lo, hi))
    return np.interp(lam_um, lam_tab, val_tab)


def interp_phase_deg(lam_um, lam_tab, phase_deg_tab, ctx):
    """Interpolate a table of phase angles (degrees) at lam_um, returning
    radians (matches np.angle()'s convention, so callers can feed the
    result straight into np.exp(1j * ...)).

    Phase angles CANNOT be linearly interpolated as raw numbers across a
    +-180 deg branch cut (two adjacent rows of e.g. +178 deg and -178 deg
    are 4 deg apart physically but ~356 deg apart numerically). Instead
    this interpolates the complex UNIT vector (cos + i sin) of each row's
    angle component-wise via interp_hard, then takes the angle of the
    result -- exact for a smoothly-varying physical phase curve, branch
    cut or not (P2 coating-phase columns; materials.py's table-coating
    loader documents the matching all-or-none ars/arp/ats/atp_deg
    columns)."""
    rad = np.deg2rad(np.asarray(phase_deg_tab, dtype=np.float64))
    re = interp_hard(lam_um, lam_tab, np.cos(rad), ctx)
    im = interp_hard(lam_um, lam_tab, np.sin(rad), ctx)
    return np.arctan2(im, re)


def _read_registry(csv_path, required_cols, what):
    csv_path = resolve_prop_path(Path(csv_path))
    if not csv_path.exists():
        raise MaterialError("%s csv not found: %s" % (what, csv_path))
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise MaterialError("%s csv %s: missing required column(s) %s"
                                % (what, csv_path, sorted(missing)))
        rows = list(reader)
    out = []
    for i, row in enumerate(rows):
        lineno = i + 2
        name = (row.get("name") or "").strip()
        ctx = "%s line %d (%r)" % (csv_path.name, lineno,
                                   name or "<blank name>")
        if not name:
            raise MaterialError("%s: missing name" % ctx)
        if any(name.lower() == n.lower() for n, _, _ in out):
            raise MaterialError("%s: duplicate name" % ctx)
        reference = (row.get("reference") or "").strip()
        if not reference:
            raise MaterialError("%s: reference is required" % ctx)
        out.append((name, row, ctx))
    return out


def _read_table(path, columns, ctx, lam_col="wavelength_nm"):
    """Read a per-item table csv -> dict of float64 arrays, wavelength in um
    ('lam_um'), strictly increasing, >= 2 rows."""
    path = Path(path)
    resolved = resolve_prop_path(path, alt_ext=".mietab")
    if not resolved.exists():
        alt = _swap_suffix(path, ".mietab")
        raise MaterialError("%s: table not found: %s or %s"
                            % (ctx, path, alt))
    path = resolved
    data = {c: [] for c in (lam_col,) + tuple(columns)}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(data) - set(reader.fieldnames or [])
        if missing:
            raise MaterialError("%s: table %s missing column(s) %s"
                                % (ctx, path, sorted(missing)))
        for j, r in enumerate(reader):
            try:
                for c in data:
                    data[c].append(float(r[c]))
            except (TypeError, ValueError):
                raise MaterialError("%s: table %s row %d not numeric: %r"
                                    % (ctx, path, j + 2, r))
    lam_um = np.asarray(data.pop(lam_col), dtype=np.float64) * 1e-3
    if lam_um.size < 2:
        raise MaterialError("%s: table %s has fewer than 2 rows" % (ctx, path))
    if np.any(np.diff(lam_um) <= 0):
        raise MaterialError("%s: table %s %s not strictly increasing"
                            % (ctx, path, lam_col))
    out = {"lam_um": lam_um}
    for c in columns:
        out[c] = np.asarray(data[c], dtype=np.float64)
    return out


def _parse_params_field(raw, ctx):
    """';'-separated k=v pairs; v is a float or a comma-list of floats."""
    out = {}
    for kv in (raw or "").strip().split(";"):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise MaterialError("%s: bad params field %r" % (ctx, kv))
        k, v = kv.split("=", 1)
        k = k.strip()
        try:
            if "," in v:
                out[k] = [float(x) for x in v.split(",")]
            else:
                out[k] = float(v)
        except ValueError:
            raise MaterialError("%s: params %r=%r is not numeric" % (ctx, k, v))
    return out


# ---------------------------------------------------------------------------
# birefringence/uniaxial.csv
# ---------------------------------------------------------------------------
def load_uniaxial(csv_path=None, db=None):
    """-> {crystal_name: {"o": Material, "e": Material, "reference": str,
    "notes": str}}. Both material rows must exist in `db`. Call
    db.attach_uniaxial(result) to enable db.is_birefringent()."""
    csv_path = Path(csv_path) if csv_path is not None \
        else DEFAULT_BIREFRINGENCE_CSV
    if db is None:
        raise MaterialError("load_uniaxial requires a MaterialDB (db=...)")
    out = {}
    for name, row, ctx in _read_registry(
            csv_path, {"name", "n_o_material", "n_e_material", "reference"},
            "uniaxial birefringence"):
        o_name = (row.get("n_o_material") or "").strip()
        e_name = (row.get("n_e_material") or "").strip()
        for mn in (o_name, e_name):
            if mn not in db:
                raise MaterialError(
                    "%s: references unknown material %r (must be a "
                    "materials.csv row)" % (ctx, mn))
        if name in db and name.lower() not in (o_name.lower(),
                                               e_name.lower()):
            # a crystal name may shadow a scalar material row only if that
            # row IS one of its own o/e rows — otherwise lookups get
            # ambiguous (which index does body.material mean?)
            raise MaterialError(
                "%s: crystal name collides with unrelated materials.csv "
                "row %r" % (ctx, name))
        out[name] = {"o": db.get(o_name), "e": db.get(e_name),
                     "reference": (row.get("reference") or "").strip(),
                     "notes": (row.get("notes") or "").strip()}
    return out


def load_biaxial(csv_path=None, db=None):
    """-> {crystal_name: {"x","y","z": Material, "reference": str,
    "notes": str}}. All three principal-index material rows must exist in
    `db`. Call db.attach_biaxial(result) to enable db.is_biaxial()."""
    csv_path = Path(csv_path) if csv_path is not None \
        else DEFAULT_BIAXIAL_CSV
    if db is None:
        raise MaterialError("load_biaxial requires a MaterialDB (db=...)")
    out = {}
    for name, row, ctx in _read_registry(
            csv_path, {"name", "n_x_material", "n_y_material",
                       "n_z_material", "reference"},
            "biaxial birefringence"):
        axes = {ax: (row.get("n_%s_material" % ax) or "").strip()
                for ax in ("x", "y", "z")}
        for mn in axes.values():
            if mn not in db:
                raise MaterialError(
                    "%s: references unknown material %r (must be a "
                    "materials.csv row)" % (ctx, mn))
        if name in db and name.lower() not in (
                v.lower() for v in axes.values()):
            # same shadowing rule as uniaxial crystal names
            raise MaterialError(
                "%s: crystal name collides with unrelated materials.csv "
                "row %r" % (ctx, name))
        out[name] = {ax: db.get(mn) for ax, mn in axes.items()}
        out[name]["reference"] = (row.get("reference") or "").strip()
        out[name]["notes"] = (row.get("notes") or "").strip()
    return out


# ---------------------------------------------------------------------------
# polarizer/polarizers.csv + tables/
# ---------------------------------------------------------------------------
def load_polarizers(csv_path=None):
    """-> {name: {"type": linear|circular_left|circular_right,
    "lam_um": arr, "T_par": arr, "T_perp": arr,
    "retardance_waves": float (circular only, default 0.25),
    "reference": str}}.  T columns are power-transmission fractions of the
    polarizer film for light polarized parallel / perpendicular to the
    transmission axis; extinction ratio = T_par/T_perp."""
    csv_path = Path(csv_path) if csv_path is not None \
        else DEFAULT_POLARIZERS_CSV
    tables_dir = csv_path.parent / "tables"
    out = {}
    for name, row, ctx in _read_registry(
            csv_path, {"name", "type", "table_csv", "reference"},
            "polarizers"):
        ptype = (row.get("type") or "").strip()
        if ptype not in POLARIZER_TYPES:
            raise MaterialError("%s: type %r must be one of %s"
                                % (ctx, ptype, ", ".join(POLARIZER_TYPES)))
        table = _read_table(tables_dir / (row.get("table_csv") or "").strip(),
                            ("T_parallel", "T_perpendicular"), ctx)
        T_par, T_perp = table["T_parallel"], table["T_perpendicular"]
        if np.any((T_par <= 0) | (T_par > 1)) \
                or np.any((T_perp <= 0) | (T_perp > 1)):
            raise MaterialError("%s: transmissions must be in (0, 1]" % ctx)
        if np.any(T_perp >= T_par):
            raise MaterialError(
                "%s: T_perpendicular must be < T_parallel everywhere "
                "(otherwise the transmission axis is mislabeled)" % ctx)
        ret_raw = (row.get("retardance_waves") or "").strip()
        retardance = float(ret_raw) if ret_raw else 0.25
        out[name] = {"type": ptype, "lam_um": table["lam_um"],
                     "T_par": T_par, "T_perp": T_perp,
                     "retardance_waves": retardance,
                     "reference": (row.get("reference") or "").strip()}
    return out


# ---------------------------------------------------------------------------
# filter/filters.csv + tables/  (BULK spectral filters)
# ---------------------------------------------------------------------------
def load_filters(csv_path=None):
    """-> {name: {"lam_um": arr, "alpha_per_m": arr,
    "ref_thickness_m": float, "reference": str}}.

    The table's transmittance_internal T(lambda) at ref_thickness_mm is
    converted to an additive absorption coefficient
    alpha = -ln(T)/d_ref [1/m], applied in the tracer's Beer-Lambert bulk
    step — so a filter body 2x thicker than d_ref absorbs correctly, and
    energy lands in the absorbed_bulk ledger bucket. Interpolate
    alpha_per_m linearly (== linear in ln T)."""
    csv_path = Path(csv_path) if csv_path is not None else DEFAULT_FILTERS_CSV
    tables_dir = csv_path.parent / "tables"
    out = {}
    for name, row, ctx in _read_registry(
            csv_path, {"name", "table_csv", "ref_thickness_mm", "reference"},
            "filters"):
        try:
            d_ref_m = float((row.get("ref_thickness_mm") or "").strip()) * 1e-3
        except ValueError:
            raise MaterialError("%s: ref_thickness_mm not numeric" % ctx)
        if d_ref_m <= 0:
            raise MaterialError("%s: ref_thickness_mm must be > 0" % ctx)
        table = _read_table(tables_dir / (row.get("table_csv") or "").strip(),
                            ("transmittance_internal",), ctx)
        T = table["transmittance_internal"]
        if np.any((T <= 0) | (T > 1)):
            raise MaterialError(
                "%s: transmittance_internal must be in (0, 1] — use a small "
                "floor like 1e-6 instead of 0 in stopbands" % ctx)
        out[name] = {"lam_um": table["lam_um"],
                     "alpha_per_m": -np.log(T) / d_ref_m,
                     "ref_thickness_m": d_ref_m,
                     "reference": (row.get("reference") or "").strip()}
    return out


# ---------------------------------------------------------------------------
# detector/detectors.csv + tables/  (quantum-efficiency curves)
# ---------------------------------------------------------------------------
def load_detectors(csv_path=None):
    """-> {name: {"lam_um": arr, "qe": arr, "reference": str, "notes": str}}.

    Each row references a per-detector table wavelength_nm,qe giving the
    fractional quantum efficiency QE(lambda) in (0, 1]. post_process weights
    a detector body's spectral cube by QE(lambda) (via
    detector.spectral_cube_to_photocurrent) to report a photocurrent -- a
    display-stage diagnostic, never a tracer closure bucket."""
    csv_path = Path(csv_path) if csv_path is not None \
        else DEFAULT_DETECTORS_CSV
    tables_dir = csv_path.parent / "tables"
    out = {}
    for name, row, ctx in _read_registry(
            csv_path, {"name", "table_csv", "reference"}, "detectors"):
        table = _read_table(tables_dir / (row.get("table_csv") or "").strip(),
                            ("qe",), ctx)
        qe = table["qe"]
        if np.any(qe <= 0):
            raise MaterialError(
                "%s: qe must be > 0 (a dead band is a data hole, not a "
                "0 -- floor it or drop the point)" % ctx)
        if np.any(qe > 1):
            raise MaterialError(
                "%s: qe must be <= 1 (fractional quantum efficiency, not a "
                "percentage or a gain)" % ctx)
        out[name] = {"lam_um": table["lam_um"], "qe": qe,
                     "reference": (row.get("reference") or "").strip(),
                     "notes": (row.get("notes") or "").strip()}
    return out


# ---------------------------------------------------------------------------
# emission/emitters.miesrc + tables/  (tabulated source emission spectra)
# ---------------------------------------------------------------------------
def load_emission(csv_path=None):
    """-> {name: {"kind": str, "lam_um": arr, "lam_nm": arr,
    "relative_power": arr, "reference": str, "notes": str}}.

    Each row references a per-emitter table wavelength_nm,relative_power
    giving the RELATIVE spectral power density P(lambda) of a source's
    emission (arbitrary units — the sampler in sources.wavelength_strata
    normalizes it to a PDF and places equal-power quantile strata, so only
    the SHAPE matters). Only kind='continuous' (piecewise-linear PDF) is
    supported this round; other staged kinds (blackbody, line) are rejected
    naming the kind. Validation: relative_power >= 0 everywhere, integral of
    P over lambda > 0, >= 2 rows (via _read_table)."""
    csv_path = Path(csv_path) if csv_path is not None \
        else DEFAULT_EMISSION_CSV
    tables_dir = csv_path.parent / "tables"
    out = {}
    for name, row, ctx in _read_registry(
            csv_path, {"name", "kind", "table_csv", "reference"}, "emission"):
        kind = (row.get("kind") or "").strip()
        if kind not in EMISSION_KINDS:
            raise MaterialError(
                "%s: kind %r needs engine support (only %s supported this "
                "round)" % (ctx, kind, ", ".join(EMISSION_KINDS)))
        table = _read_table(tables_dir / (row.get("table_csv") or "").strip(),
                            ("relative_power",), ctx)
        rel = table["relative_power"]
        if np.any(rel < 0):
            raise MaterialError(
                "%s: relative_power must be >= 0 everywhere (a spectral "
                "power density is non-negative)" % ctx)
        if np.trapezoid(rel, table["lam_um"]) <= 0:
            raise MaterialError(
                "%s: relative_power integrates to <= 0 — the table carries no "
                "power" % ctx)
        out[name] = {"kind": kind, "lam_um": table["lam_um"],
                     "lam_nm": table["lam_um"] * 1e3,
                     "relative_power": rel,
                     "reference": (row.get("reference") or "").strip(),
                     "notes": (row.get("notes") or "").strip()}
    return out


# ---------------------------------------------------------------------------
# grating/gratings.csv + tables/
# ---------------------------------------------------------------------------
def load_gratings(csv_path=None):
    """-> {name: {"model": str, "lines_per_mm": float, "params": dict,
    "table": None | {order:int -> {"lam_um": arr, "eta_s": arr,
    "eta_p": arr}}, "reference": str}}.

    Models: bragg_kogelnik (params thickness_um, dn, slant_deg),
    dammann (params transitions=[...] in (0,1)), table (per-order
    polarization-resolved efficiency tables). 'lamellar' rows are allowed
    for completeness but the CLI form is the usual way to get one."""
    csv_path = Path(csv_path) if csv_path is not None else DEFAULT_GRATINGS_CSV
    tables_dir = csv_path.parent / "tables"
    out = {}
    for name, row, ctx in _read_registry(
            csv_path, {"name", "model", "lines_per_mm", "params",
                       "table_csv", "reference"}, "gratings"):
        model = (row.get("model") or "").strip()
        if model not in GRATING_MODELS:
            raise MaterialError("%s: model %r must be one of %s"
                                % (ctx, model, ", ".join(GRATING_MODELS)))
        try:
            lines_per_mm = float((row.get("lines_per_mm") or "").strip())
        except ValueError:
            raise MaterialError("%s: lines_per_mm not numeric" % ctx)
        if lines_per_mm <= 0:
            raise MaterialError("%s: lines_per_mm must be > 0" % ctx)
        params = _parse_params_field(row.get("params"), ctx)
        table = None
        if model == "table":
            tname = (row.get("table_csv") or "").strip()
            if not tname:
                raise MaterialError("%s: model=table requires table_csv" % ctx)
            table = _load_grating_table(tables_dir / tname, ctx)
        elif model == "bragg_kogelnik":
            for req in ("thickness_um", "dn"):
                if req not in params:
                    raise MaterialError(
                        "%s: bragg_kogelnik requires params %r" % (ctx, req))
        elif model == "dammann":
            tr = params.get("transitions")
            if tr is None:
                raise MaterialError("%s: dammann requires params "
                                    "transitions=x1,x2,..." % ctx)
            tr = [tr] if isinstance(tr, float) else list(tr)
            if not all(0.0 < x < 1.0 for x in tr) \
                    or any(b <= a for a, b in zip(tr, tr[1:])):
                raise MaterialError(
                    "%s: dammann transitions must be strictly increasing "
                    "in (0, 1)" % ctx)
            params["transitions"] = tr
        out[name] = {"model": model, "lines_per_mm": lines_per_mm,
                     "params": params, "table": table,
                     "reference": (row.get("reference") or "").strip()}
    return out


def _load_grating_table(path, ctx):
    """Grating efficiency/amplitude table loader.

    Two on-disk formats are supported, distinguished by a schema marker on
    the FIRST line:

      * LEGACY (v1) — no marker; header ``wavelength_nm,order,eta_s,eta_p``.
        REAL per-order power efficiencies interpolated on wavelength only
        (cos_i / azimuth ignored). Returns
        ``{order:int -> {"lam_um", "eta_s", "eta_p"}}`` exactly as before —
        full backward compatibility.

      * v2 (RCWA) — first line ``# mietab grating v2 [side=transmission]``;
        header ``wavelength_nm,theta_deg,phi_deg,order,amp_s_re,amp_s_im,
        amp_p_re,amp_p_im``. COMPLEX per-order amplitudes on a regular
        (lambda, theta, phi) grid, s and p, with |amp|^2 = co-polarized
        order efficiency and arg(amp) = the diffracted-order phase
        (Zemax/Lumerical complex-amplitude interpolation, engine3 §7.5).
        Returns a gridded dict with ``"schema": "v2"``.
    """
    path = Path(path)
    resolved = resolve_prop_path(path, alt_ext=".mietab")
    if not resolved.exists():
        alt = _swap_suffix(path, ".mietab")
        raise MaterialError("%s: grating table not found: %s or %s"
                            % (ctx, path, alt))
    path = resolved
    with open(path, newline="") as fh:
        first = fh.readline()
    if first.lstrip().startswith("#") and "grating v2" in first:
        return _load_grating_table_v2(path, first, ctx)
    return _load_grating_table_v1(path, ctx)


def _load_grating_table_v2(path, marker_line, ctx):
    """Parse a v2 (RCWA complex-amplitude) grating table. See
    _load_grating_table for the format. Returns
    {"schema":"v2", "side":str, "lam_um":arr, "theta_deg":arr,
     "phi_deg":arr, "orders":[int], "amp_s":{m:cplx(nl,nt,np)},
     "amp_p":{m:cplx(nl,nt,np)}}."""
    # marker key=value options (side=transmission|reflection)
    side = "transmission"
    for tok in marker_line.lstrip("#").split():
        if tok.startswith("side="):
            side = tok.split("=", 1)[1].strip()
    if side not in ("transmission", "reflection"):
        raise MaterialError("%s: grating table %s: side=%r must be "
                            "'transmission' or 'reflection'"
                            % (ctx, path, side))
    need = ["wavelength_nm", "theta_deg", "phi_deg", "order",
            "amp_s_re", "amp_s_im", "amp_p_re", "amp_p_im"]
    rows = []
    with open(path, newline="") as fh:
        fh.readline()                       # skip marker line
        reader = csv.DictReader(fh)
        missing = set(need) - set(reader.fieldnames or [])
        if missing:
            raise MaterialError("%s: grating v2 table %s missing column(s) %s"
                                % (ctx, path, sorted(missing)))
        for j, r in enumerate(reader):
            try:
                rows.append((
                    float(r["wavelength_nm"]) * 1e-3,   # lam_um
                    float(r["theta_deg"]), float(r["phi_deg"]),
                    int(r["order"]),
                    complex(float(r["amp_s_re"]), float(r["amp_s_im"])),
                    complex(float(r["amp_p_re"]), float(r["amp_p_im"]))))
            except (TypeError, ValueError, KeyError):
                raise MaterialError("%s: grating v2 table %s row %d not "
                                    "numeric: %r" % (ctx, path, j + 2, r))
    if not rows:
        raise MaterialError("%s: grating v2 table %s is empty" % (ctx, path))
    lam_ax = np.array(sorted({r[0] for r in rows}))
    th_ax = np.array(sorted({r[1] for r in rows}))
    ph_ax = np.array(sorted({r[2] for r in rows}))
    orders = sorted({r[3] for r in rows})
    nl, nt, nph = lam_ax.size, th_ax.size, ph_ax.size
    li = {v: i for i, v in enumerate(lam_ax)}
    ti = {v: i for i, v in enumerate(th_ax)}
    pj = {v: i for i, v in enumerate(ph_ax)}
    amp_s = {m: np.full((nl, nt, nph), np.nan, dtype=np.complex128)
             for m in orders}
    amp_p = {m: np.full((nl, nt, nph), np.nan, dtype=np.complex128)
             for m in orders}
    for lam, th, ph, m, a_s, a_p in rows:
        amp_s[m][li[lam], ti[th], pj[ph]] = a_s
        amp_p[m][li[lam], ti[th], pj[ph]] = a_p
    # every order must fill the FULL regular grid (no holes)
    for m in orders:
        if np.isnan(amp_s[m]).any() or np.isnan(amp_p[m]).any():
            raise MaterialError(
                "%s: grating v2 table %s order %d is not a complete regular "
                "(lambda,theta,phi) grid (%d x %d x %d expected)"
                % (ctx, path, m, nl, nt, nph))
    # energy sanity: co-polarized sum over orders must not exceed 1
    for pol, amp in (("s", amp_s), ("p", amp_p)):
        tot = sum(np.abs(amp[m]) ** 2 for m in orders)
        if np.any(tot > 1.0 + 1e-6):
            worst = float(np.max(tot))
            raise MaterialError(
                "%s: grating v2 table %s: summed |amp_%s|^2 over orders = "
                "%.6f > 1 — energy would not close" % (ctx, path, pol, worst))
    return {"schema": "v2", "side": side, "lam_um": lam_ax,
            "theta_deg": th_ax, "phi_deg": ph_ax, "orders": orders,
            "amp_s": amp_s, "amp_p": amp_p}


def _load_grating_table_v1(path, ctx):
    """LEGACY tables/<name>.mietab: wavelength_nm,order,eta_s,eta_p ->
    {order: {"lam_um", "eta_s", "eta_p"}} with per-order strictly
    increasing wavelength grids."""
    rows_by_order = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        needed = {"wavelength_nm", "order", "eta_s", "eta_p"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise MaterialError("%s: grating table %s missing column(s) %s"
                                % (ctx, path, sorted(missing)))
        for j, r in enumerate(reader):
            try:
                m = int(r["order"])
                lam_um = float(r["wavelength_nm"]) * 1e-3
                es, ep = float(r["eta_s"]), float(r["eta_p"])
            except (TypeError, ValueError):
                raise MaterialError("%s: grating table %s row %d not "
                                    "numeric: %r" % (ctx, path, j + 2, r))
            if not (0.0 <= es <= 1.0 and 0.0 <= ep <= 1.0):
                raise MaterialError("%s: grating table %s row %d eta outside "
                                    "[0,1]" % (ctx, path, j + 2))
            rows_by_order.setdefault(m, []).append((lam_um, es, ep))
    if not rows_by_order:
        raise MaterialError("%s: grating table %s is empty" % (ctx, path))
    out = {}
    for m, rows in rows_by_order.items():
        rows.sort(key=lambda t: t[0])
        lam = np.asarray([t[0] for t in rows])
        if lam.size < 2:
            raise MaterialError("%s: grating table %s order %d has fewer "
                                "than 2 rows" % (ctx, path, m))
        if np.any(np.diff(lam) <= 0):
            raise MaterialError("%s: grating table %s order %d has duplicate "
                                "wavelengths" % (ctx, path, m))
        out[m] = {"lam_um": lam,
                  "eta_s": np.asarray([t[1] for t in rows]),
                  "eta_p": np.asarray([t[2] for t in rows])}
    # energy sanity on the wavelengths shared by every order
    orders = sorted(out)
    common = out[orders[0]]["lam_um"]
    for m in orders[1:]:
        common = np.intersect1d(common, out[m]["lam_um"])
    if common.size:
        tot_s = sum(interp_hard(common, out[m]["lam_um"], out[m]["eta_s"],
                                ctx) for m in orders)
        tot_p = sum(interp_hard(common, out[m]["lam_um"], out[m]["eta_p"],
                                ctx) for m in orders)
        if np.any(tot_s > 1.0 + 1e-9) or np.any(tot_p > 1.0 + 1e-9):
            raise MaterialError(
                "%s: grating table %s: summed order efficiencies exceed 1 "
                "(per polarization) — energy would not close" % (ctx, path))
    return out


# ---------------------------------------------------------------------------
# one-call entry
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# diffuser/diffusers.miedif
# ---------------------------------------------------------------------------
def load_diffusers(csv_path=None):
    """-> {name: {"slope_rms": float, "grit": int|None, "reference": str}}.

    Ground-glass diffuser registry: each row gives EITHER a catalog grit
    number (mapped to an RMS microfacet slope by roughness.slope_for_grit
    at scene-build time) or an explicit slope_rms; if both are present
    they must agree with the mapping to within 20% (a mislabeled row is a
    data error worth failing loudly on)."""
    from .roughness import slope_for_grit
    csv_path = Path(csv_path) if csv_path is not None \
        else DEFAULT_OPTPROPS_DIR / "diffuser" / "diffusers.miedif"
    out = {}
    for name, row, ctx in _read_registry(
            csv_path, {"name", "reference"}, "diffusers"):
        grit_raw = (row.get("grit") or "").strip()
        slope_raw = (row.get("slope_rms") or "").strip()
        grit = int(grit_raw) if grit_raw else None
        slope = float(slope_raw) if slope_raw else None
        if grit is None and slope is None:
            raise MaterialError("%s: needs grit or slope_rms" % ctx)
        if slope is None:
            slope = slope_for_grit(grit)
        elif grit is not None:
            expect = slope_for_grit(grit)
            if abs(slope - expect) > 0.2 * expect:
                raise MaterialError(
                    "%s: slope_rms %.4g disagrees with grit %d "
                    "(calibration gives %.4g)" % (ctx, slope, grit, expect))
        if not 0.0 < slope < 1.0:
            raise MaterialError("%s: slope_rms must be in (0, 1)" % ctx)
        out[name] = {"slope_rms": slope, "grit": grit,
                     "reference": row["reference"]}
    return out


# ---------------------------------------------------------------------------
# scatter/bsdf.miebsdf  (ABg / Harvey-Shack measured-scatter surfaces)
# ---------------------------------------------------------------------------
SCATTER_MODELS = ("abg",)


def _abg_params_or_die(A_raw, B_raw, g_raw, ctx, what):
    """Parse + energy-validate one ABg triple (A>0, B>0, g>0, TIS<=1 at
    normal incidence). Shared by the reflected BRDF and the optional BTDF
    block. Returns (A, B, g)."""
    from .scatter import abg_tis
    try:
        A = float((A_raw or "").strip())
        B = float((B_raw or "").strip())
        g = float((g_raw or "").strip())
    except ValueError:
        raise MaterialError("%s: %s A, B, g must be numeric" % (ctx, what))
    if not A > 0.0:
        raise MaterialError("%s: %s A must be > 0" % (ctx, what))
    if not B > 0.0:
        raise MaterialError("%s: %s B must be > 0" % (ctx, what))
    if not g > 0.0:
        raise MaterialError("%s: %s g must be > 0" % (ctx, what))
    tis0 = abg_tis(A, B, g, 1.0)          # normal incidence = widest umax
    if tis0 > 1.0 + 1e-9:
        raise MaterialError(
            "%s: %s total integrated scatter %.4g exceeds 1 (energy) — the "
            "ABg fit scatters more than the incident power"
            % (ctx, what, tis0))
    return A, B, g


def load_scatter(csv_path=None):
    """-> {name: {"model": "abg", "A": float, "B": float, "g": float,
    "tis_cap": float|None, "btdf": None|{"A","B","g","tis_cap"},
    "reference": str, "notes": str}}.

    ABg BSDF registry for polished optical surfaces (raytracer/scatter.py):
    BSDF(u) = A/(B + u^g), u = |beta - beta0| the direction-cosine offset
    from specular. Each row is validated: A > 0, B > 0, g > 0, and the total
    integrated scatter at normal incidence (scatter.abg_tis, the fraction of
    reflected power that leaves the specular direction) must not exceed 1
    (energy). tis_cap, if given, is an OPTIONAL per-entry ceiling on the TIS
    used by the tracer split (a measured scatter fraction the ABg fit may
    over-integrate); it must itself be in (0, 1].

    BTDF (transmitted-side scatter, OPTIONAL, backward compatible): a row may
    carry a `btdf` flag column plus optional `btdf_A`/`btdf_B`/`btdf_g`/
    `btdf_tis_cap`. When `btdf` is truthy (1/true/yes/on) the transmitted
    child is ALSO split into a specular remainder + a scattered lobe about
    the REFRACTED direction, using the btdf_* ABg triple (each field defaults
    to the reflected A/B/g when left blank). Rows with no `btdf` column (or a
    falsey one) behave EXACTLY as before — reflected-side only."""
    csv_path = Path(csv_path) if csv_path is not None \
        else DEFAULT_OPTPROPS_DIR / "scatter" / "bsdf.miebsdf"
    out = {}
    for name, row, ctx in _read_registry(
            csv_path, {"name", "model", "A", "B", "g", "reference"},
            "scatter"):
        model = (row.get("model") or "").strip()
        if model not in SCATTER_MODELS:
            raise MaterialError("%s: model %r must be one of %s"
                                % (ctx, model, ", ".join(SCATTER_MODELS)))
        A, B, g = _abg_params_or_die(row.get("A"), row.get("B"),
                                     row.get("g"), ctx, "reflected")
        cap_raw = (row.get("tis_cap") or "").strip()
        tis_cap = float(cap_raw) if cap_raw else None
        if tis_cap is not None and not 0.0 < tis_cap <= 1.0:
            raise MaterialError("%s: tis_cap must be in (0, 1]" % ctx)

        # ---- optional transmissive-scatter (BTDF) block ----
        btdf = None
        btdf_raw = (row.get("btdf") or "").strip().lower()
        if btdf_raw in ("1", "true", "yes", "on"):
            # each btdf_* field defaults to the reflected value when blank
            bA = (row.get("btdf_A") or "").strip() or repr(A)
            bB = (row.get("btdf_B") or "").strip() or repr(B)
            bg = (row.get("btdf_g") or "").strip() or repr(g)
            tA, tB, tg = _abg_params_or_die(bA, bB, bg, ctx, "btdf")
            bcap_raw = (row.get("btdf_tis_cap") or "").strip()
            btdf_cap = float(bcap_raw) if bcap_raw else None
            if btdf_cap is not None and not 0.0 < btdf_cap <= 1.0:
                raise MaterialError("%s: btdf_tis_cap must be in (0, 1]" % ctx)
            btdf = {"A": tA, "B": tB, "g": tg, "tis_cap": btdf_cap}
        elif btdf_raw not in ("", "0", "false", "no", "off"):
            raise MaterialError(
                "%s: btdf %r must be a boolean (1/true/yes/on or blank/0)"
                % (ctx, row.get("btdf")))

        out[name] = {"model": model, "A": A, "B": B, "g": g,
                     "tis_cap": tis_cap, "btdf": btdf,
                     "reference": (row.get("reference") or "").strip(),
                     "notes": (row.get("notes") or "").strip()}
    return out


# ---------------------------------------------------------------------------
# nonlinear/nonlinear.mienlo  (chi2 tensors + SHG process rows + Pockels +
# Kerr n2 + saturable absorbers)
# ---------------------------------------------------------------------------
def _read_registry_commented(csv_path, required_cols, what):
    """_read_registry variant that skips full-line '#' comments (the .mienlo
    registry documents its d_il packing in a file-header comment block).
    Same return shape [(name, row, ctx)], with ctx line numbers counted
    against the ORIGINAL file. Quoted fields must not span lines here."""
    csv_path = resolve_prop_path(Path(csv_path))
    if not csv_path.exists():
        raise MaterialError("%s csv not found: %s" % (what, csv_path))
    kept, linenos = [], []
    with open(csv_path, newline="") as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.lstrip().startswith("#"):
                continue
            kept.append(line)
            linenos.append(lineno)
    reader = csv.DictReader(kept)
    missing = required_cols - set(reader.fieldnames or [])
    if missing:
        raise MaterialError("%s csv %s: missing required column(s) %s"
                            % (what, csv_path, sorted(missing)))
    out = []
    for i, row in enumerate(reader):
        lineno = linenos[i + 1] if i + 1 < len(linenos) else linenos[-1]
        name = (row.get("name") or "").strip()
        ctx = "%s line %d (%r)" % (csv_path.name, lineno,
                                   name or "<blank name>")
        if not name:
            raise MaterialError("%s: missing name" % ctx)
        if any(name.lower() == n.lower() for n, _, _ in out):
            raise MaterialError("%s: duplicate name" % ctx)
        reference = (row.get("reference") or "").strip()
        if not reference:
            raise MaterialError("%s: reference is required" % ctx)
        out.append((name, row, ctx))
    return out


def _nlo_float(row, col, ctx, positive=False, nonzero=False,
               nonnegative=False):
    raw = (row.get(col) or "").strip()
    if not raw:
        raise MaterialError("%s: %s is required" % (ctx, col))
    try:
        val = float(raw)
    except ValueError:
        raise MaterialError("%s: %s %r not numeric" % (ctx, col, raw))
    if not np.isfinite(val):
        raise MaterialError("%s: %s must be finite" % (ctx, col))
    if positive and val <= 0:
        raise MaterialError("%s: %s must be > 0 (got %g)" % (ctx, col, val))
    if nonnegative and val < 0:
        raise MaterialError("%s: %s must be >= 0 (got %g)" % (ctx, col, val))
    if nonzero and val == 0:
        raise MaterialError("%s: %s must be non-zero" % (ctx, col))
    return val


def _parse_d_il(raw, ctx):
    """Unpack a d_il_pm_V cell into a (3, 6) float64 array.

    Packing (documented in the .mienlo header too): the full 3x6 contracted
    (Voigt) d-matrix ROW-MAJOR — three '|'-separated rows i = 1..3
    (polarization component), each exactly six ';'-separated floats
    l = 1..6 (11->1, 22->2, 33->3, 23/32->4, 13/31->5, 12/21->6), pm/V,
    crystal principal frame:  d11;d12;...;d16|d21;...;d26|d31;...;d36
    """
    rows = [r.strip() for r in (raw or "").strip().split("|")]
    if len(rows) != 3 or not all(rows):
        raise MaterialError(
            "%s: d_il_pm_V must pack exactly 3 '|'-separated rows "
            "(d11;..;d16|d21;..;d26|d31;..;d36), got %r" % (ctx, raw))
    mat = []
    for i, r in enumerate(rows):
        parts = [p.strip() for p in r.split(";")]
        if len(parts) != 6:
            raise MaterialError(
                "%s: d_il_pm_V row %d must have exactly 6 ';'-separated "
                "entries (got %d)" % (ctx, i + 1, len(parts)))
        try:
            mat.append([float(p) for p in parts])
        except ValueError:
            raise MaterialError("%s: d_il_pm_V row %d not numeric: %r"
                                % (ctx, i + 1, r))
    arr = np.asarray(mat, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise MaterialError("%s: d_il_pm_V entries must be finite" % ctx)
    return arr


def _parse_r_coeffs(raw, ctx):
    """'r63=26.4' / 'r33=30.8;r13=8.6' -> {'r63': 26.4, ...} (pm/V)."""
    out = {}
    for kv in (raw or "").strip().split(";"):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise MaterialError(
                "%s: bad r_coeffs_pm_V entry %r (want rNN=value)" % (ctx, kv))
        k, v = kv.split("=", 1)
        k = k.strip()
        if len(k) < 2 or k[0] != "r" or not k[1:].isdigit():
            raise MaterialError(
                "%s: r_coeffs_pm_V key %r must be an rNN electro-optic "
                "coefficient name" % (ctx, k))
        try:
            out[k] = float(v)
        except ValueError:
            raise MaterialError("%s: r_coeffs_pm_V %r=%r is not numeric"
                                % (ctx, k, v))
    if not out:
        raise MaterialError(
            "%s: r_coeffs_pm_V is required (named ';'-packed pairs, e.g. "
            "'r63=26.4')" % ctx)
    return out


def load_nonlinear(csv_path=None, uniaxial=None, biaxial=None):
    """-> {name: row-dict} from nonlinear/nonlinear.mienlo. One 'kind' key
    discriminates row types (NLO_KINDS); per kind the parsed fields are:

      chi2_tensor : crystal, point_group, d_il_pm_V ((3,6) float64 array —
                    see _parse_d_il for the ';'/'|' packing), kleinman
                    (bool), lam_ref_nm
      chi2_process: crystal, process (NLO_PROCESSES), lam_pump_nm,
                    theta_deg, phi_deg, d_eff_pm_V
      pockels     : crystal, r_pm_V ({'r63': 26.4, ...}), geometry
                    (NLO_GEOMETRIES)
      n2          : material, n2_m2_W, lam_ref_nm
      saturable   : I_sat_W_cm2 (> 0), T0 (unsaturated transmission/
                    reflectance in (0, 1]), tau_recovery_s (>= 0),
                    alpha0_per_mm (float > 0, OPTIONAL -- pulsed-optics
                    Phase P8: the bulk unsaturated absorption coefficient
                    per millimetre, consumed by nlo.saturable_alpha0_per_m.
                    None when the column is blank, which falls back to
                    reading T0 itself as a per-millimetre transmission
                    (alpha0 = -ln(T0)/mm) -- the shipped sam_1550_16_2ps
                    row (a SESAM MIRROR device whose T0 is really a
                    whole-device reflectance) leaves this blank)

    plus 'reference' (required, hard-validated) and 'notes' on every row.
    Full-line '#' comments in the csv are skipped.

    Validation of cross-registry names: 'crystal' (chi2_*/pockels rows) is
    checked against the uniaxial/biaxial registries when either handle is
    passed (load_optical_properties passes both); with neither handle the
    check is skipped so the registry can be loaded standalone. 'material'
    (n2 rows) is NOT resolved here — the Kerr consumer resolves it against
    MaterialDB at use time, LAZILY BY DESIGN: staged rows (n2_yag) may
    precede their materials.miemat index row.
    """
    csv_path = Path(csv_path) if csv_path is not None else DEFAULT_NONLINEAR_CSV
    known = None
    if uniaxial is not None or biaxial is not None:
        known = {str(k).strip().lower() for k in (uniaxial or {})} \
            | {str(k).strip().lower() for k in (biaxial or {})}
    out = {}
    for name, row, ctx in _read_registry_commented(
            csv_path, {"name", "kind", "reference"}, "nonlinear"):
        kind = (row.get("kind") or "").strip()
        if kind not in NLO_KINDS:
            raise MaterialError("%s: kind %r must be one of %s"
                                % (ctx, kind, ", ".join(NLO_KINDS)))
        entry = {"kind": kind,
                 "reference": (row.get("reference") or "").strip(),
                 "notes": (row.get("notes") or "").strip()}
        if kind in ("chi2_tensor", "chi2_process", "pockels"):
            crystal = (row.get("crystal") or "").strip()
            if not crystal:
                raise MaterialError("%s: crystal is required for kind=%s"
                                    % (ctx, kind))
            if known is not None and crystal.lower() not in known:
                raise MaterialError(
                    "%s: crystal %r is not a birefringence registry row "
                    "(uniaxial.miebrf / biaxial.mibiax)" % (ctx, crystal))
            entry["crystal"] = crystal
        if kind == "chi2_tensor":
            point_group = (row.get("point_group") or "").strip()
            if not point_group:
                raise MaterialError("%s: point_group is required" % ctx)
            kl = (row.get("kleinman") or "").strip().lower()
            if kl not in ("true", "false"):
                raise MaterialError("%s: kleinman must be 'true' or 'false' "
                                    "(got %r)" % (ctx, kl))
            entry.update(
                point_group=point_group,
                d_il_pm_V=_parse_d_il(row.get("d_il_pm_V"), ctx),
                kleinman=(kl == "true"),
                lam_ref_nm=_nlo_float(row, "lam_ref_nm", ctx, positive=True))
        elif kind == "chi2_process":
            process = (row.get("process") or "").strip()
            if process not in NLO_PROCESSES:
                raise MaterialError("%s: process %r must be one of %s"
                                    % (ctx, process, ", ".join(NLO_PROCESSES)))
            entry.update(
                process=process,
                lam_pump_nm=_nlo_float(row, "lam_pump_nm", ctx,
                                       positive=True),
                theta_deg=_nlo_float(row, "theta_deg", ctx),
                phi_deg=_nlo_float(row, "phi_deg", ctx),
                d_eff_pm_V=_nlo_float(row, "d_eff_pm_V", ctx, nonzero=True))
        elif kind == "pockels":
            geometry = (row.get("geometry") or "").strip()
            if geometry not in NLO_GEOMETRIES:
                raise MaterialError("%s: geometry %r must be one of %s"
                                    % (ctx, geometry,
                                       ", ".join(NLO_GEOMETRIES)))
            entry.update(
                r_pm_V=_parse_r_coeffs(row.get("r_coeffs_pm_V"), ctx),
                geometry=geometry)
        elif kind == "n2":
            material = (row.get("material") or "").strip()
            if not material:
                raise MaterialError("%s: material is required for kind=n2"
                                    % ctx)
            entry.update(
                material=material,
                n2_m2_W=_nlo_float(row, "n2_m2_W", ctx, nonzero=True),
                lam_ref_nm=_nlo_float(row, "lam_ref_nm", ctx, positive=True))
        elif kind == "saturable":
            T0 = _nlo_float(row, "T0", ctx, positive=True)
            if T0 > 1.0:
                raise MaterialError(
                    "%s: T0 must be in (0, 1] (a fractional unsaturated "
                    "transmission/reflectance, got %g)" % (ctx, T0))
            alpha0_raw = (row.get("alpha0_per_mm") or "").strip()
            alpha0_per_mm = (_nlo_float(row, "alpha0_per_mm", ctx,
                                        positive=True)
                             if alpha0_raw else None)
            entry.update(
                I_sat_W_cm2=_nlo_float(row, "I_sat_W_cm2", ctx,
                                       positive=True),
                T0=T0,
                tau_recovery_s=_nlo_float(row, "tau_recovery_s", ctx,
                                          nonnegative=True),
                alpha0_per_mm=alpha0_per_mm)
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# instrument/instruments.mieinst + tables/  (virtual instrument layer,
# engine3.md §9 -- P2.5)
#
# A "row" here is a parametrized RESPONSE model of a real (or not-yet-owned)
# bench instrument, read as a POST-PROCESS layer over an ideal detector
# plane (post_process.render_instrument). Every row carries a 'class'
# discriminator (INSTRUMENT_CLASSES); each class has its own required
# columns, validated below. Because the classes need very different
# parameters, the registry csv is intentionally WIDE (one header covers
# every class; a given row leaves the columns of every OTHER class blank)
# -- the same "one shared header, sparse per-row" shape as this repo's
# nonlinear/nonlinear.mienlo registry.
#
# PLACEHOLDER CLASSES: 'polarimeter', 'wavefront_sensor', 'autocorrelator'
# have their column schemas defined and validated below (so a future
# authored row is hard-checked exactly like the shipped ones), but the
# shipped instruments.mieinst carries NO rows of these classes yet -- the
# owner does not have this bench gear (engine3.md §9.2 table, "bench twin
# exists: not yet owned"). Do not remove the placeholder validation branches
# just because len(instruments) shows no polarimeter/wavefront_sensor/
# autocorrelator rows; they exist so the FIRST row of that class is caught
# by the loader instead of silently mis-parsed by post_process.
# ---------------------------------------------------------------------------
DEFAULT_INSTRUMENTS_CSV = DEFAULT_OPTPROPS_DIR / "instrument" / "instruments.mieinst"

INSTRUMENT_CLASSES = ("camera", "powermeter", "spectrometer",
                      "polarimeter", "wavefront_sensor", "autocorrelator")
# classes with a schema but (by design) no shipped rows this round -- see
# module note above.
PLACEHOLDER_INSTRUMENT_CLASSES = ("polarimeter", "wavefront_sensor",
                                  "autocorrelator")


def _reg_int(row, col, ctx, positive=False):
    """Like _nlo_float but for an integer-valued registry column (pixel
    counts, bit depth, digit counts, analyzer-state counts)."""
    raw = (row.get(col) or "").strip()
    if not raw:
        raise MaterialError("%s: %s is required" % (ctx, col))
    try:
        val = int(raw)
    except ValueError:
        raise MaterialError("%s: %s %r is not an integer" % (ctx, col, raw))
    if positive and val <= 0:
        raise MaterialError("%s: %s must be > 0 (got %d)" % (ctx, col, val))
    return val


def load_instruments(csv_path=None):
    """-> {name: row-dict} from instrument/instruments.mieinst. One 'class'
    key discriminates row types (INSTRUMENT_CLASSES); per class the parsed
    fields are:

      camera      : pixel_pitch_um, width_px, height_px (int),
                    fill_factor (0,1], qe (from qe_table: {"lam_um","qe"}),
                    full_well_e, read_noise_e (>=0), dark_current_e_per_s
                    (>=0), bit_depth (int>0), adc_gain_e_per_dn,
                    integration_time_s_default
      powermeter  : EXACTLY ONE of {"resp_table": {"lam_um",
                    "responsivity_a_w"}, "flat_responsivity_a_w": float} is
                    set (the other is None), aperture_mm, nep_w_per_sqrthz,
                    bandwidth_hz (>0), display_digits (int>=1)
      spectrometer: lam_lo_nm < lam_hi_nm, resolution_fwhm_nm (>0),
                    slit_um (>0), stray_light_floor ([0,1)),
                    qe (from detector_qe_table: {"lam_um","qe"})
      polarimeter (PLACEHOLDER, no shipped rows): analyzer_states (int>=2),
                    extinction_ratio (>0), retarder_error_deg (>=0)
      wavefront_sensor (PLACEHOLDER, no shipped rows): opd_sampling_um (>0),
                    reference_arm_model (non-empty str)
      autocorrelator (PLACEHOLDER, no shipped rows): shg_crystal (non-empty
                    str), delay_range_fs (>0)

    plus 'reference' (required, hard-validated) and 'notes' on every row.
    Table columns (qe_table/responsivity_table/detector_qe_table) resolve
    against tables/ next to the csv via _read_table, exactly like
    load_detectors/load_polarizers."""
    csv_path = Path(csv_path) if csv_path is not None \
        else DEFAULT_INSTRUMENTS_CSV
    tables_dir = csv_path.parent / "tables"
    out = {}
    for name, row, ctx in _read_registry(
            csv_path, {"name", "class", "reference"}, "instruments"):
        klass = (row.get("class") or "").strip()
        if klass not in INSTRUMENT_CLASSES:
            raise MaterialError("%s: class %r must be one of %s"
                                % (ctx, klass, ", ".join(INSTRUMENT_CLASSES)))
        entry = {"class": klass,
                 "reference": (row.get("reference") or "").strip(),
                 "notes": (row.get("notes") or "").strip()}
        if klass == "camera":
            table = _read_table(
                tables_dir / (row.get("qe_table") or "").strip(),
                ("qe",), ctx)
            qe = table["qe"]
            if np.any(qe <= 0) or np.any(qe > 1):
                raise MaterialError(
                    "%s: qe_table qe must be in (0, 1]" % ctx)
            entry.update(
                pixel_pitch_um=_nlo_float(row, "pixel_pitch_um", ctx,
                                          positive=True),
                width_px=_reg_int(row, "width_px", ctx, positive=True),
                height_px=_reg_int(row, "height_px", ctx, positive=True),
                fill_factor=_nlo_float(row, "fill_factor", ctx,
                                       positive=True),
                lam_um=table["lam_um"], qe=qe,
                full_well_e=_nlo_float(row, "full_well_e", ctx,
                                       positive=True),
                read_noise_e=_nlo_float(row, "read_noise_e", ctx,
                                        nonnegative=True),
                dark_current_e_per_s=_nlo_float(
                    row, "dark_current_e_per_s", ctx, nonnegative=True),
                bit_depth=_reg_int(row, "bit_depth", ctx, positive=True),
                adc_gain_e_per_dn=_nlo_float(row, "adc_gain_e_per_dn", ctx,
                                             positive=True),
                integration_time_s_default=_nlo_float(
                    row, "integration_time_s_default", ctx, positive=True))
            if entry["fill_factor"] > 1.0:
                raise MaterialError("%s: fill_factor must be in (0, 1]" % ctx)
        elif klass == "powermeter":
            resp_raw = (row.get("responsivity_table") or "").strip()
            flat_raw = (row.get("flat_responsivity_a_w") or "").strip()
            if bool(resp_raw) == bool(flat_raw):
                raise MaterialError(
                    "%s: exactly one of responsivity_table / "
                    "flat_responsivity_a_w is required" % ctx)
            resp_table = None
            flat_resp = None
            if resp_raw:
                table = _read_table(tables_dir / resp_raw,
                                    ("responsivity_a_w",), ctx)
                r = table["responsivity_a_w"]
                if np.any(r <= 0):
                    raise MaterialError(
                        "%s: responsivity_a_w must be > 0" % ctx)
                resp_table = {"lam_um": table["lam_um"],
                             "responsivity_a_w": r}
            else:
                flat_resp = _nlo_float(row, "flat_responsivity_a_w", ctx,
                                       positive=True)
            entry.update(
                resp_table=resp_table, flat_responsivity_a_w=flat_resp,
                aperture_mm=_nlo_float(row, "aperture_mm", ctx,
                                       positive=True),
                nep_w_per_sqrthz=_nlo_float(row, "nep_w_per_sqrthz", ctx,
                                            positive=True),
                bandwidth_hz=_nlo_float(row, "bandwidth_hz", ctx,
                                        positive=True),
                display_digits=_reg_int(row, "display_digits", ctx,
                                        positive=True))
        elif klass == "spectrometer":
            lam_lo = _nlo_float(row, "lam_lo_nm", ctx, positive=True)
            lam_hi = _nlo_float(row, "lam_hi_nm", ctx, positive=True)
            if lam_hi <= lam_lo:
                raise MaterialError(
                    "%s: lam_hi_nm must be > lam_lo_nm" % ctx)
            stray = _nlo_float(row, "stray_light_floor", ctx,
                               nonnegative=True)
            if stray >= 1.0:
                raise MaterialError(
                    "%s: stray_light_floor must be in [0, 1)" % ctx)
            table = _read_table(
                tables_dir / (row.get("detector_qe_table") or "").strip(),
                ("qe",), ctx)
            qe = table["qe"]
            if np.any(qe <= 0) or np.any(qe > 1):
                raise MaterialError(
                    "%s: detector_qe_table qe must be in (0, 1]" % ctx)
            entry.update(
                lam_lo_nm=lam_lo, lam_hi_nm=lam_hi,
                resolution_fwhm_nm=_nlo_float(row, "resolution_fwhm_nm", ctx,
                                              positive=True),
                slit_um=_nlo_float(row, "slit_um", ctx, positive=True),
                stray_light_floor=stray,
                lam_um=table["lam_um"], qe=qe)
        elif klass == "polarimeter":
            entry.update(
                analyzer_states=_reg_int(row, "analyzer_states", ctx,
                                         positive=True),
                extinction_ratio=_nlo_float(row, "extinction_ratio", ctx,
                                            positive=True),
                retarder_error_deg=_nlo_float(row, "retarder_error_deg", ctx,
                                              nonnegative=True))
            if entry["analyzer_states"] < 2:
                raise MaterialError(
                    "%s: analyzer_states must be >= 2" % ctx)
        elif klass == "wavefront_sensor":
            ref_arm = (row.get("reference_arm_model") or "").strip()
            if not ref_arm:
                raise MaterialError(
                    "%s: reference_arm_model is required" % ctx)
            entry.update(
                opd_sampling_um=_nlo_float(row, "opd_sampling_um", ctx,
                                           positive=True),
                reference_arm_model=ref_arm)
        elif klass == "autocorrelator":
            crystal = (row.get("shg_crystal") or "").strip()
            if not crystal:
                raise MaterialError("%s: shg_crystal is required" % ctx)
            entry.update(
                shg_crystal=crystal,
                delay_range_fs=_nlo_float(row, "delay_range_fs", ctx,
                                          positive=True))
        out[name] = entry
    return out


class OpticalProperties:
    """Everything loaded from an opticalproperties/ root. Attributes:
    matdb (MaterialDB, with uniaxial attached), coatings, polarizers,
    filters, gratings, uniaxial — shapes per the load_* docstrings."""

    __slots__ = ("root", "matdb", "coatings", "polarizers", "filters",
                 "gratings", "uniaxial", "biaxial", "diffusers", "detectors",
                 "scatter", "emission", "nonlinear", "instruments")

    def __init__(self, root, matdb, coatings, polarizers, filters, gratings,
                 uniaxial, diffusers=None, detectors=None, biaxial=None,
                 scatter=None, emission=None, nonlinear=None,
                 instruments=None):
        self.root = root
        self.matdb = matdb
        self.coatings = coatings
        self.polarizers = polarizers
        self.filters = filters
        self.gratings = gratings
        self.uniaxial = uniaxial
        self.biaxial = biaxial if biaxial is not None else {}
        self.diffusers = diffusers if diffusers is not None else {}
        self.detectors = detectors if detectors is not None else {}
        self.scatter = scatter if scatter is not None else {}
        self.emission = emission if emission is not None else {}
        self.nonlinear = nonlinear if nonlinear is not None else {}
        self.instruments = instruments if instruments is not None else {}


def load_optical_properties(root=None, db=None):
    """Load the full opticalproperties/ library. Missing OPTIONAL category
    csvs (polarizer/filter/grating/birefringence) load as empty dicts so a
    trimmed-down library still works; materials.csv and coatings.csv are
    required."""
    root = Path(root) if root is not None else DEFAULT_OPTPROPS_DIR
    if db is None:
        db = MaterialDB.load(csv_path=root / "materials.miemat",
                             nk_dir=root / "nk")
    coatings = load_coatings(csv_path=root / "coating" / "coatings.miecoat",
                             db=db)

    def optional(loader, path, **kw):
        # path is the preferred (new-extension) name; also accept a
        # pure-legacy library that only has the .csv sibling. Check
        # existence directly (no NOTE here) -- the loader's own
        # resolve_prop_path call emits the one-line legacy NOTE.
        present = path.exists() or _swap_suffix(path, ".csv").exists()
        return loader(csv_path=path, **kw) if present else {}

    uniaxial = optional(load_uniaxial,
                        root / "birefringence" / "uniaxial.miebrf", db=db)
    db.attach_uniaxial(uniaxial)
    biaxial = optional(load_biaxial,
                       root / "birefringence" / "biaxial.mibiax", db=db)
    db.attach_biaxial(biaxial)
    return OpticalProperties(
        root=root, matdb=db, coatings=coatings,
        polarizers=optional(load_polarizers,
                            root / "polarizer" / "polarizers.miepol"),
        filters=optional(load_filters, root / "filter" / "filters.miefilt"),
        gratings=optional(load_gratings, root / "grating" / "gratings.miegrat"),
        uniaxial=uniaxial,
        biaxial=biaxial,
        diffusers=optional(load_diffusers,
                           root / "diffuser" / "diffusers.miedif"),
        detectors=optional(load_detectors,
                           root / "detector" / "detectors.miedet"),
        scatter=optional(load_scatter, root / "scatter" / "bsdf.miebsdf"),
        emission=optional(load_emission,
                          root / "emission" / "emitters.miesrc"),
        nonlinear=optional(load_nonlinear,
                           root / "nonlinear" / "nonlinear.mienlo",
                           uniaxial=uniaxial, biaxial=biaxial),
        instruments=optional(load_instruments,
                             root / "instrument" / "instruments.mieinst"))


# ---------------------------------------------------------------------------
# Self-check: /home3/optics/env/bin/python -m raytracer.optprops  (from scripts/)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    props = load_optical_properties()
    print("opticalproperties root:", props.root)
    print("  materials : %d" % len(props.matdb))
    print("  coatings  : %d  (%s)" % (len(props.coatings),
                                      ", ".join(sorted(props.coatings))))
    print("  polarizers: %d" % len(props.polarizers))
    print("  filters   : %d" % len(props.filters))
    print("  gratings  : %d" % len(props.gratings))
    print("  uniaxial  : %d" % len(props.uniaxial))
    print("  biaxial   : %d" % len(props.biaxial))
    print("  diffusers : %d" % len(props.diffusers))
    print("  detectors : %d" % len(props.detectors))
    print("  scatter   : %d" % len(props.scatter))
    print("  emission  : %d  (%s)" % (len(props.emission),
                                      ", ".join(sorted(props.emission))))
    print("  nonlinear : %d  (%s)" % (len(props.nonlinear),
                                      ", ".join(sorted(props.nonlinear))))
    print("  instruments: %d  (%s)" % (len(props.instruments),
                                       ", ".join(sorted(props.instruments))))
