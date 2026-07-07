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
DEFAULT_POLARIZERS_CSV = DEFAULT_OPTPROPS_DIR / "polarizer" / "polarizers.miepol"
DEFAULT_FILTERS_CSV = DEFAULT_OPTPROPS_DIR / "filter" / "filters.miefilt"
DEFAULT_GRATINGS_CSV = DEFAULT_OPTPROPS_DIR / "grating" / "gratings.miegrat"

POLARIZER_TYPES = ("linear", "circular_left", "circular_right")
GRATING_MODELS = ("lamellar", "bragg_kogelnik", "dammann", "table")


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
    """tables/<name>.csv: wavelength_nm,order,eta_s,eta_p ->
    {order: {"lam_um", "eta_s", "eta_p"}} with per-order strictly
    increasing wavelength grids."""
    path = Path(path)
    resolved = resolve_prop_path(path, alt_ext=".mietab")
    if not resolved.exists():
        alt = _swap_suffix(path, ".mietab")
        raise MaterialError("%s: grating table not found: %s or %s"
                            % (ctx, path, alt))
    path = resolved
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


class OpticalProperties:
    """Everything loaded from an opticalproperties/ root. Attributes:
    matdb (MaterialDB, with uniaxial attached), coatings, polarizers,
    filters, gratings, uniaxial — shapes per the load_* docstrings."""

    __slots__ = ("root", "matdb", "coatings", "polarizers", "filters",
                 "gratings", "uniaxial", "diffusers")

    def __init__(self, root, matdb, coatings, polarizers, filters, gratings,
                 uniaxial, diffusers=None):
        self.root = root
        self.matdb = matdb
        self.coatings = coatings
        self.polarizers = polarizers
        self.filters = filters
        self.gratings = gratings
        self.uniaxial = uniaxial
        self.diffusers = diffusers if diffusers is not None else {}


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
    return OpticalProperties(
        root=root, matdb=db, coatings=coatings,
        polarizers=optional(load_polarizers,
                            root / "polarizer" / "polarizers.miepol"),
        filters=optional(load_filters, root / "filter" / "filters.miefilt"),
        gratings=optional(load_gratings, root / "grating" / "gratings.miegrat"),
        uniaxial=uniaxial,
        diffusers=optional(load_diffusers,
                           root / "diffuser" / "diffusers.miedif"))


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
    print("  diffusers : %d" % len(props.diffusers))
