#!/usr/bin/env python3
# =============================================================================
# materials.py -- optical materials database + dispersion evaluation for the
# ray tracer pipeline.
#
# Reads materials.csv (dispersion model + parameters per material) and
# nk_data/*.csv (tabulated n,k spectra for metals / TiO2 / water-k / etc.)
# and exposes:
#
#   Material     one row of materials.csv -> n_complex(lambda_m) evaluator
#   MaterialDB   name -> Material, loaded + hard-validated from materials.csv
#   CoatingSpec  one TMM coatings.csv row (ordered list of material:thickness)
#   load_coatings(csv_path, db) -> {name: {"kind": "tmm"|"table", ...}}
#                (see load_coatings docstring for the exact spec shapes)
#   plot_dispersion(db, names, out_png)  diagnostic n/k vs wavelength plot
#
# Only numpy is a hard import dependency (matplotlib is imported lazily
# inside plot_dispersion). No pandas, no FreeCAD, no torch.
#
# Units: materials.csv Sellmeier/Cauchy formulas take lambda in micrometres
# (um), matching the standard literature convention (refractiveindex.info
# etc). n_complex() takes/returns SI wavelengths in metres, as used
# throughout the rest of the ray tracer.
# =============================================================================
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Default paths (this file lives at <project>/scripts/raytracer/materials.py)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent.parent
DEFAULT_OPTPROPS_DIR = _PROJECT_DIR / "opticalproperties"
DEFAULT_MATERIALS_CSV = DEFAULT_OPTPROPS_DIR / "materials.miemat"
DEFAULT_COATINGS_CSV = DEFAULT_OPTPROPS_DIR / "coating" / "coatings.miecoat"
DEFAULT_NK_DIR = DEFAULT_OPTPROPS_DIR / "nk"


# ---------------------------------------------------------------------------
# Self-describing extensions (.miemat/.mienk/.miecoat/.mietab/...) with a
# backward-compatible '.csv' fallback -- content is plain CSV either way.
#
# resolve_prop_path() handles the top-level registry files (materials,
# coatings): callers always ask for the new extension first; if that's
# missing, the legacy same-stem .csv sibling is used instead (one NOTE to
# stderr), so an external all-.csv --optical-properties library keeps
# working unmodified.
#
# _swap_suffix()/resolve_prop_path() also handle per-item files named in a
# registry row (nk_file / coating 'table' column): the row may name either
# extension while the file on disk uses the other; exact name is tried
# first, then the swapped extension, then a hard error naming both.
# ---------------------------------------------------------------------------
def _swap_suffix(path, alt_ext):
    path = Path(path)
    alt_ext = alt_ext if alt_ext.startswith(".") else "." + alt_ext
    other = ".csv" if path.suffix.lower() == alt_ext else alt_ext
    return path.with_suffix(other)


def resolve_prop_path(path, alt_ext=".csv"):
    """Return an existing sibling of `path`: `path` itself if present,
    else the sibling with its suffix swapped between '.csv' and alt_ext
    if THAT exists (emitting a legacy-format NOTE to stderr when we fall
    back to a plain '.csv' file), else `path` unchanged so the caller's
    own "not found" error names the originally intended file."""
    path = Path(path)
    if path.exists():
        return path
    alt = _swap_suffix(path, alt_ext)
    if alt.exists():
        if alt.suffix.lower() == ".csv" and path.suffix.lower() != ".csv":
            sys.stderr.write(
                "NOTE: using legacy %s; rename to %s\n" % (alt, path))
        return alt
    return path

VALID_CLASSES = {"gas", "glass", "liquid", "polymer", "metal", "oxide",
                  "film", "special"}
VALID_MODELS = {"sellmeier", "schott", "cauchy", "constant", "tabulated"}

# Reference temperature (deg C) assumed for a material's dispersion data when
# the row carries no explicit thermo_t_ref. Optical glass catalogs (Schott,
# Ohara) tabulate their index at 20 C.
DEFAULT_T_REF_C = 20.0

# Thermo-optic column names in a .miemat row (all optional; a row is treated
# as having no dn/dT model unless at least one of D0..E1 is present).
_THERMO_KEYS = ("thermo_d0", "thermo_d1", "thermo_d2",
                "thermo_e0", "thermo_e1", "thermo_lambda_tk")

# Materials allowed to have density <= 0 (vacuum / sentinel detector row).
ZERO_DENSITY_OK = {"vacuum", "detector"}

_PARAM_KEYS = ("p1", "p2", "p3", "p4", "p5", "p6")


class MaterialError(ValueError):
    """Hard-validation failure. Always names the offending row/material."""


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------
class Material:
    """A single material: dispersion model + params, optional nk table.

    n_complex(lam_m) evaluates the model (vectorized) and returns
    complex128 n + i*k. k >= 0 by convention (absorbing media).
    """

    def __init__(self, name, cls, model, params, density, nk_file=None,
                 trans_min=None, trans_max=None, notes="", reference="",
                 thermo=None, t_ref_c=DEFAULT_T_REF_C):
        self.name = name
        self.cls = cls
        self.model = model
        self.params = tuple(params)          # 6 floats, unused slots = nan
        self.density = density
        self.nk_file = nk_file               # filename (str) or None
        self.trans_min = trans_min           # um or None
        self.trans_max = trans_max           # um or None
        self.notes = notes
        self.reference = reference
        # thermo-optic (Schott TIE-19 dn_abs/dT model), or None if the row
        # carried no D0..E1 coefficients. tuple (D0, D1, D2, E0, E1, lam_tk_um).
        self.thermo = tuple(thermo) if thermo is not None else None
        self.t_ref_c = float(t_ref_c)        # dispersion-data reference temp
        # tabulated nk data, populated by MaterialDB.load if nk_file is set
        self.nk_lambda_um = None             # ascending float array
        self.nk_n = None
        self.nk_k = None

    @property
    def has_thermo(self):
        return self.thermo is not None

    def __repr__(self):
        return "Material(%r, class=%r, model=%r)" % (self.name, self.cls,
                                                       self.model)

    # -- dispersion model (n from parametric formula) --------------------
    def _n_from_model(self, lam_um):
        p = self.params
        if self.model == "sellmeier":
            l2 = lam_um ** 2
            n2 = np.ones_like(l2)
            for b, c in ((p[0], p[3]), (p[1], p[4]), (p[2], p[5])):
                n2 = n2 + b * l2 / (l2 - c)
            return np.sqrt(n2)
        if self.model == "schott":
            # Legacy Schott power-series: n^2 = a0 + a1*l^2 + a2*l^-2
            #                                  + a3*l^-4 + a4*l^-6 + a5*l^-8
            # (params p1..p6 = a0..a5; lam in um). Used by many older glass
            # catalog rows before the Sellmeier-1 form.
            l2 = lam_um ** 2
            n2 = (p[0] + p[1] * l2 + p[2] / l2
                  + p[3] / l2 ** 2 + p[4] / l2 ** 3 + p[5] / l2 ** 4)
            return np.sqrt(n2)
        if self.model == "cauchy":
            a, b, c = p[0], p[1], p[2]
            return a + b / lam_um ** 2 + c / lam_um ** 4
        if self.model == "constant":
            return np.full_like(lam_um, p[0], dtype=np.float64)
        raise MaterialError(
            "material %r: model %r has no parametric n formula "
            "(use n_complex only for tabulated data)" % (self.name, self.model))

    # -- k interpolation from a tabulated file (log-linear, k=0 fallback) -
    def _k_from_table(self, lam_um):
        lam_tab, k_tab = self.nk_lambda_um, self.nk_k
        lo, hi = lam_tab[0], lam_tab[-1]
        out_of_range = (lam_um < lo) | (lam_um > hi)
        if np.any(out_of_range):
            bad = np.atleast_1d(lam_um)[out_of_range][0]
            raise MaterialError(
                "material %r: k requested at %.6g um is outside tabulated "
                "nk_file %r range [%.6g, %.6g] um -- no extrapolation"
                % (self.name, bad, self.nk_file, lo, hi))
        if np.all(k_tab > 0):
            return np.exp(np.interp(lam_um, lam_tab, np.log(k_tab)))
        if np.all(k_tab == 0):
            return np.zeros_like(lam_um)
        # mixed zero / nonzero k values -- log-linear is undefined, fall
        # back to plain linear interpolation for the whole table.
        return np.interp(lam_um, lam_tab, k_tab)

    def _n_from_table(self, lam_um):
        lam_tab, n_tab = self.nk_lambda_um, self.nk_n
        lo, hi = lam_tab[0], lam_tab[-1]
        out_of_range = (lam_um < lo) | (lam_um > hi)
        if np.any(out_of_range):
            bad = np.atleast_1d(lam_um)[out_of_range][0]
            raise MaterialError(
                "material %r: n requested at %.6g um is outside tabulated "
                "nk_file %r range [%.6g, %.6g] um -- no extrapolation"
                % (self.name, bad, self.nk_file, lo, hi))
        return np.interp(lam_um, lam_tab, n_tab)

    def _warn_outside_transmission(self, lam_um):
        if self.trans_min is None or self.trans_max is None:
            return
        bad = (lam_um < self.trans_min) | (lam_um > self.trans_max)
        if np.any(bad):
            first_bad = float(np.atleast_1d(lam_um)[bad][0])
            warnings.warn(
                "material %r: evaluating n at %.6g um, outside stated "
                "transmission window [%.6g, %.6g] um (%s)"
                % (self.name, first_bad, self.trans_min, self.trans_max,
                   self.reference),
                stacklevel=3)

    def _dn_thermal(self, lam_um, n_ref, T_c):
        """Schott TIE-19 absolute-index change dn_abs(lambda, T) relative to
        the dispersion-data reference temperature. lam_um in um, n_ref the
        index at t_ref_c, T_c the target temperature in deg C. Returns the
        (signed) index increment to add to n_ref. Requires self.thermo."""
        D0, D1, D2, E0, E1, lam_tk = self.thermo
        dT = T_c - self.t_ref_c
        l2 = lam_um ** 2
        denom = l2 - lam_tk ** 2
        return ((n_ref ** 2 - 1.0) / (2.0 * n_ref)
                * (D0 * dT + D1 * dT ** 2 + D2 * dT ** 3
                   + (E0 * dT + E1 * dT ** 2) / denom))

    def n_complex(self, lam_m, T=None):
        """lam_m: scalar or array-like of wavelengths in metres (SI).
        T: optional temperature in deg C. When given (and the material carries
        a thermo-optic model), the real index is shifted by the Schott TIE-19
        dn_abs(lambda, T) term relative to t_ref_c; k is unaffected. T=None (or
        a material without thermo data, or T == t_ref_c) leaves n unchanged.
        Returns complex128 (scalar or array, matching input shape) n + i*k.
        Hard-raises MaterialError if a tabulated nk_file's range is
        exceeded (no silent extrapolation). Warns (does not raise) if a
        parametric model is evaluated outside transmission_um_min/max.
        """
        lam_m = np.asarray(lam_m, dtype=np.float64)
        scalar_in = (lam_m.ndim == 0)
        lam_um = np.atleast_1d(lam_m).astype(np.float64) * 1e6

        if self.model == "tabulated":
            n = self._n_from_table(lam_um)
            k = self._k_from_table(lam_um)
        else:
            self._warn_outside_transmission(lam_um)
            n = self._n_from_model(lam_um)
            if self.nk_lambda_um is not None:
                k = self._k_from_table(lam_um)
            elif self.model == "constant":
                k = np.full_like(lam_um, self.params[1])
            else:
                k = np.zeros_like(lam_um)

        if T is not None and self.has_thermo and float(T) != self.t_ref_c:
            n = n + self._dn_thermal(lam_um, n, float(T))

        out = n.astype(np.complex128) + 1j * k.astype(np.complex128)
        return complex(out[0]) if scalar_in else out

    # -- dispersion derivatives (group index / GDD support) ---------------
    def _dn_dlam_um(self, lam_um):
        """d Re(n)/d(lambda) per um. Analytic for sellmeier/cauchy, zero for
        constant; for tabulated, the linear interpolant's knot-gradient
        interpolated back onto lam_um (no extrapolation -- the table-range
        check in _n_from_table applies to the same range)."""
        if self.model == "sellmeier":
            l2 = lam_um ** 2
            n = self._n_from_model(lam_um)
            p = self.params
            acc = np.zeros_like(lam_um)
            for b, c in ((p[0], p[3]), (p[1], p[4]), (p[2], p[5])):
                acc = acc + (-2.0 * b * c * lam_um) / (l2 - c) ** 2
            return acc / (2.0 * n)
        if self.model == "schott":
            # d(n^2)/dlam = 2 a1 lam - 2 a2 lam^-3 - 4 a3 lam^-5
            #               - 6 a4 lam^-7 - 8 a5 lam^-9 ; dn/dlam = that/(2n)
            p = self.params
            n = self._n_from_model(lam_um)
            dn2 = (2.0 * p[1] * lam_um - 2.0 * p[2] / lam_um ** 3
                   - 4.0 * p[3] / lam_um ** 5 - 6.0 * p[4] / lam_um ** 7
                   - 8.0 * p[5] / lam_um ** 9)
            return dn2 / (2.0 * n)
        if self.model == "cauchy":
            b, c = self.params[1], self.params[2]
            return -2.0 * b / lam_um ** 3 - 4.0 * c / lam_um ** 5
        if self.model == "constant":
            return np.zeros_like(lam_um)
        # tabulated
        self._n_from_table(lam_um)           # range check (hard-raises)
        grad = np.gradient(self.nk_n, self.nk_lambda_um)
        return np.interp(lam_um, self.nk_lambda_um, grad)

    def _stencil_um(self, lam_um):
        """(centre, h) for numeric second/third derivatives, clamped so the
        centre +- h stencil stays inside a tabulated table's range."""
        h = lam_um * 1e-3
        if self.model == "tabulated":
            lam_tab = self.nk_lambda_um
            spacing = np.interp(lam_um, lam_tab[:-1], np.diff(lam_tab))
            h = np.maximum(h, 2.0 * spacing)
            lam_um = np.clip(lam_um, lam_tab[0] + h, lam_tab[-1] - h)
        return lam_um, h

    def dn_dlam(self, lam_m):
        """First derivative d Re(n)/d(lambda) in 1/m (SI). Vectorized;
        scalar in -> scalar out, like n_complex."""
        lam_m = np.asarray(lam_m, dtype=np.float64)
        scalar_in = (lam_m.ndim == 0)
        lam_um = np.atleast_1d(lam_m).astype(np.float64) * 1e6
        out = self._dn_dlam_um(lam_um) * 1e6
        return float(out[0]) if scalar_in else out

    def n_group(self, lam_m):
        """Group index n_g = Re(n) - lambda * dn/dlambda (real; Im(n) does
        not participate). Governs envelope/energy transport speed c/n_g."""
        lam_m = np.asarray(lam_m, dtype=np.float64)
        scalar_in = (lam_m.ndim == 0)
        n = np.real(np.atleast_1d(self.n_complex(lam_m)))
        out = n - np.atleast_1d(lam_m) * np.atleast_1d(self.dn_dlam(lam_m))
        return float(out[0]) if scalar_in else out

    def d2n_dlam2(self, lam_m):
        """Second derivative d2 Re(n)/d(lambda)2 in 1/m^2, via central
        difference of the (analytic where available) first derivative.
        Tabulated models give a knot-scale approximation only."""
        lam_m = np.asarray(lam_m, dtype=np.float64)
        scalar_in = (lam_m.ndim == 0)
        lam_um = np.atleast_1d(lam_m).astype(np.float64) * 1e6
        lam_um, h = self._stencil_um(lam_um)
        d2_um = (self._dn_dlam_um(lam_um + h)
                 - self._dn_dlam_um(lam_um - h)) / (2.0 * h)
        out = d2_um * 1e12
        return float(out[0]) if scalar_in else out

    def d3n_dlam3(self, lam_m):
        """Third derivative d3 Re(n)/d(lambda)3 in 1/m^3 (second central
        difference of the first derivative)."""
        lam_m = np.asarray(lam_m, dtype=np.float64)
        scalar_in = (lam_m.ndim == 0)
        lam_um = np.atleast_1d(lam_m).astype(np.float64) * 1e6
        lam_um, h = self._stencil_um(lam_um)
        d3_um = (self._dn_dlam_um(lam_um + h)
                 - 2.0 * self._dn_dlam_um(lam_um)
                 + self._dn_dlam_um(lam_um - h)) / h ** 2
        out = d3_um * 1e18
        return float(out[0]) if scalar_in else out


C_LIGHT_M_S = 299_792_458.0


def gdd_per_length(mat, lam_m):
    """Material group-delay dispersion per unit length, s^2/m:
    phi2/L = lambda^3/(2 pi c^2) * d2n/dlambda2. Positive = normal
    dispersion (red leads blue)."""
    lam_m = np.asarray(lam_m, dtype=np.float64)
    return lam_m ** 3 / (2.0 * np.pi * C_LIGHT_M_S ** 2) * mat.d2n_dlam2(lam_m)


def tod_per_length(mat, lam_m):
    """Material third-order dispersion per unit length, s^3/m:
    phi3/L = -lambda^4/(4 pi^2 c^3) * (3 d2n/dlambda2 + lambda d3n/dlambda3)."""
    lam_m = np.asarray(lam_m, dtype=np.float64)
    return (-lam_m ** 4 / (4.0 * np.pi ** 2 * C_LIGHT_M_S ** 3)
            * (3.0 * mat.d2n_dlam2(lam_m) + lam_m * mat.d3n_dlam3(lam_m)))


# ---------------------------------------------------------------------------
# MaterialDB
# ---------------------------------------------------------------------------
class MaterialDB:
    def __init__(self, materials):
        # materials: dict[str canonical-name -> Material]; keep insertion order
        self._by_lower = {name.lower(): mat for name, mat in materials.items()}
        self._order = list(materials.keys())
        # uniaxial birefringence registry, populated by
        # optprops.load_uniaxial via attach_uniaxial():
        #   {crystal_name_lower: {"o": Material, "e": Material, ...}}
        self._uniaxial = {}
        # biaxial registry (optprops.load_biaxial via attach_biaxial):
        #   {crystal_name_lower: {"x","y","z": Material, ...}}
        self._biaxial = {}

    # -- uniaxial birefringence (opticalproperties/birefringence) ---------
    def attach_uniaxial(self, mapping):
        """mapping: {crystal_name: {"o": Material, "e": Material, ...}}."""
        self._uniaxial = {k.strip().lower(): v for k, v in mapping.items()}

    def is_birefringent(self, name):
        return (isinstance(name, str)
                and name.strip().lower() in self._uniaxial)

    def get_uniaxial(self, name):
        """(mat_o, mat_e) for a birefringent crystal name. KeyError if the
        name is not in the uniaxial registry (check is_birefringent)."""
        entry = self._uniaxial[name.strip().lower()]
        return entry["o"], entry["e"]

    # -- biaxial birefringence ---------------------------------------------
    def attach_biaxial(self, mapping):
        """mapping: {crystal_name: {"x","y","z": Material, ...}}."""
        self._biaxial = {k.strip().lower(): v for k, v in mapping.items()}

    def is_biaxial(self, name):
        return (isinstance(name, str)
                and name.strip().lower() in self._biaxial)

    def get_biaxial(self, name):
        """(mat_x, mat_y, mat_z) principal-index Materials for a biaxial
        crystal name. KeyError if not registered (check is_biaxial)."""
        entry = self._biaxial[name.strip().lower()]
        return entry["x"], entry["y"], entry["z"]

    @classmethod
    def load(cls, csv_path=None, nk_dir=None):
        csv_path = Path(csv_path) if csv_path is not None else DEFAULT_MATERIALS_CSV
        csv_path = resolve_prop_path(csv_path)
        nk_dir = Path(nk_dir) if nk_dir is not None else DEFAULT_NK_DIR
        if not csv_path.exists():
            raise MaterialError("materials csv not found: %s" % csv_path)

        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            required_cols = {"name", "class", "model", "p1", "p2", "p3",
                              "p4", "p5", "p6", "nk_file", "density_kg_m3",
                              "transmission_um_min", "transmission_um_max",
                              "notes", "reference"}
            missing_cols = required_cols - set(reader.fieldnames or [])
            if missing_cols:
                raise MaterialError(
                    "materials csv %s: missing required column(s) %s"
                    % (csv_path, sorted(missing_cols)))
            rows = list(reader)

        materials = {}
        for i, row in enumerate(rows):
            lineno = i + 2  # header is line 1
            name = (row.get("name") or "").strip()
            ctx = "materials.csv line %d (%r)" % (lineno, name or "<blank name>")
            if not name:
                raise MaterialError("%s: missing material name" % ctx)
            if any(existing.lower() == name.lower() for existing in materials):
                raise MaterialError("%s: duplicate material name" % ctx)

            cls_ = (row.get("class") or "").strip()
            if cls_ not in VALID_CLASSES:
                raise MaterialError(
                    "%s: unknown class %r (must be one of %s)"
                    % (ctx, cls_, sorted(VALID_CLASSES)))

            model = (row.get("model") or "").strip()
            if model not in VALID_MODELS:
                raise MaterialError(
                    "%s: unknown model %r (must be one of %s)"
                    % (ctx, model, sorted(VALID_MODELS)))

            params = _parse_params(row, model, ctx)

            nk_file = (row.get("nk_file") or "").strip()
            if model == "tabulated" and not nk_file:
                raise MaterialError(
                    "%s: model=tabulated requires a non-empty nk_file" % ctx)

            density = _parse_float(row.get("density_kg_m3"), ctx,
                                    "density_kg_m3", default=0.0)
            if density <= 0 and name.lower() not in ZERO_DENSITY_OK:
                raise MaterialError(
                    "%s: density_kg_m3 must be > 0 (got %r)" % (ctx, density))
            if density < 0:
                raise MaterialError(
                    "%s: density_kg_m3 must be >= 0 (got %r)" % (ctx, density))

            trans_min = _parse_optional_float(row.get("transmission_um_min"), ctx,
                                               "transmission_um_min")
            trans_max = _parse_optional_float(row.get("transmission_um_max"), ctx,
                                               "transmission_um_max")
            if trans_min is not None and trans_max is not None \
                    and trans_min >= trans_max:
                raise MaterialError(
                    "%s: transmission_um_min (%r) must be < transmission_um_max (%r)"
                    % (ctx, trans_min, trans_max))

            reference = (row.get("reference") or "").strip()
            if not reference:
                raise MaterialError("%s: reference is required" % ctx)
            notes = (row.get("notes") or "").strip()

            thermo, t_ref_c = _parse_thermo(row, ctx)

            mat = Material(name, cls_, model, params, density,
                            nk_file=nk_file or None, trans_min=trans_min,
                            trans_max=trans_max, notes=notes,
                            reference=reference, thermo=thermo,
                            t_ref_c=t_ref_c)

            if nk_file:
                _attach_nk_table(mat, nk_dir, ctx)

            materials[name] = mat

        return cls(materials)

    def get(self, name):
        try:
            return self._by_lower[name.strip().lower()]
        except (KeyError, AttributeError):
            available = ", ".join(sorted(self._order))
            raise KeyError(
                "unknown material %r. Available materials: %s" % (name, available))

    def used_names(self):
        """Names of all materials currently loaded in the DB (insertion order)."""
        return list(self._order)

    def __contains__(self, name):
        return isinstance(name, str) and name.strip().lower() in self._by_lower

    def __len__(self):
        return len(self._order)

    def __iter__(self):
        return iter(self._order)


def _parse_float(raw, ctx, field, default=None):
    s = (raw or "").strip()
    if s == "":
        if default is None:
            raise MaterialError("%s: missing required field %r" % (ctx, field))
        return default
    try:
        return float(s)
    except ValueError:
        raise MaterialError("%s: field %r=%r is not a number" % (ctx, field, s))


def _parse_optional_float(raw, ctx, field):
    s = (raw or "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        raise MaterialError("%s: field %r=%r is not a number" % (ctx, field, s))


def _parse_params(row, model, ctx):
    p = [float("nan")] * 6

    def gf(key, default=None):
        s = (row.get(key) or "").strip()
        if s == "":
            if default is None:
                raise MaterialError(
                    "%s: model=%r requires parameter %r" % (ctx, model, key))
            return default
        try:
            return float(s)
        except ValueError:
            raise MaterialError("%s: parameter %r=%r is not a number" % (ctx, key, s))

    if model == "sellmeier":
        for i, key in enumerate(_PARAM_KEYS):
            p[i] = gf(key)
        # C_i (p4..p6) are squared resonance wavelengths; positive for the
        # usual Sellmeier-1 fit, but a genuine catalog fit may carry a small
        # negative or zero C (no real pole -> mathematically well-behaved).
        # Reject only non-finite values; an in-band positive-C pole is a real
        # physical resonance handled/warned by the transmission-window check.
        for idx, cname in ((3, "C1"), (4, "C2"), (5, "C3")):
            if not np.isfinite(p[idx]):
                raise MaterialError(
                    "%s: sellmeier %s (p%d) must be finite (got %r)"
                    % (ctx, cname, idx + 1, p[idx]))
    elif model == "schott":
        # Legacy power-series a0..a5 -> p1..p6; any sign, all required.
        for i, key in enumerate(_PARAM_KEYS):
            p[i] = gf(key)
    elif model == "cauchy":
        p[0] = gf("p1")
        p[1] = gf("p2")
        p[2] = gf("p3")
        # p4..p6 unused for cauchy
        p[3] = gf("p4", 0.0)
        p[4] = gf("p5", 0.0)
        p[5] = gf("p6", 0.0)
    elif model == "constant":
        p[0] = gf("p1")
        p[1] = gf("p2", 0.0)
        p[2] = gf("p3", 0.0)
        p[3] = gf("p4", 0.0)
        p[4] = gf("p5", 0.0)
        p[5] = gf("p6", 0.0)
    elif model == "tabulated":
        pass  # params unused
    return tuple(p)


def _parse_thermo(row, ctx):
    """Parse the optional Schott TIE-19 thermo-optic columns from a .miemat
    row. Returns (thermo_tuple_or_None, t_ref_c). thermo is
    (D0, D1, D2, E0, E1, lam_tk_um) if ANY of D0..E1/lam_tk is present (all
    six then required so a partial model can't silently mis-evaluate); None
    otherwise. t_ref_c defaults to DEFAULT_T_REF_C (20 C) when absent."""
    present = {k: (row.get(k) or "").strip() for k in _THERMO_KEYS}
    if not any(present.values()):
        # no thermo-optic model on this row; t_ref only meaningful with one
        return None, DEFAULT_T_REF_C
    missing = [k for k, v in present.items() if v == ""]
    if missing:
        raise MaterialError(
            "%s: partial thermo-optic model -- missing column(s) %s (all of "
            "%s are required once any is set)"
            % (ctx, sorted(missing), list(_THERMO_KEYS)))
    vals = []
    for k in _THERMO_KEYS:
        try:
            vals.append(float(present[k]))
        except ValueError:
            raise MaterialError(
                "%s: thermo-optic %r=%r is not a number" % (ctx, k, present[k]))
    t_ref_c = _parse_optional_float(row.get("thermo_t_ref_c"), ctx,
                                     "thermo_t_ref_c")
    if t_ref_c is None:
        t_ref_c = DEFAULT_T_REF_C
    return tuple(vals), t_ref_c


def _attach_nk_table(mat, nk_dir, ctx):
    path = Path(nk_dir) / mat.nk_file
    resolved = resolve_prop_path(path, alt_ext=".mienk")
    if not resolved.exists():
        alt = _swap_suffix(path, ".mienk")
        raise MaterialError(
            "%s: nk_file %r not found at %s or %s"
            % (ctx, mat.nk_file, path, alt))
    path = resolved
    lam, n, k = [], [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        needed = {"wavelength_nm", "n", "k"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise MaterialError(
                "%s: nk_file %s missing column(s) %s" % (ctx, path, sorted(missing)))
        for j, r in enumerate(reader):
            try:
                lam.append(float(r["wavelength_nm"]) * 1e-3)  # nm -> um
                n.append(float(r["n"]))
                k.append(float(r["k"]))
            except (TypeError, ValueError):
                raise MaterialError(
                    "%s: nk_file %s row %d is not numeric: %r"
                    % (ctx, path, j + 2, r))
    if len(lam) < 2:
        raise MaterialError("%s: nk_file %s has fewer than 2 data rows" % (ctx, path))
    lam_arr = np.asarray(lam, dtype=np.float64)
    if np.any(np.diff(lam_arr) <= 0):
        raise MaterialError(
            "%s: nk_file %s wavelength_nm column is not strictly increasing"
            % (ctx, path))
    mat.nk_lambda_um = lam_arr
    mat.nk_n = np.asarray(n, dtype=np.float64)
    mat.nk_k = np.asarray(k, dtype=np.float64)


# ---------------------------------------------------------------------------
# Coatings
# ---------------------------------------------------------------------------
class CoatingSpec:
    """One coatings.csv row: name + ordered layer stack.

    layers: list of (material_name, thickness_spec) where thickness_spec is
    either a float (metres) or a tuple ('qw', lam0_m) for a quarter-wave
    layer. This is the exact form consumed by
    raytracer.thinfilm.resolve_coating_layers(), which resolves qw specs
    itself at trace time (thickness depends on n(lambda0), so it cannot be
    baked in once and reused if the material's n is dispersive).
    """

    def __init__(self, name, layers, reference=""):
        self.name = name
        self.layers = layers
        self.reference = reference

    def validate_materials(self, db):
        """Raise MaterialError if any layer references a material not in db."""
        for mat_name, _ in self.layers:
            if mat_name not in db:
                raise MaterialError(
                    "coating %r: layer references unknown material %r"
                    % (self.name, mat_name))

    def resolve(self, db):
        """Convenience helper: return [(material_name, thickness_m), ...]
        with qw specs eagerly resolved to a thickness in metres using
        n(lambda0) from `db`. (load_coatings() does NOT call this -- it
        returns the unresolved form so callers with their own lambda can
        resolve dispersively.)"""
        self.validate_materials(db)
        out = []
        for mat_name, spec in self.layers:
            if isinstance(spec, tuple) and spec[0] == "qw":
                lam0_m = spec[1]
                mat = db.get(mat_name)
                n0 = mat.n_complex(lam0_m).real
                thickness_m = lam0_m / (4.0 * n0)
            else:
                thickness_m = float(spec)
            out.append((mat_name, float(thickness_m)))
        return out


def _parse_thickness_spec(spec_str, ctx):
    s = spec_str.strip()
    if s.lower().startswith("qw@"):
        try:
            lam0_nm = float(s[3:])
        except ValueError:
            raise MaterialError("%s: bad quarter-wave spec %r" % (ctx, s))
        if lam0_nm <= 0:
            raise MaterialError("%s: quarter-wave lambda0 must be > 0 (got %r)" % (ctx, s))
        return ("qw", lam0_nm * 1e-9)
    try:
        thickness_nm = float(s)
    except ValueError:
        raise MaterialError("%s: bad thickness spec %r" % (ctx, s))
    if thickness_nm <= 0:
        raise MaterialError("%s: thickness must be > 0 (got %r)" % (ctx, s))
    return thickness_nm * 1e-9


def _parse_layers_field(layers_raw, ctx):
    layers = []
    for layer_str in layers_raw.split(";"):
        layer_str = layer_str.strip()
        if not layer_str:
            continue
        if ":" not in layer_str:
            raise MaterialError(
                "%s: bad layer %r (expected material:thickness_spec)"
                % (ctx, layer_str))
        mat_name, spec_str = layer_str.split(":", 1)
        mat_name = mat_name.strip()
        if not mat_name:
            raise MaterialError("%s: layer %r missing material name" % (ctx, layer_str))
        thickness_spec = _parse_thickness_spec(spec_str, ctx)
        layers.append((mat_name, thickness_spec))
    return layers


def _load_coating_table(path, ctx):
    """coating/tables/<name>.csv: wavelength_nm,Rs,Rp,Ts,Tp (fractions).
    Hard-validated: strictly increasing lambda, all in [0,1], Rs+Ts<=1 and
    Rp+Tp<=1 per row (remainder = absorption)."""
    path = Path(path)
    if not path.exists():
        raise MaterialError("%s: coating table not found: %s" % (ctx, path))
    cols = {"wavelength_nm": [], "Rs": [], "Rp": [], "Ts": [], "Tp": []}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(cols) - set(reader.fieldnames or [])
        if missing:
            raise MaterialError("%s: coating table %s missing column(s) %s"
                                % (ctx, path, sorted(missing)))
        for j, r in enumerate(reader):
            try:
                for key in cols:
                    cols[key].append(float(r[key]))
            except (TypeError, ValueError):
                raise MaterialError("%s: coating table %s row %d not numeric: %r"
                                    % (ctx, path, j + 2, r))
    lam_um = np.asarray(cols["wavelength_nm"], dtype=np.float64) * 1e-3
    if lam_um.size < 2:
        raise MaterialError("%s: coating table %s has fewer than 2 rows"
                            % (ctx, path))
    if np.any(np.diff(lam_um) <= 0):
        raise MaterialError("%s: coating table %s wavelength_nm not strictly "
                            "increasing" % (ctx, path))
    out = {"lam_um": lam_um}
    for key in ("Rs", "Rp", "Ts", "Tp"):
        arr = np.asarray(cols[key], dtype=np.float64)
        if np.any((arr < 0) | (arr > 1)):
            raise MaterialError("%s: coating table %s column %s outside [0,1]"
                                % (ctx, path, key))
        out[key] = arr
    if np.any(out["Rs"] + out["Ts"] > 1.0 + 1e-9) \
            or np.any(out["Rp"] + out["Tp"] > 1.0 + 1e-9):
        raise MaterialError(
            "%s: coating table %s violates R+T<=1 (per polarization) — "
            "energy would not close" % (ctx, path))
    return out


def _parse_coatings_csv(csv_path):
    csv_path = resolve_prop_path(Path(csv_path))
    if not csv_path.exists():
        raise MaterialError("coatings csv not found: %s" % csv_path)
    tables_dir = csv_path.parent / "tables"
    specs = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        required_cols = {"name", "layers", "reference"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise MaterialError(
                "coatings csv %s: missing required column(s) %s"
                % (csv_path, sorted(missing)))
        for i, row in enumerate(reader):
            lineno = i + 2
            name = (row.get("name") or "").strip()
            ctx = "coatings.csv line %d (%r)" % (lineno, name or "<blank name>")
            if not name:
                raise MaterialError("%s: missing coating name" % ctx)
            reference = (row.get("reference") or "").strip()
            if not reference:
                raise MaterialError("%s: reference is required" % ctx)
            layers_raw = (row.get("layers") or "").strip()
            table_raw = (row.get("table") or "").strip()
            if bool(layers_raw) == bool(table_raw):
                raise MaterialError(
                    "%s: exactly one of 'layers' (TMM stack) or 'table' "
                    "(measured Rs/Rp/Ts/Tp csv) must be set" % ctx)
            if layers_raw:
                layers = _parse_layers_field(layers_raw, ctx)
                specs.append(("tmm", name, layers, None, None, reference))
            else:
                aoi = (row.get("aoi_deg") or "").strip()
                aoi_deg = float(aoi) if aoi else 0.0
                table_path = tables_dir / table_raw
                resolved = resolve_prop_path(table_path, alt_ext=".mietab")
                if not resolved.exists():
                    alt = _swap_suffix(table_path, ".mietab")
                    raise MaterialError(
                        "%s: coating table not found: %s or %s"
                        % (ctx, table_path, alt))
                table = _load_coating_table(resolved, ctx)
                specs.append(("table", name, None, table, aoi_deg, reference))
    return specs


def load_coatings(csv_path=None, db=None):
    """Parse coatings.csv into tagged coating specs.

    Returns {coating_name: spec} where spec is one of
      {"kind": "tmm", "layers": [(material_name, thickness_spec), ...],
       "reference": str}
          thickness_spec is a float in metres, or ('qw', lam0_m) for a
          quarter-wave layer -- deliberately left UNRESOLVED so dispersive
          n(lambda0) applies at trace time (thinfilm.resolve_coating_layers
          consumes exactly this shape);
      {"kind": "table", "lam_um": arr, "Rs": arr, "Rp": arr, "Ts": arr,
       "Tp": arr, "aoi_deg": float, "reference": str}
          measured/tabulated coating (hot/cold mirrors, PBS interfaces,
          dichroics) -- amplitudes carry the bare-interface Fresnel phase
          at trace time (documented approximation, README §6).
    Every TMM layer's material is hard-validated against `db`.
    """
    csv_path = Path(csv_path) if csv_path is not None else DEFAULT_COATINGS_CSV
    if db is None:
        raise MaterialError("load_coatings requires a MaterialDB (db=...)")
    out = {}
    for kind, name, layers, table, aoi_deg, reference in \
            _parse_coatings_csv(csv_path):
        if name in out:
            raise MaterialError("coatings.csv: duplicate coating name %r" % name)
        if kind == "tmm":
            CoatingSpec(name, layers, reference).validate_materials(db)
            out[name] = {"kind": "tmm", "layers": list(layers),
                         "reference": reference}
        else:
            spec = {"kind": "table", "aoi_deg": aoi_deg,
                    "reference": reference}
            spec.update(table)
            out[name] = spec
    return out


# ---------------------------------------------------------------------------
# Diagnostic plot (not required for the loader to function; matplotlib is
# imported lazily so it stays an optional dependency).
# ---------------------------------------------------------------------------
def plot_dispersion(db, names, out_png, lam_range_nm=(300.0, 1100.0), n_points=300):
    """Plot n(lambda) and k(lambda) for the given material names over
    lam_range_nm, clipped to each material's own valid range (tabulated
    nk_file extent, or transmission_um_min/max, or the full requested
    range if unconstrained). Saves a two-panel PNG to out_png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo_req, hi_req = lam_range_nm
    fig, (ax_n, ax_k) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    for name in names:
        mat = db.get(name)
        if mat.model == "tabulated":
            lo = max(lo_req, mat.nk_lambda_um[0] * 1000.0)
            hi = min(hi_req, mat.nk_lambda_um[-1] * 1000.0)
        else:
            lo = mat.trans_min * 1000.0 if mat.trans_min is not None else lo_req
            hi = mat.trans_max * 1000.0 if mat.trans_max is not None else hi_req
            lo = max(lo_req, lo)
            hi = min(hi_req, hi)
        if hi <= lo:
            continue
        lam_nm = np.linspace(lo, hi, n_points)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nk = mat.n_complex(lam_nm * 1e-9)
        ax_n.plot(lam_nm, nk.real, label=name)
        k = nk.imag.copy()
        k[k <= 0] = np.nan
        ax_k.plot(lam_nm, k, label=name)

    ax_n.set_ylabel("n")
    ax_n.set_title("Refractive index n(lambda)")
    ax_n.legend(fontsize=7, loc="best")
    ax_k.set_ylabel("k")
    ax_k.set_xlabel("wavelength (nm)")
    ax_k.set_yscale("log")
    ax_k.set_title("Extinction coefficient k(lambda)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return str(out_png)


# ---------------------------------------------------------------------------
# Self-check: python3 scripts/raytracer/materials.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    db = MaterialDB.load()
    print("loaded %d materials: %s" % (len(db), ", ".join(db.used_names())))
    for nm in ("bk7", "fused_silica", "water", "aluminum"):
        mat = db.get(nm)
        n = mat.n_complex(np.array(589e-9))
        print("  %-14s n(589nm) = %s" % (nm, n))
    coatings = load_coatings(db=db)
    for name, spec in coatings.items():
        detail = spec["layers"] if spec["kind"] == "tmm" else \
            "table %d pts, aoi=%g deg" % (len(spec["lam_um"]), spec["aoi_deg"])
        print("coating %-16s [%s] %s" % (name, spec["kind"], detail))
