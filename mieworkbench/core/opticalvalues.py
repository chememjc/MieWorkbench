"""Context-sensitive "insert optical value" menu model — pure, no Qt.

Given the train, a target element and the field being edited, returns
the list of optical values a designer would otherwise compute on the
side: the previous element's focal distances, afocal spacings, paraxial
image distances, system values, grating diffraction angles, prism
minimum deviation. The GUI panes (train editor cells, element editor
dim/property fields) render this as a right-click submenu; entries can
be inserted as a literal value or captured into a miewb_vars variable
(`suggest_var`) so the design stays re-tunable.

Same pattern as facemaps.menu_model: plain data out, oracle-tested,
thin QMenu builders in the panes.

Entry shape:
  {"group": str,        # submenu grouping ("Previous element", ...)
   "label": str,        # human text incl. the value
   "value": float,      # mm or deg
   "kind": "length"|"angle",
   "suggest_var": str|None,   # proposed miewb_vars name
   "expr": str|None}          # live expression, only when genuinely
                              # expressible over existing variables

Honest limits: everything paraxial (see core/paraxial.py's header);
grating entries parse the inline `N:v|h:orders=a..b` grating syntax
(registry `@row` gratings are resolved when an optprops object is
supplied, else skipped); prism minimum deviation assumes the equilateral
(A=60 deg) `prism` primitive.
"""

import math
import re

from . import paraxial, wizards
import train_solver  # noqa: E402  (scripts/ on sys.path via paraxial)

# "span N Airy zeros" detector-sizing entries: N * 1.22 * lambda *
# image_distance / aperture_diameter, offered for these N (a coarse
# linear-in-N scaling of the first Airy null radius, not the exact
# non-uniformly-spaced zero positions of the Airy pattern).
AIRY_ZERO_COUNTS = (2, 4, 8)

# which optical-value kind fits each train edge field
FIELD_KIND = {
    "distance": "length",
    "decenter_x": "length",
    "decenter_y": "length",
    "tilt_rx": "angle",
    "tilt_ry": "angle",
    "tilt_rz": "angle",
    "fold_deviation": "angle",
    "fold_azimuth": "angle",
}

# dim-sheet aliases that are lengths (radius aliases are case-sensitive
# R/R_*/R2-style so e.g. `rotation` never matches; the rest are plain
# lowercase names)
_RADIUS_ALIAS_RE = re.compile(r"^R(_\w+|\d*)?$")
_LENGTH_ALIAS_RE = re.compile(
    r"^(ct\w*|aperture|diameter|width|height|length|thickness|"
    r"f_design|rfl|gap|spacing\w*|core_diameter|clad_diameter|"
    r"hole_diameter|side|back)$")


def field_kind(field, is_sheet_alias=False):
    """'length' | 'angle' | None for a train field or dim-sheet alias."""
    if not is_sheet_alias:
        return FIELD_KIND.get(field)
    f = field or ""
    if _RADIUS_ALIAS_RE.match(f) or _LENGTH_ALIAS_RE.match(f):
        return "length"
    return None  # rotation/base_angle/conic k/...: no generic value fits


def _var_name(prefix, element):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", element).strip("_").lower()
    return "%s_%s" % (prefix, slug)


def _fmt(v):
    return "%.6g" % v


_GRATING_INLINE_RE = re.compile(
    r"(?:^|=)\s*(\d+(?:\.\d+)?)\s*:\s*[vh]\s*:\s*orders\s*=\s*"
    r"(-?\d+)\s*\.\.\s*(-?\d+)", re.IGNORECASE)


def _grating_lines_per_mm(primary, optprops=None):
    """(lines_per_mm, orders) from a body's grating property, or None."""
    from .train import _prop_value
    raw = _prop_value(primary, "grating")
    if not raw:
        return None
    m = _GRATING_INLINE_RE.search(str(raw))
    if m:
        return float(m.group(1)), range(int(m.group(2)),
                                        int(m.group(3)) + 1)
    if optprops is not None and "@" in str(raw):
        row = str(raw).split("@", 1)[1].split(";")[0].strip()
        try:
            g = optprops.gratings[row]
            lpm = float(g.get("lines_per_mm"))
            return lpm, range(-1, 2)
        except (KeyError, AttributeError, TypeError, ValueError):
            return None
    return None


def value_menu_model(train_model, element, field, variables=None,
                     index_fn=None, lam_nm=None, is_sheet_alias=False,
                     optprops=None):
    """List of insertable optical values for (element, field). Empty when
    the field has no meaningful optical values. Never raises on solver
    trouble — degrades to fewer entries."""
    kind = field_kind(field, is_sheet_alias=is_sheet_alias)
    if kind is None:
        return []
    variables = variables or {}
    if index_fn is None:
        matdb = wizards._default_matdb()

        def index_fn(mat, lam):
            return wizards.index_at(matdb, mat, lam)

    if lam_nm is None:
        try:
            lam_nm = paraxial.design_wavelength_nm(train_model)
        except Exception:
            lam_nm = paraxial.DESIGN_LAMBDA_FALLBACK_NM

    entries = []
    records = {}
    try:
        records = train_model.records()
    except train_solver.TrainError:
        pass
    rec = records.get(element) or {}
    ref = rec.get("ref") if rec.get("mode") == "chained" else None

    def _card(el):
        try:
            primary = train_model.primary_body(el)
            from .train import _prop_value
            k = _prop_value(primary, "miewb_primitive")
            params = train_model._sheet_params(el) or {}
            if not k:
                return None
            c = paraxial.element_cardinals(k, params, index_fn, lam_nm,
                                           material=_prop_value(primary,
                                                                "material"))
            return None if c.get("passthrough") or c.get("afocal") else c
        except Exception:
            return None

    # ---- previous-element focal values (length fields) -------------------
    if kind == "length" and ref:
        c_ref = _card(ref)
        if c_ref:
            # UXNOTES_ROUND3 #28: for a NEGATIVE rear group (diverging
            # element, BFL < 0) "distance to its focus" reads like a
            # forward placement distance, but the value is actually
            # negative — the focus is a VIRTUAL one upstream of the exit
            # vertex, and chaining the next element "at" that distance
            # doesn't track the real image (use "Paraxial image distance
            # after X" / "System BFL" for that). Make the sign explicit
            # instead of implying a placement distance.
            if c_ref["bfl"] < 0:
                bfl_label = ("BFL of %s (%s mm — NEGATIVE: a virtual "
                            "focus upstream of %s's exit vertex, not a "
                            "forward placement distance)"
                            % (ref, _fmt(c_ref["bfl"]), ref))
            else:
                bfl_label = ("BFL of %s — distance to its focus (%s mm)"
                            % (ref, _fmt(c_ref["bfl"])))
            entries.append({
                "group": "Previous element (%s)" % ref,
                "label": bfl_label,
                "value": c_ref["bfl"], "kind": "length",
                "suggest_var": _var_name("bfl", ref), "expr": None})
            entries.append({
                "group": "Previous element (%s)" % ref,
                "label": "EFL of %s (%s mm)" % (ref, _fmt(c_ref["efl"])),
                "value": c_ref["efl"], "kind": "length",
                "suggest_var": _var_name("efl", ref), "expr": None})
            c_self = _card(element)
            if c_self:
                afocal = c_ref["bfl"] - c_self["ffl"]
                entries.append({
                    "group": "Previous element (%s)" % ref,
                    "label": "Afocal spacing %s -> %s (paraxial, %s mm)"
                             % (ref, element, _fmt(afocal)),
                    "value": afocal, "kind": "length",
                    "suggest_var": _var_name("afocal", ref),
                    "expr": None})
        # paraxial image distance of the upstream subsystem through ref
        try:
            sub = paraxial.system_summary(
                train_model, variables, index_fn, lam_nm,
                through_element=ref)
            img = sub.get("image_distance_mm")
            if img is not None and math.isfinite(img) \
                    and sub["n_optical_elements"] > 0:
                entries.append({
                    "group": "Previous element (%s)" % ref,
                    "label": "Paraxial image distance after %s "
                             "(collimated input, %s mm)" % (ref, _fmt(img)),
                    "value": img, "kind": "length",
                    "suggest_var": _var_name("img", ref), "expr": None})
        except Exception:
            pass

    # ---- system values ----------------------------------------------------
    try:
        s = paraxial.system_summary(train_model, variables, index_fn,
                                    lam_nm)
    except Exception:
        s = None
    if s and s.get("n_optical_elements") and not s.get("afocal"):
        if kind == "length":
            for key, label, prefix in (
                    ("efl", "System EFL", "sys_efl"),
                    ("bfl", "System BFL", "sys_bfl"),
                    ("image_distance_mm", "System image distance",
                     "sys_img")):
                v = s.get(key)
                if v is not None and math.isfinite(v):
                    entries.append({
                        "group": "System",
                        "label": "%s (%s mm)" % (label, _fmt(v)),
                        "value": v, "kind": "length",
                        "suggest_var": prefix, "expr": None})

            # ---- diffraction-scale detector sizing: "span N Airy zeros"
            # (future.md (a2) / UXNOTES_ROUND3 #17). Needs the same
            # limiting-aperture stop search system_summary already ran,
            # plus a finite image distance — both already computed above,
            # so this only ever appears when they resolve.
            img_mm = s.get("image_distance_mm")
            lim = s.get("limiting_element")
            ap_mm = (paraxial.element_aperture_mm(train_model, lim)
                    if lim else None)
            if ap_mm and ap_mm > 0 and img_mm is not None \
                    and math.isfinite(img_mm):
                lam_mm = lam_nm * 1e-6
                for n in AIRY_ZERO_COUNTS:
                    r = n * 1.22 * lam_mm * img_mm / ap_mm
                    entries.append({
                        "group": "System",
                        "label": "Span %d Airy zeros @ %.0f nm, stop %s "
                                 "(%s mm radius)"
                                 % (n, lam_nm, lim, _fmt(r)),
                        "value": r, "kind": "length",
                        "suggest_var": "airy_%d" % n, "expr": None})

    # ---- grating / prism angles (angle fields) ----------------------------
    if kind == "angle" and ref:
        try:
            primary_ref = train_model.primary_body(ref)
        except train_solver.TrainError:
            primary_ref = None
        if primary_ref is not None:
            g = _grating_lines_per_mm(primary_ref, optprops=optprops)
            if g:
                lpm, orders = g
                d_um = 1000.0 / lpm            # groove spacing, um
                lam_um = lam_nm / 1000.0
                for m in orders:
                    if m == 0:
                        continue
                    sin_t = m * lam_um / d_um
                    if abs(sin_t) >= 1.0:
                        continue
                    ang = math.degrees(math.asin(sin_t))
                    entries.append({
                        "group": "Grating (%s)" % ref,
                        "label": "Order %+d diffraction angle @ %.0f nm "
                                 "(%s deg)" % (m, lam_nm, _fmt(ang)),
                        "value": ang, "kind": "angle",
                        "suggest_var": _var_name(
                            "grat_m%s" % str(m).replace("-", "n"), ref),
                        "expr": None})
            from .train import _prop_value
            if _prop_value(primary_ref, "miewb_primitive") == "prism":
                try:
                    n = float(index_fn(
                        _prop_value(primary_ref, "material") or "bk7",
                        lam_nm))
                    A = 60.0
                    dmin = 2.0 * math.degrees(
                        math.asin(n * math.sin(math.radians(A / 2.0)))) - A
                    entries.append({
                        "group": "Prism (%s)" % ref,
                        "label": "Minimum deviation @ %.0f nm (%s deg)"
                                 % (lam_nm, _fmt(dmin)),
                        "value": dmin, "kind": "angle",
                        "suggest_var": _var_name("dev", ref),
                        "expr": None})
                except (ValueError, TypeError):
                    pass

    return entries
