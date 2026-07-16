"""Pre-run validation framework: catch missing info and likely errors
BEFORE launching the simulation.

Qt-free and worker-free: checks run on the Project's structure dict + a
loaded optical-property library + the intended run configuration, reusing
the pipeline's own parsers/loaders as oracles (common.parse_*,
raytracer.optprops/materials, common.estimate) instead of reimplementing
their rules. The problems pane renders Findings; errors gate the Run
button, warnings are overridable.

Deep geometric checks (recompute errors, open solids via OCC, pairwise
overlaps) live in the FreeCAD worker's 'check' op; run_deep_checks() wraps
it when a live Project is available.
"""

import os
import sys

_SCRIPTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import common  # noqa: E402

ERROR = "error"
WARNING = "warning"
INFO = "info"


class Finding:
    def __init__(self, severity, message, body=None, face=None,
                 fix_hint=None, check=None):
        self.severity = severity
        self.message = message
        self.body = body
        self.face = face
        self.fix_hint = fix_hint
        self.check = check

    def __repr__(self):
        return "<%s %s: %s>" % (self.severity.upper(), self.check,
                                self.message)

    def as_dict(self):
        return {"severity": self.severity, "message": self.message,
                "body": self.body, "face": self.face,
                "fix_hint": self.fix_hint, "check": self.check}


CHECKS = []


def check(name):
    def wrap(fn):
        fn.check_name = name
        CHECKS.append(fn)
        return fn
    return wrap


class Validator:
    """validate(structure, optprops, config) -> [Finding].

    structure: the FreeCAD worker's get_structure dict.
    optprops:  raytracer.optprops.OpticalProperties for the PROJECT library
               (or None -> library checks are skipped with a warning).
    config:    dict of run options (rays, nlambda, resolution, backend,
               min_eff_samples, seeds ...); missing keys use pipeline
               defaults.
    """

    def __init__(self, structure, optprops=None, config=None):
        self.structure = structure or {"bodies": []}
        self.optprops = optprops
        self.config = dict(config or {})

    def validate(self):
        findings = []
        for fn in CHECKS:
            try:
                findings.extend(fn(self) or [])
            except Exception as exc:   # a broken check must not hide others
                findings.append(Finding(
                    WARNING, "check %r itself failed: %s"
                    % (fn.check_name, exc), check=fn.check_name))
        order = {ERROR: 0, WARNING: 1, INFO: 2}
        findings.sort(key=lambda f: order.get(f.severity, 3))
        return findings

    # -- helpers -------------------------------------------------------------
    def bodies(self):
        return self.structure.get("bodies", [])

    def prop(self, body, name):
        rec = body.get("properties", {}).get(name)
        return None if rec is None else rec.get("value")

    def role(self, body):
        # pulsed-optics Phase P3: pulse_energy is a valid alternative to
        # power (extract_geometry.classify_body's OR-gate mirrored here);
        # check_pulse_params below enforces the power/pulse_energy XOR.
        if self.prop(body, "lambdac") is not None \
                and (self.prop(body, "power") is not None
                     or self.prop(body, "pulse_energy") is not None):
            return "source"
        mat = self.prop(body, "material")
        if mat is None or str(mat).strip().lower() in ("", "none"):
            return "ignored"
        if str(mat).strip().lower() == "detector":
            return "detector"
        return "optic"

    def source_range_nm(self):
        """(lo, hi) union over sources, mirroring run_trace.lam_range_nm's
        3-sigma + pad policy."""
        lo, hi = 1e9, 0.0
        for b in self.bodies():
            if self.role(b) != "source":
                continue
            lc = float(self.prop(b, "lambdac"))
            lmin = float(self.prop(b, "lambdamin") or lc)
            lmax = float(self.prop(b, "lambdamax") or lc)
            lo = min(lo, lc - 3.0 * max(lc - lmin, 0.0))
            hi = max(hi, lc + 3.0 * max(lmax - lc, 0.0))
        if hi <= 0.0:
            return None
        pad = max(5.0, 0.02 * (hi - lo))
        return lo - pad, hi + pad

    def facemap_values(self, body, prop_name):
        """All values of a facemap-form property ('Face3=X;...' or bare)."""
        raw = self.prop(body, prop_name)
        if raw is None:
            return {}
        return common.parse_facemap_spec(
            str(raw), body=body.get("name"), feature=body.get("tip"))

    def sheet_params(self, body):
        """This body's dim-sheet aliases (sheet labeled 'dim_<label>'),
        offline mirror of core.train.TrainModel._sheet_params: the
        worker's get_structure already returns every sheet flat in
        structure['sheets'], so no live TrainModel/worker is needed."""
        label = "dim_%s" % body.get("label")
        for sheet in self.structure.get("sheets", []):
            if sheet.get("label") == label:
                out = {}
                for alias, cell in (sheet.get("aliases") or {}).items():
                    try:
                        out[alias] = float(cell.get("value"))
                    except (TypeError, ValueError):
                        pass
                return out
        return None

    def aperture_mm(self, body):
        """Clear-aperture diameter (mm) from this body's dim sheet, same
        alias preference order as core.paraxial.element_aperture_mm /
        _APERTURE_ALIASES (hole_diameter wins over outer diameter for
        iris/pinhole/annular stops); None if no sheet or no positive
        alias is found."""
        from . import paraxial
        params = self.sheet_params(body)
        if not params:
            return None
        for key in paraxial._APERTURE_ALIASES:
            if key in params and params[key] and params[key] > 0:
                return float(params[key])
        return None


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
@check("roles")
def check_roles(v):
    out = []
    roles = [v.role(b) for b in v.bodies()]
    if "source" not in roles:
        out.append(Finding(ERROR, "scene has no light source (a source "
                           "body needs 'lambdac' plus either 'power' or "
                           "'pulse_energy')",
                           fix_hint="add a laser/source element or set "
                           "power+lambdac (or pulse_energy+lambdac) on a "
                           "body", check="roles"))
    if "detector" not in roles:
        out.append(Finding(ERROR, "scene has no detector (set "
                           "material='detector' on a screen body)",
                           fix_hint="add a detector element",
                           check="roles"))
    for b in v.bodies():
        if v.role(b) == "ignored":
            out.append(Finding(INFO, "%s has no material and is ignored "
                               "by the simulation" % b["label"],
                               body=b["name"], check="roles"))
    return out


@check("pulse-params")
def check_pulse_params(v):
    """Pulsed-source body-property contract (pulsed-optics Phase P3):
    catches the power/pulse_energy XOR and the pulse_energy-needs-
    rep_rate rule (raytracer.scene._parse_pulse_source enforces the same
    rules engine-side; this surfaces them pre-run instead of as a
    SceneError mid-trace) plus positivity and a stray-property warning."""
    out = []
    for b in v.bodies():
        power = v.prop(b, "power")
        pulse_energy = v.prop(b, "pulse_energy")
        pulse_duration = v.prop(b, "pulse_duration")
        rep_rate = v.prop(b, "rep_rate")
        if power is None and pulse_energy is None \
                and pulse_duration is None and rep_rate is None:
            continue    # no pulse properties at all on this body
        if v.role(b) != "source":
            out.append(Finding(
                WARNING, "%s: pulse_energy/pulse_duration/rep_rate are "
                "only meaningful on source bodies (need 'lambdac' too); "
                "ignoring" % b["label"], body=b["name"],
                check="pulse-params"))
            continue
        if power is not None and pulse_energy is not None:
            out.append(Finding(
                ERROR, "%s: both power (%.6g mW) and pulse_energy "
                "(%.6g uJ) are set — pick one (average power is "
                "ambiguous otherwise)"
                % (b["label"], float(power), float(pulse_energy)),
                body=b["name"], fix_hint="remove one of power/"
                "pulse_energy", check="pulse-params"))
        if pulse_energy is not None and rep_rate is None:
            out.append(Finding(
                ERROR, "%s: pulse_energy needs rep_rate — average power "
                "is underdetermined without it" % b["label"],
                body=b["name"], fix_hint="set rep_rate (Hz)",
                check="pulse-params"))
        for name, val, unit in (("pulse_energy", pulse_energy, "uJ"),
                                ("pulse_duration", pulse_duration, "ps"),
                                ("rep_rate", rep_rate, "Hz")):
            if val is not None and float(val) <= 0:
                out.append(Finding(
                    ERROR, "%s: %s must be > 0 %s (got %g)"
                    % (b["label"], name, unit, float(val)),
                    body=b["name"], check="pulse-params"))
    return out


@check("detector-face")
def check_detector_face(v):
    """A detector body's detector_face pin must name a face that exists on
    the body (bare 'FaceN' or a full 'Body.Tip.FaceN' id)."""
    out = []
    for b in v.bodies():
        raw = v.prop(b, "detector_face")
        if raw is None or str(raw).strip().lower() in ("", "none"):
            continue
        if v.role(b) != "detector":
            out.append(Finding(WARNING, "%s: detector_face is only "
                               "meaningful on detector bodies (material="
                               "'detector'); it will be ignored" % b["label"],
                               body=b["name"], check="detector-face"))
            continue
        bare = str(raw).strip().rsplit(".", 1)[-1]
        try:
            idx = common.parse_face_spec(
                "B.T.%s" % bare if bare.startswith("Face") else str(raw).strip()
            )["face_index"]
        except ValueError:
            out.append(Finding(ERROR, "%s: bad detector_face %r (expected "
                               "'FaceN' or 'Body.Tip.FaceN')"
                               % (b["label"], raw), body=b["name"],
                               check="detector-face"))
            continue
        n = int(b.get("face_count", 0) or 0)
        if n and not 1 <= idx <= n:
            out.append(Finding(ERROR, "%s: detector_face names Face%d but "
                               "the body has %d face%s"
                               % (b["label"], idx, n,
                                  "s" if n != 1 else ""), body=b["name"],
                               fix_hint="pick an existing FaceN or clear "
                               "detector_face to restore the auto-pick",
                               check="detector-face"))
    return out


@check("placement-traps")
def check_placement_traps(v):
    """Two placement conventions that silently zero a run (both found the
    hard way in the demo shakedown, UXNOTES_ROUND3):
      * a SOURCE anchored at/near the world origin — emit directions pick
        the 'toward the origin' hemisphere, which degenerates AT the
        origin (rays spray backwards; detected power reads ~0);
      * a ROTATED detector without a detector_face pin — the closest-to-
        origin face auto-pick tends to land on a thin edge face (a strip
        image / 0 mW)."""
    import math as _math
    out = []
    for b in v.bodies():
        role = v.role(b)
        pos = (b.get("placement") or {}).get("pos_mm") or [0.0, 0.0, 0.0]
        quat = (b.get("placement") or {}).get("quat") or [0, 0, 0, 1]
        if role == "source" \
                and _math.sqrt(sum(c * c for c in pos)) < 1.0:
            out.append(Finding(
                WARNING, "%s: source sits at the world origin — the emit "
                "direction convention ('toward the origin hemisphere') "
                "degenerates there and rays can spray backwards"
                % b["label"], body=b["name"],
                fix_hint="move the source off the origin (any nonzero "
                "position; downstream chained elements follow)",
                check="placement-traps"))
        if role == "detector" and v.prop(b, "detector_face") in (None, ""):
            # rotation that tips the local +x axis off +/-x by > ~1 deg
            qx, qy, qz, qw = [float(c) for c in quat]
            # local +x in world coordinates
            ax = 1.0 - 2.0 * (qy * qy + qz * qz)
            if abs(abs(ax) - 1.0) > 1.5e-4:      # cos(1 deg) ~ 0.99985
                out.append(Finding(
                    WARNING, "%s: rotated detector without a "
                    "detector_face pin — the closest-to-origin face "
                    "auto-pick can land on a thin EDGE face (strip "
                    "image / 0 mW)" % b["label"], body=b["name"],
                    fix_hint="set detector_face to the recording face "
                    "(element editor face combo)",
                    check="placement-traps"))
    return out


@check("geometry")
def check_geometry(v):
    out = []
    for b in v.bodies():
        if v.role(b) == "ignored":
            continue
        if not b.get("solid_closed", True):
            out.append(Finding(ERROR, "%s is not a closed solid — rays "
                               "would leak through the seam" % b["label"],
                               body=b["name"],
                               fix_hint="fix the sketch/pad so the body "
                               "is watertight", check="geometry"))
        pl = b.get("placement", {}).get("pos_mm", [0, 0, 0])
        if any(x != x for x in pl):   # NaN
            out.append(Finding(ERROR, "%s has a NaN placement"
                               % b["label"], body=b["name"],
                               check="geometry"))
    return out


@check("library-names")
def check_library_names(v):
    if v.optprops is None:
        return [Finding(WARNING, "no optical-property library loaded; "
                        "name checks skipped", check="library-names")]
    p = v.optprops
    out = []
    for b in v.bodies():
        role = v.role(b)
        if role == "ignored":
            continue
        mat = str(v.prop(b, "material") or "").strip()
        if role != "source" and mat and mat.lower() != "detector":
            biaxial = getattr(p, "biaxial", {}) or {}
            known = (mat in p.matdb) or (mat in p.uniaxial) \
                or (mat in biaxial)
            if not known:
                out.append(Finding(ERROR, "%s: unknown material %r"
                                   % (b["label"], mat), body=b["name"],
                                   fix_hint="add it to materials.miemat "
                                   "or pick an existing one",
                                   check="library-names"))
            if mat in biaxial and v.prop(b, "crystal_axis2") is None:
                out.append(Finding(
                    ERROR, "%s: material %r is biaxial and needs a "
                    "crystal_axis2 property (the Y principal axis; "
                    "crystal_axis is the X axis, Z is derived)"
                    % (b["label"], mat), body=b["name"],
                    fix_hint="add crystal_axis2='x,y,z' (body-local, "
                    "not parallel to crystal_axis) on this body",
                    check="library-names"))
        for prop_name, registry, what in (
                ("coating", p.coatings, "coating"),
                ("filter", p.filters, "filter"),
                ("polarizer", p.polarizers, "polarizer")):
            raw = v.prop(b, prop_name)
            if raw is None:
                continue
            try:
                values = v.facemap_values(b, prop_name) \
                    if prop_name == "coating" else \
                    {None: str(raw)}
            except ValueError as exc:
                out.append(Finding(ERROR, "%s: bad %s spec: %s"
                                   % (b["label"], what, exc),
                                   body=b["name"], check="library-names"))
                continue
            for face, name in values.items():
                # the tagging contract treats ''/'none' as "no <what>"
                if str(name).strip().lower() in ("", "none"):
                    continue
                if name not in registry:
                    out.append(Finding(
                        ERROR, "%s: unknown %s %r%s"
                        % (b["label"], what, name,
                           "" if face in (None, common.FACEMAP_ALL)
                           else " on %s" % face),
                        body=b["name"],
                        fix_hint="add it to the library or fix the name",
                        check="library-names"))
        graw = v.prop(b, "grating")
        if graw is not None:
            try:
                gmap = v.facemap_values(b, "grating")
                for face, gval in gmap.items():
                    g = common.parse_grating_value(str(gval))
                    reg = g.get("registry")
                    if reg and reg not in p.gratings:
                        out.append(Finding(
                            ERROR, "%s: unknown grating registry entry "
                            "%r" % (b["label"], reg), body=b["name"],
                            check="library-names"))
            except ValueError as exc:
                out.append(Finding(ERROR, "%s: bad grating spec: %s"
                                   % (b["label"], exc), body=b["name"],
                                   check="library-names"))
    return out


@check("coating-phase")
def check_coating_phase(v):
    """P2: a table coating (materials.py 'kind'=='table') carries phase
    only when its csv has all four ars_deg/arp_deg/ats_deg/atp_deg
    columns (phase_valid=True); without them the tracer borrows the
    bare-interface Fresnel phase (documented approximation, harmless for
    an incoherent/power-only scene). On a COHERENT scene that borrowed
    phase can be badly wrong -- a real coating's retardance is NOT
    derivable from its Rs/Rp/Ts/Tp alone (Blaschke factors), e.g. a
    dielectric NPBS at 45 deg carries ~20 deg of retardance a bare
    table can't express (engine3.md Sec 7.3) -- so this WARNS (never
    gates the Run button; the run is still physically valid, just less
    accurate for interference through that coating)."""
    if v.optprops is None:
        return []
    if not any(v.role(b) == "source" and bool(v.prop(b, "coherent"))
               for b in v.bodies()):
        return []
    out = []
    seen = set()
    for b in v.bodies():
        if v.role(b) == "ignored":
            continue
        try:
            values = v.facemap_values(b, "coating")
        except ValueError:
            continue    # check_library_names already reports bad specs
        for face, name in values.items():
            cname = str(name).strip()
            if cname.lower() in ("", "none"):
                continue
            spec = v.optprops.coatings.get(cname)
            if spec is None:
                continue    # check_library_names already reports unknown names
            key = (b["name"], cname)
            if spec.get("kind") == "table" and not spec.get("phase_valid") \
                    and key not in seen:
                seen.add(key)
                out.append(Finding(
                    WARNING,
                    "%s: coating %r has no phase columns -- interference "
                    "through this coating uses bare-interface phase — "
                    "supply Ars/Arp/Ats/Atp columns or a TMM stack for "
                    "phase-accurate results" % (b["label"], cname),
                    body=b["name"],
                    fix_hint="add ars_deg/arp_deg/ats_deg/atp_deg columns "
                    "to %s's table csv (Zemax TABLE convention) or switch "
                    "to a TMM layer stack" % cname,
                    check="coating-phase"))
    return out


@check("nlo-props")
def check_nlo_props(v):
    """Pulsed-optics Phase P8: Pockels cell / saturable absorber / TPA /
    Kerr thin-lens body properties. Registry-name validation follows the
    same pattern as check_library_names (coating/filter/polarizer/
    grating): unknown 'nonlinear'/'saturable'/'kerr_n2' registry rows are
    hard errors here too (same severity the engine itself gates on at
    Scene construction). pockels_voltage set without an attached
    kind=pockels 'nonlinear' row, and saturable/tpa_beta/kerr_n2/
    pockels_voltage placed on a SOURCE body, are advisory smells
    (WARNING) — the engine's own hard validation (role checks, crystal_
    axis/geometry/gap requirements) is the actual gate; this check exists
    so the GUI catches likely mistakes before a run."""
    if v.optprops is None:
        return [Finding(WARNING, "no optical-property library loaded; "
                        "nonlinear-property name checks skipped",
                        check="nlo-props")]
    p = v.optprops
    nonlinear_reg = getattr(p, "nonlinear", {}) or {}
    out = []
    for b in v.bodies():
        role = v.role(b)
        if role == "ignored":
            continue

        nl_raw = v.prop(b, "nonlinear")
        nl_row = None
        if nl_raw is not None:
            name = str(nl_raw).strip()
            if name.lower() in ("", "none"):
                nl_raw = None
            elif name not in nonlinear_reg:
                out.append(Finding(
                    ERROR, "%s: unknown nonlinear entry %r"
                    % (b["label"], name), body=b["name"],
                    fix_hint="add it to opticalproperties/nonlinear/"
                    "nonlinear.mienlo or fix the name",
                    check="nlo-props"))
            else:
                nl_row = nonlinear_reg[name]
            if role == "source":
                out.append(Finding(
                    WARNING, "%s: nonlinear is only meaningful on optic "
                    "bodies (this is a source)" % b["label"],
                    body=b["name"], check="nlo-props"))

        pv = v.prop(b, "pockels_voltage")
        if pv is not None:
            try:
                pv_val = float(pv)
            except (TypeError, ValueError):
                pv_val = None
            if role == "source":
                out.append(Finding(
                    WARNING, "%s: pockels_voltage is only meaningful on "
                    "optic bodies (this is a source)" % b["label"],
                    body=b["name"], check="nlo-props"))
            elif pv_val and (nl_row is None or nl_row.get("kind") != "pockels"):
                out.append(Finding(
                    WARNING, "%s: pockels_voltage is set but no "
                    "kind=pockels 'nonlinear' row is attached — it has "
                    "no effect" % b["label"], body=b["name"],
                    fix_hint="set nonlinear=<a kind=pockels registry row>",
                    check="nlo-props"))

        for prop_name, parser, kind, what in (
                ("saturable", common.parse_saturable_value, "saturable",
                 "saturable absorber"),
                ("kerr_n2", common.parse_kerr_n2_value, "n2", "Kerr n2")):
            raw = v.prop(b, prop_name)
            if raw is None:
                continue
            if role == "source":
                out.append(Finding(
                    WARNING, "%s: %s is only meaningful on optic bodies "
                    "(this is a source)" % (b["label"], prop_name),
                    body=b["name"], check="nlo-props"))
                continue
            try:
                spec = parser(str(raw))
            except ValueError as exc:
                out.append(Finding(ERROR, "%s: bad %s spec: %s"
                                   % (b["label"], prop_name, exc),
                                   body=b["name"], check="nlo-props"))
                continue
            reg_name = spec.get("registry")
            if reg_name is not None:
                row = nonlinear_reg.get(reg_name)
                if row is None or row.get("kind") != kind:
                    out.append(Finding(
                        ERROR, "%s: unknown %s registry entry %r"
                        % (b["label"], what, reg_name), body=b["name"],
                        fix_hint="add it to opticalproperties/nonlinear/"
                        "nonlinear.mienlo (kind=%s) or fix the name"
                        % kind, check="nlo-props"))

        tpa = v.prop(b, "tpa_beta")
        if tpa is not None and role == "source":
            out.append(Finding(WARNING, "%s: tpa_beta is only meaningful "
                               "on optic bodies (this is a source)"
                               % b["label"], body=b["name"],
                               check="nlo-props"))
    return out


@check("spec-syntax")
def check_spec_syntax(v):
    out = []
    for b in v.bodies():
        if v.role(b) == "ignored":
            continue
        pol = v.prop(b, "polarization")
        if pol is not None:
            try:
                common.parse_polarization_spec(str(pol))
            except ValueError as exc:
                out.append(Finding(ERROR, "%s: %s" % (b["label"], exc),
                                   body=b["name"], check="spec-syntax"))
        for name in ("polarizer_axis", "crystal_axis", "crystal_axis2"):
            axis = v.prop(b, name)
            if axis is None:
                continue
            try:
                parts = [float(x) for x in str(axis).split(",")]
                if len(parts) != 3 or not any(parts):
                    raise ValueError("need 3 comma-separated non-zero "
                                     "components")
            except ValueError as exc:
                out.append(Finding(ERROR, "%s: bad %s %r (%s)"
                                   % (b["label"], name, axis, exc),
                                   body=b["name"], check="spec-syntax"))
        for name in ("mirror", "absorbance"):
            val = v.prop(b, name)
            if val is None:
                continue
            try:
                x = float(val)
                if not 0.0 <= x <= 1.0:
                    raise ValueError
            except (TypeError, ValueError):
                out.append(Finding(ERROR, "%s: %s must be a number in "
                                   "[0, 1] (got %r)"
                                   % (b["label"], name, val),
                                   body=b["name"], check="spec-syntax"))
        for name in ("roughness", "surface_override", "scatter"):
            raw = v.prop(b, name)
            if raw is None:
                continue
            try:
                v.facemap_values(b, name)
            except ValueError as exc:
                out.append(Finding(ERROR, "%s: bad %s spec: %s"
                                   % (b["label"], name, exc),
                                   body=b["name"], check="spec-syntax"))
        apod = v.prop(b, "apodization")
        if apod is not None:
            try:
                common.parse_apodization_spec(str(apod))
            except ValueError as exc:
                out.append(Finding(ERROR, "%s: bad apodization spec: %s"
                                   % (b["label"], exc),
                                   body=b["name"], check="spec-syntax"))
        bw = v.prop(b, "beam_waist")
        if bw is not None:
            try:
                if not float(bw) > 0.0:
                    raise ValueError
            except (TypeError, ValueError):
                out.append(Finding(
                    ERROR, "%s: beam_waist must be a number > 0 mm "
                    "(got %r)" % (b["label"], bw), body=b["name"],
                    check="spec-syntax"))
        m2 = v.prop(b, "m2")
        if m2 is not None:
            try:
                if not float(m2) >= 1.0:
                    raise ValueError
            except (TypeError, ValueError):
                out.append(Finding(
                    ERROR, "%s: m2 must be a number >= 1.0 (got %r)"
                    % (b["label"], m2), body=b["name"],
                    check="spec-syntax"))
        sraw = v.prop(b, "scatter")
        if sraw is not None:
            try:
                smap = v.facemap_values(b, "scatter")
            except ValueError:
                smap = None   # already reported above
            if smap is not None:
                if v.optprops is not None:
                    scatter_reg = getattr(v.optprops, "scatter", {}) or {}
                    for face, name in smap.items():
                        if str(name).strip().lower() in ("", "none"):
                            continue
                        if name not in scatter_reg:
                            out.append(Finding(
                                ERROR, "%s: unknown scatter entry %r%s"
                                % (b["label"], name,
                                   "" if face in (None, common.FACEMAP_ALL)
                                   else " on %s" % face),
                                body=b["name"],
                                fix_hint="add it to opticalproperties/"
                                "scatter/bsdf.miebsdf or pick an "
                                "existing entry", check="spec-syntax"))
                for other in ("roughness", "diffuser"):
                    if v.prop(b, other) is None:
                        continue
                    try:
                        omap = v.facemap_values(b, other)
                    except ValueError:
                        continue   # already reported above
                    clash = (common.FACEMAP_ALL in smap
                             or common.FACEMAP_ALL in omap
                             or set(smap) & set(omap))
                    if clash:
                        out.append(Finding(
                            ERROR, "%s: scatter and %s cover the same "
                            "face(s) — they are alternative models of "
                            "one surface, pick one per face"
                            % (b["label"], other), body=b["name"],
                            check="spec-syntax"))
        draw = v.prop(b, "diffuser")
        if draw is not None:
            try:
                dmap = v.facemap_values(b, "diffuser")
                for value in dmap.values():
                    spec = common.parse_diffuser_value(value)
                    if "registry" in spec and v.optprops is not None:
                        reg = getattr(v.optprops, "diffusers", {}) or {}
                        if spec["registry"] not in reg:
                            raise ValueError(
                                "unknown diffuser registry entry %r"
                                % spec["registry"])
            except ValueError as exc:
                out.append(Finding(ERROR, "%s: bad diffuser spec: %s"
                                   % (b["label"], exc),
                                   body=b["name"], check="spec-syntax"))
            if v.prop(b, "roughness") is not None:
                try:
                    dmap = v.facemap_values(b, "diffuser")
                    rmap = v.facemap_values(b, "roughness")
                    clash = (common.FACEMAP_ALL in dmap
                             or common.FACEMAP_ALL in rmap
                             or set(dmap) & set(rmap))
                except ValueError:
                    clash = True   # unparseable maps already reported
                if clash:
                    out.append(Finding(
                        ERROR, "%s: diffuser and roughness cover the same "
                        "face(s) — they are alternative models of one "
                        "surface, pick one per face" % b["label"],
                        body=b["name"], check="spec-syntax"))
    return out


@check("wavelength-coverage")
def check_wavelength_coverage(v):
    if v.optprops is None:
        return []
    rng = v.source_range_nm()
    if rng is None:
        return []
    lo_um, hi_um = rng[0] * 1e-3, rng[1] * 1e-3
    p = v.optprops
    out = []

    def check_table(lam_um, what, name, body):
        if lam_um is None or len(lam_um) == 0:
            return
        tlo, thi = float(lam_um[0]), float(lam_um[-1])
        if lo_um < tlo or hi_um > thi:
            out.append(Finding(
                ERROR, "%s: %s %r covers %.0f-%.0f nm but the sources "
                "span %.0f-%.0f nm — the trace would hard-error "
                "(tables never extrapolate)"
                % (body["label"], what, name, tlo * 1e3, thi * 1e3,
                   lo_um * 1e3, hi_um * 1e3),
                body=body["name"],
                fix_hint="extend the table or narrow the source band",
                check="wavelength-coverage"))

    for b in v.bodies():
        role = v.role(b)
        if role == "ignored":
            continue
        mat = str(v.prop(b, "material") or "").strip()
        if role != "source" and mat and mat.lower() != "detector" \
                and mat in p.matdb:
            m = p.matdb.get(mat)
            try:
                m.n_complex(lo_um * 1e-6)
                m.n_complex(hi_um * 1e-6)
            except Exception as exc:
                out.append(Finding(ERROR, "%s: material %r cannot be "
                                   "evaluated across the source band: %s"
                                   % (b["label"], mat, exc),
                                   body=b["name"],
                                   check="wavelength-coverage"))
            tmin = getattr(m, "trans_min", None)
            tmax = getattr(m, "trans_max", None)
            if tmin is not None and tmax is not None \
                    and (lo_um < tmin or hi_um > tmax):
                out.append(Finding(
                    WARNING, "%s: source band %.0f-%.0f nm leaves %r's "
                    "stated transmission window %.0f-%.0f nm"
                    % (b["label"], lo_um * 1e3, hi_um * 1e3, mat,
                       tmin * 1e3, tmax * 1e3),
                    body=b["name"], check="wavelength-coverage"))
        f = v.prop(b, "filter")
        if f is not None and str(f) in p.filters:
            check_table(p.filters[str(f)].get("lam_um"), "filter",
                        str(f), b)
        pol = v.prop(b, "polarizer")
        if pol is not None and str(pol) in p.polarizers:
            check_table(p.polarizers[str(pol)].get("lam_um"), "polarizer",
                        str(pol), b)
    return out


@check("sampling")
def check_sampling(v):
    out = []
    rays = float(v.config.get("rays") or common.PRESETS["quick"]["rays"])
    nlambda = int(v.config.get("nlambda")
                  or common.PRESETS["quick"]["nlambda"])
    min_eff = float(v.config.get("min_eff_samples", 1000.0))
    n_sources = sum(1 for b in v.bodies() if v.role(b) == "source")
    if n_sources == 0:
        return []
    per_key = rays / max(nlambda, 1)
    if per_key < min_eff:
        out.append(Finding(
            WARNING, "~%.0f rays per (source, wavelength) stratum is "
            "below the gather gate of %.0f effective samples — the trace "
            "would abort at the gather stage"
            % (per_key, min_eff),
            fix_hint="raise --rays to at least %.0f, lower --nlambda, or "
            "disable the gate" % (min_eff * nlambda),
            check="sampling"))
    return out


@check("gather-preflight")
def check_gather_preflight(v):
    """Aperture-diffraction ray-budget preflight (design-usability round
    UXNOTES_ROUND3 #18/#29, future.md (a2)): a coherent source clipped
    down to a small transmitted footprint by a downstream aperture (or
    any element's own clear aperture) needs its TRANSMITTED population —
    not just its total rays/stratum — above the coherent gather's M_eff
    gate. Today that only surfaces as a GatherError AFTER a failed
    trace. This is a deliberately COARSE preflight: a plane aperture-area
    ratio (smallest downstream clear-aperture opening vs the source's
    own emitting footprint, both from dim-sheet aliases — see
    core.paraxial._APERTURE_ALIASES) estimates the transmitted fraction,
    with no geometry/divergence/propagation considered at all. WARNING
    only, never gates the Run button — it can both over- and
    under-estimate the real clipping."""
    out = []
    coherent_sources = [b for b in v.bodies()
                        if v.role(b) == "source"
                        and bool(v.prop(b, "coherent"))]
    if not coherent_sources:
        return out
    rays = float(v.config.get("rays") or common.PRESETS["quick"]["rays"])
    nlambda = int(v.config.get("nlambda")
                  or common.PRESETS["quick"]["nlambda"])
    # same default as check_sampling above, and the actual hard gate:
    # raytracer.gather.render_coherent's min_eff_samples kwarg (gather.py,
    # "SAMPLING GATE" header comment: M_eff >= 1000 by default).
    min_eff = float(v.config.get("min_eff_samples", 1000.0))
    per_stratum = rays / max(nlambda, 1)

    smallest_ap, smallest_body = None, None
    for b in v.bodies():
        if v.role(b) != "optic":
            continue
        d = v.aperture_mm(b)
        if d is not None and (smallest_ap is None or d < smallest_ap):
            smallest_ap, smallest_body = d, b
    if smallest_ap is None:
        return out

    for src in coherent_sources:
        beam_d = v.aperture_mm(src)
        if beam_d is None or beam_d <= 0 or smallest_ap >= beam_d:
            continue    # no aperture data, or no clipping to estimate
        transmitted_fraction = (smallest_ap / beam_d) ** 2
        eff_samples = transmitted_fraction * per_stratum
        if eff_samples < min_eff:
            mult = min_eff / max(eff_samples, 1e-9)
            out.append(Finding(
                WARNING,
                "%s (~Ø%.3g mm beam) through %s's ~Ø%.3g mm "
                "opening: a coarse aperture-area estimate gives only "
                "~%.0f effective transmitted rays/stratum, below the "
                "coherent gather's gate of %.0f effective samples — the "
                "trace may abort with GatherError: undersampled"
                % (src["label"], beam_d, smallest_body["label"],
                   smallest_ap, eff_samples, min_eff),
                body=src["name"],
                fix_hint="raise --rays by roughly %.0fx (to ~%.0f) to "
                "clear the gather gate through this aperture"
                % (mult, rays * mult),
                check="gather-preflight"))
    return out


@check("estimate")
def check_estimate(v):
    out = []
    rays = float(v.config.get("rays") or common.PRESETS["quick"]["rays"])
    nlambda = int(v.config.get("nlambda")
                  or common.PRESETS["quick"]["nlambda"])
    resolution = int(v.config.get("resolution")
                     or common.PRESETS["quick"]["resolution"])
    backend = str(v.config.get("backend") or "auto")
    n_coh = sum(1 for b in v.bodies() if v.role(b) == "source"
                and bool(v.prop(b, "coherent")))
    n_det = sum(1 for b in v.bodies() if v.role(b) == "detector")
    est = common.estimate(rays, resolution, nlambda, n_coh, backend,
                          n_detectors=max(n_det, 1))
    try:
        avail_gb = (os.sysconf("SC_AV_PHYS_PAGES")
                    * os.sysconf("SC_PAGE_SIZE")) / 2 ** 30
    except (ValueError, OSError, AttributeError):
        avail_gb = None
    if avail_gb is not None and est["accumulator_GB"] > 0.8 * avail_gb:
        out.append(Finding(
            ERROR, "detector accumulators need ~%.1f GB but only "
            "%.1f GB RAM is available"
            % (est["accumulator_GB"], avail_gb),
            fix_hint="lower --resolution or --spectral-bins",
            check="estimate"))
    out.append(Finding(
        INFO, "estimated runtime: trace %s + gather %s (accumulators "
        "%.2f GB)" % (common.fmt_duration(est["trace_s"]),
                      common.fmt_duration(est["gather_s"]),
                      est["accumulator_GB"]),
        check="estimate"))
    return out


def has_errors(findings):
    return any(f.severity == ERROR for f in findings)


def run_deep_checks(project):
    """Worker-side geometric checks (recompute errors, open solids,
    pairwise overlaps) via the live FreeCAD session. Returns [Finding].

    If no problems are found, returns a single INFO Finding describing
    the successful check."""
    result = project.fc.request("check", {"doc": project.doc})
    out = []
    for name in result.get("invalid", []):
        out.append(Finding(ERROR, "%s failed to recompute" % name,
                           body=name, check="deep-geometry"))
    for name in result.get("open_solids", []):
        out.append(Finding(ERROR, "%s is not a closed solid" % name,
                           body=name, check="deep-geometry"))
    n_overlaps = 0
    for ov in result.get("overlaps", []):
        out.append(Finding(
            WARNING, "%s and %s overlap by %.3g mm^3 — optically "
            "contacted surfaces need a small air gap (5 um convention)"
            % (ov["a"], ov["b"], ov["volume_mm3"]),
            body=ov["a"], check="deep-geometry"))
        n_overlaps += 1

    # If no problems were found, append an INFO Finding describing the check
    if not out:
        n_bodies = len(project.structure.get("bodies", []))
        n_face_pairs = result.get("face_pairs_checked", 0)
        msg = "Deep check passed — %d bodies recomputed, 0 open solids, " \
              "no overlapping pairs" % n_bodies
        if n_face_pairs > 0:
            msg += " (%d face pairs tested)" % n_face_pairs
        out.append(Finding(INFO, msg, check="deep-geometry"))

    return out
