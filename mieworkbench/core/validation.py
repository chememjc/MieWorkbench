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
        if self.prop(body, "power") is not None \
                and self.prop(body, "lambdac") is not None:
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


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
@check("roles")
def check_roles(v):
    out = []
    roles = [v.role(b) for b in v.bodies()]
    if "source" not in roles:
        out.append(Finding(ERROR, "scene has no light source (a source "
                           "body needs both 'power' and 'lambdac' "
                           "properties)",
                           fix_hint="add a laser/source element or set "
                           "power+lambdac on a body", check="roles"))
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
            known = (mat in p.matdb) or (mat in p.uniaxial)
            if not known:
                out.append(Finding(ERROR, "%s: unknown material %r"
                                   % (b["label"], mat), body=b["name"],
                                   fix_hint="add it to materials.miemat "
                                   "or pick an existing one",
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
        for name in ("polarizer_axis", "crystal_axis"):
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
        for name in ("roughness", "surface_override"):
            raw = v.prop(b, name)
            if raw is None:
                continue
            try:
                v.facemap_values(b, name)
            except ValueError as exc:
                out.append(Finding(ERROR, "%s: bad %s spec: %s"
                                   % (b["label"], name, exc),
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
