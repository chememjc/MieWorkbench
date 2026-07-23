# =============================================================================
# scene.py — build the runtime scene from a validated model.json contract.
#
# Responsibilities:
#   * resolve bodies/faces into AnalyticFace objects (mesh faces are a hard
#     error in v1 — the extractor canonicalizes revolutions, so every optical
#     face in a lens/sphere/slit bench is analytic; see future.md)
#   * resolve materials & coatings against the MaterialDB
#   * apply --suppress-body (body treated as ignored, no FCStd edit)
#   * attach per-face physics options (grating, roughness, extra detectors)
#   * nearest-hit intersection across all active faces for a ray batch
#
# Face indexing: faces get integer ids (position in scene.faces); -1 = none.
# Body indexing: position in scene.bodies; AMBIENT (-1) = surrounding air.
# =============================================================================
import numpy as np

from .surfaces import make_surface, AnalyticFace
from . import nlo


FACEMAP_ALL = "__all__"     # matches common.FACEMAP_ALL (contract sentinel)


def _parse_pulse_source(label, src):
    """Pulsed-optics Phase P3: resolve a source's power specification and
    (optional) pulse metadata IN PLACE on `src` (the extracted source
    record — see extract_geometry.py's source_dict). Body properties
    arrive as (all optional except lambdac, checked upstream):
    power_mW, pulse_energy_uJ, pulse_duration_ps, rep_rate_hz.

    Rules:
      * power XOR pulse_energy: both present is a hard error (which
        average-power definition should win is ambiguous); neither
        present is ALSO an error — this is the pre-existing "a source
        needs a power spec" contract, now phrased as an XOR since
        pulse_energy is a second valid way to satisfy it.
      * pulse_energy (+ mandatory rep_rate — average power is otherwise
        underdetermined) derives power_W = energy_J * rep_rate_Hz,
        written back to src["power_mW"] so every downstream consumer
        (sampling in sources.py, the power ledger, ...) sees ordinary
        average power and needs no pulse-awareness at all.
      * power + rep_rate (no pulse_energy) reverse-derives
        pulse_energy_J = power_W / rep_rate_Hz.
      * pulse_duration alone (no energy, no rep_rate) is legal: CW
        virtual-pulse mode, a bare tau0 annotation for a later envelope/
        chirp model — the source is still continuous-wave average power.

    Whenever ANY of {pulse_energy, pulse_duration, rep_rate} is present,
    sets src["pulse"] = {energy_J, duration_s, rep_rate_Hz, peak_power_W,
    avg_power_W, derived, kappa}. All 7 keys are ALWAYS present (matching
    the lambdamin_nm/lambdamax_nm convention elsewhere in this contract:
    key present, value None when not computable) rather than being
    omitted — simpler for downstream code to consume with a plain
    .get()/indexing. peak_power_W = 0.94 * energy_J / duration_s (Gaussian
    pulse shape factor) needs BOTH energy_J and duration_s; kappa =
    peak_power_W / avg_power_W needs peak_power_W. `derived` records
    which of {power_mW, pulse_energy_J} this function computed
    ("power"/"pulse_energy"/None — CW-with-duration-only derives
    neither). A plain non-pulsed source (power only, nothing else) gets
    no "pulse" key at all — zero footprint for the overwhelming majority
    of existing scenes."""
    power_mw = src.get("power_mW")
    energy_uj = src.get("pulse_energy_uJ")
    duration_ps = src.get("pulse_duration_ps")
    rep_rate_hz = src.get("rep_rate_hz")

    # power_mW == 0.0 means "not authored" (extract_geometry's sentinel
    # for a pulse_energy-only source — common.validate_model requires
    # source.power_mW to already be a real float, so extract_geometry
    # can't leave it None; see that file's comment at the source_dict
    # build site). None is accepted too, for hand-built model dicts
    # (tests) that skip the sentinel and simply omit the key.
    has_power = power_mw is not None and power_mw != 0.0
    has_energy = energy_uj is not None
    has_duration = duration_ps is not None
    has_rep_rate = rep_rate_hz is not None

    if has_power and has_energy:
        raise ValueError(
            "source %s: both power (%.6g mW) and pulse_energy (%.6g uJ) "
            "are set — pick one (average power is ambiguous otherwise)"
            % (label, power_mw, energy_uj))
    if not has_power and not has_energy:
        raise ValueError(
            "source %s: needs either 'power' (mW) or 'pulse_energy' (uJ) "
            "— neither is present" % label)

    if has_energy and energy_uj <= 0:
        raise ValueError("source %s: pulse_energy must be > 0 uJ (got %g)"
                         % (label, energy_uj))
    if has_duration and duration_ps <= 0:
        raise ValueError("source %s: pulse_duration must be > 0 ps (got %g)"
                         % (label, duration_ps))
    if has_rep_rate and rep_rate_hz <= 0:
        raise ValueError("source %s: rep_rate must be > 0 Hz (got %g)"
                         % (label, rep_rate_hz))

    energy_j = None
    derived = None
    if has_energy:
        if not has_rep_rate:
            raise ValueError(
                "source %s: pulse_energy needs rep_rate — average power "
                "is underdetermined without it" % label)
        energy_j = energy_uj * 1e-6
        power_w = energy_j * rep_rate_hz
        src["power_mW"] = power_w * 1e3
        derived = "power"
    else:
        power_w = power_mw * 1e-3
        if has_rep_rate:
            energy_j = power_w / rep_rate_hz
            derived = "pulse_energy"

    if not (has_energy or has_duration or has_rep_rate):
        return   # ordinary source: no pulse annotation at all

    duration_s = duration_ps * 1e-12 if has_duration else None
    peak_power_w = (0.94 * energy_j / duration_s
                    if (energy_j is not None and duration_s is not None)
                    else None)
    avg_power_w = power_w
    kappa = (peak_power_w / avg_power_w
             if (peak_power_w is not None and avg_power_w > 0) else None)

    src["pulse"] = {
        "energy_J": energy_j,
        "duration_s": duration_s,
        "rep_rate_Hz": rep_rate_hz,
        "peak_power_W": peak_power_w,
        "avg_power_W": avg_power_w,
        "derived": derived,
        "kappa": kappa,
    }


class Body:
    __slots__ = ("index", "name", "label", "role", "material", "coating",
                 "mirror", "absorbance", "absorbance_faces",
                 "roughness_nm", "roughness_faces",
                 "diffuser_faces", "scatter_faces", "scatter_targets",
                 "grating_map", "figure_map", "source",
                 "detector", "closed", "face_ids", "polarizer",
                 "polarizer_axis", "filter", "crystal_axis", "birefringent",
                 "filter_lam_um", "filter_alpha_per_m", "crystal_axis2",
                 "crystal_frame", "biaxial", "gyration",
                 # pulsed-optics Phase P8 (Pockels / saturable / TPA / Kerr):
                 "nonlinear", "pockels_voltage", "pockels_gap_mm",
                 "pockels_mats", "saturable_raw", "saturable_spec",
                 "tpa_beta", "kerr_n2_raw", "kerr_n2_value", "shg_spec",
                 "temperature_c",
                 # samples-instruments round: body-bound sample media
                 "sample", "bbox_m")

    def __init__(self, index, rec):
        self.index = index
        self.name = rec["name"]
        self.label = rec["label"]
        self.role = rec["role"]
        self.material = rec.get("material")
        # coating: v1 = bare string (whole body), v2 = per-face dict.
        # Normalize to a dict {face_id_or_FACEMAP_ALL: name} (None if none).
        coat = rec.get("coating")
        if coat in (None, "", "none"):
            self.coating = None
        elif isinstance(coat, str):
            self.coating = {FACEMAP_ALL: coat}
        else:
            self.coating = dict(coat)
        self.mirror = float(rec.get("mirror") or 0.0)
        self.absorbance = float(rec.get("absorbance") or 0.0)
        # per-face absorbance (edge blackening, engine3 Sec 11 / P8): {face_id:
        # frac} -- overrides the whole-body absorbance on those faces only (a
        # blackened lens EDGE that swallows ghost/stray light while the optical
        # surfaces stay clear). dict or None.
        self.absorbance_faces = rec.get("absorbance_faces")
        # NOTE the contract key is roughness_nm ('roughness' was a latent
        # mismatch — body-tagged roughness silently never reached the trace)
        self.roughness_nm = float(rec.get("roughness_nm") or 0.0)
        self.roughness_faces = rec.get("roughness_faces")   # dict or None
        self.diffuser_faces = rec.get("diffuser_faces")     # dict or None
        self.scatter_faces = rec.get("scatter_faces")       # dict or None
        # optional explicit importance-scatter targets: list of detector
        # labels this body's scatter faces aim at (None => every detector)
        st = rec.get("scatter_targets")
        self.scatter_targets = ([s.strip() for s in st.split(",") if s.strip()]
                                if isinstance(st, str) else (st or None))
        self.grating_map = rec.get("grating")               # dict or None
        # figure error (Zernike surface figure, engine3 Sec 11 / P8): per-face
        # map {face_id_or_FACEMAP_ALL: registry_name}, resolved to a
        # PerturbedSurface wrapper in the face-build loop.
        fig = rec.get("figure_error")
        if fig in (None, "", "none"):
            self.figure_map = None
        elif isinstance(fig, str):
            self.figure_map = {FACEMAP_ALL: fig}
        else:
            self.figure_map = dict(fig)
        self.polarizer = rec.get("polarizer") or None
        pa = rec.get("polarizer_axis")
        self.polarizer_axis = np.asarray(pa, dtype=np.float64) \
            if pa is not None else np.array([0.0, 0.0, 1.0])
        self.filter = rec.get("filter") or None
        ca = rec.get("crystal_axis")
        # v1 models carry no crystal_axis: default global +x (README §5)
        self.crystal_axis = np.asarray(ca, dtype=np.float64) \
            if ca is not None else np.array([1.0, 0.0, 0.0])
        self.birefringent = False           # set by Scene from the matdb
        # (rho_deg_per_mm, ref_lam_nm) for an optically-active (gyrotropic)
        # uniaxial crystal, else None. Set by Scene from the matdb; drives
        # scene-level natural optical activity in the tracer (bulk rotation
        # of the polarization plane along the optic axis).
        self.gyration = None
        ca2 = rec.get("crystal_axis2")
        self.crystal_axis2 = np.asarray(ca2, dtype=np.float64) \
            if ca2 is not None else None
        self.crystal_frame = None           # (3,3) rows = principal axes,
        self.biaxial = False                # both set by Scene (biaxial)
        self.filter_lam_um = None           # set by Scene from optprops
        self.filter_alpha_per_m = None
        self.source = rec.get("source")
        self.detector = rec.get("detector")
        self.closed = bool(rec.get("solid_closed", True))
        # optional per-body operating-temperature override (deg C); None ->
        # use the scene-global temperature. Only shifts materials that carry
        # a thermo-optic model (Material.has_thermo).
        t = rec.get("temperature")
        self.temperature_c = float(t) if t is not None and t != "" else None
        self.face_ids = []

        # ---- pulsed-optics Phase P8: Pockels / saturable / TPA / Kerr ----
        # 'nonlinear' names a nonlinear.mienlo registry row (kind=pockels or
        # chi2_*; resolved+validated by Scene, which also sets pockels_mats
        # for an attached pockels row). 'saturable'/'kerr_n2' are resolved
        # by Scene too (registry row OR inline spec, common.
        # parse_saturable_value/parse_kerr_n2_value) into saturable_spec
        # ({I_sat_W_cm2, T0, alpha0_per_mm}) / kerr_n2_value (float, m^2/W).
        self.nonlinear = rec.get("nonlinear") or None
        self.pockels_voltage = float(rec.get("pockels_voltage") or 0.0)
        pg = rec.get("pockels_gap_mm")
        self.pockels_gap_mm = float(pg) if pg is not None else None
        self.pockels_mats = None            # (mat_o, mat_e) shifted proxies
        self.saturable_raw = rec.get("saturable") or None
        self.saturable_spec = None
        self.tpa_beta = float(rec.get("tpa_beta") or 0.0)   # cm/GW
        self.shg_spec = None     # P7b: resolved chi2_process row (Scene)
        self.kerr_n2_raw = rec.get("kerr_n2") or None
        self.kerr_n2_value = None           # m^2/W

        # ---- samples-instruments round: body-bound sample media ----------
        # 'sample' names a sample/samples.miesamp registry row; the tracer
        # builds a particle medium bounded by THIS body's interior with the
        # body's own material as the host (raytracer/particles.py
        # BodyParticleMedium). bbox_m is the extractor's world-space AABB
        # (metres) — placement bounds for explicit realizations.
        self.sample = rec.get("sample") or None
        bb = rec.get("bbox_m")
        self.bbox_m = ((np.asarray(bb["min"], dtype=np.float64),
                        np.asarray(bb["max"], dtype=np.float64))
                       if bb else None)

    def filter_alpha(self, lam_m):
        """Additive bulk absorption coefficient [1/m] at wavelength(s) [m]
        from this body's spectral filter table (0 if no filter). Hard error
        outside the tabulated range — no extrapolation."""
        if self.filter_lam_um is None:
            return 0.0
        from .optprops import interp_hard
        return interp_hard(np.asarray(lam_m) * 1e6, self.filter_lam_um,
                           self.filter_alpha_per_m,
                           "body %s filter %r" % (self.label, self.filter))


class Scene:
    def __init__(self, model, matdb, coatings, suppress_bodies=(),
                 extra_detector_faces=(), grating_specs=(), rough_specs=(),
                 optprops=None, geometry_dir=None, strict_analytic=False,
                 mesh_flat_normals=False, temperature_c=None):
        """model: validated model.json dict; matdb: MaterialDB;
        coatings: {name: {"kind": "tmm"|"table", ...}} from load_coatings;
        optprops: optional OpticalProperties (polarizer/filter/grating
        registries — required when the model uses those properties);
        geometry_dir: directory containing the model's faces/*.stl (needed
        to trace mesh-type faces via the BVH — typically the model.json's
        parent); strict_analytic restores the v1 hard error on mesh faces."""
        self.matdb = matdb
        self.coatings = coatings
        self.optprops = optprops
        # scene-global operating temperature (deg C); None -> each material's
        # own reference temperature (no thermo-optic shift). A per-body
        # 'temperature' property overrides this for that body.
        self.temperature_c = (float(temperature_c)
                              if temperature_c is not None else None)
        polarizers = optprops.polarizers if optprops is not None else {}
        filters = optprops.filters if optprops is not None else {}
        grating_registry = optprops.gratings if optprops is not None else {}
        emission = optprops.emission if optprops is not None else {}
        nonlinear_registry = optprops.nonlinear if optprops is not None else {}
        self.polarizers = polarizers
        # needed inside the body loop below (nonlinear/saturable/kerr_n2
        # value-string parsing) — imported here (not at module level) to
        # match the existing convention lower in this method.
        import common as _common
        self.ambient = matdb.get(model.get("ambient_material", "air"))
        suppress = set(suppress_bodies)

        self.bodies = []
        self.faces = []                    # AnalyticFace, index = face id
        self.face_body = []                # face id -> body index
        self.face_records = []             # face id -> contract face dict
        self.sources = []                  # (body_index, source dict)
        self.emit_faces = {}               # body_index -> AnalyticFace
        self.detector_faces = {}           # face id -> owning body index
        self.face_by_name = {}             # "Body.Feature.FaceN" -> face id

        unknown = [s for s in suppress
                   if s not in {b["name"] for b in model["bodies"]}
                   and s not in {b["label"] for b in model["bodies"]}]
        if unknown:
            raise ValueError("--suppress-body names not in model: %r"
                             % unknown)

        for rec in model["bodies"]:
            if rec["role"] == "ignored":
                continue
            if rec["name"] in suppress or rec["label"] in suppress:
                continue
            body = Body(len(self.bodies), rec)
            # resolve material/coating/polarizer/filter now — unknown names
            # must fail here, not mid-trace
            if body.role == "optic":
                if matdb.is_birefringent(body.material):
                    body.birefringent = True
                    nrm = np.linalg.norm(body.crystal_axis)
                    if nrm < 1e-9:
                        raise ValueError("body %s: zero crystal_axis"
                                         % body.label)
                    body.crystal_axis = body.crystal_axis / nrm
                    # natural optical activity (gyrotropic crystals, e.g.
                    # alpha-quartz): carry the registry rotatory power so the
                    # tracer can rotate the polarization plane along the axis
                    body.gyration = matdb.gyration(body.material)
                elif matdb.is_biaxial(body.material):
                    # full principal frame: crystal_axis = X, crystal_axis2
                    # = Y (Gram-Schmidt orthogonalized), Z = X x Y
                    if body.crystal_axis2 is None:
                        raise ValueError(
                            "body %s: biaxial material %r needs BOTH "
                            "crystal_axis (X principal axis) and "
                            "crystal_axis2 (Y)" % (body.label,
                                                   body.material))
                    x = body.crystal_axis
                    nx = np.linalg.norm(x)
                    if nx < 1e-9:
                        raise ValueError("body %s: zero crystal_axis"
                                         % body.label)
                    x = x / nx
                    y = body.crystal_axis2
                    y = y - np.dot(y, x) * x
                    ny = np.linalg.norm(y)
                    if ny < 1e-6:
                        raise ValueError(
                            "body %s: crystal_axis2 is (near-)parallel to "
                            "crystal_axis — principal frame undefined"
                            % body.label)
                    y = y / ny
                    body.crystal_frame = np.stack([x, y, np.cross(x, y)])
                    body.biaxial = True
                else:
                    matdb.get(body.material)
            if body.coating is not None:
                for cname in body.coating.values():
                    if cname not in coatings:
                        raise ValueError(
                            "body %s: unknown coating %r (coatings.csv "
                            "has: %s)" % (body.label, cname,
                                          ", ".join(sorted(coatings))))
            if body.polarizer is not None:
                if body.polarizer not in polarizers:
                    raise ValueError(
                        "body %s: unknown polarizer %r (polarizers.csv "
                        "has: %s)" % (body.label, body.polarizer,
                                      ", ".join(sorted(polarizers)) or
                                      "<none loaded — pass optprops>"))
                nrm = np.linalg.norm(body.polarizer_axis)
                if nrm < 1e-9:
                    raise ValueError("body %s: zero polarizer_axis"
                                     % body.label)
                body.polarizer_axis = body.polarizer_axis / nrm
            if body.filter is not None:
                if body.filter not in filters:
                    raise ValueError(
                        "body %s: unknown filter %r (filters.csv has: %s)"
                        % (body.label, body.filter,
                           ", ".join(sorted(filters)) or
                           "<none loaded — pass optprops>"))
                fentry = filters[body.filter]
                body.filter_lam_um = fentry["lam_um"]
                body.filter_alpha_per_m = fentry["alpha_per_m"]

            # ---- pulsed-optics Phase P8: Pockels / chi2-accept / --------
            # ---- saturable absorber / TPA / Kerr n2 ---------------------
            if body.nonlinear is not None:
                if body.role != "optic":
                    raise ValueError(
                        "body %s: nonlinear is only meaningful on optic "
                        "bodies (role=%s)" % (body.label, body.role))
                if body.nonlinear not in nonlinear_registry:
                    raise ValueError(
                        "body %s: unknown nonlinear entry %r "
                        "(opticalproperties/nonlinear/nonlinear.mienlo "
                        "has: %s)" % (body.label, body.nonlinear,
                                     ", ".join(sorted(nonlinear_registry))
                                     or "<none loaded — pass optprops>"))
                nrow = nonlinear_registry[body.nonlinear]
                if nrow["kind"] == "chi2_tensor":
                    raise ValueError(
                        "body %s: chi2_tensor row %r cannot drive the "
                        "SHG event directly — it has no resolved d_eff/"
                        "phase-matching geometry. Attach a chi2_process "
                        "row instead (derive one from the tensor with "
                        "nlo.d_eff_tensor + nlo.phase_match_angle; see "
                        "docs/RAYTRACER.md)" % (body.label,
                                                body.nonlinear))
                elif nrow["kind"] == "chi2_process":
                    # pulsed-optics P7b: the deterministic per-segment
                    # SHG transfer (tracer.step). Design pump wavelength
                    # is exactly phase-matched (delta_k = 0); detuned rays
                    # get the scalar-index sinc^2 falloff (tracer's
                    # _shg_delta_k — walk-off/angular detuning are
                    # documented out of scope).
                    if not str(nrow["process"]).startswith("shg"):
                        raise ValueError(
                            "body %s: chi2_process row %r is %r — only "
                            "SHG processes drive the bulk event this "
                            "phase" % (body.label, body.nonlinear,
                                       nrow["process"]))
                    if (str(nrow.get("crystal") or "").strip().lower()
                            != str(body.material or "").strip().lower()):
                        # soft warning (unlike pockels' hard error): the
                        # row only supplies d_eff + the design pump; the
                        # INDICES (detuning, Fresnel) come from the
                        # body's material — mixing them is legitimate in
                        # test benches but usually an authoring mistake
                        import warnings
                        warnings.warn(
                            "body %s: chi2 row %r is for crystal %r but "
                            "the body material is %r — detuning/Fresnel "
                            "use the BODY's indices"
                            % (body.label, body.nonlinear,
                               nrow.get("crystal"), body.material))
                    body.shg_spec = {
                        "name": body.nonlinear,
                        "d_eff_m_V": nrow["d_eff_pm_V"] * 1e-12,
                        "lam_pump_m": nrow["lam_pump_nm"] * 1e-9,
                    }
                elif nrow["kind"] == "pockels":
                    if not body.birefringent:
                        raise ValueError(
                            "body %s: pockels row %r needs a birefringent "
                            "material (body material is %r)"
                            % (body.label, body.nonlinear, body.material))
                    if (nrow["crystal"].strip().lower()
                            != body.material.strip().lower()):
                        raise ValueError(
                            "body %s: pockels row %r crystal %r does not "
                            "match the body's birefringent material %r"
                            % (body.label, body.nonlinear, nrow["crystal"],
                               body.material))
                    if nrow["geometry"] != "transverse":
                        raise ValueError(
                            "body %s: pockels row %r uses %r geometry — "
                            "this engine phase (P8) implements the "
                            "TRANSVERSE Pockels geometry ONLY (documented "
                            "scope); a %r cell needs a later engine phase"
                            % (body.label, body.nonlinear, nrow["geometry"],
                               nrow["geometry"]))
                    if "r33" not in nrow["r_pm_V"] \
                            or "r13" not in nrow["r_pm_V"]:
                        raise ValueError(
                            "body %s: pockels row %r needs BOTH r33 and "
                            "r13 coefficients for the transverse geometry "
                            "(got %s)" % (body.label, body.nonlinear,
                                         sorted(nrow["r_pm_V"])))
                    if body.pockels_gap_mm is None:
                        raise ValueError(
                            "body %s: pockels row %r needs the "
                            "pockels_gap body property (mm) — the "
                            "transverse-field gap distance d in E = V/d"
                            % (body.label, body.nonlinear))
                    mo, me = matdb.get_uniaxial(body.material)
                    gap_m = body.pockels_gap_mm * 1e-3
                    body.pockels_mats = nlo.pockels_shifted_materials(
                        mo, me, nrow["r_pm_V"], gap_m, body.pockels_voltage)
                else:
                    raise ValueError(
                        "body %s: the 'nonlinear' body property expects "
                        "a pockels or chi2_* registry row (got kind=%r "
                        "for %r) — saturable absorption and the Kerr "
                        "effect use their own 'saturable'/'kerr_n2' body "
                        "properties" % (body.label, nrow["kind"],
                                       body.nonlinear))

            if body.saturable_raw is not None:
                if body.role != "optic":
                    raise ValueError(
                        "body %s: saturable is only meaningful on optic "
                        "bodies (role=%s)" % (body.label, body.role))
                parsed = _common.parse_saturable_value(body.saturable_raw)
                if "registry" in parsed:
                    sat_names = sorted(k for k, v in
                                       nonlinear_registry.items()
                                       if v["kind"] == "saturable")
                    srow = nonlinear_registry.get(parsed["registry"])
                    if srow is None or srow["kind"] != "saturable":
                        raise ValueError(
                            "body %s: unknown saturable registry entry "
                            "%r (kind=saturable rows: %s)"
                            % (body.label, parsed["registry"],
                               ", ".join(sat_names) or "<none>"))
                    body.saturable_spec = {
                        "I_sat_W_cm2": srow["I_sat_W_cm2"], "T0": srow["T0"],
                        "alpha0_per_mm": srow.get("alpha0_per_mm")}
                else:
                    body.saturable_spec = {
                        "I_sat_W_cm2": parsed["I_sat_W_cm2"],
                        "T0": parsed["T0"], "alpha0_per_mm": None}

            if body.kerr_n2_raw is not None:
                if body.role != "optic":
                    raise ValueError(
                        "body %s: kerr_n2 is only meaningful on optic "
                        "bodies (role=%s)" % (body.label, body.role))
                parsed = _common.parse_kerr_n2_value(body.kerr_n2_raw)
                if "registry" in parsed:
                    n2_names = sorted(k for k, v in
                                      nonlinear_registry.items()
                                      if v["kind"] == "n2")
                    krow = nonlinear_registry.get(parsed["registry"])
                    if krow is None or krow["kind"] != "n2":
                        raise ValueError(
                            "body %s: unknown kerr_n2 registry entry %r "
                            "(kind=n2 rows: %s)"
                            % (body.label, parsed["registry"],
                               ", ".join(n2_names) or "<none>"))
                    mat_name = krow["material"]
                    if mat_name not in matdb:
                        raise ValueError(
                            "body %s: kerr_n2 registry row %r's material "
                            "%r is not in materials.miemat (a STAGED n2 "
                            "row — see "
                            "library_data/staged/nonlinear_staging_notes.md)"
                            % (body.label, parsed["registry"], mat_name))
                    body.kerr_n2_value = krow["n2_m2_W"]
                else:
                    body.kerr_n2_value = parsed["n2_m2_W"]

            self.bodies.append(body)

            if body.role == "source":
                src = rec["source"]
                # pulsed-optics Phase P3: resolve power vs pulse_energy
                # (XOR + derivation) BEFORE anything downstream reads
                # src["power_mW"] — must run before body.source is stored
                # and before sample_source ever sees this dict.
                _parse_pulse_source(body.label, src)
                spec_name = src.get("spectrum")
                if spec_name is not None:
                    if spec_name not in emission:
                        raise ValueError(
                            "source %s: unknown spectrum %r (emission "
                            "registry has: %s)"
                            % (body.label, spec_name,
                               ", ".join(sorted(emission)) or
                               "<none loaded — pass optprops>"))
                    entry = emission[spec_name]
                    if entry["kind"] == "lines":
                        # discrete emission lines -- NO lam_nm/relative_power
                        # keys (optprops.load_emission's docstring); the
                        # per-line stratum allocation lives in
                        # sources.wavelength_strata, keyed off these three
                        # arrays/scalar instead of a tabulated PDF.
                        src["_lines_nm"] = np.asarray(entry["lines_nm"],
                                                      dtype=np.float64)
                        src["_lines_intensity"] = np.asarray(
                            entry["intensity"], dtype=np.float64)
                        src["_lines_linewidth_nm"] = float(
                            entry["linewidth_nm"])
                    else:
                        # continuous or blackbody (blackbody is synthesized
                        # to a dense table AT LOAD -- see load_emission's
                        # docstring -- so it carries the SAME lam_nm/
                        # relative_power keys and needs no special case here)
                        src["_spectrum_lam_nm"] = np.asarray(
                            entry["lam_nm"], dtype=np.float64)
                        src["_spectrum_pdf"] = np.asarray(
                            entry["relative_power"], dtype=np.float64)
                # samples-instruments round: extended image-emitting source.
                # `image` names an image/images.mieimg registry row; pixels
                # are loaded HERE (once per scene build) into the greyscale
                # radiance map sources._sample_image_points consumes.
                image_name = src.get("image")
                if image_name is not None:
                    images = (self.optprops.images
                              if self.optprops is not None else {})
                    if image_name not in images:
                        raise ValueError(
                            "source %s: unknown image %r (image registry "
                            "has: %s)"
                            % (body.label, image_name,
                               ", ".join(sorted(images)) or
                               "<none loaded — pass optprops>"))
                    from .sources import load_image_gray
                    src["_image_gray"] = load_image_gray(
                        images[image_name]["path"])
                    if src.get("beam"):
                        raise ValueError(
                            "source %s: image and beam_waist are mutually "
                            "exclusive (a bitmap radiance map has no "
                            "Gaussian mode)" % body.label)
                    if src.get("apodization"):
                        raise ValueError(
                            "source %s: image and apodization are mutually "
                            "exclusive (the bitmap IS the transverse "
                            "profile)" % body.label)
                    if src.get("coherent"):
                        import warnings
                        warnings.warn(
                            "source %s: coherent extended image source — "
                            "physical for a laser-illuminated transparency; "
                            "an incandescent/diffuse scene should set "
                            "coherent=false" % body.label)
                self.sources.append((body.index, src))
                # source bodies contribute no intersectable geometry (the
                # housing is not traced), but the emitting face itself is
                # built for area sampling and kept OUT of self.faces
                emit_name = rec["source"]["emit_face"]
                emit_rec = next((f for f in rec["faces"]
                                 if f["id"] == emit_name), None)
                if emit_rec is None:
                    raise ValueError("source %s: emit face %r not among "
                                     "its faces" % (body.label, emit_name))
                if emit_rec["surface"]["type"] == "mesh":
                    raise NotImplementedError(
                        "source %s: emitting face is mesh-type" % body.label)
                self.emit_faces[body.index] = AnalyticFace(
                    emit_rec["id"], make_surface(emit_rec["surface"]),
                    emit_rec["trim_polylines_xyz"],
                    emit_rec["orientation_outward"], body.index, -1,
                    area_m2=emit_rec["area_m2"])
                continue

            for f in rec["faces"]:
                if f["surface"]["type"] == "mesh":
                    if strict_analytic:
                        raise NotImplementedError(
                            "face %s is mesh-type (non-analytic) and "
                            "--strict-analytic is set" % f["id"])
                    if body.role == "detector" \
                            and f["id"] == rec["detector"]["face"]:
                        raise NotImplementedError(
                            "detector screen face %s is mesh-type — "
                            "detector grids need an analytic plane"
                            % f["id"])
                    stl = f.get("mesh_stl") or ""
                    path = None
                    if geometry_dir is not None and stl:
                        from pathlib import Path
                        path = Path(geometry_dir) / stl
                    if path is None or not path.exists():
                        raise ValueError(
                            "mesh face %s: STL %r not found — pass "
                            "geometry_dir (the model.json's directory)"
                            % (f["id"], stl))
                    from .mesh import MeshFace
                    face = MeshFace(f, path,
                                    flat_normals=mesh_flat_normals)
                else:
                    surf = make_surface(f["surface"])
                    fig_name = self._face_figure_name(body, f["id"])
                    if fig_name is not None:
                        from .surfaces import PerturbedSurface
                        reg = (self.optprops.figures
                               if self.optprops is not None else {})
                        if fig_name not in reg:
                            raise ValueError(
                                "body %s face %s: unknown figure_error %r "
                                "(figures registry has: %s)"
                                % (body.label, f["id"], fig_name,
                                   ", ".join(sorted(reg))
                                   or "<none loaded — pass optprops>"))
                        fspec = reg[fig_name]
                        surf = PerturbedSurface(
                            surf, fspec["coeffs"], fspec["r_norm_m"])
                    face = AnalyticFace(
                        f["id"], surf,
                        f["trim_polylines_xyz"], f["orientation_outward"],
                        body.index, len(self.faces), area_m2=f["area_m2"])
                fid = len(self.faces)
                self.faces.append(face)
                self.face_body.append(body.index)
                self.face_records.append(f)
                self.face_by_name[f["id"]] = fid
                body.face_ids.append(fid)

            if body.role == "detector":
                det_face = rec["detector"]["face"]
                if det_face not in self.face_by_name:
                    raise ValueError("detector face %r not found on body %s"
                                     % (det_face, body.label))
                self.detector_faces[self.face_by_name[det_face]] = body.index

        if not self.sources:
            raise ValueError("no active sources (all suppressed?)")
        if not self.detector_faces:
            raise ValueError("no active detectors (all suppressed?)")

        # extra CLI detector faces: transparent zero-effect screens on any
        # existing face (including optical-element faces)
        self.extra_detector_faces = set()
        for spec in extra_detector_faces:
            fid = self._face_id_or_die(spec, "detector")
            self.extra_detector_faces.add(fid)

        # ---- per-face physics options ---------------------------------
        # precedence everywhere: CLI spec > body per-face entry > body
        # whole-body value. (_common imported earlier in this method.)
        self.gratings = {}
        # body 'grating' property (per-face dict, values from
        # common.parse_grating_value)
        for body in self.bodies:
            if body.grating_map:
                for face_name, value in body.grating_map.items():
                    spec = _common.parse_grating_value(value)
                    fid = self._face_id_or_die(face_name, "grating")
                    spec["face"] = {"id": face_name}
                    self.gratings[fid] = spec
        # CLI --grating overrides
        for g in grating_specs:
            self.gratings[self._face_id_or_die(g["face"]["id"],
                                               "grating")] = g
        # resolve @registry refs against opticalproperties/grating/
        for fid, spec in self.gratings.items():
            reg = spec.get("registry")
            if reg is None:
                continue
            if reg not in grating_registry:
                raise ValueError(
                    "grating on %s: unknown registry entry %r (gratings.csv "
                    "has: %s)" % (self.faces[fid].id, reg,
                                  ", ".join(sorted(grating_registry)) or
                                  "<none loaded — pass optprops>"))
            entry = grating_registry[reg]
            spec["model"] = entry["model"]
            spec["lines_per_mm"] = entry["lines_per_mm"]
            spec["params"] = entry["params"]
            spec["table"] = entry["table"]
        for fid in self.gratings:
            if self.faces[fid].surface is None:
                raise ValueError(
                    "grating on mesh face %s: gratings need an analytic "
                    "surface (groove/UV geometry)" % self.faces[fid].id)

        self.roughness = {}
        # body per-face roughness strings, then whole-body float
        for body in self.bodies:
            if body.roughness_faces:
                for face_name, value in body.roughness_faces.items():
                    rv = _common.parse_rough_value(value)
                    if face_name == FACEMAP_ALL:
                        for fid in body.face_ids:
                            self.roughness[fid] = dict(
                                rv, face={"id": self.faces[fid].id})
                    else:
                        fid = self._face_id_or_die(face_name, "roughness")
                        self.roughness[fid] = dict(rv,
                                                   face={"id": face_name})
            if body.roughness_nm > 0:
                for fid in body.face_ids:
                    self.roughness.setdefault(
                        fid, {"face": {"id": self.faces[fid].id},
                              "sigma_nm": body.roughness_nm,
                              "lcorr_um": 10.0})
        # CLI --rough overrides everything
        for r in rough_specs:
            self.roughness[self._face_id_or_die(r["face"]["id"],
                                                "roughness")] = r

        # ---- ground-glass diffusers: the deep-rough limit of the same
        # microfacet model. Each diffuser face resolves to an RMS slope
        # (grit table / explicit / registry) and lands in self.roughness
        # as a sigma>>lambda entry (specular retention exactly 0, every
        # ray scattered through one Beckmann facet with full
        # per-polarization Fresnel). A face carrying BOTH diffuser and
        # roughness is a contract error, not a merge.
        from .roughness import diffuser_equivalent, slope_for_grit
        for body in self.bodies:
            if not body.diffuser_faces:
                continue
            for face_name, value in body.diffuser_faces.items():
                spec = _common.parse_diffuser_value(value)
                if "registry" in spec:
                    reg = (self.optprops.diffusers
                           if self.optprops is not None else {})
                    entry = reg.get(spec["registry"])
                    if entry is None:
                        raise ValueError(
                            "body %s: unknown diffuser registry entry %r "
                            "(opticalproperties/diffuser/diffusers.miedif)"
                            % (body.label, spec["registry"]))
                    slope = entry["slope_rms"]
                elif "grit" in spec:
                    slope = slope_for_grit(spec["grit"])
                else:
                    slope = spec["slope"]
                sigma_nm, lcorr_um = diffuser_equivalent(slope)
                if face_name == FACEMAP_ALL:
                    fids = list(body.face_ids)
                else:
                    fids = [self._face_id_or_die(face_name, "diffuser")]
                for fid in fids:
                    if fid in self.roughness:
                        raise ValueError(
                            "body %s: face %s carries BOTH a diffuser and "
                            "a roughness declaration — they are "
                            "alternative models of one surface, pick one"
                            % (body.label, self.faces[fid].id))
                    self.roughness[fid] = {
                        "face": {"id": self.faces[fid].id},
                        "sigma_nm": sigma_nm, "lcorr_um": lcorr_um,
                        "diffuser": True}

        # ---- measured-scatter (ABg/BSDF) faces: reflected-side lobe from
        # a registry entry (opticalproperties/scatter/). Resolves to
        # self.scatter = {fid: entry}. A face carrying scatter AND roughness
        # OR diffuser is a contract error — they are alternative surface
        # models (the roughness map already holds any diffuser entries).
        self.scatter = {}
        # {fid: [detector label, ...]} explicit importance-scatter aim
        # subsets (from the body's scatter_targets); absent => aim at every
        # detector (Tracer._scatter_target_list). Never affects the physics
        # unless --importance-scatter is on.
        self.scatter_targets = {}
        scatter_registry = (self.optprops.scatter
                            if self.optprops is not None else {})
        for body in self.bodies:
            if not body.scatter_faces:
                continue
            for face_name, name in body.scatter_faces.items():
                if name not in scatter_registry:
                    raise ValueError(
                        "body %s: unknown scatter entry %r "
                        "(opticalproperties/scatter/bsdf.miebsdf has: %s)"
                        % (body.label, name,
                           ", ".join(sorted(scatter_registry)) or
                           "<none loaded — pass optprops>"))
                if face_name == FACEMAP_ALL:
                    fids = list(body.face_ids)
                else:
                    fids = [self._face_id_or_die(face_name, "scatter")]
                for fid in fids:
                    if fid in self.roughness:
                        raise ValueError(
                            "body %s: face %s carries BOTH a scatter and a "
                            "roughness/diffuser declaration — they are "
                            "alternative models of one surface, pick one"
                            % (body.label, self.faces[fid].id))
                    self.scatter[fid] = scatter_registry[name]
                    if body.scatter_targets:
                        self.scatter_targets[fid] = list(body.scatter_targets)

        # per-face coating map: {int fid: coating name}
        self.face_coatings = {}
        for body in self.bodies:
            if not body.coating:
                continue
            default = body.coating.get(FACEMAP_ALL)
            if default is not None:
                for fid in body.face_ids:
                    self.face_coatings[fid] = default
            for face_name, cname in body.coating.items():
                if face_name == FACEMAP_ALL:
                    continue
                self.face_coatings[self._face_id_or_die(
                    face_name, "coating")] = cname

        # per-face absorbance map: {int fid: absorbance frac} (edge blackening)
        self.face_absorbance = {}
        for body in self.bodies:
            if not body.absorbance_faces:
                continue
            for face_name, frac in body.absorbance_faces.items():
                self.face_absorbance[self._face_id_or_die(
                    face_name, "absorbance")] = float(frac)

        self.face_body = np.asarray(self.face_body, dtype=np.int32)

    @staticmethod
    def _face_figure_name(body, face_id):
        """Figure-error registry name for one face of `body`, or None:
        an exact per-face key wins over the FACEMAP_ALL whole-body default."""
        fm = body.figure_map
        if not fm:
            return None
        if face_id in fm:
            return fm[face_id]
        return fm.get(FACEMAP_ALL)

    def _face_id_or_die(self, face_name, kind):
        if face_name not in self.face_by_name:
            raise ValueError(
                "%s face %r not found in scene. Available faces:\n  %s"
                % (kind, face_name,
                   "\n  ".join(sorted(self.face_by_name))))
        return self.face_by_name[face_name]

    # ------------------------------------------------------------------
    def source_bodies(self):
        return [(self.bodies[i], src) for i, src in self.sources]

    def body_of_face(self, fid):
        return self.bodies[self.face_body[fid]]

    def medium_index(self, body_index, lam):
        """Complex n for rays inside body body_index (-1 = ambient).

        Birefringent bodies return the ORDINARY index n_o: o-rays and any
        non-mode-tagged path (e.g. a grating on a crystal face) use it;
        e-rays override the real part via RayBatch.n_eff, and absorption
        uses Im(n_o) for both modes (documented approximation)."""
        if body_index < 0:
            return self.ambient.n_complex(lam)
        body = self.bodies[body_index]
        if body.role == "detector" or body.material in (None, "detector"):
            # detector solids are ideal thin screens; treat interior as
            # ambient (rays never legitimately travel "inside" them)
            return self.ambient.n_complex(lam)
        T = self._body_temperature(body)
        if body.birefringent:
            return self.uniaxial_materials(body)[0].n_complex(lam, T=T)
        if body.biaxial:
            # scalar bookkeeping index (medium stack / seam accounting):
            # the geometric mean keeps it sheet-neutral
            mx, my, mz = self.matdb.get_biaxial(body.material)
            return (np.real(mx.n_complex(lam, T=T))
                    * np.real(my.n_complex(lam, T=T))
                    * np.real(mz.n_complex(lam, T=T))) ** (1.0 / 3.0)
        return self.matdb.get(body.material).n_complex(lam, T=T)

    def _body_temperature(self, body):
        """Operating temperature (deg C) for a body: its own override if set,
        else the scene-global temperature (None -> material reference temp)."""
        return body.temperature_c if body.temperature_c is not None \
            else self.temperature_c

    def medium_group_index(self, body_index, lam):
        """Real GROUP index n_g for rays inside body body_index (-1 =
        ambient), mirroring medium_index's medium resolution. Ambient and
        detector interiors return exactly 1.0 (the vacuum-like envelope
        reference — the group delay of an ambient path is its geometric
        length, so gopl == d in air by construction; the ~3e-4 constant-
        model air index carries no dispersion anyway). Birefringent bodies
        return the ORDINARY material's group index (e-rays override via
        RayBatch.n_g_eff, same convention as n_eff); biaxial bodies return
        the geometric-mean bookkeeping value (sheet rays always carry
        n_g_eff)."""
        if body_index < 0:
            return np.ones_like(np.asarray(lam, dtype=np.float64))
        body = self.bodies[body_index]
        if body.role == "detector" or body.material in (None, "detector"):
            return np.ones_like(np.asarray(lam, dtype=np.float64))
        if body.birefringent:
            return self.uniaxial_materials(body)[0].n_group(lam)
        if body.biaxial:
            mx, my, mz = self.matdb.get_biaxial(body.material)
            return (mx.n_group(lam) * my.n_group(lam)
                    * mz.n_group(lam)) ** (1.0 / 3.0)
        return self.matdb.get(body.material).n_group(lam)

    def medium_gdd_per_length(self, body_index, lam):
        """Material group-delay dispersion per unit length [s^2/m] for rays
        inside body body_index (-1 = ambient), mirroring medium_index.
        Ambient/detector interiors: 0.0. Birefringent: the o material
        (same isotropic fallback as medium_index); biaxial: arithmetic
        mean of the three principal materials (diagnostic-grade scalar
        bookkeeping, like medium_index's geometric-mean phase index)."""
        from .materials import gdd_per_length
        lam = np.asarray(lam, dtype=np.float64)
        if body_index < 0:
            return np.zeros_like(lam)
        body = self.bodies[body_index]
        if body.role == "detector" or body.material in (None, "detector"):
            return np.zeros_like(lam)
        if body.birefringent:
            return gdd_per_length(self.uniaxial_materials(body)[0], lam)
        if body.biaxial:
            mats = self.matdb.get_biaxial(body.material)
            return sum(gdd_per_length(m, lam) for m in mats) / 3.0
        return gdd_per_length(self.matdb.get(body.material), lam)

    def medium_tod_per_length(self, body_index, lam):
        """Material third-order dispersion per unit length [s^3/m] for rays
        inside body body_index (-1 = ambient), with EXACTLY
        medium_gdd_per_length's medium-resolution fallbacks (ambient/
        detector 0.0; birefringent -> o material; biaxial -> arithmetic
        mean). Used by the --gdd-budget table, not the hot path."""
        from .materials import tod_per_length
        lam = np.asarray(lam, dtype=np.float64)
        if body_index < 0:
            return np.zeros_like(lam)
        body = self.bodies[body_index]
        if body.role == "detector" or body.material in (None, "detector"):
            return np.zeros_like(lam)
        if body.birefringent:
            return tod_per_length(self.uniaxial_materials(body)[0], lam)
        if body.biaxial:
            mats = self.matdb.get_biaxial(body.material)
            return sum(tod_per_length(m, lam) for m in mats) / 3.0
        return tod_per_length(self.matdb.get(body.material), lam)

    def uniaxial_materials(self, body):
        """(mat_o, mat_e) Material-like objects for a birefringent body —
        the Pockels-SHIFTED proxy pair (nlo._ShiftedIndex) when the body
        carries an attached EO cell (body.pockels_mats, set at Scene
        construction from a 'nonlinear'=pockels row + pockels_voltage/
        pockels_gap), else the bare uniaxial registry pair. Single choke
        point so every consumer (medium_index, medium_group_index,
        medium_gdd_per_length, uniaxial_indices, and the tracer's
        directional-group-index lookups in _birefringent_children) sees
        the Pockels shift transparently — retardance, dispersion and
        group delay all follow from the SAME two proxies."""
        if body.pockels_mats is not None:
            return body.pockels_mats
        return self.matdb.get_uniaxial(body.material)

    def uniaxial_indices(self, body, lam):
        """(n_o_real, n_e_real) arrays for a birefringent body at lam [m]."""
        mo, me = self.uniaxial_materials(body)
        return (np.real(mo.n_complex(lam)), np.real(me.n_complex(lam)))

    def biaxial_eps(self, body, lam):
        """(n,3) principal permittivities (n_x^2, n_y^2, n_z^2), real, for
        a biaxial body at lam [m] (per-ray arrays for dispersion)."""
        mx, my, mz = self.matdb.get_biaxial(body.material)
        return np.stack([np.real(m.n_complex(lam)) ** 2
                         for m in (mx, my, mz)], axis=-1)

    # ------------------------------------------------------------------
    def intersect(self, pos, direction):
        """Nearest hit across all faces.

        Self-intersection is handled by the faces' t_eps guard (100 nm),
        NOT by excluding the last-hit face — a ray reflected internally in
        a sphere legitimately re-hits the same face.
        Returns (t (N,), face_id (N,) int32 with -1 = miss).
        """
        n = len(pos)
        best_t = np.full(n, np.inf)
        best_f = np.full(n, -1, dtype=np.int32)
        for fid, face in enumerate(self.faces):
            t, hit = face.intersect(pos, direction)
            better = hit & (t < best_t)
            best_t[better] = t[better]
            best_f[better] = fid
        return best_t, best_f

    def point_inside_body(self, pts, body_index, direction=None,
                          max_hits=64):
        """Parity (even-odd) containment test: True where pts (N,3, metres)
        lie strictly inside the closed solid of body body_index.

        Marches a probe ray from each point along `direction` (default a
        fixed irrational-ish direction that avoids axis-aligned tangency on
        boxes/cylinders), counting crossings with THIS BODY'S faces only.
        Each march step advances 100 nm past the hit — the same t_eps scale
        the face intersectors use — so a quadric face hit twice by one
        probe is counted twice (nearest-hit-per-face alone would break
        parity). Used for sample-medium particle placement (rejection
        sampling), not per-bounce physics — cost is one-off.
        """
        body = self.bodies[body_index]
        if not body.closed:
            raise ValueError(
                "point_inside_body: body %s is not a closed solid"
                % body.label)
        pts = np.asarray(pts, dtype=np.float64)
        if direction is None:
            direction = np.array([0.912871, 0.365148, 0.182574])
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        n = len(pts)
        crossings = np.zeros(n, dtype=np.int64)
        pos = pts.copy()
        active = np.ones(n, dtype=bool)
        dirs = np.broadcast_to(d, (n, 3)).copy()
        for _ in range(max_hits):
            idx = np.where(active)[0]
            if len(idx) == 0:
                break
            best_t = np.full(len(idx), np.inf)
            for fid in body.face_ids:
                t, hit = self.faces[fid].intersect(pos[idx], dirs[idx])
                better = hit & (t < best_t)
                best_t[better] = t[better]
            hit_any = np.isfinite(best_t)
            crossings[idx[hit_any]] += 1
            adv = idx[hit_any]
            pos[adv] = pos[adv] + (best_t[hit_any] + 1e-7)[:, None] \
                * dirs[adv]
            active[idx[~hit_any]] = False
        return (crossings % 2) == 1
