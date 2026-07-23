# =============================================================================
# cengine.py — Python glue for the C/OpenMP(/CUDA) engine (cengine/).
#
# Three responsibilities:
#   1. Feature detection + engine routing (choose_engine): scan the BUILT
#      Scene + CLI args for every physics feature the case uses; route to
#      the C engine only when ALL of them are in the PORTED set. The Python
#      engine remains the permanent reference — anything unported (or any
#      doubt) runs there.
#   2. Request building (build_request): pre-resolve every dispersive
#      quantity at the fixed stratum wavelengths (plan D1 — rays only ever
#      carry those exact values, sources.py:331) and serialize scene
#      geometry + parameters to <case>/cengine/request_seed<k>.json.
#      Detector grid geometry comes from REAL DetectorGrid objects so both
#      engines share the pixel mapping bit-for-bit.
#   3. Output conversion (run_c_case): spawn miewb-trace once per seed,
#      then convert its raw outputs (npy cubes + ledger/detected JSON) into
#      the existing case contract — rays.npy, detectors/<label>.h5 (via
#      run_trace.save_detectors, the SAME writer the Python engine uses),
#      audit.json, case.json blocks.
#
# The C engine binary: cengine/build/miewb-trace (override: MIEWB_CENGINE).
# Missing binary => engine=auto silently routes to Python.
# =============================================================================
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# scripts/ is on sys.path by the time every known caller imports this module
# (run_trace.py, the cengine parity/registry tests) — but guard it here too
# so `import common` below is safe standalone (same bare-module convention
# run_trace.py uses, not a package-relative import).
_scripts_dir = str(Path(__file__).resolve().parent.parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
import common                                             # noqa: E402

# ---------------------------------------------------------------------------
# Ported-feature registry. Grows phase by phase; a feature is added ONLY
# after its side-by-side parity tests pass (see plan: cengine round).
# ---------------------------------------------------------------------------
PORTED = frozenset({
    "surface:plane",            # phase A
    "surface:sphere",           # phase A
    "filter",                   # phase A (bulk alpha pre-resolved per lam)
    "surface:cylinder",         # phase B
    "surface:cone",             # phase B
    "surface:torus",            # phase B
    "surface:asphere",          # phase B
    "coating",                  # phase B (TMM per-ray; tables per-lam)
    "polarizer",                # phase B
    "surface:mesh",             # phase C (triangle BLAS)
    "coherent",                 # phase D (Huygens gather, CPU/CUDA)
    "save_fields",              # phase D (complex Ex/Ey field maps)
    "grating",                  # phase E (all models; kogelnik per-ray)
    "roughness",                # phase E (Beckmann lobes, micro Fresnel)
    "scatter",                  # phase E (ABg lobes, g == 2)
    "birefringence",            # phase F (uniaxial o/e; biaxial stays
                                #   Python-routed via its own feature)
    "particles",                # phase G (continuum mode; the explicit
                                #   realization keeps its own feature)
    "sample_body",              # samples-instruments: body-bound CONTINUUM
                                #   sample medium (the `sample` body property).
                                #   Same continuum kernel as the CLI box cloud,
                                #   region-gated by the medium stack (top ==
                                #   the host body) instead of a slab; host
                                #   material effects are pre-baked into the
                                #   per-medium tables the glue serializes, so
                                #   the C side needs zero body/solvent
                                #   knowledge. Explicit/lattice sample rows
                                #   emit the unported "sample_explicit" token.
    "export_rays",              # phase H (per-detector landing records)
    "ghost_analysis",           # phase H (refl_hist face-id history)
    "viz_pattern",              # phase H (glue-level: Python viz-only
                                #   pass supplies the overlay rays)
    "image_source",             # samples-instruments: extended image-emitting
                                #   source. The glue serializes a Vose alias
                                #   table over the bitmap pixels (built by the
                                #   SAME sources._build_alias_table — one
                                #   implementation) + the face-UV bbox + cone
                                #   half-angle; the C sampler (trace.c
                                #   sample_image_pos_dir) alias-draws a pixel,
                                #   jitters in-pixel, and emits Lambertian/cone
                                #   about the signed emit normal. Requires a
                                #   planar emit face (both engines error else).
    "gdd_budget",               # P7 tranche 1: per-body power-weighted bulk
                                #   path tallied in C; ALL dispersion (group
                                #   index / GDD / TOD) resolved Python-side
                                #   in build_gdd_budget (untouched)
    "time_products",            # P7 tranche 1: per-ray gopl/gdd accumulators
                                #   (group index / GDD-per-length pre-resolved
                                #   in the request) + per-detector arrival
                                #   records; Python finalize_time bins them
                                #   UNCHANGED. Crystal scenes emit the unported
                                #   time_directional_index token instead.
    "ray_differentials",        # P7 tranche 2: Igehy ray differentials seeded
                                #   + transported through free flight / specular
                                #   reflect+refract (kernels/diffk.h, a
                                #   formula-for-formula port of differentials.py
                                #   + surfaces.normal_derivative, oracle-pinned
                                #   at 1e-5 vs finite differences). Sizes the
                                #   coherent gather's per-sample dA from
                                #   |dPdx x dPdy|; grating orders / scatter lobes
                                #   / o-e splits drop the differential (NaN ->
                                #   source-area fallback), same as Python.
    "saturable",                # P7 tranche 2: intensity-dependent saturable
                                #   absorption alpha0/(1+I/I_sat) on the
                                #   homogeneous-propagator alpha_add hook
                                #   (nlo.saturable_alpha_per_m; glue pre-resolves
                                #   alpha0_per_m + I_sat to SI). Energy lands in
                                #   absorbed_bulk.
    "tpa",                      # P7 tranche 2: two-photon absorption
                                #   alpha_TPA(I)=beta_SI*I on the SAME alpha_add
                                #   hook (nlo.tpa_alpha_per_m).
    "kerr",                     # P7 tranche 2: Kerr thin-element bulk phase
                                #   Delta_opl = n2*I*L added to opl for COHERENT
                                #   rays (tracer.py:394-436). Per-ray intensity
                                #   (p/dA)*kappa from the ported ray_differentials
                                #   dA, else the source flat-top area + warning.
    # NOTE: "surface:qforbes" (raytracer.surfaces.QForbes, the ISO 10110-12
    # Forbes Q-type asphere -- engine3.md Sec 7.6) is DELIBERATELY absent.
    # detect_features()'s per-face loop below emits it automatically
    # (type(face.surface).__name__.lower(), same generic path every other
    # "surface:*" token comes from) -- no code change was needed to make
    # every qforbes scene route Python-only. Do not add it here until the
    # C engine actually implements the Forbes recurrence (see the P8
    # incident note ~line 104: a feature silently routing to C without its
    # physics is the one failure mode this registry exists to prevent).
})

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def binary_path():
    """Path to miewb-trace, or None if not built. Resolution order:
    the MIEWB_CENGINE env var (checked explicitly here, for clarity, even
    though common.CENGINE_BINARY already folds it in) > common.CENGINE_BINARY
    (miewb.env's MIEWB_CENGINE line, if any — repo machine-config
    convention, like MIEWB_FREECAD etc.) > the repo-default build path."""
    env = os.environ.get("MIEWB_CENGINE") or common.CENGINE_BINARY
    p = Path(env) if env else _REPO_ROOT / "cengine" / "build" / "miewb-trace"
    return p if p.is_file() and os.access(p, os.X_OK) else None


# ---------------------------------------------------------------------------
# feature detection
# ---------------------------------------------------------------------------
def detect_features(args, scene):
    """Every gate-relevant physics feature this case would exercise.
    Detection runs on the BUILT scene (post CLI overrides), so --grating /
    --rough / --suppress-body etc. are already folded in."""
    feats = set()
    for face in scene.faces:
        surf = getattr(face, "surface", None)
        name = "mesh" if surf is None else type(surf).__name__.lower()
        feats.add("surface:%s" % name)
        # Zernike surface figure error (engine3 Sec 11 / P8): the
        # "surface:perturbedsurface" token above already forces Python (it is
        # deliberately absent from PORTED, exactly like "surface:qforbes"), but
        # emit an explicit, self-documenting "figure_error" token too -- same
        # "every feature emits its token" rule the registry exists to enforce.
        if type(surf).__name__ == "PerturbedSurface":
            feats.add("figure_error")
    for bidx, src in scene.sources:
        feats.add("surface:%s"
                  % type(scene.emit_faces[bidx].surface).__name__.lower())
        if src.get("coherent"):
            feats.add("coherent")
        if src.get("beam"):
            feats.add("beam")
        if src.get("apodization"):
            feats.add("apodization")
        # samples-instruments round: extended image-emitting source
        # (per-pixel alias-method emission, sources.py). PORTED: the glue
        # serializes the Vose alias table + face-UV bbox + cone half-angle
        # and the C sampler (trace.c sample_image_pos_dir) reproduces the
        # density + Lambertian/cone emission statistically.
        if src.get("image"):
            feats.add("image_source")
    biref_approx = getattr(args, "biref_approx", False)
    for body in scene.bodies:
        if body.birefringent:
            # The C engine's birefk.h computes interface amplitudes with the
            # legacy effective-index approximation (trace.c fresnel_eval on
            # n_phase_e). The default Python path now uses the EXACT
            # Lekner-1991 amplitudes, which are NOT ported -> emit the
            # unported "biref_exact" token so exact-uniaxial scenes honestly
            # route to Python. Under --biref-approx both engines agree, so
            # emit the ported "birefringence" token (C stays available).
            feats.add("birefringence" if biref_approx else "biref_exact")
            # natural optical activity (gyrotropic crystals) is a Python-only
            # bulk polarization-transport effect (tracer._apply_optical_
            # activity) with NO C counterpart -- force Python even under
            # --biref-approx so a gyrotropic scene never silently loses its
            # rotation to the C engine.
            if body.gyration is not None:
                feats.add("gyration")
        if body.biaxial:
            feats.add("biaxial")
            # P9: exact biaxial interface amplitudes ride the Berreman 4x4
            # (berreman.py). Under --biref-approx the legacy effective-index
            # Fresnel is used and both engines agree on that approximation, so
            # only the plain "biaxial" (Python-only) token is emitted. The
            # `berreman` token is a C-registry seam STUB (trace.c INTERACTIONS,
            # match=m_never): it is NOT in registry_supported_token, so a
            # forced --engine c hard-errors naming it and --engine auto routes
            # Python (verified: registry_dump_tokens skips stub tokens, so it
            # never appears in --tokens). Emit it whenever the new-physics path
            # is actually taken so the stub's hard-error contract stays honest.
            if not biref_approx:
                feats.add("berreman")
        if body.polarizer:
            feats.add("polarizer")
        if body.filter:
            feats.add("filter")
        # pulsed-optics NLO elements. The `nonlinear` body property (chi2 SHG
        # or a Pockels cell) still forces Python:
        #   * chi2 SHG: the harmonic child is born at lam/2 in a NEW wavelength
        #     stratum (n_lambda + parent) with its own detector-array slot;
        #     that lambda-union / stratum plumbing is a later tranche.
        #   * Pockels: the transverse EO cell IS a birefringent body whose
        #     n_o/n_e ride the ported uniaxial kernel (the index shift is
        #     pre-baked into the tables the glue serializes) — BUT the Python
        #     reference engine cannot currently trace a Pockels body end-to-end
        #     (scene.medium_index passes T= to the _ShiftedIndex proxy, which
        #     nlo.pockels_shifted_materials builds WITHOUT a temperature kwarg;
        #     masked only because the sole e2e Pockels test is @slow + xfail).
        #     With no working reference there is nothing to gate C parity
        #     against, so the token stays Python-only until that reference bug
        #     is fixed (Python NLO physics is out of this tranche's scope).
        # saturable/tpa/kerr are ported bulk effects (P7 tranche 2). The
        # "every feature emits its token" rule is why each still emits: a
        # kerr body whose OTHER features were all ported once silently routed
        # to C and skipped the physics.
        if body.nonlinear:
            feats.add("nonlinear")
        if body.saturable_raw:
            feats.add("saturable")
        if body.tpa_beta:
            feats.add("tpa")
        if body.kerr_n2_raw:
            feats.add("kerr")
        # samples-instruments round: body-bound sample media (the `sample`
        # body property -> a particle population bounded by this body's
        # interior, host = the body's material). CONTINUUM mode is ported
        # (medium-stack-gated particle medium reusing the ensemble tables);
        # EXPLICIT/lattice realizations stay Python-routed. The effective mode
        # is resolved exactly like BodyParticleMedium (registry mode override,
        # else count vs threshold) so routing can never disagree with the
        # medium the trace actually builds. The defensive getattr keeps this
        # inert until scene bodies grow the attribute.
        if getattr(body, "sample", None) is not None:
            if _sample_body_mode(args, scene, body) == "continuum":
                feats.add("sample_body")
            else:
                feats.add("sample_explicit")
        # BTDF (transmitted-side measured scatter, this round): the ported
        # "scatter" token covers BRDF-only ABg rows; a row carrying
        # transmitted-side (A_t/B_t/g_t) columns must emit the reserved
        # "scatter_btdf" token (already in the test_registry_tokens
        # PYTHON_ONLY partition) where scatter rows are resolved below --
        # C port deferred, documented in future.md.
    # thermo-optic shift: any optic body whose effective operating
    # temperature differs from its material reference AND carries a
    # thermo-optic model changes the index -> Python engine only (the C
    # index path mirrors n(lambda) at reference temperature, no dn/dT term).
    for body in scene.bodies:
        if body.role != "optic":
            continue
        T_eff = body.temperature_c if body.temperature_c is not None \
            else scene.temperature_c
        if T_eff is None:
            continue
        try:
            mat = scene.matdb.get(body.material)
        except Exception:
            continue
        if getattr(mat, "has_thermo", False) and float(T_eff) != mat.t_ref_c:
            feats.add("temperature")
            break
    if scene.gratings:
        feats.add("grating")
        # v2 RCWA tables carry COMPLEX per-order amplitudes interpolated on
        # (lambda, theta, phi); the C grating kernel only bakes a lambda-only
        # real [order][lam] efficiency table (no theta/phi axes, no phase), so
        # a v2 table is not yet ported. Emit its own token -> --engine auto
        # routes such scenes to Python (honest routing; C port is a later
        # tranche). NEVER let a v2 grating silently run the C lambda-only path.
        for gspec in scene.gratings.values():
            tbl = gspec.get("table")
            if isinstance(tbl, dict) and tbl.get("schema") == "v2":
                feats.add("grating_table_v2")
                break
    if scene.roughness:
        feats.add("roughness")
        if (getattr(args, "rough_fresnel", None) or "micro") == "macro":
            # legacy nominal-angle scalar model: Python engine only
            feats.add("rough_fresnel_macro")
    if scene.scatter:
        feats.add("scatter")
        for entry in scene.scatter.values():
            if abs(float(entry["g"]) - 2.0) >= 1e-12:
                # numeric inverse-CDF sampler: Python engine only
                feats.add("scatter_g_ne_2")
            # transmissive scatter (BTDF): the refracted-side scattered lobe
            # is Python-only this round (the C scatter kernel is reflected-
            # side ABg). Every scene using it MUST fall back, never silently
            # skip the transmitted lobes (the P8-comment rule above).
            if entry.get("btdf") is not None:
                feats.add("scatter_btdf")
    # measured-scatter importance sampling (§7.1): detector-aimed cone
    # children + rejection-sampled remainder are a Python-engine feature this
    # round. Emit the token so --engine auto routes to Python rather than
    # running the C full-lobe sampler and losing the variance reduction.
    if getattr(args, "importance_scatter", False) and scene.scatter:
        feats.add("scatter_importance")
    if scene.face_coatings:
        feats.add("coating")
        # P2: a phase-carrying table coating (materials.py phase_valid)
        # changes the emitted amplitude's phase, not just its magnitude --
        # the C engine's table-coating path (still the phase-invalid
        # bare-Fresnel-phase borrow) would silently give a DIFFERENT
        # (wrong) coherent answer than Python for the same scene. Force
        # Python routing until the C side implements it (not this round's
        # scope) -- same "every feature emits its token" rule that the P8
        # NLO incident above enforces (a ported-looking scene silently
        # skipping unported physics).
        if any(scene.coatings[cname].get("phase_valid")
               for cname in set(scene.face_coatings.values())):
            feats.add("coating_phase")
    if scene.extra_detector_faces:
        feats.add("extra_detector_faces")
    for fid in scene.detector_faces:
        if type(scene.faces[fid].surface).__name__ != "Plane":
            feats.add("curved_detector")
    # CLI-flag features (trace-behavior-changing only; post-stage flags
    # like --photometric never gate the engine)
    if args.particles:
        feats.add("particles")
        # continuum mode is ported; explicit realizations (count under
        # the threshold: frozen spheres, complex S1/S2 speckle) stay on
        # the Python engine
        import common
        from .mie import LogNormalDistribution, number_density, MieEvaluator
        spec = common.parse_particles_spec(args.particles)
        dist = LogNormalDistribution(
            median_r=spec["median_um"] * 1e-6 / 2.0, gsd=spec["gsd"])
        mat_p = scene.matdb.get(spec["material"])
        rho_h = scene.ambient.density if scene.ambient.density > 0 \
            else 1.204
        phi = spec["phi"]
        if phi is None:
            # tau= spec: resolve the target optical depth to a phi exactly
            # like ParticleCloud will (same lam set, same closed form) so
            # the explicit-vs-continuum routing decision here can never
            # disagree with the mode the trace actually runs. (Previously
            # this path crashed number_density(None, ...) — a tau spec
            # could not route through --engine auto at all.)
            from .particles import resolve_tau_phi
            from .sources import wavelength_strata
            lam_list = sorted({
                float(l) for _, src in scene.sources
                for l in wavelength_strata(src, args.nlambda)})
            phi, _info = resolve_tau_phi(
                spec["tau"], float(spec["box_size_m"][0]),
                MieEvaluator(mat_p, scene.ambient), dist,
                mat_p.density, rho_h, lam_list)
        N, _ = number_density(phi, mat_p.density, rho_h, dist)
        count = N * float(np.prod(spec["box_size_m"]))
        thr = args.particle_threshold if args.particle_threshold \
            is not None else common.DEFAULTS["particle_threshold"]
        if count <= thr:
            feats.add("particles_explicit")
    if args.ray_differentials:
        feats.add("ray_differentials")
    if args.export_rays:
        feats.add("export_rays")
    if args.ghost_analysis:
        feats.add("ghost_analysis")
    # P2 parallel-transport polarization analysis: RayBatch.Qmat/Jmat
    # bookkeeping exists in the Python engine only (same silent-skip rule
    # as every other engine-diagnostic feature above — the token forces
    # 'auto' off C so --pol-transport is never quietly dropped).
    if getattr(args, "pol_transport", False):
        feats.add("pol_transport")
    # samples-instruments round: internal conical refraction (biaxial
    # optic-axis fan) is Python-only physics — biaxial scenes already
    # Python-route via their own tokens, but the flag emits its token
    # anyway ("every feature emits its token"): a --conical run on a
    # uniaxial-only scene must not silently drop the fan by routing to C.
    if getattr(args, "conical", False):
        feats.add("conical")
    if args.viz_pattern:
        feats.add("viz_pattern")
    if args.save_fields:
        feats.add("save_fields")
    # pulsed-optics time products (P4): time-binned detector recording
    # (track_time + arrival records) exists in the Python engine only.
    # resolve_time_products folds in BOTH triggers — an explicit
    # --time-products AND the auto-enable rule (pulsed source present, no
    # flag) — so routing can never disagree with run_trace's activation.
    from .detector import resolve_time_products
    if resolve_time_products(args, scene):
        feats.add("time_products")
        # The C gopl accumulator uses the medium's SCALAR group index. A
        # uniaxial e-ray / biaxial slow-fast ray carries a DIRECTIONAL group
        # index frozen at the crystal entry interface (rays.py n_g_eff,
        # birefringence.n_group_e_theta) that the C engine does not carry —
        # so a crystal + time-products scene must route to Python. Emit an
        # unported token (never a silent wrong group delay through a crystal).
        if any(b.birefringent or b.biaxial for b in scene.bodies):
            feats.add("time_directional_index")
    # --gdd-budget forces group-delay tracking (per-body path tally) even
    # on a CW scene with no time products — Python engine only (P5)
    if getattr(args, "gdd_budget", False):
        feats.add("gdd_budget")
    return feats


def choose_engine(args, scene):
    """(engine, reason) — 'c' only when the binary exists AND every
    detected feature is ported. --engine c with unported features is a
    hard error naming them (never a silent wrong answer)."""
    mode = getattr(args, "engine", None) or "auto"
    if mode == "python":
        return "python", "forced by --engine python"
    feats = detect_features(args, scene)
    unported = sorted(feats - PORTED)
    binary = binary_path()
    if mode == "c":
        if binary is None:
            raise SystemExit(
                "--engine c: miewb-trace binary not found — build it "
                "(cd cengine && ./build.sh) or set MIEWB_CENGINE")
        if unported:
            raise SystemExit(
                "--engine c: this scene uses features not yet ported to "
                "the C engine: %s (use --engine auto/python)"
                % ", ".join(unported))
        return "c", "forced by --engine c"
    if binary is None:
        return "python", "miewb-trace binary not built"
    if unported:
        return "python", "unported: %s" % ", ".join(unported)
    return "c", "all scene features ported"


# ---------------------------------------------------------------------------
# request building
# ---------------------------------------------------------------------------
def _surface_spec(face_rec, geometry_dir):
    """The contract's analytic surface dict (already SI metres); mesh
    faces become {"type": "mesh", "stl": <abs path>} — the C engine reads
    the binary STL directly."""
    surf = face_rec["surface"]
    if surf["type"] == "mesh":
        stl = face_rec.get("mesh_stl") or ""
        path = Path(geometry_dir) / stl if geometry_dir else None
        if path is None or not path.exists():
            raise SystemExit(
                "cengine: mesh face %s: STL %r not found under %s"
                % (face_rec["id"], stl, geometry_dir))
        return {"type": "mesh", "stl": str(path)}
    return surf


def _face_aabb(face_rec, geometry_dir):
    """Conservative world AABB [[lo],[hi]] for the scene TLAS (plan D5):
    union of the trim-polyline bbox and the face's own STL bbox (the STL
    tracks the true surface to chord tolerance), padded; analytic
    full-primitive bounds when no STL exists (synthetic test scenes).
    Returns None for 'unknown' — the C engine then never culls the face
    (always correct, just unaccelerated)."""
    pts = [np.asarray(lp, dtype=float)
           for lp in face_rec.get("trim_polylines_xyz") or [] if len(lp)]
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    if pts:
        allp = np.concatenate(pts)
        lo = np.minimum(lo, allp.min(axis=0))
        hi = np.maximum(hi, allp.max(axis=0))
    stl = face_rec.get("mesh_stl") or ""
    stl_path = Path(geometry_dir) / stl if (geometry_dir and stl) else None
    if stl_path is not None and stl_path.exists():
        from .mesh import read_stl
        tris, _ = read_stl(stl_path)
        if len(tris):
            v = tris.reshape(-1, 3)
            lo = np.minimum(lo, v.min(axis=0))
            hi = np.maximum(hi, v.max(axis=0))
    else:
        surf = face_rec["surface"]
        t = surf["type"]
        if t == "sphere":
            c = np.asarray(surf["center"], dtype=float)
            lo = np.minimum(lo, c - surf["radius"])
            hi = np.maximum(hi, c + surf["radius"])
        elif t == "torus":
            c = np.asarray(surf["center"], dtype=float)
            rr = surf["major_r"] + surf["minor_r"]
            lo = np.minimum(lo, c - rr)
            hi = np.maximum(hi, c + rr)
        elif t != "plane":
            # cylinder/cone/asphere without an STL: rim polylines do not
            # bound the surface bulge in general — no culling
            return None
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return None
    diag = float(np.linalg.norm(hi - lo))
    pad = 1e-3 * diag + 1e-6      # chord-error + epsilon margin
    return [(lo - pad).tolist(), (hi + pad).tolist()]


def _face_entry(fid, face_rec, body_index, det_index, coat_index=-1,
                geometry_dir=None):
    return {
        "id": face_rec["id"],
        "body": int(body_index),
        "surface": _surface_spec(face_rec, geometry_dir),
        "orientation_outward": bool(face_rec["orientation_outward"]),
        "area_m2": float(face_rec["area_m2"] or 0.0),
        "trim": face_rec["trim_polylines_xyz"]
                if face_rec["surface"]["type"] != "mesh" else [],
        "detector": int(det_index),
        "coating": int(coat_index),
        "aabb": _face_aabb(face_rec, geometry_dir),
    }


def _emit_dir_policy(face):
    """Emission-direction policy for the request (sources.py:240-266).
    Planar faces: the collimated direction, sign toward the origin — the
    Python engine derives the sign from the mean of the SAMPLED points;
    the trim-loop centroid is the same decision for any non-pathological
    face (documented deviation, parity-tested).
    Curved faces: per-sample normal-toward-origin in C; flip_all covers
    the whole-face-flipped case (frac_neg == 1.0)."""
    surf = face.surface
    if type(surf).__name__ == "Plane":
        loops = np.concatenate([np.asarray(lp) for lp in face.trim.loops])
        c_uv = loops.mean(axis=0)
        center = surf.origin + c_uv[0] * surf.t1 + c_uv[1] * surf.t2
        sign = 1.0 if float(np.dot(surf.n, -center)) >= 0.0 else -1.0
        return {"emit_policy": "collimated",
                "emit_dir": [float(x) for x in sign * surf.n],
                "flip_all": False}
    # curved: probe the face-centroid normal orientation
    loops = np.concatenate([np.asarray(lp) for lp in face.trim.loops]) \
        if face.trim.mode == "polygon" else None
    if loops is not None:
        from .sources import _uv_to_xyz
        c_uv = loops.mean(axis=0)
        center = _uv_to_xyz(surf, c_uv[0:1], c_uv[1:2])[0]
    else:
        center = getattr(surf, "c", np.zeros(3))
    nrm = surf.normal(center[None, :])[0]
    flip_all = bool(np.dot(nrm, -center) < 0.0)
    return {"emit_policy": "curved", "emit_dir": [0.0, 0.0, 1.0],
            "flip_all": flip_all}


def _track_time_active(args, scene):
    """True when this case tracks pulsed-optics time-domain quantities —
    time products (explicit --time-products OR the pulsed-source auto-enable,
    both folded by resolve_time_products) OR an explicit --gdd-budget. The
    single authority both the request builder and detect_features consult so
    routing and the C trace flag can never disagree (mirrors run_trace's
    track_time = bool(time_products) or args.gdd_budget)."""
    from .detector import resolve_time_products
    return bool(resolve_time_products(args, scene)) or \
        bool(getattr(args, "gdd_budget", False))


def build_request(args, scene, seed, lam_range, grids, out_dir,
                  export_this_seed=False, track_this_seed=False,
                  time_this_seed=False,
                  primary_lo=0, primary_hi=None, gather_skip=False,
                  gather_only=False, gather_input=None):
    """Serialize one seed's trace request. grids: {fid: DetectorGrid} from
    run_trace.build_detectors — the SAME objects later filled with the C
    cubes, so grid geometry is shared by construction."""
    from .sources import wavelength_strata, n_pol_strata, jones_for
    import common

    det_order = list(grids.keys())          # face id -> detector index
    det_index = {fid: i for i, fid in enumerate(det_order)}

    # ---- global wavelength union (plan D1) ----
    lams = []
    src_meta = []
    for sid, (bidx, src) in enumerate(scene.sources):
        strata = wavelength_strata(src, args.nlambda)
        src_meta.append((sid, bidx, src, len(lams), len(strata)))
        lams.extend(float(x) for x in strata)
    lams = np.asarray(lams)

    def n_table(body_index):
        n = scene.medium_index(body_index, lams)
        n = np.asarray(n, dtype=np.complex128)
        if n.ndim == 0:
            n = np.full(len(lams), complex(n))
        return n

    amb = n_table(-1)

    # ---- pulsed-optics P7 time products: group index / GDD-per-length,
    # pre-resolved at every stratum wavelength through the SAME Python
    # material stencil the Python engine uses (scene.medium_group_index /
    # medium_gdd_per_length). The C trace only multiplies these by the
    # segment length, so gopl/gdd_acc match Python bit-for-bit. Resolved
    # only when time products run on THIS seed (seed 0). ----
    def ng_table(body_index):
        return np.asarray(scene.medium_group_index(body_index, lams),
                          dtype=np.float64)

    def gdd_table(body_index):
        return np.asarray(scene.medium_gdd_per_length(body_index, lams),
                          dtype=np.float64)

    # ---- coatings, pre-resolved at every stratum wavelength (D1) ----
    # scene.face_coatings: {fid: coating name}; serialize each distinct
    # coating once and reference by index from the face entries.
    coat_names = sorted(set(scene.face_coatings.values()))
    coat_index = {n: i for i, n in enumerate(coat_names)}
    coatings = []
    for cname in coat_names:
        cspec = scene.coatings[cname]
        if cspec["kind"] == "tmm":
            from .thinfilm import resolve_coating_layers
            layer_n, layer_d = resolve_coating_layers(
                cspec["layers"], scene.matdb, lams)
            coatings.append({
                "kind": "tmm",
                "layer_d": [float(d) for d in layer_d],
                # flattened [layer][lam], matching the C engine's layout
                "layer_n_re": [float(x) for n in layer_n
                               for x in np.real(np.broadcast_to(
                                   n, lams.shape))],
                "layer_n_im": [float(x) for n in layer_n
                               for x in np.imag(np.broadcast_to(
                                   n, lams.shape))],
            })
        else:
            from .optprops import interp_hard
            lam_um = lams * 1e6
            ctx = "coating %r (cengine request)" % cname
            coatings.append({
                "kind": "table",
                "Rs": [float(x) for x in interp_hard(
                    lam_um, cspec["lam_um"], cspec["Rs"], ctx)],
                "Rp": [float(x) for x in interp_hard(
                    lam_um, cspec["lam_um"], cspec["Rp"], ctx)],
                "Ts": [float(x) for x in interp_hard(
                    lam_um, cspec["lam_um"], cspec["Ts"], ctx)],
                "Tp": [float(x) for x in interp_hard(
                    lam_um, cspec["lam_um"], cspec["Tp"], ctx)],
            })

    bodies = []
    for body in scene.bodies:
        if body.role == "source":
            # rays never travel inside source housings (their faces are
            # not in scene.faces); table content is irrelevant but must
            # exist — use ambient
            n = amb
        else:
            n = n_table(body.index)
        entry = {
            "label": body.label,
            "name": body.name,
            "role": body.role,
            "mirror": float(body.mirror),
            "absorbance": float(body.absorbance),
            "n_re": [float(x) for x in np.real(n)],
            "n_im": [float(x) for x in np.imag(n)],
            "filter_alpha": None,
        }
        if body.filter_lam_um is not None:
            alpha = body.filter_alpha(lams)
            entry["filter_alpha"] = [float(x) for x in np.asarray(alpha)]
        if time_this_seed:
            # source housings are never traced through — mirror n_table's
            # ambient fallback so the array is always present & lam-sized
            bidx = -1 if body.role == "source" else body.index
            entry["n_g"] = [float(x) for x in ng_table(bidx)]
            entry["gdd_per_m"] = [float(x) for x in gdd_table(bidx)]
        entry["birefringence"] = None
        if body.birefringent:
            n_o, n_e = scene.uniaxial_indices(body, lams)
            entry["birefringence"] = {
                "axis": [float(x) for x in body.crystal_axis],
                "n_o": [float(x) for x in np.broadcast_to(n_o,
                                                          lams.shape)],
                "n_e": [float(x) for x in np.broadcast_to(n_e,
                                                          lams.shape)],
            }
        entry["polarizer"] = None
        if body.polarizer is not None:
            from .optprops import interp_hard
            pol = scene.polarizers[body.polarizer]
            lam_um = lams * 1e6
            ctx = "polarizer %r (cengine request)" % body.polarizer
            entry["polarizer"] = {
                "type": pol["type"],
                "retardance_waves": float(pol.get("retardance_waves")
                                          or 0.0),
                "axis": [float(x) for x in body.polarizer_axis],
                "T_par": [float(x) for x in interp_hard(
                    lam_um, pol["lam_um"], pol["T_par"], ctx)],
                "T_perp": [float(x) for x in interp_hard(
                    lam_um, pol["lam_um"], pol["T_perp"], ctx)],
            }
        # pulsed-optics P7 tranche 2 NLO bulk effects (saturable / TPA / Kerr).
        # Absent on a plain body. saturable is pre-resolved to SI here through
        # nlo.saturable_alpha0_per_m (the SAME function the Python tracer uses),
        # so the C side only evaluates the intensity-dependent law. SHG
        # (body.shg_spec) and Pockels are NOT serialized: an SHG body emits the
        # unported "nonlinear_shg" token (harmonic-child strata are a later
        # tranche) and a Pockels cell's index shift already rides the
        # pre-resolved birefringence n_o/n_e tables (scene.uniaxial_indices).
        if getattr(body, "saturable_spec", None) is not None:
            from . import nlo as _nlo
            spec = body.saturable_spec
            entry["saturable"] = {
                "alpha0_per_m": float(_nlo.saturable_alpha0_per_m(spec)),
                "I_sat_W_m2": float(spec["I_sat_W_cm2"]) * 1e4,
            }
        if getattr(body, "tpa_beta", 0.0):
            entry["tpa_beta_si"] = float(body.tpa_beta) * 1e-11
        if getattr(body, "kerr_n2_value", None):
            entry["kerr_n2"] = float(body.kerr_n2_value)
        bodies.append(entry)

    # ---- per-face roughness / ABg scatter / grating tables (phase E) ----
    from .roughness import slope_from_sigma_lcorr
    roughs = []
    rough_of_fid = {}
    for fid, rspec in scene.roughness.items():
        rough_of_fid[fid] = len(roughs)
        sigma_m = float(rspec["sigma_nm"]) * 1e-9
        lcorr_m = float(rspec["lcorr_um"]) * 1e-6
        roughs.append({"sigma_m": sigma_m,
                       "slope": float(slope_from_sigma_lcorr(sigma_m,
                                                             lcorr_m))})
    scatters = []
    scat_of_fid = {}
    for fid, sspec in scene.scatter.items():
        scat_of_fid[fid] = len(scatters)
        scatters.append({"A": float(sspec["A"]), "B": float(sspec["B"]),
                         "tis_cap": (float(sspec["tis_cap"])
                                     if sspec.get("tis_cap") is not None
                                     else None)})
    gratings = []
    grat_of_fid = {}
    for fid, gspec in scene.gratings.items():
        grat_of_fid[fid] = len(gratings)
        lo, hi = gspec["orders"]
        orders = list(range(lo, hi + 1))
        face = scene.faces[fid]
        body = scene.body_of_face(fid)
        # groove base vector (grating.groove_vector's 'u'/'v'/explicit)
        gv = gspec["groove"]
        if gv == "u":
            base = face.surface.t1
        elif gv == "v":
            base = face.surface.t2
        else:
            base = np.array([float(x) for x in gv.split(",")])
        # far-side index exactly like apply_to_batch (grating.py:414-417)
        if body.material not in (None, "detector"):
            n2 = np.real(scene.matdb.get(body.material).n_complex(lams))
            n2 = np.broadcast_to(n2, lams.shape)
        else:
            n2 = np.ones(len(lams))
        entry = {
            "lo": int(lo), "hi": int(hi),
            "lines_per_mm": float(gspec["lines_per_mm"]),
            "groove_base": [float(x) for x in base],
            "n2": [float(x) for x in n2],
        }
        if (gspec.get("model") or "lamellar") == "bragg_kogelnik":
            p = gspec.get("params", {})
            entry["model"] = "kogelnik"
            entry["thickness_m"] = float(p["thickness_um"]) * 1e-6
            entry["dn"] = float(p["dn"])
            entry["slant_rad"] = float(
                np.deg2rad(float(p.get("slant_deg", 0.0))))
        else:
            tbl = gspec.get("table")
            if isinstance(tbl, dict) and tbl.get("schema") == "v2":
                # unreachable under normal routing (grating_table_v2 forces
                # Python); guard so a future dispatch bug can't bake a v2
                # table's phase/theta/phi structure into a lambda-only table.
                raise NotImplementedError(
                    "v2 RCWA grating table is not C-portable (complex "
                    "amplitude, theta/phi axes); scene must route to Python "
                    "via the grating_table_v2 feature token")
            # lambda-only models: pre-resolve through the SAME Python
            # code (order_efficiencies) at every stratum wavelength
            from .grating import order_efficiencies
            eta_s, eta_p = order_efficiencies(
                gspec, lams, np.ones(len(lams)), orders)
            entry["model"] = "fixed"
            # [order][lam] flattening matching the C layout
            entry["eta_s"] = [float(x) for x in eta_s.T.ravel()]
            entry["eta_p"] = [float(x) for x in eta_p.T.ravel()]
        gratings.append(entry)

    geometry_dir = Path(args.model_json).parent
    faces = []
    for fid in range(len(scene.faces)):
        fe = _face_entry(fid, scene.face_records[fid],
                         scene.face_body[fid], det_index.get(fid, -1),
                         coat_index.get(scene.face_coatings.get(fid), -1),
                         geometry_dir=geometry_dir)
        fe["rough"] = rough_of_fid.get(fid, -1)
        fe["scatter"] = scat_of_fid.get(fid, -1)
        fe["grating"] = grat_of_fid.get(fid, -1)
        faces.append(fe)

    # viz caps (int or {sid: int} — run_trace.compute_viz_caps)
    from run_trace import compute_viz_caps
    viz_caps = compute_viz_caps(scene, args, None)

    # emit-face records come from the model contract (source bodies are
    # not in scene.face_records)
    model = common.load_model(args.model_json)

    sources = []
    for sid, bidx, src, lam_off, n_strata in src_meta:
        body = scene.bodies[bidx]
        face = scene.emit_faces[bidx]
        pol = src.get("polarization") or {"kind": "unpolarized"}
        n_pol = n_pol_strata(src)
        jones = []
        for ps in range(n_pol):
            js, jp = jones_for(pol, ps)
            jones.append([float(np.real(js)), float(np.imag(js)),
                          float(np.real(jp)), float(np.imag(jp))])
        cap = viz_caps.get(sid, 500) if isinstance(viz_caps, dict) \
            else int(viz_caps)
        rec = next(b for b in model["bodies"] if b["name"] == body.name)
        emit_rec = next(f for f in rec["faces"]
                        if f["id"] == src["emit_face"])
        entry = {
            "label": body.label,
            "body_index": int(bidx),
            # power_mW None/0.0 is the extractor's "unset" sentinel for
            # pulse_energy-XOR-power sources (CLAUDE.md P3/P6): the C
            # engine bills the derived AVERAGE power exactly like the
            # Python engine. Exposed when tranche 1 flipped time-product
            # scenes to C routing (pre-existing hole in build_request).
            "power_W": (float(src["power_mW"]) * 1e-3
                        if src.get("power_mW") not in (None, 0.0)
                        else float((src.get("pulse") or {})
                                   .get("avg_power_W") or 0.0)),
            # pulsed-optics P7 tranche 2: pulse peak/avg power ratio (kappa)
            # multiplies every per-ray local intensity (nlo.ray_intensity);
            # 1.0 for CW / non-pulsed sources.
            # .get default doesn't cover an explicit kappa=None (pulse dict
            # present, duration unset -> no peak/avg ratio derivable): CW law
            "kappa_pulse": float((src.get("pulse") or {}).get("kappa")
                                 or 1.0),
            "coherent": bool(src.get("coherent", False)),
            "lam_offset": int(lam_off),
            "n_strata": int(n_strata),
            "n_pol": int(n_pol),
            "jones": jones,
            "viz_cap": int(cap),
            "emit_face": _face_entry(-1, emit_rec, bidx, -1),
        }
        entry.update(_emit_dir_policy(face))
        # samples-instruments round: extended image-emitting source. The scene
        # resolved the registry bitmap into src["_image_gray"] at build; the C
        # sampler needs the Vose alias table (built by the SAME
        # sources._build_alias_table so there is exactly ONE implementation),
        # the image W/H, the emission cone half-angle (0.0 = Lambertian
        # sentinel), and the face-UV bounding rectangle the bitmap fills
        # (u_lo/u_hi/v_lo/v_hi over the trim loops, exactly as
        # sources._sample_image_points computes them). The emit_dir policy above
        # already resolved the SIGNED emit normal the image directions fan
        # about. Requires a planar emit face (Python raises the same).
        img = src.get("_image_gray")
        if img is not None:
            if type(face.surface).__name__ != "Plane":
                raise SystemExit(
                    "cengine: image source %s requires a planar emit face "
                    "(got %s)" % (body.label,
                                  type(face.surface).__name__))
            from .sources import _build_alias_table
            img = np.asarray(img, dtype=np.float64)
            H, W = img.shape
            prob_table, alias_idx = _build_alias_table(img.ravel())
            loops = face.trim.loops
            allu = np.concatenate([np.asarray(lp)[:, 0] for lp in loops])
            allv = np.concatenate([np.asarray(lp)[:, 1] for lp in loops])
            entry["image"] = {
                "W": int(W), "H": int(H),
                "cone_deg": float(src.get("image_cone_deg") or 0.0),
                "u_lo": float(allu.min()), "u_hi": float(allu.max()),
                "v_lo": float(allv.min()), "v_hi": float(allv.max()),
                "prob": [float(x) for x in prob_table],
                "alias": [int(x) for x in alias_idx],
            }
        # per-(stratum, pol) gather normalization areas (compute_sample_area)
        from run_trace import compute_sample_area
        sa = compute_sample_area(scene, args)
        entry["sample_area"] = [
            float(sa[(sid, st, ps)])
            for st in range(n_strata) for ps in range(n_pol)]
        # pulsed-optics P7 SPM chirp: per-stratum birth-time offset [s]
        # (sources.install_spm sets src["_stratum_t0"]; absent otherwise).
        # Only carried when time products run on this seed.
        if time_this_seed:
            t0 = src.get("_stratum_t0")
            if t0 is not None:
                entry["stratum_t0"] = [float(x) for x in np.asarray(t0)]
        sources.append(entry)

    detectors = []
    for i, fid in enumerate(det_order):
        g = grids[fid]
        detectors.append({
            "label": g.label,
            "face_id": int(fid),
            "xhat": [float(x) for x in g.xhat],
            "yhat": [float(x) for x in g.yhat],
            "normal": [float(x) for x in g.normal],
            "x_lo": float(g.x_lo), "y_lo": float(g.y_lo),
            "pixel_m": float(g.pixel_m),
            "W": int(g.W), "H": int(g.H),
            "spectral_bins": int(g.spectral_bins),
            "lam_lo_m": float(g.lam_lo), "lam_hi_m": float(g.lam_hi),
        })

    from run_trace import resolve_workers
    return {
        "schema": 1,
        # P3 interaction registry (REGISTRY.md §2.2): the detected feature
        # tokens travel with the request so the C engine can hard-error on
        # any token it has no implementation for (the belt-and-suspenders
        # backstop to choose_engine's routing — never a silent skip).
        "features": sorted(detect_features(args, scene)),
        "out_dir": str(out_dir),
        "params": {
            "max_reflections": int(args.max_reflections),
            "power_floor": float(args.power_floor),
            "rays": int(args.rays),
            # P1 chunked-run contract: this invocation traces primaries
            # [lo,hi); p_ray stays power_W/rays so chunks sum to one run.
            "primary_lo": int(primary_lo),
            "primary_hi": int(args.rays if primary_hi is None
                              else primary_hi),
            "gather_skip": bool(gather_skip),
            # P1 final stage: no tracing — load the merged sample dump +
            # accumulator snapshots from gather_input, run the in-binary
            # gather (tiled kernel; gather.mode=exact still honored)
            "gather_only": bool(gather_only),
            "gather_input": str(gather_input) if gather_input else None,
            "seed": int(seed),
            "batch_size": 1 << 20,
            "threads": 0 if args.workers == "auto"
                       else resolve_workers(args.workers),
            "mesh_flat_normals": bool(args.mesh_flat_normals),
            "export_rays": bool(export_this_seed),
            "importance_aim": bool(getattr(args, "importance_aim",
                                           False)),
            # P7 ray-differentials port: seed + transport the Igehy ray
            # differentials and size the coherent gather's per-sample dA.
            "ray_differentials": bool(getattr(args, "ray_differentials",
                                              False)),
            "track_history": bool(track_this_seed),
            # pulsed-optics P7: group-delay accumulators + the per-body
            # power-weighted bulk-path tally (the GDD-budget input). Active
            # whenever the case runs time products OR --gdd-budget, on EVERY
            # chunk/seed (path_tally is a linear tally; the merge sums it and
            # build_gdd_budget consumes seed 0). resolve_time_products folds
            # in both the explicit flag and the pulsed-source auto-enable, so
            # this can never disagree with run_trace's activation.
            "track_time": bool(_track_time_active(args, scene)),
            # pulsed-optics P7 time products: also accumulate the gopl/gdd
            # group-delay ray slots + record per-detector arrival records.
            # Seed 0 only (like the Python engine's time_rec), so the request
            # carries the n_g/gdd tables + stratum_t0 only then.
            "time_products": bool(time_this_seed),
            "linear_scan": bool(os.environ.get("MIEWB_CENGINE_LINEAR")
                                == "1"),
        },
        "lams_m": [float(x) for x in lams],
        "ambient_n_re": [float(x) for x in np.real(amb)],
        "ambient_n_im": [float(x) for x in np.imag(amb)],
        **({"ambient_n_g": [float(x) for x in ng_table(-1)],
            "ambient_gdd_per_m": [float(x) for x in gdd_table(-1)]}
           if time_this_seed else {}),
        "bodies": bodies,
        "faces": faces,
        "sources": sources,
        "detectors": detectors,
        "coatings": coatings,
        "roughs": roughs,
        "scatters": scatters,
        "gratings": gratings,
        "particles": _particles_block(args, scene, lams),
        "sample_media": _sample_media_block(args, scene, lams),
        "gather": {
            # map the Python gather's --backend to the C engine's kernels
            "backend": {"auto": "auto", "torch": "cuda",
                        "numpy": "cpu"}.get(args.backend or "auto",
                                            "auto"),
            "min_eff_samples": float(args.min_eff_samples),
            "enforce_gate": not args.no_gather_gate,
            "save_fields": bool(args.save_fields),
            "occlusion": bool(args.gather_occlusion),
            "occlusion_tile": 16,
            # P1 tile-factorized kernel is the default; --gather-exact
            # selects the plain fp64 reference kernel (bit-exact anchor)
            "mode": ("exact" if getattr(args, "gather_exact", False)
                     else "tiled"),
            # EXPERIMENTAL NUFFT angular-spectrum route (cuFINUFFT); OFF by
            # default (opt-in via --gather-nufft). The per-key runtime gate
            # is the real switch; see cengine/src/gather_nufft.c.
            "nufft": bool(getattr(args, "gather_nufft", False)),
        },
    }


def _sample_body_mode(args, scene, body):
    """Effective mode ('continuum' | 'explicit') of a `sample`-tagged body,
    resolved EXACTLY as raytracer.particles.BodyParticleMedium does so the
    routing decision matches the medium the trace actually builds:
      * a registry `mode` of 'continuum' / 'explicit' forces that mode;
      * 'auto' compares the phi/tau-derived particle count to the threshold
        (count <= threshold -> explicit, like ParticleCloud), host = the body
        material (real solvent contrast/density).
    Any resolution failure returns 'explicit' — the conservative choice: the
    Python engine then runs and raises the real error rather than the C engine
    silently mis-tracing. Only the continuum verdict routes to C."""
    import common
    name = getattr(body, "sample", None)
    reg = scene.optprops.samples if scene.optprops is not None else {}
    row = reg.get(name)
    if row is None:
        return "explicit"
    mode = row.get("mode", "auto")
    if mode in ("continuum", "explicit"):
        return mode
    try:
        from .mie import (LogNormalDistribution, MieEvaluator,
                          number_density)
        from .particles import resolve_tau_phi
        from .sources import wavelength_strata
        if body.bbox_m is None or body.material in (None, "", "none",
                                                    "detector"):
            return "explicit"
        thr = args.particle_threshold if args.particle_threshold is not None \
            else common.DEFAULTS["particle_threshold"]
        lo, hi = body.bbox_m
        box_size = [float(x) for x in (hi - lo)]
        dist = LogNormalDistribution(
            median_r=row["median_um"] * 1e-6 / 2.0, gsd=row["gsd"])
        mat_p = scene.matdb.get(row["particle_material"])
        host = scene.matdb.get(body.material)
        rho_h = host.density if host.density > 0 else 1.204
        phi = row.get("phi")
        if phi is None:
            lam_list = sorted({
                float(l) for _, src in scene.sources
                for l in wavelength_strata(src, args.nlambda)})
            phi, _info = resolve_tau_phi(
                row["tau"], float(box_size[0]),
                MieEvaluator(mat_p, host), dist,
                mat_p.density, rho_h, lam_list)
        N, _ = number_density(phi, mat_p.density, rho_h, dist)
        count = N * float(np.prod(box_size))
        return "explicit" if count <= thr else "continuum"
    except Exception:
        return "explicit"


def _medium_tables(cloud, lams, n_u):
    """Continuum tables at the stratum wavelengths (plan D1) for ONE medium
    (a box cloud or a body-bound sample medium — both are ParticleCloud
    subclasses): mu_ext / albedo per lam (S(q)-corrected in-place by
    EnsembleTables when a structure factor is set), the radius-node CDF, and
    the per-(lam, node) INVERSE direction CDF. Returns
    (n_quad, mu_ext, albedo, radius_cdf, inv_phase) with the flat [lam],
    [lam][node], [lam][node][u] layouts the C ParticleC reader expects.

    S(q) note (samples-instruments): when a structure factor is present the
    Python sampler (mie.sample_direction) draws the cosine from a SINGLE
    size-averaged, S(q)-corrected inverse CDF (`sq_cdf`/`sq_mu`) that ignores
    the per-ray radius node. That reshaping lives ENTIRELY in the serialized
    table: inv_phase is built by REPLICATING that one corrected CDF across
    every radius node, so the C kernel (which still draws a node, then reads
    inv_phase[node]) reproduces the S(q) direction law with no C changes.
    mu_ext / albedo are already the S(q)-corrected values in `_nearest`."""
    u_grid = np.linspace(0.0, 1.0, n_u)
    radii = cloud.tables.radii
    has_sq = cloud.tables._sq is not None
    mu_ext, albedo, radius_cdf, inv_phase = [], [], [], []
    for lam in lams:
        t = cloud.tables._nearest(float(lam))
        mu_ext.append(float(t["mu_ext"]))
        albedo.append(float(t["albedo"]))
        radius_cdf.extend(
            float(x) for x in np.cumsum(t["radius_weights"]))
        if has_sq:
            inv = [float(x)
                   for x in np.interp(u_grid, t["sq_cdf"], t["sq_mu"])]
            for _rv in radii:
                inv_phase.extend(inv)
        else:
            for rv in radii:
                mu_g, _p, cdf = cloud.evaluator.phase_function(
                    float(rv), float(lam))
                inv_phase.extend(
                    float(x) for x in np.interp(u_grid, cdf, mu_g))
    return int(len(radii)), mu_ext, albedo, radius_cdf, inv_phase


def _particles_block(args, scene, lams):
    """CLI --particles world-box continuum tables at the stratum wavelengths
    (D1). Byte-identical to the pre-samples engine (sq is always None on the
    CLI path)."""
    if not args.particles:
        return None
    import common
    from .particles import ParticleCloud
    spec = common.parse_particles_spec(args.particles)
    thr = args.particle_threshold if args.particle_threshold is not None \
        else common.DEFAULTS["particle_threshold"]
    cloud = ParticleCloud(spec, scene, threshold=thr,
                          seed=int(args.seed0),
                          lam_list=[float(x) for x in lams])
    if cloud.mode != "continuum":
        raise SystemExit(
            "cengine: explicit-mode particles reached the C request "
            "builder — feature routing bug")
    n_u = 512
    n_quad, mu_ext, albedo, radius_cdf, inv_phase = \
        _medium_tables(cloud, lams, n_u)
    return {
        "box_lo": [float(x) for x in cloud.lo],
        "box_hi": [float(x) for x in cloud.hi],
        "n_quad": n_quad,
        "n_u": n_u,
        "mu_ext": mu_ext,
        "albedo": albedo,
        "radius_cdf": radius_cdf,
        "inv_phase": inv_phase,
    }


def _sample_media_block(args, scene, lams):
    """One continuum-medium block per `sample`-tagged body (samples-
    instruments round). Mirrors run_trace.build_particle_media /
    particles.build_body_sample_media, but only for CONTINUUM-mode bodies —
    routing (detect_features) guarantees an explicit-mode body forces Python,
    so an explicit medium reaching here is a routing bug (hard error, never a
    silent wrong answer). Each block carries the host body index (the C
    medium-stack region test), the `sample:<label>` ledger key, and the
    per-medium continuum tables (host solvent effects pre-baked)."""
    import common
    from .particles import BodyParticleMedium
    reg = scene.optprops.samples if scene.optprops is not None else {}
    thr = args.particle_threshold if args.particle_threshold is not None \
        else common.DEFAULTS["particle_threshold"]
    n_u = 512
    blocks = []
    for body in scene.bodies:
        name = getattr(body, "sample", None)
        if not name:
            continue
        if _sample_body_mode(args, scene, body) != "continuum":
            raise SystemExit(
                "cengine: explicit-mode sample body %s reached the C request "
                "builder — feature routing bug" % body.label)
        if name not in reg:
            raise SystemExit(
                "cengine: body %s sample %r not in the samples registry"
                % (body.label, name))
        medium = BodyParticleMedium(
            name, reg[name], body, scene, threshold=thr,
            seed=int(args.seed0), lam_list=[float(x) for x in lams])
        if medium.mode != "continuum":
            raise SystemExit(
                "cengine: sample body %s resolved to %s in the C request "
                "builder — routing bug" % (body.label, medium.mode))
        n_quad, mu_ext, albedo, radius_cdf, inv_phase = \
            _medium_tables(medium, lams, n_u)
        blocks.append({
            "body_index": int(body.index),
            "label": "sample:%s" % body.label,
            "n_quad": n_quad,
            "n_u": n_u,
            "mu_ext": mu_ext,
            "albedo": albedo,
            "radius_cdf": radius_cdf,
            "inv_phase": inv_phase,
        })
    return blocks or None


# ---------------------------------------------------------------------------
# run + convert
# ---------------------------------------------------------------------------
def _run_binary(request_path, log_level=None):
    """Spawn miewb-trace; stdout (incl. @MIEWB lines) is inherited so the
    pipeline's progress parsing keeps working. Returns the exit code."""
    cmd = [str(binary_path()), "--config", str(request_path)]
    if log_level:
        cmd += ["--log-level", log_level]
    print("[trace] cengine: %s" % " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    return proc.returncode


# ---------------------------------------------------------------------------
# P3 persistent worker (REGISTRY.md §6). One miewb-trace --serve process per
# run_trace process amortizes the spawn + CUDA context init + device-buffer
# pool over every chunk trace AND the final gather of a case (V chunks x S
# seeds pay ONE init). The client feeds request-file paths on the worker's
# stdin and reads its `@MIEWB-WORKER {json}` protocol lines from stdout with
# a mid-line prefix scan (the fcclient discipline: the worker emits a LEADING
# newline so engine noise cannot glue onto the response). @MIEWB progress
# lines the worker emits are forwarded to our stdout so run_pipeline's
# progress parsing is unchanged. A dead/malformed/timed-out worker raises
# WorkerError; the caller kills it and falls back to one-shot _run_binary.
#
# MIEWB_CENGINE_ONESHOT=1 forces the classic per-invocation path (the escape
# hatch documented in cengine/README.md).
# ---------------------------------------------------------------------------
_WORKER_PREFIX = "@MIEWB-WORKER "
# per-request wall-clock ceiling; generous (a chunk of a big case can be
# slow) but bounded so a hung worker can't wedge the run. Override via env.
_WORKER_TIMEOUT_S = float(os.environ.get("MIEWB_WORKER_TIMEOUT", "1800"))


class WorkerError(RuntimeError):
    """The persistent worker died, timed out, or emitted a malformed line."""


class Worker:
    """A single miewb-trace --serve child. run(request_path) feeds the path
    and returns the request's exit code (rc). Not thread-safe: one request in
    flight at a time (run_c_case drives it sequentially)."""

    def __init__(self, log_level=None, timeout=_WORKER_TIMEOUT_S):
        import queue
        import threading
        self.timeout = timeout
        cmd = [str(binary_path()), "--serve"]
        if log_level:
            cmd += ["--log-level", log_level]
        print("[trace] cengine: %s (persistent worker)" % " ".join(cmd),
              flush=True)
        # stderr inherited (cengine.log-style noise + crash backtraces show);
        # only stdout is piped (protocol + forwarded @MIEWB progress).
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        self._q = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self):
        """Feed the worker's stdout lines into a queue; None marks EOF."""
        try:
            for line in self.proc.stdout:
                self._q.put(line)
        finally:
            self._q.put(None)

    def run(self, request_path):
        import queue
        req = str(request_path)
        try:
            self.proc.stdin.write(req + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise WorkerError("worker stdin closed: %s" % exc)
        deadline = time.time() + self.timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise WorkerError("worker timed out after %.0fs on %s"
                                  % (self.timeout, req))
            try:
                line = self._q.get(timeout=remaining)
            except queue.Empty:
                raise WorkerError("worker timed out after %.0fs on %s"
                                  % (self.timeout, req))
            if line is None:
                raise WorkerError("worker exited (EOF) before responding "
                                  "to %s" % req)
            idx = line.find(_WORKER_PREFIX)
            if idx < 0:
                # engine noise / @MIEWB progress — forward it verbatim so the
                # pipeline's progress parser still sees it.
                sys.stdout.write(line)
                sys.stdout.flush()
                continue
            payload = line[idx + len(_WORKER_PREFIX):].strip()
            try:
                resp = json.loads(payload)
                return int(resp["rc"])
            except (ValueError, KeyError, TypeError) as exc:
                raise WorkerError("malformed worker response %r: %s"
                                  % (payload, exc))

    def kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass
        self.close()

    def close(self):
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()        # EOF -> clean worker exit
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# P1 chunked-run contract: checkpoint / resume / additive extension
#
# The C engine traces primaries in CHUNKS (gather_skip mode: each invocation
# dumps its coherent samples + incoherent cube + ledger instead of gathering).
# The Python driver merges the chunk payloads into DetectorGrid accumulators,
# sorts the samples canonically, writes the MERGED dump back to disk, and
# invokes the binary once more in gather_only mode: it loads the dump +
# accumulator snapshots and runs the normal in-binary gather — the
# tile-factorized kernel (~75x the Python torch gather; --gather-exact
# still selects the plain fp64 reference kernel). checkpoint.json makes an
# interrupted trace resumable and a completed run extendable, and the
# per-chunk sample dumps on disk ARE the durable accumulator (rebuilt into
# grids on resume).
#
# Bit-identity: coherent samples are sorted by the UNIQUE (ray_key,event_ctr)
# key before the gather, so a merged N-chunk sample set is byte-identical to a
# 1-chunk set at the same target (and a resumed run == an uninterrupted one).
# --extend rescales the already-traced chunks by old_target/new_target (an
# extra fp multiply absent from a fresh run), so extend-vs-fresh is
# statistically equivalent, not bit-equal (documented in test + report).
# ---------------------------------------------------------------------------
import math

CHECKPOINT_SCHEMA = 1
# args whose value changes the traced physics or detector geometry — a
# resume/extend must match them. `rays` is deliberately absent (--extend
# raises it); gather-only knobs (backend/workers/min_eff_samples/
# no_gather_gate) may differ between resume invocations.
_HASH_ARG_KEYS = (
    "nlambda", "resolution", "spectral_bins", "max_reflections",
    "power_floor", "seeds", "seed0", "mesh_flat_normals", "strict_analytic",
    "ray_differentials", "importance_aim", "temperature", "grating", "rough",
    "particles", "particle_threshold", "suppress_body", "source_face",
    "detector_face", "no_pol_scatter",
)
_PROP_EXTS = (".miemat", ".mienk", ".miecoat", ".miepol", ".miefilt",
              ".miegrat", ".miebrf", ".miedet", ".miesrc", ".mietab", ".csv")
# the standard sample-record fields render_coherent.merged_samples consumes
_SAMPLE_FIELDS = ("pos", "dir", "s_hat", "Es", "Ep", "lam", "opl", "power",
                  "scattered", "dA")


def _align_stride(scene, args):
    """LCM over sources of n_strata*n_pol — the alignment every chunk
    boundary (cursor) must respect so [0,cursor) has equal per-key counts
    and the gather normalization (cursor / n_strata*n_pol) is exact."""
    from raytracer.sources import wavelength_strata, n_pol_strata
    stride = 1
    for _, src in scene.sources:
        s = max(len(wavelength_strata(src, args.nlambda))
                * n_pol_strata(src), 1)
        stride = stride * s // math.gcd(stride, s)
    return stride


def _default_chunk_rays(total, stride):
    """Whole run in ONE chunk when small (<=2e5: zero per-chunk overhead);
    else ~8 chunks aligned up to the stride. MIEWB_CHUNK_RAYS overrides
    (the gate tests force chunk counts through it)."""
    env = os.environ.get("MIEWB_CHUNK_RAYS")
    if env:
        return max(int(env), 1)
    if total <= 200000:
        return total
    step = max(total // 8, 100000)
    return ((step + stride - 1) // stride) * stride


def _chunk_step(total, stride, chunk_rays):
    """Aligned chunk width (multiple of stride, >= stride)."""
    if chunk_rays >= total:
        return total
    return max(((chunk_rays + stride - 1) // stride) * stride, stride)


def scene_hash(args, scene):
    """sha256 over model.json bytes + the (relpath,size,mtime) of the optical
    property files + the physics-relevant args subset. A --resume/--extend
    that no longer matches refuses rather than silently mixing incompatible
    samples."""
    import hashlib
    import common
    h = hashlib.sha256()
    with open(args.model_json, "rb") as fh:
        h.update(fh.read())
    root = Path(args.optical_properties) if args.optical_properties \
        else Path(common.OPTPROPS_DIR)
    files = []
    if root.is_dir():
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in _PROP_EXTS:
                st = p.stat()
                files.append((str(p.relative_to(root)), st.st_size,
                              int(st.st_mtime)))
    h.update(json.dumps(files, sort_keys=True).encode())
    subset = {k: getattr(args, k, None) for k in _HASH_ARG_KEYS}
    h.update(json.dumps(subset, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _merge_chunk(grids, det_order, chunk_dir, scale):
    """Fold one trace-only chunk's payload into the in-memory accumulator
    grids: incoherent cube (+= scale), per-key incoherent tallies (+= scale,
    counts += raw), and the coherent sample records (appended, power*scale +
    amplitude*sqrt(scale) — the additive-extension renormalization). scale is
    EXACTLY 1.0 for a same-target run (fp identity), so a chunked/ resumed
    merge is bit-identical to a single trace."""
    detected = json.loads((chunk_dir / "detected.json").read_text())
    sqrt_scale = math.sqrt(scale)
    for i, fid in enumerate(det_order):
        g = grids[fid]
        inc = np.load(chunk_dir / ("det_%d_inc.npy" % i))
        if scale == 1.0:
            g.inc += inc
        else:
            g.inc += inc * scale
        for skey, entry in detected.get(g.label, {}).items():
            key = tuple(int(x) for x in skey.split("/"))
            if "incoherent_W" in entry:
                g.detected_incoherent[key] = (
                    g.detected_incoherent.get(key, 0.0)
                    + float(entry["incoherent_W"]) * scale)
                g.detected_incoherent_n[key] = (
                    g.detected_incoherent_n.get(key, 0) + int(entry["n"]))
    manifest = json.loads((chunk_dir / "gkeys.json").read_text())
    for i, fid in enumerate(det_order):
        g = grids[fid]
        for src, ls, ps, n in manifest.get(str(i), []):
            if n == 0:
                continue
            key = (int(src), int(ls), int(ps))
            base = str(chunk_dir / ("gk_%d_%d_%d_%d_" % (i, src, ls, ps)))
            Es = np.load(base + "Es.npy")
            Ep = np.load(base + "Ep.npy")
            power = np.load(base + "power.npy")
            if scale != 1.0:
                Es = Es * sqrt_scale
                Ep = Ep * sqrt_scale
                power = power * scale
            rec = {
                "pos": np.load(base + "pos.npy"),
                "dir": np.load(base + "dir.npy"),
                "s_hat": np.load(base + "shat.npy"),
                "Es": Es, "Ep": Ep,
                "lam": np.load(base + "lam.npy"),
                "opl": np.load(base + "opl.npy"),
                "power": power,
                "scattered": np.load(base + "scat.npy").astype(bool),
                # --ray-differentials per-sample wavefront patch area (NaN
                # where the differential was lost). A GEOMETRIC quantity, so
                # unscaled by the ray-count weighting; NaN when the run had no
                # differentials (the C dump wrote all-NaN then).
                "dA": np.load(base + "dA.npy"),
                "_ray_key": np.load(base + "key.npy"),
                "_evt": np.load(base + "evt.npy"),
            }
            g.samples.setdefault(key, []).append(rec)


def _finalize_sorted_samples(grids):
    """Collapse each detector's per-chunk sample records into ONE record with
    the samples in the canonical (ray_key,event_ctr) order — a total order
    (each is unique), so the merged multi-chunk order is byte-identical to a
    single trace's, making the final gather bit-reproducible. The _ray_key/
    _evt columns are KEPT: the C gather's cross-estimator groups by
    (ray_key & 3), and the merged dump round-trips them."""
    for g in grids.values():
        for key, recs in list(g.samples.items()):
            rk = np.concatenate([r["_ray_key"] for r in recs])
            ev = np.concatenate([r["_evt"] for r in recs])
            order = np.lexsort((ev, rk))       # primary key rk, tiebreak ev
            rec = {f: np.concatenate([r[f] for r in recs])[order]
                   for f in _SAMPLE_FIELDS}
            rec["_ray_key"] = rk[order]
            rec["_evt"] = ev[order]
            g.samples[key] = [rec]
            g.detected_geometric[key] = float(np.sum(rec["power"]))


def _tally_dims(scene, args):
    """(n_sources, max_strata, max_pol) — the C engine's flat detected-tally
    dimensions (request.c mirrors this exactly)."""
    from raytracer.sources import wavelength_strata, n_pol_strata
    n_src = len(scene.sources)
    max_strata = max((len(wavelength_strata(src, args.nlambda))
                      for _, src in scene.sources), default=1)
    max_pol = max((n_pol_strata(src) for _, src in scene.sources),
                  default=1)
    return n_src, max_strata, max_pol


def _write_merged_dump(scene, args, grids, det_order, dump_dir):
    """Serialize the canonically sorted merged sample sets + accumulator
    snapshots for the C engine's gather_only stage (det_load_gather_state's
    exact input layout: gkeys.json + gk_* SoA arrays + acc_<i>_inc /
    acc_<i>_tinc_W / acc_<i>_tinc_n snapshots)."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    n_src, max_strata, max_pol = _tally_dims(scene, args)
    manifest = {}
    for i, fid in enumerate(det_order):
        g = grids[fid]
        np.save(dump_dir / ("acc_%d_inc.npy" % i),
                np.ascontiguousarray(g.inc, dtype=np.float64))
        tw = np.zeros(n_src * max_strata * max_pol)
        tn = np.zeros(n_src * max_strata * max_pol, dtype=np.int64)
        for (s, l, p), v in g.detected_incoherent.items():
            tw[(s * max_strata + l) * max_pol + p] = v
        for (s, l, p), v in g.detected_incoherent_n.items():
            tn[(s * max_strata + l) * max_pol + p] = v
        np.save(dump_dir / ("acc_%d_tinc_W.npy" % i), tw)
        np.save(dump_dir / ("acc_%d_tinc_n.npy" % i), tn)
        entries = []
        for (s, l, p), recs in sorted(g.samples.items()):
            rec = recs[0]
            n = int(len(rec["power"]))
            entries.append([int(s), int(l), int(p), n])
            if n == 0:
                continue
            base = str(dump_dir / ("gk_%d_%d_%d_%d_" % (i, s, l, p)))
            np.save(base + "pos.npy", np.ascontiguousarray(
                rec["pos"], dtype=np.float64))
            np.save(base + "dir.npy", np.ascontiguousarray(
                rec["dir"], dtype=np.float64))
            np.save(base + "shat.npy", np.ascontiguousarray(
                rec["s_hat"], dtype=np.float64))
            np.save(base + "Es.npy", np.ascontiguousarray(
                rec["Es"], dtype=np.complex128))
            np.save(base + "Ep.npy", np.ascontiguousarray(
                rec["Ep"], dtype=np.complex128))
            np.save(base + "lam.npy", np.ascontiguousarray(
                rec["lam"], dtype=np.float64))
            np.save(base + "opl.npy", np.ascontiguousarray(
                rec["opl"], dtype=np.float64))
            np.save(base + "power.npy", np.ascontiguousarray(
                rec["power"], dtype=np.float64))
            # --ray-differentials per-sample patch area (NaN where lost); the
            # gather_only stage (det_load_gather_state) reads gk_..._dA.npy.
            np.save(base + "dA.npy", np.ascontiguousarray(
                rec["dA"], dtype=np.float64))
            np.save(base + "scat.npy", np.ascontiguousarray(
                rec["scattered"], dtype=np.uint8))
            np.save(base + "key.npy", np.ascontiguousarray(
                rec["_ray_key"], dtype=np.uint64))
            np.save(base + "evt.npy", np.ascontiguousarray(
                rec["_evt"], dtype=np.uint32))
        manifest[str(i)] = entries
    (dump_dir / "gkeys.json").write_text(json.dumps(manifest))


def _merge_ledger(reports):
    """Merge per-chunk C ledger.json reports (each paired with its extend
    scale) into one per-seed report. Numeric leaves sum (scaled); per-source
    closure_error is recomputed from the summed emitted/buckets."""
    out = {"sources": {}, "by_surface_W": {}, "by_body_W": {},
           "element_flux_W": {}, "detected_W": {}, "closure_gate": 1e-3}
    # pulsed-optics P7: the per-body bulk-path tally rides along as a plain
    # label->W*m numeric dict (present only under time tracking). It is a
    # linear tally, so it sums (scaled) exactly like by_body_W.
    has_path = any("path_tally_Wm" in rep for rep, _ in reports)
    if has_path:
        out["path_tally_Wm"] = {}
    for rep, scale in reports:
        for label, sd in rep.get("sources", {}).items():
            dst = out["sources"].setdefault(label, {})
            for k, v in sd.items():
                if k == "closure_error":
                    continue
                dst[k] = dst.get(k, 0.0) + float(v) * scale
        sects = ["by_surface_W", "by_body_W", "detected_W"]
        if has_path:
            sects.append("path_tally_Wm")
        for sect in sects:
            for label, v in rep.get(sect, {}).items():
                out[sect][label] = out[sect].get(label, 0.0) \
                    + float(v) * scale
        for label, fx in rep.get("element_flux_W", {}).items():
            dst = out["element_flux_W"].setdefault(
                label, {"in_W": 0.0, "out_W": 0.0})
            dst["in_W"] += float(fx.get("in_W", 0.0)) * scale
            dst["out_W"] += float(fx.get("out_W", 0.0)) * scale
    gate = out["closure_gate"]
    all_ok = True
    for label, sd in out["sources"].items():
        emitted = sd.get("emitted_W", 0.0)
        buckets = sum(v for k, v in sd.items() if k != "emitted_W")
        err = abs(1.0 - buckets / emitted) if emitted > 0.0 else 0.0
        sd["closure_error"] = err
        if err > gate:
            all_ok = False
    out["closure_ok"] = all_ok
    return out


class _LedgerShim:
    """Just enough of audit.PowerLedger for run_trace.build_gdd_budget:
    .emitted (array by source index) and .flux (label -> {in_W}). Built from
    a merged C ledger.json report."""
    def __init__(self, emitted, flux):
        self.emitted = emitted
        self.flux = flux


class _ResultShim:
    """Just enough of tracer.TraceResult for build_gdd_budget: .path_tally
    and .ledger."""
    def __init__(self, path_tally, ledger):
        self.path_tally = path_tally
        self.ledger = ledger


def _gdd_budget_from_report(scene, merged_rep):
    """Build case.json's 'gdd_budget' block from a merged C ledger report,
    reusing run_trace.build_gdd_budget UNCHANGED — ALL dispersion resolution
    (group index / GDD / TOD, the finite-difference stencil) stays in Python
    exactly as the Python engine computes it. The C engine supplies only the
    geometric per-body power-weighted bulk path (path_tally_Wm). Returns None
    when nothing was tallied (matches the Python None case)."""
    from run_trace import build_gdd_budget
    path_tally = {k: float(v)
                  for k, v in merged_rep.get("path_tally_Wm", {}).items()}
    if not path_tally:
        return None
    flux = merged_rep.get("element_flux_W", {})
    srcs = merged_rep.get("sources", {})
    emitted = np.array(
        [float(srcs.get(scene.bodies[bidx].label, {}).get("emitted_W", 0.0))
         for bidx, _ in scene.sources], dtype=np.float64)
    result = _ResultShim(path_tally, _LedgerShim(emitted, flux))
    return build_gdd_budget(scene, result)


def _load_time_records(grids, det_order, chunk_dir):
    """Fold one trace chunk's per-detector time-product arrival records
    (detector.c det_write_times: time_<i>_*.npy columns) into the grids'
    time_records lists, in the SAME compact-column dict shape the Python
    DetectorGrid._record_time_arrivals appends (detector.py:181-194). The
    Python finalize_time then bins the C records with NO code change. Silent
    no-op when a chunk wrote no time files (non-time seed). t is already
    gopl/c (seconds) — the C writer did the division."""
    _dt = {"t": np.float64, "fx": np.float32, "fy": np.float32,
           "lam": np.float32, "power": np.float64,
           "source_id": np.int16, "lam_stratum": np.int16, "gdd": np.float32}
    for i, fid in enumerate(det_order):
        tpath = chunk_dir / ("time_%d_t.npy" % i)
        if not tpath.exists() or len(np.load(tpath)) == 0:
            continue
        g = grids[fid]
        cols = ["t", "fx", "fy", "lam", "power", "source_id", "lam_stratum"]
        # analytic-envelope grids carry per-record GDD (histogram mode
        # ignores it — finalize_time keys off self.time_envelope); the C
        # engine always dumps it, so include the column iff the grid wants it
        if g.time_envelope == "analytic":
            cols.append("gdd")
        rec = {c: np.load(chunk_dir / ("time_%d_%s.npy" % (i, c))).astype(
            _dt[c]) for c in cols}
        g.time_records.append(rec)


def _load_checkpoint(ckpt_path):
    if ckpt_path.exists():
        try:
            return json.loads(ckpt_path.read_text())
        except (OSError, ValueError):
            return None
    return None


def _seed_cursor(chunks, seed):
    """Contiguous covered primary count [0,cursor) for a seed from the
    checkpoint's completed-chunk list (chunks are appended in order, so the
    max hi is the contiguous frontier)."""
    hi = 0
    for c in chunks:
        if c["seed"] == seed:
            hi = max(hi, int(c["hi"]))
    return hi


def run_c_case(args, case_dir, scene, lam_range, case):
    """Run the case on the C engine under the P1 chunked-run contract and
    write the exact output contract _main_locked would. Returns the exit
    code, or None on C-engine failure (auto falls back to Python).

    Honors getattr(args,'resume') / getattr(args,'extend') and auto-resumes a
    matching checkpoint. The C engine traces each chunk (gather_skip);
    Python accumulates, then hands the merged sorted samples BACK to the
    binary for the single final in-binary gather (gather_only mode — the
    tile-factorized kernel)."""
    import common
    from run_trace import (build_detectors, build_detected_block,
                           save_detectors)

    cdir = case_dir / "cengine"
    cdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cdir / "checkpoint.json"

    # P3: one persistent worker for the WHOLE case (every chunk trace + the
    # final gather_only stage), unless the one-shot escape hatch is set. On
    # any worker failure we kill it and fall back to per-invocation _run_binary
    # for the rest of the case (the killed request re-runs one-shot).
    _log_level = getattr(args, "cengine_log_level", None)
    _worker = None
    if os.environ.get("MIEWB_CENGINE_ONESHOT") != "1":
        try:
            _worker = Worker(log_level=_log_level)
        except Exception as exc:                # spawn failed: stay one-shot
            print("[trace] cengine: worker spawn failed (%s) — one-shot"
                  % exc, flush=True)
            _worker = None

    def invoke(req_path):
        """Run one request via the worker; on worker trouble, kill it and
        fall back to one-shot for this and every subsequent request."""
        nonlocal _worker
        if _worker is not None:
            try:
                return _worker.run(req_path)
            except WorkerError as exc:
                print("[trace] cengine: worker failed (%s) — falling back to "
                      "one-shot invocations" % exc, flush=True)
                _worker.kill()
                _worker = None
        return _run_binary(req_path, _log_level)

    def _shutdown_worker():
        nonlocal _worker
        if _worker is not None:
            _worker.close()
            _worker = None

    resume = bool(getattr(args, "resume", False))
    extend = getattr(args, "extend", None)
    target = int(args.rays)
    stride = _align_stride(scene, args)
    shash = scene_hash(args, scene)

    # pulsed-optics P7 time products: recorded on seed 0 only (like the
    # Python engine's time_rec). resolve_time_products folds in the explicit
    # flag AND the pulsed-source auto-enable, so routing never disagrees.
    from run_trace import build_time_cfg, set_time_products_case
    from .detector import resolve_time_products
    time_products = resolve_time_products(args, scene)
    time_cfg = build_time_cfg(args, scene, time_products) \
        if time_products else None
    if time_products:
        # the case.json 'time_products' block, byte-identical to the Python
        # path (run_trace uses the same helper)
        set_time_products_case(case, args, time_products)

    # export/ghost/save-fields/importance/viz-pattern keep their existing
    # seed-0 diagnostic paths simplest by running the whole trace in ONE
    # chunk (still gather_skip + single Python gather; just not split).
    force_one = bool(args.export_rays or args.ghost_analysis
                     or args.save_fields or args.importance_aim
                     or args.viz_pattern)
    chunk_rays = target if force_one else _default_chunk_rays(target, stride)
    step = _chunk_step(target, stride, chunk_rays)

    # ---- checkpoint reconcile (fresh / resume / extend / auto-resume) ----
    ckpt = _load_checkpoint(ckpt_path)
    if ckpt is not None and ckpt.get("scene_hash") != shash:
        # scene/library/knobs changed under us — the old chunks are invalid.
        if resume or extend is not None:
            raise SystemExit(
                "run_trace.py: --resume/--extend but the scene hash no longer "
                "matches %s (model, library, or a physics option changed) — "
                "start a fresh run" % ckpt_path)
        print("[trace] cengine: stale checkpoint (scene changed) — starting "
              "fresh", flush=True)
        ckpt = None
    if extend is not None:
        if ckpt is None or ckpt.get("status") != "completed":
            raise SystemExit(
                "run_trace.py: --extend needs a COMPLETED matching case "
                "(no completed checkpoint in %s)" % cdir)
        old_target = int(ckpt["target_rays"])
        if target <= old_target:
            raise SystemExit(
                "run_trace.py: --extend target %d must exceed the current %d"
                % (target, old_target))
        if old_target % stride != 0:
            raise SystemExit(
                "run_trace.py: --extend boundary %d is not aligned to the "
                "stride %d (n_strata*n_pol) — cannot extend this case"
                % (old_target, stride))
        ckpt.setdefault("extensions", []).append(
            {"from": old_target, "to": target})
        ckpt["target_rays"] = target
        ckpt["status"] = "tracing"
    elif resume:
        if ckpt is None:
            raise SystemExit(
                "run_trace.py: --resume but no checkpoint.json in %s" % cdir)
        if int(ckpt["target_rays"]) != target:
            raise SystemExit(
                "run_trace.py: --resume target %d != checkpoint %d "
                "(use --extend to raise it)"
                % (target, ckpt["target_rays"]))
    if ckpt is None:
        ckpt = {"schema_version": CHECKPOINT_SCHEMA, "scene_hash": shash,
                "target_rays": target, "seeds": int(args.seeds),
                "seed0": int(args.seed0), "align_stride": int(stride),
                "chunk_step": int(step), "status": "tracing",
                "chunks": [], "extensions": []}
    ckpt["status"] = "tracing"
    common.write_json(ckpt_path, ckpt)

    grids_by_seed = {}
    det_order = None
    # (--save-fields-detectors already forced the Python engine at routing
    # time — the C gather saves fields for EVERY detector under
    # --save-fields, so no per-detector subset is needed here.)

    # ================= Phase 1: TRACE (gather_skip chunks) =================
    for si in range(args.seeds):
        seed = args.seed0 + si
        # seed-0 grids record time products (self.time_record on) so the
        # C arrival records can be folded in + finalize_time can bin them.
        time_rec = ({"envelope": args.time_envelope}
                    if (time_products and seed == args.seed0) else None)
        grids = build_detectors(scene, args, lam_range, time_rec=time_rec)
        if det_order is None:
            det_order = list(grids.keys())
        # re-merge already-completed chunks (resume/extend) from disk
        for c in sorted((c for c in ckpt["chunks"] if c["seed"] == seed),
                        key=lambda c: c["lo"]):
            _merge_chunk(grids, det_order, cdir / c["dir"],
                         float(c["rays_denom"]) / target)
            if time_products and seed == args.seed0:
                _load_time_records(grids, det_order, cdir / c["dir"])
        cursor = _seed_cursor(ckpt["chunks"], seed)
        while cursor < target:
            lo = cursor
            hi = min(lo + step, target)
            if hi < target:
                hi = (hi // stride) * stride
                if hi <= lo:
                    hi = min(lo + stride, target)
            chunk_name = "seed%d/chunk_%d_%d" % (seed, lo, hi)
            out_dir = cdir / chunk_name
            out_dir.mkdir(parents=True, exist_ok=True)
            export_on = args.export_rays or args.ghost_analysis
            time_this = bool(time_products) and seed == args.seed0
            req = build_request(
                args, scene, seed, lam_range, grids, out_dir,
                export_this_seed=(export_on and seed == args.seed0),
                track_this_seed=(args.ghost_analysis
                                 and seed == args.seed0),
                time_this_seed=time_this,
                primary_lo=lo, primary_hi=hi, gather_skip=True)
            req_path = out_dir / "request.json"
            req_path.write_text(json.dumps(req))
            print("[trace] seed %d chunk [%d,%d) of %d [C engine]"
                  % (seed, lo, hi, target), flush=True)
            done = (si * target + hi) / max(args.seeds * target, 1)
            common.progress_emit("trace", 0.92 * done,
                                 "seed %d/%d rays %d/%d"
                                 % (si + 1, args.seeds, hi, target),
                                 case_dir=case_dir)
            t0 = time.time()
            rc = invoke(req_path)
            wall_s = time.time() - t0
            if rc != 0:
                print("[trace] ERROR: miewb-trace exited %d (see %s)"
                      % (rc, out_dir / "cengine.log"), flush=True)
                _shutdown_worker()
                return None
            _merge_chunk(grids, det_order, out_dir, 1.0)
            if time_this:
                _load_time_records(grids, det_order, out_dir)
            ckpt["chunks"].append(
                {"seed": seed, "lo": lo, "hi": hi, "rays_denom": target,
                 "dir": chunk_name, "wall_s": wall_s})
            common.write_json(ckpt_path, ckpt)      # atomic (os.replace)
            # test hook: simulate a hard kill (SIGKILL) after N chunks of THIS
            # process, AFTER the checkpoint is durably written — exercises the
            # resume path (the case lock is left behind; acquire steals it as
            # stale on resume).
            _stop = os.environ.get("MIEWB_CHUNK_STOP_AFTER")
            if _stop:
                _stop_after = int(_stop)
                if not hasattr(run_c_case, "_traced_this_proc"):
                    run_c_case._traced_this_proc = 0
                run_c_case._traced_this_proc += 1
                if run_c_case._traced_this_proc >= _stop_after:
                    print("[trace] MIEWB_CHUNK_STOP_AFTER=%d reached — "
                          "simulating a hard kill" % _stop_after, flush=True)
                    os._exit(137)
            # rewrite the detector .h5 snapshot (incoherent-only progress;
            # the coherent gather only lands at completion)
            try:
                save_detectors(case_dir, [grids], 1)
            except Exception as exc:                # snapshot is cosmetic
                print("[trace] snapshot skipped: %s" % exc, flush=True)
            cursor = hi
        grids_by_seed[seed] = grids

    ckpt["status"] = "trace_complete"
    common.write_json(ckpt_path, ckpt)

    # ================= Phase 2: single final gather (C, gather_only) ======
    # The merged, canonically sorted samples + accumulator snapshots are
    # handed BACK to the binary, which runs the normal in-binary gather —
    # the tile-factorized kernel (or the plain fp64 one under
    # --gather-exact). Routing through the Python torch gather here would
    # be a ~75x regression on exactly the long coherent runs chunking
    # exists for.
    common.progress_emit("trace", 0.94, "final gather", case_dir=case_dir)
    grids_list = []
    audits = []
    detected_all = {}
    gather_diags_all = {}
    all_viz = None
    trace_s_total = sum(c["wall_s"] for c in ckpt["chunks"])
    gather_s_total = 0.0
    for si in range(args.seeds):
        seed = args.seed0 + si
        grids = grids_by_seed[seed]
        _finalize_sorted_samples(grids)
        merged_dir = cdir / ("seed%d" % seed) / "merged"
        _write_merged_dump(scene, args, grids, det_order, merged_dir)
        gout = cdir / ("seed%d" % seed) / "gather"
        gout.mkdir(parents=True, exist_ok=True)
        req = build_request(args, scene, seed, lam_range, grids, gout,
                            gather_only=True, gather_input=merged_dir)
        req_path = gout / "request.json"
        req_path.write_text(json.dumps(req))
        print("[trace] seed %d final gather over %d chunk(s) "
              "[C engine, gather_only]"
              % (seed, sum(1 for c in ckpt["chunks"] if c["seed"] == seed)),
              flush=True)
        rc = invoke(req_path)
        if rc != 0:
            print("[trace] ERROR: miewb-trace (gather_only) exited %d "
                  "(see %s)" % (rc, gout / "cengine.log"), flush=True)
            _shutdown_worker()
            return None
        gather_json = gout / "gather.json"
        gdiags = json.loads(gather_json.read_text()) \
            if gather_json.exists() else {}
        detected = json.loads((gout / "detected.json").read_text())
        for i, fid in enumerate(det_order):
            g = grids[fid]
            cube = np.load(gout / ("det_%d_inc.npy" % i))
            if cube.shape != g.inc.shape:
                print("[trace] ERROR: gather_only cube shape %s != "
                      "expected %s for detector %s"
                      % (cube.shape, g.inc.shape, g.label), flush=True)
                _shutdown_worker()
                return None
            g.inc = cube          # snapshot + gathered coherent intensity
            # adopt the binary's tallies verbatim (identical values —
            # the snapshots round-tripped through the dump)
            g.detected_incoherent.clear()
            g.detected_incoherent_n.clear()
            g.detected_geometric.clear()
            for skey, entry in detected.get(g.label, {}).items():
                key = tuple(int(x) for x in skey.split("/"))
                if "incoherent_W" in entry:
                    g.detected_incoherent[key] = \
                        float(entry["incoherent_W"])
                    g.detected_incoherent_n[key] = int(entry["n"])
                if "coherent_W" in entry:
                    g.detected_geometric[key] = float(entry["coherent_W"])
            # --save-fields: complex Ex/Ey maps (seed0 only, matching the
            # Python engine's save_detectors contract)
            if args.save_fields and seed == args.seed0:
                fields = {}
                for skey in gdiags.get(g.label, {}):
                    key = tuple(int(x) for x in skey.split("/"))
                    ex_p = gout / ("det_%d_field_%d_%d_%d_Ex.npy"
                                   % ((i,) + key))
                    ey_p = gout / ("det_%d_field_%d_%d_%d_Ey.npy"
                                   % ((i,) + key))
                    if ex_p.exists() and ey_p.exists():
                        fields[key] = (np.load(ex_p), np.load(ey_p))
                if fields:
                    g.fields = fields
        summary = json.loads((gout / "summary.json").read_text())
        gather_s_total += float(summary.get("gather_seconds") or 0.0)
        grids_list.append(grids)
        gather_diags_all["seed%d" % seed] = gdiags
        detected_all["seed%d" % seed] = build_detected_block(grids, gdiags)
        # merge this seed's chunk ledgers (scaled) into one per-seed report
        reps = []
        for c in sorted((c for c in ckpt["chunks"] if c["seed"] == seed),
                        key=lambda c: c["lo"]):
            rep = json.loads((cdir / c["dir"] / "ledger.json").read_text())
            reps.append((rep, float(c["rays_denom"]) / target))
        merged_rep = _merge_ledger(reps)
        audits.append(merged_rep)
        # pulsed-optics P7 GDD budget: built from seed 0's per-body bulk-path
        # tally (Python run_trace does the same, seed 0 only). All dispersion
        # math is Python-side (build_gdd_budget); C supplied only path_tally.
        if seed == args.seed0 and _track_time_active(args, scene):
            budget = _gdd_budget_from_report(scene, merged_rep)
            if budget is not None:
                case["gdd_budget"] = budget
        # viz overlay: the lo==0 chunk of the first seed holds the first
        # viz_cap primaries (identical to a single run's viz prefix)
        if seed == args.seed0:
            first = min((c for c in ckpt["chunks"] if c["seed"] == seed),
                        key=lambda c: c["lo"])
            vz = cdir / first["dir"] / "rays_viz.npy"
            if vz.exists():
                all_viz = np.load(vz)

    # --export-rays / --ghost-analysis: reconstruct seed-0 ray records from
    # its single chunk dir (force_one guarantees one chunk when exporting)
    if (args.export_rays or args.ghost_analysis) and grids_list:
        seed0 = args.seed0
        cdir0 = None
        for c in ckpt["chunks"]:
            if c["seed"] == seed0:
                cdir0 = cdir / c["dir"]
                break
        if cdir0 is not None:
            _load_export_records(grids_list[0], det_order, cdir0)
        from run_trace import write_rays_full
        write_rays_full(case_dir, grids_list[0], args,
                        Path(args.model_json).parent.name, scene=scene)

    # --viz-pattern deterministic overlay (separate Python viz-only pass)
    if args.viz_pattern:
        from raytracer.tracer import Tracer, TraceConfig
        from raytracer.sources import sample_viz_pattern
        pattern = common.parse_viz_pattern_spec(args.viz_pattern)
        viz_cfg = TraceConfig(max_reflections=args.max_reflections,
                              power_floor=args.power_floor,
                              n_lambda=args.nlambda, rays=1,
                              seed=int(args.seed0), viz_rays=1 << 30,
                              rough_fresnel=args.rough_fresnel)
        viz_tracer = Tracer(scene, viz_cfg, {})
        viz_batches = []
        for sid, (bidx, src) in enumerate(scene.sources):
            vb = sample_viz_pattern(scene, scene.bodies[bidx], src, sid,
                                    pattern, args.nlambda)
            if vb is not None:
                viz_batches.append(vb)
        if viz_batches:
            all_viz = viz_tracer.run(viz_batches).viz.as_array()

    # pulsed-optics P7 time products: bin the seed-0 arrival records into the
    # selected products (seed 0 only, exactly like run_trace's Python path);
    # save_detectors then writes the time_data/time_attrs alongside the cubes.
    if time_cfg is not None and grids_list:
        for grid in grids_list[0].values():
            grid.finalize_time(time_cfg)

    common.progress_emit("trace", 0.97, "writing detectors",
                         case_dir=case_dir)
    np.save(case_dir / "rays.npy",
            all_viz if all_viz is not None else np.zeros((0, 13)))
    save_detectors(case_dir, grids_list, args.seeds)
    common.write_json(case_dir / "audit.json",
                      {"per_seed": audits, "gate": 1e-3})
    case["status"] = "completed"
    case["diagnostics"] = {}
    case["gather"] = gather_diags_all
    case["detected"] = detected_all
    case["timing"] = {"trace_s": trace_s_total, "gather_s": gather_s_total}
    case["chunked_run"] = {
        "target_rays": target, "align_stride": stride,
        "chunk_step": step, "n_chunks": len(ckpt["chunks"]),
        "extensions": ckpt.get("extensions", [])}
    common.write_json(case_dir / "case.json", case)
    ckpt["status"] = "completed"
    common.write_json(ckpt_path, ckpt)
    closure_ok = all(a["closure_ok"] for a in audits)
    if trace_s_total > 0:
        rate_c = (args.seeds * len(scene.sources) * target
                  / trace_s_total)
        common.record_calibration("trace_c", rate_c)
        common.record_calibration(
            "trace_rps_c:" + Path(args.model_json).parent.name, rate_c)
    # gather calibration writeback (p0/quick-wins gather-law rewrite): same
    # "pairs" quantity as the Python engine's _do_gather (surviving
    # coherent samples, summed across every (source, lambda-stratum,
    # pol-stratum) key — they partition, not multiply — times detector
    # pixels; cengine/src/gather.c:617 bills the identical total_pairs).
    # The binary's cuda and OpenMP gather kernels differ ~15x in rate and
    # are calibrated under DISTINCT keys (c_cuda / c_cpu), read off the
    # per-key "backend" field the C engine writes into gather.json.
    total_samples_c = 0
    c_backends = set()
    for gdiags in gather_diags_all.values():
        for keys in gdiags.values():
            for entry in keys.values():
                total_samples_c += entry["n_samples"]
                c_backends.add(entry.get("backend", "cuda"))
    if total_samples_c > 0 and gather_s_total > 0:
        bk_c = "c_cpu" if c_backends == {"cpu"} else "c_cuda"
        pairs_c = total_samples_c * (args.resolution ** 2)
        gather_init_s = common.calibrated_rate(
            "gather_init_s_" + bk_c,
            common.FALLBACK_GATHER_INIT_S_BY[bk_c])
        marginal_s = gather_s_total - gather_init_s
        # only record when the marginal part dominates — an init-dominated
        # measurement calibrates the init constant's noise, not the rate
        if marginal_s > max(0.01, 0.3 * gather_s_total):
            common.record_calibration("gather_pairs_per_s_" + bk_c,
                                      pairs_c / marginal_s)
        # spr denominator: bare per-source rays (matching estimate()'s
        # `pairs = npix * rays * spr`, see run_trace.py._do_gather's
        # identical comment), scaled by args.seeds since total_samples_c
        # aggregates every seed's surviving samples (unlike the Python
        # engine's per-seed _do_gather calls, which each record one
        # seed's spr individually).
        total_rays_c = args.seeds * target
        if total_rays_c > 0:
            common.record_calibration(
                "spr:" + Path(args.model_json).parent.name,
                total_samples_c / total_rays_c)
    _shutdown_worker()          # all invoke()s done; release the GPU context
    print("[trace] done: %d seed(s), %d chunk(s), closure %s, outputs in %s "
          "[C engine]" % (args.seeds, len(ckpt["chunks"]),
                          "OK" if closure_ok else "FAILED", case_dir),
          flush=True)
    common.progress_emit("trace", 1.0,
                         "completed" if closure_ok else "closure FAILED",
                         case_dir=case_dir,
                         status="completed" if closure_ok else "failed")
    return 0 if closure_ok else 3


def _load_export_records(grids, det_order, out_dir):
    """--export-rays / --ghost-analysis: rebuild seed-0 per-detector ray
    records from a chunk dir so run_trace.write_rays_full packs rays_full.npz
    (the SAME writer the Python engine uses)."""
    for i, fid in enumerate(det_order):
        g = grids[fid]
        pos_p = out_dir / ("exp_%d_pos.npy" % i)
        if not pos_p.exists():
            continue
        rec = {
            "pos": np.load(pos_p),
            "dir": np.load(out_dir / ("exp_%d_dir.npy" % i)),
            "birth_pos": np.load(out_dir / ("exp_%d_birth_pos.npy" % i)),
            "opl": np.load(out_dir / ("exp_%d_opl.npy" % i)),
            "lam": np.load(out_dir / ("exp_%d_lam.npy" % i)),
            "power": np.load(out_dir / ("exp_%d_power.npy" % i)),
            "source_id": np.load(
                out_dir / ("exp_%d_source_id.npy" % i)).astype(np.int16),
            "lam_stratum": np.load(
                out_dir / ("exp_%d_lam_stratum.npy" % i)).astype(np.int16),
            "pol_stratum": np.load(
                out_dir / ("exp_%d_pol_stratum.npy" % i)).astype(np.int16),
            "generation": np.load(
                out_dir / ("exp_%d_generation.npy" % i)).astype(np.int16),
            "pol_mode": np.load(
                out_dir / ("exp_%d_pol_mode.npy" % i)).astype(np.int8),
            "scattered": np.load(
                out_dir / ("exp_%d_scattered.npy" % i)).astype(bool),
            "coherent": np.load(
                out_dir / ("exp_%d_coherent.npy" % i)).astype(bool),
        }
        hist_p = out_dir / ("exp_%d_refl_hist.npy" % i)
        if hist_p.exists():
            rec["refl_hist"] = np.load(hist_p)
        if len(rec["pos"]):
            g.ray_records.append(rec)
