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
    "export_rays",              # phase H (per-detector landing records)
    "ghost_analysis",           # phase H (refl_hist face-id history)
    "viz_pattern",              # phase H (glue-level: Python viz-only
                                #   pass supplies the overlay rays)
})

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def binary_path():
    """Path to miewb-trace, or None if not built. MIEWB_CENGINE overrides
    (repo convention, like MIEWB_FREECAD etc. in common.py)."""
    env = os.environ.get("MIEWB_CENGINE")
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
    for bidx, src in scene.sources:
        feats.add("surface:%s"
                  % type(scene.emit_faces[bidx].surface).__name__.lower())
        if src.get("coherent"):
            feats.add("coherent")
        if src.get("beam"):
            feats.add("beam")
        if src.get("apodization"):
            feats.add("apodization")
    for body in scene.bodies:
        if body.birefringent:
            feats.add("birefringence")
        if body.biaxial:
            feats.add("biaxial")
        if body.polarizer:
            feats.add("polarizer")
        if body.filter:
            feats.add("filter")
        # pulsed-optics NLO elements (P8 Pockels/saturable/TPA/Kerr, P7b
        # chi2 SHG): none exist in the C engine — each forces Python.
        # These were MISSING for P8's elements at first: a kerr_n2 body
        # whose other features were all ported routed to C and silently
        # skipped the physics ("every feature emits its token" is the
        # round's own locked rule).
        if body.nonlinear:
            feats.add("nonlinear")
        if body.saturable_raw:
            feats.add("saturable")
        if body.tpa_beta:
            feats.add("tpa")
        if body.kerr_n2_raw:
            feats.add("kerr")
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
    if scene.face_coatings:
        feats.add("coating")
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
        from .mie import LogNormalDistribution, number_density
        spec = common.parse_particles_spec(args.particles)
        dist = LogNormalDistribution(
            median_r=spec["median_um"] * 1e-6 / 2.0, gsd=spec["gsd"])
        mat_p = scene.matdb.get(spec["material"])
        rho_h = scene.ambient.density if scene.ambient.density > 0 \
            else 1.204
        N, _ = number_density(spec["phi"], mat_p.density, rho_h, dist)
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


def build_request(args, scene, seed, lam_range, grids, out_dir,
                  export_this_seed=False, track_this_seed=False,
                  primary_lo=0, primary_hi=None, gather_skip=False):
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
            "power_W": float(src["power_mW"]) * 1e-3,
            "coherent": bool(src.get("coherent", False)),
            "lam_offset": int(lam_off),
            "n_strata": int(n_strata),
            "n_pol": int(n_pol),
            "jones": jones,
            "viz_cap": int(cap),
            "emit_face": _face_entry(-1, emit_rec, bidx, -1),
        }
        entry.update(_emit_dir_policy(face))
        # per-(stratum, pol) gather normalization areas (compute_sample_area)
        from run_trace import compute_sample_area
        sa = compute_sample_area(scene, args)
        entry["sample_area"] = [
            float(sa[(sid, st, ps)])
            for st in range(n_strata) for ps in range(n_pol)]
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
            "seed": int(seed),
            "batch_size": 1 << 20,
            "threads": 0 if args.workers == "auto"
                       else resolve_workers(args.workers),
            "mesh_flat_normals": bool(args.mesh_flat_normals),
            "export_rays": bool(export_this_seed),
            "importance_aim": bool(getattr(args, "importance_aim",
                                           False)),
            "track_history": bool(track_this_seed),
            "linear_scan": bool(os.environ.get("MIEWB_CENGINE_LINEAR")
                                == "1"),
        },
        "lams_m": [float(x) for x in lams],
        "ambient_n_re": [float(x) for x in np.real(amb)],
        "ambient_n_im": [float(x) for x in np.imag(amb)],
        "bodies": bodies,
        "faces": faces,
        "sources": sources,
        "detectors": detectors,
        "coatings": coatings,
        "roughs": roughs,
        "scatters": scatters,
        "gratings": gratings,
        "particles": _particles_block(args, scene, lams),
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
        },
    }


def _particles_block(args, scene, lams):
    """Continuum particle-cloud tables at the stratum wavelengths (D1):
    mu_ext / albedo per lam, the radius-node CDF, and a per-(lam, node)
    inverse phase-function CDF built from the SAME
    MieEvaluator.phase_function tables the Python engine samples from."""
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
    u_grid = np.linspace(0.0, 1.0, n_u)
    radii = cloud.tables.radii
    mu_ext, albedo, radius_cdf, inv_phase = [], [], [], []
    for lam in lams:
        t = cloud.tables._nearest(float(lam))
        mu_ext.append(float(t["mu_ext"]))
        albedo.append(float(t["albedo"]))
        radius_cdf.extend(
            float(x) for x in np.cumsum(t["radius_weights"]))
        for rv in radii:
            mu_g, _p, cdf = cloud.evaluator.phase_function(
                float(rv), float(lam))
            inv_phase.extend(
                float(x) for x in np.interp(u_grid, cdf, mu_g))
    return {
        "box_lo": [float(x) for x in cloud.lo],
        "box_hi": [float(x) for x in cloud.hi],
        "n_quad": int(len(radii)),
        "n_u": n_u,
        "mu_ext": mu_ext,
        "albedo": albedo,
        "radius_cdf": radius_cdf,
        "inv_phase": inv_phase,
    }


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
# P1 chunked-run contract: checkpoint / resume / additive extension
#
# The C engine traces primaries in CHUNKS (gather_skip mode: each invocation
# dumps its coherent samples + incoherent cube + ledger instead of gathering).
# The Python driver merges the chunk payloads into DetectorGrid accumulators
# and runs the SINGLE final coherent gather ONCE at the end (the reference
# run_trace._do_gather kernel; the C in-binary gather stays for the
# direct-binary parity tests). checkpoint.json makes an interrupted trace
# resumable and a completed run extendable, and the per-chunk sample dumps on
# disk ARE the durable accumulator (rebuilt into grids on resume).
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
                "dA": np.full(int(n), np.nan),
                "_ray_key": np.load(base + "key.npy"),
                "_evt": np.load(base + "evt.npy"),
            }
            g.samples.setdefault(key, []).append(rec)


def _finalize_sorted_samples(grids):
    """Collapse each detector's per-chunk sample records into ONE record with
    the samples in the canonical (ray_key,event_ctr) order — a total order
    (each is unique), so the merged multi-chunk order is byte-identical to a
    single trace's, making the final gather bit-reproducible. The hidden
    _ray_key/_evt columns are dropped afterward (render_coherent never sees
    them)."""
    for g in grids.values():
        for key, recs in list(g.samples.items()):
            rk = np.concatenate([r["_ray_key"] for r in recs])
            ev = np.concatenate([r["_evt"] for r in recs])
            order = np.lexsort((ev, rk))       # primary key rk, tiebreak ev
            rec = {f: np.concatenate([r[f] for r in recs])[order]
                   for f in _SAMPLE_FIELDS}
            g.samples[key] = [rec]
            g.detected_geometric[key] = float(np.sum(rec["power"]))


def _merge_ledger(reports):
    """Merge per-chunk C ledger.json reports (each paired with its extend
    scale) into one per-seed report. Numeric leaves sum (scaled); per-source
    closure_error is recomputed from the summed emitted/buckets."""
    out = {"sources": {}, "by_surface_W": {}, "by_body_W": {},
           "element_flux_W": {}, "detected_W": {}, "closure_gate": 1e-3}
    for rep, scale in reports:
        for label, sd in rep.get("sources", {}).items():
            dst = out["sources"].setdefault(label, {})
            for k, v in sd.items():
                if k == "closure_error":
                    continue
                dst[k] = dst.get(k, 0.0) + float(v) * scale
        for sect in ("by_surface_W", "by_body_W", "detected_W"):
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
    matching checkpoint. The C engine traces each chunk (gather_skip); Python
    accumulates and runs the single final gather."""
    import common
    from run_trace import (build_detectors, build_detected_block,
                           save_detectors, compute_sample_area, _do_gather,
                           resolve_save_fields_detectors)

    cdir = case_dir / "cengine"
    cdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cdir / "checkpoint.json"

    resume = bool(getattr(args, "resume", False))
    extend = getattr(args, "extend", None)
    target = int(args.rays)
    stride = _align_stride(scene, args)
    shash = scene_hash(args, scene)

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
    save_fields_labels = resolve_save_fields_detectors(
        args, sorted(scene.faces[fid].id
                     for fid in (set(scene.detector_faces)
                                 | set(scene.extra_detector_faces))))

    # ================= Phase 1: TRACE (gather_skip chunks) =================
    for si in range(args.seeds):
        seed = args.seed0 + si
        grids = build_detectors(scene, args, lam_range)
        if det_order is None:
            det_order = list(grids.keys())
        # re-merge already-completed chunks (resume/extend) from disk
        for c in sorted((c for c in ckpt["chunks"] if c["seed"] == seed),
                        key=lambda c: c["lo"]):
            _merge_chunk(grids, det_order, cdir / c["dir"],
                         float(c["rays_denom"]) / target)
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
            req = build_request(
                args, scene, seed, lam_range, grids, out_dir,
                export_this_seed=(export_on and seed == args.seed0),
                track_this_seed=(args.ghost_analysis
                                 and seed == args.seed0),
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
            rc = _run_binary(req_path)
            wall_s = time.time() - t0
            if rc != 0:
                print("[trace] ERROR: miewb-trace exited %d (see %s)"
                      % (rc, out_dir / "cengine.log"), flush=True)
                return None
            _merge_chunk(grids, det_order, out_dir, 1.0)
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

    # ================= Phase 2: single final gather =======================
    common.progress_emit("trace", 0.94, "final gather", case_dir=case_dir)
    grids_list = []
    audits = []
    detected_all = {}
    gather_diags_all = {}
    all_viz = None
    trace_s_total = sum(c["wall_s"] for c in ckpt["chunks"])
    gather_s_total = 0.0
    sample_area = compute_sample_area(scene, args, total_rays=target)
    for si in range(args.seeds):
        seed = args.seed0 + si
        grids = grids_by_seed[seed]
        _finalize_sorted_samples(grids)
        gdiags, gather_s = _do_gather(
            grids, args, scene, sample_area,
            save_fields_labels=save_fields_labels)
        gather_s_total += gather_s
        grids_list.append(grids)
        gather_diags_all["seed%d" % seed] = gdiags
        detected_all["seed%d" % seed] = build_detected_block(grids, gdiags)
        # merge this seed's chunk ledgers (scaled) into one per-seed report
        reps = []
        for c in sorted((c for c in ckpt["chunks"] if c["seed"] == seed),
                        key=lambda c: c["lo"]):
            rep = json.loads((cdir / c["dir"] / "ledger.json").read_text())
            reps.append((rep, float(c["rays_denom"]) / target))
        audits.append(_merge_ledger(reps))
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
