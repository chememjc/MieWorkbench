#!/usr/bin/env python
# =============================================================================
# run_trace.py — the optical solver stage.
#
# Interpreter: /home3/optics/env/bin/python   (numpy/scipy/torch/miepython)
#
# Reads  : geometry/<model>/model.json (validated contract from
#          extract_geometry.py), materials.csv, coatings.csv
# Writes : results/<model>/<case>/
#            case.json         — options echo + status + diagnostics
#            audit.json        — energy ledger (closure gated at 1e-3)
#            rays.npy          — viz polyline segments (N,13):
#                                [source_id, lam_m, power_W, x0..z0,
#                                 x1..z1, pol_mode (0=iso/o, 1=e),
#                                 rel_power (power/birth_power, [0,1]),
#                                 opl0_m, opl1_m (optical path Σn·ds at
#                                 the segment start/end; t = opl/c)]
#            detectors/<label>.h5 — spectral cube + grid metadata
#                                (per-seed mean and std when --seeds > 1)
#
# post_process.py renders images/plots from these files; make_viz.py does
# 3D. This script never imports FreeCAD or paraview.
# =============================================================================
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import common                                            # noqa: E402
import cli_specs                                          # noqa: E402
from raytracer.materials import MaterialDB, load_coatings  # noqa: E402
from raytracer.scene import Scene                        # noqa: E402
from raytracer.sources import (sample_source, wavelength_strata,  # noqa: E402
                               n_pol_strata, sample_viz_pattern)
from raytracer.tracer import Tracer, TraceConfig         # noqa: E402
from raytracer.detector import DetectorGrid              # noqa: E402
from raytracer import gather                             # noqa: E402


def parse_args(argv=None):
    p = cli_specs.build_parser("trace")
    return p.parse_args(argv)


def apply_source_overrides(model, overrides):
    """--source-face Body001.Pad.Face2 retargets that body's emit face."""
    for ov in overrides:
        spec = common.parse_face_spec(ov)
        hit = False
        for b in model["bodies"]:
            if b["name"] == spec["body"] and b.get("source"):
                if not any(f["id"] == ov for f in b["faces"]):
                    raise SystemExit(
                        "--source-face %s: not a face of source body %s. "
                        "Faces: %s" % (ov, b["name"],
                                       [f["id"] for f in b["faces"]]))
                b["source"]["emit_face"] = ov
                b["source"]["emit_face_autodetected"] = False
                hit = True
        if not hit:
            raise SystemExit("--source-face %s: no source body %r"
                             % (ov, spec["body"]))


def lam_range_nm(scene):
    lo, hi = 1e9, 0.0
    for _, src in scene.sources:
        lc = src["lambdac_nm"]
        lmin = src.get("lambdamin_nm") or lc
        lmax = src.get("lambdamax_nm") or lc
        span_lo = lc - 3.0 * max(lc - lmin, 0.0)
        span_hi = lc + 3.0 * max(lmax - lc, 0.0)
        lo = min(lo, span_lo)
        hi = max(hi, span_hi)
    pad = max(5.0, 0.02 * (hi - lo))
    return (lo - pad) * 1e-9, (hi + pad) * 1e-9


def build_detectors(scene, args, lam_range):
    grids = {}
    for fid in scene.detector_faces:
        grids[fid] = DetectorGrid(scene.faces[fid], args.resolution,
                                  args.spectral_bins, lam_range,
                                  label=scene.faces[fid].id)
    for fid in scene.extra_detector_faces:
        if fid not in grids:
            grids[fid] = DetectorGrid(scene.faces[fid], args.resolution,
                                      args.spectral_bins, lam_range,
                                      label=scene.faces[fid].id)
    return grids


def run_one_seed(scene, args, seed, lam_range, particle_lams, case_diag):
    # --viz-pattern: deterministic overlay rays come from a SEPARATE
    # viz-only pass after the physical trace (throwaway ledger, no
    # detector grids), so the physical pass records no viz rays at all
    pattern = (common.parse_viz_pattern_spec(args.viz_pattern)
               if args.viz_pattern else None)
    # viz-ray budget: explicit --viz-rays wins; otherwise density * area
    if pattern is not None:
        viz_caps = 0
    elif args.viz_rays is not None:
        viz_caps = args.viz_rays
    else:
        viz_caps = {}
        for sid, (bidx, _src) in enumerate(scene.sources):
            area_mm2 = (scene.emit_faces[bidx].area_m2 or 1e-6) * 1e6
            viz_caps[sid] = int(min(
                max(np.ceil(args.viz_density * area_mm2), 1),
                args.viz_rays_max))
    cfg = TraceConfig(max_reflections=args.max_reflections,
                      power_floor=args.power_floor,
                      n_lambda=args.nlambda, rays=int(args.rays),
                      seed=seed, viz_rays=viz_caps,
                      rough_fresnel=args.rough_fresnel)
    grids = build_detectors(scene, args, lam_range)
    particles = None
    if args.particles:
        from raytracer.particles import ParticleCloud
        spec = common.parse_particles_spec(args.particles)
        particles = ParticleCloud(spec, scene,
                                  threshold=args.particle_threshold,
                                  seed=seed, lam_list=particle_lams,
                                  pol_scatter=not args.no_pol_scatter)
        case_diag.setdefault("particles", particles.diagnostics())
    tracer = Tracer(scene, cfg, grids, particle_medium=particles)
    rng = np.random.default_rng(seed)
    batches = []
    sample_area = {}
    for sid, (bidx, src) in enumerate(scene.sources):
        b = sample_source(scene, scene.bodies[bidx], src, sid,
                          cfg.rays, cfg.n_lambda, rng,
                          ledger=tracer.ledger,
                          differentials=args.ray_differentials)
        batches.append(b)
        area = scene.emit_faces[bidx].area_m2 or 1e-6
        n_strata = len(wavelength_strata(src, cfg.n_lambda))
        n_pol = n_pol_strata(src)
        rays_per_key = max(cfg.rays / (n_strata * n_pol), 1)
        for st in range(n_strata):
            for ps in range(n_pol):
                sample_area[(sid, st, ps)] = area / rays_per_key
    t0 = time.time()
    result = tracer.run(batches)
    trace_s = time.time() - t0
    common.record_calibration("trace",
                              args.rays * len(scene.sources)
                              / max(trace_s, 1e-9))

    if pattern is not None:
        # viz-only pass: fresh Tracer (own throwaway ledger), no detector
        # grids, no particle medium (its RNG state must stay untouched by
        # visualization), every pattern ray recorded. Detector cubes and
        # the energy audit come exclusively from the physical pass above.
        viz_cfg = TraceConfig(max_reflections=args.max_reflections,
                              power_floor=args.power_floor,
                              n_lambda=args.nlambda, rays=1,
                              seed=seed, viz_rays=1 << 30,
                              rough_fresnel=args.rough_fresnel)
        viz_tracer = Tracer(scene, viz_cfg, {})
        viz_batches = []
        for sid, (bidx, src) in enumerate(scene.sources):
            vb = sample_viz_pattern(scene, scene.bodies[bidx], src, sid,
                                    pattern, cfg.n_lambda)
            if vb is not None:
                viz_batches.append(vb)
        if viz_batches:
            n_viz = sum(len(b.pos) for b in viz_batches)
            print("[trace] viz pattern: %d overlay ray(s) in a separate "
                  "viz-only pass" % n_viz, flush=True)
            result.viz = viz_tracer.run(viz_batches).viz

    occlusion = None
    if args.gather_occlusion:
        occ_faces = [scene.faces[fid] for fid in range(len(scene.faces))
                     if fid not in grids]
        occlusion = {"faces": occ_faces, "exclude_last": None}

    gather_diags = {}
    t0 = time.time()
    for fid, det in grids.items():
        d = gather.render_coherent(
            det, sample_area, backend=args.backend,
            enforce_gate=not args.no_gather_gate,
            min_eff_samples=args.min_eff_samples,
            occlusion=occlusion, save_fields=args.save_fields)
        if d:
            gather_diags[det.label] = {
                "/".join(str(x) for x in k): v for k, v in d.items()}
    gather_s = time.time() - t0
    ops = sum(v["n_samples"] for dd in gather_diags.values()
              for v in dd.values()) * (args.resolution ** 2)
    if ops > 0 and gather_s > 0:
        bk = next(iter(next(iter(gather_diags.values())).values()))[
            "backend"] if gather_diags else "numpy"
        common.record_calibration("gather_" + bk, ops / gather_s)
    return result, grids, gather_diags, {"trace_s": trace_s,
                                         "gather_s": gather_s}


def build_detected_block(grids, gather_diags):
    """Per-seed case.json["detected"] block: {detector label: {"s/l/p":
    {"coherent_W", "incoherent_W", "n_samples"}}}. Merges the coherent
    per-key tally (DetectorGrid.detected_geometric, already surfaced via
    gather_diags' "detected_geometric_W"/"n_samples") with the incoherent
    per-key tally added alongside it (detected_incoherent/_n) — same key
    shape (source_id, lam_stratum, pol_stratum), so a key present in only
    one population simply omits the other's *_W field."""
    out = {}
    for det in grids.values():
        keys = set(det.detected_geometric) | set(det.detected_incoherent)
        if not keys:
            continue
        diag_for_label = gather_diags.get(det.label, {})
        rows = {}
        for key in keys:
            skey = "/".join(str(x) for x in key)
            entry = {}
            if key in det.detected_geometric:
                entry["coherent_W"] = float(det.detected_geometric[key])
            if key in det.detected_incoherent:
                entry["incoherent_W"] = float(det.detected_incoherent[key])
            d = diag_for_label.get(skey)
            entry["n_samples"] = int(d["n_samples"]) if d is not None \
                else int(det.detected_incoherent_n.get(key, 0))
            rows[skey] = entry
        out[det.label] = rows
    return out


def save_detectors(case_dir, grids_list, seeds):
    """grids_list: one dict per seed. Writes mean/std cubes per label."""
    import h5py
    ddir = case_dir / "detectors"
    ddir.mkdir(parents=True, exist_ok=True)
    labels = {g.label: fid for fid, g in grids_list[0].items()}
    for label, fid in labels.items():
        cubes = np.stack([g[fid].inc for g in grids_list])
        g0 = grids_list[0][fid]
        safe = label.replace(".", "_")
        with h5py.File(ddir / (safe + ".h5"), "w") as h:
            h["spectral_cube_mean"] = cubes.mean(axis=0)
            if seeds > 1:
                h["spectral_cube_std"] = cubes.std(axis=0)
            h["mask"] = g0.mask
            # --save-fields: complex Ex/Ey per gather key (seed0 only —
            # cross-seed field averaging is phase-meaningless). Layout
            # consumed by post_process.render_stokes_maps.
            for key, (Ex, Ey) in getattr(g0, "fields", {}).items():
                grp_name = "fields/%s" % "_".join(str(k) for k in key)
                h[grp_name + "/Ex"] = Ex
                h[grp_name + "/Ey"] = Ey
            h.attrs.update({
                "label": label, "H": g0.H, "W": g0.W,
                "pixel_m": g0.pixel_m,
                "lam_lo_m": g0.lam_lo, "lam_hi_m": g0.lam_hi,
                "xhat": g0.xhat, "yhat": g0.yhat,
                "normal": g0.normal,
                "x_lo": g0.x_lo, "y_lo": g0.y_lo,
                "seeds": seeds,
            })


def main(argv=None):
    args = parse_args(argv)
    case_dir = Path(args.case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    # one writer per case: refuse (exit 4) rather than corrupt a live run
    try:
        common.acquire_case_lock(case_dir)
    except common.CaseLocked as exc:
        print("[trace] REFUSED: %s (rerun when it finishes, or remove "
              "%s if you are sure it is dead)"
              % (exc, case_dir / ".lock.json"), flush=True)
        return 4
    try:
        return _main_locked(args, case_dir)
    except BaseException:
        common.progress_emit("trace", None, "failed", case_dir=case_dir,
                             status="failed")
        raise
    finally:
        common.release_case_lock(case_dir)


def _main_locked(args, case_dir):
    common.progress_emit("trace", 0.0, "loading scene", case_dir=case_dir)
    model = common.load_model(args.model_json)
    apply_source_overrides(model, args.source_face)
    from raytracer.optprops import load_optical_properties
    props = load_optical_properties(root=args.optical_properties)
    db = props.matdb
    coatings = props.coatings
    scene = Scene(model, db, coatings,
                  suppress_bodies=args.suppress_body,
                  extra_detector_faces=args.detector_face,
                  grating_specs=[common.parse_grating_spec(g)
                                 for g in args.grating],
                  rough_specs=[common.parse_rough_spec(r)
                               for r in args.rough],
                  optprops=props,
                  geometry_dir=Path(args.model_json).parent,
                  strict_analytic=args.strict_analytic,
                  mesh_flat_normals=args.mesh_flat_normals)
    lam_range = lam_range_nm(scene)
    n_coh = sum(1 for _, s in scene.sources if s.get("coherent"))
    n_pol_max = max((n_pol_strata(s) for _, s in scene.sources), default=1)
    est = common.estimate(args.rays, args.resolution, args.nlambda,
                          n_coh, args.backend,
                          n_detectors=len(scene.detector_faces),
                          n_pol_strata=n_pol_max)
    case = {
        "options": {k: v for k, v in vars(args).items()},
        "estimates": est,
        "status": "estimated",
        "sources": [scene.bodies[b].label for b, _ in scene.sources],
        "detectors": [scene.faces[f].id for f in scene.detector_faces],
    }
    common.write_json(case_dir / "case.json", case)
    print("[trace] estimate: trace %s + gather %s, accumulators %.2f GB"
          % (common.fmt_duration(est["trace_s"]),
             common.fmt_duration(est["gather_s"]),
             est["accumulator_GB"]), flush=True)
    if args.dry_run:
        print("[trace] --dry-run: stopping after estimates", flush=True)
        common.progress_emit("trace", 1.0, "dry-run estimates written",
                             case_dir=case_dir, status="estimated")
        return 0

    # wavelengths the particle tables must cover: all strata of all sources
    particle_lams = sorted({
        float(l) for _, src in scene.sources
        for l in wavelength_strata(src, args.nlambda)})

    case_diag = {}
    grids_list = []
    audits = []
    all_viz = []
    gather_diags_all = {}
    detected_all = {}
    for s in range(args.seeds):
        seed = args.seed0 + s
        print("[trace] seed %d/%d (seed=%d)"
              % (s + 1, args.seeds, seed), flush=True)
        # reserve the last 5% for detector/audit writes after the loop
        common.progress_emit("trace", 0.95 * s / args.seeds,
                             "seed %d/%d" % (s + 1, args.seeds),
                             case_dir=case_dir)
        result, grids, gdiags, times = run_one_seed(
            scene, args, seed, lam_range, particle_lams, case_diag)
        grids_list.append(grids)
        rep = result.ledger.report(result.source_names)
        audits.append(rep)
        gather_diags_all["seed%d" % seed] = gdiags
        detected_all["seed%d" % seed] = build_detected_block(grids, gdiags)
        if s == 0:
            all_viz = result.viz.as_array()
        if not rep["closure_ok"]:
            print("[trace] WARNING: energy closure gate FAILED: %s"
                  % {k: v["closure_error"]
                     for k, v in rep["sources"].items()}, flush=True)

    common.progress_emit("trace", 0.95, "writing detectors",
                         case_dir=case_dir)
    np.save(case_dir / "rays.npy", all_viz)
    save_detectors(case_dir, grids_list, args.seeds)
    common.write_json(case_dir / "audit.json",
                      {"per_seed": audits, "gate": 1e-3})
    case["status"] = "completed"
    case["diagnostics"] = case_diag
    case["gather"] = gather_diags_all
    case["detected"] = detected_all
    case["timing"] = times
    common.write_json(case_dir / "case.json", case)
    closure_ok = all(a["closure_ok"] for a in audits)
    print("[trace] done: %d seed(s), closure %s, outputs in %s"
          % (args.seeds, "OK" if closure_ok else "FAILED",
             case_dir), flush=True)
    common.progress_emit("trace", 1.0,
                         "completed" if closure_ok else "closure FAILED",
                         case_dir=case_dir,
                         status="completed" if closure_ok else "failed")
    return 0 if closure_ok else 3


if __name__ == "__main__":
    sys.exit(main())
