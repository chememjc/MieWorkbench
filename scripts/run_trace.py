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
#            rays.npy          — viz polyline segments (N,10):
#                                [source_id, lam_m, power_W, x0..z0,
#                                 x1..z1, pol_mode (0=iso/o, 1=e)]
#            detectors/<label>.h5 — spectral cube + grid metadata
#                                (per-seed mean and std when --seeds > 1)
#
# post_process.py renders images/plots from these files; make_viz.py does
# 3D. This script never imports FreeCAD or paraview.
# =============================================================================
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import common                                            # noqa: E402
from raytracer.materials import MaterialDB, load_coatings  # noqa: E402
from raytracer.scene import Scene                        # noqa: E402
from raytracer.sources import (sample_source, wavelength_strata,  # noqa: E402
                               n_pol_strata)
from raytracer.tracer import Tracer, TraceConfig         # noqa: E402
from raytracer.detector import DetectorGrid              # noqa: E402
from raytracer import gather                             # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="optical ray trace stage")
    p.add_argument("--model-json", required=True)
    p.add_argument("--case-dir", required=True)
    p.add_argument("--rays", type=float, default=1e5,
                   help="primary rays PER SOURCE")
    p.add_argument("--nlambda", type=int, default=5)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--spectral-bins", type=int, default=16)
    p.add_argument("--max-reflections", type=int, default=6)
    p.add_argument("--power-floor", type=float, default=1e-4)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--backend", default="auto",
                   choices=["auto", "torch", "numpy"])
    p.add_argument("--viz-rays", type=int, default=None,
                   help="absolute viz-ray cap per source (overrides "
                        "--viz-density when set)")
    p.add_argument("--viz-density", type=float, default=1.0,
                   help="viz rays per mm^2 of source emit area "
                        "(visualization only — physics unaffected)")
    p.add_argument("--viz-rays-max", type=int, default=20000,
                   help="hard cap on density-derived viz rays per source")
    p.add_argument("--ray-differentials", action="store_true",
                   help="track per-ray wavefront patch areas (Igehy) so "
                        "the gather uses exact per-sample dA instead of "
                        "the source-referenced approximation (+96 B/ray)")
    p.add_argument("--no-pol-scatter", action="store_true",
                   help="legacy unpolarized Mie azimuth sampling "
                        "(default: sample azimuth from the polarized "
                        "differential cross-section)")
    p.add_argument("--rough-fresnel", default="micro",
                   choices=["micro", "macro"],
                   help="roughness-lobe Fresnel: microfacet-local per-"
                        "polarization (physical) or legacy nominal-angle "
                        "scalar average")
    p.add_argument("--source-face", action="append", default=[],
                   help="override: Body.Feature.FaceN (matched to the "
                        "source body owning that face)")
    p.add_argument("--detector-face", action="append", default=[])
    p.add_argument("--grating", action="append", default=[])
    p.add_argument("--rough", action="append", default=[])
    p.add_argument("--particles", default=None)
    p.add_argument("--particle-threshold", type=float, default=2e5,
                   help="explicit-sphere mode below this count (matches "
                        "the brute-force traversal cap), continuum above")
    p.add_argument("--suppress-body", action="append", default=[])
    p.add_argument("--min-eff-samples", type=float, default=1000.0)
    p.add_argument("--no-gather-gate", action="store_true")
    p.add_argument("--save-fields", action="store_true",
                   help="save per-(source,lam,pol) complex Ex/Ey field "
                        "maps into detectors/<label>.h5 fields/ groups "
                        "(post_process renders Stokes maps from them; "
                        "seed0 only; large files at high resolution)")
    p.add_argument("--gather-occlusion", action="store_true",
                   help="ray-cast each gather sample->detector-tile segment "
                        "against scene bodies and shadow blocked pairs "
                        "(opaque occluders, tile-quantized; see gather.py)")
    p.add_argument("--optical-properties", default=None,
                   help="override the opticalproperties/ library root")
    p.add_argument("--strict-analytic", action="store_true",
                   help="hard-error on mesh-type faces (v1 behavior) "
                        "instead of tracing them with the BVH")
    p.add_argument("--mesh-flat-normals", action="store_true",
                   help="flat facet normals on mesh faces (default: "
                        "angle-weighted smoothed vertex normals)")
    p.add_argument("--dry-run", action="store_true")
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
    # viz-ray budget: explicit --viz-rays wins; otherwise density * area
    if args.viz_rays is not None:
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
