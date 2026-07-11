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
import multiprocessing
import os
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
from raytracer.tracer import (Tracer, TraceConfig,       # noqa: E402
                              TraceResult, VizStore)
from raytracer.detector import (DetectorGrid,            # noqa: E402
                                CurvedDetectorGrid)
from raytracer.rays import HIST_DEPTH                    # noqa: E402
from raytracer.audit import PowerLedger                  # noqa: E402
# NB: `gather` (the torch-CUDA coherent Huygens gather) is imported LAZILY
# inside _do_gather so it stays PARENT-ONLY: a spawned trace-shard worker
# re-imports this module, and keeping torch out of that import path is what
# makes multiprocessing sharding CUDA-safe.


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
        lam_tab = src.get("_spectrum_lam_nm")
        if lam_tab is not None:
            # a tabulated spectrum defines its own full range; detector
            # spectral bins must cover the whole table.
            span_lo = float(np.min(lam_tab))
            span_hi = float(np.max(lam_tab))
        else:
            lc = src["lambdac_nm"]
            lmin = src.get("lambdamin_nm") or lc
            lmax = src.get("lambdamax_nm") or lc
            span_lo = lc - 3.0 * max(lc - lmin, 0.0)
            span_hi = lc + 3.0 * max(lmax - lc, 0.0)
        lo = min(lo, span_lo)
        hi = max(hi, span_hi)
    pad = max(5.0, 0.02 * (hi - lo))
    return (lo - pad) * 1e-9, (hi + pad) * 1e-9


def _make_grid(face, args, lam_range):
    """Dispatch a detector face to the right grid class by its surface type:
    planar -> DetectorGrid (unchanged), Sphere/Cylinder -> CurvedDetectorGrid;
    any other analytic/mesh surface falls through to DetectorGrid, which
    raises the existing clear 'planar detector screens only' error."""
    stype = face.surface.__class__.__name__
    cls = CurvedDetectorGrid if stype in ("Sphere", "Cylinder") \
        else DetectorGrid
    return cls(face, args.resolution, args.spectral_bins, lam_range,
               label=face.id)


def build_detectors(scene, args, lam_range):
    grids = {}
    for fid in scene.detector_faces:
        grids[fid] = _make_grid(scene.faces[fid], args, lam_range)
    for fid in scene.extra_detector_faces:
        if fid not in grids:
            grids[fid] = _make_grid(scene.faces[fid], args, lam_range)
    return grids


def resolve_workers(val):
    """--workers value ('auto' | int) -> trace-shard process count. 'auto' =
    max(1, cpu_count-2). Any int is clamped to >= 1."""
    if val == "auto":
        return max(1, (os.cpu_count() or 1) - 2)
    return max(1, int(val))


def compute_viz_caps(scene, args, pattern):
    """Per-source viz-ray cap (int or {source_id: int}); 0 when a
    --viz-pattern overlay is requested (its rays come from a separate pass).
    Factored out so both the single-process path and each shard worker build
    an identical budget."""
    if pattern is not None:
        return 0
    if args.viz_rays is not None:
        return args.viz_rays
    viz_caps = {}
    for sid, (bidx, _src) in enumerate(scene.sources):
        area_mm2 = (scene.emit_faces[bidx].area_m2 or 1e-6) * 1e6
        viz_caps[sid] = int(min(
            max(np.ceil(args.viz_density * area_mm2), 1),
            args.viz_rays_max))
    return viz_caps


def compute_sample_area(scene, args):
    """Per-(source, lam-stratum, pol-stratum) gather normalization area,
    derived from the FULL ray count (int(args.rays)) so it is independent of
    how the trace is sharded across workers."""
    total = int(args.rays)
    sample_area = {}
    for sid, (bidx, src) in enumerate(scene.sources):
        area = scene.emit_faces[bidx].area_m2 or 1e-6
        n_strata = len(wavelength_strata(src, args.nlambda))
        n_pol = n_pol_strata(src)
        rays_per_key = max(total / (n_strata * n_pol), 1)
        for st in range(n_strata):
            for ps in range(n_pol):
                sample_area[(sid, st, ps)] = area / rays_per_key
    return sample_area


def _viz_pattern_pass(scene, args, seed, pattern, result):
    """--viz-pattern deterministic overlay rays: a SEPARATE viz-only pass
    (throwaway ledger, no detector grids), run in the PARENT so it is
    unaffected by sharding. Replaces result.viz with the pattern rays."""
    viz_cfg = TraceConfig(max_reflections=args.max_reflections,
                          power_floor=args.power_floor,
                          n_lambda=args.nlambda, rays=1,
                          seed=seed, viz_rays=1 << 30,
                          rough_fresnel=args.rough_fresnel)
    viz_tracer = Tracer(scene, viz_cfg, {})
    viz_batches = []
    for sid, (bidx, src) in enumerate(scene.sources):
        vb = sample_viz_pattern(scene, scene.bodies[bidx], src, sid,
                                pattern, args.nlambda)
        if vb is not None:
            viz_batches.append(vb)
    if viz_batches:
        n_viz = sum(len(b.pos) for b in viz_batches)
        print("[trace] viz pattern: %d overlay ray(s) in a separate "
              "viz-only pass" % n_viz, flush=True)
        result.viz = viz_tracer.run(viz_batches).viz


def _do_gather(grids, args, scene, sample_area):
    """Coherent Huygens gather over the (merged) detector grids. ALWAYS runs
    single-process in the parent — this is the only place `gather` (torch)
    is imported, keeping the trace-shard workers torch-free / CUDA-safe."""
    from raytracer import gather                          # lazy; parent-only
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
    return gather_diags, gather_s


# ---------------------------------------------------------------------------
# Trace-shard payload extract/merge (multi-process --workers path)
# ---------------------------------------------------------------------------
def _extract_detector_payload(grid):
    """Picklable snapshot of one worker's detector accumulators. inc adds,
    the per-(s,l,p) tallies add per key, gather sample lists concatenate,
    ray-export records concatenate."""
    return {
        "inc": grid.inc,
        "detected_geometric": dict(grid.detected_geometric),
        "detected_incoherent": dict(grid.detected_incoherent),
        "detected_incoherent_n": dict(grid.detected_incoherent_n),
        "samples": grid.samples,
        "ray_records": grid.ray_records,
    }


def _merge_detector_payload(grid, dp):
    """Fold one worker's detector payload into the parent's grid."""
    grid.inc += dp["inc"]
    for k, v in dp["detected_geometric"].items():
        grid.detected_geometric[k] = grid.detected_geometric.get(k, 0.0) + v
    for k, v in dp["detected_incoherent"].items():
        grid.detected_incoherent[k] = \
            grid.detected_incoherent.get(k, 0.0) + v
    for k, v in dp["detected_incoherent_n"].items():
        grid.detected_incoherent_n[k] = \
            grid.detected_incoherent_n.get(k, 0) + v
    for k, recs in dp["samples"].items():
        grid.samples.setdefault(k, []).extend(recs)
    grid.ray_records.extend(dp["ray_records"])


def _shard_worker(args, child_seq, worker_index, rays_i, total_rays,
                  lam_range, particle_lams, export, viz_caps,
                  track_history=False):
    """One trace shard, run in a spawned process. Rebuilds the Scene from
    args.model_json (cheap; Scene is never pickled), traces rays_i primaries
    per source with an independent RNG stream, and returns a picklable
    accumulator payload for the parent to merge. NEVER imports/uses gather
    (torch) so spawning stays CUDA-safe. Only worker 0 records viz rays."""
    scene = build_scene(args)
    grids = build_detectors(scene, args, lam_range)
    rng = np.random.default_rng(child_seq)
    cfg = TraceConfig(max_reflections=args.max_reflections,
                      power_floor=args.power_floor,
                      n_lambda=args.nlambda, rays=rays_i,
                      seed=worker_index, viz_rays=viz_caps,
                      rough_fresnel=args.rough_fresnel,
                      export_rays=export, track_history=track_history)
    particles = None
    part_diag = None
    if args.particles:
        from raytracer.particles import ParticleCloud
        spec = common.parse_particles_spec(args.particles)
        pseed = int(child_seq.generate_state(1)[0])
        particles = ParticleCloud(spec, scene,
                                  threshold=args.particle_threshold,
                                  seed=pseed, lam_list=particle_lams,
                                  pol_scatter=not args.no_pol_scatter)
        part_diag = particles.diagnostics()
    tracer = Tracer(scene, cfg, grids, particle_medium=particles)
    # one RNG stream drives BOTH sample_source and the Tracer's internal
    # roughness/scatter draws (SeedSequence-spawned per worker).
    tracer.rng = rng
    # Power share of this shard: sample_source gives each ray power_W/rays_i
    # and emits the FULL source power into this worker's ledger. Scaling the
    # Jones amplitudes by sqrt(f) (f = rays_i/total) drops every ray to
    # power_W/total_rays — the single-process per-ray amplitude — so the
    # concatenated coherent samples sum correctly and every loss bucket
    # produced by the trace is already f-scaled; then scale the emitted /
    # emission-clipped credits (booked at full power inside sample_source)
    # to match, keeping per-shard (hence merged) closure exact.
    f = rays_i / float(total_rays)
    sqrt_f = np.sqrt(f)
    batches = []
    for sid, (bidx, src) in enumerate(scene.sources):
        b = sample_source(scene, scene.bodies[bidx], src, sid,
                          cfg.rays, cfg.n_lambda, rng,
                          ledger=tracer.ledger,
                          differentials=args.ray_differentials,
                          export_rays=export)
        b.Es *= sqrt_f
        b.Ep *= sqrt_f
        b.birth_power *= f
        batches.append(b)
    tracer.ledger.emitted *= f
    tracer.ledger.buckets["emission_clipped"] *= f
    result = tracer.run(batches)
    return {
        "ledger": result.ledger,
        "detectors": {int(fid): _extract_detector_payload(g)
                      for fid, g in grids.items()},
        "viz": result.viz.as_array() if worker_index == 0 else None,
        "particles": part_diag,
    }


def _run_single(scene, args, seed, particle_lams, case_diag, export,
                viz_caps, grids, track_history=False):
    """Single-process trace (the pre-sharding code path, unchanged). Populates
    `grids` in place and returns (TraceResult, trace_s)."""
    cfg = TraceConfig(max_reflections=args.max_reflections,
                      power_floor=args.power_floor,
                      n_lambda=args.nlambda, rays=int(args.rays),
                      seed=seed, viz_rays=viz_caps,
                      rough_fresnel=args.rough_fresnel,
                      export_rays=export, track_history=track_history)
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
    for sid, (bidx, src) in enumerate(scene.sources):
        b = sample_source(scene, scene.bodies[bidx], src, sid,
                          cfg.rays, cfg.n_lambda, rng,
                          ledger=tracer.ledger,
                          differentials=args.ray_differentials,
                          export_rays=export)
        batches.append(b)
    t0 = time.time()
    result = tracer.run(batches)
    trace_s = time.time() - t0
    common.record_calibration("trace",
                              args.rays * len(scene.sources)
                              / max(trace_s, 1e-9))
    return result, trace_s


def _run_sharded(scene, args, seed, lam_range, particle_lams, case_diag,
                 export, workers, viz_caps, grids, track_history=False):
    """Multi-process trace: N spawned shards trace rays/N primaries each with
    independent RNG streams; the parent merges their accumulators into `grids`
    and a fresh ledger. Returns (TraceResult, trace_s). The gather still runs
    single-process in the parent (caller's _do_gather)."""
    total = int(args.rays)
    n_workers = min(workers, max(total, 1))
    base, rem = divmod(total, n_workers)
    rays_list = [base + (1 if i < rem else 0) for i in range(n_workers)]
    children = np.random.SeedSequence(seed).spawn(n_workers)
    tasks = [(args, children[i], i, rays_list[i], total, lam_range,
              particle_lams, export, (viz_caps if i == 0 else 0),
              track_history)
             for i in range(n_workers)]
    print("[trace] --workers %d: sharding %d rays/source (%s) across "
          "spawned processes; gather runs single-process in the parent"
          % (n_workers, total, "+".join(str(r) for r in rays_list)),
          flush=True)
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        payloads = pool.starmap(_shard_worker, tasks)
    trace_s = time.time() - t0
    common.record_calibration("trace",
                              args.rays * len(scene.sources)
                              / max(trace_s, 1e-9))
    ledger = PowerLedger(len(scene.sources))
    for pl in payloads:
        ledger.merge(pl["ledger"])
        for fid, dp in pl["detectors"].items():
            _merge_detector_payload(grids[fid], dp)
    viz = VizStore()
    v0 = payloads[0]["viz"] if payloads else None
    if v0 is not None and len(v0):
        viz.chunks.append(v0)
    if payloads and payloads[0].get("particles") is not None:
        case_diag.setdefault("particles", payloads[0]["particles"])
    names = [scene.bodies[i].label for i, _ in scene.sources]
    result = TraceResult(grids, ledger, viz, names)
    return result, trace_s


def run_one_seed(scene, args, seed, lam_range, particle_lams, case_diag,
                 export=False, workers=1, track_history=False):
    """Trace one seed. workers<=1 keeps the exact pre-sharding single-process
    path (bit-identical); workers>1 shards the trace across spawned processes
    and merges. Either way the coherent gather runs ONCE in the parent.
    track_history allocates RayBatch.refl_hist (ghost/stray-light face-id
    history); it rides in the export ray records exactly like every other
    field, so the --workers merge concatenates it too."""
    pattern = (common.parse_viz_pattern_spec(args.viz_pattern)
               if args.viz_pattern else None)
    viz_caps = compute_viz_caps(scene, args, pattern)
    grids = build_detectors(scene, args, lam_range)
    sample_area = compute_sample_area(scene, args)

    if workers > 1 and int(args.rays) > 1:
        result, trace_s = _run_sharded(
            scene, args, seed, lam_range, particle_lams, case_diag,
            export, workers, viz_caps, grids, track_history=track_history)
    else:
        result, trace_s = _run_single(
            scene, args, seed, particle_lams, case_diag, export,
            viz_caps, grids, track_history=track_history)

    if pattern is not None:
        _viz_pattern_pass(scene, args, seed, pattern, result)

    gather_diags, gather_s = _do_gather(grids, args, scene, sample_area)
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


_EXPORT_KEYS = ("pos", "dir", "opl", "lam", "source_id", "lam_stratum",
                "pol_stratum", "generation", "pol_mode", "power",
                "scattered", "coherent", "birth_pos")


def write_rays_full(case_dir, grids, args, model_name, scene=None):
    """--export-rays: concatenate seed-0's per-detector ray records into one
    results/<case>/rays_full.npz. Per-detector arrays are namespaced
    '<safe_label>/<field>' (safe_label = label with '.' -> '_', matching the
    detector .h5 files); a global 'meta' key holds a JSON string with each
    detector's grid basis (xhat/yhat/normal/x_lo/y_lo), the seed, the cap,
    per-detector kept counts/fraction, and the model name.

    Under --ghost-analysis (track_history) each detector also gets a
    '<safe_label>/refl_hist' (N, HIST_DEPTH) int32 face-id history array, and
    the meta carries a global 'face_labels' list (face index -> element-label
    "<body>.<FaceN>") so the post ghost renderer maps a path signature to
    surfaces without reconstructing the scene's face ordering.

    Per-detector cap args.export_rays_max: above it a uniform-random subset
    drawn with the run seed is kept (a NOTE is logged) so a huge focal spot
    does not blow the file up."""
    cap = int(args.export_rays_max)
    rng = np.random.default_rng(args.seed0)
    payload = {}
    meta = {"seed": int(args.seed0), "model": model_name,
            "max_reflections": int(args.max_reflections),
            "export_rays_max": cap, "detectors": {}}
    if scene is not None:
        # face index -> "<element label>.<FaceN>" (the FaceN is the last
        # dotted component of the extractor's face id "<Body>.<Feat>.FaceN")
        meta["face_labels"] = [
            "%s.%s" % (scene.body_of_face(i).label,
                       scene.faces[i].id.rsplit(".", 1)[-1])
            for i in range(len(scene.faces))]
    # refl_hist rides along only when the trace tracked history (ghost mode).
    any_hist = any("refl_hist" in r for det in grids.values()
                   for r in det.ray_records)
    for det in grids.values():
        safe = det.label.replace(".", "_")
        recs = det.ray_records
        keys = _EXPORT_KEYS + (("refl_hist",) if any_hist else ())
        cols = {}
        for k in keys:
            if recs and k in recs[0]:
                cols[k] = np.concatenate([r[k] for r in recs])
            elif k == "refl_hist":
                cols[k] = np.zeros((0, HIST_DEPTH), dtype=np.int32)
            else:
                cols[k] = np.zeros((0, 3)) if k in ("pos", "dir",
                                                    "birth_pos") \
                    else np.zeros(0)
        n_total = len(cols["pos"])
        kept_fraction = 1.0
        if n_total > cap:
            idx = np.sort(rng.choice(n_total, size=cap, replace=False))
            cols = {k: v[idx] for k, v in cols.items()}
            kept_fraction = cap / float(n_total)
            print("[trace] NOTE: detector %s exported %d/%d rays "
                  "(subsampled, kept_fraction=%.4g)"
                  % (det.label, cap, n_total, kept_fraction), flush=True)
        for k, v in cols.items():
            payload["%s/%s" % (safe, k)] = v
        meta["detectors"][safe] = {
            "label": det.label,
            "xhat": [float(x) for x in det.xhat],
            "yhat": [float(x) for x in det.yhat],
            "normal": [float(x) for x in det.normal],
            "x_lo": float(det.x_lo), "y_lo": float(det.y_lo),
            "n_total": int(n_total), "n_kept": int(len(cols["pos"])),
            "kept_fraction": float(kept_fraction),
        }
    payload["meta"] = np.array(json.dumps(meta))
    np.savez(case_dir / "rays_full.npz", **payload)
    print("[trace] --export-rays: wrote rays_full.npz (%d detector(s))"
          % len(grids), flush=True)


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
            # curved (Sphere/Cylinder) detectors: extra attrs + a true
            # per-pixel metric area map so post_process divides power by the
            # right area for irradiance. PLANAR files get NONE of this, so
            # their .h5 stays byte-compatible.
            if isinstance(g0, CurvedDetectorGrid):
                h["pixel_area_map"] = g0.pixel_area_map
                h.attrs.update({
                    "surface_type": g0.surface_type,
                    "radius_m": g0.radius,
                    "u_lo": g0.u_lo, "u_hi": g0.u_hi,
                    "v_lo": g0.v_lo, "v_hi": g0.v_hi,
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


def build_scene(args):
    """Construct the Scene from args (model.json + CLI overrides). Factored
    out of _main_locked so each spawned trace-shard worker rebuilds an
    IDENTICAL scene from the same args (Scene construction is cheap and is
    never pickled across the process boundary)."""
    model = common.load_model(args.model_json)
    apply_source_overrides(model, args.source_face)
    from raytracer.optprops import load_optical_properties
    props = load_optical_properties(root=args.optical_properties)
    return Scene(model, props.matdb, props.coatings,
                 suppress_bodies=args.suppress_body,
                 # extra_detector_faces adds transparent screens on top of the
                 # scene (NOT in the C engine's PORTED set -> forces Python).
                 # To pin a detector's PRIMARY face while staying C-routable,
                 # set the detector_face BODY PROPERTY (baked at extract time).
                 extra_detector_faces=args.detector_face,
                 grating_specs=[common.parse_grating_spec(g)
                                for g in args.grating],
                 rough_specs=[common.parse_rough_spec(r)
                              for r in args.rough],
                 optprops=props,
                 geometry_dir=Path(args.model_json).parent,
                 strict_analytic=args.strict_analytic,
                 mesh_flat_normals=args.mesh_flat_normals)


def _main_locked(args, case_dir):
    common.progress_emit("trace", 0.0, "loading scene", case_dir=case_dir)
    scene = build_scene(args)
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

    # ---- engine routing (C engine, cengine/) ----
    # choose_engine picks the compiled engine only when every feature this
    # scene uses is ported+verified; the Python engine below remains the
    # reference. A C-engine runtime failure under --engine auto falls
    # back to Python with a loud warning (crash isolation: the C engine
    # is a separate process).
    from raytracer import cengine
    engine, engine_reason = cengine.choose_engine(args, scene)
    case["engine"] = engine
    case["engine_reason"] = engine_reason
    print("[trace] engine=%s (%s)" % (engine, engine_reason), flush=True)
    if engine == "c":
        rc = cengine.run_c_case(args, case_dir, scene, lam_range, case)
        if rc is not None:
            return rc
        print("[trace] WARNING: C engine failed — falling back to the "
              "Python engine for this case", flush=True)
        case["engine"] = "python"
        case["engine_reason"] = "c-engine runtime failure fallback"

    # wavelengths the particle tables must cover: all strata of all sources
    particle_lams = sorted({
        float(l) for _, src in scene.sources
        for l in wavelength_strata(src, args.nlambda)})

    workers = resolve_workers(args.workers)

    # --ghost-analysis implies export-rays behavior (it needs the seed-0 ray
    # records) AND turns on refl_hist tracking; a bare --export-rays does NOT
    # track history (zero overhead). Both are seed-0 only, like --save-fields.
    export_on = args.export_rays or args.ghost_analysis

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
            scene, args, seed, lam_range, particle_lams, case_diag,
            export=(export_on and s == 0), workers=workers,
            track_history=(args.ghost_analysis and s == 0))
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
    if export_on and grids_list:
        write_rays_full(case_dir, grids_list[0], args,
                        Path(args.model_json).parent.name, scene=scene)
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
