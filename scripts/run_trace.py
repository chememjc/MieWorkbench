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
from raytracer.materials import (MaterialDB, load_coatings,  # noqa: E402
                                 C_LIGHT_M_S)
from raytracer.scene import Scene                        # noqa: E402
from raytracer.sources import (sample_source, wavelength_strata,  # noqa: E402
                               n_pol_strata, sample_viz_pattern,
                               apply_stratum_t0, stratum_domega,
                               install_spm)
from raytracer.tracer import (Tracer, TraceConfig,       # noqa: E402
                              TraceResult, VizStore)
from raytracer.detector import (DetectorGrid,            # noqa: E402
                                CurvedDetectorGrid,
                                resolve_time_products)
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
    # P7b: a chi2 (SHG) body emits harmonic children at lam/2 — the
    # detector spectral range must cover them or every 532 lands in the
    # bottom bin edge (the harmonic of the LOWEST pump still fits at
    # lo/2). CW scenes without nonlinear bodies are untouched.
    if any(b.shg_spec for b in scene.bodies):
        lo = 0.5 * lo
    pad = max(5.0, 0.02 * (hi - lo))
    return (lo - pad) * 1e-9, (hi + pad) * 1e-9


def _make_grid(face, args, lam_range, time_rec=None):
    """Dispatch a detector face to the right grid class by its surface type:
    planar -> DetectorGrid (unchanged), Sphere/Cylinder -> CurvedDetectorGrid;
    any other analytic/mesh surface falls through to DetectorGrid, which
    raises the existing clear 'planar detector screens only' error."""
    stype = face.surface.__class__.__name__
    cls = CurvedDetectorGrid if stype in ("Sphere", "Cylinder") \
        else DetectorGrid
    return cls(face, args.resolution, args.spectral_bins, lam_range,
               label=face.id, time_rec=time_rec)


def build_detectors(scene, args, lam_range, time_rec=None):
    """time_rec: None (the pre-existing zero-overhead default) or
    {'envelope': ...} to buffer time-product arrival records in every grid
    (pulsed-optics P4; seed 0 only, like --save-fields/--export-rays)."""
    grids = {}
    for fid in scene.detector_faces:
        grids[fid] = _make_grid(scene.faces[fid], args, lam_range,
                                time_rec=time_rec)
    for fid in scene.extra_detector_faces:
        if fid not in grids:
            grids[fid] = _make_grid(scene.faces[fid], args, lam_range,
                                    time_rec=time_rec)
    return grids


def build_time_cfg(args, scene, products):
    """The finalize_time config (pulsed-optics P4), or None when no time
    product is active. Carries the CLI knobs plus the two per-source
    tables the analytic envelope needs at finalize time — records carry
    (source_id, lam_stratum), and this maps them to the source pulse
    duration tau0 [s] and the stratum angular bandwidth [rad/s]
    (sources.stratum_domega over wavelength_strata's edges). Chosen over
    threading per-record columns: 16 bytes/record saved, and the mapping
    is exact (strata are deterministic per source)."""
    if not products:
        return None
    n_src = len(scene.sources)
    strata = [wavelength_strata(src, args.nlambda)
              for _, src in scene.sources]
    n_max = max((len(s) for s in strata), default=1)
    tau0 = np.zeros(max(n_src, 1))
    dom = np.zeros((max(n_src, 1), n_max))
    for sid, (_, src) in enumerate(scene.sources):
        pulse = src.get("pulse") or {}
        tau0[sid] = float(pulse.get("duration_s") or 0.0)
        dw = stratum_domega(strata[sid])
        dom[sid, :len(dw)] = dw
    window = None
    if args.time_window is not None:
        window = (args.time_window[0] * 1e-9, args.time_window[1] * 1e-9)
    return {"products": tuple(products), "bins": int(args.time_bins),
            "window": window, "envelope": args.time_envelope,
            "cube_res": int(args.time_cube_res),
            "tau0_by_source": tau0, "domega": dom, "n_sources": n_src}


_FWHM_GAUSS = 4.0 * np.log(2.0)      # tau_out = tau0*sqrt(1+(K*phi2/tau0^2)^2)


def set_time_products_case(case, args, time_products):
    """Write the case.json 'time_products' block (pulsed-optics P4). Shared by
    the Python path here and cengine.run_c_case so both engines emit the
    identical block."""
    case["time_products"] = {
        "products": list(time_products),
        "bins": int(args.time_bins),
        "envelope": args.time_envelope,
        "cube_res": int(args.time_cube_res),
        "window_ns": list(args.time_window)
        if args.time_window is not None else None,
        "auto_enabled": args.time_products is None,
    }
    print("[trace] time products: %s (bins=%d, envelope=%s%s)"
          % (",".join(time_products), args.time_bins, args.time_envelope,
             ", auto-enabled by pulsed source"
             if args.time_products is None else ""), flush=True)


def build_gdd_budget(scene, result):
    """The case.json 'gdd_budget' block (pulsed-optics P5): per traversed
    body, the power-weighted mean bulk path L_bar = path_tally / flux_in
    and the MATERIAL dispersion it contributes (group delay / GDD / TOD,
    scene.medium_* resolution incl. the birefringent/biaxial fallbacks) at
    the reference wavelength, plus totals and a Gaussian pulse-broadening
    estimate per pulsed source AT ITS OWN center wavelength. Geometric
    dispersion (gratings, prisms, angular chirp) is deliberately absent —
    it shows up in the traced time products, not this table. Reference =
    the highest-emitted-power pulsed source (any source when none is
    pulsed). Returns None when nothing was tallied (e.g. an all-air
    scene)."""
    if not result.path_tally:
        return None
    label_to_index = {b.label: b.index for b in scene.bodies}
    emitted = result.ledger.emitted

    def _src_key(k):
        _, src = scene.sources[k]
        pulsed = bool((src.get("pulse") or {}).get("duration_s"))
        return (pulsed, float(emitted[k]) if k < len(emitted) else 0.0)

    k_ref = max(range(len(scene.sources)), key=_src_key)
    lam_ref = float(scene.sources[k_ref][1]["lambdac_nm"]) * 1e-9

    # significance floors: a metal mirror body admits a ~nm evanescent
    # bulk path whose table row would carry metal-dispersion GD/GDD
    # numbers that NO meaningful power ever experiences (the fs_oap
    # contrast demo showed -163,000 fs^2 of "aluminum GDD"). Two guards:
    # a body must see >= 0.1% of the emitted power (flux_in books the
    # SURFACE arrival, so mirrors pass this one) AND its power-weighted
    # mean bulk path must be >= 1 um (a micron of any real glass is
    # < 0.2 fs^2 -- irrelevant; the aluminum skin depth is nm).
    total_emitted = float(np.sum(emitted)) if len(emitted) else 0.0
    flux_floor = 1e-3 * total_emitted
    l_floor = 1e-6

    def _rows_at(lam_m):
        lam = np.asarray([lam_m])
        rows = []
        for label, tally in sorted(result.path_tally.items()):
            flux_in = result.ledger.flux.get(label, {}).get("in_W", 0.0)
            bi = label_to_index.get(label)
            if flux_in <= flux_floor or bi is None:
                continue
            L = float(tally) / float(flux_in)
            if L < l_floor:
                continue
            n_g = float(scene.medium_group_index(bi, lam)[0])
            rows.append({
                "label": label,
                "material": scene.bodies[bi].material,
                "L_bar_mm": L * 1e3,
                "n_g": n_g,
                "gd_fs": n_g * L / C_LIGHT_M_S * 1e15,
                "gdd_fs2": float(
                    scene.medium_gdd_per_length(bi, lam)[0]) * L * 1e30,
                "tod_fs3": float(
                    scene.medium_tod_per_length(bi, lam)[0]) * L * 1e45,
            })
        return rows

    rows = _rows_at(lam_ref)
    if not rows:
        return None
    total = {k: sum(r[k] for r in rows)
             for k in ("gd_fs", "gdd_fs2", "tod_fs3")}
    pulses = []
    for k, (bi_src, src) in enumerate(scene.sources):
        tau0 = float((src.get("pulse") or {}).get("duration_s") or 0.0)
        if tau0 <= 0.0:
            continue
        lam_c = float(src["lambdac_nm"]) * 1e-9
        phi2_fs2 = sum(r["gdd_fs2"]
                       for r in (rows if lam_c == lam_ref
                                 else _rows_at(lam_c)))
        tau0_fs = tau0 * 1e15
        pulses.append({
            "source": scene.bodies[bi_src].label,
            "lambda_c_nm": lam_c * 1e9,
            "tau0_fs": tau0_fs,
            "phi2_fs2": phi2_fs2,
            "tau_out_fs": tau0_fs * float(np.sqrt(
                1.0 + (_FWHM_GAUSS * phi2_fs2 / tau0_fs ** 2) ** 2)),
        })
    return {"lambda_ref_nm": lam_ref * 1e9,
            "reference_source": scene.bodies[scene.sources[k_ref][0]].label,
            "rows": rows, "total": total, "pulses": pulses}


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


def compute_sample_area(scene, args, total_rays=None):
    """Per-(source, lam-stratum, pol-stratum) gather normalization area,
    derived from the emitted ray count so it is independent of how the trace
    is sharded across workers OR chunked across primary ranges.

    total_rays: the number of primaries actually emitted per source at gather
    time (the P1 chunked/extend CURSOR). Defaults to int(args.rays) — the
    single-shot path where cursor == target, unchanged. The C-engine chunked
    driver passes the cursor so an --extend renormalizes for free."""
    total = int(args.rays) if total_rays is None else int(total_rays)
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


def resolve_save_fields_detectors(args, available_labels):
    """--save-fields-detectors LABEL[,LABEL...]: restrict --save-fields'
    complex Ex/Ey field-map writes to these detector labels (matching
    DetectorGrid.label / face.id, e.g. 'Body001.Pad.Face3' — the same
    string --detector-face uses and post_process.py 'safes' into
    det_<label>_*.png / detectors/<safe_label>.h5).

    Returns None when the flag is absent (meaning "every detector" — the
    pre-existing --save-fields behavior, unchanged) or the parsed set.
    A label not present in `available_labels` is a HARD ERROR (never a
    silent no-op) naming what IS available, so a typo doesn't quietly
    produce zero field groups."""
    if not args.save_fields_detectors:
        return None
    wanted = {s.strip() for s in args.save_fields_detectors.split(",")
              if s.strip()}
    unknown = wanted - set(available_labels)
    if unknown:
        raise SystemExit(
            "run_trace.py: --save-fields-detectors unknown label(s): %s "
            "— available detector labels: %s"
            % (", ".join(sorted(unknown)),
               ", ".join(sorted(available_labels))
               or "(no detectors in this scene)"))
    return wanted


def _do_gather(grids, args, scene, sample_area, save_fields_labels=None):
    """Coherent Huygens gather over the (merged) detector grids. ALWAYS runs
    single-process in the parent — this is the only place `gather` (torch)
    is imported, keeping the trace-shard workers torch-free / CUDA-safe.

    save_fields_labels: None (every detector eligible, args.save_fields'
    pre-existing behavior) or a set of detector labels (--save-fields-
    detectors) — a detector not in the set is gathered normally but never
    gets its complex Ex/Ey field maps recorded."""
    from raytracer import gather                          # lazy; parent-only
    occlusion = None
    if args.gather_occlusion:
        occ_faces = [scene.faces[fid] for fid in range(len(scene.faces))
                     if fid not in grids]
        occlusion = {"faces": occ_faces, "exclude_last": None}
    gather_diags = {}
    t0 = time.time()
    for fid, det in grids.items():
        want_fields = args.save_fields and (
            save_fields_labels is None or det.label in save_fields_labels)
        d = gather.render_coherent(
            det, sample_area, backend=args.backend,
            enforce_gate=not args.no_gather_gate,
            min_eff_samples=args.min_eff_samples,
            occlusion=occlusion, save_fields=want_fields)
        if d:
            gather_diags[det.label] = {
                "/".join(str(x) for x in k): v for k, v in d.items()}
    gather_s = time.time() - t0
    # "pairs" = surviving coherent samples (summed across every (source,
    # lambda-stratum, pol-stratum) key — they PARTITION the samples, they
    # don't multiply them, cengine/src/gather.c:617) * detector pixels.
    # This is the exact quantity common.estimate()'s new gather law bills
    # against; see common.py's FALLBACK_GATHER_PAIRS_PER_S comment.
    total_samples = sum(v["n_samples"] for dd in gather_diags.values()
                        for v in dd.values())
    pairs = total_samples * (args.resolution ** 2)
    if pairs > 0 and gather_s > 0:
        bk = next(iter(next(iter(gather_diags.values())).values()))[
            "backend"] if gather_diags else "numpy"
        gather_init_s = common.calibrated_rate(
            "gather_init_s_" + bk,
            common.FALLBACK_GATHER_INIT_S_BY.get(
                bk, common.FALLBACK_GATHER_INIT_S))
        marginal_s = gather_s - gather_init_s
        # only record when the marginal part dominates — an init-dominated
        # measurement calibrates the init constant's noise, not the rate
        if marginal_s > max(0.01, 0.3 * gather_s):
            common.record_calibration("gather_pairs_per_s_" + bk,
                                      pairs / marginal_s)
        # spr is scaled against the bare `rays` value (args.rays, the
        # per-source --rays count) — the SAME scalar common.estimate()
        # multiplies pairs by (it has no n_sources parameter and never
        # multiplies by source count, matching the pre-existing trace_s
        # law above). A multi-source scene's extra surviving samples are
        # therefore folded into that scene's own spr, not divided back
        # out — spr is looked up per model_stem, so this is exactly the
        # scene-characteristic ratio estimate() needs.
        if args.rays > 0:
            model_stem = Path(args.model_json).parent.name
            common.record_calibration("spr:" + model_stem,
                                      total_samples / float(args.rays))
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
        "time_records": grid.time_records,
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
    grid.time_records.extend(dp.get("time_records", []))


def _shard_worker(args, child_seq, worker_index, rays_i, total_rays,
                  lam_range, particle_lams, export, viz_caps,
                  track_history=False, track_time=False, time_rec=None):
    """One trace shard, run in a spawned process. Rebuilds the Scene from
    args.model_json (cheap; Scene is never pickled), traces rays_i primaries
    per source with an independent RNG stream, and returns a picklable
    accumulator payload for the parent to merge. NEVER imports/uses gather
    (torch) so spawning stays CUDA-safe. Only worker 0 records viz rays.
    time_rec: every shard buffers arrival records (they concatenate in the
    parent's merge, order-independent by construction)."""
    scene = build_scene(args)
    grids = build_detectors(scene, args, lam_range, time_rec=time_rec)
    rng = np.random.default_rng(child_seq)
    cfg = TraceConfig(max_reflections=args.max_reflections,
                      power_floor=args.power_floor,
                      n_lambda=args.nlambda, rays=rays_i,
                      seed=worker_index, viz_rays=viz_caps,
                      rough_fresnel=args.rough_fresnel,
                      export_rays=export, track_history=track_history,
                      track_time=track_time,
                      importance_scatter=getattr(
                          args, "importance_scatter", False),
                      importance_limit=getattr(
                          args, "importance_limit", 1.0),
                      pol_transport=getattr(args, "pol_transport", False),
                      biref_approx=getattr(args, "biref_approx", False))
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
        if cfg.track_time:
            # pre-allocate + apply the pulsed-optics birth-time offset
            # HERE, before tracer.run() ever sees this batch: sample_source
            # always runs before Tracer.run() allocates gopl (its own
            # alloc_time() call is an idempotent no-op once we've already
            # set it), so this is the only place a source's optional
            # _stratum_t0 hook can land before any propagation accumulates
            # on top of it. See apply_stratum_t0's docstring.
            b.alloc_time()
            apply_stratum_t0(b, src)
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
        # per-body power-weighted bulk path (track_time only; linear
        # tally, shards add per key like the ledger flux)
        "path_tally": result.path_tally,
        # P7b: per-body SHG transferred power (linear tally, shards add)
        "shg_converted": result.shg_converted,
        # conical-point guard: per-crystal ray count (linear tally, shards
        # add; see TraceResult.conical_guard)
        "conical_guard": result.conical_guard,
    }


def _run_single(scene, args, seed, particle_lams, case_diag, export,
                viz_caps, grids, track_history=False, track_time=False):
    """Single-process trace (the pre-sharding code path, unchanged). Populates
    `grids` in place and returns (TraceResult, trace_s)."""
    cfg = TraceConfig(max_reflections=args.max_reflections,
                      power_floor=args.power_floor,
                      n_lambda=args.nlambda, rays=int(args.rays),
                      seed=seed, viz_rays=viz_caps,
                      rough_fresnel=args.rough_fresnel,
                      export_rays=export, track_history=track_history,
                      track_time=track_time,
                      importance_scatter=getattr(
                          args, "importance_scatter", False),
                      importance_limit=getattr(
                          args, "importance_limit", 1.0),
                      pol_transport=getattr(args, "pol_transport", False),
                      biref_approx=getattr(args, "biref_approx", False))
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
        if cfg.track_time:
            # see the matching comment in _shard_worker: this is the only
            # place a source's optional _stratum_t0 birth-time offset can
            # land before tracer.run() begins propagating rays.
            b.alloc_time()
            apply_stratum_t0(b, src)
        batches.append(b)
    t0 = time.time()
    result = tracer.run(batches)
    trace_s = time.time() - t0
    rate = args.rays * len(scene.sources) / max(trace_s, 1e-9)
    common.record_calibration("trace_rays_per_s_v2", rate)
    common.record_calibration(
        "trace_rps_py:" + Path(args.model_json).parent.name, rate)
    return result, trace_s


def _run_sharded(scene, args, seed, lam_range, particle_lams, case_diag,
                 export, workers, viz_caps, grids, track_history=False,
                 track_time=False, time_rec=None):
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
              track_history, track_time, time_rec)
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
    rate = args.rays * len(scene.sources) / max(trace_s, 1e-9)
    common.record_calibration("trace_rays_per_s_v2", rate)
    common.record_calibration(
        "trace_rps_py:" + Path(args.model_json).parent.name, rate)
    ledger = PowerLedger(len(scene.sources))
    path_tally = {}
    shg_converted = {}
    conical_guard = {}
    for pl in payloads:
        ledger.merge(pl["ledger"])
        for fid, dp in pl["detectors"].items():
            _merge_detector_payload(grids[fid], dp)
        for k, v in pl.get("path_tally", {}).items():
            path_tally[k] = path_tally.get(k, 0.0) + v
        for k, v in pl.get("shg_converted", {}).items():
            shg_converted[k] = shg_converted.get(k, 0.0) + v
        for k, v in pl.get("conical_guard", {}).items():
            conical_guard[k] = conical_guard.get(k, 0) + v
    viz = VizStore()
    v0 = payloads[0]["viz"] if payloads else None
    if v0 is not None and len(v0):
        viz.chunks.append(v0)
    if payloads and payloads[0].get("particles") is not None:
        case_diag.setdefault("particles", payloads[0]["particles"])
    names = [scene.bodies[i].label for i, _ in scene.sources]
    result = TraceResult(grids, ledger, viz, names, path_tally=path_tally,
                         shg_converted=shg_converted,
                         conical_guard=conical_guard)
    return result, trace_s


def run_one_seed(scene, args, seed, lam_range, particle_lams, case_diag,
                 export=False, workers=1, track_history=False,
                 track_time=False, save_fields_labels=None, time_rec=None):
    """Trace one seed. workers<=1 keeps the exact pre-sharding single-process
    path (bit-identical); workers>1 shards the trace across spawned processes
    and merges. Either way the coherent gather runs ONCE in the parent.
    track_history allocates RayBatch.refl_hist (ghost/stray-light face-id
    history); it rides in the export ray records exactly like every other
    field, so the --workers merge concatenates it too. save_fields_labels:
    see _do_gather / resolve_save_fields_detectors. time_rec: time-product
    arrival recording config for the detector grids (seed 0 only, set by
    the caller — see build_detectors)."""
    pattern = (common.parse_viz_pattern_spec(args.viz_pattern)
               if args.viz_pattern else None)
    viz_caps = compute_viz_caps(scene, args, pattern)
    grids = build_detectors(scene, args, lam_range, time_rec=time_rec)
    sample_area = compute_sample_area(scene, args)

    if workers > 1 and int(args.rays) > 1:
        result, trace_s = _run_sharded(
            scene, args, seed, lam_range, particle_lams, case_diag,
            export, workers, viz_caps, grids, track_history=track_history,
            track_time=track_time, time_rec=time_rec)
    else:
        result, trace_s = _run_single(
            scene, args, seed, particle_lams, case_diag, export,
            viz_caps, grids, track_history=track_history,
            track_time=track_time)

    if pattern is not None:
        _viz_pattern_pass(scene, args, seed, pattern, result)

    gather_diags, gather_s = _do_gather(grids, args, scene, sample_area,
                                        save_fields_labels=save_fields_labels)
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

    Under --pol-transport each detector also gets '<safe_label>/Qmat'
    (N,3,3) float64, '/Jmat' (N,2,2) complex128, and '/s_hat' (N,3) float64
    — post_process.render_pol_transport reads these to compute honest
    per-ray retardance/diattenuation/fast-axis (M = Q^T P).

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
    # gopl/gdd_acc ride along only when the trace tracked time (pulsed
    # phases) — same conditional pattern as refl_hist.
    any_time = any("gopl" in r for det in grids.values()
                   for r in det.ray_records)
    # Qmat/Jmat/s_hat ride along only under --pol-transport (P2) — same
    # conditional pattern again.
    any_pol_transport = any("Qmat" in r for det in grids.values()
                            for r in det.ray_records)
    for det in grids.values():
        safe = det.label.replace(".", "_")
        recs = det.ray_records
        keys = _EXPORT_KEYS + (("refl_hist",) if any_hist else ()) \
            + (("gopl", "gdd_acc") if any_time else ()) \
            + (("Qmat", "Jmat", "s_hat") if any_pol_transport else ())
        cols = {}
        for k in keys:
            if recs and k in recs[0]:
                cols[k] = np.concatenate([r[k] for r in recs])
            elif k == "refl_hist":
                cols[k] = np.zeros((0, HIST_DEPTH), dtype=np.int32)
            elif k == "Qmat":
                cols[k] = np.zeros((0, 3, 3))
            elif k == "Jmat":
                cols[k] = np.zeros((0, 2, 2), dtype=np.complex128)
            else:
                cols[k] = np.zeros((0, 3)) if k in ("pos", "dir",
                                                    "birth_pos", "s_hat") \
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
            # time products (pulsed-optics P4; seed0 only, populated by
            # finalize_time): time_profile / time_profile_by_source /
            # time_spectrogram / time_streak / time_cube datasets + the
            # t_lo_s/t_hi_s/time_* attrs. Absent entirely (byte-compatible
            # .h5) when no time product was active.
            for name, arr in getattr(g0, "time_data", {}).items():
                h[name] = arr
            h.attrs.update(getattr(g0, "time_attrs", {}) or {})
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

    # P1 chunked-run contract: --extend RAYS raises the target; downstream
    # (estimate, chunk driver) reads it as the new args.rays, while args.extend
    # stays set as the "this is an extension of a completed case" signal.
    if getattr(args, "extend", None) is not None:
        args.rays = float(args.extend)

    # one writer per case: refuse (exit 4) rather than corrupt a live run.
    # P1 --resume/--extend continue an EXISTING case: a lock left by a hard
    # kill has a recent heartbeat (not yet "stale" by age) but a DEAD pid, so
    # steal it only when its owner is truly gone on this host.
    force_lock = False
    if getattr(args, "resume", False) or getattr(args, "extend", None) \
            is not None:
        info = common.lock_info(case_dir)
        if info is not None:
            import socket
            same_host = info.get("host") == socket.gethostname()
            pid = info.get("pid")
            dead = False
            if same_host and pid:
                try:
                    os.kill(int(pid), 0)
                except (OSError, ValueError):
                    dead = True
            force_lock = dead or common.lock_is_stale(case_dir, info)
    try:
        common.acquire_case_lock(case_dir, force=force_lock)
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
    scene = Scene(model, props.matdb, props.coatings,
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
                 mesh_flat_normals=args.mesh_flat_normals,
                 temperature_c=getattr(args, "temperature", None))
    # pulsed-optics P6: source-side SPM transform (spm property). Runs
    # HERE, inside the one factored scene builder, so the parent process
    # and every spawned shard install bit-identical spectra/birth-time
    # offsets (deterministic, RNG-free).
    install_spm(scene, args.nlambda)
    return scene


def _main_locked(args, case_dir):
    common.progress_emit("trace", 0.0, "loading scene", case_dir=case_dir)
    scene = build_scene(args)
    lam_range = lam_range_nm(scene)
    n_coh = sum(1 for _, s in scene.sources if s.get("coherent"))
    n_pol_max = max((n_pol_strata(s) for _, s in scene.sources), default=1)
    # every detector this case will build a grid for (build_detectors'
    # exact set: the auto/pinned primary faces plus any --detector-face
    # extras) — the universe --save-fields-detectors validates against and
    # (absent a subset) the "all detectors get fields" default.
    all_detector_fids = set(scene.detector_faces) | set(
        scene.extra_detector_faces)
    available_detector_labels = sorted(
        scene.faces[fid].id for fid in all_detector_fids)
    save_fields_labels = resolve_save_fields_detectors(
        args, available_detector_labels)
    n_field_dets = (len(save_fields_labels)
                    if save_fields_labels is not None
                    else len(available_detector_labels))
    if args.save_fields and n_coh == 0:
        # complex Ex/Ey field maps only exist for the coherent gather —
        # an all-incoherent scene writes EMPTY fields groups and the
        # Stokes/PSF/MTF renderers silently no-op (UXNOTES_ROUND3 #21)
        print("[trace] WARNING: --save-fields with NO coherent source: "
              "the fields/ groups will be empty and no Stokes/PSF/MTF "
              "products will render — set coherent=true on a source if "
              "you want field maps", file=sys.stderr)
    if n_coh > 0:
        # P2: table coatings without ars/arp/ats/atp_deg columns borrow
        # the bare-interface Fresnel phase (materials.py phase_valid) --
        # fine for incoherent power, silently approximate for coherent
        # interference. Headless-CLI mirror of mieworkbench.core.
        # validation.check_coating_phase (GUI pre-run check).
        no_phase_coatings = sorted({
            cname for cname in set(scene.face_coatings.values())
            if scene.coatings.get(cname, {}).get("kind") == "table"
            and not scene.coatings.get(cname, {}).get("phase_valid")})
        if no_phase_coatings:
            print("[trace] WARNING: coherent source(s) present with "
                  "phase-invalid table coating(s) %s: interference "
                  "through %s coating uses bare-interface phase — supply "
                  "Ars/Arp/Ats/Atp columns or a TMM stack for "
                  "phase-accurate results"
                  % (", ".join(repr(c) for c in no_phase_coatings),
                     "this" if len(no_phase_coatings) == 1 else "these"),
                  file=sys.stderr)
    # ---- engine routing decision (hoisted above the estimate so the
    # estimate bills against the engine that will actually run — the C
    # gather/trace rates differ from the torch/numpy ones by ~6x/8x) ----
    from raytracer import cengine
    engine, engine_reason = cengine.choose_engine(args, scene)
    # P1: resume/extend are a C-engine-only contract. The Python engine's
    # numpy RNG is stateful (each seed's stream is consumed in one pass), so
    # there is no primary cursor to resume from or extend past — refuse
    # clearly rather than silently re-running or double-counting.
    if engine != "c" and (getattr(args, "resume", False)
                          or getattr(args, "extend", None) is not None):
        raise SystemExit(
            "run_trace.py: --resume/--extend require the C engine "
            "(engine=%s here: %s). The Python engine's numpy RNG is "
            "stateful — checkpoint/resume/extend is C-engine-only. Re-run "
            "with --engine c, or run a fresh --rays <N>."
            % (engine, engine_reason))
    if engine == "c" and args.save_fields and save_fields_labels is not None:
        # the C engine always saves fields for every detector under
        # --save-fields (no per-detector subset plumbed through its
        # request/output contract yet) — --engine c is a hard error (never
        # a silent wrong answer), --engine auto quietly falls back.
        if (getattr(args, "engine", None) or "auto") == "c":
            raise SystemExit(
                "--engine c: --save-fields-detectors is not yet supported "
                "by the C engine (it saves fields for every detector "
                "under --save-fields) — use --engine auto/python")
        engine = "python"
        engine_reason = ("--save-fields-detectors requires the Python "
                         "engine (C engine saves fields for every "
                         "detector)")
    est_backend = (("c_cpu" if args.backend == "numpy" else "c_cuda")
                   if engine == "c" else args.backend)
    est = common.estimate(args.rays, args.resolution, args.nlambda,
                          n_coh, est_backend,
                          n_detectors=n_field_dets,
                          save_fields=args.save_fields,
                          n_pol_strata=n_pol_max,
                          model_stem=Path(args.model_json).parent.name)
    case = {
        "options": {k: v for k, v in vars(args).items()},
        "estimates": est,
        "status": "estimated",
        "sources": [scene.bodies[b].label for b, _ in scene.sources],
        # pulsed-optics Phase P3: per-source pulse metadata (energy_J,
        # duration_s, rep_rate_Hz, peak_power_W, avg_power_W, derived,
        # kappa — see raytracer.scene._parse_pulse_source), keyed by
        # source label, for post-process/GUI consumption. A source with
        # no pulse properties at all contributes no entry (empty dict
        # when the case has no pulsed sources -- unchanged schema for the
        # overwhelming majority of existing scenes). Per-pulse DETECTED
        # quantities are a later phase; this is metadata visibility only.
        "source_pulse": {scene.bodies[b].label: src["pulse"]
                         for b, src in scene.sources if "pulse" in src},
        # pulsed-optics P6: SPM sources echo the resolved peak nonlinear
        # phase (phi_max, rad) alongside the raw spec string
        "source_spm": {scene.bodies[b].label:
                       {"spec": src["spm"],
                        "phimax_rad": src.get("_spm_phimax")}
                       for b, src in scene.sources if "spm" in src},
        # pulsed-optics P7b: harmonic stratum-id map (child stratum =
        # n_lambda + parent stratum) — present only when a chi2 body is
        # in the scene; consumers (post/GUI spectra labels) use it to
        # tell fundamental strata from SHG children
        **({"harmonic_strata": {
                "n_lambda": int(args.nlambda),
                "map": {str(k): int(args.nlambda) + k
                        for k in range(int(args.nlambda))},
                "bodies": [b.label for b in scene.bodies if b.shg_spec]}}
           if any(b.shg_spec for b in scene.bodies) else {}),
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
    # the routing DECISION was made above (before the estimate); the Python
    # engine below remains the reference. A C-engine runtime failure under
    # --engine auto falls back to Python with a loud warning (crash
    # isolation: the C engine is a separate process).
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
    # track history (zero overhead). --pol-transport likewise implies
    # export-rays (it needs the seed-0 ray records to carry Qmat/Jmat). All
    # three are seed-0 only, like --save-fields.
    export_on = (args.export_rays or args.ghost_analysis
                or getattr(args, "pol_transport", False))

    # track_time: pulsed-optics time-domain accumulators (gopl/gdd_acc per
    # ray, per-body path tally), driven by the resolved time products
    # (--time-products / the pulsed-source auto-rule — P4). When active the
    # engine was already forced to Python above (cengine.detect_features
    # adds the 'time_products' feature via the SAME resolver).
    time_products = resolve_time_products(args, scene)
    time_cfg = build_time_cfg(args, scene, time_products)
    # --gdd-budget forces group-delay tracking even with no time product
    # active (CW dispersion audit); with any product active the budget
    # comes for free off the same tally (engine routing already Python
    # via the 'gdd_budget' / 'time_products' feature tokens)
    track_time = bool(time_products) or args.gdd_budget
    if time_products:
        set_time_products_case(case, args, time_products)

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
            track_history=(args.ghost_analysis and s == 0),
            track_time=track_time,
            save_fields_labels=save_fields_labels,
            time_rec=({"envelope": args.time_envelope}
                      if time_products and s == 0 else None))
        grids_list.append(grids)
        rep = result.ledger.report(result.source_names)
        if track_time:
            # per-body power-weighted bulk path [W*m] (GDD budget input;
            # key present only when time tracking ran, so the audit.json
            # schema is unchanged otherwise)
            rep["path_tally_Wm"] = {
                k: float(v) for k, v in sorted(result.path_tally.items())}
        if result.shg_converted:
            # P7b diagnostic: per-body SHG transferred power (a stratum-
            # to-stratum transfer, never a closure bucket)
            rep["shg_converted_W"] = {
                k: float(v)
                for k, v in sorted(result.shg_converted.items())}
        if result.conical_guard:
            # conical-point runtime guard (engine3.md Sec 7.2): per-crystal
            # count of rays whose incident k fell inside the optic-axis
            # degeneracy cone this seed (basis-arbitrary o/e split; never a
            # closure bucket). Also rolled into case["diagnostics"]
            # ("conical_guard") below, summed over every seed.
            rep["conical_guard_rays"] = {
                k: int(v) for k, v in sorted(result.conical_guard.items())}
            guard_diag = case_diag.setdefault("conical_guard", {})
            for k, v in result.conical_guard.items():
                guard_diag[k] = guard_diag.get(k, 0) + int(v)
        audits.append(rep)
        gather_diags_all["seed%d" % seed] = gdiags
        detected_all["seed%d" % seed] = build_detected_block(grids, gdiags)
        if s == 0:
            all_viz = result.viz.as_array()
            if track_time:
                budget = build_gdd_budget(scene, result)
                if budget is not None:
                    case["gdd_budget"] = budget
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
    if time_cfg is not None and grids_list:
        # bin the seed-0 arrival records into the selected time products
        # (seed 0 only, like --save-fields; save_detectors writes the
        # populated time_data/time_attrs alongside the spectral cubes)
        for grid in grids_list[0].values():
            grid.finalize_time(time_cfg)
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
