#!/usr/bin/env python
# =============================================================================
# run_dls.py — real traced-dynamics dynamic light scattering (DLS) driver.
#
# Interpreter: /home3/optics/env/bin/python  (numpy/scipy/torch/h5py)
#
# Traces a TIME SEQUENCE of coherent speckle fields off a Brownian particle
# suspension and persists them so scripts/dls_correlate.py can extract the
# field autocorrelation g1(tau), the Siegert g2, and — via the Stokes-
# Einstein relation — the hydrodynamic diameter. The suspension is ONE
# body-bound sample medium (`sample` body property -> a samples.miesamp
# registry row, raytracer/particles.py BodyParticleMedium) in EXPLICIT
# mode: a frozen realization of discrete spheres that a random walk moves
# between frames, so a fixed illumination sees a physically evolving
# speckle pattern.
#
# FLOW
#  1. Build the Scene ONCE (run_trace.build_scene) and locate the single
#     sampled body's BodyParticleMedium (EXPLICIT mode required; exactly one
#     sampled body in v1; at least one COHERENT source required).
#  2. Frame 0 = the medium's explicit realization. Pre-generate N frames of
#     Brownian positions SEQUENTIALLY: per particle D_i = kB*T/(6*pi*eta*r_i)
#     (eta = the sample row's solvent_visc_pas, T = --temp-k), per-axis step
#     sigma_i = sqrt(2*D_i*dt). A REFLECTIVE body wall is enforced by
#     rejection: a step landing outside the body (Scene.point_inside_body)
#     is redrawn up to _REFLECT_TRIES times, else that particle holds still
#     this frame (the walk is microseconds — never the bottleneck).
#  3. Trace the frames EMBARRASSINGLY PARALLEL over a spawn process pool
#     (numpy.random.SeedSequence.spawn per worker, the CUDA-safe convention
#     of run_trace._run_sharded). Each worker rebuilds the scene + medium,
#     injects that frame's positions into the frozen realization (radii are
#     drawn ONCE at frame 0 and passed in — frame-invariant), traces with
#     the Python engine, and reconstructs the RAW coherent field at each
#     requested detector.
#  4. Persist <case>/dls/frames.h5 + <case>/dls/manifest.json.
#
# GATHER-BACKEND ROUTING (deliberate choice): the coherent field per frame
# is evaluated with gather.points_numpy — the RAW Rayleigh-Sommerfeld field
# (UNnormalized, UNgated) — directly inside each worker on the NUMPY
# backend. DLS needs the frame-to-frame PHASE evolution of the field, not a
# power-calibrated image, so render_coherent's per-population power
# renormalization and its M_eff sampling gate are deliberately bypassed
# (they would inject a per-frame amplitude/abort artifact). numpy keeps
# torch/CUDA out of the spawned workers entirely, so trace AND field-gather
# both parallelize with no parent serialization; DLS detectors are tiny
# (default 32x32) so the per-frame numpy evaluation is cheap.
#
# FRAMES.H5 SCHEMA
#   /positions              (N, n_p, 3)   float32   Brownian positions [m]
#   /radii                  (n_p,)        float32   frozen particle radii [m]
#   /dt_s                   ()            float64   inter-frame time [s]
#   /temp_k                 ()            float64   temperature [K]
#   /detectors/<safe>/frames   (N, nkeys, 2, H, W) complex64
#                           per-frame RAW coherent field; axis 1 = gather
#                           key (see the group 'keys_json' attr, "s/l/p"
#                           order), axis 2 = [Ex, Ey] detector-frame Jones
#                           components. Keys are MUTUALLY INCOHERENT
#                           populations; correlations sum, fields do not.
#   /detectors/<safe>/q_vector (3,)       float64   mean scattering vector
#                           k_s - k_i [1/m] (source beam axis + sample->
#                           detector geometry); |q| in the q_magnitude_per_m
#                           group attr.
#   group attrs: label,H,W,pixel_m,xhat,yhat,normal,x_lo,y_lo,
#                q_magnitude_per_m,lam_lo_m,lam_hi_m,keys_json
#   root attrs : seed,engine,sample,body,host_material,solvent_visc_pas,
#                n_particles,rays,nlambda,frames,dt_ms,sample_row_json
#
# CLI
#   run_dls.py --model-json geometry/<stem>/model.json
#              --case-dir results/<stem>/<case>
#              --frames N --dt-ms X [--temp-k 293.15] [--rays R]
#              [--nlambda 1] [--detectors LABEL,..] [--resolution 32]
#              [--workers auto] [--seed S] [--max-gb 2]
#              [--optical-properties DIR] [--particle-threshold N]
#
# HONEST LIMITS: single-scattering DLS (keep the sample optically thin,
# tau <~ 0.1, or multiple scattering corrupts g1); FROZEN radii (no size
# evolution); NO hydrodynamic interactions and NO structure-factor collective
# slow-down (dilute Stokes-Einstein); NO sedimentation / flow (drift-free
# Brownian only). g1 phase is geometric (the illumination is frozen), so
# decorrelation is driven purely by particle displacement.
#
# SPARSE-CLOUD REQUIREMENT (important): the explicit medium samples each
# scatter event with a SHARED Monte-Carlo RNG stream whose draw order
# depends on the exact set of ray-particle collisions. In a DENSE cloud a
# nm-scale particle move flips a collision somewhere, desynchronising the
# whole stream, so the speckle field goes DELTA-correlated frame-to-frame
# (g1 collapses in one frame) instead of decaying at the physical rate
# D*q^2. For a clean, quantitative g1 = exp(-D q^2 tau) the sample must be
# DILUTE — a small number of WELL-SEPARATED spheres (tens, not thousands),
# so the collision set is stable across a frame and the field decorrelates
# purely through the q.r geometric phase. Load the samples.miesamp row to a
# very small optical depth (few x 1e-3) accordingly, and index-match the
# ambient to the solvent so scattered rays are not TIR-trapped at cell
# walls. Dense suspensions still run and persist, but their g1 is a
# delta-plus-noise and the cumulant D/d_H are unreliable — the correlator
# math itself is validated independently against synthetic fields
# (scripts/raytracer/tests/test_dls.py).
# =============================================================================
import argparse
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
import cli_specs                                         # noqa: E402
import run_trace                                         # noqa: E402  (reuse)
from raytracer.sources import (sample_source,            # noqa: E402
                               wavelength_strata, n_pol_strata)
from raytracer.tracer import Tracer, TraceConfig         # noqa: E402
from raytracer.audit import PowerLedger                  # noqa: E402
from raytracer.particles import build_body_sample_media  # noqa: E402
from raytracer import gather                             # noqa: E402  (numpy only)

# CODATA 2018 exact Boltzmann constant [J/K]
KB = 1.380649e-23
_REFLECT_TRIES = 8       # reflective-wall rejection redraws per particle/frame


# ---------------------------------------------------------------------------
# argument plumbing
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Traced-dynamics DLS: a Brownian frame sequence of "
                    "coherent speckle fields off a body-bound sample medium")
    p.add_argument("--model-json", required=True,
                   help="geometry/<stem>/model.json (any valid extractor / "
                        "scenehelpers model.json with one `sample` body)")
    p.add_argument("--case-dir", required=True,
                   help="results/<stem>/<case> output directory")
    p.add_argument("--frames", type=int, required=True,
                   help="number of Brownian frames N (frame 0 = the frozen "
                        "realization)")
    p.add_argument("--dt-ms", type=float, required=True,
                   help="inter-frame time in milliseconds")
    p.add_argument("--temp-k", type=float, default=293.15,
                   help="temperature [K] for D_i = kB*T/(6*pi*eta*r_i) "
                        "(default 293.15)")
    p.add_argument("--rays", type=float, default=1e5,
                   help="primary rays per source per frame (default 1e5)")
    p.add_argument("--nlambda", type=int, default=1,
                   help="wavelength strata (default 1 — DLS is quasi-"
                        "monochromatic)")
    p.add_argument("--detectors", default=None, metavar="LABEL,..",
                   help="comma list of detector face-id labels to persist "
                        "(default: every detector in the scene)")
    p.add_argument("--resolution", type=int, default=32,
                   help="detector grid resolution (default 32 — DLS point "
                        "detectors are small; larger grids balloon "
                        "frames.h5)")
    p.add_argument("--workers", default="auto",
                   help="parallel frame workers ('auto' = max(1, "
                        "cpu_count-2), or an int)")
    p.add_argument("--seed", type=int, default=12345,
                   help="master seed (realization + Brownian walk + per-"
                        "frame spawn streams)")
    p.add_argument("--max-gb", type=float, default=2.0,
                   help="refuse to run if the estimated frames.h5 exceeds "
                        "this many GB (default 2)")
    p.add_argument("--optical-properties", default=None,
                   help="override the opticalproperties/ library root")
    p.add_argument("--particle-threshold", type=float, default=2e5,
                   help="explicit/continuum auto threshold (the sample "
                        "row's mode= overrides it; explicit is required "
                        "here regardless)")
    p.add_argument("--max-reflections", type=int, default=6)
    p.add_argument("--power-floor", type=float, default=1e-4)
    return p.parse_args(argv)


def build_trace_args(args):
    """A full trace-stage argparse.Namespace (every default filled) via
    cli_specs, overridden with the DLS knobs. Reused by the parent AND every
    spawned worker so the Scene / detector build is byte-identical."""
    p = cli_specs.build_parser("trace")
    argv = [
        "--model-json", args.model_json,
        "--case-dir", args.case_dir,
        "--rays", repr(float(args.rays)),
        "--nlambda", str(int(args.nlambda)),
        "--resolution", str(int(args.resolution)),
        "--spectral-bins", "1",
        "--seeds", "1",
        "--engine", "python",
        "--backend", "numpy",
        "--max-reflections", str(int(args.max_reflections)),
        "--power-floor", repr(float(args.power_floor)),
        "--particle-threshold", repr(float(args.particle_threshold)),
    ]
    if args.optical_properties:
        argv += ["--optical-properties", args.optical_properties]
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# medium + geometry helpers
# ---------------------------------------------------------------------------
def particle_lams(scene, nlambda):
    """Wavelengths (metres) the sample Mie tables must cover: every stratum
    of every source."""
    return sorted({float(l) for _, src in scene.sources
                   for l in wavelength_strata(src, nlambda)})


def build_sample_medium(scene, args, seed):
    """The one BodyParticleMedium of the scene. Hard-errors (naming what is
    missing) on: no sampled body, more than one sampled body, or a sample
    row that resolves to a non-explicit medium."""
    lam_list = particle_lams(scene, args.nlambda)
    samples_reg = scene.optprops.samples if scene.optprops is not None else {}
    n_sampled = sum(1 for b in scene.bodies if getattr(b, "sample", None))
    if n_sampled == 0:
        raise SystemExit(
            "run_dls.py: no body carries a `sample` property — DLS needs "
            "exactly one body-bound sample medium (samples.miesamp row on a "
            "liquid-fill solid). See docs/RAYTRACER.md §5.")
    if n_sampled > 1:
        raise SystemExit(
            "run_dls.py: %d bodies carry a `sample` property — v1 supports "
            "exactly one sampled body per DLS run (%s)"
            % (n_sampled, ", ".join(b.label for b in scene.bodies
                                    if getattr(b, "sample", None))))
    media = build_body_sample_media(
        scene, samples_reg, threshold=args.particle_threshold,
        seed=seed, lam_list=lam_list,
        pol_scatter=True)
    if not media:
        raise SystemExit("run_dls.py: sample medium construction returned "
                         "nothing (internal error)")
    medium = media[0]
    if medium.mode != "explicit" or medium.explicit is None:
        raise SystemExit(
            "run_dls.py: sample %r resolved to '%s' mode, but DLS requires "
            "EXPLICIT mode (discrete spheres a random walk can move). Set "
            "mode=explicit on the samples.miesamp row, or lower the loading "
            "so the count falls below the explicit threshold."
            % (medium.sample_name, medium.mode))
    return medium, lam_list


def require_coherent(scene):
    n_coh = sum(1 for _, s in scene.sources if s.get("coherent"))
    if n_coh == 0:
        raise SystemExit(
            "run_dls.py: no COHERENT source in the scene — DLS reconstructs "
            "a coherent speckle field and needs coherent=true on a source.")
    return n_coh


def solvent_viscosity(medium):
    eta = medium.sample_row.get("solvent_visc_pas")
    if eta is None or not (float(eta) > 0):
        raise SystemExit(
            "run_dls.py: sample %r has no solvent_visc_pas — DLS needs the "
            "solvent viscosity [Pa*s] to convert the decay rate to a "
            "diffusion coefficient. Add solvent_visc_pas to the "
            "samples.miesamp row." % medium.sample_name)
    return float(eta)


# ---------------------------------------------------------------------------
# Brownian pre-generation (sequential, reflective body wall)
# ---------------------------------------------------------------------------
def brownian_frames(scene, body_index, centers0, radii, n_frames, dt_s,
                    temp_k, eta_pas, seed):
    """(N, n_p, 3) float32 Brownian trajectory. Frame 0 = centers0. Per
    particle sigma_i = sqrt(2*D_i*dt), D_i = kB*T/(6*pi*eta*r_i). Reflective
    wall by rejection: a proposed step leaving the body is redrawn up to
    _REFLECT_TRIES times, else the particle holds still that frame."""
    rng = np.random.default_rng(seed ^ 0x5A17)
    n_p = len(radii)
    D = KB * temp_k / (6.0 * np.pi * eta_pas * np.asarray(radii))
    sigma = np.sqrt(2.0 * D * dt_s)                      # (n_p,) per-axis std
    pos = np.zeros((n_frames, n_p, 3), dtype=np.float64)
    pos[0] = centers0
    for f in range(1, n_frames):
        cur = pos[f - 1]
        step = rng.normal(0.0, 1.0, size=(n_p, 3)) * sigma[:, None]
        cand = cur + step
        inside = scene.point_inside_body(cand, body_index)
        for _ in range(_REFLECT_TRIES):
            if np.all(inside):
                break
            bad = ~inside
            step_bad = rng.normal(0.0, 1.0, size=(int(bad.sum()), 3)) \
                * sigma[bad, None]
            cand[bad] = cur[bad] + step_bad
            inside2 = scene.point_inside_body(cand[bad], body_index)
            fix = np.where(bad)[0]
            inside[fix] = inside2
        # any particle still outside after the redraws holds still this frame
        cand[~inside] = cur[~inside]
        pos[f] = cand
    return pos.astype(np.float32)


# ---------------------------------------------------------------------------
# scattering geometry (q vector per detector)
# ---------------------------------------------------------------------------
def incident_direction(scene, ledger_dummy):
    """Mean emitted ray direction of the first coherent source (k_i hat)."""
    for sid, (bidx, src) in enumerate(scene.sources):
        if not src.get("coherent"):
            continue
        rng = np.random.default_rng(7)
        b = sample_source(scene, scene.bodies[bidx], src, sid, 512, 1, rng,
                          ledger=ledger_dummy)
        d = np.asarray(b.dir, dtype=np.float64).mean(axis=0)
        n = np.linalg.norm(d)
        return (d / n if n > 0 else np.array([1.0, 0.0, 0.0])), src
    raise SystemExit("run_dls.py: no coherent source for the q geometry")


def q_vector_for(det, k_i_hat, sample_centroid, n_host, lam_m):
    """Mean scattering vector q = k_s - k_i for one detector. k_s aims from
    the sample centroid to the detector's (masked) pixel-center centroid;
    both wavevectors have magnitude 2*pi*n_host/lam."""
    centers = det.pixel_centers.reshape(-1, 3)
    mask = det.mask.reshape(-1)
    pts = centers[mask] if np.any(mask) else centers
    det_center = pts.mean(axis=0)
    ks_dir = det_center - sample_centroid
    nrm = np.linalg.norm(ks_dir)
    k_s_hat = ks_dir / nrm if nrm > 0 else k_i_hat
    k0 = 2.0 * np.pi * n_host / lam_m
    q = k0 * (k_s_hat - k_i_hat)
    return q, float(np.linalg.norm(q))


# ---------------------------------------------------------------------------
# per-frame worker: RAW coherent field per detector per gather key
# ---------------------------------------------------------------------------
def _inject_realization(medium, positions, radii):
    """Overwrite the frozen realization's centers/radii and recompute the
    frame-invariant per-particle extinction collision radius + albedo (adds
    NOTHING to particles.py — set attributes directly, per the contract)."""
    ex = medium.explicit
    ex.centers = np.ascontiguousarray(positions, dtype=np.float64)
    ex.radii = np.ascontiguousarray(radii, dtype=np.float64)
    lam0 = float(np.mean(medium.tables.lam_list))
    qext, qsca, _ = medium.evaluator.efficiencies(
        ex.radii, np.full_like(ex.radii, lam0))
    ex.r_col = ex.radii * np.sqrt(np.maximum(qext, 1e-12))
    ex.albedo_p = np.where(qext > 0, qsca / np.maximum(qext, 1e-12), 0.0)


def _field_for_detector(det, sample_area):
    """RAW Rayleigh-Sommerfeld coherent field per gather key at this
    detector -> {keystr: (2, H, W) complex64} ([Ex, Ey]). Bypasses
    render_coherent's gate/normalization (see the GATHER-BACKEND ROUTING
    note): DLS wants the un-renormalized field's phase evolution. A key with
    fewer than the 4 samples the kernel needs is skipped (all-zero field)."""
    merged = det.merged_samples()
    out = {}
    grid_pts = det.pixel_centers.reshape(-1, 3)
    for key, s in merged.items():
        pos = s["pos"]
        if len(pos) < 4:
            continue
        p_hat = np.cross(s["dir"], s["s_hat"])
        dA_fb = sample_area.get(key, 1.0)
        dA = s.get("dA")
        if dA is not None and np.any(np.isfinite(dA)):
            good = np.isfinite(dA) & (dA > 0)
            dA = np.where(good, dA, dA_fb)
        else:
            dA = np.full(len(pos), dA_fb)
        E3 = (s["Es"][:, None] * s["s_hat"]
              + s["Ep"][:, None] * p_hat) * np.sqrt(dA)[:, None]
        Ex, Ey = gather.points_numpy(
            pos, E3, s["lam"], s["opl"], grid_pts,
            det.xhat, det.yhat, det.normal, s["dir"])
        keystr = "/".join(str(x) for x in key)
        out[keystr] = np.stack([Ex.reshape(det.H, det.W),
                                Ey.reshape(det.H, det.W)]).astype(np.complex64)
    return out


def _frame_worker(targs, child_seq, det_fids, radii, positions_chunk,
                  frame_idx_chunk, sample_seed, base_seed):
    """One process: build scene+medium ONCE, then for each assigned frame
    inject that frame's positions, trace (Python engine), and reconstruct
    the RAW coherent field per detector. RNG seeds for the source sampling,
    the tracer, and the medium are FROZEN across frames so ONLY particle
    motion decorrelates the speckle. Returns {frame_idx: {fid: {keystr:
    (2,H,W) c64}}}."""
    scene = run_trace.build_scene(targs)
    lam_range = run_trace.lam_range_nm(scene)
    lam_list = particle_lams(scene, targs.nlambda)
    samples_reg = scene.optprops.samples if scene.optprops is not None else {}
    media = build_body_sample_media(
        scene, samples_reg, threshold=targs.particle_threshold,
        seed=sample_seed, lam_list=lam_list, pol_scatter=True)
    medium = media[0]
    sample_area = run_trace.compute_sample_area(scene, targs)
    # frozen per-frame RNG seeds (independent of child_seq: the child stream
    # only decorrelates nothing here — kept for the CUDA-safe spawn contract)
    src_seed = base_seed + 101
    trc_seed = base_seed + 202
    med_seed = base_seed + 303

    out = {}
    for fi, pos in zip(frame_idx_chunk, positions_chunk):
        _inject_realization(medium, pos, radii)
        medium.rng = np.random.default_rng(med_seed)
        grids = run_trace.build_detectors(scene, targs, lam_range)
        cfg = TraceConfig(max_reflections=targs.max_reflections,
                          power_floor=targs.power_floor,
                          n_lambda=targs.nlambda, rays=int(targs.rays),
                          seed=trc_seed, viz_rays=0,
                          rough_fresnel=targs.rough_fresnel)
        tracer = Tracer(scene, cfg, grids, particle_medium=[medium])
        tracer.rng = np.random.default_rng(trc_seed)
        rng_src = np.random.default_rng(src_seed)
        batches = []
        for sid, (bidx, src) in enumerate(scene.sources):
            batches.append(sample_source(
                scene, scene.bodies[bidx], src, sid, cfg.rays, cfg.n_lambda,
                rng_src, ledger=tracer.ledger))
        tracer.run(batches)
        frame_out = {}
        for fid in det_fids:
            frame_out[fid] = _field_for_detector(grids[fid], sample_area)
        out[int(fi)] = frame_out
    return out


def resolve_workers(val):
    if val == "auto":
        return max(1, (os.cpu_count() or 1) - 2)
    return max(1, int(val))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    case_dir = Path(args.case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    dls_dir = case_dir / "dls"
    dls_dir.mkdir(parents=True, exist_ok=True)
    t_wall0 = time.time()

    common.progress_emit("dls", 0.0, "building scene", case_dir=case_dir)
    targs = build_trace_args(args)
    scene = run_trace.build_scene(targs)
    require_coherent(scene)
    medium, lam_list = build_sample_medium(scene, args, args.seed)
    eta_pas = solvent_viscosity(medium)

    body_index = medium.body_index
    centers0 = np.asarray(medium.explicit.centers, dtype=np.float64)
    radii = np.asarray(medium.explicit.radii, dtype=np.float64)
    n_p = len(radii)
    dt_s = args.dt_ms * 1e-3

    # ---- detectors + q geometry -------------------------------------------
    lam_range = run_trace.lam_range_nm(scene)
    all_grids = run_trace.build_detectors(scene, targs, lam_range)
    labels_by_fid = {fid: g.label for fid, g in all_grids.items()}
    if args.detectors:
        want = {s.strip() for s in args.detectors.split(",") if s.strip()}
        det_fids = [fid for fid, lab in labels_by_fid.items() if lab in want]
        unknown = want - {labels_by_fid[fid] for fid in det_fids}
        if unknown:
            raise SystemExit(
                "run_dls.py: --detectors unknown label(s): %s — scene has: %s"
                % (", ".join(sorted(unknown)),
                   ", ".join(sorted(labels_by_fid.values())) or "(none)"))
    else:
        det_fids = list(all_grids)
    if not det_fids:
        raise SystemExit("run_dls.py: no detector faces in the scene")

    ledger_dummy = PowerLedger(len(scene.sources))
    k_i_hat, coh_src = incident_direction(scene, ledger_dummy)
    lam_m = float(coh_src["lambdac_nm"]) * 1e-9
    n_host = float(np.real(medium.host.n_complex(np.array([lam_m]))[0]))
    sample_centroid = 0.5 * (medium.lo + medium.hi)
    q_by_fid = {}
    for fid in det_fids:
        q_vec, q_mag = q_vector_for(all_grids[fid], k_i_hat, sample_centroid,
                                    n_host, lam_m)
        q_by_fid[fid] = (q_vec, q_mag)

    # canonical gather-key universe (coherent sources x strata x pol strata)
    canonical_keys = []
    for sid, (bidx, src) in enumerate(scene.sources):
        if not src.get("coherent"):
            continue
        nst = len(wavelength_strata(src, args.nlambda))
        npol = n_pol_strata(src)
        for st in range(nst):
            for ps in range(npol):
                canonical_keys.append("%d/%d/%d" % (sid, st, ps))
    nkeys = len(canonical_keys)
    key_index = {k: i for i, k in enumerate(canonical_keys)}

    # grid shape (all requested detectors are planar and share resolution
    # semantics, but H/W can differ per face aspect ratio -> keep per-fid)
    HW = {fid: (all_grids[fid].H, all_grids[fid].W) for fid in det_fids}

    # ---- file-size guard --------------------------------------------------
    bytes_est = sum(args.frames * nkeys * 2 * H * W * 8
                    for (H, W) in HW.values())
    gb_est = bytes_est / 1e9
    if gb_est > args.max_gb:
        raise SystemExit(
            "run_dls.py: estimated frames.h5 ~%.2f GB exceeds --max-gb %.2f "
            "(%d frames x %d keys x 2 pol x sum(H*W) x complex64). Lower "
            "--frames/--resolution, restrict --detectors, or raise --max-gb."
            % (gb_est, args.max_gb, args.frames, nkeys))

    print("[dls] %d frame(s), dt=%.4g ms, %d particle(s), %d detector(s), "
          "%d gather key(s), ~%.3f GB"
          % (args.frames, args.dt_ms, n_p, len(det_fids), nkeys, gb_est),
          flush=True)

    # ---- Brownian pre-generation (sequential) -----------------------------
    common.progress_emit("dls", 0.05, "Brownian pre-generation",
                         case_dir=case_dir)
    positions = brownian_frames(scene, body_index, centers0, radii,
                                args.frames, dt_s, args.temp_k, eta_pas,
                                args.seed)

    # ---- parallel frame trace ---------------------------------------------
    workers = resolve_workers(args.workers)
    n_workers = max(1, min(workers, args.frames))
    frame_ids = list(range(args.frames))
    chunks = [frame_ids[i::n_workers] for i in range(n_workers)]
    chunks = [c for c in chunks if c]
    children = np.random.SeedSequence(args.seed).spawn(len(chunks))
    sample_seed = (int(args.seed) * 1000003) & 0x7fffffff
    tasks = [
        (targs, children[i], det_fids, radii,
         [positions[f] for f in chunks[i]], chunks[i],
         sample_seed, int(args.seed))
        for i in range(len(chunks))]

    # per-detector output array (N, nkeys, 2, H, W) complex64
    frames_arr = {fid: np.zeros((args.frames, nkeys, 2) + HW[fid],
                                dtype=np.complex64) for fid in det_fids}

    common.progress_emit("dls", 0.1, "tracing %d frames" % args.frames,
                         case_dir=case_dir)
    if n_workers == 1:
        payloads = [_frame_worker(*tasks[0])]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=len(chunks)) as pool:
            payloads = pool.starmap(_frame_worker, tasks)
    done = 0
    for pl in payloads:
        for fi, per_fid in pl.items():
            for fid in det_fids:
                for keystr, arr in per_fid.get(fid, {}).items():
                    ki = key_index.get(keystr)
                    if ki is not None:
                        frames_arr[fid][fi, ki] = arr
            done += 1
        common.progress_emit("dls", 0.1 + 0.8 * done / max(args.frames, 1),
                             "traced %d/%d frames" % (done, args.frames),
                             case_dir=case_dir)

    # ---- write frames.h5 --------------------------------------------------
    common.progress_emit("dls", 0.92, "writing frames.h5", case_dir=case_dir)
    import h5py
    frames_path = dls_dir / "frames.h5"
    with h5py.File(frames_path, "w") as h:
        h["positions"] = positions
        h["radii"] = radii.astype(np.float32)
        h["dt_s"] = np.float64(dt_s)
        h["temp_k"] = np.float64(args.temp_k)
        dgrp = h.create_group("detectors")
        for fid in det_fids:
            g = all_grids[fid]
            safe = g.label.replace(".", "_").replace("/", "_")
            sub = dgrp.create_group(safe)
            sub["frames"] = frames_arr[fid]
            q_vec, q_mag = q_by_fid[fid]
            sub["q_vector"] = np.asarray(q_vec, dtype=np.float64)
            sub.attrs.update({
                "label": g.label, "H": int(g.H), "W": int(g.W),
                "pixel_m": float(g.pixel_m),
                "xhat": np.asarray(g.xhat), "yhat": np.asarray(g.yhat),
                "normal": np.asarray(g.normal),
                "x_lo": float(g.x_lo), "y_lo": float(g.y_lo),
                "lam_lo_m": float(g.lam_lo), "lam_hi_m": float(g.lam_hi),
                "q_magnitude_per_m": float(q_mag),
                "keys_json": json.dumps(canonical_keys),
            })
        h.attrs.update({
            "seed": int(args.seed),
            "engine": "python-dls",
            "sample": medium.sample_name,
            "body": medium.body_label,
            "host_material": medium.host_material_name,
            "solvent_visc_pas": float(eta_pas),
            "n_particles": int(n_p),
            "rays": float(args.rays),
            "nlambda": int(args.nlambda),
            "frames": int(args.frames),
            "dt_ms": float(args.dt_ms),
            "temp_k": float(args.temp_k),
            "k_i_hat": np.asarray(k_i_hat),
            "n_host": float(n_host),
            "lam_c_m": float(lam_m),
            "sample_row_json": json.dumps(
                {k: (list(v) if isinstance(v, (list, tuple)) else v)
                 for k, v in medium.sample_row.items()
                 if k != "sq_params"}, default=str),
        })

    wall = time.time() - t_wall0
    manifest = {
        "frames": int(args.frames),
        "dt_s": float(dt_s), "dt_ms": float(args.dt_ms),
        "temp_k": float(args.temp_k),
        "sample": medium.sample_name, "body": medium.body_label,
        "host_material": medium.host_material_name,
        "solvent_visc_pas": float(eta_pas),
        "n_particles": int(n_p),
        "detectors": [all_grids[fid].label for fid in det_fids],
        "gather_keys": canonical_keys,
        "rays": float(args.rays), "nlambda": int(args.nlambda),
        "resolution": int(args.resolution),
        "seed": int(args.seed), "engine": "python-dls",
        "wall_time_s": float(wall),
    }
    with open(dls_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("[dls] done: %d frame(s) in %s -> %s"
          % (args.frames, common.fmt_duration(wall), frames_path), flush=True)
    common.progress_emit("dls", 1.0, "completed", case_dir=case_dir,
                         status="completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
