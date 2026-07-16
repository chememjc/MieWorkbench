#!/usr/bin/env python3
"""run_pipeline.py — top-level orchestrator for the optical ray tracer
pipeline.

Runs under **plain system python3** (stdlib only): unlike the stage scripts
it touches none of the FreeCAD / optics-env / ParaView python stacks
itself. It only composes argv lists and launches each stage as a
subprocess, each under its own pinned interpreter (see common.py):

    0. permute  FreeCAD headless   scripts/permute_model.py   (FREECAD)
                (only if --var is given: sweeps one base model into a
                cross-product of variant .FCStd files in basemodels/)
    1. extract  FreeCAD headless   scripts/extract_geometry.py (FREECAD)
    2. trace    optics env python  scripts/run_trace.py        (OPTICS_PYTHON)
    3. post     optics env python  scripts/post_process.py     (OPTICS_PYTHON)
    4. viz      ParaView pvpython  scripts/make_viz.py         (PVPYTHON)

Key design decisions (cloned from the antenna project's run_pipeline.py):
  * extract runs ONCE for the whole model batch (one FreeCAD launch handles
    every .FCStd file, original or permuted); trace/post/viz then loop per
    model SEQUENTIALLY — a single trace run can already saturate every
    core/GPU, so running models in parallel would only oversubscribe the
    machine.
  * The case directory name is computed here with the SAME rule
    common.case_name() uses (results/<model_stem>/<preset>[-<tag>]/), so
    post/viz always look in the directory trace actually wrote to.
  * Physics options (--rays/--resolution/--nlambda/--seeds/--backend/
    --source-face/--detector-face/--grating/--rough/--particles/
    --particle-threshold/--suppress-body/--save-fields/
    --save-fields-detectors/--dry-run) are forwarded to the trace stage
    ONLY, verbatim; --preset fills rays/resolution/nlambda/spectral-bins/
    viz-rays from common.PRESETS unless explicitly overridden on this
    command line.
  * --viz-generations forwards to the post stage; --views/--smoke forward
    to the viz stage (post_process.py / make_viz.py's own options of the
    same name — see their --help for exact semantics).
  * --dry-run means "trace estimates only, does not actually run": trace's
    case.json status then stays 'estimated' (never 'completed'), so post
    and viz are skipped for that model with a NOTICE — this is enforced
    generically via the case.json status gate (common.read_case_status),
    not a special-cased dry-run branch, so it also covers any other reason
    trace didn't reach 'completed'.
  * --keep-going turns a stage failure into a FAILED notice + skip-to-
    next-model instead of an abort; the process still exits nonzero if
    anything failed.
  * --print-only composes and prints every stage command WITHOUT
    executing anything.

Logs: extract/permute (batch-level) log to results/log.<step>; trace/post/
viz (per model) log to results/<model_stem>/<case>/log.<step>.
"""

import json
import subprocess
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (stdlib-only shared contract hub)
import cli_specs  # noqa: E402  (stdlib-only; single source of truth for CLIs)

STEPS_ORDER = ["extract", "trace", "post", "viz"]
GLOB_CHARS = "*?["


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = cli_specs.build_parser("pipeline")
    return p.parse_args(argv)


def validate_var_counts(args):
    counts = {"--var": len(args.var), "--min": len(args.min),
              "--max": len(args.max), "--n": len(args.n)}
    if len(set(counts.values())) != 1:
        raise SystemExit(
            "run_pipeline.py: --var/--min/--max/--n must appear the same "
            "number of times (got %s)" % counts)
    return list(zip(args.var, args.min, args.max, args.n))


# ---------------------------------------------------------------------------
# Model / step resolution
# ---------------------------------------------------------------------------
def resolve_steps(spec):
    requested = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STEPS_ORDER]
    if unknown:
        raise SystemExit(
            "run_pipeline.py: unknown step(s) %s — valid steps are: %s"
            % (", ".join(unknown), ", ".join(STEPS_ORDER)))
    if not requested:
        raise SystemExit("run_pipeline.py: --steps is empty")
    return [s for s in STEPS_ORDER if s in requested]


def expand_models(patterns):
    """Expand globs, resolve to absolute paths, dedup preserving order.
    Returns a list of Path objects."""
    resolved = []
    for pat in patterns:
        matches = sorted(glob(pat))
        if matches:
            candidates = matches
        elif any(c in pat for c in GLOB_CHARS):
            raise SystemExit("run_pipeline.py: pattern %r matched no files"
                             % pat)
        else:
            candidates = [pat]  # literal; existence checked below
        for m in candidates:
            path = Path(m)
            if not path.is_absolute():
                path = Path.cwd() / path
            path = path.resolve()
            if not path.exists():
                # bare model names may still resolve under BASEMODELS_DIR
                alt = common.BASEMODELS_DIR / Path(m).name
                if alt.exists():
                    path = alt
                else:
                    raise SystemExit(
                        "run_pipeline.py: file not found: %s" % path)
            resolved.append(path)

    out, seen = [], set()
    for path in resolved:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    if not out:
        raise SystemExit("run_pipeline.py: no models to process")
    return out


def variant_output_names(stem, varspecs, sweep_mode="product"):
    """Replicate permute_model.py's naming loop EXACTLY (same
    common.sweep_values / common.variant_name calls, same --var order, and
    combination order from common.sweep_combos(value_lists, sweep_mode))
    so run_pipeline can predict the variant .FCStd filenames
    permute_model.py will write without parsing its stdout. Because both
    call sites route combination order through the single
    common.sweep_combos function, the two can never drift out of sync
    again — sweep_mode must match what was passed to permute_cmd()."""
    value_lists = [common.sweep_values(vmin, vmax, n)
                   for (_, vmin, vmax, n) in varspecs]
    names = [v[0] for v in varspecs]
    out = []
    for combo in common.sweep_combos(value_lists, mode=sweep_mode):
        out_name = stem
        for var, value in zip(names, combo):
            out_name = common.variant_name(out_name, var, value)
        out.append(out_name)
    return out


# ---------------------------------------------------------------------------
# Per-stage command builders (single source of truth for flag forwarding)
# ---------------------------------------------------------------------------
def permute_cmd(model_path, varspecs, sweep_mode="product"):
    cmd = [common.FREECAD_APPIMAGE, "-c",
           str(common.SCRIPTS_DIR / "permute_model.py"), "--",
           "--model", str(model_path)]
    for var, vmin, vmax, n in varspecs:
        cmd += ["--var", var, "--min", repr(vmin), "--max", repr(vmax),
               "--n", str(n)]
    cmd += ["--outdir", str(common.BASEMODELS_DIR)]
    if sweep_mode != "product":
        cmd += ["--sweep-mode", sweep_mode]
    return cmd


def extract_cmd(model_paths):
    cmd = [common.FREECAD_APPIMAGE, "-c",
           str(common.SCRIPTS_DIR / "extract_geometry.py"), "--",
           "--models"] + [str(p) for p in model_paths]
    return cmd


def _preset_val(args, key, attr):
    v = getattr(args, attr)
    return v if v is not None else common.PRESETS[args.preset][key]


def trace_cmd(stem, case_dir, args):
    model_json = common.GEOMETRY_DIR / stem / "model.json"
    cmd = [common.OPTICS_PYTHON, str(common.SCRIPTS_DIR / "run_trace.py"),
           "--model-json", str(model_json), "--case-dir", str(case_dir),
           "--rays", repr(_preset_val(args, "rays", "rays")),
           "--resolution", str(int(_preset_val(args, "resolution",
                                                "resolution"))),
           "--nlambda", str(int(_preset_val(args, "nlambda", "nlambda"))),
           "--spectral-bins", str(int(_preset_val(
               args, "spectral_bins", "spectral_bins")))]
    # viz budget: explicit --viz-rays wins; otherwise density-driven with
    # the preset's viz_rays as the per-source cap
    if args.viz_rays is not None:
        cmd += ["--viz-rays", str(int(args.viz_rays))]
    else:
        cmd += ["--viz-density",
                repr(args.viz_density if args.viz_density is not None
                     else 1.0),
                "--viz-rays-max",
                str(int(common.PRESETS[args.preset]["viz_rays"]))]
    if args.viz_pattern is not None:
        common.parse_viz_pattern_spec(args.viz_pattern)  # fail fast here
        cmd += ["--viz-pattern", args.viz_pattern]
    if args.max_reflections is not None:
        cmd += ["--max-reflections", str(int(args.max_reflections))]
    cmd += ["--seeds", str(args.seeds if args.seeds is not None
                           else common.DEFAULTS["seeds"])]
    cmd += ["--backend", args.backend if args.backend is not None
           else common.DEFAULTS["backend"]]
    if args.engine is not None:
        cmd += ["--engine", args.engine]
    if getattr(args, "resume", False):
        cmd += ["--resume"]
    if getattr(args, "extend", None) is not None:
        cmd += ["--extend", repr(args.extend)]
    if args.importance_aim:
        cmd += ["--importance-aim"]
    if getattr(args, "importance_scatter", False):
        cmd += ["--importance-scatter"]
    if getattr(args, "importance_limit", 1.0) != 1.0:
        cmd += ["--importance-limit", repr(float(args.importance_limit))]
    cmd += ["--workers", str(args.workers)]
    for f in args.source_face:
        cmd += ["--source-face", f]
    for f in args.detector_face:
        cmd += ["--detector-face", f]
    for g in args.grating:
        cmd += ["--grating", g]
    for r in args.rough:
        cmd += ["--rough", r]
    if args.particles:
        cmd += ["--particles", args.particles]
    if args.particle_threshold is not None:
        cmd += ["--particle-threshold", repr(args.particle_threshold)]
    for b in args.suppress_body:
        cmd += ["--suppress-body", b]
    if args.rough_fresnel is not None:
        cmd += ["--rough-fresnel", args.rough_fresnel]
    if args.ray_differentials:
        cmd += ["--ray-differentials"]
    if args.gather_occlusion:
        cmd += ["--gather-occlusion"]
    if args.gather_exact:
        cmd += ["--gather-exact"]
    if getattr(args, "gather_nufft", False):
        cmd += ["--gather-nufft"]
    if args.no_pol_scatter:
        cmd += ["--no-pol-scatter"]
    if args.mesh_flat_normals:
        cmd += ["--mesh-flat-normals"]
    if getattr(args, "temperature", None) is not None:
        cmd += ["--temperature", repr(float(args.temperature))]
    if args.save_fields:
        cmd += ["--save-fields"]
    if args.save_fields_detectors:
        cmd += ["--save-fields-detectors", args.save_fields_detectors]
    if args.export_rays:
        cmd += ["--export-rays"]
    if args.ghost_analysis:
        cmd += ["--ghost-analysis"]
    if args.export_rays or args.ghost_analysis:
        cmd += ["--export-rays-max", str(int(args.export_rays_max))]
    if args.strict_analytic:
        cmd += ["--strict-analytic"]
    if args.optical_properties:
        cmd += ["--optical-properties", args.optical_properties]
    # pulsed-optics time products: forward the selection verbatim (the
    # empty tuple round-trips as 'none'); --time-bins is preset-scaled
    # when not given explicitly (cli_specs.TIME_BINS_PRESET — the flags
    # are inert on a CW scene with no --time-products, so forwarding the
    # bins default unconditionally never changes existing physics).
    if args.time_products is not None:
        cmd += ["--time-products", ",".join(args.time_products) or "none"]
    cmd += ["--time-bins", str(int(
        args.time_bins if args.time_bins is not None
        else cli_specs.TIME_BINS_PRESET[args.preset]))]
    if args.time_window is not None:
        cmd += ["--time-window",
                "%g,%g" % (args.time_window[0], args.time_window[1])]
    if args.time_cube_res != 256:
        cmd += ["--time-cube-res", str(int(args.time_cube_res))]
    if args.time_envelope != "analytic":
        cmd += ["--time-envelope", args.time_envelope]
    if args.gdd_budget:
        cmd += ["--gdd-budget"]
    if args.dry_run:
        cmd += ["--dry-run"]
    return cmd


def _dim_rays_args(args):
    """--dim-rays/--dim-rays-floor forwarding shared by post and viz;
    empty at the defaults so existing command lines stay unchanged."""
    if args.dim_rays == "off":
        return []
    cmd = ["--dim-rays", args.dim_rays]
    if args.dim_rays_floor:
        cmd += ["--dim-rays-floor", repr(args.dim_rays_floor)]
    return cmd


def post_cmd(stem, case_dir, args):
    model_json = common.GEOMETRY_DIR / stem / "model.json"
    cmd = [common.OPTICS_PYTHON, str(common.SCRIPTS_DIR / "post_process.py"),
          "--case-dir", str(case_dir), "--model-json", str(model_json)] \
        + _dim_rays_args(args)
    if args.photometric:
        cmd += ["--photometric"]
    if args.spectrometer:
        cmd += ["--spectrometer"]
    if args.instruments is not None:
        cmd += ["--instruments", args.instruments]
    if args.emit_csv:
        cmd += ["--emit-csv"]
    if args.wavefront_point is not None:
        cmd += ["--wavefront-point",
                "%g,%g" % (args.wavefront_point[0], args.wavefront_point[1])]
    if args.wavefront_pupil != "source":
        cmd += ["--wavefront-pupil", args.wavefront_pupil]
    if args.imaging_products:
        cmd += ["--imaging-products", ",".join(args.imaging_products)]
    if args.image_sim:
        cmd += ["--image-sim", args.image_sim]
        if args.image_sim_coherence != "incoherent":
            cmd += ["--image-sim-coherence", args.image_sim_coherence]
        if args.image_sim_sigma != 0.5:
            cmd += ["--image-sim-sigma", repr(float(args.image_sim_sigma))]
    if args.viz_generations is not None:
        cmd += ["--viz-generations", str(int(args.viz_generations))]
    return cmd


def viz_cmd(stem, case_dir, args):
    model_json = common.GEOMETRY_DIR / stem / "model.json"
    cmd = [common.PVPYTHON, "--force-offscreen-rendering",
          str(common.SCRIPTS_DIR / "make_viz.py"),
          "--case-dir", str(case_dir), "--model-json", str(model_json)] \
        + _dim_rays_args(args)
    if args.views:
        cmd += ["--views", args.views]
    if args.smoke:
        cmd += ["--smoke"]
    return cmd


STAGE_BUILDERS = {"trace": trace_cmd, "post": post_cmd, "viz": viz_cmd}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_logged(cmd, log_path, stdin_devnull=False):
    """Run `cmd`, tee-ing combined stdout/stderr to both this process's
    stdout and `log_path`. Raises subprocess.CalledProcessError on a
    nonzero exit."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, stdin=(subprocess.DEVNULL if stdin_devnull else None),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def _run_stage(cmd, log_path, tag, failures, stdin_devnull=False):
    try:
        run_logged(cmd, log_path, stdin_devnull=stdin_devnull)
        return True
    except subprocess.CalledProcessError as exc:
        print("FAILED: %s (exit %s)" % (tag, exc.returncode), flush=True)
        failures.append(tag)
        return False


def resolve_models(args):
    """Expand --models, then (if --var given) run permute_model.py per
    input model and replace it with its cross-product of variant .FCStd
    paths under basemodels/. Returns (list of Path, failures, aborted)."""
    models = expand_models(args.models)
    varspecs = validate_var_counts(args)
    failures = []
    if not varspecs:
        return models, failures, False

    final = []
    for model_path in models:
        stem = model_path.stem
        cmd = permute_cmd(model_path, varspecs, args.sweep_mode)
        log = common.RESULTS_DIR / ("log.permute-%s" % stem)
        if not _run_stage(cmd, log, "permute (%s)" % stem, failures,
                          stdin_devnull=True):
            if args.keep_going:
                continue
            return final, failures, True
        for name in variant_output_names(stem, varspecs, args.sweep_mode):
            final.append(common.BASEMODELS_DIR / (name + ".FCStd"))
    return final, failures, False


def run_pipeline(args, steps, model_paths, case):
    """Execute the requested stages for the resolved model list. Returns
    (failures, aborted)."""
    failures = []
    stems = [p.stem for p in model_paths]

    # pipeline-level progress: one unit per (model, step), extract counts 1
    per_model = [s for s in steps if s != "extract"]
    total_units = (1 if "extract" in steps else 0) \
        + len(stems) * len(per_model)
    done_units = 0

    def tick(msg):
        common.progress_emit("pipeline", done_units / max(1, total_units),
                             msg)

    if "extract" in steps:
        tick("extract (batch of %d)" % len(stems))
        cmd = extract_cmd(model_paths)
        log = common.RESULTS_DIR / "log.extract"
        if not _run_stage(cmd, log, "extract (batch)", failures,
                          stdin_devnull=True) and not args.keep_going:
            return failures, True
        done_units += 1

    for stem in stems:
        case_dir = common.case_dir(stem, case)
        model_failed = False
        for step in per_model:
            if step in ("post", "viz"):
                status = common.read_case_status(case_dir / "case.json")
                if status != "completed":
                    print("NOTICE: skipping %s for %s — case status is "
                          "%r (need 'completed'; trace was --dry-run, "
                          "failed, or has not run yet)"
                          % (step, stem, status), flush=True)
                    done_units += 1
                    continue
            tick("%s/%s" % (stem, step))
            cmd = STAGE_BUILDERS[step](stem, case_dir, args)
            log = case_dir / ("log.%s" % step)
            if not _run_stage(cmd, log, "%s/%s" % (stem, step), failures):
                model_failed = True
                if args.keep_going:
                    break            # skip the rest of this model
                return failures, True
            done_units += 1
        if model_failed and not args.keep_going:
            return failures, True
    common.progress_emit("pipeline", 1.0,
                         "all requested stages finished",
                         status="completed" if not failures else "failed")
    return failures, False


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def detector_power_cell(report):
    if not isinstance(report, dict) or not report.get("detectors"):
        return "-"
    parts = []
    for label, d in sorted(report["detectors"].items()):
        p_mw = d.get("total_power_W")
        if p_mw is None:
            continue
        parts.append("%s=%.3gmW" % (label, p_mw * 1e3))
    return ", ".join(parts) if parts else "-"


def summary_row(stem, case):
    case_dir = common.case_dir(stem, case)
    status = common.read_case_status(case_dir / "case.json")
    report = _load_json(case_dir / "report.json")
    return [stem, case, status, detector_power_cell(report)]


def print_summary(stems, case):
    header = ["model", "case", "status", "detected power"]
    rows = [summary_row(stem, case) for stem in stems]
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows
              else len(header[i]) for i in range(len(header))]

    def fmt(cols):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    line = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print("\n" + line)
    print(fmt(header))
    print(line)
    for r in rows:
        print(fmt(r))
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    steps = resolve_steps(args.steps)
    varspecs = validate_var_counts(args)
    if args.imaging_products and not (args.export_rays
                                      or args.ghost_analysis):
        # same style as the post stage's own gate: the imaging products
        # are computed from rays_full.npz, which only --export-rays (or
        # --ghost-analysis, which implies it) makes the trace write.
        raise SystemExit(
            "run_pipeline.py: --imaging-products requires --export-rays "
            "(the distortion/vignetting/field-curves/telecentricity "
            "products are computed from rays_full.npz)")
    if args.image_sim and not args.save_fields:
        # same up-front gate style: the image simulation's amplitude PSF
        # is read from a detector's saved coherent field map, which only
        # --save-fields makes the trace write.
        raise SystemExit(
            "run_pipeline.py: --image-sim requires --save-fields (the "
            "system's amplitude PSF is read from the dominant coherent "
            "gather key's saved detector field map)")
    case = common.case_name(args.preset, args.tag)

    if args.print_only:
        models = expand_models(args.models)
        print("run_pipeline.py: %d input model(s) [%s], steps=%s, case=%s"
              % (len(models), ", ".join(p.stem for p in models),
                 ",".join(steps), case), flush=True)
        print("# print-only: commands that WOULD run (no execution)")
        stems = [p.stem for p in models]
        if varspecs:
            for model_path in models:
                print("+ " + " ".join(
                    permute_cmd(model_path, varspecs, args.sweep_mode)))
            stems = [n for p in models
                    for n in variant_output_names(p.stem, varspecs,
                                                  args.sweep_mode)]
            model_paths = [common.BASEMODELS_DIR / (s + ".FCStd")
                          for s in stems]
            print("#   (permute produces variant model(s): %s)"
                  % ", ".join(stems))
        else:
            model_paths = models
        if "extract" in steps:
            print("+ " + " ".join(extract_cmd(model_paths)))
        for stem in stems:
            case_dir = common.case_dir(stem, case)
            for step in (s for s in steps if s != "extract"):
                print("+ " + " ".join(STAGE_BUILDERS[step](
                    stem, case_dir, args)))
        return 0

    model_paths, failures, aborted = resolve_models(args)
    if aborted:
        print_summary([p.stem for p in model_paths], case)
        print("\n%d stage(s) FAILED: %s (aborted)"
              % (len(failures), ", ".join(failures)), flush=True)
        return 1
    stems = [p.stem for p in model_paths]

    print("run_pipeline.py: %d model(s) [%s], steps=%s, case=%s"
          % (len(model_paths), ", ".join(stems), ",".join(steps), case),
          flush=True)

    more_failures, aborted = run_pipeline(args, steps, model_paths, case)
    failures += more_failures
    print_summary(stems, case)

    if failures:
        print("\n%d stage(s) FAILED: %s%s"
              % (len(failures), ", ".join(failures),
                 "" if not aborted else " (aborted)"), flush=True)
        return 1
    print("\nAll requested stages completed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
