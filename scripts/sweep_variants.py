#!/usr/bin/env python3
"""sweep_variants.py — run several run_pipeline.py jobs back-to-back, then
overlay their results with compare_runs.py.

Runs under **plain system python3** (stdlib only): it only builds and
launches subprocesses (run_pipeline.py under python3, compare_runs.py
under OPTICS_PYTHON), exactly like run_pipeline.py itself. It imports
run_pipeline.py directly (also stdlib-only, same interpreter) to reuse its
model-glob-expansion and --var variant-naming logic, so the case
directories this script predicts for the final compare_runs.py call are
computed by the EXACT SAME code path run_pipeline.py itself uses — no
parallel reimplementation to drift out of sync.

Jobs run SEQUENTIALLY — a single trace run can already saturate every
core/GPU, so running variants in parallel would only oversubscribe the
machine.

Two ways to specify the batch:
  --jobs jobs.json    a JSON list of per-job dicts; each dict's keys are
                      run_pipeline.py option names WITHOUT the leading
                      dashes and with dashes turned to underscores, e.g.
                        [{"models": "example.FCStd", "preset": "quick"},
                         {"models": "example.FCStd", "preset": "normal",
                          "tag": "hires"}]
                      List-valued run_pipeline flags (source_face,
                      detector_face, grating, rough, suppress_body, var,
                      min, max, n) may be given as a JSON list.
  --job "k=v,k=v"     shorthand, repeatable: one job per --job flag,
                      inline "key=value,key=value" pairs (scalar values
                      only -- use --jobs for list-valued flags like
                      --var/--source-face).

Common flags on THIS command line seed every job; a per-job entry
overrides them. After the batch, compare_runs.py overlays every case
directory that actually produced a report.json (--no-compare skips this).

--keep-going lets one failing job print FAILED and move on instead of
aborting; the process still exits nonzero if anything failed.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                    # noqa: E402
import run_pipeline as rp        # noqa: E402  (reuse its pure helpers)

RUN_PIPELINE = common.SCRIPTS_DIR / "run_pipeline.py"
COMPARE = common.SCRIPTS_DIR / "compare_runs.py"

# job keys -> run_pipeline.py flag (scalar, single value)
SCALAR_KEYS = ["preset", "tag", "steps", "seeds", "rays", "resolution",
              "nlambda", "spectral_bins", "viz_rays", "backend",
              "particles", "particle_threshold"]
# job keys -> run_pipeline.py flag (repeatable, list value)
LIST_KEYS = ["source_face", "detector_face", "grating", "rough",
            "suppress_body"]
# job keys -> run_pipeline.py boolean flag (no value)
FLAG_KEYS = ["dry_run"]
# parameter-sweep triplet keys (parallel lists, paired positionally)
VAR_KEYS = ["var", "min", "max", "n"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("job selection (choose one form)")
    src.add_argument("--jobs", metavar="JSON",
                     help="JSON file: a list of per-job option dicts")
    src.add_argument("--job", action="append", default=[],
                     metavar="k=v,k=v", dest="inline_jobs",
                     help="inline 'key=value,key=value' job (repeatable, "
                          "scalar values only)")

    g = p.add_argument_group("common defaults seeded into every job "
                             "(a --jobs/--job entry overrides these)")
    g.add_argument("--models", nargs="+", default=None, metavar="FCSTD")
    for key in SCALAR_KEYS:
        g.add_argument("--%s" % key.replace("_", "-"), default=None)
    g.add_argument("--dry-run", action="store_true", default=False)

    g = p.add_argument_group("execution / orchestration")
    g.add_argument("--keep-going", action="store_true",
                   help="continue after a failing job (exit nonzero at "
                        "the end)")
    g.add_argument("--no-compare", action="store_true",
                   help="skip the final compare_runs.py overlay")
    g.add_argument("--compare-out", default=None,
                   help="--out forwarded to the final compare_runs.py call")
    g.add_argument("--print-only", action="store_true",
                   help="print the composed commands without running them")
    return p.parse_args(argv)


def base_defaults(args):
    d = {}
    if args.models:
        d["models"] = list(args.models)
    for key in SCALAR_KEYS:
        val = getattr(args, key)
        if val is not None:
            d[key] = val
    if args.dry_run:
        d["dry_run"] = True
    return d


def parse_inline_job(spec):
    """'k=v,k=v' -> {k: v} (string values; run_pipeline_cmd stringifies
    everything anyway)."""
    out = {}
    for kv in spec.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise SystemExit("sweep_variants.py: bad --job entry %r "
                             "(expected key=value)" % kv)
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_jobs(args):
    base = base_defaults(args)
    jobs = []
    if args.jobs:
        raw = json.loads(Path(args.jobs).read_text())
        if not isinstance(raw, list):
            raise SystemExit("--jobs must be a JSON list of objects")
        for entry in raw:
            if not isinstance(entry, dict):
                raise SystemExit("--jobs entries must be objects")
            j = dict(base)
            j.update(entry)
            jobs.append(j)
    for spec in args.inline_jobs:
        j = dict(base)
        j.update(parse_inline_job(spec))
        jobs.append(j)
    return jobs


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def pipeline_cmd(job, keep_going):
    models = _as_list(job.get("models"))
    if not models:
        raise SystemExit("sweep_variants.py: job missing required "
                         "'models' key: %r" % job)
    cmd = [sys.executable, str(RUN_PIPELINE), "--models"] + \
        [str(m) for m in models]
    for key in SCALAR_KEYS:
        val = job.get(key)
        if val in (None, ""):
            continue
        cmd += ["--%s" % key.replace("_", "-"), str(val)]
    for key in LIST_KEYS:
        for v in _as_list(job.get(key)):
            cmd += ["--%s" % key.replace("_", "-"), str(v)]
    for key in FLAG_KEYS:
        if job.get(key):
            cmd += ["--%s" % key.replace("_", "-")]

    varspecs = job_varspecs(job)
    for var, vmin, vmax, n in varspecs:
        cmd += ["--var", str(var), "--min", repr(vmin), "--max", repr(vmax),
               "--n", str(n)]
    if keep_going:
        cmd += ["--keep-going"]
    return cmd


def job_varspecs(job):
    """Parallel (var, min, max, n) triplets from a job dict, list-valued
    or scalar (a job with exactly one swept var may give plain scalars)."""
    if not job.get("var"):
        return []
    vs = _as_list(job["var"])
    mins = _as_list(job.get("min"))
    maxs = _as_list(job.get("max"))
    ns = _as_list(job.get("n"))
    counts = {len(vs), len(mins), len(maxs), len(ns)}
    if len(counts) != 1:
        raise SystemExit("sweep_variants.py: job's var/min/max/n must have "
                         "matching lengths: %r" % job)
    return list(zip(vs, [float(x) for x in mins], [float(x) for x in maxs],
                    [int(x) for x in ns]))


def job_case_dirs(job):
    """Predicted results/<model_stem>/<case> directories for a job, using
    run_pipeline.py's OWN expand_models/variant_output_names/case_name so
    this never drifts out of sync with what run_pipeline.py actually
    writes to."""
    models = _as_list(job.get("models"))
    paths = rp.expand_models(models)
    stems = [p.stem for p in paths]
    varspecs = job_varspecs(job)
    if varspecs:
        stems = [n for stem in stems
                for n in rp.variant_output_names(stem, varspecs)]
    case = common.case_name(job.get("preset", "quick"), job.get("tag"))
    return [common.case_dir(s, case) for s in stems]


def main():
    args = parse_args()
    jobs = load_jobs(args)
    if not jobs:
        raise SystemExit("sweep_variants.py: no jobs to run (use --jobs "
                         "or --job)")

    print("sweep_variants.py: %d job(s)" % len(jobs))
    cmds = [pipeline_cmd(j, args.keep_going) for j in jobs]
    case_dirs_per_job = [job_case_dirs(j) for j in jobs]

    if args.print_only:
        for i, (cmd, dirs) in enumerate(zip(cmds, case_dirs_per_job), 1):
            print("+ job %d/%d: %s" % (i, len(jobs), " ".join(cmd)))
            for d in dirs:
                print("    -> %s" % d)
        all_dirs = [d for dirs in case_dirs_per_job for d in dirs]
        if not args.no_compare and all_dirs:
            cmp_cmd = [common.OPTICS_PYTHON, str(COMPARE), "--cases"] + \
                [str(d) for d in all_dirs]
            if args.compare_out:
                cmp_cmd += ["--out", args.compare_out]
            print("+ " + " ".join(cmp_cmd))
        return 0

    # TODO(P3 persistent worker, REGISTRY.md §6): each variant already
    # amortizes the C-engine worker WITHIN its case — run_pipeline.py ->
    # run_trace.py -> cengine.run_c_case spawns ONE `miewb-trace --serve`
    # child that serves every chunk trace + the final gather for that case
    # (V chunks x S seeds pay one context init). Reusing a SINGLE worker
    # ACROSS variants is deliberately NOT done here: this loop spawns
    # run_pipeline.py subprocesses (extract -> trace -> post -> viz, each a
    # different geometry/case), and sweep_variants.py is system-python3
    # stdlib-ONLY by contract while cengine.Worker lives in the optics env —
    # driving the binary directly across variants would cross that
    # interpreter boundary and duplicate run_trace's request-building.
    # Interface if ever wanted: a long-lived cengine.Worker whose stdin is
    # fed request.json paths (worker.run(path) -> rc); every variant's
    # per-seed/per-chunk request.json is already on disk under
    # <case>/cengine/, so a future optics-env driver could pool one worker
    # over the whole sweep. The per-case amortization already captures the
    # dominant win (context + buffer-pool reuse).
    failures = []
    produced = []
    for i, (cmd, dirs) in enumerate(zip(cmds, case_dirs_per_job), 1):
        tag = "job %d/%d" % (i, len(jobs))
        log = common.RESULTS_DIR / ("log.sweep-job%02d" % i)
        log.parent.mkdir(parents=True, exist_ok=True)
        print("=== %s: %s ===" % (tag, " ".join(cmd)), flush=True)
        try:
            with open(log, "w") as logf:
                subprocess.run(cmd, check=True, stdout=logf,
                              stderr=subprocess.STDOUT)
            produced += dirs
        except subprocess.CalledProcessError as exc:
            print("FAILED: %s (exit %s) — see %s"
                  % (tag, exc.returncode, log), flush=True)
            failures.append(tag)
            if not args.keep_going:
                break

    ran = [d for d in produced if (d / "report.json").exists()]
    if not args.no_compare and ran:
        cmp_cmd = [common.OPTICS_PYTHON, str(COMPARE), "--cases"] + \
            [str(d) for d in ran]
        if args.compare_out:
            cmp_cmd += ["--out", args.compare_out]
        log = common.RESULTS_DIR / "log.sweep-compare"
        print("=== compare: %s ===" % " ".join(cmp_cmd), flush=True)
        try:
            with open(log, "w") as logf:
                subprocess.run(cmp_cmd, check=True, stdout=logf,
                              stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            print("FAILED: compare (exit %s) — see %s" % (exc.returncode, log),
                  flush=True)
            failures.append("compare")
    elif not args.no_compare:
        print("NOTICE: no job produced a report.json — skipping compare")

    if failures:
        print("\n%d step(s) FAILED: %s" % (len(failures), ", ".join(failures)),
              flush=True)
        return 1
    print("\nsweep complete: %d job(s) produced results." % len(ran),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
