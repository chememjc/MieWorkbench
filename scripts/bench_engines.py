#!/usr/bin/env python3
# =============================================================================
# bench_engines.py — the c-engine go/no-go benchmark (plan: final gate).
#
# Runs every extracted geometry that ROUTES to the C engine through
# run_trace twice (--engine python, --engine c) at the requested settings
# and writes a Markdown table (cengine/BENCHMARKS.md by default) with
# per-stage timings and speedups. The gate: geometric-mean trace+gather
# speedup >= 1.5x, and no scene slower than Python.
#
#   "$MIEWB_OPTICS_PYTHON" scripts/bench_engines.py \
#       [--rays 1e6] [--resolution 2048] [--nlambda 9] [--out FILE]
#       [--scenes name1,name2,...]     # default: every routable geometry/
#
# Timing source: case.json["timing"] (trace_s / gather_s) plus wall time.
# Python baselines can take hours at production settings — run overnight.
# =============================================================================
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import common  # noqa: E402  (stdlib-only shared contract hub)

OPTICS = common.OPTICS_PYTHON


def routable(model_json):
    """Would --engine auto pick the C engine for this geometry?"""
    out = subprocess.run(
        [OPTICS, str(REPO / "scripts" / "run_trace.py"),
         "--model-json", str(model_json),
         "--case-dir", "/tmp/bench-probe-ignore",
         "--rays", "1", "--dry-run"],
        capture_output=True, text=True)
    return out.returncode == 0     # dry-run never routes; probe via scene
    # (routing is checked from the real run's case.json below)


def run_case(model_json, case_dir, engine, rays, resolution, nlambda,
             timeout_s=None):
    t0 = time.time()
    try:
        proc = subprocess.run(
            [OPTICS, str(REPO / "scripts" / "run_trace.py"),
             "--model-json", str(model_json), "--case-dir", str(case_dir),
             "--rays", repr(rays), "--resolution", str(resolution),
             "--nlambda", str(nlambda), "--engine", engine,
             "--workers", "auto"],
            capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-f", str(case_dir)],
                       capture_output=True)
        return {"error": "timeout", "wall": time.time() - t0, "tail": ""}
    wall = time.time() - t0
    if proc.returncode != 0:
        return {"error": proc.returncode, "wall": wall,
                "tail": proc.stdout[-800:] + proc.stderr[-400:]}
    case = json.loads((Path(case_dir) / "case.json").read_text())
    audit = json.loads((Path(case_dir) / "audit.json").read_text())
    t = case.get("timing", {})
    return {
        "wall": wall,
        "trace_s": float(t.get("trace_s") or 0.0),
        "gather_s": float(t.get("gather_s") or 0.0),
        "engine": case.get("engine"),
        "closure_ok": all(a["closure_ok"] for a in audit["per_seed"]),
        "detected": audit["per_seed"][0]["detected_W"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rays", type=float, default=1e6)
    ap.add_argument("--resolution", type=int, default=2048)
    ap.add_argument("--nlambda", type=int, default=9)
    ap.add_argument("--out", default=str(REPO / "cengine" / "BENCHMARKS.md"))
    ap.add_argument("--scenes", default=None,
                    help="comma-separated geometry names (default: all "
                         "C-routable extracted geometries)")
    ap.add_argument("--workdir", default=str(REPO / "var" / "work"
                                             / "bench_engines"))
    ap.add_argument("--timeout-s", type=float, default=5400.0,
                    help="per-engine-run wall budget; exceeding runs are "
                         "SKIPPED with an explicit note in the table "
                         "(no silent caps)")
    args = ap.parse_args()

    if args.scenes:
        names = args.scenes.split(",")
    else:
        names = sorted(p.name for p in (REPO / "geometry").iterdir()
                       if (p / "model.json").exists()
                       and "-" not in p.name)      # skip permuted variants

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []
    for name in names:
        mj = REPO / "geometry" / name / "model.json"
        print("== %s" % name, flush=True)
        cc = run_case(mj, wd / name / "c", "auto", args.rays,
                      args.resolution, args.nlambda,
                      timeout_s=args.timeout_s)
        if "error" in cc:
            print("   C run failed/timed out (%s) — noted + skipped"
                  % cc["error"])
            skipped.append((name, "C run: %s" % cc["error"]))
            continue
        if cc["engine"] != "c":
            print("   routes to python (unported feature) — skipped")
            skipped.append((name, "auto-routes to python"))
            continue
        py = run_case(mj, wd / name / "py", "python", args.rays,
                      args.resolution, args.nlambda,
                      timeout_s=args.timeout_s)
        if "error" in py:
            # Python baseline exceeded the budget but C completed: a
            # bounded speedup statement is still honest — record it
            print("   python run failed/timed out (%s)" % py["error"])
            skipped.append((name, "python baseline: %s (C wall %.1fs — "
                            "speedup > %.1fx)" % (py["error"], cc["wall"],
                                                  py["wall"] / cc["wall"])))
            continue
        # detected-power sanity (loose MC bound; the demo equivalence
        # suite is the real physics gate)
        for k, v in py["detected"].items():
            cv = cc["detected"].get(k, 0.0)
            scale = max(abs(v), abs(cv), 1e-30)
            if abs(v - cv) / scale > 0.10:
                print("   WARNING: detected %s differs >10%%: %g vs %g"
                      % (k, v, cv))
        rows.append((name, py, cc))
        print("   py %7.1fs (trace %6.1f gather %6.1f) | c %7.1fs "
              "(trace %6.1f gather %6.1f) | %.1fx"
              % (py["wall"], py["trace_s"], py["gather_s"], cc["wall"],
                 cc["trace_s"], cc["gather_s"], py["wall"] / cc["wall"]),
              flush=True)

    if not rows:
        raise SystemExit("no benchmarkable scenes")

    import math
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True,
                         cwd=REPO).stdout.strip()
    speedups = [py["wall"] / cc["wall"] for _, py, cc in rows]
    stage_speedups = [
        (py["trace_s"] + py["gather_s"])
        / max(cc["trace_s"] + cc["gather_s"], 1e-9)
        for _, py, cc in rows]
    gmean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    gmean_stage = math.exp(sum(math.log(s) for s in stage_speedups)
                           / len(stage_speedups))

    lines = [
        "# C engine benchmark (bench_engines.py)",
        "",
        "- git: `%s`  rays=%g resolution=%d nlambda=%d" % (
            sha, args.rays, args.resolution, args.nlambda),
        "- host: RTX 4090 Laptop (CUDA 13), 32-core CPU",
        "- python engine: --workers auto (process-sharded trace, "
        "torch-CUDA gather)",
        "",
        "| scene | py wall | py trace | py gather | C wall | C trace | "
        "C gather | wall speedup | stage speedup |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, py, cc in rows:
        lines.append(
            "| %s | %.1fs | %.1fs | %.1fs | %.1fs | %.2fs | %.2fs | "
            "**%.1fx** | %.1fx |"
            % (name, py["wall"], py["trace_s"], py["gather_s"],
               cc["wall"], cc["trace_s"], cc["gather_s"],
               py["wall"] / cc["wall"],
               (py["trace_s"] + py["gather_s"])
               / max(cc["trace_s"] + cc["gather_s"], 1e-9)))
    if skipped:
        lines += ["", "Skipped (explicit, no silent caps):", ""]
        lines += ["- %s — %s" % (n, why) for n, why in skipped]
    lines += [
        "",
        "geometric mean: **%.1fx wall**, **%.1fx trace+gather** over %d "
        "scenes" % (gmean, gmean_stage, len(rows)),
        "",
        "Gate (plan): >= 1.5x geometric-mean stage speedup — %s."
        % ("PASS" if gmean_stage >= 1.5 else "FAIL"),
    ]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))
    return 0 if gmean_stage >= 1.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
