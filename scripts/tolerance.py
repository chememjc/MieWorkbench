#!/usr/bin/env python
# =============================================================================
# tolerance.py — sensitivity analysis + Monte-Carlo yield tolerancing (with
# an optional per-draw focus compensator) over the fast evaluator.
#
# Interpreter: "$MIEWB_OPTICS_PYTHON" (the OPTICS env: fast_eval.py
# shells out to FreeCAD / run_trace.py / post_process.py under their own
# pinned interpreters, and the nested compensator reuses optimize.py's
# scipy engine).
#
# Three phases over one shared fast_eval.Evaluator session:
#
#   nominal       one evaluation of the as-designed parameter set (every
#                 tolerance at its NOMINAL, the compensator at its START).
#                 The baseline merit every sensitivity/impact number is
#                 relative to.
#
#   sensitivity   for each --tolerance NAME:NOMINAL:DIST:BAND, evaluate
#                 the merit at NOMINAL +/- (sens_delta * BAND) and report
#                   derivative = (M+ - M-) / (2*delta)      [central diff]
#                   impact     = max(|M+ - M0|, |M- - M0|)  [ranking key]
#                 The table is RANKED BY IMPACT, not the derivative: a
#                 design toleranced at its merit minimum (the usual case —
#                 e.g. defocus at best focus) has derivative ~ 0 while the
#                 band still costs real merit; impact is the number a
#                 tolerancing engineer budgets with. The derivative column
#                 still carries the signed local gradient.
#
#   monte-carlo   --draws N random perturbation sets, each tolerance
#                 sampled from its distribution (normal(mu=NOMINAL,
#                 sigma=BAND) or uniform(NOMINAL +/- BAND)), merit
#                 evaluated per draw. With --compensator VAR:LO:HI each
#                 draw first runs a nested optimize.OptimizationEngine
#                 (local Nelder-Mead, --comp-budget evals) over VAR with
#                 the perturbations held fixed — the recorded merit is the
#                 COMPENSATED one, exactly how an as-built system is
#                 refocused before test. Aggregates: merit distribution
#                 stats, a histogram, and (given --merit-threshold X) the
#                 yield fraction = draws with merit <= X, over ALL draws
#                 (a failed evaluation counts as a failed unit).
#
# A failed/incomplete evaluation is PENALIZED (optimize.PENALTY), never
# fatal — penalized draws are excluded from the distribution stats and
# histogram but count against yield.
#
# Progress: '@MIEWB {...}' lines (stage "tolerance") via
# common.progress_emit when MIEWB_PROGRESS=1 — per sensitivity parameter
# (extra phase="sensitivity", plus one phase="sensitivity_done" event
# carrying the compact ranked table the GUI bar chart renders), and per
# Monte-Carlo draw with frac = draws done / N (extras: draw/draws/merit/
# yield-so-far/params — the GUI histogram consumes these). Output:
# <out>/report.json with the full sensitivity table, per-draw detail and
# the aggregated Monte-Carlo block.
#
# CLI is built by cli_specs.build_parser("tolerance"); --config JSON
# mirrors the CLI (explicit CLI flags win).
#
# Typical use:
#     "$MIEWB_OPTICS_PYTHON" scripts/tolerance.py \
#         --model tolerance_lens.FCStd \
#         --tolerance lenspos:0:normal:0.5 --tolerance lensdy:0:normal:0.5 \
#         --operand spot_rms:0:1 --draws 40 --merit-threshold 1.2e5 \
#         --compensator detpos:40:60
# =============================================================================
import json
import math
import random
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import cli_specs  # noqa: E402  (stdlib-only; the one CLI authority)
import common     # noqa: E402  (stdlib-only shared contract hub)
import optimize   # noqa: E402  (MeritFunction/OptimizationEngine/PENALTY)

PENALTY = optimize.PENALTY


class ToleranceError(RuntimeError):
    """Configuration/setup error (bad tolerances, nothing to do)."""


# ---------------------------------------------------------------------------
# the tolerancing engine (evaluate_fn injectable -> unit-testable)
# ---------------------------------------------------------------------------
class ToleranceEngine:
    """Owns the three phases (nominal / sensitivity / Monte-Carlo), the
    penalized-merit scoring and the eval count. evaluate_fn(params_dict)
    -> fast_eval-shaped output dict (may raise); injectable, so the
    statistics are unit-testable against analytic merits without a trace.

    tolerances : list of {"name","nominal","dist","band"} dicts
        (cli_specs.parse_tolerance_spec).
    operands : optimize.MeritFunction operand specs.
    compensator : optional {"name","start","lo","hi"}
        (cli_specs.parse_compensator_spec); its name must not collide
        with a tolerance name.
    """

    def __init__(self, tolerances, operands, evaluate_fn, draws=50,
                 threshold=None, compensator=None, comp_budget=10,
                 sens_delta=1.0, hist_bins=20, mc_seed=42, case_dir=None,
                 progress=True):
        if not tolerances:
            raise ToleranceError("at least one tolerance is required")
        self.tolerances = [dict(t) for t in tolerances]
        names = [t["name"] for t in self.tolerances]
        if len(set(names)) != len(names):
            raise ToleranceError("duplicate tolerance names: %s" % names)
        for t in self.tolerances:
            if t["dist"] not in cli_specs.TOLERANCE_DISTS:
                raise ToleranceError("unknown distribution %r on %r"
                                     % (t["dist"], t["name"]))
            if not t["band"] > 0.0:
                raise ToleranceError("tolerance %r: band must be > 0"
                                     % t["name"])
        self.merit = optimize.MeritFunction(operands)
        self.evaluate_fn = evaluate_fn
        self.draws = int(draws)
        if self.draws < 0:
            raise ToleranceError("draws must be >= 0")
        self.threshold = None if threshold is None else float(threshold)
        self.compensator = dict(compensator) if compensator else None
        if self.compensator and self.compensator["name"] in names:
            raise ToleranceError(
                "compensator %r is also a tolerance parameter — it would "
                "just undo its own perturbation; compensate with a "
                "DIFFERENT variable" % self.compensator["name"])
        self.comp_budget = int(comp_budget)
        self.sens_delta = float(sens_delta)
        if not self.sens_delta > 0.0:
            raise ToleranceError("sens-delta must be > 0")
        self.hist_bins = max(1, int(hist_bins))
        self.mc_seed = int(mc_seed)
        self.case_dir = case_dir
        self.progress = progress

        # every evaluation carries the FULL parameter vector (fast_eval
        # locks the param set): tolerances at nominal + the compensator
        # at its resting START value
        self.base_params = {t["name"]: float(t["nominal"])
                            for t in self.tolerances}
        if self.compensator:
            self.base_params[self.compensator["name"]] = float(
                self.compensator["start"])

        self.nominal = None       # {"params","merit","operands",...}
        self.sensitivity = []     # ranked rows
        self.mc = None            # aggregated Monte-Carlo block
        self.n_evals = 0

    # -- one scored evaluation ---------------------------------------------------
    def _score(self, params):
        """(merit, rows, penalized, note): a failed evaluation or missing
        operand is PENALIZED, never fatal."""
        note = None
        try:
            out = self.evaluate_fn(dict(params))
            merit, rows, missing = self.merit.score(out)
            penalized = bool(missing)
            if missing:
                note = "; ".join(str(m) for m in missing)
        except Exception as exc:
            merit, rows, penalized = PENALTY, [], True
            note = "evaluation failed: %s: %s" % (type(exc).__name__, exc)
        self.n_evals += 1
        return merit, rows, penalized, note

    def _emit(self, frac, msg, **extra):
        if self.progress:
            common.progress_emit("tolerance", frac, msg,
                                 case_dir=self.case_dir, extra=extra)

    # -- phase 1: nominal ---------------------------------------------------------
    def run_nominal(self):
        self._emit(0.0, "evaluating the nominal design",
                   phase="nominal")
        merit, rows, penalized, note = self._score(self.base_params)
        self.nominal = {"params": dict(self.base_params), "merit": merit,
                        "operands": rows, "penalized": penalized,
                        "note": note}
        if penalized:
            raise ToleranceError(
                "the NOMINAL design failed to evaluate (%s) — nothing to "
                "tolerance against" % note)
        return self.nominal

    # -- phase 2: sensitivity -----------------------------------------------------
    def run_sensitivity(self):
        """Finite-difference each tolerance at nominal +/- delta; rank by
        merit impact over the band (see the module docstring for why the
        ranking key is impact, not the signed derivative)."""
        if self.nominal is None:
            self.run_nominal()
        m0 = self.nominal["merit"]
        rows = []
        n = len(self.tolerances)
        for i, tol in enumerate(self.tolerances):
            delta = self.sens_delta * tol["band"]
            merits = {}
            for sign, key in ((+1.0, "plus"), (-1.0, "minus")):
                params = dict(self.base_params)
                params[tol["name"]] = tol["nominal"] + sign * delta
                merit, _, penalized, note = self._score(params)
                merits[key] = {"merit": merit, "penalized": penalized,
                               "note": note}
            usable = not (merits["plus"]["penalized"]
                          or merits["minus"]["penalized"])
            mp, mm = merits["plus"]["merit"], merits["minus"]["merit"]
            row = {
                "name": tol["name"], "nominal": tol["nominal"],
                "dist": tol["dist"], "band": tol["band"], "delta": delta,
                "merit_nominal": m0,
                "merit_plus": mp if not merits["plus"]["penalized"]
                else None,
                "merit_minus": mm if not merits["minus"]["penalized"]
                else None,
                "derivative": ((mp - mm) / (2.0 * delta) if usable
                               else None),
                "impact": (max(abs(mp - m0), abs(mm - m0)) if usable
                           else PENALTY),
                "penalized": not usable,
                "note": (merits["plus"]["note"]
                         or merits["minus"]["note"]),
            }
            rows.append(row)
            self._emit(None, "sensitivity %d/%d: %s impact=%.6g"
                       % (i + 1, n, tol["name"],
                          row["impact"] if usable else float("nan")),
                       phase="sensitivity", param=tol["name"],
                       impact=(row["impact"] if usable else None),
                       derivative=row["derivative"])
        # a penalized parameter ranks LAST (impact=PENALTY would rank it
        # first and lie); sort usable rows by impact descending
        usable_rows = [r for r in rows if not r["penalized"]]
        broken_rows = [r for r in rows if r["penalized"]]
        usable_rows.sort(key=lambda r: -r["impact"])
        self.sensitivity = usable_rows + broken_rows
        for rank, row in enumerate(self.sensitivity, start=1):
            row["rank"] = rank
        self._emit(None, "sensitivity ranking: %s"
                   % " > ".join(r["name"] for r in self.sensitivity),
                   phase="sensitivity_done",
                   sensitivity=[{"name": r["name"], "rank": r["rank"],
                                 "impact": (None if r["penalized"]
                                            else r["impact"]),
                                 "derivative": r["derivative"]}
                                for r in self.sensitivity])
        return self.sensitivity

    # -- phase 3: Monte-Carlo yield ------------------------------------------------
    def _draw_params(self, rng):
        params = dict(self.base_params)
        for tol in self.tolerances:
            if tol["dist"] == "normal":
                params[tol["name"]] = rng.gauss(tol["nominal"],
                                                tol["band"])
            else:  # uniform (validated in __init__)
                params[tol["name"]] = rng.uniform(
                    tol["nominal"] - tol["band"],
                    tol["nominal"] + tol["band"])
        return params

    def _compensate(self, draw_params):
        """Nested focus-compensator optimization for one draw: optimize
        the compensator variable with the perturbations held fixed; the
        draw's merit is the recovered best. Returns (merit, penalized,
        note, comp_record)."""
        comp = self.compensator
        fixed = dict(draw_params)

        def evaluate(cp):
            merged = dict(fixed)
            merged.update(cp)
            return self.evaluate_fn(merged)

        eng = optimize.OptimizationEngine(
            [{"name": comp["name"], "start": comp["start"],
              "lo": comp["lo"], "hi": comp["hi"]}],
            self.merit.operands, evaluate, algorithm="local",
            budget=self.comp_budget, progress=False)
        try:
            best = eng.run()
        except optimize.OptimizeError as exc:
            self.n_evals += eng.n_evals
            return PENALTY, True, "compensation failed: %s" % exc, {
                "name": comp["name"], "value": None,
                "evals": eng.n_evals}
        self.n_evals += eng.n_evals
        return best["merit"], False, None, {
            "name": comp["name"],
            "value": best["params"][comp["name"]],
            "evals": eng.n_evals}

    def run_monte_carlo(self):
        if self.nominal is None:
            self.run_nominal()
        rng = random.Random(self.mc_seed)
        detail = []
        n_pass = 0
        for i in range(1, self.draws + 1):
            params = self._draw_params(rng)
            if self.compensator:
                merit, penalized, note, comp = self._compensate(params)
            else:
                merit, _, penalized, note = self._score(params)
                comp = None
            passed = (self.threshold is not None and not penalized
                      and merit <= self.threshold)
            if passed:
                n_pass += 1
            entry = {"draw": i, "params": {k: params[k] for k
                                           in self.base_params},
                     "merit": merit, "penalized": penalized, "note": note,
                     "passed": (passed if self.threshold is not None
                                else None),
                     "compensator": comp}
            detail.append(entry)
            yield_so_far = (n_pass / i if self.threshold is not None
                            else None)
            self._emit(i / self.draws,
                       "draw %d/%d merit=%.6g%s"
                       % (i, self.draws, merit,
                          ("" if yield_so_far is None
                           else " yield=%.3f" % yield_so_far)),
                       phase="mc", draw=i, draws=self.draws, merit=merit,
                       penalized=penalized,
                       merit_yield=yield_so_far,
                       params=entry["params"])
        self.mc = self._aggregate(detail, n_pass)
        return self.mc

    def _aggregate(self, detail, n_pass):
        ok = [e["merit"] for e in detail if not e["penalized"]]
        n_penalized = len(detail) - len(ok)
        stats = None
        histogram = None
        if ok:
            mean = sum(ok) / len(ok)
            var = (sum((m - mean) ** 2 for m in ok) / (len(ok) - 1)
                   if len(ok) > 1 else 0.0)
            srt = sorted(ok)

            def pct(p):
                if len(srt) == 1:
                    return srt[0]
                x = p * (len(srt) - 1)
                lo = int(math.floor(x))
                hi = min(lo + 1, len(srt) - 1)
                return srt[lo] + (x - lo) * (srt[hi] - srt[lo])

            stats = {"n": len(ok), "mean": mean,
                     "std": math.sqrt(var), "min": srt[0], "max": srt[-1],
                     "p10": pct(0.10), "p50": pct(0.50), "p90": pct(0.90)}
            histogram = self._histogram(ok)
        return {
            "draws": self.draws, "seed": self.mc_seed,
            "threshold": self.threshold,
            "compensated": bool(self.compensator),
            "n_penalized": n_penalized,
            "n_pass": (n_pass if self.threshold is not None else None),
            "yield_fraction": (n_pass / self.draws
                               if self.threshold is not None
                               and self.draws else None),
            "stats": stats,
            "histogram": histogram,
            "detail": detail,
        }

    def _histogram(self, merits):
        lo, hi = min(merits), max(merits)
        if hi <= lo:                     # all draws identical: one fat bin
            pad = abs(lo) * 1e-9 + 1e-12
            lo, hi = lo - pad, hi + pad
        nb = self.hist_bins
        width = (hi - lo) / nb
        edges = [lo + k * width for k in range(nb)] + [hi]
        counts = [0] * nb
        for m in merits:
            k = min(int((m - lo) / width), nb - 1)
            counts[k] += 1
        return {"bin_edges": edges, "counts": counts}


# ---------------------------------------------------------------------------
# orchestration (real fast_eval evaluator + report writing)
# ---------------------------------------------------------------------------
def _apply_config_file(parser, args):
    """--config JSON: values fill in wherever the CLI kept the parser
    default (explicit CLI flags win). tolerance/operand/compensator
    entries may be spec strings (parsed here) or ready dicts."""
    if not args.config:
        return
    with open(args.config) as fh:
        cfg = json.load(fh)
    for key, value in cfg.items():
        dest = key.replace("-", "_")
        if dest == "config" or not hasattr(args, dest):
            raise SystemExit("--config: unknown key %r" % key)
        if getattr(args, dest) != parser.get_default(dest):
            continue                       # explicit CLI wins
        if dest == "tolerance":
            value = [cli_specs.parse_tolerance_spec(v)
                     if isinstance(v, str) else v for v in value]
        elif dest == "operand":
            value = [cli_specs.parse_operand_spec(v) if isinstance(v, str)
                     else v for v in value]
        elif dest == "compensator" and isinstance(value, str):
            value = cli_specs.parse_compensator_spec(value)
        setattr(args, dest, value)


def _write_report(out_dir, args, engine, wall_s, status="completed",
                  error=None):
    report = {
        "model": str(args.model),
        "tolerances": engine.tolerances,
        "operands": engine.merit.operands,
        "compensator": engine.compensator,
        "comp_budget": args.comp_budget,
        "draws": args.draws,
        "merit_threshold": engine.threshold,
        "sens_delta": args.sens_delta,
        "hist_bins": args.hist_bins,
        "mc_seed": args.mc_seed,
        "preset": args.preset,
        "eval_backend": args.eval_backend,
        "nominal": engine.nominal,
        "sensitivity": engine.sensitivity,
        "mc": engine.mc,
        "n_evals": engine.n_evals,
        "wall_s": round(wall_s, 3),
        "status": status,
    }
    if error:
        report["error"] = error
    common.write_json(out_dir / "report.json", report)
    return report


def main(argv=None):
    parser = cli_specs.build_parser("tolerance")
    args = parser.parse_args(argv)
    _apply_config_file(parser, args)
    if not args.tolerance:
        parser.error("at least one --tolerance NAME:NOMINAL:DIST:BAND is "
                     "required")
    if not args.operand:
        parser.error("at least one --operand OPERAND:TARGET:WEIGHT is "
                     "required")
    if args.draws <= 0 and args.skip_sensitivity:
        parser.error("--draws 0 with --skip-sensitivity leaves nothing "
                     "to do")

    from fast_eval import Evaluator   # optics env; imports fcclient

    model_stem = Path(args.model).stem
    out_dir = Path(args.out) if args.out else (
        PROJECT_DIR / "var" / "tolerance" / model_stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequential = args.eval_backend == "sequential"
    if sequential:
        mc_ops = optimize.mc_only_operands(args.operand)
        if mc_ops:
            parser.error(
                "operand(s) %s need the Monte-Carlo backend and cannot be "
                "served by --eval-backend sequential; drop them or use "
                "--eval-backend worker" % ", ".join(sorted(set(mc_ops))))

    needs = optimize.operand_needs(args.operand)
    trace_args = []
    keep_coherent = False
    if not sequential:
        if "export_rays" in needs:
            trace_args.append("--export-rays")
        if "save_fields" in needs:
            trace_args.append("--save-fields")
            keep_coherent = True

    names = [t["name"] for t in args.tolerance]
    param_names = list(names)
    if args.compensator:
        param_names.append(args.compensator["name"])
    print("[tolerance] model=%s tolerances=%s operands=%s draws=%d "
          "compensator=%s"
          % (args.model, names, [o["operand"] for o in args.operand],
             args.draws,
             args.compensator["name"] if args.compensator else "none"),
          flush=True)
    common.progress_emit("tolerance", 0.0,
                         "starting tolerancing (%d parameter(s), %d "
                         "draw(s))" % (len(names), args.draws),
                         case_dir=out_dir)

    evaluator = Evaluator(args.model, params=param_names,
                          backend=args.eval_backend,
                          keep_coherent=keep_coherent,
                          trace_args=trace_args, workdir=args.workdir,
                          preset=args.preset, rays=args.rays,
                          resolution=args.resolution,
                          nlambda=args.nlambda, seeds=args.seeds,
                          seed0=args.seed0)
    engine = ToleranceEngine(
        args.tolerance, args.operand, evaluator.evaluate,
        draws=args.draws, threshold=args.merit_threshold,
        compensator=args.compensator, comp_budget=args.comp_budget,
        sens_delta=args.sens_delta, hist_bins=args.hist_bins,
        mc_seed=args.mc_seed, case_dir=out_dir)
    t0 = time.monotonic()
    try:
        with evaluator:
            engine.run_nominal()
            if not args.skip_sensitivity:
                engine.run_sensitivity()
            if args.draws > 0:
                engine.run_monte_carlo()
    except (ToleranceError, optimize.OptimizeError) as exc:
        common.progress_emit("tolerance", None, str(exc),
                             case_dir=out_dir, status="failed")
        print("[tolerance] FAILED: %s" % exc, flush=True)
        _write_report(out_dir, args, engine, wall_s=time.monotonic() - t0,
                      status="failed", error=str(exc))
        return 1

    report = _write_report(out_dir, args, engine,
                           wall_s=time.monotonic() - t0)
    mc = report["mc"] or {}
    yf = mc.get("yield_fraction")
    common.progress_emit(
        "tolerance", 1.0,
        "done: %d evals%s" % (engine.n_evals,
                              ("" if yf is None
                               else ", yield %.3f" % yf)),
        case_dir=out_dir, status="completed",
        extra={"n_evals": engine.n_evals, "merit_yield": yf,
               "nominal_merit": engine.nominal["merit"]})
    if engine.sensitivity:
        print("[tolerance] sensitivity ranking (by merit impact over the "
              "band):", flush=True)
        for row in engine.sensitivity:
            print("[tolerance]   %d. %-20s impact=%-12.6g d(merit)/d(%s)="
                  "%s" % (row["rank"], row["name"],
                          row["impact"] if not row["penalized"]
                          else float("nan"), row["name"],
                          ("%.6g" % row["derivative"])
                          if row["derivative"] is not None else "n/a"),
                  flush=True)
    if mc:
        stats = mc.get("stats") or {}
        print("[tolerance] Monte-Carlo: %d draws, %d penalized, merit "
              "mean=%.6g p50=%.6g p90=%.6g"
              % (mc["draws"], mc["n_penalized"],
                 stats.get("mean", float("nan")),
                 stats.get("p50", float("nan")),
                 stats.get("p90", float("nan"))), flush=True)
        if yf is not None:
            print("[tolerance] yield: %d/%d = %.3f (merit <= %g)"
                  % (mc["n_pass"], mc["draws"], yf, mc["threshold"]),
                  flush=True)
    print("[tolerance] %d evaluations in %.1f s -> %s"
          % (engine.n_evals, report["wall_s"], out_dir / "report.json"),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
