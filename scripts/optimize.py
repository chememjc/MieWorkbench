#!/usr/bin/env python
# =============================================================================
# optimize.py — merit-function optimizer over the fast evaluator.
#
# Interpreter: /home3/optics/env/bin/python (the OPTICS env: scipy +
# nevergrad/cma live there, and fast_eval.py shells out to FreeCAD /
# run_trace.py / post_process.py under their own pinned interpreters).
#
# Drives one or more design variables (spreadsheet cell aliases — exactly
# what permute_model --var / fast_eval address) to minimize a weighted
# scalar merit built from operands read off the pipeline's report.json:
#
#   spot_rms / focus    detector spot RMS radius [um] (report.json
#                       detectors.<label>.spot rows; needs --export-rays,
#                       which this script adds to the trace automatically).
#                       Multiple (source, lambda-stratum) rows combine as
#                       the n_rays-weighted RMS. MINIMIZED.
#   encircled_energy    ee_r80_um from the coherent field analysis
#                       (detectors.<label>.analysis; needs --save-fields
#                       AND a coherent gather, so the INNER loop runs
#                       keep_coherent=True for this operand — slow).
#                       MINIMIZED.
#   mtf50               mtf50_tan_cy_mm from the same analysis block.
#                       MAXIMIZED.
#   detected_power      detectors.<label>.total_power_W (summed over the
#                       matched detectors). MAXIMIZED.
#   <raw.merit.key>     any flattened report.json merit key (fast_eval.
#                       flatten_merits naming, e.g.
#                       'Body003.Pad001.Face5.total_power_W'). MINIMIZED
#                       toward its target.
#
# Merit: sum over operands of weight*(value-target)^2 for minimize
# operands (and for maximize operands with a nonzero target — "reach this
# value"); a pure maximize operand (target 0) contributes -weight*value.
# A failed/incomplete evaluation is PENALIZED (PENALTY), never fatal.
#
# Algorithms: --algorithm local = scipy Nelder-Mead within the variable
# bounds; --algorithm global = nevergrad CMA-ES. Both work in normalized
# [0,1] coordinates so differently-scaled variables condition equally.
#
# Inner loop: fast_eval.Evaluator (backend "worker" by default) with every
# source patched incoherent (unless an operand needs the coherent field
# analysis), so the Huygens gather never runs. At the end the best design
# is RE-EVALUATED once with keep_coherent=True for a faithful final
# number (skippable via --no-final-coherent; a final-eval failure — e.g.
# the gather's undersampling gate at low ray budgets — is recorded in the
# report, never fatal).
#
# Progress: '@MIEWB {...}' lines (stage "optimize", frac = evals/budget)
# via common.progress_emit when MIEWB_PROGRESS=1, plus a progress.json
# heartbeat in the --out dir. Output: <out>/report.json with the full
# convergence history (per-eval merit + best-so-far + params) and the
# final best design.
#
# CLI is built by cli_specs.build_parser("optimize"); --config JSON mirrors
# the CLI (explicit CLI flags win).
#
# Typical use:
#     /home3/optics/env/bin/python scripts/optimize.py \
#         --model example.FCStd --var lenspos:-4:-8:8 \
#         --operand spot_rms:0:1 --algorithm local --budget 20
# =============================================================================
import json
import math
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import cli_specs  # noqa: E402  (stdlib-only; the one CLI authority)
import common     # noqa: E402  (stdlib-only shared contract hub)

# merit assigned to a candidate whose evaluation failed or whose report is
# missing a requested operand — large enough that any real design beats it,
# finite so the optimizers keep moving instead of dying
PENALTY = 1e9

# operands whose merit value is better when LARGER
_MAXIMIZE = {"detected_power", "mtf50"}

# operand -> extra fast_eval needs
_NEEDS = {
    "spot_rms": ("export_rays",),
    "focus": ("export_rays",),
    "encircled_energy": ("save_fields",),
    "mtf50": ("save_fields",),
}

# ---------------------------------------------------------------------------
# Operand -> evaluator-backend routing (engine3.md Sec 8: "sequential operands
# evaluate on the Optiland trace; MC-only operands stay on fast_eval's path").
# SINGLE source of truth for which operands the deterministic sequential
# backend can serve; the GUI and optimize.py both read it.
#
#   spot_rms / focus    -> sequential: RMS of Optiland ray landings at the
#                          detector plane (deterministic, noise-free).
#   encircled_energy    -> sequential: a GEOMETRIC ray-density encircled radius
#                          (ee_r80 proxy), NOT the diffraction PSF integral.
#   mtf50               -> MC only: needs the coherent field analysis.
#   detected_power      -> MC only: needs full MC energy transport
#                          (Fresnel/scatter/absorption ledger).
#   <raw report.json>   -> MC only: an arbitrary MC-produced report field.
# ---------------------------------------------------------------------------
_SEQUENTIAL_OK = {"spot_rms", "focus", "encircled_energy"}


def operand_backend(operand):
    """'sequential' if the deterministic Optiland trace can serve this operand,
    else 'mc' (needs the Monte-Carlo pipeline)."""
    return "sequential" if operand in _SEQUENTIAL_OK else "mc"


def mc_only_operands(operands):
    """The operand names in the list that ONLY the MC backend can serve."""
    return [o["operand"] for o in operands
            if operand_backend(o["operand"]) == "mc"]


class OptimizeError(RuntimeError):
    """Configuration/setup error (bad operands, no evaluations ran)."""


# ---------------------------------------------------------------------------
# operand extraction from one fast_eval output dict
# ---------------------------------------------------------------------------
def _match_detectors(detectors, wanted):
    """Detector blocks matching an operand's optional @DETECTOR qualifier
    (exact label first, then unambiguous dotted-suffix, e.g. 'Face5' or
    'Pad001.Face5' for 'Body003.Pad001.Face5')."""
    if not wanted:
        return dict(detectors)
    if wanted in detectors:
        return {wanted: detectors[wanted]}
    hits = {label: block for label, block in detectors.items()
            if label.endswith("." + wanted)}
    return hits


def _combined_spot_rms(block):
    """detectors.<label>.spot rows -> the n_rays-weighted combined RMS
    radius [um] (sqrt of the ray-weighted mean square), or None when the
    block is absent/empty (no landed rays, or --export-rays missing).
    Prefers the POWER-WEIGHTED per-row RMS (rms_pw_radius_um): the plain
    geometric RMS counts near-zero-power ghost/stray records equally and
    then barely responds to focus; the energy-weighted one is the merit
    a designer means (falls back to rms_radius_um on older reports)."""
    rows = block.get("spot") or []
    num = den = 0.0
    for row in rows:
        try:
            r = float(row.get("rms_pw_radius_um",
                              row.get("rms_radius_um")))
            n = float(row.get("n_rays", 0.0))
        except (TypeError, ValueError):
            continue
        n = n if n > 0 else 1.0
        num += n * r * r
        den += n
    if den <= 0:
        return None
    return math.sqrt(num / den)


def operand_value(out, spec):
    """One operand's scalar for one evaluation output. Returns (value,
    note): value None when the operand cannot be read (note says why)."""
    op = spec["operand"]
    detectors = out.get("detectors") or {}

    if op not in cli_specs.OPTIMIZE_OPERANDS:
        # raw flattened merit key
        merits = out.get("merits") or {}
        if op in merits:
            return float(merits[op]), None
        return None, "merit key %r not in report" % op

    blocks = _match_detectors(detectors, spec.get("detector"))
    if not blocks:
        return None, ("no detector matches %r (have: %s)"
                      % (spec.get("detector"), sorted(detectors) or "none"))

    values = []
    for label, block in sorted(blocks.items()):
        if op in ("spot_rms", "focus"):
            v = _combined_spot_rms(block)
        elif op == "encircled_energy":
            v = (block.get("analysis") or {}).get("ee_r80_um")
        elif op == "mtf50":
            v = (block.get("analysis") or {}).get("mtf50_tan_cy_mm")
        elif op == "detected_power":
            v = block.get("total_power_W")
        else:  # pragma: no cover - registry and OPTIMIZE_OPERANDS agree
            return None, "unhandled operand %r" % op
        if v is not None:
            values.append(float(v))
    if not values:
        return None, ("operand %r has no value on detector(s) %s (spot/"
                      "analysis blocks need --export-rays/--save-fields "
                      "artifacts and landed rays)" % (op, sorted(blocks)))
    if op == "detected_power":
        return float(sum(values)), None
    return float(sum(values) / len(values)), None


class MeritFunction:
    """Weighted scalar merit over a list of operand specs (each
    {"operand","detector","target","weight"} as parse_operand_spec
    yields). score(out) -> (merit, rows, missing_notes)."""

    def __init__(self, operands):
        if not operands:
            raise OptimizeError("at least one operand is required")
        self.operands = [dict(spec) for spec in operands]

    def score(self, out):
        merit = 0.0
        rows = []
        missing = []
        for spec in self.operands:
            v, note = operand_value(out, spec)
            row = {"operand": spec["operand"],
                   "detector": spec.get("detector"),
                   "target": spec["target"], "weight": spec["weight"],
                   "value": v, "contribution": None}
            if v is None:
                missing.append(note)
            else:
                w, t = spec["weight"], spec["target"]
                if spec["operand"] in _MAXIMIZE and t == 0.0:
                    c = -w * v
                else:
                    c = w * (v - t) ** 2
                row["contribution"] = c
                merit += c
            rows.append(row)
        if missing:
            return PENALTY, rows, missing
        return merit, rows, missing

    def dls_supported(self):
        """(ok, reason): DLS represents the merit as a sum of squared
        residuals sqrt(w)*(v-target), so a PURE-maximize operand (in _MAXIMIZE
        with target 0, whose merit is the non-square -w*v) has no residual
        form. Everything else is DLS-representable."""
        for spec in self.operands:
            if spec["operand"] in _MAXIMIZE and spec["target"] == 0.0:
                return False, ("operand %r is a pure-maximize (target 0); "
                               "damped least-squares needs a squared-residual "
                               "form -- give it a nonzero target, or use "
                               "--algorithm simplex/global" % spec["operand"])
        return True, None

    def residuals(self, out):
        """(resid, rows, missing): the least-squares residual VECTOR aligned to
        the operands, r_i = sqrt(w_i)*(v_i - target_i), so sum(r_i^2) == score()
        for the DLS-supported operand set. A missing operand fills a large
        finite sentinel so the LM step backs off (never NaN/Inf, never fatal)."""
        resid = []
        rows = []
        missing = []
        for spec in self.operands:
            v, note = operand_value(out, spec)
            row = {"operand": spec["operand"],
                   "detector": spec.get("detector"),
                   "target": spec["target"], "weight": spec["weight"],
                   "value": v, "contribution": None}
            if v is None:
                missing.append(note)
                resid.append(None)
            else:
                w, t = spec["weight"], spec["target"]
                r = math.sqrt(max(w, 0.0)) * (v - t)
                resid.append(r)
                row["contribution"] = r * r
            rows.append(row)
        if missing:
            big = math.sqrt(PENALTY / max(1, len(self.operands)))
            resid = [big if r is None else r for r in resid]
        return resid, rows, missing


def operand_needs(operands):
    """Union of the fast_eval extras the operand set requires."""
    needs = set()
    for spec in operands:
        needs.update(_NEEDS.get(spec["operand"], ()))
    return needs


# ---------------------------------------------------------------------------
# the optimizer loop (algorithm-agnostic objective + two drivers)
# ---------------------------------------------------------------------------
class OptimizationEngine:
    """Owns the objective (normalized coords -> penalized scalar merit),
    the eval memo, the convergence history and the best-so-far tracking.
    evaluate_fn(params_dict) -> fast_eval-shaped output dict (may raise);
    injectable, so the loop is unit-testable against an analytic bowl
    without a trace."""

    def __init__(self, variables, operands, evaluate_fn, algorithm="local",
                 budget=40, tol=1e-3, seed=42, case_dir=None,
                 progress=True):
        if not variables:
            raise OptimizeError("at least one variable is required")
        self.variables = [dict(v) for v in variables]
        self.merit = MeritFunction(operands)
        self.evaluate_fn = evaluate_fn
        self.algorithm = algorithm
        self.budget = int(budget)
        if self.budget < 1:
            raise OptimizeError("budget must be >= 1")
        self.tol = float(tol)
        self.seed = int(seed)
        self.case_dir = case_dir
        self.progress = progress

        self.history = []
        self.best = None          # history entry of the best real merit
        self.n_evals = 0
        self._memo = {}           # rounded param tuple -> merit

    # -- coordinate mapping ----------------------------------------------------
    def _denorm(self, x):
        params = {}
        for xi, v in zip(x, self.variables):
            xi = min(1.0, max(0.0, float(xi)))
            params[v["name"]] = v["lo"] + xi * (v["hi"] - v["lo"])
        return params

    def _x0(self):
        return [(v["start"] - v["lo"]) / (v["hi"] - v["lo"])
                for v in self.variables]

    # -- the objective -----------------------------------------------------------
    def _record(self, params, merit, rows, penalized, note, out):
        """Append one convergence-history entry, update best-so-far and emit
        progress. Shared by the scalar (_objective) and residual (_residuals)
        paths so the two algorithms produce identical bookkeeping."""
        self.n_evals += 1
        entry = {"eval": self.n_evals, "params": params, "merit": merit,
                 "operands": rows, "penalized": penalized, "note": note,
                 "backend_used": (out or {}).get("backend_used")}
        if not penalized and (self.best is None
                              or merit < self.best["merit"]):
            self.best = entry
        entry["best_merit"] = self.best["merit"] if self.best else None
        self.history.append(entry)
        if self.progress:
            best = self.best["merit"] if self.best else float("nan")
            common.progress_emit(
                "optimize", self.n_evals / self.budget,
                "eval %d/%d merit=%.6g best=%.6g"
                % (self.n_evals, self.budget, merit, best),
                case_dir=self.case_dir,
                extra={"eval": self.n_evals, "budget": self.budget,
                       "merit": merit,
                       "best": self.best["merit"] if self.best else None,
                       "params": params,
                       "best_params": (self.best["params"]
                                       if self.best else None)})
        return entry

    def _objective(self, x):
        params = self._denorm(x)
        key = tuple(round(params[v["name"]], 12) for v in self.variables)
        if key in self._memo:
            return self._memo[key]
        if self.n_evals >= self.budget:
            # budget exhausted: keep the driver's bookkeeping happy without
            # paying for another trace
            return self.best["merit"] if self.best else PENALTY

        note = None
        out = None
        try:
            out = self.evaluate_fn(params)
            merit, rows, missing = self.merit.score(out)
            penalized = bool(missing)
            if missing:
                note = "; ".join(str(m) for m in missing)
        except Exception as exc:   # a single failed eval never kills the run
            merit, rows, penalized = PENALTY, [], True
            note = "evaluation failed: %s: %s" % (type(exc).__name__, exc)

        self._record(params, merit, rows, penalized, note, out)
        self._memo[key] = merit
        return merit

    def _residuals(self, x):
        """Residual VECTOR for scipy least_squares (the DLS driver). Records
        history exactly like _objective (merit = sum of squared residuals)."""
        import numpy as np
        params = self._denorm(x)
        n_op = len(self.merit.operands)
        if self.n_evals >= self.budget:
            big = math.sqrt(PENALTY / max(1, n_op))
            return np.full(n_op, (self.best and
                                  math.sqrt(self.best["merit"] / n_op)) or big)
        note = None
        out = None
        try:
            out = self.evaluate_fn(params)
            resid, rows, missing = self.merit.residuals(out)
            penalized = bool(missing)
            if missing:
                note = "; ".join(str(m) for m in missing)
        except Exception as exc:
            big = math.sqrt(PENALTY / max(1, n_op))
            resid = [big] * n_op
            rows, penalized = [], True
            note = "evaluation failed: %s: %s" % (type(exc).__name__, exc)
        merit = float(sum(r * r for r in resid))
        self._record(params, merit, rows, penalized, note, out)
        return np.asarray(resid, dtype=float)

    # -- drivers ------------------------------------------------------------------
    def run(self):
        """Run the configured algorithm; returns the best history entry
        (raises OptimizeError if not even the start design evaluated)."""
        x0 = self._x0()
        # both algorithms first evaluate the START design: the convergence
        # history then always begins at the user's baseline
        self._objective(x0)
        if self.algorithm in ("local", "simplex"):
            self._run_local(x0)
        elif self.algorithm == "dls":
            self._run_dls(x0)
        elif self.algorithm == "global":
            self._run_global(x0)
        else:
            raise OptimizeError("unknown algorithm %r" % (self.algorithm,))
        if self.best is None:
            raise OptimizeError(
                "no evaluation produced a usable merit (all %d were "
                "penalized) — check the operands against the scene's "
                "detectors" % self.n_evals)
        return self.best

    def _run_local(self, x0):
        import numpy as np
        from scipy.optimize import minimize
        n = len(self.variables)
        x0 = np.asarray(x0, dtype=float)
        # domain-scale initial simplex (scipy's default is 5% of x0 —
        # useless against a typical eval budget of a few dozen): step a
        # quarter of each normalized bound range, flipped at the edges
        step = 0.25
        simplex = [x0]
        for i in range(n):
            v = x0.copy()
            v[i] = v[i] + step if v[i] + step <= 1.0 else v[i] - step
            simplex.append(v)
        minimize(self._objective, x0, method="Nelder-Mead",
                 bounds=[(0.0, 1.0)] * n,
                 options={"maxfev": self.budget, "fatol": self.tol,
                          "xatol": 1e-4, "adaptive": n > 2,
                          "initial_simplex": np.asarray(simplex)})

    def _run_dls(self, x0):
        """Damped least-squares over the operand residual vector (scipy
        least_squares 'trf' -- the bounded trust-region Levenberg-Marquardt
        core Optiland's DLS wraps). Deterministic sequential merits make the
        finite-difference Jacobian meaningful (MC speckle would make it
        garbage), which is the whole reason sequential mode unlocks DLS
        (engine3.md Sec 8). Operates in normalized [0,1] coords."""
        import numpy as np
        from scipy.optimize import least_squares
        ok, reason = self.merit.dls_supported()
        if not ok:
            raise OptimizeError("DLS is not applicable: %s" % reason)
        x0 = np.clip(np.asarray(x0, dtype=float), 0.0, 1.0)
        # a finite-difference step large enough to see past sequential
        # round-off yet local; normalized coords, so one step spans the bound
        # range * diff_step.
        least_squares(
            self._residuals, x0, bounds=(0.0, 1.0), method="trf",
            diff_step=1e-3, xtol=self.tol, ftol=self.tol, gtol=1e-12,
            max_nfev=self.budget)

    def _run_global(self, x0):
        import warnings

        import nevergrad as ng
        import numpy as np
        param = ng.p.Array(init=np.asarray(x0, dtype=float),
                           lower=0.0, upper=1.0)
        opt = ng.optimizers.CMA(parametrization=param, budget=self.budget)
        opt.parametrization.random_state = np.random.RandomState(self.seed)
        with warnings.catch_warnings():
            try:
                from cma.evolution_strategy import InjectionWarning
                warnings.simplefilter("ignore", InjectionWarning)
            except ImportError:  # pragma: no cover - cma always present
                pass
            # bounded by real evals AND by iterations (memo hits are free
            # but must not spin forever)
            for _ in range(self.budget * 4):
                if self.n_evals >= self.budget:
                    break
                cand = opt.ask()
                value = self._objective(
                    np.asarray(cand.value, dtype=float).ravel())
                opt.tell(cand, float(value))


# ---------------------------------------------------------------------------
# orchestration (real fast_eval evaluators + report writing)
# ---------------------------------------------------------------------------
def _fidelity_kwargs(args):
    return dict(preset=args.preset, rays=args.rays,
                resolution=args.resolution, nlambda=args.nlambda,
                seeds=args.seeds, seed0=args.seed0)


def _apply_config_file(parser, args):
    """--config JSON: values fill in wherever the CLI kept the parser
    default (explicit CLI flags win). var/operand entries may be spec
    strings (parsed here) or ready dicts."""
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
        if dest == "var":
            value = [cli_specs.parse_var_spec(v) if isinstance(v, str)
                     else v for v in value]
        elif dest == "operand":
            value = [cli_specs.parse_operand_spec(v) if isinstance(v, str)
                     else v for v in value]
        setattr(args, dest, value)


def main(argv=None):
    parser = cli_specs.build_parser("optimize")
    args = parser.parse_args(argv)
    _apply_config_file(parser, args)
    if not args.var:
        parser.error("at least one --var NAME:START:LO:HI is required")
    if not args.operand:
        parser.error("at least one --operand OPERAND:TARGET:WEIGHT is "
                     "required")

    from fast_eval import Evaluator   # optics env; imports fcclient

    model_stem = Path(args.model).stem
    out_dir = Path(args.out) if args.out else (
        PROJECT_DIR / "var" / "optimize" / model_stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequential = args.eval_backend == "sequential"
    algorithm = args.algorithm
    if sequential:
        # the deterministic backend PROMOTES the default derivative-free
        # 'local' to Optiland-backed DLS; 'simplex' still forces Nelder-Mead.
        if algorithm == "local":
            algorithm = "dls"
        mc_ops = mc_only_operands(args.operand)
        if mc_ops:
            parser.error(
                "operand(s) %s need the Monte-Carlo backend (detected_power / "
                "mtf50 / raw report keys) and cannot be served by "
                "--eval-backend sequential; drop them or use --eval-backend "
                "worker" % ", ".join(sorted(set(mc_ops))))

    needs = operand_needs(args.operand)
    trace_args = []
    keep_coherent_inner = False
    if not sequential:
        # the sequential Optiland backend runs no MC trace, so --export-rays /
        # --save-fields / coherence do not apply to it.
        if "export_rays" in needs:
            trace_args.append("--export-rays")
        if "save_fields" in needs:
            # the field-analysis block (ee_r80/mtf50) only exists when the
            # coherent gather ran and saved field maps
            trace_args.append("--save-fields")
            keep_coherent_inner = True

    names = [v["name"] for v in args.var]
    print("[optimize] model=%s vars=%s operands=%s backend=%s algorithm=%s "
          "budget=%d"
          % (args.model, names,
             [o["operand"] for o in args.operand], args.eval_backend,
             algorithm, args.budget), flush=True)
    common.progress_emit("optimize", 0.0,
                         "starting %s optimization (%d-var, budget %d)"
                         % (algorithm, len(names), args.budget),
                         case_dir=out_dir)

    evaluator = Evaluator(args.model, params=names,
                          backend=args.eval_backend,
                          keep_coherent=keep_coherent_inner,
                          trace_args=trace_args, workdir=args.workdir,
                          **_fidelity_kwargs(args))
    engine = OptimizationEngine(args.var, args.operand, evaluator.evaluate,
                                algorithm=algorithm,
                                budget=args.budget, tol=args.tol,
                                seed=args.optimizer_seed, case_dir=out_dir)
    t0 = time.monotonic()
    try:
        with evaluator:
            best = engine.run()
    except OptimizeError as exc:
        common.progress_emit("optimize", None, str(exc),
                             case_dir=out_dir, status="failed")
        print("[optimize] FAILED: %s" % exc, flush=True)
        _write_report(out_dir, args, engine, wall_s=time.monotonic() - t0,
                      status="failed", error=str(exc))
        return 1

    report = _write_report(out_dir, args, engine,
                           wall_s=time.monotonic() - t0)

    # faithful final number: re-evaluate the best design with coherence
    # as authored (the inner loop forced incoherent unless save_fields). The
    # sequential backend runs no MC gather, so there is no coherent re-eval.
    if not args.no_final_coherent and not keep_coherent_inner and not sequential:
        print("[optimize] re-evaluating best design with coherent "
              "sources...", flush=True)
        common.progress_emit("optimize", 1.0,
                             "final coherent re-evaluation",
                             case_dir=out_dir)
        try:
            final_ev = Evaluator(args.model, params=names,
                                 backend=args.eval_backend,
                                 keep_coherent=True,
                                 trace_args=trace_args,
                                 workdir=args.workdir,
                                 **_fidelity_kwargs(args))
            with final_ev:
                out = final_ev.evaluate(best["params"])
            merit, rows, missing = engine.merit.score(out)
            report["final_coherent"] = {
                "merit": merit, "operands": rows,
                "penalized": bool(missing),
                "note": "; ".join(str(m) for m in missing) or None,
                "case_dir": out.get("case_dir"),
            }
        except Exception as exc:
            # e.g. the coherent gather's undersampling gate at low ray
            # budgets — the optimization result stands, honestly annotated
            report["final_coherent"] = {
                "error": "%s: %s" % (type(exc).__name__, exc)}
        common.write_json(out_dir / "report.json", report)

    fc = report.get("final_coherent") or {}
    common.progress_emit(
        "optimize", 1.0,
        "done: best merit %.6g after %d evals" % (best["merit"],
                                                  engine.n_evals),
        case_dir=out_dir, status="completed",
        extra={"best": best["merit"], "best_params": best["params"],
               "final_coherent_merit": fc.get("merit")})
    print("[optimize] best merit %.6g at %s (%d evals, %.1f s) -> %s"
          % (best["merit"],
             {k: round(v, 6) for k, v in best["params"].items()},
             engine.n_evals, report["wall_s"], out_dir / "report.json"),
          flush=True)
    if "merit" in fc:
        print("[optimize] final coherent merit: %.6g" % fc["merit"],
              flush=True)
    elif "error" in fc:
        print("[optimize] final coherent re-eval FAILED (result stands "
              "on incoherent numbers): %s" % fc["error"], flush=True)
    return 0


def _write_report(out_dir, args, engine, wall_s, status="completed",
                  error=None):
    report = {
        "model": str(args.model),
        "algorithm": engine.algorithm,
        "budget": args.budget,
        "tol": args.tol,
        "preset": args.preset,
        "eval_backend": args.eval_backend,
        "variables": engine.variables,
        "operands": engine.merit.operands,
        "history": engine.history,
        "best": engine.best,
        "n_evals": engine.n_evals,
        "wall_s": round(wall_s, 3),
        "status": status,
    }
    if error:
        report["error"] = error
    common.write_json(out_dir / "report.json", report)
    return report


if __name__ == "__main__":
    sys.exit(main())
