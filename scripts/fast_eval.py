#!/usr/bin/env python
# =============================================================================
# fast_eval.py — fast merit-function evaluator for the optimizer round.
#
# Interpreter: "$MIEWB_OPTICS_PYTHON" (the OPTICS env; it shells out
# to the FreeCAD AppImage and to run_trace.py/post_process.py under their
# own pinned interpreters, and imports the Qt-free
# mieworkbench.core.fcclient for the persistent-worker protocol client).
#
# An Evaluator maps a dict of scalar design parameters (spreadsheet cell
# aliases, exactly what permute_model.py --var sweeps address: bare
# "alias" on the default 'dim' sheet, or "sheetlabel.alias") to a dict of
# merit scalars read from the pipeline's report.json. Two backends:
#
#   backend="full"    the REFERENCE. Per evaluate(): permute_model.py
#                     (fresh FreeCAD launch, writes a variant .FCStd) ->
#                     extract_geometry.py (fresh launch) -> run_trace.py
#                     -> post_process.py -> report.json. Always correct;
#                     pays ~2 FreeCAD AppImage launches (each of which
#                     runs its script twice — the -c double-execution
#                     quirk) per evaluation.
#
#   backend="worker"  the FAST path. One persistent headless FreeCAD
#                     worker (scripts/fcserver/fc_server.py) keeps the
#                     base document open across evaluations; per
#                     evaluate() it applies only the parameter cell edits
#                     (fcops.op_apply_params == permute_model.
#                     apply_assignments, the SAME function the full path
#                     runs) and extracts model.json in place
#                     (fcops.op_extract_model == extract_geometry.
#                     extract_document, the SAME function the full path
#                     runs), with a quantized shape-fingerprint face
#                     cache so unchanged bodies skip re-tessellation.
#                     The trace/post stages are the identical subprocess
#                     commands the full path uses.
#
# Equivalence contract (pinned by scripts/raytracer/tests/test_fast_eval.py,
# the PARITY ORACLE): for the same params, the worker backend's extracted
# model.json must equal the full backend's up to OCC recompute noise, and
# the merits must match. NOTE bit-identity is NOT attainable even
# full-vs-full: two identical permute_model.py runs on this machine
# produce variant BReps differing at the last ULP (~1e-16 relative, OCC
# boolean/recompute nondeterminism), so the oracle compares numerically
# (tight tolerances) rather than byte-wise. WITHIN one worker session the
# fingerprint cache absorbs that noise, so repeated identical params are
# bit-stable.
#
# Fast trace mode: unless keep_coherent=True, every source in the
# extracted model.json is patched to coherent=false before tracing
# (coherence is a per-source scene property — raytracer/sources.py — and
# with no coherent sources the expensive Huygens gather never runs;
# detectors get direct geometric deposit). Both backends apply the same
# patch, so it never breaks parity. --save-fields is never passed.
#
# Crash recovery (worker backend): every protocol call has a timeout (a
# HUNG worker == a dead worker); on death the client relaunches the
# AppImage, re-opens the base document, replays the CURRENT cumulative
# parameter state (one apply_params — each eval carries the full
# parameter vector, so no journal is needed), and rotates the extract
# cache directory (fingerprint entries can't be trusted across a
# restart). The in-flight op is then retried; if the worker path still
# fails after bounded retries, THAT ONE evaluation falls back to the
# always-correct full path. Only if even the fallback fails does
# evaluate() raise (EvalError, with the underlying causes named). A
# single eval failure never crashes the caller's optimization loop.
#
# Typical use:
#     from fast_eval import Evaluator
#     ev = Evaluator("example.FCStd", params=["lenspos", "sphered"],
#                    backend="worker", rays=2e4, resolution=256)
#     with ev:
#         out = ev.evaluate({"lenspos": 2.5, "sphered": 28.0})
#         out["merits"]["Body003.Pad001.Face5.total_power_W"]
#
# CLI (mostly for benchmarking / smoke use):
#     "$MIEWB_OPTICS_PYTHON" scripts/fast_eval.py \
#         --model example.FCStd --backend worker \
#         --eval lenspos=2.5,sphered=28 --eval lenspos=0,sphered=30
# =============================================================================
import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_DIR))   # for mieworkbench.core.fcclient

import common  # noqa: E402
from mieworkbench.core.fcclient import (FcClient, FcDead,  # noqa: E402
                                        FcError)

PERMUTE = SCRIPTS_DIR / "permute_model.py"
EXTRACT = SCRIPTS_DIR / "extract_geometry.py"
RUN_TRACE = SCRIPTS_DIR / "run_trace.py"
POST_PROCESS = SCRIPTS_DIR / "post_process.py"
FC_SERVER = SCRIPTS_DIR / "fcserver" / "fc_server.py"
OPTILAND_EVAL = SCRIPTS_DIR / "raytracer" / "optiland_eval.py"

CASE = "fasteval"          # one case dir per variant, reused across repeats

# optiland_eval.py's "BridgeUnsupported -> fall back to MC" exit code.
_UNSUPPORTED_EXIT = 3


class EvalError(RuntimeError):
    """A merit evaluation failed on every available path."""


class SequentialUnsupported(RuntimeError):
    """The Optiland bridge rejected the scene (out of sequential scope); the
    sequential backend falls back to the MC path for this evaluation."""


# ---------------------------------------------------------------------------
# Worker client: FcClient with evaluator-owned crash recovery
# ---------------------------------------------------------------------------
class _WorkerClient(FcClient):
    """FcClient whose relaunch-and-replay restores the evaluator's state.

    FcClient already detects dead AND hung workers (every op has a
    timeout; a hang raises FcDead and recovery kills the process), and
    request() transparently relaunches + retries once. Its stock replay
    is an op journal, which is wrong for the evaluator (each eval carries
    the full parameter vector, so the correct replay is: re-open the base
    document, apply the CURRENT cumulative parameters once). recover_hook
    does exactly that — and rotates the extract cache dir first, since
    fingerprint-cache entries can't be trusted across a restart.
    """

    def __init__(self, *args, recover_hook=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recover_hook = recover_hook
        self.n_recoveries = 0

    def _recover_locked(self):
        self.kill()
        self.start()
        self.n_recoveries += 1
        for _doc, path in list(self._open_docs.items()):
            self._request_locked("open_document", {"path": path}, None)
        if self.recover_hook is not None:
            self.recover_hook(self)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class Evaluator:
    """evaluate(param_dict) -> merit dict, as fast as possible, correctly.

    Parameters
    ----------
    model : str/Path — the base .FCStd (bare names resolve against the
        project root, then basemodels/).
    params : optional list of parameter names, fixing the parameter order
        (== permute_model --var order == variant naming order). If omitted
        it is locked from the first evaluate() call (sorted). EVERY
        evaluate() must then supply exactly this set — the worker document
        accumulates cell edits, so a partial vector would silently mean
        "keep the previous eval's leftovers" and diverge from the full
        backend; requiring the full vector removes that class of bug.
    backend : "worker" (fast, persistent FreeCAD) or "full" (reference).
    preset : fidelity preset for any of rays/resolution/nlambda/
        spectral_bins not given explicitly (common.PRESETS; default
        "quick").
    seeds/seed0 : forwarded to run_trace.py (deterministic per seed).
    keep_coherent : leave source coherence as authored (default False =
        patch every source to incoherent so the gather never runs).
    detectors : optional list of detector labels to keep in "merits"
        (default: all).
    trace_args/post_args : extra CLI args appended verbatim to the trace/
        post commands (both backends, so parity is preserved).
    workdir : evaluation workspace (default var/fasteval/<stem>-<backend>).
        Layout: models/ (full-backend variants), geometry/<variant>/,
        results/<variant>/fasteval/, cache/ (worker face cache), logs/.
    """

    def __init__(self, model, params=None, backend="worker",
                 preset="quick", rays=None, resolution=None, nlambda=None,
                 spectral_bins=None, seeds=1, seed0=42, unit="mm",
                 strict=False, keep_coherent=False, detectors=None,
                 trace_args=(), post_args=(), workdir=None,
                 freecad=None, op_timeout=300.0, worker_retries=1,
                 gui_python=None, model_stop=True, seq_n_rays=4096,
                 seq_seed=0, seq_ee_frac=0.8):
        self.model_path = self._resolve_model(model)
        self.backend = backend
        if backend not in ("worker", "full", "sequential"):
            raise ValueError("backend must be 'worker', 'full' or "
                             "'sequential' (got %r)" % (backend,))
        # sequential backend: the Optiland merit path (env/ interpreter),
        # driven off the SAME worker-extracted model.json the MC path uses.
        self.gui_python = str(gui_python or common.GUI_PYTHON)
        self.model_stop = bool(model_stop)
        self.seq_n_rays = int(seq_n_rays)
        self.seq_seed = int(seq_seed)
        self.seq_ee_frac = float(seq_ee_frac)
        pre = common.PRESETS[preset]
        self.rays = float(rays if rays is not None else pre["rays"])
        self.resolution = int(resolution if resolution is not None
                              else pre["resolution"])
        self.nlambda = int(nlambda if nlambda is not None else pre["nlambda"])
        self.spectral_bins = int(spectral_bins if spectral_bins is not None
                                 else pre["spectral_bins"])
        self.seeds = int(seeds)
        self.seed0 = int(seed0)
        self.unit = unit
        self.strict = bool(strict)
        self.keep_coherent = bool(keep_coherent)
        self.detectors = list(detectors) if detectors else None
        self.trace_args = [str(a) for a in trace_args]
        self.post_args = [str(a) for a in post_args]
        self.freecad = str(freecad or common.FREECAD_APPIMAGE)
        self.op_timeout = float(op_timeout)
        self.worker_retries = int(worker_retries)

        self.workdir = Path(workdir) if workdir else (
            PROJECT_DIR / "var" / "fasteval"
            / ("%s-%s" % (self.model_path.stem, backend)))
        for sub in ("models", "geometry", "results", "logs"):
            (self.workdir / sub).mkdir(parents=True, exist_ok=True)

        self._param_names = list(params) if params else None
        self._applied = {}          # last successfully-requested param dict
        self._client = None
        self._doc = None            # worker-side document name
        self._cache_dir = None      # rotated on every worker (re)launch
        self.last_cache = None      # {"hits": [...], "misses": [...]}
        self.n_evals = 0
        self.n_fallbacks = 0

    # -- public --------------------------------------------------------------
    def evaluate(self, params):
        """One merit evaluation. Never raises for a single recoverable
        failure — the worker backend falls back to the full path for this
        eval; EvalError only when every path failed."""
        params = self._check_params(params)
        variant = self._variant(params)
        if self.backend == "sequential":
            return self._evaluate_sequential(params, variant)
        if self.backend == "full":
            return self._finish(params, variant,
                                self._prepare_full(params, variant), "full")

        worker_exc = None
        for _attempt in range(self.worker_retries + 1):
            try:
                model_json = self._prepare_worker(params, variant)
                return self._finish(params, variant, model_json, "worker")
            except (FcDead, FcError, EvalError, OSError) as exc:
                worker_exc = exc
                # hard reset: kill the worker; the next attempt (or a
                # later evaluate()) relaunches from scratch with a fresh
                # cache dir and replays the base doc + params.
                self._teardown_worker()
        # bounded worker retries exhausted -> the always-correct full path
        self.n_fallbacks += 1
        try:
            out = self._finish(params, variant,
                               self._prepare_full(params, variant),
                               "full-fallback")
            out["fallback_reason"] = "%s: %s" % (
                type(worker_exc).__name__, worker_exc)
            return out
        except Exception as exc:
            raise EvalError(
                "evaluate(%r) failed on the worker path (%s: %s) AND on "
                "the full fallback (%s: %s). Nothing left to try — check "
                "the logs under %s."
                % (params, type(worker_exc).__name__, worker_exc,
                   type(exc).__name__, exc, self.workdir / "logs"))

    # -- sequential (Optiland) backend -----------------------------------------
    def _model_json_for(self, params, variant):
        """Produce the variant's model.json the cheapest correct way: the
        persistent worker (with bounded retries), falling back to a fresh
        FreeCAD launch. Shared by the sequential backend and the MC fallback."""
        if self.backend == "full":
            return self._prepare_full(params, variant)
        worker_exc = None
        for _attempt in range(self.worker_retries + 1):
            try:
                return self._prepare_worker(params, variant)
            except (FcDead, FcError, EvalError, OSError) as exc:
                worker_exc = exc
                self._teardown_worker()
        # worker exhausted -> the always-correct full path
        self.n_fallbacks += 1
        try:
            return self._prepare_full(params, variant)
        except Exception as exc:
            raise EvalError(
                "model.json build failed on the worker path (%s: %s) AND the "
                "full fallback (%s: %s)"
                % (type(worker_exc).__name__, worker_exc,
                   type(exc).__name__, exc))

    def _evaluate_sequential(self, params, variant):
        """One sequential (Optiland) merit evaluation: build model.json, then
        evaluate operands DIRECTLY in Optiland via the env/ bridge. A scene
        the bridge rejects (BridgeUnsupported) transparently falls back to the
        MC path for THIS evaluation, clearly annotated."""
        model_json = self._model_json_for(params, variant)
        try:
            return self._finish_sequential(params, variant, model_json)
        except SequentialUnsupported as exc:
            self.n_fallbacks += 1
            out = self._finish(params, variant, model_json,
                               "sequential-mc-fallback")
            out["fallback_reason"] = "BridgeUnsupported: %s" % exc
            return out

    def _finish_sequential(self, params, variant, model_json):
        """Shell out to env/bin/python optiland_eval.py (Optiland lives in the
        GUI venv ONLY). Returns a fast_eval-shaped output dict; raises
        SequentialUnsupported on the bridge's out-of-scope exit."""
        cmd = [self.gui_python, str(OPTILAND_EVAL),
               "--model-json", str(model_json),
               "--n-rays", str(self.seq_n_rays),
               "--seed", str(self.seq_seed),
               "--ee-frac", repr(self.seq_ee_frac)]
        if self.model_stop:
            cmd.append("--model-stop")
        log_path = self._log_path(variant, "optiland")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run([str(c) for c in cmd], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True)
        with open(log_path, "w") as fh:
            fh.write("CMD: %s\n\nSTDOUT:\n%s\nSTDERR:\n%s"
                     % (" ".join(str(c) for c in cmd), proc.stdout,
                        proc.stderr))
        if proc.returncode == _UNSUPPORTED_EXIT:
            reason = proc.stdout.strip()
            try:
                reason = json.loads(reason)["bridge_unsupported"]
            except Exception:
                reason = reason or proc.stderr[-400:]
            raise SequentialUnsupported(reason)
        if proc.returncode != 0:
            raise EvalError("optiland_eval.py failed (exit %d) — log %s\n%s"
                            % (proc.returncode, log_path, proc.stderr[-800:]))
        try:
            rep = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as exc:
            raise EvalError("optiland_eval.py returned unparseable output "
                            "(%s) — log %s" % (exc, log_path))
        detectors = rep.get("detectors", {})
        self.n_evals += 1
        return {
            "params": dict(params),
            "variant": variant,
            "backend_used": "sequential",
            "model_json": str(model_json),
            "case_dir": None,
            "closure_ok": None,
            "detectors": detectors,
            "merits": flatten_merits(detectors, only=self.detectors),
            "sequential": {"paraxial_f2_mm": rep.get("paraxial_f2_mm"),
                           "epd_mm": rep.get("epd_mm"),
                           "engine": rep.get("engine"),
                           "model_stop": rep.get("model_stop")},
            "cache": self.last_cache,
        }

    def worker_pid(self):
        """PID of the live FreeCAD worker process (fault-injection tests
        kill it), or None."""
        if self._client is not None and self._client.ready_info:
            return self._client.ready_info.get("pid")
        return None

    def close(self):
        self._teardown_worker(shutdown=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- shared plumbing -------------------------------------------------------
    @staticmethod
    def _resolve_model(model):
        p = Path(model)
        if p.exists():
            return p.resolve()
        for cand in (PROJECT_DIR / p.name, common.BASEMODELS_DIR / p.name):
            if cand.exists():
                return cand.resolve()
        raise FileNotFoundError("no such model: %s" % model)

    def _check_params(self, params):
        if not params:
            raise ValueError("evaluate() needs a non-empty param dict")
        if self._param_names is None:
            self._param_names = sorted(params)
        if set(params) != set(self._param_names):
            raise ValueError(
                "evaluate() must supply exactly the locked parameter set "
                "%s every time (got %s): the worker document accumulates "
                "cell edits, so partial vectors would diverge from the "
                "full backend" % (self._param_names, sorted(params)))
        return {k: float(params[k]) for k in self._param_names}

    def _variant(self, params):
        stem = self.model_path.stem
        for name in self._param_names:
            stem = common.variant_name(stem, name, params[name])
        return common.shorten_variant(stem)

    def _log_path(self, variant, stage):
        return self.workdir / "logs" / ("%s.%s.log" % (variant, stage))

    def _run_logged(self, cmd, log_path, what, stdin_devnull=False):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                [str(c) for c in cmd],
                stdin=(subprocess.DEVNULL if stdin_devnull else None),
                stdout=logf, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            tail = "".join(
                open(log_path).readlines()[-12:]) if log_path.exists() else ""
            raise EvalError("%s failed (exit %d) — log %s\n%s"
                            % (what, proc.returncode, log_path, tail))

    def _geom_dir(self, variant):
        return self.workdir / "geometry" / variant

    def _case_dir(self, variant):
        return self.workdir / "results" / variant / CASE

    # -- full backend ----------------------------------------------------------
    def _prepare_full(self, params, variant):
        """permute + extract via fresh FreeCAD launches; returns the
        variant's model.json path."""
        models_dir = self.workdir / "models"
        cmd = [self.freecad, "-c", PERMUTE, "--",
               "--model", self.model_path, "--outdir", models_dir,
               "--unit", self.unit]
        for name in self._param_names:
            cmd += ["--var", name, "--min", repr(params[name]),
                    "--max", repr(params[name]), "--n", "0"]
        self._run_logged(cmd, self._log_path(variant, "permute"),
                         "permute_model.py", stdin_devnull=True)
        variant_fcstd = models_dir / (variant + ".FCStd")
        if not variant_fcstd.exists():
            raise EvalError("permute_model.py reported success but %s was "
                            "not written" % variant_fcstd)

        geom = self._geom_dir(variant)
        shutil.rmtree(geom, ignore_errors=True)
        cmd = [self.freecad, "-c", EXTRACT, "--",
               "--models", variant_fcstd,
               "--outdir", self.workdir / "geometry"]
        if self.strict:
            cmd += ["--strict"]
        self._run_logged(cmd, self._log_path(variant, "extract"),
                         "extract_geometry.py", stdin_devnull=True)
        model_json = geom / "model.json"
        if not model_json.exists():
            raise EvalError("extract_geometry.py reported success but %s "
                            "was not written" % model_json)
        return model_json

    # -- worker backend ----------------------------------------------------------
    def _ensure_worker(self):
        if self._client is not None and self._client.is_alive():
            return
        # a dead-but-not-torn-down client (e.g. the worker was killed
        # BETWEEN evals, so no request noticed) still owns reader threads
        # and a zombie Popen; reap it before relaunching
        self._teardown_worker()
        self._rotate_cache()
        self._client = _WorkerClient(
            appimage=self.freecad, server_script=str(FC_SERVER),
            op_timeout=self.op_timeout, recover_hook=self._on_recover)
        self._client.start()
        result = self._client.open_document(str(self.model_path))
        self._doc = result["doc"]

    def _rotate_cache(self):
        # a fresh dir per worker lifetime: entries from a dead worker are
        # never trusted (and the old dir is deleted, not just abandoned)
        if self._cache_dir is not None:
            shutil.rmtree(self._cache_dir, ignore_errors=True)
        self._cache_dir = (self.workdir / "cache"
                           / ("worker-%s" % uuid.uuid4().hex[:8]))
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _on_recover(self, client):
        """Called by _WorkerClient inside its relaunch (documents already
        re-opened): rotate the fingerprint cache and replay the cumulative
        parameter state so the in-flight op retries against a document
        equivalent to the pre-crash one."""
        self._rotate_cache()
        if self._applied:
            client._request_locked("apply_params", {
                "doc": self._doc,
                "assignments": [[k, self._applied[k]]
                                for k in self._param_names
                                if k in self._applied],
                "unit": self.unit}, None)

    def _teardown_worker(self, shutdown=False):
        if self._client is not None:
            try:
                if shutdown:
                    self._client.shutdown()
                else:
                    self._client.kill()
            except Exception:
                pass
        self._client = None
        self._doc = None

    def _prepare_worker(self, params, variant):
        """apply params + extract in place via the persistent worker;
        returns the variant's model.json path."""
        self._ensure_worker()
        # commit the target state BEFORE the request: if the worker dies
        # mid-op, recovery replays exactly this state and the op retry is
        # a (by-construction idempotent) re-application.
        self._applied = dict(params)
        self._client.request("apply_params", {
            "doc": self._doc,
            "assignments": [[k, params[k]] for k in self._param_names],
            "unit": self.unit})

        geom = self._geom_dir(variant)
        shutil.rmtree(geom, ignore_errors=True)
        result = self._client.request("extract_model", {
            "doc": self._doc,
            "out_dir": str(geom),
            "stem": variant,
            "strict": self.strict,
            # provenance echo: the worker never writes a variant .FCStd,
            # so name the base model (the full backend names its variant
            # file here — the parity oracle treats the field as
            # provenance, not geometry)
            "source_fcstd": str(self.model_path),
            "cache_dir": str(self._cache_dir)})
        self.last_cache = {"hits": result["cache_hits"],
                           "misses": result["cache_misses"]}
        model_json = geom / "model.json"
        if not model_json.exists():
            raise EvalError("extract_model op succeeded but %s was not "
                            "written" % model_json)
        return model_json

    # -- trace + post + merit read (identical for both backends) ---------------
    def _patch_incoherent(self, model_json):
        """Force every source incoherent so the Huygens gather never runs
        (the merit fast mode; identical on both backends)."""
        with open(model_json) as fh:
            model = json.load(fh)
        changed = False
        for body in model["bodies"]:
            src = body.get("source")
            if src and src.get("coherent"):
                src["coherent"] = False
                changed = True
        if changed:
            common.write_json(model_json, model)
        return changed

    def _finish(self, params, variant, model_json, backend_used):
        if not self.keep_coherent:
            self._patch_incoherent(model_json)

        case_dir = self._case_dir(variant)
        trace_cmd = [common.OPTICS_PYTHON, RUN_TRACE,
                     "--model-json", model_json, "--case-dir", case_dir,
                     "--rays", repr(self.rays),
                     "--resolution", str(self.resolution),
                     "--nlambda", str(self.nlambda),
                     "--spectral-bins", str(self.spectral_bins),
                     "--seeds", str(self.seeds),
                     "--seed0", str(self.seed0)] + self.trace_args
        self._run_logged(trace_cmd, self._log_path(variant, "trace"),
                         "run_trace.py")
        status = common.read_case_status(case_dir / "case.json")
        if status != "completed":
            raise EvalError("trace finished but case status is %r (see %s)"
                            % (status, self._log_path(variant, "trace")))

        post_cmd = [common.OPTICS_PYTHON, POST_PROCESS,
                    "--case-dir", case_dir,
                    "--model-json", model_json] + self.post_args
        self._run_logged(post_cmd, self._log_path(variant, "post"),
                         "post_process.py")

        with open(case_dir / "report.json") as fh:
            report = json.load(fh)
        self.n_evals += 1
        return {
            "params": dict(params),
            "variant": variant,
            "backend_used": backend_used,
            "model_json": str(model_json),
            "case_dir": str(case_dir),
            "closure_ok": report.get("closure_ok"),
            "detectors": report.get("detectors", {}),
            "merits": flatten_merits(report.get("detectors", {}),
                                     only=self.detectors),
            "cache": (self.last_cache if backend_used == "worker"
                      else None),
        }


def flatten_merits(detectors, only=None):
    """report.json['detectors'] -> flat {'label.path.to.scalar': float}
    (numbers only; bools and non-scalar leaves are skipped; lists of
    dicts, e.g. spot rows / per_source, are indexed numerically)."""
    out = {}

    def walk(prefix, node):
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            out[prefix] = float(node)
        elif isinstance(node, dict):
            for k in node:
                walk("%s.%s" % (prefix, k), node[k])
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk("%s.%d" % (prefix, i), item)

    for label, block in detectors.items():
        if only is not None and label not in only:
            continue
        walk(label, block)
    return out


# ---------------------------------------------------------------------------
# CLI (benchmark / smoke)
# ---------------------------------------------------------------------------
def _parse_eval_spec(spec):
    out = {}
    for kv in spec.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise SystemExit("bad --eval entry %r (expected name=value)"
                             % kv)
        k, v = kv.split("=", 1)
        out[k.strip()] = float(v)
    if not out:
        raise SystemExit("empty --eval spec %r" % spec)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Fast merit evaluator (see module docstring).")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", default="worker",
                   choices=["worker", "full", "sequential"])
    p.add_argument("--eval", action="append", required=True,
                   metavar="k=v,k=v", dest="evals",
                   help="one evaluation per flag (repeatable)")
    p.add_argument("--preset", default="quick",
                   choices=sorted(common.PRESETS))
    p.add_argument("--rays", type=float, default=None)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--nlambda", type=int, default=None)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--keep-coherent", action="store_true")
    p.add_argument("--workdir", default=None)
    args = p.parse_args(argv)

    evals = [_parse_eval_spec(s) for s in args.evals]
    names = sorted(evals[0])
    ev = Evaluator(args.model, params=names, backend=args.backend,
                   preset=args.preset, rays=args.rays,
                   resolution=args.resolution, nlambda=args.nlambda,
                   seeds=args.seeds, keep_coherent=args.keep_coherent,
                   workdir=args.workdir)
    with ev:
        for pd in evals:
            t0 = time.monotonic()
            out = ev.evaluate(pd)
            dt = time.monotonic() - t0
            print(json.dumps({
                "params": out["params"], "variant": out["variant"],
                "backend_used": out["backend_used"],
                "eval_s": round(dt, 3),
                "cache": out["cache"],
                "merits": out["merits"]}, indent=1, sort_keys=True),
                flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
