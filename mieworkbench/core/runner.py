"""RunController - Qt-only wrapper around QProcess for driving
scripts/run_pipeline.py from the GUI.

Deliberately the mirror image of core/fcclient.py's design note: fcclient
is Qt-free because it must be unit-testable under plain pytest; this class
is Qt-only (QObject + QProcess + signals) because its entire job is to
bridge a subprocess's stdout onto Qt signals the GUI panes can connect to
(the console pane, the stage chips, the progress bar). There is no
Qt-independent core to extract - QProcess already does the subprocess
plumbing (non-blocking reads driven by the Qt event loop) that FcClient
has to hand-roll with threads because it has no event loop of its own.

Protocol: run_pipeline.py is launched under plain system python3 (it is
stdlib-only by design, see scripts/common.py's header) with MIEWB_PROGRESS
=1 set so every stage emits '@MIEWB {...}' progress lines on stdout
(scripts/common.py: progress_emit/parse_progress_line). Stdout+stderr are
merged (MergedChannels) so ordering between a stage's own log lines and
its progress lines is preserved as they were written. Each line is
classified by common.parse_progress_line(): progress lines become
progress(dict); everything else becomes line(str) for the console pane.

build_args() is the single place a config-matrix value dict turns into
pipeline argv; it walks the SAME cli_specs.build_parser("pipeline")
actions the pipeline script itself parses argv with, so option names,
defaults, and append/store_true handling can never drift from the real
CLI.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import common     # noqa: E402  (stdlib-only shared contract hub)
import cli_specs   # noqa: E402  (stdlib-only; single source of truth for CLIs)

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

REPO_DIR = Path(__file__).resolve().parent.parent.parent
RUN_PIPELINE_SCRIPT = REPO_DIR / "scripts" / "run_pipeline.py"

STOP_GRACE_MS = 5000


class RunController(QObject):
    """Owns at most one run_pipeline.py QProcess at a time."""

    started = Signal()
    line = Signal(str)
    progress = Signal(dict)
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = settings   # core.settings.Settings, or None
        self._proc = None
        self._stop_requested = False

    # -- state ----------------------------------------------------------------
    def is_running(self):
        return (self._proc is not None
                and self._proc.state() != QProcess.NotRunning)

    # -- lifecycle --------------------------------------------------------------
    def start(self, model_path, extra_args=None, steps=None,
              extra_env=None):
        """Launch run_pipeline.py --models <model_path> <extra_args>
        (+ --steps <steps> if given). extra_env: dict merged over the
        settings-derived environment (workspace runs point
        MIEWB_GEOMETRY_DIR/MIEWB_RESULTS_DIR into the .MieWB workspace).
        Returns True if launched, False if a run was already in progress
        (refused, no-op)."""
        if self.is_running():
            return False
        self._extra_env = dict(extra_env or {})

        argv = ["python3", str(RUN_PIPELINE_SCRIPT),
               "--models", str(model_path)]
        argv += [str(a) for a in (extra_args or [])]
        if steps:
            steps_str = steps if isinstance(steps, str) else ",".join(steps)
            argv += ["--steps", steps_str]

        self._stop_requested = False
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.setProgram(argv[0])
        self._proc.setArguments(argv[1:])
        self._proc.setProcessEnvironment(self._build_environment())
        self._proc.readyReadStandardOutput.connect(self._on_ready_read)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error_occurred)
        self._proc.start()
        if not self._proc.waitForStarted(5000):
            self.error.emit("failed to start run_pipeline.py")
            self._proc = None
            return False
        self.started.emit()
        return True

    def stop(self):
        """terminate(), then kill() after a grace period if still alive."""
        if self._proc is None:
            return
        self._stop_requested = True
        self._proc.terminate()
        if not self._proc.waitForFinished(STOP_GRACE_MS):
            self._proc.kill()
            self._proc.waitForFinished(1000)

    # -- P1 chunked-run contract: resume/extend ----------------------------------
    def start_resume(self, model_path, case_dir, extra_env=None,
                     steps="trace,post,viz"):
        """Resume a DEAD (interrupted mid-trace) case from its
        cengine/checkpoint.json: reissue the case's own trace options at
        the checkpoint's exact target_rays plus --resume, skipping extract
        (the case's geometry/model.json is untouched). Same single-flight
        refusal as start() -- returns False, no-op, if a run is already in
        progress. steps defaults to the full downstream refresh (post/viz
        pick up the merged result); tests override it to "trace" alone to
        stay fast."""
        from . import checkpointinfo
        config = checkpointinfo.build_resume_config(case_dir)
        args = self.build_args(config)
        return self.start(model_path, extra_args=args, steps=steps,
                          extra_env=extra_env)

    def start_extend(self, model_path, case_dir, new_rays, extra_env=None,
                     steps="trace,post,viz"):
        """Additively extend a COMPLETED C-engine case to new_rays total
        primaries (must exceed its current checkpoint target_rays -- see
        raytracer.cengine.run_c_case). Same single-flight refusal as
        start(); steps as in start_resume()."""
        from . import checkpointinfo
        config = checkpointinfo.build_extend_config(case_dir, new_rays)
        args = self.build_args(config)
        return self.start(model_path, extra_args=args, steps=steps,
                          extra_env=extra_env)

    # -- environment ------------------------------------------------------------
    def _build_environment(self):
        env = QProcessEnvironment.systemEnvironment()
        env.insert("MIEWB_PROGRESS", "1")
        if self.settings is not None:
            for key, value in self.settings.env_overrides().items():
                env.insert(key, value)
        for key, value in getattr(self, "_extra_env", {}).items():
            env.insert(key, str(value))
        return env

    # -- QProcess callbacks -------------------------------------------------------
    def _on_ready_read(self):
        proc = self._proc
        if proc is None:
            return
        while proc.canReadLine():
            raw = bytes(proc.readLine()).decode("utf-8", errors="replace")
            raw = raw.rstrip("\r\n")
            ev = common.parse_progress_line(raw)
            if ev is not None:
                self.progress.emit(ev)
            else:
                self.line.emit(raw)

    def _on_finished(self, exit_code, _exit_status):
        self.finished.emit(int(exit_code))
        self._proc = None

    def _on_error_occurred(self, qprocess_error):
        # QProcess still emits `finished` after most errors; only surface a
        # standalone error() for the "never even started" case, since a
        # normal terminate()/kill() during stop() also routes through here.
        if self._stop_requested:
            return
        self.error.emit(str(qprocess_error))

    # -- argv construction (static; no QObject state needed) ---------------------
    @staticmethod
    def build_args(config):
        """{dest: value} (as produced by ConfigMatrix.values()) -> pipeline
        argv, using cli_specs.build_parser("pipeline") as the single source
        of truth for option strings / defaults / append vs store_true vs
        scalar handling. Values equal to the parser default are skipped."""
        parser = cli_specs.build_parser("pipeline")
        args = []
        for action in parser._actions:
            dest = action.dest
            if dest in ("help", "models", "print_only"):
                continue
            if dest not in config or not action.option_strings:
                continue
            value = config[dest]
            opt = action.option_strings[-1]
            kind = type(action).__name__
            if kind == "_StoreTrueAction":
                if value:
                    args.append(opt)
            elif kind == "_AppendAction":
                for item in (value or []):
                    args.append(opt)
                    args.append(_fmt_value(item))
            else:
                if value is None or value == action.default:
                    continue
                args.append(opt)
                args.append(_fmt_value(value))
        return args


    @staticmethod
    def write_sweep_manifest(model_path, config, results_root=None):
        """Predict a sweep's variants and write
        <results_root>/<stem>/sweep-<case>.manifest.json for the Compare
        pane / compare_sweep.py. Returns the manifest path, or None when
        the config sweeps nothing. Prediction rides run_pipeline's own
        variant_output_names, which shares common.sweep_combos with
        permute_model — names cannot drift."""
        sweep_vars = list(config.get("var") or [])
        if not sweep_vars:
            return None
        import json
        import run_pipeline as rp
        stem = Path(model_path).stem
        mode = config.get("sweep_mode") or "product"
        varspecs = list(zip(sweep_vars,
                            [float(v) for v in config.get("min") or []],
                            [float(v) for v in config.get("max") or []],
                            [int(v) for v in config.get("n") or []]))
        if len({len(sweep_vars), len(config.get("min") or []),
                len(config.get("max") or []),
                len(config.get("n") or [])}) != 1:
            raise ValueError("sweep var/min/max/n counts differ")
        value_lists = [common.sweep_values(vmin, vmax, n)
                       for (_, vmin, vmax, n) in varspecs]
        combos = common.sweep_combos(value_lists, mode)
        stems = rp.variant_output_names(stem, varspecs, mode)
        case = common.case_name(config.get("preset") or "quick",
                                config.get("tag"))
        root = Path(results_root) if results_root else Path(
            os.environ.get("MIEWB_RESULTS_DIR", common.RESULTS_DIR))
        variants = []
        for vstem, combo in zip(stems, combos):
            variants.append({
                "stem": vstem,
                "values": {var: float(v)
                           for var, v in zip(sweep_vars, combo)},
                "case_dir": str(root / vstem / case),
            })
        manifest = {"model": stem, "case": case, "mode": mode,
                    "order": sweep_vars, "variants": variants}
        out = root / stem / ("sweep-%s.manifest.json" % case)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        return out


def _fmt_value(value):
    if isinstance(value, float):
        return repr(value)
    return str(value)
