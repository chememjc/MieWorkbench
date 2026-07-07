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
    def start(self, model_path, extra_args=None, steps=None):
        """Launch run_pipeline.py --models <model_path> <extra_args>
        (+ --steps <steps> if given). Returns True if launched, False if a
        run was already in progress (refused, no-op)."""
        if self.is_running():
            return False

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

    # -- environment ------------------------------------------------------------
    def _build_environment(self):
        env = QProcessEnvironment.systemEnvironment()
        env.insert("MIEWB_PROGRESS", "1")
        if self.settings is not None:
            for key, value in self.settings.env_overrides().items():
                env.insert(key, value)
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


def _fmt_value(value):
    if isinstance(value, float):
        return repr(value)
    return str(value)
