"""ToleranceController - Qt-only wrapper around QProcess for driving
scripts/tolerance.py (sensitivity + Monte-Carlo tolerancing) from the GUI.

The mirror image of core/optimize_controller.py's OptimizeController:

* the program is the OPTICS env python (common.OPTICS_PYTHON —
  tolerance.py builds on fast_eval + optimize.py's scipy engine), not
  system python3;
* build_args() walks cli_specs.build_parser("tolerance"), so the pane's
  config dict and the real CLI can never drift.

Protocol: MIEWB_PROGRESS=1 is set so tolerance.py emits
'@MIEWB {"stage":"tolerance",...}' lines — per-sensitivity-parameter
events, one phase="sensitivity_done" event carrying the compact ranked
table (the TolerancePane bar chart's feed), and per-draw phase="mc"
events with frac = draws done / N plus merit/yield extras (the yield
histogram's feed); those become progress(dict) signals. Everything else
becomes line(str) for the console pane. Cancellation mirrors
RunController: terminate(), then kill() after a grace period.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import common     # noqa: E402  (stdlib-only shared contract hub)
import cli_specs   # noqa: E402  (stdlib-only; single source of truth)

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

REPO_DIR = Path(__file__).resolve().parent.parent.parent
TOLERANCE_SCRIPT = REPO_DIR / "scripts" / "tolerance.py"

STOP_GRACE_MS = 5000


class ToleranceController(QObject):
    """Owns at most one scripts/tolerance.py QProcess at a time."""

    started = Signal()
    line = Signal(str)
    progress = Signal(dict)
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, settings=None, parent=None, python=None,
                 script=None):
        super().__init__(parent)
        self.settings = settings   # core.settings.Settings, or None
        self.python = str(python or common.OPTICS_PYTHON)
        self.script = str(script or TOLERANCE_SCRIPT)
        self._proc = None
        self._stop_requested = False

    # -- state ----------------------------------------------------------------
    def is_running(self):
        return (self._proc is not None
                and self._proc.state() != QProcess.NotRunning)

    # -- lifecycle --------------------------------------------------------------
    def start(self, model_path, extra_args=None, extra_env=None):
        """Launch tolerance.py --model <model_path> <extra_args>. Returns
        True if launched, False if a tolerance run was already in
        progress (refused, no-op)."""
        if self.is_running():
            return False
        self._extra_env = dict(extra_env or {})

        argv = [self.python, self.script, "--model", str(model_path)]
        argv += [str(a) for a in (extra_args or [])]

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
            self.error.emit("failed to start tolerance.py")
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
        """{dest: value} (as produced by TolerancePane.config()) ->
        tolerance.py argv, using cli_specs.build_parser("tolerance") as
        the single source of truth for option strings / defaults /
        append vs scalar handling. Values equal to the parser default are
        skipped; --model is passed separately by start()."""
        parser = cli_specs.build_parser("tolerance")
        args = []
        for action in parser._actions:
            dest = action.dest
            if dest in ("help", "model"):
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
