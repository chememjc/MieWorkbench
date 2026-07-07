"""RayPreviewController - QProcess chain that builds a lightweight,
viz-only ray-overlay (.vtp) for the GUI's 3D views WITHOUT running the
physical trace (that's run_pipeline.py / RunController's job).

Pipeline (mirrors CLAUDE.md's pinned-interpreter contract for this repo):
  1. `save_copy` the live FreeCAD document to <workspace>/preview/model.FCStd
     via project.fc.request() — synchronous, goes through the persistent
     worker the project already has open (no second FreeCAD process here).
  2. FreeCAD AppImage BATCH extract (scripts/extract_geometry.py, `-c`
     mode, `--` before its own args, stdin explicitly closed — see
     CLAUDE.md's "FreeCAD `-c`" trap) -> <workspace>/preview/geometry/
     model/model.json.
  3. scripts/preview_rays.py under the optics-env python -> rays.vtp.

Deliberately mirrors core/runner.py's QProcess plumbing (env constructed
the same way, MergedChannels, errorOccurred handling) but chains two
short-lived one-shot processes instead of one long pipeline run, and has
no journal/replay machinery (this is a read-only, throwaway preview - if a
stage dies the caller just retries `start()`).

This module is Qt-only (QObject + QProcess + signals), like runner.py: it
never imports anything from scripts/raytracer/* — only scripts/common.py
(stdlib-only) for path defaults and progress-line parsing.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import common     # noqa: E402  (stdlib-only shared contract hub)

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

REPO_DIR = Path(__file__).resolve().parent.parent.parent
EXTRACT_SCRIPT = REPO_DIR / "scripts" / "extract_geometry.py"
PREVIEW_SCRIPT = REPO_DIR / "scripts" / "preview_rays.py"

STOP_GRACE_MS = 5000


def default_freecad_appimage():
    return os.environ.get("MIEWB_FREECAD", common.FREECAD_APPIMAGE)


def default_optics_python():
    return os.environ.get("MIEWB_OPTICS_PYTHON", common.OPTICS_PYTHON)


class RayPreviewController(QObject):
    """Owns at most one preview QProcess (of the two one-shot stages that
    make up a single run) at a time."""

    finished = Signal(str)      # path to rays.vtp
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = None
        self._stage = None            # "extract" | "preview" | None
        self._cancelled = False
        # per-run state carried from stage 1 (save_copy) through stage 3
        self._geometry_dir = None
        self._optical_properties = None
        self._pattern = None
        self._only_bodies = None
        self._rays_path = None
        self._output = ""

    # -- state ----------------------------------------------------------------
    def is_running(self):
        return (self._proc is not None
                and self._proc.state() != QProcess.NotRunning)

    # -- lifecycle --------------------------------------------------------------
    def start(self, project, workspace_dir, pattern="fan:n=5",
             only_bodies=None, optical_properties=None):
        """Kick off the save_copy -> extract -> preview_rays chain.

        project: the GUI's core.project.Project (must have an open
        document; project.fc is the FcClient, project.doc the open name).
        workspace_dir: a directory the caller owns (e.g. the .MieWB
        workspace); a `preview/` subdirectory is created under it and
        reused/overwritten on every call.
        pattern: a --viz-pattern spec string (scripts/common.
        parse_viz_pattern_spec), e.g. "fan:n=5" or "rings:dr=1:nper=8".
        only_bodies: iterable of body names/labels to pass through to
        preview_rays.py's --only-bodies (sources/detectors are always
        kept by that script regardless).
        optical_properties: opticalproperties/ root dir; defaults to
        common.OPTPROPS_DIR (same default the pipeline scripts use).

        Returns True if the chain was launched, False if a preview was
        already running (refused, no-op; call cancel() first).
        """
        if self.is_running():
            return False
        self._cancelled = False
        self._output = ""

        workspace_dir = Path(workspace_dir)
        preview_dir = workspace_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        model_copy = preview_dir / "model.FCStd"
        geometry_root = preview_dir / "geometry"
        self._rays_path = str(preview_dir / "rays.vtp")
        self._pattern = pattern
        self._only_bodies = list(only_bodies) if only_bodies else None
        self._optical_properties = str(
            optical_properties if optical_properties is not None
            else common.OPTPROPS_DIR)
        self._geometry_dir = str(geometry_root / model_copy.stem)

        self.progress.emit("saving model copy")
        try:
            project.fc.request(
                "save_copy", {"doc": project.doc, "path": str(model_copy)})
        except Exception as exc:
            self.failed.emit("save_copy failed: %s" % exc)
            return False

        return self._start_extract(model_copy, geometry_root)

    def cancel(self):
        """Kill whichever stage is currently running. finished()/failed()
        are NOT emitted for a cancelled run."""
        self._cancelled = True
        self._stage = None
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        proc.terminate()
        if not proc.waitForFinished(STOP_GRACE_MS):
            proc.kill()
            proc.waitForFinished(1000)

    # -- stage 1: FreeCAD batch extract -----------------------------------------
    def _start_extract(self, model_copy, geometry_root):
        self._stage = "extract"
        self.progress.emit("extracting geometry")
        argv = [default_freecad_appimage(), "-c", str(EXTRACT_SCRIPT), "--",
               "--models", str(model_copy), "--outdir", str(geometry_root)]
        proc = self._make_process(argv)
        # batch FreeCAD runs need stdin explicitly closed (CLAUDE.md: the
        # persistent fc_server worker keeps stdin open as its request
        # channel, but a one-shot `-c script -- args` batch run must not
        # inherit an open stdin or it can hang waiting on it)
        proc.setStandardInputFile(os.devnull)
        proc.finished.connect(self._on_extract_finished)
        return self._launch(proc, "failed to start FreeCAD extract")

    def _on_extract_finished(self, exit_code, _exit_status):
        proc = self._take_proc()
        if self._cancelled or proc is None:
            return
        if exit_code != 0:
            self.failed.emit("geometry extract failed (exit %d): %s"
                             % (exit_code, self._tail()))
            return
        self._start_preview()

    # -- stage 2: preview_rays.py under the optics env --------------------------
    def _start_preview(self):
        self._stage = "preview"
        self.progress.emit("tracing preview rays")
        argv = [default_optics_python(), str(PREVIEW_SCRIPT),
               "--geometry", self._geometry_dir,
               "--optical-properties", self._optical_properties,
               "--out", self._rays_path,
               "--pattern", self._pattern]
        if self._only_bodies:
            argv += ["--only-bodies", ",".join(self._only_bodies)]
        proc = self._make_process(argv)
        proc.finished.connect(self._on_preview_finished)
        self._launch(proc, "failed to start preview_rays.py")

    def _on_preview_finished(self, exit_code, _exit_status):
        proc = self._take_proc()
        if self._cancelled or proc is None:
            return
        if exit_code != 0:
            self.failed.emit("preview_rays.py failed (exit %d): %s"
                             % (exit_code, self._tail()))
            return
        self.finished.emit(self._rays_path)

    # -- shared QProcess plumbing -------------------------------------------------
    def _make_process(self, argv):
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.setProgram(argv[0])
        proc.setArguments(argv[1:])
        env = QProcessEnvironment.systemEnvironment()
        env.insert("MIEWB_PROGRESS", "1")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._on_ready_read)
        proc.errorOccurred.connect(self._on_error_occurred)
        return proc

    def _launch(self, proc, start_error):
        self._proc = proc
        proc.start()
        if not proc.waitForStarted(5000):
            self._proc = None
            self.failed.emit(start_error)
            return False
        return True

    def _take_proc(self):
        proc, self._proc = self._proc, None
        return proc

    def _on_ready_read(self):
        proc = self.sender()
        if proc is None:
            return
        while proc.canReadLine():
            raw = bytes(proc.readLine()).decode("utf-8", errors="replace")
            raw = raw.rstrip("\r\n")
            self._output += raw + "\n"
            ev = common.parse_progress_line(raw)
            if ev is not None:
                self.progress.emit(str(ev.get("msg") or ev.get("stage")))
            elif raw:
                self.progress.emit(raw)

    def _on_error_occurred(self, qprocess_error):
        if self._cancelled:
            return
        stage = self._stage
        self._proc = None
        self.failed.emit("%s process error: %s" % (stage or "preview",
                                                    qprocess_error))

    def _tail(self, n=4000):
        return self._output[-n:]
