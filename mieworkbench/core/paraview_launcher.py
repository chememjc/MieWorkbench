"""Launch interactive ParaView on a case's viz/ data (detached).

The GUI ships offscreen renders (viz/*.png); for detailed interactive
analysis the user opens the same .vtp files in full ParaView. The binary
is derived from the pinned pvpython path (its sibling 'paraview'),
overridable via settings key 'paraview'."""

import os
import sys
from glob import glob

_SCRIPTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import common  # noqa: E402


def find_paraview(settings=None):
    """Best available interactive paraview binary path, or None."""
    if settings is not None:
        explicit = settings.get("paraview")
        if explicit and os.path.isfile(explicit):
            return explicit
    sibling = os.path.join(os.path.dirname(common.PVPYTHON), "paraview")
    if os.path.isfile(sibling):
        return sibling
    for candidate in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(candidate, "paraview")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def viz_files(case_dir):
    """The ParaView-loadable artifacts of a case, rays first."""
    viz = os.path.join(str(case_dir), "viz")
    rays = sorted(glob(os.path.join(viz, "rays.vtp")))
    dets = sorted(glob(os.path.join(viz, "det_*.vtp")))
    return rays + dets


def launch(case_dir, settings=None):
    """Start ParaView detached on the case's .vtp files.

    Returns (ok, message). Uses QProcess.startDetached when a Qt app is
    running, subprocess otherwise (so it is testable headless)."""
    binary = find_paraview(settings)
    if binary is None:
        return False, ("no interactive 'paraview' binary found next to "
                       "%s (set it in Settings)" % common.PVPYTHON)
    files = viz_files(case_dir)
    if not files:
        return False, ("no viz/*.vtp files in %s — run the viz stage "
                       "first" % case_dir)
    try:
        from PySide6.QtCore import QCoreApplication, QProcess
        if QCoreApplication.instance() is not None:
            ok = QProcess.startDetached(binary, ["--data=%s" % f
                                                 for f in files])
            if not ok:
                return False, "failed to start %s" % binary
            return True, "%s (%d file(s))" % (binary, len(files))
    except ImportError:
        pass
    import subprocess
    subprocess.Popen([binary] + ["--data=%s" % f for f in files],
                     start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True, "%s (%d file(s))" % (binary, len(files))
