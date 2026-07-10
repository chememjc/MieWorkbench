"""app.py - MieWorkbench entry point.

`python -m mieworkbench [model.FCStd]` builds the QApplication, the
MainWindow, and runs the Qt event loop. The optional positional argument
is just stashed on the window (open_model()) for later phases - this
phase does not yet load geometry from it.

MIEWB_SMOKE=1 makes main() auto-quit after 3 seconds instead of blocking
on the event loop forever, so `-m mieworkbench` can be used as a
non-interactive smoke test (see mieworkbench/tests and the verification
step in the build task): launch, build every pane, tear down cleanly,
exit 0.
"""

import argparse
import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .mainwindow import MainWindow

SMOKE_QUIT_MS = 3000


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)

    app = QApplication(argv)
    app.setApplicationName("MieWorkbench")
    app.setOrganizationName("CurtisAnalytical")

    parser = argparse.ArgumentParser(
        prog="mieworkbench",
        description="MieWorkbench - optical ray-tracing pipeline GUI")
    parser.add_argument(
        "model", nargs="?", default=None,
        help="optional .FCStd/.MieWB/.MieSim model to open at startup")
    args = parser.parse_args(argv[1:])

    window = MainWindow()
    if args.model:
        window.open_model(args.model)
    window.show()

    # quit paths that bypass MainWindow.closeEvent (app.quit(), SMOKE)
    # still must release VTK interactors/timers AND the FreeCAD worker or
    # the interpreter hangs after exec() returns; both are idempotent
    def _teardown():
        window.shutdown_resources()
        try:
            window.project.shutdown()
        except Exception:
            pass
    app.aboutToQuit.connect(_teardown)

    if os.environ.get("MIEWB_SMOKE") == "1":
        QTimer.singleShot(SMOKE_QUIT_MS, app.quit)

    return app.exec()
