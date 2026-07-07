"""PreviewScheduler - debounced auto-refresh policy for the live ray
preview.

Pure Qt policy object; it knows nothing about subprocesses. The
mainwindow wires it between Project.opticsChanged and
RayPreviewController:

    project.opticsChanged        -> scheduler.notify_change()
    raypreview.finished/failed   -> scheduler.notify_run_finished()
    scheduler.previewWanted      -> mainwindow launches the preview chain
                                    (guarding on raypreview.is_running(),
                                    no pipeline run in flight, etc.) and
                                    calls notify_busy(True/False) around
                                    it.

Policy:
- DEBOUNCE-RESTART: every notify_change() restarts a single-shot timer
  (default 1000 ms); a burst of edits (drag, macro, multi-step undo)
  coalesces into one previewWanted after the scene goes idle.
- QUEUE-ONE-MORE, NEVER CANCEL: if the chain is busy when the timer
  fires, remember exactly one pending request and re-arm the debounce
  when the run finishes -- the extract stage is a whole FreeCAD launch,
  so cancel-and-restart on every keystroke would thrash the machine, and
  the preview chain refuses concurrent starts anyway. At most one stale
  chain-length of lag.
- NO LOOPS BY CONSTRUCTION: preview completion mutates nothing in the
  Project, so finishing a run can only service the pending flag, never
  create one.
- Disabling stops the timer and drops any pending request.
"""

from PySide6.QtCore import QObject, QTimer, Signal

DEFAULT_DEBOUNCE_MS = 1000


class PreviewScheduler(QObject):
    previewWanted = Signal()

    def __init__(self, debounce_ms=DEFAULT_DEBOUNCE_MS, parent=None):
        super().__init__(parent)
        self._enabled = True
        self._busy = False
        self._pending = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(int(debounce_ms))
        self._timer.timeout.connect(self._on_timeout)

    # -- configuration ----------------------------------------------------
    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        if not self._enabled:
            self._timer.stop()
            self._pending = False

    def is_enabled(self):
        return self._enabled

    # -- inputs -------------------------------------------------------------
    def notify_change(self):
        """An optics-affecting edit happened: (re)start the debounce."""
        if not self._enabled:
            return
        self._timer.start()

    def notify_busy(self, busy):
        """Mirror of the preview chain's running state (the mainwindow
        calls notify_busy(True) when it actually launches a run)."""
        self._busy = bool(busy)

    def notify_run_finished(self):
        """A preview run ended (success or failure). Service the pending
        request, if any, by re-arming the debounce -- a run kicked off
        during continued editing still waits for idle."""
        self._busy = False
        if self._pending and self._enabled:
            self._pending = False
            self._timer.start()

    notify_run_failed = notify_run_finished

    # -- internals ------------------------------------------------------------
    def _on_timeout(self):
        if not self._enabled:
            return
        if self._busy:
            self._pending = True
            return
        self.previewWanted.emit()

    # test/introspection hooks
    def has_pending(self):
        return self._pending

    def timer_active(self):
        return self._timer.isActive()
