"""UndoStack - command-pattern undo/redo over Project mutations.

Not QUndoStack: every redo/undo here is a synchronous FreeCAD-worker
round-trip that can raise (FcError, ProjectError), and the failure policy
must be explicit rather than Qt's silent best-effort:

  - push_and_do(): the command executes FIRST; if it raises, nothing is
    pushed and the exception propagates to the caller (the UI shows it).
  - undo()/redo(): if a stored command fails mid-stack, the stack no
    longer describes reality -> it is CLEARED and `error` is emitted;
    never silently diverge.

Commands carry plain callables (redo_fn/undo_fn) that must invoke the
Project's private _do_* bodies, NOT its public methods - public methods
push commands, so calling them from inside a command would re-record.

Macros group several commands into one user-visible step ("Add lens1"):
begin_macro()/end_macro(); child commands still execute immediately
through push_and_do; abort_macro() rolls back executed children in
reverse order (used when a multi-step flow fails halfway).

Depth eviction and clear() call each dropped command's `cleanup`
callable (delete_element uses it to remove stale stash .FCStd files).

Dirty tracking: mark_clean() pins the current index (called on save);
is_clean() says whether the document matches its last-saved state, and
indexChanged fires after every push/undo/redo so the owner can sync its
dirty flag.
"""

from PySide6.QtCore import QObject, Signal

DEFAULT_DEPTH = 20


class Command:
    def __init__(self, text, redo_fn, undo_fn, cleanup=None):
        self.text = text
        self.redo_fn = redo_fn
        self.undo_fn = undo_fn
        self.cleanup = cleanup

    def redo(self):
        self.redo_fn()

    def undo(self):
        self.undo_fn()

    def dispose(self):
        if self.cleanup is not None:
            try:
                self.cleanup()
            except Exception:
                pass


class MacroCommand(Command):
    def __init__(self, text):
        super().__init__(text, None, None)
        self.children = []

    def redo(self):
        for cmd in self.children:
            cmd.redo()

    def undo(self):
        for cmd in reversed(self.children):
            cmd.undo()

    def dispose(self):
        for cmd in self.children:
            cmd.dispose()


class UndoStack(QObject):
    canUndoChanged = Signal(bool, str)   # (enabled, command text)
    canRedoChanged = Signal(bool, str)
    indexChanged = Signal()
    error = Signal(str)                  # mid-stack failure; stack cleared

    def __init__(self, depth=DEFAULT_DEPTH, parent=None):
        super().__init__(parent)
        self.depth = depth
        self._stack = []          # oldest .. newest
        self._index = 0           # commands [0.._index) are applied
        self._clean_index = 0     # index at last save; -1 = unreachable
        self._macro = None

    # -- recording -----------------------------------------------------------
    def push_and_do(self, command):
        """Execute, then record. On exception nothing is recorded."""
        command.redo()
        self._push_done(command)

    def push_done(self, command):
        """Record a command whose effect has ALREADY been applied (used by
        apply_operation, which executes the move before recording it)."""
        self._push_done(command)

    def _push_done(self, command):
        if self._macro is not None:
            self._macro.children.append(command)
            return
        # a new command invalidates the redo tail
        for cmd in self._stack[self._index:]:
            cmd.dispose()
        del self._stack[self._index:]
        if self._clean_index > self._index:
            self._clean_index = -1    # the saved state is no longer reachable
        self._stack.append(command)
        self._index += 1
        while len(self._stack) > self.depth:
            self._stack.pop(0).dispose()
            self._index -= 1
            self._clean_index = (self._clean_index - 1
                                 if self._clean_index > 0 else -1)
        self._notify()

    # -- macros ----------------------------------------------------------------
    def begin_macro(self, text):
        if self._macro is not None:
            raise RuntimeError("macro already open")
        self._macro = MacroCommand(text)

    def end_macro(self):
        macro, self._macro = self._macro, None
        if macro is None:
            raise RuntimeError("no macro open")
        if macro.children:
            self._push_done(macro)

    def abort_macro(self):
        """Roll back the executed children of the open macro (reverse
        order) and drop it. Rollback failures are reported, not raised."""
        macro, self._macro = self._macro, None
        if macro is None:
            return
        try:
            macro.undo()
        except Exception as exc:
            self.error.emit("could not roll back %r: %s" % (macro.text, exc))
        macro.dispose()

    def in_macro(self):
        return self._macro is not None

    # -- walking -----------------------------------------------------------------
    def can_undo(self):
        return self._macro is None and self._index > 0

    def can_redo(self):
        return self._macro is None and self._index < len(self._stack)

    def undo_text(self):
        return self._stack[self._index - 1].text if self.can_undo() else ""

    def redo_text(self):
        return self._stack[self._index].text if self.can_redo() else ""

    def undo(self):
        if not self.can_undo():
            return False
        cmd = self._stack[self._index - 1]
        try:
            cmd.undo()
        except Exception as exc:
            self._fail("undo of %r failed: %s" % (cmd.text, exc))
            return False
        self._index -= 1
        self._notify()
        return True

    def redo(self):
        if not self.can_redo():
            return False
        cmd = self._stack[self._index]
        try:
            cmd.redo()
        except Exception as exc:
            self._fail("redo of %r failed: %s" % (cmd.text, exc))
            return False
        self._index += 1
        self._notify()
        return True

    def _fail(self, message):
        # the document no longer matches the stack; drop everything
        self.clear()
        self.error.emit(message)

    # -- state ----------------------------------------------------------------
    def clear(self):
        for cmd in self._stack:
            cmd.dispose()
        self._stack = []
        self._index = 0
        self._clean_index = 0
        self._macro = None
        self._notify()

    def mark_clean(self):
        self._clean_index = self._index
        self.indexChanged.emit()

    def is_clean(self):
        return self._index == self._clean_index

    def _notify(self):
        self.canUndoChanged.emit(self.can_undo(), self.undo_text())
        self.canRedoChanged.emit(self.can_redo(), self.redo_text())
        self.indexChanged.emit()
