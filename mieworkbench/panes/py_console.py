"""PyConsolePane - an in-app Python REPL bound to the live GUI session.

A dependency-free console (stdlib `code.InteractiveConsole` + a QPlainTextEdit
transcript and a QLineEdit prompt) whose namespace holds the live objects the
window owns -- `project` (the core.project.Project session), `window`, `runner`,
and `np`. Power users can query and mutate the scene programmatically; because
every Project mutation flows through its undoable Command path, console edits get
undo/redo for free.

Design notes:
- Everything runs on the Qt main thread (synchronous). A long-running statement
  will block the event loop while it executes -- that is an accepted tradeoff for
  a zero-dependency console; there is no data race with the QProcess-based
  pipeline runner (that is a separate OS process).
- The pane is self-contained and unit-testable without an event loop: call
  `run_source(text)` directly and read `transcript_text()` (see
  tests/test_py_console.py). `set_context(**objs)` injects/updates the namespace.
- Tab completion (rlcompleter over the live namespace) and Up/Down history are
  provided by the small QLineEdit subclass below.
"""

import code
import contextlib
import io
import rlcompleter
import sys
import traceback
from collections import deque

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLineEdit, QPlainTextEdit, QVBoxLayout, QWidget,
)

_BANNER = ("MieWorkbench Python console -- `project`, `window`, `runner`, `np` "
           "are in scope. Edits via `project` are undoable.")
MAX_HISTORY = 500


class _PromptEdit(QLineEdit):
    """Single-line prompt with Up/Down history recall and Tab completion,
    delegated to the owning pane (kept here so the pane stays testable
    without simulating key events)."""

    def __init__(self, pane):
        super().__init__(pane)
        self._pane = pane

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            self._pane._recall_history(-1 if key == Qt.Key.Key_Up else +1)
            return
        if key == Qt.Key.Key_Tab:
            self._pane._complete()
            return
        super().keyPressEvent(event)


class PyConsolePane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("py_console_host")

        self._namespace = {"__name__": "__console__", "__doc__": None}
        self._console = code.InteractiveConsole(self._namespace)
        # capture InteractiveConsole's own writes (syntax errors, banners).
        self._console.write = self._append_raw
        # showtraceback must be overridden too: in Py3.10 the stock version
        # routes to sys.excepthook when a custom one is installed (pytest-qt,
        # some Qt apps), which would escape the console instead of printing
        # into it. Format runtime tracebacks straight into the transcript.
        self._console.showtraceback = self._show_traceback
        self._more = False                 # in a multi-line block?
        self._history = deque(maxlen=MAX_HISTORY)
        self._hist_pos = 0                 # index into history during recall
        self._pending = ""                 # edit-in-progress saved on recall

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        self.output.setObjectName("py_console_output")
        self.output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.prompt = _PromptEdit(self)
        self.prompt.setObjectName("py_console_prompt")
        self.prompt.setPlaceholderText(">>> ")
        self.prompt.returnPressed.connect(self._on_return)
        layout.addWidget(self.output, 1)
        layout.addWidget(self.prompt)

        self._append_raw(_BANNER + "\n")

    # -- public API ---------------------------------------------------------
    def set_context(self, **objects):
        """Inject/replace live objects in the console namespace (called by
        MainWindow with project/window/runner/np)."""
        self._namespace.update(objects)

    def transcript_text(self):
        return self.output.toPlainText()

    def run_source(self, source):
        """Feed one input line to the console (as if typed at the prompt) and
        append the echoed prompt + any stdout/stderr/traceback to the
        transcript. Returns True if the console now expects a continuation
        line (an open block), else False. Directly unit-testable."""
        self._append_raw(("... " if self._more else ">>> ") + source + "\n")
        if source.strip() and not self._more:
            if not self._history or self._history[-1] != source:
                self._history.append(source)
        self._hist_pos = len(self._history)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), \
                    contextlib.redirect_stderr(buf):
                self._more = self._console.push(source)
        except SystemExit:
            self._append_raw("(SystemExit ignored in console)\n")
            self._more = False
        out = buf.getvalue()
        if out:
            self._append_raw(out if out.endswith("\n") else out + "\n")
        self.prompt.setPlaceholderText("... " if self._more else ">>> ")
        return self._more

    # -- internals ----------------------------------------------------------
    def _on_return(self):
        source = self.prompt.text()
        self.prompt.clear()
        self.run_source(source)

    def _show_traceback(self):
        typ, val, tb = sys.exc_info()
        # drop the console's own exec frame (tb.tb_next), like the stock
        # InteractiveInterpreter.showtraceback does.
        nxt = tb.tb_next if tb is not None else None
        self._append_raw("".join(traceback.format_exception(typ, val, nxt)))

    def _append_raw(self, text):
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)
        self.output.insertPlainText(text)
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)

    def _recall_history(self, direction):
        if not self._history:
            return
        if self._hist_pos == len(self._history):
            self._pending = self.prompt.text()
        self._hist_pos = max(0, min(len(self._history),
                                    self._hist_pos + direction))
        if self._hist_pos == len(self._history):
            self.prompt.setText(self._pending)
        else:
            self.prompt.setText(self._history[self._hist_pos])

    def _complete(self):
        """rlcompleter completion over the live namespace on the token to the
        left of the cursor. Single match -> insert; multiple -> list them."""
        text = self.prompt.text()
        # token = trailing run of identifier/./attribute characters
        i = len(text)
        while i > 0 and (text[i - 1].isalnum() or text[i - 1] in "._"):
            i -= 1
        token = text[i:]
        if not token:
            return
        completer = rlcompleter.Completer(self._namespace)
        matches = []
        j = 0
        while True:
            m = completer.complete(token, j)
            if m is None:
                break
            if m not in matches:
                matches.append(m)
            j += 1
        if not matches:
            return
        if len(matches) == 1:
            self.prompt.setText(text[:i] + matches[0].rstrip("("))
        else:
            self._append_raw("    ".join(sorted(matches)) + "\n")
            # insert the longest common prefix
            prefix = matches[0]
            for m in matches[1:]:
                while not m.startswith(prefix):
                    prefix = prefix[:-1]
            if len(prefix) > len(token):
                self.prompt.setText(text[:i] + prefix)
