"""ConsolePane - the bottom dock's log view for pipeline runs.

Self-contained (no dependency on core.runner): MainWindow wires
RunController.line(str) into append_line() and lets this pane own all the
presentation logic (colorizing, filtering, scrollback). That keeps the
pane reusable/testable on its own (see tests/test_console.py) and keeps
RunController Qt-plumbing-only.

Every incoming line is kept in an in-memory ring buffer (last 20000 lines,
each tagged with the pipeline stage it looks like it came from) so the
stage filter combo can re-render the visible text without re-running the
pipeline. Stage tagging is a best-effort read of the log-line prefixes the
stage scripts actually print (see scripts/common.py's PROGRESS_PREFIX doc
and each stage script's print() calls):

    [trace]                  -> "trace"   (run_trace.py)
    [post]                   -> "post"    (post_process.py)
    [prep] [setup] [render] [done] -> "viz"  (make_viz.py)
    [optimize]               -> "optimize" (optimize.py)
    anything else            -> "extract" (extract_geometry.py's plain
                                 prints/WARNING/ERROR carry no bracket tag,
                                 and so does run_pipeline.py's own batch-
                                 level narration; both are bucketed here
                                 as a catch-all rather than inventing a
                                 fifth filter category)

'@MIEWB {...}' progress lines are consumed entirely by RunController (they
become progress() signal emissions, never line() emissions) but
append_line() also refuses them defensively, so this pane is safe even if
something upstream forwards a raw stdout line unfiltered.
"""

import html
from collections import deque

from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout,
    QWidget,
)

PROGRESS_PREFIX = "@MIEWB "
MAX_LINES = 20000

STAGE_CHOICES = ["All", "extract", "trace", "post", "viz", "optimize",
                 "tolerance"]

_VIZ_PREFIXES = ("[prep]", "[setup]", "[render]", "[done]")

_COLOR_DEFAULT = "#d4d4d4"
_COLOR_TRACE = "#22d3ee"     # cyan
_COLOR_POST = "#e879f9"      # magenta
_COLOR_VIZ = "#eab308"       # yellow
_COLOR_OPTIMIZE = "#34d399"   # green
_COLOR_TOLERANCE = "#a78bfa"  # violet
_COLOR_ERROR = "#f87171"     # red
_COLOR_NOTICE = "#fb923c"    # orange


def classify_stage(text):
    if text.startswith("[trace]"):
        return "trace"
    if text.startswith("[post]"):
        return "post"
    if text.startswith(_VIZ_PREFIXES):
        return "viz"
    if text.startswith("[optimize]"):
        return "optimize"
    if text.startswith("[tolerance]"):
        return "tolerance"
    return "extract"


def classify_color(text):
    if "FAILED" in text or "ERROR" in text:
        return _COLOR_ERROR
    if text.startswith("[trace]"):
        return _COLOR_TRACE
    if text.startswith("[post]"):
        return _COLOR_POST
    if text.startswith(_VIZ_PREFIXES):
        return _COLOR_VIZ
    if text.startswith("[optimize]"):
        return _COLOR_OPTIMIZE
    if text.startswith("[tolerance]"):
        return _COLOR_TOLERANCE
    if "NOTICE" in text:
        return _COLOR_NOTICE
    return _COLOR_DEFAULT


class ConsolePane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._buffer = deque(maxlen=MAX_LINES)   # (stage, text, color)
        self._at_bottom = True

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(STAGE_CHOICES)
        self.filter_combo.setToolTip(
            "Show only log lines from one pipeline stage")
        self.filter_combo.currentTextChanged.connect(self._rerender)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Clear the console (does not stop "
                                     "a running pipeline)")
        self.clear_button.clicked.connect(self.clear)

        top_row = QHBoxLayout()
        top_row.addWidget(self.filter_combo)
        top_row.addStretch(1)
        top_row.addWidget(self.clear_button)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setMaximumBlockCount(MAX_LINES)
        self.text_edit.setToolTip("Pipeline stdout/stderr, colorized by "
                                  "stage")
        font = self.text_edit.font()
        font.setFamily("monospace")
        font.setStyleHint(font.StyleHint.Monospace)
        self.text_edit.setFont(font)
        self.text_edit.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: %s; }"
            % _COLOR_DEFAULT)
        self.text_edit.verticalScrollBar().valueChanged.connect(
            self._on_scroll)

        layout = QVBoxLayout(self)
        layout.addLayout(top_row)
        layout.addWidget(self.text_edit)

    # -- public API -------------------------------------------------------------
    def append_line(self, text):
        """Add one already-split line of pipeline output. Progress lines
        ('@MIEWB ...') are dropped - they're consumed by RunController."""
        if text.startswith(PROGRESS_PREFIX):
            return
        stage = classify_stage(text)
        color = classify_color(text)
        self._buffer.append((stage, text, color))
        if self._matches_filter(stage):
            self._render_one(text, color)

    def clear(self):
        self._buffer.clear()
        self.text_edit.clear()

    # -- filtering / rendering ---------------------------------------------------
    def _matches_filter(self, stage):
        current = self.filter_combo.currentText()
        return current == "All" or current == stage

    def _rerender(self, _current_text=None):
        self.text_edit.clear()
        for stage, text, color in self._buffer:
            if self._matches_filter(stage):
                self._render_one(text, color)

    def _render_one(self, text, color):
        was_bottom = self._at_bottom
        cursor = self.text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.text_edit.setTextCursor(cursor)
        cursor.insertHtml(
            '<span style="color:%s;">%s</span><br/>'
            % (color, html.escape(text)))
        if was_bottom:
            sb = self.text_edit.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _on_scroll(self, value):
        sb = self.text_edit.verticalScrollBar()
        self._at_bottom = value >= sb.maximum() - 2
