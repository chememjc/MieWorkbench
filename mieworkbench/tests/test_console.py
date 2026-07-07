"""ConsolePane tests: colorized append, stage filter combo, progress-line
suppression, ring-buffer clear."""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.panes.console import (  # noqa: E402
    ConsolePane, classify_stage,
)


def test_classify_stage():
    assert classify_stage("[trace] seed 1/3") == "trace"
    assert classify_stage("[post] wrote images") == "post"
    assert classify_stage("[render] overview3d ...") == "viz"
    assert classify_stage("[prep] vtkexport") == "viz"
    assert classify_stage("run_pipeline.py: 1 model(s)") == "extract"


def test_append_and_filter(qtbot):
    console = ConsolePane()
    qtbot.addWidget(console)

    console.append_line("[trace] seed 1/3")
    console.append_line("[post] wrote images")
    console.append_line("[render] overview3d ...")
    console.append_line("run_pipeline.py: notice")

    full_text = console.text_edit.toPlainText()
    assert "[trace] seed 1/3" in full_text
    assert "[post] wrote images" in full_text
    assert "[render] overview3d ..." in full_text
    assert "run_pipeline.py: notice" in full_text

    console.filter_combo.setCurrentText("trace")
    filtered = console.text_edit.toPlainText()
    assert "[trace] seed 1/3" in filtered
    assert "[post] wrote images" not in filtered
    assert "[render] overview3d ..." not in filtered
    assert "run_pipeline.py: notice" not in filtered

    console.filter_combo.setCurrentText("All")
    restored = console.text_edit.toPlainText()
    assert "[trace] seed 1/3" in restored
    assert "[post] wrote images" in restored
    assert "run_pipeline.py: notice" in restored


def test_progress_lines_never_rendered(qtbot):
    console = ConsolePane()
    qtbot.addWidget(console)

    console.append_line(
        '@MIEWB {"ev":"progress","stage":"trace","frac":0.5}')
    console.append_line("[trace] a real line")

    text = console.text_edit.toPlainText()
    assert "@MIEWB" not in text
    assert "[trace] a real line" in text
    assert len(console._buffer) == 1


def test_clear(qtbot):
    console = ConsolePane()
    qtbot.addWidget(console)

    console.append_line("[trace] hello")
    assert len(console._buffer) == 1

    console.clear()
    assert console.text_edit.toPlainText() == ""
    assert len(console._buffer) == 0
