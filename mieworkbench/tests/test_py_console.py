"""Unit tests for the in-app Python console (panes/py_console.py): expression
evaluation, statement/multi-line handling, error capture, namespace injection
(project/window/runner), command history, and tab completion -- all driven
directly (no event loop), matching the offscreen GUI-test discipline.
"""
import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from PySide6.QtCore import Qt  # noqa: E402

from mieworkbench.panes.py_console import PyConsolePane  # noqa: E402


def test_evaluates_expression(qtbot):
    pane = PyConsolePane()
    qtbot.addWidget(pane)
    pane.run_source("1 + 2")
    assert "3" in pane.transcript_text()


def test_statement_and_stdout(qtbot):
    pane = PyConsolePane()
    qtbot.addWidget(pane)
    pane.run_source("x = 40 + 2")
    pane.run_source("print(x)")
    assert "42" in pane.transcript_text()


def test_multiline_block(qtbot):
    pane = PyConsolePane()
    qtbot.addWidget(pane)
    assert pane.run_source("for i in range(3):") is True     # wants more
    assert pane.run_source("    print(i * 10)") is True
    assert pane.run_source("") is False                      # block closes
    t = pane.transcript_text()
    assert "0" in t and "10" in t and "20" in t


def test_error_is_captured_not_raised(qtbot):
    pane = PyConsolePane()
    qtbot.addWidget(pane)
    pane.run_source("1/0")                                   # must not raise
    assert "ZeroDivisionError" in pane.transcript_text()


def test_namespace_injection(qtbot):
    pane = PyConsolePane()
    qtbot.addWidget(pane)
    sentinel = object()
    pane.set_context(project=sentinel, answer=42)
    pane.run_source("answer")
    assert "42" in pane.transcript_text()
    pane.run_source("project is not None")
    assert "True" in pane.transcript_text()


def test_history_recall(qtbot):
    pane = PyConsolePane()
    qtbot.addWidget(pane)
    pane.run_source("aaa = 1")
    pane.run_source("bbb = 2")
    pane._recall_history(-1)                                 # Up -> last cmd
    assert pane.prompt.text() == "bbb = 2"
    pane._recall_history(-1)                                 # Up -> older
    assert pane.prompt.text() == "aaa = 1"
    pane._recall_history(+1)                                 # Down -> newer
    assert pane.prompt.text() == "bbb = 2"


def test_tab_completion_single_match(qtbot):
    pane = PyConsolePane()
    qtbot.addWidget(pane)
    pane.set_context(unique_symbol_xyz=123)
    pane.prompt.setText("unique_sym")
    pane._complete()
    assert pane.prompt.text() == "unique_symbol_xyz"


def test_systemexit_is_swallowed(qtbot):
    pane = PyConsolePane()
    qtbot.addWidget(pane)
    pane.run_source("raise SystemExit")                      # must not exit
    assert "SystemExit" in pane.transcript_text()


def test_console_wired_into_mainwindow(qtbot):
    # end-to-end "working demo": the app's console is bound to the LIVE
    # session objects and can execute against them.
    from mieworkbench.mainwindow import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    ns = w.py_console._namespace
    assert ns["project"] is w.project
    assert ns["window"] is w
    assert ns["runner"] is w.runner
    w.py_console.run_source("project.is_open()")             # live Project call
    assert "False" in w.py_console.transcript_text()         # no doc open yet
    w.py_console.run_source("np.array([1, 2, 3]).sum()")     # np in scope
    assert "6" in w.py_console.transcript_text()
