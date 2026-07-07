"""PreviewScheduler policy tests: debounce-restart, queue-one-more,
enable/disable, and the no-loop property. Uses a 0/short-interval timer +
qtbot.waitUntil so nothing sleeps for a real second."""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.previewscheduler import PreviewScheduler  # noqa: E402


def _make(qtbot, debounce_ms=0):
    sched = PreviewScheduler(debounce_ms=debounce_ms)
    fired = []
    sched.previewWanted.connect(lambda: fired.append(1))
    return sched, fired


def test_change_fires_once_after_idle(qtbot):
    sched, fired = _make(qtbot)
    sched.notify_change()
    qtbot.waitUntil(lambda: len(fired) == 1, timeout=2000)
    qtbot.wait(20)
    assert fired == [1]   # one edit -> exactly one request


def test_burst_of_changes_coalesces(qtbot):
    sched, fired = _make(qtbot, debounce_ms=50)
    for _ in range(10):
        sched.notify_change()
    qtbot.waitUntil(lambda: len(fired) == 1, timeout=2000)
    qtbot.wait(80)
    assert fired == [1]


def test_busy_at_timeout_queues_exactly_one_rerun(qtbot):
    sched, fired = _make(qtbot)
    sched.notify_busy(True)
    sched.notify_change()
    qtbot.waitUntil(sched.has_pending, timeout=2000)
    assert fired == []            # not fired while busy
    sched.notify_change()         # more edits while busy
    qtbot.wait(20)
    assert sched.has_pending()

    sched.notify_run_finished()   # run ends -> pending re-arms debounce
    qtbot.waitUntil(lambda: len(fired) == 1, timeout=2000)
    qtbot.wait(20)
    assert fired == [1]           # exactly one queued rerun
    assert not sched.has_pending()


def test_run_finished_with_nothing_pending_is_silent(qtbot):
    sched, fired = _make(qtbot)
    sched.notify_busy(True)
    sched.notify_run_finished()
    qtbot.wait(20)
    assert fired == []            # completion alone never triggers a run


def test_disable_stops_timer_and_drops_pending(qtbot):
    sched, fired = _make(qtbot, debounce_ms=50)
    sched.notify_change()
    sched.set_enabled(False)
    qtbot.wait(80)
    assert fired == []
    assert not sched.timer_active()

    sched.notify_busy(True)
    sched.notify_change()         # ignored while disabled
    sched.notify_run_finished()
    qtbot.wait(20)
    assert fired == []

    sched.set_enabled(True)
    sched.notify_busy(False)
    sched.notify_change()
    qtbot.waitUntil(lambda: len(fired) == 1, timeout=2000)


def test_reenabling_does_not_resurrect_old_pending(qtbot):
    sched, fired = _make(qtbot)
    sched.notify_busy(True)
    sched.notify_change()
    qtbot.waitUntil(sched.has_pending, timeout=2000)
    sched.set_enabled(False)
    assert not sched.has_pending()
    sched.set_enabled(True)
    sched.notify_run_finished()
    qtbot.wait(20)
    assert fired == []
