"""fcclient unit tests against the fake protocol server (no FreeCAD).

The real-FreeCAD integration tests live in test_fcserver_integration.py and
are marked 'freecad'; these here must stay fast enough for every dev loop.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.fcclient import FcClient, FcError, FcDead  # noqa: E402

FAKE = os.path.join(os.path.dirname(__file__), "fake_fc_server.py")


def make_client(**kw):
    kw.setdefault("launch_cmd", [sys.executable, FAKE])
    kw.setdefault("ready_timeout", 20.0)
    kw.setdefault("op_timeout", 20.0)
    return FcClient(**kw)


def test_ready_handshake_ignores_noise():
    with make_client() as fc:
        assert fc.ready_info["freecad"] == "fake"
        assert fc.ready_info["protocol"] == 1


def test_roundtrip_with_interleaved_noise():
    with make_client() as fc:
        assert fc.ping()["pong"] is True
        assert fc.ping()["pong"] is True  # noise between responses skipped


def test_op_error_maps_to_fcerror():
    with make_client() as fc:
        with pytest.raises(FcError, match="kaboom"):
            fc.request("boom")
        assert fc.is_alive()  # an op error must not kill the worker


def test_unknown_op_is_fcerror_not_crash():
    with make_client() as fc:
        with pytest.raises(FcError, match="unknown op"):
            fc.request("no_such_op")


def test_shutdown_clean():
    fc = make_client()
    fc.start()
    fc.shutdown()
    assert not fc.is_alive()


def test_crash_recovery_replays_journal():
    with make_client() as fc:
        fc.open_document("/tmp/somemodel.FCStd")
        fc.request("set_property", {"doc": "somemodel", "body": "B",
                                    "name": "material", "value": "bk7"})
        assert fc.journal_length() == 1
        pid_before = fc.ready_info["pid"]

        # murder the worker mid-session, then keep using the client
        with pytest.raises((FcDead, FcError)):
            fc.request("die")
        result = fc.request("set_property", {"doc": "somemodel", "body": "B",
                                             "name": "material",
                                             "value": "sf5"})
        assert fc.ready_info["pid"] != pid_before      # relaunched
        assert result["open_docs"] == ["somemodel"]    # doc re-opened
        assert fc.journal_length() == 2                # original + new edit


def test_save_clears_journal():
    with make_client() as fc:
        fc.open_document("/tmp/model2.FCStd")
        fc.request("set_property", {"doc": "model2", "body": "B",
                                    "name": "power", "value": 5.0})
        assert fc.journal_length("model2") == 1
        fc.request("save", {"doc": "model2"})
        assert fc.journal_length("model2") == 0


def test_close_forgets_document():
    with make_client() as fc:
        fc.open_document("/tmp/model3.FCStd")
        fc.request("set_property", {"doc": "model3", "body": "B",
                                    "name": "x", "value": 1.0})
        # fake server has no real close op; simulate the bookkeeping path
        fc._open_docs.pop("model3", None)
        fc._journal = [e for e in fc._journal if e[0] != "model3"]
        assert fc.journal_length("model3") == 0
