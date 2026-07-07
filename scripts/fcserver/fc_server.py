#!/usr/bin/env python3
# fc_server.py - persistent headless FreeCAD worker for MieWorkbench.
#
# Launch (stdin is the request channel - do NOT redirect from /dev/null):
#   /home3/freecad/FreeCAD.AppImage -c scripts/fcserver/fc_server.py
#
# Protocol: newline-delimited JSON.
#   request  : {"id": N, "op": "<name>", "params": {...}}
#   response : @FCJSON {"id": N, "ok": true, "result": {...}}
#            | @FCJSON {"id": N, "ok": false, "error": "...", "traceback": "..."}
#   events   : @FCJSON {"event": "ready", "protocol": 1, "freecad": "...", "pid": N}
#
# Every protocol line is prefixed "@FCJSON " so clients can discard FreeCAD's
# own console noise (QThreadStorage warnings etc.). Anything on stdout
# WITHOUT the prefix is noise; anything on stderr is noise.
#
# Headless traps handled (see freecad-headless skill / CLAUDE.md):
#   - `-c` executes the script TWICE (headless pass, then a GUI-spinup pass).
#     The serve loop runs in the first pass and finishes with os._exit(0)
#     at EOF/shutdown, so the second pass never begins.
#   - No `if __name__ == "__main__"` guard (would silently skip everything).
#   - Never sys.exit (swallowed); always os._exit.
#   - print() can drop under the AppImage: protocol writes go through
#     os.write on fd 1 with an explicit flush-free single write.

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import FreeCAD  # noqa: E402
import fcops    # noqa: E402

PROTOCOL_VERSION = 1


def _emit(obj):
    # The LEADING newline matters: FreeCAD progress observers print noise
    # to stdout WITHOUT a trailing newline ("Importing project files...."),
    # and a response appended to such a partial line would not start with
    # the @FCJSON prefix and be discarded as noise by the client (a real,
    # history-dependent 300s-timeout bug). Clients skip blank lines.
    line = "\n@FCJSON " + json.dumps(obj, separators=(",", ":")) + "\n"
    # single atomic write straight to fd 1; bypasses the AppImage's
    # sometimes-lossy sys.stdout buffering
    os.write(1, line.encode("utf-8", "replace"))


def _serve():
    _emit({"event": "ready", "protocol": PROTOCOL_VERSION,
           "freecad": ".".join(FreeCAD.Version()[0:3]),
           "pid": os.getpid()})
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except ValueError as exc:
            _emit({"id": None, "ok": False,
                   "error": "bad JSON request: %s" % exc})
            continue
        rid = req.get("id")
        op = req.get("op")
        if op == "shutdown":
            _emit({"id": rid, "ok": True, "result": {"shutdown": True}})
            break
        ok, result = fcops.dispatch(op, req.get("params"))
        resp = {"id": rid, "ok": ok}
        if ok:
            resp["result"] = result
        else:
            resp.update(result)
        try:
            _emit(resp)
        except (TypeError, ValueError):
            # non-serializable result - never die on a response
            _emit({"id": rid, "ok": False,
                   "error": "op %r returned non-JSON-serializable data" % op})
    _emit({"event": "bye", "pid": os.getpid()})


try:
    _serve()
finally:
    # first pass ends here; never let FreeCAD start the second (GUI) pass
    os._exit(0)
