#!/usr/bin/env python3
"""A stand-in for fc_server.py used by fcclient unit tests.

Speaks the same @FCJSON protocol over stdin/stdout, mixed with FreeCAD-style
console noise, without needing the FreeCAD AppImage. Behaviours are keyed
off the op name / params so tests can provoke errors, slowness, and crashes.
"""

import json
import os
import sys

OPEN_DOCS = {}


def emit(obj):
    os.write(1, ("@FCJSON " + json.dumps(obj) + "\n").encode())


print("FreeCAD 9.9.9 fake console banner")           # noise
print("QThreadStorage: entry 3 destroyed", file=sys.stderr)  # noise
emit({"event": "ready", "protocol": 1, "freecad": "fake", "pid": os.getpid()})

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    req = json.loads(raw)
    rid, op = req.get("id"), req.get("op")
    params = req.get("params") or {}
    if op == "shutdown":
        emit({"id": rid, "ok": True, "result": {"shutdown": True}})
        break
    if op == "ping":
        print("interleaved noise line")               # noise mid-stream
        emit({"id": rid, "ok": True, "result": {"pong": True}})
    elif op == "open_document":
        name = os.path.splitext(os.path.basename(params["path"]))[0]
        OPEN_DOCS[name] = params["path"]
        emit({"id": rid, "ok": True,
              "result": {"doc": name, "bodies": [], "sheets": [],
                         "file": params["path"]}})
    elif op == "boom":
        emit({"id": rid, "ok": False, "error": "OpError: kaboom",
              "traceback": "Traceback ..."})
    elif op == "die":
        os._exit(9)                                   # simulate worker crash
    elif op == "set_property":
        # record mutations so the replay test can count them
        emit({"id": rid, "ok": True,
              "result": {"changed_bodies": [], "moved_bodies": [],
                         "invalid": [], "placements": {},
                         "open_docs": sorted(OPEN_DOCS),
                         "marker": params.get("value")}})
    elif op == "save":
        emit({"id": rid, "ok": True, "result": {"file": "fake.FCStd"}})
    else:
        emit({"id": rid, "ok": False, "error": "unknown op %r" % op})

emit({"event": "bye"})
