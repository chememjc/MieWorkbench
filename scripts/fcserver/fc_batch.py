#!/usr/bin/env python3
# fc_batch.py - one-shot batch runner for fcops (CLI / tests / fallback).
#
# Usage (args via env to dodge the FreeCAD `-c` arg-parsing hang; always
# close stdin for one-shot runs):
#   FC_REQUEST_FILE=/path/req.json FC_RESPONSE_FILE=/path/resp.json \
#     "$MIEWB_FREECAD" -c scripts/fcserver/fc_batch.py < /dev/null
#
# req.json : {"ops": [{"op": "<name>", "params": {...}}, ...]}
# resp.json: {"responses": [{"ok": true, "result": ...} | {"ok": false, ...}]}
#
# Ops run in order; a failed op does NOT stop the batch (its response
# records the error) unless it carries {"required": true}.
#
# Idempotency note: the AppImage executes this script twice per invocation.
# The response file is written atomically at the end of each pass, so the
# second pass simply overwrites it with identical content computed from the
# same request - callers see one consistent result. os._exit(0) after the
# first pass prevents the second pass entirely.

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import FreeCAD  # noqa: E402
import fcops    # noqa: E402


def _log(msg):
    FreeCAD.Console.PrintMessage(msg + "\n")
    print(msg, flush=True)


def _main():
    req_path = os.environ.get("FC_REQUEST_FILE")
    resp_path = os.environ.get("FC_RESPONSE_FILE")
    if not req_path or not os.path.isfile(req_path):
        _log("fc_batch: FC_REQUEST_FILE missing or not a file")
        os._exit(2)
    with open(req_path, "r") as fh:
        request = json.load(fh)

    responses = []
    for entry in request.get("ops", []):
        ok, result = fcops.dispatch(entry.get("op"), entry.get("params"))
        resp = {"op": entry.get("op"), "ok": ok}
        if ok:
            resp["result"] = result
        else:
            resp.update(result)
        responses.append(resp)
        if not ok and entry.get("required"):
            _log("fc_batch: required op %r failed: %s"
                 % (entry.get("op"), result.get("error")))
            break

    payload = json.dumps({"responses": responses}, indent=1)
    if resp_path:
        tmp = resp_path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(payload)
        os.replace(tmp, resp_path)
        _log("fc_batch: wrote %d responses to %s"
             % (len(responses), resp_path))
    else:
        for line in payload.splitlines():
            _log(line)


try:
    _main()
finally:
    os._exit(0)
