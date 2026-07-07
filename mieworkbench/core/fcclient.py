"""FcClient - GUI-side client for the persistent headless FreeCAD worker.

Deliberately Qt-free (subprocess + threading) so it is unit-testable under
plain pytest and reusable from CLI tooling; the GUI wraps calls in worker
threads / signals at the pane layer.

Protocol: newline-delimited JSON to the worker's stdin; responses arrive on
stdout as lines prefixed "@FCJSON " (everything else is FreeCAD noise and is
discarded, though the last few noise lines are kept for diagnostics).

Reliability model:
  - every MUTATING op is appended to an edit journal after it succeeds;
  - if the worker dies, `ensure()` relaunches it, re-opens the documents
    that were open, and replays the journal (all ops are idempotent);
  - the journal is cleared on save/save_as (the file now holds the edits).
"""

import json
import os
import queue
import subprocess
import threading
import time

FCJSON_PREFIX = "@FCJSON "

MUTATING_OPS = {
    "set_property", "remove_property", "set_spreadsheet", "set_placement",
    "import_primitive",
}
# ops that reset the journal for a document when they succeed
JOURNAL_CLEARING_OPS = {"save", "save_as"}


class FcError(RuntimeError):
    """An op-level error reported by the worker."""

    def __init__(self, message, traceback_text=None):
        super().__init__(message)
        self.traceback_text = traceback_text


class FcDead(RuntimeError):
    """The worker process is gone / unresponsive."""


def default_freecad_appimage():
    return os.environ.get("MIEWB_FREECAD", "/home3/freecad/FreeCAD.AppImage")


def _server_script():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(
        here, "..", "..", "scripts", "fcserver", "fc_server.py"))


class FcClient:
    def __init__(self, appimage=None, server_script=None,
                 ready_timeout=60.0, op_timeout=300.0, launch_cmd=None):
        """launch_cmd overrides the full worker argv (tests use it to run a
        fake protocol server under plain python instead of the AppImage)."""
        self.appimage = appimage or default_freecad_appimage()
        self.server_script = server_script or _server_script()
        self.launch_cmd = launch_cmd or [self.appimage, "-c",
                                         self.server_script]
        self.ready_timeout = ready_timeout
        self.op_timeout = op_timeout

        self._proc = None
        self._responses = queue.Queue()
        self._reader = None
        self._stderr_reader = None
        self._noise_tail = []          # last noise lines, for diagnostics
        self._lock = threading.Lock()  # serializes request/response cycles
        self._next_id = 1
        self.ready_info = None

        # crash-recovery state
        self._open_docs = {}     # doc name -> source path
        self._journal = []       # [(doc, op, params)] mutations since save

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if self.is_alive():
            return self.ready_info
        self._responses = queue.Queue()
        self._proc = subprocess.Popen(
            self.launch_cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._drain, args=(self._proc.stderr,), daemon=True)
        self._stderr_reader.start()

        msg = self._next_message(timeout=self.ready_timeout)
        if not msg or msg.get("event") != "ready":
            self.kill()
            raise FcDead("FreeCAD worker did not become ready "
                         "(last noise: %r)" % (self._noise_tail[-3:],))
        self.ready_info = msg
        return msg

    def is_alive(self):
        return self._proc is not None and self._proc.poll() is None

    def shutdown(self):
        if not self.is_alive():
            return
        try:
            self._send({"id": self._take_id(), "op": "shutdown"})
            self._proc.wait(timeout=15)
        except Exception:
            self.kill()
        finally:
            self._proc = None
            self.ready_info = None

    def kill(self):
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None
        self.ready_info = None

    # -- public op API -----------------------------------------------------
    def request(self, op, params=None, timeout=None, _replaying=False):
        """Run one op, with transparent relaunch-and-replay on worker death."""
        with self._lock:
            try:
                result = self._request_locked(op, params, timeout)
            except FcDead:
                if _replaying:
                    raise
                self._recover_locked()
                result = self._request_locked(op, params, timeout)
            self._record(op, params, result)
            return result

    # convenience wrappers (thin; keep the op names greppable)
    def ping(self):
        return self.request("ping")

    def open_document(self, path):
        result = self.request("open_document", {"path": path})
        self._open_docs[result["doc"]] = os.path.abspath(path)
        return result

    def close(self, doc):
        result = self.request("close", {"doc": doc})
        self._open_docs.pop(doc, None)
        self._journal = [e for e in self._journal if e[0] != doc]
        return result

    # -- internals ----------------------------------------------------------
    def _take_id(self):
        rid = self._next_id
        self._next_id += 1
        return rid

    def _send(self, obj):
        if not self.is_alive():
            raise FcDead("FreeCAD worker is not running")
        try:
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise FcDead("FreeCAD worker pipe broken: %s" % exc)

    def _request_locked(self, op, params, timeout):
        if not self.is_alive():
            raise FcDead("FreeCAD worker is not running")
        rid = self._take_id()
        self._send({"id": rid, "op": op, "params": params or {}})
        deadline = time.monotonic() + (timeout or self.op_timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FcDead("timeout waiting for op %r" % op)
            msg = self._next_message(timeout=remaining)
            if msg is None:
                raise FcDead("FreeCAD worker died during op %r "
                             "(last noise: %r)" % (op, self._noise_tail[-3:]))
            if msg.get("event"):
                continue  # stray event; keep waiting
            if msg.get("id") != rid:
                continue  # stale response from a timed-out predecessor
            if not msg.get("ok"):
                raise FcError(msg.get("error", "unknown FreeCAD error"),
                              msg.get("traceback"))
            return msg.get("result")

    def _next_message(self, timeout):
        try:
            item = self._responses.get(timeout=timeout)
        except queue.Empty:
            return None if not self.is_alive() else self._raise_slow()
        return item

    def _raise_slow(self):
        raise FcDead("no response from FreeCAD worker within timeout "
                     "(worker still alive; op too slow or wedged)")

    def _read_stdout(self):
        proc = self._proc
        for line in proc.stdout:
            if line.startswith(FCJSON_PREFIX):
                try:
                    self._responses.put(json.loads(line[len(FCJSON_PREFIX):]))
                except ValueError:
                    self._note_noise(line)
            else:
                self._note_noise(line)
        # EOF -> worker exited; wake any waiter
        self._responses.put(None)

    def _drain(self, stream):
        for line in stream:
            self._note_noise(line)

    def _note_noise(self, line):
        line = line.rstrip("\n")
        if line:
            self._noise_tail.append(line)
            del self._noise_tail[:-20]

    # -- journal / crash recovery -------------------------------------------
    def _record(self, op, params, result):
        doc = (params or {}).get("doc")
        if op in MUTATING_OPS and doc:
            self._journal.append((doc, op, dict(params)))
        elif op in JOURNAL_CLEARING_OPS and doc:
            self._journal = [e for e in self._journal if e[0] != doc]
            if op == "save_as" and result and result.get("file"):
                self._open_docs[doc] = result["file"]

    def _recover_locked(self):
        """Relaunch the worker, re-open documents, replay the journal."""
        self.kill()
        self.start()
        for doc, path in list(self._open_docs.items()):
            self._request_locked("open_document", {"path": path}, None)
        for doc, op, params in list(self._journal):
            self._request_locked(op, params, None)

    def journal_length(self, doc=None):
        if doc is None:
            return len(self._journal)
        return sum(1 for e in self._journal if e[0] == doc)

    # -- context manager ------------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False
