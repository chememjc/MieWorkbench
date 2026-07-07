"""RayPreviewController tests: the save_copy -> FreeCAD-batch-extract ->
preview_rays.py chain, driven entirely by stub executables (no real
FreeCAD, no real optics env) so these stay fast/offline like the rest of
mieworkbench/tests. The stubs mimic the two real subprocess programs'
calling convention closely enough to exercise the REAL argv-building and
QProcess-sequencing code in core/raypreview.py:

  fake appimage:  argv = ["-c", "<extract_script>", "--", "--models", M,
                          "--outdir", D]  (a real FreeCAD.AppImage's own
                  '-c <script>' console-mode convention, not python's -c)
  fake optics py: argv = ["<preview_script>", "--geometry", G,
                          "--optical-properties", P, "--out", O,
                          "--pattern", SPEC, [--only-bodies ...]]

A real end-to-end run (real FreeCAD + real optics env) is covered by the
smoke test called out in the task instructions, run manually/CI-side; it
is intentionally NOT part of this fast offline suite.
"""

import json
import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core import raypreview                    # noqa: E402
from mieworkbench.core.raypreview import RayPreviewController  # noqa: E402


STUB_APPIMAGE_OK = """#!/usr/bin/env python3
import json
import os
import sys

argv = sys.argv[1:]
i = argv.index("--")
rest = argv[i + 1:]
d = {}
it = iter(rest)
for tok in it:
    if tok.startswith("--"):
        d[tok[2:]] = next(it)
outdir = d["outdir"]
model = d["models"]
stem = os.path.splitext(os.path.basename(model))[0]
os.makedirs(os.path.join(outdir, stem), exist_ok=True)
with open(os.path.join(outdir, stem, "model.json"), "w") as fh:
    json.dump({"stub": True}, fh)
print("stub extract ok")
sys.exit(0)
"""

STUB_APPIMAGE_FAIL = """#!/usr/bin/env python3
import sys
print("stub extract FAILED on purpose", file=sys.stderr)
sys.exit(3)
"""

STUB_APPIMAGE_SLEEP = """#!/usr/bin/env python3
import time
time.sleep(30)
"""

STUB_OPTICS_OK = """#!/usr/bin/env python3
import sys

argv = sys.argv[2:]   # skip our own path + the preview_rays.py placeholder
d = {}
it = iter(argv)
for tok in it:
    if tok.startswith("--"):
        d[tok[2:]] = next(it)
with open(d["out"], "w") as fh:
    fh.write("<VTKFile/>")
print("stub preview ok pattern=%s" % d.get("pattern"))
sys.exit(0)
"""

STUB_OPTICS_FAIL = """#!/usr/bin/env python3
import sys
print("stub preview FAILED on purpose", file=sys.stderr)
sys.exit(4)
"""


def _write_stub(path, body):
    path.write_text(body)
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


class FakeFc:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def request(self, op, params=None):
        self.calls.append((op, params))
        if self.fail:
            raise RuntimeError("save_copy kaboom")
        return {"file": (params or {}).get("path")}


class FakeProject:
    def __init__(self, fail_save=False):
        self.fc = FakeFc(fail=fail_save)
        self.doc = "Doc"


def _patch_stubs(monkeypatch, tmp_path, appimage_body, optics_body):
    appimage = _write_stub(tmp_path / "fake_appimage.py", appimage_body)
    optics = _write_stub(tmp_path / "fake_optics.py", optics_body)
    monkeypatch.setattr(raypreview, "default_freecad_appimage",
                        lambda: appimage)
    monkeypatch.setattr(raypreview, "default_optics_python",
                        lambda: optics)


def test_full_chain_success(qtbot, tmp_path, monkeypatch):
    _patch_stubs(monkeypatch, tmp_path, STUB_APPIMAGE_OK, STUB_OPTICS_OK)
    controller = RayPreviewController()
    project = FakeProject()
    workspace = tmp_path / "workspace"

    with qtbot.waitSignal(controller.finished, timeout=10000) as blocker:
        started = controller.start(project, workspace, pattern="fan:n=5")
        assert started

    rays_path = blocker.args[0]
    assert rays_path == str(workspace / "preview" / "rays.vtp")
    assert os.path.exists(rays_path)
    assert project.fc.calls == [
        ("save_copy",
         {"doc": "Doc", "path": str(workspace / "preview" / "model.FCStd")})]
    assert not controller.is_running()


def test_only_bodies_forwarded_to_preview_stage(qtbot, tmp_path, monkeypatch):
    # capture argv the "optics" stub was actually called with, by having it
    # dump its own sys.argv to a side file we can inspect afterward
    optics_body = """#!/usr/bin/env python3
import json
import sys

argv = sys.argv[2:]
d = {}
it = iter(argv)
for tok in it:
    if tok.startswith("--"):
        d[tok[2:]] = next(it)
with open(d["out"] + ".argv.json", "w") as fh:
    json.dump(sys.argv, fh)
with open(d["out"], "w") as fh:
    fh.write("<VTKFile/>")
sys.exit(0)
"""
    _patch_stubs(monkeypatch, tmp_path, STUB_APPIMAGE_OK, optics_body)
    controller = RayPreviewController()
    project = FakeProject()
    workspace = tmp_path / "workspace"

    with qtbot.waitSignal(controller.finished, timeout=10000):
        controller.start(project, workspace, pattern="rings:dr=1:nper=4",
                         only_bodies=["Lens1", "Lens2"])

    # the stub dumps its own argv next to the --out path it was given
    dump_path = str(workspace / "preview" / "rays.vtp") + ".argv.json"
    dumped = json.loads(open(dump_path).read())
    assert "--only-bodies" in dumped
    assert dumped[dumped.index("--only-bodies") + 1] == "Lens1,Lens2"
    assert "--pattern" in dumped
    assert dumped[dumped.index("--pattern") + 1] == "rings:dr=1:nper=4"


def test_save_copy_failure_reports_failed_without_launching_subprocess(
        qtbot, tmp_path, monkeypatch):
    _patch_stubs(monkeypatch, tmp_path, STUB_APPIMAGE_OK, STUB_OPTICS_OK)
    controller = RayPreviewController()
    project = FakeProject(fail_save=True)
    workspace = tmp_path / "workspace"

    with qtbot.waitSignal(controller.failed, timeout=5000) as blocker:
        started = controller.start(project, workspace)
        assert not started

    assert "save_copy failed" in blocker.args[0]
    assert not controller.is_running()


def test_extract_failure_propagates_and_stops_the_chain(
        qtbot, tmp_path, monkeypatch):
    _patch_stubs(monkeypatch, tmp_path, STUB_APPIMAGE_FAIL, STUB_OPTICS_OK)
    controller = RayPreviewController()
    project = FakeProject()
    workspace = tmp_path / "workspace"

    with qtbot.waitSignal(controller.failed, timeout=10000) as blocker:
        started = controller.start(project, workspace)
        assert started

    assert "geometry extract failed" in blocker.args[0]
    assert not (workspace / "preview" / "rays.vtp").exists()
    assert not controller.is_running()


def test_preview_stage_failure_propagates(qtbot, tmp_path, monkeypatch):
    _patch_stubs(monkeypatch, tmp_path, STUB_APPIMAGE_OK, STUB_OPTICS_FAIL)
    controller = RayPreviewController()
    project = FakeProject()
    workspace = tmp_path / "workspace"

    with qtbot.waitSignal(controller.failed, timeout=10000) as blocker:
        started = controller.start(project, workspace)
        assert started

    assert "preview_rays.py failed" in blocker.args[0]


def test_cancel_during_extract_stops_process_and_suppresses_signals(
        qtbot, tmp_path, monkeypatch):
    _patch_stubs(monkeypatch, tmp_path, STUB_APPIMAGE_SLEEP, STUB_OPTICS_OK)
    controller = RayPreviewController()
    project = FakeProject()
    workspace = tmp_path / "workspace"

    seen = {"finished": [], "failed": []}
    controller.finished.connect(lambda p: seen["finished"].append(p))
    controller.failed.connect(lambda m: seen["failed"].append(m))

    started = controller.start(project, workspace)
    assert started
    qtbot.waitUntil(lambda: controller.is_running(), timeout=5000)

    controller.cancel()
    assert not controller.is_running()

    # give any stray queued signal a chance to fire, then confirm silence
    qtbot.wait(500)
    assert seen == {"finished": [], "failed": []}


def test_start_refused_while_a_preview_is_already_running(
        qtbot, tmp_path, monkeypatch):
    _patch_stubs(monkeypatch, tmp_path, STUB_APPIMAGE_SLEEP, STUB_OPTICS_OK)
    controller = RayPreviewController()
    project = FakeProject()
    workspace = tmp_path / "workspace"

    started = controller.start(project, workspace)
    assert started
    qtbot.waitUntil(lambda: controller.is_running(), timeout=5000)

    assert controller.start(project, workspace) is False

    controller.cancel()
