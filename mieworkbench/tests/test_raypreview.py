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
from pathlib import Path

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core import raypreview                    # noqa: E402
from mieworkbench.core.raypreview import RayPreviewController  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


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


# =============================================================================
# P4b preview unification: the sequential fast path (core/sequential_
# preview.py), driven through the SAME stub-appimage harness above but with
# the stub extract COPYING A REAL committed geometry/<stem> cache (rather
# than writing the placeholder {"stub": True} JSON the tests above use) so
# the fast path has a genuine model.json to bridge. The "optics" stub is
# STUB_OPTICS_FAIL in the bridgeable case specifically to PROVE the fallback
# subprocess is never launched (it would fail loudly and fail the test if it
# ran) -- the task's "without touching the FreeCAD extract when geometry is
# fresh" requirement, restated precisely: after a single successful extract,
# the trace stage never re-invokes _start_extract, and (in the bridgeable
# case) never needs the fallback subprocess at all.
# =============================================================================
pytest.importorskip("optiland")

STUB_APPIMAGE_COPY = """#!/usr/bin/env python3
import os
import shutil
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
dst = os.path.join(outdir, stem)
shutil.copytree(%r, dst)
# count invocations (a side-effect file next to outdir) so the test can
# assert the extract stage never re-runs
counter = os.path.join(os.path.dirname(outdir), "extract_calls.txt")
with open(counter, "a") as fh:
    fh.write("1\\n")
print("stub extract ok (copied real geometry %s)")
sys.exit(0)
""" % (str(REPO / "geometry" / "lens_dcx"), "lens_dcx")

STUB_APPIMAGE_COPY_CRYSTAL = STUB_APPIMAGE_COPY.replace(
    repr(str(REPO / "geometry" / "lens_dcx")),
    repr(str(REPO / "geometry" / "ktp_walkoff")))


def _skip_if_no_geometry(stem):
    if not (REPO / "geometry" / stem / "model.json").exists():
        pytest.skip("geometry/%s cache absent" % stem)


def test_sequential_fast_path_used_for_bridgeable_scene_no_fallback_subprocess(
        qtbot, tmp_path, monkeypatch):
    """A bridgeable scene (lens_dcx): the sequential fast path succeeds
    in-process, finished() reports 'sequential (exact)', and the fallback
    preview_rays.py subprocess is NEVER launched -- STUB_OPTICS_FAIL would
    fail the whole chain if it ran, so a passing test proves it didn't."""
    _skip_if_no_geometry("lens_dcx")
    _patch_stubs(monkeypatch, tmp_path, STUB_APPIMAGE_COPY, STUB_OPTICS_FAIL)
    controller = RayPreviewController()
    project = FakeProject()
    workspace = tmp_path / "workspace"

    with qtbot.waitSignal(controller.finished, timeout=15000) as blocker:
        started = controller.start(project, workspace, pattern="fan:n=5")
        assert started

    rays_path, engine = blocker.args
    assert engine == "sequential (exact)"
    assert os.path.exists(rays_path)

    counter = workspace / "preview" / "extract_calls.txt"
    assert counter.read_text().count("1") == 1, \
        "the FreeCAD extract stage ran more than once for one preview cycle"


def test_fallback_engages_for_unbridgeable_crystal_scene(
        qtbot, tmp_path, monkeypatch):
    """An unbridgeable scene (ktp_walkoff, a birefringent crystal train):
    the sequential fast path declines (BridgeUnsupported), the chain falls
    back to the original preview_rays.py subprocess exactly once, and
    finished() reports 'engine fan'."""
    _skip_if_no_geometry("ktp_walkoff")
    _patch_stubs(monkeypatch, tmp_path, STUB_APPIMAGE_COPY_CRYSTAL,
                STUB_OPTICS_OK)
    controller = RayPreviewController()
    project = FakeProject()
    workspace = tmp_path / "workspace"

    with qtbot.waitSignal(controller.finished, timeout=15000) as blocker:
        started = controller.start(project, workspace, pattern="fan:n=5")
        assert started

    rays_path, engine = blocker.args
    assert engine == "engine fan"
    assert os.path.exists(rays_path)

    counter = workspace / "preview" / "extract_calls.txt"
    assert counter.read_text().count("1") == 1


def test_sequential_build_can_be_disabled_via_monkeypatch(
        qtbot, tmp_path, monkeypatch):
    """default_sequential_build() is the ONE seam a test (or a future
    settings toggle) needs to force the general chain even for a
    bridgeable scene."""
    _skip_if_no_geometry("lens_dcx")
    _patch_stubs(monkeypatch, tmp_path, STUB_APPIMAGE_COPY, STUB_OPTICS_OK)
    monkeypatch.setattr(raypreview, "default_sequential_build", lambda: None)
    controller = RayPreviewController()
    project = FakeProject()
    workspace = tmp_path / "workspace"

    with qtbot.waitSignal(controller.finished, timeout=15000) as blocker:
        started = controller.start(project, workspace, pattern="fan:n=5")
        assert started

    _rays_path, engine = blocker.args
    assert engine == "engine fan"


# =============================================================================
# user-selectable preview engine (engine="sequential"|"full" on start()):
# same stub harness as the P4b block above, reusing the lens_dcx bridgeable
# geometry cache and the "STUB_OPTICS_FAIL proves the fallback didn't run"
# trick where applicable.
# =============================================================================
def test_engine_full_forces_fallback_subprocess_and_skips_the_bridge(
        qtbot, tmp_path, monkeypatch):
    """engine="full" must skip the sequential bridge entirely -- even for
    a scene that WOULD bridge (lens_dcx) -- and always take the full
    Monte-Carlo preview_rays.py subprocess, reporting the honest 'full
    trace' label."""
    _skip_if_no_geometry("lens_dcx")
    _patch_stubs(monkeypatch, tmp_path, STUB_APPIMAGE_COPY, STUB_OPTICS_OK)

    def _boom():
        raise AssertionError("sequential bridge must not be consulted "
                             "when engine='full'")
    monkeypatch.setattr(raypreview, "default_sequential_build", _boom)
    controller = RayPreviewController()
    project = FakeProject()
    workspace = tmp_path / "workspace"

    with qtbot.waitSignal(controller.finished, timeout=15000) as blocker:
        started = controller.start(project, workspace, pattern="fan:n=5",
                                   engine="full")
        assert started

    rays_path, engine = blocker.args
    assert engine == raypreview.ENGINE_FULL
    assert os.path.exists(rays_path)


def test_explicit_engine_sequential_matches_default_auto_behavior(
        qtbot, tmp_path, monkeypatch):
    """engine="sequential" passed explicitly is just the current default
    auto behavior spelled out: the sequential fast path is used on a
    bridgeable scene, the fallback subprocess never launches (STUB_OPTICS_
    FAIL would fail the test if it ran), and the label stays 'sequential
    (exact)'."""
    _skip_if_no_geometry("lens_dcx")
    _patch_stubs(monkeypatch, tmp_path, STUB_APPIMAGE_COPY, STUB_OPTICS_FAIL)
    controller = RayPreviewController()
    project = FakeProject()
    workspace = tmp_path / "workspace"

    with qtbot.waitSignal(controller.finished, timeout=15000) as blocker:
        started = controller.start(project, workspace, pattern="fan:n=5",
                                   engine="sequential")
        assert started

    rays_path, engine = blocker.args
    assert engine == "sequential (exact)"
    assert os.path.exists(rays_path)
