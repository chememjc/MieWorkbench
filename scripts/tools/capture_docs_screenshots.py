#!/usr/bin/env python
"""capture_docs_screenshots.py -- render docs/guide/img/*.png by driving
the real MainWindow offscreen.

Interpreter: env/bin/python (GUI venv), with QT_QPA_PLATFORM=offscreen set
in the environment BEFORE this process starts (the module sets it as a
defensive default too, but Qt reads the platform plugin at QApplication
construction time, so exporting it is the reliable way):

    QT_QPA_PLATFORM=offscreen env/bin/python \
        scripts/tools/capture_docs_screenshots.py

Drives the SAME MainWindow class the app and the test suite use (see
mieworkbench/tests/test_mainwindow*.py for the construction/teardown
pattern this mirrors: QApplication -> MainWindow() -> open_model(...) ->
... -> window.shutdown_resources() + window.project.shutdown()). Never
triggers a modal: every pane exposes a dialog-free path for the state a
screenshot needs (CLAUDE.md's "never show an unguarded modal in a pane
code path" rule), and this script only ever calls .show()/non-modal
setters, never .exec().

SHOT LIST: a declarative list of Shot records (name, scene, setup
callable -> the widget to grab, deferred flag + reason). This list IS the
"machine-readable manifest of needed shots" the task brief asks for --
run with --manifest to dump it as JSON without capturing anything.

Two fixture prerequisites, both derived from basemodels/ (never demos/ --
that tree is being rewritten by another round in parallel):
  * SCENE = basemodels/example-lenspos0.FCStd -- a real multi-element
    scene (2 sources, 1 lens with a coating, 3 detectors) good for most
    static panes.
  * FIXTURE_CASE = results/example-lenspos0/quick-docsshot -- a finished
    quick-preset run (--export-rays --smoke) of that scene, used for the
    Results/Animation shots. Regenerate it with:
        python3 scripts/run_pipeline.py --models basemodels/example-lenspos0.FCStd \
            --preset quick --export-rays --smoke --tag docsshot
    (results/ is gitignored; shots needing it degrade to deferred if it's
    absent instead of running the pipeline inline -- this tool captures
    screenshots, it does not orchestrate simulations).

Qt offscreen grab caveat: VtkSceneView does real OpenGL work only in
Initialize()/Render(), which is safe to construct but may rasterize to a
solid/black frame under the "offscreen" platform plugin (no real GPU
context). Each grabbed frame is checked for near-uniform (mostly-black)
content (`looks_blank`); a shot that comes back blank is recorded
"black-vtk" in the manifest and its PNG is NOT written (per the task
brief: don't fight VTK offscreen rendering -- note it and move on). The
committed demos/gallery/*.png (ParaView-rendered) are the fallback source
of real viewport imagery for anything that needs one.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QToolBar  # noqa: E402

from mieworkbench.mainwindow import MainWindow  # noqa: E402

IMG_DIR = os.path.join(REPO, "docs", "guide", "img")
SCENE_ROOT_DEFAULT = os.path.join(REPO, "basemodels")
SCENE = "example-lenspos0.FCStd"
FIXTURE_CASE = os.path.join(REPO, "results", "example-lenspos0",
                            "quick-docsshot")
WINDOW_SIZE = QSize(1600, 1000)

# a body known to carry a whole-body coating on SCENE (see the module
# docstring's "Body Lens {'coating': 'MgF2', ...}" fcclient probe) -- used
# by the element-editor shot to show a populated Active Properties row
COATED_BODY = "Body"


# ---------------------------------------------------------------------------
# offscreen-blank detection
# ---------------------------------------------------------------------------
def looks_blank(pixmap, sample=2000):
    """True if `pixmap` is (close to) a single flat color -- the offscreen-
    VTK-grab failure mode. Samples up to `sample` pixels on a grid rather
    than every pixel (cheap, plenty for a uniformity check)."""
    img = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB32)
    w, h = img.width(), img.height()
    if w == 0 or h == 0:
        return True
    step = max(1, int((w * h / max(sample, 1)) ** 0.5))
    seen = set()
    for y in range(0, h, step):
        for x in range(0, w, step):
            seen.add(img.pixel(x, y))
            if len(seen) > 1:
                return False
    return True


# ---------------------------------------------------------------------------
# per-shot setup callables: window -> QWidget to grab (or None to skip)
# ---------------------------------------------------------------------------
def _select(window, label_or_name):
    """Select a body by internal name (falls back to a label match against
    the open project's structure) through the same SelectionModel API a
    real pick uses."""
    name = label_or_name
    if window.project.structure:
        names = {b["name"] for b in window.project.structure["bodies"]}
        if label_or_name not in names:
            for b in window.project.structure["bodies"]:
                if b.get("label") == label_or_name:
                    name = b["name"]
                    break
    window.selection.select(name, ())
    QApplication.processEvents()


def shot_viewport_3d(window):
    window.central_tabs.setCurrentWidget(window.scene3d)
    window.scene3d.view.fit_camera()
    QApplication.processEvents()
    # capture the whole pane (toolbar + view) but blank-check only the
    # inner VTK widget -- the toolbar's own chrome is never blank, which
    # would otherwise mask a genuinely blank (offscreen-GL) render below it
    return window.scene3d, window.scene3d.view


def shot_element_editor(window):
    _select(window, COATED_BODY)
    window.element_editor_dock.raise_()
    return window.element_editor


def shot_train_editor(window):
    window.train_editor_dock.raise_()
    return window.train_editor


def shot_variables(window):
    if window.variables_dock is None:
        return None
    window.variables_dock.raise_()
    return window.variables_pane


def shot_library_browser(window):
    window.library_dock.raise_()
    return window.library


def shot_property_library_editor(window):
    window._open_prop_editor()
    QApplication.processEvents()
    return window._prop_editor_window


def shot_run_and_validate(window):
    window.config_matrix.resize(900, 700)
    return window.config_matrix


def shot_console_and_problems(window):
    window.console_dock.raise_()
    for line in (
        "[extract] example-lenspos0  bodies=7 sources=2 detectors=3"
        " overlaps=0 warnings=0",
        "[trace] seed 42 chunk [0,100000) of 100000 [C engine]",
        "[trace] done -- 6238670 ray interactions in 1.992 s"
        " (3.13e+06 rays/s), closure err max 2.16e-07",
        "[post] wrote images/spectra/plots + report.json",
    ):
        window.console.append_line(line)
    QApplication.processEvents()
    return window.console


def _load_fixture_case(window):
    if not os.path.isdir(FIXTURE_CASE):
        return False
    window.results.load_case(FIXTURE_CASE)
    QApplication.processEvents()
    return True


def shot_results_power(window):
    if not _load_fixture_case(window):
        return None
    window.central_tabs.setCurrentWidget(window.results)
    window.results.tabs.setCurrentIndex(1)   # Power
    return window.results


def shot_results_analysis(window):
    if not _load_fixture_case(window):
        return None
    window.central_tabs.setCurrentWidget(window.results)
    window.results.tabs.setCurrentIndex(2)   # Analysis
    return window.results


def shot_animation(window):
    rays_vtp = os.path.join(FIXTURE_CASE, "viz", "rays.vtp")
    if not os.path.isfile(rays_vtp):
        return None
    window.scene3d.load_rays_vtp(rays_vtp)
    window.anim_enable_action.setChecked(True)
    window.anim_controller.step()
    QApplication.processEvents()
    return window.findChild(QToolBar, "animation_toolbar")


# ---------------------------------------------------------------------------
# the shot list / manifest
# ---------------------------------------------------------------------------
class Shot:
    def __init__(self, name, setup, needs_scene=True, deferred=False,
                reason=""):
        self.name = name
        self.setup = setup
        self.needs_scene = needs_scene
        self.deferred = deferred
        self.reason = reason


SHOT_LIST = [
    Shot("viewport-3d-1", shot_viewport_3d),
    Shot("element-editor-1", shot_element_editor),
    Shot("train-editor-1", shot_train_editor),
    Shot("variables-1", shot_variables),
    Shot("library-browser-1", shot_library_browser),
    Shot("property-library-editor-1", shot_property_library_editor),
    Shot("run-and-validate-1", shot_run_and_validate),
    Shot("console-and-problems-1", shot_console_and_problems),
    Shot("results-1", shot_results_power),
    Shot("results-2", shot_results_analysis),
    Shot("animation-1", shot_animation),
    Shot("optimize-1", None, deferred=True,
        reason="needs a completed optimize.py run for a non-empty "
               "convergence plot -- capture against a demo run in "
               "Phase B"),
    Shot("tolerance-1", None, deferred=True,
        reason="needs a completed tolerance.py run for non-empty "
               "sensitivity/yield plots -- capture against a demo run "
               "in Phase B"),
]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(only=None, scene_root=SCENE_ROOT_DEFAULT, out_dir=IMG_DIR):
    os.makedirs(out_dir, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(WINDOW_SIZE)
    window.show()
    QApplication.processEvents()

    scene_path = os.path.join(scene_root, SCENE)
    scene_loaded = False
    try:
        if os.path.isfile(scene_path):
            window.open_model(scene_path)
            scene_loaded = True
            QApplication.processEvents()

        results = []
        for shot in SHOT_LIST:
            if only and shot.name not in only:
                continue
            entry = {"name": shot.name}
            if shot.deferred or shot.setup is None:
                entry["status"] = "deferred"
                entry["reason"] = shot.reason or "no setup callable"
                results.append(entry)
                continue
            if shot.needs_scene and not scene_loaded:
                entry["status"] = "deferred"
                entry["reason"] = "scene %s not found under %s" % (
                    SCENE, scene_root)
                results.append(entry)
                continue
            try:
                result = shot.setup(window)
            except Exception as exc:                     # pragma: no cover
                entry["status"] = "error"
                entry["reason"] = "%s: %s" % (type(exc).__name__, exc)
                results.append(entry)
                continue
            widget, check_widget = (result if isinstance(result, tuple)
                                    else (result, result))
            if widget is None:
                entry["status"] = "deferred"
                entry["reason"] = "setup returned no widget (missing " \
                                  "fixture case or dock)"
                results.append(entry)
                continue
            QApplication.processEvents()
            pixmap = widget.grab()
            if pixmap.isNull() or pixmap.width() == 0:
                entry["status"] = "error"
                entry["reason"] = "grab() returned an empty pixmap"
                results.append(entry)
                continue
            if looks_blank(check_widget.grab()):
                entry["status"] = "black-vtk"
                entry["reason"] = "grab() rendered a single flat color " \
                                  "(offscreen VTK/GL limitation)"
                results.append(entry)
                continue
            out_path = os.path.join(out_dir, shot.name + ".png")
            pixmap.save(out_path, "PNG")
            entry["status"] = "captured"
            entry["path"] = os.path.relpath(out_path, REPO)
            results.append(entry)
        return results
    finally:
        window.shutdown_resources()
        try:
            window.project.shutdown()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", default=None,
                    help="capture just this shot name (repeatable)")
    ap.add_argument("--scene-root", default=SCENE_ROOT_DEFAULT,
                    help="directory SCENE is resolved against (default: "
                         "basemodels/; Phase B points this at demos/)")
    ap.add_argument("--out-dir", default=IMG_DIR,
                    help="where PNGs are written (default: docs/guide/img)")
    ap.add_argument("--manifest", action="store_true",
                    help="print the shot list as JSON and exit without "
                         "capturing anything")
    args = ap.parse_args()

    if args.manifest:
        print(json.dumps(
            [{"name": s.name, "deferred": s.deferred, "reason": s.reason}
             for s in SHOT_LIST], indent=2))
        return 0

    only = set(args.only) if args.only else None
    results = run(only=only, scene_root=args.scene_root,
                 out_dir=args.out_dir)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(results, fh, indent=2)
    for entry in results:
        print("%-28s %-10s %s" % (entry["name"], entry["status"],
                                  entry.get("path", entry.get("reason", ""))))
    print("manifest written to %s" % os.path.relpath(manifest_path, REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
