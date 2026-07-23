#!/usr/bin/env python
"""gui_verify.py -- interactive-GUI verification harness, run under a REAL
X server (xvfb-run) so VTK gets a real GL context and screenshots show
actual rendered frames, not the offscreen-platform blank-frame failure
mode scripts/tools/capture_docs_screenshots.py works around.

    xvfb-run -a env/bin/python scripts/tools/gui_verify.py

Do NOT set QT_QPA_PLATFORM=offscreen (this script actively strips it if
inherited) -- under xvfb the default "xcb" platform plugin gets a real,
if software-rendered (llvmpipe), GL context via GLX. Run without xvfb-run
(no DISPLAY) and the script prints the invocation hint and exits nonzero
rather than silently falling back to a blank-frame run.

Reuses the proven driving machinery from capture_docs_screenshots.py:
MainWindow construction/teardown, event-loop pumping, grab-and-save,
blank-frame detection (looks_blank) -- extended with a VTK-render-window
fallback (vtkWindowToImageFilter) for 3D-view screenshots that come back
blank from a plain widget.grab() (some window managers/compositors under
Xvfb still rasterize the QWidget backing store oddly even though the GL
context itself is real).

Organized as SCENARIOS: an ordered list of steps (ctx.step("name") marks
the current step; actions follow; ctx.check(cond, msg) asserts; ctx.shot()
screenshots). A scenario runs against a FRESH MainWindow (never reused
across scenarios, avoiding state bleed) under a single shared QApplication.
A failing ctx.check() raises StepFailure, caught by the scenario runner:
a "FAIL <scenario> @ <step> (<elapsed>s): <message>" line is printed and
the harness moves on to the NEXT SCENARIO (a failure never aborts the
whole run). Exit code is nonzero iff any scenario did not PASS.

CLI:
    --only NAME      run just this scenario (repeatable)
    --list           print scenario names and exit
    --out DIR        screenshot root (default var/gui_verify, gitignored)
"""
import argparse
import os
import shutil
import sys
import time
import traceback

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

DEMOS_DIR = os.path.join(REPO, "demos")
DEFAULT_OUT = os.path.join(REPO, "var", "gui_verify")

TELEPHOTO = os.path.join(DEMOS_DIR, "telephoto_zoom.MieWB")
CAMERA_TRIPLET = os.path.join(DEMOS_DIR, "camera_triplet.MieWB")
IRIS_DEMO = os.path.join(DEMOS_DIR, "bladed_iris_star.MieWB")

PRIMITIVES_DIR = os.path.join(REPO, "primitives")
CUVETTE_SQUARE = os.path.join(PRIMITIVES_DIR, "cuvette_square.FCStd")
SOURCE_IMAGE = os.path.join(PRIMITIVES_DIR, "source_image.FCStd")

WINDOW_W, WINDOW_H = 1600, 1000


# ---------------------------------------------------------------------------
# DISPLAY / platform guard -- must run BEFORE importing PySide6/mieworkbench
# ---------------------------------------------------------------------------
def _require_real_display():
    if not os.environ.get("DISPLAY"):
        print(
            "gui_verify.py needs a real X display for VTK's GL context "
            "(the offscreen Qt platform rasterizes VTK to a blank frame).\n"
            "Run it under Xvfb:\n\n"
            "    xvfb-run -a env/bin/python scripts/tools/gui_verify.py\n",
            file=sys.stderr)
        raise SystemExit(2)
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        print("gui_verify.py: QT_QPA_PLATFORM=offscreen was set in the "
              "environment; removing it so Qt uses the real xcb platform "
              "under this X display.", file=sys.stderr)
        del os.environ["QT_QPA_PLATFORM"]


_require_real_display()

from PySide6.QtCore import Qt, QSettings  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QLineEdit, QScrollArea, QSpinBox,
    QToolBar,
)
from PySide6.QtGui import QColor, QImage  # noqa: E402

import common  # noqa: E402  (stdlib-only shared contract hub)
from mieworkbench.mainwindow import MainWindow  # noqa: E402
from mieworkbench.widgets.vtkview import _ABSORBER_STYLE  # noqa: E402
from mieworkbench.panes.tolerance_pane import (  # noqa: E402
    HAVE_QTCHARTS as TOL_HAVE_QTCHARTS,
)


# ---------------------------------------------------------------------------
# blank-frame detection (verbatim from capture_docs_screenshots.py) +
# a VTK render-window fallback for 3D-view steps
# ---------------------------------------------------------------------------
def looks_blank(pixmap, sample=2000):
    """True if `pixmap` is (close to) a single flat color."""
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


def _capture_vtk_render_window(vtk_view, path):
    """Render `vtk_view`'s (a VtkSceneView) actual render window straight
    to a PNG via vtkWindowToImageFilter, bypassing the QWidget backing
    store entirely. Returns True on success."""
    try:
        from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
        from vtkmodules.vtkIOImage import vtkPNGWriter
    except Exception:
        return False
    try:
        render_window = vtk_view.interactor.GetRenderWindow()
        render_window.Render()
        w2i = vtkWindowToImageFilter()
        w2i.SetInput(render_window)
        w2i.ReadFrontBufferOff()
        w2i.Update()
        writer = vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# event-loop pumping helpers
# ---------------------------------------------------------------------------
def pump(seconds, interval_s=0.02):
    """Pump the Qt event loop for `seconds` of wall-clock time (lets
    QTimers -- the bead-animation playback clock -- actually tick under
    the real xcb platform, and lets QProcess signals for the ray-preview
    subprocess chain arrive)."""
    deadline = time.monotonic() + seconds
    while True:
        QApplication.processEvents()
        if time.monotonic() >= deadline:
            return
        time.sleep(interval_s)


def pump_until(predicate, timeout_s=30.0, interval_s=0.05):
    """Pump until `predicate()` is true or `timeout_s` elapses. Returns
    the final predicate() value (so a timeout is distinguishable from a
    predicate that flipped true right at the deadline only by the caller
    re-checking, which every caller here does via ctx.check)."""
    deadline = time.monotonic() + timeout_s
    while True:
        QApplication.processEvents()
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return predicate()
        time.sleep(interval_s)


# ---------------------------------------------------------------------------
# scenario context: steps, checks, screenshots
# ---------------------------------------------------------------------------
class StepFailure(Exception):
    pass


class ScenarioContext:
    def __init__(self, name, out_dir):
        self.name = name
        self.out_dir = out_dir
        self.step_name = "-"
        self.screenshots = []
        os.makedirs(out_dir, exist_ok=True)

    def step(self, name):
        self.step_name = name
        print("    [%s] %s" % (self.name, name))

    def check(self, cond, message):
        if not cond:
            raise StepFailure(message)

    def shot(self, widget, filename, vtk_view=None):
        """Screenshot `widget` (a full-window widget.grab()) to
        out_dir/filename. If `vtk_view` (a VtkSceneView) is given, its
        blankness is probed SEPARATELY from `widget` (a wrapping pane's
        grab() can look non-blank purely from its toolbar chrome while the
        embedded native VTK subwindow -- composited via its own GLX
        surface, not Qt's backing store -- comes back white/blank in the
        SAME grab()); when the vtk_view probe is blank, fall back to a
        direct render-window capture (vtkWindowToImageFilter) of
        vtk_view itself, which bypasses widget compositing entirely."""
        path = os.path.join(self.out_dir, filename)
        pixmap = widget.grab()
        self.check(not pixmap.isNull() and pixmap.width() > 0,
                  "grab() of %r returned an empty pixmap" % filename)
        if vtk_view is not None and looks_blank(vtk_view.grab()):
            if _capture_vtk_render_window(vtk_view, path):
                self.screenshots.append((path, "vtk-fallback"))
                return path
        pixmap.save(path, "PNG")
        blank = looks_blank(pixmap)
        self.screenshots.append((path, "blank" if blank else "ok"))
        return path


def _isolate_qsettings(out_root):
    """Point every QSettings(IniFormat, UserScope, ...) lookup -- which is
    how MainWindow's core/settings.Settings wraps QSettings("Curtis
    Analytical", "MieWorkbench") -- at a scratch dir under `out_root`,
    wiped fresh. MainWindow persists toggles like anim_enabled /
    show_face_indicators / anim_bead_opacity_mode; writing to the REAL
    per-user settings file would both pollute the developer's actual GUI
    config and make scenario PASS/FAIL depend on what a PRIOR run (or a
    prior scenario in this same run, e.g. toolbar-contrast checking
    anim_enable_action) last left on disk. Called once per scenario (a
    fresh MainWindow deserves a fresh settings slate too, not just a
    fresh Python object)."""
    settings_dir = os.path.join(out_root, ".qsettings")
    shutil.rmtree(settings_dir, ignore_errors=True)
    os.makedirs(settings_dir, exist_ok=True)
    # NativeFormat, NOT IniFormat: QSettings(org, app) with no explicit
    # format argument (what core/settings.Settings uses) resolves to
    # NativeFormat, and setPath keys its redirect table by the exact
    # format passed in -- pointing IniFormat elsewhere is a silent no-op
    # on the format actually in use (verified: it kept writing to the
    # real ~/.config/CurtisAnalytical/MieWorkbench.conf). NativeFormat IS
    # ini-on-disk on Linux, just a different table entry.
    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope,
                      settings_dir)


def new_window(out_root):
    _isolate_qsettings(out_root)
    window = MainWindow()
    window.resize(WINDOW_W, WINDOW_H)
    window.show()
    QApplication.processEvents()
    return window


def _write_dummy_png(path, color):
    """A tiny non-trivial (non-single-color-page) PNG fixture: a flat
    color swatch is plenty to prove a gallery's thumbnail loader actually
    picked the file up (looks_blank() is checked against the whole PANE
    grab, which always has table/toolbar chrome around the thumbnails --
    never against the swatch alone)."""
    img = QImage(64, 64, QImage.Format.Format_RGB32)
    img.fill(color)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, "PNG")


def _tab_index_by_label(tabs, label):
    for i in range(tabs.count()):
        if tabs.tabText(i) == label:
            return i
    return -1


def teardown_window(window):
    try:
        window.shutdown_resources()
    except Exception:
        pass
    try:
        window.project.shutdown()
    except Exception:
        pass
    try:
        window.close()
    except Exception:
        pass
    QApplication.processEvents()


# ===========================================================================
# Scenario 1: anim-autopreview (THE acceptance scenario)
# ===========================================================================
def scenario_anim_autopreview(ctx, window):
    ctx.step("open demos/telephoto_zoom.MieWB")
    ctx.check(os.path.isfile(TELEPHOTO), "fixture missing: %s" % TELEPHOTO)
    window.open_model(TELEPHOTO)
    pump(2.0)
    ctx.check(window.project.is_open(), "project failed to open telephoto_zoom")

    ctx.step("baseline screenshot")
    ctx.shot(window.scene3d, "01-baseline.png", vtk_view=window.scene3d.view)

    ctx.step("toggle anim_enable_action ON")
    fail_box = {}
    window.raypreview.failed.connect(lambda msg: fail_box.setdefault("msg", msg))
    ctx.check(not window.anim_enable_action.isChecked(),
              "anim_enable_action already checked before the test")
    window.anim_enable_action.setChecked(True)
    QApplication.processEvents()
    running = window.raypreview.is_running()
    status = window.statusBar().currentMessage()
    ctx.check(running or "preview" in status.lower(),
              "ray preview did not start: is_running()=%s status=%r"
              % (running, status))

    ctx.step("wait for preview completion (anim_controller.has_segments())")
    def _done():
        return bool(fail_box) or window.anim_controller.has_segments()
    pump_until(_done, timeout_s=120.0)
    ctx.check(not fail_box,
              "ray preview failed: %s" % fail_box.get("msg"))
    ctx.check(window.anim_controller.has_segments(),
              "anim_controller.has_segments() still False after 120s "
              "(preview never finished, or produced an overlay with no "
              "opl/timing data)")

    ctx.step("assert no stale-timing-data warning")
    msg = window.statusBar().currentMessage()
    ctx.check("predate timing data" not in msg,
              "bead-animation-unavailable warning is showing: %r" % msg)

    ctx.step("screenshot beads at t=0")
    ctx.shot(window.scene3d, "02-beads-t0.png", vtk_view=window.scene3d.view)

    ctx.step("set bead-opacity combo to 'By power'")
    idx = window.anim_opacity_combo.findData("power")
    ctx.check(idx >= 0, "anim_opacity_combo has no 'power' entry")
    window.anim_opacity_combo.setCurrentIndex(idx)
    QApplication.processEvents()

    ctx.step("step animation 5 frames")
    for _ in range(5):
        window.anim_step_action.trigger()
        QApplication.processEvents()

    ctx.step("screenshot stepped frames (power opacity)")
    ctx.shot(window.scene3d, "03-stepped-power-opacity.png",
             vtk_view=window.scene3d.view)

    ctx.step("assert BeadLayer received a 4-comp RGBA array")
    ctx.check(window.anim_controller.power_available(),
              "loaded preview rays carry no per-segment power (rel_power); "
              "'By power' opacity mode falls back to opaque -- statusBar: %r"
              % window.statusBar().currentMessage())
    ncomp = window.scene3d.view.beads._rgb.GetNumberOfComponents()
    ctx.check(ncomp == 4,
              "BeadLayer scalar array has %d component(s), expected 4 "
              "(RGBA) once 'By power' opacity is active" % ncomp)

    ctx.step("play ~1s")
    window.anim_play_action.trigger()
    pump(1.0)

    ctx.step("pause")
    window.anim_pause_action.trigger()
    QApplication.processEvents()

    ctx.step("screenshot after play/pause")
    ctx.shot(window.scene3d, "04-play-pause.png", vtk_view=window.scene3d.view)


# ===========================================================================
# Scenario 2: selection (whole-element -> sub-select -> clear)
# ===========================================================================
def scenario_selection(ctx, window):
    ctx.step("open demos/telephoto_zoom.MieWB")
    ctx.check(os.path.isfile(TELEPHOTO), "fixture missing: %s" % TELEPHOTO)
    window.open_model(TELEPHOTO)
    pump(2.0)
    ctx.check(window.project.is_open(), "project failed to open telephoto_zoom")

    ctx.step("locate FrontGroup member bodies")
    body_names = [b["name"] for b in window.project.structure["bodies"]]
    front_members = [b for b in body_names
                     if window.project.element_group(b) == "FrontGroup"]
    ctx.check(len(front_members) >= 2,
              "expected >=2 member bodies for element 'FrontGroup' "
              "(a multi-body achromat), found %r" % front_members)

    ctx.step("selection.select(member, origin='scene3d') through the "
            "real dispatcher")
    window.selection.select(front_members[0], (), origin="scene3d")
    QApplication.processEvents()
    ctx.check(window.selection.element == "FrontGroup",
              "selection.element=%r, expected 'FrontGroup' (a bare-body "
              "scene3d pick should be expanded to the whole element)"
              % window.selection.element)
    ctx.check(set(window.selection.bodies) == set(front_members),
              "selection.bodies=%r, expected %r"
              % (window.selection.bodies, front_members))

    ctx.step("assert view._selection covers BOTH members' face ids")
    expected = set()
    for b in front_members:
        expected |= {f["id"] for f in
                    window.project.faces.get(b, {}).get("faces", [])}
    actual = window.scene3d.view._selection
    ctx.check(bool(expected), "computed an empty expected face-id set "
                              "(fixture/face lookup problem)")
    ctx.check(actual == expected,
              "view._selection (%d face ids) != expected member-union "
              "(%d face ids)" % (len(actual), len(expected)))

    ctx.step("screenshot whole-element selection")
    ctx.shot(window.scene3d, "01-element-selected.png",
             vtk_view=window.scene3d.view)

    ctx.step("sub-select via inspector member list (click first row)")
    member_list = window.inspector.member_list.list
    ctx.check(member_list.count() >= 1,
              "inspector member list is empty after an element selection")
    item = member_list.item(0)
    clicked_name = item.data(Qt.ItemDataRole.UserRole)
    window.inspector.member_list._on_item_clicked(item)
    QApplication.processEvents()

    ctx.step("assert single-body sub-selection")
    ctx.check(window.selection.element is None,
              "selection.element=%r, expected None after a member-list "
              "sub-select" % window.selection.element)
    ctx.check(window.selection.body == clicked_name,
              "selection.body=%r, expected the clicked member %r"
              % (window.selection.body, clicked_name))

    ctx.step("screenshot sub-selection")
    ctx.shot(window.scene3d, "02-sub-selected.png", vtk_view=window.scene3d.view)

    ctx.step("trigger clear_selection_action")
    window.clear_selection_action.trigger()
    QApplication.processEvents()

    ctx.step("assert selection empty + copy/delete actions disabled")
    ctx.check(window.selection.body is None and not window.selection.bodies,
              "selection not cleared: body=%r bodies=%r"
              % (window.selection.body, window.selection.bodies))
    ctx.check(not window.clear_selection_action.isEnabled(),
              "clear_selection_action still enabled after clearing")
    ctx.check(not window.copy_action.isEnabled(),
              "copy_action still enabled with an empty selection")
    ctx.check(not window.delete_action.isEnabled(),
              "delete_action still enabled with an empty selection")

    ctx.step("screenshot cleared selection")
    ctx.shot(window.scene3d, "03-cleared.png", vtk_view=window.scene3d.view)


# ===========================================================================
# Scenario 3: absorbing-stop
# ===========================================================================
def scenario_absorbing_stop(ctx, window):
    ctx.step("open demos/bladed_iris_star.MieWB")
    ctx.check(os.path.isfile(IRIS_DEMO), "fixture missing: %s" % IRIS_DEMO)
    window.open_model(IRIS_DEMO)
    pump(2.0)
    ctx.check(window.project.is_open(), "project failed to open bladed_iris_star")

    ctx.step("locate the Iris body")
    body_name = None
    bodies = window.project.structure["bodies"]
    names = {b["name"] for b in bodies}
    if "Iris" in names:
        body_name = "Iris"
    else:
        for b in bodies:
            if b.get("label") == "Iris":
                body_name = b["name"]
                break
    ctx.check(body_name is not None,
              "no body named/labeled 'Iris' found in bladed_iris_star "
              "(bodies: %r)" % sorted(names))

    ctx.step("assert the iris disc actor's base style is the dark "
            "absorber style")
    view = window.scene3d.view
    actors = view._body_actors.get(body_name) or []
    ctx.check(bool(actors), "no rendered actors for body %r" % body_name)
    styles = {view._actor_base_style.get(a) for a in actors}
    ctx.check(_ABSORBER_STYLE in styles,
              "Iris actor base style(s)=%r, expected to include the "
              "absorber style %r" % (styles, _ABSORBER_STYLE))

    ctx.step("screenshot showing the dark iris")
    ctx.shot(window.scene3d, "01-dark-iris.png", vtk_view=window.scene3d.view)


# ===========================================================================
# Scenario 4: preview-config
# ===========================================================================
def scenario_preview_config(ctx, window):
    ctx.step("open demos/telephoto_zoom.MieWB")
    ctx.check(os.path.isfile(TELEPHOTO), "fixture missing: %s" % TELEPHOTO)
    window.open_model(TELEPHOTO)
    pump(2.0)
    ctx.check(window.project.is_open(), "project failed to open telephoto_zoom")

    spec = "rings:dr=2:nper=8:nrings=3"

    ctx.step("project.set_preview_config(rings spec + sequential engine)")
    window.project.set_preview_config({"spec": spec,
                                       "engine": "sequential"})

    ctx.step("assert get_preview_config round-trip")
    cfg = window.project.get_preview_config()
    ctx.check(cfg is not None and cfg.get("spec") == spec
              and cfg.get("engine") == "sequential",
              "get_preview_config()=%r, expected spec=%r engine=sequential"
              % (cfg, spec))

    ctx.step("open the Preview Configuration dialog (launch=False)")
    dialog = window._open_preview_dialog(launch=False, exec_dialog=False)
    dialog.show()
    QApplication.processEvents()
    got = dialog.values()
    ctx.check(got["spec"] == spec and got["engine"] == "sequential",
              "dialog prefill=%r, expected spec=%r engine=sequential"
              % (got, spec))

    ctx.step("screenshot the dialog as opened")
    ctx.shot(dialog, "01-preview-dialog.png")

    ctx.step("drive engine=Full trace; extinction Off auto-flips to log")
    off_idx = dialog.dim_mode_combo.findData("off")
    dialog.dim_mode_combo.setCurrentIndex(off_idx)
    dialog.engine_combo.setCurrentIndex(
        dialog.engine_combo.findData("full"))
    QApplication.processEvents()
    ctx.check(dialog.dim_mode_combo.currentData() == "log",
              "dim mode=%r after switching to full trace, expected "
              "auto-log" % dialog.dim_mode_combo.currentData())

    ctx.step("advanced text row round-trips a bare integer")
    dialog._on_spec_text_edited("9")
    ctx.check(dialog.values()["spec"] == "fan:n=9",
              "advanced '9' gave spec=%r, expected fan:n=9"
              % dialog.values()["spec"])

    ctx.step("screenshot the dialog after the edits")
    ctx.shot(dialog, "02-preview-dialog-full-trace.png")
    dialog.reject()


# ===========================================================================
# Scenario 4b: full-trace-ghosts -- the engine=full preview shows Fresnel
# ghost reflections under log extinction
# ===========================================================================
def scenario_full_trace_ghosts(ctx, window):
    ctx.step("open demos/telephoto_zoom.MieWB")
    ctx.check(os.path.isfile(TELEPHOTO), "fixture missing: %s" % TELEPHOTO)
    window.open_model(TELEPHOTO)
    pump(2.0)
    ctx.check(window.project.is_open(), "project failed to open telephoto_zoom")

    ctx.step("persist engine=full into the project; log extinction 40 dB")
    window.project.set_preview_config({"spec": "fan:n=5",
                                       "engine": "full"})
    window._on_ray_dimming_mode("log")
    window._set_ray_dimming_range_db(40.0)

    ctx.step("launch the preview through _start_scene_preview (the "
             "resolved-config path every launcher shares)")
    box = {}
    window.raypreview.finished.connect(
        lambda path, label: box.setdefault("label", label))
    window.raypreview.failed.connect(
        lambda msg: box.setdefault("fail", msg))
    ctx.check(window._start_scene_preview(),
              "_start_scene_preview() did not launch")

    ctx.step("wait for completion; label must be 'full trace'")
    pump_until(lambda: box, timeout_s=180.0)
    ctx.check("fail" not in box, "preview failed: %s" % box.get("fail"))
    ctx.check(box.get("label") == "full trace",
              "engine label=%r, expected 'full trace'" % box.get("label"))

    ctx.step("overlay carries reflection children (rel_power well "
             "below the transmitted chain)")
    from vtkmodules.util.numpy_support import vtk_to_numpy
    polydata = window.scene3d.view._rays_polydata
    ctx.check(polydata is not None, "no rays polydata loaded")
    rel_arr = polydata.GetCellData().GetArray("rel_power")
    ctx.check(rel_arr is not None, "overlay lacks rel_power")
    rel = vtk_to_numpy(rel_arr)
    n_weak = int((rel < 0.5).sum())
    ctx.check(n_weak > 0,
              "no segment with rel_power < 0.5 among %d -- the full "
              "trace produced no visible reflection branches" % rel.size)

    ctx.step("log extinction composed an RGBA (alpha) array")
    ctx.check(polydata.GetCellData().GetArray("rgba_dim") is not None,
              "rgba_dim missing -- log dimming did not compose")

    ctx.step("screenshot the 3D view with ghost branches")
    ctx.shot(window.scene3d, "01-full-trace-ghosts.png",
             vtk_view=window.scene3d.view)


# ===========================================================================
# Scenario 5: plot-inspect (no scene needed -- synthetic run events)
# ===========================================================================
def scenario_plot_inspect(ctx, window):
    ctx.step("feed OptimizePane a synthetic 6-eval run")
    pane = window.optimize_pane
    pane.on_started()
    best_merit, best_params = None, None
    for i in range(1, 7):
        merit = 10.0 / i          # monotonically improving series
        params = {"z": 90.0 + i, "efl": 200.0 - i}
        if best_merit is None or merit < best_merit:
            best_merit, best_params = merit, params
        pane.on_progress({
            "stage": "optimize", "eval": i, "budget": 6, "merit": merit,
            "best": best_merit, "params": params, "best_params": best_params,
        })
    pane.on_finished(0)
    QApplication.processEvents()

    ctx.step("assert Apply-optimum button enabled")
    ctx.check(pane.apply_btn.isEnabled(),
              "apply_btn not enabled after a clean 6-eval run with a real "
              "best (best_merit=%r best_params=%r)"
              % (pane._best_merit, pane._best_params))

    ctx.step("open the optimize Show-data dialog programmatically")
    pane.plot._show_data_dialog()
    QApplication.processEvents()
    dialog = pane.plot._data_dialog
    ctx.check(dialog is not None, "plot._show_data_dialog() left no dialog")

    ctx.step("assert table contents/rank column")
    ctx.check(dialog.table.rowCount() == 6,
              "optimize data table has %d rows, expected 6"
              % dialog.table.rowCount())
    last_header = dialog.table.horizontalHeaderItem(
        dialog.table.columnCount() - 1)
    ctx.check(last_header is not None and last_header.text() == "rank",
              "last column header=%r, expected 'rank'"
              % (last_header.text() if last_header else None))

    ctx.step("screenshot the optimize Show-data dialog")
    ctx.shot(dialog, "01-optimize-data-dialog.png")

    ctx.step("feed TolerancePane a synthetic sensitivity + MC run")
    tpane = window.tolerance_pane
    tpane.on_started()
    tpane.on_progress({
        "stage": "tolerance", "phase": "sensitivity_done",
        "sensitivity": [
            {"name": "train.L1.decenter_x", "rank": 1,
             "impact": 0.020, "derivative": 0.5},
            {"name": "train.L2.decenter_x", "rank": 2,
             "impact": 0.012, "derivative": 0.3},
            {"name": "train.L3.decenter_x", "rank": 3,
             "impact": 0.006, "derivative": 0.1},
        ],
    })
    for i in range(1, 9):
        merit = 0.010 + 0.002 * (i % 4)
        tpane.on_progress({
            "stage": "tolerance", "phase": "mc", "draw": i, "draws": 8,
            "merit": merit, "params": {"train.L1.decenter_x": 0.001 * i},
        })
    tpane.on_progress({"stage": "tolerance", "status": "completed",
                       "n_evals": 8})
    tpane.on_finished(0)
    QApplication.processEvents()

    ctx.step("screenshot the merit-distribution plot (polygon+CDF)")
    ctx.shot(tpane.hist_plot, "02-tolerance-merit-distribution.png")

    ctx.step("assert x-axis title 'merit' where QtCharts is available")
    if TOL_HAVE_QTCHARTS:
        title = tpane.hist_plot._ax_x.titleText()
        ctx.check(title == "merit",
                  "MeritDistributionPlot x-axis title=%r, expected 'merit'"
                  % title)
    else:
        print("    [%s] QtCharts unavailable in this interpreter -- "
             "skipping the axis-title assertion (QPainter fallback has "
             "no axis-title API)" % ctx.name)


# ===========================================================================
# Scenario 6: toolbar-contrast
# ===========================================================================
def scenario_toolbar_contrast(ctx, window):
    ctx.step("assert window styleSheet contains QToolButton:checked")
    ctx.check("QToolButton:checked" in window.styleSheet(),
              "window.styleSheet() is missing the 'QToolButton:checked' "
              "contrast rule (widgets/style.checked_toolbutton_stylesheet)")

    anim_tb = window.findChild(QToolBar, "animation_toolbar")
    main_tb = window.findChild(QToolBar, "main_toolbar")
    ctx.check(anim_tb is not None, "animation_toolbar not found")
    ctx.check(main_tb is not None, "main_toolbar not found")

    ctx.step("screenshot toolbars unchecked")
    ctx.shot(anim_tb, "01-animation-toolbar-unchecked.png")
    ctx.shot(main_tb, "02-main-toolbar-unchecked.png")

    ctx.step("toggle anim enable + face indicators ON (no scene needed)")
    window.anim_enable_action.setChecked(True)
    QApplication.processEvents()
    window.face_indicators_action.setChecked(True)
    QApplication.processEvents()
    ctx.check(window.anim_enable_action.isChecked(),
              "anim_enable_action did not stay checked")
    ctx.check(window.face_indicators_action.isChecked(),
              "face_indicators_action did not stay checked")

    ctx.step("screenshot toolbars checked")
    ctx.shot(anim_tb, "03-animation-toolbar-checked.png")
    ctx.shot(main_tb, "04-main-toolbar-checked.png")


# ===========================================================================
# Scenario 7: conical-toggle -- samples-instruments round: the --conical /
# --conical-fan / --conical-delta CLI flags (internal conical refraction at
# biaxial optic axes) surface in the ConfigMatrix's "Run Pipeline" dialog
# purely via cli_specs.py parser introspection (no hand-authored widget
# code in config_matrix.py); this scenario is really exercising that
# introspection path, not conical-refraction physics.
# ===========================================================================
def scenario_conical_toggle(ctx, window):
    ctx.step("open the Run Pipeline config dialog (non-modal: the private "
             "_config_matrix_dialog builder, never .exec())")
    dialog, _buttons = window._config_matrix_dialog("Run Pipeline", "Run")
    dialog.show()
    QApplication.processEvents()
    cm = window.config_matrix

    ctx.step("assert --conical/--conical-fan/--conical-delta were "
             "introspected into config_matrix.widgets")
    for dest in ("conical", "conical_fan", "conical_delta"):
        ctx.check(dest in cm.widgets,
                  "config_matrix.widgets has no entry for %r -- "
                  "cli_specs.py's --%s flag did not surface in the "
                  "auto-built dialog" % (dest, dest.replace("_", "-")))

    ctx.step("assert widget kinds (checkbox / spinbox / validated line edit)")
    ctx.check(isinstance(cm.widgets["conical"], QCheckBox),
              "widgets['conical'] is a %s, expected QCheckBox"
              % type(cm.widgets["conical"]).__name__)
    ctx.check(isinstance(cm.widgets["conical_fan"], QSpinBox),
              "widgets['conical_fan'] is a %s, expected QSpinBox"
              % type(cm.widgets["conical_fan"]).__name__)
    ctx.check(isinstance(cm.widgets["conical_delta"], QLineEdit),
              "widgets['conical_delta'] is a %s, expected QLineEdit"
              % type(cm.widgets["conical_delta"]).__name__)

    ctx.step("assert the three dests live in the 'physics options "
             "(stage: trace)' parser group")
    group_of = {}
    for group in cm._parser._action_groups:
        for action in group._group_actions:
            group_of[action.dest] = group.title
    for dest in ("conical", "conical_fan", "conical_delta"):
        ctx.check(group_of.get(dest) == "physics options (stage: trace)",
                  "%r is in parser group %r, expected "
                  "'physics options (stage: trace)'"
                  % (dest, group_of.get(dest)))

    ctx.step("toggle --conical on; set conical-fan=24, conical-delta=2e-4")
    cm.widgets["conical"].setChecked(True)
    cm.widgets["conical_fan"].setValue(24)
    cm.widgets["conical_delta"].setText("2e-4")
    QApplication.processEvents()

    ctx.step("assert config_matrix.values() round-trips the edits")
    values = cm.values()
    ctx.check(values.get("conical") is True,
              "values()['conical']=%r, expected True" % values.get("conical"))
    ctx.check(values.get("conical_fan") == 24,
              "values()['conical_fan']=%r, expected 24"
              % values.get("conical_fan"))
    ctx.check(values.get("conical_delta") == 2e-4,
              "values()['conical_delta']=%r, expected 2e-4"
              % values.get("conical_delta"))

    ctx.step("scroll the dialog to the conical fields + screenshot")
    scroll = dialog.findChild(QScrollArea)
    if scroll is not None:
        scroll.ensureWidgetVisible(cm.widgets["conical_fan"])
        QApplication.processEvents()
    ctx.shot(dialog, "01-conical-fields.png")
    dialog.close()


# ===========================================================================
# Scenario 8: sample-property -- import a cuvette_square (nested WALL+
# LIQUID pair), sub-select the liquid body, and assign a `sample` registry
# row through the element editor's registry-combo pattern.
# ===========================================================================
def scenario_sample_property(ctx, window):
    ctx.step("new .FCStd scene + import cuvette_square from the catalog")
    ctx.check(os.path.isfile(CUVETTE_SQUARE),
              "fixture missing: %s" % CUVETTE_SQUARE)
    scene_path = os.path.join(ctx.out_dir, "scene.FCStd")
    window.project.new_document(scene_path)
    window.model_path = scene_path
    window.project.import_primitive(CUVETTE_SQUARE, "Cuvette1")
    QApplication.processEvents()
    ctx.check(window.project.is_open(), "project failed to open a fresh document")

    ctx.step("identify the wall (glass) and liquid (water) bodies")
    bodies = window.project.element_bodies("Cuvette1")
    ctx.check(len(bodies) == 2,
              "cuvette_square import produced %d body(ies), expected 2 "
              "(wall+liquid): %r" % (len(bodies), bodies))
    wall_body = liquid_body = None
    for b in bodies:
        material = window.project.body(b)["properties"].get(
            "material", {}).get("value")
        if material == "water":
            liquid_body = b
        else:
            wall_body = b
    ctx.check(liquid_body is not None and wall_body is not None,
              "could not tell the cuvette's wall/liquid bodies apart by "
              "material among %r" % bodies)

    ctx.step("select the element, then sub-select the liquid body via the "
             "inspector member list (same path as the 'selection' scenario)")
    window.selection.select(wall_body, (), origin="scene3d")
    QApplication.processEvents()
    ctx.check(window.selection.element == "Cuvette1",
              "selection.element=%r, expected 'Cuvette1' (a bare-body pick "
              "on a 2-body element should expand)" % window.selection.element)
    member_list = window.inspector.member_list.list
    target_item = None
    for i in range(member_list.count()):
        item = member_list.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == liquid_body:
            target_item = item
            break
    ctx.check(target_item is not None,
              "inspector member list has no row for the liquid body %r"
              % liquid_body)
    window.inspector.member_list._on_item_clicked(target_item)
    QApplication.processEvents()
    ctx.check(window.selection.body == liquid_body,
              "selection.body=%r, expected the liquid body %r"
              % (window.selection.body, liquid_body))

    ctx.step("assert the element editor offers all 7 sample/samples.miesamp "
             "rows in the 'sample' registry combo")
    names = window.element_editor._registry_names("sample")
    ctx.check(len(names) == 7,
              "sample registry combo has %d row(s), expected 7: %r"
              % (len(names), names))

    ctx.step("add the 'sample' property, then set an explicit row")
    window.element_editor.add_prop_combo.setCurrentText("sample")
    window.element_editor.add_prop_button.click()
    pump(0.5)
    chosen = "colloidal_crystal_fcc" if "colloidal_crystal_fcc" in names \
        else names[0]
    window.element_editor._commit_property("sample", chosen)
    pump(0.5)
    value = window.project.body(liquid_body)["properties"].get(
        "sample", {}).get("value")
    ctx.check(value == chosen,
              "liquid body's 'sample' property=%r, expected %r"
              % (value, chosen))

    ctx.step("screenshot the element editor showing the sample assignment")
    ctx.shot(window.element_editor, "01-sample-assigned.png")


# ===========================================================================
# Scenario 9: image-source -- import source_image (bakes image=
# usaf_style_target), assert the `image` registry combo + add the optional
# `image_cone_deg` numeric property.
# ===========================================================================
def scenario_image_source(ctx, window):
    ctx.step("new .FCStd scene + import source_image from the catalog")
    ctx.check(os.path.isfile(SOURCE_IMAGE),
              "fixture missing: %s" % SOURCE_IMAGE)
    scene_path = os.path.join(ctx.out_dir, "scene.FCStd")
    window.project.new_document(scene_path)
    window.model_path = scene_path
    window.project.import_primitive(SOURCE_IMAGE, "Src1")
    QApplication.processEvents()
    ctx.check(window.project.is_open(), "project failed to open a fresh document")

    ctx.step("resolve the imported body's internal name via miewb_group "
             "(Name != Label: FreeCAD keeps the template's original "
             "internal Name, only Label is rewritten to 'Src1')")
    src_bodies = window.project.element_bodies("Src1")
    ctx.check(len(src_bodies) == 1,
              "element_bodies('Src1') returned %r, expected exactly 1 "
              "body (source_image is single-body)" % src_bodies)
    src_name = src_bodies[0]

    ctx.step("select the imported source body")
    window.selection.select(src_name, (), origin="scene3d")
    QApplication.processEvents()
    ctx.check(window.selection.body == src_name,
              "selection.body=%r, expected %r"
              % (window.selection.body, src_name))

    ctx.step("assert the baked image=usaf_style_target property + the "
             "single-row 'image' registry combo")
    body = window.project.body(src_name)
    image_value = body["properties"].get("image", {}).get("value")
    ctx.check(image_value == "usaf_style_target",
              "Src1 image property=%r, expected 'usaf_style_target'"
              % image_value)
    names = window.element_editor._registry_names("image")
    ctx.check(names == ["usaf_style_target"],
              "image registry combo=%r, expected ['usaf_style_target']"
              % names)

    ctx.step("add the optional image_cone_deg field")
    window.element_editor.add_prop_combo.setCurrentText("image_cone_deg")
    window.element_editor.add_prop_button.click()
    pump(0.5)
    cone = window.project.body(src_name)["properties"].get(
        "image_cone_deg", {}).get("value")
    ctx.check(isinstance(cone, float) and 0.0 < cone <= 90.0,
              "image_cone_deg default value=%r, expected a float in (0, 90]"
              % cone)

    ctx.step("screenshot the element editor showing image + image_cone_deg")
    ctx.shot(window.element_editor, "01-image-source.png")


# ===========================================================================
# Scenario 10: library-tabs -- the PropEditorPane "Samples"/"Images" tabs
# (opened via LibraryPane's Project/System-library "Open in editor", or
# directly here through the private _open_prop_editor helper mainwindow
# itself uses) render the 7 sample rows and the 1 image row.
# ===========================================================================
def scenario_library_tabs(ctx, window):
    ctx.step("open the Property Library Editor at the Samples tab")
    window._open_prop_editor("samples", "system")
    QApplication.processEvents()
    editor = window._prop_editor_window
    ctx.check(editor is not None,
              "_open_prop_editor() did not create a PropEditorPane")
    samples_table = editor.editor("samples").table
    ctx.check(samples_table.rowCount() == 7,
              "Samples tab table has %d row(s), expected 7"
              % samples_table.rowCount())

    ctx.step("screenshot the Samples tab")
    ctx.shot(editor, "01-samples-tab.png")

    ctx.step("switch to the Images tab")
    window._open_prop_editor("images", "system")
    QApplication.processEvents()
    images_table = editor.editor("images").table
    ctx.check(images_table.rowCount() == 1,
              "Images tab table has %d row(s), expected 1"
              % images_table.rowCount())

    ctx.step("screenshot the Images tab")
    ctx.shot(editor, "02-images-tab.png")


# ===========================================================================
# Scenario 11: results-galleries -- a synthetic case dir with dummy PNGs
# under instrument/ analysis/ imaging/ dls/ populates all four Results
# galleries, including the NEW dls tab (dls_correlate.py's products).
# ===========================================================================
def scenario_results_galleries(ctx, window):
    ctx.step("fixture a synthetic case dir with a dummy PNG under each of "
             "instrument/ analysis/ imaging/ dls/")
    case_dir = os.path.join(ctx.out_dir, "case")
    swatches = (("instrument", QColor(220, 60, 60)),
                ("analysis", QColor(60, 200, 90)),
                ("imaging", QColor(70, 110, 230)),
                ("dls", QColor(230, 170, 40)))
    for sub, color in swatches:
        _write_dummy_png(os.path.join(case_dir, sub, "%s_demo.png" % sub),
                         color)

    ctx.step("point the Results pane at the fixture (load_case, no monitor)")
    window.central_tabs.setCurrentWidget(window.results)
    window.results.load_case(case_dir, monitor=False)
    QApplication.processEvents()

    ctx.step("assert all four galleries picked up exactly one thumbnail")
    for sub, _color in swatches:
        gallery = window.results.galleries.get(sub)
        ctx.check(gallery is not None,
                  "ResultsPane has no gallery named %r" % sub)
        ctx.check(len(gallery._paths) == 1,
                  "%r gallery has %d thumbnail(s), expected 1: %r"
                  % (sub, len(gallery._paths), gallery._paths))

    ctx.step("screenshot each populated tab, incl. the new DLS tab")
    tab_labels = {"instrument": "Instrument", "analysis": "Analysis",
                 "imaging": "Imaging", "dls": "DLS"}
    for i, (sub, label) in enumerate(tab_labels.items(), start=1):
        idx = _tab_index_by_label(window.results.tabs, label)
        ctx.check(idx >= 0, "Results pane has no %r tab" % label)
        window.results.tabs.setCurrentIndex(idx)
        QApplication.processEvents()
        ctx.shot(window.results, "%02d-%s-tab.png" % (i, sub))


# ===========================================================================
# Scenario 12: lamp-primitives -- the catalog (LibraryPane Elements tab)
# shows the 10 new samples-instruments primitives grouped under their
# categories (Sources: tungsten_halogen/d2_lamp/hg_calibration/
# source_image; Samples & Cells: sample_region + 5 more cells/baths).
# ===========================================================================
def scenario_lamp_primitives(ctx, window):
    ctx.step("read the catalog tree's top-level categories + children")
    tree = window.library.tree
    by_category = {}
    for i in range(tree.topLevelItemCount()):
        group_item = tree.topLevelItem(i)
        by_category[group_item.text(0)] = [
            group_item.child(j).text(0) for j in range(group_item.childCount())]

    ctx.step("assert the new Sources lamps/image-source primitives are present")
    expected_sources = ["Tungsten-halogen (QTH) lamp", "Deuterium (D2) UV lamp",
                        "Hg pen-lamp (calibration)", "Extended image source"]
    ctx.check("Sources" in by_category,
              "catalog tree has no 'Sources' category (found: %r)"
              % sorted(by_category))
    for label in expected_sources:
        ctx.check(label in by_category["Sources"],
                  "'Sources' category missing %r (has: %r)"
                  % (label, by_category["Sources"]))

    ctx.step("assert 'Samples & Cells' has sample_region + >=5 total cells")
    ctx.check("Samples & Cells" in by_category,
              "catalog tree has no 'Samples & Cells' category (found: %r)"
              % sorted(by_category))
    cells = by_category["Samples & Cells"]
    ctx.check("Bare sample region (air)" in cells,
              "'Samples & Cells' category missing 'Bare sample region "
              "(air)' (has: %r)" % cells)
    ctx.check(len(cells) >= 5,
              "'Samples & Cells' category has %d item(s), expected >=5: %r"
              % (len(cells), cells))

    ctx.step("screenshot the catalog tree scrolled to 'Sources' (Elements tab)")
    window.library.tabs.setCurrentIndex(0)
    for i in range(tree.topLevelItemCount()):
        if tree.topLevelItem(i).text(0) == "Sources":
            tree.scrollToItem(tree.topLevelItem(i))
            break
    QApplication.processEvents()
    ctx.shot(window.library, "01-catalog-tree.png")


# ===========================================================================
# Scenario: settings-toolpaths -- the Tool Paths page edits miewb.env
# (single source of truth) in place, preserves comments, and locks
# fields overridden by an exported MIEWB_* variable.
# ===========================================================================
def scenario_settings_toolpaths(ctx, window):
    from mieworkbench.core.settings import Settings, SettingsDialog

    ctx.step("scratch miewb.env copy (never touch the real file)")
    scratch = os.path.join(ctx.out_dir, "scratch-miewb.env")
    with open(scratch, "w") as fh:
        fh.write("# scratch header comment\n"
                 "MIEWB_FREECAD=%s\n"
                 "MIEWB_OPTICS_PYTHON=%s\n"
                 "MIEWB_PVPYTHON=\n"
                 % (common.FREECAD_APPIMAGE or "/missing/FreeCAD.AppImage",
                    common.OPTICS_PYTHON or "/missing/python"))

    ctx.step("dialog shows resolved values + configured-absent state")
    settings = Settings(env_file=scratch)
    dialog = SettingsDialog(settings, window)
    dialog.show()
    QApplication.processEvents()
    ctx.check(dialog._edits["freecad"].text()
              == (common.FREECAD_APPIMAGE or "/missing/FreeCAD.AppImage"),
              "freecad field=%r" % dialog._edits["freecad"].text())
    ctx.check(dialog._status_labels["pvpython"].text()
              == "absent (configured)",
              "pvpython status=%r, expected absent (configured)"
              % dialog._status_labels["pvpython"].text())
    ctx.shot(dialog, "01-toolpaths-resolved.png")

    ctx.step("edit pvpython + OK writes miewb.env, comments survive")
    dialog._edits["pvpython"].setText("/bogus/bin/pvpython")
    QApplication.processEvents()
    ctx.check(dialog._status_labels["pvpython"].text() == "missing",
              "pvpython status=%r after bogus edit, expected missing"
              % dialog._status_labels["pvpython"].text())
    dialog._on_accept()
    text = open(scratch).read()
    ctx.check("MIEWB_PVPYTHON=/bogus/bin/pvpython" in text,
              "edit not written to scratch miewb.env: %r" % text)
    ctx.check("# scratch header comment" in text,
              "comment lost on rewrite: %r" % text)
    ctx.check(settings.pvpython() == "/bogus/bin/pvpython",
              "Settings does not re-read the edited file")

    ctx.step("exported MIEWB_FREECAD locks the field read-only")
    os.environ["MIEWB_FREECAD"] = "/locked/by/env/FreeCAD.AppImage"
    try:
        dialog2 = SettingsDialog(Settings(env_file=scratch), window)
        dialog2.show()
        QApplication.processEvents()
        ctx.check(not dialog2._edits["freecad"].isEnabled(),
                  "freecad field editable despite exported MIEWB_FREECAD")
        ctx.check(dialog2._edits["freecad"].text()
                  == "/locked/by/env/FreeCAD.AppImage",
                  "locked field shows %r, expected the exported value"
                  % dialog2._edits["freecad"].text())
        ctx.shot(dialog2, "02-toolpaths-env-locked.png")
        dialog2.reject()
    finally:
        del os.environ["MIEWB_FREECAD"]
    dialog.reject()


# ---------------------------------------------------------------------------
# scenario registry
# ---------------------------------------------------------------------------
SCENARIOS = [
    ("anim-autopreview", scenario_anim_autopreview),
    ("selection", scenario_selection),
    ("absorbing-stop", scenario_absorbing_stop),
    ("preview-config", scenario_preview_config),
    ("full-trace-ghosts", scenario_full_trace_ghosts),
    ("plot-inspect", scenario_plot_inspect),
    ("toolbar-contrast", scenario_toolbar_contrast),
    ("conical-toggle", scenario_conical_toggle),
    ("sample-property", scenario_sample_property),
    ("image-source", scenario_image_source),
    ("library-tabs", scenario_library_tabs),
    ("results-galleries", scenario_results_galleries),
    ("lamp-primitives", scenario_lamp_primitives),
    ("settings-toolpaths", scenario_settings_toolpaths),
]


def run_scenario(name, func, out_root):
    out_dir = os.path.join(out_root, name)
    ctx = ScenarioContext(name, out_dir)
    window = new_window(out_root)
    t0 = time.monotonic()
    status, message = "PASS", ""
    try:
        func(ctx, window)
    except StepFailure as exc:
        status, message = "FAIL", str(exc)
    except Exception as exc:                          # pragma: no cover
        status, message = "ERROR", "%s: %s" % (type(exc).__name__, exc)
        traceback.print_exc()
    finally:
        teardown_window(window)
    dt = time.monotonic() - t0
    if status == "PASS":
        print("PASS %-20s (%.1fs)" % (name, dt))
    else:
        print("%s %-20s @ %-55s (%.1fs)\n     %s"
             % (status, name, ctx.step_name, dt, message))
    return {"name": name, "status": status, "step": ctx.step_name,
           "message": message, "duration_s": dt,
           "screenshots": ctx.screenshots}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=None,
                    help="run just this scenario (repeatable)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="screenshot root directory (default: var/gui_verify)")
    ap.add_argument("--list", action="store_true",
                    help="print scenario names and exit")
    args = ap.parse_args()

    if args.list:
        for name, _ in SCENARIOS:
            print(name)
        return 0

    only = set(args.only) if args.only else None
    selected = [(n, f) for n, f in SCENARIOS if not only or n in only]
    if only:
        missing = only - {n for n, _ in SCENARIOS}
        if missing:
            print("unknown scenario(s): %s" % ", ".join(sorted(missing)),
                 file=sys.stderr)
            return 2

    os.makedirs(args.out, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    platform = app.platformName()
    print("Qt platform: %s   DISPLAY: %s" % (platform, os.environ.get("DISPLAY")))
    if platform == "offscreen":
        print("WARNING: Qt fell back to the 'offscreen' platform plugin "
             "despite a real DISPLAY -- VTK screenshots will likely come "
             "back blank. Check that Xvfb/GLX is actually reachable "
             "(xvfb-run -a glxinfo).", file=sys.stderr)

    t_start = time.monotonic()
    results = []
    for name, func in selected:
        results.append(run_scenario(name, func, args.out))
    total_dt = time.monotonic() - t_start

    print("\n%-20s %-8s %s" % ("SCENARIO", "STATUS", "DETAIL"))
    print("-" * 78)
    n_fail = 0
    for r in results:
        detail = "" if r["status"] == "PASS" else "@ %s -- %s" % (
            r["step"], r["message"])
        print("%-20s %-8s %s" % (r["name"], r["status"], detail))
        if r["status"] != "PASS":
            n_fail += 1
        for path, kind in r["screenshots"]:
            print("    %-9s %s" % (kind, os.path.relpath(path, REPO)))
    print("-" * 78)
    print("%d/%d scenarios passed  (%.1fs total)"
         % (len(results) - n_fail, len(results), total_dt))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
