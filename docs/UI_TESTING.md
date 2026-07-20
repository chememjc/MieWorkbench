# Testing the MieWorkbench GUI

> This is the GUI test-authoring reference. For the engine/pipeline see
> `docs/RAYTRACER.md`; for the GUI architecture map see the top-level
> `CLAUDE.md`.

`mieworkbench/tests/` (1219 `def test_` functions — ~1550 collected after parametrization; see `--collect-only -q` for the exact count) runs under the **GUI
venv** interpreter (`env/bin/python`, PySide6 6.11 + vtk 9.6), never the
optics env or system python3 — see CLAUDE.md's interpreter table. Almost
everything runs **offscreen** (no real X server, no GPU): Qt's `offscreen`
platform plugin gives real `QWidget`/`QMainWindow`/`QAction` objects with
real signal/slot wiring, but no window ever appears and no OpenGL context
exists. That single constraint (no GL) shapes every convention below.

## 1. Test tiers

### (a) Pure / offscreen widget & pane tests
No FreeCAD, no real project — a `FakeProject` (see §2) or canned scene
dicts stand in wherever a `Project` is needed. This is the bulk of the
suite: `test_vtkview.py`, `test_facepicker.py`, `test_facemaps.py`,
`test_faceindicators.py`, `test_scalebar.py`, `test_transforms.py`,
`test_prop_editor.py`, `test_outliner.py`, `test_element_editor.py`,
`test_config_matrix.py`, `test_console.py`, `test_units.py`,
`test_previewscheduler.py`, `test_wizards.py`, etc.

```bash
QT_QPA_PLATFORM=offscreen env/bin/python -m pytest mieworkbench/tests -q
```

### (b) MainWindow offscreen tests
`test_mainwindow.py` — constructs the real `MainWindow` (docks, menus,
runner, results pane, ray-dimming menu) with no document open and drives
it directly (`window._on_progress(...)`, `window._on_dry_run()`,
`window._set_ray_dimming_floor(12.5)`). Same command as tier (a) — it's
the same `pytest` invocation, just a slower-to-construct fixture (one full
`MainWindow` per test).

Both (a) and (b) together: **~1490 passed, 57 skipped in ~150 s** on this
machine (the skips are `needs_gl` + `freecad`-marked tests, see below).
Run this tier on every change to `mieworkbench/`; it's cheap enough to run
before every commit.

### (c) FreeCAD integration tier
`test_mainwindow_integration.py`, `test_fcserver_integration.py`,
`test_view3d_freecad.py` and anything marked `@pytest.mark.freecad` opens
a real `example.FCStd`/`.MieWB` through the real `fc_server.py` AppImage
worker (`Project.open_model`, `import_primitive`, `undo`/`redo`,
`rebuild_primitive`, real body counts). `conftest.py` registers the marker
and skips it by default:

```python
def pytest_collection_modifyitems(config, items):
    if os.environ.get("MIEWB_RUN_FREECAD") == "1":
        return
    skip = pytest.mark.skip(reason="set MIEWB_RUN_FREECAD=1 to run "
                                   "FreeCAD integration tests")
    for item in items:
        if "freecad" in item.keywords:
            item.add_marker(skip)
```

Run with:

```bash
MIEWB_RUN_FREECAD=1 QT_QPA_PLATFORM=offscreen env/bin/python -m pytest mieworkbench/tests -q
```

Slow (each test launches/tears down a real FreeCAD AppImage worker
process) — budget tens of seconds per test, not milliseconds. Required
before anything touching `fcclient.py`, `fcops.py`, `project.py`'s worker
plumbing, or the add/copy/paste/delete/undo element flows — tier (a)/(b)
can't catch a real protocol mismatch or a real recompute side effect.

`needs_gl`-marked tests (`test_facepicker_gl.py`) are declared as a marker
in `conftest.py` but **not** auto-skipped there — the skip condition
(`QT_QPA_PLATFORM == "offscreen"`) lives in the test file itself, at the
point where it's actually true, since unlike the FreeCAD tier there's no
env var to opt back in on this sandbox (no GL is ever available here).

## 2. Core conventions

**`is_offscreen()` no-op guards** (`mieworkbench/widgets/vtkview.py`
~line 86): building `VtkSceneView` — actors, mappers, STL/VTP readers, the
orientation-marker widget — never touches the GPU; only
`vtkRenderWindow.Initialize()`/`Render()`/`EnabledOn()` do real OpenGL work,
and those crash immediately (`BadWindow`) under the offscreen plugin.
`is_offscreen()` gates every such call site so the widget stays fully
constructible and its Python-side state fully mutable — just without a
real repaint — headless. The corollary for test-writing: **assertions
read actor/mapper/transform state, never pixels** (`actor.GetProperty().
GetColor()`, `mapper.GetScalarVisibility()`, `transform.GetMatrix()`,
`view._dim_mode`) — see every test in `test_vtkview.py`. If a test needs
to prove something happens on *screen*, it belongs in the `needs_gl` tier
instead, not faked with pixel assumptions.

**qtbot usage pattern**: construct the widget, `qtbot.addWidget(widget)`
(hands teardown to pytest-qt so the widget is properly deleted between
tests), then call its methods directly — no event-loop waiting needed for
the offscreen tiers since nothing here is asynchronous animation:

```python
view = VtkSceneView()
qtbot.addWidget(view)
view.load_bodies(faces, structure)
```

**Fixtures in `vtk_test_support.py`** (not a test module — no `test_`
prefix, so pytest never collects it):
- `write_triangle_stl(path, base=(0,0,0), size=0.01)` — a trivial valid
  binary STL (one triangle), so tests never need a real
  FreeCAD/tessellation pipeline to get a face mesh on disk.
- `make_two_body_scene(tmp_path)` — a Lens (optic, 1 face) + Screen
  (detector, 1 face) `Project.structure`/`Project.faces`-shaped pair; the
  minimal scene for widget-construction tests.
- `make_lens_two_faces_scene(tmp_path)` — a Lens with two faces, for
  facemap partial-assignment tests (assigning only one of several faces
  must NOT collapse to the bare whole-body form).
- `write_simple_vtp(path, with_rgb=False, rel_power=None)` — a minimal
  `.vtp` polyline writer for the ray-overlay tests; `with_rgb` adds the
  `rgb` cell array as a plain (non-active) array on purpose — real
  pre-fix `rays.vtp` files carry it that way, and the GUI's field-data
  coloring path has to handle exactly that.
- `FakeProject` — a tiny `QObject` with the real `Project` signals
  (`sceneLoaded`, `bodiesReshaped`, `bodiesMoved`, `propertiesChanged`,
  `dirtyChanged`) and canned in-memory `structure`/`faces`, used by every
  offscreen *pane* test that needs a project-shaped object but must never
  touch FreeCAD. It logs every mutating call to `self.calls` so tests can
  assert call order/arguments as well as end state.

**Modal-dialog-in-teardown trap**: a `QDialog.exec()` (or `QInputDialog`,
`QMessageBox`) opened during a test, or during a widget's `closeEvent`,
blocks forever offscreen — no user to dismiss it, no real desktop event
loop. `mainwindow.py` guards every teardown-path modal on `isVisible()`
first (`mainwindow.py:742,1256,1311,1732`, e.g. `if self.isVisible():
...`) — an offscreen-constructed, never-`.show()`n window is never
visible, so the prompt is skipped and the action falls back to its
non-interactive default (see `test_unsaved_prompt_skipped_when_hidden`
and `test_revert_discards_unsaved_changes` in
`test_mainwindow_integration.py`: "hidden windows never block on the
modal: treated as discard"). The complementary idiom for *driving* a
dialog-fronted action from a test: **dialog-free setters**.
`MainWindow._set_ray_dimming_floor(pct)` exists solely so
`test_ray_dimming_menu_exists_and_persists` never has to open the real
"set floor %" `QInputDialog`. When a menu action pops a modal for a
single value/confirmation, add (or reuse) a private `_set_*`/`_on_*`
method that performs the mutation without the dialog, and test through
that.

**`needs_gl` marker**: for tests that must genuinely rasterize and pick
against real pixels (`test_facepicker_gl.py`'s center-click-picks-the-
framed-face test, which calls `interactor.Initialize()` and
`GetRenderWindow().Render()` for real). Mark with
`pytestmark = pytest.mark.needs_gl` at module level, then guard the actual
test(s) with `@pytest.mark.skipif(_OFFSCREEN, reason="needs a real OpenGL
context")` where `_OFFSCREEN = os.environ.get("QT_QPA_PLATFORM", "").
lower() == "offscreen"`. Keep the pure selection-set logic those GL tests
exercise (`pick_to_selection`, `normalize_pick_mode`, `event_pick_mode`,
`select_all`) in a separate Qt/VTK-free module (`widgets/facepicker.py`)
so it's covered by ordinary offscreen tests too (`test_facepicker.py`) —
don't let "needs a render window to pick" become an excuse to leave the
selection arithmetic untested.

**QSettings pollution**: tests that touch `window.settings` run against
the *real* `CurtisAnalytical/MieWorkbench` `QSettings` store on this
machine — there is no test-only settings sandbox. Save every key you're
about to mutate, mutate/assert, then restore in a `finally:`. The pattern,
from `test_mainwindow.py::test_ray_dimming_menu_exists_and_persists`:

```python
saved_mode = window.settings._qs.value("ray_dimming_mode", None)
saved_floor = window.settings._qs.value("ray_dimming_floor", None)
try:
    ... trigger actions, assert window.settings.get(...) ...
finally:
    for key, val in (("ray_dimming_mode", saved_mode),
                     ("ray_dimming_floor", saved_floor)):
        if val is None:
            window.settings._qs.remove(key)
        else:
            window.settings._qs.setValue(key, val)
```

Any new test that reads or writes a persisted `QSettings` key must follow
this save/mutate/restore-in-finally shape, or it will leak state into
whichever test (or developer's real GUI session) runs next.

## 3. How to write a new test

**Widget test** (VtkSceneView overlay style — state assertions only, no
pixels):

```python
def test_vtp_overlay_uses_rgb_when_present_else_uniform_color(qtbot, tmp_path):
    view = VtkSceneView()
    qtbot.addWidget(view)
    plain_path = tmp_path / "rays_plain.vtp"
    write_simple_vtp(plain_path, with_rgb=False)
    actor = view.load_vtp_overlay(plain_path)
    assert actor.GetMapper().GetScalarVisibility() == 0
    view.remove_overlay()
    assert view._rays_actor is None
```

**Pane test** (`FakeProject` style — no FreeCAD, assert on the fake's call
log and in-memory structure):

```python
def test_set_property_updates_structure_and_emits(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = SomePane(project)          # whatever pane under test
    qtbot.addWidget(pane)

    pane.apply_material("Lens", "BK7")   # drive the pane's own API
    assert project.calls[-1] == ("set_property", "Lens", "material", "BK7")
    assert project.body("Lens")["properties"]["material"]["value"] == "BK7"
```

**MainWindow test** (menu/action assertion style — construct once, drive
methods, no document needed unless the test says so):

```python
def test_close_and_revert_actions_exist_and_start_disabled(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert not window.close_action.isEnabled()
    assert not window.revert_action.isEnabled()
    window._on_close_model()      # nothing open: must be a harmless no-op
    window._on_revert()
    assert not window.project.is_open()
```

## 4. Manual smoke checklist

Some behavior is genuinely visual (animation, live re-color, camera
framing) and offscreen tests can't see it — no GL context means no actual
frame to look at. Run this checklist by hand after any change that
touches rendering, the ray-preview pipeline, or session lifecycle:

1. Launch: `env/bin/python -m mieworkbench <model.FCStd|X.MieWB>` (or
   `bin/mieworkbench`).
2. Model loads: bodies render with correct role coloring (red-ish
   sources, gray-blue translucent detectors, glassy light-blue optics).
3. Live ray-preview fan appears shortly after load and auto-updates
   (~1 s debounce) after editing a body property or spreadsheet value.
4. View ▸ Ray Dimming: Off / Linear / Perceptual (sqrt) each visibly
   re-fade the currently loaded rays live, with no reload needed.
5. The toolbar Extinction/dimming combo (if present) stays in sync with
   the View menu's checked state in both directions.
6. Rays menu ▸ "Live ray preview…" opens the **Preview Configuration**
   dialog; confirm all five section groups appear (Ray pattern, Trace
   engine, Overlay display, Tracer-bead animation, Advanced) and that the
   Advanced "Pattern spec" text field stays in sync with the Ray pattern
   widget's Fan/Rings controls in BOTH directions (edit one, watch the
   other update; an invalid typed spec shows an inline error and leaves
   the widget at its last valid state).
7. In the same dialog: with Trace engine on "Sequential (fast, no
   reflections)" and Ray extinction Off, accept and launch — the preview
   shows only the primary transmitted chain (no reflection/Fresnel
   ghosts). Reopen, switch Trace engine to "Full trace (shows
   reflections)" — Ray extinction auto-jumps to Logarithmic (dB). Reopen
   again, explicitly set Ray extinction to Linear or Perceptual, then
   flip the engine Sequential→Full again — the explicit choice is NOT
   overridden this time. Launch on Full — reflection/ghost rays appear
   that Sequential didn't show.
8. Per-document persistence round-trip: accept the Preview Configuration
   dialog with a non-default engine/pattern, then File ▸ Open the same
   model again (or reload) — reopening the dialog shows the SAME
   engine/pattern it was left in, not the QSettings/app default.
9. Edit geometry: overlay rays grey out (stale) immediately; re-running
   the preview (or the sim) restores full color.
10. Run a quick simulation (Simulation ▸ Run, `quick` preset) and confirm
    the Results pane populates (lightbox galleries, per-element Power
    tab).
11. File ▸ Open a second model: confirm Results pane, console, stage
    chips, ray overlay, AND run config all reset to the new model's state
    (no stale rays/results bleeding across models).
12. Scale bar updates to the new model's extent; face-indicator toggles
    (View menu) show/hide the emit/detector red half-discs and +x
    blue/green dots.
13. View ▸ Tracer Bead Animation, then Play: beads ride the loaded rays,
    visibly slower inside glass than in air/vacuum; Pause holds them in
    place, Step advances exactly one frame, Stop rewinds to the sources;
    the `t = ...` / path readout advances during Play and the animation
    loops at the last arrival instead of stopping dead.
14. File ▸ Settings ▸ Defaults tab: change the bead speed and the
    extinction (ray dimming) mode, then OK — the Animation toolbar and
    View menu update live to match, and both values survive an app
    restart.
15. Position / Orientation ▸ Absolute: select an element; the world
    X/Y/Z and Rx/Ry/Rz fields show its live pose and follow it as it
    moves. Type a new position and **Set position** (then **Set
    orientation**) — the element jumps to exactly that pose. The
    **Relative to:** readout tracks the optical-center offset from the
    chosen reference.
16. Snap-to-axis (open the `prism_spectrometer` demo): select the iris,
    **Pick target face…**, click the Camera lens — the iris rotates onto
    the deviated-beam axis and centers on it. Drag along the axis in the
    3D view (Esc cancels; a click commits), or type an offset and **Apply
    offset**. A single **Undo** restores the iris to where it started
    (the validated pain case that used to need offline trigonometry).
17. Optical train round-trip (open the `michelson_folded` demo): in the
    **Optical Train** dock, uncheck FoldA's and FoldB's fold boxes — the
    M1 arm re-collinearizes onto the straight michelson layout, both fold
    mirrors ghost in the 3D view and gain "(excluded)" badges in the
    outliner, and the ray preview straightens. Re-check both — the folded
    layout restores EXACTLY. One **Undo** per toggle.
18. Chain ripple: still in michelson_folded, edit M1's Distance cell
    (e.g. `arm1 - fold_in - fold_up` → a plain number) — downstream
    stays consistent and ONE Undo restores. Drag M2 in the 3D view — the
    Screen follows rigidly and M2's chain fields re-derive to literals.
19. Variables + sweep: in the **Variables** dock, tick `arm2`'s Sweep
    box, then **Run Pipeline** — the pre-sweep summary dialog shows the
    run count and time estimate (Cancel aborts cleanly). Run it; when
    the sweep finishes the **Compare** dock populates (metric-vs-arm2
    plots, per-variant gallery, signed difference maps, scrub slider).
20. Export: **File → Export FCStd…** writes a standalone copy in the
    CURRENT fold state; open it in plain FreeCAD — bodies, dim sheets,
    the miewb_vars sheet and the MieTrain property group are all
    visible/editable there.

> This checklist grows as rounds land; add new manual steps here.

## 5. Where tests live / naming

One file per module under test: `test_<module>.py` for `mieworkbench/
widgets/<module>.py`, `mieworkbench/panes/<module>.py`, or
`mieworkbench/core/<module>.py` (e.g. `widgets/vtkview.py` ↔
`test_vtkview.py`, `core/transforms.py` ↔ `test_transforms.py`,
`panes/prop_editor.py` ↔ `test_prop_editor.py`). One behavior per test,
named for what it asserts, not what it calls (`test_ctrl_click_removes_
when_present`, not `test_pick_to_selection_2`).

Where a GL-dependent widget wraps pure logic, split the logic out into a
plain Python/numpy module with no Qt or VTK import and test *that*
directly and exhaustively — the GL-touching wrapper only needs a thin
smoke test (or the `needs_gl` tier). `mieworkbench/core/transforms.py` is
the model: quaternion/matrix math, rotate-about-point, the reference
resolver, and `element_bounds` are pure functions tested in
`test_transforms.py` with plain numpy arrays and no `qtbot`/`QWidget` at
all — the same split `widgets/facepicker.py`'s `pick_to_selection` /
`normalize_pick_mode` follows for the pick-mode arithmetic behind the
GL-only `FacePicker` class. Keep new physics- or logic-heavy code Qt-free
by default; only reach for a real widget/window fixture when the thing
under test IS the Qt/VTK wiring itself.
