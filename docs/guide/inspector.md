# Element Inspector

`mieworkbench/panes/inspector3d.py` (`InspectorPane`, dock).

## What it does

The single-element 3D view: shows only the currently selected body,
centered and camera-fit. This is the **primary face-selection surface** —
the [3D Viewport](viewport-3d.md) can pick faces too, but this pane is
where a face selection for the [Element Editor](element-editor.md)'s
Active Properties table is expected to be built.

`set_body(project, body_name)` tears down and rebuilds the view around
just that body, and (re-)connects to the `Project` so a reshape/move/
property edit of the shown body refreshes the view automatically — no
need to call `set_body()` again after every edit.

## Multi-body elements

A single-body element routes straight to `set_body` (its one body IS the
face-selection surface). A **multi-body element** (or an empty
selection) routes to `set_element(project, element, bodies)` instead: the
3D view is blanked and swapped for a neutral state — a hint line
("Element *Name* — *N* bodies. Pick one to inspect it." or "No element
selected.") over a clickable **member list** (`MemberListWidget`, shared
with the [Element Editor](element-editor.md)). Clicking a member is an
explicit **sub-selection** of that one body — it routes through the
shared `SelectionModel` with a distinct origin so the dispatcher treats
it as a sub-selection rather than re-expanding back to the whole element,
then behaves exactly like a single-body element (`set_body`). Face
picking on a single body is otherwise unchanged.

## How to use it

- Click a face to select it; **Shift+click** extends, **Ctrl+click**
  toggles (same semantics as the 3D viewport).
- **Select all faces** button selects every face of the current body.
- **Clear** clears the face selection.
- **Rays** toggle requests a ray-overlay preview scoped to this element
  (asks the orchestrator to run one if none is loaded).
- **Right-click** opens the Active Properties apply/remove menu tree
  (`build_active_properties_menu`, wired by the main window) — the same
  menu the Element Editor's table exposes, so per-face coating/roughness/
  grating/etc. assignments can be made straight from the 3D pick.

`faceSelectionChanged(body_name, {face_id, ...})` feeds
`ElementEditorPane.set_face_selection`, which filters its Active
Properties table to assignments touching the selection.

## Gotchas

- Right-click in this pane trades away the stock right-drag dolly (VTK's
  trackball style) for the context menu — that trade is scoped to this
  pane only; the 3D viewport keeps right-drag dolly.
