# Outliner

`mieworkbench/panes/outliner.py` (`OutlinerPane`, dock).

## What it does

A list/tree of every element in the scene, so elements can be selected,
deleted, copied and pasted **by name** instead of hunting for them in the
3D view. Rows are *elements*: bodies sharing a `miewb_group` collapse
into one top-level row (label = the group) with member bodies as
children; ungrouped bodies are one row each.

Columns: element / role / primitive kind.

- **Role** (`role_for_body`, shared logic with `Scene3DPane`): `source`
  (has both `power` and `lambdac`), `detector` (`material == "detector"`),
  `ignored` (no/`none` material), else `optic`.
- **Primitive kind**: the body's `miewb_primitive` tag, empty for
  hand-authored bodies.

## How to use it

- Click the **top-level row** to select the **whole element**
  (`selectBodyRequested`) — all member bodies highlight together in the
  3D views.
- Click a **child row** (a member body of a multi-body element) to
  **explicitly sub-select just that one body** — the same sub-selection
  the [Inspector](inspector.md)/[Element Editor](element-editor.md)
  member lists offer, routed through a distinct selection origin so it
  is never re-expanded back to the whole element.
- Double-click to open the editor/wizard for that element
  (`customizeRequested`).
- **Del** key or right-click → Delete removes the element group
  (`deleteRequested`) — undoable, pre-image stashed under
  `<workspace>/undo/`.
- Right-click → Copy / Paste (`copyRequested`/`pasteRequested`); paste
  offsets the new element +x past the occupied bounding boxes so it never
  lands exactly on top of the source.

## Train status badges

`set_train_info` overlays badges on element labels: a chain-link glyph
for chained elements, a fold arrow for fold elements; excluded
(unfolded-away) elements and elements with an active problem finding are
colored distinctly. Badges are computed against a pristine (badge-free)
label captured on first use, so repeated `set_train_info` calls never
compound the decoration.

## Gotchas

- The outliner is dependency-free of the VTK widget on purpose (imports
  cheaply) — it reimplements only the pure classification logic
  (`role_for_body`), not the 3D rendering.
