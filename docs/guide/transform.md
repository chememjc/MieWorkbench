# Position / Orientation

`mieworkbench/panes/transform_panel.py` (`TransformPanel`, dock).

## What it does

Translate and rotate the selected element with **repeatable operations**
built on `core/transforms.Operation`:

- **Translate** by a vector, or toward a reference point by a distance.
- **Rotate** about an axis (global / custom / face normal / optical
  axis) around a reference point (origin default, or any fixed/element
  point).

Reference points are resolved **live at apply time**
(`project.resolver().resolve_point(spec)`), so clicking "Apply again"
after other moves keeps meaning "toward the lens" rather than replaying a
stale coordinate.

## Reference point kinds (`ReferencePointPicker`)

| Kind | Anchored to |
|---|---|
| Origin | world origin |
| Fixed point… | a typed `x, y, z` (mm) |
| Element: optical center | another element's optical center |
| Element: center of mass | another element's CoM |
| Element: bbox center | another element's bounding-box center |
| Element: point on face normal | a face centroid offset along its normal by a typed distance (mm) |

The picker shows the live resolved coordinate (`→ (x, y, z) mm`) so a
reference spec's meaning is always visible before applying.

## How to use it

Select an element (outliner, 3D viewport, or train editor), fill in the
translate/rotate fields and reference points, then apply. A compact train
positioning strip also lives here for chained elements — the full editing
surface is the [Train Editor](train-editor.md).

## Gotchas

- A body whose `Placement` is expression-bound (`.Placement.Base.y =
  <<dim>>.lenspos`) cannot be written directly — FreeCAD silently undoes
  such a write on recompute. `fcops` refuses and names the driving alias;
  this panel routes the move through the correct API instead of a raw
  placement write, but a hand-authored scene with such a binding will
  surface the refusal.
- For a **chained** element, moving it here does not detach it from the
  train — `Project.move_element`/`sync_chain_from_pose` re-derive the
  chain's edge fields (distance/decenter/tilt) from the spatial drag, so
  downstream elements still follow rigidly.
