"""FacePicker - click-to-select face picking for VtkSceneView.

Two independent halves:

* `pick_to_selection` / `select_all` - pure Python set arithmetic
  implementing the selection semantics (plain click replaces the
  selection, Ctrl+click toggles membership, Shift+click extends -- adds
  without ever removing). No VTK involved, so this is exercised directly
  by unit tests with no GL/offscreen concerns.

* `FacePicker` - the vtkCellPicker wiring: installed on a
  QVTKRenderWindowInteractor, resolves left-button-press events to a face
  via an {actor: (body_name, face_id)} map (owned by VtkSceneView) and
  calls back with (body_name, face_id, mode); mode is "toggle" when Ctrl
  was held (with or without Shift), "extend" for Shift alone, else
  "replace". A miss (empty space, or an actor that isn't tracked) calls
  back with (None, None, mode) so callers can decide whether an
  empty-space click should clear the current selection. Optionally
  (enable_context) it also intercepts right-button presses and reports
  them as context-menu requests INSTEAD of letting the trackball style
  start a dolly (a popup menu would swallow the matching button-release
  and leave the camera stuck mid-dolly; scroll-wheel zoom still works).
  This half needs a real render window to do anything useful, so it's
  only exercised by @pytest.mark.needs_gl tests (skipped offscreen).
"""

from vtkmodules.vtkRenderingCore import vtkCellPicker

PICK_MODES = ("replace", "toggle", "extend")


def normalize_pick_mode(mode):
    """Named mode string, accepting the legacy boolean `additive` flag
    (False -> "replace", True -> "toggle") for back-compat."""
    if mode is True:
        return "toggle"
    if mode is False or mode is None:
        return "replace"
    if mode not in PICK_MODES:
        raise ValueError("unknown pick mode %r" % (mode,))
    return mode


def pick_to_selection(current_selection, picked_face_id, mode,
                      all_faces=None):
    """New selection set after one pick.

    - picked_face_id is None: a miss. A "replace" miss clears the
      selection; "toggle"/"extend" misses leave it untouched (a modified
      click on empty space shouldn't discard your selection).
    - mode "replace" (plain click): selection becomes exactly {face}.
    - mode "toggle" (Ctrl+click): toggles membership of face in the
      existing selection.
    - mode "extend" (Shift+click): adds face, never removes -- there is
      no meaningful "range" on an unordered 3D surface, so Shift is
      add-only (matching mainstream CAD behavior).
    The legacy boolean `additive` third argument still works
    (False -> replace, True -> toggle).

    `all_faces`, if given, is used only to validate picked_face_id is a
    real face of the current scene (defends against a stale actor->face
    map after a reload); it never changes the *shape* of the result.
    """
    mode = normalize_pick_mode(mode)
    current = set(current_selection or [])
    if picked_face_id is None:
        return set() if mode == "replace" else current
    if all_faces is not None and picked_face_id not in set(all_faces):
        raise ValueError("picked face %r is not in all_faces" % picked_face_id)
    if mode == "replace":
        return {picked_face_id}
    if mode == "extend":
        return current | {picked_face_id}
    new_selection = set(current)
    if picked_face_id in new_selection:
        new_selection.discard(picked_face_id)
    else:
        new_selection.add(picked_face_id)
    return new_selection


def select_all(all_faces):
    """Every face id in `all_faces` -> a fresh selection set."""
    return set(all_faces or [])


def event_pick_mode(ctrl, shift):
    """Modifier keys -> pick mode (Ctrl wins over Shift)."""
    if ctrl:
        return "toggle"
    if shift:
        return "extend"
    return "replace"


class FacePicker:
    """vtkCellPicker glue: installs a LeftButtonPressEvent observer on
    `interactor`; on each press, picks against `renderer` and resolves the
    hit actor through `actor_face_map` (a live dict owned by the caller --
    read at pick time, never copied, so it always reflects the current
    scene). Calls `on_pick(body_name, face_id, mode)`."""

    def __init__(self, interactor, renderer, actor_face_map, on_pick):
        self.interactor = interactor
        self.renderer = renderer
        self.actor_face_map = actor_face_map
        self.on_pick = on_pick
        self.on_context = None
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(0.0005)
        self._observer_id = interactor.AddObserver(
            "LeftButtonPressEvent", self._on_left_button_press)
        self._context_observer_id = None

    def enable_context(self, on_context):
        """Opt in to right-button interception: `on_context(x, y)` is
        called with the VTK event position (origin bottom-left) and the
        trackball style is kept out of its right-drag dolly mode (see
        module docstring for why). Opt-in only -- views that don't pop a
        menu keep VTK's stock right-drag zoom."""
        self.on_context = on_context
        if self._context_observer_id is None:
            # priority above the style's (0.0) so the abort flag below
            # stops the event before the style ever sees it
            self._context_observer_id = self.interactor.AddObserver(
                "RightButtonPressEvent", self._on_right_button_press, 10.0)

    def _on_left_button_press(self, obj, event):
        interactor = self.interactor
        x, y = interactor.GetEventPosition()
        mode = event_pick_mode(bool(interactor.GetControlKey()),
                               bool(interactor.GetShiftKey()))
        self.picker.Pick(x, y, 0, self.renderer)
        actor = self.picker.GetActor()
        entry = self.actor_face_map.get(actor) if actor is not None else None
        if entry is not None:
            body_name, face_id = entry
            self.on_pick(body_name, face_id, mode)
        else:
            self.on_pick(None, None, mode)
        # let the trackball style still handle camera rotation/pan
        style = interactor.GetInteractorStyle()
        if style is not None:
            style.OnLeftButtonDown()

    def _on_right_button_press(self, obj, event):
        obj.SetAbortFlag(1)   # keep the trackball style out of dolly mode
        if self.on_context is not None:
            x, y = self.interactor.GetEventPosition()
            self.on_context(x, y)

    def detach(self):
        if self._observer_id is not None:
            self.interactor.RemoveObserver(self._observer_id)
            self._observer_id = None
        if self._context_observer_id is not None:
            self.interactor.RemoveObserver(self._context_observer_id)
            self._context_observer_id = None
