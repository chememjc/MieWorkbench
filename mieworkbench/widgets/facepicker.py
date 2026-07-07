"""FacePicker - click-to-select face picking for VtkSceneView.

Two independent halves:

* `pick_to_selection` / `select_all` - pure Python set arithmetic
  implementing the selection semantics (plain click replaces the
  selection, Ctrl+click toggles membership). No VTK involved, so this is
  exercised directly by unit tests with no GL/offscreen concerns.

* `FacePicker` - the vtkCellPicker wiring: installed on a
  QVTKRenderWindowInteractor, resolves left-button-press events to a face
  via an {actor: (body_name, face_id)} map (owned by VtkSceneView) and
  calls back with (body_name, face_id, additive); additive is True when
  Ctrl was held. A miss (empty space, or an actor that isn't tracked)
  calls back with (None, None, additive) so callers can decide whether an
  empty-space click should clear the current selection. This half needs a
  real render window to do anything useful, so it's only exercised by
  @pytest.mark.needs_gl tests (skipped offscreen).
"""

from vtkmodules.vtkRenderingCore import vtkCellPicker


def pick_to_selection(current_selection, picked_face_id, additive,
                      all_faces=None):
    """New selection set after one pick.

    - picked_face_id is None: a miss. additive misses leave the selection
      untouched (Ctrl+click on empty space shouldn't discard your
      selection); a plain miss clears it.
    - additive=False (plain click): selection becomes exactly {face}.
    - additive=True (Ctrl+click): toggles membership of face in the
      existing selection.

    `all_faces`, if given, is used only to validate picked_face_id is a
    real face of the current scene (defends against a stale actor->face
    map after a reload); it never changes the *shape* of the result.
    """
    current = set(current_selection or [])
    if picked_face_id is None:
        return current if additive else set()
    if all_faces is not None and picked_face_id not in set(all_faces):
        raise ValueError("picked face %r is not in all_faces" % picked_face_id)
    if not additive:
        return {picked_face_id}
    new_selection = set(current)
    if picked_face_id in new_selection:
        new_selection.discard(picked_face_id)
    else:
        new_selection.add(picked_face_id)
    return new_selection


def select_all(all_faces):
    """Every face id in `all_faces` -> a fresh selection set."""
    return set(all_faces or [])


class FacePicker:
    """vtkCellPicker glue: installs a LeftButtonPressEvent observer on
    `interactor`; on each press, picks against `renderer` and resolves the
    hit actor through `actor_face_map` (a live dict owned by the caller --
    read at pick time, never copied, so it always reflects the current
    scene). Calls `on_pick(body_name, face_id, additive)`."""

    def __init__(self, interactor, renderer, actor_face_map, on_pick):
        self.interactor = interactor
        self.renderer = renderer
        self.actor_face_map = actor_face_map
        self.on_pick = on_pick
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(0.0005)
        self._observer_id = interactor.AddObserver(
            "LeftButtonPressEvent", self._on_left_button_press)

    def _on_left_button_press(self, obj, event):
        interactor = self.interactor
        x, y = interactor.GetEventPosition()
        additive = bool(interactor.GetControlKey())
        self.picker.Pick(x, y, 0, self.renderer)
        actor = self.picker.GetActor()
        entry = self.actor_face_map.get(actor) if actor is not None else None
        if entry is not None:
            body_name, face_id = entry
            self.on_pick(body_name, face_id, additive)
        else:
            self.on_pick(None, None, additive)
        # let the trackball style still handle camera rotation/pan
        style = interactor.GetInteractorStyle()
        if style is not None:
            style.OnLeftButtonDown()

    def detach(self):
        if self._observer_id is not None:
            self.interactor.RemoveObserver(self._observer_id)
            self._observer_id = None
