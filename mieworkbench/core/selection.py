"""SelectionModel - the single source of truth for "which element/faces
are selected", shared by the 3D view, the outliner, the inspector, the
element editor and the transform panel.

Selection used to live implicitly in Scene3DPane (3D clicks only); panes
now subscribe to this model instead, so list-based selection (outliner,
problems pane) and 3D picking stay in sync. `origin` names the pane that
initiated the change - subscribers that would echo the selection back
(the 3D view re-highlighting, the outliner re-selecting its row) skip
updates they originated, breaking feedback loops without blockSignals
gymnastics.

Element vs sub-selection (WP4): a selection is normally a whole ELEMENT
(all member bodies of a multi-body group). `select_element()` records the
member `bodies` (ordered) and an `element` identity while keeping the
scalar `body` pointing at the element's PRIMARY body (so every existing
`body`-consumer keeps working). `select()` is the SUB-selection / single-
body path: it clears `element` and sets `bodies` to just that one body.
The `changed(str, object, str)` signature is unchanged (primary body name,
faces, origin); subscribers that need the element read `element`/`bodies`
off the model directly.
"""

from PySide6.QtCore import QObject, Signal


class SelectionModel(QObject):
    changed = Signal(str, object, str)   # body_name ('' = none), faces, origin

    def __init__(self, parent=None):
        super().__init__(parent)
        self.body = None       # PRIMARY body (element) or the sub-selected body
        self.faces = set()
        self.element = None    # element identity when a whole element is picked
        self.bodies = ()       # ordered member bodies of the current selection

    def select(self, body_name, faces=(), origin=""):
        """Single-body / SUB-selection: `body_name` alone, element cleared."""
        body_name = body_name or None
        faces = set(faces or ())
        bodies = (body_name,) if body_name else ()
        if (body_name == self.body and faces == self.faces
                and self.element is None and self.bodies == bodies):
            return
        self.body = body_name
        self.faces = faces
        self.element = None
        self.bodies = bodies
        self.changed.emit(body_name or "", set(faces), origin)

    def select_element(self, element, bodies, faces=(), origin="",
                       primary=None):
        """Whole-ELEMENT selection: `bodies` are the member bodies (ordered),
        `element` their shared identity. `body` stays the primary (explicit
        `primary`, else bodies[0]) so scalar-body consumers keep working."""
        bodies = tuple(bodies or ())
        faces = set(faces or ())
        primary = primary or (bodies[0] if bodies else None)
        if (element == self.element and primary == self.body
                and faces == self.faces and bodies == self.bodies):
            return
        self.element = element or None
        self.bodies = bodies
        self.body = primary
        self.faces = faces
        self.changed.emit(primary or "", set(faces), origin)

    def is_element(self):
        """True only for a genuine MULTI-body element selection (a single-
        body element still reads as a plain body selection)."""
        return self.element is not None and len(self.bodies) > 1

    def clear(self, origin=""):
        if (self.body is None and not self.faces
                and self.element is None and not self.bodies):
            return
        self.body = None
        self.faces = set()
        self.element = None
        self.bodies = ()
        self.changed.emit("", set(), origin)
