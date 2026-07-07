"""Pure (Qt-free) per-face property assignment logic for the element
editor's "Active Properties" view and context menu.

The contract grammar for per-face ("facemap") property strings lives in
scripts/common.py (parse_facemap_spec); this module is the GUI-side
inverse and set-arithmetic on top of it: grouping a body's raw facemap
strings into Assignment rows (one per distinct property+value), merging a
value onto a face selection, removing faces from an assignment, and
describing the apply/remove context menu as plain data. Every composed
string is re-parsed with parse_facemap_spec as an oracle before it is
returned, so a bug here surfaces as a ValueError instead of a corrupt
property reaching the FreeCAD worker.

Everything here is unit-tested directly (tests/test_facemaps.py) with no
Qt involved; panes/element_editor.py re-exports the names it historically
owned (merge_facemap, active_face_index, validate_facemap_value).
"""

import os
import sys
from collections import namedtuple

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import common  # noqa: E402  (stdlib-only shared contract hub)

# The five per-face properties of the body-tagging contract, in display
# order (docs/RAYTRACER.md §5; keep in sync with extract_geometry.py).
FACEMAP_PROPERTIES = ("coating", "roughness", "diffuser", "grating",
                      "surface_override")

# One row of the Active Properties view: a (property, value) pair and the
# full face ids it applies to. whole_body is True when the value is STORED
# in the bare whole-body form (face_ids then holds every face, expanded).
# An unparseable raw string yields Assignment(prop, raw, frozenset(),
# False) -- empty face_ids with whole_body False marks it invalid; the UI
# shows it as such so hand-typed garbage stays visible and removable.
Assignment = namedtuple("Assignment", "prop value face_ids whole_body")


def assignment_is_invalid(assignment):
    return not assignment.whole_body and not assignment.face_ids


def _bare_face_key(face_id):
    """'Lens.Tip.Face3' -> 'Face3'."""
    spec = common.parse_face_spec(face_id)
    return "Face%d" % spec["face_index"]


def _face_sort_key(face_id):
    return common.parse_face_spec(face_id)["face_index"]


def face_label(face_id):
    """Display label for a full face id ('Body.Pad.Face3' -> 'Face3')."""
    return _bare_face_key(face_id)


def sorted_face_ids(face_ids):
    return sorted(face_ids, key=_face_sort_key)


def _compose_facemap(expanded, all_face_ids, body_name, feature,
                     collapse=True):
    """{full_face_id: value} -> raw string in the bare 'FaceN=value;...'
    form, collapsed to the bare whole-value shorthand when every face of
    the body maps to the same value. Oracle-checked via re-parse.
    collapse=False keeps the explicit per-face form even at full uniform
    coverage -- required for `grating`, whose contract REJECTS the
    whole-body shorthand (it must name specific faces)."""
    all_face_ids = set(all_face_ids or [])
    values_on_all = ({expanded.get(fid) for fid in all_face_ids}
                     if all_face_ids else set())
    if (collapse and all_face_ids and len(expanded) == len(all_face_ids)
            and all_face_ids <= set(expanded) and len(values_on_all) == 1):
        new_raw = next(iter(values_on_all))
    else:
        parts = ["%s=%s" % (_bare_face_key(fid), expanded[fid])
                 for fid in sorted(expanded, key=_face_sort_key)]
        new_raw = ";".join(parts)

    reparsed = common.parse_facemap_spec(new_raw, body=body_name,
                                         feature=feature)
    if common.FACEMAP_ALL in reparsed:
        check = {fid: reparsed[common.FACEMAP_ALL] for fid in all_face_ids}
    else:
        check = reparsed
    if check != expanded:
        raise ValueError("facemap compose failed to round-trip (%r != %r)"
                         % (check, expanded))
    return new_raw


def _expand(existing_raw, body_name, feature, all_face_ids):
    """Parse a raw facemap string into a fully-expanded
    {full_face_id: value} dict (FACEMAP_ALL spread over all_face_ids)."""
    if not existing_raw:
        return {}
    current = common.parse_facemap_spec(str(existing_raw), body=body_name,
                                        feature=feature)
    if common.FACEMAP_ALL in current:
        return {fid: current[common.FACEMAP_ALL]
                for fid in set(all_face_ids or [])}
    return dict(current)


def merge_facemap(existing_raw, body_name, feature, all_face_ids,
                  selected_face_ids, value, collapse=True):
    """Merge `value` onto `selected_face_ids` (full 'Body.Feature.FaceN'
    ids) within a per-face property whose current raw string is
    `existing_raw` (falsy if the property doesn't exist yet). Returns the
    new raw string in the bare 'FaceN=value;...' form, collapsed to the
    bare whole-value form when every face of the body ends up mapped to
    the same value (matches common.py's 'apply to every face' shorthand;
    pass collapse=False for `grating`, which must name explicit faces).
    Re-parses the result with common.parse_facemap_spec as an oracle and
    raises ValueError if it doesn't round-trip.
    """
    expanded = _expand(existing_raw, body_name, feature, all_face_ids)
    for fid in set(selected_face_ids or []):
        expanded[fid] = value
    return _compose_facemap(expanded, all_face_ids, body_name, feature,
                            collapse=collapse)


def remove_faces(existing_raw, body_name, feature, all_face_ids,
                 faces_to_remove, collapse=True):
    """Remove `faces_to_remove` from a facemap string. Returns the new
    raw string, or None when the map empties -- the caller must then
    remove the property from the body instead of setting it."""
    expanded = _expand(existing_raw, body_name, feature, all_face_ids)
    for fid in set(faces_to_remove or []):
        expanded.pop(fid, None)
    if not expanded:
        return None
    return _compose_facemap(expanded, all_face_ids, body_name, feature,
                            collapse=collapse)


def facemap_collapse_allowed(prop):
    """Whether `prop` may use the bare whole-body shorthand. The grating
    contract requires explicit face names (extract_geometry rejects a
    bare grating value)."""
    return prop != "grating"


def assignments_for_body(properties, body_name, feature, all_face_ids):
    """Group a body's facemap property strings into Assignment rows: one
    per distinct (property, value) pair, in FACEMAP_PROPERTIES order then
    sorted by value. `properties` is the Project body-dict form
    {name: {"value": raw, ...}, ...}."""
    props = properties or {}
    out = []
    for prop in FACEMAP_PROPERTIES:
        raw = (props.get(prop, {}) or {}).get("value")
        if raw is None or raw == "":
            continue
        try:
            parsed = common.parse_facemap_spec(str(raw), body=body_name,
                                               feature=feature)
        except ValueError:
            out.append(Assignment(prop, str(raw), frozenset(), False))
            continue
        if common.FACEMAP_ALL in parsed:
            out.append(Assignment(prop, parsed[common.FACEMAP_ALL],
                                  frozenset(all_face_ids or []), True))
            continue
        by_value = {}
        for fid, value in parsed.items():
            by_value.setdefault(value, set()).add(fid)
        for value in sorted(by_value):
            out.append(Assignment(prop, value,
                                  frozenset(by_value[value]), False))
    return out


def filter_assignments(assignments, selected_faces):
    """Assignments touching any of `selected_faces`. A falsy selection
    means "no filter" (the caller shows everything); whole-body
    assignments always match a non-empty selection."""
    selected = set(selected_faces or [])
    if not selected:
        return list(assignments)
    return [a for a in assignments
            if a.whole_body or (a.face_ids & selected)]


def value_check_state(assignments, prop, value, selected_faces,
                      all_face_ids=None):
    """Is (prop, value) applied to "all", "some", or "none" of the
    selected faces? With no selection the target is the whole body
    (all_face_ids). Drives the context-menu checkmarks."""
    target = frozenset(selected_faces or []) or frozenset(all_face_ids or [])
    if not target:
        return "none"
    coverage = set()
    for a in assignments:
        if a.prop == prop and a.value == value:
            coverage |= a.face_ids
    if target <= coverage:
        return "all"
    if target & coverage:
        return "some"
    return "none"


def menu_model(assignments, selected_faces, registry_values, all_face_ids):
    """Describe the "Active Properties" context menu as plain data (no
    Qt): one entry per facemap property, each with value items carrying
    their checked/partial state against the current face selection.

    registry_values: {prop: iterable of offerable value strings} -- the
    caller supplies registry names (as stored-form values, e.g. '@dg_600'
    for diffusers) plus any template entries it wants offered. Values
    already assigned on the body are listed first so hand-typed ones are
    always visible (and removable) even when absent from the registry.
    """
    model = []
    for prop in FACEMAP_PROPERTIES:
        current = []
        for a in assignments:
            if a.prop == prop and not assignment_is_invalid(a) \
                    and a.value not in current:
                current.append(a.value)
        items, seen = [], set()
        for value in list(current) + list(registry_values.get(prop, ())):
            if value in seen or value == "":
                continue
            seen.add(value)
            state = value_check_state(assignments, prop, value,
                                      selected_faces, all_face_ids)
            items.append({"value": value,
                          "checked": state == "all",
                          "partial": state == "some"})
        model.append({"prop": prop, "items": items, "custom": True})
    return model


def active_face_index(properties, faces_meta):
    """The 'working face' of a source/detector body: the face whose
    centroid is closest to the origin -- the same auto-detection heuristic
    extract_geometry uses for the emit/detector face. Returns the face
    INDEX (1-based) or None for plain optics/no-geometry bodies."""
    props = properties or {}
    is_source = "power" in props and "lambdac" in props
    is_detector = (props.get("material", {}).get("value") == "detector")
    if not (is_source or is_detector) or not faces_meta:
        return None
    best_id, best_d = None, None
    for f in faces_meta:
        c = f.get("centroid_m")
        if c is None:
            continue
        d = sum(x * x for x in c)
        if best_d is None or d < best_d:
            best_id, best_d = f["id"], d
    if best_id is None:
        return None
    return common.parse_face_spec(best_id)["face_index"]


def validate_facemap_value(raw, body_name, feature, face_count):
    """Error-check a user-typed facemap value BEFORE it is committed:
    must parse under the contract grammar, and every named face must
    exist on the body. Returns None if ok, else a message."""
    try:
        parsed = common.parse_facemap_spec(str(raw), body=body_name,
                                           feature=feature)
    except ValueError as exc:
        return str(exc)
    for key in parsed:
        if key == common.FACEMAP_ALL:
            continue
        idx = common.parse_face_spec(key)["face_index"]
        if not 1 <= idx <= face_count:
            return ("Face%d does not exist on %s (it has %d face%s)"
                    % (idx, body_name, face_count,
                       "s" if face_count != 1 else ""))
    return None
