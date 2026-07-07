"""librarymgr.py -- system vs. project optical-property (and primitive)
libraries for the MieWorkbench GUI.

Two libraries:
  system  - the repo's opticalproperties/ + primitives/ (read from, written
            to only by promote_to_system(), which is opt-in and validated).
  project - <project_root>/opticalproperties/, a possibly-partial copy that
            holds just the registry rows + table/nk files a given model
            actually uses (ensure_project_item / ensure_project_library_
            selfcontained), so a project directory can be handed to someone
            else and traced with --optical-properties <project>/opticalproperties
            without dragging along the whole system library.

Every write path here goes through core.proplib.Transaction +
validate_and_commit(): touched files are backed up before the write, and
if the post-write PropLibrary.validate() (i.e. the real
load_optical_properties loader) rejects the result, every touched file is
rolled back and LibraryWriteError is raised. The one exception is
ensure_project_item() alone: copying a single item on its own can leave a
still-incomplete (therefore invalid) project library on purpose -- e.g. a
coating row copied before its layer material -- so it does the raw,
idempotent copy without validating; ensure_project_library_selfcontained()
is what pulls in the full transitive closure and validates once at the end.
"""
import csv
import json
import os
import shutil
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import common   # noqa: E402  (stdlib-only shared contract hub)

from .proplib import (CATEGORY_INFO, PropLibrary, PropLibraryError,   # noqa: E402
                      Transaction, validate_and_commit)

try:
    import primitivelib   # noqa: E402  (metadata dict importable w/o FreeCAD)
except Exception:                          # pragma: no cover
    primitivelib = None


class LibraryError(RuntimeError):
    """LibraryManager-level failure (unknown item, no project root, ...)."""


def _find_row(rows, name):
    name_l = name.strip().lower()
    for row in rows:
        if (row.get("name") or "").strip().lower() == name_l:
            return row
    return None


def _write_registry_rows(path, fieldnames, rows):
    """Atomic (tmp + os.replace) full rewrite of a registry csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _append_registry_row(path, fieldnames, row):
    exists = Path(path).exists()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _prop_value(props, key):
    entry = (props or {}).get(key)
    if isinstance(entry, dict):
        v = entry.get("value")
    else:
        v = entry
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def used_names_from_structure(structure):
    """Given a Project.structure-like dict ({"bodies": [{"properties":
    {name: {"value": ...}}}, ...]}), collect the optical-property registry
    names it references.

    -> {"materials": set(), "coatings": set(), "polarizers": set(),
        "filters": set(), "gratings": set()}

    'material'/'polarizer'/'filter' are whole-body scalar properties.
    'coating'/'grating' are per-face maps in facemap syntax ('MgF2' for
    every face, or 'Face3=MgF2;Face5=X' -- common.parse_facemap_spec); a
    grating facemap value may itself be a '@registry_name[:orders=..]'
    reference (common.parse_grating_value), in which case the registry
    name (not the raw facemap value) is what gets collected.
    """
    used = {"materials": set(), "coatings": set(), "polarizers": set(),
            "filters": set(), "gratings": set()}
    for body in (structure or {}).get("bodies", []):
        props = body.get("properties", {}) or {}
        name = body.get("name") or "Body"

        mat = _prop_value(props, "material")
        if mat:
            used["materials"].add(mat)
        pol = _prop_value(props, "polarizer")
        if pol:
            used["polarizers"].add(pol)
        filt = _prop_value(props, "filter")
        if filt:
            used["filters"].add(filt)

        coating = _prop_value(props, "coating")
        if coating:
            facemap = common.parse_facemap_spec(coating, body=name,
                                                feature="Feat")
            for val in facemap.values():
                used["coatings"].add(val.strip())

        grating = _prop_value(props, "grating")
        if grating:
            facemap = common.parse_facemap_spec(grating, body=name,
                                                feature="Feat")
            for val in facemap.values():
                g = common.parse_grating_value(val)
                if g["registry"]:
                    used["gratings"].add(g["registry"])
    return used


def _coating_layer_materials(row):
    """Material tokens named in a TMM coating row's 'layers' field
    ('mgf2:qw@550;sio2_film:25' -> {'mgf2', 'sio2_film'})."""
    out = set()
    for entry in (row.get("layers") or "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        mat = entry.split(":", 1)[0].strip()
        if mat:
            out.add(mat)
    return out


def _uniaxial_materials(row):
    out = set()
    for col in ("n_o_material", "n_e_material"):
        v = (row.get(col) or "").strip()
        if v:
            out.add(v)
    return out


class LibraryManager:
    def __init__(self, system_optprops, system_primitives, project_root=None):
        self.system_optprops = Path(system_optprops)
        self.system_primitives = Path(system_primitives)
        self.system_lib = PropLibrary(self.system_optprops)

        self._project_root = None
        self._project_lib = None
        if project_root is not None:
            self.set_project_root(project_root)

    # -- project root ------------------------------------------------------
    def set_project_root(self, path):
        """Point the project library at <path>/opticalproperties, creating
        that (empty) directory if needed. Copies nothing -- use
        ensure_project_item()/ensure_project_library_selfcontained() to
        populate it."""
        self._project_root = Path(path)
        proj_optprops = self._project_root / "opticalproperties"
        proj_optprops.mkdir(parents=True, exist_ok=True)
        self._project_lib = PropLibrary(proj_optprops)

    @property
    def project_root(self):
        return self._project_root

    @property
    def project_lib(self):
        return self._project_lib

    def _require_project(self):
        if self._project_lib is None:
            raise LibraryError("no project root set (call set_project_root)")
        return self._project_lib

    def _system_row(self, category, name):
        row = _find_row(self.system_lib.registry_rows(category), name)
        if row is None:
            raise LibraryError("%s: no such %s in the system library"
                               % (name, category))
        return row

    # -- project <- system --------------------------------------------------
    def ensure_project_item(self, category, name):
        """Copy `name`'s registry row (and its referenced table/nk file, if
        any) from the system library into the project library. Idempotent:
        a no-op if the row is already present. Returns the list of paths
        written (empty if nothing needed copying). Does NOT validate on its
        own -- a lone item can legitimately leave the project library
        incomplete (e.g. a coating copied before its layer material)."""
        project_lib = self._require_project()
        row = self._system_row(category, name)
        info = CATEGORY_INFO[category]
        written = []

        proj_path = project_lib.registry_path(category)
        existing = project_lib.registry_rows(category) if proj_path.exists() else []
        if _find_row(existing, name) is None:
            fieldnames = self.system_lib.registry_fieldnames(category)
            _append_registry_row(proj_path, fieldnames, row)
            written.append(proj_path)

        if info["file_dir"]:
            for col in info["file_cols"]:
                fname = (row.get(col) or "").strip()
                if not fname:
                    continue
                src = self.system_lib.table_path(category, fname)
                if not src.exists():
                    continue
                dst_dir = project_lib.root / info["file_dir"]
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                    written.append(dst)
        return written

    def _ensure_registry_exists(self, category, txn):
        """materials.miemat and coatings.miecoat are hard-required by
        load_optical_properties even when empty; create a header-only file
        if the project doesn't have one yet."""
        project_lib = self._require_project()
        path = project_lib.registry_path(category)
        if path.exists():
            return
        txn.track(path)
        fieldnames = self.system_lib.registry_fieldnames(category)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    def ensure_project_library_selfcontained(self, used):
        """used: dict of category -> iterable of names (as returned by
        used_names_from_structure(), plus optionally "uniaxial"). Pulls in
        every named item AND its transitive dependencies (TMM coating layer
        materials; uniaxial o/e materials), then validates the resulting
        project library with load_optical_properties -- rolling every
        touched file back and raising LibraryWriteError if it still isn't
        self-contained.

        Returns the list of paths written (only on success)."""
        project_lib = self._require_project()
        used = {k: set(v) for k, v in (used or {}).items()}

        extra_materials = set()
        for cname in used.get("coatings", ()):
            row = self._system_row("coatings", cname)
            extra_materials |= _coating_layer_materials(row)
        for uname in used.get("uniaxial", ()):
            row = self._system_row("uniaxial", uname)
            extra_materials |= _uniaxial_materials(row)

        txn = Transaction()
        written = []
        try:
            all_materials = set(used.get("materials", ())) | extra_materials
            for mname in all_materials:
                row = self._system_row("materials", mname)
                proj_path = project_lib.registry_path("materials")
                txn.track(proj_path)
                nk = (row.get("nk_file") or "").strip()
                if nk:
                    src = self.system_lib.table_path("materials", nk)
                    if src.exists():
                        txn.track(project_lib.root / "nk" / src.name)
                written += self.ensure_project_item("materials", mname)

            for category in ("coatings", "polarizers", "filters", "gratings",
                             "uniaxial"):
                for iname in used.get(category, ()):
                    row = self._system_row(category, iname)
                    info = CATEGORY_INFO[category]
                    txn.track(project_lib.registry_path(category))
                    if info["file_dir"]:
                        for col in info["file_cols"]:
                            fname = (row.get(col) or "").strip()
                            if not fname:
                                continue
                            src = self.system_lib.table_path(category, fname)
                            if src.exists():
                                txn.track(project_lib.root
                                         / info["file_dir"] / src.name)
                    written += self.ensure_project_item(category, iname)

            self._ensure_registry_exists("materials", txn)
            self._ensure_registry_exists("coatings", txn)

            validate_and_commit(project_lib, txn)
        except Exception:
            txn.rollback()
            raise
        return written

    # -- project -> system ---------------------------------------------------
    def promote_to_system(self, category, name, force=False):
        """Copy `name`'s row (+ referenced file) from the project library
        into the system library.

        If the system already has a row named `name` with DIFFERENT
        content, returns {'system_row': {...}, 'project_row': {...}}
        WITHOUT writing anything -- the caller (UI) asks the user to
        confirm; pass force=True to overwrite anyway. On success (or when
        no conflict existed) returns the list of paths written, validated
        via PropLibrary.validate() with rollback on failure.
        """
        project_lib = self._require_project()
        info = CATEGORY_INFO[category]
        proj_row = _find_row(project_lib.registry_rows(category), name)
        if proj_row is None:
            raise LibraryError("%s: no such %s in the project library"
                               % (name, category))

        sys_rows = self.system_lib.registry_rows(category)
        sys_row = _find_row(sys_rows, name)
        if sys_row is not None and sys_row != proj_row and not force:
            return {"system_row": sys_row, "project_row": proj_row}

        txn = Transaction()
        written = []
        try:
            sys_path = self.system_lib.registry_path(category)
            txn.track(sys_path)
            fieldnames = self.system_lib.registry_fieldnames(category)
            rows = list(sys_rows)
            replaced = False
            for i, row in enumerate(rows):
                if (row.get("name") or "").strip().lower() \
                        == name.strip().lower():
                    rows[i] = proj_row
                    replaced = True
                    break
            if not replaced:
                rows.append(proj_row)
            _write_registry_rows(sys_path, fieldnames, rows)
            written.append(sys_path)

            if info["file_dir"]:
                for col in info["file_cols"]:
                    fname = (proj_row.get(col) or "").strip()
                    if not fname:
                        continue
                    src = project_lib.table_path(category, fname)
                    if not src.exists():
                        continue
                    dst_dir = self.system_lib.root / info["file_dir"]
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst = dst_dir / src.name
                    txn.track(dst)
                    shutil.copy2(src, dst)
                    written.append(dst)

            validate_and_commit(self.system_lib, txn)
        except Exception:
            txn.rollback()
            raise
        return written

    # -- primitives ----------------------------------------------------------
    def primitives_list(self):
        """Merged list of {kind, label, category, tooltip, path, params,
        props} for every primitives/*.FCStd -- sidecar .meta.json first,
        falling back to scripts.primitivelib.PRIMITIVES metadata (for a
        .FCStd shipped without one), and finally a bare 'User' entry (label
        = filename, no params) for a user-dropped .FCStd with neither."""
        out = []
        if not self.system_primitives.is_dir():
            return out
        for fcstd in sorted(self.system_primitives.glob("*.FCStd")):
            stem = fcstd.stem
            meta_path = fcstd.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path) as fh:
                    meta = json.load(fh)
                out.append({
                    "kind": meta.get("kind", stem),
                    "label": meta.get("label", stem),
                    "category": meta.get("category", "Other"),
                    "tooltip": meta.get("tooltip", ""),
                    "path": str(fcstd),
                    "params": meta.get("params", {}),
                    "props": meta.get("props", {}),
                })
            elif primitivelib is not None and stem in primitivelib.PRIMITIVES:
                spec = primitivelib.PRIMITIVES[stem]
                out.append({
                    "kind": stem,
                    "label": spec.get("label", stem),
                    "category": spec.get("category", "Other"),
                    "tooltip": spec.get("tooltip", ""),
                    "path": str(fcstd),
                    "params": {k: dict(v) for k, v in
                              spec.get("params", {}).items()},
                    "props": dict(spec.get("props", {})),
                })
            else:
                out.append({
                    "kind": stem,
                    "label": stem,
                    "category": "User",
                    "tooltip": "",
                    "path": str(fcstd),
                    "params": {},
                    "props": {},
                })
        return out
