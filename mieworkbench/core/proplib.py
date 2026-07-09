"""proplib.py -- cache-aware wrapper around one opticalproperties/ library
root for the MieWorkbench GUI.

This module never reimplements the validation rules that live in
scripts/raytracer/optprops.py + scripts/raytracer/materials.py -- those two
modules ARE the oracle: load_optical_properties(root) either succeeds (the
library is trace-ready) or raises MaterialError naming the offending row.
Everything here either calls straight through to that loader or reads the
registry/table CSVs directly with csv.DictReader for DISPLAY purposes (the
loader's return objects intentionally drop columns like nk_file/table_csv
paths once they're resolved into arrays, so the editor pane needs its own
raw read for those).

CATEGORY_INFO is the one place that knows the on-disk layout for each of
the six registries: where the registry csv lives (registry_rel, resolved
new-extension-first / legacy-.csv-fallback exactly like the loaders), where
its rows' referenced table/nk files live (file_dir, relative to root) and
which registry column(s) name those files (file_cols), plus the per-item
file's own self-describing/".csv" fallback extension (file_alt_ext).
LibraryManager (librarymgr.py) reuses this table when copying items between
the system and project libraries, so the two modules never disagree about
where a file belongs.
"""
import csv
import shutil
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from raytracer.optprops import load_optical_properties        # noqa: E402
from raytracer.materials import MaterialError, resolve_prop_path  # noqa: E402

CATEGORIES = ("materials", "coatings", "polarizers", "filters", "gratings",
              "uniaxial", "diffusers", "detectors")

# subdir=None means the registry lives directly at <root>/<filename>.
CATEGORY_INFO = {
    "materials": {
        "registry_rel": "materials.miemat",
        "file_dir": "nk",
        "file_cols": ("nk_file",),
        "file_alt_ext": ".mienk",
    },
    "coatings": {
        "registry_rel": "coating/coatings.miecoat",
        "file_dir": "coating/tables",
        "file_cols": ("table",),
        "file_alt_ext": ".mietab",
    },
    "polarizers": {
        "registry_rel": "polarizer/polarizers.miepol",
        "file_dir": "polarizer/tables",
        "file_cols": ("table_csv",),
        "file_alt_ext": ".mietab",
    },
    "filters": {
        "registry_rel": "filter/filters.miefilt",
        "file_dir": "filter/tables",
        "file_cols": ("table_csv",),
        "file_alt_ext": ".mietab",
    },
    "gratings": {
        "registry_rel": "grating/gratings.miegrat",
        "file_dir": "grating/tables",
        "file_cols": ("table_csv",),
        "file_alt_ext": ".mietab",
    },
    "uniaxial": {
        "registry_rel": "birefringence/uniaxial.miebrf",
        "file_dir": None,           # rows reference OTHER materials, not files
        "file_cols": (),
        "file_alt_ext": None,
    },
    "diffusers": {
        "registry_rel": "diffuser/diffusers.miedif",
        "file_dir": None,           # rows are self-contained (grit or slope)
        "file_cols": (),
        "file_alt_ext": None,
    },
    "detectors": {
        "registry_rel": "detector/detectors.miedet",
        "file_dir": "detector/tables",
        "file_cols": ("table_csv",),
        "file_alt_ext": ".mietab",
    },
}


def _check_category(category):
    if category not in CATEGORY_INFO:
        raise KeyError("unknown optical-property category %r (must be one "
                       "of %s)" % (category, ", ".join(CATEGORIES)))


class PropLibraryError(RuntimeError):
    """Raised for GUI-level library-layer mistakes (bad category name,
    missing table row, ...). Loader validation failures surface as
    MaterialError (re-exported here) or via validate()'s (ok, msg) tuple."""


class PropLibrary:
    """Thin, cache-aware wrapper around one opticalproperties/ root.

    `root` need not exist yet (a not-yet-populated project library):
    registry_rows()/table_data() simply raise/return empty as appropriate,
    and validate() reports the loader's "not found" error.
    """

    def __init__(self, root):
        self.root = Path(root)
        self._props = None

    # -- loading --------------------------------------------------------
    def load(self):
        """The real OpticalProperties object (scripts.raytracer.optprops),
        cached until reload()."""
        if self._props is None:
            self._props = load_optical_properties(self.root)
        return self._props

    def reload(self):
        """Invalidate the cache; the next load() re-reads from disk."""
        self._props = None

    def validate(self):
        """(True, "") if load_optical_properties(root) succeeds right now
        (fresh read, bypassing the cache); else (False, error_message).
        Never raises."""
        try:
            load_optical_properties(self.root)
        except MaterialError as exc:
            return False, str(exc)
        except Exception as exc:                      # pragma: no cover
            return False, "%s: %s" % (type(exc).__name__, exc)
        return True, ""

    # -- category summaries ----------------------------------------------
    def categories(self):
        """{"materials": [names], "coatings": [...], "polarizers": [...],
        "filters": [...], "gratings": [...], "uniaxial": [...]} -- names in
        registry order, from the loaded (validated) library."""
        props = self.load()
        return {
            "materials": list(props.matdb.used_names()),
            "coatings": list(props.coatings.keys()),
            "polarizers": list(props.polarizers.keys()),
            "filters": list(props.filters.keys()),
            "gratings": list(props.gratings.keys()),
            "uniaxial": list(props.uniaxial.keys()),
            "diffusers": list(getattr(props, "diffusers", {}) or {}),
            "detectors": list(getattr(props, "detectors", {}) or {}),
        }

    def material_names(self):
        """Convenience for other panes: material names in registry order."""
        return list(self.load().matdb.used_names())

    # -- raw registry / table access (for display + editing) -------------
    def registry_path(self, category):
        """Existing path to the category's registry csv (new extension
        preferred, legacy same-stem .csv sibling as a fallback), or the
        preferred (new-extension) path unchanged if neither exists yet --
        i.e. always a valid *target* path for a not-yet-created registry."""
        _check_category(category)
        return resolve_prop_path(self.root / CATEGORY_INFO[category]["registry_rel"])

    def registry_fieldnames(self, category):
        """Column header of the category's registry csv."""
        path = self.registry_path(category)
        if not path.exists():
            raise PropLibraryError("%s registry not found: %s"
                                   % (category, path))
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh).fieldnames or [])

    def registry_rows(self, category):
        """List of dict rows (csv.DictReader) for the category's registry
        -- the raw csv content, not the loader's parsed/validated form (the
        loader drops columns once resolved, e.g. table_csv/nk_file paths)."""
        path = self.registry_path(category)
        if not path.exists():
            return []
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))

    def table_path(self, category, filename):
        """Resolve a per-row referenced file (nk_file / table / table_csv
        column value) against the category's file_dir, with the same
        exact-name-first / alt-extension-fallback rule the loaders use."""
        _check_category(category)
        info = CATEGORY_INFO[category]
        if not info["file_dir"]:
            raise PropLibraryError(
                "category %r rows do not reference files" % category)
        path = self.root / info["file_dir"] / filename
        return resolve_prop_path(path, alt_ext=info["file_alt_ext"])

    def table_data(self, category, filename):
        """(headers, rows) for a per-row table file: headers is the csv
        column list in file order; rows is a list of tuples of floats, one
        per data row, for QtCharts plotting (first column is normally
        wavelength_nm)."""
        path = self.table_path(category, filename)
        if not path.exists():
            raise PropLibraryError("table file not found: %s" % path)
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            headers = list(reader.fieldnames or [])
            rows = []
            for r in reader:
                rows.append(tuple(float(r[h]) for h in headers))
        return headers, rows


# ---------------------------------------------------------------------------
# Shared write-transaction helper (used by librarymgr.py and prop_editor.py):
# stage file writes, then validate a PropLibrary against the result; roll
# every touched file back to its pre-transaction state on failure.
# ---------------------------------------------------------------------------
class LibraryWriteError(RuntimeError):
    """A post-write PropLibrary.validate() failed; by the time this is
    raised the touched files have already been rolled back."""


class Transaction:
    """track(path) BEFORE writing to path (captures the pre-write state);
    commit() discards backups, rollback() restores/removes touched files."""

    def __init__(self):
        self._entries = []   # (path, existed_before, backup_path)
        self._tracked = set()

    def track(self, path):
        path = Path(path)
        if path in self._tracked:
            return
        self._tracked.add(path)
        existed = path.exists()
        backup = None
        if existed:
            backup = path.with_name(path.name + ".mieworkbench.bak")
            shutil.copy2(path, backup)
        self._entries.append((path, existed, backup))

    def commit(self):
        for _, _, backup in self._entries:
            if backup is not None:
                try:
                    backup.unlink()
                except OSError:
                    pass
        self._entries = []
        self._tracked = set()

    def rollback(self):
        for path, existed, backup in self._entries:
            try:
                if existed:
                    shutil.copy2(backup, path)
                    backup.unlink()
                elif path.exists():
                    path.unlink()
            except OSError:
                pass
        self._entries = []
        self._tracked = set()


def validate_and_commit(proplib, txn):
    """Validate `proplib` (a PropLibrary) after txn's writes. Commits and
    invalidates the library's cache on success; rolls back and raises
    LibraryWriteError(message) on failure."""
    ok, msg = proplib.validate()
    if not ok:
        txn.rollback()
        raise LibraryWriteError(msg)
    txn.commit()
    proplib.reload()
