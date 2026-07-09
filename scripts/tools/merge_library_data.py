#!/usr/bin/env python3
# =============================================================================
# merge_library_data.py -- merges the untracked library_data/ drop-in rows
# (see library_data/README.md and library.md Sec.7) into the live
# opticalproperties/ registries: materials.miemat, coating/coatings.miecoat,
# filter/filters.miefilt, polarizer/polarizers.miepol, grating/gratings.miegrat,
# birefringence/uniaxial.miebrf, plus the nk/ tables and per-item .mietab
# tables the appended rows reference.
#
# STATUS: this script has already been run once against the original
# library_data/ drop and the source files it consumed were then DELETED
# (moved out of library_data/, see library_data/README.md for what
# remains staged and why). It is kept in the tree to document the merge
# provenance and stays SAFE TO RE-RUN: for every source file it looks
# for, if the file is missing it prints "<file>: not found -- already
# merged (skipping)" and moves on rather than failing, and any row whose
# name already exists in the target registry with IDENTICAL content is
# skipped (never double-appended). A name collision with DIFFERENT
# content is treated as a hard pre-flight error -- the script never
# silently overwrites or guesses.
#
# stdlib-only (system python3, per CLAUDE.md's pinned-interpreter table).
#
# Run:  python3 scripts/tools/merge_library_data.py   (from anywhere)
# =============================================================================
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPT = ROOT / "opticalproperties"
LIB = ROOT / "library_data"
MANIFEST_PATH = ROOT / "demos" / "library_tests" / "new_items.json"

TABLE_LINE_ENDING = "\r\n"  # shipped .mietab convention (bs_5050_vis_45 etc.)


# ---------------------------------------------------------------------------
# generic CSV helpers
# ---------------------------------------------------------------------------
def _read_raw(path):
    with open(path, "r", newline="") as fh:
        return fh.read()


def detect_line_ending(path):
    data = open(path, "rb").read()
    return "\r\n" if b"\r\n" in data else "\n"


def target_header(path):
    with open(path, "r", newline="") as fh:
        first = fh.readline()
    return [c.strip() for c in first.rstrip("\r\n").split(",")]


def read_registry_repaired(path):
    """Read a registry CSV with csv.reader (not DictReader) so a row with
    MORE fields than the header (an unescaped comma inside the last
    column -- seen in materials_films.miemat's ta2o5/al2o3_film reference
    fields) can be repaired by folding the overflow back into the last
    column, instead of silently misaligning every subsequent field.
    Also strips any '#' comment lines (none currently in the registry
    files, but the merge contract says strip them everywhere).
    Returns (header, [ (name, row_dict, lineno) ])."""
    with open(path, newline="") as fh:
        lines = [l for l in fh if not l.lstrip().startswith("#")]
    reader = csv.reader(io.StringIO("".join(lines)))
    header = next(reader)
    n = len(header)
    out = []
    for i, raw in enumerate(reader, start=2):
        if not raw:
            continue
        if len(raw) > n:
            raw = raw[: n - 1] + [",".join(raw[n - 1 :])]
        elif len(raw) < n:
            raw = raw + [""] * (n - len(raw))
        row = dict(zip(header, raw))
        out.append((row.get("name", "").strip(), row, i))
    return header, out


def read_table_csv_stripped(path):
    """Consolidated *_tables.csv: strip '#' comment lines (csv.DictReader
    does not skip them), then DictReader on the real header row."""
    with open(path, newline="") as fh:
        lines = [l for l in fh if not l.lstrip().startswith("#")]
    return list(csv.DictReader(io.StringIO("".join(lines))))


def csv_line(fields, line_ending):
    buf = io.StringIO()
    csv.writer(buf, lineterminator=line_ending).writerow(fields)
    return buf.getvalue()


def write_table_content(header, rows):
    """rows: list of lists of raw string values, same order as header."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=TABLE_LINE_ENDING)
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# category configs
# ---------------------------------------------------------------------------
MATERIALS_TARGET = OPT / "materials.miemat"
MATERIALS_SOURCES = [
    LIB / "materials_glasses.miemat",
    LIB / "materials_metals_semiconductors_ir.miemat",
    LIB / "materials_polymers_liquids_gases_bio.miemat",
    LIB / "materials_films.miemat",
    LIB / "materials_crystals.miemat",
]
NK_SRC_DIR = LIB / "nk"
NK_DST_DIR = OPT / "nk"

SIMPLE_CATEGORIES = {
    # key: (target, source, table_key_col, table_ref_col, table_dir,
    #       table_header, source_table_csv, floor_transmittance)
    "coatings": dict(
        target=OPT / "coating" / "coatings.miecoat",
        source=LIB / "coatings.miecoat",
        table_key_col="coating",
        table_ref_col="table",
        table_dir=OPT / "coating" / "tables",
        table_header=["wavelength_nm", "Rs", "Rp", "Ts", "Tp"],
        source_table_csv=LIB / "coating_tables.csv",
        floor_transmittance=False,
    ),
    "filters": dict(
        target=OPT / "filter" / "filters.miefilt",
        source=LIB / "filters.miefilt",
        table_key_col="filter",
        table_ref_col="table_csv",
        table_dir=OPT / "filter" / "tables",
        table_header=["wavelength_nm", "transmittance_internal"],
        source_table_csv=LIB / "filter_tables.csv",
        floor_transmittance=True,
    ),
    "polarizers": dict(
        target=OPT / "polarizer" / "polarizers.miepol",
        source=LIB / "polarizers.miepol",
        table_key_col="polarizer",
        table_ref_col="table_csv",
        table_dir=OPT / "polarizer" / "tables",
        table_header=["wavelength_nm", "T_parallel", "T_perpendicular"],
        source_table_csv=LIB / "polarizer_tables.csv",
        floor_transmittance=False,
    ),
    "gratings": dict(
        target=OPT / "grating" / "gratings.miegrat",
        source=LIB / "gratings.miegrat",
        table_key_col="grating",
        table_ref_col="table_csv",
        table_dir=OPT / "grating" / "tables",
        table_header=["wavelength_nm", "order", "eta_s", "eta_p"],
        source_table_csv=LIB / "grating_tables.csv",
        floor_transmittance=False,
    ),
}

UNIAXIAL_TARGET = OPT / "birefringence" / "uniaxial.miebrf"
UNIAXIAL_SOURCE = LIB / "birefringence_uniaxial.miebrf"

# staged [needs-engine] files this script MUST NOT touch (consumed by other
# workstreams, or waiting on engine support not yet built):
SKIP_NEEDS_ENGINE = [
    "birefringence_biaxial.mibiax",
    "emission_emitters.miesrc",
    "emission_spectra.csv",
    "emission_led_monochromatic.csv",  # consumed by the LED-presets workstream
    "detector_vlambda.csv",
    "detector_qe.csv",
]


# ---------------------------------------------------------------------------
# plan container
# ---------------------------------------------------------------------------
class Plan:
    def __init__(self):
        self.errors = []
        self.notices = []
        # registry appends: category -> {"header", "line_ending",
        #   "target", "append_lines": [str,...], "appended_names": [...],
        #   "already_present": [...]}
        self.registry = {}
        # table files to write: path -> content (only if new/changed)
        self.table_writes = {}
        self.table_unchanged = []
        # nk copies: path -> bytes
        self.nk_writes = {}
        self.nk_unchanged = []
        self.dropped = []   # (source, name, reason)
        self.floored = []   # (table_name, wavelength_nm, old, new)


def err(plan, msg):
    plan.errors.append(msg)


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------
def plan_materials(plan):
    cat = "materials"
    if not MATERIALS_TARGET.exists():
        err(plan, "materials: target %s does not exist" % MATERIALS_TARGET)
        return
    live_header = target_header(MATERIALS_TARGET)
    _, live_rows = read_registry_repaired(MATERIALS_TARGET)
    live_by_name = {name.lower(): row for name, row, _ in live_rows}
    line_ending = detect_line_ending(MATERIALS_TARGET)

    all_incoming = []  # (source_name, name, row, lineno)
    any_present = False
    for src in MATERIALS_SOURCES:
        if not src.exists():
            plan.notices.append(
                "%s: not found -- already merged (skipping)" % src.name)
            continue
        any_present = True
        header = target_header(src)
        if header != live_header:
            err(plan, "materials: header mismatch: %s has %r, target %s "
                      "has %r" % (src.name, header, MATERIALS_TARGET.name,
                                  live_header))
            continue
        _, rows = read_registry_repaired(src)
        for name, row, lineno in rows:
            # transform: drop the known-intentional BAK4 duplicate of N-BAK4
            if src.name == "materials_glasses.miemat" and name == "BAK4":
                plan.dropped.append((src.name, name,
                                     "duplicate of N-BAK4 (see library_data/README.md)"))
                continue
            # transform: nk_file values in the source carry an "nk/" prefix;
            # the live convention (and MaterialDB.load's nk_dir join) is a
            # bare filename resolved under opticalproperties/nk/.
            nk = row.get("nk_file", "").strip()
            if nk.startswith("nk/"):
                row = dict(row)
                row["nk_file"] = nk[len("nk/") :]
            all_incoming.append((src.name, name, row, lineno))

    if not any_present:
        plan.notices.append("materials: no source files present -- already merged")
        return  # leave plan.registry["materials"] unset -> manifest preserves prior list

    if plan.errors:
        return

    # non-empty name/reference
    for src_name, name, row, lineno in all_incoming:
        if not name:
            err(plan, "materials: %s line %d: missing name" % (src_name, lineno))
        if not row.get("reference", "").strip():
            err(plan, "materials: %s line %d (%r): missing reference"
                      % (src_name, lineno, name))

    # sellmeier C>0 gate (materials.py:393-397)
    for src_name, name, row, lineno in all_incoming:
        if row.get("model", "").strip() == "sellmeier":
            for col in ("p4", "p5", "p6"):
                raw = row.get(col, "").strip()
                try:
                    v = float(raw)
                except ValueError:
                    err(plan, "materials: %s line %d (%r): sellmeier %s=%r "
                              "is not numeric" % (src_name, lineno, name, col, raw))
                    continue
                if not (v > 0):
                    err(plan, "materials: %s line %d (%r): sellmeier %s=%r "
                              "must be > 0" % (src_name, lineno, name, col, raw))

    # duplicate-name scan: among incoming, and vs live (identical content ->
    # idempotent skip; different content -> hard error)
    seen = {}
    to_append = []
    already_present = []
    for src_name, name, row, lineno in all_incoming:
        key = name.lower()
        if key in seen:
            err(plan, "materials: duplicate name %r among incoming rows "
                      "(%s line %d and %s)" % (name, src_name, lineno, seen[key]))
            continue
        seen[key] = "%s line %d" % (src_name, lineno)
        if key in live_by_name:
            live_row = live_by_name[key]
            if live_row == row:
                already_present.append(name)
            else:
                diff = [c for c in live_header if live_row.get(c, "") != row.get(c, "")]
                err(plan, "materials: name %r already exists in %s with "
                          "DIFFERENT content (fields differ: %s) -- refusing "
                          "to guess, resolve manually" % (name, MATERIALS_TARGET.name, diff))
        else:
            to_append.append((name, row))

    plan.registry[cat] = dict(
        header=live_header, line_ending=line_ending, target=MATERIALS_TARGET,
        append_lines=[csv_line([r.get(c, "") for c in live_header], line_ending)
                      for _, r in to_append],
        appended_names=[n for n, _ in to_append],
        already_present=already_present,
        # manifest order: source order (glasses, metals, polymers, films, crystals)
        all_names_in_order=[n for _, n, _, _ in all_incoming],
    )


# ---------------------------------------------------------------------------
# nk copies
# ---------------------------------------------------------------------------
def plan_nk(plan):
    if not NK_SRC_DIR.exists():
        plan.notices.append("nk/: source dir not found -- already merged (skipping)")
        return
    for src in sorted(NK_SRC_DIR.glob("*.mienk")):
        dst = NK_DST_DIR / src.name
        data = src.read_bytes()
        if dst.exists():
            if dst.read_bytes() == data:
                plan.nk_unchanged.append(src.name)
            else:
                err(plan, "nk/%s: target exists with DIFFERENT content -- "
                          "refusing to overwrite" % src.name)
        else:
            plan.nk_writes[dst] = data


# ---------------------------------------------------------------------------
# simple registries (coatings / filters / polarizers / gratings)
# ---------------------------------------------------------------------------
def plan_simple_registry(plan, cat, cfg):
    target = cfg["target"]
    source = cfg["source"]
    if not target.exists():
        err(plan, "%s: target %s does not exist" % (cat, target))
        return
    live_header = target_header(target)
    _, live_rows = read_registry_repaired(target)
    live_by_name = {name.lower(): row for name, row, _ in live_rows}
    line_ending = detect_line_ending(target)

    if not source.exists():
        plan.notices.append("%s: not found -- already merged (skipping)" % source.name)
        return

    src_header = target_header(source)
    if src_header != live_header:
        err(plan, "%s: header mismatch: %s has %r, target %s has %r"
                  % (cat, source.name, src_header, target.name, live_header))
        return

    _, rows = read_registry_repaired(source)

    for name, row, lineno in rows:
        if not name:
            err(plan, "%s: %s line %d: missing name" % (cat, source.name, lineno))
        if not row.get("reference", "").strip():
            err(plan, "%s: %s line %d (%r): missing reference"
                      % (cat, source.name, lineno, name))

    seen = {}
    to_append = []
    already_present = []
    for name, row, lineno in rows:
        key = name.lower()
        if key in seen:
            err(plan, "%s: duplicate name %r among incoming rows "
                      "(%s line %d and %s)" % (cat, name, source.name, lineno, seen[key]))
            continue
        seen[key] = "%s line %d" % (source.name, lineno)
        if key in live_by_name:
            live_row = live_by_name[key]
            if live_row == row:
                already_present.append(name)
            else:
                diff = [c for c in live_header if live_row.get(c, "") != row.get(c, "")]
                err(plan, "%s: name %r already exists in %s with DIFFERENT "
                          "content (fields differ: %s) -- refusing to guess, "
                          "resolve manually" % (cat, name, target.name, diff))
        else:
            to_append.append((name, row))

    plan.registry[cat] = dict(
        header=live_header, line_ending=line_ending, target=target,
        append_lines=[csv_line([r.get(c, "") for c in live_header], line_ending)
                      for _, r in to_append],
        appended_names=[n for n, _ in to_append],
        already_present=already_present,
        all_names_in_order=[n for n, _, _ in rows],
    )

    # ---- table split (if this category has one) ----
    src_table_csv = cfg["source_table_csv"]
    key_col = cfg["table_key_col"]
    ref_col = cfg["table_ref_col"]
    table_dir = cfg["table_dir"]
    table_header = cfg["table_header"]

    split_names = set()
    if src_table_csv.exists():
        table_rows = read_table_csv_stripped(src_table_csv)
        groups = {}
        for r in table_rows:
            item = r[key_col].strip()
            values = []
            for col in table_header:
                v = r[col]
                if cfg["floor_transmittance"] and col == "transmittance_internal":
                    fv = float(v)
                    if fv <= 0:
                        plan.floored.append((item, r["wavelength_nm"], v, "1e-6"))
                        v = "1e-6"
                values.append(v)
            groups.setdefault(item, []).append(values)
        for item, data_rows in groups.items():
            split_names.add(item)
            out_path = table_dir / (item + ".mietab")
            content = write_table_content(table_header, data_rows)
            if out_path.exists():
                if _read_raw(out_path) == content:
                    plan.table_unchanged.append(out_path.name)
                else:
                    err(plan, "%s: table %s exists with DIFFERENT content -- "
                              "refusing to overwrite" % (cat, out_path.name))
            else:
                plan.table_writes[out_path] = content
    else:
        plan.notices.append(
            "%s: not found -- already merged (skipping)" % src_table_csv.name)

    # ---- verify every table/table_csv referenced by an incoming row exists ----
    existing_table_files = {p.name for p in table_dir.glob("*.mietab")} if table_dir.exists() else set()
    for name, row, lineno in rows:
        tref = row.get(ref_col, "").strip()
        if not tref:
            continue
        stem = tref[: -len(".mietab")] if tref.endswith(".mietab") else tref
        if stem in split_names or tref in existing_table_files:
            continue
        err(plan, "%s: %s line %d (%r): references table %r which is not "
                  "in the split output and does not already exist in %s"
                  % (cat, source.name, lineno, name, tref, table_dir))


# ---------------------------------------------------------------------------
# uniaxial birefringence (no table split)
# ---------------------------------------------------------------------------
def plan_uniaxial(plan):
    cat = "uniaxial"
    target = UNIAXIAL_TARGET
    source = UNIAXIAL_SOURCE
    if not target.exists():
        err(plan, "uniaxial: target %s does not exist" % target)
        return
    live_header = target_header(target)
    _, live_rows = read_registry_repaired(target)
    live_by_name = {name.lower(): row for name, row, _ in live_rows}
    line_ending = detect_line_ending(target)

    if not source.exists():
        plan.notices.append("%s: not found -- already merged (skipping)" % source.name)
        return

    src_header = target_header(source)
    if src_header != live_header:
        err(plan, "uniaxial: header mismatch: %s has %r, target %s has %r"
                  % (source.name, src_header, target.name, live_header))
        return

    _, rows = read_registry_repaired(source)
    for name, row, lineno in rows:
        if not name:
            err(plan, "uniaxial: %s line %d: missing name" % (source.name, lineno))
        if not row.get("reference", "").strip():
            err(plan, "uniaxial: %s line %d (%r): missing reference"
                      % (source.name, lineno, name))

    seen = {}
    to_append = []
    already_present = []
    for name, row, lineno in rows:
        key = name.lower()
        if key in seen:
            err(plan, "uniaxial: duplicate name %r among incoming rows "
                      "(%s line %d and %s)" % (name, source.name, lineno, seen[key]))
            continue
        seen[key] = "%s line %d" % (source.name, lineno)
        if key in live_by_name:
            live_row = live_by_name[key]
            if live_row == row:
                already_present.append(name)
            else:
                diff = [c for c in live_header if live_row.get(c, "") != row.get(c, "")]
                err(plan, "uniaxial: name %r already exists in %s with "
                          "DIFFERENT content (fields differ: %s) -- refusing "
                          "to guess, resolve manually" % (name, target.name, diff))
        else:
            to_append.append((name, row))

    plan.registry[cat] = dict(
        header=live_header, line_ending=line_ending, target=target,
        append_lines=[csv_line([r.get(c, "") for c in live_header], line_ending)
                      for _, r in to_append],
        appended_names=[n for n, _ in to_append],
        already_present=already_present,
        all_names_in_order=[n for n, _, _ in rows],
    )


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def build_manifest(plan):
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text())
        except Exception:
            manifest = {}
    else:
        manifest = {}
    for key in ("materials", "coatings", "filters", "polarizers", "gratings", "uniaxial"):
        manifest.setdefault(key, [])
    if "materials" in plan.registry:
        manifest["materials"] = plan.registry["materials"]["all_names_in_order"]
    for cat in ("coatings", "filters", "polarizers", "gratings"):
        if cat in plan.registry:
            manifest[cat] = plan.registry[cat]["all_names_in_order"]
    if "uniaxial" in plan.registry:
        manifest["uniaxial"] = plan.registry["uniaxial"]["all_names_in_order"]
    return manifest


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------
def execute(plan, manifest):
    for cat, info in plan.registry.items():
        if info["append_lines"]:
            with open(info["target"], "a", newline="") as fh:
                fh.writelines(info["append_lines"])

    for path, content in plan.table_writes.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            fh.write(content)

    for path, data in plan.nk_writes.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    # defensive: none of the staged [needs-engine] files should ever be
    # among the sources this script reads (they're a different workstream's
    # or a future loader's responsibility -- see library_data/README.md).
    all_sources = MATERIALS_SOURCES + [UNIAXIAL_SOURCE] + [
        cfg["source"] for cfg in SIMPLE_CATEGORIES.values()
    ] + [cfg["source_table_csv"] for cfg in SIMPLE_CATEGORIES.values()]
    touched = {p.name for p in all_sources} & set(SKIP_NEEDS_ENGINE)
    assert not touched, "merge script must never touch staged files: %s" % touched

    plan = Plan()
    plan_materials(plan)
    plan_nk(plan)
    for cat, cfg in SIMPLE_CATEGORIES.items():
        plan_simple_registry(plan, cat, cfg)
    plan_uniaxial(plan)

    for n in plan.notices:
        print(n)

    if plan.errors:
        print("\nPRE-FLIGHT FAILED -- nothing written. Problems:", file=sys.stderr)
        for e in plan.errors:
            print("  - %s" % e, file=sys.stderr)
        sys.exit(1)

    manifest = build_manifest(plan)
    execute(plan, manifest)

    print()
    if plan.dropped:
        print("dropped rows:")
        for src, name, reason in plan.dropped:
            print("  %s: %s (%s)" % (src, name, reason))
    if plan.floored:
        print("floored transmittance_internal <= 0 -> 1e-6: %d value(s)" % len(plan.floored))
        for item, lam, old, new in plan.floored:
            print("  %s @ %snm: %s -> %s" % (item, lam, old, new))

    print()
    for cat in ("materials", "coatings", "filters", "polarizers", "gratings", "uniaxial"):
        if cat not in plan.registry:
            print("%-10s: source missing, skipped" % cat)
            continue
        info = plan.registry[cat]
        total_rows = sum(1 for _ in open(info["target"], newline="")) - 1
        print("%-10s: appended %d, already present %d -> target now has %d rows"
              % (cat, len(info["appended_names"]), len(info["already_present"]),
                 total_rows))

    if plan.nk_writes or plan.nk_unchanged:
        print("\nnk/: copied %d new file(s), %d already identical"
              % (len(plan.nk_writes), len(plan.nk_unchanged)))

    if plan.table_writes or plan.table_unchanged:
        print("tables: wrote %d new file(s), %d already identical"
              % (len(plan.table_writes), len(plan.table_unchanged)))

    print("\nmanifest written: %s" % MANIFEST_PATH)


if __name__ == "__main__":
    main()
