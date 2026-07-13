#!/usr/bin/env python3
# =============================================================================
# verify_miemat_preserved.py -- guardrail check for the AGF glass-catalog
# import (scripts/tools/import_agf.py --merge-into): proves that every row
# present in the OLD opticalproperties/materials.miemat (as of `git show
# HEAD:...` -- i.e. before the import touched the file) still has
# byte-identical values in its original 15 columns
# (name,class,model,p1..p6,nk_file,density_kg_m3,transmission_um_min,
# transmission_um_max,notes,reference) in the NEW (working-tree) file. The
# only permitted changes are (a) 7 new thermo_* columns appended to the
# header and to every row, and (b) those 7 cells being non-empty on rows
# the importer could confidently backfill.
#
# stdlib-only. Exit code 0 = guardrail passed, 1 = a preserved cell was
# corrupted (details printed), 2 = usage/setup error.
#
# Run:  python3 scripts/tools/verify_miemat_preserved.py
#       python3 scripts/tools/verify_miemat_preserved.py --old-ref HEAD~3
# =============================================================================
import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "opticalproperties" / "materials.miemat"

ORIGINAL_COLS = ["name", "class", "model", "p1", "p2", "p3", "p4", "p5",
                  "p6", "nk_file", "density_kg_m3", "transmission_um_min",
                  "transmission_um_max", "notes", "reference"]


def load_csv_text(text):
    reader = csv.DictReader(io.StringIO(text))
    return reader.fieldnames, list(reader)


def git_show(ref, path):
    rel = path.relative_to(ROOT)
    out = subprocess.run(["git", "show", "%s:%s" % (ref, rel)],
                          cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        print("git show %s:%s failed:\n%s" % (ref, rel, out.stderr),
              file=sys.stderr)
        sys.exit(2)
    return out.stdout


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-ref", default="HEAD",
                     help="git ref to diff the CURRENT working-tree file "
                          "against (default: %(default)s)")
    ap.add_argument("--target", default=str(TARGET))
    args = ap.parse_args(argv)

    target_path = Path(args.target)
    old_text = git_show(args.old_ref, target_path)
    new_text = target_path.read_text()

    old_fields, old_rows = load_csv_text(old_text)
    new_fields, new_rows = load_csv_text(new_text)

    missing_old_cols = set(ORIGINAL_COLS) - set(old_fields or [])
    if missing_old_cols:
        print("FAIL: old file is missing expected column(s) %s -- wrong "
              "--old-ref?" % sorted(missing_old_cols))
        return 2
    missing_new_cols = set(ORIGINAL_COLS) - set(new_fields or [])
    if missing_new_cols:
        print("FAIL: new file is missing original column(s) %s"
              % sorted(missing_new_cols))
        return 1

    thermo_cols = ["thermo_d0", "thermo_d1", "thermo_d2", "thermo_e0",
                   "thermo_e1", "thermo_lambda_tk", "thermo_t_ref_c"]
    missing_thermo = [c for c in thermo_cols if c not in (new_fields or [])]
    if missing_thermo:
        print("FAIL: new file is missing thermo column(s) %s "
              "(expected the AGF import to have added them)"
              % missing_thermo)
        return 1

    new_by_name = {}
    dupe_names = []
    for r in new_rows:
        n = r["name"]
        if n in new_by_name:
            dupe_names.append(n)
        new_by_name[n] = r
    if dupe_names:
        print("FAIL: new file has duplicate name(s): %s" % dupe_names)
        return 1

    errors = []
    missing_names = []
    for old_row in old_rows:
        name = old_row["name"]
        new_row = new_by_name.get(name)
        if new_row is None:
            missing_names.append(name)
            continue
        for col in ORIGINAL_COLS:
            ov = (old_row.get(col) or "")
            nv = (new_row.get(col) or "")
            if ov != nv:
                errors.append((name, col, ov, nv))

    print("Old file: %d rows, %d original columns" % (len(old_rows), len(ORIGINAL_COLS)))
    print("New file: %d rows, %d columns (%d original + %d thermo)"
          % (len(new_rows), len(new_fields), len(ORIGINAL_COLS), len(thermo_cols)))
    print("New rows added: %d" % (len(new_rows) - len(old_rows)))

    if missing_names:
        print("FAIL: %d old row(s) missing entirely from the new file: %s"
              % (len(missing_names), missing_names))

    if errors:
        print("FAIL: %d cell(s) changed in %d row(s) that must be "
              "byte-preserved:" % (len(errors), len({e[0] for e in errors})))
        for name, col, ov, nv in errors[:50]:
            print("  %-20s %-14s OLD=%r  NEW=%r" % (name, col, ov, nv))
        if len(errors) > 50:
            print("  ... and %d more" % (len(errors) - 50))

    if missing_names or errors:
        return 1

    thermo_filled = sum(1 for r in new_rows if r.get("thermo_d0"))
    print("PASS: all %d original rows are preserved byte-identically in "
          "their 15 original columns." % len(old_rows))
    print("      %d rows (old+new) now carry thermo-optic data."
          % thermo_filled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
