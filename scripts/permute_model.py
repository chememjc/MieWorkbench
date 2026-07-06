#!/usr/bin/env python3
# =============================================================================
# permute_model.py — FreeCAD headless parameter permutation for the optical
# ray tracer pipeline.
#
# Reads  : a single .FCStd model with a Spreadsheet::Sheet object (label
#          "dim") driving geometry via aliased cells (see extract_geometry.py
#          for the sibling convention).
# Writes : basemodels/<stem>-<var1><v1>-<var2><v2>-....FCStd — one file per
#          combination in the cross-product of all swept --var values.
#
# Run with the FreeCAD AppImage (its bundled Python has the FreeCAD modules).
# Custom args MUST be preceded by a bare "--" so FreeCAD's own arg parser
# hands the remainder to sys.argv untouched:
#
#   /home3/freecad/FreeCAD.AppImage -c scripts/permute_model.py -- \
#       --model example.FCStd \
#       --var lenspos --min -5 --max 5 --n 2 \
#       --var sphered --min 20 --max 40 --n 0 \
#       [--outdir basemodels] [--unit mm] < /dev/null
#
# --var/--min/--max/--n use argparse action="append": the Nth --var pairs
# with the Nth --min/--max/--n (validated to have equal counts). The output
# set is the cross-product of common.sweep_values() for each var, in the
# order the --var flags were given.
#
# NOTE on double execution: this AppImage build's -c mode runs the whole
# script TWICE per invocation (silent headless pass, then a GUI-spinup
# pass). All work here is idempotent (each output .FCStd is simply
# overwritten with identical content on the second pass), so this is
# harmless but doubles wall-clock time and log lines.
#
# Value semantics (common.sweep_values, project-wide law):
#   n=0   -> values = [min]                                (max ignored)
#   n=1   -> values = [min, max]
#   n>1   -> n+1 evenly spaced values: min + i*(max-min)/n for i in 0..n
#
# Variant naming: common.variant_name() applied once per swept variable, in
# --var order, chaining the previous name as the next stem, e.g.
#   example -> example-lenspos2 -> example-lenspos2-sphered30
# =============================================================================
import argparse
import itertools
import os
import sys
from pathlib import Path

import FreeCAD

# scripts/ dir on sys.path so "import common" works when run via the
# FreeCAD AppImage (which does not automatically add the script's own
# directory the way plain `python3 scripts/foo.py` does).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common   # noqa: E402

SPREADSHEET_LABEL = "dim"   # object label; internal name is usually "Spreadsheet"

# Overwriting an existing output file is expected on every rerun (idempotent
# writes); disable FreeCAD's "keep a .FCBak of the file I'm replacing"
# preference so reruns don't silently accumulate backup clutter in outdir.
FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
    "CreateBackupFiles", False)


def log(msg):
    # print() alone has been observed to buffer/drop under the AppImage
    # console in some invocations; use both PrintMessage and print.
    FreeCAD.Console.PrintMessage(msg + "\n")
    print(msg, flush=True)


def warn(msg):
    FreeCAD.Console.PrintWarning("WARNING: " + msg + "\n")
    print("WARNING: " + msg, flush=True)


def die(msg):
    FreeCAD.Console.PrintError("ERROR: %s\n" % msg)
    print("ERROR: %s" % msg, flush=True)
    os._exit(1)   # sys.exit() is swallowed under FreeCAD -c; force a real exit.


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    argv = sys.argv
    rest = argv[argv.index("--") + 1:] if "--" in argv else []

    p = argparse.ArgumentParser(
        prog="permute_model.py",
        description="Sweep one or more spreadsheet-aliased parameters across "
                     "a FreeCAD model (cross-product of all swept values).")
    p.add_argument("--model", required=True,
                    help="input .FCStd filename (bare name resolved under the "
                         "project root, or an absolute/relative path)")
    p.add_argument("--var", action="append", required=True,
                    help="spreadsheet cell alias to sweep (repeatable)")
    p.add_argument("--min", action="append", type=float, required=True,
                    help="minimum value (repeatable, paired with --var in order)")
    p.add_argument("--max", action="append", type=float, required=True,
                    help="maximum value (repeatable, paired with --var in order)")
    p.add_argument("--n", action="append", type=int, required=True,
                    help="0 -> [min] only; 1 -> [min, max]; >1 -> n+1 evenly "
                         "spaced values (repeatable, paired with --var in order)")
    p.add_argument("--outdir", default=None,
                    help="output directory (default: %s)" % common.BASEMODELS_DIR)
    p.add_argument("--unit", default="mm", help="dimension unit (default: %(default)s)")
    try:
        args = p.parse_args(rest)
    except SystemExit:
        os._exit(2)

    counts = {"--var": len(args.var), "--min": len(args.min),
              "--max": len(args.max), "--n": len(args.n)}
    if len(set(counts.values())) != 1:
        die("--var/--min/--max/--n must appear the same number of times "
            "(got %s)" % counts)
    return args


def resolve_model_path(model):
    p = Path(model)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    cand = common.PROJECT_DIR / p
    return cand


def find_sheet(doc):
    """Find the spreadsheet with Label == SPREADSHEET_LABEL, falling back to
    any Spreadsheet::Sheet object; hard error if none exists at all."""
    sheets = [o for o in doc.Objects if o.TypeId == "Spreadsheet::Sheet"]
    if not sheets:
        die("%s: document has no Spreadsheet::Sheet object" % doc.Name)
    for s in sheets:
        if s.Label == SPREADSHEET_LABEL:
            return s
    warn("%s: no spreadsheet labeled '%s'; falling back to '%s'"
         % (doc.Name, SPREADSHEET_LABEL, sheets[0].Label))
    return sheets[0]


def resolve_alias_cell(sheet, var):
    """Return the cell address for alias `var`, or hard-error listing the
    aliases that do exist on the sheet."""
    try:
        cell = sheet.getCellFromAlias(var)
    except Exception as exc:
        cell, err = None, exc
    else:
        err = None
    if cell:
        return cell

    aliases = []
    try:
        for c in sheet.getUsedCells():
            a = sheet.getAlias(c)
            if a:
                aliases.append(a)
        aliases = sorted(set(aliases))
    except Exception:
        pass
    die("alias '%s' not found on spreadsheet '%s'%s (available aliases: %s)"
        % (var, sheet.Label,
           (": %s" % err) if err is not None else "",
           ", ".join(aliases) if aliases else "<none found>"))


def check_recompute(doc):
    """Warn (do not fail) if any object is left Invalid/Error after recompute."""
    bad = []
    for obj in doc.Objects:
        state = getattr(obj, "State", None) or []
        if any(("Invalid" in s or "Error" in s) for s in state):
            bad.append("%s (%s)" % (obj.Name, ", ".join(state)))
    if bad:
        warn("recompute left %d object(s) in an error state: %s"
             % (len(bad), "; ".join(bad)))


def permute(model_path, varspecs, outdir, unit):
    """varspecs: list of (var, vmin, vmax, n) in --var order.

    Writes one .FCStd per combination in the cross-product of
    common.sweep_values() for each varspec.
    """
    stem = model_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    value_lists = [common.sweep_values(vmin, vmax, n) for (_, vmin, vmax, n) in varspecs]
    names = [v[0] for v in varspecs]
    total = 1
    for vl in value_lists:
        total *= len(vl)
    log("permute_model.py: %s  vars=%s  -> %d combination(s) -> %s"
        % (model_path.name, names, total, outdir))

    n_written = 0
    for combo in itertools.product(*value_lists):
        out_name = stem
        for var, value in zip(names, combo):
            out_name = common.variant_name(out_name, var, value)
        out_path = outdir / ("%s.FCStd" % out_name)

        doc = FreeCAD.openDocument(str(model_path))
        try:
            sheet = find_sheet(doc)
            for var, value in zip(names, combo):
                cell = resolve_alias_cell(sheet, var)
                sheet.set(cell, "=%.10g %s" % (value, unit))
            doc.recompute()
            check_recompute(doc)

            doc.saveAs(str(out_path))
            log("wrote %s (%s)"
                % (out_path, ", ".join("%s=%.10g %s" % (v, x, unit)
                                        for v, x in zip(names, combo))))
            n_written += 1
        finally:
            FreeCAD.closeDocument(doc.Name)

    return n_written


def main():
    args = parse_args()

    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        die("file not found: %s" % model_path)

    varspecs = list(zip(args.var, args.min, args.max, args.n))
    for var, vmin, vmax, n in varspecs:
        if n < 0:
            die("--n must be >= 0 (got %d for --var %s)" % (n, var))

    outdir = Path(args.outdir) if args.outdir else common.BASEMODELS_DIR

    n_written = permute(model_path, varspecs, outdir, args.unit)

    log("PERMUTE MODEL OK (%d file(s) written)" % n_written)


# NOTE: no `if __name__ == "__main__"` guard — FreeCAD's console mode (-c)
# executes scripts with __name__ set to the module's basename, not
# "__main__", which would silently skip main() if guarded.
main()
sys.exit(0)
