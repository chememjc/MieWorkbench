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
# --sweep-mode (common.sweep_combos, project-wide law):
#   product (default) -> cross-product of every swept variable's values
#   zip                -> variables advance together, one variant per
#                         index (value lists must have equal length, or
#                         length 1 to broadcast)
#
# Variant naming: common.variant_name() applied once per swept variable, in
# --var order, chaining the previous name as the next stem, e.g.
#   example -> example-lenspos2 -> example-lenspos2-sphered30
# =============================================================================
import argparse
import os
import sys
from pathlib import Path

import FreeCAD

# scripts/ dir on sys.path so "import common" works when run via the
# FreeCAD AppImage (which does not automatically add the script's own
# directory the way plain `python3 scripts/foo.py` does).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common   # noqa: E402
import train_fcstd   # noqa: E402

SPREADSHEET_LABEL = "dim"   # object label; internal name is usually "Spreadsheet"

# Overwriting an existing output file is expected on every rerun (idempotent
# writes); disable FreeCAD's "keep a .FCBak of the file I'm replacing"
# preference so reruns don't silently accumulate backup clutter in outdir.
FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
    "CreateBackupFiles", False)


class PermuteError(ValueError):
    """A fatal permutation error (die()).

    Raised instead of exiting the process so this module is importable as
    a library: scripts/fcserver/fcops.py applies parameter assignments to
    an already-open document (the fast evaluator's persistent worker),
    which must survive a bad alias/sheet name. The CLI entry point at the
    bottom of this file catches it and preserves the historical exit(1)."""


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
    raise PermuteError(msg)


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
    p.add_argument("--sweep-mode", default="product",
                    choices=["product", "zip"],
                    help="how multiple --var combinations combine: "
                         "'product' = cartesian (default); 'zip' = "
                         "variables advance together (equal-length value "
                         "lists, or length 1 to broadcast) — see "
                         "common.sweep_combos")
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


def find_sheet(doc, label=None):
    """Find the spreadsheet with the given Label (default SPREADSHEET_LABEL),
    falling back to any Spreadsheet::Sheet object when no label was asked
    for explicitly; hard error otherwise."""
    sheets = [o for o in doc.Objects if o.TypeId == "Spreadsheet::Sheet"]
    if not sheets:
        die("%s: document has no Spreadsheet::Sheet object" % doc.Name)
    want = label or SPREADSHEET_LABEL
    for s in sheets:
        if s.Label == want or s.Name == want:
            return s
    if label is not None:
        die("%s: no spreadsheet labeled '%s' (have: %s)"
            % (doc.Name, label, ", ".join(s.Label for s in sheets)))
    warn("%s: no spreadsheet labeled '%s'; falling back to '%s'"
         % (doc.Name, SPREADSHEET_LABEL, sheets[0].Label))
    return sheets[0]


def split_var(var):
    """'alias' -> (None, 'alias'); 'sheetlabel.alias' -> ('sheetlabel',
    'alias'). Sheet-qualified names address per-element parameter sheets
    (dim_<element>) written by MieWorkbench primitives."""
    if "." in var:
        sheet_label, _, alias = var.partition(".")
        return sheet_label, alias
    return None, var


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


def rebuild_primitive_groups(doc, touched_sheets):
    """MieWorkbench primitive elements (bodies tagged miewb_primitive /
    miewb_group) bake their geometry from their dim_<group> sheet instead
    of driving it through cell expressions, so an alias edit alone leaves
    the shape unchanged. Rebuild every primitive group whose sheet was
    touched (no-op for ordinary expression-driven models)."""
    groups = set()
    for sheet in touched_sheets:
        if sheet.Label.startswith("dim_"):
            groups.add(sheet.Label[len("dim_"):])
    if not groups:
        return
    import primitivelib
    for group in sorted(groups):
        bodies = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"
                  and getattr(o, "miewb_group", None) == group]
        if not bodies:
            continue
        kind = getattr(bodies[0], "miewb_primitive", None)
        if kind not in primitivelib.PRIMITIVES:
            continue
        sheet = [s for s in touched_sheets
                 if s.Label == "dim_%s" % group][0]
        primitivelib.rebuild_element(doc, sheet, kind, group)
        log("rebuilt primitive group '%s' (%s)" % (group, kind))


def extend_touched_for_miewb_vars(doc, touched_sheets):
    """KNOWN BUG fix: a dim_<El> cell can read an EXPRESSION like
    "=<<miewb_vars>>.gap * 1mm" instead of being directly swept, so its
    owning dim_<El> sheet never lands in touched_sheets and
    rebuild_primitive_groups() skips the primitive — the recompute updates
    the cell's evaluated value, but the primitive's baked geometry (which
    only tracks rebuild-on-edit, not recompute) goes stale.

    When the sweep touched the miewb_vars sheet itself, scan every dim_*
    sheet's raw cell CONTENT (not value) for a "miewb_vars" reference and
    fold those sheets into touched_sheets too, so
    rebuild_primitive_groups() rebuilds them in the same pass. Cheap: one
    getUsedCells()/getContents() scan per dim_ sheet in the document, and
    only when miewb_vars was actually part of this sweep."""
    if not any(s.Label == train_fcstd.VARIABLES_SHEET for s in touched_sheets):
        return
    touched_names = {s.Name for s in touched_sheets}
    found = []
    for sheet in doc.Objects:
        if sheet.TypeId != "Spreadsheet::Sheet":
            continue
        if not sheet.Label.startswith("dim_") or sheet.Name in touched_names:
            continue
        for cell in sheet.getUsedCells():
            try:
                content = sheet.getContents(cell)
            except Exception:
                continue
            if content and train_fcstd.VARIABLES_SHEET in content:
                found.append(sheet)
                break
    if found:
        log("miewb_vars swept: expression-linked dim sheet(s) also "
            "touched: %s" % ", ".join(s.Label for s in found))
        touched_sheets.extend(found)


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


def apply_assignments(doc, assignments, unit="mm"):
    """Apply parameter assignments to an OPEN document exactly like one
    sweep-variant iteration of permute() below: set each aliased cell
    ("alias" on the default 'dim' sheet, or "sheetlabel.alias"), recompute,
    rebuild any touched primitive groups (including miewb_vars-driven
    ones), then re-solve the optical train.

    `assignments`: iterable of (var, value) in --var order. Raises
    PermuteError (via die()) on an unknown sheet/alias. Returns the number
    of train-solved elements.

    Shared BY CONTRACT with fcops.op_apply_params (the fast evaluator's
    persistent-worker path): both the write-a-variant-file flow and the
    in-place flow MUST mutate the document through this one function so
    they can never drift apart (the fast evaluator's parity oracle pins
    this equivalence end-to-end)."""
    default_sheet = None
    touched_sheets = []
    for var, value in assignments:
        sheet_label, alias = split_var(var)
        if sheet_label is None:
            if default_sheet is None:
                default_sheet = find_sheet(doc)
            sheet = default_sheet
        else:
            sheet = find_sheet(doc, sheet_label)
        cell = resolve_alias_cell(sheet, alias)
        # global-variables cells are UNITLESS by contract (they feed
        # expressions and train fields that expect plain numbers); dim
        # cells keep the length unit
        if sheet.Label == train_fcstd.VARIABLES_SHEET:
            sheet.set(cell, "=%.10g" % value)
        else:
            sheet.set(cell, "=%.10g %s" % (value, unit))
        if sheet not in touched_sheets:
            touched_sheets.append(sheet)
    doc.recompute()
    extend_touched_for_miewb_vars(doc, touched_sheets)
    rebuild_primitive_groups(doc, touched_sheets)
    # optical train: re-bake expression-driven props and re-solve every
    # chained placement against the variant's variable values (the GUI's
    # exact solver — see train_fcstd.py)
    n_train = train_fcstd.apply_train(doc, log=log)
    if n_train:
        doc.recompute()
        log("train: re-solved %d chained element(s)" % n_train)
    check_recompute(doc)
    return n_train


def permute(model_path, varspecs, outdir, unit, sweep_mode="product"):
    """varspecs: list of (var, vmin, vmax, n) in --var order.

    Writes one .FCStd per combination of common.sweep_values() for each
    varspec, combined via common.sweep_combos(value_lists, sweep_mode)
    ("product" = cartesian, the historical default; "zip" = variables
    advance together).
    """
    stem = model_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    value_lists = [common.sweep_values(vmin, vmax, n) for (_, vmin, vmax, n) in varspecs]
    names = [v[0] for v in varspecs]
    combos = common.sweep_combos(value_lists, mode=sweep_mode)
    log("permute_model.py: %s  vars=%s  sweep_mode=%s  -> %d combination(s) "
        "-> %s" % (model_path.name, names, sweep_mode, len(combos), outdir))

    n_written = 0
    for combo in combos:
        out_name = stem
        for var, value in zip(names, combo):
            out_name = common.variant_name(out_name, var, value)
        out_path = outdir / ("%s.FCStd" % out_name)

        doc = FreeCAD.openDocument(str(model_path))
        try:
            apply_assignments(doc, list(zip(names, combo)), unit=unit)

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

    n_written = permute(model_path, varspecs, outdir, args.unit,
                        sweep_mode=args.sweep_mode)

    log("PERMUTE MODEL OK (%d file(s) written)" % n_written)


# NOTE: no `if __name__ == "__main__"` guard — FreeCAD's console mode (-c)
# executes scripts with __name__ set to the module's basename, not
# "__main__", which would silently skip main() if guarded. Instead, run
# main() only when THIS file is the script FreeCAD (or plain python) was
# asked to execute: under `AppImage -c scripts/permute_model.py -- ...`
# sys.argv contains this file's path, while a library import (fcops'
# apply_params op inside fc_server.py) has fc_server.py there instead —
# so importing this module never triggers a permutation run.
def _run_as_script():
    base = os.path.basename(__file__)
    return any(os.path.basename(str(a)) == base for a in sys.argv)


if _run_as_script():
    try:
        main()
    except PermuteError:
        # die() already printed the ERROR line; preserve the historical
        # hard exit(1) (sys.exit is swallowed under FreeCAD -c).
        os._exit(1)
    sys.exit(0)
