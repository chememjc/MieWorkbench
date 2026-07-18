"""GUI-side view over the miewb_vars sweep-variables sheet (Qt-free).

Mirrors mieworkbench.core.train's translation-layer pattern: pure
functions/classes over the worker's sheet-echo dicts (see
core/train.variables_from_sheets for the echo shape),  no FreeCAD, no Qt.
Shared solver math (expression evaluation, cycle detection) and sweep
combination semantics come from scripts/train_solver.py and
scripts/common.py respectively, so the GUI can never drift from the
headless permuter (permute_model.py) or run_pipeline.py.

Sheet layout contract (one row per variable, columns fixed):
    A<row>  comment (plain text, no alias)
    B<row>  value        alias `<name>`      "=<num>" or "=<expr>" (UNITLESS)
    C<row>  min           alias `<name>__min`
    D<row>  max           alias `<name>__max`
    E<row>  n              alias `<name>__n`
    F<row>  enabled 0/1   alias `<name>__on`

A variable's `row` is derived from its VALUE cell's address (the B<row>
alias' "cell" field in the worker echo).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import common       # noqa: E402  (stdlib-only shared contract hub)
import train_solver  # noqa: E402

from .train import VARIABLES_SHEET, _VAR_META_SUFFIXES, is_variable_meta  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------
_COL_COMMENT = "A"
_COL_VALUE = "B"
_COL_MIN = "C"
_COL_MAX = "D"
_COL_N = "E"
_COL_ON = "F"

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# same cell-address collision rule as fcserver/fcops._validate_alias
_CELL_ADDR_RE = re.compile(r"^[A-Za-z]{1,2}[0-9]+$")
_CELL_ROW_RE = re.compile(r"^[A-Za-z]+([0-9]+)$")


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------
def validate_name(name):
    """None if `name` is a valid miewb_vars variable name, else an error
    string. Rules: must match ^[A-Za-z_][A-Za-z0-9_]*$; must not look like
    a spreadsheet cell address (FreeCAD hard-rejects those as aliases —
    the same trap as fcops._validate_alias, e.g. "R1", "A2", "ab12" all
    collide); must not contain "__" (reserved for the meta-suffix
    columns) and must not itself collide with a meta suffix."""
    name = "" if name is None else str(name)
    if not name:
        return "name must not be empty"
    if _CELL_ADDR_RE.match(name):
        return ("name %r looks like a spreadsheet cell address (matches "
                "^[A-Za-z]{1,2}[0-9]+$); FreeCAD rejects aliases shaped "
                "like a cell reference" % name)
    if not _NAME_RE.match(name):
        return ("name %r is not a valid identifier: must match "
                "^[A-Za-z_][A-Za-z0-9_]*$" % name)
    if "__" in name:
        return "name %r must not contain '__' (reserved for meta suffixes)" % name
    for suffix in _VAR_META_SUFFIXES:
        if name.endswith(suffix):
            return "name %r collides with the meta suffix %r" % (name, suffix)
    return None


# ---------------------------------------------------------------------------
# Sheet parsing
# ---------------------------------------------------------------------------
@dataclass
class VarRow:
    name: str
    value_raw: str          # expression text, leading '=' stripped
    value: float             # FreeCAD-evaluated float (None if unevaluable)
    vmin: float
    vmax: float
    nstep: int
    enabled: bool
    row: int                 # spreadsheet row, from the value cell address


def _cell_row(cell_addr):
    m = _CELL_ROW_RE.match(str(cell_addr))
    if not m:
        raise ValueError("cell address %r not recognized" % cell_addr)
    return int(m.group(1))


def _meta_float(aliases, name, suffix, default):
    entry = aliases.get(name + suffix)
    if entry is None:
        return default
    try:
        return float(entry.get("value"))
    except (TypeError, ValueError):
        return default


def parse_sheet(sheet_echo):
    """miewb_vars worker sheet-echo dict (one sheet, as returned by
    fcops._sheet_dict / carried in structure["sheets"]) -> {name: VarRow}.

    Meta-suffixed aliases (__min/__max/__n/__on) are consumed as row
    metadata, not returned as their own entries. Aliases that fail
    validate_name are skipped defensively (the GUI itself never writes
    one, but a hand-edited sheet might)."""
    aliases = (sheet_echo or {}).get("aliases") or {}
    out = {}
    for alias, cell in aliases.items():
        if is_variable_meta(alias):
            continue
        if validate_name(alias) is not None:
            continue
        raw = cell.get("raw") or ""
        value_raw = raw[1:] if raw.startswith("=") else raw
        try:
            value = float(cell.get("value"))
        except (TypeError, ValueError):
            value = None
        row = _cell_row(cell.get("cell"))

        default_bound = value if value is not None else 0.0
        vmin = _meta_float(aliases, alias, "__min", default_bound)
        vmax = _meta_float(aliases, alias, "__max", default_bound)
        nstep = int(_meta_float(aliases, alias, "__n", 0))
        enabled = _meta_float(aliases, alias, "__on", 1.0) != 0.0

        out[alias] = VarRow(name=alias, value_raw=value_raw, value=value,
                            vmin=vmin, vmax=vmax, nstep=nstep,
                            enabled=enabled, row=row)
    return out


def next_free_row(sheet_echo):
    """First unoccupied row (1-based) for a new variable — one past the
    highest row any existing alias' value/meta cell already occupies.
    (Comment-only rows with no alias are invisible to the worker echo and
    so cannot be detected here; the GUI writes the comment in the same
    op batch as the value cell, so this is not a practical gap.)"""
    aliases = (sheet_echo or {}).get("aliases") or {}
    rows = []
    for entry in aliases.values():
        addr = entry.get("cell")
        if not addr:
            continue
        try:
            rows.append(_cell_row(addr))
        except ValueError:
            continue
    return (max(rows) + 1) if rows else 1


# ---------------------------------------------------------------------------
# Cell plan (create/update one variable row)
# ---------------------------------------------------------------------------
def cell_plan(name, row=None, value=None, vmin=None, vmax=None,
              nstep=None, enabled=None, comment=None):
    """The set_cell request list ([{cell, raw, alias?}, ...]) that
    creates or updates one miewb_vars variable row.

    Only parameters passed as not-None are included in the plan — pass
    every field you want (re)written; anything omitted leaves whatever is
    already in that cell alone. `value` is the value_raw form (no leading
    '=', as produced by parse_sheet/VarRow.value_raw) — a plain number or
    an expression over other variable names; cell_plan adds the leading
    '='. `vmin`/`vmax`/`nstep` are plain floats/int; `enabled` is a bool
    (stored as a 0/1 cell). `comment` is written to column A with no
    alias.

    `row` is REQUIRED: pass next_free_row(sheet_echo) when creating a new
    variable, or the existing VarRow.row to update one in place — this
    function has no sheet state of its own to resolve it from `name`
    alone.
    """
    err = validate_name(name)
    if err:
        raise ValueError(err)
    if row is None:
        raise ValueError(
            "cell_plan requires an explicit row: pass "
            "next_free_row(sheet_echo) for a new variable, or the "
            "existing VarRow.row to update one in place")
    if row < 1:
        raise ValueError("row must be >= 1 (got %r)" % row)

    plan = []
    if comment is not None:
        plan.append({"cell": "%s%d" % (_COL_COMMENT, row), "raw": str(comment)})
    if value is not None:
        text = value if isinstance(value, str) else ("%.10g" % float(value))
        raw = text if text.startswith("=") else ("=" + text)
        plan.append({"cell": "%s%d" % (_COL_VALUE, row), "raw": raw,
                     "alias": name})
    if vmin is not None:
        plan.append({"cell": "%s%d" % (_COL_MIN, row),
                     "raw": "=%.10g" % float(vmin), "alias": name + "__min"})
    if vmax is not None:
        plan.append({"cell": "%s%d" % (_COL_MAX, row),
                     "raw": "=%.10g" % float(vmax), "alias": name + "__max"})
    if nstep is not None:
        plan.append({"cell": "%s%d" % (_COL_N, row),
                     "raw": "=%d" % int(nstep), "alias": name + "__n"})
    if enabled is not None:
        plan.append({"cell": "%s%d" % (_COL_ON, row),
                     "raw": "=%d" % (1 if enabled else 0),
                     "alias": name + "__on"})
    return plan


# ---------------------------------------------------------------------------
# Sweep spec / run-count / estimate
# ---------------------------------------------------------------------------
def qualify_var_name(name, varrows):
    """Sheet-qualify a variable-name token so permute_model.split_var
    routes it to the right spreadsheet.

    permute_model.split_var reads a bare (dot-free) name against the
    per-element `dim` sheet and "<sheet>.<alias>" against the named
    sheet; a global miewb_vars variable must therefore be emitted
    "miewb_vars.<name>" or it fails to resolve (the observed
    "alias '<name>' not found on spreadsheet 'dim'"). Rule:

      * a name that already contains "." passes through unchanged
        (already sheet-qualified, e.g. "dim_Lens1.ct");
      * a bare name that IS a key in `varrows` (the scene's miewb_vars
        variables from parse_sheet) is qualified "miewb_vars.<name>";
      * any other bare name passes through unchanged (a free-typed
        `dim`-sheet alias the user meant literally).

    This is the ONE authority the Optimize + Tolerance panes share with
    the sweep path (sweep_spec below), so the qualification rule cannot
    drift between the three."""
    if "." in name:
        return name
    if varrows and name in varrows:
        return "%s.%s" % (VARIABLES_SHEET, name)
    return name


def sweep_spec(rows):
    """{name: VarRow} -> (vars, mins, maxs, ns): only enabled rows, vars
    named "miewb_vars.<name>" (sheet-qualified, matching permute_model's
    --var convention), in sorted-name order — parallel lists ready to
    become repeated --var/--min/--max/--n flags."""
    names = sorted(name for name, r in (rows or {}).items() if r.enabled)
    varnames = ["%s.%s" % (VARIABLES_SHEET, name) for name in names]
    mins = [rows[name].vmin for name in names]
    maxs = [rows[name].vmax for name in names]
    ns = [rows[name].nstep for name in names]
    return varnames, mins, maxs, ns


def run_count(rows, mode="product"):
    """Number of variant runs a sweep over `rows` (enabled only) produces
    under `mode` ("product"/"zip") — common.sweep_values lengths combined
    via common.sweep_combos, so this can never disagree with what
    permute_model.py / run_pipeline.py will actually produce. Raises
    ValueError (same message as common.sweep_combos) on a zip length
    mismatch."""
    _, mins, maxs, ns = sweep_spec(rows)
    if not mins:
        return 1
    value_lists = [common.sweep_values(vmin, vmax, n)
                   for vmin, vmax, n in zip(mins, maxs, ns)]
    return len(common.sweep_combos(value_lists, mode=mode))


def estimate_sweep(rows, mode, single_run_estimate_s):
    """Pre-sweep summary math for the Phase D dialog: given the current
    variable rows, the chosen sweep mode, and one run's estimated
    duration (e.g. common.estimate(...)["total_s"]), return the run
    count, per-run seconds, total seconds, and a common.fmt_duration
    string ready to display."""
    runs = run_count(rows, mode)
    per_run_s = float(single_run_estimate_s)
    total_s = runs * per_run_s
    return {
        "runs": runs,
        "per_run_s": per_run_s,
        "total_s": total_s,
        "text": "%d run%s x %s = %s" % (
            runs, "" if runs == 1 else "s",
            common.fmt_duration(per_run_s), common.fmt_duration(total_s)),
    }


# ---------------------------------------------------------------------------
# Cycle / expression validation
# ---------------------------------------------------------------------------
def check_cycles(rows):
    """[] if every variable's expression resolves cleanly, else a list of
    error strings (circular references named a -> b -> a, or unparseable/
    unknown-variable expressions) — delegates to
    train_solver.resolve_variables so the GUI reports the exact same
    failures the headless re-solve would hit."""
    raw = {name: r.value_raw for name, r in (rows or {}).items()}
    try:
        train_solver.resolve_variables(raw)
    except train_solver.TrainError as e:
        return [str(e)]
    return []
