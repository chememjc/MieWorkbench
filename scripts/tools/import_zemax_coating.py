#!/usr/bin/env python3
# =============================================================================
# import_zemax_coating.py -- Zemax OpticStudio "TABLE" coating file importer
# for opticalproperties/coating/coatings.miecoat (+ its per-item table).
#
# Zemax's measured/tabulated coating format (Optical Coating manual, "Table
# coatings", corroborated by the Zemax community/Ansys-support knowledge
# base -- e.g. community.zemax.com "Beam splitter coating as function of
# angle" and support.zemax.com "How to model a dichroic beam splitter"):
#
#   TABLE <coating name>
#   ANGL <angle of incidence, degrees>
#   WAVE <wavelength, MICROMETERS> Rs Rp Ts Tp [Ars Arp Ats Atp]
#   WAVE <wavelength, MICROMETERS> Rs Rp Ts Tp [Ars Arp Ats Atp]
#   ...
#   ANGL <next angle of incidence, degrees>
#   WAVE ...
#   ...
#
# One TABLE block, ONE OR MORE ANGL sub-blocks (angle of incidence), each
# followed by one WAVE line per tabulated wavelength. Rs/Rp/Ts/Tp are the
# power reflectance/transmittance per polarization (the |r|^2/|t|^2 -- same
# quantity as this repo's coating table columns). Ars/Arp/Ats/Atp are
# OPTIONAL trailing columns: the s/p reflection and transmission phase
# ANGLE IN DEGREES ("phase rotation angles" -- omitted means "no phase
# change", i.e. exactly this repo's phase_valid=False table-coating
# default). Angles and wavelengths must each be listed in ascending order
# within their block (Zemax's own documented requirement) -- this importer
# doesn't re-sort, it just enforces the same ordering materials.py's table
# loader requires (a mis-ordered or duplicate-wavelength source file is a
# hard error, not silently sorted -- a silently reordered/deduped table
# could quietly change which value a given row means).
#
# This tool converts exactly ONE (coating name, angle of incidence) pair
# per invocation -- opticalproperties/coating/coatings.miecoat's schema is
# single-AOI per row (aoi_deg column), matching how every hand-authored
# .miecoat table row in this repo already works (a multi-angle Zemax file
# with several ANGL blocks needs one importer run per angle you want, each
# writing its own <name>_<aoi>.mietab row -- --aoi-deg picks which block).
#
# stdlib-only (system python3, per CLAUDE.md's pinned-interpreter table --
# this is a registry-authoring tool, not part of the trace pipeline).
# numpy is OPTIONAL (only used for the --dry-run energy-closure sanity
# print; the parser/writer never needs it).
#
# Usage:
#   python3 scripts/tools/import_zemax_coating.py my_coating.dat \
#       --aoi-deg 45 --name my_bs_45 \
#       --reference "vendor coating.dat, retrieved 2026-07-16" \
#       --merge-into opticalproperties/coating/coatings.miecoat
#
#   # inspect without writing anything:
#   python3 scripts/tools/import_zemax_coating.py my_coating.dat --dry-run
# =============================================================================
import argparse
import csv
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CSV_KW = dict(lineterminator="\r\n")   # matches the shipped .miecoat/.mietab files


class ZemaxCoatingError(Exception):
    pass


def fmt(v):
    """Number formatting matching gen_registry_rows.py's convention (up to
    9 significant digits, shortest representation)."""
    return "%.9g" % float(v)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def parse_zemax_table(text, ctx="<input>"):
    """Parse a Zemax TABLE coating block.

    Returns (coating_name, [block, ...]) where each block is
      {"aoi_deg": float, "lam_um": [float, ...],
       "Rs": [...], "Rp": [...], "Ts": [...], "Tp": [...],
       "has_phase": bool,
       # present only when has_phase:
       "Ars": [...], "Arp": [...], "Ats": [...], "Atp": [...]}
    in file order. Comment lines ('!' or '#' first non-blank char) and
    blank lines are skipped (Zemax's own coating.dat files use '!'
    comments; this is a permissive superset, never required)."""
    lines = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        s = raw.strip()
        if not s or s[0] in "!#":
            continue
        lines.append((lineno, s))
    if not lines:
        raise ZemaxCoatingError("%s: empty file" % ctx)

    lineno0, first = lines[0]
    parts = first.split(None, 1)
    if not parts or parts[0].upper() != "TABLE":
        raise ZemaxCoatingError(
            "%s:%d: expected 'TABLE <name>' as the first line, got %r"
            % (ctx, lineno0, first))
    name = parts[1].strip() if len(parts) > 1 else ""
    if not name:
        raise ZemaxCoatingError("%s:%d: TABLE line has no coating name"
                                % (ctx, lineno0))

    blocks = []
    cur = None
    for lineno, s in lines[1:]:
        toks = s.split()
        kw = toks[0].upper()
        if kw == "ANGL":
            if len(toks) != 2:
                raise ZemaxCoatingError(
                    "%s:%d: ANGL expects exactly one value (angle deg), "
                    "got %r" % (ctx, lineno, s))
            aoi = float(toks[1])
            if cur is not None and cur["lam_um"] == []:
                raise ZemaxCoatingError(
                    "%s:%d: ANGL %g has no WAVE rows before the next ANGL"
                    % (ctx, lineno, cur["aoi_deg"]))
            if blocks and aoi <= blocks[-1]["aoi_deg"]:
                raise ZemaxCoatingError(
                    "%s:%d: ANGL values must be strictly ascending (%.6g "
                    "after %.6g)" % (ctx, lineno, aoi, blocks[-1]["aoi_deg"]))
            cur = {"aoi_deg": aoi, "lam_um": [], "Rs": [], "Rp": [],
                   "Ts": [], "Tp": [], "has_phase": None,
                   "Ars": [], "Arp": [], "Ats": [], "Atp": []}
            blocks.append(cur)
        elif kw == "WAVE":
            if cur is None:
                raise ZemaxCoatingError(
                    "%s:%d: WAVE row before any ANGL block" % (ctx, lineno))
            nums = toks[1:]
            if len(nums) not in (5, 9):
                raise ZemaxCoatingError(
                    "%s:%d: WAVE expects 5 numbers (lam Rs Rp Ts Tp) or 9 "
                    "(+ Ars Arp Ats Atp), got %d" % (ctx, lineno, len(nums)))
            try:
                nums = [float(x) for x in nums]
            except ValueError:
                raise ZemaxCoatingError(
                    "%s:%d: WAVE row not all numeric: %r" % (ctx, lineno, s))
            has_phase = len(nums) == 9
            if cur["has_phase"] is None:
                cur["has_phase"] = has_phase
            elif cur["has_phase"] != has_phase:
                raise ZemaxCoatingError(
                    "%s:%d: WAVE rows in ANGL %g mix phase and no-phase "
                    "columns -- all rows in one ANGL block must agree"
                    % (ctx, lineno, cur["aoi_deg"]))
            lam_um = nums[0]
            if cur["lam_um"] and lam_um <= cur["lam_um"][-1]:
                raise ZemaxCoatingError(
                    "%s:%d: WAVE wavelengths must be strictly ascending "
                    "within an ANGL block (%.6g after %.6g)"
                    % (ctx, lineno, lam_um, cur["lam_um"][-1]))
            cur["lam_um"].append(lam_um)
            cur["Rs"].append(nums[1])
            cur["Rp"].append(nums[2])
            cur["Ts"].append(nums[3])
            cur["Tp"].append(nums[4])
            if has_phase:
                cur["Ars"].append(nums[5])
                cur["Arp"].append(nums[6])
                cur["Ats"].append(nums[7])
                cur["Atp"].append(nums[8])
        else:
            raise ZemaxCoatingError(
                "%s:%d: unrecognized keyword %r (expected ANGL or WAVE)"
                % (ctx, lineno, toks[0]))
    if not blocks:
        raise ZemaxCoatingError("%s: TABLE %r has no ANGL blocks" % (ctx, name))
    if cur["lam_um"] == []:
        raise ZemaxCoatingError(
            "%s: ANGL %g has no WAVE rows" % (ctx, cur["aoi_deg"]))
    return name, blocks


def select_block(blocks, aoi_deg):
    """Pick the ANGL block matching aoi_deg (default: the only block, or a
    hard error naming the available angles if there's more than one and
    none was requested -- opticalproperties/coating/coatings.miecoat is
    single-AOI-per-row, so an ambiguous multi-angle file must be resolved
    explicitly, never silently averaged/nearest-picked)."""
    if aoi_deg is None:
        if len(blocks) > 1:
            raise ZemaxCoatingError(
                "file has %d ANGL blocks (%s deg) -- pass --aoi-deg to "
                "pick one (coatings.miecoat is single-AOI-per-row)"
                % (len(blocks), ", ".join("%.6g" % b["aoi_deg"]
                                          for b in blocks)))
        return blocks[0]
    for b in blocks:
        if abs(b["aoi_deg"] - aoi_deg) < 1e-9:
            return b
    raise ZemaxCoatingError(
        "no ANGL %.6g block in file -- available angles: %s"
        % (aoi_deg, ", ".join("%.6g" % b["aoi_deg"] for b in blocks)))


# ---------------------------------------------------------------------------
# .mietab table writer
# ---------------------------------------------------------------------------
def write_mietab(block, out_path):
    fieldnames = ["wavelength_nm", "Rs", "Rp", "Ts", "Tp"]
    if block["has_phase"]:
        fieldnames += ["ars_deg", "arp_deg", "ats_deg", "atp_deg"]
    buf = io.StringIO()
    writer = csv.writer(buf, **CSV_KW)
    writer.writerow(fieldnames)
    n = len(block["lam_um"])
    for i in range(n):
        row = [fmt(block["lam_um"][i] * 1000.0),   # um -> nm
               fmt(block["Rs"][i]), fmt(block["Rp"][i]),
               fmt(block["Ts"][i]), fmt(block["Tp"][i])]
        if block["has_phase"]:
            row += [fmt(block["Ars"][i]), fmt(block["Arp"][i]),
                    fmt(block["Ats"][i]), fmt(block["Atp"][i])]
        writer.writerow(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        fh.write(buf.getvalue())


# ---------------------------------------------------------------------------
# coatings.miecoat row upsert (same idempotent by-name upsert convention
# as gen_registry_rows.py)
# ---------------------------------------------------------------------------
def upsert_coating_row(registry_path, row):
    with open(registry_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)
    by_name = {r["name"]: i for i, r in enumerate(rows)}
    full = {c: row.get(c, "") for c in fieldnames}
    if row["name"] in by_name:
        rows[by_name[row["name"]]] = full
    else:
        rows.append(full)

    buf = io.StringIO()
    writer = csv.writer(buf, **CSV_KW)
    writer.writerow(fieldnames)
    for r in rows:
        writer.writerow([r[c] for c in fieldnames])
    with open(registry_path, "w", newline="") as fh:
        fh.write(buf.getvalue())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert a Zemax TABLE coating file into an "
        "opticalproperties/ .mietab + coatings.miecoat row.")
    ap.add_argument("input", help="Zemax TABLE coating .dat/.txt file")
    ap.add_argument("--aoi-deg", type=float, default=None,
                    help="which ANGL block to import (required if the "
                    "file has more than one)")
    ap.add_argument("--name", default=None,
                    help="coatings.miecoat row name (default: the TABLE "
                    "name from the file, lowercased/underscored)")
    ap.add_argument("--reference", default=None,
                    help="reference/citation for coatings.miecoat's "
                    "REQUIRED reference column (e.g. the vendor/source "
                    "and retrieval date of the Zemax coating.dat file). "
                    "Required unless --dry-run.")
    ap.add_argument("--out-table", default=None,
                    help="output .mietab path (default: "
                    "opticalproperties/coating/tables/<name>.mietab)")
    ap.add_argument("--merge-into", default=None,
                    help="coatings.miecoat registry to upsert the new row "
                    "into (default: don't touch any registry, just write "
                    "the table)")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse + report only, write nothing")
    args = ap.parse_args(argv)

    src = Path(args.input)
    text = src.read_text()
    try:
        zname, blocks = parse_zemax_table(text, ctx=str(src))
        block = select_block(blocks, args.aoi_deg)
    except ZemaxCoatingError as exc:
        print("import_zemax_coating: %s" % exc, file=sys.stderr)
        return 1

    name = args.name or re.sub(r"[^a-z0-9_]+", "_", zname.strip().lower())
    n = len(block["lam_um"])
    print("parsed TABLE %r: AOI=%.6g deg, %d wavelength rows (%.4g-%.4g "
          "um), phase columns: %s"
          % (zname, block["aoi_deg"], n, block["lam_um"][0],
             block["lam_um"][-1], block["has_phase"]))

    if args.dry_run:
        try:
            import numpy as np
            Rs = np.array(block["Rs"]); Ts = np.array(block["Ts"])
            Rp = np.array(block["Rp"]); Tp = np.array(block["Tp"])
            print("  Rs+Ts range: [%.6g, %.6g]" % ((Rs + Ts).min(),
                                                    (Rs + Ts).max()))
            print("  Rp+Tp range: [%.6g, %.6g]" % ((Rp + Tp).min(),
                                                    (Rp + Tp).max()))
        except ImportError:
            pass
        return 0

    if not args.reference:
        print("import_zemax_coating: --reference is required (not "
              "--dry-run) -- coatings.miecoat hard-validates every row "
              "has one", file=sys.stderr)
        return 1

    out_table = Path(args.out_table) if args.out_table else (
        ROOT / "opticalproperties" / "coating" / "tables" / ("%s.mietab" % name))
    write_mietab(block, out_table)
    print("wrote %s" % out_table)

    if args.merge_into:
        reg_path = Path(args.merge_into)
        table_ref = out_table.name
        row = {"name": name, "layers": "", "table": table_ref,
               "aoi_deg": fmt(block["aoi_deg"]), "reference": args.reference}
        upsert_coating_row(reg_path, row)
        print("upserted %r into %s" % (name, reg_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
