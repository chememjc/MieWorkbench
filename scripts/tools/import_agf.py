#!/usr/bin/env python3
# =============================================================================
# import_agf.py -- Zemax/OpticStudio AGF glass-catalog importer for
# opticalproperties/materials.miemat.
#
# Parses one or more Zemax ".agf" glass-catalog text files (the ANSI Glass
# Format used by Zemax OpticStudio and mirrored by most glass vendors --
# Schott, Ohara, Hoya, CDGM, ...) and converts every glass record whose
# dispersion formula the engine supports into a materials.miemat row:
#
#   AGF formula code 1 "Schott"     -> materials.miemat model=schott
#       n^2 = a0 + a1*l^2 + a2/l^2 + a3/l^4 + a4/l^6 + a5/l^8   (a0..a5 = CD[0:6])
#   AGF formula code 2 "Sellmeier1" -> materials.miemat model=sellmeier
#       n^2 = 1 + K1*l^2/(l^2-L1) + K2*l^2/(l^2-L2) + K3*l^2/(l^2-L3)
#       CD line lists K1,L1,K2,L2,K3,L3 interleaved; materials.miemat wants
#       p1..p3 = K1,K2,K3 (the B_i) and p4..p6 = L1,L2,L3 (the C_i).
# Any OTHER formula code (Sellmeier2/3/4/5, Herzberger, Extended, Handbook
# of Optics, Conrady, ...) is reported and SKIPPED -- never approximated by
# a different formula.
#
# Also parses, per glass:
#   ED  -> density_kg_m3 = ED[2] (g/cm^3, 0-indexed after the "ED" token) * 1000
#   LD  -> transmission_um_min/max (already in um)
#   TD  -> optional Schott TIE-19 thermo-optic coefficients
#          (D0 D1 D2 E0 E1 lambda_tk T_ref) -> thermo_d0..thermo_t_ref_c.
#          A blank/absent TD line (some Schott entries carry a bare "TD"
#          with no numbers) means "no thermo-optic model for this glass" --
#          the row's thermo_* columns are left empty, never guessed.
#
# Two independent things this script can do:
#   1. Parse + convert -> flat rows, optionally written to --out as a
#      standalone .miemat-shaped CSV (dry-run friendly, no merge).
#   2. --merge-into TARGET.miemat: apply those rows to a LIVE registry file
#      under a hard preservation guardrail (see merge_into_registry()):
#        - every pre-existing row's original 15 columns are byte-preserved
#          (new rows are appended to the raw line text; the 15 original
#          cells are never re-serialized through csv.writer);
#        - a pre-existing glass row is backfilled with AGF thermo data
#          ONLY when an AGF glass of the same name (case-insensitive, plus
#          a tiny built-in alias table for shorthand names like our own
#          "bk7" -> AGF "N-BK7") reproduces its n(587.6nm) to within
#          --match-tol (default 1e-3) -- i.e. it's confidently the same
#          glass -- and never touches p1..p6/density/notes/reference;
#        - every AGF glass not already present (by name) is appended as a
#          brand-new row.
#
# stdlib-only (system python3, per CLAUDE.md's pinned-interpreter table --
# this is a registry-authoring tool, not part of the trace pipeline).
#
# Usage:
#   python3 scripts/tools/import_agf.py library_data/agf/schott.agf \
#       library_data/agf/ohara.agf --dry-run
#
#   python3 scripts/tools/import_agf.py library_data/agf/schott.agf \
#       library_data/agf/ohara.agf \
#       --url schott.agf=https://www.schott.com/en-gb/products/optical-glass-p1000267/downloads \
#       --url ohara.agf=https://oharacorp.com/glass-catalog/ \
#       --retrieved 2026-07-13 \
#       --merge-into opticalproperties/materials.miemat
#
# Run with --help for the full flag list.
# =============================================================================
import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# AGF formula codes we can map onto materials.miemat's engine models.
# (raytracer/materials.py Material._n_from_model: only 'sellmeier' and
# 'schott' are parametric multi-term glass formulas; 'cauchy'/'constant'/
# 'tabulated' are not AGF dispersion forms.)
# ---------------------------------------------------------------------------
SUPPORTED_FORMULAS = {1: "schott", 2: "sellmeier"}
FORMULA_NAMES = {
    1: "Schott", 2: "Sellmeier1", 3: "Herzberger", 4: "Sellmeier2",
    5: "Conrady", 6: "Sellmeier3", 7: "Handbook of Optics 1",
    8: "Handbook of Optics 2", 9: "Sellmeier4", 10: "Extended1",
    11: "Sellmeier5", 12: "Extended2", 13: "Extended3",
}
STATUS_DESC = {0: "standard", 1: "preferred", 2: "obsolete", 3: "special",
               4: "melt"}

MATERIALS_HEADER = ["name", "class", "model", "p1", "p2", "p3", "p4", "p5",
                     "p6", "nk_file", "density_kg_m3",
                     "transmission_um_min", "transmission_um_max", "notes",
                     "reference"]
THERMO_COLS = ["thermo_d0", "thermo_d1", "thermo_d2", "thermo_e0",
               "thermo_e1", "thermo_lambda_tk", "thermo_t_ref_c"]
FULL_HEADER = MATERIALS_HEADER + THERMO_COLS

# Manual aliases for existing materials.miemat rows whose name does not
# case-insensitively equal the AGF glass name it corresponds to (verified
# by dispersion agreement in merge_into_registry -- this table only points
# the matcher at a candidate, it does not force a backfill).
BUILTIN_ALIASES = {
    "bk7": "N-BK7",
}


class AGFError(ValueError):
    pass


# ---------------------------------------------------------------------------
# AGF parsing
# ---------------------------------------------------------------------------
def _floats(tokens):
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except ValueError:
            out.append(None)
    return out


def parse_agf(path):
    """Parse one .agf file. Returns dict: name -> record dict with keys
    name, formula (int), nd, vd, status (int), cd (list[float], raw CD
    line values), density_g_cm3 (float or None), td (list[float] of
    exactly 7, or None if absent/blank), ld (tuple(min,max) or None).
    Records are returned in file order (dict preserves insertion order)."""
    records = {}
    cur = None
    with open(path, "r", encoding="latin-1", newline="") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n")
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "NM":
                if len(parts) < 6:
                    raise AGFError("%s: malformed NM line: %r" % (path, line))
                name = parts[1]
                if name in records:
                    raise AGFError(
                        "%s: duplicate glass name %r in one catalog file"
                        % (path, name))
                try:
                    formula = int(float(parts[2]))
                    nd = float(parts[4])
                    vd = float(parts[5])
                except ValueError:
                    raise AGFError("%s: malformed NM line: %r" % (path, line))
                status = 0
                if len(parts) > 7:
                    try:
                        status = int(float(parts[7]))
                    except ValueError:
                        status = 0
                cur = name
                records[name] = {
                    "name": name, "formula": formula, "nd": nd, "vd": vd,
                    "status": status, "cd": None, "density_g_cm3": None,
                    "td": None, "ld": None, "source_file": str(path),
                }
            elif tag == "CD":
                if cur is None:
                    continue
                records[cur]["cd"] = _floats(parts[1:])
            elif tag == "ED":
                if cur is None:
                    continue
                vals = _floats(parts[1:])
                # ED <tce_low> <tce_high> <density_g_cm3> <dPgF> <ignore_flag>
                if len(vals) >= 3 and vals[2] is not None:
                    records[cur]["density_g_cm3"] = vals[2]
            elif tag == "TD":
                if cur is None:
                    continue
                vals = parts[1:]
                if len(vals) == 7:
                    fvals = _floats(vals)
                    if all(v is not None for v in fvals):
                        records[cur]["td"] = fvals
                # else: blank/partial TD line -- no thermo model, leave None
            elif tag == "LD":
                if cur is None:
                    continue
                vals = _floats(parts[1:])
                if len(vals) >= 2 and vals[0] is not None and vals[1] is not None:
                    records[cur]["ld"] = (vals[0], vals[1])
            # GC / MD / OD / IT / CC / Re / BD / other tags: not needed for
            # a materials.miemat row, intentionally ignored.
    return records


# ---------------------------------------------------------------------------
# record -> materials.miemat row conversion
# ---------------------------------------------------------------------------
def fmt(v):
    """Shortest-round-trip-ish numeric formatting matching the convention
    already used in materials.miemat / gen_registry_rows.py's fmt()."""
    return "%.9g" % float(v)


def convert_params(formula, cd):
    """Return (p1..p6) tuple of strings for the given AGF formula code, or
    raise AGFError if unsupported / insufficient coefficients."""
    if formula not in SUPPORTED_FORMULAS:
        raise AGFError(
            "unsupported formula code %d (%s) -- only Schott(1)/Sellmeier1(2) "
            "are implemented by raytracer/materials.py"
            % (formula, FORMULA_NAMES.get(formula, "unknown")))
    if cd is None or len(cd) < 6 or any(v is None for v in cd[:6]):
        raise AGFError("CD line missing/short (need >=6 coefficients)")
    a = cd[:6]
    if SUPPORTED_FORMULAS[formula] == "schott":
        p = a  # a0..a5 map straight to p1..p6
    else:  # sellmeier: CD = K1,L1,K2,L2,K3,L3 -> p1..p3=K, p4..p6=L
        p = [a[0], a[2], a[4], a[1], a[3], a[5]]
    return tuple(fmt(x) for x in p)


def convert_thermo(td):
    """Return 7 formatted thermo cell strings, or 7 empty strings if td is
    None. td = [D0, D1, D2, E0, E1, lambda_tk_um, T_ref_c]."""
    if td is None:
        return ("",) * 7
    return tuple(fmt(x) for x in td)


def build_reference(catalog_label, source_file, url, retrieved):
    src = Path(source_file).name
    if url:
        return ("%s AGF glass catalog (%s; mirror file %s, retrieved %s)"
                % (catalog_label, url, src, retrieved))
    return ("%s AGF glass catalog (mirror file %s, retrieved %s)"
            % (catalog_label, src, retrieved))


def glass_to_row(rec, catalog_label, url, retrieved):
    """Convert one parsed AGF record to a full 22-column row dict, or raise
    AGFError naming why it can't be represented."""
    p1_6 = convert_params(rec["formula"], rec["cd"])
    density_g = rec["density_g_cm3"]
    if density_g is None or density_g <= 0:
        raise AGFError(
            "no usable density in AGF ED record (density_g_cm3=%r) -- "
            "engine requires density_kg_m3 > 0" % (density_g,))
    if rec["ld"] is None:
        raise AGFError("no LD (transmission range) record")
    tmin, tmax = rec["ld"]
    if not (tmin < tmax):
        raise AGFError("LD transmission range invalid (min>=max): %r" % (rec["ld"],))

    status_txt = STATUS_DESC.get(rec["status"], "status=%d" % rec["status"])
    notes = ("Zemax AGF import: formula=%s, Nd=%.6g, Vd=%.6g, "
             "catalog status=%s" % (FORMULA_NAMES[rec["formula"]], rec["nd"],
                                     rec["vd"], status_txt))
    row = dict(zip(MATERIALS_HEADER, [
        rec["name"], "glass", SUPPORTED_FORMULAS[rec["formula"]],
        p1_6[0], p1_6[1], p1_6[2], p1_6[3], p1_6[4], p1_6[5],
        "", fmt(density_g * 1000.0), fmt(tmin), fmt(tmax), notes,
        build_reference(catalog_label, rec["source_file"], url, retrieved),
    ]))
    thermo = convert_thermo(rec["td"])
    row.update(dict(zip(THERMO_COLS, thermo)))
    return row


def n_at(model, p, lam_um):
    """Evaluate n(lam_um) for model in {'schott','sellmeier'} with 6 float
    params p1..p6 -- pure-stdlib re-implementation of
    raytracer/materials.py's Material._n_from_model, used only to score
    dispersion agreement for the backfill-match confidence gate."""
    l2 = lam_um * lam_um
    if model == "sellmeier":
        n2 = 1.0
        for b, c in ((p[0], p[3]), (p[1], p[4]), (p[2], p[5])):
            n2 += b * l2 / (l2 - c)
        return n2 ** 0.5
    if model == "schott":
        n2 = (p[0] + p[1] * l2 + p[2] / l2 + p[3] / l2 ** 2
              + p[4] / l2 ** 3 + p[5] / l2 ** 4)
        return n2 ** 0.5
    raise AGFError("n_at: unsupported model %r" % model)


# ---------------------------------------------------------------------------
# catalog-label / url CLI plumbing
# ---------------------------------------------------------------------------
def _parse_kv_list(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise AGFError("expected FILE=VALUE, got %r" % item)
        k, v = item.split("=", 1)
        out[k] = v
    return out


def default_label(path):
    stem = Path(path).stem
    return stem[:1].upper() + stem[1:]


# ---------------------------------------------------------------------------
# registry read/splice helpers (byte-preservation guardrail)
# ---------------------------------------------------------------------------
def read_registry_raw(path):
    """Read a .miemat file, returning (line_ending, header_line_text,
    header_cols, [ (raw_line_text_no_eol, row_dict) ... ]). Detects LF vs
    CRLF from the raw bytes; every parsed row_dict is keyed by the ORIGINAL
    header (pre-thermo) so callers can look up by 'name' etc. Requires
    exactly one physical line per row (true of the shipped materials.miemat
    -- no field contains an embedded newline); this is verified by
    comparing line count to csv row count."""
    data = open(path, "rb").read()
    line_ending = "\r\n" if b"\r\n" in data else "\n"
    text = data.decode("utf-8")
    lines = text.split(line_ending)
    # split() on a trailing-newline file yields one empty trailing element
    trailing_blank = (lines and lines[-1] == "")
    if trailing_blank:
        lines = lines[:-1]

    header_line = lines[0]
    header_cols = next(csv.reader([header_line]))
    if "thermo_d0" in header_cols:
        raise AGFError(
            "%s: already has thermo_* columns -- refusing to double-merge "
            "(this script appends columns exactly once)" % path)

    reader = csv.reader(io.StringIO("\n".join(lines[1:])))
    parsed = list(reader)
    if len(parsed) != len(lines) - 1:
        raise AGFError(
            "%s: row count from csv parser (%d) != physical data line "
            "count (%d) -- a field must contain an embedded newline; "
            "the byte-preserving splice merge cannot be used safely"
            % (path, len(parsed), len(lines) - 1))

    rows = []
    for line_text, fields in zip(lines[1:], parsed):
        if len(fields) != len(header_cols):
            raise AGFError(
                "%s: row %r has %d fields, header has %d"
                % (path, line_text[:60], len(fields), len(header_cols)))
        rows.append((line_text, dict(zip(header_cols, fields))))
    return line_ending, header_line, header_cols, rows, trailing_blank


def csv_cell(value):
    """Format one value as a CSV cell using the same dialect (comma sep,
    QUOTE_MINIMAL) as the shipped registries."""
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow([value])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# the merge itself
# ---------------------------------------------------------------------------
def merge_into_registry(target_path, agf_rows_by_name, match_tol=1e-3,
                         aliases=None, log=print):
    """agf_rows_by_name: {agf_glass_name: full_row_dict (22 cols, from
    glass_to_row)}, name-keyed EXACTLY as in the AGF file (original case).
    Returns (new_text, report_dict); does not write anything -- caller
    decides (dry-run vs write)."""
    aliases = dict(BUILTIN_ALIASES, **(aliases or {}))
    line_ending, header_line, header_cols, rows, trailing_blank = \
        read_registry_raw(target_path)

    agf_by_lower = {name.lower(): row for name, row in agf_rows_by_name.items()}

    report = {
        "backfilled": [], "matched_no_thermo": [], "matched_mismatch": [],
        "appended": [], "append_skipped": [],
    }

    out_lines = [header_line + "," + ",".join(THERMO_COLS)]
    consumed_agf_lower = set()

    for line_text, row in rows:
        name = row.get("name", "")
        candidate_name = aliases.get(name, name)
        cand = agf_by_lower.get(candidate_name.lower())
        thermo_suffix = ",,,,,,,"  # 7 empty cells
        if cand is not None:
            consumed_agf_lower.add(candidate_name.lower())
            eligible = (row.get("class") == "glass"
                        and row.get("model") in ("schott", "sellmeier"))
            if eligible:
                try:
                    p_existing = tuple(float(row[k])
                                        for k in ("p1", "p2", "p3", "p4", "p5", "p6"))
                    n_existing = n_at(row["model"], p_existing, 0.5876)
                    p_agf = tuple(float(cand[k])
                                  for k in ("p1", "p2", "p3", "p4", "p5", "p6"))
                    n_agf = n_at(cand["model"], p_agf, 0.5876)
                    agree = abs(n_existing - n_agf) <= match_tol
                except (KeyError, ValueError, ZeroDivisionError):
                    agree = False
                if agree:
                    if cand["thermo_d0"] != "":
                        thermo_suffix = "," + ",".join(cand[c] for c in THERMO_COLS)
                        report["backfilled"].append((name, candidate_name))
                    else:
                        report["matched_no_thermo"].append((name, candidate_name))
                else:
                    report["matched_mismatch"].append((name, candidate_name))
            else:
                report["matched_no_thermo"].append((name, candidate_name))
        out_lines.append(line_text + thermo_suffix)

    # append brand-new AGF glasses (case-insensitive name not already used
    # by ANY existing row, matched or not, and not consumed above)
    existing_lower = {row.get("name", "").lower() for _, row in rows}
    writer_buf = io.StringIO()
    writer = csv.writer(writer_buf, lineterminator=line_ending)
    for name, row in agf_rows_by_name.items():
        if name.lower() in consumed_agf_lower:
            # already used to backfill an existing row (direct name match
            # or via an alias, e.g. AGF 'N-BK7' backfilling our 'bk7') --
            # do not ALSO append it as a separate new glass.
            continue
        if name.lower() in existing_lower:
            report["append_skipped"].append(
                (name, "name collides with an existing row of a "
                        "different glass (not matched/backfilled)"))
            continue
        out_lines.append(",".join(csv_cell(row[c]) for c in FULL_HEADER))
        report["appended"].append(name)

    new_text = line_ending.join(out_lines)
    if trailing_blank:
        new_text += line_ending
    return new_text, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert Zemax .agf glass catalogs to materials.miemat "
                     "rows, optionally merging them into a live registry "
                     "under a preservation guardrail (existing rows' first "
                     "15 columns are never rewritten).")
    ap.add_argument("agf_files", nargs="+", help=".agf catalog file(s)")
    ap.add_argument("--label", action="append", default=[],
                     metavar="FILE=LABEL",
                     help="human catalog label for FILE (default: "
                          "capitalized filename stem, e.g. schott.agf -> "
                          "'Schott'); used in the reference column")
    ap.add_argument("--url", action="append", default=[],
                     metavar="FILE=URL",
                     help="provenance URL for FILE's reference column")
    ap.add_argument("--retrieved", default="2026-07-13",
                     help="retrieval date stamped into the reference column "
                          "(default: %(default)s)")
    ap.add_argument("--alias", action="append", default=[],
                     metavar="EXISTING_NAME=AGF_NAME",
                     help="extra manual name alias for --merge-into matching "
                          "(beyond the built-in bk7=N-BK7); repeatable")
    ap.add_argument("--match-tol", type=float, default=1e-3,
                     help="max |delta n| at 587.6nm for a name match to be "
                          "trusted for thermo backfill (default: %(default)s)")
    ap.add_argument("--out", metavar="PATH",
                     help="write ALL convertible AGF glasses (full 22-col "
                          "materials.miemat-shaped CSV, LF, one row per "
                          "glass) to PATH -- independent of --merge-into")
    ap.add_argument("--merge-into", metavar="MATERIALS_MIEMAT",
                     help="apply the parsed catalogs to this live registry "
                          "file in place (adds the 7 thermo_* columns, "
                          "backfills matched existing rows, appends new "
                          "glasses). Combine with --dry-run to preview.")
    ap.add_argument("--dry-run", action="store_true",
                     help="parse + report, write nothing")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    labels = _parse_kv_list(args.label)
    urls = _parse_kv_list(args.url)
    aliases = _parse_kv_list(args.alias)

    def log(*a):
        if not args.quiet:
            print(*a)

    all_rows = {}          # agf name -> full row dict
    skip_report = []       # (catalog, name, reason)
    total_parsed = 0
    for fp in args.agf_files:
        fp = Path(fp)
        label = labels.get(fp.name) or labels.get(str(fp)) or default_label(fp)
        url = urls.get(fp.name) or urls.get(str(fp)) or ""
        records = parse_agf(fp)
        total_parsed += len(records)
        for name, rec in records.items():
            try:
                row = glass_to_row(rec, label, url, args.retrieved)
            except AGFError as e:
                skip_report.append((fp.name, name, str(e)))
                continue
            if name in all_rows:
                skip_report.append(
                    (fp.name, name,
                     "duplicate name across input catalogs (kept the first "
                     "occurrence: %s)" % all_rows[name]["reference"]))
                continue
            all_rows[name] = row
        log("%s: parsed %d glass records (label=%r)"
            % (fp, len(records), label))

    log("Total glasses parsed: %d" % total_parsed)
    log("Convertible to a materials.miemat row: %d" % len(all_rows))
    log("Skipped: %d" % len(skip_report))
    if skip_report:
        by_reason = {}
        for _, _, reason in skip_report:
            key = reason.split(" -- ")[0].split("(")[0].strip()
            by_reason.setdefault(key, 0)
            by_reason[key] += 1
        log("  skip reasons (grouped):")
        for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            log("    %4d  %s" % (count, reason))
        if not args.quiet:
            log("  (first 10 individual skips)")
            for cat, name, reason in skip_report[:10]:
                log("    %s / %s: %s" % (cat, name, reason))

    with_thermo = sum(1 for r in all_rows.values() if r["thermo_d0"] != "")
    log("Of those, %d carry Schott TIE-19 thermo-optic data" % with_thermo)

    if args.out and not args.dry_run:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(FULL_HEADER)
        for row in all_rows.values():
            writer.writerow([row[c] for c in FULL_HEADER])
        Path(args.out).write_text(buf.getvalue())
        log("Wrote %d rows to %s" % (len(all_rows), args.out))
    elif args.out and args.dry_run:
        log("[dry-run] would write %d rows to %s" % (len(all_rows), args.out))

    if args.merge_into:
        new_text, report = merge_into_registry(
            args.merge_into, all_rows, match_tol=args.match_tol,
            aliases=aliases, log=log)
        log("")
        log("=== merge report for %s ===" % args.merge_into)
        log("  backfilled thermo onto %d existing rows: %s"
            % (len(report["backfilled"]), report["backfilled"]))
        log("  matched by name but AGF glass has no thermo data (%d): %s"
            % (len(report["matched_no_thermo"]), report["matched_no_thermo"]))
        log("  matched by name but dispersion disagrees > tol (%d, NOT "
            "backfilled): %s"
            % (len(report["matched_mismatch"]), report["matched_mismatch"]))
        log("  appended new glass rows: %d" % len(report["appended"]))
        if report["append_skipped"]:
            log("  append skipped (name collision, not backfilled either) "
                "(%d): %s" % (len(report["append_skipped"]),
                               report["append_skipped"]))
        if args.dry_run:
            log("[dry-run] would write updated registry to %s "
                "(%d bytes)" % (args.merge_into, len(new_text)))
        else:
            Path(args.merge_into).write_bytes(new_text.encode("utf-8"))
            log("Wrote updated registry to %s" % args.merge_into)

    return 0


if __name__ == "__main__":
    sys.exit(main())
