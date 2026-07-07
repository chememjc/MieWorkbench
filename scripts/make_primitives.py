#!/usr/bin/env python3
# =============================================================================
# make_primitives.py — generate the MieWorkbench element library
# (primitives/*.FCStd + *.meta.json sidecars) from primitivelib.PRIMITIVES.
#
# Interpreter: the FreeCAD AppImage:
#   /home3/freecad/FreeCAD.AppImage -c scripts/make_primitives.py -- \
#       [--outdir primitives] [--kind <name>|all] < /dev/null
#
# Adding a NEW primitive: either add a builder to primitivelib.py and rerun
# this, or just drop any hand-authored single-element .FCStd (+ optional
# .meta.json) into primitives/ — the GUI library rescans the directory.
#
# All the usual FreeCAD -c caveats apply (double execution: writes are
# idempotent; no __main__ guard; os._exit; bare -- before args).
# =============================================================================
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import primitivelib  # noqa: E402

import FreeCAD as App  # noqa: E402

App.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
    "CreateBackupFiles", False)


def log(msg):
    App.Console.PrintMessage(msg + "\n")
    print(msg, flush=True)


def parse_args():
    # Under `AppImage -c script.py -- <args>` our args follow the bare
    # `--`; without one, sys.argv holds FreeCAD's own argv — use defaults.
    argv = sys.argv[1:]
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=str(common.PROJECT_DIR / "primitives"))
    p.add_argument("--kind", default="all",
                   help="one primitive kind, or 'all'")
    try:
        return p.parse_args(argv)
    except SystemExit as exc:      # argparse exit is swallowed under -c
        os._exit(exc.code if isinstance(exc.code, int) else 2)


def build_one(kind, outdir):
    spec = primitivelib.PRIMITIVES[kind]
    outpath = outdir / (kind + ".FCStd")
    doc = App.newDocument(kind)
    try:
        primitivelib.make_sheet(doc, kind, label="dim")
        primitivelib.build_primitive(doc, kind)
        doc.recompute()
        bad = [o.Name for o in doc.Objects
               if any("Invalid" in s or "Error" in s
                      for s in (getattr(o, "State", None) or []))]
        if bad:
            log("ERROR: %s: invalid objects after recompute: %s"
                % (kind, bad))
            os._exit(1)
        doc.saveAs(str(outpath))
    finally:
        App.closeDocument(doc.Name)

    meta = {
        "kind": kind,
        "category": spec["category"],
        "label": spec["label"],
        "tooltip": spec["tooltip"],
        "params": {a: {"default": s["default"], "unit": s["unit"],
                       "help": s["help"]}
                   for a, s in spec["params"].items()},
        "props": spec["props"],
    }
    meta_path = outdir / (kind + ".meta.json")
    tmp = str(meta_path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    os.replace(tmp, meta_path)
    log("wrote %s (+ meta)" % outpath)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kinds = (list(primitivelib.PRIMITIVES) if args.kind == "all"
             else [args.kind])
    for kind in kinds:
        if kind not in primitivelib.PRIMITIVES:
            log("ERROR: unknown primitive kind %r" % kind)
            os._exit(2)
        build_one(kind, outdir)
    log("PRIMITIVES OK (%d)" % len(kinds))


try:
    main()
except BaseException:
    import traceback
    for line in traceback.format_exc().splitlines():
        log(line)
    os._exit(1)
finally:
    os._exit(0)
