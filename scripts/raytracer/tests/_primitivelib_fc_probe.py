#!/usr/bin/env python3
# =============================================================================
# _primitivelib_fc_probe.py -- FreeCAD-embedded probe script for
# test_primitivelib.py's FreeCAD-gated checks (legacy-alias fallback in
# primitivelib.read_params, and rebuild_element's round_flag round-trip +
# label/placement/prop preservation).
#
# Interpreter: the FreeCAD AppImage (see CLAUDE.md):
#   /home3/freecad/FreeCAD.AppImage -c _primitivelib_fc_probe.py -- \
#       --out <result.json> < /dev/null
#
# Usual -c caveats: bare '--' before args; script runs TWICE (writing the
# result JSON is idempotent -- just overwritten each time); no __main__
# guard; os._exit instead of sys.exit; print() may be dropped so also log
# via FreeCAD.Console.
# =============================================================================
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPTS_DIR = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, SCRIPTS_DIR)

import FreeCAD as App  # noqa: E402
import primitivelib as pl  # noqa: E402

App.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
    "CreateBackupFiles", False)


def log(msg):
    App.Console.PrintMessage(msg + "\n")
    print(msg, flush=True)


def parse_args():
    argv = sys.argv[1:]
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    try:
        return p.parse_args(argv)
    except SystemExit as exc:
        os._exit(exc.code if isinstance(exc.code, int) else 2)


def probe_legacy_fallback():
    """A hand-built 'legacy' sheet carrying only the OLD 'radius' alias (no
    'diameter', no 'round_flag') for laser_collimated. read_params must
    report diameter = radius * 2 and round_flag = its spec default."""
    doc = App.newDocument("legacy_probe")
    try:
        sheet = doc.addObject("Spreadsheet::Sheet", "Spreadsheet")
        sheet.Label = "dim"
        sheet.set("A1", "radius")
        sheet.set("B1", "=5 mm")
        sheet.setAlias("B1", "radius")
        sheet.set("A2", "length")
        sheet.set("B2", "=12 mm")
        sheet.setAlias("B2", "length")
        doc.recompute()
        params = pl.read_params(sheet, "laser_collimated")
        return {"diameter": params["diameter"], "length": params["length"],
                "round_flag": params["round_flag"]}
    finally:
        App.closeDocument(doc.Name)


def probe_rebuild_round_flag_roundtrip():
    """Open the regenerated primitives/window.FCStd (round_flag=1 default,
    a cylinder -> 3 faces), give its body a placement/label/extra prop,
    flip round_flag to 0 in the sheet, rebuild_element, and check: (a) the
    body becomes a box (6 faces), (b) label/placement/extra prop survive."""
    win_path = os.path.join(SCRIPTS_DIR, "..", "primitives", "window.FCStd")
    doc = App.openDocument(os.path.normpath(win_path))
    try:
        body = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"
                and getattr(o, "miewb_group", None) == "window"][0]
        round_faces = len(body.Shape.Faces)

        placement = App.Placement(App.Vector(1.0, 2.0, 3.0), App.Rotation())
        body.Placement = placement
        body.Label = "MyWindowLabel"
        pl.safe_set_props(body, {"filter": "probe_marker"})

        sheet = doc.getObject("Spreadsheet")
        cell = sheet.getCellFromAlias("round_flag")
        sheet.set(cell, "0")
        doc.recompute()

        new_bodies = pl.rebuild_element(doc, sheet, "window", "window")
        nb = new_bodies[0]
        rect_faces = len(nb.Shape.Faces)
        return {
            "round_faces": round_faces,
            "rect_faces": rect_faces,
            "label": nb.Label,
            "placement_base": [nb.Placement.Base.x, nb.Placement.Base.y,
                               nb.Placement.Base.z],
            "filter_prop": getattr(nb, "filter", None),
        }
    finally:
        App.closeDocument(doc.Name)


def main():
    args = parse_args()
    result = {
        "legacy_fallback": probe_legacy_fallback(),
        "rebuild_roundtrip": probe_rebuild_round_flag_roundtrip(),
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=1)
    os.replace(tmp, args.out)
    log("PROBE OK -> %s" % args.out)


try:
    main()
except BaseException:
    import traceback
    for line in traceback.format_exc().splitlines():
        log(line)
    os._exit(1)
finally:
    os._exit(0)
