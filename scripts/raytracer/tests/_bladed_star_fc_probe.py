#!/usr/bin/env python3
# =============================================================================
# _bladed_star_fc_probe.py -- FreeCAD-embedded authoring probe for
# test_bladed_iris_star.py's end-to-end coherent diffraction-star gate.
#
# Authors two tiny scenes into <outdir>:
#   star_hex.FCStd     coherent collimated source -> 6-blade iris -> far screen
#   star_circle.FCStd  same, with a CIRCULAR iris (control: rings, no star)
# The source beam over-fills the aperture so the polygon/circle edge clips a
# coherent wavefront -> Fraunhofer far field on the screen.
#
# Interpreter: the FreeCAD AppImage (CLAUDE.md rules -- bare '--', runs twice /
# idempotent, no __main__ guard, os._exit, log to Console+print):
#   /home3/freecad/FreeCAD.AppImage -c _bladed_star_fc_probe.py -- \
#       --outdir <dir> < /dev/null
# =============================================================================
import argparse
import os
import sys
from pathlib import Path

SCRIPTS_DIR = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, SCRIPTS_DIR)

import FreeCAD as App  # noqa: E402
import primitivelib as pl  # noqa: E402

App.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
    "CreateBackupFiles", False)
mts = pl.mts


def log(msg):
    App.Console.PrintMessage(msg + "\n")
    print(msg, flush=True)


def parse_args():
    argv = sys.argv[1:]
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    try:
        return p.parse_args(argv)
    except SystemExit as exc:
        os._exit(exc.code if isinstance(exc.code, int) else 2)


APERTURE_MM = 0.30      # inscribed clear aperture
SCREEN_X_MM = 300.0


def author(outdir, shape):
    doc = App.newDocument("star_" + shape)
    try:
        mts.new_body_pad(doc, "Source", "Source",
                         circle=(0.0, 0.0, 0.8), x_start=-2.0, length=1.0,
                         props={"power": 5.0, "lambdac": 633.0,
                                "coherent": True})
        if shape == "hex":
            pl.make_sheet(doc, "iris_bladed", label="dim")
            pl.build_primitive(doc, "iris_bladed", group="Iris", params={
                "n_blades": 6, "outer_diameter": 6.0,
                "aperture_diameter": APERTURE_MM, "thickness": 0.2,
                "blade_rotation": 0.0, "blackness": 1.0})
        else:
            pl.make_sheet(doc, "iris", label="dim")
            pl.build_primitive(doc, "iris", group="Iris", params={
                "outer_diameter": 6.0, "thickness": 0.2,
                "hole_diameter": APERTURE_MM, "blackness": 1.0})
        mts.new_body_pad(doc, "Screen", "Screen",
                         rects=[(-6.0, -6.0, 12.0, 12.0)],
                         x_start=SCREEN_X_MM, length=1.0,
                         props={"material": "detector"})
        doc.recompute()
        out = Path(outdir) / ("star_%s.FCStd" % shape)
        doc.saveAs(str(out))
        log("AUTHORED %s" % out)
    finally:
        App.closeDocument(doc.Name)


def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for shape in ("hex", "circle"):
        author(args.outdir, shape)
    log("PROBE OK")


try:
    main()
except BaseException:
    import traceback
    for line in traceback.format_exc().splitlines():
        log(line)
    os._exit(1)
finally:
    os._exit(0)
