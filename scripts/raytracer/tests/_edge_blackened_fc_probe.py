#!/usr/bin/env python3
# =============================================================================
# _edge_blackened_fc_probe.py -- FreeCAD authoring probe for
# test_edge_blackened.py. Authors two identical diverging-lens scenes into
# <outdir>, differing only in the edge_blackened flag:
#   edge2_0.FCStd  clear rim (ideal transmitting edge)
#   edge2_1.FCStd  blackened rim (barrel absorbs)
# A collimated beam OVER-FILLS the clear aperture; a strong biconcave lens
# bends its rim rays outward into the cylindrical barrel -> a real edge
# interaction that the blackening swallows (ghost/stray-light suppression).
#
#   /home3/freecad/FreeCAD.AppImage -c _edge_blackened_fc_probe.py -- \
#       --outdir <dir> < /dev/null
# =============================================================================
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import FreeCAD as App  # noqa: E402
import primitivelib as pl  # noqa: E402

App.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
    "CreateBackupFiles", False)


def log(m):
    App.Console.PrintMessage(m + "\n")
    print(m, flush=True)


def parse_args():
    argv = sys.argv[1:]
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    try:
        return p.parse_args(argv)
    except SystemExit as exc:
        os._exit(exc.code if isinstance(exc.code, int) else 2)


def author(outdir, blk):
    doc = App.newDocument("edge2_%d" % blk)
    try:
        pl.make_sheet(doc, "lens_dcv", label="dim")
        pl.build_primitive(doc, "lens_dcv", group="Lens", params={
            "R_front": 18.0, "R_back": 18.0, "ct": 8.0, "aperture": 20.0,
            "edge_blackened": blk})
        pl.mts.new_body_pad(doc, "Src", "Src", circle=(0.0, 0.0, 9.7),
                            x_start=-15.0, length=1.0,
                            props={"power": 10.0, "lambdac": 633.0,
                                   "coherent": False})
        pl.mts.new_body_pad(doc, "Det", "Det",
                            rects=[(-60.0, -60.0, 120.0, 120.0)],
                            x_start=60.0, length=1.0,
                            props={"material": "detector"})
        doc.recompute()
        out = Path(outdir) / ("edge2_%d.FCStd" % blk)
        doc.saveAs(str(out))
        log("AUTHORED %s" % out)
    finally:
        App.closeDocument(doc.Name)


def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    author(args.outdir, 0)
    author(args.outdir, 1)
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
