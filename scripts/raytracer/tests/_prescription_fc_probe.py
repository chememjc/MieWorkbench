#!/usr/bin/env python3
# =============================================================================
# _prescription_fc_probe.py -- FreeCAD-embedded probe for test_prescription.py's
# FreeCAD-gated cross-check (engine3 Sec 3, P5). Builds a lens_pcx scene, then
# extracts it three ways and dumps the result JSON:
#   base   : no prescription                 -> native-OCC surfaces
#   presc  : matching prescription sidecar   -> emitted-from-prescription
#   drift  : deliberately corrupted (radius +5 um) -> must raise ExtractError
#
# Interpreter: the FreeCAD AppImage (see CLAUDE.md):
#   "$MIEWB_FREECAD" -c _prescription_fc_probe.py -- \
#       --out <result.json> < /dev/null
# Usual -c caveats: bare '--'; runs TWICE (idempotent writes); no __main__
# guard; os._exit not sys.exit; print() may drop (log via Console too).
# =============================================================================
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

SCRIPTS_DIR = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, SCRIPTS_DIR)

import FreeCAD as App  # noqa: E402
import primitivelib as pl  # noqa: E402
import make_test_scenes as mts  # noqa: E402
import extract_geometry as eg  # noqa: E402
from raytracer import prescription as pr  # noqa: E402

App.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
    "CreateBackupFiles", False)


def log(m):
    App.Console.PrintMessage(m + "\n")
    print(m, flush=True)


def parse_args():
    argv = sys.argv[1:]
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    try:
        return p.parse_args(argv)
    except SystemExit as exc:
        os._exit(exc.code if isinstance(exc.code, int) else 2)


def _surfaces(model):
    out = {}
    for b in model["bodies"]:
        for f in b["faces"]:
            if "Lens1" in f["id"]:
                out[f["id"].split(".")[-1]] = f["surface"]
    return out


def build_scene(path):
    doc = App.newDocument("presc_probe")
    try:
        bodies = pl.build_primitive(doc, "lens_pcx", group="Lens1")
        # exercise the local->global transform with a real placement
        bodies[0].Placement = App.Placement(
            App.Vector(11.0, 2.0, -3.0),
            App.Rotation(App.Vector(0, 0, 1), 13.0))
        mts.add_source(doc, "Source", -40.0, 1.5,
                       {"power": 5.0, "lambdac": 633.0})
        mts.add_detector(doc, "Screen", 80.0, 20.0)
        mts.finalize(doc, path)
    finally:
        App.closeDocument(doc.Name)
    # build_primitive() builds from default params (no dim sheet is authored
    # here), so the prescription is generated from the same defaults.
    return {a: s["default"]
            for a, s in pl.PRIMITIVES["lens_pcx"]["params"].items()}


def run():
    args = parse_args()
    work = Path(tempfile.mkdtemp(prefix="presc-probe-"))
    scene = work / "lens_pcx.FCStd"
    params = build_scene(str(scene))
    entry = pl.build_prescription_entry("lens_pcx", params)
    doc = pr.new_document({"Lens1": entry})

    result = {"kind": "lens_pcx", "entry_surfaces": len(entry["surfaces"])}

    # base (no prescription)
    m = eg.extract_document(App.openDocument(str(scene)), "base",
                            work / "geo_base", False, str(scene))
    App.closeDocument(App.ActiveDocument.Name)
    result["base"] = _surfaces(m)

    # prescription-emitted
    m = eg.extract_document(App.openDocument(str(scene)), "presc",
                            work / "geo_presc", False, str(scene),
                            prescription=doc)
    App.closeDocument(App.ActiveDocument.Name)
    result["presc"] = _surfaces(m)

    # deliberate mismatch -> ExtractError
    bad = pr.new_document({"Lens1": pl.build_prescription_entry(
        "lens_pcx", params)})
    bad["elements"]["Lens1"]["surfaces"][0]["radius"] += 5e-6
    try:
        eg.extract_document(App.openDocument(str(scene)), "drift",
                            work / "geo_drift", False, str(scene),
                            prescription=bad)
        result["drift_raised"] = False
    except eg.ExtractError as exc:
        result["drift_raised"] = True
        result["drift_msg"] = str(exc)
    finally:
        if App.ActiveDocument is not None:
            App.closeDocument(App.ActiveDocument.Name)

    with open(args.out, "w") as fh:
        json.dump(result, fh)
    log("PRESC PROBE wrote %s" % args.out)


run()
