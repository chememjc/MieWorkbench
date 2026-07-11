# =============================================================================
# bake_detector_faces.py — bake resolved detector-face pins as `detector_face`
# body properties on the saved demo scene (called by make_demos.py).
#
# Interpreter: the FreeCAD AppImage:
#   /home3/freecad/FreeCAD.AppImage -c scripts/tools/bake_detector_faces.py \
#       -- --model demos/michelson.FCStd --pin 'Screen=Body.Tip.Face3' ... \
#       < /dev/null
#
# make_demos already resolves each pinned detector's recording FaceN id from a
# batch extraction of the SAVED scene (resolve_detector_pins). This tool writes
# that id back onto the detector body as the `detector_face` string property
# (group "Base"), so extract_geometry bakes it into model.json's detector dict
# in place of the closest-to-origin auto-pick — no extra transparent screen,
# so the scene stays C-engine-routable (unlike the CLI --detector-face path).
#
# All the usual FreeCAD -c caveats apply (runs twice: idempotent set/save; no
# __main__ guard; os._exit).
# =============================================================================
import os
import sys
from pathlib import Path

import FreeCAD as App  # noqa: E402

App.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
    "CreateBackupFiles", False)


def log(msg):
    App.Console.PrintMessage(msg + "\n")
    print(msg, flush=True)


def parse_args():
    argv = sys.argv[1:]
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    model = None
    pins = []
    i = 0
    while i < len(argv):
        if argv[i] == "--model":
            model = argv[i + 1]
            i += 2
        elif argv[i] == "--pin":
            label, _, face_id = argv[i + 1].partition("=")
            pins.append((label, face_id))
            i += 2
        else:
            log("ERROR: unexpected arg %r" % argv[i])
            os._exit(2)
    if not model or not pins:
        log("ERROR: usage -- --model <scene.FCStd> --pin 'Label=FaceId' ...")
        os._exit(2)
    return model, pins


def main():
    model, pins = parse_args()
    doc = App.openDocument(model)
    try:
        by_label = {}
        for obj in doc.Objects:
            if obj.TypeId == "PartDesign::Body":
                by_label.setdefault(getattr(obj, "Label", ""), obj)
        for label, face_id in pins:
            body = by_label.get(label)
            if body is None:
                log("ERROR: no PartDesign::Body labelled %r" % label)
                os._exit(1)
            if not hasattr(body, "detector_face"):
                body.addProperty("App::PropertyString", "detector_face",
                                 "Base")
            body.detector_face = face_id
            log("PIN %s <- detector_face=%s" % (label, face_id))
        doc.recompute()
        doc.save()
        log("BAKE OK (%d pins)" % len(pins))
    finally:
        App.closeDocument(doc.Name)


try:
    main()
except BaseException:
    import traceback
    for line in traceback.format_exc().splitlines():
        log(line)
    os._exit(1)
finally:
    os._exit(0)
