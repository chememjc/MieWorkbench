# =============================================================================
# add_schmidt_corrector.py — add the hand-authored quartic Schmidt corrector
# plate to the schmidt_cassegrain demo scene (called by make_demos.py).
#
# Interpreter: the FreeCAD AppImage:
#   /home3/freecad/FreeCAD.AppImage -c scripts/tools/add_schmidt_corrector.py \
#       -- --model demos/schmidt_cassegrain.FCStd < /dev/null
#
# Why hand-authored: the catalog lens_asphere primitive exposes only the
# conic constant k, but a Schmidt corrector's profile is DOMINATED by the
# quartic term: the classic single-wavelength design (Schroeder,
# "Astronomical Optics"; Rutten & van Venrooij) is
#     z(r) = K [ r^4 - (3/2) a^2 r^2 ],   K = 1 / (4 (n-1) R_m^3)
# with neutral zone at r = (sqrt(3)/2) a = 0.866 a. Mapped onto the
# engine's surface_override asphere grammar (sag = r^2/(R(1+sqrt(1-(1+k)
# (r/R)^2))) + A4 r^4): A4 = K and the r^2 term folds into the paraxial
# radius via R = 1/(2 A2) with A2 = -(3/2) a^2 K. The BSpline profile
# below interpolates the EXACT override sag at 61 stations, so the
# extractor's <1 um verification gate passes (same technique as
# primitivelib._build_lens_asphere / make_test_scenes.make_lens_asphere).
#
# All the usual FreeCAD -c caveats apply (runs twice: the build is
# idempotent via a remove-first; no __main__ guard; os._exit).
# =============================================================================
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MIEWB_MTS_LIBRARY_ONLY", "1")

import FreeCAD as App  # noqa: E402
import Part  # noqa: E402
import make_test_scenes as mts  # noqa: E402

App.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
    "CreateBackupFiles", False)

# C8-class corrector: full 203.2 mm aperture, BK7 at the 550 nm design
# wavelength, primary R_m = 812.8 mm (Suiter's table / the classic form).
APERTURE_MM = 203.2
CT_MM = 5.0
RM_MM = 812.8
N_DESIGN = 1.51852          # BK7 at 550 nm (shipped Sellmeier row)
LABEL = "Corrector"


def log(msg):
    App.Console.PrintMessage(msg + "\n")
    print(msg, flush=True)


def corrector_coeffs():
    a = APERTURE_MM / 2.0
    K = 1.0 / (4.0 * (N_DESIGN - 1.0) * RM_MM ** 3)     # A4 [mm^-3]
    A2 = -1.5 * a * a * K                               # [mm^-1]
    R = 1.0 / (2.0 * A2)                                # paraxial radius
    return R, K, a


def override_sag(r, R, k, a4):
    """The engine's asphere sag formula (mm units, matches
    extract_geometry.asphere_sag_m)."""
    conic = r * r / (R * (1.0 + math.sqrt(1.0 - (1.0 + k) * (r / R) ** 2)))
    return conic + a4 * r ** 4


def parse_args():
    argv = sys.argv[1:]
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(argv) != 2 or argv[0] != "--model":
        log("ERROR: usage -- --model <scene.FCStd>")
        os._exit(2)
    return argv[1]


def main():
    model = parse_args()
    doc = App.openDocument(model)
    try:
        # idempotent: the -c double-execution (and reruns) replace the body
        for obj in list(doc.Objects):
            if getattr(obj, "Label", "") == LABEL and \
                    obj.TypeId == "PartDesign::Body":
                feats = list(getattr(obj, "Group", []) or [])
                doc.removeObject(obj.Name)
                for f in feats:
                    try:
                        doc.removeObject(f.Name)
                    except Exception:
                        pass
        doc.recompute()

        R, a4, sa = corrector_coeffs()
        n_samp = 61
        pts = [App.Vector(override_sag(sa * i / (n_samp - 1), R, 0.0, a4),
                          sa * i / (n_samp - 1), 0)
               for i in range(n_samp)]
        bs = Part.BSplineCurve()
        bs.interpolate(pts)
        xfr = pts[-1].x
        edges = [bs,
                 mts._line(xfr, sa, CT_MM, sa),
                 mts._line(CT_MM, sa, CT_MM, 0.0),
                 mts._line(CT_MM, 0.0, 0.0, 0.0)]
        body = mts.revolve_body(doc, LABEL, edges)
        body.Label = LABEL
        # the extractor's verifier measures sag along the face's OUTWARD
        # (-x) normal, so the -x-bulging profile built above reads as
        # POSITIVE sag in the canonical frame: declare with flipped signs
        # (verified: unflipped fails with a pure sign-mirror residual)
        mts.set_props(body, {
            "material": "bk7",
            "surface_override":
                "Face1=asphere:R=%.6f;k=0;A4=%.10e;r_max=%.4f"
                % (-R, -a4, sa),
        })
        doc.recompute()
        bad = [o.Name for o in doc.Objects
               if any("Invalid" in s or "Error" in s
                      for s in (getattr(o, "State", None) or []))]
        if bad:
            log("ERROR: invalid objects: %s" % bad)
            os._exit(1)
        doc.save()
        log("CORRECTOR OK R=%.1f A4=%.3e sag_edge=%.4f mm"
            % (R, a4, override_sag(sa, R, 0.0, a4)))
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
