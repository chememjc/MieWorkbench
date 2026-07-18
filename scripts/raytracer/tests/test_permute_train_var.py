# =============================================================================
# test_permute_train_var.py — the "train.<ElementLabel>.<field>" variable
# address form added to the permutation backend (permute_model.py), which
# lets tolerance / optimize runs perturb an element's optical-train chain-
# recipe pose fields (per-element decenter / despace / tilt).
#
# Two layers, both FreeCAD-gated (permute_model.py imports FreeCAD at module
# scope, so it can only be exercised under the AppImage interpreter — the
# same reason mieworkbench/tests/test_train_parity.py is env-gated):
#
#   1. UNIT + INTEGRATION (fast, no trace): an in-AppImage driver imports
#      permute_model + train_solver directly and checks
#        - parse_train_var / split_var recognize all three address forms,
#          reject a malformed train var and an unknown/anchored field with a
#          clear PermuteError, and leave the existing forms unchanged;
#        - apply_assignments applied to a hand-built chained scene routes a
#          train.<El>.decenter_x + train.<El>.tilt_ry perturbation into the
#          element's placement, and that placement equals a DIRECT
#          train_solver.solve_chain prediction of the perturbed recipe to
#          1e-9 (the same solver the GUI is pinned against).
#
#   2. TOLERANCE PATH (slow): fast_eval.Evaluator (the worker backend the
#      tolerance/optimize CLIs drive) evaluates one train field over a real
#      chained scene at two values and produces finite, differing detected
#      power — proving the new address form flows opaquely through the
#      Evaluator -> op_apply_params -> apply_assignments plumbing end to end.
# =============================================================================
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import common  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(common.FREECAD_APPIMAGE),
    reason="FreeCAD AppImage not available (%s)" % common.FREECAD_APPIMAGE)


# ---------------------------------------------------------------------------
# 1. unit + integration — one in-AppImage driver
# ---------------------------------------------------------------------------
# The driver runs under FreeCAD's bundled python (which has FreeCAD but no
# numpy); it prints TRAINVAR_OK on success and TRAINVAR_FAIL:<why> + exits 1
# on any failure. os._exit at the end short-circuits the AppImage's usual
# double execution of -c scripts.
_DRIVER = r'''
import os
import sys

argv = sys.argv
rest = argv[argv.index("--") + 1:] if "--" in argv else []
SCRIPTS = rest[0]
sys.path.insert(0, SCRIPTS)

import FreeCAD  # noqa: E402
import permute_model as pm       # noqa: E402
import train_solver             # noqa: E402


def fail(msg):
    print("TRAINVAR_FAIL: %s" % msg, flush=True)
    os._exit(1)


def expect_permute_error(fn, what):
    try:
        fn()
    except pm.PermuteError:
        return
    except Exception as exc:  # noqa: BLE001
        fail("%s: raised %s, expected PermuteError" % (what, type(exc).__name__))
    fail("%s: no error raised (expected PermuteError)" % what)


# -- parse_train_var: the new form, plus non-train pass-through --------------
if pm.parse_train_var("train.L1.decenter_x") != ("L1", "decenter_x"):
    fail("parse_train_var basic")
# element labels may contain spaces; the field is the segment after the
# LAST dot
if pm.parse_train_var("train.Fold Mirror.tilt_ry") != ("Fold Mirror", "tilt_ry"):
    fail("parse_train_var spaced label")
if pm.parse_train_var("lenspos") is not None:
    fail("parse_train_var bare alias should be None")
if pm.parse_train_var("dim_Lens1.ct") is not None:
    fail("parse_train_var sheet.alias should be None")
if pm.parse_train_var("trainish.x.y") is not None:
    fail("parse_train_var non-prefix should be None")

# -- split_var: existing forms unaffected ------------------------------------
if pm.split_var("lenspos") != (None, "lenspos"):
    fail("split_var bare")
if pm.split_var("miewb_vars.gap") != ("miewb_vars", "gap"):
    fail("split_var sheet-qualified")
if pm.split_var("dim_Lens1.ct") != ("dim_Lens1", "ct"):
    fail("split_var per-element sheet")

# -- malformed train var: clear error ----------------------------------------
expect_permute_error(lambda: pm.parse_train_var("train.L1"), "malformed (no field)")
expect_permute_error(lambda: pm.parse_train_var("train."), "malformed (empty)")

# -- build a minimal chained scene: anchored SRC -> chained L1 ---------------
doc = FreeCAD.newDocument("trainvar")


def add_body(label):
    b = doc.addObject("PartDesign::Body", label)
    b.Label = label
    return b


def add_train_prop(body, name, value):
    body.addProperty("App::PropertyString", name, "MieTrain")
    setattr(body, name, value)


src = add_body("SRC")
src.Placement = FreeCAD.Placement(FreeCAD.Vector(0.0, 0.0, 0.0),
                                  FreeCAD.Rotation())
l1 = add_body("L1")
add_train_prop(l1, "miewb_train_mode", "chained")
add_train_prop(l1, "miewb_train_ref", "SRC")
add_train_prop(l1, "miewb_train_distance", "30")
doc.recompute()

# -- apply_train_field validation -------------------------------------------
expect_permute_error(lambda: pm.apply_train_field(doc, "L1", "bogus", 1.0),
                     "unknown field")
expect_permute_error(lambda: pm.apply_train_field(doc, "NOPE", "distance", 1.0),
                     "missing element")
expect_permute_error(lambda: pm.apply_train_field(doc, "SRC", "decenter_x", 1.0),
                     "anchored element (no chained recipe)")

# -- integration: apply the two train perturbations via apply_assignments ----
DX, RY = 2.5, 5.0
pm.apply_assignments(doc,
                     [("train.L1.decenter_x", DX), ("train.L1.tilt_ry", RY)],
                     unit="mm")

placed = doc.getObject(l1.Name).Placement
got_pos = [placed.Base.x, placed.Base.y, placed.Base.z]
got_quat = list(placed.Rotation.Q)

# -- independent oracle: call train_solver.solve_chain DIRECTLY with the
#    perturbed recipe (hand-built, not read back from the doc) ---------------
records = {
    "SRC": {"label": "SRC", "mode": "anchored"},
    "L1": {"label": "L1", "mode": "chained", "ref": "SRC",
           "distance": "30", "decenter_x": "%.10g" % DX,
           "tilt_ry": "%.10g" % RY},
}
identity_q = list(FreeCAD.Rotation().Q)
anchors = {
    "SRC": {"pos_mm": [0.0, 0.0, 0.0], "quat": identity_q},
    "L1": {"pos_mm": [0.0, 0.0, 0.0], "quat": identity_q},
}
pred = train_solver.solve_chain(records, anchors, {})["placements"]["L1"]

# also prove the perturbation actually MOVED the element off the unperturbed
# on-axis chain position (distance 30 along +? -> decenter puts it off axis)
if abs(got_pos[0]) + abs(got_pos[1]) + abs(got_pos[2]) < 1e-6:
    fail("placement stayed at the origin (perturbation not applied)")

for i in range(3):
    if abs(got_pos[i] - pred["pos_mm"][i]) > 1e-9:
        fail("pos[%d] %.12g != solver %.12g" % (i, got_pos[i], pred["pos_mm"][i]))

# sign-insensitive quaternion compare
dplus = sum((got_quat[i] - pred["quat"][i]) ** 2 for i in range(4)) ** 0.5
dminus = sum((got_quat[i] + pred["quat"][i]) ** 2 for i in range(4)) ** 0.5
if min(dplus, dminus) > 1e-9:
    fail("quat %s != solver %s" % (got_quat, pred["quat"]))

print("TRAINVAR_OK", flush=True)
os._exit(0)
'''


def test_train_var_unit_and_parity(tmp_path):
    driver = tmp_path / "train_var_driver.py"
    driver.write_text(_DRIVER)
    proc = subprocess.run(
        [common.FREECAD_APPIMAGE, "-c", str(driver), "--", str(SCRIPTS)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    out = proc.stdout + "\n" + proc.stderr
    assert "TRAINVAR_FAIL" not in out, out[-4000:]
    assert proc.returncode == 0, "driver exit %d:\n%s" % (proc.returncode,
                                                          out[-4000:])
    assert "TRAINVAR_OK" in proc.stdout, out[-4000:]


# ---------------------------------------------------------------------------
# 2. tolerance path — fast_eval.Evaluator over one train field (slow)
# ---------------------------------------------------------------------------
DEMO = SCRIPTS.parent / "demos" / "beam_expander.MieWB"


@pytest.mark.slow
@pytest.mark.skipif(not DEMO.exists(), reason="beam_expander.MieWB demo absent")
def test_train_field_flows_through_fast_eval(tmp_path):
    """The exact plumbing tolerance.py / optimize.py drive: fast_eval's
    worker Evaluator applies a train.<El>.<field> perturbation (opaque name
    -> op_apply_params -> permute_model.apply_assignments -> train re-solve
    -> extract) and traces. Two decenter values of the beam expander's
    second lens must both yield a finite detected power, and the
    perturbation must MOVE it (a finite, non-zero merit delta)."""
    import fast_eval  # noqa: E402  (optics-env module)

    # unpack the baked chained model from the .MieWB gallery archive
    with zipfile.ZipFile(DEMO) as zf:
        inner = next(n for n in zf.namelist() if n.endswith("model.FCStd"))
        model = tmp_path / "beam_expander.FCStd"
        model.write_bytes(zf.read(inner))

    # "L2" is a chained element in beam_expander (it is a ref target of the
    # detector, and itself chains to L1); decenter it transversely.
    param = "train.L2.decenter_y"
    ev = fast_eval.Evaluator(
        model, params=[param], backend="worker", preset="quick",
        rays=4000, resolution=128, nlambda=3,
        workdir=tmp_path / "fasteval")
    try:
        def detected(res):
            return sum(v for k, v in res["merits"].items()
                       if k.endswith(".total_power_W"))

        p0 = detected(ev.evaluate({param: 0.0}))
        p1 = detected(ev.evaluate({param: 6.0}))
    finally:
        ev.close()

    import math
    assert math.isfinite(p0) and p0 >= 0.0, "nominal detected power %r" % p0
    assert math.isfinite(p1) and p1 >= 0.0, "perturbed detected power %r" % p1
    assert abs(p1 - p0) > 0.0, (
        "a 6 mm lens decenter left detected power unchanged "
        "(%.6g vs %.6g) — perturbation did not propagate" % (p0, p1))
