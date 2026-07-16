#!/usr/bin/env python
# =============================================================================
# optiland_eval.py -- the P4b SEQUENTIAL merit evaluator (engine3.md Sec.5/8).
#
# Turns a geometry/<stem>/model.json (the SAME data the MC engine reads -- NOT
# FreeCAD) into deterministic, noise-free, differentiable merit operands via
# the Optiland sequential kernel (scripts/raytracer/optiland_bridge.py):
#   * spot RMS radius at the detector plane,
#   * best-focus axial shift from the detector,
#   * a GEOMETRIC encircled-energy radius (ray-density proxy, NOT a PSF).
# The output is shaped like the MC pipeline's report.json 'detectors' block so
# scripts/optimize.py's operand catalog reads it UNCHANGED (single-sourced
# merit definitions across the sequential and MC paths):
#
#   detectors.<label>.spot           = [{rms_radius_um, rms_pw_radius_um,
#                                        n_rays}]   (spot_rms / focus operands)
#   detectors.<label>.analysis       = {ee_r80_um (geometric proxy),
#                                        focus_shift_mm, paraxial_f2_mm, epd_mm}
#   detectors.<label>.total_power_W  = null   (MC-only; sequential cannot serve)
#
# Interpreter: env/bin/python (the GUI venv, where Optiland is installed --
# NOT the optics engine env). scripts/fast_eval.py (optics env) shells out to
# THIS module under env/bin/python and parses the JSON on stdout. On an
# out-of-scope scene the bridge raises BridgeUnsupported; this CLI then prints
# a one-line JSON {"bridge_unsupported": "<reason>"} and exits 3, which
# fast_eval reads as "fall back to the MC backend for this evaluation".
#
# CLI:
#   env/bin/python scripts/raytracer/optiland_eval.py \
#       --model-json geometry/lens_pcx/model.json [--model-stop] \
#       [--n-rays 4096] [--seed 0] [--ee-frac 0.8]
# =============================================================================
import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from raytracer import optiland_bridge as ob   # noqa: E402  (env has optiland)

# fast_eval reads this exit code as "BridgeUnsupported -> fall back to MC".
UNSUPPORTED_EXIT = 3


def evaluate_model_json(model_json, model_stop=True, n_rays=4096, seed=0,
                        ee_frac=0.8, optprops_dir=None):
    """model.json path -> report-shaped operand dict (see the module
    docstring). Raises ob.BridgeUnsupported for out-of-scope scenes."""
    geom_dir = Path(model_json).resolve().parent
    system = ob.load_sequential_system(geom_dir, model_stop=model_stop)
    m = ob.evaluate_geometry(system, n_rays=n_rays, seed=seed,
                             ee_frac=ee_frac, optprops_dir=optprops_dir)

    spot_um = m["spot_rms_m"] * 1e6
    block = {
        # equal-weight geometric bundle: the power-weighted and plain RMS are
        # the same (optimize.py prefers rms_pw_radius_um, falls back to
        # rms_radius_um).
        "spot": [{
            "source": 0, "lambda_stratum": 0,
            "rms_radius_um": spot_um,
            "rms_pw_radius_um": spot_um,
            "n_rays": m["n_rays"],
        }],
        "analysis": {
            # geometric encircled-energy radius, exposed under the operand's
            # key so encircled_energy reads it (documented proxy, not a PSF).
            "ee_r80_um": m["ee_radius_m"] * 1e6,
            "ee_frac": ee_frac,
            "focus_shift_mm": m["focus_shift_m"] * 1e3,
            "afocal": m["afocal"],
            "paraxial_f2_mm": m["paraxial_f2_mm"],
            "epd_mm": m["epd_mm"],
            "geometric": True,
        },
        # MC-only operands the sequential path cannot serve: explicit null so a
        # caller that reads them gets None (penalized) rather than a wrong 0.
        "total_power_W": None,
    }
    return {
        "backend": "sequential",
        "engine": "optiland-%s" % _optiland_version(),
        "model_stop": bool(model_stop),
        "detectors": {m["detector"]: block},
        "paraxial_f2_mm": m["paraxial_f2_mm"],
        "epd_mm": m["epd_mm"],
    }


def _optiland_version():
    try:
        import optiland
        return getattr(optiland, "__version__", "?")
    except Exception:
        return "?"


def main(argv=None):
    p = argparse.ArgumentParser(description="Sequential Optiland merit "
                                            "evaluator (see module docstring).")
    p.add_argument("--model-json", required=True)
    p.add_argument("--model-stop", action="store_true",
                   help="model an opaque iris annulus as a real aperture stop "
                        "(entrance pupil derived from its clear bore)")
    p.add_argument("--n-rays", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ee-frac", type=float, default=0.8)
    p.add_argument("--optical-properties", default=None,
                   help="override optical-properties root")
    args = p.parse_args(argv)

    try:
        out = evaluate_model_json(
            args.model_json, model_stop=args.model_stop, n_rays=args.n_rays,
            seed=args.seed, ee_frac=args.ee_frac,
            optprops_dir=args.optical_properties)
    except ob.BridgeUnsupported as exc:
        print(json.dumps({"bridge_unsupported": str(exc)}))
        return UNSUPPORTED_EXIT
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
