#!/usr/bin/env python
"""Capture pre-rebuild demo baselines for the object-placer equivalence gate.

Two artifacts per demo, committed under demos/baselines/:

  <name>.placements.json   every body's world placement + element grouping,
                           read through the fc worker (the same structure
                           the GUI sees)
  <name>.power.json        per-detector headline metrics from a quick-preset
                           run with --seeds N (per-seed std from the
                           detector .h5 gives the Monte-Carlo spread)

Run under the GUI venv (fcclient + h5py, both Qt-free):

  env/bin/python scripts/tools/capture_demo_baselines.py            # all demos
  env/bin/python scripts/tools/capture_demo_baselines.py --demos michelson
  env/bin/python scripts/tools/capture_demo_baselines.py --placements-only

Restartable: existing baseline files are skipped unless --force.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import miewb_tool  # noqa: E402

from mieworkbench.core.fcclient import FcClient  # noqa: E402

DEMOS_DIR = REPO / "demos"
BASELINE_DIR = DEMOS_DIR / "baselines"
WORK_DIR = REPO / "var" / "work" / "baseline_runs"

DEMO_NAMES = [
    "beam_expander", "camera_triplet", "czerny_turner", "dobsonian",
    "fiber_coupler", "michelson", "microscope_objective", "newtonian",
    "prism_spectrometer", "schmidt_cassegrain",
    # Phase-12 new-physics demos
    "ktp_walkoff", "gaussian_bench", "ghost_doublet", "scatter_plate",
    "curved_focal",
    # optimize/tolerance-round showcase demos (new)
    "double_gauss", "fiber_coupling_doublet",
    # WP7 beyond-sequential showcase demos
    "fizeau_flats", "fs_shg_spectrogram", "quartz_rotator",
    "speckle_mie_combo",
    # samples-instruments round demos (baseline placement + power gated)
    "conical_refraction", "colloidal_crystal", "goniometer_bath",
    "uvvis_spectrometer", "forward_scatter_diffraction_sizer", "imaging_bench",
]


def capture_placements(fc, name):
    """Open the demo .FCStd read-only and dump every body's placement."""
    structure = fc.open_document(str(DEMOS_DIR / ("%s.FCStd" % name)))
    doc = structure["doc"]
    try:
        bodies = {}
        for b in structure.get("bodies", []):
            props = b.get("properties", {})
            bodies[b["name"]] = {
                "label": b.get("label"),
                "placement": b.get("placement"),
                "group": (props.get("miewb_group") or {}).get("value"),
                "primitive": (props.get("miewb_primitive") or {}).get("value"),
            }
        sheets = {}
        for s in structure.get("sheets", []):
            sheets[s.get("label") or s.get("name")] = {
                alias: cell.get("raw")
                for alias, cell in (s.get("aliases") or {}).items()
            }
        return {"demo": name, "bodies": bodies, "sheets": sheets}
    finally:
        fc.close(doc)


def harvest_power(results_dir, stem, seeds):
    """report.json headline metrics + per-seed std from the detector cubes."""
    import h5py
    case_dirs = sorted((results_dir / stem).glob("*"))
    case_dirs = [d for d in case_dirs if (d / "report.json").exists()]
    if not case_dirs:
        raise RuntimeError("no finished case under %s/%s"
                           % (results_dir, stem))
    case = case_dirs[0]
    with open(case / "report.json") as fh:
        report = json.load(fh)
    detectors = {}
    for label, det in (report.get("detectors") or {}).items():
        entry = {
            "total_power_W": det.get("total_power_W"),
            "peak_irradiance_W_m2": det.get("peak_irradiance_W_m2"),
            "profile_visibility": det.get("profile_visibility"),
        }
        h5_path = case / "detectors" / ("%s.h5" % label.replace(".", "_"))
        if h5_path.exists():
            with h5py.File(h5_path, "r") as h5:
                if "spectral_cube_std" in h5:
                    # cube cells are watts (report total_power_W is a bare
                    # cube.sum()); treat pixel-bin cells as independent for
                    # a total-power MC-spread estimate
                    import numpy as np
                    std = h5["spectral_cube_std"][...]
                    entry["power_seed_std_W"] = float(
                        np.sqrt(np.sum(std.astype(np.float64) ** 2)))
        detectors[label] = entry
    return {
        "case": case.name,
        "seeds": seeds,
        "closure_ok": report.get("closure_ok"),
        "detectors": detectors,
    }


def run_demo(name, seeds):
    workdir = WORK_DIR / name
    if workdir.exists():
        import shutil
        shutil.rmtree(str(workdir))
    out_sim = workdir.with_suffix(".MieSim")
    rc, _ = miewb_tool.run_miewb(
        DEMOS_DIR / ("%s.MieWB" % name), out_sim, workdir=workdir,
        extra_args=["--seeds", str(seeds),
                    "--steps", "extract,trace,post"])
    if rc != 0:
        raise RuntimeError("pipeline failed for %s (exit %d)" % (name, rc))
    stem = miewb_tool.read_manifest(
        DEMOS_DIR / ("%s.MieWB" % name)).get("model_stem") or name
    return harvest_power(workdir / "results", stem, seeds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", default=",".join(DEMO_NAMES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--placements-only", action="store_true")
    ap.add_argument("--power-only", action="store_true")
    args = ap.parse_args()

    names = [n.strip() for n in args.demos.split(",") if n.strip()]
    unknown = [n for n in names if n not in DEMO_NAMES]
    if unknown:
        ap.error("unknown demos: %s" % ", ".join(unknown))
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    fc = None
    try:
        for name in names:
            pfile = BASELINE_DIR / ("%s.placements.json" % name)
            if not args.power_only and (args.force or not pfile.exists()):
                if fc is None:
                    fc = FcClient()
                    fc.start()
                data = capture_placements(fc, name)
                pfile.write_text(json.dumps(data, indent=1, sort_keys=True))
                print("[baseline] wrote %s" % pfile, flush=True)

            wfile = BASELINE_DIR / ("%s.power.json" % name)
            if not args.placements_only and (args.force
                                             or not wfile.exists()):
                data = run_demo(name, args.seeds)
                wfile.write_text(json.dumps(data, indent=1, sort_keys=True))
                print("[baseline] wrote %s" % wfile, flush=True)
    finally:
        if fc is not None:
            fc.shutdown()
    print("[baseline] done")


if __name__ == "__main__":
    main()
