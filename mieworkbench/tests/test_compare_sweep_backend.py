"""Backend tests for scripts/compare_sweep.py.

compare_sweep.py needs numpy/matplotlib/h5py (the optics env), which the
GUI venv this test suite normally runs under does NOT provide (see
CLAUDE.md's pinned-interpreter table). So this test builds tiny synthetic
case directories (report.json + detectors/*.h5, written with the GUI
venv's own h5py — that IS available there) and then runs compare_sweep.py
as a SUBPROCESS under /home3/optics/env/bin/python, exactly the way the
GUI's ComparePane does. Skipped outright if that interpreter is missing
(e.g. a bare clone with no optics env installed yet).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_SWEEP = REPO_ROOT / "scripts" / "compare_sweep.py"
OPTICS_PYTHON = os.environ.get("MIEWB_OPTICS_PYTHON",
                               "/home3/optics/env/bin/python")

pytestmark = pytest.mark.skipif(
    not os.path.exists(OPTICS_PYTHON),
    reason="optics env python not present on this machine")


def _write_case(case_dir, power_scale, shift_px):
    """A tiny synthetic case dir: report.json + detectors/D.h5, mirroring
    the attrs run_trace.py's save_detectors() writes (label, H, W,
    pixel_m, lam_lo_m, lam_hi_m, xhat, yhat, normal, x_lo, y_lo)."""
    case_dir.mkdir(parents=True)
    H = W = 8
    pixel_m = 5e-5
    cube = np.zeros((3, H, W), dtype=np.float64)
    cy, cx = H // 2, min(max(W // 2 + shift_px, 0), W - 1)
    cube[:, cy, cx] = power_scale * 1e-6 / 3.0
    mask = np.ones((H, W), dtype=bool)
    ddir = case_dir / "detectors"
    ddir.mkdir()
    with h5py.File(ddir / "D.h5", "w") as h:
        h["spectral_cube_mean"] = cube
        h["mask"] = mask
        h.attrs.update({
            "label": "D", "H": H, "W": W, "pixel_m": pixel_m,
            "lam_lo_m": 500e-9, "lam_hi_m": 600e-9,
            "xhat": [1.0, 0.0, 0.0], "yhat": [0.0, 1.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "x_lo": -W / 2 * pixel_m, "y_lo": -H / 2 * pixel_m,
            "seeds": 1,
        })
    total_w = float(cube.sum())
    irr = cube.sum(axis=0) / pixel_m ** 2
    report = {
        "closure_ok": True,
        "detectors": {
            "D": {
                "total_power_W": total_w,
                "peak_irradiance_W_m2": float(irr.max()),
                "profile_visibility": 0.9,
            }
        },
    }
    (case_dir / "report.json").write_text(json.dumps(report))


@pytest.fixture
def two_cases(tmp_path):
    v1 = tmp_path / "results" / "modelA-gap10" / "quick"
    v2 = tmp_path / "results" / "modelA-gap20" / "quick"
    _write_case(v1, power_scale=1.0, shift_px=0)
    _write_case(v2, power_scale=1.5, shift_px=1)
    return v1, v2


def _run(args, cwd=None):
    proc = subprocess.run(
        [OPTICS_PYTHON, str(COMPARE_SWEEP)] + args,
        cwd=cwd, capture_output=True, text=True, timeout=120)
    return proc


def test_manifest_mode_end_to_end(tmp_path, two_cases):
    v1, v2 = two_cases
    manifest = {
        "model": "modelA", "case": "quick", "mode": "product",
        "order": ["miewb_vars.gap"],
        "variants": [
            {"stem": "modelA-gap10", "values": {"miewb_vars.gap": 10.0},
             "case_dir": str(v1)},
            {"stem": "modelA-gap20", "values": {"miewb_vars.gap": 20.0},
             "case_dir": str(v2)},
        ],
    }
    manifest_path = tmp_path / "results" / "modelA" / "sweep-quick.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest))

    out_dir = tmp_path / "out_manifest"
    proc = _run(["--manifest", str(manifest_path), "--out", str(out_dir)])
    assert proc.returncode == 0, proc.stdout + proc.stderr

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["mode"] == "product"
    assert summary["model"] == "modelA"
    assert summary["case"] == "quick"
    assert summary["ref"] == "modelA-gap10"
    assert summary["variables_varying"] == ["miewb_vars.gap"]
    assert len(summary["variants"]) == 2
    for v in summary["variants"]:
        d = v["detectors"]["D"]
        for key in ("total_power_W", "peak_irradiance_W_m2",
                   "profile_visibility", "centroid_x_mm", "centroid_y_mm",
                   "rms_spot_radius_mm"):
            assert key in d

    assert (out_dir / "metrics.csv").exists()

    # one metric-vs-variable plot per (metric, detector) x the one
    # varying variable
    assert len(summary["plots"]) == 6
    for rel in summary["plots"]:
        assert (out_dir / rel).exists()

    # gallery: both variants rendered for detector D
    gallery = summary["gallery"]["D"]
    assert {e["stem"] for e in gallery} == {"modelA-gap10", "modelA-gap20"}
    for e in gallery:
        assert (out_dir / e["image"]).exists()

    # diff maps: ref is skipped, the other variant gets a diff image
    diffs = summary["diffs"]["D"]
    assert {e["stem"] for e in diffs} == {"modelA-gap20"}
    for e in diffs:
        assert (out_dir / e["image"]).exists()


def test_cases_mode_skips_variable_plots(tmp_path, two_cases):
    v1, v2 = two_cases
    out_dir = tmp_path / "out_cases"
    proc = _run(["--cases", str(v1), str(v2), "--out", str(out_dir)])
    assert proc.returncode == 0, proc.stdout + proc.stderr

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["mode"] == "cases"
    assert summary["variables_varying"] == []
    assert summary["plots"] == []          # no variable axis -> no plots
    assert (out_dir / "metrics.csv").exists()
    assert "D" in summary["gallery"]
    assert len(summary["gallery"]["D"]) == 2


def test_no_report_json_anywhere_exits_nonzero(tmp_path):
    empty = tmp_path / "results" / "nothing" / "quick"
    empty.mkdir(parents=True)
    out_dir = tmp_path / "out_empty"
    proc = _run(["--cases", str(empty), "--out", str(out_dir)])
    assert proc.returncode != 0
    assert "report.json" in (proc.stdout + proc.stderr)
