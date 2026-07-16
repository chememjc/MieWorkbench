"""P1 tile-factorized gather: tiled-vs-exact A/B gate (C engine).

The tiled kernel (cengine/src/kernels/gatherk.h, tile-factorized path)
replaces per-pair fp64 with per-(tile, sample) fp64 staging + an fp32
inner loop built on the exact algebraic identity
r - R = (2R(u.dp) + |dp|^2)/(r + R). The residual error is fp32
roundoff only, bounded by ~5e-7*k*dpmax (reported per key in
gather.json as phase_err_bound_rad); at fringe slopes that phase error
maps to an intensity deviation of at most a few times the bound,
relative to peak.

Gate pinned here (doubleslit, the fringe-physics oracle scene):
  - detected powers bit-identical between tiled and exact modes;
  - cube max deviation <= 4x the reported phase bound, rel-to-peak;
  - fringe visibility > 0.85 in BOTH modes;
  - no near-field fallbacks on this scene (near_exact_pairs == 0);
  - gather.json declares the mode actually run.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "geometry" / "doubleslit" / "model.json"
BINARY = REPO / "cengine" / "build" / "miewb-trace"

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and BINARY.exists()),
    reason="needs the doubleslit geometry cache and a built C engine")


def _run(case_dir, extra):
    cmd = [sys.executable, str(REPO / "scripts" / "run_trace.py"),
           "--model-json", str(MODEL), "--case-dir", str(case_dir),
           "--rays", "40000", "--nlambda", "3", "--resolution", "256",
           "--engine", "c"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    with h5py.File(next((case_dir / "detectors").glob("*.h5"))) as f:
        cube = f["spectral_cube_mean"][...]
    detected = json.loads((case_dir / "case.json").read_text())["detected"]
    gj = json.loads(next(case_dir.glob("cengine/seed*/gather.json"))
                    .read_text())
    return cube, detected, gj


def _visibility(cube):
    img = cube.sum(axis=0)
    h = img.shape[0]
    prof = np.clip(img[h // 2 - 3: h // 2 + 3].mean(axis=0), 0, None)
    lit = prof[prof > prof.max() * 0.02]
    return (prof.max() - lit.min()) / (prof.max() + lit.min())


def test_tiled_matches_exact_within_phase_budget(tmp_path):
    cube_t, det_t, gj_t = _run(tmp_path / "tiled", [])
    cube_e, det_e, gj_e = _run(tmp_path / "exact", ["--gather-exact"])

    keys_t = [e for d in gj_t.values() for e in d.values()]
    keys_e = [e for d in gj_e.values() for e in d.values()]
    assert all(e["gather_mode"] == "tiled" for e in keys_t)
    assert all(e["gather_mode"] == "exact" for e in keys_e)
    assert all(e["near_exact_pairs"] == 0 for e in keys_t)

    # detected power: the gather renormalizes to geometric power, so the
    # detected tallies must agree exactly
    assert json.dumps(det_t, sort_keys=True) == \
        json.dumps(det_e, sort_keys=True)

    bound = max(e["phase_err_bound_rad"] for e in keys_t)
    assert 0.0 < bound < 2e-3          # adaptive tiling honors the budget
    dev = np.abs(cube_t - cube_e).max() / np.abs(cube_e).max()
    assert dev <= 4.0 * bound, (dev, bound)

    assert _visibility(cube_t) > 0.85
    assert _visibility(cube_e) > 0.85
