# =============================================================================
# test_checkpoint_extend.py — P1 chunked-run contract gates (C engine only).
#
# Exercises the checkpoint / resume / additive-extension machinery on the REAL
# doubleslit geometry (coherent, C-engine-routable, fast at 1e4 rays). Runs
# run_trace.py as a SUBPROCESS so the kill-simulation (os._exit) doesn't take
# pytest down with it.
#
# Gates (spec: scratchpad/p1_specs/chunked_run_spec.md):
#   (a) N-chunk == 1-chunk : BIT-identical detector cube (same seed/target)
#   (b) kill mid-trace + --resume == uninterrupted : BIT-identical
#   (c) --extend base->target == fresh target : bit-identical for the 2x
#       (exact-halving) ratio here; statistically equivalent in general
#   (d) energy closure < 1e-3 at completion; per-chunk emitted scales exactly
#       with the chunk width (cursor)
#   (e) a misaligned primary_lo makes the C engine hard-error (exit != 0), and
#       the Python driver only ever emits aligned chunk boundaries
#
# --no-gather-gate is used throughout: these gates test the chunk PLUMBING,
# not the coherent-gather sampling quality (which has its own gate + tests).
# =============================================================================
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent.parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import common                                              # noqa: E402
from raytracer import cengine                              # noqa: E402

MODEL = REPO / "geometry" / "doubleslit" / "model.json"
RAYS = 8000
BASE = ["--nlambda", "3", "--resolution", "96", "--spectral-bins", "4",
        "--engine", "c", "--seeds", "1", "--no-gather-gate"]

pytestmark = pytest.mark.skipif(
    cengine.binary_path() is None or not MODEL.exists(),
    reason="needs the built C engine + extracted geometry/doubleslit")


def _run(case_dir, rays, chunk_rays, extra=(), stop_after=None, check=True):
    env = dict(os.environ)
    env["MIEWB_CHUNK_RAYS"] = str(int(chunk_rays))
    if stop_after is not None:
        env["MIEWB_CHUNK_STOP_AFTER"] = str(int(stop_after))
    else:
        env.pop("MIEWB_CHUNK_STOP_AFTER", None)
    cmd = [common.OPTICS_PYTHON, str(SCRIPTS / "run_trace.py"),
           "--model-json", str(MODEL), "--case-dir", str(case_dir),
           "--rays", str(int(rays))] + BASE + list(extra)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise AssertionError("run_trace failed (%d):\n%s\n%s"
                             % (p.returncode, p.stdout[-3000:],
                                p.stderr[-3000:]))
    return p


def _cube(case_dir):
    import h5py
    h5 = next((case_dir / "detectors").glob("*.h5"))
    with h5py.File(h5, "r") as h:
        return h["spectral_cube_mean"][...], h["mask"][...]


def test_a_chunked_matches_one_chunk(tmp_path):
    """(a) a 4-chunk trace is BIT-identical to the same run in one chunk."""
    c1 = tmp_path / "one"
    c4 = tmp_path / "four"
    _run(c1, RAYS, RAYS)                 # one chunk
    _run(c4, RAYS, RAYS // 4)            # four chunks
    n_chunks = len(json.loads(
        (c4 / "cengine" / "checkpoint.json").read_text())["chunks"])
    assert n_chunks == 4, n_chunks
    cube1, mask1 = _cube(c1)
    cube4, mask4 = _cube(c4)
    assert np.array_equal(mask1, mask4)
    assert np.array_equal(cube1, cube4), \
        "4-chunk cube differs from 1-chunk (max |d|=%.3e)" \
        % np.abs(cube1 - cube4).max()


def test_b_kill_resume_matches_uninterrupted(tmp_path):
    """(b) kill after >=1 chunk boundary + --resume == uninterrupted run."""
    ref = tmp_path / "ref"
    _run(ref, RAYS, RAYS // 4)           # clean 4-chunk reference
    cube_ref, _ = _cube(ref)

    killed = tmp_path / "killed"
    p = _run(killed, RAYS, RAYS // 4, stop_after=2, check=False)
    assert p.returncode != 0, "kill simulation should exit nonzero"
    ck = json.loads((killed / "cengine" / "checkpoint.json").read_text())
    assert ck["status"] == "tracing" and len(ck["chunks"]) == 2

    _run(killed, RAYS, RAYS // 4, extra=["--resume"])   # resume
    ck2 = json.loads((killed / "cengine" / "checkpoint.json").read_text())
    assert ck2["status"] == "completed" and len(ck2["chunks"]) == 4
    cube_res, _ = _cube(killed)
    assert np.array_equal(cube_ref, cube_res), \
        "resumed cube differs from uninterrupted (max |d|=%.3e)" \
        % np.abs(cube_ref - cube_res).max()


def test_c_extend_matches_fresh(tmp_path):
    """(c) --extend base->2*base == a fresh 2*base run. The 2x ratio makes
    the extend rescale an EXACT halving (scale=0.5), so this comes out
    bit-identical here; a general (non-power-of-two) ratio is only
    statistically equivalent (extra fp multiply on the reused chunks)."""
    base = RAYS
    tgt = 2 * RAYS
    ext = tmp_path / "ext"
    _run(ext, base, base)                                   # complete at base
    assert json.loads((ext / "cengine" / "checkpoint.json").read_text()
                      )["status"] == "completed"
    _run(ext, base, base, extra=["--extend", str(tgt)])     # extend to 2x
    ck = json.loads((ext / "cengine" / "checkpoint.json").read_text())
    assert ck["target_rays"] == tgt and ck["extensions"] == \
        [{"from": base, "to": tgt}]

    fresh = tmp_path / "fresh"
    _run(fresh, tgt, tgt)
    cube_ext, _ = _cube(ext)
    cube_fresh, _ = _cube(fresh)
    peak = max(abs(cube_fresh).max(), 1e-30)
    # statistical gate (always holds); bit-identity is asserted too because
    # the 2x ratio is an exact halving — if a future change breaks it this
    # message documents why the fallback is statistical.
    assert np.allclose(cube_ext, cube_fresh, rtol=1e-6, atol=1e-6 * peak), \
        "extend vs fresh reldiff too large (max |d|=%.3e, peak=%.3e)" \
        % (np.abs(cube_ext - cube_fresh).max(), peak)
    assert np.array_equal(cube_ext, cube_fresh), \
        "2x extend expected bit-identical (exact halving); got max |d|=%.3e" \
        % np.abs(cube_ext - cube_fresh).max()


def test_d_closure_and_emitted_scaling(tmp_path):
    """(d) closure < 1e-3 at completion; each chunk's emitted power scales
    exactly with its width (emitted_chunk == P_source * width / target)."""
    case = tmp_path / "case"
    _run(case, RAYS, RAYS // 4)
    audit = json.loads((case / "audit.json").read_text())["per_seed"][0]
    assert audit["closure_ok"]
    for sname, sd in audit["sources"].items():
        assert sd["closure_error"] < 1e-3, (sname, sd["closure_error"])
    ck = json.loads((case / "cengine" / "checkpoint.json").read_text())
    cdir = case / "cengine"
    # P_source (full emitted) from the completed audit
    p_src = {s: sd["emitted_W"] for s, sd in audit["sources"].items()}
    for c in ck["chunks"]:
        rep = json.loads((cdir / c["dir"] / "ledger.json").read_text())
        width = c["hi"] - c["lo"]
        for sname, sd in rep["sources"].items():
            expect = p_src[sname] * width / ck["target_rays"]
            assert abs(sd["emitted_W"] - expect) <= 1e-9 * max(expect, 1e-30) \
                + 1e-15, (sname, c, sd["emitted_W"], expect)


def test_e_misaligned_primary_lo_hard_errors(tmp_path):
    """(e) a misaligned primary_lo makes the C engine hard-error, and the
    Python driver only ever writes stride-aligned chunk boundaries."""
    case = tmp_path / "case"
    _run(case, RAYS, RAYS // 4)
    ck = json.loads((case / "cengine" / "checkpoint.json").read_text())
    stride = ck["align_stride"]
    target = ck["target_rays"]
    # driver never emits a misaligned boundary
    for c in ck["chunks"]:
        assert c["lo"] % stride == 0, c
        assert c["hi"] % stride == 0 or c["hi"] == target, c

    # craft a misaligned request (primary_lo not a multiple of stride) and
    # run the binary directly — it must refuse (exit != 0).
    chunk = cengine.binary_path()
    a_chunk = sorted((case / "cengine" / "seed42").glob("chunk_*"))[0]
    req = json.loads((a_chunk / "request.json").read_text())
    req["params"]["primary_lo"] = stride + 1          # guaranteed misaligned
    req["params"]["primary_hi"] = req["params"]["rays"]
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    req["out_dir"] = str(bad_dir)
    bad_req = tmp_path / "bad_req.json"
    bad_req.write_text(json.dumps(req))
    p = subprocess.run([str(chunk), "--config", str(bad_req)],
                       capture_output=True, text=True)
    assert p.returncode != 0, "misaligned primary_lo should hard-error"
    assert "aligned" in (p.stdout + p.stderr).lower()
