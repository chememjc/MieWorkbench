# =============================================================================
# test_persistent_worker.py — P3 persistent-worker gates (C engine only).
#
# The `miewb-trace --serve` worker (REGISTRY.md §6) is the DEFAULT path for a
# C-engine case: one child process per run_trace process serves every chunk
# trace + the final gather_only stage, amortizing spawn + CUDA context + the
# device-buffer pool. MIEWB_CENGINE_ONESHOT=1 is the escape hatch (the classic
# per-invocation path). These gates prove the worker is byte-exact vs one-shot,
# recovers from a bad request, and lets the driver fall back when it dies.
#
# Runs run_trace.py as a SUBPROCESS (real doubleslit geometry, coherent,
# C-routable, fast at 8e3 rays); MIEWB_CHUNK_RAYS forces the chunk count.
#
# Gates:
#   (a) a 4-chunk case via the worker is BIT-identical (.h5 cube + mask) to
#       the same case via one-shot invocations (the P1 checkpoint template).
#   (b) a fabricated bad path AND a malformed-JSON request each report rc!=0
#       on the protocol line and the worker SURVIVES to serve the next
#       (valid) request with rc==0.
#   (c) the worker killed mid-case -> the driver falls back to one-shot and
#       completes; the result is bit-identical to an uninterrupted run.
#   (d) the existing test_checkpoint_extend + test_cengine_parity suites run
#       green with the worker as the default path (they drive run_trace as a
#       subprocess, so they exercise the worker automatically — no duplication
#       here; run them with the worker default to close gate (d)).
#
# --no-gather-gate throughout: these gates test the worker PLUMBING, not the
# coherent-gather sampling quality (its own tests own that).
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


def _run(case_dir, rays, chunk_rays, env_extra=None, check=True):
    env = dict(os.environ)
    env["MIEWB_CHUNK_RAYS"] = str(int(chunk_rays))
    # a clean baseline: neither hook set unless the caller asks
    env.pop("MIEWB_CENGINE_ONESHOT", None)
    env.pop("MIEWB_WORKER_DIE_AFTER", None)
    if env_extra:
        env.update(env_extra)
    cmd = [common.OPTICS_PYTHON, str(SCRIPTS / "run_trace.py"),
           "--model-json", str(MODEL), "--case-dir", str(case_dir),
           "--rays", str(int(rays))] + BASE
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


def _n_chunks(case_dir):
    return len(json.loads(
        (case_dir / "cengine" / "checkpoint.json").read_text())["chunks"])


def test_a_worker_matches_oneshot(tmp_path):
    """(a) 4-chunk worker run == 4-chunk one-shot run, bit-for-bit."""
    worker = tmp_path / "worker"
    oneshot = tmp_path / "oneshot"
    pw = _run(worker, RAYS, RAYS // 4)                       # worker default
    _run(oneshot, RAYS, RAYS // 4,
         env_extra={"MIEWB_CENGINE_ONESHOT": "1"})           # one-shot

    # the worker path must have genuinely used the worker (not silently
    # fallen back to one-shot, which would also match)
    assert "persistent worker" in pw.stdout, pw.stdout[-2000:]
    assert "falling back to one-shot" not in pw.stdout, pw.stdout[-2000:]
    assert _n_chunks(worker) == 4 and _n_chunks(oneshot) == 4

    cw, mw = _cube(worker)
    co, mo = _cube(oneshot)
    assert np.array_equal(mw, mo)
    assert np.array_equal(cw, co), \
        "worker cube differs from one-shot (max |d|=%.3e)" \
        % np.abs(cw - co).max()


def test_b_bad_request_survives(tmp_path):
    """(b) a bad path and a malformed-JSON request each get rc!=0 and the
    worker survives to serve the next valid request."""
    # produce a valid request.json (one-chunk worker run)
    seed = tmp_path / "seedcase"
    _run(seed, RAYS, RAYS)
    good = next((seed / "cengine").rglob("request.json"))

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{ this is not valid json ")

    w = cengine.Worker()
    try:
        rc_missing = w.run(str(tmp_path / "does_not_exist.json"))
        assert rc_missing != 0, "missing request should report rc!=0"
        assert w.proc.poll() is None, "worker died on a missing request"

        rc_malformed = w.run(str(malformed))
        assert rc_malformed != 0, "malformed JSON should report rc!=0"
        assert w.proc.poll() is None, "worker died on malformed JSON"

        rc_good = w.run(str(good))
        assert rc_good == 0, "valid request after two bad ones should be rc==0"
        assert w.proc.poll() is None
    finally:
        w.close()


def test_c_worker_death_falls_back(tmp_path):
    """(c) worker dies mid-case -> driver falls back to one-shot, completes,
    and the cube is bit-identical to an uninterrupted run."""
    ref = tmp_path / "ref"
    _run(ref, RAYS, RAYS // 4)                               # clean reference
    cube_ref, _ = _cube(ref)

    killed = tmp_path / "killed"
    # worker exits before responding to the 2nd request -> fallback for the
    # rest of the case (the un-answered request re-runs one-shot).
    pk = _run(killed, RAYS, RAYS // 4,
              env_extra={"MIEWB_WORKER_DIE_AFTER": "2"})
    assert "falling back to one-shot" in pk.stdout, pk.stdout[-2000:]
    ck = json.loads((killed / "cengine" / "checkpoint.json").read_text())
    assert ck["status"] == "completed" and len(ck["chunks"]) == 4, ck["status"]

    cube_killed, _ = _cube(killed)
    assert np.array_equal(cube_ref, cube_killed), \
        "post-fallback cube differs from uninterrupted (max |d|=%.3e)" \
        % np.abs(cube_ref - cube_killed).max()
