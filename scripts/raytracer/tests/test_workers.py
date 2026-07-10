# =============================================================================
# test_workers.py — multi-process trace sharding (--workers, lowhanging #12).
#
# The trace loop can be sharded across spawned processes: each worker traces
# rays/N primaries with an independent RNG stream, the parent merges the
# per-detector accumulators + power ledger and runs the coherent Huygens
# gather ONCE. These tests pin the merge identity, the ledger merge, the
# workers=1 short-circuit (must NOT spawn) and coherent/export correctness.
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_workers.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "raytracer" / "tests"))

import common                                             # noqa: E402
import cli_specs                                          # noqa: E402
import run_trace                                          # noqa: E402
import scenehelpers as sh                                 # noqa: E402
from raytracer.audit import PowerLedger                   # noqa: E402
from raytracer.sources import wavelength_strata           # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _prep(tmp_path, bodies, rays=8000, nlambda=1, resolution=64,
          extra_argv=None):
    """Write a synthetic model.json, build args + Scene, return everything
    run_one_seed needs."""
    model = sh.make_model(bodies)
    common.validate_model(model)
    mj = tmp_path / "model.json"
    common.write_json(mj, model)
    argv = ["--model-json", str(mj), "--case-dir", str(tmp_path / "case"),
            "--rays", str(rays), "--nlambda", str(nlambda),
            "--resolution", str(resolution), "--seed0", "42"]
    argv += extra_argv or []
    args = cli_specs.build_parser("trace").parse_args(argv)
    scene = run_trace.build_scene(args)
    lam_range = run_trace.lam_range_nm(scene)
    particle_lams = sorted({
        float(l) for _, src in scene.sources
        for l in wavelength_strata(src, args.nlambda)})
    return scene, args, lam_range, particle_lams


def _run(scene, args, lam_range, particle_lams, workers, seed=42,
         export=False):
    return run_trace.run_one_seed(
        scene, args, seed, lam_range, particle_lams, {},
        export=export, workers=workers)


def _bucket_totals(rep):
    from raytracer.audit import BUCKETS
    return {b: sum(s[b] for s in rep["sources"].values()) for b in BUCKETS}


def _emitted(rep):
    return sum(s["emitted_W"] for s in rep["sources"].values())


# ---------------------------------------------------------------------------
# PowerLedger.merge unit test
# ---------------------------------------------------------------------------
def test_powerledger_merge_sums_credits():
    a = PowerLedger(2)
    a.emit(np.array([0, 1]), np.array([1.0, 2.0]))
    a.credit("escaped", np.array([0]), np.array([0.25]), where="bodyA")
    a.detect("Det", 0.5)
    a.flux_in("L", 0.3)

    b = PowerLedger(2)
    b.emit(np.array([0, 1]), np.array([3.0, 4.0]))
    b.credit("escaped", np.array([1]), np.array([0.75]), where="bodyA")
    b.detect("Det", 1.5)
    b.flux_in("L", 0.7)

    a.merge(b)
    assert np.allclose(a.emitted, [4.0, 6.0])
    assert np.isclose(a.buckets["escaped"].sum(), 1.0)
    assert np.isclose(a.by_body["bodyA"], 1.0)
    assert np.isclose(a.detected["Det"], 2.0)
    assert np.isclose(a.flux["L"]["in_W"], 1.0)

    # report() closure of the merged ledger is well-formed
    rep = a.merge(PowerLedger(2)).report(["s0", "s1"])
    assert set(rep["sources"]) == {"s0", "s1"}

    with pytest.raises(ValueError):
        a.merge(PowerLedger(3))


# ---------------------------------------------------------------------------
# Merge identity: workers=2 vs workers=1 on a slab scene
# ---------------------------------------------------------------------------
def test_merge_identity_slab(tmp_path):
    bodies = [sh.source_body(coherent=False, power_mW=2.0),
              sh.slab_body("L", "bk7", 0.0, 0.005, half=0.01),
              sh.detector_body(x=0.03, half=0.01)]
    scene, args, lr, pl = _prep(tmp_path, bodies, rays=8000)

    r1, g1, _, _ = _run(scene, args, lr, pl, workers=1)
    r2, g2, _, _ = _run(scene, args, lr, pl, workers=2)

    rep1 = r1.ledger.report(r1.source_names)
    rep2 = r2.ledger.report(r2.source_names)

    # emitted power exactly equal (both = the full source power)
    np.testing.assert_allclose(_emitted(rep2), _emitted(rep1), rtol=1e-9)

    # closure holds in both
    assert rep1["closure_ok"] and rep2["closure_ok"]
    for rep in (rep1, rep2):
        for s in rep["sources"].values():
            assert s["closure_error"] < 1e-3

    # per-bucket ledger totals within max(3sigma, 1%). A planar source at
    # normal incidence gives every ray identical Fresnel physics, so the
    # partition is deterministic and the shards agree very tightly.
    t1, t2 = _bucket_totals(rep1), _bucket_totals(rep2)
    emit = _emitted(rep1)
    for b in t1:
        tol = max(3.0 * np.sqrt(max(t1[b], 0.0) * emit / 8000.0),
                  0.01 * emit)
        assert abs(t2[b] - t1[b]) <= tol, (b, t1[b], t2[b], tol)

    # detected power (ledger + detector cube) agrees within 1%
    det1 = sum(g1[f].inc.sum() for f in g1)
    det2 = sum(g2[f].inc.sum() for f in g2)
    assert abs(det2 - det1) <= max(0.01 * det1, 3e-12)


# ---------------------------------------------------------------------------
# workers=1 short-circuit: must NOT spawn any process
# ---------------------------------------------------------------------------
def test_workers1_does_not_spawn(tmp_path, monkeypatch):
    bodies = [sh.source_body(coherent=False),
              sh.slab_body("L", "bk7", 0.0, 0.005, half=0.01),
              sh.detector_body()]
    scene, args, lr, pl = _prep(tmp_path, bodies, rays=4000)

    def _boom(*a, **k):
        raise AssertionError("workers=1 must not touch multiprocessing")
    monkeypatch.setattr(run_trace.multiprocessing, "get_context", _boom)

    # two workers=1 runs on the same seed are byte-identical (shared path)
    r1a, g1a, _, _ = _run(scene, args, lr, pl, workers=1)
    r1b, g1b, _, _ = _run(scene, args, lr, pl, workers=1)
    for f in g1a:
        assert np.array_equal(g1a[f].inc, g1b[f].inc)


# ---------------------------------------------------------------------------
# Coherent correctness: gathered cube power + sample-key sets
# ---------------------------------------------------------------------------
def test_coherent_shard_matches(tmp_path):
    # A DIVERGING Gaussian-beam coherent source (not a perfectly collimated
    # plane wave, which is degenerate for the Huygens gather — see
    # CLAUDE.md) so the single-process gather is itself stable and a
    # meaningful reference.
    bodies = [sh.source_body(coherent=True, power_mW=1.0, half=0.004,
                             polarization={"kind": "linear",
                                           "angle_deg": 0.0},
                             beam_waist_mm=0.05),
              sh.detector_body(x=0.03, half=0.02)]
    scene, args, lr, pl = _prep(tmp_path, bodies, rays=20000, resolution=96,
                                extra_argv=["--no-gather-gate"])

    r1, g1, _, _ = _run(scene, args, lr, pl, workers=1)
    r2, g2, _, _ = _run(scene, args, lr, pl, workers=2)

    # identical (source, lam, pol) sample-key sets on every detector
    for f in g1:
        assert set(g1[f].samples) == set(g2[f].samples)
        assert set(g1[f].samples)          # non-empty coherent population

    # gathered detector-cube total power within max(3sigma, 1%)
    p1 = sum(g1[f].inc.sum() for f in g1)
    p2 = sum(g2[f].inc.sum() for f in g2)
    sigma = p1 / np.sqrt(20000.0)
    assert abs(p2 - p1) <= max(3.0 * sigma, 0.01 * p1), (p1, p2)


# ---------------------------------------------------------------------------
# Export-rays + workers: records merge across shards, cap respected
# ---------------------------------------------------------------------------
def test_export_rays_shard_merge(tmp_path):
    bodies = [sh.source_body(coherent=False, power_mW=1.0),
              sh.detector_body(x=0.03, half=0.01)]
    scene, args, lr, pl = _prep(tmp_path, bodies, rays=6000)

    _, g1, _, _ = _run(scene, args, lr, pl, workers=1, export=True)
    _, g2, _, _ = _run(scene, args, lr, pl, workers=3, export=True)

    n1 = sum(sum(len(r["pos"]) for r in g1[f].ray_records) for f in g1)
    n2 = sum(sum(len(r["pos"]) for r in g2[f].ray_records) for f in g2)
    assert n1 > 0
    # every landing ray is captured under both (all primaries reach the
    # detector on this open path), so the merged shard counts match
    assert n2 == n1

    # cap: write_rays_full subsamples above export_rays_max
    args.export_rays_max = 500
    run_trace.write_rays_full(tmp_path, g2, args, "synthetic")
    npz = np.load(tmp_path / "rays_full.npz", allow_pickle=True)
    import json as _json
    meta = _json.loads(str(npz["meta"]))
    for d in meta["detectors"].values():
        assert d["n_kept"] <= 500
        if d["n_total"] > 500:
            assert d["n_kept"] == 500
