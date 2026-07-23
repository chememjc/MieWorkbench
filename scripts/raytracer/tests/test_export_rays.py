# =============================================================================
# test_export_rays.py — Phase-3 --export-rays / spot-diagram / ray-fan feature.
#
# Covers:
#   * RayBatch.birth_pos optional slot: select copies it, mixed concatenate
#     NaN-fills the batches that lack it (the _DIFF_SLOTS lifecycle).
#   * rays.npy (13-col viz contract) is BIT-identical with export on vs off.
#   * PCX-lens focus oracle: exported spot RMS at the paraxial focus is far
#     smaller than 15 mm past it; the export ray count equals the detector's
#     incoherent tally count; Sum(power) equals the ledger detected power.
#   * write_rays_full subsample cap + meta kept_fraction bookkeeping.
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest raytracer/tests/test_export_rays.py -q
# =============================================================================
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import common                                              # noqa: E402
import run_trace                                           # noqa: E402
from raytracer.rays import RayBatch                        # noqa: E402
from raytracer.scene import Scene                          # noqa: E402
from raytracer.sources import sample_source               # noqa: E402
from raytracer.tracer import Tracer, TraceConfig          # noqa: E402
from raytracer.detector import DetectorGrid               # noqa: E402
from raytracer.optprops import load_optical_properties    # noqa: E402
from . import scenehelpers as sh                           # noqa: E402

GEO = SCRIPTS.parent / "geometry"


# --------------------------------------------------------------------------- #
# 1. RayBatch.birth_pos optional slot
# --------------------------------------------------------------------------- #
def _mk_batch(n, birth=False, x0=0.0):
    b = RayBatch(n)
    b.pos[:] = np.arange(3 * n).reshape(n, 3) + x0
    b.lam[:] = 500e-9
    b.Es[:] = 1.0
    if birth:
        b.birth_pos = b.pos.copy() + 100.0
    return b


def test_birth_pos_select_copies():
    b = _mk_batch(5, birth=True)
    sel = b.select(np.array([0, 2, 4]))
    assert sel.birth_pos is not None
    assert np.array_equal(sel.birth_pos, b.birth_pos[[0, 2, 4]])
    # independent copy, not a view
    sel.birth_pos[0, 0] = -999.0
    assert b.birth_pos[0, 0] != -999.0


def test_birth_pos_absent_stays_none():
    b = _mk_batch(4, birth=False)
    sel = b.select(np.array([1, 3]))
    assert sel.birth_pos is None


def test_birth_pos_mixed_concat_nan_fills():
    a = _mk_batch(3, birth=True)
    c = _mk_batch(2, birth=False)
    out = RayBatch.concatenate([a, c])
    assert out.birth_pos is not None
    assert np.array_equal(out.birth_pos[:3], a.birth_pos)
    assert np.all(np.isnan(out.birth_pos[3:]))     # batch lacking it -> NaN


def test_birth_pos_all_absent_concat_none():
    out = RayBatch.concatenate([_mk_batch(3), _mk_batch(2)])
    assert out.birth_pos is None


# --------------------------------------------------------------------------- #
# 2. rays.npy viz contract bit-identical with export on/off
# --------------------------------------------------------------------------- #
def _box_scene():
    bodies = [
        sh.source_body("Src", x=-0.02, half=0.002, coherent=False),
        sh.slab_body("Glass", "bk7", 0.0, 0.004, half=0.01),
        sh.detector_body("Det", x=0.03, half=0.01),
    ]
    model = sh.make_model(bodies)
    common.validate_model(model)
    opt = load_optical_properties()
    return Scene(model, opt.matdb, opt.coatings, optprops=opt)


def _trace(scene, export, rays=3000, seed=7):
    cfg = TraceConfig(rays=rays, n_lambda=1, seed=seed, power_floor=1e-9,
                      viz_rays=200, export_rays=export)
    grids = {fid: DetectorGrid(scene.faces[fid], 128, 8,
                               (400e-9, 700e-9), label=scene.faces[fid].id)
             for fid in scene.detector_faces}
    tr = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(seed)
    batches = [sample_source(scene, scene.bodies[b], s, i, cfg.rays,
                             cfg.n_lambda, rng, ledger=tr.ledger,
                             export_rays=export)
               for i, (b, s) in enumerate(scene.sources)]
    res = tr.run(batches)
    return res, grids, tr


def test_rays_npy_bit_identical_export_on_off():
    viz_off, _, _ = _trace(_box_scene(), export=False)
    viz_on, _, _ = _trace(_box_scene(), export=True)
    a = viz_off.viz.as_array()
    b = viz_on.viz.as_array()
    assert a.shape == b.shape and a.shape[1] == 13
    assert np.array_equal(a, b), "viz (rays.npy) perturbed by --export-rays"


# --------------------------------------------------------------------------- #
# 3. PCX-lens focus oracle (reuses geometry/lens_pcx)
# --------------------------------------------------------------------------- #
requires_pcx = pytest.mark.skipif(
    not (GEO / "lens_pcx" / "model.json").exists(),
    reason="author+extract lens_pcx (make_test_scenes.py + extract_geometry.py)")


def _lens_pcx_scene(det_x):
    """lens_pcx with the Screen box translated so its detecting plane sits at
    global x = det_x [m] (the injected collimated bundle below replaces the
    scene source, so the source is left untouched)."""
    model = common.load_model(GEO / "lens_pcx" / "model.json")
    dx = None
    for b in model["bodies"]:
        if b.get("role") == "detector":
            xs = [f["surface"]["origin"][0] for f in b["faces"]]
            dx = det_x - min(xs)               # detecting plane = min-x face
            for f in b["faces"]:
                f["surface"]["origin"][0] += dx
                f["trim_polylines_xyz"] = [
                    [[c[0] + dx, c[1], c[2]] for c in loop]
                    for loop in f["trim_polylines_xyz"]]
    assert dx is not None
    common.validate_model(model)
    opt = load_optical_properties()
    return Scene(model, opt.matdb, opt.coatings, optprops=opt,
                 geometry_dir=str(GEO / "lens_pcx"))


def _collimated_disk(rmax=2e-3, x0=-0.02, m=4000, seed=5, lam=633e-9,
                     export=True):
    rng = np.random.default_rng(seed)
    r = rmax * np.sqrt(rng.random(m))
    th = rng.uniform(0, 2 * np.pi, m)
    b = RayBatch(m)
    b.pos[:] = np.stack([np.full(m, x0), r * np.cos(th), r * np.sin(th)],
                        axis=-1)
    b.dir[:] = [1.0, 0.0, 0.0]
    b.s_hat[:] = [0.0, 0.0, 1.0]
    b.Es[:] = 1.0
    b.Ep[:] = 1.0
    b.lam[:] = lam
    b.birth_power[:] = b.power
    b.coherent[:] = False                      # incoherent -> direct deposit
    if export:
        b.birth_pos = b.pos.copy()
    return b


def _trace_bundle(scene, batch, export, seed=5):
    grids = {fid: DetectorGrid(scene.faces[fid], 256, 8,
                               (500e-9, 750e-9), label=scene.faces[fid].id)
             for fid in scene.detector_faces}
    cfg = TraceConfig(rays=len(batch), n_lambda=1, seed=seed,
                      power_floor=1e-9, export_rays=export)
    tr = Tracer(scene, cfg, grids)
    res = tr.run([batch])
    return res, grids, tr


def _spot_rms(det):
    """Geometric RMS spot radius of the PRIMARY (generation-0) exported rays
    in the detector grid frame. Filtering to gen 0 excludes the weak
    multi-bounce lens etalon ghosts (gens 2/4/6) that otherwise smear the
    spot — the export captures them faithfully, they are just not the focus
    metric here."""
    pos = np.concatenate([r["pos"] for r in det.ray_records])
    gen = np.concatenate([r["generation"] for r in det.ray_records])
    g0 = gen == 0
    u = pos[g0] @ det.xhat
    v = pos[g0] @ det.yhat
    uc, vc = u.mean(), v.mean()
    return float(np.sqrt(np.mean((u - uc) ** 2 + (v - vc) ** 2)))


@requires_pcx
def test_pcx_focus_spot_and_export_consistency():
    # paraxial focus of the extracted PCX lens: front vertex x=0, thickness
    # 5 mm, bfl 45.236 mm -> x_focus ~= 50.24 mm; a plane 15 mm further back
    # is strongly defocused.
    x_focus, x_defocus = 0.050236, 0.065
    _, grids_f, tr_f = _trace_bundle(
        _lens_pcx_scene(x_focus), _collimated_disk(), export=True)
    _, grids_d, _ = _trace_bundle(
        _lens_pcx_scene(x_defocus), _collimated_disk(), export=True)

    det_f = next(iter(grids_f.values()))
    det_d = next(iter(grids_d.values()))
    rms_f = _spot_rms(det_f)
    rms_d = _spot_rms(det_d)
    assert rms_f < 0.25 * rms_d, (rms_f, rms_d)     # focus << defocus

    # export ray count == the detector's incoherent tally count (all rays
    # incoherent here) ...
    n_export = sum(len(r["pos"]) for r in det_f.ray_records)
    n_inc = sum(det_f.detected_incoherent_n.values())
    assert n_export == n_inc and n_export > 0

    # ... and Sum(exported power) == ledger detected power for this seed
    p_export = sum(float(np.sum(r["power"])) for r in det_f.ray_records)
    p_ledger = tr_f.ledger.detected[det_f.label]
    assert p_export == pytest.approx(p_ledger, rel=1e-9)

    # birth_pos populated on every exported record (pupil coord present)
    bp = np.concatenate([r["birth_pos"] for r in det_f.ray_records])
    assert np.all(np.isfinite(bp))


# --------------------------------------------------------------------------- #
# 4. write_rays_full subsample cap + kept_fraction meta
# --------------------------------------------------------------------------- #
def _fake_args(**kw):
    d = dict(export_rays_max=100, seed0=42, max_reflections=6)
    d.update(kw)
    return types.SimpleNamespace(**d)


def _one_detector():
    bodies = [sh.source_body("Src", x=-0.02, half=0.002),
              sh.detector_body("Det", x=0.03, half=0.01)]
    model = sh.make_model(bodies)
    common.validate_model(model)
    opt = load_optical_properties()
    scene = Scene(model, opt.matdb, opt.coatings, optprops=opt)
    fid = next(iter(scene.detector_faces))
    det = DetectorGrid(scene.faces[fid], 64, 8, (400e-9, 700e-9),
                       label=scene.faces[fid].id)
    return {fid: det}, det


def test_write_rays_full_subsample_cap(tmp_path):
    grids, det = _one_detector()
    n = 550
    rng = np.random.default_rng(0)
    det.ray_records.append({
        "pos": rng.normal(size=(n, 3)), "dir": rng.normal(size=(n, 3)),
        "opl": rng.random(n), "lam": np.full(n, 550e-9),
        "source_id": np.zeros(n, np.int16),
        "lam_stratum": np.zeros(n, np.int16),
        "pol_stratum": np.zeros(n, np.int16),
        "generation": np.zeros(n, np.int16),
        "pol_mode": np.zeros(n, np.int8),
        "power": rng.random(n),
        "scattered": np.zeros(n, bool), "coherent": np.zeros(n, bool),
        "birth_pos": rng.normal(size=(n, 3))})

    args = _fake_args(export_rays_max=100)
    run_trace.write_rays_full(tmp_path, grids, args, "unitmodel")

    z = np.load(tmp_path / "rays_full.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    safe = det.label.replace(".", "_")
    dm = meta["detectors"][safe]
    assert dm["n_total"] == n
    assert dm["n_kept"] == 100
    assert dm["kept_fraction"] == pytest.approx(100 / n)
    assert z["%s/pos" % safe].shape == (100, 3)
    assert z["%s/birth_pos" % safe].shape == (100, 3)
    assert meta["seed"] == 42 and meta["model"] == "unitmodel"
    # grid basis round-tripped
    assert np.allclose(dm["xhat"], det.xhat)


def test_write_rays_full_no_cap(tmp_path):
    grids, det = _one_detector()
    n = 40
    det.ray_records.append({
        "pos": np.zeros((n, 3)), "dir": np.zeros((n, 3)),
        "opl": np.zeros(n), "lam": np.full(n, 550e-9),
        "source_id": np.zeros(n, np.int16),
        "lam_stratum": np.zeros(n, np.int16),
        "pol_stratum": np.zeros(n, np.int16),
        "generation": np.zeros(n, np.int16),
        "pol_mode": np.zeros(n, np.int8), "power": np.ones(n),
        "scattered": np.zeros(n, bool), "coherent": np.zeros(n, bool),
        "birth_pos": np.zeros((n, 3))})
    run_trace.write_rays_full(tmp_path, grids, _fake_args(), "m")
    z = np.load(tmp_path / "rays_full.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    dm = meta["detectors"][det.label.replace(".", "_")]
    assert dm["n_kept"] == n and dm["kept_fraction"] == 1.0
