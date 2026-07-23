# =============================================================================
# test_ghosts.py — Phase-4 ghost / stray-light analysis (--ghost-analysis).
#
# Covers:
#   * RayBatch.refl_hist optional slot: select copies it, mixed concatenate
#     -1-fills the batches that lack it, all-absent stays None (the birth_pos/
#     _DIFF_SLOTS lifecycle).
#   * Two-uncoated-BK7-slab oracle: the strongest generation-2 ghost path's
#     detected power == direct-beam power * R^2 (the enumerated 2-bounce
#     Fresnel product), the top path is one of the three degenerate dominant
#     paths, the gap ghost (slab2.front -> slab1.back) is present, closure OK,
#     and the post render_ghost_analysis report block agrees.
#   * History cap: a ~200-bounce etalon does not overflow refl_hist (slot 7
#     reused, no crash).
#   * Flag OFF: track_history=False leaves refl_hist None and rays_full.npz
#     carries no refl_hist key (tracer output otherwise untouched).
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_ghosts.py -q
# =============================================================================
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import common                                              # noqa: E402
import run_trace                                           # noqa: E402
import post_process                                        # noqa: E402
from raytracer.rays import RayBatch, HIST_DEPTH            # noqa: E402
from raytracer.scene import Scene                          # noqa: E402
from raytracer.sources import sample_source               # noqa: E402
from raytracer.tracer import Tracer, TraceConfig          # noqa: E402
from raytracer.detector import DetectorGrid               # noqa: E402
from raytracer.optprops import load_optical_properties    # noqa: E402
from . import scenehelpers as sh                           # noqa: E402


# --------------------------------------------------------------------------- #
# 1. RayBatch.refl_hist optional-slot lifecycle
# --------------------------------------------------------------------------- #
def _mk_hist_batch(n, hist=False, fill=0):
    b = RayBatch(n)
    b.lam[:] = 500e-9
    b.Es[:] = 1.0
    if hist:
        b.alloc_history()
        b.refl_hist[:, 0] = fill        # a distinguishable first-slot value
    return b


def test_refl_hist_select_copies():
    b = _mk_hist_batch(5, hist=True, fill=7)
    sel = b.select(np.array([0, 2, 4]))
    assert sel.refl_hist is not None
    assert sel.refl_hist.shape == (3, HIST_DEPTH)
    assert np.array_equal(sel.refl_hist, b.refl_hist[[0, 2, 4]])
    # independent copy, not a view
    sel.refl_hist[0, 0] = -999
    assert b.refl_hist[0, 0] == 7


def test_refl_hist_absent_stays_none():
    b = _mk_hist_batch(4, hist=False)
    sel = b.select(np.array([1, 3]))
    assert sel.refl_hist is None


def test_refl_hist_mixed_concat_fills_minus_one():
    a = _mk_hist_batch(3, hist=True, fill=2)
    c = _mk_hist_batch(2, hist=False)
    out = RayBatch.concatenate([a, c])
    assert out.refl_hist is not None
    assert out.refl_hist.dtype == np.int32
    assert np.array_equal(out.refl_hist[:3], a.refl_hist)
    assert np.all(out.refl_hist[3:] == -1)          # batch lacking it -> -1


def test_refl_hist_all_absent_concat_none():
    out = RayBatch.concatenate([_mk_hist_batch(3), _mk_hist_batch(2)])
    assert out.refl_hist is None


# --------------------------------------------------------------------------- #
# Ghost trace helper
# --------------------------------------------------------------------------- #
def _trace_ghost(model, rays=4000, seed=5, max_reflections=6,
                 track_history=True, export=True, power_floor=1e-12,
                 resolution=128):
    common.validate_model(model)
    opt = load_optical_properties()
    scene = Scene(model, opt.matdb, opt.coatings, optprops=opt)
    grids = {fid: DetectorGrid(scene.faces[fid], resolution, 8,
                               (500e-9, 750e-9), label=scene.faces[fid].id)
             for fid in scene.detector_faces}
    cfg = TraceConfig(rays=rays, n_lambda=1, seed=seed,
                      power_floor=power_floor,
                      max_reflections=max_reflections,
                      export_rays=export, track_history=track_history)
    tr = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(seed)
    batches = [sample_source(scene, scene.bodies[b], s, i, cfg.rays,
                             cfg.n_lambda, rng, ledger=tr.ledger,
                             export_rays=export)
               for i, (b, s) in enumerate(scene.sources)]
    res = tr.run(batches)
    return res, grids, scene


def _stack(records, key):
    return np.concatenate([r[key] for r in records]) if records \
        else np.zeros(0)


def _face_labels(scene):
    return ["%s.%s" % (scene.body_of_face(i).label,
                       scene.faces[i].id.rsplit(".", 1)[-1])
            for i in range(len(scene.faces))]


# --------------------------------------------------------------------------- #
# 2. Two-uncoated-BK7-slab ghost oracle
# --------------------------------------------------------------------------- #
def _two_slab_model():
    return sh.make_model([
        sh.source_body("Src", x=-0.02, half=0.003, power_mW=10.0,
                       lambdac_nm=633.0, coherent=False),
        sh.slab_body("Glass1", "bk7", 0.000, 0.004, half=0.01),
        sh.slab_body("Glass2", "bk7", 0.008, 0.012, half=0.01),
        sh.detector_body("Det", x=0.03, half=0.01),
    ])


def test_two_slab_ghost_oracle():
    res, grids, scene = _trace_ghost(_two_slab_model(), rays=6000, seed=11)
    det = next(iter(grids.values()))
    recs = det.ray_records
    assert recs, "no detector ray records"

    gen = _stack(recs, "generation").astype(int)
    scat = _stack(recs, "scattered").astype(bool)
    power = _stack(recs, "power")
    hist = np.concatenate([r["refl_hist"] for r in recs])

    # normal-incidence Fresnel R at the air/BK7 interface (633 nm)
    g1 = next(b for b in scene.bodies if b.label == "Glass1")
    n_glass = float(np.real(scene.medium_index(g1.index,
                                               np.array([633e-9]))[0]))
    R = ((n_glass - 1.0) / (n_glass + 1.0)) ** 2

    # direct (gen 0) beam power reaching the detector: P_emit * T^4. The
    # dominant gen-2 ghost adds exactly two air/BK7 reflections (R each) and
    # the same four transmissions, so its power is direct_power * R^2 --
    # independent of the emitted power and (to first order) of bulk
    # absorption, which cancels between the two.
    direct_power = float(np.sum(power[gen == 0]))
    assert direct_power > 0
    expected_ghost = direct_power * R ** 2

    # group generation-2 specular detector hits by their face-id path
    groups = {}
    cand = (gen >= 2) & (~scat)
    for i in np.where(cand)[0]:
        sig = tuple(int(x) for x in hist[i] if x >= 0)
        if len(sig) >= 2:
            groups.setdefault(sig, {"p": 0.0, "n": 0})
            groups[sig]["p"] += float(power[i])
            groups[sig]["n"] += 1
    assert groups, "no generation-2 ghost paths recorded"
    ranked = sorted(groups.items(), key=lambda kv: kv[1]["p"], reverse=True)

    fl = _face_labels(scene)

    def path(sig):
        return " -> ".join(fl[f] for f in sig)

    # the three degenerate dominant 2-bounce paths (all == P*T^4*R^2):
    #   gap ghost      : slab2 front (Glass2.Face1) then slab1 back (Glass1.Face2)
    #   slab2 internal : Glass2.Face2 then Glass2.Face1
    #   slab1 internal : Glass1.Face2 then Glass1.Face1
    dominant = {
        "Glass2.Face1 -> Glass1.Face2",
        "Glass2.Face2 -> Glass2.Face1",
        "Glass1.Face2 -> Glass1.Face1",
    }
    top_sig, top_val = ranked[0]
    assert len(top_sig) == 2, "top ghost must be a 2-reflection path"
    assert path(top_sig) in dominant, path(top_sig)
    # top ghost power matches the enumerated Fresnel product within MC tol
    assert top_val["p"] == pytest.approx(expected_ghost, rel=0.05), \
        (top_val["p"], expected_ghost, R)

    # the specifically-enumerated gap ghost is present with the same power
    gap = {path(s): v for s, v in groups.items()}
    assert "Glass2.Face1 -> Glass1.Face2" in gap
    assert gap["Glass2.Face1 -> Glass1.Face2"]["p"] == pytest.approx(
        expected_ghost, rel=0.05)

    # energy closure holds
    rep = res.ledger.report([scene.bodies[b].label for b, _ in scene.sources])
    assert rep["closure_ok"], rep["sources"]


def test_two_slab_ghost_render(tmp_path):
    res, grids, scene = _trace_ghost(_two_slab_model(), rays=6000, seed=11)
    det = next(iter(grids.values()))
    recs = det.ray_records
    gen = _stack(recs, "generation").astype(int)
    scat = _stack(recs, "scattered").astype(bool)
    power = _stack(recs, "power")
    hist = np.concatenate([r["refl_hist"] for r in recs])
    cols = {"pos": _stack(recs, "pos"), "generation": gen,
            "scattered": scat, "power": power, "refl_hist": hist}
    dm = {"label": det.label, "xhat": list(det.xhat), "yhat": list(det.yhat)}
    report = {"detectors": {det.label: {}}}
    fl = _face_labels(scene)
    safe = det.label.replace(".", "_")

    class _CSV:
        def __init__(self):
            self.emitted = []

        def emit(self, filename, header, rows, **kw):
            self.emitted.append((filename, header, rows, kw))

    csv = _CSV()
    post_process.render_ghost_analysis(safe, dm, cols, tmp_path, report,
                                       fl, csv)
    ghosts = report["detectors"][det.label]["ghosts"]
    assert ghosts["n_paths"] >= 1
    assert 0.0 < ghosts["ghost_fraction"] < 1.0
    # top row is a 2-reflection path with a nonzero fraction
    top = ghosts["top"][0]
    assert top["ghost_order"] == 2
    assert top["detected_W"] > 0
    # images + CSV produced
    assert (tmp_path / ("ghost_table_%s.png" % safe)).exists()
    assert (tmp_path / ("ghost_footprint_%s_1.png" % safe)).exists()
    assert csv.emitted and csv.emitted[0][0] == "ghost_table_%s.csv" % safe


# --------------------------------------------------------------------------- #
# 3. History cap: a many-bounce etalon reuses slot 7 without overflow
# --------------------------------------------------------------------------- #
def test_history_cap_no_overflow():
    # one strongly-reflecting slab -> a long internal etalon reflection
    # ladder; max_reflections well past HIST_DEPTH forces slot-7 reuse.
    model = sh.make_model([
        sh.source_body("Src", x=-0.02, half=0.002, coherent=False),
        sh.slab_body("Etalon", "bk7", 0.0, 0.004, half=0.01, mirror=0.9),
        sh.detector_body("Det", x=0.03, half=0.01),
    ])
    res, grids, scene = _trace_ghost(model, rays=800, seed=3,
                                     max_reflections=210,
                                     power_floor=1e-30)
    det = next(iter(grids.values()))
    recs = det.ray_records
    assert recs
    hist = np.concatenate([r["refl_hist"] for r in recs])
    gen = _stack(recs, "generation").astype(int)
    assert hist.shape[1] == HIST_DEPTH
    # the ladder climbs well past the slot count ...
    assert int(gen.max()) > HIST_DEPTH
    # ... and every deep ray has slot 7 populated (reused, never overflowed)
    deep = gen >= HIST_DEPTH
    assert np.any(deep)
    assert np.all(hist[deep, HIST_DEPTH - 1] >= 0)
    # closure still holds through the deep ladder
    rep = res.ledger.report([scene.bodies[b].label for b, _ in scene.sources])
    assert rep["closure_ok"], rep["sources"]


# --------------------------------------------------------------------------- #
# 4. Flag OFF: no history allocated, no npz key
# --------------------------------------------------------------------------- #
def test_flag_off_leaves_refl_hist_none():
    model = _two_slab_model()
    _, grids, _ = _trace_ghost(model, rays=2000, seed=7,
                               track_history=False, export=True)
    det = next(iter(grids.values()))
    assert det.ray_records
    # export ran, but with history OFF the records carry no refl_hist field
    assert all("refl_hist" not in r for r in det.ray_records)


def test_flag_off_no_npz_key(tmp_path):
    import types
    model = _two_slab_model()
    _, grids, scene = _trace_ghost(model, rays=2000, seed=7,
                                   track_history=False, export=True)
    args = types.SimpleNamespace(export_rays_max=1000000, seed0=5,
                                 max_reflections=6)
    run_trace.write_rays_full(tmp_path, grids, args, "m", scene=scene)
    z = np.load(tmp_path / "rays_full.npz", allow_pickle=True)
    assert not any(k.endswith("/refl_hist") for k in z.files)
    # face_labels still written (scene passed), so the meta is well-formed
    meta = json.loads(str(z["meta"]))
    assert "face_labels" in meta and len(meta["face_labels"]) > 0
