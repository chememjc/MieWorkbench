# =============================================================================
# test_time_core.py — pulsed-optics Phase P2: the engine time-domain core.
#
# Covers:
#   * RayBatch gopl/gdd_acc optional slots (alloc_time, select copies,
#     mixed concatenate NaN-fills) + the mandatory n_g_eff slot.
#   * Slab group delay: gopl - geometric path == (n_g - 1) * L for a
#     fused-silica slab, against Material.n_group (1e-9 relative).
#   * CW impulse-response semantics: gopl = 0 at birth; gopl == d after
#     propagating d in ambient air (group reference is vacuum-like 1.0).
#   * Calcite o/e group split: e-children freeze n_g_eff ==
#     birefringence.n_group_e_theta at the traced internal wavevector
#     angle; accumulated group path per unit ray length matches it; the
#     o and e group paths through the crystal differ.
#   * Biaxial (KTP) sheet children carry distinct positive n_g_eff.
#   * Bit-identity: opl / detector cubes / ledger / viz identical with
#     track_time on vs off (np.array_equal, not allclose).
#   * Children inherit gopl (Fresnel split: no reset at an interface).
#   * path_tally: per-body power-weighted bulk path lands under the body
#     label and matches power * L for a single-pass slab.
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_time_core.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import common                                              # noqa: E402
from raytracer.rays import RayBatch                        # noqa: E402
from raytracer.scene import Scene                          # noqa: E402
from raytracer.sources import sample_source                # noqa: E402
from raytracer.tracer import Tracer, TraceConfig           # noqa: E402
from raytracer.detector import DetectorGrid                # noqa: E402
from raytracer.optprops import load_optical_properties     # noqa: E402
from raytracer import birefringence as bir                 # noqa: E402
from raytracer.materials import gdd_per_length             # noqa: E402
from . import scenehelpers as sh                           # noqa: E402


# --------------------------------------------------------------------------- #
# harness: build Scene + Tracer with track_time control
# --------------------------------------------------------------------------- #
def _build(model, rays=400, n_lambda=1, seed=5, track_time=True,
           export=False, power_floor=1e-9, max_reflections=6):
    common.validate_model(model)
    opt = load_optical_properties()
    scene = Scene(model, opt.matdb, opt.coatings, optprops=opt)
    cfg = TraceConfig(rays=rays, n_lambda=n_lambda, seed=seed,
                      power_floor=power_floor, viz_rays=50,
                      export_rays=export, track_time=track_time,
                      max_reflections=max_reflections)
    grids = {fid: DetectorGrid(scene.faces[fid], 64, 8,
                               (400e-9, 900e-9), label=scene.faces[fid].id)
             for fid in scene.detector_faces}
    tracer = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(seed)
    batches = [sample_source(scene, scene.bodies[b], s, i, cfg.rays,
                             cfg.n_lambda, rng, ledger=tracer.ledger,
                             export_rays=export)
               for i, (b, s) in enumerate(scene.sources)]
    return scene, tracer, batches, grids


def _trace(model, **kw):
    scene, tracer, batches, grids = _build(model, **kw)
    result = tracer.run(batches)
    return result, grids, scene, tracer


def _det_records(grids):
    """Concatenated per-ray record columns of the (single) detector."""
    det = next(iter(grids.values()))
    assert det.ray_records, "no detector ray records (export off?)"
    return {k: np.concatenate([r[k] for r in det.ray_records])
            for k in det.ray_records[0]}


def _slab_model(material="fused_silica", L=0.01, det_x=0.03, **extra):
    return sh.make_model([
        sh.source_body(x=-0.02, half=0.001, coherent=False),
        sh.slab_body("Slab", material, 0.0, L, half=0.01, **extra),
        sh.detector_body(x=det_x, half=0.02),
    ])


# --------------------------------------------------------------------------- #
# 1. RayBatch slot lifecycle (mirrors the birth_pos contract)
# --------------------------------------------------------------------------- #
def _mk_batch(n, time=False):
    b = RayBatch(n)
    b.pos[:] = np.arange(3 * n).reshape(n, 3)
    b.lam[:] = 500e-9
    b.Es[:] = 1.0
    if time:
        b.alloc_time()
        b.gopl[:] = np.arange(n) + 1.0
        b.gdd_acc[:] = (np.arange(n) + 1.0) * 1e-27
    return b


def test_time_slots_default_none_alloc_zero():
    b = RayBatch(5)
    assert b.gopl is None and b.gdd_acc is None
    b.alloc_time()
    assert b.gopl.dtype == np.float64 and b.gdd_acc.dtype == np.float64
    assert b.gopl.shape == (5,) and b.gdd_acc.shape == (5,)
    assert np.all(b.gopl == 0.0) and np.all(b.gdd_acc == 0.0)
    # n_g_eff is MANDATORY, zero-initialized like n_eff
    assert b.n_g_eff.dtype == np.float64
    assert np.all(b.n_g_eff == 0.0)


def test_time_slots_select_copies():
    b = _mk_batch(5, time=True)
    b.n_g_eff[:] = np.arange(5) * 0.1
    sel = b.select(np.array([0, 2, 4]))
    assert np.array_equal(sel.gopl, b.gopl[[0, 2, 4]])
    assert np.array_equal(sel.gdd_acc, b.gdd_acc[[0, 2, 4]])
    assert np.array_equal(sel.n_g_eff, b.n_g_eff[[0, 2, 4]])
    sel.gopl[0] = -999.0                      # independent copy, not a view
    assert b.gopl[0] != -999.0
    # absent stays None
    assert RayBatch(3).select(np.array([1])).gopl is None


def test_time_slots_mixed_concat_nan_fills():
    a = _mk_batch(3, time=True)
    c = _mk_batch(2, time=False)
    out = RayBatch.concatenate([a, c])
    assert np.array_equal(out.gopl[:3], a.gopl)
    assert np.array_equal(out.gdd_acc[:3], a.gdd_acc)
    assert np.all(np.isnan(out.gopl[3:])) and np.all(np.isnan(out.gdd_acc[3:]))
    # all-absent -> stays None
    both = RayBatch.concatenate([_mk_batch(2), _mk_batch(2)])
    assert both.gopl is None and both.gdd_acc is None


# --------------------------------------------------------------------------- #
# 2. slab group delay against Material.n_group
# --------------------------------------------------------------------------- #
def test_slab_group_delay_fused_silica():
    L = 0.01
    result, grids, scene, _ = _trace(_slab_model(L=L), rays=300,
                                     track_time=True, export=True)
    rep = result.ledger.report(result.source_names)
    assert rep["closure_ok"]
    recs = _det_records(grids)
    assert "gopl" in recs and "gdd_acc" in recs
    prim = recs["generation"] == 0            # no internal reflections
    assert np.count_nonzero(prim) > 100
    geom = np.linalg.norm(recs["pos"][prim] - recs["birth_pos"][prim],
                          axis=-1)
    lam = recs["lam"][prim]
    silica = scene.matdb.get("fused_silica")
    n_g = silica.n_group(lam)
    extra = recs["gopl"][prim] - geom
    want = (n_g - 1.0) * L
    assert np.allclose(extra, want, rtol=1e-9, atol=0.0), \
        "max rel err %.3g" % np.max(np.abs(extra / want - 1.0))
    # and the accumulated GDD is the material's phi2/L times the glass path
    want_gdd = gdd_per_length(silica, lam) * L
    assert np.allclose(recs["gdd_acc"][prim], want_gdd, rtol=1e-9)
    assert np.all(want_gdd > 0)               # silica: normal dispersion


# --------------------------------------------------------------------------- #
# 3. CW impulse-response semantics in ambient air
# --------------------------------------------------------------------------- #
def test_gopl_zero_at_birth_and_equals_distance_in_air():
    model = sh.make_model([
        sh.source_body(x=-0.02, half=0.001, coherent=False),
        sh.detector_body(x=0.03, half=0.02),
    ])
    scene, tracer, batches, grids = _build(model, rays=200,
                                           track_time=True, export=True)
    b = batches[0]
    b.alloc_time()
    assert np.all(b.gopl == 0.0) and np.all(b.gdd_acc == 0.0)
    tracer.run(batches)
    recs = _det_records(grids)
    geom = np.linalg.norm(recs["pos"] - recs["birth_pos"], axis=-1)
    # ambient group index is exactly 1.0: gopl == geometric distance
    assert np.allclose(recs["gopl"], geom, rtol=1e-12, atol=0.0)
    assert np.allclose(recs["gopl"], 0.05, rtol=1e-12)
    assert np.all(recs["gdd_acc"] == 0.0)     # no dispersion in ambient


# --------------------------------------------------------------------------- #
# 4. calcite o/e group split (uniaxial directional group index)
# --------------------------------------------------------------------------- #
def test_calcite_oe_group_split():
    L = 0.01
    c45 = np.sqrt(0.5)
    model = sh.make_model([
        sh.source_body(x=-0.02, half=0.0005, coherent=False,
                       polarization={"kind": "linear", "angle_deg": 45.0}),
        sh.slab_body("Calcite", "calcite", 0.0, L, half=0.01,
                     crystal_axis=[c45, c45, 0.0]),
        sh.detector_body(x=0.03, half=0.02),
    ])
    scene, tracer, batches, _ = _build(model, rays=6, track_time=True)
    b = batches[0]
    b.alloc_time()
    c1 = tracer.step(b)                       # source -> crystal entry
    med1 = c1.current_medium()
    e_mask = c1.pol_mode == 1
    o_mask = (c1.pol_mode == 0) & (med1 >= 0)
    assert np.any(e_mask) and np.any(o_mask)
    # entry gopl: 20 mm of ambient, inherited by every child (no reset)
    assert np.allclose(c1.gopl, 0.02, rtol=1e-12)

    mo, me = scene.matdb.get_uniaxial("calcite")
    axis = np.array([c45, c45, 0.0])

    e = c1.select(e_mask)
    assert np.all(e.n_g_eff > 0.0)
    n_o = np.real(mo.n_complex(e.lam))
    n_e = np.real(me.n_complex(e.lam))
    # internal wavevector recovered from the traced RAY direction
    k_int = bir.k_from_ray(e.dir, axis, n_o, n_e)
    cos_kc = np.sum(k_int * axis, axis=-1)
    ng_want = bir.n_group_e_theta(cos_kc, mo, me, e.lam)
    assert np.allclose(e.n_g_eff, ng_want, rtol=1e-6)
    # o-children stay on the scalar medium path (n_g_eff sentinel 0)
    o = c1.select(o_mask)
    assert np.all(o.n_g_eff == 0.0)

    # ---- through the crystal: group path per unit ray length ----
    e_pos0, e_gopl0 = e.pos.copy(), e.gopl.copy()
    c2e = tracer.step(e)
    oute = c2e.select(c2e.current_medium() == -1)   # transmitted, in order
    assert len(oute) == len(e)
    seg_e = np.linalg.norm(oute.pos - e_pos0, axis=-1)
    dg_e = (oute.gopl - e_gopl0) / seg_e
    assert np.allclose(dg_e, e.n_g_eff, rtol=1e-12)
    assert np.allclose(dg_e, ng_want, rtol=1e-6)
    # e exit resets the directional group index with n_eff
    assert np.all(oute.n_g_eff == 0.0) and np.all(oute.n_eff == 0.0)

    o_pos0, o_gopl0 = o.pos.copy(), o.gopl.copy()
    c2o = tracer.step(o)
    outo = c2o.select(c2o.current_medium() == -1)
    assert len(outo) == len(o)
    seg_o = np.linalg.norm(outo.pos - o_pos0, axis=-1)
    dg_o = (outo.gopl - o_gopl0) / seg_o
    assert np.allclose(dg_o, mo.n_group(o.lam), rtol=1e-12)
    # the two modes accumulate DIFFERENT group path per unit length
    assert np.all(np.abs(dg_e - dg_o[:len(dg_e)]) > 1e-3)
    # bulk GDD uses the o material for both modes (documented approx)
    assert np.allclose(oute.gdd_acc,
                       gdd_per_length(mo, oute.lam) * seg_e, rtol=1e-9)


# --------------------------------------------------------------------------- #
# 5. biaxial (KTP) sheets carry distinct directional group indices
# --------------------------------------------------------------------------- #
def test_biaxial_sheets_freeze_group_index():
    c45 = np.sqrt(0.5)
    model = sh.make_model([
        sh.source_body(x=-0.01, half=0.0002, coherent=False),
        sh.slab_body("KTP", "ktp", 0.0, 0.015, half=0.008,
                     crystal_axis=[c45, 0.0, c45],
                     crystal_axis2=[0.0, 1.0, 0.0]),
        sh.detector_body(x=0.02, half=0.01),
    ])
    scene, tracer, batches, _ = _build(model, rays=6, track_time=True)
    b = batches[0]
    b.alloc_time()
    c1 = tracer.step(b)
    slow = c1.select(c1.pol_mode == 2)
    fast = c1.select(c1.pol_mode == 3)
    assert len(slow) and len(fast)
    assert np.all(slow.n_g_eff > 1.0) and np.all(fast.n_g_eff > 1.0)
    # distinct sheets -> distinct group indices, both >= their own n - a
    # weak sanity bound (normal dispersion at 633 nm: n_g > n_phase)
    assert np.all(np.abs(slow.n_g_eff - fast.n_g_eff) > 1e-4)
    assert np.all(slow.n_g_eff > slow.n_eff)
    assert np.all(fast.n_g_eff > fast.n_eff)


# --------------------------------------------------------------------------- #
# 6. bit-identity with track_time off vs on
# --------------------------------------------------------------------------- #
def test_bit_identity_track_time_on_off():
    kw = dict(rays=2000, seed=11, export=True)
    r_off, g_off, _, t_off = _trace(_slab_model(), track_time=False, **kw)
    r_on, g_on, _, t_on = _trace(_slab_model(), track_time=True, **kw)

    # per-ray records: opl (and everything else) bit-identical
    off, on = _det_records(g_off), _det_records(g_on)
    for k in ("opl", "pos", "dir", "lam", "power", "generation"):
        assert np.array_equal(off[k], on[k]), "record %r perturbed" % k
    assert "gopl" not in off and "gdd_acc" not in off
    assert "gopl" in on and "gdd_acc" in on

    # detector spectral cubes bit-identical
    for fid in g_off:
        assert np.array_equal(g_off[fid].inc, g_on[fid].inc)

    # ledger bit-identical (buckets, emitted, diagnostics)
    assert np.array_equal(r_off.ledger.emitted, r_on.ledger.emitted)
    for bk in r_off.ledger.buckets:
        assert np.array_equal(r_off.ledger.buckets[bk],
                              r_on.ledger.buckets[bk]), bk
    assert r_off.ledger.by_surface == r_on.ledger.by_surface
    assert r_off.ledger.by_body == r_on.ledger.by_body
    assert r_off.ledger.flux == r_on.ledger.flux
    assert r_off.ledger.detected == r_on.ledger.detected

    # viz store (rays.npy contract) bit-identical
    assert np.array_equal(r_off.viz.as_array(), r_on.viz.as_array())

    # off run: zero footprint (no tally, no slots anywhere)
    assert t_off.path_tally == {}
    assert t_on.path_tally != {}


# --------------------------------------------------------------------------- #
# 7. children inherit the parent's gopl at a split
# --------------------------------------------------------------------------- #
def test_children_inherit_gopl_at_fresnel_split():
    model = _slab_model(material="bk7", L=0.004)
    scene, tracer, batches, _ = _build(model, rays=50, track_time=True)
    b = batches[0]
    b.alloc_time()
    d1 = 0.02                                  # source -> slab entry
    c1 = tracer.step(b)
    med = c1.current_medium()
    refl = c1.select((med == -1))              # Fresnel-reflected children
    trans = c1.select((med >= 0))              # refracted into the slab
    assert len(refl) and len(trans)
    # BOTH children continue from the parent's accumulated group path —
    # no reset to 0 at the interface
    assert np.allclose(refl.gopl, d1, rtol=1e-12)
    assert np.allclose(trans.gopl, d1, rtol=1e-12)
    assert np.all(refl.gopl >= d1 - 1e-15)
    assert np.all(trans.gopl >= d1 - 1e-15)


# --------------------------------------------------------------------------- #
# 8. path_tally: per-body power-weighted bulk path
# --------------------------------------------------------------------------- #
def test_path_tally_single_pass_slab():
    L = 0.01
    result, _, _, tracer = _trace(_slab_model(L=L), rays=1000,
                                  track_time=True)
    assert set(tracer.path_tally) == {"Slab"}
    assert result.path_tally is tracer.path_tally
    got = tracer.path_tally["Slab"]
    emitted = float(np.sum(result.ledger.emitted))
    # dominant term: (power entering the slab) * L; entry Fresnel loss at
    # normal incidence on silica is ~3.5%, internal double-bounces add a
    # small positive tail
    assert 0.90 * emitted * L < got < 1.02 * emitted * L
    # track_time off -> empty (pinned again here for the tally specifically)
    _, _, _, t2 = _trace(_slab_model(L=L), rays=200, track_time=False)
    assert t2.path_tally == {}
