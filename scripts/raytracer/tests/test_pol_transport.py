# =============================================================================
# test_pol_transport.py -- P2: parallel-transport Q matrix + honest
# retardance/diattenuation oracle tests (Yun, McClain & Chipman, "Three-
# dimensional polarization ray-tracing calculus II", Appl. Opt. 50, 2866
# (2011)). See raytracer/poltransport.py for the construction (Qmat = the
# running parallel-transported frame; Jmat = the running interface-
# convention cumulative Jones matrix; M = Q^T P recovers the physical
# retardance/diattenuation with the geometric spin removed).
#
# Geometry: single-interaction oracles (1, 3, 4) call Tracer.step() ONCE
# directly and inspect the returned children (no detector needed — this
# sidesteps detector-placement precision entirely). Oracle 2 (periscope)
# chains two step() calls through two real mirror bodies, since it needs
# the SECOND interaction's geometry to actually depend on the first
# reflection's direction.
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_pol_transport.py -q
# =============================================================================
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from raytracer.rays import RayBatch                        # noqa: E402
from raytracer.scene import Scene                           # noqa: E402
from raytracer.sources import sample_source                 # noqa: E402
from raytracer.tracer import Tracer, TraceConfig             # noqa: E402
from raytracer.optprops import load_optical_properties      # noqa: E402
from raytracer import fresnel as fr                          # noqa: E402
from raytracer import poltransport as pt                     # noqa: E402
from raytracer import cengine                                 # noqa: E402
from . import scenehelpers as sh                              # noqa: E402


# ---------------------------------------------------------------------------
# geometry helpers: arbitrarily oriented thin boxes (mirrors / tilted slabs)
# ---------------------------------------------------------------------------
def _rotated_box_faces(name, R, center, x0=0.0, x1=0.002, half=0.01):
    faces = sh.box_faces(name, x0, x1, half)
    center = np.asarray(center, dtype=float)
    out = []
    for f in faces:
        origin = R @ np.asarray(f["surface"]["origin"], dtype=float) + center
        normal = R @ np.asarray(f["surface"]["normal"], dtype=float)
        corners = [(R @ np.asarray(c, dtype=float) + center).tolist()
                  for c in f["trim_polylines_xyz"][0]]
        nf = dict(f)
        nf["surface"] = {"type": "plane", "origin": origin.tolist(),
                         "normal": normal.tolist()}
        nf["trim_polylines_xyz"] = [corners]
        out.append(nf)
    return out


def _fold_mirror_body(name, center, d_in, d_out, half=0.01,
                      material="aluminum"):
    """A thin box whose FRONT face specularly reflects d_in -> d_out
    exactly: the mirror normal bisects -d_in and d_out, i.e.
    n_hat = normalize(d_out - d_in) (verified: reflect_dir(d_in, n_hat)
    == d_out for this n_hat, for any unit d_in != d_out)."""
    d_in = np.asarray(d_in, dtype=float)
    d_out = np.asarray(d_out, dtype=float)
    n_target = d_out - d_in
    n_target = n_target / np.linalg.norm(n_target)
    R = pt.rotation_between(np.array([[-1.0, 0.0, 0.0]]),
                            n_target[None, :])[0]
    faces = _rotated_box_faces(name, R, center, half=half)
    return {"name": name, "label": name, "role": "optic",
           "material": material, "faces": faces}


def _tilted_slab_body(name, material, center, theta_deg, half=0.01,
                      thickness=0.004, **extra):
    """A slab box whose FRONT face normal is tilted by theta_deg (about z,
    in the x-y plane) from -x, i.e. a ray along +x hits it at theta_deg
    angle of incidence."""
    th = np.deg2rad(theta_deg)
    n_target = np.array([-np.cos(th), np.sin(th), 0.0])
    R = pt.rotation_between(np.array([[-1.0, 0.0, 0.0]]),
                            n_target[None, :])[0]
    faces = _rotated_box_faces(name, R, center, x0=0.0, x1=thickness,
                               half=half)
    body = {"name": name, "label": name, "role": "optic",
           "material": material, "faces": faces}
    body.update(extra)
    return body


def _scene(bodies, optprops=None):
    """Scene() requires >=1 active detector; every oracle here inspects
    Tracer.step() children directly, so this detector is a structural
    placeholder that no ray ever reaches (parked far off-axis)."""
    if optprops is None:
        optprops = load_optical_properties()
    bodies = list(bodies) + [sh.detector_body("_Placeholder", x=1.0,
                                              half=0.001)]
    model = sh.make_model(bodies)
    scene = Scene(model, optprops.matdb, optprops.coatings, optprops=optprops)
    return scene, optprops


def _tracer_and_batch(scene, rays=300, n_lambda=1, seed=1,
                      max_reflections=6):
    """A Tracer (no detectors — the oracles inspect step() children
    directly) + the primary batch with Qmat/Jmat initialized exactly as
    Tracer.run() would (poltransport.init_birth from the freshly-sampled
    birth (s_hat, dir))."""
    cfg = TraceConfig(rays=rays, n_lambda=n_lambda, seed=seed,
                      power_floor=1e-12, max_reflections=max_reflections,
                      pol_transport=True)
    tracer = Tracer(scene, cfg, {})
    rng = np.random.default_rng(seed)
    batches = [sample_source(scene, scene.bodies[b], s, i, cfg.rays,
                             cfg.n_lambda, rng, ledger=tracer.ledger)
              for i, (b, s) in enumerate(scene.sources)]
    assert len(batches) == 1
    b = batches[0]
    pt.init_birth(b)
    return tracer, b


def _M(batch):
    """M = Q^T P restricted to the transverse block, for every ray in
    `batch` (uses the ray's OWN current (s_hat, dir) as the arrival
    frame — see poltransport.py)."""
    O_final = pt.frame(batch.s_hat, batch.dir)
    Delta = np.einsum('nji,njk->nik', batch.Qmat, O_final)   # Q^T @ O_final
    return np.einsum('nij,njk->nik', Delta[:, :2, :2], batch.Jmat)


AL_LAM_NM = 700.0    # inside the aluminum table's 207-1240 nm range


def _mat_index(optprops, name, lam_nm):
    mat = optprops.matdb.get(name)
    return complex(mat.n_complex(np.array([lam_nm * 1e-9]))[0])


def _fresnel_45deg(optprops, lam_nm):
    # n1 = the SCENE's actual ambient index (air, not exactly vacuum) --
    # the analytic reference must match what the tracer itself used.
    n1 = _mat_index(optprops, "air", lam_nm)
    n2 = _mat_index(optprops, "aluminum", lam_nm)
    cos_i = np.array([np.cos(np.deg2rad(45.0))])
    rs, rp, ts, tp, ct = fr.fresnel_coeffs(cos_i, np.array([n1]),
                                          np.array([n2]))
    return rs[0], rp[0]


# ---------------------------------------------------------------------------
# Oracle 1: single 45-degree aluminum fold mirror
# ---------------------------------------------------------------------------
def test_oracle1_single_45deg_al_mirror():
    optprops = load_optical_properties()
    bodies = [
        sh.source_body(x=-0.02, half=0.001, power_mW=1.0, coherent=False,
                       polarization={"kind": "linear", "angle_deg": 30.0},
                       lambdac_nm=AL_LAM_NM),
        _fold_mirror_body("M1", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          (0.0, 1.0, 0.0)),
    ]
    scene, optprops = _scene(bodies, optprops)
    tracer, batch = _tracer_and_batch(scene)
    children = tracer.step(batch)
    refl = children.select(children.current_medium() < 0)
    assert len(refl) > 0

    # Q's own 3rd column IS the ray's current direction, by construction
    # (invariant of the parallel-transport accumulation) -- and the fold
    # rotates that direction by exactly 90 degrees.
    assert np.allclose(refl.Qmat[:, :, 2], refl.dir, atol=1e-10)
    assert np.max(np.abs(np.sum(refl.dir * batch.dir[0], axis=-1))) < 1e-9

    M = _M(refl)
    gamma, diatten, axis = pt.polar_decompose(M)
    # every ray hit the SAME flat mirror at the SAME angle -> identical M
    assert np.ptp(gamma) < 1e-9
    assert np.ptp(diatten) < 1e-9

    rs, rp = _fresnel_45deg(optprops, AL_LAM_NM)
    delta_analytic = abs(float(np.angle(rp) - np.angle(rs)))
    if delta_analytic > np.pi:
        delta_analytic = 2 * np.pi - delta_analytic
    assert abs(gamma[0] - delta_analytic) < 1e-6

    diatten_analytic = abs((abs(rs) ** 2 - abs(rp) ** 2)
                           / (abs(rs) ** 2 + abs(rp) ** 2))
    assert abs(diatten[0] - diatten_analytic) < 1e-6


# ---------------------------------------------------------------------------
# Oracle 2: two orthogonal folds (periscope) -- retardance sums, geometric
# image rotation shows up ONLY in Q, never as fake retardance in M.
# ---------------------------------------------------------------------------
def test_oracle2_periscope_orthogonal_folds():
    """Two identical 45-degree aluminum folds with ORTHOGONAL fold planes
    (x-y then y-z: a genuine 3-D image rotation, not a planar zigzag).

    Closed-form composition of two retarders (Chipman's homogeneous-
    retarder / Pauli parametrization) gives
      cos(Gamma_total/2) = cos(Gamma_A/2)cos(Gamma_B/2)
                          - sin(Gamma_A/2)sin(Gamma_B/2)*cos(phi)
    where phi is the angle between the two retarders' fast axes AFTER
    parallel transport. A NAIVE accumulation (reading retardance straight
    off Jmat, the interface-convention cumulative Jones matrix, without
    ever applying the Q correction) instead measures phi as if it were 0
    or pi purely from the RAW interface (s,p) bookkeeping — which for this
    exact geometry works out to phi_naive = pi (the entry rotation into
    mirror B's own interface frame is a full 90-degree swap of s and p),
    giving a bogus near-maximal retardance of pi even though the mirrors
    are identical and should leave the beam's retardance UNCHANGED by any
    reflection off two rotationally-symmetric interactions structured this
    way. The Q-corrected M uses the TRUE parallel-transported phi (0 here:
    Delta, the mismatch between Q's parallel transport and the interface
    convention, is itself the missing 90-degree twist) and reports the
    physically correct Gamma_total = |Gamma_A - Gamma_B| = 0 for two
    IDENTICAL mirrors — this exact contrast (pi if you skip Q, 0 once you
    apply it) is the regression the Q matrix exists to prevent.

    NOTE for future readers: a naive "retardance == SUM of the two
    mirrors' Fresnel retardances" claim is only correct for a SAME-PLANE
    (coplanar) fold pair, where phi == 0 (fast axes already aligned after
    parallel transport, so cos(Gamma_total/2) == cos((Gamma_A+Gamma_B)/2)
    reduces to a plain sum) -- e.g. two folds that zig-zag within one
    plane (x-y then y-x), not the ORTHOGONAL-plane periscope tested here.
    Do not "fix" this test back to a sum assertion without re-deriving phi
    for whatever geometry is in play."""
    optprops = load_optical_properties()
    bodies = [
        sh.source_body(x=-0.02, half=0.001, power_mW=1.0, coherent=False,
                       polarization={"kind": "linear", "angle_deg": 30.0},
                       lambdac_nm=AL_LAM_NM),
        _fold_mirror_body("M1", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          (0.0, 1.0, 0.0)),
        _fold_mirror_body("M2", (0.0, 0.02, 0.0), (0.0, 1.0, 0.0),
                          (0.0, 0.0, 1.0)),
    ]
    scene, optprops = _scene(bodies, optprops)
    tracer, batch = _tracer_and_batch(scene)
    c1 = tracer.step(batch)
    refl1 = c1.select(c1.current_medium() < 0)
    assert len(refl1) > 0
    assert np.allclose(refl1.dir, [0.0, 1.0, 0.0], atol=1e-6)

    c2 = tracer.step(refl1)
    refl2 = c2.select(c2.current_medium() < 0)
    assert len(refl2) > 0
    assert np.allclose(refl2.dir, [0.0, 0.0, 1.0], atol=1e-6)

    # Q captures the geometric image rotation: its own 3rd column is the
    # ray's actual current direction (parallel-transport invariant), and
    # Q is a genuine proper rotation (det +1, orthonormal).
    Qf = refl2.Qmat[0]
    assert np.allclose(Qf[:, 2], refl2.dir[0], atol=1e-8)
    assert abs(np.linalg.det(Qf) - 1.0) < 1e-9
    assert np.allclose(Qf.T @ Qf, np.eye(3), atol=1e-9)

    # Delta (Q vs the ray's actual interface-convention frame) is EXACTLY
    # a 90-degree rotation for this two-orthogonal-fold geometry -- the
    # quantitative "image rotation appears in Q" check.
    O_final = pt.frame(refl2.s_hat, refl2.dir)
    Delta = np.einsum('nji,njk->nik', refl2.Qmat, O_final)
    R90 = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert np.allclose(Delta[0, :2, :2], R90, atol=1e-6) \
        or np.allclose(Delta[0, :2, :2], -R90, atol=1e-6)

    M = _M(refl2)
    gamma, diatten, axis = pt.polar_decompose(M)
    assert np.ptp(gamma) < 1e-8
    # honest (Q-corrected): two IDENTICAL mirrors leave zero net retardance
    assert gamma[0] < 1e-6

    # THE regression: reading retardance straight off Jmat (skipping the Q
    # correction entirely) reports a bogus near-maximal retardance, purely
    # from the geometric axis swap between the two interfaces' own (s,p)
    # conventions -- proving Q is doing necessary, non-trivial work.
    gamma_naive, _, _ = pt.polar_decompose(refl2.Jmat)
    assert gamma_naive[0] > np.pi - 1e-6


# ---------------------------------------------------------------------------
# Oracle 3: ideal linear polarizer at theta -- diattenuation == 1, axis ==
# theta (relative to the ray's OWN birth (s_hat, p_hat) frame, unrotated by
# Q since this is a straight, unbent, normal-incidence beam).
# ---------------------------------------------------------------------------
def test_oracle3_ideal_polarizer_axis_and_diattenuation():
    axis_vec = np.array([0.0, 0.6, 0.8])          # unit, in the y-z plane
    bodies = [
        sh.source_body(x=-0.02, half=0.001, power_mW=1.0, coherent=False,
                       polarization={"kind": "linear", "angle_deg": 10.0},
                       lambdac_nm=550.0),
        sh.slab_body("Pol", "pmma", 0.0, 0.002, half=0.01,
                     polarizer="ideal_linear", polarizer_axis=list(axis_vec)),
    ]
    scene, optprops = _scene(bodies)
    tracer, batch = _tracer_and_batch(scene)
    children = tracer.step(batch)
    trans = children.select(children.current_medium() >= 0)
    assert len(trans) > 0

    # straight, unbent, normal-incidence beam -> Q accumulates NO rotation
    assert np.allclose(trans.Qmat[0], pt.frame(batch.s_hat[0], batch.dir[0]),
                       atol=1e-9)

    M = _M(trans)
    gamma, diatten, axis = pt.polar_decompose(M)
    assert np.ptp(diatten) < 1e-9
    assert diatten[0] > 1.0 - 1e-3          # ideal_linear: T_perp ~ 1e-6

    s_hat0 = batch.s_hat[0]
    p_hat0 = np.cross(batch.dir[0], s_hat0)
    axis_predicted = np.arctan2(np.dot(axis_vec, p_hat0),
                                np.dot(axis_vec, s_hat0))
    # compare mod pi (a linear axis has no distinguished sign)
    d = (axis[0] - axis_predicted + np.pi / 2.0) % np.pi - np.pi / 2.0
    assert abs(d) < 1e-3, (axis[0], axis_predicted)


# ---------------------------------------------------------------------------
# Oracle 4: plain uncoated refraction at 30 degrees -- retardance 0 (real
# Fresnel), diattenuation == the analytic |ts|/|tp| amplitude asymmetry.
# ---------------------------------------------------------------------------
def test_oracle4_refraction_30deg_zero_retardance():
    bodies = [
        sh.source_body(x=-0.02, half=0.001, power_mW=1.0, coherent=False,
                       polarization={"kind": "linear", "angle_deg": 20.0},
                       lambdac_nm=587.6),
        _tilted_slab_body("Slab", "bk7", (0.0, 0.0, 0.0), 30.0),
    ]
    scene, optprops = _scene(bodies)
    tracer, batch = _tracer_and_batch(scene)
    children = tracer.step(batch)
    trans = children.select(children.current_medium() >= 0)
    assert len(trans) > 0

    M = _M(trans)
    gamma, diatten, axis = pt.polar_decompose(M)
    assert np.ptp(diatten) < 1e-8
    assert gamma[0] < 1e-8       # real (lossless dielectric) Fresnel: no
                                 # retardance, only diattenuation

    n1 = _mat_index(optprops, "air", 587.6)
    n2 = _mat_index(optprops, "bk7", 587.6)
    cos_i = np.array([np.cos(np.deg2rad(30.0))])
    rs, rp, ts, tp, ct = fr.fresnel_coeffs(cos_i, np.array([n1]),
                                          np.array([n2]))
    diatten_analytic = abs((abs(ts[0]) ** 2 - abs(tp[0]) ** 2)
                           / (abs(ts[0]) ** 2 + abs(tp[0]) ** 2))
    assert abs(diatten[0] - diatten_analytic) < 1e-6


# ---------------------------------------------------------------------------
# Oracle 5: Qmat/Jmat slot lifecycle (select/concatenate NaN-fill), the
# differentials'/birth_pos's pattern.
# ---------------------------------------------------------------------------
def _mk_batch(n, transport=False):
    b = RayBatch(n)
    b.pos[:] = 0.0
    b.dir[:, 0] = 1.0
    b.s_hat[:, 2] = 1.0
    b.lam[:] = 500e-9
    b.Es[:] = 1.0
    if transport:
        pt.init_birth(b)
    return b


def test_pol_transport_absent_by_default():
    b = _mk_batch(4)
    assert b.Qmat is None and b.Jmat is None


def test_pol_transport_select_copies():
    b = _mk_batch(5, transport=True)
    sel = b.select(np.array([0, 2, 4]))
    assert sel.Qmat is not None and sel.Jmat is not None
    assert np.array_equal(sel.Qmat, b.Qmat[[0, 2, 4]])
    assert np.array_equal(sel.Jmat, b.Jmat[[0, 2, 4]])
    sel.Qmat[0, 0, 0] = -999.0
    assert b.Qmat[0, 0, 0] != -999.0            # independent copy


def test_pol_transport_mixed_concat_nan_fills():
    a = _mk_batch(3, transport=True)
    c = _mk_batch(2, transport=False)
    out = RayBatch.concatenate([a, c])
    assert out.Qmat is not None and out.Jmat is not None
    assert np.array_equal(out.Qmat[:3], a.Qmat)
    assert np.all(np.isnan(out.Qmat[3:]))
    assert np.all(np.isnan(out.Jmat[3:].real))
    assert np.all(np.isnan(out.Jmat[3:].imag))


def test_pol_transport_all_absent_concat_none():
    out = RayBatch.concatenate([_mk_batch(3), _mk_batch(2)])
    assert out.Qmat is None and out.Jmat is None


def test_kill_nan_fills_both():
    b = _mk_batch(4, transport=True)
    pt.kill(b)
    assert np.all(np.isnan(b.Qmat))
    assert np.all(np.isnan(b.Jmat.real)) and np.all(np.isnan(b.Jmat.imag))


# ---------------------------------------------------------------------------
# routing: --pol-transport forces the Python engine (feature token)
# ---------------------------------------------------------------------------
def _fake_args(**over):
    base = dict(rough_fresnel="micro", particles=None,
               particle_threshold=None, ray_differentials=False,
               export_rays=False, ghost_analysis=False, viz_pattern=None,
               save_fields=False, pol_transport=False)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_pol_transport_feature_token_forces_python():
    scene, _ = _scene([
        sh.source_body(x=-0.02, half=0.001),
        sh.slab_body("Glass", "bk7", 0.0, 0.002, half=0.01),
    ])
    feats_off = cengine.detect_features(_fake_args(pol_transport=False),
                                        scene)
    feats_on = cengine.detect_features(_fake_args(pol_transport=True),
                                       scene)
    assert "pol_transport" not in feats_off
    assert "pol_transport" in feats_on


# ---------------------------------------------------------------------------
# cross-feature: --pol-transport with --importance-scatter (P2 integration).
# The importance sampler re-sites the ABg lobes into _emit_scatter_side with
# THREE child populations (aimed NEE children, full-lobe remainder, and the
# BTDF transmitted side) — all must carry valid transport per the shipped
# real-transport policy for ABg lobes (no NaN Q/J on any diffuse child in a
# non-birefringent scene), and closure must stay exact.
# ---------------------------------------------------------------------------
def _rect_detector(name, centre, normal, up, half):
    """Single-face detector plane (copied from test_scatter_importance_btdf:
    lets the detector sit OFF the specular axis so it sees scatter only)."""
    c = np.asarray(centre, float)
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    up = np.asarray(up, float)
    up = up - np.dot(up, n) * n
    up = up / np.linalg.norm(up)
    right = np.cross(up, n)
    corners = [c - half * right - half * up, c + half * right - half * up,
               c + half * right + half * up, c - half * right + half * up]
    face = {"id": "%s.Pad.Face1" % name,
            "surface": {"type": "plane", "origin": list(c),
                        "normal": list(n)},
            "orientation_outward": True, "area_m2": float((2 * half) ** 2),
            "fingerprint": {}, "mesh_stl": "",
            "trim_polylines_xyz": [[list(p) for p in corners]]}
    return {"name": name, "label": name, "role": "detector",
            "detector": {"face": face["id"]}, "faces": [face]}


def test_pol_transport_with_importance_scatter():
    import common
    from raytracer.detector import DetectorGrid
    optprops = load_optical_properties()
    slab = sh.slab_body("Win", "bk7", 0.0, 0.002, half=0.02)
    # BRDF + BTDF scatter row => _emit_scatter_side runs on BOTH sides
    slab["scatter_faces"] = {"Win.Pad.Face2": "lightly_ground_glass_window"}
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False, half=0.001,
                       polarization={"kind": "linear", "angle_deg": 25.0}),
        slab,
        # off-axis detector: sees scattered (incl. aimed) children only
        _rect_detector("Stray", (0.0, 0.06, 0.0), (0.0, -1.0, 0.0),
                       (0.0, 0.0, 1.0), 0.02),
        sh.detector_body(x=0.1, half=0.05),
    ])
    common.validate_model(model)
    scene = Scene(model, optprops.matdb, optprops.coatings,
                  optprops=optprops)
    grids = {fid: DetectorGrid(scene.faces[fid], 96, 8, (500e-9, 760e-9),
                               label=scene.faces[fid].id)
             for fid in scene.detector_faces}
    cfg = TraceConfig(rays=8000, n_lambda=1, seed=5, power_floor=1e-12,
                      importance_scatter=True, pol_transport=True,
                      export_rays=True)
    tracer = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(5)
    batches = [sample_source(scene, scene.bodies[b], s, i, cfg.rays,
                             cfg.n_lambda, rng, ledger=tracer.ledger,
                             export_rays=True)
              for i, (b, s) in enumerate(scene.sources)]
    result = tracer.run(batches)

    # closure stays exact with both features on
    rep = result.ledger.report(result.source_names)
    for name, srep in rep["sources"].items():
        assert srep["closure_error"] < 1e-3, (name, srep["closure_error"])

    # the off-axis detector saw scattered children (aimed and/or lobes)
    stray = next(g for g in grids.values() if g.label.startswith("Stray"))
    recs = stray.ray_records
    assert recs, "no scattered rays reached the off-axis detector"
    scat_flags = np.concatenate([r["scattered"] for r in recs])
    assert np.any(scat_flags)

    # NaN semantics: NO ray in this non-birefringent scene may carry NaN
    # transport (the shipped ABg policy is REAL transport on every diffuse
    # population — aimed NEE children, full-lobe remainder, BTDF side);
    # and every carried Q must still be a proper rotation whose 3rd column
    # is the ray's own arrival direction.
    for det in grids.values():
        for r in det.ray_records:
            Q = r["Qmat"]
            J = r["Jmat"]
            assert np.all(np.isfinite(Q)), det.label
            assert np.all(np.isfinite(J.real)) and np.all(
                np.isfinite(J.imag)), det.label
            assert np.allclose(
                np.einsum('nji,njk->nik', Q, Q), np.eye(3), atol=1e-9)
            assert np.allclose(Q[:, :, 2], r["dir"], atol=1e-9)
