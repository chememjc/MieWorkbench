# =============================================================================
# test_coating_phase.py — P2: coating phase columns + phase-invalid flag.
#
# materials.py's table-coating loader now accepts OPTIONAL per-row phase
# columns (ars_deg, arp_deg, ats_deg, atp_deg — Zemax TABLE coating
# convention, s/p reflection+transmission phase angle in DEGREES) that are
# all-or-none; a table with them sets phase_valid=True and the tracer uses
# the table's own phase instead of borrowing the bare-interface Fresnel
# phase. Covers:
#   * loader all-or-none validation + phase_valid flag
#   * branch-cut-safe complex interpolation (optprops.interp_phase_deg)
#   * the tracer actually APPLIES the table phase (a real two-beam
#     interferometer built from box-on-the-x-axis synthetic geometry, no
#     FreeCAD needed — see scenehelpers.py)
#   * C-engine routing: a phase_valid table forces Python (coating_phase
#     token, unported)
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import common                                                   # noqa: E402
from raytracer.materials import MaterialDB, MaterialError, _load_coating_table  # noqa: E402
from raytracer.optprops import interp_phase_deg                 # noqa: E402
from raytracer.tests import scenehelpers as sh                  # noqa: E402
from raytracer import cengine                                   # noqa: E402


# ---------------------------------------------------------------------------
# 1. loader: all-or-none phase columns, phase_valid flag
# ---------------------------------------------------------------------------
def _write_table(tmp_path, name, header, rows):
    path = tmp_path / name
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def test_loader_no_phase_columns_is_phase_invalid(tmp_path):
    path = _write_table(
        tmp_path, "t1.csv", "wavelength_nm,Rs,Rp,Ts,Tp",
        ["400,0.5,0.5,0.5,0.5", "700,0.5,0.5,0.5,0.5"])
    out = _load_coating_table(path, "test")
    assert out["phase_valid"] is False
    assert "ars_deg" not in out


def test_loader_all_four_phase_columns_is_phase_valid(tmp_path):
    path = _write_table(
        tmp_path, "t2.csv",
        "wavelength_nm,Rs,Rp,Ts,Tp,ars_deg,arp_deg,ats_deg,atp_deg",
        ["400,0.5,0.5,0.5,0.5,10,20,30,40",
         "700,0.5,0.5,0.5,0.5,15,25,35,45"])
    out = _load_coating_table(path, "test")
    assert out["phase_valid"] is True
    assert out["ars_deg"].tolist() == [10.0, 15.0]
    assert out["atp_deg"].tolist() == [40.0, 45.0]


@pytest.mark.parametrize("missing", ["ars_deg", "arp_deg", "ats_deg", "atp_deg"])
def test_loader_partial_phase_columns_is_an_error(tmp_path, missing):
    cols = ["ars_deg", "arp_deg", "ats_deg", "atp_deg"]
    present = [c for c in cols if c != missing]
    header = "wavelength_nm,Rs,Rp,Ts,Tp," + ",".join(present)
    row_vals = ",".join(["1"] * len(present))
    path = _write_table(
        tmp_path, "t3.csv", header,
        ["400,0.5,0.5,0.5,0.5," + row_vals,
         "700,0.5,0.5,0.5,0.5," + row_vals])
    with pytest.raises(MaterialError, match="all-or-none|together or not at all"):
        _load_coating_table(path, "test")


# ---------------------------------------------------------------------------
# 2. branch-cut-safe complex interpolation
# ---------------------------------------------------------------------------
def test_interp_phase_deg_branch_cut_symmetric():
    """Two rows symmetric about the +/-180 branch cut (170 deg, -170 deg =
    190 deg): the TRUE continuous phase at the midpoint wavelength is 180
    deg exactly (by symmetry — both rows are 10 deg from the cut on
    opposite sides). A NAIVE linear average of the raw degree values,
    (170 + -170)/2 = 0 deg, is off by a full 180 deg — as wrong as
    possible. interp_phase_deg (complex/unit-vector interpolation) must
    land on 180 deg, not 0."""
    lam_tab = np.array([0.5, 0.6])
    phase_tab = np.array([170.0, -170.0])
    mid = np.array([0.55])
    got_rad = interp_phase_deg(mid, lam_tab, phase_tab, "test")
    got_deg = np.degrees(got_rad)[0]
    # wrap to (-180, 180] and compare against +/-180 (same point)
    wrapped = (got_deg + 180.0) % 360.0 - 180.0
    assert abs(abs(wrapped) - 180.0) < 1e-6, got_deg
    naive = (170.0 + -170.0) / 2.0
    assert abs(wrapped - naive) > 90.0        # nowhere near the naive (wrong) answer


def test_interp_phase_deg_matches_endpoints_exactly():
    lam_tab = np.array([0.4, 0.5, 0.7])
    phase_tab = np.array([12.0, -45.0, 178.0])
    got = np.degrees(interp_phase_deg(lam_tab, lam_tab, phase_tab, "test"))
    assert got == pytest.approx(phase_tab, abs=1e-9)


def test_interp_phase_deg_out_of_range_hard_errors():
    lam_tab = np.array([0.4, 0.7])
    phase_tab = np.array([0.0, 10.0])
    with pytest.raises(MaterialError, match="outside tabulated range"):
        interp_phase_deg(np.array([0.9]), lam_tab, phase_tab, "test")


# ---------------------------------------------------------------------------
# 3. tracer applies the table phase — a real two-beam interferometer
#
# Collinear, on-axis "common-path" interferometer built from ONLY
# axis-aligned box geometry (no 45-degree tilts needed):
#
#   Source (x=-Ls, coherent, linear pol) --> Plate (coated Face1 ONLY,
#   material 'air' so every OTHER face is a trivial index-matched no-op)
#   --> arm A: reflects straight back to the Detector (behind the source)
#       arm B: transmits through the Plate, hits a perfect Mirror
#       (mirror=1.0), reflects, transmits back OUT through the Plate,
#       continues to the same Detector.
#
# max_reflections=2 keeps EXACTLY these two paths (arm A: 1 reflection;
# arm B: 1 reflection at the mirror, transmissions don't count against the
# cap) — any higher-order cavity bounce (reflect again off the coated
# Face1 from inside) would need a 2nd reflection AFTER the mirror's, i.e.
# generation 2 at the point it's attempted, which the cap blocks (credited
# to truncated_generation, never reaches the detector). So the detector
# sees exactly a clean two-beam sum: E_A (one reflection off Face1) + E_B
# (one transmission through Face1, one mirror reflection, one transmission
# back through Face1 — TWO Face1 transmissions, so arm B picks up the
# table's transmission phase TWICE per round trip).
#
# Rather than reading detected INTENSITY off the detector grid (whose
# total is phase-INVARIANT by energy conservation, and whose per-pixel
# value mixes in a wavefront-curvature-mismatch spatial pattern), this
# reads the RAW per-arm coherent samples straight off the detector's
# gather-sample buffer (before the diffraction-integral gather runs) and
# sums each arm's complex field directly — exact (not Monte-Carlo-noisy):
# every ray in an arm picks up the identical phase shift from a phase
# change on Face1, so the arm's complex SUM rotates by exactly that shift,
# for any sample count.
# ---------------------------------------------------------------------------
LAM_NM = 633.0


def _interf_model(coatings_extra):
    """coatings_extra: the coatings dict (just 'phase_plate')."""
    src = sh.source_body(name="Src", x=-0.01, half=0.00002, power_mW=1.0,
                         lambdac_nm=LAM_NM, coherent=True,
                         polarization={"kind": "linear", "angle_deg": 0.0})
    plate = sh.slab_body("Plate", "air", 0.0, 0.00003, half=0.0001,
                         coating={"Plate.Pad.Face1": "phase_plate"})
    mirror = sh.slab_body("Mirror", "bk7", 0.0003, 0.00032, half=0.0001,
                          mirror=1.0)
    det_face = sh._rect_face(
        "Det.Pad.Face1", [-0.01, 0, 0], [1, 0, 0],
        [[-0.01, -0.002, -0.002], [-0.01, 0.002, -0.002],
         [-0.01, 0.002, 0.002], [-0.01, -0.002, 0.002]], (0.004) ** 2)
    det = {"name": "Det", "label": "Det", "role": "detector",
           "detector": {"face": det_face["id"]}, "faces": [det_face]}
    return sh.make_model([src, plate, mirror, det])


def _phase_valid_coating(ats_deg, ars_deg=0.0):
    lam_um = np.array([0.5, 0.7])
    Rs = Rp = 0.5
    Ts = Tp = 0.5
    return {"phase_plate": {
        "kind": "table", "lam_um": lam_um,
        "Rs": np.full(2, Rs), "Rp": np.full(2, Rp),
        "Ts": np.full(2, Ts), "Tp": np.full(2, Tp),
        "aoi_deg": 0.0, "reference": "synthetic test", "phase_valid": True,
        "ars_deg": np.full(2, ars_deg), "arp_deg": np.full(2, ars_deg),
        "ats_deg": np.full(2, ats_deg), "atp_deg": np.full(2, ats_deg),
    }}


def _phase_invalid_coating():
    lam_um = np.array([0.5, 0.7])
    Rs = Rp = 0.5
    Ts = Tp = 0.5
    return {"phase_plate": {
        "kind": "table", "lam_um": lam_um,
        "Rs": np.full(2, Rs), "Rp": np.full(2, Rp),
        "Ts": np.full(2, Ts), "Tp": np.full(2, Tp),
        "aoi_deg": 0.0, "reference": "synthetic test", "phase_valid": False,
    }}


def _run_and_split_arms(coatings, rays=20000, seed=7):
    """Trace the interferometer, return (rel_phase_deg, |E_A|, |E_B|,
    closure_error). rel_phase_deg = arg(sum_B) - arg(sum_A), wrapped to
    [0, 360)."""
    model = _interf_model(coatings)
    common.validate_model(model)
    from raytracer.scene import Scene
    from raytracer.sources import sample_source
    from raytracer.tracer import Tracer, TraceConfig
    from raytracer.detector import DetectorGrid

    db = MaterialDB.load()
    scene = Scene(model, db, coatings)
    grids = {fid: DetectorGrid(scene.faces[fid], 32, 4, (600e-9, 660e-9),
                               label=scene.faces[fid].id)
             for fid in scene.detector_faces}
    cfg = TraceConfig(rays=rays, n_lambda=1, seed=seed, max_reflections=2)
    tracer = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(seed)
    bidx, src = scene.sources[0]
    batch = sample_source(scene, scene.bodies[bidx], src, 0, cfg.rays, 1,
                          rng, ledger=tracer.ledger)
    result = tracer.run([batch])

    rep = result.ledger.report(result.source_names)
    closure_err = next(iter(rep["sources"].values()))["closure_error"]

    det = list(grids.values())[0]
    merged = det.merged_samples()
    s = merged[(0, 0, 0)]
    opl = s["opl"]
    mid = (opl.min() + opl.max()) / 2.0
    armA = opl < mid                    # short path: direct reflection
    armB = ~armA                        # long path: transmit-mirror-transmit
    assert armA.any() and armB.any()

    p_hat = np.cross(s["dir"], s["s_hat"])
    E3 = s["Es"][:, None] * s["s_hat"] + s["Ep"][:, None] * p_hat
    zhat = np.array([0.0, 0.0, 1.0])     # transverse axis, both arms ~ -x
    Ez = E3 @ zhat
    sumA = Ez[armA].sum()
    sumB = Ez[armB].sum()
    rel_deg = np.degrees(np.angle(sumB) - np.angle(sumA)) % 360.0
    return rel_deg, abs(sumA), abs(sumB), closure_err


def test_tracer_applies_table_phase_fringe_shift():
    """The defining P2 physics check: changing the phase-valid table's
    transmission phase (ats_deg/atp_deg) by DELTA shifts the two-beam
    interferometer's relative phase by exactly 2*DELTA (arm B crosses the
    coated face TWICE per round trip — once outbound, once on return —
    so it picks up the table's transmission phase twice)."""
    DELTA = 40.0
    rel0, ampA0, ampB0, err0 = _run_and_split_arms(_phase_valid_coating(0.0))
    rel1, ampA1, ampB1, err1 = _run_and_split_arms(_phase_valid_coating(DELTA))

    shift = (rel1 - rel0) % 360.0
    assert shift == pytest.approx(2.0 * DELTA, abs=1e-6)

    # |amplitude| (hence power) is untouched by a pure phase change --
    # phases don't move power, closure stays exact (P2 requirement).
    assert ampA1 == pytest.approx(ampA0, rel=1e-9)
    assert ampB1 == pytest.approx(ampB0, rel=1e-9)
    assert err0 < 1e-3 and err1 < 1e-3


def test_tracer_applies_table_phase_wraps_correctly_past_branch_cut():
    """Same check with a DELTA that pushes 2*ats_deg past +/-180 (the
    exact branch-cut regime the loader's complex interpolation exists
    for) -- the tracer-level effect must still track exactly."""
    rel_a, *_ = _run_and_split_arms(_phase_valid_coating(170.0))
    rel_b, *_ = _run_and_split_arms(_phase_valid_coating(-170.0))
    # ats_deg 170 -> -170 is a +20 deg step the short way (170 -> 180 ==
    # -180 -> -170); the tracer-level shift is 2x that input step (arm B
    # crosses the coated face twice per round trip) = 40 deg.
    diff = (rel_b - rel_a + 180.0) % 360.0 - 180.0
    assert abs(diff) == pytest.approx(40.0, abs=1e-6)


def test_tracer_phase_invalid_table_ignores_intended_phase():
    """A phase_valid=False table (no ars/arp/ats/atp_deg columns) can't
    carry a programmable phase at all -- the tracer falls back to the
    trivial bare-interface Fresnel phase for BOTH the reflected and
    transmitted amplitude (documented approximation), which for this
    scene's index-matched (air/air) coated face is exactly zero extra
    phase -- i.e. it reproduces the phase_valid=True, ats_deg=0 baseline,
    proving the phase-invalid path does NOT invent a coating-specific
    retardance."""
    rel_invalid, *_ = _run_and_split_arms(_phase_invalid_coating())
    rel_baseline, *_ = _run_and_split_arms(_phase_valid_coating(0.0))
    assert rel_invalid == pytest.approx(rel_baseline, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. C-engine routing: coating_phase forces Python
# ---------------------------------------------------------------------------
def _scene_for_routing(coatings):
    model = _interf_model(coatings)
    common.validate_model(model)
    from raytracer.scene import Scene
    db = MaterialDB.load()
    return Scene(model, db, coatings)


class _Args:
    engine = "auto"
    particles = None
    particle_threshold = None
    ray_differentials = False
    export_rays = False
    ghost_analysis = False
    viz_pattern = None
    save_fields = False
    rough_fresnel = "micro"
    gdd_budget = False


def test_coating_phase_token_detected():
    scene = _scene_for_routing(_phase_valid_coating(30.0))
    feats = cengine.detect_features(_Args(), scene)
    assert "coating_phase" in feats
    assert "coating_phase" not in cengine.PORTED


def test_coating_phase_not_detected_when_phase_invalid():
    scene = _scene_for_routing(_phase_invalid_coating())
    feats = cengine.detect_features(_Args(), scene)
    assert "coating" in feats
    assert "coating_phase" not in feats


def test_coating_phase_forces_python_routing(monkeypatch, tmp_path):
    fake_binary = tmp_path / "miewb-trace"
    fake_binary.write_text("#!/bin/sh\n")
    fake_binary.chmod(0o755)
    monkeypatch.setattr(cengine, "binary_path", lambda: fake_binary)

    scene = _scene_for_routing(_phase_valid_coating(30.0))
    engine, reason = cengine.choose_engine(_Args(), scene)
    assert engine == "python"
    assert "coating_phase" in reason
