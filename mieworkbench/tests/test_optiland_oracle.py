# =============================================================================
# test_optiland_oracle.py -- the P4a PARITY ORACLE (engine3.md Sec.5 / Sec.15
# P4a): Optiland (sequential) vs the MieWorkbench C engine (non-sequential
# Monte-Carlo) on GEOMETRIC ground truths that both compute exactly. This
# arbiter MUST exist before any Optiland-based optimizer/designer (P4b) adds a
# second physics truth -- "an arbiter exists before a second physics truth
# does" (engine3.md Sec.15 P4a).
#
# Run under env/bin/python (the GUI venv, which has Optiland + numpy):
#   QT_QPA_PLATFORM=offscreen env/bin/python -m pytest \
#       mieworkbench/tests/test_optiland_oracle.py -q
# The oracle cases additionally shell out to the optics env's C engine
# (/home3/optics/env + cengine/build/miewb-trace); absent -> those cases skip.
# The bridge structural + unit-contract tests need only Optiland + a
# geometry/<stem>/model.json cache.
#
# Deterministic-fan ROUTE (documented per the task): the MieWorkbench side uses
# the C engine's --export-rays output of a small coherent=false Monte-Carlo run
# (run_trace.py). Each exported ray's landing is EXACT geometry given its birth
# point (the source is planar -> perfectly collimated, no divergence -- see
# raytracer/sources.py); the only stochastic thing is WHERE in the pupil each
# ray is born. Optiland is then traced at those SAME per-ray pupil positions,
# so every metric is a clean engine-vs-engine comparison over one identical
# input ray set. The GUI's --viz-pattern "fan" overlay was NOT reused: it emits
# a fixed cardinal cross from the source for 3D preview and is not a
# controllable field/height fan headlessly; --export-rays gives the full
# deterministic pupil->image map an aberration oracle needs.
#
# ================= ADJUDICATIONS (engine3.md: correctness is not conformance)
# Two disagreements were found beyond the naive "both are exact geometry"
# tolerance and each was chased to root with an INDEPENDENT exact vector-Snell
# meridional trace (and the repo's lensmaker/thick-lens pins) as the third
# arbiter. NEITHER engine's geometry is wrong; both were reconciled in the
# BRIDGE/HARNESS, and the C engine was NOT modified (task contract):
#
#   (A) Ambient index. First pass showed the C-engine landing radius short of
#       the analytic by ~1e-3 (few um), growing with pupil height. Root cause:
#       the C engine traces in the scene's REAL ambient -- the materials
#       registry's "air" is n=1.000272 at 633 nm, not vacuum -- while the
#       bridge initially used Optiland's default n=1.0 "air". The implied index
#       offset (n_air-1 = 2.72e-4) matched the discrepancy exactly. Fix:
#       optiland_bridge resolves the ambient ("air") through the SAME registry
#       and installs it as the object-space + gap index. After the fix the
#       C-engine landing matches the analytic to ~0.1 um (MC binning floor) and
#       Optiland matches it to <1 nm.
#
#   (B) Reported image-space direction. Best-focus derived from each engine's
#       reported ray DIRECTION differed by ~4 um. Root: Optiland's *reported*
#       image-surface direction cosines carry an (n_air-1)-scale artifact in
#       the transverse slope (a reduced/normalized-direction convention at the
#       image surface); its ACTUAL geometry is exact -- proven because its
#       LANDING at any plane matches the analytic and the C engine to machine
#       precision, and its best focus computed from POSITIONS at two image
#       planes (convention-free) equals the C engine's direction-derived focus
#       to <1 nm. The C engine's reported direction matches the analytic. Fix:
#       the harness computes Optiland's focus from two-plane positions; the C
#       engine keeps its (correct) reported direction.
#
# After (A)+(B), all three ground truths -- best-focus z, per-ray landing
# position across the full pupil fan, and spot RMS -- agree to floating-point
# round-off (~1e-16 m) on all four scenes, INCLUDING the SF5/BK7 Cooke triplet.
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("optiland")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "mieworkbench" / "tests"))
sys.path.insert(0, str(REPO / "scripts"))

import optiland_oracle_support as H          # noqa: E402
from raytracer import optiland_bridge as ob   # noqa: E402

SCENES = H.SCENES                              # pcx, dcx, beam_expander, triplet


def _has_geometry(stem):
    return (REPO / "geometry" / stem / "model.json").exists()


# --------------------------------------------------------------------------
# unit contract -- the ONE place metres<->millimetres is pinned
# --------------------------------------------------------------------------
def test_unit_contract_constant():
    """model.json is SI metres; Optiland is millimetres; MM_PER_M is the single
    conversion and it is exactly 1000."""
    assert ob.MM_PER_M == 1000.0


@pytest.mark.skipif(not _has_geometry("lens_pcx"),
                    reason="geometry/lens_pcx cache absent (run extract)")
def test_unit_contract_lengths_convert_once():
    """Every length crossing the bridge is scaled by MM_PER_M exactly once: a
    25 mm (0.025 m) front radius and a 65 mm detector become 25 and 65 in the
    Optiland model, and the paraxial focal length lands in the mm regime."""
    system = ob.load_sequential_system(REPO / "geometry" / "lens_pcx")
    front = system.surfaces[0]
    assert abs(front.radius_m - 0.025) < 1e-9                  # metres in
    assert abs(front.radius_m * ob.MM_PER_M - 25.0) < 1e-9     # mm out
    assert abs(system.image_z_m * ob.MM_PER_M - 65.0) < 1e-6
    opt = ob.build_optic(system, epd_mm=8.0)
    assert 40.0 < float(opt.paraxial.f2()) < 60.0              # mm, not m or um


# --------------------------------------------------------------------------
# bridge structural round-trips (Optiland only; no MC needed)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _has_geometry("lens_pcx"),
                    reason="geometry/lens_pcx cache absent")
def test_bridge_pcx_signs_and_focus():
    """Plano-convex: front sphere R=+25 mm (centre downstream), flat back;
    thick-lens focus matches the lensmaker equation to 0.1% (the repo's own
    invariant), cross-checking the signed-radius + ambient-index mapping."""
    system = ob.load_sequential_system(REPO / "geometry" / "lens_pcx")
    assert len(system.surfaces) == 2
    front, back = system.surfaces
    assert abs(front.radius_m * ob.MM_PER_M - 25.0) < 1e-6
    assert back.is_flat
    assert front.material_after == "bk7" and back.material_after == "air"
    # lensmaker (thick, PCX in air): 1/f = (n-1)/R1  (R2 = inf)
    n = ob.resolve_index("bk7", system.wavelength_nm)
    n_air = ob.resolve_index("air", system.wavelength_nm)
    f_lensmaker = 1.0 / ((n / n_air - 1.0) * (1.0 / 25.0))
    f_opt = float(ob.build_optic(system, epd_mm=8.0).paraxial.f2())
    assert abs(f_opt - f_lensmaker) / f_lensmaker < 1e-3, (f_opt, f_lensmaker)


@pytest.mark.skipif(not _has_geometry("camera_triplet"),
                    reason="geometry/camera_triplet cache absent")
def test_bridge_triplet_drops_stop_keeps_glass():
    """The Cooke triplet: the aluminium iris annulus AND its air plug are
    dropped (not refracting surfaces); exactly the three glass elements
    (BK7 / SF5 / BK7, six surfaces) survive, with the SF5 flint biconcave
    (front R<0, back R>0)."""
    system = ob.load_sequential_system(REPO / "geometry" / "camera_triplet")
    bodies = [s.body_label for s in system.surfaces]
    assert set(bodies) == {"L1", "L2", "L3"}
    assert len(system.surfaces) == 6
    assert "aluminum" not in {s.material_after for s in system.surfaces}
    l2 = [s for s in system.surfaces if s.body_label == "L2"]
    assert l2[0].radius_m < 0 and l2[1].radius_m > 0     # biconcave flint
    assert l2[0].material_after == "sf5"


def test_bridge_rejects_unsupported():
    """A non-sequential feature (a tilted plane) is a hard BridgeUnsupported,
    never a silent wrong answer."""
    with pytest.raises(ob.BridgeUnsupported):
        ob._plane_vertex({"surface": {"type": "plane", "origin": [0, 0, 0],
                                      "normal": [0.7, 0.7, 0.0]}})


# --------------------------------------------------------------------------
# THE ORACLE: Optiland vs the C engine on geometric ground truths
# --------------------------------------------------------------------------
_TOL_FOCUS_M = 1e-6      # task 1 um (met with ~1e10 margin: machine precision)
_TOL_LAND_M = 1e-9      # task 1e-6 m (met with ~1e7 margin)
_TOL_SPOT_REL = 1e-6    # task 1e-4 (met with ~1e9 margin)


@pytest.mark.slow
@pytest.mark.skipif(not H.mc_available(),
                    reason="C engine / optics-env python unavailable")
@pytest.mark.parametrize("stem", SCENES)
def test_oracle_optiland_vs_cengine(stem, tmp_path):
    """Adjudicated parity on best-focus z, per-ray landing position (the full
    pupil fan -> marginal + chief + every height), and spot RMS. See the module
    docstring for the two adjudications (ambient index; reported-direction
    convention) that reduced the disagreement from a few um to machine
    precision. All four scenes now agree to floating-point round-off."""
    if not _has_geometry(stem):
        pytest.skip("geometry/%s cache absent (run extract_geometry)" % stem)
    r = H.run_oracle(stem, tmp_path, rays=20000)
    print("\n" + H._fmt(r))

    # (1) per-ray landing position across the full pupil fan
    assert r["land_resid_max_m"] < _TOL_LAND_M, r
    # (2) spot RMS at the detector
    assert r["spot_reldiff"] < _TOL_SPOT_REL, r
    # (3) best-focus axial position
    assert abs(r["focus_delta_m"]) < _TOL_FOCUS_M, r
