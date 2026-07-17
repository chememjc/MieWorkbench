# =============================================================================
# test_sequential_preview.py -- P4b PREVIEW UNIFICATION (engine3.md Sec.5 /
# Sec.15 P4b: "preview unified"): tests for core/sequential_preview.py, the
# in-process Optiland fast path that core/raypreview.py tries before falling
# back to the general (subprocess) Python-engine preview chain.
#
# Run under env/bin/python (the GUI venv, which has Optiland + numpy):
#   QT_QPA_PLATFORM=offscreen env/bin/python -m pytest \
#       mieworkbench/tests/test_sequential_preview.py -q
#
# THE PHYSICS-CONSISTENCY GATE (the point of unification): the module docstring
# promises "the physics you preview is the physics you run". Two tiers:
#   (1) FAST, always-run, no MC dependency: trace_pupil_world_path's final
#       vertex reproduces trace_pupil_world's endpoint EXACTLY (same
#       underlying Optiland call; a regression pin on the new multi-segment
#       function against the pre-existing, P4a-oracle-validated one).
#   (2) @pytest.mark.slow, gated on the C engine + optics env being present
#       (mirrors test_optiland_sequential.py's own pattern): traces the SAME
#       fan -- the C engine's own MC-generated direct-ray birth positions --
#       through BOTH the sequential fast path and the real MC engine, and
#       asserts the final landing position agrees to < 1e-9 m. This is the
#       literal "same fan through both paths" gate: it reuses
#       optiland_oracle_support.py's already-adjudicated harness (ambient
#       index, two-plane focus, ... -- see test_optiland_oracle.py's
#       ADJUDICATIONS block) rather than re-deriving a fresh comparison
#       against the Python engine's own branching (Fresnel-ghost) viz trace,
#       whose reflected/transmitted children share a source_id and cannot be
#       disambiguated from the outside without duplicating the tracer's own
#       physics.
# =============================================================================
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("optiland")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "mieworkbench" / "tests"))
sys.path.insert(0, str(REPO))

from raytracer import optiland_bridge as ob            # noqa: E402
from mieworkbench.core import sequential_preview as sp  # noqa: E402


def _geo(stem):
    return REPO / "geometry" / stem


def _has(stem):
    return (_geo(stem) / "model.json").exists()


BRIDGEABLE_SCENES = ["lens_dcx", "beam_expander"]


# --------------------------------------------------------------------------
# build() end-to-end: writes a valid rays.vtp, reports the right engine
# --------------------------------------------------------------------------
@pytest.mark.parametrize("stem", BRIDGEABLE_SCENES)
def test_build_writes_valid_vtp(tmp_path, stem):
    if not _has(stem):
        pytest.skip("geometry/%s absent" % stem)
    out = tmp_path / "rays.vtp"
    ok, engine = sp.build(_geo(stem), out, pattern="fan:n=5")
    assert ok, engine
    assert engine == sp.ENGINE_SEQUENTIAL
    assert out.exists()
    text = out.read_text()
    assert "<VTKFile" in text and "PolyData" in text
    assert 'NumberOfLines="0"' not in text.split("\n")[3]


@pytest.mark.parametrize("stem", BRIDGEABLE_SCENES)
def test_build_only_bodies_still_bridgeable(tmp_path, stem):
    """--only-bodies filtering (mirrors preview_rays.py's own contract) does
    not break the fast path: sources/detectors are always kept."""
    if not _has(stem):
        pytest.skip("geometry/%s absent" % stem)
    with open(_geo(stem) / "model.json") as fh:
        model = json.load(fh)
    optic_labels = [b["label"] for b in model["bodies"] if b["role"] == "optic"]
    out = tmp_path / "rays.vtp"
    ok, engine = sp.build(_geo(stem), out, pattern="fan:n=5",
                          only_bodies=optic_labels)
    assert ok, engine
    assert out.exists()


# --------------------------------------------------------------------------
# out-of-scope scenes fall back cleanly (never raise)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("stem", ["ktp_walkoff", "calcite_displacer",
                                  "waveplate_quartz", "wollaston"])
def test_build_returns_false_on_crystal_scenes(tmp_path, stem):
    """Task requirement: fallback engages on an unbridgeable scene (any
    crystal scene). build() must return (False, reason), never raise."""
    if not _has(stem):
        pytest.skip("geometry/%s absent" % stem)
    out = tmp_path / "rays.vtp"
    ok, reason = sp.build(_geo(stem), out, pattern="fan:n=5")
    assert ok is False
    assert isinstance(reason, str) and reason
    assert not out.exists()


def test_build_returns_false_on_missing_geometry(tmp_path):
    out = tmp_path / "rays.vtp"
    ok, reason = sp.build(tmp_path / "nonexistent", out, pattern="fan:n=5")
    assert ok is False
    assert "model.json" in reason


def test_build_returns_false_never_raises_on_garbage_model_json(tmp_path):
    """The exact shape of the GUI's stub-based offline tests (a malformed/
    minimal model.json from an early extract stage) -- build() must degrade
    to a plain (False, reason), never propagate an exception."""
    geom = tmp_path / "geom"
    geom.mkdir()
    (geom / "model.json").write_text(json.dumps({"stub": True}))
    out = tmp_path / "rays.vtp"
    ok, reason = sp.build(geom, out, pattern="fan:n=5")
    assert ok is False
    assert isinstance(reason, str) and reason


# --------------------------------------------------------------------------
# GATE (1): trace_pupil_world_path pins against the pre-existing,
# P4a-oracle-validated trace_pupil_world (fast, deterministic, no MC)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("stem", BRIDGEABLE_SCENES)
def test_trace_pupil_world_path_matches_trace_pupil_world(stem):
    if not _has(stem):
        pytest.skip("geometry/%s absent" % stem)
    system = ob.load_sequential_system(_geo(stem))
    opt = ob.build_optic(system)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epd_mm = float(opt.paraxial.EPD())
    rng = np.random.default_rng(0)
    n = 64
    r = np.sqrt(rng.random(n)) * (epd_mm / 2.0 / ob.MM_PER_M) * 0.95
    th = 2.0 * np.pi * rng.random(n)
    py = r * np.cos(th)
    pz = r * np.sin(th)
    lam_um = system.wavelength_nm * 1e-3

    ref_pos, _ref_dir = ob.trace_pupil_world(
        opt, py, pz, epd_mm, lam_um, system.image_z_m)

    src = np.column_stack([np.full(n, system.surfaces[0].vertex_x_m - 0.01),
                           py, pz])
    path, valid = ob.trace_pupil_world_path(opt, py, pz, epd_mm, lam_um, src)
    assert np.all(valid)
    new_pos = path[:, -1, :]
    assert np.allclose(new_pos, ref_pos, atol=1e-9), \
        np.max(np.abs(new_pos - ref_pos))


# --------------------------------------------------------------------------
# GATE (2): the literal "same fan through both paths" gate, < 1e-9 m
# --------------------------------------------------------------------------
def _mc_gate_available():
    try:
        import optiland_oracle_support as H
    except Exception:
        return False
    return H.mc_available()


@pytest.mark.slow
@pytest.mark.skipif(not _mc_gate_available(),
                    reason="C engine / optics env unavailable")
@pytest.mark.parametrize("stem", BRIDGEABLE_SCENES)
def test_sequential_preview_matches_mc_engine_landing(tmp_path, stem):
    """The physics-consistency gate the task requires: trace the SAME fan
    (the C engine's own MC-generated direct-ray birth positions) through
    BOTH the sequential fast path (trace_pupil_world_path, the function
    core/sequential_preview.py calls) and the real MC/C engine, and assert
    the final landing position agrees to < 1e-9 m -- pinning that the
    preview path cannot silently drift from "the physics you run"."""
    import optiland_oracle_support as H

    if not _has(stem):
        pytest.skip("geometry/%s absent" % stem)
    npz = H.run_cengine_export(stem, tmp_path / stem, rays=20000, seed=11)
    birth, pos, _dirn, _det_meta = H.load_mc_direct_bundle(npz)
    assert len(birth) >= 8, "too few direct rays exported"

    system = ob.load_sequential_system(_geo(stem))
    lam_um = system.wavelength_nm * 1e-3
    rho_m = np.sqrt(birth[:, 1] ** 2 + birth[:, 2] ** 2)
    epd_mm = 2.0 * float(rho_m.max()) * ob.MM_PER_M * (1.0 + 1e-6)
    opt = ob.build_optic(system, epd_mm=epd_mm)

    path, valid = ob.trace_pupil_world_path(
        opt, birth[:, 1], birth[:, 2], epd_mm, lam_um, birth)
    assert np.all(valid), "sequential trace unexpectedly vignetted a direct ray"
    seq_landing = path[:, -1, :]

    resid = np.linalg.norm(seq_landing - pos, axis=1)
    print("\n%s: max landing residual %.3e m over %d rays"
          % (stem, float(resid.max()), len(resid)))
    assert float(resid.max()) < 1e-9, (stem, float(resid.max()))
