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
# WP1: 13-column viz rows carry the cumulative optical path (opl0/opl1) so the
# bead animation has timing data. Before this, the sequential path emitted
# 11-column rows and beadanim reported "Loaded rays predate timing data (opl)".
# opl is Σ n·ds in metres (beadanim/preview_rays contract: t = opl/c); the
# traced polyline is V0=source -> V1..VS=system.surfaces -> V_{S+1}=image, and
# segment k (V[k]->V[k+1]) carries the DOWNSTREAM medium of V[k]: ambient for
# the source->first-surface leg, then each surface's material_after.
# --------------------------------------------------------------------------
def _read_polydata(path):
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def _read_segments(path):
    """Return (p0, p1, opl0, opl1) as numpy arrays, one row per line-cell
    segment, in the file's cell order (which is _build's per-ray, per-segment
    emit order: all of ray 0's segments, then ray 1's, ...)."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    pd = _read_polydata(path)
    cd = pd.GetCellData()
    opl0 = vtk_to_numpy(cd.GetArray("opl0")).astype(np.float64)
    opl1 = vtk_to_numpy(cd.GetArray("opl1")).astype(np.float64)
    conn = vtk_to_numpy(pd.GetLines().GetConnectivityArray())
    off = vtk_to_numpy(pd.GetLines().GetOffsetsArray())
    pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
    starts = off[:-1]
    p0 = pts[conn[starts]]
    p1 = pts[conn[starts + 1]]
    return p0, p1, opl0, opl1


@pytest.mark.parametrize("stem", BRIDGEABLE_SCENES)
def test_build_emits_opl_cell_arrays(tmp_path, stem):
    """(1) the written vtp carries opl0/opl1 cell arrays -- the timing data
    the bead animation requires."""
    if not _has(stem):
        pytest.skip("geometry/%s absent" % stem)
    out = tmp_path / "rays.vtp"
    ok, engine = sp.build(_geo(stem), out, pattern="fan:n=5")
    assert ok, engine
    pd = _read_polydata(out)
    cd = pd.GetCellData()
    assert cd.GetArray("opl0") is not None, "opl0 cell array missing"
    assert cd.GetArray("opl1") is not None, "opl1 cell array missing"


@pytest.mark.parametrize("stem", BRIDGEABLE_SCENES)
def test_opl_monotone_and_root_zero(tmp_path, stem):
    """(2) per-ray opl is monotone non-decreasing and the first segment of
    every ray starts at opl0 == 0 (a source-born root, opl = Σ n·ds from the
    emit point)."""
    if not _has(stem):
        pytest.skip("geometry/%s absent" % stem)
    out = tmp_path / "rays.vtp"
    ok, engine = sp.build(_geo(stem), out, pattern="fan:n=5")
    assert ok, engine
    _p0, _p1, opl0, opl1 = _read_segments(out)
    assert len(opl0) > 0
    # each segment is itself non-decreasing (n >= 1, ds >= 0)
    assert np.all(opl1 >= opl0 - 1e-15)
    # ray boundaries: a new ray begins at each opl0 == 0 cell; within a ray
    # the running opl must chain (this cell's opl0 == previous cell's opl1)
    # and stay monotone.
    starts = np.where(opl0 == 0.0)[0]
    assert starts.size > 0, "no source-born (opl0==0) root segment found"
    assert starts[0] == 0, "first cell is not a ray root"
    bounds = list(starts) + [len(opl0)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        assert np.all(np.diff(opl0[a:b]) >= -1e-15), "opl0 not monotone in ray"
        # continuity: segment start opl == previous segment end opl
        assert np.allclose(opl0[a + 1:b], opl1[a:b - 1], rtol=0, atol=1e-15)


@pytest.mark.parametrize("stem", BRIDGEABLE_SCENES)
def test_glass_segment_opl_equals_index_times_length(tmp_path, stem):
    """(3) the surface/media off-by-one pin: every segment's Δopl/length matches
    EITHER the ambient index or a glass surface's material_after index -- so no
    segment is ever booked against the wrong medium -- and, per ray, the glass
    index appears ONLY on interior legs (never the source->first-surface leg or
    the last-surface->image leg), which is exactly what a segment/media
    off-by-one shift would violate.

    Tolerance note: the design pins this at rel tol 1e-9, which is the tolerance
    of the FLOAT64 opl computation in _build. The rays.vtp cell arrays store opl
    as FLOAT32 (vtkexport.write_vtp_polylines, out of scope for this WP), so the
    file round-trip caps the recoverable precision at ~1e-6 relative (float32
    catastrophic cancellation of Σ n·ds against a large cumulative offset -- the
    far lens in beam_expander is the worst case). 1e-6 is still >5 orders of
    magnitude tighter than the ~0.5-relative signal an air/glass off-by-one
    produces, so the media pin is unambiguous; the per-ray medium-sequence
    assertion below independently catches any index shift."""
    if not _has(stem):
        pytest.skip("geometry/%s absent" % stem)
    out = tmp_path / "rays.vtp"
    ok, engine = sp.build(_geo(stem), out, pattern="fan:n=5")
    assert ok, engine
    p0, p1, opl0, opl1 = _read_segments(out)

    # resolve the allowed per-segment indices at the SYSTEM's own wavelength
    # (load_sequential_system derives it from the source; lens_dcx=633 nm,
    # beam_expander=650 nm -- NOT a fixed constant), the same wavelength
    # _build resolves n_seg at.
    system = ob.load_sequential_system(_geo(stem))
    wl = system.wavelength_nm
    n_amb = ob.resolve_index(system.ambient, wl)
    glass_idx = sorted({
        ob.resolve_index(s.material_after, wl)
        for s in system.surfaces
        if s.material_after and s.material_after != system.ambient})
    assert glass_idx, "scene has no glass surface to pin"
    allowed = np.asarray([n_amb] + glass_idx)

    seg_len = np.linalg.norm(p1 - p0, axis=1)
    nz = seg_len > 1e-12
    ratio = (opl1[nz] - opl0[nz]) / seg_len[nz]

    # (3a) every segment's index matches ONE allowed medium (float32-honest)
    rel_err = np.abs(ratio[:, None] - allowed[None, :]) / allowed[None, :]
    nearest = np.argmin(rel_err, axis=1)
    assert np.max(np.min(rel_err, axis=1)) < 1e-5, \
        ("segment index off every allowed medium", float(np.max(np.min(rel_err, axis=1))))
    # (3b) at least one segment carries a glass index (the in-glass leg exists)
    is_glass = nearest > 0                    # index 0 is ambient in `allowed`
    assert np.any(is_glass), "no segment carries a glass index"

    # (3c) the off-by-one guard: split cells back into per-ray runs (a new ray
    # starts at each opl0==0 root) and assert the FIRST (source->surface) and
    # LAST (surface->image) leg of every ray are ambient, never glass -- a
    # segment/media index shift would land glass on one of these end legs.
    roots = np.where(opl0[nz] == 0.0)[0]
    bounds = list(roots) + [len(ratio)]
    assert roots.size > 0 and roots[0] == 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        assert not is_glass[a], "source->first-surface leg booked as glass"
        assert not is_glass[b - 1], "last-surface->image leg booked as glass"


@pytest.mark.parametrize("stem", BRIDGEABLE_SCENES)
def test_beadanim_precompute_nonnull(tmp_path, stem):
    """(4) the user-facing payoff: beadanim.precompute_segments accepts the
    written polydata (opl0/opl1 present) and yields an animatable SegmentSet
    with t_max > 0 -- the "predate timing data" error is gone."""
    if not _has(stem):
        pytest.skip("geometry/%s absent" % stem)
    from mieworkbench.core import beadanim
    out = tmp_path / "rays.vtp"
    ok, engine = sp.build(_geo(stem), out, pattern="fan:n=5")
    assert ok, engine
    seg = beadanim.precompute_segments(_read_polydata(out))
    assert seg is not None, "precompute_segments returned None (no timing data)"
    assert seg.t_max > 0.0


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
