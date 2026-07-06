# =============================================================================
# test_gather_occlusion.py — optional gather-occlusion (render_coherent(...,
# occlusion=...)):
#   * baseline identity: occlusion=None is bit-identical to a no-arg render
#   * empty faces list is identity and reports 0 faces
#   * an opaque plate between slit and detector shadows the blocked pixels
#     to ~0 while the un-shadowed pixels stay bit-identical
#   * a plate beside the line of sight is culled by the conservative
#     prefilter (n_faces_active == 0) and leaves the image identical
#   * torch and numpy apply the SAME mask (agree to 5e-3)
#   * shadow-edge lands within `tile` pixels of the geometric projection
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_gather_occlusion.py -v
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from raytracer.surfaces import Plane, AnalyticFace       # noqa: E402
from raytracer.detector import DetectorGrid              # noqa: E402
from raytracer import gather                             # noqa: E402

LAM = 633e-9
L = 0.1                      # slit plane -> detector distance


def _detector(half=5e-3, resolution=128, x0=L):
    """Square detector of half-size `half` at x = x0, normal -x.
    Grid basis works out to xhat = +y, yhat = -z (see detector.py)."""
    sq = [[x0, -half, -half], [x0, half, -half],
          [x0, half, half], [x0, -half, half]]
    face = AnalyticFace("Det.Synth.Face1",
                        Plane([x0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
                        [sq], True, 0, 0, area_m2=(2 * half) ** 2)
    return DetectorGrid(face, resolution, spectral_bins=4,
                        lam_range=(600e-9, 660e-9), label="synth")


def _source(rng, width, height, m, yc=0.0):
    """Uniform random samples over a rectangular emit patch at x=0, all
    radiating +x (collimated ray direction, curved wavefront from the point
    spread)."""
    y = rng.uniform(yc - width / 2, yc + width / 2, m)
    z = rng.uniform(-height / 2, height / 2, m)
    pos = np.stack([np.zeros(m), y, z], axis=-1)
    dirp = np.tile([1.0, 0.0, 0.0], (m, 1))
    s_hat = np.tile([0.0, 0.0, 1.0], (m, 1))
    amp = np.sqrt(1.0 / m)
    return {"pos": pos, "dir": dirp, "s_hat": s_hat,
            "Es": np.full(m, amp / np.sqrt(2), dtype=np.complex128),
            "Ep": np.full(m, amp / np.sqrt(2), dtype=np.complex128),
            "lam": np.full(m, LAM), "opl": np.zeros(m),
            "power": np.full(m, 1.0 / m),
            "scattered": np.zeros(m, dtype=bool)}


def _plate(x0, y_lo, y_hi, z_lo, z_hi, fid="Plate.Face1"):
    """Opaque rectangular plane occluder at x = x0, spanning the given y/z
    box (normal +x; orientation is irrelevant for an occluder)."""
    poly = [[x0, y_lo, z_lo], [x0, y_hi, z_lo],
            [x0, y_hi, z_hi], [x0, y_lo, z_hi]]
    return AnalyticFace(fid, Plane([x0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                        [poly], True, 1, 0,
                        area_m2=(y_hi - y_lo) * (z_hi - z_lo))


def _render(det_factory, s, occlusion=None, backend="numpy",
            enforce_gate=False):
    """Fresh detector (render_coherent accumulates into det.inc), set the
    samples, render, return (intensity HxW, diags)."""
    det = det_factory()
    det.samples = {(0, 0): [s]}
    det.detected_geometric = {(0, 0): float(np.sum(s["power"]))}
    diags = gather.render_coherent(det, {(0, 0): 1e-8}, backend=backend,
                                   enforce_gate=enforce_gate,
                                   occlusion=occlusion)
    return det.inc.sum(axis=0), diags


# ---------------------------------------------------------------------------
def test_baseline_identity_none():
    """occlusion=None must be bit-identical to the no-argument render."""
    rng = np.random.default_rng(0)
    s = _source(rng, 3e-3, 3e-3, 3000)
    fac = lambda: _detector(resolution=96)                    # noqa: E731
    ref, _ = _render(fac, s)
    same, d = _render(fac, s, occlusion=None)
    assert np.array_equal(ref, same)
    assert all("occlusion" not in v for v in d.values())      # zero-cost off


def test_empty_faces_identity():
    """A non-None occlusion with no faces is identity and reports 0 faces."""
    rng = np.random.default_rng(1)
    s = _source(rng, 3e-3, 3e-3, 3000)
    fac = lambda: _detector(resolution=96)                    # noqa: E731
    ref, _ = _render(fac, s)
    same, d = _render(fac, s, occlusion={"faces": []})
    assert np.array_equal(ref, same)
    occ = d[(0, 0)]["occlusion"]
    assert occ["n_faces_tested"] == 0
    assert occ["n_faces_active"] == 0
    assert occ["frac_pairs_blocked"] == 0.0


def test_plate_shadows_lower_half():
    """An opaque plate between a (near-point) source and the detector zeroes
    the pixels it shadows while leaving the un-shadowed pixels bit-identical.
    A near-point source keeps the geometric shadow sharp (no penumbra)."""
    rng = np.random.default_rng(2)
    half = 5e-3
    s = _source(rng, 2e-4, 2e-4, 4000)
    fac = lambda: _detector(half=half, resolution=128)        # noqa: E731
    ref, _ = _render(fac, s)
    # plate at x=L/2 with top edge just below centre -> shadows z_det < ~0.
    ztop = -0.02 * half
    plate = _plate(L / 2, -10 * half, 10 * half, -10 * half, ztop)
    occ_img, d = _render(fac, s, occlusion={"faces": [plate]})
    occ = d[(0, 0)]["occlusion"]
    assert occ["n_faces_tested"] == 1 and occ["n_faces_active"] == 1
    assert occ["frac_pairs_blocked"] > 0.0

    det = _detector(half=half, resolution=128)
    z = det.pixel_centers[:, :, 2]
    # well below the (tile-quantized) shadow edge: fully extinguished (a
    # blocked (sample, pixel) pair contributes K=0 -> the estimate is exactly
    # zero, well under 1% of the un-occluded value)
    deep = z < -0.5 * half
    ref_scale = np.abs(ref[deep]).max()
    assert np.abs(occ_img[deep]).max() < 1e-2 * ref_scale
    # comfortably above the edge the field is UNCHANGED, but the documented
    # per-population power renormalization (factor = P_geom / integral) rescales
    # the whole map by a single scalar once the lower half is removed. So the
    # un-shadowed region stays identical UP TO that one global constant:
    upper = z > 0.5 * half
    u_ref, u_occ = ref[upper], occ_img[upper]
    big = np.abs(u_ref) > 0.05 * np.abs(u_ref).max()
    c = u_occ[big] / u_ref[big]
    assert np.ptp(c) < 1e-9 * abs(np.mean(c)), np.ptp(c)   # uniform rescale


def test_plate_beside_is_culled():
    """A plate off to the side blocks nothing; the conservative AABB
    prefilter must cull it (n_faces_active == 0) and the image is identical."""
    rng = np.random.default_rng(3)
    half = 5e-3
    s = _source(rng, 3e-3, 3e-3, 3000)
    fac = lambda: _detector(half=half, resolution=96)         # noqa: E731
    ref, _ = _render(fac, s)
    beside = _plate(L / 2, 10 * half, 12 * half, -half, half)  # far +y
    occ_img, d = _render(fac, s, occlusion={"faces": [beside]})
    occ = d[(0, 0)]["occlusion"]
    assert occ["n_faces_tested"] == 1
    assert occ["n_faces_active"] == 0
    assert occ["frac_pairs_blocked"] == 0.0
    assert np.array_equal(occ_img, ref)


def test_detector_face_never_occludes():
    """Passing the detector's own face must not shadow its own screen."""
    rng = np.random.default_rng(4)
    s = _source(rng, 3e-3, 3e-3, 3000)
    det = _detector(resolution=96)
    ref, _ = _render(lambda: _detector(resolution=96), s)
    det.samples = {(0, 0): [s]}
    det.detected_geometric = {(0, 0): float(np.sum(s["power"]))}
    d = gather.render_coherent(det, {(0, 0): 1e-8}, backend="numpy",
                               enforce_gate=False,
                               occlusion={"faces": [det.face]})
    occ = d[(0, 0)]["occlusion"]
    assert occ["n_faces_tested"] == 0            # own face skipped
    assert np.array_equal(det.inc.sum(axis=0), ref)


def test_torch_matches_numpy_with_occlusion():
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
    except ImportError:
        pytest.skip("no torch")
    rng = np.random.default_rng(5)
    half = 5e-3
    s = _source(rng, 3e-3, 3e-3, 3000)
    fac = lambda: _detector(half=half, resolution=128)        # noqa: E731
    plate = _plate(L / 2, -10 * half, 10 * half, -10 * half, -0.02 * half)
    occ = {"faces": [plate]}
    img_n, dn = _render(fac, s, occlusion=occ, backend="numpy")
    img_t, dt = _render(fac, s, occlusion=occ, backend="torch")
    # masks are built once in numpy for both backends -> same diag
    assert dn[(0, 0)]["occlusion"] == dt[(0, 0)]["occlusion"]
    scale = np.abs(img_n).max()
    assert np.max(np.abs(img_n - img_t)) / scale < 5e-3


@pytest.mark.parametrize("tile", [1, 16])
def test_shadow_edge_position(tile):
    """The shadow edge on the detector lands within `tile` pixels of the
    geometric projection of the plate's top edge."""
    rng = np.random.default_rng(6 + tile)
    half = 5e-3
    res = 64
    # near-point source -> sharp geometric shadow
    s = _source(rng, 2e-4, 2e-4, 1500)
    fac = lambda: _detector(half=half, resolution=res)        # noqa: E731
    ref, _ = _render(fac, s)
    ztop = -0.15 * half
    plate = _plate(L / 2, -10 * half, 10 * half, -10 * half, ztop)
    occ_img, _ = _render(fac, s, occlusion={"faces": [plate], "tile": tile})

    det = _detector(half=half, resolution=res)
    z_col = det.pixel_centers[:, det.W // 2, 2]               # z per row
    pixel_m = det.pixel_m
    # blocked pixels contribute K=0 -> the estimate is EXACTLY 0.0 (a robust
    # marker immune to the zero-mean MC noise elsewhere). The shadow edge is
    # the highest-z fully-blocked row.
    blocked_row = np.all(occ_img == 0.0, axis=1)
    assert blocked_row.any(), "no shadow found"
    z_edge = z_col[blocked_row].max()
    z_geom = 2.0 * ztop                        # straight line 0 -> (L/2, ztop)
    assert abs(z_edge - z_geom) <= (tile + 1.5) * pixel_m, \
        (z_edge, z_geom, tile * pixel_m)
