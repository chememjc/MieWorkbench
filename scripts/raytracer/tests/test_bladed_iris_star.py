# =============================================================================
# test_bladed_iris_star.py -- the N-blade iris diffraction STAR (engine3 Sec 11
# / P8). A coherent beam clipped by N straight blade edges diffracts into the
# classic N-fold star (N spikes for even N, 2N for odd), NOT the circular Airy
# rings. We quantify the 6-fold azimuthal symmetry, never eyeball it.
#
# Two tiers:
#   * FAST/deterministic: scalar Fraunhofer (FFT) of the EXACT regular-hexagon
#     aperture the iris_bladed builder defines (inscribed circle = clear
#     aperture) vs a circle of the same clear aperture. The m=6 azimuthal
#     Fourier component dominates for the hexagon and vanishes for the circle.
#   * ENGINE end-to-end (FreeCAD + slow-gated): author the real scenes through
#     primitivelib, extract with extract_geometry, trace COHERENTLY through the
#     full engine + Huygens gather, and assert the hexagon's detector image
#     carries a strong 6-fold modulation the circle's does not.
#
# Run (fast tier only):
#   "$MIEWB_OPTICS_PYTHON" -m pytest \
#       scripts/raytracer/tests/test_bladed_iris_star.py -q
# Run (+ engine tier):
#   MIEWB_RUN_FREECAD=1 "$MIEWB_OPTICS_PYTHON" -m pytest \
#       scripts/raytracer/tests/test_bladed_iris_star.py -q -m ''
# =============================================================================
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import common  # noqa: E402

RUN_FREECAD = os.environ.get("MIEWB_RUN_FREECAD") == "1"
FREECAD = common.FREECAD_APPIMAGE
OPTPROPS = str(common.OPTPROPS_DIR)

LAM = 633e-9


# --------------------------------------------------------------------------- #
# shared 6-fold metric: azimuthal Fourier spectrum of an annulus of the image
# --------------------------------------------------------------------------- #
def sixfold_ratio(I, px, r_lo, r_hi, nbins=120):
    """(m6/AC, others/AC) for image I (pixel pitch px), over the annulus
    [r_lo, r_hi] centered on the intensity centroid. m6/AC is the fraction of
    azimuthal AC power in the 6-fold component; `others` are the NON-multiple-
    of-6 modes 1..12 (m=12, the second harmonic of the 6-fold, belongs to the
    star signature, NOT to the competing background)."""
    H, W = I.shape
    yy, xx = np.mgrid[0:H, 0:W]
    pos = np.maximum(I, 0.0)
    tot = pos.sum()
    cx = (xx * pos).sum() / tot
    cy = (yy * pos).sum() / tot
    x = (xx - cx) * px
    y = (yy - cy) * px
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    ann = (r >= r_lo) & (r <= r_hi)
    b = (((th[ann] + math.pi) / (2 * math.pi) * nbins).astype(int)) % nbins
    prof = np.zeros(nbins)
    cnt = np.zeros(nbins)
    np.add.at(prof, b, I[ann])
    np.add.at(cnt, b, 1)
    prof = prof / np.maximum(cnt, 1)
    prof -= prof.mean()
    F = np.abs(np.fft.rfft(prof))
    ac = math.sqrt(float((F[1:] ** 2).sum())) or 1.0
    # competing background = modes 1..12 that are NOT harmonics of 6
    others = np.array([F[m] / ac for m in range(1, 13) if m % 6 != 0])
    return F[6] / ac, others


# --------------------------------------------------------------------------- #
# FAST: scalar Fraunhofer of the exact aperture the primitive defines
# --------------------------------------------------------------------------- #
def _regular_polygon_mask(n, r_in, X, Y, rot=0.0):
    """Regular n-gon (INSCRIBED-circle radius r_in) as an intersection of n
    half-planes -- exactly the clear aperture iris_bladed builds
    (aperture_diameter = inscribed / flat-to-flat diameter)."""
    inside = np.ones(X.shape, dtype=bool)
    for k in range(n):
        a = rot + 2 * math.pi * k / n + math.pi / n
        inside &= (X * math.cos(a) + Y * math.sin(a)) <= r_in
    return inside


def _fraunhofer_sixfold(mask_fn):
    Nap = 2048
    half = 1.2e-3
    xa = np.linspace(-half, half, Nap)
    dxa = xa[1] - xa[0]
    Xa, Ya = np.meshgrid(xa, xa)
    field = mask_fn(Xa, Ya).astype(complex)
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    I = np.abs(F) ** 2
    fx = np.fft.fftshift(np.fft.fftfreq(Nap, d=dxa))
    px = (fx[1] - fx[0]) * LAM * 0.30            # screen coord at L = 0.30 m
    return sixfold_ratio(I, px, r_lo=8 * px, r_hi=45 * px)


def test_hexagon_fraunhofer_has_sixfold_star():
    r_in = 0.15e-3          # iris_bladed default-ish clear aperture (0.30 mm)
    m6_hex, others_hex = _fraunhofer_sixfold(
        lambda X, Y: _regular_polygon_mask(6, r_in, X, Y))
    m6_circ, _ = _fraunhofer_sixfold(
        lambda X, Y: (X ** 2 + Y ** 2) <= r_in ** 2)
    # the hexagon's far field is strongly 6-fold; the circle's is not
    assert m6_hex > 0.4, m6_hex
    assert m6_circ < 0.05, m6_circ
    # m=6 dominates every other low-order azimuthal mode of the hexagon
    assert m6_hex > 5.0 * float(np.max(others_hex))


def test_odd_blade_count_doubles_the_spikes():
    # a pentagon (odd N=5) has NO 6-fold term but a strong 10-fold one, and a
    # 5-fold one -- the even/odd spike-count rule. Just check m6 is NOT how a
    # pentagon reads (guards against a metric that fires on any polygon).
    r_in = 0.15e-3
    m6_pent, _ = _fraunhofer_sixfold(
        lambda X, Y: _regular_polygon_mask(5, r_in, X, Y))
    assert m6_pent < 0.15, m6_pent


# --------------------------------------------------------------------------- #
# ENGINE end-to-end: real coherent run through the full pipeline
# --------------------------------------------------------------------------- #
def _trace_star(geo_dir, shape, rays):
    import common
    from raytracer.scene import Scene
    from raytracer.sources import sample_source
    from raytracer.tracer import Tracer, TraceConfig
    from raytracer.detector import DetectorGrid
    from raytracer import gather
    from raytracer.optprops import load_optical_properties
    opt = load_optical_properties(root=OPTPROPS)
    model = common.load_model(geo_dir / ("star_" + shape) / "model.json")
    common.validate_model(model)
    sc = Scene(model, opt.matdb, opt.coatings, optprops=opt,
               geometry_dir=str(geo_dir / ("star_" + shape)))
    fid = list(sc.detector_faces)[0]
    grids = {fid: DetectorGrid(sc.faces[fid], 400, 4, (600e-9, 660e-9),
                               label=sc.faces[fid].id)}
    cfg = TraceConfig(rays=rays, n_lambda=1, seed=3, power_floor=1e-12)
    tr = Tracer(sc, cfg, grids)
    rng = np.random.default_rng(3)
    bidx, src = sc.sources[0]
    batch = sample_source(sc, sc.bodies[bidx], src, 0, cfg.rays, 1, rng,
                          ledger=tr.ledger)
    tr.run([batch])
    det = grids[fid]
    area = sc.emit_faces[bidx].area_m2 / cfg.rays
    gather.render_coherent(det, {(0, 0): area}, backend="auto",
                           min_eff_samples=200, enforce_gate=False)
    return det.inc.sum(axis=0), det.pixel_m


@pytest.mark.slow
@pytest.mark.skipif(not RUN_FREECAD,
                    reason="set MIEWB_RUN_FREECAD=1 for the FreeCAD e2e star")
def test_hexagon_star_through_engine(tmp_path):
    probe = Path(__file__).resolve().parent / "_bladed_star_fc_probe.py"
    # author both scenes
    subprocess.run([FREECAD, "-c", str(probe), "--", "--outdir", str(tmp_path)],
                   stdin=subprocess.DEVNULL, check=True,
                   timeout=600, capture_output=True)
    # extract both
    geo = tmp_path / "geo"
    subprocess.run(
        [FREECAD, "-c", str(SCRIPTS / "extract_geometry.py"), "--",
         "--models", str(tmp_path / "star_hex.FCStd"),
         str(tmp_path / "star_circle.FCStd"), "--outdir", str(geo)],
        stdin=subprocess.DEVNULL, check=True, timeout=600,
        capture_output=True)
    I_hex, px = _trace_star(geo, "hex", rays=150000)
    I_circ, _ = _trace_star(geo, "circle", rays=150000)
    m6_hex, _ = sixfold_ratio(I_hex, px, 0.6e-3, 3.5e-3)
    m6_circ, _ = sixfold_ratio(I_circ, px, 0.6e-3, 3.5e-3)
    # THE physical, artifact-free discriminator: the SAME m=6 azimuthal bin is
    # strong for the hexagon's coherent far field and near-zero for the
    # circle's rings (a >4x contrast in the same bin -- any square-detector-grid
    # aliasing, e.g. an m=4 mode, hits BOTH images equally and cancels out of
    # this comparison). The clean intra-mode dominance of the star is pinned by
    # the deterministic Fraunhofer tier above.
    assert m6_hex > 0.45, (m6_hex, m6_circ)
    assert m6_hex > 4.0 * m6_circ, (m6_hex, m6_circ)
