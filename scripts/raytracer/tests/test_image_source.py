# =============================================================================
# test_image_source.py -- extended image-emitting source (samples-
# instruments round).
#
# A source body's `image` property binds a greyscale bitmap (registry
# image/images.mieimg) as a per-position radiance map over the emitting
# face: rays are drawn with probability proportional to pixel value
# (Vose alias method) at EQUAL per-ray power, jittered within their
# pixel, and emitted Lambertian (or into an image_cone_deg cone) so an
# imaging bench can form a real image of the bitmap. See
# sources.load_image_gray/_sample_image_points/_image_emission_dirs.
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_image_source.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import common                                                    # noqa: E402
from raytracer.scene import Scene                                # noqa: E402
from raytracer.sources import (sample_source, load_image_gray,   # noqa: E402
                               _build_alias_table, _alias_draw)
from raytracer.optprops import load_optical_properties           # noqa: E402
from . import scenehelpers as sh                                 # noqa: E402


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------
def test_load_image_gray_npy_and_rejects_zero(tmp_path):
    arr = np.array([[0.0, 1.0], [2.0, 3.0]])
    p = tmp_path / "t.npy"
    np.save(p, arr)
    assert np.array_equal(load_image_gray(p), arr)
    z = tmp_path / "z.npy"
    np.save(z, np.zeros((4, 4)))
    with pytest.raises(ValueError, match="all-zero"):
        load_image_gray(z)


def test_load_image_gray_rgb_luma(tmp_path):
    rgb = np.zeros((2, 2, 3))
    rgb[0, 0] = [1.0, 0.0, 0.0]
    rgb[1, 1] = [0.0, 1.0, 0.0]
    p = tmp_path / "c.npy"
    np.save(p, rgb)
    g = load_image_gray(p)
    assert g[0, 0] == pytest.approx(0.299)
    assert g[1, 1] == pytest.approx(0.587)


# ---------------------------------------------------------------------------
# alias method
# ---------------------------------------------------------------------------
def test_alias_table_reproduces_distribution():
    prob = np.array([0.5, 0.25, 0.125, 0.125])
    pt, ai = _build_alias_table(prob)
    rng = np.random.default_rng(0)
    draws = _alias_draw(pt, ai, rng, 200000)
    freq = np.bincount(draws, minlength=4) / 200000.0
    assert freq == pytest.approx(prob, abs=5e-3)


# ---------------------------------------------------------------------------
# end-to-end sampling through sample_source
# ---------------------------------------------------------------------------
def _image_scene(img, cone_deg=None, coherent=False):
    model = sh.make_model([
        sh.source_body("Src", x=-0.02, half=0.004, power_mW=2.0,
                       lambdac_nm=550.0, coherent=coherent),
        sh.detector_body("Det", x=0.03, half=0.03),
    ])
    common.validate_model(model)
    optprops = load_optical_properties()
    scene = Scene(model, optprops.matdb, optprops.coatings,
                  optprops=optprops)
    bidx, src = scene.sources[0]
    src["_image_gray"] = np.asarray(img, dtype=np.float64)
    if cone_deg is not None:
        src["image_cone_deg"] = cone_deg
    return scene, bidx, src


def test_image_positions_follow_bitmap_orientation():
    """Top-left bright pixel only -> every ray in the (u < mid, v > mid)
    quadrant: row 0 is the TOP of the picture (max v), col 0 at u_lo."""
    scene, bidx, src = _image_scene([[1.0, 0.0], [0.0, 0.0]])
    rng = np.random.default_rng(7)
    batch = sample_source(scene, scene.bodies[bidx], src, 0, 2000, 1, rng)
    # emit face: x = -0.02 plane, square half=0.004 in (y, z). Map ray
    # positions into the face's own (t1, t2) frame for the quadrant test.
    face = scene.emit_faces[bidx]
    surf = face.surface
    rel = batch.pos - surf.origin
    u = rel @ surf.t1
    v = rel @ surf.t2
    assert np.all(u < 1e-12)          # bright column = low-u half
    assert np.all(v > -1e-12)         # bright row 0 = top = high-v half
    # equal power per ray (the bitmap rides the DENSITY, not the power)
    assert np.allclose(batch.birth_power, batch.birth_power[0])


def test_image_density_proportional_to_pixel_value():
    scene, bidx, src = _image_scene([[3.0, 1.0]])
    rng = np.random.default_rng(11)
    batch = sample_source(scene, scene.bodies[bidx], src, 0, 40000, 1, rng)
    face = scene.emit_faces[bidx]
    surf = face.surface
    u = (batch.pos - surf.origin) @ surf.t1
    n_lo = int(np.sum(u < 0))
    n_hi = int(np.sum(u >= 0))
    assert n_lo / max(n_hi, 1) == pytest.approx(3.0, rel=0.05)


def test_image_dirs_lambertian_default():
    scene, bidx, src = _image_scene([[1.0]])
    rng = np.random.default_rng(3)
    batch = sample_source(scene, scene.bodies[bidx], src, 0, 20000, 1, rng)
    # emit normal is +x (toward the origin/detector); Lambertian
    # cosine-weighted hemisphere has <cos theta> = 2/3
    cos_t = batch.dir[:, 0]
    assert np.all(cos_t > 0)
    assert np.mean(cos_t) == pytest.approx(2.0 / 3.0, abs=0.01)


def test_image_dirs_cone_restriction():
    scene, bidx, src = _image_scene([[1.0]], cone_deg=10.0)
    rng = np.random.default_rng(3)
    batch = sample_source(scene, scene.bodies[bidx], src, 0, 5000, 1, rng)
    cos_min = np.cos(np.deg2rad(10.0))
    assert np.all(batch.dir[:, 0] >= cos_min - 1e-12)
    # uniform in solid angle within the cone: <cos> = (1+cos_max)/2
    assert np.mean(batch.dir[:, 0]) == pytest.approx(
        (1.0 + cos_min) / 2.0, abs=0.002)


def test_image_beam_mutually_exclusive():
    scene, bidx, src = _image_scene([[1.0]])
    src["beam"] = {"waist_mm": 1.0, "m2": 1.0}
    with pytest.raises(ValueError, match="mutually exclusive"):
        sample_source(scene, scene.bodies[bidx], src, 0, 100, 1,
                      np.random.default_rng(0))
