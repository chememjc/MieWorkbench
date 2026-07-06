# =============================================================================
# test_polarization.py — source polarization states + polarization strata.
#   * Jones vectors for linear/circular/elliptical/unpolarized
#   * circular handedness verified NUMERICALLY against the documented
#     convention (clockwise facing the oncoming beam = right)
#   * unpolarized = two mutually-incoherent orthogonal populations:
#     equal power split, orthogonal Jones, and NO interference between
#     pol strata in the gather (co-located co-polarized populations in
#     different strata add in intensity, not amplitude)
# Run: /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_polarization.py -v
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from raytracer.sources import (jones_for, n_pol_strata,      # noqa: E402
                               _pol_reference_frame)
from raytracer.surfaces import Plane, AnalyticFace           # noqa: E402
from raytracer.detector import DetectorGrid                  # noqa: E402
from raytracer import gather                                 # noqa: E402

LAM = 633e-9
L = 0.1


# ---------------------------------------------------------------------------
# reference frame
# ---------------------------------------------------------------------------
def test_reference_frame_orthonormal():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(200, 3))
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    e_ref, e_perp = _pol_reference_frame(d)
    assert np.allclose(np.linalg.norm(e_ref, axis=-1), 1.0, atol=1e-12)
    assert np.allclose(np.sum(e_ref * d, axis=-1), 0.0, atol=1e-12)
    assert np.allclose(np.sum(e_ref * e_perp, axis=-1), 0.0, atol=1e-12)
    assert np.allclose(np.cross(d, e_ref), e_perp, atol=1e-12)


def test_reference_frame_z_fallback():
    d = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    e_ref, e_perp = _pol_reference_frame(d)
    # z projects to zero -> falls back to global y
    assert np.allclose(np.abs(e_ref @ np.array([0.0, 1.0, 0.0])), 1.0)
    assert np.allclose(np.linalg.norm(e_perp, axis=-1), 1.0)


# ---------------------------------------------------------------------------
# Jones states
# ---------------------------------------------------------------------------
def test_jones_unit_power_all_kinds():
    for pol, ps in [({"kind": "unpolarized"}, 0),
                    ({"kind": "unpolarized"}, 1),
                    ({"kind": "linear", "angle_deg": 37.0}, 0),
                    ({"kind": "circular", "handedness": "left"}, 0),
                    ({"kind": "circular", "handedness": "right"}, 0),
                    ({"kind": "elliptical", "psi_deg": 20.0,
                      "chi_deg": 15.0}, 0)]:
        es, ep = jones_for(pol, ps)
        assert abs(abs(es) ** 2 + abs(ep) ** 2 - 1.0) < 1e-12, pol


def test_linear_angles():
    es, ep = jones_for({"kind": "linear", "angle_deg": 0.0}, 0)
    assert abs(es - 1.0) < 1e-12 and abs(ep) < 1e-12
    es, ep = jones_for({"kind": "linear", "angle_deg": 90.0}, 0)
    assert abs(es) < 1e-12 and abs(ep - 1.0) < 1e-12
    es, ep = jones_for({"kind": "linear", "angle_deg": 45.0}, 0)
    assert abs(es - ep) < 1e-12


def test_unpolarized_strata_orthogonal():
    assert n_pol_strata({}) == 2
    assert n_pol_strata({"polarization": {"kind": "linear",
                                          "angle_deg": 0}}) == 1
    e0 = jones_for({"kind": "unpolarized"}, 0)
    e1 = jones_for({"kind": "unpolarized"}, 1)
    # orthogonal Jones vectors
    assert abs(e0[0] * np.conj(e1[0]) + e0[1] * np.conj(e1[1])) < 1e-12


def test_elliptical_limits():
    # chi = 0 -> linear at psi
    es, ep = jones_for({"kind": "elliptical", "psi_deg": 30.0,
                        "chi_deg": 0.0}, 0)
    assert abs(es.imag) < 1e-12 and abs(ep.imag) < 1e-12
    assert abs(np.arctan2(ep.real, es.real) - np.deg2rad(30.0)) < 1e-12
    # chi = 45 -> circular (equal magnitudes, +/-90 deg phase)
    es, ep = jones_for({"kind": "elliptical", "psi_deg": 0.0,
                        "chi_deg": 45.0}, 0)
    assert abs(abs(es) - abs(ep)) < 1e-12
    assert abs(np.angle(ep / es) - np.pi / 2) < 1e-12


def test_circular_handedness_numeric():
    """'right' = E rotates clockwise as seen facing the ONCOMING beam.

    Beam along +x, observer at +x looking back along -x with up = z:
    the observer's screen-right is r = up x view = z x (-x) = -y.
    Field convention Re[E exp(-i w t)], basis e_ref = z (s), e_perp =
    dir x e_ref = -y (p). Track the field vector over a quarter period
    and check it moves from up toward screen-right (clockwise).
    """
    d = np.array([[1.0, 0.0, 0.0]])
    e_ref, e_perp = _pol_reference_frame(d)
    assert np.allclose(e_ref[0], [0, 0, 1])
    assert np.allclose(e_perp[0], [0, -1, 0])
    up = np.array([0.0, 0.0, 1.0])
    screen_right = np.array([0.0, -1.0, 0.0])

    def field(es, ep, wt):
        E = (es * e_ref[0] + ep * e_perp[0]) * np.exp(-1j * wt)
        return E.real

    es, ep = jones_for({"kind": "circular", "handedness": "right"}, 0)
    E0 = field(es, ep, 0.0)
    E1 = field(es, ep, 0.4)          # a bit later in the cycle
    # starts along up, rotates toward screen-right => clockwise
    assert E0 @ up > 0.9 / np.sqrt(2)
    assert E1 @ screen_right > 0.0
    es, ep = jones_for({"kind": "circular", "handedness": "left"}, 0)
    E1 = field(es, ep, 0.4)
    assert E1 @ screen_right < 0.0   # counterclockwise


# ---------------------------------------------------------------------------
# strata never interfere in the gather
# ---------------------------------------------------------------------------
def _detector(half=5e-3, resolution=256, x0=L):
    sq = [[x0, -half, -half], [x0, half, -half],
          [x0, half, half], [x0, -half, half]]
    face = AnalyticFace("Det.Synth.Face1",
                        Plane([x0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
                        [sq], True, 0, 0, area_m2=(2 * half) ** 2)
    return DetectorGrid(face, resolution, spectral_bins=4,
                        lam_range=(600e-9, 660e-9), label="synth")


def _slit_samples(rng, slits, width, height, m_per_slit):
    pos = []
    for yc in slits:
        y = rng.uniform(yc - width / 2, yc + width / 2, m_per_slit)
        z = rng.uniform(-height / 2, height / 2, m_per_slit)
        pos.append(np.stack([np.zeros(m_per_slit), y, z], axis=-1))
    pos = np.concatenate(pos)
    m = len(pos)
    return {"pos": pos, "dir": np.tile([1.0, 0.0, 0.0], (m, 1)),
            "s_hat": np.tile([0.0, 0.0, 1.0], (m, 1)),
            "Es": np.full(m, np.sqrt(1.0 / m), dtype=np.complex128),
            "Ep": np.zeros(m, dtype=np.complex128),
            "lam": np.full(m, LAM), "opl": np.zeros(m),
            "power": np.full(m, 1.0 / m),
            "scattered": np.zeros(m, dtype=bool)}


def _fringe_visibility(det, inten):
    """Fringe contrast in the CENTRAL region only (several fringe periods,
    well inside the single-slit envelope) so envelope falloff toward the
    detector edges does not masquerade as fringe visibility."""
    prof = inten[inten.shape[0] // 2]
    x = (np.arange(det.W) + 0.5) * det.pixel_m + det.x_lo
    ctr = np.abs(x) < 1.2e-3          # ~7 fringe periods at 316 um pitch
    p = prof[ctr]
    lo = max(float(p.min()), 0.0)
    hi = float(p.max())
    return (hi - lo) / (hi + lo)


def test_pol_strata_do_not_interfere():
    """One slit's field in stratum 0, the other slit's in stratum 1 —
    both co-polarized. Same stratum -> Young fringes; different strata
    -> the fringes must vanish (fields add in intensity only)."""
    rng = np.random.default_rng(7)
    d_sep = 200e-6
    sA = _slit_samples(rng, [-d_sep / 2], 10e-6, 1e-3, 3000)
    sB = _slit_samples(rng, [+d_sep / 2], 10e-6, 1e-3, 3000)

    # same stratum: interference
    det = _detector(resolution=128)
    det.samples = {(0, 0, 0): [sA, sB]}
    det.detected_geometric = {(0, 0, 0): 2.0}
    gather.render_coherent(det, {(0, 0, 0): 1e-8}, backend="numpy",
                           enforce_gate=False)
    vis_same = _fringe_visibility(det, det.inc.sum(axis=0))

    # different pol strata: no interference
    det2 = _detector(resolution=128)
    det2.samples = {(0, 0, 0): [sA], (0, 0, 1): [sB]}
    det2.detected_geometric = {(0, 0, 0): 1.0, (0, 0, 1): 1.0}
    gather.render_coherent(det2, {(0, 0, 0): 1e-8, (0, 0, 1): 1e-8},
                           backend="numpy", enforce_gate=False)
    vis_diff = _fringe_visibility(det2, det2.inc.sum(axis=0))

    assert vis_same > 0.7, vis_same
    assert vis_diff < 0.35, vis_diff


# ---------------------------------------------------------------------------
# sample_source end-to-end (needs an extracted model)
# ---------------------------------------------------------------------------
MODEL_JSON = SCRIPTS.parent / "geometry" / "doubleslit" / "model.json"


@pytest.mark.skipif(not MODEL_JSON.exists(),
                    reason="geometry/doubleslit/model.json not extracted")
def test_sample_source_polarization_states():
    import common
    from raytracer.materials import MaterialDB, load_coatings
    from raytracer.scene import Scene
    from raytracer.sources import sample_source

    model = common.load_model(MODEL_JSON)
    db = MaterialDB.load()
    scene = Scene(model, db, load_coatings(db=db))
    bidx, src = scene.sources[0]
    rng = np.random.default_rng(1)

    # default: unpolarized -> 2 strata, equal power, orthogonal fields
    b = sample_source(scene, scene.bodies[bidx], dict(src), 0, 4000, 1, rng)
    assert set(np.unique(b.pol_stratum)) == {0, 1}
    p0 = float(np.sum(b.power[b.pol_stratum == 0]))
    p1 = float(np.sum(b.power[b.pol_stratum == 1]))
    assert abs(p0 - p1) / (p0 + p1) < 0.02
    s0 = b.pol_stratum == 0
    assert np.all(np.abs(b.Ep[s0]) < 1e-15)      # stratum 0 = pure e_ref
    assert np.all(np.abs(b.Es[~s0]) < 1e-15)     # stratum 1 = pure e_perp
    total = float(np.sum(b.power))
    assert total == pytest.approx(src["power_mW"] * 1e-3, rel=1e-9)

    # linear:30 -> single stratum, Jones ratio tan(30)
    src2 = dict(src)
    src2["polarization"] = {"kind": "linear", "angle_deg": 30.0}
    b = sample_source(scene, scene.bodies[bidx], src2, 0, 1000, 1, rng)
    assert set(np.unique(b.pol_stratum)) == {0}
    ratio = np.abs(b.Ep) / np.abs(b.Es)
    assert np.allclose(ratio, np.tan(np.deg2rad(30.0)), atol=1e-12)

    # circular -> 90 deg phase between components
    src3 = dict(src)
    src3["polarization"] = {"kind": "circular", "handedness": "right"}
    b = sample_source(scene, scene.bodies[bidx], src3, 0, 1000, 1, rng)
    assert np.allclose(np.abs(np.angle(b.Ep / b.Es)), np.pi / 2,
                       atol=1e-12)
    assert np.allclose(np.abs(b.Es), np.abs(b.Ep), rtol=1e-12)
