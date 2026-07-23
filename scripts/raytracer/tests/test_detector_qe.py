# =============================================================================
# test_detector_qe.py -- QE-weighted detector photocurrent.
#
# detector.spectral_cube_to_photocurrent is a pure function over a synthetic
# (bins, H, W) power cube [W] and a (lam_um, qe) curve; no tracer or case
# directory needed. post_process.detector_qe_curve_for_label pins the
# label -> owning-body ownership match.
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest scripts/raytracer/tests/test_detector_qe.py -v
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

from raytracer.detector import spectral_cube_to_photocurrent    # noqa: E402
import post_process as pp                                       # noqa: E402

# CODATA exact SI (must match detector.py)
Q = 1.602176634e-19
H = 6.62607015e-34
C = 299792458.0


def _one_bin_cube(power_W):
    """(1, 2, 2) cube whose total power is power_W (spread over 4 pixels)."""
    return np.full((1, 2, 2), power_W / 4.0, dtype=np.float64)


def test_flat_qe_photocurrent_hand_computed():
    # 1 W entirely at 800 nm, QE = 0.8 flat across a table that brackets it.
    # R = QE*q*lambda/(h*c) = 0.8 * q * 800e-9 / (h*c) ~ 0.5162 A/W
    lam = 800e-9
    cube = _one_bin_cube(1.0)                 # single bin centered at 800 nm
    lam_lo, lam_hi = lam - 1e-9, lam + 1e-9   # narrow bin around 800 nm
    qe_lam_um = np.array([0.5, 1.0])          # brackets 0.8 um
    qe_vals = np.array([0.8, 0.8])
    i_a, p_w, cov = spectral_cube_to_photocurrent(
        cube, lam_lo, lam_hi, qe_lam_um, qe_vals)
    R_expect = 0.8 * Q * lam / (H * C)
    assert i_a == pytest.approx(R_expect, rel=1e-6)      # ~0.5162 A for 1 W
    assert i_a == pytest.approx(0.5162, abs=2e-4)
    assert p_w == pytest.approx(0.8, rel=1e-9)           # QE-weighted power
    assert cov == pytest.approx(1.0)


def test_coverage_straddling_table_edge():
    # two equal-power bins, one inside the QE table's range and one past its
    # red edge -> QE=0 (interp right=0) on the outside bin, coverage = 0.5.
    qe_lam_um = np.array([0.60, 0.90])       # 600..900 nm
    qe_vals = np.array([0.5, 0.5])
    bins = 2
    # bin centers at 700 nm (inside) and 1100 nm (outside): span 500..1300 nm
    lam_lo, lam_hi = 500e-9, 1300e-9
    cube = np.zeros((bins, 1, 1))
    cube[0, 0, 0] = 1.0                       # 700 nm bin, 1 W
    cube[1, 0, 0] = 1.0                       # 1100 nm bin, 1 W
    i_a, p_w, cov = spectral_cube_to_photocurrent(
        cube, lam_lo, lam_hi, qe_lam_um, qe_vals)
    assert cov == pytest.approx(0.5)
    # only the inside bin (700 nm, QE 0.5) contributes power/current
    assert p_w == pytest.approx(0.5, rel=1e-9)
    R_700 = 0.5 * Q * 700e-9 / (H * C)
    assert i_a == pytest.approx(R_700, rel=1e-6)


def test_zero_power_cube_is_all_zero():
    cube = np.zeros((4, 3, 3))
    i_a, p_w, cov = spectral_cube_to_photocurrent(
        cube, 400e-9, 800e-9, np.array([0.4, 0.8]), np.array([0.9, 0.9]))
    assert (i_a, p_w, cov) == (0.0, 0.0, 0.0)


def test_spectrum_and_cube_agree():
    # a (bins,) spectrum and a (bins,H,W) cube with the same per-bin totals
    # give identical scalars (trailing axes are summed out).
    qe_lam_um = np.array([0.40, 0.80])
    qe_vals = np.array([0.7, 0.9])
    spectrum = np.array([0.2, 0.5, 0.3])
    cube = np.zeros((3, 2, 2))
    for b in range(3):
        cube[b] = spectrum[b] / 4.0
    args = (400e-9, 800e-9, qe_lam_um, qe_vals)
    assert spectral_cube_to_photocurrent(spectrum, *args) == pytest.approx(
        spectral_cube_to_photocurrent(cube, *args))


# ---------------------------------------------------------------------------
# label -> owning body ownership match (post_process)
# ---------------------------------------------------------------------------
def test_qe_curve_ownership_prefix_match():
    qe_bodies = {"Detector": "hamamatsu_s1223"}
    # h5 label is a face id "<BodyName>.<Tip>.FaceN"
    assert pp.detector_qe_curve_for_label(
        "Detector.Pad.Face3", qe_bodies) == "hamamatsu_s1223"
    # exact label (no dotted suffix) also resolves
    assert pp.detector_qe_curve_for_label(
        "Detector", qe_bodies) == "hamamatsu_s1223"
    # a body with no qe_curve -> None
    assert pp.detector_qe_curve_for_label("Other.X.Face1", qe_bodies) is None


def test_qe_curve_ownership_longest_prefix_wins():
    qe_bodies = {"Det": "curve_a", "Detector": "curve_b"}
    assert pp.detector_qe_curve_for_label(
        "Detector.Pad.Face1", qe_bodies) == "curve_b"
    assert pp.detector_qe_curve_for_label(
        "Det.Pad.Face1", qe_bodies) == "curve_a"
