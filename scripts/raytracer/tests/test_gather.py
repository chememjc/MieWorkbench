# =============================================================================
# test_gather.py — coherent-gather validation:
#   * kernel exactness vs a brute-force reference
#   * point source: 1/r^2 intensity + spherical phase
#   * DOUBLE SLIT: fringe pitch lambda*L/d within 1%, visibility > 0.9
#   * single slit: sinc^2 first zero within 2%
#   * undersampling gate raises; torch and numpy backends agree
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest scripts/raytracer/tests/test_gather.py -v
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


def _detector(half=5e-3, resolution=512, x0=L):
    """Square detector of half-size `half` at x = x0, normal -x."""
    sq = [[x0, -half, -half], [x0, half, -half],
          [x0, half, half], [x0, -half, half]]
    face = AnalyticFace("Det.Synth.Face1",
                        Plane([x0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
                        [sq], True, 0, 0, area_m2=(2 * half) ** 2)
    return DetectorGrid(face, resolution, spectral_bins=4,
                        lam_range=(600e-9, 660e-9), label="synth")


def _slit_samples(rng, slits, width, height, m_per_slit):
    """Uniform random samples over vertical slits at x=0.
    slits: list of y-centers; width: slit width in y; height: in z."""
    pos = []
    for yc in slits:
        y = rng.uniform(yc - width / 2, yc + width / 2, m_per_slit)
        z = rng.uniform(-height / 2, height / 2, m_per_slit)
        pos.append(np.stack([np.zeros(m_per_slit), y, z], axis=-1))
    pos = np.concatenate(pos)
    m = len(pos)
    dirp = np.tile([1.0, 0.0, 0.0], (m, 1))
    s_hat = np.tile([0.0, 0.0, 1.0], (m, 1))
    amp = np.sqrt(1.0 / m)                     # total power 1 W
    return {"pos": pos, "dir": dirp, "s_hat": s_hat,
            "Es": np.full(m, amp / np.sqrt(2), dtype=np.complex128),
            "Ep": np.full(m, amp / np.sqrt(2), dtype=np.complex128),
            "lam": np.full(m, LAM), "opl": np.zeros(m),
            "power": np.full(m, 1.0 / m),
            "scattered": np.zeros(m, dtype=bool)}


def _render(det, s, backend="numpy"):
    det.samples = {(0, 0): [s]}
    det.detected_geometric = {(0, 0): float(np.sum(s["power"]))}
    diags = gather.render_coherent(det, {(0, 0): 1e-8}, backend=backend)
    inten = det.inc.sum(axis=0)
    return inten, diags


def test_kernel_matches_bruteforce():
    rng = np.random.default_rng(0)
    det = _detector(half=1e-3, resolution=16)
    m = 20
    pos = rng.uniform(-1e-3, 1e-3, (m, 3)) * [0, 1, 1]
    dirp = np.tile([1.0, 0.0, 0.0], (m, 1))
    E3 = (rng.normal(size=(m, 3)) + 1j * rng.normal(size=(m, 3))) * 0.1
    E3[:, 0] = 0.0                               # transverse
    lam = np.full(m, LAM)
    opl = rng.uniform(0, 1e-4, m)
    Ex, Ey = gather.accumulate_numpy(pos, E3, lam, opl, det, dirp)
    # brute force in float128-ish (python floats)
    k = 2 * np.pi / LAM
    pix = det.pixel_centers.reshape(-1, 3)
    ref_x = np.zeros(len(pix), dtype=complex)
    ref_y = np.zeros(len(pix), dtype=complex)
    for i in range(m):
        d = pix - pos[i]
        r = np.linalg.norm(d, axis=-1)
        rhat = d / r[:, None]
        cprop = rhat @ dirp[i]
        cdet = np.abs(rhat @ det.normal)
        K = np.clip(0.5 * (cprop + cdet), 0, 1)
        K[cprop <= 0] = 0.0
        w = K / r * np.exp(1j * k * (opl[i] + gather.C_AMBIENT_N * r))
        ref_x += w * (E3[i] @ det.xhat)
        ref_y += w * (E3[i] @ det.yhat)
    ref_x *= -1j / LAM
    ref_y *= -1j / LAM
    scale = np.abs(ref_x).max()
    assert np.max(np.abs(Ex.reshape(-1) - ref_x)) / scale < 1e-4
    assert np.max(np.abs(Ey.reshape(-1) - ref_y)) / scale < 1e-4


def test_point_source_inverse_square():
    det = _detector(half=20e-3, resolution=64)
    m = 1
    pos = np.zeros((1, 3))
    dirp = np.array([[1.0, 0.0, 0.0]])
    E3 = np.array([[0.0, 1.0, 0.0]], dtype=complex)
    Ex, Ey = gather.accumulate_numpy(pos, E3, np.array([LAM]),
                                     np.zeros(1), det, dirp)
    inten = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    pix = det.pixel_centers
    r2 = np.sum(pix ** 2, axis=-1)
    rhat_x = pix[..., 0] / np.sqrt(r2)
    K = 0.5 * (rhat_x + rhat_x)                  # cprop == cdet here
    expect = (K / np.sqrt(r2)) ** 2 / LAM ** 2   # |Ey| ~ K/r * 1/lam
    ratio = inten / expect
    ratio /= ratio[32, 32]
    # fp32 amplitude accumulation bounds this at ~1e-7 relative; phases
    # are fp64 so there is no systematic error, only rounding
    assert np.max(np.abs(ratio - 1)) < 1e-5
    corner_gain = (inten[0, 0] / inten[32, 32]) \
        / ((K[0, 0] ** 2 / r2[0, 0]) / (K[32, 32] ** 2 / r2[32, 32]))
    assert abs(corner_gain - 1) < 1e-5


def _fringe_pitch(row, pixel_m):
    """Dominant spatial period via FFT peak (excluding DC)."""
    row = row - row.mean()
    f = np.abs(np.fft.rfft(row))
    freqs = np.fft.rfftfreq(len(row), d=pixel_m)
    peak = 1 + np.argmax(f[1:])
    # parabolic interpolation around the peak for sub-bin accuracy
    if 1 <= peak < len(f) - 1:
        a, b, c = f[peak - 1], f[peak], f[peak + 1]
        shift = 0.5 * (a - c) / (a - 2 * b + c)
    else:
        shift = 0.0
    fpk = freqs[peak] + shift * (freqs[1] - freqs[0])
    return 1.0 / fpk


def test_double_slit_fringes():
    rng = np.random.default_rng(7)
    d_sep = 100e-6
    width = 10e-6
    det = _detector(half=4e-3, resolution=1024)
    s = _slit_samples(rng, [-d_sep / 2, d_sep / 2], width,
                      height=2e-3, m_per_slit=3000)
    inten, diags = _render(det, s)
    assert diags[(0, 0)]["effective_samples"] > 1000
    # central horizontal cut (y axis of the grid = global y here)
    # find grid row nearest z=0
    zrow = np.argmin(np.abs(det.pixel_centers[:, 0, :][:, 2] - 0.0)) \
        if det.pixel_centers.shape[0] > 1 else 0
    # average a few central rows for noise robustness
    H = inten.shape[0]
    band = inten[H // 2 - 4:H // 2 + 4].mean(axis=0)
    pitch = _fringe_pitch(band, det.pixel_m)
    expect = LAM * L / d_sep                     # 633 um
    assert pitch == pytest.approx(expect, rel=0.01), (pitch, expect)
    # visibility over the central few fringes
    n_half = int(1.6 * expect / det.pixel_m)
    mid = len(band) // 2
    seg = band[mid - n_half:mid + n_half]
    V = (seg.max() - seg.min()) / (seg.max() + seg.min())
    assert V > 0.9, V


def test_single_slit_first_zero():
    rng = np.random.default_rng(8)
    a = 50e-6                                    # slit width
    det = _detector(half=4e-3, resolution=1024)
    s = _slit_samples(rng, [0.0], a, height=2e-3, m_per_slit=6000)
    inten, _ = _render(det, s)
    H = inten.shape[0]
    # heavy row averaging + light smoothing: the unbiased map carries
    # zero-mean noise that a bare threshold-crossing search trips over
    band = np.maximum(inten[H // 2 - 40:H // 2 + 40].mean(axis=0), 0.0)
    band = np.convolve(band, np.ones(5) / 5, mode="same")
    expect_zero = LAM * L / a                    # 1.266 mm from center
    mid = len(band) // 2
    half = band[mid:]
    # first LOCAL minimum below 15% of the central peak (structure-based,
    # not threshold-crossing)
    mins = [i for i in range(3, len(half) - 3)
            if half[i] <= half[i - 1] and half[i] <= half[i + 1]
            and half[i] < 0.15 * band[mid]
            and i * det.pixel_m > 0.5 * expect_zero]
    assert mins, "no diffraction minimum found"
    first_zero = mins[0] * det.pixel_m
    assert first_zero == pytest.approx(expect_zero, rel=0.05), \
        (first_zero, expect_zero)


def test_incoherent_source_no_fringes():
    # same double slit but with randomized per-sample phases: visibility ~ 0
    rng = np.random.default_rng(9)
    d_sep = 100e-6
    det = _detector(half=4e-3, resolution=512)
    s = _slit_samples(rng, [-d_sep / 2, d_sep / 2], 10e-6,
                      height=2e-3, m_per_slit=4000)
    ph = rng.uniform(0, 2 * np.pi, len(s["Es"]))
    s["Es"] = s["Es"] * np.exp(1j * ph)
    s["Ep"] = s["Ep"] * np.exp(1j * ph)
    # random-phase samples are the physical-speckle population: their
    # incoherent pedestal is real intensity, not MC noise to subtract
    s["scattered"] = np.ones(len(s["Es"]), dtype=bool)
    inten, _ = _render(det, s)
    H = inten.shape[0]
    band = inten[H // 2 - 4:H // 2 + 4].mean(axis=0)
    expect = LAM * L / d_sep
    n_half = int(1.6 * expect / det.pixel_m)
    mid = len(band) // 2
    seg = band[mid - n_half:mid + n_half]
    V = (seg.max() - seg.min()) / (seg.max() + seg.min())
    assert V < 0.35, V     # speckle noise keeps this well below coherent 0.9+


def test_gate_raises_when_undersampled():
    rng = np.random.default_rng(10)
    det = _detector(half=4e-3, resolution=64)
    s = _slit_samples(rng, [0.0], 50e-6, height=2e-3, m_per_slit=5)
    with pytest.raises(gather.GatherError):
        _render(det, s)


def test_torch_backend_matches_numpy():
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
    except ImportError:
        pytest.skip("no torch")
    rng = np.random.default_rng(11)
    det_n = _detector(half=2e-3, resolution=256)
    det_t = _detector(half=2e-3, resolution=256)
    s = _slit_samples(rng, [-50e-6, 50e-6], 10e-6, height=1e-3,
                      m_per_slit=2000)
    inten_n, _ = _render(det_n, s, backend="numpy")
    inten_t, _ = _render(det_t, s, backend="torch")
    scale = inten_n.max()
    assert np.max(np.abs(inten_n - inten_t)) / scale < 5e-3
