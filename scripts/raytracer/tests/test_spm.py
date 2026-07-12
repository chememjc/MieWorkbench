# =============================================================================
# test_spm.py — pulsed-optics Phase P6: the source-side self-phase-
# modulation transform (sources.install_spm) + the sc_superk SPD entry.
#
# Covers (the plan's locked gates):
#   * RMS spectral width vs Delta_omega_in * sqrt(1 + (4/(3 sqrt 3))
#     phi_max^2) (Agrawal, Gaussian pulse) at 10%.
#   * multi-peak count ~= phi_max/pi + 1.
#   * chirp tilt sign: the reddest stratum is born EARLIEST (leading edge
#     red, trailing blue).
#   * SPD normalization + supersession of lambdamin/lambdamax.
#   * spm grammar: phimax form, gamma:length form (phi_max = gamma *
#     P_pk * L_eff), error cases.
#   * sc_superk emission row loads (37 points, 400-2400 nm).
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_spm.py -q
# =============================================================================
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from raytracer.sources import (install_spm, parse_spm_spec,   # noqa: E402
                               wavelength_strata, C_LIGHT_MPS)
from raytracer.scene import _parse_pulse_source               # noqa: E402
from raytracer.optprops import load_optical_properties        # noqa: E402


def _spm_source(phimax=None, gamma=None, length=None, tau_ps=0.1,
                lam_nm=1560.0, energy_uj=0.01, rep_hz=8e7):
    src = {"lambdac_nm": lam_nm, "power_mW": 0.0,
           "pulse_energy_uJ": energy_uj, "pulse_duration_ps": tau_ps,
           "rep_rate_hz": rep_hz}
    _parse_pulse_source("Src", src)
    if phimax is not None:
        src["spm"] = "phimax:%g" % phimax
    else:
        src["spm"] = "gamma:%g:length:%g" % (gamma, length)
    return src


def _install(src, n_lambda=9):
    scene = SimpleNamespace(sources=[(0, src)])
    install_spm(scene, n_lambda)
    return src


def _rms_omega(lam_nm, pdf):
    """Power-weighted RMS width of the SPD in angular frequency."""
    w = 2.0 * np.pi * C_LIGHT_MPS / (lam_nm * 1e-9)
    order = np.argsort(w)
    w, p = w[order], pdf[order]
    mean = np.trapezoid(p * w, w) / np.trapezoid(p, w)
    var = np.trapezoid(p * (w - mean) ** 2, w) / np.trapezoid(p, w)
    return np.sqrt(var)


# --------------------------------------------------------------------------- #
# 1. RMS broadening factor (Agrawal) at 10%
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("phimax", [1.5, 4.0, 8.0])
def test_rms_broadening_matches_agrawal(phimax):
    ref = _install(_spm_source(phimax=1e-6))     # ~no SPM: input width
    out = _install(_spm_source(phimax=phimax))
    r_in = _rms_omega(ref["_spectrum_lam_nm"], ref["_spectrum_pdf"])
    r_out = _rms_omega(out["_spectrum_lam_nm"], out["_spectrum_pdf"])
    expect = np.sqrt(1.0 + (4.0 / (3.0 * np.sqrt(3.0))) * phimax ** 2)
    assert r_out / r_in == pytest.approx(expect, rel=0.10)


# --------------------------------------------------------------------------- #
# 2. multi-peak count ~ phi_max/pi + 1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("phimax,lo,hi", [
    (np.pi * 1.5, 2, 3), (np.pi * 3.5, 4, 5), (np.pi * 6.5, 7, 8)])
def test_peak_count(phimax, lo, hi):
    src = _install(_spm_source(phimax=phimax))
    p = src["_spectrum_pdf"]
    peaks = np.sum((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:])
                   & (p[1:-1] > 0.02))
    assert lo <= peaks <= hi, "phimax=%.3g: %d peaks" % (phimax, peaks)


# --------------------------------------------------------------------------- #
# 3. chirp tilt: leading edge red, trailing blue
# --------------------------------------------------------------------------- #
def test_stratum_t0_red_first():
    src = _install(_spm_source(phimax=5.0), n_lambda=9)
    lam = np.asarray(wavelength_strata(src, 9))
    t0 = src["_stratum_t0"]
    assert t0.shape == (9,)
    # reddest (longest lambda) stratum born earliest, bluest last
    assert t0[np.argmax(lam)] < 0 < t0[np.argmin(lam)]
    # monotonic: t0 decreases with wavelength (increases with omega)
    order = np.argsort(lam)
    assert np.all(np.diff(t0[order]) <= 1e-18)
    # birth offsets live inside the pulse: |t0| < a few tau
    assert np.all(np.abs(t0) < 5e-13)


# --------------------------------------------------------------------------- #
# 4. SPD install details
# --------------------------------------------------------------------------- #
def test_spd_normalized_and_supersedes_bounds():
    src = _spm_source(phimax=4.0)
    src["lambdamin_nm"], src["lambdamax_nm"] = 1500.0, 1620.0
    _install(src)
    assert src["_spectrum_pdf"].max() == pytest.approx(1.0)
    assert src["lambdamin_nm"] is None and src["lambdamax_nm"] is None
    lam = src["_spectrum_lam_nm"]
    assert np.all(np.diff(lam) > 0)               # ascending, no dupes
    # phimax=4 broadens the ~36 nm transform-limited width ~3.7x (the
    # Agrawal factor): the table must span well beyond the input band on
    # both sides of the carrier
    assert lam.min() < 1500.0 < 1560.0 < 1620.0 < lam.max()
    assert src["_spm_phimax"] == pytest.approx(4.0)
    # strata sample inside the installed table
    s = np.asarray(wavelength_strata(src, 5))
    assert lam.min() * 1e-9 <= s.min() <= s.max() <= lam.max() * 1e-9


# --------------------------------------------------------------------------- #
# 5. grammar + gamma form
# --------------------------------------------------------------------------- #
def test_gamma_length_form_derives_phimax():
    # 10 nJ / 100 fs Gaussian: P_pk = 0.94 * 1e-8 / 1e-13 = 94 kW;
    # gamma 11.5 W^-1 km^-1 * 0.1 m -> phi_max = 11.5e-3 * 94e3 * 0.1
    src = _install(_spm_source(gamma=11.5, length=0.1))
    assert src["_spm_phimax"] == pytest.approx(11.5e-3 * 94e3 * 0.1,
                                               rel=1e-9)


def test_parse_spm_spec_errors():
    assert parse_spm_spec("phimax:3.5") == {"phimax": 3.5}
    assert parse_spm_spec("gamma:11.5:length:0.1") == {
        "gamma_W_km": 11.5, "length_m": 0.1}
    for bad in ("phimax:0", "phimax:x", "gamma:1:len:2", "nope", ""):
        with pytest.raises(ValueError):
            parse_spm_spec(bad)


def test_spm_needs_pulse_duration():
    src = {"lambdac_nm": 1560.0, "power_mW": 5.0, "spm": "phimax:3"}
    with pytest.raises(ValueError, match="pulse_duration"):
        _install(src)


# --------------------------------------------------------------------------- #
# 6. sc_superk emission entry
# --------------------------------------------------------------------------- #
def test_sc_superk_row_loads():
    e = load_optical_properties().emission["sc_superk"]
    lam = np.asarray(e["lam_nm"], dtype=float)
    assert len(lam) == 37
    assert lam[0] == 400.0 and lam[-1] == 2400.0
    assert e["kind"] == "continuous"
