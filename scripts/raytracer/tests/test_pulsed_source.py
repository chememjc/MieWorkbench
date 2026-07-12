# =============================================================================
# test_pulsed_source.py — pulsed-optics Phase P3: the pulsed source model.
#
# Covers:
#   * raytracer.scene._parse_pulse_source (exercised through Scene(), the
#     real call site — see scene.py's Scene.__init__ source-processing
#     block): power/pulse_energy XOR error, missing-neither error,
#     pulse_energy+rep_rate -> derived power_mW (exact value),
#     power+rep_rate -> derived pulse_energy_J, peak_power_W = 0.94*E/tau,
#     kappa = peak/avg, CW-with-duration-only mode, positivity errors.
#   * raytracer.sources.apply_stratum_t0: the birth-time-offset hook a
#     later SPM/chirp phase will drive — no-op when _stratum_t0 is
#     absent or batch.gopl isn't allocated yet, otherwise
#     gopl += C_LIGHT_MPS * t0[lam_stratum].
#   * the case.json "source_pulse" block (run_trace.py) — a scene-level
#     dict assert of the exact expression run_trace.py uses, since
#     spinning up the full CLI pipeline here would be out of scope.
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_pulsed_source.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import common                                              # noqa: E402
from raytracer.scene import Scene                           # noqa: E402
from raytracer.sources import (sample_source, apply_stratum_t0,  # noqa: E402
                               C_LIGHT_MPS)
from raytracer.optprops import load_optical_properties      # noqa: E402
from . import scenehelpers as sh                            # noqa: E402


def _scene(src_kwargs):
    """Build a trivial source+detector Scene with the given source_body()
    kwargs; returns (scene, src dict). validate_model runs first, exactly
    like the real pipeline (power_mW=0.0 is a valid float, so a
    pulse_energy-only source still clears that pre-existing gate)."""
    model = sh.make_model([
        sh.source_body(**src_kwargs),
        sh.detector_body(x=0.03, half=0.02),
    ])
    common.validate_model(model)
    opt = load_optical_properties()
    scene = Scene(model, opt.matdb, opt.coatings, optprops=opt)
    bidx, src = scene.sources[0]
    return scene, src


# --------------------------------------------------------------------------- #
# 1. power XOR pulse_energy
# --------------------------------------------------------------------------- #
def test_both_power_and_pulse_energy_is_an_error():
    with pytest.raises(ValueError) as exc:
        _scene(dict(power_mW=5.0, pulse_energy_uJ=10.0, rep_rate_hz=1000.0))
    msg = str(exc.value)
    assert "both power" in msg and "pulse_energy" in msg
    assert "5" in msg and "10" in msg


def test_neither_power_nor_pulse_energy_is_an_error():
    with pytest.raises(ValueError) as exc:
        _scene(dict(power_mW=0.0))
    assert "needs either" in str(exc.value)


# --------------------------------------------------------------------------- #
# 2. pulse_energy + rep_rate -> derives power_mW (exact value)
# --------------------------------------------------------------------------- #
def test_pulse_energy_and_rep_rate_derives_power_mw():
    # 10 uJ at 1 kHz -> 10e-6 J * 1000 Hz = 0.01 W = 10 mW
    scene, src = _scene(dict(power_mW=0.0, pulse_energy_uJ=10.0,
                             rep_rate_hz=1000.0))
    assert src["power_mW"] == pytest.approx(10.0, rel=1e-12)
    assert src["pulse"]["derived"] == "power"
    assert src["pulse"]["energy_J"] == pytest.approx(10e-6, rel=1e-12)
    assert src["pulse"]["avg_power_W"] == pytest.approx(0.01, rel=1e-12)
    assert src["pulse"]["rep_rate_Hz"] == 1000.0
    # body.source is the SAME dict object -- mutation is visible there too
    assert scene.bodies[scene.sources[0][0]].source["power_mW"] == \
        pytest.approx(10.0, rel=1e-12)


def test_pulse_energy_requires_rep_rate():
    with pytest.raises(ValueError) as exc:
        _scene(dict(power_mW=0.0, pulse_energy_uJ=10.0))
    assert "pulse_energy needs rep_rate" in str(exc.value)


# --------------------------------------------------------------------------- #
# 3. power + rep_rate -> reverse-derives pulse_energy_J
# --------------------------------------------------------------------------- #
def test_power_and_rep_rate_derives_pulse_energy():
    # 5 mW at 1 kHz -> 0.005 W / 1000 Hz = 5e-6 J per pulse
    scene, src = _scene(dict(power_mW=5.0, rep_rate_hz=1000.0))
    assert src["power_mW"] == 5.0             # power itself is untouched
    assert src["pulse"]["derived"] == "pulse_energy"
    assert src["pulse"]["energy_J"] == pytest.approx(5e-6, rel=1e-12)
    assert src["pulse"]["avg_power_W"] == pytest.approx(0.005, rel=1e-12)


# --------------------------------------------------------------------------- #
# 4/5. peak power (0.94*E/tau) and kappa (peak/avg)
# --------------------------------------------------------------------------- #
def test_peak_power_and_kappa():
    # 10 uJ @ 1 kHz (-> 10 mW avg), 100 ps FWHM
    scene, src = _scene(dict(power_mW=0.0, pulse_energy_uJ=10.0,
                             rep_rate_hz=1000.0, pulse_duration_ps=100.0))
    energy_j = 10e-6
    duration_s = 100e-12
    want_peak = 0.94 * energy_j / duration_s
    assert src["pulse"]["duration_s"] == pytest.approx(duration_s, rel=1e-12)
    assert src["pulse"]["peak_power_W"] == pytest.approx(want_peak, rel=1e-12)
    avg = src["pulse"]["avg_power_W"]
    assert src["pulse"]["kappa"] == pytest.approx(want_peak / avg, rel=1e-12)


def test_peak_power_none_when_duration_absent():
    scene, src = _scene(dict(power_mW=0.0, pulse_energy_uJ=10.0,
                             rep_rate_hz=1000.0))
    assert src["pulse"]["duration_s"] is None
    assert src["pulse"]["peak_power_W"] is None
    assert src["pulse"]["kappa"] is None


# --------------------------------------------------------------------------- #
# 6. CW virtual-pulse mode: duration only, no energy, no rep_rate
# --------------------------------------------------------------------------- #
def test_cw_duration_only_mode_is_legal():
    scene, src = _scene(dict(power_mW=5.0, pulse_duration_ps=100.0))
    assert src["power_mW"] == 5.0
    pulse = src["pulse"]
    assert pulse["duration_s"] == pytest.approx(100e-12, rel=1e-12)
    assert pulse["energy_J"] is None
    assert pulse["rep_rate_Hz"] is None
    assert pulse["peak_power_W"] is None
    assert pulse["kappa"] is None
    assert pulse["derived"] is None
    assert pulse["avg_power_W"] == pytest.approx(0.005, rel=1e-12)


def test_ordinary_source_gets_no_pulse_key():
    scene, src = _scene(dict(power_mW=5.0))
    assert "pulse" not in src


# --------------------------------------------------------------------------- #
# 7. positivity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs,msg", [
    (dict(power_mW=0.0, pulse_energy_uJ=-1.0, rep_rate_hz=1000.0),
     "pulse_energy must be > 0"),
    (dict(power_mW=5.0, pulse_duration_ps=0.0), "pulse_duration must be > 0"),
    (dict(power_mW=0.0, pulse_energy_uJ=1.0, rep_rate_hz=-5.0),
     "rep_rate must be > 0"),
])
def test_bad_pulse_values_are_errors(kwargs, msg):
    with pytest.raises(ValueError) as exc:
        _scene(kwargs)
    assert msg in str(exc.value)


# --------------------------------------------------------------------------- #
# 8. apply_stratum_t0: birth-time offset hook
# --------------------------------------------------------------------------- #
def test_stratum_t0_shifts_gopl_by_c_times_t0():
    # lambdamin/lambdamax bracket lambdac so wavelength_strata actually
    # returns n_lambda distinct strata (a bare monochromatic source
    # collapses to a single stratum regardless of n_lambda)
    scene, src = _scene(dict(power_mW=1.0, lambdamin_nm=620.0,
                             lambdamax_nm=645.0))
    bidx, _ = scene.sources[0]
    n_lambda = 3
    t0 = np.array([0.0, 1e-12, 2.5e-12])
    src["_stratum_t0"] = t0
    rng = np.random.default_rng(7)
    batch = sample_source(scene, scene.bodies[bidx], src, 0, 300, n_lambda,
                          rng)
    assert np.any(batch.lam_stratum == 0)
    assert np.any(batch.lam_stratum == 1)
    assert np.any(batch.lam_stratum == 2)

    # no-op before gopl is allocated (the real pipeline's actual state
    # right after sample_source returns -- see apply_stratum_t0's
    # docstring for why)
    assert batch.gopl is None
    apply_stratum_t0(batch, src)
    assert batch.gopl is None

    # caller pre-allocates (mirrors run_trace.py's batch-building loops
    # under cfg.track_time), THEN the hook applies
    batch.alloc_time()
    assert np.all(batch.gopl == 0.0)
    apply_stratum_t0(batch, src)
    want = C_LIGHT_MPS * t0[batch.lam_stratum]
    assert np.allclose(batch.gopl, want, rtol=0.0, atol=1e-30)
    assert np.any(batch.gopl != 0.0)   # strata 1 and 2 actually moved


def test_stratum_t0_absent_is_a_no_op():
    scene, src = _scene(dict(power_mW=1.0))    # no _stratum_t0 key
    bidx, _ = scene.sources[0]
    rng = np.random.default_rng(3)
    batch = sample_source(scene, scene.bodies[bidx], src, 0, 50, 1, rng)
    batch.alloc_time()
    before = batch.gopl.copy()
    apply_stratum_t0(batch, src)
    assert np.array_equal(batch.gopl, before)
    assert np.all(batch.gopl == 0.0)


# --------------------------------------------------------------------------- #
# 9. case.json "source_pulse" block (run_trace.py) -- scene-level dict
#    assert of the exact expression run_trace.py's _main_locked uses
# --------------------------------------------------------------------------- #
def test_case_json_source_pulse_block_shape():
    scene, src = _scene(dict(power_mW=0.0, pulse_energy_uJ=10.0,
                             rep_rate_hz=1000.0))
    source_pulse = {scene.bodies[b].label: s["pulse"]
                    for b, s in scene.sources if "pulse" in s}
    assert set(source_pulse) == {"Src"}
    pulse = source_pulse["Src"]
    assert set(pulse) == {"energy_J", "duration_s", "rep_rate_Hz",
                         "peak_power_W", "avg_power_W", "derived", "kappa"}
    assert pulse["derived"] == "power"


def test_case_json_source_pulse_block_empty_for_ordinary_source():
    scene, src = _scene(dict(power_mW=5.0))
    source_pulse = {scene.bodies[b].label: s["pulse"]
                    for b, s in scene.sources if "pulse" in s}
    assert source_pulse == {}
