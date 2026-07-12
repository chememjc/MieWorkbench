# =============================================================================
# test_time_products.py — pulsed-optics Phase P4: the time-binned detector,
# the four selectable time products, and their CLI/engine-routing glue.
#
# Covers:
#   * discrete energy conservation: sum(product) * dt == the summed
#     in-window arrival-record power (time_total_W) for BOTH envelope
#     modes, and that total matches the detected per-key tallies exactly
#     (1e-12 relative) — the coherent population via its GEOMETRIC power.
#   * marginals: spectrogram/streak/by-source marginals equal the profile
#     (float64), cube marginal within float32 accumulation tolerance.
#   * auto window covers every record (nothing excluded); explicit
#     --time-window is respected and a window-CLIPPED kernel still
#     conserves energy over the in-window bins.
#   * analytic vs histogram: same total, analytic smoother.
#   * CW virtual pulse: a delta at t = d/c (degenerate-window guard).
#   * broadening sanity: 100 fs through 20 mm fused silica vs
#     tau(phi2) = tau0*sqrt(1+(4 ln2 phi2/tau0^2)^2) within 5%.
#   * cengine routing: any active product records engine=python.
#   * the auto-enable rule (pulsed -> pulse,spectrogram; CW -> nothing;
#     explicit 'none' -> nothing).
#   * wavelength_strata's new StratumWavelengths return type (edges +
#     stratum_domega) and its backward compatibility.
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_time_products.py -q
# =============================================================================
import json
import sys
from pathlib import Path

import numpy as np
import h5py
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "raytracer" / "tests"))

import common                                              # noqa: E402
import cli_specs                                            # noqa: E402
import run_trace                                            # noqa: E402
import run_pipeline                                         # noqa: E402
import scenehelpers as sh                                  # noqa: E402
from raytracer.sources import (wavelength_strata,          # noqa: E402
                               stratum_domega)
from raytracer.materials import gdd_per_length             # noqa: E402
from raytracer.optprops import load_optical_properties     # noqa: E402
from raytracer import cengine                              # noqa: E402
from post_process import _fwhm_interp                      # noqa: E402

C = 299792458.0
DET = "Det.Pad.Face1"


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
def _run(tmp_path, bodies, extra, name="case", rays=3000, nlambda=3):
    model = sh.make_model(bodies)
    common.validate_model(model)
    mj = tmp_path / (name + "_model.json")
    common.write_json(mj, model)
    case = tmp_path / name
    rc = run_trace.main([
        "--model-json", str(mj), "--case-dir", str(case),
        "--rays", str(rays), "--nlambda", str(nlambda),
        "--resolution", "48", "--workers", "1"] + extra)
    assert rc == 0
    return case


def _h5cols(case):
    with h5py.File(case / "detectors" / "Det_Pad_Face1.h5") as h:
        data = {k: h[k][...] for k in h.keys() if k.startswith("time_")}
        attrs = dict(h.attrs)
    return data, attrs


def _case_json(case):
    return json.loads((case / "case.json").read_text())


def _detected_W(case, kind):
    """Summed per-key detected power ('incoherent_W'/'coherent_W') for the
    single detector, from case.json's per-seed detected block."""
    cj = _case_json(case)
    seed_block = next(iter(cj["detected"].values()))
    return sum(v.get(kind, 0.0) for v in seed_block[DET].values())


def _two_slab(pulsed=True, coherent=False):
    src = dict(x=-0.02, half=0.001, power_mW=1.0, lambdac_nm=633.0,
               lambdamin_nm=630.0, lambdamax_nm=636.0, coherent=coherent)
    if pulsed:
        src["pulse_duration_ps"] = 0.5
    return [sh.source_body(**src),
            sh.slab_body("SlabA", "bk7", 0.0, 0.004, half=0.01),
            sh.slab_body("SlabB", "fused_silica", 0.006, 0.011, half=0.01),
            sh.detector_body(x=0.03, half=0.02)]


# --------------------------------------------------------------------------- #
# 1. discrete energy conservation, both envelope modes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("envelope", ["analytic", "histogram"])
def test_energy_conservation_every_product(tmp_path, envelope):
    case = _run(tmp_path, _two_slab(), [
        "--time-products", "all", "--time-bins", "64",
        "--time-envelope", envelope], name=envelope)
    data, attrs = _h5cols(case)
    dt = float(attrs["time_dt_s"])
    total = float(attrs["time_total_W"])
    assert total > 0.0
    # the auto window covers everything (nothing excluded here)
    assert attrs["time_excluded_W"] == 0.0
    # every float64 product integrates to the in-window record power
    for name in ("time_profile", "time_profile_by_source",
                 "time_spectrogram", "time_streak"):
        got = float(data[name].sum()) * dt
        assert abs(got - total) <= 1e-12 * total, (name, got, total)
    # float32 cube: accumulation tolerance
    got = float(data["time_cube"].astype(np.float64).sum()) * dt
    assert abs(got - total) <= 1e-5 * total
    # ... and the record total IS the detected (incoherent) power
    det = _detected_W(case, "incoherent_W")
    assert abs(total - det) <= 1e-12 * det


def test_coherent_records_carry_geometric_power(tmp_path):
    """Coherent rays contribute their GEOMETRIC power at their geometric
    arrival time (fringe-resolved timing out of scope): the profile total
    equals the detected_geometric tally exactly, and the arrival lands at
    d/c for the free-space path."""
    d = 0.05                       # source at -0.02, detector at +0.03
    case = _run(tmp_path, [
        sh.source_body(coherent=True, pulse_duration_ps=0.5),
        sh.detector_body(x=0.03, half=0.02),
    ], ["--time-products", "pulse", "--time-bins", "64",
        "--no-gather-gate"], rays=2000, nlambda=1)
    data, attrs = _h5cols(case)
    dt = float(attrs["time_dt_s"])
    total = float(attrs["time_total_W"])
    got = float(data["time_profile"].sum()) * dt
    assert abs(got - total) <= 1e-12 * total
    det = _detected_W(case, "coherent_W")
    assert abs(total - det) <= 1e-12 * det
    prof = data["time_profile"]
    t_pk = float(attrs["t_lo_s"]) + (np.argmax(prof) + 0.5) * dt
    assert abs(t_pk - d / C) <= 3.0 * dt + 0.5e-12


# --------------------------------------------------------------------------- #
# 2. marginals
# --------------------------------------------------------------------------- #
def test_marginals_match_profile(tmp_path):
    case = _run(tmp_path, _two_slab(), [
        "--time-products", "all", "--time-bins", "64"])
    data, _ = _h5cols(case)
    prof = data["time_profile"]
    atol = 1e-11 * prof.max()
    assert np.allclose(data["time_spectrogram"].sum(axis=0), prof,
                       rtol=1e-9, atol=atol)
    assert np.allclose(data["time_streak"].sum(axis=1), prof,
                       rtol=1e-9, atol=atol)
    assert np.allclose(data["time_profile_by_source"].sum(axis=0), prof,
                       rtol=1e-12, atol=0.0)
    # float32 cube
    assert np.allclose(
        data["time_cube"].astype(np.float64).sum(axis=(1, 2)), prof,
        rtol=2e-3, atol=1e-3 * prof.max())


# --------------------------------------------------------------------------- #
# 3. windows: auto coverage / explicit override / clipped kernels
# --------------------------------------------------------------------------- #
def test_auto_window_covers_all_records(tmp_path):
    case = _run(tmp_path, _two_slab(), [
        "--time-products", "pulse", "--time-bins", "64"])
    _, attrs = _h5cols(case)
    assert attrs["time_excluded_W"] == 0.0
    assert not attrs["time_window_explicit"]
    assert attrs["t_lo_s"] <= attrs["t_p001_s"] \
        <= attrs["t_p999_s"] <= attrs["t_hi_s"]


def test_explicit_window_respected_and_clipped_kernels_conserve(tmp_path):
    bodies = _two_slab()
    auto = _run(tmp_path, bodies, [
        "--time-products", "pulse", "--time-bins", "64"], name="auto")
    _, a0 = _h5cols(auto)
    det = _detected_W(auto, "incoherent_W")
    # cut through the arrival distribution: from just above the earliest
    # arrivals to midway — kernels straddling either edge get clipped
    w0 = float(a0["t_p001_s"]) + 0.2e-12
    w1 = 0.5 * (float(a0["t_p001_s"]) + float(a0["t_p999_s"]))
    spec = "%.12g,%.12g" % (w0 / 1e-9, w1 / 1e-9)
    case = _run(tmp_path, bodies, [
        "--time-products", "pulse", "--time-bins", "64",
        "--time-window", spec], name="win")
    data, attrs = _h5cols(case)
    assert attrs["time_window_explicit"]
    assert attrs["t_lo_s"] == pytest.approx(w0, rel=1e-9)
    assert attrs["t_hi_s"] == pytest.approx(w1, rel=1e-9)
    dt = float(attrs["time_dt_s"])
    total = float(attrs["time_total_W"])
    excl = float(attrs["time_excluded_W"])
    # the window genuinely clipped something...
    assert excl > 0.0
    # ...clipped kernels are renormalized over the in-window bins
    got = float(data["time_profile"].sum()) * dt
    assert abs(got - total) <= 1e-12 * max(total, 1e-30)
    # ...and in-window + excluded still accounts for every detected watt
    assert abs((total + excl) - det) <= 1e-12 * det


# --------------------------------------------------------------------------- #
# 4. analytic vs histogram: same total, analytic smoother
# --------------------------------------------------------------------------- #
def test_analytic_smoother_than_histogram(tmp_path):
    bodies = _two_slab()
    auto = _run(tmp_path, bodies, [
        "--time-products", "pulse", "--time-bins", "128"], name="auto")
    _, a0 = _h5cols(auto)
    # shared explicit window so the two runs bin identically
    spec = "%.12g,%.12g" % (float(a0["t_lo_s"]) / 1e-9,
                            float(a0["t_hi_s"]) / 1e-9)
    prof = {}
    tot = {}
    for env in ("analytic", "histogram"):
        case = _run(tmp_path, bodies, [
            "--time-products", "pulse", "--time-bins", "128",
            "--time-window", spec, "--time-envelope", env], name=env)
        data, attrs = _h5cols(case)
        prof[env] = data["time_profile"]
        tot[env] = float(attrs["time_total_W"])
    assert tot["analytic"] == pytest.approx(tot["histogram"], rel=1e-12)
    v_a = float(np.var(np.diff(prof["analytic"])))
    v_h = float(np.var(np.diff(prof["histogram"])))
    assert v_a < v_h


# --------------------------------------------------------------------------- #
# 5. CW virtual pulse: delta at t = d/c
# --------------------------------------------------------------------------- #
def test_cw_source_is_a_delta_at_d_over_c(tmp_path):
    d = 0.05
    case = _run(tmp_path, [
        sh.source_body(),                    # plain CW, monochromatic
        sh.detector_body(x=0.03, half=0.02),
    ], ["--time-products", "pulse", "--time-bins", "64"],
        rays=2000, nlambda=1)
    data, attrs = _h5cols(case)
    prof = data["time_profile"]
    dt = float(attrs["time_dt_s"])
    # collimated free-space path: every ray arrives at exactly d/c — the
    # spread is path differences only, which are zero here, so the delta
    # occupies at most two adjacent bins (degenerate-window guard opens a
    # minimal window around the single arrival time)
    nz = np.flatnonzero(prof > 0)
    assert 1 <= len(nz) <= 2
    assert nz.max() - nz.min() <= 1
    t_pk = float(attrs["t_lo_s"]) + (np.argmax(prof) + 0.5) * dt
    assert abs(t_pk - d / C) <= max(dt, 1e-12)
    total = float(attrs["time_total_W"])
    assert float(prof.sum()) * dt == pytest.approx(total, rel=1e-12)


# --------------------------------------------------------------------------- #
# 6. broadening sanity: 100 fs through 20 mm fused silica (5% gate; the
#    tight 2% gate against the GDD-budget table is the NEXT phase's)
# --------------------------------------------------------------------------- #
def test_pulse_broadening_fused_silica_within_5pct(tmp_path):
    tau0 = 100e-15
    lam0 = 400.0                                    # nm (strong GDD)
    L = 0.02
    # transform-limited Gaussian: sigma_omega = (4 ln2 / tau0)/(2 sqrt(2 ln2))
    sig_w = 4.0 * np.log(2.0) / tau0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sig_lam_nm = (lam0 * 1e-9) ** 2 * sig_w / (2.0 * np.pi * C) / 1e-9

    silica = load_optical_properties().matdb.get("fused_silica")
    phi2 = float(gdd_per_length(silica, lam0 * 1e-9)) * L
    expect = tau0 * np.sqrt(1.0 + (4.0 * np.log(2.0) * phi2
                                   / tau0 ** 2) ** 2)
    # window: +-1 ps around the main-pulse group delay (excludes the tiny
    # double-bounce slab echoes ~200 ps later that would blow up dt)
    n_g = float(silica.n_group(lam0 * 1e-9))
    t_c = (0.03 + n_g * L) / C
    spec = "%.12g,%.12g" % ((t_c - 1e-12) / 1e-9, (t_c + 1e-12) / 1e-9)

    case = _run(tmp_path, [
        sh.source_body(x=-0.02, half=0.001, power_mW=1.0,
                       lambdac_nm=lam0,
                       lambdamin_nm=lam0 - sig_lam_nm,
                       lambdamax_nm=lam0 + sig_lam_nm,
                       pulse_duration_ps=tau0 / 1e-12),
        sh.slab_body("Slab", "fused_silica", 0.0, L, half=0.01),
        sh.detector_body(x=0.03, half=0.02),
    ], ["--time-products", "pulse", "--time-bins", "512",
        "--time-window", spec], rays=10000, nlambda=9)
    data, attrs = _h5cols(case)
    prof = data["time_profile"]
    tc = float(attrs["t_lo_s"]) \
        + (np.arange(len(prof)) + 0.5) * float(attrs["time_dt_s"])
    fwhm, _, _ = _fwhm_interp(tc, prof)
    assert fwhm is not None
    assert abs(fwhm - expect) / expect < 0.05, \
        "measured %.4g fs vs theory %.4g fs" % (fwhm / 1e-15,
                                                expect / 1e-15)


# --------------------------------------------------------------------------- #
# 7. cengine routing: time products force the Python engine
# --------------------------------------------------------------------------- #
def test_time_products_route_python_with_reason(tmp_path):
    case = _run(tmp_path, [
        sh.source_body(pulse_duration_ps=1.0),
        sh.detector_body(x=0.03, half=0.02),
    ], [], rays=500, nlambda=1)                     # auto-enabled
    cj = _case_json(case)
    assert cj["engine"] == "python"
    if cengine.binary_path() is not None:
        assert "time_products" in cj["engine_reason"]


def test_explicit_none_does_not_gate_the_engine(tmp_path):
    case = _run(tmp_path, [
        sh.source_body(pulse_duration_ps=1.0),
        sh.detector_body(x=0.03, half=0.02),
    ], ["--time-products", "none"], rays=500, nlambda=1)
    cj = _case_json(case)
    assert "time_products" not in cj.get("engine_reason", "")


# --------------------------------------------------------------------------- #
# 8. auto-enable rule
# --------------------------------------------------------------------------- #
def test_auto_enable_pulsed_defaults_to_pulse_and_spectrogram(tmp_path):
    case = _run(tmp_path, [
        sh.source_body(pulse_duration_ps=1.0),
        sh.detector_body(x=0.03, half=0.02),
    ], [], rays=500, nlambda=1)
    data, attrs = _h5cols(case)
    assert set(data) == {"time_profile", "time_profile_by_source",
                         "time_spectrogram"}
    assert attrs["time_products"] == "pulse,spectrogram"
    cj = _case_json(case)
    assert cj["time_products"]["auto_enabled"] is True
    assert cj["time_products"]["products"] == ["pulse", "spectrogram"]


def test_cw_scene_without_flag_gets_no_time_datasets(tmp_path):
    case = _run(tmp_path, [
        sh.source_body(), sh.detector_body(x=0.03, half=0.02),
    ], [], rays=500, nlambda=1)
    data, attrs = _h5cols(case)
    assert data == {}
    assert not any(str(k).startswith(("time_", "t_lo", "t_hi", "t_p"))
                   for k in attrs)
    assert "time_products" not in _case_json(case)


def test_explicit_none_suppresses_the_auto_default(tmp_path):
    case = _run(tmp_path, [
        sh.source_body(pulse_duration_ps=1.0),
        sh.detector_body(x=0.03, half=0.02),
    ], ["--time-products", "none"], rays=500, nlambda=1)
    data, _ = _h5cols(case)
    assert data == {}
    assert "time_products" not in _case_json(case)


# --------------------------------------------------------------------------- #
# 9. wavelength_strata: edges + stratum_domega (API change, backward compat)
# --------------------------------------------------------------------------- #
def test_strata_uniform_band_values_and_edges():
    src = {"lambdac_nm": 633.0, "lambdamin_nm": 630.0}
    n = 4
    strata = wavelength_strata(src, n)
    # backward compat: same center values as the pre-edges formula
    q = (np.arange(n) + 0.5) / n
    want = (633.0 - 3.0 + 6.0 * q) * 1e-9
    assert np.allclose(np.asarray(strata), want, rtol=0.0, atol=0.0)
    assert len(strata) == n and float(strata[0]) > 0
    # edges: exact band quantiles
    want_e = (630.0 + 6.0 * np.arange(n + 1) / n) * 1e-9
    assert np.allclose(strata.edges, want_e, rtol=0.0, atol=0.0)
    dw = stratum_domega(strata)
    assert dw.shape == (n,) and np.all(dw > 0)


def test_strata_monochromatic_zero_width():
    strata = wavelength_strata({"lambdac_nm": 633.0}, 7)
    assert len(strata) == 1
    assert np.allclose(strata.edges, 633e-9)
    assert stratum_domega(strata) == pytest.approx(0.0, abs=0.0)


def test_strata_gaussian_edges_finite_and_bracketing():
    src = {"lambdac_nm": 633.0, "lambdamin_nm": 630.0,
           "lambdamax_nm": 638.0}
    n = 9
    strata = wavelength_strata(src, n)
    e = strata.edges
    assert e.shape == (n + 1,)
    assert np.all(np.isfinite(e))
    assert np.all(np.diff(e) > 0)
    # every center lies inside its own edge bracket
    c = np.asarray(strata)
    assert np.all((e[:-1] < c) & (c < e[1:]))
    assert np.all(stratum_domega(strata) > 0)


def test_strata_tabulated_spectrum_edges():
    src = {"lambdac_nm": 550.0,
           "_spectrum_lam_nm": np.array([500.0, 550.0, 600.0]),
           "_spectrum_pdf": np.array([0.0, 1.0, 0.0])}
    n = 5
    strata = wavelength_strata(src, n)
    e = strata.edges
    assert e.shape == (n + 1,)
    assert e[0] == pytest.approx(500e-9, rel=1e-6)
    assert e[-1] == pytest.approx(600e-9, rel=1e-6)
    assert np.all(np.diff(e) > 0)
    c = np.asarray(strata)
    assert np.all((e[:-1] <= c) & (c <= e[1:]))


# --------------------------------------------------------------------------- #
# 10. CLI plumbing: parsers + run_pipeline forwarding
# --------------------------------------------------------------------------- #
def test_cli_specs_parse_time_products():
    assert cli_specs.parse_time_products("all") == cli_specs.TIME_PRODUCTS
    assert cli_specs.parse_time_products("none") == ()
    assert cli_specs.parse_time_products("streak,pulse") == \
        ("pulse", "streak")
    with pytest.raises(Exception):
        cli_specs.parse_time_products("bogus")
    assert cli_specs.parse_time_window("0.1,0.4") == (0.1, 0.4)
    with pytest.raises(Exception):
        cli_specs.parse_time_window("0.4,0.1")


def test_run_pipeline_forwards_time_flags():
    p = cli_specs.build_parser("pipeline")
    args = p.parse_args(["--models", "x.FCStd", "--preset", "quick",
                         "--time-products", "pulse,cube",
                         "--time-window", "0.1,0.4",
                         "--time-envelope", "histogram"])
    cmd = run_pipeline.trace_cmd("x", Path("/tmp/case"), args)
    s = " ".join(cmd)
    assert "--time-products pulse,cube" in s
    assert "--time-bins 128" in s                    # quick preset scale
    assert "--time-window 0.1,0.4" in s
    assert "--time-envelope histogram" in s
    # explicit bins win over the preset; 'none' round-trips
    args = p.parse_args(["--models", "x.FCStd", "--preset", "detailed",
                         "--time-bins", "77", "--time-products", "none"])
    s = " ".join(run_pipeline.trace_cmd("x", Path("/tmp/case"), args))
    assert "--time-bins 77" in s
    assert "--time-products none" in s


# --------------------------------------------------------------------------- #
# 10. splat numerics: fs kernels binned over a ns window (regression)
# --------------------------------------------------------------------------- #
def test_splat_sub_bin_kernels_finite_and_conserve():
    # sigma << dt (a 100 fs pulse under an auto window spanning several
    # path lengths -> ps bins): evaluating the Gaussian at bin centres
    # tens of sigma away used to drive rs subnormal, overflow pc/rs to
    # inf and NaN the neighbour bins via 0*inf. Discretely these kernels
    # ARE deltas — the splat must stay finite and conserve exactly.
    from raytracer.detector import _splat_records
    n_t = 256
    t_lo, dt = 0.0, 4e-12
    n = 1000
    t = np.linspace(1e-12, 1e-9, n)              # all in-window
    sigma = np.full(n, 4.25e-14)                 # 100 fs FWHM kernel
    power = np.full(n, 1e-3)
    acc = np.zeros((1, n_t))
    excl = _splat_records(acc, t, sigma, power,
                          np.zeros(n, np.int64), t_lo, dt)
    assert np.all(np.isfinite(acc))
    assert excl == 0.0
    assert float(acc.sum()) == pytest.approx(n * 1e-3, rel=1e-12)


def test_splat_mixed_width_chunk_finite_and_conserves():
    # one wide kernel drags the chunk's halfK out to many bins, so the
    # narrow-but-still-Gaussian records (sigma just above dt/6) evaluate
    # bins far outside their own support — the rs floor must keep the
    # deposit finite and every record either binned or booked excluded.
    from raytracer.detector import _splat_records
    n_t = 64
    t_lo, dt = 0.0, 1.0
    sigma = np.array([0.2] * 50 + [20.0])        # pre-sorted by sigma
    t = np.concatenate([np.linspace(2.0, 62.0, 50), [32.0]])
    power = np.full(51, 1e-3)
    acc = np.zeros((1, n_t))
    excl = _splat_records(acc, t, sigma, power,
                          np.zeros(51, np.int64), t_lo, dt)
    assert np.all(np.isfinite(acc))
    assert float(acc.sum()) + excl == pytest.approx(51e-3, rel=1e-12)
