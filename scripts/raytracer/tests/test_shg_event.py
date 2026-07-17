# =============================================================================
# test_shg_event.py — pulsed-optics Phase P7b: the chi2 SHG bulk transfer
# (tracer.step) + its stratum/detector/engine plumbing.
#
# Covers (the plan's locked gates):
#   * shg_efficiency_vec == the scalar shg_efficiency (incl. the clamp).
#   * eta ∝ I * L^2 at delta_k = 0 (pump at the row's design wavelength).
#   * sinc^2 detuning: traced conversion at a detuned pump matches the
#     closed-form ratio from the scalar-index delta_k.
#   * transfer closure: energy closure stays OK, the harmonic strata
#     carry exactly the shg_converted tally minus their own downstream
#     losses, and total detected power is conserved vs the no-chi2 run.
#   * harmonic children land at lam/2 in the right spectral bin
#     (run_trace path: lam-range extension + harmonic_strata map).
#   * child gopl continuity: harmonic arrival time rides the pump's
#     group delay (spectrogram row within the pump pulse window).
#   * engine routing: nonlinear/kerr/saturable/tpa bodies force Python
#     (the P8 elements previously emitted NO token — regression).
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_shg_event.py -q
# =============================================================================
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import h5py
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "raytracer" / "tests"))

import common                                              # noqa: E402
import run_trace                                            # noqa: E402
import scenehelpers as sh                                  # noqa: E402
from raytracer import nlo, cengine                          # noqa: E402
from raytracer.optprops import load_optical_properties     # noqa: E402

# the shipped BBO type-I row: design pump 800 nm, d_eff 1.95 pm/V
ROW = "bbo_shg_800_type1"
D_EFF = 1.95e-12
LAM_D = 800.0


def _bench(L_mm=5.0, lam_nm=LAM_D, power_mW=1000.0, half_src=0.0005):
    """source -> fused-silica chi2 slab -> detector. Fused silica (with
    the documented crystal-mismatch warning) keeps the indices isotropic
    so the power bookkeeping has no o/e split; the row supplies only
    d_eff + the design wavelength."""
    slab = sh.slab_body("Xtal", "fused_silica", 0.0, L_mm * 1e-3,
                        half=0.01, nonlinear=ROW)
    return [sh.source_body(power_mW=power_mW, lambdac_nm=lam_nm,
                           half=half_src, coherent=False),
            slab,
            sh.detector_body(x=0.03, half=0.02)]


def _trace(bodies, rays=5000, n_lambda=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # crystal-mismatch (documented)
        return sh.trace_scene(sh.make_model(bodies), rays=rays,
                              n_lambda=n_lambda)


def _harmonic_detected(result, grids, n_lambda):
    det = next(iter(grids.values()))
    fund = harm = 0.0
    for (sid, k, p), w in det.detected_incoherent.items():
        if k >= n_lambda:
            harm += w
        else:
            fund += w
    return fund, harm


# --------------------------------------------------------------------------- #
# 1. vectorized efficiency == scalar (incl. clamp)
# --------------------------------------------------------------------------- #
def test_efficiency_vec_matches_scalar():
    rng = np.random.default_rng(7)
    L = rng.uniform(1e-4, 1e-2, 50)
    I = 10 ** rng.uniform(6, 16, 50)          # spans into the clamp
    dk = rng.uniform(-2e4, 2e4, 50)
    eta_v, cl_v = nlo.shg_efficiency_vec(D_EFF, L, I, 1.45, 1.47,
                                         800e-9, dk)
    for i in range(50):
        eta_s, cl_s = nlo.shg_efficiency(D_EFF, L[i], I[i], 1.45, 1.47,
                                         800e-9, dk[i])
        assert eta_v[i] == pytest.approx(eta_s, rel=1e-12)
        assert bool(cl_v[i]) == cl_s
    assert np.any(cl_v) and not np.all(cl_v)


# --------------------------------------------------------------------------- #
# 2. eta ∝ I * L^2 at delta_k = 0
# --------------------------------------------------------------------------- #
def test_eta_scales_with_intensity_and_length_squared():
    def converted(L_mm, power_mW):
        result, grids, _ = _trace(_bench(L_mm=L_mm, power_mW=power_mW))
        return result.shg_converted["Xtal"]

    base = converted(2.0, 200.0)
    assert base > 0
    # double the length: eta x4 (sinc^2 stays 1 at the design pump)
    assert converted(4.0, 200.0) / base == pytest.approx(4.0, rel=1e-2)
    # double the power: eta doubles (∝ I) AND the converted pump power
    # doubles -> transferred power x4; the EFFICIENCY (conv/P) doubles
    assert (converted(2.0, 400.0) / 400.0) / (base / 200.0) \
        == pytest.approx(2.0, rel=1e-2)


# --------------------------------------------------------------------------- #
# 3. sinc^2 detuning sweep (scalar-index delta_k)
# --------------------------------------------------------------------------- #
def test_detuning_follows_sinc2():
    L = 2e-3
    silica = load_optical_properties().matdb.get("fused_silica")

    def dk_of(lam_nm):
        lam = lam_nm * 1e-9
        n1 = float(np.real(silica.n_complex(np.array([lam]))[0]))
        n2 = float(np.real(silica.n_complex(np.array([lam / 2]))[0]))
        n1d = float(np.real(silica.n_complex(np.array([LAM_D * 1e-9]))[0]))
        n2d = float(np.real(silica.n_complex(
            np.array([LAM_D * 1e-9 / 2]))[0]))
        return (4.0 * np.pi * (n1 - n2) / lam
                - 4.0 * np.pi * (n1d - n2d) / (LAM_D * 1e-9))

    def conv(lam_nm):
        result, _, _ = _trace(_bench(L_mm=L * 1e3, lam_nm=lam_nm))
        return result.shg_converted["Xtal"]

    base = conv(LAM_D)
    for lam_nm in (801.0, 803.0):
        # closed-form expected ratio: the full eta formula at fixed I, L
        def eta_form(lam):
            n1 = float(np.real(silica.n_complex(
                np.array([lam * 1e-9]))[0]))
            n2 = float(np.real(silica.n_complex(
                np.array([lam * 1e-9 / 2]))[0]))
            e, _ = nlo.shg_efficiency(D_EFF, L, 1.0, n1, n2, lam * 1e-9,
                                      dk_of(lam))
            return e
        want = eta_form(lam_nm) / eta_form(LAM_D)
        got = conv(lam_nm) / base
        assert got == pytest.approx(want, rel=2e-2), \
            "lam=%g: got %.4g want %.4g" % (lam_nm, got, want)
    # sanity: 3 nm off the design pump through 2 mm is deep in the sinc
    assert conv(803.0) / base < 0.5


# --------------------------------------------------------------------------- #
# 4. transfer closure + harmonic bookkeeping
# --------------------------------------------------------------------------- #
def test_transfer_closure_and_harmonic_strata():
    result, grids, _ = _trace(_bench(L_mm=5.0, power_mW=2000.0),
                              n_lambda=1)
    rep = result.ledger.report(["Src"])
    assert rep["closure_ok"]
    conv = result.shg_converted["Xtal"]
    assert conv > 0
    fund, harm = _harmonic_detected(result, grids, n_lambda=1)
    assert harm > 0
    # the harmonic detected power is the transfer minus its own exit-
    # face Fresnel/backreflection losses: within 15% of the tally and
    # never above it
    assert harm <= conv * (1.0 + 1e-9)
    assert harm == pytest.approx(conv, rel=0.15)
    # total detected power conserved vs the no-chi2 bench (pure
    # transfer): the harmonic's slightly different Fresnel at lam/2 is
    # the only difference
    bodies = _bench(L_mm=5.0, power_mW=2000.0)
    del bodies[1]["nonlinear"]
    r0, g0, _ = _trace(bodies, n_lambda=1)
    det_with = fund + harm
    det_without = sum(next(iter(g0.values())).detected_incoherent.values())
    assert det_with == pytest.approx(det_without, rel=1e-2)


# --------------------------------------------------------------------------- #
# 5+6. run_trace path: spectral bin at lam/2, gopl continuity, routing
# --------------------------------------------------------------------------- #
def _run(tmp_path, bodies, extra, name="case", rays=4000, nlambda=3):
    model = sh.make_model(bodies)
    common.validate_model(model)
    mj = tmp_path / (name + "_model.json")
    common.write_json(mj, model)
    case = tmp_path / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = run_trace.main([
            "--model-json", str(mj), "--case-dir", str(case),
            "--rays", str(rays), "--nlambda", str(nlambda),
            "--resolution", "48", "--workers", "1"] + extra)
    assert rc == 0
    return case


def test_harmonic_lands_at_half_wavelength_and_rides_pump_delay(tmp_path):
    bodies = _bench(L_mm=5.0, power_mW=0.0)
    src = bodies[0]["source"]
    src.update(pulse_energy_uJ=0.05, pulse_duration_ps=0.5,
               rep_rate_hz=8e7)                  # pulsed -> auto products
    case = _run(tmp_path, bodies, ["--time-products", "spectrogram",
                                   "--spectral-bins", "32"], nlambda=3)
    cj = json.loads((case / "case.json").read_text())
    assert cj["engine"] == "python"
    hs = cj["harmonic_strata"]
    assert hs["n_lambda"] == 3 and hs["map"]["0"] == 3
    assert hs["bodies"] == ["Xtal"]
    with h5py.File(case / "detectors" / "Det_Pad_Face1.h5") as h:
        sg = h["time_spectrogram"][...]
        lam_lo = float(h.attrs["lam_lo_m"]) * 1e9
        lam_hi = float(h.attrs["lam_hi_m"]) * 1e9
        t_lo, dt = float(h.attrs["t_lo_s"]), float(h.attrs["time_dt_s"])
    assert lam_lo < 400.0 < lam_hi          # range extended to lam/2
    n_bins = sg.shape[0]
    edges = np.linspace(lam_lo, lam_hi, n_bins + 1)
    lam_c = 0.5 * (edges[:-1] + edges[1:])
    rows = sg.sum(axis=1)
    harm_rows = (lam_c > 390) & (lam_c < 410) & (rows > 0)
    pump_rows = (lam_c > 780) & (lam_c < 820) & (rows > 0)
    assert harm_rows.any(), "no power in the 400 nm spectral rows"
    assert pump_rows.any()
    # gopl continuity: the harmonic's mean arrival sits inside the pump
    # pulse's own arrival window (it inherited the pump group delay)
    tc = t_lo + (np.arange(sg.shape[1]) + 0.5) * dt
    t_harm = float((sg[harm_rows] @ tc).sum() / sg[harm_rows].sum())
    t_pump = float((sg[pump_rows] @ tc).sum() / sg[pump_rows].sum())
    assert abs(t_harm - t_pump) < 2e-12, \
        "harmonic arrival %.3g vs pump %.3g" % (t_harm, t_pump)


def test_nlo_chi2_body_routes_python(tmp_path):
    # chi2 SHG (P7b) still forces Python — the harmonic-child strata are not
    # ported to C (P7 tranche 2 ported saturable/tpa/kerr only; a chi2 body
    # emits the unported "nonlinear" token).
    bodies = [sh.source_body(power_mW=1.0, lambdac_nm=LAM_D, coherent=False),
              sh.slab_body("Xtal", "fused_silica", 0.0, 0.002,
                           half=0.01, nonlinear=ROW),
              sh.detector_body(x=0.03, half=0.02)]
    case = _run(tmp_path, bodies, [], name="route_nonlinear",
                rays=200, nlambda=1)
    cj = json.loads((case / "case.json").read_text())
    assert cj["engine"] == "python"
    if cengine.binary_path() is not None:
        assert "nonlinear" in cj["engine_reason"], \
            "chi2 SHG should force Python routing"


@pytest.mark.skipif(cengine.binary_path() is None,
                    reason="miewb-trace not built")
def test_nlo_bulk_bodies_route_c(tmp_path):
    # P7 tranche 2: saturable / TPA / Kerr are ported — their bodies route to
    # the C engine under auto (previously all NLO forced Python). Kerr on an
    # incoherent source warns+skips but still routes to C (the token is
    # ported; the physics is a coherent-gather phase term).
    for prop, val, token in (("kerr_n2", "@n2_fused_silica", "kerr"),
                             ("saturable", "@sam_1550_16_2ps", "saturable"),
                             ("tpa_beta", 2.0, "tpa")):
        bodies = [sh.source_body(power_mW=1.0, lambdac_nm=LAM_D,
                                 coherent=False),
                  sh.slab_body("Xtal", "fused_silica", 0.0, 0.002,
                               half=0.01, **{prop: val}),
                  sh.detector_body(x=0.03, half=0.02)]
        case = _run(tmp_path, bodies, [], name="route_%s" % token,
                    rays=200, nlambda=1)
        cj = json.loads((case / "case.json").read_text())
        assert cj["engine"] == "c", \
            "%s should route to C (reason: %s)" % (
                prop, cj.get("engine_reason"))
