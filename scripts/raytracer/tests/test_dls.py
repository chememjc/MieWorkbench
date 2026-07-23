# =============================================================================
# test_dls.py — DLS correlator validation + tiny end-to-end smoke.
#
# (a) VALIDATION-FIRST: synthesize a complex AR(1)/Ornstein-Uhlenbeck field
#     E_{n+1} = a*E_n + noise (a = exp(-Gamma_true*dt), stationary-variance-
#     matched noise) whose field autocorrelation is exactly exp(-Gamma*tau).
#     Assert the correlator's own cumulant fit recovers Gamma within 3% and
#     the coherence factor beta ~ 1 for a fully coherent complex field.
# (b) g2 intercept ~ 2 and the tail -> 1 (Siegert, beta ~ 1).
# (c) @slow tiny end-to-end: a scenehelpers cuvette with a `sample` medium
#     (explicit spheres) + a coherent source + an off-axis detector, run
#     through run_dls.main + dls_correlate.main; assert the frames.h5
#     shape/keys and that the fitted Gamma is positive and within a factor
#     of 3 of D*q^2.
#
# Run: /home3/optics/env/bin/python -m pytest \
#        scripts/raytracer/tests/test_dls.py -q -m "not slow"   (fast subset)
#      ... (drop -m) once with the slow smoke.
# =============================================================================
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPTS))

import dls_correlate as dc                                # noqa: E402


# ---------------------------------------------------------------------------
# synthetic complex AR(1) / OU field with a known decay rate
# ---------------------------------------------------------------------------
def _ou_field(n, dt, gamma_true, var=1.0, seed=0):
    """E_{n+1} = a*E_n + noise, a = exp(-gamma*dt). Complex noise variance
    chosen so the stationary field variance is `var`: Var(E) =
    sigma_n^2/(1-a^2)."""
    rng = np.random.default_rng(seed)
    a = np.exp(-gamma_true * dt)
    sigma_n = np.sqrt(var * (1.0 - a * a))
    E = np.zeros(n, dtype=np.complex128)
    # start in the stationary distribution
    E[0] = (rng.normal(0, np.sqrt(var / 2))
            + 1j * rng.normal(0, np.sqrt(var / 2)))
    for i in range(1, n):
        w = (rng.normal(0, sigma_n / np.sqrt(2))
             + 1j * rng.normal(0, sigma_n / np.sqrt(2)))
        E[i] = a * E[i - 1] + w
    return E


# ---------------------------------------------------------------------------
# (a) cumulant fit recovers Gamma_true within 3%; beta ~ 1
# ---------------------------------------------------------------------------
def test_cumulant_recovers_gamma():
    dt = 1e-4
    gamma_true = 2000.0
    E = _ou_field(20000, dt, gamma_true, seed=7)
    lags, g1 = dc.field_autocorr(E)
    tau = lags * dt
    gamma_fit, mu2, c0 = dc.cumulant_fit(tau, np.abs(g1))
    assert gamma_fit > 0
    rel = abs(gamma_fit - gamma_true) / gamma_true
    assert rel < 0.03, "Gamma %.1f vs true %.1f (%.1f%%)" % (
        gamma_fit, gamma_true, 100 * rel)


def test_beta_is_unity_for_coherent_field():
    E = _ou_field(20000, 1e-4, 2000.0, seed=11)
    I = np.abs(E) ** 2
    beta = dc.intensity_beta(I)
    assert abs(beta - 1.0) < 0.1, "beta %.3f" % beta


# ---------------------------------------------------------------------------
# (b) g2 intercept ~ 2, tail -> 1
# ---------------------------------------------------------------------------
def test_g2_siegert_intercept_and_tail():
    dt = 1e-4
    E = _ou_field(20000, dt, 2000.0, seed=13)
    lags, g1 = dc.field_autocorr(E)
    beta = dc.intensity_beta(np.abs(E) ** 2)
    g2 = 1.0 + beta * np.abs(g1) ** 2
    assert 1.85 < g2[0] < 2.15, "g2(0) = %.3f" % g2[0]
    # tail (well past 1/Gamma): the last quarter of lags -> 1
    tail = g2[len(g2) * 3 // 4:]
    assert np.mean(tail) < 1.05, "g2 tail mean %.4f" % np.mean(tail)


def test_field_autocorr_multichannel_sums_incoherently():
    # two independent OU channels with the SAME Gamma still recover it
    dt = 1e-4
    E1 = _ou_field(8000, dt, 1500.0, seed=1)
    E2 = _ou_field(8000, dt, 1500.0, seed=2)
    E = np.stack([E1, E2], axis=1)
    lags, g1 = dc.field_autocorr(E)
    gamma_fit, _, _ = dc.cumulant_fit(lags * dt, np.abs(g1))
    assert abs(gamma_fit - 1500.0) / 1500.0 < 0.06


def test_stokes_einstein_roundtrip():
    # D = Gamma/q^2, d_H = kB T/(3 pi eta D) -> recover a known diameter
    T, eta, r = 293.15, 1.0e-3, 250e-9      # 250 nm radius sphere in water
    D_true = dc.KB * T / (6 * np.pi * eta * r)
    q = 2.3e7
    Gamma = D_true * q ** 2
    D, d_H = dc.diffusion_and_diameter(Gamma, q, T, eta)
    assert abs(D - D_true) / D_true < 1e-9
    assert abs(d_H - 2 * r) / (2 * r) < 1e-9


# ---------------------------------------------------------------------------
# (c) tiny end-to-end smoke
# ---------------------------------------------------------------------------
def _make_optprops_root(tmp_path):
    """Copy the shipped library and add a samples.miesamp DLS row."""
    root = tmp_path / "opticalproperties"
    shutil.copytree(REPO / "opticalproperties", root)
    sample_dir = root / "sample"
    sample_dir.mkdir(exist_ok=True)
    # A DILUTE handful of large explicit polystyrene spheres (1.5 um radius
    # = 3 um diameter) in water, loaded to a very small optical depth
    # tau=8e-4 -> ~11 well-separated spheres. Few, widely-spaced scatterers
    # are ESSENTIAL for the traced speckle to evolve smoothly: the explicit
    # medium samples each scatter event with a shared MC RNG stream whose
    # draw ORDER depends on the exact set of ray-particle collisions, so a
    # DENSE cloud desynchronises that stream on nm-scale particle motion and
    # the speckle goes delta-correlated (g1 collapses in one frame). With a
    # sparse cloud the collision set is stable frame-to-frame and the field
    # decorrelates via the physical q.r geometric phase (see the run_dls.py
    # header 'HONEST LIMITS').
    header = ("name,particle_material,dist,median_um,gsd,phi,tau,mode,"
              "sq_model,solvent_visc_pas,reference,notes\n")
    row = ("dls_test,polystyrene,mono,3.0,,,8e-4,explicit,none,"
           "1.0e-3,test,dls smoke\n")
    (sample_dir / "samples.miesamp").write_text(header + row)
    return root


def _make_model(tmp_path):
    """A water cuvette (sample body) with a coherent source and a 90-deg
    side detector."""
    from raytracer.tests import scenehelpers as sh
    # cuvette: 0.4 mm cube of water centered at origin. The ambient is set to
    # water too (index-matched, like a DLS cell immersed in matching fluid)
    # so scattered speckle rays are not TIR-trapped at cuvette walls but fly
    # straight to an off-axis detector.
    hx = 0.0002
    cuvette = sh.slab_body("Cuvette", "water", -hx, hx, half=hx)
    cuvette["sample"] = "dls_test"
    cuvette["bbox_m"] = {"min": [-hx, -hx, -hx], "max": [hx, hx, hx]}
    cuvette["solid_closed"] = True
    src = sh.source_body("Src", x=-0.01, half=0.00008,
                         lambdac_nm=633.0, coherent=True)
    # off-axis forward detector just past the cuvette exit (x = 0.4 mm),
    # normal -x, spanning y in [0.12, 3] mm (above the 0.08 mm-half direct
    # beam, so it sees ONLY forward-scattered speckle) x z in [-3, 3] mm
    xd = 0.0004
    ylo, yhi, zh = 0.00012, 0.003, 0.003
    yc = 0.5 * (ylo + yhi)
    face = sh._rect_face(
        "Det.Pad.Face1", [xd, yc, 0], [-1, 0, 0],
        [[xd, ylo, -zh], [xd, yhi, -zh], [xd, yhi, zh], [xd, ylo, zh]],
        (yhi - ylo) * 2 * zh)
    det = {"name": "Det", "label": "Det", "role": "detector",
           "detector": {"face": face["id"]}, "faces": [face]}
    model = sh.make_model([src, cuvette, det])
    model["ambient_material"] = "water"      # index-matched immersion
    return model


@pytest.mark.slow
def test_dls_end_to_end(tmp_path):
    import h5py
    import common
    import run_dls

    optroot = _make_optprops_root(tmp_path)
    model = _make_model(tmp_path)
    common.validate_model(model)
    geom_dir = tmp_path / "geometry" / "dls"
    geom_dir.mkdir(parents=True)
    model_json = geom_dir / "model.json"
    model_json.write_text(json.dumps(model))
    case_dir = tmp_path / "results" / "dls" / "case"

    run_dls.main([
        "--model-json", str(model_json),
        "--case-dir", str(case_dir),
        "--frames", "30", "--dt-ms", "5.0",
        "--temp-k", "293.15",
        "--rays", "60000", "--nlambda", "1",
        "--resolution", "16",
        "--workers", "2", "--seed", "3",
        "--optical-properties", str(optroot),
    ])

    frames_path = case_dir / "dls" / "frames.h5"
    assert frames_path.exists()
    with h5py.File(frames_path, "r") as h:
        pos = h["positions"][()]
        radii = h["radii"][()]
        n_p = len(radii)
        assert pos.shape == (30, n_p, 3)
        assert 4 <= n_p < 2e5      # explicit, enough for the gather
        dgrp = h["detectors"]
        assert len(dgrp) >= 1
        label0 = list(dgrp)[0]
        g = dgrp[label0]
        fr = g["frames"]
        N, nkeys, npol, H, W = fr.shape
        assert N == 30 and npol == 2 and H >= 8 and W >= 8
        assert nkeys >= 1
        q_mag = float(g.attrs["q_magnitude_per_m"])
        assert q_mag > 0
        keys = json.loads(g.attrs["keys_json"])
        assert len(keys) == nkeys
        eta = float(h.attrs["solvent_visc_pas"])
        r_med = float(np.median(radii))

    # run the offline correlator
    rc = dc.main(["--case-dir", str(case_dir)])
    assert rc == 0
    report = json.loads((case_dir / "dls" / "report.json").read_text())
    dets = report["detectors"]
    assert dets
    # expected order of magnitude: Gamma ~ D * q^2 for the median radius
    D_expected = dc.KB * 293.15 / (6 * np.pi * eta * r_med)
    ok = False
    for label, r in dets.items():
        gamma = r["Gamma_per_s"]
        q = r["q_magnitude_per_m"]
        assert np.isfinite(gamma)
        gamma_expected = D_expected * q ** 2
        if gamma > 0 and 1.0 / 3.0 < gamma / gamma_expected < 3.0:
            ok = True
    assert ok, ("no detector matched Gamma ~ D*q^2 within 3x: "
                + ", ".join("%s Gamma=%.3g (exp %.3g)"
                            % (l, r["Gamma_per_s"],
                               D_expected * r["q_magnitude_per_m"] ** 2)
                            for l, r in dets.items()))
    # g2_*.csv + plots exist
    assert list((case_dir / "dls").glob("g2_*.csv"))
    assert (case_dir / "dls" / "correlogram.png").exists()
    assert (case_dir / "dls" / "gamma_vs_q2.png").exists()
