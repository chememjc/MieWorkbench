#!/usr/bin/env python
# =============================================================================
# dls_correlate.py — dynamic light scattering (DLS) correlator.
#
# Interpreter: "$MIEWB_OPTICS_PYTHON"  (numpy/scipy/h5py/matplotlib)
#
# Reads the persisted speckle field sequence written by run_dls.py
# (<case>/dls/frames.h5) and computes, per detector:
#   * g1(tau) — the normalized first-order (field) autocorrelation, via an
#     FFT autocorrelation over the frame axis. The complex speckle field is
#     summed over a central KxK aperture per (source, lambda, pol) gather
#     key + polarization component; those channels are MUTUALLY INCOHERENT
#     populations, so their per-channel field autocorrelations are summed
#     (unbiased per-lag mean) and then normalized to g1(0) = 1.
#   * g2(tau) = 1 + beta * |g1(tau)|^2 — the Siegert-reconstructed intensity
#     correlation, with the coherence factor beta fitted from the tau->0
#     intercept of the MEASURED aperture-intensity fluctuation
#     (beta = <I^2>/<I>^2 - 1).
#   * a second-order cumulant fit  ln|g1| = -Gamma*tau + mu2*tau^2/2  by
#     weighted least squares over the decade where |g1| > 0.1, giving the
#     decay rate Gamma, the translational diffusion coefficient
#     D = Gamma / q^2 (q = |q_vector| from the h5), and the hydrodynamic
#     diameter d_H = kB*T / (3*pi*eta*D) (Stokes-Einstein; eta/T echoed in
#     the h5).
#
# This stage is fully OFFLINE and RE-RUNNABLE: it never traces — it only
# reads frames.h5. Change --aperture-px and re-run to trade coherence
# factor (beta) against SNR without re-simulating.
#
# Outputs (under <case>/dls/):
#   g2_<label>.csv       tau_s, g1, g2   (per detector)
#   correlogram.png      |g1|(tau) for every detector, log-x
#   gamma_vs_q2.png      Gamma vs q^2 scatter + through-origin weighted fit
#                        (slope = D, annotated d_H)  [multi-angle]
#   report.json          per-detector blocks + fitted D, d_H, beta, Gamma
#
# CLI:
#   dls_correlate.py --case-dir <case> [--aperture-px K] [--emit-csv]
#     --aperture-px K   central KxK detector aperture summed into E(t)
#                       (default: the full grid). Smaller K -> fewer speckles
#                       averaged -> higher beta, lower SNR.
#     --emit-csv        (default on for g2_*.csv; kept for symmetry with the
#                       pipeline's other stages — the CSVs are always written)
#
# HONEST LIMITS (shared with run_dls.py): single-scattering DLS only (frame-
# to-frame speckle must stay in the g1 decay regime, tau <~ 0.1 mean free
# path); frozen radii (no polydispersity evolution); no hydrodynamic
# interactions / no structure-factor slow-down of D (dilute limit); no
# sedimentation or flow (pure Brownian, drift-free). Gamma, D and d_H are
# the z-average cumulant estimates, valid for modest polydispersity.
# =============================================================================
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

# CODATA 2018 exact Boltzmann constant [J/K]
KB = 1.380649e-23


# ---------------------------------------------------------------------------
# Correlator math (importable, engine-free — validated by test_dls.py)
# ---------------------------------------------------------------------------
def field_autocorr(E, max_lag=None):
    """First-order (field) autocorrelation of a complex field series.

    E : (N,) or (N, C) complex — N frames, C mutually-incoherent channels
        (e.g. every gather key x polarization component). Channels are
        summed at the CORRELATION level, not the field level (they never
        interfere), which is the physically correct combination for a
        multi-mode / multi-key detector.

    Returns (lags, g1):
      lags : (M+1,) integer frame lags 0..M  (M = N//2 by default)
      g1   : (M+1,) complex, normalized so g1[0] = 1.

    Per channel c the unbiased autocorrelation is
        R_c(m) = (1/(N-m)) * sum_{t=0}^{N-1-m} conj(E_c[t]) E_c[t+m]
    computed by zero-padded FFT; g1(m) = sum_c R_c(m) / sum_c R_c(0).
    """
    E = np.asarray(E)
    if E.ndim == 1:
        E = E[:, None]
    E = E.astype(np.complex128, copy=False)
    N = E.shape[0]
    M = N // 2 if max_lag is None else min(int(max_lag), N - 1)
    # zero-pad to >= 2N so the circular FFT autocorr equals the linear one
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    F = np.fft.fft(E, n=nfft, axis=0)
    ac = np.fft.ifft(F * np.conj(F), axis=0)          # (nfft, C)
    R = ac[:M + 1, :]                                  # lags 0..M
    counts = (N - np.arange(M + 1)).astype(np.float64)  # unbiased overlap
    Rhat = R / counts[:, None]                          # per-channel mean/lag
    numer = Rhat.sum(axis=1)                            # incoherent channel sum
    denom = numer[0].real
    if denom <= 0:
        return np.arange(M + 1), np.zeros(M + 1, dtype=np.complex128)
    return np.arange(M + 1), numer / denom


def intensity_beta(I):
    """Coherence factor beta from the tau->0 intercept of the measured
    aperture-intensity fluctuation: beta = <I^2>/<I>^2 - 1. For a fully
    coherent single-mode complex-Gaussian speckle this is ~1; averaging
    several independent speckles over the aperture drives it toward 0."""
    I = np.asarray(I, dtype=np.float64)
    m1 = float(np.mean(I))
    if m1 <= 0:
        return 0.0
    return float(np.mean(I * I) / (m1 * m1) - 1.0)


def cumulant_fit(tau_s, g1_abs, floor=0.1, min_window=5):
    """Second-order cumulant fit  ln|g1| = c0 - Gamma*tau + 0.5*mu2*tau^2
    by WEIGHTED least squares (weights |g1|^2 — the standard
    heteroscedastic weighting for the log transform), anchored at lag 0.

    Fit window: the CONTIGUOUS initial run of lags (from lag 0) over which
    |g1| stays above `floor` — the classic "fit the first decade of the
    decay" cumulant window — widened to at least `min_window` lags so the
    fit is never degenerate (never returns NaN). The tail, where |g1| has
    fallen into the estimator noise, is deliberately excluded (fitting it
    biases Gamma). Returns (Gamma [1/s], mu2 [1/s^2], c0).
    """
    tau = np.asarray(tau_s, dtype=np.float64)
    g = np.asarray(g1_abs, dtype=np.float64)
    n = len(g)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    # contiguous initial window lag 0..L: extend while |g1| > floor
    L = 1
    while L < n - 1 and np.isfinite(g[L]) and g[L] > floor:
        L += 1
    L = min(max(L, min_window), n - 1)
    idx = np.arange(0, L + 1)
    x = tau[idx]
    gi = np.clip(g[idx], 1e-6, None)
    y = np.log(gi)
    w = gi ** 2
    # design matrix for [c0, Gamma, mu2] with column signs baked in so the
    # returned coefficients are (c0, Gamma, mu2)
    A = np.stack([np.ones_like(x), -x, 0.5 * x ** 2], axis=1)
    WA = A * w[:, None]
    ATA = A.T @ WA
    ATy = WA.T @ y
    try:
        coeffs = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        coeffs, *_ = np.linalg.lstsq(WA, y * w, rcond=None)
    c0, Gamma, mu2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    return Gamma, mu2, c0


def diffusion_and_diameter(Gamma, q_mag, T_k, eta_pas):
    """(D, d_H) from a cumulant decay rate. D = Gamma / q^2 [m^2/s];
    d_H = kB*T / (3*pi*eta*D) [m] (Stokes-Einstein sphere). Returns
    (nan, nan) when q or eta are unusable."""
    if not (q_mag > 0 and eta_pas and eta_pas > 0 and np.isfinite(Gamma)):
        return float("nan"), float("nan")
    D = Gamma / (q_mag ** 2)
    if D <= 0:
        return float(D), float("nan")
    d_H = KB * T_k / (3.0 * np.pi * eta_pas * D)
    return float(D), float(d_H)


# ---------------------------------------------------------------------------
# frames.h5 -> per-detector series
# ---------------------------------------------------------------------------
def _aperture_slice(H, W, K):
    """Central KxK index ranges (row, col). K None or >= grid -> full grid."""
    if K is None or K >= H or K >= W:
        return slice(0, H), slice(0, W)
    y0 = (H - K) // 2
    x0 = (W - K) // 2
    return slice(y0, y0 + K), slice(x0, x0 + K)


def detector_series(frames, H, W, aperture_px=None):
    """frames: (N, nkeys, 2, H, W) complex -> (E_channels, I) where
    E_channels is (N, nkeys*2) complex aperture-summed field per channel and
    I is (N,) the total aperture intensity summed over channels + pixels."""
    ys, xs = _aperture_slice(H, W, aperture_px)
    sub = frames[:, :, :, ys, xs]                       # (N, nkeys, 2, k, k)
    N = sub.shape[0]
    E = sub.sum(axis=(3, 4))                             # (N, nkeys, 2)
    E = E.reshape(N, -1)                                 # (N, nkeys*2)
    # total detected intensity: |E|^2 per pixel, summed over pixels+channels
    I = np.abs(sub).astype(np.float64) ** 2
    I = I.sum(axis=(1, 2, 3, 4))                         # (N,)
    return E, I


def analyze_detector(frames, H, W, dt_s, q_mag, T_k, eta_pas,
                     aperture_px=None):
    """Full per-detector reduction -> dict of arrays + fitted scalars."""
    E, I = detector_series(frames, H, W, aperture_px)
    lags, g1 = field_autocorr(E)
    tau = lags * dt_s
    beta = intensity_beta(I)
    g1_abs = np.abs(g1)
    g2 = 1.0 + beta * g1_abs ** 2
    Gamma, mu2, c0 = cumulant_fit(tau, g1_abs)
    D, d_H = diffusion_and_diameter(Gamma, q_mag, T_k, eta_pas)
    pdi = float(mu2 / Gamma ** 2) if (np.isfinite(mu2) and np.isfinite(Gamma)
                                      and Gamma != 0) else float("nan")
    return {
        "tau_s": tau, "g1": g1_abs, "g2": g2,
        "beta": float(beta), "Gamma_per_s": float(Gamma),
        "mu2_per_s2": float(mu2), "pdi": pdi,
        "D_m2_per_s": float(D), "d_H_m": float(d_H),
        "q_magnitude_per_m": float(q_mag),
        "mean_intensity": float(np.mean(I)),
        "n_frames": int(frames.shape[0]),
    }


# ---------------------------------------------------------------------------
# I/O + rendering
# ---------------------------------------------------------------------------
def _write_g2_csv(path, tau, g1, g2):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tau_s", "g1", "g2"])
        for t, a, b in zip(tau, g1, g2):
            w.writerow(["%.9g" % t, "%.9g" % a, "%.9g" % b])


def _render_correlogram(path, per_det):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, r in sorted(per_det.items()):
        tau = r["tau_s"]
        pos = tau > 0
        ax.semilogx(tau[pos] * 1e3, r["g1"][pos], marker=".", ms=3,
                    lw=1.2, label="%s (d_H=%s)"
                    % (label, _fmt_len(r["d_H_m"])))
    ax.set_xlabel("lag time tau [ms]")
    ax.set_ylabel("|g1(tau)|  (field autocorrelation)")
    ax.set_title("DLS correlogram")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    if per_det:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _render_gamma_vs_q2(path, per_det, T_k, eta_pas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    q2, gam = [], []
    for r in per_det.values():
        q = r["q_magnitude_per_m"]
        if q > 0 and np.isfinite(r["Gamma_per_s"]) and r["Gamma_per_s"] > 0:
            q2.append(q ** 2)
            gam.append(r["Gamma_per_s"])
    fig, ax = plt.subplots(figsize=(8, 5))
    D_fit = float("nan")
    d_H_fit = float("nan")
    if q2:
        q2 = np.asarray(q2)
        gam = np.asarray(gam)
        ax.scatter(q2, gam, s=40, color="#1f77b4", zorder=3,
                   label="per-detector Gamma")
        # through-origin weighted fit: Gamma = D * q^2
        denom = float(np.sum(q2 ** 2))
        if denom > 0:
            D_fit = float(np.sum(q2 * gam) / denom)
            xline = np.linspace(0, float(q2.max()) * 1.05, 50)
            ax.plot(xline, D_fit * xline, "-", color="#d62728", lw=1.5,
                    label="fit  D = %.3g um^2/ms" % (D_fit * 1e12 / 1e3))
            if eta_pas and eta_pas > 0 and D_fit > 0:
                d_H_fit = KB * T_k / (3.0 * np.pi * eta_pas * D_fit)
                ax.annotate("d_H = %s" % _fmt_len(d_H_fit),
                            xy=(0.05, 0.9), xycoords="axes fraction",
                            fontsize=11)
    ax.set_xlabel("q^2  [1/m^2]")
    ax.set_ylabel("Gamma  [1/s]")
    ax.set_title("Multi-angle Gamma vs q^2  (slope = D)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    if q2 is not None and len(q2):
        ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return D_fit, d_H_fit


def _fmt_len(m):
    if not np.isfinite(m):
        return "n/a"
    if m >= 1e-6:
        return "%.3g um" % (m * 1e6)
    return "%.3g nm" % (m * 1e9)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Offline DLS correlator over run_dls.py's frames.h5")
    p.add_argument("--case-dir", required=True,
                   help="results/<model>/<case> directory holding dls/"
                        "frames.h5")
    p.add_argument("--aperture-px", type=int, default=None, metavar="K",
                   help="central KxK detector aperture summed into E(t) "
                        "(default: full grid). Smaller K -> higher beta, "
                        "lower SNR")
    p.add_argument("--emit-csv", action="store_true",
                   help="(g2_<label>.csv are always written; flag kept for "
                        "pipeline symmetry)")
    args = p.parse_args(argv)

    import h5py
    case_dir = Path(args.case_dir)
    dls_dir = case_dir / "dls"
    frames_path = dls_dir / "frames.h5"
    if not frames_path.exists():
        raise SystemExit("dls_correlate.py: no %s — run run_dls.py first"
                         % frames_path)

    per_det = {}
    with h5py.File(frames_path, "r") as h:
        dt_s = float(h["dt_s"][()])
        T_k = float(h["temp_k"][()])
        eta_pas = None
        if "solvent_visc_pas" in h.attrs:
            v = h.attrs["solvent_visc_pas"]
            eta_pas = float(v) if v is not None and float(v) > 0 else None
        dgrp = h["detectors"]
        for label in dgrp:
            g = dgrp[label]
            frames = g["frames"][()]                    # (N,nkeys,2,H,W) c64
            H = int(g.attrs["H"])
            W = int(g.attrs["W"])
            q_mag = float(g.attrs.get("q_magnitude_per_m", 0.0))
            real_label = str(g.attrs.get("label", label))
            r = analyze_detector(frames, H, W, dt_s, q_mag, T_k, eta_pas,
                                 aperture_px=args.aperture_px)
            per_det[real_label] = r

    dls_dir.mkdir(parents=True, exist_ok=True)
    for label, r in per_det.items():
        safe = label.replace(".", "_").replace("/", "_")
        _write_g2_csv(dls_dir / ("g2_%s.csv" % safe),
                      r["tau_s"], r["g1"], r["g2"])

    _render_correlogram(dls_dir / "correlogram.png", per_det)
    D_fit, d_H_fit = _render_gamma_vs_q2(
        dls_dir / "gamma_vs_q2.png", per_det, T_k, eta_pas)

    report = {
        "temp_k": T_k, "dt_s": dt_s,
        "solvent_visc_pas": eta_pas,
        "aperture_px": args.aperture_px,
        "multi_angle_fit": {"D_m2_per_s": D_fit, "d_H_m": d_H_fit},
        "detectors": {
            label: {k: v for k, v in r.items()
                    if k not in ("tau_s", "g1", "g2")}
            for label, r in per_det.items()},
    }
    with open(dls_dir / "report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("[dls_correlate] %d detector(s); multi-angle D=%.3g m^2/s, "
          "d_H=%s -> %s"
          % (len(per_det), D_fit, _fmt_len(d_H_fit), dls_dir), flush=True)
    for label, r in sorted(per_det.items()):
        print("  %s: Gamma=%.4g /s  beta=%.3f  D=%.4g m^2/s  d_H=%s"
              % (label, r["Gamma_per_s"], r["beta"], r["D_m2_per_s"],
                 _fmt_len(r["d_H_m"])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
