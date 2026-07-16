"""meent adoption gate for P6 RCWA tables (engine3 §7.5).

(a) Li inverse-rule convergence on a metallic TM lamellar grating
(b) energy conservation for a lossless dielectric lamellar case across angles
(c) reciprocity check.

Run: env/bin/python adoption_gate.py
"""
import numpy as np
import meent


def lamellar_ucell(n_ridge, n_groove, nx=4000, duty=0.5):
    row = np.full(nx, n_groove, dtype=complex)
    row[: int(round(duty * nx))] = n_ridge
    return row.reshape(1, 1, nx)


def solve(pol, theta_deg, N, n_ridge, n_groove, P, depth, wl,
          n_top=1.0, n_bot=1.0, phi_deg=0.0, nx=4000, duty=0.5, ft=0):
    mee = meent.call_mee(
        backend=0, pol=pol, n_top=n_top, n_bot=n_bot,
        theta=np.deg2rad(theta_deg), phi=np.deg2rad(phi_deg),
        fto=(N, 0), wavelength=wl, period=(P, P),
        ucell=lamellar_ucell(n_ridge, n_groove, nx=nx, duty=duty),
        thickness=(depth,), type_complex=np.complex128, fourier_type=ft)
    return mee.conv_solve()


def oaxis(N):
    return np.arange(-N, N + 1)


print("meent 0.12.0  —  factorization (verified from installed source):")
print("  meent/on_numpy/emsolver/fourier_analysis.py cfs2d()/dfs2d() apply the")
print("  Li INVERSE RULE to field components discontinuous across a material")
print("  interface (source comments literally: 'discontinuous ... inverse rule")
print("  is applied'; the epz convolution matrix is inverted via meeinv()).")
print("  This is Li (1996) JOSA A 13:1870 proper Fourier factorization.")
print("  connecting_algo default 'TMM' (transfer matrix); 'SMM' scattering-")
print("  matrix formulation is also selectable.\n")

# ---------------------------------------------------------------------------
# (a) metallic TM lamellar convergence — the case the conventional (Laurent)
# rule converges on only after hundreds of orders; Li's inverse rule (meent)
# converges by ~20. Gold at 633 nm, n = 0.18 + 3.07i (Johnson & Christy 1972).
# Shallow lamellar (depth 50 nm), duty 0.5, period 1.0 um, normal incidence,
# TM (p) polarization. Orders 0, +-1 propagate.
# ---------------------------------------------------------------------------
n_au = 0.18 + 3.07j
P, DEPTH, WL = 1.0, 0.05, 0.633
print("=" * 72)
print("(a) Li inverse-rule convergence — Au lamellar TM, normal incidence")
print("    n_Au=%s  period=%.3g um  depth=%.3g um  wl=%.3g um  duty=0.5"
      % (n_au, P, DEPTH, WL))
sweep = {}
for N in (5, 10, 20, 40, 60, 100):
    r = solve(pol=1, theta_deg=0.0, N=N, n_ridge=n_au, n_groove=1.0,
              P=P, depth=DEPTH, wl=WL)
    o = oaxis(N)
    de = np.real(r.de_ri).ravel()
    sweep[N] = np.array([de[o == 0][0], de[o == -1][0], de[o == 1][0]])
    print("    N=%3d  R0=%.8f  R-1=%.8f  R+1=%.8f  sumR=%.6f"
          % (N, sweep[N][0], sweep[N][1], sweep[N][2], de.sum()))
diff_20_100 = float(np.max(np.abs(sweep[20] - sweep[100])))
diff_60_100 = float(np.max(np.abs(sweep[60] - sweep[100])))
print("    max|N=20 - N=100| over (R0,R-1,R+1) = %.3e   (gate: < 1e-3)"
      % diff_20_100)
print("    max|N=60 - N=100|                   = %.3e" % diff_60_100)
print("    GATE(a): %s" % ("PASS" if diff_20_100 < 1e-3 else "FAIL"))

# ---------------------------------------------------------------------------
# (b) energy conservation — lossless dielectric lamellar (n_ridge=1.5), all
# propagating reflected+transmitted orders must sum to 1.
# ---------------------------------------------------------------------------
print("=" * 72)
print("(b) energy conservation — lossless dielectric lamellar (n_ridge=1.5,"
      " n_groove=1.0)")
Pb, Db, WLb = 1.2, 0.6, 0.55
worst = 0.0
for theta in (0.0, 10.0, 25.0, 40.0):
    for pol in (0, 1):
        r = solve(pol=pol, theta_deg=theta, N=40, n_ridge=1.5, n_groove=1.0,
                  P=Pb, depth=Db, wl=WLb)
        tot = float(np.real(r.de_ri).sum() + np.real(r.de_ti).sum())
        worst = max(worst, abs(tot - 1.0))
        print("    theta=%4.1f  pol=%s  sum(de_ri+de_ti)=%.12f  |err|=%.2e"
              % (theta, "TE" if pol == 0 else "TM", tot, abs(tot - 1.0)))
print("    worst |sum - 1| = %.3e   (gate: < 1e-6)" % worst)
print("    GATE(b): %s" % ("PASS" if worst < 1e-6 else "FAIL"))

# ---------------------------------------------------------------------------
# (c) reciprocity — a lossless transmission grating in a symmetric surround.
# Forward: incidence theta_i, +1 transmitted order exits at theta_1. Reverse
# the ray: incidence theta_1, the -1 order returns to theta_i. de_ti carries
# the obliquity factor, so the two efficiencies must be equal (reciprocity).
# ---------------------------------------------------------------------------
print("=" * 72)
print("(c) reciprocity — lossless dielectric transmission grating")
Pc, Dc, WLc = 1.4, 0.5, 0.55
theta_i = 12.0
rf = solve(pol=0, theta_deg=theta_i, N=40, n_ridge=1.5, n_groove=1.0,
           P=Pc, depth=Dc, wl=WLc)
o = oaxis(40)
ep1_f = float(np.real(rf.de_ti).ravel()[o == 1][0])
theta_1 = np.degrees(np.arcsin(np.sin(np.deg2rad(theta_i)) + WLc / Pc))
rr = solve(pol=0, theta_deg=theta_1, N=40, n_ridge=1.5, n_groove=1.0,
           P=Pc, depth=Dc, wl=WLc)
em1_r = float(np.real(rr.de_ti).ravel()[o == -1][0])
recip = abs(ep1_f - em1_r)
print("    forward:  +1 order @ theta_i=%.2f deg -> eff=%.8f (exits %.3f deg)"
      % (theta_i, ep1_f, theta_1))
print("    reverse:  -1 order @ theta_1=%.3f deg -> eff=%.8f"
      % (theta_1, em1_r))
print("    |eff_fwd - eff_rev| = %.3e   (gate: < 1e-3)" % recip)
print("    GATE(c): %s" % ("PASS" if recip < 1e-3 else "FAIL"))
print("=" * 72)
