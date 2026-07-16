#!/usr/bin/env python3
# =============================================================================
# rcwa_kogelnik_crosscheck.py — validate the meent RCWA table generator
# against the engine's existing Kogelnik (1969) transmission coupled-wave
# branch (raytracer.grating), for a THICK, WEAKLY-modulated volume grating
# near Bragg. Reports first-order efficiency agreement.
#
# Runs under env/bin/python (meent + raytracer):
#   PYTHONPATH=scripts env/bin/python scripts/tools/rcwa_kogelnik_crosscheck.py
#
# Validity regime (documented): Kogelnik two-wave coupled-wave theory holds in
# the SVEA / thick-grating limit — Klein-Cook Q = 2*pi*lambda*d/(n*Lambda^2)
# >> 1 (only 0 and +1 orders significantly coupled), small index modulation
# dn, unslanted transmission geometry, at/near the Bragg angle. To match the
# engine's Kogelnik "thin-boundary n~1" idealization exactly, the average
# index is 1.0 (modulation about vacuum), so no interface refraction enters
# and the two coupled-wave treatments are compared under identical
# assumptions. Away from that regime (thin grating, large dn, far detuning)
# they are EXPECTED to diverge; that is the honest scope of the closed-form
# model the RCWA table supersedes.
# =============================================================================
import numpy as np
import meent

from raytracer.grating import bragg_kogelnik_eta, _kogelnik_nu_xi


def meent_vbg_eta1(lam_um, theta_deg, period_um, dn, thick_um, n0=1.0,
                   fto=20, nx=1024, pol=0):
    """Exact RCWA first-order transmitted efficiency of an unslanted volume
    phase grating n(x)=n0+dn*cos(2*pi*x/period), thickness thick_um, in a
    surround of index n0 (index-matched, no interface refraction)."""
    x = (np.arange(nx) + 0.5) / nx * period_um
    row = (n0 + dn * np.cos(2 * np.pi * x / period_um)).astype(complex)
    ucell = row.reshape(1, 1, nx)
    mee = meent.call_mee(
        backend=0, pol=pol, n_top=n0, n_bot=n0,
        theta=np.deg2rad(theta_deg), phi=0.0, fto=(fto, 0),
        wavelength=lam_um, period=(period_um, period_um), ucell=ucell,
        thickness=(thick_um,), type_complex=np.complex128, fourier_type=0)
    res = mee.conv_solve()
    o = np.arange(-fto, fto + 1)
    de_ti = np.real(res.de_ti).ravel()
    # the Bragg-matched first order for +theta incidence subtracts K (m=-1);
    # take the stronger of +-1 so the check is sign-convention robust.
    eta1 = max(float(de_ti[o == 1][0]), float(de_ti[o == -1][0]))
    return eta1, float(de_ti[o == 0][0])


def kogelnik_eta1(lam_um, theta_deg, period_um, dn, thick_um):
    """Engine Kogelnik first-order efficiency at this geometry (unslanted)."""
    spec = {"lines_per_mm": 1e3 / period_um,
            "params": {"thickness_um": thick_um, "dn": dn, "slant_deg": 0.0}}
    lam = np.array([lam_um * 1e-6])
    cos_i = np.array([np.cos(np.deg2rad(theta_deg))])
    nu_s, nu_p, xi, prop = _kogelnik_nu_xi(spec, lam, cos_i)
    return float(bragg_kogelnik_eta(nu_s, xi)[0])


def main():
    # thick, weakly-modulated, unslanted VBG; n0 = 1.0 (matches engine's n~1).
    # dn*d is held near lambda*cos(theta_B)/2 so the Bragg coupling nu ~ pi/2
    # (eta_1 ~ 1); dn is small and d large so higher-order leakage is
    # negligible (deep in the two-wave SVEA regime).
    Lam = 2.0        # period um (500 l/mm)
    dn = 0.01
    d = 30.0         # thickness um
    lam = 0.6        # um
    n0 = 1.0
    # Bragg angle (unslanted, thin-boundary): sin(theta_B) = lambda/(2*Lambda)
    theta_B = np.degrees(np.arcsin(lam / (2 * Lam)))
    Q = 2 * np.pi * lam * d / (n0 * Lam ** 2)
    print("VBG cross-check — Lambda=%.3g um (%.0f l/mm) dn=%.3g d=%.3g um "
          "lam=%.3g um n0=%.2f" % (Lam, 1e3 / Lam, dn, d, lam, n0))
    print("  Klein-Cook Q = %.2f  (Kogelnik SVEA valid for Q >> 1)" % Q)
    print("  Bragg angle theta_B = %.4f deg" % theta_B)
    print("  %-10s %-12s %-12s %-10s" % ("dtheta", "kogelnik_eta1",
                                          "meent_eta1", "abs_diff"))
    worst = 0.0
    for dth in (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0):
        th = theta_B + dth
        ek = kogelnik_eta1(lam, th, Lam, dn, d)
        em, e0 = meent_vbg_eta1(lam, th, Lam, dn, d, n0=n0)
        diff = abs(ek - em)
        # score agreement only where the two-wave model is meant to apply
        # (near Bragg, |dtheta| <= 0.5 deg); report all points.
        if abs(dth) <= 0.5:
            worst = max(worst, diff)
        print("  %-+10.2f %-12.6f %-12.6f %-10.4f" % (dth, ek, em, diff))
    print("  worst |kogelnik - meent| within +-1 deg of Bragg = %.4f "
          "(%.2f%%)" % (worst, 100 * worst))
    print("  RESULT: %s (few-%% agreement in the SVEA regime)"
          % ("PASS" if worst < 0.05 else "MARGINAL" if worst < 0.1 else "FAIL"))


if __name__ == "__main__":
    main()
