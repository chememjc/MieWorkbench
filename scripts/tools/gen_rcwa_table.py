#!/usr/bin/env python3
# =============================================================================
# gen_rcwa_table.py — generate a v2 RCWA grating table (.mietab) + registry
# row from a structural grating spec, via meent (Li inverse-rule RCWA).
#
# Runs under the GUI venv: env/bin/python (meent + numpy + scipy).
#
# Precompute-and-interpolate is what Zemax/Lumerical do (engine3 §7.5): a full
# RCWA solve per ray is O(N^3) and impossible; instead we tabulate COMPLEX
# per-order amplitudes on a (lambda, theta, phi) grid and interpolate the
# re/im components at trace time (raytracer.grating._v2_amplitudes).
#
# Amplitude normalization (matches the consumer's |amp|^2 = efficiency, arg =
# phase contract): amp = sqrt(de_co) * exp(i*arg(coeff)) where de_co is the
# CO-POLARIZED order efficiency (de_ti_s / de_ri_s for s, de_ti_p / de_ri_p
# for p) and coeff is the corresponding complex Rayleigh amplitude (T_s/T_p or
# R_s/R_p). Cross-polarized power at conical mount (phi != 0) is booked as loss
# by the diagonal engine model — documented in RAYTRACER.md.
#
# ADAPTIVE REFINEMENT: Rayleigh/Wood anomalies (a diffracted order passing off,
# |n_top*sin(theta) + m*lambda/period| = n_side) are the known weak spot of
# interpolated RCWA tables. Their loci are analytic; this tool densifies the
# theta and lambda axes locally around every onset inside the sampled range.
#
# Example (the shipped 600 l/mm binary fused-silica transmission grating):
#   env/bin/python scripts/tools/gen_rcwa_table.py \
#       --kind lamellar --period-um 1.6667 --duty 0.5 --depth-um 0.5 \
#       --n-ridge 1.46 --n-groove 1.0 --side transmission \
#       --lam-min-nm 450 --lam-max-nm 750 --n-lam 7 \
#       --theta-max-deg 50 --n-theta 6 --phi-max-deg 20 --n-phi 3 \
#       --orders -1..1 --name rcwa_fs_600_v2 \
#       --out opticalproperties/grating/tables/rcwa_fs_600_v2.mietab
# =============================================================================
import argparse
import sys
from pathlib import Path

import numpy as np


def build_ucell_lamellar(n_ridge, n_groove, duty, nx):
    """1-layer binary lamellar profile as a (1,1,nx) index raster (ridge over
    the first `duty` fraction of the period)."""
    row = np.full(nx, n_groove, dtype=complex)
    row[: int(round(duty * nx))] = n_ridge
    return row.reshape(1, 1, nx)


def anomaly_loci_theta(lam_um, period_um, n_top, sides, orders_lo, orders_hi,
                       theta_min, theta_max):
    """Onset incidence angles (deg) at wavelength lam_um where some order
    m passes off on some side n_side: |n_top sin th + m lam/period| = n_side.
    Returns those within [theta_min, theta_max]."""
    out = []
    lo = lam_um / period_um
    for m in range(orders_lo, orders_hi + 1):
        for n_side in sides:
            for s in (+1.0, -1.0):
                val = (s * n_side - m * lo) / n_top
                if -1.0 <= val <= 1.0:
                    th = np.degrees(np.arcsin(val))
                    if theta_min <= th <= theta_max:
                        out.append(th)
    return out


def anomaly_loci_lambda(theta_deg, period_um, n_top, sides, orders_lo,
                        orders_hi, lam_min, lam_max):
    """Onset wavelengths (um) at incidence theta_deg where some order passes
    off: n_top sin th + m lam/period = ± n_side  ->  lam = (±n_side -
    n_top sin th)*period/m. Returns those within [lam_min, lam_max]."""
    out = []
    st = np.sin(np.deg2rad(theta_deg))
    for m in range(orders_lo, orders_hi + 1):
        if m == 0:
            continue
        for n_side in sides:
            for s in (+1.0, -1.0):
                lam = (s * n_side - n_top * st) * period_um / m
                if lam_min <= lam <= lam_max:
                    out.append(lam)
    return out


def refine_axis(base, loci, span, n_cluster=3, frac=0.006):
    """Return base axis plus, around each locus, a small symmetric cluster of
    nodes so the sampling is visibly denser at the analytic anomaly. `span` is
    the full axis range; `frac` sets the cluster half-width (fraction of span).
    """
    extra = []
    hw = frac * span
    for c in loci:
        for k in range(1, n_cluster + 1):
            d = hw * k / n_cluster
            extra.extend([c - d, c + d, c])
    lo, hi = base[0], base[-1]
    allv = [v for v in list(base) + extra if lo - 1e-12 <= v <= hi + 1e-12]
    # dedupe to a tolerance so near-coincident nodes don't explode the grid
    allv = np.array(sorted(allv))
    keep = [allv[0]]
    tol = 1e-4 * span
    for v in allv[1:]:
        if v - keep[-1] > tol:
            keep.append(v)
    return np.array(keep)


def solve_amps(ucell, thickness_um, n_top, n_bot, lam_um, theta_deg, phi_deg,
               period_um, fto, side, orders):
    """One meent solve -> {m: (amp_s, amp_p)} complex, |amp|^2 = co-pol eff."""
    import meent
    mee = meent.call_mee(
        backend=0, pol=0, n_top=n_top, n_bot=n_bot,
        theta=np.deg2rad(theta_deg), phi=np.deg2rad(phi_deg),
        fto=(fto, 0), wavelength=lam_um, period=(period_um, period_um),
        ucell=ucell, thickness=(thickness_um,), type_complex=np.complex128,
        fourier_type=0)
    res = mee.conv_solve()
    te, tm = res.res_te_inc, res.res_tm_inc          # TE / TM incidence
    o = np.arange(-fto, fto + 1)
    if side == "transmission":
        de_s = np.real(te.de_ti_s).ravel(); c_s = np.asarray(te.T_s).ravel()
        de_p = np.real(tm.de_ti_p).ravel(); c_p = np.asarray(tm.T_p).ravel()
    else:
        de_s = np.real(te.de_ri_s).ravel(); c_s = np.asarray(te.R_s).ravel()
        de_p = np.real(tm.de_ri_p).ravel(); c_p = np.asarray(tm.R_p).ravel()
    out = {}
    for m in orders:
        i = int(np.where(o == m)[0][0])
        es = max(float(de_s[i]), 0.0)
        ep = max(float(de_p[i]), 0.0)
        amp_s = np.sqrt(es) * np.exp(1j * np.angle(c_s[i]))
        amp_p = np.sqrt(ep) * np.exp(1j * np.angle(c_p[i]))
        out[m] = (amp_s, amp_p)
    return out


def parse_orders(text):
    lo, hi = text.split("..")
    return int(lo), int(hi)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", default="lamellar", choices=["lamellar"])
    ap.add_argument("--period-um", type=float, required=True)
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--depth-um", type=float, required=True)
    ap.add_argument("--n-ridge", type=float, required=True)
    ap.add_argument("--n-groove", type=float, default=1.0)
    ap.add_argument("--n-top", type=float, default=1.0)
    ap.add_argument("--n-bot", type=float, default=1.0)
    ap.add_argument("--side", choices=["transmission", "reflection"],
                    default="transmission")
    ap.add_argument("--lam-min-nm", type=float, required=True)
    ap.add_argument("--lam-max-nm", type=float, required=True)
    ap.add_argument("--n-lam", type=int, default=7)
    ap.add_argument("--theta-max-deg", type=float, default=50.0)
    ap.add_argument("--theta-min-deg", type=float, default=0.0)
    ap.add_argument("--n-theta", type=int, default=6)
    ap.add_argument("--phi-max-deg", type=float, default=20.0)
    ap.add_argument("--n-phi", type=int, default=3)
    ap.add_argument("--orders", type=parse_orders, default=(-1, 1))
    ap.add_argument("--fto", type=int, default=40)
    ap.add_argument("--nx", type=int, default=2000)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--refine-cluster", type=int, default=2,
                    help="nodes per side clustered around each anomaly locus")
    ap.add_argument("--refine-frac", type=float, default=0.006,
                    help="cluster half-width as a fraction of the axis span")
    ap.add_argument("--out", required=True, help="output .mietab path")
    ap.add_argument("--name", required=True, help="registry entry name")
    ap.add_argument("--reference", default=None,
                    help="citation string for the registry row")
    ap.add_argument("--registry", default=None,
                    help="gratings.miegrat to append the row to (optional)")
    args = ap.parse_args(argv)

    lo, hi = args.orders
    orders = list(range(lo, hi + 1))
    period_um = args.period_um
    lines_per_mm = 1000.0 / period_um
    lam_min_um = args.lam_min_nm * 1e-3
    lam_max_um = args.lam_max_nm * 1e-3
    sides = sorted({args.n_top, args.n_bot})

    base_lam = np.linspace(lam_min_um, lam_max_um, args.n_lam)
    base_th = np.linspace(args.theta_min_deg, args.theta_max_deg, args.n_theta)

    if args.no_refine:
        lam_ax, th_ax = base_lam, base_th
        n_ref_th = n_ref_lam = 0
    else:
        # theta loci: union over the wavelength nodes; lambda loci: union over
        # the base theta nodes. Densify each axis around its analytic onsets.
        th_loci = []
        for lam in base_lam:
            th_loci += anomaly_loci_theta(lam, period_um, args.n_top, sides,
                                          lo, hi, args.theta_min_deg,
                                          args.theta_max_deg)
        lam_loci = []
        for th in base_th:
            lam_loci += anomaly_loci_lambda(th, period_um, args.n_top, sides,
                                            lo, hi, lam_min_um, lam_max_um)
        th_ax = refine_axis(base_th, th_loci,
                            args.theta_max_deg - args.theta_min_deg,
                            n_cluster=args.refine_cluster,
                            frac=args.refine_frac)
        lam_ax = refine_axis(base_lam, lam_loci, lam_max_um - lam_min_um,
                             n_cluster=args.refine_cluster,
                             frac=args.refine_frac)
        n_ref_th = th_ax.size - base_th.size
        n_ref_lam = lam_ax.size - base_lam.size

    ph_ax = (np.array([0.0]) if args.n_phi <= 1
             else np.linspace(0.0, args.phi_max_deg, args.n_phi))

    ucell = build_ucell_lamellar(args.n_ridge, args.n_groove, args.duty,
                                 args.nx)

    n_solve = lam_ax.size * th_ax.size * ph_ax.size
    sys.stderr.write(
        "gen_rcwa_table: %s grating, %.4g l/mm, side=%s\n"
        "  grid lam=%d (base %d +%d refined)  theta=%d (base %d +%d)  phi=%d "
        "-> %d solves\n"
        % (args.kind, lines_per_mm, args.side, lam_ax.size, args.n_lam,
           n_ref_lam, th_ax.size, args.n_theta, n_ref_th, ph_ax.size,
           n_solve))

    rows = []
    done = 0
    for lam in lam_ax:
        for th in th_ax:
            for ph in ph_ax:
                amps = solve_amps(ucell, args.depth_um, args.n_top, args.n_bot,
                                  float(lam), float(th), float(ph), period_um,
                                  args.fto, args.side, orders)
                for m in orders:
                    a_s, a_p = amps[m]
                    rows.append((lam * 1e3, th, ph, m,
                                 a_s.real, a_s.imag, a_p.real, a_p.imag))
                done += 1
        sys.stderr.write("  %d/%d solves\r" % (done, n_solve))
        sys.stderr.flush()
    sys.stderr.write("\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("# mietab grating v2 side=%s\n" % args.side)
        fh.write("wavelength_nm,theta_deg,phi_deg,order,"
                 "amp_s_re,amp_s_im,amp_p_re,amp_p_im\n")
        for (lnm, th, ph, m, sr, si, pr, pi) in rows:
            fh.write("%.6g,%.6g,%.6g,%d,%.8g,%.8g,%.8g,%.8g\n"
                     % (lnm, th, ph, m, sr, si, pr, pi))
    sys.stderr.write("wrote %s (%d rows)\n" % (out_path, len(rows)))

    ref = args.reference or (
        "%s binary %s grating, RCWA (meent 0.12.0, Li inverse-rule) — "
        "period=%.4g um duty=%.2f depth=%.4g um n_ridge=%.4g n_groove=%.4g; "
        "v2 complex-amplitude table on a (lambda,theta,phi) grid with adaptive "
        "Wood-anomaly refinement (gen_rcwa_table.py)"
        % (args.kind, args.side, period_um, args.duty, args.depth_um,
           args.n_ridge, args.n_groove))
    row = '%s,table,%.6g,,%s,"%s"' % (args.name, lines_per_mm,
                                      out_path.name, ref)
    sys.stderr.write("\nregistry row (append to gratings.miegrat):\n")
    print(row)

    if args.registry:
        reg = Path(args.registry)
        existing = reg.read_text() if reg.exists() else ""
        if (args.name + ",") in existing:
            sys.stderr.write("registry already has %r; not appending\n"
                             % args.name)
        else:
            with open(reg, "a") as fh:
                if existing and not existing.endswith("\n"):
                    fh.write("\n")
                fh.write(row + "\n")
            sys.stderr.write("appended row to %s\n" % reg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
