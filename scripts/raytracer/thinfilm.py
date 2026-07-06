# =============================================================================
# thinfilm.py — multilayer thin-film coatings via the characteristic-matrix
# (transfer-matrix) method, Macleod formulation with tilted admittances.
#
# For a stack of layers j = 1..m (ordered from the INCIDENT side toward the
# substrate), each with complex index n_j and thickness d_j, at vacuum
# wavelength lambda and incidence angle theta_0 in the incident medium n_0:
#
#   Snell invariant:  beta = n_0 sin(theta_0)          (real: n_0 lossless)
#   n_j cos(theta_j) = sqrt(n_j^2 - beta^2)            branch Im >= 0
#   delta_j = 2 pi d_j (n_j cos theta_j) / lambda
#   eta_j(s) = n_j cos(theta_j);   eta_j(p) = n_j^2 / (n_j cos theta_j)
#   M_j = [[cos d_j,  i sin d_j / eta_j],
#          [i eta_j sin d_j,  cos d_j]]
#   [B, C]^T = (M_1 M_2 ... M_m) [1, eta_sub]^T
#   r = (eta_0 B - C) / (eta_0 B + C)
#   t = 2 eta_0 / (eta_0 B + C)
#   R = |r|^2 ;  T = 4 eta_0 Re(eta_sub) / |eta_0 B + C|^2
#
# Zero layers degenerates exactly to the bare Fresnel interface (tested).
# All functions are vectorized over rays (lambda and cos_i vary per ray).
# =============================================================================
import numpy as np


def _n_cos(n, beta):
    """n_j cos(theta_j) = sqrt(n_j^2 - beta^2), branch Im >= 0."""
    val = np.sqrt(np.asarray(n, dtype=np.complex128) ** 2 - beta ** 2 + 0j)
    return np.where(np.imag(val) < 0.0, -val, val)


def tmm_coeffs(lam, cos_i, n_in, n_sub, layer_n, layer_d):
    """Amplitude coefficients (rs, rp, ts, tp) for a coated interface.

    lam     : (N,) vacuum wavelength [m]
    cos_i   : (N,) cosine of incidence angle in the incident medium (>= 0)
    n_in    : (N,) incident-medium index (real / weakly absorbing)
    n_sub   : (N,) complex substrate index
    layer_n : list of (N,) complex arrays, incident side first
    layer_d : list of float or (N,) thicknesses [m], same order

    Returns complex (rs, rp, ts, tp). Power: R=|r|^2,
    T = Re(eta_sub)/eta_0 * |t|^2 per polarization (with the matching
    tilted admittances), provided n_in is real.
    """
    lam = np.asarray(lam, dtype=np.float64)
    ci = np.asarray(cos_i, dtype=np.float64)
    n0 = np.real(np.asarray(n_in, dtype=np.complex128))
    beta = n0 * np.sqrt(np.maximum(1.0 - ci ** 2, 0.0))

    eta0_s = (n0 * ci).astype(np.complex128)
    eta0_p = (n0 ** 2 / np.maximum(n0 * ci, 1e-300)).astype(np.complex128)

    nc_sub = _n_cos(n_sub, beta)
    ns = np.asarray(n_sub, dtype=np.complex128)
    eta_sub_s = nc_sub
    eta_sub_p = ns ** 2 / nc_sub

    out = []
    for eta0, eta_sub, pol in ((eta0_s, eta_sub_s, "s"),
                               (eta0_p, eta_sub_p, "p")):
        # accumulate [B, C] = M_total . [1, eta_sub]
        B = np.ones_like(eta_sub)
        C = eta_sub.copy()
        # multiply from the substrate side outward:
        # [B,C] <- M_j [B,C] applied j = m..1 gives the same product
        # M_1..M_m [1, eta_sub] when iterated in reverse layer order.
        for nj, dj in zip(reversed(layer_n), reversed(layer_d)):
            ncj = _n_cos(nj, beta)
            if pol == "s":
                eta_j = ncj
            else:
                eta_j = np.asarray(nj, dtype=np.complex128) ** 2 / ncj
            delta = 2.0 * np.pi * np.asarray(dj) * ncj / lam
            cd = np.cos(delta)
            sd = np.sin(delta)
            B2 = cd * B + 1j * sd / eta_j * C
            C2 = 1j * eta_j * sd * B + cd * C
            B, C = B2, C2
        denom = eta0 * B + C
        r = (eta0 * B - C) / denom
        t = 2.0 * eta0 / denom
        out.append((r, t, eta0, eta_sub))

    (rs, ts, e0s, ems), (rp, tp, e0p, emp) = out
    # Sign convention: the Macleod admittance rp is opposite to the
    # Born & Wolf / fresnel.py convention (which this tracer uses for its
    # (s_hat, p_hat = d x s_hat) basis transport). Flip so a zero-layer
    # stack reproduces fresnel.fresnel_coeffs exactly, phases included.
    rp = -rp
    return rs, rp, ts, tp, (e0s, ems, e0p, emp)


def tmm_power(rs, rp, ts, tp, etas):
    """Power (Rs, Rp, Ts, Tp) from tmm_coeffs outputs. Lossless stacks
    satisfy R + T = 1 to machine precision (tested)."""
    e0s, ems, e0p, emp = etas
    Rs = np.abs(rs) ** 2
    Rp = np.abs(rp) ** 2
    Ts = np.real(ems) / np.real(e0s) * np.abs(ts) ** 2
    Tp = np.real(emp) / np.real(e0p) * np.abs(tp) ** 2
    return Rs, Rp, Ts, Tp


def resolve_coating_layers(coating_layers, matdb, lam):
    """Materialize a coating spec into (layer_n, layer_d) arrays at lam.

    coating_layers: list of (material_name, thickness) where thickness is
    either a float [m] or the tuple ('qw', lam0_m) meaning quarter-wave at
    lam0: d = lam0 / (4 * Re(n(lam0))).
    """
    lam = np.asarray(lam, dtype=np.float64)
    layer_n, layer_d = [], []
    for mat_name, thick in coating_layers:
        mat = matdb.get(mat_name)
        n_lam = mat.n_complex(lam)
        if isinstance(thick, tuple) and thick[0] == "qw":
            lam0 = float(thick[1])
            n0 = float(np.real(mat.n_complex(np.array([lam0]))[0]))
            d = lam0 / (4.0 * n0)
        else:
            d = float(thick)
        layer_n.append(n_lam)
        layer_d.append(d)
    return layer_n, layer_d
