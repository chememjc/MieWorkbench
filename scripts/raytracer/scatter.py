# =============================================================================
# scatter.py — measured-scatter (ABg / Harvey-Shack) BSDF sampling for
# polished optical surfaces. Numpy-only, float64, vectorized over ray
# batches. Pure functions: no tracer/scene imports.
#
# Model (v1, REFLECTED side only — BRDF; BTDF is out of scope): the
# scattered light forms a lobe about the specular direction whose bidirectional
# reflectance distribution function follows the empirical ABg form
#
#     BSDF(beta) = A / (B + |beta - beta0|^g)
#
# where beta is the direction-cosine PROJECTION of a scattered ray onto the
# surface plane (beta = sin(theta_s) * azimuth_hat, |beta| in [0, 1]) and
# beta0 is the specular direction cosine (|beta0| = sin(theta_i)). The lobe
# variable u = |beta - beta0| is the radial offset from specular in
# direction-cosine space. A, B fix the amplitude and the width of the
# specular core; g (typically ~2 for polished surfaces) fixes the roll-off
# slope. This is the standard scatter model used in stray-light packages
# (FRED/ASAP "ABg"); see Pfisterer, "Approximated Scatter Models for Stray
# Light Analysis," Optics & Photonics News (2011), and J. E. Harvey's
# Harvey-Shack surface-scatter formulation (Opt. Eng. 51, 013402 (2012)).
#
# Energy: the KEY identity is that the projected-solid-angle measure equals
# the direction-cosine area element,  cos(theta_s) dOmega = d^2 beta, so the
# total integrated scatter (the fraction of incident power that leaves the
# specular direction) is a flat area integral of the BSDF over the accessible
# beta-disk:
#
#     TIS = INT_disk  A / (B + |beta - beta0|^g)  d^2 beta
#         = INT_0^umax  A / (B + u^g) * 2*pi*u du          (radial, about beta0)
#
# We integrate radially about beta0 out to umax = 1 - |beta0|, the largest
# offset guaranteed to stay inside the unit beta-disk for every azimuth. By
# the triangle inequality |beta| <= |beta0| + u <= 1, so EVERY sampled ray is
# above the horizon by construction and no energy is lost to clipping. For a
# low-scatter surface (small B) the integrand is concentrated near u = 0, so
# this radial truncation is numerically irrelevant; it only trims the far
# crescent of the lobe at oblique incidence. TIS is used by the tracer to
# split the reflected Fresnel share into a specular remainder (1 - TIS) and a
# scattered lobe population carrying TIS.
#
# Inverse-CDF decision: the radial CDF of A/(B+u^g)*2*pi*u is CLOSED FORM for
# g == 2 (F(u) = ln(1 + u^2/B) / ln(1 + umax^2/B), inverted analytically), so
# the shipped g == 2 entries sample with no tabulation. For g != 2 we build a
# per-call numeric inverse CDF (trapezoid cumulant on a normalized radial
# grid, row-wise inverted) — O(n * Ngrid); documented and exercised by the
# tests, but not used by any shipped registry entry.
# =============================================================================
import numpy as np

from .roughness import _tangent_frame

_NUMERIC_NGRID = 1025          # radial grid for the g != 2 numeric inverse CDF
_TIS_NGRID = 4001             # (denser) grid for the g != 2 TIS quadrature


def _radial_tis_umax(A, B, g, umax):
    """INT_0^umax A/(B+u^g) * 2*pi*u du, vectorized over umax (>= 0).

    Closed form for g == 2; trapezoid quadrature otherwise."""
    umax = np.asarray(umax, dtype=np.float64)
    out = np.zeros(umax.shape, dtype=np.float64)
    pos = umax > 0.0
    if not np.any(pos):
        return out
    um = umax[pos]
    if abs(g - 2.0) < 1e-12:
        out[pos] = np.pi * A * np.log1p(um * um / B)
    else:
        # grid clustered near u=0 (s**2 warp): the integrand
        # A/(B+u^g)*2*pi*u is sharply peaked near the origin for small B, so
        # a uniform grid under-resolves it — the warp restores ~1e-6 accuracy
        s = np.linspace(0.0, 1.0, _TIS_NGRID) ** 2
        u = um[:, None] * s[None, :]                 # (npos, Ngrid)
        integ = (A / (B + u ** g)) * (2.0 * np.pi * u)
        out[pos] = np.trapezoid(integ, x=u, axis=1)
    return out


def abg_tis(A, B, g, cos_i):
    """Total integrated scatter (fraction of the incident/reflected power
    that leaves the specular direction) for an ABg surface at incidence
    cosine cos_i. Scalar or array cos_i; returns the matching shape.

    Integrated radially about the specular direction out to
    umax = 1 - sin(theta_i) in direction-cosine space (see module header)."""
    cos_i = np.asarray(cos_i, dtype=np.float64)
    scalar = cos_i.ndim == 0
    ci = np.atleast_1d(cos_i)
    beta0 = np.sqrt(np.clip(1.0 - ci * ci, 0.0, 1.0))
    umax = np.clip(1.0 - beta0, 0.0, 1.0)
    tis = _radial_tis_umax(A, B, g, umax)
    return float(tis[0]) if scalar else tis


def _sample_radius(rng, A, B, g, umax):
    """Draw u = |beta - beta0| in [0, umax] from p(u) ~ A/(B+u^g)*2*pi*u,
    vectorized over the per-ray umax (>= 0). Rows with umax == 0 return 0."""
    umax = np.asarray(umax, dtype=np.float64)
    n = umax.shape[0]
    u = np.zeros(n, dtype=np.float64)
    pos = umax > 0.0
    if not np.any(pos):
        return u
    um = umax[pos]
    r = rng.random(int(np.count_nonzero(pos)))
    if abs(g - 2.0) < 1e-12:
        # F(u) = ln(1 + u^2/B) / ln(1 + umax^2/B); invert analytically
        u[pos] = np.sqrt(B * (np.power(1.0 + um * um / B, r) - 1.0))
    else:
        # per-ray numeric inverse CDF on a normalized radial grid
        s = np.linspace(0.0, 1.0, _NUMERIC_NGRID)
        ug = um[:, None] * s[None, :]                # (npos, Ngrid)
        dens = ug / (B + ug ** g)                     # p(u) up to constants
        cum = np.concatenate(
            [np.zeros((ug.shape[0], 1)),
             np.cumsum(0.5 * (dens[:, 1:] + dens[:, :-1])
                       * np.diff(ug, axis=1), axis=1)], axis=1)
        cum /= cum[:, -1:]                            # -> [0, 1] per row
        # row-wise inverse: locate r in each row's CDF, linear-interp the grid
        idx = np.clip((cum < r[:, None]).sum(axis=1) - 1, 0,
                      _NUMERIC_NGRID - 2)
        rows = np.arange(ug.shape[0])
        c0 = cum[rows, idx]
        c1 = cum[rows, idx + 1]
        frac = np.where(c1 > c0, (r - c0) / (c1 - c0), 0.0)
        u[pos] = ug[rows, idx] + frac * (ug[rows, idx + 1] - ug[rows, idx])
    return u


def sample_abg(rng, n, A, B, g, d_spec, n_hat):
    """Sample n unit scatter directions from the ABg reflected lobe.

    rng     : numpy Generator
    n       : number of directions (== len(d_spec))
    A,B,g   : ABg parameters (BSDF = A/(B+u^g), u = |beta - beta0|)
    d_spec  : (n,3) unit SPECULAR reflected directions (on the +n_hat side)
    n_hat   : (n,3) unit surface normals oriented toward the incident /
              reflected hemisphere (d_spec . n_hat = cos(theta_i) >= 0)

    Returns (n,3) unit directions. The lobe is radially symmetric about the
    specular direction in direction-cosine space; azimuth is uniform. Every
    returned ray is above the horizon (n_hat . dir >= 0) by construction
    (umax = 1 - |beta0| guarantees |beta| <= 1), so the split is energy-exact.
    """
    d_spec = np.ascontiguousarray(d_spec, dtype=np.float64).reshape(n, 3)
    n_hat = np.ascontiguousarray(n_hat, dtype=np.float64).reshape(n, 3)
    # specular tangential (direction-cosine) vector beta0 and its magnitude
    dn = np.sum(d_spec * n_hat, axis=-1, keepdims=True)
    t_spec = d_spec - dn * n_hat
    beta0 = np.linalg.norm(t_spec, axis=-1)
    umax = np.clip(1.0 - beta0, 0.0, 1.0)

    u = _sample_radius(rng, A, B, g, umax)
    psi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    t1, t2 = _tangent_frame(n_hat)
    offset = ((u * np.cos(psi))[:, None] * t1
              + (u * np.sin(psi))[:, None] * t2)
    beta_vec = t_spec + offset
    beta2 = np.sum(beta_vec * beta_vec, axis=-1)
    # fold any float-edge overshoot back onto the horizon (keeps w real)
    over = beta2 > 1.0
    if np.any(over):
        beta_vec[over] *= (np.sqrt(1.0 - 1e-15) / np.sqrt(beta2[over]))[:, None]
        beta2 = np.clip(beta2, 0.0, 1.0 - 1e-15)
    w = np.sqrt(np.clip(1.0 - beta2, 0.0, 1.0))
    d = beta_vec + w[:, None] * n_hat
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d
