# =============================================================================
# roughness.py — surface-roughness scattering models. Pure functions only:
# no tracer/scene imports. The tracer applies sqrt(specular_power_factor)
# to the specular amplitude and routes the scattered remainder through
# beckmann_sample microfacet directions.
#
# Conventions match fresnel.py: n_hat is a unit normal, cos_i = -d.n_hat.
# =============================================================================
import numpy as np


def specular_power_factor(sigma_m, cos_i, lam):
    """Davies / Bennett-Porteus total-integrated-scatter specular retention.

    A = exp(-(4*pi*sigma*cos_i/lam)^2)

    sigma_m : RMS surface height [m], scalar or (N,)
    cos_i   : (N,) cosine of incidence angle (w.r.t. n_hat, >= 0)
    lam     : (N,) vacuum wavelength [m]

    Returns the POWER retained in the specular direction (N,); the tracer
    applies sqrt(A) to the specular amplitude (Es, Ep) so that |E|^2 scales
    by A. The scattered remainder (1-A) is redistributed via beckmann_sample.
    """
    sigma_m = np.asarray(sigma_m, dtype=np.float64)
    cos_i = np.asarray(cos_i, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    x = 4.0 * np.pi * sigma_m * cos_i / lam
    return np.exp(-(x ** 2))


def slope_from_sigma_lcorr(sigma_m, lcorr_m):
    """RMS microfacet slope from RMS height sigma and correlation length
    lcorr, assuming a Gaussian surface-height autocorrelation function
    C(r) = sigma^2 exp(-r^2/lcorr^2). For that convention the RMS slope of
    the surface is m_rms = sqrt(2)*sigma/lcorr (standard Beckmann-model
    result, e.g. Ogilvy "Theory of Wave Scattering from Random Rough
    Surfaces").

    sigma_m, lcorr_m : [m], scalar or array (broadcastable)
    """
    sigma_m = np.asarray(sigma_m, dtype=np.float64)
    lcorr_m = np.asarray(lcorr_m, dtype=np.float64)
    return np.sqrt(2.0) * sigma_m / lcorr_m


# ---------------------------------------------------------------------------
# Ground-glass diffusers: the DEEP-ROUGH limit of this same microfacet
# model (sigma >> lambda so the specular retention A is exactly 0 in
# float; every ray refracts/reflects through one Beckmann-sampled
# microfacet with full per-polarization Fresnel + Jones rotation, so
# depolarization emerges from the ensemble of rotated bases). A diffuser
# is therefore parameterized by its RMS microfacet SLOPE alone; grit
# numbers map to slopes via an approximate calibration against published
# ground-glass diffusing angles (Thorlabs DG series, transmitted FWHM at
# 633 nm through N-BK7, n ~ 1.515): small-angle refraction through a
# tilted facet deflects by ~(n-1)*tilt, so FWHM ~ 2.355*(n-1)*m_rms.
# ---------------------------------------------------------------------------
DIFFUSER_SIGMA_NM = 20000.0     # 20 um RMS height: A = exp(-(4pi*sigma/lam)^2)
                                # underflows to 0.0 for any optical lambda

GRIT_FWHM_DEG = {               # approximate catalog diffusing angles
    120: 12.0,
    220: 9.0,
    600: 5.0,
    1500: 2.5,
}


def slope_for_grit(grit, n_design=1.515):
    """RMS microfacet slope for a catalog grit number: FWHM(grit) from
    the calibration table (log-log interpolated between entries, clamped
    at the ends) inverted through FWHM = 2*sqrt(2*ln2)*(n-1)*m_rms."""
    grit = float(grit)
    gs = np.array(sorted(GRIT_FWHM_DEG), dtype=np.float64)
    fw = np.array([GRIT_FWHM_DEG[int(g)] for g in gs])
    lg = np.log(np.clip(grit, gs[0], gs[-1]))
    fwhm_deg = float(np.exp(np.interp(lg, np.log(gs), np.log(fw))))
    fwhm_rad = np.deg2rad(fwhm_deg)
    return fwhm_rad / (2.0 * np.sqrt(2.0 * np.log(2.0))
                       * (n_design - 1.0))


def diffuser_equivalent(slope_rms):
    """(sigma_nm, lcorr_um) whose slope_from_sigma_lcorr equals slope_rms
    with sigma deep in the rough regime -- the roughness-map entry that
    makes the standard scattering path behave as a pure diffuser."""
    if not 0.0 < slope_rms < 1.0:
        raise ValueError("diffuser RMS slope must be in (0, 1), got %r"
                         % slope_rms)
    sigma_m = DIFFUSER_SIGMA_NM * 1e-9
    lcorr_m = np.sqrt(2.0) * sigma_m / slope_rms
    return DIFFUSER_SIGMA_NM, lcorr_m * 1e6


def _tangent_frame(n_hat):
    """Deterministic orthonormal in-plane frame (t1, t2) per row of n_hat.

    Same pattern as fresnel.pol_basis's degenerate-case fallback / the
    surfaces.py _plane_frame helper: pick the global axis most orthogonal
    to n (smallest |component|), cross to build an orthonormal pair.
    """
    n = np.asarray(n_hat, dtype=np.float64)
    ax = np.zeros_like(n)
    idx = np.argmin(np.abs(n), axis=-1)
    ax[np.arange(len(n)), idx] = 1.0
    t1 = np.cross(ax, n)
    t1 /= np.linalg.norm(t1, axis=-1, keepdims=True)
    t2 = np.cross(n, t1)
    return t1, t2


def beckmann_sample(n_hat, sigma_slope, rng, k):
    """Sample k microfacet normals per input normal from the Beckmann
    slope distribution with RMS slope sigma_slope.

    p(theta) ~ (2 tan(theta) / sigma_slope^2) * exp(-tan^2(theta)/sigma_slope^2)
             * (1/cos^3(theta))          [in solid angle]

    which is sampled via the standard inversion tan(theta) =
    sqrt(-sigma_slope^2 * ln(U)), U ~ Uniform(0,1) (equivalently
    tan^2(theta)/sigma_slope^2 ~ Exponential(1), so E[tan^2] = sigma_slope^2
    exactly). phi is uniform on [0, 2*pi). Samples tilting more than 89 deg
    from n_hat are rejected and resampled (the microfacet model breaks down
    at grazing tilts; this is a negligible-probability tail for physically
    reasonable slopes).

    n_hat       : (N,3) unit normals
    sigma_slope : RMS microfacet slope (scalar)
    rng         : numpy Generator (e.g. np.random.default_rng(seed))
    k           : number of microfacet samples per input normal

    Returns (N,k,3) unit microfacet normals.
    """
    n_hat = np.asarray(n_hat, dtype=np.float64)
    N = n_hat.shape[0]
    t1, t2 = _tangent_frame(n_hat)

    tan2_max = np.tan(np.deg2rad(89.0)) ** 2
    tan2 = np.zeros((N, k), dtype=np.float64)
    remaining = np.ones((N, k), dtype=bool)
    while np.any(remaining):
        n_remain = int(np.count_nonzero(remaining))
        u = rng.random(n_remain)
        u = np.clip(u, 1e-300, 1.0)          # guard log(0)
        cand = -(sigma_slope ** 2) * np.log(u)
        rows, cols = np.where(remaining)
        tan2[rows, cols] = cand
        ok = cand <= tan2_max
        remaining[rows[ok], cols[ok]] = False

    theta = np.arctan(np.sqrt(tan2))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(N, k))
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    m = (cos_t[..., None] * n_hat[:, None, :]
         + (sin_t * cos_p)[..., None] * t1[:, None, :]
         + (sin_t * sin_p)[..., None] * t2[:, None, :])
    m /= np.linalg.norm(m, axis=-1, keepdims=True)
    return m
