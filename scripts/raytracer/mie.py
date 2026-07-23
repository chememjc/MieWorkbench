# =============================================================================
# mie.py — Mie scattering tables built on miepython.
#
# Conventions:
#   * size parameter x = 2 pi r n_medium / lambda_vacuum
#   * relative index m = n_particle / n_medium (complex; Im >= 0 absorbing)
#   * Qext/Qsca/g are dimensionless efficiencies vs the geometric cross
#     section pi r^2
#   * the single-particle phase function p(mu) (mu = cos theta) is
#     normalized NUMERICALLY so that 2 pi Integral p(mu) dmu = 1 — we do
#     not rely on miepython's normalization convention (pinned by test)
#
# Two consumers:
#   * continuum medium (particles.py): ensemble tables — mu_ext(lambda),
#     single-scatter albedo, radius-resolved scatter weighting, and
#     inverse-CDF phase-function sampling
#   * explicit spheres: complex amplitudes S1(mu), S2(mu) for exact
#     polarized scattering off an individual particle. particles.py draws
#     the scattering azimuth phi from the polarized differential cross
#     section |S1|^2|E_perp(phi)|^2 + |S2|^2|E_par(phi)|^2 (no azimuthal
#     flattening); the phi-integral is polarization-independent, so the
#     theta-marginal remains the phase function p(mu) below.
# =============================================================================
import numpy as np
import miepython


class MieEvaluator:
    """Single-particle Mie quantities for one (particle, host) pair,
    evaluated lazily and cached on (radius, lambda) grids."""

    def __init__(self, particle_material, host_material):
        self.mat_p = particle_material
        self.mat_h = host_material
        self._cache = {}

    def m_rel(self, lam):
        n_p = self.mat_p.n_complex(np.atleast_1d(lam))
        n_h = np.real(self.mat_h.n_complex(np.atleast_1d(lam)))
        return n_p / n_h

    def size_param(self, r, lam):
        n_h = np.real(self.mat_h.n_complex(np.atleast_1d(lam)))
        return 2.0 * np.pi * np.asarray(r) * n_h / np.asarray(lam)

    def efficiencies(self, r, lam):
        """(qext, qsca, g) for scalar or matched-shape r, lam [m]."""
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        lam = np.broadcast_to(np.atleast_1d(lam), r.shape).astype(float)
        qext = np.empty_like(r)
        qsca = np.empty_like(r)
        g = np.empty_like(r)
        for i in range(len(r)):
            m = complex(self.m_rel(lam[i])[0])
            x = float(self.size_param(r[i], lam[i])[0])
            # miepython 3.x defines m = n - ik for absorbing spheres;
            # our materials carry n + ik with k >= 0, so conjugate
            qe, qs, _, gg = miepython.efficiencies_mx(np.conj(m), x)
            qext[i], qsca[i], g[i] = qe, qs, gg
        return qext, qsca, g

    def amplitudes(self, r, lam, mu):
        """Complex S1(mu), S2(mu) for ONE particle radius/wavelength.
        mu: (K,) array of cos(theta)."""
        m = complex(self.m_rel(lam)[0])
        x = float(self.size_param(r, lam)[0])
        S1, S2 = miepython.S1_S2(np.conj(m), x, np.asarray(mu))
        return S1, S2

    def phase_function(self, r, lam, n_mu=1024):
        """Numerically normalized p(mu): 2 pi Int p dmu = 1.
        Cached per (r, lam) rounded keys."""
        key = (round(float(r), 12), round(float(lam), 15), n_mu)
        if key in self._cache:
            return self._cache[key]
        mu = np.linspace(-1.0, 1.0, n_mu)
        S1, S2 = self.amplitudes(r, lam, mu)
        p = 0.5 * (np.abs(S1) ** 2 + np.abs(S2) ** 2)
        norm = 2.0 * np.pi * np.trapezoid(p, mu)
        p = p / norm
        cdf = np.concatenate([[0.0], np.cumsum(
            0.5 * (p[1:] + p[:-1]) * np.diff(mu))])
        cdf /= cdf[-1]
        self._cache[key] = (mu, p, cdf)
        return mu, p, cdf

    def sample_scatter_mu(self, r, lam, rng, n):
        """Draw n cos(theta) values from the phase function."""
        mu, p, cdf = self.phase_function(r, lam)
        u = rng.uniform(0.0, 1.0, n)
        return np.interp(u, cdf, mu)


class LogNormalDistribution:
    """Number-weighted log-normal over particle RADIUS.

    median_r: median radius [m]; gsd: geometric standard deviation (>1).
    Hard-truncated to [r_min, r_max] (spec: 1 nm .. 1 mm diameters by
    default at the CLI layer)."""

    def __init__(self, median_r, gsd, r_min=0.5e-9, r_max=0.5e-3):
        if gsd < 1.0:
            raise ValueError("gsd must be >= 1")
        self.mu = np.log(median_r)
        self.sigma = np.log(gsd)
        self.r_min = r_min
        self.r_max = r_max

    def sample(self, rng, n):
        if self.sigma == 0.0:
            return np.full(n, np.exp(self.mu))
        out = np.exp(rng.normal(self.mu, self.sigma, size=int(n * 1.3) + 8))
        out = out[(out >= self.r_min) & (out <= self.r_max)]
        while len(out) < n:
            extra = np.exp(rng.normal(self.mu, self.sigma, size=n))
            extra = extra[(extra >= self.r_min) & (extra <= self.r_max)]
            out = np.concatenate([out, extra])
        return out[:n]

    def mean_volume(self):
        """E[(4/3) pi r^3] of the (untruncated) log-normal:
        E[r^3] = exp(3 mu + 4.5 sigma^2)."""
        return (4.0 / 3.0) * np.pi * np.exp(
            3.0 * self.mu + 4.5 * self.sigma ** 2)

    def quadrature(self, n=48):
        """(radii, weights) Gauss-Legendre in log-r for ensemble
        integrals, truncated to +-4 sigma within [r_min, r_max]."""
        if self.sigma == 0.0:
            return np.array([np.exp(self.mu)]), np.array([1.0])
        lo = max(self.mu - 4 * self.sigma, np.log(self.r_min))
        hi = min(self.mu + 4 * self.sigma, np.log(self.r_max))
        xg, wg = np.polynomial.legendre.leggauss(n)
        lr = 0.5 * (hi + lo) + 0.5 * (hi - lo) * xg
        w = wg * 0.5 * (hi - lo) * (
            np.exp(-0.5 * ((lr - self.mu) / self.sigma) ** 2)
            / (self.sigma * np.sqrt(2 * np.pi)))
        w /= w.sum()
        return np.exp(lr), w


def number_density(phi_mass, rho_particle, rho_host, dist):
    """Number density [1/m^3] from mass fraction phi.

    Volume fraction f_v = (phi/rho_p) / (phi/rho_p + (1-phi)/rho_h);
    N = f_v / E[V_particle]."""
    if not (0.0 < phi_mass < 1.0):
        raise ValueError("phi (mass fraction) must be in (0,1)")
    f_v = (phi_mass / rho_particle) / (
        phi_mass / rho_particle + (1.0 - phi_mass) / rho_host)
    return f_v / dist.mean_volume(), f_v


class EnsembleTables:
    """Ensemble-averaged optics of a log-normal cloud at fixed lambda list.

    mu_ext(lambda) = N * Int pi r^2 Qext(r,lambda) p(r) dr        [1/m]
    albedo(lambda) = Int r^2 Qsca p / Int r^2 Qext p
    scatter radius sampling weight ~ r^2 Qsca(r) p(r)

    STRUCTURE FACTOR S(q) (samples-instruments round). If `sq` is supplied
    the medium is a CORRELATED suspension (inter-particle interference),
    not the default independent-scatterer cloud. `sq` is either
      * a prebuilt callable S(q_per_um) -> array, or
      * a (sq_model, sq_params, context) triple dispatched through
        structure.sq_evaluate (units: q in 1/um).
    Two things change (both energy-exact by construction — particles.py
    _continuum reads only mu_ext + albedo):
      * the ANGULAR sampling density becomes p_Mie(theta)*S(q(theta)) —
        forward scattering is suppressed where S<1 at low q;
      * the SCATTERING coefficient is scaled by the phase-function-weighted
        mean structure factor <S>_p = Int p(theta) S(q) dOmega /
        Int p(theta) dOmega, so mu_sca' = mu_sca*<S>_p, mu_abs unchanged,
        mu_ext = mu_abs + mu_sca'.  The polarized azimuth law is untouched
        (S(q) is theta-only; azimuth stays uniform in the continuum path).

    HONEST LIMIT — polydispersity decoupling: a single S(q) at the
    effective structure scale multiplies the SIZE-AVERAGED phase function
    (the local monodisperse approximation / "decoupling approximation").
    The registry supplies the structure scale through context
    {"phi_v": ensemble volume fraction, "r_mean_um": volume-weighted mean
    radius}; per-size S(q) coupling (the full Kotlarchyk-Chen cross term)
    is NOT modelled.
    """

    def __init__(self, evaluator, dist, n_density, lam_list, n_quad=48,
                 sq=None):
        self.ev = evaluator
        self.dist = dist
        self.N = n_density
        self.lam_list = np.asarray(sorted(set(float(l) for l in lam_list)))
        self.radii, self.weights = dist.quadrature(n_quad)
        self._by_lam = {}
        for lam in self.lam_list:
            qext, qsca, g = evaluator.efficiencies(
                self.radii, np.full_like(self.radii, lam))
            sig_e = np.pi * self.radii ** 2 * qext
            sig_s = np.pi * self.radii ** 2 * qsca
            mu_ext = self.N * np.sum(sig_e * self.weights)
            mu_sca = self.N * np.sum(sig_s * self.weights)
            w_r = sig_s * self.weights
            w_sum = w_r.sum()
            w_r = w_r / w_sum if w_sum > 0 else self.weights
            self._by_lam[float(lam)] = {
                "mu_ext": mu_ext, "mu_sca": mu_sca,
                "albedo": (mu_sca / mu_ext) if mu_ext > 0 else 0.0,
                "radius_weights": w_r, "qext": qext, "qsca": qsca, "g": g,
            }

        # --- optional inter-particle structure factor S(q) ---
        self._sq = None
        self.sq_model = None
        if sq is not None:
            if callable(sq):
                self._sq = sq
                self.sq_model = getattr(sq, "sq_model", "callable")
            else:
                sq_model, sq_params, context = sq

                def _make(_m, _p, _c):
                    from . import structure
                    return lambda qv: structure.sq_evaluate(_m, _p, qv, _c)
                self._sq = _make(sq_model, sq_params, context)
                self.sq_model = sq_model
            # reference lambda for scalar diagnostics = nearest to the mean
            self.sq_ref_lam = float(self.lam_list[int(np.argmin(np.abs(
                self.lam_list - float(np.mean(self.lam_list)))))])
            for lam in self.lam_list:
                base = self._by_lam[float(lam)]
                mu_grid, S_vals, q_um, mean_S, cdf = self._sq_lam(lam, base)
                mu_ext_old = base["mu_ext"]
                mu_sca_old = base["mu_sca"]
                mu_abs = mu_ext_old - mu_sca_old        # absorption unchanged
                mu_sca_new = mu_sca_old * mean_S
                mu_ext_new = mu_abs + mu_sca_new
                base["mu_ext"] = mu_ext_new
                base["mu_sca"] = mu_sca_new
                base["albedo"] = ((mu_sca_new / mu_ext_new)
                                  if mu_ext_new > 0 else 0.0)
                base["sq_mean_S"] = mean_S
                base["sq_qmax"] = float(q_um.max()) if q_um.size else 0.0
                base["sq_mu"] = mu_grid
                base["sq_cdf"] = cdf

    def _sq_lam(self, lam, base):
        """Build the S(q)-corrected angular sampler + <S>_p at one lambda.

        Returns (mu_grid, S_vals, q_per_um, mean_S, cdf).

        Decoupling: the size-AVERAGED phase function P_ens(mu) =
        sum_r radius_weights[r] * p_r(mu) (scattering-weighted, the same
        weights the two-step radius/mu sampler marginalizes to) is what
        S(q) multiplies — one structure factor for the whole size mix.
        """
        w_r = base["radius_weights"]
        mu_grid = None
        P_ens = None
        for i, r in enumerate(self.radii):
            mu_i, p_i, _ = self.ev.phase_function(r, lam)
            if P_ens is None:
                mu_grid = mu_i
                P_ens = np.zeros_like(mu_i)
            P_ens = P_ens + w_r[i] * p_i

        # --- q-UNIT CONVERSION (the honest bit). mie.py is METRES-internal
        # (lam in metres, radii in metres) but structure.py wants q in
        # INVERSE MICROMETRES. So convert lambda to um first:
        #   lambda_um = lam_m * 1e6
        #   q(theta) = 2 * (2 pi n_host / lambda_um) * sin(theta/2)  [1/um]
        # with sin(theta/2) = sqrt((1-mu)/2). n_host = Re(host index) at
        # this lambda (the evaluator's host material). Backscatter (mu=-1)
        # gives q_max = 4 pi n_host / lambda_um exactly — the grid top,
        # pinned by test_sample_sq.py to catch a 1e-6/1e-9 unit slip. ---
        n_host = float(np.real(
            self.ev.mat_h.n_complex(np.atleast_1d(lam))[0]))
        lam_um = float(lam) * 1e6
        q_um = (4.0 * np.pi * n_host / lam_um) * np.sqrt(
            np.clip((1.0 - mu_grid) / 2.0, 0.0, None))
        S_vals = np.asarray(self._sq(q_um), dtype=np.float64)

        # <S>_p = Int p S dOmega / Int p dOmega. Azimuthal symmetry makes
        # dOmega = 2 pi dmu, and the 2 pi cancels in the ratio -> a plain
        # trapezoid ratio over mu (numpy quadrature on the 1024-pt grid).
        den = float(np.trapezoid(P_ens, mu_grid))
        num = float(np.trapezoid(P_ens * S_vals, mu_grid))
        mean_S = (num / den) if den != 0.0 else 1.0

        pdf = np.clip(P_ens * S_vals, 0.0, None)
        cdf = np.concatenate([[0.0], np.cumsum(
            0.5 * (pdf[1:] + pdf[:-1]) * np.diff(mu_grid))])
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        return mu_grid, S_vals, q_um, mean_S, cdf

    def _nearest(self, lam):
        i = int(np.argmin(np.abs(self.lam_list - float(lam))))
        return self._by_lam[float(self.lam_list[i])]

    def mu_ext(self, lam):
        return self._nearest(lam)["mu_ext"]

    def albedo(self, lam):
        return self._nearest(lam)["albedo"]

    def sample_radius(self, lam, rng, n):
        t = self._nearest(lam)
        idx = rng.choice(len(self.radii), size=n, p=t["radius_weights"])
        return self.radii[idx]

    def sample_direction(self, lam, r, d_in, rng):
        """Scatter directions for rays with incoming unit dirs d_in (N,3),
        each off a particle of radius r[i]. Azimuth uniform: the continuum
        child is incoherent with random phase (disordered medium), so its
        polarization azimuth is unbiased by construction. Polarization-
        resolved azimuth sampling lives in the explicit-particle path
        (particles.py ExplicitRealization.intercept)."""
        n = len(d_in)
        mu = np.empty(n)
        if self._sq is not None:
            # S(q)-correlated suspension: draw theta directly from the
            # size-averaged p(theta)*S(q(theta)) CDF (decoupling — the
            # per-ray radius is not used for the direction here).
            t = self._nearest(lam)
            u = rng.uniform(0.0, 1.0, n)
            mu = np.interp(u, t["sq_cdf"], t["sq_mu"])
        else:
            # group by radius for phase-CDF reuse (radii come from the
            # shared quadrature grid, so few uniques)
            for rv in np.unique(r):
                sel = r == rv
                mu[sel] = self.ev.sample_scatter_mu(rv, lam, rng,
                                                    int(sel.sum()))
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
        # build frames around d_in
        a = np.zeros_like(d_in)
        a[np.arange(n), np.argmin(np.abs(d_in), axis=-1)] = 1.0
        t1 = np.cross(d_in, a)
        t1 /= np.linalg.norm(t1, axis=-1, keepdims=True)
        t2 = np.cross(d_in, t1)
        st = np.sqrt(np.clip(1.0 - mu ** 2, 0.0, 1.0))
        return (mu[:, None] * d_in
                + (st * np.cos(phi))[:, None] * t1
                + (st * np.sin(phi))[:, None] * t2)

    def diagnostics(self):
        d = {
            "N_per_m3": self.N,
            "radii_um": (self.radii / 1e-6).tolist(),
            "lambda_nm": (self.lam_list / 1e-9).tolist(),
            "mu_ext_per_m": {("%.1f" % (l / 1e-9)):
                             self._by_lam[float(l)]["mu_ext"]
                             for l in self.lam_list},
            "albedo": {("%.1f" % (l / 1e-9)):
                       self._by_lam[float(l)]["albedo"]
                       for l in self.lam_list},
        }
        if self._sq is not None:
            d["sq_model"] = self.sq_model
            d["sq_mean_S"] = {("%.1f" % (l / 1e-9)):
                              self._by_lam[float(l)]["sq_mean_S"]
                              for l in self.lam_list}
            # <S>_p at the reference lambda (the mu_sca scale factor)
            d["mu_sca_scale"] = self._by_lam[self.sq_ref_lam]["sq_mean_S"]
        return d
