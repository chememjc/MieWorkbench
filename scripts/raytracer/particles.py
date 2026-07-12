# =============================================================================
# particles.py — log-normal particle clouds, hybrid explicit/continuum.
#
# CONTINUUM mode (count > threshold): the box is a participating medium.
# Deterministic-splitting estimator per box traversal of length Delta:
#   * the parent ray continues UNSCATTERED with amplitude
#     * exp(-mu_ext Delta / 2)  — the ballistic/coherent component decays
#     exactly as Beer-Lambert exp(-mu_ext Delta) in power, staying coherent
#   * one scattered child carries P (1 - e^{-mu Delta}) * albedo from a
#     scatter point sampled ~ mu e^{-mu s} along the segment, direction
#     from the ensemble Mie phase function; it is flagged INCOHERENT
#     (random-phase medium) so it never enters the coherent gather;
#     multiple scattering happens naturally as the child re-traverses.
#   * P (1 - e^{-mu Delta}) (1 - albedo) -> 'particle_absorbed'
#
# EXPLICIT mode (count <= threshold): a frozen random realization of
# non-overlapping spheres. Rays collide when passing within the EXTINCTION
# radius r*sqrt(Qext) (the extinction paradox makes sigma_ext up to 2x
# geometric). On collision the Jones vector is scattered with the complex
# Mie amplitudes S1/S2 in the scattering plane — phase is deterministic,
# so a fixed realization produces physical speckle; --seeds N re-draws the
# realization. Intersection is brute-force chunked with AABB pre-cull;
# counts > ~5e4 get a cost warning (grid/numba traversal: future.md).
# =============================================================================
import numpy as np

from .rays import RayBatch
from . import fresnel as fr
from .mie import (MieEvaluator, LogNormalDistribution, number_density,
                  EnsembleTables)


def _slab_overlap(pos, direction, t_max, lo, hi):
    """Per-ray [t0, t1] overlap of segment [0, t_max] with an AABB."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / direction
    t_lo = (lo[None, :] - pos) * inv
    t_hi = (hi[None, :] - pos) * inv
    tmin = np.minimum(t_lo, t_hi)
    tmax = np.maximum(t_lo, t_hi)
    # axes with zero direction: inside-slab check
    par = np.abs(direction) < 1e-300
    inside = (pos >= lo[None, :]) & (pos <= hi[None, :])
    tmin = np.where(par, np.where(inside, -np.inf, np.inf), tmin)
    tmax = np.where(par, np.where(inside, np.inf, -np.inf), tmax)
    t0 = np.maximum(tmin.max(axis=-1), 0.0)
    t1 = np.minimum(tmax.min(axis=-1), t_max)
    return t0, t1


def _sample_phi_polarized(pa, pb, pc, rng, ngrid=256):
    """Sample azimuth phi in [0, 2 pi) from the per-ray polarized law
        pdf(phi) proportional to  pa + pb cos(2 phi) + pc sin(2 phi)
    via a shared-grid inverse CDF (vectorized over rays).

    pa, pb, pc : (m,) real arrays.  Returns (m,) phi in [0, 2 pi).
    The pdf is non-negative analytically (it is a sum of squared field
    projections weighted by |S1|^2, |S2|^2); float noise is clipped.
    """
    m = len(pa)
    edges = np.linspace(0.0, 2.0 * np.pi, ngrid + 1)          # (ngrid+1,)
    c2 = np.cos(2.0 * edges)
    s2 = np.sin(2.0 * edges)
    pdf = pa[:, None] + pb[:, None] * c2[None, :] \
        + pc[:, None] * s2[None, :]                           # (m, ngrid+1)
    pdf = np.clip(pdf, 0.0, None)
    dphi = edges[1] - edges[0]
    seg = 0.5 * (pdf[:, 1:] + pdf[:, :-1]) * dphi             # (m, ngrid)
    cdf = np.concatenate([np.zeros((m, 1)), np.cumsum(seg, axis=1)], axis=1)
    tot = cdf[:, -1:].copy()
    # degenerate rows (zero incident power / all-zero pdf): fall back to
    # a uniform CDF so the draw stays well-defined.
    bad = tot[:, 0] <= 0.0
    if np.any(bad):
        cdf[bad] = np.linspace(0.0, 1.0, ngrid + 1)[None, :]
        tot[bad, 0] = 1.0
    cdf = cdf / tot
    u = rng.uniform(0.0, 1.0, m)
    # per-row bin k with cdf[k] <= u < cdf[k+1], then linear interp in-bin
    k = np.sum(cdf <= u[:, None], axis=1) - 1
    k = np.clip(k, 0, ngrid - 1)
    c_lo = np.take_along_axis(cdf, k[:, None], axis=1)[:, 0]
    c_hi = np.take_along_axis(cdf, (k + 1)[:, None], axis=1)[:, 0]
    frac = np.where(c_hi > c_lo, (u - c_lo) / np.maximum(c_hi - c_lo, 1e-300),
                    0.0)
    return edges[k] + frac * dphi


def resolve_tau_phi(tau, box_length_m, evaluator, dist, rho_p, rho_h,
                    lam_list, n_quad=48):
    """Solve the particle mass fraction phi that gives a target ballistic
    (Beer-Lambert) optical depth `tau` along a box of length
    `box_length_m`, evaluated at the mean of `lam_list`.

    EnsembleTables defines mu_ext(lambda) = N * Int pi r^2 Qext(r) p(r) dr
    with N = f_v / dist.mean_volume() (see `number_density`) — i.e.
    mu_ext is EXACTLY linear in the volume fraction f_v (the quadrature
    integral K = Int pi r^2 Qext p dr doesn't depend on any assumed
    phi/N at all). That makes
        mu_ext_per_funit = K / mean_volume
    a single per-ensemble constant, so f_v_target = tau / (box_length_m *
    mu_ext_per_funit) is exact, not a small-phi linearization. What IS
    nonlinear is the mass-fraction -> volume-fraction map
        f_v = (phi/rho_p) / (phi/rho_p + (1-phi)/rho_h)
    (number_density's formula); inverting it in closed form then gives
    phi exactly for the target f_v, with no iterative root-find needed.
    """
    if box_length_m <= 0:
        raise ValueError("particles tau resolution needs a box length > 0 "
                         "(got %.3g m)" % box_length_m)
    lam_ref = float(np.mean(np.asarray(list(lam_list), dtype=float)))
    radii, weights = dist.quadrature(n_quad)
    qext, _, _ = evaluator.efficiencies(radii, np.full_like(radii, lam_ref))
    K = float(np.sum(np.pi * radii ** 2 * qext * weights))
    mean_vol = dist.mean_volume()
    mu_ext_per_funit = K / mean_vol if mean_vol > 0 else 0.0
    if mu_ext_per_funit <= 0:
        raise ValueError("particles tau resolution: degenerate ensemble "
                         "(zero extinction) at %.1f nm" % (lam_ref * 1e9))
    f_v = tau / (box_length_m * mu_ext_per_funit)
    if not (0.0 < f_v < 1.0):
        raise ValueError(
            "particles tau=%.4g is not achievable in a %.4g mm box (would "
            "need volume fraction %.3g, must be in (0,1)) — shrink the "
            "box, raise the particle density, or lower tau"
            % (tau, box_length_m * 1e3, f_v))
    a, b = 1.0 / rho_p, 1.0 / rho_h
    phi = (f_v * b) / (a * (1.0 - f_v) + f_v * b)
    return phi, {"tau_target": tau, "mu_ext_target_per_m": tau / box_length_m,
                "volume_fraction": f_v, "lambda_ref_nm": lam_ref * 1e9,
                "box_length_mm": box_length_m * 1e3}


class ParticleCloud:
    """Facade: builds the right mode from a parsed --particles spec."""

    def __init__(self, spec, scene, threshold=1e6, seed=0,
                 lam_list=(633e-9,), pol_scatter=True):
        # local copy: a tau= spec gets its resolved phi filled in below,
        # without mutating the caller's dict.
        self.spec = spec = dict(spec)
        # explicit-mode azimuth: True samples phi from the polarized Mie
        # differential cross-section; False = legacy uniform azimuth. The
        # lead wires this to the --pol-scatter CLI flag.
        self.pol_scatter = pol_scatter
        self.lo = np.asarray(spec["box_corner_m"])
        self.hi = self.lo + np.asarray(spec["box_size_m"])
        self.box_volume = float(np.prod(spec["box_size_m"]))
        mat_p = scene.matdb.get(spec["material"])
        self.dist = LogNormalDistribution(
            median_r=spec["median_um"] * 1e-6 / 2.0,   # median DIAMETER um
            gsd=spec["gsd"])
        rho_p = mat_p.density
        rho_h = scene.ambient.density if scene.ambient.density > 0 else 1.204
        self.evaluator = MieEvaluator(mat_p, scene.ambient)
        self.tau_resolved = None
        if spec.get("phi") is None:
            tau = spec.get("tau")
            if tau is None:
                raise ValueError(
                    "particles spec has neither 'phi' nor 'tau' — nothing "
                    "to solve a number density from")
            # along-beam box length is the FIRST box dimension (the
            # default/documented box convention: dx is the through-beam
            # thickness, dy/dz are the cross-section).
            box_length_m = float(spec["box_size_m"][0])
            phi, info = resolve_tau_phi(tau, box_length_m, self.evaluator,
                                        self.dist, rho_p, rho_h, lam_list)
            spec["phi"] = phi
            self.tau_resolved = info
        self.N, self.f_v = number_density(spec["phi"], rho_p, rho_h,
                                          self.dist)
        self.count = self.N * self.box_volume
        self.tables = EnsembleTables(self.evaluator, self.dist, self.N,
                                     lam_list)
        self.rng = np.random.default_rng(seed)
        self.mode = "explicit" if self.count <= threshold else "continuum"
        self.explicit = None
        if self.mode == "explicit":
            n_exp = int(round(self.count))
            if n_exp < 1:
                raise ValueError(
                    "particle cloud is empty: phi=%.3g of %s (density %g "
                    "kg/m3) with median %.3g um in a %.3g mm^3 box gives "
                    "%.3g particles. Remember phi is a MASS fraction vs "
                    "the ambient medium — dense particles in air need "
                    "surprisingly large phi."
                    % (spec["phi"], spec["material"], rho_p,
                       spec["median_um"], self.box_volume * 1e9,
                       self.count))
            self.explicit = ExplicitRealization(
                self, scene, n_exp, self.rng)

    def diagnostics(self):
        d = self.tables.diagnostics()
        d.update({
            "mode": self.mode,
            "count": self.count,
            "volume_fraction": self.f_v,
            "box_lo_mm": (self.lo / 1e-3).tolist(),
            "box_hi_mm": (self.hi / 1e-3).tolist(),
            "median_diameter_um": self.spec["median_um"],
            "gsd": self.spec["gsd"],
            "phi": self.spec["phi"],
        })
        if self.tau_resolved is not None:
            d["tau_resolved"] = dict(self.tau_resolved, resolved_phi=
                                     self.spec["phi"])
        return d

    # ------------------------------------------------------------------
    # tracer hook
    # ------------------------------------------------------------------
    def intercept(self, tracer, batch, t, fid):
        """Called inside Tracer.step before surface interactions.
        Returns (t, fid, batch, children_or_None)."""
        if self.mode == "continuum":
            return self._continuum(tracer, batch, t, fid)
        return self.explicit.intercept(tracer, batch, t, fid)

    def _continuum(self, tracer, batch, t, fid):
        seg_max = np.where(fid >= 0, t, 1.0)     # escapers still traverse
        t0, t1 = _slab_overlap(batch.pos, batch.dir, seg_max,
                               self.lo, self.hi)
        cross = t1 > t0
        if not np.any(cross):
            return t, fid, batch, None
        idx = np.where(cross)[0]
        mu = np.array([self.tables.mu_ext(l) for l in batch.lam[idx]])
        delta = t1[idx] - t0[idx]
        tau = mu * delta
        p_col = 1.0 - np.exp(-tau)
        p_in = batch.power[idx]

        # scattered children (one per crossing ray with meaningful power)
        alb = np.array([self.tables.albedo(l) for l in batch.lam[idx]])
        p_scat = p_in * p_col * alb
        p_abs = p_in * p_col * (1.0 - alb)
        if np.any(p_abs > 0):
            tracer.ledger.credit("particle_absorbed",
                                 batch.source_id[idx], p_abs,
                                 where="particles")
        make = p_scat > 0
        children = None
        if np.any(make):
            src_rows = idx[make]
            m = len(src_rows)
            child = batch.select(src_rows)
            # scatter point ~ truncated exponential on [t0, t1]
            u = self.rng.uniform(0.0, 1.0, m)
            tm = tau[make]
            s = -np.log(1.0 - u * (1.0 - np.exp(-tm))) / (mu[make])
            s_abs = t0[src_rows] + s
            child.pos = batch.pos[src_rows] + s_abs[:, None] \
                * batch.dir[src_rows]
            r = self.tables.sample_radius(child.lam[0], self.rng, m)
            child.dir = self.tables.sample_direction(
                child.lam[0], r, batch.dir[src_rows], self.rng)
            # rebuild s_hat perpendicular to the new direction
            s_new, _ = fr.pol_basis(child.dir, np.roll(child.dir, 1,
                                                       axis=-1))
            child.s_hat = s_new
            # incoherent, power-weighted, random phase
            amp = np.sqrt(p_scat[make] / 2.0)
            ph = self.rng.uniform(0, 2 * np.pi, m)
            child.Es = amp * np.exp(1j * ph)
            child.Ep = amp * np.exp(1j * ph)
            child.coherent[:] = False
            children = child

        # ballistic parent: coherent Beer-Lambert amplitude decay
        att = np.exp(-tau / 2.0)
        batch.Es[idx] *= att
        batch.Ep[idx] *= att
        return t, fid, batch, children


class ExplicitRealization:
    """Frozen non-overlapping sphere realization + brute-force collision."""

    MAX_BRUTE = 200_000

    def __init__(self, cloud, scene, count, rng):
        import warnings
        self.cloud = cloud
        if count > self.MAX_BRUTE:
            raise ValueError(
                "explicit particle count %d exceeds the brute-force cap "
                "%d — raise --particle-threshold to force continuum mode "
                "or reduce phi/box size" % (count, self.MAX_BRUTE))
        if count > 50_000:
            warnings.warn("explicit mode with %d spheres will be slow "
                          "(brute-force traversal)" % count)
        self.radii = cloud.dist.sample(rng, count)
        self.centers = self._place(scene, count, rng)
        # per-particle extinction collision radius at the mean lambda
        lam0 = float(np.mean(cloud.tables.lam_list))
        qext, qsca, _ = cloud.evaluator.efficiencies(
            self.radii, np.full_like(self.radii, lam0))
        self.r_col = self.radii * np.sqrt(np.maximum(qext, 1e-12))
        self.albedo_p = np.where(qext > 0, qsca / np.maximum(qext, 1e-12),
                                 0.0)

    def _place(self, scene, count, rng):
        """Dart-throwing with a hash grid; rejects overlaps and points
        inside optic solids (exact for sphere bodies, bbox otherwise)."""
        lo, hi = self.cloud.lo, self.cloud.hi
        cell = max(2.0 * self.radii.max(), 1e-6)
        dims = np.maximum(((hi - lo) / cell).astype(int), 1)
        grid = {}
        centers = np.empty((count, 3))
        # optic-body rejection geometry
        sph = []
        for b in scene.bodies:
            if b.role != "optic":
                continue
            for fidx in b.face_ids:
                s = scene.faces[fidx].surface
                if s.__class__.__name__ == "Sphere":
                    sph.append((s.c, s.r))
        placed = 0
        attempts = 0
        while placed < count and attempts < count * 200:
            attempts += 1
            p = rng.uniform(lo, hi)
            r = self.radii[placed]
            if any(np.linalg.norm(p - c) < R + r for c, R in sph):
                continue
            key = tuple(((p - lo) / cell).astype(int))
            ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in grid.get((key[0] + dx, key[1] + dy,
                                           key[2] + dz), ()):
                            if np.linalg.norm(p - centers[j]) \
                                    < r + self.radii[j]:
                                ok = False
                                break
            if not ok:
                continue
            centers[placed] = p
            grid.setdefault(key, []).append(placed)
            placed += 1
        if placed < count:
            raise RuntimeError(
                "particle placement failed: %d/%d after %d attempts — "
                "volume fraction too high for non-overlapping spheres?"
                % (placed, count, attempts))
        return centers

    def intercept(self, tracer, batch, t, fid):
        seg_max = np.where(fid >= 0, t, 1.0)
        t0, t1 = _slab_overlap(batch.pos, batch.dir, seg_max,
                               self.cloud.lo, self.cloud.hi)
        cross = np.where(t1 > t0)[0]
        if len(cross) == 0:
            return t, fid, batch, None
        # nearest particle collision per crossing ray (chunked brute force)
        hit_t = np.full(len(cross), np.inf)
        hit_j = np.full(len(cross), -1, dtype=np.int64)
        C, R = self.centers, self.r_col
        for lo_i in range(0, len(cross), 4096):
            rows = cross[lo_i:lo_i + 4096]
            o = batch.pos[rows]
            d = batch.dir[rows]
            oc = o[:, None, :] - C[None, :, :]         # (q, M, 3)
            b = np.einsum("qmk,qk->qm", oc, d)
            c = np.einsum("qmk,qmk->qm", oc, oc) - R[None, :] ** 2
            disc = b * b - c
            ok = disc >= 0
            sq = np.sqrt(np.where(ok, disc, 0.0))
            tc = -b - sq
            tc = np.where(ok & (tc > 1e-9), tc, np.inf)
            jbest = np.argmin(tc, axis=1)
            tbest = tc[np.arange(len(rows)), jbest]
            sl = slice(lo_i, lo_i + len(rows))
            take = tbest < hit_t[sl]
            hit_t[sl] = np.where(take, tbest, hit_t[sl])
            hit_j[sl] = np.where(take, jbest, hit_j[sl])
        in_seg = (hit_t >= t0[cross]) & (hit_t <= t1[cross])
        coll_rows = cross[in_seg]
        if len(coll_rows) == 0:
            return t, fid, batch, None
        jj = hit_j[in_seg]
        tt = hit_t[in_seg]

        # collided rays scatter at the particle; they no longer reach the
        # surface this step — retarget them
        child = batch.select(coll_rows)
        child.pos = batch.pos[coll_rows] + tt[:, None] * batch.dir[coll_rows]
        child.opl += 0.0   # opl handled by tracer for the segment; the
        # tracer computes opl on ITS segment — since we terminate the
        # parent here, add the collided sub-segment explicitly:
        # (ambient index ~ air)
        n_amb = np.real(self.cloud.evaluator.mat_h.n_complex(child.lam))
        child.opl = batch.opl[coll_rows] + n_amb * tt

        d_in = batch.dir[coll_rows]
        m = len(coll_rows)
        mu_s = np.empty(m)
        S1v = np.empty(m, dtype=np.complex128)
        S2v = np.empty(m, dtype=np.complex128)
        for j in np.unique(jj):
            sel = jj == j
            r_p = self.radii[j]
            lam0 = child.lam[sel][0]
            mu_sel = self.cloud.evaluator.sample_scatter_mu(
                r_p, lam0, self.cloud.rng, int(sel.sum()))
            S1, S2 = self.cloud.evaluator.amplitudes(r_p, lam0, mu_sel)
            mu_s[sel] = mu_sel
            S1v[sel] = S1
            S2v[sel] = S2

        # Azimuth frame (t1v, t2v) perpendicular to d_in; phi is measured
        # from t1v toward t2v.  Because d_in x t1v = t2v and d_in x t2v =
        # -t1v, the scattering-plane perpendicular / parallel directions for
        # a given phi are
        #   perp   = -sin(phi) t1v + cos(phi) t2v
        #   par_in =  cos(phi) t1v + sin(phi) t2v .
        ax = np.zeros_like(d_in)
        ax[np.arange(m), np.argmin(np.abs(d_in), axis=-1)] = 1.0
        t1v = np.cross(d_in, ax)
        t1v /= np.linalg.norm(t1v, axis=-1, keepdims=True)
        t2v = np.cross(d_in, t1v)

        # Azimuth from the polarized differential cross-section.  Project the
        # incident Jones field E = Es*s_hat + Ep*p_old onto the frame:
        #   A = E . t1v,  B = E . t2v .
        # Then E_perp(phi) = -sin(phi) A + cos(phi) B and
        #      E_par(phi)  =  cos(phi) A + sin(phi) B, so the scattered
        # intensity dI(phi) = |S1|^2 |E_perp|^2 + |S2|^2 |E_par|^2 reduces to
        #   dI(phi) = pa + pb cos(2 phi) + pc sin(2 phi),   s1=|S1|^2 s2=|S2|^2
        #   pa = (s1+s2)/2 (|A|^2+|B|^2)   ( = (s1+s2)/2 * P_in )
        #   pb = (s2-s1)/2 (|A|^2-|B|^2)
        #   pc = (s2-s1)  Re(A conj(B)).
        # The phi-integral is 2*pi*pa proportional to (s1+s2)*P_in, i.e.
        # INDEPENDENT of the polarization state, so the theta-marginal of the
        # joint law equals the unpolarized phase function used above to draw
        # mu_s — sampling theta then phi|theta is exact (asserted in tests).
        p_old = np.cross(d_in, child.s_hat)
        s_hat = child.s_hat
        A = child.Es * np.sum(s_hat * t1v, axis=-1) \
            + child.Ep * np.sum(p_old * t1v, axis=-1)
        B = child.Es * np.sum(s_hat * t2v, axis=-1) \
            + child.Ep * np.sum(p_old * t2v, axis=-1)
        if self.cloud.pol_scatter:
            s1 = np.abs(S1v) ** 2
            s2 = np.abs(S2v) ** 2
            A2 = np.abs(A) ** 2
            B2 = np.abs(B) ** 2
            pa = 0.5 * (s1 + s2) * (A2 + B2)
            pb = 0.5 * (s2 - s1) * (A2 - B2)
            pc = (s2 - s1) * np.real(A * np.conj(B))
            phi = _sample_phi_polarized(pa, pb, pc, self.cloud.rng)
        else:
            phi = self.cloud.rng.uniform(0, 2 * np.pi, m)

        st = np.sqrt(np.clip(1 - mu_s ** 2, 0, 1))
        d_out = (mu_s[:, None] * d_in
                 + (st * np.cos(phi))[:, None] * t1v
                 + (st * np.sin(phi))[:, None] * t2v)
        # scattering plane basis: perpendicular = normalize(d_in x d_out)
        perp = np.cross(d_in, d_out)
        nrm = np.linalg.norm(perp, axis=-1, keepdims=True)
        fallback = t1v
        perp = np.where(nrm > 1e-12, perp / np.maximum(nrm, 1e-300),
                        fallback)
        par_in = np.cross(perp, d_in)      # in-plane, incident side
        # rotate Jones into scattering-plane basis (s = perp); with the
        # sampled phi this yields exactly (E_perp, E_par)
        Es, Ep = fr.rotate_jones(child.Es, child.Ep, s_hat, p_old,
                                 perp, par_in)
        # Apply the Mie amplitudes, then rescale to power albedo * P_in.
        # With phi drawn from the polarized law above this is a faithful
        # single-sample importance estimator: the DIRECTION density carries
        # the cos(2 phi) polarization modulation while each child still
        # carries exactly albedo * P_in, so per-event energy is exact (no
        # longer merely exact-on-average).  pol_scatter=False falls back to
        # uniform phi (the legacy azimuth-flattened behavior).
        alb = self.albedo_p[jj]
        p_in = (np.abs(Es) ** 2 + np.abs(Ep) ** 2)
        p_raw = np.abs(Es * S1v) ** 2 + np.abs(Ep * S2v) ** 2
        C = np.sqrt(alb * p_in / np.maximum(p_raw, 1e-300))
        Es2 = Es * S1v * C
        Ep2 = Ep * S2v * C
        child.dir = d_out
        child.s_hat = perp
        child.Es = Es2
        child.Ep = Ep2
        child.scattered[:] = True
        # coherent flag preserved: frozen realization -> physical speckle
        absorbed = p_in * (1.0 - alb)
        if np.any(absorbed > 0):
            tracer.ledger.credit("particle_absorbed",
                                 child.source_id, absorbed,
                                 where="particles")

        # remove collided rays from the surface-bound batch
        keep = np.ones(len(batch), dtype=bool)
        keep[coll_rows] = False
        batch2 = batch.select(keep)
        return t[keep], fid[keep], batch2, child
