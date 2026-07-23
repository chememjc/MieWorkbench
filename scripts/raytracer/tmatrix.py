# =============================================================================
# tmatrix.py — T-matrix evaluator for aspherical (spheroid) particles,
# duck-typing raytracer.mie.MieEvaluator so particles.py's EnsembleTables /
# ExplicitRealization can use either interchangeably (see make_evaluator()
# at the bottom). Built on the pytmatrix package (Mishchenko's T-matrix
# Fortran code via a Python wrapper); pytmatrix is a soft dependency of the
# optics env only (see _require_pytmatrix() below).
#
# -----------------------------------------------------------------------
# PHYSICS: random orientation
# -----------------------------------------------------------------------
# Particles are modeled as ORIENTATION-AVERAGED spheroids — the physically
# right model for particles tumbling freely in suspension (Brownian
# rotation on timescales far faster than any macroscopic observation). We
# average over ALL 3D orientations (pytmatrix's `orient_averaged_fixed`
# with a uniform_pdf() over the polar Euler angle `beta` and a uniform
# sweep of the azimuthal Euler angle `alpha`); a spheroid's third
# (self-rotation) Euler angle is physically irrelevant by symmetry, so
# this is the full isotropic average. For a TRUE random-orientation
# ensemble of an axisymmetric particle, the orientation-averaged phase
# matrix depends on the scattering angle theta ALONE (Mishchenko's
# random-orientation symmetry result) — not on the scattering azimuth or
# the incident polarization state — which is exactly what lets a single
# 1-D theta sweep stand in for the full 4*pi solid angle (see `_solve`).
#
# -----------------------------------------------------------------------
# CONVENTION (loud, because it changes what a size distribution means):
#   `r` is the VOLUME-EQUIVALENT SPHERE RADIUS, i.e. the radius of a
#   sphere with the same volume as the spheroid (pytmatrix's
#   RADIUS_EQUAL_VOLUME, which is the Scatterer default). A LogNormalDistribution
#   built over `r` for a TMatrixEvaluator therefore describes a size
#   distribution of EQUIVALENT VOLUMES, not of any linear dimension of the
#   spheroid itself (its actual semi-axes are r*aspect_ratio**(-1/3) and
#   r*aspect_ratio**(2/3)-scaled per pytmatrix's own convention — see
#   pytmatrix.tmatrix.Scatterer docs). Qext/Qsca (efficiencies vs
#   pi*r**2, matching mie.py's convention) are therefore NOT efficiencies
#   against the spheroid's true (orientation-dependent) geometric cross
#   section; they are whatever numbers make sigma = Q * pi * r**2
#   reproduce the correct orientation-averaged cross sections — which is
#   all particles.py ever does with them (`sig_e = pi*r**2*qext`), so the
#   convention is self-consistent end to end.
#
#   `aspect_ratio` is pytmatrix's "horizontal-to-rotational axis ratio":
#   > 1 is OBLATE (flattened, like a lentil/M&M), < 1 is PROLATE
#   (elongated, like a rugby ball/rod), == 1 is a sphere (falls back to
#   pytmatrix's single-orientation solver, since orientation-averaging a
#   sphere is a costly no-op — and axis_ratio values only *infinitesimally*
#   off 1.0 are numerically pathological in the underlying Fortran solver,
#   so we special-case EXACTLY 1.0 rather than "close to").
#
# -----------------------------------------------------------------------
# pytmatrix -> mie.py convention mapping (VERIFIED, not assumed):
# -----------------------------------------------------------------------
#   Geometry: incidence fixed at (thet0=90, phi0=0); the scattering angle
#   theta (mu = cos(theta)) is swept via (thet=90, phi=degrees(theta)),
#   i.e. phi=0 is forward (mu=1) and phi=180 is backward (mu=-1). This
#   in-plane sweep is pytmatrix's own `geom_horiz_forw`/`geom_horiz_back`
#   axis (tmatrix_aux.py).
#
#   Amplitude mapping: mie.py's S1 (perpendicular) <-> pytmatrix's S[0,0];
#   S2 (parallel) <-> pytmatrix's S[1,1]. Verified for a sphere
#   (axis_ratio=1) against miepython.S1_S2 at m=3+0.5j, x=2*pi: the
#   forward values agree (S1(0)==S2(0), the standard sphere degeneracy),
#   and |S1(mu)|/|S2(mu)| tracks miepython's ratio to <0.2% at mu in
#   {1,0.7,0.3,0,-0.3,-0.7,-1}. The two libraries carry DIFFERENT overall
#   complex normalizations (miepython's default "albedo" norm is an
#   arbitrary per-(r,lambda) scalar tied to Qext; pytmatrix's S is the
#   bare physical amplitude satisfying the optical theorem directly) and
#   an angle-dependent absolute phase gauge that differs between the two
#   codes (a harmless freedom: mie.py and particles.py only ever consume
#   |S1|, |S2| — never Re/Im or the S1*conj(S2) cross term — so this gauge
#   difference has zero physical consequence here; see the phase_function
#   / intercept() call sites in mie.py / particles.py). Cross-polarization
#   terms S[0,1]/S[1,0] vanish (<1e-15) for a single orientation, matching
#   the sphere's exact zero.
#
#   IMPORTANT SUBTLETY (found by verification, not assumed): pytmatrix's
#   `orient_averaged_fixed` orientation-averages S *coherently* (a
#   straight weighted sum of the complex per-orientation S) because that
#   coherent mean amplitude is exactly what the optical-theorem forward
#   value needs to give the correct Qext (extinction genuinely is a
#   coherent forward-interference effect, even for a random ensemble —
#   confirmed: `ext_xsect` reproduces the Mie/pytmatrix reference exactly).
#   But total SCATTERED POWER from independently, randomly oriented
#   particles is an INCOHERENT sum: the right ensemble quantity is
#   <|S|^2>, not |<S>|^2 (Jensen's inequality — the latter under-counts).
#   Squaring the coherently-averaged S for Qsca/g/amplitudes was tried
#   first and measured ~15% LOW against pytmatrix's own (slow) sca_xsect/
#   asym reference for a non-sphere case (axis_ratio=1.3) — confirming the
#   gap is exactly this coherent-vs-incoherent averaging effect (it
#   vanishes for axis_ratio=1, where there is only one orientation to
#   average and no gap can appear — which is why the sphere-limit checks
#   above didn't catch it). The fix: use pytmatrix's phase matrix Z
#   (`get_Z()`/`get_SZ()`), which `orient_averaged_fixed` DOES average
#   incoherently (it is bilinear in the fields, i.e. an intensity/Mueller
#   quantity, and stays correct under linear averaging). For unpolarized
#   incidence, Z[0,0](theta) IS the correct ensemble-averaged unpolarized
#   differential scattering intensity, and (verified directly against
#   get_S() for a single orientation, where S is unambiguous — see
#   `_solve_uncached` for the derivation comment; this is the OPPOSITE
#   pairing from pytmatrix's own `scatter.sca_intensity(h_pol=True)`,
#   whose h/v radar-polarization labeling does not itself determine which
#   of our S1/S2 it matches):
#       |S1(theta)|^2 = Z[0,0](theta) + Z[0,1](theta)   (perp, S[0,0])
#       |S2(theta)|^2 = Z[0,0](theta) - Z[0,1](theta)   (par, S[1,1])
#   Because the ensemble's SCATTERING PHASE is randomized by the random
#   orientation (only the ensemble-averaged intensity is well-defined),
#   amplitudes() reports these as REAL non-negative values (a phase
#   convention of 0) — exactly matching what particles.py and
#   mie.py.phase_function ever consume (|S1|, |S2| only, see above).
#
#   Efficiencies: Qext is the FAST optical-theorem evaluate at forward
#   scattering (`pytmatrix.scatter.ext_xsect`, a single coherent-S(0)
#   call — cheap and exact, verified above). Qsca and g are NOT computed
#   via `pytmatrix.scatter.sca_xsect`/`asym` — those run a 2-D adaptive
#   scipy.dblquad over the full 4*pi sphere and were measured at
#   ~140 SECONDS for a single (r, lambda) point (n_alpha=5, n_beta=10,
#   axis_ratio=1.3) — utterly impractical for a size-distribution
#   quadrature grid. Instead we reuse the SAME theta-sweep computed for
#   amplitudes() (a few hundred ms) and integrate the (correct, Z-based)
#   unpolarized intensity ourselves:
#       sigma_sca = 2*pi * Int_0^pi Z00(theta) sin(theta) dtheta
#       g         = Int Z00 cos(theta) sin(theta) dtheta / (that)
#   (NOTE: no 1/k^2 factor — pytmatrix's Z is normalized so this holds
#   directly; empirically verified against pytmatrix's own dblquad
#   sca_xsect/asym reference for axis_ratio=1.3: matched to <0.02% — see
#   test_tmatrix.py::test_fast_qsca_g_matches_slow_dblquad_reference).
#
# -----------------------------------------------------------------------
# PERFORMANCE: disk + memory cache
# -----------------------------------------------------------------------
# particles.py calls efficiencies()/amplitudes() over (r, lambda)
# quadrature grids (EnsembleTables: ~48 radii x N wavelengths). Each
# distinct (r, lambda) pair costs one "solve" (~50-250 ms depending on
# n_mu/aspect_ratio) that computes efficiencies AND the amplitude table
# TOGETHER (they share the same orientation-averaged theta sweep) and
# caches the result:
#   * in-memory, for the lifetime of the TMatrixEvaluator instance
#   * on disk, under <cache_root>/tmatrix/<sha1[:2]>/<sha1>.json, keyed by
#     a hash of (particle n,k at lambda, host n at lambda, aspect_ratio,
#     r, lambda, n_mu, n_alpha, n_beta) — i.e. keyed on the RESOLVED
#     optical constants, not material names, so two differently-named
#     materials with identical n,k share a cache entry (and a stale
#     dispersion-model edit invalidates it automatically).
# `cache_root` defaults to <repo>/var/cache/tmatrix, honoring the same
# MIEWB_CACHE_DIR env var mieworkbench/core/geomcache.py uses for its own
# <root>/var/cache tree (this module just adds a "tmatrix" subdirectory).
# =============================================================================
import hashlib
import json
import os
import sys

import numpy as np

from .mie import MieEvaluator

_HAVE_PYTMATRIX = False
_PYTMATRIX_IMPORT_ERROR = None
try:
    from pytmatrix.tmatrix import Scatterer as _Scatterer
    from pytmatrix import orientation as _orientation
    from pytmatrix import scatter as _scatter
    _HAVE_PYTMATRIX = True
except ImportError as _exc:                                    # pragma: no cover
    _PYTMATRIX_IMPORT_ERROR = _exc


def _require_pytmatrix():
    if _HAVE_PYTMATRIX:
        return
    raise ImportError(
        "TMatrixEvaluator needs the 'pytmatrix' package (T-matrix "
        "scattering for aspherical/spheroid particles) — it is not "
        "importable in this interpreter (%s).\n"
        "pytmatrix is an OPTICS-ENV-ONLY dependency (never the GUI venv, "
        "never FreeCAD's python — see INSTALL.md 'The optics environment'"
        "). Install it there:\n"
        "    $MIEWB_OPTICS_PYTHON -m pip install pytmatrix==0.3.3\n"
        "(substitute your local optics env's bin/pip if MIEWB_OPTICS_PYTHON "
        "points elsewhere). Original import error: %r"
        % (sys.executable, _PYTMATRIX_IMPORT_ERROR))


def _repo_root():
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def default_cache_root():
    """<repo>/var/cache/tmatrix, overridable via MIEWB_CACHE_DIR (the same
    env var mieworkbench/core/geomcache.py roots its own tessellation
    cache under) — this module keeps its entries in a private "tmatrix"
    subdirectory of that shared root."""
    root = os.environ.get(
        "MIEWB_CACHE_DIR", os.path.join(_repo_root(), "var", "cache"))
    return os.path.join(root, "tmatrix")


class TMatrixEvaluator(MieEvaluator):
    """Orientation-averaged T-matrix optics for one (particle, host,
    aspect_ratio) spheroid family — duck-types MieEvaluator (subclasses it
    to inherit phase_function()/sample_scatter_mu()/m_rel()/size_param()
    unchanged; only efficiencies() and amplitudes() are overridden).

    r means the volume-equivalent-sphere radius; see the module docstring
    above for the full convention writeup and the pytmatrix<->mie.py
    amplitude-mapping verification.
    """

    def __init__(self, particle_material, host_material, aspect_ratio,
                 cache_dir=None, n_mu=181, n_alpha=5, n_beta=10):
        _require_pytmatrix()
        super().__init__(particle_material, host_material)
        if not (aspect_ratio > 0):
            raise ValueError(
                "aspect_ratio must be > 0 (horizontal/rotational axis "
                "ratio; got %r)" % (aspect_ratio,))
        self.aspect_ratio = float(aspect_ratio)
        self.cache_dir = cache_dir or default_cache_root()
        self.n_mu = int(n_mu)
        self.n_alpha = int(n_alpha)
        self.n_beta = int(n_beta)
        self._solve_cache = {}   # in-memory (r, lam) -> solve dict

    # -- cache plumbing ---------------------------------------------------
    def _cache_key(self, r, lam):
        n_p = complex(self.mat_p.n_complex(np.array([lam], dtype=float))[0])
        n_h = float(np.real(
            self.mat_h.n_complex(np.array([lam], dtype=float))[0]))
        payload = ("%.8e|%.8e|%.8e|%.10f|%.10e|%.10e|%d|%d|%d" % (
            n_p.real, n_p.imag, n_h, self.aspect_ratio, float(r), float(lam),
            self.n_mu, self.n_alpha, self.n_beta))
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return digest, payload

    def _disk_path(self, digest):
        return os.path.join(self.cache_dir, digest[:2], digest + ".json")

    def _load_disk(self, digest):
        path = self._disk_path(digest)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            return None
        try:
            mu = np.asarray(raw["mu"], dtype=np.float64)
            S1 = (np.asarray(raw["S1_re"], dtype=np.float64)
                  + 1j * np.asarray(raw["S1_im"], dtype=np.float64))
            S2 = (np.asarray(raw["S2_re"], dtype=np.float64)
                  + 1j * np.asarray(raw["S2_im"], dtype=np.float64))
            return {"qext": float(raw["qext"]), "qsca": float(raw["qsca"]),
                    "g": float(raw["g"]), "mu": mu, "S1": S1, "S2": S2}
        except (KeyError, ValueError):
            return None

    def _save_disk(self, digest, payload, d):
        path = self._disk_path(digest)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = {
            "key": payload,
            "qext": d["qext"], "qsca": d["qsca"], "g": d["g"],
            "mu": d["mu"].tolist(),
            "S1_re": d["S1"].real.tolist(), "S1_im": d["S1"].imag.tolist(),
            "S2_re": d["S2"].real.tolist(), "S2_im": d["S2"].imag.tolist(),
        }
        tmp = path + ".tmp.%d" % os.getpid()
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, path)   # atomic on the same filesystem

    def _solve(self, r, lam):
        """The one expensive step: an orientation-averaged theta sweep at
        a single (r, lam), giving efficiencies AND an amplitude table
        together. Memory-cached per instance, disk-cached across runs."""
        r = float(r)
        lam = float(lam)
        mem_key = (round(r, 15), round(lam, 15))
        cached = self._solve_cache.get(mem_key)
        if cached is not None:
            return cached
        digest, payload = self._cache_key(r, lam)
        d = self._load_disk(digest)
        if d is None:
            d = self._solve_uncached(r, lam)
            self._save_disk(digest, payload, d)
        self._solve_cache[mem_key] = d
        return d

    # -- the actual pytmatrix work -----------------------------------------
    def _solve_uncached(self, r, lam):
        m = complex(self.m_rel(lam)[0])
        n_h = float(np.real(
            self.mat_h.n_complex(np.array([lam], dtype=float))[0]))
        wl_medium = lam / n_h   # pytmatrix wants the wavelength IN THE
        # MEDIUM, matching this module's own size_param()/m_rel() convention
        # (x = 2 pi r n_h / lam_vacuum == 2 pi r / (lam_vacuum/n_h)).
        sca = _Scatterer(radius=r, wavelength=wl_medium, m=m,
                          axis_ratio=self.aspect_ratio,
                          suppress_warning=True)
        is_sphere = (self.aspect_ratio == 1.0)
        if is_sphere:
            # orientation-averaging a sphere is a costly no-op; axis_ratio
            # values only infinitesimally off 1.0 are numerically
            # pathological in the underlying Fortran solver (verified:
            # 1.0000001 gives garbage ~1e6x off), so we special-case
            # EXACTLY 1.0 rather than "close to".
            sca.orient = _orientation.orient_single
        else:
            sca.orient = _orientation.orient_averaged_fixed
            sca.or_pdf = _orientation.uniform_pdf()
            sca.n_alpha = self.n_alpha
            sca.n_beta = self.n_beta

        # Qext: cheap single-point optical theorem (see module docstring).
        sca.thet0 = 90.0
        sca.phi0 = 0.0
        qext = float(_scatter.ext_xsect(sca) / (np.pi * r ** 2))

        # Qsca, g, and the amplitude table: one shared theta sweep, using
        # the phase matrix Z (INCOHERENTLY orientation-averaged — see the
        # module docstring for why this must not be derived from squaring
        # the coherently-averaged S).
        sca.thet = 90.0
        theta = np.linspace(0.0, np.pi, self.n_mu)
        Z00 = np.empty(self.n_mu, dtype=np.float64)
        Z01 = np.empty(self.n_mu, dtype=np.float64)
        for i, th in enumerate(theta):
            sca.phi = float(np.degrees(th))
            Z = sca.get_Z()
            Z00[i] = Z[0, 0]
            Z01[i] = Z[0, 1]

        sin_t = np.sin(theta)
        sigma_sca = 2.0 * np.pi * np.trapezoid(Z00 * sin_t, theta)
        qsca = float(sigma_sca / (np.pi * r ** 2))
        num_g = 2.0 * np.pi * np.trapezoid(Z00 * sin_t * np.cos(theta), theta)
        g = float(num_g / sigma_sca) if sigma_sca > 0 else 0.0

        # |S1|^2 = Z00+Z01 (perp, matches S[0,0]), |S2|^2 = Z00-Z01 (par,
        # matches S[1,1]) — verified directly against get_S() for a sphere
        # (single orientation, so S is unambiguous): |S[0,0]|^2 == Z00+Z01
        # and |S[1,1]|^2 == Z00-Z01 to float precision at every mu tested.
        # (NOTE this is the OPPOSITE pairing from pytmatrix's own
        # `scatter.sca_intensity(h_pol=True)` == Z00-Z01 -- that h/v
        # labeling is pytmatrix's radar convention and does not itself
        # determine which of our S1/S2 it corresponds to; we verified
        # against get_S() directly rather than trusting the h_pol name.)
        # Reported as real non-negative "amplitudes" (phase convention 0)
        # since the ensemble phase is randomized by construction — only
        # the magnitude is physically meaningful here (see docstring).
        S1 = np.sqrt(np.clip(Z00 + Z01, 0.0, None)).astype(np.complex128)
        S2 = np.sqrt(np.clip(Z00 - Z01, 0.0, None)).astype(np.complex128)

        mu = np.cos(theta)                    # descending: 1 -> -1
        order = np.argsort(mu)                # store ascending for np.interp
        return {"qext": qext, "qsca": qsca, "g": g,
                "mu": mu[order], "S1": S1[order], "S2": S2[order]}

    # -- public duck-typed surface (mirrors MieEvaluator exactly) ----------
    def efficiencies(self, r, lam):
        """(qext, qsca, g) for scalar or matched-shape r, lam [m]. `g` is
        the orientation-averaged asymmetry parameter <cos theta> — same
        third-return-value semantics as MieEvaluator.efficiencies (NOT
        Qabs; Qabs = qext - qsca, exactly as with MieEvaluator)."""
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        lam = np.broadcast_to(np.atleast_1d(lam), r.shape).astype(float)
        qext = np.empty_like(r)
        qsca = np.empty_like(r)
        g = np.empty_like(r)
        for i in range(len(r)):
            d = self._solve(r[i], lam[i])
            qext[i], qsca[i], g[i] = d["qext"], d["qsca"], d["g"]
        return qext, qsca, g

    def amplitudes(self, r, lam, mu):
        """Complex S1(mu), S2(mu) for ONE particle radius/wavelength,
        orientation-averaged. mu: (K,) array of cos(theta). Interpolated
        (real/imag parts separately) off the cached theta-sweep table —
        see the module docstring for the pytmatrix<->mie.py mapping and
        why only |S1|, |S2| (never the raw phase) are physically load-
        bearing downstream."""
        d = self._solve(r, lam)
        mu_q = np.asarray(mu, dtype=np.float64)
        S1 = (np.interp(mu_q, d["mu"], d["S1"].real)
              + 1j * np.interp(mu_q, d["mu"], d["S1"].imag))
        S2 = (np.interp(mu_q, d["mu"], d["S2"].real)
              + 1j * np.interp(mu_q, d["mu"], d["S2"].imag))
        return S1, S2

    # phase_function() and sample_scatter_mu() are inherited unchanged from
    # MieEvaluator: both only ever call self.amplitudes(...), so they work
    # correctly (and share the same in-memory self._cache dict) as-is.


def make_evaluator(particle_material, host_material, shape,
                    aspect_ratio=None, **tmatrix_kwargs):
    """Factory: MieEvaluator for shape='sphere', TMatrixEvaluator for
    shape='spheroid'. `aspect_ratio` (pytmatrix convention: horizontal-to-
    rotational axis ratio, >1 oblate / <1 prolate) is required for
    'spheroid' and ignored for 'sphere'. Extra keyword arguments
    (cache_dir, n_mu, n_alpha, n_beta) are forwarded to TMatrixEvaluator."""
    shape_norm = (shape or "sphere").strip().lower()
    if shape_norm == "sphere":
        return MieEvaluator(particle_material, host_material)
    if shape_norm == "spheroid":
        if aspect_ratio is None:
            raise ValueError(
                "make_evaluator(shape='spheroid') needs an aspect_ratio "
                "(pytmatrix convention: horizontal-to-rotational axis "
                "ratio, >1 oblate / <1 prolate / ==1 sphere)")
        return TMatrixEvaluator(particle_material, host_material,
                                aspect_ratio, **tmatrix_kwargs)
    raise ValueError(
        "make_evaluator: unknown particle shape %r (expected 'sphere' or "
        "'spheroid')" % (shape,))
