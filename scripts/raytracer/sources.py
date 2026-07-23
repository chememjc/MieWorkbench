# =============================================================================
# sources.py — light-source sampling.
#
# Emitting face: the face named in the contract's source.emit_face (default:
# closest-to-origin, chosen at extract time; CLI-overridable upstream).
# Sampling is stratified-jittered in the face's canonical UV, rejected
# against the trim, so per-sample area weights dA are uniform-in-area over
# the face (needed as the coherent-gather quadrature weight).
#
# Emission direction ("light emitted toward the origin only"):
#   * planar face  -> collimated along the face normal, sign chosen so the
#     beam heads toward the origin hemisphere (per-face).
#   * curved face  -> local surface normal per sample, sign chosen per
#     sample toward the origin; a mixed-sign face triggers a loud warning
#     and the against-origin samples are dropped with their power credited
#     to the 'emission_clipped' audit bucket.
#
# Wavelengths (nm params from the contract):
#   * neither lambdamin nor lambdamax  -> monochromatic at lambdac
#   * both        -> asymmetric Gaussian: sigma- = lambdac - lambdamin,
#                    sigma+ = lambdamax - lambdac (each side a half-normal,
#                    side chosen with probability sigma/(sigma-+sigma+))
#   * exactly one -> uniform on [lambdac - w, lambdac + w] where w is the
#                    given half-width (spec: "uniformly distributed around
#                    the center wavelength")
# Sampling is STRATIFIED: n_lambda equal-probability quantiles, one
# deterministic lambda per stratum, equal power weights. Rays are assigned
# a stratum id so the gather keeps per-stratum coherent accumulators
# (different optical frequencies never interfere stationarily).
#
# Phase reference: opl = 0 on the emitting surface — the surface IS the
# source wavefront (plane wave for a flat laser, sphere wave for the
# divergent laser). 'coherent' sources get zero initial phase; incoherent
# sources get a uniform random phase per ray (fringe visibility ~ 0).
#
# Polarization (contract source.polarization, parsed dict from
# common.parse_polarization_spec; absent -> unpolarized):
#   * unpolarized -> TWO mutually-incoherent orthogonal populations
#     (pol_stratum 0/1), rays alternating between them; the gather keeps
#     per-(source, lam, pol) accumulators so the populations never
#     interfere. This is exact for polarizer/retarder chains (Malus etc.),
#     unlike the old equal-split single Jones vector (which was really
#     45-degree linear light).
#   * linear:<deg> / circular:left|right / elliptical:<psi>:<chi> -> one
#     population with the exact Jones vector.
# Angle reference frame per ray: e_ref = global +z projected transverse
# to the emission direction (fallback +y when emitting along z);
# e_perp = dir x e_ref. s_hat is set to e_ref so (Es, Ep) IS the Jones
# vector in that frame. linear:<deg> rotates from e_ref toward e_perp.
# Circular handedness: 'right' means the E-vector rotates clockwise as
# seen by an observer facing the ONCOMING beam (optics/Hecht convention);
# with the field convention Re[E exp(-i w t)] and p_hat = dir x s_hat
# that is Jones (1, +i)/sqrt(2) in the (e_ref, e_perp) basis.
# Elliptical (psi, chi): standard orientation/ellipticity angles,
# E = (cos psi cos chi - i sin psi sin chi,
#      sin psi cos chi + i cos psi sin chi).
# =============================================================================
import numpy as np

from .rays import RayBatch

C_LIGHT_MPS = 299792458.0


def apply_stratum_t0(batch, src):
    """Pulsed-optics birth-time offset hook (Phase P3 stub consumed by a
    later SPM/chirp phase). If src carries an optional "_stratum_t0"
    array (seconds, one entry per wavelength stratum — same stratum
    indexing as wavelength_strata/batch.lam_stratum), add
    C_LIGHT_MPS * t0[stratum] metres to batch.gopl for every ray, keyed
    by its own lam_stratum. gopl is a pure Sum(n_g*ds) accumulator that
    is exactly 0 at birth by contract (RayBatch.alloc_time's docstring)
    and never read to make trace decisions (track_time is diagnostic
    only — see tracer.py) — so nudging it once, right at birth, by a
    per-stratum constant is equivalent to starting the group-delay
    integration from a nonzero t0 for that stratum.

    No-op when "_stratum_t0" is absent, OR when batch.gopl is still None.
    The second guard is the one that actually matters in this codebase's
    real call order: sample_source() (this module) ALWAYS runs and
    returns a fresh batch BEFORE scripts/run_trace.py ever calls
    Tracer.run(batches) — and Tracer.run() is what allocates gopl
    (`if self.cfg.track_time and b.gopl is None: b.alloc_time()`,
    tracer.py Tracer.run()). So batch.gopl is unconditionally None right
    after sample_source() builds a batch; this function is a plain
    no-op there. It only fires for a caller that pre-allocates —
    run_trace.py's batch-building loops call b.alloc_time() themselves
    (under the same cfg.track_time gate) right after sample_source()
    returns and BEFORE tracer.run(), then call this function — the same
    idempotent-alloc dance test_time_core.py's _build() harness already
    does by hand. Kept as a free function (not inlined at the
    batch.lam_stratum assignment below) so callers control exactly when
    it fires and a batch with track_time off stays a true zero-cost
    no-op."""
    t0 = src.get("_stratum_t0")
    if t0 is None or batch.gopl is None:
        return
    t0 = np.asarray(t0, dtype=np.float64)
    batch.gopl += C_LIGHT_MPS * t0[batch.lam_stratum]


def parse_spm_spec(raw):
    """'phimax:<rad>' or 'gamma:<W^-1 km^-1>:length:<m>' -> dict.
    The authoring grammar for the `spm` source property (pulsed-optics
    P6): either the peak nonlinear phase directly, or the fiber
    nonlinearity gamma (the datasheet unit, per-watt-per-KILOMETRE) and
    an effective length, from which phi_max = gamma * P_pk * L_eff
    (P_pk = the source's derived Gaussian peak power)."""
    parts = [p.strip() for p in str(raw).split(":")]
    try:
        if len(parts) == 2 and parts[0] == "phimax":
            v = float(parts[1])
            if v <= 0:
                raise ValueError("phimax must be > 0 rad")
            return {"phimax": v}
        if len(parts) == 4 and parts[0] == "gamma" and parts[2] == "length":
            g, L = float(parts[1]), float(parts[3])
            if g <= 0 or L <= 0:
                raise ValueError("gamma and length must be > 0")
            return {"gamma_W_km": g, "length_m": L}
    except ValueError as e:
        raise ValueError("bad spm spec %r: %s" % (raw, e))
    raise ValueError(
        "bad spm spec %r (expected 'phimax:<rad>' or "
        "'gamma:<W^-1km^-1>:length:<m>')" % raw)


# SPM synthesis grid: 4096 points spanning +-8 tau captures the Gaussian
# envelope to below 1e-38 and resolves phi_max up to ~50 rad
_SPM_N = 4096
_SPM_SPAN_TAU = 8.0
# SPD floor relative to the spectral peak kept in the installed table
_SPM_SPD_FLOOR = 1e-5


def install_spm(scene, n_lambda):
    """Pulsed-optics P6: apply the SOURCE-SIDE self-phase-modulation
    transform to every source carrying an `spm` property, IN PLACE on the
    scene's source dicts. Two artifacts per source:

      * the EXACT pure-SPM power spectrum — E(t) = sqrt(I(t)) *
        exp(i phi_max I(t)/I0) on a 4096-pt grid, one FFT, converted to a
        wavelength-density table installed as the source's tabulated SPD
        (_spectrum_lam_nm/_spectrum_pdf, superseding lambdamin/lambdamax
        exactly like an emission-registry `spectrum` row);
      * the chirp — per-stratum birth-time offsets (src['_stratum_t0'],
        the P3 hook) from the analytic instantaneous-frequency curve
        delta_omega(t) = -d(phi_NL)/dt evaluated on its CENTRAL monotonic
        branch (between the two extrema), so the spectrogram shows the
        SPM S-curve with the physical tilt: leading edge red, trailing
        edge blue. The outer branches (each frequency also recurs in the
        wings) are folded onto the branch ends — a quasi-classical
        single-time-per-frequency approximation, documented in
        docs/RAYTRACER.md.

    Deterministic and RNG-free by construction: shard workers rebuild the
    scene per process and MUST arrive at bit-identical tables. Call right
    after Scene construction, before anything reads wavelength strata."""
    for _, src in scene.sources:
        raw = src.get("spm")
        if raw is None:
            continue
        spec = parse_spm_spec(raw)
        pulse = src.get("pulse") or {}
        tau = pulse.get("duration_s")
        if not tau:
            raise ValueError(
                "spm source: needs pulse_duration (the transform is "
                "defined on the Gaussian pulse envelope)")
        if "phimax" in spec:
            phimax = spec["phimax"]
        else:
            p_pk = pulse.get("peak_power_W")
            if not p_pk:
                raise ValueError(
                    "spm 'gamma:...:length:...' needs the derived peak "
                    "power — give the source pulse_energy + rep_rate (or "
                    "power + rep_rate) alongside pulse_duration")
            phimax = (spec["gamma_W_km"] * 1e-3) * p_pk * spec["length_m"]
        lam0 = src["lambdac_nm"] * 1e-9
        w0 = 2.0 * np.pi * C_LIGHT_MPS / lam0

        # ---- exact SPM spectrum: one FFT of the analytic field ----
        t = np.linspace(-_SPM_SPAN_TAU * tau, _SPM_SPAN_TAU * tau, _SPM_N,
                        endpoint=False)
        dt = t[1] - t[0]
        g = np.exp(-4.0 * np.log(2.0) * (t / tau) ** 2)   # I(t)/I0, FWHM tau
        field = np.sqrt(g) * np.exp(1j * phimax * g)
        spec_w = np.abs(np.fft.fftshift(np.fft.fft(field))) ** 2
        dw = np.fft.fftshift(np.fft.fftfreq(_SPM_N, dt)) * 2.0 * np.pi
        w_abs = w0 + dw
        keep = (spec_w > _SPM_SPD_FLOOR * spec_w.max()) & (w_abs > 0)
        lam_nm = 2.0 * np.pi * C_LIGHT_MPS / w_abs[keep] * 1e9
        # spectral density per wavelength: S(lam) = S(w) * dw/dlam
        pdf = spec_w[keep] * (2.0 * np.pi * C_LIGHT_MPS
                              / (lam_nm * 1e-9) ** 2)
        order = np.argsort(lam_nm)
        src["_spectrum_lam_nm"] = lam_nm[order]
        src["_spectrum_pdf"] = pdf[order] / pdf.max()
        src["lambdamin_nm"] = None
        src["lambdamax_nm"] = None
        src["_spm_phimax"] = float(phimax)     # echoed into case.json

        # ---- chirp: per-stratum birth time from the central branch ----
        # delta_omega(t) = -d/dt (phimax g(t)) = phimax 8ln2 t/tau^2 g(t);
        # monotonic increasing between its extrema (leading edge t<0 ->
        # red, trailing -> blue)
        dwt = phimax * 8.0 * np.log(2.0) * t / tau ** 2 * g
        i_lo, i_hi = int(np.argmin(dwt)), int(np.argmax(dwt))
        branch = slice(i_lo, i_hi + 1)
        lam_k = np.asarray(wavelength_strata(src, n_lambda),
                           dtype=np.float64)
        dw_k = 2.0 * np.pi * C_LIGHT_MPS / lam_k - w0
        src["_stratum_t0"] = np.interp(dw_k, dwt[branch], t[branch])


def n_pol_strata(src):
    """Number of mutually-incoherent polarization populations a source
    emits: 2 for unpolarized (the default), 1 for any explicit state."""
    pol = src.get("polarization") or {"kind": "unpolarized"}
    return 2 if pol.get("kind", "unpolarized") == "unpolarized" else 1


def _pol_reference_frame(dirs):
    """Per-ray transverse reference frame: e_ref = z projected transverse
    (fallback y when |z x dir| ~ 0), e_perp = dir x e_ref."""
    z = np.array([0.0, 0.0, 1.0])
    y = np.array([0.0, 1.0, 0.0])
    ref = z - np.sum(dirs * z, axis=-1, keepdims=True) * dirs
    nrm = np.linalg.norm(ref, axis=-1)
    fallback = nrm < 1e-9
    if np.any(fallback):
        alt = y - np.sum(dirs[fallback] * y, axis=-1, keepdims=True) \
            * dirs[fallback]
        ref[fallback] = alt
        nrm = np.linalg.norm(ref, axis=-1)
    e_ref = ref / nrm[:, None]
    e_perp = np.cross(dirs, e_ref)
    return e_ref, e_perp


def jones_for(pol, pol_stratum):
    """Unit Jones vector (Es, Ep) complex pair in the (e_ref, e_perp)
    basis for a polarization dict + stratum index. |Es|^2+|Ep|^2 = 1."""
    kind = (pol or {"kind": "unpolarized"}).get("kind", "unpolarized")
    if kind == "unpolarized":
        # two orthogonal fully-polarized populations of equal power
        return (1.0 + 0j, 0j) if pol_stratum == 0 else (0j, 1.0 + 0j)
    if kind == "linear":
        th = np.deg2rad(pol["angle_deg"])
        return (np.cos(th) + 0j, np.sin(th) + 0j)
    if kind == "circular":
        # 'right' = clockwise facing the oncoming beam (module header)
        s = 1.0 if pol["handedness"] == "right" else -1.0
        return (1.0 / np.sqrt(2) + 0j, s * 1j / np.sqrt(2))
    if kind == "elliptical":
        psi = np.deg2rad(pol["psi_deg"])
        chi = np.deg2rad(pol["chi_deg"])
        return (np.cos(psi) * np.cos(chi) - 1j * np.sin(psi) * np.sin(chi),
                np.sin(psi) * np.cos(chi) + 1j * np.cos(psi) * np.sin(chi))
    raise ValueError("unknown polarization kind %r" % kind)


class StratumWavelengths(np.ndarray):
    """Return type of wavelength_strata: behaves EXACTLY like the plain
    (n_strata,) float64 wavelength array every existing caller consumes
    (len / iteration / indexing / arithmetic are all inherited), plus an
    `edges` attribute: the (n_strata + 1,) per-stratum wavelength EDGES [m]
    (edges[k]..edges[k+1] brackets stratum k; monotonic non-decreasing).

    Added for the pulsed-optics time products (P4): the per-stratum
    ANGULAR-FREQUENCY bandwidth (stratum_domega) sets the analytic time-
    envelope kernel width for GDD-broadened arrivals. Views/slices carry
    the parent's edges along (diagnostic attribute only — never used to
    reindex a sliced array)."""

    def __array_finalize__(self, obj):
        self.edges = getattr(obj, "edges", None)


def _with_edges(lam_m, edges_m):
    out = np.ascontiguousarray(lam_m, dtype=np.float64).view(
        StratumWavelengths)
    out.edges = np.ascontiguousarray(edges_m, dtype=np.float64)
    return out


def stratum_domega(strata):
    """(n_strata,) angular-frequency bandwidth [rad/s] of each wavelength
    stratum, from a StratumWavelengths' edges: |2*pi*c*(1/lam_lo - 1/lam_hi)|
    over the stratum's wavelength bracket. Zero for a zero-width
    (monochromatic) stratum."""
    e = strata.edges
    return np.abs(2.0 * np.pi * C_LIGHT_MPS * (1.0 / e[:-1] - 1.0 / e[1:]))


# CDF quantile the OPEN (infinite-tail) edges of the asymmetric-Gaussian
# regime are clamped to: the outermost stratum's finite edge sits halfway
# (in probability) between its own center quantile (0.5/n from the end)
# and the open end — i.e. at 0.25/n from the end. A half-normal tail has
# no finite support, but its kernel-bandwidth proxy must be finite; the
# tail's half-mass point is the natural width surrogate (documented,
# pinned by test_time_products.py's broadening gate).
_EDGE_TAIL_FRACTION = 0.25


def _lines_stratum_counts(n_lambda, intensity):
    """(counts, keep_idx, dropped_idx) for a 'lines' source's n_lambda
    stratum budget over len(intensity) discrete lines, intensity-
    proportional, every KEPT line getting >= 1 stratum.

    n_lambda >= n_lines: every line is kept; counts[i] via the largest-
    remainder (Hamilton) apportionment method on the ideal shares
    n_lambda*weight_i (so an exactly-divisible ratio like 3:1:5 over 9
    strata reproduces 3,1,5 exactly), then any line that still landed on
    zero strata (possible for a very skewed weight, e.g. 4 strata over
    weights 0.9/0.05/0.05) is bumped to 1 by stealing one stratum from
    the currently largest-allocated line, repeated until no zero remains
    — keeps the "every kept line >= 1" invariant without disturbing the
    common (non-degenerate) case computed above.

    n_lambda < n_lines: only the strongest n_lambda lines (by intensity)
    survive, each getting exactly 1 stratum; keep_idx/dropped_idx (both
    ascending original-index order) let the caller renormalize power
    over the kept subset and warn about the drop."""
    intensity = np.asarray(intensity, dtype=np.float64)
    n_lines = intensity.size
    weights = intensity / intensity.sum()
    if n_lambda < n_lines:
        rank = np.argsort(-intensity, kind="stable")
        keep_idx = np.sort(rank[:n_lambda])
        dropped_idx = np.sort(rank[n_lambda:])
        return np.ones(n_lambda, dtype=np.int64), keep_idx, dropped_idx
    raw = n_lambda * weights
    floor = np.floor(raw).astype(np.int64)
    remainder = int(n_lambda - floor.sum())
    frac = raw - floor
    order = np.argsort(-frac, kind="stable")
    counts = floor.copy()
    counts[order[:remainder]] += 1
    while np.any(counts == 0):
        zero_i = int(np.argmin(counts))
        donor_i = int(np.argmax(counts))
        counts[donor_i] -= 1
        counts[zero_i] += 1
    return counts, np.arange(n_lines), np.array([], dtype=np.int64)


def _lines_strata(src, n_lambda):
    """wavelength_strata's 'lines' regime: n_lambda strata distributed
    over the source's discrete emission lines proportional to intensity
    (see _lines_stratum_counts), one deterministic wavelength per
    stratum (the line's own center — never an interpolated/synthetic
    value; a physical lamp line is narrower than any bench resolution,
    so every stratum belonging to one line samples that EXACT center)
    and equal power per stratum (sample_source's ray-count-per-stratum
    convention is what turns "more strata" into "more power" for a
    louder line — this function only ever returns wavelength POSITIONS).

    Edges (diagnostic — stratum_domega / time products only, see the
    StratumWavelengths class doc): each line contributes a
    linewidth_nm-wide band centered on it, split into that line's own
    stratum-count equal contiguous sub-bands (all exact). Bands across
    DIFFERENT lines are never adjacent in reality (the loader's overlap
    check guarantees a real gap) but the returned edges array is a
    single (n_strata+1,) shared-boundary sequence by contract — so at
    each line-to-line transition the boundary is pinned to the NEXT
    line's true lower edge (never a value inside a real line's own
    band, so no stratum's width can go to zero or negative): this
    inflates exactly one stratum per non-final line (its own last
    stratum, which absorbs the inter-line gap into its reported width)
    while every other stratum — everything interior to a line's own
    split, and the entire final line — keeps its exact linewidth_nm (or
    linewidth_nm/count) width. Absolute positions are otherwise the
    line's own true nm values throughout; only that one inflated
    boundary per gap is not "about its line center" by construction."""
    lines_nm = np.asarray(src["_lines_nm"], dtype=np.float64)
    intensity = np.asarray(src["_lines_intensity"], dtype=np.float64)
    width_nm = float(src["_lines_linewidth_nm"])
    counts, keep_idx, dropped_idx = _lines_stratum_counts(n_lambda, intensity)
    if dropped_idx.size:
        total = float(intensity.sum())
        share = float(intensity[dropped_idx].sum()) / total if total else 0.0
        import warnings
        warnings.warn(
            "lines source: n_lambda=%d < n_lines=%d — dropping the %d "
            "weakest line(s) at %s nm (%.1f%% of total line power) to "
            "fit the wavelength-stratum budget, and the %d kept line(s) "
            "get EQUAL power (one stratum each — intensity ratios cannot "
            "be represented below one stratum per line). Raise --nlambda "
            "to at least the line count for proportional line powers."
            % (n_lambda, intensity.size, dropped_idx.size,
               ", ".join("%.4g" % v for v in lines_nm[dropped_idx]),
               100.0 * share, keep_idx.size))
    lines_nm = lines_nm[keep_idx]
    lam_nm_list = []
    edges_nm_list = []
    for i, (center_nm, k) in enumerate(zip(lines_nm, counts)):
        lo_nm = center_nm - 0.5 * width_nm
        hi_nm = center_nm + 0.5 * width_nm
        local = np.linspace(lo_nm, hi_nm, int(k) + 1)
        if i == 0:
            edges_nm_list.extend(local.tolist())
        else:
            edges_nm_list.pop()          # drop the previous true-hi
            edges_nm_list.extend(local.tolist())   # incl. this true-lo
        lam_nm_list.extend([center_nm] * int(k))
    lam_m = np.asarray(lam_nm_list, dtype=np.float64) * 1e-9
    edges_m = np.asarray(edges_nm_list, dtype=np.float64) * 1e-9
    return _with_edges(lam_m, edges_m)


def wavelength_strata(src, n_lambda):
    """Deterministic per-stratum wavelengths [m] (equal probability each),
    returned as a StratumWavelengths (an ndarray subclass carrying the
    per-stratum wavelength EDGES — see the class docstring; every
    pre-existing caller keeps using the result as the plain array).

    Three regimes, each placing strata at CDF centers (k+0.5)/n so every
    stratum carries the same probability mass (equal-power stratified
    sampling; per-ray birth_power is untouched). Edges sit at the CDF
    quantiles k/n mapped through the same inverse transform:
      * a tabulated emission spectrum (_spectrum_lam_nm/_spectrum_pdf, from
        the emission registry — 'continuous' rows and 'blackbody' rows
        alike, the latter synthesized to a dense table AT LOAD by
        optprops.load_emission so it needs no special case here):
        inverse-CDF of the piecewise-linear PDF — densify each linear
        segment x16, cumulative-trapezoid CDF, invert;
      * discrete emission lines (_lines_nm/_lines_intensity/
        _lines_linewidth_nm, an emission registry 'lines' row): see
        _lines_strata — per-line stratum counts proportional to
        intensity, each stratum at its line's exact center;
      * an asymmetric-Gaussian line (lambdamin/lambdamax bracket lambdac):
        two glued half-normals (the two OPEN tail edges are clamped to the
        _EDGE_TAIL_FRACTION quantile — a half-normal has no finite rim);
      * a single bound: a symmetric uniform band around lambdac."""
    if src.get("_lines_nm") is not None:
        return _lines_strata(src, n_lambda)
    lam_c = src["lambdac_nm"]
    lam_lo = src.get("lambdamin_nm")
    lam_hi = src.get("lambdamax_nm")
    q = (np.arange(n_lambda) + 0.5) / n_lambda      # stratum centers in CDF
    q_edge = np.arange(n_lambda + 1) / n_lambda     # stratum edges in CDF
    lam_tab = src.get("_spectrum_lam_nm")
    if lam_tab is not None:
        # inverse-CDF sampling of a tabulated piecewise-linear PDF: densify
        # each segment (x16 points) so the trapezoid CDF is smooth, then map
        # the stratum-center quantiles through cdf^-1.
        lam_tab = np.asarray(lam_tab, dtype=np.float64)
        pdf_tab = np.asarray(src["_spectrum_pdf"], dtype=np.float64)
        segs = [np.linspace(lam_tab[i], lam_tab[i + 1], 16, endpoint=False)
                for i in range(lam_tab.size - 1)]
        lam_dense = np.concatenate(segs + [lam_tab[-1:]])
        pdf_dense = np.interp(lam_dense, lam_tab, pdf_tab)
        cdf = np.concatenate([[0.0], np.cumsum(
            0.5 * (pdf_dense[1:] + pdf_dense[:-1]) * np.diff(lam_dense))])
        cdf /= cdf[-1]
        return _with_edges(np.interp(q, cdf, lam_dense) * 1e-9,
                           np.interp(q_edge, cdf, lam_dense) * 1e-9)
    if lam_lo is None and lam_hi is None:
        # monochromatic: 1 zero-width stratum (edges collapse onto lam_c)
        return _with_edges(np.full(1, lam_c * 1e-9),
                           np.full(2, lam_c * 1e-9))
    if lam_lo is not None and lam_hi is not None:
        sig_m = lam_c - lam_lo
        sig_p = lam_hi - lam_c
        if sig_m < 0 or sig_p < 0:
            raise ValueError("source %r: lambdamin/lambdamax must bracket "
                             "lambdac" % src)
        if sig_m + sig_p == 0.0:
            # lambdamin == lambdamax == lambdac: a zero-width band is a
            # valid way to spell "monochromatic" (used to divide by zero)
            return _with_edges(np.full(1, lam_c * 1e-9),
                               np.full(2, lam_c * 1e-9))
        # two half-normals glued at lambda_c with weights sig-/sig+
        from scipy.stats import norm
        w_m = sig_m / (sig_m + sig_p)

        def gauss_ppf(qv):
            lam = np.empty(len(qv))
            left = qv < w_m
            # left side: q in [0,w_m) -> half-normal below lambda_c
            qq = qv[left] / max(w_m, 1e-300)
            lam[left] = lam_c - np.abs(
                norm.ppf(0.5 + 0.5 * (1 - qq))) * sig_m
            qq = (qv[~left] - w_m) / max(1 - w_m, 1e-300)
            lam[~left] = lam_c + np.abs(norm.ppf(0.5 + 0.5 * qq)) * sig_p
            return lam * 1e-9
        # clamp the two OPEN tail edges (q = 0, 1 map to -/+inf) to the
        # half-mass quantile of the outermost stratum's tail
        qe = q_edge.copy()
        qe[0] = _EDGE_TAIL_FRACTION / n_lambda
        qe[-1] = 1.0 - _EDGE_TAIL_FRACTION / n_lambda
        return _with_edges(gauss_ppf(q), gauss_ppf(qe))
    # exactly one bound: symmetric uniform around lambda_c
    w = (lam_c - lam_lo) if lam_lo is not None else (lam_hi - lam_c)
    return _with_edges((lam_c - w + 2.0 * w * q) * 1e-9,
                       (lam_c - w + 2.0 * w * q_edge) * 1e-9)


def _face_center_xyz(face):
    """The emitting face's reference center point in world xyz — the
    origin for apodization/beam-mode transverse radius, chosen the same
    way sample_viz_pattern picks a face centroid (mean of the trim loop
    vertices in the surface's own uv), so both features agree on what
    'center of the face' means."""
    surf = face.surface
    if face.trim.mode == "untrimmed":
        if surf.__class__.__name__ == "Sphere":
            return surf.c + surf.r * surf.axis
        raise NotImplementedError(
            "apodization/beam center undefined for an untrimmed %s face"
            % surf.__class__.__name__)
    if face.trim.mode == "band":
        v_lo, v_hi = face.trim.v_band
        return _uv_to_xyz(surf, np.zeros(1), np.array([0.5 * (v_lo + v_hi)]))[0]
    loop_uv = np.concatenate([np.asarray(lp) for lp in face.trim.loops])
    c_uv = loop_uv.mean(axis=0)
    return _uv_to_xyz(surf, c_uv[0:1], c_uv[1:2])[0]


def _sample_beam_points(face, w0_m, n_rays, rng):
    """Transverse position sampling for a beam_waist source: rejection-
    sample (dx, dy) ~ independent N(0, sigma) in the face tangent frame
    about the face center, sigma = w0/2 (matching the intensity profile
    I(r) ~ exp(-2 r^2/w0^2): a radially-symmetric 2D Gaussian factors into
    per-axis exp(-2 x^2/w0^2), i.e. std = w0/2), redrawing candidates that
    land outside the physical aperture.

    Truncation choice: REJECTION SAMPLING (not clip/clamp) — clamping would
    pile spurious power at the rim and distort the profile; rejecting from
    the untruncated Gaussian yields samples distributed exactly as the
    aperture-TRUNCATED Gaussian, so uniform per-ray power (the caller keeps
    p_ray = P/N unweighted) already integrates to the right shape without
    any additional per-ray reweighting — position density IS the physical
    photon density. Requires a planar emitting face (checked by the
    caller): the tangent frame (t1, t2) is then a fixed, ray-independent
    basis, which keeps this a simple 2D rejection sampler."""
    surf = face.surface
    center = _face_center_xyz(face)
    pts = np.empty((0, 3))
    target = n_rays
    tries = 0
    while len(pts) < target and tries < 200:
        m = int((target - len(pts)) * 1.5) + 16
        dx = rng.normal(0.0, w0_m / 2.0, size=m)
        dy = rng.normal(0.0, w0_m / 2.0, size=m)
        cand = center + dx[:, None] * surf.t1 + dy[:, None] * surf.t2
        ok = face.trim.contains(surf.to_uv(cand))
        pts = np.concatenate([pts, cand[ok]], axis=0)
        tries += 1
    if len(pts) < target:
        raise RuntimeError(
            "beam-mode source: Gaussian position sampling failed to "
            "converge (%d/%d after %d rounds) — w0 too large relative to "
            "the emitting face aperture?" % (len(pts), target, tries))
    pts = pts[:target]
    normals = surf.normal(pts)
    return pts, normals


def sample_source(scene, body, src, source_id, n_rays, n_lambda, rng,
                  ledger=None, differentials=False, export_rays=False):
    """Sample a RayBatch for one source. Power split equally across rays;
    each ray belongs to one wavelength stratum. differentials=True
    allocates Igehy ray differentials (wavefront patch h = sqrt(A/N)
    along the transverse frame; curvature from the emit surface's shape
    operator) for --ray-differentials dA tracking.

    Optional source.beam {waist_mm, m2}: Gaussian-beam mode — position
    sampled from the waist intensity profile (see _sample_beam_points)
    instead of uniformly over the face, plus a per-ray angular divergence
    added below (requires a planar emitting face). Optional
    source.apodization {kind:'gaussian', w0_mm, order}: transverse FIELD
    apodization — position sampling is untouched (any face shape), each
    ray's POWER is reweighted by the profile and renormalized to the exact
    source power instead."""
    face = scene.emit_faces.get(body.index)
    if face is None:
        raise ValueError("source %s has no emit face built — extractor/"
                         "scene mismatch" % body.label)
    surf = face.surface
    lam_strata = wavelength_strata(src, n_lambda)
    n_strata = len(lam_strata)
    power_W = src["power_mW"] * 1e-3
    coherent = bool(src.get("coherent", False))
    beam = src.get("beam")
    apod = src.get("apodization")

    if beam is not None:
        if surf.__class__.__name__ != "Plane":
            raise NotImplementedError(
                "source %s: beam_waist requires a planar emitting face "
                "(got %s)" % (body.label, surf.__class__.__name__))
        w0_m = beam["waist_mm"] * 1e-3
        pts, normals = _sample_beam_points(face, w0_m, n_rays, rng)
    else:
        pts, normals = _sample_face_points(face, n_rays, rng)
    n = len(pts)

    # direction: toward-origin sign policy
    to_origin = -pts                                  # origin - point
    flat = surf.__class__.__name__ == "Plane"
    if flat:
        n0 = normals[0]
        sign = 1.0 if np.dot(n0, np.mean(to_origin, axis=0)) >= 0 else -1.0
        dirs = np.tile(sign * n0, (n, 1))
        clipped = np.zeros(n, dtype=bool)
    else:
        dots = np.sum(normals * to_origin, axis=-1)
        sign = np.where(dots >= 0.0, 1.0, -1.0)
        # per-sample flip would fold the wavefront: emit only the samples
        # whose natural normal faces the origin; drop (and account) others
        frac_neg = np.mean(sign < 0)
        if 0.0 < frac_neg < 1.0:
            import warnings
            warnings.warn(
                "source %s: emitting face normals straddle the origin "
                "direction (%.1f%% clipped) — emission clipped to the "
                "origin-facing side" % (body.label, 100 * frac_neg))
        if frac_neg == 1.0:
            dirs = -normals
            clipped = np.zeros(n, dtype=bool)
        else:
            dirs = normals
            clipped = sign < 0
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    if apod is not None:
        # apodization weights ALL n samples (pre-clip) so the ledger.emit
        # call below (which records all n) still sums to power_W exactly
        # by construction: p_ray_i = power_W * w_i / sum(w) over all n.
        # (separate var from beam's w0_m — a source could carry both)
        center = _face_center_xyz(face)
        r_all = np.linalg.norm(pts - center, axis=-1)
        apod_w0_m = apod["w0_mm"] * 1e-3
        order = apod["order"]
        weight = np.exp(-2.0 * (r_all / apod_w0_m) ** (2 * order))
        p_ray = power_W * weight / weight.sum()       # (n,) exact-sum array
    else:
        p_ray = power_W / n                           # per-sample power

    keep = ~clipped
    pts, dirs = pts[keep], dirs[keep]
    n_kept = len(pts)
    if n_kept == 0:
        raise ValueError("source %s: all emission samples clipped — "
                         "check geometry orientation" % body.label)

    if beam is not None:
        # per-ray divergence half-angle theta0 = M2*lambda/(pi*w0) — the
        # STANDARD Gaussian-beam far-field 1/e^2-intensity half-angle
        # (same "1/e^2 in intensity" convention as w0 itself), evaluated
        # per ray at its own sampled wavelength for correct chromatic
        # divergence. Independent per-axis angle std = theta0/2: matching
        # the same near-field/far-field second-moment propagation
        # w(z)^2 = w0^2 + (theta0*z)^2 that _sample_beam_points' sigma =
        # w0/2 position sampling assumes (position std^2 + z^2*angle std^2
        # must equal w(z)^2/4 at every z, and w0/zR = theta0 exactly, which
        # forces angle std = theta0/2 — NOT theta0/sqrt(2)).
        lam_of_ray = lam_strata[np.arange(n_kept) % n_strata]
        theta0 = beam["m2"] * lam_of_ray / (np.pi * w0_m)
        ang_std = theta0 / 2.0
        ax = rng.normal(0.0, ang_std)
        ay = rng.normal(0.0, ang_std)
        tilted = dirs + np.tan(ax)[:, None] * surf.t1 + np.tan(ay)[:, None] * surf.t2
        dirs = tilted / np.linalg.norm(tilted, axis=-1, keepdims=True)

    if ledger is not None:
        # emitted = the FULL source power; clipped samples immediately
        # balance into their bucket so closure holds
        ledger.emit(np.full(n, source_id), np.broadcast_to(p_ray, (n,)))
        if np.any(clipped):
            p_clip = p_ray[clipped] if apod is not None \
                else np.full(int(np.sum(clipped)), p_ray)
            ledger.credit("emission_clipped",
                          np.full(int(np.sum(clipped)), source_id), p_clip)
    if apod is not None:
        p_ray = p_ray[keep]                           # (n_kept,) kept subset

    pol = src.get("polarization") or {"kind": "unpolarized"}
    n_pol = n_pol_strata(src)

    batch = RayBatch(n_kept)
    batch.pos[:] = pts
    batch.dir[:] = dirs
    if export_rays:
        # birth position on the source face (world metres); inherited
        # unchanged by every child so a detected ray keeps its pupil coord
        batch.birth_pos = batch.pos.copy()
    idx = np.arange(n_kept)
    batch.lam[:] = lam_strata[idx % n_strata]
    batch.lam_stratum[:] = idx % n_strata
    # NOTE: the src["_stratum_t0"] birth-time offset (apply_stratum_t0,
    # module top) is NOT applied here — batch.gopl is still None at this
    # point in the real pipeline (see that function's docstring for why);
    # the caller applies it after pre-allocating gopl.
    # interleave so every (lam, pol) combination is uniformly filled
    batch.pol_stratum[:] = (idx // n_strata) % n_pol
    batch.source_id[:] = source_id
    batch.coherent[:] = coherent
    batch.birth_power[:] = p_ray
    # polarization basis: s_hat = the global-z-referenced transverse frame
    # (module header) so (Es, Ep) IS the Jones vector in that frame
    e_ref, _ = _pol_reference_frame(dirs)
    batch.s_hat[:] = e_ref
    # broadcast to (n_kept,) unconditionally so the per-stratum masking
    # below (amp[sel]) works whether p_ray is the ordinary uniform scalar
    # or an apodization-weighted per-ray array
    amp = np.broadcast_to(np.sqrt(p_ray), n_kept)
    if coherent:
        phase = np.ones(n_kept, dtype=np.complex128)
    else:
        phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=n_kept))
    for ps in range(n_pol):
        js, jp = jones_for(pol, ps)
        sel = batch.pol_stratum == ps
        batch.Es[sel] = amp[sel] * js * phase[sel]
        batch.Ep[sel] = amp[sel] * jp * phase[sel]

    if differentials:
        # NOTE (scoped limitation): beam/apodization sources still seed the
        # differential patch from the WHOLE face area, same as an ordinary
        # source — the coherent gather's per-ray dA therefore falls back to
        # this uniform footprint rather than the true (non-uniform) sampling
        # density. Position-only accuracy (this function's actual contract)
        # and incoherent detection are unaffected; a beam-mode source fed
        # through the COHERENT Huygens gather gets an approximate (not
        # density-corrected) diffraction-pattern shape. Fine-grained fix
        # would need a per-ray dA at emission, which is out of scope here.
        from .differentials import init_flat, init_curved
        e_perp = np.cross(dirs, e_ref)
        h = np.sqrt((face.area_m2 or 1e-6) / n)
        if flat:
            dPdx, dDdx, dPdy, dDdy = init_flat(dirs, e_ref, e_perp, h)
        else:
            S = surf.normal_derivative(pts)
            n_can = surf.normal(pts)
            sign = np.sign(np.sum(dirs * n_can, axis=-1))
            dPdx, dDdx, dPdy, dDdy = init_curved(dirs, e_ref, e_perp, h,
                                                 S, sign)
        batch.alloc_differentials()
        batch.dPdx[:] = dPdx
        batch.dDdx[:] = dDdx
        batch.dPdy[:] = dPdy
        batch.dDdy[:] = dDdy
    return batch


def _emit_face_from_record(scene, body, src):
    raise ValueError(
        "source %s: emit_face %r is not in the scene's face table — "
        "extractor/scene mismatch" % (body.label, src["emit_face"]))


def _sample_face_points(face, n_rays, rng):
    """Stratified-jittered area sampling of an analytic face.

    Strategy: rejection-sample in a UV bounding box using the trim test,
    with area weights corrected by the local surface metric (first
    fundamental form). For the surfaces used here the metric is:
      plane: 1;  sphere: R^2 cos(v);  cylinder: R;  cone: |v| tan-ish;
    handled by importance-correcting the v coordinate analytically for
    sphere (sample sin(v) uniformly) and uniformly otherwise.
    Returns (points (M,3), normals (M,3) canonical).
    """
    surf = face.surface
    cls = surf.__class__.__name__
    # UV bounds from the trim polygon loops
    if face.trim.mode == "untrimmed":
        if cls == "Sphere":
            u_lo, u_hi = -np.pi, np.pi
            v_lo, v_hi = -np.pi / 2, np.pi / 2
        else:
            raise NotImplementedError(
                "untrimmed emitting face of type %s" % cls)
    elif face.trim.mode == "band":
        u_lo, u_hi = -np.pi, np.pi
        v_lo, v_hi = face.trim.v_band
    else:
        allu = np.concatenate([lp[:, 0] for lp in face.trim.loops])
        allv = np.concatenate([lp[:, 1] for lp in face.trim.loops])
        u_lo, u_hi = float(allu.min()), float(allu.max())
        v_lo, v_hi = float(allv.min()), float(allv.max())

    pts = np.empty((0, 3))
    target = n_rays
    tries = 0
    while len(pts) < target and tries < 60:
        m = int((target - len(pts)) * 1.8) + 16
        u = rng.uniform(u_lo, u_hi, size=m)
        if cls == "Sphere":
            # uniform in area: sample sin(v) uniformly
            sv = rng.uniform(np.sin(v_lo), np.sin(v_hi), size=m)
            v = np.arcsin(sv)
        else:
            v = rng.uniform(v_lo, v_hi, size=m)
        cand = _uv_to_xyz(surf, u, v)
        # containment evaluated through the same to_uv convention the trim
        # polygon itself was built with
        ok = face.trim.contains(surf.to_uv(cand))
        pts = np.concatenate([pts, cand[ok]], axis=0)
        tries += 1
    if len(pts) < target:
        raise RuntimeError(
            "source face %s: area sampling failed to converge "
            "(%d/%d after %d rounds) — trim geometry suspect"
            % (face.id, len(pts), target, tries))
    pts = pts[:target]
    normals = surf.normal(pts)
    return pts, normals


def _wrap(surf, u):
    return (u + np.pi) % (2 * np.pi) - np.pi


def _uv_to_xyz(surf, u, v):
    cls = surf.__class__.__name__
    if cls == "Plane":
        return surf.origin + u[:, None] * surf.t1 + v[:, None] * surf.t2
    if cls == "Sphere":
        cu_, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        return (surf.c
                + surf.r * (cv * cu_)[:, None] * surf.t1
                + surf.r * (cv * su)[:, None] * surf.t2
                + surf.r * sv[:, None] * surf.axis)
    if cls == "Cylinder":
        cu_, su = np.cos(u), np.sin(u)
        return (surf.o
                + surf.r * cu_[:, None] * surf.t1
                + surf.r * su[:, None] * surf.t2
                + v[:, None] * surf.a)
    raise NotImplementedError("emitting face of type %s" % cls)


def _rings_uv(loop_uv, c_uv, pattern):
    """rings:dr=<mm>:nper=<N>[:nrings=<K>] -> centroid + concentric rings,
    every dr mm, nper rays per ring, out to the trim rim (or K rings)."""
    r_rim = float(np.max(np.linalg.norm(loop_uv - c_uv, axis=-1)))
    dr = pattern["dr_mm"] * 1e-3
    n_rings = pattern["nrings"]
    if n_rings is None:
        n_rings = int(np.floor(r_rim / dr + 1e-9))
    uv = [c_uv]
    for k in range(1, n_rings + 1):
        theta = 2.0 * np.pi * np.arange(pattern["nper"]) / pattern["nper"]
        ring = c_uv + (k * dr) * np.stack(
            [np.cos(theta), np.sin(theta)], axis=-1)
        uv.append(ring)
    return np.concatenate([np.atleast_2d(p) for p in uv], axis=0)


def _fan_uv(loop_uv, c_uv, pattern):
    """fan[:n=<K>] (default K=5) -> centroid, then up to 4 cardinal rays
    along the face's local +y/-y/+x/-x directions at 95% of the trim's
    AXIS-ALIGNED extent in that direction (not the corner-to-corner rim
    radius rings uses) so the cardinal points land inside non-circular
    (e.g. square) apertures instead of past their corners. Any rays beyond
    the 4 cardinals fill the largest inscribed circle (95% of the smallest
    of the four axial extents) evenly spaced, offset by 45 deg so they
    don't coincide with the cardinal directions.
    """
    n = pattern["n"]
    u_lo, v_lo = loop_uv.min(axis=0)
    u_hi, v_hi = loop_uv.max(axis=0)
    ext_px = float(u_hi - c_uv[0])
    ext_mx = float(c_uv[0] - u_lo)
    ext_py = float(v_hi - c_uv[1])
    ext_my = float(c_uv[1] - v_lo)

    cardinals = [((0.0, 1.0), ext_py),    # +y (top)
                 ((0.0, -1.0), ext_my),   # -y (bottom)
                 ((1.0, 0.0), ext_px),    # +x (right)
                 ((-1.0, 0.0), ext_mx)]   # -x (left)

    uv = [c_uv]
    n_cardinal = min(max(n - 1, 0), 4)
    for (du, dv), ext in cardinals[:n_cardinal]:
        uv.append(c_uv + 0.95 * ext * np.array([du, dv]))

    extra = n - 1 - n_cardinal
    if extra > 0:
        r_fill = 0.95 * min(ext_px, ext_mx, ext_py, ext_my)
        theta = 2.0 * np.pi * np.arange(extra) / extra + np.pi / 4.0
        ring = c_uv + r_fill * np.stack(
            [np.cos(theta), np.sin(theta)], axis=-1)
        uv.append(ring)
    return np.concatenate([np.atleast_2d(p) for p in uv], axis=0)


def _pattern_uv_points(kind, loop_uv, c_uv, pattern):
    if kind == "rings":
        return _rings_uv(loop_uv, c_uv, pattern)
    if kind == "fan":
        return _fan_uv(loop_uv, c_uv, pattern)
    raise ValueError("sample_viz_pattern: unknown pattern kind %r"
                     % (kind,))


def _sphere_pattern_points(surf, loop_xyz, kind, pattern):
    """Pattern points on a spherical cap (divergent-laser emit face):
    generate the 2D pattern in the rim's best-fit plane, then lift each
    point onto the sphere along the plane normal, choosing the
    intersection on the cap side (nearer the cap apex o + R*Ŵ, W = rim
    centroid - sphere center)."""
    c3 = loop_xyz.mean(axis=0)
    M = loop_xyz - c3
    _, _, vt = np.linalg.svd(M, full_matrices=False)
    e1, e2, n_pl = vt[0], vt[1], vt[2]
    loop2 = np.stack([M @ e1, M @ e2], axis=-1)
    uv = _pattern_uv_points(kind, loop2, loop2.mean(axis=0), pattern)
    pts_plane = c3 + uv[:, :1] * e1 + uv[:, 1:] * e2

    centre, radius = surf.c, surf.r
    w = c3 - centre
    wn = np.linalg.norm(w)
    apex = centre + radius * (w / wn if wn > 1e-12 else n_pl)
    d = pts_plane - centre
    b = d @ n_pl
    c = np.sum(d * d, axis=-1) - radius * radius
    disc = b * b - c
    ok = disc >= 0.0
    pts_plane, b, disc = pts_plane[ok], b[ok], disc[ok]
    root = np.sqrt(disc)
    cand1 = pts_plane + (-b + root)[:, None] * n_pl
    cand2 = pts_plane + (-b - root)[:, None] * n_pl
    d1 = np.linalg.norm(cand1 - apex, axis=-1)
    d2 = np.linalg.norm(cand2 - apex, axis=-1)
    return np.where((d1 <= d2)[:, None], cand1, cand2)


def sample_viz_pattern(scene, body, src, source_id, pattern, n_lambda):
    """Deterministic viz-overlay ray positions: one central ray plus either
    concentric rings or a small cardinal fan (pattern from
    common.parse_viz_pattern_spec: {"kind": "rings", "dr_mm", "nper",
    "nrings"} or {"kind": "fan", "n"}).

    VISUAL HELPER ONLY: callers trace the returned batch in a separate
    viz-only pass (throwaway ledger, no detector grids), so these rays
    can never affect flux, detector images, or the energy audit.

    Planar emit faces get the pattern directly in their metric uv space;
    SPHERICAL caps (divergent lasers) get it in the rim's best-fit plane
    lifted onto the cap, with per-point normal directions (a diverging
    fan, matching sample_source's curved-face emission). Other surface
    types return None with a warning so the caller falls back to the
    default random viz rays.
    """
    face = scene.emit_faces.get(body.index)
    if face is None:
        raise ValueError("source %s has no emit face built" % body.label)
    surf = face.surface
    cls = surf.__class__.__name__
    if cls not in ("Plane", "Sphere"):
        import warnings
        warnings.warn("source %s: --viz-pattern needs a planar or "
                      "spherical emit face (got %s); falling back to "
                      "default viz rays" % (body.label, cls))
        return None

    if face.trim.mode == "untrimmed" or not getattr(face.trim, "loops", None):
        raise ValueError("source %s: emitting face has no trim loops"
                         % body.label)
    kind = pattern["kind"]

    if cls == "Plane":
        # uv == metres for a Plane: t1/t2 are orthonormal
        loop_uv = np.concatenate([np.asarray(lp) for lp in face.trim.loops])
        c_uv = loop_uv.mean(axis=0)
        uv = _pattern_uv_points(kind, loop_uv, c_uv, pattern)
        inside = face.trim.contains(uv)
        uv = uv[inside]
        if len(uv) == 0:
            raise ValueError("source %s: viz pattern produced no rays "
                             "inside the emit face (pattern too large for "
                             "the aperture?)" % body.label)
        pts = _uv_to_xyz(surf, uv[:, 0], uv[:, 1])
        n = len(pts)
        # same toward-origin direction policy as sample_source's flat branch
        n0 = surf.normal(pts)[0]
        sign = 1.0 if np.dot(n0, -np.mean(pts, axis=0)) >= 0 else -1.0
        dirs = np.tile(sign * n0, (n, 1))
    else:
        loop_uv = [np.asarray(lp) for lp in face.trim.loops]
        loop_xyz = np.concatenate(
            [_uv_to_xyz(surf, lp[:, 0], lp[:, 1]) for lp in loop_uv])
        pts = _sphere_pattern_points(surf, loop_xyz, kind, pattern)
        if len(pts) == 0:
            raise ValueError("source %s: viz pattern produced no rays on "
                             "the emitting cap" % body.label)
        n = len(pts)
        # per-point normals, origin-facing side (sample_source's curved
        # policy: flip wholesale when the natural normal faces away)
        normals = surf.normal(pts)
        dots = np.sum(normals * (-pts), axis=-1)
        if np.all(dots < 0):
            normals = -normals
        dirs = normals
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    lam_strata = wavelength_strata(src, n_lambda)
    n_strata = len(lam_strata)
    # Replicate every pattern point across ALL wavelength strata: a
    # broadband ("white") source then emits a red/green/blue bundle from
    # each fan/ring position, so chromatic behavior (lens dispersion,
    # prism/grating splits) reads directly off the overlay. Monochromatic
    # sources have exactly one stratum, so nothing changes for them.
    # Visual-only rays: the extra count never touches physics/audits.
    if n_strata > 1:
        pts = np.repeat(pts, n_strata, axis=0)
        dirs = np.repeat(dirs, n_strata, axis=0)
    n_total = len(pts)
    pol = src.get("polarization") or {"kind": "unpolarized"}
    power_W = src["power_mW"] * 1e-3
    p_ray = power_W / n_total

    batch = RayBatch(n_total)
    batch.pos[:] = pts
    batch.dir[:] = dirs
    idx = np.arange(n_total)
    batch.lam[:] = lam_strata[idx % n_strata]
    batch.lam_stratum[:] = idx % n_strata
    batch.pol_stratum[:] = 0
    batch.source_id[:] = source_id
    batch.coherent[:] = bool(src.get("coherent", False))
    batch.birth_power[:] = p_ray
    e_ref, _ = _pol_reference_frame(dirs)
    batch.s_hat[:] = e_ref
    js, jp = jones_for(pol, 0)
    amp = np.sqrt(p_ray)
    batch.Es[:] = amp * js
    batch.Ep[:] = amp * jp
    return batch
