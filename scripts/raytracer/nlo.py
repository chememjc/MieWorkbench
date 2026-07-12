#!/usr/bin/env python3
# =============================================================================
# nlo.py -- chi(2) nonlinear-optics math for the pulsed-optics round.
#
# Numpy-only. Consumes rows from optprops.load_nonlinear (the
# opticalproperties/nonlinear/nonlinear.mienlo registry) plus the uniaxial
# dispersion machinery in birefringence.py. This module is PURE MATH: the
# tracer-side SHG event (phase P7b) calls into it but lives elsewhere.
#
# Conventions (each cross-checked against Boyd, "Nonlinear Optics" 3rd ed.):
#   * d-matrices are the contracted (Voigt) 3x6 form, pm/V in the registry,
#     CRYSTAL principal frame. Contracted index l: 11->1, 22->2, 33->3,
#     23/32->4, 13/31->5, 12/21->6 (Boyd Sec. 1.5).
#   * Field vectors passed to d_eff_tensor are unit E-FIELD directions in
#     the crystal frame (not D-field). For extraordinary waves E tilts off
#     transversality by the walk-off angle rho; passing the transverse
#     D-direction instead perturbs d_eff at the cos(rho) level (<0.5% for
#     rho < 5 deg) -- inside the accuracy of any scalar-d_eff treatment.
#   * SHG collinear phase mismatch: delta_k = k(2w) - 2 k(w).
#   * Conversion efficiency: undepleted plane-wave result, clamped at 0.5
#     where the undepleted approximation has long since died.
# =============================================================================
import numpy as np

from .birefringence import n_e_theta
from .materials import MaterialError

EPS0 = 8.8541878128e-12    # vacuum permittivity [F/m]
C0 = 299792458.0           # speed of light [m/s]

# Voigt contracted index l -> tensor index pair (j, k), 0-based.
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _unit(v, what):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.sqrt(np.dot(v, v)))
    if n < 1e-12:
        raise MaterialError("nlo: %s is a zero vector" % what)
    return v / n


# ---------------------------------------------------------------------------
# d_eff from the full tensor
# ---------------------------------------------------------------------------
def d_eff_tensor(d_il, point_group, k_hat_crystal, e_pump1, e_pump2,
                 e_harmonic):
    """Scalar effective nonlinearity d_eff = e_harm . d : (e_p1 (x) e_p2).

    d_il: (3, 6) contracted d-matrix (registry chi2_tensor row, pm/V; any
    units -- the result carries them through). The pump pair is symmetrized
    into the Voigt vector

        v_l = e1_j e2_k + e1_k e2_j   (j != k)
        v_l = e1_j e2_j               (j == k)

    so degenerate SHG (e_pump1 == e_pump2 == e) reproduces the textbook
    (e_x^2, e_y^2, e_z^2, 2 e_y e_z, 2 e_x e_z, 2 e_x e_y) column (Boyd,
    Nonlinear Optics 3rd ed., Eq. 1.5.27 region), and a type-II distinct
    pump pair counts each mixed product once per ordering.

    All vectors are unit E-FIELD directions in the CRYSTAL principal frame
    (see module header for the E-vs-D walk-off caveat). k_hat_crystal is
    carried for interface/documentation symmetry -- the contraction itself
    never needs it (the field vectors already encode the geometry); it is
    only normalized to catch degenerate input. point_group is provenance
    metadata: the symmetry is already baked into the zeros of d_il.

    Oracle (pinned in tests): for 3m ooe this contraction equals the
    closed-form d_eff = d31 sin(theta) - d22 cos(theta) sin(3 phi) exactly,
    with o = (sin phi, -cos phi, 0) and
    e = (-cos theta cos phi, -cos theta sin phi, sin theta).
    """
    d = np.asarray(d_il, dtype=np.float64)
    if d.shape != (3, 6):
        raise MaterialError("nlo: d_il must be a (3, 6) contracted matrix, "
                            "got shape %s" % (d.shape,))
    _unit(k_hat_crystal, "k_hat_crystal")
    e1 = _unit(e_pump1, "e_pump1")
    e2 = _unit(e_pump2, "e_pump2")
    eh = _unit(e_harmonic, "e_harmonic")
    v = np.empty(6, dtype=np.float64)
    for l, (j, k) in enumerate(_VOIGT_PAIRS):
        v[l] = e1[j] * e2[k] + (e1[k] * e2[j] if j != k else 0.0)
    return float(eh @ (d @ v))


# ---------------------------------------------------------------------------
# phase mismatch + phase-matching solve
# ---------------------------------------------------------------------------
def delta_k(n1, n2, lam1_m):
    """SHG collinear phase mismatch [1/m], written explicitly as
    delta_k = k(2w) - 2 k(w) with k = 2 pi n / lam:

        k(w)  = 2 pi n1 / lam1
        k(2w) = 2 pi n2 / (lam1 / 2) = 4 pi n2 / lam1
        delta_k = (4 pi / lam1) (n2 - n1)

    n1 = index at the fundamental, n2 = index at the harmonic. The sign
    only matters to callers via sinc^2 (even), but the k2 - 2k1 orientation
    is the one Boyd Eq. 2.2.13 uses.
    """
    lam1_m = np.asarray(lam1_m, dtype=np.float64)
    k1 = 2.0 * np.pi * np.asarray(n1, dtype=np.float64) / lam1_m
    k2 = 4.0 * np.pi * np.asarray(n2, dtype=np.float64) / lam1_m
    out = k2 - 2.0 * k1
    return float(out) if out.ndim == 0 else out


def _resolve_uniaxial_entry(crystal_entry, matdb_or_props):
    """Accept a crystal name (looked up in an OpticalProperties, a
    MaterialDB with attached uniaxial registry, or a plain
    {name: {"o","e"}} dict) or a direct uniaxial entry dict; return the
    {"o": Material, "e": Material} entry."""
    if isinstance(crystal_entry, dict) and "o" in crystal_entry \
            and "e" in crystal_entry:
        return crystal_entry
    if not isinstance(crystal_entry, str):
        raise MaterialError(
            "nlo.phase_match_angle: crystal_entry must be a uniaxial "
            "registry entry dict or a crystal name (got %r)"
            % (type(crystal_entry).__name__,))
    src = matdb_or_props
    uni = getattr(src, "uniaxial", None)          # OpticalProperties
    if uni is None and hasattr(src, "is_birefringent"):   # MaterialDB
        if src.is_birefringent(crystal_entry):
            mat_o, mat_e = src.get_uniaxial(crystal_entry)
            return {"o": mat_o, "e": mat_e}
        uni = {}
    if uni is None and isinstance(src, dict):
        uni = src
    if uni is None:
        raise MaterialError(
            "nlo.phase_match_angle: cannot resolve crystal %r -- pass an "
            "OpticalProperties, a MaterialDB with attach_uniaxial, or the "
            "uniaxial registry dict" % crystal_entry)
    for key in (crystal_entry, crystal_entry.strip().lower()):
        if key in uni:
            return uni[key]
    raise MaterialError(
        "nlo.phase_match_angle: crystal %r is not in the uniaxial registry"
        % crystal_entry)


def phase_match_angle(crystal_entry, matdb_or_props, lam_pump_m, process):
    """Collinear SHG phase-matching angle.

    process='shg_type1' (ooe, NEGATIVE uniaxial): solves
    n_e(2w, theta) = n_o(w) by bisection on theta in (0, 90] deg using
    birefringence.n_e_theta and the crystal's o/e Material dispersion.
    crystal_entry: a uniaxial registry entry ({"o","e"} Materials) or a
    crystal name resolved through matdb_or_props (OpticalProperties /
    MaterialDB with attached uniaxial / plain registry dict).

    process='shg_type2': the general two-branch average-index solve is OUT
    OF SCOPE this phase -- pass a kind='chi2_process' registry row as
    crystal_entry and its pre-solved cut angles are returned with
    source='registry' (matdb_or_props may be None then).

    Returns {"theta_deg", "phi_deg" (None when the solve is
    phi-degenerate), "source" ('solved'|'registry'), "residual_dn"
    (n_e(2w, theta) - n_o(w) at the returned angle; None for registry
    rows)}. Raises MaterialError when the crystal cannot phase-match
    (positive uniaxial, or birefringence too weak to defeat dispersion).
    """
    if process not in ("shg_type1", "shg_type2"):
        raise MaterialError("nlo.phase_match_angle: unknown process %r"
                            % (process,))
    if process == "shg_type2":
        if isinstance(crystal_entry, dict) \
                and crystal_entry.get("kind") == "chi2_process":
            return {"theta_deg": float(crystal_entry["theta_deg"]),
                    "phi_deg": float(crystal_entry["phi_deg"]),
                    "source": "registry", "residual_dn": None}
        raise MaterialError(
            "nlo.phase_match_angle: the general type-II angle solve is not "
            "implemented (P7a ships type-I only) -- pass a chi2_process "
            "registry row to use its pre-solved cut angles")
    entry = _resolve_uniaxial_entry(crystal_entry, matdb_or_props)
    lam1 = float(lam_pump_m)
    lam2 = lam1 / 2.0
    n_o1 = float(np.real(entry["o"].n_complex(lam1)))
    n_o2 = float(np.real(entry["o"].n_complex(lam2)))
    n_e2 = float(np.real(entry["e"].n_complex(lam2)))
    if n_e2 >= n_o2:
        raise MaterialError(
            "nlo.phase_match_angle: type-I ooe needs a NEGATIVE uniaxial "
            "crystal (n_e(2w)=%.4f >= n_o(2w)=%.4f)" % (n_e2, n_o2))

    def f(theta):
        return float(n_e_theta(np.cos(theta), n_o2, n_e2)) - n_o1

    lo, hi = 0.0, 0.5 * np.pi
    f_lo, f_hi = f(lo), f(hi)   # f(0) = n_o2 - n_o1, f(90) = n_e2 - n_o1
    if not (f_lo > 0.0 and f_hi < 0.0):
        raise MaterialError(
            "nlo.phase_match_angle: no type-I phase-matching angle at "
            "lam=%.1f nm (n_o(w)=%.4f, n_o(2w)=%.4f, n_e(2w)=%.4f -- "
            "birefringence cannot bridge the dispersion)"
            % (lam1 * 1e9, n_o1, n_o2, n_e2))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    theta = 0.5 * (lo + hi)
    return {"theta_deg": float(np.degrees(theta)), "phi_deg": None,
            "source": "solved", "residual_dn": f(theta)}


# ---------------------------------------------------------------------------
# conversion efficiency
# ---------------------------------------------------------------------------
def sinc2(x):
    """sin^2(x)/x^2 with the exact limit 1 at x = 0. Vectorized. Built on
    np.sinc (= sin(pi t)/(pi t)), which handles the origin without a
    0/0 -- no epsilon fudging."""
    x = np.asarray(x, dtype=np.float64)
    out = np.sinc(x / np.pi) ** 2
    return float(out) if out.ndim == 0 else out


ETA_CLAMP = 0.5


def shg_efficiency(d_eff, L_m, I_W_m2, n1, n2, lam1_m, delta_k):
    """Undepleted plane-wave SHG conversion efficiency.

        eta = 8 pi^2 d_eff^2 L^2 I
              -----------------------  * sinc^2(delta_k L / 2)
              n1^2 n2 eps0 c lam1^2

    Source: Boyd, "Nonlinear Optics" 3rd ed., Sec. 2.2 -- Eq. 2.2.19 for
    the harmonic amplitude in the undepleted-pump limit; expressing
    eta = I(2w)/I(w) with I = 2 n eps0 c |E|^2 and lam1 the VACUUM pump
    wavelength gives the prefactor above (the same engineering form as
    Sutherland, Handbook of Nonlinear Optics, Eq. 2.100).

    Units are strict SI: d_eff [m/V] (registry values are pm/V -- multiply
    by 1e-12), L_m [m], I_W_m2 [W/m^2] pump intensity, lam1_m [m],
    delta_k [1/m] (from delta_k()).

    Returns (eta, clamped): eta is capped at ETA_CLAMP = 0.5 with
    clamped=True, because the quadratic undepleted growth is unphysical
    well before 50% conversion (pump depletion turns it into a tanh^2;
    Boyd Sec. 2.6) -- the tracer-side event treats a clamped value as
    "strong conversion, not to be trusted quantitatively".
    """
    if L_m < 0 or I_W_m2 < 0:
        raise MaterialError("nlo.shg_efficiency: L_m and I_W_m2 must be "
                            ">= 0")
    eta = (8.0 * np.pi ** 2 * float(d_eff) ** 2 * float(L_m) ** 2
           * float(I_W_m2)
           / (float(n1) ** 2 * float(n2) * EPS0 * C0 * float(lam1_m) ** 2)
           ) * sinc2(0.5 * float(delta_k) * float(L_m))
    if eta > ETA_CLAMP:
        return ETA_CLAMP, True
    return float(eta), False


def local_intensity(power_W, dA_m2, kappa_pulse):
    """Local intensity I = (power / dA) * kappa_pulse [W/m^2].

    Centralizes the ray-power -> intensity convention for P7b/P8:
    power_W is the ray's carried power (a CW-average in this engine),
    dA_m2 the local beam-patch area it represents, and kappa_pulse the
    pulse peak-enhancement factor (1.0 for CW; P_peak/P_avg ~
    1/(f_rep * tau_eff) for a mode-locked train -- computed by the
    time-domain caller, NOT here)."""
    if dA_m2 <= 0:
        raise MaterialError("nlo.local_intensity: dA_m2 must be > 0")
    return float(power_W) / float(dA_m2) * float(kappa_pulse)


# =============================================================================
# Phase P8: Pockels cell / saturable absorber / two-photon absorption /
# Kerr thin lens. Pure math + the shared per-ray intensity estimator; the
# tracer-side hooks (bulk alpha_add, opl advance, uniaxial index lookup)
# live in tracer.py/scene.py and call into this module.
# =============================================================================

# ---------------------------------------------------------------------------
# Pockels cell: index-shifted Material proxy
# ---------------------------------------------------------------------------
class _ShiftedIndex:
    """Material-like proxy: `base`'s n_complex shifted by a real,
    wavelength-dependent Delta_n(lam) (added to the REAL part only).
    Drop-in substitute anywhere a raytracer.materials.Material is
    expected -- n_complex/n_group/dn_dlam/d2n_dlam2/d3n_dlam3, the full
    interface scene.py's medium_index/medium_group_index/
    medium_gdd_per_length/uniaxial_indices consume (via
    Scene.uniaxial_materials) -- so retardance, dispersion and group
    delay all pick up the shift automatically through the EXISTING
    machinery.

    v1 approximation (documented): n(lam) and n_group(lam) = n -
    lam*dn/dlam both get the SAME delta_fn(lam) added directly (i.e.
    d(Delta_n)/dlam is neglected in the group-index formula), and
    d2n_dlam2/d3n_dlam3 pass straight through to `base` unperturbed.
    Delta_n is already a tiny (~1e-4-scale) index correction, so its own
    curvature vs wavelength is a negligible second-order effect on
    GDD/TOD next to the host crystal's own dispersion."""

    def __init__(self, base, delta_fn):
        self._base = base
        self._delta_fn = delta_fn

    def n_complex(self, lam_m):
        return self._base.n_complex(lam_m) + self._delta_fn(lam_m)

    def n_group(self, lam_m):
        return self._base.n_group(lam_m) + self._delta_fn(lam_m)

    def dn_dlam(self, lam_m):
        return self._base.dn_dlam(lam_m)

    def d2n_dlam2(self, lam_m):
        return self._base.d2n_dlam2(lam_m)

    def d3n_dlam3(self, lam_m):
        return self._base.d3n_dlam3(lam_m)


def pockels_delta_n(mat, r_pm_V, e_field_v_per_m):
    """Return a callable lam_m -> Delta_n(lam) = -0.5 * n_base(lam)^3 * r *
    E for ONE electro-optic coefficient (Boyd/Yariv transverse-Pockels
    convention; the absolute sign is not pinned by the registry -- same
    documented caveat as nonlinear.mienlo's d_il_pm_V column -- only the
    sin^2(pi V / 2 Vpi) MAGNITUDE law is asserted by the P8 oracle).
    r_pm_V: electro-optic coefficient [pm/V]. e_field_v_per_m: applied
    field [V/m] (scalar; E = V/d for the transverse geometry)."""
    r_si = float(r_pm_V) * 1e-12          # pm/V -> m/V

    def delta_fn(lam_m):
        n0 = np.real(mat.n_complex(lam_m))
        return -0.5 * n0 ** 3 * r_si * float(e_field_v_per_m)
    return delta_fn


def pockels_shifted_materials(mat_o, mat_e, r_pm_V, gap_m, voltage):
    """Build the (mat_o', mat_e') _ShiftedIndex pair for a TRANSVERSE
    Pockels cell (P8 v1 scope; longitudinal geometries such as KD*P r63
    are rejected earlier, at Scene construction, with a clear "needs a
    later engine phase" error):

        Delta_n_e = -0.5 n_e^3 r33 E
        Delta_n_o = -0.5 n_o^3 r13 E
        E = V / d

    the optic axis (body.crystal_axis) is UNCHANGED -- the induced
    birefringence rides on the EXISTING o/e principal frame, it does not
    rotate it (a longitudinal 45-degree induced-axis cell would need
    that, which is exactly why it is out of v1 scope).

    r_pm_V: {'r33': .., 'r13': ..} (nonlinear.mienlo pockels row,
    geometry='transverse'; both keys required -- checked by the caller).
    gap_m: electrode gap d [m] (pockels_gap body property, mm -> m).
    voltage: applied V [V] (pockels_voltage body property; 0 -> Delta_n
    == 0 everywhere, the cell reads as an ordinary passive crystal).
    Returns (mat_o_shifted, mat_e_shifted)."""
    if gap_m <= 0:
        raise MaterialError(
            "nlo.pockels_shifted_materials: gap_m must be > 0")
    e_field = float(voltage) / float(gap_m)
    mat_o_s = _ShiftedIndex(mat_o,
                            pockels_delta_n(mat_o, r_pm_V["r13"], e_field))
    mat_e_s = _ShiftedIndex(mat_e,
                            pockels_delta_n(mat_e, r_pm_V["r33"], e_field))
    return mat_o_s, mat_e_s


# ---------------------------------------------------------------------------
# Saturable absorber / two-photon absorption: bulk alpha(I) contributions
# ---------------------------------------------------------------------------
def saturable_alpha0_per_m(spec):
    """Unsaturated bulk absorption coefficient alpha0 [1/m] for a
    resolved saturable spec (scene.Body.saturable_spec: {I_sat_W_cm2, T0,
    alpha0_per_mm}), either a nonlinear.mienlo kind=saturable registry row
    or the inline 'sat:I_sat=..:T0=..' spec (common.parse_saturable_value)
    -- both resolve to the same shape.

    Preference order: the explicit alpha0_per_mm column when the registry
    row carries one; otherwise T0 is read as a PER-MILLIMETRE unsaturated
    transmission: alpha0 = -ln(T0) * 1e3 [1/m]. Documented v1
    approximation -- rows authored before alpha0_per_mm existed (e.g. the
    shipped sam_1550_16_2ps SESAM row, whose T0 is really a whole-DEVICE
    mirror reflectance, not a bulk transmission) fall back to this
    per-mm reading, which is only a scale convention until a device
    actually specifies alpha0_per_mm directly."""
    alpha0_per_mm = spec.get("alpha0_per_mm")
    if alpha0_per_mm is not None:
        return float(alpha0_per_mm) * 1e3
    return -np.log(float(spec["T0"])) * 1e3


def saturable_alpha_per_m(spec, I_W_m2):
    """Saturated bulk absorption coefficient alpha(I) = alpha0 / (1 +
    I/I_sat) [1/m]. I_W_m2: per-ray local intensity [W/m^2] (scalar or
    array; NaN/negative treated as 0 by the caller, see tracer.step)."""
    alpha0 = saturable_alpha0_per_m(spec)
    I_sat = float(spec["I_sat_W_cm2"]) * 1e4      # W/cm^2 -> W/m^2
    return alpha0 / (1.0 + np.asarray(I_W_m2, dtype=np.float64) / I_sat)


def tpa_alpha_per_m(beta_cm_GW, I_W_m2):
    """Two-photon-absorption EFFECTIVE linear coefficient alpha_TPA(I) =
    beta_SI * I [1/m], where beta_SI [m/W] = beta_cm_GW * 1e-11 (cm/GW ->
    m/W: 1 cm/GW = 1e-2 m / 1e9 W = 1e-11 m/W).

    This is the exact rewrite of the standard TPA law dI/dz = -beta I^2
    as a Beer-Lambert form dI/dz = -alpha_TPA(I) I with alpha_TPA(I) =
    beta*I -- lets TPA ride the SAME per-segment Beer-Lambert bulk-
    absorption hook as ordinary/filter/saturable absorption (tracer.step
    alpha_add), with the correct intensity-squared physics folded into
    alpha_TPA's own I-dependence."""
    beta_si = float(beta_cm_GW) * 1e-11
    return beta_si * np.asarray(I_W_m2, dtype=np.float64)


# ---------------------------------------------------------------------------
# shared per-ray local-intensity estimator
# ---------------------------------------------------------------------------
def ray_intensity(batch, scene):
    """Per-ray local intensity estimate [W/m^2] for every ray in `batch`
    (a RayBatch, or RayBatch-like selection already narrowed to a shared-
    medium group -- see tracer.step's bulk loop / the Kerr hook).

    Preference order:
      1. Ray differentials (--ray-differentials): per-ray wavefront patch
         area dA from differentials.patch_area(dPdx, dPdy, dir); I =
         power / dA. Accurate transverse profile -- REQUIRED for the Kerr
         thin lens to reproduce a physical beam profile.
      2. Uniform-beam fallback: this ray's SOURCE emitting-face area
         (scene.emit_faces[body_index].area_m2); I = power / area. A
         flat-top approximation (same area for every ray from that
         source regardless of where on the optic it lands) -- adequate
         for a scalar saturable/TPA estimate, NOT for the Kerr transverse
         profile (documented; the Kerr hook still warns+uses it, giving
         an on-axis-only estimate rather than a real lens).
      3. Neither available (no differentials, no resolvable source/area):
         NaN for those rays -- the caller treats NaN as I=0 (saturable ->
         unsaturated alpha0; TPA -> zero; Kerr -> no added phase) and
         warns ONCE.

    kappa_pulse (pulse peak/avg power ratio, per-ray via its SOURCE's
    pulse dict -- see scene._parse_pulse_source) multiplies every
    estimate, same convention as local_intensity().

    Returns (I, warn_reason): I is an (n,) float64 array (n = len(batch)),
    NaN where no estimate was possible; warn_reason is None, or a short
    string the caller should warn about ONCE."""
    n = len(batch)
    I = np.full(n, np.nan, dtype=np.float64)
    kappa = np.ones(n, dtype=np.float64)
    sids = np.unique(batch.source_id)
    for sid in sids:
        sel = batch.source_id == sid
        src = scene.sources[int(sid)][1]
        pulse = src.get("pulse")
        k = pulse.get("kappa") if pulse is not None else None
        kappa[sel] = k if k is not None else 1.0

    if batch.has_differentials:
        from . import differentials as diff_mod
        dA = diff_mod.patch_area(batch.dPdx, batch.dPdy, batch.dir)
        good = np.isfinite(dA) & (dA > 0)
        if np.any(good):
            I[good] = batch.power[good] / dA[good] * kappa[good]

    missing = ~np.isfinite(I)
    if np.any(missing):
        for sid in np.unique(batch.source_id[missing]):
            sel = missing & (batch.source_id == sid)
            body_idx, src = scene.sources[int(sid)]
            face = scene.emit_faces.get(body_idx)
            if face is not None and face.area_m2:
                # flat-top approximation: the SOURCE's total emitted power
                # over its whole aperture area, the SAME scalar for every
                # ray from this source (NOT a per-ray power share divided
                # by the full area -- that would underestimate I by ~N,
                # the ray count).
                total_power_w = src["power_mW"] * 1e-3
                I[sel] = total_power_w / face.area_m2 * kappa[sel]

    warn_reason = None
    if np.any(~np.isfinite(I)):
        warn_reason = ("no ray differentials and no resolvable "
                       "source emit-face area")
    return I, warn_reason
