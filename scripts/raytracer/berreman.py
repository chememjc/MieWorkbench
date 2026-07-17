# =============================================================================
# berreman.py — full-anisotropy interface/slab optics via the Berreman 4x4
# formalism (Berreman, J. Opt. Soc. Am. 62, 502 (1972)).  Numpy-only,
# complex128, vectorized over ray batches.  This is the P9 "full anisotropy"
# tier that sits BEHIND the P6 exact-uniaxial Lekner path (birefringence.py):
#
#   isotropic                       -> fresnel.py            (cheapest)
#   uniaxial, NON-absorbing         -> birefringence.py Lekner 4x4 (pinned)
#   biaxial / ANY absorbing         -> HERE (Berreman 4x4)
#   gyrotropic (natural activity)   -> HERE
#
# WHY A SECOND 4x4.  The Lekner solve (birefringence.uniaxial_interface_*)
# assembles tangential-E/H continuity for the SPECIAL uniaxial normal surface
# (two closed-form o/e roots, real indices).  Berreman solves the SAME
# boundary-value problem for a GENERAL 3x3 permittivity tensor eps (biaxial,
# complex/absorbing, or gyrotropic-Hermitian) by turning Maxwell's tangential
# equations into a 4x4 first-order system whose eigenvalues are the four
# partial-wave normal indices q_z.  Everything the two share — the field
# conventions, the s/p basis, the flux normalization, the branch discipline —
# is deliberately IDENTICAL to birefringence.py so the two arbitrate each
# other (test_berreman ORACLE 1: uniaxial-through-Berreman == Lekner to 1e-10;
# ORACLE 2: isotropic == fresnel to 1e-10).
#
# =====================  CONVENTIONS (shared with P6)  ========================
#   * d / k_hat point INTO the interface (along propagation).  n_hat is the
#     unit surface normal AGAINST the incident ray: cos_i = -d . n_hat >= 0.
#     zhat = -n_hat points INTO the transmission medium (P6's zhat).
#   * s/p basis is EXACTLY fresnel.pol_basis: s_hat = normalize(d x n_hat)
#     (perpendicular to the plane of incidence); p_hat = d x s_hat.  Returned
#     reflection Jones (r_ss, r_sp, r_ps, r_pp) is in this basis for BOTH the
#     incident and reflected wave and reduces to fresnel rs/rp in the
#     isotropic limit with NO extra sign flips.
#   * every field is a plane wave exp(i(k.r - w t)); in k0 = w/c units
#     (|k| = refractive index).  With H' == Z0 H, Faraday gives H' = q x E and
#     Ampere gives q x H' = -eps E for every mode (q = k/k0).
#
# =====================  THE BERREMAN 4x4  ====================================
# Work in a LOCAL interface frame (xhat in the plane of incidence and
# tangential to the surface, yhat = s_hat perpendicular to that plane, zhat =
# -n_hat into the transmission medium).  A plane wave shares the tangential
# reduced wavevector xi = k_x/k0 = n_incident * sin(theta_i) (k_y = 0 by the
# frame choice).  The state vector is
#         Psi = (Ex, Hy', Ey, -Hx')            (Berreman ordering)
# and d/dz Psi = i k0 Delta Psi, so a partial wave exp(i k0 q_z z) has
# Delta Psi = q_z Psi.  Delta(eps, xi) is built by eliminating Ez, Hz' from
# Maxwell (DERIVED in _delta_matrix's comment; independently re-derived, not
# transcribed).  The four eigenvalues q_z are the partial-wave normal indices;
# the eigenvectors give the full E, H' of each wave.
#
# =====================  DEGENERACY-ROBUST MODE SORTING  ======================
# The merge hinges on this (engine3 Sec 7.4/17: "the mode-sorting is where it
# will break").  Split the four waves into 2 forward (+z: transmitted side)
# and 2 backward (-z: reflected side) by:
#   * propagating wave (|Im q_z| below tol): sign of the NORMAL time-averaged
#     Poynting flux  S_z ~ Re(Psi0 conj(Psi1) + Psi2 conj(Psi3))  (forward =
#     S_z > 0).  NEVER by Re(q_z) ordering (fails for backward-energy modes in
#     strongly anisotropic media).
#   * evanescent / absorbing wave: sign of Im(q_z) (forward = Im > 0, decays
#     into +z).
# We NEVER match partial waves one-to-one across the interface (that is what
# breaks at exceptional points).  Instead each side contributes a 2-D forward
# (or backward) SUBSPACE, and the interface solve works in whatever basis the
# eigensolver returns for that subspace — any independent pair spans it, so
# near-degenerate eigenvectors (repeated real q_z: isotropic, principal-plane
# biaxial, on-axis gyrotropic) are fine as long as the pair is independent.
# The ONE place this genuinely fails is a true exceptional point (defective
# Delta: two eigenvectors coalesce, algebraic mult 2 / geometric mult 1);
# there the forward field matrix is singular.  partial_waves flags it
# (defective) and interface_solve falls back to a least-squares solve and sets
# `defective` so callers can report rather than silently return garbage.  The
# oracle set stays away from EPs (engine3: "ships with an absorbing oracle and
# last") — this is an honest guard, not a claim to resolve EPs exactly.
#
# =====================  SLABS / STACKS  ======================================
# A waveplate IS a slab.  Products of transfer matrices OVERFLOW with
# absorbing layers (a decaying partial wave's exp(+|Im q_z| k0 d) blows up);
# engine3 pins the fix: SCATTERING-matrix (S-matrix) recursion, which only
# ever multiplies by exp(-|Im q_z| k0 d) <= 1.  slab_smatrix does the Redheffer
# star recursion over layers.
#
# =====================  GYROTROPY (natural optical activity)  ================
# McClain-Hillman-Chipman JOSA A 10, 2371 (1993): the gyration vector G is
# FROZEN per ray (g = G_vec . k_hat is the local scalar gyration seen along the
# ray).  Reciprocal (natural) activity only this round: the added tensor is
# eps_ij += i * eps_ijk * G_k, i.e. i (G x .) — antisymmetric & imaginary, so
# eps stays HERMITIAN (lossless) and the effect is reciprocal.  Faraday
# (non-reciprocal, magnetized) activity would break the antisymmetry and is a
# documented seam, NOT implemented here.  ORACLE 3: alpha-quartz rotatory
# power 21.77 deg/mm @ 589.3 nm.
# =============================================================================
import numpy as np

from . import fresnel  # noqa: E402  (isotropic limit / s-p basis share)

_DEGEN = 1e-9            # |Im q_z| below this = "propagating" (sort by S_z)
_EP_TOL = 1e-7          # forward/backward subspace conditioning gate for EPs


# ---------------------------------------------------------------------------
# small vectorization helpers (mirror birefringence.py)
# ---------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _bcast_vec(v, n):
    v = np.asarray(v)
    if v.ndim == 1:
        v = np.broadcast_to(v, (n, v.shape[-1]))
    return np.ascontiguousarray(v)


def _bcast_scalar(x, n):
    x = np.asarray(x)
    if x.ndim == 0:
        x = np.broadcast_to(x, (n,))
    return np.ascontiguousarray(x)


def _dot(a, b):
    return np.sum(a * b, axis=-1)


# ---------------------------------------------------------------------------
# permittivity tensor construction
# ---------------------------------------------------------------------------
def eps_tensor(principal_eps, frame):
    """Global-frame 3x3 permittivity tensor(s) from principal permittivities
    and a crystal frame.

    principal_eps : (3,) or (n,3) COMPLEX principal values (n_i^2; imaginary
                    parts = absorption/dichroism).  Order matches `frame`'s
                    rows.
    frame         : (3,3) or (n,3,3), rows = principal axes in GLOBAL coords
                    (v_crystal = frame @ v_global), the same convention as
                    birefringence._bcast_frame.

    Returns (n,3,3) complex eps in GLOBAL coords: eps = frame^T diag(p) frame.
    (Rows-are-principal-axes => R = frame maps global->crystal, so the global
    tensor is R^T D R.)  Reduces to p * I for isotropic principal values.
    """
    p = np.asarray(principal_eps)
    if p.ndim == 1:
        n = 1 if np.asarray(frame).ndim == 2 else np.asarray(frame).shape[0]
        p = np.broadcast_to(p, (n, 3))
    n = p.shape[0]
    R = np.asarray(frame)
    if R.ndim == 2:
        R = np.broadcast_to(R, (n, 3, 3))
    p = p.astype(np.complex128)
    R = R.astype(np.complex128)
    # eps_global = R^T diag(p) R  ->  einsum: sum_k R[k,i] p_k R[k,j]
    return np.einsum("nki,nk,nkj->nij", R, p, R)


def add_gyration(eps, G_vec):
    """Add reciprocal (natural) optical activity to a permittivity tensor.

    eps   : (n,3,3) complex (global frame).
    G_vec : (3,) or (n,3) real gyration VECTOR in global coords.  The added
            term is eps_ij += i * eps_ijk * G_k = i (G x .), keeping eps
            Hermitian (lossless, reciprocal).  For a ray, G_vec is the frozen
            g = G * k_hat (McClain 1993) supplied by the caller.

    Returns a NEW (n,3,3) complex array (input untouched).
    """
    eps = np.array(eps, dtype=np.complex128, copy=True)
    n = eps.shape[0]
    G = _bcast_vec(np.asarray(G_vec, dtype=np.float64), n)
    gx, gy, gz = G[:, 0], G[:, 1], G[:, 2]
    # antisymmetric i*(G x .):  A_ij = i eps_ijk G_k
    # A = [[0, -i gz, i gy],[i gz, 0, -i gx],[-i gy, i gx, 0]]
    eps[:, 0, 1] += -1j * gz
    eps[:, 0, 2] += 1j * gy
    eps[:, 1, 0] += 1j * gz
    eps[:, 1, 2] += -1j * gx
    eps[:, 2, 0] += -1j * gy
    eps[:, 2, 1] += 1j * gx
    return eps


# ---------------------------------------------------------------------------
# local interface frame
# ---------------------------------------------------------------------------
def local_frame(d, n_hat):
    """Right-handed interface frame (xhat, yhat, zhat).

    zhat = -n_hat (into the transmission medium); yhat = fresnel s_hat
    (perpendicular to the plane of incidence); xhat = yhat x zhat (in the
    plane of incidence, tangential to the surface, with d.xhat >= 0).
    Returns (xhat, yhat, zhat) each (n,3), and s_hat/p_hat (== fresnel's).
    """
    d = _unit(d)
    n = d.shape[0]
    nh = _unit(_bcast_vec(n_hat, n))
    s_hat, p_hat = fresnel.pol_basis(d, nh)
    zhat = -nh
    yhat = s_hat
    xhat = np.cross(yhat, zhat)
    xhat = xhat / np.linalg.norm(xhat, axis=-1, keepdims=True)
    return xhat, yhat, zhat, s_hat, p_hat


def eps_to_local(eps, xhat, yhat, zhat):
    """Rotate a global-frame eps (n,3,3) into the local interface frame whose
    axis rows are xhat, yhat, zhat: eps_local = L eps L^T with L rows =
    (xhat, yhat, zhat)."""
    L = np.stack([xhat, yhat, zhat], axis=1).astype(np.complex128)  # (n,3,3)
    return np.einsum("nik,nkl,njl->nij", L, eps.astype(np.complex128), L)


# ---------------------------------------------------------------------------
# the Berreman Delta matrix and partial-wave solve
# ---------------------------------------------------------------------------
def _delta_matrix(epsL, xi):
    """Berreman Delta (n,4,4) for local-frame permittivity epsL and reduced
    tangential wavevector xi (k_x/k0, k_y = 0).

    DERIVED (independently, not transcribed).  Maxwell in q = k/k0 units with
    q = (xi, 0, q_z), H' = Z0 H:
        q x E = H'                 (Faraday)
        q x H' = -eps E            (Ampere)
    Eliminate Ez = -(xi Hy' + e_zx Ex + e_zy Ey)/e_zz  and  Hz' = xi Ey, then
    read q_z * (Ex, Hy', Ey, -Hx') as linear combinations of the state:
        row Ex:  q_z Ex = (1 - xi^2/e_zz) Hy' - (xi e_zx/e_zz) Ex
                          - (xi e_zy/e_zz) Ey
        row Hy': q_z Hy'= (e_xx - e_xz e_zx/e_zz) Ex - (xi e_xz/e_zz) Hy'
                          + (e_xy - e_xz e_zy/e_zz) Ey
        row Ey:  q_z Ey = (-Hx')
        row -Hx':q_z(-Hx')=(e_yx - e_yz e_zx/e_zz) Ex - (xi e_yz/e_zz) Hy'
                          + (e_yy - e_yz e_zy/e_zz - xi^2) Ey
    """
    n = epsL.shape[0]
    e = epsL
    exx, exy, exz = e[:, 0, 0], e[:, 0, 1], e[:, 0, 2]
    eyx, eyy, eyz = e[:, 1, 0], e[:, 1, 1], e[:, 1, 2]
    ezx, ezy, ezz = e[:, 2, 0], e[:, 2, 1], e[:, 2, 2]
    xi = _bcast_scalar(np.asarray(xi, dtype=np.complex128), n)
    D = np.zeros((n, 4, 4), dtype=np.complex128)
    # row 0 (Ex)
    D[:, 0, 0] = -xi * ezx / ezz
    D[:, 0, 1] = 1.0 - xi ** 2 / ezz
    D[:, 0, 2] = -xi * ezy / ezz
    # row 1 (Hy')
    D[:, 1, 0] = exx - exz * ezx / ezz
    D[:, 1, 1] = -xi * exz / ezz
    D[:, 1, 2] = exy - exz * ezy / ezz
    # row 2 (Ey)
    D[:, 2, 3] = 1.0
    # row 3 (-Hx')
    D[:, 3, 0] = eyx - eyz * ezx / ezz
    D[:, 3, 1] = -xi * eyz / ezz
    D[:, 3, 2] = eyy - eyz * ezy / ezz - xi ** 2
    return D


def _fields_from_state(Psi, epsL, xi, qz):
    """Reconstruct full local-frame E (n,4,3) and H' (n,4,3) for a stack of
    partial waves.  Psi (n,4,K): state vectors as COLUMNS (Psi[:, :, m] is the
    m-th wave's (Ex,Hy',Ey,-Hx')).  qz (n,K) the eigenvalues.  epsL (n,3,3),
    xi (n,)."""
    n = Psi.shape[0]
    Ex = Psi[:, 0, :]
    Hy = Psi[:, 1, :]
    Ey = Psi[:, 2, :]
    Hx = -Psi[:, 3, :]
    ezx = epsL[:, 2, 0][:, None]
    ezy = epsL[:, 2, 1][:, None]
    ezz = epsL[:, 2, 2][:, None]
    xic = _bcast_scalar(np.asarray(xi, dtype=np.complex128), n)[:, None]
    Ez = -(xic * Hy + ezx * Ex + ezy * Ey) / ezz
    Hz = xic * Ey
    E = np.stack([Ex, Ey, Ez], axis=-1)          # (n,K,3)
    H = np.stack([Hx, Hy, Hz], axis=-1)          # (n,K,3)
    return E, H


def _state_flux_z(Psi):
    """Normal time-averaged Poynting flux S_z (x2; sign is what matters) of
    each partial wave from its state vector: S_z ~ Re(Ex Hy'* - Ey Hx'*) =
    Re(Psi0 conj(Psi1) + Psi2 conj(Psi3)).  Psi (n,4,K) columns per wave."""
    return np.real(Psi[:, 0, :] * np.conj(Psi[:, 1, :])
                   + Psi[:, 2, :] * np.conj(Psi[:, 3, :]))


def partial_waves(epsL, xi):
    """Eigensplit the Berreman Delta into forward (+z) and backward (-z)
    partial waves with degeneracy-robust sorting.

    Returns dict:
      qz        (n,4)   eigenvalues, columns re-ordered [f0,f1,b0,b1]
      Psi       (n,4,4) state vectors as COLUMNS, same ordering
      Sz        (n,4)   normal flux of each (ordered)
      defective (n,)    bool: forward or backward 2x2 field block near-singular
                        (exceptional point) — callers should report.
    """
    D = _delta_matrix(epsL, xi)
    n = D.shape[0]
    w, V = np.linalg.eig(D)                       # w (n,4), V (n,4,4) columns
    Sz = _state_flux_z(V)
    imq = np.imag(w)
    prop = np.abs(imq) < _DEGEN
    # forward: propagating -> S_z > 0 ; evanescent -> Im q_z > 0
    forward = np.where(prop, Sz > 0.0, imq > 0.0)

    # sort each row so the two forward waves come first.  Exactly two of each
    # is the generic case; guard pathological counts by falling back to the
    # (Im q_z, then S_z) ranking so we always return a 2/2 split.
    order = np.zeros((n, 4), dtype=np.intp)
    for i in range(n):
        fwd_idx = np.where(forward[i])[0]
        bwd_idx = np.where(~forward[i])[0]
        if len(fwd_idx) != 2:
            key = imq[i] + 1e-12 * Sz[i]
            srt = np.argsort(-key)
            fwd_idx, bwd_idx = srt[:2], srt[2:]
        order[i] = np.concatenate([fwd_idx, bwd_idx])
    idx = np.arange(n)
    qz = w[idx[:, None], order]
    Psi = V[idx[:, None, None], np.arange(4)[None, :, None], order[:, None, :]]
    Sz = Sz[idx[:, None], order]

    # exceptional-point guard: forward (and backward) 2-vectors in the
    # (Ex,Ey) tangential-E plane must be independent.  Cheap proxy = the 2x2
    # tangential-E Gram determinant relative to its norm.
    defective = np.zeros(n, dtype=bool)
    for sl in (slice(0, 2), slice(2, 4)):
        A = Psi[:, [0, 2], sl]                    # (n,2,2) tangential-E cols
        det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        scale = (np.linalg.norm(A[:, :, 0], axis=1)
                 * np.linalg.norm(A[:, :, 1], axis=1))
        defective |= np.abs(det) < _EP_TOL * np.maximum(scale, 1e-300)
    return {"qz": qz, "Psi": Psi, "Sz": Sz, "defective": defective}


# ---------------------------------------------------------------------------
# isotropic analytic partial waves (deterministic s/p, avoids eig's arbitrary
# degenerate basis so reflection Jones read out cleanly)
# ---------------------------------------------------------------------------
def _iso_states(n_med, xi):
    """The isotropic partial-wave states in a deterministic s/p basis.

    Returns (Phi, qz) with Phi (n,4,4) columns [forward-s, forward-p,
    backward-s, backward-p] and qz (n,4) the matching eigenvalues
    (+qz,+qz,-qz,-qz).  E_s along +yhat; E_p in the x-z plane.  Deterministic,
    so callers read r_ss / r_pp with no basis ambiguity."""
    n = xi.shape[0]
    nm = np.asarray(n_med, dtype=np.complex128)
    qz = np.sqrt(nm ** 2 - xi ** 2)               # forward normal index (+root)
    z = np.zeros(n, dtype=np.complex128)
    o = np.ones(n, dtype=np.complex128)
    # The p-mode E-fields are aligned with fresnel.pol_basis's p convention so
    # the returned reflection Jones equals fresnel rs/rp (isotropic) and the
    # Lekner amplitudes (uniaxial) with NO extra sign flips.  In the local
    # frame s_hat=yhat=(0,1,0): incident p_i_local = normalize(d_local x s_hat)
    # = (-qz,0,xi)/nm, reflected p_r_local = (qz,0,xi)/nm.
    # forward s: E=(0,1,0), q=(xi,0,qz) -> H'=(-qz,0,xi): Hy'=0,-Hx'=qz
    Psi_fs = np.stack([z, z, o, qz], axis=1)
    # forward p: E=(-qz,0,xi)/nm  ==  fresnel p_i -> H'=(0,-nm,0)
    Psi_fp = np.stack([-qz / nm, -nm, z, z], axis=1)
    # backward s: q=(xi,0,-qz), E=(0,1,0) -> H'=(qz,0,xi): -Hx'=-qz
    Psi_bs = np.stack([z, z, o, -qz], axis=1)
    # backward p: E=(qz,0,xi)/nm  ==  fresnel p_r (reflect_dir) -> H'=(0,-nm,0)
    Psi_bp = np.stack([qz / nm, -nm, z, z], axis=1)
    Phi = np.stack([Psi_fs, Psi_fp, Psi_bs, Psi_bp], axis=2)
    return Phi, np.stack([qz, qz, -qz, -qz], axis=1)


# ---------------------------------------------------------------------------
# single-interface amplitude solve
# ---------------------------------------------------------------------------
def interface_solve(epsL_a, epsL_b, xi, n_a_iso=None, n_b_iso=None):
    """General single-interface 4x4 boundary-value solve in the local frame.

    epsL_a, epsL_b : (n,3,3) complex local-frame permittivities (incidence /
                     transmission media).  Ignored on a side given as iso.
    xi             : (n,) reduced tangential wavevector.
    n_a_iso        : if not None, (n,) — incidence medium ISOTROPIC with this
                     index; its states are the deterministic s/p pair so
                     reflection Jones read out directly.
    n_b_iso        : likewise for the transmission medium.

    Returns dict (see anis_interface_in / _out for the consumer contract):
      Phi_a, Phi_b : (n,4,4) field matrices, columns [f0,f1,b0,b1]
      qz_a, qz_b   : (n,4)
      r            : (n,2,2) reflected-mode amplitudes; r[:,j,k] = amplitude of
                     backward mode j in medium a for unit forward mode k
      t            : (n,2,2) transmitted-mode amplitudes (forward mode j in b)
      Sz_a, Sz_b   : (n,4) normal fluxes (ordered)
      defective    : (n,) bool
    """
    if n_a_iso is not None:
        Phi_a, qz_a = _iso_states(n_a_iso, xi)
        Sz_a = _state_flux_z(Phi_a)
        def_a = np.zeros(Phi_a.shape[0], dtype=bool)
    else:
        pw_a = partial_waves(epsL_a, xi)
        Phi_a, qz_a, Sz_a, def_a = (pw_a["Psi"], pw_a["qz"], pw_a["Sz"],
                                    pw_a["defective"])
    if n_b_iso is not None:
        Phi_b, qz_b = _iso_states(n_b_iso, xi)
        Sz_b = _state_flux_z(Phi_b)
        def_b = np.zeros(Phi_b.shape[0], dtype=bool)
    else:
        pw_b = partial_waves(epsL_b, xi)
        Phi_b, qz_b, Sz_b, def_b = (pw_b["Psi"], pw_b["qz"], pw_b["Sz"],
                                    pw_b["defective"])
    n = Phi_a.shape[0]
    defective = def_a | def_b

    # continuity Phi_a c_a = Phi_b c_b, c_a=(i0,i1,r0,r1), c_b=(t0,t1,0,0).
    # unknown u = (r0,r1,t0,t1); system  M u = Phi_a[:,forward] @ (i0,i1).
    #   M = [ -Phi_a[:,back] | Phi_b[:,forward] ]   (n,4,4)
    M = np.concatenate([-Phi_a[:, :, 2:4], Phi_b[:, :, 0:2]], axis=2)
    rhs = Phi_a[:, :, 0:2]                                     # (n,4,2) two inc
    if np.any(defective):
        u = np.empty((n, 4, 2), dtype=np.complex128)
        good = ~defective
        if np.any(good):
            u[good] = np.linalg.solve(M[good], rhs[good])
        for i in np.where(defective)[0]:
            u[i] = np.linalg.lstsq(M[i], rhs[i], rcond=None)[0]
    else:
        u = np.linalg.solve(M, rhs)
    r = u[:, 0:2, :]                                          # (n,2,2)
    t = u[:, 2:4, :]
    return {"Phi_a": Phi_a, "Phi_b": Phi_b, "qz_a": qz_a, "qz_b": qz_b,
            "r": r, "t": t, "Sz_a": Sz_a, "Sz_b": Sz_b,
            "defective": defective}


# ---------------------------------------------------------------------------
# high-level interface: isotropic -> anisotropic (ENTRY)
# ---------------------------------------------------------------------------
def _epsG_bcast(epsG, n):
    epsG = np.asarray(epsG, dtype=np.complex128)
    if epsG.ndim == 2:
        epsG = np.broadcast_to(epsG, (n, 3, 3))
    return np.ascontiguousarray(epsG)


def anis_interface_in(d_in, n_hat, epsG, n1):
    """EXACT Berreman amplitudes for an ISOTROPIC (n1) -> anisotropic (epsG)
    interface.  epsG (3,3) or (n,3,3) GLOBAL-frame permittivity (biaxial /
    absorbing / gyrotropic).

    Returns a dict of per-ray arrays (mirrors birefringence.uniaxial_
    interface_in's s/p contract):
      rss,rsp,rps,rpp : complex reflection Jones in fresnel's pol_basis(d)
                        s/p basis, flux-normalized (|.|^2 = reflected power
                        fraction per unit input).
      t : (n,2,2) complex flux-normalized transmission amplitudes; t[:,m,k] =
          coupling of incident s(k=0)/p(k=1) into crystal forward mode m.
      Ts, Tp : (n,) transmitted power fraction for unit s / p input.
      Rs, Rp : (n,) reflected power fraction for unit s / p input.
      A : (n,) absorbed fraction for an s-input reference (1 - Rs - Ts).
      qz_t : (n,2) transmitted-mode normal indices q_z (for sheet matching).
      Et, Ht : (n,2,3) transmitted-mode E / H' in the LOCAL frame.
      Dt : (n,2,3) transmitted-mode D = eps E in the GLOBAL frame (unit-ish).
      xhat,yhat,zhat, s_new,p_new : local / s-p basis vectors (n,3).
      defective : (n,) bool exceptional-point flag.
    """
    d = _unit(d_in)
    n = d.shape[0]
    nh = _unit(_bcast_vec(n_hat, n))
    n1 = _bcast_scalar(np.asarray(n1, dtype=np.complex128), n)
    xhat, yhat, zhat, s_new, p_new = local_frame(d, nh)
    epsG = _epsG_bcast(epsG, n)
    epsL_b = eps_to_local(epsG, xhat, yhat, zhat)
    cos_i = np.real(-_dot(d, nh))
    xi = np.real(n1) * np.sqrt(np.clip(1.0 - cos_i ** 2, 0.0, None))

    sol = interface_solve(None, epsL_b, xi, n_a_iso=np.real(n1))
    r, t = sol["r"], sol["t"]
    Phi_a, Phi_b = sol["Phi_a"], sol["Phi_b"]

    Sz_inc = np.maximum(np.abs(_state_flux_z(Phi_a[:, :, 0:2])), 1e-300)  # (n,2)
    Sz_ref = np.abs(_state_flux_z(Phi_a[:, :, 2:4]))                       # (n,2)
    Sz_t = np.abs(_state_flux_z(Phi_b[:, :, 0:2]))                         # (n,2)

    # flux-normalize: r[:,j,k] reflected mode j (flux Sz_ref_j) per input k
    # (flux Sz_inc_k); |amp|^2 = power fraction.
    rN = r * np.sqrt(Sz_ref[:, :, None]) / np.sqrt(Sz_inc[:, None, :])
    tN = t * np.sqrt(Sz_t[:, :, None]) / np.sqrt(Sz_inc[:, None, :])
    rss, rps = rN[:, 0, 0], rN[:, 1, 0]           # s-input -> s / p reflected
    rsp, rpp = rN[:, 0, 1], rN[:, 1, 1]           # p-input -> s / p reflected
    Ts = np.sum(np.abs(tN[:, :, 0]) ** 2, axis=1)
    Tp = np.sum(np.abs(tN[:, :, 1]) ** 2, axis=1)
    Rs = np.abs(rss) ** 2 + np.abs(rps) ** 2
    Rp = np.abs(rsp) ** 2 + np.abs(rpp) ** 2
    A = 1.0 - Rs - Ts

    E_t, H_t = _fields_from_state(Phi_b[:, :, 0:2], epsL_b, xi,
                                  sol["qz_b"][:, 0:2])   # (n,2,3) local
    # D = eps E (local), then to global for the tracer's D-basis bookkeeping
    D_t_local = np.einsum("nij,nmj->nmi", epsL_b, E_t)
    L = np.stack([xhat, yhat, zhat], axis=1)             # rows local axes
    D_t = np.einsum("nji,nmj->nmi", L.astype(np.complex128), D_t_local)
    return {
        "rss": rss, "rsp": rsp, "rps": rps, "rpp": rpp,
        "t": tN, "Ts": Ts, "Tp": Tp, "Rs": Rs, "Rp": Rp, "A": A,
        "qz_t": sol["qz_b"][:, 0:2], "Et": E_t, "Ht": H_t, "Dt": D_t,
        "Sz_t": Sz_t, "xhat": xhat, "yhat": yhat, "zhat": zhat,
        "s_new": s_new, "p_new": p_new, "defective": sol["defective"],
    }


# ---------------------------------------------------------------------------
# slab S-matrix (Redheffer star recursion; absorbing-stable)
# ---------------------------------------------------------------------------
def _star(SA, SB):
    """Redheffer star product of two S-matrices, each a 4-tuple of 2x2 blocks
    (Tuu, Rdu, Rud, Tdd) with u=up/backward, d=down/forward, all (n,2,2)."""
    Tuu_A, Rdu_A, Rud_A, Tdd_A = SA
    Tuu_B, Rdu_B, Rud_B, Tdd_B = SB
    n = Tdd_A.shape[0]
    Ii = np.broadcast_to(np.eye(2, dtype=np.complex128), (n, 2, 2))
    M1 = np.linalg.inv(Ii - Rud_A @ Rdu_B)
    M2 = np.linalg.inv(Ii - Rdu_B @ Rud_A)
    Tdd = Tdd_B @ M1 @ Tdd_A
    Rdu = Rdu_A + Tuu_A @ Rdu_B @ M1 @ Tdd_A
    Rud = Rud_B + Tdd_B @ M1 @ Rud_A @ Tuu_B
    Tuu = Tuu_A @ M2 @ Tuu_B
    return (Tuu, Rdu, Rud, Tdd)


def slab_smatrix(eps_layers, thick_layers, lam, d_in, n_hat, n_in, n_out):
    """Reflection / transmission of a stack of anisotropic layers between two
    isotropic half-spaces, via S-matrix recursion (never transfer-matrix
    products — absorbing-stable, engine3 Sec 7.4).

    eps_layers   : list of (3,3) or (n,3,3) GLOBAL-frame permittivities.
    thick_layers : list of scalars or (n,) layer thicknesses [m].
    lam          : (n,) wavelength [m].
    d_in, n_hat  : (n,3) incident direction / entry-face normal (P6 convention).
    n_in, n_out  : (n,) isotropic indices of the bounding half-spaces.

    Returns dict: rss,rsp,rps,rpp (reflection Jones, s/p basis),
      tss,tsp,tps,tpp (transmission Jones), R (n,), T (n,), A (n,),
      s_new,p_new (n,3).  Amplitudes flux-normalized (|.|^2 = power fraction),
      so R + T + A == 1 with A the absorptive deficit.
    """
    d = _unit(d_in)
    n = d.shape[0]
    nh = _unit(_bcast_vec(n_hat, n))
    lam = _bcast_scalar(np.asarray(lam, dtype=np.float64), n)
    n_in = _bcast_scalar(np.asarray(n_in, dtype=np.complex128), n)
    n_out = _bcast_scalar(np.asarray(n_out, dtype=np.complex128), n)
    xhat, yhat, zhat, s_new, p_new = local_frame(d, nh)
    cos_i = np.real(-_dot(d, nh))
    xi = np.real(n_in) * np.sqrt(np.clip(1.0 - cos_i ** 2, 0.0, None))
    k0 = 2.0 * np.pi / lam

    def layer_states(epsG):
        epsL = eps_to_local(_epsG_bcast(epsG, n), xhat, yhat, zhat)
        pw = partial_waves(epsL, xi)
        return pw["Psi"], pw["qz"]

    # interface S-matrix from field matrices Phi_L (left) / Phi_R (right).
    # Continuity Phi_L a = Phi_R b, a=(a_f,a_b), b=(b_f,b_b). Scatter:
    #   incoming (a_f, b_b) -> outgoing (b_f, a_b).
    #   Lf a_f + Lb a_b = Rf b_f + Rb b_b
    #   [ -Rf | Lb ] (b_f; a_b) = [ -Lf | Rb ] (a_f; b_b)
    def interface_S(Phi_L, Phi_R):
        Lf, Lb = Phi_L[:, :, 0:2], Phi_L[:, :, 2:4]
        Rf, Rb = Phi_R[:, :, 0:2], Phi_R[:, :, 2:4]
        Mout = np.concatenate([-Rf, Lb], axis=2)
        Min = np.concatenate([-Lf, Rb], axis=2)
        S = np.linalg.solve(Mout, Min)
        return (S[:, 2:4, 2:4], S[:, 2:4, 0:2], S[:, 0:2, 2:4], S[:, 0:2, 0:2])

    def prop_S(qz, thick):
        pf = np.exp(1j * k0 * qz[:, 0:2] * thick[:, None])   # forward +z
        pb = np.exp(-1j * k0 * qz[:, 2:4] * thick[:, None])  # backward -z
        Z = np.zeros((n, 2, 2), dtype=np.complex128)
        Pf = Z.copy(); Pf[:, 0, 0] = pf[:, 0]; Pf[:, 1, 1] = pf[:, 1]
        Pb = Z.copy(); Pb[:, 0, 0] = pb[:, 0]; Pb[:, 1, 1] = pb[:, 1]
        return (Pb, Z.copy(), Z.copy(), Pf)

    Phi_in, _ = _iso_states(n_in, xi)
    S = None
    Phi_prev = Phi_in
    for epsG, thick in zip(eps_layers, thick_layers):
        Phi_l, qz_l = layer_states(epsG)
        Sif = interface_S(Phi_prev, Phi_l)
        S = Sif if S is None else _star(S, Sif)
        th = _bcast_scalar(np.asarray(thick, dtype=np.float64), n)
        S = _star(S, prop_S(qz_l, th))
        Phi_prev = Phi_l
    Phi_out, _ = _iso_states(n_out, xi)
    Sif = interface_S(Phi_prev, Phi_out)
    S = _star(S, Sif)

    Tuu, Rdu, Rud, Tdd = S
    fin = np.abs(_state_flux_z(Phi_in[:, :, 0:2]))       # (n,2) s,p incident
    fout = np.abs(_state_flux_z(Phi_out[:, :, 0:2]))     # (n,2) transmitted
    rss = Rdu[:, 0, 0]; rps = Rdu[:, 1, 0]
    rsp = Rdu[:, 0, 1]; rpp = Rdu[:, 1, 1]
    tN = Tdd * np.sqrt(fout[:, :, None]) / np.sqrt(fin[:, None, :])
    tss = tN[:, 0, 0]; tps = tN[:, 1, 0]
    tsp = tN[:, 0, 1]; tpp = tN[:, 1, 1]
    R = np.abs(rss) ** 2 + np.abs(rps) ** 2
    T = np.abs(tss) ** 2 + np.abs(tps) ** 2
    A = 1.0 - R - T
    return {"rss": rss, "rsp": rsp, "rps": rps, "rpp": rpp,
            "tss": tss, "tsp": tsp, "tps": tps, "tpp": tpp,
            "R": R, "T": T, "A": A, "s_new": s_new, "p_new": p_new}


# ---------------------------------------------------------------------------
# gyration bookkeeping helper
# ---------------------------------------------------------------------------
def gyration_from_rotatory_power(rho_deg_per_mm, lam_m, n_o):
    """Gyration SCALAR G (dimensionless, adds to eps) that reproduces a
    measured specific rotation (rotatory power) rho along the optic axis.

    Derivation (module header + McClain 1993): along the optic axis the two
    circular eigenmodes have n_+^2 = eps_o + G, n_-^2 = eps_o - G, so
    n_+ - n_- = G / n_o.  A linear input rotates by pi (n_+ - n_-) / lam per
    unit length = pi G / (n_o lam).  Hence
        G = n_o * lam * rho / pi          (rho in rad per metre)
    rho supplied in DEG/MM (the registry / literature unit).  Positive G ->
    right-handed rotation with the +i(G x .) convention of add_gyration.
    """
    rho = np.deg2rad(np.asarray(rho_deg_per_mm, dtype=np.float64)) * 1e3  # rad/m
    return n_o * np.asarray(lam_m, dtype=np.float64) * rho / np.pi
