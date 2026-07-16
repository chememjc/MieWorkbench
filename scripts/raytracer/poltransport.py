# =============================================================================
# poltransport.py -- P2: parallel-transport Q matrix + honest retardance/
# diattenuation maps (Yun, McClain & Chipman, "Three-dimensional polarization
# ray-tracing calculus II", Appl. Opt. 50, 2866 (2011)).
#
# Two per-ray RayBatch optional slots (rays.py: Qmat (N,3,3) float64, Jmat
# (N,2,2) complex128; --pol-transport, TraceConfig.pol_transport):
#
#   Qmat -- the running PARALLEL-TRANSPORTED (s,p,k) frame, in GLOBAL xyz
#           coordinates (columns s_hat, p_hat, k_hat). Initialized at birth
#           to the ray's OWN emitted frame (NOT identity -- see init_birth).
#           At every interaction that changes the ray's direction it is
#           left-multiplied by the MINIMAL (shortest-arc / Rodrigues)
#           rotation taking the old direction to the new one -- a PURELY
#           GEOMETRIC update that knows nothing about the interaction's
#           amplitude physics or the interface's own (s,p) Fresnel
#           convention (fresnel.pol_basis). Free flight, and any bare
#           same-direction basis re-expression, contribute identity.
#   Jmat -- the running INTERFACE-CONVENTION cumulative Jones matrix:
#           J <- J_q . R(basis) . J at every amplitude-affecting
#           interaction, where R(basis) is the SAME 2x2 re-expression
#           fresnel.rotate_jones applies to the field (via
#           fresnel.basis_rotation_matrix, so the two updates cannot
#           drift) and J_q is that interaction's local diag(as, ap) /
#           diattenuator / grating-order 2x2. This is the naive
#           "cumulative Jones matrix" of the Chipman formalism: on its
#           own it still conflates real retardance with whatever
#           geometric spin the chain of interface (s,p) conventions
#           introduces relative to true parallel transport.
#
# Post-process (post_process.render_pol_transport) combines them per ray as
#   Delta = (Qmat^T @ frame(s_hat_arrival, dir_arrival))[:2, :2]
#   M     = Delta @ Jmat
# and reads M's retardance / diattenuation / fast-axis off its SVD polar
# decomposition (polar_decompose below). Delta is always a pure 2x2
# rotation (Qmat's and the arrival frame's 3rd column are both the ray's
# actual final direction), so it strips exactly the geometric spin baked
# into Jmat's R(basis) factors without touching M's singular values
# (diattenuation) or its unitary part's eigenvalue phase gap (retardance)
# -- those are invariant under any real-orthogonal similarity; only the
# reported fast-axis ORIENTATION needs Delta.
#
# Slot lifecycle: exactly the differentials/k_dir/refl_hist optional-slot
# pattern (rays.py) -- RayBatch.select() copies both arrays generically;
# RayBatch.concatenate() NaN-fills batches that lack them. Sites whose
# amplitude physics is not (yet) modeled here -- birefringent/biaxial o/e
# channel splits -- call kill() on every child they spawn, exactly mirroring
# tracer._kill_differentials's precedent for the differentials slots at
# those same sites; post-process drops NaN rays from the maps and reports
# the dropped fraction.
#
# KNOWN SEAM (not a hard limitation): Q itself needs nothing but the ray's
# own direction change at a birefringent/biaxial split, so it is trivially
# definable there too -- it is J that is hard (each o/e or slow/fast child
# is a genuine multi-channel linear combination of the incident field, not
# a single diag(as, ap); see tracer._birefringent_children's eigenbasis
# decomposition). The exact per-channel J_step IS mechanically derivable
# from the same coefficients that function already computes (the entry/
# eigenbasis rotate_jones calls + the per-channel amplitude scalars) via
# the same basis_rot2/diag2 composition used everywhere else in this
# module; it was simply out of scope for this round. A future round can
# land it site-by-site without touching Qmat's own update() call at all.
# =============================================================================
import numpy as np


def frame(s_hat, d_hat):
    """Orthonormal (s, p, k) frame as a (...,3,3) matrix with COLUMNS
    s_hat, p_hat = d_hat x s_hat, d_hat -- matches the tracer's p_hat
    convention (cross(dir, s_hat)) everywhere."""
    s_hat = np.asarray(s_hat, dtype=np.float64)
    d_hat = np.asarray(d_hat, dtype=np.float64)
    p_hat = np.cross(d_hat, s_hat)
    return np.stack([s_hat, p_hat, d_hat], axis=-1)


def rotation_between(a, b):
    """Batched (n,3,3) MINIMAL (shortest-arc) rotation matrix R with
    R @ a == b for unit vectors a, b (n,3) -- the pure geometric parallel-
    transport step, independent of any polarization/amplitude bookkeeping.
    Antiparallel rows (a ~ -b, e.g. exact normal-incidence retroreflection)
    fall back to an explicit 180-degree rotation about a deterministic
    axis perpendicular to a (same degenerate-case tie-break style as
    fresnel.pol_basis)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.shape[0]
    v = np.cross(a, b)
    c = np.sum(a * b, axis=-1)
    K = np.zeros((n, 3, 3))
    K[:, 0, 1] = -v[:, 2]
    K[:, 0, 2] = v[:, 1]
    K[:, 1, 0] = v[:, 2]
    K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]
    K[:, 2, 1] = v[:, 0]
    eye = np.eye(3)
    antipar = c < (-1.0 + 1e-9)
    denom = np.where(antipar, 1.0, 1.0 + c)
    R = eye + K + np.einsum('nij,njk->nik', K, K) \
        * (1.0 / denom)[:, None, None]
    if np.any(antipar):
        aa = a[antipar]
        ax = np.zeros_like(aa)
        pick = np.argmin(np.abs(aa), axis=-1)
        ax[np.arange(len(aa)), pick] = 1.0
        perp = np.cross(aa, ax)
        perp /= np.linalg.norm(perp, axis=-1, keepdims=True)
        R180 = 2.0 * perp[:, :, None] * perp[:, None, :] - eye
        R[antipar] = R180
    return R


def init_birth(batch):
    """Allocate Qmat/Jmat on a freshly sampled primary batch: Qmat0 = the
    ray's own emitted (s_hat, p_hat, dir) frame (NOT identity), Jmat0 =
    identity. Call once, from the batch's CURRENT (s_hat, dir) at
    emission -- the spec's 'O_in,initial from the source's emitted
    (s_hat, p_hat, dir); J = I'."""
    batch.Qmat = frame(batch.s_hat, batch.dir).copy()
    n = len(batch)
    batch.Jmat = np.tile(np.eye(2, dtype=np.complex128), (n, 1, 1))


def kill(batch):
    """Mark a child batch's transport as lost (NaN) -- mirrors
    tracer._kill_differentials exactly. Used at sites whose amplitude
    physics is not modeled here (birefringent/biaxial channel splits): Q
    alone would still be well-defined (it only needs the direction
    change), but a Q without its matching Jmat cannot feed M = Q^T P, so
    both are NaN-filled together."""
    if batch is not None and len(batch) > 0 and batch.Qmat is not None:
        batch.Qmat[:] = np.nan
        batch.Jmat[:] = np.nan + 1j * np.nan


def update(child, s_old, d_old, s_new, d_new, j_step):
    """In place: child.Qmat/Jmat currently hold the PARENT's accumulated
    values (RayBatch.select() already copied them before the caller
    mutated child.dir/s_hat) -- overwrite with the new cumulative product
      Q_child = Rstep(d_old -> d_new) @ Q_parent
      J_child = j_step @ J_parent
    s_old/d_old: the ray's basis just BEFORE this interaction (grp.s_hat/
    grp.dir at the call site, i.e. the PARENT's state). s_new/d_new: this
    SPECIFIC child's basis just after (e.g. refl.s_hat/refl.dir). j_step:
    (n,2,2) complex, already composed as J_q @ R(basis) for this
    interaction (see j_step_diag / basis_rot2 below). No-op when Qmat is
    not allocated (--pol-transport off)."""
    if child.Qmat is None:
        return
    Rstep = rotation_between(np.asarray(d_old, dtype=np.float64),
                             np.asarray(d_new, dtype=np.float64))
    child.Qmat = np.einsum('nij,njk->nik', Rstep, child.Qmat)
    child.Jmat = np.einsum('nij,njk->nik', j_step, child.Jmat)


def basis_rot2(s_old, p_old, s_new, p_new):
    """(n,2,2) complex-dtype rotation matrix with EXACTLY the content of
    the 2x2 fresnel.rotate_jones applies to the field (via
    fresnel.basis_rotation_matrix) -- the 'R(basis)' factor of J_step, so
    the field and Jmat updates can never drift apart."""
    from . import fresnel as fr
    css, csp, cps, cpp = fr.basis_rotation_matrix(s_old, p_old, s_new, p_new)
    n = css.shape
    R = np.zeros(n + (2, 2), dtype=np.complex128)
    R[..., 0, 0] = css
    R[..., 0, 1] = csp
    R[..., 1, 0] = cps
    R[..., 1, 1] = cpp
    return R


def diag2(a, b):
    """(n,2,2) complex diagonal matrix from two per-ray amplitude arrays
    (the interaction's local J_q = diag(as, ap))."""
    a = np.asarray(a)
    b = np.asarray(b)
    J = np.zeros(a.shape + (2, 2), dtype=np.complex128)
    J[..., 0, 0] = a
    J[..., 1, 1] = b
    return J


def j_step_diag(amp_s, amp_p, s_old, p_old, s_new, p_new):
    """Convenience: the common 'diag(amp) @ R(basis)' per-interaction J
    step (plain Fresnel reflect/transmit, screens, gratings, roughness/
    ABg scatter lobes)."""
    R = basis_rot2(s_old, p_old, s_new, p_new)
    D = diag2(amp_s, amp_p)
    return np.einsum('nij,njk->nik', D, R)


def polar_decompose(M):
    """M: (...,2,2) complex. Returns (retardance [rad, 0..pi],
    diattenuation [0..1], axis [rad, mod pi]) via SVD polar decomposition
    (M = W @ P_pos, W unitary, P_pos positive-semidefinite).

    Retardance comes from the Pauli/Chipman parametrization of the
    homogeneous retarder W = e^{i*alpha} * (cos(G/2) I - i sin(G/2)
    n . sigma): G = retardance.

    Axis is the ORIENTATION of the dominant right-singular vector (the
    diattenuator's own transmission axis -- a Stokes-parameter-style
    formula, phase-ambiguity-free since it only uses the phase-invariant
    combination v (x) v^H of the singular vector v), used whenever the
    diattenuation is resolvable. This is deliberately NOT read off W's
    Pauli n -- for a pure diattenuator (no retarder at all, e.g. an ideal
    polarizer) W is the identity and n is undefined, even though the
    diattenuator's own axis is perfectly well defined. For any HOMOGENEOUS
    element (real physical Fresnel/polarizer/retarder stacks: M = R(-th)
    diag(lambda1,lambda2) R(th)) the two axes coincide whenever both are
    resolvable, so this is a strict improvement, not just a special case.
    When diattenuation is negligible (M ~ a pure retarder or ~ scalar),
    falls back to W's own Pauli axis (0 when that too is degenerate --
    e.g. a near-pure circular retarder or M ~ scalar * identity, where
    orientation is genuinely ill-defined)."""
    U, S, Vh = np.linalg.svd(M)
    W = np.einsum('...ij,...jk->...ik', U, Vh)
    detW = np.linalg.det(W)
    root = np.exp(-1j * np.angle(detW) / 2.0)
    w = W * root[..., None, None]
    flip = np.real(w[..., 0, 0]) < 0
    w = np.where(flip[..., None, None], -w, w)
    a = w[..., 0, 0]
    b = w[..., 0, 1]
    cos_half = np.clip(np.real(a), -1.0, 1.0)
    gamma = 2.0 * np.arccos(cos_half)
    sin_half = np.sin(gamma / 2.0)
    safe = np.abs(sin_half) > 1e-9
    sin_safe = np.where(safe, sin_half, 1.0)
    nx = np.where(safe, -np.imag(b) / sin_safe, 0.0)
    ny = np.where(safe, -np.real(b) / sin_safe, 0.0)
    axis_w = 0.5 * np.arctan2(ny, nx)

    s1 = S[..., 0]
    s2 = S[..., 1]
    denom = np.clip(s1 ** 2 + s2 ** 2, 1e-300, None)
    diatten = (s1 ** 2 - s2 ** 2) / denom

    v0 = Vh[..., 0, 0]
    v1 = Vh[..., 0, 1]
    s1p = np.abs(v0) ** 2 - np.abs(v1) ** 2
    s2p = 2.0 * np.real(v0 * np.conj(v1))
    axis_v = 0.5 * np.arctan2(s2p, s1p)

    use_v = np.abs(diatten) > 1e-6
    axis = np.where(use_v, axis_v, axis_w)
    return gamma, diatten, axis
