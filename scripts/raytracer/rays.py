# =============================================================================
# rays.py — RayBatch: struct-of-arrays state for a batch of rays.
#
# Power/phase convention: the ray's power [W] is |Es|^2 + |Ep|^2; the phase
# of (Es, Ep) plus k * opl is the field phase used by the coherent gather.
# opl is the accumulated optical path length Sum(Re(n) * ds) with opl = 0 on
# the source's emitting surface (the emitting surface IS the reference
# wavefront). Everything float64 — see the phase-precision note in gather.py.
#
# medium: per-ray stack of body indices (-1 = ambient) with explicit depth;
# push on entering a solid, pop (validated) on leaving. Depth 4 is enough for
# solids nested 4 deep, which no sane optical bench exceeds; overflow is a
# hard error rather than silent corruption.
# =============================================================================
import numpy as np

MEDIUM_STACK_DEPTH = 4
AMBIENT = -1
# refl_hist width: the ghost/stray-light history records the face id of each
# reflection event, one slot per reflection generation, capped at this depth
# (generation >= HIST_DEPTH-1 reuse the last slot).
HIST_DEPTH = 8


class RayBatch:
    __slots__ = ("pos", "dir", "s_hat", "Es", "Ep", "lam", "opl",
                 "medium", "depth", "source_id", "lam_stratum",
                 "pol_stratum", "pol_mode", "n_eff", "n_g_eff",
                 "generation", "last_face", "coherent", "birth_power",
                 "viz_flag", "scattered",
                 "dPdx", "dDdx", "dPdy", "dDdy", "birth_pos", "k_dir",
                 "refl_hist", "gopl", "gdd_acc", "Qmat", "Jmat")

    # ray-differential slots (Igehy): allocated ONLY under
    # --ray-differentials (None otherwise — +96 B/ray when on). NaN rows
    # mean "differential lost" (grating/scatter/birefringent children);
    # the gather falls back to the source-referenced area per sample.
    _DIFF_SLOTS = ("dPdx", "dDdx", "dPdy", "dDdy")

    # birth_pos: (N,3) world-metres position of each ray's birth point on
    # its source face — allocated ONLY under --export-rays (None otherwise,
    # +24 B/ray when on), following the same optional-slot lifecycle as the
    # differential slots (select copies it; a mixed concat NaN-fills the
    # batches that lack it). Inherited unchanged by every child ray, so a
    # detected ray carries the pupil coordinate of the primary it came from.

    # k_dir: (N,3) unit WAVEVECTOR direction for rays inside a biaxial
    # crystal (pol_mode 2/3), where ray (Poynting) and wavevector differ
    # and — unlike uniaxial — the ray->k inversion has no closed form, so
    # k must be carried explicitly. Allocated only by the biaxial entry
    # interface; NaN rows in a mixed concat are rays that never entered a
    # biaxial medium (their pol_mode is 0/1 and k_dir is never read).

    # gopl / gdd_acc: time-domain accumulators (TraceConfig.track_time;
    # None otherwise, +16 B/ray when on, same optional-slot lifecycle as
    # birth_pos). gopl = accumulated GROUP optical path Sum(n_g * ds)
    # [metres] (envelope arrival time = gopl / c); gdd_acc = accumulated
    # group-delay dispersion Sum((phi2/L) * ds) [s^2]. Zero at birth
    # (alloc_time); children inherit through select, so a detected ray
    # carries the full group delay of its lineage. A mixed concat
    # NaN-fills the batches that lack them.

    # refl_hist: (N, HIST_DEPTH) int32 face-id history for ghost/stray-light
    # analysis — allocated ONLY under --ghost-analysis (TraceConfig.
    # track_history; None otherwise, +32 B/ray when on). -1-filled; the
    # tracer writes the FACE id of each reflection event into slot
    # min(pre-increment generation, HIST_DEPTH-1). Same optional-slot
    # lifecycle as birth_pos (select copies it; a mixed concat -1-fills the
    # batches that lack it). Transmitted children inherit it unchanged, so a
    # detected ray carries the ordered sequence of surfaces it reflected off.

    # Qmat / Jmat: P2 parallel-transport polarization bookkeeping —
    # allocated ONLY under --pol-transport (TraceConfig.pol_transport; None
    # otherwise, +80 B/ray when on: (3,3) float64 + (2,2) complex128). See
    # poltransport.py for the full construction. Qmat is the running
    # parallel-transported (s,p,k) frame (init'd to the ray's OWN emitted
    # frame, not identity); Jmat is the running interface-convention
    # cumulative Jones matrix (init'd to identity). Same optional-slot
    # lifecycle as birth_pos (select copies both; a mixed concat NaN-fills
    # the batches that lack them — sites whose amplitude physics is not
    # modeled here, e.g. birefringent/biaxial channel splits, NaN both
    # explicitly via poltransport.kill()).

    def __init__(self, n):
        self.pos = np.zeros((n, 3), dtype=np.float64)
        self.dir = np.zeros((n, 3), dtype=np.float64)
        self.s_hat = np.zeros((n, 3), dtype=np.float64)
        self.Es = np.zeros(n, dtype=np.complex128)
        self.Ep = np.zeros(n, dtype=np.complex128)
        self.lam = np.zeros(n, dtype=np.float64)
        self.opl = np.zeros(n, dtype=np.float64)
        self.medium = np.full((n, MEDIUM_STACK_DEPTH), AMBIENT,
                              dtype=np.int16)
        self.depth = np.zeros(n, dtype=np.int8)
        self.source_id = np.zeros(n, dtype=np.int16)
        self.lam_stratum = np.zeros(n, dtype=np.int16)
        # pol_stratum: mutually-incoherent polarization population id.
        # Unpolarized sources emit two orthogonal populations (0/1) that
        # NEVER interfere — the gather keeps per-(source, lam, pol)
        # accumulators, exactly like wavelength strata. Polarized sources
        # emit a single stratum 0.
        self.pol_stratum = np.zeros(n, dtype=np.int16)
        # pol_mode / n_eff: uniaxial birefringence state. pol_mode 0 =
        # isotropic or ordinary ray, 1 = extraordinary ray. n_eff > 0
        # overrides the medium's phase index in the bulk OPL step (the
        # e-ray's direction-dependent index is fixed at the entry
        # interface and constant along the segment inside the crystal).
        self.pol_mode = np.zeros(n, dtype=np.int8)
        self.n_eff = np.zeros(n, dtype=np.float64)
        # n_g_eff: directional GROUP index frozen at a crystal entry
        # interface (uniaxial e-rays, biaxial slow/fast), the group-index
        # sibling of n_eff. 0 = "not directional; use the medium's scalar
        # group index". MANDATORY slot (like n_eff) so it survives
        # select/concatenate even when time tracking is off.
        self.n_g_eff = np.zeros(n, dtype=np.float64)
        self.generation = np.zeros(n, dtype=np.int16)
        self.last_face = np.full(n, -1, dtype=np.int32)
        self.coherent = np.zeros(n, dtype=bool)
        # birth_power: the primary ray's power at emission — the reference
        # for the relative power floor (children inherit it unchanged).
        self.birth_power = np.zeros(n, dtype=np.float64)
        # viz_flag: record this ray's segments in the visualization store
        # (set on the first viz_rays primaries per source; inherited by
        # children through select/concatenate)
        self.viz_flag = np.zeros(n, dtype=bool)
        # scattered: ray underwent a random-direction event (explicit
        # particle scatter, roughness lobe). In the coherent gather these
        # form the physical-speckle population whose incoherent pedestal
        # is REAL intensity; smooth (non-scattered) samples get the MC
        # pedestal expectation subtracted instead (see gather.py).
        self.scattered = np.zeros(n, dtype=bool)
        self.dPdx = None
        self.dDdx = None
        self.dPdy = None
        self.dDdy = None
        self.birth_pos = None
        self.k_dir = None
        self.refl_hist = None
        self.gopl = None
        self.gdd_acc = None
        self.Qmat = None
        self.Jmat = None

    def alloc_differentials(self):
        for name in self._DIFF_SLOTS:
            setattr(self, name, np.zeros((len(self), 3), dtype=np.float64))

    def alloc_history(self):
        self.refl_hist = np.full((len(self), HIST_DEPTH), -1, dtype=np.int32)

    def alloc_time(self):
        """Allocate + zero-fill the time-domain accumulators (gopl,
        gdd_acc) on this batch (TraceConfig.track_time; mirrors
        alloc_history). gopl = 0 at birth by definition: the emitting
        surface is the group-delay reference plane."""
        self.gopl = np.zeros(len(self), dtype=np.float64)
        self.gdd_acc = np.zeros(len(self), dtype=np.float64)

    @property
    def has_differentials(self):
        return self.dPdx is not None

    def __len__(self):
        return len(self.lam)

    @property
    def power(self):
        return np.abs(self.Es) ** 2 + np.abs(self.Ep) ** 2

    def current_medium(self):
        """Body index of the medium each ray is travelling in (-1 ambient)."""
        idx = np.maximum(self.depth - 1, 0)
        top = self.medium[np.arange(len(self)), idx]
        return np.where(self.depth > 0, top, AMBIENT)

    def push_medium(self, mask, body_index):
        if not np.any(mask):
            return
        if np.any(self.depth[mask] >= MEDIUM_STACK_DEPTH):
            raise RuntimeError(
                "medium stack overflow (solids nested > %d deep) — "
                "check for overlapping solids" % MEDIUM_STACK_DEPTH)
        rows = np.where(mask)[0]
        self.medium[rows, self.depth[rows]] = body_index[rows] \
            if isinstance(body_index, np.ndarray) else body_index
        self.depth[rows] += 1

    def pop_medium(self, mask, expect_body):
        """Pop the top medium; hard error if it isn't the expected body
        (catches non-manifold nesting / overlapping solids at runtime)."""
        if not np.any(mask):
            return
        rows = np.where(mask)[0]
        if np.any(self.depth[rows] <= 0):
            raise RuntimeError(
                "medium stack underflow — ray exits a solid it never "
                "entered (overlapping solids or orientation bug)")
        top = self.medium[rows, self.depth[rows] - 1]
        exp = expect_body[rows] if isinstance(expect_body, np.ndarray) \
            else expect_body
        bad = top != exp
        if np.any(bad):
            raise RuntimeError(
                "medium stack pop mismatch: expected body %r got %r — "
                "non-manifold nesting (overlapping solids?)"
                % (np.unique(np.asarray(exp)[bad] if np.ndim(exp) else exp),
                   np.unique(top[bad])))
        self.medium[rows, self.depth[rows] - 1] = AMBIENT
        self.depth[rows] -= 1

    def select(self, mask_or_idx):
        """New RayBatch holding the selected rays (copies)."""
        idx = np.where(mask_or_idx)[0] if np.asarray(
            mask_or_idx).dtype == bool else np.asarray(mask_or_idx)
        out = RayBatch(len(idx))
        for name in self.__slots__:
            src = getattr(self, name)
            if src is None:
                continue
            # optional slots (_DIFF_SLOTS, birth_pos) start as None on the
            # fresh out batch — allocate them by copying the selection;
            # the mandatory slots are pre-allocated, so assign into them
            if getattr(out, name) is None:
                setattr(out, name, src[idx].copy())
            else:
                getattr(out, name)[...] = src[idx]
        return out

    @staticmethod
    def concatenate(batches):
        batches = [b for b in batches if len(b) > 0]
        if not batches:
            return RayBatch(0)
        out = RayBatch(sum(len(b) for b in batches))
        if any(b.has_differentials for b in batches):
            # mixed batches: rays without differentials get NaN (lost)
            out.alloc_differentials()
            for name in RayBatch._DIFF_SLOTS:
                getattr(out, name)[:] = np.nan
        if any(b.birth_pos is not None for b in batches):
            # mixed batches: rays from a batch without birth_pos NaN-fill
            out.birth_pos = np.full((len(out), 3), np.nan)
        if any(b.k_dir is not None for b in batches):
            # mixed batches: NaN marks rays that never entered a biaxial
            # medium (pol_mode 0/1 — k_dir is never read for them)
            out.k_dir = np.full((len(out), 3), np.nan)
        if any(b.refl_hist is not None for b in batches):
            # mixed batches: -1 marks rays from a batch without history
            out.refl_hist = np.full((len(out), HIST_DEPTH), -1,
                                    dtype=np.int32)
        if any(b.gopl is not None for b in batches):
            # mixed batches: rays from a batch without time tracking
            # NaN-fill (their group delay is undefined, not zero)
            out.gopl = np.full(len(out), np.nan)
        if any(b.gdd_acc is not None for b in batches):
            out.gdd_acc = np.full(len(out), np.nan)
        if any(b.Qmat is not None for b in batches):
            # mixed batches: rays from a batch without pol-transport
            # tracking NaN-fill (their Q/J history is undefined, not
            # identity) — poltransport.polar_decompose drops NaN rays.
            out.Qmat = np.full((len(out), 3, 3), np.nan)
        if any(b.Jmat is not None for b in batches):
            out.Jmat = np.full((len(out), 2, 2), np.nan + 1j * np.nan,
                               dtype=np.complex128)
        at = 0
        for b in batches:
            n = len(b)
            for name in RayBatch.__slots__:
                src = getattr(b, name)
                if src is None:
                    continue
                getattr(out, name)[at:at + n] = src
            at += n
        return out
