# =============================================================================
# audit.py — power ledger. Every watt a ray loses is credited to exactly one
# bucket at the moment of loss, so closure (sum of buckets == emitted power)
# is an invariant of the trace loop rather than a post-hoc reconciliation.
# Closure is gated at 1e-3 relative per source.
# =============================================================================
import numpy as np

# Buckets partition every watt LOST by rays; their sum must equal emitted
# power (closure). Detector screens are transparent measurement planes —
# power ARRIVING there is a per-detector diagnostic ("detected_geometric",
# tracked in DetectorGrid and by_surface), NOT a loss bucket, otherwise a
# ray passing two detectors would double-count.
BUCKETS = (
    "absorbed_surface",      # mirror/absorbance + metal/film absorption
    "absorbed_bulk",         # Beer-Lambert in media
    "particle_absorbed",     # Mie albedo losses
    "escaped",               # left the scene without hitting anything
    "truncated_generation",  # killed by the reflection-generation cap
    "truncated_power",       # killed by the relative power floor
    "emission_clipped",      # source samples whose hemisphere was clipped
    "polarizer_absorbed",    # dichroic rejection in polarizer elements
                             # (crossed-polarizer scenes park ~all power here)
    "seam_loss",             # rays killed crossing a face-face seam whose
                             # trim tests disagreed (rare; large values
                             # indicate broken geometry)
)


class PowerLedger:
    def __init__(self, n_sources):
        self.n_sources = n_sources
        self.emitted = np.zeros(n_sources)
        self.buckets = {b: np.zeros(n_sources) for b in BUCKETS}
        # sub-ledgers for diagnostics (keyed by body/face name)
        self.by_surface = {}
        self.by_body = {}

    def emit(self, source_id, power):
        np.add.at(self.emitted, source_id, power)

    def credit(self, bucket, source_id, power, where=None):
        """Credit power [W] (arrays ok) to a bucket, optionally tagged with
        a surface/body name for the diagnostic breakdown."""
        np.add.at(self.buckets[bucket], source_id, power)
        if where is not None:
            sub = self.by_surface if bucket in (
                "detected_geometric", "absorbed_surface") else self.by_body
            sub[where] = sub.get(where, 0.0) + float(np.sum(power))

    def closure(self):
        """Per-source relative closure error |1 - sum(buckets)/emitted|."""
        total = sum(self.buckets.values())
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.abs(1.0 - total / self.emitted)
        return np.where(self.emitted > 0, err, 0.0)

    def report(self, source_names, gate=1e-3):
        err = self.closure()
        rep = {
            "sources": {},
            "by_surface_W": {k: v for k, v in sorted(self.by_surface.items())},
            "by_body_W": {k: v for k, v in sorted(self.by_body.items())},
            "closure_gate": gate,
            "closure_ok": bool(np.all(err <= gate)),
        }
        for i, name in enumerate(source_names):
            rep["sources"][name] = {
                "emitted_W": float(self.emitted[i]),
                "closure_error": float(err[i]),
                **{b: float(self.buckets[b][i]) for b in BUCKETS},
            }
        return rep
