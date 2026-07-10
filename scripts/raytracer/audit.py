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
        # per-element boundary flux (diagnostic TALLIES, not closure
        # buckets): power arriving at an element from outside ("in_W") and
        # power leaving it back into the surroundings ("out_W"). in - out
        # ~= power absorbed inside/at the element; small shortfalls from
        # generation/power-truncated rays are expected and documented.
        self.flux = {}
        # detected power per detector label, kept separately from
        # by_surface (which historically mixes surface absorption and
        # detection under one name)
        self.detected = {}

    def emit(self, source_id, power):
        np.add.at(self.emitted, source_id, power)

    def flux_in(self, label, power):
        entry = self.flux.setdefault(label, {"in_W": 0.0, "out_W": 0.0})
        entry["in_W"] += float(power)

    def flux_out(self, label, power):
        entry = self.flux.setdefault(label, {"in_W": 0.0, "out_W": 0.0})
        entry["out_W"] += float(power)

    def detect(self, label, power):
        self.detected[label] = self.detected.get(label, 0.0) + float(power)

    def credit(self, bucket, source_id, power, where=None):
        """Credit power [W] (arrays ok) to a bucket, optionally tagged with
        a surface/body name for the diagnostic breakdown."""
        np.add.at(self.buckets[bucket], source_id, power)
        if where is not None:
            sub = self.by_surface if bucket in (
                "detected_geometric", "absorbed_surface") else self.by_body
            sub[where] = sub.get(where, 0.0) + float(np.sum(power))

    def merge(self, other):
        """Fold another ledger into this one (multi-process --workers trace
        sharding). Every quantity is a linear tally, so shards add: the
        emitted / bucket arrays add elementwise, and the diagnostic
        by_surface/by_body/detected dicts and the per-element flux in/out
        sub-totals add per key. Returns self for chaining."""
        if other.n_sources != self.n_sources:
            raise ValueError(
                "PowerLedger.merge: source-count mismatch (%d vs %d)"
                % (self.n_sources, other.n_sources))
        self.emitted += other.emitted
        for b in BUCKETS:
            self.buckets[b] += other.buckets[b]
        for k, v in other.by_surface.items():
            self.by_surface[k] = self.by_surface.get(k, 0.0) + v
        for k, v in other.by_body.items():
            self.by_body[k] = self.by_body.get(k, 0.0) + v
        for k, v in other.flux.items():
            entry = self.flux.setdefault(k, {"in_W": 0.0, "out_W": 0.0})
            entry["in_W"] += v["in_W"]
            entry["out_W"] += v["out_W"]
        for k, v in other.detected.items():
            self.detected[k] = self.detected.get(k, 0.0) + v
        return self

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
            "element_flux_W": {k: dict(v)
                               for k, v in sorted(self.flux.items())},
            "detected_W": {k: v for k, v in sorted(self.detected.items())},
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
