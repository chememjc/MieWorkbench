/* ===========================================================================
 * ledger.h — the power ledger: every watt a ray loses is credited to
 * exactly one bucket at the moment of loss, so closure (sum of buckets ==
 * emitted) is an invariant of the trace loop.
 *
 * EXACT port of scripts/raytracer/audit.py PowerLedger: same nine buckets,
 * same by_surface/by_body/flux/detected diagnostic split, same closure
 * definition, gated at 1e-3 downstream. Each OpenMP thread owns a private
 * ledger; ledger_merge folds them in thread order (all quantities are
 * linear tallies — audit.py merge()).
 *
 * Diagnostic maps are label-keyed dicts in Python; here they are arrays
 * indexed by body/detector index (labels known up front), rendered back to
 * dicts by ledger_write_json.
 * =========================================================================== */
#ifndef MIEWB_LEDGER_H
#define MIEWB_LEDGER_H

#include "scene.h"

/* Bucket order matches audit.BUCKETS (audit.py:14-27) — the JSON output
 * must carry the same names. */
enum {
    BK_ABSORBED_SURFACE = 0,
    BK_ABSORBED_BULK,
    BK_PARTICLE_ABSORBED,
    BK_ESCAPED,
    BK_TRUNCATED_GENERATION,
    BK_TRUNCATED_POWER,
    BK_EMISSION_CLIPPED,
    BK_POLARIZER_ABSORBED,
    BK_SEAM_LOSS,
    N_BUCKETS
};

extern const char *const LEDGER_BUCKET_NAMES[N_BUCKETS];

typedef struct {
    int n_sources, n_bodies, n_dets;
    double *emitted;            /* [n_sources] */
    double *buckets;            /* [N_BUCKETS * n_sources] */
    /* audit.py by_surface: absorbed_surface per body label + the detector
     * arrival diagnostic per detector label */
    double *surf_by_body;       /* [n_bodies] */
    double *surf_by_det;        /* [n_dets] */
    /* audit.py by_body: bulk/seam/polarizer losses; slot 0 = "ambient",
     * slot i+1 = body i */
    double *by_body;            /* [n_bodies + 1] */
    double *flux_in, *flux_out; /* [n_bodies] element boundary tallies */
    double *detected;           /* [n_dets] detected_W */
    int n_particles;            /* participating-media count (0 = none) */
    double *by_particles;       /* [n_particles] absorbed W per medium; the
                                 * by_body_W key is the medium's label
                                 * ("particles" / "sample:<label>"). NULL when
                                 * n_particles==0. */
    /* pulsed-optics P7 (track_time only): per-body power-weighted bulk path
     * Sum(surviving_power * segment) [W*m], the GDD-budget mean-path input
     * (tracer.py:466-468 path_tally). Linear tally: merges by sum, weighted
     * by post-bulk-absorption power so a mirror's nm evanescent skin does
     * not book a spurious long metal path. */
    double *path_tally;         /* [n_bodies]; NULL unless track_time */
} LedgerC;

void ledger_init(LedgerC *l, const SceneC *s);
void ledger_free(LedgerC *l);
void ledger_merge(LedgerC *into, const LedgerC *from);

static inline void ledger_credit(LedgerC *l, int bucket, int source_id,
                                 double power) {
    l->buckets[bucket * l->n_sources + source_id] += power;
}

/* Write the PowerLedger.report() JSON shape (audit.py:104-122) to
 * <out_dir>/ledger.json; source names come from the scene. gate mirrors
 * report(gate=1e-3). */
void ledger_write_json(const LedgerC *l, const SceneC *s, const char *path,
                       double gate);

/* Per-source relative closure error (audit.py closure()). */
double ledger_closure_max(const LedgerC *l);

#endif /* MIEWB_LEDGER_H */
