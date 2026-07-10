/* ===========================================================================
 * trace.h — the propagation loop (port of tracer.py Tracer) and source
 * sampling (port of sources.py sample_source).
 * =========================================================================== */
#ifndef MIEWB_TRACE_H
#define MIEWB_TRACE_H

#include "scene.h"
#include "ledger.h"
#include "detector.h"

/* viz polyline record — one row of the (M, 13) rays.npy contract
 * (tracer.py VizStore/run_trace.py:12-17) */
typedef struct {
    double row[13];
} VizRec;

typedef struct {
    VizRec *v;
    int64_t n, cap;
} VizVec;

typedef struct {
    LedgerC ledger;         /* merged result */
    VizVec viz;             /* merged viz polylines */
    int64_t rays_traced;    /* total ray-interactions processed */
    double trace_seconds;
} TraceResultC;

/* Sample all sources (deterministic per (seed, source, primary index) —
 * see rng.h) and run the trace to completion. Detector cubes/tallies
 * accumulate into scene->dets. */
void trace_run(SceneC *scene, TraceResultC *out);

void trace_result_free(TraceResultC *r);

#endif /* MIEWB_TRACE_H */
