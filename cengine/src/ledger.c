/* ledger.c — see ledger.h. */
#include "ledger.h"
#include "log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *const LEDGER_BUCKET_NAMES[N_BUCKETS] = {
    "absorbed_surface",
    "absorbed_bulk",
    "particle_absorbed",
    "escaped",
    "truncated_generation",
    "truncated_power",
    "emission_clipped",
    "polarizer_absorbed",
    "seam_loss",
};

void ledger_init(LedgerC *l, const SceneC *s) {
    l->n_sources = s->n_sources;
    l->n_bodies = s->n_bodies;
    l->n_dets = s->n_dets;
    l->emitted = (double *)calloc((size_t)l->n_sources, sizeof(double));
    l->buckets = (double *)calloc((size_t)N_BUCKETS * l->n_sources,
                                  sizeof(double));
    l->surf_by_body = (double *)calloc((size_t)l->n_bodies, sizeof(double));
    l->surf_by_det = (double *)calloc((size_t)(l->n_dets ? l->n_dets : 1),
                                      sizeof(double));
    l->by_body = (double *)calloc((size_t)l->n_bodies + 1, sizeof(double));
    l->flux_in = (double *)calloc((size_t)l->n_bodies, sizeof(double));
    l->flux_out = (double *)calloc((size_t)l->n_bodies, sizeof(double));
    l->detected = (double *)calloc((size_t)(l->n_dets ? l->n_dets : 1),
                                   sizeof(double));
    /* pulsed-optics P7: the bulk-path tally exists only under time tracking */
    l->path_tally = s->track_time
        ? (double *)calloc((size_t)l->n_bodies, sizeof(double)) : NULL;
    if (!l->emitted || !l->buckets || !l->surf_by_body || !l->surf_by_det
            || !l->by_body || !l->flux_in || !l->flux_out || !l->detected
            || (s->track_time && !l->path_tally))
        die(EXIT_PHYSICS, "ledger: allocation failed");
}

void ledger_free(LedgerC *l) {
    free(l->emitted);
    free(l->buckets);
    free(l->surf_by_body);
    free(l->surf_by_det);
    free(l->by_body);
    free(l->flux_in);
    free(l->flux_out);
    free(l->detected);
    free(l->path_tally);
    memset(l, 0, sizeof *l);
}

void ledger_merge(LedgerC *into, const LedgerC *from) {
    for (int i = 0; i < into->n_sources; i++)
        into->emitted[i] += from->emitted[i];
    for (int i = 0; i < N_BUCKETS * into->n_sources; i++)
        into->buckets[i] += from->buckets[i];
    for (int i = 0; i < into->n_bodies; i++) {
        into->surf_by_body[i] += from->surf_by_body[i];
        into->flux_in[i] += from->flux_in[i];
        into->flux_out[i] += from->flux_out[i];
    }
    if (into->path_tally && from->path_tally)
        for (int i = 0; i < into->n_bodies; i++)
            into->path_tally[i] += from->path_tally[i];
    for (int i = 0; i <= into->n_bodies; i++)
        into->by_body[i] += from->by_body[i];
    for (int i = 0; i < into->n_dets; i++) {
        into->surf_by_det[i] += from->surf_by_det[i];
        into->detected[i] += from->detected[i];
    }
    into->by_particles += from->by_particles;
}

static double closure_err(const LedgerC *l, int i) {
    if (l->emitted[i] <= 0.0) return 0.0;
    double total = 0.0;
    for (int b = 0; b < N_BUCKETS; b++)
        total += l->buckets[b * l->n_sources + i];
    return fabs(1.0 - total / l->emitted[i]);
}

double ledger_closure_max(const LedgerC *l) {
    double worst = 0.0;
    for (int i = 0; i < l->n_sources; i++) {
        double e = closure_err(l, i);
        if (e > worst) worst = e;
    }
    return worst;
}

/* JSON string escaping for labels (quotes/backslashes only — labels come
 * from FreeCAD body names which the contract already restricts). */
static void jesc(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') fputc('\\', f);
        fputc(*s, f);
    }
    fputc('"', f);
}

void ledger_write_json(const LedgerC *l, const SceneC *s, const char *path,
                       double gate) {
    FILE *f = fopen(path, "w");
    if (!f) die(EXIT_PHYSICS, "ledger: cannot write %s", path);
    int ok = 1;

    fprintf(f, "{\n  \"sources\": {\n");
    for (int i = 0; i < l->n_sources; i++) {
        fprintf(f, "    ");
        jesc(f, s->sources[i].label);
        fprintf(f, ": {\"emitted_W\": %.17g, \"closure_error\": %.17g",
                l->emitted[i], closure_err(l, i));
        for (int b = 0; b < N_BUCKETS; b++)
            fprintf(f, ", \"%s\": %.17g", LEDGER_BUCKET_NAMES[b],
                    l->buckets[b * l->n_sources + i]);
        fprintf(f, "}%s\n", i + 1 < l->n_sources ? "," : "");
        if (closure_err(l, i) > gate) ok = 0;
    }
    /* by_surface_W: absorbed_surface per body + detector arrivals; keys
     * sorted lexically like Python's sorted() — the glue re-sorts on load
     * anyway, so plain order is fine for machine consumption. Zero entries
     * are omitted (Python only creates keys on first credit). */
    fprintf(f, "  },\n  \"by_surface_W\": {");
    int first = 1;
    for (int i = 0; i < l->n_bodies; i++) {
        if (l->surf_by_body[i] == 0.0) continue;
        fprintf(f, "%s\n    ", first ? "" : ",");
        jesc(f, s->bodies[i].label);
        fprintf(f, ": %.17g", l->surf_by_body[i]);
        first = 0;
    }
    for (int i = 0; i < l->n_dets; i++) {
        if (l->surf_by_det[i] == 0.0) continue;
        fprintf(f, "%s\n    ", first ? "" : ",");
        jesc(f, s->dets[i].label);
        fprintf(f, ": %.17g", l->surf_by_det[i]);
        first = 0;
    }
    fprintf(f, "\n  },\n  \"by_body_W\": {");
    first = 1;
    for (int i = 0; i <= l->n_bodies; i++) {
        if (l->by_body[i] == 0.0) continue;
        fprintf(f, "%s\n    ", first ? "" : ",");
        if (i == 0) fprintf(f, "\"ambient\"");
        else jesc(f, s->bodies[i - 1].label);
        fprintf(f, ": %.17g", l->by_body[i]);
        first = 0;
    }
    if (l->by_particles != 0.0) {
        fprintf(f, "%s\n    \"particles\": %.17g", first ? "" : ",",
                l->by_particles);
        first = 0;
    }
    fprintf(f, "\n  },\n  \"element_flux_W\": {");
    first = 1;
    for (int i = 0; i < l->n_bodies; i++) {
        if (l->flux_in[i] == 0.0 && l->flux_out[i] == 0.0) continue;
        fprintf(f, "%s\n    ", first ? "" : ",");
        jesc(f, s->bodies[i].label);
        fprintf(f, ": {\"in_W\": %.17g, \"out_W\": %.17g}",
                l->flux_in[i], l->flux_out[i]);
        first = 0;
    }
    fprintf(f, "\n  },\n  \"detected_W\": {");
    first = 1;
    for (int i = 0; i < l->n_dets; i++) {
        if (l->detected[i] == 0.0) continue;
        fprintf(f, "%s\n    ", first ? "" : ",");
        jesc(f, s->dets[i].label);
        fprintf(f, ": %.17g", l->detected[i]);
        first = 0;
    }
    /* pulsed-optics P7: per-body power-weighted bulk path [W*m] (present only
     * under time tracking; the Python audit key is path_tally_Wm). Consumed by
     * run_c_case -> build_gdd_budget, which does ALL dispersion resolution in
     * Python (this side is a pure geometric power*length tally). */
    if (l->path_tally) {
        fprintf(f, "\n  },\n  \"path_tally_Wm\": {");
        first = 1;
        for (int i = 0; i < l->n_bodies; i++) {
            if (l->path_tally[i] == 0.0) continue;
            fprintf(f, "%s\n    ", first ? "" : ",");
            jesc(f, s->bodies[i].label);
            fprintf(f, ": %.17g", l->path_tally[i]);
            first = 0;
        }
    }
    fprintf(f, "\n  },\n  \"closure_gate\": %.17g,\n"
            "  \"closure_ok\": %s\n}\n", gate, ok ? "true" : "false");
    if (fclose(f) != 0)
        die(EXIT_PHYSICS, "ledger: short write to %s", path);
}
