/* detector.c — see detector.h. */
#include "detector.h"
#include "log.h"
#include "npyio.h"
#include "../vendor/yyjson/yyjson.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void dethits_init(DetHitVec *h) {
    h->cap = 1024;
    h->n = 0;
    h->v = (DetHit *)malloc((size_t)h->cap * sizeof(DetHit));
    if (!h->v) die(EXIT_PHYSICS, "dethits: allocation failed");
}

void dethits_free(DetHitVec *h) {
    free(h->v);
    memset(h, 0, sizeof *h);
}

void dethits_push(DetHitVec *h, const DetHit *hit) {
    if (h->n == h->cap) {
        h->cap *= 2;
        DetHit *p = (DetHit *)realloc(h->v, (size_t)h->cap * sizeof(DetHit));
        if (!p) die(EXIT_PHYSICS, "dethits: growth to %lld failed",
                    (long long)h->cap);
        h->v = p;
    }
    h->v[h->n++] = *hit;
}

void dethits_clear(DetHitVec *h) { h->n = 0; }

void gathhits_init(GatherHitVec *h) {
    h->cap = 1024;
    h->n = 0;
    h->v = (GatherHit *)malloc((size_t)h->cap * sizeof(GatherHit));
    if (!h->v) die(EXIT_PHYSICS, "gathhits: allocation failed");
}

void gathhits_free(GatherHitVec *h) {
    free(h->v);
    memset(h, 0, sizeof *h);
}

void gathhits_push(GatherHitVec *h, const GatherHit *hit) {
    if (h->n == h->cap) {
        h->cap *= 2;
        GatherHit *p = (GatherHit *)realloc(
            h->v, (size_t)h->cap * sizeof(GatherHit));
        if (!p) die(EXIT_PHYSICS, "gathhits: growth to %lld failed",
                    (long long)h->cap);
        h->v = p;
    }
    h->v[h->n++] = *hit;
}

void gathhits_clear(GatherHitVec *h) { h->n = 0; }

void exportvec_init(ExportVec *e) {
    e->cap = 1024;
    e->n = 0;
    e->v = (ExportRec *)malloc((size_t)e->cap * sizeof(ExportRec));
    if (!e->v) die(EXIT_PHYSICS, "exports: allocation failed");
}

void exportvec_free(ExportVec *e) {
    free(e->v);
    memset(e, 0, sizeof *e);
}

void exportvec_push(ExportVec *e, const ExportRec *r) {
    if (e->n == e->cap) {
        e->cap *= 2;
        ExportRec *p = (ExportRec *)realloc(
            e->v, (size_t)e->cap * sizeof(ExportRec));
        if (!p) die(EXIT_PHYSICS, "exports: growth failed");
        e->v = p;
    }
    e->v[e->n++] = *r;
}

void exportvec_clear(ExportVec *e) { e->n = 0; }

void det_collect_exports(SceneC *s, const ExportVec *e) {
    for (int64_t i = 0; i < e->n; i++) {
        DetC *d = &s->dets[e->v[i].det];
        if (d->n_exports == d->cap_exports) {
            d->cap_exports = d->cap_exports ? d->cap_exports * 2 : 4096;
            void *p = realloc(d->exports, (size_t)d->cap_exports
                              * sizeof(ExportRec));
            if (!p) die(EXIT_PHYSICS, "exports: detector growth failed");
            d->exports = p;
        }
        ((ExportRec *)d->exports)[d->n_exports++] = e->v[i];
    }
}

void det_write_exports(const SceneC *s) {
    if (!s->export_rays) return;
    char path[1200];
    for (int di = 0; di < s->n_dets; di++) {
        const DetC *d = &s->dets[di];
        int64_t n = d->n_exports;
        const ExportRec *recs = (const ExportRec *)d->exports;
        /* SoA staging buffers */
        double *v3buf = (double *)malloc((size_t)(n > 0 ? n : 1) * 3
                                         * sizeof(double));
        double *sc = (double *)malloc((size_t)(n > 0 ? n : 1)
                                      * sizeof(double));
        int32_t *hist = (int32_t *)malloc(
            (size_t)(n > 0 ? n : 1) * HIST_DEPTH * sizeof(int32_t));
        if (!v3buf || !sc || !hist)
            die(EXIT_PHYSICS, "exports: staging allocation failed");
        struct { const char *name; int kind; } fields[] = {
            {"pos", 0}, {"dir", 1}, {"birth_pos", 2},
            {"opl", 3}, {"lam", 4}, {"power", 5}, {"source_id", 6},
            {"lam_stratum", 7}, {"pol_stratum", 8}, {"generation", 9},
            {"pol_mode", 10}, {"scattered", 11}, {"coherent", 12},
        };
        for (size_t f = 0; f < sizeof(fields) / sizeof(fields[0]); f++) {
            int kind = fields[f].kind;
            if (kind <= 2) {
                for (int64_t i = 0; i < n; i++) {
                    kvec3 v = kind == 0 ? recs[i].pos
                            : kind == 1 ? recs[i].dir
                                        : recs[i].birth_pos;
                    v3buf[i * 3] = v.x;
                    v3buf[i * 3 + 1] = v.y;
                    v3buf[i * 3 + 2] = v.z;
                }
                snprintf(path, sizeof path, "%s/exp_%d_%s.npy",
                         s->out_dir, di, fields[f].name);
                npy_write_f64_2d(path, v3buf, (size_t)n, 3);
            } else {
                for (int64_t i = 0; i < n; i++) {
                    const ExportRec *r = &recs[i];
                    double x = 0.0;
                    switch (kind) {
                    case 3: x = r->opl; break;
                    case 4: x = r->lam; break;
                    case 5: x = r->power; break;
                    case 6: x = r->source_id; break;
                    case 7: x = r->lam_stratum; break;
                    case 8: x = r->pol_stratum; break;
                    case 9: x = r->generation; break;
                    case 10: x = r->pol_mode; break;
                    case 11: x = r->scattered; break;
                    case 12: x = r->coherent; break;
                    }
                    sc[i] = x;
                }
                snprintf(path, sizeof path, "%s/exp_%d_%s.npy",
                         s->out_dir, di, fields[f].name);
                npy_write_f64_1d(path, sc, (size_t)n);
            }
        }
        if (s->track_history) {
            for (int64_t i = 0; i < n; i++)
                for (int k = 0; k < HIST_DEPTH; k++)
                    hist[i * HIST_DEPTH + k] = recs[i].refl_hist[k];
            snprintf(path, sizeof path, "%s/exp_%d_refl_hist.npy",
                     s->out_dir, di);
            size_t shape[2] = {(size_t)n, HIST_DEPTH};
            npy_write(path, hist, "<i4", 2, shape);
        }
        free(v3buf);
        free(sc);
        free(hist);
    }
}

void det_free_exports(SceneC *s) {
    for (int i = 0; i < s->n_dets; i++) {
        free(s->dets[i].exports);
        s->dets[i].exports = NULL;
        s->dets[i].n_exports = s->dets[i].cap_exports = 0;
    }
}

/* ---------------------------------------------------------------------------
 * pulsed-optics P7 time-product arrival records — same collect/write/free
 * lifecycle as the --export-rays records above.
 * ------------------------------------------------------------------------ */
void timevec_init(TimeVec *t) {
    t->cap = 1024;
    t->n = 0;
    t->v = (TimeRec *)malloc((size_t)t->cap * sizeof(TimeRec));
    if (!t->v) die(EXIT_PHYSICS, "times: allocation failed");
}

void timevec_free(TimeVec *t) {
    free(t->v);
    memset(t, 0, sizeof *t);
}

void timevec_push(TimeVec *t, const TimeRec *r) {
    if (t->n == t->cap) {
        t->cap *= 2;
        TimeRec *p = (TimeRec *)realloc(t->v,
                                        (size_t)t->cap * sizeof(TimeRec));
        if (!p) die(EXIT_PHYSICS, "times: growth failed");
        t->v = p;
    }
    t->v[t->n++] = *r;
}

void timevec_clear(TimeVec *t) { t->n = 0; }

void det_collect_times(SceneC *s, const TimeVec *t) {
    for (int64_t i = 0; i < t->n; i++) {
        DetC *d = &s->dets[t->v[i].det];
        if (d->n_times == d->cap_times) {
            d->cap_times = d->cap_times ? d->cap_times * 2 : 4096;
            void *p = realloc(d->times,
                              (size_t)d->cap_times * sizeof(TimeRec));
            if (!p) die(EXIT_PHYSICS, "times: detector growth failed");
            d->times = p;
        }
        ((TimeRec *)d->times)[d->n_times++] = t->v[i];
    }
}

void det_write_times(const SceneC *s) {
    if (!s->time_products) return;
    char path[1200];
    for (int di = 0; di < s->n_dets; di++) {
        const DetC *d = &s->dets[di];
        int64_t n = d->n_times;
        const TimeRec *recs = (const TimeRec *)d->times;
        double *sc = (double *)malloc((size_t)(n > 0 ? n : 1)
                                      * sizeof(double));
        if (!sc) die(EXIT_PHYSICS, "times: staging allocation failed");
        /* one f64 column per field (fx/fy/lam/gdd are f32 in the record but
         * dumped as f64 — run_c_case recasts to the Python dtypes, and the
         * finalize_time math is fp64 anyway). */
        struct { const char *name; int kind; } fields[] = {
            {"t", 0}, {"fx", 1}, {"fy", 2}, {"lam", 3}, {"power", 4},
            {"source_id", 5}, {"lam_stratum", 6}, {"gdd", 7},
        };
        for (size_t f = 0; f < sizeof(fields) / sizeof(fields[0]); f++) {
            int kind = fields[f].kind;
            for (int64_t i = 0; i < n; i++) {
                const TimeRec *r = &recs[i];
                double x = 0.0;
                switch (kind) {
                case 0: x = r->t; break;
                case 1: x = r->fx; break;
                case 2: x = r->fy; break;
                case 3: x = r->lam; break;
                case 4: x = r->power; break;
                case 5: x = r->source_id; break;
                case 6: x = r->lam_stratum; break;
                case 7: x = r->gdd; break;
                }
                sc[i] = x;
            }
            snprintf(path, sizeof path, "%s/time_%d_%s.npy",
                     s->out_dir, di, fields[f].name);
            npy_write_f64_1d(path, sc, (size_t)n);
        }
        free(sc);
    }
}

void det_free_times(SceneC *s) {
    for (int i = 0; i < s->n_dets; i++) {
        free(s->dets[i].times);
        s->dets[i].times = NULL;
        s->dets[i].n_times = s->dets[i].cap_times = 0;
    }
}

/* find-or-create the GKey sample set for a (source, stratum, pol) triple */
static GKey *det_gkey(DetC *d, int16_t src, int16_t ls, int16_t ps) {
    for (int32_t i = 0; i < d->n_gkeys; i++) {
        GKey *g = &d->gkeys[i];
        if (g->source_id == src && g->lam_stratum == ls
                && g->pol_stratum == ps)
            return g;
    }
    if (d->n_gkeys == d->cap_gkeys) {
        d->cap_gkeys = d->cap_gkeys ? d->cap_gkeys * 2 : 8;
        GKey *p = (GKey *)realloc(d->gkeys,
                                  (size_t)d->cap_gkeys * sizeof(GKey));
        if (!p) die(EXIT_PHYSICS, "detector: gkey growth failed");
        d->gkeys = p;
    }
    GKey *g = &d->gkeys[d->n_gkeys++];
    memset(g, 0, sizeof *g);
    g->source_id = src;
    g->lam_stratum = ls;
    g->pol_stratum = ps;
    g->cap = 4096;
    g->pos = (double *)malloc((size_t)g->cap * 3 * sizeof(double));
    g->dir = (double *)malloc((size_t)g->cap * 3 * sizeof(double));
    g->s_hat = (double *)malloc((size_t)g->cap * 3 * sizeof(double));
    g->Es = (kcplx *)malloc((size_t)g->cap * sizeof(kcplx));
    g->Ep = (kcplx *)malloc((size_t)g->cap * sizeof(kcplx));
    g->lam = (double *)malloc((size_t)g->cap * sizeof(double));
    g->opl = (double *)malloc((size_t)g->cap * sizeof(double));
    g->power = (double *)malloc((size_t)g->cap * sizeof(double));
    g->scattered = (uint8_t *)malloc((size_t)g->cap);
    g->ray_key = (uint64_t *)malloc((size_t)g->cap * sizeof(uint64_t));
    g->event_ctr = (uint32_t *)malloc((size_t)g->cap * sizeof(uint32_t));
    if (!g->pos || !g->dir || !g->s_hat || !g->Es || !g->Ep || !g->lam
            || !g->opl || !g->power || !g->scattered || !g->ray_key
            || !g->event_ctr)
        die(EXIT_PHYSICS, "detector: gkey sample allocation failed");
    return g;
}

static void gkey_grow(GKey *g) {
    int64_t cap = g->cap * 2;
    g->pos = (double *)realloc(g->pos, (size_t)cap * 3 * sizeof(double));
    g->dir = (double *)realloc(g->dir, (size_t)cap * 3 * sizeof(double));
    g->s_hat = (double *)realloc(g->s_hat,
                                 (size_t)cap * 3 * sizeof(double));
    g->Es = (kcplx *)realloc(g->Es, (size_t)cap * sizeof(kcplx));
    g->Ep = (kcplx *)realloc(g->Ep, (size_t)cap * sizeof(kcplx));
    g->lam = (double *)realloc(g->lam, (size_t)cap * sizeof(double));
    g->opl = (double *)realloc(g->opl, (size_t)cap * sizeof(double));
    g->power = (double *)realloc(g->power, (size_t)cap * sizeof(double));
    g->scattered = (uint8_t *)realloc(g->scattered, (size_t)cap);
    g->ray_key = (uint64_t *)realloc(g->ray_key,
                                     (size_t)cap * sizeof(uint64_t));
    g->event_ctr = (uint32_t *)realloc(g->event_ctr,
                                       (size_t)cap * sizeof(uint32_t));
    if (!g->pos || !g->dir || !g->s_hat || !g->Es || !g->Ep || !g->lam
            || !g->opl || !g->power || !g->scattered || !g->ray_key
            || !g->event_ctr)
        die(EXIT_PHYSICS, "detector: gather sample growth to %lld failed "
            "— out of memory; reduce --rays", (long long)cap);
    g->cap = cap;
}

void det_apply_gather_hits(SceneC *s, const GatherHitVec *hits) {
    for (int64_t i = 0; i < hits->n; i++) {
        const GatherHit *h = &hits->v[i];
        DetC *d = &s->dets[h->det];
        GKey *g = det_gkey(d, h->source_id, h->lam_stratum,
                           h->pol_stratum);
        if (g->n == g->cap) gkey_grow(g);
        int64_t j = g->n++;
        g->pos[j * 3] = h->pos.x;
        g->pos[j * 3 + 1] = h->pos.y;
        g->pos[j * 3 + 2] = h->pos.z;
        g->dir[j * 3] = h->dir.x;
        g->dir[j * 3 + 1] = h->dir.y;
        g->dir[j * 3 + 2] = h->dir.z;
        g->s_hat[j * 3] = h->s_hat.x;
        g->s_hat[j * 3 + 1] = h->s_hat.y;
        g->s_hat[j * 3 + 2] = h->s_hat.z;
        g->Es[j] = h->Es;
        g->Ep[j] = h->Ep;
        g->lam[j] = h->lam;
        g->opl[j] = h->opl;
        g->power[j] = h->power;
        g->scattered[j] = h->scattered;
        g->ray_key[j] = h->ray_key;
        g->event_ctr[j] = h->event_ctr;
        size_t key = ((size_t)h->source_id * s->max_strata
                      + h->lam_stratum) * s->max_pol + h->pol_stratum;
        d->det_geom_W[key] += h->power;
    }
}

void det_free_gkeys(DetC *d) {
    for (int32_t i = 0; i < d->n_gkeys; i++) {
        GKey *g = &d->gkeys[i];
        free(g->pos); free(g->dir); free(g->s_hat);
        free(g->Es); free(g->Ep);
        free(g->lam); free(g->opl); free(g->power);
        free(g->scattered); free(g->ray_key); free(g->event_ctr);
    }
    free(d->gkeys);
    d->gkeys = NULL;
    d->n_gkeys = d->cap_gkeys = 0;
}

/* P1: serialize coherent sample sets for the Python driver's chunk
 * accumulation. Layout under out_dir: gkeys.json manifest + per-key SoA
 * .npy arrays. The Python side (cengine._merge_chunk) reads these into
 * DetectorGrid .samples records, sorts by (ray_key,event_ctr), writes the
 * MERGED dump back in the same layout, and the binary re-loads it via
 * det_load_gather_state (gather_only mode) for the single final gather. */
void det_dump_gkeys(const SceneC *s) {
    char path[1300];
    /* manifest keyed by detector index -> [[src,ls,ps,n], ...] */
    snprintf(path, sizeof path, "%s/gkeys.json", s->out_dir);
    FILE *mf = fopen(path, "w");
    if (!mf) die(EXIT_PHYSICS, "detector: cannot write %s", path);
    fprintf(mf, "{");
    int first_det = 1;
    for (int i = 0; i < s->n_dets; i++) {
        const DetC *d = &s->dets[i];
        fprintf(mf, "%s\"%d\":[", first_det ? "" : ",", i);
        first_det = 0;
        for (int32_t k = 0; k < d->n_gkeys; k++) {
            const GKey *g = &d->gkeys[k];
            fprintf(mf, "%s[%d,%d,%d,%lld]", k ? "," : "",
                    g->source_id, g->lam_stratum, g->pol_stratum,
                    (long long)g->n);
            size_t n = (size_t)g->n;
            /* Es/Ep are kcplx (== C _Complex double, layout of complex128) */
            snprintf(path, sizeof path, "%s/gk_%d_%d_%d_%d_pos.npy",
                     s->out_dir, i, g->source_id, g->lam_stratum,
                     g->pol_stratum);
            { size_t sh[2] = {n, 3}; npy_write(path, g->pos, "<f8", 2, sh); }
            snprintf(path, sizeof path, "%s/gk_%d_%d_%d_%d_dir.npy",
                     s->out_dir, i, g->source_id, g->lam_stratum,
                     g->pol_stratum);
            { size_t sh[2] = {n, 3}; npy_write(path, g->dir, "<f8", 2, sh); }
            snprintf(path, sizeof path, "%s/gk_%d_%d_%d_%d_shat.npy",
                     s->out_dir, i, g->source_id, g->lam_stratum,
                     g->pol_stratum);
            { size_t sh[2] = {n, 3}; npy_write(path, g->s_hat, "<f8", 2, sh); }
            snprintf(path, sizeof path, "%s/gk_%d_%d_%d_%d_Es.npy",
                     s->out_dir, i, g->source_id, g->lam_stratum,
                     g->pol_stratum);
            npy_write(path, g->Es, "<c16", 1, &n);
            snprintf(path, sizeof path, "%s/gk_%d_%d_%d_%d_Ep.npy",
                     s->out_dir, i, g->source_id, g->lam_stratum,
                     g->pol_stratum);
            npy_write(path, g->Ep, "<c16", 1, &n);
            snprintf(path, sizeof path, "%s/gk_%d_%d_%d_%d_lam.npy",
                     s->out_dir, i, g->source_id, g->lam_stratum,
                     g->pol_stratum);
            npy_write(path, g->lam, "<f8", 1, &n);
            snprintf(path, sizeof path, "%s/gk_%d_%d_%d_%d_opl.npy",
                     s->out_dir, i, g->source_id, g->lam_stratum,
                     g->pol_stratum);
            npy_write(path, g->opl, "<f8", 1, &n);
            snprintf(path, sizeof path, "%s/gk_%d_%d_%d_%d_power.npy",
                     s->out_dir, i, g->source_id, g->lam_stratum,
                     g->pol_stratum);
            npy_write(path, g->power, "<f8", 1, &n);
            snprintf(path, sizeof path, "%s/gk_%d_%d_%d_%d_scat.npy",
                     s->out_dir, i, g->source_id, g->lam_stratum,
                     g->pol_stratum);
            npy_write(path, g->scattered, "|u1", 1, &n);
            snprintf(path, sizeof path, "%s/gk_%d_%d_%d_%d_key.npy",
                     s->out_dir, i, g->source_id, g->lam_stratum,
                     g->pol_stratum);
            npy_write(path, g->ray_key, "<u8", 1, &n);
            snprintf(path, sizeof path, "%s/gk_%d_%d_%d_%d_evt.npy",
                     s->out_dir, i, g->source_id, g->lam_stratum,
                     g->pol_stratum);
            npy_write(path, g->event_ctr, "<u4", 1, &n);
        }
        fprintf(mf, "]");
    }
    fprintf(mf, "}\n");
    if (fclose(mf) != 0)
        die(EXIT_PHYSICS, "detector: short write to gkeys.json");
}

/* P1 gather_only: the symmetric loader. Reads the Python driver's MERGED
 * dump from s->gather_input into the DetC/GKey structures the normal
 * gather_run consumes:
 *   gkeys.json + gk_<det>_<src>_<ls>_<ps>_<field>.npy  (det_dump_gkeys
 *     layout; samples arrive canonically sorted by (ray_key,event_ctr))
 *   acc_<det>_inc.npy      merged incoherent cube snapshot (pre-gather)
 *   acc_<det>_tinc_W.npy / acc_<det>_tinc_n.npy   per-key incoherent
 *     tallies, flat [n_sources][max_strata][max_pol]
 * det_geom_W is recomputed from the loaded sample powers (the same values
 * the dump carries), so detected.json/gather.json stay complete. */
static void *load_gk(const char *in, int di, int src, int ls, int ps,
                     const char *field, const char *dtype, size_t expect) {
    char path[1400];
    snprintf(path, sizeof path, "%s/gk_%d_%d_%d_%d_%s.npy",
             in, di, src, ls, ps, field);
    size_t n = 0;
    void *buf = npy_read(path, dtype, &n);
    if (n != expect)
        die(EXIT_INPUT, "gather_only: %s has %zu elements, manifest says "
            "%zu", path, n, expect);
    return buf;
}

void det_load_gather_state(SceneC *s) {
    const char *in = s->gather_input;
    char path[1400];
    size_t tally = (size_t)s->n_sources * s->max_strata * s->max_pol;

    /* accumulator snapshots (a detector with no incoherent arrivals may
     * legitimately have an all-zero snapshot — file still required, the
     * Python driver always writes it) */
    for (int i = 0; i < s->n_dets; i++) {
        DetC *d = &s->dets[i];
        size_t n = 0;
        snprintf(path, sizeof path, "%s/acc_%d_inc.npy", in, i);
        double *inc = (double *)npy_read(path, "<f8", &n);
        size_t want = (size_t)d->spectral_bins * d->H * d->W;
        if (n != want)
            die(EXIT_INPUT, "gather_only: %s has %zu elements, detector "
                "cube wants %zu", path, n, want);
        memcpy(d->inc, inc, want * sizeof(double));
        free(inc);
        snprintf(path, sizeof path, "%s/acc_%d_tinc_W.npy", in, i);
        double *tw = (double *)npy_read(path, "<f8", &n);
        if (n != tally)
            die(EXIT_INPUT, "gather_only: %s has %zu tallies, want %zu",
                path, n, tally);
        memcpy(d->det_inc_W, tw, tally * sizeof(double));
        free(tw);
        snprintf(path, sizeof path, "%s/acc_%d_tinc_n.npy", in, i);
        int64_t *tn = (int64_t *)npy_read(path, "<i8", &n);
        if (n != tally)
            die(EXIT_INPUT, "gather_only: %s has %zu tallies, want %zu",
                path, n, tally);
        memcpy(d->det_inc_n, tn, tally * sizeof(int64_t));
        free(tn);
    }

    /* coherent sample sets from the manifest */
    snprintf(path, sizeof path, "%s/gkeys.json", in);
    yyjson_doc *doc = yyjson_read_file(path, 0, NULL, NULL);
    if (!doc)
        die(EXIT_INPUT, "gather_only: cannot parse %s", path);
    yyjson_val *root = yyjson_doc_get_root(doc);
    for (int i = 0; i < s->n_dets; i++) {
        DetC *d = &s->dets[i];
        char key[16];
        snprintf(key, sizeof key, "%d", i);
        yyjson_val *arr = yyjson_obj_get(root, key);
        if (!arr || !yyjson_is_arr(arr))
            continue;
        size_t ki, kmax;
        yyjson_val *entry;
        yyjson_arr_foreach(arr, ki, kmax, entry) {
            if (!yyjson_is_arr(entry) || yyjson_arr_size(entry) != 4)
                die(EXIT_INPUT, "gather_only: bad gkeys.json entry for "
                    "detector %d", i);
            int src = (int)yyjson_get_sint(yyjson_arr_get(entry, 0));
            int ls = (int)yyjson_get_sint(yyjson_arr_get(entry, 1));
            int ps = (int)yyjson_get_sint(yyjson_arr_get(entry, 2));
            int64_t n = yyjson_get_sint(yyjson_arr_get(entry, 3));
            if (n <= 0)
                continue;
            GKey *g = det_gkey(d, (int16_t)src, (int16_t)ls, (int16_t)ps);
            while (g->cap < n) gkey_grow(g);
            size_t un = (size_t)n;
            void *buf;
            buf = load_gk(in, i, src, ls, ps, "pos", "<f8", un * 3);
            memcpy(g->pos, buf, un * 3 * sizeof(double)); free(buf);
            buf = load_gk(in, i, src, ls, ps, "dir", "<f8", un * 3);
            memcpy(g->dir, buf, un * 3 * sizeof(double)); free(buf);
            buf = load_gk(in, i, src, ls, ps, "shat", "<f8", un * 3);
            memcpy(g->s_hat, buf, un * 3 * sizeof(double)); free(buf);
            buf = load_gk(in, i, src, ls, ps, "Es", "<c16", un);
            memcpy(g->Es, buf, un * sizeof(kcplx)); free(buf);
            buf = load_gk(in, i, src, ls, ps, "Ep", "<c16", un);
            memcpy(g->Ep, buf, un * sizeof(kcplx)); free(buf);
            buf = load_gk(in, i, src, ls, ps, "lam", "<f8", un);
            memcpy(g->lam, buf, un * sizeof(double)); free(buf);
            buf = load_gk(in, i, src, ls, ps, "opl", "<f8", un);
            memcpy(g->opl, buf, un * sizeof(double)); free(buf);
            buf = load_gk(in, i, src, ls, ps, "power", "<f8", un);
            memcpy(g->power, buf, un * sizeof(double)); free(buf);
            buf = load_gk(in, i, src, ls, ps, "scat", "|u1", un);
            memcpy(g->scattered, buf, un); free(buf);
            buf = load_gk(in, i, src, ls, ps, "key", "<u8", un);
            memcpy(g->ray_key, buf, un * sizeof(uint64_t)); free(buf);
            buf = load_gk(in, i, src, ls, ps, "evt", "<u4", un);
            memcpy(g->event_ctr, buf, un * sizeof(uint32_t)); free(buf);
            g->n = n;
            /* rebuild the geometric tally from the loaded powers (same
             * values the dump carries; detected.json stays complete) */
            size_t tkey = ((size_t)src * s->max_strata + ls)
                          * s->max_pol + ps;
            double w = 0.0;
            for (int64_t j = 0; j < n; j++) w += g->power[j];
            d->det_geom_W[tkey] += w;
        }
    }
    yyjson_doc_free(doc);
    LOGI("gather_only: loaded merged samples + accumulator snapshots "
         "from %s", in);
}

/* ---------------------------------------------------------------- mask */
void det_compute_mask(DetC *d, const SceneC *s) {
    const FaceC *face = &s->faces[d->face_id];
    if (face->surf.kind != SURF_PLANE)
        die(EXIT_INPUT, "detector '%s': face %s is not planar (curved "
            "detectors are routed to the Python engine)", d->label,
            face->id);
    /* pixel centers, exactly detector.py:148-162: world = gx*xhat +
     * gy*yhat + (origin minus its (xhat,yhat) components) */
    kvec3 origin = face->surf.u.plane.origin;
    kvec3 n_comp = origin;
    n_comp = v3_sub(n_comp, v3_scale(d->xhat, v3_dot(origin, d->xhat)));
    n_comp = v3_sub(n_comp, v3_scale(d->yhat, v3_dot(origin, d->yhat)));
    for (int32_t iy = 0; iy < d->H; iy++) {
        double gy = d->y_lo + (iy + 0.5) * d->pixel_m;
        for (int32_t ix = 0; ix < d->W; ix++) {
            double gx = d->x_lo + (ix + 0.5) * d->pixel_m;
            kvec3 p = n_comp;
            p = v3_fma(p, gx, d->xhat);
            p = v3_fma(p, gy, d->yhat);
            double u, v;
            surf_to_uv(&face->surf, p, &u, &v);
            d->mask[(size_t)iy * d->W + ix] =
                (uint8_t)trim_contains(&face->trim, u, v);
        }
    }
}

/* ------------------------------------------------------------- splat */
void det_apply_hits(SceneC *s, const DetHitVec *hits) {
    for (int64_t i = 0; i < hits->n; i++) {
        const DetHit *h = &hits->v[i];
        DetC *d = &s->dets[h->det];
        /* deposit_incoherent math (detector.py:53-69): center-offset then
         * bilinear over the 4 surrounding pixels, bounds-checked */
        double fx = (double)h->fx - 0.5;
        double fy = (double)h->fy - 0.5;
        double x0 = floor(fx), y0 = floor(fy);
        double wx = fx - x0, wy = fy - y0;
        int ix0 = (int)x0, iy0 = (int)y0;
        double *plane = d->inc + (size_t)h->bin * d->H * d->W;
        const double w[4] = {(1 - wx) * (1 - wy), wx * (1 - wy),
                             (1 - wx) * wy, wx * wy};
        const int ox[4] = {0, 1, 0, 1};
        const int oy[4] = {0, 0, 1, 1};
        for (int c = 0; c < 4; c++) {
            int xi = ix0 + ox[c], yi = iy0 + oy[c];
            if (xi >= 0 && xi < d->W && yi >= 0 && yi < d->H)
                plane[(size_t)yi * d->W + xi] += h->power * w[c];
        }
        /* per-(source, lam_stratum, pol_stratum) tallies
         * (detector.py:70-85) */
        size_t key = ((size_t)h->source_id * s->max_strata
                      + h->lam_stratum) * s->max_pol + h->pol_stratum;
        d->det_inc_W[key] += h->power;
        d->det_inc_n[key] += 1;
    }
}

/* ------------------------------------------------------------- output */
void det_write_outputs(const SceneC *s) {
    char path[1200];
    for (int i = 0; i < s->n_dets; i++) {
        const DetC *d = &s->dets[i];
        snprintf(path, sizeof path, "%s/det_%d_inc.npy", s->out_dir, i);
        npy_write_f64_3d(path, d->inc, (size_t)d->spectral_bins,
                         (size_t)d->H, (size_t)d->W);
        snprintf(path, sizeof path, "%s/det_%d_mask.npy", s->out_dir, i);
        npy_write_u8_2d(path, d->mask, (size_t)d->H, (size_t)d->W);
    }
    /* detected.json: per detector, per (source, stratum, pol) tallies in
     * the "src/lam/pol" key shape build_detected_block uses
     * (run_trace.py:422-451) */
    snprintf(path, sizeof path, "%s/detected.json", s->out_dir);
    FILE *f = fopen(path, "w");
    if (!f) die(EXIT_PHYSICS, "detector: cannot write %s", path);
    fprintf(f, "{\n");
    for (int i = 0; i < s->n_dets; i++) {
        const DetC *d = &s->dets[i];
        fprintf(f, "  \"%s\": {", d->label);
        int first = 1;
        for (int src = 0; src < s->n_sources; src++)
            for (int ls = 0; ls < s->max_strata; ls++)
                for (int ps = 0; ps < s->max_pol; ps++) {
                    size_t key = ((size_t)src * s->max_strata + ls)
                                 * s->max_pol + ps;
                    int has_inc = d->det_inc_n[key] != 0;
                    int has_coh = d->det_geom_W[key] != 0.0;
                    if (!has_inc && !has_coh) continue;
                    fprintf(f, "%s\n    \"%d/%d/%d\": {", first ? "" : ",",
                            src, ls, ps);
                    if (has_inc)
                        fprintf(f, "\"incoherent_W\": %.17g, \"n\": %lld%s",
                                d->det_inc_W[key],
                                (long long)d->det_inc_n[key],
                                has_coh ? ", " : "");
                    if (has_coh)
                        fprintf(f, "\"coherent_W\": %.17g",
                                d->det_geom_W[key]);
                    fprintf(f, "}");
                    first = 0;
                }
        fprintf(f, "\n  }%s\n", i + 1 < s->n_dets ? "," : "");
    }
    fprintf(f, "}\n");
    if (fclose(f) != 0)
        die(EXIT_PHYSICS, "detector: short write to %s", path);
}
