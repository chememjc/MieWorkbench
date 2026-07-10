/* detector.c — see detector.h. */
#include "detector.h"
#include "log.h"
#include "npyio.h"

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
    if (!g->pos || !g->dir || !g->s_hat || !g->Es || !g->Ep || !g->lam
            || !g->opl || !g->power || !g->scattered || !g->ray_key)
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
    if (!g->pos || !g->dir || !g->s_hat || !g->Es || !g->Ep || !g->lam
            || !g->opl || !g->power || !g->scattered || !g->ray_key)
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
        free(g->scattered); free(g->ray_key);
    }
    free(d->gkeys);
    d->gkeys = NULL;
    d->n_gkeys = d->cap_gkeys = 0;
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
