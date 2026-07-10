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
                    if (d->det_inc_n[key] == 0) continue;
                    fprintf(f, "%s\n    \"%d/%d/%d\": {\"incoherent_W\": "
                            "%.17g, \"n\": %lld}", first ? "" : ",",
                            src, ls, ps, d->det_inc_W[key],
                            (long long)d->det_inc_n[key]);
                    first = 0;
                }
        fprintf(f, "\n  }%s\n", i + 1 < s->n_dets ? "," : "");
    }
    fprintf(f, "}\n");
    if (fclose(f) != 0)
        die(EXIT_PHYSICS, "detector: short write to %s", path);
}
