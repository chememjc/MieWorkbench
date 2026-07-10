/* raybuf.c — growable ray arenas (see raybuf.h). */
#include "raybuf.h"
#include "log.h"

#include <stdlib.h>
#include <string.h>

/* Hard cap: 256M rays * ~200 B = ~50 GB, far beyond the 62 GB host — any
 * arena reaching this is a runaway split cascade, not a real workload. */
#define RAYVEC_HARD_CAP (256ll << 20)

void rayvec_init(RayVec *v, int64_t cap) {
    if (cap < 16) cap = 16;
    v->rays = (Ray *)malloc((size_t)cap * sizeof(Ray));
    if (!v->rays)
        die(EXIT_PHYSICS, "rayvec: allocation of %lld rays (%.1f MB) failed",
            (long long)cap, (double)cap * sizeof(Ray) / 1e6);
    v->n = 0;
    v->cap = cap;
}

void rayvec_free(RayVec *v) {
    free(v->rays);
    v->rays = NULL;
    v->n = v->cap = 0;
}

static void rayvec_grow(RayVec *v, int64_t need) {
    int64_t cap = v->cap;
    while (cap < need) cap *= 2;
    if (cap > RAYVEC_HARD_CAP)
        die(EXIT_PHYSICS,
            "rayvec: child buffer would exceed %lld rays — runaway ray "
            "splitting; reduce --rays or --max-reflections",
            (long long)RAYVEC_HARD_CAP);
    Ray *p = (Ray *)realloc(v->rays, (size_t)cap * sizeof(Ray));
    if (!p)
        die(EXIT_PHYSICS, "rayvec: growth to %lld rays (%.1f MB) failed — "
            "out of memory; reduce --rays",
            (long long)cap, (double)cap * sizeof(Ray) / 1e6);
    v->rays = p;
    v->cap = cap;
}

void rayvec_push(RayVec *v, const Ray *r) {
    if (v->n == v->cap) rayvec_grow(v, v->n + 1);
    v->rays[v->n++] = *r;
}

void rayvec_clear(RayVec *v) { v->n = 0; }

void rayvec_extend(RayVec *dst, const RayVec *src) {
    if (src->n == 0) return;
    if (dst->n + src->n > dst->cap) rayvec_grow(dst, dst->n + src->n);
    memcpy(dst->rays + dst->n, src->rays, (size_t)src->n * sizeof(Ray));
    dst->n += src->n;
}
