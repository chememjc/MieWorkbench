/* ===========================================================================
 * bvh.h — median-split AABB BVH, shared by the scene-level TLAS (leaves =
 * face indices) and the per-mesh-face triangle BLAS (leaves = triangles).
 *
 * Build is a port of mesh.py BVH.__init__ (surfaces median-split over
 * centroids, iterative, degenerate split halved by stable sort order);
 * traversal is per-ray scalar with nearest-t pruning (the batched numpy
 * traversal in mesh.py:219-253 collapses to a simple stack walk per ray).
 * =========================================================================== */
#ifndef MIEWB_BVH_H
#define MIEWB_BVH_H

#include <stdint.h>
#include "kernels/kmath.h"

typedef struct {
    kvec3 bbmin, bbmax;
    int32_t left, right;    /* -1 = leaf */
    int32_t start, count;   /* leaf: range into order[] */
} BvhNode;

typedef struct {
    BvhNode *nodes;
    int32_t n_nodes;
    int32_t *order;         /* item ids in leaf order */
    int32_t n_items;
} BvhC;

/* Build from per-item AABBs (item centroid = box center). leaf_size:
 * mesh.py uses 8 for triangles; the TLAS uses 2 (faces are expensive to
 * test — asphere Newton — so split fine). */
void bvh_build(BvhC *b, const kvec3 *lo, const kvec3 *hi, int32_t n,
               int leaf_size);
void bvh_free(BvhC *b);

/* slab test (mesh.py _ray_box, scalar): hit iff
 * tmax >= max(tmin, t_eps) and tmin <= best_t. inv = 1/d componentwise
 * (IEEE inf for zero components handles axis-parallel rays). */
static inline int bvh_ray_box(kvec3 o, kvec3 inv, double best_t,
                              kvec3 bbmin, kvec3 bbmax, double t_eps) {
    double t1, t2, tmin, tmax, lo, hi;
    t1 = (bbmin.x - o.x) * inv.x;
    t2 = (bbmax.x - o.x) * inv.x;
    tmin = t1 < t2 ? t1 : t2;
    tmax = t1 < t2 ? t2 : t1;
    t1 = (bbmin.y - o.y) * inv.y;
    t2 = (bbmax.y - o.y) * inv.y;
    lo = t1 < t2 ? t1 : t2;
    hi = t1 < t2 ? t2 : t1;
    if (lo > tmin) tmin = lo;
    if (hi < tmax) tmax = hi;
    t1 = (bbmin.z - o.z) * inv.z;
    t2 = (bbmax.z - o.z) * inv.z;
    lo = t1 < t2 ? t1 : t2;
    hi = t1 < t2 ? t2 : t1;
    if (lo > tmin) tmin = lo;
    if (hi < tmax) tmax = hi;
    double floor_t = tmin > t_eps ? tmin : t_eps;
    return tmax >= floor_t && tmin <= best_t;
}

#define BVH_STACK_MAX 128

#endif /* MIEWB_BVH_H */
