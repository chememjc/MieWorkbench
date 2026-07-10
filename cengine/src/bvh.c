/* bvh.c — median-split builder (port of mesh.py BVH.__init__:161-216). */
#include "bvh.h"
#include "log.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    double key;
    int32_t idx;
} SortItem;

static int sort_cmp(const void *a, const void *b) {
    const SortItem *x = (const SortItem *)a, *y = (const SortItem *)b;
    if (x->key < y->key) return -1;
    if (x->key > y->key) return 1;
    /* stable tie-break on original index (numpy kind="stable") */
    return (x->idx < y->idx) ? -1 : (x->idx > y->idx);
}

void bvh_build(BvhC *b, const kvec3 *lo, const kvec3 *hi, int32_t n,
               int leaf_size) {
    b->n_items = n;
    /* worst case 2n-1 nodes for leaf_size >= 1 */
    int32_t cap = n > 0 ? 2 * n : 1;
    b->nodes = (BvhNode *)malloc((size_t)cap * sizeof(BvhNode));
    b->order = (int32_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
    if (!b->nodes || !b->order)
        die(EXIT_PHYSICS, "bvh: allocation failed (%d items)", n);
    b->n_nodes = 0;

    kvec3 *cen = (kvec3 *)malloc((size_t)(n > 0 ? n : 1) * sizeof(kvec3));
    int32_t *idxbuf = (int32_t *)malloc(
        (size_t)(n > 0 ? n : 1) * sizeof(int32_t));
    if (!cen || !idxbuf) die(EXIT_PHYSICS, "bvh: allocation failed");
    for (int32_t i = 0; i < n; i++) {
        cen[i] = v3_scale(v3_add(lo[i], hi[i]), 0.5);
        idxbuf[i] = i;
    }

    /* explicit stack of (node, [start, end) into idxbuf) */
    typedef struct { int32_t node, s, e; } Frame;
    Frame *stack = (Frame *)malloc((size_t)(2 * (n > 0 ? n : 1) + 8)
                                   * sizeof(Frame));
    if (!stack) die(EXIT_PHYSICS, "bvh: allocation failed");
    int sp = 0;
    int32_t order_at = 0;

    int32_t root = b->n_nodes++;
    stack[sp].node = root;
    stack[sp].s = 0;
    stack[sp].e = n;
    sp++;

    while (sp > 0) {
        Frame f = stack[--sp];
        BvhNode *node = &b->nodes[f.node];
        kvec3 bmin = v3(INFINITY, INFINITY, INFINITY);
        kvec3 bmax = v3(-INFINITY, -INFINITY, -INFINITY);
        for (int32_t i = f.s; i < f.e; i++) {
            int32_t id = idxbuf[i];
            if (lo[id].x < bmin.x) bmin.x = lo[id].x;
            if (lo[id].y < bmin.y) bmin.y = lo[id].y;
            if (lo[id].z < bmin.z) bmin.z = lo[id].z;
            if (hi[id].x > bmax.x) bmax.x = hi[id].x;
            if (hi[id].y > bmax.y) bmax.y = hi[id].y;
            if (hi[id].z > bmax.z) bmax.z = hi[id].z;
        }
        node->bbmin = bmin;
        node->bbmax = bmax;
        int32_t cnt = f.e - f.s;
        if (cnt <= leaf_size) {
            node->left = node->right = -1;
            node->start = order_at;
            node->count = cnt;
            for (int32_t i = f.s; i < f.e; i++)
                b->order[order_at++] = idxbuf[i];
            continue;
        }
        /* widest centroid axis, median split (mesh.py:194-204) */
        kvec3 cmin = v3(INFINITY, INFINITY, INFINITY);
        kvec3 cmax = v3(-INFINITY, -INFINITY, -INFINITY);
        for (int32_t i = f.s; i < f.e; i++) {
            kvec3 c = cen[idxbuf[i]];
            if (c.x < cmin.x) cmin.x = c.x;
            if (c.y < cmin.y) cmin.y = c.y;
            if (c.z < cmin.z) cmin.z = c.z;
            if (c.x > cmax.x) cmax.x = c.x;
            if (c.y > cmax.y) cmax.y = c.y;
            if (c.z > cmax.z) cmax.z = c.z;
        }
        double ex = cmax.x - cmin.x, ey = cmax.y - cmin.y,
               ez = cmax.z - cmin.z;
        int ax = (ex >= ey && ex >= ez) ? 0 : (ey >= ez ? 1 : 2);
        /* sort the range by centroid[ax] (stable) and split at the
         * median position — the halved fallback and the <=median split
         * coincide within a sorted range, so one code path covers both */
        SortItem *items = (SortItem *)malloc((size_t)cnt
                                             * sizeof(SortItem));
        if (!items) die(EXIT_PHYSICS, "bvh: allocation failed");
        for (int32_t i = 0; i < cnt; i++) {
            kvec3 c = cen[idxbuf[f.s + i]];
            items[i].key = ax == 0 ? c.x : (ax == 1 ? c.y : c.z);
            items[i].idx = idxbuf[f.s + i];
        }
        qsort(items, (size_t)cnt, sizeof(SortItem), sort_cmp);
        /* split "<= median": count items with key <= median value */
        double med = (cnt & 1) ? items[cnt / 2].key
                   : 0.5 * (items[cnt / 2 - 1].key + items[cnt / 2].key);
        int32_t nleft = 0;
        while (nleft < cnt && items[nleft].key <= med) nleft++;
        if (nleft == 0 || nleft == cnt) nleft = cnt / 2;   /* degenerate */
        for (int32_t i = 0; i < cnt; i++)
            idxbuf[f.s + i] = items[i].idx;
        free(items);

        int32_t li = b->n_nodes++;
        int32_t ri = b->n_nodes++;
        if (b->n_nodes > cap)
            die(EXIT_PHYSICS, "bvh: node overflow (internal error)");
        node->left = li;
        node->right = ri;
        node->start = -1;
        node->count = 0;
        stack[sp].node = li; stack[sp].s = f.s; stack[sp].e = f.s + nleft;
        sp++;
        stack[sp].node = ri; stack[sp].s = f.s + nleft; stack[sp].e = f.e;
        sp++;
    }
    free(stack);
    free(cen);
    free(idxbuf);
}

void bvh_free(BvhC *b) {
    free(b->nodes);
    free(b->order);
    memset(b, 0, sizeof *b);
}
