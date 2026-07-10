/* mesh.c — see mesh.h. Ports mesh.py with the same constants:
 * weld grid 1e-9 m, BLAS leaf 8, t_eps 1e-7, MT epsilons 1e-12/1e-300. */
#include "mesh.h"
#include "log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WELD_GRID 1e-9
#define MESH_LEAF 8
#define MESH_T_EPS 1e-7

/* ------------------------------------------------------------- STL read */
typedef struct {
    int64_t qx, qy, qz;
    int32_t orig;
} WeldKey;

static int weld_cmp(const void *a, const void *b) {
    const WeldKey *x = (const WeldKey *)a, *y = (const WeldKey *)b;
    if (x->qx != y->qx) return x->qx < y->qx ? -1 : 1;
    if (x->qy != y->qy) return x->qy < y->qy ? -1 : 1;
    if (x->qz != y->qz) return x->qz < y->qz ? -1 : 1;
    return 0;
}

MeshC *mesh_load(const char *stl_path, int flat_normals,
                 const char *face_id) {
    FILE *f = fopen(stl_path, "rb");
    if (!f)
        die(EXIT_INPUT, "mesh face %s: cannot open STL %s", face_id,
            stl_path);
    uint8_t header[84];
    if (fread(header, 1, 84, f) != 84)
        die(EXIT_INPUT, "mesh face %s: STL %s truncated", face_id,
            stl_path);
    uint32_t count;
    memcpy(&count, header + 80, 4);
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    if (fsize != 84l + (long)count * 50l)
        die(EXIT_INPUT, "mesh face %s: %s is not a binary STL (the "
            "extractor writes binary; ASCII unsupported here)", face_id,
            stl_path);
    fseek(f, 84, SEEK_SET);

    /* raw facets: 12 f32 (normal + 3 verts) + u16 attr, packed 50 bytes */
    double *tris = (double *)malloc((size_t)count * 9 * sizeof(double));
    double *fnorm = (double *)malloc((size_t)count * 3 * sizeof(double));
    if (!tris || !fnorm)
        die(EXIT_INPUT, "mesh face %s: allocation failed (%u tris)",
            face_id, count);
    for (uint32_t i = 0; i < count; i++) {
        uint8_t rec[50];
        if (fread(rec, 1, 50, f) != 50)
            die(EXIT_INPUT, "mesh face %s: STL %s truncated at facet %u",
                face_id, stl_path, i);
        float v[12];
        memcpy(v, rec, 48);
        for (int k = 0; k < 3; k++) fnorm[i * 3 + k] = (double)v[k];
        for (int k = 0; k < 9; k++) tris[i * 9 + k] = (double)v[3 + k];
    }
    fclose(f);

    /* drop degenerates + align winding to the stored facet normals
     * (IndexedMesh, mesh.py:99-125) */
    MeshC *m = (MeshC *)calloc(1, sizeof(MeshC));
    if (!m) die(EXIT_INPUT, "mesh: allocation failed");
    m->flat_normals = (uint8_t)flat_normals;
    int32_t kept = 0;
    m->face_normals = (double *)malloc((size_t)count * 3 * sizeof(double));
    if (!m->face_normals) die(EXIT_INPUT, "mesh: allocation failed");
    for (uint32_t i = 0; i < count; i++) {
        kvec3 p0 = v3(tris[i * 9 + 0], tris[i * 9 + 1], tris[i * 9 + 2]);
        kvec3 p1 = v3(tris[i * 9 + 3], tris[i * 9 + 4], tris[i * 9 + 5]);
        kvec3 p2 = v3(tris[i * 9 + 6], tris[i * 9 + 7], tris[i * 9 + 8]);
        kvec3 gn = v3_cross(v3_sub(p1, p0), v3_sub(p2, p0));
        double gl = v3_norm(gn);
        if (gl <= 1e-20) continue;              /* zero-area: drop */
        kvec3 fn = v3(fnorm[i * 3], fnorm[i * 3 + 1], fnorm[i * 3 + 2]);
        if (v3_dot(gn, fn) < 0.0) {             /* flip winding */
            kvec3 tmp = p1;
            p1 = p2;
            p2 = tmp;
            gn = v3_scale(gn, -1.0);
        }
        double *dst = tris + (size_t)kept * 9;
        dst[0] = p0.x; dst[1] = p0.y; dst[2] = p0.z;
        dst[3] = p1.x; dst[4] = p1.y; dst[5] = p1.z;
        dst[6] = p2.x; dst[7] = p2.y; dst[8] = p2.z;
        kvec3 u = v3_scale(gn, 1.0 / gl);
        m->face_normals[kept * 3] = u.x;
        m->face_normals[kept * 3 + 1] = u.y;
        m->face_normals[kept * 3 + 2] = u.z;
        kept++;
    }
    if ((uint32_t)kept < count)
        LOGW("mesh face %s: dropped %u degenerate triangle(s)", face_id,
             count - (uint32_t)kept);
    if (kept == 0)
        die(EXIT_INPUT, "mesh face %s: STL has no valid triangles",
            face_id);
    m->n_tris = kept;
    m->tri_v = tris;
    free(fnorm);

    /* weld vertices onto the 1e-9 grid -> indexed mesh (mesh.py:127-134) */
    int32_t nv_raw = kept * 3;
    WeldKey *keys = (WeldKey *)malloc((size_t)nv_raw * sizeof(WeldKey));
    m->tri_vidx = (int32_t *)malloc((size_t)nv_raw * sizeof(int32_t));
    if (!keys || !m->tri_vidx) die(EXIT_INPUT, "mesh: allocation failed");
    for (int32_t i = 0; i < nv_raw; i++) {
        keys[i].qx = (int64_t)llround(tris[i * 3 + 0] / WELD_GRID);
        keys[i].qy = (int64_t)llround(tris[i * 3 + 1] / WELD_GRID);
        keys[i].qz = (int64_t)llround(tris[i * 3 + 2] / WELD_GRID);
        keys[i].orig = i;
    }
    qsort(keys, (size_t)nv_raw, sizeof(WeldKey), weld_cmp);
    m->verts = (double *)malloc((size_t)nv_raw * 3 * sizeof(double));
    if (!m->verts) die(EXIT_INPUT, "mesh: allocation failed");
    int32_t nv = 0;
    for (int32_t i = 0; i < nv_raw; i++) {
        if (i == 0 || weld_cmp(&keys[i], &keys[i - 1]) != 0) {
            m->verts[nv * 3] = tris[keys[i].orig * 3];
            m->verts[nv * 3 + 1] = tris[keys[i].orig * 3 + 1];
            m->verts[nv * 3 + 2] = tris[keys[i].orig * 3 + 2];
            nv++;
        }
        m->tri_vidx[keys[i].orig] = nv - 1;
    }
    m->n_verts = nv;
    free(keys);

    /* angle-weighted vertex normals (mesh.py:140-155) */
    m->vert_normals = (double *)calloc((size_t)nv * 3, sizeof(double));
    if (!m->vert_normals) die(EXIT_INPUT, "mesh: allocation failed");
    for (int32_t t = 0; t < m->n_tris; t++) {
        kvec3 fn = v3(m->face_normals[t * 3], m->face_normals[t * 3 + 1],
                      m->face_normals[t * 3 + 2]);
        for (int corner = 0; corner < 3; corner++) {
            int a = corner, b = (corner + 1) % 3, c = (corner + 2) % 3;
            const double *pa = m->verts + (size_t)m->tri_vidx[t * 3 + a] * 3;
            const double *pb = m->verts + (size_t)m->tri_vidx[t * 3 + b] * 3;
            const double *pc = m->verts + (size_t)m->tri_vidx[t * 3 + c] * 3;
            kvec3 u = v3_unit(v3(pb[0] - pa[0], pb[1] - pa[1],
                                 pb[2] - pa[2]));
            kvec3 w = v3_unit(v3(pc[0] - pa[0], pc[1] - pa[1],
                                 pc[2] - pa[2]));
            double cosang = v3_dot(u, w);
            if (cosang > 1.0) cosang = 1.0;
            if (cosang < -1.0) cosang = -1.0;
            double ang = acos(cosang);
            double *vn = m->vert_normals
                         + (size_t)m->tri_vidx[t * 3 + a] * 3;
            vn[0] += fn.x * ang;
            vn[1] += fn.y * ang;
            vn[2] += fn.z * ang;
        }
    }
    for (int32_t i = 0; i < nv; i++) {
        kvec3 n = v3(m->vert_normals[i * 3], m->vert_normals[i * 3 + 1],
                     m->vert_normals[i * 3 + 2]);
        double l = v3_norm(n);
        if (l > 1e-300) {
            m->vert_normals[i * 3] = n.x / l;
            m->vert_normals[i * 3 + 1] = n.y / l;
            m->vert_normals[i * 3 + 2] = n.z / l;
        }
    }

    /* triangle BLAS */
    kvec3 *lo = (kvec3 *)malloc((size_t)kept * sizeof(kvec3));
    kvec3 *hi = (kvec3 *)malloc((size_t)kept * sizeof(kvec3));
    if (!lo || !hi) die(EXIT_INPUT, "mesh: allocation failed");
    for (int32_t t = 0; t < kept; t++) {
        const double *v = tris + (size_t)t * 9;
        kvec3 a = v3(v[0], v[1], v[2]);
        kvec3 bb = v3(v[3], v[4], v[5]);
        kvec3 c = v3(v[6], v[7], v[8]);
        lo[t] = v3(fmin(a.x, fmin(bb.x, c.x)), fmin(a.y, fmin(bb.y, c.y)),
                   fmin(a.z, fmin(bb.z, c.z)));
        hi[t] = v3(fmax(a.x, fmax(bb.x, c.x)), fmax(a.y, fmax(bb.y, c.y)),
                   fmax(a.z, fmax(bb.z, c.z)));
    }
    bvh_build(&m->bvh, lo, hi, kept, MESH_LEAF);
    free(lo);
    free(hi);
    return m;
}

void mesh_free(MeshC *m) {
    if (!m) return;
    free(m->tri_v);
    free(m->face_normals);
    free(m->tri_vidx);
    free(m->verts);
    free(m->vert_normals);
    bvh_free(&m->bvh);
    free(m);
}

/* -------------------------------------------------- Moller-Trumbore hit */
/* mesh.py _leaf_mt epsilons: |a| > 1e-300, u,v >= -1e-12,
 * u+v <= 1+1e-12, t > t_eps. Backface culling OFF. */
static double tri_mt(const double *v, kvec3 o, kvec3 d) {
    kvec3 v0 = v3(v[0], v[1], v[2]);
    kvec3 e1 = v3(v[3] - v[0], v[4] - v[1], v[5] - v[2]);
    kvec3 e2 = v3(v[6] - v[0], v[7] - v[1], v[8] - v[2]);
    kvec3 h = v3_cross(d, e2);
    double a = v3_dot(e1, h);
    if (fabs(a) <= 1e-300) return INFINITY;
    double f = 1.0 / a;
    kvec3 sv = v3_sub(o, v0);
    double u = f * v3_dot(sv, h);
    if (u < -1e-12) return INFINITY;
    kvec3 q = v3_cross(sv, e1);
    double w = f * v3_dot(d, q);
    if (w < -1e-12 || u + w > 1.0 + 1e-12) return INFINITY;
    double t = f * v3_dot(e2, q);
    return t > MESH_T_EPS ? t : INFINITY;
}

double mesh_intersect(const MeshC *m, kvec3 o, kvec3 d) {
    kvec3 inv = v3(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
    double best_t = INFINITY;
    int32_t stack[BVH_STACK_MAX];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const BvhNode *nd = &m->bvh.nodes[stack[--sp]];
        if (!bvh_ray_box(o, inv, best_t, nd->bbmin, nd->bbmax, MESH_T_EPS))
            continue;
        if (nd->left < 0) {
            for (int32_t i = 0; i < nd->count; i++) {
                int32_t tid = m->bvh.order[nd->start + i];
                double t = tri_mt(m->tri_v + (size_t)tid * 9, o, d);
                if (t < best_t) best_t = t;
            }
        } else {
            if (sp + 2 > BVH_STACK_MAX)
                die(EXIT_PHYSICS, "mesh: BVH stack overflow");
            stack[sp++] = nd->left;
            stack[sp++] = nd->right;
        }
    }
    return best_t;
}

/* ------------------------------------------ nearest-tri normal relocation */
/* squared distance from p to triangle (Ericson region test, scalar port
 * of mesh.py _closest_tri_dist2), also returns the closest point */
static double closest_tri_dist2(kvec3 p, kvec3 a, kvec3 b, kvec3 c) {
    kvec3 ab = v3_sub(b, a), ac = v3_sub(c, a), ap = v3_sub(p, a);
    double d1 = v3_dot(ab, ap), d2 = v3_dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { kvec3 dv = v3_sub(p, a);
        return v3_dot(dv, dv); }
    kvec3 bp = v3_sub(p, b);
    double d3 = v3_dot(ab, bp), d4 = v3_dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { kvec3 dv = v3_sub(p, b);
        return v3_dot(dv, dv); }
    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        double vv = d1 / (d1 - d3);
        kvec3 q = v3_fma(a, vv, ab);
        kvec3 dv = v3_sub(p, q);
        return v3_dot(dv, dv);
    }
    kvec3 cp = v3_sub(p, c);
    double d5 = v3_dot(ab, cp), d6 = v3_dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { kvec3 dv = v3_sub(p, c);
        return v3_dot(dv, dv); }
    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        double ww = d2 / (d2 - d6);
        kvec3 q = v3_fma(a, ww, ac);
        kvec3 dv = v3_sub(p, q);
        return v3_dot(dv, dv);
    }
    double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        double ww = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        kvec3 q = v3_fma(b, ww, v3_sub(c, b));
        kvec3 dv = v3_sub(p, q);
        return v3_dot(dv, dv);
    }
    double denom = 1.0 / (va + vb + vc);
    double vv = vb * denom, ww = vc * denom;
    kvec3 q = v3_add(a, v3_add(v3_scale(ab, vv), v3_scale(ac, ww)));
    kvec3 dv = v3_sub(p, q);
    return v3_dot(dv, dv);
}

static double point_box_dist2(kvec3 p, kvec3 lo, kvec3 hi) {
    double dx = p.x < lo.x ? lo.x - p.x : (p.x > hi.x ? p.x - hi.x : 0.0);
    double dy = p.y < lo.y ? lo.y - p.y : (p.y > hi.y ? p.y - hi.y : 0.0);
    double dz = p.z < lo.z ? lo.z - p.z : (p.z > hi.z ? p.z - hi.z : 0.0);
    return dx * dx + dy * dy + dz * dz;
}

static int32_t nearest_tri(const MeshC *m, kvec3 p) {
    double best_d2 = INFINITY;
    int32_t best = -1;
    int32_t stack[BVH_STACK_MAX];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const BvhNode *nd = &m->bvh.nodes[stack[--sp]];
        if (point_box_dist2(p, nd->bbmin, nd->bbmax) > best_d2)
            continue;
        if (nd->left < 0) {
            for (int32_t i = 0; i < nd->count; i++) {
                int32_t tid = m->bvh.order[nd->start + i];
                const double *v = m->tri_v + (size_t)tid * 9;
                double d2 = closest_tri_dist2(
                    p, v3(v[0], v[1], v[2]), v3(v[3], v[4], v[5]),
                    v3(v[6], v[7], v[8]));
                if (d2 < best_d2) { best_d2 = d2; best = tid; }
            }
        } else {
            if (sp + 2 > BVH_STACK_MAX)
                die(EXIT_PHYSICS, "mesh: BVH stack overflow");
            stack[sp++] = nd->left;
            stack[sp++] = nd->right;
        }
    }
    return best;
}

kvec3 mesh_normal(const MeshC *m, kvec3 p) {
    int32_t tid = nearest_tri(m, p);
    if (tid < 0)
        die(EXIT_PHYSICS, "mesh: normal query found no triangle "
            "(point not on the mesh?)");
    if (m->flat_normals)
        return v3(m->face_normals[tid * 3], m->face_normals[tid * 3 + 1],
                  m->face_normals[tid * 3 + 2]);
    /* barycentric weights of p on the containing triangle
     * (mesh.py _barycentric) */
    const int32_t *vi = m->tri_vidx + (size_t)tid * 3;
    kvec3 a = v3(m->verts[vi[0] * 3], m->verts[vi[0] * 3 + 1],
                 m->verts[vi[0] * 3 + 2]);
    kvec3 b = v3(m->verts[vi[1] * 3], m->verts[vi[1] * 3 + 1],
                 m->verts[vi[1] * 3 + 2]);
    kvec3 c = v3(m->verts[vi[2] * 3], m->verts[vi[2] * 3 + 1],
                 m->verts[vi[2] * 3 + 2]);
    kvec3 v0 = v3_sub(b, a), v1 = v3_sub(c, a), v2 = v3_sub(p, a);
    double d00 = v3_dot(v0, v0), d01 = v3_dot(v0, v1);
    double d11 = v3_dot(v1, v1);
    double d20 = v3_dot(v2, v0), d21 = v3_dot(v2, v1);
    double denom = d00 * d11 - d01 * d01;
    if (fabs(denom) <= 1e-300) denom = 1.0;
    double vv = (d11 * d20 - d01 * d21) / denom;
    double ww = (d00 * d21 - d01 * d20) / denom;
    double uu = 1.0 - vv - ww;
    kvec3 n = v3(0.0, 0.0, 0.0);
    const double *vn = m->vert_normals;
    n = v3_fma(n, uu, v3(vn[vi[0] * 3], vn[vi[0] * 3 + 1],
                         vn[vi[0] * 3 + 2]));
    n = v3_fma(n, vv, v3(vn[vi[1] * 3], vn[vi[1] * 3 + 1],
                         vn[vi[1] * 3 + 2]));
    n = v3_fma(n, ww, v3(vn[vi[2] * 3], vn[vi[2] * 3 + 1],
                         vn[vi[2] * 3 + 2]));
    return v3_unit(n);
}
