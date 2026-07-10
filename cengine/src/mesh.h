/* ===========================================================================
 * mesh.h — tessellated (STL) face support: binary STL reader, welded
 * indexed mesh, triangle BLAS, Moller-Trumbore, nearest-triangle normal
 * relocation. Port of scripts/raytracer/mesh.py (see its header for the
 * honest sag-error limit; mesh faces are incoherent-only there and the
 * same limit holds here).
 * =========================================================================== */
#ifndef MIEWB_MESH_H
#define MIEWB_MESH_H

#include "bvh.h"

typedef struct {
    int32_t n_tris, n_verts;
    double *tri_v;          /* n_tris * 9: v0.xyz v1.xyz v2.xyz (welded) */
    double *face_normals;   /* n_tris * 3, unit, winding-aligned */
    int32_t *tri_vidx;      /* n_tris * 3 vertex ids */
    double *verts;          /* n_verts * 3 */
    double *vert_normals;   /* n_verts * 3, angle-weighted, unit */
    uint8_t flat_normals;   /* --mesh-flat-normals */
    BvhC bvh;               /* triangle BLAS, leaf 8 (mesh.py _LEAF) */
} MeshC;

/* Load a binary STL (the extractor writes binary; ASCII dies with a clear
 * message), weld/clean exactly like mesh.IndexedMesh, build the BLAS. */
MeshC *mesh_load(const char *stl_path, int flat_normals,
                 const char *face_id);
void mesh_free(MeshC *m);

/* Nearest triangle hit (t > t_eps 1e-7); +INF on miss.
 * Port of BVH.intersect + _leaf_mt (mesh.py:219-282), scalar. */
double mesh_intersect(const MeshC *m, kvec3 o, kvec3 d);

/* Canonical (winding-aligned) unit normal at a surface point: relocate
 * the containing triangle (nearest-tri BVH search) then facet normal or
 * barycentric-interpolated vertex normals (mesh.py MeshFace.normal). */
kvec3 mesh_normal(const MeshC *m, kvec3 p);

#endif /* MIEWB_MESH_H */
