# =============================================================================
# mesh.py — tessellated-face fallback: STL reader, welded indexed mesh,
# median-split AABB BVH, batched Moller-Trumbore, and a MeshFace object that
# is duck-type compatible with surfaces.AnalyticFace (same .intersect and
# .normal_out_of_solid contract) so the tracer can shoot rays at genuinely
# non-analytic faces.
#
# HONEST LIMIT (README): a tessellated face carries sag error ~ the linear
# deflection of the mesh (tens of microns), which is >> lambda. Optical-path
# phase through such a face is therefore meaningless for coherent gather —
# every MeshFace warns once at construction. Meshes are for incoherent power
# accounting / geometry-limited faces only.
#
# UNITS: the extractor (extract_geometry.mesh_and_write_stl) scales each face
# to METRES *before* tessellating and writing the binary STL (see its module
# header: "binary STL, metres, per face"). So STL vertices are already SI
# metres and we load them as-is (scale=1.0). The optional `scale` argument is
# provided only for STLs authored in other units.
#
# Conventions mirror surfaces.py: all geometry SI metres / float64; the
# canonical mesh normal is the triangle-winding normal, aligned at load to the
# STL's stored per-facet normals (which follow FreeCAD's face orientation), so
# orientation_outward describes it exactly as for the analytic quadrics.
# =============================================================================
import struct
import warnings

import numpy as np

_WELD_GRID = 1e-9          # vertex weld quantum [m]
_LEAF = 8                  # BVH leaf triangle count
_T_EPS = 1e-7             # self-intersection guard [m], identical to AnalyticFace


# ---------------------------------------------------------------------------
# STL reading (binary + ASCII, auto-sniffed)
# ---------------------------------------------------------------------------
_BIN_REC = np.dtype([("normal", "<f4", (3,)),
                     ("verts", "<f4", (3, 3)),
                     ("attr", "<u2")])


def _looks_binary(buf):
    """Binary STL is 80-byte header + uint32 count + count*50 bytes."""
    if len(buf) < 84:
        return False
    count = struct.unpack_from("<I", buf, 80)[0]
    return len(buf) == 84 + count * 50


def read_stl(path, scale=1.0):
    """Read a binary or ASCII STL. Returns (tris, facet_normals) with tris of
    shape (n_tri, 3, 3) float64 in metres (after `scale`) and facet_normals
    (n_tri, 3) float64 as stored in the file (unnormalized; may be zero)."""
    with open(path, "rb") as fh:
        buf = fh.read()
    if _looks_binary(buf):
        count = struct.unpack_from("<I", buf, 80)[0]
        data = np.frombuffer(buf, dtype=_BIN_REC, count=count, offset=84)
        tris = data["verts"].astype(np.float64) * scale
        normals = data["normal"].astype(np.float64)
        return tris, normals
    return _read_ascii_stl(buf, scale)


def _read_ascii_stl(buf, scale):
    text = buf.decode("ascii", errors="ignore")
    verts = []
    normals = []
    cur = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0].lower()
        if key == "facet" and len(parts) >= 5 and parts[1].lower() == "normal":
            normals.append([float(parts[2]), float(parts[3]), float(parts[4])])
        elif key == "vertex" and len(parts) >= 4:
            cur.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(cur) == 3:
                verts.append(cur)
                cur = []
    tris = np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3) * scale
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if len(normals) != len(tris):        # tolerate missing/mismatched normals
        normals = np.zeros((len(tris), 3), dtype=np.float64)
    return tris, normals


# ---------------------------------------------------------------------------
# Indexed mesh: weld vertices, drop degenerates, align winding, vertex normals
# ---------------------------------------------------------------------------
class IndexedMesh:
    def __init__(self, tris, facet_normals=None, face_id="<mesh>"):
        tris = np.asarray(tris, dtype=np.float64)
        if facet_normals is None:
            facet_normals = np.zeros((len(tris), 3), dtype=np.float64)

        # geometric (winding) normal; drop zero-area triangles
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        gn = np.cross(e1, e2)
        gn_len = np.linalg.norm(gn, axis=-1)
        keep = gn_len > 1e-20
        n_drop = int((~keep).sum())
        if n_drop:
            warnings.warn(
                "mesh %s: dropped %d degenerate (zero-area) triangle(s)"
                % (face_id, n_drop), stacklevel=2)
        tris = tris[keep]
        gn = gn[keep]
        gn_len = gn_len[keep]
        facet_normals = facet_normals[keep]

        # align winding to the stored facet normals (FreeCAD's face
        # orientation) so the canonical normal matches orientation_outward
        dots = np.sum(gn * facet_normals, axis=-1)
        flip = dots < 0.0
        if np.any(flip):
            tris[flip] = tris[flip][:, [0, 2, 1], :]
            gn[flip] = -gn[flip]

        self.face_normals = gn / gn_len[:, None]          # (T,3) unit
        self.tri_areas = 0.5 * gn_len                     # (T,)
        self.area_m2 = float(self.tri_areas.sum())

        # weld vertices onto a grid -> indexed mesh
        q = np.round(tris.reshape(-1, 3) / _WELD_GRID).astype(np.int64)
        _, inv = np.unique(q, axis=0, return_inverse=True)
        uverts = np.zeros((inv.max() + 1, 3), dtype=np.float64)
        flat = tris.reshape(-1, 3)
        uverts[inv] = flat                                # any representative
        self.vertices = uverts
        self.faces = inv.reshape(-1, 3).astype(np.int64)  # (T,3) vertex ids
        self.tris = tris                                  # (T,3,3) coords

        # angle-weighted vertex normals
        self.vertex_normals = self._angle_weighted_normals()

    def _angle_weighted_normals(self):
        V = self.vertices
        F = self.faces
        vn = np.zeros_like(V)
        p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        fn = self.face_normals
        for (a, b, c) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            pa, pb, pc = V[F[:, a]], V[F[:, b]], V[F[:, c]]
            u = pb - pa
            w = pc - pa
            u /= (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-300)
            w /= (np.linalg.norm(w, axis=-1, keepdims=True) + 1e-300)
            ang = np.arccos(np.clip(np.sum(u * w, axis=-1), -1.0, 1.0))
            np.add.at(vn, F[:, a], fn * ang[:, None])
        nrm = np.linalg.norm(vn, axis=-1, keepdims=True)
        return vn / np.where(nrm > 1e-300, nrm, 1.0)


# ---------------------------------------------------------------------------
# BVH: median-split over centroids, arrays-only, batched traversal
# ---------------------------------------------------------------------------
class BVH:
    def __init__(self, tris, leaf=_LEAF):
        self.tris = np.ascontiguousarray(tris, dtype=np.float64)
        T = len(tris)
        centroids = tris.mean(axis=1)
        bbmin_l, bbmax_l = [], []
        left_l, right_l, start_l, count_l = [], [], [], []
        order = []

        # iterative build (avoids recursion-limit issues on big meshes)
        self._n = 0

        def alloc():
            bbmin_l.append(None); bbmax_l.append(None)
            left_l.append(-1); right_l.append(-1)
            start_l.append(-1); count_l.append(0)
            i = self._n
            self._n += 1
            return i

        # stack of (node_id, triangle-index array)
        root = alloc()
        stack = [(root, np.arange(T))]
        while stack:
            ni, idxs = stack.pop()
            pts = tris[idxs]
            bbmin_l[ni] = pts.min(axis=(0, 1))
            bbmax_l[ni] = pts.max(axis=(0, 1))
            if len(idxs) <= leaf:
                start_l[ni] = len(order)
                count_l[ni] = len(idxs)
                order.extend(idxs.tolist())
                continue
            cen = centroids[idxs]
            ext = cen.max(axis=0) - cen.min(axis=0)
            ax = int(np.argmax(ext))
            med = np.median(cen[:, ax])
            lmask = cen[:, ax] <= med
            if lmask.all() or not lmask.any():   # degenerate split -> halve
                srt = idxs[np.argsort(cen[:, ax], kind="stable")]
                half = len(srt) // 2
                lidx, ridx = srt[:half], srt[half:]
            else:
                lidx, ridx = idxs[lmask], idxs[~lmask]
            li, ri = alloc(), alloc()
            left_l[ni], right_l[ni] = li, ri
            stack.append((li, lidx))
            stack.append((ri, ridx))

        self.bbmin = np.array(bbmin_l, dtype=np.float64)
        self.bbmax = np.array(bbmax_l, dtype=np.float64)
        self.left = np.array(left_l, dtype=np.int32)
        self.right = np.array(right_l, dtype=np.int32)
        self.start = np.array(start_l, dtype=np.int32)
        self.count = np.array(count_l, dtype=np.int32)
        self.order = np.array(order, dtype=np.int32)      # leaf tri ids

    # ---- ray batch nearest hit ----------------------------------------
    def intersect(self, o, d, t_eps=_T_EPS, best_t=None):
        """Nearest triangle hit per ray. Returns (t (N,), tri_id (N,) int32,
        -1 = miss). Backface culling OFF (watertight solids)."""
        N = len(o)
        if best_t is None:
            best_t = np.full(N, np.inf)
        else:
            best_t = best_t.copy()
        best_tri = np.full(N, -1, dtype=np.int32)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
        stack = [(0, np.arange(N))]
        while stack:
            ni, ridx = stack.pop()
            if ridx.size == 0:
                continue
            hitbox = self._ray_box(o[ridx], inv[ridx], best_t[ridx],
                                   self.bbmin[ni], self.bbmax[ni], t_eps)
            sub = ridx[hitbox]
            if sub.size == 0:
                continue
            if self.left[ni] < 0:                # leaf
                self._leaf_mt(ni, sub, o, d, t_eps, best_t, best_tri)
            else:
                stack.append((self.left[ni], sub))
                stack.append((self.right[ni], sub))
        return best_t, best_tri

    @staticmethod
    def _ray_box(o, inv, best_t, bbmin, bbmax, t_eps):
        t1 = (bbmin[None] - o) * inv
        t2 = (bbmax[None] - o) * inv
        tmin = np.minimum(t1, t2).max(axis=-1)
        tmax = np.maximum(t1, t2).min(axis=-1)
        return (tmax >= np.maximum(tmin, t_eps)) & (tmin <= best_t)

    def _leaf_mt(self, ni, sub, o, d, t_eps, best_t, best_tri):
        tri_ids = self.order[self.start[ni]:self.start[ni] + self.count[ni]]
        tri = self.tris[tri_ids]                          # (L,3,3)
        os = o[sub]                                       # (S,3)
        ds = d[sub]
        # broadcast S x L
        v0 = tri[:, 0][None]                              # (1,L,3)
        e1 = (tri[:, 1] - tri[:, 0])[None]
        e2 = (tri[:, 2] - tri[:, 0])[None]
        dd = ds[:, None, :]                               # (S,1,3)
        h = np.cross(dd, e2)
        a = np.sum(e1 * h, axis=-1)                       # (S,L)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = 1.0 / a
            s = os[:, None, :] - v0
            u = f * np.sum(s * h, axis=-1)
            q = np.cross(s, e1)
            v = f * np.sum(dd * q, axis=-1)
            t = f * np.sum(e2 * q, axis=-1)
        ok = (np.abs(a) > 1e-300) & (u >= -1e-12) & (v >= -1e-12) \
            & (u + v <= 1.0 + 1e-12) & (t > t_eps)
        t = np.where(ok, t, np.inf)
        jbest = np.argmin(t, axis=1)                      # (S,)
        tbest = t[np.arange(len(sub)), jbest]
        improve = tbest < best_t[sub]
        gi = sub[improve]
        best_t[gi] = tbest[improve]
        best_tri[gi] = tri_ids[jbest[improve]]

    # ---- nearest triangle to points (for normal relocation) -----------
    def nearest_tri(self, pts):
        """Nearest triangle id per point (points assumed ~on the surface)."""
        M = len(pts)
        best_d2 = np.full(M, np.inf)
        best_tri = np.full(M, -1, dtype=np.int32)
        stack = [(0, np.arange(M))]
        while stack:
            ni, pidx = stack.pop()
            if pidx.size == 0:
                continue
            d2box = _point_box_dist2(pts[pidx], self.bbmin[ni], self.bbmax[ni])
            sub = pidx[d2box <= best_d2[pidx]]
            if sub.size == 0:
                continue
            if self.left[ni] < 0:
                tri_ids = self.order[self.start[ni]:
                                     self.start[ni] + self.count[ni]]
                tri = self.tris[tri_ids]                  # (L,3,3)
                P = pts[sub][:, None, :]                  # (S,1,3)
                d2 = _closest_tri_dist2(
                    P, tri[:, 0][None], tri[:, 1][None], tri[:, 2][None])
                jb = np.argmin(d2, axis=1)
                db = d2[np.arange(len(sub)), jb]
                imp = db < best_d2[sub]
                gi = sub[imp]
                best_d2[gi] = db[imp]
                best_tri[gi] = tri_ids[jb[imp]]
            else:
                stack.append((self.left[ni], sub))
                stack.append((self.right[ni], sub))
        return best_tri


def _point_box_dist2(pts, bbmin, bbmax):
    cl = np.minimum(np.maximum(pts, bbmin[None]), bbmax[None])
    d = pts - cl
    return np.sum(d * d, axis=-1)


def _closest_tri_dist2(P, A, B, C):
    """Squared distance from points P to triangles (A,B,C), broadcasting.
    Ericson 'Real-Time Collision Detection' region test, fully vectorized."""
    ab = B - A
    ac = C - A
    ap = P - A
    d1 = np.sum(ab * ap, axis=-1)
    d2 = np.sum(ac * ap, axis=-1)
    bp = P - B
    d3 = np.sum(ab * bp, axis=-1)
    d4 = np.sum(ac * bp, axis=-1)
    cp = P - C
    d5 = np.sum(ab * cp, axis=-1)
    d6 = np.sum(ac * cp, axis=-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        v_face = vb / np.where(denom != 0.0, denom, 1.0)
        w_face = vc / np.where(denom != 0.0, denom, 1.0)
        v_ab = d1 / np.where((d1 - d3) != 0.0, d1 - d3, 1.0)
        v_ac = d2 / np.where((d2 - d6) != 0.0, d2 - d6, 1.0)
        v_bc = (d4 - d3) / np.where((d4 - d3 + d5 - d6) != 0.0,
                                    d4 - d3 + d5 - d6, 1.0)
    shp = np.broadcast_shapes(P.shape[:-1], A.shape[:-1])
    A_, B_, C_ = np.broadcast_to(A, shp + (3,)), np.broadcast_to(B, shp + (3,)),\
        np.broadcast_to(C, shp + (3,))
    P_ = np.broadcast_to(P, shp + (3,))
    d1, d2, d3, d4, d5, d6 = (np.broadcast_to(x, shp)
                              for x in (d1, d2, d3, d4, d5, d6))
    va, vb, vc = (np.broadcast_to(x, shp) for x in (va, vb, vc))
    v_face, w_face, v_ab, v_ac, v_bc = (
        np.broadcast_to(x, shp) for x in
        (v_face, w_face, v_ab, v_ac, v_bc))
    cpt = A_ + v_face[..., None] * (B_ - A_) + w_face[..., None] * (C_ - A_)
    # region overrides (vertex/edge)
    cpt = np.where(((d1 <= 0) & (d2 <= 0))[..., None], A_, cpt)
    cpt = np.where(((d3 >= 0) & (d4 <= d3))[..., None], B_, cpt)
    cpt = np.where(((d6 >= 0) & (d5 <= d6))[..., None], C_, cpt)
    e_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    cpt = np.where(e_ab[..., None], A_ + v_ab[..., None] * ab_b(A_, B_), cpt)
    e_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    cpt = np.where(e_ac[..., None], A_ + v_ac[..., None] * (C_ - A_), cpt)
    e_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    cpt = np.where(e_bc[..., None], B_ + v_bc[..., None] * (C_ - B_), cpt)
    dvec = P_ - cpt
    return np.sum(dvec * dvec, axis=-1)


def ab_b(A_, B_):
    return B_ - A_


def _barycentric(p, a, b, c):
    """Barycentric weights (wa, wb, wc) of p projected onto triangle abc."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.sum(v0 * v0, axis=-1)
    d01 = np.sum(v0 * v1, axis=-1)
    d11 = np.sum(v1 * v1, axis=-1)
    d20 = np.sum(v2 * v0, axis=-1)
    d21 = np.sum(v2 * v1, axis=-1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) > 1e-300, denom, 1.0)
    vv = (d11 * d20 - d01 * d21) / denom
    ww = (d00 * d21 - d01 * d20) / denom
    uu = 1.0 - vv - ww
    return uu, vv, ww


# ---------------------------------------------------------------------------
# MeshFace — AnalyticFace-compatible tessellated face
# ---------------------------------------------------------------------------
class MeshFace:
    """Duck-type compatible with surfaces.AnalyticFace for the pipeline's
    purposes: exposes .id, .area_m2, .intersect(o,d,...) -> (t, hit), and
    .normal_out_of_solid(p). Not analytic, so .surface is None and it must not
    be used as a source/detector/grating face (those consume face.surface /
    face.trim); the lead wires MeshFace only into optic bodies.

    Normals: intersect() returns only (t, hit) to match AnalyticFace, so
    normal_out_of_solid re-locates the containing triangle for each query
    point via a BVH nearest-triangle search (points are on the surface, so
    this is exact and independent of any per-call caching). flat_normals=True
    returns the facet normal; otherwise angle-weighted vertex normals are
    barycentrically interpolated across the containing triangle.
    """

    surface = None            # not an analytic primitive
    trim = None

    def __init__(self, face_record, stl_path, flat_normals=False, scale=1.0):
        self.id = face_record["id"]
        self.flat_normals = bool(flat_normals)
        tris, facet_normals = read_stl(stl_path, scale=scale)
        self.mesh = IndexedMesh(tris, facet_normals, face_id=self.id)
        self.bvh = BVH(self.mesh.tris)
        self.area_m2 = face_record.get("area_m2") or self.mesh.area_m2
        self.outward_sign = 1.0 if face_record["orientation_outward"] else -1.0
        self.body_index = face_record.get("body_index")
        self.face_index = face_record.get("face_index")
        warnings.warn(
            "MeshFace %s: tessellated face — coherent optical-path phase "
            "through it carries sag error >> lambda (README honest limit); "
            "use for incoherent/geometry-limited accounting only." % self.id,
            stacklevel=2)

    def intersect(self, o, d, t_eps=_T_EPS, exclude_mask=None):
        """Nearest contained hit per ray. Returns (t, hit_mask); t = inf where
        no hit. exclude_mask (N,) True suppresses this face for those rays."""
        o = np.asarray(o, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        t, tri = self.bvh.intersect(o, d, t_eps=t_eps)
        if exclude_mask is not None:
            t = np.where(exclude_mask, np.inf, t)
        return t, np.isfinite(t)

    def normal(self, p):
        """Canonical (winding-aligned) unit normal at surface points p."""
        p = np.asarray(p, dtype=np.float64)
        tri = self.bvh.nearest_tri(p)
        if self.flat_normals:
            n = self.mesh.face_normals[tri]
            return n / np.linalg.norm(n, axis=-1, keepdims=True)
        F = self.mesh.faces[tri]
        A = self.mesh.vertices[F[:, 0]]
        B = self.mesh.vertices[F[:, 1]]
        C = self.mesh.vertices[F[:, 2]]
        wa, wb, wc = _barycentric(p, A, B, C)
        vn = self.mesh.vertex_normals
        n = (wa[:, None] * vn[F[:, 0]] + wb[:, None] * vn[F[:, 1]]
             + wc[:, None] * vn[F[:, 2]])
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def normal_out_of_solid(self, p):
        return self.outward_sign * self.normal(p)
