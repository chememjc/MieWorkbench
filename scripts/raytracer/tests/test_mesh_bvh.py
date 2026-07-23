# =============================================================================
# test_mesh_bvh.py — validation of mesh.py: STL reading (binary + ASCII),
# welded indexed mesh, BVH + batched Moller-Trumbore, and MeshFace against an
# analytic sphere. Run:
#   "$MIEWB_OPTICS_PYTHON" -m pytest raytracer/tests/test_mesh_bvh.py -q
# =============================================================================
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer import mesh as M                              # noqa: E402
from raytracer.surfaces import Sphere                        # noqa: E402


# ---------------------------------------------------------------------------
# in-test icosphere generator (unit sphere), scaled to a radius in metres
# ---------------------------------------------------------------------------
def icosphere(n_sub, radius):
    t = (1 + 5 ** 0.5) / 2
    v = [np.array(x, float) for x in (
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
        [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1],
        [-t, 0, 1])]
    v = [x / np.linalg.norm(x) for x in v]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
         [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
         [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
         [8, 6, 7], [9, 8, 1]]
    for _ in range(n_sub):
        mid = {}
        nf = []

        def mp(a, b):
            k = (min(a, b), max(a, b))
            if k in mid:
                return mid[k]
            p = v[a] + v[b]
            v.append(p / np.linalg.norm(p))
            mid[k] = len(v) - 1
            return mid[k]
        for a, b, c in f:
            ab, bc, ca = mp(a, b), mp(b, c), mp(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    V = np.array(v) * radius
    F = np.array(f)
    return V[F], V, F                                        # (T,3,3), verts, idx


def write_binary_stl(path, tris, outward=True):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    if outward:                                             # orient by centroid
        flip = np.sum(n * tris.mean(1), axis=1) < 0
        n[flip] *= -1
    with open(path, "wb") as fh:
        fh.write(b"opticalraytracer test STL".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(tris)))
        for i in range(len(tris)):
            fh.write(struct.pack("<3f", *n[i]))
            for p in tris[i]:
                fh.write(struct.pack("<3f", *p))
            fh.write(struct.pack("<H", 0))


def write_ascii_stl(path, tris):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    with open(path, "w") as fh:
        fh.write("solid test\n")
        for i in range(len(tris)):
            fh.write("facet normal %g %g %g\n" % tuple(n[i]))
            fh.write("outer loop\n")
            for p in tris[i]:
                fh.write("vertex %g %g %g\n" % tuple(p))
            fh.write("endloop\nendfacet\n")
        fh.write("endsolid test\n")


@pytest.fixture(scope="module")
def ico(tmp_path_factory):
    d = tmp_path_factory.mktemp("mesh")
    tris, V, F = icosphere(4, 0.05)                          # ~5120 tris
    stl = d / "ico.stl"
    write_binary_stl(str(stl), tris)
    return {"dir": d, "tris": tris, "stl": str(stl), "radius": 0.05}


def _mkface(stl, flat=False):
    rec = {"id": "B.F.Face1", "orientation_outward": True, "area_m2": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return M.MeshFace(rec, stl, flat_normals=flat)


# ---------------------------------------------------------------------------
# MeshFace vs analytic sphere
# ---------------------------------------------------------------------------
def test_meshface_vs_analytic_sphere(ico):
    R = ico["radius"]
    mf = _mkface(ico["stl"])
    sph = Sphere([0, 0, 0], R)
    rng = np.random.default_rng(1)
    o = rng.normal(scale=0.15, size=(500, 3))
    tgt = rng.normal(scale=0.02, size=(500, 3))
    d = tgt - o
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    tm, hitm = mf.intersect(o, d)
    ts, vs = sph.intersect(o, d)
    ta = np.where(vs & (ts > 1e-7), ts, np.inf).min(axis=1)
    hita = np.isfinite(ta)

    # tessellation sag bound: an inscribed icosphere chord sits below the true
    # surface by up to ~edge^2/(8R). Along a ray that radial sag maps to a t
    # error ~ sag / |cos(incidence)|, so normalise by the incidence cosine.
    tris = ico["tris"]
    edges = np.concatenate([
        np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1),
        np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1),
        np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)])
    sag_bound = edges.max() ** 2 / (8.0 * R)

    ta_safe = np.where(hita, ta, 0.0)
    phit = o + ta_safe[:, None] * d
    cos_graze = np.abs(np.sum(d * phit, axis=-1)
                       / (np.linalg.norm(phit, axis=-1) + 1e-30))
    both = hitm & hita
    solid = both & (cos_graze > 0.2)                        # not near-tangent
    assert solid.sum() > 300
    # incidence-normalised radial error stays within a couple of sag bounds
    assert np.max(np.abs(tm[solid] - ta[solid]) * cos_graze[solid]) \
        < 3.0 * sag_bound
    # hit/miss agreement away from the silhouette
    clear = (cos_graze > 0.4)
    assert np.mean(hitm[clear] == hita[clear]) > 0.98


def test_interpolated_and_flat_normals(ico):
    R = ico["radius"]
    mf = _mkface(ico["stl"])
    flatf = _mkface(ico["stl"], flat=True)
    sph = Sphere([0, 0, 0], R)
    rng = np.random.default_rng(2)
    # points exactly on the analytic sphere
    u = rng.normal(size=(300, 3))
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    p = R * u
    n_true = sph.normal(p)

    n_interp = mf.normal(p)
    ang = np.degrees(np.arccos(np.clip(np.sum(n_interp * n_true, axis=-1),
                                       -1, 1)))
    assert ang.max() < 2.0                                  # within 2 degrees

    n_flat = flatf.normal(p)
    assert np.max(np.abs(np.linalg.norm(n_flat, axis=-1) - 1.0)) < 1e-12
    # flat facet normals are quantized -> differ from the smooth interp
    assert np.mean(np.abs(np.sum(n_flat * n_true, axis=-1)) < 0.9999) > 0.5

    # outward orientation: normal_out_of_solid points away from the centre
    nout = mf.normal_out_of_solid(p)
    assert np.all(np.sum(nout * u, axis=-1) > 0.0)


# ---------------------------------------------------------------------------
# BVH == brute force
# ---------------------------------------------------------------------------
def test_bvh_equals_bruteforce(ico):
    tris, _ = M.read_stl(ico["stl"])
    bvh = M.BVH(tris)
    rng = np.random.default_rng(3)
    o = rng.normal(scale=0.15, size=(200, 3))
    tgt = rng.normal(scale=0.02, size=(200, 3))
    d = tgt - o
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    tb, trib = bvh.intersect(o, d)

    # brute force all triangles
    v0 = tris[:, 0]
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    best = np.full(200, np.inf)
    btri = np.full(200, -1)
    for i in range(200):
        h = np.cross(d[i], e2)
        a = np.sum(e1 * h, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = 1.0 / a
            s = o[i] - v0
            uu = f * np.sum(s * h, axis=1)
            q = np.cross(s, e1)
            vv = f * np.sum(d[i] * q, axis=1)
            tt = f * np.sum(e2 * q, axis=1)
        ok = ((np.abs(a) > 1e-300) & (uu >= -1e-12) & (vv >= -1e-12)
              & (uu + vv <= 1 + 1e-12) & (tt > 1e-7))
        tt = np.where(ok, tt, np.inf)
        j = int(np.argmin(tt))
        if np.isfinite(tt[j]):
            best[i] = tt[j]
            btri[i] = j
    assert np.all(np.isfinite(best) == np.isfinite(tb))
    fin = np.isfinite(best)
    assert np.max(np.abs(best[fin] - tb[fin])) == 0.0
    assert np.all(btri[fin] == trib[fin])


# ---------------------------------------------------------------------------
# STL reader round-trip (binary and ASCII agree)
# ---------------------------------------------------------------------------
def test_binary_ascii_roundtrip(ico):
    tris_b, _ = M.read_stl(ico["stl"])
    ascii_path = ico["dir"] / "ico_ascii.stl"
    write_ascii_stl(str(ascii_path), ico["tris"])
    tris_a, _ = M.read_stl(str(ascii_path))
    assert tris_a.shape == tris_b.shape
    # ASCII is written with %g (finite precision) -> compare loosely
    assert np.max(np.abs(np.sort(tris_a.reshape(-1))
                         - np.sort(tris_b.reshape(-1)))) < 1e-4

    ib, ha = M.MeshFace, None
    facea = _mkface(str(ascii_path))
    faceb = _mkface(ico["stl"])
    rng = np.random.default_rng(9)
    o = rng.normal(scale=0.15, size=(200, 3))
    d = -o / np.linalg.norm(o, axis=-1, keepdims=True)       # aim at centre
    ta, _ = facea.intersect(o, d)
    tbv, _ = faceb.intersect(o, d)
    fin = np.isfinite(ta) & np.isfinite(tbv)
    assert fin.sum() > 150
    assert np.max(np.abs(ta[fin] - tbv[fin])) < 1e-4


# ---------------------------------------------------------------------------
# t_eps: a ray starting on a triangle does not self-hit
# ---------------------------------------------------------------------------
def test_t_eps_no_self_hit(ico):
    mf = _mkface(ico["stl"])
    R = ico["radius"]
    tris, _ = M.read_stl(ico["stl"])
    # start exactly at triangle centroids, fire outward -> no immediate self-hit
    cen = tris.mean(axis=1)[:50]
    nrm = np.cross(tris[:50, 1] - tris[:50, 0], tris[:50, 2] - tris[:50, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    outward = np.sign(np.sum(nrm * cen, axis=1))[:, None]
    d = nrm * outward
    t, hit = mf.intersect(cen, d)
    # outward rays from the surface leave the (convex) sphere: no re-hit
    assert not np.any(t < 1e-6)


# ---------------------------------------------------------------------------
# degenerate (zero-area) triangles are dropped with a warning
# ---------------------------------------------------------------------------
def test_degenerate_triangles_dropped(ico):
    tris = ico["tris"].copy()
    # append a zero-area (collapsed) triangle
    bad = np.zeros((1, 3, 3))
    bad[0] = np.array([[0.01, 0, 0], [0.01, 0, 0], [0.01, 0, 0]])
    tris2 = np.concatenate([tris, bad], axis=0)
    stl2 = ico["dir"] / "ico_degen.stl"
    write_binary_stl(str(stl2), ico["tris"])                # base
    # write manually with the degenerate appended (normals zero for the bad one)
    with open(stl2, "wb") as fh:
        fh.write(b"degen".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(tris2)))
        n = np.cross(tris2[:, 1] - tris2[:, 0], tris2[:, 2] - tris2[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
        for i in range(len(tris2)):
            fh.write(struct.pack("<3f", *n[i]))
            for p in tris2[i]:
                fh.write(struct.pack("<3f", *p))
            fh.write(struct.pack("<H", 0))
    tr, _ = M.read_stl(str(stl2))
    with pytest.warns(UserWarning, match="degenerate"):
        im = M.IndexedMesh(tr, face_id="B.F.Face1")
    assert len(im.tris) == len(ico["tris"])                 # bad one dropped
    assert np.all(np.isfinite(im.vertex_normals))
