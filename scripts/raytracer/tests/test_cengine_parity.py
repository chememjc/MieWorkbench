# =============================================================================
# test_cengine_parity.py — side-by-side C engine vs Python engine.
#
# Runs each synthetic feature scene (cengine_scenes.py) through the FULL
# run_trace CLI path twice (--engine python, --engine c) and compares the
# output artifacts:
#
#   deterministic scenes (collimated normal incidence — every ray behaves
#   identically regardless of sampled position): emitted power, every
#   ledger bucket, and detected power must agree to 1e-9 relative even
#   though the two engines use different RNGs.
#
#   statistical scenes (position-dependent physics): totals agree to
#   percent-level Monte-Carlo tolerances.
#
#   always: energy closure holds in both engines; detector cube shapes and
#   grid attrs match; the C engine's trim mask equals the Python grid mask
#   bit-for-bit (same algorithm, ported).
#
# Skipped entirely when the miewb-trace binary is not built.
# =============================================================================
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from raytracer import cengine                                # noqa: E402
import cengine_scenes                                        # noqa: E402

pytestmark = pytest.mark.skipif(
    cengine.binary_path() is None,
    reason="miewb-trace not built (cd cengine && ./build.sh)")

RAYS = 20000
RESOLUTION = 128
NLAMBDA = 3


def run_engine(model_json, case_dir, engine):
    import run_trace
    rc = run_trace.main([
        "--model-json", str(model_json),
        "--case-dir", str(case_dir),
        "--rays", str(RAYS),
        "--resolution", str(RESOLUTION),
        "--nlambda", str(NLAMBDA),
        "--spectral-bins", "8",
        "--engine", engine,
        "--workers", "1",
    ])
    assert rc == 0, "run_trace --engine %s exited %s" % (engine, rc)
    return {
        "case": json.loads((case_dir / "case.json").read_text()),
        "audit": json.loads((case_dir / "audit.json").read_text()),
        "rays": np.load(case_dir / "rays.npy"),
        "case_dir": case_dir,
    }


def load_cube(case_dir):
    import h5py
    h5s = sorted((case_dir / "detectors").glob("*.h5"))
    assert len(h5s) == 1
    with h5py.File(h5s[0], "r") as h:
        return (h["spectral_cube_mean"][...], h["mask"][...],
                dict(h.attrs))


def rel_close(a, b, tol, what):
    scale = max(abs(a), abs(b), 1e-300)
    assert abs(a - b) / scale <= tol, \
        "%s: %.12g vs %.12g (rel %.3g > %.3g)" % (
            what, a, b, abs(a - b) / scale, tol)


@pytest.mark.parametrize("name", sorted(cengine_scenes.SCENES))
def test_scene_parity(name, tmp_path):
    _, klass = cengine_scenes.SCENES[name]
    model_json = cengine_scenes.write_scene(name, tmp_path / "geometry")

    py = run_engine(model_json, tmp_path / "case_py", "python")
    cc = run_engine(model_json, tmp_path / "case_c", "c")

    assert py["case"]["engine"] == "python"
    assert cc["case"]["engine"] == "c", cc["case"].get("engine_reason")

    # ---- both engines close energy ----
    for tag, r in (("python", py), ("c", cc)):
        for seed_rep in r["audit"]["per_seed"]:
            assert seed_rep["closure_ok"], \
                "%s engine failed closure: %s" % (tag, seed_rep)

    # ---- ledger comparison ----
    tol = 1e-9 if klass == "deterministic" else 0.02
    py_src = py["audit"]["per_seed"][0]["sources"]
    c_src = cc["audit"]["per_seed"][0]["sources"]
    assert set(py_src) == set(c_src)
    from raytracer.audit import BUCKETS
    for sname in py_src:
        emitted = py_src[sname]["emitted_W"]
        rel_close(emitted, c_src[sname]["emitted_W"],
                  1e-12, "%s emitted (%s)" % (name, sname))
        for b in BUCKETS:
            pv, cv = py_src[sname][b], c_src[sname][b]
            # micro-buckets (< 0.1% of emitted) are MC-noise-dominated in
            # statistical scenes — hold them to an ABSOLUTE floor vs
            # emitted instead of a relative bar on the bucket itself
            floor = (1e-12 if klass == "deterministic" else 1e-3) * emitted
            if abs(pv - cv) <= floor:
                continue
            rel_close(pv, cv, tol, "%s bucket %s (%s)" % (name, b, sname))

    # ---- detected power ----
    py_det = py["audit"]["per_seed"][0]["detected_W"]
    c_det = cc["audit"]["per_seed"][0]["detected_W"]
    assert set(py_det) == set(c_det)
    for label in py_det:
        rel_close(py_det[label], c_det[label], tol,
                  "%s detected_W %s" % (name, label))

    # ---- detector cube: shape/attrs identical, integral matches ----
    py_cube, py_mask, py_attrs = load_cube(py["case_dir"])
    c_cube, c_mask, c_attrs = load_cube(cc["case_dir"])
    assert py_cube.shape == c_cube.shape
    assert np.array_equal(py_mask, c_mask)
    for k in ("H", "W", "pixel_m", "x_lo", "y_lo"):
        assert np.allclose(py_attrs[k], c_attrs[k]), k
    rel_close(float(py_cube.sum()), float(c_cube.sum()), tol,
              "%s cube integral" % name)
    # spatial agreement: irradiance-weighted centroid within a pixel-ish
    if float(py_cube.sum()) > 0:
        py_img = py_cube.sum(axis=0)
        c_img = c_cube.sum(axis=0)
        yy, xx = np.mgrid[0:py_img.shape[0], 0:py_img.shape[1]]
        pcx = (py_img * xx).sum() / py_img.sum()
        ccx = (c_img * xx).sum() / c_img.sum()
        pcy = (py_img * yy).sum() / py_img.sum()
        ccy = (c_img * yy).sum() / c_img.sum()
        atol = 1.5 if klass == "deterministic" else 3.0
        assert abs(pcx - ccx) < atol and abs(pcy - ccy) < atol, \
            "%s centroid (%.2f, %.2f) vs (%.2f, %.2f)" % (
                name, pcx, pcy, ccx, ccy)

    # ---- C engine's own trim mask == Python grid mask (bit-for-bit) ----
    c_mask_own = np.load(cc["case_dir"] / "cengine" / "seed42"
                         / "det_0_mask.npy")
    assert np.array_equal(c_mask_own.astype(bool), py_mask.astype(bool)), \
        "%s: C trim mask differs from Python DetectorGrid mask" % name

    # ---- viz rows: same shape contract, sane values ----
    assert py["rays"].shape[1] == 13 and cc["rays"].shape[1] == 13
    assert cc["rays"].shape[0] > 0
    assert np.all(np.isfinite(cc["rays"]))


REPO = SCRIPTS.parent


@pytest.mark.parametrize("name", sorted(cengine_scenes.REAL_SCENES))
def test_real_geometry_parity(name, tmp_path):
    """Side-by-side on REAL extracted geometries (repo geometry/ dirs) —
    the closest thing to a demo without FreeCAD in the loop. Statistical
    tolerances (curved surfaces => position-dependent physics)."""
    model_json = REPO / "geometry" / name / "model.json"
    if not model_json.exists():
        pytest.skip("geometry/%s not extracted" % name)

    py = run_engine(model_json, tmp_path / "case_py", "python")
    cc = run_engine(model_json, tmp_path / "case_c", "c")
    assert cc["case"]["engine"] == "c", cc["case"].get("engine_reason")

    for tag, r in (("python", py), ("c", cc)):
        for seed_rep in r["audit"]["per_seed"]:
            assert seed_rep["closure_ok"], \
                "%s engine failed closure: %s" % (tag, seed_rep)

    py_rep = py["audit"]["per_seed"][0]
    c_rep = cc["audit"]["per_seed"][0]
    for sname in py_rep["sources"]:
        rel_close(py_rep["sources"][sname]["emitted_W"],
                  c_rep["sources"][sname]["emitted_W"], 1e-12,
                  "%s emitted (%s)" % (name, sname))
    assert set(py_rep["detected_W"]) == set(c_rep["detected_W"])
    for label in py_rep["detected_W"]:
        rel_close(py_rep["detected_W"][label], c_rep["detected_W"][label],
                  0.03, "%s detected_W %s" % (name, label))


_SURF_SPECS = {
    "sphere": ("sphere 0.01 0 0 0.006",
               ("Sphere", ([0.01, 0, 0], 0.006))),
    "cylinder": ("cylinder 0.01 0 0 0 0.3 0.954 0.004",
                 ("Cylinder", ([0.01, 0, 0], [0, 0.3, 0.954], 0.004))),
    "cone": ("cone 0.02 0 0 -0.9 0.1 0.42 0.35",
             ("Cone", ([0.02, 0, 0], [-0.9, 0.1, 0.42], 0.35))),
    "torus": ("torus 0.03 0 0 0 0 1 0.02 0.004",
              ("Torus", ([0.03, 0, 0], [0, 0, 1], 0.02, 0.004))),
    "asphere": ("asphere 0.01 0 0 1 0 0 0.05 -1.2 0.012 2 5e3 -2e6",
                ("Asphere", ([0.01, 0, 0], [1, 0, 0], 0.05, -1.2,
                             [5e3, -2e6], 0.012))),
}


@pytest.mark.parametrize("kind", sorted(_SURF_SPECS))
def test_surface_roots_parity(kind, tmp_path):
    """Root-level fuzz: C surf_roots() vs surfaces.py on random rays —
    same hit/miss decisions, first-hit t within 1e-9 relative. This is the
    test that caught the quartic resolvent-root selection bug."""
    import subprocess
    from raytracer import surfaces as S
    tool = Path(cengine.binary_path()).parent / "tests" / "tool_surf_roots"
    if not tool.exists():
        pytest.skip("tool_surf_roots not built")
    spec_line, (cls_name, args) = _SURF_SPECS[kind]
    surf = getattr(S, cls_name)(*args)
    rng = np.random.default_rng(7)
    N = 20000
    o = np.stack([rng.uniform(-0.05, 0.0, N), rng.uniform(-0.03, 0.03, N),
                  rng.uniform(-0.01, 0.01, N)], axis=1)
    phi = rng.uniform(-0.4, 0.4, N)
    th = rng.uniform(-0.4, 0.4, N)
    d = np.stack([np.cos(phi) * np.cos(th), np.sin(phi),
                  np.sin(th) * np.cos(phi)], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t_py, ok = surf.intersect(o, d)
    t_py = np.where(ok & (t_py > 1e-7), t_py, np.inf)
    t_py = np.sort(t_py, axis=1)[:, 0]
    lines = [spec_line] + [
        "%.17g %.17g %.17g %.17g %.17g %.17g"
        % (o[i, 0], o[i, 1], o[i, 2], d[i, 0], d[i, 1], d[i, 2])
        for i in range(N)]
    out = subprocess.run([str(tool)], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True)
    t_c = np.array([
        min([float(x) for x in ln.split()
             if np.isfinite(float(x)) and float(x) > 1e-7],
            default=np.inf)
        for ln in out.stdout.splitlines()])
    assert len(t_c) == N
    hit_py = np.isfinite(t_py)
    hit_c = np.isfinite(t_c)
    n_dis = int((hit_py != hit_c).sum())
    # grazing rays may legitimately flip near tangency; allow a whisker
    assert n_dis <= max(2, N // 10000), \
        "%s: %d hit/miss disagreements" % (kind, n_dis)
    both = hit_py & hit_c
    rel = np.abs(t_py[both] - t_c[both]) / t_py[both]
    assert rel.max() < 1e-9, "%s: max rel root diff %g" % (kind, rel.max())


def test_thread_count_invariance(tmp_path):
    """Plan D2: results are a pure function of ray lineage — the detector
    cube must be BIT-identical across thread counts, the viz segment SET
    identical (ordering may differ), ledger sums within reordering ulps."""
    import subprocess
    model_json = cengine_scenes.write_scene("c_plate", tmp_path / "g")
    run_engine(model_json, tmp_path / "case", "c")
    req_path = tmp_path / "case" / "cengine" / "request_seed42.json"
    req = json.loads(req_path.read_text())
    cubes, vizzes, ledgers = [], [], []
    for th in (1, 7, 32):
        od = tmp_path / ("threads%d" % th)
        od.mkdir()
        req["out_dir"] = str(od)
        req["params"]["threads"] = th
        rp = od / "request.json"
        rp.write_text(json.dumps(req))
        subprocess.run([str(cengine.binary_path()), "--config", str(rp)],
                       check=True, capture_output=True)
        cubes.append(np.load(od / "det_0_inc.npy"))
        vizzes.append(np.array(sorted(map(tuple,
                                          np.load(od / "rays_viz.npy")))))
        ledgers.append(json.loads((od / "ledger.json").read_text()))
    assert np.array_equal(cubes[0], cubes[1])
    assert np.array_equal(cubes[0], cubes[2])
    assert np.array_equal(vizzes[0], vizzes[1])
    assert np.array_equal(vizzes[0], vizzes[2])
    s0 = ledgers[0]["sources"]["Src"]
    for lg in ledgers[1:]:
        st = lg["sources"]["Src"]
        for k, v in s0.items():
            # watt-scale fields: 1e-9 rel; closure_error itself is a
            # dimensionless ~1e-13 quantity — absolute floor covers it
            assert abs(st[k] - v) <= max(1e-9 * abs(v), 1e-12), k


def test_tlas_matches_linear_scan(tmp_path, monkeypatch):
    """The scene TLAS must be a pure accelerator: identical detector cube
    and ledger vs the brute-force linear scan (the phase-C analogue of
    test_mesh_bvh's BVH == brute force gate), on a real multi-face
    geometry."""
    model_json = REPO / "geometry" / "lens_pcx" / "model.json"
    if not model_json.exists():
        pytest.skip("geometry/lens_pcx not extracted")
    r_tlas = run_engine(model_json, tmp_path / "case_tlas", "c")
    monkeypatch.setenv("MIEWB_CENGINE_LINEAR", "1")
    r_lin = run_engine(model_json, tmp_path / "case_linear", "c")
    import h5py
    with h5py.File(next((tmp_path / "case_tlas" / "detectors").glob(
            "*.h5")), "r") as h:
        cube_t = h["spectral_cube_mean"][...]
    with h5py.File(next((tmp_path / "case_linear" / "detectors").glob(
            "*.h5")), "r") as h:
        cube_l = h["spectral_cube_mean"][...]
    assert np.array_equal(cube_t, cube_l), \
        "TLAS changed results vs linear scan"
    s_t = r_tlas["audit"]["per_seed"][0]["sources"]
    s_l = r_lin["audit"]["per_seed"][0]["sources"]
    for sname in s_t:
        for k, v in s_t[sname].items():
            assert abs(s_l[sname][k] - v) <= max(1e-12 * abs(v), 1e-15), \
                "%s.%s" % (sname, k)


def test_routing_reasons(tmp_path):
    """--engine auto routes deterministically and records the reason."""
    model_json = cengine_scenes.write_scene("c_plate", tmp_path / "g")
    case = tmp_path / "case_auto"
    r = run_engine(model_json, case, "auto")
    assert r["case"]["engine"] == "c"
    assert "ported" in r["case"]["engine_reason"]
