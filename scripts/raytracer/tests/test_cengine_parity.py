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


def c_chunk_dir(case_dir, seed=42):
    """First C-engine trace chunk dir (P1 chunked-run layout:
    cengine/seed<seed>/chunk_<lo>_<hi>/). The direct-binary tests read the
    request/mask/inc/viz artifacts from here."""
    chunks = sorted((case_dir / "cengine" / ("seed%d" % seed)).glob(
        "chunk_*"), key=lambda p: int(p.name.split("_")[1]))
    return chunks[0]


def _uniaxial_material_names():
    """Crystal names in the uniaxial birefringence registry (first CSV
    column)."""
    import csv as _csv
    import common as _common
    path = _common.OPTPROPS_DIR / "birefringence" / "uniaxial.miebrf"
    with open(path, newline="") as fh:
        return {row["name"].strip().lower()
                for row in _csv.DictReader(fh) if row.get("name")}


def _is_uniaxial(model_json):
    """True if the scene has a uniaxial-crystal body. The C engine's
    birefringence kernel implements the legacy effective-index amplitudes,
    while the Python default is the EXACT Lekner-1991 path (which routes to
    Python, not C). Parity for these scenes is tested on the SHARED approx
    path via --biref-approx; biaxial bodies always Python-route regardless."""
    names = _uniaxial_material_names()
    doc = json.loads(Path(model_json).read_text())
    for b in doc.get("bodies", []):
        mat = b.get("material")
        if mat is not None and str(mat).strip().lower() in names:
            return True
    return False


def run_engine(model_json, case_dir, engine):
    import run_trace
    argv = [
        "--model-json", str(model_json),
        "--case-dir", str(case_dir),
        "--rays", str(RAYS),
        "--resolution", str(RESOLUTION),
        "--nlambda", str(NLAMBDA),
        "--spectral-bins", "8",
        "--engine", engine,
        "--workers", "1",
    ]
    if _is_uniaxial(model_json):
        argv.append("--biref-approx")
    rc = run_trace.main(argv)
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
    c_mask_own = np.load(c_chunk_dir(cc["case_dir"]) / "det_0_mask.npy")
    assert np.array_equal(c_mask_own.astype(bool), py_mask.astype(bool)), \
        "%s: C trim mask differs from Python DetectorGrid mask" % name

    # ---- viz rows: same shape contract, sane values ----
    assert py["rays"].shape[1] == 13 and cc["rays"].shape[1] == 13
    assert cc["rays"].shape[0] > 0
    assert np.all(np.isfinite(cc["rays"]))


REPO = SCRIPTS.parent


def test_coherent_doubleslit_parity(tmp_path):
    """Phase D gate: the C engine's Huygens gather vs the Python engine on
    the REAL doubleslit geometry — Young fringes must agree in placement
    and contrast. C-vs-Python differ by RNG realization only, so the bar
    is the Python seed-to-seed level (measured: 2D corr ~0.86 at 5e4
    rays; peak-row correlation is far tighter)."""
    import h5py
    model_json = REPO / "geometry" / "doubleslit" / "model.json"
    if not model_json.exists():
        pytest.skip("geometry/doubleslit not extracted")

    def run(engine, case):
        import run_trace
        rc = run_trace.main([
            "--model-json", str(model_json), "--case-dir", str(case),
            "--rays", "100000", "--resolution", "256", "--nlambda", "1",
            "--engine", engine, "--workers", "1"])
        assert rc == 0
        with h5py.File(next((case / "detectors").glob("*.h5")), "r") as h:
            return h["spectral_cube_mean"][...].sum(axis=0)

    a = run("python", tmp_path / "py")
    b = run("c", tmp_path / "c")
    case = json.loads((tmp_path / "c" / "case.json").read_text())
    assert case["engine"] == "c", case.get("engine_reason")
    # energy: integrals agree within MC bounds
    ra = float(a.sum())
    rb = float(b.sum())
    assert abs(ra - rb) / max(ra, rb) < 0.05, (ra, rb)
    # fringe structure: peak-row profiles strongly correlated
    pa = a[a.sum(axis=1).argmax()]
    pb = b[b.sum(axis=1).argmax()]
    corr = float(np.corrcoef(pa, pb)[0, 1])
    assert corr > 0.9, "fringe profile correlation %.3f" % corr


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

    # spatial: irradiance-weighted centroid within ~2 px per detector
    # (catches e.g. a wrong calcite walk-off displacement that power
    # totals would miss); scenes may have several detectors (hot_mirror)
    import h5py
    for h5p in sorted((py["case_dir"] / "detectors").glob("*.h5")):
        with h5py.File(h5p, "r") as h:
            py_img = h["spectral_cube_mean"][...].sum(axis=0)
        with h5py.File(cc["case_dir"] / "detectors" / h5p.name, "r") as h:
            c_img = h["spectral_cube_mean"][...].sum(axis=0)
        if py_img.sum() <= 0 or c_img.sum() <= 0:
            continue                    # empty arm at these test params
        yy, xx = np.mgrid[0:py_img.shape[0], 0:py_img.shape[1]]
        for ax, gr in (("x", xx), ("y", yy)):
            pc = (py_img * gr).sum() / py_img.sum()
            ccn = (c_img * gr).sum() / c_img.sum()
            assert abs(pc - ccn) < 2.0, \
                "%s %s centroid_%s %.2f vs %.2f px" % (
                    name, h5p.name, ax, pc, ccn)


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


def test_particles_continuum_parity(tmp_path):
    """Phase G: continuum particle medium — ballistic decay, scattered
    children, particle_absorbed bucket — vs the Python engine.
    Statistical (MC scattering)."""
    import run_trace
    model_json = cengine_scenes.write_scene("c_plate", tmp_path / "g")
    spec = ("box=8,-15,-15:10,30,30;material=water;phi=0.05;"
            "median_um=10;gsd=1.6")

    def run(engine, case):
        rc = run_trace.main([
            "--model-json", str(model_json), "--case-dir", str(case),
            "--rays", "20000", "--resolution", "64", "--nlambda", "1",
            "--spectral-bins", "4", "--engine", engine, "--workers", "1",
            "--particles", spec])
        assert rc == 0
        return json.loads((case / "audit.json").read_text())["per_seed"][0]

    py = run("python", tmp_path / "py")
    case_c = tmp_path / "c"
    cc = run("c", case_c)
    assert json.loads((case_c / "case.json").read_text())["engine"] == "c"
    for rep in (py, cc):
        assert rep["closure_ok"], rep
    ps = py["sources"]["Src"]
    cs = cc["sources"]["Src"]
    emitted = ps["emitted_W"]
    for b in ("particle_absorbed", "escaped", "absorbed_bulk"):
        assert abs(ps[b] - cs[b]) <= max(0.03 * abs(ps[b]),
                                         2e-3 * emitted), \
            "%s: %g vs %g" % (b, ps[b], cs[b])
    for label in py["detected_W"]:
        rel_close(py["detected_W"][label], cc["detected_W"][label], 0.05,
                  "particles detected_W %s" % label)


def test_thread_count_invariance(tmp_path):
    """Plan D2: results are a pure function of ray lineage — the detector
    cube must be BIT-identical across thread counts, the viz segment SET
    identical (ordering may differ), ledger sums within reordering ulps."""
    import subprocess
    model_json = cengine_scenes.write_scene("c_plate", tmp_path / "g")
    run_engine(model_json, tmp_path / "case", "c")
    req_path = c_chunk_dir(tmp_path / "case") / "request.json"
    req = json.loads(req_path.read_text())
    # exercise the C engine's OWN in-binary gather here (the P1 pipeline
    # routes coherent gather to Python; this direct-binary test still covers
    # the C gather's thread-invariance by turning gather_skip back off).
    req["params"]["gather_skip"] = False
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


def test_spectrum_scene_strata_parity(tmp_path):
    """Feature D5: a tabulated-emission source routes to C automatically
    (spectrum is no gate feature) and the C request's lambda union is the
    SAME inverse-CDF equal-power quantile strata that Python's
    wavelength_strata produces — so the two engines sample identical
    wavelengths, and detected power matches to the deterministic bar."""
    model_json = cengine_scenes.write_scene("c_spectrum", tmp_path / "geo")
    cc = run_engine(model_json, tmp_path / "case_c", "c")
    assert cc["case"]["engine"] == "c", cc["case"].get("engine_reason")

    # Python reference strata for the same source at NLAMBDA
    import common
    from raytracer.scene import Scene
    from raytracer.sources import wavelength_strata
    from raytracer.optprops import load_optical_properties
    model = json.loads(model_json.read_text())
    common.validate_model(model)
    op = load_optical_properties()
    scene = Scene(model, op.matdb, op.coatings, optprops=op)
    _, src = scene.sources[0]
    ref = wavelength_strata(src, NLAMBDA)

    req = json.loads(
        (c_chunk_dir(tmp_path / "case_c") / "request.json").read_text())
    lams_c = np.asarray(req["lams_m"])
    assert lams_c.shape == ref.shape
    assert np.allclose(lams_c, ref, rtol=0, atol=1e-15), (lams_c, ref)
    # strata land inside the LED-B1 table range (400-700 nm)
    assert lams_c.min() * 1e9 >= 400.0 and lams_c.max() * 1e9 <= 700.0


def test_routing_reasons(tmp_path):
    """--engine auto routes deterministically and records the reason."""
    model_json = cengine_scenes.write_scene("c_plate", tmp_path / "g")
    case = tmp_path / "case_auto"
    r = run_engine(model_json, case, "auto")
    assert r["case"]["engine"] == "c"
    assert "ported" in r["case"]["engine_reason"]


# ---------------------------------------------------------------------------
# P7 tranche 1: pulsed-optics time domain
# ---------------------------------------------------------------------------
def _run_time(model_json, case_dir, engine, extra):
    """run_trace over a scene with the P7 time-domain flags. extra is the
    extra CLI list (--gdd-budget and/or --time-products ...)."""
    import run_trace
    argv = [
        "--model-json", str(model_json),
        "--case-dir", str(case_dir),
        "--rays", str(RAYS),
        "--resolution", "64",
        "--nlambda", str(NLAMBDA),
        "--spectral-bins", "8",
        "--engine", engine,
        "--workers", "1",
        "--seeds", "1",
    ] + list(extra)
    rc = run_trace.main(argv)
    assert rc == 0, "run_trace --engine %s exited %s" % (engine, rc)
    return json.loads((case_dir / "case.json").read_text())


def test_gdd_budget_parity(tmp_path):
    """P7 tranche 1 (gdd_budget): the C engine tallies the per-body
    power-weighted bulk path; build_gdd_budget resolves the dispersion in
    Python UNCHANGED. On a CW glass slab (deterministic normal incidence)
    the C case.json gdd_budget block must match the Python one — path lengths
    to ~1e-9, and every DISPERSION quantity (n_g / gd / gdd / tod) BIT-exact
    since both engines call the identical Python material stencil."""
    model_json = cengine_scenes.write_scene("c_plate", tmp_path / "geometry")
    py = _run_time(model_json, tmp_path / "case_py", "python", ["--gdd-budget"])
    cc = _run_time(model_json, tmp_path / "case_c", "c", ["--gdd-budget"])
    assert py["engine"] == "python"
    assert cc["engine"] == "c", cc.get("engine_reason")
    pb, cb = py.get("gdd_budget"), cc.get("gdd_budget")
    assert pb is not None and cb is not None, (pb, cb)
    assert pb["reference_source"] == cb["reference_source"]
    rel_close(pb["lambda_ref_nm"], cb["lambda_ref_nm"], 1e-12, "lambda_ref")
    assert [r["label"] for r in pb["rows"]] == [r["label"] for r in cb["rows"]]
    for pr, cr in zip(pb["rows"], cb["rows"]):
        assert pr["material"] == cr["material"]
        # geometric path: statistical (different RNG realizations), 1e-9 bar
        rel_close(pr["L_bar_mm"], cr["L_bar_mm"], 1e-9,
                  "%s L_bar_mm" % pr["label"])
        # dispersion quantities: n_g is BIT-identical (Python stencil on the
        # same reference wavelength); gd/gdd/tod inherit L_bar's tiny drift
        rel_close(pr["n_g"], cr["n_g"], 1e-14, "%s n_g" % pr["label"])
        for k in ("gd_fs", "gdd_fs2", "tod_fs3"):
            rel_close(pr[k], cr[k], 1e-9, "%s %s" % (pr["label"], k))
    for k in ("gd_fs", "gdd_fs2", "tod_fs3"):
        rel_close(pb["total"][k], cb["total"][k], 1e-9, "total %s" % k)


def _load_time(case_dir):
    """(time_attrs dict, time_profile array or None) from the single
    detector .h5."""
    import h5py
    h5s = sorted((case_dir / "detectors").glob("*.h5"))
    assert len(h5s) == 1
    with h5py.File(h5s[0], "r") as h:
        attrs = {k: h.attrs[k] for k in h.attrs}
        prof = h["time_profile"][...] if "time_profile" in h else None
    return attrs, prof


def _profile_moments(attrs, prof):
    """(mean_t, rms_width, integral_W) of a time_profile density [W/s]."""
    t_lo = float(attrs["t_lo_s"])
    dt = float(attrs["time_dt_s"])
    tc = t_lo + (np.arange(len(prof)) + 0.5) * dt
    w = prof.sum()
    mean = float((tc * prof).sum() / w)
    rms = float(np.sqrt(((tc - mean) ** 2 * prof).sum() / w))
    return mean, rms, float(prof.sum() * dt)


def test_time_products_cw_parity(tmp_path):
    """P7 tranche 1 (time_products): a CW glass slab with explicit
    --time-products. Every ray's group path is identical at collimated normal
    incidence, so the arrival-time attrs AND the binned profile agree to fp
    precision between engines (the gopl accumulator is deterministic given
    the geometric path; the Python finalize_time bins the C records)."""
    model_json = cengine_scenes.write_scene("c_plate", tmp_path / "geometry")
    extra = ["--time-products", "pulse,spectrogram", "--time-bins", "128"]
    py = _run_time(model_json, tmp_path / "case_py", "python", extra)
    cc = _run_time(model_json, tmp_path / "case_c", "c", extra)
    assert py["engine"] == "python"
    assert cc["engine"] == "c", cc.get("engine_reason")
    pa, pp = _load_time(tmp_path / "case_py")
    ca, cp = _load_time(tmp_path / "case_c")
    for k in ("t_p001_s", "t_p999_s", "time_total_W", "time_dt_s",
              "t_lo_s", "t_hi_s"):
        rel_close(float(pa[k]), float(ca[k]), 1e-9, "time attr %s" % k)
    assert int(pa["time_n_records"]) == int(ca["time_n_records"])
    pm, pw, pi = _profile_moments(pa, pp)
    cm, cw, ci = _profile_moments(ca, cp)
    rel_close(pm, cm, 1e-9, "profile mean-t")
    rel_close(pi, ci, 1e-9, "profile integral")


def test_time_products_pulse_gdd_parity(tmp_path):
    """P7 tranche 1: a broadband VIRTUAL-pulse source (pulse_duration on a
    power source) auto-enables time products; the analytic-envelope FWHM
    folds in each stratum's angular bandwidth AND the material GDD the ray
    accumulated through the slab (gdd_acc). Deterministic collimated scene =>
    the pulse profile (mean, width, integral) matches to fp precision, and
    the co-active gdd_budget block agrees too."""
    from scenehelpers import (source_body, slab_body, detector_body,
                              make_model)
    model = make_model([
        source_body("Src", x=-0.02, half=0.004, power_mW=2.0,
                    lambdac_nm=800.0, lambdamin_nm=760.0, lambdamax_nm=840.0,
                    pulse_duration_ps=0.1),
        slab_body("Plate", "bk7", 0.0, 0.01, half=0.02),
        detector_body("Det", x=0.03, half=0.025)])
    geo = tmp_path / "geometry"
    geo.mkdir(parents=True, exist_ok=True)
    (geo / "model.json").write_text(json.dumps(model))
    py = _run_time(geo / "model.json", tmp_path / "case_py", "python",
                   ["--gdd-budget"])
    cc = _run_time(geo / "model.json", tmp_path / "case_c", "c",
                   ["--gdd-budget"])
    assert cc["engine"] == "c", cc.get("engine_reason")
    # auto-enable fired (no explicit --time-products flag)
    assert py["time_products"]["auto_enabled"]
    pa, pp = _load_time(tmp_path / "case_py")
    ca, cp = _load_time(tmp_path / "case_c")
    assert int(pa["time_n_records"]) == int(ca["time_n_records"])
    pm, pw, pi = _profile_moments(pa, pp)
    cm, cw, ci = _profile_moments(ca, cp)
    rel_close(pm, cm, 1e-9, "pulse profile mean-t")
    rel_close(pw, cw, 1e-9, "pulse profile rms-width")
    rel_close(pi, ci, 1e-9, "pulse profile integral")
    # gdd_budget rides along on the same track_time tally
    rel_close(py["gdd_budget"]["rows"][0]["gdd_fs2"],
              cc["gdd_budget"]["rows"][0]["gdd_fs2"], 1e-9, "co-active gdd")


def test_time_products_spm_chirp_parity(tmp_path):
    """P7 tranche 1: an SPM source installs an exact-FFT SPD + an S-curve
    chirp via per-stratum birth-time offsets (sources.install_spm ->
    _stratum_t0). The C engine births each stratum's gopl at c*t0[stratum]
    (apply_stratum_t0), spreading the arrival window; the profile matches
    Python to fp precision."""
    from scenehelpers import (source_body, slab_body, detector_body,
                              make_model)
    sb = source_body("Src", x=-0.02, half=0.004, power_mW=5.0,
                     lambdac_nm=800.0, pulse_duration_ps=0.05)
    sb["source"]["spm"] = "phimax:3.0"
    model = make_model([sb, slab_body("Plate", "bk7", 0.0, 0.005, half=0.02),
                        detector_body("Det", x=0.03, half=0.025)])
    geo = tmp_path / "geometry"
    geo.mkdir(parents=True, exist_ok=True)
    (geo / "model.json").write_text(json.dumps(model))
    py = _run_time(geo / "model.json", tmp_path / "case_py", "python", [])
    cc = _run_time(geo / "model.json", tmp_path / "case_c", "c", [])
    assert cc["engine"] == "c", cc.get("engine_reason")
    pa, pp = _load_time(tmp_path / "case_py")
    ca, cp = _load_time(tmp_path / "case_c")
    # the chirp actually spread the arrival window (t0 offsets are nonzero)
    assert float(pa["t_hi_s"]) - float(pa["t_lo_s"]) > float(pa["time_dt_s"])
    for k in ("t_lo_s", "t_hi_s", "t_p001_s", "t_p999_s"):
        rel_close(float(pa[k]), float(ca[k]), 1e-9, "spm attr %s" % k)
    pm, pw, pi = _profile_moments(pa, pp)
    cm, cw, ci = _profile_moments(ca, cp)
    rel_close(pm, cm, 1e-9, "spm profile mean-t")
    rel_close(pw, cw, 1e-9, "spm profile rms-width")
