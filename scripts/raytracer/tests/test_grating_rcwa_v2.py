# =============================================================================
# test_grating_rcwa_v2.py — oracles for the P6 v2 RCWA grating tables
# (engine3 §7.5): complex per-order amplitude tables interpolated over
# (lambda, theta, phi), superseding the lambda-only real-efficiency format.
#
#   * v2 loader round-trip (schema, axes, orders, energy sanity)
#   * interpolation exactness at grid nodes + analytic trilinear midpoint
#   * OLD-format (v1) tables still behave BIT-IDENTICALLY (cos_i/azimuth
#     ignored; amplitude == sqrt(eta), real)
#   * diffracted-order PHASE from the table is carried into the child Jones
#     vector, and the energy ledger partition still closes exactly
#   * adaptive Wood/Rayleigh-anomaly refinement densifies the axis near the
#     analytic onset loci (asserted programmatically)
#
# Runs under the optics env (no meent needed — uses inline / shipped tables):
#   "$MIEWB_OPTICS_PYTHON" -m pytest raytracer/tests/test_grating_rcwa_v2.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer import grating as gr                        # noqa: E402
from raytracer import fresnel as fr                        # noqa: E402
from raytracer import optprops as op                       # noqa: E402
from raytracer.materials import MaterialError              # noqa: E402
from raytracer.rays import RayBatch                        # noqa: E402
from raytracer.audit import PowerLedger                    # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SHIPPED = (REPO / "opticalproperties" / "grating" / "tables"
           / "rcwa_fs_600_v2.mietab")


# ---------------------------------------------------------------------------
# helpers to author a v2 .mietab on disk
# ---------------------------------------------------------------------------
def _write_v2(path, lams_nm, thetas, phis, orders, amp_fn,
              side="transmission"):
    """amp_fn(lam_nm, th, ph, m) -> (amp_s complex, amp_p complex)."""
    with open(path, "w") as fh:
        fh.write("# mietab grating v2 side=%s\n" % side)
        fh.write("wavelength_nm,theta_deg,phi_deg,order,"
                 "amp_s_re,amp_s_im,amp_p_re,amp_p_im\n")
        for lam in lams_nm:
            for th in thetas:
                for ph in phis:
                    for m in orders:
                        a_s, a_p = amp_fn(lam, th, ph, m)
                        fh.write("%.6g,%.6g,%.6g,%d,%.10g,%.10g,%.10g,%.10g\n"
                                 % (lam, th, ph, m, a_s.real, a_s.imag,
                                    a_p.real, a_p.imag))


def _v2_spec(table):
    return {"model": "table", "lines_per_mm": 600.0, "groove": "1,0,0",
            "orders": (min(table["orders"]), max(table["orders"])),
            "efficiencies": None, "params": {}, "table": table}


# ---------------------------------------------------------------------------
# 1. loader round-trip
# ---------------------------------------------------------------------------
def test_v2_loader_roundtrip(tmp_path):
    p = tmp_path / "t.mietab"

    def amp(lam, th, ph, m):
        # magnitude 0.5 on +-1 so sum |amp|^2 = 0.5 <= 1
        a = 0.5 * np.exp(1j * np.deg2rad(lam * 0.1 + th + ph + 30 * m))
        return (a, 0.4 * a)

    _write_v2(p, [500, 600, 700], [0, 20, 40], [0, 15], [-1, 0, 1], amp)
    tbl = op._load_grating_table(str(p), "ctx")
    assert tbl["schema"] == "v2" and tbl["side"] == "transmission"
    assert np.allclose(tbl["lam_um"], [0.5, 0.6, 0.7])
    assert np.allclose(tbl["theta_deg"], [0, 20, 40])
    assert np.allclose(tbl["phi_deg"], [0, 15])
    assert tbl["orders"] == [-1, 0, 1]
    assert tbl["amp_s"][1].shape == (3, 3, 2)


def test_v2_loader_rejects_incomplete_grid(tmp_path):
    p = tmp_path / "hole.mietab"
    with open(p, "w") as fh:
        fh.write("# mietab grating v2 side=transmission\n")
        fh.write("wavelength_nm,theta_deg,phi_deg,order,"
                 "amp_s_re,amp_s_im,amp_p_re,amp_p_im\n")
        # two thetas for order 0 but only one for order 1 -> incomplete grid
        fh.write("500,0,0,0,0.1,0,0.1,0\n")
        fh.write("500,20,0,0,0.1,0,0.1,0\n")
        fh.write("500,0,0,1,0.1,0,0.1,0\n")
    with pytest.raises(MaterialError):
        op._load_grating_table(str(p), "ctx")


def test_v2_loader_rejects_energy_overflow(tmp_path):
    p = tmp_path / "hot.mietab"

    def amp(lam, th, ph, m):
        return (0.9 + 0j, 0.9 + 0j)          # sum over 2 orders = 1.62 > 1
    _write_v2(p, [500, 600], [0, 20], [0], [-1, 1], amp)
    with pytest.raises(MaterialError):
        op._load_grating_table(str(p), "ctx")


# ---------------------------------------------------------------------------
# 2. interpolation exactness
# ---------------------------------------------------------------------------
def test_v2_node_exact(tmp_path):
    p = tmp_path / "n.mietab"

    def amp(lam, th, ph, m):
        a = 0.5 * np.exp(1j * np.deg2rad(0.2 * lam + 2 * th + ph + 40 * m))
        return (a, 0.3 * a)
    _write_v2(p, [500, 600, 700], [0, 25, 50], [0, 10, 20], [-1, 0, 1], amp)
    tbl = op._load_grating_table(str(p), "ctx")
    spec = _v2_spec(tbl)
    # query exactly on nodes: lam=600 nm, theta=25, phi=10
    lam = np.array([600e-9]); cos_i = np.array([np.cos(np.deg2rad(25.0))])
    az = np.array([10.0])
    a_s, a_p = gr.order_amplitudes(spec, lam, cos_i, [-1, 0, 1], az)
    for j, m in enumerate([-1, 0, 1]):
        exp_s, exp_p = amp(600, 25, 10, m)
        assert abs(a_s[0, j] - exp_s) < 1e-9
        assert abs(a_p[0, j] - exp_p) < 1e-9
    # efficiency == |amp|^2
    es, ep = gr.order_efficiencies(spec, lam, cos_i, [-1, 0, 1], az)
    assert np.allclose(es, np.abs(a_s) ** 2)
    assert np.allclose(ep, np.abs(a_p) ** 2)


def test_v2_trilinear_midpoint_exact(tmp_path):
    # amplitude LINEAR in (lam, theta, phi) -> multilinear interp reproduces
    # the analytic midpoint value exactly.
    p = tmp_path / "lin.mietab"

    def lin(lam, th, ph):
        return (0.001 * lam) + (0.002 * th) + (0.003 * ph)

    def amp(lam, th, ph, m):
        v = lin(lam, th, ph)
        return (complex(v, 0.5 * v), complex(0.2 * v, -0.1 * v))
    _write_v2(p, [500, 700], [0, 40], [0, 20], [0], amp)
    tbl = op._load_grating_table(str(p), "ctx")
    spec = _v2_spec(tbl)
    lam = np.array([600e-9]); cos_i = np.array([np.cos(np.deg2rad(20.0))])
    az = np.array([10.0])
    a_s, a_p = gr.order_amplitudes(spec, lam, cos_i, [0], az)
    v = lin(600, 20, 10)
    assert abs(a_s[0, 0] - complex(v, 0.5 * v)) < 1e-9
    assert abs(a_p[0, 0] - complex(0.2 * v, -0.1 * v)) < 1e-9


def test_v2_clamps_out_of_grid(tmp_path):
    # theta/phi beyond the sampled range clamp to the nearest edge (no error);
    # lambda out of range still hard-errors would be nice, but v2 clamps all
    # three — assert the clamp holds the edge value.
    p = tmp_path / "c.mietab"

    def amp(lam, th, ph, m):
        return (complex(0.01 * th, 0.0), 0j)
    _write_v2(p, [500, 600], [0, 30], [0], [0], amp)
    tbl = op._load_grating_table(str(p), "ctx")
    spec = _v2_spec(tbl)
    lam = np.array([550e-9]); cos_i = np.array([np.cos(np.deg2rad(80.0))])
    a_s, _ = gr.order_amplitudes(spec, lam, cos_i, [0], np.array([0.0]))
    assert abs(a_s[0, 0] - complex(0.01 * 30, 0.0)) < 1e-9   # held at theta=30


# ---------------------------------------------------------------------------
# 3. legacy v1 tables unchanged (bit-identical)
# ---------------------------------------------------------------------------
def test_v1_amplitude_is_real_sqrt_eta_and_ignores_geometry():
    lam_um = np.array([0.4, 0.8])
    table = {-1: {"lam_um": lam_um, "eta_s": np.array([0.5, 0.5]),
                  "eta_p": np.array([0.2, 0.2])}}
    spec = {"model": "table", "lines_per_mm": 600.0, "groove": "1,0,0",
            "orders": (-1, 1), "params": {}, "table": table}
    lam = np.array([600e-9])
    # amplitude == sqrt(eta), purely real, and azimuth is ignored
    a_s, a_p = gr.order_amplitudes(spec, lam, np.array([0.9]), [-1, 0, 1],
                                   azimuth=np.array([33.0]))
    assert abs(a_s[0, 0] - np.sqrt(0.5)) < 1e-15 and a_s[0, 0].imag == 0.0
    assert abs(a_p[0, 0] - np.sqrt(0.2)) < 1e-15
    # cos_i does NOT change a v1 table result (lambda-only)
    e1, _ = gr.order_efficiencies(spec, lam, np.array([0.2]), [-1, 0, 1])
    e2, _ = gr.order_efficiencies(spec, lam, np.array([0.99]), [-1, 0, 1])
    assert np.array_equal(e1, e2)


# ---------------------------------------------------------------------------
# 4. apply_to_batch: phase carried + energy closes
# ---------------------------------------------------------------------------
class _FakeFace:
    surface = None

    def __init__(self, n_out):
        self._n = np.asarray(n_out, dtype=np.float64)

    def normal_out_of_solid(self, pos):
        return np.broadcast_to(self._n, (len(pos), 3)).copy()


class _FakeBody:
    material = None
    mirror = 0.0
    index = 0
    label = "grat"


class _FakeScene:
    def __init__(self, fid, spec, face, body):
        self.gratings = {fid: spec}
        self.faces = {fid: face}
        self._body = body

    def body_of_face(self, fid):
        return self._body

    def medium_index(self, mm, lam):
        return np.ones(np.shape(lam))


class _FakeCfg:
    max_reflections = 6


class _FakeTracer:
    def __init__(self, scene):
        self.scene = scene
        self.cfg = _FakeCfg()
        self.ledger = PowerLedger(1)


def _normal_incidence_batch(nrays=2):
    d = np.array([0.0, 0.0, -1.0])
    n_hat = np.array([0.0, 0.0, 1.0])
    s_hat = fr.pol_basis(d[None, :], n_hat[None, :])[0][0]
    grp = RayBatch(nrays)
    grp.pos[:] = 0.0
    grp.dir[:] = d
    grp.s_hat[:] = s_hat
    grp.lam[:] = 600e-9
    grp.source_id[:] = 0
    grp.Es[:] = 1.0 + 0.0j
    grp.Ep[:] = 0.0
    return grp, s_hat


def _make_v2_table_single_order(tmp_path, phase_deg, eta=0.5):
    p = tmp_path / "phase.mietab"

    a = np.sqrt(eta) * np.exp(1j * np.deg2rad(phase_deg))

    def amp(lam, th, ph, m):
        return (a, a) if m == 1 else (0j, 0j)
    _write_v2(p, [550, 650], [0, 30], [0], [0, 1], amp)
    return op._load_grating_table(str(p), "ctx")


def test_v2_apply_carries_phase_and_closes(tmp_path):
    tbl = _make_v2_table_single_order(tmp_path, phase_deg=57.0, eta=0.5)
    spec = _v2_spec(tbl)
    fid = 3
    grp, s_hat = _normal_incidence_batch()
    # remember incident s phase (0) for the phase-carry check
    scene = _FakeScene(fid, spec, _FakeFace([0.0, 0.0, 1.0]), _FakeBody())
    tracer = _FakeTracer(scene)
    p_in = float(grp.power.sum())

    out = gr.apply_to_batch(tracer, fid, grp)
    assert out is not None
    # only order +1 carries power (eta=0.5 for the pure-s rays)
    assert abs(float(out.power.sum()) - 0.5 * p_in) < 1e-9
    # the diffracted child's s-amplitude phase == the table phase (57 deg),
    # since the incident s phase was 0 and |amp|=sqrt(0.5).
    ph = np.degrees(np.angle(out.Es[0]))
    assert abs(((ph - 57.0 + 180) % 360) - 180) < 1e-6
    assert abs(abs(out.Es[0]) ** 2 - 0.5) < 1e-9
    # energy partition closes: children + absorbed == incident
    p_children = float(out.power.sum())
    p_absorbed = float(tracer.ledger.buckets["absorbed_surface"].sum())
    assert abs(p_children + p_absorbed - p_in) < 1e-9


# ---------------------------------------------------------------------------
# 5. adaptive Wood/Rayleigh-anomaly refinement
# ---------------------------------------------------------------------------
def _load_generator():
    sys.path.insert(0, str(REPO / "scripts"))
    from tools import gen_rcwa_table as g          # noqa: E402
    return g


def test_wood_anomaly_refinement_densifies_axis():
    g = _load_generator()
    period_um = 1.0e3 / 600.0
    n_top = 1.0
    sides = [1.0]
    lo, hi = -1, 1
    theta_min, theta_max = 0.0, 50.0
    base = np.linspace(theta_min, theta_max, 6)
    # analytic onset loci across the band
    loci = []
    for lam_um in np.linspace(0.45, 0.75, 7):
        loci += g.anomaly_loci_theta(lam_um, period_um, n_top, sides, lo, hi,
                                     theta_min, theta_max)
    assert loci, "expected at least one Wood anomaly in the band"
    refined = g.refine_axis(base, loci, theta_max - theta_min)
    assert refined.size > base.size                 # refinement added nodes
    # local spacing AROUND a locus is finer than the base spacing far from it.
    base_dtheta = (theta_max - theta_min) / 5.0
    for c in loci:
        near = refined[np.abs(refined - c) < 0.05 * (theta_max - theta_min)]
        if near.size >= 2:
            assert np.min(np.diff(np.sort(near))) < 0.5 * base_dtheta


def test_anomaly_loci_theta_matches_grating_equation():
    g = _load_generator()
    period_um = 1.6667
    # at the onset theta the |m| order must be exactly grazing: |sin th_in +
    # m lam/period| = 1 (n_top=n_side=1)
    lam_um = 0.5
    loci = g.anomaly_loci_theta(lam_um, period_um, 1.0, [1.0], -1, 1, 0.0, 89.0)
    for th in loci:
        vals = [abs(np.sin(np.deg2rad(th)) + m * lam_um / period_um)
                for m in (-1, 1)]
        assert min(abs(v - 1.0) for v in vals) < 1e-9


# ---------------------------------------------------------------------------
# 6. shipped table (skips if not generated in this tree)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SHIPPED.exists(), reason="shipped v2 table absent")
def test_shipped_table_loads_and_conserves():
    tbl = op._load_grating_table(str(SHIPPED), "shipped")
    assert tbl["schema"] == "v2"
    # co-polarized order sum <= 1 everywhere (loader already checks; re-assert)
    for pol in ("amp_s", "amp_p"):
        tot = sum(np.abs(tbl[pol][m]) ** 2 for m in tbl["orders"])
        assert np.all(tot <= 1.0 + 1e-6)


@pytest.mark.skipif(not SHIPPED.exists(), reason="shipped v2 table absent")
def test_e2e_scene_energy_closes_with_v2_table():
    """Full Scene build + trace through a transmission grating carrying the
    shipped @rcwa_fs_600_v2 table: the energy ledger partition must close and
    the v2 grating must actually fire."""
    import warnings
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import scenehelpers as sh

    g = sh.slab_body("G", "air", 0.0, 0.001, half=0.02)
    g["grating"] = {"G.Pad.Face1": "@rcwa_fs_600_v2:orders=-1..1"}
    src = sh.source_body(power_mW=1.0, lambdac_nm=550.0, coherent=False,
                         half=0.0008, x=-0.02,
                         polarization={"kind": "linear", "angle_deg": 0.0})
    model = sh.make_model([src, g, sh.detector_body(x=0.05, half=0.08)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, grids, scene = sh.trace_scene(model, rays=4000,
                                              max_reflections=6)
    rep = result.ledger.report(["Src"])
    assert rep["closure_ok"], rep["sources"]["Src"]
    assert "G:grating" in rep["by_surface_W"]
