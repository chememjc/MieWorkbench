# =============================================================================
# test_grating_models.py — validation of the polarization- and wavelength-
# resolved grating models added to grating.py (README §1.9):
#   * bragg_kogelnik : Kogelnik (1969) transmission coupled-wave efficiency
#   * dammann        : exact Fourier orders of a binary +-pi phase profile
#   * table          : per-order eta_s/eta_p interpolation at the ray lambda
#   * polarized apply_to_batch: Jones rotation into the interface basis and
#     per-polarization amplitudes, with exact energy partition.
# The legacy lamellar behavior stays pinned by test_grating_roughness.py.
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          raytracer/tests/test_grating_models.py -q   (from scripts/)
# =============================================================================
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer import grating as gr                      # noqa: E402
from raytracer import fresnel as fr                       # noqa: E402
from raytracer.materials import MaterialError             # noqa: E402
from raytracer.rays import RayBatch                       # noqa: E402
from raytracer.audit import PowerLedger                   # noqa: E402


# ---------------------------------------------------------------------------
# bragg_kogelnik — pure coupled-wave efficiency formula
# ---------------------------------------------------------------------------
def test_kogelnik_efficiency_formula():
    # nu = pi/2 at exact Bragg (xi=0) -> full transfer to the first order.
    assert abs(gr.bragg_kogelnik_eta(np.pi / 2, 0.0) - 1.0) < 1e-9
    # nu = pi/4 -> sin^2(pi/4) = 1/2.
    assert abs(gr.bragg_kogelnik_eta(np.pi / 4, 0.0) - 0.5) < 1e-12
    # nu=xi=0 -> 0 (no modulation), and the [0,1] clamp holds.
    assert gr.bragg_kogelnik_eta(0.0, 0.0) == 0.0
    assert 0.0 <= gr.bragg_kogelnik_eta(3.0, 0.0) <= 1.0

    # Kogelnik Fig.6-style detuning zero: at nu=pi/2 with xi=sqrt(3)*nu the
    # argument sqrt(nu^2+xi^2) = 2*nu = pi, so sin^2(pi) = 0 exactly.
    nu = np.pi / 2
    xi = np.sqrt(3.0) * nu
    # analytic expectation (do NOT hand-wave): sin^2(2 nu) * nu^2/(4 nu^2)
    expect = np.sin(2 * nu) ** 2 * (nu * nu) / (nu * nu + xi * xi)
    assert abs(expect) < 1e-15
    assert abs(gr.bragg_kogelnik_eta(nu, xi) - expect) < 1e-12

    # detuning strictly reduces the first-order efficiency below the peak
    assert gr.bragg_kogelnik_eta(np.pi / 2, 0.3) < gr.bragg_kogelnik_eta(
        np.pi / 2, 0.0)


def _kogelnik_spec(lines_per_mm, dn, thickness_um, slant_deg=0.0,
                   orders=(-1, 0, 1, 2)):
    return {"model": "bragg_kogelnik", "lines_per_mm": lines_per_mm,
            "groove": "1,0,0", "orders": (orders[0], orders[-1]),
            "efficiencies": None, "table": None,
            "params": {"thickness_um": thickness_um, "dn": dn,
                       "slant_deg": slant_deg}}


def test_kogelnik_geometry_bragg_peak():
    # Construct exact-Bragg incidence for an unslanted transmission grating and
    # tune dn*d so that nu_s = pi/2 -> the +1 order takes ALL the power.
    lines_per_mm = 1000.0
    period = 1e-3 / lines_per_mm                 # 1 um
    lam = 500e-9
    sin_b = lam / (2 * period)                   # unslanted Bragg condition
    theta_b = np.arcsin(sin_b)
    cos_b = np.cos(theta_b)
    thickness_um = 100.0
    d = thickness_um * 1e-6
    # nu_s = pi*dn*d/(lam*cos_b) = pi/2  ->  dn = lam*cos_b/(2 d)
    dn = lam * cos_b / (2.0 * d)

    orders = [-1, 0, 1, 2]
    spec = _kogelnik_spec(lines_per_mm, dn, thickness_um, orders=orders)
    eta_s, eta_p = gr.order_efficiencies(spec, np.array([lam]),
                                         np.array([cos_b]), orders)
    col = {m: j for j, m in enumerate(orders)}

    # first order carries everything; zeroth is empty; energy is conserved.
    assert abs(eta_s[0, col[1]] - 1.0) < 1e-9
    assert abs(eta_s[0, col[0]] - 0.0) < 1e-9
    assert abs(eta_s[0, col[0]] + eta_s[0, col[1]] - 1.0) < 1e-12
    assert abs(eta_p[0, col[0]] + eta_p[0, col[1]] - 1.0) < 1e-12
    # no power leaks into the non-Bragg orders
    assert eta_s[0, col[-1]] == 0.0 and eta_s[0, col[2]] == 0.0

    # p-polarization coupling is reduced by cos(2 theta') in (0,90) deg, so the
    # first-order p efficiency sits strictly below the s efficiency here.
    two_theta = np.rad2deg(2 * theta_b)
    assert 0.0 < two_theta < 90.0
    assert eta_p[0, col[1]] < eta_s[0, col[1]]

    # detuning off Bragg (larger incidence angle) reduces the first order
    cos_off = np.cos(theta_b + np.deg2rad(3.0))
    eta_s_off, _ = gr.order_efficiencies(spec, np.array([lam]),
                                         np.array([cos_off]), orders)
    assert eta_s_off[0, col[1]] < eta_s[0, col[1]]


# ---------------------------------------------------------------------------
# dammann — exact Fourier orders of a binary +-pi phase profile
# ---------------------------------------------------------------------------
def _parseval_sum(tr, M):
    """Vectorized sum_{|m|<=M} |c_m|^2 for the binary +-pi Dammann profile."""
    xs = np.array([0.0] + list(tr) + [1.0])
    signs = np.array([(-1) ** j for j in range(len(xs) - 1)], dtype=float)
    m = np.arange(-M, M + 1)
    mm = m[m != 0]
    total = float((signs * (xs[1:] - xs[:-1])).sum()) ** 2      # |c_0|^2
    c = np.zeros(mm.shape, dtype=np.complex128)
    for j, s in enumerate(signs):
        c += s * (np.exp(-2j * np.pi * mm * xs[j + 1])
                  - np.exp(-2j * np.pi * mm * xs[j])) / (-2j * np.pi * mm)
    return total + float(np.sum(np.abs(c) ** 2))


def test_dammann_parseval_and_symmetry():
    tr = [0.03863, 0.39084]
    # The +-pi profile is DISCONTINUOUS (exp(i phi) jumps +1<->-1), so its
    # Fourier tail decays only as 1/m and Parseval converges as ~1/M. Reaching
    # 1e-6 needs a large summation window (|m| ~ 6e5), not just a few orders.
    assert _parseval_sum(tr, 200) < 1.0            # energy bound at any window
    assert abs(_parseval_sum(tr, 600_000) - 1.0) < 1e-6

    coeffs = gr.dammann_coefficients(tr, list(range(-6, 7)))
    # real (binary +-pi) profile => |c_m| = |c_{-m}| exactly.
    for m in range(1, 6):
        assert abs(abs(coeffs[m]) - abs(coeffs[-m])) < 1e-14
    eta = gr.dammann_efficiencies(tr, range(-5, 6))
    for m in range(1, 6):
        assert abs(eta[m] - eta[-m]) < 1e-14


def test_dammann_reduces_to_square_wave():
    # A single transition at 0.5 is the +/- pi square wave: its orders must
    # coincide with the lamellar duty=0.5 model (cross-model consistency).
    eta = gr.dammann_efficiencies([0.5], range(-3, 4))
    lam_eta, _ = gr.lamellar_efficiencies(list(range(-3, 4)), duty=0.5)
    for m in range(-3, 4):
        assert abs(eta[m] - lam_eta[m]) < 1e-12


def test_dammann_equal_intensity_1x5():
    # A genuine 1x5 equal-intensity Dammann design. NOTE: the literature
    # half-period pair 0.03863/0.39084 does NOT yield five equal orders under
    # the plain single-period +,-,+ construction this model uses (the +-1
    # orders dominate, max/min ~ 5); a true uniform 1x5 needs the optimized
    # full-period transition set below (the first point, 0.019304, is exactly
    # 0.03863/2). Uniformity here is < 0.1%.
    tr = [0.0193035565, 0.3676573095, 0.6323426905, 0.9806964435]
    eta = gr.dammann_efficiencies(tr, [-2, -1, 0, 1, 2])
    five = np.array([eta[m] for m in (-2, -1, 0, 1, 2)])
    assert five.min() > 0.15                       # each order well populated
    assert five.max() / five.min() - 1.0 < 0.01    # < 1% (published: assert 2%)
    # and symmetric
    assert abs(eta[1] - eta[-1]) < 1e-14
    assert abs(eta[2] - eta[-2]) < 1e-14


# ---------------------------------------------------------------------------
# table — per-order polarization-resolved efficiency interpolation
# ---------------------------------------------------------------------------
def _table_spec(orders=(-1, 0, 1), eta_s=0.8, eta_p=0.2, order=-1,
                lam_nm=(400.0, 800.0)):
    lam_um = np.array(lam_nm) * 1e-3
    table = {order: {"lam_um": lam_um,
                     "eta_s": np.full(len(lam_um), eta_s),
                     "eta_p": np.full(len(lam_um), eta_p)}}
    return {"model": "table", "lines_per_mm": 600.0, "groove": "1,0,0",
            "orders": (orders[0], orders[-1]), "efficiencies": None,
            "params": {}, "table": table}


def test_table_interpolation_roundtrip():
    # linearly-varying table -> exact interpolation at an interior wavelength
    lam_um = np.array([0.4, 0.8])
    table = {0: {"lam_um": lam_um, "eta_s": np.array([0.2, 0.6]),
                 "eta_p": np.array([0.1, 0.5])}}
    spec = {"model": "table", "lines_per_mm": 600.0, "groove": "1,0,0",
            "orders": (-1, 1), "efficiencies": None, "params": {},
            "table": table}
    orders = [-1, 0, 1]
    lam = np.array([600e-9])                   # 0.6 um -> midpoint
    eta_s, eta_p = gr.order_efficiencies(spec, lam, np.array([0.9]), orders)
    col = {m: j for j, m in enumerate(orders)}
    assert abs(eta_s[0, col[0]] - 0.4) < 1e-12
    assert abs(eta_p[0, col[0]] - 0.3) < 1e-12
    # orders absent from the table read zero
    assert eta_s[0, col[1]] == 0.0 and eta_s[0, col[-1]] == 0.0


def test_table_out_of_range_raises():
    spec = _table_spec()
    with pytest.raises(MaterialError):
        gr.order_efficiencies(spec, np.array([1000e-9]), np.array([0.9]),
                              [-1, 0, 1])          # 1.0 um > 0.8 um table max


def test_table_order_outside_table_is_zero():
    # a table order (-1) outside the requested spec order range is ignored,
    # and a requested order missing from the table reads 0.
    spec = _table_spec(orders=(0, 1), order=-1)    # table has -1 only
    eta_s, eta_p = gr.order_efficiencies(spec, np.array([500e-9]),
                                         np.array([0.9]), [0, 1])
    assert np.all(eta_s == 0.0) and np.all(eta_p == 0.0)


# ---------------------------------------------------------------------------
# polarized application through apply_to_batch (Jones rotation + energy)
# ---------------------------------------------------------------------------
class _FakeFace:
    surface = None

    def __init__(self, n_out):
        self._n = np.asarray(n_out, dtype=np.float64)

    def normal_out_of_solid(self, pos):
        return np.broadcast_to(self._n, (len(pos), 3)).copy()


class _FakeBody:
    def __init__(self):
        self.material = None      # -> n2 = 1 (no matdb needed)
        self.mirror = 0.0         # transmissive
        self.index = 0
        self.label = "grat"


class _FakeScene:
    def __init__(self, fid, spec, face, body):
        self.gratings = {fid: spec}
        self.faces = {fid: face}
        self._body = body

    def body_of_face(self, fid):
        return self._body

    def medium_index(self, mm, lam):
        return np.ones(np.shape(lam))    # ambient index 1


class _FakeCfg:
    max_reflections = 6


class _FakeTracer:
    def __init__(self, scene):
        self.scene = scene
        self.cfg = _FakeCfg()
        self.ledger = PowerLedger(1)


def _oblique_batch():
    # oblique incidence in the x-z plane so pol_basis is non-degenerate;
    # s_hat is set to the interface s so a pure-s / pure-p state stays pure.
    theta = np.deg2rad(20.0)
    d = np.array([np.sin(theta), 0.0, -np.cos(theta)])
    n_hat = np.array([0.0, 0.0, 1.0])
    s_hat = fr.pol_basis(d[None, :], n_hat[None, :])[0][0]
    grp = RayBatch(2)
    grp.pos[:] = 0.0
    grp.dir[:] = d
    grp.s_hat[:] = s_hat
    grp.lam[:] = 500e-9
    grp.source_id[:] = 0
    grp.Es[:] = [1.0, 0.0]       # ray0 pure-s, ray1 pure-p
    grp.Ep[:] = [0.0, 1.0]
    return grp, n_hat


def test_polarized_table_application_and_energy():
    fid = 7
    spec = _table_spec(orders=(-1, 0, 1), eta_s=0.8, eta_p=0.2, order=-1)
    grp, n_hat = _oblique_batch()
    scene = _FakeScene(fid, spec, _FakeFace([0.0, 0.0, 1.0]), _FakeBody())
    tracer = _FakeTracer(scene)

    p_in = float(grp.power.sum())                 # 1 + 1 = 2 W
    out = gr.apply_to_batch(tracer, fid, grp)
    assert out is not None and len(out) == 2      # both rays -> order -1

    # transmitted order powers: pure-s ray gets eta_s=0.8, pure-p gets eta_p=0.2
    assert abs(out.power[0] - 0.8) < 1e-12
    assert abs(out.power[1] - 0.2) < 1e-12
    # the child inherits the interface s-basis; pure-s stays pure-s
    assert abs(abs(out.Es[0]) ** 2 - 0.8) < 1e-12
    assert abs(out.Ep[0]) < 1e-12
    assert abs(abs(out.Ep[1]) ** 2 - 0.2) < 1e-12
    assert abs(out.Es[1]) < 1e-12

    # energy closes: children + credited absorption == incident power.
    p_children = float(out.power.sum())
    p_absorbed = float(tracer.ledger.buckets["absorbed_surface"].sum())
    assert abs(p_children + p_absorbed - p_in) < 1e-12
    # absorbed remainder is per-polarization: (1-0.8) + (1-0.2) = 1.0
    assert abs(p_absorbed - 1.0) < 1e-12
    # and it was tagged to the grating surface diagnostic
    assert abs(tracer.ledger.by_surface["grat:grating"] - 1.0) < 1e-12


def test_m0_lamellar_still_snell():
    # regression: with the default (lamellar) model, order m=0 diffracts along
    # the Snell-refracted direction (grating equation degenerates correctly).
    theta_i = np.deg2rad(30.0)
    d = np.array([[np.sin(theta_i), 0.0, -np.cos(theta_i)]])
    n_hat = np.array([[0.0, 0.0, 1.0]])
    g_hat = np.array([[1.0, 0.0, 0.0]])
    lam = np.array([633e-9])
    n1 = np.array([1.0])
    n2 = np.array([1.5])
    out = gr.order_directions(d, n_hat, g_hat, 600.0, lam, [0], n1, n2)
    dirs_t = out[0][0]
    cos_i = -np.sum(d * n_hat, axis=-1)
    expect = fr.refract_dir(d, n_hat, cos_i, n1, n2)
    assert np.max(np.abs(dirs_t - expect)) < 1e-12

    # and the default lamellar dispatch has eta_s == eta_p (scalar model)
    spec = {"lines_per_mm": 600.0, "groove": "v", "orders": (-1, 1),
            "efficiencies": None}       # no "model" key -> legacy lamellar
    eta_s, eta_p = gr.order_efficiencies(spec, lam, cos_i, [-1, 0, 1])
    assert np.array_equal(eta_s, eta_p)


def test_normal_incidence_grating_pair_closure():
    """Regression (pulsed-optics round, treacy_compressor demo): grating
    children inherited the INCIDENT s_hat verbatim; at exactly normal
    incidence pol_basis's degenerate fallback s is arbitrary-transverse,
    so a diffracted child's s_hat was not perpendicular to its own
    direction and the SECOND grating's Jones rotation silently lost
    cos^2(theta_d) of the power (9.3% closure leak). apply_to_batch now
    rebuilds each child's frame (n x d, sign-aligned)."""
    import warnings
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import scenehelpers as sh

    # G1 at normal incidence; the m=-1 order travels down-left and hits
    # G2 (a full-height slab behind the source plane, grating on its +x
    # cap) at theta_d = 28.7 deg -- the exact treacy leak topology. The
    # source sits between the two plates (source bodies are not
    # intersectable geometry, only the emission point matters).
    g1 = sh.slab_body("G1", "aluminum", 0.0, 0.002, half=0.01, mirror=1.0)
    g1["grating"] = {"G1.Pad.Face1": "600:0,1,0:orders=-1..1"}
    g2 = sh.slab_body("G2", "aluminum", -0.065, -0.063, half=0.06,
                      mirror=1.0)
    g2["grating"] = {"G2.Pad.Face2": "600:0,1,0:orders=-1..1"}
    src = sh.source_body(power_mW=1.0, lambdac_nm=800.0, coherent=False,
                         half=0.0005, x=-0.03,
                         polarization={"kind": "linear", "angle_deg": 0.0})
    # tilt the emit normal by 1e-7 in z: real FreeCAD-extracted scenes
    # carry ~1e-8 direction noise, so |d x n| at "normal" incidence is
    # TINY-but-nonzero -- pol_basis then normalizes noise into an
    # arbitrary transverse s (here forced to ~(0,-1,0)), which is the
    # exact preconditon of the leak (exact zeros take the clean
    # degenerate fallback and hide it)
    f = src["faces"][0]
    n = [1.0, 0.0, 1e-7]
    f["fingerprint"]["normal_hint"] = n
    f["surface"]["normal"] = [c / (1 + 5e-15) for c in n]
    model = sh.make_model([
        src,
        g1, g2,
        sh.detector_body(x=0.06, half=0.08),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, grids, scene = sh.trace_scene(model, rays=4000,
                                              max_reflections=8)
    rep = result.ledger.report(["Src"])
    # pre-fix this read closure_error ~ 0.05-0.09 (the skewed-basis
    # rotation at G2); the partition must be exact again
    assert rep["closure_ok"], rep["sources"]["Src"]
    # and the twice-diffracted arm really exists (G2's grating fired)
    assert "G2:grating" in rep["by_surface_W"]
