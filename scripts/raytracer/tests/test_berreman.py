# =============================================================================
# test_berreman.py — P9 full-anisotropy Berreman 4x4 validation
# (scripts/raytracer/berreman.py).  These are the merge-gate oracles from
# engine3 Sec 7.4 stage 2 / Sec 15 P9.  Two independent 4x4 formulations
# (Berreman here, Lekner in birefringence.py) arbitrate each other; Fresnel
# arbitrates the isotropic limit; energy (Poynting-flux) closure arbitrates the
# branch/normalization; the quartz activity + SiC reststrahlen numbers pin the
# gyrotropic and absorbing physics.
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_berreman.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer import fresnel as fr             # noqa: E402
from raytracer import birefringence as bi       # noqa: E402
from raytracer import berreman as bz            # noqa: E402
from . import scenehelpers as sh                # noqa: E402

CAL_NO, CAL_NE = 1.658, 1.486          # calcite (negative uniaxial)
QTZ_NO, QTZ_NE = 1.5443, 1.5534        # quartz  (positive uniaxial)
KTP = np.array([1.7377 ** 2, 1.7453 ** 2, 1.8297 ** 2])   # biaxial @1064


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _geom(thetas_deg, alpha_deg, phi_deg):
    """Interface normal +z; plane of incidence x-z; incident ray d_z<0.  Optic
    axis at polar alpha from the normal, azimuth phi from +x (== P6's helper)."""
    th = np.deg2rad(np.atleast_1d(thetas_deg).astype(float))
    n = th.shape[0]
    nh = np.tile([0, 0, 1.0], (n, 1))
    d = np.stack([np.sin(th), np.zeros(n), -np.cos(th)], axis=1)
    al, ph = np.deg2rad(alpha_deg), np.deg2rad(phi_deg)
    c = np.tile([np.sin(al) * np.cos(ph), np.sin(al) * np.sin(ph),
                 np.cos(al)], (n, 1))
    return d, nh, c


def _uniaxial_epsG(c, n_o, n_e):
    """Global eps tensor(s) for a uniaxial crystal with optic axis c: the two
    transverse principal axes carry n_o, the c-axis carries n_e."""
    c = _unit(np.atleast_2d(c))
    n = c.shape[0]
    ax = np.zeros_like(c)
    ax[np.arange(n), np.argmin(np.abs(c), axis=1)] = 1.0
    e1 = _unit(np.cross(c, ax))
    e2 = np.cross(c, e1)
    frame = np.stack([e1, e2, c], axis=1)           # rows principal axes; z=c
    ep = np.tile([n_o ** 2, n_o ** 2, n_e ** 2], (n, 1)).astype(complex)
    return bz.eps_tensor(ep, frame)


# ===========================================================================
# ORACLE 2 — isotropic reduction to Fresnel (1e-10)
# ===========================================================================
def test_isotropic_reduction_to_fresnel():
    N = 200
    thetas = np.linspace(0.1, 80, N)
    d, nh, _ = _geom(thetas, 0, 0)
    ncr = 1.55
    epsG = np.eye(3) * ncr ** 2
    a = bz.anis_interface_in(d, nh, epsG, np.full(N, 1.0))
    cos_i = -np.sum(d * nh, axis=1)
    rs, rp, ts, tp, ct = fr.fresnel_coeffs(cos_i, np.full(N, 1.0, complex),
                                           np.full(N, ncr, complex))
    Rs, Rp, Ts, Tp = fr.power_coeffs(rs, rp, ts, tp, cos_i, ct,
                                     np.full(N, 1.0, complex),
                                     np.full(N, ncr, complex))
    assert np.max(np.abs(a["rss"] - rs)) < 1e-10
    assert np.max(np.abs(a["rpp"] - rp)) < 1e-10
    assert np.max(np.abs(a["rsp"])) < 1e-10
    assert np.max(np.abs(a["rps"])) < 1e-10
    assert np.max(np.abs(a["Ts"] - Ts)) < 1e-10
    assert np.max(np.abs(a["Tp"] - Tp)) < 1e-10


# ===========================================================================
# ORACLE 1 — uniaxial-through-Berreman == Lekner (1e-10), both directions of
# the full reflection Jones + transmitted power, dense (theta, alpha, phi) grid
# ===========================================================================
@pytest.mark.parametrize("no,ne", [(CAL_NO, CAL_NE), (QTZ_NO, QTZ_NE)])
def test_reduction_to_lekner(no, ne):
    worst_jones = 0.0
    worst_T = 0.0
    for alpha in np.linspace(0, 90, 10):
        for phi in np.linspace(0, 90, 10):
            thetas = np.linspace(0.5, 78, 30)
            d, nh, c = _geom(thetas, alpha, phi)
            n1 = np.full(len(thetas), 1.0)
            lek = bi.uniaxial_interface_in(d, nh, c, n1,
                                           np.full(len(thetas), no),
                                           np.full(len(thetas), ne))
            epsG = _uniaxial_epsG(c, no, ne)
            a = bz.anis_interface_in(d, nh, epsG, n1)
            for k in ("rss", "rsp", "rps", "rpp"):
                worst_jones = max(worst_jones, np.max(np.abs(a[k] - lek[k])))
            T_lek = np.abs(lek["tos"]) ** 2 + np.abs(lek["tes"]) ** 2
            worst_T = max(worst_T, np.max(np.abs(a["Ts"] - T_lek)))
    assert worst_jones < 1e-10, worst_jones
    assert worst_T < 1e-10, worst_T


# ===========================================================================
# ORACLE 6 — Poynting-flux energy closure
# ===========================================================================
@pytest.mark.parametrize("eps_princ", [
    KTP,                                   # biaxial
    np.array([CAL_NO ** 2, CAL_NO ** 2, CAL_NE ** 2]),   # uniaxial calcite
])
def test_lossless_energy_closure(eps_princ):
    worst = 0.0
    for alpha in np.linspace(0, 90, 8):
        for phi in np.linspace(0, 90, 8):
            thetas = np.linspace(0.5, 72, 20)
            d, nh, c = _geom(thetas, alpha, phi)
            cc = _unit(c)
            ax = np.zeros_like(cc)
            ax[np.arange(len(cc)), np.argmin(np.abs(cc), axis=1)] = 1.0
            e1 = _unit(np.cross(cc, ax))
            e2 = np.cross(cc, e1)
            frame = np.stack([e1, e2, cc], axis=1)
            epsG = bz.eps_tensor(
                np.tile(eps_princ, (len(thetas), 1)).astype(complex), frame)
            a = bz.anis_interface_in(d, nh, epsG, np.full(len(thetas), 1.0))
            worst = max(worst, np.max(np.abs(a["Rs"] + a["Ts"] - 1.0)),
                        np.max(np.abs(a["Rp"] + a["Tp"] - 1.0)))
    assert worst < 1e-10, worst


def test_absorbing_slab_energy_closure():
    """R + T + A == 1 with A the flux deficit, for a tilted absorbing biaxial
    slab over a range of angles (mode-mixing, no closed form — closure is the
    constraint).  S-matrix recursion stays finite for the absorbing layer."""
    na, nb, nc = complex(2.0, 0.05), complex(2.3, 0.02), complex(1.7, 0.08)
    al = np.deg2rad(35.0)
    x = np.array([np.cos(al), 0, np.sin(al)])
    y = np.array([0, 1.0, 0])
    frame = np.stack([x, y, np.cross(x, y)])[None]
    epsG = bz.eps_tensor(np.array([[na ** 2, nb ** 2, nc ** 2]], complex),
                         frame)
    for thdeg in (0.1, 20, 40, 55, 70):
        th = np.deg2rad(thdeg)
        d = np.array([[np.sin(th), 0, -np.cos(th)]])
        nh = np.array([[0, 0, 1.0]])
        res = bz.slab_smatrix([epsG], [5e-6], np.array([633e-9]), d, nh,
                              np.array([1.0]), np.array([1.0]))
        s = res["R"][0] + res["T"][0] + res["A"][0]
        assert abs(s - 1.0) < 1e-10, (thdeg, s)
        assert res["A"][0] > 0.1        # genuinely absorbing


# ===========================================================================
# ORACLE 3 — alpha-quartz optical activity 21.77 deg/mm @ 589.3 nm
# ===========================================================================
QUARTZ_RHO = 21.77          # deg/mm rotatory power (registry datum)
QUARTZ_LAM = 589.3e-9


def test_quartz_gyration_eigenindex_split():
    """The Berreman eps->Delta->eigensplit chain, with the gyration calibrated
    to the cited rotatory power, produces two CIRCULAR eigenmodes whose index
    split reproduces 21.77 deg/mm.  Validates the whole gyrotropic machinery
    end to end (a bug in eps construction / Delta / sorting breaks it even with
    calibrated G)."""
    G = bz.gyration_from_rotatory_power(QUARTZ_RHO, QUARTZ_LAM, QTZ_NO)
    epsG = bz.eps_tensor(np.array([[QTZ_NO ** 2, QTZ_NO ** 2, QTZ_NE ** 2]],
                                  complex), np.eye(3)[None])
    khat = np.array([[0, 0, 1.0]])              # propagation along optic axis
    epsG = bz.add_gyration(epsG, G * khat)
    d = np.array([[0, 0, 1.0]])
    nh = np.array([[0, 0, -1.0]])
    xh, yh, zh, _, _ = bz.local_frame(d, nh)
    epsL = bz.eps_to_local(epsG, xh, yh, zh)
    pw = bz.partial_waves(epsL, np.array([0.0]))
    qz = np.real(pw["qz"][0])
    fwd = np.sort(qz[0:2])
    dn = fwd[1] - fwd[0]
    rho = np.rad2deg(np.pi * dn / QUARTZ_LAM) * 1e-3
    assert abs(rho - QUARTZ_RHO) < 0.02, rho
    # the two forward eigenmodes are (near-)circular: |E_x| ~ |E_y|, 90 deg
    E, _ = bz._fields_from_state(pw["Psi"][:, :, 0:2], epsL, np.array([0.0]),
                                 pw["qz"][:, 0:2])
    for m in (0, 1):
        ex, ey = E[0, m, 0], E[0, m, 1]
        assert abs(abs(ex) - abs(ey)) < 1e-3 * abs(ex)
        phase = np.angle(ey / ex)
        assert min(abs(phase - np.pi / 2), abs(phase + np.pi / 2)) < 1e-2


def test_quartz_slab_rotation():
    """A 1 mm quartz slab between index-matched (n_o) half-spaces rotates a
    linear input by 21.77 deg (interface reflections nulled; pure bulk
    circular birefringence through slab_smatrix)."""
    G = bz.gyration_from_rotatory_power(QUARTZ_RHO, QUARTZ_LAM, QTZ_NO)
    epsG = bz.eps_tensor(np.array([[QTZ_NO ** 2, QTZ_NO ** 2, QTZ_NE ** 2]],
                                  complex), np.eye(3)[None])
    epsG = bz.add_gyration(epsG, G * np.array([[0, 0, 1.0]]))
    d = np.array([[0, 0, 1.0]])
    nh = np.array([[0, 0, -1.0]])
    res = bz.slab_smatrix([epsG], [1e-3], np.array([QUARTZ_LAM]), d, nh,
                          np.array([QTZ_NO]), np.array([QTZ_NO]))
    tss, tps = res["tss"][0], res["tps"][0]
    theta = np.rad2deg(0.5 * np.arctan2(
        2 * np.real(np.conj(tss) * tps),
        np.abs(tss) ** 2 - np.abs(tps) ** 2))
    assert abs(abs(theta) - QUARTZ_RHO) < 0.05, theta
    assert res["A"][0] < 1e-6                 # lossless
    assert res["T"][0] > 0.999


# ===========================================================================
# ORACLE 4 — absorbing anisotropic vs analytic complex-index Fresnel.
# 4H-SiC reststrahlen (polar-dielectric Lorentz phonon model; the Passler-
# Paarmann material class, JOSA B 34, 2128 (2017)).  At normal incidence and
# in the c-perp-to-plane geometry the modes decouple into isotropic complex
# Fresnel, giving an exact reference; Berreman matches to machine precision
# (far better than the "few %" gate).
# ===========================================================================
def _sic_eps(wn, branch):
    # (eps_inf, wLO, wTO, gamma) cm^-1 for 4H-SiC ordinary / extraordinary
    p = {"o": (6.56, 970.0, 797.0, 4.0), "e": (6.72, 967.0, 782.0, 4.0)}[branch]
    ei, wLO, wTO, g = p
    return ei * (wLO ** 2 - wn ** 2 - 1j * g * wn) \
        / (wTO ** 2 - wn ** 2 - 1j * g * wn)


def test_sic_reststrahlen_normal_incidence():
    d = np.array([[0, 0, 1.0]])
    nh = np.array([[0, 0, -1.0]])
    for wn in (750.0, 800.0, 850.0, 900.0, 950.0, 1000.0):
        eo, ee = _sic_eps(wn, "o"), _sic_eps(wn, "e")
        n_o = np.sqrt(eo)
        epsG = bz.eps_tensor(np.array([[eo, eo, ee]], complex), np.eye(3)[None])
        a = bz.anis_interface_in(d, nh, epsG, np.array([1.0]))
        Rb = abs(a["rss"][0]) ** 2
        Rf = abs((1 - n_o) / (1 + n_o)) ** 2
        assert abs(Rb - Rf) < 1e-9, (wn, Rb, Rf)


def test_absorbing_uniaxial_c_perp_poi_decouples():
    """Absorbing uniaxial with c perpendicular to the plane of incidence: the
    s-mode (E||c) is the extraordinary (complex n_e), the p-mode is ordinary
    (complex n_o); Berreman reproduces both isotropic complex-Fresnel curves to
    machine precision with zero cross coupling, at oblique incidence."""
    eo, ee = _sic_eps(820.0, "o"), _sic_eps(820.0, "e")
    n_o, n_e = np.sqrt(eo), np.sqrt(ee)
    for thdeg in (20.0, 45.0, 60.0):
        th = np.deg2rad(thdeg)
        d = np.array([[np.sin(th), 0, -np.cos(th)]])
        nh = np.array([[0, 0, 1.0]])
        epsG = bz.eps_tensor(np.array([[eo, ee, eo]], complex), np.eye(3)[None])
        a = bz.anis_interface_in(d, nh, epsG, np.array([1.0]))
        rs_e, _, _, _, _ = fr.fresnel_coeffs(np.array([np.cos(th)]),
                                             np.array([1.0], complex),
                                             np.array([n_e]))
        _, rp_o, _, _, _ = fr.fresnel_coeffs(np.array([np.cos(th)]),
                                             np.array([1.0], complex),
                                             np.array([n_o]))
        assert abs(abs(a["rss"][0]) - abs(rs_e[0])) < 1e-9
        assert abs(abs(a["rpp"][0]) - abs(rp_o[0])) < 1e-9
        assert abs(a["rsp"][0]) < 1e-9 and abs(a["rps"][0]) < 1e-9


# ===========================================================================
# eps-tensor construction sanity
# ===========================================================================
# ===========================================================================
# ORACLE 5 — biaxial e2e (tracer routes the biaxial interface through Berreman
# by default; effective-index under --biref-approx).  Energy closure < 1e-3 on
# BOTH paths, and the amplitude CHANGE vs effective-index is reported (the
# finding table, engine3 Sec 7.4 prediction: exact at principal alignment /
# near-normal, O(1%) at steep oblique off-principal incidence).
# ===========================================================================
def _ktp_slab_scene(biref_approx, rays=40000, seed=3):
    import common
    from raytracer.scene import Scene
    from raytracer.sources import sample_source
    from raytracer.tracer import Tracer, TraceConfig
    from raytracer.detector import DetectorGrid
    from raytracer.optprops import load_optical_properties
    c = np.sqrt(0.5)
    bodies = [
        sh.source_body("Src", x=-0.02, half=0.002, power_mW=1.0,
                       lambdac_nm=633.0,
                       polarization={"kind": "linear", "angle_deg": 45.0}),
        sh.slab_body("Ktp", "ktp", 0.0, 0.004, half=0.01,
                     crystal_axis=[c, 0.0, c], crystal_axis2=[0.0, 1.0, 0.0]),
        sh.detector_body("Det", x=0.03, half=0.02),
    ]
    model = sh.make_model(bodies)
    common.validate_model(model)
    op = load_optical_properties()
    scene = Scene(model, op.matdb, op.coatings, optprops=op)
    grids = {fid: DetectorGrid(scene.faces[fid], 128, 8, (500e-9, 760e-9),
                               label=scene.faces[fid].id)
             for fid in scene.detector_faces}
    cfg = TraceConfig(rays=rays, n_lambda=1, seed=seed, power_floor=1e-12,
                      max_reflections=6, biref_approx=biref_approx)
    tracer = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(seed)
    batches = [sample_source(scene, scene.bodies[b], s, i, cfg.rays,
                             cfg.n_lambda, rng, ledger=tracer.ledger)
               for i, (b, s) in enumerate(scene.sources)]
    result = tracer.run(batches)
    rep = result.ledger.report(result.source_names)
    return max(s["closure_error"] for s in rep["sources"].values())


def test_biaxial_e2e_energy_closure_both_paths():
    """A KTP slab traced end-to-end closes to < 1e-3 with the Berreman path
    (default) AND with --biref-approx: the exact power difference lands in
    absorbed_surface, so closure holds either way (the tracer integration
    contract)."""
    assert _ktp_slab_scene(False) < 1e-3      # Berreman (default)
    assert _ktp_slab_scene(True) < 1e-3       # effective-index


def test_biaxial_effective_index_finding():
    """FINDING (engine3 Sec 7.4, not a hard gate): the reflected-Jones error
    |exact Berreman - effective-index| for KTP with the X principal axis at
    polar alpha in the x-z plane, s-input, over theta.  Representative table
    (L2 error of the reflected (rss, rps)):

        alpha  th=5     20       40       60       75
        0.0    ~1e-16   ~1e-15   ~1e-16   ~1e-16   1.1e-2
        22.5   ~1e-16   ~1e-16   ~1e-15   7.2e-3   4.2e-3
        45.0   ~1e-16   ~1e-15   1.8e-3   ~1e-14   9.2e-4

    Exact (== effective-index) at principal alignment + near-normal; grows to
    O(1%) at steep oblique off-principal incidence (the biaxial analogue of
    P6's uniaxial azimuth finding)."""
    eps = KTP.astype(complex)
    errs = {}
    for alpha in (0.0, 22.5, 45.0):
        al = np.deg2rad(alpha)
        xax = np.array([np.sin(al), 0.0, np.cos(al)])
        frame = np.stack([xax, [0.0, 1.0, 0.0], np.cross(xax, [0, 1.0, 0])])
        row = []
        for theta in (5, 20, 40, 60, 75):
            th = np.deg2rad(theta)
            d = np.array([[np.sin(th), 0.0, -np.cos(th)]])
            nh = np.array([[0.0, 0.0, 1.0]])
            s_new, p_new = fr.pol_basis(d, nh)
            ci = -np.sum(d * nh, axis=1)
            # exact Berreman reflected (rss, rps) for s-input
            a = bz.anis_interface_in(d, nh, bz.eps_tensor(eps[None], frame),
                                     np.array([1.0]))
            # effective-index reflected (mirror the --biref-approx tracer path)
            fr3 = np.broadcast_to(frame, (1, 3, 3))
            res = bi.refract_in_biaxial(d, nh, fr3, 1.0, KTP[None])
            im = bi.biaxial_modes_for_k(d, fr3, KTP[None])
            E1, E2 = fr.rotate_jones(np.ones(1, complex), np.zeros(1, complex),
                                     s_new, p_new, im["D_slow"], im["D_fast"])
            cs = np.sum(im["D_slow"] * s_new, 1)
            sn = np.sum(im["D_slow"] * p_new, 1)
            n1c = np.array([1.0], complex)
            rs1, rp1, _, _, _ = fr.fresnel_coeffs(
                ci, n1c, res["n_phase_slow"].astype(complex))
            rs2, rp2, _, _, _ = fr.fresnel_coeffs(
                ci, n1c, res["n_phase_fast"].astype(complex))
            rEs = E1 * cs * rs1 - E2 * sn * rs2
            rEp = E1 * sn * rp1 + E2 * cs * rp2
            row.append(float(np.hypot(abs(a["rss"][0] - rEs[0]),
                                      abs(a["rps"][0] - rEp[0]))))
        errs[alpha] = np.array(row)
    # SHAPE assertions (engine3 Sec 7.4 prediction)
    #  - principal-aligned + near-normal is exact
    assert np.max(errs[0.0][:3]) < 1e-9
    #  - the overall discrepancy is O(1%), reached at steep oblique incidence
    allmax = max(np.max(v) for v in errs.values())
    assert 1e-3 < allmax < 5e-2, allmax


def test_eps_tensor_isotropic_and_hermitian_gyration():
    epsG = bz.eps_tensor(np.array([[2.25, 2.25, 2.25]], complex),
                         np.eye(3)[None])
    assert np.max(np.abs(epsG[0] - 2.25 * np.eye(3))) < 1e-13
    # rotation similarity preserves eigenvalues (principal indices)
    c = _unit(np.array([[0.3, 0.5, 0.8]]))
    epsU = _uniaxial_epsG(c, CAL_NO, CAL_NE)
    ev = np.sort(np.real(np.linalg.eigvals(epsU[0])))
    assert np.allclose(ev, np.sort([CAL_NO ** 2, CAL_NO ** 2, CAL_NE ** 2]),
                       atol=1e-10)
    # gyration keeps eps Hermitian (lossless reciprocal activity)
    g = bz.add_gyration(epsU.astype(complex), np.array([[0.0, 0.0, 1e-4]]))
    assert np.max(np.abs(g[0] - g[0].conj().T)) < 1e-15
