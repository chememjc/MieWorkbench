# =============================================================================
# test_biaxial.py — closed-form validation of the biaxial two-sheet solver in
# birefringence.py: quartic normal-surface roots, D eigenmodes, walk-off,
# uniaxial/isotropic limits, interface in/out/internal-reflection.
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_biaxial.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raytracer import fresnel as fr            # noqa: E402
from raytracer import birefringence as bi      # noqa: E402
from . import scenehelpers as sh               # noqa: E402

# KTP @ 1064 nm (Kato & Takaoka 2002), the canonical biaxial oracle
KTP_NX, KTP_NY, KTP_NZ = 1.7377, 1.7453, 1.8297
KTP_EPS = np.array([KTP_NX ** 2, KTP_NY ** 2, KTP_NZ ** 2])
# calcite / quartz as (n_o, n_o, n_e) biaxial degenerations
CAL_NO, CAL_NE = 1.658, 1.486
QTZ_NO, QTZ_NE = 1.5443, 1.5534

I3 = np.eye(3)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _H(K, eps):
    """The quartic normal surface (crystal frame == global here)."""
    u = np.sum(K ** 2, axis=-1)
    P = np.sum(eps * K ** 2, axis=-1)
    w = eps * (np.sum(eps) - eps)
    Q = np.sum(w * K ** 2, axis=-1)
    return u * P - Q + np.prod(eps)


def _incident_fan(n=40, max_deg=75.0):
    th = np.deg2rad(np.linspace(1.0, max_deg, n))
    d = np.stack([np.sin(th), np.zeros(n), np.cos(th)], axis=-1)
    nh = np.broadcast_to([0.0, 0.0, -1.0], (n, 3)).copy()
    return d, nh


# ---------------------------------------------------------------------------
# normal surface + transversality
# ---------------------------------------------------------------------------
def test_biaxial_normal_surface_residual():
    d, nh = _incident_fan()
    out = bi.refract_in_biaxial(d, nh, I3, 1.0, KTP_EPS)
    for m in ("slow", "fast"):
        assert not out["tir_%s" % m].any()
        K = out["n_phase_%s" % m][:, None] * out["k_%s" % m]
        res = _H(K, KTP_EPS)
        # relative residual against the O(eps^3) scale of H
        assert np.max(np.abs(res)) < 1e-9 * np.prod(KTP_EPS)


def test_biaxial_eigenmode_orthogonality_and_transversality():
    rng = np.random.default_rng(7)
    k = _unit(rng.normal(size=(200, 3)))
    modes = bi.biaxial_modes_for_k(k, I3, KTP_EPS)
    assert np.max(np.abs(np.sum(modes["D_slow"] * k, axis=-1))) < 1e-12
    assert np.max(np.abs(np.sum(modes["D_fast"] * k, axis=-1))) < 1e-12
    assert np.max(np.abs(
        np.sum(modes["D_slow"] * modes["D_fast"], axis=-1))) < 1e-12
    assert np.all(modes["n_slow"] >= modes["n_fast"] - 1e-12)
    assert np.all(modes["n_slow"] <= KTP_NZ + 1e-9)
    assert np.all(modes["n_fast"] >= KTP_NX - 1e-9)


def test_refract_in_matches_modes_for_k_per_sheet():
    d, nh = _incident_fan()
    out = bi.refract_in_biaxial(d, nh, I3, 1.0, KTP_EPS)
    for m in ("slow", "fast"):
        modes = bi.biaxial_modes_for_k(out["k_%s" % m], I3, KTP_EPS)
        n_sheet = modes["n_%s" % m]
        assert np.max(np.abs(n_sheet - out["n_phase_%s" % m])) < 1e-9


# ---------------------------------------------------------------------------
# limits: isotropic and uniaxial
# ---------------------------------------------------------------------------
def test_biaxial_isotropic_limit_is_snell():
    n2 = 1.52
    eps = np.array([n2 ** 2] * 3)
    d, nh = _incident_fan()
    out = bi.refract_in_biaxial(d, nh, I3, 1.0, eps)
    cos_i = -np.sum(d * nh, axis=-1)
    snell = fr.refract_dir(d, nh, cos_i, np.full(len(d), 1.0),
                           np.full(len(d), n2))
    for m in ("slow", "fast"):
        # double-root (fully degenerate) case: accuracy is bounded by the
        # sqrt(machine-eps) eigenvalue splitting, not the 1e-9 of the
        # simple-root paths
        assert np.max(np.abs(out["k_%s" % m] - snell)) < 1e-7
        assert np.max(np.abs(out["s_%s" % m] - snell)) < 1e-7
        assert np.max(np.abs(out["n_phase_%s" % m] - n2)) < 1e-7
        assert np.max(np.abs(out["n_ray_%s" % m] - n2)) < 1e-7


def _check_uniaxial_limit(n_o, n_e):
    eps = np.array([n_o ** 2, n_o ** 2, n_e ** 2])   # optic axis = z
    c = np.array([0.0, 0.0, 1.0])
    d, nh = _incident_fan()
    n = len(d)
    uni = bi.refract_in(d, nh, c, 1.0, np.full(n, n_o), np.full(n, n_e))
    bia = bi.refract_in_biaxial(d, nh, I3, 1.0, eps)
    # sheet <-> o/e assignment by phase index (negative uniaxial: o is
    # slow; positive uniaxial: e is slow)
    for i in range(n):
        pairs = {
            "o": (uni["k_o"][i], n_o, d[i] * 0),          # s_o == k_o
            "e": (uni["k_e"][i], uni["n_phase_e"][i], uni["s_e"][i]),
        }
        for m in ("slow", "fast"):
            n_b = bia["n_phase_%s" % m][i]
            # match to the closer uniaxial mode
            mode = "o" if abs(n_b - n_o) <= abs(n_b - pairs["e"][1]) else "e"
            k_u, n_u, s_u = pairs[mode]
            assert abs(n_b - n_u) < 5e-9
            assert np.max(np.abs(bia["k_%s" % m][i] - k_u)) < 5e-9
            if mode == "e":
                assert np.max(np.abs(bia["s_%s" % m][i] - s_u)) < 5e-9
            else:
                assert np.max(np.abs(bia["s_%s" % m][i] - k_u)) < 5e-9


def test_biaxial_uniaxial_limit_negative():
    _check_uniaxial_limit(CAL_NO, CAL_NE)


def test_biaxial_uniaxial_limit_positive():
    _check_uniaxial_limit(QTZ_NO, QTZ_NE)


# ---------------------------------------------------------------------------
# principal-plane walk-off oracle (KTP): in the x-z plane the in-plane mode
# behaves EXACTLY like a uniaxial e-wave with (n_o=n_x, n_e=n_z, c=z), and
# the y-polarized mode has n = n_y with zero walk-off.
# ---------------------------------------------------------------------------
def test_ktp_principal_plane_walkoff_matches_uniaxial_formula():
    # angle of the optic axes from z in the x-z plane; keep clear of it
    inv = 1.0 / KTP_EPS
    tan2 = (inv[0] - inv[1]) / (inv[1] - inv[2])
    th_axis = np.arctan(np.sqrt(tan2))
    for th_deg in (10.0, 30.0, 75.0):
        th = np.deg2rad(th_deg)
        assert abs(th - th_axis) > np.deg2rad(5.0)
        k = np.array([[np.sin(th), 0.0, np.cos(th)]])
        modes = bi.biaxial_modes_for_k(k, I3, KTP_EPS)

        # in-plane sheet: uniaxial-like n(theta) with n_o=n_x, n_e=n_z
        n_inplane = bi.n_e_theta(np.cos(th), KTP_NX, KTP_NZ)[()]
        which = ("slow" if abs(modes["n_slow"][0] - n_inplane)
                 < abs(modes["n_fast"][0] - n_inplane) else "fast")
        other = "fast" if which == "slow" else "slow"
        assert abs(modes["n_%s" % which][0] - n_inplane) < 1e-9
        assert abs(modes["n_%s" % other][0] - KTP_NY) < 1e-9
        # y-mode is polarized along y and walks off nowhere
        assert abs(abs(modes["D_%s" % other][0, 1]) - 1.0) < 1e-9

        K_in = n_inplane * k
        s_ray, n_phase, n_ray = bi.biaxial_ray_from_k(K_in, I3, KTP_EPS)
        rho_bi = np.arccos(np.clip(np.sum(s_ray[0] * k[0]), -1, 1))
        rho_uni = bi.walkoff_angle(np.cos(th), KTP_NX, KTP_NZ)[()]
        assert abs(rho_bi - abs(rho_uni)) < 1e-9
        assert abs(n_ray[0] - n_phase[0] * np.cos(rho_bi)) < 1e-12

        K_y = KTP_NY * k
        s_y, _, n_ray_y = bi.biaxial_ray_from_k(K_y, I3, KTP_EPS)
        assert np.max(np.abs(s_y[0] - k[0])) < 1e-9
        assert abs(n_ray_y[0] - KTP_NY) < 1e-9


# ---------------------------------------------------------------------------
# interface: TIR ordering, internal reflection, exit
# ---------------------------------------------------------------------------
def test_biaxial_tir_fast_sheet_first():
    # steep incidence from a dense medium: the fast (lower-index) sheet
    # goes evanescent before the slow one
    n1 = 1.90
    th = np.deg2rad(np.linspace(60.0, 89.0, 60))
    d = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=-1)
    nh = np.broadcast_to([0.0, 0.0, -1.0], d.shape).copy()
    out = bi.refract_in_biaxial(d, nh, I3, n1, KTP_EPS)
    # whenever the slow sheet is evanescent the fast one must be too
    assert not np.any(out["tir_slow"] & ~out["tir_fast"])
    # and the sweep must actually exhibit the fast-only-TIR band
    assert np.any(out["tir_fast"] & ~out["tir_slow"])
    assert np.any(out["tir_fast"] & out["tir_slow"])


def test_reflect_internal_biaxial_isotropic_is_specular():
    n2 = 1.52
    eps = np.array([n2 ** 2] * 3)
    rng = np.random.default_rng(3)
    k_hat = _unit(rng.normal(size=(50, 3)))
    nh = _unit(-k_hat + 0.3 * rng.normal(size=(50, 3)))
    # ensure the normal is against the wave
    flip = np.sum(k_hat * nh, axis=-1) > 0
    nh[flip] = -nh[flip]
    K_in = n2 * k_hat
    K_refl, ok = bi.reflect_internal_biaxial(K_in, nh, I3, eps)
    assert ok.all()
    cos_i = -np.sum(k_hat * nh, axis=-1)
    spec = K_in + 2.0 * (n2 * cos_i)[:, None] * nh
    assert np.max(np.abs(K_refl - spec)) < 1e-7


def test_reflect_internal_biaxial_stays_on_surface():
    th = np.deg2rad(np.array([20.0, 40.0, 60.0]))
    k = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=-1)
    modes = bi.biaxial_modes_for_k(k, I3, KTP_EPS)
    K_in = modes["n_slow"][:, None] * k
    # reflect off a face whose normal opposes the wave at 45 deg
    nh = _unit(np.broadcast_to([-np.sin(np.pi / 4), 0.0,
                                -np.cos(np.pi / 4)], k.shape).copy())
    K_refl, ok = bi.reflect_internal_biaxial(K_in, nh, I3, KTP_EPS)
    assert ok.all()
    res = _H(K_refl, KTP_EPS)
    assert np.max(np.abs(res)) < 1e-9 * np.prod(KTP_EPS)
    # tangential wavevector conserved
    t_in = K_in - np.sum(K_in * nh, axis=-1)[:, None] * nh
    t_out = K_refl - np.sum(K_refl * nh, axis=-1)[:, None] * nh
    assert np.max(np.abs(t_in - t_out)) < 1e-9
    # mode-preserving: the reflected phase index stays on the slow sheet
    k_r = _unit(K_refl)
    m_r = bi.biaxial_modes_for_k(k_r, I3, KTP_EPS)
    n_r = np.linalg.norm(K_refl, axis=-1)
    assert np.max(np.abs(n_r - m_r["n_slow"])) < 1e-9


def test_parallel_plate_roundtrip_restores_direction():
    # air -> biaxial plate -> air with parallel faces: both sheets exit
    # parallel to the incident beam (lateral offset only)
    d, nh = _incident_fan(n=25, max_deg=70.0)
    out = bi.refract_in_biaxial(d, nh, I3, 1.0, KTP_EPS)
    for m in ("slow", "fast"):
        K_int = out["n_phase_%s" % m][:, None] * out["k_%s" % m]
        # exit face normal against the internal (+z-going) wave: for a
        # plate with parallel faces that is the SAME direction as the
        # entry-side normal (both point back toward -z)
        d_out, tir = bi.refract_out_biaxial(K_int, nh, 1.0)
        assert not tir.any()
        assert np.max(np.abs(d_out - d)) < 1e-9


def test_biaxial_registry_loads_and_pins_ktp_indices():
    from raytracer import optprops
    props = optprops.load_optical_properties()
    assert set(props.biaxial) >= {"ktp", "kta", "lbo", "bibo"}
    assert props.matdb.is_biaxial("ktp")
    assert not props.matdb.is_biaxial("calcite")   # uniaxial stays uniaxial
    mx, my, mz = props.matdb.get_biaxial("ktp")
    lam = 1.064e-6
    got = [m.n_complex(lam).real for m in (mx, my, mz)]
    for g, want in zip(got, (KTP_NX, KTP_NY, KTP_NZ)):
        assert abs(g - want) < 2e-3, (got, (KTP_NX, KTP_NY, KTP_NZ))
    assert props.biaxial["ktp"]["reference"]


# ---------------------------------------------------------------------------
# scene-level tracer integration: KTP walk-off plate
# ---------------------------------------------------------------------------
def _ktp_plate_model(polarization=None, t=0.015):
    """Normal-incidence KTP slab with the X principal axis at 45 deg in the
    global x-z plane (Y principal = global y): the beam propagates in the
    X-Z principal plane at 45 deg — maximum-walk-off geometry. The
    y-polarized sheet goes straight (n = n_y); the in-plane sheet walks
    off in global z."""
    c = np.sqrt(0.5)
    bodies = [
        sh.source_body(x=-0.01, half=0.00015, coherent=False,
                       polarization=polarization),
        sh.slab_body("KTP", "ktp", 0.0, t, half=0.008,
                     crystal_axis=[c, 0.0, c],
                     crystal_axis2=[0.0, 1.0, 0.0]),
        sh.detector_body(x=t + 0.005, half=0.01),
    ]
    return sh.make_model(bodies)


def _spots_z(det, thresh=0.2):
    """Intensity-centroid spot positions [m] along global z (simplified
    from test_scenes_e2e._spots_along)."""
    img = det.inc.sum(axis=0)
    ax_is_x = abs(det.xhat[2]) > abs(det.yhat[2])
    if ax_is_x:
        prof = img.sum(axis=0)
        coord = det.x_lo + (np.arange(det.W) + 0.5) * det.pixel_m
        sgn = np.sign(det.xhat[2])
    else:
        prof = img.sum(axis=1)
        coord = det.y_lo + (np.arange(det.H) + 0.5) * det.pixel_m
        sgn = np.sign(det.yhat[2])
    above = prof > thresh * prof.max()
    spots, i, n = [], 0, len(prof)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        w = prof[i:j]
        spots.append(float(np.sum(coord[i:j] * w) / np.sum(w)))
        i = j
    return sorted(sgn * s for s in spots)


def _expected_walkoff_dz(t, lam_nm=633.0):
    """Predicted in-plane-sheet transverse displacement from the (already
    unit-pinned) solver itself, for the _ktp_plate_model geometry."""
    from raytracer import optprops
    props = optprops.load_optical_properties()
    mx, my, mz = props.matdb.get_biaxial("ktp")
    lam = lam_nm * 1e-9
    eps = np.array([[np.real(m.n_complex(lam)) ** 2
                     for m in (mx, my, mz)]])
    c = np.sqrt(0.5)
    x_ax = np.array([c, 0.0, c])
    y_ax = np.array([0.0, 1.0, 0.0])
    frame = np.stack([x_ax, y_ax, np.cross(x_ax, y_ax)])
    k = np.array([[1.0, 0.0, 0.0]])
    modes = bi.biaxial_modes_for_k(k, frame, eps)
    # the in-plane sheet is the one NOT polarized along global y
    name = "slow" if abs(modes["D_slow"][0, 1]) < 0.5 else "fast"
    K = modes["n_%s" % name][:, None] * k
    s_ray, _, _ = bi.biaxial_ray_from_k(K, frame, eps)
    return t * s_ray[0, 2] / s_ray[0, 0]


def test_ktp_plate_scene_double_spot_and_closure():
    model = _ktp_plate_model()
    res, grids, scene = sh.trace_scene(model, rays=15000, resolution=400)
    rep = res.ledger.report(res.source_names)
    assert max(s["closure_error"] for s in rep["sources"].values()) < 1e-3

    det = list(grids.values())[0]
    spots = _spots_z(det)
    assert len(spots) == 2, "expected 2 spots, got %r (mm)" \
        % [round(s * 1e3, 3) for s in spots]
    dz = _expected_walkoff_dz(t=0.015)
    got = max(spots, key=abs)
    straight = min(spots, key=abs)
    assert abs(straight) < 1e-4
    assert abs(got - dz) < 0.05 * abs(dz), \
        "walk-off spot at %.4f mm vs solver-predicted %.4f mm" \
        % (got * 1e3, dz * 1e3)


def test_ktp_plate_polarization_selects_sheet():
    # linear:90 = global +y -> the straight (n_y) sheet only;
    # linear:0 = global +z -> the walking in-plane sheet only
    dz = _expected_walkoff_dz(t=0.015)
    _, g_y, _ = sh.trace_scene(
        _ktp_plate_model({"kind": "linear", "angle_deg": 90}),
        rays=12000, resolution=400)
    _, g_z, _ = sh.trace_scene(
        _ktp_plate_model({"kind": "linear", "angle_deg": 0}),
        rays=12000, resolution=400)
    sy = _spots_z(list(g_y.values())[0])
    sz = _spots_z(list(g_z.values())[0])
    assert len(sy) == 1 and abs(sy[0]) < 1e-4, \
        "y-pol should give one straight spot, got %r" % sy
    assert len(sz) == 1, "z-pol should give one spot, got %r" % sz
    assert abs(sz[0] - dz) < 0.05 * abs(dz)


def test_biaxial_frame_rotation_equivariance():
    # rotating the crystal frame and the geometry together must rotate
    # every output vector identically
    rng = np.random.default_rng(11)
    ang = 0.7
    R = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                  [np.sin(ang), np.cos(ang), 0.0],
                  [0.0, 0.0, 1.0]])
    d, nh = _incident_fan(n=15)
    base = bi.refract_in_biaxial(d, nh, I3, 1.0, KTP_EPS)
    # frame rows = principal axes in global coords; rotating the crystal
    # by R makes the axes R@e_i, i.e. frame = R.T? No: rows are axes ->
    # frame' = (R @ I3.T).T = R.T ... verify by equivariance instead:
    rot = bi.refract_in_biaxial(
        (d @ R.T), (nh @ R.T), I3 @ R.T, 1.0, KTP_EPS)
    for m in ("slow", "fast"):
        assert np.max(np.abs(rot["k_%s" % m] - base["k_%s" % m] @ R.T)) \
            < 1e-9
        assert np.max(np.abs(rot["s_%s" % m] - base["s_%s" % m] @ R.T)) \
            < 1e-9
        assert np.max(np.abs(rot["n_phase_%s" % m]
                             - base["n_phase_%s" % m])) < 1e-9
