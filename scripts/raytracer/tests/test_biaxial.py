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


# ===========================================================================
# CONICAL REFRACTION: optic axes, cone angle, per-ray proximity, and the
# perturbed two-sheet fan (birefringence.biaxial_optic_axes / cone_half_angle
# / axis_proximity / conical_fan).
# ===========================================================================
# all four registry crystals @ 1064 nm (crystal principal frame == global)
CRYSTALS = {
    "ktp": np.array([1.7377, 1.7453, 1.8297]),
    "kta": np.array([1.782, 1.787, 1.868]),
    "lbo": np.array([1.566, 1.591, 1.606]),
    "bibo": np.array([1.757, 1.784, 1.917]),
}


def _closed_cone_angle(n):
    n1, n2, n3 = n
    return np.arctan(np.sqrt((n2 ** 2 - n1 ** 2)
                             * (n3 ** 2 - n2 ** 2)) / (n1 * n3))


# ---------------------------------------------------------------------------
# optic axes
# ---------------------------------------------------------------------------
def test_optic_axes_are_degeneracies_all_crystals():
    # at both optic axes the two sheets of biaxial_modes_for_k must coincide
    for name, n in CRYSTALS.items():
        eps = n ** 2
        axes = bi.biaxial_optic_axes(eps)
        assert axes.shape == (2, 3)
        m = bi.biaxial_modes_for_k(axes, I3, eps)
        rel = np.abs(m["n_slow"] - m["n_fast"]) / m["n_slow"]
        assert np.max(rel) < 1e-10, (name, rel)


def test_optic_axes_symmetric_in_principal_plane():
    # nx<ny<nz for every registry crystal -> axes lie in the x-z plane
    # (zero y-component) and are mirror images about z
    for name, n in CRYSTALS.items():
        axes = bi.biaxial_optic_axes(n ** 2)
        assert np.max(np.abs(axes[:, 1])) < 1e-14, name       # in x-z plane
        assert abs(axes[0, 2] - axes[1, 2]) < 1e-14, name     # same z
        assert abs(axes[0, 0] + axes[1, 0]) < 1e-14, name     # opposite x
        assert np.max(np.abs(np.linalg.norm(axes, axis=1) - 1.0)) < 1e-14
        # tilt angle from z matches the closed form used by the KTP oracle
        inv = 1.0 / n ** 2
        th = np.arctan2(np.sqrt(inv[0] - inv[1]), np.sqrt(inv[1] - inv[2]))
        assert abs(np.arctan2(abs(axes[0, 0]), axes[0, 2]) - th) < 1e-12


def test_optic_axes_uniaxial_limits_degenerate():
    # two smallest eps equal (positive uniaxial n_o<n_e, axis = z) -> both
    # optic axes collapse onto z
    axes = bi.biaxial_optic_axes(np.array([QTZ_NO, QTZ_NO, QTZ_NE]) ** 2)
    assert np.max(np.abs(axes - np.array([0.0, 0.0, 1.0]))) < 1e-12
    # two largest eps equal (negative uniaxial n_e<n_o, axis = z) -> both
    # optic axes collapse onto the small-index axis (x here)
    axes = bi.biaxial_optic_axes(np.array([CAL_NE, CAL_NO, CAL_NO]) ** 2)
    assert np.max(np.abs(np.abs(axes) - np.array([1.0, 0.0, 0.0]))) < 1e-12
    assert np.max(np.abs(axes[0] - axes[1])) < 1e-12    # SAME vector twice


# ---------------------------------------------------------------------------
# cone half-angle: closed form vs numerical limit
# ---------------------------------------------------------------------------
def _numeric_cone_angle(eps, delta):
    """Max angle between the four s_ray directions reached by perturbing the
    first optic axis by +-delta WITHIN the plane of the axes (rotation about
    the mid-index principal axis), both sheets each side."""
    axis = bi.biaxial_optic_axes(eps)[0]
    # the plane of the axes is x-z (y is the mid-index principal axis); a
    # rotation about y keeps the perturbed normal in that plane
    c, s = np.cos(delta), np.sin(delta)
    Rp = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    ss = []
    for R in (Rp, Rp.T):
        k = _unit((R @ axis)[None, :])
        mm = bi.biaxial_modes_for_k(k, I3, eps)
        for sh in ("slow", "fast"):
            K = mm["n_%s" % sh][:, None] * k
            sray, _, _ = bi.biaxial_ray_from_k(K, I3, eps)
            ss.append(sray[0])
    ss = np.array(ss)
    G = np.clip(ss @ ss.T, -1.0, 1.0)
    return np.arccos(G).max()


def test_cone_angle_closed_form_vs_numeric_limit():
    # Romberg (two-level Richardson) of the finite-perturbation limit cancels
    # the leading O(delta) and O(delta^2) errors of the tangent-cone approach;
    # deterministic, no RNG.
    for name, n in CRYSTALS.items():
        eps = n ** 2
        Ac = _closed_cone_angle(n)
        d = 2e-4
        a0 = _numeric_cone_angle(eps, d)
        a1 = _numeric_cone_angle(eps, d / 2)
        a2 = _numeric_cone_angle(eps, d / 4)
        r1a, r1b = 2 * a1 - a0, 2 * a2 - a1
        A_num = (4 * r1b - r1a) / 3.0
        assert abs(A_num - Ac) < 1e-8 * Ac, (name, A_num, Ac)
        # and cone_half_angle returns exactly that closed form
        assert abs(bi.cone_half_angle(eps) - Ac) < 1e-14, name


def test_cone_angle_ktp_magnitude_sane():
    A_deg = np.degrees(bi.cone_half_angle(CRYSTALS["ktp"] ** 2))
    assert 0.1 < A_deg < 5.0, A_deg          # order of a degree or two
    assert abs(A_deg - 1.610) < 1e-2, A_deg  # KTP @ 1064 nm


# ---------------------------------------------------------------------------
# axis proximity (dispatch criterion)
# ---------------------------------------------------------------------------
def test_axis_proximity_zero_on_axes_and_vectorized():
    eps = CRYSTALS["ktp"] ** 2
    axes = bi.biaxial_optic_axes(eps)           # global == crystal (I3)
    # exactly on the axes (and their negatives) -> ~0
    k = np.concatenate([axes, -axes])
    prox = bi.axis_proximity(k, eps, I3)
    assert prox.shape == (4,)
    assert np.max(prox) < 1e-12
    # a hand-computed off-axis angle: 3 deg off the first axis in x-z
    th = np.deg2rad(3.0)
    a = axes[0]
    b = a * np.cos(th) + np.array([a[2], 0.0, -a[0]]) * np.sin(th)  # rot in x-z
    prox1 = bi.axis_proximity(_unit(b[None, :]), eps, I3)
    assert abs(prox1[0] - th) < 1e-9
    # rotated crystal frame: rotate frame and wave normals together, angle
    # to the nearest axis is frame-invariant
    ang = 0.6
    R = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                  [np.sin(ang), np.cos(ang), 0.0],
                  [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(5)
    kk = _unit(rng.normal(size=(30, 3)))
    base = bi.axis_proximity(kk, eps, I3)
    rot = bi.axis_proximity(kk @ R.T, eps, I3 @ R.T)
    assert np.max(np.abs(base - rot)) < 1e-12
    assert base.shape == (30,)


# ---------------------------------------------------------------------------
# conical fan geometry
# ---------------------------------------------------------------------------
def test_conical_fan_on_ktp_axis_geometry():
    eps = CRYSTALS["ktp"] ** 2
    A = bi.cone_half_angle(eps)
    axis = bi.biaxial_optic_axes(eps)[0]
    n_fan, ea = 24, 1e-3
    fan = bi.conical_fan(axis, I3, eps, n_fan, ea)
    for key in ("k", "s_ray", "D_hat"):
        assert fan[key].shape == (2 * n_fan, 3)
    for key in ("n_phase", "n_ray", "sheet", "azimuth"):
        assert fan[key].shape == (2 * n_fan,)
    s = fan["s_ray"]
    # (a) all rays lie on the Hamilton cone: the largest angle between any
    # two generators equals the opening angle A within 2*eps_angle
    G = np.clip(s @ s.T, -1.0, 1.0)
    max_pair = np.arccos(G).max()
    assert abs(max_pair - A) < 2 * ea, (max_pair, A)
    # and the optic-axis (wave-normal) direction is itself a generator: some
    # sampled ray sits within the azimuth-sampling resolution (~eps_angle)
    min_to_axis = np.arccos(np.clip(np.abs(s @ axis).max(), -1.0, 1.0))
    assert min_to_axis < 2 * ea, min_to_axis
    # (c) transverse completeness: slow.fast D orthogonal at every azimuth
    Ds, Df = fan["D_hat"][:n_fan], fan["D_hat"][n_fan:]
    assert np.max(np.abs(np.sum(Ds * Df, axis=-1))) < 1e-10
    assert np.max(np.abs(np.sum(fan["D_hat"] * fan["k"], axis=-1))) < 1e-12


def test_conical_fan_polarization_half_turn_law():
    # D rotates by phi/2 around the ring (classic conical-refraction law):
    # fit angle(D) vs azimuth -> slope 1/2 on each sheet.
    eps = CRYSTALS["ktp"] ** 2
    axis = bi.biaxial_optic_axes(eps)[0]
    n_fan, ea = 60, 1e-3
    fan = bi.conical_fan(axis, I3, eps, n_fan, ea)
    phi = fan["azimuth"][:n_fan]
    # transverse reference frame about the wave normal (same construction
    # conical_fan uses, so azimuth 0 is well defined)
    ax = np.zeros(3)
    ax[int(np.argmin(np.abs(axis)))] = 1.0
    e1 = _unit(np.cross(axis, ax))
    e2 = np.cross(axis, e1)
    for blk in (slice(0, n_fan), slice(n_fan, 2 * n_fan)):
        D = fan["D_hat"][blk]
        psi = np.arctan2(D @ e2, D @ e1)     # defined mod pi (D ~ -D)
        slope = np.polyfit(phi, np.unwrap(2 * psi), 1)[0] / 2.0
        assert abs(slope - 0.5) < 1e-2, slope


def test_conical_fan_matches_plain_two_sheet_solve():
    # continuity: each fan child is exactly the plain solver evaluated at
    # that perturbed wave normal (no packaging / sheet-assignment drift).
    eps = CRYSTALS["ktp"] ** 2
    axis = bi.biaxial_optic_axes(eps)[0]
    n_fan, ea = 12, 2e-3
    fan = bi.conical_fan(axis, I3, eps, n_fan, ea)
    dirs = fan["k"][:n_fan]
    modes = bi.biaxial_modes_for_k(dirs, I3, eps)
    for j, sheet in enumerate(("slow", "fast")):
        blk = slice(j * n_fan, (j + 1) * n_fan)
        K = modes["n_%s" % sheet][:, None] * dirs
        s_ray, n_phase, n_ray = bi.biaxial_ray_from_k(K, I3, eps)
        assert np.max(np.abs(fan["s_ray"][blk] - s_ray)) < 1e-12
        assert np.max(np.abs(fan["n_phase"][blk] - n_phase)) < 1e-12
        assert np.max(np.abs(fan["n_ray"][blk] - n_ray)) < 1e-12
        assert np.max(np.abs(fan["D_hat"][blk] - modes["D_%s" % sheet])) \
            < 1e-12
        assert np.all(fan["sheet"][blk] == j)


# ===========================================================================
# CONICAL REFRACTION — tracer integration (--conical): end-to-end ring
# formation, closure, and the guard/fanned counters (samples-instruments
# round; the pure-math fan is covered above).
# ===========================================================================
def _on_axis_ktp_model(t=0.020, lambdac_nm=1064.0):
    """KTP slab oriented so one OPTIC AXIS lies along the global +x beam:
    principal Y = global y, principal X/Z rotated about y so that the
    crystal-frame optic axis [sin(theta), 0, cos(theta)] maps onto global
    x. Rows of the crystal frame are the principal axes in global coords
    (v_c = frame @ v_g), so crystal_axis (X_p) = [sin t, 0, -cos t] and
    Z_p = X_p x Y_p = [cos t, 0, sin t] gives frame @ x_hat = the optic
    axis exactly."""
    from raytracer import optprops
    props = optprops.load_optical_properties()
    mo = props.matdb.get_biaxial("ktp")
    lam = lambdac_nm * 1e-9
    eps = np.array([np.real(m.n_complex(lam)) ** 2 for m in mo])
    axes_c = bi.biaxial_optic_axes(eps)
    ax = axes_c[np.argmax(axes_c[:, 0])]        # the +x-leaning axis
    st, ct_ = ax[0], ax[2]
    bodies = [
        sh.source_body(x=-0.01, half=0.00005, coherent=False,
                       power_mW=1.0, lambdac_nm=lambdac_nm),
        sh.slab_body("KTP", "ktp", 0.0, t, half=0.008,
                     crystal_axis=[st, 0.0, -ct_],
                     crystal_axis2=[0.0, 1.0, 0.0]),
        sh.detector_body(x=t + 0.003, half=0.003),
    ]
    return sh.make_model(bodies), eps, t


def _radial_stats(det):
    """(mean_r, std_r) of the detected-power radial distribution about
    its own power centroid, in metres on the detector plane."""
    img = det.inc.sum(axis=0)
    tot = img.sum()
    assert tot > 0
    xs = det.x_lo + (np.arange(det.W) + 0.5) * det.pixel_m
    ys = det.y_lo + (np.arange(det.H) + 0.5) * det.pixel_m
    X, Y = np.meshgrid(xs, ys)
    cx = (img * X).sum() / tot
    cy = (img * Y).sum() / tot
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mean_r = (img * R).sum() / tot
    var_r = (img * (R - mean_r) ** 2).sum() / tot
    return float(mean_r), float(np.sqrt(var_r))


def test_conical_off_counts_guard_single_spot():
    model, eps, t = _on_axis_ktp_model()
    res, grids, scene = sh.trace_scene(model, rays=4000, resolution=300)
    rep = res.ledger.report(res.source_names)
    assert max(s["closure_error"] for s in rep["sources"].values()) < 1e-3
    # off: degenerate entries counted, nothing fanned
    assert res.conical_guard.get("KTP", 0) > 0
    assert res.conical_fanned == {}
    # arbitrary-basis two-sheet pass: a tight on-axis spot, no ring
    mean_r, _ = _radial_stats(list(grids.values())[0])
    A = bi.cone_half_angle(eps)
    assert mean_r < 0.15 * t * np.tan(A)


def test_conical_on_forms_poggendorff_ring_with_closure():
    """--conical: the on-axis beam through 20 mm of KTP fans into the
    Hamilton cone; the exit-face footprint is a circle through the axis
    of DIAMETER t*tan(A) (one cone generator IS the wave normal), which
    the slab exit freezes — so about its own centroid the detected ring
    has radius t*tan(A)/2, tight."""
    model, eps, t = _on_axis_ktp_model()
    res, grids, scene = sh.trace_scene(model, rays=4000, resolution=300,
                                       conical=True, conical_fan=24,
                                       conical_delta=1e-4)
    rep = res.ledger.report(res.source_names)
    assert max(s["closure_error"] for s in rep["sources"].values()) < 1e-3
    assert res.conical_fanned.get("KTP", 0) > 0
    assert res.conical_guard == {}
    A = bi.cone_half_angle(eps)
    r_ring = 0.5 * t * np.tan(A)
    mean_r, std_r = _radial_stats(list(grids.values())[0])
    # ring radius within 10% (beam pencil width + fan discretization),
    # and RADIALLY TIGHT (a ring, not a disc: std << radius)
    assert abs(mean_r - r_ring) < 0.10 * r_ring, \
        "ring radius %.4g mm vs expected %.4g mm" \
        % (mean_r * 1e3, r_ring * 1e3)
    assert std_r < 0.35 * r_ring


def test_conical_fan_energy_conservation_detailed():
    """Fan child powers per parent sum EXACTLY to the mean-index
    transmitted power (unit-level, via the tracer's ledger closure at
    1e-9 on a lossless path: source -> crystal entry fan -> exit ->
    detector; absorbed_surface picks up only the tiny mean-index vs
    exact-sheet Fresnel difference)."""
    model, eps, t = _on_axis_ktp_model()
    res, grids, scene = sh.trace_scene(model, rays=1500, resolution=200,
                                       conical=True, conical_fan=12)
    rep = res.ledger.report(res.source_names)
    # detected + all loss buckets == emitted to the standard gate
    assert max(s["closure_error"] for s in rep["sources"].values()) < 1e-3
    # most of the power must actually reach the detector (AR-free KTP:
    # ~8% Fresnel loss per face -> ~84% through two faces, minus the
    # multi-bounce remainder)
    det = list(grids.values())[0]
    p_det = float(sum(det.detected_incoherent.values()))
    assert p_det > 0.75e-3
