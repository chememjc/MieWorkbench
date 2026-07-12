# =============================================================================
# test_nlo.py — pulsed-optics Phase P7a: chi(2) registry + d_eff tensor math
# + phase-matching solver.
#
# Covers:
#   * load_nonlinear round-trip of the shipped nonlinear.mienlo (all five
#     kinds, parsed values, '#' header comments skipped).
#   * Hard-validation rejections: missing reference, unknown kind, bad d_il
#     packing, negative I_sat, T0 > 1, bad r-coeff names, unknown crystal
#     when the birefringence handles are passed (and lazy acceptance when
#     they are not).
#   * d_eff_tensor vs the closed-form 3m type-I ooe formula
#     d_eff = d31 sin(theta) - d22 cos(theta) sin(3 phi)  (1e-12 — same
#     math, so exact).
#   * KTP type-II SHG 1064 at (theta=90, phi=23.5): contraction with
#     measured d15/d24 reproduces the Eckardt & Byer d_eff = 3.2 pm/V
#     within 15%.
#   * phase_match_angle: BBO type-I at 800 nm within 0.5 deg of the 29.2
#     deg vendor cut, using the shipped bbo_o/bbo_e dispersion; registry
#     passthrough for type-II; unmatchable/positive-uniaxial rejections.
#   * sinc2 limits, delta_k explicit form, shg_efficiency scaling
#     (linear in I, quadratic in L at delta_k = 0, sinc^2 detuning null,
#     0.5 clamp), local_intensity convention.
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_nlo.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from raytracer import nlo, optprops                        # noqa: E402
from raytracer.materials import MaterialError              # noqa: E402

NLO_HEADER = ("kind,name,crystal,point_group,d_il_pm_V,kleinman,lam_ref_nm,"
              "process,lam_pump_nm,theta_deg,phi_deg,d_eff_pm_V,"
              "r_coeffs_pm_V,geometry,material,n2_m2_W,I_sat_W_cm2,T0,"
              "tau_recovery_s,reference,notes")

COLS = NLO_HEADER.split(",")


def write_registry(tmp_path, rows):
    """rows: list of {col: value} dicts -> a temp .mienlo file path."""
    lines = ["# comment line, must be skipped", NLO_HEADER]
    for r in rows:
        lines.append(",".join('"%s"' % r.get(c, "") if "," in r.get(c, "")
                              else r.get(c, "") for c in COLS))
    p = tmp_path / "nonlinear.mienlo"
    p.write_text("\n".join(lines) + "\n")
    return p


def tensor_row(**over):
    base = {"kind": "chi2_tensor", "name": "t1", "crystal": "bbo",
            "point_group": "3m",
            "d_il_pm_V": "0;0;0;0;0.08;-2.2|-2.2;2.2;0;0.08;0;0"
                         "|0.08;0.08;0;0;0;0",
            "kleinman": "true", "lam_ref_nm": "1064", "reference": "synth"}
    base.update(over)
    return base


def saturable_row(**over):
    base = {"kind": "saturable", "name": "s1", "I_sat_W_cm2": "6e7",
            "T0": "0.84", "tau_recovery_s": "2e-12", "reference": "synth"}
    base.update(over)
    return base


@pytest.fixture(scope="module")
def registry():
    return optprops.load_nonlinear()


@pytest.fixture(scope="module")
def props():
    return optprops.load_optical_properties()


# ---------------------------------------------------------------------------
# loader round-trip (shipped registry)
# ---------------------------------------------------------------------------
def test_shipped_registry_loads(registry):
    assert set(registry) == {
        "linbo3_d", "bbo_d", "ktp_d", "lbo_d", "kdp_d",
        "ktp_shg_1064_type2", "bbo_shg_800_type1",
        "kdp_star_q_switch", "linbo3_eo",
        "n2_fused_silica", "n2_sapphire", "n2_yag", "n2_bk7",
        "sam_1550_16_2ps"}
    kinds = {r["kind"] for r in registry.values()}
    assert kinds == set(optprops.NLO_KINDS)
    for row in registry.values():
        assert row["reference"]


def test_tensor_rows_parse(registry):
    ktp = registry["ktp_d"]
    assert ktp["kind"] == "chi2_tensor" and ktp["crystal"] == "ktp"
    d = ktp["d_il_pm_V"]
    assert d.shape == (3, 6) and d.dtype == np.float64
    assert d[0, 4] == 1.9 and d[1, 3] == 3.5          # measured d15, d24
    assert d[2, 0] == 2.2 and d[2, 1] == 3.9 and d[2, 2] == 15.3
    assert ktp["kleinman"] is False                    # KTP violates Kleinman
    assert ktp["lam_ref_nm"] == 1064.0
    lnb = registry["linbo3_d"]
    assert lnb["kleinman"] is True and lnb["d_il_pm_V"][2, 2] == 25.2
    kdp = registry["kdp_d"]
    assert kdp["point_group"] == "-42m"
    assert kdp["d_il_pm_V"][2, 5] == 0.38 and kdp["d_il_pm_V"][0, 3] == 0.38


def test_process_rows_parse(registry):
    ktp = registry["ktp_shg_1064_type2"]
    assert (ktp["process"], ktp["lam_pump_nm"]) == ("shg_type2", 1064.0)
    assert (ktp["theta_deg"], ktp["phi_deg"]) == (90.0, 23.5)
    assert ktp["d_eff_pm_V"] == 3.2
    bbo = registry["bbo_shg_800_type1"]
    assert (bbo["process"], bbo["theta_deg"]) == ("shg_type1", 29.2)
    assert bbo["d_eff_pm_V"] == 1.95


def test_pockels_rows_parse(registry):
    kdp = registry["kdp_star_q_switch"]
    assert kdp["geometry"] == "longitudinal"
    assert kdp["r_pm_V"] == {"r63": 26.4}
    lnb = registry["linbo3_eo"]
    assert lnb["geometry"] == "transverse"
    assert lnb["r_pm_V"] == {"r33": 30.8, "r13": 8.6}


def test_n2_rows_parse(registry):
    fs = registry["n2_fused_silica"]
    assert fs["material"] == "fused_silica"
    assert fs["n2_m2_W"] == 2.7e-20 and fs["lam_ref_nm"] == 1053.0
    assert registry["n2_sapphire"]["n2_m2_W"] == 2.9e-20
    assert registry["n2_yag"]["n2_m2_W"] == 6.0e-20
    assert registry["n2_bk7"]["n2_m2_W"] == 3.4e-20
    # n2_yag's material row is staged: the field is lazy BY DESIGN
    assert registry["n2_yag"]["material"] == "yag"


def test_saturable_row_parses(registry):
    sam = registry["sam_1550_16_2ps"]
    assert sam["I_sat_W_cm2"] == 6.0e7
    assert sam["T0"] == 0.84
    assert sam["tau_recovery_s"] == 2.0e-12


def test_optical_properties_slot(props, registry):
    # load_optical_properties wires the registry in (with crystal
    # validation against the live uniaxial+biaxial registries)
    assert set(props.nonlinear) == set(registry)


# ---------------------------------------------------------------------------
# loader hard-validation
# ---------------------------------------------------------------------------
def test_missing_reference_rejected(tmp_path):
    p = write_registry(tmp_path, [tensor_row(reference="")])
    with pytest.raises(MaterialError, match="reference is required"):
        optprops.load_nonlinear(csv_path=p)


def test_unknown_kind_rejected(tmp_path):
    p = write_registry(tmp_path, [tensor_row(kind="chi3_tensor")])
    with pytest.raises(MaterialError, match="kind"):
        optprops.load_nonlinear(csv_path=p)


def test_bad_d_il_packing_rejected(tmp_path):
    # five entries in a row
    p = write_registry(tmp_path, [tensor_row(
        d_il_pm_V="0;0;0;0;0.08|-2.2;2.2;0;0.08;0;0|0.08;0.08;0;0;0;0")])
    with pytest.raises(MaterialError, match="d_il_pm_V"):
        optprops.load_nonlinear(csv_path=p)
    # two rows instead of three
    p = write_registry(tmp_path, [tensor_row(
        d_il_pm_V="0;0;0;0;0.08;-2.2|-2.2;2.2;0;0.08;0;0")])
    with pytest.raises(MaterialError, match=r"3 '\|'-separated rows"):
        optprops.load_nonlinear(csv_path=p)
    # non-numeric entry
    p = write_registry(tmp_path, [tensor_row(
        d_il_pm_V="0;0;0;0;d15;-2.2|-2.2;2.2;0;0.08;0;0|0.08;0.08;0;0;0;0")])
    with pytest.raises(MaterialError, match="not numeric"):
        optprops.load_nonlinear(csv_path=p)


def test_bad_kleinman_rejected(tmp_path):
    p = write_registry(tmp_path, [tensor_row(kleinman="yes")])
    with pytest.raises(MaterialError, match="kleinman"):
        optprops.load_nonlinear(csv_path=p)


def test_negative_I_sat_rejected(tmp_path):
    p = write_registry(tmp_path, [saturable_row(I_sat_W_cm2="-6e7")])
    with pytest.raises(MaterialError, match="I_sat_W_cm2"):
        optprops.load_nonlinear(csv_path=p)


def test_T0_over_unity_rejected(tmp_path):
    p = write_registry(tmp_path, [saturable_row(T0="84")])
    with pytest.raises(MaterialError, match="T0"):
        optprops.load_nonlinear(csv_path=p)


def test_bad_r_coeff_name_rejected(tmp_path):
    p = write_registry(tmp_path, [
        {"kind": "pockels", "name": "p1", "crystal": "kdp",
         "r_coeffs_pm_V": "d63=26.4", "geometry": "longitudinal",
         "reference": "synth"}])
    with pytest.raises(MaterialError, match="r_coeffs_pm_V"):
        optprops.load_nonlinear(csv_path=p)


def test_crystal_validated_against_handles(tmp_path):
    p = write_registry(tmp_path, [tensor_row(crystal="unobtainium")])
    # lazy without handles (standalone load)
    reg = optprops.load_nonlinear(csv_path=p)
    assert reg["t1"]["crystal"] == "unobtainium"
    # hard error once the birefringence registries are in hand
    with pytest.raises(MaterialError, match="unobtainium"):
        optprops.load_nonlinear(csv_path=p, uniaxial={"bbo": {}},
                                biaxial={"ktp": {}})
    # and a known crystal passes the same gate
    p2 = write_registry(tmp_path, [tensor_row()])
    reg2 = optprops.load_nonlinear(csv_path=p2, uniaxial={"bbo": {}},
                                   biaxial={})
    assert reg2["t1"]["crystal"] == "bbo"


def test_duplicate_name_rejected(tmp_path):
    p = write_registry(tmp_path, [tensor_row(), tensor_row()])
    with pytest.raises(MaterialError, match="duplicate name"):
        optprops.load_nonlinear(csv_path=p)


# ---------------------------------------------------------------------------
# d_eff tensor contraction
# ---------------------------------------------------------------------------
def _ooe_vectors(theta, phi):
    """Standard uniaxial field frame (c || z): k at (theta, phi),
    o = (sin phi, -cos phi, 0), e = (-cos t cos p, -cos t sin p, sin t)."""
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    k = np.array([st * cp, st * sp, ct])
    o = np.array([sp, -cp, 0.0])
    e = np.array([-ct * cp, -ct * sp, st])
    return k, o, e


def test_3m_closed_form_identity(registry):
    """The general contraction IS the 3m type-I ooe closed form
    d_eff = d31 sin(theta) - d22 cos(theta) sin(3 phi) — assert exactly."""
    d = registry["bbo_d"]["d_il_pm_V"]
    d31, d22 = d[2, 0], -d[0, 5]
    for theta_deg in (10.0, 22.8, 29.2, 45.0, 70.0):
        for phi_deg in (0.0, 30.0, 90.0, 123.0, 251.0):
            th, ph = np.radians(theta_deg), np.radians(phi_deg)
            k, o, e = _ooe_vectors(th, ph)
            got = nlo.d_eff_tensor(d, "3m", k, o, o, e)
            want = d31 * np.sin(th) - d22 * np.cos(th) * np.sin(3 * ph)
            assert abs(got - want) < 1e-12


def test_bbo_process_row_consistent(registry):
    """The shipped bbo_shg_800_type1 d_eff matches the contraction at its
    own cut angles (theta=29.2, phi=90)."""
    d = registry["bbo_d"]["d_il_pm_V"]
    row = registry["bbo_shg_800_type1"]
    th = np.radians(row["theta_deg"])
    ph = np.radians(row["phi_deg"])
    k, o, e = _ooe_vectors(th, ph)
    got = abs(nlo.d_eff_tensor(d, "3m", k, o, o, e))
    assert abs(got - row["d_eff_pm_V"]) / row["d_eff_pm_V"] < 0.02


def test_ktp_type2_d_eff(registry):
    """KTP type-II SHG 1064 at theta=90, phi=23.5 (XY principal plane):
    pump photons split between the z-polarized slow mode and the in-plane
    fast mode, harmonic in-plane. Contraction with the measured d15/d24
    must land within 15% of the Eckardt & Byer 3.2 pm/V (field-vector vs
    walk-off subtleties allowed)."""
    d = registry["ktp_d"]["d_il_pm_V"]
    row = registry["ktp_shg_1064_type2"]
    th = np.radians(row["theta_deg"])
    ph = np.radians(row["phi_deg"])
    st = np.sin(th)
    k = np.array([st * np.cos(ph), st * np.sin(ph), np.cos(th)])
    e_z = np.array([0.0, 0.0, 1.0])
    e_ip = np.array([-np.sin(ph), np.cos(ph), 0.0])
    got = abs(nlo.d_eff_tensor(d, "mm2", k, e_z, e_ip, e_ip))
    assert abs(got - 3.2) / 3.2 < 0.15
    # and the same contraction equals the mm2 XY-plane closed form
    # d_eff = d15 sin^2(phi) + d24 cos^2(phi) exactly
    want = d[0, 4] * np.sin(ph) ** 2 + d[1, 3] * np.cos(ph) ** 2
    assert abs(got - want) < 1e-12


def test_d_eff_tensor_input_validation(registry):
    d = registry["bbo_d"]["d_il_pm_V"]
    k, o, e = _ooe_vectors(0.5, 0.3)
    with pytest.raises(MaterialError, match=r"\(3, 6\)"):
        nlo.d_eff_tensor(d[:, :5], "3m", k, o, o, e)
    with pytest.raises(MaterialError, match="zero vector"):
        nlo.d_eff_tensor(d, "3m", k, np.zeros(3), o, e)


# ---------------------------------------------------------------------------
# phase matching
# ---------------------------------------------------------------------------
def test_bbo_type1_phase_match_800(props):
    """Oracle: BBO type-I SHG at 800 nm phase-matches at theta = 29.2 deg
    (EKSMA/dmphotonics stock cuts) with the library's bbo_o/bbo_e
    dispersion."""
    res = nlo.phase_match_angle("bbo", props, 800e-9, "shg_type1")
    assert res["source"] == "solved" and res["phi_deg"] is None
    assert abs(res["theta_deg"] - 29.2) < 0.5
    assert abs(res["residual_dn"]) < 1e-12
    # the solved angle really equates the indices
    entry = props.uniaxial["bbo"]
    from raytracer.birefringence import n_e_theta
    n_o1 = float(np.real(entry["o"].n_complex(800e-9)))
    n_e2 = float(n_e_theta(np.cos(np.radians(res["theta_deg"])),
                           float(np.real(entry["o"].n_complex(400e-9))),
                           float(np.real(entry["e"].n_complex(400e-9)))))
    assert abs(n_o1 - n_e2) < 1e-10


def test_bbo_type1_phase_match_1064(props):
    """Secondary oracle: 1064 -> 532 phase-matches near the classic
    22.8 deg (Eckardt 1990 / EKSMA re-derivation 22.85)."""
    res = nlo.phase_match_angle("bbo", props, 1064e-9, "shg_type1")
    assert abs(res["theta_deg"] - 22.8) < 0.5


def test_phase_match_accepts_entry_and_matdb(props):
    a = nlo.phase_match_angle(props.uniaxial["bbo"], None, 800e-9,
                              "shg_type1")
    b = nlo.phase_match_angle("bbo", props.matdb, 800e-9, "shg_type1")
    c = nlo.phase_match_angle("bbo", props.uniaxial, 800e-9, "shg_type1")
    assert a["theta_deg"] == b["theta_deg"] == c["theta_deg"]


def test_positive_uniaxial_rejected(props):
    with pytest.raises(MaterialError, match="NEGATIVE uniaxial"):
        nlo.phase_match_angle("quartz", props, 800e-9, "shg_type1")


def test_type2_registry_passthrough(registry):
    row = registry["ktp_shg_1064_type2"]
    res = nlo.phase_match_angle(row, None, 1064e-9, "shg_type2")
    assert res == {"theta_deg": 90.0, "phi_deg": 23.5,
                   "source": "registry", "residual_dn": None}
    with pytest.raises(MaterialError, match="type-II"):
        nlo.phase_match_angle("ktp", None, 1064e-9, "shg_type2")


def test_unknown_crystal_rejected(props):
    with pytest.raises(MaterialError, match="unobtainium"):
        nlo.phase_match_angle("unobtainium", props, 800e-9, "shg_type1")


# ---------------------------------------------------------------------------
# sinc2 / delta_k / shg_efficiency / local_intensity
# ---------------------------------------------------------------------------
def test_sinc2_limits():
    assert nlo.sinc2(0.0) == 1.0
    assert nlo.sinc2(np.pi) < 1e-30
    assert abs(nlo.sinc2(1e-8) - 1.0) < 1e-12
    assert nlo.sinc2(0.7) == nlo.sinc2(-0.7)          # even
    arr = nlo.sinc2(np.array([0.0, np.pi / 2]))
    assert arr.shape == (2,) and arr[0] == 1.0
    assert abs(arr[1] - (2 / np.pi) ** 2) < 1e-15


def test_delta_k_explicit_form():
    assert nlo.delta_k(1.5, 1.5, 1e-6) == 0.0
    # k(2w) - 2 k(w) = (4 pi / lam1)(n2 - n1)
    lam = 1.064e-6
    got = nlo.delta_k(1.5, 1.6, lam)
    assert abs(got - 4 * np.pi * 0.1 / lam) / abs(got) < 1e-14
    assert got > 0                                     # n2 > n1


def test_eta_scales_linearly_with_intensity():
    args = dict(d_eff=3.2e-12, L_m=5e-3, n1=1.78, n2=1.79, lam1_m=1.064e-6,
                delta_k=0.0)
    e1, c1 = nlo.shg_efficiency(I_W_m2=1e10, **args)
    e2, c2 = nlo.shg_efficiency(I_W_m2=2e10, **args)
    assert not c1 and not c2
    assert abs(e2 / e1 - 2.0) < 1e-12


def test_eta_scales_quadratically_with_length():
    args = dict(d_eff=3.2e-12, I_W_m2=1e10, n1=1.78, n2=1.79,
                lam1_m=1.064e-6, delta_k=0.0)
    e1, _ = nlo.shg_efficiency(L_m=1e-3, **args)
    e4, _ = nlo.shg_efficiency(L_m=2e-3, **args)
    assert abs(e4 / e1 - 4.0) < 1e-12


def test_eta_detuning_null():
    """At delta_k L / 2 = pi the sinc^2 kills the conversion."""
    L = 5e-3
    dk = 2 * np.pi / L
    e0, _ = nlo.shg_efficiency(3.2e-12, L, 1e10, 1.78, 1.79, 1.064e-6, 0.0)
    ed, _ = nlo.shg_efficiency(3.2e-12, L, 1e10, 1.78, 1.79, 1.064e-6, dk)
    assert ed < 1e-25 * e0 + 1e-40
    # half-detuning: sinc^2(pi/2) = (2/pi)^2
    eh, _ = nlo.shg_efficiency(3.2e-12, L, 1e10, 1.78, 1.79, 1.064e-6,
                               dk / 2)
    assert abs(eh / e0 - (2 / np.pi) ** 2) < 1e-12


def test_eta_dimensional_ballpark():
    """KTP, 5 mm, 100 MW/cm^2, perfect matching: the undepleted formula
    predicts >100% (clamped) — and at 10 MW/cm^2 lands near 12%, the
    textbook order of magnitude for a single-pass ns doubler."""
    eta, clamped = nlo.shg_efficiency(3.2e-12, 5e-3, 1e12, 1.78, 1.79,
                                      1.064e-6, 0.0)
    assert clamped and eta == nlo.ETA_CLAMP == 0.5
    eta, clamped = nlo.shg_efficiency(3.2e-12, 5e-3, 1e11, 1.78, 1.79,
                                      1.064e-6, 0.0)
    assert not clamped and 0.05 < eta < 0.3


def test_local_intensity():
    assert abs(nlo.local_intensity(1e-3, 1e-6, 1.0) - 1e3) < 1e-9    # W/m^2
    # mode-locked peak enhancement rides through linearly
    assert abs(nlo.local_intensity(1e-3, 1e-6, 1.25e5) / 1.25e8 - 1) < 1e-12
    with pytest.raises(MaterialError, match="dA_m2"):
        nlo.local_intensity(1e-3, 0.0, 1.0)
