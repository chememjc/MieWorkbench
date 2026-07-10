# =============================================================================
# test_scatter.py — measured-scatter (ABg/BSDF) import: registry loader,
# the sample_abg sampler, and the reflected-side specular/scatter split in
# the tracer (lowhanging.md #10).
#
# Gates (validation scope: the scatter path + what it touches):
#   a  loader: reference required, TIS>1 rejected, unknown name -> scene error
#   b  sampler: sampled |beta| KS-matches the ABg radial density (g==2 closed
#      form AND g!=2 numeric inverse CDF); TIS quadrature matches abg_tis
#   c  single interface: specular + scattered children == incident * R (mirror)
#   d  scene energy closure < 1e-3 with scatter on
#   e  scatter + roughness on one face is a contract error at Scene build
# =============================================================================
import numpy as np
import pytest
from scipy import integrate, stats

import common
from raytracer.materials import MaterialError
from raytracer.optprops import load_optical_properties, load_scatter
from raytracer.scene import Scene
from raytracer import scatter as scatter_mod

from . import scenehelpers as sh


@pytest.fixture(scope="module")
def props():
    return load_optical_properties()


# ---------------------------------------------------------------------------
# (a) loader validation
# ---------------------------------------------------------------------------
_HEADER = "name,model,A,B,g,tis_cap,reference,notes\n"


def _write_reg(tmp_path, body):
    p = tmp_path / "bsdf.miebsdf"
    p.write_text(_HEADER + body)
    return p


def test_registry_ships_cited_entries(props):
    assert {"polished_fused_silica", "polished_bk7_glass",
            "diamond_turned_aluminum"} <= set(props.scatter)
    for name, e in props.scatter.items():
        assert e["reference"]                     # citation mandatory
        assert e["A"] > 0 and e["B"] > 0 and e["g"] > 0
        tis = scatter_mod.abg_tis(e["A"], e["B"], e["g"], 1.0)
        assert 0.0 < tis <= 1.0
    assert props.scatter["diamond_turned_aluminum"]["tis_cap"] == \
        pytest.approx(0.1)


def test_loader_requires_reference(tmp_path):
    p = _write_reg(tmp_path, "s1,abg,1e-3,1e-3,2,,,note\n")
    with pytest.raises(MaterialError, match="reference is required"):
        load_scatter(csv_path=p)


def test_loader_rejects_tis_over_one(tmp_path):
    # A large enough that INT A/(B+u^2)*2piu du over the disk exceeds 1
    p = _write_reg(tmp_path, "hot,abg,10,1e-3,2,,ref here,note\n")
    with pytest.raises(MaterialError, match="exceeds 1"):
        load_scatter(csv_path=p)


def test_loader_rejects_nonpositive_params(tmp_path):
    for bad in ("s,abg,0,1e-3,2,,ref,n\n", "s,abg,1e-3,0,2,,ref,n\n",
                "s,abg,1e-3,1e-3,0,,ref,n\n"):
        with pytest.raises(MaterialError):
            load_scatter(csv_path=_write_reg(tmp_path, bad))


def test_loader_rejects_bad_tis_cap(tmp_path):
    p = _write_reg(tmp_path, "s,abg,1e-3,1e-3,2,1.5,ref,n\n")
    with pytest.raises(MaterialError, match="tis_cap"):
        load_scatter(csv_path=p)


def test_unknown_scatter_name_errors_at_scene_build(props):
    slab = sh.slab_body("S", "bk7", 0.0, 0.002, half=0.05)
    slab["scatter_faces"] = {"S.Pad.Face2": "no_such_entry"}
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False),
        slab, sh.detector_body(x=0.1, half=0.05)])
    common.validate_model(model)
    with pytest.raises(ValueError, match="unknown scatter entry"):
        Scene(model, props.matdb, props.coatings, optprops=props)


# ---------------------------------------------------------------------------
# (b) sampler
# ---------------------------------------------------------------------------
def _radial_cdf_func(A, B, g, umax=1.0):
    """Analytic radial CDF of p(u) ~ A/(B+u^g)*2*pi*u on [0, umax], as a
    grid-interpolated callable (independent of the sampler's inversion)."""
    grid = np.linspace(0.0, umax, 4001)
    dens = (A / (B + grid ** g)) * (2.0 * np.pi * grid)
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    cum /= cum[-1]
    return lambda u: np.interp(u, grid, cum)


@pytest.mark.parametrize("A,B,g,n", [(1.0, 0.01, 2.0, 20000),
                                     (1.0, 0.05, 2.5, 8000)])
def test_sampled_beta_ks_matches_radial_density(A, B, g, n):
    """At normal incidence beta0=0, so |beta| == u; the sampled radial
    distribution must match the analytic ABg radial CDF (KS)."""
    rng = np.random.default_rng(7)
    n_hat = np.tile([0.0, 0.0, 1.0], (n, 1))
    d_spec = n_hat.copy()                          # specular = normal
    dirs = scatter_mod.sample_abg(rng, n, A, B, g, d_spec, n_hat)
    # all above horizon, unit length
    assert np.all(dirs[:, 2] >= -1e-12)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-9)
    beta = np.linalg.norm(dirs[:, :2], axis=1)     # == u at normal incidence
    D, _ = stats.kstest(beta, _radial_cdf_func(A, B, g))
    assert D < 0.02, "KS statistic %.4g too large" % D


@pytest.mark.parametrize("A,B,g", [(1e-3, 1e-3, 2.0), (2e-3, 5e-2, 2.5),
                                   (5e-4, 1e-4, 1.8)])
@pytest.mark.parametrize("cos_i", [1.0, 0.8, 0.5])
def test_abg_tis_matches_independent_quadrature(A, B, g, cos_i):
    beta0 = np.sqrt(1.0 - cos_i ** 2)
    umax = 1.0 - beta0
    ref, _ = integrate.quad(
        lambda u: (A / (B + u ** g)) * 2.0 * np.pi * u, 0.0, umax,
        epsabs=1e-12, epsrel=1e-12)
    got = scatter_mod.abg_tis(A, B, g, cos_i)
    assert got == pytest.approx(ref, rel=1e-6, abs=1e-12)


def test_abg_tis_vectorized_matches_scalar():
    cos = np.array([1.0, 0.9, 0.6, 0.3])
    vec = scatter_mod.abg_tis(1e-3, 1e-3, 2.0, cos)
    for i, c in enumerate(cos):
        assert vec[i] == pytest.approx(scatter_mod.abg_tis(1e-3, 1e-3, 2.0, c))


# ---------------------------------------------------------------------------
# (c) single-interface energy split
# ---------------------------------------------------------------------------
def test_single_interface_split_conserves_reflected_power(props):
    """Perfect mirror (R=1) with scatter on its front face: specular +
    scattered children carry the full incident power (== incident * R), so
    no power lands in absorbed_surface and all reflected light escapes."""
    slab = sh.slab_body("Mir", "bk7", 0.0, 0.002, half=0.05,
                        mirror=1.0, absorbance=0.0)
    slab["scatter_faces"] = {"Mir.Pad.Face1": "diamond_turned_aluminum"}
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False, half=0.001),
        slab,
        sh.detector_body(x=0.1, half=0.05)])
    result, _, _ = sh.trace_scene(model, rays=20000, optprops=props)
    rep = result.ledger.report(["Src"])
    src = rep["sources"]["Src"]
    emitted = src["emitted_W"]
    assert rep["closure_ok"], rep
    # reflected side loses nothing at the interface (exact split)
    assert src["absorbed_surface"] < 1e-6 * emitted
    assert src["seam_loss"] < 1e-9 * emitted
    # all reflected power (R=1) leaves backward and escapes
    assert src["escaped"] == pytest.approx(emitted, rel=1e-6)


def test_higher_scatter_entry_broadens_lobe(props):
    """A rougher (larger-B) ABg surface produces a wider scatter lobe: the
    mean off-specular angle of the sampled directions grows with B."""
    rng = np.random.default_rng(11)
    n = 40000
    n_hat = np.tile([0.0, 0.0, 1.0], (n, 1))
    d_spec = n_hat.copy()

    def mean_offset(entry):
        e = props.scatter[entry]
        dirs = scatter_mod.sample_abg(rng, n, e["A"], e["B"], e["g"],
                                      d_spec, n_hat)
        return float(np.mean(np.linalg.norm(dirs[:, :2], axis=1)))

    wide = mean_offset("diamond_turned_aluminum")     # B = 1e-2
    narrow = mean_offset("polished_fused_silica")     # B = 1e-4
    assert wide > narrow > 0.0


# ---------------------------------------------------------------------------
# (d) scene closure with a transmissive optic
# ---------------------------------------------------------------------------
def test_scene_closure_with_scatter_on_dielectric(props):
    slab = sh.slab_body("Win", "bk7", 0.0, 0.002, half=0.05)
    slab["scatter_faces"] = {"Win.Pad.Face2": "polished_bk7_glass"}
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False, half=0.001),
        slab, sh.detector_body(x=0.1, half=0.05)])
    result, grids, _ = sh.trace_scene(model, rays=30000, optprops=props)
    rep = result.ledger.report(["Src"])
    assert rep["closure_ok"], rep
    assert rep["sources"]["Src"]["closure_error"] < 1e-3


def test_no_scatter_builds_no_entries(props):
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False),
        sh.slab_body("Win", "bk7", 0.0, 0.002, half=0.05),
        sh.detector_body(x=0.1, half=0.05)])
    scene = Scene(model, props.matdb, props.coatings, optprops=props)
    assert scene.scatter == {}


# ---------------------------------------------------------------------------
# (e) conflict rule
# ---------------------------------------------------------------------------
def test_scatter_plus_roughness_same_face_errors(props):
    slab = sh.slab_body("S", "bk7", 0.0, 0.002, half=0.05)
    slab["scatter_faces"] = {"S.Pad.Face2": "polished_bk7_glass"}
    slab["roughness_faces"] = {"S.Pad.Face2": "50"}
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False),
        slab, sh.detector_body(x=0.1, half=0.05)])
    with pytest.raises(ValueError, match="alternative models"):
        Scene(model, props.matdb, props.coatings, optprops=props)


def test_scatter_plus_diffuser_same_face_errors(props):
    slab = sh.slab_body("S", "bk7", 0.0, 0.002, half=0.05)
    slab["scatter_faces"] = {"S.Pad.Face2": "polished_bk7_glass"}
    slab["diffuser_faces"] = {"S.Pad.Face2": "grit:600"}
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False),
        slab, sh.detector_body(x=0.1, half=0.05)])
    with pytest.raises(ValueError, match="alternative models"):
        Scene(model, props.matdb, props.coatings, optprops=props)
