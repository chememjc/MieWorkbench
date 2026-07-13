# =============================================================================
# test_image_sim.py -- partial-coherence imaging / image simulation oracle
# (imaging-analysis round, greenfield half of F4).
#
# Two layers:
#   (1) analysis_field.py Fourier-optics oracle -- a synthetic hard-edged
#       circular pupil P (radius R frequency pixels) defines the ground
#       truth: h = IFT(P) is the amplitude PSF, the coherent ATF support
#       radius must be R while the incoherent OTF support radius (FT of
#       |h|^2 = the pupil autocorrelation) must be 2R -- THE classic
#       factor of two (Goodman, Introduction to Fourier Optics ch. 6).
#       Plus delta-object self-consistency, the image_partial sigma->0 /
#       sigma->large limits, coherent edge ringing vs the incoherent
#       overshoot bound, and real/finite/>=0 sanity on every output.
#   (2) post_process.render_image_sim end-to-end against a SYNTHETIC case
#       directory (the test_post_analysis.py pattern): a detector .h5
#       carrying a diffraction-limited fields/<key>/{Ex,Ey} pair built
#       from that same circular pupil + a bar-target input image ->
#       imaging/*.png outputs, the report.json 'image_sim' block, the
#       hard --save-fields error when no fields exist, and the coherent
#       vs incoherent physics (ringing overshoot) THROUGH the real
#       render path.
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_image_sim.py -q
# =============================================================================
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import post_process                                        # noqa: E402
from raytracer.analysis_field import (                     # noqa: E402
    coherent_transfer, image_coherent, image_incoherent, image_partial,
    mtf2d, psf_from_fields, pupil_radius_px)


# ---------------------------------------------------------------------------
# shared synthetic optics: hard-edged circular pupil -> amplitude PSF
# ---------------------------------------------------------------------------
N = 128
R_PUPIL = 12.0   # frequency pixels; 2R + sigma*R stays well inside N/2


def _disc_pupil(n=N, radius=R_PUPIL):
    ys, xs = np.indices((n, n))
    r = np.hypot(ys - n // 2, xs - n // 2)
    return (r <= radius).astype(np.complex128)


def _amp_psf_from_pupil(P):
    """Centered amplitude PSF h = fftshift(IFT2(ifftshift(P))) -- the
    exact inverse of coherent_transfer, so the ATF roundtrips to P."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(P)))


def _support_radius(mag, threshold):
    """Max pixel radius from the (H//2, W//2) DC pixel where a centered
    non-negative map exceeds threshold * its peak."""
    H, W = mag.shape
    ys, xs = np.indices((H, W))
    r = np.hypot(ys - H // 2, xs - W // 2)
    return float(r[mag > threshold * mag.max()].max())


def _bar_target(n=N, period=32, duty=0.5, lo=0.0, hi=1.0):
    """Vertical binary bar chart (intensity object), values in {lo, hi}."""
    xs = np.arange(n)
    bars = ((xs % period) < duty * period)
    obj = np.full((n, n), lo)
    obj[:, bars] = hi
    return obj


def _delta_object(n=N):
    obj = np.zeros((n, n))
    obj[n // 2, n // 2] = 1.0
    return obj


def _corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


# ---------------------------------------------------------------------------
# (1a) THE classic factor of 2: incoherent OTF cutoff = 2x coherent ATF
# ---------------------------------------------------------------------------
def test_incoherent_cutoff_is_twice_coherent_cutoff():
    P = _disc_pupil()
    h = _amp_psf_from_pupil(P)

    # coherent ATF: coherent_transfer must roundtrip to the pupil itself
    atf = coherent_transfer(h)
    assert np.allclose(atf, P, atol=1e-10)
    r_coh = _support_radius(np.abs(atf), threshold=0.5)
    assert abs(r_coh - R_PUPIL) <= 1.5

    # incoherent OTF: |FT(|h|^2)| == the pupil autocorrelation, support 2R
    # (exactly zero outside up to FFT roundoff ~1e-12 relative, so a 1e-6
    # relative threshold measures the true support)
    otf = mtf2d(np.abs(h) ** 2, pixel_m=1e-6)["mtf"]
    r_inc = _support_radius(otf, threshold=1e-6)
    assert abs(r_inc - 2.0 * R_PUPIL) <= 3.0
    assert r_inc / r_coh == pytest.approx(2.0, abs=0.2)

    # and pupil_radius_px (image_partial's own support estimator) agrees
    assert pupil_radius_px(atf) == pytest.approx(r_coh, abs=1.0)


# ---------------------------------------------------------------------------
# (1b) delta-object self-consistency
# ---------------------------------------------------------------------------
def test_delta_object_returns_the_psf():
    h = _amp_psf_from_pupil(_disc_pupil())
    psf = np.abs(h) ** 2
    obj = _delta_object()

    # incoherent: the unit-sum-normalized intensity PSF, centered on the
    # delta (which sits at the kernel-origin pixel -> no translation)
    img_inc = image_incoherent(obj, psf)
    assert np.allclose(img_inc, psf / psf.sum(), atol=1e-12)

    # coherent: sqrt(delta)=delta in amplitude -> U = h -> |h|^2 exactly
    img_coh = image_coherent(obj, h)
    assert np.allclose(img_coh, psf, atol=1e-12)


# ---------------------------------------------------------------------------
# (1c) partial-coherence limits
# ---------------------------------------------------------------------------
def test_partial_sigma_zero_equals_coherent():
    h = _amp_psf_from_pupil(_disc_pupil())
    obj = _bar_target()
    img_coh = image_coherent(obj, h)
    img_p0 = image_partial(obj, h, sigma=0.0)
    assert np.array_equal(img_p0, img_coh)
    # a sub-pixel source disc is still exactly coherent
    img_tiny = image_partial(obj, h, sigma=0.02)
    assert np.array_equal(img_tiny, img_coh)


def test_partial_large_sigma_approaches_incoherent():
    h = _amp_psf_from_pupil(_disc_pupil())
    psf = np.abs(h) ** 2
    obj = _bar_target()

    img_coh = image_coherent(obj, h)
    img_inc = image_incoherent(obj, psf)
    img_mid = image_partial(obj, h, sigma=0.5)
    img_big = image_partial(obj, h, sigma=2.0)

    c_coh = _corr(img_coh, img_inc)
    c_mid = _corr(img_mid, img_inc)
    c_big = _corr(img_big, img_inc)

    # coherent and incoherent images of a bar target genuinely differ...
    assert c_coh < 0.97
    # ...and the Abbe sum marches monotonically toward incoherent
    # (measured here: c_coh ~ 0.962, c_mid ~ 0.972, c_big ~ 0.998)
    assert c_big > c_mid > c_coh
    assert c_big > 0.995

    # a genuine multi-point source: sigma=0.5 must NOT collapse to the
    # coherent image (the sub-pixel guard didn't fire)
    assert not np.allclose(img_mid, img_coh)


# ---------------------------------------------------------------------------
# (1d) coherent edge ringing vs the incoherent overshoot bound
# ---------------------------------------------------------------------------
def test_coherent_edge_ringing_incoherent_bounded():
    h = _amp_psf_from_pupil(_disc_pupil())
    psf = np.abs(h) ** 2
    obj = _bar_target()

    img_inc = image_incoherent(obj, psf)
    img_coh = image_coherent(obj, h)

    # incoherent: unit-sum kernel on a binary [0,1] object can NEVER
    # exceed 1 -- it is MTF-limited and smooth
    assert img_inc.max() <= 1.0 + 1e-9
    # coherent: Gibbs-type edge ringing overshoots in intensity by ~19%
    # ((1.09)^2) for a sharp edge; demand a solid >5% overshoot
    assert img_coh.max() > 1.05
    # and the coherent image is 'sharper': more total variation across
    # the bar direction than the incoherent one
    tv = lambda a: np.abs(np.diff(a, axis=1)).sum()   # noqa: E731
    assert tv(img_coh) > tv(img_inc)


# ---------------------------------------------------------------------------
# (1e) energy / positivity sanity on every output
# ---------------------------------------------------------------------------
def test_outputs_real_finite_nonnegative():
    h = _amp_psf_from_pupil(_disc_pupil())
    psf = np.abs(h) ** 2
    obj = _bar_target(lo=0.1, hi=0.9)
    for img in (image_coherent(obj, h),
                image_incoherent(obj, psf),
                image_partial(obj, h, sigma=0.7)):
        assert img.dtype == np.float64
        assert np.all(np.isfinite(img))
        assert np.all(img >= 0.0)
        assert img.shape == obj.shape


def test_shape_mismatch_and_bad_args_raise():
    h = _amp_psf_from_pupil(_disc_pupil())
    with pytest.raises(ValueError):
        image_coherent(np.zeros((32, 32)), h)
    with pytest.raises(ValueError):
        image_incoherent(np.ones((N, N)), np.zeros((N, N)))   # zero-sum PSF
    with pytest.raises(ValueError):
        image_partial(np.ones((N, N)), h, sigma=-0.1)


# ---------------------------------------------------------------------------
# (2) render_image_sim end-to-end against a synthetic case directory
# ---------------------------------------------------------------------------
LABEL = "Det.Pad.Face1"


def _write_fields_h5(case_dir, label=LABEL, x_pol=True):
    """detectors/<label>.h5 with one coherent gather key whose Ex (or Ey)
    is the diffraction-limited amplitude PSF of the disc pupil -- the
    exact layout run_trace's --save-fields writes and
    post_process._iter_field_keys reads."""
    ddir = case_dir / "detectors"
    ddir.mkdir(parents=True, exist_ok=True)
    h = _amp_psf_from_pupil(_disc_pupil()).astype(np.complex128)
    zero = np.zeros_like(h)
    path = ddir / "det.h5"
    with h5py.File(path, "w") as f:
        f["fields/0_0_0/Ex"] = h if x_pol else zero
        f["fields/0_0_0/Ey"] = zero if x_pol else h
        f.attrs["label"] = label
        f.attrs["pixel_m"] = 2e-6
        f.attrs["H"] = N
        f.attrs["W"] = N
    return path


def _write_object_npy(case_dir, n=48):
    obj = _bar_target(n=n, period=12)
    path = case_dir / "target.npy"
    np.save(path, obj)
    return path


@pytest.mark.parametrize("mode", ["incoherent", "coherent", "partial"])
def test_render_image_sim_end_to_end(tmp_path, mode):
    h5path = _write_fields_h5(tmp_path)
    obj_path = _write_object_npy(tmp_path)
    report = {"detectors": {}}

    post_process.render_image_sim(tmp_path, [h5path], report,
                                  str(obj_path), coherence=mode, sigma=0.6)

    idir = tmp_path / "imaging"
    assert (idir / ("image_sim_%s.png" % mode)).exists()
    assert (idir / "image_sim_input.png").exists()

    block = report["image_sim"]
    assert block["mode"] == mode
    assert block["detector"] == LABEL
    assert block["key"] == "0_0_0"
    assert block["amp_source"] == "Ex"
    assert block["input"] == str(obj_path)
    assert block["shape"] == [N, N]
    assert block["output"] == "imaging/image_sim_%s.png" % mode
    assert np.isfinite(block["rms_contrast"]) and block["rms_contrast"] > 0
    assert ("sigma" in block) == (mode == "partial")
    if mode == "partial":
        assert block["sigma"] == pytest.approx(0.6)


def test_render_image_sim_coherent_vs_incoherent_physics(tmp_path):
    """The demo-level physics check THROUGH the real render path: the
    coherent simulation of a bar target shows edge ringing (max above
    the object's plateau) while the incoherent one is MTF-limited and
    bounded by it."""
    h5path = _write_fields_h5(tmp_path)
    obj_path = _write_object_npy(tmp_path)

    rep_c, rep_i = {"detectors": {}}, {"detectors": {}}
    post_process.render_image_sim(tmp_path, [h5path], rep_c,
                                  str(obj_path), coherence="coherent")
    post_process.render_image_sim(tmp_path, [h5path], rep_i,
                                  str(obj_path), coherence="incoherent")

    img_c = rep_c["image_sim"]["image_max"]
    img_i = rep_i["image_sim"]["image_max"]
    assert img_i <= 1.0 + 1e-9       # incoherent never overshoots a 0/1 bar
    assert img_c > 1.05              # coherent rings past the plateau
    # and the two outputs genuinely differ
    assert rep_c["image_sim"]["rms_contrast"] != pytest.approx(
        rep_i["image_sim"]["rms_contrast"], rel=1e-3)


def test_render_image_sim_ey_polarized_field(tmp_path):
    """A pure y-polarized coherent key: the amplitude PSF must come from
    Ey (full phase kept), not a zeroed Ex."""
    h5path = _write_fields_h5(tmp_path, x_pol=False)
    obj_path = _write_object_npy(tmp_path)
    report = {"detectors": {}}
    post_process.render_image_sim(tmp_path, [h5path], report,
                                  str(obj_path), coherence="coherent")
    assert report["image_sim"]["amp_source"] == "Ey"
    assert report["image_sim"]["image_max"] > 1.05   # phase kept -> ringing


def test_render_image_sim_png_input(tmp_path):
    """PNG object input path (Pillow); values round-trip as greyscale."""
    PIL = pytest.importorskip("PIL.Image")
    obj = (_bar_target(n=40, period=10) * 255).astype(np.uint8)
    png = tmp_path / "target.png"
    PIL.fromarray(obj, mode="L").save(png)
    h5path = _write_fields_h5(tmp_path)
    report = {"detectors": {}}
    post_process.render_image_sim(tmp_path, [h5path], report, str(png),
                                  coherence="incoherent")
    assert (tmp_path / "imaging" / "image_sim_incoherent.png").exists()
    assert report["image_sim"]["input"] == str(png)


def test_render_image_sim_requires_saved_fields(tmp_path):
    """No fields/ group anywhere -> the documented hard error naming
    --save-fields."""
    ddir = tmp_path / "detectors"
    ddir.mkdir(parents=True)
    path = ddir / "det.h5"
    with h5py.File(path, "w") as f:
        f["mask"] = np.ones((4, 4), dtype=bool)
        f.attrs["label"] = LABEL
        f.attrs["pixel_m"] = 1e-6
    obj_path = _write_object_npy(tmp_path)
    with pytest.raises(SystemExit, match="--save-fields"):
        post_process.render_image_sim(tmp_path, [path], {"detectors": {}},
                                      str(obj_path))


def test_render_image_sim_missing_input_errors(tmp_path):
    h5path = _write_fields_h5(tmp_path)
    with pytest.raises(SystemExit, match="image-sim"):
        post_process.render_image_sim(tmp_path, [h5path], {"detectors": {}},
                                      str(tmp_path / "nope.npy"))


def test_render_image_sim_noop_without_flag(tmp_path):
    report = {"detectors": {}}
    post_process.render_image_sim(tmp_path, [], report, None)
    assert not (tmp_path / "imaging").exists()
    assert "image_sim" not in report


# ---------------------------------------------------------------------------
# (3) pipeline-level demo: a real coherent trace (run_trace.main with
# --save-fields) followed by post_process.main --image-sim -- the whole
# flag path (cli_specs post parser -> main() wiring -> render_image_sim ->
# imaging/*.png + report.json). The strict ringing-overshoot physics is
# pinned by (1d)/(2) against the diffraction-limited pupil; here we pin
# that the FULL pipeline produces both modes and that they differ.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def coherent_traced_case(tmp_path_factory):
    from raytracer.tests import scenehelpers as sh
    import common
    import run_trace
    tmp_path = tmp_path_factory.mktemp("image_sim_case")
    src = sh.source_body(power_mW=2.0, coherent=True, half=5e-4,
                         lambdac_nm=633.0, x=-0.02,
                         polarization={"kind": "linear", "angle_deg": 0.0})
    det = sh.detector_body(x=0.02, half=0.004)
    model = sh.make_model([src, det])
    common.validate_model(model)
    mj = tmp_path / "model.json"
    mj.write_text(json.dumps(model))
    case = tmp_path / "case"
    rc = run_trace.main([
        "--model-json", str(mj), "--case-dir", str(case),
        "--rays", "20000", "--nlambda", "1", "--resolution", "64",
        "--power-floor", "1e-8", "--save-fields", "--no-gather-gate",
    ])
    assert rc == 0
    obj = tmp_path / "target.npy"
    np.save(obj, _bar_target(n=64, period=16))
    return mj, case, obj


def test_pipeline_image_sim_both_modes(coherent_traced_case):
    mj, case, obj = coherent_traced_case
    rc = post_process.main([
        "--case-dir", str(case), "--model-json", str(mj),
        "--image-sim", str(obj), "--image-sim-coherence", "incoherent"])
    assert rc == 0
    rep_inc = json.loads((case / "report.json").read_text())
    assert (case / "imaging" / "image_sim_incoherent.png").exists()
    assert (case / "imaging" / "image_sim_input.png").exists()
    assert rep_inc["image_sim"]["mode"] == "incoherent"
    # incoherent: unit-sum kernel on a [0,1] object stays bounded by 1
    assert rep_inc["image_sim"]["image_max"] <= 1.0 + 1e-9

    rc = post_process.main([
        "--case-dir", str(case), "--model-json", str(mj),
        "--image-sim", str(obj), "--image-sim-coherence", "coherent"])
    assert rc == 0
    rep_coh = json.loads((case / "report.json").read_text())
    assert (case / "imaging" / "image_sim_coherent.png").exists()
    assert rep_coh["image_sim"]["mode"] == "coherent"

    # the two illumination models genuinely differ through the REAL
    # traced field (amplitude vs intensity convolution)
    assert rep_coh["image_sim"]["rms_contrast"] != pytest.approx(
        rep_inc["image_sim"]["rms_contrast"], rel=1e-3)


def test_pipeline_image_sim_without_save_fields_errors(tmp_path):
    """post stage on a case traced WITHOUT --save-fields: the documented
    hard error naming the fix."""
    from raytracer.tests import scenehelpers as sh
    import common
    import run_trace
    src = sh.source_body(power_mW=2.0, coherent=True, half=5e-4,
                         lambdac_nm=633.0, x=-0.02,
                         polarization={"kind": "linear", "angle_deg": 0.0})
    det = sh.detector_body(x=0.02, half=0.004)
    model = sh.make_model([src, det])
    common.validate_model(model)
    mj = tmp_path / "model.json"
    mj.write_text(json.dumps(model))
    case = tmp_path / "case"
    rc = run_trace.main([
        "--model-json", str(mj), "--case-dir", str(case),
        "--rays", "5000", "--nlambda", "1", "--resolution", "32",
        "--power-floor", "1e-8", "--no-gather-gate",
    ])
    assert rc == 0
    obj = tmp_path / "target.npy"
    np.save(obj, _bar_target(n=32, period=8))
    with pytest.raises(SystemExit, match="--save-fields"):
        post_process.main([
            "--case-dir", str(case), "--model-json", str(mj),
            "--image-sim", str(obj)])
