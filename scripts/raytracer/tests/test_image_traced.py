# =============================================================================
# test_image_traced.py -- render_image_traced (samples-instruments round):
# the extended image-source follow-on that publishes the traced end-to-end
# detector image into imaging/ and, when --image-sim also ran, a
# traced-vs-sim side-by-side with an NCC agreement metric (direct vs
# 180-degree-rotated orientations — a real bench inverts).
# =============================================================================
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import post_process as pp                                        # noqa: E402


def _case(tmp_path, img, label="Det.Face1"):
    case = tmp_path / "case"
    (case / "detectors").mkdir(parents=True)
    H, W = img.shape
    cube = img[None, :, :].astype(np.float64)
    with h5py.File(case / "detectors" / (label + ".h5"), "w") as h:
        h["spectral_cube_mean"] = cube
        h["mask"] = np.ones((H, W), dtype=bool)
        h.attrs.update({"label": label, "H": H, "W": W, "pixel_m": 5e-5,
                        "lam_lo_m": 400e-9, "lam_hi_m": 800e-9,
                        "x_lo": 0.0, "y_lo": 0.0, "seeds": 1})
    return case, [case / "detectors" / (label + ".h5")]


def _model(with_image=True):
    src = {"power_mW": 1.0, "lambdac_nm": 550.0}
    if with_image:
        src["image"] = "usaf_style_target"
    return {"bodies": [{"name": "Src", "source": src},
                       {"name": "Det"}]}


def _asym_img():
    img = np.zeros((24, 24))
    img[2:8, 2:8] = 1.0        # bright top-left block
    img[16:20, 14:22] = 0.4
    return img


def test_noop_without_image_source(tmp_path):
    case, h5s = _case(tmp_path, _asym_img())
    report = {}
    pp.render_image_traced(case, h5s, report, _model(with_image=False))
    assert "image_traced" not in report
    assert not (case / "imaging").exists()


def test_traced_png_published_standalone(tmp_path):
    case, h5s = _case(tmp_path, _asym_img())
    report = {}
    pp.render_image_traced(case, h5s, report, _model())
    block = report["image_traced"]["Det.Face1"]
    assert (case / block["output"]).exists()
    assert "ncc_vs_sim" not in block        # no sim ran


def test_ncc_prefers_rotated_orientation(tmp_path):
    """The sim image is object-oriented; feed a 180-rotated copy of the
    traced image as the 'sim' and the comparison must pick rotated_180
    with NCC ~ 1."""
    img = _asym_img()
    case, h5s = _case(tmp_path, img)
    idir = case / "imaging"
    idir.mkdir()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(idir / "image_sim_incoherent.png", img[::-1, ::-1],
               cmap="gray")
    report = {"image_sim": {"output": "imaging/image_sim_incoherent.png",
                            "detector": "Det.Face1"}}
    pp.render_image_traced(case, h5s, report, _model())
    block = report["image_traced"]["Det.Face1"]
    assert block["sim_orientation"] == "rotated_180"
    assert block["ncc_vs_sim"] > 0.95
    assert (case / block["comparison"]).exists()


def test_ncc_direct_orientation_and_metric_range(tmp_path):
    img = _asym_img()
    case, h5s = _case(tmp_path, img)
    idir = case / "imaging"
    idir.mkdir()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(idir / "image_sim_incoherent.png", img, cmap="gray")
    report = {"image_sim": {"output": "imaging/image_sim_incoherent.png",
                            "detector": "Det.Face1"}}
    pp.render_image_traced(case, h5s, report, _model())
    block = report["image_traced"]["Det.Face1"]
    assert block["sim_orientation"] == "direct"
    assert 0.95 < block["ncc_vs_sim"] <= 1.0


def test_ncc_helper_bounds():
    rng = np.random.default_rng(0)
    a = rng.random((16, 16))
    assert pp._ncc(a, a) == pytest.approx(1.0)
    assert pp._ncc(a, 3.0 * a + 2.0) == pytest.approx(1.0)   # scale/offset
    assert pp._ncc(a, -a) == pytest.approx(-1.0)
