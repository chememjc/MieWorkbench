# =============================================================================
# test_sources_lines.py -- discrete emission-line sources (samples-
# instruments round).
#
# optprops.load_emission's 'lines' kind carries {lines_nm, lines_um,
# intensity, linewidth_nm} and deliberately has NO lam_um/relative_power
# keys (a consumer that treats a line source as continuous fails loudly
# instead of mis-sampling -- see test_opticalproperties.py). This file
# covers the engine-side sampling that kind needs:
#   raytracer.sources._lines_stratum_counts -- intensity-proportional
#       stratum-budget allocation over the lines, every KEPT line >= 1,
#       weakest lines dropped (with a warning) when n_lambda < n_lines.
#   raytracer.sources._lines_strata / wavelength_strata -- the actual
#       StratumWavelengths (lam + edges) a 'lines' source dict produces.
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_sources_lines.py -q
# =============================================================================
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import common                                                    # noqa: E402
from raytracer.scene import Scene                                # noqa: E402
from raytracer.sources import (wavelength_strata, sample_source,  # noqa: E402
                               _lines_strata, _lines_stratum_counts)
from raytracer.optprops import load_optical_properties           # noqa: E402
from . import scenehelpers as sh                                 # noqa: E402


def _lines_src(lines_nm, intensity, linewidth_nm=0.1, lambdac_nm=None):
    """A source dict carrying the resolved _lines_* arrays -- what
    Scene attaches for a 'lines'-kind spectrum (scene.py), built
    directly here the same way test_emission_spectrum.py's
    _spectrum_src injects _spectrum_lam_nm/_spectrum_pdf without going
    through the registry/Scene machinery."""
    lines_nm = np.asarray(lines_nm, dtype=np.float64)
    return {"lambdac_nm": lambdac_nm if lambdac_nm is not None
            else float(lines_nm[0]),
            "_lines_nm": lines_nm,
            "_lines_intensity": np.asarray(intensity, dtype=np.float64),
            "_lines_linewidth_nm": float(linewidth_nm)}


# ---------------------------------------------------------------------------
# (b) n_lambda=9 over 3 lines, intensities 3:1:5 -> exact proportional
# stratum counts, all stratum lambdas at line centers, per-stratum power
# equal within a line, total per-line power ratio == intensity ratio.
# ---------------------------------------------------------------------------
def test_stratum_counts_proportional_to_intensity():
    lines_nm = [500.0, 550.0, 600.0]
    intensity = [3.0, 1.0, 5.0]
    counts, keep_idx, dropped_idx = _lines_stratum_counts(
        9, np.asarray(intensity))
    assert dropped_idx.size == 0
    assert keep_idx.tolist() == [0, 1, 2]
    assert counts.tolist() == [3, 1, 5]


def test_lines_strata_lambdas_and_power_ratio():
    lines_nm = [500.0, 550.0, 600.0]
    intensity = [3.0, 1.0, 5.0]
    src = _lines_src(lines_nm, intensity)
    lam_m = wavelength_strata(src, 9)
    assert lam_m.shape == (9,)
    lam_nm = lam_m * 1e9
    # every stratum sits EXACTLY at one of the three line centers -- no
    # interpolated/synthetic wavelength ever appears
    uniq = np.unique(np.round(lam_nm, 6))
    assert uniq.tolist() == pytest.approx([500.0, 550.0, 600.0])
    # per-line stratum counts (== per-line ray-count share sample_source
    # draws from, at equal per-ray power -- this IS the power split)
    counts = {ln: int(np.sum(np.isclose(lam_nm, ln))) for ln in lines_nm}
    assert counts == {500.0: 3, 550.0: 1, 600.0: 5}
    # total per-line power ratio (== stratum-count ratio here) exactly
    # matches the intensity ratio 3:1:5
    total = sum(counts.values())
    for ln, inten in zip(lines_nm, intensity):
        assert counts[ln] / total == pytest.approx(inten / sum(intensity))


def test_lines_strata_edges_finite_and_nondecreasing_multiline():
    """General multi-line safety net: even though only a single-line
    fixture (below) gets an EXACT per-stratum width guarantee (see
    _lines_strata's docstring on the shared-boundary compromise at
    line-to-line transitions), the returned edges must always stay
    finite and monotonically non-decreasing (no zero/negative-width
    stratum, whatever the real inter-line gaps are)."""
    lines_nm = [500.0, 550.0, 600.0]
    intensity = [3.0, 1.0, 5.0]
    src = _lines_src(lines_nm, intensity, linewidth_nm=0.05)
    lam_m = wavelength_strata(src, 9)
    edges_nm = lam_m.edges * 1e9
    assert edges_nm.shape == (10,)
    assert np.all(np.isfinite(edges_nm))
    assert np.all(np.diff(edges_nm) > 0)


# ---------------------------------------------------------------------------
# (c) n_lambda=2 over 3 lines -> strongest 2 kept + a one-line warning
# naming the dropped line(s) + their total power share; power
# renormalizes over the kept lines.
# ---------------------------------------------------------------------------
def test_too_few_strata_drops_weakest_line_with_warning():
    lines_nm = [500.0, 550.0, 600.0]
    intensity = [3.0, 1.0, 5.0]           # weakest is 550.0 nm (1.0)
    src = _lines_src(lines_nm, intensity)
    with pytest.warns(UserWarning) as rec:
        lam_m = wavelength_strata(src, 2)
    msgs = [str(w.message) for w in rec.list]
    assert any("550" in m and "dropping" in m for m in msgs), msgs
    # the warning names the dropped line's power share: 1/(3+1+5) ~ 11.1%
    assert any("11.1%" in m for m in msgs), msgs
    lam_nm = np.sort(lam_m * 1e9)
    assert lam_nm.tolist() == pytest.approx([500.0, 600.0])


def test_too_few_strata_counts_helper():
    counts, keep_idx, dropped_idx = _lines_stratum_counts(
        2, np.asarray([3.0, 1.0, 5.0]))
    assert counts.tolist() == [1, 1]
    assert keep_idx.tolist() == [0, 2]     # strongest two, ascending index
    assert dropped_idx.tolist() == [1]


# ---------------------------------------------------------------------------
# (d) stratum edges: finite, non-overlapping, width == linewidth_nm.
# Single-line fixture sidesteps the line-to-line shared-boundary
# compromise entirely (see _lines_strata's docstring) so every stratum's
# width is exactly right, not just approximately so.
# ---------------------------------------------------------------------------
def test_single_line_single_stratum_edges_exact_width():
    src = _lines_src([500.0], [1.0], linewidth_nm=0.2)
    lam_m = wavelength_strata(src, 1)
    assert lam_m.shape == (1,)
    assert lam_m[0] * 1e9 == pytest.approx(500.0)
    edges_nm = lam_m.edges * 1e9
    assert edges_nm.shape == (2,)
    assert np.all(np.isfinite(edges_nm))
    assert edges_nm[1] > edges_nm[0]
    assert edges_nm[1] - edges_nm[0] == pytest.approx(0.2)
    assert edges_nm[0] == pytest.approx(499.9)
    assert edges_nm[1] == pytest.approx(500.1)


def test_single_line_multi_stratum_edges_split_evenly():
    src = _lines_src([500.0], [1.0], linewidth_nm=0.2)
    lam_m = wavelength_strata(src, 4)
    assert lam_m.shape == (4,)
    assert np.all(lam_m * 1e9 == pytest.approx(500.0))
    edges_nm = lam_m.edges * 1e9
    assert edges_nm.shape == (5,)
    assert np.all(np.isfinite(edges_nm))
    widths = np.diff(edges_nm)
    assert np.all(np.isfinite(widths)) and np.all(widths > 0)
    # each of the 4 strata gets an EXACT equal quarter of the line's band
    assert widths == pytest.approx(0.05 * np.ones(4))
    assert widths.sum() == pytest.approx(0.2)          # == linewidth_nm


# ---------------------------------------------------------------------------
# (e) end-to-end smoke: a real source body + Scene, with the _lines_*
# fields attached the way Scene() would for a spectrum naming a 'lines'
# emission row -- sample_source must hand out ONLY line-center
# wavelengths, nothing interpolated in between.
# ---------------------------------------------------------------------------
def test_sample_source_rays_get_line_center_wavelengths_only():
    model = sh.make_model([
        sh.source_body("Src", x=-0.02, half=0.004, power_mW=2.0,
                       lambdac_nm=500.0),
        sh.detector_body("Det", x=0.03, half=0.03),
    ])
    common.validate_model(model)
    optprops = load_optical_properties()
    scene = Scene(model, optprops.matdb, optprops.coatings,
                 optprops=optprops)
    bidx, src = scene.sources[0]
    # attach the resolved _lines_* fields directly (bypassing the
    # registry/spectrum-property plumbing, same as _lines_src above)
    src["_lines_nm"] = np.array([500.0, 600.0])
    src["_lines_intensity"] = np.array([1.0, 1.0])
    src["_lines_linewidth_nm"] = 0.1

    rng = np.random.default_rng(5)
    n_lambda = 6
    batch = sample_source(scene, scene.bodies[bidx], src, 0, 3000, n_lambda,
                          rng)
    lam_nm = np.unique(np.round(batch.lam * 1e9, 6))
    assert lam_nm.tolist() == pytest.approx([500.0, 600.0])
    # equal intensity -> equal stratum split -> equal ray share (up to the
    # ordinary idx % n_strata rounding for n_rays not a clean multiple)
    n0 = int(np.sum(np.isclose(batch.lam * 1e9, 500.0)))
    n1 = int(np.sum(np.isclose(batch.lam * 1e9, 600.0)))
    assert n0 + n1 == len(batch)
    assert n0 == pytest.approx(n1, rel=0.02)
