# =============================================================================
# test_emission_spectrum.py -- tabulated emission-spectrum sources.
#
# The `spectrum` source property names an emission/emitters.miesrc registry
# row (a tabulated relative-power SPD); sources.wavelength_strata samples it
# by inverse-CDF, placing one wavelength per equal-power quantile stratum so
# per-ray birth_power is untouched (unbiased stratified sampling).
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_emission_spectrum.py -q
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common                                                   # noqa: E402
from raytracer.sources import wavelength_strata                 # noqa: E402
from raytracer.optprops import load_optical_properties          # noqa: E402
import run_trace                                                # noqa: E402
from scenehelpers import (source_body, detector_body, make_model,  # noqa: E402
                          trace_scene)


SPEC = "led_white_2733k"


@pytest.fixture(scope="module")
def optprops():
    return load_optical_properties()


@pytest.fixture(scope="module")
def table(optprops):
    e = optprops.emission[SPEC]
    return np.asarray(e["lam_nm"]), np.asarray(e["relative_power"])


def _spectrum_src(optprops):
    """A source dict carrying the resolved _spectrum_* arrays (what Scene
    attaches), without needing to build a full Scene."""
    e = optprops.emission[SPEC]
    return {"lambdac_nm": 584.6,
            "_spectrum_lam_nm": np.asarray(e["lam_nm"], dtype=np.float64),
            "_spectrum_pdf": np.asarray(e["relative_power"], dtype=np.float64)}


def _reference_cdf(lam_nm, pdf, n_dense=8000):
    """Independent fine piecewise-linear CDF of the table PDF: (lam, cdf)
    with cdf normalized to [0, 1]. Used to check the sampler's quantiles."""
    lam_ref = np.linspace(lam_nm[0], lam_nm[-1], n_dense)
    pdf_ref = np.interp(lam_ref, lam_nm, pdf)
    cdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (pdf_ref[1:] + pdf_ref[:-1]) * np.diff(lam_ref))])
    cdf /= cdf[-1]
    return lam_ref, cdf


# ---------------------------------------------------------------------------
# (a) strata sit at the CDF centers (k+0.5)/n of the tabulated PDF
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 3, 5, 16, 64])
def test_strata_are_cdf_centers(optprops, table, n):
    lam_nm, pdf = table
    lam_ref, cdf = _reference_cdf(lam_nm, pdf)
    lam_m = wavelength_strata(_spectrum_src(optprops), n)
    assert lam_m.shape == (n,)
    lam_strata_nm = lam_m * 1e9
    # strata are strictly increasing and inside the table range
    assert np.all(np.diff(lam_strata_nm) > 0)
    assert lam_strata_nm.min() >= lam_nm[0] - 1e-6
    assert lam_strata_nm.max() <= lam_nm[-1] + 1e-6
    # CDF evaluated at each stratum equals its quantile center (k+0.5)/n
    q_expect = (np.arange(n) + 0.5) / n
    q_at_strata = np.interp(lam_strata_nm, lam_ref, cdf)
    assert np.allclose(q_at_strata, q_expect, atol=2e-3)


# ---------------------------------------------------------------------------
# (b) equal power per stratum: integrated PDF between adjacent stratum CDF
#     boundaries (quantiles k/n) is identical across strata
# ---------------------------------------------------------------------------
def test_equal_power_between_stratum_boundaries(optprops, table):
    n = 8
    lam_nm, pdf = table
    lam_ref, cdf = _reference_cdf(lam_nm, pdf)
    total = np.trapezoid(pdf, lam_nm)
    # stratum boundaries are the quantiles 0, 1/n, 2/n, ..., 1 in lambda
    q_bounds = np.arange(n + 1) / n
    lam_bounds = np.interp(q_bounds, cdf, lam_ref)
    powers = []
    for i in range(n):
        mask = (lam_ref >= lam_bounds[i]) & (lam_ref <= lam_bounds[i + 1])
        lo, hi = lam_bounds[i], lam_bounds[i + 1]
        xs = np.concatenate([[lo], lam_ref[mask], [hi]])
        xs = np.unique(xs)
        ys = np.interp(xs, lam_nm, pdf)
        powers.append(np.trapezoid(ys, xs))
    powers = np.asarray(powers)
    assert np.allclose(powers, total / n, rtol=5e-3), powers


# ---------------------------------------------------------------------------
# (d) lam_range covers the whole table
# ---------------------------------------------------------------------------
def test_lam_range_covers_table(optprops, table):
    lam_nm, _ = table
    sb = source_body("Src", x=-0.02, half=0.004, power_mW=2.0,
                     lambdac_nm=584.6, spectrum=SPEC)
    model = make_model([sb, detector_body("Det", x=0.03, half=0.025)])
    common.validate_model(model)
    from raytracer.scene import Scene
    scene = Scene(model, optprops.matdb, optprops.coatings, optprops=optprops)
    lo_m, hi_m = run_trace.lam_range_nm(scene)
    assert lo_m * 1e9 <= lam_nm[0]
    assert hi_m * 1e9 >= lam_nm[-1]


# ---------------------------------------------------------------------------
# (c) + (e) end-to-end: white-LED source -> detector; detected spectral
#     profile follows the LED-B1 shape and energy closes
# ---------------------------------------------------------------------------
def test_end_to_end_profile_and_closure(optprops, table):
    lam_nm, pdf = table
    sb = source_body("Src", x=-0.02, half=0.004, power_mW=2.0,
                     lambdac_nm=584.6, spectrum=SPEC)
    model = make_model([sb, detector_body("Det", x=0.03, half=0.03)])
    result, grids, scene = trace_scene(
        model, rays=6000, n_lambda=96, seed=11, optprops=optprops,
        resolution=64)

    # (e) energy closure < 1e-3
    rep = result.ledger.report(result.source_names)
    for name, s in rep["sources"].items():
        assert s["emitted_W"] == pytest.approx(2e-3, rel=1e-9), name
        assert s["closure_error"] < 1e-3, (name, s["closure_error"])

    # detected spectral profile: sum the cube over space -> per-bin power
    grid = next(iter(grids.values()))
    spec = grid.inc.sum(axis=(1, 2))
    bins = grid.spectral_bins
    lam_c = (grid.lam_lo + (np.arange(bins) + 0.5)
             * (grid.lam_hi - grid.lam_lo) / bins) * 1e9   # nm
    assert spec.sum() > 0
    # nearly all the 2 mW arrives (source small, detector large, air path)
    assert spec.sum() == pytest.approx(2e-3, rel=0.02)

    # expected per-bin shape from the table PDF (0 outside the table)
    expect = np.interp(lam_c, lam_nm, pdf, left=0.0, right=0.0)
    inband = expect > 0
    corr = float(np.corrcoef(spec[inband], expect[inband])[0, 1])
    assert corr > 0.9, "detected-vs-table spectral correlation %.3f" % corr

    # (c) coarse features: blue pump band (420-490 nm) AND phosphor hump
    # (560-660 nm) both carry power; the phosphor hump dominates, matching
    # the table's own red/blue power ratio within tolerance.
    def band(arr, x, lo, hi):
        m = (x >= lo) & (x < hi)
        return float(arr[m].sum())

    blue_det = band(spec, lam_c, 420, 490)
    red_det = band(spec, lam_c, 560, 660)
    assert blue_det > 0 and red_det > 0
    assert red_det > blue_det                       # phosphor hump dominates

    blue_tab = np.trapezoid(*(lambda m: (pdf[m], lam_nm[m]))(
        (lam_nm >= 420) & (lam_nm <= 490)))
    red_tab = np.trapezoid(*(lambda m: (pdf[m], lam_nm[m]))(
        (lam_nm >= 560) & (lam_nm <= 660)))
    ratio_det = red_det / blue_det
    ratio_tab = red_tab / blue_tab
    assert ratio_det == pytest.approx(ratio_tab, rel=0.30), \
        (ratio_det, ratio_tab)


def test_zero_width_band_is_monochromatic():
    """lambdamin == lambdamax == lambdac is a valid spelling of
    'monochromatic' (used to divide by zero in the half-normal split)."""
    src = {"lambdac_nm": 633.0, "lambdamin_nm": 633.0,
           "lambdamax_nm": 633.0}
    lam = wavelength_strata(src, 5)
    assert lam.shape == (1,)
    assert lam[0] == pytest.approx(633e-9)
