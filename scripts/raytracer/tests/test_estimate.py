# =============================================================================
# test_estimate.py — common.estimate()'s fields_h5_GB prediction
# (design-usability round): wired to save_fields/n_detectors/n_pol_strata
# instead of silently ignoring them. See common.estimate()'s field_bytes
# comment for the exact formula this pins.
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_estimate.py -q
# =============================================================================
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import common                                              # noqa: E402


def test_fields_h5_GB_zero_when_save_fields_off():
    est = common.estimate(1e5, 64, 3, 1, "numpy", n_detectors=4,
                          save_fields=False, n_pol_strata=2)
    assert est["fields_h5_GB"] == 0.0


def test_fields_h5_GB_zero_by_default():
    # save_fields defaults to False even with a nonzero n_detectors.
    est = common.estimate(1e5, 64, 3, 1, "numpy", n_detectors=4)
    assert est["fields_h5_GB"] == 0.0


def test_fields_h5_GB_positive_when_save_fields_on():
    est = common.estimate(1e5, 64, 3, 1, "numpy", n_detectors=1,
                          save_fields=True, n_pol_strata=1)
    assert est["fields_h5_GB"] > 0.0


def test_fields_h5_GB_scales_linearly_with_detector_count():
    # this is the --save-fields-detectors seam: fewer detectors saving
    # fields must predict proportionally less disk.
    est1 = common.estimate(1e5, 64, 3, 1, "numpy", n_detectors=1,
                           save_fields=True, n_pol_strata=1)
    est2 = common.estimate(1e5, 64, 3, 1, "numpy", n_detectors=2,
                           save_fields=True, n_pol_strata=1)
    est3 = common.estimate(1e5, 64, 3, 1, "numpy", n_detectors=3,
                           save_fields=True, n_pol_strata=1)
    assert est2["fields_h5_GB"] == pytest.approx(2 * est1["fields_h5_GB"])
    assert est3["fields_h5_GB"] == pytest.approx(3 * est1["fields_h5_GB"])


def test_fields_h5_GB_zero_detectors_is_zero():
    est = common.estimate(1e5, 64, 3, 1, "numpy", n_detectors=0,
                          save_fields=True, n_pol_strata=1)
    assert est["fields_h5_GB"] == 0.0


def test_fields_h5_GB_scales_with_resolution_squared():
    est_lo = common.estimate(1e5, 32, 3, 1, "numpy", n_detectors=1,
                             save_fields=True, n_pol_strata=1)
    est_hi = common.estimate(1e5, 64, 3, 1, "numpy", n_detectors=1,
                             save_fields=True, n_pol_strata=1)
    # resolution doubles -> npix (= resolution^2) quadruples
    assert est_hi["fields_h5_GB"] == pytest.approx(
        4 * est_lo["fields_h5_GB"])


def test_fields_h5_GB_scales_with_gather_key_count():
    # n_keys = n_coherent_sources * nlambda * n_pol_strata
    base = common.estimate(1e5, 40, 2, 1, "numpy", n_detectors=1,
                           save_fields=True, n_pol_strata=1)
    more_lambda = common.estimate(1e5, 40, 4, 1, "numpy", n_detectors=1,
                                  save_fields=True, n_pol_strata=1)
    more_pol = common.estimate(1e5, 40, 2, 1, "numpy", n_detectors=1,
                               save_fields=True, n_pol_strata=2)
    more_sources = common.estimate(1e5, 40, 2, 2, "numpy", n_detectors=1,
                                   save_fields=True, n_pol_strata=1)
    assert more_lambda["fields_h5_GB"] == pytest.approx(
        2 * base["fields_h5_GB"])
    assert more_pol["fields_h5_GB"] == pytest.approx(2 * base["fields_h5_GB"])
    assert more_sources["fields_h5_GB"] == pytest.approx(
        2 * base["fields_h5_GB"])


def test_fields_h5_GB_matches_documented_formula():
    rays, res, nlambda, n_coh, n_pol, n_det = 1e5, 40, 5, 2, 2, 3
    est = common.estimate(rays, res, nlambda, n_coh, "numpy",
                          n_detectors=n_det, save_fields=True,
                          n_pol_strata=n_pol)
    npix = res * res
    n_keys = n_coh * nlambda * n_pol
    expected_bytes = npix * 2 * 16 * n_keys * n_det   # Ex+Ey, complex128
    assert est["fields_h5_GB"] == pytest.approx(expected_bytes / 1e9)


def test_accumulator_GB_unaffected_by_save_fields():
    # fields_h5_GB is additive/independent — the existing coherent
    # accumulator estimate must not change when save_fields flips.
    est_off = common.estimate(1e5, 64, 3, 1, "numpy", n_detectors=2,
                              save_fields=False, n_pol_strata=1)
    est_on = common.estimate(1e5, 64, 3, 1, "numpy", n_detectors=2,
                             save_fields=True, n_pol_strata=1)
    assert est_off["accumulator_GB"] == pytest.approx(
        est_on["accumulator_GB"])
