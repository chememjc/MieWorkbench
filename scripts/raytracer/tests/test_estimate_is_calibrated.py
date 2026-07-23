# =============================================================================
# test_estimate_is_calibrated.py — common.estimate_is_calibrated(), the GUI
# "estimate (calibrated)" vs "estimate (uncalibrated fallback)" label helper
# added for the P1 per-run accuracy-vs-time dialog. It must NEVER feed the
# runtime law itself (see estimate() in common.py) — it only reports
# whether estimate() would have found a real .calibration.json sample.
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_estimate_is_calibrated.py -q
# =============================================================================
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import common                                              # noqa: E402


@pytest.fixture(autouse=True)
def isolated_calibration(tmp_path, monkeypatch):
    """Every test in this file gets its own empty .calibration.json —
    never read or write the real repo results/.calibration.json."""
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(common, "CALIBRATION_JSON",
                        tmp_path / ".calibration.json")
    yield


def test_false_when_no_calibration_file_exists():
    assert common.estimate_is_calibrated("torch") is False
    assert common.estimate_is_calibrated("torch", model_stem="x") is False


def test_false_with_empty_calibration_file():
    common.write_json(common.CALIBRATION_JSON, [])
    assert common.estimate_is_calibrated("torch") is False


def test_true_after_global_trace_calibration():
    common.record_calibration("trace_rays_per_s_v2", 3e5)
    assert common.estimate_is_calibrated("torch") is True


def test_true_after_global_gather_calibration():
    common.record_calibration("gather_pairs_per_s_torch", 1.2e9)
    assert common.estimate_is_calibrated("torch") is True
    # a different backend's key must not count
    assert common.estimate_is_calibrated("numpy") is False


def test_true_after_per_scene_trace_calibration_only_for_that_scene():
    common.record_calibration("trace_rps_py:some_scene", 4e5)
    assert common.estimate_is_calibrated("torch", model_stem="some_scene") \
        is True
    assert common.estimate_is_calibrated("torch", model_stem="other_scene") \
        is False
    # without a model_stem, the per-scene sample doesn't apply
    assert common.estimate_is_calibrated("torch") is False


def test_true_after_per_scene_spr_calibration():
    common.record_calibration("spr:some_scene", 5.7)
    assert common.estimate_is_calibrated("torch", model_stem="some_scene") \
        is True


def test_c_backend_uses_c_engine_kinds():
    common.record_calibration("trace_c", 3e5)
    assert common.estimate_is_calibrated("c") is True
    # the python-engine kind must not satisfy the c-engine backend
    assert common.estimate_is_calibrated("numpy") is False


def test_zero_rate_entries_are_ignored():
    common.record_calibration("trace_rays_per_s_v2", 0)
    assert common.estimate_is_calibrated("torch") is False
