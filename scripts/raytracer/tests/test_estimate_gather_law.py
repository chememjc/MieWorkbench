# =============================================================================
# test_estimate_gather_law.py — common.estimate()'s coherent-gather runtime
# law (p0/quick-wins rewrite).
#
# The old law multiplied gather cost by nlambda * n_pol_strata *
# n_coherent_sources, but the C/Python gather bills per (source,
# lambda-stratum, pol-stratum) KEY — surviving samples are PARTITIONED
# across those keys, never multiplied by their count
# (cengine/src/gather.c:617 `total_pairs += n_sel * Q`). The new law is
#     pairs    = npix * rays * spr
#     gather_s = gather_init_s + pairs / rate_gather   (any coherent source)
#              = ~0.05                                  (no coherent source)
# and spr/rate/init are all calibratable via common.calibrated_rate() /
# common.record_calibration() against RESULTS_DIR/.calibration.json.
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_estimate_gather_law.py -q
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


def test_no_coherent_source_gives_tiny_gather_floor():
    est = common.estimate(1e5, 512, 5, 0, "torch")
    assert est["gather_s"] == pytest.approx(0.05)
    # and it must be far below what even a trivial coherent gather costs
    # (the ~1.9s CUDA-context-init floor alone)
    assert est["gather_s"] < common.FALLBACK_GATHER_INIT_S


def test_nlambda_does_not_change_the_estimate():
    # nlambda used to multiply gather_ops directly; the gather bills per
    # KEY (partition, not product), so it must be a no-op on gather_s/
    # trace_s/total_s now (fields_h5_GB is the one place nlambda still
    # legitimately matters, and that's covered by test_estimate.py).
    base = common.estimate(1e5, 256, 3, 1, "torch")
    more_lambda = common.estimate(1e5, 256, 9, 1, "torch")
    assert more_lambda["gather_s"] == pytest.approx(base["gather_s"])
    assert more_lambda["trace_s"] == pytest.approx(base["trace_s"])
    assert more_lambda["total_s"] == pytest.approx(base["total_s"])


def test_n_pol_strata_does_not_change_the_estimate():
    base = common.estimate(1e5, 256, 3, 1, "torch", n_pol_strata=1)
    more_pol = common.estimate(1e5, 256, 3, 1, "torch", n_pol_strata=2)
    assert more_pol["gather_s"] == pytest.approx(base["gather_s"])


def test_n_coherent_sources_count_does_not_change_the_estimate():
    # only WHETHER a coherent source exists matters (the >=1 gate), not
    # how many — the old law multiplied pairs by the raw count.
    one_source = common.estimate(1e5, 256, 3, 1, "torch")
    three_sources = common.estimate(1e5, 256, 3, 3, "torch")
    assert three_sources["gather_s"] == pytest.approx(one_source["gather_s"])


def test_gather_s_scales_linearly_with_pairs():
    # pairs = npix * rays * spr: doubling rays (spr/npix held fixed) must
    # double the marginal (post-init) gather cost.
    est1 = common.estimate(1e5, 256, 3, 1, "torch")
    est2 = common.estimate(2e5, 256, 3, 1, "torch")
    marginal1 = est1["gather_s"] - common.FALLBACK_GATHER_INIT_S
    marginal2 = est2["gather_s"] - common.FALLBACK_GATHER_INIT_S
    assert marginal2 == pytest.approx(2 * marginal1)


def test_spr_lookup_path_uses_recorded_calibration():
    # no calibration recorded yet -> DEFAULT_SPR (1.0)
    default_est = common.estimate(1e5, 256, 3, 1, "torch",
                                  model_stem="some_scene")
    assert default_est["spr"] == pytest.approx(common.DEFAULT_SPR)

    # record a scene-specific spr (as run_trace.py's _do_gather does after
    # a completed run) and confirm estimate() picks it up and scales
    # pairs/gather_s proportionally
    common.record_calibration("spr:some_scene", 5.7)
    calibrated_est = common.estimate(1e5, 256, 3, 1, "torch",
                                     model_stem="some_scene")
    assert calibrated_est["spr"] == pytest.approx(5.7)
    assert calibrated_est["gather_pairs"] == pytest.approx(
        default_est["gather_pairs"] * 5.7)

    # a DIFFERENT model_stem must not pick up some_scene's calibration
    other_est = common.estimate(1e5, 256, 3, 1, "torch",
                                model_stem="other_scene")
    assert other_est["spr"] == pytest.approx(common.DEFAULT_SPR)


def test_record_calibration_caps_history_at_five_and_medians():
    for rate in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        common.record_calibration("gather_pairs_per_s_torch", rate)
    with open(common.CALIBRATION_JSON) as fh:
        import json
        entries = json.load(fh)
    same_kind = [e for e in entries
                 if e["kind"] == "gather_pairs_per_s_torch"]
    assert len(same_kind) == 5
    # oldest two (1.0, 2.0) were pruned; median of the remaining
    # [3, 4, 5, 6, 7] is 5.0
    assert common.calibrated_rate("gather_pairs_per_s_torch", -1) == \
        pytest.approx(5.0)


def test_backend_c_uses_its_own_fallback():
    # bare "c" resolves to the CUDA gather kernel's calibration keys;
    # "c_cpu" selects the OpenMP kernel's (~15x slower, distinct key)
    pairs = 256 * 256 * 1e5 * common.DEFAULT_SPR
    est = common.estimate(1e5, 256, 3, 1, "c")
    expected = (common.FALLBACK_GATHER_INIT_S_BY["c_cuda"]
                + pairs / common.FALLBACK_GATHER_PAIRS_PER_S["c_cuda"])
    assert est["gather_s"] == pytest.approx(expected)
    est_cpu = common.estimate(1e5, 256, 3, 1, "c_cpu")
    expected_cpu = (common.FALLBACK_GATHER_INIT_S_BY["c_cpu"]
                    + pairs / common.FALLBACK_GATHER_PAIRS_PER_S["c_cpu"])
    assert est_cpu["gather_s"] == pytest.approx(expected_cpu)
    assert est_cpu["gather_s"] > est["gather_s"]


def test_gpu_free_vram_bytes_never_raises():
    result = common.gpu_free_vram_bytes()
    assert result is None or isinstance(result, int)
