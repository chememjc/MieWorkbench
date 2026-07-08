# =============================================================================
# test_ray_dimming.py — the rel_power viz column (attenuation dimming data).
#
# Gates:
#   1. VizStore records rel_power = power/birth_power per segment: exactly
#      1.0 leaving the source, Fresnel-split children summing back to the
#      parent's value at an interface (splits dim consistently).
#   2. birth_power == 0 is guarded (rel 0.0, no NaN).
#   3. write_vtp_polylines emits the rel_power float32 CELL array for
#      (N,11) input, keeps rgb the active scalars, and still accepts
#      legacy (N,9)/(N,10) arrays without it.
#
# Run: /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_ray_dimming.py -v
# =============================================================================
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

from raytracer.tracer import VizStore                    # noqa: E402
from raytracer.vtkexport import write_vtp_polylines      # noqa: E402

from .scenehelpers import (make_model, source_body, slab_body,  # noqa: E402
                           detector_body, trace_scene)


X_SRC, X_SLAB0, X_SLAB1 = -0.02, -0.005, 0.005


def _traced_viz():
    model = make_model([
        source_body("Src", x=X_SRC, half=0.001, power_mW=1.0,
                    lambdac_nm=633.0),
        slab_body("Slab", "BK7", X_SLAB0, X_SLAB1),
        detector_body("Det", x=0.03),
    ])
    result, _grids, _scene = trace_scene(model, rays=2000, seed=7)
    return result.viz.as_array()


def test_rel_power_column_semantics():
    viz = _traced_viz()
    assert viz.shape[1] == 13
    rel = viz[:, 10]
    assert np.all((rel >= 0.0) & (rel <= 1.0))

    # segments leaving the emit plane: full power relative to birth
    at_source = np.isclose(viz[:, 3], X_SRC, atol=1e-9)
    assert np.any(at_source)
    assert np.allclose(rel[at_source], 1.0, atol=1e-9)

    # normal-incidence Fresnel split at the slab front face: the two
    # FIRST-generation children (reflected ~R, transmitted ~T) must carry
    # rel values that sum back to the parent's 1.0. Later generations
    # also start segments here (internal bounces re-exiting the front
    # face, e.g. T*R*T), so take the two largest distinct values -- R and
    # T dominate every descendant product of themselves. (Bulk BK7
    # absorption over <=10 mm is within the tolerance.)
    at_front = np.isclose(viz[:, 3], X_SLAB0, atol=1e-9) & ~at_source
    assert np.any(at_front)
    split = np.unique(np.round(rel[at_front], 6))[-2:]
    assert split.min() < 0.1 < 0.8 < split.max()
    assert np.isclose(split.sum(), 1.0, atol=1e-3)


def test_vizstore_zero_birth_power_guarded():
    store = VizStore()
    batch = SimpleNamespace(
        viz_flag=np.array([True, True]),
        source_id=np.array([0, 0]),
        lam=np.array([633e-9, 633e-9]),
        power=np.array([0.0, 0.5]),
        birth_power=np.array([0.0, 1.0]),
        pos=np.zeros((2, 3)),
        pol_mode=np.array([0, 0]),
    )
    store.add(batch, np.ones((2, 3)),
              np.array([0.0, 1e-3]), np.array([5e-4, 2e-3]))
    arr = store.as_array()
    assert arr.shape == (2, 13)
    assert np.all(np.isfinite(arr[:, 10]))
    assert arr[0, 10] == 0.0
    assert arr[1, 10] == 0.5
    assert arr[1, 11] == 1e-3 and arr[1, 12] == 2e-3


def test_vizstore_empty_shape():
    assert VizStore().as_array().shape == (0, 13)


def _seg_rows(ncols, n=3):
    rows = np.zeros((n, ncols))
    rows[:, 0] = 0                       # source_id
    rows[:, 1] = 633e-9                  # lam
    rows[:, 2] = 1e-3                    # power
    rows[:, 3:9] = np.arange(6) * 1e-3   # endpoints
    if ncols >= 10:
        rows[:, 9] = 0                   # pol_mode
    if ncols >= 11:
        rows[:, 10] = [1.0, 0.5, 0.0][:n]
    if ncols >= 13:
        rows[:, 11] = np.arange(n) * 1e-3          # opl0
        rows[:, 12] = np.arange(n) * 1e-3 + 5e-4   # opl1
    return rows


def test_vtp_export_rel_power_cell_array(tmp_path):
    out = tmp_path / "rays.vtp"
    write_vtp_polylines(out, _seg_rows(11))
    text = out.read_text()
    assert 'Name="rel_power"' in text
    assert 'type="Float32" Name="rel_power"' in text
    assert 'Scalars="rgb"' in text       # rgb stays the active scalars
    assert 'Name="rgba"' not in text     # appearance-neutral by default


def test_vtp_export_dim_mode_bakes_rgba(tmp_path):
    out = tmp_path / "rays_dim.vtp"
    write_vtp_polylines(out, _seg_rows(11), dim_mode="linear",
                        dim_floor=5.0)
    text = out.read_text()
    assert 'Name="rgba" NumberOfComponents="4"' in text
    assert 'Scalars="rgb"' in text       # active scalars unchanged
    # legacy widths ignore the dim flags rather than failing
    out10 = tmp_path / "rays10_dim.vtp"
    write_vtp_polylines(out10, _seg_rows(10), dim_mode="linear")
    assert 'Name="rgba"' not in out10.read_text()


def test_vtp_export_legacy_widths_have_no_rel_power(tmp_path):
    for ncols in (9, 10):
        out = tmp_path / ("rays%d.vtp" % ncols)
        write_vtp_polylines(out, _seg_rows(ncols))
        assert 'rel_power' not in out.read_text()


def test_vtp_export_rejects_unknown_width(tmp_path):
    with pytest.raises(ValueError):
        write_vtp_polylines(tmp_path / "bad.vtp", np.zeros((2, 12)))


def test_vtp_export_empty_11col(tmp_path):
    out = tmp_path / "empty.vtp"
    write_vtp_polylines(out, np.zeros((0, 11)))
    assert 'Name="rel_power"' in out.read_text()
