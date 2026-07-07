# =============================================================================
# test_preview_rays.py — scripts/preview_rays.py (the GUI's lightweight,
# viz-only ray-overlay generator) exercised in-process on a synthetic
# model.json fixture (same scenehelpers bodies test_viz_pattern.py's
# build_scene() uses), so it never needs a real FreeCAD extract in CI.
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_preview_rays.py -q
# =============================================================================
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import preview_rays                                       # noqa: E402

from .scenehelpers import (detector_body, make_model,      # noqa: E402
                           slab_body, source_body)


def write_geometry(tmp_path, model, name="scene"):
    geometry_dir = tmp_path / "geometry" / name
    geometry_dir.mkdir(parents=True)
    with open(geometry_dir / "model.json", "w") as fh:
        json.dump(model, fh)
    return geometry_dir


def build_model():
    """Same 10x10mm-square-source -> detector layout as
    test_viz_pattern.py's build_scene()."""
    return make_model([
        source_body("Src", x=-0.02, half=0.005, power_mW=5.0,
                    lambdac_nm=633.0),
        detector_body("Det", x=0.02, half=0.01),
    ])


def test_main_writes_vtp_with_expected_polyline_count(tmp_path):
    geometry_dir = write_geometry(tmp_path, build_model())
    out = tmp_path / "rays.vtp"
    rc = preview_rays.main([
        "--geometry", str(geometry_dir),
        "--out", str(out),
        "--pattern", "fan:n=5",
    ])
    assert rc == 0
    assert out.exists()
    # 5 rays * 2 polyline legs each (source -> detector, then the
    # transparent-screen pass-through -> scene escape), matching
    # test_viz_pattern.py's own bit-identity conventions for this scene.
    text = out.read_text()
    assert 'NumberOfLines="10"' in text


def test_main_default_pattern_is_fan(tmp_path):
    geometry_dir = write_geometry(tmp_path, build_model())
    out = tmp_path / "rays.vtp"
    rc = preview_rays.main([
        "--geometry", str(geometry_dir),
        "--out", str(out),
    ])
    assert rc == 0
    assert 'NumberOfLines="10"' in out.read_text()   # fan:n=5 default


def test_main_rejects_missing_model_json(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    rc = preview_rays.main([
        "--geometry", str(empty_dir),
        "--out", str(tmp_path / "rays.vtp"),
    ])
    assert rc != 0
    assert "model.json" in capsys.readouterr().err


def test_main_rejects_model_with_no_sources(tmp_path, capsys):
    model = make_model([detector_body("Det", x=0.02, half=0.01)])
    geometry_dir = write_geometry(tmp_path, model)
    rc = preview_rays.main([
        "--geometry", str(geometry_dir),
        "--out", str(tmp_path / "rays.vtp"),
    ])
    assert rc != 0
    assert "source" in capsys.readouterr().err


def test_main_rejects_bad_pattern(tmp_path, capsys):
    geometry_dir = write_geometry(tmp_path, build_model())
    rc = preview_rays.main([
        "--geometry", str(geometry_dir),
        "--out", str(tmp_path / "rays.vtp"),
        "--pattern", "spiral:whatever",
    ])
    assert rc != 0
    assert "pattern" in capsys.readouterr().err


def test_filter_only_bodies_keeps_sources_and_detectors_and_named():
    model = make_model([
        source_body("Src", x=-0.02, half=0.005),
        slab_body("Lens1", "BK7", -0.01, -0.005, half=0.01),
        slab_body("Lens2", "BK7", 0.0, 0.005, half=0.01),
        detector_body("Det", x=0.02, half=0.01),
    ])
    preview_rays._filter_only_bodies(model, ["Lens1"])
    names = {b["name"] for b in model["bodies"]}
    assert names == {"Src", "Lens1", "Det"}


def test_filter_only_bodies_noop_when_none_given():
    model = build_model()
    before = [b["name"] for b in model["bodies"]]
    preview_rays._filter_only_bodies(model, None)
    assert [b["name"] for b in model["bodies"]] == before


def test_only_bodies_end_to_end_drops_unlisted_optic(tmp_path):
    model = make_model([
        source_body("Src", x=-0.02, half=0.005, power_mW=5.0,
                    lambdac_nm=633.0),
        slab_body("Lens1", "BK7", -0.01, -0.005, half=0.01),
        detector_body("Det", x=0.02, half=0.01),
    ])
    geometry_dir = write_geometry(tmp_path, model)
    out = tmp_path / "rays.vtp"
    rc = preview_rays.main([
        "--geometry", str(geometry_dir),
        "--out", str(out),
        "--pattern", "fan:n=1",
        "--only-bodies", "NoSuchBody",   # drops Lens1, keeps Src + Det
    ])
    assert rc == 0
    assert out.exists()


def test_main_previews_detectorless_scene(tmp_path):
    """A half-built scene (source + lens, no detector yet) is exactly when
    a preview matters — a synthetic transparent far-field detector is
    injected so Scene()'s invariant holds without touching ray paths."""
    model = make_model([
        source_body("Src", x=-0.02, half=0.005, power_mW=5.0,
                    lambdac_nm=633.0),
        slab_body("Window", "bk7", 0.0, 0.005),
    ])
    geometry_dir = write_geometry(tmp_path, model)
    out = tmp_path / "rays.vtp"
    rc = preview_rays.main([
        "--geometry", str(geometry_dir),
        "--out", str(out),
        "--pattern", "fan:n=5",
    ])
    assert rc == 0
    assert out.exists()
    assert 'NumberOfLines' in out.read_text()
