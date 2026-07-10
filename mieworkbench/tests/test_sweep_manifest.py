"""RunController.write_sweep_manifest — prediction/format tests (pure,
no QProcess launch)."""

import json

from mieworkbench.core.runner import RunController


def test_no_sweep_returns_none(tmp_path):
    assert RunController.write_sweep_manifest(
        tmp_path / "m.FCStd", {"preset": "quick"}, tmp_path) is None


def test_product_manifest(tmp_path):
    config = {"preset": "quick", "var": ["miewb_vars.gap", "miewb_vars.t"],
              "min": [10.0, 0.0], "max": [20.0, 1.0], "n": [1, 1],
              "sweep_mode": "product"}
    path = RunController.write_sweep_manifest(
        tmp_path / "model.FCStd", config, tmp_path)
    m = json.loads(path.read_text())
    assert path.name == "sweep-quick.manifest.json"
    assert m["mode"] == "product"
    assert m["order"] == ["miewb_vars.gap", "miewb_vars.t"]
    assert len(m["variants"]) == 4          # 2 x 2 (n=1 -> [min, max])
    v0 = m["variants"][0]
    assert v0["values"] == {"miewb_vars.gap": 10.0, "miewb_vars.t": 0.0}
    assert v0["stem"].startswith("model-")
    assert v0["case_dir"].endswith("/%s/quick" % v0["stem"])
    # every stem unique
    assert len({v["stem"] for v in m["variants"]}) == 4


def test_zip_manifest_counts(tmp_path):
    config = {"preset": "normal", "tag": "night",
              "var": ["miewb_vars.a", "miewb_vars.b"],
              "min": [0.0, 5.0], "max": [2.0, 7.0], "n": [2, 2],
              "sweep_mode": "zip"}
    path = RunController.write_sweep_manifest(
        tmp_path / "model.FCStd", config, tmp_path)
    m = json.loads(path.read_text())
    assert m["case"] == "normal-night"
    assert len(m["variants"]) == 3          # zipped, not 9
    assert [v["values"]["miewb_vars.a"] for v in m["variants"]] \
        == [0.0, 1.0, 2.0]
    assert [v["values"]["miewb_vars.b"] for v in m["variants"]] \
        == [5.0, 6.0, 7.0]
