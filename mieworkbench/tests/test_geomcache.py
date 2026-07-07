"""GeomCache unit tests with an in-process fake FcClient (no FreeCAD)."""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.geomcache import GeomCache  # noqa: E402


class FakeClient:
    """Answers get_structure/tessellate from a mutable in-memory model."""

    def __init__(self):
        self.tessellate_calls = []
        self.bodies = {
            "Lens": {"shape_key": "k1", "pos": [0.0, 0.0, 0.0]},
            "Screen": {"shape_key": "k9", "pos": [50.0, 0.0, 0.0]},
        }

    def request(self, op, params=None):
        params = params or {}
        if op == "get_structure":
            return {"doc": "d", "bodies": [
                {"name": n, "shape_key": b["shape_key"],
                 "placement": {"pos_mm": b["pos"], "quat": [0, 0, 0, 1]}}
                for n, b in self.bodies.items()], "sheets": []}
        if op == "tessellate":
            names = params.get("bodies") or list(self.bodies)
            self.tessellate_calls.append(sorted(names))
            out = {}
            for n in names:
                b = self.bodies[n]
                stl = os.path.join(params["out_dir"],
                                   "%s.Tip.Face1.stl" % n)
                with open(stl, "wb") as fh:
                    fh.write(b"\x00" * 84)  # minimal binary STL shell
                out[n] = {"faces": [{"id": "%s.Tip.Face1" % n, "stl": stl,
                                     "area_m2": 1.0}],
                          "shape_key": b["shape_key"],
                          "placement": {"pos_mm": b["pos"],
                                        "quat": [0, 0, 0, 1]}}
            return {"bodies": out}
        raise AssertionError("unexpected op %r" % op)


def test_miss_then_hit(tmp_path):
    client = FakeClient()
    cache = GeomCache(client, cache_root=str(tmp_path))
    r1 = cache.faces_for("d", "/proj/model.FCStd")
    assert len(client.tessellate_calls) == 1
    assert set(r1) == {"Lens", "Screen"}
    assert os.path.isfile(r1["Lens"]["faces"][0]["stl"])

    r2 = cache.faces_for("d", "/proj/model.FCStd")
    assert len(client.tessellate_calls) == 1  # pure cache hit
    assert r2["Lens"]["faces"][0]["stl"] == r1["Lens"]["faces"][0]["stl"]


def test_placement_move_is_cache_hit_with_fresh_placement(tmp_path):
    client = FakeClient()
    cache = GeomCache(client, cache_root=str(tmp_path))
    cache.faces_for("d", "/proj/model.FCStd")
    client.bodies["Lens"]["pos"] = [7.5, 0.0, 0.0]  # move, same shape_key
    r = cache.faces_for("d", "/proj/model.FCStd")
    assert len(client.tessellate_calls) == 1  # no re-tessellation
    assert r["Lens"]["placement"]["pos_mm"] == [7.5, 0.0, 0.0]  # live value


def test_shape_change_retessellates_only_that_body(tmp_path):
    client = FakeClient()
    cache = GeomCache(client, cache_root=str(tmp_path))
    cache.faces_for("d", "/proj/model.FCStd")
    client.bodies["Lens"]["shape_key"] = "k2"
    cache.faces_for("d", "/proj/model.FCStd")
    assert client.tessellate_calls[-1] == ["Lens"]  # Screen untouched


def test_invalidate(tmp_path):
    client = FakeClient()
    cache = GeomCache(client, cache_root=str(tmp_path))
    cache.faces_for("d", "/proj/model.FCStd")
    cache.invalidate("/proj/model.FCStd")
    cache.faces_for("d", "/proj/model.FCStd")
    assert len(client.tessellate_calls) == 2


def test_missing_stl_forces_retessellation(tmp_path):
    client = FakeClient()
    cache = GeomCache(client, cache_root=str(tmp_path))
    r = cache.faces_for("d", "/proj/model.FCStd")
    os.remove(r["Lens"]["faces"][0]["stl"])
    cache.faces_for("d", "/proj/model.FCStd")
    assert client.tessellate_calls[-1] == ["Lens"]
