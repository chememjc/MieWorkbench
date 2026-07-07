"""GeomCache - persistent per-face tessellation cache for the 3D views.

Layout (under the cache root, default <repo>/var/cache):
    <sha1(abs fcstd path)>/<body_name>/<shape_key>/
        meta.json                    {faces:[{id, stl, area_m2, ...}], placement}
        <face_id>.stl                body-local, metres

The shape_key comes from the FreeCAD worker (placement-independent geometric
fingerprint), so a placement-only move never invalidates the cache, while a
spreadsheet/param edit that rebuilds a body's shape produces a new key and
only that body is re-tessellated.
"""

import hashlib
import json
import os
import shutil


def _repo_root():
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def default_cache_root():
    return os.environ.get(
        "MIEWB_CACHE_DIR", os.path.join(_repo_root(), "var", "cache"))


class GeomCache:
    def __init__(self, fcclient, cache_root=None):
        self.client = fcclient
        self.cache_root = cache_root or default_cache_root()

    def _model_dir(self, fcstd_path):
        digest = hashlib.sha1(
            os.path.abspath(fcstd_path).encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_root, digest)

    def _body_dir(self, fcstd_path, body_name, shape_key):
        safe_key = shape_key.replace("/", "_")
        return os.path.join(self._model_dir(fcstd_path), body_name, safe_key)

    # -- public ------------------------------------------------------------
    def faces_for(self, doc, fcstd_path, structure=None, bodies=None):
        """Return {body_name: {"faces": [...], "placement": {...}}} with every
        face STL present on disk, tessellating only stale/missing bodies.

        `structure` is the worker's get_structure result (fetched if None);
        `bodies` optionally restricts to a subset of body names.
        """
        if structure is None:
            structure = self.client.request("get_structure", {"doc": doc})
        out = {}
        stale = []
        for body in structure["bodies"]:
            name = body["name"]
            if bodies is not None and name not in bodies:
                continue
            bdir = self._body_dir(fcstd_path, name, body["shape_key"])
            meta = self._load_meta(bdir)
            if meta is None:
                stale.append(name)
            else:
                # live placement always comes from the structure, not the cache
                out[name] = {"faces": meta["faces"],
                             "placement": body["placement"],
                             "shape_key": body["shape_key"]}
        if stale:
            fresh = self._tessellate(doc, fcstd_path, stale)
            out.update(fresh)
        return out

    def invalidate(self, fcstd_path, body_name=None):
        target = self._model_dir(fcstd_path)
        if body_name is not None:
            target = os.path.join(target, body_name)
        shutil.rmtree(target, ignore_errors=True)

    # -- internals -----------------------------------------------------------
    def _load_meta(self, bdir):
        meta_path = os.path.join(bdir, "meta.json")
        try:
            with open(meta_path, "r") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            return None
        for face in meta.get("faces", []):
            if not os.path.isfile(face.get("stl", "")):
                return None
        return meta

    def _tessellate(self, doc, fcstd_path, body_names):
        out = {}
        # tessellate into a scratch dir, then move per-body into keyed dirs
        scratch = os.path.join(self._model_dir(fcstd_path), "_scratch")
        shutil.rmtree(scratch, ignore_errors=True)
        os.makedirs(scratch, exist_ok=True)
        result = self.client.request(
            "tessellate", {"doc": doc, "out_dir": scratch,
                           "bodies": body_names})
        for name, info in result["bodies"].items():
            bdir = self._body_dir(fcstd_path, name, info["shape_key"])
            shutil.rmtree(bdir, ignore_errors=True)
            os.makedirs(bdir, exist_ok=True)
            faces = []
            for face in info["faces"]:
                dest = os.path.join(bdir, os.path.basename(face["stl"]))
                shutil.move(face["stl"], dest)
                face = dict(face, stl=dest)
                faces.append(face)
            meta = {"faces": faces, "shape_key": info["shape_key"]}
            with open(os.path.join(bdir, "meta.json"), "w") as fh:
                json.dump(meta, fh, indent=1)
            out[name] = {"faces": faces, "placement": info["placement"],
                         "shape_key": info["shape_key"]}
        shutil.rmtree(scratch, ignore_errors=True)
        return out
