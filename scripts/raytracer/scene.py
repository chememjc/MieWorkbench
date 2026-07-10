# =============================================================================
# scene.py — build the runtime scene from a validated model.json contract.
#
# Responsibilities:
#   * resolve bodies/faces into AnalyticFace objects (mesh faces are a hard
#     error in v1 — the extractor canonicalizes revolutions, so every optical
#     face in a lens/sphere/slit bench is analytic; see future.md)
#   * resolve materials & coatings against the MaterialDB
#   * apply --suppress-body (body treated as ignored, no FCStd edit)
#   * attach per-face physics options (grating, roughness, extra detectors)
#   * nearest-hit intersection across all active faces for a ray batch
#
# Face indexing: faces get integer ids (position in scene.faces); -1 = none.
# Body indexing: position in scene.bodies; AMBIENT (-1) = surrounding air.
# =============================================================================
import numpy as np

from .surfaces import make_surface, AnalyticFace


FACEMAP_ALL = "__all__"     # matches common.FACEMAP_ALL (contract sentinel)


class Body:
    __slots__ = ("index", "name", "label", "role", "material", "coating",
                 "mirror", "absorbance", "roughness_nm", "roughness_faces",
                 "diffuser_faces", "scatter_faces", "grating_map", "source",
                 "detector", "closed", "face_ids", "polarizer",
                 "polarizer_axis", "filter", "crystal_axis", "birefringent",
                 "filter_lam_um", "filter_alpha_per_m", "crystal_axis2",
                 "crystal_frame", "biaxial")

    def __init__(self, index, rec):
        self.index = index
        self.name = rec["name"]
        self.label = rec["label"]
        self.role = rec["role"]
        self.material = rec.get("material")
        # coating: v1 = bare string (whole body), v2 = per-face dict.
        # Normalize to a dict {face_id_or_FACEMAP_ALL: name} (None if none).
        coat = rec.get("coating")
        if coat in (None, "", "none"):
            self.coating = None
        elif isinstance(coat, str):
            self.coating = {FACEMAP_ALL: coat}
        else:
            self.coating = dict(coat)
        self.mirror = float(rec.get("mirror") or 0.0)
        self.absorbance = float(rec.get("absorbance") or 0.0)
        # NOTE the contract key is roughness_nm ('roughness' was a latent
        # mismatch — body-tagged roughness silently never reached the trace)
        self.roughness_nm = float(rec.get("roughness_nm") or 0.0)
        self.roughness_faces = rec.get("roughness_faces")   # dict or None
        self.diffuser_faces = rec.get("diffuser_faces")     # dict or None
        self.scatter_faces = rec.get("scatter_faces")       # dict or None
        self.grating_map = rec.get("grating")               # dict or None
        self.polarizer = rec.get("polarizer") or None
        pa = rec.get("polarizer_axis")
        self.polarizer_axis = np.asarray(pa, dtype=np.float64) \
            if pa is not None else np.array([0.0, 0.0, 1.0])
        self.filter = rec.get("filter") or None
        ca = rec.get("crystal_axis")
        # v1 models carry no crystal_axis: default global +x (README §5)
        self.crystal_axis = np.asarray(ca, dtype=np.float64) \
            if ca is not None else np.array([1.0, 0.0, 0.0])
        self.birefringent = False           # set by Scene from the matdb
        ca2 = rec.get("crystal_axis2")
        self.crystal_axis2 = np.asarray(ca2, dtype=np.float64) \
            if ca2 is not None else None
        self.crystal_frame = None           # (3,3) rows = principal axes,
        self.biaxial = False                # both set by Scene (biaxial)
        self.filter_lam_um = None           # set by Scene from optprops
        self.filter_alpha_per_m = None
        self.source = rec.get("source")
        self.detector = rec.get("detector")
        self.closed = bool(rec.get("solid_closed", True))
        self.face_ids = []

    def filter_alpha(self, lam_m):
        """Additive bulk absorption coefficient [1/m] at wavelength(s) [m]
        from this body's spectral filter table (0 if no filter). Hard error
        outside the tabulated range — no extrapolation."""
        if self.filter_lam_um is None:
            return 0.0
        from .optprops import interp_hard
        return interp_hard(np.asarray(lam_m) * 1e6, self.filter_lam_um,
                           self.filter_alpha_per_m,
                           "body %s filter %r" % (self.label, self.filter))


class Scene:
    def __init__(self, model, matdb, coatings, suppress_bodies=(),
                 extra_detector_faces=(), grating_specs=(), rough_specs=(),
                 optprops=None, geometry_dir=None, strict_analytic=False,
                 mesh_flat_normals=False):
        """model: validated model.json dict; matdb: MaterialDB;
        coatings: {name: {"kind": "tmm"|"table", ...}} from load_coatings;
        optprops: optional OpticalProperties (polarizer/filter/grating
        registries — required when the model uses those properties);
        geometry_dir: directory containing the model's faces/*.stl (needed
        to trace mesh-type faces via the BVH — typically the model.json's
        parent); strict_analytic restores the v1 hard error on mesh faces."""
        self.matdb = matdb
        self.coatings = coatings
        self.optprops = optprops
        polarizers = optprops.polarizers if optprops is not None else {}
        filters = optprops.filters if optprops is not None else {}
        grating_registry = optprops.gratings if optprops is not None else {}
        self.polarizers = polarizers
        self.ambient = matdb.get(model.get("ambient_material", "air"))
        suppress = set(suppress_bodies)

        self.bodies = []
        self.faces = []                    # AnalyticFace, index = face id
        self.face_body = []                # face id -> body index
        self.face_records = []             # face id -> contract face dict
        self.sources = []                  # (body_index, source dict)
        self.emit_faces = {}               # body_index -> AnalyticFace
        self.detector_faces = {}           # face id -> owning body index
        self.face_by_name = {}             # "Body.Feature.FaceN" -> face id

        unknown = [s for s in suppress
                   if s not in {b["name"] for b in model["bodies"]}
                   and s not in {b["label"] for b in model["bodies"]}]
        if unknown:
            raise ValueError("--suppress-body names not in model: %r"
                             % unknown)

        for rec in model["bodies"]:
            if rec["role"] == "ignored":
                continue
            if rec["name"] in suppress or rec["label"] in suppress:
                continue
            body = Body(len(self.bodies), rec)
            # resolve material/coating/polarizer/filter now — unknown names
            # must fail here, not mid-trace
            if body.role == "optic":
                if matdb.is_birefringent(body.material):
                    body.birefringent = True
                    nrm = np.linalg.norm(body.crystal_axis)
                    if nrm < 1e-9:
                        raise ValueError("body %s: zero crystal_axis"
                                         % body.label)
                    body.crystal_axis = body.crystal_axis / nrm
                elif matdb.is_biaxial(body.material):
                    # full principal frame: crystal_axis = X, crystal_axis2
                    # = Y (Gram-Schmidt orthogonalized), Z = X x Y
                    if body.crystal_axis2 is None:
                        raise ValueError(
                            "body %s: biaxial material %r needs BOTH "
                            "crystal_axis (X principal axis) and "
                            "crystal_axis2 (Y)" % (body.label,
                                                   body.material))
                    x = body.crystal_axis
                    nx = np.linalg.norm(x)
                    if nx < 1e-9:
                        raise ValueError("body %s: zero crystal_axis"
                                         % body.label)
                    x = x / nx
                    y = body.crystal_axis2
                    y = y - np.dot(y, x) * x
                    ny = np.linalg.norm(y)
                    if ny < 1e-6:
                        raise ValueError(
                            "body %s: crystal_axis2 is (near-)parallel to "
                            "crystal_axis — principal frame undefined"
                            % body.label)
                    y = y / ny
                    body.crystal_frame = np.stack([x, y, np.cross(x, y)])
                    body.biaxial = True
                else:
                    matdb.get(body.material)
            if body.coating is not None:
                for cname in body.coating.values():
                    if cname not in coatings:
                        raise ValueError(
                            "body %s: unknown coating %r (coatings.csv "
                            "has: %s)" % (body.label, cname,
                                          ", ".join(sorted(coatings))))
            if body.polarizer is not None:
                if body.polarizer not in polarizers:
                    raise ValueError(
                        "body %s: unknown polarizer %r (polarizers.csv "
                        "has: %s)" % (body.label, body.polarizer,
                                      ", ".join(sorted(polarizers)) or
                                      "<none loaded — pass optprops>"))
                nrm = np.linalg.norm(body.polarizer_axis)
                if nrm < 1e-9:
                    raise ValueError("body %s: zero polarizer_axis"
                                     % body.label)
                body.polarizer_axis = body.polarizer_axis / nrm
            if body.filter is not None:
                if body.filter not in filters:
                    raise ValueError(
                        "body %s: unknown filter %r (filters.csv has: %s)"
                        % (body.label, body.filter,
                           ", ".join(sorted(filters)) or
                           "<none loaded — pass optprops>"))
                fentry = filters[body.filter]
                body.filter_lam_um = fentry["lam_um"]
                body.filter_alpha_per_m = fentry["alpha_per_m"]
            self.bodies.append(body)

            if body.role == "source":
                self.sources.append((body.index, rec["source"]))
                # source bodies contribute no intersectable geometry (the
                # housing is not traced), but the emitting face itself is
                # built for area sampling and kept OUT of self.faces
                emit_name = rec["source"]["emit_face"]
                emit_rec = next((f for f in rec["faces"]
                                 if f["id"] == emit_name), None)
                if emit_rec is None:
                    raise ValueError("source %s: emit face %r not among "
                                     "its faces" % (body.label, emit_name))
                if emit_rec["surface"]["type"] == "mesh":
                    raise NotImplementedError(
                        "source %s: emitting face is mesh-type" % body.label)
                self.emit_faces[body.index] = AnalyticFace(
                    emit_rec["id"], make_surface(emit_rec["surface"]),
                    emit_rec["trim_polylines_xyz"],
                    emit_rec["orientation_outward"], body.index, -1,
                    area_m2=emit_rec["area_m2"])
                continue

            for f in rec["faces"]:
                if f["surface"]["type"] == "mesh":
                    if strict_analytic:
                        raise NotImplementedError(
                            "face %s is mesh-type (non-analytic) and "
                            "--strict-analytic is set" % f["id"])
                    if body.role == "detector" \
                            and f["id"] == rec["detector"]["face"]:
                        raise NotImplementedError(
                            "detector screen face %s is mesh-type — "
                            "detector grids need an analytic plane"
                            % f["id"])
                    stl = f.get("mesh_stl") or ""
                    path = None
                    if geometry_dir is not None and stl:
                        from pathlib import Path
                        path = Path(geometry_dir) / stl
                    if path is None or not path.exists():
                        raise ValueError(
                            "mesh face %s: STL %r not found — pass "
                            "geometry_dir (the model.json's directory)"
                            % (f["id"], stl))
                    from .mesh import MeshFace
                    face = MeshFace(f, path,
                                    flat_normals=mesh_flat_normals)
                else:
                    face = AnalyticFace(
                        f["id"], make_surface(f["surface"]),
                        f["trim_polylines_xyz"], f["orientation_outward"],
                        body.index, len(self.faces), area_m2=f["area_m2"])
                fid = len(self.faces)
                self.faces.append(face)
                self.face_body.append(body.index)
                self.face_records.append(f)
                self.face_by_name[f["id"]] = fid
                body.face_ids.append(fid)

            if body.role == "detector":
                det_face = rec["detector"]["face"]
                if det_face not in self.face_by_name:
                    raise ValueError("detector face %r not found on body %s"
                                     % (det_face, body.label))
                self.detector_faces[self.face_by_name[det_face]] = body.index

        if not self.sources:
            raise ValueError("no active sources (all suppressed?)")
        if not self.detector_faces:
            raise ValueError("no active detectors (all suppressed?)")

        # extra CLI detector faces: transparent zero-effect screens on any
        # existing face (including optical-element faces)
        self.extra_detector_faces = set()
        for spec in extra_detector_faces:
            fid = self._face_id_or_die(spec, "detector")
            self.extra_detector_faces.add(fid)

        # ---- per-face physics options ---------------------------------
        # precedence everywhere: CLI spec > body per-face entry > body
        # whole-body value.
        import common as _common
        self.gratings = {}
        # body 'grating' property (per-face dict, values from
        # common.parse_grating_value)
        for body in self.bodies:
            if body.grating_map:
                for face_name, value in body.grating_map.items():
                    spec = _common.parse_grating_value(value)
                    fid = self._face_id_or_die(face_name, "grating")
                    spec["face"] = {"id": face_name}
                    self.gratings[fid] = spec
        # CLI --grating overrides
        for g in grating_specs:
            self.gratings[self._face_id_or_die(g["face"]["id"],
                                               "grating")] = g
        # resolve @registry refs against opticalproperties/grating/
        for fid, spec in self.gratings.items():
            reg = spec.get("registry")
            if reg is None:
                continue
            if reg not in grating_registry:
                raise ValueError(
                    "grating on %s: unknown registry entry %r (gratings.csv "
                    "has: %s)" % (self.faces[fid].id, reg,
                                  ", ".join(sorted(grating_registry)) or
                                  "<none loaded — pass optprops>"))
            entry = grating_registry[reg]
            spec["model"] = entry["model"]
            spec["lines_per_mm"] = entry["lines_per_mm"]
            spec["params"] = entry["params"]
            spec["table"] = entry["table"]
        for fid in self.gratings:
            if self.faces[fid].surface is None:
                raise ValueError(
                    "grating on mesh face %s: gratings need an analytic "
                    "surface (groove/UV geometry)" % self.faces[fid].id)

        self.roughness = {}
        # body per-face roughness strings, then whole-body float
        for body in self.bodies:
            if body.roughness_faces:
                for face_name, value in body.roughness_faces.items():
                    rv = _common.parse_rough_value(value)
                    if face_name == FACEMAP_ALL:
                        for fid in body.face_ids:
                            self.roughness[fid] = dict(
                                rv, face={"id": self.faces[fid].id})
                    else:
                        fid = self._face_id_or_die(face_name, "roughness")
                        self.roughness[fid] = dict(rv,
                                                   face={"id": face_name})
            if body.roughness_nm > 0:
                for fid in body.face_ids:
                    self.roughness.setdefault(
                        fid, {"face": {"id": self.faces[fid].id},
                              "sigma_nm": body.roughness_nm,
                              "lcorr_um": 10.0})
        # CLI --rough overrides everything
        for r in rough_specs:
            self.roughness[self._face_id_or_die(r["face"]["id"],
                                                "roughness")] = r

        # ---- ground-glass diffusers: the deep-rough limit of the same
        # microfacet model. Each diffuser face resolves to an RMS slope
        # (grit table / explicit / registry) and lands in self.roughness
        # as a sigma>>lambda entry (specular retention exactly 0, every
        # ray scattered through one Beckmann facet with full
        # per-polarization Fresnel). A face carrying BOTH diffuser and
        # roughness is a contract error, not a merge.
        from .roughness import diffuser_equivalent, slope_for_grit
        for body in self.bodies:
            if not body.diffuser_faces:
                continue
            for face_name, value in body.diffuser_faces.items():
                spec = _common.parse_diffuser_value(value)
                if "registry" in spec:
                    reg = (self.optprops.diffusers
                           if self.optprops is not None else {})
                    entry = reg.get(spec["registry"])
                    if entry is None:
                        raise ValueError(
                            "body %s: unknown diffuser registry entry %r "
                            "(opticalproperties/diffuser/diffusers.miedif)"
                            % (body.label, spec["registry"]))
                    slope = entry["slope_rms"]
                elif "grit" in spec:
                    slope = slope_for_grit(spec["grit"])
                else:
                    slope = spec["slope"]
                sigma_nm, lcorr_um = diffuser_equivalent(slope)
                if face_name == FACEMAP_ALL:
                    fids = list(body.face_ids)
                else:
                    fids = [self._face_id_or_die(face_name, "diffuser")]
                for fid in fids:
                    if fid in self.roughness:
                        raise ValueError(
                            "body %s: face %s carries BOTH a diffuser and "
                            "a roughness declaration — they are "
                            "alternative models of one surface, pick one"
                            % (body.label, self.faces[fid].id))
                    self.roughness[fid] = {
                        "face": {"id": self.faces[fid].id},
                        "sigma_nm": sigma_nm, "lcorr_um": lcorr_um,
                        "diffuser": True}

        # ---- measured-scatter (ABg/BSDF) faces: reflected-side lobe from
        # a registry entry (opticalproperties/scatter/). Resolves to
        # self.scatter = {fid: entry}. A face carrying scatter AND roughness
        # OR diffuser is a contract error — they are alternative surface
        # models (the roughness map already holds any diffuser entries).
        self.scatter = {}
        scatter_registry = (self.optprops.scatter
                            if self.optprops is not None else {})
        for body in self.bodies:
            if not body.scatter_faces:
                continue
            for face_name, name in body.scatter_faces.items():
                if name not in scatter_registry:
                    raise ValueError(
                        "body %s: unknown scatter entry %r "
                        "(opticalproperties/scatter/bsdf.miebsdf has: %s)"
                        % (body.label, name,
                           ", ".join(sorted(scatter_registry)) or
                           "<none loaded — pass optprops>"))
                if face_name == FACEMAP_ALL:
                    fids = list(body.face_ids)
                else:
                    fids = [self._face_id_or_die(face_name, "scatter")]
                for fid in fids:
                    if fid in self.roughness:
                        raise ValueError(
                            "body %s: face %s carries BOTH a scatter and a "
                            "roughness/diffuser declaration — they are "
                            "alternative models of one surface, pick one"
                            % (body.label, self.faces[fid].id))
                    self.scatter[fid] = scatter_registry[name]

        # per-face coating map: {int fid: coating name}
        self.face_coatings = {}
        for body in self.bodies:
            if not body.coating:
                continue
            default = body.coating.get(FACEMAP_ALL)
            if default is not None:
                for fid in body.face_ids:
                    self.face_coatings[fid] = default
            for face_name, cname in body.coating.items():
                if face_name == FACEMAP_ALL:
                    continue
                self.face_coatings[self._face_id_or_die(
                    face_name, "coating")] = cname

        self.face_body = np.asarray(self.face_body, dtype=np.int32)

    def _face_id_or_die(self, face_name, kind):
        if face_name not in self.face_by_name:
            raise ValueError(
                "%s face %r not found in scene. Available faces:\n  %s"
                % (kind, face_name,
                   "\n  ".join(sorted(self.face_by_name))))
        return self.face_by_name[face_name]

    # ------------------------------------------------------------------
    def source_bodies(self):
        return [(self.bodies[i], src) for i, src in self.sources]

    def body_of_face(self, fid):
        return self.bodies[self.face_body[fid]]

    def medium_index(self, body_index, lam):
        """Complex n for rays inside body body_index (-1 = ambient).

        Birefringent bodies return the ORDINARY index n_o: o-rays and any
        non-mode-tagged path (e.g. a grating on a crystal face) use it;
        e-rays override the real part via RayBatch.n_eff, and absorption
        uses Im(n_o) for both modes (documented approximation)."""
        if body_index < 0:
            return self.ambient.n_complex(lam)
        body = self.bodies[body_index]
        if body.role == "detector" or body.material in (None, "detector"):
            # detector solids are ideal thin screens; treat interior as
            # ambient (rays never legitimately travel "inside" them)
            return self.ambient.n_complex(lam)
        if body.birefringent:
            return self.matdb.get_uniaxial(body.material)[0].n_complex(lam)
        if body.biaxial:
            # scalar bookkeeping index (medium stack / seam accounting):
            # the geometric mean keeps it sheet-neutral
            mx, my, mz = self.matdb.get_biaxial(body.material)
            return (np.real(mx.n_complex(lam))
                    * np.real(my.n_complex(lam))
                    * np.real(mz.n_complex(lam))) ** (1.0 / 3.0)
        return self.matdb.get(body.material).n_complex(lam)

    def uniaxial_indices(self, body, lam):
        """(n_o_real, n_e_real) arrays for a birefringent body at lam [m]."""
        mo, me = self.matdb.get_uniaxial(body.material)
        return (np.real(mo.n_complex(lam)), np.real(me.n_complex(lam)))

    def biaxial_eps(self, body, lam):
        """(n,3) principal permittivities (n_x^2, n_y^2, n_z^2), real, for
        a biaxial body at lam [m] (per-ray arrays for dispersion)."""
        mx, my, mz = self.matdb.get_biaxial(body.material)
        return np.stack([np.real(m.n_complex(lam)) ** 2
                         for m in (mx, my, mz)], axis=-1)

    # ------------------------------------------------------------------
    def intersect(self, pos, direction):
        """Nearest hit across all faces.

        Self-intersection is handled by the faces' t_eps guard (100 nm),
        NOT by excluding the last-hit face — a ray reflected internally in
        a sphere legitimately re-hits the same face.
        Returns (t (N,), face_id (N,) int32 with -1 = miss).
        """
        n = len(pos)
        best_t = np.full(n, np.inf)
        best_f = np.full(n, -1, dtype=np.int32)
        for fid, face in enumerate(self.faces):
            t, hit = face.intersect(pos, direction)
            better = hit & (t < best_t)
            best_t[better] = t[better]
            best_f[better] = fid
        return best_t, best_f
