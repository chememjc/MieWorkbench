# engine.md — BREP-native C++/OptiX engine: design, plan, and verdict

> ## ⚠ SUPERSEDED by `engine2.md` (2026-07-15)
>
> **Do not act on this document.** It answered the wrong question. Kept for its
> research record — the OptiX/OCCT/MDL/prior-art findings in §2, §6, §7 and the
> References are sound and were reused.
>
> What it got wrong:
>
> - **§1.2(b) — the "~3.3× Amdahl ceiling" is an artifact of benchmarking at 1e6
>   rays**, where runs are 1.5–45 s and the fixed ~2 s Python driver is 4–30 % of
>   wall. At production scale the driver is noise. The method was right; the anchor
>   point was wrong.
> - **§11.3 "Amdahl honesty" is wrong.** It claimed the gather-moving options carry
>   "new physics approximation risk and belong in their own round." In fact ~20–60×
>   is available from a precision fix plus an error-controlled local expansion that is
>   **bit-exact at tile=1**, with zero change to the physics model. The gather is not
>   a later round — it is the main event, and the only lever that touches the
>   multi-hour scenes.
> - **The whole BREP premise dissolves** under prescription-primary optics +
>   CAD-primary mechanical: nothing needing phase is ever meshed, and nothing meshed
>   ever needs phase. OCCT survives host-side only, for import/fitting.
> - **The stated UI motive was a ~10-line VTK bug** (`vtkview.py:752-754`
>   `EdgeVisibilityOn()`), not a geometry-representation problem. Refining the mesh
>   makes it ~56× worse.
>
> §1.5 (a NURBS cannot exactly represent an asphere; the prescription is more exact
> than the CAD) **still holds** — it just argues for deleting BREP rather than
> structuring it.

Status: **design document**. No code exists. Superseded by `engine2.md`.
Audience: engine maintainers. Companion to `docs/RAYTRACER.md` (physics
reference) and `cengine/README.md` (current C engine).

---

## 1. Scope & verdict

### 1.1 What is proposed

A third engine, `brepengine/`, selected by `--engine brep`, that traces the
**OpenCASCADE BREP directly** instead of the derived geometry
`extract_geometry.py` emits today. Exact trims from real pcurves, analytic
faces as closed-form primitives, freeform faces as rational Bézier patches,
OptiX traversal on the RTX 4090, and direct ingestion of mechanical CAD for
stray-light work.

### 1.2 The honest verdict, stated first

**Do this for capability, not for speed.** The speed case is much weaker than
it looks, and the measured evidence is partly against the architecture in the
original brief.

Three findings drive this:

**(a) The literature says direct patch casting loses.** Schollmeyer & Fröhlich
(2009) measured, on a 53,554-Bézier-patch model, tessellate-then-bisect at
**14.2 ms** vs direct ray casting at **93.6 ms** — direct is **6.6× slower**.
Sloup & Havran (2021), the state of the art at 49–499 MRays/s, deliberately
used raw CUDA, **not** RTX/OptiX. Zemax OpticStudio — which embeds a real ACIS
kernel and therefore *could* trace BREP exactly — does not, by default. Its
documentation, verbatim: *"rays are traced to the faceted model, then an
iterative method is used to converge to the true intersection to the underlying
NURBS geometry."* Its exact-kernel mode is *"extremely slow… can serve as a
reference."*

**(b) The tracer is not the bottleneck in half the scene catalog.** From
`cengine/BENCHMARKS.md`, decomposing the C engine's own wall time (1e6 rays,
2048², 9λ, RTX 4090 Laptop):

| scene | C wall | C trace | C gather | residual¹ | ceiling if trace→0² |
|---|---|---|---|---|---|
| camera_triplet | 25.1 s | 23.25 s | 0.01 s | 1.84 s | **13.6×** |
| microscope_objective | 45.5 s | 41.16 s | 1.91 s | 2.43 s | **10.5×** |
| ghost_doublet | 10.2 s | 7.98 s | 0.01 s | 2.21 s | 4.6× |
| beam_expander | 7.0 s | 4.83 s | 0.06 s | 2.11 s | 3.2× |
| dobsonian | 2.9 s | 1.86 s | 0.01 s | 1.03 s | 2.8× |
| fiber_coupler | 6.8 s | 4.18 s | 0.01 s | 2.61 s | 2.6× |
| czerny_turner | 1.5 s | 0.90 s | 0.02 s | 0.58 s | 2.5× |
| schmidt_cassegrain | 10.2 s | 5.96 s | 1.91 s | 2.33 s | 2.4× |
| prism_spectrometer | 9.4 s | 5.29 s | 1.91 s | 2.20 s | 2.3× |
| scatter_plate | 7.2 s | 3.28 s | 1.92 s | 2.00 s | 1.8× |
| newtonian | 4.4 s | 1.39 s | 1.94 s | 1.07 s | 1.5× |
| michelson_folded³ | 716.8 s | 1.60 s | 713.42 s | 1.78 s | **1.002×** |

¹ `wall − trace_s − gather_s`. ² `wall / (wall − trace_s)` — the Amdahl bound
on an *infinitely fast tracer*. ³ measured at 2e5 rays; the 1e6 run exceeds the
5400 s budget.

**Geometric mean ceiling over the 11 benchmark scenes: ~3.3×.** That is the
*maximum* an infinitely fast tracer can deliver over today's C engine. A
realistic OptiX tracer lands well below it.

**(c) The residual is the Python driver, not the engine.** `bench_engines.py:45-58`
times the whole `run_trace.py` subprocess. The ~1–2.6 s floor present in *every*
scene is the optics-env interpreter start plus numpy/scipy/torch-CUDA import
plus scene build — paid even when the C binary does the work, because
`run_trace.py` is the driver that invokes it.

### 1.3 What this actually decomposes into

There are **three independent cost centers**, and BREP/OptiX attacks exactly one:

| cost center | magnitude | attacked by |
|---|---|---|
| Python driver overhead | ~1–2.6 s fixed/run | a **standalone `brepengine` binary** that reads `scene.json` + `.brep` with no Python in the trace path |
| trace | 0.2 %–93 % of wall, scene-dependent | **BREP/OptiX** (this document) |
| coherent gather | up to 99.5 % of wall (michelson) | **gather rework** (§11) — orthogonal to BREP |

Chasing the tracer alone, in a michelson-class scene, buys **1.002×**. This is
the single most important number in this document.

### 1.4 Which of the four stated motives BREP actually serves

| motive | verdict |
|---|---|
| **Real-world CAD ingestion** (stray light on as-designed hardware) | **Yes — the strongest case.** Nothing else delivers it. Today a mechanical assembly must be meshed, and mesh faces are incoherent-only by contract. |
| **Kill the tessellation contract** | **Yes, in the sense that matters.** The contract worth killing is *mesh-as-geometry* (`RAYTRACER.md §6.2`: "sag ≫ λ, coherent phase meaningless"). A proxy triangle used *only* to seed Newton onto the exact patch is not that: the hit point, normal, and OPL all come from the exact surface. Phase is valid. |
| **Phase-exact arbitrary surfaces** | **Partly — and it makes prescribed optics *worse*, not better.** See §1.5. |
| **Speed** | **Weakest case.** Ceiling ~3.3× geomean over cengine; ~1× on gather-bound scenes. Much of the remaining headroom is in the driver and the gather, neither of which is a BREP problem. |

### 1.5 The asphere inversion — why analytic must win

A NURBS surface **cannot exactly represent an even-order polynomial asphere**.
Only conics of revolution are rationally representable. The current contract
therefore authors an asphere as a BSpline through exact sag points *plus* a
`surface_override` declaring the true prescription, and
`extract_geometry.build_asphere_surface` verifies the fit to
`ASPHERE_TOL_M = 1e-6` (`extract_geometry.py:133`).

**1 µm ≈ 2 waves at 500 nm.** The CAD BSpline is therefore *not* phase-valid for
aspheres. The `surface_override` prescription is the exact geometry; the BREP is
the approximation.

This generalizes: **for any prescribed optic, the analytic form is more exact
than its BREP representation.** "Trace what's actually modeled" is the *less*
accurate choice for designed optics. Hence the load-time authority order in §3.3
— analytic first, BREP as fallback — which is a **correctness** decision that
happens to also be faster.

### 1.6 Go/no-go

**Go, with the scope reordered.** Build it for CAD ingestion and true-freeform
optics. Do not sell it as a speed project. Two cheaper, orthogonal wins should be
taken regardless of whether this engine is ever built:

1. **A standalone binary** with no Python in the trace path (~1–2.6 s/run).
2. **The gather rework** (§11) — the only thing that touches michelson-class scenes.

And one finding that should be acted on immediately, independent of everything
else: **the Python engine has no scene BVH** (`scene.py:910-926` is a linear scan
over every face for every batch). A large fraction of cengine's measured 8.3× is
attributable to acceleration structure, not to C. Any claim that "BREP+OptiX gave
us N×" must subtract that.

---

## 2. Corrections to the premise

The original brief contained six assertions that research contradicted. They are
recorded here so the design is not re-litigated from the same wrong priors.

| brief said | actual |
|---|---|
| Use NVIDIA MDL SDK | **Unusable.** The MDL spec contains **zero** mentions of polarization. No Jones vectors, no Mueller matrices. `df::thin_film` is *single-layer* Airy returning `0.5*(R_s+R_p)` — the unpolarized average — with **phase discarded**, and normal-form rules forbid nesting, so it cannot express a multilayer stack. `material_geometry` carries only displacement/cutout/normal — no surfaces, no intersection. MDL is an authoring layer for RGB *plausible* rendering; every quantity this engine needs is precisely what MDL discards. **Dropped.** The `.miemat`/`.miecoat`/`.mienk` loaders remain the material authority. |
| Embree + OptiX | **OptiX only.** Two traversal backends = two intersection code paths for no gain on a machine with a 4090. |
| "Straight C where possible for maximum speed" | **No measurable effect.** Modern compilers generate identical code for equivalent C and C++. OCCT is C++-only; OptiX device code is C++. The C-vs-C++ axis is not a performance decision. `cengine/`'s existing header-only kernels are already `static inline` C that compiles as CUDA device code nearly verbatim — *that* is the real asset. |
| Subdivide all patches, custom-prim IS everywhere | **Analytic faces get closed-form custom prims; only freeform gets patches, via tessellate-and-refine.** See §1.2(a), §1.5. |
| OptiX displaced micro-meshes | **Removed.** OptiX 9.0 release notes: *"The DMM API is now deprecated and is being replaced by the OptiX Clusters API. The DMM API and samples have been removed from the SDK."* Clusters accelerate *triangle* BVH builds and do not help a BREP tracer. **There is no built-in parametric-patch/NURBS/Bézier-surface primitive in OptiX.** |
| 2D interval tree for trim | **2D kd-tree over bi-monotonic segments** (Sloup & Havran 2021). See §4.3. |

Two further corrections, to *this repo's own documentation*:

- **`CLAUDE.md` and `RAYTRACER.md` overstate the parity bar.** They claim
  "1e-12 deterministic / 3-seed ±max(3σ, 1%)". `test_cengine_parity.py:101`
  actually enforces `tol = 1e-9 if klass == "deterministic" else 0.02`. 1e-12
  applies only to `emitted_W` and as the deterministic absolute floor. **No
  3-seed/3σ harness exists** — statistical scenes use a flat 2 % relative bar on
  one seed.
- **There is no Philox in the Python engine.** `tracer.py:166` is plain
  `np.random.default_rng(config.seed)`. Lineage-keyed Philox is C-engine-only
  (`cengine/src/rng.h`). Python↔C agreement is **statistical only**, and always
  has been.
- **`CLAUDE.md` says "eleven classic-system demos".** There are **34**
  `demos/*.MieWB`, 15 with committed baselines in `demos/baselines/`. The
  benchmark set is 11; the demo set is not.

### 2.1 Unresolved — flagged, not asserted

Two independent research passes **disagree** on OCCT's LGPL-2.1 exception. One
reads it as covering only header material incorporated into object code (so
§6's relink obligation still binds the library, implying dynamic linking). The
other cites SPDX `OCCT-exception-1.0` as permitting static linking into
closed-source. This repository is not closed-source, so the question is likely
moot here. **Recommendation: dynamic linking**, as the safe default, pending a
reading of `OCCT_LGPL_EXCEPTION.txt` in the vendored tree.

---

## 3. Geometry pipeline (load time)

### 3.1 Ingest

The engine links OCCT directly. FreeCAD supplies **tags only**.

```
.FCStd (zip)
├── Document.xml            → FreeCAD/fcserver → scene.json  (roles, tags, placements, spreadsheets)
└── <shape>.brp             → copied out       → brep/<body>.brep
                                                      │
                                          brepengine (links OCCT 8.0.0_p1)
```

`extract_geometry.py`'s tessellation, per-face STL emission, area tripwire
(`AREA_TOL_REL`), and `trim_polylines_xyz` discretization **do not run** on this
path. Its *classification* logic (`classify_body`, role assignment, tag parsing,
`surface_override` parsing) is retained and moves into the `scene.json` producer.

### 3.2 Threading hazard — mandatory

OCCT caches BSpline polynomial coefficients **lazily, mutating on read**.
`D0`/`D1`/`EvalD0` on a `Geom_BSplineSurface` looks const and is not thread-safe.

What 8.0 **did** fix (verified against the upgrade guide and V8_0_0 release notes):
`Standard_Mutex` → `std::mutex`; `TopTools_MutexForShapeProvider` removed; several
`BRepCheck_*`/Foundation internals converted to `thread_local`; the exception/error
handler stack is now `thread_local`; poles/weights/knots moved from
`NCollection_HArray*` handles to direct value-member arrays, which structurally
removes some sharing hazards. **8.0.0_p1** additionally fixed a `Standard_Strtod`
shared-Bigint-pool race that hit **parallel BRep reads** — the reason the pin is
p1, not 8.0.0.

What 8.0 **did not** do: **nowhere is concurrent read-only `Geom_*` evaluation
documented as safe, and the eval-cache mutation-on-read hazard is never stated as
eliminated.** Research could find no such guarantee in the guide or the release
notes. Treat it as **unresolved and undocumented**, not fixed.

**Rule:** every worker thread gets its own `BRepAdaptor_Surface` / geometry handle
copy, or the eval cache is warmed single-threaded before fan-out. Never share an
adaptor. **Verify empirically with ThreadSanitizer during P1** rather than trusting
either the docs or this note.

### 3.3 Face classification — authority order

Per face, in order; first match wins:

1. **`surface_override` present** → exact analytic prim from the *declared
   prescription*. The BREP surface is used only to verify the fit and to source
   the trim. (§1.5 — this is the accurate path, not a legacy hack.)
2. **`BRepAdaptor_Surface::GetType()` ∈ {`GeomAbs_Plane`, `GeomAbs_Cylinder`,
   `GeomAbs_Cone`, `GeomAbs_Sphere`, `GeomAbs_Torus`}** → exact analytic prim
   from OCCT's own canonical params.
3. **`GeomAbs_SurfaceOfRevolution`** → attempt canonicalization to sphere/cylinder
   (port `extract_geometry.canonicalize_revolution:518`). On success → analytic.
4. **Otherwise** (`GeomAbs_BSplineSurface`, `GeomAbs_BezierSurface`,
   `GeomAbs_SurfaceOfExtrusion`, `GeomAbs_OffsetSurface`, `GeomAbs_OtherSurface`)
   → `GeomConvert_BSplineSurfaceToBezierSurface` → rational Bézier patch grid →
   subdivision (§3.4).

The `GeomAbs_SurfaceType` enum is **unchanged in 8.0** — 11 values, verified
against the 8.0.0 refman: `Plane, Cylinder, Cone, Sphere, Torus, BezierSurface,
BSplineSurface, SurfaceOfRevolution, SurfaceOfExtrusion, OffsetSurface,
OtherSurface`. Note it is `SurfaceOfExtrusion`, **not** `SurfaceOfLinearExtrusion`
(that is the `Geom_` class name, not the enum tag).

`--strict-analytic` (existing flag semantics) makes case 4 a hard error.

### 3.4 Subdivision

Recursive, at load, until **both** hold per sub-patch:

- **Flatness**: `max_i dist(P_i, Π)` over interior control points, where `Π` is
  the plane through the four corner control points, below `ε_flat`.
- **Size**: the sub-patch AABB diagonal is small relative to the scene AABB
  diagonal, below `ε_size`.

Both thresholds are **correctness parameters, not speed knobs** — see §7.4
(wrong-root convergence). They are recorded in `case.json` and surfaced in the
anomaly block.

Rational weights are preserved (`Geom_BezierSurface` is rational). Bicubic
sub-patches store 16 control points as `float4` (xyzw = x·w, y·w, z·w, w).

**Extraction API (8.0-verified).** `Geom_BezierSurface`/`Geom_BSplineSurface` gained
**zero-copy** accessors alongside the 7.x copy-out forms:

- `const NCollection_Array2<gp_Pnt>& Poles() const` — new, zero-copy. The
  deprecated `void Poles(NCollection_Array2<gp_Pnt>&)` copy-out form still exists.
- `const NCollection_Array2<double>* Weights() const` — **nullable**, returns
  `nullptr` for non-rational surfaces (unchanged from 7.x), **or**
  `const NCollection_Array2<double>& WeightsArray() const` — new, always valid,
  returns a static unit-weight view for non-rational surfaces with zero alloc.

Use `WeightsArray()`: it removes the non-rational null branch from the extraction
path entirely. **There is no `PolesAndWeights()` combined accessor** in either 7.x
or 8.0 — do not plan around one.

For the host-side verifier and the `surface_override` fit check (which *evaluate*
rather than extract), 8.0 adds virtual `EvalD0`/`EvalD1`/`EvalD2`/`EvalD3`/`EvalDN`
across ~32 Geom/Geom2d leaf classes, returning POD structs named
`Geom_Surface::ResD1`/`ResD2`/`ResD3` (the upgrade guide's "Geom_SurfD1/D2/D3"
prose is a loose paraphrase; the nested `Res*` names are what the refman lists).
The 7.x out-param `D0(U,V,gp_Pnt&)` forms are **retained** as non-virtual inline
wrappers, so either style compiles.

Expected yield: typical CAD NURBS faces give tens to a few thousand sub-patches.
Scene ceiling ~10⁶ patches ⇒ ~256 MB of control points at 16×`float4`. Comfortable
in 16 GB.

### 3.5 Degenerate geometry — role-based policy

Real `.FCStd` and STEP files contain zero-area faces and sliver trims from bad
booleans. Policy keys off the **existing role classification**:

| body role | policy |
|---|---|
| `ignored` / mechanical | **filter + loud warn.** Face dropped, counted in `degenerate_faces_filtered`, logged with face id and reason. |
| `optic` / `source` / `detector` | **hard error.** A sliver in an optical path is a physics bug, not noise. Named face, named defect, non-zero exit. |

Thresholds: area `< 1e-12 m²`; aspect ratio `> 1e4`; trim loop shorter than the
subdivision tolerance. `--strict-geometry` promotes the mechanical case to a hard
error (mirroring the existing `--strict-analytic`).

This is the one place the engine is *deliberately permissive*, because the
stray-light motive (§1.4) requires ingesting CAD the optics contract would reject.

### 3.6 Instancing

Reuse the **existing placement-independent `shape_key`** computed by
`fcops`/`geomcache.py`, extended to hash the `.brep` bytes. Same key ⇒ one BLAS,
N instance transforms in the TLAS. 400 identical bolts hold one bolt BLAS.

`CLAUDE.md`'s quantization contract for shape fingerprints applies unchanged
(0.1 µm CoM, `-0.0` folded, **absolute** not relative formatting — `%g`-style
relative formatting turns 1e-15 recompute noise into spurious invalidations).

---

## 4. Trim preprocessor

The single largest correctness gain in this document. Today's trim is a
**0.05 mm-chord UV polygon** (`extract_geometry.py:128`, `:767-816`). Exact trim
replaces it.

### 4.1 Extraction

- `BRepTools::OuterWire(F)` for the outer loop; remaining wires are holes.
- **`BRepTools_WireExplorer`** — *ordered*. `TopExp_Explorer` is **not** ordered
  and must not be used here.
- Per edge: `BRep_Tool::CurveOnSurface(E, F, first, last)` → `Handle(Geom2d_Curve)`.
- `GeomConvert_BSplineCurveToBezierCurve` → rational Bézier segments in UV.
- `TopAbs_REVERSED` on the face flips both the normal and the trim in/out sense.
  Honor it explicitly.

The head-to-tail orientation correction that
`extract_geometry.trim_polylines_xyz:802-816` had to hand-roll (because
`OrderedEdges` orders edges but does not flip reversed edges' point sequences —
the bug that killed half of every pad face) is **structurally absent** here:
`BRepTools_WireExplorer` yields edges in order *with* orientation, and pcurves are
taken in that orientation.

### 4.2 Seams — explicit, not incidental

A periodic surface's seam edge appears **twice** in the wire (FORWARD and
REVERSED) with **two distinct pcurves on the same face**.

- Detect: `Geom_Surface::IsUPeriodic()`/`IsVPeriodic()`, `BRep_Tool::IsClosed(E, F)`.
- Fetch both pcurves via the **indexed** overload
  `BRep_Tool::CurveOnSurface(E, C2d&, S&, L&, First&, Last&, const int Index)`
  called with `Index = 1` and `Index = 2`. **There is no simultaneous `(C1, C2)`
  two-out-param overload** — the 8.0.0 refman lists exactly four `CurveOnSurface`
  overloads and none has that shape. (`ShapeAnalysis_Edge::PCurve(..., orient)` is
  the alternative.)
- Unwrap the domain at load so no segment straddles the period.
- The crossing test must **neither double-count the seam nor leak through it**.
  Seam hits are counted in the anomaly block (`seam_hits`) — the existing
  `seam_loss` ledger bucket exists precisely because this goes wrong.

### 4.3 Device-side point location — 2D kd-tree

Following Sloup & Havran (2021):

1. Split each Bézier pcurve segment at its `u` and `v` extrema into
   **bi-monotonic** pieces (Schollmeyer's construction). Monotonicity makes the
   crossing test exact and branch-light.
2. Build a **2D kd-tree** over the segments in UV.
3. Query: descend to the leaf containing `(u,v)`, count crossings along a UV ray
   against the candidate segments, apply even-odd with the outer/hole sense.

Chosen over the brief's interval tree because it is the published, measured
approach (49–499 MRays/s) and degrades better on faces with many long segments.
Schollmeyer's own data shows trimming is **cheap** — 14.2 ms total including
tessellation; the ray casting is what costs. The trim structure is not the risk.

---

## 5. Device data model

Everything the intersection program touches is a **flat, indexed device buffer
built at load**. No pointers. No per-ray allocation.

| buffer | contents |
|---|---|
| `surfaces[]` | tagged union: plane{origin,normal} · sphere{center,R} · cylinder{origin,axis,R} · cone{apex,axis,half_angle} · torus{center,axis,R_maj,R_min} · asphere{vertex,axis,R,k,coeffs[],r_max} — mirroring `extract_geometry.classify_surface:598` schemas |
| `patches[]` | 16×`float4` control points (rational, w-premultiplied) · uv-domain rect · face index · trim offset |
| `trim_segments[]` | bi-monotonic 2D Bézier segments, per-face contiguous |
| `trim_nodes[]` | 2D kd-tree nodes, per-face contiguous, indexed by `face.trim_offset` |
| `faces[]` | surface idx · trim offset/count · `outward_sign` · body idx · material binding idx · role |
| `materials[]` | per-λ pre-resolved n/k, coating stack refs, filter α — **pre-resolved at fixed stratum λ**, exactly as `cengine`'s request protocol does today |
| `instances[]` | BLAS ref · 3×4 transform · body idx |

`orientation_outward` retains its current contract semantics: it describes the
**canonical normal derived from stored analytic params**, *not* OCCT's
`normalAt()`. `CLAUDE.md` flags this explicitly; do not "simplify" it.

---

## 6. Acceleration structure

Two-level, per the brief:

- **TLAS** over solid instances (`OptixInstance`, 3×4 transforms, `shape_key`-deduped BLASes).
- **BLAS** per solid, over primitives, where a primitive is either:
  - an **analytic face** — `OptixBuildInputCustomPrimitives`, one AABB per face;
  - a **freeform sub-patch** — proxy triangles (§7.3).

### 6.1 Tight analytic bounds

A trimmed cylinder face is bounded by **its swept trim loops, not the full
cylinder**. Compute from the trim curves' UV bounding boxes mapped through the
surface. Same for cone/torus/sphere zones. This matters: a loose bound on a large
cylinder makes every ray in the scene a candidate.

Analytic AABBs are **conservative by construction** — they contain the true
trimmed patch, so they cannot cause a miss.

### 6.2 Why analytic faces get custom prims and freeform does not

Custom prims get hardware AABB traversal but their IS program runs **in software
on the SM**, breaking the hardware traversal loop. Measured penalty for analytic
intersectors vs ray-triangle: **~2×**.

For analytic faces this is the right trade anyway: the IS is a closed-form root
solve (a quadratic, or `quartic.h` for the torus), it is cheap, it is exact, and
it needs no proxy — so there is no miss risk and no seed-density correctness
parameter. For freeform, the IS would be an fp64 Newton solve, which is exactly
where the software-IS penalty compounds and where §1.2(a)'s 6.6× lives.

> **A skeptical note on a number you will encounter.** A widely-cited result
> claims custom AABBs are "up to 4× faster" than built-ins ([arXiv 2408.14247]).
> It is measured on an **A100, which has no RT cores at all**. It is irrelevant
> to any RTX question. Do not let it into a design discussion.

---

## 7. Intersection

### 7.1 Analytic — closed form

Lift `cengine/src/kernels/surf.h` (21 KB, all six surface types) and
`quartic.h` (torus). These are `static inline` C, already parity-tested against
the Python reference to 1e-9, and compile as CUDA device code nearly verbatim.

Retain `t_eps = 1e-7` (100 nm) as the self-hit guard, and retain the contract
that there is **no last-face exclusion** — a ray reflected internally in a sphere
legitimately re-hits the same face (`scene.py:910-916`).

### 7.2 Trim test

Hit → surface-specific `(u,v)` → kd-tree point location (§4.3) → even-odd.
`trim.h` provides the current canonical-UV regimes (untrimmed / band / polygon);
the band regime's periodic-winding logic informs §4.2 but the polygon regime is
**replaced** by the exact test.

### 7.3 Freeform — proxy-seeded Newton

1. Each Bézier sub-patch is tessellated to a small triangle fan (2–8 tris),
   built into the BLAS as **built-in triangles** → RT-core hardware traversal.
2. Closest-hit takes the barycentric hit and the triangle's `(u,v)` corners → an
   initial `(u,v,t)` seed.
3. **2×2 Newton** on `S(u,v) − (O + tD) = 0`, in **fp64**, to convergence.
4. Report the **exact** hit point, normal, and `t` from the rational patch. The
   triangle is discarded. **The mesh is never the geometry.**

This is Martin et al. (2000) and it is what Zemax ships (§1.2(a)).

### 7.4 Failure modes and mitigations

| failure | detection | mitigation |
|---|---|---|
| **Tangential contact** (ray grazes a cylinder) — Newton converges slowly or oscillates | near-zero determinant in the 2×2 Jacobian solve | fall back to **interval bisection on `t`**; count `tangential_fallback` |
| **Ray ∥ plane at tolerance distance** | `\|D·n\| < ε` | **edge-tube capture** — explicit tube geometry along the trim edge, *not* epsilon fudging |
| **Wrong-root convergence** — Newton lands on a different intersection than the seed implies | seed too coarse relative to patch curvature | this is a **correctness** parameter, not a speed knob. Zemax tightened its 21.3.2 chord-tolerance default *"to enable more accurate ray tracing"* — i.e. they shipped this bug. Drive `ε_flat`/`ε_size` (§3.4) from patch curvature; verify against the §16-P1 oracle |
| **Proxy miss at grazing incidence** — ray hits the true patch, misses the triangle | — | inflate the proxy shell by the subdivision bound, then reject in CH if the Newton hit falls outside the true `(u,v)` domain. Count `proxy_miss` |
| **Multiple equivalent intersections** (Efremov 2005) | Newton returns two roots within tolerance | keep nearest; count `trim_ambiguous` |
| **Rational weights corrupt the ε criterion** (Efremov 2005) | — | scale the convergence test by `w`; never test on premultiplied coordinates |

Efremov, Havran & Seidel (2005) is a *correctness* paper — it reports no
performance numbers and states its implementation is deliberately unoptimized.
Read it for the degeneracies, not the speed.

---

## 8. Precision contract

The 4090 runs fp64 at **1/64** the fp32 rate. Naively porting the Python engine's
all-fp64 `RayBatch` would make this engine *slower* than `cengine`.

| quantity | precision | rationale |
|---|---|---|
| BVH traversal, AABBs, proxy triangles | **fp32** | hardware-accelerated; geometric tolerance ≫ fp32 ulp at scene scale |
| Newton on rational patches | **fp64** | convergence + wrong-root avoidance |
| hit point, normal, `t` | **fp64** | feeds OPL |
| **OPL accumulation** | **fp64, segment-relative** | see below |
| Jones `Es`/`Ep` | **fp64 complex** | |
| gather phase | **fp64 → `mod 2π` → fp32 trig** | see below |

### 8.1 Invariants that must not break

- **`opl = 0` on the emitting surface** — the emitting surface *is* the reference
  wavefront (`rays.py:6-8`). Non-negotiable; the entire coherent model rests on it.
- **`mod 2π` in fp64 *before* any fp32 trig** (`gather.py:159-161`; rationale at
  `:28`). Paths are 1e5–1e6 waves; fp32 trig on an unreduced phase injects O(1) rad.
  This is a documented, load-bearing line of the existing engine.
- **Segment-relative OPL**: accumulate per-segment in fp64 and reduce, rather than
  carrying a metre-scale absolute accumulator that needs 1e-10 m resolution over
  metre paths (~1e10 dynamic range). Bounds the accumulator's dynamic range without
  changing semantics.
- **4-deep LIFO medium stack** (`rays.py:MEDIUM_STACK_DEPTH = 4`), `AMBIENT = -1`,
  overflow and pop-mismatch are **hard errors**, not silent corruption.

---

## 9. Trace loop

**Wavefront queues, with inline closest-hit for cheap events.**

OptiX's native model is per-ray recursion with a small payload. That is wrong for
this engine: children fan out at every event (reflect+transmit, o+e, grating
orders), and the ray state — Jones vectors, 4-deep medium stack, differentials,
stratum ids, `refl_hist` — blows the payload/register budget. Divergent
NLO/birefringence code inside closest-hit would thrash.

```
for bounce in 0..max_reflections:
    optixLaunch(nearest_hit)      # whole live wavefront, traversal only
    physics_kernel()              # applies the interface event, SoA
    compact_children()            # scatter into next wavefront
    if wavefront empty: break
```

- **Inline in CH**: trivial events only — pure absorb, pure mirror — to skip a
  queue round-trip.
- **Preserved verbatim**: the 9-bucket ledger and its **1e-3 closure gate**
  (`audit.py:14-27`, exit 3 on failure); the boundary-flux tally
  (`in − out = absorbed`, diagnostic side-table, **never a closure bucket**);
  `(source_id, lam_stratum, pol_stratum)` gather keys; the medium-stack push/pop
  discipline.
- **Detected power stays a diagnostic**, not a closure bucket — detector screens
  are transparent measurement planes; two detectors in a path would double-count
  (`audit.py:9-13`).

---

## 10. Physics port map

`PORTED` (`scripts/raytracer/cengine.py:37-61`) is the **one authority** for what
is C-ported today. Verbatim, 20 tokens:

```
surface:plane, surface:sphere, filter,                            # phase A
surface:cylinder, surface:cone, surface:torus, surface:asphere,
coating, polarizer,                                               # phase B
surface:mesh,                                                     # phase C
coherent, save_fields,                                            # phase D
grating, roughness, scatter,                                      # phase E
birefringence,                                                    # phase F
particles,                                                        # phase G
export_rays, ghost_analysis, viz_pattern                          # phase H
```

**A feature must emit a token or the C engine silently skips its physics.** This
rule was violated once already (P8 NLO bodies, `cengine.py:105-110`). It carries
over to `brepengine` unchanged.

### 10.1 Liftable near-verbatim

`cengine/src/kernels/` — already `static inline` C, already parity-pinned:

| header | size | covers |
|---|---|---|
| `surf.h` | 21 KB | all six analytic surface types |
| `fresnel.h` | 5.4 KB | complex-n Fresnel, TIR, metals |
| `birefk.h` | 5.1 KB | uniaxial o/e |
| `quartic.h` | 5.4 KB | torus quartic |
| `scatterk.h` | 4.4 KB | ABg lobes |
| `kmath.h` | 4.6 KB | shared math |
| `gratingk.h` | 3.4 KB | all grating models |
| `trim.h` | 3.4 KB | canonical-UV regimes (polygon regime **replaced**, §7.2) |
| `thinfilm.h` | 4.1 KB | TMM stacks |
| `gatherk.h` | 3.2 KB | Huygens kernel |
| `rng.h` | 6.9 KB | lineage-keyed Philox (§12) |

### 10.2 Must be written new (Python-only today)

`biaxial` · `particles_explicit` · `beam` · `apodization` · `ray_differentials` ·
`curved_detector` · `extra_detector_faces` · `rough_fresnel_macro` ·
`scatter_g_ne_2` · `temperature` (thermo-optic dn/dT) · `nonlinear` (SHG/Pockels) ·
`saturable` · `tpa` · `kerr` · `time_products` · `gdd_budget`

This is the bulk of the remaining work and the reason §16 phases it. Until the
last token lands, `--engine auto` routes these to Python exactly as it does today.

### 10.3 Coherence-critical facts to preserve

- Mesh faces carry **no usable phase** — and under this design there are no mesh
  faces in the physics path at all, which *removes* the restriction rather than
  preserving it.
- **Differentials die** at gratings, scatter events, and o/e splits (NaN-filled).
- **Table coatings carry no phase** — they borrow bare-interface Fresnel phase.
  Only TMM stacks are coherent (`RAYTRACER.md §6.2`). BREP does not change this.
- Two `xfail`s encode **real physics gaps**, not test debt: circular-polarizer
  retardance (`test_scenes_e2e.py:601`) and a Pockels-adjacent issue
  (`test_nlo_elements.py:192-193`). Do not "fix" them by matching.

---

## 11. Gather rework

The gather is O(pixels × samples) Huygens summation — a GEMM-shaped problem.
**OptiX does nothing for it**, and per §1.2(b) it is 99.5 % of michelson_folded's
wall time.

Current model (`gather.py`, Rayleigh–Sommerfeld-I, *not* full Fresnel–Kirchhoff):

```
E(p) = (1/(iλ)) Σ_i E_i √(dA_i) K_i exp(i k (opl_i + n_amb r_ip)) / r_ip
```

### 11.1 What BREP changes

`RAYTRACER.md §6.2` documents the current normalization as an approximation:
*"source-referenced sample-area normalization by default (renormalized to
geometric detected power)"*, with true `dA` only under `--ray-differentials`.

Exact BREP surfaces + Igehy differentials give **true per-ray `dA` analytically**,
from the exact surface Jacobian rather than a source-referenced estimate. This
removes a documented approximation from the physics — a real correctness gain,
independent of speed.

### 11.2 What must not change

- The `M_eff = (Σ|a|)²/Σ|a|² ≥ 1000` undersampling gate
  (`gather.py:446` `min_eff_samples=1000.0`, computed `:502`, `GatherError:50`),
  and its dependence on **jittered** sampling — a regular grid re-enables coherent
  aliasing. Note `check_sampling`'s π-step check (`:503-512`) is **diagnostic only**
  — it warns, never raises; `M_eff` is the enforcement.
- The two-population split: `smooth` (unbiased G=4 cross-estimator) and `speckle`.
- **Negatives unclipped.** Detector maps are unbiased with zero-mean negative MC
  noise; clip only for display.
- Occlusion mask built once and shared, so backends are identical by construction.

### 11.3 Amdahl honesty

Even a perfect gather rework leaves michelson_folded at 713 s of gather. The
options that actually move it — batched-GEMM/tensor-core restructuring, or an
FFT-based angular-spectrum propagator where geometry permits — are **new physics
approximation risk** and belong in their own round, not this one.

---

## 12. Determinism

Lift `cengine/src/rng.h` **verbatim** as device code.

```c
rng_primary_key(seed, source_id, primary_index)
    = k_mix64(seed ^ k_mix64((source_id << 40) ^ primary_index));
rng_child_key(parent_key, event_ctr, child_slot)
    = k_mix64(parent_key ^ ((event_ctr << 32) | child_slot) ^ 0xA5A5A5A5DEADBEEF);
```

Counter block: key = ray_key split 2×u32; counter = `(event_ctr, draw_block,
0x4D494557 "MIEW", 0x42454E47 "BENG")`.

**Thread-invariant by construction** — every draw is a pure function of
`(seed, ray_key, event_ctr, draw)`, with no sequential stream state. This ports to
GPU thread counts unchanged, which is the whole point of the design.

**`child_slot` enum values must be appended, never renumbered** — renumbering
silently re-keys every downstream ray.

KATs pin canonical Random123 vectors (`cengine/tests/test_rng.c:22-42`); they port
as-is.

**No attempt is made to match numpy's stream.** Python↔`brepengine` is
statistical-only, as Python↔C already is (§2).

---

## 13. Anomaly reporting

Two surfaces, neither of which is a closure bucket.

### 13.1 Diagnostic block

An `anomalies` block in `audit.json`, alongside the 9 buckets:

```json
"anomalies": {
  "nan_rejected":              0,
  "newton_nonconverged":       0,
  "tangential_fallback":       0,
  "proxy_miss":                0,
  "trim_ambiguous":            0,
  "seam_hits":                 0,
  "degenerate_faces_filtered": 0
}
```

Non-zero counts warn. A configurable fraction (default ~1e-6 of rays) fails the
run. Surfaced in the GUI Results pane alongside the existing Power tab.

### 13.2 Per-pixel anomaly image

A detector-resolution `uint32` image counting anomalies per pixel, saved beside
the irradiance cube and plumbed through `detector.py` / `save_detectors`. This is
how you **see** where geometry is failing rather than inferring it from a scalar.

### 13.3 NaN policy

One NaN in an accumulation buffer poisons a pixel forever. **Reject non-finite
radiance at splat time**, count it, surface it. Rejection is not silent and not
free — it is a counted, reported event.

---

## 14. Interchange & integration

### 14.1 `scene.json` — schema_version 3

Carries roles, materials, all body tags, spreadsheet values, placements, and
`shape_key`/instance refs. **No face geometry.** Face ids (`Body.Tip.FaceN`) are
resolved by the engine from the `.brep` via `TopExp` ordering.

```
geometry/<stem>/
├── scene.json          schema_version: 3   (tags only)
└── brep/<body>.brep    the real OCCT shape
```

STLs disappear from the physics path entirely. `faces/*.stl` and `mesh_stl`
references are not produced on this route.

**Face-index fragility carries over and gets worse.** `CLAUDE.md` already warns
that *"rebuilds renumber FaceN"*, so preserved face-mapped props can land on an
edge face after a size edit. Under BREP the `FaceN` ↔ `TopExp` ordering contract
must be pinned explicitly and tested, because it is now the **only** binding
between a tag and a surface.

### 14.2 Routing

`--engine {auto, python, c, brep}` during transition. `auto` picks `brep` only if
every scene feature is in `brepengine`'s own `PORTED` set — same gate function as
today (`cengine.choose_engine:202`), same hard-error-on-forced-unported behavior.

Collapses to `{auto, brep}` plus `c` (retained as the CPU oracle, §14.4) after
cutover.

### 14.3 Env pins

`MIEWB_OCCT_ROOT`, `MIEWB_OPTIX_ROOT`, `MIEWB_BREPENGINE` — matching the existing
`MIEWB_FREECAD` / `MIEWB_OPTICS_PYTHON` / `MIEWB_PVPYTHON` / `MIEWB_CENGINE`
convention in `common.py`.

### 14.4 Engine roles after cutover

| engine | role |
|---|---|
| `brepengine` | **production** |
| `cengine` | **permanent CPU reference/oracle** — replaces Python in that role. 8.3× faster than Python, debuggable, no GPU required. |
| Python | retired **only** when the last `PORTED` token lands *and* the §16 cutover gate passes |

### 14.5 Standalone binary

Per §1.3, `brepengine` must be invocable **without Python in the trace path** —
reading `scene.json` + `.brep`, writing the same npy cubes + ledger JSON that
`run_trace.save_detectors` consumes. This recovers the ~1–2.6 s/run driver floor
that `cengine` still pays. `run_pipeline.py` and the GUI continue to drive it as a
subprocess; that is a *convenience* path, not the only path.

---

## 15. Build

### 15.1 Vendored OCCT — 8.0.0_p1

Tag `V8_0_0_p1` (2026-06-17). **p1 specifically, not bare 8.0.0** — it fixes the
`Standard_Strtod` shared-Bigint-pool race that hits **parallel BRep reads** (§3.2).

Minimal kernel + STEP build:

```
-DBUILD_MODULE_FoundationClasses=ON
-DBUILD_MODULE_ModelingData=ON
-DBUILD_MODULE_ModelingAlgorithms=ON
-DBUILD_MODULE_DataExchange=ON
-DBUILD_MODULE_Visualization=OFF
-DBUILD_MODULE_Draw=OFF
-DBUILD_MODULE_ApplicationFramework=OFF
-DUSE_VTK=OFF -DUSE_TK=OFF -DUSE_FREETYPE=OFF
```

Toolkits needed: `TKernel` `TKMath` (Foundation) · `TKG2d` `TKG3d` `TKGeomBase`
`TKBRep` (ModelingData) · `TKGeomAlgo` `TKTopAlgo` `TKShHealing`
(ModelingAlgorithms) · `TKXSBase` `TKSTEPBase` `TKSTEP` `TKXDESTEP`
(DataExchange, for STEP ingestion).

**Verified minimums (8.0 upgrade guide):** C++17 **mandatory**, no C++20
requirement anywhere. GCC **8.0** minimum — this machine's 11.4 is comfortable.
CMake **3.10** enforced, 3.16+ recommended (and required if `BUILD_USE_PCH=ON`) —
this machine's 3.22.1 is fine.

Other 8.0 changes: `Standard_Failure` derives from `std::exception` and
`Standard_Failure::Raise()` is **removed** (must `throw`). `Handle(T)` is replaced
site-wide by `occ::handle<T>` — the macro is retained for source compatibility, but
new code should use the template spelling. `Standard_Boolean` → `bool`. Global math
wrappers (`Sin`, `ACos`, `Sqrt`) deprecated for `std::` equivalents. NCollection
reworked; `TColStd_*`/`TopTools_*` typedefs deprecated; `NCollection_Map::Seek()`
removed in favor of `Contained()` → `std::optional`. **BRepMesh fully restructured**
(`BRepMesh_FastDiscret` removed, plugin system deleted) — irrelevant to us, since we
do not tessellate for physics.

**Our eight load-bearing classes were verified individually against the 8.0.0
refman and are unchanged in substance:** `GeomConvert_BSplineSurfaceToBezierSurface`
(`NbUPatches`/`NbVPatches`/`Patch(I,J)`/`Patches`/`UKnots`/`VKnots` — same header,
same shape; `Patch()` returns `occ::handle<Geom_BezierSurface>`),
`GeomConvert_BSplineCurveToBezierCurve` (`NbArcs`/`Arc(I)`/`Arcs`/`Knots`),
`BRepTools_WireExplorer` (**unchanged**; note the *new, different*
`BRepGraph_WireExplorer` exists — do not conflate), `BRep_Tool::CurveOnSurface`
(4 overloads, §4.2), `BRep_Tool::IsClosed`, `BRepAdaptor_Surface`
(`Plane()`/`Cylinder()`/`Cone()`/`Sphere()`/`Torus()`/`Bezier()`/`BSpline()`/
`GetType()`), `GeomAbs_SurfaceType` (§3.3), `IntCurvesFace_ShapeIntersector`
(§16-P1). Because `brepengine` is a **new** codebase rather than a 7.x port, the
broad 8.0 source-incompatibility is largely a non-issue for us.

**Deliberately independent of FreeCAD's bundled OCCT 7.8.1**
(`libTKernel.so.7.8.1` inside the AppImage). OCCT's BRep format is
read-backward-compatible — an 8.0 kernel reads 7.8.1-written `.brep` — so this
decouples the engine from FreeCAD's release cadence at the cost of owning the
build and the 8.0 migration surface.

Dynamic linking, per §2.1.

### 15.2 Vendored OptiX

Newest release at implementation time (9.1 as of Dec 2025), vendored in-repo like
`cengine` vendors `yyjson` — **pending a license check**, as NVIDIA's OptiX SDK
redistribution terms must be read before committing headers. OptiX is header-only
against the driver, so this is headers plus a `MIEWB_OPTIX_ROOT` override.

Do **not** use the OptiX 8.0.0 headers currently loose in `~/Downloads/include`
(`OPTIX_VERSION 80000`) — two majors stale, and not an SDK install.

### 15.3 Host toolchain (this machine)

| tool | version | note |
|---|---|---|
| GPU | RTX 4090 Laptop, 16 GB, SM 8.9 | fp64 at 1/64 rate (§8) |
| driver | 580.159.03 | |
| CUDA | 13.0.48 at `/usr/local/cuda-13` | **system nvcc is 11.5 — CMake must pin the right one**, exactly as `cengine/build.sh` already does |
| gcc | 11.4.0 | C++17 OK |
| CMake | 3.22.1 | ≥3.10 required by OCCT 8.0; 3.16+ recommended |
| ninja | 1.10.1 | |
| disk | 2.4 TB free on `/home3` | **`/` is chronically ~97 % full** — everything lives on `/home3` |

---

## 16. Phased plan with acceptance gates

Every phase ends in a measurable gate. **P0 exists to kill the project cheaply if
the numbers say so.**

### P0 — spikes (gate: real numbers, confirm or kill)

1. OCCT 8.0.0_p1 → Bézier patch grid + exact trim extraction on one real demo
   face and one imported STEP part.
2. One fp64 Newton custom-prim IS in OptiX, benchmarked head-to-head against a
   proxy-triangle + CH-refine path on the same patch.

**Gate:** measured MRays/s for both paths. If custom-prim IS is within ~2× of the
proxy path, §7.3 may be simplified to custom prims throughout. If it is 5×+ worse,
§1.2(a) is confirmed on this hardware and the hybrid stands. **Either result is a
win — this phase buys information, not code.**

### P1 — loader + trim preprocessor

**Gate:** for a corpus of demo faces + imported STEP parts, hit points agree with
`IntCurvesFace_ShapeIntersector` (OCCT's own exact intersector, used as the
**oracle**, never the engine) to <1e-9 m. Seam, periodic, and degenerate cases
explicitly represented in the corpus. Plus: a ThreadSanitizer run over concurrent
face evaluation (§3.2).

The oracle's entry point is **`Load(const TopoDS_Shape&, double Tol)`, not
`Init()`** — then `Perform(gp_Lin, PInf, PSup)` / `PerformNearest(...)`,
`IsDone()`, `NbPnt()`, `Pnt(I)`, `UParameter(I)`, `VParameter(I)`,
`WParameter(I)`, `Face(I)`, `State(I)`, `SortResult()`.

### P2 — analytic custom prims + TLAS/BLAS

**Gate:** parity vs `cengine` on the analytic demo scenes, at the *real* bar —
`1e-9` deterministic / `2 %` statistical (`test_cengine_parity.py:101`), not the
overstated one. C trim mask == brep trim mask on untrimmed/band/polygon regimes.

### P3 — wavefront trace loop + lifted kernels

**Gate:** ledger closure **<1e-3 in every scene** (`audit.py:104`); thread-count
invariance (`np.array_equal` across launch configs); RNG KATs pass.

### P4 — freeform hybrid + gather rework

**Gate:** double-slit fringe pitch λL/d ±1 px and **visibility >0.85** end-to-end
from a BREP scene; exact-`dA` gather agrees with the differential path to <1e-4.

### P5 — physics migration of the Python-only tail

**Gate:** each token from §10.2 lands with its own parity tests **before** being
added to `PORTED` — the existing rule, unchanged.

### P6 — OptiX viewport

Sequenced last, deliberately. The engine ships against the existing VTK viewport
(fed by `geomcache` STLs) so the physics win is not bet on a UI rewrite. Replacing
`widgets/vtkview.py` (1,117 LOC) plus `facepicker`, `faceindicators`, ray overlays,
chain/fold linkage lines, ghosting, scale bar, and manipulators is its own round.

### Cutover gate

1. **Physics invariants**, retargeted to `brepengine` — verified against the test
   files, not the docs:

   | invariant | tolerance | source |
   |---|---|---|
   | Fresnel R+T=1 over 200 angles | 1e-12 | `test_kernels.py:43` |
   | TIR Fresnel-rhomb δp−δs | 1e-10 | `test_kernels.py:67-78` |
   | MgF₂ QW on BK7 | 1e-6 | `test_kernels.py:154-158` |
   | torus quartic vs `np.roots` | 1e-7 | `test_kernels.py` |
   | calcite walk-off 6.23° @45°/590nm | ±0.05° | `test_birefringence.py:109` |
   | Kogelnik η=1 at ν=π/2 | 1e-9 | `test_grating_models.py` |
   | Dammann Parseval | 1e-6 | `test_grating_models.py` |
   | lamellar duty=0.5 η₁=4/π² | 1e-12 | `test_grating_roughness.py` |
   | Wiscombe MIEV0 Qext (x=10, x=100) | 2e-3 abs | `test_mie_particles.py` |
   | Igehy differentials vs finite diff | 1e-5 rel | `test_ray_differentials.py` |
   | double-slit visibility | >0.85 | `test_doubleslit_e2e.py` |
   | energy closure, every scene | <1e-3 | `test_scenes_e2e.py:305` |
   | element flux in−out==absorbed | ≤1e-3·emitted | `test_element_flux.py` |

2. **Demos**: all 34 `demos/*.MieWB` at `quick` preset (1e5 rays / 512² / 5λ),
   power + placement within the existing 2 % statistical bar, reusing
   `run_demo_equivalence.py`'s structure and the 15 committed
   `demos/baselines/*.json` oracles. Michelson fringe visibility included.

3. **Adjudication rule — the important one.** Any disagreement >2 % triggers an
   investigation to **establish which engine is correct before making them
   match**. Correctness is not conformance. The Python engine is a reference, not
   an oracle: it has two known-`xfail` physics gaps (§10.3), a documented
   normalization approximation (§11.1), and a lossy trim (§4). A BREP disagreement
   may well be BREP being **right**. Every adjudication and its resolution is
   recorded in this document's changelog.

---

## 17. Risks & mitigations

| risk | severity | mitigation |
|---|---|---|
| **Speedup disappoints** — ceiling is ~3.3× geomean, ~1× gather-bound (§1.2) | **high, and already realized** | Framed up front (§1.6). Sell capability. Take the standalone-binary and gather wins separately. |
| **Custom-prim IS penalty** compounds on fp64 Newton | high | P0 measures it before commitment (§16). Hybrid §7.3 is the fallback and is what Zemax ships. |
| **Wrong-root convergence** — Newton lands on the wrong intersection | **high, silent** | Seed density is a correctness parameter (§7.4). Zemax shipped this bug and fixed it by tightening chord tolerance. P1 oracle catches it. |
| **Proxy miss at grazing** | medium | Inflated shell + CH domain rejection + counted `proxy_miss` (§7.4). |
| **OCCT eval-cache race** — mutation on read | **high, silent, nondeterministic** | Per-thread adaptor copies, mandatory (§3.2). Pin p1 for the `Standard_Strtod` fix. |
| **Periodic seam mishandling** — double-count or leak | medium | Explicit two-pcurve handling + domain unwrap at load (§4.2). The existing `seam_loss` bucket is the tripwire. |
| **Degenerate CAD** | medium | Role-based policy: filter+warn mechanical, hard-error optical (§3.5). |
| **NaN propagation** | medium | Reject at splat, count, per-pixel image (§13). |
| **Tangential contact** | medium | Jacobian-determinant detection → interval bisection on `t` (§7.4). |
| **OCCT 8.0 migration churn** — C++17, `Handle()`→`occ::handle<>`, `Standard_Boolean`→`bool`, `Standard_Failure::Raise()` removed (must `throw`), `TColStd_*`/`TopTools_*` typedefs deprecated, `NCollection_Map::Seek()` removed | low **for our 8 target classes** (all verified unchanged, §15.1); medium for anything else | Minimal module set (§15.1). We are a **new** codebase, not a 7.x migration, so most of the churn never applies. Automated toolkit ships at `adm/scripts/migration_800/`. |
| **`BRep_Tool::IsClosed`/`IsPeriodic` tolerance changed in 8.0** — from `gp::Resolution()` (~1e-290, "practically unusable") to `Precision::Computational()` (~DBL_EPSILON) | **medium, silent** | A **real semantic behavior change** that lands squarely on our seam detection (§4.2): edges that were never reported closed under 7.x may now be. Test seam classification explicitly against known-periodic faces in the P1 corpus. |
| **16 GB VRAM** | low | ~10⁶ patches ≈ 256 MB of control points. `shape_key` instancing (§3.6). Existing `pixel_chunk`/`sample_chunk` OOM knobs carry over. |
| **Face-index binding** is now the only tag↔surface link | medium | Pin and test the `FaceN` ↔ `TopExp` ordering contract (§14.1). Rebuilds already renumber `FaceN`. |
| **Dual-maintenance** during P1–P5 | low | `--engine auto` routes unported features to Python, exactly as today. `cengine` stays as the CPU oracle. |
| **`PORTED` token discipline** — a feature without a token silently skips physics | **high, silent** | The rule carries over unchanged and was violated once already (`cengine.py:105-110`). Token-emission tests per feature. |

---

## 18. Honest limits

What this engine will **not** do, in the style of `RAYTRACER.md §6.2`:

- **It does not make prescribed optics more exact.** Analytic prescriptions beat
  their BREP representations (§1.5). `surface_override` survives, and should.
- **It does not fix the physics-model gaps.** Scalar polarization-blind lamellar/
  Dammann gratings (no RCWA); table coatings carrying no phase; the effective-index
  Fresnel approximation at birefringent interfaces; no dichroism, optical activity,
  gyrotropy, or conical refraction; BRDF-only single isotropic scatter; continuum
  particles incoherent by construction. **Geometry exactness is orthogonal to all
  of these.**
- **It does not speed up gather-bound scenes.** michelson_folded ceiling: 1.002×.
- **It does not remove tessellation from the GUI** until P6, and even then the
  question is whether an OptiX-rendered viewport is worth rewriting `widgets/`.
- **It does not give bit-exact agreement with Python.** Never has been possible
  (§12); `default_rng` vs Philox.
- **It does not eliminate the Python driver** unless §14.5 is actually built.
- **Curved detectors remain incoherent-only** unless P5 changes that model — a
  physics decision, not a geometry one.
- **It cannot represent optical contact.** Proper nesting works (LIFO medium
  stack); optically-contacted solids still don't exist, and the ~5 µm air gap /
  nested-thin-plate workarounds (`CLAUDE.md`) still apply.

---

## References

- Martin, Cohen, Fish, Shirley (2000). "Practical Ray Tracing of Trimmed NURBS
  Surfaces." *JGT* 5(1):27–52. — the canonical refine→BVH→Newton architecture;
  17–22 µs/ray-NURBS test on a 300 MHz R12K.
- Nishita, Sederberg, Kakimoto (1990). "Ray Tracing Trimmed Rational Surface
  Patches." *CG* 24(4):337–345. [doi:10.1145/97879.97916] — Bézier clipping.
- Efremov, Havran, Seidel (2005). "Robust and Numerically Stable Bézier Clipping
  Method for Ray Tracing NURBS Surfaces." [doi:10.1145/1090122.1090144] —
  correctness/degeneracies. No performance numbers by the authors' own statement.
- Schollmeyer & Fröhlich (2009). "Direct Trimming of NURBS Surfaces on the GPU."
  *TOG* 28(3):47. — bi-monotonic trim segments; VW Beetle (53,554 patches):
  tessellation+bisection 14.2 ms vs ray casting 93.6 ms.
- Sloup & Havran (2021). "Optimizing Ray Tracing of Trimmed NURBS Surfaces on the
  GPU." *CGF* 40(7):161–172. — 2D kd-tree trim location; CUDA (not OptiX);
  49–499 MRays/s on RTX 2080 Ti/3090.
- Schulz (2009). [doi:10.1016/j.cagd.2007.12.006] — proof of Bézier clipping's
  quadratic convergence (empirical until then).
- Zemax OpticStudio ACIS CAD modes.
  <https://community.zemax.com/got-a-question-7/using-new-acis-cad-libraries-in-opticstudio-20-3-225>
- OptiX 9.0 release notes (DMM removal).
  <https://forums.developer.nvidia.com/t/optix-9-0-release/322842>
- NVIDIA MDL SDK. <https://github.com/NVIDIA/MDL-SDK> (BSD-3-Clause)
- OCCT 8.0.0_p1. <https://github.com/Open-Cascade-SAS/OCCT/releases/tag/V8_0_0_p1>
- OCCT upgrade guide. <https://dev.opencascade.org/doc/overview/html/occt__upgrade.html>
- OCCT licensing. <https://dev.opencascade.org/resources/licensing>
- OCCT thread-safety overview. <https://dev.opencascade.org/content/thread-safety-overview>
- Embree 4. <https://github.com/RenderKit/embree> (Apache-2.0) — evaluated and
  rejected (§2); no NURBS/trimmed-surface primitives, and its subdivision-surface
  support is Catmull-Clark, a different representation with no trim loops or
  rational weights. Not a path to BREP faces.

---

## Changelog

| date | change |
|---|---|
| 2026-07-15 | Initial design. MDL dropped; Embree dropped; analytic-first authority order established; hybrid patch strategy adopted over the brief's custom-prim-everywhere; Amdahl ceiling (~3.3×) quantified and the verdict reframed from speed to capability. |
