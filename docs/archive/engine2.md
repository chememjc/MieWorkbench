# engine2.md — redesign for physical realism and speed

Status: **design document**. No code exists.
Supersedes `engine.md` (kept as a research record; its §1.2(b) and §11.3 are
**wrong** — see §2).
Companions: `docs/RAYTRACER.md` (physics reference), `cengine/README.md`.

The question this answers: **knowing what we know now, how would we rebuild this to
be better in physical realism (first) and speed (second)?**

---

## 0. Do these now, regardless of everything else

Three defects found while planning. All independent of the redesign, all small, all
worth shipping immediately. None changes any physics.

### 0.1 The selection "spiderweb" is a display bug (~10 lines)

`mieworkbench/widgets/vtkview.py:752-754`:

```python
prop.EdgeVisibilityOn()
prop.SetEdgeColor(*_SELECTED_EDGE_COLOR)   # (0.10, 0.10, 0.10) — near-black
prop.SetLineWidth(2.0)
```

On selection this draws **every triangle edge of the STL** in near-black at 2 px over
the orange face. On a doubly-curved lens at 15° angular deflection that is, literally,
a black web on orange.

**Refining the tessellation makes it worse, not better** — 15°→2° multiplies triangles
~56× and therefore web lines ~56×. Any instinct to "just mesh it finer" is exactly
backwards.

Fix: the face is already recolored orange on selection, so the edges are redundant —
delete or condition the three lines, or switch to `vtkFeatureEdges` to draw only the
face's silhouette.

**Second, separate defect:** `grep -rn "PolyDataNormals|FeatureAngle|Phong|Gouraud"
mieworkbench/` returns **zero hits**. `vtkview.py:663-666` wires `vtkSTLReader` →
`vtkPolyDataMapper` directly, so every curved surface is **flat-shaded off per-facet
STL normals**. That is the faceted *look*, independent of the web. Fix is ~5 lines:
`vtkCleanPolyData` (STL has unshared vertices; points must be merged first) →
`vtkPolyDataNormals(FeatureAngle=60, SplittingOff, ComputePointNormals)`.

Both are display-only, touch no physics, and are cached per `shape_key`.

### 0.2 The runtime estimator is wrong by ~8× (`common.py:926`)

```python
nsamples = max(1.0, rays * 0.5)          # bug 2
gather_ops = (nsamples * npix * max(1, n_coherent_sources) * nlambda
              * max(1, n_pol_strata))    # bug 1
```

- **Bug 1:** gather keys are `(source, λ-stratum, pol-stratum)` and each key renders
  the full grid against **its own sample subset**. Strata **partition** M; they do not
  duplicate it. The `nlambda * n_pol_strata` factor is spurious. (`gather.c:617`
  `total_pairs += n_sel * Q` — the code itself counts `Q × M`.)
- **Bug 2:** `rays * 0.5` ("assume half the primaries survive") — a measured
  interferometer yields **~6 samples/ray** (both arms recombine, every child
  survives).

Measured: the estimator overshoots by **8.21×** on `results/example/quick-phase0`
(715.4 s predicted vs 87.1 s actual) — a ~10× overcount against a ~12× undercount.

**Correct law**, validated blind against a withheld benchmark number to **−0.076 %**:

```
gather_s ≈ 1.9 (CUDA context init) + Q · M_total / 6.71e9
Q = resolution²,  M_total = actual surviving gather samples
```

**nlambda does not enter.**

### 0.3 Free gather wins (~2×, physics-neutral)

See §6.2. Three mechanical fixes to `cengine/src/kernels/gatherk.h` that change no
math: fuse `sqrt`+`1/r` into one `rsqrt`; strength-reduce `fmod`; hoist the
sample-invariant `fmod(k·opl_i, 2π)` to the host.

---

## 1. Verdict and priority ranking

### 1.1 The finding that reorders everything

**The gather is the entire performance story, and the trace is not.**

Gather cost is `n_src × resolution² × rays`. Because the λ and pol strata *cancel*
(§0.2), the scaling is:

| preset | rays | Q | trace | gather |
|---|---|---|---|---|
| quick → normal | ×10 | ×16 | ×10 | **×160** |
| normal → detailed | ×10 | ×4 | ×10 | **×40** |

The gather dominates **harder the more accuracy you ask for**. Projected at
`detailed`:

| scene | trace | gather | wall | gather % |
|---|---|---|---|---|
| camera_triplet | 232 s | ~2 s | ~4 min | 0.2 % |
| microscope_objective | 412 s | ~78 s | ~8 min | 16 % |
| **michelson_folded** | 80 s | **142,300 s** | **39.6 h** | **99.94 %** |
| **michelson** | — | — | **~60 h** | ~100 % |

Only 2 of 13 scenes go multi-hour. Both are interferometers, both are ~100 % gather,
for one structural reason: an interferometer recombines *both arms* on *one* screen,
so ~5.7 samples/ray survive coherently and every sample meets every pixel. Elsewhere
sources are incoherent and `gather_s` is a fixed ~1.9 s CUDA-init floor (the five
benchmark scenes reading 1.91/1.91/1.91/1.92/1.94 s are *not* doing gather work —
five identical values cannot be real).

**A tracer rewrite — BREP, OptiX, or otherwise — is aimed at 0.22 % of the problem
in the only scenes that hurt.**

### 1.2 The gather's bug is precision architecture, not code quality

The CUDA gather is **not** badly written. From SASS disassembly: **74 fp64
instructions/pair → 497 G fp64 instr/s against a 472 G hardware peak.** It is running
at ~100 % of the RTX 4090's fp64 instruction-issue rate.

**fp64 is the mistake.** Ada does fp64 at **1/64** of fp32 (CUDA C++ Programming Guide
Table 7, compute capability 8.9: fp32 FMA = 128/clock/SM, fp64 = 2/clock/SM).

The proof is seven lines of `cengine/src/kernels/gatherk.h:44-58`:

```c
double r = sqrt(r2);
double inv_r = 1.0 / r;                                    // 2nd MUFU Newton chain
double rhat_dot_dir = (dx*dir.x + ...) * inv_r;            // fp64
double cos_det = fabs(dx*nrm.x + ...) * inv_r;             // fp64
double K = 0.5 * (rhat_dot_dir + cos_det);                 // fp64, clamped to [0,1]
...
double phase = fmod(k * (opl + GATHER_C_AMBIENT_N * r), K_TWO_PI);
float w = (float)(K * inv_r);                              // ← CAST STRAIGHT TO FLOAT
```

**`K` and `inv_r` are computed in fp64 and immediately cast to `float`.** Every fp64
operation feeding them is discarded to 24 bits on the next line. `K` is an obliquity
in [0,1]; `inv_r` is a 1/r amplitude needing ~1e-3 relative. Neither ever needed fp64.

The only quantity that genuinely needs fp64 is **`r` inside the phase** — paths are
1e5–1e6 waves, and the `mod 2π` reduction before fp32 trig is a load-bearing contract
(`gather.py:28`, `:159-161`).

### 1.3 Priority ranking

Two rankings, because "physical realism first, speed second" has two independent
bottlenecks: coherent scenes are gather-bound, stray-light scenes are ray-budget-bound.

**Tier 1 — high value, small effort. Do these first.**

| # | work | why | § |
|---|---|---|---|
| 1 | **Gather precision architecture** | The only lever touching the multi-hour scenes. 20–60×. Converts *directly* into accuracy (§6.4). | 6.3 |
| 2 | **Scatter importance sampling** (+ BTDF) | Unblocks stray light. The 46,000-year argument (§7.1). Multiple scatter comes nearly free afterward. **Was not on the original list.** | 7.1 |
| 3 | **Q matrix**, **coating phase columns**, **conical-point runtime guard** | Three small, independent correctness fixes. The conical guard removes a *silently wrong* answer. | 4.1, 7.3, 7.2 |
| 4 | **Forbes Q-bfs/Q-con** | Port prysm (MIT). Closes an ISO-normative gap. **Was not on the original list.** | 7.7 |

**Tier 2 — larger, and each unlocks a class of work.**

| # | work | why | § |
|---|---|---|---|
| 5 | **Sequential mode** | The keystone. Turns the *existing* optimizer and tolerancer from toys into tools — confirmed from the code, not assumed (§1.4). | 5 |
| 6 | **Dispatch registry + one-source/two-target build** | The seam every future physics feature needs, and the end of the 16 Python-only features. **Not PRT** — that is already done (§4.1). | 4.2, 4.4 |
| 7 | **Lekner-1991 closed-form uniaxial Fresnel** | Exact, at ~Fresnel cost. Triage by optic-axis **azimuth**, not incidence angle. | 7.4 |
| 8 | **RCWA tables** via meent (MIT) | Into the `order_efficiencies` seam that already exists. | 7.5 |
| 9 | **AD Jacobians** for the optimizer | Full Jacobian at the same complexity as the primal trace. | 8 |
| 10 | **Prescription-primary data model** | "A 100 mm radius *is* 100 mm." Exact by construction; deletes BREP. | 3 |
| 11 | **Element realism** (iris leaves, mounts, figure error) | Real hardware, not idealizations. | 9 |

**Do not build:** conical refraction (§7.2 — not a ray feature, irrelevant to this
engine's crystals, and KTP's gyrotropy destroys the singularity anyway); UTD (§7.6 —
the gather already computes it); KK phase retrieval (§7.3 — provably unsound).

**Full rewrite is not indicated.** The polarization transport is already
PRT-equivalent; the deficits are additive modules plus one data-reduction matrix.

### 1.4 Sequential mode is the keystone — confirmed from the code

The hypothesis was that sequential mode unlocks the most. Testing it against the repo
rather than assuming:

**The optimizer and tolerancer already exist.** `scripts/optimize.py` (554 LOC),
`scripts/tolerance.py` (562 LOC), `scripts/fast_eval.py` (592 LOC),
`mieworkbench/core/{optimize,tolerance}_controller.py`, plus Optimize and Tolerance
GUI panes. `tolerance.py` does sensitivity + Monte-Carlo yield + a per-draw focus
compensator. The operand catalog covers spot RMS, encircled energy, MTF50, detected
power, and arbitrary `report.json` keys.

**So the gap is not capability. It is cost per merit evaluation, and gradients.**

- Even `fast_eval`'s fast "worker" backend costs, per evaluation: apply params →
  re-extract `model.json` → **a full Monte-Carlo trace** → post-process. It cleverly
  patches every source to `coherent=false` so the gather never runs — but it is still
  a full MC trace. **Zemax evaluates a merit function in microseconds.**
- `scripts/optimize.py:351` `--algorithm local` is **scipy Nelder-Mead** — a
  *derivative-free simplex*. It has no gradients at all. Nelder-Mead degrades badly past
  ~10 variables; a real lens design drives 50+. `--algorithm global` is nevergrad CMA-ES
  (`scripts/optimize.py:364`), also derivative-free.

**Hypothesis confirmed.** Sequential mode + a differentiable trace turns an existing,
well-built optimizer from a toy into something that designs lenses. Nothing else in
this document unlocks as much per unit effort.

### 1.5 Go/no-go

**Go — but as a physics project with a performance core, and *not* as the project
`engine.md` described.** Rank order above. The BREP tracer is cancelled (§2.2).

**The strategic frame: selective dominance, not parity.** Matching Zemax outright is
~110–190 engineer-weeks (§12) and pointless. The asymmetry that decides the strategy:

- **Every viable open-source tracer is sequential-only.** Optiland's non-sequential is
  roadmap; KrakenOS is GPL-3; Batoid is narrower than the existing C engine. **The OSS
  world has a sequential kernel we can legally take (Optiland, MIT), and no coherent
  non-sequential engine at all.**
- **Zemax's rigorous diffraction is sequential-only.** Huygens PSF and POP do not exist
  in their non-sequential mode; their NSC coherent sum references phase to the *centre
  of the pixel struck*, and their own staff concede it is *"beyond the scope of the ray
  model."*

So: **take theirs, keep ours.** Adopt a sequential kernel for the ~80 % of daily design
work; pour the rest into the moat Zemax structurally cannot cross — **NLO, time domain,
and honest coherent diffraction inside a non-sequential CAD scene.** Be the best tool in
existence for coherent/nonlinear/stray-light physics; be *adequate* for classical lens
design.

---

## 2. Corrections to `engine.md` and to the repo's documentation

### 2.1 `engine.md` was measured at the wrong end of the curve

`engine.md §1.2(b)` derived a "~3.3× geometric-mean ceiling" for an infinitely fast
tracer. **That number is an artifact of benchmarking at 1e6 rays**, where runs are
1.5–45 s and the fixed ~2 s Python driver is 4–30 % of wall. At production scale the
driver is noise and the ceiling grows with problem size. The methodology was right;
the anchor point was wrong.

`engine.md §11.3` ("Amdahl honesty") asserted the gather-moving options are *"new
physics approximation risk and belong in their own round."* **Wrong.** The 20–60× is
available from a precision fix and an error-controlled local expansion that is
**bit-exact at tile=1**, with zero change to the physics model. It is not a separate
round; it is the main event.

### 2.2 BREP is unnecessary — the motive dissolves

Given prescription-primary optics and CAD-primary mechanical (§3):

- **Optical surfaces are analytic prescriptions and are never meshed** → phase is
  always valid. The tessellation contract's defect (`RAYTRACER.md §6.2`: mesh faces
  carry no usable phase) never arises.
- **Mechanical surfaces are meshed and never need phase** → stray-light scatter off a
  lens barrel is incoherent. A triangle mesh is not merely acceptable there, it is the
  *fastest possible* representation (RT-core hardware traversal).
- The **stated UI motive does not survive contact with the code** — it is a ~10-line
  VTK bug (§0.1).
- A NURBS **cannot exactly represent an even-order asphere** anyway; only conics of
  revolution are rationally representable. `engine.md §1.5` established this and it
  still holds — it just now argues for deleting BREP rather than for structuring it.

OCCT survives **host-side only**, at load time, for importing and fitting (§3.3).
This deletes ~80 % of `engine.md` for zero physics loss.

### 2.3 Errors in the repo's own documentation

| location | claim | actual |
|---|---|---|
| `CLAUDE.md`, `RAYTRACER.md §13`, `cengine/README.md` | parity bar "1e-12 deterministic / 3-seed ±max(3σ,1%)" | `test_cengine_parity.py:101`: `tol = 1e-9 if deterministic else 0.02`. **No 3-seed/3σ harness exists.** 1e-12 applies only to `emitted_W`. |
| `CLAUDE.md` | "Philox … RNG is lineage-keyed" implied engine-wide | **No Philox in the Python engine.** `tracer.py:166` is `np.random.default_rng`. Philox is C-only (`cengine/src/rng.h`). Python↔C was **never** bit-exact. |
| `CLAUDE.md` | "eleven classic-system `.MieWB` galleries" | **34** `demos/*.MieWB`; 15 with committed baselines. The *benchmark* set is 11. |
| `common.py:926` | runtime estimator | Wrong by 8.21× (§0.2). |

These should be corrected in place as a follow-up.

---

## 3. Data model: prescription-primary

### 3.1 The principle

**The optical prescription is the truth. The CAD is a view of it.**

A lens with a 100 mm radius of curvature *has* a 100 mm radius, regardless of what
FreeCAD's BSpline approximation of it says. Today this is inverted: CAD is primary and
`surface_override` is an escape hatch bolted on for aspheres, verified to
`ASPHERE_TOL_M = 1e-6` (`extract_geometry.py:133`) — **1 µm ≈ 2 waves at 500 nm**,
i.e. the CAD is *not* phase-valid for aspheres and the override already *is* the truth.
Formalize what is already true.

### 3.2 The split

| role | truth | representation | phase |
|---|---|---|---|
| **Optical surfaces** | prescription | analytic, closed-form intersection | exact, always valid |
| **Mechanical geometry** | CAD | triangle mesh, RT-core traversal | never needed (incoherent scatter) |

This is how Zemax is built: analytic surface types plus imported CAD for mechanical.

### 3.3 Prescription surface catalog

Authored in **basic terms**, converted internally to the modern basis:

- conic (radius, conic constant) — sphere/paraboloid/ellipsoid/hyperboloid
- even/odd polynomial asphere (today's `surface_override`)
- **Forbes Q-type** (Q-bfs, Q-con) — orthogonal, well-conditioned, the modern standard
- Zernike sag (Noll/Fringe — the *analysis* code already exists, `analysis.py:47-119`)
- XY polynomial / freeform
- grid sag (measured interferogram → §9.3)
- GRIN (index as a function of position)
- plane / cylinder / cone / torus

**Authoring contract:** enter a radius and conic; ask later for it to be re-expressed
as Forbes Q-type without re-authoring. Conversion is internal and lossless where the
bases permit, reported where they do not.

### 3.4 OCCT's remaining role

Host-side, load-time, never at trace time:

1. **Import** mechanical CAD (`.FCStd` `.brp` members, STEP) → tessellate → mesh BLAS.
2. **Fit** prescriptions out of imported *optical* geometry where possible: an OCCT
   `Geom_SphericalSurface` → a sphere prescription with an exact radius; a BSpline
   optical face → attempt an asphere/Forbes fit, report the residual, refuse silently
   never.

Pin: OCCT 8.0.0_p1 (`V8_0_0_p1`, 2026-06-17 — its `Standard_Strtod` fix specifically
addresses a race in **parallel BRep reads**). Verified unchanged in 8.0:
`BRepAdaptor_Surface`, `GeomAbs_SurfaceType` (11 values; note the tag is
`SurfaceOfExtrusion`), `BRepTools_WireExplorer`, `BRep_Tool::CurveOnSurface`.
**Hazard:** OCCT caches BSpline polynomial coefficients **mutating on read** —
concurrent evaluation of a shared surface is unsafe and 8.0 does *not* document a fix.
Copy adaptors per thread; verify with ThreadSanitizer.

### 3.5 Formats

`.MieWB`/`.MieSim` keep their ZIP structure and gain **`prescription.json`** as the
optical source of truth. `model.FCStd` remains the mechanical/visual representation,
**generated from the prescription** for optical elements. Existing files keep opening;
the FreeCAD round-trip constraint is preserved.

---

## 4. Physics core architecture

### 4.1 Polarization: you already have PRT. Do not rewrite it.

**This reverses an earlier decision in this document's own planning, and it is the most
important correction here.**

The plan was to adopt the Chipman/Yun/McClain 3×3 polarization ray-tracing calculus to
replace "chained 2×2 Jones." **That premise is false. The engine's transport is already
PRT-equivalent.**

PRT is `P_q = O_out·J_q·O_in⁻¹` with `O = [ŝ|p̂|k̂]`, `J` the 2×2 Jones matrix bordered
with a 1, and `P_total = P_N···P_1`. The engine:

- carries `s_hat` as an explicit **3D global vector** — `rays.py:75`,
  `self.s_hat = np.zeros((n, 3), dtype=np.float64)`;
- re-expresses amplitudes between bases by **explicit 3D dot products** —
  `fresnel.py:139` `rotate_jones`: `css = Σ(s_new·s_old)`, `csp = Σ(s_new·p_old)`, …

That is the PRT transport. The 3×3's third dimension lies along `k̂`, where the
transverse field is identically zero, so bordering `J` with a 1 and conjugating by `O`
is algebraically what `rotate_jones` already does. **Nothing enters P that is not
already in `{J_q, k̂_in, k̂_out, n̂}`; the map PRT ⇔ Jones+basis is invertible.**

What a *naive* 2×2 chain gets wrong is asserting that surface q's exit basis equals
surface q+1's entry basis. **This code does not make that error.** Geometric
(Pancharatnam) phase and skew aberration (Yun et al., *Opt. Lett.* **36**, 4062)
**emerge from correct basis transport** — they are consequences, not extra terms.
`pol_basis` (`fresnel.py:114-132`) even handles the normal-incidence singularity, which
is removable anyway: `a_s = a_p` exactly where `ŝ` is undefined.

**Zemax itself uses two-pass s/p Jones, not 3×3 PRT.**

#### The real gap: the parallel-transport Q matrix

Yun/McClain/Chipman Paper II (*Appl. Opt.* **50**, 2866) identifies the actual
deficiency: *"Two rays with different ray paths can have the same PRT matrix but
different retardances."* **P alone conflates physical retardance with geometric
rotation.**

Separating them requires the **parallel-transport matrix Q** — the same construction
with `J → I` — after which honest retardance comes from **M = QᵀP**. The engine has no
Q, so it cannot report trustworthy retardance/diattenuation maps.

**This is a small additive module, not a rewrite.** Cost of a full 3×3 chain would have
been 198 flops vs 56 (3.5×) and 144 B vs 64 B of state — noise against BVH traversal
and TMM transcendentals — but it buys nothing we don't have. Write Q; keep the tracer.

#### Partial polarization is already nearly free

Sum **outer products** (coherency `C_i = (Ps)(Ps)†`), not scalar power: averaging
rank-1 Hermitian PSD matrices *is* a partially-polarized coherency matrix. Conditions:
coherent subsets must be summed **as fields first** (which the `(source, λ-stratum,
pol-stratum)` key structure already enforces — §4.5); **DOP is biased high at low N**
(it is a ratio of averages); wide-solid-angle ensembles need the 3×3 coherency.

The detector side needs no change: `gather.py:500-501` already expands to
`E3 = (Es·ŝ + Ep·p̂)·√dA` — the gather consumes 3-vectors today.

**Library note:** [poke](https://github.com/Jashcraf/poke) (BSD-3, CuPy/JAX) implements
Yun exactly — but requires Zemax/CODE V for ray data. **Read it; do not depend on it.**
Polaris-M (Chipman's own) is commercial Mathematica.

### 4.2 Dispatch: replace the 443-line if-chain

Today's interaction dispatch is a hardcoded chain, not a table:

- `tracer.py:526-556`: `if int(f) in self.detectors: … elif body.role == "detector": …
  elif int(f) in scene.gratings: … else: _optic_children`
- `_optic_children` (`tracer.py:716-1159`) — **443 lines** containing a second chain:
  `if body.birefringent:` → `if body.biaxial:` → `if coat … kind == "tmm"` → `elif
  coat` → `if np.any(tir)` → `if grp.has_differentials` → `if rough` → `if scat` → …

There is no registry, no plugin seam, no surface-interaction interface. Adding one
physics feature touches ~10 sites.

**Replace with a registry**: an interaction is a self-contained unit declaring what it
consumes, what children it produces, and its energy bookkeeping. Conical refraction
(§7) becomes a new registration, not a new branch in a 443-line function.

### 4.3 Kill the silent-failure feature-token mechanism

`cengine.py`'s `PORTED` frozenset (20 tokens) gates C routing: `choose_engine` picks C
only if `feats ⊆ PORTED`. **A feature that forgets to emit a token silently skips its
physics.** This is a wrong answer, not a crash — and `cengine.py:105-110` records that
**it has already happened once** (P8 NLO bodies).

Any mechanism where the failure mode is "silently wrong" is unacceptable in a physics
engine. Replace with construction-time registration: an interaction that exists in the
scene and has no implementation on the active backend is a **hard error**, by default,
with no way to forget.

### 4.4 One source, two targets

Write the physics **once**, compile it for both CUDA device and host CPU. The CPU build
is the debug oracle: single-steppable, runs anywhere, no GPU.

**This is not speculative — the repo already does it.** `cengine/src/kernels/kmath.h:25`:

```c
#define KFN __host__ __device__ static inline   // under nvcc
#define KFN static inline                       // otherwise
```

`gather_pair` is already called from **both** `cengine/src/gather.c` (CPU/OpenMP) and
`cengine/src/cuda/gather.cu` (CUDA), and `cengine/CMakeLists.txt` already makes CUDA
optional with a CPU fallback. The proposal is to **extend a proven in-repo pattern**
from the 1,509 LOC of `kernels/` to the whole engine — not to invent one.

This permanently kills the dual-implementation problem: **20 of 36 features are
C-ported; 16 are Python-only** (`biaxial`, `nonlinear`, `saturable`, `tpa`, `kerr`,
`time_products`, `gdd_budget`, `ray_differentials`, `curved_detector`, `temperature`,
`beam`, `apodization`, `particles_explicit`, `extra_detector_faces`, `scatter_g_ne_2`,
`rough_fresnel_macro`) — i.e. **44 % of the feature surface, and the entire recent
roadmap, can never be fast.**

### 4.5 What is preserved verbatim

Non-negotiable, and all currently correct:

- `opl = 0` on the emitting surface — **the emitting surface IS the reference
  wavefront** (`rays.py:6-8`).
- fp64 phase, `mod 2π` **before** any fp32 trig (`gather.py:28`, `:159-161`).
- The 9-bucket energy ledger and its **1e-3 closure gate** (`audit.py:14`, `:104`),
  exit 3 on failure.
- **Detected power is a diagnostic, not a closure bucket** — detector screens are
  transparent measurement planes; two in a path would double-count (`audit.py:9-13`).
- The boundary-flux tally (`in − out = absorbed`), a diagnostic side-table, never a
  closure bucket.
- `(source, λ-stratum, pol-stratum)` gather keys — the mutual-coherence equivalence
  class. Unpolarized sources emit 2 orthogonal populations that **can never interfere**.
- The 4-deep LIFO medium stack; overflow is a hard error.
- Lineage-keyed Philox (`cengine/src/rng.h`), thread-count-invariant by construction.
  `child_slot` values **append-only**.

---

## 5. Sequential and non-sequential modes

Two traversal strategies over **one** physics core, one surface catalog, one material
library. This is how Zemax and CODE V are built.

| | sequential | non-sequential |
|---|---|---|
| geometry | ordered surface list | full 3D scene, BVH |
| rays | ~10²–10³ deterministic, aimed | 1e5–1e7 Monte-Carlo |
| cost | **microseconds** | seconds–hours |
| use | design, optimization, tolerancing, aberration analysis, **preview** | stray light, ghosts, coherent recombination, scatter |

Sequential mode is what makes the *existing* optimizer (§1.4) usable: it replaces a
full MC pipeline per merit evaluation with a microsecond trace, a ~1e6× change in the
inner loop.

It also makes **preview exact**: deterministic, instant, no MC noise, and the physics
you preview is the physics you run. Today preview is a separate 3-process chain
(`raypreview.py`: `save_copy` → AppImage extract → `preview_rays.py`) that can drift
out of sync with the engine. The photon-bead animation (`core/beadanim.py`) replays
returned paths and is unaffected.

Non-sequential remains the default for everything the demo catalog does today.

---

## 6. The gather — the dominant cost

### 6.1 What it is

Rayleigh–Sommerfeld-I, over ray samples:

```
E(p) = (1/(iλ)) Σ_i E_i √(dA_i) K_i exp(i k (opl_i + n_amb r_ip)) / r_ip
K = clip(0.5(cosθ_prop + cosθ_det), 0, 1),  K = 0 for back-radiation
```

**This is exactly what Zemax's Huygens PSF does.** Ansys' own documentation, verbatim:
*"a grid of rays is traced through the optical system"*, each *"a particular amplitude
and phase wavelet"*, summed by *"direct integration of Huygens wavelets"*; cost is
*"the pupil grid size squared times the image grid size squared, times the number of
wavelengths"*; and *"The only disadvantage of the Huygens PSF is speed."*

Their **FFT PSF is the approximation** — it requires the system be F/1.5 or slower, the
image surface in the far field, and the chief ray near-normal (>20° invalidates it).
When the two disagree, Zemax's guidance is that *"the Huygens method is more reliable
because it does not make similar assumptions."*

**On the coherent path we are not behind Zemax. We are doing the reference-grade thing,
slowly.**

### 6.2 The free wins (~2×, zero math change)

| defect | cost | fix |
|---|---|---|
| `sqrt(r2)` then separate `1.0/r` → two MUFU Newton chains | ~10 fp64 ops | one `rsqrt`, then `r = r2·inv_r` |
| `fmod(x, 2π)` → two more `MUFU.RCP64H` chains | **~27 % of the kernel** | `x − 2π·trunc(x·inv2π)`. The compiler *cannot* do this — `fmod` is IEEE-exact by contract |
| `fmod(k·opl_i, 2π)` is **loop-invariant per sample** | Q×M instead of M | hoist to host |

### 6.3 The real fix: precision architecture (20–60×)

Only the **phase** needs fp64. Demote everything else, and amortize the fp64 part.

**Tile-reference factorization.** Tile the detector; for tile centre `p₀`:

```
R_i = |p₀ − s_i|                    ← fp64, ONCE per (tile, sample)
u_i = (p₀ − s_i)/R_i
r   = R_i + u_i·δp + (|δp|² − (u_i·δp)²)/(2R_i) + O(δp³/R²)
```

`k·R_i mod 2π` is computed once in fp64 — **preserving the load-bearing `gather.py:28`
reduction contract**. The residual `k·(r − R_i)` is O(k · tile_size): for a 64-px tile
at 5 µm pitch that is ~4000 rad, which **fp32 carries to ~4e-4 rad**. With a 64×64
tile the fp64 work is amortized 4096×, and the per-pair inner loop becomes fp32 +
`__sincosf` (already used, `gatherk.h:61`).

**This is error-controlled, not a physics change:**

- Tile size and expansion order are the knobs.
- At **tile = 1 it reduces to the current kernel bit-for-bit** — so it is verifiable
  against itself, not merely plausible.
- Gate: bit-exact parity at tile=1; a written phase-error budget; and the pinned
  double-slit visibility > 0.85 invariant as the physics check.

The first-order term `u_i·δp` is **bilinear** in (sample direction, pixel offset) —
i.e. a GEMM, and exactly the "phase-added stereogram" of the computer-generated-
holography literature, which has independently solved this identical sum.

**Default: error-controlled. Exact O(Q×M) available via CLI flag and the GUI's
simulation configuration** (not preview).

### 6.4 The speedup is accuracy, not wall clock

This is the point, and it is why the gather is a *physics* project:

**The gather is a Monte-Carlo estimator of a wave field, not a discretization of one.**
At λ=500 nm over a 25 mm aperture the Nyquist DOF is `(2L/λ)² ≈ 1e10`; we have ~1e6
samples — **1e-4 of Nyquist**. Ray-sample spacing is ~50λ; detector pitch ~10λ. That
is precisely why `gather.py` needs the `M_eff ≥ 1000` gate (`gather.py:446`, `:502`),
mandatory jitter (a regular grid re-enables coherent aliasing), and a 1/M_eff speckle
pedestal. The samples do not resolve the wavefront; the MC estimator launders the
aliasing into zero-mean noise.

Gather cost is **linear in samples**. So a 30× faster gather is 30× more samples is a
real reduction in the speckle floor.

**Default: spend the speedup on accuracy.** Configurable, with a **visual
accuracy-vs-time representation** so the tradeoff is never implicit, and the resulting
M_eff / speckle pedestal always reported.

### 6.5 NUFFT angular spectrum — an auto-gated exact fast path

The RS sum is **not** a type-3 NUFFT directly (type-3 has a *bilinear* phase `s_k·x_j`;
ours is `k|p − s|`, nonlinear in both). But via the Weyl identity:

```
exp(ikr)/r = (i/2π) ∫∫ (1/k_z) exp(i k·(p − s)) dk_x dk_y
```

When a **separating plane** exists — all samples on one side, all pixels on the other,
**true for a detector** — `|z − z'|` linearizes and the exponent separates:

**type-1 NUFFT (samples → uniform k-grid) → × propagator (1/k_z) → type-2 NUFFT
(k-grid → pixels).**

**Exact** to quadrature + NUFFT tolerance (settable to 1e-9), and **cost independent of
both M and Q**.

The gate is the space-bandwidth product `N_k = (2L·sinθ_max/λ)²`:

| scene | N_k | complex64 | verdict |
|---|---|---|---|
| 5 mm beam, ±5° | 3.0e6 | 24 MB | trivially feasible |
| 25 mm aperture, ±20° | 1.2e9 | 9.4 GB | infeasible on 16 GB |

This is the *same* wall Zemax POP hits — hence its guard bands and Rayleigh-range
gating. So: **compute the SBP per coherence key at runtime; take NUFFT if it fits VRAM
and beats Q×M; else fall back to §6.3.** Same discipline as the existing `--engine
auto` routing, and error-controlled so it validates against brute force.

Library: cuFINUFFT (Apache-2.0; GPU type-3 shipped v2.4.0, current v2.5.1;
~1e9 nonuniform points/s).

### 6.6 Rejected, with reasons

| method | why rejected |
|---|---|
| **FMM / directional FMM** | The problem is **1e-4 of Nyquist**. At 1λ leaves over a 25 mm aperture that is 2.5e9 leaf cells holding 1e6 samples → **4e-4 samples per leaf**. FMM's entire premise is amortizing one expansion over many points in a box; at <1 point/box you build 1e6 expansions for 1e6 points and pay pure overhead. FMM literature assumes N ~ K² (Nyquist-sampled surfaces); we are at N ~ 1e-4·K². No GPU library; directional FMM is CPU/MPI only. |
| **Gaussian beam decomposition** (FRED/ASAP/CODE V BSP) | Documented failure mode is **hard apertures in the near field** — precisely the pinned double-slit invariant. *"Due to the soft edges of the Gaussian beams, they cannot completely reconstruct the field of a sharp aperture edge."* Clipping forces re-synthesis of *"several hundred to several thousand new beams."* Also paraxial/ABCD, which will not compose with Jones/birefringence/medium-stack. |
| **Fresnel / type-3 NUFFT** | Paraxial. Changes RS-I → Fresnel. Rejected on the stated priority. |
| **Angular spectrum on a binned grid** | Nearest-cell binning at Δx = 0.73 µm gives phase error `k·δ ≈ 4.6 rad`. Destroys fringes. Only NUFFT-quality kernel spreading is admissible — which is what type-1 *is*. |

**None of these change the MC speckle pedestal** (§6.4). Physical accuracy is set by M
and the `M_eff` gate, not by the summation algorithm — which is exactly why the
cheapest, most verifiable option wins.

---

## 7. Physics realism upgrades

**Research overturned the premise on four of these.** Two named "gaps" are not gaps,
one is not implementable as a ray feature at all, and the highest-value item was not on
the original list. Recorded honestly, because the original priorities were wrong.

| item | verdict |
|---|---|
| PRT 3×3 | **Not a gap** — already PRT-equivalent (§4.1). Real gap: the Q matrix (small). |
| ABg → Harvey-Shack | **Not a gap** — ABg **is** the Harvey-Shack shift-invariant model. Real gap: **importance sampling**. |
| Conical refraction | **Not implementable as a ray feature**, and irrelevant to this engine's crystals. **Do not build.** |
| Coating phase via KK | **Provably unsound.** Ask the user for phase columns, as Zemax does. |
| Berreman 4×4 | Right — but uniaxial has a **cheaper exact** closed form. |
| RCWA tables | Sound; the seam already exists. |
| Forbes Q-type | **Real gap, cheap to close** — was not on the list. |

### 7.1 Scatter: ABg *is* Harvey-Shack. The real gap is importance sampling.

**ABg is not an approximation to Harvey-Shack — it is the HS shift-invariant model.**
Its β/β₀ vectors are Harvey-Shack's; it is isotropic precisely because `|β − β₀|` is
independent of incident direction. And it is not merely empirical: Stover et al.,
*Proc. SPIE* **9961**, 996102 (2016) — *"the empirically observed near shift invariance…
is a direct consequence of the Rayleigh-Rice theory."* ABg's form is the ABC/K-
correlation PSD pushed through the RR golden rule.

**Citation correction:** GHS is **Krywonos, Harvey & Choi**, *JOSA A* **28**(6), 1121
(2011) — **not Vernold** (that is the modified Beckmann-Kirchhoff line).

RR and Kirchhoff are **complementary, not ranked**: RR needs `4πσcosθ ≪ λ` (height);
Kirchhoff needs `4πR_c cosθ ≫ λ` (curvature) — the same phase quantity gated in
opposite directions. RR goes unphysical outside its range: Krywonos Fig. 21 gives
**TIS = 32.19** at 5°/488 nm against GHS's 0.999.

**The decisive finding — the stated priority was wrong.** Multiple scatter is two
different things:

- **within one BRDF interaction** — first-order perturbation, **physically correct for
  smooth surfaces**, and what every commercial tool ships. Fixing it means a better
  *model* (GHS), not more rays.
- **multiple surface bounces** — which any tracer gets **free** by continuing to trace
  scattered rays. (ASAP's `LEVEL` parameter exists to *limit* it.)

**So the real constraint is the ray budget, and the answer is importance sampling.**
BRO's own arithmetic: 1e6 pupil rays × a 1°-density hemisphere ≈ **4×10¹⁴ rays ≈ 46,000
years** for one field at one wavelength — *"Importance area sampling is an essential
feature of any serious stray light analysis program"* (BRO-PN-1157).

Zemax's recipe: target solid angle, **equal flux per ray, average the BSDF over the
cone, rescale power**; the `limit` parameter is the bias knob.

**Do importance sampling before multiple scatter.** Also: **no BTDF is a genuine gap**
— all four commercial tools have it; the transmitted child at a scattering face is
untouched today (`RAYTRACER.md §6.2`). Anisotropy = ellipticize `|β − β₀|`.

Tooling reality check: **Zemax has no Harvey model and no built-in PSD model.** FRED is
richest (Harvey, Extended HS, ABg, K-correlation, Mie, multiple models per surface).

### 7.2 Conical refraction: confirmed not a ray phenomenon — and do not build it

The claim was load-bearing, so it was proved rather than asserted.

Berry's paraxial Hamiltonian (*J. Opt. A* **6**, 289 (2004), Eq. 2.8) is a spin-½
coupled to momentum: `H(P) = ½P²·1 + A S·P`, eigenvalues `H± = ½P² ± A|P|`. **The
±A|P| term is non-analytic at P = 0** — that is the conical touching. Degeneracy needs
two conditions on a 2D parameter space → codimension 2 → an isolated **diabolical
point**. Eigenvectors carry a **half-angle** θ_P/2 → winding ½ → polarization undefined
at P = 0.

Geometric optics fails on three counts (Berry §7): the axial spike **diverges**; ring
asymmetry is wrong; the secondary oscillations are wrong. Berry explicitly forecloses
the ray reading: interpreting them as two-ray interference *"is wrong, because the
polarization states associated with the two rays are orthogonal and so cannot
interfere."*

**The nuance that kills the obvious objection:** ρ₀ = R₀/w → ∞ does **not** recover
Hamilton's thin cylinder. In the focal image `f(ξ)` is **universal — independent of
ρ₀**. Large ρ₀ makes the ring *fractionally* thin while the Poggendorff fine structure
persists at fixed width forever. Geometric optics is recovered only in the far field —
never on axis, never at focus.

Correct treatment: the **Belsky–Khapalyuk–Berry integral** (Berry Eq. 3.3), reducing
for circularly-symmetric input to two 1D Hankel-type integrals. Berry: *"(3.8)–(3.11)
constitute the exact solution of the paraxial model."* It **cannot be a ray feature**:
the crystal acts as a **nonlocal k-space Jones operator** depending on propagation
direction, requiring the incident **angular spectrum**, not a ray.

**Two facts that settle the roadmap question:**

1. **Conical refraction is irrelevant to KTP/LBO/BiBO as this engine uses them** — they
   are cut for phase-matching, never along an optic axis.
2. **KTP is gyrotropic**, and optical activity is *"a singular perturbation that
   destroys the conical singularity"* (*Phys. Rev. Materials* **4**, 055203 (2020)),
   splitting each optic axis into two singular axes. KGW is the CR workhorse *precisely
   because* it lacks optical activity.

**Verdict: do not build.** No open-source BKB implementation exists; only VirtualLab
Fusion (field tracing) does this commercially.

**But fix the real bug regardless:** `birefringence.py:80` `_DEGEN = 1e-9` returns an
**arbitrary** orthonormal basis at the conical point — silently plausible-but-wrong
output. Add a **runtime guard**, not a cone.

⚠️ Cone half-angle `A = (1/n₂)√((n₂−n₁)(n₃−n₂))`, `R₀ = A·l`. Wikipedia's formula is
the **full** cone angle = 2A.

### 7.3 Coating phase: confirmed impossible; ask the user, as Zemax does

`ln r = ln|r| + iθ`; `R = |r|²` gives **only the real part**. The Kramers-Kronig/Bode
relation returns the **minimum** phase — it is an *inequality, not an equality*. Any
`G` factors as `G_MP·G_AP`, and the **Blaschke all-pass factor has unit magnitude at
every frequency** — provably invisible to R while carrying arbitrary phase.

**The crux: multilayers are not minimum phase.** *"Unfortunately, for a thin film, the
zeros associated with Fabry–Perot resonances are typical at all angles."* The
authoritative statement — Tikhonravov, Baumeister & Popov, *Appl. Opt.* **36**, 4382
(1997) — is that phase is KK-derivable *"provided the radiant reflectance **and the
Blaschke factors** are known."* **The Blaschke factors are exactly what a table lacks.**

**Do not implement KK retrieval.** Two sound options, both industry-standard:

1. **Accept user-supplied phase columns.** This is exactly what Zemax's TABLE coating
   does: `ANGL/WAVE/Rs Rp Ts Tp **Ars Arp Ats Atp**` (phases in degrees). **Zemax did
   not solve the inverse problem — it declined to, and asked the user.**
2. **Fit a TMM stack** to the table (as OptiLayer OptiRE / Essential Macleod do) —
   strictly more powerful, and yields GDD for free (§7 ties to the existing
   `gdd_budget`).

Ellipsometry gives `Δ = φ_p − φ_s` — **retardance absolutely; common phase never.**

**It matters, quantified:** a dielectric NPBS at 45° ≈ **20° retardance**; a silver fold
mirror at 450 nm swings **16°→70° across 30–60° AOI**; an F/1.5 Cassegrain that is
diffraction-limited unpolarized shows **half a wave of astigmatism** when polarized.

**Minimum fix, cheap:** flag table coatings **phase-invalid** and refuse or warn on
interferometric/ultrafast/polarimetric runs. Today they silently borrow bare-interface
Fresnel phase (`RAYTRACER.md §6.2`) — which is precisely how a coherent result gets
quietly corrupted.

### 7.4 Anisotropic interfaces: Berreman is right, but uniaxial has a cheaper exact form

The effective-index approximation fails because (a) eigenmodes are orthogonal in **D**,
not **E** — and boundary conditions are imposed on E/H; (b) cross terms are real;
(c) flux normalization needs **Poynting**, not k (percent-level for calcite's 6.2°
walk-off).

**The triage axis is azimuth, not incidence angle.** Lekner, *JOSA A* **40**, 722
(2023): `r_sp` and `r_ps` are **odd in optic-axis azimuth φ** — they **vanish
identically at φ = 0 or 90°** (most textbook scenes, which is exactly why the current
approximation looks fine) and are **maximal near 45°**.

**For uniaxial non-absorbing, skip the eigensolver entirely:** the Booker quartic
factors analytically and **Lekner (1991)** gives closed-form exact o/e amplitudes at
~Fresnel cost. **That is the correct engineering answer for calcite/quartz.**

**Berreman 4×4** (*JOSA* **62**, 502 (1972)): `dΨ/dz = i k₀ Δ Ψ`, `Ψ = (Ex,Hy,Ey,−Hx)`
— exactly the tangentially-continuous components. Subsumes the single interface,
handles biaxial and absorbing natively. Thick/absorbing stacks need an **S-matrix**.
Reserve it for biaxial/absorbing/coated-anisotropic.

**Optical activity, resolved:** `g = G·k` is nonlocal — **but a ray has a known k**.
**Freeze g per ray** → local gyrotropic ε → standard Berreman. This is exactly
**McClain, Hillman & Chipman, *JOSA A* **10**, 2371 & 2383 (1993)** (quartz, calcite,
HgS) — the single most applicable reference. Keep reciprocal (natural, g ∝ k) and
non-reciprocal (Faraday, g ∝ B) paths distinct. Quartz **21.77 deg/mm @ 589.3 nm**
is the validation target.

**Dichroism is not "just complex n":** complex `k_z` → **inhomogeneous waves**
(constant-phase ∦ constant-amplitude); eigenmodes go elliptical and **non-orthogonal**;
complex-symmetric ε → a **non-normal** eigenproblem with exceptional points. This
destroys exactly the orthogonal-linear-mode structure the effective-index scheme
assumes.

**Library:** GeneralTmm (MIT, C++/Eigen, active v1.3.1 Mar 2026) — best for embedding.
pyGTM / Passler–Paarmann (*JOSA B* **34**, 2128 + **erratum 36**, 3246) has
degeneracy-robust mode sorting — the failure mode that bites a tracer sweeping
orientations.

### 7.5 RCWA: precompute-and-interpolate is exactly what industry does

**Per-ray RCWA is impossible.** The core is a dense complex eigendecomposition,
**O(N³)**. Measured (`zgeev`): N=101 → **15 ms**; N=441 → 0.32 s; **N=882 (21×21
crossed) → 2.1 s**; fitted exponent 2.97. At 1e6 rays × 15 ms ≈ **4 CPU-hours per layer
per bounce**; crossed gratings ≈ decades.

**The precompute approach is validated by the vendor mechanism itself.** The
Zemax/Lumerical dynamic link samples on a **51×51 (or 101×101) direction-cosine grid**,
computes missing points on demand, caches in RAM, and — critically — **interpolates the
electric field, carrying phase and polarization**, not a scalar efficiency.
Interpolating complex real/imag **sidesteps phase unwrapping**. Storage is trivial
(~65 MB for 101² × 11 orders × Jones × 9λ); the build is ~51 CPU-hours, embarrassingly
parallel.

**Failure mode: Rayleigh/Wood anomalies** — efficiency swings tens of percent across a
fraction of a degree, and RCWA itself struggles there. Their (λ,θ) loci are **analytic**,
so refine adaptively.

Essential details: **Li's inverse rule** for metallic TM; **S-matrix** for stability.
Convergence: correct factorization at **20 orders** beats conventional at **400**.

**The seam already exists.** `grating.py:269`
`order_efficiencies(spec, lam, cos_i, orders)` already takes `cos_i` — but the `table`
branch **interpolates on λ only and ignores it**, and there is no azimuth φ. Extending
the table to (λ, θ, φ, pol) with complex amplitude **is the whole job.**

**RCWA subsumes Kogelnik** (the two-wave SVEA limit). Keep Kogelnik for thick,
weakly-modulated, near-Bragg VBGs; RCWA elsewhere.

**Library (verified via GitHub API, 2026-07-15):** **meent** (kc-ml2) — **MIT, pushed
2026-07-14, numpy/JAX/torch, autodiff + vector FMM** — the live choice. **FMMAX is
archived on both forks** (invrs-io Oct 2025; facebookresearch 2026-03-08) — MIT and
excellent, but unmaintained. **S4 is effectively dead** (last push 2021-01) **and
GPL-2.0**. grcwa GPLv3; torcwa LGPLv3; Inkstone AGPLv3; nannos GPLv3. **Note the
license spread — MIT only for meent/FMMAX.**

**Buy the tables (meent, MIT). Write the interpolator.**

### 7.6 Edge diffraction / UTD: your gather already computes it

**The Huygens gather *is* the Kirchhoff/Rayleigh–Sommerfeld integral; edge diffraction
is its content, not an add-on.** Du et al. (2023) prove that for an **absorbing** screen,
UTD-under-the-Fresnel-approximation is **equivalent** to the knife-edge model = the
Kirchhoff integral. The edge-diffracted ray is the *endpoint contribution* of the
integral already being sampled. UTD's Fresnel transition function exists to fix the
shadow-boundary divergence — *the regime the gather already handles best*. Its PEC-wedge
coefficient is **wrong for black baffles** anyway.

**No commercial optical tool implements GTD/UTD.** ASAP uses a stationary-phase macro;
TracePro a stochastic BDDF envelope (*"will not show diffraction rings… but the energy
under the curve is correct"*). Zemax NSC has essentially none.

**The one real argument is stray light:** a gather goes **straight to the detector** —
the diffracted field is *evaluated, never propagated*. It cannot scatter off a baffle
and land somewhere. A diffracted **ray** re-enters the tracer. If that is ever wanted,
the answer is a **Keller-cone launch with plain GTD weights away from the shadow
boundary** — not UTD, not the PEC coefficient.

### 7.7 Forbes Q-type: a real gap, cheap to close — and it was not on the list

Forbes (*Opt. Express* **15**, 5218 (2007); **18**, 13851 and **18**, 19700 (2010)).
**ISO 10110-12:2019 §4.3.2.2 is normative**: §4.3.2.2.1 *"Orthonormal in **slope**"* =
Q-bfs; §4.3.2.2.3 *"Orthonormal in **amplitude**"* = Q-con = `T_m(2w²−1)`.

Why it matters: `r⁴, r⁶, r⁸` are **near-linearly-dependent over a finite aperture** →
huge correlated cancelling coefficients → an ill-conditioned optimizer and a
manufacturing spec nobody can hold. `Σa_n²` maps to weighted RMS slope, which is
directly a manufacturability metric.

**[prysm](https://github.com/brandondube/prysm) (MIT)** has `Qbfs`/`Qcon`/`Q2d` with
analytic derivatives — **port it into `surfaces.py`.** (Freeform Q-2D is ISO 10110-**19**,
not -12.)

This composes with §3.3's "author in basic terms, convert internally" contract: enter a
radius and conic; ask later for Forbes without re-authoring.

### 7.8 GRIN: there is no spec to match, and one real trap

Zemax's GRIN step is a **fixed z-axis geometric step with no adaptive control**
("Sharma" appears **zero** times in its manual — the attribution is folklore).
Open-source GRIN is **empty**: Optiland literally `raise NotImplementedError`; KrakenOS
returns a hardcoded n = 1.25.

**The trap that matters here:** OPD accumulation (`d(OPD)/ds = n`) is the
accuracy-critical term **for the coherent gather**. First-order integrators surface as
**phase noise** — Zemax's own manual concedes OPD convergence lags ray convergence.
**Use RK4.**

---

## 8. Optimization and tolerancing

The machinery exists and is well-built (§1.4). What it needs, in order:

1. **A deterministic, noise-free evaluator** — i.e. sequential mode (§5). This is not
   primarily about speed. **It is about gradients existing at all.** MC speckle noise
   makes finite differences garbage; Zemax recommends Orthogonal Descent over DLS
   *specifically for noisy non-sequential problems*, and this engine is permanently in
   that regime (§12.3). A sequential trace is deterministic, so DLS and exact autodiff
   both become possible.
2. **Adopt Optiland (MIT)** as that kernel rather than writing one — it brings DLS, an
   operand catalog, paraxial, Seidel, Zernike, MTF, torch autograd/GPU, MC tolerancing,
   and `.zmx` import. Port to C++ if it profiles hot; a sequential trace is ~100 rays ×
   ~20 surfaces, so the Python cost is bounded. **Build the parity oracle first**
   (§12.3) — two engines without an arbiter is the largest architectural risk here.
3. **Damped least squares** — Spencer (*Appl. Opt.* **2**, 1257, 1963) ≡
   Levenberg-Marquardt, with Meiron per-parameter damping. Replaces the current
   derivative-free Nelder-Mead simplex (`optimize.py:351`), which degrades badly past
   ~10 variables where real designs drive 50+.
4. **AD Jacobians** — Seger et al. (*Opt. Express* **33**, 3054, 2025): autodiff of the
   ray–surface intersection gives the **full Jacobian at the same complexity as the
   primal trace**. The industry's finite-difference habit is optional.
5. **Global search** — keep nevergrad CMA-ES (`optimize.py:364`); it becomes usable once
   evaluations are microseconds, and it is the *correct* choice on the MC path, which
   stays noisy by nature.
6. **Tolerancing** — `tolerance.py` already does sensitivity + MC yield + a compensator.
   It inherits the speedup for free, and only then becomes real: **N ≥ 4,602 MC draws**
   are needed for 99.9 % yield at 99 % confidence (Laville & Aymard, arXiv:2607.03067).
   Infeasible at MC-trace cost; seconds at sequential-trace cost.

Attribution correction: Zemax's **Global Synthesis is Kuper & Harris** (*Proc. SPIE*
**1780**, 1993), **not Isshiki** — a separate escape-function lineage. Hammer/Global
Search internals: **NOT FOUND** (deliberately vague in the vendor docs).

---

## 9. Element library realism

Each toggleable; ideal behavior remains the default.

### 9.1 Real iris — a new library part

An N-blade iris with actual leaf geometry: straight or curved leaves, the polygonal
aperture they form, leaf thickness and overlap. This changes the diffraction pattern
(the classic N-pointed star), the ghost paths, and the vignetting — none of which a
perfect circle reproduces.

### 9.2 Real mounts and mechanical geometry

Barrels, retaining rings, bevels, chamfers, edge blackening — the actual scatter and
vignetting sources in a real instrument. This is what CAD ingestion (§3.4) is for, and
it is incoherent, so meshes are correct.

### 9.3 Surface figure error

Real optics are not perfect. Mid-spatial-frequency figure error and measured
interferogram maps (Zernike coefficients or grid sag) dominate real PSF and Strehl.
Today the engine has ideal surfaces plus statistical roughness only — the deterministic
middle is missing. Enters as a prescription layer (§3.3), so it composes with any base
surface.

### 9.4 Colors and affordances

An iris should look black. Cheap, and worth doing — but strictly after the physics.

---

## 10. Display and preview

- **Exact analytic rendering for prescription surfaces.** Since the optical truth *is*
  the prescription, render it exactly rather than as an STL approximation of a CAD
  approximation of it. Faces become single smooth entities; selection is clean by
  construction.
- **Exotic/mechanical geometry keeps meshed display** — accepting imperfect graphics
  rather than lag.
- **The §0.1 fixes ship first** and may resolve the complaint entirely on their own.
- **Preview unified with sequential mode** (§5): exact, instant, no MC noise, same
  physics as the run. Photon-bead animation (`core/beadanim.py`) replays the paths.
- **ParaView stays** for deep analysis — genuinely the best free option on Linux, and
  nothing in this redesign displaces it.

---

## 11. Migration plan with gates

Evolve in place. Keep the GUI (23.2k LOC), scripts, library data, demos, and the 37k
LOC of tests — none of these is the problem, and all encode hard-won correctness.
Rewrite only the engine core.

| phase | work | gate |
|---|---|---|
| **P0** | §0 quick wins | Spiderweb gone; estimator within ~20 % of the measured law; gather ~2× with **bit-identical** output |
| **P1** | Gather precision architecture (§6.3) | **Bit-exact at tile=1**; written phase-error budget; double-slit visibility > 0.85; michelson_folded `detailed` under ~1 h |
| **P2** | Tier-1 correctness: scatter importance sampling + BTDF (§7.1); Q matrix (§4.1); coating phase columns + phase-invalid flag (§7.3); conical-point guard (§7.2); Forbes Q-type (§7.7) | Stray-light scene tractable at a real ray budget; retardance maps validated; quartz 21.77 deg/mm; Forbes vs prysm to 1e-12 |
| **P3** | Physics core: dispatch registry + one-source/two-target build (§4.2, §4.4) | Every pinned invariant passes (Fresnel 1e-12, Kogelnik 1e-9, calcite 6.23°, Wiscombe Qext, closure < 1e-3); CPU and GPU builds agree; **no feature can route without an implementation** |
| **P4a** | **Parity oracle first** — Optiland vs the MC engine on shared `demos/` scenes, gated like `test_cengine_parity.py` (§12.3) | An arbiter exists **before** a second physics truth does. Non-negotiable ordering. |
| **P4b** | Sequential mode via Optiland (§5, §12.3) + DLS + AD Jacobians (§8) | Merit evaluation in microseconds; a known lens design converges to its published prescription; `.zmx` import round-trips |
| **P5** | Prescription-primary data model (§3) | All 34 demos reproduce within the real 2 % statistical bar; prescriptions round-trip through FreeCAD |
| **P6** | Lekner uniaxial (§7.4), RCWA tables (§7.5), Berreman for biaxial/absorbing/gyrotropic (§7.4) | Per-feature validation against published data |
| **P7** | Element realism (§9), display (§10) | — |
| **P8** | Freeze Python at a git tag as an archive oracle | Golden vectors committed as fixtures |

**Adjudication rule.** Any disagreement > 2 % triggers an investigation to **establish
which engine is correct before making them match**. Correctness is not conformance. The
Python engine is a reference, not an oracle: it has two known-`xfail` physics gaps
(circular-polarizer retardance, `test_scenes_e2e.py:601`; a Pockels-adjacent issue,
`test_nlo_elements.py:192-193`), a documented normalization approximation, and a lossy
trim. A disagreement may well be the new engine being **right**.

---

## 12. What it would take to match or beat Zemax

**Full parity is ~110–190 engineer-weeks (2–4 engineer-years) and is strategically
pointless. Selective dominance costs ~20–30 and is achievable.**

Effort estimates are anchored on this repo's own measured size — `scripts/raytracer/`
is 14,554 LOC against 19,006 LOC of tests, a **~1.3× test multiplier this project
actually sustains**. One experienced engineer at that test density. **The dominant
error term is validation, not implementation**: in this codebase physics features are
gated by oracle tests, and that is where schedules die.

### 12.1 Corrections to our own premises

| premise | verdict |
|---|---|
| "Zemax does not do RCWA" | **FALSE.** Premium ships a **1D in-house RCWA DLL** (`srg_trapezoid_RCWA.DLL`, `srg_step_RCWA.DLL`, `srg_user_defined_RCWA.dll`) + a sequential slanted-grating RCWA surface; Enterprise adds the **Lumerical 2D RCWA Dynamic Link**. RCWA is a **gap**, not a lead. |
| "Zemax sequential diffractives are weak" | **TRUE, and narrower than it sounds.** Sequential Binary 1/2/3 model **no efficiency at all** — *"the efficiency to the specified diffraction order is assumed to be 100%."* Our Kogelnik/Dammann/table beats that. But Zemax matches by leaving sequential mode (NSC RCWA DLLs). **`future.md` reads this as a gap; in sequential mode it is a lead.** |
| "Mie continuum is a differentiator" | **WEAK — parity.** Zemax NSC Volume Physics has Rayleigh, Henyey-Greenstein, and a **Mie DLL with correct polarization tracking**. |
| "Zemax does no NLO / time-domain" | **CONFIRMED — the moat.** See §12.4. |

### 12.2 The gap table

Scale: **TRIVIAL** <1 wk · **MODERATE** weeks · **HARD** months · **RESEARCH** open problem.

| # | gap | cplx | eng-wk | basis | deps | OSS | risk |
|---|---|---|---|---|---|---|---|
| 1 | **Sequential mode** (surface list, paraxial, ray aiming, pupils, solves, pickups) | HARD | **12–20** write / **6–10** adopt | minimal seq kernel ≈3–4k LOC + 4k tests | fields/stop/pupil absent from the FCStd contract | **Optiland, MIT** | **Two engines = two physics truths.** Budget **+3–4 wk** for a parity oracle |
| 2 | Paraxial / first-order (EFL, BFL, pupils, Lagrange) | MODERATE | 2–3 | Welford | 1 | Optiland | low |
| 3 | Seidel coefficients | MODERATE | 2–3 | closed-form sums over the paraxial trace | 1, 2 | Optiland | low |
| 4 | FFT PSF / FFT MTF | MODERATE | 2–4 | `opd_exit_pupil()` already exists (`raytracer/analysis_imaging.py:253`) | 1 | prysm | sampling / guard band |
| 5 | **DLS + operand catalog** | MODERATE | 4–6 | LM is textbook; Zemax's 300+ operands are 30 yrs of accretion — chase 10 % | **1 + noise-free eval** | Optiland | **§12.3** |
| 6 | Differentiable trace | HARD | 6–10 (free w/ Optiland) | torch rewrite of intersection + Fresnel | — | Optiland, DeepLens | stochastic ≠ deterministic gradients |
| 7 | Surfaces: Zernike sag, extended poly, grid sag, biconic | MODERATE | 1–2 **each** (8–12) | the bracketed-Newton asphere solver extends | — | Optiland (partial) | Newton convergence on freeform |
| 8 | **Forbes Q-type** | HARD | 3–5 | Forbes 2007; Jacobi recursions, numerically delicate | 7 | **prysm, MIT** | recursion conditioning |
| 9 | **GRIN** | HARD | 4–8 | Sharma RK4 ray-equation ODE | ODE in the trace loop; **OPL along a curved path** | none mature | **coherent-gather + ledger integration**; §7.8 |
| 10 | Diffractive Binary 1/2/3 | MODERATE | 2–4 | phase-polynomial OPD + Sweatt | — | no | low |
| 11 | **RCWA 1D** | HARD | 8–16 | Moharam & Gaylord 1995 | seam exists (§7.5) | **meent, MIT** | convergence vs Fourier order |
| 12 | RCWA 2D crossed | RESEARCH | 16–30 | **Zemax itself outsources this to Lumerical** | 11 | meent | very high |
| 13 | **POP** | HARD | 6–10 (2–4 via prysm) | Goodman; angular spectrum + pilot beam | beam-grid concept absent | **prysm, MIT** | sampling fragility — Zemax's own #1 POP complaint |
| 14 | **Importance sampling** | MODERATE | 4–6 | aim scattered rays at a target solid angle, re-weight | scatter refactor | no | **must not break the 1e-3 ledger** |
| 15 | Path analysis / ghost focus | MODERATE | 3–4 | lineage tagging — **Philox lineage keys already exist** | — | no | low |
| 16 | Inverse sensitivity + yield | MODERATE | 3–5 | root-find tolerance for a ΔMerit budget | **1** (needs 1000s of evals) | Optiland | speed-bound without #1 |
| 17 | Multi-config / zoom | MODERATE | 3–4 | **`miewb_vars` + sweeps are ~60 % there** | — | — | low |
| 18 | Stock lens matching | MODERATE | 3–5 | code is easy | 1 | no | **legal, not technical** — catalog redistribution rights |
| 19 | ISO 10110 drawings | MODERATE | 3–4 | pure output; FreeCAD TechDraw helps | — | no | low |
| 20 | Phosphor / fluorescence | MODERATE | 4–6 | λ-shifting volume emission | — | no | ledger closure |
| 21 | Coating optimizer | TRIVIAL–MOD | 2–3 | TMM exists; wrap LM over thicknesses | — | yes | low |
| 22 | Scatter: BTDF, GHS, pol-sensitive | MODERATE | 4–6 | §7.1 | — | no | low |
| 23 | Thermal / STOP | HARD | 8–12 | dn/dT + FEA deformation import | multi-config | no | FEA interchange |

### 12.3 The critical path — and why sequential mode is the keystone

The hypothesis was that sequential mode unlocks the most. **Confirmed, but it needs
sharpening: the keystone is not "sequential mode" as a feature — it is a fast,
deterministic, noise-free, differentiable evaluator.** Sequential mode is merely the
standard vehicle.

The decisive evidence is not speed. It is that **gradients do not exist today**:

> **Monte-Carlo speckle noise makes finite-difference gradients garbage.** This is
> precisely why Zemax recommends Orthogonal Descent over DLS *specifically for noisy
> non-sequential problems* — **and this engine is permanently in that regime.**

A sequential trace is deterministic, so DLS *and* exact autodiff gradients both work.
That is the deepest reason for §5, deeper than the ~1e6× inner-loop speedup.

Tolerancing makes it starker: **N ≥ 4,602 MC draws** are needed for 99.9 % yield at
99 % confidence (Laville & Aymard, arXiv:2607.03067). At MC-trace cost that is
infeasible; at sequential-trace cost it is seconds.

Gaps **2, 3, 4, 5, 16, 18 all depend on #1.** Nothing else unlocks more than one
downstream item.

**The counter-hypothesis fails.** A torch/GPU differentiable *non-sequential* trace
speeds evaluation but yields **stochastic** gradients, and delivers no paraxial, no
pupils, no Seidel. Sequential wins.

**And the estimate is lower than it looks, because the hard part is already done.**
The conceptually difficult piece of sequential mode is **imposing an order on a CAD
scene** — and `train_solver.py`'s anchored/chained optical-train model **is** a
sequential surface ordering. The bridge from a `miewb_train_*` recipe to a surface list
is far shorter than "write a sequential mode" implies. This materially lowers #1's risk.

**Recommended path (~20–30 eng-weeks to a credible designer's tool):**

1. **Adopt Optiland (MIT) as the sequential kernel** — 6–10 wk including the
   train-chain → surface-list mapping and the unit contract. Free-rides paraxial,
   Seidel, Zernike, MTF, DLS + operands, autograd/GPU, MC tolerancing, and **.zmx
   import** (instant interop with the incumbent). Port to C++ later if it profiles hot;
   a sequential trace is ~100 rays × ~20 surfaces, so the Python cost is bounded.
2. **Build the parity oracle FIRST, not last** — 3–4 wk. Optiland vs the MC engine on
   shared `demos/` scenes, gated like `test_cengine_parity.py`. **Without this you have
   two physics truths and no arbiter. This is the single largest architectural risk in
   the plan**, and it is exactly the risk this repo's one-solver culture
   (`train_solver`, cengine parity) exists to prevent.
3. Then: FFT MTF (#4), inverse sensitivity (#16), multi-config (#17).
4. **Deprioritize:** RCWA-2D, STOP, stock lens matching (legal), Forbes Q-2D freeform.

**Value/effort order:** sequential-via-Optiland ≫ DLS + operands > FFT MTF > inverse
sensitivity ≈ multi-config > importance sampling > POP-via-prysm > surface catalog >
GRIN > RCWA-1D ≫ RCWA-2D.

### 12.4 Where this engine genuinely exceeds Zemax

| candidate | verdict |
|---|---|
| **NLO — SHG / Pockels / Kerr / TPA / saturable** | **REAL. The moat.** Zemax staff, verbatim: *"OpticStudio does not currently model the optical effects of non-linear crystals."* Their only nonlinear capability is TPA via Volume Physics. |
| **Time domain — pulse / spectrogram / streak / GDD budget** | **REAL.** No Zemax equivalent found after targeted search. Users are forced to reason manually about pulse overlap. |
| **Coherent diffraction in a NON-SEQUENTIAL scene** | **REAL, and better physics than theirs.** Zemax's NSC coherent irradiance references phase **to the centre of the pixel struck**; staff concede it *"cannot account for conservation of energy… without making assumptions"* and *"These cases are simply beyond the scope of the ray model."* **Zemax's rigorous diffraction (Huygens PSF, POP) is sequential-only.** We do a true Rayleigh–Sommerfeld-I gather with a **reported, unbiased noise floor** and 1e-3 ledger closure, inside a full 3D CAD scene. **This is the differentiator.** |
| **Biaxial media** | **NARROW but real.** Zemax birefringence is **uniaxial only**, cannot ray-split in sequential mode, and models no optical activity. But §7.4 concedes our effective-index Fresnel; §7.2 concedes no conical refraction. A real, shallow edge. |
| **Open, scriptable, FreeCAD-native, parametric** | **REAL on cost/openness; CONTESTED on CAD.** Zemax Premium supports native Creo/Inventor parts and a full ZOS-API. The edge is parametric round-trip and price, not CAD access as such. |
| Mie continuum | **Parity.** (§12.1) |
| RCWA | **Gap, not lead.** (§12.1) |

**The strategic asymmetry, stated plainly:** every viable open-source tracer is
**sequential-only** — Optiland's non-sequential is roadmap, KrakenOS is GPL-3, Batoid is
narrower than the existing C engine. **The OSS world has a sequential kernel we can
legally take, and no coherent non-sequential engine at all.** Take theirs; keep ours.
The moat to widen — NLO, time-domain, and honest non-sequential coherent diffraction —
is one Zemax cannot cross without building a new engine, and it is confirmed by their
own staff.

---

## 12A. Licensing

**The repo has no LICENSE file today**, which by default means all rights reserved.
The stated intent is to share source while **keeping the option not to**. That
requirement is decisive: **any GPL/AGPL/LGPL link would force disclosure.** So:
**permissive dependencies only.** The permissive set covers every need in this
document — there is no capability that requires a copyleft dependency.

### Approved (permissive)

| library | license | role | status (2026-07-15) |
|---|---|---|---|
| **Optiland** | **MIT** | sequential kernel (§12.3) | 770★, committed 2026-07-15 |
| **prysm** | **MIT** | Forbes Q-type (§7.7), POP (#13), Zernike/MTF | 345★, 2026-07-12. ⚠️ **pin a git SHA** — PyPI is stale at v0.21.1 (2022) |
| **meent** | **MIT** | RCWA tables (§7.5) | pushed 2026-07-14; numpy/JAX/torch, autodiff + vector FMM |
| **GeneralTmm** | **MIT** | Berreman 4×4 (§7.4) | C++/Eigen, v1.3.1 Mar 2026 |
| **cuFINUFFT** | **Apache-2.0** | NUFFT gather path (§6.5) | v2.5.1, GPU type-3 since v2.4.0 |
| **Poke** | **BSD-3** | PRT/Q-matrix reference (§4.1) | **vendor the algorithms, not the package** — its `raytrace.py` only wraps Zemax/CODE V |
| **RayOptics** | **BSD-3** | `.zmx`/`.seq` importers, paraxial | 398★, 2026-07-05 |
| **yyjson** | MIT | already vendored (`cengine/vendor/yyjson`) | — |
| OCCT | LGPL-2.1 + exception | host-side import/fitting only (§3.4) | **dynamic-link** — see below |

### Rejected — copyleft or unlicensed

| library | license | why rejected |
|---|---|---|
| **Optika** | **NO LICENSE** | ⚠️⚠️ All rights reserved — **legally unusable** regardless of intent |
| **KrakenOS** | **GPL-3** | The only OSS non-sequential + STL tracer — but would relicense MieWorkbench. **Read, do not link.** |
| **S4** | GPL-2 | Also effectively dead (last push 2021-01) |
| **Inkstone** | AGPL-3 | Network copyleft — the most aggressive |
| **grcwa**, **nannos** | GPL-3 | meent (MIT) covers the same ground |
| **torcwa** | LGPL-3 | GPU RCWA, but LGPL complicates static linking |
| **rayopt** | LGPL-3 | Dead anyway (PyPI v0.2, 2017) |
| **Goptical** | GPL-3 | Abandoned (upstream dead 2012) |
| **FMMAX** | MIT | *Not* a license problem — **archived on both forks** (invrs-io Oct 2025; facebookresearch 2026-03-08). Unmaintained. |

### OCCT's exception — flagged, not asserted

Two independent research passes **disagree** on whether OCCT's LGPL-2.1 exception
permits static linking. One reads it as covering only header material incorporated into
object code (so §6's relink obligation still binds); the other cites SPDX
`OCCT-exception-1.0` as permitting static linking into closed-source.
**Recommendation: dynamic linking**, which is unambiguous under both readings. OCCT is
host-side and load-time only (§3.4), so this costs nothing.

### Action

Pick a license and add the file. **MIT or BSD-3** preserves every option — including
relicensing later and shipping a closed build — and matches the dependency set above.
Apache-2.0 additionally grants patent rights, at the cost of some GPL-2 incompatibility.

---

## 13. Honest limits

What this redesign does **not** fix:

- **The MC speckle floor is fundamental.** At 1e-4 of Nyquist sampling, the coherent
  reconstruction is a Monte-Carlo estimator. More samples lower the floor as 1/M_eff;
  nothing removes it. The gather speedup buys accuracy — it does not buy exactness.
- **Conical refraction at the optic axis is not a ray phenomenon** (§7.2). No ray tracer
  resolves it; it needs the Belsky–Khapalyuk–Berry integral over an angular spectrum.
  We will *guard* the singularity, not model it.
- **Table coatings still cannot carry phase** from amplitude data alone (§7.3). This is
  an **information-theoretic limit** — the Blaschke all-pass factors are invisible to R
  — not an implementation gap. The fix is to ask the user, which is what Zemax does.
- **Rayleigh/Wood anomalies** will remain the weak spot of interpolated RCWA tables
  (§7.5) — efficiency swings tens of percent across a fraction of a degree, and RCWA
  itself struggles there. Adaptive refinement mitigates; it does not eliminate.
- **Dichroic (absorbing) anisotropic media are genuinely hard** (§7.4): inhomogeneous
  waves, non-orthogonal elliptical eigenmodes, a non-normal eigenproblem with
  exceptional points. Berreman handles it; the mode-sorting is where it will break.
- **DOP from a Monte-Carlo ensemble is biased high at low N** (§4.1) — it is a ratio of
  averages. Report N alongside it or it will mislead.
- **Continuum particle scattering stays incoherent by construction** — physically
  correct for a disordered medium, but it never contributes fringe structure.
- **NUFFT is scene-gated** (§6.5), not a universal win.
- **Sequential mode does not make every system sequential.** Stray light, ghosts, and
  coherent recombination remain non-sequential and expensive.
- **A 39.6 h run becomes ~1 h, not ~1 min.** 20–60× is transformative, not magic.
- **We will not match Zemax's breadth.** ~110–190 engineer-weeks (§12), and the
  300+-operand merit catalog alone is 30 years of accretion. Chasing ~10 % of it is the
  plan; the rest is conceded deliberately.
- **Adopting Optiland creates a second physics truth.** This is the largest
  architectural risk in the document, and it cuts against this repo's one-solver
  culture (`train_solver.py` pinned to 1e-9 by a parity oracle; cengine parity). The
  mitigation — build the oracle **before** the second engine (§11 P4a) — is a
  discipline, not a guarantee.
- **Gradients will never exist on the Monte-Carlo path.** Speckle noise makes finite
  differences meaningless there, permanently. Non-sequential optimization stays
  derivative-free (CMA-ES), and that is correct rather than a deficiency.
- **RCWA tables are only as good as their sampling** near Rayleigh/Wood anomalies
  (§7.5), and RCWA-2D is where Zemax itself gave up and outsourced to Lumerical.

---

## References

- Zemax Huygens PSF — <https://ansyshelp.ansys.com/public//Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Huygens_PSF.html>
- FFT vs Huygens PSF — <https://support.zemax.com/hc/en-us/articles/1500005576202-What-is-the-difference-between-the-FFT-and-Huygens-PSF>
- Zemax POP — <https://optics.ansys.com/hc/en-us/articles/42661957754131-Exploring-Physical-Optics-Propagation-POP-in-OpticStudio>
- Interferometers in Zemax NSC (validated vs experiment) — Photonics 12(3):206, <https://www.mdpi.com/2304-6732/12/3/206>
- CODE V Beam Synthesis Propagation — <https://www.keysight.com/us/en/assets/3125-1210/data-sheets/CODE-V-Beam-Synthesis-Propagation.pdf>
- Ashcraft & Douglas, open-source GBD — arXiv:2106.09162
- Truncated Gaussian beams / GBD hard-edge failure — JOSA A 36(5):859; arXiv:2404.12454
- Engquist & Ying, directional FMM — <https://web.stanford.edu/~lexing/Wave07Long.pdf>; arXiv:0802.4115
- FINUFFT / cuFINUFFT — <https://github.com/flatironinstitute/finufft>; arXiv:2102.08463
- CUDA C++ Programming Guide 12.9.1, Table 7 §8.4.1 (fp32/fp64/SFU throughput, CC 8.9)
- Shimobaba et al., wavefront recording plane — Opt. Express 20(4):4018
- CGH recurrence / low-precision GPU CGH — Comput. Phys. Commun.; Appl. Sci. 11:6235
- OCCT 8.0.0_p1 — <https://github.com/Open-Cascade-SAS/OCCT/releases/tag/V8_0_0_p1>

**Polarization (§4.1)**
- Yun, Crabtree & Chipman, *Appl. Opt.* **50**(18), 2855 (2011) — PRT I, [10.1364/AO.50.002855](https://doi.org/10.1364/AO.50.002855)
- Yun, **McClain** & Chipman, *Appl. Opt.* **50**(18), 2866 (2011) — PRT II (retardance; the Q matrix), [10.1364/AO.50.002866](https://doi.org/10.1364/AO.50.002866)
- Yun et al., *Opt. Lett.* **36**, 4062 — skew aberration, [10.1364/OL.36.004062](https://doi.org/10.1364/OL.36.004062)
- Zhang et al., *Opt. Express* **25**, 26973 — 3×3 coherency, [10.1364/OE.25.026973](https://doi.org/10.1364/OE.25.026973)
- Chipman, Lam & Young, *Polarized Light and Optical Systems* (2018), ISBN 9781498700566
- poke — <https://github.com/Jashcraf/poke> (BSD-3)

**Conical refraction (§7.2)**
- Berry, *J. Opt. A* **6**, 289 (2004) — <https://iopscience.iop.org/article/10.1088/1464-4258/6/4/001>
- Berry, Jeffrey & Lunney, *Proc. R. Soc. A* **462**, 1629 (2006), [10.1098/rspa.2006.1680](https://royalsocietypublishing.org/doi/10.1098/rspa.2006.1680)
- Belskii & Khapalyuk, *Opt. Spectrosc.* **44**, 312/436 (1978)
- Turpin et al., *Laser Photonics Rev.* **10** (2016), [10.1002/lpor.201600112](https://onlinelibrary.wiley.com/doi/10.1002/lpor.201600112)
- Gyrotropy destroys the singularity — *Phys. Rev. Materials* **4**, 055203 (2020), <https://link.aps.org/doi/10.1103/PhysRevMaterials.4.055203>

**Coating phase (§7.3)**
- Tikhonravov, Baumeister & Popov, *Appl. Opt.* **36**, 4382 (1997), [10.1364/AO.36.004382](https://doi.org/10.1364/AO.36.004382)
- Bechhoefer, *Am. J. Phys.* (2011) — Kramers-Kronig/Bode and the zero, <https://www.sfu.ca/chaos/assets/papers/2011/KK-Bode-MeaningZero.pdf>
- Grosse & Offermann (1991), [10.1007/BF00323731](https://doi.org/10.1007/BF00323731)

**Anisotropic interfaces (§7.4)**
- Berreman, *JOSA* **62**, 502 (1972), [10.1364/JOSA.62.000502](https://doi.org/10.1364/JOSA.62.000502)
- Lekner, *JOSA A* **40**, 722 (2023) — azimuth-odd cross terms, [10.1364/JOSAA.40.000722](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-40-4-722)
- Lekner (1991) — closed-form uniaxial, [10.1088/0953-8984/3/32/017](https://doi.org/10.1088/0953-8984/3/32/017)
- McClain, Hillman & Chipman, *JOSA A* **10**, 2371 & 2383 (1993) — optical activity in a ray tracer
- Passler & Paarmann, *JOSA B* **34**, 2128 + **erratum 36**, 3246, [10.1364/JOSAB.34.002128](https://doi.org/10.1364/JOSAB.34.002128)
- GeneralTmm — <https://github.com/ardiloot/GeneralTmm> (MIT)

**Gratings (§7.5)**
- Moharam & Gaylord (1981/1995); Li, *JOSA A* **13**, 1870 (inverse rule); Li, *JOSA A* **13**, 1024 (S-matrix)
- Lalanne & Morris, *JOSA A* **13**, 779 — convergence
- meent — <https://github.com/kc-ml2/meent> (MIT, active)
- Zemax↔Lumerical RCWA link — <https://support.zemax.com/hc/en-us/articles/6367505128979>

**Scatter (§7.1)**
- Krywonos, Harvey & **Choi**, *JOSA A* **28**(6), 1121 (2011) — GHS, [10.1364/JOSAA.28.001121](https://doi.org/10.1364/JOSAA.28.001121)
- Stover et al., *Proc. SPIE* **9961**, 996102 (2016) — ABg ⇐ Rayleigh-Rice
- Stover, *Optical Scattering* PM224, [10.1117/3.975276](https://doi.org/10.1117/3.975276)
- BRO-PN-1157 — importance sampling / the 46,000-year argument, <https://lavinia.as.arizona.edu/~minimaestro/MAESTRO/Construction/References/bropn1157_straylight.pdf>

**Edge diffraction (§7.6)**
- Du et al., *Electron. Lett.* (2023) — UTD ≡ knife-edge ≡ Kirchhoff, [10.1049/ell2.13014](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ell2.13014)
- Kouyoumjian & Pathak, *Proc. IEEE* **62**, 1448 (1974)
- Worku & Gross, *JOSA A* **36**, 859 (2019) — Gaussian basis vs step discontinuity, [10.1364/JOSAA.36.000859](https://doi.org/10.1364/JOSAA.36.000859)

**Surfaces & optimization (§7.7, §8)**
- Forbes, *Opt. Express* **15**, 5218 (2007); **18**, 13851 (2010); **18**, 19700 (2010)
- ISO 10110-12:2019 §4.3.2.2 (Q-bfs / Q-con); ISO 10110-19 (freeform Q-2D)
- prysm — <https://github.com/brandondube/prysm> (MIT)
- Spencer, *Appl. Opt.* **2**, 1257 (1963) — DLS ≡ Levenberg-Marquardt, [10.1364/AO.2.001257](https://doi.org/10.1364/AO.2.001257)
- Kuper & Harris, *Proc. SPIE* **1780** (1993) — Global Synthesis
- **Seger et al., *Opt. Express* **33**, 3054 (2025)** — AD of ray–surface intersection: full Jacobian at the cost of the primal trace, [10.1364/OE.546049](https://doi.org/10.1364/OE.546049)

**Zemax parity (§12)**
- OpticStudio feature comparison chart (RCWA rows) — <https://downloads.zemax.com/zemax-portal/os/files/OpticStudio-Feature-Comparison-Chart.pdf>
- Sequential 1D RCWA grating DLL (Premium/Enterprise) — <https://community.zemax.com/code-exchange-10/dll-user-defined-surface-sequential-rcwa-1d-grating-3980>
- **"OpticStudio does not currently model the optical effects of non-linear crystals"** — <https://community.zemax.com/got-a-question-7/how-to-model-harmonic-generation-1000>
- NSC coherent detector limits ("beyond the scope of the ray model") — <https://community.zemax.com/got-a-question-7/nonsequential-mode-coherent-optics-problem-with-detector-rectangle-and-nsdc-operand-240>
- Orthogonal Descent vs DLS for noisy problems — <https://community.zemax.com/got-a-question-7/od-vs-dls-464>
- Polarization-sensitive (Mie) scattering — <https://support.zemax.com/hc/en-us/articles/1500005576902-Polarization-sensitive-scattering-in-OpticStudio>
- Jones Matrix surface (loses E_z) — <https://optics.ansys.com/hc/en-us/articles/43071140222099-How-to-use-the-Jones-Matrix-surface>
- Laville & Aymard (2026), MC tolerancing draw counts — arXiv:2607.03067

**FOSS landscape (§12A)**
- Optiland — <https://github.com/optiland/optiland> (MIT)
- prysm — <https://github.com/brandondube/prysm> (MIT)
- meent — <https://github.com/kc-ml2/meent> (MIT)
- GeneralTmm — <https://github.com/ardiloot/GeneralTmm> (MIT)
- Poke — <https://github.com/Jashcraf/poke> (BSD-3)
- RayOptics — <https://github.com/mjhoptics/ray-optics> (BSD-3)
- KrakenOS — <https://github.com/Garchupiter/Kraken-Optical-Simulator> (**GPL-3 — read, do not link**)
- S4 — <https://github.com/victorliu/S4> (**GPL-2, dead**)

---

## Changelog

| date | change |
|---|---|
| 2026-07-15 | Initial. Supersedes `engine.md`: BREP cancelled (§2.2), the Amdahl ceiling corrected (§2.1), the gather identified as the dominant cost and its fp64 diagnosis established (§1.2, §6), sequential mode confirmed as the keystone from the code (§1.4). |
| 2026-07-15 | **Four premises overturned by research, recorded rather than quietly dropped:** (1) **PRT 3×3 is already implemented** — `s_hat` is a 3D global vector and `rotate_jones` is the PRT transport; the real gap is the Q matrix (§4.1). (2) **ABg *is* Harvey-Shack**; the real gap is importance sampling (§7.1). (3) **Conical refraction is not a ray phenomenon** and is irrelevant to this engine's crystals — do not build; fix the `_DEGEN` guard instead (§7.2). (4) **Zemax DOES ship RCWA** (Premium 1D DLL, Enterprise Lumerical 2D) — a gap, not a lead (§12.1). Added §12 (parity + effort), §12A (licensing), Forbes Q-type and importance sampling to Tier 1 (§1.3). |
