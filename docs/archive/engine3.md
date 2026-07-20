# engine3.md — the engine overhaul: physics first, speed second, reality as the referee

Status: **approved design**. Supersedes `engine2.md` (kept as a research record; its
findings were independently re-verified against the code — see §2 for the short list of
corrections) and `engine.md` (superseded twice over; BREP stays cancelled).
Companions: `docs/RAYTRACER.md` (physics reference), `cengine/README.md`.

This document is the synthesis of `engine2.md`, a ~45-claim line-by-line verification of
it against this repository (2026-07-16), a re-audit of every proposed dependency, and the
owner's locked decisions (§1.2). Every quantitative claim herein was either verified this
session against the code/benchmarks or is explicitly marked with the gate that will
verify it.

The question it answers: **how do we rebuild this engine to be as physically true as
possible, running as fast as possible on this machine (RTX 4090 Laptop, 16 GB, SM 8.9),
with simulations in the hours-to-week range, a quick-pass-before-commit workflow, and an
architecture where future physics is added to a seam instead of a rewrite — validated
against the optical bench, not against other simulators?**

---

## 0. Do these now, regardless of everything else

Small, independent, all verified against the code. None changes any physics. All land
directly on `master` behind the existing test suite.

### 0.1 The selection "spiderweb" and the faceted look are two display bugs

Verified at `mieworkbench/widgets/vtkview.py:751-754`: selection turns on
`EdgeVisibilityOn()` with `_SELECTED_EDGE_COLOR = (0.10, 0.10, 0.10)` at 2 px — every
STL triangle edge in near-black over the orange face (`_SELECTED_COLOR` at :78 already
recolors the face, so the edges are redundant). Refining the tessellation multiplies web
lines ~56× — the "mesh it finer" instinct is exactly backwards. Fix: delete/condition
the three lines, or `vtkFeatureEdges` for silhouette only.

Separately, `vtkview.py:663-666` wires `vtkSTLReader → vtkPolyDataMapper` directly;
`grep -rn "PolyDataNormals|FeatureAngle|CleanPolyData" mieworkbench/` returns zero hits
(verified). Every curved surface is flat-shaded off per-facet STL normals. Fix (~5
lines, cached per `shape_key`): `vtkCleanPolyData` (STL vertices are unshared — merge
first) → `vtkPolyDataNormals(FeatureAngle=60, SplittingOff, ComputePointNormals)`.

### 0.2 The runtime estimator is wrong in its *formula*, not its calibration

`common.py:924-926` (verified):

```python
nsamples = max(1.0, rays * 0.5)          # bug 2
gather_ops = (nsamples * npix * max(1, n_coherent_sources) * nlambda
              * max(1, n_pol_strata))    # bug 1
```

- **Bug 1:** gather keys `(source, λ-stratum, pol-stratum)` **partition** the samples;
  they do not duplicate them. `cengine/src/gather.c:617` counts `total_pairs += n_sel *
  Q` — the code itself bills Q × M. The `nlambda × n_pol_strata` factor is spurious.
- **Bug 2:** "half the primaries survive" — a measured interferometer yields ~6
  samples/ray (both arms recombine; children survive).

Measured on `results/example/quick-phase0` (verified in `case.json`): gather predicted
715.4 s vs 87.1 s actual (8.2×) — **and the trace was worse: 590 s predicted vs 26.4 s
actual (22×)**, a fact engine2.md under-reported. An important correction to engine2's
framing: `common.py` **already has a calibration layer** (`calibrated_rate()`,
`RESULTS_DIR/.calibration.json`, fallbacks at `common.py:880-881`) — the fix is a
corrected *law* feeding the existing calibration, not a constant:

```
gather_s ≈ t_init + Q · M_total / rate_gather      (Q = resolution², strata cancel)
trace_s  ≈ rays_total / rate_trace                 (rays_total incl. children)
```

validated blind by engine2 to −0.076 % on a withheld benchmark. `t_init` ≈ 1.9 s CUDA
context (measured: five non-coherent scenes report gather_s = 1.91–1.94 s — a floor, not
work). Each completed case writes its measured rates back through the existing
calibration file. **nlambda does not enter.** This estimator becomes load-bearing in the
per-run accuracy dialog (§10.3), so "within ~20 % on the benchmark set" is its gate.

### 0.3 Free gather wins (physics-neutral, bit-identical output)

Verified in `cengine/src/kernels/gatherk.h:42-64`, with one correction to engine2 §6.2:

| defect | fix |
|---|---|
| `sqrt(r2)` then `1.0/r` — two fp64 MUFU Newton chains | one `rsqrt`, then `r = r2·inv_r` |
| `fmod(x, 2π)` — ~27 % of the kernel | `x − 2π·trunc(x·inv2π)`; the compiler cannot do this (`fmod` is IEEE-exact by contract) |
| `k·opl_i` recomputed per pixel | precompute per sample. **Correction to engine2:** the `fmod` is over the *combined* `k·(opl + n·r)` (gatherk.h:57) and cannot be hoisted — only the raw product can. This saves a multiply, not the fmod chain. |

Net expectation ~1.5–2× (dominated by the fmod strength-reduction), verified by an
A/B harness. Gate correction (implementation finding, 2026-07-16): strict bit-identity
is not achievable — `rsqrt` and the trunc-based reduction are not IEEE-identical to
the ops they replace — so the honest gate is **fp32-indistinguishable**: detected
powers bit-identical, cube max deviation at the fp32-epsilon level (measured ≤8e-8 of
peak), pinned visibility unchanged. The k·opl precompute was dropped: it saves one FMA
per pair (the fmod chain it was thought to remove is not separable), not worth buffer
plumbing.

### 0.4 Add the LICENSE file: MIT

The repo has no LICENSE (verified) — legally all-rights-reserved, which contradicts the
sharing intent. Decision (owner, tentative but sufficient): **MIT**, now. Every
dependency in §14 is compatible; MIT preserves every later option including relicensing
and closed builds.

### 0.5 Correct the repo's own documentation

| location | claim | actual (verified) |
|---|---|---|
| `CLAUDE.md`, `RAYTRACER.md` §13, `cengine/README.md`, **and `cengine/src/rng.h:17-19`** | parity bar "1e-12 deterministic / 3-seed ±max(3σ,1%)" | `test_cengine_parity.py:101`: `tol = 1e-9 if deterministic else 0.02`, single seed. No 3-seed harness exists anywhere (the only `max(3σ,1%)` code is `test_workers.py`, unrelated). 1e-12 applies to `emitted_W` only. |
| `CLAUDE.md` | "Philox … lineage-keyed" implied engine-wide | Python engine is `np.random.default_rng` (`tracer.py:166`). Philox is C-only. Python↔C was never bit-exact. |
| `CLAUDE.md` | "eleven classic-system demos" | 34 `demos/*.MieWB`; 15 baseline *designs* (30 JSON files — placements+power pairs). The *benchmark* set is 11. |
| `common.py:924` | estimator | §0.2. |

---

## 1. Verdict, strategy, and priorities

### 1.1 The finding that reorders everything (verified)

**The gather is the entire performance story; the trace is not.** Gather cost is
`Q × M_total` (resolution² × surviving samples; strata cancel). Because Q and M both grow
with preset:

| preset step | rays | Q | trace cost | gather cost |
|---|---|---|---|---|
| quick → normal | ×10 | ×16 | ×10 | **×160** |
| normal → detailed | ×10 | ×4 | ×10 | **×40** |

The gather dominates harder the more accuracy you ask for. Verified from
`cengine/BENCHMARKS.md` (RTX 4090 Laptop, CUDA 13): only the interferometers go
multi-hour — michelson_folded is 713.4 s of gather against 1.6 s of trace at 2e5 rays;
projected ~39.6 h at `detailed`, ~99.9 % gather. Every non-coherent scene shows a fixed
~1.9 s gather floor (CUDA init, not work — five scenes reading 1.91–1.94 s identically).
The structural reason: an interferometer recombines both arms on one screen (~5.7
surviving samples/ray) and every sample meets every pixel.

**A tracer rewrite is aimed at 0.2 % of the problem in the only scenes that hurt.** The
CUDA gather kernel is not badly written — engine2's SASS analysis showed it saturating
~100 % of Ada's fp64 instruction-issue rate. **fp64 is the mistake** (Ada does fp64 at
1/64 of fp32 — verified, CC 8.9: 128 vs 2 FMA/clk/SM), and the kernel computes obliquity
and 1/r in fp64 only to cast them to `float` on the next line (gatherk.h:58, verified).
Only the phase needs fp64. §6 is the fix, and it converts directly into accuracy (§6.4).

### 1.2 Strategy and owner decisions (locked 2026-07-16)

**Selective dominance, not parity.** Be the best tool in existence for coherent
diffraction, NLO, time-domain, and stray light inside a full CAD scene — the moat Zemax
structurally cannot cross (their own staff: non-sequential coherent detection is
*"beyond the scope of the ray model"*; *"OpticStudio does not currently model the
optical effects of non-linear crystals"*). Be *adequate* at classical sequential design
by adopting an OSS kernel. Matching Zemax outright is ~110–190 engineer-weeks (§13) and
pointless.

The decisions that shape everything below:

1. **Sequential kernel: adopt Optiland (MIT)**; parity oracle built *first* (§5).
2. **Bench validation is a first-class deliverable** (§12): power meter/photodiode,
   camera/beam profiler, and spectrometer exist on the owner's bench today;
   interferometers, imaging systems, and spectrometers are the systems that will
   actually be built. NLO bench work comes later.
3. **Virtual instrument layer** (§9): every real instrument class gets a parametrized
   virtual twin — including instruments *not yet owned* — in both `full` (noise) and
   `ideal` (response-only) modes, so any future acquisition compares 1:1 on day one.
4. **Run contract** (§10): checkpoint/resume and additive ray extension are engine
   contract requirements from the start; accuracy-vs-time is an explicit per-run choice.
5. **Headless-first** (§16): long runs happen over SSH; every feature has CLI parity
   *and* a GUI surface, and the GUI surface is proven by offscreen UI simulation.
6. **NLO/time-domain gets fast in a later phase** — the registry is designed so the port
   is fill-in work; the physics stays correct-but-Python-speed meanwhile.
7. **Full anisotropy (Berreman 4×4) is an option, last** — but its seam is designed in
   the core round so it integrates without a rewrite (§7.4).
8. **Fluorescence and GRIN**: designed-for seams, implemented later (§4.2). Coherent
   curved detectors: deprioritized, likely never.
9. **This 16 GB laptop is the target**; VRAM is a budget with mandatory headroom (§16).
10. **Cadence**: quick wins and additive features merge to master freely behind gates;
    the core rewrite lives on a long branch merged at functional equivalence (§15).

### 1.3 Priority ranking

Two independent bottlenecks: coherent scenes are gather-bound; stray-light scenes are
ray-budget-bound. Physics-correctness items rank by silent-wrongness first.

**Tier 1 — high value, small effort, first:**

| # | work | why | § |
|---|---|---|---|
| 1 | Gather precision architecture | the only lever on the multi-hour scenes; 20–60×; converts to accuracy | 6 |
| 2 | Scatter importance sampling + BTDF | unblocks stray light (the 46,000-year argument); multiple scatter then ~free | 7.1 |
| 3 | Q matrix; coating phase columns + phase-invalid flag; conical-point guard | three small independent fixes; two remove *silently wrong* answers | 4.1, 7.3, 7.2 |
| 4 | Forbes Q-bfs/Q-con via prysm | closes an ISO-normative gap cheaply | 7.6 |
| 5 | Run contract: checkpoint/extend + honest estimator | required for week-scale runs; cheap now, painful to retrofit | 10 |

**Tier 2 — larger, each unlocks a class of work:**

| # | work | § |
|---|---|---|
| 6 | Virtual instrument layer (early — enables the bench program) | 9 |
| 7 | Dispatch registry + one-source/two-target core (the long branch) | 4.2–4.4 |
| 8 | Sequential mode via Optiland + DLS + AD Jacobians | 5, 8 |
| 9 | Prescription-primary data model | 3 |
| 10 | Lekner-1991 closed-form uniaxial Fresnel | 7.4 |
| 11 | RCWA tables via meent | 7.5 |
| 12 | NLO/time-domain port to the fast core | 15 P7 |
| 13 | Element realism (iris leaves, mounts, figure error); exact prescription display | 11 |

**Optional, last:** full-anisotropy Berreman 4×4 (absorbing/gyrotropic) against the §4.2
seam — owner-requested as a completeness option, not a need.

**Do not build:** conical refraction (§7.2 — not a ray feature; irrelevant to
phase-matched crystal cuts; gyrotropy destroys the singularity in KTP anyway); UTD edge
diffraction (§7.7 — the gather already computes it); KK phase retrieval (§7.3 — provably
unsound); BREP tracing (§3 — the motive dissolved under prescription-primary; the
"geometry looks bad" complaint was the §0.1 display bug).

### 1.4 Sequential mode is the keystone — confirmed from the code

The optimizer and tolerancer already exist and are well-built (verified: `optimize.py`
554 LOC, `tolerance.py` 562 LOC with sensitivity + MC yield + per-draw compensator,
`fast_eval.py` 592 LOC, two GUI panes; operands cover spot RMS, encircled energy, MTF50,
detected power, arbitrary `report.json` keys). The gap is **cost per merit evaluation
and the existence of gradients**: even `fast_eval`'s fast path re-extracts and runs a
full MC trace per evaluation (it patches sources to `coherent=false` so the gather never
runs — verified `fast_eval.py:48-51` — but the trace remains), and the optimizers are
derivative-free (scipy Nelder-Mead at `optimize.py:351`, nevergrad CMA-ES at `:364`).
MC speckle noise makes finite-difference gradients garbage *permanently* on the
non-sequential path; a deterministic sequential trace makes DLS and exact autodiff both
possible. Nothing else unlocks as much per unit effort.

The hard part is smaller than it looks: `train_solver.py`'s anchored/chained optical
train **is** a sequential surface ordering — `solve_chain()` returns a topologically
ordered element list with vertex-to-vertex distances along the beam (verified). The
bridge from a `miewb_train_*` recipe to an Optiland surface list is a mapping, not an
architecture.

---

## 2. Corrections log

### 2.1 To engine2.md (found by this verification pass)

| engine2 claim | correction |
|---|---|
| §6.2: "`fmod(k·opl_i, 2π)` is loop-invariant — hoist to host" | The `fmod` is over the combined `k·(opl + n·r)` (gatherk.h:57); `r` varies per pixel, so the fmod cannot be hoisted. Only the raw product `k·opl_i` is precomputable — a multiply, not a MUFU chain. |
| §0.2 implied a bare-constant estimator | A calibration layer already exists (`calibrated_rate`, `.calibration.json`); the *formula* is what's wrong. The trace estimate was also off (22× on quick-phase0), not just the gather. |
| "15 committed baselines" | 15 baseline *designs* = 30 JSON files (placements + power pairs). |
| Parity-bar overstatement located in three docs | Also in `cengine/src/rng.h:17-19` — a fourth site for the §0.5 sweep. |
| poke "current" | Last pushed 2025-05 — stale. "Read the algorithms, do not depend" stands, more strongly. |
| GeneralTmm as "Berreman 4×4 … best for embedding" | It implements the Hodgkinson–Kassam–Wu (1997) eigen-formulation (equivalent class, not literally Berreman) and is an 8-star Cython project. Downgraded to *reference oracle*; the in-house 4×4 (§7.4) is ~500 LOC. |
| FMMAX "MIT, archived on both forks" | facebookresearch fork is clean MIT; the invrs-io fork's license resolves to NOASSERTION. Both archived. Moot — meent is the choice. |
| torch gather unmentioned in the precision plan | The Python engine already has a CUDA gather (`gather.points_torch`, verified). The precision architecture lands in the C-CUDA kernel **only**; the torch gather follows the existing sunset roadmap (`cengine/README.md`). One precision effort, not two. |
| §6.3 tile error "O(δp³/R²)" stated without a control law | Made explicit in §6.3: adaptive tile sizing against a phase-error budget; at R = 10 mm a fixed 64-px tile is ~0.5 rad — unacceptable. |

Everything else in engine2.md checked out — including every file:line citation tested,
the PORTED token list (20 tokens verified verbatim), the fp64 diagnosis, the Q×M law,
the PRT-equivalence finding, the xfail physics gaps, and the LOC counts (14,554 engine /
19,006 engine-test / 23,184 GUI — all exact).

### 2.2 To engine.md

Carried from engine2 §2, still correct: the "~3.3× Amdahl ceiling" was an artifact of
benchmarking at 1e6 rays where the ~2 s driver floor is 4–30 % of wall; the
gather-moving options are *not* "new physics approximation risk" (bit-exact at tile=1);
BREP dissolves under prescription-primary. One engine.md idea is **reinstated** here
that engine2 dropped: the standalone/persistent driver (§10.4) — the ~2 s Python import
+ ~1.9 s CUDA init per case is real money in sweeps and optimizer loops.

---

## 3. Data model: prescription-primary

**The optical prescription is the truth; the CAD is a view of it.** Today this is
inverted: CAD is primary and `surface_override` is the escape hatch, verified to
`ASPHERE_TOL_M = 1e-6` (`extract_geometry.py:133`) — 1 µm ≈ 2 waves at 500 nm, so for
aspheres the override already *is* the phase-valid truth. Formalize it.

| role | truth | representation | phase |
|---|---|---|---|
| optical surfaces | prescription | analytic, closed-form intersection | exact, always valid |
| mechanical geometry | CAD | triangle mesh, BVH | never needed (incoherent scatter) |

Surface catalog (authored in basic terms, converted internally, re-expressible without
re-authoring): conic; even/odd asphere; **Forbes Q-bfs/Q-con**; Zernike sag; XY
polynomial; grid sag (measured maps, §11.3); plane/cylinder/cone/torus; GRIN as a
*medium* prescription (seam, §4.2). `.MieWB`/`.MieSim` gain `prescription.json`;
`model.FCStd` remains the mechanical/visual representation generated from it for optical
elements; existing files keep opening; the FreeCAD round-trip is preserved.

OCCT survives host-side only, at load time: import mechanical CAD (STEP/`.brp`) → mesh;
fit prescriptions out of imported optical geometry (sphere → exact radius; BSpline →
attempted asphere/Forbes fit with reported residual, silent refusal never). Pin
8.0.0_p1; per-thread adaptor copies (the eval-cache mutation-on-read hazard is
undocumented-unfixed); this is the one C++-facing dependency and it stays out of the
trace path.

---

## 4. Physics core architecture

### 4.1 Polarization: it is already PRT. Add the Q matrix; do not rewrite.

Verified: the engine carries `s_hat` as a 3D global vector (`rays.py:75`) and
re-expresses amplitudes between bases by explicit 3D dot products
(`fresnel.py:139-153` `rotate_jones`), with the normal-incidence singularity handled
(`fresnel.py:114-136`). That *is* the Yun/Chipman PRT transport — the 3×3's third
dimension lies along k̂ where the transverse field is zero; the map PRT ⇔ Jones+basis is
invertible. Geometric (Pancharatnam) phase and skew aberration emerge from correct basis
transport. Zemax itself uses two-pass s/p Jones.

The real gap (Yun/McClain/Chipman Paper II): **P alone conflates physical retardance
with geometric rotation.** Add the parallel-transport matrix **Q** (same construction
with J → I); honest retardance/diattenuation maps come from M = QᵀP. Small additive
module. Partial polarization: sum coherency outer products `(Ps)(Ps)†` per gather key —
the key structure already enforces "coherent subsets sum as fields first"; the detector
side already consumes 3-vectors (`gather.py:500-501`, verified). Report N with DOP (it
is biased high at low N).

### 4.2 The interaction registry — the extensibility contract

Today's dispatch is a hardcoded chain (verified: `tracer.py:525-555` +
`_optic_children` at `:716-1159`, ~442 lines of nested `if`), and C-routing is gated by
the `PORTED` frozenset where **a feature that forgets to emit a token silently skips its
physics** — which has already happened once (P8 NLO, `cengine.py:104-109`, verified).
A mechanism whose failure mode is "silently wrong" is unacceptable in a physics engine.

Replace both with a registry. An **interaction** is a self-contained unit declaring:

- **token** — its feature name (the extractor emits it; the scene carries it);
- **match** — what triggers it (body property pattern, face map, medium);
- **contract** — parent ray + interface/segment state → child rays (Jones vectors,
  medium-stack ops, differential fate) + **ledger deltas** (which of the 9 buckets, or
  none); energy bookkeeping is part of the type, not a convention;
- **implementations** — per backend, from one source (§4.3);
- **oracle** — the validation test that pins it (registration without a pinned test
  fails CI).

Construction-time rule: a scene whose feature set contains a token with no
implementation on the *active* backend is a **hard error** — routing falls back (under
`auto`) or refuses (under a forced engine), never skips. This is the end of the PORTED
silent-skip class.

Two seam *kinds*, both designed in the core round:

1. **Surface interactions** (everything in `_optic_children` today, plus future:
   fluorescence at a phosphor face, exact-uniaxial SHG walk-off, full-anisotropy
   Berreman §7.4).
2. **Volume propagators** — the segment between surfaces is itself dispatchable:
   homogeneous (today's closed form), absorbing/scattering continuum (today's
   particles), **GRIN** (RK4 ray ODE + fp64 OPL integration — the accuracy-critical
   term for the coherent gather is `d(OPL)/ds = n`, so first-order integrators surface
   as phase noise), thermal-gradient index, and time-dependent media. GRIN and
   fluorescence get registry stubs + named oracles in the core round (GRIN: Luneburg /
   Maxwell fish-eye analytic foci; fluorescence: ledger closure with λ-shifted
   emission), so the later work is fill-in, not design.

### 4.3 One source, two targets

Write the physics once; compile for CUDA device and host CPU. **This extends a proven
in-repo pattern, not an invention**: `kmath.h:24-28` defines `KFN` as
`__host__ __device__ static inline` under `__CUDACC__`, and `gather_pair` is already
called from both `gather.c` (OpenMP) and `cuda/gather.cu` (verified). Extend from the
1,509 LOC of `kernels/` to the whole interaction set. The CPU build is the debug
oracle: single-steppable, runs anywhere, no GPU — which also serves the headless/remote
requirement. This permanently kills the dual-implementation problem: 16 of 36 feature
tokens are Python-only today (verified list in `cengine.py`), i.e. the entire recent
roadmap can never be fast until this lands.

### 4.4 Language

**C++17 for host-side orchestration; C-style header kernels for physics.** Rationale
(owner-ratified): compilers generate identical code for equivalent C and C++ — there is
no speed axis here — so readability and line count decide. The kernels stay exactly the
proven `KFN` C-style pattern (they must compile as CUDA device code anyway, and they
port verbatim). The host side — scene construction, the registry, buffer management,
checkpointing — is where RAII, `std::vector`, and a typed registry genuinely shrink the
code and delete manual-memory bug classes. No exotic C++ (no exceptions across the
kernel boundary, no virtual dispatch in hot loops — the registry resolves to flat
function tables at scene build).

### 4.5 Preserved verbatim (all verified correct today)

- `opl = 0` on the emitting surface — the emitting surface IS the reference wavefront.
- fp64 phase, `mod 2π` **before** any fp32 trig (`gather.py:27-30`, `:158-161`).
- The 9-bucket energy ledger, 1e-3 closure gate, exit 3 on failure (`audit.py`,
  `run_trace.py:1095`). Detected power stays a diagnostic, never a closure bucket.
- Boundary-flux tally (in − out = absorbed), diagnostic side-table only.
- `(source, λ-stratum, pol-stratum)` gather keys — the mutual-coherence equivalence
  classes.
- 4-deep LIFO medium stack; overflow/pop-mismatch are hard errors.
- Lineage-keyed Philox (`cengine/src/rng.h`, verified) — adopted **engine-wide** in the
  new core (it is what makes §10.2's additive extension possible); `child_slot` values
  append-only; KATs carry over. No attempt to match numpy streams; the Python engine
  remains statistical-only relative to the core, as it always was.
- `t_eps = 1e-7` self-hit guard; no last-face exclusion (`scene.py:913-915`).

---

## 5. Sequential and non-sequential modes

Two traversal strategies over one physics core, one surface catalog, one material
library — the Zemax/CODE V architecture:

| | sequential | non-sequential |
|---|---|---|
| geometry | ordered surface list (from the train recipe) | full 3D scene, BVH |
| rays | ~10²–10³ deterministic, aimed | 1e5–1e7 Monte-Carlo |
| cost | microseconds | seconds–hours |
| use | design, optimization, tolerancing, aberration analysis, exact preview | stray light, ghosts, coherent recombination, scatter — the default |

**Adopt Optiland (MIT) as the sequential kernel.** Verified 2026-07-16: active
(pushed 2026-07-15), and its optimizer inventory is better than engine2 knew —
`least_squares` (the DLS class), **`orthogonal_descent`** (Zemax's own recommendation
for noisy merit functions — directly usable on our MC path), `basin_hopping`,
`differential_evolution`, `dual_annealing`, `shgo`, `glass_expert` (scipy), CMA-ES +
particle swarm (custom), Adam/SGD (torch autograd). Plus paraxial/Seidel/Zernike/MTF,
MC tolerancing, and `.zmx`/`.seq`/`.len` import. Non-sequential is roadmap-only for
them — which is exactly the asymmetry: **take theirs (sequential), keep ours
(non-sequential coherent), and never confuse the two.**

**The parity oracle comes first — non-negotiable ordering.** Optiland vs the MC engine
on shared `demos/` scenes, gated like `test_cengine_parity.py`. Two engines without an
arbiter is the largest architectural risk in this document, and it is exactly the risk
this repo's one-solver culture (`train_solver` pinned at 1e-9; cengine parity) exists to
prevent. The train-recipe → surface-list mapping and the unit contract are the real
integration work.

Sequential mode also replaces the drift-prone 3-process preview chain
(`core/raypreview.py`, verified): preview becomes exact, instant, and the same physics
as the run.

---

## 6. The gather — the dominant cost, and an accuracy budget

### 6.1 What it is, and where it stands

Rayleigh–Sommerfeld-I over ray samples (verified `gather.py:2-16`):

```
E(p) = (1/(iλ)) Σ_i E_i √(dA_i) K_i exp(i k (opl_i + n_amb r_ip)) / r_ip
K = clip(0.5(cosθ_prop + cosθ_det), 0, 1),  K = 0 for back-radiation
```

This is exactly Zemax's Huygens PSF — their reference-grade method, whose *"only
disadvantage … is speed"* (their words), and their FFT PSF is the approximation with
F/# and field-angle validity limits. On the coherent path we are not behind; we are
doing the reference-grade thing, slowly, in a non-sequential scene where they cannot do
it at all.

### 6.2 Precision architecture (the 20–60×)

Only the phase needs fp64. **Tile-reference factorization**: tile the detector; per tile
centre `p₀` and sample `i`, compute `R_i = |p₀ − s_i|` and `k·R_i mod 2π` **once in
fp64** (preserving the load-bearing reduction contract), then for pixel offsets `δp`
within the tile:

```
r ≈ R_i + u_i·δp + (|δp|² − (u_i·δp)²)/(2R_i),   u_i = (p₀ − s_i)/R_i
```

The residual phase `k·(r − R_i)` is computed *as the expansion terms directly* (never as
a difference of large numbers — that would be catastrophic cancellation) in fp32, then
fp32 `__sincosf`. With a 64² tile the fp64 work is amortized 4096×.

**Implementation upgrade (landed 2026-07-16, supersedes the Taylor plan above):** the
residual uses the **exact algebraic identity** `r − R = (2R(u·δp) + |δp|²)/(r + R)` —
valid for any stored (R, u) with |u| ≈ 1, so there is **no truncation term at all**;
the error is fp32 representation/roundoff only, bounded by ~5e-7·k·δp_max and
**reported per key** (`gather.json: phase_err_bound_rad`). Tile size adapts per
detector to keep the bound inside a 1e-3 rad budget at the detector's shortest λ;
samples within 8×δp_max of a tile route through the exact fp64 kernel (counted:
`near_exact_pairs`). `--gather-exact` selects the plain fp64 kernel — verified
byte-identical to the pre-change baseline — as the permanent reference path. Measured
(MD co-tenant, indicative): michelson_folded gather 713.4 s → 62.0 s (**11.5×**,
8.16e10 pairs/s); doubleslit detected powers bit-identical, cube deviation 1.0e-3 of
peak against a 5.8e-4 rad reported bound, visibility 0.9607 unchanged. Gate pinned in
`test_gather_tiled.py`. The `detailed`-preset michelson <1 h target rides on the NUFFT
path (§6.3) — collimated interferometer arms are its design case.

The first-order term `u_i·δp` is bilinear — a GEMM, exactly the phase-added stereogram
of the CGH literature. If profiling shows the fp32 inner loop shy of peak, tensor-core
(TF32) tiles are a follow-on inside the same error budget — noted, not promised.

This lands in the **C-CUDA kernel only**; the torch gather follows its existing sunset
roadmap. The CPU (OpenMP) build gets the same factorization from the same source (§4.3).

### 6.3 NUFFT angular-spectrum fast path — CLOSED as a route for MC gathers
### (implementation finding, 2026-07-16; infrastructure retained opt-in)

The premise above ("exact to quadrature + NUFFT tolerance" with the k-grid sized to
the geometric NA) is **wrong for Monte-Carlo point-sample gathers**, and this section
records why so it is not re-attempted: MC Huygens samples are ideal point emitters
whose plane-wave decomposition spans the **entire propagating disc** |k_t| ≤ k
regardless of detector geometry. A k-grid truncated to the geometric space-bandwidth
product band-limits each spherical wave and leaves an **irreducible Gibbs floor**,
measured ~8e-2 of peak — insensitive to sample count (2e3→2e4: 0.087→0.066) and to
4×→8× band over-coverage, the signature of the mechanism. Exactness requires the full
Nyquist disc `N_k = (2L/λ)²` — precisely the 1e10-DOF wall of §6.4. This violates
same-physics-on-every-path, so: the route ships **off by default** behind
`--gather-nufft`, its runtime gate (separating plane, obliquity-separability bound,
VRAM budget via cudaMemGetInfo, cost model) rejects real scenes with reasons logged in
`gather.json`, and cuFINUFFT v2.5.1 remains an optional build-time fetch. The
infrastructure is correct and reusable for **band-limited fields** (POP-style beam
propagation, §8/P4b+), where the white-spectrum problem does not arise. Consequence
for the P1 wall-time target: `detailed` michelson rides on the tiled kernel
(~11.5× today; projected ~3 h from 39.6 h) plus SFU-level kernel tuning and idle-GPU
headroom — the <1 h figure is no longer promised by this section.

### 6.4 The speedup is spent on accuracy — per-run choice

The gather is a Monte-Carlo estimator of a wave field at ~1e-4 of Nyquist sampling; the
`M_eff ≥ 1000` gate, mandatory jitter, and the 1/M_eff speckle pedestal (all verified in
`gather.py`) exist because of that. Gather cost is linear in samples, so 30× faster is
30× more samples is a real reduction of the speckle floor. **Owner decision: the run
dialog always asks** — accuracy (more rays, same wall) vs speed (same rays, shorter
wall) — with the corrected §0.2 estimator's predicted wall time and the projected
M_eff/pedestal shown. During development, the fastest settings are the default posture
(§16.4); the long runs come at the end.

### 6.5 Rejected, with reasons (verified reasoning, carried)

FMM/directional FMM (at 1e-4 of Nyquist the boxes hold <1 sample — pure overhead; no
GPU library); Gaussian-beam decomposition (documented failure on hard apertures in the
near field — precisely the pinned double-slit invariant; paraxial; composes badly with
Jones/medium-stack); Fresnel/type-3-direct (paraxial — changes the physics); binned
angular spectrum (nearest-cell binning ⇒ ~4.6 rad phase error — destroys fringes; only
NUFFT-quality spreading is admissible, which is what type-1 *is*). None of these change
the speckle pedestal; the cheapest verifiable option wins.

---

## 7. Physics realism upgrades

Engine2's research overturned four premises; all four reversals were re-verified and
stand: PRT is not a gap (§4.1); ABg *is* Harvey-Shack; conical refraction is not a ray
feature; KK phase retrieval is unsound. Details below only where this document adds or
changes something.

### 7.1 Scatter: importance sampling first, then BTDF, then model breadth

ABg is the Harvey-Shack shift-invariant model (Stover 2016: a direct consequence of
Rayleigh-Rice), so the model is not the gap — **the ray budget is**. BRO's arithmetic:
1e6 pupil rays × a 1°-density hemisphere ≈ 4×10¹⁴ rays ≈ 46,000 years; *"importance
area sampling is an essential feature of any serious stray light analysis program."*
Implement target-solid-angle importance sampling (equal flux per ray, BSDF averaged
over the cone, power rescaled; the bias knob explicit) — **without breaking the 1e-3
ledger**, which is the acceptance gate. Then: **BTDF** (the transmitted child at a
scattering face is untouched today — a genuine gap all four commercial tools cover);
anisotropy by ellipticizing `|β − β₀|`; GHS (Krywonos-Harvey-Choi 2011) as a
better-model follow-on where RR leaves its validity range. Multiple surface bounces then
come free by continuing to trace scattered children.

### 7.2 Conical-point runtime guard

Do not build conical refraction (Berry's BKB integral needs the incident angular
spectrum — a field-tracing feature; and the engine's crystals are phase-matching cuts
that never see the optic axis). But `birefringence.py:80` `_DEGEN = 1e-9` silently
returns an arbitrary transverse basis when k ∥ optic axis (verified, no warning) —
plausible-but-wrong output. Add a counted runtime guard: rays inside the degeneracy
cone increment an anomaly counter (§12.1's mechanism) and warn with the crystal named;
a configurable fraction fails the run.

### 7.3 Coating phase: ask the user, as Zemax does

Table coatings carry no phase and silently borrow bare-interface Fresnel phase
(verified: `materials.py:856-860` — no phase columns; `tracer.py:810-813`). Phase from R
alone is information-theoretically impossible (Blaschke all-pass factors; multilayers
are not minimum-phase — Tikhonravov 1997). Two sound fixes, both shipped: (1) accept
user-supplied phase columns — the Zemax TABLE format (`Rs Rp Ts Tp Ars Arp Ats Atp`,
phases in degrees — format verified) becomes importable; (2) fit a TMM stack to the
table (OptiLayer-style), which also yields GDD for the existing `gdd_budget` machinery.
Minimum fix ships first: flag phase-less table coatings **phase-invalid** and warn/refuse
on interferometric/ultrafast/polarimetric runs — ending a silent corruption path.

### 7.4 Anisotropic interfaces: Lekner now; full Berreman as the final option

Today's effective-index approximation is worst where `r_sp`/`r_ps` are maximal — optic
axis at ~45° azimuth (Lekner 2023: cross terms are odd in azimuth, vanishing at 0°/90°,
which is why textbook scenes look fine). Two stages:

1. **Lekner (1991) closed-form exact uniaxial** o/e amplitudes at ~Fresnel cost — the
   correct engineering answer for calcite/quartz; lands in P6. Validation: calcite
   walk-off 6.226° stays; add a 45°-azimuth interface oracle from the Lekner paper.
2. **Full anisotropy — Berreman 4×4** (biaxial, absorbing/dichroic, gyrotropic with
   g frozen per ray à la McClain 1993; quartz 21.77 deg/mm @589.3 nm as the activity
   oracle): **an option, implemented last** (owner decision), against a surface-
   interaction seam designed in the core round. In-house implementation (~500 LOC:
   build Δ(ε, k_t), eigensplit with degeneracy-robust sorting, S-matrix for thick/
   absorbing stacks), validated against GeneralTmm and pyGTM as *oracles only*
   (GeneralTmm is a niche Hodgkinson-formulation project; pyGTM is GPL — never linked).
   Dichroism note stands: inhomogeneous waves, non-orthogonal elliptical eigenmodes —
   the mode-sorting is where it will break, and the oracle set must include an
   absorbing case (Passler-Paarmann's examples).

### 7.5 RCWA tables via meent

Per-ray RCWA is impossible (O(N³) eigendecomposition; measured ~15 ms at N=101 ⇒
CPU-hours per surface per bounce). Precompute-and-interpolate is what
Zemax/Lumerical themselves do (51×51 direction grid, complex field interpolation
carrying phase and polarization — sidesteps unwrapping). The seam exists:
`grating.py:269` `order_efficiencies(spec, lam, cos_i, orders)` already takes `cos_i`
but the table branch interpolates on λ only (verified). The job: extend tables to
(λ, θ, φ, pol) with complex amplitude; adaptive refinement near Rayleigh/Wood anomaly
loci (analytic); Kogelnik kept for thick weakly-modulated near-Bragg VBGs. Library:
**meent** (MIT verified via license API, active). Two engine2 sub-claims about meent —
Li's inverse rule and S-matrix formulation — were **not verifiable from its docs**;
they become adoption-gate checks (convergence on a metallic TM lamellar case vs
published Li 1996 curves) before tables ship. Fallback if meent fails the gate: keep
the table format, generate with a different tool later — the interpolator is ours
either way.

### 7.6 Forbes Q-type (Q-bfs, Q-con)

ISO 10110-12 normative; `r⁴,r⁶,r⁸` are near-collinear over a finite aperture (ill-
conditioned optimization and untranslatable manufacturing specs); `Σa_n²` maps to RMS
slope. Port from prysm (MIT; **pin a git SHA** — PyPI is stale at 2022) into the
surface catalog with analytic derivatives; parity vs prysm at 1e-12 is the gate.
Composes with §3's "author basic, convert internally" contract.

### 7.7 Edge diffraction

No build. The Huygens gather *is* the Kirchhoff/RS integral; edge diffraction is its
content (Du 2023: UTD-under-Fresnel ≡ knife-edge ≡ Kirchhoff for absorbing screens).
No commercial optical tool implements GTD/UTD. If a *propagating* diffracted ray is
ever wanted for stray light, the answer is a Keller-cone launch with plain GTD weights
— recorded as a non-seam (a new registry interaction, when justified by a real scene).

---

## 8. Optimization and tolerancing

The machinery exists (§1.4); what it needs, in order: (1) the deterministic noise-free
evaluator — sequential mode; (2) DLS via Optiland's `least_squares` (+
`orthogonal_descent` for merit functions that stay on the MC path); (3) AD Jacobians —
Seger 2025: autodiff of the ray–surface intersection gives the full Jacobian at primal-
trace complexity (free-rides Optiland's torch backend); (4) keep nevergrad CMA-ES for
global search on the MC path, which stays noisy by nature and where derivative-free is
*correct*; (5) tolerancing inherits everything — N ≥ 4,602 MC draws for 99.9 % yield at
99 % confidence (Laville & Aymard) is infeasible at MC-trace cost and seconds at
sequential cost. DeepLens (Apache-2.0, active) and dO (MIT) are watched as
differentiable-design references — read for curriculum/end-to-end ideas; not
dependencies.

The merit-evaluation bridge: sequential operands evaluate on the Optiland trace;
MC-only operands (detected power through a scatter path, ghost irradiance) stay on
`fast_eval`'s path and get the §10.4 persistent worker so the per-evaluation floor
drops from ~4 s to milliseconds-after-warm.

---

## 9. The virtual instrument layer (new)

**Purpose (owner requirement): every simulated detector can be read through a
parametrized model of a real instrument, so a bench measurement and a simulation
compare 1:1 in the instrument's own units — including instruments not yet owned, so a
future acquisition is comparable on day one.**

### 9.1 Architecture

A **post-process layer over the existing ideal detector planes**. The physics products
(W/m² maps, spectra, Stokes, time cubes — unbiased, unclipped, negative-noise-preserving)
remain untouched underneath; the instrument view is derived, never a replacement. This
keeps the ledger and every existing invariant exactly as they are, and means N virtual
instruments can read one detector plane.

Each instrument profile has two modes:

- **`ideal`** — response only: spectral responsivity/QE, geometric sampling (pixel
  pitch/fill/aperture), spectral resolution function. Deterministic.
- **`full`** — adds the noise/transfer chain, synthesized reproducibly (seeded by run
  seed × instrument id × exposure index).

### 9.2 Instrument classes (initial set matches the owner's bench)

| class | response parameters | full-mode additions | bench twin exists |
|---|---|---|---|
| **camera / beam profiler** | pixel pitch & count, fill factor, QE(λ), window transmission | shot noise (Poisson on detected e⁻), read noise (e⁻ rms), dark current (e⁻/s, integration time), full-well + saturation, bit depth/ADC gain, optional pixel MTF | yes |
| **power meter / photodiode** | responsivity(λ) A/W or thermal flat, aperture geometry, angular acceptance | NEP/bandwidth noise, digits/resolution, calibration uncertainty (reported, not synthesized) | yes |
| **spectrometer** | slit/resolution function (FWHM vs λ), grating efficiency envelope, detector QE | stray-light floor, dark + read noise, wavelength calibration error | yes |
| **polarimeter** (placeholder) | analyzer sequence → Stokes; extinction ratio, retarder errors | detector noise per state | not yet owned |
| **wavefront/interferometric** (placeholder) | OPD map sampling, reference-arm model | camera-chain noise | not yet owned |
| **autocorrelator/FROG** (placeholder, pulsed) | consumes existing time products | SHG-detector chain | not yet owned |

The engine already produces the physical quantities each placeholder needs (Stokes
maps, OPD via `opd_exit_pupil`, time cubes) — the placeholders are pure readout models,
which is why they are cheap to keep honest.

### 9.3 Data model and surfaces

Profiles live in the property library: `opticalproperties/instrument/*.mieinst` (CSV
like the rest; `reference` column REQUIRED — datasheet or measurement source; loaders
hard-validate in `optprops.py` style). Bodies opt in via an `instrument` property on
detector bodies (stackable with `qe_curve`, which remains for the bare photocurrent
path). Outputs land beside the physical products: counts images (integer, saturated,
noisy), photocurrent A, instrument-resolution spectra. CLI: post-process flag +
simparams key; GUI: instrument assignment in the detector's property panel + a Results
sub-tab per instrument view with an ideal/full toggle. Headless parity is total: the
instrument layer is a `post_process.py` stage.

### 9.4 Contract with the bench (§12.2)

A bench comparison always names its instrument profile. When real gear is characterized
(measured QE, gain, read noise), the profile is updated from the datasheet placeholder
to measured values — the profile file’s `reference` column records which. Simulated
counts vs measured counts is then a like-for-like comparison, and residual disagreement
is *physics*, which is the point.

---

## 10. The run contract (new): quick pass → commit, without waste

### 10.1 Checkpoint/resume

Week-scale runs must survive crashes and power loss. Periodic checkpoints capture:
detector accumulators (they are additive sums — cheap), the RNG cursor (with lineage-
keyed Philox this is just the primary-index high-water mark per source), the ledger,
and the progress block. Resume re-opens the case, verifies the scene hash, and
continues from the high-water mark. Deterministic gate: an interrupted+resumed run is
**bit-identical** to an uninterrupted one (Philox makes this a construction property,
not a hope). Checkpoint cadence is time-based (default ~5 min), written atomically
beside `progress.json`.

### 10.2 Additive extension

A finished run can be **extended** with more rays instead of rerun: new primaries take
the next disjoint primary-index range under the same seed lineage; accumulators merge;
M_eff and the speckle pedestal drop monotonically. "Quick pass at 1e5, looks right,
extend to 1e7 overnight" reuses everything already computed. Statistical gate: an
extended run (1e5 + 9e5) matches a fresh 1e6 run within the statistical bar. This is
the concrete payoff of adopting Philox engine-wide (§4.5); the Python engine cannot
offer it (sequential-stream RNG) and is not asked to.

### 10.3 The accuracy dialog

Every run (GUI dialog; CLI flags with the same fields) shows: predicted wall time from
the corrected estimator (§0.2), predicted M_eff/speckle pedestal for coherent
detectors, and the VRAM plan (§16.2). The choice — faster vs deeper vs extend-existing
— is always explicit (owner decision). After the run, actual-vs-predicted feeds the
calibration file, so the dialog gets honest over time.

### 10.4 Persistent worker / standalone driver (reinstated from engine.md)

The ~2 s optics-env import + ~1.9 s CUDA-context floor is paid per case today —
dominating sweeps of cheap variants and any MC-path optimizer loop. The new core ships
as a standalone binary reading `scene.json` and writing the same npy/ledger products,
plus a **persistent mode** (serve loop over a socket/stdin, one CUDA context, N cases)
that `sweep_variants.py`, `fast_eval.py`, and the GUI runner all reuse. Mirrors the
`fcserver` pattern already proven in this repo.

---

## 11. Element library realism and display

- **Real iris** (N-blade leaf geometry — diffraction star, ghosts, vignetting), **real
  mounts** (barrels, retaining rings, edge blackening — via mechanical CAD ingestion,
  incoherent, meshes correct), **surface figure error** (Zernike/grid-sag maps as a
  prescription layer — the deterministic middle between ideal surfaces and statistical
  roughness, which dominates real PSFs). Each toggleable; ideal remains default.
- **Display**: prescription surfaces render exactly (the truth is analytic — no STL of
  a CAD of a prescription); mechanical stays meshed; the §0.1 fixes ship first and may
  resolve the visual complaint alone. Preview unifies with sequential mode (§5).
  ParaView stays for deep analysis.
- An iris should look black (§9.4 of engine2 — kept, strictly after physics).

---

## 12. Validation: invariants, oracles, and the optical bench

### 12.1 In-silico (gates, all pre-existing ones preserved)

The pinned invariant suite carries verbatim (Fresnel 1e-12, TIR phase, TMM λ/4,
thick-lens 0.5 %, double-slit λL/d ±1 px + visibility >0.85, Malus, calcite walk-off,
Kogelnik, Dammann Parseval, Igehy vs finite differences, BVH == brute force, Wiscombe
Qext, closure <1e-3 everywhere, flux in−out=absorbed, undo torture, demo equivalence).
New oracles per feature ship with the feature (registry rule, §4.2). **Independent
cross-validation** (new): simple-scene fields checked against independent
implementations — LightPipes (BSD-3) and prysm propagation for apertures/PSFs, batoid
(BSD-2) for geometric traces, torchoptics (MIT) for polarized Fourier cases — so
"plausible in silico" never rests on self-agreement. Anomaly accounting from engine.md
§13 carries: counted `nan_rejected`/`degenerate`/`conical_guard` events, per-pixel
anomaly image, non-silent NaN rejection at splat.

### 12.2 The bench protocol (new; the referee)

**Rule: measured reality outranks every simulator, including ours.** Getting close to
Zemax is nice; getting close to the bench is the requirement. Three campaigns matched
to owned gear, each executable incrementally as systems get built (owner: bench tests
happen during development, not at a fixed date):

1. **Power/transmission budget** (power meter): source → elements → detector power for
   lens/filter/beamsplitter trains. Acceptance: within combined instrument+source
   uncertainty, target ±5 % absolute (calibration-limited), tighter for ratios.
2. **Irradiance structure** (camera/profiler): double-slit and Michelson fringe pitch
   (±1 px) and visibility (±0.05); PSF/spot cross-sections and encircled energy for an
   imaging train; comparisons through the §9 camera profile in counts.
3. **Spectral** (spectrometer): grating/prism dispersion (peak λ within instrument
   resolution), filter edges, source SPDs through the spectrometer profile.

Every comparison is recorded in a committed ledger `docs/VALIDATION_BENCH.md`: date,
scene file, instrument profile + its provenance, measured vs simulated, residual,
disposition. Disagreements >bar trigger the adjudication rule: establish *which is
wrong* (instrument model, scene authoring, or engine physics) before touching anything
— correctness is not conformance, in either direction. When the polarimeter/
interferometer class gear arrives, campaigns extend with zero engine work (§9.2).

---

## 13. Zemax positioning (condensed; full analysis in engine2 §12, verified)

Where this engine genuinely exceeds Zemax (their staff's own words in the record): NLO
(SHG/Pockels/Kerr/TPA/saturable), time domain (pulse/spectrogram/streak/GDD), and
honest coherent diffraction in a non-sequential CAD scene (their NSC coherent sum
references phase to the pixel-centre and is conceded *"beyond the scope of the ray
model"*). Where it is behind and closing: sequential design workflow (Optiland),
RCWA (a gap, not a lead — Zemax ships 1D DLLs and outsources 2D to Lumerical), scatter
model breadth. Where it stays behind deliberately: the 300+-operand catalog, stock-lens
matching (legal, not technical), STOP, POP breadth. Effort calibration: this repo's
measured ~1.3× test-to-code ratio is the honest multiplier on all estimates; the
dominant schedule risk is validation, not implementation.

---

## 14. Licensing and dependencies

Repo license: **MIT** (§0.4). Dependency policy: permissive for anything linked or
vendored; copyleft tools usable as *offline generators or oracles only* (their outputs
— tables, test vectors — are not derivative works). All statuses verified 2026-07-16:

### Adopted

| library | license | role | notes |
|---|---|---|---|
| Optiland | MIT | sequential kernel (§5) | active; optimizer inventory verified in source |
| prysm | MIT | Forbes Q (§7.6), POP/MTF analysis oracles | **pin a git SHA** — PyPI stale (2022) |
| meent | MIT (license API; README omits it) | RCWA table generator (§7.5) | Li-rule/S-matrix claims unverified → adoption gate |
| cuFINUFFT | Apache-2.0 (LICENSE file; badge wrong) | NUFFT gather path (§6.3) | v2.5.1; GPU type-3 since 2.4.0 |
| yyjson | MIT | already vendored | — |
| OCCT | LGPL-2.1 + exception | host-side import/fit only (§3) | **dynamic-link**; never in trace path |
| RefractiveIndex.info data | CC0 | materials library expansion | public domain |

### Oracles / read-only references (never linked)

| library | license | role |
|---|---|---|
| poke | BSD-3 | PRT/Q-matrix algorithm reference — **stale (2025-05)**; vendor the algorithms |
| GeneralTmm | MIT | 4×4 anisotropic oracle (Hodgkinson formulation; 8-star niche — oracle, not dependency) |
| pyGTM | **GPL-3** | Passler-Paarmann 4×4 oracle — outputs only |
| LightPipes / batoid / torchoptics | BSD-3 / BSD-2 / MIT | independent cross-validation (§12.1) |
| ray-optics | BSD-3 | `.zmx`/`.seq` import reference |
| DeepLens / dO | Apache-2.0 / MIT | differentiable-design references (§8) |
| KrakenOS | **GPL-3** | read, do not link |
| Zemax TABLE coating format | — | phase-column import compatibility target (§7.3) |

### Rejected

Optika (no license — legally unusable); S4 (GPL-2, dead); Inkstone (AGPL); grcwa/nannos
(GPL — meent covers); torcwa (LGPL); FMMAX (archived; invrs-io fork's license is
NOASSERTION); rayopt/Goptical (dead); NVIDIA MDL (discards every quantity this engine
needs); OptiX (moot — BREP cancelled; headers would be redistributable but
proprietary-EULA, never relicensable); Embree (no relevant primitives).

---

## 15. Phased plan, gates, and branch strategy

Evolve in place. Keep the GUI (23.2k LOC), scripts, library data, demos, and 37k LOC of
tests — they encode hard-won correctness. Rewrite the engine core only.

**Branch strategy (owner decision):** P0–P2.5 land on `master` via short feature
branches (additive, existing tests gate). P3 — the core rewrite — lives on a long-lived
branch (`core-v3`) merged **only at functional equivalence with today** (all 34 demos,
parity at the real bar 1e-9/2 %, every invariant, closure everywhere). P4+ return to
feature branches / clearly-delineated commits. Milestone demos gate merges; automated
gates decide, hands-on shakedowns at big milestones only.

Every phase additionally ships: **CLI/headless parity; a GUI surface; an offscreen UI
simulation proving the GUI surface** (existing `QT_QPA_PLATFORM=offscreen` pattern);
and **a UI-authored demo** where the capability warrants one (authored through the
Project/session path as `make_demos.py` does — demo creation itself exercises the GUI).

| phase | work | gate |
|---|---|---|
| **P0** | §0 quick wins + MIT LICENSE + doc corrections | spiderweb/faceting gone; estimator within ~20 % on the benchmark set; gather micro-opts bit-identical; all tests green |
| **P1** | Gather precision architecture (§6.2) + NUFFT path (§6.3) + run contract (§10.1–10.3) | bit-exact at tile=1 (--gather-exact); per-key phase-error budget reported; double-slit visibility >0.85; michelson_folded benchmark gather ≥10× (measured 11.5×; `detailed` ~3 h from 39.6 h — the <1 h stretch target closed with §6.3's finding, revisit via kernel SFU tuning at idle-GPU benchmark time); resume bit-identical; extension bit-identical (achieved, stronger than the statistical gate) |
| **P2** | Correctness tier: importance sampling + BTDF (§7.1); Q matrix (§4.1); coating phase columns + phase-invalid flag (§7.3); conical guard (§7.2); Forbes (§7.6) | stray-light scene tractable at a real budget with ledger closure; retardance maps validated; Forbes vs prysm 1e-12; a UI-authored stray-light demo |
| **P2.5** | Virtual instrument layer (§9) + instrument demos | ideal-mode round-trip (counts→W inverts within quantization); noise statistics match parameters (χ²); camera/power-meter/spectrometer profiles from real datasheets; GUI Results instrument tab offscreen-tested |
| **P3** | Core: registry + one-source/two-target (§4.2–4.4) + persistent worker (§10.4), on `core-v3` | **functional equivalence**: 34 demos, parity 1e-9/2 %, all invariants, closure <1e-3; CPU and GPU builds agree; no feature can route without an implementation; fluorescence/GRIN/Berreman seams present with stub tests; then merge to master |
| **P4a** | Parity oracle: Optiland vs MC engine on shared demos | the arbiter exists before the second physics truth — non-negotiable order |
| **P4b** | Sequential mode via Optiland + DLS + AD (§5, §8) | merit evals in microseconds; a known design converges to its published prescription; `.zmx` round-trips; preview unified |
| **P5** | Prescription-primary data model (§3) | 34 demos reproduce within the statistical bar; prescriptions round-trip through FreeCAD |
| **P6** | Lekner uniaxial + RCWA tables via meent (§7.4-1, §7.5) | Lekner vs published amplitudes; meent adoption gate (Li-rule convergence); Wood-anomaly refinement demonstrated |
| **P7** | NLO/time-domain port to the fast core | each token lands with its parity test before registration; pulsed demos at C-engine speed |
| **P8** | Element realism + exact prescription display (§11) | figure-error maps round-trip; iris diffraction star visible in a coherent run |
| **P9 (optional)** | Full-anisotropy Berreman 4×4 against the P3 seam (§7.4-2) | quartz activity 21.77 deg/mm; absorbing-case oracle vs pyGTM outputs; owner opts in |

Bench-validation checkpoints (§12.2) interleave wherever gear and built systems allow —
they gate nothing on the schedule but everything on credibility.

**Adjudication rule (unchanged, load-bearing):** any disagreement >2 % between engines
triggers an investigation to establish which is *correct* before making them match. The
Python engine is a reference, not an oracle — it has two known-xfail physics gaps
(verified: circular-polarizer handedness `test_scenes_e2e.py:601`; Pockels coherent
reconstruction `test_nlo_elements.py:192-193`), a documented dA normalization
approximation, and a lossy trim. After P7 (the last Python-only tokens ported and
parity-pinned) the Python engine freezes at a git tag as the archive oracle with
committed golden vectors.

---

## 16. Development conduct (new)

### 16.1 Language

C++17 host / C-style `KFN` kernels (§4.4). Decided on readability and line count;
performance is provably a non-axis. Owner preference for C is honored where it costs
nothing (the kernels — which is where all the physics lives) and overridden where C++
deletes code (host orchestration).

### 16.2 VRAM is a budget, not a capacity

Never plan against 16 GB. The engine probes free VRAM at run start, subtracts a fixed
headroom (default 2 GB), and sizes `pixel_chunk`/`sample_chunk`/NUFFT admission against
the *budget*. Rationale beyond co-tenancy: the desktop compositor, the GUI's own VTK
context, and future remote machines all make "assume the whole card" a crash-shaped
assumption.

### 16.3 Benchmarks under co-tenancy

The GPU currently shares with an MD refinement (~250 MB, plus thermal/IO effects).
Rules: benchmarks taken during co-tenancy are *indicative only* and marked so;
`.calibration.json` is refreshed only from runs on an otherwise-idle GPU; no
VRAM-pegging tests while the co-tenant runs; performance regressions are only declared
from idle-GPU A/B pairs.

### 16.4 Testing posture

Minimal time testing, maximal time coding — **with the physics floor non-negotiable**:
the pinned invariant suite, parity gates, ledger closure, and per-feature oracles
always run; test ceremony beyond that is skipped. Order work fast-to-test-first:
bit-identical refactors and display fixes (verifiable in seconds) before
behavior-changing physics (verifiable in minutes-hours). During development, runs use
the fastest presets that still exercise the code path; the long-run validation sweeps
come at the end of a phase, not the middle.

### 16.5 Headless-first, GUI-proven

Every feature works over SSH with no display (CLI flags/simparams keys are the primary
interface; the GUI is a client of the same paths). Every GUI surface added is then
driven in offscreen UI simulation to prove the planned functionality exists — and new
demos are authored *through the UI/Project path*, which is itself the test that the UI
supports the feature.

---

## 17. Honest limits

- **The MC speckle floor is fundamental** — 1/M_eff, reported, never removed. The
  gather speedup buys a lower floor, not exactness.
- **Table coatings without phase columns stay phase-invalid** — information theory, not
  implementation (§7.3). The fix is asking the user, which is what Zemax does too.
- **Conical refraction at the axis is guarded, not modeled** (§7.2).
- **Rayleigh/Wood anomalies remain the weak spot of interpolated RCWA tables**;
  adaptive refinement mitigates, never eliminates.
- **Dichroic anisotropic media are genuinely hard** — the Berreman option's
  mode-sorting is where it will break; that is why it ships with an absorbing oracle
  and last.
- **Gradients never exist on the MC path** — non-sequential optimization stays
  derivative-free, and that is correct rather than deficient.
- **A 39.6 h interferometer becomes ~1 h, not ~1 min** — 20–60× is transformative, not
  magic.
- **Adopting Optiland creates a second physics truth** — mitigated by oracle-first
  ordering (P4a), which is a discipline, not a guarantee.
- **Instrument profiles are parametric models** — shot/read/dark chains, not sensor
  emulation; fixed-pattern noise, blooming, and nonlinearity enter only if the bench
  demands them.
- **Extension cannot fix a wrong scene** — more rays converge to the model, not to
  reality; only §12.2 checks reality.
- **DOP from small ensembles is biased high** — always reported with N.
- **We will not match Zemax's breadth** — deliberately (§13).

---

## References

Carried from engine2.md (all verified or re-verified): the polarization set (Yun,
Crabtree & Chipman *Appl. Opt.* 50:2855; Yun, McClain & Chipman 50:2866; Chipman, Lam &
Young 2018), Lekner 1991 (10.1088/0953-8984/3/32/017) and 2023 (*JOSA A* 40:722),
McClain-Hillman-Chipman *JOSA A* 10:2371/2383, Berreman *JOSA* 62:502, Passler &
Paarmann *JOSA B* 34:2128 + erratum, Tikhonravov-Baumeister-Popov *Appl. Opt.* 36:4382,
Krywonos-Harvey-Choi *JOSA A* 28:1121, Stover *Proc. SPIE* 9961:996102, BRO-PN-1157,
Berry *J. Opt. A* 6:289, *Phys. Rev. Materials* 4:055203, Moharam & Gaylord, Li *JOSA
A* 13:1870/1024, Forbes *Opt. Express* 15:5218 + 18:13851/19700, ISO 10110-12/-19,
Spencer *Appl. Opt.* 2:1257, Seger et al. *Opt. Express* 33:3054 (2025), Laville &
Aymard arXiv:2607.03067, Du et al. *Electron. Lett.* 2023, the Zemax
Huygens-PSF/POP/NSC-coherent/nonlinear-crystal documentation set, and the CUDA C++
Programming Guide Table 7 (CC 8.9 throughputs).

Library links: Optiland, prysm, meent, cuFINUFFT/finufft, poke, GeneralTmm, pyGTM,
ray-optics, LightPipes, batoid, torchoptics, DeepLens, dO, KrakenOS, RefractiveIndex.info
— URLs as in engine2.md §References/§12A, statuses re-verified 2026-07-16 (§14).

---

## Changelog

| date | change |
|---|---|
| 2026-07-16 | Initial. Synthesis of engine2.md + full verification pass (~45 claims, 3 audits — engine2 sound; corrections in §2.1) + dependency re-audit (§14) + owner decisions (§1.2). New relative to engine2: virtual instrument layer (§9), run contract with checkpoint/extend (§10), bench validation protocol (§12.2), persistent worker reinstated (§10.4), interaction-registry extensibility contract with volume-propagator seams (§4.2), adaptive tile-error control law (§6.2), C++17-host/C-kernel language decision (§4.4, §16.1), VRAM-budget and co-tenancy rules (§16.2–16.3), phased plan restructured with branch strategy and GUI-proof obligations (§15). Berreman moved to optional-last (owner); fluorescence/GRIN made designed seams; coherent curved detectors deprioritized. |
| 2026-07-17 | **OVERHAUL COMPLETE — every phase P0–P9 implemented, gated, and merged** (master d9427d0, one continuous session from design approval). Final gate: engine 1093/0 (2 xfail = the pre-existing honest physics gaps), GUI 1235/0, demos 15/15. Headlines: michelson gather 713→55 s (12.9×, error-bounded, exact path preserved); NUFFT closed with the band-truncation finding; checkpoint/resume/extend bit-identical; DLS designs lenses (0.0012% recovery, 1.18 s) against a machine-precision Optiland oracle; prescriptions are ground truth; virtual instruments validated live; exact uniaxial (Lekner-equivalent 4×4) and full-anisotropy Berreman (Lekner-arbitrated 9.4e-16, quartz 21.7700°/mm) both default-on with the effective-index errors quantified (1% @45° azimuth / O(1%) off-principal); time-domain + bulk NLO bit-identical at C speed; registry hard-error routing ended the silent-skip class (and caught its own first stale-binary incident at close-out); bladed-iris star 12×, Zernike figure error, preview physics == run physics at 6e-17 m. Deferred honestly (backlog in the session memory): χ² SHG C port, Lekner/Berreman C ports, biaxial exit interface, Optimize-pane wiring, Q-Forbes↔Optiland mapping, o/e transport channels, small parked items. Bench campaigns await built systems — the instrument layer is ready for them. |
