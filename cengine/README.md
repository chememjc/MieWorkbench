# MieWorkbench C engine (`miewb-trace`)

A compiled OpenMP (and, from phase D, CUDA) implementation of the
computationally intensive core of the MieWorkbench ray tracer. The Python
engine (`scripts/raytracer/`) remains the **permanent reference
implementation** — this engine exists purely for speed, is verified against
the Python engine feature-by-feature, and is only ever selected for scenes
whose every feature has passed those parity gates.

## Build

```bash
cd cengine
./build.sh          # → build/miewb-trace   (cmake + ninja, gcc, OpenMP)
./build.sh test     # + run the C unit tests (ctest)
./build.sh clean
```

Requirements: gcc ≥ 11, cmake ≥ 3.22, ninja, OpenMP. CUDA 13
(`/usr/local/cuda-13`) joins in phase D. The build is **optional**: without
the binary, `--engine auto` silently runs everything on the Python engine.

Binary discovery: `cengine/build/miewb-trace`, overridable with the
`MIEWB_CENGINE` env var (repo convention).

## How it plugs in

```
run_trace.py --engine {auto,python,c}      (default auto)
   └─ scripts/raytracer/cengine.py
        choose_engine(): detect every feature the scene uses; route to C
                         only if all ∈ PORTED (else Python, reason logged)
        build_request(): pre-resolve ALL dispersion at the fixed stratum
                         wavelengths; serialize scene+params to
                         <case>/cengine/request_seed<k>.json
        run_c_case():    spawn miewb-trace once per seed; convert raw
                         outputs into the standard case contract
                         (rays.npy, detectors/<label>.h5 via the SAME
                         save_detectors writer, audit.json, case.json)
```

The engine choice and reason are logged (`[trace] engine=c (...)`) and
recorded in `case.json` (`engine`, `engine_reason`). `--engine c` with
unported features is a hard error naming them. A C-engine crash under
`--engine auto` falls back to the Python engine with a loud warning —
the engine is a separate process, so failures are isolated.

`--workers` (Python process sharding) does not apply to the C engine; it
threads internally with OpenMP (`--threads` / request `params.threads`,
0 = all cores).

## Design decisions (locked; see the c-engine plan)

- **D1 — λ-table pre-resolution.** Every ray's wavelength is one of the
  fixed stratum values (`sources.py:331`), so the Python glue evaluates all
  dispersive quantities (body n(λ), filter α(λ), coating stacks, …) at
  those wavelengths and ships plain tables. The C engine never parses
  property CSVs or evaluates Sellmeier/Mie — ~1,600 lines that never need
  porting, and material-range errors keep their good Python messages.
- **D2 — Counter-based Philox4x32-10 RNG keyed by ray lineage** (`rng.h`).
  Every draw is a pure function of `(seed, ray_key, event_ctr, draw index)`;
  children derive keys from `(parent_key, event_ctr, child_slot)`. Results
  are therefore **independent of thread count and scheduling** (pinned by
  `test_cengine_parity.py::test_thread_count_invariance`: bit-identical
  detector cubes across 1/7/32 threads). It does NOT reproduce numpy PCG64
  streams — the agreed parity bar is deterministic (~1e-12) for non-random
  physics and statistical (3-seed ±max(3σ,1%)) for MC aggregates.
- **D4 — No HDF5 in C.** The engine writes raw `.npy` + JSON into
  `<case>/cengine/seed<k>/`; the Python glue packs the standard `.h5`
  through the same `save_detectors` code path the Python engine uses.
- **D6 — Strict floating point.** `-O3 -march=native` but NO `-ffast-math`
  and `-ffp-contract=off`: physics kernels are parity-gated at 1e-12 and
  FMA contraction changes results. Per-file relaxation only after parity
  is green and only where proven insensitive. `-g` always (crash
  backtraces stay resolvable).
- **Ray storage is array-of-structs** (`raybuf.h`), a documented deviation
  from the plan's SoA sketch: the trace loop is per-ray scalar (BVH
  traversal + branchy physics), where AoS is the cache-friendly layout.
  SoA appears where lockstep vectorization actually happens (the phase-D
  gather sample buffers).

## Threading & determinism model

Per batch: `#pragma omp parallel for schedule(static)` over rays; each
thread appends children/ledger credits/detector hits/viz rows to private
buffers; a serial pass merges them **in thread order** and replays detector
splats into the float64 cube. Consequences:

- Detector cubes and viz segment sets: bit-identical for any thread count.
- Ledger sums: reordering-level (~1 ulp) variation across thread counts —
  far inside the 1e-3 closure gate and the demo-equivalence tolerances.

## Error handling contract

- Exit codes: **2** invalid input, **3** physics runtime error, **4** CUDA,
  **1** crash (signal). The Python glue reports them with the log path.
- Every validation error names its context (face id, body label, sizes).
- SIGSEGV/SIGBUS/SIGFPE/SIGABRT print a backtrace + `addr2line` hint to
  stderr and the log — a bare segfault is a bug in this contract.
- Log: `<case>/cengine/cengine.log` always records at DEBUG level; stderr
  respects `--log-level` / `MIEWB_LOG_LEVEL`.
- Progress: `@MIEWB {json}` stdout lines under `MIEWB_PROGRESS=1`
  (run_trace re-broadcasts; Python keeps owning `progress.json`).

## Source map

```
src/kernels/   host+device-shared physics headers (C ∩ C++ subset, KFN):
    kmath.h    vec3/complex primitives, plane frames (numpy conventions)
    surf.h     analytic surfaces: roots/normal/UV   [phase A: plane,sphere]
    trim.h     trim containment (untrimmed/band/polygon regimes)
    fresnel.h  complex Fresnel/Snell/TIR/pol-basis/Jones rotation
src/rng.h      Philox4x32-10 + lineage keying (KATs in tests/test_rng.c)
src/request.c  request.json → SceneC (strict validation, trim_build)
src/trace.c    source sampling + the OpenMP wavefront loop (tracer.py port)
src/ledger.c   9-bucket power ledger (audit.py port), closure
src/detector.c planar grid mask + bilinear splat (detector.py port)
src/npyio.c    minimal .npy writer
src/log.c      levels, @MIEWB progress, crash backtraces
src/main.c     entry point; one invocation = one seed
tests/         ctest golden tests (Fresnel invariants, Philox KATs)
```

Every kernel header opens with the Python file:line contract it ports;
constants are copied verbatim and commented.

## Feature status

The routing source of truth is `PORTED` in `scripts/raytracer/cengine.py`.
Per-phase parity gates live in
`scripts/raytracer/tests/test_cengine_parity.py` (side-by-side vs the
Python engine on synthetic feature scenes) plus the C unit tests.

| Phase | Features | Status |
|---|---|---|
| A | plane/sphere surfaces, trim, Fresnel, mirror/absorbance, medium stack, bulk absorption + filters, incoherent detectors, viz, ledger | **done** |
| B | cylinder/cone/torus/asphere, TMM + table coatings, polarizers | **done** |
| C | scene-wide TLAS BVH + mesh BLAS (TLAS == linear scan pinned) | **done** |
| D | CUDA coherent gather + GPU occlusion + save-fields | pending |
| E | gratings, roughness/diffusers, ABg scatter | pending |
| F | birefringence (uniaxial, biaxial) | pending |
| G | particles/Mie | pending |
| H | ray differentials, ghost analysis, export-rays, viz-pattern | pending |
| I | importance aiming (opt-in), perf polish | pending |

## Sunset roadmap (documented per project decision)

1. Through shakedown and merge: torch gather stays as the 100% fallback
   and three-way parity reference (numpy/torch/CUDA).
2. After the C engine has survived a post-merge shakedown period: retire
   the torch gather backend (and its ~5 GB dependency from the optics
   env).
3. Eventually: sunset the Python compute paths for day-to-day use; the
   numpy engine remains indefinitely as the slow, readable reference that
   parity tests run against.
