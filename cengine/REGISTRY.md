# REGISTRY.md — the P3 interaction registry (core-v3)

Status: **binding spec** for the core-v3 round (engine3.md §4.2–4.4, §15 P3).
The goal: every physics feature becomes a self-contained registration instead
of a branch in a hardcoded chain, and **no scene feature can ever route
without an implementation** — the failure mode of the `PORTED` token set
("forget the token → silently skip the physics", violated once, P8 NLO) is
structurally eliminated.

## 1. The two seam kinds

### 1.1 Surface interactions

What happens when a ray meets a face. Today: `trace.c::process_ray`'s
if-chain (mirroring `tracer.py:525-555` + `_optic_children`). Becomes:

```c
typedef struct {
    const char *token;      /* feature token — MUST equal the string
                             * cengine.py::detect_features emits for the
                             * feature; the parity of these two lists is
                             * pinned by a test (see §4) */
    /* Scene-build time: does this interaction apply to face fid?
     * Pure function of the scene description — never of ray state. */
    int (*match)(const SceneC *s, int32_t fid);
    /* Trace time: consume the parent ray at the hit, push children via
     * push_child(), book energy ONLY through the ledger API. The handler
     * owns its complete energy bookkeeping: sum(children power) +
     * sum(ledger credits) == parent power (the 1e-3 closure gate is the
     * runtime instrument; per-handler unit oracles pin it exactly). */
    void (*apply)(const SceneC *s, ThreadCtx *cx, const Ray *ray,
                  const HitInfo *hit);
} InteractionDef;
```

### 1.2 Volume propagators

What happens to a ray BETWEEN hits. Today implicit (straight line, fp64 OPL
accumulation, bulk absorption via the medium stack). Becomes dispatchable so
GRIN / fluorescence / time-dependent media are fill-in work:

```c
typedef struct {
    const char *token;
    int (*match_medium)(const SceneC *s, const Ray *ray);   /* build/push time */
    /* advance the ray by the segment to t_hit: position, fp64 OPL
     * (d(OPL)/ds = n — phase-critical), bulk absorption (ledger),
     * and for curved propagation (GRIN) the actual path integration.
     * The homogeneous propagator is the identity-cost default. */
    void (*advance)(const SceneC *s, ThreadCtx *cx, Ray *ray, double t_hit);
} PropagatorDef;
```

## 2. Construction-time resolution (the hard-error contract)

At scene build (`request.c` → scene finalize):

1. Every face gets an **ordered handler list** resolved from the registry
   (order encodes today's precedence exactly: detector-screen → grating →
   the optic chain; §3's port preserves behavior branch-for-branch).
2. Every feature token present in the scene (the request carries the
   detect_features list — new field `features[]`) is checked against the
   registry. **Unknown token → hard error at load, exit 2, naming the
   token and the body/face that produced it.** Never a silent skip.
3. `choose_engine` (Python) keeps routing under `auto` by consulting the
   SAME token list: a token the C registry lacks routes the case to Python
   BEFORE the binary launches (unchanged UX), and the C binary's own check
   is the belt-and-suspenders backstop for forced `--engine c`.

`PORTED` in `cengine.py` remains the Python-side mirror of the C registry,
but gains a parity test (§4) so the two lists cannot drift.

## 3. Porting order (feature-by-feature, parity after each)

Restructure `process_ray` into registrations WITHOUT changing behavior —
each step moves one branch into a handler and must pass the full parity
suite (1e-9 deterministic / 2% statistical) plus its own oracle before the
next step starts:

1. `detector` screens (transparent measurement planes)
2. `grating` (all models)
3. absorber / `mirror` / `filter` (the simple optic-chain heads)
4. `coating` (TMM + table) + bare Fresnel refract/reflect (the big one —
   the reflect/transmit core becomes the DEFAULT terminal handler)
5. `polarizer`
6. `roughness` + `scatter` (specular attenuation + lobes)
7. `birefringence` (uniaxial o/e)
8. `particles` (continuum + explicit is Python-only; continuum registers)
9. volume propagators: homogeneous (default) + bulk absorption

Then the **seam stubs** (registered, match never fires yet, oracle files
named): `fluorescence` (volume, λ-shifting emission; oracle: ledger closure
with λ-shifted output), `grin` (propagator, RK4 + fp64 OPL; oracle:
Luneburg/Maxwell-fisheye analytic foci), `berreman` (surface, full
anisotropy; oracle: quartz activity 21.77°/mm @589.3 nm + a
Passler-Paarmann absorbing case — P9, owner opt-in).

## 4. Tests that pin the architecture

- **Token parity**: the C binary dumps its registry tokens (`--tokens`);
  a pytest asserts the dump ⊇ `cengine.PORTED` and that every token in
  `detect_features`' emission surface appears in exactly one of
  {C registry, Python-only set} — no orphans, no silent gaps.
- **Hard-error**: a request carrying a fabricated token exits 2 naming it.
- **Behavior freeze**: after each porting step, the parity suite + the
  pinned invariants (closure <1e-3 everywhere, thread invariance,
  RNG KATs) — the same bar every earlier round used.

## 5. Host-side C++17 conversion (scope-limited)

Where it deletes code: the registry tables and scene-build resolution
(std::vector/string_view instead of hand-rolled arrays), request/scene
buffer management (RAII), the persistent-worker serve loop (§6). The
kernels (`kernels/*.h`) stay C-style KFN headers — they compile as CUDA
device code and are the physics; no virtual dispatch anywhere in the hot
path (handler lists resolve to flat function-pointer arrays at build).

## 6. Persistent worker (P3 item 5, spec'd in the P1 chunk round)

Serve loop in `main.c(pp)`: read request paths over stdin (the `fcserver`
pattern), hoist CUDA context + reusable device buffers across requests;
`cengine.py::_run_binary` gains worker mode; `sweep_variants.py` and
`fast_eval.py` reuse it. Gate: V×S-case sweep pays ONE context init;
per-case results byte-identical to one-shot invocations.

## 7. Gate for the P3 merge (unchanged from engine3 §15)

Functional equivalence: all 34 demos, parity 1e-9/2%, every invariant,
closure everywhere, thread invariance, CPU and GPU builds agree; then
core-v3 merges to master.
