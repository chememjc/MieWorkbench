# Pipeline CLI

`scripts/run_pipeline.py` — the headless orchestrator every GUI run
ultimately shells out to (via `core/runner.py`'s `RunController`, or
directly on a machine with no GUI). Full flag reference:
`run_pipeline.py --help`, or `scripts/cli_specs.py` (the single source of
truth `ConfigMatrix` introspects). Full engine/physics reference:
[`../RAYTRACER.md`](../RAYTRACER.md). This page is a condensed pointer,
not a duplicate.

## Basic invocation

```bash
python3 scripts/run_pipeline.py --models example.FCStd --preset quick
```

Interpreter: plain system `python3` (stdlib-only by design — the same
interpreter FreeCAD's embedded Python uses, see CLAUDE.md's pinned-
interpreter table).

## Presets

`quick` = 1e5 rays / 512² / 5λ · `normal` = 1e6 / 2048² / 9λ · `detailed`
= 1e7 / 4096² / 17λ. `--max-reflections N` raises the 6-bounce default cap
(some demo scenes need ~200).

## Sweeps

`--var NAME --min LO --max HI --n N` (repeatable; sheet-qualified names
work, e.g. `--var dim_Lens1.ct`); `--sweep-mode product|zip` picks the
combination order (`common.sweep_combos` is the one authority, shared
with the [Variables](variables.md) pane).

## Detector outputs

`--photometric` (lux maps), `--spectrometer` (λ-vs-x profiles). QE-weighted
photocurrent has no CLI flag — set `qe_curve` on the detector body
instead.

## Time domain (pulsed sources)

`--time-products pulse,spectrogram,streak,cube` (a pulsed source
auto-enables `pulse,spectrogram`; `none` suppresses); `--time-bins`,
`--time-window`, `--time-envelope analytic|histogram`; `--gdd-budget`
(per-element group-delay/GDD/TOD table, free when time products run, or
forces group-delay tracking on a CW run).

## Ray differentials / imaging

`--ray-differentials` (needed for Kerr bulk NLO and the imaging
products' PSF-convolution path); `--image-sim PATH` (requires
`--save-fields`).

## Engine selection

`--engine {auto,python,c}` — default `auto` picks the compiled C engine
when every scene feature the run needs is in the `PORTED` set
(`scripts/raytracer/cengine.py`), else falls back to Python. See
[`../RAYTRACER.md`](../RAYTRACER.md) §13 for what's ported.

## Related

- [run-and-validate.md](run-and-validate.md) — the GUI surface over this
  CLI (auto-generated form, pre-run dialog, dry run).
- [optimize.md](optimize.md) / [tolerance.md](tolerance.md) —
  `scripts/optimize.py` / `scripts/tolerance.py`, separate CLIs built on
  the same `fast_eval` evaluator, not `run_pipeline.py` itself.
- [headless-remote.md](headless-remote.md) — running any of this without
  the GUI at all.
