# Tolerance

`mieworkbench/panes/tolerance_pane.py` (`TolerancePane`, central
"Tolerance" tab, also **Simulation → Tolerance…**) — full GUI parity with
`scripts/tolerance.py`, driven through `core/tolerance_controller.py`
(`ToleranceController` owns the `QProcess`).

## Variable address forms

Identical grammar to [Optimize](optimize.md): bare `alias` (element's
`dim` sheet), `sheetlabel.alias`, `miewb_vars.<name>` (auto-qualified by
`core.variables.qualify_var_name` when picked from the dropdown), or
`train.<ElementLabel>.<field>` for a chained element's pose — exactly the
per-element decenter/despace/tilt tolerancing wants.

## Three phases (`ToleranceEngine`)

1. **nominal** — one evaluation at every tolerance's nominal value and
   the compensator (if any) at its start. The baseline every
   sensitivity/impact number is relative to.
2. **sensitivity** — for each `--tolerance NAME:NOMINAL:DIST:BAND`,
   evaluate the merit at `NOMINAL ± sens_delta*BAND` and report:
   - `derivative = (M+ - M-) / (2*delta)` (central difference, signed
     local gradient)
   - `impact = max(|M+ - M0|, |M- - M0|)` — the table is **ranked by
     impact, not derivative**: a design toleranced at its merit minimum
     (e.g. defocus at best focus) has derivative ≈ 0 while the band still
     costs real merit; impact is the number a tolerancing engineer
     budgets with.
3. **monte-carlo** — `--draws N` random perturbation sets, each tolerance
   sampled from its distribution, merit evaluated per draw. With a
   compensator, each draw first runs a nested `optimize.OptimizationEngine`
   (local Nelder-Mead, `comp-budget` evals) over the compensator variable
   with the perturbations held fixed — the recorded merit is the
   **compensated** one, exactly how an as-built system is refocused
   before test.

A failed/incomplete evaluation is penalized (`optimize.PENALTY`), never
fatal — excluded from distribution stats/histogram but still counts
against yield.

## Tolerance table

Columns: name / nominal / distribution / band. Distributions
(`TOLERANCE_DISTS`): `normal` (BAND = 1-σ width) or `uniform` (BAND =
half-width), both in NAME's units — mm for distance/decenter, degrees for
tilts. The name-cell dropdown is `miewb_vars`-fed like Optimize's
variable table; picking a global auto-fills nominal/band from its
`__min`/`__max` sweep meta.

## Operand table

Same grammar and semantics as [Optimize](optimize.md)'s operand table —
`spot_rms`/`focus`/`encircled_energy`/`mtf50`/`detected_power`, or a raw
`report.json` key.

## Compensator

`VAR:LO:HI` (or `VAR:START:LO:HI`, START defaults to the midpoint) — a
design variable the Monte-Carlo phase re-optimizes per draw with the
random perturbations fixed. Its name must not collide with any tolerance
name.

## Monte-Carlo settings

Draws, RNG seed, merit threshold (for yield), compensator + comp-budget,
sens-delta (the fraction of BAND used for the sensitivity finite
difference), skip-sensitivity, histogram bin count, plus the shared
preset/rays/backend fidelity combos.

**Yield** = fraction of draws with merit ≤ merit-threshold, over **all**
draws (a failed evaluation counts as a failed unit).

## Live result views

Mirrors Optimize's plotting: `QtCharts` when importable, else a
dependency-free `QPainter` fallback. Both charts share Optimize's hover +
right-click "Show data…" (non-modal table with an **Export CSV…**
button, `plot_inspect.DataTableDialog`).

- **Sensitivity bar chart** — ranked by impact, fed by the run's
  `phase="sensitivity_done"` progress event (a compact ranked table).
  Still bars; hover shows a `name — impact I (derivative D)` tooltip,
  right-click → Show data opens `name | impact | derivative | rank`.
- **Monte-Carlo merit distribution** (`MeritDistributionPlot`, the former
  "yield histogram") — no longer bars: a **frequency polygon** (a line
  through the histogram bin centres, x-axis labeled "merit") plus a
  **cumulative-distribution (CDF)** curve on a right-hand 0–1 axis, fed
  incrementally per draw (`draw`/`draws`/`merit`/`yield-so-far`/`params`
  extras). Draws with `merit >= PENALTY_FLOOR (1e8)` are excluded from
  both curves but counted separately. Hover either curve for a per-draw
  tooltip; right-click → Show data opens `draw# | <variables…> | merit |
  pass/fail | rank`.

## Failure banner

Same non-modal error-surfacing banner as Optimize (`match_error_line`,
shared `_ERROR_LINE_RE`).

## Config persistence

The pane's full configuration (tolerances, operands, compensator,
Monte-Carlo settings) is stashed as JSON on the `miewb_vars` sheet
(`Project.set_tolerance_config`/`get_tolerance_config`, document property
`miewb_tolerance_config`), travels with the `.FCStd`/`.MieWB`, and
re-populates the pane on reopen via `apply_config()`. Round-trips exactly
(`config()` → `apply_config()` → `config()`).

![Tolerance pane](img/tolerance-1.png)
*(`camera_triplet`'s shipped tolerance table + the gate's trimmed
3-row sensitivity smoke run — see
[walkthroughs/camera-triplet.md](walkthroughs/camera-triplet.md) for the
decenter-ranking caveat this specific system illustrates.)*
