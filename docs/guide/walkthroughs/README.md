# Demo walkthroughs

One page per optimize/tolerance **showcase demo** (the four
`scripts/run_demo_equivalence.py` gate smoke-runs, see
`demos/README.md`'s "Optimization & tolerancing" section): load the demo
→ what the pre-configured Optimize/Tolerance panes already show →
run a short optimize → run a tolerance sensitivity pass → interpret the
physics story, including the one showcase whose textbook result does
**not** reproduce cleanly (say so plainly rather than overclaiming).

| Page | System | Optimize operand | The story |
|---|---|---|---|
| [camera-triplet.md](camera-triplet.md) | Cooke triplet, ~50 mm EFL, f/5.6 | `spot_rms` (sequential) | middle-element decenter *should* dominate — broadband/stray-ray noise makes the ranking MC-unstable in practice |
| [schmidt-cassegrain.md](schmidt-cassegrain.md) | C8-class 203 mm f/10 catadioptric | `spot_rms` (worker/MC) | secondary despace-to-focus magnification |
| [double-gauss.md](double-gauss.md) | Symmetric double-Gauss, ~53 mm f/2.6 | `spot_rms` (sequential) | symmetry-breaking decenters dominate a fast near-symmetric form |
| [fiber-coupling-doublet.md](fiber-coupling-doublet.md) | Achromat → 0.22 NA fiber | `detected_power` (worker/MC) | lateral decenter kills coupling far faster than despace |

Every page's optimize/tolerance step mirrors exactly what
`scripts/run_demo_equivalence.py`'s gate smoke-runs (`SHOWCASE`,
`SMOKE_TOL_ROWS`, `SMOKE_BUDGET=3`, `SMOKE_RAYS=30000`) — the screenshots
on [optimize.md](../optimize.md)/[tolerance.md](../tolerance.md) and on
`camera-triplet.md` are real output from that exact smoke study, not a
mockup. See [`../demo-gallery.md`](../demo-gallery.md) for the full
38-demo gallery this sits inside, and `demos/README.md` for every demo's
prescription + citation.
