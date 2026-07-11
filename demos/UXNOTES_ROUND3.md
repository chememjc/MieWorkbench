# UXNOTES_ROUND3 — design-usability round shakedown log

Pain points hit while authoring the round's 11 new demos through the
Project/chain-op layer (the same code paths the GUI docks call), plus the
offscreen GUI passes. Companion to UXNOTES.md (object-placer round) and
UXNOTES_ROUND2.md. Each entry: what hurt, why, and what was (or should
be) done about it. Consolidates the per-batch agent notes.

Legend: FIXED-THIS-ROUND / MITIGATED (helper exists, could be smoother) /
OPEN (candidate for future.md).

## From the telephoto design work (main session)

1. **Chain distances are vertex-to-vertex, paraxial gaps are
   plane-to-plane** — translating a solved ABCD layout (gaps between
   surfaces/planes) into chain `distance` fields requires knowing each
   primitive's port-vertex convention (`primitivelib.port_frames`): an
   iris's entry/exit vertex vs its stop plane, an achromat's exit vertex
   vs its rear principal plane. MITIGATED: computing the offsets in the
   demo script via `port_frames` + `core/paraxial`; a human in the GUI
   gets the paraxial readout (BFL/FFL are vertex-referenced, matching
   the chain convention) and the insert-value menu ("distance to
   previous element's focus" = ref BFL, exactly the chain semantics).
2. **No negative achromat primitive** — the telephoto's rear group is a
   classic negative cemented doublet; `lens_achromat` only solves f > 0
   (it scales a fixed positive design). Worked around with a
   ν-split dcv(bk7) + dcx(sf5) pair 0.5 mm apart, achromatized by hand
   (well: by `wizards`/`paraxial` in the demo script). OPEN: a
   `solve_achromat` extension for f < 0 (crown-negative split) + builder.
3. **Two-group-zoom detector tracking needs BFL(z)** — a rational
   function of the zoom gap. The expression grammar's `/` makes
   `-(p + q*z)/(r + s*z)` directly expressible (coefficients from the
   two group matrices), so the sensor follows the zoom exactly —
   genuinely pleasant. MITIGATED-BY-DESIGN; noting that WITHOUT the
   expression engine this demo would need per-position re-anchoring.
   OPEN (small): nothing computes p,q,r,s for you — a "zoom pair"
   calculator could.

## Consolidated pain points (batches 1+2 + main session), with dispositions

### Real bugs found by authoring (all FIXED-THIS-ROUND)
4. **Non-square detectors crashed the field-analysis MTF plotter**
   (36×24 mm → 512×341 grid; tangential axis applied to the sagittal
   slice). Fixed: `mtf2d` gained `freq_y_cy_mm`; `post_process` uses it.
5. **`lambdamin == lambdamax` divided by zero** in
   `sources.wavelength_strata`. Fixed: zero-width band = monochromatic.
6. **`led_white` was a catalog entry without a `primitives/led_white.FCStd`**
   — import_primitive failed and the multiled demo fell back to hand-setting
   the `spectrum` property. Fixed: primitive generated + committed.
7. **Iris/pinhole/annular openings invisible to the paraxial stop search**
   (`hole_diameter` wasn't in the aperture-alias list). Fixed in
   `core/paraxial._APERTURE_ALIASES` (found by the telephoto demo: the iris
   IS its aperture stop).

### Chain semantics that cost a build-fail each (MITIGATED by expect(); GUI echo below)
8. **Distance from a source measures from the source's exit vertex (its
   origin)**, not from world x=0 — off-by-source-position slip.
9. **Distance is exit-vertex→entry-vertex**: downstream elements land past
   the *back face* of a thick reference — off-by-thickness slip.
10. **Flip moves the body origin**: a flipped lens's body origin sits `ct`
   past the beam-side vertex (the solver flips about the port).
11. **Fold azimuth direction is unguessable headless** (azimuth 0 → +y turn
   for a +x beam; up = azimuth 90). The GUI's 3D preview + fold dialog is
   the affordance; headless authoring leans on expect().
   → FIX-THIS-ROUND: train-editor Distance cells get a "resolves to world
   (x, y, z)" echo in the tooltip, and the decenter/tilt tooltips gain a
   concrete "+x beam: decenter_x → world +y, decenter_y → +z" line.
12. **Chain `decenter` offsets the element BODY, not the beam axis** — you
   cannot re-center a train onto the world axis with it; the axial
   reference must itself be on-axis. (Deliberately exploitable for partial
   apertures.) → tooltip line added this round; deeper "beam-axis decenter"
   is future.md material.

### Placement affordances still missing (→ future.md unless noted)
13. **Anchored placements take only literal xyz/quat** — no variables, no
   expressions, no polar form; the moment a pose isn't a beam-chain
   relationship you drop to hand-computed literals (aerosol side detector,
   conoscopy field-source fan y_off = L·tan θ).
14. **A `--particles` cloud is not a chain-referenceable body** — no way to
   say "detector 40 mm from the cloud center at 90°" (nephelometer ring).
15. **No "fan of field angles" affordance** (sources at a common pivot
   overlap as solids; must be spread on an arc by hand).
16. **Co-located transparent detectors overlap-fail extraction** — no clean
   "measure the same plane two ways" authoring path.
17. **No "sensor at the system's paraxial image plane" chain intent** — the
   right-click insert-value menu now offers the system image distance +
   previous-element BFL (exactly this), but nothing expresses diffraction
   scales ("span N Airy zeros").

### Physics-vs-preset tensions (documented in the demos; helpers → future.md)
18. **Demo-spec numbers vs `quick`**: aerosol phi=1e-6 → τ≈5e-5 (invisible;
   demo uses 2e-2 → τ≈1.14); Ø6 beam through a Ø0.2 pinhole → GatherError
   undersampled (demo uses Ø0.6); triplet at Ø14 is spherical-aberration
   soup (analysis demo uses Ø5). A validate-time "this scene needs ~N rays
   for its coherent gathers" preflight and a `--particles` target-τ knob
   (solve phi for a wanted optical depth) both belong in future.md.
19. **Coherent gather does not resolve Airy ring NULLS at quick budgets**
   (disk scale right, rings washed) — documented honest limit in the demo.
20. **Thick retarders silently collapse the reconstructed field**
   (phase_step ≫ π: displayed 0 mW while detected_geometric_W is right).
   → FIX-THIS-ROUND: run-time warning when the gather's phase step blows
   past π so the 0 mW isn't a head-scratcher.
21. **Incoherent sources + --save-fields = empty fields group, no Stokes**
   → FIX-THIS-ROUND: explicit warning at trace time.
22. **A full final analyzer flattens S1/S2 maps** (uniform projection);
   angle-varying (conoscopic) geometry is what survives a coherent gather —
   physics, documented in the stokes demo.

### Defaults / discoverability (small FIXES-THIS-ROUND)
23. **Pinhole/slit blackness default 0.98 leaks 2%** through the very
   screens the aperture-diffraction contract is for → default to 1.0
   (iris keeps 0.98 — a photographic stop is a physical part).
24. **The emission-spectrum vocabulary was undiscoverable** from the
   palette → led_white primitive now ships; source_broadband tooltip points
   at the emitters registry.
25. **Detector face auto-pick lands on edge faces** for any non-trivial
   detector orientation (bit both batches) → pre-run validation now warns
   when the auto-picked face looks like an edge face and suggests
   `detector_face`; demos pin via pin_detector/detector_face.
26. `particle_threshold` direction ("explicit if count ≤ threshold") reads
   backwards; `laser_collimated` divergence semantics not in the params;
   LED primitive name suffixes are bin nicknames, actual CWL in lambdac;
   model.json/rays_full.npz are METRES (1000× foot-gun) — doc lines.

### Frictionless (credit where due)
- `--photometric`/`--emit-csv`/per-source tables flowed with zero friction.
- The expression grammar carried the zoom demo's rational BFL(z) tracking
  and the telephoto's every-dimension efl-scaling without workarounds.
- expect() build-time self-checks caught every chain-semantics slip before
  a single trace ran.
