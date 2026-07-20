# UX friction log — consolidated open items

Consolidated 2026-07-19 (docs round). This file supersedes the four
per-round shakedown logs (round-1 `UXNOTES.md`, `UXNOTES_ROUND2.md`,
`UXNOTES_ROUND3.md`, `UXNOTES_PULSED.md`): every observation in those
files was re-verified against the current code; everything not listed
below was confirmed FIXED (with the fixing code located), obsolete, or
non-actionable historical color, and was dropped. Git history keeps the
originals.

## Placement / transform panel

1. **No axis+angle rotation readout.** The Absolute-pose group in
   `transform_panel.py` only shows intrinsic XYZ Euler angles at 2
   decimal places — a sub-mrad tilt (e.g. a Michelson fringe-tilt of
   0.158 mrad ≈ 0.009°) rounds away and can't be visually verified.
   Wants an axis+angle display with 4+ decimals.
2. **Anchored (absolute) placement fields are literal-only.** "Set
   position"/"Set orientation" in the Absolute group are plain
   `QDoubleSpinBox`es — no expression/variable support, unlike the chain
   edge cells and the newer polar "Place about point" tool (which DOES
   support expressions and a genuine polar form). A side-scatter
   detector or field-source fan still can't be swept parametrically
   from an anchored pose.
3. **`--particles` clouds are not chain-referenceable.** No way to say
   "detector 40 mm at 90° from the cloud center" (nephelometer-ring
   authoring) without hand-computed literals — needs a lightweight
   region-anchor/virtual element the reference resolver can target.
4. **Co-located transparent detectors overlap-fail extraction.** No
   authoring path exists for "measure the same plane two ways."

## Catalog / wizard gaps

5. **Achromat wizard can build an unbuildable lens.** `solve_achromat`/
   `design_lens("achromat", …)` never maps or clamps `aperture` against
   the solved interface radius — a small-f solve against the default
   18 mm aperture still dies with a bare "math domain error" instead of
   a clamp or warning.
6. **No "slit lamp" catalog preset.** A divergent+broadband source is
   still assembled by hand from `laser_divergent` + two properties.
7. **No `grating_mirror` catalog primitive.** A reflective grating
   (Czerny-Turner-style) still needs three manual switches on
   `grating_plate` (`mirror=1.0`, explicit periodicity vector, face
   pin).

## Rebuild-on-edit

8. **Face-map re-resolution after rebuild is unimplemented.**
   Rebuilding a primitive still renumbers `FaceN` indices with no
   geometric old→new matching, silently orphaning coating/grating/
   surface_override face-map properties. The Czerny-Turner demo still
   works around this by importing un-rebuilt geometry.

## Pulsed-optics round

9. **`pockels_switch` demo is still dropped.** The coherent o/e-gather
   recombination that would turn EO retardance into an observable
   detected-power modulation is phase-noise-dominated at gallery scale
   (0.6% observed vs 61% predicted by sin²); needs either a much larger
   coherent ray budget or a new incoherent-path observable (e.g. a
   Stokes/DOP readout between circular polarizers). Not currently
   tracked in future.md.
10. **Auto time-window sizing covers echo-train ghosts** instead of
    clustering around the dominant pulse (e.g. a ~1 ns window for a
    100 fs pulse when a double-bounce ghost arrives ~60 ps later).
    Tracked in future.md as "Dominant-cluster auto time window" but
    still unimplemented.
11. **Transmission-grating truncated diffraction orders leak ~8%** past
    the closure gate — only the reflective/aluminum booking branch
    accounts for the truncated remainder correctly. Tracked in
    future.md as "Transmission-grating truncated-order booking" but
    still open.

## Minor / unverified

12. **Small doc/naming gaps** (open-unverified): `particle_threshold`'s
    "explicit if count ≤ threshold" direction reads backwards;
    `laser_collimated` divergence semantics aren't documented on the
    params; LED primitive name suffixes are bin nicknames while the
    actual CWL is in `lambdac`; `model.json`/`rays_full.npz` units are
    metres (a 1000× foot-gun) with no doc callout.
