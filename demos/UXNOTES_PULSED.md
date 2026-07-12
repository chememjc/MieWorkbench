# UX notes — pulsed-optics round demo shakedown (2026-07-12)

Friction found while building/running the nine pulsed-round demos, in
build order. Engine bugs found here were fixed in the round itself;
authoring traps are listed for the next demo author.

1. **Chained `reflect` port only exists on mirror-ish kinds.** The SHG
   bench's dichroic was first authored as a `window` with a fold — a
   window's ports are `out`/`transmit` only, so `port="reflect"` is a
   TrainError. `bs_plate` (whose KIND declares the reflect port) with a
   `coating` override is the idiom for any dichroic/splitter pick-off,
   exactly like michelson's BS.

2. **`fold_deviation`/`fold_azimuth` vs `tilt_ry` conventions differ.**
   A hand-authored `fold_deviation="90", tilt_ry=45` plate folded into
   −z, not the intended −y (the azimuth convention rotates the fold
   plane out of the layout). For in-plane pick-offs, michelson's
   `tilt_ry=45` + `port="reflect"` chain is far harder to get wrong
   than raw deviate fields.

3. **Spreadsheet alias `V` is rejected by FreeCAD** ("Invalid alias" —
   single capital letters collide with the column namespace, same
   family as the documented R1/A2 trap). The pockels sweep variable is
   `volt`.

4. **The Pockels gap property is `pockels_gap`,** not `pockels_gap_mm`
   (the `_mm` suffix appears only on the extracted model.json echo).
   Scene errors name the missing property clearly.

5. **A monochromatic fs source shows no GDD broadening.** Wavelength
   strata carry the envelope bandwidth: a mono source is one zero-width
   stratum → Δω=0 → the analytic kernel never widens, no matter the
   accumulated GDD (the budget table still predicts τ_out, silently
   disagreeing with the trace). The fs laser primitives now bake their
   transform-limited (Mai Tai: ±4 nm σ) or datasheet (FemtoFiber:
   80 nm) bandwidths; a demo that overrides `lambdac` must override
   `lambdamin`/`lambdamax` too or it inherits the 800 nm bounds.

6. **Detector spectra of tabulated/SPM sources are stratum combs.**
   Strata are equal-power by design, so the spectral SHAPE lives in
   their density: at `nlambda=17` the SPM demo rendered a flat comb.
   `nlambda` must comfortably exceed `--spectral-bins` for the detected
   spectrum to look like the SPD (erfiber_spm uses 129 vs 64).

7. **Auto time windows cover the echo train.** A lens bench's window
   stretched to ~1 ns because the double-bounce ghosts arrive ~60 ps
   after the main pulse — 2 ps bins cannot resolve a 100 fs pulse. The
   fs contrast pair pins `--time-window` to ±1.5 ps around the computed
   main-pulse group delay. (A future "dominant-cluster" auto window is
   listed in future.md.)

8. **Transmission `grating_plate` + `orders=-1..1` leaks the truncated
   orders past the closure gate** (~8% at 800 nm/600 g/mm): the
   REFLECTIVE (aluminum, `mirror=1.0`) build books the lamellar
   remainder into absorbed_surface exactly, so the Treacy pair is
   reflective. (Transmission truncation booking is a known seam.)

9. **ENGINE BUG (fixed this round): grating children kept the incident
   s_hat.** At (near-)normal incidence, `pol_basis` normalizes
   FreeCAD-extraction noise (~1e-8 direction components) into an
   arbitrary transverse s; a diffracted child then carried an s_hat not
   perpendicular to its own direction, and the SECOND grating of the
   Treacy pair rotate_jones'd through the skewed basis — a silent
   cos²θ_d (9.3%) closure leak. `grating.apply_to_batch` now rebuilds
   each child's frame (n×d, sign-aligned); regression
   `test_normal_incidence_grating_pair_closure` tilts the emit normal
   by 1e-7 to reproduce the noise (exact zeros hide the bug in the
   clean degenerate fallback).

10. **GDD budget rows from metal mirror bodies** — the nanowatt
    evanescent fraction entering an aluminum mirror earned a table row
    with −163,000 fs² of "aluminum GDD" that no meaningful power ever
    sees. The budget now requires ≥0.1% of emitted power through a body
    before it earns a row (run_trace.build_gdd_budget flux floor).

11. **Coherent sources + many strata trip the gather budget.** 129
    strata (SPM demo) or a diffraction-split arm (Treacy) leave <1000
    effective coherent samples per (source, stratum) key at the quick
    preset. Neither demo needs interference: `coherent=False` on the
    source is the right call, not a rays bump.

12. **The pockels_switch demo was DROPPED after two full builds.** The
    o/e recombination that turns EO retardance into detected-power
    modulation exists only in the coherent gather; a mm-scale cell
    carries thousands of radians of static o/e phase (gather phase-step
    gate), and even the beat-length-thin cell on a gallery-scale bench
    leaves the reconstruction phase-noise-dominated — the 0..V_π sweep
    moved detected power 0.6% where sin² predicts 61%. The physics
    stays pinned by the engine oracle (test_nlo_elements, 1%), which
    uses a deliberately tiny bench for exactly this reason. A shippable
    EO demo needs either a much larger coherent budget or an
    incoherent-path observable (e.g. a Stokes/DOP readout between
    CIRCULAR polarizers) — future.md.
