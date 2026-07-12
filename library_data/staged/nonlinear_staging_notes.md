# Nonlinear registry (P7a) — staging notes

The `opticalproperties/nonlinear/nonlinear.mienlo` registry landed in the
pulsed-optics round (Phase P7a: registry + `raytracer/nlo.py` math +
`raytracer/optprops.load_nonlinear`). Source data:
`library_data_pinned.md` research notes (2026-07-11), RECOMMENDED values
carried with condensed one-line citations. Items still waiting on other work:

| Item | What's missing | Waits on |
|--|--|--|
| `n2_yag` row (`material=yag`) | No `yag` index row exists in `materials.miemat` — undoped YAG Sellmeier (Zelmon, Small & Page 1998, Appl. Opt. 37, 4933 is the standard citation) was never merged. The n2 row ships anyway because the loader resolves `material` LAZILY by design (the Kerr consumer looks it up against MaterialDB at use time). | A `yag` materials.miemat row (data-only merge; any future materials pass). |
| Tracer-side SHG event | `nlo.py` is pure math; nothing in `scene.py`/`tracer.py` consumes `chi2_tensor`/`chi2_process` rows yet. | Phase P7b (owned by the tracer-side agent). |
| Kerr / n2 propagation | `n2` rows load but no B-integral / self-focusing consumer exists. | Phase P8 (uses `nlo.local_intensity` for the intensity convention). |
| Pockels cells | `pockels` rows load; no EO retardance element consumes `r_pm_V` yet. V_pi formulas per geometry are documented in the row notes. | Future EO-element phase. |
| `sam_1550_16_2ps` | Saturable-absorber row loads; no absorber element/time-domain consumer. `T0` = unsaturated REFLECTANCE (1 − A, mirror device); modulation depth 0.09 and non-saturable loss 0.07 ride in the row notes; `I_sat` is the F_sat/τ_pulse scaling estimate at τ_pulse = 1 ps. | Mode-locking/time-domain phase. |
| Type-II phase-matching solve | `nlo.phase_match_angle` solves type-I (ooe, negative uniaxial) generally; type-II returns the pre-solved `chi2_process` registry angles with `source="registry"`. | A general two-branch (biaxial-capable) index-surface solver, if ever needed. |
| Fiber γ rows (HNLF 11.5 W⁻¹km⁻¹, SMF-28 1.2 W⁻¹km⁻¹, pinned file §10) | Not transcribed — no fiber-waveguide nonlinearity consumer or schema slot in `.mienlo` this phase. | A fiber/waveguide propagation model. |
