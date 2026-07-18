# Authoring

Pointer page — new elements and new optical-property entries are covered
in depth elsewhere; this page just routes you.

## New primitives (parametric elements)

`primitives/*.FCStd` + `.meta.json`, built by `scripts/primitivelib.py`.
See [`../../CUSTOMIZE.md`](../../CUSTOMIZE.md) §1 for primitive anatomy
(the `dim` sheet, the `miewb_primitive`/`miewb_group` GUI-internal tags,
why rebuild-on-edit rather than live constraint expressions) and the
rest of that document for the builder registration steps. The GUI
surfaces a finished primitive through the
[Library browser + element wizard](library-browser.md); per-body/per-face
tagging semantics live in [`../RAYTRACER.md`](../RAYTRACER.md) §5 (the
full authoring contract).

## New optical-property entries

`opticalproperties/` registries (materials, coatings, polarizers,
filters, gratings, birefringence, figure errors, nonlinear, scatter,
instruments, …), loaded by `scripts/raytracer/optprops.py` and
`scripts/raytracer/materials.py`. See
[`../../CUSTOMIZE.md`](../../CUSTOMIZE.md) for the how-to (adding a row,
attaching a spectral table, the citation requirement) and
[`../RAYTRACER.md`](../RAYTRACER.md) §7 for full schema semantics and the
physics each category feeds into. The GUI surface is the
[Property Library Editor](property-library-editor.md) — per-column
tooltips there are generated from the same schema documented in
`mieworkbench/core/libschema.py`.
