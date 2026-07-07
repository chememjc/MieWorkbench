# Customizing MieWorkbench: new elements and optical properties

This document covers extending the two libraries MieWorkbench ships with:
the parametric **element library** (`primitives/*.FCStd` + `.meta.json`,
built by `scripts/primitivelib.py`) and the **optical property library**
(`opticalproperties/`, loaded by `scripts/raytracer/optprops.py` and
`scripts/raytracer/materials.py`). For the underlying tagging contract and
physics those two libraries feed into, see
[docs/RAYTRACER.md](docs/RAYTRACER.md) §5 (authoring contract) and §7
(optical properties); this document only covers *how to add to the
library*, not the full semantics of each tag or property model.

---

## 1. Primitive anatomy

Every entry in the Library pane's "Elements" tab (README.md §3.5) is one
`.FCStd` file under `primitives/`, plus an optional `.meta.json` sidecar
with the same stem. A primitive `.FCStd` contains:

- one `Spreadsheet::Sheet` labeled `dim`, with one aliased cell per
  geometry parameter — raw cell content `"=<value> mm"` / `"=<value>
  deg"` / a bare number for unitless counts (facet counts, etc.);
- one `PartDesign::Body` built from those parameter values (two, for a
  multi-body element like an achromat doublet or a PBS cube), tagged with
  the usual Base-group contract properties (`material`/`power`/`lambdac`/
  … — docs/RAYTRACER.md §5.1) **plus** two GUI-internal tags:
  - `miewb_primitive` (string) — which `PRIMITIVES` registry entry built
    this body,
  - `miewb_group` (string) — shared by every body of one multi-body
    element, so the GUI can rebuild them together.

**Why rebuild-on-edit, not constraint expressions:** changing a
primitive's parameters can change *topology* — a radius of curvature
going to zero turns a curved surface into a flat one, a facet-count change
adds/removes faces, some forms flip sign conventions (e.g. plano-convex
vs. plano-concave). No FreeCAD expression binding can restructure a
document's feature tree like that. So the `dim` sheet is treated as the
single source of truth, and editing a primitive's parameters in the
Element Properties pane (README.md §3.3) re-runs the whole builder
function with the new alias values — `fcserver`'s `rebuild_primitive` op —
rather than recomputing expressions in place. `rebuild_element()` in
`scripts/primitivelib.py` saves off each body's Label, Placement, and any
extra custom properties before deleting and rebuilding, then restores
them, so a rebuild is otherwise non-destructive. **Hand-authored
primitives that use real FreeCAD cell expressions** (Route A below) are
not subject to this — they go through the ordinary `set_spreadsheet` →
document recompute path, since their geometry genuinely is
expression-driven.

---

## 2. Route A: drop in a hand-authored `.FCStd` primitive

The simplest way to add an element: author it in the FreeCAD GUI like any
other `PartDesign` model, bind its geometry to a `dim`-labeled spreadsheet
via expressions (e.g. a Pad length set to `<<dim>>.length`), tag the body
with the usual Base contract properties, save it as `primitives/<kind>.FCStd`,
and press **Refresh** in the Library pane's Elements tab — it picks up any
`.FCStd` dropped into `primitives/` with no code changes.

Add an optional `primitives/<kind>.meta.json` sidecar to control how it's
labeled and parameterized in the GUI. Exact schema (every key as actually
read/written by `scripts/primitivelib.py` and `scripts/make_primitives.py`
— confirmed against `primitives/lens_dcx.meta.json`):

```json
{
 "kind": "lens_dcx",
 "category": "Lenses",
 "label": "Biconvex lens",
 "tooltip": "Convex both sides (R1 front, -R2 back).",
 "params": {
  "R_front": {"default": 40.0, "unit": "mm", "help": "front radius (>0)"},
  "R_back":  {"default": 40.0, "unit": "mm", "help": "back radius magnitude (>0)"},
  "ct":      {"default": 6.0,  "unit": "mm", "help": "center thickness"},
  "aperture":{"default": 20.0, "unit": "mm", "help": "clear aperture diameter"}
 },
 "props": {"material": "bk7"}
}
```

| Key | Meaning |
|---|---|
| `kind` | the primitive's identifier (matches the `.FCStd` stem) |
| `category` | grouping shown in the Library tree (`Sources`, `Lenses`, `Mirrors`, …) |
| `label` | human-readable name shown in the Library and used to build a default element label |
| `tooltip` | one-line description shown in the Library's details panel |
| `params` | `{alias: {"default": number, "unit": "mm"\|"deg"\|"", "help": text}}` — must match the aliases actually present on the `.FCStd`'s `dim` sheet; `unit` is `""`/omitted for unitless counts |
| `props` | default Base-group tag values applied when the element is added to a scene (e.g. a lens's default `material`) |

Without a `.meta.json`, the Library still lists the primitive (it falls
back to introspecting the `.FCStd`'s own `dim` sheet aliases/defaults
directly), but you lose per-parameter help text and a curated default
category/label/tooltip.

---

## 3. Route B: a coded builder in `primitivelib.py`

For a primitive whose geometry is genuinely parametric in a way that's
awkward to hand-author (or that you want to keep regeneratable from
source), add an entry to the `PRIMITIVES` dict in `scripts/primitivelib.py`
and a builder function. `PRIMITIVES` is a plain dict of dicts, importable
without FreeCAD (guarded imports) so the GUI can list primitives, params,
and defaults under plain Python — only *building* one needs the FreeCAD
AppImage. A real entry (`lens_dcx`):

```python
"lens_dcx": {
    "category": "Lenses", "label": "Biconvex lens",
    "tooltip": "Convex both sides (R1 front, -R2 back).",
    "params": {"R_front": P(40.0, "mm", "front radius (>0)"),
               "R_back": P(40.0, "mm", "back radius magnitude (>0)"),
               "ct": P(6.0, "mm", "center thickness"),
               "aperture": P(20.0, "mm", "clear aperture diameter")},
    "props": {"material": "bk7"},
    "meridian": lambda p: (p["R_front"], -p["R_back"]),
},
```

(`P(default, unit, help)` is a tiny helper that builds the
`{"default":…, "unit":…, "help":…}` dict; spherical-lens entries carry an
extra `"meridian"` key a shared factory turns into a lens-revolve builder
— most other primitives instead point at their own builder function
directly.)

A builder function has the signature `fn(doc, group, p) -> [bodies]`
(`doc` = the FreeCAD document, `group` = the element's group/label, `p` =
the resolved parameter dict) and is responsible for: creating the
geometry (reusing `make_test_scenes.py`'s helpers — `lens_meridian`,
`revolve_body`, `pad_body`, `new_body_pad`, …), applying `props` (the
default tags) via `safe_set_props`, and tagging every produced body with
`miewb_primitive`/`miewb_group` via `_tag()`. `build_primitive()` ties
this together: build → apply props → tag → `doc.recompute()`. The `dim`
spreadsheet itself is built generically by `make_sheet()`, which writes
one row per `params` entry (`A<row>` = alias name, `B<row>` = the raw
`"=<value> <unit>"` cell, aliased to the parameter name) — you don't write
sheet-construction code per primitive, only the geometry builder.

**FreeCAD alias restriction:** FreeCAD's spreadsheet rejects an alias that
looks like a cell address (`R1`, `A2`, …) with "Invalid alias" — this is
native FreeCAD behavior, not something this repo validates for you. Name
parameters to avoid the collision: use `R_front`/`R_back` rather than
`R1`/`R2` (every primitive in the registry already follows this
convention — note `lens_dcx`'s *tooltip* says "R1 front, -R2 back" as
prose shorthand, but its actual parameter aliases are `R_front`/`R_back`).

Once your entry and builder are in place, regenerate the library:

```bash
/home3/freecad/FreeCAD.AppImage -c scripts/make_primitives.py -- \
    --kind lens_dcx < /dev/null   # or --kind all to rebuild everything
```

This writes both `primitives/<kind>.FCStd` (via `doc.saveAs`) and
`primitives/<kind>.meta.json` (a JSON dump of the same `category`/`label`/
`tooltip`/`params`/`props` — minus the non-serializable `meridian` lambda,
if present), after a recompute sanity check that hard-fails if any object
came out `Invalid`/`Error`.

---

## 4. Body-tagging contract — quick reference

Every optic/source/detector body carries some subset of these
`App::Property*` custom properties in group "Base" (full semantics,
classification rules, and precedence in **docs/RAYTRACER.md §5.1** — this
is only a name/type/purpose summary):

| Property | Type | Purpose |
|---|---|---|
| `material` | String | registry row in `materials.miemat`, a crystal name in `uniaxial.miebrf`, `"detector"`, or absent/`"none"` (body ignored) |
| `power` (mW) + `lambdac` (nm) | Float | presence of both marks the body a **source** |
| `coating` | String | whole-body name, or per-face map `'Face3=MgF2;Face5=x'` |
| `roughness` | Float or String | whole-body RMS nm, or per-face map `'Face1=200:lcorr=5;Face2=50'` |
| `filter` | String | a `filters.miefilt` row name |
| `polarizer` + `polarizer_axis` | String, `'x,y,z'` | a `polarizers.miepol` row name + body-local transmission axis |
| `polarizer_axis`/`crystal_axis` default | — | `0,0,1` / `+x` respectively when absent |
| `crystal_axis` | String `'x,y,z'` | body-local optic axis for a birefringent `material` |
| `grating` | String | per-face map only: `'Face2=600:v:orders=-1..1'` or `'Face2=@registryname'` |
| `surface_override` | String | per-face asphere declaration: `'FaceN=asphere:R=..;k=..;A4=..;...;r_max=..'` |
| `mirror` | Float `[0,1]` | achromatic partial-reflector fraction |
| `absorbance` | Float `[0,1]` | fraction of the non-mirror remainder absorbed |
| `polarization`, `lambdamin`/`lambdamax`, `coherent` | source-only | emission spectrum/polarization — see docs/RAYTRACER.md §5.2 |

`coating`/`roughness`/`grating`/`surface_override` are the four properties
that support a per-face map form (`FaceN=value;FaceM=value`); the Element
Properties pane's "Per-face assignments" section (README.md §3.3) edits
exactly these four.

---

## 5. Adding optical properties

### 5.1 File anatomy per category

Each category is one registry file (rows = named entries) plus, for
categories whose entries reference tabulated spectral data, a
`tables/` (or `nk/`) subdirectory of per-item CSV files. All content is
plain CSV under self-describing extensions — see README.md §4.4 for the
exact required-column table per category, and docs/RAYTRACER.md §7 for
full physical-model semantics of each column. Loaders:
`scripts/raytracer/optprops.py` (polarizer/filter/grating/birefringence)
and `scripts/raytracer/materials.py` (materials/coatings, since those two
predate the others and have their own loader path).

### 5.2 Citation policy

Every registry row **requires a non-empty `reference` column** — every
loader hard-validates this and raises rather than silently accepting an
uncredited entry (`optprops.py`'s shared `_read_registry()` helper, and
the equivalent checks in `materials.py` for `materials.miemat` and
`coatings.miecoat`). Put a real citation (paper, standard, datasheet, or
"n=1 by definition" for vacuum) — a placeholder string will pass the
loader but defeats the point; reviewers should treat an unconvincing
`reference` the same as a missing one.

### 5.3 Using the GUI Property Library Editor

Library pane → **Open in editor** (either the Project or System library
tab) opens the property editor: one tab per registry, each showing the
registry's own CSV header as table columns (so a schema change shows up
automatically, nothing to keep in sync in the GUI). Cells are read-only
until you toggle **Edit**; committing gathers the table back into rows,
refuses if any `reference` cell is empty, atomically rewrites the
registry CSV, and re-validates through the real loader — on any failure
every touched file is rolled back and the loader's own error is shown, so
you can't commit a change that would break `load_optical_properties()`.

**Import table…** maps an arbitrary external CSV's columns onto a
category's required table schema (e.g. `wavelength_nm,n,k` for a
materials nk-table, `wavelength_nm,Rs,Rp,Ts,Tp` for a coating table) via a
column-mapping dialog: you pick which source column feeds each required
destination column, and the (independently unit-tested) mapping function
converts every mapped cell to a float, raising a clear error naming the
row/column if something isn't numeric — so importing a vendor datasheet
export with oddly-named columns doesn't require pre-editing the CSV by
hand.

Selecting a row that references a spectral table plots it inline
(PySide6 QtCharts — no matplotlib in the GUI process).

### 5.4 Project vs. system library, and promoting entries

Two libraries exist side by side (README.md §3.5): the **system library**
at `<repo>/opticalproperties/`, and the **project library** inside a
`.MieWB` workspace's own `opticalproperties/` — normally a trimmed subset
containing just the rows/tables a given model actually uses, so the
project (or its `.MieWB`) is self-contained and portable. The editor's
library selector switches every tab between the two (the project tabs are
disabled when no project root is open).

Adding an entry while a project is open writes it to the *project*
library by default. To make a project-local addition available to every
future project, **promote it to the system library**: this copies the
row (and any referenced table/nk file it points at) from the project
registry into the system one, through the same validated
write-with-rollback path as an ordinary edit. If the system library
already has a same-named entry with *different* content, promotion stops
and reports the conflicting rows instead of silently overwriting — you
decide whether to force the overwrite.

---

## 6. Lens wizard forms

`mieworkbench/core/wizards.py` solves the inverse problem — target focal
length → radii — for a fixed set of lens forms, then cross-checks the
result against the real thick-lens EFL/BFL formula. `LENS_FORMS` maps each
form to the primitive it designs and how its solver's output maps onto
that primitive's `dim`-sheet aliases:

| Form | Label | Primitive | Solver inputs | Maps to params |
|---|---|---|---|---|
| `pcx` | Plano-convex | `lens_pcx` | f>0, n, d | `R_front`, `ct` |
| `pcv` | Plano-concave | `lens_pcv` | f<0, n, d | `R_back`, `ct` |
| `dcx` | Biconvex (symmetric) | `lens_dcx` | f>0, n, d | `R_front`, `R_back` (negated), `ct` |
| `dcv` | Biconcave (symmetric) | `lens_dcv` | f<0, n, d | `R_front` (negated), `R_back`, `ct` |
| `best` | Best-form singlet | `lens_meniscus` | f>0, n, d | `R_front`, `R_back`, `ct` |
| `asphere` | Aspheric (conic) | `lens_asphere` | f>0, n, d | `R`, `k`, `ct` |
| `ball` | Ball lens | `lens_ball` | f>0, n | `diameter` |
| `achromat` | Achromatic doublet | `lens_achromat` | f (scales a reference BK7/SF5 f=50mm design) | `R_front`, `R_iface`, `R_back`, `ct_crown`, `ct_flint` |
| `fresnel` | Fresnel lens | `lens_fresnel` | f, n, aperture, n_facets | `f_design`, `n_design`, `aperture`, `n_facets` |
| `cyl` | Cylindrical | `lens_cyl` | f, n, d | `R` |

`design_lens(form, f_mm, matdb=None, material="bk7", ...)` is the one-call
entry point: it resolves the chosen material's real index at the design
wavelength from the property library (when a `matdb` is supplied), runs
the form's solver, and applies the form's `map()` lambda to produce
`{alias: value}` — exactly the aliases the target primitive's `dim` sheet
uses.

In the GUI (`panes/wizard_dialog.py`'s `ElementWizardDialog`), choosing a
lens primitive that has a matching form shows a "Design by focal length"
box (focal length + material + **Compute radii**); computing overwrites
the matching rows of the dialog's own parameter table and shows the
resulting EFL/BFL as a cross-check. Accepting the dialog writes only the
parameters that differ from the primitive's shipped defaults, via
`project.set_spreadsheet("dim_<label>", alias, "=<value> <unit>")`,
followed by `project.rebuild_primitive(label)` to regenerate the actual
FreeCAD geometry from the new `dim`-sheet values — the same rebuild-on-
edit path described in §1.
