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

**Plate-like primitives reuse two shared builders instead of one-off
geometry code.** `_build_plate(doc, group, width_mm, thickness_mm,
round_flag, name=None)` builds a round (cylinder) or rectangular (box) pad
of diameter/edge-length `width_mm` and `thickness_mm`, front (−x) face at
x=0 — every plain plate primitive (`window`, `mirror_flat`,
`polarizer_plate`, `waveplate`, `filter_plate`, `grating_plate`,
`detector_plane`, `nd_filter`, `filter_bandpass`/`longpass`/`shortpass`/
`notch`, `diffuser_plate` before its face-property is applied, …) is
literally this function via the `_plate_from_params(doc, group, p)`
adapter, which just reads `p["width"]`/`p["thickness"]`/
`p.get("round_flag", 0)` off the primitive's own params dict.
`_build_wedge_plate(doc, group, width_mm, thickness_mm, wedge_deg,
round_flag, name=None)` builds the same shape but with the back face
tilted by `wedge_deg` (thickness increases toward +y; `wedge_deg == 0`
degenerates to a plain `_build_plate` call) — used by `window_wedged`,
`bs_plate`, `prism_wedge`, and shares its tilted-back-face technique with
`anamorphic_pair`'s hand-rolled 2-D profile. The **round_flag convention**
(`P(1|0, "", "1 = circular, 0 = rectangular")` in `params`, read via
`p.get("round_flag", <default>)` in the builder) is what the wizard's
`ParamTableWidget` (`mieworkbench/panes/element_wizard.py`) renders as a
"Circular shape" checkbox instead of a bare 0/1 number — add a
`round_flag` param to any new plate-like or source primitive to get that
checkbox for free. Follow the existing **width-as-diameter/edge-length**
wording in every `round_flag` primitive's `width`/`diameter` help text
(the v2 rename retired `radius`/`half-size` params in favor of full
diameters and widths — see `LEGACY_ALIASES` in `primitivelib.py` for how
old saved scenes built with the old params keep rebuilding).

For a plate primitive that needs a per-face property set on a
dynamically-located face (a coating on the front cap, a diffuser on the
back cap, …) rather than a whole-body prop, `_plate_with_face_prop(doc,
group, width_mm, thickness_mm, round_flag, prop_name, front_value=None,
back_value=None, name=None)` builds a plain `_build_plate` and then
locates the front (−x) and/or back (+x) face with
`_find_face_by_signed_normal` (sign-sensitive — the plain
`mts._find_face_by_normal` helper scores by `abs(dot)` and can't tell a
plate's front cap from its back cap, since both are exactly antiparallel
to the x-axis) before writing `"Face%d=%s" % (face_index, value)` into
`prop_name`. `pbs_plate`, `dichroic_plate`, `nd_reflective`, and
`diffuser_plate` (§5.4.1 of docs/RAYTRACER.md) are all one-line callers of
this helper; a new plate primitive that needs a coating/diffuser only on
one specific face should do the same instead of hardcoding a `FaceN=`
literal that would silently drift if `new_body_pad`'s face numbering ever
changed.

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

### 3.1 `derived_props`: syncing a sheet param to a body property across rebuilds

Some primitives compute a *contract* property (a real Base-group tag the
engine reads, like `absorbance`) directly from a *geometry* parameter on
the `dim` sheet, rather than exposing both separately. The `iris`,
`pinhole`, and `slit` apertures are the shipped example: their
`blackness` param (0.95–1.0, "fraction of incident power absorbed") is
turned into the disc/plate body's `absorbance` property by the builder
(`_build_iris`/`_build_pinhole`/`_build_slit` pass `"absorbance":
p["blackness"]` straight into `mts.pad_body`'s `props=`), so editing
`blackness` in the Element Properties pane and rebuilding changes
`absorbance` too, with no separate edit.

The wrinkle is `rebuild_element()`'s normal user-customization
preservation: on every rebuild it snapshots each old body's extra
`App::Property*` values (beyond `miewb_primitive`/`miewb_group`) and
restores them onto the freshly-built replacement, so hand-added tags and
edits made outside the sheet survive a topology-changing rebuild. Without
an escape hatch, that snapshot would restore the **old** `absorbance`
value onto the new body — overwriting the value the builder just
recomputed from the current `blackness`, and silently freezing
`absorbance` at whatever it happened to be the first time.

A primitive spec declares which of its properties are derived-from-sheet,
not preserve-across-rebuild, via a `"derived_props"` tuple:

```python
"iris": {
    ...
    "derived_props": ("absorbance",),
},
```

`rebuild_element()` unions `derived_props` into its `baseline` exclusion
set alongside `miewb_primitive`/`miewb_group`, so a listed property is
*never* included in the "extra props to restore" snapshot — the builder's
freshly-computed value always wins instead. Use this whenever a new
primitive derives a real contract property from one of its own `dim`-sheet
params: add the property name to `derived_props` so a rebuild can't
resurrect a stale value.

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
`scripts/raytracer/optprops.py` (polarizer/filter/grating/birefringence,
plus diffusers as of B6) and `scripts/raytracer/materials.py`
(materials/coatings, since those two predate the others and have their
own loader path).

The newest category, `diffuser/diffusers.miedif`, has no `tables/`
subdirectory — its rows are flat, `name,grit,slope_rms,reference`. Each
row supplies **either** `grit` (a catalog grit number, e.g. `120`) **or**
`slope_rms` (an RMS microfacet slope given directly), never both; the
shipped rows (`dg_120`, `dg_220`, `dg_600`, `dg_1500`) all use `grit`. As
with every other category, `reference` is mandatory (§5.2) — see
`opticalproperties/diffuser/diffusers.miedif` for the shipped rows and
`raytracer/roughness.py`'s `GRIT_FWHM_DEG` for the grit→scatter-angle
calibration that consumes them (docs/RAYTRACER.md §5.4.1).

#### 5.1.1 The registry-row generator (`scripts/tools/gen_registry_rows.py`)

Some registry families are large enough (a whole beamsplitter-ratio
ladder, a run of ND densities) that hand-typing each row invites
transcription drift between siblings. `scripts/tools/gen_registry_rows.py`
(stdlib-only, run from anywhere as `python3 scripts/tools/gen_registry_rows.py`)
generates and **idempotently upserts** several such families by deriving
every row from either a shipped source table or a closed-form formula, so
re-running it produces byte-identical output (`git diff` stays empty on a
repeat run) — it's meant to be safe to re-run after touching its source
tables, not a one-shot migration script:

- **BS ratio family** (`coating/coatings.miecoat` + per-row
  `coating/tables/bs_XXYY_vis_45.mietab`): `bs_3070`/`4060`/`6040`/`7030`/
  `9010`/`1090_vis_45`, each derived from the shipped `bs_5050_vis_45`
  table by rescaling its average R/T to the target ratio while keeping
  `Rs+Ts == Rp+Tp` equal to the source's per-wavelength total insertion
  loss, and scaling its s/p asymmetry by `4*rR*rT` so every channel stays
  in `[0,1]` even at extreme ratios (`gen_bs_ratio_family()`).
- **Pellicle rows** (`pellicle_4555_45`, `pellicle_uncoated_45`): flat
  R/T across the same wavelength grid as `bs_5050_vis_45`
  (`gen_pellicle_rows()`).
- **Reflective ND** (`coating/tables/nd_refl_od03..30.mietab`, 0° AOI,
  400–1100 nm): `T=10^-OD`, metallic absorption `A=0.25*(1-T)`, `R=1-T-A`
  (`gen_nd_refl_rows()`).
- **Absorptive ND filters** (`filter/filters.miefilt` +
  `filter/tables/nd_od01..40.mietab`): flat Beer-Lambert
  `T=10^-OD` at a 2 mm reference thickness (`gen_nd_filter_rows()`).
- **`shortpass_600`**: an exact mirror of the shipped `longpass_600`
  table about 600 nm (`gen_shortpass_row()`).
- **`notch_633_25`**: a synthetic OD4, 25 nm-FWHM Gaussian rejection
  notch centered at 633 nm, using the same Gaussian-FWHM parametrization
  `bp_550_40` uses for its passband, inverted into a notch
  (`gen_notch_row()`).

Every generated row's `reference` column names the generator and the
formula/basis used (`GEN_TAG = "generated by scripts/tools/gen_registry_rows.py"`),
so a generated row is never mistaken for a digitized vendor curve. Adding
a new generated family means writing one `gen_*()` function returning
`(rows, changed_table_filenames)` and wiring it into `main()`'s generator
list — `upsert_registry()`/`write_table()` handle the idempotent CSV
read-modify-write for you (matching the shipped CRLF line endings and
`csv.QUOTE_MINIMAL` quoting so untouched rows round-trip byte-for-byte).

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

### 6.1 Waveplate thickness solver

`core/wizards.py` also ships a standalone solver for the other inverse
problem a retarder primitive poses — target retardance → thickness —
alongside the lens-focal-length solvers above:

```python
waveplate_thickness(kind, lambda_nm, order=0, matdb=None, crystal="quartz")
```

`kind` is `"half"` (0.5-wave retardance) or `"quarter"` (0.25-wave);
`order` is the non-negative integer number of extra whole waves (a
first-order half-wave plate, `order=1`, retards 1.5 waves total). It looks
up the crystal's ordinary/extraordinary indices at `lambda_nm` from the
**same** birefringence registry the ray tracer itself uses
(`raytracer.optprops.load_optical_properties(...).matdb.get_uniaxial`,
lazily loaded and cached if no `matdb` is passed in, so
`waveplate_thickness("half", 633.0)` works standalone with no setup), then
solves `thickness_mm = (order + retardance_waves) * lambda_nm * 1e-6 /
|n_e - n_o|` — evaluating `|n_e - n_o|` **at** `lambda_nm` rather than a
fixed d-line constant, so a design at a non-visible wavelength stays exact
despite quartz's (weak) birefringence dispersion. It returns a dict
(`thickness`, `waves`, `n_o`, `n_e`, `delta_n`, plus the inputs echoed
back) so a caller can show the solved thickness alongside the indices that
produced it, the same way the lens designer's "Compute radii" shows
EFL/BFL. This maps directly onto the `waveplate` primitive's own
`thickness` dim-sheet alias (§2/§3): feed the result's `thickness` through
`project.set_spreadsheet("dim_<label>", "thickness", "=<value> mm")` +
`project.rebuild_primitive(label)`, exactly like the lens forms' `map()`
output.

Unlike the lens forms, this solver is **not yet wired to a dedicated box
in `ElementWizardDialog`** the way `design_lens` drives "Design by focal
length" — there is no `waveplate` entry in `LENS_FORMS`/`_FORM_FOR_PRIMITIVE`
(§6 above) triggering a "Design by retardance" section. It is fully
implemented and tested (`mieworkbench/tests/test_wizards.py`) and callable
directly, so it's ready to back such a box (or a scripted preset table of
common half-/quarter-wave designs) the same way `design_lens` backs the
lens forms — that GUI wiring is future work, not present behavior.
