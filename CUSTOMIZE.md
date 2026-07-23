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
old saved scenes built with the old params keep rebuilding). The mapping
is a small inline dict, `{"diameter": ("radius", 2.0), "width": ("half",
2.0)}` (new-alias → (old-alias, scale-factor-to-new)), read by both
`read_params()` (rebuilding a body saved under the old param name) and
`port_frames()`'s `_port_params()` helper (§3b below, same fallback so
chain placement works on an old scene too). Adding a new legacy alias
means one new dict entry; a brand-new param with no legacy predecessor
needs no entry at all — it just falls back to the spec default.

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

Once your entry and builder are in place, regenerate the library. (Commands
below assume a one-time `scripts/setup_env.sh` and, per shell, `source
scripts/miewb_env.sh` — INSTALL.md §5 — which puts `$MIEWB_FREECAD` in
your environment.)

```bash
"$MIEWB_FREECAD" -c scripts/make_primitives.py -- \
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

The asphere-backed primitives are the second shipped example, and the
sharper trap: `lens_asphere` and `mirror_parabolic` write a
`surface_override` string computed from their R/k/rfl/aperture params.
Without `derived_props: ("surface_override",)`, editing a parabolic
mirror's focal length rebuilds the geometry but restores the OLD override
string — and the next extraction **hard-fails** on the <1 µm asphere
verification (this is exactly how the Newtonian demo first died). If your
builder writes `surface_override` (or any face-map string tied to
geometry), list it here.

Two more shipped examples follow the same pattern with a different
recomputed property: the **cube-beamsplitter primitives** (`pbs_cube`,
`bs_cube`) derive their `coating` string (`'Face<N>=<row>'`) from the
splitter-diagonal plate's own freshly-built face index — `derived_props:
("coating",)` — so a rebuild after resizing the cube can't leave the
coating pointing at a stale/renumbered `FaceN`. The **simple-lens
family**'s `edge_blackened` flag (`lens_pcx`/`lens_dcx`/`lens_pcv`/
`lens_dcv`/`lens_meniscus`) is `derived_props: ("edge_blackened",)` for a
subtler reason: the builder just sets the bool itself (the *extractor*,
not the builder, turns a truthy flag into per-face absorbance on the
cylindrical barrel at extract time), but it still needs the derived-prop
escape hatch so toggling `edge_blackened` in the sheet and rebuilding
doesn't have the generic prop-preservation path restore the old value
over the new one.

---

## 3b. Port frames — make a new primitive chain-able

The optical-train chain model positions elements by vertex-to-vertex
distances along the beam, using each primitive's ELEMENT-LOCAL port
geometry from `primitivelib.port_frames(kind, params)`. When you add a
primitive (either route), give it a `port_frames` entry or chained
placement falls back to a thin-element approximation at the optical
center (distances become center-to-center):

```python
# in port_frames()'s kind table — pure math over the dim params,
# NO FreeCAD, NEVER FaceN indices (they renumber on rebuild):
#   {"entry": [x,y,z], "exit": [x,y,z],  # axis-pierce points of the
#                                        # first/last optical surfaces
#    "axis": [1,0,0], "up": [0,0,1],     # library convention: beam +x
#    "reflect_plane": {"point": ..., "normal": ...} or None}
```

Conventions (every shipped builder follows them): body-local beam along
+x with the FRONT vertex at x=0; `up` = +z; mirrors have entry == exit
== the axis/surface intersection and a `reflect_plane` whose normal
points back INTO the incoming beam; sources emit from local x=0 (the
body extends toward -x); reflective gratings carry the reflect plane of
the grating face. Derive the formulas from the builder's own sketch
math and cross-check against the shipped `.FCStd` bbox (see the
verification notes in `port_frames`'s docstring). Hand-authored bodies
can instead carry a `miewb_train_ports` JSON property with the same
dict.

As of this writing, `lens_cyl`, `lens_fresnel`, `retro_corner_cube`, the
right-angle/dove/penta/rhomboid prisms, `anamorphic_pair`, and the
Glan-Taylor polarizer still fall back to the bbox heuristic (thin-element,
center-to-center chaining) instead of a real `port_frames` formula —
contributing one for any of these is a good first authoring task.

---

## 4. Body-tagging contract — quick reference

Every optic/source/detector body carries some subset of these
`App::Property*` custom properties in group "Base" (full semantics,
classification rules, and precedence in **docs/RAYTRACER.md §5.1** — this
is only a name/type/purpose summary):

| Property | Type | Purpose |
|---|---|---|
| `material` | String | registry row in `materials.miemat`, a crystal name in `uniaxial.miebrf` or `biaxial.mibiax`, `"detector"`, or absent/`"none"` (body ignored) |
| `power` (mW) + `lambdac` (nm) | Float | presence of both marks the body a **source** |
| `coating` | String | whole-body name, or per-face map `'Face3=MgF2;Face5=x'` |
| `roughness` | Float or String | whole-body RMS nm, or per-face map `'Face1=200:lcorr=5;Face2=50'` |
| `diffuser` | String | ground glass, whole-body or per-face map: `'grit:120'` \| `'slope:0.08'` \| `'@dg_600'` |
| `scatter` | String | measured ABg/BSDF registry name, whole-body or per-face map — mutually exclusive with `roughness`/`diffuser` on the same face (§9 below) |
| `filter` | String | a `filters.miefilt` row name |
| `polarizer` + `polarizer_axis` | String, `'x,y,z'` | a `polarizers.miepol` row name + body-local transmission axis |
| `polarizer_axis`/`crystal_axis` default | — | `0,0,1` / `+x` respectively when absent |
| `crystal_axis` | String `'x,y,z'` | body-local optic axis for a uniaxial birefringent `material`; X principal axis for a biaxial one |
| `crystal_axis2` | String `'x,y,z'` | body-local Y principal axis — REQUIRED (with `crystal_axis`) when `material` is biaxial (§10 below); no default |
| `grating` | String | per-face map only: `'Face2=600:v:orders=-1..1'` or `'Face2=@registryname'` |
| `surface_override` | String | per-face asphere declaration: `'FaceN=asphere:R=..;k=..;A4=..;...;r_max=..'` |
| `mirror` | Float `[0,1]` | achromatic partial-reflector fraction |
| `absorbance` | Float `[0,1]` | fraction of the non-mirror remainder absorbed |
| `polarization`, `lambdamin`/`lambdamax`, `coherent` | source-only | emission spectrum/polarization — see docs/RAYTRACER.md §5.2 |
| `apodization`, `beam_waist` + `m2` | String / Float | source-only: transverse field taper and Gaussian-beam mode — see docs/RAYTRACER.md §5.2 |

`coating`/`roughness`/`diffuser`/`scatter`/`grating`/`surface_override` are
the six properties that support a per-face map form
(`FaceN=value;FaceM=value`); the Element Properties pane's "Per-face
assignments" section (README.md §3.3, `core/facemaps.py`'s
`FACEMAP_PROPERTIES`) edits exactly these six.

---

## 5. Adding optical properties

### 5.1 File anatomy per category

Each category is one registry file (rows = named entries) plus, for
categories whose entries reference tabulated spectral data, a
`tables/` (or `nk/`) subdirectory of per-item CSV files. All content is
plain CSV under self-describing extensions — see README.md §4.4 for the
exact required-column table per category, and docs/RAYTRACER.md §7 for
full physical-model semantics of each column. Loaders:
`scripts/raytracer/optprops.py` (polarizer/filter/grating/birefringence
— both uniaxial and biaxial, §10 below — plus diffusers as of B6 and
scatter, §9 below, as of the `lowhanging-improvements` round; and, from
later rounds, detector QE curves (§7 below), emission SPDs (§8b below),
figure error (§11b below), nonlinear/EO rows (§11 below), and virtual
instruments (§7b below) — see each section's own `load_*` docstring for
its exact column set) and `scripts/raytracer/materials.py` (materials/
coatings, since those two predate the others and have their own loader
path).

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

---

## 7. Adding detector quantum-efficiency (QE) curves

Detector bodies can carry a `qe_curve` property (string) naming a registered
quantum-efficiency curve from `detector/detectors.miedet`. When set,
`post_process.py` adds a `qe` block to that detector's `report.json` entry:
`photocurrent_A` (responsivity R(λ)=QE·qλ/hc applied to the spectral cube),
`qe_weighted_power_W`, and `coverage_frac` — the fraction of detected power
inside the QE table's wavelength range (QE is zero-filled outside the table,
never extrapolated; coverage_frac makes the truncation visible). There is no
CLI flag — the body property alone drives it. (The unrelated `--photometric`
flag produces lux maps from the CIE V(λ) luminosity function.)

To add a new QE curve:

1. **Append a row to `detector/detectors.miedet`:**
   - `name`: registry key (e.g., `my_detector_qe`)
   - `table_csv`: filename reference (e.g., `my_detector_qe.mietab`)
   - `reference`: required citation (manufacturer datasheet, peer-reviewed measurement, etc.)
   - `notes`: optional description (e.g., "silicon photodiode, 20–1100 nm")

2. **Create the wavelength-QE table as `detector/tables/<name>.mietab`:**
   - Two columns: `wavelength_nm, qe` (values in (0,1], strictly increasing wavelength)
   - Minimum 2 rows; linear interpolation between points, zero-fill outside
     the range (reported via `coverage_frac`)
   - Example:
     ```
     wavelength_nm,qe
     660,0.65
     960,0.55
     ```

3. **Tag a detector body:** in the GUI's **Properties** pane, set the body's
   Base-group `qe_curve` property to your registry name. On run, `post_process.py`
   will compute the wavelength-weighted photocurrent and emit the metrics above.

See `detector/detectors.miedet` (shipped `hamamatsu_s1223` row) and the
staged-but-unshipped CMOS rows in `library_data/staged/detector_qe.csv`.

---

## 7b. Adding a virtual instrument (camera / powermeter / spectrometer)

Detector bodies can also carry an `instrument` property (string, `'row'`
or `'row:mode'`, mode `ideal`\|`full`, default `full`) naming a row in
`instrument/instruments.mieinst`. Where `qe_curve` (§7 above) is a fixed
responsivity weighting folded into the plain power report, `instrument`
drives a full **post-process response model of a real bench instrument**
over the SAME already-computed ideal spectral cube —
`post_process.py`'s `render_instrument` dispatcher writes an
`instrument` block into that detector's `report.json` entry (`ideal`
mode: deterministic response only; `full` mode: additionally draws a
reproducible noise chain seeded from the run's own seed). See
docs/RAYTRACER.md §7.11 for the full output/report-field reference.

`instrument/instruments.mieinst` is one WIDE CSV — a single header row
covers every `class`, and a given row only fills the columns its own
class uses (blank elsewhere), the same sparse-row shape as
`nonlinear/nonlinear.mienlo` (§11 below). `class` discriminates the
schema: `camera`, `powermeter`, `spectrometer`, `diode_array` (P12/
samples-instruments: a physical linear-array readout — reuses `camera`'s
electron/noise/ADC columns plus `spectrometer`'s `stray_light_floor`/
`detector_qe_table`, adding its own `pixel_height_um`/`n_px` real-array
geometry columns) ship rows today;
`polarimeter`, `wavefront_sensor`, `autocorrelator` are schema-defined
**placeholder classes** (fully validated, no shipped rows — bench gear
the project doesn't own yet). Don't remove their validation branches just
because `len(instruments)` shows none of them.

To add a new instrument row:

1. **Append a row to `instrument/instruments.mieinst`** with `name`,
   `class`, `reference` (required citation — a real datasheet), optional
   `notes`, plus your class's own columns (`optprops.load_instruments`'s
   docstring is the source of truth; verify against it before authoring):
   - `camera`: `pixel_pitch_um`, `width_px`/`height_px` (int, >0),
     `fill_factor` (0,1], `qe_table` (→ a `wavelength_nm,qe` table, values
     in (0,1]), `full_well_e`, `read_noise_e` (>=0),
     `dark_current_e_per_s` (>=0), `bit_depth` (int>0),
     `adc_gain_e_per_dn`, `integration_time_s_default`.
   - `powermeter`: EXACTLY ONE of `responsivity_table` (→ a
     `wavelength_nm,responsivity_a_w` table, values >0) or
     `flat_responsivity_a_w` (a single float) — leave the other column
     blank — plus `aperture_mm`, `nep_w_per_sqrthz`, `bandwidth_hz` (>0),
     `display_digits` (int>=1).
   - `spectrometer`: `lam_lo_nm` < `lam_hi_nm`, `resolution_fwhm_nm`
     (>0), `slit_um` (>0), `stray_light_floor` ([0,1)),
     `detector_qe_table` (→ a `wavelength_nm,qe` table, values in (0,1]).
   - `diode_array`: `pixel_pitch_um`, `pixel_height_um`, `n_px` (int>0,
     real array pixel count), plus the SAME `full_well_e`/`read_noise_e`/
     `bit_depth`/`adc_gain_e_per_dn`/`integration_time_s_default` columns
     `camera` uses and the same `stray_light_floor`/`detector_qe_table`
     columns `spectrometer` uses (no separate columns for these — the
     loader reuses the identical fields across classes).
   - `polarimeter`/`wavefront_sensor`/`autocorrelator` (placeholder,
     validated but no shipped consumer in `post_process.py` yet):
     `analyzer_states` (int>=2)/`extinction_ratio`/`retarder_error_deg`;
     `opd_sampling_um`/`reference_arm_model`; `shg_crystal`/
     `delay_range_fs` respectively — author one only alongside the render
     path that would consume it.

2. **Create any referenced table(s)** under `instrument/tables/<file>` in
   the two-column shape named above (header row, strictly increasing
   wavelength, minimum 2 rows — the same `_read_table` convention as
   every other tabulated registry).

3. **Tag a detector body:** in the GUI's **Properties** pane, set the
   body's `instrument` property to `<row>` or `<row>:ideal`/`<row>:full`.
   On run, the Results pane's "Instrument" tab shows the rendered
   output; `--instruments off` on the CLI skips the layer even when a
   detector carries the property.

See `opticalproperties/instrument/instruments.mieinst` for the four
shipped rows (`camera_generic`, `powermeter_generic`,
`spectrometer_generic`, `tcd1304_array`, each citing a real datasheet) and
docs/RAYTRACER.md §7.11 for the full report-field reference.

---

## 8. LED monochromatic source presets

Eight LED monochromatic-source primitives ship as `led_*` entries in
`scripts/primitivelib.py`, driven by published center wavelength (CWL) and
full-width-at-half-maximum (FWHM) data in `library_data/emission_led_monochromatic.csv`.
Each preset is a `source_broadband` geometry (circular or rectangular emit face,
same builders as the generic broadband source) with:
- `lambdac`: the LED's center wavelength (nm)
- `lambdamin`/`lambdamax`: CWL ± FWHM/2.3548 (Gaussian approximation, normalized so the
  integral of a Gaussian with FWHM σ is correct)
- `coherent`: False (direct deposit, not coherent gather)
- `power`: default 5 mW (tunable per instance)

The eight presets are:
- `led_deep_red_660`: 660 nm, FWHM 20 nm
- `led_red_630`: 625 nm, FWHM 20 nm
- `led_amber_590`: 590 nm, FWHM 20 nm
- `led_green_525`: 527 nm, FWHM 30 nm
- `led_blue_470`: 472 nm, FWHM 20 nm
- `led_royal_blue_450`: 452 nm, FWHM 20 nm
- `led_uv_365`: 365 nm, FWHM 9.0 nm
- `led_uv_385`: 385 nm, FWHM 11 nm

To add a new LED preset:

1. **Source data:** add a row to `library_data/emission_led_monochromatic.csv` with
   the LED's name, CWL, and FWHM (from datasheet).

2. **Primitive entry:** add an entry to `PRIMITIVES` dict in `scripts/primitivelib.py`:
   ```python
   "led_wavelength_abbrev": {
       "category": "Sources", "label": "Color LED (λ nm)",
       "tooltip": "Monochromatic LED source (CWL λ nm, FWHM XX nm; reference datasheet).",
       "params": {"diameter": P(10.0, "mm", "…"),
                  "length": P(10.0, "mm", "…"),
                  "round_flag": P(1, "", "…")},
       "props": {"power": 5.0, "lambdac": <CWL>,
                 "lambdamin": <CWL - FWHM/2.3548>,
                 "lambdamax": <CWL + FWHM/2.3548>,
                 "coherent": False},
   }
   ```
   (The parameters and tooltip follow the existing LED pattern; all eight current
   presets use the same geometry and differ only in their source properties.)

The existing presets' data come from Cree/Lumileds/Nichia datasheets, cited in
the CSV header. Treat any non-datasheet source as UNVERIFIED and flag it accordingly.

---

## 8b. Adding an emission spectrum (SPD) registry entry

A source body's `spectrum` property (string) names a row in
`emission/emitters.miesrc` — a tabulated relative spectral power density
that **supersedes** `lambdamin`/`lambdamax` (the sampler in
`sources.wavelength_strata` normalizes the table to a PDF and draws
equal-power inverse-CDF wavelength strata, so only the table's SHAPE
matters, not its absolute scale). This is how a source gets a realistic
digitized/tabulated spectral profile instead of the LED presets' (§8
above) Gaussian approximation.

Only `kind=continuous` (piecewise-linear PDF) is supported this round —
`blackbody` and `line` are staged kinds the loader REJECTS by name; don't
author a row with either yet.

To add a new emission spectrum:

1. **Append a row to `emission/emitters.miesrc`:**
   - `name`: registry key (e.g., `my_source_spd`)
   - `kind`: `continuous` (the only value the loader accepts today)
   - `table_csv`: filename reference (e.g., `my_source_spd.mietab`)
   - `reference`: required citation (datasheet spectral plot, published
     measurement, etc.)
   - `notes`: optional — state digitization confidence/error bars if the
     table was read off a plot rather than a tabulated dataset

2. **Create the table as `emission/tables/<name>.mietab`:**
   - Two columns: `wavelength_nm, relative_power` (arbitrary units — only
     the SHAPE matters), strictly increasing wavelength, minimum 2 rows
   - Validation: `relative_power >= 0` everywhere; the table's integral
     over wavelength must be `> 0` (an all-zero table carries no power
     and is rejected)

3. **Tag a source body:** set the body's `spectrum` property to your
   registry name. `spectrum` supersedes `lambdamin`/`lambdamax` — leave
   them as-authored (ignored once `spectrum` is set) or drop them for
   clarity.

See `opticalproperties/emission/emitters.miesrc` for the two shipped rows
(`led_white_2733k` — CIE 015:2018 std illuminant LED-B1; `sc_superk` —
NKT SuperK EXTREME datasheet SPD, digitized with a noted ±20–30%
visual-digitization caveat and a clipped 1064 nm residual pump spike) and
§11 below for how a supercontinuum-style pulsed-source primitive pairs
`spectrum=<row>` with its own emission table.

---

## 9. Adding a measured-scatter (BSDF/ABg) registry entry

`opticalproperties/scatter/bsdf.miebsdf` (loader: `optprops.load_scatter()`,
docs/RAYTRACER.md §7.9/§5.4.2) is the ABg-model measured-scatter registry
consumed by a body's `scatter` property. **Scope: reflected-side (BRDF)
always; an optional transmitted-side (BTDF) lobe** via the `btdf`,
`btdf_A`, `btdf_B`, `btdf_g`, `btdf_tis_cap` columns (each `btdf_*` column
defaults to its reflected-side `A`/`B`/`g` counterpart when left blank —
see the shipped `lightly_ground_glass_window` row for a worked BTDF
example). No per-azimuth anisotropy is modeled on either side, so don't
try to model a grooved/turned surface's directional lobe with this schema.

1. **Append a row to `scatter/bsdf.miebsdf`:**
   - `name`: registry key (e.g., `my_ground_aluminum`)
   - `model`: `abg` (the only supported value today)
   - `A`, `B`, `g`: the reflected-side (BRDF) ABg fit coefficients
     (`BSDF(u) = A/(B + u^g)`, `u` the direction-cosine offset from
     specular) — all three **must be `> 0`**. `g` is typically close to 2
     for polished surfaces (2 gives a closed-form radial CDF, so sampling
     needs no per-call tabulation; other `g` values fall back to a
     numeric inverse-CDF, which works but costs more per gather call).
   - `tis_cap` (optional): a ceiling in `(0, 1]` on the total integrated
     scatter the loader computes from `A`/`B`/`g` at normal incidence —
     use this when a measured total-scatter number is known but the raw
     ABg fit would over-integrate it (see the shipped
     `diamond_turned_aluminum` row, capped at 0.1).
   - `btdf`, `btdf_A`, `btdf_B`, `btdf_g`, `btdf_tis_cap` (optional, all
     five): set `btdf` truthy (`1`/`true`/`yes`/`on`) to add a
     transmitted-side lobe about the refracted direction, split from the
     specular transmitted remainder the same way the BRDF splits the
     reflected one. Each `btdf_A`/`btdf_B`/`btdf_g` defaults to the
     row's own reflected `A`/`B`/`g` when left blank, so a symmetric
     scatterer needs only `btdf=1`; an asymmetric one (e.g. a rougher
     transmissive side) overrides the ones that differ. `btdf_tis_cap`
     is the BTDF's own optional TIS ceiling, independent of the
     reflected-side `tis_cap`. Leave all five blank for reflected-only
     scatter (the pre-BTDF row shape still works unchanged).
   - `reference`: **required** citation — cite the actual goniophotometer
     measurement or published ABg fit if you have one. If you are only
     approximating a "representative" surface (as the four shipped rows
     currently do, per Pfisterer's general ABg methodology rather than a
     specific measured curve), say so explicitly and flag the row
     **UNVERIFIED** in `notes` — do not present an engineering guess as a
     measured curve.
   - `notes`: free text; state the verification status plainly.

2. **Validation at load time** (`optprops.load_scatter()`): `A`, `B`, `g`
   numeric and `> 0`; `tis_cap` (if given) in `(0, 1]`; and
   `scatter.abg_tis(A, B, g, cos_i=1)` (widest, normal-incidence total
   integrated scatter) must not exceed 1 — a fit that would scatter more
   power than it receives is a hard `MaterialError` at load time, not a
   silent energy leak discovered mid-trace.

3. **Tag a body:** in the GUI's **Properties** pane (or directly as an
   `App::PropertyString`), set the face's `scatter` property to your
   registry name — whole-body (`'my_ground_aluminum'`) or per-face
   (`'Face2=my_ground_aluminum'`), same generic facemap grammar as
   `coating`/`roughness`/`grating`. **`scatter` is mutually exclusive
   with `roughness`/`diffuser` on the same face** (a `Scene`-build-time
   `ValueError` naming the face) — they are alternative models of one
   surface; pick one.

See `opticalproperties/scatter/bsdf.miebsdf` for the four shipped rows
(`polished_fused_silica`, `polished_bk7_glass`, `diamond_turned_aluminum`,
and `lightly_ground_glass_window` — the BTDF-column worked example) and
`scripts/raytracer/scatter.py`'s module header for the full ABg
energy/sampling derivation.

---

## 10. Adding a biaxial crystal

Biaxial crystals (`n_x != n_y != n_z`: KTP, KTA, LBO, BiBO today) need
**two** registry additions plus a full principal-axis frame on the body
— more moving parts than a uniaxial crystal (§4's `crystal_axis` alone).
See docs/RAYTRACER.md §5.6b/§7.7 for the physics model and honest limits
(internal conical refraction near an optic axis is modeled as a perturbed
two-sheet fan behind `--conical`, off by default; external conical
refraction is not modeled).

1. **Add three principal-index rows to `materials.miemat`** (§5.1 above),
   one per axis, named `<crystal>_nx`/`_ny`/`_nz` (matching the shipped
   `ktp_nx`/`ktp_ny`/`ktp_nz` convention) — `model=sellmeier` is typical
   for a published Sellmeier fit, each with its own `reference`. Cross-
   check the fit reproduces the reference's stated index at at least one
   wavelength (the project's spot-check policy, §5.1) and note the
   result in `notes`.

2. **Append a row to `birefringence/biaxial.mibiax`:**
   - `name`: the crystal name a body's `material` property will use
     (e.g. `ktp`) — this name must **not** collide with an unrelated
     `materials.miemat` row (same shadowing rule as uniaxial crystal
     names).
   - `n_x_material`, `n_y_material`, `n_z_material`: the three
     `materials.miemat` row names from step 1.
   - `reference`: required citation for the crystal identification (can
     repeat the Sellmeier paper if it's the same source).
   - `notes`: free text (e.g., index values at a reference wavelength,
     positive/negative biaxial sign, fit confidence).

3. **Tag a body:** set `material` to the crystal name from step 2, plus
   **both** `crystal_axis` (the X principal axis, body-local `x,y,z`) and
   `crystal_axis2` (the Y principal axis) — the `Scene` loader
   Gram-Schmidt-orthogonalizes Y against X and derives Z = X × Y; a
   missing `crystal_axis2`, or one (near-)parallel to `crystal_axis`, is
   a hard error at scene-build time (the GUI's validation pane also flags
   a biaxial `material` missing `crystal_axis2` before a run, §5.1 of
   docs/RAYTRACER.md). Unlike `crystal_axis` (which defaults to local
   `+x` on every optic), `crystal_axis2` has **no default** — it must be
   authored explicitly.

4. **Validate:** `load_biaxial()` requires all three principal-index
   material rows to exist and requires a non-empty `reference`; a
   crystal name colliding with an unrelated `materials.miemat` row is a
   hard error naming both. Add a scene/test scene exercising the new
   crystal if it will be relied on (the shipped four are pinned by
   `scripts/raytracer/tests/test_biaxial.py`'s closed-form solver tests,
   but there is no dedicated end-to-end FreeCAD scene gate the way
   `doubleslit` has — treat a new crystal's authored scene as manually
   validated, same caveat as the uniaxial Wollaston/waveplate scenes).

See `opticalproperties/birefringence/biaxial.mibiax` for the four shipped
rows and `library.md` §3.2 for the citation/confidence notes (the 5
mineral-placeholder rows there are UNVERIFIED and not yet promoted —
promoting one means clearing that flag against a real source first).

---

## 11. Nonlinear registry rows + pulsed-source primitives (pulsed round)

**χ²/EO/Kerr/saturable rows** live in
`opticalproperties/nonlinear/nonlinear.mienlo` (plain CSV, full-line `#`
comments allowed, `reference` mandatory; `raytracer/optprops.py
load_nonlinear` hard-validates). Five row kinds:

- `chi2_tensor` — 3×6 d_il (pm/V, semicolon-packed row-major) + point
  group. Authoring/derivation only: `nlo.d_eff_tensor` contracts it for
  an arbitrary geometry and `nlo.phase_match_angle` solves type-I ooe
  angles. A tensor row on a body is a hard error.
- `chi2_process` — pre-solved scalar process: crystal, process
  (`shg_type1`/`shg_type2`), `lam_pump_nm` (the exactly-phase-matched
  design pump), `theta_deg`/`phi_deg`, `d_eff_pm_V`. This is what the
  `nonlinear` body property consumes for the tracer's SHG transfer
  (docs/RAYTRACER.md §6.12). Derive new ones from a tensor row + the
  solver, cite both the d-coefficient source and the angle source.
- `pockels` (r coefficients, transverse geometry), `n2` (Kerr, m²/W),
  `saturable` (SESAM-style I_sat/T0/modulation) — consumed by the
  `nonlinear`/`kerr_n2`/`saturable` body properties respectively.

**Pulsed-laser primitives** reuse `_build_laser_collimated`; the entire
personality lives in the catalog entry's `props` dict
(`scripts/primitivelib.py`, "pulsed lasers" block): either
`power` + `rep_rate` (+ `pulse_duration`) — energy/pulse derives — or
`pulse_energy` + `rep_rate` with NO power key (the XOR contract,
docs/RAYTRACER.md §5.2.1). Datasheet provenance goes in the tooltip
(these ship with citations: Mai Tai HP, FemtoFiber pro, Q-smart 850,
SuperK EXR-20). A supercontinuum-style source pairs `spectrum=<row>`
with a digitized SPD table in `emission/tables/*.mietab` (§8b above);
an SPM-broadened source sets `spm='gamma:<W⁻¹km⁻¹>:length:<m>'`.
Regenerate with
`"$MIEWB_FREECAD" -c scripts/make_primitives.py -- --kind <name> < /dev/null`.

---

## 11b. Adding a surface figure-error (Zernike) registry entry

An optic body's `figure_error` property (string) names a row in
`figure/figures.miefig` — a Noll-indexed Zernike coefficient set
describing how a real polished surface's sag DEVIATES from its nominal
(CAD) shape, applied at scene-build time as a
`raytracer.surfaces.PerturbedSurface` sag perturbation over the
transverse pupil. Like `coating`/`roughness`/`grating` it accepts either
a whole-body value or a per-face map (`'FaceN=name;FaceM=name'`) —
though it is not yet one of the six properties the GUI's per-face
"Active Properties" editor exposes (§4 above: `coating`/`roughness`/
`diffuser`/`scatter`/`grating`/`surface_override`), so author a per-face
`figure_error` value directly on the `App::PropertyString`. Because the
CAD body stays the UNPERTURBED nominal shape by design, the extractor's
<1 µm asphere/override verification (§3.1 above) checks base-vs-CAD only
and never sees the perturbation. **This feature is Python-routed** —
`figure_error` is not in the C engine's `PORTED` token set
(`scripts/raytracer/cengine.py`), so any scene using it falls back to
the Python engine under `--engine auto`.

To add a new figure-error entry:

1. **Append a row to `figure/figures.miefig`:**
   - `name`: registry key (e.g., `my_mirror_figure`)
   - `coeffs`: `;`-separated `j:rms_nm` terms (Noll index `j`, SURFACE
     sag RMS in nm — a mirror's WAVEFRONT error is 2× this and falls out
     of the tracer's OPL naturally, no separate accounting needed). Noll
     `j >= 2` — `j=1` (piston) is a meaningless constant offset and is
     rejected; a duplicate `j` within one row is rejected.
   - `r_norm_mm`: pupil radius (mm) the coefficients are referenced to,
     must be `> 0`
   - `reference`: required citation (interferometer report, a spec sheet
     stating a fringe/RMS figure spec, etc.)
   - `notes`: optional

2. **Tag a body:** set the optic's `figure_error` property to your
   registry name (whole-body or per-face, per the note above).

See `opticalproperties/figure/figures.miefig` for the four shipped rows
(`fig_lambda4_defocus_633`, `fig_astig_633`, `fig_lambda10_typical`,
`fig_trefoil_633`) and docs/RAYTRACER.md §5.8b for the physics model.

---

## 12. Adding a scattering-sample registry entry (`sample`, S(q))

`opticalproperties/sample/samples.miesamp` (loader: `optprops.
load_samples()`, docs/RAYTRACER.md §5.13/§7.16) is the scattering-sample
registry consumed by an optic body's `sample` property (§4 above): a
particle population bound to that body's interior (the body's own
`material` is the host medium). 16 columns: `name, particle_material,
dist, median_um, gsd, phi, tau, mode, count, sq_model, sq_params, shape,
aspect_ratio, solvent_visc_pas, reference, notes`.

1. **Append a row to `sample/samples.miesamp`:**
   - `name`: registry key
   - `particle_material`: must exist in `materials.miemat` (§5 above)
   - `dist`: `mono` or a log-normal-over-radius kind; `median_um` (median
     DIAMETER, matching `--particles`' own convention, §9 of
     docs/RAYTRACER.md); `gsd` (optional, defaults 1.6, forced to 1.0 for
     `dist=mono`)
   - **exactly one** of `phi` (mass fraction, `(0,1)`) or `tau` (target
     Beer-Lambert optical depth along the host body's own AABB x-extent)
   - `mode` (optional, default `auto`): `auto`/`continuum`/`explicit`;
     `count` (optional int `>0`) pins an explicit-mode site count directly
   - `sq_model` (optional, default `none`): `none`/`py`/`baxter`/
     `fractal`/`paracrystal`/`table`, with `sq_params` (`;`-separated
     `key:val`) validated per model — see docs/RAYTRACER.md §7.16 for the
     exact required/optional keys per model (each model's physics is in
     §5.13). `mode=explicit` + `sq_model=paracrystal` places a REAL
     fcc/bcc/sc lattice realization instead of an independent-sphere dart
     throw.
   - `shape` (optional, default `sphere`): `sphere` or `spheroid` (with
     `aspect_ratio`, required `!= 1.0` for `spheroid`, forced `1.0` for
     `sphere`) — `spheroid` routes through the T-matrix evaluator
     (§5.13; needs `pytmatrix` in the optics env, INSTALL.md §3.4).
   - `solvent_visc_pas` (optional): required if the row will be used with
     `run_dls.py` (§8.7 of docs/RAYTRACER.md) — the host solvent's
     dynamic viscosity for the Stokes-Einstein diffusion coefficient.
   - `reference`: required citation for the particle/size-distribution/
     S(q) data; `notes`: optional.

2. **Create an S(q) table** if `sq_model=table`: `sample/tables/
   <name>.mietab` with columns `q_per_um, s` (`q` strictly increasing
   `>= 0`, `s > 0`, `>= 2` rows) — the abscissa is momentum transfer
   (1/µm), not wavelength, so this is a distinct table shape from every
   other `*.mietab` in the repo.

3. **Tag a body:** set an optic body's `sample` property to your
   registry name — the body's shape bounds the cloud and its `material`
   is the host medium, so pick (or build) a host body whose interior
   volume is what you want the sample to fill (§13 below for ready-made
   nested-cell primitives).

4. **Validate:** `load_samples()` hard-validates every field per the
   schema above; a sample used with `run_dls.py` additionally requires
   EXPLICIT mode (`mode=explicit` or `count` small enough that `auto`
   resolves to it) and at least one coherent source in the scene.

See `opticalproperties/sample/samples.miesamp` for the 7 shipped rows and
docs/RAYTRACER.md §5.13/§7.16 for the full physics model and honest
limits (decoupling approximation for polydisperse S(q); `tau` resolves
along the host body's AABB x-extent, exact only for an on-axis
rectangular cell).

## 13. Building a cuvette-style nested-solid sample-cell primitive

The samples-instruments round's sample-cell primitives
(`cuvette_square`/`cuvette_capillary`/`flow_cell`/`vial_cylindrical`/
`vat_cylindrical`) all follow the SAME pattern as `bs_cube`/`pbs_cube`
(§3 above): a full WALL solid (glass) with a full LIQUID solid nested
strictly inside it, glass-to-liquid contact, no air gap — the extractor
classifies the pair `validation.nested_solids` and the tracer's LIFO
medium stack recovers the wall as the shell outside the liquid volume
(docs/RAYTRACER.md's "Optically-contacted solids" quirk note). The wall
body is always the PRIMARY body (carries the element label + train
props, named `group` by convention); the liquid body is a second body
named `group + "_liquid"`.

**Rectangular cells** (`_build_cuvette_box` in `primitivelib.py`, shared
by `cuvette_square`/`cuvette_capillary`; `_build_flow_cell` is the same
idea with a smaller flowing-liquid channel instead of a full-cross-
section liquid fill): the outer wall box spans `(path_length + 2*wall) x
(width + 2*wall) x (height + 2*wall)`; the liquid box (`path_length x
width x height`) sits centered inside it, inset by `wall` on every face:

```python
def _build_cuvette_box(doc, group, p, wall_material="glass",
                       liquid_material="water"):
    pl_, w, h, wall = p["path_length"], p["width"], p["height"], p["wall"]
    outer_x, outer_y, outer_z = pl_ + 2*wall, w + 2*wall, h + 2*wall
    wall_body = mts.new_body_pad(
        doc, group, group,
        rects=[(-outer_y/2, -outer_z/2, outer_y, outer_z)],
        x_start=0.0, length=outer_x, props={"material": wall_material})
    liquid = mts.new_body_pad(
        doc, group + "_liquid", group + "_liquid",
        rects=[(-w/2, -h/2, w, h)],
        x_start=wall, length=pl_, props={"material": liquid_material})
    return [wall_body, liquid]
```

**Cylindrical cells** (`_build_cyl_nested`, shared by `vial_cylindrical`/
`vat_cylindrical`): a vertical (local z) glass cylinder, radius =
`diameter/2`, x-centered at its own radius so the near tangent point
sits at local x=0 (the `lens_rod` convention, §3's primitive-anatomy
note) and the far tangent at x=`diameter`; the liquid is a smaller
concentric cylinder inset by `wall` radially and on both z ends.

**A bare-cloud host with no cell walls at all**: `sample_region` is a
single `material=air` box (no nesting, no second body) — the anchor for
a `sample` cloud that shouldn't be walled at all, and it carries a
`port_frames` pass-through entry so a downstream chained element (e.g. a
goniometer detector) can reference it like any other element (closes the
`future.md` a2 backlog item on chain-referenceable particle clouds).

**To add a new nested-pair primitive**: write a builder following either
pattern above (probe-verify the resulting volumes and confirm zero
overlaps the way the samples-instruments wave did — see its test suite
for the exact probe-point recipe), add a `port_frames` entry (§3b above)
so the primitive is chain-able, and register both builder + `PRIMITIVES`
metadata (§3). Set the wall body's `material` to a real glass/polymer row
and the liquid body's `material` to the solvent (or a `sample`-cloud
host, §12) — never both walls AND a bare-liquid-only design for the same
cell; pick one topology.

## 14. Adding an extended image-source registry entry

`opticalproperties/image/images.mieimg` (loader: `optprops.
load_images()`, docs/RAYTRACER.md §5.14/§7.17) names a greyscale bitmap
consumed by a source body's `image` property (§4 above). Columns `name,
file, reference, notes`.

1. **Add the bitmap file** next to the registry CSV
   (`opticalproperties/image/`), one of `.png`/`.jpg`/`.jpeg`/`.tif`/
   `.tiff`/`.bmp`/`.npy` (`IMAGE_EXTENSIONS`) — keep it inside
   `opticalproperties/` so a `.MieWB` project library carries it with the
   scene. `scripts/tools/gen_usaf_target.py` is a worked example generator
   (a MIL-STD-150A-style-alike resolution target) if you need a
   synthetic test pattern rather than a real captured image; row 0 of the
   bitmap is the picture's TOP (max-value convention — see the loader's
   own orientation note, §5.14).
2. **Append a row to `image/images.mieimg`**: `name`, `file` (the
   filename from step 1), `reference` (required — cite the real image's
   provenance, or the generator script + its own convention notes for a
   synthetic target), `notes` (optional).
3. **Tag a body:** set a source body's `image` property to your registry
   name; optionally `image_cone_deg` (0,90] to restrict emission to a
   cone instead of full Lambertian (§5.14 — this is a variance
   optimization, not a physical claim about the object's emission
   pattern). `image` is mutually exclusive with `beam`/`apodization` and
   needs a planar emitting face.
4. **Validate:** `load_images()` only checks file existence/extension and
   the `reference` citation contract; the actual pixel data loads once at
   scene build (`sources.load_image_gray`), so a corrupt/unreadable
   bitmap fails at trace time, not at registry-load time.

See `opticalproperties/image/images.mieimg` for the shipped
`usaf_style_target` row and the `source_image` primitive.
