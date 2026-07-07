# UX friction log — building the demo gallery through the interface

Every demo in this folder was assembled with the exact op sequence the GUI
issues (`scripts/make_demos.py` drives `mieworkbench.core.fcclient`:
import_primitive → set_spreadsheet → rebuild_primitive → set_placement →
set_property → save). This file records what was easy, what hurt, and what
should change — the raw material for the round-2 UX proposal.

## What worked well

- **Type-first add + parameter sheets.** Every element except the Schmidt
  corrector came straight from the catalog; setting prescription radii/
  thicknesses through the `dim` sheet aliases was exactly the "element
  parameters box" experience — no typing beyond numbers.
- **`solve_achromat` / lens wizards.** The microscope objective's two
  doublets were one function call each (f=20/f=40, scaled BK7/SF5 design).
- **Multi-body elements move rigidly by group.** Placing the iris, fiber,
  achromats and BS cube (all 2-body elements) needed one set_placement.
- **`max_reflections` as a per-demo simparam.** The fiber's ~60-bounce TIR
  guiding needed one line in simparams.json once the pipeline exposed it.
- **The aperture contract.** iris/slit plugs (material=air) came built-in;
  no manual plug bodies anywhere.

## Friction points (ranked by how much time they cost)

1. **No "aim this element at that one".** Every folded system needed hand
   trig for rotation quaternions: the Newtonian/Dobsonian diagonal
   (-135° about z), the Czerny-Turner's four aimed bodies (mirror-law
   bisector normals computed offline), the Michelson's M2/detector, the
   prism-spectrometer camera arm. This was BY FAR the largest cost of
   building the gallery. A transform-panel "point local -x at element E /
   at world point P" action (plus "reflect A onto B" for mirrors) would
   have removed ~80 lines of layout math.
2. **No system-level focus readout.** The camera triplet and microscope
   needed an offline paraxial solver to place the sensor (the engine knows
   the answer, but only after a full run). An on-demand "paraxial trace
   through the current train" readout (EFL/BFL/image plane) would remove
   the round-trip. The thick-lens wizard solves single elements only.
3. **No "flip element" affordance.** Orienting a PCX lens convex-out, the
   SCT secondary convex-toward-primary, etc. means knowing the primitive's
   local axis convention and applying a 180° rotation. A one-click "flip
   about local y/z" (or an orientation indicator — now added in this
   round: the blue +x dot helps diagnose, but not fix) would be quicker.
4. **Min-deviation prism setup is trig homework.** `prism.rotation` is the
   right knob, but the value (30° − (A+Dmin)/2) had to be derived offline
   from the glass index. A tiny wizard ("orient for minimum deviation at
   λ") fits the existing wizard registry pattern.
5. **Tiny angles are invisible.** The Michelson's 0.158 mrad tilt (5
   fringes across the detector) can't be verified visually in the 3D view;
   trust-the-number only. The transform panel showing the rotation as
   axis+angle (not just a quaternion) with 4+ decimals would help.
6. **D-shaped mirror default clips an on-axis cone.** `mirror_d_shaped`
   with cut_offset=0 is a half-disc — correct for beam packing, wrong as a
   Newtonian diagonal for an on-axis bundle (half the cone misses). The
   demos use a round flat instead; a `cut_offset` preset note (or an
   elliptical-diagonal primitive) would prevent the trap.
7. **Achromat aperture vs scaled radii.** `solve_achromat(20)` yields
   |R_iface| = 8.8 mm; leaving the default aperture (18 mm) makes the
   builder fail with a bare "math domain error". The wizard knows both
   numbers — it should clamp/warn instead of letting the build die.
8. **Diverging broadband source is a property recipe, not a type.** A
   "slit lamp" (divergent + lambdamin/lambdamax) is laser_divergent plus
   two hand-added properties. Worth a catalog preset.
9. **A reflection grating is three separate switches.** The catalog
   `grating_plate` is a bk7 TRANSMISSION grating; making it reflective
   for a Czerny-Turner needed `mirror=1.0` (the engine's reflect/transmit
   switch — the aluminum material alone does nothing), an explicit
   `0,1,0` periodicity vector (`v` = the face-frame t2 tangent = out of
   the layout plane here), and a face pin (see the FaceN instability
   above). A `grating_mirror` catalog primitive with in-plane defaults
   would make this one click.

## Real bugs the shakedown caught (fixed immediately)

- **Multi-body elements tore apart on placement.** `set_placement` looked
  a body up by label BEFORE trying the element-group match, and an
  imported multi-body element's primary body carries the element label
  itself — so moving "Slit"/"Stop"/"Fiber"/"BS" moved only the first body
  and left the plug/cladding/second prism at the origin (extraction then
  failed on overlapping solids, or the Michelson silently traced garbage).
  Group match now wins; the GUI inherits the fix.
- **Rebuild-on-edit kept stale `surface_override` strings.** Editing a
  `mirror_parabolic`'s focal length rebuilt the geometry but re-applied
  the OLD override via the extra-prop preservation path, so extraction
  died on the <1 µm asphere verification. `surface_override` is now in
  `derived_props` for the asphere-backed primitives (same mechanism as
  the iris `blackness` → `absorbance` fix).
- **The prism's `rotation` param silently vanished on rebuild.** The
  builder baked it into the body *Placement*, which rebuild_element
  rightly preserves from before the rebuild — so setting rotation via the
  sheet did nothing (the spectrometer beam came out 12° instead of 54°).
  The builder now bakes rotation into the sketch vertices. Rule of thumb:
  a sheet param must never live in the Placement.
- **Rebuilding renumbers faces, orphaning face-mapped properties.** A
  `grating_plate` resized through the sheet gets fresh FaceN indices, and
  the preserved `Face1=600:v` grating property can silently land on an
  edge face (0 diffracted power). The Czerny-Turner demo imports the
  shipped geometry un-rebuilt; the real fix (round 2) is re-resolving
  face-map indices geometrically after every rebuild.

## Detector-face gotcha (engine contract, worth a round-2 look)

The emit/detector face auto-pick is "face centroid closest to the world
origin". On a rotated, off-axis detector (a folded telescope's eyepiece)
that is a thin EDGE face — the run completes and detects 0 mW with no
warning. The folded demos pin `detector_face` in simparams, resolved at
build time from tessellation normals (`Demo.detector_face()`) because
FaceN numbering is not even stable across rebuilds. A `detector_face`
BODY PROPERTY (part of the authoring contract, set once in the GUI)
would remove the whole failure class.

## Round-2 proposal (prioritized)

1. **"Aim at" transform action** — point an element's local −x (or a
   chosen axis) at another element / a world point; "reflect A onto B"
   variant that sets a mirror's normal to the bisector. Removes the hand
   trig that dominated gallery construction. (transform_panel + a pure
   solver in core/transforms.py)
2. **`detector_face`/`emit_face` as body properties** — extractor-level
   support so the recording face is authored once in the GUI (with the
   indicator glyph showing it) instead of pinned per-run; kills the
   0-mW-edge-face failure class AND the FaceN instability, if stored as
   a geometric rule (e.g. `facing:+y` or `local:-x`) rather than an index.
3. **Paraxial system readout** — EFL/BFL/image-plane of the current train
   along an axis, computed from the extracted surfaces (a pure-python
   paraxial chain; make_demos.py already contains the 25-line prototype).
4. **Face-map re-resolution after rebuild** — match old→new faces
   geometrically (centroid/normal/area) and rewrite FaceN indices in
   coating/grating/override strings; warn when unmatched.
5. **"Flip element" button** (180° about local y/z) in the transform
   panel, paired with the orientation indicators.
6. **Prism min-deviation wizard** (given glass + λ, set `rotation`) and a
   **slit-lamp source preset** (divergent + λ band) in the catalog.
7. **Transform panel shows axis+angle** with enough digits for mrad-scale
   alignment (Michelson fringe tilts are invisible otherwise).

## Fixed during this round (feedback already applied)

- `detector_plane` grew a `height` param (36×24 CMOS sensor demo).
- `--max-reflections` is now a first-class pipeline/simparams option.
- `fiber_optic` and `mirror_annular` became catalog primitives.
- Face-orientation indicators now show emit/detector/+x faces in both 3D
  views (the "which way is this facing" questions that motivated several
  of the flips above are at least *visible* now).
