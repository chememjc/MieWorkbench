# PRESCRIPTION.md — the prescription-primary data model (engine3.md §3, P5)

**The optical prescription is the truth; the CAD is a view of it.**

Historically MieWorkbench inverted this: the CAD (a tagged `.FCStd`) was
primary, and `surface_override` was an escape hatch used only for aspheres —
the extractor *derived* optical surfaces from FreeCAD geometry
(`extract_geometry.canonicalize_revolution`, native OCC recognition), verifying
an override against the mesh to 1 µm. P5 formalizes what was already half-true:
the primitive builders (`scripts/primitivelib.py`) author elements in **basic
terms** (the `dim` spreadsheet) and already know the exact intended optical
surfaces. Those surfaces become the recorded truth; the CAD is verified against
them and the `model.json` optical surface is emitted **from the prescription**.

This document is the schema + contract. See `docs/RAYTRACER.md` §5 for the
body-tagging contract, and `engine3.md` §3 for the design rationale.

---

## 1. What a prescription is

A `prescription.json` records, per optical **element**, the basic terms it was
authored in and the exact analytic optical surfaces those terms imply — in the
**same surface language** `model.json` already carries (`common._SURFACE_REQ`;
there is no second surface language).

```json
{
  "schema_version": 1,
  "elements": {
    "Lens1": {
      "kind": "lens_pcx",
      "params": { "R_front": 0.025, "ct": 0.005, "aperture": 0.020 },
      "surfaces": [
        { "role": "front", "material": "bk7", "type": "sphere",
          "center": [0.025, 0.0, 0.0], "radius": 0.025 },
        { "role": "edge",  "material": "bk7", "type": "cylinder",
          "origin": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
          "radius": 0.010 }
      ]
    }
  }
}
```

- **key** — the element's `miewb_group` (stable across rebuilds and multi-body
  elements; the same key `set_placement` resolves by). Falls back to the body
  Label.
- **kind** — the `primitivelib` primitive kind (or `"custom"`).
- **params** — the basic terms, **SI** (metres / radians). Informational +
  displayed; the surfaces are the operative content.
- **surfaces** — a list of model.json surface dicts (`type` + the geometry keys
  required by `common._SURFACE_REQ` for that type), each augmented with:
  - **role** — a descriptive label (`front` / `back` / `edge` / `sphere` /
    `barrel` / …). Not used for matching — matching is geometric.
  - **material** *(optional)* — the glass/medium binding.

### Coordinate frame

Surface geometry is **body-local, SI metres** — the frame the builder builds in
(optical axis local **+x**, front vertex at the origin; the primitive
convention). The extractor transforms local → global through the body's FreeCAD
`Placement` (the exact transform OCC applies to the geometry), so the stored
prescription is **placement-independent** and survives the element being moved,
chained, or folded in the optical train.

---

## 2. Storage & precedence

| container | how the prescription travels |
|---|---|
| `.MieWB` | a `prescription.json` member (`miewb_tool` pack/unpack round-trips it; `sniff` unaffected; absent for prescription-free scenes = full backward compat) |
| bare `.FCStd` | a sidecar `<stem>.prescription.json` next to the model |
| `.MieSim` | carried inside the embedded `input.MieWB`; the packed `geometry/*/model.json` already holds the prescription-emitted surfaces |

**Extractor precedence** (`extract_geometry.load_prescription_for`): an explicit
`--prescription PATH` (or a dict passed to `extract_document(prescription=…)`)
wins; else the `<stem>.prescription.json` sidecar; else none (every existing
scene extracts exactly as before).

`miewb_tool run` copies the unpacked `prescription.json` beside the named model
as the sidecar, so the headless pipeline picks it up with no `run_pipeline`
flag.

---

## 3. The extractor cross-check (engine3 §3)

When a body carries a matching prescription entry, `extract_geometry`:

1. **Matches** each prescription surface to a FreeCAD face — gated by the
   face's native OCC type (`sphere` ↔ `Sphere`, `cylinder` ↔ `Cylinder`,
   `asphere` ↔ `SurfaceOfRevolution`; `SurfaceOfRevolution` is also accepted
   for sphere/cylinder to cover OCC failing to natively recognize a
   spline-approximated cap). A single element never carries both an asphere and
   a sphere/cylinder surface, so there is no ambiguity.
2. **Verifies** the tessellated face against the prescription surface to the
   **1 µm** gate (`PRESCRIPTION_TOL_M`, the same gate the asphere-override
   verifier uses). A residual > 1 µm — or a prescription surface that matches no
   face at all (topology drift) — is a **hard error** (`ExtractError`) naming
   the body, surface role, and residual. The CAD is never silently preferred
   over its prescription, nor vice versa.
3. **Emits** the `model.json` surface **from the prescription**:

   | prescription type | emit policy |
   |---|---|
   | `sphere`, `asphere`, `qforbes` | **emitted from the prescription** (exact params, transformed to global). Asphere/qforbes reuse the existing `build_asphere_surface` / `build_qforbes_surface` machinery: **R/k/coeffs** come from the prescription; **vertex/axis** are recovered from the placed geometry (exact). |
   | `cylinder`, `plane`, `cone` | **verified only**; the native OCC surface is kept (already exact; a ruled/flat surface's origin-along-axis is a free parameter, so re-emitting buys nothing). |

Bodies **without** a prescription extract exactly as today.

**Figure error is orthogonal to this gate (P8, engine3 §11).** A `figure_error`
body property (§5.8b of `RAYTRACER.md`) adds a Zernike SAG perturbation to the
analytic surface *at scene build*, wrapping it in a `surfaces.PerturbedSurface`.
By design the CAD (and any prescription) is the **UNPERTURBED** shape, so this
1 µm cross-check verifies the base surface against the CAD **only** and never
sees the perturbation — the figure map is a trace-time overlay, not a geometry
edit, and carries no verification of its own.

---

## 4. Emission from the builders (the single authoring path)

`primitivelib.build_prescription_entry(kind, params_mm)` is **pure** (no
FreeCAD) and computes the entry from the *same* dim params the geometry builders
consume — so editing a dim param regenerates the geometry **and** its
prescription through one function. It is importable from every interpreter
stack (GUI venv, optics env, system python3, FreeCAD).

Covered kinds this round — the clean analytic lens/mirror family:

| kind | prescription surfaces |
|---|---|
| `lens_pcx` / `lens_pcv` | front **or** back sphere + edge cylinder |
| `lens_dcx` / `lens_dcv` / `lens_meniscus` | front + back spheres + edge cylinder |
| `lens_ball` | one sphere |
| `lens_rod` | barrel cylinder (axis +z) |
| `lens_cyl` | front cylinder (axis +z) |
| `lens_asphere` | front asphere + edge cylinder |
| `mirror_parabolic` | front asphere (R = 2·rfl, k = −1) + edge cylinder |

Flat backs stay with the extractor's native-OCC plane classification (already
exact, no canonicalization risk) and are not carried in the prescription.

Generation is wired into the GUI (`Project.build_prescription()`, packed into
the `.MieWB` by every `mainwindow` save/export path) and available headless via
`miewb_tool pack --prescription`. The GUI element editor shows a read-only
**Prescription** group for covered elements; editing stays via the dim
parameters (which regenerate the prescription — the single-authoring-path
invariant).

---

## 5. Loader API (`scripts/raytracer/prescription.py`, pure stdlib)

```python
from raytracer import prescription as pr
doc  = pr.new_document({key: entry, ...})   # build + validate
pr.validate(doc)                            # raises PrescriptionError
pr.save(path, doc); doc = pr.load(path)     # atomic, stable key order
text = pr.dumps(doc); doc = pr.loads(text)  # byte-stable round-trip
entry = pr.element_for(doc, key)            # lookup, never raises
side  = pr.sidecar_path("scene.FCStd")      # scene.prescription.json
```

Surface geometry keys are validated with `common._check_surface_params`, so the
prescription and `model.json` surface schemas can never drift apart. `type`
`"mesh"` is rejected — a prescription surface is always analytic.

---

## 6. Findings on clean primitives (verified 2026-07-16)

For the covered lens family the FreeCAD builders already produce **native exact
OCC** surfaces — a plano-convex front revolves to a true `Sphere` (center =
vertex + signed R, radius = |R|), the rim to a `Cylinder`, the flat back to a
`Plane`; only `lens_asphere` and `mirror_parabolic` produce a
`SurfaceOfRevolution` (they already used `surface_override`). Consequences,
measured end-to-end (build → extract → compare):

- **Spherical caps**: the prescription-emitted sphere agrees with the
  native-OCC extraction to **≈1×10⁻¹⁶ m** at zero placement, and to
  **≈5×10⁻¹⁶ m** under a real translate + 13–17° rotation — i.e. the local→global
  transform matches OCC's own to machine epsilon, far inside the 1×10⁻¹² gate.
- **Aspheres**: the prescription supplies the identical `R/k/coeffs`
  the builder's `surface_override` string did (`lens_asphere` R = 20.6033 mm,
  k = −1, A4 = 6.586562×10⁻⁶ mm⁻³; `mirror_parabolic` R = 100 mm, k = −1) — the
  prescription now *is* the truth the override string encoded.
- **Physics**: a C-engine `quick` trace of the pcx lens scene reports
  **bit-identical** detected power with vs without the prescription
  (0.004597940105661133 W), confirming the emit is physics-neutral for clean
  primitives.

No param/canonicalization disagreement was found for the covered kinds — the
clean-primitive prescription and the FreeCAD canonicalization agree to
floating-point precision, exactly as engine3 §3 anticipated. The strict value
of P5 here is (a) the prescription becomes the recorded, placement-independent
truth and (b) the 1 µm cross-check now hard-errors on any CAD drift from it.
The larger structural win is the asphere family, where FreeCAD has no native
analytic form and the prescription is the only phase-valid representation.
