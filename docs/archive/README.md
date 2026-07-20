# docs/archive — historical design ledgers

These are the point-in-time design/planning documents that drove past
rounds. They are kept for provenance — tracked docs cite them by section
number (e.g. "engine3 §7.4") — but they are NOT maintained: the shipped
behavior they specified is authoritatively described in
`docs/RAYTRACER.md` (engine) and `README.md` (workbench), and later
rounds may have superseded details in place.

- `engine.md` — BREP-native C++/OptiX engine: design, plan, and verdict
  (pre-cengine exploration).
- `engine2.md` — engine redesign for physical realism and speed
  (prescription-primary data model; corrections to engine.md).
- `engine3.md` — the engine overhaul: physics first, speed second — the
  design ledger behind the P0–P9 phases (Lekner uniaxial, RCWA tables,
  virtual instruments, figure error, Berreman 4×4, sequential preview
  unification).
- `UI_COORDINATE_PROPOSAL.md` — coordinate-system UI proposal that
  seeded the transform-panel work.
