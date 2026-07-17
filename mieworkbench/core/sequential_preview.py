"""sequential_preview.py - P4b preview unification (engine3.md Sec.5 /
Sec.15 P4b: "preview unified"): an ADDITIVE fast path for the GUI's live
ray preview.

When the current scene is sequential-expressible (the same scope
scripts/raytracer/optiland_bridge.py already enforces via
BridgeUnsupported -- rotationally-symmetric on-axis refractive/reflective
trains, no tilts/decenters/gratings/birefringence) AND its source emits a
collimated, on-axis fan, trace the preview fan directly through the
Optiland bridge -- IN-PROCESS, under the GUI's own env/bin/python
interpreter (which already carries Optiland; see optiland_bridge.py's
module docstring: "Interpreter: env/bin/python (the project GUI venv...)")
-- instead of shelling out to scripts/preview_rays.py under the optics env.
Same fan definition (raytracer.sources.sample_viz_pattern, the exact
function preview_rays.py uses), same rays.vtp output contract
(raytracer.vtkexport.write_vtp_polylines) -- the GUI's ray-overlay
consumer (widgets/vtkview.py et al.) never has to know which engine
produced the file.

Process boundary: deliberately IN-PROCESS, not a QProcess. Optiland is a
pure-numpy geometric tracer (no CUDA/native extension in this call path),
the fan is a handful of rays (n=O(10)), and P4a's parity oracle
(mieworkbench/tests/test_optiland_oracle.py) already pins Optiland's
geometry against the C-engine MC trace to floating-point round-off on
exactly this scene class -- so the crash-isolation rationale that justifies
raypreview.py's subprocess chain (an isolated FreeCAD/optics-env process
per stage) does not apply here to the same degree, and paying a Python
subprocess-spawn + full optics-env import (~1-3 s, per raypreview.py's
docstring) for a sub-millisecond trace would defeat the point. The
trade-off is made SAFE, not ignored: build() below is the ONLY entry point,
and it catches every exception (not just BridgeUnsupported) so a bug in
this path degrades to "fall back to the general chain", never a GUI crash.

This is a FAST PATH, not a replacement: build() returns (False, reason) for
any scene/pattern outside the bridge's documented scope (tilts, folds,
mirrors needing double-pass ordering, divergent/non-collimated sources,
crystals/birefringence, gratings, multiple sources/detectors, missing
geometry, an Optiland import failure, ...) or on ANY unexpected error.
core/raypreview.py's caller falls back to the existing
save_copy -> extract -> preview_rays.py subprocess chain unchanged.

Known simplification (documented, not a bug): the sequential trace follows
only the PRIMARY transmitted/reflected ray through each surface -- no
Fresnel-ghost or scatter branches (that is what "sequential" ray tracing
means, engine3.md Sec.5's own table: sequential = "ordered surface list...
exact preview"; non-sequential = "stray light, ghosts... the default").
Every ray segment therefore carries rel_power=1.0; a scene where the ghost
overlay itself is the point of previewing is exactly the kind of scene this
bridge already declines (or the user reads as expected from the "sequential
(exact)" status hint, which names the engine that produced the overlay).
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import common     # noqa: E402  (stdlib-only; parse_viz_pattern_spec)

# Status-hint labels (core/raypreview.py forwards one of these verbatim on
# its `finished` signal; mainwindow.py shows it in the status bar).
ENGINE_SEQUENTIAL = "sequential (exact)"
ENGINE_FALLBACK = "engine fan"

# axial (world +x) alignment tolerance for "the source emits straight down
# the optical axis" -- anything else (a tilted/folded train) is outside the
# bridge's on-axis scope. Matches optiland_bridge.py's own plane/asphere
# axis tolerances (1e-6).
_AXIS_TOL = 1e-6
# collimation tolerance: every ray in the pattern must share the source's
# emit direction (unit vectors) for the object-at-infinity assumption
# trace_pupil_world_path relies on to hold. A Plane emit face gives exactly
# IDENTICAL directions (raytracer/sources.py: `dirs = np.tile(...)`); a
# Sphere (divergent) cap gives materially different per-point normals, so
# this cleanly separates the two without special-casing the source class.
_COLLIMATION_TOL = 1e-9


# Cache the optical-properties registry load (168 materials + coatings +
# filters + gratings + ... -- ~130 ms of CSV parsing per raytracer.optprops.
# load_optical_properties call, measured; profiled as the dominant cost of
# this fast path once Optiland/numpy imports are warm) by resolved root
# path. Reused across every debounced preview cycle in a session -- the
# property library changes far less often than the geometry does, and a
# live library edit still lands on the NEXT preview cycle (the debounce
# already coalesces edits; this cache is not a correctness hazard, only a
# same-session speed one, and the general (non-sequential) preview chain
# never benefits from or is affected by it -- it always reloads fresh in
# its own subprocess). Keyed on the resolved root string; a caller that
# wants to force a reload (e.g. after promoting a library entry) can call
# invalidate_props_cache().
_PROPS_CACHE = {}


def invalidate_props_cache(root=None):
    """Drop the cached optical-properties registry for `root` (or every
    cached root when None) -- call after editing the library so the next
    preview cycle picks up the change immediately instead of waiting for
    process restart."""
    if root is None:
        _PROPS_CACHE.clear()
    else:
        _PROPS_CACHE.pop(str(root), None)


def _cached_optprops(root):
    from raytracer.optprops import load_optical_properties
    key = str(root) if root is not None else ""
    props = _PROPS_CACHE.get(key)
    if props is None:
        props = load_optical_properties(root=root)
        _PROPS_CACHE[key] = props
    return props


def _filter_only_bodies(model, only_bodies):
    """Mirrors scripts/preview_rays.py's own _filter_only_bodies exactly
    (an independent copy rather than importing a private helper from a
    script module, so this fast path has no coupling to preview_rays.py's
    internals -- it only shares the output CONTRACT with it, per the task's
    "do not touch preview_rays.py" boundary)."""
    if not only_bodies:
        return
    keep = {b.strip() for b in only_bodies if b.strip()}
    kept = []
    for b in model["bodies"]:
        if (b["role"] in ("source", "detector")
                or b["name"] in keep or b["label"] in keep):
            kept.append(b)
    model["bodies"] = kept


def build(geometry_dir, out_path, pattern="fan:n=5", only_bodies=None,
         optical_properties=None):
    """Attempt the sequential fast path.

    Returns (True, ENGINE_SEQUENTIAL) and writes rays.vtp to out_path on
    success. Returns (False, reason) -- a short human-readable string, for
    the console/log -- if the scene/pattern is out of scope or on ANY
    failure; NEVER raises. The caller (core/raypreview.py) treats a False
    return exactly like "the bridge doesn't apply here" and runs its
    existing subprocess chain.
    """
    try:
        return _build(geometry_dir, out_path, pattern, only_bodies,
                     optical_properties)
    except Exception as exc:      # deliberately broad: see module docstring
        return False, "%s: %s" % (type(exc).__name__, exc)


def _build(geometry_dir, out_path, pattern_spec, only_bodies,
          optical_properties):
    # Imported lazily (inside the try/except in build()): an Optiland
    # import failure -- or absence entirely on some future minimal install
    # -- is just another "fall back" reason, not a hard dependency of the
    # preview chain.
    from raytracer import optiland_bridge as ob
    from raytracer.scene import Scene
    from raytracer.sources import sample_viz_pattern
    from raytracer.vtkexport import write_vtp_polylines

    geometry_dir = Path(geometry_dir)
    model_json = geometry_dir / "model.json"
    if not model_json.exists():
        raise ob.BridgeUnsupported("no model.json in %s" % geometry_dir)
    with open(model_json) as fh:
        model = json.load(fh)
    _filter_only_bodies(model, list(only_bodies) if only_bodies else None)

    if not any(b["role"] == "source" for b in model["bodies"]):
        raise ob.BridgeUnsupported("scene has no source bodies")

    pattern = common.parse_viz_pattern_spec(pattern_spec)
    props = _cached_optprops(optical_properties)
    scene = Scene(model, props.matdb, props.coatings, optprops=props,
                  geometry_dir=geometry_dir)
    if len(scene.sources) != 1:
        raise ob.BridgeUnsupported(
            "sequential preview needs exactly one source, got %d"
            % len(scene.sources))

    sid = 0
    bidx, src = scene.sources[0]
    # n_lambda=3: matches preview_rays.py's own dispersion-preview strata
    # (red/green/blue fan rays through dispersive glass); a monochromatic
    # source still collapses to one stratum (wavelength_strata's contract).
    batch = sample_viz_pattern(scene, scene.bodies[bidx], src, sid, pattern, 3)
    if batch is None or len(batch) == 0:
        raise ob.BridgeUnsupported("pattern produced no rays for the source")

    dirs = batch.dir
    d0 = dirs[0]
    if abs(abs(float(d0[0])) - 1.0) > _AXIS_TOL:
        raise ob.BridgeUnsupported(
            "source does not emit along the world +/-x optical axis "
            "(tilted/folded train) -- outside the sequential bridge's "
            "on-axis scope")
    if d0[0] < 0:
        raise ob.BridgeUnsupported(
            "source emits toward -x; the bridge's surface ordering "
            "assumes propagation toward +x")
    if not np.allclose(dirs, d0, atol=_COLLIMATION_TOL):
        raise ob.BridgeUnsupported(
            "source emit face is divergent (e.g. a spherical cap) -- "
            "outside the sequential bridge's collimated "
            "object-at-infinity scope")

    rows = []
    for lam_stratum in np.unique(batch.lam_stratum):
        m = batch.lam_stratum == lam_stratum
        lam_m = float(batch.lam[m][0])
        lam_nm = lam_m * 1e9
        # model_stop=True: model a real iris/aperture-stop annulus as a
        # physical clip (RadialAperture), exactly like the run the preview
        # is standing in for -- a fan ray outside a modelled iris's bore
        # is dropped (valid[i] False below) precisely as it would vignette
        # in the actual trace.
        system = ob.load_sequential_system(geometry_dir, wavelength_nm=lam_nm,
                                           model_stop=True)
        opt = ob.build_optic(system, optprops_dir=optical_properties)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            epd_mm = float(opt.paraxial.EPD())
        pts = batch.pos[m]
        path, valid = ob.trace_pupil_world_path(
            opt, pts[:, 1], pts[:, 2], epd_mm, lam_nm * 1e-3, pts)
        sid_col = batch.source_id[m]
        power_col = batch.birth_power[m]
        for i in range(path.shape[0]):
            if not valid[i]:
                continue
            seq = path[i]
            for k in range(seq.shape[0] - 1):
                rows.append([
                    float(sid_col[i]), lam_m, float(power_col[i]),
                    seq[k, 0], seq[k, 1], seq[k, 2],
                    seq[k + 1, 0], seq[k + 1, 1], seq[k + 1, 2],
                    0.0, 1.0])       # pol_mode=0 (ordinary), rel_power=1.0

    if not rows:
        raise ob.BridgeUnsupported(
            "every fan ray was vignetted or failed in the sequential trace")

    rays = np.asarray(rows, dtype=np.float64)
    write_vtp_polylines(Path(out_path), rays)
    return True, ENGINE_SEQUENTIAL
