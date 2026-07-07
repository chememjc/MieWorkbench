#!/usr/bin/env python
# =============================================================================
# preview_rays.py — lightweight ray-overlay preview for the GUI's 3D views.
#
# Interpreter: /home3/optics/env/bin/python   (numpy/scipy; NO PySide/vtk-py
# GUI deps — this module may import scripts/raytracer/* freely, but the GUI
# side (mieworkbench/core/raypreview.py) only ever shells out to it via
# QProcess, it never imports it directly).
#
# Reads  : <geometry>/model.json (+ faces/*.stl for mesh faces) — the same
#          contract extract_geometry.py writes and run_trace.py reads.
# Writes : a single .vtp polyline file (raytracer.vtkexport.write_vtp_
#          polylines) built from a deterministic --pattern overlay
#          (common.parse_viz_pattern_spec / raytracer.sources.
#          sample_viz_pattern) traced in a THROWAWAY viz-only pass: no
#          detector grids, no energy audit, no physics side effects at
#          all — this script exists purely so the GUI can show "what would
#          the beam do" without paying for a full run_trace.py run.
#
# --only-bodies NAME[,NAME...]: keep only the named bodies (by name or
# label) plus every source AND every detector body (Scene() hard-requires
# at least one of each — dropping every detector isn't "just a filter",
# it makes the scene unconstructible), so rays interact with just the
# requested subset of optics.
#
# Progress: common.progress_emit() already gates stdout '@MIEWB {...}'
# lines on MIEWB_PROGRESS=1 itself (see common.py) — no extra gating here.
# =============================================================================
import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import common                                              # noqa: E402
from raytracer.optprops import load_optical_properties     # noqa: E402
from raytracer.scene import Scene                          # noqa: E402
from raytracer.sources import sample_viz_pattern            # noqa: E402
from raytracer.tracer import Tracer, TraceConfig            # noqa: E402
from raytracer.vtkexport import write_vtp_polylines          # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Deterministic ray-overlay preview (visualization "
                    "only; never runs the physical trace).")
    p.add_argument("--geometry", required=True, metavar="DIR",
                   help="directory containing model.json (extract_geometry.py "
                        "output, e.g. geometry/<stem>)")
    p.add_argument("--optical-properties", default=None, metavar="DIR",
                   help="opticalproperties/ root (default: common.OPTPROPS_DIR)")
    p.add_argument("--out", required=True, metavar="RAYS.VTP",
                   help="output .vtp path")
    p.add_argument("--pattern", default="fan:n=5", metavar="SPEC",
                   help="viz pattern spec, see common.parse_viz_pattern_spec "
                        "('rings:dr=..:nper=..[:nrings=..]' or "
                        "'fan[:n=..]'); default 'fan:n=5'")
    p.add_argument("--only-bodies", default=None, metavar="A,B,...",
                   help="comma-separated body names/labels to keep as "
                        "optics (sources and detectors are always kept)")
    p.add_argument("--max-bounces", type=int, default=None, metavar="N",
                   help="max reflections/refractions per ray (default: "
                        "engine default, currently 6)")
    return p.parse_args(argv)


def _filter_only_bodies(model, only_bodies):
    if not only_bodies:
        return
    keep = {b.strip() for b in only_bodies if b.strip()}
    kept = []
    for b in model["bodies"]:
        if (b["role"] in ("source", "detector")
                or b["name"] in keep or b["label"] in keep):
            kept.append(b)
    model["bodies"] = kept


def main(argv=None):
    args = parse_args(argv)
    common.progress_emit("preview", 0.0, "loading scene")

    geometry_dir = Path(args.geometry)
    model_json = geometry_dir / "model.json"
    if not model_json.exists():
        print("preview_rays: no model.json in %s (run extract_geometry.py "
              "first)" % geometry_dir, file=sys.stderr)
        return 1

    try:
        model = common.load_model(model_json)
    except Exception as exc:
        print("preview_rays: failed to load %s: %s" % (model_json, exc),
              file=sys.stderr)
        return 1

    only_bodies = (args.only_bodies.split(",") if args.only_bodies else None)
    _filter_only_bodies(model, only_bodies)
    try:
        common.validate_model(model)
    except Exception as exc:
        print("preview_rays: --only-bodies left an invalid scene: %s" % exc,
              file=sys.stderr)
        return 1

    if not any(b["role"] == "source" for b in model["bodies"]):
        print("preview_rays: model has no source bodies (nothing to "
              "preview)", file=sys.stderr)
        return 1

    try:
        pattern = common.parse_viz_pattern_spec(args.pattern)
    except ValueError as exc:
        print("preview_rays: bad --pattern %r: %s" % (args.pattern, exc),
              file=sys.stderr)
        return 1

    try:
        props = load_optical_properties(root=args.optical_properties)
    except Exception as exc:
        print("preview_rays: failed to load optical properties: %s" % exc,
              file=sys.stderr)
        return 1

    try:
        scene = Scene(model, props.matdb, props.coatings, optprops=props,
                      geometry_dir=geometry_dir)
    except Exception as exc:
        print("preview_rays: failed to build scene: %s" % exc,
              file=sys.stderr)
        return 1

    common.progress_emit("preview", 0.4, "tracing viz pattern")
    kwargs = {}
    if args.max_bounces is not None:
        kwargs["max_reflections"] = args.max_bounces
    viz_cfg = TraceConfig(rays=1, n_lambda=1, seed=0, viz_rays=1 << 30,
                          **kwargs)
    tracer = Tracer(scene, viz_cfg, {})
    viz_batches = []
    for sid, (bidx, src) in enumerate(scene.sources):
        vb = sample_viz_pattern(scene, scene.bodies[bidx], src, sid,
                                pattern, viz_cfg.n_lambda)
        if vb is not None:
            viz_batches.append(vb)
    if not viz_batches:
        print("preview_rays: --pattern produced no rays for any source "
              "(non-planar emit faces fall back to random viz rays, which "
              "this preview does not draw)", file=sys.stderr)
        return 1

    result = tracer.run(viz_batches)
    rays = result.viz.as_array()

    common.progress_emit("preview", 0.9, "writing %s" % args.out)
    out_path = Path(args.out)
    write_vtp_polylines(out_path, rays)
    common.progress_emit("preview", 1.0, "done", status="completed")
    print("[preview] %d ray polyline(s) -> %s" % (len(rays), out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
