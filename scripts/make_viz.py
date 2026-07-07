#!/usr/bin/env python3
"""make_viz.py -- batch ParaView 3D visualization driver for the FreeCAD ->
ray tracer -> ParaView optical simulation pipeline (see ``viz_common.py``
for the shared pvpython helper library and ``viz_configs.py`` for the
declarative view registry -- add new views there, see its module
docstring).

Must run under pvpython, e.g.::

    /home3/paraview/ParaView-6.1.1-MPI-Linux-Python3.12-x86_64/bin/pvpython \
        --force-offscreen-rendering scripts/make_viz.py \
        --case-dir results/example/quick --model-json geometry/example/model.json

USAGE
-----
    pvpython make_viz.py --case-dir results/<model>/<case> \
        --model-json geometry/<model>/model.json \
        [--views overview3d,top,side,detector_closeup,turntable] \
        [--resolution 1920x1080] [--smoke]

FLOW
----
1. PREP (subprocess, OPTICS_PYTHON, never pvpython itself): run
   ``python -m raytracer.vtkexport --case-dir <case_dir> --model-json
   <model_json> --out-dir <case_dir>/viz`` to convert rays.npy and
   detectors/*.h5 into viz/rays.vtp + viz/det_*.vtp. This is the ONLY
   place numpy/h5py-dependent project code runs; pvpython from here on
   only loads .vtp files with ParaView's own XML reader and loads body
   STLs natively (ParaView reads STL directly -- no conversion needed).
2. Load model.json (validated via common.load_model) and compute the
   scene bounds (viz_common.scene_bounds_m, refined per-render by
   unioning with each Shown reader's actual data bounds).
3. Select views: default is every view in viz_configs.VIEWS; ``--views``
   filters by name. ``--smoke`` forces the view list down to
   ``overview3d`` alone at 800x600, ignoring --views/--resolution.
4. Render each selected view's PNG(s) into ``<case_dir>/viz/``.
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import common  # noqa: E402
import cli_specs  # noqa: E402

from viz_common import (  # noqa: E402
    load_model_json, geometry_dir, default_viz_dir, scene_bounds_m,
    merge_bounds, show_geometry, load_vtp, find_detector_vtps,
    load_detector_meta, show_rgb_cells, show_scalar_cells, has_cell_array,
    new_view, apply_camera,
    camera_three_quarter, camera_top, camera_side, camera_faceon_bounds,
    camera_faceon_meta, camera_turntable, save_png, reset_session,
)
from viz_configs import (  # noqa: E402
    VIEWS, VIEWS_BY_NAME, select_views, DEFAULT_RESOLUTION,
    DETECTOR_CLOSEUP_RESOLUTION, SMOKE_RESOLUTION, SMOKE_VIEW_NAME,
)

from paraview.simple import Render  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Prep step: rays.npy + detectors/*.h5 -> viz/*.vtp (runs under OPTICS_PYTHON)
# ---------------------------------------------------------------------------
def run_vtkexport(case_dir, model_json, viz_dir):
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(common.SCRIPTS_DIR) + (
        os.pathsep + existing if existing else "")
    cmd = [common.OPTICS_PYTHON, "-m", "raytracer.vtkexport",
           "--case-dir", str(case_dir), "--model-json", str(model_json),
           "--out-dir", str(viz_dir)]
    print("[prep] " + " ".join(cmd))
    subprocess.check_call(cmd, env=env)


# ---------------------------------------------------------------------------
# Shared scene assembly (bodies + rays + detectors), reused by every
# static/multi_frame view builder -- only the camera differs between them.
# ---------------------------------------------------------------------------
def _build_scene(ctx, view):
    bounds_list = [ctx["model_bounds"]]
    show_geometry(ctx["model_json"], ctx["geom_dir"], view)
    # (bounds from STL readers are folded into ctx["model_bounds"] already
    #  via scene_bounds_m; body STL bounds should match model.json's bbox_m
    #  closely, so we don't re-query every reader here for speed.)

    rays_path = os.path.join(ctx["viz_dir"], "rays.vtp")
    if os.path.exists(rays_path):
        reader = load_vtp(rays_path)
        show_rgb_cells(reader, view)
        bounds_list.append(reader.GetDataInformation().GetBounds())
    else:
        print("[warn] no rays.vtp at %s" % rays_path)

    for _label, path in find_detector_vtps(ctx["viz_dir"]):
        reader = load_vtp(path)
        show_rgb_cells(reader, view)
        bounds_list.append(reader.GetDataInformation().GetBounds())

    return merge_bounds(bounds_list)


def build_overview3d(view_cfg, ctx):
    reset_session()
    view = new_view(ctx["resolution"])
    bounds = _build_scene(ctx, view)
    Render(view)  # warm-up: consume the implicit reset-camera-on-first-render
    apply_camera(view, camera_three_quarter(bounds))
    Render(view)
    path = os.path.join(ctx["out_dir"], "overview3d.png")
    save_png(view, path, ctx["resolution"])
    return [path]


def build_top(view_cfg, ctx):
    reset_session()
    view = new_view(ctx["resolution"])
    bounds = _build_scene(ctx, view)
    Render(view)
    apply_camera(view, camera_top(bounds, ctx["resolution"]))
    Render(view)
    path = os.path.join(ctx["out_dir"], "top.png")
    save_png(view, path, ctx["resolution"])
    return [path]


def build_side(view_cfg, ctx):
    reset_session()
    view = new_view(ctx["resolution"])
    bounds = _build_scene(ctx, view)
    Render(view)
    apply_camera(view, camera_side(bounds, ctx["resolution"]))
    Render(view)
    path = os.path.join(ctx["out_dir"], "side.png")
    save_png(view, path, ctx["resolution"])
    return [path]


def build_detector_closeup(view_cfg, ctx):
    """One closeup PNG per det_*.vtp: rays + bodies for context, camera
    zoomed face-on to that single detector's own bounding box."""
    detectors = find_detector_vtps(ctx["viz_dir"])
    if not detectors:
        print("[skip] detector_closeup: no det_*.vtp files in %s"
              % ctx["viz_dir"])
        return []

    written = []
    resolution = DETECTOR_CLOSEUP_RESOLUTION
    for label, path in detectors:
        reset_session()
        view = new_view(resolution)
        show_geometry(ctx["model_json"], ctx["geom_dir"], view)
        rays_path = os.path.join(ctx["viz_dir"], "rays.vtp")
        if os.path.exists(rays_path):
            show_rgb_cells(load_vtp(rays_path), view)
        det_reader = load_vtp(path)
        show_rgb_cells(det_reader, view)
        det_bounds = det_reader.GetDataInformation().GetBounds()

        meta = load_detector_meta(path)
        cam = camera_faceon_meta(det_bounds, meta) if meta is not None \
            else camera_faceon_bounds(det_bounds)

        Render(view)  # warm-up
        apply_camera(view, cam)
        Render(view)

        fname = "detector_closeup_%s.png" % label
        out_path = os.path.join(ctx["out_dir"], fname)
        save_png(view, out_path, resolution)
        written.append(out_path)
    return written


def build_turntable(view_cfg, ctx):
    """The overview3d scene built ONCE, then N frames varying only the
    camera azimuth around z at a fixed elevation."""
    reset_session()
    view = new_view(ctx["resolution"])
    bounds = _build_scene(ctx, view)
    Render(view)  # warm-up

    n_frames = view_cfg["n_frames"]
    written = []
    for i in range(n_frames):
        apply_camera(view, camera_turntable(bounds, i, n_frames))
        Render(view)
        fname = "turntable_frame%d.png" % i
        out_path = os.path.join(ctx["out_dir"], fname)
        save_png(view, out_path, ctx["resolution"])
        written.append(out_path)
    return written


def build_rays_polmode(view_cfg, ctx):
    """Same content as overview3d (bodies + rays + detectors), but rays
    are colored by their 'pol_mode' CELL array (0=ordinary/isotropic,
    1=extraordinary o/e-split ray) instead of wavelength -- a diagnostic
    view for birefringent-crystal scenes. Silently skipped (with a
    warning) if rays.vtp doesn't exist yet, or predates the pol_mode
    array (e.g. a stale viz/ reused via --skip-vtkexport from before this
    view existed) -- no visual regression to the other (wavelength-RGB)
    views either way."""
    rays_path = os.path.join(ctx["viz_dir"], "rays.vtp")
    if not os.path.exists(rays_path):
        print("[skip] rays_polmode: no rays.vtp at %s" % rays_path)
        return []
    rays_reader = load_vtp(rays_path)
    if not has_cell_array(rays_reader, "pol_mode"):
        print("[skip] rays_polmode: rays.vtp has no 'pol_mode' CELL array "
              "(stale viz/ predating birefringence support -- re-run "
              "without --skip-vtkexport)")
        return []

    reset_session()
    view = new_view(ctx["resolution"])
    bounds_list = [ctx["model_bounds"]]
    show_geometry(ctx["model_json"], ctx["geom_dir"], view)
    # rays_reader was created before reset_session() cleared prior sources
    # -- rebuild it fresh against the new session
    rays_reader = load_vtp(rays_path)
    show_scalar_cells(rays_reader, view, "pol_mode", value_range=(0.0, 1.0))
    bounds_list.append(rays_reader.GetDataInformation().GetBounds())
    for _label, path in find_detector_vtps(ctx["viz_dir"]):
        det_reader = load_vtp(path)
        show_rgb_cells(det_reader, view)
        bounds_list.append(det_reader.GetDataInformation().GetBounds())
    bounds = merge_bounds(bounds_list)

    Render(view)  # warm-up
    apply_camera(view, camera_three_quarter(bounds))
    Render(view)
    path = os.path.join(ctx["out_dir"], "rays_polmode.png")
    save_png(view, path, ctx["resolution"])
    return [path]


BUILDERS = {
    "build_overview3d": build_overview3d,
    "build_top": build_top,
    "build_side": build_side,
    "build_detector_closeup": build_detector_closeup,
    "build_turntable": build_turntable,
    "build_rays_polmode": build_rays_polmode,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = cli_specs.build_parser("viz")
    return p.parse_args(argv)


def resolve_view_names(requested):
    if requested is None:
        return None
    names = []
    for name in requested.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in VIEWS_BY_NAME:
            print("[warn] unknown view '%s' (known: %s), skipping"
                  % (name, ", ".join(VIEWS_BY_NAME)), file=sys.stderr)
            continue
        names.append(name)
    return names


def main():
    args = parse_args()
    case_dir = os.path.abspath(args.case_dir.rstrip("/"))
    if not os.path.isdir(case_dir):
        raise SystemExit("case directory not found: %s" % case_dir)
    model_json_path = os.path.abspath(args.model_json)
    if not os.path.isfile(model_json_path):
        raise SystemExit("model.json not found: %s" % model_json_path)

    viz_dir = str(default_viz_dir(case_dir))
    out_dir = os.path.abspath(args.out) if args.out else viz_dir
    os.makedirs(out_dir, exist_ok=True)

    if not args.skip_vtkexport:
        run_vtkexport(case_dir, model_json_path, viz_dir)
    else:
        print("[prep] --skip-vtkexport: assuming %s already has current "
              "rays.vtp / det_*.vtp" % viz_dir)

    model_json = load_model_json(model_json_path)
    geom_dir = geometry_dir(model_json_path)
    model_bounds = scene_bounds_m(model_json)

    if args.smoke:
        view_names = [SMOKE_VIEW_NAME]
        resolution = SMOKE_RESOLUTION
    else:
        view_names = resolve_view_names(args.views)
        resolution = args.resolution

    selected = select_views(view_names)

    print("[setup] case dir  = %s" % case_dir)
    print("[setup] model json = %s" % model_json_path)
    print("[setup] viz dir   = %s" % viz_dir)
    print("[setup] out dir   = %s" % out_dir)
    print("[setup] resolution = %dx%d" % resolution)
    print("[setup] views: %s" % ", ".join(v["name"] for v in selected))

    ctx = {
        "case_dir": case_dir, "model_json": model_json, "geom_dir": geom_dir,
        "viz_dir": viz_dir, "out_dir": out_dir, "resolution": resolution,
        "model_bounds": model_bounds, "smoke": args.smoke,
    }

    if not selected:
        print("[warn] no views selected")
    for i, view_cfg in enumerate(selected):
        builder = BUILDERS[view_cfg["builder"]]
        print("[render] %s ..." % view_cfg["name"])
        common.progress_emit("viz", i / max(1, len(selected)),
                             view_cfg["name"], case_dir=case_dir)
        written = builder(view_cfg, ctx)
        print("[render] %s: wrote %d file(s)" % (view_cfg["name"], len(written)))

    common.progress_emit("viz", 1.0, "%d view(s) rendered" % len(selected),
                         case_dir=case_dir, status="completed")
    print("[done]")


if __name__ == "__main__":
    main()
