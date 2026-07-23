"""viz_common.py -- shared pvpython library for the optical ray tracer's
ParaView batch visualization suite (ParaView 6.1.1). See ``viz_configs.py``
for the declarative view registry and ``make_viz.py`` for the driver that
ties them together.

Ported from the antenna project's ``viz_common.py`` (same pvpython/ParaView
conventions: offscreen rendering, one reader/pipeline per render, the
camera warm-up Render() fix); the reader/geometry/camera helpers are
rewritten for this project's data model:

  * body geometry: geometry/<model>/faces/<face_id>.stl, per-FACE binary
    STL files ALREADY in SI metres (no mm->m Transform needed -- unlike
    the antenna project, extract_geometry.py here scales a COPY of each
    face to metres before tessellating).
  * rays / detectors: results/<model>/<case>/viz/{rays,det_*}.vtp, produced
    by scripts/raytracer/vtkexport.py from rays.npy / detectors/*.h5 (see
    that module's docstring). pvpython never imports the raytracer package
    itself -- make_viz.py shells out to OPTICS_PYTHON to run
    ``python -m raytracer.vtkexport`` as a prep step, then this module only
    LOADS the resulting .vtp files with ParaView's own XML reader.
  * both ray segments and detector quads carry a 3-component uint8 CELL
    array named 'rgb' -- colored directly (MapScalars = 0), not through a
    scalar lookup table. Ray segments additionally carry 'rel_power'
    (float32, power/birth_power) and, when the vtkexport prep step ran
    with --dim-rays, a baked 4-component 'rgba' array for
    attenuation-dimmed renders (alpha computed under OPTICS_PYTHON, NOT
    via a pvpython ProgrammableFilter -- that leaks numpy_interface
    names into __main__ and shadows builtins like max()).

PV 6.1.1 notes carried over from the antenna project:
  * The first Render() after new representations are Shown auto-resets the
    camera to fit visible data, clobbering any camera set beforehand; a
    warm-up Render() must precede ``apply_camera()``.
  * A CELL array of exactly 3 (or 4) unsigned-char components is rendered
    as direct RGB(A) color once ``disp.MapScalars = 0`` is set after
    ``ColorBy`` -- no lookup table / scalar range involved.

This module is a PURE LIBRARY: importing it has no side effects (no server
connection, no view/source creation at import time).

Must run under::

    "$MIEWB_PVPYTHON" \
        --force-offscreen-rendering <driver>.py ...
"""

import math
import os
import stat
import sys

from paraview.simple import *  # noqa: F401,F403  (standard ParaView idiom)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import common  # noqa: E402

from viz_configs import body_appearance  # noqa: E402


# ---------------------------------------------------------------------------
# model.json / filesystem discovery
# ---------------------------------------------------------------------------
def load_model_json(model_json_path):
    """Validated model.json dict, via the project's own contract validator
    (common.load_model) -- the SAME gate run_trace.py itself passes
    through, so a viz run on a bad model.json fails loudly here too."""
    return common.load_model(model_json_path)


def geometry_dir(model_json_path):
    """geometry/<model>/ -- the directory containing model.json and the
    faces/<id>.stl files it references (mesh_stl paths are relative to
    this directory)."""
    from pathlib import Path
    return Path(model_json_path).resolve().parent


def default_viz_dir(case_dir):
    from pathlib import Path
    return Path(case_dir) / "viz"


def scene_bounds_m(model_json):
    """(xmin,xmax,ymin,ymax,zmin,zmax) in metres, unioned across every
    body's bbox_m (cheap -- no STL read needed)."""
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = float("-inf")
    for body in model_json["bodies"]:
        bb = body.get("bbox_m")
        if not bb:
            continue
        (x0, y0, z0), (x1, y1, z1) = bb["min"], bb["max"]
        xmin, xmax = min(xmin, x0), max(xmax, x1)
        ymin, ymax = min(ymin, y0), max(ymax, y1)
        zmin, zmax = min(zmin, z0), max(zmax, z1)
    if xmin > xmax:
        # degenerate (no bodies had a bbox) -- a small unit box around the
        # origin so camera math never divides by zero
        return (-1e-3, 1e-3, -1e-3, 1e-3, -1e-3, 1e-3)
    return (xmin, xmax, ymin, ymax, zmin, zmax)


def merge_bounds(bounds_list):
    """Union of several 6-tuples (xmin,xmax,ymin,ymax,zmin,zmax), ignoring
    any that are empty/invalid (VTK reports (1,-1,...) style inverted
    bounds for empty datasets)."""
    valid = [b for b in bounds_list
             if b[0] <= b[1] and b[2] <= b[3] and b[4] <= b[5]]
    if not valid:
        return (-1e-3, 1e-3, -1e-3, 1e-3, -1e-3, 1e-3)
    xmin = min(b[0] for b in valid)
    xmax = max(b[1] for b in valid)
    ymin = min(b[2] for b in valid)
    ymax = max(b[3] for b in valid)
    zmin = min(b[4] for b in valid)
    zmax = max(b[5] for b in valid)
    return (xmin, xmax, ymin, ymax, zmin, zmax)


# ---------------------------------------------------------------------------
# Body geometry (STL, already in metres -- no scaling)
# ---------------------------------------------------------------------------
def iter_face_stls(model_json, geom_dir):
    """Yield (face_id, role, stl_abs_path) for every face of every non-
    ignored body in model_json['bodies'] (model.json already excludes
    'ignored'-role bodies)."""
    for body in model_json["bodies"]:
        role = body.get("role")
        for face in body.get("faces", []):
            stl_rel = face.get("mesh_stl")
            if not stl_rel:
                continue
            yield face["id"], role, str(geom_dir / stl_rel)


def show_geometry(model_json, geom_dir, view):
    """Show every body face STL as a translucent 'glass' surface, GROUPED
    by owning body role into at most a handful of merged representations
    (one AppendDatasets per role actually present) rather than one Show()
    per face.

    Why: a typical scene has 20-30+ individual face STLs. Showing each as
    its own representation, together with the (large, opaque, direct-RGB)
    detector quads, was observed to exhaust a hardware texture-unit
    budget -- ParaView logged 'Hardware does not support the number of
    textures defined' and the view rendered as a solid magenta frame
    (VTK/OpenGL's shader-compile-failure color), reproducible specifically
    when translucent body geometry AND more than one detector quad were
    shown in the same view (isolated by bisection: bodies+rays alone
    render fine; bodies+all 3 detector quads together reproduce the
    failure even with rays omitted). Merging faces by role cuts the body
    representation count ~10x (e.g. 28 faces -> 3 roles) while preserving
    per-role coloring, which resolved the failure in testing.

    Returns (list of Show() displays, list of per-representation bounds
    tuples)."""
    by_role = {}
    for face_id, role, stl_path in iter_face_stls(model_json, geom_dir):
        reader = STLReader(FileNames=[stl_path])
        reader.UpdatePipeline()
        by_role.setdefault(role, []).append(reader)

    disps = []
    bounds_list = []
    for role, readers in by_role.items():
        if len(readers) == 1:
            merged = readers[0]
        else:
            merged = AppendDatasets(Input=readers)
            merged.UpdatePipeline()
        appearance = body_appearance(role)
        disp = Show(merged, view)
        disp.Representation = "Surface"
        disp.ColorArrayName = [None, ""]
        disp.DiffuseColor = list(appearance["color"])
        disp.AmbientColor = list(appearance["color"])
        disp.Opacity = appearance["opacity"]
        disps.append(disp)
        bounds_list.append(merged.GetDataInformation().GetBounds())
    return disps, bounds_list


# ---------------------------------------------------------------------------
# rays.vtp / det_*.vtp loaders
# ---------------------------------------------------------------------------
def load_vtp(path):
    """XMLPolyDataReader for a .vtp file written by raytracer.vtkexport."""
    if not os.path.exists(path):
        raise FileNotFoundError("vtp file not found: %s" % path)
    reader = XMLPolyDataReader(FileName=[path])
    reader.UpdatePipeline()
    return reader


def find_detector_vtps(viz_dir):
    """Sorted list of (label_guess, path) for every det_*.vtp in viz_dir.
    label_guess is the filename stem with 'det_' stripped (safe_label form,
    '.' already turned to '_' by vtkexport -- not the original face id)."""
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(str(viz_dir), "det_*.vtp"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        out.append((stem[len("det_"):], p))
    return out


def load_detector_meta(vtp_path):
    """{'normal','xhat','yhat','center'} sidecar JSON written by
    raytracer.vtkexport.write_detector_quads next to det_*.vtp, or None if
    missing (e.g. viz/ produced by an older run, or --skip-vtkexport
    against stale files) -- callers fall back to an AABB-based guess."""
    import json
    meta_path = vtp_path[:-4] + ".json" if vtp_path.endswith(".vtp") \
        else vtp_path + ".json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as fh:
        return json.load(fh)


def show_rgb_cells(reader, view, representation="Surface", array="rgb"):
    """Show `reader`'s polydata, colored directly by its 3-component uint8
    CELL array 'rgb' (no lookup table -- MapScalars = 0 per VTK/ParaView's
    direct-color convention). `array` may name a 4-component uint8 cell
    array instead (e.g. dim_rays_filter's 'rgba'), rendered as direct
    RGBA with per-cell opacity."""
    disp = Show(reader, view)
    disp.Representation = representation
    ColorBy(disp, ("CELLS", array))
    disp.MapScalars = 0
    return disp


def has_cell_array(reader, name):
    """True if `reader`'s polydata has a CELL array named `name`. Used to
    gracefully skip a scalar-colored view (e.g. rays.vtp's 'pol_mode')
    against a stale viz/ produced by an older vtkexport run (or
    --skip-vtkexport reusing one) that predates that array, rather than
    raising a ParaView/VTK lookup error."""
    try:
        return name in reader.CellData.keys()
    except Exception:
        return False


def show_scalar_cells(reader, view, array_name, value_range=(0.0, 1.0),
                      representation="Surface"):
    """Show `reader`'s polydata colored by a scalar CELL array through a
    ParaView lookup table -- unlike show_rgb_cells's direct-color path
    (used for precomputed wavelength/detector-irradiance RGB triples),
    this is for diagnostic scalar overlays where no RGB is precomputed on
    the writing side, e.g. rays.vtp's 'pol_mode' (0=ordinary/isotropic,
    1=extraordinary -- a birefringent crystal's o/e split). A 2-point
    blue (low) -> orange (high) transfer function spans `value_range`
    (default (0,1), matching pol_mode's two states) with a labeled scalar
    bar so the two colors are identifiable in the rendered PNG."""
    disp = Show(reader, view)
    disp.Representation = representation
    ColorBy(disp, ("CELLS", array_name))
    disp.MapScalars = 1
    lut = GetColorTransferFunction(array_name)
    lut.RGBPoints = [value_range[0], 0.20, 0.55, 0.95,
                     value_range[1], 0.95, 0.45, 0.10]
    lut.RescaleTransferFunction(value_range[0], value_range[1])
    disp.SetScalarBarVisibility(view, True)
    sb = GetScalarBar(lut, view)
    sb.Title = array_name
    sb.ComponentTitle = ""
    return disp, lut


# ---------------------------------------------------------------------------
# Views / cameras
# ---------------------------------------------------------------------------
def new_view(resolution=(1920, 1080)):
    """A fresh offscreen RenderView with a neutral dark background."""
    view = CreateView("RenderView")
    view.ViewSize = [int(resolution[0]), int(resolution[1])]
    view.UseColorPaletteForBackground = 0
    view.BackgroundColorMode = "Single Color"
    view.Background = [0.07, 0.08, 0.10]
    view.OrientationAxesVisibility = 1
    return view


def apply_camera(view, cam):
    """Apply a camera dict (see camera_* builders below)."""
    view.CameraFocalPoint = list(cam["focal"])
    view.CameraPosition = list(cam["position"])
    view.CameraViewUp = list(cam["up"])
    if cam.get("parallel"):
        view.CameraParallelProjection = 1
        if cam.get("parallel_scale") is not None:
            view.CameraParallelScale = float(cam["parallel_scale"])
    else:
        view.CameraParallelProjection = 0
        if cam.get("view_angle") is not None:
            view.CameraViewAngle = float(cam["view_angle"])
    return view


def _bounds_center(bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)


def _bounds_radius(bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return 0.5 * math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2
                           + (zmax - zmin) ** 2) or 1e-6


def camera_three_quarter(bounds, view_angle=30.0, margin=1.35):
    """Perspective off-axis 3/4 camera framing `bounds`, positioned
    diagonally above/behind/right of the domain center."""
    center = _bounds_center(bounds)
    radius = _bounds_radius(bounds)
    span = margin * radius
    position = [center[0] - 0.85 * span, center[1] - 1.05 * span,
               center[2] + 0.65 * span]
    return {"parallel": False, "focal": list(center), "position": position,
            "up": [0.0, 0.0, 1.0], "view_angle": view_angle}


def _parallel_scale_for(bounds, h_ax, v_ax, resolution, margin):
    """Half-height parallel_scale that fits both in-plane extents of
    `bounds` given the render aspect ratio (same idea as the antenna
    project's camera_faceon)."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    extents = (xmax - xmin, ymax - ymin, zmax - zmin)
    half_w = 0.5 * extents[h_ax]
    half_h = 0.5 * extents[v_ax]
    aspect = float(resolution[0]) / float(resolution[1])
    scale = margin * max(half_h, half_w / aspect)
    return scale if scale > 0 else 1.0


def camera_top(bounds, resolution=(1920, 1080), margin=1.15):
    """Parallel-projection camera looking straight down -z."""
    center = _bounds_center(bounds)
    radius = _bounds_radius(bounds)
    position = [center[0], center[1], center[2] + max(radius, 1e-6) * 3.0]
    scale = _parallel_scale_for(bounds, 0, 1, resolution, margin)
    return {"parallel": True, "focal": list(center), "position": position,
            "up": [0.0, 1.0, 0.0], "parallel_scale": scale}


def camera_side(bounds, resolution=(1920, 1080), margin=1.15):
    """Parallel-projection camera looking down -y."""
    center = _bounds_center(bounds)
    radius = _bounds_radius(bounds)
    position = [center[0], center[1] + max(radius, 1e-6) * 3.0, center[2]]
    scale = _parallel_scale_for(bounds, 0, 2, resolution, margin)
    return {"parallel": True, "focal": list(center), "position": position,
            "up": [0.0, 0.0, 1.0], "parallel_scale": scale}


def _norm3(v):
    length = math.sqrt(sum(c * c for c in v))
    return [c / length for c in v] if length > 0 else [0.0, 0.0, 1.0]


def camera_faceon_meta(bounds, meta, margin=1.2):
    """Parallel-projection camera looking straight at a detector plane
    using its ACTUAL basis (meta = the JSON sidecar
    raytracer.vtkexport.write_detector_quads writes next to det_*.vtp).
    Unlike camera_faceon_bounds' bbox-thinnest-axis guess, this is exact
    even for a detector whose normal is not aligned with a world axis (a
    tilted plane's own axis-aligned bounding box need not be thin along
    any single coordinate axis, only along its true normal)."""
    center = _bounds_center(bounds)
    radius = _bounds_radius(bounds)
    normal = _norm3(meta["normal"])
    up = _norm3(meta["yhat"])
    standoff = max(radius, 1e-6) * 4.0
    position = [center[i] + normal[i] * standoff for i in range(3)]
    return {"parallel": True, "focal": list(center), "position": position,
            "up": up, "parallel_scale": margin * max(radius, 1e-6)}


def camera_faceon_bounds(bounds, margin=1.2):
    """Fallback parallel-projection camera framing a detector's own
    (small) bounds when no orientation sidecar is available: looks along
    the bounds' thinnest axis (an exact face-on view only when the
    detector plane happens to be axis-aligned -- prefer
    camera_faceon_meta when a sidecar JSON is available)."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    extents = [xmax - xmin, ymax - ymin, zmax - zmin]
    thin_axis = extents.index(min(extents))
    center = _bounds_center(bounds)
    radius = _bounds_radius(bounds)
    standoff = max(radius, 1e-6) * 4.0
    position = list(center)
    up = [0.0, 0.0, 1.0]
    if thin_axis == 2:
        position[2] += standoff
        up = [0.0, 1.0, 0.0]
    elif thin_axis == 1:
        position[1] += standoff
        up = [0.0, 0.0, 1.0]
    else:
        position[0] += standoff
        up = [0.0, 0.0, 1.0]
    parallel_scale = margin * max(radius, 1e-6)
    return {"parallel": True, "focal": list(center), "position": position,
            "up": up, "parallel_scale": parallel_scale}


def camera_turntable(bounds, frame_i, n_frames,
                     elevation_deg=25.0, view_angle=30.0, margin=1.35):
    """Perspective camera at a fixed elevation, azimuth = frame_i/n_frames
    of a full turn around the +z axis through the scene center."""
    center = _bounds_center(bounds)
    radius = _bounds_radius(bounds)
    span = margin * radius
    az = 2.0 * math.pi * (frame_i / float(n_frames))
    el = math.radians(elevation_deg)
    dx = span * math.cos(el) * math.cos(az)
    dy = span * math.cos(el) * math.sin(az)
    dz = span * math.sin(el)
    position = [center[0] + dx, center[1] + dy, center[2] + dz]
    return {"parallel": False, "focal": list(center), "position": position,
            "up": [0.0, 0.0, 1.0], "view_angle": view_angle}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def save_png(view, path, resolution):
    """Render and save one screenshot, creating the parent directory as
    needed."""
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    Render(view)
    SaveScreenshot(str(path), view,
                   ImageResolution=[int(resolution[0]), int(resolution[1])])
    return path


def reset_session():
    """Delete all sources and views so successive views do not leak
    pipeline state between builds."""
    for view in GetViews():
        try:
            Delete(view)
        except Exception:
            pass
    for src in list(GetSources().values()):
        try:
            Delete(src)
        except Exception:
            pass
