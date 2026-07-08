# =============================================================================
# vtkexport.py — hand-rolled VTK XML PolyData (.vtp) writer for the
# orchestration/visualization layer. No `vtk` python package dependency:
# ParaView's pvpython reads these files with its own vtkXMLPolyDataReader,
# but nothing on the writing side needs to link against VTK — this module
# only needs numpy (+ h5py for write_detector_quads) and is meant to run
# under OPTICS_PYTHON, never under pvpython (see make_viz.py's module
# docstring for the reasoning: pvpython never imports the raytracer package).
#
# Contents:
#   write_vtp_polylines(path, rays)          rays.npy (N,9|10|11) ->
#                                             ray segments (10th column,
#                                             if present, is pol_mode:
#                                             0=ordinary/isotropic,
#                                             1=extraordinary o/e-split ray;
#                                             11th, if present, is
#                                             rel_power = power/birth_power)
#   write_vtp_mesh(path, vertices, triangles, color=None)
#                                             triangle soup -> surface mesh
#   write_detector_quads(path, h5_path, model=None, max_cells=512)
#                                             detector spectral cube -> a
#                                             decimated colored quad grid,
#                                             positioned in world 3D space
#
# CLI (invoked as `python -m raytracer.vtkexport ...` under OPTICS_PYTHON by
# make_viz.py's prep step — pvpython itself only ever LOADS the .vtp files
# this produces, it never runs this module):
#
#   python -m raytracer.vtkexport --case-dir results/<model>/<case> \
#       --model-json geometry/<model>/model.json [--out-dir DIR] \
#       [--max-cells 512]
#
# writes <out-dir>/rays.vtp (if rays.npy present) and one
# <out-dir>/det_<safe-label>.vtp per results/<case>/detectors/*.h5.
#
# VTK XML inline-binary format notes (verified against ParaView 6.1.1's
# vtkXMLDataParser): a DataArray with format="binary" stores ONE
# base64-encoded blob whose decoded bytes are a little-endian header (here
# UInt32, matching the VTKFile-level header_type="UInt32" declared on the
# root element) giving the *uncompressed byte length of the payload*,
# immediately followed by the raw payload bytes. There is no line
# wrapping requirement; a single long base64 line is valid XML character
# data and parses fine.
# =============================================================================
import argparse
import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

from .detector import cie_xyz_weights, _XYZ_TO_SRGB, spectral_cube_to_srgb

_VTK_TYPES = {
    np.dtype("float64"): "Float64",
    np.dtype("float32"): "Float32",
    np.dtype("int64"): "Int64",
    np.dtype("int32"): "Int32",
    np.dtype("int16"): "Int16",
    np.dtype("uint8"): "UInt8",
}


# ---------------------------------------------------------------------------
# wavelength -> sRGB (same CIE 1931 approach as post_process.wavelength_rgb,
# reused here so 3D ray colors and the 2D XY cross-section plot agree)
# ---------------------------------------------------------------------------
def wavelength_to_rgb8(lam_m):
    """(N,) wavelengths in metres -> (N,3) uint8 sRGB, hue-normalized (each
    ray's color is scaled to full brightness; a wavelength outside the
    visible CIE range with zero response falls back to mid-gray 0.3)."""
    lam_m = np.atleast_1d(np.asarray(lam_m, dtype=np.float64))
    lam_nm = lam_m / 1e-9
    w = cie_xyz_weights(lam_nm)
    rgb = np.clip(w @ _XYZ_TO_SRGB.T, 0, None)
    mx = rgb.max(axis=-1, keepdims=True)
    rgb = np.where(mx > 0, rgb / mx, 0.3)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Low-level XML / inline-binary helpers
# ---------------------------------------------------------------------------
def _b64_block(data_bytes):
    header = struct.pack("<I", len(data_bytes))
    return base64.b64encode(header + data_bytes).decode("ascii")


def _data_array(name, arr, ncomp=1):
    arr = np.ascontiguousarray(arr)
    vtk_type = _VTK_TYPES.get(arr.dtype)
    if vtk_type is None:
        raise ValueError("unsupported dtype %s for array %r" % (arr.dtype, name))
    blob = _b64_block(arr.tobytes())
    return ('      <DataArray type="%s" Name="%s" NumberOfComponents="%d" '
            'format="binary">\n%s\n      </DataArray>\n'
            % (vtk_type, name, ncomp, blob))


def _cell_data_block(cell_data):
    if not cell_data:
        return "      <CellData>\n      </CellData>\n"
    # Declare 'rgb' (when present) as the ACTIVE cell scalars: without the
    # Scalars= attribute no array is active after reading, and any VTK
    # mapper in UseCellData mode silently falls back to the actor's flat
    # default color (rays rendered WHITE in the GUI despite a perfectly
    # good rgb array sitting in the file).
    names = [name for name, _arr, _n in cell_data]
    active = ' Scalars="rgb"' if "rgb" in names else ""
    parts = ["      <CellData%s>\n" % active]
    for name, arr, ncomp in cell_data:
        parts.append(_data_array(name, arr, ncomp))
    parts.append("      </CellData>\n")
    return "".join(parts)


def _point_data_block(point_data):
    if not point_data:
        return "      <PointData>\n      </PointData>\n"
    parts = ["      <PointData>\n"]
    for name, arr, ncomp in point_data:
        parts.append(_data_array(name, arr, ncomp))
    parts.append("      </PointData>\n")
    return "".join(parts)


def _write_vtp(path, points, lines=None, polys=None, verts=None,
               point_data=None, cell_data=None):
    """Generic single-Piece PolyData writer. `lines`/`polys`/`verts` are each
    (connectivity, offsets) int32 array pairs, or None."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n_points = points.shape[0]

    def cell_counts(pair):
        if pair is None:
            return 0
        _, offsets = pair
        return int(offsets.shape[0])

    n_verts = cell_counts(verts)
    n_lines = cell_counts(lines)
    n_polys = cell_counts(polys)

    out = []
    out.append('<?xml version="1.0"?>\n')
    out.append('<VTKFile type="PolyData" version="1.0" '
                'byte_order="LittleEndian" header_type="UInt32">\n')
    out.append("  <PolyData>\n")
    out.append('    <Piece NumberOfPoints="%d" NumberOfVerts="%d" '
                'NumberOfLines="%d" NumberOfStrips="0" NumberOfPolys="%d">\n'
                % (n_points, n_verts, n_lines, n_polys))
    out.append(_point_data_block(point_data))
    out.append(_cell_data_block(cell_data))
    out.append("      <Points>\n")
    out.append(_data_array("points", points, 3))
    out.append("      </Points>\n")

    def cell_block(tag, pair):
        out.append("      <%s>\n" % tag)
        if pair is not None:
            connectivity, offsets = pair
            out.append(_data_array(
                "connectivity", np.asarray(connectivity, dtype=np.int32), 1))
            out.append(_data_array(
                "offsets", np.asarray(offsets, dtype=np.int32), 1))
        out.append("      </%s>\n" % tag)

    cell_block("Verts", verts)
    cell_block("Lines", lines)
    cell_block("Strips", None)
    cell_block("Polys", polys)

    out.append("    </Piece>\n")
    out.append("  </PolyData>\n")
    out.append("</VTKFile>\n")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(out))
    return path


# ---------------------------------------------------------------------------
# write_vtp_polylines — rays.npy (N,9|10|11) -> N line cells
# ---------------------------------------------------------------------------
def write_vtp_polylines(path, rays, dim_mode="off", dim_floor=0.0):
    """rays: (N,9) array [source_id, lam_m, power_W, x0,y0,z0, x1,y1,z1],
    (N,10) with a trailing pol_mode column (0=isotropic/ordinary ray,
    1=extraordinary ray -- a birefringent crystal's o/e split), or (N,11)
    with a further rel_power column (power/birth_power in [0,1] -- the
    per-segment attenuation the renderers map to opacity). Older narrower
    files are still accepted; missing columns default to 0 (NOTE: the
    caller must pass the array with its ACTUAL column count -- this
    function does not force-reshape, since that would silently corrupt a
    legitimately wider array).

    Writes N 2-point line cells (no point data), with CELL data:
      rgb (3x uint8, wavelength color), power (float32 W), source_id
      (int16), pol_mode (uint8, 0=ordinary/isotropic, 1=extraordinary),
      and, for (N,11) input, rel_power (float32, [0,1]). rgb stays the
      ACTIVE scalars; rel_power is appearance-neutral data so each
      consumer applies its own dimming curve at render time (the GUI
      composes its own RGBA from it per its View-menu setting).

    dim_mode 'linear'|'sqrt' additionally bakes an rgba (4x uint8) cell
    array -- rgb plus alpha = f(rel_power) floored at dim_floor percent
    -- for ParaView's direct-RGBA path (make_viz --dim-rays; the curve
    must be applied HERE because pvpython's ProgrammableFilter leaks
    numpy_interface names into __main__, shadowing builtins). Ignored
    for input narrower than (N,11).
    """
    rays = np.asarray(rays, dtype=np.float64)
    if rays.ndim != 2 or rays.shape[1] not in (9, 10, 11):
        raise ValueError(
            "write_vtp_polylines: rays must be (N,9), (N,10) or (N,11), "
            "got shape %r" % (rays.shape,))
    has_pol = rays.shape[1] >= 10
    has_rel = rays.shape[1] >= 11
    n = rays.shape[0]
    if n == 0:
        points = np.zeros((0, 3), dtype=np.float64)
        connectivity = np.zeros(0, dtype=np.int32)
        offsets = np.zeros(0, dtype=np.int32)
        rgb = np.zeros((0, 3), dtype=np.uint8)
        power = np.zeros(0, dtype=np.float32)
        source_id = np.zeros(0, dtype=np.int16)
        pol_mode = np.zeros(0, dtype=np.uint8)
        rel_power = np.zeros(0, dtype=np.float32)
    else:
        p0 = rays[:, 3:6]
        p1 = rays[:, 6:9]
        points = np.empty((2 * n, 3), dtype=np.float64)
        points[0::2] = p0
        points[1::2] = p1
        connectivity = np.arange(2 * n, dtype=np.int32)
        offsets = np.arange(2, 2 * n + 1, 2, dtype=np.int32)
        rgb = wavelength_to_rgb8(rays[:, 1])
        power = rays[:, 2].astype(np.float32)
        source_id = rays[:, 0].astype(np.int16)
        pol_mode = (rays[:, 9] if has_pol
                    else np.zeros(n)).astype(np.uint8)
        if has_rel:
            rel_power = np.clip(rays[:, 10], 0.0, 1.0).astype(np.float32)

    cell_data = [("rgb", rgb, 3), ("power", power, 1),
                 ("source_id", source_id, 1),
                 ("pol_mode", pol_mode, 1)]
    if has_rel:
        cell_data.append(("rel_power", rel_power, 1))
        if dim_mode != "off":
            a = np.sqrt(rel_power) if dim_mode == "sqrt" else rel_power
            a = np.maximum(a, float(dim_floor) / 100.0)
            rgba = np.empty((n, 4), dtype=np.uint8)
            rgba[:, :3] = rgb
            rgba[:, 3] = np.clip(np.round(255.0 * a), 0,
                                 255).astype(np.uint8)
            cell_data.append(("rgba", rgba, 4))
    return _write_vtp(path, points, lines=(connectivity, offsets),
                      cell_data=cell_data)


# ---------------------------------------------------------------------------
# write_vtp_mesh — arbitrary triangle soup -> surface mesh
# ---------------------------------------------------------------------------
def write_vtp_mesh(path, vertices, triangles, color=None):
    """vertices: (M,3) float; triangles: (T,3) int vertex indices.

    `color`, if given, is either a single (r,g,b) 0-255 uint8-ish triple
    (broadcast to every triangle as CELL data) or a (T,3) per-triangle
    array.
    """
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    ntri = triangles.shape[0]
    connectivity = triangles.reshape(-1).astype(np.int32)
    offsets = (np.arange(1, ntri + 1, dtype=np.int32) * 3)

    cell_data = []
    if color is not None:
        color = np.asarray(color)
        if color.ndim == 1:
            rgb = np.tile(np.clip(color, 0, 255).astype(np.uint8), (ntri, 1))
        else:
            rgb = np.clip(color, 0, 255).astype(np.uint8)
        cell_data.append(("rgb", rgb, 3))

    return _write_vtp(path, vertices, polys=(connectivity, offsets),
                      cell_data=cell_data)


# ---------------------------------------------------------------------------
# write_detector_quads — spectral cube -> decimated colored quad grid,
# positioned in the scene's world 3D frame
# ---------------------------------------------------------------------------
def _find_face_origin(model, label):
    """Look up the plane origin (world, metres) of the face named `label`
    in a loaded model.json dict, or None if not found / not a plane."""
    if model is None:
        return None
    for body in model.get("bodies", []):
        for f in body.get("faces", []):
            if f.get("id") == label:
                surf = f.get("surface", {})
                if surf.get("type") == "plane" and "origin" in surf:
                    return np.asarray(surf["origin"], dtype=np.float64)
                return None
    return None


def _decimate_block_mean(arr, stride_h, stride_w):
    """arr: (H,W) or (H,W,C). Block-mean downsample by integer strides,
    truncating any remainder rows/cols (H,W need not be exact multiples)."""
    h, w = arr.shape[0], arr.shape[1]
    h_trim = (h // stride_h) * stride_h
    w_trim = (w // stride_w) * stride_w
    a = arr[:h_trim, :w_trim]
    new_h = h_trim // stride_h
    new_w = w_trim // stride_w
    if arr.ndim == 2:
        a = a.reshape(new_h, stride_h, new_w, stride_w)
        return a.mean(axis=(1, 3))
    c = arr.shape[2]
    a = a.reshape(new_h, stride_h, new_w, stride_w, c)
    return a.mean(axis=(1, 3))


def write_detector_quads(path, h5_path, model=None, max_cells=512):
    """Read a run_trace.py detectors/<label>.h5 file and write a decimated
    (<= max_cells x max_cells) quad grid .vtp, with per-cell `rgb` (sRGB
    from the spectral cube, via detector.spectral_cube_to_srgb) and
    `irradiance_W_m2` CELL data arrays, positioned in world 3D space.

    World placement replicates DetectorGrid.__init__'s pixel_centers
    construction (raytracer/detector.py): world = x*xhat + y*yhat + n_comp,
    where n_comp is the component of the detector plane's origin
    orthogonal to xhat/yhat. The h5 file itself does not store the plane
    origin (only xhat/yhat/normal/x_lo/y_lo/pixel_m), so `model` (a loaded
    model.json dict) is needed to recover it by matching the h5 attrs
    'label' against a face id; if `model` is None or the face/origin can't
    be found, n_comp defaults to zero (a warning is printed) and the quad
    grid is placed in the xhat/yhat plane through the world origin instead
    — still internally consistent, just not aligned with the rest of the
    scene.
    """
    import h5py

    with h5py.File(h5_path, "r") as h:
        cube = h["spectral_cube_mean"][...]
        mask = h["mask"][...]
        attrs = dict(h.attrs)

    label = attrs["label"]
    H, W = int(attrs["H"]), int(attrs["W"])
    pixel_m = float(attrs["pixel_m"])
    xhat = np.asarray(attrs["xhat"], dtype=np.float64)
    yhat = np.asarray(attrs["yhat"], dtype=np.float64)
    normal = np.asarray(attrs["normal"], dtype=np.float64)
    x_lo, y_lo = float(attrs["x_lo"]), float(attrs["y_lo"])
    lam_lo, lam_hi = float(attrs["lam_lo_m"]), float(attrs["lam_hi_m"])

    origin = _find_face_origin(model, label)
    if origin is not None:
        n_comp = (origin - np.dot(origin, xhat) * xhat
                  - np.dot(origin, yhat) * yhat)
    else:
        print("[vtkexport] WARNING: no plane origin found for detector "
              "%r (pass --model-json to place it correctly) — placing "
              "at n_comp=0" % label, file=sys.stderr)
        n_comp = np.zeros(3)

    rgb_full = spectral_cube_to_srgb(cube, lam_lo, lam_hi)   # (H,W,3) float
    irr_full = cube.sum(axis=0) / (pixel_m ** 2)             # (H,W) W/m^2
    mask_f = mask.astype(np.float64)

    stride_h = max(1, -(-H // max_cells))     # ceil division
    stride_w = max(1, -(-W // max_cells))
    rgb = _decimate_block_mean(rgb_full, stride_h, stride_w)
    irr = _decimate_block_mean(irr_full, stride_h, stride_w)
    mfrac = _decimate_block_mean(mask_f, stride_h, stride_w)

    Hc, Wc = irr.shape
    cell_pixel_m = pixel_m * max(stride_h, stride_w)

    # corner grid: (Hc+1, Wc+1) points, in (x,y) detector-plane coordinates
    xs = x_lo + np.arange(Wc + 1) * (pixel_m * stride_w)
    ys = y_lo + np.arange(Hc + 1) * (pixel_m * stride_h)
    gx, gy = np.meshgrid(xs, ys)               # (Hc+1, Wc+1)
    corners = (gx[..., None] * xhat + gy[..., None] * yhat + n_comp)
    points = corners.reshape(-1, 3)

    # quad connectivity: corner (i,j) index = i*(Wc+1) + j
    def cidx(i, j):
        return i * (Wc + 1) + j

    ii, jj = np.meshgrid(np.arange(Hc), np.arange(Wc), indexing="ij")
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)
    quads = np.stack([cidx(ii, jj), cidx(ii, jj + 1),
                      cidx(ii + 1, jj + 1), cidx(ii + 1, jj)], axis=1)
    n_quads = quads.shape[0]
    connectivity = quads.reshape(-1).astype(np.int32)
    offsets = (np.arange(1, n_quads + 1, dtype=np.int32) * 4)

    rgb_u8 = np.clip(rgb.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)
    irr_flat = irr.reshape(-1).astype(np.float32)
    mask_flat = (mfrac.reshape(-1) > 0.5)
    # cells entirely outside the detector's active trim area are dimmed to
    # a dark neutral gray so the quad grid still reads as "screen shaped"
    rgb_u8 = np.where(mask_flat[:, None], rgb_u8, np.uint8(20))

    cell_data = [("rgb", rgb_u8, 3), ("irradiance_W_m2", irr_flat, 1)]
    out_path = _write_vtp(path, points, polys=(connectivity, offsets),
                          cell_data=cell_data)

    # Orientation sidecar (plain JSON, no vtk/paraview involved): make_viz.py
    # needs xhat/yhat/normal to frame a precise face-on camera for
    # detector_closeup, but a TILTED detector's plain axis-aligned bounding
    # box (what pvpython sees from the .vtp alone) does NOT reveal which
    # world axis is "thin" the way it would for an axis-aligned detector --
    # for a plane whose normal isn't aligned with x/y/z, all three bbox
    # extents can be comparably nonzero. Writing the actual basis here lets
    # make_viz.py build an exact face-on camera without guessing from bounds.
    meta_path = Path(str(path)[:-4] + ".json") if str(path).endswith(".vtp") \
        else Path(str(path) + ".json")
    center_world = points.mean(axis=0) if len(points) else n_comp
    with open(meta_path, "w") as fh:
        json.dump({"label": label, "normal": normal.tolist(),
                  "xhat": xhat.tolist(), "yhat": yhat.tolist(),
                  "center": center_world.tolist()}, fh, indent=1)
    return out_path


# ---------------------------------------------------------------------------
# CLI: rays.npy + detectors/*.h5 -> viz/*.vtp (run under OPTICS_PYTHON;
# make_viz.py shells out to this so pvpython itself never imports numpy/h5py
# via the raytracer package)
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m raytracer.vtkexport",
        description="Convert a run_trace.py case dir's rays.npy + "
                    "detectors/*.h5 into ParaView-ready .vtp files.")
    p.add_argument("--case-dir", required=True)
    p.add_argument("--model-json", default=None,
                   help="geometry/<model>/model.json (needed to place "
                        "detector quads correctly in world space)")
    p.add_argument("--out-dir", default=None,
                   help="default: <case-dir>/viz")
    p.add_argument("--max-cells", type=int, default=512)
    p.add_argument("--dim-rays", default="off",
                   choices=["off", "linear", "sqrt"],
                   help="bake an rgba cell array into rays.vtp with "
                        "alpha from rel_power (attenuation dimming for "
                        "make_viz --dim-rays)")
    p.add_argument("--dim-rays-floor", type=float, default=0.0,
                   metavar="PCT",
                   help="minimum opacity percent for --dim-rays")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    case_dir = Path(args.case_dir)
    out_dir = Path(args.out_dir) if args.out_dir else case_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if args.model_json:
        with open(args.model_json) as fh:
            model = json.load(fh)

    rays_path = case_dir / "rays.npy"
    if rays_path.exists():
        rays = np.load(rays_path)
        out_path = out_dir / "rays.vtp"
        write_vtp_polylines(out_path, rays, dim_mode=args.dim_rays,
                            dim_floor=args.dim_rays_floor)
        print("[vtkexport] wrote %s (%d ray segments)" % (out_path, len(rays)))
    else:
        print("[vtkexport] no rays.npy at %s, skipping" % rays_path)

    det_dir = case_dir / "detectors"
    n_det = 0
    if det_dir.is_dir():
        for h5path in sorted(det_dir.glob("*.h5")):
            import h5py
            with h5py.File(h5path, "r") as h:
                label = h.attrs["label"]
            safe = label.replace(".", "_")
            out_path = out_dir / ("det_%s.vtp" % safe)
            write_detector_quads(out_path, h5path, model=model,
                                 max_cells=args.max_cells)
            print("[vtkexport] wrote %s" % out_path)
            n_det += 1
    else:
        print("[vtkexport] no detectors/ dir at %s, skipping" % det_dir)

    print("[vtkexport] done: rays=%s detectors=%d -> %s"
          % (rays_path.exists(), n_det, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
