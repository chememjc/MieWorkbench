"""viz_configs.py -- declarative view registry for the optical ray tracer's
ParaView batch visualization suite (see ``make_viz.py`` / ``viz_common.py``).

Mirrors the antenna project's ``viz_configs.py`` driver/library/declarative
split, rewritten around the FreeCAD -> ray tracer -> ParaView pipeline:

  * body geometry: geometry/<model>/faces/<face_id>.stl, ALREADY in SI
    metres (extract_geometry.py scales a copy of the face to metres before
    tessellating) -- unlike the antenna project's mm-authored STLs, there
    is NO unit conversion needed here; body meshes, rays.npy and the
    detector quads all share the same metre frame.
  * rays: results/<model>/<case>/viz/rays.vtp (produced by
    scripts/raytracer/vtkexport.py from rays.npy), CELL data 'rgb'
    (uint8x3, wavelength color), 'power' (float32 W), 'source_id' (int16),
    'pol_mode' (uint8, 0=ordinary/isotropic, 1=extraordinary o/e-split ray
    -- absent/all-zero for a rays.npy predating birefringence support),
    'rel_power' (float32, power/birth_power in [0,1] -- absent for a
    rays.npy predating attenuation dimming; drives --dim-rays).
  * detectors: results/<model>/<case>/viz/det_<safe-label>.vtp, a
    decimated (<=512x512) colored quad grid positioned in world space,
    CELL data 'rgb' and 'irradiance_W_m2'.

This module is a PURE DECLARATIVE LIBRARY: no ``paraview`` imports, no I/O
beyond plain dict/function definitions. Safe to import from anywhere.

HOW TO ADD A NEW VIEW
----------------------
1. Add one dict to ``VIEWS`` below.
2. Add one builder function ``build_<name>(...)`` in ``make_viz.py`` and
   register it in that file's ``BUILDERS`` dict under the same string key
   used in step 1's ``"builder"`` field.
"""

# ---------------------------------------------------------------------------
# Body-role -> appearance. All body STL faces render as translucent
# "glass-like" surfaces per the task spec (0.25 opacity for every role);
# a mild per-role tint keeps sources/detectors/optics visually distinct
# without breaking the "glass" look.
# ---------------------------------------------------------------------------
BODY_COLORS = {
    "optic":    {"color": (0.65, 0.80, 0.95), "opacity": 0.25},  # pale blue glass
    "source":   {"color": (0.95, 0.45, 0.35), "opacity": 0.25},  # warm red glass
    "detector": {"color": (0.95, 0.75, 0.30), "opacity": 0.25},  # amber glass
}
DEFAULT_BODY_COLOR = {"color": (0.7, 0.7, 0.7), "opacity": 0.25}

DEFAULT_RESOLUTION = (1920, 1080)
DETECTOR_CLOSEUP_RESOLUTION = (2048, 2048)
SMOKE_RESOLUTION = (800, 600)
SMOKE_VIEW_NAME = "overview3d"

TURNTABLE_N_FRAMES = 8
TURNTABLE_ELEVATION_DEG = 25.0

# ---------------------------------------------------------------------------
# View registry
# ---------------------------------------------------------------------------
# kind:
#   'static'        -- rendered once, one PNG
#   'per_detector'  -- one PNG per results/<case>/viz/det_*.vtp file found
#   'multi_frame'   -- N PNGs (view_cfg['n_frames']), camera-only variation

VIEWS = [
    {
        "name": "overview3d",
        "builder": "build_overview3d",
        "kind": "static",
        "description": "All body STLs (translucent glass, colored by role), "
                        "rays.vtp colored by wavelength, every det_*.vtp "
                        "colored by its own sRGB cell data. Three-quarter "
                        "perspective camera framing the whole scene. "
                        "overview3d.png.",
    },
    {
        "name": "top",
        "builder": "build_top",
        "kind": "static",
        "description": "Same content as overview3d, parallel-projection "
                        "camera looking straight down -z. top.png.",
    },
    {
        "name": "side",
        "builder": "build_side",
        "kind": "static",
        "description": "Same content as overview3d, parallel-projection "
                        "camera looking down -y. side.png.",
    },
    {
        "name": "detector_closeup",
        "builder": "build_detector_closeup",
        "kind": "per_detector",
        "description": "For each det_*.vtp: rays + bodies for context, "
                        "camera zoomed face-on to just that detector's "
                        "bounding box. detector_closeup_<label>.png.",
    },
    {
        "name": "turntable",
        "builder": "build_turntable",
        "kind": "multi_frame",
        "n_frames": TURNTABLE_N_FRAMES,
        "description": "The overview3d scene rendered from %d evenly "
                        "spaced azimuths around z at a fixed elevation "
                        "(camera-only variation, scene built once). "
                        "turntable_frame<i>.png." % TURNTABLE_N_FRAMES,
    },
    {
        "name": "rays_polmode",
        "builder": "build_rays_polmode",
        "kind": "static",
        "description": "Same scene as overview3d, but rays.vtp colored by "
                        "its 'pol_mode' CELL array (0=ordinary/isotropic, "
                        "1=extraordinary -- birefringent o/e split) via a "
                        "ParaView lookup table instead of the wavelength "
                        "'rgb' array used everywhere else; skipped (with a "
                        "warning) if rays.vtp predates the pol_mode array "
                        "(stale viz/ + --skip-vtkexport). rays_polmode.png.",
    },
]

VIEWS_BY_NAME = {v["name"]: v for v in VIEWS}


# ---------------------------------------------------------------------------
# Pure helpers (no paraview import needed -- safe to unit test standalone)
# ---------------------------------------------------------------------------
def select_views(names=None):
    """Return the ordered list of view-config dicts named in ``names`` (a
    list of strings), or every registered view if ``names`` is None.
    Unknown names are silently skipped (caller may warn)."""
    if names is None:
        return list(VIEWS)
    selected = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        v = VIEWS_BY_NAME.get(name)
        if v is not None:
            selected.append(v)
    return selected


def body_appearance(role):
    """{'color': (r,g,b) in 0..1, 'opacity': float} for a body role string
    ('optic'/'source'/'detector'; anything else -> DEFAULT_BODY_COLOR)."""
    return BODY_COLORS.get(role, DEFAULT_BODY_COLOR)


def safe_label(label):
    """Match run_trace.py's save_detectors() filename sanitization exactly
    ('.' -> '_') so det_<safe_label>.vtp round-trips to detector labels."""
    return label.replace(".", "_")
