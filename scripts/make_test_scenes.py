# =============================================================================
# make_test_scenes.py — author validation FCStd scenes programmatically.
#
# Interpreter: the FreeCAD AppImage:
#   "$MIEWB_FREECAD" -c scripts/make_test_scenes.py -- \
#       [--outdir .] [--scene doubleslit] < /dev/null
#
# --scene <name> builds one scene; --scene all builds every scene in SCENES.
# Output <name>.FCStd in --outdir (default the repo root, matching the
# original doubleslit behavior; the WS-I scene batch is authored into
# basemodels/ by passing --outdir basemodels).
#
# The module-level SCENES dict is importable WITHOUT FreeCAD (the FreeCAD
# imports are guarded) so tests can read each scene's physical parameters and
# expected values as plain metadata:
#     import make_test_scenes; make_test_scenes.SCENES["lens_pcx"]
#
# All the usual FreeCAD -c caveats apply (double execution: all writes
# idempotent; no __main__ guard; os._exit; log via FreeCAD.Console).
#
# ---------------------------------------------------------------------------
# GEOMETRY CONVENTIONS (all scenes)
# ---------------------------------------------------------------------------
# * Optical axis is global +x. Sources sit at negative x and emit toward the
#   origin (+x); detectors sit at positive x (or off-axis for reflected arms).
# * A spherical refracting surface is described by a SIGNED radius R using the
#   optical convention "R>0 iff the centre of curvature is on the +x side of
#   the vertex" (light travels +x). A plano surface has R=None.
# * Lenses are built by revolving a meridian profile 360 deg about the global
#   x-axis (H_Axis of a sketch on the body's XZ_Plane). A circular-arc meridian
#   revolves to a native OCC Sphere; a straight radial meridian revolves to a
#   Plane (a flat lens face); a straight slanted meridian revolves to a Cone
#   (axicon / Fresnel facet); a constant-radius meridian revolves to a Cylinder
#   (the lens rim). This is exactly what the example.FCStd lens does, and what
#   extract_geometry.classify_surface / canonicalize_revolution recover.
# * Cylinder lenses / prisms / cubes are PADDED sketches (a circular-arc edge
#   padded along z gives a native Cylinder face; a straight polygon padded
#   gives Plane faces).
# =============================================================================
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (stdlib-only, always importable)

# FreeCAD is only importable under the AppImage's embedded python; guard it so
# `import make_test_scenes` works under plain python for SCENES metadata.
try:
    import FreeCAD as App
    import Part
    import Sketcher  # noqa: F401  (registers sketch object types)
    _HAVE_FREECAD = True
    App.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
        "CreateBackupFiles", False)
except Exception:            # pragma: no cover - metadata-only import path
    App = None
    Part = None
    _HAVE_FREECAD = False


def log(msg):
    if _HAVE_FREECAD:
        App.Console.PrintMessage(msg + "\n")
    print(msg, flush=True)


# =============================================================================
# SCENES metadata (importable without FreeCAD) — physical parameters + the
# expected values tests assert on. Lengths mm, angles deg unless noted.
# Numeric expecteds computed from BK7/SF5/quartz/calcite dispersion at the
# stated wavelengths (see the module docstring / commit notes for formulas).
# =============================================================================
SCENES = {
    "lens_pcx": {
        "description": "plano-convex BK7, convex (R=25) toward source, flat "
                       "toward detector; collimated 633nm Ø10 beam.",
        "material": "bk7", "lambda_nm": 633.0,
        "R1_mm": 25.0, "R2_mm": None, "thickness_mm": 5.0,
        "aperture_mm": 20.0, "beam_dia_mm": 10.0,
        "n_633": 1.51508, "expected_efl_mm": 48.536, "expected_bfl_mm": 45.236,
        "detector_x_mm": 65.0,
        "faces": {"sphere": 1, "plane": 1, "cylinder": 1},
    },
    "lens_dcx": {
        "description": "symmetric biconvex BK7, R=+/-40, collimated 633nm.",
        "material": "bk7", "lambda_nm": 633.0,
        "R1_mm": 40.0, "R2_mm": -40.0, "thickness_mm": 6.0,
        "aperture_mm": 20.0, "beam_dia_mm": 10.0,
        "n_633": 1.51508, "expected_efl_mm": 39.845, "detector_x_mm": 55.0,
        "faces": {"sphere": 2, "cylinder": 1},
    },
    "lens_pcv": {
        "description": "plano-concave BK7 (flat front, concave R=+25 back); "
                       "diverging, expands the collimated 633nm beam.",
        "material": "bk7", "lambda_nm": 633.0,
        "R1_mm": None, "R2_mm": 25.0, "thickness_mm": 3.0,
        "aperture_mm": 20.0, "beam_dia_mm": 10.0,
        "n_633": 1.51508, "expected_efl_mm": -48.536,
        "note": "virtual focus 48.5mm in front of the lens (diverging).",
        "detector_x_mm": 60.0,
        "faces": {"sphere": 1, "plane": 1, "cylinder": 1},
    },
    "lens_dcv": {
        "description": "symmetric biconcave BK7, R=-40/+40, diverging 633nm.",
        "material": "bk7", "lambda_nm": 633.0,
        "R1_mm": -40.0, "R2_mm": 40.0, "thickness_mm": 3.0,
        "aperture_mm": 20.0, "beam_dia_mm": 10.0,
        "n_633": 1.51508, "expected_efl_mm": -38.340, "detector_x_mm": 60.0,
        "faces": {"sphere": 2, "cylinder": 1},
    },
    "lens_achromat": {
        "description": "cemented crown(BK7)+flint(SF5) doublet, f~50mm "
                       "achromatized at F(486.1)/C(656.3); 5um air gap at the "
                       "interface to avoid a coincident-face seam.",
        "crown_material": "bk7", "flint_material": "sf5",
        "design_wavelengths_nm": [486.1, 656.3], "d_line_nm": 587.6,
        "R1_mm": 31.0, "R2_interface_mm": -21.956, "R3_mm": -64.497,
        "crown_thickness_mm": 6.0, "flint_thickness_mm": 3.0,
        "air_gap_mm": 0.005, "aperture_mm": 18.0, "beam_dia_mm": 10.0,
        "V_bk7": 64.14, "V_sf5": 32.24,
        "expected_efl_mm": 50.0, "detector_x_mm": 72.0,
        "faces_crown": {"sphere": 2, "cylinder": 1},
        "faces_flint": {"sphere": 2, "cylinder": 1},
    },
    "lens_sphere_control": {
        "description": "plano-convex spherical BK7, f~40mm at 633nm, Ø20 "
                       "aperture; spherical control for lens_asphere RMS-spot "
                       "comparison (same f, same aperture, k=0).",
        "material": "bk7", "lambda_nm": 633.0,
        "R1_mm": 20.6033, "R2_mm": None, "thickness_mm": 6.0,
        "aperture_mm": 20.0, "beam_dia_mm": 18.0,
        "n_633": 1.51508, "expected_efl_mm": 40.0, "conic_k": 0.0,
        "detector_x_mm": 46.0,
        "faces": {"sphere": 1, "plane": 1, "cylinder": 1},
    },
    "lens_asphere": {
        "description": "plano-convex BK7 asphere, f~40mm at 633nm, convex side "
                       "toward source, flat toward the focus. The front "
                       "conic+A4 profile is solved for the COMPLETE lens (front "
                       "asphere + flat exit) — not just a front-surface "
                       "stigmatic conic — minimizing best-focus RMS spot for "
                       "the collimated on-axis 18mm beam; k=-2.29547 (=-n^2) "
                       "corrected only the front surface and the flat exit "
                       "re-added spherical aberration (over-corrected ~3x worse "
                       "than the sphere control). Solved offline by exact "
                       "meridional ray trace + Nelder-Mead on (k, A4); reaches "
                       "~1um RMS (~83x better than the sphere control). Authored "
                       "via a revolved BSpline through exact sag samples + a "
                       "matching surface_override (verified <1um by the "
                       "extractor).",
        "material": "bk7", "lambda_nm": 633.0,
        "R1_mm": 20.6033, "R2_mm": None, "thickness_mm": 6.0,
        "aperture_mm": 20.0, "beam_dia_mm": 18.0,
        "n_633": 1.51508, "expected_efl_mm": 40.0,
        "conic_k": -1.0, "asphere_A4_mm": 6.586562e-06,   # A4 units mm^-3
        "detector_x_mm": 46.0,
        "faces": {"asphere": 1, "plane": 1, "cylinder": 1},
    },
    "lens_cyl_pos": {
        "description": "plano-convex cylinder lens (cylinder axis along z, "
                       "R=25), line focus in x-y; collimated 633nm.",
        "material": "bk7", "lambda_nm": 633.0,
        "R_mm": 25.0, "thickness_mm": 5.0, "aperture_mm": 20.0,
        "height_z_mm": 20.0, "beam_dia_mm": 10.0,
        "n_633": 1.51508, "expected_efl_mm": 48.536, "detector_x_mm": 60.0,
        "faces": {"cylinder": 1, "plane": 5},
    },
    "lens_cyl_neg": {
        "description": "plano-concave cylinder lens (axis along z, R=-25), "
                       "diverging line; collimated 633nm.",
        "material": "bk7", "lambda_nm": 633.0,
        "R_mm": -25.0, "thickness_mm": 4.0, "aperture_mm": 20.0,
        "height_z_mm": 20.0, "beam_dia_mm": 10.0,
        "n_633": 1.51508, "expected_efl_mm": -48.536, "detector_x_mm": 60.0,
        "faces": {"cylinder": 1, "plane": 5},
    },
    "axicon_pcx": {
        "description": "plano-convex BK7 axicon, base angle 10deg, Ø22 "
                       "aperture; conical front (apex toward source), flat base "
                       "toward detector; collimated 633nm Ø10 beam.",
        "material": "bk7", "lambda_nm": 633.0,
        "base_angle_deg": 10.0, "cone_semiangle_from_axis_deg": 80.0,
        "aperture_mm": 22.0, "beam_dia_mm": 10.0, "n_633": 1.51508,
        "detector_z_mm": 30.0,
        "expected_ring_radius_mm": 2.7043,   # z*tan((n-1)*alpha), z=30
        "note": "ring radius = z*tan((n-1)*alpha); z=30mm from the flat base.",
        "detector_x_mm": 30.0,
        "faces": {"cone": 1, "plane": 1},
    },
    "lens_ball": {
        "description": "8mm BK7 ball lens (full sphere); collimated 587.6nm "
                       "Ø4 beam.",
        "material": "bk7", "lambda_nm": 587.6,
        "diameter_mm": 8.0, "beam_dia_mm": 4.0, "n_d": 1.51680,
        "expected_bfl_mm": 1.870, "expected_efl_mm": 5.870,
        "note": "BFL = R(2-n)/(2(n-1)) from the rear surface; R=4mm.",
        "detector_x_mm": 12.0,
        "faces": {"sphere": 1},
    },
    "lens_rod": {
        "description": "8mm-dia BK7 rod (cylinder, axis along z) acting as a "
                       "cylinder lens in the x-y plane; collimated 587.6nm.",
        "material": "bk7", "lambda_nm": 587.6,
        "diameter_mm": 8.0, "length_z_mm": 20.0, "beam_dia_mm": 4.0,
        "n_d": 1.51680, "detector_x_mm": 20.0,
        "faces": {"cylinder": 1, "plane": 2},
    },
    "lens_fresnel": {
        "description": "collapsed plano-convex Fresnel lens, f~50mm at 633nm, "
                       "8 annular conical facets (each facet slope = local "
                       "slope of the ideal PCX at that zone); flat back toward "
                       "detector; collimated 633nm.",
        "material": "bk7", "lambda_nm": 633.0,
        "expected_efl_mm": 50.0, "n_facets": 8, "aperture_mm": 20.0,
        "facet_height_mm": 1.0, "beam_dia_mm": 18.0, "n_633": 1.51508,
        "detector_x_mm": 55.0,
        "faces_note": ">=8 cone faces (one per facet) + flat back + rim.",
        "min_cone_faces": 8,
    },
    "prism_equilateral": {
        "description": "60deg equilateral BK7 prism at minimum deviation for "
                       "550nm; broadband incoherent source 420-680nm; detector "
                       "off-axis, face-on to the dispersed fan.",
        "material": "bk7", "apex_deg": 60.0,
        "lambdac_nm": 550.0, "lambdamin_nm": 420.0, "lambdamax_nm": 680.0,
        "n_550": 1.51852, "min_deviation_deg": 38.798,
        "entrance_aoi_deg": 49.399, "side_mm": 20.0, "height_z_mm": 20.0,
        # For min-deviation the +x beam must strike the entrance (left) face at
        # AOI=(A+dmin)/2=49.399deg. The unrotated left face presents 30deg AOI
        # to +x, so the prism is rotated by theta about z with normal angle
        # 150+theta; -cos(150+theta)=cos(49.399) => theta=-19.399deg (the old
        # +19.4 put the beam at ~10deg AOI, TIR at the exit). Solved offline by
        # a 2D polygon-prism ray trace (brentq on AOI).
        "prism_rotation_deg": -19.3991,
        # Detector: face-on to the 550nm exit ray (exit dir (0.7794,-0.6266),
        # deviation 38.798deg toward -y), centered 25mm downstream of the exit
        # vertex. Its front broad face is the closest detector face to the world
        # origin for ANY in-plane spin (33.37mm vs >=33.87mm for every edge
        # face), so the extractor's closest-centroid auto-pick lands on the
        # screen, not an edge (which is why the old rotated plate caught 0 W).
        "detector_center_mm": [26.3987, -21.2226, 0.0],
        "detector_normal": [-0.779387, 0.626542, 0.0],
        "detector_half_mm": 10.0,
        "faces": {"plane": 5},
    },
    "pol_linear": {
        "description": "Thorlabs LPVISE100-A film polarizer (PMMA substrate "
                       "2mm) with body-local polarizer_axis '0,0,1'; linear:30 "
                       "source. Spreadsheet alias 'polangle' rotates the "
                       "polarizer body about x (default 0 -> axis stays +z) so "
                       "the Malus angle is permutable.",
        "substrate_material": "pmma", "polarizer": "thorlabs_lpvise100a",
        "polarizer_axis_local": "0,0,1", "source_polarization": "linear:30",
        "lambda_nm": 550.0, "substrate_mm": 2.0, "polangle_deg": 0.0,
        "expected_malus": 0.75,  # cos^2(30) at polangle=0
        "note": "Malus factor cos^2(30-polangle); Fresnel at the PMMA faces "
                "and the film ER curve modify absolute transmission.",
        "detector_x_mm": 40.0,
        "faces": {"plane": 6},
    },
    "pol_crossed": {
        "description": "two ideal_linear polarizers, axes +z then +y (crossed) "
                       "on near-index air substrates; unpolarized 550nm source. "
                       "Transmission ~= 0.5 * (ideal_linear leakage)^2.",
        "polarizer": "ideal_linear", "substrate_material": "air",
        "axis1_local": "0,0,1", "axis2_local": "0,1,0",
        "lambda_nm": 550.0, "expected_transmission": 5e-7,
        "note": "ideal 0.5*ER^-2; ideal_linear.csv leakage ~1e-6 -> ~0.5e-6.",
        "detector_x_mm": 40.0,
        "faces_each": {"plane": 6},
    },
    "pol_circular": {
        "description": "Thorlabs CP1L532 left-handed circular polarizer (BK7 "
                       "substrate 2mm); source circular:right (blocked). The "
                       "e2e test flips the source to circular:left in-memory to "
                       "check the pass state.",
        "substrate_material": "bk7", "polarizer": "thorlabs_cp1l532",
        "source_polarization": "circular:right", "lambda_nm": 532.0,
        "substrate_mm": 2.0, "detector_x_mm": 40.0,
        "note": "left-handed CP passes circular:left, blocks circular:right.",
        "faces": {"plane": 6},
    },
    "waveplate_quartz": {
        "description": "multi-order (m=30) half-wave quartz plate at 589nm "
                       "between crossed ideal_linear polarizers at +/-45deg; "
                       "crystal_axis '0,0,1'. HWP rotates the +45 input to -45, "
                       "which the -45 analyzer passes: sin^2(delta/2)~=1.",
        "material": "quartz", "crystal_axis_local": "0,0,1",
        "lambda_nm": 589.0, "order_m": 30,
        "n_o_589": 1.54422, "n_e_589": 1.55332, "delta_n": 0.009100,
        "thickness_mm": 1.9740, "retardance_waves": 30.5,
        "pol_in_axis_local": "0,0.70711,0.70711",   # +45 in the y-z plane
        "pol_out_axis_local": "0,-0.70711,0.70711",  # -45, crossed vs pol_in
        "substrate_material": "air",
        "expected_transmission_factor": 1.0,  # sin^2(delta/2), delta=pi
        "detector_x_mm": 45.0,
        "faces_plate": {"plane": 6},
    },
    "waveplate_mgf2": {
        "description": "multi-order (m=30) half-wave MgF2 plate at 589nm "
                       "between crossed ideal_linear polarizers at +/-45deg; "
                       "crystal_axis '0,0,1'. Same physics as "
                       "waveplate_quartz but MgF2 carries NO gyration data, "
                       "so the scene stays C-engine routable (quartz is "
                       "gyrotropic and reference-routes by design) — this is "
                       "the C-parity uniaxial-retardance coverage scene.",
        "material": "mgf2", "crystal_axis_local": "0,0,1",
        "lambda_nm": 589.0, "order_m": 30,
        "n_o_589": 1.37772, "n_e_589": 1.38953, "delta_n": 0.011812,
        "thickness_mm": 1.5209, "retardance_waves": 30.5,
        "pol_in_axis_local": "0,0.70711,0.70711",   # +45 in the y-z plane
        "pol_out_axis_local": "0,-0.70711,0.70711",  # -45, crossed vs pol_in
        "substrate_material": "air",
        "expected_transmission_factor": 1.0,  # sin^2(delta/2), delta=pi
        "detector_x_mm": 45.0,
        "faces_plate": {"plane": 6},
    },
    "pbs_cube": {
        "description": "20mm BK7 polarizing-beamsplitter cube: two 45deg "
                       "right-angle prisms, first prism's hypotenuse coated "
                       "pbs_visible_45, 5um air gap (cemented interface modeled "
                       "as a thin gap); unpolarized 550nm source; TWO detectors "
                       "(transmitted +x arm, reflected -y arm).",
        "material": "bk7", "coating": "pbs_visible_45", "cube_mm": 20.0,
        "air_gap_mm": 0.005, "lambda_nm": 550.0, "height_z_mm": 20.0,
        "det_trans_x_mm": 40.0, "det_refl_y_mm": -25.0,
        "note": "transmitted arm ~p-pol, reflected arm ~s-pol.",
        "faces_each_prism": {"plane": 5},
    },
    "calcite_displacer": {
        "description": "10mm calcite slab, crystal_axis 45deg in the x-z plane "
                       "('0.70711,0,0.70711'); unpolarized 590nm Ø0.5 beam; "
                       "detector 1mm behind -> two spots ~1.09mm apart "
                       "(o- and e-ray walk-off).",
        "material": "calcite", "crystal_axis_local": "0.70711,0,0.70711",
        "lambda_nm": 590.0, "thickness_mm": 10.0, "beam_dia_mm": 0.5,
        "n_o_590": 1.65830, "n_e_590": 1.48611,
        "walkoff_deg": 6.232, "expected_displacement_mm": 1.0919,
        "detector_x_mm": 11.0,
        "faces": {"plane": 6},
    },
    "wollaston": {
        "description": "Wollaston prism: two 30deg calcite wedges with "
                       "orthogonal optic axes ('0,0,1' and '0,1,0'), 5um gap; "
                       "590nm unpolarized narrow beam; detector 50mm "
                       "downstream. Symmetric split ~2(n_o-n_e)tan(30).",
        "material": "calcite", "wedge_angle_deg": 30.0, "air_gap_mm": 0.005,
        "axis1_local": "0,0,1", "axis2_local": "0,1,0",
        "lambda_nm": 590.0, "beam_dia_mm": 0.5, "height_z_mm": 12.0,
        "n_o_590": 1.65830, "n_e_590": 1.48611,
        "expected_split_rad": 0.19883, "expected_split_deg": 11.392,
        "detector_x_mm": 50.0,
        "faces_each_wedge": {"plane": 5},
    },
    "filter_bandpass": {
        "description": "bp_550_40 bandpass filter on a BK7 3.5mm slab; "
                       "broadband incoherent 450-650nm source; detector "
                       "straight through.",
        "material": "bk7", "filter": "bp_550_40", "substrate_mm": 3.5,
        "lambdac_nm": 550.0, "lambdamin_nm": 450.0, "lambdamax_nm": 650.0,
        "detector_x_mm": 40.0,
        "faces": {"plane": 6},
    },
    "hot_mirror": {
        "description": "BK7 plate at 45deg with hot_mirror_45 coating; "
                       "broadband 450-1000nm source; TWO detectors (visible "
                       "pass +x arm, IR reflect +y arm).",
        "material": "bk7", "coating": "hot_mirror_45", "plate_thickness_mm": 3.0,
        "plate_size_mm": 30.0, "plate_rotation_deg": 45.0,
        "lambdac_nm": 700.0, "lambdamin_nm": 450.0, "lambdamax_nm": 1000.0,
        "det_pass_x_mm": 40.0, "det_refl_y_mm": 30.0,
        "faces": {"plane": 6},
    },
    "mesh_freeform": {
        "description": "deliberately non-analytic BK7 optic (prolate ellipsoid "
                       "of revolution via a revolved BSpline meridian) that "
                       "falls back to a 'mesh' surface; used by the all-flags "
                       "integration test. Extractor WARNS (mesh fallback) — do "
                       "NOT use --strict.",
        "material": "bk7", "lambda_nm": 633.0,
        "semi_major_x_mm": 8.0, "semi_minor_r_mm": 5.0, "beam_dia_mm": 6.0,
        "detector_x_mm": 40.0, "expects_mesh_fallback": True,
        "faces": {"mesh": 1},
    },

    # ---------------------------------------------------------------------
    # Phase-12 new-physics scenes (biaxial / Gaussian / ghost / scatter /
    # curved detector). "As simple as possible but physically real": solids
    # with real materials, sources/detectors per the §5 body-tagging
    # contract, energy closes <1e-3 in every one.
    # ---------------------------------------------------------------------
    "ktp_walkoff": {
        "description": "15mm KTP biaxial plate, X principal axis at 45deg in "
                       "the global x-z plane ('0.70711,0,0.70711'), Y "
                       "principal = global y ('0,1,0'); 633nm unpolarized "
                       "narrow beam propagates in the X-Z principal plane "
                       "(maximum walk-off). The y-polarized sheet goes "
                       "straight (n=n_y); the in-plane sheet walks off in "
                       "global z -> two spots.",
        "material": "ktp",
        "crystal_axis_local": "0.70711,0,0.70711",
        "crystal_axis2_local": "0,1,0",
        "lambda_nm": 633.0, "thickness_mm": 15.0, "beam_dia_mm": 0.3,
        "walkoff_axis": "z",
        "note": "two-spot separation is the solver-predicted in-plane-sheet "
                "transverse displacement (biaxial_ray_from_k); see "
                "test_biaxial._expected_walkoff_dz.",
        "detector_x_mm": 20.0,
        "faces": {"plane": 6},
    },
    "gaussian_bench": {
        "description": "Gaussian-beam source (beam_waist 50um at the emitting "
                       "face, M2=1.0) propagating 62mm (=5 Rayleigh ranges) "
                       "through empty air to a screen where the beam has "
                       "expanded ~5x. Incoherent direct-deposit beam mode "
                       "(coherent=False).",
        "lambda_nm": 633.0, "beam_waist_mm": 0.05, "m2": 1.0,
        "source_x_mm": -2.0, "source_aperture_mm": 2.0,
        "detector_x_mm": 60.0, "det_half_mm": 2.0,
        "note": "z from waist = detector_x - source_x = 62mm; "
                "w(z)=w0*sqrt(1+(z/zR)^2), zR=pi*w0^2/lambda ~= 12.4mm.",
        "faces": {"plane": 1},
    },
    "ghost_doublet": {
        "description": "Two uncoated N-BK7 flat windows (4mm thick, 8mm air "
                       "gap) in a collimated 633nm incoherent beam; a screen "
                       "downstream records the direct beam plus the natural "
                       "Fresnel ghosts. The dominant 2-bounce ghost carries "
                       "direct_power * R^2 (R = normal-incidence air/BK7 "
                       "Fresnel reflectance).",
        "material": "bk7", "lambda_nm": 633.0,
        "window_thickness_mm": 4.0, "gap_mm": 4.0, "window_size_mm": 20.0,
        "beam_dia_mm": 6.0,
        "g1_x0_mm": 0.0, "g2_x0_mm": 8.0, "detector_x_mm": 30.0,
        "note": "ghost oracle: strongest gen-2 path power == direct * R^2 "
                "(enumerated 2-reflection Fresnel product).",
        "faces_each": {"plane": 6},
    },
    "scatter_plate": {
        "description": "Flat BK7 window at 45deg with measured ABg scatter "
                       "(scatter=polished_bk7_glass) on its beam-facing "
                       "front face; collimated 633nm beam. The Fresnel "
                       "reflection folds to +y where a screen catches the "
                       "specular spot plus the diffuse scatter lobe.",
        "material": "bk7", "scatter": "polished_bk7_glass",
        "lambda_nm": 633.0, "plate_size_mm": 24.0, "plate_thickness_mm": 3.0,
        "plate_rotation_deg": -45.0, "beam_dia_mm": 6.0,
        "det_refl_y_mm": 30.0,
        "note": "scattered rays are flagged scattered=True at the detector; "
                "reflected-side specular + scatter split conserves R.",
        "faces": {"plane": 6},
    },
    "curved_focal": {
        "description": "Collimated 633nm Ø10 beam -> plano-convex BK7 lens "
                       "(R=25, f~48.5, flat toward focus) -> CONCAVE "
                       "cylindrical detector (material=detector, axis along "
                       "z) whose curved face hugs the focus at x~50mm. The "
                       "curved screen catches >90% of the focused power. "
                       "coherent=False (geometric focus).",
        "material": "bk7", "lambda_nm": 633.0,
        "R1_mm": 25.0, "R2_mm": None, "thickness_mm": 5.0,
        "aperture_mm": 20.0, "beam_dia_mm": 10.0, "n_633": 1.51508,
        "expected_efl_mm": 48.536, "expected_bfl_mm": 45.236,
        "det_center_of_curvature_x_mm": 30.0, "det_radius_mm": 20.0,
        "det_half_y_mm": 5.0, "det_height_z_mm": 10.0, "det_back_pad_mm": 3.0,
        "focus_x_mm": 50.236,
        "note": "concave-toward-beam cylinder: center of curvature in FRONT "
                "(x=30) so the vertex face at x=cx+R=50 sits on the focus; "
                "curved detector face auto-picks or is pinned via "
                "detector_face.",
        "faces": {"cylinder": 1, "plane": 3},
    },

    # ---------------------------------------------------------------------
    # Depth-4 nesting spike (cuvette-in-bath de-risk). Four concentric
    # axis-aligned cube shells on the +x beam axis, each strictly inside the
    # previous with a 2mm clearance (>>5um): outer vat glass wall > bath
    # liquid > inner cuvette glass wall > sample liquid. Beam travels
    # straight through the common centre at normal incidence on every face
    # -> 8 Fresnel interfaces + one Beer-Lambert absorbing chord (the
    # innermost body carries an nd_od01 bulk filter on top of material=water
    # so the test has a clean closed-form expected transmission).
    # ---------------------------------------------------------------------
    "nested4": {
        "description": "depth-4 concentric nesting: 20mm BK7 outer vat wall "
                       "> 16mm water bath > 12mm BK7 cuvette wall > 8mm water "
                       "sample (nd_od01 bulk filter for a measurable Beer-"
                       "Lambert chord), each pair with a 2mm clearance on "
                       "every face (extractor classifies 6 pairwise "
                       "validation.nested_solids entries: outer/bath, "
                       "outer/inner_glass, outer/sample, bath/inner_glass, "
                       "bath/sample, inner_glass/sample). Collimated "
                       "unpolarized 550nm beam on-axis; straight through -> "
                       "8 normal-incidence Fresnel interfaces + one "
                       "Beer-Lambert chord through the sample body.",
        "lambda_nm": 550.0, "beam_dia_mm": 3.0,
        "outer_material": "bk7", "bath_material": "water",
        "inner_material": "bk7", "sample_material": "water",
        "sample_filter": "nd_od01",
        "outer_half_mm": 10.0, "bath_half_mm": 8.0,
        "inner_half_mm": 6.0, "sample_half_mm": 4.0,
        "clearance_mm": 2.0,
        "outer_x0_mm": 40.0, "outer_len_mm": 20.0,
        "bath_x0_mm": 42.0, "bath_len_mm": 16.0,
        "inner_x0_mm": 44.0, "inner_len_mm": 12.0,
        "sample_x0_mm": 46.0, "sample_len_mm": 8.0,
        "sample_chord_mm": 8.0,
        "source_x_mm": -20.0, "detector_x_mm": 90.0, "detector_half_mm": 20.0,
        "faces": {"plane": 6 * 4},
    },
    "auto_designed_lens": {
        "description": "optimizer demo: the lens_pcx singlet with its "
                       "axial position spreadsheet-driven (dim.lenspos "
                       "alias, expression-bound Placement.Base.x). The "
                       "detector sits 4mm PAST the lenspos=0 focal plane, "
                       "so the spot-minimizing lens position is lenspos="
                       "+4mm (collimated input: the focus translates 1:1 "
                       "with the lens). coherent=False (geometric focus).",
        "material": "bk7", "lambda_nm": 633.0,
        "R1_mm": 25.0, "R2_mm": None, "thickness_mm": 5.0,
        "aperture_mm": 20.0, "beam_dia_mm": 10.0,
        "n_633": 1.51508, "expected_efl_mm": 48.536,
        "expected_bfl_mm": 45.236,
        "focus_x_at_lenspos0_mm": 50.236,
        "detector_x_mm": 54.236,
        "expected_focus_lenspos_mm": 4.0,
        "faces": {"sphere": 1, "plane": 1, "cylinder": 1},
    },
    "tolerance_lens": {
        "description": "tolerancing demo: the auto_designed_lens singlet "
                       "with THREE spreadsheet-driven degrees of freedom "
                       "— dim.lenspos (lens axial position, "
                       "Placement.Base.x), dim.lensdy (lens transverse "
                       "decenter, Placement.Base.y) and dim.detpos "
                       "(detector axial position, Placement.Base.x). The "
                       "nominal design is IN FOCUS (detector at the "
                       "lenspos=0 focal plane), so lenspos errors defocus "
                       "1:1 (collimated input), lensdy errors mostly "
                       "TRANSLATE the spot (spot RMS is first-order "
                       "insensitive to decenter), and detpos is the "
                       "classic refocus compensator. coherent=False "
                       "(geometric spot).",
        "material": "bk7", "lambda_nm": 633.0,
        "R1_mm": 25.0, "R2_mm": None, "thickness_mm": 5.0,
        "aperture_mm": 20.0, "beam_dia_mm": 10.0,
        "n_633": 1.51508, "expected_efl_mm": 48.536,
        "expected_bfl_mm": 45.236,
        "focus_x_at_lenspos0_mm": 50.236,
        "faces": {"sphere": 1, "plane": 1, "cylinder": 1},
    },
}


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args():
    argv = sys.argv
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=str(common.PROJECT_DIR))
    p.add_argument("--scene", default="doubleslit")
    try:
        return p.parse_args(rest)
    except SystemExit as e:
        os._exit(int(e.code or 0))


# =============================================================================
# Low-level sketch/geometry helpers (FreeCAD-only; called only under -c)
# =============================================================================
def add_rect(sk, u0, v0, du, dv):
    """Closed 4-line rectangle in sketch-local coordinates."""
    V = App.Vector
    pts = [V(u0, v0, 0), V(u0 + du, v0, 0), V(u0 + du, v0 + dv, 0),
           V(u0, v0 + dv, 0)]
    for i in range(4):
        sk.addGeometry(Part.LineSegment(pts[i], pts[(i + 1) % 4]), False)


def add_circle(sk, u0, v0, r):
    sk.addGeometry(Part.Circle(App.Vector(u0, v0, 0),
                               App.Vector(0, 0, 1), r), False)


def set_props(body, props):
    """Attach group-'Base' custom properties, typed by python value."""
    for k, v in (props or {}).items():
        if isinstance(v, bool):
            body.addProperty("App::PropertyBool", k, "Base")
        elif isinstance(v, (int, float)):
            body.addProperty("App::PropertyFloat", k, "Base")
        else:
            body.addProperty("App::PropertyString", k, "Base")
        setattr(body, k, v)


def new_body_pad(doc, name, label, rects=None, circle=None,
                 x_start=0.0, length=1.0, props=None):
    """PartDesign Body with one sketch on its YZ plane (normal = +x),
    offset to x_start [mm], padded `length` mm along +x. Sketch local
    (u, v) = global (y, z)."""
    body = doc.addObject("PartDesign::Body", name)
    sk = body.newObject("Sketcher::SketchObject", name + "_sk")
    yz = [o for o in body.Origin.OriginFeatures
          if o.Name.startswith("YZ_Plane")][0]
    sk.AttachmentSupport = [(yz, "")]
    sk.MapMode = "FlatFace"
    sk.AttachmentOffset = App.Placement(App.Vector(0, 0, x_start),
                                        App.Rotation())
    for r in rects or []:
        add_rect(sk, *r)
    if circle is not None:
        add_circle(sk, *circle)
    pad = body.newObject("PartDesign::Pad", name + "_pad")
    pad.Profile = sk
    pad.Length = length
    sk.Visibility = False
    set_props(body, props)
    return body


_PLANE_PREFIX = {"YZ": "YZ_Plane", "XZ": "XZ_Plane", "XY": "XY_Plane"}


def _origin_plane(body, which):
    pref = _PLANE_PREFIX[which]
    return [o for o in body.Origin.OriginFeatures
            if o.Name.startswith(pref)][0]


def pad_body(doc, name, geoms, plane="XY", offset=0.0, length=1.0,
             props=None, placement=None):
    """Generic padded body: `geoms` are Part edge geometries (LineSegment/
    ArcOfCircle/Circle) in the sketch-local (u,v) frame of `plane`. The sketch
    is attached to that origin plane at `offset` along its normal, and padded
    `length` along the normal. `placement` (App.Placement) rotates/positions
    the whole body afterwards."""
    body = doc.addObject("PartDesign::Body", name)
    sk = body.newObject("Sketcher::SketchObject", name + "_sk")
    pl = _origin_plane(body, plane)
    sk.AttachmentSupport = [(pl, "")]
    sk.MapMode = "FlatFace"
    if offset:
        sk.AttachmentOffset = App.Placement(App.Vector(0, 0, offset),
                                            App.Rotation())
    for g in geoms:
        sk.addGeometry(g, False)
    pad = body.newObject("PartDesign::Pad", name + "_pad")
    pad.Profile = sk
    pad.Length = length
    sk.Visibility = False
    if placement is not None:
        body.Placement = placement
    set_props(body, props)
    return body


def revolve_body(doc, name, geoms, props=None, placement=None):
    """Revolve a meridian profile 360 deg about the global x-axis. `geoms` are
    Part edges in the (u=x, v=radial) meridian frame; the sketch lives on the
    body's XZ_Plane (local u=x, local v=z) and revolves about its H_Axis."""
    body = doc.addObject("PartDesign::Body", name)
    sk = body.newObject("Sketcher::SketchObject", name + "_sk")
    xz = _origin_plane(body, "XZ")
    sk.AttachmentSupport = [(xz, "")]
    sk.MapMode = "FlatFace"
    for g in geoms:
        sk.addGeometry(g, False)
    rev = body.newObject("PartDesign::Revolution", name + "_rev")
    rev.Profile = sk
    rev.ReferenceAxis = (sk, ["H_Axis"])
    rev.Angle = 360.0
    sk.Visibility = False
    if placement is not None:
        body.Placement = placement
    set_props(body, props)
    return body


# ---- meridian / arc math (used by lens builders) --------------------------
def _sign(x):
    return 1.0 if x >= 0.0 else -1.0


def surf_u(R, u_vertex, v):
    """u-coordinate of a spherical surface (signed radius R, vertex on the
    axis at u_vertex) at radial height v. R>0 -> centre on +u side."""
    cu = u_vertex + R
    return cu - _sign(R) * math.sqrt(R * R - v * v)


def _arc3(u1, v1, um, vm, u2, v2):
    """Arc of circle through three (u,v) points, in the sketch plane."""
    return Part.Arc(App.Vector(u1, v1, 0), App.Vector(um, vm, 0),
                    App.Vector(u2, v2, 0))


def _line(u1, v1, u2, v2):
    return Part.LineSegment(App.Vector(u1, v1, 0), App.Vector(u2, v2, 0))


def _surface_edge(R, u_vertex, sa, start_at_axis):
    """One refracting-surface meridian edge from the axis vertex (v=0) to the
    rim (v=sa), or reversed. R=None -> flat radial line. Returns (edge, u_rim)."""
    if R is None:
        u_rim = u_vertex
        if start_at_axis:
            return _line(u_vertex, 0.0, u_vertex, sa), u_rim
        return _line(u_vertex, sa, u_vertex, 0.0), u_rim
    u_rim = surf_u(R, u_vertex, sa)
    u_mid = surf_u(R, u_vertex, sa / 2.0)
    if start_at_axis:
        return _arc3(u_vertex, 0.0, u_mid, sa / 2.0, u_rim, sa), u_rim
    return _arc3(u_rim, sa, u_mid, sa / 2.0, u_vertex, 0.0), u_rim


def lens_meridian(front_R, back_R, ct, sa, xf):
    """Closed meridian for a lens: front surface (toward -x) at vertex xf, back
    surface at vertex xf+ct, semi-aperture sa. Returns Part edges."""
    xfv, xbv = xf, xf + ct
    front_edge, xfr = _surface_edge(front_R, xfv, sa, start_at_axis=True)
    back_edge, xbr = _surface_edge(back_R, xbv, sa, start_at_axis=False)
    edges = [front_edge]
    if abs(xbr - xfr) > 1e-7:
        edges.append(_line(xfr, sa, xbr, sa))     # rim (revolves -> cylinder)
    edges.append(back_edge)
    edges.append(_line(xbv, 0.0, xfv, 0.0))       # axis closing segment
    return edges, (xfr, xbr, xfv, xbv)


# =============================================================================
# Scene-authoring convenience (source + detector bodies)
# =============================================================================
def add_source(doc, name, x_face, radius, props, length=1.0):
    """Collimated disc source: circle pad whose +x cap (closest to origin) is
    the emitting face, at x=x_face."""
    return new_body_pad(doc, name, name, circle=(0.0, 0.0, radius),
                        x_start=x_face - length, length=length, props=props)


def add_detector(doc, name, x_face, half, thick=1.0):
    """Planar detector screen; front (-x) face at x_face is the recorded face."""
    return new_body_pad(doc, name, name,
                        rects=[(-half, -half, 2 * half, 2 * half)],
                        x_start=x_face, length=thick,
                        props={"material": "detector"})


def add_detector_plane(doc, name, half, thick, placement):
    """Detector screen with an arbitrary placement (for off-axis arms)."""
    return pad_body(doc, name,
                    [_line(-half, -half, half, -half), _line(half, -half, half, half),
                     _line(half, half, -half, half), _line(-half, half, -half, -half)],
                    plane="XY", offset=-thick / 2.0, length=thick,
                    props={"material": "detector"}, placement=placement)


def finalize(doc, outpath):
    doc.recompute()
    bad = [o.Name for o in doc.Objects
           if any("Invalid" in s or "Error" in s
                  for s in (getattr(o, "State", None) or []))]
    if bad:
        log("ERROR: recompute left invalid objects in %s: %s"
            % (outpath, bad))
        os._exit(1)
    doc.saveAs(str(outpath))
    log("wrote %s" % outpath)


# =============================================================================
# doubleslit (unchanged from the original)
# =============================================================================
def make_doubleslit(outpath):
    doc = App.newDocument("doubleslit")
    try:
        slit_w = 0.1
        slit_sep = 0.5
        slit_h = 8.0
        plate_th = 1.0
        plate = 10.0
        holes = []
        for yc in (-slit_sep / 2, slit_sep / 2):
            holes.append((yc - slit_w / 2, -slit_h / 2, slit_w, slit_h))
        rects = [(-plate / 2, -plate / 2, plate, plate)] + holes
        new_body_pad(doc, "SlitPlate", "SlitPlate", rects=rects,
                     x_start=0.0, length=plate_th,
                     props={"material": "aluminum"})
        for i, yc in enumerate((-slit_sep / 2, slit_sep / 2)):
            new_body_pad(doc, "SlitFill%d" % i, "SlitFill%d" % i,
                         rects=[(yc - slit_w / 2, -slit_h / 2,
                                 slit_w, slit_h)],
                         x_start=0.0, length=plate_th,
                         props={"material": "air"})
        new_body_pad(doc, "Laser", "Laser", circle=(0.0, 0.0, 1.5),
                     x_start=-51.0, length=1.0,
                     props={"power": 5.0, "lambdac": 633.0,
                            "coherent": True})
        new_body_pad(doc, "Screen", "Screen",
                     rects=[(-6.0, -6.0, 12.0, 12.0)],
                     x_start=100.0, length=1.0,
                     props={"material": "detector"})
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# Revolved-lens scenes (pcx / dcx / pcv / dcv / sphere_control / achromat)
# =============================================================================
def _build_simple_lens(doc, s, name="Lens", extra_props=None):
    edges, _ = lens_meridian(s["R1_mm"], s["R2_mm"], s["thickness_mm"],
                             s["aperture_mm"] / 2.0, 0.0)
    props = {"material": s["material"]}
    if extra_props:
        props.update(extra_props)
    revolve_body(doc, name, edges, props=props)


def make_lens_scene(name, outpath):
    s = SCENES[name]
    doc = App.newDocument(name)
    try:
        _build_simple_lens(doc, s)
        add_source(doc, "Source", -30.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"]})
        add_detector(doc, "Screen", s["detector_x_mm"], 15.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def make_lens_sphere_control(outpath):
    make_lens_scene("lens_sphere_control", outpath)


def make_auto_designed_lens(outpath):
    """The optimizer round's demo scene (see the SCENES entry): lens_pcx
    whose body Placement.Base.x is expression-bound to a 'dim' spreadsheet
    'lenspos' alias — exactly the example.FCStd pattern, so permute_model
    --var / fast_eval apply_params drive it. Detector fixed 4mm past the
    lenspos=0 focus; scripts/optimize.py should find lenspos ~= +4."""
    s = SCENES["auto_designed_lens"]
    doc = App.newDocument("auto_designed_lens")
    try:
        sh = doc.addObject("Spreadsheet::Sheet", "dim")
        sh.Label = "dim"
        sh.set("A1", "lens axial offset [mm]")
        sh.set("B1", "=0 mm")
        sh.setAlias("B1", "lenspos")
        doc.recompute()
        _build_simple_lens(doc, s)
        lens = doc.getObject("Lens")
        lens.setExpression(".Placement.Base.x", "<<dim>>.lenspos")
        add_source(doc, "Source", -30.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "coherent": False})
        add_detector(doc, "Screen", s["detector_x_mm"], 15.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def make_tolerance_lens(outpath):
    """The tolerancing round's demo scene (see the SCENES entry): the
    auto_designed_lens singlet with lens position AND decenter AND
    detector position spreadsheet-driven (all expression-bound
    Placement.Base fields on the 'dim' sheet), nominally in focus.
    scripts/tolerance.py perturbs lenspos/lensdy and refocuses with
    detpos."""
    s = SCENES["tolerance_lens"]
    doc = App.newDocument("tolerance_lens")
    try:
        sh = doc.addObject("Spreadsheet::Sheet", "dim")
        sh.Label = "dim"
        sh.set("A1", "lens axial offset [mm]")
        sh.set("B1", "=0 mm")
        sh.setAlias("B1", "lenspos")
        sh.set("A2", "lens y decenter [mm]")
        sh.set("B2", "=0 mm")
        sh.setAlias("B2", "lensdy")
        sh.set("A3", "detector x position [mm]")
        sh.set("B3", "=%g mm" % s["focus_x_at_lenspos0_mm"])
        sh.setAlias("B3", "detpos")
        doc.recompute()
        _build_simple_lens(doc, s)
        lens = doc.getObject("Lens")
        lens.setExpression(".Placement.Base.x", "<<dim>>.lenspos")
        lens.setExpression(".Placement.Base.y", "<<dim>>.lensdy")
        add_source(doc, "Source", -30.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "coherent": False})
        # detector geometry built at x=0; its whole placement rides on
        # the detpos alias (the focus-compensator variable)
        det = add_detector(doc, "Screen", 0.0, 15.0)
        det.setExpression(".Placement.Base.x", "<<dim>>.detpos")
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def make_lens_achromat(outpath):
    s = SCENES["lens_achromat"]
    doc = App.newDocument("lens_achromat")
    try:
        sa = s["aperture_mm"] / 2.0
        R1, Ri, R3 = s["R1_mm"], s["R2_interface_mm"], s["R3_mm"]
        ctc, ctf, gap = (s["crown_thickness_mm"], s["flint_thickness_mm"],
                         s["air_gap_mm"])
        # crown: front R1 (convex), back Ri (interface); front vertex x=0
        crown_edges, crown_x = lens_meridian(R1, Ri, ctc, sa, 0.0)
        revolve_body(doc, "Crown", crown_edges, props={"material":
                                                        s["crown_material"]})
        # flint: front Ri (same radius, +gap), back R3; front vertex = ctc+gap
        flint_edges, _ = lens_meridian(Ri, R3, ctf, sa, ctc + gap)
        revolve_body(doc, "Flint", flint_edges, props={"material":
                                                       s["flint_material"]})
        add_source(doc, "Source", -30.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["d_line_nm"]})
        add_detector(doc, "Screen", s["detector_x_mm"], 12.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# lens_asphere — revolved BSpline through exact sag samples + surface_override
# =============================================================================
def _asphere_sag(r, R, k, a4=0.0):
    """Even asphere sag: conic + A4 r^4 (a4 in mm^-3, r in mm). Same sag
    convention as extract_geometry.asphere_sag_m / the surface_override."""
    c = 1.0 / R
    conic = c * r * r / (1.0 + math.sqrt(1.0 - (1.0 + k) * c * c * r * r))
    return conic + a4 * r ** 4


def make_lens_asphere(outpath):
    s = SCENES["lens_asphere"]
    doc = App.newDocument("lens_asphere")
    try:
        sa = s["aperture_mm"] / 2.0
        R, k, ct = s["R1_mm"], s["conic_k"], s["thickness_mm"]
        a4 = s["asphere_A4_mm"]
        # convex front toward source: vertex at x=0, sag increases toward +x.
        n_samp = 41
        pts = []
        for i in range(n_samp):
            v = sa * i / (n_samp - 1)
            u = _asphere_sag(v, R, k, a4)     # >=0, vertex at u=0
            pts.append(App.Vector(u, v, 0))
        bs = Part.BSplineCurve()
        bs.interpolate(pts)
        xfr = pts[-1].x                    # front rim x at v=sa
        xbv = ct                           # flat back vertex
        edges = [bs,
                 _line(xfr, sa, xbv, sa),          # rim -> cylinder
                 _line(xbv, sa, xbv, 0.0),         # flat back -> plane
                 _line(xbv, 0.0, 0.0, 0.0)]        # axis closing segment
        # surface_override declares Face1 (the revolved BSpline) an analytic
        # asphere; the extractor verifies it against the geometry to <1um.
        ov = "Face1=asphere:R=%.6f;k=%.6f;A4=%.9g;r_max=%.4f" % (R, k, a4, sa)
        revolve_body(doc, "Lens", edges,
                     props={"material": s["material"], "surface_override": ov})
        add_source(doc, "Source", -30.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"]})
        add_detector(doc, "Screen", s["detector_x_mm"], 15.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# Cylinder lenses (padded arc profile -> native Cylinder face)
# =============================================================================
def _cyl_lens_profile(R, ct, sa):
    """Plano-convex/concave cylinder-lens profile in the (x, y) plane: curved
    front surface (radius R, vertex on axis at x=0), flat back at x=xb.
    Returns Part edges (padded along z gives a Cylinder + Plane faces)."""
    # front arc from (xfr,-sa) through vertex (0,0)-ish to (xfr,+sa)
    u_mid = surf_u(R, 0.0, sa)         # rim x
    u_vtx = 0.0
    # keep the flat back beyond the deepest point of the curved surface
    xb = max(u_mid, u_vtx) + ct
    top = _arc3(u_mid, sa, surf_u(R, 0.0, 0.0), 0.0, u_mid, -sa) \
        if False else None
    # arc from (u_mid,-sa) via vertex (u_vtx,0) to (u_mid,+sa)
    arc = _arc3(u_mid, -sa, u_vtx, 0.0, u_mid, sa)
    edges = [arc,
             _line(u_mid, sa, xb, sa),
             _line(xb, sa, xb, -sa),
             _line(xb, -sa, u_mid, -sa)]
    return edges, xb


def make_cyl_lens(name, outpath):
    s = SCENES[name]
    doc = App.newDocument(name)
    try:
        sa = s["aperture_mm"] / 2.0
        H = s["height_z_mm"]
        edges, _ = _cyl_lens_profile(s["R_mm"], s["thickness_mm"], sa)
        pad_body(doc, "Lens", edges, plane="XY", offset=-H / 2.0, length=H,
                 props={"material": s["material"]})
        add_source(doc, "Source", -30.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"]})
        add_detector(doc, "Screen", s["detector_x_mm"], 15.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# axicon (revolved slant meridian -> native Cone face)
# =============================================================================
def make_axicon(outpath):
    s = SCENES["axicon_pcx"]
    doc = App.newDocument("axicon_pcx")
    try:
        sa = s["aperture_mm"] / 2.0
        alpha = math.radians(s["base_angle_deg"])
        axial = sa * math.tan(alpha)        # apex(x=0) to base plane
        # meridian: apex(0,0) -> slant to rim(axial,sa) -> radial base to
        # (axial,0) -> axis back to apex.
        edges = [_line(0.0, 0.0, axial, sa),
                 _line(axial, sa, axial, 0.0),
                 _line(axial, 0.0, 0.0, 0.0)]
        revolve_body(doc, "Axicon", edges, props={"material": s["material"]})
        add_source(doc, "Source", -30.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"]})
        add_detector(doc, "Screen", axial + s["detector_z_mm"], 8.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# ball lens (revolved semicircle -> native Sphere) and rod (padded circle)
# =============================================================================
def make_lens_ball(outpath):
    s = SCENES["lens_ball"]
    doc = App.newDocument("lens_ball")
    try:
        R = s["diameter_mm"] / 2.0
        xc = R                                  # centre at x=R -> touches x=0
        # semicircle meridian: (0,0) via (xc,R) to (2R,0), closed by the axis.
        edges = [_arc3(0.0, 0.0, xc, R, 2 * R, 0.0),
                 _line(2 * R, 0.0, 0.0, 0.0)]
        revolve_body(doc, "Ball", edges, props={"material": s["material"]})
        add_source(doc, "Source", -30.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"]})
        add_detector(doc, "Screen", s["detector_x_mm"], 6.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def make_lens_rod(outpath):
    s = SCENES["lens_rod"]
    doc = App.newDocument("lens_rod")
    try:
        R = s["diameter_mm"] / 2.0
        H = s["length_z_mm"]
        circ = [Part.Circle(App.Vector(R, 0.0, 0.0), App.Vector(0, 0, 1), R)]
        pad_body(doc, "Rod", circ, plane="XY", offset=-H / 2.0, length=H,
                 props={"material": s["material"]})
        add_source(doc, "Source", -30.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"]})
        add_detector(doc, "Screen", s["detector_x_mm"], 8.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# Fresnel lens (sawtooth meridian of conical facets -> native Cone faces)
# =============================================================================
def make_lens_fresnel(outpath):
    s = SCENES["lens_fresnel"]
    doc = App.newDocument("lens_fresnel")
    try:
        sa = s["aperture_mm"] / 2.0
        n = s["n_facets"]
        f = s["expected_efl_mm"]
        nglass = s["n_633"]
        h = s["facet_height_mm"]     # facet draft height (x extent of each tooth)
        back = h + 2.0               # flat back plane x
        # Each facet spans radial [v0,v1]; its front slope matches the local
        # slope of the ideal plano-convex sag  z(v)=v^2/(2 (n-1) f)  (thin-lens
        # limit): dz/dv = v/((n-1) f). The facet is a straight (conical) slope
        # from the outer edge (front, x=0) up-and-in, then a riser back to x=0.
        edges = []
        v_prev = 0.0
        x_prev = 0.0
        pts = []   # build the closed sawtooth polyline (u=x, v)
        pts.append((0.0, 0.0))       # axis vertex, front
        for i in range(n):
            v0 = sa * i / n
            v1 = sa * (i + 1) / n
            slope = 0.5 * (v0 + v1) / ((nglass - 1.0) * f)   # dz/dv at zone mid
            # facet front surface: from (x=0, v0) slanting out to (x=dx, v1)
            dx = slope * (v1 - v0)
            # riser from previous facet tip back down to front plane at v0
            pts.append((0.0, v0))            # front plane at inner edge of facet
            pts.append((dx, v1))            # slanted cone face out to v1
        # close: from outer rim (dx_last, sa) across to back plane, down, back
        last_dx = pts[-1][0]
        pts.append((back, sa))
        pts.append((back, 0.0))
        # build edges from consecutive points (dedupe identical)
        poly = []
        for p in pts:
            if not poly or (abs(poly[-1][0] - p[0]) > 1e-9
                            or abs(poly[-1][1] - p[1]) > 1e-9):
                poly.append(p)
        for a, b in zip(poly, poly[1:]):
            edges.append(_line(a[0], a[1], b[0], b[1]))
        edges.append(_line(poly[-1][0], poly[-1][1], poly[0][0], poly[0][1]))
        revolve_body(doc, "Fresnel", edges, props={"material": s["material"]})
        add_source(doc, "Source", -30.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"]})
        add_detector(doc, "Screen", s["detector_x_mm"], 12.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# prism (equilateral, padded triangle, rotated near minimum deviation)
# =============================================================================
def make_prism_equilateral(outpath):
    s = SCENES["prism_equilateral"]
    doc = App.newDocument("prism_equilateral")
    try:
        L = s["side_mm"]
        H = s["height_z_mm"]
        R = L / math.sqrt(3.0)               # circumradius
        # equilateral triangle centred at origin, apex pointing +y
        verts = [(R * math.cos(math.radians(a)), R * math.sin(math.radians(a)))
                 for a in (90.0, 210.0, 330.0)]
        edges = [_line(verts[i][0], verts[i][1],
                       verts[(i + 1) % 3][0], verts[(i + 1) % 3][1])
                 for i in range(3)]
        rot = App.Rotation(App.Vector(0, 0, 1), s["prism_rotation_deg"])
        pl = App.Placement(App.Vector(0, 0, 0), rot)
        pad_body(doc, "Prism", edges, plane="XY", offset=-H / 2.0, length=H,
                 props={"material": s["material"]}, placement=pl)
        add_source(doc, "Source", -40.0, 3.0,
                   {"power": 5.0, "lambdac": s["lambdac_nm"],
                    "lambdamin": s["lambdamin_nm"], "lambdamax": s["lambdamax_nm"]})
        # Face-on screen for the dispersed fan (which bends toward -y). Normal
        # anti-parallel to the 550nm exit ray; shortest-arc rotation from the
        # plate's local +z. See SCENES for the placement derivation.
        det_c = s["detector_center_mm"]
        det_n = s["detector_normal"]
        pl_det = App.Placement(App.Vector(*det_c),
                               App.Rotation(App.Vector(0, 0, 1),
                                            App.Vector(*det_n)))
        add_detector_plane(doc, "Screen", s["detector_half_mm"], 1.0, pl_det)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# Polarization / crystal-optic slabs (thin plane-faced plates)
# =============================================================================
def _plate_edges(sa):
    return [_line(-sa, -sa, sa, -sa), _line(sa, -sa, sa, sa),
            _line(sa, sa, -sa, sa), _line(-sa, sa, -sa, -sa)]


def make_pol_linear(outpath):
    s = SCENES["pol_linear"]
    doc = App.newDocument("pol_linear")
    try:
        sh = doc.addObject("Spreadsheet::Sheet", "Spreadsheet")
        sh.Label = "dim"
        sh.set("A1", "=%g deg" % s["polangle_deg"])
        sh.setAlias("A1", "polangle")
        doc.recompute()
        # polarizer plate on YZ (normal +x), body-local axis '0,0,1'.
        pol = new_body_pad(doc, "Polarizer", "Polarizer",
                           rects=[(-10.0, -10.0, 20.0, 20.0)],
                           x_start=0.0, length=s["substrate_mm"],
                           props={"material": s["substrate_material"],
                                  "polarizer": s["polarizer"],
                                  "polarizer_axis": s["polarizer_axis_local"]})
        pol.Placement = App.Placement(
            App.Vector(0, 0, 0),
            App.Rotation(App.Vector(1, 0, 0), s["polangle_deg"]))
        try:
            pol.setExpression("Placement.Rotation.Angle", "<<dim>>.polangle")
        except Exception as exc:
            log("pol_linear: polangle expression not bound (%s); axis fixed"
                % exc)
        add_source(doc, "Source", -30.0, 5.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "polarization": s["source_polarization"]})
        add_detector(doc, "Screen", s["detector_x_mm"], 12.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def make_pol_crossed(outpath):
    s = SCENES["pol_crossed"]
    doc = App.newDocument("pol_crossed")
    try:
        new_body_pad(doc, "Pol1", "Pol1",
                     rects=[(-10.0, -10.0, 20.0, 20.0)],
                     x_start=0.0, length=1.0,
                     props={"material": s["substrate_material"],
                            "polarizer": s["polarizer"],
                            "polarizer_axis": s["axis1_local"]})
        new_body_pad(doc, "Pol2", "Pol2",
                     rects=[(-10.0, -10.0, 20.0, 20.0)],
                     x_start=5.0, length=1.0,
                     props={"material": s["substrate_material"],
                            "polarizer": s["polarizer"],
                            "polarizer_axis": s["axis2_local"]})
        add_source(doc, "Source", -30.0, 5.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "polarization": "unpolarized"})
        add_detector(doc, "Screen", s["detector_x_mm"], 12.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def make_pol_circular(outpath):
    s = SCENES["pol_circular"]
    doc = App.newDocument("pol_circular")
    try:
        new_body_pad(doc, "CircPol", "CircPol",
                     rects=[(-10.0, -10.0, 20.0, 20.0)],
                     x_start=0.0, length=s["substrate_mm"],
                     props={"material": s["substrate_material"],
                            "polarizer": s["polarizer"]})
        add_source(doc, "Source", -30.0, 5.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "polarization": s["source_polarization"]})
        add_detector(doc, "Screen", s["detector_x_mm"], 12.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def make_waveplate_quartz(outpath, scene_key="waveplate_quartz"):
    s = SCENES[scene_key]
    doc = App.newDocument(scene_key)
    try:
        # input polarizer +45
        new_body_pad(doc, "PolIn", "PolIn",
                     rects=[(-8.0, -8.0, 16.0, 16.0)], x_start=0.0, length=1.0,
                     props={"material": s["substrate_material"],
                            "polarizer": "ideal_linear",
                            "polarizer_axis": s["pol_in_axis_local"]})
        # quartz half-wave plate, crystal_axis +z
        new_body_pad(doc, "Waveplate", "Waveplate",
                     rects=[(-8.0, -8.0, 16.0, 16.0)], x_start=3.0,
                     length=s["thickness_mm"],
                     props={"material": s["material"],
                            "crystal_axis": s["crystal_axis_local"]})
        # analyzer -45 (crossed vs input)
        new_body_pad(doc, "PolOut", "PolOut",
                     rects=[(-8.0, -8.0, 16.0, 16.0)],
                     x_start=3.0 + s["thickness_mm"] + 2.0, length=1.0,
                     props={"material": s["substrate_material"],
                            "polarizer": "ideal_linear",
                            "polarizer_axis": s["pol_out_axis_local"]})
        add_source(doc, "Source", -30.0, 5.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "polarization": "unpolarized"})
        add_detector(doc, "Screen", s["detector_x_mm"], 10.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# PBS cube (two 45deg prisms, hypotenuse coated, 5um gap, two detectors)
# =============================================================================
def make_pbs_cube(outpath):
    s = SCENES["pbs_cube"]
    doc = App.newDocument("pbs_cube")
    try:
        c = s["cube_mm"]
        H = s["height_z_mm"]
        gap = s["air_gap_mm"]
        half = c / 2.0
        # cube corners in x-y: A(0,-half) B(c,-half) C(c,half) D(0,half).
        # Diagonal \ from D(0,half) to B(c,-half).
        # Prism1 (entrance, left): A,D,B -> right angle at A, hypotenuse D-B.
        A, B, C, D = (0.0, -half), (c, -half), (c, half), (0.0, half)
        p1 = [A, D, B]
        p1_edges = [_line(*p1[i], *p1[(i + 1) % 3]) for i in range(3)]
        b1 = pad_body(doc, "Prism1", p1_edges, plane="XY", offset=-H / 2.0,
                      length=H, props={"material": s["material"]})
        # Prism2 (exit): D,B,C -> right angle at C, shifted +gap along the
        # hypotenuse outward normal (1,1)/sqrt2 to open a 5um air gap.
        p2 = [D, B, C]
        p2_edges = [_line(*p2[i], *p2[(i + 1) % 3]) for i in range(3)]
        shift = gap / math.sqrt(2.0)
        pl2 = App.Placement(App.Vector(shift, shift, 0.0), App.Rotation())
        pad_body(doc, "Prism2", p2_edges, plane="XY", offset=-H / 2.0,
                 length=H, props={"material": s["material"]}, placement=pl2)
        # coat prism1's hypotenuse (the D-B face). Find it after recompute by
        # locating the plane face whose normal is ~ (1,1,0)/sqrt2.
        doc.recompute()
        hyp = _find_face_by_normal(b1, (1.0, 1.0, 0.0))
        if hyp is None:
            log("ERROR: pbs_cube could not find hypotenuse face on Prism1")
            os._exit(1)
        b1.addProperty("App::PropertyString", "coating", "Base")
        b1.coating = "%s.%s=%s" % (b1.Name, b1.Tip.Name if b1.Tip else b1.Name,
                                   s["coating"])
        # rewrite as FaceN=coating
        b1.coating = "Face%d=%s" % (hyp, s["coating"])
        add_source(doc, "Source", -30.0, 4.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "polarization": "unpolarized"})
        add_detector(doc, "DetTrans", s["det_trans_x_mm"], 12.0)
        # reflected arm exits the -y (bottom) face; detector below, facing +y.
        pl_r = App.Placement(App.Vector(half, s["det_refl_y_mm"], 0.0),
                             App.Rotation(App.Vector(1, 0, 0), 90.0))
        add_detector_plane(doc, "DetRefl", 12.0, 1.0, pl_r)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def _find_face_by_normal(body, target, tol=1e-3):
    """Return the 1-based face index whose outward-ish normal is ~parallel to
    `target` (a lattice direction), else None. Uses the face centre normal."""
    import FreeCAD as _fc
    t = _fc.Vector(*target)
    t.normalize()
    best, best_dot = None, tol
    for i, f in enumerate(body.Shape.Faces, start=1):
        try:
            u0, u1, v0, v1 = f.ParameterRange
            nrm = f.normalAt((u0 + u1) / 2.0, (v0 + v1) / 2.0)
            nrm.normalize()
        except Exception:
            continue
        d = abs(nrm.dot(t))
        if d > best_dot:
            best_dot, best = d, i
    return best


# =============================================================================
# calcite displacer (single birefringent slab, off-axis crystal axis)
# =============================================================================
def make_calcite_displacer(outpath):
    s = SCENES["calcite_displacer"]
    doc = App.newDocument("calcite_displacer")
    try:
        new_body_pad(doc, "Calcite", "Calcite",
                     rects=[(-6.0, -6.0, 12.0, 12.0)], x_start=0.0,
                     length=s["thickness_mm"],
                     props={"material": s["material"],
                            "crystal_axis": s["crystal_axis_local"]})
        add_source(doc, "Source", -20.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "polarization": "unpolarized"})
        add_detector(doc, "Screen", s["detector_x_mm"], 4.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# Wollaston prism (two orthogonal-axis calcite wedges, 5um gap)
# =============================================================================
def make_wollaston(outpath):
    s = SCENES["wollaston"]
    doc = App.newDocument("wollaston")
    try:
        H = s["height_z_mm"]
        gap = s["air_gap_mm"]
        beta = math.radians(s["wedge_angle_deg"])
        halfz = H / 2.0
        w = 6.0                       # half-width in y
        t = 2 * w * math.tan(beta)    # thickness swing across the wedge
        # Wedge1: right triangle in x-y, hypotenuse rising +x with y.
        # verts: (0,-w),(t,-w)... actually build two complementary wedges whose
        # slanted faces mate at a 5um gap.
        # Wedge1 occupies x in [0, t] with slant from (0,-w) to (t, w) ... use:
        w1 = [(0.0, -w), (t, w), (0.0, w)]
        e1 = [_line(*w1[i], *w1[(i + 1) % 3]) for i in range(3)]
        pad_body(doc, "Wedge1", e1, plane="XY", offset=-halfz, length=H,
                 props={"material": s["material"],
                        "crystal_axis": s["axis1_local"]})
        # Wedge2 mates on the slant, forming a rectangular block x in [~0, t+..]
        w2 = [(0.0, -w), (t, -w), (t, w)]
        e2 = [_line(*w2[i], *w2[(i + 1) % 3]) for i in range(3)]
        # separate along the slant normal to open the gap
        nrm = (math.sin(beta), -math.cos(beta))
        pl2 = App.Placement(App.Vector(gap * nrm[0], gap * nrm[1], 0.0),
                            App.Rotation())
        pad_body(doc, "Wedge2", e2, plane="XY", offset=-halfz, length=H,
                 props={"material": s["material"],
                        "crystal_axis": s["axis2_local"]}, placement=pl2)
        add_source(doc, "Source", -20.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "polarization": "unpolarized"})
        add_detector(doc, "Screen", s["detector_x_mm"], 12.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# filter (bandpass) and hot mirror
# =============================================================================
def make_filter_bandpass(outpath):
    s = SCENES["filter_bandpass"]
    doc = App.newDocument("filter_bandpass")
    try:
        new_body_pad(doc, "Filter", "Filter",
                     rects=[(-10.0, -10.0, 20.0, 20.0)], x_start=0.0,
                     length=s["substrate_mm"],
                     props={"material": s["material"], "filter": s["filter"]})
        add_source(doc, "Source", -30.0, 5.0,
                   {"power": 5.0, "lambdac": s["lambdac_nm"],
                    "lambdamin": s["lambdamin_nm"], "lambdamax": s["lambdamax_nm"]})
        add_detector(doc, "Screen", s["detector_x_mm"], 12.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def make_hot_mirror(outpath):
    s = SCENES["hot_mirror"]
    doc = App.newDocument("hot_mirror")
    try:
        S = s["plate_size_mm"] / 2.0
        th = s["plate_thickness_mm"]
        # plate on YZ (normal +x) then rotate 45deg about z so +x beam hits at
        # 45deg AOI; transmitted -> +x, reflected -> +y.
        pl = App.Placement(App.Vector(0, 0, 0),
                           App.Rotation(App.Vector(0, 0, 1),
                                        s["plate_rotation_deg"]))
        pad_body(doc, "HotMirror",
                 [_line(-S, -S, S, -S), _line(S, -S, S, S),
                  _line(S, S, -S, S), _line(-S, S, -S, -S)],
                 plane="YZ", offset=-th / 2.0, length=th,
                 props={"material": s["material"], "coating": s["coating"]},
                 placement=pl)
        add_source(doc, "Source", -40.0, 6.0,
                   {"power": 5.0, "lambdac": s["lambdac_nm"],
                    "lambdamin": s["lambdamin_nm"], "lambdamax": s["lambdamax_nm"]})
        add_detector(doc, "DetPass", s["det_pass_x_mm"], 15.0)
        pl_r = App.Placement(App.Vector(0.0, s["det_refl_y_mm"], 0.0),
                             App.Rotation(App.Vector(1, 0, 0), 90.0))
        add_detector_plane(doc, "DetRefl", 15.0, 1.0, pl_r)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# mesh_freeform (revolved BSpline ellipsoid -> 'mesh' fallback, no override)
# =============================================================================
def make_mesh_freeform(outpath):
    s = SCENES["mesh_freeform"]
    doc = App.newDocument("mesh_freeform")
    try:
        a = s["semi_major_x_mm"]
        b = s["semi_minor_r_mm"]
        # prolate half-ellipse meridian: (0,0) .. (a,b) .. (2a,0), a!=b so the
        # revolved SurfaceOfRevolution cannot canonicalize to sphere/cylinder.
        n_samp = 31
        pts = []
        for i in range(n_samp):
            th = math.pi * i / (n_samp - 1)     # 0..pi
            x = a * (1.0 - math.cos(th))         # 0..2a
            r = b * math.sin(th)                 # 0..b..0
            pts.append(App.Vector(x, r, 0))
        bs = Part.BSplineCurve()
        bs.interpolate(pts)
        edges = [bs, _line(2 * a, 0.0, 0.0, 0.0)]
        revolve_body(doc, "Freeform", edges, props={"material": s["material"]})
        add_source(doc, "Source", -30.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"]})
        add_detector(doc, "Screen", s["detector_x_mm"], 12.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# Phase-12 new-physics scenes
# =============================================================================
def make_ktp_walkoff(outpath):
    s = SCENES["ktp_walkoff"]
    doc = App.newDocument("ktp_walkoff")
    try:
        t = s["thickness_mm"]
        new_body_pad(doc, "KTP", "KTP",
                     rects=[(-6.0, -6.0, 12.0, 12.0)], x_start=0.0, length=t,
                     props={"material": s["material"],
                            "crystal_axis": s["crystal_axis_local"],
                            "crystal_axis2": s["crystal_axis2_local"]})
        add_source(doc, "Source", -10.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "polarization": "unpolarized"})
        add_detector(doc, "Screen", s["detector_x_mm"], 4.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def make_gaussian_bench(outpath):
    s = SCENES["gaussian_bench"]
    doc = App.newDocument("gaussian_bench")
    try:
        add_source(doc, "Source", s["source_x_mm"],
                   s["source_aperture_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "coherent": False,
                    "beam_waist": s["beam_waist_mm"], "m2": s["m2"]})
        add_detector(doc, "Screen", s["detector_x_mm"], s["det_half_mm"])
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def make_ghost_doublet(outpath):
    s = SCENES["ghost_doublet"]
    doc = App.newDocument("ghost_doublet")
    try:
        w = s["window_size_mm"] / 2.0
        th = s["window_thickness_mm"]
        for i, x0 in enumerate((s["g1_x0_mm"], s["g2_x0_mm"])):
            new_body_pad(doc, "Glass%d" % (i + 1), "Glass%d" % (i + 1),
                         rects=[(-w, -w, 2 * w, 2 * w)], x_start=x0, length=th,
                         props={"material": s["material"]})
        add_source(doc, "Source", -20.0, s["beam_dia_mm"] / 2.0,
                   {"power": 10.0, "lambdac": s["lambda_nm"],
                    "coherent": False})
        add_detector(doc, "Screen", s["detector_x_mm"], 10.0)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def make_scatter_plate(outpath):
    s = SCENES["scatter_plate"]
    doc = App.newDocument("scatter_plate")
    try:
        S = s["plate_size_mm"] / 2.0
        th = s["plate_thickness_mm"]
        # plate on YZ (normal +x) then rotate 45deg about z so the +x beam
        # hits at 45deg AOI; the Fresnel reflection folds to +y.
        pl = App.Placement(App.Vector(0, 0, 0),
                           App.Rotation(App.Vector(0, 0, 1),
                                        s["plate_rotation_deg"]))
        plate = pad_body(doc, "Window",
                         [_line(-S, -S, S, -S), _line(S, -S, S, S),
                          _line(S, S, -S, S), _line(-S, S, -S, -S)],
                         plane="YZ", offset=-th / 2.0, length=th,
                         props={"material": s["material"]}, placement=pl)
        # measured scatter on the beam-facing front face (outward normal has
        # the most-negative dot with the +x beam, i.e. points back at source).
        doc.recompute()
        # beam-facing front cap: outward normal = (-cos(theta), -sin(theta), 0)
        # for a +x face rotated theta about z; theta=-45 -> (-0.707, +0.707, 0).
        th = math.radians(s["plate_rotation_deg"])
        front = _find_face_by_signed_normal(
            plate, (-math.cos(th), -math.sin(th), 0.0))
        if front is None:
            log("ERROR: scatter_plate could not find front face")
            os._exit(1)
        plate.addProperty("App::PropertyString", "scatter", "Base")
        plate.scatter = "Face%d=%s" % (front, s["scatter"])
        add_source(doc, "Source", -30.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "coherent": False})
        # reflected-arm screen at +y, facing -y (records the folded beam).
        pl_r = App.Placement(App.Vector(0.0, s["det_refl_y_mm"], 0.0),
                             App.Rotation(App.Vector(1, 0, 0), 90.0))
        add_detector_plane(doc, "DetRefl", 15.0, 1.0, pl_r)
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def _find_face_by_signed_normal(body, target, tol=1e-3):
    """Return the 1-based face index whose OUTWARD normal is closest (signed)
    to `target` (a direction), else None. Unlike _find_face_by_normal this
    respects sign, so a front cap is distinguished from a parallel back cap."""
    import FreeCAD as _fc
    t = _fc.Vector(*target)
    t.normalize()
    best, best_dot = None, -2.0
    for i, f in enumerate(body.Shape.Faces, start=1):
        try:
            u0, u1, v0, v1 = f.ParameterRange
            nrm = f.normalAt((u0 + u1) / 2.0, (v0 + v1) / 2.0)
            nrm.normalize()
        except Exception:
            continue
        d = nrm.dot(t)
        if d > best_dot:
            best_dot, best = d, i
    return best if best_dot > tol else None


def make_curved_focal(outpath):
    s = SCENES["curved_focal"]
    doc = App.newDocument("curved_focal")
    try:
        _build_simple_lens(doc, s, name="Lens")
        add_source(doc, "Source", -30.0, s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "coherent": False})
        # concave-toward-beam cylindrical screen: center of curvature at
        # (cx, 0) in FRONT of the surface, so the surface x = cx + sqrt(R^2 -
        # y^2) is deepest (+x) on axis (vertex) and curves forward (-x) at the
        # rim -> concave as seen from the -x beam side. Axis along z.
        cx = s["det_center_of_curvature_x_mm"]
        R = s["det_radius_mm"]
        sa = s["det_half_y_mm"]
        Hz = s["det_height_z_mm"]
        x_vtx = cx + R                         # deepest point (on the focus)
        x_edge = cx + math.sqrt(R * R - sa * sa)
        x_back = x_vtx + s["det_back_pad_mm"]
        # meridian in the (x, y) sketch plane (padded along z): concave arc
        # from (x_edge,-sa) via vertex (x_vtx,0) to (x_edge,+sa), then a flat
        # back closing the solid.
        arc = _arc3(x_edge, -sa, x_vtx, 0.0, x_edge, sa)
        edges = [arc,
                 _line(x_edge, sa, x_back, sa),
                 _line(x_back, sa, x_back, -sa),
                 _line(x_back, -sa, x_edge, -sa)]
        pad_body(doc, "CurvedDet", edges, plane="XY", offset=-Hz / 2.0,
                 length=Hz, props={"material": "detector"})
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


def make_nested4(outpath):
    """Depth-4 concentric nesting spike: four axis-aligned cube shells, each
    strictly inside the previous with a clearance gap (never coincident
    faces -- see CLAUDE.md's nested-solids trap). Built directly with boxes
    (new_body_pad rects) rather than a hollow shell: the tracer's LIFO
    medium stack recovers the "wall" as the region inside the outer solid
    but outside the next nested one, exactly like bs_cube's coated plate."""
    s = SCENES["nested4"]
    doc = App.newDocument("nested4")
    try:
        def cube(name, half, x0, length, material, extra_props=None):
            props = {"material": material}
            if extra_props:
                props.update(extra_props)
            return new_body_pad(
                doc, name, name,
                rects=[(-half, -half, 2 * half, 2 * half)],
                x_start=x0, length=length, props=props)

        cube("OuterVat", s["outer_half_mm"], s["outer_x0_mm"],
             s["outer_len_mm"], s["outer_material"])
        cube("Bath", s["bath_half_mm"], s["bath_x0_mm"], s["bath_len_mm"],
             s["bath_material"])
        cube("InnerCuvette", s["inner_half_mm"], s["inner_x0_mm"],
             s["inner_len_mm"], s["inner_material"])
        cube("Sample", s["sample_half_mm"], s["sample_x0_mm"],
             s["sample_len_mm"], s["sample_material"],
             extra_props={"filter": s["sample_filter"]})
        add_source(doc, "Source", s["source_x_mm"], s["beam_dia_mm"] / 2.0,
                   {"power": 5.0, "lambdac": s["lambda_nm"],
                    "polarization": "unpolarized", "coherent": False})
        add_detector(doc, "Screen", s["detector_x_mm"], s["detector_half_mm"])
        finalize(doc, outpath)
    finally:
        App.closeDocument(doc.Name)


# =============================================================================
# Dispatch
# =============================================================================
BUILDERS = {
    "doubleslit": make_doubleslit,
    "lens_pcx": lambda p: make_lens_scene("lens_pcx", p),
    "lens_dcx": lambda p: make_lens_scene("lens_dcx", p),
    "lens_pcv": lambda p: make_lens_scene("lens_pcv", p),
    "lens_dcv": lambda p: make_lens_scene("lens_dcv", p),
    "lens_achromat": make_lens_achromat,
    "lens_sphere_control": make_lens_sphere_control,
    "auto_designed_lens": make_auto_designed_lens,
    "tolerance_lens": make_tolerance_lens,
    "lens_asphere": make_lens_asphere,
    "lens_cyl_pos": lambda p: make_cyl_lens("lens_cyl_pos", p),
    "lens_cyl_neg": lambda p: make_cyl_lens("lens_cyl_neg", p),
    "axicon_pcx": make_axicon,
    "lens_ball": make_lens_ball,
    "lens_rod": make_lens_rod,
    "lens_fresnel": make_lens_fresnel,
    "prism_equilateral": make_prism_equilateral,
    "pol_linear": make_pol_linear,
    "pol_crossed": make_pol_crossed,
    "pol_circular": make_pol_circular,
    "waveplate_quartz": make_waveplate_quartz,
    "waveplate_mgf2": lambda outpath: make_waveplate_quartz(
        outpath, scene_key="waveplate_mgf2"),
    "pbs_cube": make_pbs_cube,
    "calcite_displacer": make_calcite_displacer,
    "wollaston": make_wollaston,
    "filter_bandpass": make_filter_bandpass,
    "hot_mirror": make_hot_mirror,
    "mesh_freeform": make_mesh_freeform,
    "ktp_walkoff": make_ktp_walkoff,
    "gaussian_bench": make_gaussian_bench,
    "ghost_doublet": make_ghost_doublet,
    "scatter_plate": make_scatter_plate,
    "curved_focal": make_curved_focal,
    "nested4": make_nested4,
}


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.scene == "all":
        names = list(BUILDERS.keys())
    elif args.scene in BUILDERS:
        names = [args.scene]
    else:
        log("ERROR: unknown scene %r (known: %s)"
            % (args.scene, ", ".join(sorted(BUILDERS))))
        os._exit(1)
    for nm in names:
        BUILDERS[nm](outdir / (nm + ".FCStd"))
    log("MAKE TEST SCENES OK (%d scene(s))" % len(names))


# Module-scope autorun (no __main__ guard: FreeCAD -c would skip it).
# MIEWB_MTS_LIBRARY_ONLY=1 suppresses it so primitivelib.py can import the
# geometry helpers under the AppImage without building a scene as a side
# effect.
if _HAVE_FREECAD and not os.environ.get("MIEWB_MTS_LIBRARY_ONLY"):
    main()
    os._exit(0)
