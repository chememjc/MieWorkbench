# =============================================================================
# cengine_scenes.py — minimal synthetic scenes for C-engine parity testing.
#
# Each scene isolates one feature set of the C engine (plan: "simplest
# possible while covering all features — the goal is comparing the two
# implementations, not simulating a real instrument"). Scenes are written
# as geometry dirs (model.json only — analytic faces need no STLs) so BOTH
# engines run through the full run_trace CLI path.
#
# Scene classes:
#   deterministic — collimated normal-incidence stacks where every ray
#     behaves identically regardless of its sampled position: totals
#     (emitted / buckets / detected) must agree between engines to ~1e-9
#     even though the position RNGs differ.
#   statistical — position-dependent physics (curved surfaces): totals
#     agree only statistically; the test uses percent-level tolerances.
# =============================================================================
import json
from pathlib import Path

from scenehelpers import (box_faces, source_body, slab_body, detector_body,
                          make_model, _rect_face)


def _sphere_body(name, material, center, radius):
    """A full (untrimmed) sphere optic — e.g. a ball lens. The trim wire is
    a degenerate meridian seam; area_m2 == full sphere area triggers the
    'untrimmed' trim regime in both engines."""
    import numpy as np
    th = np.linspace(-np.pi / 2, np.pi / 2, 33)
    seam = np.stack([center[0] + radius * np.cos(th),
                     np.full_like(th, center[1]),
                     center[2] + radius * np.sin(th)], axis=1)
    face = {
        "id": "%s.Pad.Face1" % name,
        "surface": {"type": "sphere", "center": list(center),
                    "radius": radius},
        "orientation_outward": True,
        "area_m2": float(4.0 * np.pi * radius ** 2),
        "fingerprint": {},
        "mesh_stl": "",
        "trim_polylines_xyz": [seam.tolist()],
    }
    return {"name": name, "label": name, "role": "optic",
            "material": material, "faces": [face]}


def scene_c_plate():
    """Glass slab at normal incidence: Fresnel split, medium stack,
    internal-reflection generations, escaped side losses. Deterministic."""
    return make_model([
        source_body("Src", x=-0.02, half=0.004, power_mW=2.0,
                    lambdac_nm=633.0),
        slab_body("Plate", "bk7", 0.0, 0.005, half=0.02),
        detector_body("Det", x=0.03, half=0.025),
    ])


def scene_c_mirror_screen():
    """Partial mirror + absorbance on an optic plate (the mirror/absorbance
    amplitude model, tracer.py header) in front of a detector screen.
    Deterministic."""
    return make_model([
        source_body("Src", x=-0.02, half=0.004, power_mW=1.0,
                    lambdac_nm=550.0),
        slab_body("HalfMirror", "bk7", 0.0, 0.002, half=0.02,
                  mirror=0.4, absorbance=0.1),
        detector_body("Det", x=0.03, half=0.025),
    ])


def scene_c_filter():
    """Bandpass-filter slab with a broadband source: per-stratum bulk
    absorption tables (plan D1) + spectral binning. Deterministic."""
    return make_model([
        source_body("Src", x=-0.02, half=0.004, power_mW=1.0,
                    lambdac_nm=550.0, lambdamin_nm=450.0,
                    lambdamax_nm=650.0),
        slab_body("Filter", "bk7", 0.0, 0.003, half=0.02,
                  filter="bp_550_40"),
        detector_body("Det", x=0.03, half=0.025),
    ])


def scene_c_spectrum():
    """Tabulated-emission white-LED source (spectrum=led_white_2733k) through
    a glass plate: the C engine derives its lambda union from the SAME
    wavelength_strata inverse-CDF call as Python (equal-power quantile
    strata), so the per-stratum wavelengths and detected power must match.
    Deterministic (collimated normal incidence)."""
    return make_model([
        source_body("Src", x=-0.02, half=0.004, power_mW=2.0,
                    lambdac_nm=584.6, spectrum="led_white_2733k"),
        slab_body("Plate", "bk7", 0.0, 0.003, half=0.02),
        detector_body("Det", x=0.03, half=0.025),
    ])


def scene_c_ball_lens():
    """Full-sphere ball lens (untrimmed sphere trim regime, curved-surface
    refraction). Statistical: refraction depends on the sampled position."""
    return make_model([
        source_body("Src", x=-0.03, half=0.003, power_mW=1.0,
                    lambdac_nm=633.0),
        _sphere_body("Ball", "bk7", [0.0, 0.0, 0.0], 0.005),
        detector_body("Det", x=0.02, half=0.02),
    ])


def scene_c_torus():
    """Full (untrimmed) glass torus with its tube crossing the beam — the
    quartic intersection path. No repo geometry carries a torus face, so
    this synthetic scene is its only coverage. Statistical."""
    import numpy as np
    R, r = 0.02, 0.004
    cx = 0.01 + R          # near tube segment centered on the beam axis
    # degenerate seam polyline: one tube-circle at azimuth u=0
    th = np.linspace(-np.pi, np.pi, 49)
    seam = np.stack([np.full_like(th, cx - R) + r * np.cos(th) * (-1.0),
                     np.zeros_like(th),
                     r * np.sin(th)], axis=1)
    # place points on the tube at u=pi (x = cx - R - r*cos(v)): param by v
    seam = np.stack([np.full_like(th, cx) - (R + r * np.cos(th)),
                     np.zeros_like(th),
                     r * np.sin(th)], axis=1)
    face = {
        "id": "Donut.Pad.Face1",
        "surface": {"type": "torus", "center": [cx, 0.0, 0.0],
                    "axis": [0.0, 0.0, 1.0], "major_r": R, "minor_r": r},
        "orientation_outward": True,
        "area_m2": float(4.0 * np.pi ** 2 * R * r),
        "fingerprint": {},
        "mesh_stl": "",
        "trim_polylines_xyz": [seam.tolist()],
    }
    donut = {"name": "Donut", "label": "Donut", "role": "optic",
             "material": "bk7", "faces": [face]}
    return make_model([
        source_body("Src", x=-0.02, half=0.002, power_mW=1.0,
                    lambdac_nm=633.0),
        donut,
        detector_body("Det", x=0.03, half=0.025),
    ])


def scene_c_coating_ar():
    """Quarter-wave MgF2 AR coating (TMM) on a glass slab at normal
    incidence. Deterministic."""
    return make_model([
        source_body("Src", x=-0.02, half=0.004, power_mW=1.0,
                    lambdac_nm=550.0),
        slab_body("Coated", "bk7", 0.0, 0.004, half=0.02, coating="MgF2"),
        detector_body("Det", x=0.03, half=0.025),
    ])


def scene_c_polarizer_malus():
    """Linear source through two ideal linear polarizers: the first along
    the source axis (z), the second at 45 deg — Malus cos^2(45) = 0.5
    through the pair. Deterministic."""
    import numpy as np
    return make_model([
        source_body("Src", x=-0.03, half=0.004, power_mW=1.0,
                    lambdac_nm=633.0,
                    polarization={"kind": "linear", "angle_deg": 0.0}),
        slab_body("Pol1", "bk7", -0.01, -0.008, half=0.02,
                  polarizer="ideal_linear", polarizer_axis=[0.0, 0.0, 1.0]),
        slab_body("Pol2", "bk7", 0.0, 0.002, half=0.02,
                  polarizer="ideal_linear",
                  polarizer_axis=[0.0, np.sqrt(0.5), np.sqrt(0.5)]),
        detector_body("Det", x=0.03, half=0.025),
    ])


def scene_c_grating():
    """Transmission grating plate at normal incidence (600 l/mm, orders
    -1..1, lamellar duty-0.5 efficiencies). Deterministic in totals."""
    return make_model([
        source_body("Src", x=-0.02, half=0.003, power_mW=1.0,
                    lambdac_nm=633.0),
        slab_body("Grat", "bk7", 0.0, 0.002, half=0.02,
                  grating={"Grat.Pad.Face2": "600:v:orders=-1..1"}),
        detector_body("Det", x=0.05, half=0.045),
    ])


def scene_c_rough_plate():
    """Rough-faced glass plate: Beckmann lobes + Davies specular
    retention (per-face roughness string). Statistical."""
    return make_model([
        source_body("Src", x=-0.02, half=0.003, power_mW=1.0,
                    lambdac_nm=633.0),
        slab_body("RoughP", "bk7", 0.0, 0.002, half=0.02,
                  roughness_faces={"RoughP.Pad.Face1":
                                   "80:lcorr=5"}),
        detector_body("Det", x=0.03, half=0.028),
    ])


def scene_c_saturable():
    """Saturable-absorber slab at normal incidence (P7 tranche 2 NLO). The
    intensity-dependent bulk absorption alpha0/(1+I/I_sat) rides the same
    Beer-Lambert alpha_add hook as a filter; I is the flat-top source-area
    estimate (no --ray-differentials), the SAME scalar for every ray, so the
    scene stays deterministic. I_sat is tuned so I/I_sat ~ O(1) — the run
    exercises the saturation term, not just alpha0. Deterministic."""
    return make_model([
        source_body("Src", x=-0.02, half=0.004, power_mW=1.0,
                    lambdac_nm=633.0),
        slab_body("SatAbs", "bk7", 0.0, 0.002, half=0.02,
                  saturable="sat:I_sat=1e-3:T0=0.5"),
        detector_body("Det", x=0.03, half=0.025),
    ])


def scene_c_tpa():
    """Two-photon-absorption slab at normal incidence (P7 tranche 2 NLO).
    alpha_TPA(I) = beta_SI * I on the same Beer-Lambert hook; flat-top
    intensity, deterministic. beta is a synthetic value chosen for a
    measurable (~sub-percent) absorbed fraction."""
    return make_model([
        source_body("Src", x=-0.02, half=0.004, power_mW=1.0,
                    lambdac_nm=633.0),
        slab_body("TpaSlab", "bk7", 0.0, 0.002, half=0.02,
                  tpa_beta=5.0e9),
        detector_body("Det", x=0.03, half=0.025),
    ])


# name -> (builder, comparison class)
SCENES = {
    "c_saturable": (scene_c_saturable, "deterministic"),      # P7 tranche 2
    "c_tpa": (scene_c_tpa, "deterministic"),                  # P7 tranche 2
    "c_plate": (scene_c_plate, "deterministic"),
    "c_mirror_screen": (scene_c_mirror_screen, "deterministic"),
    "c_filter": (scene_c_filter, "deterministic"),
    "c_spectrum": (scene_c_spectrum, "deterministic"),         # feature D5
    "c_ball_lens": (scene_c_ball_lens, "statistical"),
    "c_coating_ar": (scene_c_coating_ar, "deterministic"),         # phase B
    "c_polarizer_malus": (scene_c_polarizer_malus, "deterministic"),
    "c_torus": (scene_c_torus, "statistical"),                     # phase B
    "c_grating": (scene_c_grating, "deterministic"),               # phase E
    "c_rough_plate": (scene_c_rough_plate, "statistical"),         # phase E
}

# Real extracted geometries (repo geometry/ dirs) that become routable as
# phases land — used by test_cengine_parity's real-geometry test. All are
# statistical (curved surfaces / broadband).
REAL_SCENES = {
    "lens_ball": "statistical",            # sphere + cylinder + planes (B)
    "lens_pcx": "statistical",             # plano-convex singlet (B)
    "filter_bandpass": "statistical",      # filter slab + cyl housing (B)
    "axicon_pcx": "statistical",           # CONE surface (B)
    "lens_asphere": "statistical",         # ASPHERE surface (B)
    "lens_fresnel": "statistical",         # cone facets (B)
    "prism_equilateral": "statistical",    # TIR path (B)
    "hot_mirror": "statistical",           # dielectric stack coating (B)
    "pol_crossed": "statistical",          # crossed polarizers ~zero T (B)
    "lens_achromat": "statistical",        # two glasses + 5um air gap (B)
    "ghost_doublet": "statistical",        # multi-bounce ghosts (B)
    "mesh_freeform": "statistical",        # tessellated face -> BLAS (C)
    "scatter_plate": "statistical",        # ABg measured scatter (E)
    "bench": "statistical",                # diffuser + coherent source (E)
    "calcite_displacer": "statistical",    # uniaxial o/e walk-off (F)
    # NOTE: the retarder parity scene is MgF2, not quartz — quartz carries
    # gyration data (optical activity) and gyrotropic scenes reference-route
    # to Python by design (the C engine has no rotation term). waveplate_mgf2
    # is the same m=30 HWP physics with a gyration-free uniaxial, keeping
    # C-engine coverage of uniaxial retardance. test_routing_reasons pins the
    # quartz scene's python routing.
    "waveplate_mgf2": "statistical",       # MgF2 retarder + polarizer (F)
    "wollaston": "statistical",            # crossed calcite prisms (F)
    "pol_circular": "statistical",         # circular polarizer chain (F)
}


def write_scene(name, root):
    """Write geometry/<name>/model.json under root; returns the model.json
    path (run_trace's --model-json input)."""
    builder, _ = SCENES[name]
    gdir = Path(root) / name
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "model.json").write_text(json.dumps(builder()))
    return gdir / "model.json"
