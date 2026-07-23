#!/usr/bin/env python3
# =============================================================================
# primitivelib.py — parametric single-element primitive builders for the
# MieWorkbench library ("add element" wizards / primitives/*.FCStd).
#
# Interpreter: the FreeCAD AppImage's embedded python for BUILDING; the
# module-level PRIMITIVES metadata dict is importable WITHOUT FreeCAD
# (guarded imports, same pattern as make_test_scenes.py) so the GUI can
# list primitives, parameter specs and defaults under plain python.
#
# Design: each primitive .FCStd contains
#   - one 'dim'-labeled Spreadsheet with one aliased cell per GEOMETRY
#     parameter (raw "=<val> mm" / "=<val> deg" / bare number for counts);
#   - one (or, for achromat/pbs_cube, two) PartDesign::Body built FROM
#     those parameter values, tagged with the usual Base contract props
#     (material/power/lambdac/... per README §5) plus:
#       miewb_primitive : str  — the PRIMITIVES kind that built it
#       miewb_group     : str  — shared by all bodies of one element
#     so the GUI's element editor knows how to rebuild it.
#
# WHY REBUILD-ON-EDIT (not constraint expressions): parameter changes can
# change TOPOLOGY (R -> flat surface, facet counts, sign flips pcx<->pcv),
# which no constraint expression can do. So the spreadsheet is the single
# source of truth and fcserver's 'rebuild_primitive' op re-runs the builder
# with the current alias values, preserving Label/props/Placement. Hand-
# authored user primitives with real cell expressions keep working through
# the ordinary set_spreadsheet -> recompute path instead.
#
# Reuses make_test_scenes.py's proven geometry helpers (lens_meridian,
# revolve_body, pad_body, new_body_pad, ...).
# =============================================================================
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os

# make_test_scenes runs its main() at module scope under FreeCAD (the -c
# no-__main__-guard convention); this flag tells it we only want its
# geometry helpers, not a scene build.
os.environ.setdefault("MIEWB_MTS_LIBRARY_ONLY", "1")

try:
    import FreeCAD as App
    import Part
    import make_test_scenes as mts
    _HAVE_FREECAD = True
except Exception:               # metadata-only import path (plain python)
    App = None
    Part = None
    mts = None
    _HAVE_FREECAD = False


def P(default, unit, help_text):
    return {"default": default, "unit": unit, "help": help_text}


# ---------------------------------------------------------------------------
# Primitive registry (importable without FreeCAD)
# ---------------------------------------------------------------------------
PRIMITIVES = {
    # -- sources ------------------------------------------------------------
    "laser_collimated": {
        "category": "Sources", "label": "Collimated laser",
        "tooltip": "Cylindrical (or, if round_flag=0, boxy) housing "
                   "emitting a collimated beam from its flat +x end cap.",
        "params": {"diameter": P(10.0, "mm", "beam (emit face) diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 5.0, "lambdac": 633.0, "coherent": True},
    },
    "laser_divergent": {
        "category": "Sources", "label": "Divergent laser",
        "tooltip": "Laser with a convex spherical emit cap: emitted rays "
                   "diverge from a virtual point (radius of curvature = "
                   "roc). With round_flag=0, a flat rectangular emit face "
                   "is built instead (roc does not apply).",
        "params": {"diameter": P(10.0, "mm", "emit aperture diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "roc": P(200.0, "mm", "emit cap radius of curvature "
                                         "(divergence = diameter/2/roc); "
                                         "only applies when round_flag=1"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular (spherical emit "
                                          "cap), 0 = flat rectangular "
                                          "emitter")},
        "props": {"power": 5.0, "lambdac": 633.0, "coherent": True},
    },
    "source_broadband": {
        "category": "Sources", "label": "Broadband source",
        "tooltip": "Incoherent broadband disc (or box) emitter (set "
                   "lambdamin / lambdamax in the properties). For a "
                   "MEASURED emission shape, set the `spectrum` property "
                   "to an opticalproperties/emission registry row instead "
                   "(e.g. led_white_2733k -- or import the led_white "
                   "primitive, which is exactly that).",
        "params": {"diameter": P(10.0, "mm", "emit face diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 5.0, "lambdac": 550.0,
                  "lambdamin": 450.0, "lambdamax": 650.0,
                  "coherent": False},
    },
    # -- LED presets (monochromatic-LED library_data/emission_led_monochromatic.csv;
    #    lambdamin/lambdamax = cwl -+ FWHM/2.3548 so the existing Gaussian
    #    source reproduces each LED's datasheet FWHM -- see sources.py:19-22) --
    "led_deep_red_660": {
        "category": "Sources", "label": "Deep-red LED (660 nm)",
        "tooltip": "Deep-red monochromatic LED source (CWL 660 nm, FWHM "
                   "20 nm; Lumileds LUXEON Z DS105 Table 1a/2 (LXZ1-PA01); "
                   "Cree XP-E2 CLD-DS56 Photo Red 650-670nm bins).",
        "params": {"diameter": P(10.0, "mm", "emit face diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 5.0, "lambdac": 660.0,
                  "lambdamin": 651.51, "lambdamax": 668.49,
                  "coherent": False},
    },
    "led_red_630": {
        "category": "Sources", "label": "Red LED (625 nm)",
        "tooltip": "Red monochromatic LED source (CWL 625 nm, FWHM 20 nm; "
                   "Cree XP-E2 CLD-DS56 R2-R3 (620-630nm); FWHM Lumileds "
                   "DS105 (LXZ1-PD01)).",
        "params": {"diameter": P(10.0, "mm", "emit face diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 5.0, "lambdac": 625.0,
                  "lambdamin": 616.51, "lambdamax": 633.49,
                  "coherent": False},
    },
    "led_amber_590": {
        "category": "Sources", "label": "Amber LED (590 nm)",
        "tooltip": "Amber monochromatic LED source (CWL 590 nm, FWHM "
                   "20 nm; Cree XP-E2 CLD-DS56 A2-A3 (585-595nm); FWHM "
                   "Lumileds DS105 (LXZ1-PL03)).",
        "params": {"diameter": P(10.0, "mm", "emit face diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 5.0, "lambdac": 590.0,
                  "lambdamin": 581.51, "lambdamax": 598.49,
                  "coherent": False},
    },
    "led_green_525": {
        "category": "Sources", "label": "Green LED (527 nm)",
        "tooltip": "Green monochromatic LED source (CWL 527 nm, FWHM "
                   "30 nm; Cree XP-E2 CLD-DS56 G2-G4 (520-535nm); FWHM "
                   "Lumileds DS105 (LXZ1-PM01)).",
        "params": {"diameter": P(10.0, "mm", "emit face diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 5.0, "lambdac": 527.0,
                  "lambdamin": 514.26, "lambdamax": 539.74,
                  "coherent": False},
    },
    "led_blue_470": {
        "category": "Sources", "label": "Blue LED (472 nm)",
        "tooltip": "Blue monochromatic LED source (CWL 472 nm, FWHM "
                   "20 nm; Cree XP-E2 CLD-DS56 B3-B6 (465-485nm); FWHM "
                   "Lumileds DS105 (LXZ1-PB01)).",
        "params": {"diameter": P(10.0, "mm", "emit face diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 5.0, "lambdac": 472.0,
                  "lambdamin": 463.51, "lambdamax": 480.49,
                  "coherent": False},
    },
    "led_royal_blue_450": {
        "category": "Sources", "label": "Royal-blue LED (452 nm)",
        "tooltip": "Royal-blue monochromatic LED source (CWL 452 nm, "
                   "FWHM 20 nm; Cree XP-E2 CLD-DS56 D3-D5 (450-465nm); "
                   "FWHM Lumileds DS105 (LXZ1-PR01)).",
        "params": {"diameter": P(10.0, "mm", "emit face diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 5.0, "lambdac": 452.0,
                  "lambdamin": 443.51, "lambdamax": 460.49,
                  "coherent": False},
    },
    "led_uv_365": {
        "category": "Sources", "label": "UV LED (365 nm)",
        "tooltip": "365 nm UV monochromatic LED source (CWL 365 nm, "
                   "FWHM 9.0 nm; Nichia NCSU276CT-E datasheet "
                   "lambda_p=365nm dlambda=9.0nm, grep-verified PDF).",
        "params": {"diameter": P(10.0, "mm", "emit face diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 5.0, "lambdac": 365.0,
                  "lambdamin": 361.18, "lambdamax": 368.82,
                  "coherent": False},
    },
    "led_uv_385": {
        "category": "Sources", "label": "UV LED (385 nm)",
        "tooltip": "385 nm UV monochromatic LED source (CWL 385 nm, "
                   "FWHM 11 nm; Nichia NVSU233A-D1-E U385 bin; "
                   "UNVERIFIED-lite -- FWHM secondary-sourced, not "
                   "PDF-verified).",
        "params": {"diameter": P(10.0, "mm", "emit face diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 5.0, "lambdac": 385.0,
                  "lambdamin": 380.33, "lambdamax": 389.67,
                  "coherent": False},
    },
    "led_white": {
        "category": "Sources", "label": "White LED (2733 K)",
        "tooltip": "Phosphor-converted warm-white LED source with a full "
                   "tabulated emission spectrum (blue pump peak ~450 nm + "
                   "broad phosphor hump; CIE 015:2018 Table 12.1 standard "
                   "illuminant LED-B1, CCT ~2733 K). Samples the SPD table "
                   "(spectrum=led_white_2733k); lambdac is the power-weighted "
                   "mean wavelength.",
        "params": {"diameter": P(10.0, "mm", "emit face diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        # lambdac = 584.6 nm: power-weighted mean lambda of the LED-B1 table
        # (trapezoid integral of lambda*P over the piecewise-linear PDF).
        # Required by classify_body (power+lambdac mark a source); the
        # `spectrum` table supersedes it for actual wavelength sampling.
        "props": {"power": 5.0, "lambdac": 584.6,
                  "spectrum": "led_white_2733k", "coherent": False},
    },
    # -- pulsed lasers (pulsed-optics P12; pulse props per the P3 power
    #    XOR pulse_energy contract — sources with pulse_duration auto-
    #    enable the time products. Data: library_data_pinned.md research
    #    notes 2026-07-11, datasheet-exact unless noted) ------------------
    "laser_pulsed": {
        "category": "Sources", "label": "Pulsed laser (generic)",
        "tooltip": "Generic pulsed laser: 10 nJ / 100 fs FWHM / 80 MHz at "
                   "800 nm (a typical fs oscillator). Edit pulse_energy / "
                   "pulse_duration / rep_rate freely — power stays unset "
                   "(pulse_energy x rep_rate defines average power).",
        "params": {"diameter": P(2.0, "mm", "beam (emit face) diameter "
                                            "(circular) or edge length "
                                            "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"lambdac": 800.0, "lambdamin": 796.0,
                  "lambdamax": 804.0, "pulse_energy": 0.01,
                  "pulse_duration": 0.1, "rep_rate": 8e7,
                  "coherent": True},
    },
    "laser_maitai_800": {
        "category": "Sources", "label": "Ti:sapphire fs (Mai Tai HP)",
        "tooltip": "Spectra-Physics Mai Tai HP Ti:sapphire oscillator at "
                   "its 800 nm tuning peak: 2.5 W average, <100 fs sech2 "
                   "(modelled Gaussian), 80 MHz, 1.2 mm beam, linear "
                   "polarization >500:1 (MKS/Spectra-Physics Mai Tai "
                   "datasheet rev 6/26). Energy/pulse derives as "
                   "power/rep_rate = 31 nJ.",
        "params": {"diameter": P(1.2, "mm", "1/e^2 beam diameter at "
                                            "800 nm (datasheet <1.2)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 2500.0, "lambdac": 800.0,
                  "lambdamin": 796.0, "lambdamax": 804.0,
                  "pulse_duration": 0.1, "rep_rate": 8e7,
                  "polarization": "linear:0", "coherent": True},
    },
    "laser_erfiber_1560": {
        "category": "Sources", "label": "Er-fiber fs (1560 nm)",
        "tooltip": "TOPTICA FemtoFiber pro IR/NIR Er-fiber oscillator: "
                   "350 mW average, <100 fs, 80 MHz, 3.5 mm beam at "
                   "1560 nm (TOPTICA manual M-043 v07; polarization ratio "
                   "at 1560 nm estimated from the 780 nm SHG spec). "
                   "Energy/pulse derives as 4.4 nJ.",
        "params": {"diameter": P(3.5, "mm", "1/e^2 beam diameter at "
                                            "1560 nm"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 350.0, "lambdac": 1560.0,
                  "lambdamin": 1526.0, "lambdamax": 1594.0,
                  "pulse_duration": 0.1, "rep_rate": 8e7,
                  "polarization": "linear:0", "coherent": True},
    },
    "laser_ndyag_1064": {
        "category": "Sources", "label": "Q-switched Nd:YAG (850 mJ)",
        "tooltip": "Quantel/Lumibird Q-smart 850 flashlamp Nd:YAG: "
                   "850 mJ/pulse, ~6 ns FWHM, 10 Hz, 9 mm beam, "
                   "horizontal polarization >80% (Quantel datasheet "
                   "10-32-2667 rev 05/14; linewidth <=0.7 cm^-1 — "
                   "modelled monochromatic). Average power derives as "
                   "8.5 W; peak power 133 MW.",
        "params": {"diameter": P(9.0, "mm", "beam diameter at output"),
                   "length": P(15.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"lambdac": 1064.0, "pulse_energy": 850000.0,
                  "pulse_duration": 6000.0, "rep_rate": 10.0,
                  "polarization": "linear:0", "coherent": True},
    },
    "sc_superk": {
        "category": "Sources", "label": "Supercontinuum (SuperK EXR-20)",
        "tooltip": "NKT SuperK EXTREME EXR-20 supercontinuum: 8 W total "
                   "over the tabulated 400-2400 nm SPD "
                   "(spectrum=sc_superk, digitized +-20-30% from the Jan "
                   "2011 datasheet; 1064 nm residual pump spike clipped), "
                   "~5 ps seed pulses at 80 MHz, single-mode, unpolarized "
                   "(M2<1.1). lambdac is the power-weighted mean of the "
                   "table. Clip the SPD to your bench's material window "
                   "(docs/RAYTRACER.md).",
        "params": {"diameter": P(1.5, "mm", "collimated beam diameter "
                                            "(datasheet: ~1 mm @530 nm to "
                                            "~3 mm @2000 nm; single "
                                            "mid-band value)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 8000.0, "lambdac": 1200.0,
                  "spectrum": "sc_superk", "pulse_duration": 5.0,
                  "rep_rate": 8e7, "coherent": False},
    },
    "fiber_nonlinear_output": {
        "category": "Sources", "label": "Nonlinear-fiber output (SPM)",
        "tooltip": "Er-fiber fs laser launched through 2 cm of highly-"
                   "nonlinear fiber (OFS/Lightera HNLF, gamma = 11.5 "
                   "W^-1km^-1): the source-side SPM transform (spm "
                   "property) installs the exact multi-peak self-phase-"
                   "modulated spectrum (phi_max ~ 9.5 rad here) and the "
                   "S-curve chirp. Edit the spm property "
                   "('gamma:<W^-1km^-1>:length:<m>' or 'phimax:<rad>') "
                   "to change the broadening.",
        "params": {"diameter": P(3.5, "mm", "collimated output beam "
                                            "diameter"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 350.0, "lambdac": 1560.0,
                  "pulse_duration": 0.1, "rep_rate": 8e7,
                  "spm": "gamma:11.5:length:0.02",
                  "polarization": "linear:0", "coherent": True},
    },
    # -- detectors ----------------------------------------------------------
    "detector_plane": {
        "category": "Detectors", "label": "Detector plane",
        "tooltip": "Square (or round) thin-screen detector; its -x face "
                   "records irradiance. Transparent to the beam.",
        "params": {"width": P(30.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "height": P(0.0, "mm", "rectangular height; 0 = square "
                                          "(width used for both edges); "
                                          "ignored when round_flag=1 -- "
                                          "e.g. 36 x 24 for a full-frame "
                                          "CMOS sensor"),
                   "thickness": P(1.0, "mm", "screen thickness"),
                   "round_flag": P(0, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "detector"},
    },
    # -- spherical lenses (revolved meridians) -------------------------------
    "lens_pcx": {
        "category": "Lenses", "label": "Plano-convex lens",
        "tooltip": "Convex R1 toward -x, flat back.",
        "params": {"R_front": P(25.0, "mm", "front radius of curvature (>0)"),
                   "ct": P(5.0, "mm", "center thickness"),
                   "aperture": P(20.0, "mm", "clear aperture diameter")},
        "props": {"material": "bk7"},
        "meridian": lambda p: (p["R_front"], None),
    },
    "lens_dcx": {
        "category": "Lenses", "label": "Biconvex lens",
        "tooltip": "Convex both sides (R1 front, -R2 back).",
        "params": {"R_front": P(40.0, "mm", "front radius (>0)"),
                   "R_back": P(40.0, "mm", "back radius magnitude (>0)"),
                   "ct": P(6.0, "mm", "center thickness"),
                   "aperture": P(20.0, "mm", "clear aperture diameter")},
        "props": {"material": "bk7"},
        "meridian": lambda p: (p["R_front"], -p["R_back"]),
    },
    "lens_pcv": {
        "category": "Lenses", "label": "Plano-concave lens",
        "tooltip": "Flat front, concave back — diverging.",
        "params": {"R_back": P(25.0, "mm", "back radius of curvature (>0)"),
                   "ct": P(3.0, "mm", "center thickness"),
                   "aperture": P(20.0, "mm", "clear aperture diameter")},
        "props": {"material": "bk7"},
        "meridian": lambda p: (None, p["R_back"]),
    },
    "lens_dcv": {
        "category": "Lenses", "label": "Biconcave lens",
        "tooltip": "Concave both sides — diverging.",
        "params": {"R_front": P(40.0, "mm", "front radius magnitude (>0)"),
                   "R_back": P(40.0, "mm", "back radius (>0)"),
                   "ct": P(3.0, "mm", "center thickness"),
                   "aperture": P(20.0, "mm", "clear aperture diameter")},
        "props": {"material": "bk7"},
        "meridian": lambda p: (-p["R_front"], p["R_back"]),
    },
    "lens_meniscus": {
        "category": "Lenses", "label": "Meniscus lens",
        "tooltip": "Both surfaces curve the same way (R1, R2 same sign).",
        "params": {"R_front": P(20.0, "mm", "front radius (signed)"),
                   "R_back": P(40.0, "mm", "back radius (signed)"),
                   "ct": P(4.0, "mm", "center thickness"),
                   "aperture": P(18.0, "mm", "clear aperture diameter")},
        "props": {"material": "bk7"},
        "meridian": lambda p: (p["R_front"], p["R_back"]),
    },
    "lens_ball": {
        "category": "Lenses", "label": "Ball lens",
        "tooltip": "Full sphere.",
        "params": {"diameter": P(8.0, "mm", "sphere diameter")},
        "props": {"material": "bk7"},
    },
    "lens_rod": {
        "category": "Lenses", "label": "Rod lens",
        "tooltip": "Cylinder rod (axis along z) — cylinder lens in x-y.",
        "params": {"diameter": P(8.0, "mm", "rod diameter"),
                   "length": P(20.0, "mm", "rod length along z")},
        "props": {"material": "bk7"},
    },
    "lens_cyl": {
        "category": "Lenses", "label": "Cylindrical lens",
        "tooltip": "Plano-convex (R>0) or plano-concave (R<0) cylinder "
                   "lens, cylinder axis along z: line focus.",
        "params": {"R": P(25.0, "mm", "front radius (signed: <0 concave)"),
                   "ct": P(5.0, "mm", "center thickness"),
                   "aperture": P(20.0, "mm", "aperture in y (diameter)"),
                   "height": P(20.0, "mm", "extent along z")},
        "props": {"material": "bk7"},
    },
    "lens_asphere": {
        "category": "Lenses", "label": "Aspheric lens",
        "tooltip": "Plano-convex even asphere, conic + A4 r^4 (revolved "
                   "exact-sag BSpline + surface_override; extractor "
                   "verifies <1 um). Defaults are the f=40 BK7@633 "
                   "design solved for the COMPLETE lens (front asphere + "
                   "flat exit) by exact ray trace — the old k=-n^2 "
                   "convention corrected only the front surface and "
                   "over-corrected the full lens.",
        "params": {"R": P(20.6033, "mm", "vertex radius of curvature"),
                   "k": P(-1.0, "", "conic constant (full-lens-corrected "
                                    "design; wizards.solve_asphere "
                                    "supplies (k, A4) per focal length)"),
                   # alias must not look like a cell address (A4 = cell!)
                   "A4_mm3": P(6.586562e-06, "", "even-asphere A4 "
                                                 "coefficient, mm^-3"),
                   "ct": P(6.0, "mm", "center thickness"),
                   "aperture": P(20.0, "mm", "clear aperture diameter")},
        "props": {"material": "bk7"},
        # builder-owned, re-derived from R/k/A4/aperture every rebuild
        "derived_props": ("surface_override",),
    },
    "lens_fresnel": {
        "category": "Lenses", "label": "Fresnel lens",
        "tooltip": "Collapsed plano-convex lens: annular conical facets "
                   "matching the ideal thin-lens local slope.",
        "params": {"aperture": P(24.0, "mm", "clear aperture diameter"),
                   "f_design": P(50.0, "mm", "design focal length"),
                   "n_design": P(1.51508, "", "design refractive index"),
                   "n_facets": P(12.0, "", "number of annular facets"),
                   "back": P(2.0, "mm", "substrate thickness behind the "
                                        "deepest facet")},
        "props": {"material": "bk7"},
    },
    "lens_achromat": {
        "category": "Lenses", "label": "Achromatic doublet",
        "tooltip": "Cemented crown+flint doublet (modeled with a 5 um "
                   "air gap at the interface). Two bodies.",
        "params": {"R_front": P(31.0, "mm", "crown front radius"),
                   "R_iface": P(-21.956, "mm", "cemented interface radius "
                                               "(signed)"),
                   "R_back": P(-64.497, "mm", "flint back radius (signed)"),
                   "ct_crown": P(6.0, "mm", "crown center thickness"),
                   "ct_flint": P(3.0, "mm", "flint center thickness"),
                   "gap": P(0.005, "mm", "interface air gap"),
                   "aperture": P(18.0, "mm", "clear aperture diameter")},
        "props": {},   # per-body materials set by the builder
    },
    # -- other refractives ----------------------------------------------------
    "axicon": {
        "category": "Lenses", "label": "Axicon",
        "tooltip": "Conical front (apex toward -x), flat base: turns a "
                   "beam into a ring / Bessel zone.",
        "params": {"base_angle": P(10.0, "deg", "cone base angle"),
                   "aperture": P(22.0, "mm", "base diameter")},
        "props": {"material": "bk7"},
    },
    "prism": {
        "category": "Prisms & Mirrors", "label": "Equilateral prism",
        "tooltip": "Equilateral dispersing prism, apex up (+y), length "
                   "along z.",
        "params": {"side": P(25.0, "mm", "triangle side length"),
                   "height": P(25.0, "mm", "extent along z"),
                   "rotation": P(0.0, "deg", "rotation about z")},
        "props": {"material": "bk7"},
    },
    "mirror_flat": {
        "category": "Prisms & Mirrors", "label": "Flat mirror",
        "tooltip": "Aluminum plate; combine with a 'mirror' or coating "
                   "property for partial reflectors.",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(3.0, "mm", "plate thickness"),
                   "round_flag": P(0, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "aluminum"},
    },
    "window": {
        "category": "Plates & Filters", "label": "Optical window",
        "tooltip": "Plane-parallel plate.",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(3.0, "mm", "plate thickness"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7"},
    },
    "polarizer_plate": {
        "category": "Polarization", "label": "Polarizer",
        "tooltip": "Linear polarizer plate (registry item + body-local "
                   "transmission axis).",
        "params": {"width": P(20.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(1.0, "mm", "plate thickness"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7", "polarizer": "ideal_linear",
                  "polarizer_axis": "0,0,1"},
    },
    "waveplate": {
        "category": "Polarization", "label": "Waveplate (quartz)",
        "tooltip": "Uniaxial quartz retarder; retardance set by thickness "
                   "and crystal_axis.",
        "params": {"width": P(16.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(0.0298, "mm", "plate thickness (sets "
                                                "retardance)"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "quartz", "crystal_axis": "0,0,1"},
    },
    "filter_plate": {
        "category": "Plates & Filters", "label": "Spectral filter",
        "tooltip": "Bulk (Beer-Lambert) spectral filter plate; pick the "
                   "filter registry item in the properties.",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(3.0, "mm", "plate thickness"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7", "filter": "bp_550_40"},
    },
    "grating_plate": {
        "category": "Plates & Filters", "label": "Diffraction grating",
        "tooltip": "Plate whose front (-x) face carries a grating spec "
                   "(default 600 l/mm vertical grooves; edit the "
                   "'grating' property or use the wizard).",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(3.0, "mm", "plate thickness"),
                   "round_flag": P(0, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7", "grating": "Face1=600:v"},
    },
    "pbs_cube": {
        "category": "Polarization", "label": "PBS cube",
        "tooltip": "Polarizing beamsplitter: a single BK7 cube with a "
                   "thin pbs_visible_45-coated plate NESTED inside along "
                   "the diagonal (glass-glass interface; s-pol reflects, "
                   "p-pol transmits -- the old two-prism air-gap build "
                   "TIR'd the transmitted arm at 45 deg). Two bodies "
                   "(cube + splitter plate).",
        "params": {"cube": P(20.0, "mm", "cube edge length"),
                   "height": P(20.0, "mm", "extent along z"),
                   "plate_ct": P(0.2, "mm", "internal splitter plate "
                                            "thickness")},
        "props": {},   # per-body props set by the builder
        "derived_props": ("coating",),
    },
    # =========================================================================
    # Batch 1 -- plate-likes (all on _build_plate/_build_wedge_plate)
    # =========================================================================
    "bs_plate": {
        "category": "Beamsplitters", "label": "Beamsplitter plate (50:50)",
        "tooltip": "Non-polarizing beamsplitter plate; place at 45 deg AOI. "
                   "The front (-x) face carries the beamsplitter coating "
                   "(default bs_5050_vis_45 -- swap the 'coating' property "
                   "for any bs_XXYY_vis_45 registry row to change the "
                   "split ratio). wedge_deg tilts the back face slightly "
                   "to kill parallel-plate etalon fringes (0 = "
                   "plane-parallel).",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(3.0, "mm", "plate thickness (center, if "
                                             "wedged)"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular"),
                   "wedge_deg": P(0.5, "deg", "back-face wedge angle "
                                              "(anti-etalon); 0 = "
                                              "plane-parallel")},
        "props": {"material": "bk7"},   # coating (front face) set by builder
    },
    "pbs_plate": {
        "category": "Beamsplitters", "label": "Polarizing beamsplitter plate",
        "tooltip": "Plate-form PBS; place at 45 deg AOI. Front (-x) face "
                   "carries the pbs_visible_45 coating.",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(3.0, "mm", "plate thickness"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7"},   # coating (front face) set by builder
    },
    "dichroic_plate": {
        "category": "Beamsplitters", "label": "Dichroic beamsplitter plate",
        "tooltip": "Longpass dichroic (default dichroic_567lp_45, cut-on "
                   "567nm); place at 45 deg AOI. Swap the 'coating' "
                   "property to hot_mirror_45 or cold_mirror_45 for a "
                   "vendor-style hot/cold mirror instead.",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(1.0, "mm", "plate thickness"),
                   "round_flag": P(0, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7"},   # coating (front face) set by builder
    },
    "pellicle": {
        "category": "Beamsplitters", "label": "Pellicle beamsplitter",
        "tooltip": "Ultra-thin (few-micron) nitrocellulose membrane "
                   "beamsplitter -- negligible ghosting, no wedge/etalon "
                   "concerns. Front (-x) face carries the pellicle_4555_45 "
                   "coating (swap to pellicle_uncoated_45 for a bare "
                   "membrane).",
        "params": {"diameter": P(25.0, "mm", "membrane clear aperture "
                                             "diameter"),
                   "membrane_thickness": P(0.002, "mm",
                                           "membrane thickness (2 um, "
                                           "typical pellicle film)")},
        "props": {"material": "bk7"},   # coating (front face) set by builder
    },
    "nd_filter": {
        "category": "Filters", "label": "ND filter (absorptive)",
        "tooltip": "Bulk (Beer-Lambert) absorptive neutral-density plate. "
                   "Swap the 'filter' property to any nd_odXX registry row "
                   "for other densities (OD scales with thickness from a "
                   "2mm reference).",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(2.0, "mm", "plate thickness"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7", "filter": "nd_od10"},
    },
    "nd_reflective": {
        "category": "Filters", "label": "ND filter (reflective)",
        "tooltip": "Metallic (Inconel-style) reflective neutral-density "
                   "plate; front (-x) face carries the nd_refl_od10 "
                   "coating (swap for other nd_refl_odXX rows).",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(3.0, "mm", "plate thickness"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7"},   # coating (front face) set by builder
    },
    "filter_bandpass": {
        "category": "Filters", "label": "Bandpass filter",
        "tooltip": "Bandpass spectral filter plate (default bp_550_40: "
                   "CWL=550nm, FWHM=40nm).",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(3.0, "mm", "plate thickness"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7", "filter": "bp_550_40"},
    },
    "filter_longpass": {
        "category": "Filters", "label": "Longpass filter",
        "tooltip": "Longpass spectral filter plate (default longpass_600: "
                   "cut-on 600nm).",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(3.0, "mm", "plate thickness"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7", "filter": "longpass_600"},
    },
    "filter_shortpass": {
        "category": "Filters", "label": "Shortpass filter",
        "tooltip": "Shortpass spectral filter plate (default "
                   "shortpass_600: cut-off 600nm).",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(3.0, "mm", "plate thickness"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7", "filter": "shortpass_600"},
    },
    "filter_notch": {
        "category": "Filters", "label": "Notch filter",
        "tooltip": "Narrow rejection-band spectral filter plate (default "
                   "notch_633_25: OD4 notch at 633nm, 25nm FWHM).",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(3.0, "mm", "plate thickness"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7", "filter": "notch_633_25"},
    },
    "window_wedged": {
        "category": "Plates & Filters", "label": "Wedged window",
        "tooltip": "Plane-wedge window: back face tilted by wedge_deg "
                   "(thickness increases toward +y) to walk stray "
                   "reflections off-axis and kill etalon fringes.",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(5.0, "mm", "center thickness"),
                   "wedge_deg": P(0.5, "deg", "back-face wedge angle"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7"},
    },
    "diffuser_plate": {
        "category": "Diffusers", "label": "Ground-glass diffuser",
        "tooltip": "Ground-glass diffuser plate; the exit (+x) face "
                   "carries the ground-surface scatter (default @dg_600, "
                   "600-grit). Swap the 'diffuser' property's registry "
                   "reference for @dg_120/220/1500 (coarser -> finer grit, "
                   "narrower -> wider scatter angle). Never combine with "
                   "a 'roughness' spec on the same face.",
        "params": {"width": P(25.0, "mm", "edge length (rectangular) or "
                                          "diameter (circular)"),
                   "thickness": P(2.0, "mm", "plate thickness"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"material": "bk7"},   # diffuser (exit face) set by builder
    },
    # =========================================================================
    # Batch 2 -- prisms / mirrors / apertures
    # =========================================================================
    "prism_right_angle": {
        "category": "Prisms & Mirrors", "label": "Right-angle prism",
        "tooltip": "45-45-90 right-angle prism; the hypotenuse TIRs the "
                   "beam through 90 deg (no coating needed at 45 deg AOI "
                   "in bk7).",
        "params": {"leg": P(25.0, "mm", "leg length"),
                   "height": P(25.0, "mm", "extent along z")},
        "props": {"material": "bk7"},
    },
    "prism_wedge": {
        "category": "Prisms & Mirrors", "label": "Wedge prism",
        "tooltip": "Round wedge prism (angular beam deviation without a "
                   "reflection); shares its wedged-back-face construction "
                   "with window_wedged.",
        "params": {"diameter": P(25.0, "mm", "clear aperture diameter"),
                   "thickness": P(5.0, "mm", "center thickness"),
                   "wedge_deg": P(2.0, "deg", "wedge angle")},
        "props": {"material": "bk7"},
    },
    "prism_dove": {
        "category": "Prisms & Mirrors", "label": "Dove prism",
        "tooltip": "Trapezoidal Dove prism: beam enters/exits through the "
                   "45-deg end faces and TIRs once off the long bottom "
                   "face; rotating the prism about the beam axis rotates "
                   "the image at twice the rate (image-rotation prism).",
        "params": {"aperture": P(20.0, "mm", "clear aperture (height and "
                                             "width)"),
                   "length": P(80.0, "mm", "overall length")},
        "props": {"material": "bk7"},
    },
    "prism_penta": {
        "category": "Prisms & Mirrors", "label": "Penta prism",
        "tooltip": "Pentaprism: deviates the beam by exactly 90 deg "
                   "regardless of prism orientation, without image "
                   "reversal. The two reflecting faces do NOT satisfy TIR "
                   "in bk7 at their working angle of incidence, so they "
                   "carry a real metallic mirror coating (Al_mirror_bare) "
                   "-- set by the builder from the actual face normals.",
        "params": {"aperture": P(20.0, "mm", "clear aperture (entrance/exit "
                                             "face height)")},
        "props": {"material": "bk7"},   # reflecting-face coating set by builder
    },
    "prism_rhomboid": {
        "category": "Prisms & Mirrors", "label": "Rhomboid prism",
        "tooltip": "Displaces the beam laterally while preserving its "
                   "direction and orientation (unlike a periscope pair, "
                   "one solid part): two parallel 45-deg TIR faces (no "
                   "coating needed in bk7).",
        "params": {"aperture": P(20.0, "mm", "clear aperture (entrance/exit "
                                             "face height)"),
                   "length": P(60.0, "mm", "horizontal extent (sets the "
                                           "lateral displacement)")},
        "props": {"material": "bk7"},
    },
    "mirror_concave": {
        "category": "Prisms & Mirrors", "label": "Concave mirror",
        "tooltip": "Front-surface spherical concave mirror (converging); "
                   "aluminum, same front-surface-metal convention as "
                   "mirror_flat.",
        "params": {"R": P(100.0, "mm", "concave curvature radius (>0)"),
                   "aperture": P(25.0, "mm", "clear aperture diameter"),
                   "ct": P(4.0, "mm", "edge/substrate thickness")},
        "props": {"material": "aluminum"},
    },
    "mirror_convex": {
        "category": "Prisms & Mirrors", "label": "Convex mirror",
        "tooltip": "Front-surface spherical convex mirror (diverging); "
                   "aluminum, same front-surface-metal convention as "
                   "mirror_flat.",
        "params": {"R": P(100.0, "mm", "convex curvature radius (>0)"),
                   "aperture": P(25.0, "mm", "clear aperture diameter"),
                   "ct": P(4.0, "mm", "edge/substrate thickness")},
        "props": {"material": "aluminum"},
    },
    "mirror_d_shaped": {
        "category": "Prisms & Mirrors", "label": "D-shaped mirror",
        "tooltip": "Circular mirror with one flat edge (a chord) for "
                   "close beam-packing; cut_offset=0 gives a true half-"
                   "circle, positive values shift the flat edge toward "
                   "+y (leaving more than half the disc).",
        "params": {"diameter": P(25.0, "mm", "full-circle diameter before "
                                             "the flat cut"),
                   "thickness": P(3.0, "mm", "mirror thickness"),
                   "cut_offset": P(0.0, "mm", "distance of the flat edge "
                                              "from the disc center along "
                                              "+y (0 = through center)")},
        "props": {"material": "aluminum"},
    },
    "iris": {
        "category": "Apertures", "label": "Iris (circular stop)",
        "tooltip": "Blackened circular aperture stop: an opaque annular "
                   "disc plus a thin material=air 'plug' filling the "
                   "opening (the aperture contract, docs/RAYTRACER.md "
                   "S5.10) so the coherent gather re-anchors correctly at "
                   "the aperture plane. 'blackness' (0.95-1.0) sets the "
                   "disc's absorbance property directly -- edit blackness "
                   "and rebuild to change it (a manual 'absorbance' edit "
                   "would be overwritten by the next rebuild, since it is "
                   "re-derived from blackness every time).",
        "params": {"outer_diameter": P(25.0, "mm", "disc outer diameter"),
                   "thickness": P(1.0, "mm", "disc thickness"),
                   "hole_diameter": P(5.0, "mm", "clear aperture (hole) "
                                                 "diameter"),
                   "blackness": P(0.98, "", "fraction of incident power "
                                           "absorbed by the disc "
                                           "(0.95-1.0 typical)")},
        "props": {},   # disc: material/absorbance; plug: material=air --
                       # both set by the builder
        "derived_props": ("absorbance",),
    },
    "iris_bladed": {
        "category": "Apertures", "label": "Iris (N-blade polygon stop)",
        "tooltip": "Blackened N-blade iris diaphragm: an opaque disc with a "
                   "true REGULAR-POLYGON opening (the straight blade edges) "
                   "plus a thin material=air plug filling the polygon (the "
                   "aperture contract, docs/RAYTRACER.md S5.10). The polygon "
                   "-- not a circle -- is the whole point: a coherent beam "
                   "clipped by N straight edges diffracts into the classic "
                   "N-fold diffraction STAR (N spikes for even N, 2N for odd) "
                   "instead of the circular Airy rings. 'aperture_diameter' is "
                   "the INSCRIBED-circle (flat-to-flat) clear aperture; "
                   "'blade_rotation' spins the polygon; 'blackness' drives the "
                   "disc absorbance directly (re-derived every rebuild, like "
                   "the circular iris).",
        "params": {"n_blades": P(6, "", "number of straight blades = polygon "
                                        "sides (>=3; 6 = classic 6-spike "
                                        "hexagonal star)"),
                   "outer_diameter": P(25.0, "mm", "disc outer diameter"),
                   "aperture_diameter": P(6.0, "mm", "clear aperture = "
                                          "inscribed-circle (flat-to-flat) "
                                          "diameter of the polygon"),
                   "thickness": P(1.0, "mm", "disc thickness"),
                   "blade_rotation": P(0.0, "deg", "azimuthal rotation of the "
                                       "blade polygon about the beam axis"),
                   "blackness": P(0.98, "", "fraction of incident power "
                                           "absorbed by the disc "
                                           "(0.95-1.0 typical)")},
        "props": {},   # disc: material/absorbance; plug: material=air --
                       # both set by the builder
        "derived_props": ("absorbance",),
    },
    "pinhole": {
        "category": "Apertures", "label": "Pinhole (rectangular mount)",
        "tooltip": "Small circular pinhole in a rectangular blackened "
                   "plate, plus a thin material=air plug filling the "
                   "hole (the aperture contract, docs/RAYTRACER.md "
                   "S5.10). 'blackness' drives the plate's absorbance "
                   "property directly (re-derived every rebuild).",
        "params": {"width": P(25.0, "mm", "plate width"),
                   "height": P(25.0, "mm", "plate height"),
                   "thickness": P(0.5, "mm", "plate thickness"),
                   "hole_diameter": P(0.5, "mm", "pinhole diameter"),
                   "blackness": P(1.0, "", "fraction of incident power "
                                          "absorbed by the plate "
                                          "(default 1.0: a diffraction "
                                          "SCREEN must be opaque; lower "
                                          "for a real blackened part)")},
        "props": {},   # plate: material/absorbance; plug: material=air --
                       # both set by the builder
        "derived_props": ("absorbance",),
    },
    "slit": {
        "category": "Apertures", "label": "Slit aperture",
        "tooltip": "Rectangular slit opening in a blackened plate, plus a "
                   "thin material=air plug filling the opening (the "
                   "aperture contract, docs/RAYTRACER.md S5.10). "
                   "'blackness' drives the plate's absorbance property "
                   "directly (re-derived every rebuild).",
        "params": {"width": P(25.0, "mm", "plate width"),
                   "height": P(25.0, "mm", "plate height"),
                   "thickness": P(0.5, "mm", "plate thickness"),
                   "slit_width": P(0.1, "mm", "slit opening width"),
                   "slit_height": P(10.0, "mm", "slit opening height"),
                   "blackness": P(1.0, "", "fraction of incident power "
                                          "absorbed by the plate "
                                          "(default 1.0: a diffraction "
                                          "SCREEN must be opaque; lower "
                                          "for a real blackened part)")},
        "props": {},   # plate: material/absorbance; plug: material=air --
                       # both set by the builder
        "derived_props": ("absorbance",),
    },
    "retro_corner_cube": {
        "category": "Prisms & Mirrors", "label": "Corner-cube retroreflector",
        "tooltip": "Solid glass trihedral corner-cube: three mutually "
                   "perpendicular back (TIR) faces return any incoming "
                   "ray antiparallel to itself, over a wide range of "
                   "incidence angles. Entrance face normal is rotated to "
                   "-x (beam travels +x) to match the rest of the "
                   "library's convention. 'aperture' is approximate (the "
                   "entrance face's inscribed-circle diameter).",
        "params": {"aperture": P(25.0, "mm", "approximate clear aperture "
                                             "(entrance-face inscribed-"
                                             "circle diameter)")},
        "props": {"material": "bk7"},
    },
    # =========================================================================
    # Batch 3 -- complex catalog primitives (beamsplitter cube, anamorphic
    # pair, Glan-Taylor polarizer, on-axis parabolic mirror)
    # =========================================================================
    "bs_cube": {
        "category": "Beamsplitters", "label": "Beamsplitter cube (50:50)",
        "tooltip": "Non-polarizing beamsplitter cube: a single BK7 cube "
                   "with a thin coated plate NESTED inside along the "
                   "diagonal (glass-glass interface -- the split table "
                   "applies exactly; the old two-prism air-gap build "
                   "TIR'd the transmitted arm at 45 deg and lost ~1/3 of "
                   "the power to seam loss). Default coating "
                   "bs_5050_vis_45; swap the plate body's 'coating' "
                   "property for any bs_XXYY_vis_45 registry row to "
                   "change the split ratio. Two bodies (cube + splitter "
                   "plate).",
        "params": {"cube": P(25.0, "mm", "cube edge length"),
                   "height": P(25.0, "mm", "extent along z"),
                   "plate_ct": P(0.2, "mm", "internal splitter plate "
                                            "thickness")},
        "props": {},   # per-body props (material + splitter coating) set
                       # by the builder
        # the coating string names a face index of the freshly built
        # plate -- a rebuild must re-derive it, not restore the stale one
        "derived_props": ("coating",),
    },
    "anamorphic_pair": {
        "category": "Prisms & Mirrors", "label": "Anamorphic prism pair",
        "tooltip": "Two wedge prisms in the standard anamorphic "
                   "arrangement (second prism's wedge flipped so the net "
                   "beam deviation cancels): magnifies the beam in one "
                   "axis (y) only -- e.g. circularizing a diode laser's "
                   "elliptical output. Two bk7 bodies; separation is baked "
                   "into each prism's local geometry (identity body "
                   "Placement on both, the achromat convention).",
        "params": {"wedge_deg": P(10.0, "deg", "wedge angle (both prisms)"),
                   "aperture": P(20.0, "mm", "clear aperture (y-extent and "
                                             "z-extrusion)"),
                   "separation": P(15.0, "mm", "gap between the two "
                                               "prisms' front faces along "
                                               "the beam (x)")},
        "props": {},   # material=bk7 on both, set by the builder
    },
    "polarizer_glan_taylor": {
        "category": "Polarization", "label": "Glan-Taylor polarizer",
        "tooltip": "Two calcite prisms split by an air gap along a "
                   "diagonal cut (default 40 deg from the optic-axis face; "
                   "typical Glan-Taylor range 38-42 deg): the ordinary ray "
                   "TIRs at the gap and is rejected sideways while the "
                   "extraordinary ray transmits straight through -- "
                   "extinction via TIR of the o-ray, not absorption. "
                   "crystal_axis=0,0,1 on both prisms puts the optic axis "
                   "along the extrusion (z) direction, perpendicular to "
                   "the x-y transmission plane (standard GT orientation). "
                   "Two bodies.",
        "params": {"aperture": P(15.0, "mm", "clear aperture (y-height and "
                                             "z-extrusion)"),
                   "length": P(20.0, "mm", "overall length (x)"),
                   "gap": P(0.005, "mm", "internal air gap at the cut"),
                   "cut_angle": P(40.0, "deg", "cut angle from the optic-"
                                               "axis face (typical GT "
                                               "range 38-42 deg)")},
        "props": {},   # material=calcite + crystal_axis=0,0,1 on both,
                       # set by the builder
    },
    "mirror_parabolic": {
        "category": "Prisms & Mirrors", "label": "Parabolic mirror (on-axis)",
        "tooltip": "On-axis front-surface parabolic mirror (concave toward "
                   "-x, conic k=-1, R=2*rfl -- exact paraxial AND "
                   "geometric focus at x=-rfl from the vertex, no on-axis "
                   "spherical aberration): revolved exact-sag BSpline + "
                   "surface_override, extractor-verified <1 um, same "
                   "technique as lens_asphere. (Descoped from an off-axis "
                   "OAP: the extractor's asphere vertex-locator requires "
                   "the retained face to include the r~=0 vertex, which a "
                   "90-deg off-axis segment structurally never does.)",
        "params": {"rfl": P(50.0, "mm", "reflected focal length (vertex "
                                        "to focus)"),
                   "aperture": P(25.0, "mm", "clear aperture diameter"),
                   "thickness": P(10.0, "mm", "substrate thickness behind "
                                             "the vertex")},
        "props": {"material": "aluminum"},
        # surface_override is derived from rfl/aperture by the builder --
        # rebuild-on-edit must NOT restore the stale pre-rebuild string
        # (the extractor's <1 um verifier dies on the mismatch)
        "derived_props": ("surface_override",),
    },
    # =========================================================================
    # Batch C -- fiber optics + telescope primitives (demo-gallery round)
    # =========================================================================
    "fiber_optic": {
        "category": "Fiber Optics", "label": "Step-index fiber (straight)",
        "tooltip": "Straight step-index multimode fiber segment along +x: "
                   "analytic-cylinder core plus a concentric cladding "
                   "annulus, flat polished end faces at x=0 and x=length. "
                   "Defaults model a 200 um-core 0.22-NA silica fiber "
                   "(core material fiber_core_na22 vs fused_silica "
                   "cladding; swap either body's 'material' property to "
                   "change the NA). The core/cladding boundary uses the "
                   "standard 5 um modeling air gap (optically-contacted "
                   "solids don't exist, docs/RAYTRACER.md) -- guided rays "
                   "launched inside the fiber NA still TIR at the core "
                   "wall exactly as they should; only rays OUTSIDE the NA "
                   "behave differently (they TIR at the core|gap interface "
                   "instead of refracting away into the cladding), so "
                   "leaky/cladding-mode power is not quantitative. Two "
                   "bodies.",
        "params": {"core_diameter": P(0.2, "mm", "core diameter (200 um "
                                                 "typical large-core "
                                                 "multimode)"),
                   "clad_diameter": P(0.24, "mm", "cladding outer diameter "
                                                  "(240 um typical for a "
                                                  "200 um core)"),
                   "length": P(75.0, "mm", "fiber segment length along +x"),
                   "gap": P(0.005, "mm", "core/cladding modeling air gap "
                                         "(the 5 um optical-contact "
                                         "convention; do not set to 0)")},
        "props": {},   # per-body materials set by the builder
    },
    "mirror_annular": {
        "category": "Prisms & Mirrors", "label": "Annular concave mirror",
        "tooltip": "Center-holed spherical concave mirror (a Cassegrain/"
                   "Schmidt-Cassegrain-style perforated primary): concave "
                   "toward -x like mirror_concave, with a clear circular "
                   "hole through the center so the converging beam folded "
                   "back by a secondary can pass behind the mirror. The "
                   "hole is genuinely open (revolved annular profile, no "
                   "plug body needed -- it is not an aperture stop, just "
                   "a pass-through).",
        "params": {"R": P(400.0, "mm", "concave curvature radius (>0); "
                                       "focal length = R/2"),
                   "aperture": P(100.0, "mm", "outer clear-aperture "
                                              "diameter"),
                   "hole_diameter": P(25.0, "mm", "central hole diameter"),
                   "ct": P(10.0, "mm", "substrate thickness: flat back at "
                                       "x=ct (must exceed the front sag, "
                                       "which bulges toward -x)")},
        "props": {"material": "aluminum"},
    },

    # =========================================================================
    # Samples & Cells (samples-instruments round): cuvettes/vials/vats built
    # as a NESTED PAIR of solids (the bs_cube/nested4 pattern -- a glass WALL
    # body fully containing a LIQUID body, no air gap, glass-to-liquid
    # contact). The WALL is always the PRIMARY body (element label + train
    # props); the liquid body is named '<group>_liquid' and is where the
    # user sets 'sample=' (a samples.miesamp row) to bind a particle
    # population, or overrides 'material'/'filter' for a plain absorbing
    # solution. Convention shared by every kind below: geometry params are
    # the INTERNAL (liquid) dimensions; 'wall' is added SYMMETRICALLY on
    # every face (axial AND transverse, all the way round -- exactly the
    # 'clearance_mm' convention make_test_scenes.make_nested4 already pins
    # for depth-4 nesting) to get the outer glass envelope, so e.g. 'height'
    # is the internal liquid fill height, not the external body height (no
    # headspace/meniscus modeled). Beam travels local +x.
    # =========================================================================
    "cuvette_square": {
        "category": "Samples & Cells", "label": "Cuvette (square, 10mm path)",
        "tooltip": "Standard rectangular spectrophotometer cuvette: a solid "
                   "glass WALL body with the LIQUID SAMPLE nested fully "
                   "inside it (bs_cube-style nested pair -- glass-to-liquid "
                   "contact, no air gap; the extractor classifies the pair "
                   "as validation.nested_solids and the tracer's LIFO "
                   "medium stack recovers the glass wall as the shell "
                   "outside the liquid volume). Beam travels along local "
                   "+x through path_length of liquid; width/height are the "
                   "INTERNAL (liquid) cross-section, wall is added "
                   "symmetrically on every face (x, y, AND z) to build the "
                   "outer glass envelope. The WALL body carries the "
                   "element label/train props (primary); set 'sample=' on "
                   "the second body ('<name>_liquid') to bind a "
                   "scattering/absorbing particle population, or override "
                   "its 'material'/'filter' for a plain absorbing "
                   "solution.",
        "params": {"path_length": P(10.0, "mm", "internal optical path "
                                                "length (liquid depth) "
                                                "along the beam"),
                   "width": P(10.0, "mm", "internal liquid cross-section "
                                          "width (y)"),
                   "height": P(40.0, "mm", "internal liquid fill height "
                                           "(z); no headspace modeled"),
                   "wall": P(1.25, "mm", "glass wall thickness, added "
                                         "symmetrically on every face")},
        "props": {},   # wall material=glass, liquid material=water (builder)
    },
    "cuvette_capillary": {
        "category": "Samples & Cells", "label": "Capillary cuvette (thin path)",
        "tooltip": "Thin-path rectangular cuvette (short-pathlength cell "
                   "for strongly absorbing/scattering samples): same "
                   "nested-pair construction as cuvette_square (solid "
                   "glass WALL body, LIQUID nested fully inside, wall "
                   "thickness added symmetrically on every face -- see "
                   "cuvette_square's tooltip for the full contract). "
                   "Default path_length=0.2mm.",
        "params": {"path_length": P(0.2, "mm", "internal optical path "
                                               "length (liquid depth) "
                                               "along the beam"),
                   "width": P(10.0, "mm", "internal liquid cross-section "
                                          "width (y)"),
                   "height": P(30.0, "mm", "internal liquid fill height "
                                           "(z); no headspace modeled"),
                   "wall": P(1.0, "mm", "glass wall thickness, added "
                                        "symmetrically on every face")},
        "props": {},   # wall material=glass, liquid material=water (builder)
    },
    "flow_cell": {
        "category": "Samples & Cells", "label": "Flow cell (process/Insitec)",
        "tooltip": "Process/in-line particle-sizing flow cell (Malvern "
                   "Insitec-style): a single glass block spans the full "
                   "body_width x body_height outer footprint (the window "
                   "mount along the beam), with the FLOWING LIQUID modeled "
                   "as a thin aperture x aperture channel of path_length "
                   "depth nested fully inside it (bs_cube-style nested "
                   "pair: 'wall' is the window thickness on each side of "
                   "the liquid channel along the beam/x; body_width/"
                   "body_height set the outer glass block footprint, "
                   "aperture the liquid channel's clear cross-section -- "
                   "keep aperture smaller than body_width/body_height so "
                   "the channel stays nested). The window/wall block is "
                   "the primary body (element label/train props); set "
                   "'sample=' on the '<name>_liquid' body to bind a "
                   "flowing particle population.",
        "params": {"path_length": P(2.0, "mm", "liquid channel depth "
                                               "along the beam"),
                   "aperture": P(10.0, "mm", "clear liquid-channel "
                                             "cross-section (square, y "
                                             "and z)"),
                   "body_width": P(20.0, "mm", "outer glass block width "
                                               "(y)"),
                   "body_height": P(20.0, "mm", "outer glass block height "
                                                "(z)"),
                   "wall": P(2.0, "mm", "window thickness on each side of "
                                        "the liquid channel (x)")},
        "props": {},   # wall material=glass, liquid material=water (builder)
    },
    "vial_cylindrical": {
        "category": "Samples & Cells", "label": "Cylindrical vial (DLS)",
        "tooltip": "Dynamic-light-scattering-style round sample vial: "
                   "vertical cylinder axis (local z); the beam crosses "
                   "HORIZONTALLY (local +x) through the curved glass wall "
                   "-- same local-origin convention as lens_rod (front "
                   "glass tangent at x=0, back tangent at x=diameter). "
                   "The glass WALL is a full solid cylinder; the LIQUID "
                   "sample is a smaller concentric cylinder nested fully "
                   "inside it (bs_cube-style nested pair, 'wall' added "
                   "symmetrically: radially AND on both z ends). The WALL "
                   "body is primary (element label/train props); set "
                   "'sample=' on '<name>_liquid' to bind a particle "
                   "population (e.g. the DLS suspension itself). KNOWN "
                   "ENGINE CAVEAT: at these dimensions the glass/water "
                   "curved pair is a real (aberrated) compound cylindrical "
                   "lens with an internal caustic; the C engine (the "
                   "default under --engine auto -- what the GUI/CLI always "
                   "use for this scene) traces it correctly (closure "
                   "~1e-14), but the pure-Python REFERENCE engine "
                   "(--engine python) has a reproducible energy-closure "
                   "bug on this specific glass/water/small-radius "
                   "combination (closure error blows up to ~1e16-1e20; "
                   "swapping the liquid to decalin at the identical "
                   "geometry closes fine, ruling out a geometry-authoring "
                   "mistake) -- a Python/C parity gap worth a follow-up "
                   "engine investigation, not a reason to avoid this "
                   "primitive in normal (auto-engine) use.",
        "params": {"diameter": P(12.0, "mm", "outer glass vial diameter"),
                   "height": P(45.0, "mm", "internal liquid fill height "
                                           "(z); no headspace modeled"),
                   "wall": P(1.0, "mm", "glass wall thickness, added "
                                        "symmetrically (radially and "
                                        "axially)")},
        "props": {},   # wall material=glass, liquid material=water (builder)
    },
    "vat_cylindrical": {
        "category": "Samples & Cells", "label": "Index-matching bath (vat)",
        "tooltip": "Large cylindrical index-matching bath (e.g. a DLS/SLS "
                   "goniometer vat): vertical axis (local z), same "
                   "lens_rod-style horizontal beam convention as "
                   "vial_cylindrical (glass tangent at x=0, x=diameter). "
                   "The glass WALL is a full solid cylinder; the BATH "
                   "LIQUID (default decalin, a common index-matching "
                   "fluid) is a smaller concentric cylinder nested fully "
                   "inside it (wall added symmetrically radially and "
                   "axially). Intended usage: place a vial_cylindrical "
                   "(or cuvette_*) element INSIDE the bath at the vat's "
                   "local center (x=diameter/2, y=0, z=0) via an ordinary "
                   "ANCHORED absolute placement -- train_solver has no "
                   "multi-port 'center' concept, so this is a separate "
                   "element nested by PLACEMENT, not a train port. The "
                   "extractor then sees a depth-4 concentric stack (outer "
                   "glass > bath liquid > inner vial glass > inner sample "
                   "liquid), exactly the pattern pinned by "
                   "scripts/raytracer/tests/test_nested_depth4.py. The "
                   "WALL body is primary; set 'sample=' or a 'filter'/"
                   "'material' override on '<name>_liquid' for the bath "
                   "fluid itself.",
        "params": {"diameter": P(80.0, "mm", "outer glass vat diameter"),
                   "height": P(100.0, "mm", "internal bath liquid fill "
                                            "height (z); no headspace "
                                            "modeled"),
                   "wall": P(2.5, "mm", "glass wall thickness, added "
                                        "symmetrically (radially and "
                                        "axially)")},
        "props": {},   # wall material=glass, liquid material=decalin (builder)
    },
    "sample_region": {
        "category": "Samples & Cells", "label": "Bare sample region (air)",
        "tooltip": "An air-material cube with no glass wall at all: a "
                   "chainable train anchor for a BARE particle cloud "
                   "(no cuvette/vial around it -- e.g. a free-space "
                   "aerosol/spray measurement volume). Assign 'sample=' "
                   "(a samples.miesamp row) directly on this body to bind "
                   "a scattering/absorbing particle population; the beam "
                   "passes straight through along local +x (a plain "
                   "pass-through port, like a window).",
        "params": {"width": P(20.0, "mm", "cube edge length (all three "
                                          "axes)")},
        "props": {"material": "air"},
    },

    # -- lamp sources (samples-instruments round): same disc/box-emitter
    #    builder + params as source_broadband/led_white, differing only in
    #    the emission registry row referenced by 'spectrum'. --------------
    "tungsten_halogen": {
        "category": "Sources", "label": "Tungsten-halogen (QTH) lamp",
        "tooltip": "Quartz tungsten-halogen (QTH) broadband lamp: an "
                   "analytic ~3000K blackbody continuum "
                   "(spectrum=bb_halogen_3000k) over the typical QTH "
                   "output range (350-2600nm; see the emitters.miesrc row "
                   "notes for what the pure-Planck approximation leaves "
                   "out). lambdac=850nm is a representative NIR-weighted "
                   "wavelength; the spectrum table supersedes it for "
                   "actual wavelength sampling.",
        "params": {"diameter": P(10.0, "mm", "emit face diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 50.0, "lambdac": 850.0,
                  "spectrum": "bb_halogen_3000k", "coherent": False},
    },
    "d2_lamp": {
        "category": "Sources", "label": "Deuterium (D2) UV lamp",
        "tooltip": "Deuterium arc lamp: the UV continuum only "
                   "(spectrum=d2_uv_approx, ~185-400nm smooth hump "
                   "peaking ~230-250nm) -- the real D2 Balmer-alpha/beta "
                   "atomic lines at 486/656nm are DELIBERATELY OMITTED "
                   "from this continuum-only table (see the "
                   "emitters.miesrc row notes); do not use this preset "
                   "where the visible Balmer lines matter. lambdac=250nm "
                   "is a representative UV wavelength; the spectrum table "
                   "supersedes it for actual wavelength sampling.",
        "params": {"diameter": P(10.0, "mm", "emit face diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 5.0, "lambdac": 250.0,
                  "spectrum": "d2_uv_approx", "coherent": False},
    },
    "hg_calibration": {
        "category": "Sources", "label": "Hg pen-lamp (calibration)",
        "tooltip": "Mercury pencil (pen) calibration lamp: discrete "
                   "emission lines (spectrum=hg_penlamp) at the standard "
                   "11-line Hg pen-lamp set (253.65nm and 365.01nm "
                   "strongest; see the emitters.miesrc row for the full "
                   "list + NIST ASD citation). lambdac=435.83nm (the "
                   "strong visible blue-violet line) is a representative "
                   "wavelength; the spectrum table supersedes it for "
                   "actual line sampling.",
        "params": {"diameter": P(10.0, "mm", "emit face diameter "
                                              "(circular) or edge length "
                                              "(rectangular, round_flag=0)"),
                   "length": P(10.0, "mm", "housing length"),
                   "round_flag": P(1, "", "1 = circular, 0 = rectangular")},
        "props": {"power": 1.0, "lambdac": 435.83,
                  "spectrum": "hg_penlamp", "coherent": False},
    },
    "source_image": {
        "category": "Sources", "label": "Extended image source",
        "tooltip": "Extended incoherent Lambertian emitter that projects a "
                   "named image (image=usaf_style_target, an "
                   "opticalproperties/image/images.mieimg registry row) "
                   "across its rectangular emit face instead of a uniform "
                   "disc/box -- e.g. a backlit resolution target or "
                   "reticle for imaging-system demos. Add the optional "
                   "'image_cone_deg' body property (emission half-angle in "
                   "degrees, default: full Lambertian hemisphere) to "
                   "restrict emission to a narrower cone for etendue-"
                   "matched imaging benches.",
        "params": {"width": P(20.0, "mm", "emit face width (y)"),
                   "height": P(20.0, "mm", "emit face height (z)"),
                   "length": P(5.0, "mm", "housing length")},
        "props": {"power": 10.0, "lambdac": 550.0, "coherent": False,
                  "image": "usaf_style_target"},
    },
}


# ---------------------------------------------------------------------------
# Edge blackening (engine3 Sec 11 / P8): a toggleable 'edge_blackened' flag on
# the simple-lens family (0 = ideal transmitting rim, the default; 1 = the
# lens barrel absorbs -- ghost/stray-light suppression). The builder sets a
# bool body property the extractor turns into per-face absorbance on the
# cylindrical edge; it is a derived_prop so a rebuild after toggling the sheet
# re-derives it (the same reason iris blackness is one -- the generic prop-
# preservation would otherwise restore the stale pre-rebuild value).
# ---------------------------------------------------------------------------
_EDGE_BLACKENABLE_LENSES = ("lens_pcx", "lens_dcx", "lens_pcv", "lens_dcv",
                            "lens_meniscus")
for _lk in _EDGE_BLACKENABLE_LENSES:
    PRIMITIVES[_lk]["params"]["edge_blackened"] = P(
        0, "", "1 = blacken the lens edge/barrel (absorbs stray/ghost light); "
               "0 = ideal transmitting rim")
    PRIMITIVES[_lk]["derived_props"] = ("edge_blackened",)


# ---------------------------------------------------------------------------
# Legacy alias migration (v2-feature-round rename): old saved elements'
# spreadsheets carry the pre-rename aliases below. read_params() falls back
# to <legacy alias> * <scale> when the CURRENT spec alias is missing from
# the sheet, so rebuild-on-edit keeps working on existing scenes/.MieWB
# libraries without a one-shot migration pass.
#   new alias -> (legacy alias, scale from legacy value to new alias value)
# ---------------------------------------------------------------------------
LEGACY_ALIASES = {
    "diameter": ("radius", 2.0),   # radius -> diameter: value doubles
    "width": ("half", 2.0),        # half (half-width) -> width: value doubles
}


# ---------------------------------------------------------------------------
# port_frames — element-local beam-port geometry (importable WITHOUT FreeCAD)
#
# The optical-train chain solver (scripts/train_solver.py) positions elements
# by VERTEX-TO-VERTEX distances along the beam. It needs each primitive's
# element-local port geometry, derived EXACTLY from the same dim-sheet
# parameters the builders above consume -- NEVER from faces (FaceN is unstable
# across rebuilds, and this must work in the plain-python GUI interpreter with
# no FreeCAD). Every formula below was read off the corresponding _build_*
# function and cross-checked against the shipped primitives/*.FCStd bounding
# boxes / face normals.
#
# LOCAL-AXIS CONVENTION: every library primitive is authored so the beam
# travels along local +x, with the transverse reference along local +z (the
# revolve axis for lenses/mirrors is +x; plate pads front-face at x=0 padding
# toward +x; sources emit from their +x cap). So axis=(1,0,0), up=(0,0,1) for
# all kinds. Multi-body elements (achromat, cube BS, fiber, ...) all author
# their bodies in one SHARED local frame with identity body placements (offsets
# baked into geometry; op_import_primitive/set_placement move the whole group
# rigidly), so ports are computed across the whole element in that frame:
# entry = first surface of the first optic, exit = last surface of the last.
# ---------------------------------------------------------------------------
_PORT_AXIS = [1.0, 0.0, 0.0]
_PORT_UP = [0.0, 0.0, 1.0]


def _port_unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]


def _port_params(kind, params):
    """Merge caller params (subset, mm/deg floats) over the spec defaults, so
    port_frames tolerates missing optional params exactly as the builders do
    (they read every param from the sheet, which always carries the defaults).
    Legacy alias names (radius/half) are accepted and scaled like read_params."""
    spec = PRIMITIVES[kind]
    given = dict(params or {})
    out = {}
    for alias, pspec in spec["params"].items():
        if alias in given:
            out[alias] = float(given[alias])
            continue
        legacy = LEGACY_ALIASES.get(alias)
        if legacy and legacy[0] in given:
            out[alias] = float(given[legacy[0]]) * legacy[1]
            continue
        out[alias] = float(pspec["default"])
    return out


def _port_result(entry_x, exit_x, reflect_point=None, reflect_normal=None):
    reflect_plane = None
    if reflect_normal is not None:
        pt = reflect_point if reflect_point is not None else [entry_x, 0.0, 0.0]
        reflect_plane = {"point": [float(pt[0]), float(pt[1]), float(pt[2])],
                         "normal": _port_unit(reflect_normal)}
    return {"entry": [float(entry_x), 0.0, 0.0],
            "exit": [float(exit_x), 0.0, 0.0],
            "axis": list(_PORT_AXIS), "up": list(_PORT_UP),
            "reflect_plane": reflect_plane}


# Kinds handled by generic families (thickness param name -> exit vertex x).
# Transmissive plates: front vertex at x=0, back vertex at x=thickness (on-axis
# center thickness; wedged plates are plane-parallel AT y=0). _build_plate /
# _plate_from_params / _build_wedge_plate all front-face at x=0, pad toward +x.
_PORT_TRANSMISSIVE_PLATES = {
    "window": "thickness", "window_wedged": "thickness",
    "filter_plate": "thickness", "nd_filter": "thickness",
    "filter_bandpass": "thickness", "filter_longpass": "thickness",
    "filter_shortpass": "thickness", "filter_notch": "thickness",
    "polarizer_plate": "thickness", "waveplate": "thickness",
    "diffuser_plate": "thickness", "prism_wedge": "thickness",
}
# Front-coated plates: same body as a transmissive plate, but the front (-x)
# face at x=0 carries the reflective/splitting coating (bs/pbs/dichroic/nd-
# refl/pellicle) or the grating spec. The transmit port still exits the back
# face; the reflect plane is the front face, normal -x (back into the beam).
_PORT_FRONT_COATED_PLATES = {
    "bs_plate": "thickness", "pbs_plate": "thickness",
    "dichroic_plate": "thickness", "nd_reflective": "thickness",
    "pellicle": "membrane_thickness", "grating_plate": "thickness",
}
# Flat front-surface reflectors: reflective -x face at x=0, entry==exit there.
_PORT_FLAT_MIRRORS = ("mirror_flat", "mirror_d_shaped")
# Curved front-surface mirrors: the reflective surface's on-axis VERTEX is at
# x=0 for every one of these (verified: mirror_concave/parabolic bulge toward
# -x with the axis vertex at x=0; convex apex at x=0; annular hole-center
# vertex extrapolated to x=0). entry==exit==vertex; tangent-plane normal -x.
_PORT_CURVED_MIRRORS = ("mirror_concave", "mirror_convex",
                        "mirror_parabolic", "mirror_annular")
# Center-thickness lenses: lens_meridian always puts the front vertex at x=0
# and the back vertex at x=ct, regardless of surface curvature signs, so the
# axis-pierce vertices are exactly (0,0,0) and (ct,0,0).
_PORT_CT_LENSES = ("lens_pcx", "lens_dcx", "lens_pcv", "lens_dcv",
                   "lens_meniscus", "lens_asphere")


def port_frames(kind, params):
    """Element-local beam-port geometry for a primitive kind, computed from its
    dim-sheet parameters (mm, floats). Returns
      {"entry": [x,y,z], "exit": [x,y,z],   # vertices ON the optical axis
       "axis": [x,y,z],                     # unit local beam direction entry->exit
       "up":   [x,y,z],                     # unit local transverse reference
       "reflect_plane": {"point": [x,y,z], "normal": [x,y,z]} or None}

    entry/exit are the optical VERTICES (axis-pierce points of the first and
    last optical surfaces, including sag: for the library's lenses that is
    always the front vertex at x=0 and the back vertex at x=ct). For mirrors
    entry == exit == the axis/reflective-surface intersection; for sources the
    emission point on the +x exit cap; for detectors the sensing-face center.
    reflect_plane (mirrors, the coated splitting surface of beamsplitters, and
    reflective gratings) is that surface's plane with normal signed to point
    back toward the entry side (into the incoming beam).

    axis is local +x and up is local +z for every kind: all builders construct
    the element along +x with the transverse reference in the y-z plane.

    APPROXIMATIONS (documented per the port contract):
      * prism -- a dispersing prism is a "deviate" port whose deviation is
        genuinely ray-/wavelength-dependent, so no fixed entry/exit surface
        vertex is meaningful. entry == exit == the prism's geometric center on
        the beam axis (local origin, the triangle centroid); only the port
        ORIGIN matters for the chain solver's downstream distance bookkeeping.
      * mirror_annular -- the reflect-plane POINT is the axis/vertex
        intersection even though the annular primary has a clear hole there
        (no material on axis); the plane geometry is still well-defined.

    Raises KeyError(kind) for kinds without a port formula (the caller falls
    back to a bbox heuristic): lens_cyl, lens_fresnel, axicon's cousins with
    internal TIR/reflective folds (retro_corner_cube, the right-angle/dove/
    penta/rhomboid prisms, the anamorphic pair, the Glan-Taylor polarizer)."""
    if kind not in PRIMITIVES:
        raise KeyError(kind)
    p = _port_params(kind, params)

    # -- sources: emission on the flat (or convex-cap) +x face at x=0 --------
    if PRIMITIVES[kind].get("category") == "Sources":
        return _port_result(0.0, 0.0)

    # -- thin sensing/aperture planes: entry == exit at the plane center -----
    if kind in ("detector_plane", "iris", "iris_bladed", "slit", "pinhole"):
        return _port_result(0.0, 0.0)

    # -- center-thickness lenses --------------------------------------------
    if kind in _PORT_CT_LENSES:
        return _port_result(0.0, p["ct"])
    if kind == "lens_ball":
        return _port_result(0.0, p["diameter"])            # sphere: 2R
    if kind == "lens_rod":
        return _port_result(0.0, p["diameter"])            # cylinder x-span 2R
    if kind == "lens_achromat":
        return _port_result(0.0, p["ct_crown"] + p["gap"] + p["ct_flint"])
    if kind == "axicon":
        axial = (p["aperture"] / 2.0) * math.tan(math.radians(p["base_angle"]))
        return _port_result(0.0, axial)                    # apex x=0, base x=axial

    # -- straight fiber: flat polished end faces at x=0 and x=length ---------
    if kind == "fiber_optic":
        return _port_result(0.0, p["length"])

    # -- samples & cells: nested glass-wall/liquid pairs, beam along +x ------
    # (rectangular cuvettes/flow cell: outer glass front/back face vertices
    # at x=0 and x=path_length+2*wall; cylindrical vial/vat: same lens_rod-
    # style tangent-to-tangent span, x=0 and x=diameter)
    if kind in ("cuvette_square", "cuvette_capillary", "flow_cell"):
        return _port_result(0.0, p["path_length"] + 2.0 * p["wall"])
    if kind in ("vial_cylindrical", "vat_cylindrical"):
        return _port_result(0.0, p["diameter"])
    if kind == "sample_region":
        return _port_result(0.0, p["width"])

    # -- prism (deviate port; center-origin approximation) ------------------
    if kind == "prism":
        return _port_result(0.0, 0.0)

    # -- plates -------------------------------------------------------------
    if kind in _PORT_TRANSMISSIVE_PLATES:
        return _port_result(0.0, p[_PORT_TRANSMISSIVE_PLATES[kind]])
    if kind in _PORT_FRONT_COATED_PLATES:
        thick = p[_PORT_FRONT_COATED_PLATES[kind]]
        return _port_result(0.0, thick, reflect_point=[0.0, 0.0, 0.0],
                            reflect_normal=[-1.0, 0.0, 0.0])

    # -- flat / curved front-surface mirrors --------------------------------
    if kind in _PORT_FLAT_MIRRORS or kind in _PORT_CURVED_MIRRORS:
        return _port_result(0.0, 0.0, reflect_point=[0.0, 0.0, 0.0],
                            reflect_normal=[-1.0, 0.0, 0.0])

    # -- beamsplitter cubes (nested diagonal splitter plate) ----------------
    if kind in ("bs_cube", "pbs_cube"):
        c = p["cube"]
        # cube front face x=0, back face x=cube; the splitter plane runs along
        # the D-B diagonal through the cube center (cube/2, 0). A +x axis at
        # y=0 meets that plane at x=cube/2. The coated face outward normal
        # (first face the +x beam hits) is (-1,-1)/sqrt2 -> back toward entry.
        return _port_result(0.0, c,
                            reflect_point=[c / 2.0, 0.0, 0.0],
                            reflect_normal=[-math.sqrt(0.5),
                                            -math.sqrt(0.5), 0.0])

    raise KeyError(kind)


# ---------------------------------------------------------------------------
# Builders (FreeCAD only). Each: fn(doc, group, p) -> [bodies]
# `group` is the element label stem; single-body builders name the body
# `group` itself, multi-body builders append a suffix.
# ---------------------------------------------------------------------------
def safe_set_props(body, props):
    """Like make_test_scenes.set_props but tolerates already-existing
    properties (needed on the rebuild path, where props are re-applied to
    freshly built bodies that may already carry some of them)."""
    for k, v in (props or {}).items():
        if k not in body.PropertiesList:
            if isinstance(v, bool):
                body.addProperty("App::PropertyBool", k, "Base")
            elif isinstance(v, (int, float)):
                body.addProperty("App::PropertyFloat", k, "Base")
            else:
                body.addProperty("App::PropertyString", k, "Base")
        setattr(body, k, v)


def _tag(bodies, kind, group):
    for b in bodies:
        safe_set_props(b, {"miewb_primitive": kind, "miewb_group": group})
    return bodies


def _simple_lens(doc, group, p, meridian):
    R1, R2 = meridian
    edges, _ = mts.lens_meridian(R1, R2, p["ct"], p["aperture"] / 2.0, 0.0)
    body = mts.revolve_body(doc, group, edges)
    # edge blackening flag (derived body property; extractor turns it into
    # per-face absorbance on the cylindrical barrel). Always stamped so a
    # rebuild after toggling the sheet re-derives it either way.
    mts.set_props(body, {"edge_blackened": bool(p.get("edge_blackened", 0))})
    return [body]


def _build_laser_collimated(doc, group, p):
    if p.get("round_flag", 1):
        return [mts.new_body_pad(doc, group, group,
                                 circle=(0.0, 0.0, p["diameter"] / 2.0),
                                 x_start=-p["length"], length=p["length"])]
    w = p["diameter"]
    h = w / 2.0
    return [mts.new_body_pad(doc, group, group,
                             rects=[(-h, -h, w, w)],
                             x_start=-p["length"], length=p["length"])]


def _build_laser_divergent(doc, group, p):
    if p.get("round_flag", 1):
        # rod with a convex (+x-bulging) spherical emit cap at x=0:
        # lens_meridian back surface with R=-roc bulges toward +x
        edges, _ = mts.lens_meridian(None, -p["roc"], p["length"],
                                     p["diameter"] / 2.0, -p["length"])
        return [mts.revolve_body(doc, group, edges)]
    # rectangular: a spherical cap on a box is not cheaply constructible;
    # build the flat rectangular emitter instead (roc does not apply).
    w = p["diameter"]
    h = w / 2.0
    return [mts.new_body_pad(doc, group, group,
                             rects=[(-h, -h, w, w)],
                             x_start=-p["length"], length=p["length"])]


def _build_source_image(doc, group, p):
    """Rectangular Lambertian image-projecting emitter: identical box shape
    to laser_collimated's round_flag=0 branch (emits from the +x face at
    x=0, housing extends toward -x), but with independent width/height so a
    non-square image aspect ratio (e.g. a 4:3 target) can be modeled."""
    w, h = p["width"], p["height"]
    return [mts.new_body_pad(doc, group, group,
                             rects=[(-w / 2.0, -h / 2.0, w, h)],
                             x_start=-p["length"], length=p["length"])]


# ---------------------------------------------------------------------------
# Samples & Cells: nested WALL(glass) + LIQUID pair (the bs_cube/nested4
# pattern -- one full solid strictly inside another, glass-to-liquid
# contact, no air gap -- extractor classifies it validation.nested_solids
# and the tracer's LIFO medium stack recovers the wall as the shell outside
# the liquid volume). 'wall' is added SYMMETRICALLY on every face, matching
# make_test_scenes.make_nested4's 'clearance_mm' convention. The wall body
# is always the PRIMARY body (named `group`, carries element label + train
# props); the liquid body is `group + "_liquid"`.
# ---------------------------------------------------------------------------
def _build_cuvette_box(doc, group, p, wall_material="glass",
                       liquid_material="water"):
    """Rectangular cuvette: outer glass box spans
    (path_length + 2*wall) x (width + 2*wall) x (height + 2*wall); the
    liquid box (path_length x width x height) sits centered inside it,
    inset by 'wall' on every face. Shared by cuvette_square and
    cuvette_capillary (only the sheet defaults differ)."""
    pl_, w, h, wall = p["path_length"], p["width"], p["height"], p["wall"]
    outer_x = pl_ + 2.0 * wall
    outer_y = w + 2.0 * wall
    outer_z = h + 2.0 * wall
    wall_body = mts.new_body_pad(
        doc, group, group,
        rects=[(-outer_y / 2.0, -outer_z / 2.0, outer_y, outer_z)],
        x_start=0.0, length=outer_x, props={"material": wall_material})
    liquid = mts.new_body_pad(
        doc, group + "_liquid", group + "_liquid",
        rects=[(-w / 2.0, -h / 2.0, w, h)],
        x_start=wall, length=pl_, props={"material": liquid_material})
    return [wall_body, liquid]


def _build_cuvette_square(doc, group, p):
    return _build_cuvette_box(doc, group, p)


def _build_cuvette_capillary(doc, group, p):
    return _build_cuvette_box(doc, group, p)


def _build_flow_cell(doc, group, p):
    """Process flow cell: a single glass block (body_width x body_height
    outer footprint, outer x = path_length + 2*wall) with the flowing
    liquid modeled as a thin aperture x aperture channel nested fully
    inside it, inset by 'wall' on the two beam-direction (x) window
    faces and centered within the larger body_width/body_height cross-
    section."""
    pl_, ap = p["path_length"], p["aperture"]
    bw, bh, wall = p["body_width"], p["body_height"], p["wall"]
    outer_x = pl_ + 2.0 * wall
    wall_body = mts.new_body_pad(
        doc, group, group,
        rects=[(-bw / 2.0, -bh / 2.0, bw, bh)],
        x_start=0.0, length=outer_x, props={"material": "glass"})
    liquid = mts.new_body_pad(
        doc, group + "_liquid", group + "_liquid",
        rects=[(-ap / 2.0, -ap / 2.0, ap, ap)],
        x_start=wall, length=pl_, props={"material": "water"})
    return [wall_body, liquid]


def _build_cyl_nested(doc, group, p, wall_material, liquid_material):
    """Shared vial_cylindrical / vat_cylindrical builder: vertical (local
    z) glass cylinder, a full solid, radius = diameter/2, x-centered at
    its own radius so the near tangent point sits at local x=0 (the
    lens_rod convention) and the far tangent at x=diameter; the liquid is
    a smaller concentric cylinder nested fully inside, inset by 'wall'
    radially and on both z ends."""
    d, h, wall = p["diameter"], p["height"], p["wall"]
    r_out = d / 2.0
    r_in = r_out - wall
    outer = Part.Circle(App.Vector(r_out, 0, 0), App.Vector(0, 0, 1), r_out)
    wall_body = mts.pad_body(doc, group, [outer], plane="XY",
                             offset=-(h / 2.0 + wall), length=h + 2.0 * wall,
                             props={"material": wall_material})
    inner = Part.Circle(App.Vector(r_out, 0, 0), App.Vector(0, 0, 1), r_in)
    liquid = mts.pad_body(doc, group + "_liquid", [inner], plane="XY",
                          offset=-h / 2.0, length=h,
                          props={"material": liquid_material})
    return [wall_body, liquid]


def _build_vial_cylindrical(doc, group, p):
    return _build_cyl_nested(doc, group, p, "glass", "water")


def _build_vat_cylindrical(doc, group, p):
    return _build_cyl_nested(doc, group, p, "glass", "decalin")


def _build_sample_region(doc, group, p):
    w = p["width"]
    return [mts.new_body_pad(doc, group, group,
                             rects=[(-w / 2.0, -w / 2.0, w, w)],
                             x_start=0.0, length=w,
                             props={"material": "air"})]


def _build_plate(doc, group, width_mm, thickness_mm, round_flag, name=None):
    """Shared plate builder: a round (cylinder) or rectangular (box) pad of
    diameter/edge-length `width_mm` and `thickness_mm`, front (-x) face at
    x=0. Used by every plate-like primitive (detector/mirror/window/
    polarizer/waveplate/filter/grating) so future plate primitives
    (beamsplitter/ND/filter batch) can reuse it directly."""
    name = name or group
    if round_flag:
        return [mts.new_body_pad(doc, name, name,
                                 circle=(0.0, 0.0, width_mm / 2.0),
                                 x_start=0.0, length=thickness_mm)]
    h = width_mm / 2.0
    return [mts.new_body_pad(doc, name, name,
                             rects=[(-h, -h, width_mm, width_mm)],
                             x_start=0.0, length=thickness_mm)]


def _plate_from_params(doc, group, p):
    return _build_plate(doc, group, p["width"], p["thickness"],
                        p.get("round_flag", 0), name=group)


def _build_detector_plane(doc, group, p):
    """Detector screen: square/round via the shared plate builder, or a
    true rectangle when height > 0 (round_flag=0) -- e.g. a 36 x 24 mm
    CMOS sensor; DetectorGrid derives the non-square pixel grid from the
    face bbox automatically."""
    height = p.get("height", 0.0)
    if p.get("round_flag", 0) or height <= 0.0:
        return _plate_from_params(doc, group, p)
    w = p["width"]
    return [mts.new_body_pad(doc, group, group,
                             rects=[(-w / 2.0, -height / 2.0, w, height)],
                             x_start=0.0, length=p["thickness"])]


def _build_fiber_optic(doc, group, p):
    """Core cylinder + cladding annulus along +x, separated by the 5 um
    optical-contact modeling gap (see the PRIMITIVES tooltip for the
    physics caveat). Both cylinders are native OCC analytic surfaces."""
    r_core = p["core_diameter"] / 2.0
    r_clad = p["clad_diameter"] / 2.0
    gap = p["gap"]
    length = p["length"]
    core = mts.new_body_pad(doc, group, group,
                            circle=(0.0, 0.0, r_core),
                            x_start=0.0, length=length,
                            props={"material": "fiber_core_na22"})
    outer = Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), r_clad)
    inner = Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1),
                        r_core + gap)
    clad = mts.pad_body(doc, group + "_clad", [outer, inner], plane="YZ",
                        offset=0.0, length=length,
                        props={"material": "fused_silica"})
    return [core, clad]


def _build_mirror_annular(doc, group, p):
    """Revolved annular meridian: spherical concave front (arc centered on
    the revolution axis -> a true OCC sphere), cylindrical hole wall,
    flat back at x=ct. The profile never touches the axis, so the
    revolution leaves the center genuinely open."""
    R = p["R"]
    r_out = p["aperture"] / 2.0
    r_in = p["hole_diameter"] / 2.0
    ct = p["ct"]

    def sag(v):
        return mts.surf_u(-R, 0.0, v)   # concave toward -x (mirror_concave)

    r_mid = (r_in + r_out) / 2.0
    edges = [
        mts._arc3(sag(r_in), r_in, sag(r_mid), r_mid, sag(r_out), r_out),
        mts._line(sag(r_out), r_out, ct, r_out),
        mts._line(ct, r_out, ct, r_in),
        mts._line(ct, r_in, sag(r_in), r_in),
    ]
    return [mts.revolve_body(doc, group, edges)]


def _build_lens_ball(doc, group, p):
    R = p["diameter"] / 2.0
    edges = [mts._arc3(0.0, 0.0, R, R, 2 * R, 0.0),
             mts._line(2 * R, 0.0, 0.0, 0.0)]
    return [mts.revolve_body(doc, group, edges)]


def _build_lens_rod(doc, group, p):
    R = p["diameter"] / 2.0
    circ = [Part.Circle(App.Vector(R, 0.0, 0.0), App.Vector(0, 0, 1), R)]
    return [mts.pad_body(doc, group, circ, plane="XY",
                         offset=-p["length"] / 2.0, length=p["length"])]


def _build_lens_cyl(doc, group, p):
    edges, _ = mts._cyl_lens_profile(p["R"], p["ct"], p["aperture"] / 2.0)
    return [mts.pad_body(doc, group, edges, plane="XY",
                         offset=-p["height"] / 2.0, length=p["height"])]


def _build_lens_asphere(doc, group, p):
    sa = p["aperture"] / 2.0
    R, k, ct = p["R"], p["k"], p["ct"]
    a4 = p.get("A4_mm3", 0.0)  # .get: pre-A4 saved sheets rebuild unchanged
    n_samp = 41
    pts = [App.Vector(mts._asphere_sag(sa * i / (n_samp - 1), R, k, a4),
                      sa * i / (n_samp - 1), 0)
           for i in range(n_samp)]
    bs = Part.BSplineCurve()
    bs.interpolate(pts)
    xfr = pts[-1].x
    edges = [bs,
             mts._line(xfr, sa, ct, sa),
             mts._line(ct, sa, ct, 0.0),
             mts._line(ct, 0.0, 0.0, 0.0)]
    body = mts.revolve_body(doc, group, edges)
    override = "Face1=asphere:R=%.6f;k=%.6f" % (R, k)
    if a4:
        override += ";A4=%.9g" % a4
    override += ";r_max=%.4f" % sa
    mts.set_props(body, {"surface_override": override})
    return [body]


def _build_lens_fresnel(doc, group, p):
    sa = p["aperture"] / 2.0
    n = int(round(p["n_facets"]))
    f, nglass = p["f_design"], p["n_design"]
    pts = [(0.0, 0.0)]
    for i in range(n):
        v0 = sa * i / n
        v1 = sa * (i + 1) / n
        slope = 0.5 * (v0 + v1) / ((nglass - 1.0) * f)
        dx = slope * (v1 - v0)
        pts.append((0.0, v0))
        pts.append((dx, v1))
    deepest = max(x for x, _ in pts)
    back = deepest + p["back"]
    pts.append((back, sa))
    pts.append((back, 0.0))
    poly = []
    for q in pts:
        if not poly or (abs(poly[-1][0] - q[0]) > 1e-9
                        or abs(poly[-1][1] - q[1]) > 1e-9):
            poly.append(q)
    edges = [mts._line(a[0], a[1], b[0], b[1])
             for a, b in zip(poly, poly[1:])]
    edges.append(mts._line(poly[-1][0], poly[-1][1],
                           poly[0][0], poly[0][1]))
    return [mts.revolve_body(doc, group, edges)]


def _build_lens_achromat(doc, group, p):
    sa = p["aperture"] / 2.0
    crown_edges, _ = mts.lens_meridian(p["R_front"], p["R_iface"],
                                       p["ct_crown"], sa, 0.0)
    crown = mts.revolve_body(doc, group + "_crown", crown_edges,
                             props={"material": "bk7"})
    flint_edges, _ = mts.lens_meridian(p["R_iface"], p["R_back"],
                                       p["ct_flint"], sa,
                                       p["ct_crown"] + p["gap"])
    flint = mts.revolve_body(doc, group + "_flint", flint_edges,
                             props={"material": "sf5"})
    return [crown, flint]


def _build_axicon(doc, group, p):
    sa = p["aperture"] / 2.0
    axial = sa * math.tan(math.radians(p["base_angle"]))
    edges = [mts._line(0.0, 0.0, axial, sa),
             mts._line(axial, sa, axial, 0.0),
             mts._line(axial, 0.0, 0.0, 0.0)]
    return [mts.revolve_body(doc, group, edges)]


def _build_prism(doc, group, p):
    # `rotation` is baked into the SKETCH vertices, NOT the body
    # Placement: rebuild_element preserves the pre-rebuild Placement, so a
    # placement-borne sheet param would be silently reverted on every
    # rebuild-on-edit (found by the prism_spectrometer demo, whose
    # min-deviation rotation vanished and left the beam 12 deg off).
    L, H = p["side"], p["height"]
    R = L / math.sqrt(3.0)
    rot = p.get("rotation", 0.0)
    verts = [(R * math.cos(math.radians(a + rot)),
              R * math.sin(math.radians(a + rot)))
             for a in (90.0, 210.0, 330.0)]
    edges = [mts._line(verts[i][0], verts[i][1],
                       verts[(i + 1) % 3][0], verts[(i + 1) % 3][1])
             for i in range(3)]
    return [mts.pad_body(doc, group, edges, plane="XY",
                         offset=-H / 2.0, length=H)]


def _build_cube_beamsplitter(doc, group, p, coating):
    """Single BK7 cube with a THIN COATED PLATE NESTED strictly inside
    along the D-B diagonal. The coated interface is then glass-glass
    (n1 = n2), so the registry split table applies exactly at 45 deg.

    WHY NOT the old two-prisms + 5 um cemented air gap: 45 deg internal
    incidence is past BK7's critical angle (41.2 deg) — the air gap TIRs
    the transmitted arm (there is no frustrated-TIR physics), table
    coatings then emit a grazing ghost child, and ~1/3 of the input power
    drowned in seam loss. Proper NESTING (one solid strictly inside
    another) is supported by the tracer's LIFO medium stack and is
    classified as validation.nested_solids by the extractor (partial
    overlap is still rejected). Validated: 46.7%/43.4% arms for bs, s-pol
    91%/0.04% + p-pol 2.3%/89% for pbs, zero seam loss, closure ~1e-12.

    The plate retracts from the cube edges (never touches the outer
    faces); the small uncoated margin passes a centered beam untouched.
    Shared by pbs_cube (polarizing) and bs_cube (non-polarizing): only
    the hypotenuse registry row differs."""
    c, H = p["cube"], p["height"]
    plate_ct = p.get("plate_ct", 0.2)
    half = c / 2.0
    sq = [mts._line(0.0, -half, c, -half), mts._line(c, -half, c, half),
          mts._line(c, half, 0.0, half), mts._line(0.0, half, 0.0, -half)]
    cube = mts.pad_body(doc, group, sq, plane="XY",
                        offset=-H / 2.0, length=H,
                        props={"material": "bk7"})

    ux, uy = math.sqrt(0.5), -math.sqrt(0.5)    # along the D-B diagonal
    mx, my = math.sqrt(0.5), math.sqrt(0.5)     # diagonal (split) normal
    cx, cy = half, 0.0
    margin = max(0.5, 0.025 * c)                # keep strictly inside
    hl = c * math.sqrt(2.0) / 2.0 - margin
    ht = plate_ct / 2.0
    corners = [(cx - ux * hl - mx * ht, cy - uy * hl - my * ht),
               (cx + ux * hl - mx * ht, cy + uy * hl - my * ht),
               (cx + ux * hl + mx * ht, cy + uy * hl + my * ht),
               (cx - ux * hl + mx * ht, cy - uy * hl + my * ht)]
    edges = [mts._line(*corners[i], *corners[(i + 1) % 4])
             for i in range(4)]
    z_margin = max(0.3, 0.015 * H)
    plate = mts.pad_body(doc, group + "_split", edges, plane="XY",
                         offset=-(H / 2.0 - z_margin),
                         length=H - 2.0 * z_margin,
                         props={"material": "bk7"})
    doc.recompute()
    # coat the FIRST face the +x beam hits (outward normal -(1,1)/sqrt2);
    # sign-aware lookup — the abs-dot helper can't tell the two parallel
    # plate faces apart
    f = _find_face_by_signed_normal(
        plate, (-math.sqrt(0.5), -math.sqrt(0.5), 0.0))
    if f is None:
        raise ValueError("%s: coated splitter face not found on %s"
                         % (group, plate.Name))
    safe_set_props(plate, {"coating": "Face%d=%s" % (f, coating)})
    return [cube, plate]


def _build_pbs_cube(doc, group, p):
    return _build_cube_beamsplitter(doc, group, p, "pbs_visible_45")


def _build_bs_cube(doc, group, p):
    return _build_cube_beamsplitter(doc, group, p, "bs_5050_vis_45")


def _build_wedge_plate(doc, group, width_mm, thickness_mm, wedge_deg,
                       round_flag, name=None):
    """Plate with a flat front (-x) face at x=0 and a back face tilted by
    wedge_deg (thickness increases toward +y); wedge_deg == 0 degenerates to
    the plain _build_plate. Rect: the tilt is folded directly into a 2-D
    (x,y) profile padded along z (uniform in z, same technique as
    axicon/lens_cyl). Round: a flat plate can't express a non-perpendicular
    back face as a single Sketch+Pad, so a rotated PartDesign::Plane
    (attached FlatFace to the front origin plane, offset+rotated via
    AttachmentOffset) plus a through-all Pocket cuts the tilted back face
    off an oversized blank -- verified against the expected tilted-face
    area/x-range in a scratch FreeCAD probe (front cap stays a full circle
    at x=0; back cap area = pi*(width/2)^2/cos(wedge_deg))."""
    name = name or group
    sa = width_mm / 2.0
    if not wedge_deg:
        return _build_plate(doc, group, width_mm, thickness_mm, round_flag,
                            name=name)
    tan_w = math.tan(math.radians(wedge_deg))
    if not round_flag:
        xb_hi = thickness_mm + sa * tan_w
        xb_lo = thickness_mm - sa * tan_w
        edges = [mts._line(0.0, -sa, 0.0, sa),
                 mts._line(0.0, sa, xb_hi, sa),
                 mts._line(xb_hi, sa, xb_lo, -sa),
                 mts._line(xb_lo, -sa, 0.0, -sa)]
        return [mts.pad_body(doc, name, edges, plane="XY",
                             offset=-sa, length=width_mm)]
    thick_max = thickness_mm + sa * tan_w * 1.05 + 0.5
    body = mts.new_body_pad(doc, name, name, circle=(0.0, 0.0, sa),
                            x_start=0.0, length=thick_max)
    doc.recompute()
    yz = mts._origin_plane(body, "YZ")
    plane = body.newObject("PartDesign::Plane", name + "_wedgeplane")
    plane.AttachmentSupport = [(yz, "")]
    plane.MapMode = "FlatFace"
    plane.AttachmentOffset = App.Placement(
        App.Vector(0.0, 0.0, thickness_mm),
        App.Rotation(App.Vector(0.0, 1.0, 0.0), wedge_deg))
    doc.recompute()
    big = max(width_mm, thick_max) * 10.0
    sk2 = body.newObject("Sketcher::SketchObject", name + "_wedgecut")
    sk2.AttachmentSupport = [(plane, "")]
    sk2.MapMode = "FlatFace"
    sk2.addGeometry(Part.LineSegment(App.Vector(-big, -big, 0),
                                     App.Vector(big, -big, 0)), False)
    sk2.addGeometry(Part.LineSegment(App.Vector(big, -big, 0),
                                     App.Vector(big, big, 0)), False)
    sk2.addGeometry(Part.LineSegment(App.Vector(big, big, 0),
                                     App.Vector(-big, big, 0)), False)
    sk2.addGeometry(Part.LineSegment(App.Vector(-big, big, 0),
                                     App.Vector(-big, -big, 0)), False)
    pocket = body.newObject("PartDesign::Pocket", name + "_wedgepocket")
    pocket.Profile = sk2
    pocket.Type = "ThroughAll"
    pocket.Reversed = True
    sk2.Visibility = False
    doc.recompute()
    return [body]


def _find_face_by_signed_normal(body, target, tol=1e-3):
    """Like mts._find_face_by_normal, but sign-sensitive. That helper scores
    candidates by abs(normal.dot(target)), which cannot tell a plate's front
    (-x) cap from its back (+x) cap -- both are exactly antiparallel to the
    x-axis, so an abs()-based search for the BACK face can silently return
    the FRONT face instead (caught by a probe: a diffuser_plate 'back face'
    lookup for (1,0,0) returned the same face a 'front face' lookup for
    (-1,0,0) would have). Used for every front-vs-back (-x/+x) face lookup
    in this module; _build_prism_penta's mirror-face lookup doesn't need
    this since those two targets aren't antiparallel to any other face."""
    t = App.Vector(*target)
    t.normalize()
    best, best_dot = None, tol
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
    return best


def _plate_with_face_prop(doc, group, width_mm, thickness_mm, round_flag,
                          prop_name, front_value=None, back_value=None,
                          name=None):
    """Build a plain plate (_build_plate) then set `prop_name` (coating or
    diffuser) on the dynamically-located front (-x) and/or back (+x) face --
    robust to round_flag and to the exact face numbering new_body_pad
    happens to produce (same technique _build_pbs_cube uses for its
    hypotenuse), rather than hardcoding 'FaceN='."""
    name = name or group
    body = _build_plate(doc, name, width_mm, thickness_mm, round_flag,
                        name=name)[0]
    doc.recompute()
    parts = []
    if front_value is not None:
        f = _find_face_by_signed_normal(body, (-1.0, 0.0, 0.0))
        if f is None:
            raise ValueError("%s: front face not found" % name)
        parts.append("Face%d=%s" % (f, front_value))
    if back_value is not None:
        b = _find_face_by_signed_normal(body, (1.0, 0.0, 0.0))
        if b is None:
            raise ValueError("%s: back face not found" % name)
        parts.append("Face%d=%s" % (b, back_value))
    safe_set_props(body, {prop_name: ";".join(parts)})
    return [body]


def _build_bs_plate(doc, group, p):
    body = _build_wedge_plate(doc, group, p["width"], p["thickness"],
                              p.get("wedge_deg", 0.0),
                              p.get("round_flag", 1), name=group)[0]
    doc.recompute()
    f = _find_face_by_signed_normal(body, (-1.0, 0.0, 0.0))
    if f is None:
        raise ValueError("bs_plate: front face not found")
    safe_set_props(body, {"coating": "Face%d=bs_5050_vis_45" % f})
    return [body]


def _build_pbs_plate(doc, group, p):
    return _plate_with_face_prop(doc, group, p["width"], p["thickness"],
                                 p.get("round_flag", 1), "coating",
                                 front_value="pbs_visible_45")


def _build_dichroic_plate(doc, group, p):
    return _plate_with_face_prop(doc, group, p["width"], p["thickness"],
                                 p.get("round_flag", 0), "coating",
                                 front_value="dichroic_567lp_45")


def _build_nd_reflective(doc, group, p):
    return _plate_with_face_prop(doc, group, p["width"], p["thickness"],
                                 p.get("round_flag", 1), "coating",
                                 front_value="nd_refl_od10")


def _build_diffuser_plate(doc, group, p):
    return _plate_with_face_prop(doc, group, p["width"], p["thickness"],
                                 p.get("round_flag", 1), "diffuser",
                                 back_value="@dg_600")


def _build_pellicle(doc, group, p):
    body = mts.new_body_pad(doc, group, group,
                            circle=(0.0, 0.0, p["diameter"] / 2.0),
                            x_start=0.0, length=p["membrane_thickness"])
    doc.recompute()
    f = _find_face_by_signed_normal(body, (-1.0, 0.0, 0.0))
    if f is None:
        raise ValueError("pellicle: front face not found")
    safe_set_props(body, {"coating": "Face%d=pellicle_4555_45" % f})
    return [body]


def _build_window_wedged(doc, group, p):
    return _build_wedge_plate(doc, group, p["width"], p["thickness"],
                              p["wedge_deg"], p.get("round_flag", 1),
                              name=group)


def _build_prism_wedge(doc, group, p):
    return _build_wedge_plate(doc, group, p["diameter"], p["thickness"],
                              p["wedge_deg"], 1, name=group)


def _build_prism_right_angle(doc, group, p):
    L, H = p["leg"], p["height"]
    edges = [mts._line(0.0, 0.0, L, 0.0),
             mts._line(L, 0.0, 0.0, L),
             mts._line(0.0, L, 0.0, 0.0)]
    return [mts.pad_body(doc, group, edges, plane="XY",
                         offset=-H / 2.0, length=H)]


def _build_prism_dove(doc, group, p):
    H, L = p["aperture"], p["length"]
    y0, y1 = -H / 2.0, H / 2.0
    edges = [mts._line(0.0, y0, L, y0),
             mts._line(L, y0, L - H, y1),
             mts._line(L - H, y1, H, y1),
             mts._line(H, y1, 0.0, y0)]
    return [mts.pad_body(doc, group, edges, plane="XY",
                         offset=-H / 2.0, length=H)]


def _build_prism_rhomboid(doc, group, p):
    h, L = p["aperture"], p["length"]
    A, B, C, D = (0.0, 0.0), (0.0, h), (L, h + L), (L, L)
    edges = [mts._line(A[0], A[1], B[0], B[1]),
             mts._line(B[0], B[1], C[0], C[1]),
             mts._line(C[0], C[1], D[0], D[1]),
             mts._line(D[0], D[1], A[0], A[1])]
    return [mts.pad_body(doc, group, edges, plane="XY",
                         offset=-h / 2.0, length=h)]


def _build_prism_penta(doc, group, p):
    """Canonical pentaprism cross-section (interior angles 90 deg between
    the entrance/exit faces, 112.5 deg at each of the other four vertices --
    the standard vendor-catalog penta-prism proportions), derived from the
    2-mirror-at-45-deg invariant (a beam reflected by two mirrors is
    deviated by 90 deg regardless of prism rotation): starting from the
    entrance-face direction and turning by the exterior angle (180-112.5 =
    67.5 deg) at each vertex in turn traces closed, convex EN-M1-BACK-M2-EX
    edges; picking the M1/M2 (mirror) edge lengths equal by symmetry leaves
    the BACK (non-optical) edge length k as the only free design choice
    (verified: the pentagon closure equations for the x- and y-components
    are then IDENTICAL, i.e. k truly is a free parameter). Reflecting faces
    (M1, M2) carry a real metallic coating (Al_mirror_bare, a pre-existing
    coatings.miecoat row) located dynamically from the polygon's own
    outward-edge normals -- penta-prism angles are known not to satisfy TIR
    in bk7, so a real mirror coating is the physically correct choice
    regardless of the exact working angle of incidence."""
    h = p["aperture"]
    k = 0.5 * h

    def _rot(v, deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)

    d_en = (0.0, 1.0)
    d_m1 = _rot(d_en, -67.5)
    d_back = _rot(d_m1, -67.5)
    d_m2 = _rot(d_back, -67.5)
    d_ex = _rot(d_m2, -67.5)
    cm = d_m1[0] + d_m2[0]
    const = h * d_en[0] + k * d_back[0] + h * d_ex[0]
    m = -const / cm

    A = (0.0, 0.0)
    B = (A[0] + h * d_en[0], A[1] + h * d_en[1])
    C = (B[0] + m * d_m1[0], B[1] + m * d_m1[1])
    D = (C[0] + k * d_back[0], C[1] + k * d_back[1])
    E = (D[0] + m * d_m2[0], D[1] + m * d_m2[1])
    verts = [A, B, C, D, E]
    edges = [mts._line(verts[i][0], verts[i][1],
                       verts[(i + 1) % 5][0], verts[(i + 1) % 5][1])
             for i in range(5)]
    body = mts.pad_body(doc, group, edges, plane="XY",
                        offset=-h / 2.0, length=h)
    doc.recompute()

    def _outward_normal(a, b):
        ex, ey = b[0] - a[0], b[1] - a[1]
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
        for cand in ((ey, -ex), (-ey, ex)):
            L = math.hypot(*cand)
            n = (cand[0] / L, cand[1] / L)
            if (mx - cx) * n[0] + (my - cy) * n[1] > 0:
                return n
        return None

    n_m1 = _outward_normal(B, C)
    n_m2 = _outward_normal(D, E)
    f1 = mts._find_face_by_normal(body, (n_m1[0], n_m1[1], 0.0))
    f2 = mts._find_face_by_normal(body, (n_m2[0], n_m2[1], 0.0))
    if f1 is None or f2 is None:
        raise ValueError("prism_penta: reflecting faces not found")
    safe_set_props(body, {"coating":
                         "Face%d=Al_mirror_bare;Face%d=Al_mirror_bare"
                         % (f1, f2)})
    return [body]


def _build_mirror_concave(doc, group, p):
    edges, _ = mts.lens_meridian(-p["R"], None, p["ct"],
                                 p["aperture"] / 2.0, 0.0)
    return [mts.revolve_body(doc, group, edges)]


def _build_mirror_convex(doc, group, p):
    edges, _ = mts.lens_meridian(p["R"], None, p["ct"],
                                 p["aperture"] / 2.0, 0.0)
    return [mts.revolve_body(doc, group, edges)]


def _build_mirror_d_shaped(doc, group, p):
    r = p["diameter"] / 2.0
    u0 = p["cut_offset"]
    vlim = math.sqrt(max(r * r - u0 * u0, 1e-6))
    chord = mts._line(u0, -vlim, u0, vlim)
    arc = mts._arc3(u0, vlim, r, 0.0, u0, -vlim)
    return [mts.pad_body(doc, group, [chord, arc], plane="YZ",
                         offset=0.0, length=p["thickness"])]


def _build_iris(doc, group, p):
    r_out, r_in = p["outer_diameter"] / 2.0, p["hole_diameter"] / 2.0
    outer = Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), r_out)
    inner = Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), r_in)
    disk = mts.pad_body(doc, group, [outer, inner], plane="YZ",
                        offset=0.0, length=p["thickness"],
                        props={"material": "aluminum",
                               "absorbance": p["blackness"]})
    plug = mts.new_body_pad(doc, group + "_plug", group + "_plug",
                            circle=(0.0, 0.0, r_in), x_start=0.0,
                            length=p["thickness"],
                            props={"material": "air"})
    return [disk, plug]


def _polygon_yz_edges(n, r_circ, rot_rad):
    """N Part.LineSegment edges of a regular n-gon (circumradius r_circ,
    azimuth offset rot_rad) in the YZ sketch-local (u=y, v=z) plane. A fresh
    list each call -- FreeCAD consumes a geometry when it is added to a sketch,
    so the disc and the plug each need their own."""
    verts = [(r_circ * math.cos(rot_rad + 2.0 * math.pi * k / n),
              r_circ * math.sin(rot_rad + 2.0 * math.pi * k / n))
             for k in range(n)]
    return [Part.LineSegment(App.Vector(verts[k][0], verts[k][1], 0.0),
                             App.Vector(verts[(k + 1) % n][0],
                                        verts[(k + 1) % n][1], 0.0))
            for k in range(n)]


def _build_iris_bladed(doc, group, p):
    n = int(round(p["n_blades"]))
    if n < 3:
        n = 3
    r_out = p["outer_diameter"] / 2.0
    r_in = p["aperture_diameter"] / 2.0        # inscribed-circle radius
    r_circ = r_in / math.cos(math.pi / n)      # polygon circumradius
    rot = math.radians(p["blade_rotation"])
    outer = Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), r_out)
    disk = mts.pad_body(doc, group,
                        [outer] + _polygon_yz_edges(n, r_circ, rot),
                        plane="YZ", offset=0.0, length=p["thickness"],
                        props={"material": "aluminum",
                               "absorbance": p["blackness"]})
    plug = mts.pad_body(doc, group + "_plug",
                        _polygon_yz_edges(n, r_circ, rot),
                        plane="YZ", offset=0.0, length=p["thickness"],
                        props={"material": "air"})
    return [disk, plug]


def _build_pinhole(doc, group, p):
    hw, hh = p["width"] / 2.0, p["height"] / 2.0
    r_hole = p["hole_diameter"] / 2.0
    disk = mts.new_body_pad(doc, group, group,
                            rects=[(-hw, -hh, p["width"], p["height"])],
                            circle=(0.0, 0.0, r_hole), x_start=0.0,
                            length=p["thickness"],
                            props={"material": "aluminum",
                                   "absorbance": p["blackness"]})
    plug = mts.new_body_pad(doc, group + "_plug", group + "_plug",
                            circle=(0.0, 0.0, r_hole), x_start=0.0,
                            length=p["thickness"],
                            props={"material": "air"})
    return [disk, plug]


def _build_slit(doc, group, p):
    hw, hh = p["width"] / 2.0, p["height"] / 2.0
    sw, sh = p["slit_width"] / 2.0, p["slit_height"] / 2.0
    disk = mts.new_body_pad(doc, group, group,
                            rects=[(-hw, -hh, p["width"], p["height"]),
                                   (-sw, -sh, p["slit_width"],
                                    p["slit_height"])],
                            x_start=0.0, length=p["thickness"],
                            props={"material": "aluminum",
                                   "absorbance": p["blackness"]})
    plug = mts.new_body_pad(doc, group + "_plug", group + "_plug",
                            rects=[(-sw, -sh, p["slit_width"],
                                    p["slit_height"])],
                            x_start=0.0, length=p["thickness"],
                            props={"material": "air"})
    return [disk, plug]


def _build_retro_corner_cube(doc, group, p):
    """Solid trihedral corner-cube: pad a cube corner at the origin, then
    slice it with a plane perpendicular to the (1,1,1) diagonal close
    enough to the origin (coordinate-sum cut < the cube edge) that the
    result is a clean 4-face tetrahedron (3 mutually-perpendicular faces +
    1 entrance face) rather than a 7-face corner-with-remnant-faces shape
    (verified in a scratch FreeCAD probe: coordinate-sum-cut = 0.9x the
    cube edge gives exactly 4 faces, with the 3 back faces' pairwise
    normal dot products = 0.0 to machine precision). The whole body is
    then rotated so the entrance face normal is -x, matching the rest of
    the library's 'beam travels along +x' convention."""
    aperture = p["aperture"]
    # d = tetrahedron leg length such that the entrance face's inscribed-
    # circle diameter ~= aperture (equilateral triangle side s = d*sqrt(2),
    # inradius = s/(2*sqrt(3)) = d*sqrt(6)/6).
    d = aperture / (math.sqrt(6.0) / 3.0)
    L = d * 1.2   # cube blank large enough that the cut stays a clean triangle
    body = doc.addObject("PartDesign::Body", group)
    sk = body.newObject("Sketcher::SketchObject", group + "_sk")
    xy = mts._origin_plane(body, "XY")
    sk.AttachmentSupport = [(xy, "")]
    sk.MapMode = "FlatFace"
    sk.addGeometry(Part.LineSegment(App.Vector(0, 0, 0), App.Vector(L, 0, 0)),
                   False)
    sk.addGeometry(Part.LineSegment(App.Vector(L, 0, 0), App.Vector(L, L, 0)),
                   False)
    sk.addGeometry(Part.LineSegment(App.Vector(L, L, 0), App.Vector(0, L, 0)),
                   False)
    sk.addGeometry(Part.LineSegment(App.Vector(0, L, 0), App.Vector(0, 0, 0)),
                   False)
    pad = body.newObject("PartDesign::Pad", group + "_pad")
    pad.Profile = sk
    pad.Length = L
    sk.Visibility = False
    doc.recompute()

    n = App.Vector(1.0, 1.0, 1.0)
    n.normalize()
    plane = body.newObject("PartDesign::Plane", group + "_cutplane")
    plane.MapMode = "Deactivated"
    plane.Placement = App.Placement(App.Vector(d / 3.0, d / 3.0, d / 3.0),
                                    App.Rotation(App.Vector(0, 0, 1), n))
    doc.recompute()
    big = L * 10.0
    sk2 = body.newObject("Sketcher::SketchObject", group + "_cutsk")
    sk2.AttachmentSupport = [(plane, "")]
    sk2.MapMode = "FlatFace"
    sk2.addGeometry(Part.LineSegment(App.Vector(-big, -big, 0),
                                     App.Vector(big, -big, 0)), False)
    sk2.addGeometry(Part.LineSegment(App.Vector(big, -big, 0),
                                     App.Vector(big, big, 0)), False)
    sk2.addGeometry(Part.LineSegment(App.Vector(big, big, 0),
                                     App.Vector(-big, big, 0)), False)
    sk2.addGeometry(Part.LineSegment(App.Vector(-big, big, 0),
                                     App.Vector(-big, -big, 0)), False)
    pocket = body.newObject("PartDesign::Pocket", group + "_cutpocket")
    pocket.Profile = sk2
    pocket.Type = "ThroughAll"
    pocket.Reversed = True
    sk2.Visibility = False
    doc.recompute()
    # reorient: entrance-face normal (1,1,1)/sqrt(3) -> -x
    body.Placement = App.Placement(App.Vector(0, 0, 0),
                                   App.Rotation(n, App.Vector(-1, 0, 0)))
    doc.recompute()
    return [body]


def _build_anamorphic_pair(doc, group, p):
    """Two wedge prisms in the standard beam-circularizing (anamorphic)
    arrangement: the second prism's wedge is flipped (tilt reversed
    relative to the first) so the net angular beam deviation cancels while
    the beam is magnified in y only -- e.g. circularizing a diode laser's
    elliptical output. Each prism's cross-section is the same flat-front/
    tilted-back profile _build_wedge_plate's rectangular branch uses, but
    the `separation` offset is baked directly into each prism's local
    (x, y) profile — both bodies get an IDENTITY Placement (the achromat
    convention: offsets live in geometry), unlike pbs_cube's second body,
    which is shifted via an actual Placement transform."""
    wd, ap, sep = p["wedge_deg"], p["aperture"], p["separation"]
    ct0 = 8.0   # nominal center thickness (not separately parameterized --
                # wedge_deg/aperture/separation are the only exposed knobs)
    sa = ap / 2.0
    tan_w = math.tan(math.radians(wd))

    def _wedge_edges(x0, sign):
        xb_hi = x0 + ct0 + sign * sa * tan_w
        xb_lo = x0 + ct0 - sign * sa * tan_w
        return [mts._line(x0, -sa, x0, sa),
                mts._line(x0, sa, xb_hi, sa),
                mts._line(xb_hi, sa, xb_lo, -sa),
                mts._line(xb_lo, -sa, x0, -sa)]

    b1 = mts.pad_body(doc, group + "_1", _wedge_edges(0.0, 1.0),
                      plane="XY", offset=-ap / 2.0, length=ap,
                      props={"material": "bk7"})
    b2 = mts.pad_body(doc, group + "_2", _wedge_edges(sep, -1.0),
                      plane="XY", offset=-ap / 2.0, length=ap,
                      props={"material": "bk7"})
    return [b1, b2]


def _build_polarizer_glan_taylor(doc, group, p):
    """Glan-Taylor polarizer: a rectangular calcite block cut along a
    diagonal at `cut_angle` from the y-axis (the optic-axis face), the two
    halves separated by an air gap along the cut normal (same shifted-
    second-body technique as _build_cube_beamsplitter's hypotenuse gap).
    crystal_axis '0,0,1' puts the optic axis along the z extrusion,
    perpendicular to the x-y transmission plane -- the standard GT cut
    orientation that TIRs the o-ray at the gap while the e-ray transmits."""
    h, L, gap, ca = p["aperture"], p["length"], p["gap"], p["cut_angle"]
    tan_c = math.tan(math.radians(ca))
    x_top = L / 2.0 + (h / 2.0) * tan_c
    x_bot = L / 2.0 - (h / 2.0) * tan_c
    crystal_props = {"material": "calcite", "crystal_axis": "0,0,1"}

    p1 = [(0.0, -h / 2.0), (0.0, h / 2.0), (x_top, h / 2.0), (x_bot, -h / 2.0)]
    edges1 = [mts._line(*p1[i], *p1[(i + 1) % 4]) for i in range(4)]
    b1 = mts.pad_body(doc, group + "_in", edges1, plane="XY",
                      offset=-h / 2.0, length=h, props=crystal_props)

    p2 = [(x_top, h / 2.0), (L, h / 2.0), (L, -h / 2.0), (x_bot, -h / 2.0)]
    edges2 = [mts._line(*p2[i], *p2[(i + 1) % 4]) for i in range(4)]
    ca_r = math.radians(ca)
    n = (math.cos(ca_r), -math.sin(ca_r))   # cut-plane normal, points +x-ish
    pl2 = App.Placement(App.Vector(gap * n[0], gap * n[1], 0.0),
                        App.Rotation())
    b2 = mts.pad_body(doc, group + "_out", edges2, plane="XY",
                      offset=-h / 2.0, length=h, props=crystal_props,
                      placement=pl2)
    return [b1, b2]


def _build_mirror_parabolic(doc, group, p):
    """On-axis front-surface parabolic mirror: same exact-sag BSpline +
    surface_override technique as _build_lens_asphere (k=-1 always -- a
    true parabola, not a user-tunable conic), reflecting face concave
    toward -x so a -x-traveling collimated beam converges at the paraxial
    AND geometric focus x=-rfl from the vertex (R=2*rfl kills spherical
    aberration on axis exactly, same as any conic mirror at its own focus).
    Descoped from an off-axis OAP -- see the PRIMITIVES tooltip / the
    project report for why."""
    sa = p["aperture"] / 2.0
    R_decl = 2.0 * p["rfl"]
    k, thickness = -1.0, p["thickness"]
    # NEGATIVE R for the actual geometry (unlike _build_lens_asphere's
    # convex-toward-source R>0): the reflecting face must be CONCAVE toward
    # -x -- the same sign flip _build_mirror_concave applies to its
    # (spherical) lens_meridian call -- so a -x-approaching collimated beam
    # actually converges instead of diverging. The surface_override string
    # still declares the POSITIVE R_decl: the extractor's vertex/axis
    # locator (build_asphere_surface) auto-flips ITS OWN local axis so
    # near-vertex sag reads non-negative, whichever way the real geometry
    # bulges -- so it always expects the same-sign R lens_asphere uses,
    # regardless of which global direction this body's surface opens
    # (verified empirically: declaring the geometry's own signed R here
    # fails the <1um gate with an exact sign-flipped residual).
    n_samp = 41
    pts = [App.Vector(mts._asphere_sag(sa * i / (n_samp - 1), -R_decl, k),
                      sa * i / (n_samp - 1), 0)
           for i in range(n_samp)]
    bs = Part.BSplineCurve()
    bs.interpolate(pts)
    xfr = pts[-1].x
    edges = [bs,
             mts._line(xfr, sa, thickness, sa),
             mts._line(thickness, sa, thickness, 0.0),
             mts._line(thickness, 0.0, 0.0, 0.0)]
    body = mts.revolve_body(doc, group, edges)
    mts.set_props(body, {"surface_override":
                         "Face1=asphere:R=%.6f;k=%.6f;r_max=%.4f"
                         % (R_decl, k, sa)})
    return [body]


# ---------------------------------------------------------------------------
# Prescription emission (engine3.md Sec 3, P5): the SAME dim params that drive
# the geometry builders above ALSO imply the exact analytic optical surfaces.
# build_prescription_entry() computes them in BODY-LOCAL SI-metre coordinates
# (optical axis local +x, front vertex at the origin -- the primitive
# convention), in the model.json surface language (raytracer.prescription /
# common._SURFACE_REQ). This is PURE (no FreeCAD) so the GUI, make_primitives
# and make_demos can all emit prescriptions without a worker, and it is the
# SINGLE authoring path: editing a dim param regenerates the geometry AND its
# prescription through this one function.
#
# Scope this round -- the clean analytic lens/mirror family:
#   spheres  (pcx/pcv/dcx/dcv/meniscus caps, ball)  -> emitted-from-prescription
#   asphere  (lens_asphere front, mirror_parabolic) -> emitted-from-prescription
#   cylinder (rod/cyl barrel, lens edge rims)        -> verified against the
#            prescription, kept in native OCC form (already exact, and a
#            cylinder's origin along its own axis is a free parameter)
# Flat backs stay with the extractor's native-OCC plane classification (exact,
# no canonicalization risk) and are not carried in the prescription.
# ---------------------------------------------------------------------------
_PRESCRIPTION_KINDS = frozenset((
    "lens_pcx", "lens_dcx", "lens_pcv", "lens_dcv", "lens_meniscus",
    "lens_ball", "lens_rod", "lens_cyl", "lens_asphere", "mirror_parabolic",
))

_MM = 1e-3   # mm -> m


def prescription_kinds():
    """The primitive kinds build_prescription_entry() can emit for."""
    return set(_PRESCRIPTION_KINDS)


def _presc_material(kind):
    return PRIMITIVES.get(kind, {}).get("props", {}).get("material")


def _sph_surface(role, material, cx_m, R_m):
    return {"role": role, "material": material, "type": "sphere",
            "center": [cx_m, 0.0, 0.0], "radius": abs(R_m)}


def _cyl_surface(role, material, ox_m, r_m, axis):
    return {"role": role, "material": material, "type": "cylinder",
            "origin": [ox_m, 0.0, 0.0], "axis": list(axis),
            "radius": abs(r_m)}


def _asph_surface(role, material, R_m, k, coeffs_si, r_max_m):
    return {"role": role, "material": material, "type": "asphere",
            "vertex": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
            "R": R_m, "k": float(k), "coeffs": list(coeffs_si),
            "r_max": r_max_m}


def _asphere_coeff_si(order, a_mm):
    """A_n (mm^(1-n) coefficient of the r^n sag term, r/sag in mm) -> SI
    (m^(1-n)): A_n_SI = A_n_mm * 1000^(n-1). Mirrors extract_geometry's
    build_asphere_surface coeff conversion exactly."""
    return a_mm * (1000.0 ** (order - 1))


def build_prescription_entry(kind, params_mm, key=None):
    """Pure: dim-sheet params (FreeCAD internal units: mm / deg) -> a
    prescription entry {kind, params (SI), surfaces (LOCAL SI)} for the
    covered lens/mirror family, or None for any other kind. `key` is unused
    here (the caller keys the element in the document); accepted so callers
    can pass it uniformly."""
    if kind not in _PRESCRIPTION_KINDS:
        return None
    p = params_mm
    mat = _presc_material(kind)
    surfaces = []
    si_params = {}

    if kind in ("lens_pcx", "lens_dcx", "lens_pcv", "lens_dcv",
                "lens_meniscus"):
        R1_mm, R2_mm = PRIMITIVES[kind]["meridian"](p)
        ct_m = p["ct"] * _MM
        sa_m = (p["aperture"] / 2.0) * _MM
        si_params = {"ct": ct_m, "aperture": p["aperture"] * _MM}
        if R1_mm is not None:
            si_params["R_front"] = R1_mm * _MM
            surfaces.append(_sph_surface("front", mat, R1_mm * _MM, R1_mm * _MM))
        if R2_mm is not None:
            si_params["R_back"] = R2_mm * _MM
            surfaces.append(_sph_surface(
                "back", mat, (p["ct"] + R2_mm) * _MM, R2_mm * _MM))
        surfaces.append(_cyl_surface("edge", mat, 0.0, sa_m, [1.0, 0.0, 0.0]))

    elif kind == "lens_ball":
        R_m = (p["diameter"] / 2.0) * _MM
        si_params = {"diameter": p["diameter"] * _MM}
        surfaces.append(_sph_surface("sphere", mat, R_m, R_m))

    elif kind == "lens_rod":
        R_m = (p["diameter"] / 2.0) * _MM
        si_params = {"diameter": p["diameter"] * _MM, "length": p["length"] * _MM}
        # barrel: revolved about local +z, centre-of-section at x = R
        surfaces.append(_cyl_surface("barrel", mat, R_m, R_m, [0.0, 0.0, 1.0]))

    elif kind == "lens_cyl":
        R_mm = p["R"]      # signed: <0 concave
        R_m = R_mm * _MM
        si_params = {"R": R_m, "ct": p["ct"] * _MM,
                     "aperture": p["aperture"] * _MM, "height": p["height"] * _MM}
        # cylinder axis || local +z; axis-line at x = R (like a sphere centre)
        surfaces.append(_cyl_surface("front", mat, R_m, R_m, [0.0, 0.0, 1.0]))

    elif kind == "lens_asphere":
        R_m = p["R"] * _MM
        sa_m = (p["aperture"] / 2.0) * _MM
        a4 = p.get("A4_mm3", 0.0)
        coeffs = [_asphere_coeff_si(4, a4)] if a4 else []
        si_params = {"R": R_m, "k": float(p["k"]), "ct": p["ct"] * _MM,
                     "aperture": p["aperture"] * _MM}
        if a4:
            si_params["A4"] = _asphere_coeff_si(4, a4)
        surfaces.append(_asph_surface("front", mat, R_m, p["k"], coeffs, sa_m))
        surfaces.append(_cyl_surface("edge", mat, 0.0, sa_m, [1.0, 0.0, 0.0]))

    elif kind == "mirror_parabolic":
        # builder DECLARES the positive R = 2*rfl (the extractor's asphere
        # verifier flips its own local axis to match the concave geometry),
        # k = -1 always (a true parabola). Match that declaration here.
        R_decl_m = (2.0 * p["rfl"]) * _MM
        sa_m = (p["aperture"] / 2.0) * _MM
        si_params = {"rfl": p["rfl"] * _MM, "aperture": p["aperture"] * _MM,
                     "thickness": p["thickness"] * _MM}
        surfaces.append(_asph_surface("front", mat, R_decl_m, -1.0, [], sa_m))
        surfaces.append(_cyl_surface("edge", mat, 0.0, sa_m, [1.0, 0.0, 0.0]))

    if not surfaces:
        return None
    return {"kind": kind, "params": si_params, "surfaces": surfaces}


def _lens_builder(kind):
    meridian = PRIMITIVES[kind]["meridian"]
    return lambda doc, group, p: _simple_lens(doc, group, p, meridian(p))


_BUILDERS = None


def builders():
    global _BUILDERS
    if _BUILDERS is None:
        if not _HAVE_FREECAD:
            raise RuntimeError("primitivelib builders need FreeCAD")
        _BUILDERS = {
            "laser_collimated": _build_laser_collimated,
            "laser_pulsed": _build_laser_collimated,
            "laser_maitai_800": _build_laser_collimated,
            "laser_erfiber_1560": _build_laser_collimated,
            "laser_ndyag_1064": _build_laser_collimated,
            "sc_superk": _build_laser_collimated,
            "fiber_nonlinear_output": _build_laser_collimated,
            "laser_divergent": _build_laser_divergent,
            "source_broadband": _build_laser_collimated,
            "led_deep_red_660": _build_laser_collimated,
            "led_red_630": _build_laser_collimated,
            "led_amber_590": _build_laser_collimated,
            "led_green_525": _build_laser_collimated,
            "led_blue_470": _build_laser_collimated,
            "led_royal_blue_450": _build_laser_collimated,
            "led_uv_365": _build_laser_collimated,
            "led_uv_385": _build_laser_collimated,
            "led_white": _build_laser_collimated,
            "detector_plane": _build_detector_plane,
            "lens_ball": _build_lens_ball,
            "lens_rod": _build_lens_rod,
            "lens_cyl": _build_lens_cyl,
            "lens_asphere": _build_lens_asphere,
            "lens_fresnel": _build_lens_fresnel,
            "lens_achromat": _build_lens_achromat,
            "axicon": _build_axicon,
            "prism": _build_prism,
            "mirror_flat": _plate_from_params,
            "window": _plate_from_params,
            "polarizer_plate": _plate_from_params,
            "waveplate": _plate_from_params,
            "filter_plate": _plate_from_params,
            "grating_plate": _plate_from_params,
            "pbs_cube": _build_pbs_cube,
            "bs_plate": _build_bs_plate,
            "pbs_plate": _build_pbs_plate,
            "dichroic_plate": _build_dichroic_plate,
            "pellicle": _build_pellicle,
            "nd_filter": _plate_from_params,
            "nd_reflective": _build_nd_reflective,
            "filter_bandpass": _plate_from_params,
            "filter_longpass": _plate_from_params,
            "filter_shortpass": _plate_from_params,
            "filter_notch": _plate_from_params,
            "window_wedged": _build_window_wedged,
            "diffuser_plate": _build_diffuser_plate,
            "prism_right_angle": _build_prism_right_angle,
            "prism_wedge": _build_prism_wedge,
            "prism_dove": _build_prism_dove,
            "prism_penta": _build_prism_penta,
            "prism_rhomboid": _build_prism_rhomboid,
            "mirror_concave": _build_mirror_concave,
            "mirror_convex": _build_mirror_convex,
            "mirror_d_shaped": _build_mirror_d_shaped,
            "iris": _build_iris,
            "iris_bladed": _build_iris_bladed,
            "pinhole": _build_pinhole,
            "slit": _build_slit,
            "retro_corner_cube": _build_retro_corner_cube,
            "bs_cube": _build_bs_cube,
            "anamorphic_pair": _build_anamorphic_pair,
            "polarizer_glan_taylor": _build_polarizer_glan_taylor,
            "mirror_parabolic": _build_mirror_parabolic,
            "fiber_optic": _build_fiber_optic,
            "mirror_annular": _build_mirror_annular,
            "cuvette_square": _build_cuvette_square,
            "cuvette_capillary": _build_cuvette_capillary,
            "flow_cell": _build_flow_cell,
            "vial_cylindrical": _build_vial_cylindrical,
            "vat_cylindrical": _build_vat_cylindrical,
            "sample_region": _build_sample_region,
            "tungsten_halogen": _build_laser_collimated,
            "d2_lamp": _build_laser_collimated,
            "hg_calibration": _build_laser_collimated,
            "source_image": _build_source_image,
        }
        for kind, spec in PRIMITIVES.items():
            if "meridian" in spec:
                _BUILDERS[kind] = _lens_builder(kind)
    return _BUILDERS


# ---------------------------------------------------------------------------
# Build / rebuild entry points (FreeCAD only)
# ---------------------------------------------------------------------------
def sheet_raw(value, unit):
    if unit:
        return "=%.10g %s" % (value, unit)
    return "%.10g" % value


def make_sheet(doc, kind, label="dim"):
    """Create the parameter spreadsheet for `kind` with default values."""
    sheet = doc.addObject("Spreadsheet::Sheet", "Spreadsheet")
    sheet.Label = label
    row = 1
    for alias, spec in PRIMITIVES[kind]["params"].items():
        cell_lbl = "A%d" % row
        cell_val = "B%d" % row
        sheet.set(cell_lbl, alias)
        sheet.set(cell_val, sheet_raw(spec["default"], spec["unit"]))
        sheet.setAlias(cell_val, alias)
        row += 1
    return sheet


def read_params(sheet, kind):
    """Alias values (floats, FreeCAD internal units: mm / deg) for `kind`.

    Legacy-scene fallback: if a spec alias isn't in the sheet (e.g. an
    element built before the diameter/width rename), try LEGACY_ALIASES's
    old alias name scaled to the new one; if that's not present either
    (e.g. a brand-new param like round_flag on an old element), use the
    spec default. This keeps rebuild-on-edit working on pre-existing
    scenes/.MieWB libraries without a migration pass."""
    import FreeCAD
    out = {}
    for alias, spec in PRIMITIVES[kind]["params"].items():
        cell = sheet.getCellFromAlias(alias)
        if cell:
            qty = sheet.get(alias)
            out[alias] = float(FreeCAD.Units.Quantity(qty).Value)
            continue
        legacy = LEGACY_ALIASES.get(alias)
        if legacy:
            legacy_alias, scale = legacy
            legacy_cell = sheet.getCellFromAlias(legacy_alias)
            if legacy_cell:
                qty = sheet.get(legacy_alias)
                out[alias] = float(FreeCAD.Units.Quantity(qty).Value) * scale
                continue
        out[alias] = spec["default"]
    return out


def build_primitive(doc, kind, group=None, params=None):
    """Build `kind` into doc: bodies + contract props + tagging. Returns
    the list of bodies. `params` defaults to the spec defaults."""
    spec = PRIMITIVES[kind]
    group = group or kind
    if params is None:
        params = {a: s["default"] for a, s in spec["params"].items()}
    bodies = builders()[kind](doc, group, params)
    for b in bodies:
        safe_set_props(b, spec["props"])
    _tag(bodies, kind, group)
    doc.recompute()
    return bodies


def rebuild_element(doc, sheet, kind, group):
    """Rebuild all bodies of `group` from the sheet's current parameter
    values, preserving each body's Label, Placement and any extra custom
    props the user added since — INCLUDING the MieTrain chain-recipe
    props (miewb_train_*): a variable edit that rebuilds a chained
    primitive must not tear the element out of the optical train (found
    by the telephoto demo's efl edit — the old Base-group-only snapshot
    silently dropped mode/ref/distance and the train fell apart).
    Returns the new bodies."""
    params = read_params(sheet, kind)
    old = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"
           and getattr(o, "miewb_group", None) == group]
    if not old:
        raise ValueError("no bodies with miewb_group %r" % group)
    keep = []
    # derived_props (e.g. iris/pinhole/slit's 'absorbance', re-derived from
    # the 'blackness' sheet param every rebuild): excluded from the extra-
    # props snapshot below so the builder's freshly-computed value always
    # wins, instead of the generic user-customization-preservation path
    # restoring a now-stale value from the body being replaced.
    baseline = {"miewb_primitive", "miewb_group"} \
        | set(PRIMITIVES.get(kind, {}).get("derived_props", ()))
    for b in old:
        extra = {}
        for pname in b.PropertiesList:
            if pname in baseline:
                continue
            try:
                pgroup = b.getGroupOfProperty(pname)
                if pgroup not in ("Base", "MieTrain"):
                    continue
                ptype = b.getTypeIdOfProperty(pname)
            except Exception:
                continue
            if ptype in ("App::PropertyString", "App::PropertyFloat",
                         "App::PropertyBool") \
                    and pname not in ("Label",):
                extra[pname] = (getattr(b, pname), pgroup)
        keep.append({"label": b.Label, "placement": b.Placement,
                     "extra": extra})
        # remove the body and its owned features
        feats = list(getattr(b, "Group", []) or [])
        doc.removeObject(b.Name)
        for f in feats:
            try:
                doc.removeObject(f.Name)
            except Exception:
                pass
    doc.recompute()
    bodies = builders()[kind](doc, group, params)
    _tag(bodies, kind, group)
    for b, k in zip(bodies, keep):
        b.Label = k["label"]
        b.Placement = k["placement"]
        for pname, (val, pgroup) in k["extra"].items():
            if pname not in b.PropertiesList:
                fc_type = ("App::PropertyBool" if isinstance(val, bool)
                           else "App::PropertyFloat"
                           if isinstance(val, (int, float))
                           else "App::PropertyString")
                b.addProperty(fc_type, pname, pgroup)
            setattr(b, pname, val)
    doc.recompute()
    return bodies
