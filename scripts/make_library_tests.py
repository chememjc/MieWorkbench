#!/usr/bin/env python3
# =============================================================================
# make_library_tests.py — build the demos/library_tests/ template scenes
# THROUGH THE GUI'S OWN OP PATH (same idiom as scripts/make_demos.py).
#
# These are minimal source -> element -> detector template scenes. A separate
# sweep runner (another workstream) copies each template, swaps ONE property
# (material / coating / filter / polarizer / grating registry name), and runs
# an ultra-quick pipeline to smoke every library entry. This script builds and
# ships the committed templates (demos/library_tests/<name>.FCStd + .MieWB),
# verified to RUN.
#
#   python3 scripts/make_library_tests.py [--scene NAME|all]
#                                         [--outdir demos/library_tests]
#                                         [--no-pack] [--list]
#
# All sources are coherent=False (incoherent direct deposit) so the coherent
# Huygens gather never trips GatherError at the runner's low ray counts.
# Geometry follows make_demos: the beam travels +x; a detector records on its
# -x face. Detectors on rotated/folded arms MUST pin detector_face in
# simparams (the extractor's closest-to-world-origin auto-pick grabs a thin
# EDGE face on a rotated detector and silently detects 0 mW).
#
# Reuses make_demos.Demo / rot_z / resolve_detector_pins verbatim so the two
# builders share one op path and one post-save face-resolution mechanism.
# =============================================================================
import argparse
import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import make_demos as md  # noqa: E402  (Demo, rot_z, resolve_detector_pins, FcClient, miewb_tool)
import miewb_tool  # noqa: E402

Demo = md.Demo
rot_z = md.rot_z
PRIMDIR = REPO / "primitives"


# ---------------------------------------------------------------------------
# the templates — each returns its simparams dict (detector_face pins are
# appended post-save from the pin_detector calls, exactly like make_demos)
# ---------------------------------------------------------------------------
def scene_mat_transmissive(d):
    """Material test (transmissive): laser -> BK7 window at normal incidence
    -> detector on the transmitted arm. Placeholder material 'bk7'; the runner
    swaps it for any transmissive glass/crystal."""
    d.add("laser_collimated", "Laser", pos=(-30, 0, 0),
          params={"diameter": 8.0}, props={"coherent": False})
    d.add("window", "Window", pos=(0, 0, 0),
          params={"width": 25.0, "thickness": 3.0},
          props={"material": "bk7"})
    d.add("detector_plane", "Detector", pos=(30, 0, 0),
          params={"width": 30.0})
    return {"preset": "quick"}


def scene_mat_metal_45(d):
    """Material test (metal/reflective): laser -> window plate rotated 45deg
    about z with a metal 'material' (Fresnel-reflects most incident power)
    -> detector on the 90deg REFLECTED arm (-y). Transmission through a solid
    metal plate is ~0, so the detector sits on the reflected beam; its face is
    pinned (rotated arm)."""
    d.add("laser_collimated", "Laser", pos=(-30, 0, 0),
          params={"diameter": 8.0}, props={"coherent": False})
    # +x beam off a plate whose normal is at 45deg (front normal +x rotated
    # +45 about z) reflects to -y: d - 2(d.n)n = (0,-1,0).
    d.add("window", "Mirror", pos=(0, 0, 0), rot_deg=45.0,
          params={"width": 25.0, "thickness": 3.0},
          props={"material": "aluminum"})
    # beam arrives travelling -y; detector -x face must point +y -> rot -90.
    d.add("detector_plane", "Detector", pos=(0, -30, 0), rot_deg=-90.0,
          params={"width": 30.0})
    d.pin_detector("Detector", (0.0, -1.0, 0.0))
    return {"preset": "quick"}


def scene_crystal_waveplate(d):
    """Crystal/waveplate test: 45deg-linear laser -> quartz waveplate
    (crystal_axis 0,0,1) -> ideal linear analyzer (axis 0,0,1) -> detector.
    The analyzer axis coincides with one waveplate eigen-axis (+z), so ~half
    the input transmits regardless of retardance; the runner swaps the
    waveplate 'material' (uniaxial crystal)."""
    d.add("laser_collimated", "Laser", pos=(-30, 0, 0),
          params={"diameter": 8.0},
          props={"coherent": False, "polarization": "linear:45"})
    d.add("waveplate", "Waveplate", pos=(0, 0, 0),
          props={"material": "quartz", "crystal_axis": "0,0,1"})
    d.add("polarizer_plate", "Analyzer", pos=(15, 0, 0),
          props={"polarizer": "ideal_linear", "polarizer_axis": "0,0,1"})
    d.add("detector_plane", "Detector", pos=(40, 0, 0),
          params={"width": 30.0})
    return {"preset": "quick"}


def scene_coated_plate_0(d):
    """Coating test at normal incidence: laser -> BK7 window carrying a
    whole-body AR coating -> detector on the transmitted arm. Placeholder
    coating 'MgF2' (0deg AR); the runner swaps it for any 0deg coating."""
    d.add("laser_collimated", "Laser", pos=(-30, 0, 0),
          params={"diameter": 8.0}, props={"coherent": False})
    d.add("window", "Window", pos=(0, 0, 0),
          params={"width": 25.0, "thickness": 3.0},
          props={"material": "bk7", "coating": "MgF2"})
    d.add("detector_plane", "Detector", pos=(30, 0, 0),
          params={"width": 30.0})
    return {"preset": "quick"}


def scene_coated_plate_45(d):
    """Coating test at 45deg: laser -> BK7 window rotated 45deg carrying a
    whole-body 45deg beamsplitter coating -> TWO detectors: det_r on the 90deg
    reflected arm (-y), det_t straight through (+x). BOTH faces pinned. The
    runner swaps in aoi_deg=45 dichroic / laser-mirror table coatings."""
    d.add("laser_collimated", "Laser", pos=(-30, 0, 0),
          params={"diameter": 8.0}, props={"coherent": False})
    d.add("window", "Splitter", pos=(0, 0, 0), rot_deg=45.0,
          params={"width": 25.0, "thickness": 3.0},
          props={"material": "bk7", "coating": "bs_5050_vis_45"})
    d.add("detector_plane", "det_r", pos=(0, -30, 0), rot_deg=-90.0,
          params={"width": 30.0})
    d.add("detector_plane", "det_t", pos=(30, 0, 0),
          params={"width": 30.0})
    d.pin_detector("det_r", (0.0, -1.0, 0.0))
    d.pin_detector("det_t", (1.0, 0.0, 0.0))
    return {"preset": "quick"}


def scene_filter_plate(d):
    """Spectral filter test: broadband source (band narrowed to sit inside the
    passband so the template detects plenty) -> filter_plate (default
    filter=bp_550_40) -> detector. The runner swaps the filter registry name."""
    d.add("source_broadband", "Source", pos=(-30, 0, 0),
          params={"diameter": 8.0},
          props={"coherent": False, "lambdac": 550.0,
                 "lambdamin": 535.0, "lambdamax": 565.0})
    d.add("filter_plate", "Filter", pos=(0, 0, 0),
          params={"width": 25.0, "thickness": 3.0})
    d.add("detector_plane", "Detector", pos=(30, 0, 0),
          params={"width": 30.0})
    return {"preset": "quick"}


def scene_polarizer_plate(d):
    """Polarizer test: 0deg-linear laser (E along +z = e_ref for a +x beam)
    -> polarizer_plate with transmission axis 0,0,1 (parallel -> full
    transmit) -> detector. The runner swaps the polarizer registry name."""
    d.add("laser_collimated", "Laser", pos=(-30, 0, 0),
          params={"diameter": 8.0},
          props={"coherent": False, "polarization": "linear:0"})
    d.add("polarizer_plate", "Polarizer", pos=(0, 0, 0),
          props={"polarizer": "ideal_linear", "polarizer_axis": "0,0,1"})
    d.add("detector_plane", "Detector", pos=(30, 0, 0),
          params={"width": 30.0})
    return {"preset": "quick"}


def scene_grating_plate(d):
    """Grating test: laser -> transmission grating_plate (default
    grating=Face1=600:v, CLI form so it runs standalone) -> a WIDE detector
    far downstream so the diffracted orders separate. Not rotated, so the
    detector face auto-picks fine (no pin). The runner swaps in @<name>
    registry forms on the same face key."""
    d.add("laser_collimated", "Laser", pos=(-30, 0, 0),
          params={"diameter": 8.0}, props={"coherent": False})
    # keep the primitive-default grating (Face1=600:v); do NOT edit dims.
    d.add("grating_plate", "Grating", pos=(0, 0, 0))
    d.add("detector_plane", "Detector", pos=(50, 0, 0),
          params={"width": 80.0})
    return {"preset": "quick"}


def scene_led_source(d):
    """LED source test.

    TODO(led-primitive): another workstream is generating led_*.FCStd source
    primitives (e.g. led_green_527). When a green led_* primitive exists, this
    template should import THAT primitive directly instead of the
    source_broadband fallback below. Detection logic (glob primitives/led_*)
    is already wired; only the props/params for the real primitive need to be
    filled in once its meta is known.
    """
    led = _find_green_led()
    if led is not None:
        # a real LED primitive exists — use it verbatim (its own default
        # spectrum/power carry the LED characteristics)
        d.add(led, "LED", pos=(0, 0, 0), props={"coherent": False})
        d.note("led_source: used LED primitive %r" % led)
    else:
        # fallback: broadband disc shaped to a green LED (peak 527 nm,
        # FWHM 32 nm -> +-1 sigma = +-FWHM/2.3548 ~= +-13.6 nm)
        half = 32.0 / 2.3548
        d.add("source_broadband", "LED", pos=(0, 0, 0),
              params={"diameter": 8.0},
              props={"coherent": False, "lambdac": 527.0,
                     "lambdamin": round(527.0 - half, 1),
                     "lambdamax": round(527.0 + half, 1)})
        d.note("led_source: NO led_* primitive found; used source_broadband "
               "fallback (527 nm, FWHM 32 nm). Switch to the LED primitive "
               "once it exists (see TODO in scene_led_source).")
    d.add("detector_plane", "Detector", pos=(30, 0, 0),
          params={"width": 30.0})
    return {"preset": "quick"}


def _find_green_led():
    """Return the kind name of a green LED primitive if one is shipped, else
    None. Another agent generates led_*.FCStd concurrently."""
    hits = sorted(glob.glob(str(PRIMDIR / "led_*.FCStd")))
    kinds = [Path(h).stem for h in hits]
    for k in kinds:
        if "green" in k or "527" in k:
            return k
    return kinds[0] if kinds else None


SCENES = {
    "mat_transmissive": scene_mat_transmissive,
    "mat_metal_45": scene_mat_metal_45,
    "crystal_waveplate": scene_crystal_waveplate,
    "coated_plate_0": scene_coated_plate_0,
    "coated_plate_45": scene_coated_plate_45,
    "filter_plate": scene_filter_plate,
    "polarizer_plate": scene_polarizer_plate,
    "grating_plate": scene_grating_plate,
    "led_source": scene_led_source,
}


# ---------------------------------------------------------------------------
# build one template (mirrors make_demos.build_demo, minus the corrector)
# ---------------------------------------------------------------------------
def build_scene(name, outdir, pack=True):
    outdir.mkdir(parents=True, exist_ok=True)
    fcstd = outdir / ("%s.FCStd" % name)
    if fcstd.exists():
        fcstd.unlink()
    fc = md.FcClient()
    try:
        scene = Demo(fc, fcstd)
        simparams = SCENES[name](scene)
        scene.save()
    finally:
        fc.shutdown()
    if scene.detector_pins or scene.grating_pins:
        det, grat = md.resolve_detector_pins(
            fcstd, (scene.detector_pins, scene.grating_pins))
        if det:
            simparams["detector_face"] = det
        if grat:
            simparams["grating"] = grat
    if pack:
        miewb_tool.pack_miewb(fcstd, outdir / ("%s.MieWB" % name),
                              simparams=simparams)
    print("[libtest] %-20s -> %s%s" % (name, fcstd.name,
                                       " (+MieWB)" if pack else ""),
          flush=True)
    return scene.notes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="all",
                   help="template name or 'all' (see --list)")
    p.add_argument("--outdir", default=str(REPO / "demos" / "library_tests"))
    p.add_argument("--no-pack", action="store_true",
                   help="skip the .MieWB packing step")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        for name in SCENES:
            print(name)
        return 0
    names = list(SCENES) if args.scene == "all" else [args.scene]
    all_notes = []
    for name in names:
        if name not in SCENES:
            p.error("unknown template %r" % name)
        all_notes += build_scene(name, Path(args.outdir),
                                 pack=not args.no_pack)
    for n in all_notes:
        print("  note: " + n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
