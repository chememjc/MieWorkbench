#!/usr/bin/env python3
# =============================================================================
# cli_specs.py — single source of truth for the four pipeline stage CLIs.
#
# Stdlib-only (like common.py): imported by every interpreter stack that
# builds one of these parsers (system python3 for run_pipeline.py, the
# optics env for run_trace.py/post_process.py, pvpython for make_viz.py),
# so it must import NOTHING beyond the python standard library plus this
# project's own stdlib-only modules (common.py, viz_configs.py). No numpy,
# no FreeCAD, no paraview, no argcomplete.
#
# Each stage script builds its argparse.ArgumentParser by calling
# build_parser(<stage>) and then calling .parse_args() (or .parse_args(argv))
# on the result; any post-parse validation (e.g. run_pipeline's paired
# --var/--min/--max/--n count check) stays in the owning script. Keeping the
# option definitions here — rather than duplicated across four scripts —
# lets the MieWorkbench GUI introspect exactly the same parser objects (for
# its config-matrix form generation, `--help`-equivalent tooltips, choices,
# defaults, etc.) that each stage script actually parses argv with.
#
# Self-check:  python3 scripts/cli_specs.py
#   (builds all four parsers under plain python3; must not import anything
#   heavy to do so)
# =============================================================================
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (stdlib-only shared contract hub)

STAGES = ("pipeline", "trace", "post", "viz", "optimize", "tolerance")


def _workers_arg(s):
    """argparse type for --workers: the literal string 'auto' or a positive
    int. Kept stdlib (no numpy/os) so cli_specs stays importable under every
    interpreter stack and the self-check passes."""
    if s == "auto":
        return "auto"
    try:
        v = int(s)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "--workers must be 'auto' or a positive integer (got %r)" % (s,))
    if v < 1:
        raise argparse.ArgumentTypeError(
            "--workers must be >= 1 (got %d)" % v)
    return v


def parse_wavefront_point(s):
    """'X_MM,Y_MM' -> (float, float). argparse type= for --wavefront-point
    (render_wavefront's optional image-point override, post stage)."""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "invalid --wavefront-point %r (expected X_MM,Y_MM)" % s)
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "invalid --wavefront-point %r (expected X_MM,Y_MM)" % s)


IMAGING_PRODUCTS = ("distortion", "vignetting", "field_curves",
                    "telecentricity")


def parse_imaging_products(s):
    """Comma list -> validated tuple for --imaging-products (post stage's
    field-imaging renderers; 'all' = every product). argparse type=."""
    names = [t.strip() for t in s.split(",") if t.strip()]
    if not names:
        raise argparse.ArgumentTypeError(
            "--imaging-products got an empty list (expected a comma list "
            "of %s or 'all')" % ",".join(IMAGING_PRODUCTS))
    if names == ["all"]:
        return IMAGING_PRODUCTS
    bad = [n for n in names if n not in IMAGING_PRODUCTS]
    if bad:
        raise argparse.ArgumentTypeError(
            "unknown --imaging-products entr%s %s (know: %s, or 'all')"
            % ("y" if len(bad) == 1 else "ies", ",".join(bad),
               ",".join(IMAGING_PRODUCTS)))
    # de-duplicate, keep the canonical order
    return tuple(n for n in IMAGING_PRODUCTS if n in names)


IMAGE_SIM_COHERENCE = ("incoherent", "coherent", "partial")


def _add_image_sim_args(g):
    """The three --image-sim flags (imaging-analysis round), shared
    verbatim by the pipeline and post parsers (post_process consumes
    them; run_pipeline forwards them). --image-sim REQUIRES --save-fields
    (run_pipeline.main validates the pair up front; post_process hard-
    errors if no saved coherent field exists)."""
    g.add_argument("--image-sim", default=None, metavar="PATH",
                   help="simulate imaging an input scene image through "
                        "the modeled system: PATH is a greyscale image "
                        "(PNG/JPG/TIFF via Pillow, or a 2-D .npy array) "
                        "convolved with the amplitude PSF taken from the "
                        "dominant coherent gather key's saved detector "
                        "field. REQUIRES --save-fields (the coherent "
                        "field map is the PSF source) and a coherent "
                        "point/collimated source. Outputs: imaging/"
                        "image_sim_<mode>.png + image_sim_input.png + a "
                        "report.json 'image_sim' block. post stage only")
    g.add_argument("--image-sim-coherence", default="incoherent",
                   choices=list(IMAGE_SIM_COHERENCE),
                   help="--image-sim illumination model: 'incoherent' "
                        "(default; intensities convolve with |h|^2 — the "
                        "classic MTF-limited image), 'coherent' "
                        "(amplitudes convolve with h — sharp edges with "
                        "ringing), or 'partial' (Abbe source-integration "
                        "over a disc source of normalized radius "
                        "--image-sim-sigma)")
    g.add_argument("--image-sim-sigma", type=float, default=0.5,
                   help="partial-coherence factor sigma for "
                        "--image-sim-coherence=partial: illumination-"
                        "source radius over pupil radius in pupil-"
                        "frequency coordinates (the standard "
                        "NA_cond/NA_obj). 0 = fully coherent, >~2 = "
                        "effectively incoherent (default 0.5)")


TIME_PRODUCTS = ("pulse", "spectrogram", "streak", "cube")

# Comma-list product flags, dest -> (canonical choices, whether 'none' is
# a meaningful explicit value). The GUI's ConfigMatrix renders these as a
# row of per-product checkboxes instead of a free-text QLineEdit; anything
# else introspecting the parsers can use it the same way. 'none' matters
# only for --time-products (it suppresses the pulsed-scene auto-default;
# an omitted --imaging-products already means "none").
PRODUCT_FLAG_CHOICES = {
    "time_products": (TIME_PRODUCTS, True),
    "imaging_products": (IMAGING_PRODUCTS, False),
}

# --time-bins preset scaling (pulsed-optics P4): applied by run_pipeline's
# trace_cmd when --time-bins is not given (common.PRESETS itself is
# unchanged; the trace parser's own default covers direct run_trace.py use).
TIME_BINS_PRESET = {"quick": 128, "normal": 256, "detailed": 512}


def parse_time_products(s):
    """Comma list -> validated tuple for --time-products (trace stage's
    time-binned detector products). 'all' = every product, 'none' = the
    empty tuple (explicitly suppresses the pulsed-scene auto-default).
    argparse type=."""
    names = [t.strip() for t in s.split(",") if t.strip()]
    if not names:
        raise argparse.ArgumentTypeError(
            "--time-products got an empty list (expected a comma list of "
            "%s, or 'all'/'none')" % ",".join(TIME_PRODUCTS))
    if names == ["all"]:
        return TIME_PRODUCTS
    if names == ["none"]:
        return ()
    bad = [n for n in names if n not in TIME_PRODUCTS]
    if bad:
        raise argparse.ArgumentTypeError(
            "unknown --time-products entr%s %s (know: %s, or 'all'/'none')"
            % ("y" if len(bad) == 1 else "ies", ",".join(bad),
               ",".join(TIME_PRODUCTS)))
    # de-duplicate, keep the canonical order
    return tuple(n for n in TIME_PRODUCTS if n in names)


def parse_time_window(s):
    """'T0,T1' (ns, floats, T0 < T1) -> (float, float). argparse type= for
    --time-window."""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "invalid --time-window %r (expected T0,T1 in ns)" % s)
    try:
        t0, t1 = float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "invalid --time-window %r (expected T0,T1 in ns)" % s)
    if not t1 > t0:
        raise argparse.ArgumentTypeError(
            "--time-window %r: T1 must be > T0" % s)
    return t0, t1


def _add_time_product_args(g, bins_default):
    """The five pulsed-optics time-product flags, shared verbatim by the
    pipeline and trace parsers (bins_default None on the pipeline parser:
    run_pipeline preset-scales it via TIME_BINS_PRESET before forwarding)."""
    g.add_argument("--time-products", default=None,
                   type=parse_time_products, metavar="LIST",
                   help="comma list of time-binned detector products to "
                        "compute: %s, or 'all'/'none'. Default: "
                        "'pulse,spectrogram' when the scene has a pulsed "
                        "source (pulse_duration set), nothing otherwise; "
                        "'none' suppresses the auto-default. Any active "
                        "product tracks per-ray group delay and forces "
                        "the Python engine (reason recorded in case.json)"
                        % ",".join(TIME_PRODUCTS))
    g.add_argument("--time-bins", type=int, default=bins_default,
                   metavar="N",
                   help="time bins per product (default: preset-scaled "
                        "%s via run_pipeline; %s when run_trace.py is "
                        "invoked directly)"
                        % ("/".join("%s=%d" % kv
                                    for kv in sorted(TIME_BINS_PRESET.items())),
                           TIME_BINS_PRESET["normal"]))
    g.add_argument("--time-window", default=None, type=parse_time_window,
                   metavar="T0,T1",
                   help="explicit time window in ns (floats, T0<T1); "
                        "default: auto — the exact record arrival span "
                        "padded by 3x the widest envelope kernel width. "
                        "Kernels clipped by the window edge are "
                        "renormalized over the in-window bins (energy "
                        "conserving); fully-out-of-window power is "
                        "reported in the .h5 time_excluded_W attr")
    g.add_argument("--time-cube-res", type=int, default=256, metavar="N",
                   help="spatial cap for the 'cube' product: the (t, y, x) "
                        "cube is binned down to at most NxN pixels "
                        "(default 256; binning factor recorded in attrs)")
    g.add_argument("--time-envelope", default="analytic",
                   choices=["analytic", "histogram"],
                   help="arrival-record envelope: 'analytic' (default) "
                        "deposits a per-record Gaussian whose FWHM "
                        "combines the source pulse duration with the "
                        "record's accumulated GDD x stratum bandwidth; "
                        "'histogram' is a plain weighted histogram of "
                        "arrival times")
    g.add_argument("--gdd-budget", action="store_true",
                   help="emit the per-element dispersion budget (mean "
                        "traced path, group delay, GDD, TOD at the "
                        "reference source's center wavelength + totals + "
                        "pulse-broadening estimate) into case.json; "
                        "post_process renders the table/CSV. Covers "
                        "MATERIAL dispersion only — geometric GDD "
                        "(gratings, prisms) shows up in the traced time "
                        "products instead. The budget is computed "
                        "automatically whenever time products are active; "
                        "this flag additionally forces group-delay "
                        "tracking on a CW scene (Python engine)")


# ---------------------------------------------------------------------------
# pipeline  (scripts/run_pipeline.py)
# ---------------------------------------------------------------------------
def _build_pipeline_parser():
    STEPS_ORDER = ["extract", "trace", "post", "viz"]

    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Chain FreeCAD -> optics -> ParaView optical ray "
                    "tracer stages for one or more models.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, metavar="FCSTD",
                   help="one or more .FCStd paths or globs (model stem = "
                        "file name without .FCStd)")
    p.add_argument("--preset", default="quick",
                   choices=sorted(common.PRESETS),
                   help="fidelity preset filling rays/resolution/nlambda/"
                        "spectral-bins/viz-rays defaults (default: quick)")
    p.add_argument("--tag", default=None,
                   help="appended to the case name: results/<model>/"
                        "<preset>-<tag>/")
    p.add_argument("--steps", default=",".join(STEPS_ORDER), metavar="LIST",
                   help="comma-separated subset of %s to run, executed in "
                        "that fixed order (default: all)"
                        % ",".join(STEPS_ORDER))

    g = p.add_argument_group("parameter sweep (stage: permute, before "
                             "extract)")
    g.add_argument("--var", action="append", default=[],
                   help="spreadsheet cell alias to sweep (repeatable, "
                        "paired in order with --min/--max/--n)")
    g.add_argument("--min", action="append", default=[], type=float)
    g.add_argument("--max", action="append", default=[], type=float)
    g.add_argument("--n", action="append", default=[], type=int)
    g.add_argument("--sweep-mode", default="product",
                   choices=["product", "zip"],
                   help="how multiple --var combinations combine: "
                        "'product' = cartesian, one variant per "
                        "combination of every variable's values (default, "
                        "historical behavior); 'zip' = variables advance "
                        "together, one variant per index (value lists "
                        "must have equal length, or length 1 to "
                        "broadcast) — see common.sweep_combos")

    g = p.add_argument_group("physics options (stage: trace)")
    g.add_argument("--dry-run", action="store_true",
                   help="trace builds estimates but does not run; post/viz "
                        "are then skipped per model with a NOTICE")
    g.add_argument("--seeds", type=int, default=None)
    g.add_argument("--rays", type=float, default=None,
                   help="primary rays per source (default: from --preset)")
    g.add_argument("--resolution", type=int, default=None,
                   help="detector grid resolution (default: from --preset)")
    g.add_argument("--nlambda", type=int, default=None,
                   help="wavelength strata (default: from --preset)")
    g.add_argument("--spectral-bins", type=int, default=None,
                   help="detector spectral bins (default: from --preset)")
    g.add_argument("--max-reflections", type=int, default=None,
                   help="reflection/TIR generation cap per ray (default 6; "
                        "raise for many-bounce systems, e.g. ~300 for a "
                        "75 mm fiber segment)")
    g.add_argument("--viz-rays", type=int, default=None,
                   help="absolute viz-ray cap per source (set this to "
                        "override --viz-density; preset value acts as the "
                        "density cap instead)")
    g.add_argument("--viz-density", type=float, default=None,
                   help="viz rays per mm^2 of source emit area "
                        "(default 1.0; visualization only)")
    g.add_argument("--viz-pattern", default=None, metavar="SPEC",
                   help="deterministic viz-ray layout instead of random: "
                        "'rings:dr=<mm>:nper=<N>[:nrings=<K>]' = central "
                        "ray + concentric rings every dr mm, N rays per "
                        "ring; or 'fan[:n=<K>]' (default K=5) = central ray "
                        "+ up to 4 cardinal rays + rim fillers "
                        "(visualization only — physics unaffected)")
    g.add_argument("--backend", default=None,
                   choices=["auto", "torch", "numpy"])
    g.add_argument("--engine", default=None,
                   choices=["auto", "python", "c"],
                   help="trace engine: 'c' = the compiled OpenMP/CUDA "
                        "engine (cengine/build/miewb-trace), 'python' = "
                        "the reference numpy engine, 'auto' (default) = C "
                        "when the binary exists and every feature the "
                        "scene uses is ported, else Python (choice + "
                        "reason logged and recorded in case.json)")
    g.add_argument("--importance-aim", action="store_true",
                   help="C-engine variance reduction: birth-cull source "
                        "samples that would immediately escape (unbiased; "
                        "see run_trace --help)")
    g.add_argument("--rough-fresnel", default=None,
                   choices=["micro", "macro"],
                   help="roughness-lobe Fresnel model (default micro)")
    g.add_argument("--ray-differentials", action="store_true",
                   help="per-ray wavefront-patch dA tracking (exact "
                        "gather normalization; costs memory)")
    g.add_argument("--gather-occlusion", action="store_true",
                   help="shadow-test gather samples against scene bodies")
    g.add_argument("--no-pol-scatter", action="store_true",
                   help="legacy unpolarized Mie azimuth sampling")
    g.add_argument("--mesh-flat-normals", action="store_true")
    g.add_argument("--temperature", type=float, default=None, metavar="DEG_C",
                   help="scene operating temperature in deg C; shifts glasses "
                        "with a thermo-optic model via Schott TIE-19 dn/dT "
                        "(default: each material's reference temp, no shift; "
                        "a per-body 'temperature' property overrides this). "
                        "Routes the run to the Python engine.")
    g.add_argument("--save-fields", action="store_true",
                   help="save complex Ex/Ey detector field maps "
                        "(enables Stokes polarization maps in post)")
    g.add_argument("--save-fields-detectors", default=None,
                   metavar="LABEL[,LABEL...]",
                   help="restrict --save-fields to these detector labels "
                        "(comma-separated; default: every detector). "
                        "Labels are detector face ids, e.g. "
                        "'Body001.Pad.Face3' (same string --detector-face "
                        "uses and post_process.py safes into "
                        "det_<label>_*.png); an unknown label is a hard "
                        "error naming the scene's available detector "
                        "labels. Forwarded to the trace stage verbatim")
    g.add_argument("--strict-analytic", action="store_true",
                   help="hard-error on mesh-type faces (v1 behavior)")
    g.add_argument("--optical-properties", default=None,
                   help="override the opticalproperties/ library root")
    g.add_argument("--source-face", action="append", default=[],
                   metavar="Body.Feature.FaceN")
    g.add_argument("--detector-face", action="append", default=[],
                   metavar="Body.Feature.FaceN",
                   help="add an extra, transparent detector screen on any "
                        "face without disturbing its physical interaction. "
                        "Prefer the authoring-time `detector_face` BODY "
                        "property (docs/RAYTRACER.md §5.1/§5.2) when you "
                        "just need to retarget a detector body's own "
                        "recorded primary face: it replaces that face in "
                        "place (no extra screen) and keeps the scene "
                        "C-engine-routable, whereas this CLI flag adds an "
                        "extra screen on top of the auto-pick and forces "
                        "the Python engine (extra_detector_faces is not "
                        "in the C engine's ported feature set)")
    g.add_argument("--grating", action="append", default=[], metavar="SPEC")
    g.add_argument("--rough", action="append", default=[], metavar="SPEC")
    g.add_argument("--particles", default=None, metavar="SPEC")
    g.add_argument("--particle-threshold", type=float, default=None)
    g.add_argument("--suppress-body", action="append", default=[],
                   metavar="BODY")

    g = p.add_argument_group("time-domain products (pulsed optics; "
                             "stage: trace)")
    _add_time_product_args(g, bins_default=None)

    g = p.add_argument_group("display options (stages: post, viz)")
    g.add_argument("--dim-rays", default="off",
                   choices=["off", "linear", "sqrt"],
                   help="dim ray renders by remaining power relative to "
                        "each ray's power at the source: opacity = "
                        "P/P_birth (linear) or sqrt(P/P_birth) "
                        "(perceptual); applies to rays_xy.png and the 3D "
                        "viz renders (default: off)")
    g.add_argument("--dim-rays-floor", type=float, default=0.0,
                   metavar="PCT",
                   help="minimum segment opacity in percent (0-100) when "
                        "--dim-rays is on (default 0 = fade fully to "
                        "invisible at zero power)")
    g.add_argument("--photometric", action="store_true",
                   help="also render a CIE-photopic-weighted illuminance "
                        "image per detector (det_<label>_lux.png) and a "
                        "luminous_flux_lm/peak+mean_illuminance_lux "
                        "report block (post stage only)")
    g.add_argument("--spectrometer", action="store_true",
                   help="also render a power-weighted wavelength-centroid "
                        "map and a lambda(x) dispersion fit per detector "
                        "(det_<label>_lambda_map.png, "
                        "spectra/lambda_vs_x_<label>.png; post stage only)")
    g.add_argument("--viz-generations", type=int, default=None,
                   help="post stage: declutter rays_xy.png to "
                        "reconstructed-generation <= N segments only "
                        "(default: all generations, unchanged behavior; "
                        "forwarded to post_process.py's --viz-generations)")
    g.add_argument("--views", default=None,
                   help="viz stage: comma-separated view names to render, "
                        "e.g. overview3d,top,side,detector_closeup,"
                        "turntable,rays_polmode (default: all views; "
                        "forwarded to make_viz.py's --views)")
    g.add_argument("--smoke", action="store_true",
                   help="viz stage: render only the 'overview3d' view at "
                        "800x600 for a fast end-to-end check (forwarded "
                        "to make_viz.py's --smoke)")

    g = p.add_argument_group("analysis / export options")
    g.add_argument("--emit-csv", action="store_true",
                   help="also write results/<case>/data/*.csv alongside "
                        "every chart (plus data/index.csv mapping file -> "
                        "entity/chart/units/provenance); post stage only")
    g.add_argument("--export-rays", action="store_true",
                   help="capture per-ray landing records at every detector "
                        "(seed 0 only) to results/<case>/rays_full.npz; "
                        "post then renders spot diagrams + ray/OPD fans "
                        "into results/<case>/analysis/")
    g.add_argument("--export-rays-max", type=int, default=2000000,
                   metavar="N",
                   help="per-detector cap on exported rays (default "
                        "2000000); above it a seeded uniform-random subset "
                        "is kept and the fraction recorded in the npz meta")
    g.add_argument("--ghost-analysis", action="store_true",
                   help="ghost / stray-light analysis: track reflection "
                        "face-id history (seed 0 only; implies "
                        "--export-rays) so post ranks multi-bounce ghost "
                        "paths by detected power into "
                        "results/<case>/analysis/")
    g.add_argument("--wavefront-point", default=None,
                   type=parse_wavefront_point, metavar="X_MM,Y_MM",
                   help="override render_wavefront's image (wavefront "
                        "reference) point, in detector-grid-frame mm (the "
                        "same u=pos.xhat, v=pos.yhat convention the spot/"
                        "fan renders use); default: each coherent key's "
                        "power-weighted landing centroid. post stage only, "
                        "requires --export-rays")
    g.add_argument("--wavefront-pupil", default="source",
                   choices=["source", "exit_pupil"],
                   help="render_wavefront's pupil model: 'source' "
                        "(default, unchanged: normalized birth position "
                        "on the emitting face) or 'exit_pupil' (chief-ray/"
                        "exit-pupil search over the field bundles, "
                        "analysis_imaging.py; falls back to 'source' with "
                        "a report note when the solve degenerates). post "
                        "stage only, requires --export-rays")
    g.add_argument("--imaging-products", default=None,
                   type=parse_imaging_products, metavar="LIST",
                   help="comma list of field-imaging products to render "
                        "in the post stage (requires --export-rays and a "
                        "multi-source field fan, e.g. the field-angle fan "
                        "wizard): %s, or 'all'. Each writes analysis/ "
                        "PNGs + report.json 'imaging' blocks (+ CSVs "
                        "under --emit-csv). Default: none"
                        % ",".join(IMAGING_PRODUCTS))
    _add_image_sim_args(g)

    g = p.add_argument_group("execution / orchestration")
    g.add_argument("--keep-going", action="store_true",
                   help="on a stage failure, print FAILED and continue "
                        "with the next model instead of aborting (exit "
                        "code still nonzero)")
    g.add_argument("--print-only", action="store_true",
                   help="compose and print every stage command without "
                        "running anything")
    g.add_argument("--workers", type=_workers_arg, default="auto",
                   metavar="N",
                   help="parallel trace shards forwarded to run_trace.py "
                        "(default 'auto' = max(1, cpu_count-2); '1' = single-"
                        "process). The coherent gather always runs single-"
                        "process in the parent.")
    return p


# ---------------------------------------------------------------------------
# trace  (scripts/run_trace.py)
# ---------------------------------------------------------------------------
def _build_trace_parser():
    p = argparse.ArgumentParser(description="optical ray trace stage")
    p.add_argument("--model-json", required=True)
    p.add_argument("--case-dir", required=True)
    p.add_argument("--rays", type=float, default=1e5,
                   help="primary rays PER SOURCE")
    p.add_argument("--nlambda", type=int, default=5)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--spectral-bins", type=int, default=16)
    p.add_argument("--max-reflections", type=int, default=6)
    p.add_argument("--power-floor", type=float, default=1e-4)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--backend", default="auto",
                   choices=["auto", "torch", "numpy"])
    p.add_argument("--engine", default="auto",
                   choices=["auto", "python", "c"],
                   help="trace engine: 'c' = the compiled OpenMP/CUDA "
                        "engine (cengine/build/miewb-trace), 'python' = "
                        "the reference numpy engine, 'auto' (default) = C "
                        "when the binary exists and every feature the "
                        "scene uses is ported, else Python (choice + "
                        "reason logged and recorded in case.json). "
                        "--workers applies to the Python engine only; the "
                        "C engine threads internally (OpenMP).")
    p.add_argument("--importance-aim", action="store_true",
                   help="C-engine variance reduction (opt-in): candidate "
                        "source samples whose ray misses the scene "
                        "bounding box are culled AT BIRTH with their power "
                        "credited straight to 'escaped' (exactly the fate "
                        "they would have had), and the candidate count is "
                        "raised so the requested --rays all do useful "
                        "work. Unbiased: every expectation is unchanged; "
                        "only the MC noise on detectors drops. Python "
                        "engine ignores the flag.")
    p.add_argument("--workers", type=_workers_arg, default="auto",
                   metavar="N",
                   help="parallel trace shards (spawned processes) for the "
                        "trace loop; 'auto' = max(1, cpu_count-2), '1' = "
                        "single-process, bit-identical to the pre-sharding "
                        "path. Each worker traces rays/N primaries with its "
                        "own RNG stream; the parent merges the shards and "
                        "runs the coherent Huygens gather ONCE (single-"
                        "process, torch-CUDA), so N>1 results are "
                        "statistically equivalent (not bit-identical) to N=1.")
    p.add_argument("--viz-rays", type=int, default=None,
                   help="absolute viz-ray cap per source (overrides "
                        "--viz-density when set)")
    p.add_argument("--viz-density", type=float, default=1.0,
                   help="viz rays per mm^2 of source emit area "
                        "(visualization only — physics unaffected)")
    p.add_argument("--viz-rays-max", type=int, default=20000,
                   help="hard cap on density-derived viz rays per source")
    p.add_argument("--viz-pattern", default=None, metavar="SPEC",
                   help="deterministic viz-ray layout instead of random: "
                        "'rings:dr=<mm>:nper=<N>[:nrings=<K>]' = central "
                        "ray + concentric rings every dr mm, N rays per "
                        "ring; or 'fan[:n=<K>]' (default K=5) = central ray "
                        "+ up to 4 cardinal rays + rim fillers, traced in "
                        "a separate viz-only pass "
                        "(visualization only — physics unaffected)")
    p.add_argument("--ray-differentials", action="store_true",
                   help="track per-ray wavefront patch areas (Igehy) so "
                        "the gather uses exact per-sample dA instead of "
                        "the source-referenced approximation (+96 B/ray)")
    p.add_argument("--no-pol-scatter", action="store_true",
                   help="legacy unpolarized Mie azimuth sampling "
                        "(default: sample azimuth from the polarized "
                        "differential cross-section)")
    p.add_argument("--rough-fresnel", default="micro",
                   choices=["micro", "macro"],
                   help="roughness-lobe Fresnel: microfacet-local per-"
                        "polarization (physical) or legacy nominal-angle "
                        "scalar average")
    p.add_argument("--source-face", action="append", default=[],
                   help="override: Body.Feature.FaceN (matched to the "
                        "source body owning that face)")
    p.add_argument("--detector-face", action="append", default=[],
                   help="add an extra, transparent detector screen on any "
                        "face (Body.Feature.FaceN) without disturbing its "
                        "physical interaction. Prefer the authoring-time "
                        "`detector_face` BODY property (docs/RAYTRACER.md "
                        "§5.1/§5.2) when you just need to retarget a "
                        "detector body's own recorded primary face: it "
                        "replaces that face in place (no extra screen) "
                        "and keeps the scene C-engine-routable, whereas "
                        "this CLI flag adds an extra screen on top of the "
                        "auto-pick and forces the Python engine "
                        "(extra_detector_faces is not in the C engine's "
                        "ported feature set)")
    p.add_argument("--grating", action="append", default=[])
    p.add_argument("--rough", action="append", default=[])
    p.add_argument("--particles", default=None)
    p.add_argument("--particle-threshold", type=float, default=2e5,
                   help="explicit-sphere mode below this count (matches "
                        "the brute-force traversal cap), continuum above")
    p.add_argument("--suppress-body", action="append", default=[])
    p.add_argument("--min-eff-samples", type=float, default=1000.0)
    p.add_argument("--no-gather-gate", action="store_true")
    p.add_argument("--temperature", type=float, default=None, metavar="DEG_C",
                   help="scene operating temperature in deg C; shifts glasses "
                        "with a thermo-optic model via Schott TIE-19 dn/dT "
                        "(default: material reference temp, no shift; a "
                        "per-body 'temperature' property overrides). Python "
                        "engine only.")
    p.add_argument("--save-fields", action="store_true",
                   help="save per-(source,lam,pol) complex Ex/Ey field "
                        "maps into detectors/<label>.h5 fields/ groups "
                        "(post_process renders Stokes maps from them; "
                        "seed0 only; large files at high resolution)")
    p.add_argument("--save-fields-detectors", default=None,
                   metavar="LABEL[,LABEL...]",
                   help="restrict --save-fields' field-map writes to "
                        "these detector labels (comma-separated; default: "
                        "every detector — identical to bare --save-fields). "
                        "Labels are detector face ids, e.g. "
                        "'Body001.Pad.Face3' (matching DetectorGrid.label/ "
                        "--detector-face); an unknown label is a hard "
                        "error naming the scene's available detector "
                        "labels. No effect without --save-fields. Forces "
                        "the Python engine (the C engine always saves "
                        "fields for every detector under --save-fields)")
    p.add_argument("--gather-occlusion", action="store_true",
                   help="ray-cast each gather sample->detector-tile segment "
                        "against scene bodies and shadow blocked pairs "
                        "(opaque occluders, tile-quantized; see gather.py)")
    p.add_argument("--optical-properties", default=None,
                   help="override the opticalproperties/ library root")
    p.add_argument("--strict-analytic", action="store_true",
                   help="hard-error on mesh-type faces (v1 behavior) "
                        "instead of tracing them with the BVH")
    p.add_argument("--mesh-flat-normals", action="store_true",
                   help="flat facet normals on mesh faces (default: "
                        "angle-weighted smoothed vertex normals)")
    p.add_argument("--dry-run", action="store_true")

    g = p.add_argument_group("time-domain products (pulsed optics)")
    _add_time_product_args(g, bins_default=TIME_BINS_PRESET["normal"])

    g = p.add_argument_group("analysis / export options")
    g.add_argument("--export-rays", action="store_true",
                   help="capture per-ray landing records at every detector "
                        "(SEED 0 ONLY, like --save-fields) into "
                        "results/<case>/rays_full.npz with per-detector "
                        "namespaced arrays + a JSON meta (grid basis, seed, "
                        "cap). Diagnostic only: the splat/gather math and "
                        "rays.npy viz contract are untouched")
    g.add_argument("--export-rays-max", type=int, default=2000000,
                   metavar="N",
                   help="per-detector cap on exported rays (default "
                        "2000000); above it a uniform-random subset drawn "
                        "with the run seed is kept and the kept fraction "
                        "recorded in the npz meta")
    g.add_argument("--ghost-analysis", action="store_true",
                   help="ghost / stray-light analysis: track the FACE-id "
                        "history of every reflection (RayBatch.refl_hist, "
                        "SEED 0 ONLY) and export it into rays_full.npz "
                        "(implies --export-rays behavior; a bare "
                        "--export-rays does NOT track history). post then "
                        "ranks multi-bounce ghost paths by detected power "
                        "into results/<case>/analysis/")
    return p


# ---------------------------------------------------------------------------
# post  (scripts/post_process.py)
# ---------------------------------------------------------------------------
def _build_post_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", required=True)
    p.add_argument("--model-json", required=True)
    p.add_argument("--viz-generations", type=int, default=None,
                  help="declutter rays_xy.png to reconstructed-generation "
                       "<= N segments only (default: all generations, "
                       "unchanged behavior). See _assign_generations.")
    p.add_argument("--dim-rays", default="off",
                   choices=["off", "linear", "sqrt"],
                   help="dim rays_xy.png segments by remaining power "
                        "relative to each ray's power at the source "
                        "(alpha = P/P_birth, or its sqrt for the "
                        "perceptual curve); default off = the existing "
                        "ensemble-percentile alpha")
    p.add_argument("--dim-rays-floor", type=float, default=0.0,
                   metavar="PCT",
                   help="minimum segment opacity in percent (0-100) when "
                        "--dim-rays is on")
    p.add_argument("--photometric", action="store_true",
                   help="also render det_<label>_lux.png (CIE-photopic "
                        "illuminance) and a photometric report block per "
                        "detector")
    p.add_argument("--spectrometer", action="store_true",
                   help="also render det_<label>_lambda_map.png (power-"
                        "weighted wavelength centroid) and a lambda(x) "
                        "dispersion fit per detector")

    g = p.add_argument_group("analysis / export options")
    g.add_argument("--emit-csv", action="store_true",
                   help="also write results/<case>/data/*.csv alongside "
                        "every chart (plus data/index.csv mapping file -> "
                        "entity/chart/units/provenance)")
    g.add_argument("--wavefront-point", default=None,
                   type=parse_wavefront_point, metavar="X_MM,Y_MM",
                   help="override render_wavefront's image (wavefront "
                        "reference) point, in detector-grid-frame mm (the "
                        "same u=pos.xhat, v=pos.yhat convention the spot/"
                        "fan renders use); default: each coherent key's "
                        "power-weighted landing centroid. Only matters "
                        "when rays_full.npz exists (--export-rays ran)")
    g.add_argument("--wavefront-pupil", default="source",
                   choices=["source", "exit_pupil"],
                   help="render_wavefront's pupil model: 'source' "
                        "(default, unchanged behavior) or 'exit_pupil' "
                        "(analysis_imaging.py chief-ray/exit-pupil "
                        "search; falls back to 'source' with a report "
                        "note when the solve degenerates — single field "
                        "point / telecentric image side)")
    g.add_argument("--imaging-products", default=None,
                   type=parse_imaging_products, metavar="LIST",
                   help="comma list of field-imaging products to render: "
                        "%s, or 'all'. HARD-REQUIRES rays_full.npz (run "
                        "the trace with --export-rays). Outputs: "
                        "analysis/imaging_*.png + report.json "
                        "detectors.<label>.imaging blocks (+ data/*.csv "
                        "under --emit-csv)" % ",".join(IMAGING_PRODUCTS))
    _add_image_sim_args(g)
    return p


# ---------------------------------------------------------------------------
# viz  (scripts/make_viz.py)
# ---------------------------------------------------------------------------
def parse_resolution(s):
    """'WIDTHxHEIGHT' -> (int, int). argparse type= for --resolution."""
    m = re.match(r"^\s*(\d+)x(\d+)\s*$", s)
    if not m:
        raise argparse.ArgumentTypeError(
            "invalid --resolution '%s' (expected WIDTHxHEIGHT)" % s)
    return int(m.group(1)), int(m.group(2))


def _build_viz_parser():
    # viz_configs.py is a pure-python/stdlib declarative library (no
    # imports at all) — safe to import here for --views/--resolution
    # defaults and help text without pulling in paraview.
    import viz_configs

    p = argparse.ArgumentParser(
        description="Batch ParaView 3D visualizations for an optical ray "
                    "tracer case directory.")
    p.add_argument("--case-dir", required=True,
                   help="results/<model>/<case> directory (run_trace.py + "
                        "post_process.py output)")
    p.add_argument("--model-json", required=True,
                   help="geometry/<model>/model.json")
    p.add_argument("--views", default=None,
                   help="Comma-separated view names to render (default: "
                        "all of %s)"
                        % ",".join(v["name"] for v in viz_configs.VIEWS))
    p.add_argument("--resolution", default=viz_configs.DEFAULT_RESOLUTION,
                   type=parse_resolution,
                   help="WIDTHxHEIGHT (default %dx%d; ignored by --smoke "
                        "and forced to 2048x2048 for detector_closeup)"
                        % viz_configs.DEFAULT_RESOLUTION)
    p.add_argument("--out", default=None,
                   help="Output directory (default: <case-dir>/viz)")
    p.add_argument("--smoke", action="store_true",
                   help="Render only the 'overview3d' view at 800x600 "
                        "(fast end-to-end smoke test)")
    p.add_argument("--skip-vtkexport", action="store_true",
                   help="Skip the OPTICS_PYTHON vtkexport prep step "
                        "(viz/*.vtp already produced by a previous run)")
    p.add_argument("--dim-rays", default="off",
                   choices=["off", "linear", "sqrt"],
                   help="dim 3D ray renders by remaining power relative "
                        "to each ray's power at the source (alpha = "
                        "P/P_birth, or its sqrt for the perceptual "
                        "curve); needs a rays.vtp with the rel_power "
                        "cell array")
    p.add_argument("--dim-rays-floor", type=float, default=0.0,
                   metavar="PCT",
                   help="minimum segment opacity in percent (0-100) when "
                        "--dim-rays is on")
    return p


# ---------------------------------------------------------------------------
# optimize  (scripts/optimize.py — merit-function optimizer over fast_eval)
# ---------------------------------------------------------------------------
# Named merit operands scripts/optimize.py understands (single source of
# truth: the GUI's OptimizePane populates its operand combo from this and
# optimize.py builds its registry over it). Any other operand string
# containing a '.' is treated as a raw flattened report.json merit key
# (fast_eval.flatten_merits naming), minimized toward its target.
OPTIMIZE_OPERANDS = ("spot_rms", "encircled_energy", "detected_power",
                     "mtf50", "focus")


def parse_var_spec(s):
    """'NAME:START:LO:HI' -> {"name","start","lo","hi"} with LO<=START<=HI.
    argparse type= for --var (optimize stage). NAME is a spreadsheet cell
    alias (bare 'alias' on the default dim sheet, or 'sheetlabel.alias' —
    exactly what permute_model --var / fast_eval address)."""
    parts = s.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "invalid --var %r (expected NAME:START:LO:HI)" % s)
    name = parts[0].strip()
    if not name:
        raise argparse.ArgumentTypeError("--var %r: empty NAME" % s)
    try:
        start, lo, hi = (float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "invalid --var %r (START/LO/HI must be numbers)" % s)
    if not lo < hi:
        raise argparse.ArgumentTypeError(
            "--var %r: LO must be < HI (got %g >= %g)" % (s, lo, hi))
    if not lo <= start <= hi:
        raise argparse.ArgumentTypeError(
            "--var %r: START %g outside [LO, HI] = [%g, %g]"
            % (s, start, lo, hi))
    return {"name": name, "start": start, "lo": lo, "hi": hi}


def parse_operand_spec(s):
    """'OPERAND[@DETECTOR]:TARGET:WEIGHT' -> {"operand","detector",
    "target","weight"}. argparse type= for --operand (optimize stage).
    OPERAND is one of OPTIMIZE_OPERANDS or a raw flattened merit key
    (contains a '.'). DETECTOR (optional) restricts the operand to one
    detector label; default: every detector in the report (summed for
    detected_power, averaged otherwise)."""
    parts = s.rsplit(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "invalid --operand %r (expected OPERAND[@DETECTOR]:TARGET:"
            "WEIGHT)" % s)
    head = parts[0].strip()
    if "@" in head:
        operand, detector = head.split("@", 1)
        operand, detector = operand.strip(), detector.strip()
    else:
        operand, detector = head, None
    if not operand:
        raise argparse.ArgumentTypeError("--operand %r: empty OPERAND" % s)
    if operand not in OPTIMIZE_OPERANDS and "." not in operand:
        raise argparse.ArgumentTypeError(
            "unknown operand %r (know: %s, or a raw flattened merit key "
            "containing a '.')" % (operand, ", ".join(OPTIMIZE_OPERANDS)))
    try:
        target, weight = float(parts[1]), float(parts[2])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "invalid --operand %r (TARGET/WEIGHT must be numbers)" % s)
    return {"operand": operand, "detector": detector,
            "target": target, "weight": weight}


def _build_optimize_parser():
    p = argparse.ArgumentParser(
        prog="optimize.py",
        description="Merit-function optimizer: drives design variables "
                    "(spreadsheet cell aliases) through the fast_eval "
                    "incoherent evaluator to minimize a weighted operand "
                    "merit, then re-evaluates the best design once with "
                    "coherence as authored.")
    p.add_argument("--model", required=True, metavar="FCSTD",
                   help="the base .FCStd (bare names resolve against the "
                        "project root, then basemodels/)")
    p.add_argument("--config", default=None, metavar="JSON",
                   help="JSON file whose keys mirror these options "
                        "(argparse dests); explicit CLI flags win over "
                        "config-file values")
    p.add_argument("--var", action="append", default=[],
                   type=parse_var_spec, metavar="NAME:START:LO:HI",
                   help="optimization variable (repeatable, >=1 required): "
                        "spreadsheet cell alias, start value and bounds "
                        "in the sheet's units (mm)")
    p.add_argument("--operand", action="append", default=[],
                   type=parse_operand_spec,
                   metavar="OPERAND[@DETECTOR]:TARGET:WEIGHT",
                   help="merit operand (repeatable, >=1 required). "
                        "Operands: %s (spot_rms/focus = detector spot RMS "
                        "radius um, needs --export-rays [added "
                        "automatically]; encircled_energy = ee_r80_um and "
                        "mtf50 = mtf50_tan_cy_mm, both from the coherent "
                        "field analysis [--save-fields + coherent inner "
                        "loop, slow]; detected_power = total_power_W, "
                        "maximized), or a raw flattened merit key. "
                        "Minimize operands contribute weight*(v-target)^2; "
                        "maximize operands contribute -weight*v (or "
                        "weight*(v-target)^2 when TARGET is nonzero)"
                        % ", ".join(OPTIMIZE_OPERANDS))
    p.add_argument("--algorithm", default="local",
                   choices=["local", "global"],
                   help="'local' = scipy Nelder-Mead within the bounds "
                        "(default); 'global' = nevergrad CMA-ES")
    p.add_argument("--budget", type=int, default=40, metavar="N",
                   help="maximum merit evaluations (default 40)")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="local-algorithm merit convergence tolerance "
                        "(scipy fatol; default 1e-3)")
    p.add_argument("--optimizer-seed", type=int, default=42,
                   help="RNG seed for the global (CMA-ES) algorithm")
    p.add_argument("--preset", default="quick",
                   choices=sorted(common.PRESETS),
                   help="fast_eval fidelity preset for rays/resolution/"
                        "nlambda not given explicitly (default: quick)")
    p.add_argument("--rays", type=float, default=None,
                   help="primary rays per source per evaluation "
                        "(default: from --preset)")
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--nlambda", type=int, default=None)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--eval-backend", default="worker",
                   choices=["worker", "full"],
                   help="fast_eval backend for the inner loop (default: "
                        "worker = persistent FreeCAD, fast)")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="optimizer case dir for report.json/progress.json "
                        "(default: var/optimize/<model-stem>)")
    p.add_argument("--workdir", default=None, metavar="DIR",
                   help="fast_eval evaluation workspace (default: "
                        "var/fasteval/<stem>-<backend>)")
    p.add_argument("--no-final-coherent", action="store_true",
                   help="skip the final keep_coherent=True re-evaluation "
                        "of the best design (the inner loop always runs "
                        "incoherent unless an operand needs the coherent "
                        "field analysis)")
    return p


# ---------------------------------------------------------------------------
# tolerance  (scripts/tolerance.py — sensitivity + Monte-Carlo tolerancing)
# ---------------------------------------------------------------------------
# Perturbation distributions scripts/tolerance.py understands (single
# source of truth: the GUI's TolerancePane populates its distribution
# combo from this and tolerance.py samples over it).
TOLERANCE_DISTS = ("normal", "uniform")


def parse_tolerance_spec(s):
    """'NAME:NOMINAL:DIST:BAND' -> {"name","nominal","dist","band"}.
    argparse type= for --tolerance (tolerance stage). NAME is a
    spreadsheet cell alias (exactly what permute_model --var / fast_eval
    address); DIST is one of TOLERANCE_DISTS; BAND > 0 is the 1-sigma
    width for 'normal' and the half-width for 'uniform', in the sheet's
    units (mm)."""
    parts = s.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "invalid --tolerance %r (expected NAME:NOMINAL:DIST:BAND)" % s)
    name = parts[0].strip()
    if not name:
        raise argparse.ArgumentTypeError("--tolerance %r: empty NAME" % s)
    dist = parts[2].strip().lower()
    if dist not in TOLERANCE_DISTS:
        raise argparse.ArgumentTypeError(
            "--tolerance %r: unknown distribution %r (know: %s)"
            % (s, dist, ", ".join(TOLERANCE_DISTS)))
    try:
        nominal, band = float(parts[1]), float(parts[3])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "invalid --tolerance %r (NOMINAL/BAND must be numbers)" % s)
    if not band > 0.0:
        raise argparse.ArgumentTypeError(
            "--tolerance %r: BAND must be > 0 (got %g)" % (s, band))
    return {"name": name, "nominal": nominal, "dist": dist, "band": band}


def parse_compensator_spec(s):
    """'VAR:LO:HI' (or 'VAR:START:LO:HI') -> {"name","start","lo","hi"}.
    argparse type= for --compensator (tolerance stage). START defaults to
    the midpoint of [LO, HI]; the compensator sits at START whenever it is
    not being optimized (the nominal / sensitivity evaluations)."""
    parts = s.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "invalid --compensator %r (expected VAR:LO:HI or "
            "VAR:START:LO:HI)" % s)
    name = parts[0].strip()
    if not name:
        raise argparse.ArgumentTypeError("--compensator %r: empty VAR" % s)
    try:
        nums = [float(p) for p in parts[1:]]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "invalid --compensator %r (bounds must be numbers)" % s)
    if len(nums) == 2:
        lo, hi = nums
        start = 0.5 * (lo + hi)
    else:
        start, lo, hi = nums
    if not lo < hi:
        raise argparse.ArgumentTypeError(
            "--compensator %r: LO must be < HI (got %g >= %g)"
            % (s, lo, hi))
    if not lo <= start <= hi:
        raise argparse.ArgumentTypeError(
            "--compensator %r: START %g outside [LO, HI] = [%g, %g]"
            % (s, start, lo, hi))
    return {"name": name, "start": start, "lo": lo, "hi": hi}


def _build_tolerance_parser():
    p = argparse.ArgumentParser(
        prog="tolerance.py",
        description="Sensitivity analysis + Monte-Carlo yield tolerancing "
                    "over the fast_eval evaluator: finite-difference each "
                    "tolerance's merit impact (ranked table), then draw N "
                    "random perturbation sets and report the merit "
                    "distribution / yield fraction, optionally recovering "
                    "each draw with a nested focus-compensator "
                    "optimization.")
    p.add_argument("--model", required=True, metavar="FCSTD",
                   help="the base .FCStd (bare names resolve against the "
                        "project root, then basemodels/)")
    p.add_argument("--config", default=None, metavar="JSON",
                   help="JSON file whose keys mirror these options "
                        "(argparse dests); explicit CLI flags win over "
                        "config-file values")
    p.add_argument("--tolerance", action="append", default=[],
                   type=parse_tolerance_spec,
                   metavar="NAME:NOMINAL:DIST:BAND",
                   help="tolerance parameter (repeatable, >=1 required): "
                        "spreadsheet cell alias, nominal value, "
                        "distribution (%s), and band (1-sigma for normal, "
                        "half-width for uniform) in the sheet's units (mm)"
                        % "/".join(TOLERANCE_DISTS))
    p.add_argument("--operand", action="append", default=[],
                   type=parse_operand_spec,
                   metavar="OPERAND[@DETECTOR]:TARGET:WEIGHT",
                   help="merit operand (repeatable, >=1 required); same "
                        "grammar and semantics as optimize.py: %s, or a "
                        "raw flattened report.json merit key"
                        % ", ".join(OPTIMIZE_OPERANDS))
    p.add_argument("--draws", type=int, default=50, metavar="N",
                   help="Monte-Carlo perturbation draws (default 50; "
                        "0 = sensitivity analysis only)")
    p.add_argument("--merit-threshold", type=float, default=None,
                   metavar="X",
                   help="a draw PASSES when its merit <= X; the yield "
                        "fraction is passes/draws (omit for distribution "
                        "stats only)")
    p.add_argument("--compensator", default=None, type=parse_compensator_spec,
                   metavar="VAR:LO:HI",
                   help="focus compensator: before recording each draw's "
                        "merit, optimize VAR within [LO, HI] (nested "
                        "optimize.py local engine) to recover the best "
                        "merit; 'VAR:START:LO:HI' also fixes the "
                        "uncompensated resting value (default: midpoint)")
    p.add_argument("--comp-budget", type=int, default=10, metavar="N",
                   help="merit evaluations per draw for the nested "
                        "compensator optimization (default 10)")
    p.add_argument("--sens-delta", type=float, default=1.0, metavar="FRAC",
                   help="sensitivity finite-difference step as a fraction "
                        "of each tolerance's band (default 1.0 = probe at "
                        "the band edges)")
    p.add_argument("--skip-sensitivity", action="store_true",
                   help="skip the per-parameter finite-difference "
                        "sensitivity table (Monte-Carlo only)")
    p.add_argument("--hist-bins", type=int, default=20, metavar="N",
                   help="merit-histogram bin count in the report "
                        "(default 20)")
    p.add_argument("--mc-seed", type=int, default=42,
                   help="RNG seed for the Monte-Carlo perturbation draws")
    p.add_argument("--preset", default="quick",
                   choices=sorted(common.PRESETS),
                   help="fast_eval fidelity preset for rays/resolution/"
                        "nlambda not given explicitly (default: quick)")
    p.add_argument("--rays", type=float, default=None,
                   help="primary rays per source per evaluation "
                        "(default: from --preset)")
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--nlambda", type=int, default=None)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--eval-backend", default="worker",
                   choices=["worker", "full"],
                   help="fast_eval backend (default: worker = persistent "
                        "FreeCAD, fast)")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="tolerance case dir for report.json/progress.json "
                        "(default: var/tolerance/<model-stem>)")
    p.add_argument("--workdir", default=None, metavar="DIR",
                   help="fast_eval evaluation workspace (default: "
                        "var/fasteval/<stem>-<backend>)")
    return p


_BUILDERS = {
    "pipeline": _build_pipeline_parser,
    "trace": _build_trace_parser,
    "post": _build_post_parser,
    "viz": _build_viz_parser,
    "optimize": _build_optimize_parser,
    "tolerance": _build_tolerance_parser,
}


def build_parser(stage):
    """Return a fully-configured argparse.ArgumentParser for `stage`
    (one of cli_specs.STAGES), identical in behavior to what the
    corresponding stage script built before this module existed."""
    try:
        builder = _BUILDERS[stage]
    except KeyError:
        raise ValueError("unknown stage %r (must be one of %s)"
                         % (stage, ", ".join(STAGES)))
    return builder()


# ---------------------------------------------------------------------------
# Self-check:  python3 scripts/cli_specs.py
# ---------------------------------------------------------------------------
def _selfcheck():
    ok = True
    for stage in STAGES:
        try:
            p = build_parser(stage)
            print("  [ok] build_parser(%r) -> %s" % (stage, type(p).__name__))
        except Exception as exc:
            print("  [FAIL] build_parser(%r): %s" % (stage, exc))
            ok = False
    print("SELF-CHECK", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selfcheck())
