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

STAGES = ("pipeline", "trace", "post", "viz")


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


_BUILDERS = {
    "pipeline": _build_pipeline_parser,
    "trace": _build_trace_parser,
    "post": _build_post_parser,
    "viz": _build_viz_parser,
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
