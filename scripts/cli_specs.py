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
    g.add_argument("--strict-analytic", action="store_true",
                   help="hard-error on mesh-type faces (v1 behavior)")
    g.add_argument("--optical-properties", default=None,
                   help="override the opticalproperties/ library root")
    g.add_argument("--source-face", action="append", default=[],
                   metavar="Body.Feature.FaceN")
    g.add_argument("--detector-face", action="append", default=[],
                   metavar="Body.Feature.FaceN")
    g.add_argument("--grating", action="append", default=[], metavar="SPEC")
    g.add_argument("--rough", action="append", default=[], metavar="SPEC")
    g.add_argument("--particles", default=None, metavar="SPEC")
    g.add_argument("--particle-threshold", type=float, default=None)
    g.add_argument("--suppress-body", action="append", default=[],
                   metavar="BODY")

    g = p.add_argument_group("execution / orchestration")
    g.add_argument("--keep-going", action="store_true",
                   help="on a stage failure, print FAILED and continue "
                        "with the next model instead of aborting (exit "
                        "code still nonzero)")
    g.add_argument("--print-only", action="store_true",
                   help="compose and print every stage command without "
                        "running anything")
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
    p.add_argument("--detector-face", action="append", default=[])
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
