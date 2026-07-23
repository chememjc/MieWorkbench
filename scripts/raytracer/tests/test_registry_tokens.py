# =============================================================================
# test_registry_tokens.py — the P3 interaction-registry token contract
# (REGISTRY.md §4). Pins the C binary's --tokens dump against the Python
# routing authority (cengine.PORTED + detect_features' emission surface) so
# the two can never drift, and proves the construction-time hard error.
#
#   1. Token parity: the dump ⊇ cengine.PORTED (no silent GAP — every ported
#      feature has a C token), and every STATIC token detect_features() can
#      emit is classified in exactly one of {C registry, Python-only} (no
#      ORPHAN routed to C without an implementation — the failure mode the
#      whole registry exists to kill, cengine.py's P8 NLO incident note).
#   2. Hard error: a request carrying a fabricated feature token exits 2
#      naming it (REGISTRY.md §2.2).
#
# Skipped whole when the miewb-trace binary is not built.
# =============================================================================
import json
import re
import subprocess
import sys
import inspect
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))                       # scenehelpers, scenes
_SCRIPTS = _HERE.parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))                    # raytracer, run_trace

from raytracer import cengine                             # noqa: E402
import cengine_scenes                                     # noqa: E402

pytestmark = pytest.mark.skipif(
    cengine.binary_path() is None,
    reason="miewb-trace not built (cd cengine && ./build.sh)")


# --------------------------------------------------------------------------
# The Python-only half of detect_features' emission surface: every STATIC
# token the function can emit that is NOT ported to C. Curated here and kept
# honest by test_static_emission_surface_is_partitioned below, which parses
# detect_features' source and asserts every literal token is in exactly one
# of {PORTED, PYTHON_ONLY}. (The dynamic "surface:%s" token is format-string
# and open-ended; a specific surface routes to C only if its exact token is
# in PORTED — same generic path, no per-surface bookkeeping.)
# --------------------------------------------------------------------------
PYTHON_ONLY = frozenset({
    "beam", "apodization",              # source models (sources.py)
    "biaxial",                          # biaxial crystals (uniaxial is ported)
    "figure_error",                     # P8 Zernike surface figure error
                                        #   (PerturbedSurface, surfaces.py)
    "berreman",                         # P9 full-anisotropy 4x4 (biaxial exact
                                        #   / gyrotropic / absorbing-aniso);
                                        #   C-registry seam STUB, dump skips it
                                        #   -> classified Python-only here.
    "gyration",                         # scene-level natural optical activity
                                        #   (fixround: near-axis polarization
                                        #   rotation in gyrotropic uniaxials;
                                        #   no C rotation term — reference-
                                        #   routes, pinned by
                                        #   test_routing_reasons)
    "nonlinear",                        # P8 NLO: chi2 SHG child + Pockels (SHG
                                        #   harmonic strata unported; Pockels
                                        #   rides the biref tables — split is a
                                        #   later tranche). saturable/tpa/kerr
                                        #   are ported (P7 tranche 2).
    "temperature",                      # thermo-optic dn/dT
    "rough_fresnel_macro",              # legacy nominal-angle rough model
    "scatter_g_ne_2", "scatter_btdf", "scatter_importance",
    "grating_table_v2",                 # P6 RCWA v2 complex-amplitude tables
    "coating_phase",                    # phase-carrying table coating (P2)
    "extra_detector_faces", "curved_detector",
    "particles_explicit",               # explicit realization (continuum ported)
    "pol_transport",                    # P2 parallel-transport Q/J bookkeeping
    "time_directional_index",           # crystal e-ray n_g_eff (time+crystal)
    "conical",                          # samples-instruments: internal conical
                                        #   refraction fan (biaxial optic axis;
                                        #   Python physics, --conical)
    "sample_explicit",                  # samples-instruments: EXPLICIT/lattice
                                        #   sample realization (frozen spheres /
                                        #   paracrystal). The continuum-mode
                                        #   `sample_body` token is now PORTED (C
                                        #   medium-stack-gated particle medium);
                                        #   explicit rows still Python-route.
    # NOTE: "image_source" (extended image-emitting source) is now PORTED —
    # the C sampler (trace.c sample_image_pos_dir) alias-draws pixels + emits
    # Lambertian/cone; it lives in cengine.PORTED + the C registry, not here.
})


def _dump_tokens():
    out = subprocess.run(
        [str(cengine.binary_path()), "--tokens"],
        capture_output=True, text=True, check=True)
    toks = {}
    for line in out.stdout.splitlines():
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        toks[parts[0].strip()] = parts[1].strip() if len(parts) > 1 else ""
    return toks


def _static_emission_tokens():
    """Every literal token detect_features() can emit — the STATIC part of
    its emission surface. Parses feats.add("literal") occurrences from the
    live source (drift-proof: a new feature token trips the partition test
    below until it is routed)."""
    src = inspect.getsource(cengine.detect_features)
    lits = set(re.findall(r'feats\.add\(\s*["\']([^"\']+)["\']', src))
    # drop the format-string surface token; specific surface:* tokens are
    # covered by PORTED membership (surface:plane, ... surface:qforbes note)
    return {t for t in lits if "%s" not in t}


def test_dump_is_superset_of_ported():
    """No GAP: every feature in cengine.PORTED has a C-registry token, so a
    ported scene never hard-errors at load."""
    dump = set(_dump_tokens())
    missing = set(cengine.PORTED) - dump
    assert not missing, (
        "C --tokens is missing PORTED tokens (silent gap): %s" % sorted(missing))


def test_ported_and_python_only_disjoint():
    overlap = set(cengine.PORTED) & PYTHON_ONLY
    assert not overlap, "token classified both ported and python-only: %s" \
        % sorted(overlap)


def test_python_only_absent_from_c_registry():
    """No ORPHAN routing to C: a Python-only token must NOT appear in the C
    registry, or a scene using it would run on C and silently skip the
    physics — the exact P8 NLO failure the registry abolishes."""
    dump = set(_dump_tokens())
    leaked = dump & PYTHON_ONLY
    assert not leaked, (
        "Python-only tokens present in the C registry dump: %s" % sorted(leaked))


def test_static_emission_surface_is_partitioned():
    """Every STATIC token detect_features can emit is in exactly one of
    {C registry (--tokens), Python-only}. A newly added feature token that
    forgets to pick a side fails here."""
    dump = set(_dump_tokens())
    emitted = _static_emission_tokens()
    unclassified = sorted(t for t in emitted
                          if t not in dump and t not in PYTHON_ONLY)
    both = sorted(t for t in emitted if t in dump and t in PYTHON_ONLY)
    assert not unclassified, (
        "emitted feature tokens routed nowhere (orphans — add to the C "
        "registry or PYTHON_ONLY): %s" % unclassified)
    assert not both, "emitted tokens in BOTH sides: %s" % both


# --------------------------------------------------------------------------
# Hard-error contract (REGISTRY.md §2.2)
# --------------------------------------------------------------------------
def _make_c_request(tmp_path):
    """Run a small scene through the C engine so a real request.json exists,
    and return its path."""
    import run_trace
    model_json = cengine_scenes.write_scene("c_plate", tmp_path / "geometry")
    rc = run_trace.main([
        "--model-json", str(model_json),
        "--case-dir", str(tmp_path / "case"),
        "--rays", "2000", "--resolution", "64", "--nlambda", "1",
        "--spectral-bins", "4", "--engine", "c", "--workers", "1",
    ])
    assert rc == 0, "run_trace --engine c exited %s" % rc
    reqs = sorted((tmp_path / "case" / "cengine").rglob("request.json"))
    assert reqs, "no request.json produced under the C case dir"
    return reqs[0]


def test_request_carries_feature_tokens(tmp_path):
    """build_request emits the detected feature list (so the C engine can
    check it) — c_plate is a bare Fresnel plate + detector screen."""
    req = json.loads(_make_c_request(tmp_path).read_text())
    assert "features" in req, "request.json is missing the features[] field"
    feats = set(req["features"])
    assert "surface:plane" in feats
    # every carried token must be one the C registry accepts (this scene
    # routed to C, so by construction it must)
    dump = set(_dump_tokens())
    assert feats <= dump, "routed-to-C scene carries unknown tokens: %s" \
        % sorted(feats - dump)


def test_fabricated_token_exits_2_naming_it(tmp_path):
    """A request whose features[] carries a token with no C implementation
    is a hard error, exit 2, naming the token (never a silent skip)."""
    req_path = _make_c_request(tmp_path)
    req = json.loads(req_path.read_text())
    bogus = "bogus:fabricated_feature"
    req["features"] = list(req.get("features", [])) + [bogus]
    tampered = tmp_path / "tampered_request.json"
    tampered.write_text(json.dumps(req))
    p = subprocess.run(
        [str(cengine.binary_path()), "--config", str(tampered)],
        capture_output=True, text=True)
    assert p.returncode == 2, (
        "expected exit 2 (EXIT_INPUT), got %s\nstderr:\n%s"
        % (p.returncode, p.stderr))
    assert bogus in (p.stderr + p.stdout), (
        "hard error did not name the fabricated token:\n%s" % p.stderr)
