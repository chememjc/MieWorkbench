"""P1 NUFFT angular-spectrum gather route: per-key gate + routing (C engine).

The NUFFT route (cengine/src/gather_nufft.c) is an OPT-IN, gated third gather
path. It is OFF by default and, when opted in with --gather-nufft, a per-key
runtime gate decides whether to use it — logged into gather.json as
`gather_mode` plus a `nufft_gate` object with the decision + reasons.

IMPORTANT (documented limitation): the route is band-limited, but Monte-Carlo
Huygens samples are ideal POINT emitters with a white spatial spectrum, so
the angular-spectrum band truncation cannot reproduce the exact per-pair
kernel to NUFFT tolerance (an irreducible few-% Gibbs floor — measured, see
the C ctest `nufft_route`). It is therefore NOT enabled for production and
the gate rejects every wide-angle/near-field real scene. These tests pin:

  1. default: the route is off (gather_mode == "tiled", nufft_gate disabled);
  2. --gather-nufft: the gate runs, logs its reasons, and REJECTS the
     wide-angle diffraction scenes (doubleslit, michelson_folded) — routing
     to the tiled kernel — so no scene silently takes the inaccurate route;
  3. the gate object carries the separating-plane / obliquity / VRAM / cost
     sub-decisions.

The route's own execution + accuracy floor is pinned by the C unit test
`cengine/build/tests/test_nufft_route` (ctest), which runs the full
type-1 -> propagator -> type-2 GPU path on a controlled collimated case.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
BINARY = REPO / "cengine" / "build" / "miewb-trace"
DS = REPO / "geometry" / "doubleslit" / "model.json"
MICH = REPO / "geometry" / "michelson_folded" / "model.json"

pytestmark = pytest.mark.skipif(
    not (DS.exists() and BINARY.exists()),
    reason="needs the doubleslit geometry cache and a built C engine")


def _run(model, case_dir, extra, rays=40000, res=256, nlam=3):
    cmd = [sys.executable, str(REPO / "scripts" / "run_trace.py"),
           "--model-json", str(model), "--case-dir", str(case_dir),
           "--rays", str(rays), "--nlambda", str(nlam),
           "--resolution", str(res), "--engine", "c"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    # P1 chunked-run layout: the final gather runs in a gather_only stage
    # under cengine/seed<k>/gather/ (was cengine/seed<k>/ pre-chunking)
    gj = json.loads(
        next(case_dir.glob("cengine/seed*/gather/gather.json")).read_text())
    return [e for det in gj.values() for e in det.values()]


def test_default_route_is_off(tmp_path):
    """No --gather-nufft: the route is off; nufft_gate logged as disabled."""
    keys = _run(DS, tmp_path / "def", [])
    assert keys
    for e in keys:
        assert e["gather_mode"] == "tiled"
        assert "nufft_gate" in e
        assert e["nufft_gate"]["enabled"] is False


def test_optin_gate_rejects_doubleslit(tmp_path):
    """--gather-nufft on the wide-angle doubleslit: the gate WANTS the route
    but rejects it on obliquity separability (the diffraction NA is intrinsic
    and violates the < 1e-3 fold bound) -> falls back to the tiled kernel,
    with every sub-decision logged.

    cuFINUFFT is an OPTIONAL build dependency: on a stub binary
    (nufft_gate.available == false) the linked-only assertions are skipped,
    after pinning the stub contract — the flag must NOT enable the route."""
    keys = _run(DS, tmp_path / "opt", ["--gather-nufft"])
    assert keys
    if not keys[0]["nufft_gate"].get("available", False):
        for e in keys:
            g = e["nufft_gate"]
            assert g["enabled"] is False and g["chosen"] is False
            assert e["gather_mode"] == "tiled"
        pytest.skip("binary built without cuFINUFFT — stub contract pinned")
    for e in keys:
        g = e["nufft_gate"]
        assert g["enabled"] is True          # opted in
        assert g["chosen"] is False          # but gate rejected it
        assert e["gather_mode"] == "tiled"   # so it ran tiled
        # the separating plane exists (a detector always has one)...
        assert g["separating"] is True
        assert g["reasons"]["sep"] is True
        # ...but the obliquity fold bound is violated (wide diffraction)
        assert g["reasons"]["obliq"] is False
        assert g["obliquity_var"] >= 1e-3
        # gate object carries the full sub-decision set
        for r in ("want", "sep", "obliq", "vram", "cost"):
            assert r in g["reasons"]


@pytest.mark.skipif(not MICH.exists(),
                    reason="needs the michelson_folded geometry cache")
def test_optin_gate_michelson_folded(tmp_path):
    """michelson_folded at 2e4 rays / 512^2 / 3 lambda: the folded-arm
    detector is near-field/wide-angle here, so the gate also rejects it
    (obliquity/cost) and routes tiled — the gate machinery runs and logs
    a decision either way (never silently takes the inaccurate route)."""
    keys = _run(MICH, tmp_path / "mich", ["--gather-nufft"],
                rays=20000, res=512, nlam=3)
    assert keys
    if not keys[0]["nufft_gate"].get("available", False):
        for e in keys:
            assert e["nufft_gate"]["chosen"] is False
            assert e["gather_mode"] == "tiled"
        pytest.skip("binary built without cuFINUFFT — stub contract pinned")
    for e in keys:
        g = e["nufft_gate"]
        assert g["enabled"] is True
        # decision is logged and consistent with the mode actually run
        assert (e["gather_mode"] == "nufft") == (g["chosen"] is True)
        assert isinstance(g["obliquity_var"], (int, float))
        assert isinstance(g["k_grid_N"], int)
