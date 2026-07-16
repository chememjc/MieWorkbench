"""checkpointinfo — Qt-free introspection of the P1 chunked-run contract
(scripts/raytracer/cengine.py's <case_dir>/cengine/checkpoint.json) for a
results case directory.

Pure-Python (no Qt), mirroring core/train.py's split: this module is the
reusable, plain-pytest-testable core; the Results pane (panes/results.py)
and RunController (core/runner.py) are the Qt/subprocess-facing consumers.

Case-dir layout recap (CLAUDE.md "C engine" + docs/RAYTRACER.md §13):
  <case_dir>/case.json               — status: estimated/completed/failed
  <case_dir>/cengine/checkpoint.json — C-engine chunked-run state only:
      status: tracing (mid-trace or a dead/interrupted run) ->
              trace_complete (all chunks done, gather pending) ->
              completed (case.json also 'completed'); target_rays; chunks
              (list of {seed,lo,hi,...}); extensions (list of {from,to}).
A checkpoint.json only exists for a case that ran (at least once) on the C
engine under the chunked-run contract; Python-engine cases never have one.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import common     # noqa: E402  (stdlib-only shared contract hub)


def checkpoint_path(case_dir):
    return Path(case_dir) / "cengine" / "checkpoint.json"


def read_checkpoint(case_dir):
    """checkpoint.json as a dict, or None (never ran chunked / unreadable)."""
    path = checkpoint_path(case_dir)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_case(case_dir):
    """case.json as a dict, or None (missing / unreadable)."""
    try:
        return json.loads((Path(case_dir) / "case.json").read_text())
    except (OSError, ValueError):
        return None


def is_live(case_dir):
    """True when a fresh (non-stale) lock is held by a running process --
    the exact 'live' test mainwindow.py uses to decide monitor mode."""
    info = common.lock_info(case_dir)
    return info is not None and not common.lock_is_stale(case_dir, info)


def resume_state(case_dir):
    """None, or a dict describing a resumable DEAD run: checkpoint status
    is 'tracing' (interrupted mid-trace: a hard kill, or the GUI/host
    closing under it) and the case is not currently live (lock free/stale
    -- a live 'tracing' case is a run in progress, not a dead one)."""
    ckpt = read_checkpoint(case_dir)
    if ckpt is None or ckpt.get("status") != "tracing":
        return None
    if is_live(case_dir):
        return None
    return {
        "target_rays": int(ckpt.get("target_rays", 0)),
        "n_chunks": len(ckpt.get("chunks", [])),
    }


def extend_state(case_dir):
    """None, or a dict describing an extendable COMPLETED C-engine case:
    {'current_rays', 'spr', 'm_eff_proxy'}. spr ("surviving coherent
    samples per primary ray") is the case's own MEASURED value (from its
    gather diagnostics in case.json['gather'], the same ratio
    raytracer.cengine.run_c_case's calibration writeback computes) --
    not the machine-global calibrated/fallback estimate() uses, since we
    have this case's actual number. m_eff_proxy = spr * rays is the
    effective-sample-count proxy (docs/RAYTRACER.md §6.3: speckle pedestal
    ~ 1/M_eff)."""
    ckpt = read_checkpoint(case_dir)
    if ckpt is None or ckpt.get("status") != "completed":
        return None
    case = read_case(case_dir)
    if case is None or case.get("status") != "completed":
        return None
    current_rays = int(ckpt.get("target_rays", 0))
    spr = measured_spr(case, current_rays)
    return {
        "current_rays": current_rays,
        "spr": spr,
        "m_eff_proxy": spr * current_rays,
    }


def measured_spr(case, current_rays):
    """Actual surviving-samples-per-primary-ray from a completed case's
    gather diagnostics (case['gather'], one block per seed, each a
    {detector_label: {'src/lam/pol': {'n_samples': ..., ...}}} dict --
    see raytracer.cengine.run_c_case / raytracer.gather.render_coherent).
    Falls back to common.DEFAULT_SPR when the scene has no coherent
    source (n_samples all zero / section absent) -- matches
    common.estimate()'s own fallback."""
    seeds_gather = (case or {}).get("gather") or {}
    total_samples = 0
    n_seeds = 0
    for seed_diag in seeds_gather.values():
        n_seeds += 1
        for det_keys in (seed_diag or {}).values():
            for entry in (det_keys or {}).values():
                total_samples += int(entry.get("n_samples", 0))
    total_rays = max(n_seeds, 1) * max(int(current_rays), 1)
    if total_samples <= 0:
        return common.DEFAULT_SPR
    return total_samples / total_rays


def resolve_preset_tag(case_dir):
    """Best-effort inverse of common.case_name(preset, tag): the case dir's
    basename is '<preset>' or '<preset>-<tag>'. Falls back to
    ('quick', None) for a case name that doesn't start with a known preset
    (should not happen for a GUI-launched case) -- physics-relevant values
    (rays/resolution/nlambda/...) are still pinned explicitly by the
    caller, so a wrong preset guess only risks recomputing unset defaults,
    never silently mismatched physics."""
    name = Path(case_dir).name
    for preset in sorted(common.PRESETS, key=len, reverse=True):
        if name == preset:
            return preset, None
        if name.startswith(preset + "-"):
            return preset, name[len(preset) + 1:]
    return "quick", None


_CARRIED_OPTION_KEYS = ("resolution", "nlambda", "spectral_bins", "seeds",
                        "backend")


def build_resume_config(case_dir):
    """ConfigMatrix-shaped {dest: value} for RunController.build_args(),
    reissuing THIS case's own trace options (from case.json) at the exact
    target_rays its checkpoint expects (run_c_case hard-refuses a --resume
    whose --rays doesn't match the checkpoint's target_rays exactly)."""
    case_dir = Path(case_dir)
    ckpt = read_checkpoint(case_dir) or {}
    case = read_case(case_dir) or {}
    opts = case.get("options") or {}
    preset, tag = resolve_preset_tag(case_dir)
    config = {"resume": True, "engine": "c", "preset": preset,
             "rays": float(ckpt.get("target_rays")
                           or opts.get("rays") or 0.0)}
    if tag:
        config["tag"] = tag
    for key in _CARRIED_OPTION_KEYS:
        if opts.get(key) is not None:
            config[key] = opts[key]
    return config


def build_extend_config(case_dir, new_rays):
    """Like build_resume_config, but for a COMPLETED case's --extend N
    (additive continuation past its current target_rays)."""
    config = build_resume_config(case_dir)
    del config["resume"]
    config["extend"] = float(new_rays)
    return config
