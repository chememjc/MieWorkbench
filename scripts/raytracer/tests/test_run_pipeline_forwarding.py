# =============================================================================
# test_run_pipeline_forwarding.py — run_pipeline.py's stage-command builders
# forward the flags cli_specs.py declares for them (design-usability round):
#   --views/--smoke (viz stage), --viz-generations (post stage),
#   --save-fields-detectors (trace stage, alongside --save-fields).
# Pure unit tests against trace_cmd()/post_cmd()/viz_cmd() — no subprocess,
# no FreeCAD/ParaView/optics-heavy imports (run_pipeline.py + cli_specs.py
# are stdlib-only by contract, so this runs fine under any interpreter, but
# lives under the engine test tree per repo convention).
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_run_pipeline_forwarding.py -q
# =============================================================================
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import cli_specs                                          # noqa: E402
import run_pipeline                                        # noqa: E402


def _parse(argv):
    return cli_specs.build_parser("pipeline").parse_args(
        ["--models", "x.FCStd"] + argv)


CASE_DIR = Path("/tmp/miewb-forwarding-test-case")


# ---------------------------------------------------------------------------
# --save-fields-detectors (trace stage)
# ---------------------------------------------------------------------------
def test_trace_cmd_forwards_save_fields_detectors():
    args = _parse(["--save-fields", "--save-fields-detectors", "DetA,DetB"])
    cmd = run_pipeline.trace_cmd("x", CASE_DIR, args)
    assert "--save-fields" in cmd
    i = cmd.index("--save-fields-detectors")
    assert cmd[i + 1] == "DetA,DetB"


def test_trace_cmd_omits_save_fields_detectors_by_default():
    args = _parse([])
    cmd = run_pipeline.trace_cmd("x", CASE_DIR, args)
    assert "--save-fields" not in cmd
    assert "--save-fields-detectors" not in cmd


def test_trace_cmd_save_fields_detectors_without_save_fields_still_forwards():
    # cli-level: the flag is inert without --save-fields (run_trace.py's
    # own gate handles that), but forwarding itself must not silently drop
    # it just because --save-fields was omitted.
    args = _parse(["--save-fields-detectors", "DetA"])
    cmd = run_pipeline.trace_cmd("x", CASE_DIR, args)
    assert "--save-fields" not in cmd
    i = cmd.index("--save-fields-detectors")
    assert cmd[i + 1] == "DetA"


# ---------------------------------------------------------------------------
# --viz-generations (post stage)
# ---------------------------------------------------------------------------
def test_post_cmd_forwards_viz_generations():
    args = _parse(["--viz-generations", "3"])
    cmd = run_pipeline.post_cmd("x", CASE_DIR, args)
    i = cmd.index("--viz-generations")
    assert cmd[i + 1] == "3"


def test_post_cmd_omits_viz_generations_by_default():
    args = _parse([])
    cmd = run_pipeline.post_cmd("x", CASE_DIR, args)
    assert "--viz-generations" not in cmd


def test_post_cmd_forwards_viz_generations_zero():
    # 0 is falsy but a legitimate value (only generation 0 segments) —
    # forwarding must key off "is not None", not truthiness.
    args = _parse(["--viz-generations", "0"])
    cmd = run_pipeline.post_cmd("x", CASE_DIR, args)
    i = cmd.index("--viz-generations")
    assert cmd[i + 1] == "0"


# ---------------------------------------------------------------------------
# --imaging-products / --wavefront-pupil (post stage, pulsed-optics P9)
# ---------------------------------------------------------------------------
def test_post_cmd_forwards_imaging_products_and_wavefront_pupil():
    args = _parse(["--export-rays",
                   "--imaging-products", "distortion,telecentricity",
                   "--wavefront-pupil", "exit_pupil"])
    cmd = run_pipeline.post_cmd("x", CASE_DIR, args)
    i = cmd.index("--imaging-products")
    assert cmd[i + 1] == "distortion,telecentricity"
    i = cmd.index("--wavefront-pupil")
    assert cmd[i + 1] == "exit_pupil"


def test_post_cmd_omits_imaging_flags_by_default():
    args = _parse([])
    cmd = run_pipeline.post_cmd("x", CASE_DIR, args)
    assert "--imaging-products" not in cmd
    assert "--wavefront-pupil" not in cmd


def test_imaging_products_all_expands_to_every_product():
    args = _parse(["--export-rays", "--imaging-products", "all"])
    cmd = run_pipeline.post_cmd("x", CASE_DIR, args)
    i = cmd.index("--imaging-products")
    assert cmd[i + 1] == "distortion,vignetting,field_curves,telecentricity"


def test_imaging_products_rejects_unknown_name():
    with pytest.raises(SystemExit):
        _parse(["--export-rays", "--imaging-products", "distortion,bogus"])


def test_pipeline_gates_imaging_products_without_export_rays(monkeypatch):
    argv = ["--models", "x.FCStd", "--imaging-products", "distortion",
            "--print-only"]
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py"] + argv)
    with pytest.raises(SystemExit) as exc:
        run_pipeline.main()
    assert "--export-rays" in str(exc.value)


def test_pipeline_allows_imaging_products_with_ghost_analysis(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    # --ghost-analysis implies --export-rays at the trace stage, so the
    # gate must accept it too
    model = tmp_path / "dummy.FCStd"
    model.write_bytes(b"")
    argv = ["--models", str(model), "--steps", "post",
            "--ghost-analysis", "--imaging-products", "vignetting",
            "--print-only"]
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py"] + argv)
    assert run_pipeline.main() == 0
    assert "--imaging-products vignetting" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --views / --smoke (viz stage)
# ---------------------------------------------------------------------------
def test_viz_cmd_forwards_views_and_smoke():
    args = _parse(["--views", "top,side", "--smoke"])
    cmd = run_pipeline.viz_cmd("x", CASE_DIR, args)
    i = cmd.index("--views")
    assert cmd[i + 1] == "top,side"
    assert "--smoke" in cmd


def test_viz_cmd_omits_views_and_smoke_by_default():
    args = _parse([])
    cmd = run_pipeline.viz_cmd("x", CASE_DIR, args)
    assert "--views" not in cmd
    assert "--smoke" not in cmd


def test_viz_cmd_still_forwards_dim_rays_alongside_new_flags():
    args = _parse(["--views", "top", "--dim-rays", "linear"])
    cmd = run_pipeline.viz_cmd("x", CASE_DIR, args)
    assert "--dim-rays" in cmd
    assert "--views" in cmd


# ---------------------------------------------------------------------------
# --print-only end-to-end sanity (composes without executing/raising)
# ---------------------------------------------------------------------------
def test_print_only_composes_new_flags_without_error(tmp_path, capsys,
                                                      monkeypatch):
    model = tmp_path / "dummy.FCStd"
    model.write_bytes(b"")
    argv = ["--models", str(model), "--steps", "trace,post,viz",
            "--views", "top,side", "--smoke", "--viz-generations", "2",
            "--save-fields", "--save-fields-detectors", "DetA,DetB",
            "--print-only"]
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py"] + argv)
    ret = run_pipeline.main()
    assert ret == 0
    out = capsys.readouterr().out
    assert "--views top,side" in out
    assert "--smoke" in out
    assert "--viz-generations 2" in out
    assert "--save-fields-detectors DetA,DetB" in out


# ---------------------------------------------------------------------------
# Task 4: --detector-face help text points at the detector_face BODY
# property as the preferred, C-engine-routable alternative.
# ---------------------------------------------------------------------------
def test_detector_face_help_prefers_body_property():
    for stage in ("pipeline", "trace"):
        p = cli_specs.build_parser(stage)
        action = p._option_string_actions["--detector-face"]
        helptext = action.help or ""
        assert "detector_face" in helptext
        assert "BODY property" in helptext or "body property" in helptext
        assert "C-engine" in helptext or "C engine" in helptext
