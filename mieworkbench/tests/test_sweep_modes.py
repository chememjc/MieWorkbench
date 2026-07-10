"""Phase C sweep-mode integration tests (marked 'freecad', env-gated):
drives permute_model.py through the real AppImage twice to prove

  (a) --sweep-mode zip produces exactly the variant filenames
      run_pipeline.variant_output_names() predicts for zip, with two vars
      of equal swept length, and

  (b) the KNOWN BUG fix in permute_model.rebuild_primitive_groups (an
      expression-linked dim_<El> cell like "=<<miewb_vars>>.gap * 1mm" now
      also triggers a primitive rebuild when miewb_vars itself is swept,
      not only when the primitive's own sheet is directly touched): the
      iris body's shape actually changes between two miewb_vars.gap
      variants.

Run: MIEWB_RUN_FREECAD=1 QT_QPA_PLATFORM=offscreen env/bin/python \
         -m pytest mieworkbench/tests/test_sweep_modes.py -q
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

from mieworkbench.core.project import Project  # noqa: E402

import run_pipeline  # noqa: E402

pytestmark = pytest.mark.freecad

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRIMITIVES = os.path.join(REPO, "primitives")
APPIMAGE = os.environ.get("MIEWB_FREECAD", "/home3/freecad/FreeCAD.AppImage")


def run_permute(model_path, outdir, varspecs, sweep_mode=None):
    """varspecs: list of (var, vmin, vmax, n)."""
    script = os.path.join(REPO, "scripts", "permute_model.py")
    cmd = [APPIMAGE, "-c", script, "--", "--model", str(model_path)]
    for var, vmin, vmax, n in varspecs:
        cmd += ["--var", var, "--min", str(vmin), "--max", str(vmax),
               "--n", str(n)]
    cmd += ["--outdir", str(outdir)]
    if sweep_mode is not None:
        cmd += ["--sweep-mode", sweep_mode]
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True)
    assert proc.returncode == 0, \
        "permute failed:\n%s\n%s" % (proc.stdout[-3000:], proc.stderr[-1000:])
    return proc


@pytest.fixture()
def project(tmp_path):
    p = Project()
    p.new_document(str(tmp_path / "sweepmodes.FCStd"))
    yield p
    p.shutdown()


def build_scene(project, gap=10.0):
    """A single iris element whose dim_IRIS.hole_diameter cell is an
    EXPRESSION referencing the miewb_vars sheet (not a value directly
    swept by --var), so a miewb_vars.gap sweep only moves the geometry if
    permute_model's rebuild-fix (C3) actually fires."""
    fc, doc = project.fc, project.doc
    fc.request("create_sheet", {"doc": doc, "label": "miewb_vars"})
    fc.request("set_cell", {"doc": doc, "sheet": "miewb_vars",
                            "cell": "B1", "raw": "=%g" % gap,
                            "alias": "gap"})
    project._refetch_structure()

    project.import_primitive(
        os.path.join(PRIMITIVES, "iris.FCStd"), "IRIS")
    fc.request("set_spreadsheet", {
        "doc": doc, "sheet": "dim_IRIS", "alias": "hole_diameter",
        "raw": "=<<miewb_vars>>.gap * 1mm"})
    project._refetch_structure()


def iris_body(fc, path, element="IRIS"):
    """The iris DISC body (not its material=air hole-filling plug) of the
    `element` group: import_primitive does not guarantee the disc's own
    Label equals the element label for every primitive kind (only that
    every member shares miewb_group == element), so select by group and
    exclude the "_plug" member instead of looking up by Label."""
    structure = fc.open_document(str(path))
    try:
        members = [b for b in structure["bodies"]
                  if b["properties"].get("miewb_group", {}).get("value")
                  == element]
        disc = [b for b in members if not b["label"].endswith("_plug")]
        assert len(disc) == 1, (element, [b["label"] for b in members])
        return disc[0]
    finally:
        fc.close(structure["doc"])


def test_sweep_mode_zip_filenames_match_prediction(project, tmp_path):
    build_scene(project, gap=10.0)
    project.save()
    stem = Path(project.fcstd_path).stem

    varspecs = [("miewb_vars.gap", 10.0, 20.0, 1),
                ("dim_IRIS.outer_diameter", 25.0, 30.0, 1)]
    outdir = tmp_path / "variants"
    run_permute(project.fcstd_path, outdir, varspecs, sweep_mode="zip")

    predicted = run_pipeline.variant_output_names(stem, varspecs, "zip")
    assert len(predicted) == 2, predicted   # zip of two length-2 lists

    got = sorted(p[:-len(".FCStd")] for p in os.listdir(outdir)
                if p.endswith(".FCStd"))
    assert got == sorted(predicted), (got, predicted)

    # product mode on the SAME varspecs would have produced 4 variants;
    # confirm zip really did combine them pairwise, not cartesian
    product_predicted = run_pipeline.variant_output_names(
        stem, varspecs, "product")
    assert len(product_predicted) == 4
    assert set(predicted) != set(product_predicted)


def test_miewb_vars_sweep_rebuilds_expression_linked_primitive(
        project, tmp_path):
    build_scene(project, gap=10.0)
    project.save()
    stem = Path(project.fcstd_path).stem

    varspecs = [("miewb_vars.gap", 10.0, 20.0, 1)]   # 2 values: 10, 20
    outdir = tmp_path / "variants"
    run_permute(project.fcstd_path, outdir, varspecs)

    names = run_pipeline.variant_output_names(stem, varspecs, "product")
    assert len(names) == 2
    variants = [outdir / ("%s.FCStd" % n) for n in names]
    for v in variants:
        assert v.exists(), sorted(os.listdir(outdir))

    bodies = [iris_body(project.fc, v) for v in variants]
    shape_keys = [b["shape_key"] for b in bodies]
    # the two gap values (10 vs 20) must produce geometrically DIFFERENT
    # iris discs (the hole cut is a different size) -- this is exactly
    # the case the pre-fix code missed, since dim_IRIS was never in
    # touched_sheets (only miewb_vars was directly swept).
    assert shape_keys[0] != shape_keys[1], shape_keys
