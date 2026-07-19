"""Project.apply_parameter_values (WP3 "Apply optimum" backend).

Two tiers:
  1. Scripted-worker (no FreeCAD): address routing + validation, the
     dim-cell set+rebuild-in-one-command undo ordering, and the
     fail-atomically guards (expression-bound cell, unknown address).
  2. A real-FreeCAD end-to-end (marked 'freecad', env-gated): apply a
     miewb_vars value AND a dim primitive cell in one step, then a single
     undo() restores both (worker structures compared equal).
"""

import copy
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

from mieworkbench.core.project import Project, ProjectError  # noqa: E402
from mieworkbench.core.geomcache import GeomCache  # noqa: E402
from mieworkbench.core.transforms import BodyState  # noqa: E402
from mieworkbench.tests.test_undo import FakeWorkerFc  # noqa: E402


# ---------------------------------------------------------------------------
# 1. scripted worker
# ---------------------------------------------------------------------------
def _dim_project(dim_raw="=2 mm"):
    """A Project over a fake worker whose one primitive ("Lens") is
    driven by a dim_Lens sheet cell (alias 'ct')."""
    structure = {
        "doc": "scene", "label": "scene", "file": "/nowhere/scene.FCStd",
        "bodies": [{
            "name": "Lens", "label": "Lens", "tip": "Pad", "face_count": 1,
            "solid_closed": True, "volume_mm3": 1.0,
            "center_of_mass_mm": [0, 0, 0], "bbox_mm": [0, 0, 0, 1, 1, 1],
            "placement": {"pos_mm": [0.0, 0.0, 0.0],
                          "quat": [0.0, 0.0, 0.0, 1.0]},
            "placement_bound": False, "shape_key": "k1",
            "properties": {
                "material": {"type": "App::PropertyString",
                             "group": "Base", "value": "bk7"},
                "miewb_primitive": {"type": "App::PropertyString",
                                    "group": "Base", "value": "lens_pcx"},
                "miewb_group": {"type": "App::PropertyString",
                                "group": "Base", "value": "Lens"},
            },
        }],
        "sheets": [{"name": "Sheet", "label": "dim_Lens", "aliases": {
            "ct": {"cell": "B1", "raw": dim_raw, "value": 2.0, "unit": "mm"},
        }}],
    }
    project = Project()
    fake = FakeWorkerFc(structure)
    project._fc = fake
    project._cache = GeomCache(fake, cache_root=tempfile.mkdtemp(
        prefix="miewb_apply_test_"))
    project.doc = "scene"
    project.fcstd_path = "/nowhere/scene.FCStd"
    project.structure = fake.request("get_structure", {"doc": "scene"})
    project.body_states["Lens"] = BodyState.from_worker(
        project.structure["bodies"][0], [])
    return project, fake


def test_dim_cell_applied_and_rebuilt():
    project, fake = _dim_project()
    n = project.apply_parameter_values({"dim_Lens.ct": 6.0})
    assert n == 1
    assert project._sheet_raw("dim_Lens", "ct") == "=6 mm"
    ops = [op for op, _ in fake.ops]
    assert "set_spreadsheet" in ops and "rebuild_primitive" in ops


def test_dim_cell_undo_restores_value_then_rebuilds():
    project, fake = _dim_project()
    project.apply_parameter_values({"dim_Lens.ct": 6.0})
    fake.ops.clear()
    project.undo()
    assert project._sheet_raw("dim_Lens", "ct") == "=2 mm"
    order = [op for op, _ in fake.ops
             if op in ("set_spreadsheet", "rebuild_primitive")]
    # the rebuild must come AFTER the value restore or geometry is stale
    assert order == ["set_spreadsheet", "rebuild_primitive"]


def test_dim_cell_undo_is_one_step():
    project, _ = _dim_project()
    project.apply_parameter_values({"dim_Lens.ct": 6.0})
    assert project.undo_stack.can_undo()
    project.undo()
    assert not project.undo_stack.can_undo()   # ONE composite step


def test_expression_bound_dim_cell_raises_and_changes_nothing():
    project, fake = _dim_project(dim_raw="=<<miewb_vars>>.gap * 1mm")
    fake.ops.clear()
    with pytest.raises(ProjectError) as exc:
        project.apply_parameter_values({"dim_Lens.ct": 6.0})
    assert "expression-bound" in str(exc.value)
    assert "dim_Lens.ct" in str(exc.value)
    # nothing was written (validation happens before the macro opens)
    assert not any(op in ("set_spreadsheet", "rebuild_primitive", "set_cell")
                   for op, _ in fake.ops)
    assert not project.undo_stack.can_undo()


def test_unknown_address_raises_atomically():
    project, fake = _dim_project()
    fake.ops.clear()
    with pytest.raises(ProjectError):
        project.apply_parameter_values({"nope": 1.0})
    assert not project.undo_stack.can_undo()


def test_missing_dim_alias_raises():
    project, _ = _dim_project()
    with pytest.raises(ProjectError):
        project.apply_parameter_values({"dim_Lens.absent": 1.0})


def test_empty_assignments_is_noop():
    project, _ = _dim_project()
    assert project.apply_parameter_values({}) == 0
    assert not project.undo_stack.can_undo()


def test_cell_is_literal_classifier():
    lit = Project._cell_is_literal
    assert lit("=5 mm")
    assert lit("=5")
    assert lit("1e-3 mm")
    assert not lit("=<<miewb_vars>>.gap * 1mm")
    assert not lit("=2*ct")
    assert not lit("=")


# ---------------------------------------------------------------------------
# 2. real FreeCAD worker (env-gated)
# ---------------------------------------------------------------------------
def _structure_projection(structure):
    """A stable projection for comparing worker structures across an
    apply+undo cycle: each sheet's alias raws + each body's Base
    properties and placement."""
    sheets = {s.get("label"): {a: e.get("raw")
                               for a, e in (s.get("aliases") or {}).items()}
              for s in structure.get("sheets", [])}
    bodies = {b["name"]: {"props": {k: v.get("value")
                                    for k, v in b["properties"].items()},
                          "placement": b.get("placement")}
              for b in structure.get("bodies", [])}
    return {"sheets": sheets, "bodies": bodies}


@pytest.mark.freecad
def test_apply_parameter_values_real_worker(tmp_path):
    import make_demos
    from mieworkbench.core.fcclient import FcClient
    from mieworkbench.core import variables as variables_mod

    fc = FcClient()
    try:
        demo = make_demos.Demo(fc, tmp_path / "s.FCStd")
        make_demos.DEMOS["michelson_folded"](demo)
        project = demo.project

        rows = variables_mod.parse_sheet(project.variables_sheet())
        if not rows:
            pytest.skip("demo carries no miewb_vars variable")
        varname, varrow = next(iter(rows.items()))

        dim_addr = None
        for sheet in project.sheets():
            label = sheet.get("label") or ""
            if not label.startswith("dim_"):
                continue
            for alias, entry in (sheet.get("aliases") or {}).items():
                raw = entry.get("raw") or ""
                if project._cell_is_literal(raw) and "vars" not in raw:
                    dim_addr = (label, alias, raw)
                    break
            if dim_addr:
                break
        if dim_addr is None:
            pytest.skip("demo has no literal dim_ primitive cell")
        label, alias, old_dim_raw = dim_addr

        before = _structure_projection(copy.deepcopy(project.structure))
        old_var = varrow.value

        n = project.apply_parameter_values({
            "miewb_vars.%s" % varname: old_var + 1.0,
            "%s.%s" % (label, alias): 5.0,
        })
        assert n == 2
        # applied
        after = variables_mod.parse_sheet(project.variables_sheet())
        assert abs(after[varname].value - old_var) > 0.5
        assert project._sheet_raw(label, alias) != old_dim_raw

        # ONE undo restores BOTH
        assert project.undo_stack.can_undo()
        project.undo()
        assert not project.undo_stack.can_undo()
        restored = variables_mod.parse_sheet(project.variables_sheet())
        assert abs(restored[varname].value - old_var) < 1e-9
        assert project._sheet_raw(label, alias) == old_dim_raw
        assert _structure_projection(project.structure) == before
    finally:
        fc.shutdown()
