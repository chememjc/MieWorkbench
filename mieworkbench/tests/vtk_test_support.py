"""Shared test-only support for the 3D-view test suite: a trivial binary
STL writer (so tests never need a real FreeCAD/tessellation pipeline), a
couple of canned two/three-body "scenes" shaped exactly like
Project.structure/Project.faces, a minimal .vtp writer for the rays-
overlay tests, and FakeProject - a tiny QObject stand-in for
core.project.Project used by every offscreen widget-construction test.

Not a test module itself (no test_ prefix) so pytest never collects it.
"""

import os
import struct

from PySide6.QtCore import QObject, Signal

from vtkmodules.vtkCommonCore import (vtkFloatArray, vtkPoints,
                                      vtkUnsignedCharArray)
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter


# ---------------------------------------------------------------------------
# trivial binary STL (one triangle -> a valid, tiny, body-local-metres face)
# ---------------------------------------------------------------------------
def write_triangle_stl(path, base=(0.0, 0.0, 0.0), size=0.01):
    x, y, z = base
    triangle = ((0.0, 0.0, 1.0),
               (x, y, z), (x + size, y, z), (x, y + size, z))
    with open(path, "wb") as fh:
        fh.write(b"\x00" * 80)
        fh.write(struct.pack("<I", 1))
        for vec in triangle:
            fh.write(struct.pack("<3f", *vec))
        fh.write(struct.pack("<H", 0))


# ---------------------------------------------------------------------------
# canned scenes
# ---------------------------------------------------------------------------
def _placement(pos_mm):
    return {"pos_mm": list(pos_mm), "quat": [0.0, 0.0, 0.0, 1.0]}


def make_two_body_scene(tmp_path):
    """Lens (optic, 1 face) + Screen (detector, 1 face) -- exactly 2 STL
    files, the minimal scene for widget-construction tests."""
    lens_stl = os.path.join(str(tmp_path), "lens_face1.stl")
    screen_stl = os.path.join(str(tmp_path), "screen_face1.stl")
    write_triangle_stl(lens_stl, base=(0.0, 0.0, 0.0), size=0.01)
    write_triangle_stl(screen_stl, base=(0.0, 0.0, 0.0), size=0.02)

    lens = {
        "name": "Lens", "label": "Lens", "tip": "Revolution",
        "face_count": 1, "solid_closed": True, "volume_mm3": 100.0,
        "center_of_mass_mm": [0.0, 0.0, 0.0],
        "bbox_mm": [-5.0, -5.0, -5.0, 5.0, 5.0, 5.0],
        "placement": _placement([0.0, 0.0, 0.0]),
        "placement_bound": False, "shape_key": "k1",
        "properties": {
            "material": {"type": "App::PropertyString", "group": "Base",
                        "value": "BK7"},
            "coating": {"type": "App::PropertyString", "group": "Base",
                       "value": "MgF2"},
            "miewb_primitive": {"type": "App::PropertyString",
                                "group": "Base", "value": "lens"},
            "miewb_group": {"type": "App::PropertyString", "group": "Base",
                           "value": "lensgrp"},
        },
    }
    screen = {
        "name": "Screen", "label": "Screen", "tip": "Pad",
        "face_count": 1, "solid_closed": True, "volume_mm3": 50.0,
        "center_of_mass_mm": [50.0, 0.0, 0.0],
        "bbox_mm": [45.0, -10.0, -10.0, 55.0, 10.0, 10.0],
        "placement": _placement([50.0, 0.0, 0.0]),
        "placement_bound": False, "shape_key": "k2",
        "properties": {
            "material": {"type": "App::PropertyString", "group": "Base",
                        "value": "detector"},
        },
    }
    structure = {
        "bodies": [lens, screen],
        "sheets": [
            {"name": "dim", "label": "dim", "aliases": {
                "lensth": {"cell": "B5", "raw": "=2 mm", "value": 2.0,
                          "unit": "mm"},
                "wavelength": {"cell": "B9", "raw": "633", "value": 633.0,
                              "unit": ""},
            }},
        ],
    }
    faces = {
        "Lens": {"faces": [
            {"id": "Lens.Revolution.Face1", "stl": lens_stl,
             "area_m2": 1e-4, "centroid_m": [0.0, 0.0, 0.0],
             "normal_hint": [0.0, 0.0, 1.0]},
        ], "placement": lens["placement"]},
        "Screen": {"faces": [
            {"id": "Screen.Pad.Face1", "stl": screen_stl,
             "area_m2": 4e-4, "centroid_m": [0.0, 0.0, 0.0],
             "normal_hint": [1.0, 0.0, 0.0]},
        ], "placement": screen["placement"]},
    }
    return structure, faces


def make_two_member_group_scene(tmp_path):
    """A genuine MULTI-body element: an achromat 'achro' with two member
    bodies (Crown + Flint sharing miewb_group='achro') plus an ungrouped
    Screen detector -- for the element-vs-sub-selection (WP4) tests."""
    crown_stl = os.path.join(str(tmp_path), "crown_face1.stl")
    flint_stl = os.path.join(str(tmp_path), "flint_face1.stl")
    screen_stl = os.path.join(str(tmp_path), "grp_screen_face1.stl")
    write_triangle_stl(crown_stl, base=(0.0, 0.0, 0.0), size=0.01)
    write_triangle_stl(flint_stl, base=(0.0, 0.0, 0.01), size=0.01)
    write_triangle_stl(screen_stl, base=(0.0, 0.0, 0.0), size=0.02)

    def _body(name, tip, com, mat, group):
        return {
            "name": name, "label": name, "tip": tip,
            "face_count": 1, "solid_closed": True, "volume_mm3": 100.0,
            "center_of_mass_mm": list(com),
            "bbox_mm": [com[0] - 5, -5.0, -5.0, com[0] + 5, 5.0, 5.0],
            "placement": _placement(com),
            "placement_bound": False, "shape_key": name,
            "properties": {
                "material": {"type": "App::PropertyString", "group": "Base",
                            "value": mat},
                "miewb_primitive": {"type": "App::PropertyString",
                                    "group": "Base", "value": "achromat"},
                "miewb_group": {"type": "App::PropertyString",
                               "group": "Base", "value": group},
            },
        }

    crown = _body("Crown", "Revolution", [0.0, 0.0, 0.0], "BK7", "achro")
    flint = _body("Flint", "Revolution", [3.0, 0.0, 0.0], "SF2", "achro")
    screen = {
        "name": "Screen", "label": "Screen", "tip": "Pad",
        "face_count": 1, "solid_closed": True, "volume_mm3": 50.0,
        "center_of_mass_mm": [50.0, 0.0, 0.0],
        "bbox_mm": [45.0, -10.0, -10.0, 55.0, 10.0, 10.0],
        "placement": _placement([50.0, 0.0, 0.0]),
        "placement_bound": False, "shape_key": "scr",
        "properties": {
            "material": {"type": "App::PropertyString", "group": "Base",
                        "value": "detector"},
        },
    }
    structure = {
        "bodies": [crown, flint, screen],
        "sheets": [{"name": "dim_achro", "label": "dim_achro", "aliases": {}}],
    }
    faces = {
        "Crown": {"faces": [
            {"id": "Crown.Revolution.Face1", "stl": crown_stl,
             "area_m2": 1e-4, "centroid_m": [0.0, 0.0, 0.0],
             "normal_hint": [0.0, 0.0, 1.0]}],
            "placement": crown["placement"]},
        "Flint": {"faces": [
            {"id": "Flint.Revolution.Face1", "stl": flint_stl,
             "area_m2": 1e-4, "centroid_m": [0.003, 0.0, 0.0],
             "normal_hint": [0.0, 0.0, 1.0]}],
            "placement": flint["placement"]},
        "Screen": {"faces": [
            {"id": "Screen.Pad.Face1", "stl": screen_stl,
             "area_m2": 4e-4, "centroid_m": [0.0, 0.0, 0.0],
             "normal_hint": [1.0, 0.0, 0.0]}],
            "placement": screen["placement"]},
    }
    return structure, faces


def make_lens_two_faces_scene(tmp_path):
    """Lens with TWO faces (for facemap partial-assignment tests, where
    assigning to only one of several faces must NOT collapse to the
    bare all-faces form) + a one-face Screen."""
    structure, faces = make_two_body_scene(tmp_path)
    face2_stl = os.path.join(str(tmp_path), "lens_face2.stl")
    write_triangle_stl(face2_stl, base=(0.0, 0.0, 0.01), size=0.01)
    faces["Lens"]["faces"].append({
        "id": "Lens.Revolution.Face2", "stl": face2_stl, "area_m2": 1e-4,
        "centroid_m": [0.0, 0.0, 0.01], "normal_hint": [0.0, 0.0, 1.0]})
    for body in structure["bodies"]:
        if body["name"] == "Lens":
            body["face_count"] = 2
    return structure, faces


# ---------------------------------------------------------------------------
# minimal .vtp writer for the rays-overlay tests
# ---------------------------------------------------------------------------
def write_simple_vtp(path, with_rgb=False, rel_power=None, opl=None):
    """One 2-point line cell per entry of `rel_power` (default: a single
    cell). `rel_power`, when given, also writes the float cell array of
    the same name that vtkexport.write_vtp_polylines emits for the
    attenuation-dimming feature; `opl` is a parallel list of (opl0, opl1)
    metre pairs writing the bead-animation timing arrays. Each cell i
    runs from (0, i, 0) to (1, i, 0)."""
    n_cells = max(len(rel_power) if rel_power is not None else 1,
                  len(opl) if opl is not None else 1)
    points = vtkPoints()
    lines = vtkCellArray()
    for i in range(n_cells):
        points.InsertNextPoint(0.0, float(i), 0.0)
        points.InsertNextPoint(1.0, float(i), 0.0)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(2 * i)
        lines.InsertCellPoint(2 * i + 1)

    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    if with_rgb:
        colors = vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("rgb")
        for _ in range(n_cells):
            colors.InsertNextTuple3(255, 255, 0)
        # AddArray, deliberately NOT SetScalars: real pre-fix rays.vtp
        # files carry 'rgb' as a plain (non-active) cell array, and the
        # GUI's field-data coloring must handle exactly that (the old
        # SetScalars fixture masked the white-rays bug)
        polydata.GetCellData().AddArray(colors)
    if rel_power is not None:
        rel = vtkFloatArray()
        rel.SetNumberOfComponents(1)
        rel.SetName("rel_power")
        for v in rel_power:
            rel.InsertNextValue(float(v))
        polydata.GetCellData().AddArray(rel)
    if opl is not None:
        for name, col in (("opl0", 0), ("opl1", 1)):
            arr = vtkFloatArray()
            arr.SetNumberOfComponents(1)
            arr.SetName(name)
            for pair in opl:
                arr.InsertNextValue(float(pair[col]))
            polydata.GetCellData().AddArray(arr)

    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.Write()


# ---------------------------------------------------------------------------
# FakeProject - Project stand-in (same signals, canned data, in-memory
# mutation) for offscreen pane tests that never touch FreeCAD.
# ---------------------------------------------------------------------------
class FakeProject(QObject):
    sceneLoaded = Signal()
    bodiesReshaped = Signal(list)
    bodiesMoved = Signal(dict)
    propertiesChanged = Signal(str)
    dirtyChanged = Signal(bool)

    def __init__(self, structure, faces, parent=None):
        super().__init__(parent)
        self.structure = structure
        self.faces = faces
        self.body_states = {}
        self.calls = []   # audit log of mutating calls, in call order

    def body(self, name):
        for b in self.structure.get("bodies", []):
            if b["name"] == name or b["label"] == name:
                return b
        raise KeyError("no body %r" % name)

    def body_names(self):
        return [b["name"] for b in self.structure.get("bodies", [])]

    # -- element identity (mirrors core.project.Project) --------------------
    def element_group(self, body_name):
        b = self.body(body_name)
        return (b["properties"].get("miewb_group", {}).get("value")
                or b["label"])

    def element_bodies(self, element):
        element = str(element)
        names = [b["name"] for b in self.structure.get("bodies", [])
                 if b["properties"].get("miewb_group", {}).get("value")
                 == element]
        if names:
            return names
        b = self.body(element)
        group = b["properties"].get("miewb_group", {}).get("value")
        if group:
            return [x["name"] for x in self.structure.get("bodies", [])
                    if x["properties"].get("miewb_group", {}).get("value")
                    == group]
        return [b["name"]]

    def is_open(self):
        return True

    def current_placement(self, name):
        return None

    def resolver(self):
        class _Resolver:
            def resolve_point(self, spec):
                raise ValueError("no resolver in FakeProject")

            def resolve_axis(self, spec):
                raise ValueError("no resolver in FakeProject")
        return _Resolver()

    def sheets(self):
        return self.structure.get("sheets", [])

    def sheet_for_body(self, name):
        b = self.body(name)
        group = b["properties"].get("miewb_group", {}).get("value")
        for sheet in self.sheets():
            if group and sheet["label"] == "dim_%s" % group:
                return sheet
        for sheet in self.sheets():
            if sheet["label"] == "dim":
                return sheet
        return None

    def set_property(self, body, name, value, ptype=None):
        self.calls.append(("set_property", body, name, value))
        b = self.body(body)
        b["properties"][name] = {"type": "App::PropertyString",
                                 "group": "Base", "value": value}
        self.propertiesChanged.emit(body)

    def remove_property(self, body, name):
        self.calls.append(("remove_property", body, name))
        b = self.body(body)
        b["properties"].pop(name, None)
        self.propertiesChanged.emit(body)

    def train(self):
        """Real TrainModel over the fake structure (paraxial readout +
        insert-optical-value tests)."""
        from mieworkbench.core.train import TrainModel
        return TrainModel(self.structure, self.body_states)

    def train_variables(self):
        return {}

    def set_spreadsheet(self, sheet, alias, raw, rebuild_group=None):
        self.calls.append(("set_spreadsheet", sheet, alias, raw))
        for s in self.sheets():
            if s["label"] == sheet and alias in s.get("aliases", {}):
                s["aliases"][alias]["raw"] = raw
        self.propertiesChanged.emit("")
        if rebuild_group:
            self.rebuild_primitive(rebuild_group)

    def set_element_parameters(self, sheet, values, rebuild_group=None,
                               text=None):
        for alias, raw in values.items():
            self.set_spreadsheet(sheet, alias, raw)
        if rebuild_group:
            self.rebuild_primitive(rebuild_group)

    def rebuild_primitive(self, group):
        self.calls.append(("rebuild_primitive", group))
        names = [b["name"] for b in self.structure.get("bodies", [])
                if b["properties"].get("miewb_group", {}).get("value")
                == group]
        self.bodiesReshaped.emit(names)
        self.propertiesChanged.emit(names[0] if names else "")
