#!/usr/bin/env python3
# =============================================================================
# _primitivelib_fc_probe.py -- FreeCAD-embedded probe script for
# test_primitivelib.py's FreeCAD-gated checks (legacy-alias fallback in
# primitivelib.read_params, and rebuild_element's round_flag round-trip +
# label/placement/prop preservation).
#
# Interpreter: the FreeCAD AppImage (see CLAUDE.md):
#   /home3/freecad/FreeCAD.AppImage -c _primitivelib_fc_probe.py -- \
#       --out <result.json> < /dev/null
#
# Usual -c caveats: bare '--' before args; script runs TWICE (writing the
# result JSON is idempotent -- just overwritten each time); no __main__
# guard; os._exit instead of sys.exit; print() may be dropped so also log
# via FreeCAD.Console.
# =============================================================================
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPTS_DIR = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, SCRIPTS_DIR)

import FreeCAD as App  # noqa: E402
import primitivelib as pl  # noqa: E402

App.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
    "CreateBackupFiles", False)


def log(msg):
    App.Console.PrintMessage(msg + "\n")
    print(msg, flush=True)


def parse_args():
    argv = sys.argv[1:]
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    try:
        return p.parse_args(argv)
    except SystemExit as exc:
        os._exit(exc.code if isinstance(exc.code, int) else 2)


def probe_legacy_fallback():
    """A hand-built 'legacy' sheet carrying only the OLD 'radius' alias (no
    'diameter', no 'round_flag') for laser_collimated. read_params must
    report diameter = radius * 2 and round_flag = its spec default."""
    doc = App.newDocument("legacy_probe")
    try:
        sheet = doc.addObject("Spreadsheet::Sheet", "Spreadsheet")
        sheet.Label = "dim"
        sheet.set("A1", "radius")
        sheet.set("B1", "=5 mm")
        sheet.setAlias("B1", "radius")
        sheet.set("A2", "length")
        sheet.set("B2", "=12 mm")
        sheet.setAlias("B2", "length")
        doc.recompute()
        params = pl.read_params(sheet, "laser_collimated")
        return {"diameter": params["diameter"], "length": params["length"],
                "round_flag": params["round_flag"]}
    finally:
        App.closeDocument(doc.Name)


def probe_rebuild_round_flag_roundtrip():
    """Open the regenerated primitives/window.FCStd (round_flag=1 default,
    a cylinder -> 3 faces), give its body a placement/label/extra prop,
    flip round_flag to 0 in the sheet, rebuild_element, and check: (a) the
    body becomes a box (6 faces), (b) label/placement/extra prop survive."""
    win_path = os.path.join(SCRIPTS_DIR, "..", "primitives", "window.FCStd")
    doc = App.openDocument(os.path.normpath(win_path))
    try:
        body = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"
                and getattr(o, "miewb_group", None) == "window"][0]
        round_faces = len(body.Shape.Faces)

        placement = App.Placement(App.Vector(1.0, 2.0, 3.0), App.Rotation())
        body.Placement = placement
        body.Label = "MyWindowLabel"
        pl.safe_set_props(body, {"filter": "probe_marker"})

        sheet = doc.getObject("Spreadsheet")
        cell = sheet.getCellFromAlias("round_flag")
        sheet.set(cell, "0")
        doc.recompute()

        new_bodies = pl.rebuild_element(doc, sheet, "window", "window")
        nb = new_bodies[0]
        rect_faces = len(nb.Shape.Faces)
        return {
            "round_faces": round_faces,
            "rect_faces": rect_faces,
            "label": nb.Label,
            "placement_base": [nb.Placement.Base.x, nb.Placement.Base.y,
                               nb.Placement.Base.z],
            "filter_prop": getattr(nb, "filter", None),
        }
    finally:
        App.closeDocument(doc.Name)


BATCH3_KINDS = [
    "bs_cube", "anamorphic_pair", "polarizer_glan_taylor", "mirror_parabolic",
]

BATCHC_KINDS = [
    "fiber_optic", "mirror_annular",
]

# samples-instruments round: cuvettes/vials/vat (nested WALL+LIQUID pair,
# two bodies each), the bare air sample_region (one body), and the four
# lamp/image sources (one body each).
SAMPLES_CELL_KINDS = [
    "cuvette_square", "cuvette_capillary", "flow_cell",
    "vial_cylindrical", "vat_cylindrical", "sample_region",
]
SAMPLES_SOURCE_KINDS = [
    "tungsten_halogen", "d2_lamp", "hg_calibration", "source_image",
]

NEW_KINDS = [
    "bs_plate", "pbs_plate", "dichroic_plate", "pellicle", "nd_filter",
    "nd_reflective", "filter_bandpass", "filter_longpass", "filter_shortpass",
    "filter_notch", "window_wedged", "diffuser_plate",
    "prism_right_angle", "prism_wedge", "prism_dove", "prism_penta",
    "prism_rhomboid", "mirror_concave", "mirror_convex", "mirror_d_shaped",
    "iris", "iris_bladed", "pinhole", "slit", "retro_corner_cube",
] + BATCH3_KINDS + BATCHC_KINDS + SAMPLES_CELL_KINDS + SAMPLES_SOURCE_KINDS


def probe_new_kinds_build_rebuild():
    """Every new (v2-feature-round batch 1+2) primitive: build with default
    params, stamp a Label/Placement onto the first body, rebuild_element
    from the sheet unchanged, and check the label/placement survive and the
    body count is stable (2 for iris/pinhole/slit, 1 otherwise)."""
    out = {}
    for kind in NEW_KINDS:
        doc = App.newDocument("probe_" + kind)
        try:
            sheet = pl.make_sheet(doc, kind, label="dim")
            bodies = pl.build_primitive(doc, kind, group=kind)
            n_before = len(bodies)
            b0 = bodies[0]
            placement = App.Placement(App.Vector(4.0, 5.0, 6.0), App.Rotation())
            b0.Placement = placement
            b0.Label = "Probe_" + kind
            doc.recompute()
            new_bodies = pl.rebuild_element(doc, sheet, kind, kind)
            nb0 = [b for b in new_bodies if b.Label == "Probe_" + kind][0]
            out[kind] = {
                "n_before": n_before,
                "n_after": len(new_bodies),
                "label_ok": nb0.Label == "Probe_" + kind,
                "placement_ok": [nb0.Placement.Base.x, nb0.Placement.Base.y,
                                 nb0.Placement.Base.z] == [4.0, 5.0, 6.0],
            }
        finally:
            App.closeDocument(doc.Name)
    return out


def probe_apertures():
    """iris/pinhole/slit: two bodies (disc/plate + air plug), plug
    material=air, disc absorbance == blackness -- both right after the
    initial build AND after a rebuild with a CHANGED blackness value (the
    derived_props mechanism must re-derive absorbance from the sheet every
    time, not preserve the stale pre-rebuild value)."""
    out = {}
    for kind in ("iris", "iris_bladed", "pinhole", "slit"):
        doc = App.newDocument("probe_ap_" + kind)
        try:
            sheet = pl.make_sheet(doc, kind, label="dim")
            bodies = pl.build_primitive(doc, kind, group=kind)
            disc = [b for b in bodies if not b.Name.endswith("_plug")][0]
            plug = [b for b in bodies if b.Name.endswith("_plug")][0]
            first = {
                "n_bodies": len(bodies),
                "plug_material": getattr(plug, "material", None),
                "disc_absorbance": getattr(disc, "absorbance", None),
                "blackness_param": pl.PRIMITIVES[kind]["params"]["blackness"]
                ["default"],
            }
            cell = sheet.getCellFromAlias("blackness")
            sheet.set(cell, "0.5")
            doc.recompute()
            new_bodies = pl.rebuild_element(doc, sheet, kind, kind)
            disc2 = [b for b in new_bodies if not b.Name.endswith("_plug")][0]
            plug2 = [b for b in new_bodies if b.Name.endswith("_plug")][0]
            second = {
                "n_bodies": len(new_bodies),
                "plug_material": getattr(plug2, "material", None),
                "disc_absorbance": getattr(disc2, "absorbance", None),
            }
            out[kind] = {"initial": first, "after_blackness_rebuild": second}
        finally:
            App.closeDocument(doc.Name)
    return out


def probe_bladed_iris_polygon():
    """iris_bladed: the air plug is a REGULAR-POLYGON prism (n_blades sides),
    its cross-section area matches the analytic regular-n-gon area for the
    inscribed clear aperture, and a rebuild with a changed n_blades actually
    changes the polygon (face count follows n)."""
    import math as _m
    kind = "iris_bladed"
    doc = App.newDocument("probe_bladed")
    try:
        sheet = pl.make_sheet(doc, kind, label="dim")
        bodies = pl.build_primitive(doc, kind, group=kind)
        plug = [b for b in bodies if b.Name.endswith("_plug")][0]
        p = pl.PRIMITIVES[kind]["params"]
        n = int(round(p["n_blades"]["default"]))
        r_in = p["aperture_diameter"]["default"] / 2.0
        th = p["thickness"]["default"]
        # regular n-gon inscribed-circle-radius r_in: area = n r_in^2 tan(pi/n)
        area_analytic = n * r_in * r_in * _m.tan(_m.pi / n)
        first = {
            "n_plug_side_faces": len(plug.Shape.Faces) - 2,   # minus 2 caps
            "n_blades": n,
            "plug_vol_mm3": plug.Shape.Volume,
            "plug_vol_analytic_mm3": area_analytic * th,
        }
        # rebuild with n_blades = 8
        sheet.set(sheet.getCellFromAlias("n_blades"), "8")
        doc.recompute()
        new_bodies = pl.rebuild_element(doc, sheet, kind, kind)
        plug2 = [b for b in new_bodies if b.Name.endswith("_plug")][0]
        second = {"n_plug_side_faces": len(plug2.Shape.Faces) - 2}
        return {"initial": first, "after_octagon_rebuild": second}
    finally:
        App.closeDocument(doc.Name)


def probe_edge_blackened_lens():
    """lens_pcx edge_blackened: the flag lands as a bool body property, and a
    rebuild after toggling the sheet param RE-DERIVES it (True->False), not
    restoring the stale pre-rebuild value (the derived_props mechanism)."""
    kind = "lens_pcx"
    doc = App.newDocument("probe_edge_black")
    try:
        sheet = pl.make_sheet(doc, kind, label="dim")
        pl.build_primitive(doc, kind, group=kind)
        # edit the sheet ON, then rebuild so the flag reflects the edited sheet
        sheet.set(sheet.getCellFromAlias("edge_blackened"), "1")
        doc.recompute()
        bodies = pl.rebuild_element(doc, sheet, kind, kind)
        on = bool(getattr(bodies[0], "edge_blackened", None))
        # toggle OFF and rebuild
        sheet.set(sheet.getCellFromAlias("edge_blackened"), "0")
        doc.recompute()
        bodies2 = pl.rebuild_element(doc, sheet, kind, kind)
        off = bool(getattr(bodies2[0], "edge_blackened", None))
        return {"after_on": on, "after_off": off}
    finally:
        App.closeDocument(doc.Name)


def probe_corner_cube():
    """retro_corner_cube: exactly 4 faces, 3 of which (the reflecting
    trihedral) are mutually perpendicular (pairwise normal dot ~ 0), the
    4th being the entrance face."""
    doc = App.newDocument("probe_corner_cube")
    try:
        bodies = pl.build_primitive(doc, "retro_corner_cube",
                                    group="retro_corner_cube")
        body = bodies[0]
        normals = []
        for f in body.Shape.Faces:
            u0, u1, v0, v1 = f.ParameterRange
            normals.append(f.normalAt((u0 + u1) / 2.0, (v0 + v1) / 2.0))
        n_faces = len(normals)
        # the 3 "back" faces are the ones whose pairwise dots include two
        # near-zero values each (mutually perpendicular); the entrance face
        # is the odd one out.
        dots = []
        for i in range(n_faces):
            for j in range(i + 1, n_faces):
                dots.append(abs(normals[i].dot(normals[j])))
        n_perp_pairs = sum(1 for d in dots if d < 1e-6)
        return {"n_faces": n_faces, "n_perp_pairs": n_perp_pairs}
    finally:
        App.closeDocument(doc.Name)


def _gap(body_a, body_b):
    """Min distance (mm) between two bodies' shapes -- 0.0 if touching or
    overlapping, >0 if a real air/cement gap separates them."""
    return body_a.Shape.distToShape(body_b.Shape)[0]


def probe_batch3_geometry():
    """bs_cube: two bodies, a real hypotenuse gap, coating on exactly one
    body/face (the entrance prism); anamorphic_pair: two non-overlapping
    bk7 bodies; polarizer_glan_taylor: two calcite bodies with
    crystal_axis, a real gap at the cut."""
    out = {}

    doc = App.newDocument("probe_bs_cube")
    try:
        bodies = pl.build_primitive(doc, "bs_cube", group="bs_cube")
        cube = [b for b in bodies if b.Name == "bs_cube"][0]
        plate = [b for b in bodies if b.Name.endswith("_split")][0]
        # nested-plate design: the splitter must sit STRICTLY inside the
        # cube (proper nesting -- the common volume equals the plate's)
        common_vol = cube.Shape.common(plate.Shape).Volume
        out["bs_cube"] = {
            "n_bodies": len(bodies),
            "plate_inside": abs(common_vol - plate.Shape.Volume)
                            <= 1e-6 * plate.Shape.Volume,
            "coating_cube": getattr(cube, "coating", None),
            "coating_plate": getattr(plate, "coating", None),
        }
    finally:
        App.closeDocument(doc.Name)

    doc = App.newDocument("probe_anamorphic_pair")
    try:
        bodies = pl.build_primitive(doc, "anamorphic_pair",
                                    group="anamorphic_pair")
        out["anamorphic_pair"] = {
            "n_bodies": len(bodies),
            "gap_mm": _gap(bodies[0], bodies[1]),
            "materials": [getattr(b, "material", None) for b in bodies],
            "placements_identity": [
                (b.Placement.Base.x, b.Placement.Base.y, b.Placement.Base.z)
                == (0.0, 0.0, 0.0) for b in bodies],
        }
    finally:
        App.closeDocument(doc.Name)

    doc = App.newDocument("probe_glan_taylor")
    try:
        bodies = pl.build_primitive(doc, "polarizer_glan_taylor",
                                    group="polarizer_glan_taylor")
        b_in = [b for b in bodies if b.Name.endswith("_in")][0]
        b_out = [b for b in bodies if b.Name.endswith("_out")][0]
        out["polarizer_glan_taylor"] = {
            "n_bodies": len(bodies),
            "gap_mm": _gap(b_in, b_out),
            "materials": [getattr(b, "material", None) for b in bodies],
            "crystal_axes": [getattr(b, "crystal_axis", None)
                            for b in bodies],
        }
    finally:
        App.closeDocument(doc.Name)

    return out


def probe_samples_cells_nested_pairs():
    """cuvette_square/cuvette_capillary/flow_cell/vial_cylindrical/
    vat_cylindrical: two bodies, the WALL is the PRIMARY body (Name ==
    group) and carries the glass-family material, the '<group>_liquid'
    body is nested STRICTLY inside it (common volume == the liquid's own
    volume, the bs_cube/nested4 nested-solids pattern -- no air gap,
    glass-to-liquid contact). sample_region: a single air body."""
    out = {}
    for kind in SAMPLES_CELL_KINDS:
        doc = App.newDocument("probe_cell_" + kind)
        try:
            bodies = pl.build_primitive(doc, kind, group=kind)
            if kind == "sample_region":
                b = bodies[0]
                out[kind] = {"n_bodies": len(bodies),
                            "material": getattr(b, "material", None),
                            "name_is_group": b.Name == kind}
                continue
            wall = [b for b in bodies if b.Name == kind][0]
            liquid = [b for b in bodies if b.Name == kind + "_liquid"][0]
            common_vol = wall.Shape.common(liquid.Shape).Volume
            liquid_vol = liquid.Shape.Volume
            wall_vol = wall.Shape.Volume
            out[kind] = {
                "n_bodies": len(bodies),
                "wall_is_primary": wall.Name == kind,
                "wall_material": getattr(wall, "material", None),
                "liquid_material": getattr(liquid, "material", None),
                "liquid_nested_in_wall": abs(common_vol - liquid_vol)
                                        <= 1e-6 * liquid_vol,
                "liquid_strictly_smaller": liquid_vol < wall_vol,
                "wall_vol_mm3": wall_vol,
                "liquid_vol_mm3": liquid_vol,
            }
        finally:
            App.closeDocument(doc.Name)
    return out


def probe_build_mirror_parabolic_scene(outpath):
    """Build + save a small mirror_parabolic FCStd scene (collimated
    coherent=false source upstream, mirror, a token far-away detector just
    to satisfy the 'model has no detectors' contract check) for the
    engine-level geometric focus check: a real detector plane co-axial
    with the mirror would self-shadow the incoming beam, so the actual
    focus measurement is done in test_primitivelib.py by manually
    injecting rays into the extracted model.json and harvesting the
    reflected bundle -- this probe only needs to produce a valid,
    non-overlapping .FCStd for extract_geometry.py to consume."""
    doc = App.newDocument("mirror_parabolic_focus_scene")
    try:
        pl.make_sheet(doc, "mirror_parabolic", label="dim")
        pl.build_primitive(doc, "mirror_parabolic", group="Mirror")
        rfl = pl.PRIMITIVES["mirror_parabolic"]["params"]["rfl"]["default"]
        aperture = pl.PRIMITIVES["mirror_parabolic"]["params"]["aperture"][
            "default"]
        mts = pl.mts
        mts.new_body_pad(doc, "Source", "Source",
                         circle=(0.0, 0.0, aperture / 2.0 - 1.0),
                         x_start=-150.0, length=1.0,
                         props={"power": 5.0, "lambdac": 633.0,
                                "coherent": False})
        mts.new_body_pad(doc, "Screen", "Screen",
                         rects=[(-10.0, -10.0, 20.0, 20.0)],
                         x_start=200.0, length=1.0,
                         props={"material": "detector"})
        doc.recompute()
        doc.saveAs(str(outpath))
        return {"rfl_mm": rfl, "aperture_mm": aperture, "path": str(outpath)}
    finally:
        App.closeDocument(doc.Name)


def main():
    args = parse_args()
    scene_path = Path(args.out).resolve().parent / \
        "mirror_parabolic_focus_scene.FCStd"
    result = {
        "legacy_fallback": probe_legacy_fallback(),
        "rebuild_roundtrip": probe_rebuild_round_flag_roundtrip(),
        "new_kinds_build_rebuild": probe_new_kinds_build_rebuild(),
        "apertures": probe_apertures(),
        "bladed_iris": probe_bladed_iris_polygon(),
        "edge_blackened": probe_edge_blackened_lens(),
        "corner_cube": probe_corner_cube(),
        "batch3_geometry": probe_batch3_geometry(),
        "samples_cells": probe_samples_cells_nested_pairs(),
        "mirror_parabolic_scene": probe_build_mirror_parabolic_scene(
            scene_path),
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=1)
    os.replace(tmp, args.out)
    log("PROBE OK -> %s" % args.out)


try:
    main()
except BaseException:
    import traceback
    for line in traceback.format_exc().splitlines():
        log(line)
    os._exit(1)
finally:
    os._exit(0)
