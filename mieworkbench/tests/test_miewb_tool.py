"""miewb_tool format-engine tests (stdlib logic; no Qt, no FreeCAD).

The pack side needs only files on disk, so everything runs against tmp
fixtures; the real pipeline run path is covered by the phase-8 e2e outside
pytest (it takes ~15 min on GPU).
"""

import json
import os
import sys
import zipfile

import pytest

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import miewb_tool  # noqa: E402


@pytest.fixture()
def fake_library(tmp_path):
    lib = tmp_path / "optprops"
    (lib / "nk").mkdir(parents=True)
    (lib / "materials.miemat").write_text(
        "name,class,model,p1,reference\nglass,dielectric,constant,1.5,x\n")
    (lib / "nk" / "gold.mienk").write_text(
        "wavelength_nm,n,k\n400,1.6,1.9\n700,0.1,3.5\n")
    return lib


@pytest.fixture()
def fake_model(tmp_path):
    # a .FCStd is a zip; make a minimal one so sniff() recognizes it
    model = tmp_path / "scene.FCStd"
    with zipfile.ZipFile(model, "w") as zf:
        zf.writestr("Document.xml", "<Document/>")
    return model


def test_pack_unpack_roundtrip(tmp_path, fake_model, fake_library):
    wb = tmp_path / "scene.MieWB"
    manifest = miewb_tool.pack_miewb(
        fake_model, wb, optprops_dir=fake_library,
        simparams={"preset": "quick"}, project_meta={"note": "hi"})
    assert manifest["format"] == "MieWB"
    assert manifest["model_stem"] == "scene"

    out = tmp_path / "ws"
    m2 = miewb_tool.unpack(wb, out)
    assert m2 == miewb_tool.read_manifest(wb)
    assert (out / "model.FCStd").is_file()
    assert (out / "opticalproperties" / "materials.miemat").is_file()
    assert (out / "opticalproperties" / "nk" / "gold.mienk").is_file()
    assert json.loads((out / "simparams.json").read_text()) == {
        "preset": "quick"}
    assert json.loads((out / "project.json").read_text()) == {"note": "hi"}


def test_stored_vs_deflated_members(tmp_path, fake_model, fake_library):
    wb = tmp_path / "scene.MieWB"
    miewb_tool.pack_miewb(fake_model, wb, optprops_dir=fake_library)
    with zipfile.ZipFile(wb) as zf:
        by_name = {i.filename: i.compress_type for i in zf.infolist()}
    assert by_name["model.FCStd"] == zipfile.ZIP_STORED
    assert by_name["opticalproperties/materials.miemat"] == \
        zipfile.ZIP_DEFLATED


def test_sniff(tmp_path, fake_model, fake_library):
    wb = tmp_path / "scene.MieWB"
    miewb_tool.pack_miewb(fake_model, wb, optprops_dir=fake_library)
    assert miewb_tool.sniff(wb) == "MieWB"
    assert miewb_tool.sniff(fake_model) == "FCStd"
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"\x00" * 64)
    assert miewb_tool.sniff(junk) is None


def test_version_too_new_rejected(tmp_path):
    arc = tmp_path / "future.MieWB"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.writestr("manifest.json",
                    json.dumps({"format": "MieWB", "version": 99}))
    with pytest.raises(miewb_tool.MieFormatError, match="newer"):
        miewb_tool.read_manifest(arc)


def test_corrupt_archive_rejected(tmp_path):
    bad = tmp_path / "bad.MieWB"
    bad.write_bytes(b"not a zip at all")
    with pytest.raises(miewb_tool.MieFormatError, match="not a"):
        miewb_tool.read_manifest(bad)


def test_zip_slip_guard(tmp_path):
    evil = tmp_path / "evil.MieSim"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("manifest.json",
                    json.dumps({"format": "MieSim", "version": 1}))
        zf.writestr("../escape.txt", "gotcha")
    with pytest.raises(miewb_tool.MieFormatError, match="escapes"):
        miewb_tool.unpack(evil, tmp_path / "ws")
    assert not (tmp_path / "escape.txt").exists()


def make_workspace(tmp_path, status="completed", with_viz=True):
    ws = tmp_path / "ws"
    case = ws / "results" / "m" / "quick"
    (case / "detectors").mkdir(parents=True)
    (case / "detectors" / "d.h5").write_bytes(b"\x89HDF")
    (case / "case.json").write_text(json.dumps({"status": status}))
    (case / "rays.npy").write_bytes(b"\x93NUMPY")
    (case / "log.trace").write_text("...")
    if with_viz:
        (case / "viz").mkdir()
        (case / "viz" / "rays.vtp").write_text("<VTKFile/>")
    geo = ws / "geometry" / "m"
    (geo / "faces").mkdir(parents=True)
    (geo / "model.json").write_text("{}")
    (geo / "faces" / "B.Pad.Face1.stl").write_bytes(b"\x00" * 84)
    return ws


def test_pack_miesim_and_extract_embedded(tmp_path, fake_model,
                                          fake_library):
    wb = tmp_path / "scene.MieWB"
    miewb_tool.pack_miewb(fake_model, wb, optprops_dir=fake_library)
    ws = make_workspace(tmp_path)
    sim = tmp_path / "scene.MieSim"
    manifest = miewb_tool.pack_miesim(ws, sim, wb)
    assert manifest["format"] == "MieSim"
    assert manifest["model"] == "m" and manifest["case"] == "quick"
    assert manifest["status"] == "completed"
    with zipfile.ZipFile(sim) as zf:
        names = set(zf.namelist())
    assert "input.MieWB" in names
    assert "results/m/quick/detectors/d.h5" in names
    assert "geometry/m/model.json" in names

    out_wb = tmp_path / "again.MieWB"
    miewb_tool.extract_embedded_miewb(sim, out_wb)
    assert out_wb.read_bytes() == wb.read_bytes()


def test_pack_miesim_purge(tmp_path, fake_model, fake_library):
    wb = tmp_path / "scene.MieWB"
    miewb_tool.pack_miewb(fake_model, wb, optprops_dir=fake_library)
    ws = make_workspace(tmp_path)
    sim = tmp_path / "purged.MieSim"
    manifest = miewb_tool.pack_miesim(ws, sim, wb, purge_intermediates=True)
    assert manifest["purged_intermediates"] is True
    with zipfile.ZipFile(sim) as zf:
        names = set(zf.namelist())
    # kept: the physics
    assert "results/m/quick/detectors/d.h5" in names
    assert "results/m/quick/case.json" in names
    assert "geometry/m/model.json" in names
    # purged: regenerable visuals / logs / face meshes
    assert "results/m/quick/rays.npy" not in names
    assert "results/m/quick/viz/rays.vtp" not in names
    assert "results/m/quick/log.trace" not in names
    assert "geometry/m/faces/B.Pad.Face1.stl" not in names


def test_simparams_to_args():
    args = miewb_tool.simparams_to_args(
        {"preset": "quick", "dry_run": True, "keep_going": False,
         "seeds": 3, "grating": ["a:600:v", "b:300:h"]})
    assert args == ["--dry-run", "--grating", "a:600:v",
                    "--grating", "b:300:h", "--preset", "quick",
                    "--seeds", "3"]
