#!/usr/bin/env python3
# =============================================================================
# miewb_tool.py — .MieWB / .MieSim archive engine + headless CLI.
#
# Interpreter: plain system python3 (stdlib only, like common.py). This is
# BOTH the importable format library used by the MieWorkbench GUI
# (mieworkbench.core.mieformats wraps it) and the standalone tool a headless
# / remote box uses with nothing but a repo clone.
#
# Formats (both ZIP containers; already-compressed members are STORED so we
# never double-deflate):
#
#   X.MieWB — a portable workbench: everything needed to run the simulation.
#       manifest.json        {"format":"MieWB","version":1,"created":...,
#                             "app":..., "fcstd":"model.FCStd"}
#       model.FCStd          the scene            (STORED — .FCStd is a zip)
#       opticalproperties/** the project property library (CSV-content)
#       simparams.json       config-matrix state -> run_pipeline args
#       project.json         GUI/session metadata (optional)
#
#   X.MieSim — results of one run, self-describing and re-runnable:
#       manifest.json        {"format":"MieSim","version":1,"created":...,
#                             "source_miewb":..., "model":<stem>,
#                             "case":<case>, "status":...}
#       input.MieWB          the EXACT workbench used for the run (STORED)
#       geometry/<stem>/**   extracted contract (model.json + face STLs)
#       results/<stem>/<case>/**   everything the pipeline wrote
#
# A .MieSim opened without re-running is read-only except "save as .MieWB"
# (extract input.MieWB). Re-running REPLACES input.MieWB and the result
# members. sniff() distinguishes the two by manifest, not extension.
#
# CLI:
#   miewb_tool.py pack   <model.FCStd> -o X.MieWB [--optical-properties DIR]
#                        [--simparams FILE.json]
#   miewb_tool.py unpack <X.MieWB|X.MieSim> -d WORKDIR
#   miewb_tool.py info   <X.MieWB|X.MieSim>
#   miewb_tool.py run    <X.MieWB> -o X.MieSim [--workdir DIR] [--keep]
#                        [-- <extra run_pipeline args>]
#   miewb_tool.py pack-sim -d WORKDIR -o X.MieSim --miewb X.MieWB
#                        [--purge-intermediates]
# =============================================================================
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import common  # noqa: E402

FORMAT_MIEWB = "MieWB"
FORMAT_MIESIM = "MieSim"
FORMAT_VERSION = 1
APP_ID = "MieWorkbench"

# members that are already compressed: store, don't deflate
_STORED_SUFFIXES = {".fcstd", ".miewb", ".miesim", ".h5", ".png", ".jpg",
                    ".gz", ".zip"}


class MieFormatError(ValueError):
    pass


def _compress_type(name):
    return (zipfile.ZIP_STORED
            if Path(name).suffix.lower() in _STORED_SUFFIXES
            else zipfile.ZIP_DEFLATED)


def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ---------------------------------------------------------------------------
# generic zip helpers
# ---------------------------------------------------------------------------
def _add_file(zf, src, arcname):
    zf.write(str(src), arcname, compress_type=_compress_type(arcname))


def _add_tree(zf, root, arc_prefix, exclude_names=()):
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if path.name in exclude_names:
            continue
        rel = path.relative_to(root).as_posix()
        _add_file(zf, path, "%s/%s" % (arc_prefix, rel))


def _write_manifest(zf, manifest):
    zf.writestr("manifest.json",
                json.dumps(manifest, indent=1, sort_keys=True),
                compress_type=zipfile.ZIP_DEFLATED)


def read_manifest(path):
    """Manifest of a .MieWB/.MieSim without unpacking anything else."""
    try:
        with zipfile.ZipFile(str(path)) as zf:
            with zf.open("manifest.json") as fh:
                manifest = json.load(fh)
    except (zipfile.BadZipFile, KeyError, ValueError, OSError) as exc:
        raise MieFormatError("%s: not a MieWB/MieSim archive (%s)"
                             % (path, exc))
    fmt = manifest.get("format")
    if fmt not in (FORMAT_MIEWB, FORMAT_MIESIM):
        raise MieFormatError("%s: unknown format %r" % (path, fmt))
    if int(manifest.get("version", 0)) > FORMAT_VERSION:
        raise MieFormatError(
            "%s: format version %s is newer than this tool understands "
            "(%s); update the software" % (path, manifest.get("version"),
                                           FORMAT_VERSION))
    return manifest


def sniff(path):
    """'MieWB' | 'MieSim' | 'FCStd' | None for an arbitrary file."""
    try:
        with zipfile.ZipFile(str(path)) as zf:
            names = set(zf.namelist())
    except (zipfile.BadZipFile, OSError):
        return None
    if "manifest.json" in names:
        try:
            return read_manifest(path)["format"]
        except MieFormatError:
            return None
    if "Document.xml" in names:
        return "FCStd"
    return None


def _atomic_replace(tmp_path, final_path):
    os.replace(str(tmp_path), str(final_path))


# ---------------------------------------------------------------------------
# .MieWB
# ---------------------------------------------------------------------------
def pack_miewb(fcstd_path, out_path, optprops_dir=None, simparams=None,
               project_meta=None):
    """Create X.MieWB. simparams: dict or path to a JSON file."""
    fcstd_path = Path(fcstd_path)
    if not fcstd_path.is_file():
        raise MieFormatError("no such model: %s" % fcstd_path)
    optprops_dir = Path(optprops_dir) if optprops_dir else common.OPTPROPS_DIR
    if not optprops_dir.is_dir():
        raise MieFormatError("no such property library: %s" % optprops_dir)
    if isinstance(simparams, (str, Path)):
        with open(simparams) as fh:
            simparams = json.load(fh)
    simparams = simparams or {}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    manifest = {"format": FORMAT_MIEWB, "version": FORMAT_VERSION,
                "app": APP_ID, "created": _now_iso(),
                "fcstd": "model.FCStd",
                "model_stem": fcstd_path.stem}
    with zipfile.ZipFile(str(tmp), "w") as zf:
        _write_manifest(zf, manifest)
        _add_file(zf, fcstd_path, "model.FCStd")
        _add_tree(zf, optprops_dir, "opticalproperties")
        zf.writestr("simparams.json",
                    json.dumps(simparams, indent=1, sort_keys=True))
        if project_meta:
            zf.writestr("project.json",
                        json.dumps(project_meta, indent=1, sort_keys=True))
    _atomic_replace(tmp, out_path)
    return manifest


def unpack(archive_path, workdir):
    """Explode a .MieWB or .MieSim into workdir; returns its manifest."""
    manifest = read_manifest(archive_path)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(archive_path)) as zf:
        for info in zf.infolist():
            target = workdir / info.filename
            # zip-slip guard
            if not str(target.resolve()).startswith(
                    str(workdir.resolve()) + os.sep):
                raise MieFormatError("archive member escapes workdir: %s"
                                     % info.filename)
        zf.extractall(str(workdir))
    return manifest


# ---------------------------------------------------------------------------
# .MieSim
# ---------------------------------------------------------------------------
# purge patterns for --purge-intermediates: visual/diagnostic bulk that can
# be regenerated from the kept trace outputs (detectors/*.h5 + case.json)
_PURGE_GLOBS = ["results/*/*/rays.npy", "results/*/*/viz/*",
                "results/*/*/log.*", "geometry/*/faces/*.stl"]


def pack_miesim(workdir, out_path, miewb_path, model_stem=None, case=None,
                purge_intermediates=False, run_meta=None):
    """Pack a run workspace (geometry/ + results/) into X.MieSim, embedding
    the exact .MieWB the run used."""
    workdir = Path(workdir)
    miewb_path = Path(miewb_path)
    if not miewb_path.is_file():
        raise MieFormatError("no such workbench: %s" % miewb_path)
    geo_root = workdir / "geometry"
    res_root = workdir / "results"
    if not res_root.is_dir():
        raise MieFormatError("workspace has no results/: %s" % workdir)

    stems = sorted(p.name for p in geo_root.iterdir()) \
        if geo_root.is_dir() else []
    if model_stem is None:
        if len(stems) != 1:
            raise MieFormatError(
                "cannot infer model stem (geometry/ has %s); pass "
                "model_stem" % (stems or "nothing"))
        model_stem = stems[0]
    if case is None:
        cases = sorted(p.name for p in (res_root / model_stem).iterdir()
                       if p.is_dir()) \
            if (res_root / model_stem).is_dir() else []
        if len(cases) != 1:
            raise MieFormatError(
                "cannot infer case (results/%s has %s); pass case"
                % (model_stem, cases or "nothing"))
        case = cases[0]

    case_dir = res_root / model_stem / case
    status = common.read_case_status(case_dir / "case.json")

    purged = []
    def _is_purged(rel_posix):
        if not purge_intermediates:
            return False
        from fnmatch import fnmatch
        for pat in _PURGE_GLOBS:
            if fnmatch(rel_posix, pat):
                purged.append(rel_posix)
                return True
        return False

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    manifest = {"format": FORMAT_MIESIM, "version": FORMAT_VERSION,
                "app": APP_ID, "created": _now_iso(),
                "source_miewb": miewb_path.name,
                "model": model_stem, "case": case, "status": status,
                "purged_intermediates": bool(purge_intermediates),
                "run_meta": run_meta or {}}
    with zipfile.ZipFile(str(tmp), "w") as zf:
        _write_manifest(zf, manifest)
        _add_file(zf, miewb_path, "input.MieWB")
        for root, prefix in ((geo_root / model_stem,
                              "geometry/%s" % model_stem),
                             (case_dir, "results/%s/%s"
                              % (model_stem, case))):
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_dir():
                    continue
                if path.name in (".lock.json",):
                    continue
                rel = "%s/%s" % (prefix,
                                 path.relative_to(root).as_posix())
                if _is_purged(rel):
                    continue
                _add_file(zf, path, rel)
    _atomic_replace(tmp, out_path)
    if purged:
        print("[pack-sim] purged %d intermediate file(s)" % len(purged),
              flush=True)
    return manifest


def extract_embedded_miewb(miesim_path, out_path):
    """Pull input.MieWB out of a .MieSim (the 'save as .MieWB' path)."""
    manifest = read_manifest(miesim_path)
    if manifest["format"] != FORMAT_MIESIM:
        raise MieFormatError("%s is not a .MieSim" % miesim_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(miesim_path)) as zf:
        with zf.open("input.MieWB") as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return out_path


# ---------------------------------------------------------------------------
# run: unpack -> pipeline -> pack-sim (the headless path)
# ---------------------------------------------------------------------------
def simparams_to_args(simparams):
    """simparams.json dict -> run_pipeline argv fragment. Keys are argparse
    dests (underscores); values: bool -> bare flag, list -> repeated."""
    args = []
    for dest, value in sorted((simparams or {}).items()):
        flag = "--" + dest.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, (list, tuple)):
            for v in value:
                args += [flag, str(v)]
        else:
            args += [flag, str(value)]
    return args


def run_miewb(miewb_path, out_miesim, workdir=None, extra_args=None,
              keep_workdir=False, python=None):
    """Unpack X.MieWB, run the full pipeline in an isolated workspace, and
    pack X.MieSim. Returns (exit_code, manifest_or_None)."""
    miewb_path = Path(miewb_path).resolve()
    manifest = read_manifest(miewb_path)
    if manifest["format"] != FORMAT_MIEWB:
        raise MieFormatError("%s is not a .MieWB" % miewb_path)

    owns_workdir = workdir is None
    workdir = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="miewb-run-",
                         dir=str(common.PROJECT_DIR / "var" / "work")
                         if (common.PROJECT_DIR / "var").is_dir() else None))
    workdir.mkdir(parents=True, exist_ok=True)
    unpack(miewb_path, workdir)

    model = workdir / manifest.get("fcstd", "model.FCStd")
    stem = manifest.get("model_stem") or model.stem
    # the model must carry its stem as filename so geometry/results land
    # under a meaningful name
    named_model = workdir / ("%s.FCStd" % stem)
    if not named_model.exists():
        shutil.copy2(model, named_model)

    with open(workdir / "simparams.json") as fh:
        simparams = json.load(fh)
    args = simparams_to_args(simparams) + list(extra_args or [])

    env = dict(os.environ)
    env["MIEWB_GEOMETRY_DIR"] = str(workdir / "geometry")
    env["MIEWB_RESULTS_DIR"] = str(workdir / "results")
    env.setdefault("MIEWB_PROGRESS", "1")
    cmd = [python or sys.executable or "python3",
           str(SCRIPTS_DIR / "run_pipeline.py"),
           "--models", str(named_model),
           "--optical-properties", str(workdir / "opticalproperties")]
    cmd += args
    print("[run] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        print("[run] pipeline FAILED (exit %d); workspace kept at %s"
              % (rc, workdir), flush=True)
        return rc, None

    sim_manifest = pack_miesim(workdir, out_miesim, miewb_path,
                               model_stem=stem,
                               run_meta={"pipeline_args": args})
    if owns_workdir and not keep_workdir:
        shutil.rmtree(str(workdir), ignore_errors=True)
    print("[run] wrote %s" % out_miesim, flush=True)
    return 0, sim_manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]

    p = argparse.ArgumentParser(
        description=".MieWB / .MieSim pack, unpack and headless run tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pack", help="model.FCStd -> X.MieWB")
    sp.add_argument("model")
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--optical-properties", default=None)
    sp.add_argument("--simparams", default=None,
                    help="JSON file of run_pipeline option values")

    sp = sub.add_parser("unpack", help="explode an archive into a directory")
    sp.add_argument("archive")
    sp.add_argument("-d", "--dest", required=True)

    sp = sub.add_parser("info", help="print an archive's manifest")
    sp.add_argument("archive")

    sp = sub.add_parser("run", help="X.MieWB -> run pipeline -> X.MieSim "
                        "(extra run_pipeline args after --)")
    sp.add_argument("miewb")
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--workdir", default=None)
    sp.add_argument("--keep", action="store_true",
                    help="keep the workspace directory")

    sp = sub.add_parser("pack-sim", help="pack a run workspace into X.MieSim")
    sp.add_argument("-d", "--workdir", required=True)
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--miewb", required=True,
                    help="the .MieWB this run came from (embedded)")
    sp.add_argument("--model-stem", default=None)
    sp.add_argument("--case", default=None)
    sp.add_argument("--purge-intermediates", action="store_true",
                    help="drop rays.npy/viz/logs/face STLs from the archive")

    sp = sub.add_parser("extract-miewb",
                        help="pull input.MieWB out of a .MieSim")
    sp.add_argument("miesim")
    sp.add_argument("-o", "--out", required=True)

    args = p.parse_args(argv)
    if args.cmd == "pack":
        manifest = pack_miewb(args.model, args.out,
                              optprops_dir=args.optical_properties,
                              simparams=args.simparams)
        print(json.dumps(manifest, indent=1, sort_keys=True))
    elif args.cmd == "unpack":
        manifest = unpack(args.archive, args.dest)
        print("unpacked %s (%s) -> %s"
              % (args.archive, manifest["format"], args.dest))
    elif args.cmd == "info":
        print(json.dumps(read_manifest(args.archive), indent=1,
                         sort_keys=True))
    elif args.cmd == "run":
        rc, _ = run_miewb(args.miewb, args.out, workdir=args.workdir,
                          extra_args=extra, keep_workdir=args.keep)
        return rc
    elif args.cmd == "pack-sim":
        manifest = pack_miesim(args.workdir, args.out, args.miewb,
                               model_stem=args.model_stem, case=args.case,
                               purge_intermediates=args.purge_intermediates)
        print(json.dumps(manifest, indent=1, sort_keys=True))
    elif args.cmd == "extract-miewb":
        out = extract_embedded_miewb(args.miesim, args.out)
        print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
