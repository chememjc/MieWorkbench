# MieWorkbench AppImage

`build_appimage.sh` produces `MieWorkbench-<version>-x86_64.AppImage`
(version read from `mieworkbench/__init__.py`): the GUI plus a bundled CLI
in one file.

## What is bundled

- A relocatable Python 3.12 (conda-forge, installed under `AppDir/usr`).
- GUI dependencies pinned in `requirements-gui.txt`: PySide6 6.11.x, vtk
  9.6.x, numpy, scipy, h5py (unused Qt modules such as WebEngine/QML/3D are
  pruned; QtCharts, QtWidgets and QtOpenGL are kept).
- xcb-util libraries and libxkbcommon for Qt's X11 plugin (found through an
  RPATH on the plugin, so no `LD_LIBRARY_PATH` leaks into child processes).
- The repo payload under `usr/share/mieworkbench/`: `mieworkbench/`,
  `scripts/`, `opticalproperties/`, `primitives/`, `basemodels/`, `demos/`,
  `docs/` (without `docs/archive`), `share/`, README/INSTALL/CUSTOMIZE/
  CHANGELOG/LICENSE. Tests and `__pycache__` are left out; bytecode is
  precompiled with unchecked hashes.
- `usr/bin/miewb-trace`: the C engine built CPU-only (OpenMP, no
  `-march=native`, no CUDA) with the conda-forge toolchain, so it needs only
  glibc 2.17. The highest requirement over all bundled binaries is GLIBC 2.34
  (PySide6's `qsb` tool), so the AppImage runs on glibc 2.34 or newer
  (Ubuntu 22.04 and later).

NOT bundled (configure through `miewb.env`): FreeCAD (`MIEWB_FREECAD`), the
optics-env python (`MIEWB_OPTICS_PYTHON`), ParaView `pvpython`
(`MIEWB_PVPYTHON`), CUDA / a GPU C-engine build (`MIEWB_CENGINE`).

## Building

Requirements on the build host: bash, curl, tar, git, binutils (`objdump`),
and either network access to download `appimagetool` or a prebuilt runtime.
Everything else (micromamba, compilers, cmake, ninja, patchelf) is pulled
from conda-forge into the work directory.

```bash
packaging/appimage/build_appimage.sh                 # downloads appimagetool
APPIMAGETOOL=/path/appimagetool-x86_64.AppImage packaging/appimage/build_appimage.sh
RUNTIME_FILE=/path/type2-runtime packaging/appimage/build_appimage.sh   # no appimagetool
```

With `RUNTIME_FILE` and no `APPIMAGETOOL` the image is assembled as
`runtime + mksquashfs(AppDir)` (zstd) instead of calling appimagetool.

Knobs: `WORK_DIR` (default `build/appimage`), `MM_VERSION`, `PYTHON_SPEC`,
`GLIBC_MAX` (default 2.35; the build fails if any bundled ELF needs more),
`SQUASH_COMP`, `KEEP_CACHE=1`, `SKIP_DEMOS=1`. The script rebuilds `AppDir`
from scratch each run. The glibc scan and a smoke test run before packing.

## Using it

```bash
chmod +x MieWorkbench-1.1.0-x86_64.AppImage
./MieWorkbench-1.1.0-x86_64.AppImage                       # GUI
./MieWorkbench-1.1.0-x86_64.AppImage model.MieWB           # GUI, open a file
./MieWorkbench-1.1.0-x86_64.AppImage setup-env             # write miewb.env
./MieWorkbench-1.1.0-x86_64.AppImage pipeline --models x.FCStd --preset quick
./MieWorkbench-1.1.0-x86_64.AppImage tool pack x.FCStd -o x.MieWB
./MieWorkbench-1.1.0-x86_64.AppImage cli compare_runs --help
./MieWorkbench-1.1.0-x86_64.AppImage python -c "import numpy"
./MieWorkbench-1.1.0-x86_64.AppImage trace --help
./MieWorkbench-1.1.0-x86_64.AppImage --version | --help
```

Without FUSE 2 add `--appimage-extract-and-run` (or `--appimage-extract`).
`cli <name>` runs `scripts/<name>.py` with the bundled python; scripts that
belong to the FreeCAD, optics or ParaView stacks are launched by the pipeline
with the external interpreters from `miewb.env` (script paths inside the
read-only mount are fine).

## Runtime layout and environment variables

The mount is read-only, so `AppRun` redirects everything writable. Each
default is applied only if the variable is not exported and has no line in
the env file, so your own configuration always wins.

| Variable | Default | Purpose |
|---|---|---|
| `MIEWB_ENV_FILE` | `${XDG_CONFIG_HOME:-~/.config}/mieworkbench/miewb.env` | machine paths (tools); written by `setup-env` and the Settings dialog |
| `MIEWB_GEOMETRY_DIR` | `${XDG_DATA_HOME:-~/.local/share}/mieworkbench/geometry` | extracted geometry |
| `MIEWB_RESULTS_DIR` | `.../mieworkbench/results` | case results |
| `MIEWB_VAR_DIR` | `.../mieworkbench/var` | GUI workspaces (`work/`), tessellation cache (`cache/`), optimizer/tolerance scratch |
| `MIEWB_CACHE_DIR` | `$MIEWB_VAR_DIR/cache` | tessellation and T-matrix caches |
| `MIEWB_CENGINE` | bundled `usr/bin/miewb-trace` | C engine (set in `miewb.env` to use a GPU build) |
| `MIEWB_GUI_PYTHON` | bundled python | GUI interpreter |

`MIEWB_VAR_DIR` is also honoured by a plain clone (default `<repo>/var`).
`PYTHONDONTWRITEBYTECODE=1` and `PYTHONNOUSERSITE=1` are set. On X11
sessions `QT_QPA_PLATFORM=xcb` is chosen when unset.

## Release

`.github/workflows/release-appimage.yml` builds on ubuntu-22.04, smoke
tests the image and publishes it to the GitHub release for the pushed `v*`
tag (or the `tag` input of a manual run). The notes come from
`RELEASE_NOTES.md`.
