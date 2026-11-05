# MieWorkbench v1.1.0

FreeCAD 26.3 compatibility, plus the first AppImage build of the GUI with a
bundled command-line interface.

## Highlights

- **AppImage**: one file, `MieWorkbench-1.1.0-x86_64.AppImage`. It bundles
  Python 3.12, PySide6 6.11, VTK 9.6, numpy, scipy, h5py, the library and
  primitive catalogs, the demo galleries and a CPU-only (OpenMP) build of the
  C engine `miewb-trace`. FreeCAD, the optics environment, ParaView and CUDA
  stay external and are configured through `miewb.env`.
- **Bundled CLI**: the same file runs the headless tools:
  `pipeline` (extract, trace, post, viz), `tool` (pack / run `.MieWB` and
  `.MieSim`), `cli <script>` for any `scripts/<name>.py`, `trace`, `python`
  and `setup-env`.
- **FreeCAD 1.1.3 through 26.3** supported (26.3 ships OCCT 8.0 and
  Python 3.13). PartDesign `FuzzyTolerance` is pinned to -1 on every
  Pad/Pocket/Revolution the builders create, because 26.3 changed the
  default from -1 to 0. The FreeCAD worker reports the full release suffix
  (for example `26.3.0RC2`) and refuses spreadsheet aliases that collide with
  unit tokens.
- Verified on FreeCAD 1.1.3 (OCCT 7.9.3) and 26.3rc2 (OCCT 8.0.1): all 80
  primitives rebuild with identical validity, face count, face order, areas
  and volumes; geometry extraction of every demo, basemodel and primitive
  gives identical contracts; demo-equivalence placements pass 27/27.
- Writable state never goes into the read-only AppImage: settings, geometry,
  results and GUI workspaces live under `~/.config/mieworkbench` and
  `~/.local/share/mieworkbench` (new `MIEWB_VAR_DIR` override; a normal
  clone still uses `<repo>/var`).
- `mieworkbench --version` prints `MieWorkbench 1.1.0`.

## Required dependencies

The AppImage does not contain these. Install them separately and point
`miewb.env` at them (`./MieWorkbench-1.1.0-x86_64.AppImage setup-env`, or
File > Settings in the GUI).

| What | Version | miewb.env key |
|---|---|---|
| FreeCAD AppImage | 1.1.3 through 26.3 | `MIEWB_FREECAD` (required) |
| Optics-env Python with numpy, scipy, h5py, miepython, matplotlib | torch optional but recommended; use a CUDA build of torch for GPU gathers | `MIEWB_OPTICS_PYTHON` (required) |
| ParaView `pvpython` | 6.1.1 (optional; without it the viz stage is skipped) | `MIEWB_PVPYTHON` |
| CUDA toolkit (nvcc 13) | optional, only to build the GPU C engine yourself; the bundled engine is CPU-only | `MIEWB_NVCC`, `MIEWB_CUDA_ARCH`, `MIEWB_CENGINE` |

## System requirements

Minimum:

- x86_64 Linux with glibc 2.34 or newer (Ubuntu 22.04, Debian 12, Fedora 35
  and later). The highest glibc symbol version required by any bundled
  binary was measured at 2.34.
- `libfuse2` to run the AppImage directly (Ubuntu 22.04:
  `sudo apt install libfuse2`); without it run with
  `--appimage-extract-and-run`.
- OpenGL / EGL drivers and libraries (`libgl1`, `libegl1`), fontconfig
  (`libfontconfig1`), and an X11 session (XWayland on Wayland desktops).
- A FreeCAD AppImage and an optics-env Python (see above).

Recommended:

- NVIDIA GPU with a CUDA build of torch in the optics env for large
  coherent gathers.
- 16 GB RAM or more and a multi-core CPU (the bundled engine uses OpenMP).
- ParaView 6.1.1 for 3D renders.

## Quick start

```bash
chmod +x MieWorkbench-1.1.0-x86_64.AppImage
./MieWorkbench-1.1.0-x86_64.AppImage --version
./MieWorkbench-1.1.0-x86_64.AppImage setup-env       # probe/write ~/.config/mieworkbench/miewb.env
./MieWorkbench-1.1.0-x86_64.AppImage                 # start the GUI
./MieWorkbench-1.1.0-x86_64.AppImage pipeline --models axicon_pcx.FCStd --preset quick   # bundled basemodel
./MieWorkbench-1.1.0-x86_64.AppImage tool run demo.MieWB -o demo.MieSim
```

If FUSE is unavailable, prefix any command with
`--appimage-extract-and-run`, for example
`./MieWorkbench-1.1.0-x86_64.AppImage --appimage-extract-and-run`.

Build and layout details: `packaging/appimage/README.md`.
