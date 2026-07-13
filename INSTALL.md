# Installing MieWorkbench

MieWorkbench is a repo you clone and run in place — there is no packaged
distribution. This document covers a full desktop install (GUI + engine)
on Pop!_OS / Ubuntu 22.04 and 24.04, and a minimal headless-server install
for machines that only need to run jobs, not display anything.

For what each piece does once installed, see [README.md](README.md); for
the engine itself see [docs/RAYTRACER.md](docs/RAYTRACER.md).

---

## 1. What you need

MieWorkbench composes **four independent interpreters**, each pinned by
path in `scripts/common.py` and each overridable via an `MIEWB_*`
environment variable or the GUI's **File → Settings…** dialog (see
README.md §5.13). Nothing here needs to be installed *inside* this repo's
own virtualenv except the GUI's own Python dependencies (§3) — the other
three are external tools you point the workbench at.

| # | Tool | Used by | This machine's pin |
|---|---|---|---|
| 1 | System `python3` (3.10+) | `run_pipeline.py`, `sweep_variants.py`, `miewb_tool.py`, `common.py`, `cli_specs.py` | whatever `python3` resolves to |
| 2 | FreeCAD 1.1+ AppImage | `extract_geometry.py`, `permute_model.py`, `make_test_scenes.py`, `make_primitives.py` | `/home3/freecad/FreeCAD.AppImage` (FreeCAD 1.1.1) |
| 3 | "The optics environment" — a Python env with numpy/scipy/torch(-CUDA)/h5py/miepython/matplotlib | `run_trace.py`, `post_process.py`, `compare_runs.py`, `optimize.py`, `tolerance.py`, `fast_eval.py`, the engine's own pytest suite | `/home3/optics/env/bin/python` |
| 4 | ParaView 5.13+/6.x with `pvpython` | `make_viz.py` | `/home3/paraview/ParaView-6.1.1-MPI-Linux-Python3.12-x86_64/bin/pvpython` (ParaView 6.1.1) |
| 5 | GUI venv (PySide6 + VTK + …) | `mieworkbench`, GUI pytest | `env/` inside this repo |

You do not need all five to do useful work — see §6 for a minimal
headless install (just #1, #2, #3, and optionally #4).

A GPU is **not** required, but an NVIDIA driver + CUDA-capable GPU is
**strongly recommended** for `run_trace.py`: the solver's default backend
is `auto`, which uses `torch` on CUDA when available and falls back to
plain `numpy` otherwise — the numpy path is correct but considerably
slower for anything beyond the `quick` preset.

Base OS packages (both 22.04 and 24.04):

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git
```

The GUI needs a display: an ordinary X11 desktop session or XWayland
works (PySide6/Qt on Wayland-native is not something this project has
been validated against — XWayland is the safe choice). A headless server
with no display at all is fine for the GUI test suite (`QT_QPA_PLATFORM=
offscreen`) and for `miewb_tool.py`, which never touches Qt.

---

## 2. Clone the repo

```bash
git clone <your-remote-url> raytracegui
cd raytracegui
```

`env/`, `var/`, `results/`, `geometry/`, `*.FCBak`, and `__pycache__/` are
gitignored (regenerated locally); everything else — `scripts/`,
`mieworkbench/`, `opticalproperties/`, `primitives/`, `basemodels/`,
`example.FCStd`, `doubleslit.FCStd` — comes from the clone.

---

## 3. External tools

### 3.1 FreeCAD

Download the FreeCAD 1.1+ AppImage (https://www.freecad.org/downloads.php)
and make it executable:

```bash
chmod +x FreeCAD_1.1.1-Linux-x86_64-py311.AppImage
```

Put it wherever you like; you'll point `MIEWB_FREECAD` (or the Settings
dialog) at its path if it isn't at the default
`/home3/freecad/FreeCAD.AppImage`. AppImages need FUSE to run directly
(`sudo apt install libfuse2` on 22.04, `libfuse2t64` or the `--appimage-
extract-and-run` flag on 24.04 where FUSE2 may be absent); if FUSE isn't
available, extract and run: `./FreeCAD*.AppImage --appimage-extract` then
point `MIEWB_FREECAD` at `squashfs-root/AppRun`.

FreeCAD's headless `-c` mode has a few sharp edges (a script runs twice
per invocation, `--help`/argparse `SystemExit` is swallowed, stdin must be
`/dev/null` for one-shot batch calls) — these are pipeline-internal
concerns already handled by the scripts; you shouldn't need to think
about them unless you're modifying `scripts/extract_geometry.py` et al.

### 3.2 The optics environment

The engine's trace/post/compare stages and its pytest suite need numpy,
scipy, a CUDA-capable torch build (optional but recommended), h5py,
miepython, and matplotlib. The **optical-design tools** (`optimize.py`,
`tolerance.py` — merit-function optimization and Monte-Carlo tolerancing)
additionally need **`nevergrad`** (the CMA-ES / directed global optimizer
backend) and **`cma`** (the standalone CMA-ES it wraps); both are pure-Python
wheels with no CUDA dependency. This machine's copy lives at
`/home3/optics/env/bin/python` (Python 3.11, `torch` built for CUDA 13).
To build an equivalent environment elsewhere:

```bash
python3 -m venv /path/to/optics/env
/path/to/optics/env/bin/pip install numpy scipy h5py miepython matplotlib
# optical-design tools (optimization + tolerancing):
/path/to/optics/env/bin/pip install nevergrad cma
# CUDA build (pick the index URL matching your driver's CUDA version;
# see https://pytorch.org/get-started/locally/):
/path/to/optics/env/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU-only fallback (correct, just slower):
/path/to/optics/env/bin/pip install torch
```

`nevergrad`/`cma` are only needed if you run the optimizer or tolerancer
(or their GUI docks); the trace/post/viz pipeline works without them.

Point `MIEWB_OPTICS_PYTHON` (or Settings) at `/path/to/optics/env/bin/python`.
A conda/mamba environment with the same packages works identically —
`common.py` only cares that the path is a working Python interpreter with
those packages importable.

### 3.3 ParaView

Download a ParaView 5.13+ or 6.x binary distribution
(https://www.paraview.org/download/) that bundles `pvpython` — do **not**
rely on a distro package, which typically omits the Python bindings this
project needs. This machine uses ParaView 6.1.1
(`ParaView-6.1.1-MPI-Linux-Python3.12-x86_64`). Extract it anywhere and
point `MIEWB_PVPYTHON` (or Settings) at `<extracted>/bin/pvpython`.

---

## 4. The GUI virtualenv

```bash
cd raytracegui
python3 -m venv env
env/bin/pip install --upgrade pip
env/bin/pip install PySide6 vtk numpy scipy h5py pytest pytest-qt
```

Versions in use on this machine: `PySide6 6.11.1`, `vtk 9.6.2`, `numpy
2.2.6`, `scipy 1.15.3`, `h5py 3.16.0`, `pytest 9.1.1`, `pytest-qt 4.5.0`,
under Python 3.10 (Ubuntu 22.04's system Python). Both PySide6 6.11 and
vtk 9.6 ship prebuilt wheels for Python 3.10 (22.04) and Python 3.12
(24.04) on manylinux — `pip install` picks the right wheel automatically
for whichever `python3` you ran `venv` with, no source build needed on
either release. `numpy`/`scipy`/`h5py` in the GUI venv are a *separate*
copy from the optics environment's (§3.2) — the GUI never imports
`scripts/raytracer/`'s torch-dependent code, only reads/writes plain
files (`model.json`, `.h5` detector cubes, etc.) with these lighter deps.

---

## 5. First run

```bash
# 1. sanity-check the pinned tool paths and a battery of pure-math
#    invariants (path checks will show FAIL until you fix them in step 2)
python3 scripts/common.py

# 2. if any of FreeCAD / optics-python / pvpython aren't at this
#    machine's defaults, either export overrides:
export MIEWB_FREECAD=/path/to/FreeCAD.AppImage
export MIEWB_OPTICS_PYTHON=/path/to/optics/env/bin/python
export MIEWB_PVPYTHON=/path/to/ParaView/bin/pvpython
#    ...or launch the GUI once and fix them under File > Settings... —
#    that dialog persists the same values (via QSettings) and applies
#    them to every pipeline subprocess it launches, so you don't need
#    the exports for GUI use once set there.

# 3. generate the primitive element library if primitives/ is empty
#    (it ships populated in this repo, but regenerate after editing
#    scripts/primitivelib.py — see CUSTOMIZE.md)
/path/to/FreeCAD.AppImage -c scripts/make_primitives.py -- --kind all < /dev/null

# 4. launch the GUI
env/bin/python -m mieworkbench
```

If `bin/mieworkbench` and `share/mieworkbench.desktop` exist in your
checkout, you can use the launcher script directly (it locates the repo
relative to itself and refuses with a clear message if `env/` is
missing):

```bash
bin/mieworkbench [example.FCStd | project.MieWB | results.MieSim]
```

and, optionally, install a desktop entry for your user:

```bash
share/install-desktop.sh
```

which writes `~/.local/share/applications/mieworkbench.desktop` pointing
at this checkout's `bin/mieworkbench`, associating `.FCStd`/`.MieWB`/
`.MieSim` files with it.

---

## 6. Headless-server install (no GUI)

A machine that only needs to run jobs — e.g. the target of an "Export Run
Script" from a workstation, or a CI runner — needs none of the GUI
virtualenv (§4). Just:

- system `python3` (for `run_pipeline.py`/`miewb_tool.py`),
- the FreeCAD AppImage (§3.1),
- the optics environment (§3.2),
- ParaView (§3.3), only if you want `viz/*.png` renders — omit `--steps`
  down to `extract,trace,post` (or accept that the viz stage will fail)
  if you don't have it.

Usage, given a `.MieWB` produced elsewhere (by the GUI's "Export Run
Script", or `miewb_tool.py pack`):

```bash
python3 scripts/miewb_tool.py run project.MieWB -o result.MieSim
```

This unpacks the workbench into an isolated scratch workspace, runs
`run_pipeline.py` there with the packed `simparams.json` translated back
into CLI flags, and packs a self-contained `.MieSim` — no repo-wide state
is touched, and the workspace is cleaned up afterward unless `--keep` is
given. See README.md §4 and §5.9 for the full format/CLI reference.

---

## 7. Verification

```bash
# engine suite (~906 tests currently; grows over time — see the actual
# count with --collect-only -q)
/home3/optics/env/bin/python -m pytest scripts/raytracer/tests/ -q

# GUI suite, headless
QT_QPA_PLATFORM=offscreen env/bin/python -m pytest mieworkbench/tests -q

# + slower FreeCAD integration tests
MIEWB_RUN_FREECAD=1 QT_QPA_PLATFORM=offscreen env/bin/python -m pytest mieworkbench/tests -q

# end-to-end: the shipped example scene through the quick preset
python3 scripts/run_pipeline.py --models example.FCStd --preset quick
```

A clean install should show all pytest suites passing and the pipeline
run completing with a printed summary table and `results/example/quick/`
populated with `case.json`, `audit.json`, `detectors/*.h5`, `images/`,
`spectra/`, `plots/`, and (if ParaView is configured) `viz/`.

## C engine (optional, recommended)

The compiled trace/gather engine gives ~10x on trace-bound scenes and
~6x on coherent scenes (see `cengine/BENCHMARKS.md`). Requirements:
gcc >= 11, cmake >= 3.22, ninja, OpenMP; CUDA 13 for the GPU gather
(`/usr/local/cuda-13`; without it the engine builds CPU-only).

    cd cengine && ./build.sh          # -> cengine/build/miewb-trace
    ./build.sh test                   # C unit tests

Nothing else changes: `--engine auto` (the default) uses it whenever the
scene's features are ported and falls back to the Python engine
otherwise. Set `MIEWB_CENGINE` to override the binary path.
