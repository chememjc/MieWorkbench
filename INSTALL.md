# Installing MieWorkbench

MieWorkbench is a repo you clone and run in place — there is no packaged
distribution. This document covers a full desktop install (GUI + engine)
on Pop!_OS / Ubuntu 22.04 and 24.04, and a minimal headless-server install
for machines that only need to run jobs, not display anything.

For what each piece does once installed, see [README.md](README.md); for
the engine itself see [docs/RAYTRACER.md](docs/RAYTRACER.md).

---

## 1. What you need

MieWorkbench composes **four independent interpreters**. Their paths are
recorded once, per machine, in gitignored `<repo>/miewb.env` — written by
`scripts/setup_env.sh` (§5) — which `scripts/common.py` reads at import
time; an exported `MIEWB_*` environment variable always overrides the
miewb.env entry, and the GUI's **File → Settings…** dialog (see README.md
§5.13) edits the same file in place. Nothing here needs to be installed
*inside* this repo's own virtualenv except the GUI's own Python
dependencies (§3) — the other three are external tools you point the
workbench at.

| # | Tool | Used by | miewb.env key |
|---|---|---|---|
| 1 | System `python3` (3.10+) | `run_pipeline.py`, `sweep_variants.py`, `miewb_tool.py`, `common.py`, `cli_specs.py` | *(none — whatever `python3` resolves to)* |
| 2 | FreeCAD 1.1+ AppImage | `extract_geometry.py`, `permute_model.py`, `make_test_scenes.py`, `make_primitives.py` | `MIEWB_FREECAD` (required) |
| 3 | "The optics environment" — a Python env with numpy/scipy/torch(-CUDA)/h5py/miepython/matplotlib | `run_trace.py`, `post_process.py`, `compare_runs.py`, `optimize.py`, `tolerance.py`, `fast_eval.py`, the engine's own pytest suite | `MIEWB_OPTICS_PYTHON` (required) |
| 4 | ParaView 5.13+/6.x with `pvpython` | `make_viz.py` | `MIEWB_PVPYTHON` (required key, empty value allowed — "no ParaView here") |
| 5 | GUI venv (PySide6 + VTK + …) | `mieworkbench`, GUI pytest | *(none — defaults to `env/` inside this repo; `MIEWB_GUI_PYTHON` overrides)* |

You do not need all five to do useful work — see §7 for a minimal
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
git clone https://github.com/chememjc/MieWorkbench.git
cd MieWorkbench
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

Put it wherever you like — record its path as `MIEWB_FREECAD` in step §5
(`scripts/setup_env.sh`). AppImages need FUSE to run directly
(`sudo apt install libfuse2` on 22.04, `libfuse2t64` or the `--appimage-
extract-and-run` flag on 24.04 where FUSE2 may be absent); if FUSE isn't
available, extract and run: `./FreeCAD*.AppImage --appimage-extract` and
record `squashfs-root/AppRun` as `MIEWB_FREECAD` instead.

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
wheels with no CUDA dependency. To build the environment:

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

Record `/path/to/optics/env/bin/python` as `MIEWB_OPTICS_PYTHON` in step
§5 (`scripts/setup_env.sh`). A conda/mamba environment with the same
packages works identically — `common.py` only cares that the path is a
working Python interpreter with those packages importable.

### 3.3 ParaView

Download a ParaView 5.13+ or 6.x binary distribution
(https://www.paraview.org/download/) that bundles `pvpython` — do **not**
rely on a distro package, which typically omits the Python bindings this
project needs. Extract it anywhere (e.g.
`ParaView-6.1.1-MPI-Linux-Python3.12-x86_64`) and record
`<extracted>/bin/pvpython` as `MIEWB_PVPYTHON` in step §5
(`scripts/setup_env.sh`) — or leave it configured *empty* if this machine
has no ParaView; the viz stage then skips cleanly with a NOTICE instead of
failing.

When invoking `pvpython` directly for `scripts/make_viz.py` (outside the
GUI, which always adds this flag itself), pass
`--force-offscreen-rendering` — see the module docstring:

```bash
/path/to/ParaView/bin/pvpython \
    --force-offscreen-rendering scripts/make_viz.py \
    --case-dir results/example/quick --model-json geometry/example/model.json
```

### 3.4 Optional: pytmatrix (T-matrix spheroid samples, samples-instruments round)

`scripts/raytracer/tmatrix.py`'s `TMatrixEvaluator` (orientation-averaged
T-matrix scattering for non-spherical, `shape=spheroid` sample-registry
rows, docs/RAYTRACER.md §5.13) is a **soft, optics-env-only** dependency
on [pytmatrix](https://github.com/jleinonen/pytmatrix) (MIT) — a
`sphere`-shape sample row never imports it, and every other feature in the
repo works without it. Install it into the optics environment (§3.2)
ONLY — never the GUI venv, never FreeCAD's python.

**pytmatrix 0.3.3's own `setup.py` no longer builds under numpy 2.x** (its
`numpy.distutils`-based Fortran extension build predates numpy's
`distutils` removal), so a bare `pip install pytmatrix==0.3.3` fails on
this machine's numpy 2.x optics env. The verified, REQUIRED recipe builds
the Fortran extension by hand with `numpy.f2py`'s meson backend and drops
the built package straight into `site-packages`:

```bash
OPTICS=/path/to/optics/env   # substitute your optics env root; if you've
                              # already sourced scripts/miewb_env.sh, this
                              # is $(dirname "$(dirname "$MIEWB_OPTICS_PYTHON")")

# 1. pytmatrix's PyPI sdist (0.3.3) — has the Fortran sources but is
#    MISSING pytmatrix.pyf (the f2py interface file); fetch that
#    separately from GitHub's master branch.
mkdir -p /tmp/pytmatrix_build && cd /tmp/pytmatrix_build
$OPTICS/bin/pip download --no-binary :all: --no-deps -d . pytmatrix==0.3.3
tar xzf pytmatrix-0.3.3.tar.gz
curl -L -o pytmatrix-0.3.3/pytmatrix/fortran_tm/pytmatrix.pyf \
    https://raw.githubusercontent.com/jleinonen/pytmatrix/master/pytmatrix/fortran_tm/pytmatrix.pyf

# 2. meson is f2py's build backend here — install it into the optics env
$OPTICS/bin/pip install meson

# 3. build the extension IN PLACE (the env's own bin/ must be FIRST on
#    PATH so f2py's meson backend picks up the right python/meson/ninja;
#    FFLAGS points gfortran at ampld.par.f's shared parameter block)
cd pytmatrix-0.3.3/pytmatrix/fortran_tm
PATH="$OPTICS/bin:$PATH" FFLAGS="-I$(pwd)" \
    $OPTICS/bin/python -m numpy.f2py -c pytmatrix.pyf \
    ampld.lp.f lpd.f --backend meson

# 4. copy the whole package (pure-python modules + the built .so) into
#    site-packages — there is no working `pip install .` path here, this
#    IS the install step
cp -r /tmp/pytmatrix_build/pytmatrix-0.3.3/pytmatrix \
    "$OPTICS"/lib/python3.11/site-packages/
```

(Adjust the `python3.11` site-packages path to your optics env's actual
Python version.) **Verify** with a smoke test that builds a `Scatterer`
and reads back its S-matrix:

```bash
$OPTICS/bin/python -c "
from pytmatrix.tmatrix import Scatterer
s = Scatterer(radius=1.0, wavelength=0.6328, m=complex(1.5, 0.0), axis_ratio=1.0)
print(s.get_S())
"
```

A clean run (no traceback, a 2×2 complex S-matrix printed) confirms the
build. `raytracer/tmatrix.py` soft-imports pytmatrix and only raises
(naming this exact install path) the first time a `shape=spheroid` sample
row is actually evaluated — `sphere`-shape rows, and every other feature,
are completely unaffected if this step is skipped.

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

### 4.1 Optional: the prysm oracle (Forbes Q-type surface tests only)

`mieworkbench/tests/test_qforbes_prysm_oracle.py` checks
`raytracer.surfaces.QForbes` (ISO 10110-12 Qbfs/Qcon aspheres) against
[prysm](https://github.com/brandondube/prysm)'s own Forbes-polynomial
implementation to 1e-12. prysm is MIT-licensed but not on a normal release
cadence (PyPI is stale), so it is installed from a **pinned git SHA** into
this same `env/` venv — never into the optics env, and never imported
by the engine itself (test-only oracle dependency):

```bash
env/bin/pip install \
  "git+https://github.com/brandondube/prysm@f8d72fb66f1c1e5858abdd3f4685805ef319d97b"
```

That SHA is NOT prysm's tip-of-master — the two commits after it
(`eb52449`, `26a4209`) ship a `prysm.x.raytracing` package whose own
`__init__.py` imports modules that were never committed upstream, so
`import prysm.x.raytracing.sags` (what the oracle test needs) raises
`ModuleNotFoundError` at either. `f8d72fb` is the newest commit confirmed
(2026-07-16) to import cleanly. Skip this step entirely if you don't need
that one test file — `pytest.importorskip("prysm")` at its top makes it a
no-op skip, not a failure, when prysm isn't installed.

### 4.2 Optional: the Optiland parity oracle (P4a)

`mieworkbench/tests/test_optiland_oracle.py` is the **P4a parity oracle**
(docs/archive/engine3.md §5 / §15 P4a): it cross-checks the MieWorkbench C ray-tracer
against [Optiland](https://github.com/optiland/optiland) (MIT), an
independent sequential ray tracer, on the shared `geometry/` scenes —
best-focus position, per-ray landing across the pupil fan, and spot RMS
all agree to floating-point round-off. This arbiter must exist **before**
any Optiland-based optimizer/designer (P4b) adds a second physics truth.

Optiland is a normal PyPI package, installed into the SAME `env/` GUI venv
— never into the optics env, and never imported by the engine itself
(the bridge `scripts/raytracer/optiland_bridge.py` is the ONLY module that
imports it, and it runs under `env/bin/python`):

```bash
env/bin/pip install "optiland==0.6.0"
```

**PINNED VERSION: `optiland==0.6.0`** — the newest release that supports
Python 3.10 (the GUI venv's interpreter). 0.6.1 (and later) require
Python ≥ 3.11 and will not install here; the resolver would silently fall
back to an ancient release, so pin explicitly. It pulls
numba/pandas/pyyaml/seaborn/tabulate into `env/` (no torch — that is an
optional Optiland extra we do not install).

The oracle cases additionally shell out to the optics-env C engine to
generate the Monte-Carlo comparison bundle (`run_trace.py --export-rays`);
if the optics env or `cengine/build/miewb-trace` is absent they skip.
The bridge structural + unit-contract tests need only Optiland. Skip the
whole file with `pytest.importorskip("optiland")` when Optiland isn't
installed. The two adjudications that reconciled the engines to machine
precision (the real-air ambient index n=1.000272, and Optiland's
reported-image-direction convention) are recorded in that test's module
docstring.

### 4.3 Optional: the meent RCWA table generator (P6 grating tables only)

`scripts/tools/gen_rcwa_table.py` generates v2 RCWA grating tables (complex
per-order amplitudes over a `(lambda, theta, phi)` grid; docs/archive/engine3.md §7.5,
docs/RAYTRACER.md §5.5/§7.5) using [meent](https://github.com/kc-ml2/meent)
(MIT), a rigorous coupled-wave (RCWA) solver. Like the oracles above it is
installed into the SAME `env/` GUI venv — never into the optics env —
and is **never imported by the engine**: the generated `.mietab` is committed
and the tracer only interpolates it. It is a generation-time (authoring)
dependency:

```bash
env/bin/pip install "meent==0.12.0"
```

**PINNED VERSION: `meent==0.12.0`** — the newest release, compatible with the
GUI venv's Python 3.10 and numpy 2.x (its only hard deps are `numpy>=1.23.3`
and `scipy>=1.9.1`; the `jax`/`pytorch` extras are backends we do not
install — the default numpy backend is used). Its factorization is the Li
(1996, JOSA A 13:1870) inverse rule (verified from source). The adoption gate
that qualified meent before any table shipped is
`scripts/tools/rcwa_adoption_gate.py` (Li-rule convergence, energy
conservation, reciprocity); the closed-form cross-check against the engine's
Kogelnik VBG branch is `scripts/tools/rcwa_kogelnik_crosscheck.py`. Skip this
step entirely unless you are re-generating grating tables.

---

## 5. Configure machine paths (`scripts/setup_env.sh`)

Machine-specific tool paths live in ONE gitignored file,
`<repo>/miewb.env` — never in source. `scripts/setup_env.sh` creates it;
run it once per machine, after §3/§4:

```bash
scripts/setup_env.sh
```

Run with no flags at a terminal, it walks an interactive probe: it looks
for FreeCAD/the optics Python/`pvpython`/`nvcc` at a few common
locations and on `$PATH`, prompts you to confirm or type a path for
anything it can't find on its own, and reads the GPU's SM architecture
off `nvidia-smi` for `MIEWB_CUDA_ARCH`. `MIEWB_FREECAD` and
`MIEWB_OPTICS_PYTHON` are required — the script won't write a file
missing either. `MIEWB_PVPYTHON` and the two CUDA keys can be left
empty (deliberately "not installed here"); it asks before doing so.

For CI/scripted installs, pass the paths as flags and skip the prompts:

```bash
scripts/setup_env.sh --non-interactive \
    --freecad /path/to/FreeCAD.AppImage \
    --optics-python /path/to/optics/env/bin/python \
    --pvpython /path/to/ParaView/bin/pvpython \
    --nvcc /usr/local/cuda-13/bin/nvcc --cuda-arch 89
```

`--pvpython ''`/`--nvcc ''`/`--cuda-arch ''` configure that key absent
(a missing `--freecad`/`--optics-python` under `--non-interactive` exits
2 and writes nothing). `--print` shows the file that would be written
without writing it. Re-running is idempotent: an existing miewb.env
value wins over a fresh probe unless a flag overrides it.

The file itself is flat `KEY=value` lines (see `miewb.env.example` for
the full contract and comments):

```
MIEWB_FREECAD=/path/to/FreeCAD.AppImage
MIEWB_OPTICS_PYTHON=/path/to/optics/env/bin/python
MIEWB_PVPYTHON=/path/to/ParaView/bin/pvpython
MIEWB_NVCC=
MIEWB_CUDA_ARCH=
```

**Precedence** (checked by `scripts/common.py` at import, for every
interpreter stack): an exported `MIEWB_*` environment variable beats a
miewb.env entry, which beats nothing — a required key (`MIEWB_FREECAD`,
`MIEWB_OPTICS_PYTHON`, `MIEWB_PVPYTHON`) missing from both is a hard
`UnconfiguredError` at import, naming this script. An **empty** value
(`MIEWB_PVPYTHON=`) means "configured absent": the pipeline's viz stage
skips cleanly with a NOTICE instead of erroring, and empty CUDA keys
give a CPU-only C-engine build (§ C engine, below) — this is the normal
shape of a headless-server miewb.env (§7). `MIEWB_ALLOW_UNCONFIGURED=1`
is an escape hatch that lets an unconfigured machine import anyway
(unresolved tools resolve to `None`, so only the stage that actually
needs one fails); the GUI sets it for itself, and so do the test
conftests.

To use these paths directly in a shell (rather than relying on `common.py`
to read miewb.env per invocation), source the loader — it exports every
key plus `MIEWB_INST_DIR` (the repo root), letting doc examples like
`"$MIEWB_OPTICS_PYTHON" -m pytest ...` work verbatim:

```bash
source scripts/miewb_env.sh
```

The GUI's **File → Settings… → Tool Paths** page edits this same
`miewb.env` file in place (a one-time migration folds in any paths from
an older QSettings-only install); launching the GUI on a machine with no
miewb.env opens that dialog automatically instead of failing.

## 6. First run

```bash
# 1. sanity-check the configured tool paths and a battery of pure-math
#    invariants
python3 scripts/common.py

# 2. generate the primitive element library if primitives/ is empty
#    (it ships populated in this repo, but regenerate after editing
#    scripts/primitivelib.py — see CUSTOMIZE.md)
"$MIEWB_FREECAD" -c scripts/make_primitives.py -- --kind all < /dev/null

# 3. launch the GUI
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

## 7. Headless-server install (no GUI)

A machine that only needs to run jobs — e.g. the target of an "Export Run
Script" from a workstation, or a CI runner — needs none of the GUI
virtualenv (§4). Just:

- system `python3` (for `run_pipeline.py`/`miewb_tool.py`),
- the FreeCAD AppImage (§3.1),
- the optics environment (§3.2),
- ParaView (§3.3), only if you want `viz/*.png` renders — configure
  `MIEWB_PVPYTHON` empty and omit it otherwise (§5).

Configure the machine (§5) non-interactively in one line — an empty
`--pvpython` is a deliberate "no ParaView here", after which the
pipeline's viz stage skips cleanly with a NOTICE instead of failing:

```bash
scripts/setup_env.sh --non-interactive \
    --freecad /path/to/FreeCAD.AppImage \
    --optics-python /path/to/optics/env/bin/python --pvpython ''
```

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

## 8. Verification

```bash
# commands below assume a one-time `scripts/setup_env.sh` (§5) and,
# per shell, `source scripts/miewb_env.sh`

# engine suite (~1336 tests currently; see the actual count with
# --collect-only -q)
"$MIEWB_OPTICS_PYTHON" -m pytest scripts/raytracer/tests/ -q

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
gcc >= 11, cmake >= 3.22, ninja, OpenMP; CUDA >= 13 for the GPU gather.
`build.sh` reads `MIEWB_NVCC`/`MIEWB_CUDA_ARCH` from miewb.env
(`scripts/setup_env.sh` probes both — `--nvcc`/`--cuda-arch` to set
them explicitly); leaving either empty builds CPU-only.

    cd cengine && ./build.sh          # -> cengine/build/miewb-trace
    ./build.sh test                   # C unit tests

Nothing else changes: `--engine auto` (the default) uses it whenever the
scene's features are ported and falls back to the Python engine
otherwise. Set `MIEWB_CENGINE` to override the binary path.
