# Headless / remote

Running MieWorkbench without the GUI — a repo clone plus the pinned
tools (or `MIEWB_*` overrides, see CLAUDE.md's interpreter table) is
sufficient. Commands below assume a one-time `scripts/setup_env.sh` and,
per shell, `source scripts/miewb_env.sh` (loads `miewb.env`, exports
`MIEWB_INST_DIR`).

## Pack, run, inspect

```bash
python3 scripts/miewb_tool.py pack model.FCStd -o X.MieWB --simparams p.json
python3 scripts/miewb_tool.py run X.MieWB -o X.MieSim    # unpack -> pipeline -> pack
python3 scripts/miewb_tool.py info X.MieWB
```

`miewb_tool.py run` unpacks the workbench, runs `run_pipeline.py` under
the pinned interpreters (each stage under its own — see CLAUDE.md), then
re-packs everything into a `.MieSim`. See [file-formats.md](file-formats.md)
for the full archive contents.

## Export Run Script (GUI-authored, headlessly executable)

**File → Export Run Script…** in the GUI packs the current model and
writes a portable wrapper:

```sh
#!/bin/sh
set -e
python3 <repo>/scripts/miewb_tool.py run <the>.MieWB -o <the>.MieSim
```

No simulation logic of its own — a thin handoff to a remote/CI machine.

## Locking

Exactly one writer per case directory. `run_trace.py` calls
`common.acquire_case_lock(case_dir)`, which atomically creates
`<case_dir>/.lock.json` (`{pid, host, started, cmdline}`). A locked case
**refuses and exits code 4**:

```
[trace] REFUSED: case is locked by pid <PID> on <HOST> since <TIMESTAMP>
```

A lock is stale (stealable) once its heartbeat is >120s old *and* the
pid is dead; released in a `finally` block on both success and failure.
Never included when packed into a `.MieSim`. The GUI opens a live-locked
case read-only in [monitor mode](results.md) instead of racing it.

## Testing

Two independent suites, two interpreters, never cross-imported:

```bash
# engine (pure Python + numpy/scipy/torch; no FreeCAD, no Qt)
"$MIEWB_OPTICS_PYTHON" -m pytest scripts/raytracer/tests/ -q
#   (-m "not slow" skips the end-to-end cases for a fast loop)

# GUI (PySide6 + VTK; headless via Qt's offscreen platform plugin)
QT_QPA_PLATFORM=offscreen env/bin/python -m pytest mieworkbench/tests -q

# + FreeCAD integration tests (real fc_server worker)
MIEWB_RUN_FREECAD=1 QT_QPA_PLATFORM=offscreen env/bin/python -m pytest mieworkbench/tests -q
```

`@pytest.mark.freecad` tests auto-skip unless `MIEWB_RUN_FREECAD=1`;
`needs_gl` tests skip when running offscreen.

## Environment overrides

A fresh remote clone is configured by running `scripts/setup_env.sh`
(`--non-interactive` for CI/scripted installs), which probes the machine
and writes `<repo>/miewb.env` (gitignored — see `miewb.env.example` for
the file's full contract); `source scripts/miewb_env.sh` then loads it
into the shell. All tool paths/dirs remain env-overridable on top of
that: `MIEWB_FREECAD`, `MIEWB_OPTICS_PYTHON`, `MIEWB_PVPYTHON`,
`MIEWB_GEOMETRY_DIR`, `MIEWB_RESULTS_DIR`, `MIEWB_OPTPROPS_DIR` — an
exported environment variable always beats the `miewb.env` entry, so a
machine-specific install never has to be hard-coded anywhere.
`MIEWB_PROGRESS=1` makes every stage emit `@MIEWB {json}` progress lines
(consumed by the GUI's `RunController`, otherwise just informational on
stdout).
