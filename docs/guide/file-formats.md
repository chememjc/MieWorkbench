# File formats

`scripts/miewb_tool.py` — the format library + standalone CLI, plain
system `python3` (stdlib only). Both the GUI (`mieworkbench.core`
wraps it) and a headless/remote box with nothing but a repo clone use
this same module.

## `.FCStd` — the scene

An ordinary FreeCAD document. The body/face tagging contract
(`material`, `power`/`lambdac`, `coating`, `roughness`, `filter`,
`polarizer`+`polarizer_axis`, `crystal_axis`(2), `grating`,
`surface_override`, `mirror`, `absorbance`, `temperature`, the `dim`
parameter spreadsheet, GUI-internal `miewb_primitive`/`miewb_group`) is
fully specified in [`../RAYTRACER.md`](../RAYTRACER.md) §5; a
quick-reference summary is in [`../../CUSTOMIZE.md`](../../CUSTOMIZE.md).

## `.MieWB` — a portable workbench

A ZIP archive (already-compressed members STORED, not deflated):

```
manifest.json         {"format":"MieWB","version":1,"created":...,
                        "app":..., "fcstd":"model.FCStd"}
model.FCStd            the scene (stored — .FCStd is itself a zip)
opticalproperties/**   the project property library
simparams.json         config-matrix state -> run_pipeline args
project.json           optional GUI/session metadata
```

Opening one in the GUI unpacks it into a scratch workspace
(`var/work/<name>-<hash>/`); Save re-packs the whole archive fresh each
time (not an incremental patch).

```bash
python3 scripts/miewb_tool.py pack model.FCStd -o project.MieWB \
    [--optical-properties DIR] [--simparams params.json]
python3 scripts/miewb_tool.py unpack project.MieWB -d some/dir
python3 scripts/miewb_tool.py info project.MieWB
```

## `.MieSim` — a self-contained result

```
manifest.json            {"format":"MieSim","version":1,"created":...,
                           "source_miewb":..., "model":<stem>,
                           "case":<case>, "status":...}
input.MieWB               the EXACT workbench used for this run (stored)
geometry/<stem>/**        the extracted contract (model.json + face STLs)
results/<stem>/<case>/**  everything the pipeline wrote (never .lock.json)
```

A successful rerun **replaces** `input.MieWB` and every result member of
the same `.MieSim` in place. A live-locked case opens read-only in
[monitor mode](results.md). `--purge-intermediates` drops
`rays.npy`/`viz/*`/logs/per-face STLs while keeping `detectors/*.h5`,
`case.json`, `model.json`.

```bash
python3 scripts/miewb_tool.py run project.MieWB -o result.MieSim [--workdir DIR] [--keep]
python3 scripts/miewb_tool.py pack-sim -d workdir -o result.MieSim --miewb project.MieWB \
    [--model-stem STEM] [--case CASE] [--purge-intermediates]
python3 scripts/miewb_tool.py extract-miewb result.MieSim -o project.MieWB
```

## `sniff()`

`miewb_tool.sniff(path)` distinguishes `.MieWB`/`.MieSim`/bare `.FCStd`
by manifest **content**, not file extension — a renamed archive still
opens correctly.

## Optical property files

Self-describing extensions over plain CSV content (`materials.miemat`,
`coating/coatings.miecoat`, …) — every loader falls back to a same-stem
legacy `.csv` if the new-style file isn't present. See
[`../RAYTRACER.md`](../RAYTRACER.md) §7 for schemas and
[`../../CUSTOMIZE.md`](../../CUSTOMIZE.md) for adding entries; the GUI
surface is the [Property Library Editor](property-library-editor.md).
