#!/usr/bin/env python3
# =============================================================================
# run_library_tests.py — per-item sweep runner for the expanded optical
# property library.
#
# Exercises EVERY newly-added optical-property item (materials, coatings,
# filters, polarizers, gratings, uniaxial crystals — the authoritative list
# lives in demos/library_tests/new_items.json) with an ultra-quick end-to-end
# pipeline run. For each item it:
#
#   1. picks a template scene appropriate to the item's optical role,
#   2. picks a sensible test wavelength (pick_lambda, the one pure function
#      below — reviewed by the main assistant),
#   3. copies the template .FCStd, swaps the ONE property that names the item
#      (material / coating / filter / polarizer / grating registry name) and
#      retunes the source wavelength, through the GUI's own fc_server op path
#      (one persistent FcClient for the whole sweep),
#   4. packs a .MieWB with ultra-quick simparams and runs it to a .MieSim,
#   5. harvests exit code, case status, closure_ok and per-detector power,
#   6. purges the per-item workdir/.MieWB/.MieSim (disk hygiene) — kept only
#      for FAILED items when --keep-failures is set,
#   7. writes results.csv + RESULTS.md (both rerunnable / restartable).
#
# Interpreter: plain system python3 (stdlib only, like miewb_tool.py /
# common.py). fcclient is Qt-free and safe to import here. run_pipeline is
# invoked by miewb_tool.run_miewb, which orchestrates the pinned FreeCAD /
# optics-env / pvpython interpreters itself.
#
#   python3 scripts/run_library_tests.py                 # every new item
#   python3 scripts/run_library_tests.py --category filters
#   python3 scripts/run_library_tests.py --items materials:copper,gratings:lamellar_1200
#   python3 scripts/run_library_tests.py --jobs 4        # partitioned subprocesses
#
# The sweep is RESTARTABLE: items already present in the results CSV are
# skipped unless --force. Sources are all coherent=False (direct deposit),
# so the coherent Huygens gather never trips at the runner's low ray counts.
# =============================================================================
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import common          # noqa: E402  (stdlib-only contract hub)
import miewb_tool      # noqa: E402  (pack_miewb / run_miewb, stdlib)

OPTPROPS = common.OPTPROPS_DIR
LIBTESTS = REPO / "demos" / "library_tests"
NEW_ITEMS = LIBTESTS / "new_items.json"

# reference source power (both laser_collimated and source_broadband default
# to 5.0 mW); success gating is relative to this
SRC_POWER_MW = 5.0

# glass materials that are opaque/reflective in the visible (high k) and so
# get tested on the reflective 45deg template, like the metals
SEMICONDUCTORS = {"silicon", "germanium", "gaas", "sic"}

CATEGORIES = ["materials", "coatings", "filters", "polarizers",
              "gratings", "uniaxial"]


# ---------------------------------------------------------------------------
# template metadata
#
#   swap_body / swap_prop : the body + Base property that names the item
#   source                : the source body whose wavelength is retuned
#   detectors             : recording detector body label(s) (report keys)
#   nlambda / band        : monochromatic (1) vs 3-stratum band source
#   grating               : swap value is "<FaceN>=@<name>" preserving FaceN
# ---------------------------------------------------------------------------
TEMPLATES = {
    "mat_transmissive": dict(swap_body="Window", swap_prop="material",
                             source="Laser", detectors=["Detector"],
                             nlambda=1, band=False),
    "mat_metal_45":     dict(swap_body="Mirror", swap_prop="material",
                             source="Laser", detectors=["Detector"],
                             nlambda=1, band=False),
    "crystal_waveplate": dict(swap_body="Waveplate", swap_prop="material",
                              source="Laser", detectors=["Detector"],
                              nlambda=1, band=False),
    "coated_plate_0":   dict(swap_body="Window", swap_prop="coating",
                             source="Laser", detectors=["Detector"],
                             nlambda=1, band=False),
    "coated_plate_45":  dict(swap_body="Splitter", swap_prop="coating",
                             source="Laser", detectors=["det_r", "det_t"],
                             nlambda=1, band=False),
    "filter_plate":     dict(swap_body="Filter", swap_prop="filter",
                             source="Source", detectors=["Detector"],
                             nlambda=3, band=True),
    "polarizer_plate":  dict(swap_body="Polarizer", swap_prop="polarizer",
                             source="Laser", detectors=["Detector"],
                             nlambda=1, band=False),
    "grating_plate":    dict(swap_body="Grating", swap_prop="grating",
                             source="Laser", detectors=["Detector"],
                             nlambda=1, band=False, grating=True),
    "led_source":       dict(swap_body=None, swap_prop=None,
                             source="LED", detectors=["Detector"],
                             nlambda=3, band=True),
}

# per-item wavelength / source overrides for edge cases discovered while
# smoke-testing. name -> {"lambda_nm": float, ["lambdamin_nm":..,
# "lambdamax_nm":..], "note": "..."}. Keyed by the bare item name.
PER_ITEM_OVERRIDES = {
    # germanium's nk table has a deliberate 0.83-2.5um GAP (interband edge
    # left untabulated); the geometric mean of the full 0.21-12um span lands
    # inside that gap. Pin it into the visible tabulated portion where k is
    # high (strongly reflective -> plenty of power on the 45deg arm).
    "germanium": {"lambda_nm": 500.0,
                  "note": "geomean lands in the 0.83-2.5um untabulated gap; "
                          "pinned to 500nm (visible, high k, reflective)"},
    "silicon": {"lambda_nm": 500.0,
                "note": "pinned to 500nm visible (high k on the 45deg arm)"},
    "gaas": {"lambda_nm": 500.0,
             "note": "pinned to 500nm visible (high k on the 45deg arm)"},
    "sic": {"lambda_nm": 500.0,
            "note": "pinned to 500nm visible"},
}


# ===========================================================================
# registry parsing helpers (pure; no FreeCAD, no numpy)
# ===========================================================================
def _read_csv_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


_CACHE = {}


def _materials():
    if "materials" not in _CACHE:
        _CACHE["materials"] = {r["name"]: r
                               for r in _read_csv_rows(OPTPROPS /
                                                       "materials.miemat")}
    return _CACHE["materials"]


def _coatings():
    if "coatings" not in _CACHE:
        _CACHE["coatings"] = {r["name"]: r
                              for r in _read_csv_rows(OPTPROPS / "coating" /
                                                      "coatings.miecoat")}
    return _CACHE["coatings"]


def _filters():
    if "filters" not in _CACHE:
        _CACHE["filters"] = {r["name"]: r
                             for r in _read_csv_rows(OPTPROPS / "filter" /
                                                     "filters.miefilt")}
    return _CACHE["filters"]


def _polarizers():
    if "polarizers" not in _CACHE:
        _CACHE["polarizers"] = {r["name"]: r
                                for r in _read_csv_rows(OPTPROPS /
                                                        "polarizer" /
                                                        "polarizers.miepol")}
    return _CACHE["polarizers"]


def _gratings():
    if "gratings" not in _CACHE:
        _CACHE["gratings"] = {r["name"]: r
                              for r in _read_csv_rows(OPTPROPS / "grating" /
                                                      "gratings.miegrat")}
    return _CACHE["gratings"]


def _uniaxial():
    if "uniaxial" not in _CACHE:
        _CACHE["uniaxial"] = {r["name"]: r
                              for r in _read_csv_rows(OPTPROPS /
                                                      "birefringence" /
                                                      "uniaxial.miebrf")}
    return _CACHE["uniaxial"]


def _table_wavelengths(path):
    """First (wavelength_nm) column of a .mietab, as sorted floats."""
    rows = _read_csv_rows(path)
    return sorted(float(r["wavelength_nm"]) for r in rows)


def _nk_range_nm(nk_file):
    """(min, max) wavelength [nm] of an nk table."""
    wl = _table_wavelengths(OPTPROPS / "nk" / nk_file)
    return wl[0], wl[-1]


def _material_trans_um(name):
    """(min, max) transmission window [um] of a material row, or None if the
    row leaves the window blank."""
    row = _materials()[name]
    lo, hi = row.get("transmission_um_min"), row.get("transmission_um_max")
    if lo and hi:
        return float(lo), float(hi)
    return None


def _geomean(a, b):
    return math.sqrt(a * b)


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ===========================================================================
# pick_lambda — THE test-wavelength policy (one pure function; keep it here,
# keep it commented). Returns a dict:
#     {"lambdac_nm": float,
#      "lambdamin_nm": float | None,   # None => monochromatic (1 stratum)
#      "lambdamax_nm": float | None,
#      "note": str}
# The sources treat "neither lambdamin nor lambdamax" as monochromatic at
# lambdac (sources.wavelength_strata) -- so a mono pick leaves both None
# rather than setting equal bounds (equal bounds are a degenerate zero-width
# Gaussian). A band pick sets both bounds for a 3-stratum source.
# ===========================================================================
def pick_lambda(category, name):
    ov = PER_ITEM_OVERRIDES.get(name)
    if ov and "lambda_nm" in ov:
        return {"lambdac_nm": float(ov["lambda_nm"]),
                "lambdamin_nm": ov.get("lambdamin_nm"),
                "lambdamax_nm": ov.get("lambdamax_nm"),
                "note": ov.get("note", "per-item override")}

    if category == "materials":
        return _pick_material(name)
    if category == "uniaxial":
        return _pick_uniaxial(name)
    if category == "coatings":
        return _pick_coating(name)
    if category == "filters":
        return _pick_filter(name)
    if category == "polarizers":
        return _pick_polarizer(name)
    if category == "gratings":
        return _pick_grating(name)
    raise ValueError("unknown category %r" % category)


def _mono(lam_nm, note=""):
    return {"lambdac_nm": float(lam_nm), "lambdamin_nm": None,
            "lambdamax_nm": None, "note": note}


def _pick_material(name):
    row = _materials()[name]
    model = row["model"]
    if model == "tabulated":
        # tabulated (metals, semiconductors): geometric mean of the nk
        # table's own wavelength range, clamped >=2% inside the ends so the
        # tracer never has to extrapolate.
        lo, hi = _nk_range_nm(row["nk_file"])
        lam = _clamp(_geomean(lo, hi), lo * 1.02, hi * 0.98)
        return _mono(lam, "tabulated: geomean of nk range [%g,%g]nm"
                     % (lo, hi))
    # parametric (sellmeier / cauchy / constant): geometric mean of the
    # transmission window, clamped to a broadly sensible [0.25,12]um; blank
    # window -> a safe 0.55um.
    win = _material_trans_um(name)
    if win is None:
        return _mono(550.0, "parametric: blank transmission window -> 550nm")
    lam_um = _clamp(_geomean(win[0], win[1]), 0.25, 12.0)
    return _mono(lam_um * 1000.0,
                 "parametric: geomean of transmission [%g,%g]um" % win)


def _pick_uniaxial(name):
    # geometric mean of the INTERSECTION of the o- and e-material
    # transmission windows (the crystal is only usable where both rays
    # transmit).
    row = _uniaxial()[name]
    o = _material_trans_um(row["n_o_material"])
    e = _material_trans_um(row["n_e_material"])
    if o is None or e is None:
        return _mono(550.0, "uniaxial: o/e window blank -> 550nm")
    lo, hi = max(o[0], e[0]), min(o[1], e[1])
    if lo >= hi:
        return _mono(550.0, "uniaxial: empty o/e window intersection -> 550nm")
    # the crystal_waveplate template's fixed ideal_linear analyzer only has
    # tabulated data 350-1100nm, so keep the test wavelength inside that
    # (and away from the ends) even when the crystal transmits far into the IR
    lam_um = _clamp(_geomean(lo, hi), max(0.40, lo * 1.02), min(1.05, hi))
    return _mono(lam_um * 1000.0,
                 "uniaxial: geomean of o&e window intersection [%g,%g]um, "
                 "clamped to analyzer support" % (lo, hi))


def _pick_coating(name):
    row = _coatings()[name]
    table = (row.get("table") or "").strip()
    if table:
        # table coating: mid of the table's wavelength range
        wl = _table_wavelengths(OPTPROPS / "coating" / "tables" / table)
        return _mono((wl[0] + wl[-1]) / 2.0,
                     "coating table: mid of [%g,%g]nm" % (wl[0], wl[-1]))
    # TMM layer stack: use the first quarter-wave design wavelength
    # (mgf2:qw@633 -> 633); no qw anywhere -> 550nm.
    layers = row.get("layers") or ""
    for tok in layers.replace(";", " ").split():
        if "qw@" in tok:
            try:
                return _mono(float(tok.split("qw@", 1)[1]),
                             "coating TMM: first qw@ design wavelength")
            except ValueError:
                pass
    return _mono(550.0, "coating TMM: no qw@ design wavelength -> 550nm")


def _pick_filter(name):
    # source band centred on the filter's peak transmittance, +-max(10nm,
    # 2x local table spacing), CLIPPED inside the table range. 3 strata.
    row = _filters()[name]
    tbl = _read_csv_rows(OPTPROPS / "filter" / "tables" / row["table_csv"])
    pts = sorted((float(r["wavelength_nm"]),
                  float(r["transmittance_internal"])) for r in tbl)
    wls = [w for w, _ in pts]
    tmax = max(t for _, t in pts)
    # peak wavelength (ties -> the middle one, so a flat plateau centres
    # the band rather than picking an edge)
    peak_wls = [w for w, t in pts if t == tmax]
    peak = peak_wls[len(peak_wls) // 2]
    # local spacing near the peak
    i = wls.index(peak)
    spacings = []
    if i > 0:
        spacings.append(wls[i] - wls[i - 1])
    if i < len(wls) - 1:
        spacings.append(wls[i + 1] - wls[i])
    local = min(spacings) if spacings else 10.0
    half = max(10.0, 2.0 * local)
    # the Gaussian source SAMPLES beyond [lambdamin, lambdamax] (the bounds
    # set sigma, they don't truncate), and the filter table hard-errors on
    # any out-of-range wavelength -- so the band center must sit far enough
    # inside the table that the sampling tail (~5x the half-width) never
    # leaves it. Filters whose peak T is at a table EDGE (the NIR longpass
    # rg715/780/830/850 family peaks at the last gridpoint) get pulled
    # inward; a table too narrow for even that collapses to its middle.
    safe_lo, safe_hi = wls[0] + 5.0 * half, wls[-1] - 5.0 * half
    if safe_lo <= safe_hi:
        center = _clamp(peak, safe_lo, safe_hi)
    else:
        center = 0.5 * (wls[0] + wls[-1])
        half = max(10.0, (wls[-1] - wls[0]) / 12.0)
    lo = _clamp(center - half, wls[0], center)
    hi = _clamp(center + half, center, wls[-1])
    return {"lambdac_nm": float(center), "lambdamin_nm": float(lo),
            "lambdamax_nm": float(hi),
            "note": "filter: peak T at %gnm, center %gnm (tail-safe), "
                    "band +-%gnm" % (peak, center, half)}


def _pick_polarizer(name):
    # linear mid of the polarizer table's wavelength range. Monochromatic.
    row = _polarizers()[name]
    wl = _table_wavelengths(OPTPROPS / "polarizer" / "tables" /
                            row["table_csv"])
    return _mono((wl[0] + wl[-1]) / 2.0,
                 "polarizer: linear mid of [%g,%g]nm" % (wl[0], wl[-1]))


def _parse_params(params_str):
    out = {}
    for tok in (params_str or "").split(";"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _pick_grating(name):
    # Order-landing note: grating_plate has an 80mm detector 50mm downstream;
    # with groove period d = 1/lines_per_mm, order m lands at sin(theta) =
    # m*lambda/d. If m*lambda/d >~ 0.6 the +-1 order misses/evanesces -- OK,
    # the zero order still lands (and carries plenty of power). Recorded in
    # the note, not gated.
    row = _gratings()[name]
    model = row["model"]
    lpmm = float(row["lines_per_mm"]) if row.get("lines_per_mm") else None
    if model == "table":
        wl = _table_wavelengths(OPTPROPS / "grating" / "tables" /
                                row["table_csv"])
        lam = (wl[0] + wl[-1]) / 2.0
        note = "grating table: mid of [%g,%g]nm" % (wl[0], wl[-1])
    elif model == "bragg_kogelnik":
        params = _parse_params(row.get("params"))
        design = None
        for key in ("lambda_nm", "lambda0_nm", "design_nm", "lambda0",
                    "lambda"):
            if key in params:
                try:
                    design = float(params[key])
                except ValueError:
                    design = None
        lam = design if design is not None else 550.0
        note = ("grating bragg: design wavelength" if design is not None
                else "grating bragg: no design wavelength -> 550nm")
    else:  # lamellar / dammann / anything else
        lam = 550.0
        note = "grating %s: 550nm" % model
    if lpmm:
        d_nm = 1e6 / lpmm            # groove period in nm (1/lpmm mm)
        s = lam / d_nm
        note += "; m1 sin(theta)=%.2f%s" % (
            s, " (order likely misses the detector; zero order lands)"
            if s >= 0.6 else "")
    return _mono(lam, note)


# ===========================================================================
# template selection
# ===========================================================================
def select_template(category, name):
    if category == "materials":
        row = _materials()[name]
        if row["class"] == "metal" or name in SEMICONDUCTORS:
            return "mat_metal_45"
        # crystal-axis rows (name_o/_e/_nx/... ) are plain materials -> test
        # transmissively; a genuine uniaxial crystal comes in via the
        # 'uniaxial' category, not here.
        return "mat_transmissive"
    if category == "uniaxial":
        return "crystal_waveplate"
    if category == "coatings":
        row = _coatings()[name]
        aoi = (row.get("aoi_deg") or "").strip()
        is45 = "_45" in name
        try:
            is45 = is45 or (aoi and abs(float(aoi) - 45.0) < 5.0)
        except ValueError:
            pass
        # reflectors (HR stacks, protected metals, laser-line mirrors) put
        # ~all their power in the reflected arm -- test them on the two-arm
        # 45deg template so they get a positive-power pass instead of an
        # expected-LOWPOWER on the transmission arm (their design aoi may be
        # 0deg; at 45deg a detuned HR still reflects strongly, which is all
        # this smoke test needs)
        if name.startswith(("HR_", "protected_", "laser_mirror")):
            return "coated_plate_45"
        return "coated_plate_45" if is45 else "coated_plate_0"
    if category == "filters":
        return "filter_plate"
    if category == "polarizers":
        return "polarizer_plate"
    if category == "gratings":
        return "grating_plate"
    raise ValueError("unknown category %r" % category)


# ===========================================================================
# scene instantiation + run + harvest (uses a shared FcClient)
# ===========================================================================
def _load_template_simparams(template):
    with __import__("zipfile").ZipFile(str(LIBTESTS / (template + ".MieWB"))) \
            as zf:
        return json.loads(zf.read("simparams.json"))


def _build_simparams(template, meta, feature=None):
    """Template simparams (detector_face pins etc.) merged with the ultra-
    quick sweep overrides. `feature` toggles the photometric/spectrometer
    feature runs."""
    sp = dict(_load_template_simparams(template))
    sp.update({
        "rays": 10000,
        "resolution": 128,
        "nlambda": meta["nlambda"],
        "spectral_bins": 4,
        "viz_rays": 0,
        "steps": "extract,trace,post",
    })
    if feature == "photometric":
        sp.update({"photometric": True, "nlambda": 3, "spectral_bins": 8})
    elif feature == "spectrometer":
        sp.update({"spectrometer": True, "nlambda": 5, "spectral_bins": 16})
    return sp


def instantiate_scene(fc, template, meta, item_name, lam, out_fcstd,
                      do_swap=True):
    """Copy the template .FCStd to out_fcstd, open it on the shared server,
    swap the item property and retune the source, save and close."""
    shutil.copy2(LIBTESTS / (template + ".FCStd"), out_fcstd)
    st = fc.request("open_document", {"path": str(out_fcstd)})
    doc = st["doc"]
    try:
        if do_swap and meta.get("swap_body"):
            body, prop = meta["swap_body"], meta["swap_prop"]
            if meta.get("grating"):
                # preserve the existing FaceN key, point it at the registry
                cur = None
                for b in st["bodies"]:
                    if b["label"] == body:
                        cur = (b.get("properties", {})
                               .get(prop, {}).get("value"))
                face = "Face1"
                if cur and "=" in cur:
                    face = cur.split("=", 1)[0]
                value = "%s=@%s" % (face, item_name)
            else:
                value = item_name
            fc.request("set_property", {"doc": doc, "body": body,
                                        "name": prop, "value": value})
        # retune the source wavelength
        src = meta["source"]
        fc.request("set_property", {"doc": doc, "body": src,
                                    "name": "lambdac", "value": lam["lambdac_nm"]})
        if lam["lambdamin_nm"] is not None and lam["lambdamax_nm"] is not None:
            fc.request("set_property", {"doc": doc, "body": src,
                                        "name": "lambdamin",
                                        "value": lam["lambdamin_nm"]})
            fc.request("set_property", {"doc": doc, "body": src,
                                        "name": "lambdamax",
                                        "value": lam["lambdamax_nm"]})
        else:
            # monochromatic: strip any band bounds so the source is a single
            # stratum at lambdac (equal bounds would be a degenerate Gaussian)
            for bound in ("lambdamin", "lambdamax"):
                _try_remove(fc, doc, src, bound)
        fc.request("save", {"doc": doc})
    finally:
        fc.request("close", {"doc": doc})
    return out_fcstd


def _try_remove(fc, doc, body, name):
    try:
        fc.request("remove_property", {"doc": doc, "body": body,
                                       "name": name})
    except Exception:
        pass   # property simply absent -> nothing to strip


def _harvest(workdir):
    """Pull exit-side facts out of a run workspace.

    Preferred path: post's report.json (closure_ok + per-detector cube). If
    post crashed AFTER trace wrote the detector cubes but BEFORE report.json
    (e.g. the plot_materials diagnostic dies on a TMM coating whose layer
    materials do not cover its fixed 360-1050nm plot grid), fall back to
    case.json status + a direct read of the detector .h5 cubes so the physics
    is still harvested rather than lost to a diagnostic-plot bug."""
    reports = list((workdir / "results").glob("*/*/report.json"))
    cases = list((workdir / "results").glob("*/*/case.json"))
    status = (common.read_case_status(cases[0]) if cases else "missing")
    detectors, closure_ok = {}, None
    if reports:
        rep = json.loads(reports[0].read_text())
        closure_ok = rep.get("closure_ok")
        for label, d in (rep.get("detectors") or {}).items():
            detectors[label] = d
        return status, closure_ok, detectors
    # fallback: trace sets case status "completed" ONLY when closure held, so
    # a completed case means closure was OK even though post did not finish
    if status == "completed":
        closure_ok = True
    detectors = _harvest_h5(workdir)
    return status, closure_ok, detectors


def _harvest_h5(workdir):
    """Read per-detector total_power_W straight from the trace's .h5 cubes
    via the optics-env python (h5py lives there, not in system python3).
    Keys mirror report.json: the detector face label."""
    h5s = list((workdir / "results").glob("*/*/detectors/*.h5"))
    if not h5s:
        return {}
    snippet = (
        "import sys, json, h5py\n"
        "out = {}\n"
        "for p in sys.argv[1:]:\n"
        "    with h5py.File(p) as h:\n"
        "        label = h.attrs.get('label', p)\n"
        "        out[str(label)] = {"
        "'total_power_W': float(h['spectral_cube_mean'][...].sum())}\n"
        "print(json.dumps(out))\n")
    try:
        res = subprocess.run([common.OPTICS_PYTHON, "-c", snippet]
                             + [str(p) for p in h5s],
                             capture_output=True, text=True, timeout=120)
        return json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        return {}


def run_one(fc, category, name, args):
    """Run a single library item. Returns a result-row dict."""
    template = select_template(category, name)
    meta = TEMPLATES[template]
    lam = pick_lambda(category, name)
    stem = "%s__%s" % (category, name)
    row = {"item": name, "category": category, "template": template,
           "lambda_nm": round(lam["lambdac_nm"], 2), "status": "",
           "closure_ok": "", "detected_mW": "", "detected_mW_alt": "",
           "note": lam["note"]}

    fcstd = args.outdir / (stem + ".FCStd")
    mwb = args.outdir / (stem + ".MieWB")
    workdir = args.workroot / stem
    miesim = args.workroot / (stem + ".MieSim")
    if workdir.exists():
        shutil.rmtree(str(workdir), ignore_errors=True)

    try:
        instantiate_scene(fc, template, meta, name, lam, fcstd)
        miewb_tool.pack_miewb(fcstd, mwb,
                              simparams=_build_simparams(template, meta))
        rc, _ = miewb_tool.run_miewb(mwb, miesim, workdir=str(workdir))
        status, closure_ok, detectors = _harvest(workdir)
        row["status_case"] = status
        row["closure_ok"] = "" if closure_ok is None else bool(closure_ok)

        arms = meta["detectors"]
        p0 = _det_mW(detectors, arms[0])
        row["detected_mW"] = "" if p0 is None else round(p0, 6)
        if len(arms) > 1:
            p1 = _det_mW(detectors, arms[1])
            row["detected_mW_alt"] = "" if p1 is None else round(p1, 6)
            # 45deg beamsplitter: pass if R+T lands
            power_ok = ((p0 or 0.0) + (p1 or 0.0)) > 0.10 * SRC_POWER_MW
            row["note"] += "; %s=%.4gmW %s=%.4gmW" % (
                arms[0], (p0 or 0.0), arms[1], (p1 or 0.0))
        else:
            power_ok = (p0 or 0.0) > 0.01 * SRC_POWER_MW

        # reflector coatings on the 0deg TRANSMISSION template legitimately
        # pass ~0 power to the transmitted-arm detector -- flag so a det~0
        # reading is not mistaken for a failure
        if (template == "coated_plate_0" and not power_ok
                and (name.startswith("HR_") or name.startswith("protected_")
                     or "mirror" in name)):
            row["note"] += ("; high-reflector: ~0 transmission on the 0deg "
                            "transmit template is EXPECTED (test reflectance "
                            "separately)")

        physics_ok = bool(closure_ok) and power_ok
        if physics_ok and rc == 0:
            row["status"] = "PASS"
        elif physics_ok and rc != 0:
            # trace physics closed and the expected power landed, but a LATER
            # pipeline stage (the post diagnostic plot) crashed -- harvested
            # from the h5 cubes; flag it so the post bug is visible
            row["status"] = "PASS-nopost"
        elif closure_ok is None and rc != 0:
            row["status"] = "FAIL-run(exit%d)" % rc
        elif not closure_ok:
            row["status"] = "FAIL-closure"
        else:
            # ran + closed but the expected arm is dim; record, don't hard-fail
            row["status"] = "LOWPOWER"
            if row["item"] in EXPECTED_LOW:
                row["note"] += "; " + EXPECTED_LOW[row["item"]]
        if rc != 0:
            # always surface a late-stage crash (post plot_materials etc.)
            row["note"] += "; late stage crashed (exit%d): %s" % (
                rc, _post_error(workdir))
    except Exception as exc:
        row["status"] = "ERROR"
        row["note"] += "; %s: %s" % (type(exc).__name__, exc)

    _cleanup(row, args, fcstd, mwb, workdir, miesim)
    return row


def _find_det(detectors, label):
    """report.json keys detectors by the recording FACE id
    ('<BodyLabel>.<feature>.FaceN'); match on the body-label segment (body
    labels never contain a dot)."""
    for key, d in detectors.items():
        if key == label or key.split(".", 1)[0] == label:
            return d
    return None


def _det_mW(detectors, label):
    d = _find_det(detectors, label)
    if not d or d.get("total_power_W") is None:
        return None
    return float(d["total_power_W"]) * 1000.0


PASS_STATUSES = ("PASS", "PASS-nopost", "LOWPOWER")

# items whose LOWPOWER outcome is physically expected (note appended so the
# results table explains itself)
EXPECTED_LOW = {
    "schott_ng9": "EXPECTED: strong neutral-density glass (tau_i ~0.2 at "
                  "1mm ref, plate is thicker) -- dim by design",
}


def _post_error(workdir):
    """Last exception-looking line from log.post/log.trace (for the note)."""
    for log in ("log.post", "log.trace"):
        hits = list((workdir / "results").glob("*/*/" + log))
        if not hits:
            continue
        try:
            lines = hits[0].read_text().splitlines()
        except OSError:
            continue
        for ln in reversed(lines):
            s = ln.strip()
            if "Error" in s or "error:" in s:
                return s[:200]
    return "no error line found"


def _cleanup(row, args, fcstd, mwb, workdir, miesim):
    failed = row["status"] not in PASS_STATUSES
    keep = args.keep_failures and failed
    if keep:
        return
    for p in (fcstd, mwb, miesim):
        try:
            Path(p).unlink()
        except OSError:
            pass
    shutil.rmtree(str(workdir), ignore_errors=True)


# ===========================================================================
# feature runs (photometric / QE / spectrometer) -- appended after the sweep
# ===========================================================================
def _feature_supported(flag):
    """Probe cli_specs for a pipeline flag (bool store_true)."""
    try:
        import cli_specs
        p = cli_specs.build_parser("pipeline")
        dests = {a.dest for a in p._actions}
        return flag in dests
    except Exception:
        return False


def _qe_supported():
    """QE-curve-on-detector needs a detector loader in optprops. Absent in
    this drop -> the feature run is skipped."""
    try:
        import importlib
        opt = importlib.import_module("raytracer.optprops")
        return hasattr(opt, "load_detectors") or hasattr(opt, "load_qe")
    except Exception:
        return False


def run_features(fc, args):
    rows = []

    # (a) LED source + photometric
    rows.append(_feature_photometric(fc, args))
    # (b) detector QE curve (hamamatsu_s1223) -- only if the engine supports it
    rows.append(_feature_qe(fc, args))
    # (c) grating + spectrometer
    rows.append(_feature_spectrometer(fc, args))
    return rows


def _feature_row(item, note, status="", **kw):
    r = {"item": item, "category": "feature", "template": "", "lambda_nm": "",
         "status": status, "closure_ok": "", "detected_mW": "",
         "detected_mW_alt": "", "note": note}
    r.update(kw)
    return r


def _run_feature_scene(fc, template, feature, item, args, source_lambda=None,
                       extra_props=None):
    meta = TEMPLATES[template]
    stem = "feature__%s" % item
    fcstd = args.outdir / (stem + ".FCStd")
    mwb = args.outdir / (stem + ".MieWB")
    workdir = args.workroot / stem
    miesim = args.workroot / (stem + ".MieSim")
    if workdir.exists():
        shutil.rmtree(str(workdir), ignore_errors=True)
    shutil.copy2(LIBTESTS / (template + ".FCStd"), fcstd)
    st = fc.request("open_document", {"path": str(fcstd)})
    doc = st["doc"]
    try:
        if source_lambda is not None:
            fc.request("set_property", {"doc": doc, "body": meta["source"],
                                        "name": "lambdac",
                                        "value": source_lambda["lambdac_nm"]})
            for b in ("lambdamin", "lambdamax"):
                key = b + "_nm"
                if source_lambda.get(key) is not None:
                    fc.request("set_property",
                               {"doc": doc, "body": meta["source"],
                                "name": b, "value": source_lambda[key]})
        for (body, prop, value) in (extra_props or []):
            fc.request("set_property", {"doc": doc, "body": body,
                                        "name": prop, "value": value})
        fc.request("save", {"doc": doc})
    finally:
        fc.request("close", {"doc": doc})
    sp = _build_simparams(template, meta, feature=feature)
    miewb_tool.pack_miewb(fcstd, mwb, simparams=sp)
    rc, _ = miewb_tool.run_miewb(mwb, miesim, workdir=str(workdir))
    status, closure_ok, detectors = _harvest(workdir)
    return rc, status, closure_ok, detectors, (fcstd, mwb, workdir, miesim)


def _feature_photometric(fc, args):
    if not _feature_supported("photometric"):
        return _feature_row("led_source+photometric",
                            "photometric flag absent from pipeline",
                            status="SKIPPED-notyet")
    try:
        rc, status, closure_ok, det, paths = _run_feature_scene(
            fc, "led_source", "photometric", "led_photometric", args)
        photo = (_find_det(det, "Detector") or {}).get("photometric")
        ok = rc == 0 and closure_ok and photo is not None
        row = _feature_row(
            "led_source+photometric",
            "photometric keys: %s" % (sorted(photo) if photo else "MISSING"),
            status="PASS" if ok else ("FAIL" if rc == 0 else
                                      "FAIL-run(exit%d)" % rc),
            closure_ok="" if closure_ok is None else bool(closure_ok))
        if photo:
            row["detected_mW"] = round(photo.get("luminous_flux_lm", 0.0), 6)
    except Exception as exc:
        row = _feature_row("led_source+photometric",
                           "%s: %s" % (type(exc).__name__, exc),
                           status="ERROR")
        paths = None
    _feature_cleanup(row, args, paths)
    return row


def _feature_qe(fc, args):
    if not _qe_supported():
        return _feature_row(
            "detector_qe(hamamatsu_s1223)",
            "no detector QE loader in raytracer.optprops (load_detectors/"
            "load_qe absent) -- feature not landed yet",
            status="SKIPPED-notyet")
    try:
        lam = {"lambdac_nm": 780.0, "lambdamin_nm": 770.0,
               "lambdamax_nm": 790.0}
        rc, status, closure_ok, det, paths = _run_feature_scene(
            fc, "mat_transmissive", None, "detector_qe", args,
            source_lambda=lam,
            extra_props=[("Detector", "qe_curve", "hamamatsu_s1223")])
        qe = (_find_det(det, "Detector") or {}).get("qe")
        ok = rc == 0 and closure_ok and qe is not None
        row = _feature_row("detector_qe(hamamatsu_s1223)",
                           "qe keys: %s" % (sorted(qe) if qe else "MISSING"),
                           status="PASS" if ok else "FAIL",
                           closure_ok="" if closure_ok is None
                           else bool(closure_ok))
    except Exception as exc:
        row = _feature_row("detector_qe(hamamatsu_s1223)",
                           "%s: %s" % (type(exc).__name__, exc),
                           status="ERROR")
        paths = None
    _feature_cleanup(row, args, paths)
    return row


def _feature_spectrometer(fc, args):
    if not _feature_supported("spectrometer"):
        return _feature_row("grating+spectrometer",
                            "spectrometer flag absent from pipeline",
                            status="SKIPPED-notyet")
    try:
        # broaden the grating source so there is a spectrum to disperse
        lam = {"lambdac_nm": 550.0, "lambdamin_nm": 450.0,
               "lambdamax_nm": 650.0}
        rc, status, closure_ok, det, paths = _run_feature_scene(
            fc, "grating_plate", "spectrometer", "grating_spectrometer",
            args, source_lambda=lam)
        spec = (_find_det(det, "Detector") or {}).get("spectrometer")
        ok = rc == 0 and closure_ok and spec is not None
        row = _feature_row("grating+spectrometer",
                           "spectrometer keys: %s"
                           % (sorted(spec) if spec else "MISSING"),
                           status="PASS" if ok else "FAIL",
                           closure_ok="" if closure_ok is None
                           else bool(closure_ok))
    except Exception as exc:
        row = _feature_row("grating+spectrometer",
                           "%s: %s" % (type(exc).__name__, exc),
                           status="ERROR")
        paths = None
    _feature_cleanup(row, args, paths)
    return row


def _feature_cleanup(row, args, paths):
    if paths is None:
        return
    fcstd, mwb, workdir, miesim = paths
    failed = row["status"] not in ("PASS", "SKIPPED-notyet")
    if args.keep_failures and failed:
        return
    for p in (fcstd, mwb, miesim):
        try:
            Path(p).unlink()
        except OSError:
            pass
    shutil.rmtree(str(workdir), ignore_errors=True)


# ===========================================================================
# CSV / RESULTS.md IO (restartable)
# ===========================================================================
CSV_FIELDS = ["item", "category", "template", "lambda_nm", "status",
              "closure_ok", "detected_mW", "detected_mW_alt", "status_case",
              "note"]


def _load_done(csv_path):
    done = {}
    if csv_path.exists():
        for r in _read_csv_rows(csv_path):
            done[(r["category"], r["item"])] = r
    return done


def _write_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_md(md_path, rows):
    from collections import Counter
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    lines = ["# Library expansion — per-item smoke-test results", "",
             "Generated by `scripts/run_library_tests.py` (rerunnable). "
             "Each row is one library item run end-to-end through an "
             "ultra-quick pipeline (10k rays / 128px).", ""]
    lines += ["## Summary", "", "| Category | Total | PASS | LOWPOWER | "
              "FAIL/ERROR | SKIPPED |", "|---|---|---|---|---|---|"]
    for cat in CATEGORIES + ["feature"]:
        rs = by_cat.get(cat, [])
        if not rs:
            continue
        npass = sum(1 for r in rs if r["status"] in ("PASS", "PASS-nopost"))
        nlow = sum(1 for r in rs if r["status"] == "LOWPOWER")
        nskip = sum(1 for r in rs if r["status"].startswith("SKIPPED"))
        nfail = len(rs) - npass - nlow - nskip
        lines.append("| %s | %d | %d | %d | %d | %d |"
                     % (cat, len(rs), npass, nlow, nfail, nskip))
    # failures listed
    bad = [r for r in rows if r["status"] not in PASS_STATUSES
           and not r["status"].startswith("SKIPPED")]
    lines += ["", "## Failures / errors", ""]
    if not bad:
        lines.append("_None._")
    else:
        lines += ["| Item | Category | Template | lambda (nm) | Status | Note |",
                  "|---|---|---|---|---|---|"]
        for r in bad:
            lines.append("| %s | %s | %s | %s | %s | %s |"
                         % (r["item"], r["category"], r["template"],
                            r["lambda_nm"], r["status"],
                            str(r["note"]).replace("|", "/")))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")


# ===========================================================================
# item selection + driver
# ===========================================================================
def _all_items():
    data = json.loads(NEW_ITEMS.read_text())
    items = []
    for cat in CATEGORIES:
        for name in data.get(cat, []):
            items.append((cat, name))
    return items


def _select_items(args):
    if args.items:
        out = []
        for tok in args.items.split(","):
            tok = tok.strip()
            if not tok:
                continue
            cat, name = tok.split(":", 1)
            out.append((cat.strip(), name.strip()))
        return out
    allit = _all_items()
    if args.category:
        return [(c, n) for c, n in allit if c == args.category]
    return allit


def _run_jobs_parallel(args, items):
    """Partition items across N subprocesses of this script, then merge the
    partial CSVs. Each child writes its own partial file and skips the
    feature rows (--no-features)."""
    n = args.jobs
    parts = [items[i::n] for i in range(n)]
    procs, partials = [], []
    for i, part in enumerate(parts):
        if not part:
            continue
        spec = ",".join("%s:%s" % (c, nm) for c, nm in part)
        partial = args.workroot / ("partial_%d.csv" % i)
        partials.append(partial)
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--items", spec, "--jobs", "1", "--no-features",
               "--outdir", str(args.outdir), "--workroot", str(args.workroot),
               "--results-csv", str(partial),
               "--results-md", str(args.workroot / ("partial_%d.md" % i))]
        if args.keep_failures:
            cmd.append("--keep-failures")
        if args.force:
            cmd.append("--force")
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        p.wait()
    # merge partials into the main CSV
    done = _load_done(args.results_csv)
    for partial in partials:
        for r in _read_csv_rows(partial):
            done[(r["category"], r["item"])] = r
        try:
            partial.unlink()
        except OSError:
            pass
    rows = list(done.values())
    _write_csv(args.results_csv, rows)
    _write_md(args.results_md, rows)
    print("[libtest] merged %d rows -> %s" % (len(rows), args.results_csv))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", default=None,
                   help="comma list of cat:name (e.g. materials:copper,"
                        "filters:schott_rg645); overrides --category")
    p.add_argument("--category", default=None, choices=CATEGORIES)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--outdir", default=str(LIBTESTS / "generated"))
    p.add_argument("--workroot", default=str(REPO / "var" / "library_tests"))
    p.add_argument("--keep-failures", action="store_true")
    p.add_argument("--results-csv", default=str(LIBTESTS / "results.csv"))
    p.add_argument("--results-md", default=str(LIBTESTS / "RESULTS.md"))
    p.add_argument("--force", action="store_true",
                   help="rerun items already present in the results CSV")
    p.add_argument("--no-features", action="store_true",
                   help="skip the 3 photometric/QE/spectrometer feature rows")
    args = p.parse_args(argv)
    args.outdir = Path(args.outdir)
    args.workroot = Path(args.workroot)
    args.results_csv = Path(args.results_csv)
    args.results_md = Path(args.results_md)
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.workroot.mkdir(parents=True, exist_ok=True)

    items = _select_items(args)

    if args.jobs > 1:
        _run_jobs_parallel(args, items)
        return 0

    done = _load_done(args.results_csv)
    if not args.force:
        items = [(c, n) for c, n in items if (c, n) not in done]

    from mieworkbench.core.fcclient import FcClient
    fc = FcClient()
    rows_by_key = dict(done)
    try:
        for idx, (cat, name) in enumerate(items, 1):
            t0 = time.time()
            print("[libtest] (%d/%d) %s:%s" % (idx, len(items), cat, name),
                  flush=True)
            row = run_one(fc, cat, name, args)
            rows_by_key[(cat, name)] = row
            print("[libtest]   -> %s  lambda=%s  det=%s mW  (%.1fs)"
                  % (row["status"], row["lambda_nm"], row["detected_mW"],
                     time.time() - t0), flush=True)
            # incremental checkpoint after every item (restartable)
            _write_csv(args.results_csv, list(rows_by_key.values()))
        if not args.no_features:
            for frow in run_features(fc, args):
                rows_by_key[(frow["category"], frow["item"])] = frow
                print("[libtest]   feature %s -> %s"
                      % (frow["item"], frow["status"]), flush=True)
    finally:
        fc.shutdown()

    rows = list(rows_by_key.values())
    _write_csv(args.results_csv, rows)
    _write_md(args.results_md, rows)
    print("[libtest] wrote %d rows -> %s + %s"
          % (len(rows), args.results_csv, args.results_md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
