#!/usr/bin/env python
# =============================================================================
# post_process.py — rerunnable rendering/analysis stage.
#
# Interpreter: /home3/optics/env/bin/python
#
# Reads  : results/<model>/<case>/{case.json, audit.json, rays.npy,
#          detectors/*.h5} + geometry/<model>/model.json + materials.csv
# Writes : results/<model>/<case>/{images,spectra,plots}/*.png + report.json
#
# Outputs per detector: wavelength-colored sRGB image, linear + log
# grayscale irradiance, horizontal/vertical profiles through the peak,
# per-bin spectrum. Scene-level: 2D XY cross-section ray plot with body
# outlines, per-material n/k dispersion curves, coating R(lambda), energy
# audit bars. Everything is regenerable without re-tracing.
# =============================================================================
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import h5py                                              # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import common                                            # noqa: E402
import cli_specs                                          # noqa: E402
from raytracer.detector import (spectral_cube_to_srgb,   # noqa: E402
                                cie_xyz_weights, spectral_cube_to_lux,
                                spectral_cube_to_photocurrent)
from raytracer.materials import MaterialDB, load_coatings  # noqa: E402
from raytracer import thinfilm as tf                     # noqa: E402
from raytracer import fresnel as fr                      # noqa: E402
from raytracer import grating as gr                      # noqa: E402
from raytracer.optprops import load_optical_properties   # noqa: E402
from raytracer.sources import (wavelength_strata,        # noqa: E402
                               jones_for, n_pol_strata)
from raytracer.audit import BUCKETS                      # noqa: E402
from raytracer.analysis_field import (psf_from_fields,   # noqa: E402
                                      normalize_psf, radial_profile,
                                      mtf2d, mtf50, encircled_energy,
                                      ee_radius)
from raytracer.analysis import (fit_zernike, opd_from_rays,  # noqa: E402
                                strehl_marechal, noll_name,
                                fringe_index, noll_to_nm)


# =============================================================================
# --emit-csv: unified data export (results/<case>/data/*.csv + index.csv).
# Every renderer above stays PNG-only when args.emit_csv is False (byte-
# identical to pre-CSV behavior); CsvEmitter is threaded in as an optional
# argument everywhere a chart's underlying data is already sitting in a
# numpy array, so no rendering math is duplicated or perturbed.
# =============================================================================
class CsvEmitter:
    """Writes results/<case>/data/<filename>.csv and accumulates a row per
    file for the final data/index.csv (file, entity, chart, units,
    provenance, image). Convention: a chart's CSV shares its PNG's
    basename; `image` records the PNG path (relative to the case dir) so
    index.csv can join file <-> chart."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_rows = []

    def emit(self, filename, header_cols, rows, entity="", chart="",
             units="", provenance="", image=None):
        with open(self.data_dir / filename, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header_cols)
            w.writerows(rows)
        self.index_rows.append((filename, entity, chart, units,
                                provenance or "", image or ""))

    def write_index(self):
        with open(self.data_dir / "index.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "entity", "chart", "units", "provenance",
                       "image"])
            w.writerows(self.index_rows)


def _flatten_scalars(d, prefix=""):
    """dict -> [(dotted.key, value)] for every int/float/bool leaf (recurses
    into nested dicts; skips lists/strings/None). Used to dump a report.json
    detector sub-dict into a flat metrics CSV without re-deriving anything."""
    out = []
    for k, v in d.items():
        key = "%s.%s" % (prefix, k) if prefix else k
        if isinstance(v, dict):
            out.extend(_flatten_scalars(v, key))
        elif isinstance(v, bool):
            out.append((key, int(v)))
        elif isinstance(v, (int, float)):
            out.append((key, v))
    return out


# =============================================================================
# Per-(source, detector) detected power -- surfaces case.json["detected"]
# (run_trace.py's per-seed coherent+incoherent tally, see detector.py's
# detected_geometric/detected_incoherent) into report.json + a CSV. Values
# are averaged across seeds (coherent_W/incoherent_W are powers; n_samples
# is a total ray count summed across seeds, not averaged, matching how the
# gather diagnostics themselves report n_samples per seed).
# =============================================================================
def add_per_source_detected(case, report):
    """Populate report['detectors'][label]['per_source'] from case.json's
    per-seed 'detected' block. Returns the flat list of rows (one per
    (source, detector, lam_stratum, pol_stratum)) for the CSV writer."""
    sources = case.get("sources", [])
    detected_all = case.get("detected", {})
    n = float(len(detected_all)) or 1.0
    acc = {}
    for seed_block in detected_all.values():
        for label, rows in seed_block.items():
            for skey, vals in rows.items():
                a = acc.setdefault((label, skey), {"coherent_W": 0.0,
                                                    "incoherent_W": 0.0,
                                                    "n_samples": 0})
                a["coherent_W"] += vals.get("coherent_W", 0.0) / n
                a["incoherent_W"] += vals.get("incoherent_W", 0.0) / n
                a["n_samples"] += vals.get("n_samples", 0)
    flat = []
    for (label, skey), vals in sorted(acc.items()):
        if label not in report["detectors"]:
            continue
        s, l, p = (int(x) for x in skey.split("/"))
        src_name = sources[s] if s < len(sources) else str(s)
        report["detectors"][label].setdefault("per_source", []).append({
            "source": src_name, "lam_stratum": l, "pol_stratum": p,
            "coherent_W": vals["coherent_W"],
            "incoherent_W": vals["incoherent_W"],
        })
        flat.append((src_name, label, l, p, vals["coherent_W"],
                    vals["incoherent_W"], vals["n_samples"]))
    return flat


SLOW_RAY_COLOR = "#7b2d8b"  # biaxial slow sheet (pol_mode 2), dashed
FAST_RAY_COLOR = "#0b6e4f"  # biaxial fast sheet (pol_mode 3), dashed
E_RAY_COLOR = "black"       # fixed distinct color for extraordinary (o/e
                            # split) rays in plot_rays_2d -- deliberately
                            # NOT a wavelength color so it reads unambiguously
                            # against the CIE-derived hues used everywhere
                            # else in this file.


def wavelength_rgb(lam_nm):
    w = cie_xyz_weights(np.atleast_1d(lam_nm))
    from raytracer.detector import _XYZ_TO_SRGB
    rgb = np.clip(w @ _XYZ_TO_SRGB.T, 0, None)
    mx = rgb.max(axis=-1, keepdims=True)
    rgb = np.where(mx > 0, rgb / mx, 0.3)
    return rgb


def detector_qe_curve_for_label(label, qe_bodies):
    """Map a detector h5 label (a face id "<BodyName>.<Tip>.FaceN") to the
    qe_curve of its owning body, if any. qe_bodies is {body_name:
    qe_curve_name} for detector bodies that declared a qe_curve. Ownership
    is a prefix match on the body Name (the face id always begins
    "<BodyName>."), so the run's chosen detector face -- autodetected OR an
    explicit --detector-face on some other FaceN of the same body -- still
    resolves to the right body. Returns the curve name or None."""
    exact = qe_bodies.get(label)
    if exact is not None:
        return exact
    # longest matching body-name prefix wins (guards against one body name
    # being a prefix of another)
    best = None
    for name, curve in qe_bodies.items():
        if label == name or label.startswith(name + "."):
            if best is None or len(name) > len(best[0]):
                best = (name, curve)
    return best[1] if best is not None else None


def render_detector(h5path, outdir_img, outdir_spec, report,
                    photometric=False, spectrometer=False,
                    qe_bodies=None, detector_registry=None,
                    csv_emitter=None):
    with h5py.File(h5path) as h:
        cube = h["spectral_cube_mean"][...]
        mask = h["mask"][...]
        attrs = dict(h.attrs)
        std = h["spectral_cube_std"][...] \
            if "spectral_cube_std" in h else None
        # curved (Sphere/Cylinder) detectors carry a TRUE per-pixel metric
        # area map; planar files have none (byte-compatible with pre-Phase-10)
        area_map = h["pixel_area_map"][...] \
            if "pixel_area_map" in h else None
    label = attrs["label"]
    safe = label.replace(".", "_")
    lam_lo, lam_hi = attrs["lam_lo_m"], attrs["lam_hi_m"]
    pixel_m = attrs["pixel_m"]
    curved = "surface_type" in attrs

    # per-pixel area: curved uses the stored metric map (varies with latitude
    # on a sphere); planar uses the square pixel. Irradiance = power / area
    # either way, so the total detected POWER (cube.sum) is unchanged and
    # flows into the report exactly as for a planar screen.
    if curved:
        pixel_area = np.where(area_map > 0.0, area_map, np.inf)
        W_mm = attrs["radius_m"] * (attrs["u_hi"] - attrs["u_lo"]) / 1e-3
        H_mm = (attrs["radius_m"] * (attrs["v_hi"] - attrs["v_lo"]) / 1e-3
                if attrs["surface_type"] == "sphere"
                else (attrs["v_hi"] - attrs["v_lo"]) / 1e-3)
        extent_mm = [0, W_mm, 0, H_mm]
        col_pitch_mm = W_mm / attrs["W"]
        row_pitch_mm = H_mm / attrs["H"]
        if attrs["surface_type"] == "sphere":
            xlabel, ylabel = "azimuth arc [mm]", "polar arc [mm]"
        else:
            xlabel, ylabel = "arc s [mm]", "axial z [mm]"
    else:
        pixel_area = pixel_m ** 2
        extent_mm = [0, attrs["W"] * pixel_m / 1e-3,
                     0, attrs["H"] * pixel_m / 1e-3]
        col_pitch_mm = pixel_m / 1e-3
        row_pitch_mm = pixel_m / 1e-3
        xlabel, ylabel = "x [mm]", "y [mm]"

    # the stored cube is UNBIASED and may contain zero-mean negative MC
    # noise (see gather.py); sums stay honest, displays clip at zero
    irr_raw = cube.sum(axis=0) / pixel_area             # W/m^2, unbiased
    irr = np.maximum(irr_raw, 0.0)                      # display copy
    total_W = float(cube.sum())
    report["detectors"][label] = {
        "total_power_W": total_W,
        "peak_irradiance_W_m2": float(irr.max()),
        "resolution": [int(attrs["H"]), int(attrs["W"])],
        "pixel_um": float(pixel_m / 1e-6),
    }
    if curved:
        report["detectors"][label]["surface_type"] = str(attrs["surface_type"])

    # sRGB wavelength-colored image (clip the noise for display)
    rgb = spectral_cube_to_srgb(np.maximum(cube, 0.0), lam_lo, lam_hi)
    rgb[~mask] = 0.12
    fig, ax = plt.subplots(figsize=(8, 8), dpi=max(
        128, int(attrs["W"]) // 8))
    ax.imshow(rgb, origin="lower", extent=extent_mm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("%s — wavelength-colored irradiance" % label)
    fig.savefig(outdir_img / ("det_%s.png" % safe), bbox_inches="tight")
    plt.close(fig)

    # linear + log grayscale
    for tag, data in (("lin", irr),
                      ("log", np.log10(np.maximum(
                          irr, irr[irr > 0].min() if np.any(irr > 0)
                          else 1e-30)))):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=max(
            128, int(attrs["W"]) // 8))
        im = ax.imshow(data, origin="lower", cmap="magma",
                       extent=extent_mm)
        fig.colorbar(im, ax=ax, fraction=0.046,
                     label="W/m$^2$" if tag == "lin"
                     else "log10 W/m$^2$")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("%s — irradiance (%s)" % (label, tag))
        fig.savefig(outdir_img / ("det_%s_%s.png" % (safe, tag)),
                    bbox_inches="tight")
        plt.close(fig)

    # profiles through the peak
    if np.any(irr > 0):
        iy, ix = np.unravel_index(np.argmax(irr), irr.shape)
        fig, axes = plt.subplots(2, 1, figsize=(9, 7))
        xmm = (np.arange(irr.shape[1]) + 0.5) * col_pitch_mm
        ymm = (np.arange(irr.shape[0]) + 0.5) * row_pitch_mm
        axes[0].plot(xmm, irr[iy], lw=0.8)
        axes[0].set_title("%s — horizontal profile through peak "
                          "(row %d)" % (label, iy))
        axes[0].set_xlabel(xlabel)
        axes[1].plot(ymm, irr[:, ix], lw=0.8)
        axes[1].set_title("vertical profile through peak (col %d)" % ix)
        axes[1].set_xlabel(ylabel)
        for a in axes:
            a.set_ylabel("W/m$^2$")
        fig.tight_layout()
        fig.savefig(outdir_img / ("det_%s_profiles.png" % safe),
                    bbox_inches="tight")
        plt.close(fig)
        # fringe visibility over the central band (diagnostic)
        seg = irr[iy]
        seg = seg[seg > 0]
        if len(seg) > 16:
            V = (seg.max() - seg.min()) / (seg.max() + seg.min())
            report["detectors"][label]["profile_visibility"] = float(V)
        if csv_emitter is not None:
            img = "images/det_%s_profiles.png" % safe
            csv_emitter.emit(
                "profile_h_%s.csv" % safe, ["position_m", "irradiance_W_m2"],
                zip(xmm * 1e-3, irr[iy]), entity=label,
                chart="profile_horizontal", units="W/m^2", image=img)
            csv_emitter.emit(
                "profile_v_%s.csv" % safe, ["position_m", "irradiance_W_m2"],
                zip(ymm * 1e-3, irr[:, ix]), entity=label,
                chart="profile_vertical", units="W/m^2", image=img)

    # spectrum
    bins = cube.shape[0]
    lam_c = (lam_lo + (np.arange(bins) + 0.5)
             * (lam_hi - lam_lo) / bins) / 1e-9
    pw = cube.reshape(bins, -1).sum(axis=1) * 1e3       # mW
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(lam_c, pw, width=0.9 * (lam_c[1] - lam_c[0]),
           color=wavelength_rgb(lam_c))
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("detected power [mW]")
    ax.set_title("%s — detected spectrum" % label)
    fig.savefig(outdir_spec / ("spectrum_%s.png" % safe),
                bbox_inches="tight")
    plt.close(fig)
    if std is not None:
        np.save(outdir_spec / ("std_cube_%s.npy" % safe), std)
    if csv_emitter is not None:
        csv_emitter.emit(
            "spectrum_%s.csv" % safe, ["wavelength_nm", "power_W"],
            zip(lam_c, pw * 1e-3), entity=label, chart="detected_spectrum",
            units="W", image="spectra/spectrum_%s.png" % safe)

    if photometric:
        _render_photometric(cube, mask, lam_lo, lam_hi, pixel_area,
                            extent_mm, outdir_img, safe, label, report)
    if spectrometer:
        _render_spectrometer(cube, lam_lo, lam_hi, pixel_m, extent_mm,
                             outdir_img, outdir_spec, safe, label, report)

    # QE-weighted photocurrent -- data-driven (no CLI flag): a detector body
    # tagged qe_curve=<name> gets its spectral cube weighted by the
    # registry QE(lambda) curve. Purely a display-stage diagnostic.
    curve = detector_qe_curve_for_label(label, qe_bodies or {})
    if curve is not None:
        entry = (detector_registry or {}).get(curve)
        if entry is None:
            print("[post] NOTE: unknown qe_curve %r — skipping" % curve)
        else:
            i_a, p_w, cov = spectral_cube_to_photocurrent(
                cube, lam_lo, lam_hi, entry["lam_um"], entry["qe"])
            report["detectors"][label]["qe"] = {
                "curve": curve,
                "photocurrent_A": float(i_a),
                "qe_weighted_power_W": float(p_w),
                "coverage_frac": float(cov),
            }

    if csv_emitter is not None:
        # scalar metrics: dump every already-computed number in this
        # detector's report block (total/peak/visibility/photometric/
        # spectrometer/qe) -- no re-derivation, just a flat metric,value
        # table so nothing here can drift from report.json.
        metrics = _flatten_scalars(
            {k: v for k, v in report["detectors"][label].items()
             if k != "per_source"})
        csv_emitter.emit("metrics_%s.csv" % safe, ["metric", "value"],
                         metrics, entity=label, chart="scalar_metrics")


# =============================================================================
# --photometric: CIE photopic (lux) render.
# =============================================================================
def _render_photometric(cube, mask, lam_lo, lam_hi, pixel_area, extent_mm,
                        outdir_img, safe, label, report):
    """--photometric: illuminance render of the same spectral cube the
    sRGB/irradiance images use -- no tracer changes, purely a different
    CIE weighting (y-bar instead of the full xyz -> sRGB matrix)."""
    lux_map, luminous_flux_lm = spectral_cube_to_lux(
        np.maximum(cube, 0.0), lam_lo, lam_hi, pixel_area)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=max(
        128, lux_map.shape[1] // 8))
    im = ax.imshow(lux_map, origin="lower", cmap="magma", extent=extent_mm)
    fig.colorbar(im, ax=ax, fraction=0.046, label="lux")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("%s — illuminance" % label)
    fig.savefig(outdir_img / ("det_%s_lux.png" % safe), bbox_inches="tight")
    plt.close(fig)
    report["detectors"][label]["photometric"] = {
        "luminous_flux_lm": float(luminous_flux_lm),
        "peak_illuminance_lux": float(lux_map.max()),
        "mean_illuminance_lux": float(lux_map[mask].mean())
                                if np.any(mask) else 0.0,
    }


# =============================================================================
# --spectrometer: power-weighted lambda(x,y) centroid render + a lambda(x)
# dispersion fit. The centroid/fit math is factored into pure functions
# (spectral_centroid, lambda_centroid_map, linear_fit_r2) so it is
# testable without a case directory or an h5 file.
# =============================================================================
def spectral_centroid(cube, lam_lo, lam_hi):
    """(bins, ...) power cube -> (total_power (...), lambda_bar_nm (...)),
    the power-weighted wavelength centroid collapsed along axis 0.
    lam_lo/lam_hi are in metres (h5 attrs); lambda_bar is NaN wherever
    total_power is exactly 0."""
    bins = cube.shape[0]
    lam_c_nm = (lam_lo + (np.arange(bins) + 0.5)
               * (lam_hi - lam_lo) / bins) / 1e-9
    total = cube.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_bar = np.tensordot(lam_c_nm, cube, axes=(0, 0)) / total
    return total, lam_bar


def lambda_centroid_map(cube, lam_lo, lam_hi, power_frac_threshold=1e-6):
    """(bins, H, W) power cube -> (lambda_bar (H, W) [nm], valid (H, W)
    bool). A pixel is valid if its total power is at least
    power_frac_threshold of the brightest pixel's total power (matches
    the spec's "total power < 1e-6*peak" masking rule); lambda_bar is NaN
    outside the valid mask."""
    total, lam_bar = spectral_centroid(cube, lam_lo, lam_hi)
    peak = float(total.max()) if total.size else 0.0
    valid = (total >= power_frac_threshold * peak) if peak > 0 \
        else np.zeros_like(total, dtype=bool)
    return np.where(valid, lam_bar, np.nan), valid


def linear_fit_r2(x, y):
    """Least-squares y = slope*x + intercept -> (slope, intercept, r2), or
    (None, None, None) if fewer than 2 points (a constant y gives r2=1.0
    rather than a division by zero)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return None, None, None
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


def _render_spectrometer(cube, lam_lo, lam_hi, pixel_m, extent_mm,
                         outdir_img, outdir_spec, safe, label, report):
    cube = np.maximum(cube, 0.0)
    lam_map, valid2d = lambda_centroid_map(cube, lam_lo, lam_hi)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=max(
        128, lam_map.shape[1] // 8))
    im = ax.imshow(lam_map, origin="lower", cmap="viridis", extent=extent_mm)
    fig.colorbar(im, ax=ax, fraction=0.046, label="wavelength [nm]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("%s — power-weighted wavelength centroid" % label)
    fig.savefig(outdir_img / ("det_%s_lambda_map.png" % safe),
                bbox_inches="tight")
    plt.close(fig)

    # collapse y (sum power over rows first, THEN take the centroid) --
    # equivalent to a power-weighted mean over the whole column
    col_total, col_lam = spectral_centroid(cube.sum(axis=1), lam_lo, lam_hi)
    peak = float(col_total.max()) if col_total.size else 0.0
    col_valid = (col_total >= 1e-6 * peak) if peak > 0 \
        else np.zeros_like(col_total, dtype=bool)
    xmm = (np.arange(cube.shape[2]) + 0.5) * pixel_m / 1e-3

    spec = {"lambda_min_nm": None, "lambda_max_nm": None,
           "dispersion_nm_per_mm": None, "fit_r2": None}
    if np.any(valid2d):
        spec["lambda_min_nm"] = float(np.nanmin(lam_map))
        spec["lambda_max_nm"] = float(np.nanmax(lam_map))
    slope, intercept, r2 = linear_fit_r2(xmm[col_valid], col_lam[col_valid])
    if slope is None:
        print("[post] NOTE: %s spectrometer dispersion fit skipped "
              "(fewer than 2 valid detector columns)" % label)
    else:
        spec["dispersion_nm_per_mm"] = slope
        spec["fit_r2"] = r2

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xmm[col_valid], col_lam[col_valid], "o", ms=3, color="C0")
    if slope is not None:
        fit_y = slope * xmm[col_valid] + intercept
        ax.plot(xmm[col_valid], fit_y, "-", color="C1",
               label="fit: %.4g nm/mm, R^2=%.4f" % (slope, r2))
        ax.legend(fontsize=8)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("wavelength [nm]")
    ax.set_title("%s — wavelength vs x" % label)
    fig.savefig(outdir_spec / ("lambda_vs_x_%s.png" % safe),
                bbox_inches="tight")
    plt.close(fig)

    report["detectors"][label]["spectrometer"] = spec


# =============================================================================
# Polarization-state (Stokes) maps -- DATA-DRIVEN, currently always a no-op.
#
# detector.py does not save per-(source,lam_stratum,pol_stratum) complex
# field maps yet (--save-fields is a separate lead task, see future.md); the
# reader below DEFINES the layout the tracer will need to write for this to
# activate, so WS-J and the lead can land independently:
#
#   <label>.h5:
#     fields/<key>/Ex   complex128 (H, W)   -- transverse E field, x-component
#     fields/<key>/Ey   complex128 (H, W)   -- transverse E field, y-component
#
# one 'fields/<key>' group per (source, lam_stratum, pol_stratum) gather key
# (case.json's "s/l/p" gather-diagnostics keys, joined with '_' instead of
# '/' for HDF5 path-safety, e.g. key "0_0_1"); Ex/Ey share the SAME (H, W)
# pixel grid as spectral_cube_mean/mask (attrs H, W, pixel_m, xhat, yhat).
# The exact numeric formatting of <key> is not load-bearing here -- this
# reader iterates whatever subgroups exist under 'fields' and only requires
# each to contain 'Ex' and 'Ey' datasets. If no 'fields' group is present
# (true for every existing case dir today), render_stokes_maps is a silent
# no-op.
# =============================================================================
def _iter_field_keys(h5file):
    """[(key, h5py.Group)] for every 'fields/<key>' subgroup that has both
    'Ex' and 'Ey' datasets, or [] if the 'fields' group is absent/empty."""
    if "fields" not in h5file:
        return []
    grp = h5file["fields"]
    return [(key, grp[key]) for key in grp
            if "Ex" in grp[key] and "Ey" in grp[key]]


def stokes_from_jones(Ex, Ey):
    """(H,W) complex128 Ex,Ey (an orthogonal transverse basis) -> S0,S1,S2,S3
    (each (H,W) float64), the standard optics-convention Stokes parameters:
      S0 = |Ex|^2 + |Ey|^2                  (total intensity)
      S1 = |Ex|^2 - |Ey|^2                  (linear H/V)
      S2 = 2 Re(Ex Ey*)                     (linear +-45deg)
      S3 = -2 Im(Ex Ey*)                    (circular; sign is a time-
                                              convention choice, harmless for
                                              a diagnostic map)
    """
    Ex = np.asarray(Ex)
    Ey = np.asarray(Ey)
    Ix = np.abs(Ex) ** 2
    Iy = np.abs(Ey) ** 2
    S0 = Ix + Iy
    S1 = Ix - Iy
    cross = Ex * np.conj(Ey)
    S2 = 2.0 * np.real(cross)
    S3 = -2.0 * np.imag(cross)
    return S0, S1, S2, S3


def _plot_stokes_panel(S0, S1, S2, S3, extent_mm, outpath, title):
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    im0 = axes[0, 0].imshow(S0, origin="lower", cmap="gray", extent=extent_mm)
    axes[0, 0].set_title("S0 (intensity)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    lim = max(float(np.max(np.abs(S1))), float(np.max(np.abs(S2))),
             float(np.max(np.abs(S3))), 1e-30)
    for ax, S, name in ((axes[0, 1], S1, "S1"), (axes[1, 0], S2, "S2"),
                        (axes[1, 1], S3, "S3")):
        im = ax.imshow(S, origin="lower", cmap="RdBu_r", extent=extent_mm,
                       vmin=-lim, vmax=lim)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.flat:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def render_stokes_maps(h5path, outdir_img):
    """If <label>.h5 has 'fields/<key>' groups (see the layout comment
    above -- not yet written by the tracer), render one
    stokes_<label>_<key>.png (2x2 S0/S1/S2/S3 panel, S1-S3 sharing a
    diverging scale symmetric about 0) per key, plus a single
    dop_<label>.png summed degree-of-polarization map (the incoherent sum
    of Stokes vectors across every key is valid because different
    (source,lam,pol) strata never interfere -- detector.py's own gather
    contract). No 'fields' group -> does nothing, no error."""
    with h5py.File(h5path) as h:
        keys = _iter_field_keys(h)
        if not keys:
            return
        attrs = dict(h.attrs)
        label = attrs["label"]
        safe = label.replace(".", "_")
        pixel_m = attrs["pixel_m"]
        extent_mm = [0, attrs["W"] * pixel_m / 1e-3,
                    0, attrs["H"] * pixel_m / 1e-3]
        stokes_sum = None
        for key, grp in keys:
            Ex = grp["Ex"][...]
            Ey = grp["Ey"][...]
            S0, S1, S2, S3 = stokes_from_jones(Ex, Ey)
            _plot_stokes_panel(
                S0, S1, S2, S3, extent_mm,
                outdir_img / ("stokes_%s_%s.png" % (safe, key)),
                title="%s -- field %s" % (label, key))
            if stokes_sum is None:
                stokes_sum = [S0.copy(), S1.copy(), S2.copy(), S3.copy()]
            else:
                stokes_sum[0] += S0
                stokes_sum[1] += S1
                stokes_sum[2] += S2
                stokes_sum[3] += S3
        S0, S1, S2, S3 = stokes_sum
        with np.errstate(divide="ignore", invalid="ignore"):
            dop = np.where(S0 > 0,
                          np.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2) / S0, 0.0)
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(np.clip(dop, 0, 1), origin="lower", cmap="viridis",
                       extent=extent_mm, vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046,
                     label="degree of polarization")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("%s -- summed degree of polarization" % label)
        fig.savefig(outdir_img / ("dop_%s.png" % safe), bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# 2D ray plot
# =============================================================================
def _assign_generations(rays):
    """Reconstruct a per-segment reflection/refraction generation index
    from rays.npy's flat, order-independent list of 2-point segments --
    there is no explicit generation column (VizStore in tracer.py, lead-
    owned, does not store one; see future.md's '--viz-generations' note).

    Segments are chained within the same source_id by exact endpoint match
    ((x1,y1,z1) of a parent == (x0,y0,z0) of a child): each ray's full
    bounce/transmission history flows through the same computed hit-point
    array from one tracer.step() to the next, so matching endpoints are
    bit-identical in practice. Roots (generation 0) are segments whose
    start point matches no other same-source segment's end point;
    everything else is 1 + its parent's generation.

    CAVEAT: rays from the same source that happen to cross the exact same
    point (e.g. a shared focus) can rarely be mis-chained to a sibling's
    continuation rather than their own. Harmless for the visual declutter
    this feeds (--viz-generations); not used anywhere quantitative.
    """
    n = len(rays)
    gen = np.zeros(n, dtype=np.int32)
    if n == 0:
        return gen
    src = rays[:, 0].astype(np.int64)
    p0 = np.round(rays[:, 3:6], 9)
    p1 = np.round(rays[:, 6:9], 9)
    end_index = {}
    for i in range(n):
        key = (int(src[i]), float(p1[i, 0]), float(p1[i, 1]), float(p1[i, 2]))
        end_index.setdefault(key, i)      # arbitrary tie-break among
                                          # rays that converge exactly
    parent = np.full(n, -1, dtype=np.int64)
    for j in range(n):
        key = (int(src[j]), float(p0[j, 0]), float(p0[j, 1]), float(p0[j, 2]))
        p = end_index.get(key)
        if p is not None and p != j:
            parent[j] = p
    memo = {}
    for i in range(n):
        chain = []
        cur = i
        depth = 0
        while cur not in memo and parent[cur] >= 0 and depth <= n:
            chain.append(cur)
            cur = int(parent[cur])
            depth += 1
        base = memo.get(cur, 0)
        for k, node in enumerate(reversed(chain)):
            base += 1
            memo[node] = base
        gen[i] = memo.get(i, 0)
    return gen


def plot_rays_2d(rays, model, outpath, max_generation=None,
                 dim_mode="off", dim_floor=0.0):
    """XY cross-section: segments colored by wavelength, alpha ~ power.
    Ordinary/isotropic rays (pol_mode==0, or every ray in an old 9-column
    rays.npy) draw as before; extraordinary rays (pol_mode==1 -- a
    birefringent crystal's o/e split) draw dashed in a fixed distinct
    color with an 'e-ray' legend entry. `max_generation`, if given, drops
    reconstructed-generation > N segments to declutter reflection-heavy
    scenes (see _assign_generations).

    dim_mode 'linear'|'sqrt' switches alpha from the default ensemble
    95th-percentile scaling to each segment's power relative to its own
    ray's power at the source (rel_power, column 10), so attenuation and
    splits fade the trace; dim_floor is a minimum opacity in percent."""
    from matplotlib.lines import Line2D
    if rays.shape[1] >= 10:
        pol_mode = rays[:, 9]
    else:
        pol_mode = np.zeros(len(rays))
    if dim_mode != "off" and rays.shape[1] < 11:
        print("[post] --dim-rays: rays.npy has no rel_power column "
              "(pre-dimming trace) — falling back to percentile alpha")
        dim_mode = "off"
    if max_generation is not None and len(rays):
        gen = _assign_generations(rays)
        gkeep = gen <= max_generation
        rays = rays[gkeep]
        pol_mode = pol_mode[gkeep]

    fig, ax = plt.subplots(figsize=(12, 8))
    # body outlines from trim wires (light gray, projected to XY)
    for b in model["bodies"]:
        if b["role"] == "ignored":
            continue
        for f in b.get("faces", []):
            for wire in f.get("trim_polylines_xyz") or []:
                w = np.asarray(wire)
                keep = np.abs(w[:, 2]) < 5e-3   # near the optical plane
                if keep.sum() < 2:
                    continue
                ax.plot(w[keep, 0] * 1e3, w[keep, 1] * 1e3,
                        color="0.55", lw=0.7, zorder=1)
    mode_colors = {1: E_RAY_COLOR, 2: SLOW_RAY_COLOR, 3: FAST_RAY_COLOR}
    modes_seen = set()
    if len(rays):
        if dim_mode != "off":
            rel = np.clip(rays[:, 10], 0.0, 1.0)
            a = np.sqrt(rel) if dim_mode == "sqrt" else rel
            alpha = np.clip(np.maximum(a, dim_floor / 100.0), 0.0, 1.0)
        else:
            power = rays[:, 2]
            pmax = np.percentile(power[power > 0], 95) \
                if np.any(power > 0) else 1.0
            alpha = np.clip(power / pmax, 0.02, 0.6)
        colors = wavelength_rgb(rays[:, 1] / 1e-9)
        # limit draw count for file size
        idx = np.arange(len(rays))
        if len(idx) > 20000:
            idx = np.random.default_rng(0).choice(idx, 20000,
                                                  replace=False)
        for i in idx:
            mode = int(pol_mode[i])
            special = mode in mode_colors
            if special:
                modes_seen.add(mode)
            ax.plot([rays[i, 3] * 1e3, rays[i, 6] * 1e3],
                    [rays[i, 4] * 1e3, rays[i, 7] * 1e3],
                    color=mode_colors[mode] if special else colors[i],
                    linestyle="--" if special else "-",
                    alpha=float(alpha[i]), lw=0.9 if special else 0.5,
                    zorder=3 if special else 2)
    if modes_seen:
        names = {1: "e-ray", 2: "slow sheet", 3: "fast sheet"}
        ax.legend(handles=[Line2D([0], [0], color=mode_colors[m], ls="--",
                                  lw=1.2, label=names[m])
                           for m in sorted(modes_seen)],
                  loc="upper right", fontsize=8)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    # crop to the optical bench (escaped rays otherwise dominate scale)
    boxes = [b["bbox"] for b in model["bodies"]
             if b["role"] != "ignored" and b.get("bbox")]
    if boxes:
        lo = np.min([b[0] for b in boxes], axis=0)
        hi = np.max([b[1] for b in boxes], axis=0)
        cx, cy = (lo[0] + hi[0]) / 2 * 1e3, (lo[1] + hi[1]) / 2 * 1e3
        half = max(hi[0] - lo[0], hi[1] - lo[1]) * 1e3 * 0.75
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
    ax.set_title("ray trace — XY cross-section (alpha ~ %s)"
                 % ("power / birth power" if dim_mode != "off"
                    else "ray power"))
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _coating_names(b):
    """v2 model.json allows body['coating'] to be either a legacy plain
    string (whole-body coating) or a per-face map {face_id|'__all__':
    coating_name} (common.parse_facemap_spec / _check_facemap) -- flatten
    either shape to the coating names it references."""
    c = b.get("coating")
    if c in (None, "none"):
        return []
    if isinstance(c, dict):
        return [v for v in c.values() if v not in (None, "none")]
    return [c]


def plot_materials(model, outdir, csv_emitter=None):
    props = load_optical_properties()
    db = props.matdb
    used = set()
    for b in model["bodies"]:
        name = b.get("material")
        if name in (None, "none", "detector"):
            continue
        if db.is_birefringent(name):
            # uniaxial crystal names map to their o/e materials.csv rows
            mo, me = db.get_uniaxial(name)
            used.add(mo.name)
            used.add(me.name)
        elif getattr(db, "is_biaxial", lambda _n: False)(name):
            # biaxial crystal names map to their three principal-index rows
            for m in db.get_biaxial(name):
                used.add(m.name)
        else:
            used.add(name)
    used = sorted(used)
    coats = sorted({name for b in model["bodies"]
                    for name in _coating_names(b)})
    lam = np.linspace(360e-9, 1050e-9, 300)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in used:
        try:
            n = db.get(name).n_complex(lam)
        except ValueError:      # tabulated range narrower than plot span
            continue
        axes[0].plot(lam / 1e-9, np.real(n), label=name)
        axes[1].semilogy(lam / 1e-9,
                         np.maximum(np.imag(n), 1e-12), label=name)
        if csv_emitter is not None:
            ref = db.get(name).reference
            csv_emitter.emit(
                "nk_%s.csv" % _safe_name(name),
                ["wavelength_nm", "n", "k", "reference"],
                zip(lam / 1e-9, np.real(n), np.imag(n),
                   [ref] * len(lam)),
                entity=name, chart="material_dispersion", provenance=ref,
                image="plots/materials_nk.png")
    axes[0].set_ylabel("n")
    axes[1].set_ylabel("k")
    for a in axes:
        a.set_xlabel("wavelength [nm]")
        a.legend(fontsize=8)
    fig.suptitle("dispersion of materials used in this scene")
    fig.tight_layout()
    fig.savefig(outdir / "materials_nk.png", bbox_inches="tight")
    plt.close(fig)

    if coats:
        coatings = load_coatings(db=db)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        cos_i = np.ones_like(lam)
        n1 = np.ones_like(lam)
        n_sub = db.get("bk7").n_complex(lam)
        for cname in coats:
            cspec = coatings[cname]
            if cspec["kind"] == "tmm":
                try:
                    ln, ld = tf.resolve_coating_layers(cspec["layers"], db,
                                                       lam)
                except ValueError:  # layer material tabulated range narrower
                    continue        # than the plot span (same as above)
                rs, rp, ts, tp, etas = tf.tmm_coeffs(lam, cos_i, n1, n_sub,
                                                     ln, ld)
                R = 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)
                if csv_emitter is not None:
                    Rs, Rp, Ts, Tp = tf.tmm_power(rs, rp, ts, tp, etas)
                    ref = cspec.get("reference", "")
                    csv_emitter.emit(
                        "coating_%s.csv" % _safe_name(cname),
                        ["wavelength_nm", "Rs", "Rp", "Ts", "Tp",
                         "reference"],
                        zip(lam / 1e-9, Rs, Rp, Ts, Tp, [ref] * len(lam)),
                        entity=cname, chart="coating_RT", provenance=ref,
                        image="plots/coating_reflectance.png")
            else:
                lam_um = lam * 1e6
                R = 0.5 * (np.interp(lam_um, cspec["lam_um"], cspec["Rs"])
                           + np.interp(lam_um, cspec["lam_um"], cspec["Rp"]))
            ax.plot(lam / 1e-9, 100 * R, label=cname)
        rs0, rp0, _, _, _ = fr.fresnel_coeffs(cos_i, n1, n_sub)
        ax.plot(lam / 1e-9, 100 * np.abs(rs0) ** 2, "--", color="0.5",
                label="bare BK7")

        # extend: table-kind coatings also get their own measured
        # Rs/Rp/Ts/Tp curves overlaid, on their NATIVE tabulated grid (not
        # the 360-1050nm span above -- measured tables are usually
        # narrower) and at their own AOI (e.g. the 45deg hot/cold-mirror
        # and PBS tables in coatings.csv) -- the combined-R curve plotted
        # above stays as the at-a-glance summary, these are the detail.
        for cname in coats:
            cspec = coatings[cname]
            if cspec["kind"] != "table":
                continue
            tab_lam_nm = cspec["lam_um"] * 1e3
            aoi = cspec.get("aoi_deg", 0.0)
            for key, ls in (("Rs", "-."), ("Rp", ":"),
                           ("Ts", (0, (1, 1))), ("Tp", (0, (3, 1, 1, 1)))):
                if key in cspec:
                    ax.plot(tab_lam_nm, 100 * cspec[key], linestyle=ls,
                           lw=1.0,
                           label="%s %s (AOI=%.0f°)" % (cname, key, aoi))
            if csv_emitter is not None:
                ref = cspec.get("reference", "")
                nan = np.full(len(tab_lam_nm), np.nan)
                csv_emitter.emit(
                    "coating_%s.csv" % _safe_name(cname),
                    ["wavelength_nm", "Rs", "Rp", "Ts", "Tp", "reference"],
                    zip(tab_lam_nm, cspec.get("Rs", nan),
                       cspec.get("Rp", nan), cspec.get("Ts", nan),
                       cspec.get("Tp", nan), [ref] * len(tab_lam_nm)),
                    entity=cname, chart="coating_RT", provenance=ref,
                    image="plots/coating_reflectance.png")

        ax.set_xlabel("wavelength [nm]")
        ax.set_ylabel("R or T [%]")
        ax.legend(fontsize=7)
        ax.set_title("coating reflectance (+ tabulated Rs/Rp/Ts/Tp detail)")
        fig.savefig(outdir / "coating_reflectance.png",
                    bbox_inches="tight")
        plt.close(fig)


def _safe_name(name):
    return str(name).replace(".", "_").replace("/", "_")


def _referenced_grating_registry_names(model):
    """Sorted set of opticalproperties/grating/gratings.csv registry names
    referenced by any body's per-face 'grating' map (v2 schema): only
    '@name'-form values resolve to a registry entry (common.
    parse_grating_value's 'registry' key); explicit lamellar specs
    ('600:v:...') have no catalog name to plot under and are skipped."""
    names = set()
    for b in model["bodies"]:
        gmap = b.get("grating")
        if not isinstance(gmap, dict):
            continue
        for val in gmap.values():
            try:
                spec = common.parse_grating_value(val)
            except ValueError:
                continue
            if spec.get("registry"):
                names.add(spec["registry"])
    return sorted(names)


def plot_optical_elements(model, outdir):
    """One PNG per optical element the model actually references (v2
    body properties 'polarizer' / 'filter' / per-face 'grating' registry
    entries) -- silently skips anything the model doesn't use, and
    anything a name references that isn't in the loaded opticalproperties/
    library (shouldn't happen for a model that already traced, but this is
    a rerunnable display stage, not a re-validation of the trace inputs).

      polarizer_<name>.png   T_par(lambda), T_perp(lambda) [log y] +
                              extinction ratio on a twin axis
      filter_<name>.png      internal transmittance at ref thickness vs
                              lambda [log y]
      grating_<name>.png     table model: per-order eta_s/eta_p vs lambda;
                              bragg_kogelnik: efficiency vs lambda AT Bragg
                              incidence (raytracer.grating.order_efficiencies,
                              the same dispatch the tracer itself uses);
                              dammann: bar chart of design order
                              efficiencies (wavelength-independent -- see
                              raytracer.grating.dammann_efficiencies)
    """
    polarizer_names = sorted({b["polarizer"] for b in model["bodies"]
                              if b.get("polarizer")})
    filter_names = sorted({b["filter"] for b in model["bodies"]
                           if b.get("filter")})
    grating_names = _referenced_grating_registry_names(model)
    if not (polarizer_names or filter_names or grating_names):
        return

    props = load_optical_properties()

    for name in polarizer_names:
        spec = props.polarizers.get(name)
        if spec is None:
            continue
        lam_nm = spec["lam_um"] * 1e3
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.semilogy(lam_nm, spec["T_par"], color="C0", label="T_par")
        ax1.semilogy(lam_nm, spec["T_perp"], color="C1", label="T_perp")
        ax1.set_xlabel("wavelength [nm]")
        ax1.set_ylabel("transmittance")
        ax2 = ax1.twinx()
        er = spec["T_par"] / spec["T_perp"]
        ax2.semilogy(lam_nm, er, color="0.3", ls="--", label="ER")
        ax2.set_ylabel("extinction ratio T_par / T_perp")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=8)
        ax1.set_title("polarizer %r (%s)" % (name, spec["type"]))
        fig.savefig(outdir / ("polarizer_%s.png" % _safe_name(name)),
                    bbox_inches="tight")
        plt.close(fig)

    for name in filter_names:
        spec = props.filters.get(name)
        if spec is None:
            continue
        lam_nm = spec["lam_um"] * 1e3
        T = np.exp(-spec["alpha_per_m"] * spec["ref_thickness_m"])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.semilogy(lam_nm, T)
        ax.set_xlabel("wavelength [nm]")
        ax.set_ylabel("internal transmittance @ %.3g mm"
                     % (spec["ref_thickness_m"] * 1e3))
        ax.set_title("filter %r" % name)
        fig.savefig(outdir / ("filter_%s.png" % _safe_name(name)),
                    bbox_inches="tight")
        plt.close(fig)

    for name in grating_names:
        spec = props.gratings.get(name)
        if spec is None:
            continue
        model_kind = spec["model"]
        if model_kind == "table":
            table = spec["table"]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for m in sorted(table):
                t = table[m]
                lam_nm = t["lam_um"] * 1e3
                ax.plot(lam_nm, t["eta_s"], label="order %d, s" % m)
                ax.plot(lam_nm, t["eta_p"], "--", label="order %d, p" % m)
            ax.set_xlabel("wavelength [nm]")
            ax.set_ylabel("diffraction efficiency")
            ax.legend(fontsize=7)
            ax.set_title("grating %r (measured table)" % name)
            fig.savefig(outdir / ("grating_%s.png" % _safe_name(name)),
                        bbox_inches="tight")
            plt.close(fig)

        elif model_kind == "bragg_kogelnik":
            # efficiency vs lambda AT BRAGG INCIDENCE: for each probe
            # wavelength, pick theta so the thin-hologram Bragg condition
            # cos(phi_g - theta) = lam/(2*Lambda) holds exactly (see
            # grating.py's bragg_kogelnik module docstring), then evaluate
            # the SAME order_efficiencies() dispatch the tracer itself
            # uses. Wavelengths with no solution (|lam/(2*Lambda)| > 1) are
            # left out of the Bragg-matched curve.
            p = spec["params"]
            period_m = 1e-3 / spec["lines_per_mm"]
            slant = np.deg2rad(float(p.get("slant_deg", 0.0)))
            phi_g = 0.5 * np.pi - slant
            lam = np.linspace(300e-9, 1100e-9, 400)
            cos_arg = lam / (2.0 * period_m)
            valid = np.abs(cos_arg) <= 1.0
            theta = np.full_like(lam, np.nan)
            theta[valid] = phi_g - np.arccos(cos_arg[valid])
            cos_i = np.cos(theta)
            eta_s, eta_p = gr.order_efficiencies(spec, lam, cos_i, [0, 1])
            eta_s1 = np.where(valid, eta_s[:, 1], np.nan)
            eta_p1 = np.where(valid, eta_p[:, 1], np.nan)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(lam[valid] * 1e9, eta_s1[valid], label="eta_s (order 1)")
            ax.plot(lam[valid] * 1e9, eta_p1[valid], "--",
                   label="eta_p (order 1)")
            ax.set_xlabel("wavelength [nm]")
            ax.set_ylabel("first-order diffraction efficiency")
            ax.set_ylim(0, 1.02)
            ax.legend(fontsize=8)
            ax.set_title("grating %r (bragg_kogelnik, at Bragg incidence)"
                        % name)
            fig.savefig(outdir / ("grating_%s.png" % _safe_name(name)),
                        bbox_inches="tight")
            plt.close(fig)

        elif model_kind == "dammann":
            tr = spec["params"]["transitions"]
            orders = list(range(-5, 6))
            effs = gr.dammann_efficiencies(tr, orders)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            vals = [effs[m] for m in orders]
            ax.bar([str(m) for m in orders], vals, color="C0")
            ax.set_xlabel("diffraction order")
            ax.set_ylabel("design efficiency")
            ax.set_title("grating %r (dammann design orders, sum=%.3f)"
                        % (name, sum(vals)))
            fig.savefig(outdir / ("grating_%s.png" % _safe_name(name)),
                        bbox_inches="tight")
            plt.close(fig)
        # 'lamellar' registry rows (if any) have no dedicated design plot
        # here -- explicit CLI lamellar specs already show up implicitly
        # through the traced rays/detector images.


def plot_audit(audit, outdir):
    per_seed = audit["per_seed"]
    rep = per_seed[0]
    sources = list(rep["sources"])
    buckets = [k for k in next(iter(rep["sources"].values()))
               if k not in ("emitted_W", "closure_error")]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(sources))
    for b in buckets:
        vals = np.array([rep["sources"][s].get(b, 0.0) * 1e3
                         for s in sources])
        if vals.sum() <= 0:
            continue
        ax.bar(sources, vals, bottom=bottom, label=b)
        bottom += vals
    for i, s in enumerate(sources):
        ax.text(i, bottom[i], "closure err %.1e"
                % rep["sources"][s]["closure_error"],
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("power [mW]")
    ax.legend(fontsize=8)
    ax.set_title("energy audit (seed 0)")
    fig.savefig(outdir / "energy_audit.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# --emit-csv: per-source (spectrum/Stokes/ledger) + per-element + system CSVs.
# Everything here reads data the trace/audit stages already produced (model
# source props, audit.json buckets, report['elements'] from
# common.element_power_table) -- no new physics, just a CSV projection.
# =============================================================================
def _avg_source_ledger(audit):
    """{source_name: {emitted_W, closure_error, <bucket>: W, ...}} averaged
    across seeds, same averaging convention as common.element_power_table."""
    per_seed = audit.get("per_seed", [])
    if not per_seed:
        return {}
    n = float(len(per_seed))
    names = list(per_seed[0]["sources"])
    out = {}
    for name in names:
        row = {"emitted_W": 0.0, "closure_error": 0.0}
        row.update({b: 0.0 for b in BUCKETS})
        for rep in per_seed:
            s = rep["sources"][name]
            row["emitted_W"] += s["emitted_W"] / n
            row["closure_error"] += s["closure_error"] / n
            for b in BUCKETS:
                row[b] += s.get(b, 0.0) / n
        out[name] = row
    return out


def _source_spectrum_rows(src, n_lambda):
    """[(wavelength_nm, rel_power, power_W)] -- the n_lambda deterministic
    equal-probability strata sample_source draws from (wavelength_strata),
    each carrying 1/n_lambda of the source's total power."""
    lam_m = wavelength_strata(src, n_lambda)
    power_w = src["power_mW"] * 1e-3
    rel = 1.0 / len(lam_m)
    return [(float(l / 1e-9), rel, power_w * rel) for l in lam_m]


def _source_stokes_row(src):
    """Source polarization spec -> (S0,S1,S2,S3,DOP), power-normalized to
    S0=1. Unpolarized sources average the two orthogonal pol_strata Jones
    vectors jones_for() returns for pol_stratum 0/1 (equal power each) --
    the same populations sample_source draws rays from -- which correctly
    collapses to DOP=0."""
    pol = src.get("polarization")
    n_pol = n_pol_strata({"polarization": pol})
    S = np.zeros(4)
    for ps in range(n_pol):
        Es, Ep = jones_for(pol, ps)
        S += np.array(stokes_from_jones(np.array(Es), np.array(Ep))) / n_pol
    dop = float(np.sqrt(S[1] ** 2 + S[2] ** 2 + S[3] ** 2) / S[0]) \
        if S[0] > 0 else 0.0
    return {"S0": float(S[0]), "S1": float(S[1]), "S2": float(S[2]),
            "S3": float(S[3]), "DOP": dop}


def emit_source_csvs(csv_emitter, model, case, audit):
    """Per-source emitted spectrum, Stokes state, and averaged ledger
    buckets/closure -- one CSV per chart kind, all sources concatenated
    (a 'source' column distinguishes rows) since there is no per-source
    chart today to share a basename with."""
    n_lambda = int(case.get("options", {}).get("nlambda", 1) or 1)
    src_bodies = [b for b in model["bodies"] if b.get("role") == "source"]

    spec_rows = []
    stokes_rows = []
    for b in src_bodies:
        name = b.get("label", b["name"])
        src = b["source"]
        for lam_nm, rel, p_w in _source_spectrum_rows(src, n_lambda):
            spec_rows.append((name, lam_nm, rel, p_w))
        st = _source_stokes_row(src)
        stokes_rows.append((name, st["S0"], st["S1"], st["S2"], st["S3"],
                            st["DOP"]))
    if spec_rows:
        csv_emitter.emit(
            "source_spectrum.csv",
            ["source", "wavelength_nm", "rel_power", "power_W"], spec_rows,
            entity="sources", chart="emitted_spectrum", units="W")
    if stokes_rows:
        csv_emitter.emit(
            "source_polarization.csv",
            ["source", "S0", "S1", "S2", "S3", "DOP"], stokes_rows,
            entity="sources", chart="polarization_state")

    ledger = _avg_source_ledger(audit)
    if ledger:
        ledger_rows = [(name, bucket, row[bucket])
                       for name, row in ledger.items() for bucket in BUCKETS]
        csv_emitter.emit(
            "source_ledger.csv", ["source", "bucket", "power_W"],
            ledger_rows, entity="sources", chart="energy_ledger", units="W")
        closure_rows = [(name, row["emitted_W"], row["closure_error"])
                        for name, row in ledger.items()]
        csv_emitter.emit(
            "source_closure.csv",
            ["source", "emitted_W", "closure_error"], closure_rows,
            entity="sources", chart="closure_summary", units="W")


def emit_element_csvs(csv_emitter, report, audit):
    """Per-element power table (report['elements'], already the seed-
    averaged common.element_power_table) + audit.json's per-seed-0
    boundary-flux and per-face tallies (diagnostic side-tables, not
    seed-averaged upstream -- reported as-is from seed 0, matching
    plot_audit's existing seed-0 convention)."""
    rows = [(label, r["power_in_W"], r["power_out_W"], r["absorbed_W"],
            r["detected_W"]) for label, r in report["elements"].items()]
    if rows:
        csv_emitter.emit(
            "element_power.csv",
            ["element", "power_in_W", "power_out_W", "absorbed_W",
             "detected_W"], rows, entity="elements",
            chart="element_power_table", units="W")

    rep0 = audit["per_seed"][0]
    flux_rows = [(label, fx.get("in_W", 0.0), fx.get("out_W", 0.0))
                for label, fx in sorted(rep0.get("element_flux_W", {}).items())]
    if flux_rows:
        csv_emitter.emit(
            "element_boundary_flux.csv", ["element", "in_W", "out_W"],
            flux_rows, entity="elements", chart="boundary_flux", units="W")

    face_rows = [(face_id, w)
                for face_id, w in sorted(rep0.get("by_surface_W", {}).items())]
    if face_rows:
        csv_emitter.emit(
            "element_per_face_power.csv", ["face", "power_W"], face_rows,
            entity="elements", chart="per_face_power", units="W")


def emit_system_csvs(csv_emitter, report, audit):
    """System-level data/energy_ledger.csv (source, bucket, power_W) +
    data/power_flow.csv (from_node, to_node, power_W): a simple two-tier
    flow graph SOURCES -> element (in) -> {ABSORBED, DETECTED,
    SURROUNDINGS} plus SOURCES -> <loss bucket> for buckets that are not
    tied to a specific element (escaped/truncated/etc — the ledger does
    not attribute those to an element)."""
    ledger = _avg_source_ledger(audit)
    ledger_rows = [(name, bucket, row[bucket])
                   for name, row in ledger.items() for bucket in BUCKETS]
    if ledger_rows:
        csv_emitter.emit(
            "energy_ledger.csv", ["source", "bucket", "power_W"],
            ledger_rows, entity="system", chart="energy_ledger", units="W")

    flow_rows = []
    for label, r in report["elements"].items():
        if r["power_in_W"] > 0:
            flow_rows.append(("SOURCES", label, r["power_in_W"]))
        if r["absorbed_W"] > 0:
            flow_rows.append((label, "ABSORBED", r["absorbed_W"]))
        if r["detected_W"] > 0:
            flow_rows.append((label, "DETECTED", r["detected_W"]))
        if r["power_out_W"] > 0:
            flow_rows.append((label, "SURROUNDINGS", r["power_out_W"]))
    bucket_totals = {}
    for row in ledger.values():
        for b in BUCKETS:
            bucket_totals[b] = bucket_totals.get(b, 0.0) + row[b]
    for b, w in bucket_totals.items():
        if w > 0:
            flow_rows.append(("SOURCES", b.upper(), w))
    if flow_rows:
        csv_emitter.emit(
            "power_flow.csv", ["from_node", "to_node", "power_W"],
            flow_rows, entity="system", chart="power_flow", units="W")


# =============================================================================
# --export-rays follow-on: spot diagrams + transverse ray/OPD fans from
# results/<case>/rays_full.npz (run_trace.py's seed-0 landing records). PNGs
# go under results/<case>/analysis/; CSVs go through the CsvEmitter with the
# index.csv `image` column pointing at the analysis PNG. No-op when the npz
# is absent (i.e. the trace ran without --export-rays).
# =============================================================================
def _spot_stats(u, v, power):
    """Centroid + RMS + geometric (100%) radius of landing points (u, v)
    [m] in the detector grid frame. RMS is power-weighted-agnostic (a plain
    geometric spot metric); centroid is the plain mean so a symmetric spot
    reports its geometric center."""
    uc = float(np.mean(u))
    vc = float(np.mean(v))
    dr = np.hypot(u - uc, v - vc)
    rms = float(np.sqrt(np.mean(dr ** 2)))
    geo = float(np.max(dr)) if len(dr) else 0.0
    return uc, vc, rms, geo


def render_spot_diagram(safe, dm, cols, adir, report, csv_emitter=None):
    label = dm["label"]
    xhat = np.asarray(dm["xhat"]); yhat = np.asarray(dm["yhat"])
    pos = cols["pos"]
    if len(pos) == 0:
        return
    u = pos @ xhat
    v = pos @ yhat
    sid = cols["source_id"].astype(int)
    lst = cols["lam_stratum"].astype(int)
    lam = cols["lam"]
    power = cols["power"]
    keys = sorted(set(zip(sid.tolist(), lst.tolist())))
    keys = [k for k in keys if int(np.sum((sid == k[0]) & (lst == k[1]))) > 0]
    if not keys:
        return
    rows = []
    csv_rows = []
    npanel = len(keys)
    ncol = min(4, npanel)
    nrow = int(np.ceil(npanel / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (s, l) in enumerate(keys):
        m = (sid == s) & (lst == l)
        uk, vk = u[m], v[m]
        lam_nm = float(np.mean(lam[m])) * 1e9
        uc, vc, rms, geo = _spot_stats(uk, vk, power[m])
        ax = axes.ravel()[i]
        ax.axis("on")
        rgb = np.clip(wavelength_rgb(lam_nm).ravel(), 0, 1)
        ax.scatter((uk - uc) * 1e6, (vk - vc) * 1e6, s=2,
                   color=rgb, edgecolors="none")
        ax.set_aspect("equal", "box")
        ax.set_title("src %d λ%d %.0f nm\nRMS %.2f µm  geo %.2f µm"
                     % (s, l, lam_nm, rms * 1e6, geo * 1e6), fontsize=8)
        ax.set_xlabel("x − centroid [µm]", fontsize=7)
        ax.set_ylabel("y − centroid [µm]", fontsize=7)
        ax.tick_params(labelsize=6)
        rows.append({"source_id": int(s), "lam_stratum": int(l),
                     "rms_radius_um": rms * 1e6, "geo_radius_um": geo * 1e6,
                     "centroid_x_um": uc * 1e6, "centroid_y_um": vc * 1e6,
                     "n_rays": int(m.sum())})
        csv_rows.append((int(s), int(l), rms * 1e6, geo * 1e6,
                         uc * 1e6, vc * 1e6, int(m.sum())))
    fig.suptitle("Spot diagram — %s" % label, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = adir / ("spot_%s.png" % safe)
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    if label in report["detectors"]:
        report["detectors"][label]["spot"] = rows
    if csv_emitter is not None:
        csv_emitter.emit(
            "spot_%s.csv" % safe,
            ["source_id", "lam_stratum", "rms_radius_um", "geo_radius_um",
             "centroid_x_um", "centroid_y_um", "n_rays"], csv_rows,
            entity=label, chart="spot_diagram", units="um",
            provenance="rays_full.npz",
            image="analysis/spot_%s.png" % safe)


def _pupil_xy(bp, xhat, yhat, power):
    """Power-weighted-centroid, max-radius-normalized transverse pupil
    coordinates (px, py) from a source/key's birth positions -- the
    convention render_ray_fans (and now render_wavefront) samples the
    exit/emit wavefront at. Returns None if every birth position
    coincides (rmax <= 0, e.g. a point-like or single-ray population)."""
    wsum = float(np.sum(power)) or 1.0
    c = (power[:, None] * bp).sum(axis=0) / wsum
    px = (bp - c) @ xhat
    py = (bp - c) @ yhat
    rmax = float(np.max(np.hypot(px, py))) if len(px) else 0.0
    if rmax <= 0:
        return None
    return px / rmax, py / rmax


def render_ray_fans(safe, dm, cols, adir, report, csv_emitter=None,
                    min_rays=50, slab=0.05):
    """Transverse tangential/sagittal ray fans + a chief-referenced OPD fan,
    using each ray's birth_pos to build a per-source normalized pupil
    coordinate (offset from the power-weighted birth centroid / max radius).
    Tangential = rays in the |pupil_x|<slab strip (Δy_landing vs pupil_y);
    sagittal the transpose; OPD = opl + straight-line ambient path to the
    landing centroid, referenced to the chief (min-pupil-radius) ray, in
    waves at each ray's own wavelength.

    Pupil normalization (birth centroid, max-radius scale) is factored
    into _pupil_xy() so render_wavefront samples the identical pupil
    coordinate convention."""
    label = dm["label"]
    xhat = np.asarray(dm["xhat"]); yhat = np.asarray(dm["yhat"])
    pos = cols["pos"]; bp = cols["birth_pos"]
    if len(pos) == 0:
        return
    sid = cols["source_id"].astype(int)
    opl = cols["opl"]; lam = cols["lam"]; power = cols["power"]
    u = pos @ xhat
    v = pos @ yhat
    sources = [s for s in sorted(set(sid.tolist()))
               if int(np.sum(sid == s)) > min_rays]
    # a source with any NaN birth_pos (children from a fully-synthetic
    # population) cannot be pupil-referenced; drop those rays per source
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    csv_rows = []
    fan_report = []
    plotted = False
    for s in sources:
        m = (sid == s) & np.isfinite(bp).all(axis=1)
        if int(m.sum()) <= min_rays:
            continue
        bpm = bp[m]
        pw = power[m]
        pupil = _pupil_xy(bpm, xhat, yhat, pw)
        if pupil is None:
            continue
        px, py = pupil
        um, vm = u[m], v[m]
        uc = float(np.mean(um)); vc = float(np.mean(vm))
        lamm = lam[m]; oplm = opl[m]
        # chief ray: smallest pupil radius
        chief = int(np.argmin(np.hypot(px, py)))
        dist = np.hypot(um - uc, vm - vc)          # ambient path to centroid
        opd = oplm + dist
        opd_ref = opd - opd[chief]
        opd_waves = opd_ref / lamm
        # tangential strip
        tan = np.abs(px) < slab
        if np.any(tan):
            order = np.argsort(py[tan])
            axes[0].plot(py[tan][order], (vm[tan][order] - vc) * 1e6,
                         marker=".", ms=3, lw=0.8, label="src %d" % s)
            axes[2].plot(py[tan][order], opd_waves[tan][order],
                         marker=".", ms=3, lw=0.8, label="src %d" % s)
            for pyi, dyi, wi in zip(py[tan], (vm[tan] - vc),
                                    opd_waves[tan]):
                csv_rows.append((int(s), "tangential", float(pyi),
                                 float(dyi * 1e6)))
                csv_rows.append((int(s), "opd", float(pyi), float(wi)))
            plotted = True
        # sagittal strip
        sag = np.abs(py) < slab
        if np.any(sag):
            order = np.argsort(px[sag])
            axes[1].plot(px[sag][order], (um[sag][order] - uc) * 1e6,
                         marker=".", ms=3, lw=0.8, label="src %d" % s)
            for pxi, dxi in zip(px[sag], (um[sag] - uc)):
                csv_rows.append((int(s), "sagittal", float(pxi),
                                 float(dxi * 1e6)))
            plotted = True
        fan_report.append({"source_id": int(s),
                           "opd_rms_waves": float(np.std(opd_waves)),
                           "opd_pv_waves": float(np.ptp(opd_waves)),
                           "n_rays": int(m.sum())})
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_title("Tangential fan"); axes[0].set_xlabel("pupil y")
    axes[0].set_ylabel("Δy landing [µm]")
    axes[1].set_title("Sagittal fan"); axes[1].set_xlabel("pupil x")
    axes[1].set_ylabel("Δx landing [µm]")
    axes[2].set_title("OPD fan (chief-ref)"); axes[2].set_xlabel("pupil y")
    axes[2].set_ylabel("OPD [waves]")
    for ax in axes:
        ax.axhline(0, color="0.7", lw=0.6, zorder=0)
        ax.legend(fontsize=7)
    fig.suptitle("Ray fans — %s" % label, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = adir / ("fan_%s.png" % safe)
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    if label in report["detectors"]:
        report["detectors"][label]["fan"] = fan_report
    if csv_emitter is not None:
        csv_emitter.emit(
            "fan_%s.csv" % safe,
            ["source_id", "fan_type", "pupil", "value"], csv_rows,
            entity=label, chart="ray_fan",
            units="Δlanding µm; opd waves", provenance="rays_full.npz",
            image="analysis/fan_%s.png" % safe)


# =============================================================================
# --ghost-analysis follow-on: rank multi-bounce reflection ("ghost") paths by
# detected power. Runs only when rays_full.npz carries the seed-0 refl_hist
# face-id history (i.e. the trace ran with --ghost-analysis). A ghost is a
# detector hit that reflected >= 2 times AND is purely specular (scattered
# rays -- roughness/diffuser/particle lobes -- are excluded: they are a
# continuous BSDF pedestal, not a discrete surface-reflection ghost). Rays are
# grouped by the ORDERED tuple of face ids they reflected off (the path
# signature, mapped to element labels via meta['face_labels']) and ranked by
# summed detected power. No new physics -- pure bookkeeping over refl_hist.
# =============================================================================
GHOST_TOP_N = 12          # bar-chart / report rows
GHOST_FOOTPRINTS = 3      # top-K paths that also get a footprint image


def _ghost_face_label(fid, face_labels):
    """Face index -> "<element>.<FaceN>" via the npz meta's face_labels list
    (written by run_trace from the scene); falls back to "face<idx>" if the
    map is missing or the index is out of range (defensive)."""
    if face_labels is not None and 0 <= fid < len(face_labels):
        return face_labels[fid]
    return "face%d" % fid


def _ghost_path_str(sig, face_labels):
    return " -> ".join(_ghost_face_label(f, face_labels) for f in sig)


def render_ghost_analysis(safe, dm, cols, adir, report, face_labels,
                          csv_emitter=None):
    """Ghost/stray-light table + footprints for one detector. Groups the
    generation>=2 specular detector hits by their refl_hist face-id path
    signature, ranks by summed detected power, writes:
      analysis/ghost_table_<safe>.png     (top-N bar chart)
      analysis/ghost_footprint_<safe>_<k>.png  (top-3 detector-frame maps)
      data/ghost_table_<safe>.csv         (via CsvEmitter)
      report['detectors'][label]['ghosts']  (top rows + totals)."""
    label = dm["label"]
    xhat = np.asarray(dm["xhat"]); yhat = np.asarray(dm["yhat"])
    pos = cols["pos"]
    hist = cols.get("refl_hist")
    if hist is None or len(pos) == 0:
        return
    gen = cols["generation"].astype(int)
    scat = cols["scattered"].astype(bool)
    power = cols["power"]
    total_det = float(np.sum(power))
    cand = (gen >= 2) & (~scat)
    if not np.any(cand) or total_det <= 0:
        return
    # group candidate rays by their ordered face-id reflection signature
    groups = {}
    for i in np.where(cand)[0]:
        sig = tuple(int(x) for x in hist[i] if x >= 0)
        if len(sig) < 2:            # defensive: history lost (should not happen)
            continue
        groups.setdefault(sig, []).append(i)
    if not groups:
        return
    rows = []
    for sig, idxs in groups.items():
        idxs = np.asarray(idxs)
        rows.append({"sig": sig, "idxs": idxs,
                     "power_W": float(np.sum(power[idxs])),
                     "n_rays": int(len(idxs)), "order": len(sig)})
    rows.sort(key=lambda r: r["power_W"], reverse=True)
    ghost_total = float(sum(r["power_W"] for r in rows))

    # --- top-N bar chart -------------------------------------------------
    top = rows[:GHOST_TOP_N]
    fig, ax = plt.subplots(figsize=(8, max(2.4, 0.42 * len(top) + 1.0)))
    ypos = np.arange(len(top))[::-1]
    ax.barh(ypos, [r["power_W"] for r in top], color="#c0392b")
    ax.set_yticks(ypos)
    ax.set_yticklabels([_ghost_path_str(r["sig"], face_labels) for r in top],
                       fontsize=7)
    ax.set_xlabel("detected power [W]")
    ax.set_title("Ghost paths (gen>=2 specular) — %s\n"
                 "ghost total %.3g W of %.3g W detected (%.2f%%)"
                 % (label, ghost_total, total_det,
                    100.0 * ghost_total / total_det), fontsize=9)
    fig.tight_layout()
    fig.savefig(adir / ("ghost_table_%s.png" % safe), bbox_inches="tight")
    plt.close(fig)

    # --- footprints of the top few paths (detector-frame 2-D histograms) -
    for rank, r in enumerate(rows[:GHOST_FOOTPRINTS], start=1):
        idxs = r["idxs"]
        u = (pos[idxs] @ xhat) * 1e3          # mm in the detector grid frame
        v = (pos[idxs] @ yhat) * 1e3
        figf, axf = plt.subplots(figsize=(4.2, 3.6))
        if len(idxs) >= 4 and np.ptp(u) > 0 and np.ptp(v) > 0:
            axf.hist2d(u, v, bins=40, weights=power[idxs], cmap="inferno")
        else:
            axf.scatter(u, v, s=6, c="#e67e22")
        axf.set_aspect("equal", "box")
        axf.set_xlabel("u [mm]", fontsize=8)
        axf.set_ylabel("v [mm]", fontsize=8)
        axf.set_title("Ghost #%d  %s\n%.3g W  %d rays"
                      % (rank, _ghost_path_str(r["sig"], face_labels),
                         r["power_W"], r["n_rays"]), fontsize=8)
        figf.tight_layout()
        figf.savefig(adir / ("ghost_footprint_%s_%d.png" % (safe, rank)),
                     bbox_inches="tight")
        plt.close(figf)

    # --- report block + CSV ---------------------------------------------
    report_rows = []
    csv_rows = []
    for r in top:
        path = _ghost_path_str(r["sig"], face_labels)
        frac = r["power_W"] / total_det
        report_rows.append({
            "path": path, "ghost_order": r["order"],
            "detected_W": r["power_W"], "fraction_of_detected": frac,
            "n_rays": r["n_rays"]})
        csv_rows.append((path, r["order"], r["power_W"], frac, r["n_rays"]))
    if label in report["detectors"]:
        report["detectors"][label]["ghosts"] = {
            "total_detected_W": total_det,
            "ghost_detected_W": ghost_total,
            "ghost_fraction": ghost_total / total_det,
            "n_paths": len(rows),
            "top": report_rows,
        }
    if csv_emitter is not None:
        csv_emitter.emit(
            "ghost_table_%s.csv" % safe,
            ["path", "ghost_order", "detected_W", "fraction_of_detected",
             "n_rays"], csv_rows,
            entity=label, chart="ghost_table", units="W",
            provenance="rays_full.npz",
            image="analysis/ghost_table_%s.png" % safe)


def render_ray_analysis(case_dir, report, csv_emitter=None):
    npz_path = case_dir / "rays_full.npz"
    if not npz_path.exists():
        return
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    face_labels = meta.get("face_labels")
    adir = case_dir / "analysis"
    adir.mkdir(exist_ok=True)
    for safe, dm in meta["detectors"].items():
        cols = {k: z["%s/%s" % (safe, k)]
                for k in ("pos", "dir", "opl", "lam", "source_id",
                          "lam_stratum", "pol_stratum", "generation",
                          "pol_mode", "power", "scattered", "coherent",
                          "birth_pos")}
        hist_key = "%s/refl_hist" % safe
        if hist_key in z.files:
            cols["refl_hist"] = z[hist_key]
        render_spot_diagram(safe, dm, cols, adir, report, csv_emitter)
        render_ray_fans(safe, dm, cols, adir, report, csv_emitter)
        if "refl_hist" in cols:
            render_ghost_analysis(safe, dm, cols, adir, report,
                                  face_labels, csv_emitter)


# =============================================================================
# --save-fields follow-on: PSF / FFT-MTF / encircled-energy analysis from a
# detector's coherent field maps (analysis_field.py; see the 'fields/<key>/
# {Ex,Ey}' layout documented above render_stokes_maps and _iter_field_keys).
# One row/panel per (source,lam,pol) gather key PLUS a final 'all' row that
# is the incoherent SUM of every key's PSF -- this detector's overall
# broadband/multi-source spot, legitimate because different gather keys
# never interfere (the same assumption render_stokes_maps' summed DOP panel
# already relies on). Silent no-op when the 'fields' group is empty/absent
# (the trace did not use --save-fields). SEED 0 ONLY (fields/ is written
# only for seed 0, see run_trace.save_detectors) -- every figure title says
# so explicitly.
# =============================================================================
def _field_key_metrics(psf, pixel_m):
    """One (coherent-key or 'all'-summed) PSF -> {scalars, mtf, radial, ee}
    -- the report's scalar metrics plus the raw curves the plotting/CSV
    helpers need, computed once and shared by both."""
    peak_w_m2 = float(psf.max()) / pixel_m ** 2 if psf.size else 0.0
    mtf = mtf2d(psf, pixel_m)
    W = mtf["mtf"].shape[1]
    H = mtf["mtf"].shape[0]
    freq = mtf["freq_cy_mm"]
    freq_y = mtf["freq_y_cy_mm"]      # sagittal axis (H != W safe)
    tan_half = mtf["tangential"][W // 2:]
    sag_half = mtf["sagittal"][H // 2:]
    radii, ee = encircled_energy(psf, pixel=pixel_m)
    r_prof, prof = radial_profile(psf, pixel=pixel_m)
    scalars = {
        "psf_peak_W_m2": peak_w_m2,
        "mtf50_tan_cy_mm": mtf50(freq, tan_half),
        "mtf50_sag_cy_mm": mtf50(freq_y, sag_half),
        "ee_r50_um": ee_radius(radii, ee, 0.5) * 1e6,
        "ee_r80_um": ee_radius(radii, ee, 0.8) * 1e6,
        "ee_r90_um": ee_radius(radii, ee, 0.9) * 1e6,
    }
    return {"scalars": scalars, "mtf": mtf, "radial": (r_prof, prof),
            "ee": (radii, ee)}


def _plot_psf_panels(panels, pixel_m, outpath, title):
    n = len(panels)
    fig, axes = plt.subplots(n, 3, figsize=(11.5, 3.3 * n), squeeze=False)
    for i, (name, psf) in enumerate(panels):
        norm, _ = normalize_psf(psf)
        H, W = psf.shape
        extent_um = [0, W * pixel_m * 1e6, 0, H * pixel_m * 1e6]
        ax0, ax1, ax2 = axes[i]
        im0 = ax0.imshow(norm, origin="lower", cmap="inferno",
                         extent=extent_um)
        ax0.set_title("%s — linear" % name, fontsize=9)
        fig.colorbar(im0, ax=ax0, fraction=0.046)
        log_img = np.log10(np.maximum(norm, 1e-6))
        im1 = ax1.imshow(log_img, origin="lower", cmap="inferno",
                         extent=extent_um, vmin=-6, vmax=0)
        ax1.set_title("%s — log10" % name, fontsize=9)
        fig.colorbar(im1, ax=ax1, fraction=0.046)
        r, prof = radial_profile(psf, pixel=pixel_m)
        pk = float(prof.max()) if len(prof) and prof.max() > 0 else 1.0
        ax2.semilogy(r * 1e6, np.maximum(prof / pk, 1e-8))
        ax2.set_ylim(1e-6, 2.0)
        ax2.set_title("%s — radial profile" % name, fontsize=9)
        ax2.set_xlabel("r [µm]")
        ax2.set_ylabel("normalized irradiance")
        for ax in (ax0, ax1):
            ax.set_xlabel("x [µm]")
            ax.set_ylabel("y [µm]")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _plot_mtf_panels(panels, metrics_by_key, outpath, title):
    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.3 * n), squeeze=False)
    for i, (name, _psf) in enumerate(panels):
        mtf = metrics_by_key[name]["mtf"]
        ax0, ax1 = axes[i]
        im0 = ax0.imshow(mtf["mtf"], origin="lower", cmap="viridis",
                         extent=[mtf["fx_cy_mm"][0], mtf["fx_cy_mm"][-1],
                                mtf["fy_cy_mm"][0], mtf["fy_cy_mm"][-1]])
        ax0.set_title("%s — 2D MTF" % name, fontsize=9)
        ax0.set_xlabel("fx [cyc/mm]")
        ax0.set_ylabel("fy [cyc/mm]")
        fig.colorbar(im0, ax=ax0, fraction=0.046)
        W = mtf["mtf"].shape[1]
        H = mtf["mtf"].shape[0]
        freq = mtf["freq_cy_mm"]
        freq_y = mtf["freq_y_cy_mm"]   # sagittal axis (H != W safe)
        ax1.plot(freq, mtf["tangential"][W // 2:], label="tangential")
        ax1.plot(freq_y, mtf["sagittal"][H // 2:], label="sagittal")
        ax1.axhline(0.5, color="0.7", lw=0.6, zorder=0)
        ax1.set_title("%s — 1D slices" % name, fontsize=9)
        ax1.set_xlabel("frequency [cyc/mm]")
        ax1.set_ylabel("MTF")
        ax1.set_ylim(0, 1.02)
        ax1.legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _plot_ee_panel(metrics_by_key, panels, outpath, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, _psf in panels:
        radii, ee = metrics_by_key[name]["ee"]
        ax.plot(radii * 1e6, ee, label=name, lw=1.0)
        for frac in (0.5, 0.8, 0.9):
            r = ee_radius(radii, ee, frac)
            if np.isfinite(r):
                ax.plot(r * 1e6, frac, "o", ms=4, color="0.2")
    ax.set_xlabel("radius [µm]")
    ax.set_ylabel("encircled energy fraction")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def render_field_analysis(h5path, adir, report, csv_emitter=None):
    """--save-fields follow-on (see the module comment above): PSF/MTF/
    encircled-energy analysis per coherent gather key + one incoherent-sum
    'all' row. No-op if the h5 has no populated 'fields' group."""
    with h5py.File(h5path) as h:
        keys = _iter_field_keys(h)
        if not keys:
            return
        attrs = dict(h.attrs)
        label = attrs["label"]
        pixel_m = float(attrs["pixel_m"])
        field_data = [(key, grp["Ex"][...], grp["Ey"][...])
                     for key, grp in keys]
    safe = label.replace(".", "_")
    adir.mkdir(parents=True, exist_ok=True)

    panels = [(key, psf_from_fields(Ex, Ey)) for key, Ex, Ey in field_data]
    if len(panels) > 1:
        summed = np.sum([p for _, p in panels], axis=0)
        panels = panels + [("all", summed)]

    metrics_by_key = {name: _field_key_metrics(psf, pixel_m)
                      for name, psf in panels}
    # headline scalars come from the dominant-POWER physical key (never the
    # synthetic 'all' summed row, which mixes strata that may not even
    # share a wavelength) -- per-key numbers (including 'all') live in
    # block["keys"].
    phys_names = [k for k, _, _ in field_data]
    dominant = max(phys_names,
                  key=lambda k: metrics_by_key[k]["scalars"]["psf_peak_W_m2"])

    block = {"keys": {name: m["scalars"]
                      for name, m in metrics_by_key.items()}}
    block.update(metrics_by_key[dominant]["scalars"])
    report.setdefault("detectors", {}).setdefault(
        label, {})["analysis"] = block

    caveat = "%s — seed 0 only" % label
    _plot_psf_panels(panels, pixel_m, adir / ("psf_%s.png" % safe),
                     "PSF — %s" % caveat)
    _plot_mtf_panels(panels, metrics_by_key, adir / ("mtf_%s.png" % safe),
                     "MTF — %s" % caveat)
    _plot_ee_panel(metrics_by_key, panels, adir / ("ee_%s.png" % safe),
                  "Encircled energy — %s" % caveat)

    if csv_emitter is not None:
        for name, m in metrics_by_key.items():
            r_prof, prof = m["radial"]
            csv_emitter.emit(
                "psf_radial_%s_%s.csv" % (safe, name),
                ["radius_um", "irradiance_W_m2"],
                zip(r_prof * 1e6, prof / pixel_m ** 2), entity=label,
                chart="psf_radial_profile", units="W/m^2",
                image="analysis/psf_%s.png" % safe)
            mtf = m["mtf"]
            W = mtf["mtf"].shape[1]
            H = mtf["mtf"].shape[0]
            freq = mtf["freq_cy_mm"]
            csv_emitter.emit(
                "mtf_slices_%s_%s.csv" % (safe, name),
                ["freq_cy_mm", "tangential", "sagittal"],
                zip(freq, mtf["tangential"][W // 2:],
                   mtf["sagittal"][H // 2:]), entity=label,
                chart="mtf_slices", units="cyc/mm",
                image="analysis/mtf_%s.png" % safe)
            radii, ee = m["ee"]
            csv_emitter.emit(
                "ee_%s_%s.csv" % (safe, name),
                ["radius_um", "ee_fraction"], zip(radii * 1e6, ee),
                entity=label, chart="encircled_energy", units="um",
                image="analysis/ee_%s.png" % safe)


# =============================================================================
# --export-rays follow-on: per-coherent-key wavefront (Zernike/Strehl)
# analysis from rays_full.npz's birth_pos/opl records -- the source-
# referenced pupil model documented in analysis.py's module docstring
# (exact for the collimated/laser benches this tracer models; NOT a true
# exit pupil for finite-conjugate imaging -- see that docstring). No-op
# unless rays_full.npz exists AND at least one (source,lam_stratum,
# pol_stratum) key has more than MIN_WAVEFRONT_RAYS COHERENT rays landing
# on a detector.
#
# FUTURE WORK (explicitly out of scope): a psf-peak-ratio Strehl (measured
# PSF peak / diffraction-limited reference PSF peak) would let --save-
# fields and --export-rays cross-check each other, but needs a reference
# (Airy/aberration-free) PSF model this module does not build -- left for
# a later pass, see future.md.
# =============================================================================
MIN_WAVEFRONT_RAYS = 200
ZERNIKE_JMAX = 15


def _zernike_panel(ax_map, ax_bar, px, py, opd_waves, coeffs_waves, title):
    lim = float(np.max(np.abs(opd_waves))) if len(opd_waves) else 1e-12
    lim = max(lim, 1e-12)
    sca = ax_map.scatter(px, py, c=opd_waves, s=6, cmap="RdBu_r",
                         vmin=-lim, vmax=lim)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_map.plot(np.cos(theta), np.sin(theta), color="0.4", lw=0.7)
    ax_map.set_aspect("equal", "box")
    ax_map.set_xlabel("pupil x")
    ax_map.set_ylabel("pupil y")
    ax_map.set_title("%s — OPD map [waves]" % title, fontsize=9)
    plt.colorbar(sca, ax=ax_map, fraction=0.046)

    jmax = len(coeffs_waves)
    js = np.arange(1, jmax + 1)
    ax_bar.bar(js, coeffs_waves, color="C0")
    ax_bar.set_xlabel("Noll j")
    ax_bar.set_ylabel("coeff [waves]")
    ax_bar.set_title("%s — Zernike (jmax=%d)" % (title, jmax), fontsize=9)
    ax_bar.set_xticks(js)
    ax_bar.tick_params(axis="x", labelsize=6)


def render_wavefront(case_dir, report, csv_emitter=None,
                     wavefront_point=None):
    npz_path = case_dir / "rays_full.npz"
    if not npz_path.exists():
        return
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    adir = case_dir / "analysis"
    jmax = ZERNIKE_JMAX

    for safe, dm in meta["detectors"].items():
        label = dm["label"]
        xhat = np.asarray(dm["xhat"], dtype=np.float64)
        yhat = np.asarray(dm["yhat"], dtype=np.float64)
        normal = np.asarray(dm["normal"], dtype=np.float64)
        cols = {k: z["%s/%s" % (safe, k)]
                for k in ("pos", "opl", "lam", "source_id", "lam_stratum",
                          "pol_stratum", "power", "coherent", "birth_pos")}
        if len(cols["pos"]) == 0:
            continue
        coh = cols["coherent"].astype(bool)
        if not np.any(coh):
            continue
        sid = cols["source_id"].astype(int)
        lst = cols["lam_stratum"].astype(int)
        pst = cols["pol_stratum"].astype(int)
        finite_bp = np.isfinite(cols["birth_pos"]).all(axis=1)
        combo_keys = sorted(set(zip(sid[coh & finite_bp].tolist(),
                                   lst[coh & finite_bp].tolist(),
                                   pst[coh & finite_bp].tolist())))

        rows_by_key = []
        panels = []
        csv_rows = []
        for (s, l, p) in combo_keys:
            m = coh & finite_bp & (sid == s) & (lst == l) & (pst == p)
            n = int(np.sum(m))
            if n <= MIN_WAVEFRONT_RAYS:
                continue
            bp = cols["birth_pos"][m]
            pw = cols["power"][m]
            pupil = _pupil_xy(bp, xhat, yhat, pw)
            if pupil is None:
                continue
            px, py = pupil
            pos_m = cols["pos"][m]
            opl_m = cols["opl"][m]
            lam_mean = float(np.mean(cols["lam"][m]))
            total_power = float(np.sum(pw))
            if wavefront_point is not None:
                x_mm, y_mm = wavefront_point
                n_off = float(np.dot(pos_m[0], normal))
                ref = (x_mm * 1e-3 * xhat + y_mm * 1e-3 * yhat
                      + n_off * normal)
            else:
                wsum = total_power or 1.0
                ref = (pw[:, None] * pos_m).sum(axis=0) / wsum
            opd, rho, theta = opd_from_rays(
                np.stack([px, py], axis=1), pos_m, opl_m, ref)
            fit = fit_zernike(rho, theta, opd, jmax=jmax, weights=pw)
            strehl = strehl_marechal(fit["rms_wavefront"], lam_mean)
            key_name = "%d_%d_%d" % (s, l, p)
            rows_by_key.append({
                "key": key_name, "source_id": s, "lam_stratum": l,
                "pol_stratum": p, "strehl_marechal": strehl,
                "rms_waves": fit["rms_wavefront"] / lam_mean,
                "pv_waves": fit["pv"] / lam_mean, "n_rays": n,
                "total_power_W": total_power,
            })
            panels.append((key_name, px, py, opd / lam_mean,
                          fit["coeffs"] / lam_mean))
            for j in range(1, jmax + 1):
                n_j, m_j = noll_to_nm(j)
                csv_rows.append((
                    s, l, p, j, fringe_index(n_j, m_j), noll_name(j),
                    fit["coeffs"][j - 1] / lam_mean,
                    fit["coeffs"][j - 1] * 1e9))

        if not rows_by_key:
            continue
        adir.mkdir(parents=True, exist_ok=True)
        dominant = max(rows_by_key, key=lambda r: r["total_power_W"])
        block = {
            "keys": rows_by_key,
            "strehl_marechal": dominant["strehl_marechal"],
            "rms_waves": dominant["rms_waves"],
            "pv_waves": dominant["pv_waves"],
            "n_rays": dominant["n_rays"],
        }
        report.setdefault("detectors", {}).setdefault(
            label, {})["wavefront"] = block

        n = len(panels)
        fig, axes = plt.subplots(n, 2, figsize=(9, 3.6 * n), squeeze=False)
        for i, (name, px, py, opd_w, coeffs_w) in enumerate(panels):
            _zernike_panel(axes[i, 0], axes[i, 1], px, py, opd_w, coeffs_w,
                          "%s key %s" % (label, name))
        fig.suptitle("Wavefront — %s (source-referenced pupil, seed 0 "
                     "only)" % label, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(adir / ("wavefront_%s.png" % safe), bbox_inches="tight")
        plt.close(fig)

        if csv_emitter is not None:
            csv_emitter.emit(
                "zernike_%s.csv" % safe,
                ["source_id", "lam_stratum", "pol_stratum", "noll_j",
                 "fringe_j", "name", "coeff_waves", "coeff_nm"], csv_rows,
                entity=label, chart="zernike_fit", units="waves; nm",
                provenance="rays_full.npz",
                image="analysis/wavefront_%s.png" % safe)


def main(argv=None):
    p = cli_specs.build_parser("post")
    args = p.parse_args(argv)
    case_dir = Path(args.case_dir)
    with open(case_dir / "case.json") as fh:
        case = json.load(fh)
    if case.get("status") != "completed":
        raise SystemExit("case status is %r (need 'completed') — run "
                         "the trace stage first" % case.get("status"))
    model = common.load_model(args.model_json)
    with open(case_dir / "audit.json") as fh:
        audit = json.load(fh)

    img = case_dir / "images"
    spec = case_dir / "spectra"
    plots = case_dir / "plots"
    for d in (img, spec, plots):
        d.mkdir(exist_ok=True)
    # --emit-csv off (the default): no data/ dir is created at all, and
    # every renderer below runs exactly as it did before this flag existed.
    csv_emitter = CsvEmitter(case_dir / "data") if args.emit_csv else None

    report = {"detectors": {}, "closure_ok": all(
        a["closure_ok"] for a in audit["per_seed"]),
        "elements": common.element_power_table(
            audit, {b["name"]: b.get("label", b["name"])
                    for b in model["bodies"]})}
    # detector bodies tagged qe_curve -> {BodyName: curve_name}; the QE
    # registry is loaded from the same default opticalproperties/ root
    # plot_materials uses. Both are optional -- a library without a
    # detector/ subtree just leaves photocurrent out of the report.
    qe_bodies = {b["name"]: b["detector"]["qe_curve"]
                 for b in model["bodies"]
                 if isinstance(b.get("detector"), dict)
                 and b["detector"].get("qe_curve")}
    detector_registry = {}
    if qe_bodies:
        try:
            detector_registry = load_optical_properties().detectors
        except Exception as exc:
            print("[post] NOTE: could not load detector QE registry (%s) — "
                  "skipping photocurrent" % exc)

    h5paths = sorted((case_dir / "detectors").glob("*.h5"))
    for i, h5path in enumerate(h5paths):
        common.progress_emit("post", 0.8 * i / max(1, len(h5paths)),
                             "detector %s" % h5path.stem,
                             case_dir=case_dir)
        render_detector(h5path, img, spec, report,
                        photometric=args.photometric,
                        spectrometer=args.spectrometer,
                        qe_bodies=qe_bodies,
                        detector_registry=detector_registry,
                        csv_emitter=csv_emitter)
        render_stokes_maps(h5path, img)

    # per-(source, detector) detected power: promotes case.json["detected"]
    # into report.json regardless of --emit-csv (deliverable independent of
    # the CSV flag); the CSV export of the same rows IS gated on the flag.
    per_source_flat = add_per_source_detected(case, report)
    if csv_emitter is not None and per_source_flat:
        csv_emitter.emit(
            "source_detector.csv",
            ["source", "detector", "lam_stratum", "pol_stratum",
             "coherent_W", "incoherent_W", "n_samples"], per_source_flat,
            entity="sources", chart="source_detector_power", units="W")

    # --export-rays follow-on: spot diagrams + ray/OPD fans (no-op unless
    # the trace stage wrote rays_full.npz). Runs regardless of --emit-csv;
    # the CSV export of the same data is gated on the emitter being present.
    render_ray_analysis(case_dir, report, csv_emitter)

    # --save-fields follow-on: PSF/MTF/encircled-energy analysis, one h5 at
    # a time (no-op per-detector unless that h5 has a populated 'fields'
    # group, i.e. the trace ran with --save-fields).
    adir = case_dir / "analysis"
    for h5path in h5paths:
        render_field_analysis(h5path, adir, report, csv_emitter)

    # --export-rays follow-on: per-coherent-key wavefront/Zernike/Strehl
    # analysis (no-op unless rays_full.npz exists AND some key clears
    # MIN_WAVEFRONT_RAYS). --wavefront-point overrides the default power-
    # weighted landing centroid image point.
    render_wavefront(case_dir, report, csv_emitter,
                     wavefront_point=args.wavefront_point)

    common.progress_emit("post", 0.8, "diagnostic plots",
                         case_dir=case_dir)
    rays = np.load(case_dir / "rays.npy")
    plot_rays_2d(rays, model, plots / "rays_xy.png",
                max_generation=args.viz_generations,
                dim_mode=args.dim_rays, dim_floor=args.dim_rays_floor)
    plot_materials(model, plots, csv_emitter=csv_emitter)
    plot_optical_elements(model, plots)
    plot_audit(audit, plots)
    if csv_emitter is not None:
        emit_source_csvs(csv_emitter, model, case, audit)
        emit_element_csvs(csv_emitter, report, audit)
        emit_system_csvs(csv_emitter, report, audit)
        csv_emitter.write_index()
    common.write_json(case_dir / "report.json", report)
    print("[post] wrote images/spectra/plots + report.json in %s"
          % case_dir, flush=True)
    common.progress_emit("post", 1.0, "report.json written",
                         case_dir=case_dir, status="completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
