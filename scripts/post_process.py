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
                    qe_bodies=None, detector_registry=None):
    with h5py.File(h5path) as h:
        cube = h["spectral_cube_mean"][...]
        mask = h["mask"][...]
        attrs = dict(h.attrs)
        std = h["spectral_cube_std"][...] \
            if "spectral_cube_std" in h else None
    label = attrs["label"]
    safe = label.replace(".", "_")
    lam_lo, lam_hi = attrs["lam_lo_m"], attrs["lam_hi_m"]
    pixel_m = attrs["pixel_m"]
    pixel_area = pixel_m ** 2

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
    extent_mm = [0, attrs["W"] * pixel_m / 1e-3,
                 0, attrs["H"] * pixel_m / 1e-3]

    # sRGB wavelength-colored image (clip the noise for display)
    rgb = spectral_cube_to_srgb(np.maximum(cube, 0.0), lam_lo, lam_hi)
    rgb[~mask] = 0.12
    fig, ax = plt.subplots(figsize=(8, 8), dpi=max(
        128, int(attrs["W"]) // 8))
    ax.imshow(rgb, origin="lower", extent=extent_mm)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
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
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("%s — irradiance (%s)" % (label, tag))
        fig.savefig(outdir_img / ("det_%s_%s.png" % (safe, tag)),
                    bbox_inches="tight")
        plt.close(fig)

    # profiles through the peak
    if np.any(irr > 0):
        iy, ix = np.unravel_index(np.argmax(irr), irr.shape)
        fig, axes = plt.subplots(2, 1, figsize=(9, 7))
        xmm = (np.arange(irr.shape[1]) + 0.5) * pixel_m / 1e-3
        ymm = (np.arange(irr.shape[0]) + 0.5) * pixel_m / 1e-3
        axes[0].plot(xmm, irr[iy], lw=0.8)
        axes[0].set_title("%s — horizontal profile through peak "
                          "(row %d)" % (label, iy))
        axes[0].set_xlabel("x [mm]")
        axes[1].plot(ymm, irr[:, ix], lw=0.8)
        axes[1].set_title("vertical profile through peak (col %d)" % ix)
        axes[1].set_xlabel("y [mm]")
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
    has_e_ray = False
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
            is_e = pol_mode[i] == 1
            has_e_ray = has_e_ray or is_e
            ax.plot([rays[i, 3] * 1e3, rays[i, 6] * 1e3],
                    [rays[i, 4] * 1e3, rays[i, 7] * 1e3],
                    color=E_RAY_COLOR if is_e else colors[i],
                    linestyle="--" if is_e else "-",
                    alpha=float(alpha[i]), lw=0.9 if is_e else 0.5,
                    zorder=3 if is_e else 2)
    if has_e_ray:
        ax.legend(handles=[Line2D([0], [0], color=E_RAY_COLOR, ls="--",
                                  lw=1.2, label="e-ray")],
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


def plot_materials(model, outdir):
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
                        detector_registry=detector_registry)
        render_stokes_maps(h5path, img)
    common.progress_emit("post", 0.8, "diagnostic plots",
                         case_dir=case_dir)
    rays = np.load(case_dir / "rays.npy")
    plot_rays_2d(rays, model, plots / "rays_xy.png",
                max_generation=args.viz_generations,
                dim_mode=args.dim_rays, dim_floor=args.dim_rays_floor)
    plot_materials(model, plots)
    plot_optical_elements(model, plots)
    plot_audit(audit, plots)
    common.write_json(case_dir / "report.json", report)
    print("[post] wrote images/spectra/plots + report.json in %s"
          % case_dir, flush=True)
    common.progress_emit("post", 1.0, "report.json written",
                         case_dir=case_dir, status="completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
