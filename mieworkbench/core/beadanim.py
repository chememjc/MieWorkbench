"""beadanim - the tracer-bead animation model and controller.

A "bead" is a sphere riding a ray polyline at the physical propagation
speed c/n. The engine records each viz segment's optical path window
(opl0/opl1 = Σn·ds metres at the segment start/end, rays.vtp cell arrays)
so the whole animation reduces to one global simulation clock:

    t0 = opl0 / c,  t1 = opl1 / c
    bead active while t0 <= clock < t1
    position = lerp(p0, p1, (clock - t0) / (t1 - t0))

Children inherit their parent's opl at the split, so a reflected and a
transmitted bead spawn together the instant the parent bead arrives at
the interface -- no ray ids or chaining logic needed. Beads in glass move
slower by 1/n automatically (same geometry, larger opl window).

The playback speed setting is "mm of path per real second" for a bead in
vacuum: each frame advances the sim clock by

    dt = (speed_mm / fps) / 1000 / c        [seconds of simulated time]

so a vacuum bead covers speed/fps mm per frame and a bead inside n=1.5
glass covers 1/1.5 of that -- "the tracer beads change speed with the
refractive index" for free.

Pure numpy model (SegmentSet / precompute_segments / active_positions,
unit-testable without Qt) + a QTimer-driven AnimationController that
pushes per-frame positions into a BeadLayer (widgets/vtkview.py). The
timer never starts under the offscreen test platform; step()/_tick() are
callable directly (the dialog-free-setter idiom, docs/UI_TESTING.md).

Bead opacity (opt-in; default "off" is bit-identical to always-opaque):
------------------------------------------------------------------------
`active_positions` takes a `opacity_mode` ("off" | "power"). In "power"
mode each active bead gets a per-frame alpha from a frame-normalized
log-dB power mapping over a dynamic range R dB (default 30):

    Pmax  = max absolute `power` among beads ACTIVE THIS FRAME
    db    = clamp(10*log10(Pmax / P), 0, R)
    alpha = max(1 - db/R, ALPHA_FLOOR)          # ALPHA_FLOOR = 0.08

so alpha = 1 at P = Pmax and floors at ALPHA_FLOOR for P <= Pmax/10^(R/10).
Weak beads stay faintly visible (the floor); zero/negative power floors.

LEADING-WAVEFRONT exemption: beads on "leading" segments render alpha =
1.0 regardless of power, for their whole transit. Leading flags are
computed ONCE at overlay load (`compute_leading_flags`) by lineage
reconstruction over the viz segments:

  - Each segment has a START key = quantize(p0, opl0) and an END key =
    quantize(p1, opl1). Positions quantize to POS_QUANT (1e-9 m), optical
    path to OPL_QUANT (1e-12 m) -- the split-continuity contract makes a
    child's (start point, opl0) exactly its parent's (end point, opl1),
    and consecutive segments of one ray chain the same way.
  - Roots (opl0 == 0, source-born) are leading.
  - At a SPLIT (several child segments sharing one parent END key) a
    child inherits the parent's leading flag iff its birth `power` is
    >= 0.9 x the brightest sibling's power (so a 50/50 Michelson split
    keeps BOTH arms leading; a 90/10 ghost split keeps only the 90).
  - A non-split continuation (a single child of a key) always inherits.

ray_cap in "power" mode keeps the BRIGHTEST beads per source (leading
beads are always kept, even past the cap) instead of the first-N-by-index
of "off" mode, which stays bit-identical.

Legacy overlays that predate the per-segment `power` array still animate;
"power" mode simply falls back to opaque ("off") on them.
"""

from dataclasses import dataclass

import numpy as np

from PySide6.QtCore import QObject, QTimer, Signal

C_M_S = 299_792_458.0     # speed of light in vacuum [m/s]

# toolbar defaults (QSettings keys anim_*; core/settings.py conventions)
DEFAULT_SPEED_MM_S = 2.0
DEFAULT_FPS = 15
DEFAULT_BEAD_SIZE_MM = 1.0
DEFAULT_RAY_CAP = 300

# bead-opacity ("power" mode) constants
DEFAULT_OPACITY_MODE = "off"       # "off" (opaque, today's default) | "power"
DEFAULT_RANGE_DB = 30.0            # dynamic range of the log-dB alpha map
ALPHA_FLOOR = 0.08                # weak beads stay faintly visible
SIBLING_LEAD_BAND = 0.9           # split children >= 0.9*max sibling inherit
POS_QUANT = 1e-9                  # position quantization for lineage keys, m
OPL_QUANT = 1e-12                 # optical-path quantization for keys, m


@dataclass
class SegmentSet:
    p0: np.ndarray          # (N,3) segment start, metres
    p1: np.ndarray          # (N,3) segment end, metres
    t0: np.ndarray          # (N,) window start, seconds (= opl0/c)
    t1: np.ndarray          # (N,) window end, seconds (= opl1/c)
    rgb: np.ndarray         # (N,3) uint8 wavelength color
    source_id: np.ndarray   # (N,) int
    t_max: float            # loop period: max(t1)
    power: np.ndarray = None    # (N,) f32 segment power [W], or None (legacy)
    leading: np.ndarray = None  # (N,) bool leading-wavefront flag, or None


def precompute_segments(polydata):
    """Extract a SegmentSet from a loaded rays.vtp polydata, or None when
    the file predates the opl0/opl1 timing columns (animation must then
    be disabled -- the caller surfaces a hint)."""
    from vtkmodules.util.numpy_support import vtk_to_numpy

    if polydata is None or polydata.GetNumberOfLines() == 0:
        return None
    cell_data = polydata.GetCellData()
    arrays = {}
    for name in ("opl0", "opl1", "rgb"):
        arr = cell_data.GetArray(name)
        if arr is None:
            return None
        arrays[name] = vtk_to_numpy(arr)
    sid_arr = cell_data.GetArray("source_id")
    power_arr = cell_data.GetArray("power")

    pts = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)
    conn = vtk_to_numpy(polydata.GetLines().GetConnectivityArray())
    offsets = vtk_to_numpy(polydata.GetLines().GetOffsetsArray())
    starts = offsets[:-1]
    # 2-point line cells only (what write_vtp_polylines emits); anything
    # longer would need per-cell sub-segment timing we don't have
    if not np.all((offsets[1:] - starts) == 2):
        return None
    p0 = pts[conn[starts]]
    p1 = pts[conn[starts + 1]]

    opl0 = arrays["opl0"].astype(np.float64)
    opl1 = arrays["opl1"].astype(np.float64)
    t0 = opl0 / C_M_S
    t1 = opl1 / C_M_S
    keep = t1 > t0                        # zero-duration cells can't lerp
    if not np.any(keep):
        return None
    rgb = arrays["rgb"].reshape(-1, 3).astype(np.uint8)
    sid = (vtk_to_numpy(sid_arr).astype(np.int64) if sid_arr is not None
           else np.zeros(len(t0), dtype=np.int64))

    # leading flags need the FULL segment graph (including any zero-
    # duration cells that carry lineage) but the SegmentSet only keeps the
    # animatable ones -- compute over all, then mask to `keep`.
    power = leading = None
    if power_arr is not None:
        power = vtk_to_numpy(power_arr).astype(np.float64)
        leading = compute_leading_flags(p0, p1, opl0, opl1, power)
        power = power[keep]
        leading = leading[keep]

    return SegmentSet(p0=p0[keep], p1=p1[keep], t0=t0[keep], t1=t1[keep],
                      rgb=rgb[keep], source_id=sid[keep],
                      t_max=float(t1[keep].max()),
                      power=power, leading=leading)


def _quant_key(x, y, z, opl):
    """Integer lineage key: position quantized to POS_QUANT, optical path
    to OPL_QUANT. Fold -0.0 -> 0 so signed zeros don't split a key."""
    def q(v, step):
        k = int(round(float(v) / step))
        return k + 0            # normalize -0 -> 0
    return (q(x, POS_QUANT), q(y, POS_QUANT), q(z, POS_QUANT),
            q(opl, OPL_QUANT))


def compute_leading_flags(p0, p1, opl0, opl1, power):
    """(N,) bool: which viz segments ride the LEADING wavefront.

    Reconstructs the ray lineage from the split-continuity contract: a
    segment's START key = quantize(p0, opl0) equals its parent's END key =
    quantize(p1, opl1). Roots (opl0 == 0) are leading. A split is several
    children sharing one parent END key; a child inherits the parent's
    flag iff its birth `power` >= SIBLING_LEAD_BAND x the brightest
    sibling's power (single-child continuations always inherit). O(n) via
    dict keys; processed in opl0 order so parents decide before children.
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(-1, 3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 3)
    opl0 = np.asarray(opl0, dtype=np.float64).reshape(-1)
    opl1 = np.asarray(opl1, dtype=np.float64).reshape(-1)
    power = np.asarray(power, dtype=np.float64).reshape(-1)
    n = len(opl0)
    leading = np.zeros(n, dtype=bool)
    if n == 0:
        return leading

    start_key = [None] * n
    end_map = {}                       # END key -> parent segment index
    start_groups = {}                  # START key -> [sibling indices]
    for i in range(n):
        sk = _quant_key(p0[i, 0], p0[i, 1], p0[i, 2], opl0[i])
        ek = _quant_key(p1[i, 0], p1[i, 1], p1[i, 2], opl1[i])
        start_key[i] = sk
        end_map[ek] = i                # unique parent per end key
        start_groups.setdefault(sk, []).append(i)

    order = np.argsort(opl0, kind="stable")   # parents (smaller opl0) first
    opl_key0 = int(round(0.0 / OPL_QUANT))
    for i in order:
        if int(round(opl0[i] / OPL_QUANT)) == opl_key0:
            leading[i] = True          # source-born root
            continue
        parent = end_map.get(start_key[i])
        if parent is None or not leading[parent]:
            continue                   # orphan or dim-parent lineage
        siblings = start_groups[start_key[i]]
        if len(siblings) == 1:
            leading[i] = True          # single continuation inherits
        else:
            maxp = max(power[s] for s in siblings)
            leading[i] = power[i] >= SIBLING_LEAD_BAND * maxp
    return leading


def active_positions(seg, clock, ray_cap=0, opacity_mode="off",
                     range_db=DEFAULT_RANGE_DB):
    """(points_m (M,3), rgb (M,3) uint8, alpha) of the beads alive at
    `clock` seconds.

    ray_cap > 0 bounds the SIMULTANEOUSLY DRAWN beads per source -- an
    honest render cap, not a trace cap, since viz segments carry no
    per-ray identity. In "off" mode the cap is stable first-N by segment
    index (today's behavior, bit-identical) and `alpha` is None. In
    "power" mode (only when the overlay carries `power`) the cap keeps the
    BRIGHTEST beads per source plus every leading-flagged one, and `alpha`
    is an (M,) float per the frame-normalized log-dB power map with a
    leading-wavefront exemption (see the module docstring)."""
    power_mode = opacity_mode == "power" and seg.power is not None
    mask = (seg.t0 <= clock) & (clock < seg.t1)
    idx = np.nonzero(mask)[0]
    if ray_cap and len(idx):
        kept = []
        for sid in np.unique(seg.source_id[idx]):
            of_source = idx[seg.source_id[idx] == sid]
            if len(of_source) <= ray_cap:
                kept.append(of_source)
            elif power_mode:
                lead = seg.leading[of_source]
                lead_idx = of_source[lead]
                rest = of_source[~lead]
                order = np.argsort(-seg.power[rest], kind="stable")
                need = max(0, ray_cap - len(lead_idx))
                kept.append(np.concatenate([lead_idx, rest[order[:need]]]))
            else:
                kept.append(of_source[:ray_cap])
        idx = np.concatenate(kept)
    if not len(idx):
        empty_alpha = np.zeros((0,)) if power_mode else None
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8),
                empty_alpha)
    f = ((clock - seg.t0[idx]) / (seg.t1[idx] - seg.t0[idx]))[:, None]
    points = seg.p0[idx] + f * (seg.p1[idx] - seg.p0[idx])
    alpha = None
    if power_mode:
        alpha = _power_alpha(seg.power[idx], seg.leading[idx], range_db)
    return points, seg.rgb[idx], alpha


def _power_alpha(power, leading, range_db):
    """(M,) float alpha for the active beads: frame-normalized log-dB map
    over `range_db`, floored at ALPHA_FLOOR, leading beads pinned to 1.0.
    Zero/negative power floors (clipped to a tiny positive for the log)."""
    r = max(1e-6, float(range_db))
    p = np.asarray(power, dtype=np.float64)
    pmax = p.max() if len(p) else 1.0
    if pmax <= 0.0:
        pmax = 1.0
    tiny = pmax * 1e-30
    db = 10.0 * np.log10(pmax / np.clip(p, tiny, None))
    db = np.clip(db, 0.0, r)
    alpha = np.maximum(1.0 - db / r, ALPHA_FLOOR)
    alpha[np.asarray(leading, dtype=bool)] = 1.0
    return alpha


def format_sim_time(seconds):
    """Auto-unit time label: fs/ps/ns/µs/ms picked by magnitude."""
    for factor, unit in ((1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns"),
                         (1e-12, "ps")):
        if seconds >= factor:
            return "%.2f %s" % (seconds / factor, unit)
    return "%.1f fs" % (seconds / 1e-15)


class AnimationController(QObject):
    """Owns the playback clock and QTimer; pushes per-frame bead
    positions into a BeadLayer. States: 'stopped' (clock 0, beads at the
    sources), 'playing', 'paused'. Play loops at t_max until Stop."""

    frameAdvanced = Signal(float, float)   # (sim clock [s], path [mm])
    stateChanged = Signal(str)             # 'playing' | 'paused' | 'stopped'
    availabilityChanged = Signal(bool)     # segments with timing loaded?

    def __init__(self, layer=None, render=None, parent=None):
        super().__init__(parent)
        self._layer = layer                # BeadLayer (or None in tests)
        self._render = render or (lambda: None)
        self._seg = None
        self._clock = 0.0
        self._state = "stopped"
        self.speed_mm_s = DEFAULT_SPEED_MM_S
        self.fps = DEFAULT_FPS
        self.bead_size_mm = DEFAULT_BEAD_SIZE_MM
        self.ray_cap = DEFAULT_RAY_CAP
        self.bead_opacity_mode = DEFAULT_OPACITY_MODE
        self.bead_opacity_range_db = DEFAULT_RANGE_DB
        self.enabled = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # -- data ----------------------------------------------------------------
    def set_segments(self, seg):
        """New overlay (or None for a legacy/removed/stale one): reset the
        clock, park beads at the sources, tell the shell whether the
        transport should be enabled."""
        self._seg = seg
        self._stop_internal()
        self.availabilityChanged.emit(seg is not None)

    def has_segments(self):
        return self._seg is not None

    # -- settings --------------------------------------------------------------
    def apply_settings(self, bead_size_mm=None, speed_mm_s=None, fps=None,
                       ray_cap=None, enabled=None, bead_opacity_mode=None,
                       bead_opacity_range_db=None):
        opacity_dirty = False
        if bead_size_mm is not None:
            self.bead_size_mm = max(0.01, float(bead_size_mm))
            if self._layer is not None:
                self._layer.set_radius_m(self.bead_size_mm / 1000.0)
        if speed_mm_s is not None:
            self.speed_mm_s = max(1e-6, float(speed_mm_s))
        if fps is not None:
            self.fps = max(1, int(fps))
            if self._timer.isActive():
                self._timer.setInterval(int(round(1000.0 / self.fps)))
        if ray_cap is not None:
            self.ray_cap = max(0, int(ray_cap))
        if bead_opacity_mode is not None:
            mode = str(bead_opacity_mode)
            self.bead_opacity_mode = mode if mode in ("off", "power") \
                else "off"
            opacity_dirty = True
        if bead_opacity_range_db is not None:
            self.bead_opacity_range_db = max(1.0, float(bead_opacity_range_db))
            opacity_dirty = True
        if enabled is not None:
            self.enabled = bool(enabled)
            if not self.enabled:
                self.pause()
            if self._layer is not None:
                self._layer.set_visible(self.enabled)
            self._push_frame()
        elif opacity_dirty:
            self._push_frame()         # re-tint the current frame in place
        self._render()

    def power_available(self):
        """True when the loaded overlay carries per-segment power (so the
        'power' opacity mode is live rather than falling back to opaque)."""
        return self._seg is not None and self._seg.power is not None

    @property
    def dt_per_frame(self):
        """Sim-clock seconds per frame: a vacuum bead advances
        speed_mm_s/fps mm of geometric path each frame."""
        return (self.speed_mm_s / self.fps) / 1000.0 / C_M_S

    # -- transport --------------------------------------------------------------
    def play(self):
        if self._seg is None or not self.enabled:
            return
        self._state = "playing"
        self.stateChanged.emit(self._state)
        from ..widgets.vtkview import is_offscreen
        if not is_offscreen():
            self._timer.start(int(round(1000.0 / self.fps)))

    def pause(self):
        self._timer.stop()
        if self._state == "playing":
            self._state = "paused"
            self.stateChanged.emit(self._state)

    def stop(self):
        self._stop_internal()
        self.stateChanged.emit(self._state)

    def _stop_internal(self):
        self._timer.stop()
        self._state = "stopped"
        self._clock = 0.0
        self._push_frame()

    def step(self):
        """Single-step: advance exactly one frame (speed/fps mm of vacuum
        path). Usable from any state; pauses playback."""
        if self._seg is None or not self.enabled:
            return
        self.pause()
        self._tick()

    def _tick(self):
        if self._seg is None:
            return
        self._clock += self.dt_per_frame
        if self._clock >= self._seg.t_max:
            self._clock = 0.0            # loop until Stop
        self._push_frame()

    def _push_frame(self):
        clock = self._clock
        if self._seg is not None and self._layer is not None \
                and self.enabled:
            points, rgb, alpha = active_positions(
                self._seg, clock, self.ray_cap,
                self.bead_opacity_mode, self.bead_opacity_range_db)
            self._layer.update_beads(points, rgb, alpha)
        self._render()
        self.frameAdvanced.emit(clock, clock * C_M_S * 1000.0)

    @property
    def clock(self):
        return self._clock

    @property
    def state(self):
        return self._state
