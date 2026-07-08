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


@dataclass
class SegmentSet:
    p0: np.ndarray          # (N,3) segment start, metres
    p1: np.ndarray          # (N,3) segment end, metres
    t0: np.ndarray          # (N,) window start, seconds (= opl0/c)
    t1: np.ndarray          # (N,) window end, seconds (= opl1/c)
    rgb: np.ndarray         # (N,3) uint8 wavelength color
    source_id: np.ndarray   # (N,) int
    t_max: float            # loop period: max(t1)


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

    t0 = arrays["opl0"].astype(np.float64) / C_M_S
    t1 = arrays["opl1"].astype(np.float64) / C_M_S
    keep = t1 > t0                        # zero-duration cells can't lerp
    if not np.any(keep):
        return None
    rgb = arrays["rgb"].reshape(-1, 3).astype(np.uint8)
    sid = (vtk_to_numpy(sid_arr).astype(np.int64) if sid_arr is not None
           else np.zeros(len(t0), dtype=np.int64))
    return SegmentSet(p0=p0[keep], p1=p1[keep], t0=t0[keep], t1=t1[keep],
                      rgb=rgb[keep], source_id=sid[keep],
                      t_max=float(t1[keep].max()))


def active_positions(seg, clock, ray_cap=0):
    """(points_m (M,3), rgb (M,3) uint8) of the beads alive at `clock`
    seconds. ray_cap > 0 bounds the SIMULTANEOUSLY DRAWN beads per source
    (stable first-N by segment index) -- an honest render cap, not a
    trace cap, since viz segments carry no per-ray identity."""
    mask = (seg.t0 <= clock) & (clock < seg.t1)
    idx = np.nonzero(mask)[0]
    if ray_cap and len(idx):
        kept = []
        for sid in np.unique(seg.source_id[idx]):
            of_source = idx[seg.source_id[idx] == sid]
            kept.append(of_source[:ray_cap])
        idx = np.concatenate(kept)
    if not len(idx):
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))
    f = ((clock - seg.t0[idx]) / (seg.t1[idx] - seg.t0[idx]))[:, None]
    points = seg.p0[idx] + f * (seg.p1[idx] - seg.p0[idx])
    return points, seg.rgb[idx]


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
                       ray_cap=None, enabled=None):
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
        if enabled is not None:
            self.enabled = bool(enabled)
            if not self.enabled:
                self.pause()
            if self._layer is not None:
                self._layer.set_visible(self.enabled)
            self._push_frame()
        self._render()

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
            points, rgb = active_positions(self._seg, clock, self.ray_cap)
            self._layer.update_beads(points, rgb)
        self._render()
        self.frameAdvanced.emit(clock, clock * C_M_S * 1000.0)

    @property
    def clock(self):
        return self._clock

    @property
    def state(self):
        return self._state
