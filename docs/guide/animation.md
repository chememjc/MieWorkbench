# Tracer-bead animation

`mieworkbench/core/beadanim.py` (`AnimationController` + the pure
`SegmentSet`/`precompute_segments`/`active_positions` model) + the
"Animation" toolbar in `mainwindow.py`.

## What it does

A "bead" is a sphere riding a ray polyline at the physical propagation
speed c/n. The engine records each viz segment's optical path window
(`opl0`/`opl1` = Σn·ds metres at the segment start/end, `rays.vtp` cell
arrays), so the whole animation reduces to one global simulation clock:

```
t0 = opl0 / c,  t1 = opl1 / c
bead active while t0 <= clock < t1
position = lerp(p0, p1, (clock - t0) / (t1 - t0))
```

Children inherit their parent's opl at a split, so a reflected and a
transmitted bead spawn together the instant the parent bead arrives at
the interface — no ray-id/chaining logic needed. Beads inside glass move
slower by 1/n automatically (same geometry, larger opl window).

## Toolbar

**View → Tracer Bead Animation** (also the toolbar's first button) is one
checkable action shared by the menu and toolbar (Qt keeps both in sync).
Enabling it is required before anything else is available — it also gates
on an overlay actually being loaded.

- **Play / Pause / Stop / Step** — Stop rewinds every bead to the sources
  at t = 0; Step advances exactly one frame (`speed ÷ fps` mm of vacuum
  path).
- **Bead** (radius, mm), **Speed** (mm/s — "mm of ray path per real
  second for a bead in vacuum"; beads in glass move slower by 1/n), **FPS**
  (5/10/15/24/30).
- **Bead opacity**: **Opaque** (default, bit-identical to pre-feature
  behavior) or **By power** — see below.

## Bead opacity ("By power" mode)

Opt-in; default is always-opaque. In power mode each active bead gets a
per-frame alpha from a frame-normalized log-dB power mapping over a
dynamic range R dB (toolbar spin box, default 30, range 10-60):

```
Pmax  = max absolute `power` among beads ACTIVE THIS FRAME
db    = clamp(10*log10(Pmax / P), 0, R)
alpha = max(1 - db/R, ALPHA_FLOOR)     # ALPHA_FLOOR = 0.08
```

alpha = 1 at P = Pmax and floors at `ALPHA_FLOOR` for
P ≤ Pmax/10^(R/10) — weak beads stay faintly visible rather than
vanishing.

**Leading-wavefront exemption**: beads on "leading" segments render at
alpha = 1.0 regardless of power, for their whole transit. Leading flags
are computed once at overlay load (`compute_leading_flags`) by lineage
reconstruction over the viz segments:

- roots (`opl0 == 0`, source-born) are leading;
- at a split (several children sharing one parent END key), a child
  inherits the parent's leading flag iff its birth power is ≥ 0.9× the
  brightest sibling's power (the **10% sibling rule** — a 50/50 Michelson
  split keeps *both* arms leading; a 90/10 ghost split keeps only the
  90%);
- a non-split continuation always inherits.

`ray_cap` in power mode keeps the **brightest** beads per source (leading
beads always kept, even past the cap) instead of the first-N-by-index
ordering "off" mode uses.

## Gotchas

- Legacy overlays that predate the per-segment `power` array still
  animate; power mode simply falls back to opaque on them.
- The QTimer driving playback never starts under the offscreen test
  platform — `step()`/`_tick()` are callable directly for tests
  (dialog-free-setter idiom).
- The readout shows the simulation clock (auto unit) and the
  vacuum-equivalent optical path c·t travelled so far.

![Animation toolbar](img/animation-1.png)
