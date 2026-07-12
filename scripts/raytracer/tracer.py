# =============================================================================
# tracer.py — the propagation loop.
#
# Per step, for a RayBatch:
#   1. nearest-hit intersection against all scene faces
#   2. Beer-Lambert bulk absorption + dispersion phase over the segment
#   3. escaped rays credited to the ledger
#   4. hits dispatched by face role:
#        detector screen -> gather sample (coherent) / direct deposit
#                           (incoherent), then thin-screen mirror/absorb/
#                           transmit WITHOUT refraction
#        grating face    -> diffraction orders (grating.py)
#        optic face      -> mirror/absorbance + Fresnel/TMM split into
#                           reflected + refracted children
#   5. children below the relative power floor or past the generation cap
#      are killed with their power credited to the ledger
#
# mirror/absorbance surface model (documented design choice): 'mirror' r is
# an idealized achromatic partial reflector, 'absorbance' a eats the stated
# fraction of the remainder, and what is left interacts physically
# (Fresnel/TMM). Per polarization the reflected child gets amplitude
#   sqrt(r + (1-r)(1-a)|r_F|^2) * exp(i arg(r_F))
# (power-exact, phase from the physical coefficient) and the transmitted
# child gets sqrt((1-r)(1-a)) t_F. Surface absorption is the exact power
# difference, so the audit closes by construction. r=0, a=0 reduces to pure
# Fresnel/TMM identically, phases included.
#
# Reflection generations: 'generation' counts REFLECTIONS (spec: "up to the
# 6th, user adjustable"); transmissions do not increment it.
# =============================================================================
import numpy as np

from . import fresnel as fr
from . import thinfilm as tf
from . import grating as grating_mod
from . import roughness as rough_mod
from . import differentials as diff_mod
from .rays import RayBatch, AMBIENT, HIST_DEPTH
from .audit import PowerLedger


def _kill_differentials(batch):
    """Mark a child batch's ray differentials as lost (NaN): the gather
    falls back to the source-referenced patch area for those samples.
    Used for children whose differential transport is not implemented
    (grating orders, scattered lobes, birefringent o/e splits)."""
    if batch is not None and len(batch) > 0 and batch.has_differentials:
        for name in RayBatch._DIFF_SLOTS:
            getattr(batch, name)[:] = np.nan


class TraceConfig:
    def __init__(self, max_reflections=6, power_floor=1e-4, n_lambda=5,
                 rays=int(1e5), seed=0, viz_rays=500, batch_size=1 << 20,
                 rough_fresnel="micro", export_rays=False,
                 track_history=False, track_time=False):
        self.max_reflections = max_reflections
        self.power_floor = power_floor
        self.n_lambda = n_lambda
        self.rays = int(rays)
        self.seed = seed
        # export_rays: capture per-ray landing records at every detector
        # event into DetectorGrid.ray_records (--export-rays; seed 0 only).
        # Purely diagnostic — the splat/gather math is untouched.
        self.export_rays = export_rays
        # track_history: allocate RayBatch.refl_hist on the primaries and
        # record the face id of every reflection event (ghost/stray-light
        # analysis; --ghost-analysis, seed 0 only). Zero overhead when off
        # (the slot stays None everywhere). Purely diagnostic.
        self.track_history = track_history
        # track_time: allocate RayBatch.gopl/gdd_acc on the primaries and
        # advance the GROUP optical path + accumulated GDD beside opl in
        # step() (pulsed-optics time-domain core). Zero RNG use; when off
        # NO new code executes and every existing accumulator (opl, the
        # ledger, detector cubes) is bit-identical. Diagnostic/analysis
        # only — never feeds the splat/gather.
        self.track_time = track_time
        # viz_rays: int cap for every source, or {source_id: cap} computed
        # from --viz-density (rays per mm^2 of emit area) upstream
        self.viz_rays = viz_rays
        self.batch_size = batch_size
        # 'micro': roughness-lobe Fresnel evaluated at the microfacet-local
        # incidence angle, per polarization (physical). 'macro': legacy
        # nominal-angle scalar average (README §6.2 item 2), kept for A/B.
        self.rough_fresnel = rough_fresnel


class VizStore:
    """Polyline segments for rays whose viz_flag is set (the first
    viz_rays primaries per source; children inherit the flag)."""

    def __init__(self):
        # (M, 13): source_id, lam, power, x0..z0, x1..z1, pol_mode,
        # rel_power (power/birth_power in [0,1] -- drives attenuation
        # dimming in the renderers), opl0, opl1 (optical path Σn·ds in
        # metres at the segment start/end -- t = opl/c drives the
        # tracer-bead animation; escaped stubs get a synthetic opl1)
        self.chunks = []

    @staticmethod
    def flag_primaries(batch, cap):
        """cap: int (same for every source) or {source_id: int} computed
        from the viz ray density upstream."""
        for sid in np.unique(batch.source_id):
            c = cap.get(int(sid), 500) if isinstance(cap, dict) else cap
            idx = np.where(batch.source_id == sid)[0][:c]
            batch.viz_flag[idx] = True

    def add(self, batch, p1, opl0, opl1):
        m = batch.viz_flag
        if not np.any(m):
            return
        power = batch.power[m]
        birth = batch.birth_power[m]
        rel = np.zeros_like(power)
        np.divide(power, birth, out=rel, where=birth > 0)
        np.clip(rel, 0.0, 1.0, out=rel)
        self.chunks.append(np.concatenate([
            batch.source_id[m, None].astype(np.float64),
            batch.lam[m, None], power[:, None],
            batch.pos[m], p1[m],
            batch.pol_mode[m, None].astype(np.float64),
            rel[:, None],
            opl0[m, None], opl1[m, None]], axis=1))

    def as_array(self):
        return np.concatenate(self.chunks) if self.chunks \
            else np.zeros((0, 13))


class TraceResult:
    def __init__(self, detectors, ledger, viz, source_names,
                 path_tally=None):
        self.detectors = detectors        # face_id -> DetectorGrid
        self.ledger = ledger
        self.viz = viz
        self.source_names = source_names
        # path_tally: {body label: Sum(power * ds) [W*m]} power-weighted
        # bulk path per non-ambient body (track_time only; the per-body
        # GDD budget of a later phase divides by detected power to get
        # the mean glass path). Diagnostic, zero RNG use, empty when
        # time tracking is off.
        self.path_tally = path_tally if path_tally is not None else {}


class Tracer:
    def __init__(self, scene, config, detectors, grating_module=None,
                 roughness_module=None, particle_medium=None):
        """detectors: {face_id: DetectorGrid} prepared by the caller.
        grating_module/roughness_module: optional hooks (P3-F) exposing
        apply(...) kernels; particle_medium: optional hook (P3-E)."""
        self.scene = scene
        self.cfg = config
        self.detectors = detectors
        self.grating = grating_module
        self.rough = roughness_module
        self.particles = particle_medium
        self.ledger = PowerLedger(len(scene.sources))
        self.viz = VizStore()
        self.rng = np.random.default_rng(config.seed)
        # per-body power-weighted bulk path [W*m], keyed by body label
        # (same key space as the ledger's element flux tallies). Filled
        # only under cfg.track_time; see TraceResult.path_tally.
        self.path_tally = {}

    # ------------------------------------------------------------------
    def run(self, batches):
        """batches: list of freshly sampled source RayBatches."""
        queue = list(batches)
        for b in queue:
            VizStore.flag_primaries(b, self.cfg.viz_rays)
            if self.cfg.track_history and b.refl_hist is None:
                b.alloc_history()
            if self.cfg.track_time and b.gopl is None:
                b.alloc_time()
        # a hard iteration cap guards against pathological loops; with the
        # generation cap the loop terminates naturally well before this
        for _ in range(64 * (self.cfg.max_reflections + 2)):
            if not queue:
                break
            batch = queue.pop()
            if len(batch) == 0:
                continue
            children = self.step(batch)
            if children is not None and len(children) > 0:
                # split oversized batches to bound memory
                if len(children) > self.cfg.batch_size:
                    idx = np.arange(len(children))
                    for part in np.array_split(
                            idx, len(children) // self.cfg.batch_size + 1):
                        queue.append(children.select(part))
                else:
                    queue.append(children)
        if queue:
            # drain leftovers into the ledger so closure still holds
            for b in queue:
                self.ledger.credit("truncated_generation", b.source_id,
                                   b.power)
        names = [self.scene.bodies[i].label for i, _ in self.scene.sources]
        return TraceResult(self.detectors, self.ledger, self.viz, names,
                           path_tally=self.path_tally)

    # ------------------------------------------------------------------
    def step(self, batch):
        scene = self.scene
        n = len(batch)
        t, fid = scene.intersect(batch.pos, batch.dir)

        # ---- particle medium interception (P3-E hook) ----
        # scattered/collided children start fresh from their scatter point
        # and are merged into this step's children at the end
        particle_children = None
        if self.particles is not None:
            t, fid, batch, particle_children = \
                self.particles.intercept(self, batch, t, fid)
            n = len(batch)
            if n == 0:
                return self._apply_floors(particle_children) \
                    if particle_children is not None else None

        hit = fid >= 0

        # ---- bulk absorption + phase over the segment ----
        med = batch.current_medium()
        seg = np.where(hit, t, 0.0)       # escaped rays: no traversal loss
        n_med = np.empty(n, dtype=np.complex128)
        alpha_add = np.zeros(n)
        for m in np.unique(med):
            sel = med == m
            n_med[sel] = scene.medium_index(int(m), batch.lam[sel])
            # bulk spectral filters: additive absorption coefficient from
            # the body's filter table (Beer-Lambert; energy lands in
            # absorbed_bulk with correct path-length scaling)
            if m >= 0 and scene.bodies[int(m)].filter is not None:
                alpha_add[sel] = scene.bodies[int(m)].filter_alpha(
                    batch.lam[sel])
        alpha = 4.0 * np.pi * np.imag(n_med) / batch.lam + alpha_add
        trans = np.exp(-np.clip(alpha * seg, 0.0, 700.0))
        p_before = batch.power
        batch.Es *= np.sqrt(trans)
        batch.Ep *= np.sqrt(trans)
        absorbed = p_before * (1.0 - trans)
        if np.any(absorbed > 0):
            for m in np.unique(med[absorbed > 0]):
                sel = (med == m) & (absorbed > 0)
                name = "ambient" if m == AMBIENT \
                    else scene.bodies[int(m)].label
                self.ledger.credit("absorbed_bulk", batch.source_id[sel],
                                   absorbed[sel], where=name)
        # gather samples must carry the phase at the segment START — the
        # gather kernel itself adds k*(n r) for the final leg; adding the
        # segment here AND in the kernel double-counts a per-ray-different
        # path and scrambles the Fermat phase agreement at foci
        start_opl = batch.opl.copy()
        # e-rays inside a uniaxial crystal carry their direction-dependent
        # phase index in n_eff (set at the entry interface); everyone else
        # uses the medium's scalar index. Absorption above already used
        # Im(n_med) — the o-ray k is the documented approximation for
        # absorbing crystals.
        n_phase = np.where(batch.n_eff > 0.0, batch.n_eff, np.real(n_med))
        batch.opl += n_phase * seg

        # ---- time-domain accumulators (track_time only) ----
        # GROUP optical path Sum(n_g * ds) and accumulated GDD
        # Sum((phi2/L) * ds) advance beside opl; frozen directional group
        # indices (n_g_eff, crystal e/sheet rays) override the medium
        # scalar exactly like n_eff does for the phase. Also tallies the
        # power-weighted bulk path per body (GDD budget, diagnostic).
        # STRICTLY additive: nothing here touches opl/Es/Ep/ledger buckets
        # or the RNG, and none of it runs when track_time is off.
        if self.cfg.track_time and batch.gopl is not None:
            n_grp = np.empty(n, dtype=np.float64)
            gdd_l = np.empty(n, dtype=np.float64)
            for m in np.unique(med):
                sel = med == m
                n_grp[sel] = scene.medium_group_index(int(m),
                                                      batch.lam[sel])
                gdd_l[sel] = scene.medium_gdd_per_length(int(m),
                                                         batch.lam[sel])
                if m >= 0:
                    # p_before = power at the segment start (pre bulk
                    # absorption); escaped rays have seg == 0
                    lbl = scene.bodies[int(m)].label
                    self.path_tally[lbl] = self.path_tally.get(lbl, 0.0) \
                        + float(np.sum(p_before[sel] * seg[sel]))
            batch.gopl += np.where(batch.n_g_eff > 0.0, batch.n_g_eff,
                                   n_grp) * seg
            batch.gdd_acc += gdd_l * seg

        # ray differentials: the wavefront patch area at the segment START
        # (= the last interaction point) is what the gather wavelet needs;
        # then transfer the position differential over the segment for the
        # next interaction
        if batch.has_differentials:
            start_dA = diff_mod.patch_area(batch.dPdx, batch.dPdy,
                                           batch.dir)
            batch.dPdx = batch.dPdx + seg[:, None] * batch.dDdx
            batch.dPdy = batch.dPdy + seg[:, None] * batch.dDdy
        else:
            start_dA = None

        # ---- viz segments ----
        p1 = batch.pos + np.where(hit, t, 0.25)[:, None] * batch.dir
        # per-segment optical-path window for the bead animation:
        # opl0 = value at the segment start (start_opl above), opl1 = the
        # advanced batch.opl. Escaped rays advanced by zero (seg=0), so
        # synthesize the drawn 0.25 m stub's optical path into a FRESH
        # array -- batch.opl itself (the coherence path) is never touched.
        viz_opl1 = np.where(hit, batch.opl, start_opl + n_phase * 0.25)
        self.viz.add(batch, p1, start_opl, viz_opl1)

        # ---- escaped ----
        if np.any(~hit):
            self.ledger.credit("escaped", batch.source_id[~hit],
                               batch.power[~hit])

        if not np.any(hit):
            return self._apply_floors(particle_children) \
                if particle_children is not None else None

        # keep segment-start state for gather samples
        start_pos = batch.pos.copy()
        live = batch.select(hit)
        live_t = t[hit]
        live_fid = fid[hit]
        live_start = start_pos[hit]
        live_start_opl = start_opl[hit]
        live_nmed = n_med[hit]
        live_dA = start_dA[hit] if start_dA is not None else None
        live.pos = live.pos + live_t[:, None] * live.dir

        children = []
        for f in np.unique(live_fid):
            sel = live_fid == f
            grp = live.select(sel)
            grp_start = live_start[sel]
            grp_start_opl = live_start_opl[sel]
            grp_nmed = live_nmed[sel]
            grp_dA = live_dA[sel] if live_dA is not None else None
            body = scene.body_of_face(int(f))

            if int(f) in self.detectors:
                self._detector_event(int(f), grp, grp_start,
                                     grp_start_opl, grp_nmed, grp_dA)
                if int(f) in scene.extra_detector_faces \
                        and body.role == "optic":
                    # extra detector on an optical face: record the field,
                    # then interact physically (transparent zero-effect grid)
                    out = self._optic_children(int(f), grp)
                else:
                    out = self._screen_children(int(f), grp)
            elif body.role == "detector":
                # non-screen face of a detector solid: the body is an ideal
                # thin screen — its other faces are strict no-ops (no
                # refraction, no medium change, no mirror re-application)
                out = grp
            elif int(f) in scene.gratings:
                out = grating_mod.apply_to_batch(self, int(f), grp)
                _kill_differentials(out)   # order transport unimplemented
            else:
                out = self._optic_children(int(f), grp)
            if out is not None and len(out) > 0:
                children.append(out)

        if particle_children is not None and len(particle_children) > 0:
            children.append(particle_children)
        if not children:
            return None
        merged = RayBatch.concatenate(children)
        return self._apply_floors(merged)

    # ------------------------------------------------------------------
    def _record_reflection(self, child, fid):
        """Ghost-analysis bookkeeping (--ghost-analysis): stamp the FACE id
        `fid` of this reflection event into the child batch's refl_hist,
        indexed by the child's PRE-increment generation (capped at the last
        slot). MUST be called BEFORE `child.generation += 1`. Zero overhead
        when track_history is off (refl_hist is None). Purely diagnostic —
        never touches power/phase/direction."""
        if not self.cfg.track_history or child.refl_hist is None \
                or len(child) == 0:
            return
        slot = np.minimum(child.generation, HIST_DEPTH - 1).astype(np.intp)
        child.refl_hist[np.arange(len(child)), slot] = int(fid)

    # ------------------------------------------------------------------
    def _detector_event(self, fid, grp, grp_start, grp_start_opl,
                        grp_nmed, grp_dA=None):
        det = self.detectors[fid]
        if self.cfg.export_rays and len(grp) > 0:
            self._export_records(det, grp)
        coh = grp.coherent
        if np.any(coh):
            c = np.where(coh)[0]
            keys = np.stack([grp.source_id[c], grp.lam_stratum[c],
                             grp.pol_stratum[c]], axis=1)
            for key in np.unique(keys, axis=0):
                sel = c[np.all(keys == key, axis=1)]
                if len(sel) == 0:
                    continue
                det.add_gather_samples(
                    key[0], key[1], key[2],
                    grp_start[sel], grp.dir[sel],
                    grp.Es[sel], grp.Ep[sel], grp.s_hat[sel],
                    grp.lam[sel], grp_start_opl[sel],
                    grp.power[sel], grp.scattered[sel],
                    dA=grp_dA[sel] if grp_dA is not None else None)
        if np.any(~coh):
            i = np.where(~coh)[0]
            det.deposit_incoherent(grp.pos[i], grp.power[i], grp.lam[i],
                                   source_id=grp.source_id[i],
                                   lam_stratum=grp.lam_stratum[i],
                                   pol_stratum=grp.pol_stratum[i])
        # diagnostic ledger (not a closure bucket)
        self.ledger.by_surface[det.label] = (
            self.ledger.by_surface.get(det.label, 0.0)
            + float(np.sum(grp.power)))
        # separate detected tally: by_surface historically mixes surface
        # absorption and detection under one key space
        self.ledger.detect(det.label, float(np.sum(grp.power)))

    def _export_records(self, det, grp):
        """--export-rays: append one per-detector-event record of the ray
        states AT this detector hit (grp.pos is the hit point, grp.opl the
        accumulated OPL to it — both already advanced to the intersection
        before this call). Diagnostic only; never feeds the splat/gather."""
        n = len(grp)
        bp = grp.birth_pos
        if bp is None:
            bp = np.full((n, 3), np.nan)
        rec = {
            "pos": grp.pos.copy(),
            "dir": grp.dir.copy(),
            "opl": grp.opl.copy(),
            "lam": grp.lam.copy(),
            "source_id": grp.source_id.copy(),
            "lam_stratum": grp.lam_stratum.copy(),
            "pol_stratum": grp.pol_stratum.copy(),
            "generation": grp.generation.copy(),
            "pol_mode": grp.pol_mode.copy(),
            "power": grp.power,
            "scattered": grp.scattered.copy(),
            "coherent": grp.coherent.copy(),
            "birth_pos": bp.copy(),
        }
        # --ghost-analysis: the reflection face-id history rides along like
        # every other field (so the --workers merge concatenates it too).
        # Present only when track_history is on (refl_hist allocated).
        if grp.refl_hist is not None:
            rec["refl_hist"] = grp.refl_hist.copy()
        # track_time: the accumulated group path / GDD ride along the same
        # way (present only when the time slots are allocated).
        if grp.gopl is not None:
            rec["gopl"] = grp.gopl.copy()
        if grp.gdd_acc is not None:
            rec["gdd_acc"] = grp.gdd_acc.copy()
        det.ray_records.append(rec)

    def _flux_out_children(self, body, children):
        """Per-element boundary-flux tally (diagnostic, zero RNG use):
        credit each child ray that leaves this interface OUTSIDE the body
        as power flowing out of the element. 'Outside' is read off the
        medium stack (a child created at B's boundary whose top-of-stack
        is not B is in the surroundings), so the same test covers
        specular, scattered, o/e and screen children uniformly."""
        for ch in children:
            if ch is None or len(ch) == 0:
                continue
            outside = ch.current_medium() != body.index
            if np.any(outside):
                self.ledger.flux_out(body.label,
                                     float(np.sum(ch.power[outside])))

    def _screen_children(self, fid, grp):
        """Ideal thin screen: mirror fraction specular-reflects, absorbance
        eats its share of the rest, remainder continues UNREFRACTED."""
        body = self.scene.body_of_face(fid)
        self.ledger.flux_in(body.label, float(np.sum(grp.power)))
        r_m = body.mirror
        a = body.absorbance
        face = self.scene.faces[fid]
        out = []
        # transmitted continuation
        t_frac = (1.0 - r_m) * (1.0 - a)
        if t_frac > 0:
            tr = grp.select(np.ones(len(grp), dtype=bool))
            tr.Es *= np.sqrt(t_frac)
            tr.Ep *= np.sqrt(t_frac)
            out.append(tr)
        if r_m > 0:
            n_out = face.normal_out_of_solid(grp.pos)
            # normal against the ray
            sgn = -np.sign(np.sum(n_out * grp.dir, axis=-1))
            n_hat = n_out * sgn[:, None]
            rf = grp.select(np.ones(len(grp), dtype=bool))
            _kill_differentials(rf)
            rf.dir = fr.reflect_dir(grp.dir, n_hat)
            rf.Es *= -np.sqrt(r_m)
            rf.Ep *= -np.sqrt(r_m)
            self._record_reflection(rf, fid)
            rf.generation += 1
            out.append(rf)
        ab = (1.0 - r_m) * a
        if ab > 0:
            self.ledger.credit("absorbed_surface", grp.source_id,
                               grp.power * ab, where=body.label)
        self._flux_out_children(body, out)
        return RayBatch.concatenate(out) if out else None

    # ------------------------------------------------------------------
    def _optic_children(self, fid, grp):
        scene = self.scene
        face = scene.faces[fid]
        body = scene.body_of_face(fid)
        m = len(grp)

        n_out = face.normal_out_of_solid(grp.pos)
        cos_with_out = np.sum(grp.dir * n_out, axis=-1)
        entering = cos_with_out < 0.0

        # ---- seam-leak guard ----
        # A ray whose medium stack disagrees with the crossing direction
        # slipped through the seam between two trimmed faces (one face's
        # trim rejected the true hit, a neighbor accepted a grazing one).
        # Rare by construction; kill it and account the power visibly.
        top = grp.current_medium()
        leak = np.where(entering, top == body.index, top != body.index)
        if np.any(leak):
            self.ledger.credit("seam_loss", grp.source_id[leak],
                               grp.power[leak], where=body.label)
            if not np.any(~leak):
                return None
            grp = grp.select(~leak)
            n_out = n_out[~leak]
            entering = entering[~leak]
            m = len(grp)

        n_hat = np.where(entering[:, None], n_out, -n_out)
        cos_i = np.clip(-np.sum(grp.dir * n_hat, axis=-1), 0.0, 1.0)

        if np.any(entering):
            self.ledger.flux_in(body.label,
                                float(np.sum(grp.power[entering])))

        # ---- uniaxial birefringence: dedicated o/e split path ----
        if body.birefringent:
            return self._birefringent_children(fid, grp, entering, n_hat,
                                               cos_i)
        # ---- biaxial: dedicated slow/fast two-sheet split path ----
        if body.biaxial:
            return self._biaxial_children(fid, grp, entering, n_hat, cos_i)
        if np.any(grp.pol_mode != 0):
            # a mode-tagged crystal ray (uniaxial e, biaxial slow/fast)
            # hit a non-birefringent boundary (nested body inside a
            # crystal): approximate as ordinary from here on
            if not getattr(self, "_warned_nested_mode", False):
                import warnings
                warnings.warn(
                    "mode-tagged crystal ray hit non-birefringent face %s "
                    "(nested body inside a crystal?) — continuing as "
                    "ordinary index (documented approximation)" % face.id)
                self._warned_nested_mode = True
            grp.pol_mode[:] = 0
            grp.n_eff[:] = 0.0
            grp.n_g_eff[:] = 0.0

        # media on both sides
        cur = grp.current_medium()
        n1 = np.empty(m, dtype=np.complex128)
        for mm in np.unique(cur):
            s = cur == mm
            n1[s] = scene.medium_index(int(mm), grp.lam[s])
        n2 = np.empty(m, dtype=np.complex128)
        # entering: far side is this body's material
        if np.any(entering):
            n2[entering] = scene.medium_index(body.index,
                                              grp.lam[entering])
        # exiting: far side is the medium under the top of the stack
        ex = ~entering
        if np.any(ex):
            idx = np.where(ex)[0]
            depth = grp.depth[idx]
            under = np.where(
                depth >= 2,
                grp.medium[idx, np.maximum(depth - 2, 0)],
                AMBIENT).astype(np.int64)
            tmp = np.empty(len(idx), dtype=np.complex128)
            for mm in np.unique(under):
                s = under == mm
                tmp[s] = scene.medium_index(int(mm), grp.lam[idx][s])
            n2[ex] = tmp

        # amplitude coefficients: bare Fresnel, coating TMM, or a measured
        # coating table (per-face map; whole-body coatings expand to every
        # face at Scene construction)
        coat = scene.face_coatings.get(fid)
        if coat is not None and scene.coatings[coat]["kind"] == "tmm":
            cspec = scene.coatings[coat]
            layer_n, layer_d = tf.resolve_coating_layers(
                cspec["layers"], scene.matdb, grp.lam)
            rs, rp, ts, tp, etas = tf.tmm_coeffs(
                grp.lam, cos_i, n1, n2, layer_n, layer_d)
            Rs, Rp, Ts, Tp = tf.tmm_power(rs, rp, ts, tp, etas)
        elif coat is not None:
            # tabulated coating: measured Rs/Rp/Ts/Tp at the ray wavelength.
            # Tables carry no phase, so the amplitude coefficients borrow
            # the BARE-interface Fresnel phase (documented approximation:
            # use TMM layer stacks when coating phase matters coherently).
            from .optprops import interp_hard
            cspec = scene.coatings[coat]
            rs, rp, ts, tp, ct = fr.fresnel_coeffs(cos_i, n1, n2)
            lam_um = grp.lam * 1e6
            ctx = "coating %r on %s" % (coat, face.id)
            Rs = interp_hard(lam_um, cspec["lam_um"], cspec["Rs"], ctx)
            Rp = interp_hard(lam_um, cspec["lam_um"], cspec["Rp"], ctx)
            Ts = interp_hard(lam_um, cspec["lam_um"], cspec["Ts"], ctx)
            Tp = interp_hard(lam_um, cspec["lam_um"], cspec["Tp"], ctx)
            # past the critical angle there IS no propagating transmitted
            # direction: honor TIR by folding the table's T into the
            # reflected side (energy-conserving). Without this, the table
            # branch emitted a "transmitted" child whose refract_dir was
            # a degenerate grazing ghost, silently booked as seam loss
            # (found by the BS-cube investigation).
            tir = fr.is_tir(cos_i, n1, n2)
            if np.any(tir):
                Rs = np.where(tir, np.clip(Rs + Ts, 0.0, 1.0), Rs)
                Rp = np.where(tir, np.clip(Rp + Tp, 0.0, 1.0), Rp)
                Ts = np.where(tir, 0.0, Ts)
                Tp = np.where(tir, 0.0, Tp)
            rs = np.sqrt(Rs) * np.exp(1j * np.angle(rs))
            rp = np.sqrt(Rp) * np.exp(1j * np.angle(rp))
            ts = np.sqrt(np.maximum(Ts, 0.0)) * np.exp(1j * np.angle(ts))
            tp = np.sqrt(np.maximum(Tp, 0.0)) * np.exp(1j * np.angle(tp))
        else:
            rs, rp, ts, tp, ct = fr.fresnel_coeffs(cos_i, n1, n2)
            Rs, Rp, Ts, Tp = fr.power_coeffs(rs, rp, ts, tp, cos_i, ct,
                                             n1, n2)
        Ts = np.clip(Ts, 0.0, None)
        Tp = np.clip(Tp, 0.0, None)

        # rotate Jones into this interface's (s,p) basis
        s_new, p_new = fr.pol_basis(grp.dir, n_hat)
        p_old = np.cross(grp.dir, grp.s_hat)
        Es, Ep = fr.rotate_jones(grp.Es, grp.Ep, grp.s_hat, p_old,
                                 s_new, p_new)

        r_m = body.mirror
        a = body.absorbance
        phys = (1.0 - r_m) * (1.0 - a)

        p_in = np.abs(Es) ** 2 + np.abs(Ep) ** 2
        out = []

        # ---- ray differentials at this interface ----
        # dt-correct the transferred position differential onto the surface
        # (Igehy), sign-correct the canonical shape operator to d(n_hat)/dp,
        # and precompute reflected/refracted direction differentials.
        diff = None
        if grp.has_differentials:
            surf = getattr(face, "surface", None)
            if surf is not None and hasattr(surf, "normal_derivative"):
                n_can = surf.normal(grp.pos)
                fsign = np.sign(np.sum(n_hat * n_can, axis=-1))
                S = fsign[:, None, None] * surf.normal_derivative(grp.pos)
                denom = np.sum(grp.dir * n_hat, axis=-1)
                denom = np.where(np.abs(denom) < 1e-12, np.inf, denom)

                def _to_surf(dP):
                    dt = -np.sum(dP * n_hat, axis=-1) / denom
                    return dP + dt[:, None] * grp.dir

                diff = {"dPx": _to_surf(grp.dPdx),
                        "dPy": _to_surf(grp.dPdy), "S": S}

        # ---- roughness: specular attenuation factor ----
        rough = self.scene.roughness.get(fid)
        if rough is not None:
            sigma_m = rough["sigma_nm"] * 1e-9
            A_spec = rough_mod.specular_power_factor(sigma_m, cos_i,
                                                     grp.lam)
        else:
            A_spec = np.ones(m)
        sqrtA = np.sqrt(A_spec)

        # ---- measured scatter (ABg/BSDF): reflected-side specular/scatter
        # split. TIS(cos_i) leaves the specular direction; the specular
        # reflection keeps sqrt(1-TIS) of its amplitude (REFLECTED side only —
        # the transmitted child is untouched, v1 BRDF scope), the scattered
        # remainder is emitted as sampled lobes below. Mutually exclusive
        # with roughness/diffuser by the scene contract, so sqrtA == 1 here.
        scat = self.scene.scatter.get(fid)
        if scat is not None:
            from . import scatter as scatter_mod
            tis = scatter_mod.abg_tis(scat["A"], scat["B"], scat["g"], cos_i)
            if scat["tis_cap"] is not None:
                tis = np.minimum(tis, scat["tis_cap"])
            tis = np.clip(tis, 0.0, 1.0)
            refl_scale = np.sqrt(1.0 - tis)
        else:
            refl_scale = np.ones(m)

        # ---- reflected child ----
        # power-exact amplitude, phase from the physical coefficient.
        # full_amp_r* is the UNSPLIT reflected amplitude (reused by the
        # scatter lobes); the specular child additionally carries sqrtA
        # (roughness) and refl_scale (scatter specular remainder).
        full_amp_rs = np.sqrt(r_m + phys * np.abs(rs) ** 2) \
            * np.exp(1j * np.angle(rs))
        full_amp_rp = np.sqrt(r_m + phys * np.abs(rp) ** 2) \
            * np.exp(1j * np.angle(rp))
        amp_rs = full_amp_rs * sqrtA * refl_scale
        amp_rp = full_amp_rp * sqrtA * refl_scale
        can_reflect = grp.generation < self.cfg.max_reflections
        refl = grp.select(np.ones(m, dtype=bool))
        refl.dir = fr.reflect_dir(grp.dir, n_hat)
        refl.s_hat = s_new
        refl.Es = Es * amp_rs
        refl.Ep = Ep * amp_rp
        self._record_reflection(refl, fid)
        refl.generation += 1
        if grp.has_differentials:
            if diff is None:
                _kill_differentials(refl)
            else:
                refl.dPdx, refl.dDdx = diff_mod.reflect(
                    diff["dPx"], grp.dDdx, grp.dir, n_hat, diff["S"])
                refl.dPdy, refl.dDdy = diff_mod.reflect(
                    diff["dPy"], grp.dDdy, grp.dir, n_hat, diff["S"])
        p_refl = refl.power
        if np.any(~can_reflect):
            self.ledger.credit("truncated_generation",
                               refl.source_id[~can_reflect],
                               p_refl[~can_reflect])
        keep_r = can_reflect & (p_refl > 0)
        if np.any(keep_r):
            out.append(refl.select(keep_r))

        # ---- transmitted child ----
        tir = (Ts + Tp) <= 1e-15
        trans = grp.select(np.ones(m, dtype=bool))
        trans.dir = fr.refract_dir(grp.dir, n_hat, cos_i,
                                   np.real(n1), np.real(n2))
        trans.s_hat = s_new
        if grp.has_differentials:
            if diff is None:
                _kill_differentials(trans)
            else:
                eta = np.real(n1) / np.real(n2)
                trans.dPdx, trans.dDdx = diff_mod.refract(
                    diff["dPx"], grp.dDdx, grp.dir, n_hat, diff["S"],
                    eta, trans.dir)
                trans.dPdy, trans.dDdy = diff_mod.refract(
                    diff["dPy"], grp.dDdy, grp.dir, n_hat, diff["S"],
                    eta, trans.dir)
        # amplitude via power (impedance factor folded in), phase from ts/tp
        amp_ts = np.sqrt(phys * Ts) * np.exp(1j * np.angle(ts)) * sqrtA
        amp_tp = np.sqrt(phys * Tp) * np.exp(1j * np.angle(tp)) * sqrtA
        trans.Es = Es * amp_ts
        trans.Ep = Ep * amp_tp
        # medium bookkeeping
        ent = entering & ~tir
        exi = (~entering) & ~tir
        body_idx = np.full(m, body.index)
        trans.push_medium(ent, body_idx)
        trans.pop_medium(exi, body_idx)
        # ---- polarizer: dichroic Jones diattenuator on ENTRY ----
        # applied once per traversal (entry face); the rejected component's
        # power goes to its own ledger bucket, NOT absorbed_surface, so the
        # exact-difference below must see the PRE-polarizer power.
        p_trans_pre = trans.power.copy()
        if body.polarizer is not None and np.any(ent):
            self._apply_polarizer(body, trans, ent)
            d_loss = p_trans_pre - trans.power
            lost = ent & (d_loss > 0)
            if np.any(lost):
                self.ledger.credit("polarizer_absorbed",
                                   trans.source_id[lost], d_loss[lost],
                                   where=body.label)
        keep_t = ~tir & (trans.power > 0)
        if np.any(keep_t):
            out.append(trans.select(keep_t))

        # per-ray power accounted to children so absorption stays exact
        # (pre-polarizer: the polarizer loss already has its own bucket)
        p_accounted = refl.power + p_trans_pre * (~tir)

        # ---- roughness: Beckmann-scattered lobes carry the (1-A) power --
        # Scattered children keep their OPL phase (deterministic
        # microfacets per event) so speckle is physical; --seeds averages
        # realizations. rough_fresnel='micro' (default): coefficients
        # evaluated at each MICROFACET-LOCAL incidence angle, applied per
        # polarization in the microfacet s/p basis (fixes README §6.2
        # item 2). 'macro': legacy nominal-angle scalar average, for A/B.
        if rough is not None and np.any(A_spec < 1.0 - 1e-12):
            micro = self.cfg.rough_fresnel == "micro"
            k_lobe = 2
            slope = rough_mod.slope_from_sigma_lcorr(
                rough["sigma_nm"] * 1e-9, rough["lcorr_um"] * 1e-6)
            mf = rough_mod.beckmann_sample(n_hat, slope, self.rng, k_lobe)
            # POWER FRACTIONS of the incident ray (dimensionless, <= 1);
            # child amplitude scales are sqrt(fraction / k_lobe) applied
            # to the incident Jones amplitudes — do NOT divide by p_in
            # (a fraction divided by watts once amplified every child to
            # ~fraction/k ABSOLUTE watts and blew energy up 1e9x)
            loseR = (1.0 - A_spec) * (r_m + phys * 0.5 *
                                      (np.abs(rs) ** 2 + np.abs(rp) ** 2))
            loseT = (1.0 - A_spec) * (phys * 0.5 * (Ts + Tp))
            for j in range(k_lobe):
                n_j = mf[:, j, :]
                cos_j = np.clip(-np.sum(grp.dir * n_j, axis=-1), 0.0, 1.0)
                if micro:
                    # coefficients at the local angle. TMM stacks are
                    # re-evaluated at cos_j; single-AOI measured tables
                    # keep their macro values (documented).
                    if coat is not None \
                            and scene.coatings[coat]["kind"] == "tmm":
                        rs_j, rp_j, ts_j, tp_j, etas_j = tf.tmm_coeffs(
                            grp.lam, cos_j, n1, n2, layer_n, layer_d)
                        Rs_j, Rp_j, Ts_j, Tp_j = tf.tmm_power(
                            rs_j, rp_j, ts_j, tp_j, etas_j)
                    elif coat is not None:
                        rs_j, rp_j, ts_j, tp_j = rs, rp, ts, tp
                        Rs_j, Rp_j, Ts_j, Tp_j = Rs, Rp, Ts, Tp
                    else:
                        rs_j, rp_j, ts_j, tp_j, ct_j = fr.fresnel_coeffs(
                            cos_j, n1, n2)
                        Rs_j, Rp_j, Ts_j, Tp_j = fr.power_coeffs(
                            rs_j, rp_j, ts_j, tp_j, cos_j, ct_j, n1, n2)
                    # grazing microfacets can drive the TMM/Fresnel stack
                    # into NaN (internal TIR at the local angle) — zero
                    # those lobes; their power lands in absorbed_surface
                    # via the exact difference instead of poisoning the
                    # ledger (clip() passes NaN through!)
                    rs_j = np.nan_to_num(rs_j)
                    rp_j = np.nan_to_num(rp_j)
                    ts_j = np.nan_to_num(ts_j)
                    tp_j = np.nan_to_num(tp_j)
                    Ts_j = np.clip(np.nan_to_num(Ts_j), 0.0, None)
                    Tp_j = np.clip(np.nan_to_num(Tp_j), 0.0, None)
                    # Jones into the microfacet's own s/p basis
                    s_j, p_j = fr.pol_basis(grp.dir, n_j)
                    Es_j, Ep_j = fr.rotate_jones(
                        Es, Ep, s_new, np.cross(grp.dir, s_new), s_j, p_j)
                    frac = (1.0 - A_spec) / k_lobe
                # scattered reflection
                sc = grp.select(np.ones(m, dtype=bool))
                _kill_differentials(sc)
                sc.dir = fr.reflect_dir(grp.dir, n_j)
                if micro:
                    sc.s_hat = s_j
                    sc.Es = Es_j * np.sqrt(
                        frac * (r_m + phys * np.abs(rs_j) ** 2)) \
                        * np.exp(1j * np.angle(rs_j))
                    sc.Ep = Ep_j * np.sqrt(
                        frac * (r_m + phys * np.abs(rp_j) ** 2)) \
                        * np.exp(1j * np.angle(rp_j))
                else:
                    sc.s_hat = s_new
                    amp_j = np.sqrt(loseR / k_lobe)
                    sc.Es = Es * amp_j
                    sc.Ep = Ep * amp_j
                self._record_reflection(sc, fid)
                sc.generation += 1
                sc.scattered[:] = True
                ok = ((np.sum(sc.dir * n_hat, axis=-1) > 0)
                      & (sc.generation <= self.cfg.max_reflections)
                      & (sc.power > 0))
                below = ~ok & (sc.power > 0)
                if np.any(below):
                    self.ledger.credit("absorbed_surface",
                                       sc.source_id[below],
                                       sc.power[below], where=body.label)
                p_accounted += sc.power
                if np.any(ok):
                    out.append(sc.select(ok))
                # scattered transmission (skip under TIR)
                st = trans.select(np.ones(m, dtype=bool))
                _kill_differentials(st)
                st.dir = fr.refract_dir(grp.dir, n_j, cos_j,
                                        np.real(n1), np.real(n2))
                if micro:
                    st.s_hat = s_j
                    st.Es = Es_j * np.sqrt(frac * phys * Ts_j) \
                        * np.exp(1j * np.angle(ts_j))
                    st.Ep = Ep_j * np.sqrt(frac * phys * Tp_j) \
                        * np.exp(1j * np.angle(tp_j))
                    # macro-TIR rows never pushed/popped the medium on
                    # `trans`, so a micro-transmitting lobe there would
                    # carry a wrong stack — suppress (no frustrated-TIR
                    # scatter; documented)
                    tir_j = ((Ts_j + Tp_j) <= 1e-15) | tir
                else:
                    amp_tj = np.sqrt(loseT / k_lobe)
                    st.Es = Es * amp_tj
                    st.Ep = Ep * amp_tj
                    tir_j = tir
                st.scattered[:] = True
                okt = (~tir_j & (st.power > 0)
                       & (np.sum(st.dir * n_hat, axis=-1) < 0))
                belowt = ~okt & (st.power > 0) & ~tir_j
                if np.any(belowt):
                    self.ledger.credit("absorbed_surface",
                                       st.source_id[belowt],
                                       st.power[belowt], where=body.label)
                p_accounted += st.power * (~tir_j)
                if np.any(okt):
                    out.append(st.select(okt))

        # ---- measured scatter (ABg): scattered reflected lobes carry the
        # TIS share of the reflected power, sampled around the specular
        # direction (scatter_mod.sample_abg). k_lobe matches the roughness
        # convention. Each lobe scales the FULL reflected amplitude by
        # sqrt(TIS/k), so specular (1-TIS) + scattered TIS == full R exactly.
        # Reflected side only (BRDF); transmission is not scattered in v1.
        if scat is not None and np.any(tis > 0.0):
            k_lobe = 2
            d_spec = fr.reflect_dir(grp.dir, n_hat)
            amp_lobe = np.sqrt(tis / k_lobe)
            for _j in range(k_lobe):
                sc = grp.select(np.ones(m, dtype=bool))
                _kill_differentials(sc)
                sc.dir = scatter_mod.sample_abg(
                    self.rng, m, scat["A"], scat["B"], scat["g"],
                    d_spec, n_hat)
                sc.s_hat = s_new
                sc.Es = Es * full_amp_rs * amp_lobe
                sc.Ep = Ep * full_amp_rp * amp_lobe
                self._record_reflection(sc, fid)
                sc.generation += 1
                sc.scattered[:] = True
                ok = ((np.sum(sc.dir * n_hat, axis=-1) > 0)
                      & (sc.generation <= self.cfg.max_reflections)
                      & (sc.power > 0))
                below = ~ok & (sc.power > 0)
                if np.any(below):
                    self.ledger.credit("absorbed_surface",
                                       sc.source_id[below],
                                       sc.power[below], where=body.label)
                p_accounted += sc.power
                if np.any(ok):
                    out.append(sc.select(ok))

        # ---- surface absorption = exact power difference ----
        # (generation-capped reflections were already credited above, so
        # they count as 'produced' here; tir kills the transmitted child)
        absorbed = np.clip(p_in - p_accounted, 0.0, None)
        if np.any(absorbed > 0):
            self.ledger.credit("absorbed_surface", grp.source_id, absorbed,
                               where=body.label)
        self._flux_out_children(body, out)
        return RayBatch.concatenate(out) if out else None

    # ------------------------------------------------------------------
    def _apply_polarizer(self, body, trans, mask):
        """Dichroic diattenuator Jones matrix on the transmitted rays in
        `mask` (in place). Transmission axis = body.polarizer_axis (global)
        projected transverse to each ray; T_par/T_perp interpolated from
        the polarizer's table at the ray wavelength. Circular polarizers
        add an ideal retarder (retardance_waves) with fast axis at +-45deg
        to the transmission axis: +45 -> right-circular output for light
        along the transmission axis (pinned by test)."""
        from .optprops import interp_hard
        entry = self.scene.polarizers[body.polarizer]
        idx = np.where(mask)[0]
        d = trans.dir[idx]
        axis = body.polarizer_axis
        t = axis[None, :] - np.sum(d * axis, axis=-1, keepdims=True) * d
        nrm = np.linalg.norm(t, axis=-1)
        good = nrm > 1e-9
        # rays (anti)parallel to the transmission axis see no transverse
        # axis projection: attenuate both components by the mean (the film
        # looks isotropic edge-on) — rare, but must not NaN
        t_hat = np.where(good[:, None], t / np.maximum(nrm, 1e-300)[:, None],
                         trans.s_hat[idx])
        p_of_t = np.cross(d, t_hat)
        p_old = np.cross(d, trans.s_hat[idx])
        Et, Ep_ = fr.rotate_jones(trans.Es[idx], trans.Ep[idx],
                                  trans.s_hat[idx], p_old, t_hat, p_of_t)
        ctx = "polarizer %r on %s" % (body.polarizer, body.label)
        lam_um = trans.lam[idx] * 1e6
        T_par = interp_hard(lam_um, entry["lam_um"], entry["T_par"], ctx)
        T_perp = interp_hard(lam_um, entry["lam_um"], entry["T_perp"], ctx)
        a_par = np.sqrt(np.where(good, T_par, 0.5 * (T_par + T_perp)))
        a_perp = np.sqrt(np.where(good, T_perp, 0.5 * (T_par + T_perp)))
        Et = Et * a_par
        Ep_ = Ep_ * a_perp
        if entry["type"] in ("circular_left", "circular_right"):
            # ideal retarder, fast axis at +-45deg to the transmission axis
            delta = 2.0 * np.pi * entry["retardance_waves"]
            sgn = 1.0 if entry["type"] == "circular_right" else -1.0
            # J = R(-th) diag(1, e^{-i delta}) R(th), th = sgn*45deg
            c = np.sqrt(0.5)
            s = sgn * np.sqrt(0.5)
            e = np.exp(-1j * delta)
            j11 = c * c + e * s * s
            j12 = c * s - e * s * c
            Et, Ep_ = j11 * Et + j12 * Ep_, j12 * Et + (s * s + e * c * c) \
                * Ep_
        trans.Es[idx] = Et
        trans.Ep[idx] = Ep_
        trans.s_hat[idx] = t_hat

    # ------------------------------------------------------------------
    def _birefringent_children(self, fid, grp, entering, n_hat, cos_i):
        """Uniaxial-crystal boundary: o/e double refraction.

        Entry: the transmitted field splits into an ordinary child (Snell
        with n_o, D along e_o_hat) and an extraordinary child (wavevector
        from the e normal surface, RAY along the walk-off Poynting
        direction, phase index cached in n_eff). Fresnel amplitudes use
        the isotropic-effective-index approximation (n_o for o, n(theta)
        for e) — documented in README §6; the dropped cross terms land in
        absorbed_surface via the exact power difference.
        Exit: each mode refracts out via wavevector tangential continuity;
        internal reflections are mode-preserving (k reflects specularly,
        the e-ray direction/index recomputed from the reflected k).
        Coating/roughness on birefringent faces are not modeled (warned)."""
        from . import birefringence as bir
        scene = self.scene
        body = scene.body_of_face(fid)
        face = scene.faces[fid]
        m = len(grp)
        if (scene.face_coatings.get(fid) is not None
                or scene.roughness.get(fid) is not None) \
                and not getattr(self, "_warned_bir_extras", False):
            import warnings
            warnings.warn(
                "coating/roughness on birefringent face %s is not modeled "
                "(bare-interface Fresnel used)" % face.id)
            self._warned_bir_extras = True

        n_o, n_e = scene.uniaxial_indices(body, grp.lam)
        c_axis = body.crystal_axis
        r_m = body.mirror
        a = body.absorbance
        phys = (1.0 - r_m) * (1.0 - a)
        p_in = grp.power
        p_accounted = np.zeros(m)
        out = []
        can_reflect = grp.generation < self.cfg.max_reflections

        # ---------------- ENTRY: outside -> crystal ----------------
        ent = np.where(entering)[0]
        if len(ent):
            sub = grp.select(ent)
            nh = n_hat[ent]
            ci = cos_i[ent]
            cur = sub.current_medium()
            n1 = np.empty(len(ent), dtype=np.complex128)
            for mm in np.unique(cur):
                s = cur == mm
                n1[s] = scene.medium_index(int(mm), sub.lam[s])
            res = bir.refract_in(sub.dir, nh, c_axis, np.real(n1),
                                 n_o[ent], n_e[ent])
            # incident Jones in the interface (s,p) basis
            s_new, p_new = fr.pol_basis(sub.dir, nh)
            p_old = np.cross(sub.dir, sub.s_hat)
            Es_i, Ep_i = fr.rotate_jones(sub.Es, sub.Ep, sub.s_hat, p_old,
                                         s_new, p_new)
            # ---- unitary o/e channel decomposition at the incident k ----
            # the component of the incident field along e_o couples to the
            # ordinary wave (n_o Fresnel), the e_e component to the
            # extraordinary wave (n(theta) Fresnel). Decomposing FIRST and
            # applying each channel's own R/T keeps R+T=1 per channel —
            # applying n_o reflection to the e-coupled component overcounts
            # energy by ~(Ts_e - Ts_o) and broke closure at the 1e-2 level.
            eo_i, ee_i = bir.eigenbasis(sub.dir, c_axis)
            Eo_i, Ee_i = fr.rotate_jones(Es_i, Ep_i, s_new, p_new,
                                         eo_i, ee_i)
            # each channel back in (s,p) (unit Jones direction * amplitude)
            cs_o = np.sum(eo_i * s_new, axis=-1)   # cos/sin of the eigen
            sn_o = np.sum(eo_i * p_new, axis=-1)   # rotation angle
            # channel fields in (s,p): o-channel = Eo_i*(cs_o, sn_o),
            # e-channel = Ee_i*(-sn_o, cs_o)  (orthogonal complement)
            rs_o, rp_o, ts_o, tp_o, ct_o = fr.fresnel_coeffs(
                ci, n1, n_o[ent].astype(np.complex128))
            Rs_o, Rp_o, Ts_o, Tp_o = fr.power_coeffs(
                rs_o, rp_o, ts_o, tp_o, ci, ct_o, n1,
                n_o[ent].astype(np.complex128))
            ne_th = res["n_phase_e"].astype(np.complex128)
            rs_e, rp_e, ts_e, tp_e, ct_e = fr.fresnel_coeffs(ci, n1, ne_th)
            _, _, Ts_e, Tp_e = fr.power_coeffs(rs_e, rp_e, ts_e, tp_e,
                                               ci, ct_e, n1, ne_th)

            # reflected child: coherent sum of both channels' reflections
            refl = sub.select(np.ones(len(ent), dtype=bool))
            _kill_differentials(refl)
            refl.dir = fr.reflect_dir(sub.dir, nh)
            refl.s_hat = s_new
            amp_rs_o = np.sqrt(r_m + phys * np.abs(rs_o) ** 2) \
                * np.exp(1j * np.angle(rs_o))
            amp_rp_o = np.sqrt(r_m + phys * np.abs(rp_o) ** 2) \
                * np.exp(1j * np.angle(rp_o))
            amp_rs_e = np.sqrt(r_m + phys * np.abs(rs_e) ** 2) \
                * np.exp(1j * np.angle(rs_e))
            amp_rp_e = np.sqrt(r_m + phys * np.abs(rp_e) ** 2) \
                * np.exp(1j * np.angle(rp_e))
            refl.Es = Eo_i * cs_o * amp_rs_o - Ee_i * sn_o * amp_rs_e
            refl.Ep = Eo_i * sn_o * amp_rp_o + Ee_i * cs_o * amp_rp_e
            self._record_reflection(refl, fid)
            refl.generation += 1
            cr = can_reflect[ent]
            if np.any(~cr):
                self.ledger.credit("truncated_generation",
                                   refl.source_id[~cr], refl.power[~cr])
            keep = cr & (refl.power > 0)
            if np.any(keep):
                out.append(refl.select(keep))
            p_accounted[ent] += refl.power

            # ordinary transmitted child: o-channel through n_o Fresnel,
            # D field along e_o_hat of k_o
            eo_o, ee_o = bir.eigenbasis(res["k_o"], c_axis)
            amp_ts_o = np.sqrt(phys * np.clip(Ts_o, 0.0, None)) \
                * np.exp(1j * np.angle(ts_o))
            amp_tp_o = np.sqrt(phys * np.clip(Tp_o, 0.0, None)) \
                * np.exp(1j * np.angle(tp_o))
            Eso, Epo = fr.rotate_jones(
                Eo_i * cs_o * amp_ts_o, Eo_i * sn_o * amp_tp_o,
                s_new, np.cross(res["k_o"], s_new), eo_o, ee_o)
            och = sub.select(np.ones(len(ent), dtype=bool))
            _kill_differentials(och)
            och.dir = res["k_o"]
            och.s_hat = eo_o
            och.Es = Eso                     # o mode: D along e_o_hat only
            och.Ep = np.zeros_like(Epo)
            och.pol_mode[:] = 0
            och.n_eff[:] = 0.0               # medium_index gives n_o
            och.n_g_eff[:] = 0.0             # medium_group_index: o group
            ok_o = ~res["tir_o"]
            och.push_medium(ok_o, np.full(len(ent), body.index))
            keep = ok_o & (och.power > 0)
            if np.any(keep):
                out.append(och.select(keep))
            p_accounted[ent] += och.power * ok_o

            # extraordinary transmitted child: e-channel through n(theta)
            # Fresnel; D along e_e_hat, RAY along the walk-off Poynting
            # direction, phase index cached in n_eff
            eo_e, ee_e = bir.eigenbasis(res["k_e"], c_axis)
            amp_ts_e = np.sqrt(phys * np.clip(Ts_e, 0.0, None)) \
                * np.exp(1j * np.angle(ts_e))
            amp_tp_e = np.sqrt(phys * np.clip(Tp_e, 0.0, None)) \
                * np.exp(1j * np.angle(tp_e))
            Ese, Epe = fr.rotate_jones(
                -Ee_i * sn_o * amp_ts_e, Ee_i * cs_o * amp_tp_e,
                s_new, np.cross(res["k_e"], s_new), eo_e, ee_e)
            ech = sub.select(np.ones(len(ent), dtype=bool))
            _kill_differentials(ech)
            ech.dir = res["s_e"]
            ech.s_hat = eo_e                 # _|_ s_e exactly (in-plane ray)
            ech.Es = np.zeros_like(Ese)
            ech.Ep = Epe                     # e mode: D along e_e_hat only
            ech.pol_mode[:] = 1
            ech.n_eff[:] = res["n_ray_e"]    # OPL per metre along the RAY
            if self.cfg.track_time:
                # directional group index at the frozen internal k
                # (first-cut: phase-surface derivative at fixed theta,
                # walk-off path distinction neglected — see
                # birefringence.n_group_e_theta)
                mo, me = scene.matdb.get_uniaxial(body.material)
                cos_kc = np.sum(res["k_e"] * np.broadcast_to(
                    c_axis, res["k_e"].shape), axis=-1)
                ech.n_g_eff[:] = bir.n_group_e_theta(cos_kc, mo, me,
                                                     ech.lam)
            ok_e = ~res["tir_e"]
            ech.push_medium(ok_e, np.full(len(ent), body.index))
            keep = ok_e & (ech.power > 0)
            if np.any(keep):
                out.append(ech.select(keep))
            p_accounted[ent] += ech.power * ok_e

        # ---------------- EXIT: crystal -> outside ----------------
        exi = np.where(~entering)[0]
        if len(exi):
            sub = grp.select(exi)
            nh = n_hat[exi]
            # far-side medium (under the top of the stack)
            depth = sub.depth
            under = np.where(depth >= 2,
                             sub.medium[np.arange(len(exi)),
                                        np.maximum(depth - 2, 0)],
                             AMBIENT).astype(np.int64)
            n2 = np.empty(len(exi), dtype=np.complex128)
            for mm in np.unique(under):
                s = under == mm
                n2[s] = scene.medium_index(int(mm), sub.lam[s])
            is_e = sub.pol_mode == 1
            # internal wavevector: o-rays propagate along k; e-rays store
            # the RAY direction, invert the Poynting map to recover k
            k_int = sub.dir.copy()
            if np.any(is_e):
                k_int[is_e] = bir.k_from_ray(sub.dir[is_e], c_axis,
                                             n_o[exi][is_e],
                                             n_e[exi][is_e])
            n_phase = np.where(
                is_e,
                bir.n_e_theta(np.sum(k_int * np.broadcast_to(
                    c_axis, k_int.shape), axis=-1), n_o[exi], n_e[exi]),
                n_o[exi])
            cos_k = np.clip(-np.sum(k_int * nh, axis=-1), 0.0, 1.0)
            d_out, tir = bir.refract_out(k_int, is_e, nh, c_axis,
                                         n_o[exi], n_e[exi], np.real(n2))
            # effective-index Fresnel at the exit
            rs, rp, ts, tp, ct = fr.fresnel_coeffs(
                cos_k, n_phase.astype(np.complex128), n2)
            Rs, Rp, Ts, Tp = fr.power_coeffs(
                rs, rp, ts, tp, cos_k, ct,
                n_phase.astype(np.complex128), n2)
            Ts = np.clip(Ts, 0.0, None)
            Tp = np.clip(Tp, 0.0, None)
            s_new, p_new = fr.pol_basis(k_int, nh)
            p_old = np.cross(sub.dir, sub.s_hat)
            Es_i, Ep_i = fr.rotate_jones(sub.Es, sub.Ep, sub.s_hat, p_old,
                                         s_new, p_new)

            # transmitted child: leaves the crystal, mode resets
            tr = sub.select(np.ones(len(exi), dtype=bool))
            _kill_differentials(tr)
            tr.dir = d_out
            tr.s_hat = s_new
            tr.Es = Es_i * np.sqrt(phys * Ts) * np.exp(1j * np.angle(ts))
            tr.Ep = Ep_i * np.sqrt(phys * Tp) * np.exp(1j * np.angle(tp))
            tr.pol_mode[:] = 0
            tr.n_eff[:] = 0.0
            tr.n_g_eff[:] = 0.0
            ok_t = ~tir
            tr.pop_medium(ok_t, np.full(len(exi), body.index))
            keep = ok_t & (tr.power > 0)
            if np.any(keep):
                out.append(tr.select(keep))
            p_accounted[exi] += tr.power * ok_t

            # internal reflection (partial or TIR): mode-preserving,
            # k reflects specularly; e-ray direction/index recomputed
            k_r = fr.reflect_dir(k_int, nh)
            rf = sub.select(np.ones(len(exi), dtype=bool))
            _kill_differentials(rf)
            rf.dir = k_r
            rf.s_hat = s_new
            rf.Es = Es_i * np.sqrt(r_m + phys * np.abs(rs) ** 2) \
                * np.exp(1j * np.angle(rs))
            rf.Ep = Ep_i * np.sqrt(r_m + phys * np.abs(rp) ** 2) \
                * np.exp(1j * np.angle(rp))
            self._record_reflection(rf, fid)
            rf.generation += 1
            if np.any(is_e):
                s_ray, _, n_ray = bir.ray_from_k(k_r[is_e], c_axis,
                                                 n_o[exi][is_e],
                                                 n_e[exi][is_e])
                rf.dir[is_e] = s_ray
                rf.n_eff[is_e] = n_ray
                if self.cfg.track_time:
                    mo, me = scene.matdb.get_uniaxial(body.material)
                    cos_kc = np.sum(k_r[is_e] * np.broadcast_to(
                        c_axis, k_r[is_e].shape), axis=-1)
                    rf.n_g_eff[is_e] = bir.n_group_e_theta(
                        cos_kc, mo, me, rf.lam[is_e])
            cr = can_reflect[exi]
            if np.any(~cr):
                self.ledger.credit("truncated_generation",
                                   rf.source_id[~cr], rf.power[~cr])
            keep = cr & (rf.power > 0)
            if np.any(keep):
                out.append(rf.select(keep))
            p_accounted[exi] += rf.power

        absorbed = np.clip(p_in - p_accounted, 0.0, None)
        if np.any(absorbed > 0):
            self.ledger.credit("absorbed_surface", grp.source_id, absorbed,
                               where=body.label)
        self._flux_out_children(body, out)
        return RayBatch.concatenate(out) if out else None

    # ------------------------------------------------------------------
    def _biaxial_group_index(self, body, lam, k_hat, frame, slow_mask):
        """Directional GROUP index for biaxial sheet rays: local central
        difference of the SAME sheet's phase index over lam (+-0.1%
        relative step) holding k_hat and the crystal frame fixed,
        n_g = n - lam * dn/dlam. slow_mask: bool (n,) or scalar — True
        picks the slow sheet. Same first-cut limitation as the uniaxial
        n_group_e_theta (angular dispersion / ray-vs-k path neglected)."""
        from . import birefringence as bir
        lam = np.asarray(lam, dtype=np.float64)
        h = lam * 1e-3

        def n_of(lm):
            modes = bir.biaxial_modes_for_k(
                k_hat, frame, self.scene.biaxial_eps(body, lm))
            return np.where(slow_mask, modes["n_slow"], modes["n_fast"])

        d1 = (n_of(lam + h) - n_of(lam - h)) / (2.0 * h)
        return n_of(lam) - lam * d1

    # ------------------------------------------------------------------
    def _biaxial_children(self, fid, grp, entering, n_hat, cos_i):
        """Biaxial-crystal boundary: slow/fast two-sheet double refraction.

        Structurally parallel to _birefringent_children with o/e replaced
        by slow/fast (pol_mode 2/3). Both sheets are 'extraordinary-like':
        ray (Poynting) and wavevector differ, and because the biaxial
        ray->k inversion has no closed form the unit wavevector is carried
        explicitly in batch.k_dir. Fresnel amplitudes use the same
        effective-index approximation as uniaxial (each sheet's n_phase);
        the dropped cross terms land in absorbed_surface via the exact
        power difference, so closure holds by construction. Internal
        reflections are sheet-preserving (reflect_internal_biaxial); rays
        with no same-sheet returning root (conical corner cases) drop
        their reflected share into absorbed_surface.
        Coating/roughness on biaxial faces are not modeled (warned)."""
        from . import birefringence as bir
        scene = self.scene
        body = scene.body_of_face(fid)
        face = scene.faces[fid]
        m = len(grp)
        if (scene.face_coatings.get(fid) is not None
                or scene.roughness.get(fid) is not None) \
                and not getattr(self, "_warned_bir_extras", False):
            import warnings
            warnings.warn(
                "coating/roughness on birefringent face %s is not modeled "
                "(bare-interface Fresnel used)" % face.id)
            self._warned_bir_extras = True

        frame = body.crystal_frame
        eps = scene.biaxial_eps(body, grp.lam)         # (m,3)
        r_m = body.mirror
        a = body.absorbance
        phys = (1.0 - r_m) * (1.0 - a)
        p_in = grp.power
        p_accounted = np.zeros(m)
        out = []
        can_reflect = grp.generation < self.cfg.max_reflections

        # ---------------- ENTRY: outside -> crystal ----------------
        ent = np.where(entering)[0]
        if len(ent):
            sub = grp.select(ent)
            nh = n_hat[ent]
            ci = cos_i[ent]
            cur = sub.current_medium()
            n1 = np.empty(len(ent), dtype=np.complex128)
            for mm in np.unique(cur):
                s = cur == mm
                n1[s] = scene.medium_index(int(mm), sub.lam[s])
            res = bir.refract_in_biaxial(sub.dir, nh, frame, np.real(n1),
                                         eps[ent])
            # incident Jones in the interface (s,p) basis
            s_new, p_new = fr.pol_basis(sub.dir, nh)
            p_old = np.cross(sub.dir, sub.s_hat)
            Es_i, Ep_i = fr.rotate_jones(sub.Es, sub.Ep, sub.s_hat, p_old,
                                         s_new, p_new)
            # unitary slow/fast channel decomposition at the incident k
            # (same closure-preserving pattern as the uniaxial o/e split:
            # each channel gets its own sheet-index Fresnel)
            in_modes = bir.biaxial_modes_for_k(sub.dir, frame, eps[ent])
            d_sl_i, d_fa_i = in_modes["D_slow"], in_modes["D_fast"]
            E1_i, E2_i = fr.rotate_jones(Es_i, Ep_i, s_new, p_new,
                                         d_sl_i, d_fa_i)
            cs = np.sum(d_sl_i * s_new, axis=-1)
            sn = np.sum(d_sl_i * p_new, axis=-1)

            coeffs = {}
            for name in ("slow", "fast"):
                n2eff = res["n_phase_%s" % name].astype(np.complex128)
                rs, rp, ts, tp, ct = fr.fresnel_coeffs(ci, n1, n2eff)
                _, _, Ts, Tp = fr.power_coeffs(rs, rp, ts, tp, ci, ct,
                                               n1, n2eff)
                coeffs[name] = (rs, rp, ts, tp, np.clip(Ts, 0.0, None),
                                np.clip(Tp, 0.0, None))

            # reflected child: coherent sum of both channels' reflections
            rs1, rp1 = coeffs["slow"][0], coeffs["slow"][1]
            rs2, rp2 = coeffs["fast"][0], coeffs["fast"][1]
            refl = sub.select(np.ones(len(ent), dtype=bool))
            _kill_differentials(refl)
            refl.dir = fr.reflect_dir(sub.dir, nh)
            refl.s_hat = s_new
            amp = {}
            for tag, r in (("s1", rs1), ("p1", rp1), ("s2", rs2),
                           ("p2", rp2)):
                amp[tag] = np.sqrt(r_m + phys * np.abs(r) ** 2) \
                    * np.exp(1j * np.angle(r))
            refl.Es = E1_i * cs * amp["s1"] - E2_i * sn * amp["s2"]
            refl.Ep = E1_i * sn * amp["p1"] + E2_i * cs * amp["p2"]
            self._record_reflection(refl, fid)
            refl.generation += 1
            cr = can_reflect[ent]
            if np.any(~cr):
                self.ledger.credit("truncated_generation",
                                   refl.source_id[~cr], refl.power[~cr])
            keep = cr & (refl.power > 0)
            if np.any(keep):
                out.append(refl.select(keep))
            p_accounted[ent] += refl.power

            # transmitted sheet children (slow: pol_mode 2, fast: 3)
            for name, mode_val, E_ch, ch_s, ch_p in (
                    ("slow", 2, E1_i, cs, sn),
                    ("fast", 3, E2_i, -sn, cs)):
                rs, rp, ts, tp, Ts, Tp = coeffs[name]
                k_hat = res["k_%s" % name]
                s_ray = res["s_%s" % name]
                D = res["D_%s" % name]
                amp_ts = np.sqrt(phys * Ts) * np.exp(1j * np.angle(ts))
                amp_tp = np.sqrt(phys * Tp) * np.exp(1j * np.angle(tp))
                ch = sub.select(np.ones(len(ent), dtype=bool))
                _kill_differentials(ch)
                ch.dir = s_ray
                # Jones basis: s_hat _|_ (ray, D-ish) so the full field
                # sits in Ep along the projected D direction — the same
                # single-mode bookkeeping the uniaxial e-child uses
                sh = np.cross(s_ray, D)
                nrm = np.linalg.norm(sh, axis=-1, keepdims=True)
                # degenerate guard (D ~ parallel to ray cannot really
                # happen off the optic axes; fall back to the interface s)
                bad = nrm[:, 0] < 1e-12
                sh = np.where(bad[:, None], s_new, sh / np.maximum(
                    nrm, 1e-300))
                ch.s_hat = sh
                # channel field through its own Fresnel, magnitude only
                # (the child is a single pure mode)
                Ech_s = E_ch * ch_s * amp_ts
                Ech_p = E_ch * ch_p * amp_tp
                mag = np.sqrt(np.abs(Ech_s) ** 2 + np.abs(Ech_p) ** 2)
                phase = np.exp(1j * np.angle(
                    np.where(np.abs(Ech_p) >= np.abs(Ech_s), Ech_p,
                             Ech_s)))
                ch.Es = np.zeros_like(Ech_s)
                ch.Ep = mag * phase
                ch.pol_mode[:] = mode_val
                ch.n_eff[:] = res["n_ray_%s" % name]
                if self.cfg.track_time:
                    ch.n_g_eff[:] = self._biaxial_group_index(
                        body, ch.lam, k_hat, frame, name == "slow")
                if ch.k_dir is None:
                    ch.k_dir = np.full((len(ent), 3), np.nan)
                ch.k_dir[...] = k_hat
                ok = ~res["tir_%s" % name]
                ch.push_medium(ok, np.full(len(ent), body.index))
                keep = ok & (ch.power > 0)
                if np.any(keep):
                    out.append(ch.select(keep))
                p_accounted[ent] += ch.power * ok

        # ---------------- EXIT: crystal -> outside ----------------
        exi = np.where(~entering)[0]
        if len(exi):
            sub = grp.select(exi)
            nh = n_hat[exi]
            depth = sub.depth
            under = np.where(depth >= 2,
                             sub.medium[np.arange(len(exi)),
                                        np.maximum(depth - 2, 0)],
                             AMBIENT).astype(np.int64)
            n2 = np.empty(len(exi), dtype=np.complex128)
            for mm in np.unique(under):
                s = under == mm
                n2[s] = scene.medium_index(int(mm), sub.lam[s])
            is_slow = sub.pol_mode == 2
            # the carried unit wavevector + its sheet's phase index
            k_hat = sub.k_dir
            if k_hat is None:
                # defensive: a biaxial exit without an entry can only be
                # authoring error (source inside the crystal)
                raise RuntimeError(
                    "biaxial exit through %s without carried k_dir "
                    "(source inside a biaxial solid is unsupported)"
                    % face.id)
            modes = bir.biaxial_modes_for_k(k_hat, frame, eps[exi])
            n_phase = np.where(is_slow, modes["n_slow"], modes["n_fast"])
            K_int = n_phase[:, None] * k_hat
            cos_k = np.clip(-np.sum(k_hat * nh, axis=-1), 0.0, 1.0)
            d_out, tir = bir.refract_out_biaxial(K_int, nh, np.real(n2))
            rs, rp, ts, tp, ct = fr.fresnel_coeffs(
                cos_k, n_phase.astype(np.complex128), n2)
            Rs, Rp, Ts, Tp = fr.power_coeffs(
                rs, rp, ts, tp, cos_k, ct,
                n_phase.astype(np.complex128), n2)
            Ts = np.clip(Ts, 0.0, None)
            Tp = np.clip(Tp, 0.0, None)
            s_new, p_new = fr.pol_basis(k_hat, nh)
            p_old = np.cross(sub.dir, sub.s_hat)
            Es_i, Ep_i = fr.rotate_jones(sub.Es, sub.Ep, sub.s_hat, p_old,
                                         s_new, p_new)

            # transmitted child: leaves the crystal, mode resets
            tr = sub.select(np.ones(len(exi), dtype=bool))
            _kill_differentials(tr)
            tr.dir = d_out
            tr.s_hat = s_new
            tr.Es = Es_i * np.sqrt(phys * Ts) * np.exp(1j * np.angle(ts))
            tr.Ep = Ep_i * np.sqrt(phys * Tp) * np.exp(1j * np.angle(tp))
            tr.pol_mode[:] = 0
            tr.n_eff[:] = 0.0
            tr.n_g_eff[:] = 0.0
            ok_t = ~tir
            tr.pop_medium(ok_t, np.full(len(exi), body.index))
            keep = ok_t & (tr.power > 0)
            if np.any(keep):
                out.append(tr.select(keep))
            p_accounted[exi] += tr.power * ok_t

            # internal reflection: sheet-preserving quartic re-solve
            K_refl, ok_r = bir.reflect_internal_biaxial(K_int, nh, frame,
                                                        eps[exi])
            if np.any(ok_r):
                k_r_hat = K_refl / np.linalg.norm(K_refl, axis=-1,
                                                  keepdims=True)
                s_ray, _, n_ray = bir.biaxial_ray_from_k(K_refl, frame,
                                                         eps[exi])
                rf = sub.select(np.ones(len(exi), dtype=bool))
                _kill_differentials(rf)
                rf.dir = s_ray
                rf.s_hat = s_new
                rf.Es = Es_i * np.sqrt(r_m + phys * np.abs(rs) ** 2) \
                    * np.exp(1j * np.angle(rs))
                rf.Ep = Ep_i * np.sqrt(r_m + phys * np.abs(rp) ** 2) \
                    * np.exp(1j * np.angle(rp))
                self._record_reflection(rf, fid)
                rf.generation += 1
                rf.n_eff[...] = n_ray
                if self.cfg.track_time:
                    rf.n_g_eff[...] = self._biaxial_group_index(
                        body, rf.lam, k_r_hat, frame, is_slow)
                rf.k_dir[...] = k_r_hat
                cr = can_reflect[exi]
                if np.any(~cr):
                    self.ledger.credit("truncated_generation",
                                       rf.source_id[~cr & ok_r],
                                       rf.power[~cr & ok_r])
                keep = cr & ok_r & (rf.power > 0)
                if np.any(keep):
                    out.append(rf.select(keep))
                # rays without a same-sheet return (~ok_r) fall through to
                # the absorbed_surface exact difference below
                p_accounted[exi] += rf.power * ok_r

        absorbed = np.clip(p_in - p_accounted, 0.0, None)
        if np.any(absorbed > 0):
            self.ledger.credit("absorbed_surface", grp.source_id, absorbed,
                               where=body.label)
        self._flux_out_children(body, out)
        return RayBatch.concatenate(out) if out else None

    # ------------------------------------------------------------------
    def _apply_floors(self, batch):
        floor = self.cfg.power_floor * batch.birth_power
        weak = batch.power < floor
        if np.any(weak):
            self.ledger.credit("truncated_power", batch.source_id[weak],
                               batch.power[weak])
        keep = ~weak
        return batch.select(keep) if np.any(weak) else batch
