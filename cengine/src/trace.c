/* ===========================================================================
 * trace.c — source sampling + the propagation loop.
 *
 * Faithful port of scripts/raytracer/tracer.py (step semantics, child
 * amplitudes, ledger crediting — line references inline) and
 * sources.py sample_source, restructured from numpy batch-vectorized to
 * per-ray scalar under OpenMP:
 *
 *   while queue not empty (LIFO, iteration-capped like tracer.py:158):
 *     batch = pop()
 *     parallel for over rays (schedule(static)):
 *       process_ray -> children into the thread's arena, losses into the
 *       thread's ledger, detector hits / viz rows into thread buffers
 *     serial: merge arenas/ledgers/buffers in THREAD ORDER (deterministic
 *     for a fixed thread count), splat detector hits, split oversized
 *     child batches (tracer.py:167-173)
 *
 * Determinism: every random draw is a pure function of (seed, ray lineage)
 * — see rng.h. Thread scheduling only permutes float accumulation order
 * (ledger sums, splat adds), bounded by ~1 ulp, far inside every gate.
 * =========================================================================== */
#include "trace.h"
#include "log.h"
#include "kernels/fresnel.h"
#include "kernels/thinfilm.h"
#include "rng.h"

#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* reserved event_ctr domain for emission-time draws (interactions count
 * 0, 1, 2, ... and never reach this range) */
#define EV_EMIT_POS   0xF0000000u
#define EV_EMIT_PHASE 0xF0000001u

/* ------------------------------------------------------------------ viz */
static void vizvec_init(VizVec *v) {
    v->cap = 1024;
    v->n = 0;
    v->v = (VizRec *)malloc((size_t)v->cap * sizeof(VizRec));
    if (!v->v) die(EXIT_PHYSICS, "viz: allocation failed");
}

static void vizvec_push(VizVec *v, const VizRec *r) {
    if (v->n == v->cap) {
        v->cap *= 2;
        VizRec *p = (VizRec *)realloc(v->v, (size_t)v->cap * sizeof(VizRec));
        if (!p) die(EXIT_PHYSICS, "viz: growth failed");
        v->v = p;
    }
    v->v[v->n++] = *r;
}

/* VizStore.add row layout (tracer.py:102-117, run_trace.py:12-17):
 * [source_id, lam, power, x0..z0, x1..z1, pol_mode, rel_power, opl0, opl1] */
static void viz_add(VizVec *viz, const Ray *r, kvec3 p1, double opl0,
                    double opl1) {
    double power = ray_power(r);
    double rel = 0.0;
    if (r->birth_power > 0.0) {
        rel = power / r->birth_power;
        if (rel < 0.0) rel = 0.0;
        if (rel > 1.0) rel = 1.0;
    }
    VizRec rec;
    rec.row[0] = (double)r->source_id;
    rec.row[1] = r->lam;
    rec.row[2] = power;
    rec.row[3] = r->pos.x; rec.row[4] = r->pos.y; rec.row[5] = r->pos.z;
    rec.row[6] = p1.x;     rec.row[7] = p1.y;     rec.row[8] = p1.z;
    rec.row[9] = (double)r->pol_mode;
    rec.row[10] = rel;
    rec.row[11] = opl0;
    rec.row[12] = opl1;
    vizvec_push(viz, &rec);
}

/* --------------------------------------------------------- thread context */
typedef struct {
    RayVec children;
    LedgerC ledger;
    DetHitVec hits;
    VizVec viz;
    int64_t interactions;
} ThreadCtx;

/* Power floor (tracer.py _apply_floors, 1447-1454): children below
 * power_floor * birth_power are killed into truncated_power. Applied at
 * child creation — same accounting as Python's post-concatenate pass. */
static void push_child(const SceneC *s, ThreadCtx *cx, const Ray *child) {
    double p = ray_power(child);
    if (p < s->power_floor * child->birth_power) {
        ledger_credit(&cx->ledger, BK_TRUNCATED_POWER, child->source_id, p);
        return;
    }
    rayvec_push(&cx->children, child);
}

/* medium stack ops — port of rays.py push_medium/pop_medium including the
 * hard errors (overlapping-solids diagnostics must not be lost in C) */
static void push_medium(Ray *r, int16_t body_index, const SceneC *s,
                        const char *face_id) {
    if (r->depth >= MEDIUM_STACK_DEPTH)
        die(EXIT_PHYSICS,
            "medium stack overflow at face %s (solids nested > %d deep) — "
            "check for overlapping solids", face_id, MEDIUM_STACK_DEPTH);
    r->medium[r->depth++] = body_index;
    (void)s;
}

static void pop_medium(Ray *r, int16_t expect_body, const SceneC *s,
                       const char *face_id) {
    if (r->depth <= 0)
        die(EXIT_PHYSICS,
            "medium stack underflow at face %s — ray exits body '%s' it "
            "never entered (overlapping solids or orientation bug)",
            face_id, s->bodies[expect_body].label);
    int16_t top = r->medium[r->depth - 1];
    if (top != expect_body)
        die(EXIT_PHYSICS,
            "medium stack pop mismatch at face %s: expected body '%s' got "
            "'%s' — non-manifold nesting (overlapping solids?)", face_id,
            s->bodies[expect_body].label,
            top >= 0 ? s->bodies[top].label : "ambient");
    r->medium[--r->depth] = AMBIENT;
}

/* flux_out helper (tracer.py _flux_out_children): a child leaving this
 * interface OUTSIDE the body counts as power flowing out of the element */
static void flux_out_child(LedgerC *l, const BodyC *body, const Ray *child) {
    if (ray_current_medium(child) != body->index)
        l->flux_out[body->index] += ray_power(child);
}

/* ---------------------------------------------------- screen children */
/* Port of Tracer._screen_children (tracer.py:420-454): ideal thin screen —
 * mirror fraction specular-reflects (with the -sqrt(r) half-wave sign),
 * absorbance eats its share, remainder continues UNREFRACTED. */
static void screen_children(const SceneC *s, const FaceC *face,
                            const BodyC *body, const Ray *r, ThreadCtx *cx) {
    double p_in = ray_power(r);
    cx->ledger.flux_in[body->index] += p_in;
    double r_m = body->mirror;
    double a = body->absorbance;

    double t_frac = (1.0 - r_m) * (1.0 - a);
    if (t_frac > 0.0) {
        Ray tr = *r;
        double sq = sqrt(t_frac);
        tr.Es = kc_scale(tr.Es, sq);
        tr.Ep = kc_scale(tr.Ep, sq);
        tr.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                   CHILD_SLOT_TRANSMIT);
        tr.event_ctr = 0;
        flux_out_child(&cx->ledger, body, &tr);
        push_child(s, cx, &tr);
    }
    if (r_m > 0.0) {
        kvec3 n_out = v3_scale(face_normal_canonical(face, r->pos),
                               face->outward_sign);
        double dt = v3_dot(n_out, r->dir);
        double sgn = (dt < 0.0) ? 1.0 : ((dt > 0.0) ? -1.0 : 0.0);
        kvec3 n_hat = v3_scale(n_out, sgn);
        Ray rf = *r;
        rf.dir = fresnel_reflect_dir(r->dir, n_hat);
        double sq = sqrt(r_m);
        /* -sqrt(r): the idealized mirror's half-wave phase
         * (tracer.py:444-445) */
        rf.Es = kc_scale(rf.Es, -sq);
        rf.Ep = kc_scale(rf.Ep, -sq);
        rf.generation += 1;   /* screen reflection is NOT cap-checked here —
                               * tracer.py:447 has no can_reflect guard;
                               * the cap catches it at the next optic */
        rf.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                   CHILD_SLOT_REFLECT);
        rf.event_ctr = 0;
        flux_out_child(&cx->ledger, body, &rf);
        push_child(s, cx, &rf);
    }
    double ab = (1.0 - r_m) * a;
    if (ab > 0.0) {
        ledger_credit(&cx->ledger, BK_ABSORBED_SURFACE, r->source_id,
                      p_in * ab);
        cx->ledger.surf_by_body[body->index] += p_in * ab;
    }
}

/* Port of Tracer._apply_polarizer (tracer.py:899-946): dichroic Jones
 * diattenuator in place on the transmitted ray. Transmission axis = the
 * body's global axis projected transverse to the ray; T_par/T_perp are
 * pre-resolved at the ray's stratum wavelength (plan D1). Circular
 * polarizers add an ideal retarder with fast axis at +-45 deg. */
static void apply_polarizer(const SceneC *s, const BodyC *body, Ray *trans,
                            int lam_idx) {
    kvec3 d = trans->dir;
    kvec3 axis = body->pol_axis;
    kvec3 t = v3_sub(axis, v3_scale(d, v3_dot(d, axis)));
    double nrm = v3_norm(t);
    int good = nrm > 1e-9;
    /* (anti)parallel rays see no transverse axis: attenuate both
     * components by the mean (film looks isotropic edge-on) */
    kvec3 t_hat = good ? v3_scale(t, 1.0 / (nrm > 1e-300 ? nrm : 1e-300))
                       : trans->s_hat;
    kvec3 p_of_t = v3_cross(d, t_hat);
    kvec3 p_old = v3_cross(d, trans->s_hat);
    kcplx Et, Ep_;
    fresnel_rotate_jones(trans->Es, trans->Ep, trans->s_hat, p_old,
                         t_hat, p_of_t, &Et, &Ep_);
    double T_par = body->pol_T_par[lam_idx];
    double T_perp = body->pol_T_perp[lam_idx];
    double a_par = sqrt(good ? T_par : 0.5 * (T_par + T_perp));
    double a_perp = sqrt(good ? T_perp : 0.5 * (T_par + T_perp));
    Et = kc_scale(Et, a_par);
    Ep_ = kc_scale(Ep_, a_perp);
    if (body->pol_type != POL_LINEAR) {
        /* ideal retarder, fast axis +-45 deg to the transmission axis:
         * J = R(-th) diag(1, e^{-i delta}) R(th), th = sgn*45 deg */
        double delta = K_TWO_PI * body->retardance_waves;
        double sgn = (body->pol_type == POL_CIRCULAR_RIGHT) ? 1.0 : -1.0;
        double c = sqrt(0.5);
        double sn = sgn * sqrt(0.5);
        kcplx e = kc_cis(-delta);
        kcplx j11 = kc_add(kc(c * c, 0.0), kc_scale(e, sn * sn));
        kcplx j12 = kc_sub(kc(c * sn, 0.0), kc_scale(e, sn * c));
        kcplx j22 = kc_add(kc(sn * sn, 0.0), kc_scale(e, c * c));
        kcplx Et2 = kc_add(kc_mul(j11, Et), kc_mul(j12, Ep_));
        kcplx Ep2 = kc_add(kc_mul(j12, Et), kc_mul(j22, Ep_));
        Et = Et2;
        Ep_ = Ep2;
    }
    trans->Es = Et;
    trans->Ep = Ep_;
    trans->s_hat = t_hat;
}

/* ----------------------------------------------------- optic children */
/* Port of Tracer._optic_children (tracer.py:457-896), phase-A subset:
 * bare Fresnel + mirror/absorbance + medium stack + seam guard. Coatings/
 * roughness/scatter/polarizer/birefringence arrive in phases B/E/F —
 * feature routing keeps scenes that use them on the Python engine. */
static void optic_children(const SceneC *s, const FaceC *face,
                           const BodyC *body, const Ray *r, ThreadCtx *cx) {
    kvec3 n_out = v3_scale(face_normal_canonical(face, r->pos),
                           face->outward_sign);
    double cos_with_out = v3_dot(r->dir, n_out);
    int entering = cos_with_out < 0.0;

    /* seam-leak guard (tracer.py:467-482): medium stack disagrees with the
     * crossing direction -> the ray slipped between trimmed faces */
    int top = ray_current_medium(r);
    int leak = entering ? (top == body->index) : (top != body->index);
    if (leak) {
        double p = ray_power(r);
        ledger_credit(&cx->ledger, BK_SEAM_LOSS, r->source_id, p);
        cx->ledger.by_body[body->index + 1] += p;
        return;
    }

    kvec3 n_hat = entering ? n_out : v3_scale(n_out, -1.0);
    double cos_i = -v3_dot(r->dir, n_hat);
    if (cos_i < 0.0) cos_i = 0.0;
    if (cos_i > 1.0) cos_i = 1.0;

    if (entering)
        cx->ledger.flux_in[body->index] += ray_power(r);

    /* media on both sides (tracer.py:512-536): entering -> far side is the
     * body; exiting -> the medium UNDER the top of the stack */
    kcplx n1 = scene_medium_n(s, top, r->lam_idx);
    int far;
    if (entering) {
        far = body->index;
    } else {
        far = (r->depth >= 2) ? r->medium[r->depth - 2] : AMBIENT;
    }
    kcplx n2 = scene_medium_n(s, far, r->lam_idx);

    /* amplitude coefficients: bare Fresnel, TMM coating, or measured
     * coating table (tracer.py:538-584) */
    kcplx rs, rp, ts, tp;
    double Rs, Rp, Ts, Tp;
    if (face->coating >= 0
            && s->coatings[face->coating].kind == COAT_TMM) {
        const CoatC *c = &s->coatings[face->coating];
        kcplx layer_n[COAT_MAX_LAYERS];
        for (int j = 0; j < c->n_layers; j++) {
            size_t at = (size_t)j * s->n_lams + r->lam_idx;
            layer_n[j] = kc(c->layer_n_re[at], c->layer_n_im[at]);
        }
        TmmC T = tmm_eval(r->lam, cos_i, n1, n2, layer_n, c->layer_d,
                          c->n_layers);
        rs = T.rs; rp = T.rp; ts = T.ts; tp = T.tp;
        Rs = T.Rs; Rp = T.Rp; Ts = T.Ts; Tp = T.Tp;
    } else if (face->coating >= 0) {
        /* tabulated coating: measured powers at the ray wavelength with
         * the BARE-interface Fresnel phase; past the critical angle the
         * table's T folds into R (tracer.py:549-578) */
        const CoatC *c = &s->coatings[face->coating];
        FresnelC F = fresnel_eval(cos_i, n1, n2);
        Rs = c->Rs[r->lam_idx];
        Rp = c->Rp[r->lam_idx];
        Ts = c->Ts[r->lam_idx];
        Tp = c->Tp[r->lam_idx];
        if (fresnel_is_tir(cos_i, n1, n2)) {
            Rs = Rs + Ts; if (Rs > 1.0) Rs = 1.0; if (Rs < 0.0) Rs = 0.0;
            Rp = Rp + Tp; if (Rp > 1.0) Rp = 1.0; if (Rp < 0.0) Rp = 0.0;
            Ts = 0.0;
            Tp = 0.0;
        }
        rs = kc_scale(kc_cis(kc_arg(F.rs)), sqrt(Rs));
        rp = kc_scale(kc_cis(kc_arg(F.rp)), sqrt(Rp));
        ts = kc_scale(kc_cis(kc_arg(F.ts)), sqrt(Ts > 0.0 ? Ts : 0.0));
        tp = kc_scale(kc_cis(kc_arg(F.tp)), sqrt(Tp > 0.0 ? Tp : 0.0));
    } else {
        FresnelC F = fresnel_eval(cos_i, n1, n2);
        rs = F.rs; rp = F.rp; ts = F.ts; tp = F.tp;
        Rs = F.Rs; Rp = F.Rp; Ts = F.Ts; Tp = F.Tp;
    }
    if (Ts < 0.0) Ts = 0.0;                /* tracer.py:583-584 clip */
    if (Tp < 0.0) Tp = 0.0;

    /* rotate Jones into this interface's (s,p) basis (tracer.py:586-590) */
    kvec3 s_new, p_new;
    fresnel_pol_basis(r->dir, n_hat, &s_new, &p_new);
    kvec3 p_old = v3_cross(r->dir, r->s_hat);
    kcplx Es, Ep;
    fresnel_rotate_jones(r->Es, r->Ep, r->s_hat, p_old, s_new, p_new,
                         &Es, &Ep);

    double r_m = body->mirror;
    double a = body->absorbance;
    double phys = (1.0 - r_m) * (1.0 - a);
    double p_in = kc_abs2(Es) + kc_abs2(Ep);

    /* ---- reflected child (tracer.py:647-681): power-exact amplitude
     * sqrt(r_m + phys |r|^2), phase from the physical coefficient ---- */
    kcplx amp_rs = kc_scale(kc_cis(kc_arg(rs)),
                            sqrt(r_m + phys * kc_abs2(rs)));
    kcplx amp_rp = kc_scale(kc_cis(kc_arg(rp)),
                            sqrt(r_m + phys * kc_abs2(rp)));
    Ray refl = *r;
    refl.dir = fresnel_reflect_dir(r->dir, n_hat);
    refl.s_hat = s_new;
    refl.Es = kc_mul(Es, amp_rs);
    refl.Ep = kc_mul(Ep, amp_rp);
    refl.generation = (int16_t)(r->generation + 1);
    refl.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                 CHILD_SLOT_REFLECT);
    refl.event_ctr = 0;
    double p_refl = ray_power(&refl);
    int can_reflect = r->generation < s->max_reflections;
    if (!can_reflect) {
        ledger_credit(&cx->ledger, BK_TRUNCATED_GENERATION, r->source_id,
                      p_refl);
    } else if (p_refl > 0.0) {
        flux_out_child(&cx->ledger, body, &refl);
        push_child(s, cx, &refl);
    }

    /* ---- transmitted child (tracer.py:683-726) ---- */
    int tir = (Ts + Tp) <= 1e-15;
    double p_trans_pre = 0.0;   /* PRE-polarizer power for the exact-
                                 * difference accounting (tracer.py:715) */
    if (!tir) {
        Ray trans = *r;
        trans.dir = fresnel_refract_dir(r->dir, n_hat, cos_i, n1.re, n2.re);
        trans.s_hat = s_new;
        kcplx amp_ts = kc_scale(kc_cis(kc_arg(ts)), sqrt(phys * Ts));
        kcplx amp_tp = kc_scale(kc_cis(kc_arg(tp)), sqrt(phys * Tp));
        trans.Es = kc_mul(Es, amp_ts);
        trans.Ep = kc_mul(Ep, amp_tp);
        /* medium bookkeeping (tracer.py:705-710) */
        if (entering)
            push_medium(&trans, (int16_t)body->index, s, face->id);
        else
            pop_medium(&trans, (int16_t)body->index, s, face->id);
        trans.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                      CHILD_SLOT_TRANSMIT);
        trans.event_ctr = 0;
        p_trans_pre = ray_power(&trans);
        /* ---- polarizer: dichroic Jones diattenuator on ENTRY
         * (tracer.py:711-723, _apply_polarizer 899-946) ---- */
        if (body->has_polarizer && entering) {
            apply_polarizer(s, body, &trans, r->lam_idx);
            double d_loss = p_trans_pre - ray_power(&trans);
            if (d_loss > 0.0) {
                ledger_credit(&cx->ledger, BK_POLARIZER_ABSORBED,
                              r->source_id, d_loss);
                cx->ledger.by_body[body->index + 1] += d_loss;
            }
        }
        if (ray_power(&trans) > 0.0) {
            flux_out_child(&cx->ledger, body, &trans);
            push_child(s, cx, &trans);
        }
    }

    /* ---- surface absorption = exact power difference (tracer.py:888-894)
     * pre-polarizer (that loss has its own bucket); generation-capped
     * reflection already credited; TIR kills the transmitted child ---- */
    double absorbed = p_in - (p_refl + p_trans_pre);
    if (absorbed > 0.0) {
        ledger_credit(&cx->ledger, BK_ABSORBED_SURFACE, r->source_id,
                      absorbed);
        cx->ledger.surf_by_body[body->index] += absorbed;
    }
}

/* -------------------------------------------------------- detector event */
/* Port of Tracer._detector_event (tracer.py:339-372), incoherent side only
 * (the coherent gather is phase D; request_load rejects coherent sources
 * until then). */
static void detector_event(const SceneC *s, const FaceC *face, const Ray *r,
                           ThreadCtx *cx) {
    const DetC *d = &s->dets[face->detector];
    if (r->coherent)
        die(EXIT_PHYSICS, "coherent ray reached detector '%s' — the "
            "coherent gather is not ported yet; the feature router should "
            "have kept this scene on the Python engine", d->label);
    DetHit h;
    h.det = face->detector;
    double fx, fy;
    det_to_grid(d, r->pos, &fx, &fy);
    h.fx = (float)fx;
    h.fy = (float)fy;
    h.bin = det_lam_bin(d, r->lam);
    h.power = ray_power(r);
    h.source_id = r->source_id;
    h.lam_stratum = r->lam_stratum;
    h.pol_stratum = r->pol_stratum;
    dethits_push(&cx->hits, &h);
    /* diagnostic tallies (NOT closure buckets — tracer.py:366-372) */
    cx->ledger.surf_by_det[face->detector] += h.power;
    cx->ledger.detected[face->detector] += h.power;
}

/* ------------------------------------------------------------ process_ray */
/* Port of one ray's share of Tracer.step (tracer.py:183-322). */
static void process_ray(const SceneC *s, Ray *r, ThreadCtx *cx) {
    cx->interactions++;
    int32_t fid;
    double t = scene_intersect(s, r->pos, r->dir, &fid);
    int hit = fid >= 0;

    /* ---- bulk absorption + phase over the segment (tracer.py:202-240) */
    int med = ray_current_medium(r);
    kcplx n_med = scene_medium_n(s, med, r->lam_idx);
    double seg = hit ? t : 0.0;    /* escaped rays: no traversal loss */
    double alpha = 4.0 * K_PI * n_med.im / r->lam
                   + scene_filter_alpha(s, med, r->lam_idx);
    double x = alpha * seg;
    if (x < 0.0) x = 0.0;
    if (x > 700.0) x = 700.0;
    double trans = exp(-x);
    double p_before = ray_power(r);
    double sq = sqrt(trans);
    r->Es = kc_scale(r->Es, sq);
    r->Ep = kc_scale(r->Ep, sq);
    double absorbed = p_before * (1.0 - trans);
    if (absorbed > 0.0) {
        ledger_credit(&cx->ledger, BK_ABSORBED_BULK, r->source_id, absorbed);
        cx->ledger.by_body[med + 1] += absorbed;   /* slot 0 = ambient */
    }
    double start_opl = r->opl;
    double n_phase = (r->n_eff > 0.0) ? r->n_eff : n_med.re;
    r->opl += n_phase * seg;

    /* ---- viz segment (tracer.py:254-262): escaped rays draw a 0.25 m
     * stub with a synthetic opl1 (the real opl is untouched, seg=0) ---- */
    if (r->viz_flag) {
        kvec3 p1 = v3_fma(r->pos, hit ? t : 0.25, r->dir);
        double opl1 = hit ? r->opl : start_opl + n_phase * 0.25;
        viz_add(&cx->viz, r, p1, start_opl, opl1);
    }

    /* ---- escaped (tracer.py:264-267) ---- */
    if (!hit) {
        ledger_credit(&cx->ledger, BK_ESCAPED, r->source_id, ray_power(r));
        return;
    }

    /* advance to the surface and dispatch by face role
     * (tracer.py:282-315) */
    r->pos = v3_fma(r->pos, t, r->dir);
    r->last_face = fid;
    r->event_ctr += 1;
    const FaceC *face = &s->faces[fid];
    const BodyC *body = &s->bodies[face->body];

    if (face->detector >= 0) {
        detector_event(s, face, r, cx);
        screen_children(s, face, body, r, cx);
    } else if (body->role == ROLE_DETECTOR) {
        /* non-screen face of a detector solid: strict no-op pass-through
         * (tracer.py:304-308) — the continuation is the same ray */
        Ray cont = *r;
        cont.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                     CHILD_SLOT_TRANSMIT);
        cont.event_ctr = 0;
        push_child(s, cx, &cont);
    } else {
        optic_children(s, face, body, r, cx);
    }
}

/* ---------------------------------------------------------- source sampling */
/* Port of sources.sample_source (sources.py:201-382) minus beam/apodization
 * (feature-routed to Python until ported). Every primary is a pure
 * function of (seed, source index, primary index). */

/* per-ray transverse polarization frame (sources._pol_reference_frame):
 * e_ref = global +z projected transverse to dir, fallback +y */
static kvec3 pol_reference_frame(kvec3 dir) {
    kvec3 z = v3(0.0, 0.0, 1.0);
    kvec3 ref = v3_sub(z, v3_scale(dir, v3_dot(dir, z)));
    double nrm = v3_norm(ref);
    if (nrm < 1e-9) {
        kvec3 y = v3(0.0, 1.0, 0.0);
        ref = v3_sub(y, v3_scale(dir, v3_dot(dir, y)));
        nrm = v3_norm(ref);
    }
    return v3_scale(ref, 1.0 / nrm);
}

/* rejection-sample one position on the emit face (sources.
 * _sample_face_points): uniform in the UV bbox (sphere: sin(v) uniform for
 * uniform area), accepted by the trim through the same to_uv roundtrip. */
static kvec3 sample_emit_position(const SourceC *src, uint64_t ray_key,
                                  const char *label) {
    const SurfC *surf = &src->emit_face.surf;
    int is_sphere = surf->kind == SURF_SPHERE;
    double sv_lo = 0.0, sv_hi = 0.0;
    if (is_sphere) {
        sv_lo = sin(src->v_lo);
        sv_hi = sin(src->v_hi);
    }
    /* Python retries ~60 rounds x1.8 oversampling; per-ray we allow 4096
     * attempts before declaring the trim geometry suspect */
    for (uint32_t a = 0; a < 4096; a++) {
        double du = rng_uniform(ray_key, EV_EMIT_POS, 2 * a);
        double dv = rng_uniform(ray_key, EV_EMIT_POS, 2 * a + 1);
        double u = src->u_lo + (src->u_hi - src->u_lo) * du;
        double v;
        if (is_sphere)
            v = asin(sv_lo + (sv_hi - sv_lo) * dv);
        else
            v = src->v_lo + (src->v_hi - src->v_lo) * dv;
        kvec3 p = surf_uv_to_xyz(surf, u, v);
        double qu, qv;
        surf_to_uv(surf, p, &qu, &qv);
        if (trim_contains(&src->emit_face.trim, qu, qv))
            return p;
    }
    die(EXIT_PHYSICS, "source face %s: area sampling failed to converge "
        "(4096 attempts) — trim geometry suspect", label);
}

/* Sample one source into a fresh batch. Mirrors the field assignments of
 * sources.py:323-354 one-for-one. */
static void sample_source_c(const SceneC *s, int source_index, RayVec *batch,
                            LedgerC *ledger) {
    const SourceC *src = &s->sources[source_index];
    int64_t n = s->rays;
    double p_ray = src->power_W / (double)n;

    for (int64_t i = 0; i < n; i++) {
        uint64_t key = rng_primary_key(s->seed, (uint32_t)source_index,
                                       (uint64_t)i);
        kvec3 pos = sample_emit_position(src, key, src->label);

        /* emitted power records EVERY sample (clipped ones immediately
         * balance into their bucket — sources.py:308-316) */
        ledger->emitted[source_index] += p_ray;

        kvec3 dir;
        if (src->emit_policy == EMIT_COLLIMATED) {
            dir = src->emit_dir;
        } else {
            kvec3 nrm = surf_normal(&src->emit_face.surf, pos);
            if (src->flip_all) {
                dir = v3_scale(nrm, -1.0);
            } else {
                /* per-sample toward-origin sign; against-origin samples
                 * are clipped (sources.py:249-265) */
                if (v3_dot(nrm, v3_scale(pos, -1.0)) < 0.0) {
                    ledger_credit(ledger, BK_EMISSION_CLIPPED,
                                  (int)source_index, p_ray);
                    continue;
                }
                dir = nrm;
            }
            dir = v3_unit(dir);
        }

        Ray r;
        memset(&r, 0, sizeof r);
        r.pos = pos;
        r.dir = dir;
        int16_t stratum = (int16_t)(i % src->n_strata);
        r.lam_stratum = stratum;
        r.lam_idx = (int16_t)(src->lam_offset + stratum);
        r.lam = s->lams_m[r.lam_idx];
        int16_t pol = (int16_t)((i / src->n_strata) % src->n_pol);
        r.pol_stratum = pol;
        r.source_id = (int16_t)source_index;
        r.coherent = src->coherent;
        r.birth_power = p_ray;
        r.s_hat = pol_reference_frame(dir);
        r.medium[0] = r.medium[1] = r.medium[2] = r.medium[3] = AMBIENT;
        r.depth = 0;
        r.last_face = -1;
        r.ray_key = key;
        r.event_ctr = 0;
        /* coherent: zero initial phase; incoherent: uniform random phase
         * (sources.py:346-349) */
        kcplx phase = kc(1.0, 0.0);
        if (!src->coherent)
            phase = kc_cis(K_TWO_PI * rng_uniform(key, EV_EMIT_PHASE, 0));
        double amp = sqrt(p_ray);
        r.Es = kc_scale(kc_mul(src->jones_s[pol], phase), amp);
        r.Ep = kc_scale(kc_mul(src->jones_p[pol], phase), amp);
        /* viz flag: the first viz_cap primaries of this source
         * (VizStore.flag_primaries, tracer.py:93-100) */
        r.viz_flag = (uint8_t)(batch->n < src->viz_cap);
        rayvec_push(batch, &r);
    }
}

/* ---------------------------------------------------------------- run */
void trace_run(SceneC *s, TraceResultC *out) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int n_threads = s->threads > 0 ? s->threads : omp_get_max_threads();
    LOGI("trace: %lld rays/source x %d sources, %d threads, seed %llu",
         (long long)s->rays, s->n_sources, n_threads,
         (unsigned long long)s->seed);

    ledger_init(&out->ledger, s);
    vizvec_init(&out->viz);
    out->rays_traced = 0;

    /* ---- work queue: LIFO of batches, seeded per source ---- */
    int q_cap = 64;
    int q_len = 0;
    RayVec *queue = (RayVec *)malloc((size_t)q_cap * sizeof(RayVec));
    if (!queue) die(EXIT_PHYSICS, "trace: queue allocation failed");

    for (int si = 0; si < s->n_sources; si++) {
        RayVec b;
        rayvec_init(&b, s->rays);
        sample_source_c(s, si, &b, &out->ledger);
        queue[q_len++] = b;
        log_progress("trace", 0.02 * (si + 1) / s->n_sources,
                     "sampled source %s (%lld rays)",
                     s->sources[si].label, (long long)b.n);
    }

    /* ---- per-thread contexts ---- */
    ThreadCtx *ctxs = (ThreadCtx *)calloc((size_t)n_threads,
                                          sizeof(ThreadCtx));
    if (!ctxs) die(EXIT_PHYSICS, "trace: thread context allocation failed");
    for (int i = 0; i < n_threads; i++) {
        rayvec_init(&ctxs[i].children, 4096);
        ledger_init(&ctxs[i].ledger, s);
        dethits_init(&ctxs[i].hits);
        vizvec_init(&ctxs[i].viz);
    }

    /* hard iteration cap (tracer.py:156-158): with the generation cap the
     * loop terminates naturally well before this */
    int64_t max_iter = 64ll * (s->max_reflections + 2);
    int64_t iter = 0;
    int64_t total_rays_est = (int64_t)s->n_sources * s->rays * 3;
    int64_t processed = 0;

    while (q_len > 0 && iter < max_iter) {
        iter++;
        RayVec batch = queue[--q_len];
        if (batch.n == 0) {
            rayvec_free(&batch);
            continue;
        }

        #pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            ThreadCtx *cx = &ctxs[tid];
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < batch.n; i++)
                process_ray(s, &batch.rays[i], cx);
        }
        processed += batch.n;
        rayvec_free(&batch);

        /* ---- deterministic serial merge, thread order ---- */
        int64_t n_children = 0;
        for (int i = 0; i < n_threads; i++)
            n_children += ctxs[i].children.n;

        if (n_children > 0) {
            RayVec merged;
            rayvec_init(&merged, n_children);
            for (int i = 0; i < n_threads; i++) {
                rayvec_extend(&merged, &ctxs[i].children);
                rayvec_clear(&ctxs[i].children);
            }
            /* split oversized child batches (tracer.py:166-173) */
            int n_parts = (int)(merged.n / s->batch_size) + 1;
            if (n_parts <= 1) {
                if (q_len == q_cap) {
                    q_cap *= 2;
                    queue = (RayVec *)realloc(queue,
                                              (size_t)q_cap * sizeof(RayVec));
                    if (!queue) die(EXIT_PHYSICS, "trace: queue growth "
                                    "failed");
                }
                queue[q_len++] = merged;
            } else {
                int64_t per = (merged.n + n_parts - 1) / n_parts;
                for (int p = 0; p < n_parts; p++) {
                    int64_t lo = p * per;
                    int64_t hi = lo + per;
                    if (hi > merged.n) hi = merged.n;
                    if (lo >= hi) break;
                    RayVec part;
                    rayvec_init(&part, hi - lo);
                    memcpy(part.rays, merged.rays + lo,
                           (size_t)(hi - lo) * sizeof(Ray));
                    part.n = hi - lo;
                    if (q_len == q_cap) {
                        q_cap *= 2;
                        queue = (RayVec *)realloc(
                            queue, (size_t)q_cap * sizeof(RayVec));
                        if (!queue) die(EXIT_PHYSICS, "trace: queue growth "
                                        "failed");
                    }
                    queue[q_len++] = part;
                }
                rayvec_free(&merged);
            }
        }

        for (int i = 0; i < n_threads; i++) {
            det_apply_hits(s, &ctxs[i].hits);
            dethits_clear(&ctxs[i].hits);
            ledger_merge(&out->ledger, &ctxs[i].ledger);
            /* zero the thread ledger for the next batch */
            ledger_free(&ctxs[i].ledger);
            ledger_init(&ctxs[i].ledger, s);
            for (int64_t k = 0; k < ctxs[i].viz.n; k++)
                vizvec_push(&out->viz, &ctxs[i].viz.v[k]);
            ctxs[i].viz.n = 0;
            out->rays_traced += ctxs[i].interactions;
            ctxs[i].interactions = 0;
        }

        if ((iter & 7) == 0)
            log_progress("trace",
                         0.05 + 0.9 * (double)processed
                             / (double)(processed + total_rays_est),
                         "iteration %lld: %lld rays processed",
                         (long long)iter, (long long)processed);
    }

    /* drain leftovers at the iteration cap (tracer.py:174-178) */
    for (int qi = 0; qi < q_len; qi++) {
        RayVec *b = &queue[qi];
        for (int64_t i = 0; i < b->n; i++)
            ledger_credit(&out->ledger, BK_TRUNCATED_GENERATION,
                          b->rays[i].source_id, ray_power(&b->rays[i]));
        rayvec_free(b);
    }
    free(queue);

    for (int i = 0; i < n_threads; i++) {
        rayvec_free(&ctxs[i].children);
        ledger_free(&ctxs[i].ledger);
        dethits_free(&ctxs[i].hits);
        free(ctxs[i].viz.v);
    }
    free(ctxs);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    out->trace_seconds = (double)(t1.tv_sec - t0.tv_sec)
                         + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    LOGI("trace: done — %lld ray interactions in %.3f s (%.3g rays/s), "
         "closure err max %.3g",
         (long long)out->rays_traced, out->trace_seconds,
         (double)out->rays_traced / out->trace_seconds,
         ledger_closure_max(&out->ledger));
    log_progress("trace", 1.0, "trace complete");
}

void trace_result_free(TraceResultC *r) {
    ledger_free(&r->ledger);
    free(r->viz.v);
    r->viz.v = NULL;
}
