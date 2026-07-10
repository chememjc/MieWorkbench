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
#include "kernels/scatterk.h"
#include "kernels/gratingk.h"
#include "kernels/birefk.h"
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
    GatherHitVec ghits;
    ExportVec exports;
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

/* ghost-analysis bookkeeping (tracer._record_reflection): stamp the face
 * id at slot min(pre-increment generation, HIST_DEPTH-1). Call with the
 * PARENT's generation (children here are built with generation+1). */
static void record_reflection(const SceneC *s, Ray *child,
                              int16_t parent_gen, int32_t fid) {
    if (!s->track_history) return;
    int slot = parent_gen < HIST_DEPTH - 1 ? parent_gen : HIST_DEPTH - 1;
    child->refl_hist[slot] = fid;
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
        record_reflection(s, &rf, r->generation, face - s->faces);
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

/* Interface amplitude/power coefficients at a given incidence cosine:
 * bare Fresnel, TMM coating (re-evaluated at the local angle), or a
 * measured table (macro values reused for microfacet lobes, documented
 * in tracer.py:758-768). NaN-poisoned grazing-lobe outputs are zeroed
 * exactly like tracer.py:774-784's nan_to_num. */
typedef struct {
    kcplx rs, rp, ts, tp;
    double Rs, Rp, Ts, Tp;
} IfcCoef;

static void nan_guard(IfcCoef *c) {
    if (!isfinite(c->rs.re) || !isfinite(c->rs.im)) c->rs = kc(0, 0);
    if (!isfinite(c->rp.re) || !isfinite(c->rp.im)) c->rp = kc(0, 0);
    if (!isfinite(c->ts.re) || !isfinite(c->ts.im)) c->ts = kc(0, 0);
    if (!isfinite(c->tp.re) || !isfinite(c->tp.im)) c->tp = kc(0, 0);
    if (!isfinite(c->Rs)) c->Rs = 0.0;
    if (!isfinite(c->Rp)) c->Rp = 0.0;
    if (!isfinite(c->Ts) || c->Ts < 0.0) c->Ts = 0.0;
    if (!isfinite(c->Tp) || c->Tp < 0.0) c->Tp = 0.0;
}

static IfcCoef interface_coeffs(const SceneC *s, const FaceC *face,
                                const Ray *r, double cos_x, kcplx n1,
                                kcplx n2, const IfcCoef *macro) {
    IfcCoef c;
    if (face->coating >= 0
            && s->coatings[face->coating].kind == COAT_TMM) {
        const CoatC *co = &s->coatings[face->coating];
        kcplx layer_n[COAT_MAX_LAYERS];
        for (int j = 0; j < co->n_layers; j++) {
            size_t at = (size_t)j * s->n_lams + r->lam_idx;
            layer_n[j] = kc(co->layer_n_re[at], co->layer_n_im[at]);
        }
        TmmC T = tmm_eval(r->lam, cos_x, n1, n2, layer_n, co->layer_d,
                          co->n_layers);
        c.rs = T.rs; c.rp = T.rp; c.ts = T.ts; c.tp = T.tp;
        c.Rs = T.Rs; c.Rp = T.Rp; c.Ts = T.Ts; c.Tp = T.Tp;
    } else if (face->coating >= 0 && macro) {
        c = *macro;     /* single-AOI table keeps its macro values */
    } else if (face->coating >= 0) {
        /* macro evaluation of a table coating (tracer.py:549-578) */
        const CoatC *co = &s->coatings[face->coating];
        FresnelC F = fresnel_eval(cos_x, n1, n2);
        c.Rs = co->Rs[r->lam_idx];
        c.Rp = co->Rp[r->lam_idx];
        c.Ts = co->Ts[r->lam_idx];
        c.Tp = co->Tp[r->lam_idx];
        if (fresnel_is_tir(cos_x, n1, n2)) {
            c.Rs += c.Ts; if (c.Rs > 1.0) c.Rs = 1.0; if (c.Rs < 0.0) c.Rs = 0.0;
            c.Rp += c.Tp; if (c.Rp > 1.0) c.Rp = 1.0; if (c.Rp < 0.0) c.Rp = 0.0;
            c.Ts = 0.0;
            c.Tp = 0.0;
        }
        c.rs = kc_scale(kc_cis(kc_arg(F.rs)), sqrt(c.Rs));
        c.rp = kc_scale(kc_cis(kc_arg(F.rp)), sqrt(c.Rp));
        c.ts = kc_scale(kc_cis(kc_arg(F.ts)),
                        sqrt(c.Ts > 0.0 ? c.Ts : 0.0));
        c.tp = kc_scale(kc_cis(kc_arg(F.tp)),
                        sqrt(c.Tp > 0.0 ? c.Tp : 0.0));
    } else {
        FresnelC F = fresnel_eval(cos_x, n1, n2);
        c.rs = F.rs; c.rp = F.rp; c.ts = F.ts; c.tp = F.tp;
        c.Rs = F.Rs; c.Rp = F.Rp; c.Ts = F.Ts; c.Tp = F.Tp;
    }
    nan_guard(&c);
    return c;
}

/* ------------------------------------------- birefringent children */
/* Port of Tracer._birefringent_children (tracer.py:949-1200): uniaxial
 * o/e double refraction. Entry: unitary o/e channel decomposition at the
 * incident k, per-channel effective-index Fresnel, o child (Snell, D
 * along e_o) + e child (normal-surface k, RAY along the walk-off
 * Poynting direction, phase index in n_eff) + coherent reflected child.
 * Exit: wavevector tangential continuity out; internal reflections are
 * mode-preserving. Coating/roughness on birefringent faces are not
 * modeled (same limitation as Python, which warns). */
static void biref_children(const SceneC *s, const FaceC *face,
                           const BodyC *body, const Ray *r, int entering,
                           kvec3 n_hat, double cos_i, ThreadCtx *cx) {
    double n_o = body->bir_n_o[r->lam_idx];
    double n_e = body->bir_n_e[r->lam_idx];
    kvec3 c = body->crystal_axis;
    double r_m = body->mirror;
    double a = body->absorbance;
    double phys = (1.0 - r_m) * (1.0 - a);
    double p_in = ray_power(r);
    double p_accounted = 0.0;
    int can_reflect = r->generation < s->max_reflections;

    if (entering) {
        /* ------------- ENTRY: outside -> crystal ------------- */
        kcplx n1c = scene_medium_n(s, ray_current_medium(r), r->lam_idx);
        double n1 = n1c.re;
        BirefIn bi = biref_refract_in(r->dir, n_hat, c, n1, n_o, n_e);

        kvec3 s_new, p_new;
        fresnel_pol_basis(r->dir, n_hat, &s_new, &p_new);
        kvec3 p_old = v3_cross(r->dir, r->s_hat);
        kcplx Es_i, Ep_i;
        fresnel_rotate_jones(r->Es, r->Ep, r->s_hat, p_old, s_new, p_new,
                             &Es_i, &Ep_i);
        /* unitary o/e channel decomposition at the incident k
         * (tracer.py:1005-1019) */
        kvec3 eo_i, ee_i;
        biref_eigenbasis(r->dir, c, &eo_i, &ee_i);
        kcplx Eo_i, Ee_i;
        fresnel_rotate_jones(Es_i, Ep_i, s_new, p_new, eo_i, ee_i,
                             &Eo_i, &Ee_i);
        double cs_o = v3_dot(eo_i, s_new);
        double sn_o = v3_dot(eo_i, p_new);

        FresnelC Fo = fresnel_eval(cos_i, n1c, kc(n_o, 0.0));
        FresnelC Fe = fresnel_eval(cos_i, n1c, kc(bi.n_phase_e, 0.0));
        double Ts_o = Fo.Ts > 0.0 ? Fo.Ts : 0.0;
        double Tp_o = Fo.Tp > 0.0 ? Fo.Tp : 0.0;
        double Ts_e = Fe.Ts > 0.0 ? Fe.Ts : 0.0;
        double Tp_e = Fe.Tp > 0.0 ? Fe.Tp : 0.0;

        /* reflected child: coherent sum of both channels' reflections
         * (tracer.py:1030-1054) */
        Ray refl = *r;
        refl.dir = fresnel_reflect_dir(r->dir, n_hat);
        refl.s_hat = s_new;
        kcplx ars_o = kc_scale(kc_cis(kc_arg(Fo.rs)),
                               sqrt(r_m + phys * kc_abs2(Fo.rs)));
        kcplx arp_o = kc_scale(kc_cis(kc_arg(Fo.rp)),
                               sqrt(r_m + phys * kc_abs2(Fo.rp)));
        kcplx ars_e = kc_scale(kc_cis(kc_arg(Fe.rs)),
                               sqrt(r_m + phys * kc_abs2(Fe.rs)));
        kcplx arp_e = kc_scale(kc_cis(kc_arg(Fe.rp)),
                               sqrt(r_m + phys * kc_abs2(Fe.rp)));
        refl.Es = kc_sub(kc_mul(kc_scale(Eo_i, cs_o), ars_o),
                         kc_mul(kc_scale(Ee_i, sn_o), ars_e));
        refl.Ep = kc_add(kc_mul(kc_scale(Eo_i, sn_o), arp_o),
                         kc_mul(kc_scale(Ee_i, cs_o), arp_e));
        record_reflection(s, &refl, r->generation,
                          (int32_t)(face - s->faces));
        refl.generation = (int16_t)(r->generation + 1);
        refl.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                     CHILD_SLOT_REFLECT);
        refl.event_ctr = 0;
        double p_refl = ray_power(&refl);
        if (!can_reflect) {
            ledger_credit(&cx->ledger, BK_TRUNCATED_GENERATION,
                          r->source_id, p_refl);
        } else if (p_refl > 0.0) {
            flux_out_child(&cx->ledger, body, &refl);
            push_child(s, cx, &refl);
        }
        p_accounted += p_refl;

        /* ordinary transmitted child (tracer.py:1056-1079) */
        if (!bi.tir_o) {
            kvec3 eo_o, ee_o;
            biref_eigenbasis(bi.k_o, c, &eo_o, &ee_o);
            kcplx ats_o = kc_scale(kc_cis(kc_arg(Fo.ts)),
                                   sqrt(phys * Ts_o));
            kcplx atp_o = kc_scale(kc_cis(kc_arg(Fo.tp)),
                                   sqrt(phys * Tp_o));
            kcplx Eso, Epo;
            fresnel_rotate_jones(kc_mul(kc_scale(Eo_i, cs_o), ats_o),
                                 kc_mul(kc_scale(Eo_i, sn_o), atp_o),
                                 s_new, v3_cross(bi.k_o, s_new),
                                 eo_o, ee_o, &Eso, &Epo);
            Ray och = *r;
            och.dir = bi.k_o;
            och.s_hat = eo_o;
            och.Es = Eso;               /* o mode: D along e_o only */
            och.Ep = kc(0.0, 0.0);
            och.pol_mode = 0;
            och.n_eff = 0.0;            /* medium table gives n_o */
            push_medium(&och, (int16_t)body->index, s, face->id);
            och.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                        CHILD_SLOT_ORDINARY);
            och.event_ctr = 0;
            double p_o = ray_power(&och);
            if (p_o > 0.0) {
                flux_out_child(&cx->ledger, body, &och);
                push_child(s, cx, &och);
            }
            p_accounted += p_o;
        }

        /* extraordinary transmitted child (tracer.py:1081-1105) */
        if (!bi.tir_e) {
            kvec3 eo_e, ee_e;
            biref_eigenbasis(bi.k_e, c, &eo_e, &ee_e);
            kcplx ats_e = kc_scale(kc_cis(kc_arg(Fe.ts)),
                                   sqrt(phys * Ts_e));
            kcplx atp_e = kc_scale(kc_cis(kc_arg(Fe.tp)),
                                   sqrt(phys * Tp_e));
            kcplx Ese, Epe;
            fresnel_rotate_jones(
                kc_mul(kc_scale(Ee_i, -sn_o), ats_e),
                kc_mul(kc_scale(Ee_i, cs_o), atp_e),
                s_new, v3_cross(bi.k_e, s_new), eo_e, ee_e, &Ese, &Epe);
            Ray ech = *r;
            ech.dir = bi.s_e;           /* RAY along the Poynting dir */
            ech.s_hat = eo_e;
            ech.Es = kc(0.0, 0.0);
            ech.Ep = Epe;               /* e mode: D along e_e only */
            ech.pol_mode = 1;
            ech.n_eff = bi.n_ray_e;     /* OPL per metre ALONG the ray */
            push_medium(&ech, (int16_t)body->index, s, face->id);
            ech.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                        CHILD_SLOT_EXTRAORD);
            ech.event_ctr = 0;
            double p_e = ray_power(&ech);
            if (p_e > 0.0) {
                flux_out_child(&cx->ledger, body, &ech);
                push_child(s, cx, &ech);
            }
            p_accounted += p_e;
        }
    } else {
        /* ------------- EXIT: crystal -> outside ------------- */
        int far = (r->depth >= 2) ? r->medium[r->depth - 2] : AMBIENT;
        kcplx n2c = scene_medium_n(s, far, r->lam_idx);
        int is_e = r->pol_mode == 1;
        kvec3 k_int = is_e ? biref_k_from_ray(r->dir, c, n_o, n_e)
                           : r->dir;
        double n_phase = is_e
            ? biref_n_e_theta(v3_dot(k_int, c), n_o, n_e) : n_o;
        double cos_k = -v3_dot(k_int, n_hat);
        if (cos_k < 0.0) cos_k = 0.0;
        if (cos_k > 1.0) cos_k = 1.0;
        int tir;
        kvec3 d_out = biref_refract_out(k_int, is_e, n_hat, c, n_o, n_e,
                                        n2c.re, &tir);
        FresnelC F = fresnel_eval(cos_k, kc(n_phase, 0.0), n2c);
        double Ts = F.Ts > 0.0 ? F.Ts : 0.0;
        double Tp = F.Tp > 0.0 ? F.Tp : 0.0;
        kvec3 s_new, p_new;
        fresnel_pol_basis(k_int, n_hat, &s_new, &p_new);
        kvec3 p_old = v3_cross(r->dir, r->s_hat);
        kcplx Es_i, Ep_i;
        fresnel_rotate_jones(r->Es, r->Ep, r->s_hat, p_old, s_new, p_new,
                             &Es_i, &Ep_i);

        /* transmitted child: leaves the crystal, mode resets
         * (tracer.py:1151-1165) */
        if (!tir) {
            Ray tr = *r;
            tr.dir = d_out;
            tr.s_hat = s_new;
            tr.Es = kc_mul(Es_i, kc_scale(kc_cis(kc_arg(F.ts)),
                                          sqrt(phys * Ts)));
            tr.Ep = kc_mul(Ep_i, kc_scale(kc_cis(kc_arg(F.tp)),
                                          sqrt(phys * Tp)));
            tr.pol_mode = 0;
            tr.n_eff = 0.0;
            pop_medium(&tr, (int16_t)body->index, s, face->id);
            tr.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                       CHILD_SLOT_TRANSMIT);
            tr.event_ctr = 0;
            double p_t = ray_power(&tr);
            if (p_t > 0.0) {
                flux_out_child(&cx->ledger, body, &tr);
                push_child(s, cx, &tr);
            }
            p_accounted += p_t;
        }

        /* internal reflection: mode-preserving (tracer.py:1167-1193) */
        kvec3 k_r = fresnel_reflect_dir(k_int, n_hat);
        Ray rf = *r;
        rf.dir = k_r;
        rf.s_hat = s_new;
        rf.Es = kc_mul(Es_i, kc_scale(kc_cis(kc_arg(F.rs)),
                                      sqrt(r_m + phys * kc_abs2(F.rs))));
        rf.Ep = kc_mul(Ep_i, kc_scale(kc_cis(kc_arg(F.rp)),
                                      sqrt(r_m + phys * kc_abs2(F.rp))));
        record_reflection(s, &rf, r->generation,
                          (int32_t)(face - s->faces));
        rf.generation = (int16_t)(r->generation + 1);
        rf.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                   CHILD_SLOT_REFLECT);
        rf.event_ctr = 0;
        if (is_e) {
            kvec3 s_ray;
            double np_, n_ray;
            biref_ray_from_k(k_r, c, n_o, n_e, &s_ray, &np_, &n_ray);
            rf.dir = s_ray;
            rf.n_eff = n_ray;
        }
        double p_rf = ray_power(&rf);
        if (!can_reflect) {
            ledger_credit(&cx->ledger, BK_TRUNCATED_GENERATION,
                          r->source_id, p_rf);
        } else if (p_rf > 0.0) {
            flux_out_child(&cx->ledger, body, &rf);
            push_child(s, cx, &rf);
        }
        p_accounted += p_rf;
    }

    double absorbed = p_in - p_accounted;
    if (absorbed > 0.0) {
        ledger_credit(&cx->ledger, BK_ABSORBED_SURFACE, r->source_id,
                      absorbed);
        cx->ledger.surf_by_body[body->index] += absorbed;
    }
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

    /* ---- uniaxial birefringence: dedicated o/e split path
     * (tracer.py:491-494; biaxial is Python-routed) ---- */
    if (body->birefringent) {
        biref_children(s, face, body, r, entering, n_hat, cos_i, cx);
        return;
    }
    /* mode-tagged crystal ray at a non-birefringent boundary (nested
     * body inside a crystal): continue as ordinary (tracer.py:498-510) */
    Ray fixed;
    if (r->pol_mode != 0) {
        static int warned_nested_mode = 0;
        if (!warned_nested_mode) {
            warned_nested_mode = 1;     /* benign race: warn-once flag */
            LOGW("mode-tagged crystal ray hit non-birefringent face %s "
                 "(nested body inside a crystal?) — continuing as "
                 "ordinary index (documented approximation)", face->id);
        }
        fixed = *r;
        fixed.pol_mode = 0;
        fixed.n_eff = 0.0;
        r = &fixed;
    }

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
     * coating table (tracer.py:538-584), via the shared helper (also
     * used per microfacet lobe below) */
    IfcCoef mc = interface_coeffs(s, face, r, cos_i, n1, n2, NULL);
    kcplx rs = mc.rs, rp = mc.rp, ts = mc.ts, tp = mc.tp;
    double Rs = mc.Rs, Rp = mc.Rp, Ts = mc.Ts, Tp = mc.Tp;
    (void)Rs; (void)Rp;

    /* ---- roughness: specular attenuation factor (tracer.py:620-628) */
    const RoughC *rough = face->rough >= 0 ? &s->roughs[face->rough]
                                           : NULL;
    double A_spec = rough
        ? rough_specular_factor(rough->sigma_m, cos_i, r->lam) : 1.0;
    double sqrtA = sqrt(A_spec);

    /* ---- ABg scatter: reflected-side specular/scatter split
     * (tracer.py:630-645) ---- */
    const ScatC *scat = face->scat >= 0 ? &s->scats[face->scat] : NULL;
    double tis = 0.0;
    if (scat) {
        tis = abg_tis_g2(scat->A, scat->B, cos_i);
        if (scat->tis_cap >= 0.0 && tis > scat->tis_cap)
            tis = scat->tis_cap;
        if (tis < 0.0) tis = 0.0;
        if (tis > 1.0) tis = 1.0;
    }
    double refl_scale = sqrt(1.0 - tis);

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
     * sqrt(r_m + phys |r|^2), phase from the physical coefficient; the
     * specular child additionally carries sqrtA (roughness) and
     * refl_scale (ABg specular remainder) ---- */
    kcplx full_amp_rs = kc_scale(kc_cis(kc_arg(rs)),
                                 sqrt(r_m + phys * kc_abs2(rs)));
    kcplx full_amp_rp = kc_scale(kc_cis(kc_arg(rp)),
                                 sqrt(r_m + phys * kc_abs2(rp)));
    kcplx amp_rs = kc_scale(full_amp_rs, sqrtA * refl_scale);
    kcplx amp_rp = kc_scale(full_amp_rp, sqrtA * refl_scale);
    Ray refl = *r;
    refl.dir = fresnel_reflect_dir(r->dir, n_hat);
    refl.s_hat = s_new;
    refl.Es = kc_mul(Es, amp_rs);
    refl.Ep = kc_mul(Ep, amp_rp);
    record_reflection(s, &refl, r->generation,
                      (int32_t)(face - s->faces));
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
    Ray trans;                  /* kept in scope: scattered-transmission
                                 * lobes inherit its medium stack */
    memset(&trans, 0, sizeof trans);
    if (!tir) {
        trans = *r;
        trans.dir = fresnel_refract_dir(r->dir, n_hat, cos_i, n1.re, n2.re);
        trans.s_hat = s_new;
        kcplx amp_ts = kc_scale(kc_cis(kc_arg(ts)),
                                sqrt(phys * Ts) * sqrtA);
        kcplx amp_tp = kc_scale(kc_cis(kc_arg(tp)),
                                sqrt(phys * Tp) * sqrtA);
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

    double p_accounted = p_refl + (tir ? 0.0 : p_trans_pre);

    /* ---- roughness: Beckmann-scattered lobes carry the (1-A) power
     * (tracer.py:732-852, rough_fresnel='micro': coefficients at each
     * MICROFACET-LOCAL angle, per polarization in the microfacet basis;
     * the legacy 'macro' model is feature-routed to Python) ---- */
    if (rough && A_spec < 1.0 - 1e-12) {
        const int k_lobe = 2;
        kvec3 p_of_snew = v3_cross(r->dir, s_new);
        for (int j = 0; j < k_lobe; j++) {
            kvec3 n_j = beckmann_facet(n_hat, rough->slope, r->ray_key,
                                       r->event_ctr, (uint32_t)(64 * j));
            double cos_j = -v3_dot(r->dir, n_j);
            if (cos_j < 0.0) cos_j = 0.0;
            if (cos_j > 1.0) cos_j = 1.0;
            IfcCoef lc = interface_coeffs(
                s, face, r, cos_j, n1, n2,
                (face->coating >= 0
                 && s->coatings[face->coating].kind == COAT_TABLE)
                    ? &mc : NULL);
            /* Jones into the microfacet's own s/p basis */
            kvec3 s_j, p_j;
            fresnel_pol_basis(r->dir, n_j, &s_j, &p_j);
            kcplx Es_j, Ep_j;
            fresnel_rotate_jones(Es, Ep, s_new, p_of_snew, s_j, p_j,
                                 &Es_j, &Ep_j);
            double frac = (1.0 - A_spec) / (double)k_lobe;
            /* scattered reflection */
            Ray sc = *r;
            sc.dir = fresnel_reflect_dir(r->dir, n_j);
            sc.s_hat = s_j;
            sc.Es = kc_mul(Es_j, kc_scale(
                kc_cis(kc_arg(lc.rs)),
                sqrt(frac * (r_m + phys * kc_abs2(lc.rs)))));
            sc.Ep = kc_mul(Ep_j, kc_scale(
                kc_cis(kc_arg(lc.rp)),
                sqrt(frac * (r_m + phys * kc_abs2(lc.rp)))));
            record_reflection(s, &sc, r->generation,
                              (int32_t)(face - s->faces));
            sc.generation = (int16_t)(r->generation + 1);
            sc.scattered = 1;
            sc.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                       CHILD_SLOT_ROUGH_R0 + (uint32_t)j);
            sc.event_ctr = 0;
            double p_sc = ray_power(&sc);
            int ok = v3_dot(sc.dir, n_hat) > 0.0
                     && sc.generation <= s->max_reflections && p_sc > 0.0;
            if (!ok && p_sc > 0.0) {
                ledger_credit(&cx->ledger, BK_ABSORBED_SURFACE,
                              r->source_id, p_sc);
                cx->ledger.surf_by_body[body->index] += p_sc;
            }
            p_accounted += p_sc;
            if (ok) {
                flux_out_child(&cx->ledger, body, &sc);
                push_child(s, cx, &sc);
            }
            /* scattered transmission (suppressed under macro TIR — the
             * medium stack was never pushed; tracer.py:832-836) */
            int tir_j = ((lc.Ts + lc.Tp) <= 1e-15) || tir;
            if (!tir) {
                Ray st = trans;     /* inherits the post-push/pop stack */
                st.dir = fresnel_refract_dir(r->dir, n_j, cos_j, n1.re,
                                             n2.re);
                st.s_hat = s_j;
                st.Es = kc_mul(Es_j, kc_scale(kc_cis(kc_arg(lc.ts)),
                                              sqrt(frac * phys * lc.Ts)));
                st.Ep = kc_mul(Ep_j, kc_scale(kc_cis(kc_arg(lc.tp)),
                                              sqrt(frac * phys * lc.Tp)));
                st.scattered = 1;
                st.generation = r->generation;
                st.ray_key = rng_child_key(
                    r->ray_key, r->event_ctr,
                    CHILD_SLOT_ROUGH_T0 + (uint32_t)j);
                st.event_ctr = 0;
                double p_st = ray_power(&st);
                int okt = !tir_j && p_st > 0.0
                          && v3_dot(st.dir, n_hat) < 0.0;
                if (!okt && p_st > 0.0 && !tir_j) {
                    ledger_credit(&cx->ledger, BK_ABSORBED_SURFACE,
                                  r->source_id, p_st);
                    cx->ledger.surf_by_body[body->index] += p_st;
                }
                if (!tir_j) p_accounted += p_st;
                if (okt) {
                    flux_out_child(&cx->ledger, body, &st);
                    push_child(s, cx, &st);
                }
            }
        }
    }

    /* ---- ABg scattered lobes: TIS share of the reflected power around
     * the specular direction (tracer.py:854-886) ---- */
    if (scat && tis > 0.0) {
        const int k_lobe = 2;
        kvec3 d_spec = fresnel_reflect_dir(r->dir, n_hat);
        double amp_lobe = sqrt(tis / (double)k_lobe);
        for (int j = 0; j < k_lobe; j++) {
            Ray sc = *r;
            sc.dir = abg_sample_g2(scat->A, scat->B, d_spec, n_hat,
                                   r->ray_key, r->event_ctr,
                                   (uint32_t)(256 + 2 * j));
            sc.s_hat = s_new;
            sc.Es = kc_scale(kc_mul(Es, full_amp_rs), amp_lobe);
            sc.Ep = kc_scale(kc_mul(Ep, full_amp_rp), amp_lobe);
            record_reflection(s, &sc, r->generation,
                              (int32_t)(face - s->faces));
            sc.generation = (int16_t)(r->generation + 1);
            sc.scattered = 1;
            sc.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                       CHILD_SLOT_ABG_0 + (uint32_t)j);
            sc.event_ctr = 0;
            double p_sc = ray_power(&sc);
            int ok = v3_dot(sc.dir, n_hat) > 0.0
                     && sc.generation <= s->max_reflections && p_sc > 0.0;
            if (!ok && p_sc > 0.0) {
                ledger_credit(&cx->ledger, BK_ABSORBED_SURFACE,
                              r->source_id, p_sc);
                cx->ledger.surf_by_body[body->index] += p_sc;
            }
            p_accounted += p_sc;
            if (ok) {
                flux_out_child(&cx->ledger, body, &sc);
                push_child(s, cx, &sc);
            }
        }
    }

    /* ---- surface absorption = exact power difference (tracer.py:888-894)
     * pre-polarizer (that loss has its own bucket); generation-capped
     * reflection already credited; TIR kills the transmitted child ---- */
    double absorbed = p_in - p_accounted;
    if (absorbed > 0.0) {
        ledger_credit(&cx->ledger, BK_ABSORBED_SURFACE, r->source_id,
                      absorbed);
        cx->ledger.surf_by_body[body->index] += absorbed;
    }
}

/* -------------------------------------------------------- gratings */
/* Port of grating.apply_to_batch (grating.py:379-483): one child per
 * propagating order; reflective when body.mirror >= 0.5. Order
 * efficiencies: lambda-only models arrive pre-resolved per
 * (order, lam_idx); Kogelnik is evaluated per ray. Children carry no
 * grating phase beyond OPL (documented Python limitation, same here). */
static void grating_children(const SceneC *s, const FaceC *face,
                             const BodyC *body, const Ray *r,
                             ThreadCtx *cx) {
    const GratC *g = &s->gratings[face->grating];
    kvec3 n_out = v3_scale(face_normal_canonical(face, r->pos),
                           face->outward_sign);
    double dt = v3_dot(n_out, r->dir);
    double sgn = (dt < 0.0) ? 1.0 : -1.0;
    kvec3 n_hat = v3_scale(n_out, sgn);
    int entering = sgn > 0.0;
    double cos_i = -v3_dot(r->dir, n_hat);

    double n1 = scene_medium_n(s, ray_current_medium(r), r->lam_idx).re;
    double n2g = g->n2[r->lam_idx];
    double n1s = entering ? n1 : n2g;
    double n2s = entering ? n2g : n1;

    /* groove periodicity vector projected into the local tangent plane
     * (grating.groove_vector) */
    kvec3 gh = v3_sub(g->groove_base,
                      v3_scale(n_hat, v3_dot(g->groove_base, n_hat)));
    double gn = v3_norm(gh);
    if (gn < 1e-8)
        die(EXIT_PHYSICS, "grating on face %s: groove vector nearly "
            "parallel to the local normal", face->id);
    gh = v3_scale(gh, 1.0 / gn);

    kvec3 s_new, p_new;
    fresnel_pol_basis(r->dir, n_hat, &s_new, &p_new);
    kvec3 p_old = v3_cross(r->dir, r->s_hat);
    kcplx Es, Ep;
    fresnel_rotate_jones(r->Es, r->Ep, r->s_hat, p_old, s_new, p_new,
                         &Es, &Ep);
    double Is = kc_abs2(Es);
    double Ip = kc_abs2(Ep);
    double p_in = Is + Ip;

    int reflective = body->mirror >= 0.5;
    double order_power = 0.0;

    double kog_es = 0.0, kog_ep = 0.0;
    if (g->model == GRATING_KOGELNIK)
        kogelnik_eta(g->thickness_m, g->dn, g->slant_rad, g->lines_per_mm,
                     r->lam, cos_i, &kog_es, &kog_ep);

    for (int m = g->lo; m <= g->hi; m++) {
        double eta_s, eta_p;
        if (g->model == GRATING_FIXED) {
            size_t at = (size_t)(m - g->lo) * s->n_lams + r->lam_idx;
            eta_s = g->eta_s[at];
            eta_p = g->eta_p[at];
        } else {
            /* Kogelnik: only orders 0 and +1 carry power */
            if (m == 0) { eta_s = 1.0 - kog_es; eta_p = 1.0 - kog_ep; }
            else if (m == 1) { eta_s = kog_es; eta_p = kog_ep; }
            else { eta_s = eta_p = 0.0; }
        }
        kvec3 dir_t, dir_r;
        int prop_t, prop_r;
        grating_order_dirs(r->dir, n_hat, gh, g->lines_per_mm, r->lam, m,
                           n1s, n2s, &dir_t, &prop_t, &dir_r, &prop_r);
        kvec3 d_new = reflective ? dir_r : dir_t;
        int prop = reflective ? prop_r : prop_t;
        double contrib = Is * eta_s + Ip * eta_p;
        if (prop) order_power += contrib;
        if (!prop || contrib <= 0.0) continue;

        Ray child = *r;
        child.dir = d_new;
        child.s_hat = s_new;
        child.Es = kc_scale(Es, sqrt(eta_s));
        child.Ep = kc_scale(Ep, sqrt(eta_p));
        child.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                      CHILD_SLOT_GRATING0
                                      + (uint32_t)(m + 8));
        child.event_ctr = 0;
        if (reflective) {
            record_reflection(s, &child, r->generation,
                              (int32_t)(face - s->faces));
            child.generation = (int16_t)(r->generation + 1);
            if (child.generation > s->max_reflections) {
                ledger_credit(&cx->ledger, BK_TRUNCATED_GENERATION,
                              r->source_id, ray_power(&child));
                continue;
            }
        } else {
            if (entering)
                push_medium(&child, (int16_t)body->index, s, face->id);
            else
                pop_medium(&child, (int16_t)body->index, s, face->id);
        }
        push_child(s, cx, &child);
    }
    /* efficiency losses + evanescent orders -> absorbed
     * (grating.py:474-482; Python books under label+":grating" — folded
     * into the body's surface-absorption diagnostic here) */
    double p_abs = p_in - order_power;
    if (p_abs > 0.0) {
        ledger_credit(&cx->ledger, BK_ABSORBED_SURFACE, r->source_id,
                      p_abs);
        cx->ledger.surf_by_body[body->index] += p_abs;
    }
}

/* -------------------------------------------------------- detector event */
/* Port of Tracer._detector_event (tracer.py:339-372), incoherent side only
 * (the coherent gather is phase D; request_load rejects coherent sources
 * until then). */
static void detector_event(const SceneC *s, const FaceC *face, const Ray *r,
                           kvec3 start_pos, double start_opl,
                           ThreadCtx *cx) {
    const DetC *d = &s->dets[face->detector];
    if (r->coherent) {
        /* Huygens wavelet sample at the segment START (the kernel adds
         * the final k*n*r leg itself — tracer.py:229-233 double-count
         * warning; add_gather_samples, detector.py:173-191) */
        GatherHit gh;
        gh.det = face->detector;
        gh.source_id = r->source_id;
        gh.lam_stratum = r->lam_stratum;
        gh.pol_stratum = r->pol_stratum;
        gh.pos = start_pos;
        gh.dir = r->dir;
        gh.s_hat = r->s_hat;
        gh.Es = r->Es;
        gh.Ep = r->Ep;
        gh.lam = r->lam;
        gh.opl = start_opl;
        gh.power = ray_power(r);
        gh.scattered = r->scattered;
        gh.ray_key = r->ray_key;
        gathhits_push(&cx->ghits, &gh);
        if (s->export_rays) {
            ExportRec er;
            export_fill(&er, face->detector, r);
            exportvec_push(&cx->exports, &er);
        }
        /* diagnostic tallies (mirror the incoherent path below) */
        cx->ledger.surf_by_det[face->detector] += gh.power;
        cx->ledger.detected[face->detector] += gh.power;
        return;
    }
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
    if (s->export_rays) {
        ExportRec er;
        export_fill(&er, face->detector, r);
        exportvec_push(&cx->exports, &er);
    }
    /* diagnostic tallies (NOT closure buckets — tracer.py:366-372) */
    cx->ledger.surf_by_det[face->detector] += h.power;
    cx->ledger.detected[face->detector] += h.power;
}

/* ------------------------------------------------------------ process_ray */
/* Port of one ray's share of Tracer.step (tracer.py:183-322). */
/* continuum particle-medium interception (particles.py:155-209): the
 * ballistic parent decays coherently by exp(-tau/2); one incoherent
 * scattered child per crossing carries P (1-e^-tau) albedo from a
 * truncated-exponential scatter point; the absorbed remainder books to
 * 'particle_absorbed'. Runs BEFORE the surface interaction of the step,
 * exactly like tracer.step's intercept hook. */
static void particle_intercept(const SceneC *s, Ray *r, int hit, double t,
                               ThreadCtx *cx) {
    const ParticleC *p = s->particles;
    double seg_max = hit ? t : 1.0;    /* escapers still traverse */
    /* slab overlap (particles._slab_overlap), scalar */
    double t0 = 0.0, t1 = seg_max;
    const double *lo = &p->box_lo.x, *hi = &p->box_hi.x;
    const double *o = &r->pos.x, *d = &r->dir.x;
    for (int ax = 0; ax < 3; ax++) {
        if (fabs(d[ax]) < 1e-300) {
            if (o[ax] < lo[ax] || o[ax] > hi[ax]) return;   /* miss */
            continue;
        }
        double a = (lo[ax] - o[ax]) / d[ax];
        double b = (hi[ax] - o[ax]) / d[ax];
        double mn = a < b ? a : b, mx = a < b ? b : a;
        if (mn > t0) t0 = mn;
        if (mx < t1) t1 = mx;
    }
    if (!(t1 > t0)) return;

    double mu = p->mu_ext[r->lam_idx];
    double alb = p->albedo[r->lam_idx];
    double tau = mu * (t1 - t0);
    double p_col = 1.0 - exp(-tau);
    double p_in = ray_power(r);
    double p_scat = p_in * p_col * alb;
    double p_abs = p_in * p_col * (1.0 - alb);
    if (p_abs > 0.0) {
        ledger_credit(&cx->ledger, BK_PARTICLE_ABSORBED, r->source_id,
                      p_abs);
        cx->ledger.by_particles += p_abs;
    }
    if (p_scat > 0.0) {
        /* draws live in a reserved index range of THIS event */
        uint32_t ev = r->event_ctr;
        double u = rng_uniform(r->ray_key, ev, 1024);
        double sdist = -log(1.0 - u * (1.0 - exp(-tau))) / mu;
        Ray child = *r;
        child.pos = v3_fma(r->pos, t0 + sdist, r->dir);
        /* radius node from the per-lam radius CDF */
        double ur = rng_uniform(r->ray_key, ev, 1025);
        const double *rcdf = p->radius_cdf
                             + (size_t)r->lam_idx * p->n_quad;
        int node = 0;
        while (node < p->n_quad - 1 && rcdf[node] < ur) node++;
        /* scatter cosine from the (lam, node) inverse phase CDF */
        double uu = rng_uniform(r->ray_key, ev, 1026);
        const double *inv = p->inv_phase
            + ((size_t)r->lam_idx * p->n_quad + node) * p->n_u;
        double x = uu * (p->n_u - 1);
        int i0 = (int)x;
        if (i0 > p->n_u - 2) i0 = p->n_u - 2;
        double mu_s = inv[i0] + (x - i0) * (inv[i0 + 1] - inv[i0]);
        double phi = K_TWO_PI * rng_uniform(r->ray_key, ev, 1027);
        /* frame around d_in (mie.sample_direction:216-225) */
        double axv = fabs(r->dir.x), ayv = fabs(r->dir.y),
               azv = fabs(r->dir.z);
        kvec3 a = v3(0.0, 0.0, 0.0);
        if (axv <= ayv && axv <= azv)      a.x = 1.0;
        else if (ayv <= azv)               a.y = 1.0;
        else                               a.z = 1.0;
        kvec3 t1v = v3_unit(v3_cross(r->dir, a));
        kvec3 t2v = v3_cross(r->dir, t1v);
        double st = sqrt(1.0 - mu_s * mu_s > 0.0
                         ? 1.0 - mu_s * mu_s : 0.0);
        child.dir = v3_scale(r->dir, mu_s);
        child.dir = v3_fma(child.dir, st * cos(phi), t1v);
        child.dir = v3_fma(child.dir, st * sin(phi), t2v);
        /* s_hat rebuilt like particles.py:194-196 (pol_basis against the
         * component-rolled direction) */
        kvec3 rolled = v3(child.dir.z, child.dir.x, child.dir.y);
        kvec3 s_new, p_new;
        fresnel_pol_basis(child.dir, rolled, &s_new, &p_new);
        child.s_hat = s_new;
        double amp = sqrt(p_scat / 2.0);
        kcplx ph = kc_cis(K_TWO_PI * rng_uniform(r->ray_key, ev, 1028));
        child.Es = kc_scale(ph, amp);
        child.Ep = kc_scale(ph, amp);
        child.coherent = 0;
        child.ray_key = rng_child_key(r->ray_key, ev,
                                      CHILD_SLOT_PARTICLE);
        child.event_ctr = 0;
        push_child(s, cx, &child);
    }
    /* ballistic parent: coherent Beer-Lambert amplitude decay */
    double att = exp(-tau / 2.0);
    r->Es = kc_scale(r->Es, att);
    r->Ep = kc_scale(r->Ep, att);
}

static void process_ray(const SceneC *s, Ray *r, ThreadCtx *cx) {
    cx->interactions++;
    int32_t fid;
    double t = scene_intersect(s, r->pos, r->dir, &fid);
    int hit = fid >= 0;

    /* ---- particle-medium interception (tracer.py:188-198 hook) ---- */
    if (s->particles)
        particle_intercept(s, r, hit, t, cx);

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
    kvec3 start_pos = r->pos;
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
        detector_event(s, face, r, start_pos, start_opl, cx);
        screen_children(s, face, body, r, cx);
    } else if (body->role == ROLE_DETECTOR) {
        /* non-screen face of a detector solid: strict no-op pass-through
         * (tracer.py:304-308) — the continuation is the same ray */
        Ray cont = *r;
        cont.ray_key = rng_child_key(r->ray_key, r->event_ctr,
                                     CHILD_SLOT_TRANSMIT);
        cont.event_ctr = 0;
        push_child(s, cx, &cont);
    } else if (face->grating >= 0) {
        grating_children(s, face, body, r, cx);   /* tracer.py:309-311 */
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
        r.birth_pos = pos;          /* --export-rays pupil coordinate */
        for (int hh = 0; hh < HIST_DEPTH; hh++) r.refl_hist[hh] = -1;
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
        gathhits_init(&ctxs[i].ghits);
        exportvec_init(&ctxs[i].exports);
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
            det_apply_gather_hits(s, &ctxs[i].ghits);
            gathhits_clear(&ctxs[i].ghits);
            det_collect_exports((SceneC *)s, &ctxs[i].exports);
            exportvec_clear(&ctxs[i].exports);
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
        gathhits_free(&ctxs[i].ghits);
        exportvec_free(&ctxs[i].exports);
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
