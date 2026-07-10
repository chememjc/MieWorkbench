/* ===========================================================================
 * scene.h — the C engine's scene: faces, bodies, pre-resolved wavelength
 * tables, sources, detector grids, and trace parameters.
 *
 * Built by request_load() (request.c) from <case>/cengine/request.json,
 * which scripts/raytracer/cengine.py writes. The Python side pre-resolves
 * EVERYTHING dispersive at the fixed stratum wavelengths (plan D1): this
 * struct carries plain per-(entity, lam_idx) tables and the C engine never
 * evaluates a material model.
 *
 * Mirrors scripts/raytracer/scene.py Scene/Body semantics — comments below
 * reference the Python lines each field must stay faithful to.
 * =========================================================================== */
#ifndef MIEWB_SCENE_H
#define MIEWB_SCENE_H

#include <stdint.h>
#include "kernels/kmath.h"
#include "kernels/surf.h"
#include "kernels/trim.h"
#include "raybuf.h"

#define MIEWB_MAX_NAME 96

/* body roles (scene.py Body.role) */
enum {
    ROLE_OPTIC = 0,
    ROLE_DETECTOR = 1,
    ROLE_SOURCE = 2,
};

/* polarizer types (optprops polarizer registry 'type' column) */
enum {
    POL_LINEAR = 0,
    POL_CIRCULAR_LEFT = 1,
    POL_CIRCULAR_RIGHT = 2,
};

typedef struct {
    char label[MIEWB_MAX_NAME];
    char name[MIEWB_MAX_NAME];
    uint8_t role;
    int32_t index;              /* position in SceneC.bodies == body index */
    double mirror;              /* idealized achromatic reflector fraction */
    double absorbance;
    /* per-lam tables, length SceneC.n_lams (Python computed these through
     * Scene.medium_index so detector-role bodies already read as ambient
     * — scene.py:454-477): */
    double *n_re, *n_im;
    double *filter_alpha;       /* additive bulk alpha [1/m]; NULL if none */
    /* dichroic polarizer (tracer._apply_polarizer): applied on ENTRY */
    uint8_t has_polarizer;
    uint8_t pol_type;
    double retardance_waves;
    kvec3 pol_axis;             /* unit transmission axis (global) */
    double *pol_T_par, *pol_T_perp;    /* per-lam, pre-resolved (D1) */
} BodyC;

/* coating kinds (scene.py face_coatings + tracer.py:541-578) */
enum {
    COAT_TMM = 0,
    COAT_TABLE = 1,
};

#define COAT_MAX_LAYERS 64

typedef struct {
    uint8_t kind;
    /* TMM: n_layers stacks, incident side first; indices pre-resolved at
     * every scene wavelength (flattened [layer][lam]) */
    int32_t n_layers;
    double *layer_n_re, *layer_n_im;   /* n_layers * n_lams */
    double *layer_d;                   /* n_layers */
    /* TABLE: measured power coefficients at each scene wavelength; trace
     * borrows the bare-Fresnel phase and folds T into R past TIR */
    double *Rs, *Rp, *Ts, *Tp;         /* n_lams each */
} CoatC;

typedef struct {
    char id[MIEWB_MAX_NAME];    /* "Body.Feature.FaceN" contract name */
    int32_t body;               /* owning body index */
    SurfC surf;
    TrimC trim;
    double outward_sign;        /* +1 if orientation_outward else -1 */
    double area_m2;
    int32_t detector;           /* index into SceneC.dets, or -1 */
    int32_t coating;            /* index into SceneC.coatings, or -1 */
} FaceC;

/* Planar detector grid — geometry computed by the Python glue with the
 * exact DetectorGrid.__init__ math (detector.py:88-164) and passed in, so
 * the two engines share pixel mapping bit-for-bit. The mask is recomputed
 * here from the trim (same algorithm -> same result; parity-tested). */
typedef struct {
    char label[MIEWB_MAX_NAME];
    int32_t face_id;
    kvec3 xhat, yhat, normal;
    double x_lo, y_lo, pixel_m;
    int32_t W, H, spectral_bins;
    double lam_lo, lam_hi;      /* spectral-bin range [m] */
    double *inc;                /* (bins, H, W) accumulated cube */
    uint8_t *mask;              /* (H, W) trim mask */
    /* detected-power tallies per (source, lam_stratum, pol_stratum), the
     * key shape detector.py uses; flat [src][stratum][pol] arrays sized by
     * the scene maxima. */
    double *det_inc_W;
    int64_t *det_inc_n;
} DetC;

/* Emission policy (sources.py:240-266 "toward-origin sign policy"):
 * collimated planar sources get an explicit direction from the Python
 * glue (it owns the mean-sample sign decision); curved sources choose
 * per-sample normal sign toward the origin, with flip_all covering the
 * frac_neg==1.0 whole-face-flipped case. */
enum {
    EMIT_COLLIMATED = 0,        /* planar: fixed dir for every ray */
    EMIT_CURVED = 1,            /* per-sample normal toward origin */
};

typedef struct {
    char label[MIEWB_MAX_NAME];
    int32_t body_index;         /* ledger row = position in sources array */
    double power_W;
    uint8_t coherent;
    uint8_t emit_policy;
    uint8_t flip_all;           /* EMIT_CURVED: whole face emits -normal */
    kvec3 emit_dir;             /* EMIT_COLLIMATED */
    int32_t lam_offset;         /* this source's strata start in lams_m */
    int32_t n_strata;
    int32_t n_pol;              /* 1 or 2 (sources.py n_pol_strata) */
    kcplx jones_s[2], jones_p[2];   /* per pol_stratum (sources.jones_for) */
    FaceC emit_face;            /* built like any face; not intersectable */
    /* UV sampling bounds (port of _sample_face_points' bbox logic,
     * computed at load time in request.c) */
    double u_lo, u_hi, v_lo, v_hi;
    int64_t viz_cap;            /* viz_rays cap for this source */
} SourceC;

typedef struct {
    /* trace parameters (TraceConfig, tracer.py:52-78) */
    int max_reflections;
    double power_floor;
    int64_t rays;               /* per source */
    uint64_t seed;
    int64_t batch_size;         /* children split bound (1<<20) */
    int threads;                /* 0 = OpenMP default */

    int n_lams;
    double *lams_m;             /* global wavelength union */
    double *amb_n_re, *amb_n_im;    /* ambient medium per lam */

    int n_bodies;
    BodyC *bodies;
    int n_faces;
    FaceC *faces;
    int n_sources;
    SourceC *sources;
    int n_dets;
    DetC *dets;
    int n_coatings;
    CoatC *coatings;

    int max_strata;             /* max n_strata over sources (tally dims) */
    int max_pol;                /* max n_pol over sources */

    char out_dir[1024];
} SceneC;

/* request.c */
SceneC *request_load(const char *path);
void scene_free(SceneC *s);

/* Complex index of the medium a ray is in: body index or AMBIENT, at the
 * ray's lam_idx. Port of scene.medium_index — the detector-as-ambient and
 * birefringence special cases were already folded into the tables by the
 * Python glue. */
static inline kcplx scene_medium_n(const SceneC *s, int body_index,
                                   int lam_idx) {
    if (body_index < 0)
        return kc(s->amb_n_re[lam_idx], s->amb_n_im[lam_idx]);
    const BodyC *b = &s->bodies[body_index];
    return kc(b->n_re[lam_idx], b->n_im[lam_idx]);
}

static inline double scene_filter_alpha(const SceneC *s, int body_index,
                                        int lam_idx) {
    if (body_index < 0) return 0.0;
    const double *fa = s->bodies[body_index].filter_alpha;
    return fa ? fa[lam_idx] : 0.0;
}

/* Nearest contained hit of one face — port of AnalyticFace.intersect
 * (surfaces.py:821-845): candidates filtered by t > t_eps (100 nm
 * self-intersection guard, see the comment there), sorted ascending,
 * first trim-contained root wins. Returns +INF on miss. */
static inline double face_intersect(const FaceC *f, kvec3 o, kvec3 d) {
    const double t_eps = 1e-7;
    double t[SURF_K_MAX];
    int K = surf_roots(&f->surf, o, d, t);
    for (int i = 0; i < K; i++)
        if (!(t[i] > t_eps)) t[i] = INFINITY;   /* also drops NaN */
    /* insertion sort, K <= 4 */
    for (int i = 1; i < K; i++) {
        double x = t[i];
        int j = i - 1;
        while (j >= 0 && t[j] > x) { t[j + 1] = t[j]; j--; }
        t[j + 1] = x;
    }
    for (int i = 0; i < K; i++) {
        if (!isfinite(t[i])) break;
        kvec3 p = v3_fma(o, t[i], d);
        double u, v;
        surf_to_uv(&f->surf, p, &u, &v);
        if (trim_contains(&f->trim, u, v)) return t[i];
    }
    return INFINITY;
}

/* Nearest hit across all faces — port of Scene.intersect (scene.py:492-508).
 * Linear scan in phase A; the phase-C TLAS replaces the loop body. */
static inline double scene_intersect(const SceneC *s, kvec3 o, kvec3 d,
                                     int32_t *fid_out) {
    double best_t = INFINITY;
    int32_t best_f = -1;
    for (int32_t fid = 0; fid < s->n_faces; fid++) {
        double t = face_intersect(&s->faces[fid], o, d);
        if (t < best_t) { best_t = t; best_f = fid; }
    }
    *fid_out = best_f;
    return best_t;
}

#endif /* MIEWB_SCENE_H */
