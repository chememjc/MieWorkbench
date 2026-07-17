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
#include "bvh.h"
#include "mesh.h"
#include "registry.h"

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

typedef struct BodyC {
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
    /* pulsed-optics P7 (time_products only; NULL otherwise): group index and
     * material GDD-per-length [s^2/m] per lam, PRE-RESOLVED in Python through
     * scene.medium_group_index / medium_gdd_per_length (the exact finite-
     * difference stencil). The C trace only multiplies by the segment length,
     * so gopl/gdd_acc match the Python accumulators bit-for-bit. */
    double *n_g, *gdd_per_m;    /* [n_lams] each */
    /* uniaxial birefringence (scene.py:137-144; biaxial is Python-routed) */
    uint8_t birefringent;
    kvec3 crystal_axis;         /* unit optic axis */
    double *bir_n_o, *bir_n_e;  /* per-lam REAL indices (D1) */
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

/* per-face roughness entry (scene.roughness; diffusers already resolved
 * to sigma/slope by scene.py + the glue) */
typedef struct {
    double sigma_m;             /* RMS height (Davies specular factor) */
    double slope;               /* RMS microfacet slope (Beckmann) */
} RoughC;

/* ABg measured-scatter entry (scene.scatter; g == 2 enforced by routing) */
typedef struct {
    double A, B;
    double tis_cap;             /* < 0 = uncapped */
} ScatC;

/* grating models: lambda-only efficiencies pre-resolved to tables by the
 * glue (lamellar/dammann/table -> FIXED); Kogelnik is angle-dependent and
 * evaluated per ray (kernels/gratingk.h) */
enum {
    GRATING_FIXED = 0,
    GRATING_KOGELNIK = 1,
};

typedef struct {
    uint8_t model;
    int32_t lo, hi;             /* inclusive order range */
    double lines_per_mm;
    kvec3 groove_base;          /* projected into the local tangent plane */
    double *eta_s, *eta_p;      /* FIXED: [n_orders][n_lams] */
    double thickness_m, dn, slant_rad;   /* KOGELNIK params */
    double *n2;                 /* per-lam far-side index (glue-resolved
                                 * exactly like grating.apply_to_batch) */
} GratC;

typedef struct FaceC {
    char id[MIEWB_MAX_NAME];    /* "Body.Feature.FaceN" contract name */
    int32_t body;               /* owning body index */
    SurfC surf;
    TrimC trim;
    MeshC *mesh;                /* SURF_MESH faces only, else NULL */
    double outward_sign;        /* +1 if orientation_outward else -1 */
    double area_m2;
    int32_t detector;           /* index into SceneC.dets, or -1 */
    int32_t coating;            /* index into SceneC.coatings, or -1 */
    int32_t rough;              /* index into SceneC.roughs, or -1 */
    int32_t scat;               /* index into SceneC.scats, or -1 */
    int32_t grating;            /* index into SceneC.gratings, or -1 */
    /* conservative world AABB (Python glue: STL-union-trim padded, or
     * analytic full-primitive bounds; +-INF when unknown = never culled) */
    kvec3 aabb_lo, aabb_hi;
    /* ordered surface-interaction handler list, resolved at scene build by
     * registry_resolve_faces() (REGISTRY.md §2.1). Order encodes the
     * historical process_ray precedence. Pointers into the static
     * INTERACTIONS table (program-lifetime). */
    const InteractionDef *handlers[MIEWB_MAX_FACE_HANDLERS];
    int n_handlers;
} FaceC;

/* One (source, lam_stratum, pol_stratum) coherent sample set on a
 * detector — the C analogue of DetectorGrid.samples[key]
 * (detector.py:173-201). SoA: the gather kernels stream these. */
typedef struct {
    int16_t source_id, lam_stratum, pol_stratum;
    int64_t n, cap;
    double *pos, *dir, *s_hat;      /* n * 3 */
    kcplx *Es, *Ep;                 /* complex128 Jones */
    double *lam, *opl, *power;
    double *dA;                     /* per-sample wavefront patch area [m^2]
                                     * from --ray-differentials; NaN where the
                                     * differential was lost -> gather falls
                                     * back to the source-referenced area
                                     * (gather.py:488-499). NULL when the run
                                     * has no differentials. */
    uint8_t *scattered;
    uint64_t *ray_key;              /* cross-estimator grouping (D2) */
    uint32_t *event_ctr;            /* P1: (ray_key,event_ctr) is the stable
                                     * canonical sort key the Python driver
                                     * uses to make a merged multi-chunk
                                     * sample set bit-identical to one chunk */
} GKey;

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
    /* coherent side: per-key Huygens sample sets + geometric tallies
     * (detected_geometric, detector.py:190-191) */
    GKey *gkeys;
    int32_t n_gkeys, cap_gkeys;
    double *det_geom_W;             /* same flat key layout as det_inc_W */
    /* --export-rays landing records (opaque here; detector.c owns) */
    void *exports;
    int64_t n_exports, cap_exports;
    /* pulsed-optics P7 time-product arrival records (opaque TimeRec here;
     * detector.c owns) */
    void *times;
    int64_t n_times, cap_times;
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
    /* per-(stratum, pol) gather normalization areas [m^2]
     * (run_trace.compute_sample_area) — [n_strata * n_pol] */
    double *sample_area;
    /* pulsed-optics P7 SPM chirp (time_products only): per-stratum birth-time
     * offset [s] (sources._stratum_t0). A primary is born with gopl =
     * c * stratum_t0[stratum] (sources.apply_stratum_t0). NULL => all zero. */
    double *stratum_t0;         /* [n_strata] seconds, or NULL */
} SourceC;

/* continuum-mode particle cloud (particles.py _continuum): all tables
 * pre-resolved per stratum wavelength by the glue; the phase-function
 * draw uses a per-(lam, radius-node) INVERSE CDF (mu at uniform u).
 * Explicit-realization mode is Python-routed. */
typedef struct {
    kvec3 box_lo, box_hi;
    int n_quad;                 /* radius quadrature nodes */
    int n_u;                    /* inverse-CDF resolution */
    double *mu_ext;             /* [n_lams] */
    double *albedo;             /* [n_lams] */
    double *radius_cdf;         /* [n_lams][n_quad] cumulative weights */
    double *inv_phase;          /* [n_lams][n_quad][n_u] mu(u) */
} ParticleC;

typedef struct SceneC {
    /* trace parameters (TraceConfig, tracer.py:52-78) */
    int max_reflections;
    double power_floor;
    int64_t rays;               /* per source (the NORMALIZATION denominator:
                                 * p_ray = power_W / rays, regardless of the
                                 * [lo,hi) chunk actually traced) */
    /* P1 chunked-run contract: this invocation traces primaries [lo,hi) of
     * every source (default 0..rays). Keys stay index-pure (i % n_strata,
     * (i/n_strata) % n_pol) so a chunk is a pure slice; the Python driver
     * aligns lo/hi to n_strata*n_pol and the C engine ASSERTS it. */
    int64_t primary_lo, primary_hi;
    uint8_t gather_skip;        /* 1 = trace only, dump gkey samples to disk
                                 * (accumulated across chunks by the Python
                                 * driver), no in-binary gather_run */
    uint8_t gather_only;        /* 1 = skip tracing entirely: load the merged
                                 * sample dump + accumulator snapshots from
                                 * gather_input and run the normal in-binary
                                 * gather_run (tiled kernel; gather.mode=
                                 * exact still selects the plain one) */
    char gather_input[1024];    /* gather_only: merged-dump directory */
    uint64_t seed;
    int64_t batch_size;         /* children split bound (1<<20) */
    int threads;                /* 0 = OpenMP default */

    int n_lams;
    double *lams_m;             /* global wavelength union */
    double *amb_n_re, *amb_n_im;    /* ambient medium per lam */
    /* pulsed-optics P7 (time_products only): ambient group index / GDD-per-
     * length per lam (scene.medium_group_index/medium_gdd_per_length at
     * body -1). NULL unless time_products. */
    double *amb_n_g, *amb_gdd_per_m;

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
    int n_roughs;
    RoughC *roughs;
    int n_scats;
    ScatC *scats;
    int n_gratings;
    GratC *gratings;
    ParticleC *particles;       /* NULL when no --particles */

    /* scene-level TLAS over face AABBs — the algorithmic win the Python
     * engine lacks (its Scene.intersect is a brute-force all-faces loop,
     * scene.py:492-508). linear_scan=1 (request params.linear_scan)
     * disables it for A/B validation. */
    BvhC tlas;
    uint8_t linear_scan;
    uint8_t mesh_flat_normals;  /* --mesh-flat-normals passthrough */

    /* coherent-gather parameters (render_coherent, gather.py:443-458) */
    uint8_t gather_backend;     /* 0 auto, 1 cuda, 2 cpu */
    double min_eff_samples;     /* M_eff gate (default 1000) */
    uint8_t enforce_gate;
    uint8_t save_fields;
    uint8_t occlusion;          /* --gather-occlusion */
    int occ_tile;               /* shadow tile size (default 16) */
    uint8_t gather_exact;       /* --gather-exact: plain fp64 kernel (the
                                 * bit-exact reference path); default is
                                 * the tile-factorized kernel */
    uint8_t gather_nufft;       /* NUFFT angular-spectrum route enabled
                                 * (request gather.nufft, default 0 = OFF /
                                 * opt-in); the per-key runtime gate is the
                                 * real switch, and --gather-exact/occlusion
                                 * disable it. OFF by default because the
                                 * band-limited route cannot reproduce the
                                 * exact per-pair kernel on Monte-Carlo point
                                 * samples (white spatial spectrum) — see
                                 * gather_nufft.c / cengine/README.md */
    uint8_t export_rays;        /* --export-rays (this seed) */
    uint8_t track_history;      /* --ghost-analysis (this seed) */
    uint8_t track_time;         /* pulsed-optics time tracking (P7): the
                                 * per-body power-weighted bulk-path tally
                                 * (GDD budget). Set when time_products OR
                                 * --gdd-budget is active (params.track_time). */
    uint8_t time_products;      /* pulsed-optics time products (P7): also
                                 * accumulate the gopl/gdd group-delay ray
                                 * slots and record per-detector arrival
                                 * records. Requires the n_g/gdd tables above.
                                 * (params.time_products) */
    uint8_t importance_aim;     /* --importance-aim (opt-in) */
    uint8_t ray_differentials;  /* --ray-differentials (P7 differentials port):
                                 * seed + transport the Igehy ray differentials
                                 * and size the coherent gather's per-sample dA
                                 * from |dPdx x dPdy| (params.ray_differentials) */

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

/* Group index of the medium a ray is in (time_products only). Port of
 * scene.medium_group_index — pre-resolved Python-side, so this is a table
 * read, body index or AMBIENT, at the ray's lam_idx. */
static inline double scene_medium_group_index(const SceneC *s, int body_index,
                                              int lam_idx) {
    if (body_index < 0) return s->amb_n_g[lam_idx];
    return s->bodies[body_index].n_g[lam_idx];
}

/* Material GDD-per-length [s^2/m] of the medium a ray is in (time_products
 * only). Port of scene.medium_gdd_per_length; pre-resolved Python-side. */
static inline double scene_medium_gdd_per_m(const SceneC *s, int body_index,
                                            int lam_idx) {
    if (body_index < 0) return s->amb_gdd_per_m[lam_idx];
    return s->bodies[body_index].gdd_per_m[lam_idx];
}

/* Canonical geometric normal of a face at a surface point (analytic
 * primitive normal, or the mesh's relocated winding-aligned normal).
 * Multiply by outward_sign for normal_out_of_solid. */
static inline kvec3 face_normal_canonical(const FaceC *f, kvec3 p) {
    if (f->surf.kind == SURF_MESH)
        return mesh_normal(f->mesh, p);
    return surf_normal(&f->surf, p);
}

/* Nearest contained hit of one face — port of AnalyticFace.intersect
 * (surfaces.py:821-845): candidates filtered by t > t_eps (100 nm
 * self-intersection guard, see the comment there), sorted ascending,
 * first trim-contained root wins. Mesh faces go straight to the triangle
 * BLAS (MeshFace.intersect). Returns +INF on miss. */
static inline double face_intersect(const FaceC *f, kvec3 o, kvec3 d) {
    const double t_eps = 1e-7;
    if (f->surf.kind == SURF_MESH)
        return mesh_intersect(f->mesh, o, d);
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

/* Nearest hit across all faces — the semantics of Scene.intersect
 * (scene.py:492-508), accelerated by the face-level TLAS. The linear
 * fallback stays as the A/B validation path (request params.linear_scan;
 * parity test asserts identical results). */
static inline double scene_intersect(const SceneC *s, kvec3 o, kvec3 d,
                                     int32_t *fid_out) {
    double best_t = INFINITY;
    int32_t best_f = -1;
    if (s->linear_scan || !s->tlas.nodes) {
        for (int32_t fid = 0; fid < s->n_faces; fid++) {
            double t = face_intersect(&s->faces[fid], o, d);
            if (t < best_t) { best_t = t; best_f = fid; }
        }
        *fid_out = best_f;
        return best_t;
    }
    kvec3 inv = v3(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
    int32_t stack[BVH_STACK_MAX];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const BvhNode *nd = &s->tlas.nodes[stack[--sp]];
        if (!bvh_ray_box(o, inv, best_t, nd->bbmin, nd->bbmax, 1e-7))
            continue;
        if (nd->left < 0) {
            for (int32_t i = 0; i < nd->count; i++) {
                int32_t fid = s->tlas.order[nd->start + i];
                double t = face_intersect(&s->faces[fid], o, d);
                if (t < best_t) { best_t = t; best_f = fid; }
            }
        } else {
            /* depth bound: a median-split tree over n faces is ~log2(n)
             * deep; 128 slots cover any sane scene (guarded regardless) */
            if (sp + 2 > BVH_STACK_MAX) {
                for (int32_t fid = 0; fid < s->n_faces; fid++) {
                    double t = face_intersect(&s->faces[fid], o, d);
                    if (t < best_t) { best_t = t; best_f = fid; }
                }
                *fid_out = best_f;
                return best_t;
            }
            stack[sp++] = nd->left;
            stack[sp++] = nd->right;
        }
    }
    *fid_out = best_f;
    return best_t;
}

#endif /* MIEWB_SCENE_H */
