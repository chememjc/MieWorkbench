/* ===========================================================================
 * raybuf.h — ray state and growable ray arenas.
 *
 * State mirrors scripts/raytracer/rays.py RayBatch field-for-field (same
 * dtypes, same semantics — see that file's comments), plus the two lineage
 * fields the counter-based RNG needs (plan D2):
 *   ray_key   : u64, pure function of (seed, source, primary index) for
 *               primaries and (parent_key, event_ctr, child_slot) for
 *               children.
 *   event_ctr : u32, incremented once per surface interaction.
 *
 * Layout note (documented deviation from plan D3's SoA sketch): the trace
 * loop is per-ray scalar (BVH traversal + branchy physics), so rays are
 * stored as an array-of-structs — one ray touches 3-4 consecutive cache
 * lines, all of which the interaction uses. SoA pays off only for
 * lockstep-vectorized kernels; the gather (phase D) uses SoA sample
 * buffers for exactly that reason. Everything externally observable is
 * unaffected.
 *
 * Optional Python slots not yet ported (differentials, birth_pos, k_dir,
 * refl_hist) are added in their feature phases; feature routing keeps
 * scenes that need them on the Python engine until then.
 * =========================================================================== */
#ifndef MIEWB_RAYBUF_H
#define MIEWB_RAYBUF_H

#include <stdint.h>
#include "kernels/kmath.h"

#define MEDIUM_STACK_DEPTH 4    /* rays.py:17 */
#define AMBIENT (-1)            /* rays.py:18 */

typedef struct {
    kvec3 pos, dir, s_hat;
    kcplx Es, Ep;               /* Jones; power = |Es|^2 + |Ep|^2 */
    double lam;                 /* vacuum wavelength [m] */
    double opl;                 /* accumulated Sum(Re(n) ds) [m] */
    double n_eff;               /* >0: e-ray phase-index override (phase F) */
    double birth_power;         /* primary's emission power [W] (floor ref) */
    int16_t medium[MEDIUM_STACK_DEPTH];   /* body-index stack, AMBIENT=-1 */
    int8_t depth;
    int8_t pol_mode;            /* 0 iso/o, 1 e, 2/3 biaxial (phase F) */
    int16_t source_id;
    int16_t lam_stratum;
    int16_t pol_stratum;
    int16_t generation;         /* REFLECTION count (transmits don't bump) */
    int16_t lam_idx;            /* index into SceneC.lams_m (plan D1) */
    int32_t last_face;
    uint8_t coherent, viz_flag, scattered;
    uint64_t ray_key;
    uint32_t event_ctr;
} Ray;

/* power = |Es|^2 + |Ep|^2 (rays.py:127) */
static inline double ray_power(const Ray *r) {
    return kc_abs2(r->Es) + kc_abs2(r->Ep);
}

/* Body index of the medium the ray travels in (rays.py current_medium). */
static inline int ray_current_medium(const Ray *r) {
    return (r->depth > 0) ? r->medium[r->depth - 1] : AMBIENT;
}

/* Growable arena of rays. Used for the work queue batches and the
 * per-thread child buffers; growth doubles, hard cap dies with context
 * (never a malloc-failure segfault). */
typedef struct {
    Ray *rays;
    int64_t n, cap;
} RayVec;

void rayvec_init(RayVec *v, int64_t cap);
void rayvec_free(RayVec *v);
void rayvec_push(RayVec *v, const Ray *r);
void rayvec_clear(RayVec *v);
/* append all of src to dst (arena merge, thread order) */
void rayvec_extend(RayVec *dst, const RayVec *src);

#endif /* MIEWB_RAYBUF_H */
