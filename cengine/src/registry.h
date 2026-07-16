/* ===========================================================================
 * registry.h — the P3 interaction registry (core-v3 round; REGISTRY.md).
 *
 * Two seam kinds (REGISTRY.md §1):
 *   - InteractionDef : what happens when a ray meets a face. Resolved at
 *     scene build into a per-face ORDERED handler list (order encodes the
 *     historical process_ray precedence: detector-screen -> grating -> the
 *     optic chain). Each apply() consumes the parent ray, pushes children,
 *     and books energy ONLY through the ledger.
 *   - PropagatorDef : what happens to a ray BETWEEN hits (segment advance:
 *     position, fp64 OPL, bulk absorption). The homogeneous propagator is
 *     the identity-cost default; GRIN / fluorescence are fill-in later.
 *
 * The token strings MUST mirror what scripts/raytracer/cengine.py's
 * detect_features() emits + PORTED; the binary's --tokens dump vs that
 * Python set is pinned by scripts/raytracer/tests/test_registry_tokens.py.
 *
 * LANGUAGE (REGISTRY.md §5): flat function-pointer arrays, no virtual
 * dispatch, no allocation in the hot path — the codebase idiom. The C++17
 * host conversion lands with the buffer/serve-loop work, NOT here.
 * =========================================================================== */
#ifndef MIEWB_REGISTRY_H
#define MIEWB_REGISTRY_H

#include <stdint.h>
#include <stdio.h>
#include "kernels/kmath.h"

/* forward declarations — every use below is via pointer, so the full
 * definitions (scene.h / raybuf.h / trace.c) need not be visible here.
 * C11 permits these redundant typedefs alongside the real definitions. */
typedef struct SceneC SceneC;
typedef struct FaceC FaceC;
typedef struct BodyC BodyC;
typedef struct Ray Ray;
typedef struct ThreadCtx ThreadCtx;   /* trace-local; opaque to the registry */

/* Max resolved surface handlers per face. A detector-screen face resolves
 * to two (detector_event then the thin-screen continuation); the rest to
 * one. 4 leaves headroom for the later stacked ports (coating+scatter). */
#define MIEWB_MAX_FACE_HANDLERS 4

/* The minimal hit tuple process_ray hands each surface handler — factored
 * from the locals process_ray already carries (there was no HitInfo struct
 * before; the dispatch passed face/body/start_pos/start_opl positionally). */
typedef struct HitInfo {
    int32_t fid;              /* index into SceneC.faces */
    const FaceC *face;        /* &SceneC.faces[fid] */
    const BodyC *body;        /* &SceneC.bodies[face->body] */
    double t;                 /* segment distance from start_pos to the hit */
    kvec3 start_pos;          /* segment start (coherent Huygens sample pos) */
    double start_opl;         /* segment-start OPL (Huygens sample phase) */
} HitInfo;

/* Surface interaction (REGISTRY.md §1.1). */
typedef struct InteractionDef {
    const char *token;
    /* Scene-build time: does this interaction apply to face fid? A pure
     * function of the scene description, never of ray state. */
    int (*match)(const SceneC *s, int32_t fid);
    /* Trace time: consume the parent ray, push children, book energy only
     * through the ledger. Owns its complete energy bookkeeping. */
    void (*apply)(const SceneC *s, ThreadCtx *cx, const Ray *ray,
                  const HitInfo *hit);
} InteractionDef;

/* Volume propagator (REGISTRY.md §1.2). */
typedef struct PropagatorDef {
    const char *token;
    int (*match_medium)(const SceneC *s, const Ray *ray);   /* push/build time */
    /* advance the ray by the segment to t_hit: position stays with the
     * caller, fp64 OPL and bulk absorption (ledger) are the propagator's. */
    void (*advance)(const SceneC *s, ThreadCtx *cx, Ray *ray, double t_hit);
} PropagatorDef;

/* The static tables live in trace.c, co-located with the physics handlers
 * (they reference trace-local ThreadCtx + static helpers). registry.c
 * reaches them through these accessors. */
const InteractionDef *registry_interactions(int *n_out);
const PropagatorDef *registry_propagators(int *n_out);

/* Scene-build resolution (REGISTRY.md §2.1): fill each face's ordered
 * handler list from the registry, hard-erroring on a dispatch gap or
 * overflow. */
void registry_resolve_faces(SceneC *s);

/* Construction-time hard-error contract (REGISTRY.md §2.2): every feature
 * token the request carries is checked against the C registry. An unknown
 * token is a hard error (exit 2) naming it — never a silent skip. */
int registry_supported_token(const char *token);         /* 1 if known */
void registry_check_features(const char *const *feats, int n_feats);

/* --tokens dump (REGISTRY.md §4): one "token<TAB>kind" line per known
 * token, deduplicated. The pytest parses field 1 of each non-'#' line. */
void registry_dump_tokens(FILE *out);

#endif /* MIEWB_REGISTRY_H */
