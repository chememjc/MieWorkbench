/* ===========================================================================
 * registry.c — the registry's scene-build machinery (REGISTRY.md §2, §4):
 * per-face handler resolution, the construction-time unknown-token hard
 * error, and the --tokens dump. The InteractionDef/PropagatorDef TABLES
 * themselves live in trace.c (co-located with the physics handlers, which
 * reference trace-local ThreadCtx + static helpers); this file reaches them
 * through registry_interactions()/registry_propagators().
 *
 * The full "known token" set = the interaction/propagator handler tokens
 * PLUS the tokens the C engine supports without a dedicated dispatch entry
 * yet (geometry, the still-inline optic-chain features that the terminal
 * default handler covers, the coherent gather, glue diagnostics, and the
 * bulk/volume effects the homogeneous propagator carries). It must be a
 * SUPERSET of cengine.PORTED — pinned by test_registry_tokens.py.
 * =========================================================================== */
#include "registry.h"
#include "scene.h"
#include "log.h"

#include <string.h>

/* Tokens the C engine supports but that are NOT (yet) their own dispatch
 * entry. Each carries the kind reported by --tokens. Kept here as pure data
 * so this file needs nothing from the physics. */
struct tok_ent { const char *token; const char *kind; };

static const struct tok_ent EXTRA_SUPPORTED[] = {
    /* geometry — resolved by the intersector, never a surface interaction */
    { "surface:plane",    "geometry" },
    { "surface:sphere",   "geometry" },
    { "surface:cylinder", "geometry" },
    { "surface:cone",     "geometry" },
    { "surface:torus",    "geometry" },
    { "surface:asphere",  "geometry" },
    { "surface:mesh",     "geometry" },
    /* coating (step 4) + polarizer (step 5) + roughness/scatter (step 6)
     * are registered SurfaceEffectDefs; birefringence (step 7) is its own
     * terminal InteractionDef — none live here any more. */
    /* filter = additive bulk alpha the homogeneous propagator carries.
     * (particles is now a registered PropagatorDef — step 8 — not here.) */
    { "filter",    "volume" },
    /* pulsed-optics P7 tranche 2 NLO bulk effects (nlo.py). saturable / tpa
     * ride the SAME homogeneous-propagator alpha_add hook as filter (their
     * intensity-dependent absorption lands in absorbed_bulk); kerr adds an
     * intensity-dependent phase to opl there. The per-ray intensity comes
     * from ray_local_intensity (differentials dA, else the source flat-top
     * area). Volume features — no surface dispatch entry. */
    { "saturable", "volume" },
    { "tpa",       "volume" },
    { "kerr",      "volume" },
    /* coherent recombination + glue-level diagnostics — not surface
     * interactions (the gather and the Python-side viz overlay own these) */
    { "coherent",       "gather" },
    { "save_fields",    "gather" },
    { "export_rays",    "diagnostic" },
    { "ghost_analysis", "diagnostic" },
    { "viz_pattern",    "glue" },
    /* P7 ray-differentials port: seed/transport the Igehy differentials and
     * size the coherent gather's per-sample dA from |dPdx x dPdy|. A cross-
     * cutting ray feature (like export_rays), not a surface interaction — no
     * dispatch entry; the trace + gather carry it. */
    { "ray_differentials", "diagnostic" },
    /* pulsed-optics P7 tranche 1: per-body power-weighted bulk-path tally
     * (the GDD-budget input). The trace accumulates ledger.path_tally under
     * track_time; ALL dispersion resolution stays Python-side in
     * build_gdd_budget. Diagnostic — no dispatch entry. */
    { "gdd_budget",     "diagnostic" },
    /* pulsed-optics P7 tranche 1: per-ray gopl/gdd group-delay accumulators
     * (group index / GDD pre-resolved in the request) + per-detector arrival
     * records; the Python finalize_time bins them. Diagnostic — no dispatch
     * entry. (A crystal scene emits the unported time_directional_index token
     * and routes to Python: n_g_eff is not carried in C.) */
    { "time_products",  "diagnostic" },
    /* structural body properties folded into the optic-default handler's
     * unified Fresnel/mirror/absorbance core (not emitted as gate tokens by
     * detect_features; listed so --tokens documents that the C engine owns
     * them). Their standalone split is §3 step 4 (the terminal core). */
    { "mirror",   "structural" },
    { "absorber", "structural" },
};
#define N_EXTRA ((int)(sizeof EXTRA_SUPPORTED / sizeof EXTRA_SUPPORTED[0]))

int registry_supported_token(const char *token) {
    if (!token) return 0;
    int ni;
    const InteractionDef *ia = registry_interactions(&ni);
    for (int i = 0; i < ni; i++)
        if (!ia[i].stub && strcmp(ia[i].token, token) == 0) return 1;
    int np;
    const PropagatorDef *pp = registry_propagators(&np);
    for (int i = 0; i < np; i++)
        if (!pp[i].stub && strcmp(pp[i].token, token) == 0) return 1;
    int nse;
    const SurfaceEffectDef *se = registry_surface_effects(&nse);
    for (int i = 0; i < nse; i++)
        if (strcmp(se[i].token, token) == 0) return 1;
    for (int i = 0; i < N_EXTRA; i++)
        if (strcmp(EXTRA_SUPPORTED[i].token, token) == 0) return 1;
    return 0;
}

void registry_check_features(const char *const *feats, int n_feats) {
    for (int i = 0; i < n_feats; i++) {
        const char *tok = feats[i];
        if (!tok || !*tok) continue;
        if (!registry_supported_token(tok))
            /* EXIT_INPUT == 2 (log.h): the REGISTRY.md §2.2 hard error. The
             * request carries only tokens, not the producing body, so we
             * name the token; a fabricated token has no body by definition.
             * Under --engine auto choose_engine already routed this case to
             * Python — reaching here means a forced --engine c or a stale
             * request, exactly the belt-and-suspenders backstop §2.3 wants. */
            die(EXIT_INPUT,
                "registry: scene feature token '%s' has no implementation in "
                "the C engine registry — routing must fall back to Python "
                "(use --engine auto/python; forced --engine c cannot run it)",
                tok);
    }
}

void registry_resolve_faces(SceneC *s) {
    int ni;
    const InteractionDef *ia = registry_interactions(&ni);
    for (int fid = 0; fid < s->n_faces; fid++) {
        FaceC *f = &s->faces[fid];
        f->n_handlers = 0;
        for (int i = 0; i < ni; i++) {
            if (ia[i].stub) continue;                 /* seam stub — no physics */
            if (!ia[i].match(s, (int32_t)fid)) continue;
            if (f->n_handlers >= MIEWB_MAX_FACE_HANDLERS)
                die(EXIT_INPUT,
                    "registry: face %s resolves to more than %d handlers "
                    "(raise MIEWB_MAX_FACE_HANDLERS)", f->id,
                    MIEWB_MAX_FACE_HANDLERS);
            f->handlers[f->n_handlers++] = &ia[i];
        }
        if (f->n_handlers == 0)
            die(EXIT_INPUT,
                "registry: face %s matched no interaction handler — a "
                "dispatch gap (the optic default must be total)", f->id);
    }
}

/* dedup helper for the dump: has `tok` already been printed? */
static int seen_before(const char *const *seen, int n, const char *tok) {
    for (int i = 0; i < n; i++)
        if (strcmp(seen[i], tok) == 0) return 1;
    return 0;
}

void registry_dump_tokens(FILE *out) {
    const char *seen[256];
    int n_seen = 0;
    fprintf(out, "# miewb-trace interaction registry tokens "
                 "(token<TAB>kind)\n");
    int ni;
    const InteractionDef *ia = registry_interactions(&ni);
    for (int i = 0; i < ni; i++) {
        if (ia[i].stub) continue;    /* seam stub — not a supported token */
        if (seen_before(seen, n_seen, ia[i].token)) continue;
        if (n_seen < 256) seen[n_seen++] = ia[i].token;
        fprintf(out, "%s\tinteraction\n", ia[i].token);
    }
    int np;
    const PropagatorDef *pp = registry_propagators(&np);
    for (int i = 0; i < np; i++) {
        if (pp[i].stub) continue;    /* seam stub — not a supported token */
        if (seen_before(seen, n_seen, pp[i].token)) continue;
        if (n_seen < 256) seen[n_seen++] = pp[i].token;
        fprintf(out, "%s\tpropagator\n", pp[i].token);
    }
    int nse;
    const SurfaceEffectDef *se = registry_surface_effects(&nse);
    for (int i = 0; i < nse; i++) {
        if (seen_before(seen, n_seen, se[i].token)) continue;
        if (n_seen < 256) seen[n_seen++] = se[i].token;
        fprintf(out, "%s\tinteraction\n", se[i].token);
    }
    for (int i = 0; i < N_EXTRA; i++) {
        if (seen_before(seen, n_seen, EXTRA_SUPPORTED[i].token)) continue;
        if (n_seen < 256) seen[n_seen++] = EXTRA_SUPPORTED[i].token;
        fprintf(out, "%s\t%s\n", EXTRA_SUPPORTED[i].token,
                EXTRA_SUPPORTED[i].kind);
    }
}
