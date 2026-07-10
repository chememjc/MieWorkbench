/* ===========================================================================
 * request.c — parse <case>/cengine/request.json into a SceneC.
 *
 * The request is written by scripts/raytracer/cengine.py (the single
 * source of truth for the schema; schema version checked here). Parsing is
 * strict: any missing/mistyped field dies with the JSON path named
 * (EXIT_INPUT) — silent defaults are how engines drift apart.
 *
 * Also home to trim_build(): the host-side port of TrimPolygon.__init__
 * (surfaces.py:697-758) — regime selection (untrimmed / band / polygon),
 * periodic-u unwrapping, and the sphere zone-vs-cap band disambiguation.
 * =========================================================================== */
#include "scene.h"
#include "log.h"
#include "../vendor/yyjson/yyjson.h"

#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- helpers */
static yyjson_val *need(yyjson_val *obj, const char *key, const char *ctx) {
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (!v)
        die(EXIT_INPUT, "request: missing key '%s' in %s", key, ctx);
    return v;
}

static double need_num(yyjson_val *obj, const char *key, const char *ctx) {
    yyjson_val *v = need(obj, key, ctx);
    if (!yyjson_is_num(v))
        die(EXIT_INPUT, "request: '%s' in %s is not a number", key, ctx);
    return yyjson_get_num(v);
}

static int64_t need_int(yyjson_val *obj, const char *key, const char *ctx) {
    yyjson_val *v = need(obj, key, ctx);
    if (!yyjson_is_int(v))
        die(EXIT_INPUT, "request: '%s' in %s is not an integer", key, ctx);
    return yyjson_get_sint(v);
}

static int need_bool(yyjson_val *obj, const char *key, const char *ctx) {
    yyjson_val *v = need(obj, key, ctx);
    if (!yyjson_is_bool(v))
        die(EXIT_INPUT, "request: '%s' in %s is not a bool", key, ctx);
    return yyjson_get_bool(v);
}

static void need_str_into(yyjson_val *obj, const char *key, const char *ctx,
                          char *dst, size_t cap) {
    yyjson_val *v = need(obj, key, ctx);
    const char *s = yyjson_get_str(v);
    if (!s)
        die(EXIT_INPUT, "request: '%s' in %s is not a string", key, ctx);
    if (strlen(s) >= cap)
        die(EXIT_INPUT, "request: '%s' in %s too long (%zu >= %zu)",
            key, ctx, strlen(s), cap);
    strcpy(dst, s);
}

static kvec3 need_vec3(yyjson_val *obj, const char *key, const char *ctx) {
    yyjson_val *v = need(obj, key, ctx);
    if (!yyjson_is_arr(v) || yyjson_arr_size(v) != 3)
        die(EXIT_INPUT, "request: '%s' in %s is not a 3-array", key, ctx);
    double c[3];
    for (int i = 0; i < 3; i++) {
        yyjson_val *e = yyjson_arr_get(v, (size_t)i);
        if (!yyjson_is_num(e))
            die(EXIT_INPUT, "request: '%s'[%d] in %s not a number",
                key, i, ctx);
        c[i] = yyjson_get_num(e);
    }
    return v3(c[0], c[1], c[2]);
}

/* copy a JSON number array of expected length into a fresh malloc buffer */
static double *need_dbl_array(yyjson_val *obj, const char *key,
                              const char *ctx, size_t expect) {
    yyjson_val *v = need(obj, key, ctx);
    if (!yyjson_is_arr(v))
        die(EXIT_INPUT, "request: '%s' in %s is not an array", key, ctx);
    size_t n = yyjson_arr_size(v);
    if (expect && n != expect)
        die(EXIT_INPUT, "request: '%s' in %s has %zu entries, expected %zu",
            key, ctx, n, expect);
    double *out = (double *)malloc(n * sizeof(double));
    if (!out) die(EXIT_INPUT, "request: out of memory for '%s'", key);
    size_t i, max;
    yyjson_val *e;
    yyjson_arr_foreach(v, i, max, e) {
        if (!yyjson_is_num(e))
            die(EXIT_INPUT, "request: '%s'[%zu] in %s not a number",
                key, i, ctx);
        out[i] = yyjson_get_num(e);
    }
    return out;
}

/* Python-style modulo: result in [0, m) for m > 0. */
static double pymod(double x, double m) {
    double r = fmod(x, m);
    return (r < 0.0) ? r + m : r;
}

/* ------------------------------------------------------------- trim_build */
/* Port of TrimPolygon.__init__ + _full_primitive_area + _choose_band
 * (surfaces.py:697-758). polys: JSON array of polylines, each an array of
 * [x,y,z] points. face_area may be <0 for "not provided". */
static void trim_build(TrimC *t, const SurfC *surf, yyjson_val *polys,
                       double face_area, const char *ctx) {
    t->periodic = surf->periodic_u;
    t->mode = TRIM_POLYGON;
    t->v_lo = 0.0;
    t->v_hi = 0.0;
    t->n_loops = 0;
    t->loop_off = NULL;
    t->pts_u = t->pts_v = NULL;

    size_t n_loops = yyjson_arr_size(polys);
    if (!yyjson_is_arr(polys))
        die(EXIT_INPUT, "request: trim in %s is not an array", ctx);

    /* first pass: total points */
    size_t total = 0;
    size_t li, lmax;
    yyjson_val *loop;
    yyjson_arr_foreach(polys, li, lmax, loop) {
        if (!yyjson_is_arr(loop))
            die(EXIT_INPUT, "request: trim loop %zu in %s not an array",
                li, ctx);
        total += yyjson_arr_size(loop);
    }

    double *u = (double *)malloc(total * sizeof(double));
    double *v = (double *)malloc(total * sizeof(double));
    int32_t *off = (int32_t *)malloc((n_loops + 1) * sizeof(int32_t));
    int *winding = (int *)calloc(n_loops ? n_loops : 1, sizeof(int));
    if (!u || !v || !off || !winding)
        die(EXIT_INPUT, "request: out of memory for trim in %s", ctx);

    size_t at = 0;
    yyjson_arr_foreach(polys, li, lmax, loop) {
        off[li] = (int32_t)at;
        size_t np = yyjson_arr_size(loop);
        size_t start = at;
        size_t pi, pmax;
        yyjson_val *pt;
        yyjson_arr_foreach(loop, pi, pmax, pt) {
            if (!yyjson_is_arr(pt) || yyjson_arr_size(pt) != 3)
                die(EXIT_INPUT,
                    "request: trim loop %zu point %zu in %s not [x,y,z]",
                    li, pi, ctx);
            kvec3 p = v3(yyjson_get_num(yyjson_arr_get(pt, 0)),
                         yyjson_get_num(yyjson_arr_get(pt, 1)),
                         yyjson_get_num(yyjson_arr_get(pt, 2)));
            surf_to_uv(surf, p, &u[at], &v[at]);
            at++;
        }
        /* periodic-u unwrap + winding number (surfaces.py:707-711):
         * du over the CLOSED loop in principal values; w = round(sum/2pi);
         * u <- [u0, u0 + cumsum(du[:-1])] */
        if (t->periodic && np > 0) {
            double sum = 0.0;
            double *du = (double *)malloc(np * sizeof(double));
            if (!du) die(EXIT_INPUT, "request: OOM (trim unwrap)");
            for (size_t i = 0; i < np; i++) {
                size_t j = (i + 1 < np) ? i + 1 : 0;   /* wraps to u[0] */
                double d = u[start + j] - u[start + i];
                d = pymod(d + K_PI, K_TWO_PI) - K_PI;
                du[i] = d;
                sum += d;
            }
            winding[li] = (int)lround(sum / K_TWO_PI);
            double acc = u[start];
            for (size_t i = 1; i < np; i++) {
                acc += du[i - 1];
                u[start + i] = acc;
            }
            free(du);
        }
    }
    off[n_loops] = (int32_t)at;

    /* Regime 1: untrimmed full primitive (surfaces.py:715-719,734-740) */
    double full = -1.0;
    if (surf->kind == SURF_SPHERE)
        full = 4.0 * K_PI * surf->u.sphere.r * surf->u.sphere.r;
    else if (surf->kind == SURF_TORUS)
        full = 4.0 * K_PI * K_PI * surf->u.tor.R * surf->u.tor.r;
    if (face_area > 0.0 && full > 0.0
            && fabs(face_area - full) <= 0.01 * full) {
        t->mode = TRIM_UNTRIMMED;
        free(u); free(v); free(off); free(winding);
        return;
    }

    /* Regime 2: band — some wire encircles the periodic axis
     * (surfaces.py:722-731). v-range from the WINDING loops only. */
    int any_wind = 0;
    for (size_t i = 0; i < n_loops; i++)
        if (winding[i] != 0) any_wind = 1;
    if (any_wind) {
        double v_lo = INFINITY, v_hi = -INFINITY;
        for (size_t i = 0; i < n_loops; i++) {
            if (winding[i] == 0) continue;
            for (int32_t k = off[i]; k < off[i + 1]; k++) {
                if (v[k] < v_lo) v_lo = v[k];
                if (v[k] > v_hi) v_hi = v[k];
            }
        }
        /* sphere zone-vs-polar-cap disambiguation by area matching
         * (_choose_band, surfaces.py:742-758) */
        if (surf->kind == SURF_SPHERE && face_area > 0.0) {
            double R2 = surf->u.sphere.r * surf->u.sphere.r;
            double a_zone = 2.0 * K_PI * R2 * (sin(v_hi) - sin(v_lo));
            double a_north = 2.0 * K_PI * R2 * (1.0 - sin(v_lo));
            double a_south = 2.0 * K_PI * R2 * (sin(v_hi) + 1.0);
            double dz = fabs(a_zone - face_area);
            double dn = fabs(a_north - face_area);
            double ds = fabs(a_south - face_area);
            if (dn < dz && dn <= ds)      v_hi = K_PI / 2.0;
            else if (ds < dz && ds < dn)  v_lo = -K_PI / 2.0;
        }
        t->mode = TRIM_BAND;
        t->v_lo = v_lo;
        t->v_hi = v_hi;
        /* non-winding loops become hole polygons: compact them into the
         * stored arrays */
        size_t hp = 0, hl = 0;
        int32_t *hoff = (int32_t *)malloc((n_loops + 1) * sizeof(int32_t));
        double *hu = (double *)malloc(total * sizeof(double));
        double *hv = (double *)malloc(total * sizeof(double));
        if (!hoff || !hu || !hv) die(EXIT_INPUT, "request: OOM (trim holes)");
        for (size_t i = 0; i < n_loops; i++) {
            if (winding[i] != 0) continue;
            hoff[hl] = (int32_t)hp;
            for (int32_t k = off[i]; k < off[i + 1]; k++) {
                hu[hp] = u[k];
                hv[hp] = v[k];
                hp++;
            }
            hl++;
        }
        hoff[hl] = (int32_t)hp;
        t->n_loops = (int32_t)hl;
        t->loop_off = hoff;
        t->pts_u = hu;
        t->pts_v = hv;
        free(u); free(v); free(off); free(winding);
        return;
    }

    /* Regime 3: generic polygon. */
    t->mode = TRIM_POLYGON;
    t->n_loops = (int32_t)n_loops;
    t->loop_off = off;
    t->pts_u = u;
    t->pts_v = v;
    free(winding);
}

/* ------------------------------------------------------------- face parse */
static SurfC parse_surface(yyjson_val *sobj, const char *ctx) {
    char type[32];
    need_str_into(sobj, "type", ctx, type, sizeof type);
    if (strcmp(type, "plane") == 0)
        return surf_make_plane(need_vec3(sobj, "origin", ctx),
                               need_vec3(sobj, "normal", ctx));
    if (strcmp(type, "sphere") == 0)
        return surf_make_sphere(need_vec3(sobj, "center", ctx),
                                need_num(sobj, "radius", ctx));
    if (strcmp(type, "cylinder") == 0)
        return surf_make_cylinder(need_vec3(sobj, "origin", ctx),
                                  need_vec3(sobj, "axis", ctx),
                                  need_num(sobj, "radius", ctx));
    if (strcmp(type, "cone") == 0)
        return surf_make_cone(need_vec3(sobj, "apex", ctx),
                              need_vec3(sobj, "axis", ctx),
                              need_num(sobj, "half_angle", ctx));
    if (strcmp(type, "torus") == 0)
        return surf_make_torus(need_vec3(sobj, "center", ctx),
                               need_vec3(sobj, "axis", ctx),
                               need_num(sobj, "major_r", ctx),
                               need_num(sobj, "minor_r", ctx));
    if (strcmp(type, "asphere") == 0) {
        double *coeffs = need_dbl_array(sobj, "coeffs", ctx, 0);
        yyjson_val *cv = yyjson_obj_get(sobj, "coeffs");
        int nc = (int)yyjson_arr_size(cv);
        if (nc > ASPHERE_MAX_COEFFS)
            die(EXIT_INPUT, "request: asphere in %s has %d coeffs "
                "(engine max %d)", ctx, nc, ASPHERE_MAX_COEFFS);
        SurfC s = surf_make_asphere(need_vec3(sobj, "vertex", ctx),
                                    need_vec3(sobj, "axis", ctx),
                                    need_num(sobj, "R", ctx),
                                    need_num(sobj, "k", ctx),
                                    coeffs, nc,
                                    need_num(sobj, "r_max", ctx));
        free(coeffs);
        return s;
    }
    die(EXIT_INPUT,
        "request: surface type '%s' in %s is not ported to the C engine "
        "yet — the feature router (scripts/raytracer/cengine.py) should "
        "have sent this scene to the Python engine", type, ctx);
}

static void parse_face_into(FaceC *f, yyjson_val *fobj, const char *ctx) {
    need_str_into(fobj, "id", ctx, f->id, sizeof f->id);
    f->body = (int32_t)need_int(fobj, "body", ctx);
    f->surf = parse_surface(need(fobj, "surface", ctx), ctx);
    f->outward_sign = need_bool(fobj, "orientation_outward", ctx)
                      ? 1.0 : -1.0;
    f->area_m2 = need_num(fobj, "area_m2", ctx);
    trim_build(&f->trim, &f->surf, need(fobj, "trim", ctx), f->area_m2, ctx);
    yyjson_val *det = yyjson_obj_get(fobj, "detector");
    f->detector = det && yyjson_is_int(det)
                  ? (int32_t)yyjson_get_sint(det) : -1;
    yyjson_val *coat = yyjson_obj_get(fobj, "coating");
    f->coating = coat && yyjson_is_int(coat)
                 ? (int32_t)yyjson_get_sint(coat) : -1;
}

/* Source-UV sampling bounds — port of _sample_face_points' bbox logic
 * (sources.py:404-419): untrimmed sphere = full range, band = full u +
 * v_band, polygon = loop bbox. Uses the ALREADY-BUILT trim (unwrapped u). */
static void source_uv_bounds(SourceC *src, const char *ctx) {
    const TrimC *t = &src->emit_face.trim;
    const SurfC *surf = &src->emit_face.surf;
    if (t->mode == TRIM_UNTRIMMED) {
        if (surf->kind != SURF_SPHERE)
            die(EXIT_INPUT, "%s: untrimmed emitting face of unsupported "
                "type", ctx);
        src->u_lo = -K_PI;
        src->u_hi = K_PI;
        src->v_lo = -K_PI / 2.0;
        src->v_hi = K_PI / 2.0;
        return;
    }
    if (t->mode == TRIM_BAND) {
        src->u_lo = -K_PI;
        src->u_hi = K_PI;
        src->v_lo = t->v_lo;
        src->v_hi = t->v_hi;
        return;
    }
    double u_lo = INFINITY, u_hi = -INFINITY;
    double v_lo = INFINITY, v_hi = -INFINITY;
    int32_t n = t->loop_off[t->n_loops];
    if (n == 0)
        die(EXIT_INPUT, "%s: emitting face has an empty trim polygon", ctx);
    for (int32_t i = 0; i < n; i++) {
        if (t->pts_u[i] < u_lo) u_lo = t->pts_u[i];
        if (t->pts_u[i] > u_hi) u_hi = t->pts_u[i];
        if (t->pts_v[i] < v_lo) v_lo = t->pts_v[i];
        if (t->pts_v[i] > v_hi) v_hi = t->pts_v[i];
    }
    src->u_lo = u_lo; src->u_hi = u_hi;
    src->v_lo = v_lo; src->v_hi = v_hi;
}

/* --------------------------------------------------------------- top level */
SceneC *request_load(const char *path) {
    yyjson_read_err err;
    yyjson_doc *doc = yyjson_read_file(path, 0, NULL, &err);
    if (!doc)
        die(EXIT_INPUT, "request: cannot parse %s: %s (pos %zu)",
            path, err.msg, err.pos);
    yyjson_val *root = yyjson_doc_get_root(doc);

    int64_t schema = need_int(root, "schema", "root");
    if (schema != 1)
        die(EXIT_INPUT, "request: schema %lld unsupported (engine speaks 1) "
            "— rebuild the engine or update scripts/raytracer/cengine.py",
            (long long)schema);

    SceneC *s = (SceneC *)calloc(1, sizeof(SceneC));
    if (!s) die(EXIT_INPUT, "request: out of memory");
    need_str_into(root, "out_dir", "root", s->out_dir, sizeof s->out_dir);

    yyjson_val *par = need(root, "params", "root");
    s->max_reflections = (int)need_int(par, "max_reflections", "params");
    s->power_floor = need_num(par, "power_floor", "params");
    s->rays = need_int(par, "rays", "params");
    s->seed = (uint64_t)need_int(par, "seed", "params");
    s->batch_size = need_int(par, "batch_size", "params");
    s->threads = (int)need_int(par, "threads", "params");

    /* wavelengths + ambient tables */
    yyjson_val *lams = need(root, "lams_m", "root");
    s->n_lams = (int)yyjson_arr_size(lams);
    if (s->n_lams <= 0)
        die(EXIT_INPUT, "request: lams_m is empty");
    s->lams_m = need_dbl_array(root, "lams_m", "root", 0);
    s->amb_n_re = need_dbl_array(root, "ambient_n_re", "root",
                                 (size_t)s->n_lams);
    s->amb_n_im = need_dbl_array(root, "ambient_n_im", "root",
                                 (size_t)s->n_lams);

    /* bodies */
    yyjson_val *bodies = need(root, "bodies", "root");
    s->n_bodies = (int)yyjson_arr_size(bodies);
    s->bodies = (BodyC *)calloc((size_t)s->n_bodies, sizeof(BodyC));
    size_t bi, bmax;
    yyjson_val *bobj;
    yyjson_arr_foreach(bodies, bi, bmax, bobj) {
        BodyC *b = &s->bodies[bi];
        char ctx[160];
        b->index = (int32_t)bi;
        need_str_into(bobj, "label", "body", b->label, sizeof b->label);
        need_str_into(bobj, "name", "body", b->name, sizeof b->name);
        snprintf(ctx, sizeof ctx, "body '%s'", b->label);
        char role[24];
        need_str_into(bobj, "role", ctx, role, sizeof role);
        if (strcmp(role, "optic") == 0)         b->role = ROLE_OPTIC;
        else if (strcmp(role, "detector") == 0) b->role = ROLE_DETECTOR;
        else if (strcmp(role, "source") == 0)   b->role = ROLE_SOURCE;
        else die(EXIT_INPUT, "request: %s has unknown role '%s'", ctx, role);
        b->mirror = need_num(bobj, "mirror", ctx);
        b->absorbance = need_num(bobj, "absorbance", ctx);
        b->n_re = need_dbl_array(bobj, "n_re", ctx, (size_t)s->n_lams);
        b->n_im = need_dbl_array(bobj, "n_im", ctx, (size_t)s->n_lams);
        yyjson_val *fa = yyjson_obj_get(bobj, "filter_alpha");
        b->filter_alpha = (fa && !yyjson_is_null(fa))
            ? need_dbl_array(bobj, "filter_alpha", ctx, (size_t)s->n_lams)
            : NULL;
        /* dichroic polarizer (optional) */
        yyjson_val *pol = yyjson_obj_get(bobj, "polarizer");
        if (pol && !yyjson_is_null(pol)) {
            b->has_polarizer = 1;
            char ptype[24];
            need_str_into(pol, "type", ctx, ptype, sizeof ptype);
            if (strcmp(ptype, "circular_left") == 0)
                b->pol_type = POL_CIRCULAR_LEFT;
            else if (strcmp(ptype, "circular_right") == 0)
                b->pol_type = POL_CIRCULAR_RIGHT;
            else
                b->pol_type = POL_LINEAR;
            b->retardance_waves = need_num(pol, "retardance_waves", ctx);
            b->pol_axis = v3_unit(need_vec3(pol, "axis", ctx));
            b->pol_T_par = need_dbl_array(pol, "T_par", ctx,
                                          (size_t)s->n_lams);
            b->pol_T_perp = need_dbl_array(pol, "T_perp", ctx,
                                           (size_t)s->n_lams);
        }
    }

    /* coatings (optional array; faces reference by index) */
    yyjson_val *coats = yyjson_obj_get(root, "coatings");
    s->n_coatings = coats ? (int)yyjson_arr_size(coats) : 0;
    s->coatings = (CoatC *)calloc(
        (size_t)(s->n_coatings ? s->n_coatings : 1), sizeof(CoatC));
    if (coats) {
        size_t ci, cmax;
        yyjson_val *cobj;
        yyjson_arr_foreach(coats, ci, cmax, cobj) {
            CoatC *c = &s->coatings[ci];
            char ctx[64];
            snprintf(ctx, sizeof ctx, "coating[%zu]", ci);
            char kind[16];
            need_str_into(cobj, "kind", ctx, kind, sizeof kind);
            if (strcmp(kind, "tmm") == 0) {
                c->kind = COAT_TMM;
                c->layer_d = need_dbl_array(cobj, "layer_d", ctx, 0);
                yyjson_val *ld = yyjson_obj_get(cobj, "layer_d");
                c->n_layers = (int32_t)yyjson_arr_size(ld);
                if (c->n_layers > COAT_MAX_LAYERS)
                    die(EXIT_INPUT, "request: %s has %d layers (engine "
                        "max %d)", ctx, c->n_layers, COAT_MAX_LAYERS);
                c->layer_n_re = need_dbl_array(
                    cobj, "layer_n_re", ctx,
                    (size_t)c->n_layers * s->n_lams);
                c->layer_n_im = need_dbl_array(
                    cobj, "layer_n_im", ctx,
                    (size_t)c->n_layers * s->n_lams);
            } else if (strcmp(kind, "table") == 0) {
                c->kind = COAT_TABLE;
                c->Rs = need_dbl_array(cobj, "Rs", ctx, (size_t)s->n_lams);
                c->Rp = need_dbl_array(cobj, "Rp", ctx, (size_t)s->n_lams);
                c->Ts = need_dbl_array(cobj, "Ts", ctx, (size_t)s->n_lams);
                c->Tp = need_dbl_array(cobj, "Tp", ctx, (size_t)s->n_lams);
            } else {
                die(EXIT_INPUT, "request: %s has unknown kind '%s'",
                    ctx, kind);
            }
        }
    }

    /* faces */
    yyjson_val *faces = need(root, "faces", "root");
    s->n_faces = (int)yyjson_arr_size(faces);
    s->faces = (FaceC *)calloc((size_t)(s->n_faces ? s->n_faces : 1),
                               sizeof(FaceC));
    size_t fi, fmax;
    yyjson_val *fobj;
    yyjson_arr_foreach(faces, fi, fmax, fobj) {
        char ctx[160];
        snprintf(ctx, sizeof ctx, "face[%zu]", fi);
        parse_face_into(&s->faces[fi], fobj, ctx);
        if (s->faces[fi].body < 0 || s->faces[fi].body >= s->n_bodies)
            die(EXIT_INPUT, "request: face '%s' references body %d (have "
                "%d bodies)", s->faces[fi].id, s->faces[fi].body,
                s->n_bodies);
        if (s->faces[fi].coating >= s->n_coatings)
            die(EXIT_INPUT, "request: face '%s' references coating %d "
                "(have %d)", s->faces[fi].id, s->faces[fi].coating,
                s->n_coatings);
    }

    /* sources */
    yyjson_val *sources = need(root, "sources", "root");
    s->n_sources = (int)yyjson_arr_size(sources);
    if (s->n_sources <= 0)
        die(EXIT_INPUT, "request: no sources");
    s->sources = (SourceC *)calloc((size_t)s->n_sources, sizeof(SourceC));
    size_t si, smax;
    yyjson_val *sobj;
    s->max_strata = 1;
    s->max_pol = 1;
    yyjson_arr_foreach(sources, si, smax, sobj) {
        SourceC *src = &s->sources[si];
        need_str_into(sobj, "label", "source", src->label,
                      sizeof src->label);
        char ctx[160];
        snprintf(ctx, sizeof ctx, "source '%s'", src->label);
        src->body_index = (int32_t)need_int(sobj, "body_index", ctx);
        src->power_W = need_num(sobj, "power_W", ctx);
        src->coherent = (uint8_t)need_bool(sobj, "coherent", ctx);
        char pol[24];
        need_str_into(sobj, "emit_policy", ctx, pol, sizeof pol);
        if (strcmp(pol, "collimated") == 0) {
            src->emit_policy = EMIT_COLLIMATED;
            src->emit_dir = v3_unit(need_vec3(sobj, "emit_dir", ctx));
        } else if (strcmp(pol, "curved") == 0) {
            src->emit_policy = EMIT_CURVED;
            src->flip_all = (uint8_t)need_bool(sobj, "flip_all", ctx);
        } else {
            die(EXIT_INPUT, "request: %s has unknown emit_policy '%s'",
                ctx, pol);
        }
        src->lam_offset = (int32_t)need_int(sobj, "lam_offset", ctx);
        src->n_strata = (int32_t)need_int(sobj, "n_strata", ctx);
        if (src->lam_offset < 0
                || src->lam_offset + src->n_strata > s->n_lams)
            die(EXIT_INPUT, "request: %s strata [%d..%d) outside lams_m "
                "(%d entries)", ctx, src->lam_offset,
                src->lam_offset + src->n_strata, s->n_lams);
        src->n_pol = (int32_t)need_int(sobj, "n_pol", ctx);
        if (src->n_pol < 1 || src->n_pol > 2)
            die(EXIT_INPUT, "request: %s n_pol=%d (must be 1 or 2)",
                ctx, src->n_pol);
        yyjson_val *jones = need(sobj, "jones", ctx);
        if ((int)yyjson_arr_size(jones) != src->n_pol)
            die(EXIT_INPUT, "request: %s jones has %zu rows, n_pol=%d",
                ctx, yyjson_arr_size(jones), src->n_pol);
        for (int p = 0; p < src->n_pol; p++) {
            yyjson_val *row = yyjson_arr_get(jones, (size_t)p);
            if (!yyjson_is_arr(row) || yyjson_arr_size(row) != 4)
                die(EXIT_INPUT, "request: %s jones[%d] must be "
                    "[Es_re, Es_im, Ep_re, Ep_im]", ctx, p);
            src->jones_s[p] = kc(yyjson_get_num(yyjson_arr_get(row, 0)),
                                 yyjson_get_num(yyjson_arr_get(row, 1)));
            src->jones_p[p] = kc(yyjson_get_num(yyjson_arr_get(row, 2)),
                                 yyjson_get_num(yyjson_arr_get(row, 3)));
        }
        src->viz_cap = need_int(sobj, "viz_cap", ctx);
        parse_face_into(&src->emit_face, need(sobj, "emit_face", ctx), ctx);
        source_uv_bounds(src, ctx);
        if (src->n_strata > s->max_strata) s->max_strata = src->n_strata;
        if (src->n_pol > s->max_pol) s->max_pol = src->n_pol;
    }

    /* detectors */
    yyjson_val *dets = need(root, "detectors", "root");
    s->n_dets = (int)yyjson_arr_size(dets);
    s->dets = (DetC *)calloc((size_t)(s->n_dets ? s->n_dets : 1),
                             sizeof(DetC));
    size_t di, dmax;
    yyjson_val *dobj;
    yyjson_arr_foreach(dets, di, dmax, dobj) {
        DetC *d = &s->dets[di];
        need_str_into(dobj, "label", "detector", d->label, sizeof d->label);
        char ctx[160];
        snprintf(ctx, sizeof ctx, "detector '%s'", d->label);
        d->face_id = (int32_t)need_int(dobj, "face_id", ctx);
        if (d->face_id < 0 || d->face_id >= s->n_faces)
            die(EXIT_INPUT, "request: %s face_id %d out of range", ctx,
                d->face_id);
        if (s->faces[d->face_id].detector != (int32_t)di)
            die(EXIT_INPUT, "request: %s face_id %d does not point back "
                "(face.detector=%d)", ctx, d->face_id,
                s->faces[d->face_id].detector);
        d->xhat = need_vec3(dobj, "xhat", ctx);
        d->yhat = need_vec3(dobj, "yhat", ctx);
        d->normal = need_vec3(dobj, "normal", ctx);
        d->x_lo = need_num(dobj, "x_lo", ctx);
        d->y_lo = need_num(dobj, "y_lo", ctx);
        d->pixel_m = need_num(dobj, "pixel_m", ctx);
        d->W = (int32_t)need_int(dobj, "W", ctx);
        d->H = (int32_t)need_int(dobj, "H", ctx);
        d->spectral_bins = (int32_t)need_int(dobj, "spectral_bins", ctx);
        d->lam_lo = need_num(dobj, "lam_lo_m", ctx);
        d->lam_hi = need_num(dobj, "lam_hi_m", ctx);
        if (d->W <= 0 || d->H <= 0 || d->spectral_bins <= 0
                || d->pixel_m <= 0.0)
            die(EXIT_INPUT, "request: %s has non-positive grid dims "
                "(W=%d H=%d bins=%d pixel_m=%g)", ctx, d->W, d->H,
                d->spectral_bins, d->pixel_m);
        size_t cube = (size_t)d->spectral_bins * d->H * d->W;
        d->inc = (double *)calloc(cube, sizeof(double));
        d->mask = (uint8_t *)calloc((size_t)d->H * d->W, 1);
        d->det_inc_W = (double *)calloc(
            (size_t)s->n_sources * s->max_strata * s->max_pol,
            sizeof(double));
        d->det_inc_n = (int64_t *)calloc(
            (size_t)s->n_sources * s->max_strata * s->max_pol,
            sizeof(int64_t));
        if (!d->inc || !d->mask || !d->det_inc_W || !d->det_inc_n)
            die(EXIT_INPUT, "request: %s cube allocation failed "
                "(%zu pixels)", ctx, cube);
    }
    if (s->n_dets == 0)
        die(EXIT_INPUT, "request: no detectors (mirrors scene.py:271)");

    /* basic cross-validation echo */
    LOGI("scene: %d bodies, %d faces, %d sources, %d detectors, %d lams",
         s->n_bodies, s->n_faces, s->n_sources, s->n_dets, s->n_lams);
    yyjson_doc_free(doc);
    return s;
}

void scene_free(SceneC *s) {
    if (!s) return;
    for (int i = 0; i < s->n_bodies; i++) {
        free(s->bodies[i].n_re);
        free(s->bodies[i].n_im);
        free(s->bodies[i].filter_alpha);
        free(s->bodies[i].pol_T_par);
        free(s->bodies[i].pol_T_perp);
    }
    for (int i = 0; i < s->n_coatings; i++) {
        free(s->coatings[i].layer_n_re);
        free(s->coatings[i].layer_n_im);
        free(s->coatings[i].layer_d);
        free(s->coatings[i].Rs);
        free(s->coatings[i].Rp);
        free(s->coatings[i].Ts);
        free(s->coatings[i].Tp);
    }
    free(s->coatings);
    for (int i = 0; i < s->n_faces; i++) {
        free((void *)s->faces[i].trim.loop_off);
        free((void *)s->faces[i].trim.pts_u);
        free((void *)s->faces[i].trim.pts_v);
    }
    for (int i = 0; i < s->n_sources; i++) {
        free((void *)s->sources[i].emit_face.trim.loop_off);
        free((void *)s->sources[i].emit_face.trim.pts_u);
        free((void *)s->sources[i].emit_face.trim.pts_v);
    }
    for (int i = 0; i < s->n_dets; i++) {
        free(s->dets[i].inc);
        free(s->dets[i].mask);
        free(s->dets[i].det_inc_W);
        free(s->dets[i].det_inc_n);
    }
    free(s->bodies);
    free(s->faces);
    free(s->sources);
    free(s->dets);
    free(s->lams_m);
    free(s->amb_n_re);
    free(s->amb_n_im);
    free(s);
}
