/* tool_surf_roots.c — debug/parity tool: read a surface spec + rays on
 * stdin, print each ray's candidate roots. Used by the Python-side root
 * parity check (test_cengine_parity.py) to compare surf_roots() against
 * surfaces.py root-for-root.
 *
 * stdin: first line = surface spec:
 *          torus cx cy cz ax ay az R r
 *          sphere cx cy cz r
 *          cylinder ox oy oz ax ay az r
 *          cone ax ay az vx vy vz half_angle   (apex, axis)
 *          asphere vx vy vz ax ay az R k r_max n_coeffs c...
 *        then one ray per line: ox oy oz dx dy dz
 * stdout: one line per ray: K root values (inf for invalid), %.17g
 */
#include <stdio.h>
#include <string.h>
#include "kernels/surf.h"

int main(void) {
    char kind[32];
    SurfC s;
    if (scanf("%31s", kind) != 1) return 2;
    if (strcmp(kind, "torus") == 0) {
        double c[3], a[3], R, r;
        if (scanf("%lf %lf %lf %lf %lf %lf %lf %lf", &c[0], &c[1], &c[2],
                  &a[0], &a[1], &a[2], &R, &r) != 8) return 2;
        s = surf_make_torus(v3(c[0], c[1], c[2]), v3(a[0], a[1], a[2]),
                            R, r);
    } else if (strcmp(kind, "sphere") == 0) {
        double c[3], r;
        if (scanf("%lf %lf %lf %lf", &c[0], &c[1], &c[2], &r) != 4)
            return 2;
        s = surf_make_sphere(v3(c[0], c[1], c[2]), r);
    } else if (strcmp(kind, "cylinder") == 0) {
        double o[3], a[3], r;
        if (scanf("%lf %lf %lf %lf %lf %lf %lf", &o[0], &o[1], &o[2],
                  &a[0], &a[1], &a[2], &r) != 7) return 2;
        s = surf_make_cylinder(v3(o[0], o[1], o[2]), v3(a[0], a[1], a[2]),
                               r);
    } else if (strcmp(kind, "cone") == 0) {
        double ap[3], ax[3], ha;
        if (scanf("%lf %lf %lf %lf %lf %lf %lf", &ap[0], &ap[1], &ap[2],
                  &ax[0], &ax[1], &ax[2], &ha) != 7) return 2;
        s = surf_make_cone(v3(ap[0], ap[1], ap[2]),
                           v3(ax[0], ax[1], ax[2]), ha);
    } else if (strcmp(kind, "asphere") == 0) {
        double v[3], a[3], R, k, rmax, coeffs[ASPHERE_MAX_COEFFS];
        int nc;
        if (scanf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %d", &v[0], &v[1],
                  &v[2], &a[0], &a[1], &a[2], &R, &k, &rmax, &nc) != 10)
            return 2;
        for (int i = 0; i < nc; i++)
            if (scanf("%lf", &coeffs[i]) != 1) return 2;
        s = surf_make_asphere(v3(v[0], v[1], v[2]), v3(a[0], a[1], a[2]),
                              R, k, coeffs, nc, rmax);
    } else {
        fprintf(stderr, "unknown surface kind %s\n", kind);
        return 2;
    }

    double o[3], d[3];
    while (scanf("%lf %lf %lf %lf %lf %lf", &o[0], &o[1], &o[2], &d[0],
                 &d[1], &d[2]) == 6) {
        double t[SURF_K_MAX];
        int K = surf_roots(&s, v3(o[0], o[1], o[2]), v3(d[0], d[1], d[2]),
                           t);
        for (int i = 0; i < K; i++)
            printf("%.17g%c", t[i], i + 1 < K ? ' ' : '\n');
    }
    return 0;
}
