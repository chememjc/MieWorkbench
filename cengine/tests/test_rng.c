/* test_rng.c — Philox lineage RNG contract checks:
 *   - determinism: same (key, event, draw) -> same value, independent of
 *     call order (the property that makes the OpenMP trace reproducible)
 *   - distinct keys/events/draws decorrelate
 *   - uniform moments sane; Box-Muller normal moments sane
 *   - known-answer test for Philox4x32-10 (canonical zero-input vector
 *     from Random123 / numpy)
 */
#include <stdio.h>
#include <stdlib.h>
#include "rng.h"

static int failures = 0;

static void check(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

int main(void) {
    /* ---- Philox4x32-10 known-answer: counter=0, key=0 ----
     * Reference: Random123 kat_vectors (philox4x32 R=10):
     * 6627e8d5 e169c58d bc57ac4c 9b00dbd8 */
    {
        philox4x32_ctr c = {{0, 0, 0, 0}};
        philox4x32_key k = {{0, 0}};
        philox4x32_ctr r = philox4x32_10(c, k);
        check(r.v[0] == 0x6627e8d5u && r.v[1] == 0xe169c58du
              && r.v[2] == 0xbc57ac4cu && r.v[3] == 0x9b00dbd8u,
              "Philox4x32-10 zero-vector KAT");
    }
    /* counter=ff..f, key=ff..f: 408f276d 41c83b0e a20bc7c6 6d5451fd */
    {
        philox4x32_ctr c = {{0xffffffffu, 0xffffffffu, 0xffffffffu,
                             0xffffffffu}};
        philox4x32_key k = {{0xffffffffu, 0xffffffffu}};
        philox4x32_ctr r = philox4x32_10(c, k);
        check(r.v[0] == 0x408f276du && r.v[1] == 0x41c83b0eu
              && r.v[2] == 0xa20bc7c6u && r.v[3] == 0x6d5451fdu,
              "Philox4x32-10 ones-vector KAT");
    }

    /* ---- determinism / order independence ---- */
    {
        uint64_t key = rng_primary_key(42, 3, 12345);
        double a = rng_uniform(key, 7, 4);
        double b = rng_uniform(key, 7, 5);
        /* interleave other draws, then re-ask */
        (void)rng_uniform(key, 9, 0);
        (void)rng_uniform(rng_child_key(key, 7, 1), 0, 0);
        check(rng_uniform(key, 7, 4) == a, "draw (7,4) reproducible");
        check(rng_uniform(key, 7, 5) == b, "draw (7,5) reproducible");
        check(a != b, "distinct draws differ");
    }

    /* ---- key derivation decorrelates lineage ---- */
    {
        uint64_t k1 = rng_primary_key(42, 0, 1);
        uint64_t k2 = rng_primary_key(42, 0, 2);
        uint64_t k3 = rng_primary_key(43, 0, 1);
        check(k1 != k2 && k1 != k3, "primary keys distinct");
        uint64_t c0 = rng_child_key(k1, 0, CHILD_SLOT_TRANSMIT);
        uint64_t c1 = rng_child_key(k1, 0, CHILD_SLOT_REFLECT);
        check(c0 != c1 && c0 != k1, "child keys distinct");
    }

    /* ---- uniform + normal moments ---- */
    {
        const int N = 200000;
        double sum = 0.0, sum2 = 0.0;
        uint64_t key = rng_primary_key(7, 1, 0);
        for (int i = 0; i < N; i++) {
            double u = rng_uniform(key, 0, (uint32_t)i);
            sum += u;
            sum2 += u * u;
        }
        double mean = sum / N;
        double var = sum2 / N - mean * mean;
        check(fabs(mean - 0.5) < 0.005, "uniform mean ~ 0.5");
        check(fabs(var - 1.0 / 12.0) < 0.002, "uniform var ~ 1/12");

        sum = sum2 = 0.0;
        for (int i = 0; i < N / 2; i++) {
            double z0, z1;
            rng_normal2(key, 1, (uint32_t)i, &z0, &z1);
            sum += z0 + z1;
            sum2 += z0 * z0 + z1 * z1;
        }
        double nmean = sum / N;
        double nvar = sum2 / N - nmean * nmean;
        check(fabs(nmean) < 0.01, "normal mean ~ 0");
        check(fabs(nvar - 1.0) < 0.02, "normal var ~ 1");
    }

    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("rng: all checks passed\n");
    return 0;
}
