/* ===========================================================================
 * main.c — miewb-trace entry point.
 *
 * Usage:
 *   miewb-trace --config <case>/cengine/request.json
 *               [--log-level debug|info|warn|error] [--threads N]
 *               [--version]
 *
 * One invocation = one seed (the Python wrapper loops seeds and
 * accumulates mean/std exactly like run_trace.run_one_seed). Outputs land
 * in the request's out_dir:
 *   rays_viz.npy          (M, 13) float64 — the rays.npy contract rows
 *   det_<i>_inc.npy       (bins, H, W) float64 incoherent cube
 *   det_<i>_mask.npy      (H, W) uint8 trim mask
 *   ledger.json           PowerLedger.report() shape
 *   detected.json         per-detector per-key tallies
 *   summary.json          timings + counts (consumed for calibration)
 *   cengine.log           full DEBUG-level log
 * =========================================================================== */
#include "log.h"
#include "npyio.h"
#include "scene.h"
#include "trace.h"
#include "detector.h"
#include "gather.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MIEWB_CENGINE_VERSION "0.1.0-phaseA"

int main(int argc, char **argv) {
    const char *config = NULL;
    int threads_override = -1;

    log_progress_init();
    log_install_crash_handlers(argv[0]);

    /* env default first so --log-level can override it */
    const char *env_level = getenv("MIEWB_LOG_LEVEL");
    if (env_level) {
        int lv = log_level_from_name(env_level);
        if (lv >= 0) log_set_level(lv);
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config = argv[++i];
        } else if (strcmp(argv[i], "--log-level") == 0 && i + 1 < argc) {
            int lv = log_level_from_name(argv[++i]);
            if (lv < 0)
                die(EXIT_INPUT, "unknown --log-level '%s' (debug|info|"
                    "warn|error)", argv[i]);
            log_set_level(lv);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--version") == 0) {
            printf("miewb-trace %s\n", MIEWB_CENGINE_VERSION);
            return 0;
        } else {
            die(EXIT_INPUT, "unknown argument '%s' (usage: miewb-trace "
                "--config request.json [--log-level L] [--threads N])",
                argv[i]);
        }
    }
    if (!config)
        die(EXIT_INPUT, "missing --config <request.json>");

    /* little-endian assumption of npyio ("<f8") — check once */
    {
        const uint16_t probe = 1;
        if (*(const uint8_t *)&probe != 1)
            die(EXIT_INPUT, "big-endian host unsupported (npy writer "
                "emits little-endian)");
    }

    SceneC *scene = request_load(config);

    char path[1200];
    snprintf(path, sizeof path, "%s/cengine.log", scene->out_dir);
    log_open_file(path);
    LOGI("miewb-trace %s, config %s", MIEWB_CENGINE_VERSION, config);

    if (threads_override > 0) scene->threads = threads_override;

    for (int i = 0; i < scene->n_dets; i++)
        det_compute_mask(&scene->dets[i], scene);

    TraceResultC result;
    trace_run(scene, &result);

    /* coherent Huygens gather over the collected detector samples
     * (no-op for purely incoherent scenes) */
    struct timespec g0, g1;
    clock_gettime(CLOCK_MONOTONIC, &g0);
    int64_t gather_pairs = gather_run(scene);
    clock_gettime(CLOCK_MONOTONIC, &g1);
    double gather_seconds = (double)(g1.tv_sec - g0.tv_sec)
                            + 1e-9 * (double)(g1.tv_nsec - g0.tv_nsec);
    if (gather_pairs > 0)
        LOGI("gather: %.3g (sample x point) pairs in %.2f s (%.3g "
             "pairs/s)", (double)gather_pairs, gather_seconds,
             (double)gather_pairs / gather_seconds);

    /* ---- outputs ---- */
    snprintf(path, sizeof path, "%s/rays_viz.npy", scene->out_dir);
    npy_write_f64_2d(path, (const double *)result.viz.v,
                     (size_t)result.viz.n, 13);
    det_write_outputs(scene);
    snprintf(path, sizeof path, "%s/ledger.json", scene->out_dir);
    ledger_write_json(&result.ledger, scene, path, 1e-3);

    snprintf(path, sizeof path, "%s/summary.json", scene->out_dir);
    FILE *f = fopen(path, "w");
    if (!f) die(EXIT_PHYSICS, "cannot write %s", path);
    fprintf(f,
            "{\n"
            "  \"engine\": \"miewb-trace %s\",\n"
            "  \"trace_seconds\": %.6f,\n"
            "  \"gather_seconds\": %.6f,\n"
            "  \"gather_pairs\": %lld,\n"
            "  \"ray_interactions\": %lld,\n"
            "  \"viz_segments\": %lld,\n"
            "  \"closure_err_max\": %.6g\n"
            "}\n",
            MIEWB_CENGINE_VERSION, result.trace_seconds, gather_seconds,
            (long long)gather_pairs,
            (long long)result.rays_traced, (long long)result.viz.n,
            ledger_closure_max(&result.ledger));
    fclose(f);

    double closure = ledger_closure_max(&result.ledger);
    if (closure > 1e-3)
        LOGW("energy closure %.3g exceeds the 1e-3 gate — the Python "
             "wrapper will fail this case", closure);

    trace_result_free(&result);
    for (int i = 0; i < scene->n_dets; i++)
        det_free_gkeys(&scene->dets[i]);
    scene_free(scene);
    log_close_file();
    return 0;
}
