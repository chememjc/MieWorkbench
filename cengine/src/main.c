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
 *
 * --serve (P3 persistent worker): instead of one config, read newline-
 * delimited request-file paths on stdin and process each exactly as the
 * one-shot path does, hoisting the process (hence the CUDA context + the
 * reusable device-buffer pool) across requests. Per request a single
 * protocol line is written to stdout, `@MIEWB-WORKER {"request": "<path>",
 * "rc": <int>}`, with a LEADING newline (the fcserver discipline: engine
 * noise that lacks a trailing newline cannot glue onto the protocol line,
 * and the client scans for the prefix mid-line). A recoverable per-request
 * die() (scene-load / physics validation, single-threaded, non-CUDA) reports
 * rc!=0 and the loop continues; a process-fatal signal or a CUDA fault ends
 * the worker and the client falls back to a one-shot invocation.
 * =========================================================================== */
#define _GNU_SOURCE
#include "log.h"
#include "npyio.h"
#include "scene.h"
#include "trace.h"
#include "detector.h"
#include "gather.h"
#include "ledger.h"

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MIEWB_CENGINE_VERSION "0.1.0-phaseA"

/* Process one loaded request exactly as the one-shot binary always has.
 * Returns 0 on success; a failure die()s (which in --serve mode longjmps
 * back to the serve loop, otherwise exit()s — see log.h). */
static int run_request(const char *config, int threads_override) {
    SceneC *scene = request_load(config);

    char path[1200];
    snprintf(path, sizeof path, "%s/cengine.log", scene->out_dir);
    log_open_file(path);
    LOGI("miewb-trace %s, config %s", MIEWB_CENGINE_VERSION, config);

    if (threads_override > 0) scene->threads = threads_override;

    for (int i = 0; i < scene->n_dets; i++)
        det_compute_mask(&scene->dets[i], scene);

    TraceResultC result;
    if (scene->gather_only) {
        /* P1 final stage: no tracing — load the driver's merged sample
         * dump + accumulator snapshots and run the normal gather below.
         * The ledger is all-zero here (the Python driver merged the real
         * per-chunk ledgers itself); rays_viz.npy is empty likewise. */
        ledger_init(&result.ledger, scene);
        memset(&result.viz, 0, sizeof result.viz);   /* empty viz store */
        result.rays_traced = 0;
        result.trace_seconds = 0.0;
        det_load_gather_state(scene);
    } else {
        trace_run(scene, &result);
    }

    /* coherent Huygens gather over the collected detector samples
     * (no-op for purely incoherent scenes). P1 gather_skip: this invocation
     * is a TRACE-ONLY chunk — serialize the coherent samples instead and let
     * the Python driver run the single final gather over the merged set. */
    struct timespec g0, g1;
    clock_gettime(CLOCK_MONOTONIC, &g0);
    int64_t gather_pairs = 0;
    if (scene->gather_skip) {
        det_dump_gkeys(scene);
        LOGI("gather_skip: dumped coherent samples for the Python driver's "
             "final gather (primary range [%lld,%lld))",
             (long long)scene->primary_lo, (long long)scene->primary_hi);
    } else {
        gather_pairs = gather_run(scene);
    }
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
    det_write_exports(scene);
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
            "  \"primary_lo\": %lld,\n"
            "  \"primary_hi\": %lld,\n"
            "  \"rays_total\": %lld,\n"
            "  \"gather_skip\": %s,\n"
            "  \"gather_only\": %s,\n"
            "  \"closure_err_max\": %.6g\n"
            "}\n",
            MIEWB_CENGINE_VERSION, result.trace_seconds, gather_seconds,
            (long long)gather_pairs,
            (long long)result.rays_traced, (long long)result.viz.n,
            (long long)scene->primary_lo, (long long)scene->primary_hi,
            (long long)scene->rays, scene->gather_skip ? "true" : "false",
            scene->gather_only ? "true" : "false",
            ledger_closure_max(&result.ledger));
    fclose(f);

    double closure = ledger_closure_max(&result.ledger);
    if (closure > 1e-3)
        LOGW("energy closure %.3g exceeds the 1e-3 gate — the Python "
             "wrapper will fail this case", closure);

    trace_result_free(&result);
    for (int i = 0; i < scene->n_dets; i++)
        det_free_gkeys(&scene->dets[i]);
    det_free_exports(scene);
    scene_free(scene);
    log_close_file();
    return 0;
}

/* Minimal JSON string escaping for the request path on the protocol line
 * (quotes + backslashes only — paths never carry control characters). */
static void json_escape(const char *in, char *out, size_t cap) {
    size_t j = 0;
    for (size_t i = 0; in[i] && j + 2 < cap; i++) {
        if (in[i] == '"' || in[i] == '\\') out[j++] = '\\';
        out[j++] = in[i];
    }
    out[j] = 0;
}

/* --serve worker loop. Reads request-file paths (one per line) on stdin;
 * per request runs run_request() under a die()-recovery target and reports
 * `@MIEWB-WORKER {"request": ..., "rc": ...}` on stdout. EOF -> exit 0. */
static int serve_loop(int threads_override) {
    LOGI("miewb-trace %s: --serve worker ready", MIEWB_CENGINE_VERSION);
#ifdef MIEWB_HAS_CUDA
    gather_cuda_worker_init();          /* warm the primary context up front */
#endif
    /* test hook: simulate a worker crash BEFORE responding to the Nth
     * request, so the client's fallback-to-one-shot path is exercised. */
    const char *die_after_s = getenv("MIEWB_WORKER_DIE_AFTER");
    long die_after = die_after_s ? atol(die_after_s) : 0;
    long served = 0;

    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    while ((len = getline(&line, &cap, stdin)) != -1) {
        /* trim trailing newline/CR and surrounding whitespace */
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'
                           || line[len - 1] == ' ' || line[len - 1] == '\t'))
            line[--len] = 0;
        char *path = line;
        while (*path == ' ' || *path == '\t') path++;
        if (*path == 0) continue;       /* blank line: ignore */

        served++;
        if (die_after > 0 && served >= die_after) {
            LOGW("MIEWB_WORKER_DIE_AFTER=%ld reached — simulating worker "
                 "death before responding", die_after);
            free(line);
            _exit(137);                 /* no protocol line: client sees EOF */
        }

        int rc;
        jmp_buf recov;
        int jc = setjmp(recov);
        if (jc == 0) {
            log_set_die_recovery(&recov);
            rc = run_request(path, threads_override);
        } else {
            rc = jc;                    /* die() longjmped with its exit code */
        }
        log_set_die_recovery(NULL);
        log_close_file();               /* close a per-request log left open
                                         * by a recovered die() (no-op if
                                         * run_request already closed it) */

        char esc[2400];
        json_escape(path, esc, sizeof esc);
        /* LEADING newline: guarantees the protocol line starts fresh even
         * if un-terminated engine noise preceded it on stdout. */
        printf("\n@MIEWB-WORKER {\"request\": \"%s\", \"rc\": %d}\n", esc, rc);
        fflush(stdout);
    }
    free(line);
#ifdef MIEWB_HAS_CUDA
    gather_cuda_pool_free();            /* release the device pool at exit */
#endif
    LOGI("miewb-trace: --serve stdin EOF, worker exiting");
    return 0;
}

int main(int argc, char **argv) {
    const char *config = NULL;
    int threads_override = -1;
    int serve = 0;

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
        } else if (strcmp(argv[i], "--serve") == 0) {
            serve = 1;
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
                "--config request.json | --serve [--log-level L] "
                "[--threads N])", argv[i]);
        }
    }

    /* little-endian assumption of npyio ("<f8") — check once */
    {
        const uint16_t probe = 1;
        if (*(const uint8_t *)&probe != 1)
            die(EXIT_INPUT, "big-endian host unsupported (npy writer "
                "emits little-endian)");
    }

    if (serve) {
        if (config)
            die(EXIT_INPUT, "--serve and --config are mutually exclusive "
                "(paths arrive on stdin in serve mode)");
        return serve_loop(threads_override);
    }

    if (!config)
        die(EXIT_INPUT, "missing --config <request.json> (or --serve)");
    int rc = run_request(config, threads_override);
#ifdef MIEWB_HAS_CUDA
    gather_cuda_pool_free();            /* free the pool before context exit */
#endif
    return rc;
}
