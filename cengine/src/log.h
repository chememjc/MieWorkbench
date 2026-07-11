/* ===========================================================================
 * log.h — logging, progress emission, and fatal-error discipline.
 *
 * Requirements (from the project plan — hard requirements, not niceties):
 *   - No failure may present as a bare segfault: main() installs signal
 *     handlers that print a backtrace and an addr2line hint (log_install_
 *     crash_handlers), and every input/runtime error dies through die()
 *     with named context (face id, body label, sizes).
 *   - Levels: DEBUG < INFO < WARN < ERROR, selected by --log-level or
 *     MIEWB_LOG_LEVEL. Output goes to stderr AND (once log_open_file is
 *     called) <case>/cengine/cengine.log.
 *   - When MIEWB_PROGRESS=1 the engine emits the same "@MIEWB {json}"
 *     stdout lines the Python stages emit (common.py progress_emit), so
 *     run_trace.py can re-broadcast them; the Python wrapper keeps owning
 *     progress.json (no file racing between processes).
 *
 * Exit codes (checked by scripts/raytracer/cengine.py):
 *   2 = invalid input (request/scene validation)
 *   3 = physics runtime error (medium stack, non-convergence, ...)
 *   4 = CUDA error
 *   1 = internal error / crash via signal handler
 * =========================================================================== */
#ifndef MIEWB_LOG_H
#define MIEWB_LOG_H

#include <stdarg.h>

enum {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARN = 2,
    LOG_ERROR = 3,
};

enum {
    EXIT_INPUT = 2,
    EXIT_PHYSICS = 3,
    EXIT_CUDA = 4,
};

void log_set_level(int level);
int log_level_from_name(const char *name);   /* -1 if unknown */
void log_open_file(const char *path);        /* tee to a log file */
void log_close_file(void);

void log_msg(int level, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

#define LOGD(...) log_msg(LOG_DEBUG, __VA_ARGS__)
#define LOGI(...) log_msg(LOG_INFO, __VA_ARGS__)
#define LOGW(...) log_msg(LOG_WARN, __VA_ARGS__)
#define LOGE(...) log_msg(LOG_ERROR, __VA_ARGS__)

/* Fatal: log at ERROR with file:line context, then exit(code). Never
 * returns. Use EXIT_INPUT / EXIT_PHYSICS / EXIT_CUDA. */
void die_at(int code, const char *file, int line, const char *fmt, ...)
    __attribute__((noreturn, format(printf, 4, 5)));
#define die(code, ...) die_at(code, __FILE__, __LINE__, __VA_ARGS__)

/* "@MIEWB {json}" progress lines on stdout, enabled by MIEWB_PROGRESS=1
 * (call log_progress_init once at startup to read the env). Mirrors
 * common.progress_emit's payload keys: stage, frac, msg. */
void log_progress_init(void);
void log_progress(const char *stage, double frac, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

/* SIGSEGV/SIGBUS/SIGFPE/SIGABRT -> backtrace + addr2line hint on stderr
 * and the log file, then _exit(1). */
void log_install_crash_handlers(const char *argv0);

#endif /* MIEWB_LOG_H */
