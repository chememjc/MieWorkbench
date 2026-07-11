/* log.c — see log.h for the contract. */
#define _GNU_SOURCE
#include "log.h"

#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int g_level = LOG_INFO;
static FILE *g_file = NULL;
static int g_progress = 0;
static const char *g_argv0 = "miewb-trace";

static const char *LEVEL_NAMES[] = {"DEBUG", "INFO", "WARN", "ERROR"};

void log_set_level(int level) { g_level = level; }

int log_level_from_name(const char *name) {
    for (int i = 0; i < 4; i++)
        if (strcasecmp(name, LEVEL_NAMES[i]) == 0) return i;
    return -1;
}

void log_open_file(const char *path) {
    g_file = fopen(path, "w");
    if (!g_file)
        LOGW("could not open log file %s (continuing on stderr only)", path);
}

void log_close_file(void) {
    if (g_file) { fclose(g_file); g_file = NULL; }
}

static void log_stamp(FILE *f, int level) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);
    fprintf(f, "%02d:%02d:%02d.%03ld [%s] ", tm.tm_hour, tm.tm_min,
            tm.tm_sec, ts.tv_nsec / 1000000, LEVEL_NAMES[level]);
}

static void log_vmsg(int level, const char *fmt, va_list ap) {
    if (level < g_level && !g_file) return;
    va_list ap2;
    va_copy(ap2, ap);
    if (level >= g_level) {
        log_stamp(stderr, level);
        vfprintf(stderr, fmt, ap);
        fputc('\n', stderr);
    }
    if (g_file) {
        /* the log FILE always records everything, including DEBUG — it is
         * the post-mortem record; stderr respects the level filter */
        log_stamp(g_file, level);
        vfprintf(g_file, fmt, ap2);
        fputc('\n', g_file);
        fflush(g_file);
    }
    va_end(ap2);
}

void log_msg(int level, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    log_vmsg(level, fmt, ap);
    va_end(ap);
}

void die_at(int code, const char *file, int line, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char buf[2048];
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    log_msg(LOG_ERROR, "FATAL (%s:%d): %s", file, line, buf);
    log_close_file();
    exit(code);
}

/* ------------------------------------------------------------- progress */
void log_progress_init(void) {
    const char *p = getenv("MIEWB_PROGRESS");
    g_progress = (p && strcmp(p, "1") == 0);
}

void log_progress(const char *stage, double frac, const char *fmt, ...) {
    if (!g_progress) return;
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    /* minimal JSON escaping for the msg (quotes/backslashes) — messages
     * are engine-authored so control characters never appear */
    char esc[1024];
    size_t j = 0;
    for (size_t i = 0; msg[i] && j < sizeof esc - 2; i++) {
        if (msg[i] == '"' || msg[i] == '\\') esc[j++] = '\\';
        esc[j++] = msg[i];
    }
    esc[j] = 0;
    /* same line shape as common.progress_emit: @MIEWB {json} */
    printf("@MIEWB {\"stage\": \"%s\", \"frac\": %.4f, \"msg\": \"%s\"}\n",
           stage, frac, esc);
    fflush(stdout);
}

/* --------------------------------------------------------- crash handler */
static void crash_handler(int sig) {
    void *frames[64];
    int n = backtrace(frames, 64);
    /* async-signal-safe output only: write() + backtrace_symbols_fd */
    const char *name = (sig == SIGSEGV) ? "SIGSEGV"
                     : (sig == SIGBUS)  ? "SIGBUS"
                     : (sig == SIGFPE)  ? "SIGFPE"
                     : (sig == SIGABRT) ? "SIGABRT" : "signal";
    char hdr[256];
    int m = snprintf(hdr, sizeof hdr,
                     "\n=== miewb-trace crashed: %s ===\nBacktrace (run "
                     "`addr2line -f -e %s <addr>` for source lines):\n",
                     name, g_argv0);
    ssize_t rc = write(STDERR_FILENO, hdr, (size_t)m);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);
    if (g_file) {
        int fd = fileno(g_file);
        rc = write(fd, hdr, (size_t)m);
        backtrace_symbols_fd(frames, n, fd);
    }
    (void)rc;
    _exit(1);
}

void log_install_crash_handlers(const char *argv0) {
    g_argv0 = argv0;
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = crash_handler;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
}
