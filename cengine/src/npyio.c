/* npyio.c — minimal .npy v1.0 writer. Format reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
 *   magic "\x93NUMPY", version 1.0, uint16-LE header length, then a Python
 *   dict literal padded with spaces to a 64-byte-aligned total, ending \n. */
#include "npyio.h"
#include "log.h"

#include <stdio.h>
#include <string.h>

void npy_write(const char *path, const void *data, const char *dtype,
               int ndim, const size_t *shape) {
    size_t elems = 1;
    char shape_str[128] = "";
    size_t sp = 0;
    for (int i = 0; i < ndim; i++) {
        elems *= shape[i];
        sp += (size_t)snprintf(shape_str + sp, sizeof shape_str - sp,
                               "%zu, ", shape[i]);
    }
    if (ndim == 1) {
        /* python 1-tuple: "(n,)" — the trailing comma is already there */
    } else if (sp >= 2) {
        shape_str[sp - 2] = 0;   /* drop trailing ", " for ndim >= 2 */
    }

    size_t esize = 0;
    if (strcmp(dtype, "<f8") == 0) esize = 8;
    else if (strcmp(dtype, "|u1") == 0) esize = 1;
    else die(EXIT_PHYSICS, "npy_write: unsupported dtype %s", dtype);

    char header[256];
    int hl = snprintf(header, sizeof header,
                      "{'descr': '%s', 'fortran_order': False, "
                      "'shape': (%s), }", dtype, shape_str);
    /* pad with spaces so magic(6)+ver(2)+len(2)+header is 64-aligned,
     * terminated by \n (the format spec's requirement) */
    size_t total = 10 + (size_t)hl + 1;
    size_t pad = (64 - (total % 64)) % 64;

    FILE *f = fopen(path, "wb");
    if (!f) die(EXIT_PHYSICS, "npy_write: cannot open %s for writing", path);
    const uint8_t magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    uint16_t hlen = (uint16_t)((size_t)hl + pad + 1);
    int ok = fwrite(magic, 1, 8, f) == 8
          && fwrite(&hlen, 2, 1, f) == 1
          && fwrite(header, 1, (size_t)hl, f) == (size_t)hl;
    for (size_t i = 0; ok && i < pad; i++) ok = fputc(' ', f) != EOF;
    ok = ok && fputc('\n', f) != EOF;
    if (elems > 0)
        ok = ok && fwrite(data, esize, elems, f) == elems;
    if (fclose(f) != 0) ok = 0;
    if (!ok) die(EXIT_PHYSICS, "npy_write: short write to %s (disk full?)",
                 path);
}

void npy_write_f64_1d(const char *path, const double *a, size_t n) {
    npy_write(path, a, "<f8", 1, &n);
}
void npy_write_f64_2d(const char *path, const double *a, size_t n0,
                      size_t n1) {
    size_t s[2] = {n0, n1};
    npy_write(path, a, "<f8", 2, s);
}
void npy_write_f64_3d(const char *path, const double *a, size_t n0,
                      size_t n1, size_t n2) {
    size_t s[3] = {n0, n1, n2};
    npy_write(path, a, "<f8", 3, s);
}
void npy_write_u8_2d(const char *path, const uint8_t *a, size_t n0,
                     size_t n1) {
    size_t s[2] = {n0, n1};
    npy_write(path, a, "|u1", 2, s);
}
