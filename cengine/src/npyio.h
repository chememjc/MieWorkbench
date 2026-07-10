/* ===========================================================================
 * npyio.h — minimal NumPy .npy v1.0 writer (and reader for parity tools).
 *
 * The C engine writes raw .npy arrays into <case>/cengine/; the Python
 * wrapper (scripts/raytracer/cengine.py) converts them into the existing
 * rays.npy / detectors/<label>.h5 contract. This keeps libhdf5 out of the
 * C build entirely (plan decision D4).
 *
 * Only what we need: C-contiguous float64 / uint8 arrays, 1-3 dims,
 * little-endian host (checked at startup).
 * =========================================================================== */
#ifndef MIEWB_NPYIO_H
#define MIEWB_NPYIO_H

#include <stddef.h>
#include <stdint.h>

/* Write a C-contiguous array. dtype: "<f8" or "|u1". ndim in 1..3.
 * Dies (EXIT_PHYSICS) on I/O failure — an unwritable case dir is fatal. */
void npy_write(const char *path, const void *data, const char *dtype,
               int ndim, const size_t *shape);

/* Convenience wrappers */
void npy_write_f64_1d(const char *path, const double *a, size_t n);
void npy_write_f64_2d(const char *path, const double *a, size_t n0, size_t n1);
void npy_write_f64_3d(const char *path, const double *a,
                      size_t n0, size_t n1, size_t n2);
void npy_write_u8_2d(const char *path, const uint8_t *a, size_t n0, size_t n1);

#endif /* MIEWB_NPYIO_H */
