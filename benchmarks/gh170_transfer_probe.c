// gh-ocannl-170: does routing host<->device copies through pinned staging buffers beat copying
// straight from pageable memory, as Cuda_backend.from_host / to_host (and the HIP mirror) do?
//
// Standalone on purpose, like gh1529_nvrtc_cost.c: it answers a question about the DRIVER, so it
// links the driver directly. The pageable arm is exactly what OCANNL issues today: one
// cuMemcpyHtoDAsync / cuMemcpyDtoHAsync (hipMemcpy*Async) on a non-blocking stream, from a malloc'd
// buffer whose pages were touched (an OCaml bigarray is malloc'd memory).
//
// Build (CUDA, driver API; one command line each):
//   gcc -O2 -o gh170_transfer_probe benchmarks/gh170_transfer_probe.c
//     -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda
// Build (HIP runtime):
//   gcc -O2 -DUSE_HIP -D__HIP_PLATFORM_AMD__ -o gh170_transfer_probe_hip
//     benchmarks/gh170_transfer_probe.c -I/opt/rocm/include -L/opt/rocm/lib -lamdhip64
//
// Run: ./gh170_transfer_probe [reps] [max_mib]   (defaults 15 and 256)
//
// Arms, per direction (h2d, d2h) and size (256 KiB .. max_mib MiB, x4 steps):
//   pageable    one async copy from/to the malloc'd buffer (OCANNL today).
//   pinned      one async copy from/to a page-locked buffer (cuMemHostAlloc / hipHostMalloc): the
//               ceiling of allocating the host array page-locked in the first place.
//   registered  one async copy from/to the malloc'd buffer after cuMemHostRegister /
//               hipHostRegister of it (the issue's other form: registering the host arrays of
//               nodes transferred every step). Its one-off cost is the separate `register` row:
//               call_ms = register, total_ms = register + unregister, for that size.
//   staged      a memcpy through ONE page-locked staging buffer of the full size, then the copy.
//   chunked     the bounded staging pool's shape: the copy split into CHUNK-byte pieces through two
//               page-locked buffers, each guarded by an event, so the host memcpy of one piece
//               overlaps the DMA of the other.
// Each `row` prints the medians over [reps] of
//   call_ms   how long the host was blocked issuing the transfer (until the API call(s) returned);
//   total_ms  until the stream finished the transfer (the data is usable on the far side);
//   gbps      size / total.
// For d2h, staged and chunked include the copy-out memcpy into the pageable destination, so total
// is when the destination holds the data -- what Context.to_host awaits.
// Every arm's result is verified byte for byte (a round trip through the device against a known
// pattern); a mismatch exits 1, so no row is quoted from a transfer that did not move its data.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_HIP
#include <hip/hip_runtime.h>
typedef hipStream_t stream_t;
typedef hipEvent_t event_t;
typedef void *dptr_t;
#define API "hip"
#define CHECK(x)                                                                              \
  do {                                                                                        \
    hipError_t r_ = (x);                                                                      \
    if (r_ != hipSuccess) {                                                                   \
      fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(r_), __FILE__, __LINE__); \
      exit(1);                                                                                \
    }                                                                                         \
  } while (0)
static void init_device(void) {
  CHECK(hipSetDevice(0));
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  printf("device %s integrated=%d\n", p.name, p.integrated);
}
static void host_alloc(void **p, size_t n) { CHECK(hipHostMalloc(p, n, hipHostMallocDefault)); }
static void host_register(void *p, size_t n) { CHECK(hipHostRegister(p, n, hipHostRegisterDefault)); }
static void host_unregister(void *p) { CHECK(hipHostUnregister(p)); }
static dptr_t dev_alloc(size_t n) { void *d; CHECK(hipMalloc(&d, n)); return d; }
static void stream_create(stream_t *s) { CHECK(hipStreamCreateWithFlags(s, hipStreamNonBlocking)); }
static void stream_sync(stream_t s) { CHECK(hipStreamSynchronize(s)); }
static void event_create(event_t *e) { CHECK(hipEventCreateWithFlags(e, hipEventDisableTiming)); }
static void event_record(event_t e, stream_t s) { CHECK(hipEventRecord(e, s)); }
static void event_sync(event_t e) { CHECK(hipEventSynchronize(e)); }
static void h2d(dptr_t d, size_t off, const void *h, size_t n, stream_t s) {
  CHECK(hipMemcpyHtoDAsync((char *)d + off, (void *)h, n, s));
}
static void d2h(void *h, dptr_t d, size_t off, size_t n, stream_t s) {
  CHECK(hipMemcpyDtoHAsync(h, (char *)d + off, n, s));
}
#else
#include <cuda.h>
typedef CUstream stream_t;
typedef CUevent event_t;
typedef CUdeviceptr dptr_t;
#define API "cuda"
#define CHECK(x)                                                                   \
  do {                                                                             \
    CUresult r_ = (x);                                                             \
    if (r_ != CUDA_SUCCESS) {                                                      \
      const char *s_ = NULL;                                                       \
      cuGetErrorString(r_, &s_);                                                   \
      fprintf(stderr, "cuda error %s at %s:%d\n", s_ ? s_ : "?", __FILE__, __LINE__); \
      exit(1);                                                                     \
    }                                                                              \
  } while (0)
static CUcontext ctx;
static void init_device(void) {
  CUdevice dev;
  CHECK(cuInit(0));
  CHECK(cuDeviceGet(&dev, 0));
  CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CHECK(cuCtxSetCurrent(ctx));
  char name[256];
  int integrated = 0;
  CHECK(cuDeviceGetName(name, sizeof name, dev));
  CHECK(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev));
  printf("device %s integrated=%d\n", name, integrated);
}
static void host_alloc(void **p, size_t n) { CHECK(cuMemHostAlloc(p, n, 0)); }
static void host_register(void *p, size_t n) { CHECK(cuMemHostRegister(p, n, 0)); }
static void host_unregister(void *p) { CHECK(cuMemHostUnregister(p)); }
static dptr_t dev_alloc(size_t n) { CUdeviceptr d; CHECK(cuMemAlloc(&d, n)); return d; }
static void stream_create(stream_t *s) { CHECK(cuStreamCreate(s, CU_STREAM_NON_BLOCKING)); }
static void stream_sync(stream_t s) { CHECK(cuStreamSynchronize(s)); }
static void event_create(event_t *e) { CHECK(cuEventCreate(e, CU_EVENT_DISABLE_TIMING)); }
static void event_record(event_t e, stream_t s) { CHECK(cuEventRecord(e, s)); }
static void event_sync(event_t e) { CHECK(cuEventSynchronize(e)); }
static void h2d(dptr_t d, size_t off, const void *h, size_t n, stream_t s) {
  CHECK(cuMemcpyHtoDAsync(d + off, h, n, s));
}
static void d2h(void *h, dptr_t d, size_t off, size_t n, stream_t s) {
  CHECK(cuMemcpyDtoHAsync(h, d + off, n, s));
}
#endif

#define CHUNK ((size_t)4 << 20)
enum { PAGEABLE, PINNED, STAGED, CHUNKED, REGISTERED, N_ARMS };
static const char *arm_names[N_ARMS] = {"pageable", "pinned", "staged", "chunked",
                                       "registered"};

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
static double median(double *v, int n) {
  qsort(v, n, sizeof *v, cmp_double);
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

static void fill(unsigned char *p, size_t n, unsigned seed) {
  for (size_t i = 0; i < n; i++) p[i] = (unsigned char)((i * 2654435761u + seed) >> 13);
}

static stream_t stream;
static event_t ev[2];
static unsigned char *pageable, *pinned, *stage, *chunk_buf[2];
static dptr_t dbuf;

// One h2d transfer of n bytes from [src_pageable] (or the pinned buffer) by [arm]; returns the
// host-blocked time in [*call], and leaves the transfer outstanding on the stream.
static void run_h2d(int arm, size_t n, double *call) {
  double t0 = now_ms();
  switch (arm) {
  case PAGEABLE:
  case REGISTERED: h2d(dbuf, 0, pageable, n, stream); break;
  case PINNED: h2d(dbuf, 0, pinned, n, stream); break;
  case STAGED:
    memcpy(stage, pageable, n);
    h2d(dbuf, 0, stage, n, stream);
    break;
  case CHUNKED: {
    int k = 0;
    for (size_t off = 0; off < n; off += CHUNK, k ^= 1) {
      size_t len = n - off < CHUNK ? n - off : CHUNK;
      event_sync(ev[k]); // the previous use of this buffer has been consumed by the DMA
      memcpy(chunk_buf[k], pageable + off, len);
      h2d(dbuf, off, chunk_buf[k], len, stream);
      event_record(ev[k], stream);
    }
    break;
  }
  }
  *call = now_ms() - t0;
}

// One d2h transfer into the pageable buffer (or the pinned one for PINNED); for STAGED and CHUNKED
// the copy-out into the pageable buffer is part of the transfer, so [*call] covers what the host
// had to stay for, and the caller's stream_sync only drains.
static void run_d2h(int arm, size_t n, double *call) {
  double t0 = now_ms();
  switch (arm) {
  case PAGEABLE:
  case REGISTERED: d2h(pageable, dbuf, 0, n, stream); break;
  case PINNED: d2h(pinned, dbuf, 0, n, stream); break;
  case STAGED:
    d2h(stage, dbuf, 0, n, stream);
    stream_sync(stream);
    memcpy(pageable, stage, n);
    break;
  case CHUNKED: {
    // Issue piece i into buffer i%2, then copy out piece i-1 once its event fired.
    size_t prev_off = 0, prev_len = 0;
    int k = 0;
    for (size_t off = 0; off < n; off += CHUNK, k ^= 1) {
      size_t len = n - off < CHUNK ? n - off : CHUNK;
      d2h(chunk_buf[k], dbuf, off, len, stream);
      event_record(ev[k], stream);
      if (prev_len) {
        event_sync(ev[k ^ 1]);
        memcpy(pageable + prev_off, chunk_buf[k ^ 1], prev_len);
      }
      prev_off = off;
      prev_len = len;
    }
    event_sync(ev[k ^ 1]);
    memcpy(pageable + prev_off, chunk_buf[k ^ 1], prev_len);
    break;
  }
  }
  *call = now_ms() - t0;
}

int main(int argc, char **argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 15;
  size_t max_mib = argc > 2 ? (size_t)atol(argv[2]) : 256;
  if (reps < 1 || max_mib < 1) {
    fprintf(stderr, "usage: %s [reps>=1] [max_mib>=1]\n", argv[0]);
    return 2;
  }
  size_t max_n = max_mib << 20;
  init_device();
  stream_create(&stream);
  event_create(&ev[0]);
  event_create(&ev[1]);
  event_record(ev[0], stream);
  event_record(ev[1], stream);
  pageable = malloc(max_n);
  unsigned char *expect = malloc(max_n);
  if (!pageable || !expect) return 1;
  host_alloc((void **)&pinned, max_n);
  host_alloc((void **)&stage, max_n);
  host_alloc((void **)&chunk_buf[0], CHUNK);
  host_alloc((void **)&chunk_buf[1], CHUNK);
  dbuf = dev_alloc(max_n);
  printf("header api=%s reps=%d chunk_bytes=%zu\n", API, reps, CHUNK);
  printf("header columns: row dir arm bytes call_ms total_ms gbps\n");
  double *calls = malloc(sizeof(double) * reps), *totals = malloc(sizeof(double) * reps);
  for (size_t n = 256 << 10; n <= max_n; n *= 4) {
    for (int r = -1; r < reps; r++) {
      double t0 = now_ms();
      host_register(pageable, n);
      double t1 = now_ms();
      host_unregister(pageable);
      if (r >= 0) {
        calls[r] = t1 - t0;
        totals[r] = now_ms() - t0;
      }
    }
    printf("row reg %-10s %10zu %9.4f %9.4f %7s\n", "register", n, median(calls, reps),
           median(totals, reps), "-");
    for (int dir = 0; dir < 2; dir++) {
      for (int arm = 0; arm < N_ARMS; arm++) {
        if (arm == REGISTERED) host_register(pageable, n);
        for (int r = -1; r < reps; r++) { // r = -1 is an unrecorded warm-up
          unsigned seed = (unsigned)(n + 131 * dir + 17 * arm + r);
          fill(expect, n, seed);
          double call, t0;
          if (dir == 0) {
            memcpy(arm == PINNED ? pinned : pageable, expect, n);
            t0 = now_ms();
            run_h2d(arm, n, &call);
            stream_sync(stream);
          } else {
            h2d(dbuf, 0, expect, n, stream);
            stream_sync(stream);
            memset(arm == PINNED ? pinned : pageable, 0, n);
            t0 = now_ms();
            run_d2h(arm, n, &call);
            stream_sync(stream);
          }
          double total = now_ms() - t0;
          // Verify: h2d reads the device back; d2h compares the destination.
          if (dir == 0) {
            memset(pageable, 0, n);
            d2h(pageable, dbuf, 0, n, stream);
            stream_sync(stream);
            if (memcmp(pageable, expect, n)) {
              fprintf(stderr, "MISMATCH h2d %s %zu\n", arm_names[arm], n);
              return 1;
            }
          } else if (memcmp(arm == PINNED ? pinned : pageable, expect, n)) {
            fprintf(stderr, "MISMATCH d2h %s %zu\n", arm_names[arm], n);
            return 1;
          }
          if (r >= 0) {
            calls[r] = call;
            totals[r] = total;
          }
        }
        if (arm == REGISTERED) host_unregister(pageable);
        double mc = median(calls, reps), mt = median(totals, reps);
        printf("row %s %-10s %10zu %9.4f %9.4f %7.2f\n", dir ? "d2h" : "h2d", arm_names[arm], n, mc,
               mt, n / (mt * 1e6));
        fflush(stdout);
      }
    }
  }
  printf("done\n");
  return 0;
}
