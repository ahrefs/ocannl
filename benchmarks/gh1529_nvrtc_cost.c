// gh-ocannl-1529: what does a CUDA compile cost per PROGRAM, and what per KERNEL?
//
// The HIP kernel code cache (gh-ocannl-1314, docs/agent-notes/backend-memory.md) rests on hiprtc
// costing mostly per program; this probe asks the same of nvrtc AND of the driver's PTX JIT
// (cuModuleLoadDataEx), the two halves of a CUDA compile: OCANNL's Cuda_backend emits PTX at an
// arch floor, and the driver compiles that PTX to machine code when the module loads.
//
// Standalone on purpose, like tools/nvrtc_reassoc_probe.c: it answers a question about the
// TOOLKIT, so it links nvrtc and the driver directly. Build:
//
//   gcc -O2 -o gh1529_nvrtc_cost benchmarks/gh1529_nvrtc_cost.c \
//     -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvrtc -lcuda -lpthread
//
// Run (benchmarks/gh1529_nvrtc_cost.sh drives it; CUDA_CACHE_DISABLE=1 makes every load JIT):
//
//   CUDA_CACHE_DISABLE=1 ./gh1529_nvrtc_cost <file.cu> [nvrtc option...]
//
// <file.cu> is a routine's emitted CUDA source as OCANNL compiled it (its .cu build artifact, written with
// the conditional includes already prepended): a prelude (includes, builtins), then the kernels,
// each starting at a line `extern "C" __global__ void <name>(`. The prelude is everything before
// the first kernel; a "program of k kernels" is the prelude plus those k kernel definitions. Every
// program carries the WHOLE prelude, where OCANNL's per-kernel source carries only the builtins
// that kernel uses, so a per-program cost read here is an upper bound on OCANNL's.
//
// What it prints, one `row` line per measurement (seconds, medians over the repeats):
//   (each `row` also lists its runs in order, nvrtc/load, so a cold first run shows)
//   synthetic  N trivial one-line kernels: one program of all N, vs N programs of one -- the
//              brief's "N trivial programs vs one program of N kernels", with no prelude at all.
//   joint      the routine as one program (what OCANNL compiles today).
//   subset k   k kernels, evenly spaced through the routine, as one program (what a kernel code
//              cache compiles when k kernels changed). A least-squares fit of the subsets'
//              times against kernel count gives `fit ... fixed_s=... per_kernel_s=...`.
//   perkernel  every kernel as its own program, serially (twice), and on 8 threads.
//   cubin      the joint program and one subset compiled straight to machine code
//              (--gpu-architecture=sm_<device cc>, nvrtcGetCUBIN), which moves the JIT into nvrtc.
// Every compiled program's PTX is checked to hold exactly as many `.entry` directives as kernels
// were given, and the probe exits 1 on any compile or load failure, so no row is quoted from a
// program that did not compile what it claims.
#include <cuda.h>
#include <nvrtc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NVRTC_CHECK(x)                                                                        \
  do {                                                                                        \
    nvrtcResult r_ = (x);                                                                     \
    if (r_ != NVRTC_SUCCESS) {                                                                \
      fprintf(stderr, "nvrtc error %s at %s:%d\n", nvrtcGetErrorString(r_), __FILE__, __LINE__); \
      exit(1);                                                                                \
    }                                                                                         \
  } while (0)
#define CU_CHECK(x)                                                                      \
  do {                                                                                   \
    CUresult r_ = (x);                                                                   \
    if (r_ != CUDA_SUCCESS) {                                                            \
      const char *s_ = NULL;                                                             \
      cuGetErrorString(r_, &s_);                                                         \
      fprintf(stderr, "cuda error %s at %s:%d\n", s_ ? s_ : "?", __FILE__, __LINE__);    \
      exit(1);                                                                           \
    }                                                                                    \
  } while (0)

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int n_opts;
static const char **opts;       // the caller's nvrtc options, for PTX
static const char **cubin_opts; // the same with every --gpu-architecture replaced by sm_<cc>
static int n_cubin_opts;

typedef struct {
  char *code;
  size_t size;
  double nvrtc_s;
} compiled;

static int count_entries(const char *ptx) {
  int n = 0;
  for (const char *p = ptx; (p = strstr(p, ".entry ")) != NULL; p += 7) n++;
  return n;
}

// Compiles [src]; [cubin] selects machine code over PTX. Exits on failure, printing the log.
static compiled compile_src(const char *src, const char *name, int cubin) {
  compiled c = {0};
  nvrtcProgram prog;
  double t0 = now_s();
  NVRTC_CHECK(nvrtcCreateProgram(&prog, src, name, 0, NULL, NULL));
  nvrtcResult r = cubin ? nvrtcCompileProgram(prog, n_cubin_opts, cubin_opts)
                        : nvrtcCompileProgram(prog, n_opts, opts);
  if (r != NVRTC_SUCCESS) {
    size_t log_size;
    nvrtcGetProgramLogSize(prog, &log_size);
    char *log = malloc(log_size + 1);
    nvrtcGetProgramLog(prog, log);
    log[log_size] = 0;
    fprintf(stderr, "compile of %s failed (%s):\n%s\n", name, nvrtcGetErrorString(r), log);
    exit(1);
  }
  if (cubin) {
    NVRTC_CHECK(nvrtcGetCUBINSize(prog, &c.size));
    c.code = malloc(c.size);
    NVRTC_CHECK(nvrtcGetCUBIN(prog, c.code));
  } else {
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &c.size));
    c.code = malloc(c.size);
    NVRTC_CHECK(nvrtcGetPTX(prog, c.code));
  }
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));
  c.nvrtc_s = now_s() - t0;
  return c;
}

static double load_s(const compiled *c) {
  CUmodule m;
  double t0 = now_s();
  CU_CHECK(cuModuleLoadDataEx(&m, c->code, 0, NULL, NULL));
  double t = now_s() - t0;
  CU_CHECK(cuModuleUnload(m));
  return t;
}

// The source.
static char *prelude;
static int n_kernels;
static char **kernels; // each kernel's definition text
static char **knames;

static char *read_file(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    exit(2);
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *s = malloc((size_t)n + 1);
  if (fread(s, 1, (size_t)n, f) != (size_t)n) {
    perror(path);
    exit(2);
  }
  s[n] = 0;
  fclose(f);
  return s;
}

static const char *kernel_marker = "extern \"C\" __global__";

static void split_source(char *src) {
  // Kernel starts: the marker at the beginning of a line.
  int cap = 64;
  char **starts = malloc(sizeof(char *) * (size_t)cap);
  for (char *p = src; (p = strstr(p, kernel_marker)) != NULL; p += strlen(kernel_marker)) {
    if (p != src && p[-1] != '\n') continue;
    if (n_kernels == cap) starts = realloc(starts, sizeof(char *) * (size_t)(cap *= 2));
    starts[n_kernels++] = p;
  }
  if (n_kernels == 0) {
    fprintf(stderr, "no line starts with %s: not an OCANNL CUDA routine\n", kernel_marker);
    exit(2);
  }
  prelude = strndup(src, (size_t)(starts[0] - src));
  kernels = malloc(sizeof(char *) * (size_t)n_kernels);
  knames = malloc(sizeof(char *) * (size_t)n_kernels);
  for (int i = 0; i < n_kernels; i++) {
    char *end = i + 1 < n_kernels ? starts[i + 1] : src + strlen(src);
    kernels[i] = strndup(starts[i], (size_t)(end - starts[i]));
    char *v = strstr(kernels[i], " void ");
    char *paren = v ? strchr(v, '(') : NULL;
    if (!v || !paren) {
      fprintf(stderr, "kernel %d: no `void <name>(`\n", i);
      exit(2);
    }
    v += 6;
    knames[i] = strndup(v, (size_t)(paren - v));
  }
  free(starts);
}

// A program of the prelude plus the kernels at [idx].
static char *program_of(const int *idx, int k) {
  size_t len = strlen(prelude) + 1;
  for (int j = 0; j < k; j++) len += strlen(kernels[idx[j]]) + 1;
  char *s = malloc(len);
  char *p = stpcpy(s, prelude);
  for (int j = 0; j < k; j++) {
    p = stpcpy(p, kernels[idx[j]]);
    p = stpcpy(p, "\n");
  }
  return s;
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
static double median(double *v, int n) {
  qsort(v, (size_t)n, sizeof(double), cmp_double);
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

#define REPEATS 5

// Compiles and loads one program [repeats] times; prints a row; returns the nvrtc median.
static void measure_program(const char *label, const char *src, int k, int cubin, int repeats,
                            double *nvrtc_med, double *load_med) {
  double nv[REPEATS], ld[REPEATS];
  size_t size = 0;
  for (int r = 0; r < repeats; r++) {
    compiled c = compile_src(src, "gh1529.cu", cubin);
    if (!cubin && count_entries(c.code) != k) {
      fprintf(stderr, "%s: PTX holds %d entries, %d kernels were given\n", label,
              count_entries(c.code), k);
      exit(1);
    }
    nv[r] = c.nvrtc_s;
    ld[r] = load_s(&c);
    size = c.size;
    free(c.code);
  }
  // The runs in order, before [median] sorts them: a first run slower than the rest is a warm-up
  // or a cache the later runs hit, which a median alone would hide.
  char runs[256];
  int off = 0;
  for (int r = 0; r < repeats; r++)
    off += snprintf(runs + off, sizeof runs - (size_t)off, "%s%.4f/%.4f", r ? "," : "", nv[r], ld[r]);
  *nvrtc_med = median(nv, repeats);
  *load_med = median(ld, repeats);
  printf("row %s kernels=%d src_bytes=%zu %s_bytes=%zu nvrtc_s=%.4f load_s=%.4f total_s=%.4f "
         "runs_nvrtc/load=%s\n",
         label, k, strlen(src), cubin ? "cubin" : "ptx", size, *nvrtc_med, *load_med,
         *nvrtc_med + *load_med, runs);
  fflush(stdout);
}

// Per-kernel programs, compiled (and then loaded) on [threads] threads; prints a row.
typedef struct {
  int next;
  pthread_mutex_t lock;
  double nvrtc_sum, load_sum;
  CUcontext ctx;
} work;

static void *perkernel_worker(void *arg) {
  work *w = arg;
  CU_CHECK(cuCtxSetCurrent(w->ctx));
  for (;;) {
    pthread_mutex_lock(&w->lock);
    int i = w->next++;
    pthread_mutex_unlock(&w->lock);
    if (i >= n_kernels) break;
    char *src = program_of(&i, 1);
    compiled c = compile_src(src, "gh1529.cu", 0);
    if (count_entries(c.code) != 1) {
      fprintf(stderr, "kernel %s: PTX holds %d entries\n", knames[i], count_entries(c.code));
      exit(1);
    }
    double l = load_s(&c);
    pthread_mutex_lock(&w->lock);
    w->nvrtc_sum += c.nvrtc_s;
    w->load_sum += l;
    pthread_mutex_unlock(&w->lock);
    free(c.code);
    free(src);
  }
  return NULL;
}

static void measure_perkernel(int threads, CUcontext ctx) {
  work w = {.next = 0, .nvrtc_sum = 0, .load_sum = 0, .ctx = ctx};
  pthread_mutex_init(&w.lock, NULL);
  pthread_t *ts = malloc(sizeof(pthread_t) * (size_t)threads);
  double t0 = now_s();
  for (int t = 0; t < threads; t++) pthread_create(&ts[t], NULL, perkernel_worker, &w);
  for (int t = 0; t < threads; t++) pthread_join(ts[t], NULL);
  double wall = now_s() - t0;
  printf("row perkernel threads=%d programs=%d wall_s=%.4f nvrtc_sum_s=%.4f load_sum_s=%.4f\n",
         threads, n_kernels, wall, w.nvrtc_sum, w.load_sum);
  fflush(stdout);
  free(ts);
}

static void synthetic(int n) {
  // N trivial kernels, no prelude: one program of all, then each alone.
  size_t cap = (size_t)n * 128 + 1;
  char *all = malloc(cap);
  char *p = all;
  *p = 0;
  for (int i = 0; i < n; i++)
    p += sprintf(p, "extern \"C\" __global__ void t%d(float *x) { x[threadIdx.x] += %d.0f; }\n", i,
                 i + 1);
  double nv, ld;
  measure_program("synthetic_joint", all, n, 0, REPEATS, &nv, &ld);
  double nv_sum = 0, ld_sum = 0;
  for (int i = 0; i < n; i++) {
    char one[128];
    sprintf(one, "extern \"C\" __global__ void t%d(float *x) { x[threadIdx.x] += %d.0f; }\n", i,
            i + 1);
    compiled c = compile_src(one, "gh1529_t.cu", 0);
    nv_sum += c.nvrtc_s;
    ld_sum += load_s(&c);
    free(c.code);
  }
  printf("row synthetic_separate kernels=%d programs=%d nvrtc_sum_s=%.4f load_sum_s=%.4f "
         "per_program_nvrtc_s=%.4f per_program_load_s=%.4f\n",
         n, n, nv_sum, ld_sum, nv_sum / n, ld_sum / n);
  printf("fit synthetic fixed_nvrtc_s=%.4f fixed_load_s=%.4f (separate minus joint, per extra "
         "program)\n",
         (nv_sum - nv) / (n - 1), (ld_sum - ld) / (n - 1));
  fflush(stdout);
  free(all);
}

// Least squares t = a + b k over the subset rows.
static void fit(const char *what, const int *ks, const double *ts, int n) {
  double sk = 0, st = 0, skk = 0, skt = 0;
  for (int i = 0; i < n; i++) {
    sk += ks[i];
    st += ts[i];
    skk += (double)ks[i] * ks[i];
    skt += ks[i] * ts[i];
  }
  double b = (n * skt - sk * st) / (n * skk - sk * sk);
  double a = (st - b * sk) / n;
  printf("fit subsets %s fixed_s=%.4f per_kernel_s=%.5f\n", what, a, b);
  fflush(stdout);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <file.cu> [nvrtc option...]\n", argv[0]);
    return 2;
  }
  split_source(read_file(argv[1]));
  CU_CHECK(cuInit(0));
  CUdevice dev;
  CU_CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU_CHECK(cuCtxSetCurrent(ctx));
  int major, minor, nv_major, nv_minor, drv;
  CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  NVRTC_CHECK(nvrtcVersion(&nv_major, &nv_minor));
  CU_CHECK(cuDriverGetVersion(&drv));
  char devname[256];
  CU_CHECK(cuDeviceGetName(devname, sizeof devname, dev));

  n_opts = argc - 2;
  opts = (const char **)(argv + 2);
  cubin_opts = malloc(sizeof(char *) * (size_t)(n_opts + 1));
  char arch[64];
  snprintf(arch, sizeof arch, "--gpu-architecture=sm_%d%d", major, minor);
  for (int i = 0; i < n_opts; i++)
    if (strncmp(opts[i], "--gpu-architecture", 18) != 0 && strncmp(opts[i], "-arch", 5) != 0)
      cubin_opts[n_cubin_opts++] = opts[i];
  cubin_opts[n_cubin_opts++] = arch;

  const char *cache = getenv("CUDA_CACHE_DISABLE");
  printf("header device=\"%s\" cc=%d.%d nvrtc=%d.%d driver=%d CUDA_CACHE_DISABLE=%s kernels=%d "
         "prelude_bytes=%zu file=%s options=",
         devname, major, minor, nv_major, nv_minor, drv, cache ? cache : "unset", n_kernels,
         strlen(prelude), argv[1]);
  for (int i = 0; i < n_opts; i++) printf("%s%s", i ? "," : "", opts[i]);
  printf("\n");
  fflush(stdout);

  // nvrtc's and the driver's one-time initialization, kept out of every row.
  {
    double t0 = now_s();
    compiled c = compile_src("extern \"C\" __global__ void w(float *x) { x[0] = 1.0f; }\n",
                             "warm.cu", 0);
    load_s(&c);
    free(c.code);
    printf("row warmup first_compile_and_load_s=%.4f\n", now_s() - t0);
  }

  synthetic(64);

  int *all = malloc(sizeof(int) * (size_t)n_kernels);
  for (int i = 0; i < n_kernels; i++) all[i] = i;
  char *joint = program_of(all, n_kernels);
  double joint_nv, joint_ld;
  measure_program("joint", joint, n_kernels, 0, REPEATS, &joint_nv, &joint_ld);

  // Subsets of k evenly spaced kernels.
  int ks[32], nk = 0;
  double nvs[32], lds[32];
  for (int k = 1; k < n_kernels && nk < 30; k *= 2) {
    int *idx = malloc(sizeof(int) * (size_t)k);
    for (int j = 0; j < k; j++) idx[j] = (int)((long)j * n_kernels / k);
    char *src = program_of(idx, k);
    char label[32];
    snprintf(label, sizeof label, "subset_%d", k);
    measure_program(label, src, k, 0, REPEATS, &nvs[nk], &lds[nk]);
    ks[nk++] = k;
    free(src);
    free(idx);
  }
  ks[nk] = n_kernels;
  nvs[nk] = joint_nv;
  lds[nk] = joint_ld;
  nk++;
  fit("nvrtc", ks, nvs, nk);
  fit("load", ks, lds, nk);
  double tot[32];
  for (int i = 0; i < nk; i++) tot[i] = nvs[i] + lds[i];
  fit("total", ks, tot, nk);

  // Twice serially: the second pass's loads find the driver's JIT cache warm unless
  // CUDA_CACHE_DISABLE=1, and a second pass much faster in nvrtc would mean a cache there too.
  measure_perkernel(1, ctx);
  measure_perkernel(1, ctx);
  measure_perkernel(8, ctx);

  {
    double nv, ld;
    measure_program("cubin_joint", joint, n_kernels, 1, REPEATS, &nv, &ld);
    int k = n_kernels < 4 ? n_kernels : 4;
    int idx[4];
    for (int j = 0; j < k; j++) idx[j] = (int)((long)j * n_kernels / k);
    char *src = program_of(idx, k);
    measure_program("cubin_subset_4", src, k, 1, REPEATS, &nv, &ld);
    free(src);
  }
  printf("done\n");
  return 0;
}
