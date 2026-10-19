/* Is RDNA3's WMMA exactly rounded on exact data?  Only when every product has one sign.
 * gh-ocannl-1420; the recorded verdict lives in docs/agent-notes/backend-precision-and-simd.md
 * (#simd-operations-and-rounding) and test/operations/fragment_scope_move_legality.ml.
 *
 * Standalone on purpose: it calls the gfx11 wave32 WMMA builtins directly -- no rocWMMA, no
 * OCANNL, no hiprtc -- so it answers a question about the INSTRUCTION, and it stays runnable on a
 * box where the OCaml switch is not set up.  Build and run (gfx11 only: gfx12 has other builtins):
 *
 *   hipcc -std=c++17 -O3 --offload-arch=$(rocminfo | grep -m1 -o 'gfx[0-9a-f]*') \
 *     tools/hip_wmma_sign_probe.hip -o /tmp/hip_wmma_sign_probe && /tmp/hip_wmma_sign_probe
 *
 * Its EXIT STATUS is the answer the tests rest on: 0 only if every SAME-SIGN case returned the
 * exact sum bitwise on both accumulator widths -- which is what lets a HIP tensorized leg make a
 * bitwise claim against a serial twin.  The mixed-sign cases are reported, not judged: on gfx1151
 * and gfx1102 (ROCm 7.1, at -O0 and -O3 alike) they are NOT exact, and a future target where they
 * come back exact is news for the agent note, not a failure.
 *
 * How it decides.  One 16x16 output tile over k = 16 * steps, the steps chained through the
 * accumulator operand exactly as a fragment scope chains its k blocks.  The operands are small
 * integers, so every product and every partial sum is exact in f32 (and, where noted, in f16):
 * IEEE arithmetic in ANY association returns the host's exact sum.  Measured on gfx1151 and
 * gfx1102 alike: the smallest mixed-sign case, 1*1 + 1*(-1), returns -2^-24 on both widths; the
 * issue's operands (i % 5 - 2 against i % 3 - 1) miss in 241 of 256 f32 cells (1 comes back as
 * 1 - 2^-24), and the f16 arm, whose rounding absorbs residues that small everywhere else, misses
 * in exactly its 38 exact-zero cells (subnormal multiples of 2^-24, either sign); the same
 * operands shifted to be nonnegative, or negated to be all nonpositive, are exact on both widths,
 * as are nonnegative sums up to 53760 in f32. */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

/* gfx11 wave32 layouts: lane l holds row l % 16 of A and column l % 16 of B (lanes 16-31
   replicate 0-15); accumulator element e of lane l is row 2e + l / 16, column l % 16 (the f16
   accumulator in the low half of each 32-bit register, op_sel = false). a is 16 x k, b is k x 16. */
__global__ void tile(const _Float16 *a, const _Float16 *b, int steps, float *d32, _Float16 *d16) {
  int lane = threadIdx.x, k = 16 * steps;
  v8f c32 = {};
  v16h c16 = {};
  for (int s = 0; s < steps; ++s) {
    v16h af, bf;
    for (int e = 0; e < 16; ++e) {
      af[e] = a[k * (lane % 16) + 16 * s + e];
      bf[e] = b[16 * (16 * s + e) + lane % 16];
    }
    c32 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, c32);
    c16 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(af, bf, c16, false);
  }
  for (int e = 0; e < 8; ++e) {
    int r = 2 * e + lane / 16, c = lane % 16;
    d32[16 * r + c] = c32[e];
    d16[16 * r + c] = c16[2 * e];
  }
}

struct Case {
  const char *name;
  bool same_sign;
  bool f16_exact; /* every partial sum is exact in f16 too */
  int steps;
  float (*a)(int r, int k);
  float (*b)(int k, int c);
};

static const Case cases[] = {
    {"1*1 + 1*(-1)", false, true, 1, [](int, int k) { return k < 2 ? 1.f : 0.f; },
     [](int k, int) { return k == 0 ? 1.f : (k == 1 ? -1.f : 0.f); }},
    {"i%5-2 by i%3-1, k=64", false, true, 4,
     [](int r, int k) { return (float)((64 * r + k) % 5) - 2.f; },
     [](int k, int c) { return (float)((16 * k + c) % 3) - 1.f; }},
    {"i%5 by i%3, k=64", true, true, 4, [](int r, int k) { return (float)((64 * r + k) % 5); },
     [](int k, int c) { return (float)((16 * k + c) % 3); }},
    {"-(i%5) by i%3, k=64", true, true, 4,
     [](int r, int k) { return -(float)((64 * r + k) % 5); },
     [](int k, int c) { return (float)((16 * k + c) % 3); }},
    {"i%31 by i%29, k=64 (sums to 53760)", true, false, 4,
     [](int r, int k) { return (float)((64 * r + k) % 31); },
     [](int k, int c) { return (float)((16 * k + c) % 29); }},
};

int main() {
  const int kmax = 64;
  _Float16 *da, *db, *d16;
  float *d32;
  if (hipMalloc(&da, 16 * kmax * 2) || hipMalloc(&db, 16 * kmax * 2) || hipMalloc(&d16, 512) ||
      hipMalloc(&d32, 1024)) {
    fprintf(stderr, "hipMalloc failed\n");
    return 2;
  }
  int status = 0;
  for (const Case &cs : cases) {
    int k = 16 * cs.steps;
    _Float16 a[16 * kmax], b[16 * kmax];
    float fa[16 * kmax], fb[16 * kmax];
    for (int r = 0; r < 16; ++r)
      for (int kk = 0; kk < k; ++kk) {
        fa[k * r + kk] = cs.a(r, kk);
        a[k * r + kk] = (_Float16)fa[k * r + kk];
        fb[16 * kk + r] = cs.b(kk, r);
        b[16 * kk + r] = (_Float16)fb[16 * kk + r];
      }
    (void)hipMemcpy(da, a, 16 * k * 2, hipMemcpyHostToDevice);
    (void)hipMemcpy(db, b, 16 * k * 2, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(tile, dim3(1), dim3(32), 0, 0, da, db, cs.steps, d32, d16);
    float r32[256];
    _Float16 r16[256];
    if (hipMemcpy(r32, d32, 1024, hipMemcpyDeviceToHost) ||
        hipMemcpy(r16, d16, 512, hipMemcpyDeviceToHost)) {
      fprintf(stderr, "kernel or copy failed\n");
      return 2;
    }
    int wrong32 = 0, wrong16 = 0, zeros = 0, first = -1;
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < 16; ++j) {
        float s = 0;
        for (int kk = 0; kk < k; ++kk) s += fa[k * i + kk] * fb[16 * kk + j];
        float g32 = r32[16 * i + j];
        unsigned b32, bs;
        std::memcpy(&b32, &g32, 4);
        std::memcpy(&bs, &s, 4);
        unsigned short b16, bh;
        _Float16 sh = (_Float16)s;
        std::memcpy(&b16, &r16[16 * i + j], 2);
        std::memcpy(&bh, &sh, 2);
        zeros += s == 0.f;
        if (b32 != bs) { ++wrong32; if (first < 0) first = 16 * i + j; }
        if (b16 != bh) ++wrong16;
      }
    bool exact = wrong32 == 0 && (!cs.f16_exact || wrong16 == 0);
    printf("%-36s %s: f32 accumulator %3d of 256 cells off the exact sum, f16 accumulator %3d%s; "
           "%d exact-zero cells",
           cs.name, cs.same_sign ? "same-sign " : "mixed-sign", wrong32, wrong16,
           cs.f16_exact ? "" : " (sums not f16-exact: not judged)", zeros);
    if (first >= 0) {
      float s = 0;
      for (int kk = 0; kk < k; ++kk) s += fa[k * (first / 16) + kk] * fb[16 * kk + first % 16];
      printf("; first f32 miss: exact %a, got %a", s, r32[first]);
    }
    printf("%s\n", cs.same_sign ? (exact ? "  [exact: OK]" : "  [NOT EXACT: the rule broke]")
                                : (exact ? "  [exact: news, see the note]" : ""));
    if (cs.same_sign && !exact) status = 1;
  }
  return status;
}
