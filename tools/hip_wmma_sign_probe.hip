/* Is RDNA3's WMMA exactly rounded on exact data?  Only when every product has one sign.
 * gh-ocannl-1420 (the f16 builtins), gh-ocannl-1509 (the bf16 builtins and the rounding cases);
 * the recorded verdict lives in docs/agent-notes/backend-precision-and-simd.md
 * (#simd-operations-and-rounding) and test/operations/fragment_scope_move_legality.ml.
 *
 * Standalone on purpose: it calls the gfx11 wave32 WMMA builtins directly -- no rocWMMA, no
 * OCANNL, no hiprtc -- so it answers a question about the INSTRUCTION, and it stays runnable on a
 * box where the OCaml switch is not set up.  Build and run (gfx11 only: gfx12 has other builtins):
 *
 *   hipcc -std=c++17 -O3 --offload-arch=$(rocminfo | grep -m1 -o 'gfx[0-9a-f]*') \
 *     tools/hip_wmma_sign_probe.hip -o /tmp/hip_wmma_sign_probe && /tmp/hip_wmma_sign_probe
 *
 * It names the device it ran on, on stderr.  It runs the four gfx11 builtins on every case:
 * f16 x f16 -> f32, f16 x f16 -> f16, bf16 x bf16 -> f32 and bf16 x bf16 -> bf16.  Its EXIT STATUS
 * is the answer the tests rest on: 0 only if every SAME-SIGN case returned the exact sum bitwise on
 * every arm whose partial sums are all exact at its accumulator width -- which is what lets a HIP
 * tensorized leg make a bitwise claim against a serial twin.  The mixed-sign cases are reported,
 * not judged: on gfx1151 and gfx1102 (ROCm 7.1, at -O0 and -O3 alike) they are NOT exact, and a
 * future target where they come back exact is news for the agent note, not a failure.  So are the
 * rounding cases: each names the rounding it finds at one accumulator width, beside the one
 * recorded in [arms] -- a change is news, not a failure (no test rests on inexact sums: see the
 * note's contraction-licence bullet).
 *
 * How it decides.  One 16x16 output tile over k = 16 * steps, the steps chained through the
 * accumulator operand exactly as a fragment scope chains its k blocks.  The operands are small
 * integers, exact in f16 and in bf16, so every product is an exact integer, and a partial sum of a
 * cell is an integer no larger in magnitude than the sum of that cell's |products|: when that
 * bound fits the width's significand (2^24 in f32, 2^11 in f16, 2^8 in bf16), IEEE arithmetic in
 * ANY association returns the host's exact sum.  A rounding case takes the other route: each cell
 * has two nonzero products, both exact at the width, so their sum is the one rounding, and the
 * cells' residues tell the modes apart (see [cases]).
 *
 * Measured on gfx1151 (Radeon 8060S, unified memory; ROCm 7.1, clang 21; -O0 and -O3 alike).
 * The f16 arms (gh-ocannl-1420, gfx1102 alike): the smallest mixed-sign case, 1*1 + 1*(-1),
 * returns -2^-24 on both widths; the issue's operands (i % 5 - 2 against i % 3 - 1) miss in 241
 * of 256 f32 cells (1 comes back as 1 - 2^-24), and the f16 arm, whose rounding absorbs residues
 * that small everywhere else, misses in exactly its 38 exact-zero cells (subnormal multiples of
 * 2^-24, either sign); the same operands shifted to be nonnegative, or negated to be all
 * nonpositive, are exact on both widths, as are nonnegative sums up to 53760 in f32.  The bf16
 * arms (gh-ocannl-1509) obey the same rule: every same-sign case exact on both widths, the
 * mixed-sign ones off -- 1*1 + 1*(-1) gives -2^-24 on both, the issue's operands miss in 241 f32
 * cells as f16's do and in 226 bf16 cells, where 1 comes back as 1 - 2^-8: the f32 residue cut
 * toward zero.  Rounding: the f16 accumulator rounds to nearest even; the bf16 accumulator rounds
 * TOWARD ZERO (515 -> 512, 518 -> 516, where nearest even says 516 and 520); and both f32
 * accumulators round as an adder keeping one guard bit and no sticky bit would -- 2^25 + 3, whose
 * dropped low bit would have put it past the tie, comes back 2^25 where nearest even says
 * 2^25 + 4, while 2^25 + 7 comes back 2^25 + 8 as it should. */
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
/* gfx11's bf16 builtins take and return bf16 BIT PATTERNS in i16 vectors. */
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

enum Width { F32, F16, BF16 };
enum Mode { NEAREST_EVEN, NEAREST_AWAY, TOWARD_ZERO, NO_STICKY, N_MODES };
static const char *mode_names[] = {"to nearest even", "to nearest, ties away", "toward zero",
                                   "to nearest even, no sticky bit"};

struct Arm {
  const char *name;
  Width acc;
  Mode recorded; /* the rounding a rounding case measured (gfx1151, ROCm 7.1): see the note */
};

/* The four builtins, in the order the kernel stores them. */
static const Arm arms[] = {{"f16->f32", F32, NO_STICKY},
                           {"f16->f16", F16, NEAREST_EVEN},
                           {"bf16->f32", F32, NO_STICKY},
                           {"bf16->bf16", BF16, TOWARD_ZERO}};
static const int n_arms = sizeof(arms) / sizeof(arms[0]);

/* Every arm's 256 output cells, as the raw bits of its accumulator width. */
struct Out {
  float h32[256];
  _Float16 h16[256];
  float b32[256];
  unsigned short b16[256];
};

/* gfx11 wave32 layouts: lane l holds row l % 16 of A and column l % 16 of B (lanes 16-31
   replicate 0-15); accumulator element e of lane l is row 2e + l / 16, column l % 16 (a 16-bit
   accumulator in the low half of each 32-bit register, op_sel = false). a is 16 x k, b is k x 16;
   ah/bh hold them in f16, ab/bb as bf16 bit patterns. */
__global__ void tile(const _Float16 *ah, const _Float16 *bh, const short *ab, const short *bb,
                     int steps, Out *d) {
  int lane = threadIdx.x, k = 16 * steps;
  v8f h32 = {}, b32 = {};
  v16h h16 = {};
  v16s b16 = {};
  for (int s = 0; s < steps; ++s) {
    v16h ahf, bhf;
    v16s abf, bbf;
    for (int e = 0; e < 16; ++e) {
      int ia = k * (lane % 16) + 16 * s + e, ib = 16 * (16 * s + e) + lane % 16;
      ahf[e] = ah[ia];
      bhf[e] = bh[ib];
      abf[e] = ab[ia];
      bbf[e] = bb[ib];
    }
    h32 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ahf, bhf, h32);
    h16 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(ahf, bhf, h16, false);
    b32 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(abf, bbf, b32);
    b16 = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(abf, bbf, b16, false);
  }
  for (int e = 0; e < 8; ++e) {
    int i = 16 * (2 * e + lane / 16) + lane % 16;
    d->h32[i] = h32[e];
    d->h16[i] = h16[2 * e];
    d->b32[i] = b32[e];
    d->b16[i] = (unsigned short)b16[2 * e];
  }
}

/* x rounded once to bf16, to nearest even, as bits: x finite and within f32's normal range. */
static unsigned short bf16_bits(double x) {
  uint64_t u;
  std::memcpy(&u, &x, 8);
  u += (1ull << 44) - 1 + ((u >> 45) & 1); /* keep 7 of the 52 fraction bits */
  u &= ~((1ull << 45) - 1);
  double y;
  std::memcpy(&y, &u, 8);
  float f = (float)y; /* exact: y has 8 significant bits */
  uint32_t v;
  std::memcpy(&v, &f, 4);
  return (unsigned short)(v >> 16);
}

/* x rounded once to the width, to nearest even, as bits. */
static uint32_t bits_of(Width w, double x) {
  switch (w) {
  case F32: {
    float f = (float)x;
    uint32_t v;
    std::memcpy(&v, &f, 4);
    return v;
  }
  case F16: {
    _Float16 h = (_Float16)x;
    unsigned short v;
    std::memcpy(&v, &h, 2);
    return v;
  }
  case BF16:
    return bf16_bits(x);
  }
  return 0;
}

static double value_of(Width w, uint32_t v) {
  switch (w) {
  case F32: {
    float f;
    std::memcpy(&f, &v, 4);
    return f;
  }
  case F16: {
    unsigned short s = (unsigned short)v;
    _Float16 h;
    std::memcpy(&h, &s, 2);
    return (double)h;
  }
  case BF16: {
    uint32_t u = v << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }
  }
  return 0;
}

static bool exact_in(Width w, double x) { return value_of(w, bits_of(w, x)) == x; }

/* Significand bits, the implicit one included: integers up to 2^this are exact at the width. */
static int sig_bits(Width w) { return w == F32 ? 24 : w == F16 ? 11 : 8; }

/* x rounded to the width under the mode, as bits: the nearest-even neighbour, stepped one unit
   toward zero where the mode says so (the bits are sign-magnitude, so - 1 is toward zero).
   NO_STICKY is an adder that keeps one bit below the result's last place and drops the rest (no
   sticky bit) before rounding to nearest even: x truncated to sig_bits + 1 bits, then rounded. */
static uint32_t round_bits(Width w, Mode m, double x) {
  if (m == NO_STICKY && x != 0) {
    int e;
    std::frexp(x, &e);
    double guard = std::ldexp(1.0, e - sig_bits(w) - 1);
    return bits_of(w, std::trunc(x / guard) * guard);
  }
  uint32_t ne = bits_of(w, x);
  double v = value_of(w, ne);
  if (v == x || m == NEAREST_EVEN) return ne;
  uint32_t lo = std::fabs(v) > std::fabs(x) ? ne - 1 : ne, hi = lo + 1;
  if (m == TOWARD_ZERO) return lo;
  bool tie = std::fabs(x) - std::fabs(value_of(w, lo)) == std::fabs(value_of(w, hi)) - std::fabs(x);
  return tie ? hi : ne;
}


static const int EXACT = -1;

struct Case {
  const char *name;
  int steps;
  float (*a)(int r, int k);
  float (*b)(int k, int c);
  int rounds_at; /* EXACT, or the Width whose rounding this case classifies */
};

/* A rounding case at width w: base + (r+1)(c+1), base = 2^(p+1) for w's p significand bits, so the
   sums sit where w's spacing is 4 and the residues mod 8 split them into exact cells, cells nearer
   one neighbour (above or below), and ties that round down or up to the even neighbour: each mode
   gives a different answer on some cell. */
static const Case cases[] = {
    {"1*1 + 1*(-1)", 1, [](int, int k) { return k < 2 ? 1.f : 0.f; },
     [](int k, int) { return k == 0 ? 1.f : (k == 1 ? -1.f : 0.f); }, EXACT},
    {"i%5-2 by i%3-1, k=64", 4, [](int r, int k) { return (float)((64 * r + k) % 5) - 2.f; },
     [](int k, int c) { return (float)((16 * k + c) % 3) - 1.f; }, EXACT},
    {"i%5 by i%3, k=64", 4, [](int r, int k) { return (float)((64 * r + k) % 5); },
     [](int k, int c) { return (float)((16 * k + c) % 3); }, EXACT},
    {"-(i%5) by i%3, k=64", 4, [](int r, int k) { return -(float)((64 * r + k) % 5); },
     [](int k, int c) { return (float)((16 * k + c) % 3); }, EXACT},
    {"i%31 by i%29, k=64 (sums to 53760)", 4,
     [](int r, int k) { return (float)((64 * r + k) % 31); },
     [](int k, int c) { return (float)((16 * k + c) % 29); }, EXACT},
    {"rounding at f32: 2^25 + (r+1)(c+1)", 1,
     [](int r, int k) { return k == 0 ? 8192.f : (k == 1 ? (float)(r + 1) : 0.f); },
     [](int k, int c) { return k == 0 ? 4096.f : (k == 1 ? (float)(c + 1) : 0.f); }, F32},
    {"rounding at f16: 2^12 + (r+1)(c+1)", 1,
     [](int r, int k) { return k == 0 ? 64.f : (k == 1 ? (float)(r + 1) : 0.f); },
     [](int k, int c) { return k == 0 ? 64.f : (k == 1 ? (float)(c + 1) : 0.f); }, F16},
    {"rounding at bf16: 2^9 + (r+1)(c+1)", 1,
     [](int r, int k) { return k == 0 ? 32.f : (k == 1 ? (float)(r + 1) : 0.f); },
     [](int k, int c) { return k == 0 ? 16.f : (k == 1 ? (float)(c + 1) : 0.f); }, BF16},
};

int main() {
  const int kmax = 64;
  hipDeviceProp_t prop;
  int runtime = 0;
  if (hipGetDeviceProperties(&prop, 0) || hipRuntimeGetVersion(&runtime)) {
    fprintf(stderr, "hip_wmma_sign_probe: no HIP device\n");
    return 2;
  }
  fprintf(stderr, "hip_wmma_sign_probe: backend hip, device 0: %s (%s), HIP runtime %d\n",
          prop.name, prop.gcnArchName, runtime);
  _Float16 *dah, *dbh;
  short *dab, *dbb;
  Out *dout;
  if (hipMalloc(&dah, 16 * kmax * 2) || hipMalloc(&dbh, 16 * kmax * 2) ||
      hipMalloc(&dab, 16 * kmax * 2) || hipMalloc(&dbb, 16 * kmax * 2) ||
      hipMalloc(&dout, sizeof(Out))) {
    fprintf(stderr, "hipMalloc failed\n");
    return 2;
  }
  int status = 0;
  bool footnote = false;
  printf("Per arm: the cells of 256 off the exact sum, or the rounding a rounding case finds.\n");
  for (const Case &cs : cases) {
    int k = 16 * cs.steps;
    _Float16 ah[16 * kmax], bh[16 * kmax];
    short ab[16 * kmax], bb[16 * kmax];
    double fa[16 * kmax], fb[16 * kmax];
    for (int r = 0; r < 16; ++r)
      for (int kk = 0; kk < k; ++kk) {
        fa[k * r + kk] = cs.a(r, kk);
        fb[16 * kk + r] = cs.b(kk, r);
      }
    for (int i = 0; i < 16 * k; ++i) {
      ah[i] = (_Float16)fa[i];
      bh[i] = (_Float16)fb[i];
      ab[i] = (short)bf16_bits(fa[i]);
      bb[i] = (short)bf16_bits(fb[i]);
      if (!(exact_in(F16, fa[i]) && exact_in(F16, fb[i]) && exact_in(BF16, fa[i]) &&
            exact_in(BF16, fb[i]))) {
        fprintf(stderr, "%s: an operand is not exact in f16 and bf16 -- the probe's own bug\n",
                cs.name);
        return 2;
      }
    }
    /* Per cell: the exact sum (in doubles: these integers are far inside 2^53), the bound on its
       partial sums, its nonzero products, and whether each is exact at each width.  Per width:
       exact[w] when every partial sum of every cell is exact; one_rounding[w] when at most the
       last one rounds and the result is finite. */
    double sum[256];
    bool exact[3] = {true, true, true}, one_rounding[3] = {true, true, true};
    bool has_pos = false, has_neg = false;
    for (int c = 0; c < 256; ++c) {
      double s = 0, bound = 0;
      int nonzero = 0;
      bool prod_exact[3] = {true, true, true};
      for (int kk = 0; kk < k; ++kk) {
        double p = fa[k * (c / 16) + kk] * fb[16 * kk + c % 16];
        s += p;
        bound += std::fabs(p);
        nonzero += p != 0;
        has_pos = has_pos || p > 0;
        has_neg = has_neg || p < 0;
        for (int w = 0; w < 3; ++w) prod_exact[w] = prod_exact[w] && exact_in((Width)w, p);
      }
      sum[c] = s;
      for (int w = 0; w < 3; ++w) {
        bool fits = bound <= std::ldexp(1.0, sig_bits((Width)w));
        exact[w] = exact[w] && fits;
        one_rounding[w] = one_rounding[w] && (fits || (nonzero <= 2 && prod_exact[w])) &&
                          std::isfinite(value_of((Width)w, bits_of((Width)w, s)));
      }
    }
    if (cs.rounds_at != EXACT && !one_rounding[cs.rounds_at]) {
      fprintf(stderr, "%s: more than one rounding at its width -- the probe's own bug\n", cs.name);
      return 2;
    }
    bool same_sign = !(has_pos && has_neg);
    (void)hipMemcpy(dah, ah, 16 * k * 2, hipMemcpyHostToDevice);
    (void)hipMemcpy(dbh, bh, 16 * k * 2, hipMemcpyHostToDevice);
    (void)hipMemcpy(dab, ab, 16 * k * 2, hipMemcpyHostToDevice);
    (void)hipMemcpy(dbb, bb, 16 * k * 2, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(tile, dim3(1), dim3(32), 0, 0, dah, dbh, dab, dbb, cs.steps, dout);
    Out out;
    if (hipMemcpy(&out, dout, sizeof(Out), hipMemcpyDeviceToHost)) {
      fprintf(stderr, "kernel or copy failed\n");
      return 2;
    }
    int zeros = 0;
    for (int c = 0; c < 256; ++c) zeros += sum[c] == 0;
    printf("%-36s %s:", cs.name, same_sign ? "same-sign " : "mixed-sign");
    bool all_exact = true, news = false;
    int exact_arms = 0; /* a case with none states no exactness verdict */
    char detail[n_arms][160];
    for (int a = 0; a < n_arms; ++a) {
      Width w = arms[a].acc;
      uint32_t got[256];
      for (int c = 0; c < 256; ++c) switch (a) {
        case 0: std::memcpy(&got[c], &out.h32[c], 4); break;
        case 1: { unsigned short v; std::memcpy(&v, &out.h16[c], 2); got[c] = v; break; }
        case 2: std::memcpy(&got[c], &out.b32[c], 4); break;
        default: got[c] = out.b16[c]; break;
        }
      int off[N_MODES] = {}, first = -1;
      for (int c = 0; c < 256; ++c) {
        for (int m = 0; m < N_MODES; ++m) off[m] += got[c] != round_bits(w, (Mode)m, sum[c]);
        if (first < 0 && got[c] != bits_of(w, sum[c])) first = c;
      }
      detail[a][0] = 0;
      if (first >= 0)
        snprintf(detail[a], sizeof detail[a],
                 "\n    first %s cell off nearest-even: %d, exact %a, nearest-even %a, got %a",
                 arms[a].name, first, sum[first], value_of(w, bits_of(w, sum[first])),
                 value_of(w, got[first]));
      if (exact[w]) {
        /* Every partial sum is exact: the unit owes the exact sum. */
        printf(" %s %3d", arms[a].name, off[NEAREST_EVEN]);
        ++exact_arms;
        if (off[NEAREST_EVEN]) all_exact = false;
      } else if (cs.rounds_at == (int)w) {
        int mode = -1;
        for (int m = N_MODES - 1; m >= 0; --m)
          if (!off[m]) mode = m;
        printf(" %s rounds %s", arms[a].name,
               mode < 0 ? "in no mode named here" : mode_names[mode]);
        if (mode != (int)arms[a].recorded) {
          printf(" (recorded: %s)", mode_names[arms[a].recorded]);
          news = true;
        } else
          detail[a][0] = 0;
      } else {
        printf(" %s %3d*", arms[a].name, off[NEAREST_EVEN]);
        footnote = true;
        detail[a][0] = 0;
      }
    }
    if (zeros) printf("; %d exact-zero cells", zeros);
    printf("%s%s", !exact_arms ? ""
                   : same_sign ? (all_exact ? "  [exact: OK]" : "  [NOT EXACT: the rule broke]")
                               : (all_exact ? "  [exact: news, see the note]" : ""),
           news ? "  [rounding: news, see the note]" : "");
    for (int a = 0; a < n_arms; ++a) printf("%s", detail[a]);
    printf("\n");
    if (same_sign && !all_exact) status = 1;
  }
  if (footnote)
    printf("* inexact at that width, and not the width this case rounds at: counted, not judged\n");
  return status;
}
