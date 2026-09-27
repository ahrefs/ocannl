/* Exhaustive sweeps of the e5m2 codecs, in C because the loop counts are 2^32 and 17.2e9
   (gh-ocannl-657).

   Every function these call is the one the HOST uses -- [single_to_fp8], [double_to_fp8] and
   [fp8_to_single] as compiled from arrayjit/lib/builtins.c into the [ir] library's stubs, reached
   here by an [extern] declaration rather than a copy, so the sweep cannot drift from the codec it
   verifies. (The same text is what [Builtins_cc] hands the cc backend and what [Builtins_metal]
   mirrors; the CUDA and HIP kernels use the vendor types instead, which is what tools/fp8_soak.ml
   checks on hardware.)

   The oracle for "correctly rounded" is local, and that is what makes it cheap enough to run on
   every input: [fp8_to_single] is strictly increasing over the finite magnitude codes (asserted
   separately, over all 256 codes), so a code is the round-to-nearest-even image of |x| exactly when
   |x| lies in that code's rounding interval -- between the midpoints to its two neighbours, with a
   midpoint belonging to whichever of the two has an even code. The decode table here is built from
   the FORMAT (sign, 5-bit exponent, 2-bit mantissa), not from [fp8_to_single], so a broken decoder
   cannot excuse a broken encoder. Saturation is the one asymmetry: code 0x7B has no upper midpoint,
   because a finite input above the range must land on it rather than on infinity.

   Each entry point sweeps a half-open range of bit patterns and accumulates into a caller-owned
   int64 buffer, so the OCaml side can run several of them on separate domains over disjoint ranges
   and add the results up. The runtime lock is released for the duration: the loop touches no OCaml
   value, and a domain that stayed inside a non-allocating C call would never reach a safepoint for
   another domain's GC. */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

/* From arrayjit/lib/builtins.c, linked into the [ir] library's stub archive. */
extern uint8_t single_to_fp8(float f);
extern uint8_t double_to_fp8(double f);
extern float fp8_to_single(uint8_t fp8);

/* Buffer layout, shared with fp8_codec_exhaustive.ml. */
#define OUT_CROSS 0    /* single_to_fp8 vs double_to_fp8 disagreements */
#define OUT_ROUNDING 1 /* inputs narrowed to something other than the nearest code */
#define OUT_SIGN 2     /* inputs whose sign the codec did not carry over */
#define OUT_OVERFLOW 3 /* finite inputs narrowed to an infinity or NaN code */
#define OUT_SPECIAL 4  /* infinities and NaNs not narrowed to 0x7C / 0x7F */
#define OUT_REACHED 5  /* 4 words: which of the 256 codes some input produced */
#define OUT_REPORTED 9 /* how many offender records follow */
#define OUT_RECORDS 10 /* triples: input bits, produced code, reason */
#define MAX_RECORDS 8
#define OUT_LEN (OUT_RECORDS + (3 * MAX_RECORDS))

#define REASON_CROSS 1
#define REASON_ROUNDING 2
#define REASON_SIGN 3
#define REASON_OVERFLOW 4
#define REASON_SPECIAL 5

/* The e5m2 value of each finite magnitude code, read off the format. Codes 0x7C and above are
   infinity and NaN and have no place in the ordering. */
static double e5m2_decode[0x7C];
static double e5m2_lo_mid[0x7C]; /* midpoint to the code below; unused for code 0 */
static double e5m2_hi_mid[0x7C]; /* midpoint to the code above; unused for code 0x7B */

/* Filled once, from the main domain, before any sweep starts: the sweeps run concurrently on
   several domains with the runtime lock released, so nothing they touch may be initialized lazily. */
static void e5m2_init_tables(void)
{
  int m;
  for (m = 0; m < 0x7C; m++)
  {
    int e = m >> 2;
    int q = m & 3;
    /* Subnormals are q * 2^-16; normals are (4 + q) * 2^(e - 15) / 4. Both exact in a double. */
    e5m2_decode[m] = e == 0 ? ldexp((double)q, -16) : ldexp((double)(4 + q), e - 17);
  }
  for (m = 0; m < 0x7C; m++)
  {
    e5m2_lo_mid[m] = m > 0 ? (e5m2_decode[m - 1] + e5m2_decode[m]) * 0.5 : 0.0;
    e5m2_hi_mid[m] = m < 0x7B ? (e5m2_decode[m] + e5m2_decode[m + 1]) * 0.5 : HUGE_VAL;
  }
}

/* Nonzero when magnitude code [m] is NOT the round-to-nearest-even image of [ax] >= 0. */
static int e5m2_misrounded(double ax, unsigned int m)
{
  if (m > 0x7B)
  {
    return 1; /* a finite magnitude has no business on an infinity or NaN code */
  }
  if (m > 0)
  {
    double lo = e5m2_lo_mid[m];
    if (ax < lo || (ax == lo && (m & 1u)))
    {
      return 1;
    }
  }
  if (m < 0x7B)
  {
    double hi = e5m2_hi_mid[m];
    if (ax > hi || (ax == hi && (m & 1u)))
    {
      return 1;
    }
  }
  return 0;
}

static void record(int64_t *out, int64_t bits, unsigned int code, int64_t reason)
{
  if (out[OUT_REPORTED] < MAX_RECORDS)
  {
    int64_t k = OUT_RECORDS + (3 * out[OUT_REPORTED]);
    out[k] = bits;
    out[k + 1] = (int64_t)code;
    out[k + 2] = reason;
    out[OUT_REPORTED]++;
  }
}

/* One input, already narrowed to [code]: check the sign, the specials, and the rounding.
   [sign] is the input's sign bit, positioned as e5m2's (0x80 or 0). Returns nonzero if anything
   was wrong, so the caller can skip further checks on the same input. */
static int check_narrowing(int64_t *out, int64_t bits, double d, unsigned int sign, uint8_t code)
{
  unsigned int m = code & 0x7Fu;
  out[OUT_REACHED + (code >> 6)] |= (int64_t)((uint64_t)1 << (code & 63));
  if (((unsigned int)code & 0x80u) != sign)
  {
    out[OUT_SIGN]++;
    record(out, bits, code, REASON_SIGN);
    return 1;
  }
  if (isnan(d))
  {
    if (m != 0x7Fu)
    {
      out[OUT_SPECIAL]++;
      record(out, bits, code, REASON_SPECIAL);
      return 1;
    }
    return 0;
  }
  if (isinf(d))
  {
    if (m != 0x7Cu)
    {
      out[OUT_SPECIAL]++;
      record(out, bits, code, REASON_SPECIAL);
      return 1;
    }
    return 0;
  }
  if (m >= 0x7Cu)
  {
    out[OUT_OVERFLOW]++;
    record(out, bits, code, REASON_OVERFLOW);
    return 1;
  }
  if (e5m2_misrounded(fabs(d), m))
  {
    out[OUT_ROUNDING]++;
    record(out, bits, code, REASON_ROUNDING);
    return 1;
  }
  return 0;
}

/* All f32 bit patterns in [base, base + count): [single_to_fp8] against the oracle, and
   [double_to_fp8] of the same value widened to a double -- which is the "all 2^32 f32-exact
   doubles" cross-check, the one that found a NaN-sign divergence between the two codecs. */
static void sweep_f32(uint64_t base, uint64_t count, int64_t *out)
{
  uint64_t i;
  for (i = 0; i < count; i++)
  {
    uint32_t u = (uint32_t)(base + i);
    float x;
    double d;
    uint8_t cs, cd;
    memcpy(&x, &u, sizeof(x));
    d = (double)x;
    cs = single_to_fp8(x);
    cd = double_to_fp8(d);
    if (cs != cd)
    {
      out[OUT_CROSS]++;
      record(out, (int64_t)u, ((uint32_t)cs << 8) | (uint32_t)cd, REASON_CROSS);
      continue;
    }
    (void)check_narrowing(out, (int64_t)u, d, (unsigned int)((u >> 31) << 7), cs);
  }
}

/* [double_to_fp8] over doubles that are NOT f32-exact: each top half in [base, base + count)
   crossed with four low halves -- zero, one ulp up, the mantissa's midpoint bit, and all ones.
   The midpoint pattern is the point of the exercise: gh-ocannl-648 was a double just off an f32
   tie being rounded onto it and then away by the wrong rule. */
static const uint32_t low_halves[4] = { 0u, 1u, 0x80000000u, 0xFFFFFFFFu };

static void sweep_f64(uint64_t base, uint64_t count, int64_t *out)
{
  uint64_t i;
  int k;
  for (i = 0; i < count; i++)
  {
    uint32_t hi = (uint32_t)(base + i);
    for (k = 0; k < 4; k++)
    {
      uint64_t u = ((uint64_t)hi << 32) | (uint64_t)low_halves[k];
      double d;
      uint8_t c;
      memcpy(&d, &u, sizeof(d));
      c = double_to_fp8(d);
      (void)check_narrowing(out, (int64_t)u, d, (unsigned int)((u >> 56) & 0x80u), c);
    }
  }
}

/* The OCaml side of this file is the only caller and passes the right shapes -- but an [external]
   is a hole in the type system, and a C function that trusts a length it was not given is one
   refactor away from reading past a heap block. So every entry point below checks what it is about
   to rely on, in the C, where the reliance is. Raised before the blocking section is entered, while
   the runtime lock is still held. */
static void require(int condition, const char *message)
{
  if (!condition)
  {
    caml_invalid_argument(message);
  }
}

static intnat ba_length(value v)
{
  return Caml_ba_array_val(v)->dim[0];
}

static int64_t *checked_out(value v_out, const char *where)
{
  require(ba_length(v_out) >= OUT_LEN, where);
  return (int64_t *)Caml_ba_data_val(v_out);
}

static void check_range(value v_base, value v_count, const char *where)
{
  require(Int64_val(v_base) >= 0 && Int64_val(v_count) >= 0, where);
}

/* Must be called once, from the main domain, before any sweep. */
CAMLprim value ocannl_fp8_sweep_init(value v_unit)
{
  CAMLparam1(v_unit);
  e5m2_init_tables();
  CAMLreturn(Val_unit);
}

CAMLprim value ocannl_fp8_sweep_f32(value v_base, value v_count, value v_out)
{
  CAMLparam3(v_base, v_count, v_out);
  uint64_t base, count;
  int64_t *out;
  check_range(v_base, v_count, "ocannl_fp8_sweep_f32: base and count must be non-negative");
  out = checked_out(v_out, "ocannl_fp8_sweep_f32: the counters buffer is too short");
  base = (uint64_t)Int64_val(v_base);
  count = (uint64_t)Int64_val(v_count);
  caml_enter_blocking_section();
  sweep_f32(base, count, out);
  caml_leave_blocking_section();
  CAMLreturn(Val_unit);
}

CAMLprim value ocannl_fp8_sweep_f64(value v_base, value v_count, value v_out)
{
  CAMLparam3(v_base, v_count, v_out);
  uint64_t base, count;
  int64_t *out;
  check_range(v_base, v_count, "ocannl_fp8_sweep_f64: base and count must be non-negative");
  out = checked_out(v_out, "ocannl_fp8_sweep_f64: the counters buffer is too short");
  base = (uint64_t)Int64_val(v_base);
  count = (uint64_t)Int64_val(v_count);
  caml_enter_blocking_section();
  sweep_f64(base, count, out);
  caml_leave_blocking_section();
  CAMLreturn(Val_unit);
}

/* The e5m2 value of a magnitude code, straight from the format -- the oracle's decode, exposed so
   that the OCaml side can hold [fp8_to_single] to it over all 256 codes. Codes 0x7C and up have no
   finite value; the caller does not ask for them. */
CAMLprim value ocannl_fp8_reference_decode(value v_code)
{
  CAMLparam1(v_code);
  /* The table holds the FINITE magnitudes, 0x00 to 0x7B. Masking with 0x7F -- which is what this
     line did -- lets 0x7C through 0x7F index one to four doubles past its end: infinity and the
     NaN payloads have no finite value to look up, and the caller asks about them separately. */
  require(Int_val(v_code) >= 0 && Int_val(v_code) < 0x7C,
          "ocannl_fp8_reference_decode: not a finite e5m2 magnitude code");
  CAMLreturn(caml_copy_double(e5m2_decode[Int_val(v_code)]));
}
