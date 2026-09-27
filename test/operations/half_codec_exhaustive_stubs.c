/* Exhaustive sweeps of the emulated fp16 converters, in C because the narrowing loop count is 2^32
   (gh-ocannl-985).

   [float_to_half_emulated] and [half_to_float_emulated] are what a compiler without [_Float16]
   gets, in cc kernels and in the host stubs alike. Every fleet box has [_Float16], so nothing
   shipped reaches them there; half_emulated.h is generated from the builtins definition table with
   the native switch forced off, so the functions below are the table's own text, compiled.

   Narrowing is checked against a rounding ORACLE, the one fp8_codec_exhaustive_stubs.c uses: the
   decode table is built from the FORMAT (sign, 5-bit exponent, 10-bit mantissa), not from either
   converter, and a code is the round-to-nearest-even image of |x| exactly when |x| lies between
   the midpoints to its two neighbours, a midpoint belonging to whichever neighbour has an even
   code. Unlike e5m2 there is no saturation: IEEE narrowing overflows to infinity, so infinity takes
   part in the ordering as the code after 0x7BFF with the value 2^16 -- which puts the overflow
   threshold at 65520, a tie that goes to infinity because 0x7C00 is the even neighbour. Every half
   value and every midpoint is exact in a double, as is every f32 input.

   Where the compiler has [_Float16] (tested with the compiler's own [__FLT16_MAX__], which the
   header's forced switch does not touch), each input is also narrowed by the native cast, and the
   two must agree bit for bit -- on a NaN, up to the payload, which neither IEEE nor this converter
   ties to the input's.

   The sweep entry point takes a half-open range of bit patterns and accumulates into a caller-owned
   int64 buffer, so the OCaml side can run it on several domains over disjoint ranges and add the
   results up. The runtime lock is released for the duration. */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include "half_emulated.h"

#if HAS_NATIVE_FLOAT16
#error "half_emulated.h must force the emulated converters"
#endif

#ifdef __FLT16_MAX__
#define SWEEP_HAS_NATIVE 1
#else
#define SWEEP_HAS_NATIVE 0
#endif

/* Buffer layout, shared with half_codec_exhaustive.ml. */
#define OUT_NATIVE 0   /* inputs the native cast narrows differently */
#define OUT_ROUNDING 1 /* finite inputs narrowed to something other than the nearest code */
#define OUT_SIGN 2     /* inputs whose sign the converter did not carry over */
#define OUT_SPECIAL 3  /* infinities and NaNs not narrowed to 0x7C00 / 0x7E00 */
#define OUT_REPORTED 4 /* how many offender records follow */
#define OUT_RECORDS 5  /* triples: input bits, produced code, reason */
#define MAX_RECORDS 8
#define OUT_REACHED (OUT_RECORDS + (3 * MAX_RECORDS)) /* 1024 words: which codes some input produced */
#define REACHED_WORDS 1024
#define OUT_LEN (OUT_REACHED + REACHED_WORDS)

#define REASON_NATIVE 1
#define REASON_ROUNDING 2
#define REASON_SIGN 3
#define REASON_SPECIAL 4

#define INF_CODE 0x7C00u
#define QNAN_CODE 0x7E00u

/* The half value of each magnitude code up to and including infinity, read off the format;
   infinity's entry is 2^16, the value the format would give it with one more exponent. */
static double half_decode[INF_CODE + 1];
static double half_lo_mid[INF_CODE + 1]; /* midpoint to the code below; unused for code 0 */
static double half_hi_mid[INF_CODE];     /* midpoint to the code above */

/* Filled once, from the main domain, before any sweep starts. */
static void half_init_tables(void)
{
  unsigned int m;
  for (m = 0; m <= INF_CODE; m++)
  {
    int e = (int)(m >> 10);
    int q = (int)(m & 0x3FFu);
    /* Subnormals are q * 2^-24; normals are (1024 + q) * 2^(e - 25). Both exact in a double. */
    half_decode[m] = e == 0 ? ldexp((double)q, -24) : ldexp((double)(1024 + q), e - 25);
  }
  for (m = 0; m <= INF_CODE; m++)
  {
    half_lo_mid[m] = m > 0 ? (half_decode[m - 1] + half_decode[m]) * 0.5 : 0.0;
    if (m < INF_CODE)
    {
      half_hi_mid[m] = (half_decode[m] + half_decode[m + 1]) * 0.5;
    }
  }
}

/* Nonzero when magnitude code [m] is NOT the round-to-nearest-even image of finite [ax] >= 0. */
static int half_misrounded(double ax, unsigned int m)
{
  if (m > INF_CODE)
  {
    return 1; /* a finite input has no business on a NaN code */
  }
  if (m > 0)
  {
    double lo = half_lo_mid[m];
    if (ax < lo || (ax == lo && (m & 1u)))
    {
      return 1;
    }
  }
  if (m < INF_CODE)
  {
    double hi = half_hi_mid[m];
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

static int is_nan_code(unsigned int h)
{
  return (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu) != 0;
}

#if SWEEP_HAS_NATIVE
static unsigned int native_narrow(float x)
{
  _Float16 h = (_Float16)x;
  uint16_t bits;
  memcpy(&bits, &h, sizeof(bits));
  return bits;
}
#endif

/* All f32 bit patterns in [base, base + count). */
static void sweep_narrow(uint64_t base, uint64_t count, int64_t *out)
{
  uint64_t i;
  for (i = 0; i < count; i++)
  {
    uint32_t u = (uint32_t)(base + i);
    float x;
    double d;
    unsigned int h, m, sign;
    memcpy(&x, &u, sizeof(x));
    d = (double)x;
    h = float_to_half_emulated(x);
    m = h & 0x7FFFu;
    sign = (u >> 16) & 0x8000u;
    out[OUT_REACHED + (h >> 6)] |= (int64_t)1 << (h & 63u);
#if SWEEP_HAS_NATIVE
    {
      unsigned int n = native_narrow(x);
      int agree = isnan(d) ? (is_nan_code(n) && (n & 0x8000u) == (h & 0x8000u)) : n == h;
      if (!agree)
      {
        out[OUT_NATIVE]++;
        record(out, (int64_t)u, (n << 16) | h, REASON_NATIVE);
      }
    }
#endif
    if ((h & 0x8000u) != sign)
    {
      out[OUT_SIGN]++;
      record(out, (int64_t)u, h, REASON_SIGN);
      continue;
    }
    if (isnan(d) || isinf(d))
    {
      if (m != (isnan(d) ? QNAN_CODE : INF_CODE))
      {
        out[OUT_SPECIAL]++;
        record(out, (int64_t)u, h, REASON_SPECIAL);
      }
      continue;
    }
    if (half_misrounded(fabs(d), m))
    {
      out[OUT_ROUNDING]++;
      record(out, (int64_t)u, h, REASON_ROUNDING);
    }
  }
}

/* An [external] is a hole in the type system: every entry point checks what it relies on, before
   the blocking section is entered, while the runtime lock is still held. */
static void require(int condition, const char *message)
{
  if (!condition)
  {
    caml_invalid_argument(message);
  }
}

/* Must be called once, from the main domain, before any sweep. */
CAMLprim value ocannl_half_sweep_init(value v_unit)
{
  CAMLparam1(v_unit);
  half_init_tables();
  CAMLreturn(Val_unit);
}

CAMLprim value ocannl_half_sweep_narrow(value v_base, value v_count, value v_out)
{
  CAMLparam3(v_base, v_count, v_out);
  uint64_t base, count;
  int64_t *out;
  require(Int64_val(v_base) >= 0 && Int64_val(v_count) >= 0,
          "ocannl_half_sweep_narrow: base and count must be non-negative");
  require(Caml_ba_array_val(v_out)->dim[0] >= OUT_LEN,
          "ocannl_half_sweep_narrow: the counters buffer is too short");
  out = (int64_t *)Caml_ba_data_val(v_out);
  base = (uint64_t)Int64_val(v_base);
  count = (uint64_t)Int64_val(v_count);
  caml_enter_blocking_section();
  sweep_narrow(base, count, out);
  caml_leave_blocking_section();
  CAMLreturn(Val_unit);
}

CAMLprim value ocannl_half_has_native(value v_unit)
{
  CAMLparam1(v_unit);
  CAMLreturn(Val_bool(SWEEP_HAS_NATIVE));
}

static unsigned int checked_code(value v_code, const char *where)
{
  require(Int_val(v_code) >= 0 && Int_val(v_code) <= 0xFFFF, where);
  return (unsigned int)Int_val(v_code);
}

CAMLprim value ocannl_half_emulated_widen(value v_code)
{
  CAMLparam1(v_code);
  unsigned int h = checked_code(v_code, "ocannl_half_emulated_widen: not a 16-bit code");
  CAMLreturn(caml_copy_double((double)half_to_float_emulated((uint16_t)h)));
}

CAMLprim value ocannl_half_emulated_narrow(value v_x)
{
  CAMLparam1(v_x);
  CAMLreturn(Val_int(float_to_half_emulated((float)Double_val(v_x))));
}

/* The native widening, or a refusal where the compiler has no [_Float16]: the OCaml side asks
   [ocannl_half_has_native] first. */
CAMLprim value ocannl_half_native_widen(value v_code)
{
  CAMLparam1(v_code);
  unsigned int h = checked_code(v_code, "ocannl_half_native_widen: not a 16-bit code");
#if SWEEP_HAS_NATIVE
  {
    uint16_t bits = (uint16_t)h;
    _Float16 v;
    memcpy(&v, &bits, sizeof(v));
    CAMLreturn(caml_copy_double((double)(float)v));
  }
#else
  (void)h;
  caml_failwith("ocannl_half_native_widen: the compiler has no _Float16");
#endif
}

/* The half value of a finite magnitude code, straight from the format: the oracle's decode,
   exposed so the OCaml side can hold [half_to_float_emulated] to it over all 65536 codes. */
CAMLprim value ocannl_half_reference_decode(value v_code)
{
  CAMLparam1(v_code);
  require(Int_val(v_code) >= 0 && Int_val(v_code) < (int)INF_CODE,
          "ocannl_half_reference_decode: not a finite half magnitude code");
  CAMLreturn(caml_copy_double(half_decode[Int_val(v_code)]));
}
