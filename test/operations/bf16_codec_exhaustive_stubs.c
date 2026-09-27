/* Exhaustive sweeps of the bf16 converters, in C because the narrowing loop count is 2^32
   (gh-ocannl-1069).

   [single_to_bfloat16] and the vector bridge [OCANNL_VEC_NARROW_BFLOAT16] are what every cc kernel
   storing a bf16 node runs, and the scalar one is also what the host stubs run on every backend;
   bf16_codec.h is generated from the builtins definition table, so the functions and macros below
   are the table's own text, compiled.

   Narrowing is checked against a rounding ORACLE of the same kind as half_codec_exhaustive's: the
   decode table is built from the FORMAT (sign, 8-bit exponent, 7-bit mantissa), not from either
   converter, and a code is the round-to-nearest-even image of |x| exactly when |x| lies between
   the midpoints to its two neighbours, a midpoint belonging to whichever neighbour has an even
   code. Infinity takes part in the ordering as the code after 0x7F7F with the value 2^128, so the
   overflow threshold is the tie at 0x7F7F8000, which goes to infinity because 0x7F80 is the even
   neighbour. Every bf16 value and every midpoint is exact in a double, as is every f32 input.

   A NaN has no nearest code, so it is held to one image: quieted and truncated,
   (bits >> 16) | 0x0040 -- its sign and the top of its payload kept, the quiet bit set, which is
   what clang's [__bf16] cast gives on AArch64. Where the compiler has a [__bf16] type, the native
   cast is also compared against, bit for bit on every non-NaN input and on a NaN up to the payload
   (a NaN on both sides, of one sign).

   The vector bridge is swept on the same inputs, eight lanes at a time with a partial last vector
   where the range is not a multiple of eight, and must agree with the scalar converter bit for
   bit. Where the compiler has no [__builtin_convertvector] the bridge IS a per-lane call of the
   scalar converter, so the sweep reports which arm it compiled.

   The negative control: the pre-gh-1069 rounding arithmetic, transcribed below, is swept too. It
   carries 131072 NaN inputs out of the NaN class, and it agrees with the converter under test on
   every non-NaN input -- which is the fix's claim that it changes only NaNs.

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

/* The executable also links arrayjit's host stubs (through test_utils), which export these two
   converters compiled from the same table. Renaming the header's definitions keeps the symbols
   apart, and makes certain that the functions swept are the header's text compiled here -- not the
   library's copy, which the linker could otherwise resolve to. */
#define bfloat16_to_single sweep_bfloat16_to_single
#define single_to_bfloat16 sweep_single_to_bfloat16
#include "bf16_codec.h"

/* GCC defines __BFLT16_MAX__ where it has [__bf16] (13 and later); clang does not define it, and
   its [__bf16] is an arithmetic type on x86-64 and AArch64 from 17 on. */
#if defined(__BFLT16_MAX__) \
  || (defined(__clang__) && __clang_major__ >= 17 && (defined(__aarch64__) || defined(__x86_64__)))
#define SWEEP_HAS_NATIVE 1
#else
#define SWEEP_HAS_NATIVE 0
#endif

#define SWEEP_LANES 8
typedef unsigned short sweep_u16v __attribute__((vector_size(2 * SWEEP_LANES)));
typedef uint32_t sweep_u32v __attribute__((vector_size(4 * SWEEP_LANES)));
typedef float sweep_f32v __attribute__((vector_size(4 * SWEEP_LANES)));

/* Buffer layout, shared with bf16_codec_exhaustive.ml. */
#define OUT_NATIVE 0          /* inputs the native cast narrows differently */
#define OUT_VECTOR 1          /* inputs the vector bridge narrows differently from the scalar */
#define OUT_ROUNDING 2        /* finite inputs narrowed to something other than the nearest code */
#define OUT_SIGN 3            /* inputs whose sign the converter did not carry over */
#define OUT_INF 4             /* infinite inputs not narrowed to 0x7F80 */
#define OUT_NAN_ESCAPE 5      /* NaN inputs narrowed to a non-NaN code */
#define OUT_NAN_CODE 6        /* NaN inputs narrowed to a NaN other than their quieted image */
#define OUT_CONTROL_ESCAPE 7  /* NaN inputs the pre-fix arithmetic narrows to a non-NaN code */
#define OUT_CONTROL_FINITE 8  /* non-NaN inputs where the pre-fix arithmetic differs */
#define OUT_NATIVE_PAYLOAD 9  /* NaN inputs where the native cast keeps a different payload */
#define OUT_REPORTED 10       /* how many offender records follow */
#define OUT_RECORDS 11        /* triples: input bits, produced code, reason */
#define MAX_RECORDS 8
#define OUT_REACHED (OUT_RECORDS + (3 * MAX_RECORDS)) /* 1024 words: which codes some input produced */
#define REACHED_WORDS 1024
#define OUT_LEN (OUT_REACHED + REACHED_WORDS)

#define REASON_NATIVE 1
#define REASON_VECTOR 2
#define REASON_ROUNDING 3
#define REASON_SIGN 4
#define REASON_INF 5
#define REASON_NAN_ESCAPE 6
#define REASON_NAN_CODE 7

#define INF_CODE 0x7F80u

/* The bf16 value of each magnitude code up to and including infinity, read off the format;
   infinity's entry is 2^128, the value the format would give it were the exponent not reserved. */
static double bf16_decode[INF_CODE + 1];
static double bf16_lo_mid[INF_CODE + 1]; /* midpoint to the code below; unused for code 0 */
static double bf16_hi_mid[INF_CODE];     /* midpoint to the code above */

/* Filled once, from the main domain, before any sweep starts. */
static void bf16_init_tables(void)
{
  unsigned int m;
  for (m = 0; m <= INF_CODE; m++)
  {
    int e = (int)(m >> 7);
    int q = (int)(m & 0x7Fu);
    /* Subnormals are q * 2^-133; normals are (128 + q) * 2^(e - 134). Both exact in a double. */
    bf16_decode[m] = e == 0 ? ldexp((double)q, -133) : ldexp((double)(128 + q), e - 134);
  }
  for (m = 0; m <= INF_CODE; m++)
  {
    bf16_lo_mid[m] = m > 0 ? (bf16_decode[m - 1] + bf16_decode[m]) * 0.5 : 0.0;
    if (m < INF_CODE)
    {
      bf16_hi_mid[m] = (bf16_decode[m] + bf16_decode[m + 1]) * 0.5;
    }
  }
}

/* Nonzero when magnitude code [m] is NOT the round-to-nearest-even image of finite [ax] >= 0. */
static int bf16_misrounded(double ax, unsigned int m)
{
  if (m > INF_CODE)
  {
    return 1; /* a finite input has no business on a NaN code */
  }
  if (m > 0)
  {
    double lo = bf16_lo_mid[m];
    if (ax < lo || (ax == lo && (m & 1u)))
    {
      return 1;
    }
  }
  if (m < INF_CODE)
  {
    double hi = bf16_hi_mid[m];
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

static int is_nan_code(unsigned int c)
{
  return (c & 0x7FFFu) > INF_CODE;
}

static int is_nan_bits(uint32_t u)
{
  return (u & 0x7FFFFFFFu) > 0x7F800000u;
}

/* The pre-gh-1069 [single_to_bfloat16], transcribed: the round-to-nearest-even bit-add with no NaN
   guard. Kept only as the sweep's negative control. */
static unsigned int prefix_narrow(uint32_t u)
{
  return (unsigned int)(uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

#if SWEEP_HAS_NATIVE
static unsigned int native_narrow(float x)
{
  __bf16 b = (__bf16)x;
  uint16_t bits;
  memcpy(&bits, &b, sizeof(bits));
  return bits;
}
#endif

/* Checks one input against the oracle; [c] is what the scalar converter produced. */
static void check_one(uint32_t u, float x, unsigned int c, int64_t *out)
{
  double d = (double)x;
  unsigned int m = c & 0x7FFFu;
  unsigned int sign = (u >> 16) & 0x8000u;
  unsigned int control = prefix_narrow(u);
  out[OUT_REACHED + (c >> 6)] |= (int64_t)((uint64_t)1 << (c & 63u));
  if (is_nan_bits(u))
  {
    if (!is_nan_code(control))
    {
      out[OUT_CONTROL_ESCAPE]++;
    }
  }
  else if (control != c)
  {
    out[OUT_CONTROL_FINITE]++;
  }
#if SWEEP_HAS_NATIVE
  {
    unsigned int n = native_narrow(x);
    int agree = is_nan_bits(u)
                    ? (is_nan_code(n) && is_nan_code(c) && (n & 0x8000u) == (c & 0x8000u))
                    : n == c;
    if (!agree)
    {
      out[OUT_NATIVE]++;
      record(out, (int64_t)u, (n << 16) | c, REASON_NATIVE);
    }
    else if (n != c)
    {
      out[OUT_NATIVE_PAYLOAD]++;
    }
  }
#endif
  /* Counted independently of the class checks below, so that a NaN carried into the sign bit
     (0x7FFF8000 -> 0x8000) is both a sign loss and a NaN escape: the escape count is then the whole
     of the NaN class's losses, whichever way each one left it. */
  if ((c & 0x8000u) != sign)
  {
    out[OUT_SIGN]++;
    record(out, (int64_t)u, c, REASON_SIGN);
  }
  if (is_nan_bits(u))
  {
    if (!is_nan_code(c))
    {
      out[OUT_NAN_ESCAPE]++;
      record(out, (int64_t)u, c, REASON_NAN_ESCAPE);
    }
    else if (c != ((u >> 16) | 0x0040u))
    {
      out[OUT_NAN_CODE]++;
      record(out, (int64_t)u, c, REASON_NAN_CODE);
    }
    return;
  }
  if (isinf(d))
  {
    if (m != INF_CODE)
    {
      out[OUT_INF]++;
      record(out, (int64_t)u, c, REASON_INF);
    }
    return;
  }
  if ((c & 0x8000u) == sign && bf16_misrounded(fabs(d), m))
  {
    out[OUT_ROUNDING]++;
    record(out, (int64_t)u, c, REASON_ROUNDING);
  }
}

/* All f32 bit patterns in [base, base + count), a vector's worth at a time. */
static void sweep_narrow(uint64_t base, uint64_t count, int64_t *out)
{
  uint64_t i;
  for (i = 0; i < count; i += SWEEP_LANES)
  {
    int lanes = count - i < SWEEP_LANES ? (int)(count - i) : SWEEP_LANES;
    uint32_t us[SWEEP_LANES] = {0};
    unsigned short vec_codes[SWEEP_LANES] = {0};
    sweep_f32v xv;
    int l;
    for (l = 0; l < lanes; l++)
    {
      us[l] = (uint32_t)(base + i + (uint64_t)l);
    }
    memcpy(&xv, us, sizeof(xv));
    OCANNL_VEC_NARROW_BFLOAT16(sweep_u16v, sweep_u32v, lanes, vec_codes, xv);
    for (l = 0; l < lanes; l++)
    {
      float x;
      unsigned int c;
      memcpy(&x, &us[l], sizeof(x));
      c = single_to_bfloat16(x);
      if (vec_codes[l] != c)
      {
        out[OUT_VECTOR]++;
        record(out, (int64_t)us[l], ((unsigned int)vec_codes[l] << 16) | c, REASON_VECTOR);
      }
      check_one(us[l], x, c, out);
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
CAMLprim value ocannl_bf16_sweep_init(value v_unit)
{
  CAMLparam1(v_unit);
  bf16_init_tables();
  CAMLreturn(Val_unit);
}

CAMLprim value ocannl_bf16_sweep_narrow(value v_base, value v_count, value v_out)
{
  CAMLparam3(v_base, v_count, v_out);
  uint64_t base, count;
  int64_t *out;
  require(Int64_val(v_base) >= 0 && Int64_val(v_count) >= 0,
          "ocannl_bf16_sweep_narrow: base and count must be non-negative");
  require(Int64_val(v_base) <= ((int64_t)1 << 32)
            && Int64_val(v_count) <= ((int64_t)1 << 32) - Int64_val(v_base),
          "ocannl_bf16_sweep_narrow: the range must lie within the 2^32 f32 bit patterns");
  require(Caml_ba_array_val(v_out)->dim[0] >= OUT_LEN,
          "ocannl_bf16_sweep_narrow: the counters buffer is too short");
  out = (int64_t *)Caml_ba_data_val(v_out);
  base = (uint64_t)Int64_val(v_base);
  count = (uint64_t)Int64_val(v_count);
  caml_enter_blocking_section();
  sweep_narrow(base, count, out);
  caml_leave_blocking_section();
  CAMLreturn(Val_unit);
}

CAMLprim value ocannl_bf16_has_native(value v_unit)
{
  CAMLparam1(v_unit);
  CAMLreturn(Val_bool(SWEEP_HAS_NATIVE));
}

CAMLprim value ocannl_bf16_has_convertvector(value v_unit)
{
  CAMLparam1(v_unit);
  CAMLreturn(Val_bool(OCANNL_HAS_CONVERTVECTOR));
}

static unsigned int checked_code(value v_code, const char *where)
{
  require(Int_val(v_code) >= 0 && Int_val(v_code) <= 0xFFFF, where);
  return (unsigned int)Int_val(v_code);
}

/* The f32 bit pattern [bfloat16_to_single] widens a code to, as bits: an OCaml float would carry
   the result through a float-to-double conversion, which quiets a signaling NaN. */
CAMLprim value ocannl_bf16_widen_bits(value v_code)
{
  CAMLparam1(v_code);
  unsigned int c = checked_code(v_code, "ocannl_bf16_widen_bits: not a 16-bit code");
  float f = bfloat16_to_single((uint16_t)c);
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  CAMLreturn(caml_copy_int64((int64_t)u));
}

/* The widened value, for the landmarks the golden prints (finite codes only there). */
CAMLprim value ocannl_bf16_widen(value v_code)
{
  CAMLparam1(v_code);
  unsigned int c = checked_code(v_code, "ocannl_bf16_widen: not a 16-bit code");
  CAMLreturn(caml_copy_double((double)bfloat16_to_single((uint16_t)c)));
}

/* [single_to_bfloat16] of an f32 given by its bit pattern, for the same reason as above. */
CAMLprim value ocannl_bf16_narrow_bits(value v_bits)
{
  CAMLparam1(v_bits);
  int64_t b = Int64_val(v_bits);
  uint32_t u;
  float x;
  require(b >= 0 && b <= 0xFFFFFFFFll, "ocannl_bf16_narrow_bits: not a 32-bit pattern");
  u = (uint32_t)b;
  memcpy(&x, &u, sizeof(x));
  CAMLreturn(Val_int(single_to_bfloat16(x)));
}

/* The bf16 value of a finite magnitude code, straight from the format: the oracle's decode,
   exposed so the OCaml side can hold [bfloat16_to_single] to it over all 65536 codes. */
CAMLprim value ocannl_bf16_reference_decode(value v_code)
{
  CAMLparam1(v_code);
  require(Int_val(v_code) >= 0 && Int_val(v_code) < (int)INF_CODE,
          "ocannl_bf16_reference_decode: not a finite bf16 magnitude code");
  CAMLreturn(caml_copy_double(bf16_decode[Int_val(v_code)]));
}
