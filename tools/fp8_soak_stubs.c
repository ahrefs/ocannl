/* Host half of the GPU fp8 soak (gh-ocannl-657): OCANNL's software e5m2 codec, run over the same
   inputs the device kernel narrowed with the vendor type, and compared code by code.

   The codec here is not a transcription: [single_to_fp8] and [double_to_fp8] are the functions
   arrayjit/lib/builtins.c compiles into the [ir] library, the same object code every host-side fp8
   write goes through and the same text [Builtins_cc] hands the cc backend and [Builtins_metal]
   mirrors. That is the comparison the soak exists to make -- CUDA and HIP kernels do NOT use the
   software codec, they cast to the vendor type, so a tensor whose cells are written on the host and
   read by a kernel has the two side by side inside it.

   Disagreements are classified by the INPUT, because two of the classes are known and permanent:
   CUDA saturates an already-infinite input to the largest finite where our codec keeps it infinite,
   and CUDA drops the sign of a NaN where our codec keeps it (HIP does neither). Only the finite
   class is required to be empty. */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

/* From arrayjit/lib/builtins.c, linked into the [ir] library's stub archive. */
extern uint8_t single_to_fp8(float f);
extern uint8_t double_to_fp8(double f);

/* Buffer layout, shared with fp8_soak.ml. */
#define S_FINITE 0   /* disagreements on a finite input -- the ones that must be zero */
#define S_INF 1      /* disagreements on an infinite input */
#define S_NAN 2      /* disagreements on a NaN input */
#define S_INF_SEEN 3 /* infinite inputs swept */
#define S_NAN_SEEN 4 /* NaN inputs swept */
#define S_INF_CODES 5  /* 4 words: vendor codes seen on infinite inputs */
#define S_NAN_CODES 9  /* 4 words: vendor codes seen on NaN inputs */
#define S_ALL_CODES 13 /* 4 words: every vendor code seen, over all inputs */
/* Finite disagreements the HIP guard would NOT close: those at |x| >= 2^-17, i.e. outside the range
   [ocannl_single_to_fp8_uniform] / [ocannl_double_to_fp8_uniform] pre-round to a signed zero. The
   whole claim that guard rests on is that this stays 0 -- everything below half the smallest e5m2
   subnormal rounds to zero anyway, so clamping there is exact, and a disagreement ABOVE it would be
   one the guard silently fails to fix. */
#define S_UNGUARDED 17
/* 32 words: the set of biased exponent FIELDS of the inputs that disagreed on a finite value --
   8-bit for the f32 sweep, 11-bit for the f64 one, hence 2048 bits. This is what localizes a
   vendor defect to a window instead of to a count (gh-ocannl-647), and what lets a run assert the
   window without the tool having to keep every offending input. */
#define S_DIS_EXPS 18
#define S_REPORTED 50
#define S_RECORDS 51 /* triples: input bits, our code, the vendor's */
#define S_MAX_RECORDS 8
#define S_LEN (S_RECORDS + (3 * S_MAX_RECORDS))

/* Half the smallest e5m2 subnormal (2^-17). Below this magnitude every correctly rounded narrowing
   answers a signed zero, which is why the HIP guard's clamp is exact rather than an approximation;
   it is also the boundary this file classifies disagreements against. */
#define OCANNL_FP8_GUARD_THRESHOLD 7.62939453125e-6

static void mark(int64_t *out, int slot, unsigned int code)
{
  out[slot + (code >> 6)] |= (int64_t)((uint64_t)1 << (code & 63));
}

static void record(int64_t *out, int64_t bits, unsigned int ours, unsigned int theirs)
{
  if (out[S_REPORTED] < S_MAX_RECORDS)
  {
    int64_t k = S_RECORDS + (3 * out[S_REPORTED]);
    out[k] = bits;
    out[k + 1] = (int64_t)ours;
    out[k + 2] = (int64_t)theirs;
    out[S_REPORTED]++;
  }
}

static void soak_f32(uint64_t base, uint64_t count, const uint8_t *theirs, int64_t *out)
{
  uint64_t i;
  for (i = 0; i < count; i++)
  {
    uint32_t u = (uint32_t)(base + i);
    uint32_t e32 = (u >> 23) & 0xFFu;
    float x;
    uint8_t ours, them;
    memcpy(&x, &u, sizeof(x));
    ours = single_to_fp8(x);
    them = theirs[i];
    mark(out, S_ALL_CODES, them);
    if (e32 == 0xFFu)
    {
      int is_nan = (u & 0x7FFFFFu) != 0u;
      mark(out, is_nan ? S_NAN_CODES : S_INF_CODES, them);
      out[is_nan ? S_NAN_SEEN : S_INF_SEEN]++;
      if (ours != them)
      {
        out[is_nan ? S_NAN : S_INF]++;
      }
      continue;
    }
    if (ours != them)
    {
      out[S_FINITE]++;
      mark(out, S_DIS_EXPS, e32);
      if (!(fabsf(x) < (float)OCANNL_FP8_GUARD_THRESHOLD))
      {
        out[S_UNGUARDED]++;
      }
      record(out, (int64_t)u, ours, them);
    }
  }
}

static void soak_f64(uint64_t base, uint64_t count, const uint32_t *lows, const uint8_t *theirs,
                     int64_t *out)
{
  uint64_t i;
  int k;
  for (i = 0; i < count; i++)
  {
    uint32_t hi = (uint32_t)(base + i);
    uint32_t e64 = (hi >> 20) & 0x7FFu;
    for (k = 0; k < 4; k++)
    {
      uint64_t u = ((uint64_t)hi << 32) | (uint64_t)lows[k];
      double d;
      uint8_t ours, them;
      memcpy(&d, &u, sizeof(d));
      ours = double_to_fp8(d);
      them = theirs[(4 * i) + (uint64_t)k];
      mark(out, S_ALL_CODES, them);
      if (e64 == 0x7FFu)
      {
        int is_nan = (u & 0xFFFFFFFFFFFFFULL) != 0ULL;
        mark(out, is_nan ? S_NAN_CODES : S_INF_CODES, them);
        out[is_nan ? S_NAN_SEEN : S_INF_SEEN]++;
        if (ours != them)
        {
          out[is_nan ? S_NAN : S_INF]++;
        }
        continue;
      }
      if (ours != them)
      {
        out[S_FINITE]++;
        mark(out, S_DIS_EXPS, e64);
        if (!(fabs(d) < OCANNL_FP8_GUARD_THRESHOLD))
        {
          out[S_UNGUARDED]++;
        }
        record(out, (int64_t)u, ours, them);
      }
    }
  }
}

/* An [external] is a hole in the type system: nothing in the OCaml types says how long the vendor
   buffer is, how many words the counters need, or that the low-halves array has four elements. This
   file is the only place those facts can be enforced, so it enforces all of them, before the
   blocking section and while the runtime lock is still held. */
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

CAMLprim value ocannl_fp8_soak_f32(value v_base, value v_count, value v_theirs, value v_out)
{
  CAMLparam4(v_base, v_count, v_theirs, v_out);
  uint64_t base, count;
  const uint8_t *theirs;
  int64_t *out;
  require(Int64_val(v_base) >= 0 && Int64_val(v_count) >= 0,
          "ocannl_fp8_soak_f32: base and count must be non-negative");
  require(ba_length(v_theirs) >= Int64_val(v_count),
          "ocannl_fp8_soak_f32: the vendor buffer is shorter than the sweep");
  require(ba_length(v_out) >= S_LEN, "ocannl_fp8_soak_f32: the counters buffer is too short");
  base = (uint64_t)Int64_val(v_base);
  count = (uint64_t)Int64_val(v_count);
  theirs = (const uint8_t *)Caml_ba_data_val(v_theirs);
  out = (int64_t *)Caml_ba_data_val(v_out);
  caml_enter_blocking_section();
  soak_f32(base, count, theirs, out);
  caml_leave_blocking_section();
  CAMLreturn(Val_unit);
}

CAMLprim value ocannl_fp8_soak_f64(value v_base, value v_count, value v_lows, value v_theirs,
                                   value v_out)
{
  CAMLparam5(v_base, v_count, v_lows, v_theirs, v_out);
  uint64_t base, count;
  const uint8_t *theirs;
  int64_t *out;
  uint32_t lows[4];
  int k;
  require(Int64_val(v_base) >= 0 && Int64_val(v_count) >= 0,
          "ocannl_fp8_soak_f64: base and count must be non-negative");
  /* Four low halves, and exactly four: the kernel takes them as four scalar parameters, so a
     shorter array would be read past its heap block and a longer one would silently sweep less
     than the caller asked for. */
  require(Wosize_val(v_lows) == 4, "ocannl_fp8_soak_f64: expected exactly four low halves");
  require(ba_length(v_theirs) >= 4 * Int64_val(v_count),
          "ocannl_fp8_soak_f64: the vendor buffer is shorter than four bytes per top half");
  require(ba_length(v_out) >= S_LEN, "ocannl_fp8_soak_f64: the counters buffer is too short");
  base = (uint64_t)Int64_val(v_base);
  count = (uint64_t)Int64_val(v_count);
  theirs = (const uint8_t *)Caml_ba_data_val(v_theirs);
  out = (int64_t *)Caml_ba_data_val(v_out);
  for (k = 0; k < 4; k++)
  {
    lows[k] = (uint32_t)Long_val(Field(v_lows, k));
  }
  caml_enter_blocking_section();
  soak_f64(base, count, lows, theirs, out);
  caml_leave_blocking_section();
  CAMLreturn(Val_unit);
}
