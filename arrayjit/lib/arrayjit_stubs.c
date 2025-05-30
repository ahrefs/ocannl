#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <math.h>
#include <stdint.h>

/* Pure C conversion functions for use in C backends */

/* BFloat16 to Float conversion (C function) */
static inline float bfloat16_to_float(uint16_t bf16)
{
  /* BFloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
     To convert to float32, we shift left by 16 bits */
  uint32_t f32 = ((uint32_t)bf16) << 16;
  return *((float*)&f32);
}

/* Float to BFloat16 conversion (C function) */
static inline uint16_t float_to_bfloat16(float f)
{
  uint32_t f32 = *((uint32_t*)&f);
  
  /* Round to nearest even */
  uint32_t rounded = f32 + 0x7FFF + ((f32 >> 16) & 1);
  return (uint16_t)(rounded >> 16);
}

/* FP8 E5M2 format to Float conversion (C function)
   Format: 1 sign bit, 5 exponent bits, 2 mantissa bits */
static inline float fp8_to_float(uint8_t fp8)
{
  /* Handle zero */
  if (fp8 == 0) {
    return 0.0f;
  }
  
  uint32_t sign = (fp8 >> 7) & 1;
  uint32_t exp = (fp8 >> 2) & 0x1F;
  uint32_t mant = fp8 & 0x3;
  
  /* Handle special cases */
  if (exp == 0x1F) {  /* Infinity or NaN */
    if (mant == 0) {
      return sign ? -INFINITY : INFINITY;
    } else {
      return NAN;
    }
  }
  
  /* Denormalized numbers */
  if (exp == 0) {
    float result = ldexpf((float)mant / 4.0f, -14);
    if (sign) result = -result;
    return result;
  }
  
  /* Normalized numbers */
  float result = (1.0f + (float)mant * 0.25f) * ldexpf(1.0f, (int)exp - 15);
  if (sign) result = -result;
  
  return result;
}

/* Float to FP8 E5M2 conversion (C function) */
static inline uint8_t float_to_fp8(float f)
{
  /* Handle zero */
  if (f == 0.0f) {
    return 0;
  }
  
  uint32_t sign = (f < 0) ? 1 : 0;
  f = fabsf(f);
  
  /* Handle special cases */
  if (isinf(f)) {
    return (sign << 7) | 0x7C;  /* Infinity: exp=0x1F, mant=0 */
  }
  if (isnan(f)) {
    return (sign << 7) | 0x7F;  /* NaN: exp=0x1F, mant!=0 */
  }
  
  /* Get exponent and mantissa */
  int exp_val;
  float mant_f = frexpf(f, &exp_val);
  int exp = exp_val + 14;  /* Bias is 15, but frexp gives us mantissa in [0.5, 1) */
  
  /* Clamp to representable range */
  if (exp < 0) {
    /* Underflow to zero */
    return sign << 7;
  }
  if (exp > 30) {
    /* Overflow to infinity */
    return (sign << 7) | 0x7C;
  }
  
  /* Handle denormalized numbers */
  if (exp == 0) {
    float denorm_mant = f * ldexpf(1.0f, 14) * 4.0f;
    uint32_t mant_bits = (uint32_t)(denorm_mant + 0.5f);
    if (mant_bits > 3) mant_bits = 3;
    return (sign << 7) | mant_bits;
  }
  
  /* Normalized numbers: convert mantissa from [0.5, 1) to [0, 0.75] */
  mant_f = (mant_f - 0.5f) * 4.0f;
  uint32_t mant_bits = (uint32_t)(mant_f + 0.5f);  /* Round to nearest */
  if (mant_bits > 3) mant_bits = 3;
  
  return (uint8_t)((sign << 7) | ((exp & 0x1F) << 2) | (mant_bits & 0x3));
}

/* OCaml wrapper functions */

/* BFloat16 to Float conversion (OCaml wrapper) */
CAMLprim value arrayjit_bfloat16_to_float(value v_bf16)
{
  CAMLparam1(v_bf16);
  uint16_t bf16 = (uint16_t)Int_val(v_bf16);
  float result = bfloat16_to_float(bf16);
  CAMLreturn(caml_copy_double((double)result));
}

/* Float to BFloat16 conversion (OCaml wrapper) */
CAMLprim value arrayjit_float_to_bfloat16(value v_float) 
{
  CAMLparam1(v_float);
  float f = (float)Double_val(v_float);
  uint16_t bf16 = float_to_bfloat16(f);
  CAMLreturn(Val_int(bf16));
}

/* FP8 E5M2 format to Float conversion (OCaml wrapper) */
CAMLprim value arrayjit_fp8_to_float(value v_fp8)
{
  CAMLparam1(v_fp8);
  uint8_t fp8 = (uint8_t)Int_val(v_fp8);
  float result = fp8_to_float(fp8);
  CAMLreturn(caml_copy_double((double)result));
}

/* Float to FP8 E5M2 conversion (OCaml wrapper) */
CAMLprim value arrayjit_float_to_fp8(value v_float)
{
  CAMLparam1(v_float);
  float f = (float)Double_val(v_float);
  uint8_t fp8 = float_to_fp8(f);
  CAMLreturn(Val_int(fp8));
} 