#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/bigarray.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* Threefry4x32 types and implementation */

typedef struct {
    uint32_t v[4];
} uint4x32_t;

/* Threefry4x32 constants */
const uint32_t THREEFRY_C240 = 0x1BD11BDA;

/* Rotation constants for Threefry4x32 */
const unsigned int THREEFRY_ROTATION_0_0 = 13;
const unsigned int THREEFRY_ROTATION_0_1 = 15;
const unsigned int THREEFRY_ROTATION_0_2 = 26;
const unsigned int THREEFRY_ROTATION_0_3 = 6;
const unsigned int THREEFRY_ROTATION_1_0 = 17;
const unsigned int THREEFRY_ROTATION_1_1 = 29;
const unsigned int THREEFRY_ROTATION_1_2 = 16;
const unsigned int THREEFRY_ROTATION_1_3 = 24;

/* Rotate left function */
uint32_t rotl32(uint32_t x, unsigned int n) {
    return (x << n) | (x >> (32 - n));
}

/* Threefry4x32 round function */
void threefry_round(uint32_t x[4], unsigned int r0, unsigned int r1, unsigned int r2, unsigned int r3) {
    x[0] += x[1]; x[1] = rotl32(x[1], r0); x[1] ^= x[0];
    x[2] += x[3]; x[3] = rotl32(x[3], r1); x[3] ^= x[2];
    
    uint32_t tmp = x[1];
    x[1] = x[3];
    x[3] = tmp;
    
    x[0] += x[1]; x[1] = rotl32(x[1], r2); x[1] ^= x[0];
    x[2] += x[3]; x[3] = rotl32(x[3], r3); x[3] ^= x[2];
    
    tmp = x[1];
    x[1] = x[3];
    x[3] = tmp;
}

/* Threefry4x32 implementation - 20 rounds */
extern uint4x32_t arrayjit_threefry4x32(uint4x32_t key, uint4x32_t counter) {
    uint32_t x[4];
    uint32_t ks[5];
    
    /* Initialize key schedule */
    ks[0] = key.v[0];
    ks[1] = key.v[1];
    ks[2] = key.v[2];
    ks[3] = key.v[3];
    ks[4] = ks[0] ^ ks[1] ^ ks[2] ^ ks[3] ^ THREEFRY_C240;
    
    /* Initialize state with counter */
    x[0] = counter.v[0];
    x[1] = counter.v[1];
    x[2] = counter.v[2];
    x[3] = counter.v[3];
    
    /* Initial key injection */
    x[0] += ks[0];
    x[1] += ks[1];
    x[2] += ks[2];
    x[3] += ks[3];
    
    /* 20 rounds */
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Key injection after round 4 */
    x[0] += ks[1];
    x[1] += ks[2];
    x[2] += ks[3];
    x[3] += ks[4] + 1;
    
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Key injection after round 8 */
    x[0] += ks[2];
    x[1] += ks[3];
    x[2] += ks[4];
    x[3] += ks[0] + 2;
    
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Key injection after round 12 */
    x[0] += ks[3];
    x[1] += ks[4];
    x[2] += ks[0];
    x[3] += ks[1] + 3;
    
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Key injection after round 16 */
    x[0] += ks[4];
    x[1] += ks[0];
    x[2] += ks[1];
    x[3] += ks[2] + 4;
    
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Final key injection after round 20 */
    x[0] += ks[0];
    x[1] += ks[1];
    x[2] += ks[2];
    x[3] += ks[3] + 5;
    
    uint4x32_t result;
    result.v[0] = x[0];
    result.v[1] = x[1];
    result.v[2] = x[2];
    result.v[3] = x[3];
    return result;
}

/* Conversion functions from uint4x32 to various precisions uniformly */

/* Convert to float in [0, 1) */
extern float uint32_to_single_uniform(uint32_t x) {
    /* Use upper 24 bits for float mantissa (23 bits + implicit 1) */
    return (x >> 8) * (1.0f / 16777216.0f);
}

/* Convert to double in [0, 1) */
extern double uint32_to_double_uniform(uint32_t x) {
    return x * (1.0 / 4294967296.0);
}

/* Uint4x32 to float32 uniform - uses first 32 bits */
extern float uint4x32_to_single_uniform(uint4x32_t x) {
    return uint32_to_single_uniform(x.v[0]);
}

/* Uint4x32 to float64 uniform - uses first 64 bits */
extern double uint4x32_to_double_uniform(uint4x32_t x) {
    uint64_t combined = ((uint64_t)x.v[1] << 32) | x.v[0];
    return combined * (1.0 / 18446744073709551616.0);
}

/* Uint4x32 to int32 uniform - full range */
extern int32_t uint4x32_to_int32_uniform(uint4x32_t x) {
    return (int32_t)x.v[0];
}

/* Uint4x32 to int64 uniform - full range */
extern int64_t uint4x32_to_int64_uniform(uint4x32_t x) {
    return (int64_t)(((uint64_t)x.v[1] << 32) | x.v[0]);
}

/* Uint4x32 to uint32 uniform - full range */
extern uint32_t uint4x32_to_uint32_uniform(uint4x32_t x) {
    return x.v[0];
}

/* Uint4x32 to uint64 uniform - full range */
extern uint64_t uint4x32_to_uint64_uniform(uint4x32_t x) {
    return ((uint64_t)x.v[1] << 32) | x.v[0];
}

/* Uint4x32 to int8 uniform - full range */
extern int8_t uint4x32_to_byte_uniform(uint4x32_t x) {
    return (int8_t)(x.v[0] & 0xFF);
}

/* Uint4x32 to uint16 uniform - full range */
extern uint16_t uint4x32_to_uint16_uniform(uint4x32_t x) {
    return (uint16_t)(x.v[0] & 0xFFFF);
}

/* Uint4x32 to bfloat16 uniform - uses first 16 bits */
extern uint16_t uint4x32_to_bfloat16_uniform(uint4x32_t x) {
    /* Convert to float first, then to bfloat16 */
    float f = uint32_to_single_uniform(x.v[0]);
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

/* Uint4x32 to float16 uniform - uses first 16 bits */
extern uint16_t uint4x32_to_half_uniform(uint4x32_t x) {
    /* Simplified conversion - proper fp16 would need more complex handling */
    /* This creates a uniform distribution in [0, 1) for fp16 */
    uint16_t raw = (uint16_t)(x.v[0] & 0xFFFF);
    /* Map to [0, 1) range in fp16 format */
    /* Sign bit = 0, exponent between -14 and 0, mantissa from raw bits */
    if (raw == 0) return 0;
    
    /* Find the highest set bit */
    int shift = 0;
    uint16_t temp = raw;
    while (temp >>= 1) shift++;
    
    /* Normalize mantissa */
    uint16_t mantissa = (raw << (10 - shift)) & 0x3FF;
    /* Exponent for values in [0, 1) */
    uint16_t exponent = 14 - shift;
    
    return (exponent << 10) | mantissa;
}

/* Uint4x32 to fp8 uniform - uses first 8 bits */
extern uint8_t uint4x32_to_fp8_uniform(uint4x32_t x) {
    return (uint8_t)(x.v[0] & 0xFF);
}

/* Conversion functions from various precisions to uint4x32_t */
extern  uint4x32_t single_to_uint4x32(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));
    uint4x32_t result = {{bits, 0, 0, 0}};
    return result;
}

extern uint4x32_t double_to_uint4x32(double x) {
    uint64_t bits;
    memcpy(&bits, &x, sizeof(double));
    uint4x32_t result = {{(uint32_t)(bits & 0xFFFFFFFF), (uint32_t)(bits >> 32), 0, 0}};
    return result;
}

extern uint4x32_t int32_to_uint4x32(int32_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}

extern uint4x32_t int64_to_uint4x32(int64_t x) {
    uint64_t bits = (uint64_t)x;
    uint4x32_t result = {{(uint32_t)(bits & 0xFFFFFFFF), (uint32_t)(bits >> 32), 0, 0}};
    return result;
}

extern uint4x32_t uint32_to_uint4x32(uint32_t x) {
    uint4x32_t result = {{x, 0, 0, 0}};
    return result;
}

extern uint4x32_t uint64_to_uint4x32(uint64_t x) {
    uint4x32_t result = {{(uint32_t)(x & 0xFFFFFFFF), (uint32_t)(x >> 32), 0, 0}};
    return result;
}

extern uint4x32_t byte_to_uint4x32(unsigned char x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}

extern uint4x32_t uint16_to_uint4x32(uint16_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}

extern uint4x32_t bfloat16_to_uint4x32(uint16_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}

extern uint4x32_t half_to_uint4x32(uint16_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}

extern uint4x32_t fp8_to_uint4x32(uint8_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}

/* Pure C conversion functions for use in C backends */

/* BFloat16 to Float conversion (C function) */
extern float bfloat16_to_single(uint16_t bf16)
{
  /* BFloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
     To convert to float32, we shift left by 16 bits */
  uint32_t f32 = ((uint32_t)bf16) << 16;
  return *((float *)&f32);
}

/* Float to BFloat16 conversion (C function) */
extern uint16_t single_to_bfloat16(float f)
{
  uint32_t f32 = *((uint32_t *)&f);

  /* Round to nearest even */
  uint32_t rounded = f32 + 0x7FFF + ((f32 >> 16) & 1);
  return (uint16_t)(rounded >> 16);
}

/* FP8 E5M2 format to Float conversion (C function)
   Format: 1 sign bit, 5 exponent bits, 2 mantissa bits */
extern float fp8_to_single(uint8_t fp8)
{
  /* Handle zero */
  if (fp8 == 0)
  {
    return 0.0f;
  }

  uint32_t sign = (fp8 >> 7) & 1;
  uint32_t exp = (fp8 >> 2) & 0x1F;
  uint32_t mant = fp8 & 0x3;

  /* Handle special cases */
  if (exp == 0x1F)
  { /* Infinity or NaN */
    if (mant == 0)
    {
      return sign ? -INFINITY : INFINITY;
    }
    else
    {
      return NAN;
    }
  }

  /* Denormalized numbers */
  if (exp == 0)
  {
    float result = ldexpf((float)mant / 4.0f, -14);
    if (sign)
      result = -result;
    return result;
  }

  /* Normalized numbers */
  float result = (1.0f + (float)mant * 0.25f) * ldexpf(1.0f, (int)exp - 15);
  if (sign)
    result = -result;

  return result;
}

/* Float to FP8 E5M2 conversion (C function) */
extern uint8_t single_to_fp8(float f)
{
  /* Handle zero */
  if (f == 0.0f)
  {
    return 0;
  }

  uint32_t sign = (f < 0) ? 1 : 0;
  f = fabsf(f);

  /* Handle special cases */
  if (isinf(f))
  {
    return (sign << 7) | 0x7C; /* Infinity: exp=0x1F, mant=0 */
  }
  if (isnan(f))
  {
    return (sign << 7) | 0x7F; /* NaN: exp=0x1F, mant!=0 */
  }

  /* Get exponent and mantissa */
  int exp_val;
  float mant_f = frexpf(f, &exp_val);
  int exp = exp_val + 14; /* Bias is 15, but frexp gives us mantissa in [0.5, 1) */

  /* Clamp to representable range */
  if (exp < 0)
  {
    /* Underflow to zero */
    return sign << 7;
  }
  if (exp > 30)
  {
    /* Overflow to infinity */
    return (sign << 7) | 0x7C;
  }

  /* Handle denormalized numbers */
  if (exp == 0)
  {
    float denorm_mant = f * ldexpf(1.0f, 14) * 4.0f;
    uint32_t mant_bits = (uint32_t)(denorm_mant + 0.5f);
    if (mant_bits > 3)
      mant_bits = 3;
    return (sign << 7) | mant_bits;
  }

  /* Normalized numbers: convert mantissa from [0.5, 1) to [0, 0.75] */
  mant_f = (mant_f - 0.5f) * 4.0f;
  uint32_t mant_bits = (uint32_t)(mant_f + 0.5f); /* Round to nearest */
  if (mant_bits > 3)
    mant_bits = 3;

  return (uint8_t)((sign << 7) | ((exp & 0x1F) << 2) | (mant_bits & 0x3));
}

/* OCaml wrapper functions */

/* Helper functions to convert between OCaml and C uint4x32_t */
uint4x32_t ocaml_array_to_uint4x32(value v_array) {
    uint4x32_t result;
    result.v[0] = (uint32_t)Long_val(Field(v_array, 0));
    result.v[1] = (uint32_t)Long_val(Field(v_array, 1));
    result.v[2] = (uint32_t)Long_val(Field(v_array, 2));
    result.v[3] = (uint32_t)Long_val(Field(v_array, 3));
    return result;
}

value uint4x32_to_ocaml_array(uint4x32_t x) {
    CAMLparam0();
    CAMLlocal1(result);
    result = caml_alloc(4, 0);
    Store_field(result, 0, Val_long(x.v[0]));
    Store_field(result, 1, Val_long(x.v[1]));
    Store_field(result, 2, Val_long(x.v[2]));
    Store_field(result, 3, Val_long(x.v[3]));
    CAMLreturn(result);
}

/* Threefry4x32 OCaml wrapper */
CAMLprim value arrayjit_threefry4x32_ocaml(value v_key, value v_counter)
{
  CAMLparam2(v_key, v_counter);
  uint4x32_t key = ocaml_array_to_uint4x32(v_key);
  uint4x32_t counter = ocaml_array_to_uint4x32(v_counter);
  uint4x32_t result = arrayjit_threefry4x32(key, counter);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

/* Conversion from uint4x32 to various types - OCaml wrappers */
CAMLprim value arrayjit_uint4x32_to_single_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  float result = uint4x32_to_single_uniform(x);
  CAMLreturn(caml_copy_double((double)result));
}

CAMLprim value arrayjit_uint4x32_to_double_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  double result = uint4x32_to_double_uniform(x);
  CAMLreturn(caml_copy_double(result));
}

CAMLprim value arrayjit_uint4x32_to_int32_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  int32_t result = uint4x32_to_int32_uniform(x);
  CAMLreturn(Val_long(result));
}

CAMLprim value arrayjit_uint4x32_to_int64_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  int64_t result = uint4x32_to_int64_uniform(x);
  CAMLreturn(caml_copy_int64(result));
}

CAMLprim value arrayjit_uint4x32_to_uint32_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  uint32_t result = uint4x32_to_uint32_uniform(x);
  CAMLreturn(Val_long(result));
}

CAMLprim value arrayjit_uint4x32_to_uint64_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  uint64_t result = uint4x32_to_uint64_uniform(x);
  CAMLreturn(caml_copy_int64((int64_t)result));
}

CAMLprim value arrayjit_uint4x32_to_byte_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  int8_t result = uint4x32_to_byte_uniform(x);
  CAMLreturn(Val_int(result));
}

CAMLprim value arrayjit_uint4x32_to_uint16_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  uint16_t result = uint4x32_to_uint16_uniform(x);
  CAMLreturn(Val_int(result));
}

CAMLprim value arrayjit_uint4x32_to_bfloat16_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  uint16_t result = uint4x32_to_bfloat16_uniform(x);
  CAMLreturn(Val_int(result));
}

CAMLprim value arrayjit_uint4x32_to_half_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  uint16_t result = uint4x32_to_half_uniform(x);
  CAMLreturn(Val_int(result));
}

CAMLprim value arrayjit_uint4x32_to_fp8_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  uint8_t result = uint4x32_to_fp8_uniform(x);
  CAMLreturn(Val_int(result));
}

/* Conversion to uint4x32 from various types - OCaml wrappers */
CAMLprim value arrayjit_single_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  float x = (float)Double_val(v_x);
  uint4x32_t result = single_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_double_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  double x = Double_val(v_x);
  uint4x32_t result = double_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_int32_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  int32_t x = (int32_t)Long_val(v_x);
  uint4x32_t result = int32_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_int64_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  int64_t x = Int64_val(v_x);
  uint4x32_t result = int64_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_uint32_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint32_t x = (uint32_t)Long_val(v_x);
  uint4x32_t result = uint32_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_uint64_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint64_t x = (uint64_t)Int64_val(v_x);
  uint4x32_t result = uint64_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_byte_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  unsigned char x = (unsigned char)Int_val(v_x);
  uint4x32_t result = byte_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_uint16_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint16_t x = (uint16_t)Int_val(v_x);
  uint4x32_t result = uint16_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_bfloat16_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint16_t x = (uint16_t)Int_val(v_x);
  uint4x32_t result = bfloat16_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_half_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint16_t x = (uint16_t)Int_val(v_x);
  uint4x32_t result = half_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_fp8_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint8_t x = (uint8_t)Int_val(v_x);
  uint4x32_t result = fp8_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

/* BFloat16 to Float conversion (OCaml wrapper) */
CAMLprim value arrayjit_bfloat16_to_single(value v_bf16)
{
  CAMLparam1(v_bf16);
  uint16_t bf16 = (uint16_t)Int_val(v_bf16);
  float result = bfloat16_to_single(bf16);
  CAMLreturn(caml_copy_double((double)result));
}

/* Float to BFloat16 conversion (OCaml wrapper) */
CAMLprim value arrayjit_single_to_bfloat16(value v_float)
{
  CAMLparam1(v_float);
  float f = (float)Double_val(v_float);
  uint16_t bf16 = single_to_bfloat16(f);
  CAMLreturn(Val_int(bf16));
}

/* FP8 E5M2 format to Float conversion (OCaml wrapper) */
CAMLprim value arrayjit_fp8_to_single(value v_fp8)
{
  CAMLparam1(v_fp8);
  uint8_t fp8 = (uint8_t)Int_val(v_fp8);
  float result = fp8_to_single(fp8);
  CAMLreturn(caml_copy_double((double)result));
}

/* Float to FP8 E5M2 conversion (OCaml wrapper) */
CAMLprim value arrayjit_single_to_fp8(value v_float)
{
  CAMLparam1(v_float);
  float f = (float)Double_val(v_float);
  uint8_t fp8 = single_to_fp8(f);
  CAMLreturn(Val_int(fp8));
}

// TODO: a more efficient approach would involve computing strides once and using memcpy
// for contiguous inner slices, but that adds complexity).
CAMLprim value arrayjit_copy_with_padding(value v_source, value v_target, value v_padding)
{
  CAMLparam3(v_source, v_target, v_padding);

  struct caml_ba_array *source_ba = Caml_ba_array_val(v_source);
  struct caml_ba_array *target_ba = Caml_ba_array_val(v_target);
  int ndim = source_ba->num_dims;

  if (ndim != target_ba->num_dims)
  {
    caml_failwith("Source and target must have the same number of dimensions");
  }

  if (ndim == 0)
  {
    CAMLreturn(Val_unit);
  }

  void *source_data = Caml_ba_data_val(v_source);
  void *target_data = Caml_ba_data_val(v_target);

  size_t elem_size = caml_ba_byte_size(source_ba);

  // Use source dimensions directly from bigarray
  intnat *source_shape = source_ba->dim;

  // Extract paddings
  if (Wosize_val(v_padding) != (uintnat)ndim)
  {
    caml_failwith("Padding array length mismatch");
  }
  // FIXME: this is a memory leak
  intnat *left = malloc(ndim * sizeof(intnat));
  intnat *right = malloc(ndim * sizeof(intnat));
  if (left == NULL || right == NULL)
    caml_failwith("Malloc failed");
  for (int d = 0; d < ndim; d++)
  {
    value pad = Field(v_padding, d);
    left[d] = Long_val(Field(pad, 0));
    right[d] = Long_val(Field(pad, 1));
    if (left[d] < 0 || right[d] < 0)
      caml_failwith("Negative padding");
  }

  // Verify target dimensions match source + padding
  for (int d = 0; d < ndim; d++)
  {
    if (target_ba->dim[d] != source_shape[d] + left[d] + right[d])
    {
      caml_failwith("Target dimensions do not match source + padding");
    }
  }

  // Multi-dimensional index loop
  intnat *indices = calloc(ndim, sizeof(intnat));
  if (indices == NULL)
    caml_failwith("Calloc failed");

  while (1)
  {
    // Compute source flat offset
    intnat source_offset = 0;
    intnat s_stride = 1;
    for (int d = ndim - 1; d >= 0; d--)
    {
      source_offset += indices[d] * s_stride;
      s_stride *= source_shape[d];
    }

    // Compute target flat offset with padding offset
    intnat target_offset = 0;
    intnat t_stride = 1;
    for (int d = ndim - 1; d >= 0; d--)
    {
      target_offset += (indices[d] + left[d]) * t_stride;
      t_stride *= target_ba->dim[d];
    }

    // Copy the element
    memcpy((char *)target_data + target_offset * elem_size,
           (char *)source_data + source_offset * elem_size,
           elem_size);

    // Increment indices (odometer-style)
    int carry = 1;
    for (int d = ndim - 1; d >= 0; d--)
    {
      if (carry == 0)
        break;
      indices[d] += carry;
      if (indices[d] < source_shape[d])
      {
        carry = 0;
      }
      else
      {
        indices[d] = 0;
        carry = 1;
      }
    }
    if (carry == 1)
      break; // Done
  }

  free(indices);
  free(left);
  free(right);

  CAMLreturn(Val_unit);
}