let includes =
  {|
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* No longer need export macros since we're using textual prepending */
|}

(* Each entry is (key, definition, dependencies) *)
let builtins =
  [
    (* Float16 feature detection and type definitions *)
    ( "HAS_NATIVE_FLOAT16",
      {|
#ifdef __FLT16_MAX__
  #define HAS_NATIVE_FLOAT16 1
#else
  #define HAS_NATIVE_FLOAT16 0
#endif
|},
      [] );
    ( "HALF_T",
      {|
#if HAS_NATIVE_FLOAT16
  #define HALF_T _Float16
#else
  #define HALF_T uint16_t
#endif
|},
      [ "HAS_NATIVE_FLOAT16" ] );
    ( "HALF_TO_FP",
      {|
#if HAS_NATIVE_FLOAT16
  #define HALF_TO_FP(x) (x)  /* Identity - already in floating point */
#else
  #define HALF_TO_FP(x) half_to_float_emulated(x)  /* Convert to float for computation */
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "half_to_float_emulated" ] );
    ( "FP_TO_HALF",
      {|
#if HAS_NATIVE_FLOAT16
  #define FP_TO_HALF(x) (x)  /* Identity - already half precision */
#else
  #define FP_TO_HALF(x) float_to_half_emulated(x)  /* Convert back from float */
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "float_to_half_emulated" ] );
    ( "HALF_TO_FLOAT",
      {|
#if HAS_NATIVE_FLOAT16
  #define HALF_TO_FLOAT(x) ((float)(x))
#else
  #define HALF_TO_FLOAT(x) half_to_float_emulated(x)
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "half_to_float_emulated" ] );
    ( "FLOAT_TO_HALF",
      {|
#if HAS_NATIVE_FLOAT16
  #define FLOAT_TO_HALF(x) ((_Float16)(x))
#else
  #define FLOAT_TO_HALF(x) float_to_half_emulated(x)
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "float_to_half_emulated" ] );
    ( "HALF_TO_UINT16",
      {|
#if HAS_NATIVE_FLOAT16
  #define HALF_TO_UINT16(x) ({ _Float16 _h = (x); uint16_t _r; memcpy(&_r, &_h, 2); _r; })
#else
  #define HALF_TO_UINT16(x) (x)
#endif
|},
      [ "HAS_NATIVE_FLOAT16" ] );
    ( "UINT16_TO_HALF",
      {|
#if HAS_NATIVE_FLOAT16
  #define UINT16_TO_HALF(x) ({ uint16_t _u = (x); _Float16 _h; memcpy(&_h, &_u, 2); _h; })
#else
  #define UINT16_TO_HALF(x) (x)
#endif
|},
      [ "HAS_NATIVE_FLOAT16" ] );
    (* Float16 emulation functions *)
    ( "half_to_float_emulated",
      {|
#if !HAS_NATIVE_FLOAT16
/* Convert IEEE 754 half precision (stored as uint16_t) to float */
static float half_to_float_emulated(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    if (exponent == 0) {
        if (mantissa == 0) {
            /* Zero */
            return sign ? -0.0f : 0.0f;
        } else {
            /* Subnormal */
            float result = ldexpf(mantissa / 1024.0f, -14);
            return sign ? -result : result;
        }
    } else if (exponent == 31) {
        if (mantissa == 0) {
            /* Infinity */
            return sign ? -INFINITY : INFINITY;
        } else {
            /* NaN */
            return NAN;
        }
    } else {
        /* Normal number */
        float result = ldexpf(1.0f + mantissa / 1024.0f, exponent - 15);
        return sign ? -result : result;
    }
}
#endif
|},
      [ "HAS_NATIVE_FLOAT16" ] );
    ( "float_to_half_emulated",
      {|
#if !HAS_NATIVE_FLOAT16
/* Convert float to IEEE 754 half precision (stored as uint16_t) */
static uint16_t float_to_half_emulated(float f) {
    uint32_t f32;
    memcpy(&f32, &f, sizeof(float));
    
    uint32_t sign = (f32 >> 31) & 0x1;
    uint32_t exponent = (f32 >> 23) & 0xFF;
    uint32_t mantissa = f32 & 0x7FFFFF;
    
    /* Convert exponent from float bias (127) to half bias (15) */
    int32_t new_exp = (int32_t)exponent - 127 + 15;
    
    if (exponent == 0xFF) {
        /* Infinity or NaN */
        if (mantissa == 0) {
            /* Infinity */
            return (sign << 15) | (0x1F << 10);
        } else {
            /* NaN - preserve sign and set mantissa bit */
            return (sign << 15) | (0x1F << 10) | 0x200;
        }
    } else if (new_exp <= 0) {
        /* Underflow to zero or subnormal */
        if (new_exp < -10) {
            /* Too small - flush to zero */
            return sign << 15;
        }
        /* Subnormal - with round-to-nearest-even */
        uint32_t shift = -new_exp + 1;
        mantissa = (mantissa | 0x800000);
        
        /* For subnormal, we need to shift right by (shift + 13) total bits */
        uint32_t total_shift = shift + 13;
        
        if (total_shift >= 24) {
            /* Would shift away all bits */
            return sign << 15;
        }
        
        /* Extract guard, round, and sticky bits before shifting */
        uint32_t guard_bit = (mantissa >> (total_shift - 1)) & 1;
        uint32_t round_bit = total_shift > 1 ? ((mantissa >> (total_shift - 2)) & 1) : 0;
        uint32_t sticky_mask = (1U << (total_shift - 2)) - 1;
        uint32_t sticky_bits = total_shift > 1 ? (mantissa & sticky_mask) : 0;
        
        mantissa = mantissa >> total_shift;
        
        /* Round to nearest even */
        if (guard_bit && (round_bit || sticky_bits || (mantissa & 1))) {
            mantissa++;
        }
        
        return (sign << 15) | mantissa;
    } else if (new_exp >= 0x1F) {
        /* Overflow to infinity */
        return (sign << 15) | (0x1F << 10);
    } else {
        /* Normal number - with round-to-nearest-even (banker's rounding) */
        uint32_t rounded_mantissa;
        uint32_t guard_bit = (mantissa >> 12) & 1;
        uint32_t round_bit = (mantissa >> 11) & 1;
        uint32_t sticky_bits = mantissa & 0x7FF;
        
        rounded_mantissa = mantissa >> 13;
        
        /* Round to nearest even: round up if we have:
         * - guard bit set and (round bit set OR sticky bits non-zero OR mantissa LSB set)
         */
        if (guard_bit && (round_bit || sticky_bits || (rounded_mantissa & 1))) {
            rounded_mantissa++;
        }
        
        if (rounded_mantissa > 0x3FF) {
            /* Rounding caused overflow in mantissa */
            new_exp++;
            rounded_mantissa = 0;
            if (new_exp >= 0x1F) {
                /* Overflow to infinity */
                return (sign << 15) | (0x1F << 10);
            }
        }
        return (sign << 15) | (new_exp << 10) | rounded_mantissa;
    }
}
#endif
|},
      [ "HAS_NATIVE_FLOAT16" ] );
    (* Threefry4x32 types and complete implementation *)
    ("uint4x32_t", {|
typedef struct {
    uint32_t v[4];
} uint4x32_t;
|}, []);
    ( "threefry_common",
      {|
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
|},
      [ "uint4x32_t" ] );
    ( "arrayjit_threefry4x32_crypto",
      {|

/* Threefry4x32 implementation - 20 rounds (cryptographic version) */
uint4x32_t arrayjit_threefry4x32_crypto(uint4x32_t key, uint4x32_t counter) {
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
|},
      [ "uint4x32_t"; "threefry_common" ] );
    ( "arrayjit_threefry4x32_light",
      {|
/* Threefry4x32 implementation - 2 rounds (light version, as in JAX/XLA) */
uint4x32_t arrayjit_threefry4x32_light(uint4x32_t key, uint4x32_t counter) {
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
    
    /* Only 2 rounds for light version */
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Final key injection after round 2 */
    x[0] += ks[1];
    x[1] += ks[2];
    x[2] += ks[3];
    x[3] += ks[4] + 1;
    
    uint4x32_t result;
    result.v[0] = x[0];
    result.v[1] = x[1];
    result.v[2] = x[2];
    result.v[3] = x[3];
    return result;
}
|},
      [ "uint4x32_t"; "threefry_common" ] );
    ( "arrayjit_threefry4x32",
      {|
/* Default threefry4x32 function - will be configured at runtime */
uint4x32_t arrayjit_threefry4x32(uint4x32_t key, uint4x32_t counter) {
    /* Default to light version */
    return arrayjit_threefry4x32_light(key, counter);
}
|},
      [ "uint4x32_t"; "arrayjit_threefry4x32_light" ] );
    (* Vector types with half precision *)
    ("half8_t", {|
typedef struct { HALF_T v[8]; } half8_t;
|}, [ "HALF_T" ]);
    ("float4_t", {|
typedef struct { float v[4]; } float4_t;
|}, []);
    ("double2_t", {|
typedef struct { double v[2]; } double2_t;
|}, []);
    ("int32x4_t", {|
typedef struct { int32_t v[4]; } int32x4_t;
|}, []);
    ("int64x2_t", {|
typedef struct { int64_t v[2]; } int64x2_t;
|}, []);
    ("int8x16_t", {|
typedef struct { int8_t v[16]; } int8x16_t;
|}, []);
    ("uint16x8_t", {|
typedef struct { uint16_t v[8]; } uint16x8_t;
|}, []);
    ("uint8x16_t", {|
typedef struct { uint8_t v[16]; } uint8x16_t;
|}, []);
    (* Basic conversion functions *)
    ( "uint32_to_single_uniform",
      {|
/* Convert to float in [0, 1) */
float uint32_to_single_uniform(uint32_t x) {
    /* Use upper 24 bits for float mantissa (23 bits + implicit 1) */
    return (x >> 8) * (1.0f / 16777216.0f);
}
|},
      [] );
    ( "uint32_to_double_uniform",
      {|
/* Convert to double in [0, 1) */
double uint32_to_double_uniform(uint32_t x) {
    return x * (1.0 / 4294967296.0);
}
|},
      [] );
    (* Conversion functions with dependencies *)
    ( "uint4x32_to_single_uniform",
      {|
/* Uint4x32 to float32 uniform - uses first 32 bits */
float uint4x32_to_single_uniform(uint4x32_t x) {
    return uint32_to_single_uniform(x.v[0]);
}
|},
      [ "uint4x32_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_half_uniform",
      {|
/* Uint4x32 to float16 uniform - uses first 16 bits */
uint16_t uint4x32_to_half_uniform(uint4x32_t x) {
    /* Convert through float for consistent behavior */
    float f = (x.v[0] & 0xFFFF) * (1.0f / 65536.0f);
    return FLOAT_TO_HALF(f);
}
|},
      [ "uint4x32_t"; "FLOAT_TO_HALF" ] );
    ( "uint4x32_to_half_uniform_vec",
      {|
/* Convert uint4x32 to 8 float16s uniform */
half8_t uint4x32_to_half_uniform_vec(uint4x32_t x) {
    half8_t result;
    for (int i = 0; i < 4; i++) {
        // Extract two 16-bit values and convert to float in [0, 1)
        float f1 = (x.v[i] & 0xFFFF) * (1.0f / 65536.0f);
        float f2 = ((x.v[i] >> 16) & 0xFFFF) * (1.0f / 65536.0f);
        
        // Convert to half precision - macros handle both native and emulated cases
        result.v[i*2 + 0] = FLOAT_TO_HALF(f1);
        result.v[i*2 + 1] = FLOAT_TO_HALF(f2);
    }
    return result;
}
|},
      [ "uint4x32_t"; "half8_t"; "FLOAT_TO_HALF" ] );
    (* Pure C conversion functions *)
    ( "bfloat16_to_single",
      {|
/* BFloat16 to Float conversion (C function) */
float bfloat16_to_single(uint16_t bf16)
{
  /* BFloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
     To convert to float32, we shift left by 16 bits */
  uint32_t f32 = ((uint32_t)bf16) << 16;
  return *((float *)&f32);
}
|},
      [] );
    ( "single_to_bfloat16",
      {|
/* Float to BFloat16 conversion (C function) */
uint16_t single_to_bfloat16(float f)
{
  uint32_t f32 = *((uint32_t *)&f);

  /* Round to nearest even */
  uint32_t rounded = f32 + 0x7FFF + ((f32 >> 16) & 1);
  return (uint16_t)(rounded >> 16);
}
|},
      [] );
    ( "half_to_single",
      {|
/* Half (Float16) to Float conversion (C function) */
float half_to_single(uint16_t h)
{
  HALF_T half_val = UINT16_TO_HALF(h);
  return HALF_TO_FLOAT(half_val);
}
|},
      [ "HALF_T"; "UINT16_TO_HALF"; "HALF_TO_FLOAT" ] );
    ( "single_to_half",
      {|
/* Float to Half (Float16) conversion (C function) */
uint16_t single_to_half(float f)
{
  HALF_T half_val = FLOAT_TO_HALF(f);
  return HALF_TO_UINT16(half_val);
}
|},
      [ "HALF_T"; "FLOAT_TO_HALF"; "HALF_TO_UINT16" ] );
    ( "fp8_to_single",
      {|
/* FP8 E5M2 format to Float conversion (C function)
   Format: 1 sign bit, 5 exponent bits, 2 mantissa bits */
float fp8_to_single(uint8_t fp8)
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
|},
      [] );
    ( "single_to_fp8",
      {|
/* Float to FP8 E5M2 conversion (C function) */
uint8_t single_to_fp8(float f)
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
|},
      [] );
    (* Conversion functions from various precisions to uint4x32_t *)
    ( "int32_to_uint4x32",
      {|
uint4x32_t int32_to_uint4x32(int32_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "int64_to_uint4x32",
      {|
uint4x32_t int64_to_uint4x32(int64_t x) {
    uint64_t bits = (uint64_t)x;
    uint4x32_t result = {{(uint32_t)(bits & 0xFFFFFFFF), (uint32_t)(bits >> 32), 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "uint32_to_uint4x32",
      {|
uint4x32_t uint32_to_uint4x32(uint32_t x) {
    uint4x32_t result = {{x, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "uint64_to_uint4x32",
      {|
uint4x32_t uint64_to_uint4x32(uint64_t x) {
    uint4x32_t result = {{(uint32_t)(x & 0xFFFFFFFF), (uint32_t)(x >> 32), 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "single_to_uint4x32",
      {|
uint4x32_t single_to_uint4x32(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));
    uint4x32_t result = {{bits, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "double_to_uint4x32",
      {|
uint4x32_t double_to_uint4x32(double x) {
    uint64_t bits;
    memcpy(&bits, &x, sizeof(double));
    uint4x32_t result = {{(uint32_t)(bits & 0xFFFFFFFF), (uint32_t)(bits >> 32), 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "byte_to_uint4x32",
      {|
uint4x32_t byte_to_uint4x32(unsigned char x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "uint16_to_uint4x32",
      {|
uint4x32_t uint16_to_uint4x32(uint16_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "bfloat16_to_uint4x32",
      {|
uint4x32_t bfloat16_to_uint4x32(uint16_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "half_to_uint4x32",
      {|
uint4x32_t half_to_uint4x32(uint16_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "fp8_to_uint4x32",
      {|
uint4x32_t fp8_to_uint4x32(uint8_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    (* More uint4x32 to various precision conversion functions *)
    ( "uint4x32_to_double_uniform",
      {|
/* Uint4x32 to float64 uniform - uses first 64 bits */
double uint4x32_to_double_uniform(uint4x32_t x) {
    uint64_t combined = ((uint64_t)x.v[1] << 32) | x.v[0];
    return combined * (1.0 / 18446744073709551616.0);
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_int32_uniform",
      {|
/* Uint4x32 to int32 uniform - full range */
int32_t uint4x32_to_int32_uniform(uint4x32_t x) {
    return (int32_t)x.v[0];
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_int64_uniform",
      {|
/* Uint4x32 to int64 uniform - full range */
int64_t uint4x32_to_int64_uniform(uint4x32_t x) {
    return (int64_t)(((uint64_t)x.v[1] << 32) | x.v[0]);
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_uint32_uniform",
      {|
/* Uint4x32 to uint32 uniform - full range */
uint32_t uint4x32_to_uint32_uniform(uint4x32_t x) {
    return x.v[0];
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_uint64_uniform",
      {|
/* Uint4x32 to uint64 uniform - full range */
uint64_t uint4x32_to_uint64_uniform(uint4x32_t x) {
    return ((uint64_t)x.v[1] << 32) | x.v[0];
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_byte_uniform",
      {|
/* Uint4x32 to int8 uniform - full range */
int8_t uint4x32_to_byte_uniform(uint4x32_t x) {
    return (int8_t)(x.v[0] & 0xFF);
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_uint16_uniform",
      {|
/* Uint4x32 to uint16 uniform - full range */
uint16_t uint4x32_to_uint16_uniform(uint4x32_t x) {
    return (uint16_t)(x.v[0] & 0xFFFF);
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_bfloat16_uniform",
      {|
/* Uint4x32 to bfloat16 uniform - uses first 16 bits */
uint16_t uint4x32_to_bfloat16_uniform(uint4x32_t x) {
    /* Convert to float first, then to bfloat16 */
    float f = uint32_to_single_uniform(x.v[0]);
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    /* Round to nearest even for bfloat16 */
    uint16_t bf = bits >> 16;
    if ((bits & 0x8000) && ((bits & 0x7FFF) || (bf & 1))) bf++;
    return bf;
}
|},
      [ "uint4x32_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_fp8_uniform",
      {|
/* Uint4x32 to fp8 uniform - uses first 8 bits */
uint8_t uint4x32_to_fp8_uniform(uint4x32_t x) {
    return (uint8_t)(x.v[0] & 0xFF);
}
|},
      [ "uint4x32_t" ] );
    (* Vectorized conversion functions *)
    ( "uint4x32_to_single_uniform_vec",
      {|
/* Convert uint4x32 to 4 floats in [0, 1) */
float4_t uint4x32_to_single_uniform_vec(uint4x32_t x) {
    float4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = uint32_to_single_uniform(x.v[i]);
    }
    return result;
}
|},
      [ "uint4x32_t"; "float4_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_double_uniform_vec",
      {|
/* Convert uint4x32 to 2 doubles in [0, 1) */
double2_t uint4x32_to_double_uniform_vec(uint4x32_t x) {
    double2_t result;
    uint64_t combined1 = ((uint64_t)x.v[1] << 32) | x.v[0];
    uint64_t combined2 = ((uint64_t)x.v[3] << 32) | x.v[2];
    result.v[0] = combined1 * (1.0 / 18446744073709551616.0);
    result.v[1] = combined2 * (1.0 / 18446744073709551616.0);
    return result;
}
|},
      [ "uint4x32_t"; "double2_t" ] );
    ( "uint4x32_to_int32_uniform_vec",
      {|
/* Convert uint4x32 to 4 int32s - full range */
int32x4_t uint4x32_to_int32_uniform_vec(uint4x32_t x) {
    int32x4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = (int32_t)x.v[i];
    }
    return result;
}
|},
      [ "uint4x32_t"; "int32x4_t" ] );
    ( "uint4x32_to_int64_uniform_vec",
      {|
/* Convert uint4x32 to 2 int64s - full range */
int64x2_t uint4x32_to_int64_uniform_vec(uint4x32_t x) {
    int64x2_t result;
    result.v[0] = (int64_t)(((uint64_t)x.v[1] << 32) | x.v[0]);
    result.v[1] = (int64_t)(((uint64_t)x.v[3] << 32) | x.v[2]);
    return result;
}
|},
      [ "uint4x32_t"; "int64x2_t" ] );
    ( "uint4x32_to_byte_uniform_vec",
      {|
/* Convert uint4x32 to 16 int8s - full range */
int8x16_t uint4x32_to_byte_uniform_vec(uint4x32_t x) {
    int8x16_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i*4 + 0] = (int8_t)(x.v[i] & 0xFF);
        result.v[i*4 + 1] = (int8_t)((x.v[i] >> 8) & 0xFF);
        result.v[i*4 + 2] = (int8_t)((x.v[i] >> 16) & 0xFF);
        result.v[i*4 + 3] = (int8_t)((x.v[i] >> 24) & 0xFF);
    }
    return result;
}
|},
      [ "uint4x32_t"; "int8x16_t" ] );
    ( "uint4x32_to_uint16_uniform_vec",
      {|
/* Convert uint4x32 to 8 uint16s - full range */
uint16x8_t uint4x32_to_uint16_uniform_vec(uint4x32_t x) {
    uint16x8_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i*2 + 0] = (uint16_t)(x.v[i] & 0xFFFF);
        result.v[i*2 + 1] = (uint16_t)((x.v[i] >> 16) & 0xFFFF);
    }
    return result;
}
|},
      [ "uint4x32_t"; "uint16x8_t" ] );
    ( "uint4x32_to_bfloat16_uniform_vec",
      {|
/* Convert uint4x32 to 8 bfloat16s uniform */
uint16x8_t uint4x32_to_bfloat16_uniform_vec(uint4x32_t x) {
    uint16x8_t result;
    for (int i = 0; i < 4; i++) {
        // Convert each uint32 to two bfloat16 values
        float f1 = ((x.v[i] & 0xFFFF) >> 0) * (1.0f / 65536.0f);
        float f2 = ((x.v[i] >> 16) & 0xFFFF) * (1.0f / 65536.0f);
        uint32_t bits1, bits2;
        memcpy(&bits1, &f1, sizeof(float));
        memcpy(&bits2, &f2, sizeof(float));
        // Round to nearest even for bfloat16
        uint16_t bf1 = bits1 >> 16;
        uint16_t bf2 = bits2 >> 16;
        // Check if we need to round up (guard bit set and round/sticky or LSB)
        if ((bits1 & 0x8000) && ((bits1 & 0x7FFF) || (bf1 & 1))) bf1++;
        if ((bits2 & 0x8000) && ((bits2 & 0x7FFF) || (bf2 & 1))) bf2++;
        result.v[i*2 + 0] = bf1;
        result.v[i*2 + 1] = bf2;
    }
    return result;
}
|},
      [ "uint4x32_t"; "uint16x8_t" ] );
    ( "uint4x32_to_fp8_uniform_vec",
      {|
/* Convert uint4x32 to 16 fp8s uniform */
uint8x16_t uint4x32_to_fp8_uniform_vec(uint4x32_t x) {
    uint8x16_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i*4 + 0] = (uint8_t)(x.v[i] & 0xFF);
        result.v[i*4 + 1] = (uint8_t)((x.v[i] >> 8) & 0xFF);
        result.v[i*4 + 2] = (uint8_t)((x.v[i] >> 16) & 0xFF);
        result.v[i*4 + 3] = (uint8_t)((x.v[i] >> 24) & 0xFF);
    }
    return result;
}
|},
      [ "uint4x32_t"; "uint8x16_t" ] );
  ]

let source = includes ^ String.concat "" (List.map (fun (_, def, _) -> def) builtins)
