#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/bigarray.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* Check for _Float16 support and define macros for zero-overhead abstraction */
#ifdef __FLT16_MAX__
  #define HAS_NATIVE_FLOAT16 1
  /* Native _Float16 support - use direct types and casts */
  #define HALF_T _Float16
  #define HALF_TO_FP(x) (x)  /* Identity - already in floating point */
  #define FP_TO_HALF(x) (x)  /* Identity - already half precision */
  #define HALF_TO_FLOAT(x) ((float)(x))
  #define FLOAT_TO_HALF(x) ((_Float16)(x))
  #define HALF_TO_UINT16(x) ({ _Float16 _h = (x); uint16_t _r; memcpy(&_r, &_h, 2); _r; })
  #define UINT16_TO_HALF(x) ({ uint16_t _u = (x); _Float16 _h; memcpy(&_h, &_u, 2); _h; })
#else
  #define HAS_NATIVE_FLOAT16 0
  /* No native _Float16 - use uint16_t storage and conversion functions */
  #define HALF_T uint16_t
  #define HALF_TO_FP(x) half_to_float_emulated(x)  /* Convert to float for computation */
  #define FP_TO_HALF(x) float_to_half_emulated(x)  /* Convert back from float */
  #define HALF_TO_FLOAT(x) half_to_float_emulated(x)
  #define FLOAT_TO_HALF(x) float_to_half_emulated(x)
  #define HALF_TO_UINT16(x) (x)
  #define UINT16_TO_HALF(x) (x)
#endif

/* Float16 emulation functions for systems without _Float16 */
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

#endif /* !HAS_NATIVE_FLOAT16 */

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

/* Default threefry4x32 function - will be configured at runtime */
uint4x32_t arrayjit_threefry4x32(uint4x32_t key, uint4x32_t counter) {
    /* Default to light version */
    return arrayjit_threefry4x32_light(key, counter);
}

/* Vector types for efficient extraction of multiple values */
typedef struct { float v[4]; } float4_t;
typedef struct { double v[2]; } double2_t;
typedef struct { int32_t v[4]; } int32x4_t;
typedef struct { int64_t v[2]; } int64x2_t;
typedef struct { int8_t v[16]; } int8x16_t;
typedef struct { uint16_t v[8]; } uint16x8_t;
typedef struct { uint8_t v[16]; } uint8x16_t;
typedef struct { HALF_T v[8]; } half8_t;

/* Conversion functions from uint4x32 to various precisions uniformly */
// These return vectors to efficiently use all random bits

/* Convert to float in [0, 1) */
float uint32_to_single_uniform(uint32_t x) {
    /* Use upper 24 bits for float mantissa (23 bits + implicit 1) */
    return (x >> 8) * (1.0f / 16777216.0f);
}

/* Convert to double in [0, 1) */
double uint32_to_double_uniform(uint32_t x) {
    return x * (1.0 / 4294967296.0);
}

/* Uint4x32 to float32 uniform - uses first 32 bits */
float uint4x32_to_single_uniform(uint4x32_t x) {
    return uint32_to_single_uniform(x.v[0]);
}

/* Uint4x32 to float64 uniform - uses first 64 bits */
double uint4x32_to_double_uniform(uint4x32_t x) {
    uint64_t combined = ((uint64_t)x.v[1] << 32) | x.v[0];
    return combined * (1.0 / 18446744073709551616.0);
}

/* Uint4x32 to int32 uniform - full range */
int32_t uint4x32_to_int32_uniform(uint4x32_t x) {
    return (int32_t)x.v[0];
}

/* Uint4x32 to int64 uniform - full range */
int64_t uint4x32_to_int64_uniform(uint4x32_t x) {
    return (int64_t)(((uint64_t)x.v[1] << 32) | x.v[0]);
}

/* Uint4x32 to uint32 uniform - full range */
uint32_t uint4x32_to_uint32_uniform(uint4x32_t x) {
    return x.v[0];
}

/* Uint4x32 to uint64 uniform - full range */
uint64_t uint4x32_to_uint64_uniform(uint4x32_t x) {
    return ((uint64_t)x.v[1] << 32) | x.v[0];
}

/* Uint4x32 to int8 uniform - full range */
int8_t uint4x32_to_byte_uniform(uint4x32_t x) {
    return (int8_t)(x.v[0] & 0xFF);
}

/* Uint4x32 to uint16 uniform - full range */
uint16_t uint4x32_to_uint16_uniform(uint4x32_t x) {
    return (uint16_t)(x.v[0] & 0xFFFF);
}

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

/* Uint4x32 to float16 uniform - uses first 16 bits */
uint16_t uint4x32_to_half_uniform(uint4x32_t x) {
    /* Convert through float for consistent behavior */
    float f = (x.v[0] & 0xFFFF) * (1.0f / 65536.0f);
    return FLOAT_TO_HALF(f);
}

/* Uint4x32 to fp8 uniform - uses first 8 bits */
uint8_t uint4x32_to_fp8_uniform(uint4x32_t x) {
    return (uint8_t)(x.v[0] & 0xFF);
}

/* Vectorized conversion functions that use all 128 bits efficiently */

/* Convert uint4x32 to 4 floats in [0, 1) */
float4_t uint4x32_to_single_uniform_vec(uint4x32_t x) {
    float4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = uint32_to_single_uniform(x.v[i]);
    }
    return result;
}

/* Convert uint4x32 to 2 doubles in [0, 1) */
double2_t uint4x32_to_double_uniform_vec(uint4x32_t x) {
    double2_t result;
    uint64_t combined1 = ((uint64_t)x.v[1] << 32) | x.v[0];
    uint64_t combined2 = ((uint64_t)x.v[3] << 32) | x.v[2];
    result.v[0] = combined1 * (1.0 / 18446744073709551616.0);
    result.v[1] = combined2 * (1.0 / 18446744073709551616.0);
    return result;
}

/* Convert uint4x32 to 4 int32s - full range */
int32x4_t uint4x32_to_int32_uniform_vec(uint4x32_t x) {
    int32x4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = (int32_t)x.v[i];
    }
    return result;
}

/* Convert uint4x32 to 2 int64s - full range */
int64x2_t uint4x32_to_int64_uniform_vec(uint4x32_t x) {
    int64x2_t result;
    result.v[0] = (int64_t)(((uint64_t)x.v[1] << 32) | x.v[0]);
    result.v[1] = (int64_t)(((uint64_t)x.v[3] << 32) | x.v[2]);
    return result;
}


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

/* Convert uint4x32 to 8 uint16s - full range */
uint16x8_t uint4x32_to_uint16_uniform_vec(uint4x32_t x) {
    uint16x8_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i*2 + 0] = (uint16_t)(x.v[i] & 0xFFFF);
        result.v[i*2 + 1] = (uint16_t)((x.v[i] >> 16) & 0xFFFF);
    }
    return result;
}

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

/* Conversion functions from various precisions to uint4x32_t */
uint4x32_t single_to_uint4x32(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));
    uint4x32_t result = {{bits, 0, 0, 0}};
    return result;
}

uint4x32_t double_to_uint4x32(double x) {
    uint64_t bits;
    memcpy(&bits, &x, sizeof(double));
    uint4x32_t result = {{(uint32_t)(bits & 0xFFFFFFFF), (uint32_t)(bits >> 32), 0, 0}};
    return result;
}

uint4x32_t int32_to_uint4x32(int32_t x) {
    /* Spread bits across all 4 components for better entropy with light threefry.
       Without this, consecutive counter values produce nearly identical v[0] outputs
       from 2-round threefry, causing periodicity in random number generation. */
    uint32_t u = (uint32_t)x;
    uint4x32_t result = {{
        u,
        u ^ 0x9E3779B9,              /* golden ratio constant */
        u ^ 0x6C078965,              /* Knuth's MMIX constant */
        u ^ ((u << 16) | (u >> 16))  /* bit rotation */
    }};
    return result;
}

uint4x32_t int64_to_uint4x32(int64_t x) {
    uint64_t bits = (uint64_t)x;
    uint4x32_t result = {{(uint32_t)(bits & 0xFFFFFFFF), (uint32_t)(bits >> 32), 0, 0}};
    return result;
}

uint4x32_t uint32_to_uint4x32(uint32_t x) {
    /* Spread bits across all 4 components for better entropy with light threefry.
       Without this, consecutive counter values produce nearly identical v[0] outputs
       from 2-round threefry, causing periodicity in random number generation. */
    uint4x32_t result = {{
        x,
        x ^ 0x9E3779B9,              /* golden ratio constant */
        x ^ 0x6C078965,              /* Knuth's MMIX constant */
        x ^ ((x << 16) | (x >> 16))  /* bit rotation */
    }};
    return result;
}

uint4x32_t uint64_to_uint4x32(uint64_t x) {
    uint4x32_t result = {{(uint32_t)(x & 0xFFFFFFFF), (uint32_t)(x >> 32), 0, 0}};
    return result;
}

uint4x32_t byte_to_uint4x32(unsigned char x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}

uint4x32_t uint16_to_uint4x32(uint16_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}

uint4x32_t bfloat16_to_uint4x32(uint16_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}

uint4x32_t half_to_uint4x32(uint16_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}

uint4x32_t fp8_to_uint4x32(uint8_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}

/* Pure C conversion functions for use in C backends */

/* BFloat16 to Float conversion (C function) */
float bfloat16_to_single(uint16_t bf16)
{
  /* BFloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
     To convert to float32, we shift left by 16 bits */
  uint32_t f32 = ((uint32_t)bf16) << 16;
  return *((float *)&f32);
}

/* Float to BFloat16 conversion (C function) */
uint16_t single_to_bfloat16(float f)
{
  uint32_t f32 = *((uint32_t *)&f);

  /* Round to nearest even */
  uint32_t rounded = f32 + 0x7FFF + ((f32 >> 16) & 1);
  return (uint16_t)(rounded >> 16);
}

/* Half (Float16) to Float conversion (C function) */
float half_to_single(uint16_t h)
{
  HALF_T half_val = UINT16_TO_HALF(h);
  return HALF_TO_FLOAT(half_val);
}

/* Float to Half (Float16) conversion (C function) */
uint16_t single_to_half(float f)
{
  HALF_T half_val = FLOAT_TO_HALF(f);
  return HALF_TO_UINT16(half_val);
}

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
CAMLprim value arrayjit_threefry4x32_crypto_ocaml(value v_key, value v_counter)
{
  CAMLparam2(v_key, v_counter);
  uint4x32_t key = ocaml_array_to_uint4x32(v_key);
  uint4x32_t counter = ocaml_array_to_uint4x32(v_counter);
  uint4x32_t result = arrayjit_threefry4x32_crypto(key, counter);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_threefry4x32_light_ocaml(value v_key, value v_counter)
{
  CAMLparam2(v_key, v_counter);
  uint4x32_t key = ocaml_array_to_uint4x32(v_key);
  uint4x32_t counter = ocaml_array_to_uint4x32(v_counter);
  uint4x32_t result = arrayjit_threefry4x32_light(key, counter);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

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

/* Half (Float16) to Float conversion (OCaml wrapper) */
CAMLprim value arrayjit_half_to_single(value v_half)
{
  CAMLparam1(v_half);
  uint16_t half = (uint16_t)Int_val(v_half);
  float result = half_to_single(half);
  CAMLreturn(caml_copy_double((double)result));
}

/* Float to Half (Float16) conversion (OCaml wrapper) */
CAMLprim value arrayjit_single_to_half(value v_float)
{
  CAMLparam1(v_float);
  float f = (float)Double_val(v_float);
  uint16_t half = single_to_half(f);
  CAMLreturn(Val_int(half));
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
// for contiguous inner slices, but that adds complexity.
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
  intnat *left = malloc(ndim * sizeof(intnat));
  if (left == NULL)
  {
    caml_failwith("Malloc failed");
  }
  intnat *right = malloc(ndim * sizeof(intnat));
  if (right == NULL)
  {
    free(left);
    caml_failwith("Malloc failed");
  }
  for (int d = 0; d < ndim; d++)
  {
    value pad = Field(v_padding, d);
    left[d] = Long_val(Field(pad, 0));
    right[d] = Long_val(Field(pad, 1));
    if (left[d] < 0 || right[d] < 0)
    {
      free(left);
      free(right);
      caml_failwith("Negative padding");
    }
  }

  // Verify target dimensions match source + padding
  for (int d = 0; d < ndim; d++)
  {
    if (target_ba->dim[d] != source_shape[d] + left[d] + right[d])
    {
      free(left);
      free(right);
      caml_failwith("Target dimensions do not match source + padding");
    }
  }

  // Multi-dimensional index loop
  intnat *indices = calloc(ndim, sizeof(intnat));
  if (indices == NULL)
  {
    free(left);
    free(right);
    caml_failwith("Calloc failed");
  }

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