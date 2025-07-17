#include <stdint.h>
#include <string.h>

typedef struct {
    uint32_t v[4];
} uint4x32_t;

/* Threefry4x32 constants */
static const uint32_t THREEFRY_C240 = 0x1BD11BDA;

/* Rotation constants for Threefry4x32 */
static const unsigned int THREEFRY_ROTATION_0_0 = 13;
static const unsigned int THREEFRY_ROTATION_0_1 = 15;
static const unsigned int THREEFRY_ROTATION_0_2 = 26;
static const unsigned int THREEFRY_ROTATION_0_3 = 6;
static const unsigned int THREEFRY_ROTATION_1_0 = 17;
static const unsigned int THREEFRY_ROTATION_1_1 = 29;
static const unsigned int THREEFRY_ROTATION_1_2 = 16;
static const unsigned int THREEFRY_ROTATION_1_3 = 24;

/* Rotate left function */
static inline uint32_t rotl32(uint32_t x, unsigned int n) {
    return (x << n) | (x >> (32 - n));
}

/* Threefry4x32 round function */
static inline void threefry_round(uint32_t x[4], unsigned int r0, unsigned int r1, unsigned int r2, unsigned int r3) {
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
uint4x32_t arrayjit_threefry4x32(uint4x32_t key, uint4x32_t counter) {
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
static inline float uint32_to_float_uniform(uint32_t x) {
    /* Use upper 24 bits for float mantissa (23 bits + implicit 1) */
    return (x >> 8) * (1.0f / 16777216.0f);
}

/* Convert to double in [0, 1) */
static inline double uint32_to_double_uniform(uint32_t x) {
    return x * (1.0 / 4294967296.0);
}

/* Uint4x32 to float32 uniform - uses first 32 bits */
float uint4x32_to_fp32_uniform(uint4x32_t x) {
    return uint32_to_float_uniform(x.v[0]);
}

/* Uint4x32 to float64 uniform - uses first 64 bits */
double uint4x32_to_fp64_uniform(uint4x32_t x) {
    uint64_t combined = ((uint64_t)x.v[1] << 32) | x.v[0];
    return combined * (1.0 / 18446744073709551616.0);
}

/* Uint4x32 to int32 uniform - full range */
int32_t uint4x32_to_i32_uniform(uint4x32_t x) {
    return (int32_t)x.v[0];
}

/* Uint4x32 to int64 uniform - full range */
int64_t uint4x32_to_i64_uniform(uint4x32_t x) {
    return (int64_t)(((uint64_t)x.v[1] << 32) | x.v[0]);
}

/* Uint4x32 to uint32 uniform - full range */
uint32_t uint4x32_to_u32_uniform(uint4x32_t x) {
    return x.v[0];
}

/* Uint4x32 to uint64 uniform - full range */
uint64_t uint4x32_to_u64_uniform(uint4x32_t x) {
    return ((uint64_t)x.v[1] << 32) | x.v[0];
}

/* Uint4x32 to int8 uniform - full range */
int8_t uint4x32_to_i8_uniform(uint4x32_t x) {
    return (int8_t)(x.v[0] & 0xFF);
}

/* Uint4x32 to uint8 uniform - full range */
uint8_t uint4x32_to_u8_uniform(uint4x32_t x) {
    return (uint8_t)(x.v[0] & 0xFF);
}

/* Uint4x32 to bfloat16 uniform - uses first 16 bits */
uint16_t uint4x32_to_bf16_uniform(uint4x32_t x) {
    /* Convert to float first, then to bfloat16 */
    float f = uint32_to_float_uniform(x.v[0]);
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

/* Uint4x32 to float16 uniform - uses first 16 bits */
uint16_t uint4x32_to_fp16_uniform(uint4x32_t x) {
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