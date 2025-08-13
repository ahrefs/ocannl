let source =
  {|
#include <metal_stdlib>
using namespace metal;

/* Threefry4x32 constants */
constant uint32_t THREEFRY_C240 = 0x1BD11BDA;

/* Rotation constants for Threefry4x32 */
constant uint THREEFRY_ROTATION_0_0 = 13;
constant uint THREEFRY_ROTATION_0_1 = 15;
constant uint THREEFRY_ROTATION_0_2 = 26;
constant uint THREEFRY_ROTATION_0_3 = 6;
constant uint THREEFRY_ROTATION_1_0 = 17;
constant uint THREEFRY_ROTATION_1_1 = 29;
constant uint THREEFRY_ROTATION_1_2 = 16;
constant uint THREEFRY_ROTATION_1_3 = 24;

/* Metal rotate left using built-in rotate function */
inline uint32_t rotl32(uint32_t x, uint n) {
    return rotate(x, n);
}

/* Threefry4x32 round function using SIMD operations */
inline void threefry_round(thread uint4 &x, uint r0, uint r1, uint r2, uint r3) {
    x.x += x.y; x.y = rotl32(x.y, r0); x.y ^= x.x;
    x.z += x.w; x.w = rotl32(x.w, r1); x.w ^= x.z;
    
    uint32_t tmp = x.y;
    x.y = x.w;
    x.w = tmp;
    
    x.x += x.y; x.y = rotl32(x.y, r2); x.y ^= x.x;
    x.z += x.w; x.w = rotl32(x.w, r3); x.w ^= x.z;
    
    tmp = x.y;
    x.y = x.w;
    x.w = tmp;
}

/* Threefry4x32 implementation - 20 rounds */
uint4 arrayjit_threefry4x32(uint4 key, uint4 counter) {
    uint4 x = counter;
    uint4 k = key;
    
    /* Compute ks[4] */
    uint32_t ks4 = k.x ^ k.y ^ k.z ^ k.w ^ THREEFRY_C240;
    
    /* Initial key injection */
    x += k;
    
    /* 20 rounds with key injections */
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, 
                      THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, 
                      THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, 
                      THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, 
                      THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Key injection after round 4 */
    x.x += k.y;
    x.y += k.z;
    x.z += k.w;
    x.w += ks4 + 1;
    
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, 
                      THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, 
                      THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, 
                      THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, 
                      THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Key injection after round 8 */
    x.x += k.z;
    x.y += k.w;
    x.z += ks4;
    x.w += k.x + 2;
    
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, 
                      THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, 
                      THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, 
                      THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, 
                      THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Key injection after round 12 */
    x.x += k.w;
    x.y += ks4;
    x.z += k.x;
    x.w += k.y + 3;
    
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, 
                      THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, 
                      THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, 
                      THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, 
                      THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Key injection after round 16 */
    x.x += ks4;
    x.y += k.x;
    x.z += k.y;
    x.w += k.z + 4;
    
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, 
                      THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, 
                      THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, 
                      THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, 
                      THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Final key injection after round 20 */
    x += k;
    x.w += 5;
    
    return x;
}

/* Vector types for efficient extraction of multiple values */
struct float4_t { float4 v; };
struct float2_t { float2 v; };  /* Using float2 since Metal lacks double */
struct int32x4_t { int4 v; };
struct int64x2_t { int64_t v[2]; };
struct uint64x2_t { uint64_t v[2]; };
struct int8x16_t { int8_t v[16]; };
struct uint16x8_t { uint16_t v[8]; };
struct uint8x16_t { uint8_t v[16]; };
struct half8_t { half v[8]; };

/* Conversion functions from uint4x32 to various precisions uniformly */
// These return vectors to efficiently use all random bits

/* Convert to float in [0, 1) */
inline float uint32_to_single_uniform(uint32_t x) {
    return (x >> 8) * (1.0f / 16777216.0f);
}

/* Uint4x32 to float32 uniform */
float uint4x32_to_single_uniform(uint4 x) {
    return uint32_to_single_uniform(x.x);
}

/* Uint4x32 to float64 uniform - Metal doesn't have native double support */
float uint4x32_to_double_uniform(uint4 x) {
    /* Fallback to float precision */
    uint64_t combined = (uint64_t(x.y) << 32) | x.x;
    return float(combined) * (1.0f / 18446744073709551616.0f);
}

/* Uint4x32 to int32 uniform */
int32_t uint4x32_to_int32_uniform(uint4 x) {
    return int32_t(x.x);
}

/* Uint4x32 to int64 uniform */
int64_t uint4x32_to_int64_uniform(uint4 x) {
    return int64_t((uint64_t(x.y) << 32) | x.x);
}

/* Uint4x32 to uint32 uniform */
uint32_t uint4x32_to_uint32_uniform(uint4 x) {
    return x.x;
}

/* Uint4x32 to uint64 uniform */
uint64_t uint4x32_to_uint64_uniform(uint4 x) {
    return (uint64_t(x.y) << 32) | x.x;
}

/* Uint4x32 to byte uniform */
int8_t uint4x32_to_byte_uniform(uint4 x) {
    return int8_t(x.x & 0xFF);
}

/* Uint4x32 to uint16 uniform */
uint16_t uint4x32_to_uint16_uniform(uint4 x) {
    return uint16_t(x.x & 0xFFFF);
}

/* Uint4x32 to bfloat16 uniform */
uint16_t uint4x32_to_bfloat16_uniform(uint4 x) {
    float f = uint32_to_single_uniform(x.x);
    return uint16_t(as_type<uint32_t>(f) >> 16);
}

/* Uint4x32 to float16 uniform */
half uint4x32_to_half_uniform(uint4 x) {
    float f = uint32_to_single_uniform(x.x);
    return half(f);
}

/* Uint4x32 to fp8 uniform */
uint8_t uint4x32_to_fp8_uniform(uint4 x) {
    return uint8_t(x.x & 0xFF);
}

/* Vectorized conversion functions that use all 128 bits efficiently */

/* Convert uint4x32 to 4 floats in [0, 1) */
float4_t uint4x32_to_single_uniform_vec(uint4 x) {
    float4_t result;
    result.v.x = uint32_to_single_uniform(x.x);
    result.v.y = uint32_to_single_uniform(x.y);
    result.v.z = uint32_to_single_uniform(x.z);
    result.v.w = uint32_to_single_uniform(x.w);
    return result;
}

/* Convert uint4x32 to 2 floats in [0, 1) - Metal lacks double precision */
float2_t uint4x32_to_double_uniform_vec(uint4 x) {
    float2_t result;
    uint64_t combined1 = (uint64_t(x.y) << 32) | x.x;
    uint64_t combined2 = (uint64_t(x.w) << 32) | x.z;
    result.v.x = float(combined1) * (1.0f / 18446744073709551616.0f);
    result.v.y = float(combined2) * (1.0f / 18446744073709551616.0f);
    return result;
}

/* Convert uint4x32 to 4 int32s - full range */
int32x4_t uint4x32_to_int32_uniform_vec(uint4 x) {
    int32x4_t result;
    result.v = int4(x);
    return result;
}

/* Convert uint4x32 to 2 int64s - full range */
int64x2_t uint4x32_to_int64_uniform_vec(uint4 x) {
    int64x2_t result;
    result.v[0] = (int64_t(x.y) << 32) | x.x;
    result.v[1] = (int64_t(x.w) << 32) | x.z;
    return result;
}

/* Convert uint4x32 to 4 uint32s - full range */
uint4 uint4x32_to_uint32_uniform_vec(uint4 x) {
    return x;
}

/* Convert uint4x32 to 2 uint64s - full range */
uint64x2_t uint4x32_to_uint64_uniform_vec(uint4 x) {
    uint64x2_t result;
    result.v[0] = (uint64_t(x.y) << 32) | x.x;
    result.v[1] = (uint64_t(x.w) << 32) | x.z;
    return result;
}

/* Convert uint4x32 to 16 int8s - full range */
int8x16_t uint4x32_to_byte_uniform_vec(uint4 x) {
    int8x16_t result;
    uint4 v = x;
    for (int i = 0; i < 4; i++) {
        uint32_t val = v[i];
        result.v[i*4 + 0] = int8_t(val & 0xFF);
        result.v[i*4 + 1] = int8_t((val >> 8) & 0xFF);
        result.v[i*4 + 2] = int8_t((val >> 16) & 0xFF);
        result.v[i*4 + 3] = int8_t((val >> 24) & 0xFF);
    }
    return result;
}

/* Convert uint4x32 to 8 uint16s - full range */
uint16x8_t uint4x32_to_uint16_uniform_vec(uint4 x) {
    uint16x8_t result;
    uint4 v = x;
    for (int i = 0; i < 4; i++) {
        uint32_t val = v[i];
        result.v[i*2 + 0] = uint16_t(val & 0xFFFF);
        result.v[i*2 + 1] = uint16_t((val >> 16) & 0xFFFF);
    }
    return result;
}

/* Convert uint4x32 to 8 bfloat16s uniform */
uint16x8_t uint4x32_to_bfloat16_uniform_vec(uint4 x) {
    uint16x8_t result;
    uint4 v = x;
    for (int i = 0; i < 4; i++) {
        uint32_t val = v[i];
        float f1 = float(val & 0xFFFF) * (1.0f / 65536.0f);
        float f2 = float((val >> 16) & 0xFFFF) * (1.0f / 65536.0f);
        result.v[i*2 + 0] = uint16_t(as_type<uint32_t>(f1) >> 16);
        result.v[i*2 + 1] = uint16_t(as_type<uint32_t>(f2) >> 16);
    }
    return result;
}

/* Convert uint4x32 to 8 float16s uniform */
half8_t uint4x32_to_half_uniform_vec(uint4 x) {
    half8_t result;
    uint4 v = x;
    for (int i = 0; i < 4; i++) {
        uint32_t val = v[i];
        float f1 = float(val & 0xFFFF) * (1.0f / 65536.0f);
        float f2 = float((val >> 16) & 0xFFFF) * (1.0f / 65536.0f);
        result.v[i*2 + 0] = half(f1);
        result.v[i*2 + 1] = half(f2);
    }
    return result;
}

/* Convert uint4x32 to 16 fp8s uniform */
uint8x16_t uint4x32_to_fp8_uniform_vec(uint4 x) {
    uint8x16_t result;
    uint4 v = x;
    for (int i = 0; i < 4; i++) {
        uint32_t val = v[i];
        result.v[i*4 + 0] = uint8_t(val & 0xFF);
        result.v[i*4 + 1] = uint8_t((val >> 8) & 0xFF);
        result.v[i*4 + 2] = uint8_t((val >> 16) & 0xFF);
        result.v[i*4 + 3] = uint8_t((val >> 24) & 0xFF);
    }
    return result;
}

/* Conversion functions from various precisions to uint4x32 */
uint4 single_to_uint4x32(float x) {
    uint32_t bits = as_type<uint32_t>(x);
    return uint4(bits, 0, 0, 0);
}

uint4 double_to_uint4x32(float x) {
    /* Metal doesn't have native double support, use float fallback */
    uint32_t bits = as_type<uint32_t>(x);
    return uint4(bits, 0, 0, 0);
}

uint4 int32_to_uint4x32(int32_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}

uint4 int64_to_uint4x32(int64_t x) {
    uint64_t bits = uint64_t(x);
    return uint4(uint32_t(bits & 0xFFFFFFFF), uint32_t(bits >> 32), 0, 0);
}

uint4 uint32_to_uint4x32(uint32_t x) {
    return uint4(x, 0, 0, 0);
}

uint4 uint64_to_uint4x32(uint64_t x) {
    return uint4(uint32_t(x & 0xFFFFFFFF), uint32_t(x >> 32), 0, 0);
}

uint4 byte_to_uint4x32(int8_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}

uint4 uint16_to_uint4x32(uint16_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}

uint4 bfloat16_to_uint4x32(uint16_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}

uint4 half_to_uint4x32(uint16_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}

uint4 fp8_to_uint4x32(uint8_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}
|}
