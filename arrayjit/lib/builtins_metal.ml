(* Metal builtin code split into (key, definition, dependencies) triples for filtering *)
let builtins = [
  ("METAL_HEADERS", {|#include <metal_stdlib>
using namespace metal;|}, []);

  ("THREEFRY_C240", {|constant uint32_t THREEFRY_C240 = 0x1BD11BDA;|}, []);

  ("THREEFRY_ROTATION_0_0", {|constant uint THREEFRY_ROTATION_0_0 = 13;|}, []);
  ("THREEFRY_ROTATION_0_1", {|constant uint THREEFRY_ROTATION_0_1 = 15;|}, []);
  ("THREEFRY_ROTATION_0_2", {|constant uint THREEFRY_ROTATION_0_2 = 26;|}, []);
  ("THREEFRY_ROTATION_0_3", {|constant uint THREEFRY_ROTATION_0_3 = 6;|}, []);
  ("THREEFRY_ROTATION_1_0", {|constant uint THREEFRY_ROTATION_1_0 = 17;|}, []);
  ("THREEFRY_ROTATION_1_1", {|constant uint THREEFRY_ROTATION_1_1 = 29;|}, []);
  ("THREEFRY_ROTATION_1_2", {|constant uint THREEFRY_ROTATION_1_2 = 16;|}, []);
  ("THREEFRY_ROTATION_1_3", {|constant uint THREEFRY_ROTATION_1_3 = 24;|}, []);

  ("rotl32", {|inline uint32_t rotl32(uint32_t x, uint n) {
    return rotate(x, n);
}|}, []);

  ("threefry_round", {|inline void threefry_round(thread uint4 &x, uint r0, uint r1, uint r2, uint r3) {
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
}|}, ["rotl32"]);

  ("arrayjit_threefry4x32_crypto", {|uint4 arrayjit_threefry4x32_crypto(uint4 key, uint4 counter) {
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
}|}, ["THREEFRY_C240"; "threefry_round"; "THREEFRY_ROTATION_0_0"; "THREEFRY_ROTATION_0_1"; 
      "THREEFRY_ROTATION_0_2"; "THREEFRY_ROTATION_0_3"; "THREEFRY_ROTATION_1_0"; 
      "THREEFRY_ROTATION_1_1"; "THREEFRY_ROTATION_1_2"; "THREEFRY_ROTATION_1_3"]);

  ("arrayjit_threefry4x32_light", {|uint4 arrayjit_threefry4x32_light(uint4 key, uint4 counter) {
    uint4 x = counter;
    uint4 k = key;
    
    /* Compute ks[4] */
    uint32_t ks4 = k.x ^ k.y ^ k.z ^ k.w ^ THREEFRY_C240;
    
    /* Initial key injection */
    x += k;
    
    /* Only 2 rounds for light version */
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, 
                      THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, 
                      THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Final key injection after round 2 */
    x.x += k.y;
    x.y += k.z;
    x.z += k.w;
    x.w += ks4 + 1;
    
    return x;
}|}, ["THREEFRY_C240"; "threefry_round"; "THREEFRY_ROTATION_0_0"; "THREEFRY_ROTATION_0_1"; 
      "THREEFRY_ROTATION_0_2"; "THREEFRY_ROTATION_0_3"; "THREEFRY_ROTATION_1_0"; 
      "THREEFRY_ROTATION_1_1"; "THREEFRY_ROTATION_1_2"; "THREEFRY_ROTATION_1_3"]);

  ("arrayjit_threefry4x32", {|uint4 arrayjit_threefry4x32(uint4 key, uint4 counter) {
    /* Default to light version */
    return arrayjit_threefry4x32_light(key, counter);
}|}, ["arrayjit_threefry4x32_light"]);

  ("float4_t", {|struct float4_t { float4 v; };|}, []);
  ("float2_t", {|struct float2_t { float2 v; };|}, []);
  ("int32x4_t", {|struct int32x4_t { int4 v; };|}, []);
  ("int64x2_t", {|struct int64x2_t { int64_t v[2]; };|}, []);
  ("uint64x2_t", {|struct uint64x2_t { uint64_t v[2]; };|}, []);
  ("int8x16_t", {|struct int8x16_t { int8_t v[16]; };|}, []);
  ("uint16x8_t", {|struct uint16x8_t { uint16_t v[8]; };|}, []);
  ("uint8x16_t", {|struct uint8x16_t { uint8_t v[16]; };|}, []);
  ("half8_t", {|struct half8_t { half v[8]; };|}, []);

  ("uint32_to_single_uniform", {|inline float uint32_to_single_uniform(uint32_t x) {
    return (x >> 8) * (1.0f / 16777216.0f);
}|}, []);

  ("uint4x32_to_single_uniform", {|float uint4x32_to_single_uniform(uint4 x) {
    return uint32_to_single_uniform(x.x);
}|}, ["uint32_to_single_uniform"]);

  ("uint4x32_to_double_uniform", {|float uint4x32_to_double_uniform(uint4 x) {
    /* Fallback to float precision */
    uint64_t combined = (uint64_t(x.y) << 32) | x.x;
    return float(combined) * (1.0f / 18446744073709551616.0f);
}|}, []);

  ("uint4x32_to_int32_uniform", {|int32_t uint4x32_to_int32_uniform(uint4 x) {
    return int32_t(x.x);
}|}, []);

  ("uint4x32_to_int64_uniform", {|int64_t uint4x32_to_int64_uniform(uint4 x) {
    return int64_t((uint64_t(x.y) << 32) | x.x);
}|}, []);

  ("uint4x32_to_uint32_uniform", {|uint32_t uint4x32_to_uint32_uniform(uint4 x) {
    return x.x;
}|}, []);

  ("uint4x32_to_uint64_uniform", {|uint64_t uint4x32_to_uint64_uniform(uint4 x) {
    return (uint64_t(x.y) << 32) | x.x;
}|}, []);

  ("uint4x32_to_byte_uniform", {|int8_t uint4x32_to_byte_uniform(uint4 x) {
    return int8_t(x.x & 0xFF);
}|}, []);

  ("uint4x32_to_uint16_uniform", {|uint16_t uint4x32_to_uint16_uniform(uint4 x) {
    return uint16_t(x.x & 0xFFFF);
}|}, []);

  ("uint4x32_to_bfloat16_uniform", {|uint16_t uint4x32_to_bfloat16_uniform(uint4 x) {
    float f = uint32_to_single_uniform(x.x);
    return uint16_t(as_type<uint32_t>(f) >> 16);
}|}, ["uint32_to_single_uniform"]);

  ("uint4x32_to_half_uniform", {|half uint4x32_to_half_uniform(uint4 x) {
    float f = uint32_to_single_uniform(x.x);
    return half(f);
}|}, ["uint32_to_single_uniform"]);

  ("uint4x32_to_fp8_uniform", {|uint8_t uint4x32_to_fp8_uniform(uint4 x) {
    return uint8_t(x.x & 0xFF);
}|}, []);

  ("uint4x32_to_single_uniform_vec", {|float4_t uint4x32_to_single_uniform_vec(uint4 x) {
    float4_t result;
    result.v.x = uint32_to_single_uniform(x.x);
    result.v.y = uint32_to_single_uniform(x.y);
    result.v.z = uint32_to_single_uniform(x.z);
    result.v.w = uint32_to_single_uniform(x.w);
    return result;
}|}, ["float4_t"; "uint32_to_single_uniform"]);

  ("uint4x32_to_double_uniform_vec", {|float2_t uint4x32_to_double_uniform_vec(uint4 x) {
    float2_t result;
    uint64_t combined1 = (uint64_t(x.y) << 32) | x.x;
    uint64_t combined2 = (uint64_t(x.w) << 32) | x.z;
    result.v.x = float(combined1) * (1.0f / 18446744073709551616.0f);
    result.v.y = float(combined2) * (1.0f / 18446744073709551616.0f);
    return result;
}|}, ["float2_t"]);

  ("uint4x32_to_int32_uniform_vec", {|int32x4_t uint4x32_to_int32_uniform_vec(uint4 x) {
    int32x4_t result;
    result.v = int4(x);
    return result;
}|}, ["int32x4_t"]);

  ("uint4x32_to_int64_uniform_vec", {|int64x2_t uint4x32_to_int64_uniform_vec(uint4 x) {
    int64x2_t result;
    result.v[0] = (int64_t(x.y) << 32) | x.x;
    result.v[1] = (int64_t(x.w) << 32) | x.z;
    return result;
}|}, ["int64x2_t"]);

  ("uint4x32_to_uint32_uniform_vec", {|uint4 uint4x32_to_uint32_uniform_vec(uint4 x) {
    return x;
}|}, []);

  ("uint4x32_to_uint64_uniform_vec", {|uint64x2_t uint4x32_to_uint64_uniform_vec(uint4 x) {
    uint64x2_t result;
    result.v[0] = (uint64_t(x.y) << 32) | x.x;
    result.v[1] = (uint64_t(x.w) << 32) | x.z;
    return result;
}|}, ["uint64x2_t"]);

  ("uint4x32_to_byte_uniform_vec", {|int8x16_t uint4x32_to_byte_uniform_vec(uint4 x) {
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
}|}, ["int8x16_t"]);

  ("uint4x32_to_uint16_uniform_vec", {|uint16x8_t uint4x32_to_uint16_uniform_vec(uint4 x) {
    uint16x8_t result;
    uint4 v = x;
    for (int i = 0; i < 4; i++) {
        uint32_t val = v[i];
        result.v[i*2 + 0] = uint16_t(val & 0xFFFF);
        result.v[i*2 + 1] = uint16_t((val >> 16) & 0xFFFF);
    }
    return result;
}|}, ["uint16x8_t"]);

  ("uint4x32_to_bfloat16_uniform_vec", {|uint16x8_t uint4x32_to_bfloat16_uniform_vec(uint4 x) {
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
}|}, ["uint16x8_t"]);

  ("uint4x32_to_half_uniform_vec", {|half8_t uint4x32_to_half_uniform_vec(uint4 x) {
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
}|}, ["half8_t"]);

  ("uint4x32_to_fp8_uniform_vec", {|uint8x16_t uint4x32_to_fp8_uniform_vec(uint4 x) {
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
}|}, ["uint8x16_t"]);

  ("single_to_uint4x32", {|uint4 single_to_uint4x32(float x) {
    uint32_t bits = as_type<uint32_t>(x);
    return uint4(bits, 0, 0, 0);
}|}, []);

  ("double_to_uint4x32", {|uint4 double_to_uint4x32(float x) {
    /* Metal doesn't have native double support, use float fallback */
    uint32_t bits = as_type<uint32_t>(x);
    return uint4(bits, 0, 0, 0);
}|}, []);

  ("int32_to_uint4x32", {|uint4 int32_to_uint4x32(int32_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}|}, []);

  ("int64_to_uint4x32", {|uint4 int64_to_uint4x32(int64_t x) {
    uint64_t bits = uint64_t(x);
    return uint4(uint32_t(bits & 0xFFFFFFFF), uint32_t(bits >> 32), 0, 0);
}|}, []);

  ("uint32_to_uint4x32", {|uint4 uint32_to_uint4x32(uint32_t x) {
    return uint4(x, 0, 0, 0);
}|}, []);

  ("uint64_to_uint4x32", {|uint4 uint64_to_uint4x32(uint64_t x) {
    return uint4(uint32_t(x & 0xFFFFFFFF), uint32_t(x >> 32), 0, 0);
}|}, []);

  ("byte_to_uint4x32", {|uint4 byte_to_uint4x32(int8_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}|}, []);

  ("uint16_to_uint4x32", {|uint4 uint16_to_uint4x32(uint16_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}|}, []);

  ("bfloat16_to_uint4x32", {|uint4 bfloat16_to_uint4x32(uint16_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}|}, []);

  ("half_to_uint4x32", {|uint4 half_to_uint4x32(uint16_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}|}, []);

  ("fp8_to_uint4x32", {|uint4 fp8_to_uint4x32(uint8_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}|}, []);
]
