(* Metal builtin code split into (key, definition, dependencies) triples for filtering *)
let builtins =
  [
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
    ( "threefry_round",
      {|inline void threefry_round(thread uint4 &x, uint r0, uint r1, uint r2, uint r3) {
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
}|},
      [ "rotl32" ] );
    ( "arrayjit_threefry4x32_crypto",
      {|uint4 arrayjit_threefry4x32_crypto(uint4 key, uint4 counter) {
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
}|},
      [
        "THREEFRY_C240";
        "threefry_round";
        "THREEFRY_ROTATION_0_0";
        "THREEFRY_ROTATION_0_1";
        "THREEFRY_ROTATION_0_2";
        "THREEFRY_ROTATION_0_3";
        "THREEFRY_ROTATION_1_0";
        "THREEFRY_ROTATION_1_1";
        "THREEFRY_ROTATION_1_2";
        "THREEFRY_ROTATION_1_3";
      ] );
    ( "arrayjit_threefry4x32_light",
      {|uint4 arrayjit_threefry4x32_light(uint4 key, uint4 counter) {
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
}|},
      [
        "THREEFRY_C240";
        "threefry_round";
        "THREEFRY_ROTATION_0_0";
        "THREEFRY_ROTATION_0_1";
        "THREEFRY_ROTATION_0_2";
        "THREEFRY_ROTATION_0_3";
        "THREEFRY_ROTATION_1_0";
        "THREEFRY_ROTATION_1_1";
        "THREEFRY_ROTATION_1_2";
        "THREEFRY_ROTATION_1_3";
      ] );
    ( "arrayjit_threefry4x32",
      {|uint4 arrayjit_threefry4x32(uint4 key, uint4 counter) {
    /* Default to light version */
    return arrayjit_threefry4x32_light(key, counter);
}|},
      [ "arrayjit_threefry4x32_light" ] );
    ("float4_t", {|struct float4_t { float4 v; };|}, []);
    ("float2_t", {|struct float2_t { float2 v; };|}, []);
    ( "ocannl_shfl_xor",
      (* Butterfly simdgroup shuffle for the [Workgroup_reduce] warp-shuffle rendering
         (gh-ocannl-462): the rendering requires the reduce axis to cover whole simdgroups of the
         threadgroup's .x dimension, so every lane reaches the call. *)
      {|inline float ocannl_shfl_xor(float v, ushort lane_mask) {
    return simd_shuffle_xor(v, lane_mask);
}|},
      [] );
    ("int32x4_t", {|struct int32x4_t { int4 v; };|}, []);
    ("uint32x4_t", {|struct uint32x4_t { uint4 v; };|}, []);
    ("int64x2_t", {|struct int64x2_t { int64_t v[2]; };|}, []);
    ("uint64x2_t", {|struct uint64x2_t { uint64_t v[2]; };|}, []);
    ("int8x16_t", {|struct int8x16_t { int8_t v[16]; };|}, []);
    ("uint16x8_t", {|struct uint16x8_t { uint16_t v[8]; };|}, []);
    ("uint8x16_t", {|struct uint8x16_t { uint8_t v[16]; };|}, []);
    ("half8_t", {|struct half8_t { half v[8]; };|}, []);
    (* [Set_from_vec] assigns block elements directly into the destination cells. Keep bfloat
       elements as values rather than raw uint16 bit patterns, which would be converted numerically
       on assignment (0x3F80 becoming 16256.0 rather than 1.0). *)
    ("bfloat16x8_t", {|struct bfloat16x8_t { bfloat v[8]; };|}, []);
    ( "uint32_to_single_uniform",
      {|inline float uint32_to_single_uniform(uint32_t x) {
    return (x >> 8) * (1.0f / 16777216.0f);
}|},
      [] );
    ( "uint4x32_to_single_uniform",
      {|float uint4x32_to_single_uniform(uint4 x) {
    return uint32_to_single_uniform(x.x);
}|},
      [ "uint32_to_single_uniform" ] );
    ( "uint4x32_to_double_uniform",
      {|float uint4x32_to_double_uniform(uint4 x) {
    /* Fallback to float precision: top 24 bits so the int-to-float conversion is exact and the
       result stays below 1.0 (converting all 64 bits could round up to 2^64, yielding 1.0) */
    uint64_t combined = (uint64_t(x.y) << 32) | x.x;
    return float(combined >> 40) * (1.0f / 16777216.0f);
}|},
      [] );
    ( "uint4x32_to_int32_uniform",
      {|int32_t uint4x32_to_int32_uniform(uint4 x) {
    return int32_t(x.x);
}|},
      [] );
    ( "uint4x32_to_int64_uniform",
      {|int64_t uint4x32_to_int64_uniform(uint4 x) {
    return int64_t((uint64_t(x.y) << 32) | x.x);
}|},
      [] );
    ( "uint4x32_to_uint32_uniform",
      {|uint32_t uint4x32_to_uint32_uniform(uint4 x) {
    return x.x;
}|},
      [] );
    ( "uint4x32_to_uint64_uniform",
      {|uint64_t uint4x32_to_uint64_uniform(uint4 x) {
    return (uint64_t(x.y) << 32) | x.x;
}|},
      [] );
    ( "uint4x32_to_byte_uniform",
      {|int8_t uint4x32_to_byte_uniform(uint4 x) {
    return int8_t(x.x & 0xFF);
}|},
      [] );
    ( "uint4x32_to_uint16_uniform",
      {|uint16_t uint4x32_to_uint16_uniform(uint4 x) {
    return uint16_t(x.x & 0xFFFF);
}|},
      [] );
    ( "uint4x32_to_bfloat16_uniform",
      {|bfloat uint4x32_to_bfloat16_uniform(uint4 x) {
    float f = uint32_to_single_uniform(x.x);
    return bfloat(f);
}|},
      [ "uint32_to_single_uniform" ] );
    ( "uint4x32_to_half_uniform",
      {|half uint4x32_to_half_uniform(uint4 x) {
    float f = uint32_to_single_uniform(x.x);
    return half(f);
}|},
      [ "uint32_to_single_uniform" ] );
    ( "uint4x32_to_fp8_uniform",
      {|uint8_t uint4x32_to_fp8_uniform(uint4 x) {
    return uint8_t(x.x & 0xFF);
}|},
      [] );
    ( "uint4x32_to_single_uniform_vec",
      {|float4_t uint4x32_to_single_uniform_vec(uint4 x) {
    float4_t result;
    result.v.x = uint32_to_single_uniform(x.x);
    result.v.y = uint32_to_single_uniform(x.y);
    result.v.z = uint32_to_single_uniform(x.z);
    result.v.w = uint32_to_single_uniform(x.w);
    return result;
}|},
      [ "float4_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_double_uniform_vec",
      {|float2_t uint4x32_to_double_uniform_vec(uint4 x) {
    /* Top 24 bits per lane pair, see uint4x32_to_double_uniform */
    float2_t result;
    uint64_t combined1 = (uint64_t(x.y) << 32) | x.x;
    uint64_t combined2 = (uint64_t(x.w) << 32) | x.z;
    result.v.x = float(combined1 >> 40) * (1.0f / 16777216.0f);
    result.v.y = float(combined2 >> 40) * (1.0f / 16777216.0f);
    return result;
}|},
      [ "float2_t" ] );
    ( "uint4x32_to_int32_uniform_vec",
      {|int32x4_t uint4x32_to_int32_uniform_vec(uint4 x) {
    int32x4_t result;
    result.v = int4(x);
    return result;
}|},
      [ "int32x4_t" ] );
    ( "uint4x32_to_int64_uniform_vec",
      {|int64x2_t uint4x32_to_int64_uniform_vec(uint4 x) {
    int64x2_t result;
    result.v[0] = (int64_t(x.y) << 32) | x.x;
    result.v[1] = (int64_t(x.w) << 32) | x.z;
    return result;
}|},
      [ "int64x2_t" ] );
    ( "uint4x32_to_uint32_uniform_vec",
      {|uint32x4_t uint4x32_to_uint32_uniform_vec(uint4 x) {
    uint32x4_t result;
    result.v = x;
    return result;
}|},
      [ "uint32x4_t" ] );
    ( "uint4x32_to_uint64_uniform_vec",
      {|uint64x2_t uint4x32_to_uint64_uniform_vec(uint4 x) {
    uint64x2_t result;
    result.v[0] = (uint64_t(x.y) << 32) | x.x;
    result.v[1] = (uint64_t(x.w) << 32) | x.z;
    return result;
}|},
      [ "uint64x2_t" ] );
    ( "uint4x32_to_byte_uniform_vec",
      {|int8x16_t uint4x32_to_byte_uniform_vec(uint4 x) {
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
}|},
      [ "int8x16_t" ] );
    ( "uint4x32_to_uint16_uniform_vec",
      {|uint16x8_t uint4x32_to_uint16_uniform_vec(uint4 x) {
    uint16x8_t result;
    uint4 v = x;
    for (int i = 0; i < 4; i++) {
        uint32_t val = v[i];
        result.v[i*2 + 0] = uint16_t(val & 0xFFFF);
        result.v[i*2 + 1] = uint16_t((val >> 16) & 0xFFFF);
    }
    return result;
}|},
      [ "uint16x8_t" ] );
    ( "uint4x32_to_bfloat16_uniform_vec",
      {|bfloat16x8_t uint4x32_to_bfloat16_uniform_vec(uint4 x) {
    bfloat16x8_t result;
    uint4 v = x;
    for (int i = 0; i < 4; i++) {
        uint32_t val = v[i];
        float f1 = float(val & 0xFFFF) * (1.0f / 65536.0f);
        float f2 = float((val >> 16) & 0xFFFF) * (1.0f / 65536.0f);
        result.v[i*2 + 0] = bfloat(f1);
        result.v[i*2 + 1] = bfloat(f2);
    }
    return result;
}|},
      [ "bfloat16x8_t" ] );
    ( "uint4x32_to_half_uniform_vec",
      {|half8_t uint4x32_to_half_uniform_vec(uint4 x) {
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
}|},
      [ "half8_t" ] );
    ( "uint4x32_to_fp8_uniform_vec",
      {|int8x16_t uint4x32_to_fp8_uniform_vec(uint4 x) {
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
}|},
      [ "int8x16_t" ] );
    (* Lane extraction from the packed uniform conversion (gh-509 task 4): minted by the virtualizer
       to inline packed-uniform results per cell. Implemented via the _vec builtins so the value
       stream is bitwise-identical to the vectorized stores by construction. *)
    ( "uint4x32_to_single_uniform_lane",
      {|float uint4x32_to_single_uniform_lane(uint4 x, int32_t lane) {
    return uint4x32_to_single_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_to_single_uniform_vec" ] );
    ( "uint4x32_to_double_uniform_lane",
      {|float uint4x32_to_double_uniform_lane(uint4 x, int32_t lane) {
    return uint4x32_to_double_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_to_double_uniform_vec" ] );
    ( "uint4x32_to_int32_uniform_lane",
      {|int32_t uint4x32_to_int32_uniform_lane(uint4 x, int32_t lane) {
    return uint4x32_to_int32_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_to_int32_uniform_vec" ] );
    ( "uint4x32_to_int64_uniform_lane",
      {|int64_t uint4x32_to_int64_uniform_lane(uint4 x, int32_t lane) {
    return uint4x32_to_int64_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_to_int64_uniform_vec" ] );
    ( "uint4x32_to_uint32_uniform_lane",
      {|uint32_t uint4x32_to_uint32_uniform_lane(uint4 x, int32_t lane) {
    return uint4x32_to_uint32_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_to_uint32_uniform_vec" ] );
    ( "uint4x32_to_uint64_uniform_lane",
      {|uint64_t uint4x32_to_uint64_uniform_lane(uint4 x, int32_t lane) {
    return uint4x32_to_uint64_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_to_uint64_uniform_vec" ] );
    ( "uint4x32_to_byte_uniform_lane",
      {|int8_t uint4x32_to_byte_uniform_lane(uint4 x, int32_t lane) {
    return uint4x32_to_byte_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_to_byte_uniform_vec" ] );
    ( "uint4x32_to_uint16_uniform_lane",
      {|uint16_t uint4x32_to_uint16_uniform_lane(uint4 x, int32_t lane) {
    return uint4x32_to_uint16_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_to_uint16_uniform_vec" ] );
    ( "uint4x32_to_bfloat16_uniform_lane",
      {|bfloat uint4x32_to_bfloat16_uniform_lane(uint4 x, int32_t lane) {
    return uint4x32_to_bfloat16_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_to_bfloat16_uniform_vec" ] );
    ( "uint4x32_to_half_uniform_lane",
      {|half uint4x32_to_half_uniform_lane(uint4 x, int32_t lane) {
    return uint4x32_to_half_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_to_half_uniform_vec" ] );
    ( "uint4x32_to_fp8_uniform_lane",
      {|int8_t uint4x32_to_fp8_uniform_lane(uint4 x, int32_t lane) {
    return uint4x32_to_fp8_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_to_fp8_uniform_vec" ] );
    ( "single_to_uint4x32",
      {|uint4 single_to_uint4x32(float x) {
    uint32_t bits = as_type<uint32_t>(x);
    return uint4(bits, 0, 0, 0);
}|},
      [] );
    ( "double_to_uint4x32",
      {|uint4 double_to_uint4x32(float x) {
    /* Metal doesn't have native double support, use float fallback */
    uint32_t bits = as_type<uint32_t>(x);
    return uint4(bits, 0, 0, 0);
}|},
      [] );
    ( "int32_to_uint4x32",
      {|uint4 int32_to_uint4x32(int32_t x) {
    /* Spread bits across all 4 components for better entropy with light threefry.
       Without this, consecutive counter values produce nearly identical v[0] outputs
       from 2-round threefry, causing periodicity in random number generation. */
    uint32_t u = uint32_t(x);
    return uint4(
        u,
        u ^ 0x9E3779B9,              /* golden ratio constant */
        u ^ 0x6C078965,              /* Knuth's MMIX constant */
        u ^ ((u << 16) | (u >> 16))  /* bit rotation */
    );
}|},
      [] );
    ( "int64_to_uint4x32",
      {|uint4 int64_to_uint4x32(int64_t x) {
    uint64_t bits = uint64_t(x);
    return uint4(uint32_t(bits & 0xFFFFFFFF), uint32_t(bits >> 32), 0, 0);
}|},
      [] );
    ( "uint32_to_uint4x32",
      {|uint4 uint32_to_uint4x32(uint32_t x) {
    /* Spread bits across all 4 components for better entropy with light threefry.
       Without this, consecutive counter values produce nearly identical v[0] outputs
       from 2-round threefry, causing periodicity in random number generation. */
    return uint4(
        x,
        x ^ 0x9E3779B9,              /* golden ratio constant */
        x ^ 0x6C078965,              /* Knuth's MMIX constant */
        x ^ ((x << 16) | (x >> 16))  /* bit rotation */
    );
}|},
      [] );
    ( "uint64_to_uint4x32",
      {|uint4 uint64_to_uint4x32(uint64_t x) {
    return uint4(uint32_t(x & 0xFFFFFFFF), uint32_t(x >> 32), 0, 0);
}|},
      [] );
    ( "byte_to_uint4x32",
      {|uint4 byte_to_uint4x32(int8_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}|},
      [] );
    ( "uint16_to_uint4x32",
      {|uint4 uint16_to_uint4x32(uint16_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}|},
      [] );
    ( "bfloat16_to_uint4x32",
      {|uint4 bfloat16_to_uint4x32(bfloat x) {
    return uint4(uint32_t(as_type<uint16_t>(x)), 0, 0, 0);
}|},
      [] );
    ( "half_to_uint4x32",
      {|uint4 half_to_uint4x32(uint16_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}|},
      [] );
    ( "fp8_to_uint4x32",
      {|uint4 fp8_to_uint4x32(uint8_t x) {
    return uint4(uint32_t(x), 0, 0, 0);
}|},
      [] );
    (* The e5m2 software codec. MSL has no fp8 type, so an fp8 tensor is stored as a byte and its
       arithmetic runs in f32 ([Metal_backend.C_syntax_config.compute_prec]): these two are the
       whole storage/compute seam for fp8 on Metal, called at every load and every store.

       Bit manipulation rather than the [ldexp]/[frexp] arithmetic of the C twins in
       [Builtins_cc]/[builtins.c]: Metal compiles with fast math by default, under which the
       infinity and NaN branches of a float-arithmetic codec are not reliable. The results agree
       with the C twins for all 256 codes and for every float — including their tie-away-from-zero
       rounding, their flush of everything below the smallest subnormal, and their unsigned zero —
       which is what lets an fp8 tensor written by one backend be read by another
       ([test_fp8_codec_parity]). *)
    ( "fp8_to_single",
      {|/* FP8 E5M2 (1 sign, 5 exponent, 2 mantissa bits) to float. */
inline float fp8_to_single(uint8_t v) {
    uint32_t bits = uint32_t(v);
    uint32_t sign = (bits & 0x80u) << 24;
    uint32_t exp = (bits >> 2) & 0x1Fu;
    uint32_t mant = bits & 0x3u;
    if (exp == 0x1Fu) {
        /* Infinity, or the positive quiet NaN the C codec returns for any nonzero payload. */
        return as_type<float>(mant != 0u ? 0x7FC00000u : (sign | 0x7F800000u));
    }
    if (exp == 0u) {
        /* Zero and subnormals: mant * 2^-16, exact in f32 (and the signed zero for mant = 0). */
        return as_type<float>(sign | as_type<uint32_t>(float(mant) * 1.52587890625e-05f));
    }
    /* Normals share f32's field order: rebias 15 -> 127 and put the 2 mantissa bits on top. */
    return as_type<float>(sign | ((exp + 112u) << 23) | (mant << 21));
}|},
      [] );
    ( "single_to_fp8",
      {|/* Float to FP8 E5M2: round to nearest even, subnormals rounded rather than flushed,
   signed zero preserved, finite overflow saturating to the max finite. Matches the C twins in
   builtins.c / Builtins_cc, and the native GPU fp8 types they were aligned to. */
inline uint8_t single_to_fp8(float f) {
    uint32_t bits = as_type<uint32_t>(f);
    uint32_t sign = (bits >> 24) & 0x80u;
    uint32_t e32 = (bits >> 23) & 0xFFu;
    uint32_t m32 = bits & 0x7FFFFFu;
    /* Infinity and NaN keep their sign; a NaN takes the all-mantissa code. */
    if (e32 == 0xFFu) { return uint8_t(sign | (m32 != 0u ? 0x7Fu : 0x7Cu)); }
    /* Zero, and f32 subnormals, which are far below e5m2's smallest subnormal. */
    if (e32 == 0u) { return uint8_t(sign); }
    int e = int(e32) - 112; /* the e5m2 exponent field: rebias 127 -> 15 */
    if (e >= 31) { return uint8_t(sign | 0x7Bu); } /* saturate to the largest finite, 57344 */
    if (e <= 0) {
        /* Subnormal target: the value is q * 2^-16 with q in [0, 4), rounded to nearest even. */
        uint32_t sig = 0x800000u | m32;
        int shift = 22 - e;
        if (shift > 24) { return uint8_t(sign); } /* below half the smallest subnormal */
        uint32_t q = sig >> shift;
        uint32_t rest = sig & ((1u << shift) - 1u);
        uint32_t tie = 1u << (shift - 1);
        if (rest > tie || (rest == tie && (q & 1u))) { q++; }
        if (q > 3u) { return uint8_t(sign | 0x04u); } /* carried into the smallest normal */
        return uint8_t(sign | q);
    }
    uint32_t mant = m32 >> 21;
    uint32_t rest = m32 & 0x1FFFFFu;
    if (rest > 0x100000u || (rest == 0x100000u && (mant & 1u))) { mant++; }
    if (mant > 3u) {
        mant = 0u;
        e += 1;
        if (e >= 31) { return uint8_t(sign | 0x7Bu); }
    }
    return uint8_t(sign | (uint32_t(e) << 2) | mant);
}|},
      [] );
  ]

let builtins =
  Builtins_cc.integer_power_builtins ~prefix:"inline" ~supports_double:false ~dialect:`Metal
  @ builtins
