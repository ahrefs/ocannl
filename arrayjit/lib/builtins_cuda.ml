(* CUDA builtin code split into (key, definition, dependencies) triples for filtering *)
let builtins =
  [
    ("uint4x32_t", {|typedef struct {
    unsigned int v[4];
} uint4x32_t;|}, []);
    ("float4_t", {|typedef struct { float v[4]; } float4_t;|}, []);
    ("double2_t", {|typedef struct { double v[2]; } double2_t;|}, []);
    ("int32x4_t", {|typedef struct { int v[4]; } int32x4_t;|}, []);
    ("int64x2_t", {|typedef struct { long long v[2]; } int64x2_t;|}, []);
    ("int8x16_t", {|typedef struct { signed char v[16]; } int8x16_t;|}, []);
    ("uint16x8_t", {|typedef struct { unsigned short v[8]; } uint16x8_t;|}, []);
    ("uint8x16_t", {|typedef struct { unsigned char v[16]; } uint8x16_t;|}, []);
    ("half8_t", {|typedef struct { __half v[8]; } half8_t;|}, []);
    ( "uint32_to_single_uniform",
      {|__device__ __forceinline__ float uint32_to_single_uniform(unsigned int x) {
  /* Use __uint2float_rn for correct rounding */
  return __uint2float_rn(x >> 8) * (1.0f / 16777216.0f);
}|},
      [] );
    ( "uint32_to_double_uniform",
      {|__device__ __forceinline__ double uint32_to_double_uniform(unsigned int x) {
  return __uint2double_rn(x) * (1.0 / 4294967296.0);
}|},
      [] );
    ( "uint4x32_to_single_uniform",
      {|__device__ float uint4x32_to_single_uniform(uint4x32_t x) {
  return uint32_to_single_uniform(x.v[0]);
}|},
      [ "uint4x32_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_double_uniform",
      {|__device__ double uint4x32_to_double_uniform(uint4x32_t x) {
  unsigned long long combined = __double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
  return __longlong_as_double(combined) * (1.0 / 18446744073709551616.0);
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_int32_uniform",
      {|__device__ int uint4x32_to_int32_uniform(uint4x32_t x) {
  return (int)x.v[0];
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_i64_uniform",
      {|__device__ long long uint4x32_to_i64_uniform(uint4x32_t x) {
  return __double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_u32_uniform",
      {|__device__ unsigned int uint4x32_to_u32_uniform(uint4x32_t x) {
  return x.v[0];
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_u64_uniform",
      {|__device__ unsigned long long uint4x32_to_u64_uniform(uint4x32_t x) {
  return (unsigned long long)__double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_i8_uniform",
      {|__device__ signed char uint4x32_to_i8_uniform(uint4x32_t x) {
  return (signed char)(x.v[0] & 0xFF);
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_u8_uniform",
      {|__device__ unsigned char uint4x32_to_u8_uniform(uint4x32_t x) {
  return (unsigned char)(x.v[0] & 0xFF);
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_bfloat16_uniform",
      {|__device__ unsigned short uint4x32_to_bfloat16_uniform(uint4x32_t x) {
  float f = uint32_to_single_uniform(x.v[0]);
  return (unsigned short)(__float_as_uint(f) >> 16);
}|},
      [ "uint4x32_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_half_uniform",
      {|__device__ __half uint4x32_to_half_uniform(uint4x32_t x) {
  float f = uint32_to_single_uniform(x.v[0]);
  return __float2half(f);
}|},
      [ "uint4x32_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_single_uniform_vec",
      {|__device__ float4_t uint4x32_to_single_uniform_vec(uint4x32_t x) {
  float4_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i] = uint32_to_single_uniform(x.v[i]);
  }
  return result;
}|},
      [ "uint4x32_t"; "float4_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_double_uniform_vec",
      {|__device__ double2_t uint4x32_to_double_uniform_vec(uint4x32_t x) {
  double2_t result;
  result.v[0] = __longlong_as_double(__double_as_longlong(__hiloint2double(x.v[1], x.v[0]))) * (1.0 / 18446744073709551616.0);
  result.v[1] = __longlong_as_double(__double_as_longlong(__hiloint2double(x.v[3], x.v[2]))) * (1.0 / 18446744073709551616.0);
  return result;
}|},
      [ "uint4x32_t"; "double2_t" ] );
    ( "uint4x32_to_int32_uniform_vec",
      {|__device__ int32x4_t uint4x32_to_int32_uniform_vec(uint4x32_t x) {
  int32x4_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i] = (int)x.v[i];
  }
  return result;
}|},
      [ "uint4x32_t"; "int32x4_t" ] );
    ( "uint4x32_to_i64_uniform_vec",
      {|__device__ int64x2_t uint4x32_to_i64_uniform_vec(uint4x32_t x) {
  int64x2_t result;
  result.v[0] = __double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
  result.v[1] = __double_as_longlong(__hiloint2double(x.v[3], x.v[2]));
  return result;
}|},
      [ "uint4x32_t"; "int64x2_t" ] );
    ( "uint4x32_to_i8_uniform_vec",
      {|__device__ int8x16_t uint4x32_to_i8_uniform_vec(uint4x32_t x) {
  int8x16_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i*4 + 0] = (signed char)(x.v[i] & 0xFF);
    result.v[i*4 + 1] = (signed char)((x.v[i] >> 8) & 0xFF);
    result.v[i*4 + 2] = (signed char)((x.v[i] >> 16) & 0xFF);
    result.v[i*4 + 3] = (signed char)((x.v[i] >> 24) & 0xFF);
  }
  return result;
}|},
      [ "uint4x32_t"; "int8x16_t" ] );
    ( "uint4x32_to_u16_uniform_vec",
      {|__device__ uint16x8_t uint4x32_to_u16_uniform_vec(uint4x32_t x) {
  uint16x8_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i*2 + 0] = (unsigned short)(x.v[i] & 0xFFFF);
    result.v[i*2 + 1] = (unsigned short)((x.v[i] >> 16) & 0xFFFF);
  }
  return result;
}|},
      [ "uint4x32_t"; "uint16x8_t" ] );
    ( "uint4x32_to_bfloat16_uniform_vec",
      {|__device__ uint16x8_t uint4x32_to_bfloat16_uniform_vec(uint4x32_t x) {
  uint16x8_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    // Convert each uint32 to two bfloat16 values
    float f1 = __uint2float_rn((x.v[i] & 0xFFFF) >> 0) * (1.0f / 65536.0f);
    float f2 = __uint2float_rn((x.v[i] >> 16) & 0xFFFF) * (1.0f / 65536.0f);
    result.v[i*2 + 0] = (unsigned short)(__float_as_uint(f1) >> 16);
    result.v[i*2 + 1] = (unsigned short)(__float_as_uint(f2) >> 16);
  }
  return result;
}|},
      [ "uint4x32_t"; "uint16x8_t" ] );
    ( "uint4x32_to_half_uniform_vec",
      {|__device__ half8_t uint4x32_to_half_uniform_vec(uint4x32_t x) {
  half8_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    float f1 = __uint2float_rn((x.v[i] & 0xFFFF) >> 0) * (1.0f / 65536.0f);
    float f2 = __uint2float_rn((x.v[i] >> 16) & 0xFFFF) * (1.0f / 65536.0f);
    result.v[i*2 + 0] = __float2half(f1);
    result.v[i*2 + 1] = __float2half(f2);
  }
  return result;
}|},
      [ "uint4x32_t"; "half8_t" ] );
    ( "uint4x32_to_u8_uniform_vec",
      {|__device__ uint8x16_t uint4x32_to_u8_uniform_vec(uint4x32_t x) {
  uint8x16_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i*4 + 0] = (unsigned char)(x.v[i] & 0xFF);
    result.v[i*4 + 1] = (unsigned char)((x.v[i] >> 8) & 0xFF);
    result.v[i*4 + 2] = (unsigned char)((x.v[i] >> 16) & 0xFF);
    result.v[i*4 + 3] = (unsigned char)((x.v[i] >> 24) & 0xFF);
  }
  return result;
}|},
      [ "uint4x32_t"; "uint8x16_t" ] );
    ( "single_to_uint4x32",
      {|__device__ uint4x32_t single_to_uint4x32(float x) {
  unsigned int bits = __float_as_uint(x);
  uint4x32_t result = {{bits, 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "double_to_uint4x32",
      {|__device__ uint4x32_t double_to_uint4x32(double x) {
  unsigned long long bits = __double_as_longlong(x);
  uint4x32_t result = {{(unsigned int)(bits & 0xFFFFFFFF), (unsigned int)(bits >> 32), 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "int32_to_uint4x32",
      {|__device__ uint4x32_t int32_to_uint4x32(int x) {
  uint4x32_t result = {{(unsigned int)x, 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "int64_to_uint4x32",
      {|__device__ uint4x32_t int64_to_uint4x32(long long x) {
  unsigned long long bits = (unsigned long long)x;
  uint4x32_t result = {{(unsigned int)(bits & 0xFFFFFFFF), (unsigned int)(bits >> 32), 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "uint32_to_uint4x32",
      {|__device__ uint4x32_t uint32_to_uint4x32(unsigned int x) {
  uint4x32_t result = {{x, 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "uint64_to_uint4x32",
      {|__device__ uint4x32_t uint64_to_uint4x32(unsigned long long x) {
  uint4x32_t result = {{(unsigned int)(x & 0xFFFFFFFF), (unsigned int)(x >> 32), 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "byte_to_uint4x32",
      {|__device__ uint4x32_t byte_to_uint4x32(unsigned char x) {
  uint4x32_t result = {{(unsigned int)x, 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "uint16_to_uint4x32",
      {|__device__ uint4x32_t uint16_to_uint4x32(unsigned short x) {
  uint4x32_t result = {{(unsigned int)x, 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "bfloat16_to_uint4x32",
      {|__device__ uint4x32_t bfloat16_to_uint4x32(unsigned short x) {
  uint4x32_t result = {{(unsigned int)x, 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "half_to_uint4x32",
      {|__device__ uint4x32_t half_to_uint4x32(__half x) {
  unsigned short bits = __half_as_ushort(x);
  uint4x32_t result = {{(unsigned int)bits, 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "fp8_to_uint4x32",
      {|__device__ uint4x32_t fp8_to_uint4x32(unsigned char x) {
  uint4x32_t result = {{(unsigned int)x, 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ("THREEFRY_C240", {|__device__ __constant__ unsigned int THREEFRY_C240 = 0x1BD11BDA;|}, []);
    ( "THREEFRY_ROTATION",
      {|__device__ __constant__ unsigned int THREEFRY_ROTATION[8][4] = {
    {13, 15, 26, 6},
    {17, 29, 16, 24},
    {13, 15, 26, 6},
    {17, 29, 16, 24},
    {13, 15, 26, 6},
    {17, 29, 16, 24},
    {13, 15, 26, 6},
    {17, 29, 16, 24}
};|},
      [] );
    ( "rotl32",
      {|__device__ __forceinline__ unsigned int rotl32(unsigned int x, unsigned int n) {
    return __funnelshift_l(x, x, n);
}|},
      [] );
    ( "threefry_round",
      {|__device__ __forceinline__ void threefry_round(uint4 &x, unsigned int r0, unsigned int r1, unsigned int r2, unsigned int r3) {
    x.x += x.y; x.y = rotl32(x.y, r0); x.y ^= x.x;
    x.z += x.w; x.w = rotl32(x.w, r1); x.w ^= x.z;
    
    unsigned int tmp = x.y;
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
      {|__device__ uint4x32_t arrayjit_threefry4x32_crypto(uint4x32_t key, uint4x32_t counter) {
    uint4 x = make_uint4(counter.v[0], counter.v[1], counter.v[2], counter.v[3]);
    uint4 k = make_uint4(key.v[0], key.v[1], key.v[2], key.v[3]);
    
    /* Compute ks[4] */
    unsigned int ks4 = k.x ^ k.y ^ k.z ^ k.w ^ THREEFRY_C240;
    
    /* Initial key injection */
    x.x += k.x;
    x.y += k.y;
    x.z += k.z;
    x.w += k.w;
    
    /* Unrolled 20 rounds with key injections */
    #pragma unroll
    for (int round = 0; round < 20; round += 4) {
        threefry_round(x, THREEFRY_ROTATION[0][0], THREEFRY_ROTATION[0][1], 
                          THREEFRY_ROTATION[0][2], THREEFRY_ROTATION[0][3]);
        threefry_round(x, THREEFRY_ROTATION[1][0], THREEFRY_ROTATION[1][1], 
                          THREEFRY_ROTATION[1][2], THREEFRY_ROTATION[1][3]);
        threefry_round(x, THREEFRY_ROTATION[0][0], THREEFRY_ROTATION[0][1], 
                          THREEFRY_ROTATION[0][2], THREEFRY_ROTATION[0][3]);
        threefry_round(x, THREEFRY_ROTATION[1][0], THREEFRY_ROTATION[1][1], 
                          THREEFRY_ROTATION[1][2], THREEFRY_ROTATION[1][3]);
        
        /* Key injection */
        unsigned int inj_round = (round / 4) + 1;
        if (inj_round == 1) {
            x.x += k.y;
            x.y += k.z;
            x.z += k.w;
            x.w += ks4 + inj_round;
        } else if (inj_round == 2) {
            x.x += k.z;
            x.y += k.w;
            x.z += ks4;
            x.w += k.x + inj_round;
        } else if (inj_round == 3) {
            x.x += k.w;
            x.y += ks4;
            x.z += k.x;
            x.w += k.y + inj_round;
        } else if (inj_round == 4) {
            x.x += ks4;
            x.y += k.x;
            x.z += k.y;
            x.w += k.z + inj_round;
        }
    }
    
    /* Final key injection */
    x.x += k.x;
    x.y += k.y;
    x.z += k.z;
    x.w += k.w + 5;
    
    uint4x32_t result;
    result.v[0] = x.x;
    result.v[1] = x.y;
    result.v[2] = x.z;
    result.v[3] = x.w;
    return result;
}|},
      [ "uint4x32_t"; "THREEFRY_C240"; "threefry_round"; "THREEFRY_ROTATION" ] );
    ( "arrayjit_threefry4x32_light",
      {|__device__ uint4x32_t arrayjit_threefry4x32_light(uint4x32_t key, uint4x32_t counter) {
    uint4 x = make_uint4(counter.v[0], counter.v[1], counter.v[2], counter.v[3]);
    uint4 k = make_uint4(key.v[0], key.v[1], key.v[2], key.v[3]);
    
    /* Compute ks[4] */
    unsigned int ks4 = k.x ^ k.y ^ k.z ^ k.w ^ THREEFRY_C240;
    
    /* Initial key injection */
    x.x += k.x;
    x.y += k.y;
    x.z += k.z;
    x.w += k.w;
    
    /* Only 2 rounds for light version */
    threefry_round(x, THREEFRY_ROTATION[0][0], THREEFRY_ROTATION[0][1], 
                      THREEFRY_ROTATION[0][2], THREEFRY_ROTATION[0][3]);
    threefry_round(x, THREEFRY_ROTATION[1][0], THREEFRY_ROTATION[1][1], 
                      THREEFRY_ROTATION[1][2], THREEFRY_ROTATION[1][3]);
    
    /* Final key injection after round 2 */
    x.x += k.y;
    x.y += k.z;
    x.z += k.w;
    x.w += ks4 + 1;
    
    uint4x32_t result;
    result.v[0] = x.x;
    result.v[1] = x.y;
    result.v[2] = x.z;
    result.v[3] = x.w;
    return result;
}|},
      [ "uint4x32_t"; "THREEFRY_C240"; "threefry_round"; "THREEFRY_ROTATION" ] );
    ( "arrayjit_threefry4x32",
      {|__device__ uint4x32_t arrayjit_threefry4x32(uint4x32_t key, uint4x32_t counter) {
    /* Default to light version */
    return arrayjit_threefry4x32_light(key, counter);
}|},
      [ "uint4x32_t"; "arrayjit_threefry4x32_light" ] );
  ]
