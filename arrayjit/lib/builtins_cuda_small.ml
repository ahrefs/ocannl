let source =
  {|
typedef struct {
    uint32_t v[4];
} uint4x32_t;

/* Vector types for efficient extraction of multiple values */
typedef struct { float v[4]; } float4_t;
typedef struct { double v[2]; } double2_t;
typedef struct { int32_t v[4]; } int32x4_t;
typedef struct { int64_t v[2]; } int64x2_t;
typedef struct { int8_t v[16]; } int8x16_t;
typedef struct { uint16_t v[8]; } uint16x8_t;
typedef struct { uint8_t v[16]; } uint8x16_t;
typedef struct { __half v[8]; } half8_t;

/* Conversion functions from uint4x32 to various precisions uniformly */
// These return vectors to efficiently use all random bits

/* Convert to float in [0, 1) using CUDA intrinsics */
__device__ __forceinline__ float uint32_to_single_uniform(uint32_t x) {
  /* Use __uint2float_rn for correct rounding */
  return __uint2float_rn(x >> 8) * (1.0f / 16777216.0f);
}

/* Convert to double in [0, 1) */
__device__ __forceinline__ double uint32_to_double_uniform(uint32_t x) {
  return __uint2double_rn(x) * (1.0 / 4294967296.0);
}

/* Uint4x32 to float32 uniform */
__device__ float uint4x32_to_single_uniform(uint4x32_t x) {
  return uint32_to_single_uniform(x.v[0]);
}

/* Uint4x32 to float64 uniform */
__device__ double uint4x32_to_double_uniform(uint4x32_t x) {
  uint64_t combined = __double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
  return __longlong_as_double(combined) * (1.0 / 18446744073709551616.0);
}

/* Uint4x32 to int32 uniform */
__device__ int32_t uint4x32_to_int32_uniform(uint4x32_t x) {
  return (int32_t)x.v[0];
}

/* Uint4x32 to int64 uniform */
__device__ int64_t uint4x32_to_i64_uniform(uint4x32_t x) {
  return __double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
}

/* Uint4x32 to uint32 uniform */
__device__ uint32_t uint4x32_to_u32_uniform(uint4x32_t x) {
  return x.v[0];
}

/* Uint4x32 to uint64 uniform */
__device__ uint64_t uint4x32_to_u64_uniform(uint4x32_t x) {
  return (uint64_t)__double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
}

/* Uint4x32 to int8 uniform */
__device__ int8_t uint4x32_to_i8_uniform(uint4x32_t x) {
  return (int8_t)(x.v[0] & 0xFF);
}

/* Uint4x32 to uint8 uniform */
__device__ uint8_t uint4x32_to_u8_uniform(uint4x32_t x) {
  return (uint8_t)(x.v[0] & 0xFF);
}

/* Uint4x32 to bfloat16 uniform */
__device__ uint16_t uint4x32_to_bfloat16_uniform(uint4x32_t x) {
  float f = uint32_to_single_uniform(x.v[0]);
  return (uint16_t)(__float_as_uint(f) >> 16);
}

/* Uint4x32 to float16 uniform using CUDA half intrinsics */
__device__ __half uint4x32_to_half_uniform(uint4x32_t x) {
  float f = uint32_to_single_uniform(x.v[0]);
  return __float2half(f);
}

/* Vectorized conversion functions that use all 128 bits efficiently */

/* Convert uint4x32 to 4 floats in [0, 1) */
__device__ float4_t uint4x32_to_single_uniform_vec(uint4x32_t x) {
  float4_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i] = uint32_to_single_uniform(x.v[i]);
  }
  return result;
}

/* Convert uint4x32 to 2 doubles in [0, 1) */
__device__ double2_t uint4x32_to_double_uniform_vec(uint4x32_t x) {
  double2_t result;
  result.v[0] = __longlong_as_double(__double_as_longlong(__hiloint2double(x.v[1], x.v[0]))) * (1.0 / 18446744073709551616.0);
  result.v[1] = __longlong_as_double(__double_as_longlong(__hiloint2double(x.v[3], x.v[2]))) * (1.0 / 18446744073709551616.0);
  return result;
}

/* Convert uint4x32 to 4 int32s - full range */
__device__ int32x4_t uint4x32_to_int32_uniform_vec(uint4x32_t x) {
  int32x4_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i] = (int32_t)x.v[i];
  }
  return result;
}

/* Convert uint4x32 to 2 int64s - full range */
__device__ int64x2_t uint4x32_to_i64_uniform_vec(uint4x32_t x) {
  int64x2_t result;
  result.v[0] = __double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
  result.v[1] = __double_as_longlong(__hiloint2double(x.v[3], x.v[2]));
  return result;
}


/* Convert uint4x32 to 16 int8s - full range */
__device__ int8x16_t uint4x32_to_i8_uniform_vec(uint4x32_t x) {
  int8x16_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i*4 + 0] = (int8_t)(x.v[i] & 0xFF);
    result.v[i*4 + 1] = (int8_t)((x.v[i] >> 8) & 0xFF);
    result.v[i*4 + 2] = (int8_t)((x.v[i] >> 16) & 0xFF);
    result.v[i*4 + 3] = (int8_t)((x.v[i] >> 24) & 0xFF);
  }
  return result;
}

/* Convert uint4x32 to 8 uint16s - full range */
__device__ uint16x8_t uint4x32_to_u16_uniform_vec(uint4x32_t x) {
  uint16x8_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i*2 + 0] = (uint16_t)(x.v[i] & 0xFFFF);
    result.v[i*2 + 1] = (uint16_t)((x.v[i] >> 16) & 0xFFFF);
  }
  return result;
}

/* Convert uint4x32 to 8 bfloat16s uniform */
__device__ uint16x8_t uint4x32_to_bfloat16_uniform_vec(uint4x32_t x) {
  uint16x8_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    // Convert each uint32 to two bfloat16 values
    float f1 = __uint2float_rn((x.v[i] & 0xFFFF) >> 0) * (1.0f / 65536.0f);
    float f2 = __uint2float_rn((x.v[i] >> 16) & 0xFFFF) * (1.0f / 65536.0f);
    result.v[i*2 + 0] = (uint16_t)(__float_as_uint(f1) >> 16);
    result.v[i*2 + 1] = (uint16_t)(__float_as_uint(f2) >> 16);
  }
  return result;
}

/* Convert uint4x32 to 8 float16s uniform */
__device__ half8_t uint4x32_to_half_uniform_vec(uint4x32_t x) {
  half8_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    float f1 = __uint2float_rn((x.v[i] & 0xFFFF) >> 0) * (1.0f / 65536.0f);
    float f2 = __uint2float_rn((x.v[i] >> 16) & 0xFFFF) * (1.0f / 65536.0f);
    result.v[i*2 + 0] = __float2half(f1);
    result.v[i*2 + 1] = __float2half(f2);
  }
  return result;
}

/* Convert uint4x32 to 16 uint8s uniform */
__device__ uint8x16_t uint4x32_to_u8_uniform_vec(uint4x32_t x) {
  uint8x16_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i*4 + 0] = (uint8_t)(x.v[i] & 0xFF);
    result.v[i*4 + 1] = (uint8_t)((x.v[i] >> 8) & 0xFF);
    result.v[i*4 + 2] = (uint8_t)((x.v[i] >> 16) & 0xFF);
    result.v[i*4 + 3] = (uint8_t)((x.v[i] >> 24) & 0xFF);
  }
  return result;
}
|}
