
typedef struct {
    uint32_t v[4];
} uint4x32_t;

/* Conversion functions from uint4x32 to various precisions uniformly */

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