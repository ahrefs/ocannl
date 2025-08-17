let source =
  {|
typedef struct {
    unsigned int v[4];
} uint4x32_t;

/* Vector types for efficient extraction of multiple values */
typedef struct { float v[4]; } float4_t;
typedef struct { double v[2]; } double2_t;
typedef struct { int v[4]; } int32x4_t;
typedef struct { long long v[2]; } int64x2_t;
typedef struct { signed char v[16]; } int8x16_t;
typedef struct { unsigned short v[8]; } uint16x8_t;
typedef struct { unsigned char v[16]; } uint8x16_t;
typedef struct { __half v[8]; } half8_t;

/* Conversion functions from uint4x32 to various precisions uniformly */
// These return vectors to efficiently use all random bits

/* Convert to float in [0, 1) using CUDA intrinsics */
__device__ __forceinline__ float uint32_to_single_uniform(unsigned int x) {
  /* Use __uint2float_rn for correct rounding */
  return __uint2float_rn(x >> 8) * (1.0f / 16777216.0f);
}

/* Convert to double in [0, 1) */
__device__ __forceinline__ double uint32_to_double_uniform(unsigned int x) {
  return __uint2double_rn(x) * (1.0 / 4294967296.0);
}

/* Uint4x32 to float32 uniform */
__device__ float uint4x32_to_single_uniform(uint4x32_t x) {
  return uint32_to_single_uniform(x.v[0]);
}

/* Uint4x32 to float64 uniform */
__device__ double uint4x32_to_double_uniform(uint4x32_t x) {
  unsigned long long combined = __double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
  return __longlong_as_double(combined) * (1.0 / 18446744073709551616.0);
}

/* Uint4x32 to int32 uniform */
__device__ int uint4x32_to_int32_uniform(uint4x32_t x) {
  return (int)x.v[0];
}

/* Uint4x32 to int64 uniform */
__device__ long long uint4x32_to_i64_uniform(uint4x32_t x) {
  return __double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
}

/* Uint4x32 to uint32 uniform */
__device__ unsigned int uint4x32_to_u32_uniform(uint4x32_t x) {
  return x.v[0];
}

/* Uint4x32 to uint64 uniform */
__device__ unsigned long long uint4x32_to_u64_uniform(uint4x32_t x) {
  return (unsigned long long)__double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
}

/* Uint4x32 to int8 uniform */
__device__ signed char uint4x32_to_i8_uniform(uint4x32_t x) {
  return (signed char)(x.v[0] & 0xFF);
}

/* Uint4x32 to uint8 uniform */
__device__ unsigned char uint4x32_to_u8_uniform(uint4x32_t x) {
  return (unsigned char)(x.v[0] & 0xFF);
}

/* Uint4x32 to bfloat16 uniform */
__device__ unsigned short uint4x32_to_bfloat16_uniform(uint4x32_t x) {
  float f = uint32_to_single_uniform(x.v[0]);
  return (unsigned short)(__float_as_uint(f) >> 16);
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
    result.v[i] = (int)x.v[i];
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
    result.v[i*4 + 0] = (signed char)(x.v[i] & 0xFF);
    result.v[i*4 + 1] = (signed char)((x.v[i] >> 8) & 0xFF);
    result.v[i*4 + 2] = (signed char)((x.v[i] >> 16) & 0xFF);
    result.v[i*4 + 3] = (signed char)((x.v[i] >> 24) & 0xFF);
  }
  return result;
}

/* Convert uint4x32 to 8 uint16s - full range */
__device__ uint16x8_t uint4x32_to_u16_uniform_vec(uint4x32_t x) {
  uint16x8_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i*2 + 0] = (unsigned short)(x.v[i] & 0xFFFF);
    result.v[i*2 + 1] = (unsigned short)((x.v[i] >> 16) & 0xFFFF);
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
    result.v[i*2 + 0] = (unsigned short)(__float_as_uint(f1) >> 16);
    result.v[i*2 + 1] = (unsigned short)(__float_as_uint(f2) >> 16);
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
    result.v[i*4 + 0] = (unsigned char)(x.v[i] & 0xFF);
    result.v[i*4 + 1] = (unsigned char)((x.v[i] >> 8) & 0xFF);
    result.v[i*4 + 2] = (unsigned char)((x.v[i] >> 16) & 0xFF);
    result.v[i*4 + 3] = (unsigned char)((x.v[i] >> 24) & 0xFF);
  }
  return result;
}

/* Convert int64 to uint4x32 */
__device__ uint4x32_t int64_to_uint4x32(long long x) {
  unsigned long long bits = (unsigned long long)x;
  uint4x32_t result = {{(unsigned int)(bits & 0xFFFFFFFF), (unsigned int)(bits >> 32), 0, 0}};
  return result;
}

/* Conversion functions from various precisions to uint4x32_t */
__device__ uint4x32_t single_to_uint4x32(float x) {
  unsigned int bits = __float_as_uint(x);
  uint4x32_t result = {{bits, 0, 0, 0}};
  return result;
}

__device__ uint4x32_t double_to_uint4x32(double x) {
  unsigned long long bits = __double_as_longlong(x);
  uint4x32_t result = {{(unsigned int)(bits & 0xFFFFFFFF), (unsigned int)(bits >> 32), 0, 0}};
  return result;
}

__device__ uint4x32_t int32_to_uint4x32(int x) {
  uint4x32_t result = {{(unsigned int)x, 0, 0, 0}};
  return result;
}

__device__ uint4x32_t uint32_to_uint4x32(unsigned int x) {
  uint4x32_t result = {{x, 0, 0, 0}};
  return result;
}

__device__ uint4x32_t uint64_to_uint4x32(unsigned long long x) {
  uint4x32_t result = {{(unsigned int)(x & 0xFFFFFFFF), (unsigned int)(x >> 32), 0, 0}};
  return result;
}

__device__ uint4x32_t byte_to_uint4x32(unsigned char x) {
  uint4x32_t result = {{(unsigned int)x, 0, 0, 0}};
  return result;
}

__device__ uint4x32_t uint16_to_uint4x32(unsigned short x) {
  uint4x32_t result = {{(unsigned int)x, 0, 0, 0}};
  return result;
}

__device__ uint4x32_t bfloat16_to_uint4x32(unsigned short x) {
  uint4x32_t result = {{(unsigned int)x, 0, 0, 0}};
  return result;
}

__device__ uint4x32_t half_to_uint4x32(__half x) {
  unsigned short bits = __half_as_ushort(x);
  uint4x32_t result = {{(unsigned int)bits, 0, 0, 0}};
  return result;
}

__device__ uint4x32_t fp8_to_uint4x32(unsigned char x) {
  uint4x32_t result = {{(unsigned int)x, 0, 0, 0}};
  return result;
}
|}
