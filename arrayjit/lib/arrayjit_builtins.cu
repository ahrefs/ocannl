#include <cuda_runtime.h>
#include <stdint.h>

typedef struct {
    uint32_t v[4];
} uint4x32_t;

/* Threefry4x32 constants */
__device__ __constant__ uint32_t THREEFRY_C240 = 0x1BD11BDA;

/* Rotation constants for Threefry4x32 */
__device__ __constant__ unsigned int THREEFRY_ROTATION[8][4] = {
    {13, 15, 26, 6},
    {17, 29, 16, 24},
    {13, 15, 26, 6},
    {17, 29, 16, 24},
    {13, 15, 26, 6},
    {17, 29, 16, 24},
    {13, 15, 26, 6},
    {17, 29, 16, 24}
};

/* CUDA intrinsic-based rotate left */
__device__ __forceinline__ uint32_t rotl32(uint32_t x, unsigned int n) {
    return __funnelshift_l(x, x, n);
}

/* Threefry4x32 round function using vector operations */
__device__ __forceinline__ void threefry_round(uint4 &x, unsigned int r0, unsigned int r1, unsigned int r2, unsigned int r3) {
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
__device__ uint4x32_t arrayjit_threefry4x32(uint4x32_t key, uint4x32_t counter) {
    uint4 x = make_uint4(counter.v[0], counter.v[1], counter.v[2], counter.v[3]);
    uint4 k = make_uint4(key.v[0], key.v[1], key.v[2], key.v[3]);
    
    /* Compute ks[4] */
    uint32_t ks4 = k.x ^ k.y ^ k.z ^ k.w ^ THREEFRY_C240;
    
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
        uint32_t inj_round = (round / 4) + 1;
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
}

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