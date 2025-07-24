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
__device__ uint4x32_t arrayjit_threefry4x32_impl(uint4x32_t key, uint4x32_t counter) {
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

__device__ uint4x32_t ( *arrayjit_threefry4x32)(uint4x32_t key, uint4x32_t counter) = arrayjit_threefry4x32_impl;