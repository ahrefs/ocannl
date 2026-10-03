// gh-ocannl-1073: staged fp8 m16n8k32, per-block D traffic vs held registers.
// A standalone reproduction of CUDA's plain inline-PTX fragment addressing.
// Build before reserving timing: nvcc -O3 -gencode arch=compute_89,code=compute_89
//   mma_register_scope_probe.cu -o /tmp/wave1003/1073/mma_register_scope_probe
// --dry-run validates both arms and runs one replay without reporting timings.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t e) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
    std::exit(1);
  }
}

// For a generated-kernel confirmation, compile with both emitted .cu paths:
// -DMMA_GENERATED_BASELINE='"/absolute/baseline.cu"'
// -DMMA_GENERATED_RESIDENT='"/absolute/resident.cu"'
#ifdef MMA_GENERATED_BASELINE
#define mma_register_scope_probe mma_register_scope_baseline
#include MMA_GENERATED_BASELINE
#undef mma_register_scope_probe
#define mma_register_scope_probe mma_register_scope_resident
#include MMA_GENERATED_RESIDENT
#undef mma_register_scope_probe
#else
// One warp owns a 16x32 D tile, with a 32-wide cooperative staging loop.
// Like the backend, each mma statement ends in a barrier releasing its tiles.
// The statement arm additionally brackets its D load with a leading barrier;
// the register scope moves that bracket outside the entire reduction.
template<bool resident>
__global__ void staged(const unsigned char *a, const unsigned char *b, float *d,
                       int m, int n, int k) {
  __shared__ unsigned char at[16 * 32], bt[32 * 32];
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const int row0 = blockIdx.y * 16, col0 = blockIdx.x * 32;
  float frag[1][4][4];
  float *dp = d + row0 * n + col0;
  for (int i = lane; i < 16 * 32; i += 32) dp[(i / 32) * n + i % 32] = 0;
  __syncthreads();
  if (resident) {
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      float *dr0 = dp + g * n + ni * 8 + 2 * t;
      float *dr1 = dr0 + 8 * n;
      frag[0][ni][0] = dr0[0]; frag[0][ni][1] = dr0[1];
      frag[0][ni][2] = dr1[0]; frag[0][ni][3] = dr1[1];
    }
  }
  for (int ko = 0; ko < k; ko += 32) {
    for (int i = lane; i < 16 * 32; i += 32)
      at[i] = a[(row0 + i / 32) * k + ko + i % 32];
    for (int i = lane; i < 32 * 32; i += 32)
      bt[i] = b[(ko + i / 32) * n + col0 + i % 32];
    __syncthreads();
    if (!resident) __syncthreads();
    // The backend unrolls the resident fragment loops, but leaves statement loops alone.
#pragma unroll (resident ? 4 : 1)
    for (int ni = 0; ni < 4; ++ni) {
      float *dr0 = dp + g * n + ni * 8 + 2 * t;
      float *dr1 = dr0 + 8 * n;
      float d0, d1, d2, d3;
      if (resident) {
        d0 = frag[0][ni][0]; d1 = frag[0][ni][1];
        d2 = frag[0][ni][2]; d3 = frag[0][ni][3];
      } else {
        d0 = dr0[0]; d1 = dr0[1]; d2 = dr1[0]; d3 = dr1[1];
      }
      unsigned ar[4], br[2];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int off = (g + (r % 2) * 8) * 32 + 4 * t + (r / 2) * 16;
        ar[r] = unsigned(at[off]) | (unsigned(at[off+1]) << 8)
          | (unsigned(at[off+2]) << 16) | (unsigned(at[off+3]) << 24);
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int off = (4 * t + r * 16) * 32 + ni * 8 + g;
        br[r] = unsigned(bt[off]) | (unsigned(bt[off+32]) << 8)
          | (unsigned(bt[off+64]) << 16) | (unsigned(bt[off+96]) << 24);
      }
      asm("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
          : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]), "r"(br[0]), "r"(br[1]));
      if (resident) {
        frag[0][ni][0] = d0; frag[0][ni][1] = d1;
        frag[0][ni][2] = d2; frag[0][ni][3] = d3;
      } else {
        dr0[0] = d0; dr0[1] = d1; dr1[0] = d2; dr1[1] = d3;
      }
    }
    __syncthreads();
  }
  if (resident) {
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      float *dr0 = dp + g * n + ni * 8 + 2 * t;
      float *dr1 = dr0 + 8 * n;
      dr0[0] = frag[0][ni][0]; dr0[1] = frag[0][ni][1];
      dr1[0] = frag[0][ni][2]; dr1[1] = frag[0][ni][3];
    }
    __syncthreads();
  }
}

#endif

int main(int argc, char **argv) {
  const bool dry = argc == 2 && std::strcmp(argv[1], "--dry-run") == 0;
  if (argc > 1 && !dry) { std::fprintf(stderr, "usage: %s [--dry-run]\n", argv[0]); return 2; }
  cudaDeviceProp prop;
  check(cudaGetDeviceProperties(&prop, 0));
  std::printf("backend=cuda device=%s sm_%d%d\n", prop.name, prop.major, prop.minor);
  if (prop.major * 10 + prop.minor < 89) return 2;
#ifdef MMA_GENERATED_BASELINE
  // Matches bench_mma_register_scope_emit's actual schedule: 512 warps, 128 k_o blocks.
  const int m = 8192, n = 32, k = 4096;
  std::puts("arms=generated baseline and generated resident kernels");
#else
  const int m = 512, n = 512, k = 4096;
  std::puts("arms=standalone fragment/staging reproduction");
#endif
  unsigned char *a, *b;
  float *d;
  check(cudaMalloc(&a, m*k)); check(cudaMalloc(&b, k*n));
  check(cudaMalloc(&d, m*n*sizeof(float)));
  std::vector<unsigned char> ha(m*k), hb(k*n);
  // e5m2 encodes 1 and 2 as 0x3c and 0x40. Every sum is an exact f32 integer.
  for (int i = 0; i < m*k; ++i) ha[i] = ((i / k * 17 + i % k * 13) % 7 < 3) ? 0x3c : 0x40;
  for (int i = 0; i < k*n; ++i) hb[i] = ((i / n * 11 + i % n * 19) % 11 < 5) ? 0x3c : 0x40;
  check(cudaMemcpy(a, ha.data(), ha.size(), cudaMemcpyHostToDevice));
  check(cudaMemcpy(b, hb.data(), hb.size(), cudaMemcpyHostToDevice));
  auto launch = [&](int arm) {
#ifdef MMA_GENERATED_BASELINE
    if (arm) mma_register_scope_resident<<<m/16,32>>>((__nv_fp8_e5m2 *)a,(__nv_fp8_e5m2 *)b,d);
    else mma_register_scope_baseline<<<m/16,32>>>((__nv_fp8_e5m2 *)a,(__nv_fp8_e5m2 *)b,d);
#else
    if (arm) staged<true><<<dim3(n/32,m/16),32>>>(a,b,d,m,n,k);
    else staged<false><<<dim3(n/32,m/16),32>>>(a,b,d,m,n,k);
#endif
    check(cudaGetLastError());
  };
  if (dry) {
    std::vector<float> outputs[2];
    for (int arm = 0; arm < 2; ++arm) {
      launch(arm); check(cudaDeviceSynchronize());
      outputs[arm].resize(m*n);
      check(cudaMemcpy(outputs[arm].data(), d, m*n*sizeof(float), cudaMemcpyDeviceToHost));
    }
    // Verify every cell, not just equality of two possibly broken arms.
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
      float want = 0;
      for (int l = 0; l < k; ++l)
        want += (ha[i*k+l] == 0x3c ? 1.f : 2.f) * (hb[l*n+j] == 0x3c ? 1.f : 2.f);
      if (outputs[0][i*n+j] != want || outputs[1][i*n+j] != want) {
        std::fprintf(stderr, "mismatch (%d,%d) want=%g A=%g B=%g\n", i,j,want,
                     outputs[0][i*n+j],outputs[1][i*n+j]); return 1;
      }
    }
    launch(1); check(cudaDeviceSynchronize());
    std::vector<float> replay(m*n);
    check(cudaMemcpy(replay.data(), d, m*n*sizeof(float), cudaMemcpyDeviceToHost));
    if (replay != outputs[1]) {
      std::fprintf(stderr, "resident replay differs from the validated output\n"); return 1;
    }
    std::puts("dry-run: both arms equal every exact host cell; resident replay passed");
  } else {
    const int repeats = 100, pairs = 9;
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    for (int arm = 0; arm < 2; ++arm) for (int i = 0; i < 10; ++i) launch(arm);
    check(cudaDeviceSynchronize());
    std::printf("shape=%dx%dx%d tile=16x32x32 k_o_blocks=%d repeats=%d paired_replicates=%d\n",
                m,n,k,k/32,repeats,pairs);
    for (int pair = 0; pair < pairs; ++pair) {
      float ms[2];
      for (int order = 0; order < 2; ++order) {
        const int arm = order ^ (pair & 1);
        check(cudaEventRecord(begin));
        for (int i = 0; i < repeats; ++i) launch(arm);
        check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
        check(cudaEventElapsedTime(&ms[arm], begin, end));
        ms[arm] /= repeats;
      }
      std::printf("pair=%d per_block_ms=%.6f resident_ms=%.6f resident_over_per_block=%.6f\n",
                  pair,ms[0],ms[1],ms[1]/ms[0]);
    }
    check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
  }
  check(cudaFree(a)); check(cudaFree(b)); check(cudaFree(d));
}
