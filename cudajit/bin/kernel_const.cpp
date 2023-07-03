#include <nvrtc.h>
#include <cuda.h>
#include <iostream>

// Compile:
// g++ kernel_const.cpp -o kernel_const -lnvrtc -lcuda -lstdc++

#define NUM_THREADS 1
#define NUM_BLOCKS 1
#define NVRTC_SAFE_CALL(x)                              \
  do                                                    \
  {                                                     \
    nvrtcResult result = x;                             \
    if (result != NVRTC_SUCCESS)                        \
    {                                                   \
      std::cerr << "\nerror: " #x " failed with error " \
                << nvrtcGetErrorString(result) << '\n'; \
      exit(1);                                          \
    }                                                   \
  } while (0)
#define CUDA_SAFE_CALL(x)                               \
  do                                                    \
  {                                                     \
    CUresult result = x;                                \
    if (result != CUDA_SUCCESS)                         \
    {                                                   \
      const char *msg;                                  \
      cuGetErrorName(result, &msg);                     \
      std::cerr << "\nerror: " #x " failed with error " \
                << msg << '\n';                         \
      exit(1);                                          \
    }                                                   \
  } while (0)

const char *ker_const = "\n\
extern \"C\" __constant__ float n2_value[1];\n\
extern \"C\" __constant__ float n1_value[1];\n\
extern \"C\" __global__ void ker_const(\n\
    float *n2_grad, float *n3_value, float *n3_grad, float *n1_grad) {\n\
  float n2_grad_local[1] = {0};\n\
  float n3_value_local[1];\n\
  float n3_grad_local[1] = {0};\n\
  float n1_grad_local[1] = {0};\n\
  n3_value_local[0] = ((n1_value[0]) + (n2_value[0]));\n\
  n3_grad_local[0] = (1.000000);\n\
  n1_grad_local[0] = ((n1_grad_local[0]) + (n3_grad_local[0]));\n\
  n2_grad_local[0] = ((n2_grad_local[0]) + (n3_grad_local[0]));\n\
  if (threadIdx.x == 0 && blockIdx.x == 0) { n2_grad[0] = n2_grad_local[0]; }\n\
  if (threadIdx.x == 0 && blockIdx.x == 0) { n3_value[0] = n3_value_local[0]; }\n\
  if (threadIdx.x == 0 && blockIdx.x == 0) { n3_grad[0] = n3_grad_local[0]; }\n\
  if (threadIdx.x == 0 && blockIdx.x == 0) { n1_grad[0] = n1_grad_local[0]; }\n\
}\n\
";

int main()
{
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(
      nvrtcCreateProgram(&prog,          // prog
                         ker_const,      // buffer
                         "ker_const.cu", // name
                         0,              // numHeaders
                         NULL,           // headers
                         NULL));         // includeNames
  // Note: Can specify GPU target architecture explicitly with '-arch' flag.
  const char *opts[] = {"--use_fast_math"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                  1,     // numOptions
                                                  opts); // options
  // Obtain compilation log from the program.
  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  char *log = new char[logSize];
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
  std::cout << log << '\n';
  delete[] log;
  if (compileResult != NVRTC_SUCCESS)
  {
    exit(1);
  }
  // Obtain PTX from the program.
  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  char *ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
  // Destroy the program.
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
  // Load the generated PTX and get a handle to the SAXPY kernel.
  CUdevice cuDevice;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "ker_const"));
  // Generate input for execution, and create output buffers.
  size_t n = 1;
  size_t bufferSize = n * sizeof(float);
  float n1_value[1] = {-4};
  float n1_grad[1] = {0};
  float n2_value[1] = {2};
  float n2_grad[1] = {0};
  float n3_value[1] = {0};
  float n3_grad[1] = {0};
  CUdeviceptr d_n2_grad, d_n3_value, d_n3_grad, d_n1_grad, d_n1_value, d_n2_value;
  CUDA_SAFE_CALL(cuMemAlloc(&d_n2_grad, bufferSize));
  CUDA_SAFE_CALL(cuMemAlloc(&d_n3_value, bufferSize));
  CUDA_SAFE_CALL(cuMemAlloc(&d_n3_grad, bufferSize));
  CUDA_SAFE_CALL(cuMemAlloc(&d_n1_grad, bufferSize));
  size_t d_n1_value_s, d_n2_value_s;
  CUDA_SAFE_CALL(cuModuleGetGlobal_v2(&d_n1_value, &d_n1_value_s, module, "n1_value"));
  CUDA_SAFE_CALL(cuModuleGetGlobal_v2(&d_n2_value, &d_n2_value_s, module, "n2_value"));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_n1_value, n1_value, d_n1_value_s));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_n2_value, n2_value, d_n2_value_s));
  void *args[] = {&d_n2_grad, &d_n3_value, &d_n3_grad, &d_n1_grad};
  CUDA_SAFE_CALL(
      cuLaunchKernel(kernel,
                     NUM_BLOCKS, 1, 1,  // grid dim
                     NUM_THREADS, 1, 1, // block dim
                     0, NULL,           // shared mem and stream
                     args, 0));         // arguments
  CUDA_SAFE_CALL(cuCtxSynchronize());
  // Retrieve and print output.
  CUDA_SAFE_CALL(cuMemcpyDtoH(n3_value, d_n3_value, bufferSize));
  CUDA_SAFE_CALL(cuMemcpyDtoH(n3_grad, d_n3_grad, bufferSize));
  CUDA_SAFE_CALL(cuMemcpyDtoH(n2_grad, d_n2_grad, bufferSize));
  CUDA_SAFE_CALL(cuMemcpyDtoH(n1_grad, d_n1_grad, bufferSize));
  std::cout << "n1 = " << n1_value[0] << "; n2 = " << n2_value[0]
            << "; n3 = " << n3_value[0] << "; n1_grad = " << n1_grad[0]
            << "; n2_grad = " << n2_grad[0] << std::endl;
  // Release resources.
  CUDA_SAFE_CALL(cuMemFree(d_n2_grad));
  CUDA_SAFE_CALL(cuMemFree(d_n3_value));
  CUDA_SAFE_CALL(cuMemFree(d_n3_grad));
  CUDA_SAFE_CALL(cuMemFree(d_n1_grad));
  CUDA_SAFE_CALL(cuModuleUnload(module));
  CUDA_SAFE_CALL(cuCtxDestroy(context));
  return 0;
}