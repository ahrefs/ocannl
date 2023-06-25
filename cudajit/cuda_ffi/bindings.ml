open Ctypes
open Bindings_types

module Functions (F : Ctypes.FOREIGN) = struct
  module E = Types_generated

  let cu_init = F.foreign "cuInit" F.(int @-> returning E.cu_result)
  let cu_device_get_count = F.foreign "cuDeviceGetCount" F.(ptr int @-> returning E.cu_result)
  let cu_device_get = F.foreign "cuDeviceGet" F.(ptr E.cu_device @-> int @-> returning E.cu_result)

  let cu_ctx_create =
    F.foreign "cuCtxCreate" F.(ptr cu_context @-> int @-> E.cu_device @-> returning E.cu_result)

  let cu_module_load_data_ex =
    F.foreign "cuModuleLoadDataEx"
      F.(
        ptr cu_module @-> ptr void @-> int @-> ptr E.cu_jit_option
        @-> ptr (ptr void)
        @-> returning E.cu_result)

  let cu_module_get_function =
    F.foreign "cuModuleGetFunction" F.(ptr cu_function @-> cu_module @-> string @-> returning E.cu_result)

  let cu_mem_alloc = F.foreign "cuMemAlloc" F.(ptr cu_deviceptr @-> size_t @-> returning E.cu_result)

  let cu_memcpy_H_to_D =
    F.foreign "cuMemcpyHtoD" F.(cu_deviceptr @-> ptr void @-> size_t @-> returning E.cu_result)

  let cu_launch_kernel =
    F.foreign "cuLaunchKernel"
      F.(
        cu_function @-> uint @-> uint @-> uint @-> uint @-> uint @-> uint @-> uint @-> cu_stream
        @-> ptr (ptr void)
        @-> ptr (ptr void)
        @-> returning E.cu_result)

  let cu_ctx_synchronize = F.foreign "cuCtxSynchronize" F.(void @-> returning E.cu_result)

  let cu_memcpy_D_to_H =
    F.foreign "cuMemcpyDtoH" F.(ptr void @-> cu_deviceptr @-> size_t @-> returning E.cu_result)

  let cu_mem_free = F.foreign "cuMemFree" F.(cu_deviceptr @-> returning E.cu_result)
  let cu_module_unload = F.foreign "cuModuleUnload" F.(cu_module @-> returning E.cu_result)
  let cu_ctx_destroy = F.foreign "cuCtxDestroy" F.(cu_context @-> returning E.cu_result)
  let cu_memset_d8 = F.foreign "cuMemsetD8" F.(cu_deviceptr @-> uchar @-> size_t @-> returning E.cu_result)
  let cu_memset_d16 = F.foreign "cuMemsetD16" F.(cu_deviceptr @-> ushort @-> size_t @-> returning E.cu_result)

  let cu_memset_d32 =
    F.foreign "cuMemsetD32" F.(cu_deviceptr @-> uint32_t @-> size_t @-> returning E.cu_result)

  let cu_module_get_global =
    F.foreign "cuModuleGetGlobal"
      F.(ptr cu_deviceptr @-> ptr size_t @-> cu_module @-> string @-> returning E.cu_result)

  let cu_device_get_name =
    F.foreign "cuDeviceGetName" F.(ptr char @-> int @-> E.cu_device @-> returning E.cu_result)

  let cu_device_get_attribute =
    F.foreign "cuDeviceGetAttribute"
      F.(ptr int @-> E.cu_device_attribute @-> E.cu_device @-> returning E.cu_result)

  let cu_computemode_of_int mode =
    if mode = Int64.to_int E.cu_computemode_default then CU_COMPUTEMODE_DEFAULT
    else if mode = Int64.to_int E.cu_computemode_exclusive_process then CU_COMPUTEMODE_EXCLUSIVE_PROCESS
    else if mode = Int64.to_int E.cu_computemode_prohibited then CU_COMPUTEMODE_PROHIBITED
    else CU_COMPUTEMODE_UNCATEGORIZED (Int64.of_int mode)
end
