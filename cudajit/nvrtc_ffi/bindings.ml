open Ctypes
open Bindings_types

module Functions (F : Ctypes.FOREIGN) = struct
  module E = Types_generated

  let nvrtc_version = F.foreign "nvrtcVersion" F.(ptr int @-> ptr int @-> returning E.nvrtc_result)

  let nvrtc_get_num_supported_archs =
    F.foreign "nvrtcGetNumSupportedArchs" F.(ptr int @-> returning E.nvrtc_result)

  let nvrtc_get_supported_archs =
    F.foreign "nvrtcGetNumSupportedArchs" F.(ptr int @-> returning E.nvrtc_result)

  let nvrtc_create_program =
    F.foreign "nvrtcCreateProgram"
      F.(
        ptr nvrtc_program @-> string @-> string @-> int @-> ptr string @-> ptr string
        @-> returning E.nvrtc_result)

  let nvrtc_destroy_program =
    F.foreign "nvrtcDestroyProgram" F.(ptr nvrtc_program @-> returning E.nvrtc_result)

  let nvrtc_compile_program =
    F.foreign "nvrtcCompileProgram" F.(nvrtc_program @-> int @-> ptr string @-> returning E.nvrtc_result)

  let nvrtc_get_PTX_size =
    F.foreign "nvrtcGetPTXSize" F.(nvrtc_program @-> ptr size_t @-> returning E.nvrtc_result)

  let nvrtc_get_PTX = F.foreign "nvrtcGetPTX" F.(nvrtc_program @-> ptr char @-> returning E.nvrtc_result)

  let nvrtc_get_cubin_size =
    F.foreign "nvrtcGetCUBINSize" F.(nvrtc_program @-> ptr size_t @-> returning E.nvrtc_result)

  let nvrtc_get_cubin = F.foreign "nvrtcGetCUBIN" F.(nvrtc_program @-> ptr char @-> returning E.nvrtc_result)

  let nvrtc_get_NVVM_size =
    F.foreign "nvrtcGetNVVMSize" F.(nvrtc_program @-> ptr size_t @-> returning E.nvrtc_result)

  let nvrtc_get_NVVM = F.foreign "nvrtcGetNVVM" F.(nvrtc_program @-> ptr char @-> returning E.nvrtc_result)

  let nvrtc_get_program_log_size =
    F.foreign "nvrtcGetProgramLogSize" F.(nvrtc_program @-> ptr size_t @-> returning E.nvrtc_result)

  let nvrtc_get_program_log =
    F.foreign "nvrtcGetProgramLog" F.(nvrtc_program @-> ptr char @-> returning E.nvrtc_result)

  let nvrtc_add_name_expression =
    F.foreign "nvrtcAddNameExpression" F.(nvrtc_program @-> string @-> returning E.nvrtc_result)

  let nvrtc_get_lowered_name =
    F.foreign "nvrtcGetLoweredName" F.(nvrtc_program @-> string @-> ptr string @-> returning E.nvrtc_result)
end
