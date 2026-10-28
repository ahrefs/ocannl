(** Bindings to the NVIDIA `nvrtc` library.

    NVRTC is a runtime compilation library for CUDA C++. See:
    {{:https://docs.nvidia.com/cuda/nvrtc/index.html} The User guide for the NVRTC library}. *)

type result
(** See {{:https://docs.nvidia.com/cuda/nvrtc/index.html#_CPPv411nvrtcResult} enum nvrtcResult}. *)

val sexp_of_result : result -> Sexplib0.Sexp.t

exception Nvrtc_error of { status : result; message : string }
(** Error codes returned by CUDA functions are converted to exceptions. The message stores a
    snake-case variant of the offending CUDA function name (see {!Nvrtc_ffi.Bindings.Functions} for
    the direct funciton bindings). *)

val is_success : result -> bool

type compile_to_ptx_result = {
  log : string option;
  ptx : (char Ctypes.ptr[@sexp.opaque]);
  ptx_length : int;
}
(** The values passed from {!compile_to_ptx} to {!Cuda.Module.load_data_ex}. Currently, cudajit
    converts the result of [nvrtc_compile_program] to human-readable PTX assembly before passing it
    to the [cu_module_load_data_ex] function. *)

val sexp_of_compile_to_ptx_result : compile_to_ptx_result -> Sexplib0.Sexp.t

val compile_to_ptx :
  cu_src:string -> name:string -> options:string list -> with_debug:bool -> compile_to_ptx_result
(** Performs a cascade of calls:
    {{:https://docs.nvidia.com/cuda/nvrtc/index.html#_CPPv418nvrtcCreateProgramP12nvrtcProgramPKcPKciPPCKcPPCKc}
     nvrtcCreateProgram},
    {{:https://docs.nvidia.com/cuda/nvrtc/index.html#_CPPv419nvrtcCompileProgram12nvrtcProgramiPPCKc}
     nvrtcCompileProgram},
    {{:https://docs.nvidia.com/cuda/nvrtc/index.html#_CPPv411nvrtcGetPTX12nvrtcProgramPc}
     nvrtcGetPTX}. If you store [cu_src] as a file, pass the file name including the extension as
    [name]. [options] can include for example ["--use_fast_math"] or ["--device-debug"]. If
    [with_debug] is [true], the compilation log is included even in case of compilation success (see
    {!compilation_log}).

    NOTE: [compile_to_ptx] prepends the CUDA include path to [options], so you don't need to. *)

val string_from_ptx : compile_to_ptx_result -> string
(** The stored PTX (i.e. NVIDIA assembly language) source, see
    {{:https://docs.nvidia.com/cuda/nvrtc/index.html#_CPPv411nvrtcGetPTX12nvrtcProgramPc}
     nvrtcGetPTX}. *)

val compilation_log : compile_to_ptx_result -> string option
(** The stored side output of the compilation, see
    {{:https://docs.nvidia.com/cuda/nvrtc/index.html#_CPPv418nvrtcGetProgramLog12nvrtcProgramPc}
     nvrtcGetProgramLog}. *)
