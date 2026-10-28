(** Bindings to the AMD `hiprtc` library.

    HIPRTC is a runtime compilation library for HIP C++, the direct analog of NVIDIA's NVRTC. See:
    {{:https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html} the HIP RTC
     programming guide}. *)

type result
(** See
    {{:https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hiprtc.html} enum hiprtcResult}.
*)

val sexp_of_result : result -> Sexplib0.Sexp.t

exception Hiprtc_error of { status : result; message : string }
(** Error codes returned by hiprtc functions are converted to exceptions. The message stores a
    snake-case variant of the offending hiprtc function name (see {!Hiprtc_ffi.Bindings.Functions}
    for the direct function bindings). *)

val is_success : result -> bool

val error_string : result -> string
(** The name of the error, as reported by [hiprtcGetErrorString]. *)

val version : unit -> int * int
(** The HIP runtime compilation (major, minor) version, from [hiprtcVersion]. *)

type compile_to_code_result = {
  log : string option;
  code : (char Ctypes.ptr[@sexp.opaque]);
  code_length : int;
}
(** The values passed from {!compile_to_code} to {!Hip.Module.load_data_ex}. Unlike NVRTC (which
    produces textual PTX assembly), hiprtc produces a binary GPU code object, ready to be loaded as
    a module. *)

val sexp_of_compile_to_code_result : compile_to_code_result -> Sexplib0.Sexp.t

val compile_to_code :
  hip_src:string -> name:string -> options:string list -> with_debug:bool -> compile_to_code_result
(** Performs a cascade of calls: [hiprtcCreateProgram], [hiprtcCompileProgram], [hiprtcGetCode]. If
    you store [hip_src] as a file, pass the file name including the extension as [name]. [options]
    can include for example ["-ffast-math"], ["-g"], or ["--offload-arch=gfx1100"] (see
    {!Hip.Device.attributes.gcn_arch_name}); when no [--offload-arch] is given, hiprtc targets the
    architecture of the current default device. If [with_debug] is [true], the compilation log is
    included even in case of compilation success (see {!compilation_log}).

    NOTE: unlike NVRTC, hiprtc ships built-in HIP headers, so no include path is prepended to
    [options]. *)

val string_from_code : compile_to_code_result -> string
(** The stored code object binary as a string of bytes (not human-readable, suitable for e.g.
    writing to a [.hsaco] file), see [hiprtcGetCode]. *)

val compilation_log : compile_to_code_result -> string option
(** The stored side output of the compilation, see [hiprtcGetProgramLog]. *)
