module Impl : sig
  include Ir.Backend_impl.Lowered_backend

  val hip_to_code : name:string -> string -> Hiprtc.compile_to_code_result
  (** Compile one complete HIP source through the same option assembly as backend-generated kernels.
      Exposed for hardware-backed compiler regression probes; callers normally enter through
      {!compile}. *)
end

val hip_include_options : unit -> string list
(** The [-I] options a hiprtc compile needs in order to find the HIP headers ([<hip/hip_fp16.h>],
    [<hip/hip_fp8.h>]): the no-spaces junction ocaml-hipjit creates on Windows, HIP_PATH, then
    /opt/rocm and the distro tree /usr as the fallbacks; empty on a Linux box where hiprtc's
    built-in headers suffice. Separated out of the compile path so that other hiprtc callers agree
    with it rather than reimplementing a subset — tools/fp8_soak.ml's HIP arm is one, and the CUDA
    half of that program learned in review what a divergent guess costs (a soak that probes "ready"
    and then fails to compile wherever the SDK is not where the guess looked). *)

val fp8_guard_source : unit -> string
(** The device-side source text of the two guarded narrowing helpers HIP emits in place of a bare
    float-to-e5m2 cast (gh-ocannl-647): [ocannl_single_to_fp8_uniform] and
    [ocannl_double_to_fp8_uniform], concatenated, exactly as {!Builtins_hip} hands them to a kernel.

    Exposed so that a program verifying the guard sweeps the SHIPPED definitions rather than a
    transcription of them — the same discipline as the host side of tools/fp8_soak.ml, which reaches
    [builtins.c]'s codec by [extern] instead of restating it. Raises [Utils.User_error] if either
    helper is renamed out from under it, so the verification cannot quietly become vacuous. *)
