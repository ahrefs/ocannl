module Impl : sig
  include Ir.Backend_impl.Lowered_backend

  val routine_name_probe : unit -> Ir.C_syntax.routine_name_probe
  (** What [tools/vendor_idents.exe] harvests this backend's routine-name table through
      (gh-ocannl-1338): the kernel prelude, nvrtc under the options a kernel compiles with (so it
      initializes the devices), and the routine-name mangling. *)
end

val cuda_include_options : unit -> string list
(** The [-I] options an nvrtc compile needs in order to find the CUDA toolkit headers
    ([<cuda_fp8.h>], [<cuda_fp16.h>], [<mma.h>]): CUDA_PATH where it is set, the no-spaces junction
    ocaml-cudajit creates on Windows, and /usr/local/cuda as the Linux fallback. Separated out of
    the compile path so that other nvrtc callers agree with it rather than reimplementing a subset —
    tools/fp8_soak.ml is one, and a subset that knew only /usr/local/cuda would fail to compile its
    kernel wherever the toolkit lives elsewhere. *)

val host_capabilities : unit -> (string * string) list
(** What this backend probes on the host without opening a device (gh-ocannl-1038): the toolkit
    include tree {!cuda_include_options} found ([cuda_include], in [Utils.capability_dir]'s form:
    where it resolves, which is what moves when [/usr/local/cuda] is repointed at a new toolkit, and
    whether the headers nvrtc compiles against are there, with a fingerprint of them).
    {!Context.host_capabilities} collects it. *)
