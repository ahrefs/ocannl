module Impl : Ir.Backend_impl.Lowered_backend

val cuda_include_options : unit -> string list
(** The [-I] options an nvrtc compile needs in order to find the CUDA toolkit headers
    ([<cuda_fp8.h>], [<cuda_fp16.h>], [<mma.h>]): CUDA_PATH where it is set, the no-spaces junction
    ocaml-cudajit creates on Windows, and /usr/local/cuda as the Linux fallback. Separated out of
    the compile path so that other nvrtc callers agree with it rather than reimplementing a subset —
    tools/fp8_soak.ml is one, and a subset that knew only /usr/local/cuda would fail to compile its
    kernel wherever the toolkit lives elsewhere. *)
