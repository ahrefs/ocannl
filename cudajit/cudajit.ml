open Nvrtc_ffi.Bindings_types
module Nvrtc = Nvrtc_ffi.C.Functions
open Cuda_ffi.Bindings_types

type error_code = Nvrtc_error of nvrtc_result | Cuda_error of cuda_result

exception Error of { status : error_code; message : string }

let compile_to_ptx ~cu_src ~name =
  let open Ctypes in
  let prog = allocate_n nvrtc_program ~count:1 in
  (* TODO: support headers / includes in the cuda sources. *)
  let status =
    Nvrtc.nvrtc_create_program prog cu_src name 0 (from_voidp string null) (from_voidp string null)
  in
  if status <> NVRTC_SUCCESS then
    raise @@ Error { status = Nvrtc_error status; message = "nvrtc_create_program " ^ name }
