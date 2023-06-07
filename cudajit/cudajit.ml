open Nvrtc_ffi.Bindings_types
module Nvrtc = Nvrtc_ffi.C.Functions
open Cuda_ffi.Bindings_types

type error_code = Nvrtc_error of nvrtc_result | Cuda_error of cuda_result

exception Error of { status : error_code; message : string }

type compile_to_ptx_result = { log : string option; ptx : string }

let compile_to_ptx ~cu_src ~name ~options ~with_debug =
  let open Ctypes in
  let prog = allocate_n nvrtc_program ~count:1 in
  (* TODO: support headers / includes in the cuda sources. *)
  let status =
    Nvrtc.nvrtc_create_program prog cu_src name 0 (from_voidp string null) (from_voidp string null)
  in
  if status <> NVRTC_SUCCESS then
    raise @@ Error { status = Nvrtc_error status; message = "nvrtc_create_program " ^ name };
  let num_options = List.length options in
  let c_options = CArray.make (ptr char) num_options in

  List.iteri (fun i v -> CArray.of_string v |> CArray.start |> CArray.set c_options i) options;
  let status = Nvrtc.nvrtc_compile_program !@prog num_options @@ CArray.start c_options in
  let log_msg log = Option.value log ~default:"no compilation log" in
  let error prefix status log =
    raise @@ Error { status = Nvrtc_error status; message = prefix ^ " " ^ name ^ ": " ^ log_msg log }
  in
  let log =
    if status = NVRTC_SUCCESS && not with_debug then None
    else
      let log_size = allocate size_t Unsigned.Size_t.zero in
      let status = Nvrtc.nvrtc_get_program_log_size !@prog log_size in
      if status <> NVRTC_SUCCESS then None
      else
        let count = Unsigned.Size_t.to_int !@log_size in
        let log = allocate_n char ~count in
        let status = Nvrtc.nvrtc_get_program_log !@prog log in
        if status = NVRTC_SUCCESS then Some (string_from_ptr log ~length:(count - 1)) else None
  in
  if status <> NVRTC_SUCCESS then error "nvrtc_compile_program" status log;
  let ptx_size = allocate size_t Unsigned.Size_t.zero in
  let status = Nvrtc.nvrtc_get_PTX_size !@prog ptx_size in
  if status <> NVRTC_SUCCESS then error "nvrtc_get_PTX_size" status log;
  let count = Unsigned.Size_t.to_int !@ptx_size in
  let ptx = allocate_n char ~count in
  let status = Nvrtc.nvrtc_get_PTX !@prog ptx in
  if status <> NVRTC_SUCCESS then error "nvrtc_get_PTX" status log;
  { log; ptx = string_from_ptr ptx ~length:(count - 1) }
