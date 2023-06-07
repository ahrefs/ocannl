open Nvrtc_ffi.Bindings_types
module Nvrtc = Nvrtc_ffi.C.Functions
module Cuda = Cuda_ffi.C.Functions
open Cuda_ffi.Bindings_types

type error_code = Nvrtc_error of nvrtc_result | Cuda_error of cu_result

exception Error of { status : error_code; message : string }

type compile_to_ptx_result = { log : string option; ptx : char Ctypes.ptr; ptx_length : int }

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
    ignore @@ Nvrtc.nvrtc_destroy_program prog;
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
  ignore @@ Nvrtc.nvrtc_destroy_program prog;
  { log; ptx; ptx_length = count - 1 }

let string_from_ptx prog = Ctypes.string_from_ptr prog.ptx ~length:prog.ptx_length

let check message status =
  if status <> CUDA_SUCCESS then raise @@ Error { status = Cuda_error status; message }

let cu_init flags = check "cu_init" @@ Cuda.cu_init flags

let cu_device_get_count () =
  let open Ctypes in
  let count = allocate int 0 in
  check "cu_device_get_count" @@ Cuda.cu_device_get_count count;
  !@count

let cu_device_get ~ordinal =
  let open Ctypes in
  let device = allocate Cuda_ffi.Types_generated.cu_device (Cu_device 0) in
  check "cu_device_get" @@ Cuda.cu_device_get device ordinal;
  !@device

let cu_ctx_create ~flags cu_device =
  let open Ctypes in
  let ctx = allocate_n cu_context ~count:1 in
  check "cu_ctx_create" @@ Cuda.cu_ctx_create ctx flags cu_device;
  !@ctx

type bigstring = (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(* Note: bool corresponds to C int (0=false). *)
type jit_option =
  | JIT_MAX_REGISTERS of int
  | JIT_THREADS_PER_BLOCK of int
  | JIT_WALL_TIME of { milliseconds : float }
  | JIT_INFO_LOG_BUFFER of bigstring
  | JIT_ERROR_LOG_BUFFER of bigstring
  | JIT_OPTIMIZATION_LEVEL of int
  | JIT_TARGET_FROM_CUCONTEXT
  | JIT_TARGET of cu_jit_target
  | JIT_FALLBACK_STRATEGY of cu_jit_fallback
  | JIT_GENERATE_DEBUG_INFO of bool
  | JIT_LOG_VERBOSE of bool
  | JIT_GENERATE_LINE_INFO of bool
  | JIT_CACHE_MODE of cu_jit_cache_mode
(* | JIT_POSITION_INDEPENDENT_CODE of bool *)

let uint_of_cu_jit_target c =
  let open Cuda_ffi.Types_generated in
  match c with
  | CU_TARGET_COMPUTE_30 -> Unsigned.UInt.of_int64 cu_target_compute_30
  | CU_TARGET_COMPUTE_32 -> Unsigned.UInt.of_int64 cu_target_compute_32
  | CU_TARGET_COMPUTE_35 -> Unsigned.UInt.of_int64 cu_target_compute_35
  | CU_TARGET_COMPUTE_37 -> Unsigned.UInt.of_int64 cu_target_compute_37
  | CU_TARGET_COMPUTE_50 -> Unsigned.UInt.of_int64 cu_target_compute_50
  | CU_TARGET_COMPUTE_52 -> Unsigned.UInt.of_int64 cu_target_compute_52
  | CU_TARGET_COMPUTE_53 -> Unsigned.UInt.of_int64 cu_target_compute_53
  | CU_TARGET_COMPUTE_60 -> Unsigned.UInt.of_int64 cu_target_compute_60
  | CU_TARGET_COMPUTE_61 -> Unsigned.UInt.of_int64 cu_target_compute_61
  | CU_TARGET_COMPUTE_62 -> Unsigned.UInt.of_int64 cu_target_compute_62
  | CU_TARGET_COMPUTE_70 -> Unsigned.UInt.of_int64 cu_target_compute_70
  | CU_TARGET_COMPUTE_72 -> Unsigned.UInt.of_int64 cu_target_compute_72
  | CU_TARGET_COMPUTE_75 -> Unsigned.UInt.of_int64 cu_target_compute_75
  | CU_TARGET_COMPUTE_80 -> Unsigned.UInt.of_int64 cu_target_compute_80
  | CU_TARGET_COMPUTE_86 -> Unsigned.UInt.of_int64 cu_target_compute_86
  (* | CU_TARGET_COMPUTE_87 -> Unsigned.UInt.of_int64 cu_target_compute_87
     | CU_TARGET_COMPUTE_89 -> Unsigned.UInt.of_int64 cu_target_compute_89
     | CU_TARGET_COMPUTE_90 -> Unsigned.UInt.of_int64 cu_target_compute_90
     | CU_TARGET_COMPUTE_90A -> Unsigned.UInt.of_int64 cu_target_compute_90a *)
  | CU_TARGET_UNCATEGORIZED c -> Unsigned.UInt.of_int64 c

let uint_of_cu_jit_fallback c =
  let open Cuda_ffi.Types_generated in
  match c with
  | CU_PREFER_PTX -> Unsigned.UInt.of_int64 cu_prefer_ptx
  | CU_PREFER_BINARY -> Unsigned.UInt.of_int64 cu_prefer_binary
  | CU_PREFER_UNCATEGORIZED c -> Unsigned.UInt.of_int64 c

let uint_of_cu_jit_cache_mode c =
  let open Cuda_ffi.Types_generated in
  match c with
  | CU_JIT_CACHE_OPTION_NONE -> Unsigned.UInt.of_int64 cu_jit_cache_option_none
  | CU_JIT_CACHE_OPTION_CG -> Unsigned.UInt.of_int64 cu_jit_cache_option_cg
  | CU_JIT_CACHE_OPTION_CA -> Unsigned.UInt.of_int64 cu_jit_cache_option_ca
  | CU_JIT_CACHE_OPTION_UNCATEGORIZED c -> Unsigned.UInt.of_int64 c

let cu_module_load_data_ex ptx options =
  let open Ctypes in
  let cu_mod = allocate_n cu_module ~count:1 in
  let n_opts = List.length options in
  let c_options =
    CArray.of_list Cuda_ffi.Types_generated.cu_jit_option
    @@ List.concat_map
         (function
           | JIT_MAX_REGISTERS _ -> [ CU_JIT_MAX_REGISTERS ]
           | JIT_THREADS_PER_BLOCK _ -> [ CU_JIT_THREADS_PER_BLOCK ]
           | JIT_WALL_TIME _ -> [ CU_JIT_WALL_TIME ]
           | JIT_INFO_LOG_BUFFER _ -> [ CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES; CU_JIT_INFO_LOG_BUFFER ]
           | JIT_ERROR_LOG_BUFFER _ -> [ CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES; CU_JIT_ERROR_LOG_BUFFER ]
           | JIT_OPTIMIZATION_LEVEL _ -> [ CU_JIT_OPTIMIZATION_LEVEL ]
           | JIT_TARGET_FROM_CUCONTEXT -> [ CU_JIT_TARGET_FROM_CUCONTEXT ]
           | JIT_TARGET _ -> [ CU_JIT_TARGET ]
           | JIT_FALLBACK_STRATEGY _ -> [ CU_JIT_FALLBACK_STRATEGY ]
           | JIT_GENERATE_DEBUG_INFO _ -> [ CU_JIT_GENERATE_DEBUG_INFO ]
           | JIT_LOG_VERBOSE _ -> [ CU_JIT_LOG_VERBOSE ]
           | JIT_GENERATE_LINE_INFO _ -> [ CU_JIT_GENERATE_LINE_INFO ]
           | JIT_CACHE_MODE _ ->
               [ CU_JIT_CACHE_MODE ]
               (* | JIT_POSITION_INDEPENDENT_CODE _ -> [CU_JIT_POSITION_INDEPENDENT_CODE] *))
         options
  in
  let i2u2vp i = coerce (ptr uint) (ptr void) @@ allocate uint @@ Unsigned.UInt.of_int i in
  let u2vp u = coerce (ptr uint) (ptr void) @@ allocate uint u in
  let f2vp f = coerce (ptr float) (ptr void) @@ allocate float f in
  let i2vp i = coerce (ptr int) (ptr void) @@ allocate int i in
  let bi2vp b = coerce (ptr int) (ptr void) @@ allocate int (if b then 1 else 0) in
  let ba2vp b = coerce (ptr char) (ptr void) @@ bigarray_start Ctypes.array1 b in
  let c_opts_args =
    CArray.of_list (ptr void)
    @@ List.concat_map
         (function
           | JIT_MAX_REGISTERS v -> [ i2u2vp v ]
           | JIT_THREADS_PER_BLOCK v -> [ i2u2vp v ]
           | JIT_WALL_TIME { milliseconds } -> [ f2vp milliseconds ]
           | JIT_INFO_LOG_BUFFER b ->
               let size = u2vp @@ Unsigned.UInt.of_int @@ Bigarray.Array1.size_in_bytes b in
               [ size; ba2vp b ]
           | JIT_ERROR_LOG_BUFFER b ->
               let size = u2vp @@ Unsigned.UInt.of_int @@ Bigarray.Array1.size_in_bytes b in
               [ size; ba2vp b ]
           | JIT_OPTIMIZATION_LEVEL i -> [ i2vp i ]
           | JIT_TARGET_FROM_CUCONTEXT -> [ null ]
           | JIT_TARGET t -> [ u2vp @@ uint_of_cu_jit_target t ]
           | JIT_FALLBACK_STRATEGY t -> [ u2vp @@ uint_of_cu_jit_fallback t ]
           | JIT_GENERATE_DEBUG_INFO c -> [ bi2vp c ]
           | JIT_LOG_VERBOSE c -> [ bi2vp c ]
           | JIT_GENERATE_LINE_INFO c -> [ bi2vp c ]
           | JIT_CACHE_MODE t ->
               [ u2vp @@ uint_of_cu_jit_cache_mode t ] (* | JIT_POSITION_INDEPENDENT_CODE c -> [ bi2vp c ] *))
         options
  in
  (* allocate_n Cuda_ffi.Types_generated.cu_jit_option ~count:n_opts in *)
  check "cu_module_load_data_ex"
  @@ Cuda.cu_module_load_data_ex cu_mod
       (coerce (ptr char) (ptr void) ptx.ptx)
       n_opts (CArray.start c_options) @@ CArray.start c_opts_args;
  !@cu_mod

let cu_module_get_function module_ ~name =
  let open Ctypes in
  let func = allocate_n cu_function ~count:1 in
  check "cu_module_get_function" @@ Cuda.cu_module_get_function func module_ name;
  !@func

let cu_mem_alloc ~byte_size =
  let open Ctypes in
  let device = allocate_n cu_deviceptr ~count:1 in
  check "cu_mem_alloc" @@ Cuda.cu_mem_alloc device @@ Unsigned.Size_t.of_int byte_size;
  !@device

let cu_memcpy_H_to_D ~dst_device ~src_host ~byte_size =
  check "cu_memcpy_H_to_D" @@ Cuda.cu_memcpy_H_to_D dst_device src_host @@ Unsigned.Size_t.of_int byte_size

