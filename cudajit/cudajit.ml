open Nvrtc_ffi.Bindings_types
module Nvrtc = Nvrtc_ffi.C.Functions
module Cuda = Cuda_ffi.C.Functions
open Cuda_ffi.Bindings_types
open Sexplib0.Sexp_conv

type error_code = Nvrtc_error of nvrtc_result | Cuda_error of cu_result [@@deriving sexp]

exception Error of { status : error_code; message : string }

let error_printer = function
  | Error { status; message } ->
      ignore @@ Format.flush_str_formatter ();
      Format.fprintf Format.str_formatter "%s:@ %a" message Sexplib0.Sexp.pp_hum (sexp_of_error_code status);
      Some (Format.flush_str_formatter ())
  | _ -> None

let () = Printexc.register_printer error_printer

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

let init ?(flags = 0) () = check "cu_init" @@ Cuda.cu_init flags

let device_get_count () =
  let open Ctypes in
  let count = allocate int 0 in
  check "cu_device_get_count" @@ Cuda.cu_device_get_count count;
  !@count

let device_get ~ordinal =
  let open Ctypes in
  let device = allocate Cuda_ffi.Types_generated.cu_device (Cu_device 0) in
  check "cu_device_get" @@ Cuda.cu_device_get device ordinal;
  !@device

let ctx_create ~flags device =
  let open Ctypes in
  let ctx = allocate_n cu_context ~count:1 in
  check "cu_ctx_create" @@ Cuda.cu_ctx_create ctx flags device;
  !@ctx

type bigstring = (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(* Note: bool corresponds to C int (0=false). *)
type jit_option =
  | JIT_MAX_REGISTERS of int
  | JIT_THREADS_PER_BLOCK of int
  | JIT_WALL_TIME of { milliseconds : float }
  | JIT_INFO_LOG_BUFFER of (bigstring[@sexp.opaque])
  | JIT_ERROR_LOG_BUFFER of (bigstring[@sexp.opaque])
  | JIT_OPTIMIZATION_LEVEL of int
  | JIT_TARGET_FROM_CUCONTEXT
  | JIT_TARGET of cu_jit_target
  | JIT_FALLBACK_STRATEGY of cu_jit_fallback
  | JIT_GENERATE_DEBUG_INFO of bool
  | JIT_LOG_VERBOSE of bool
  | JIT_GENERATE_LINE_INFO of bool
  | JIT_CACHE_MODE of cu_jit_cache_mode
(* | JIT_POSITION_INDEPENDENT_CODE of bool *)
[@@deriving sexp]

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

let module_load_data_ex ptx options =
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
  @@ Cuda.cu_module_load_data_ex cu_mod (coerce (ptr char) (ptr void) ptx.ptx) n_opts (CArray.start c_options)
  @@ CArray.start c_opts_args;
  !@cu_mod

let module_get_function module_ ~name =
  let open Ctypes in
  let func = allocate_n cu_function ~count:1 in
  check "cu_module_get_function" @@ Cuda.cu_module_get_function func module_ name;
  !@func

type deviceptr = Deviceptr of Unsigned.uint64

let mem_alloc ~byte_size =
  let open Ctypes in
  let device = allocate_n cu_deviceptr ~count:1 in
  check "cu_mem_alloc" @@ Cuda.cu_mem_alloc device @@ Unsigned.Size_t.of_int byte_size;
  Deviceptr !@device

let memcpy_H_to_D ?host_offset ?length ~dst:(Deviceptr dst) ~src () =
  let full_size = Bigarray.Genarray.size_in_bytes src in
  let c_typ = Ctypes.typ_of_bigarray_kind @@ Bigarray.Genarray.kind src in
  let elem_bytes = Ctypes.sizeof c_typ in
  let byte_size =
    match (host_offset, length) with
    | None, None -> full_size
    | Some offset, None -> full_size - (elem_bytes * offset)
    | None, Some length -> elem_bytes * length
    | Some offset, Some length -> elem_bytes * (length - offset)
  in
  let open Ctypes in
  let host = bigarray_start genarray src in
  let host = match host_offset with None -> host | Some offset -> host +@ offset in
  check "cu_memcpy_H_to_D"
  @@ Cuda.cu_memcpy_H_to_D dst (coerce (ptr c_typ) (ptr void) host)
  @@ Unsigned.Size_t.of_int byte_size

let alloc_and_memcpy src =
  let byte_size = Bigarray.Genarray.size_in_bytes src in
  let dst = mem_alloc ~byte_size in
  memcpy_H_to_D ~dst ~src ();
  dst

type kernel_param =
  | Tensor of deviceptr
  | Int of int
  | Size_t of Unsigned.size_t
  | Single of float
  | Double of float

let no_stream = Ctypes.(coerce (ptr void) cu_stream null)

let launch_kernel func ~grid_dim_x ?(grid_dim_y = 1) ?(grid_dim_z = 1) ~block_dim_x ?(block_dim_y = 1)
    ?(block_dim_z = 1) ~shared_mem_bytes stream kernel_params =
  let i2u = Unsigned.UInt.of_int in
  let open Ctypes in
  let c_kernel_params =
    List.map
      (function
        | Tensor (Deviceptr dev) -> coerce (ptr uint64_t) (ptr void) @@ allocate uint64_t dev
        | Int i -> coerce (ptr int) (ptr void) @@ allocate int i
        | Size_t u -> coerce (ptr size_t) (ptr void) @@ allocate size_t u
        | Single u -> coerce (ptr float) (ptr void) @@ allocate float u
        | Double u -> coerce (ptr double) (ptr void) @@ allocate double u)
      kernel_params
    |> CArray.of_list (ptr void)
    |> CArray.start
  in
  check "cu_launch_kernel"
  @@ Cuda.cu_launch_kernel func (i2u grid_dim_x) (i2u grid_dim_y) (i2u grid_dim_z) (i2u block_dim_x)
       (i2u block_dim_y) (i2u block_dim_z) (i2u shared_mem_bytes) stream c_kernel_params
  @@ coerce (ptr void) (ptr @@ ptr void) null

let ctx_synchronize () = check "cu_ctx_synchronize" @@ Cuda.cu_ctx_synchronize ()

let memcpy_D_to_H ?host_offset ?length ~dst ~src:(Deviceptr src) () =
  let full_size = Bigarray.Genarray.size_in_bytes dst in
  let c_typ = Ctypes.typ_of_bigarray_kind @@ Bigarray.Genarray.kind dst in
  let elem_bytes = Ctypes.sizeof c_typ in
  let byte_size =
    match (host_offset, length) with
    | None, None -> full_size
    | Some offset, None -> full_size - (elem_bytes * offset)
    | None, Some length -> elem_bytes * length
    | Some offset, Some length -> elem_bytes * (length - offset)
  in
  let open Ctypes in
  let host = bigarray_start genarray dst in
  let host = match host_offset with None -> host | Some offset -> host +@ offset in
  check "cu_memcpy_D_to_H"
  @@ Cuda.cu_memcpy_D_to_H (coerce (ptr c_typ) (ptr void) host) src
  @@ Unsigned.Size_t.of_int byte_size

let mem_free (Deviceptr dev) = check "cu_mem_free" @@ Cuda.cu_mem_free dev
let module_unload cu_mod = check "cu_module_unload" @@ Cuda.cu_module_unload cu_mod
let ctx_destroy ctx = check "cu_ctx_destroy" @@ Cuda.cu_ctx_destroy ctx

type device_attributes = {
  name : string;
  max_threads_per_block : int;
  max_block_dim_x : int;
  max_block_dim_y : int;
  max_block_dim_z : int;
  max_grid_dim_x : int;
  max_grid_dim_y : int;
  max_grid_dim_z : int;
  max_shared_memory_per_block : int;  (** In bytes. *)
  total_constant_memory : int;  (** In bytes. *)
  warp_size : int;  (** In threads. *)
  max_pitch : int;  (** In bytes. *)
  max_registers_per_block : int;  (** 32-bit registers. *)
  clock_rate : int;  (** In kilohertz. *)
  texture_alignment : int;
  multiprocessor_count : int;
  kernel_exec_timeout : bool;
  integrated : bool;
  can_map_host_memory : bool;
  compute_mode : cu_computemode;
  maximum_texture1d_width : int;
  maximum_texture2d_width : int;
  maximum_texture2d_height : int;
  maximum_texture3d_width : int;
  maximum_texture3d_height : int;
  maximum_texture3d_depth : int;
  maximum_texture2d_layered_width : int;
  maximum_texture2d_layered_height : int;
  maximum_texture2d_layered_layers : int;
  surface_alignment : int;
  concurrent_kernels : bool;
  ecc_enabled : bool;
  pci_bus_id : int;
  pci_device_id : int;
  tcc_driver : bool;
  memory_clock_rate : int;  (** In kilohertz. *)
  global_memory_bus_width : int;  (** In bits. *)
  l2_cache_size : int;  (** In bytes. *)
  max_threads_per_multiprocessor : int;
  async_engine_count : int;
  (* unified_addressing: bool; *)
  maximum_texture1d_layered_width : int;
  maximum_texture1d_layered_layers : int;
  maximum_texture2d_gather_width : int;
  maximum_texture2d_gather_height : int;
  maximum_texture3d_width_alternate : int;
  maximum_texture3d_height_alternate : int;
  maximum_texture3d_depth_alternate : int;
  pci_domain_id : int;
  texture_pitch_alignment : int;
  maximum_texturecubemap_width : int;
  maximum_texturecubemap_layered_width : int;
  maximum_texturecubemap_layered_layers : int;
  maximum_surface1d_width : int;
  maximum_surface2d_width : int;
  maximum_surface2d_height : int;
  maximum_surface3d_width : int;
  maximum_surface3d_height : int;
  maximum_surface3d_depth : int;
  maximum_surface1d_layered_width : int;
  maximum_surface1d_layered_layers : int;
  maximum_surface2d_layered_width : int;
  maximum_surface2d_layered_height : int;
  maximum_surface2d_layered_layers : int;
  maximum_surfacecubemap_width : int;
  maximum_surfacecubemap_layered_width : int;
  maximum_surfacecubemap_layered_layers : int;
  maximum_texture2d_linear_width : int;
  maximum_texture2d_linear_height : int;
  maximum_texture2d_linear_pitch : int;  (** In bytes. *)
  maximum_texture2d_mipmapped_width : int;
  maximum_texture2d_mipmapped_height : int;
  compute_capability_major : int;
  compute_capability_minor : int;
  maximum_texture1d_mipmapped_width : int;
  stream_priorities_supported : bool;
  global_l1_cache_supported : bool;
  local_l1_cache_supported : bool;
  max_shared_memory_per_multiprocessor : int;  (** In bytes. *)
  max_registers_per_multiprocessor : int;  (** 32-bit registers. *)
  managed_memory : bool;
  multi_gpu_board : bool;
  multi_gpu_board_group_id : int;
  host_native_atomic_supported : bool;
  single_to_double_precision_perf_ratio : int;
  pageable_memory_access : bool;
  concurrent_managed_access : bool;
  compute_preemption_supported : bool;
  can_use_host_pointer_for_registered_mem : bool;
  cooperative_launch : bool;
  max_shared_memory_per_block_optin : int;
  can_flush_remote_writes : bool;
  host_register_supported : bool;
  pageable_memory_access_uses_host_page_tables : bool;
  direct_managed_mem_access_from_host : bool;
  virtual_memory_management_supported : bool;
  handle_type_posix_file_descriptor_supported : bool;
  handle_type_win32_handle_supported : bool;
  handle_type_win32_kmt_handle_supported : bool;
  max_blocks_per_multiprocessor : int;
  generic_compression_supported : bool;
  max_persisting_l2_cache_size : int;  (** In bytes. *)
  max_access_policy_window_size : int;  (** For [CUaccessPolicyWindow::num_bytes]. *)
  gpu_direct_rdma_with_cuda_vmm_supported : bool;
  reserved_shared_memory_per_block : int;  (** In bytes. *)
  sparse_cuda_array_supported : bool;
  read_only_host_register_supported : bool;
  timeline_semaphore_interop_supported : bool;
  (* memory_pools_supported: bool; *)
  gpu_direct_rdma_supported : bool;
  gpu_direct_rdma_flush_writes_options : cu_flush_GPU_direct_RDMA_writes_options list;
  gpu_direct_rdma_writes_ordering : bool;
  mempool_supported_handle_types : bool;
  (* cluster_launch: bool; *)
  (* deferred_mapping_cuda_array_supported: bool; *)
  can_use_64_bit_stream_mem_ops : bool;
  can_use_stream_wait_value_nor : bool;
      (* dma_buf_supported: bool; *)
      (* ipc_event_supported: bool; *)
      (* mem_sync_domain_count: int; *)
      (* tensor_map_access_supported: bool; *)
      (* unified_function_pointers: bool; *)
      (* multicast_supported: bool; *)
}
[@@deriving sexp]

let device_get_attributes device =
  let open Ctypes in
  let count = 2048 in
  let name = allocate_n char ~count in
  check "cu_device_get_name" @@ Cuda.cu_device_get_name name count device;
  let name = coerce (ptr char) string name in
  let max_threads_per_block = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_threads_per_block CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK device;
  let max_threads_per_block = !@max_threads_per_block in
  let max_block_dim_x = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_block_dim_x CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X device;
  let max_block_dim_x = !@max_block_dim_x in
  let max_block_dim_y = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_block_dim_y CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y device;
  let max_block_dim_y = !@max_block_dim_y in
  let max_block_dim_z = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_block_dim_z CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z device;
  let max_block_dim_z = !@max_block_dim_z in
  let max_grid_dim_x = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_grid_dim_x CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X device;
  let max_grid_dim_x = !@max_grid_dim_x in
  let max_grid_dim_y = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_grid_dim_y CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y device;
  let max_grid_dim_y = !@max_grid_dim_y in
  let max_grid_dim_z = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_grid_dim_z CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z device;
  let max_grid_dim_z = !@max_grid_dim_z in
  let max_shared_memory_per_block = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_shared_memory_per_block CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
       device;
  let max_shared_memory_per_block = !@max_shared_memory_per_block in
  let total_constant_memory = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute total_constant_memory CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY device;
  let total_constant_memory = !@total_constant_memory in
  let warp_size = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute warp_size CU_DEVICE_ATTRIBUTE_WARP_SIZE device;
  let warp_size = !@warp_size in
  let max_pitch = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_pitch CU_DEVICE_ATTRIBUTE_MAX_PITCH device;
  let max_pitch = !@max_pitch in
  let max_registers_per_block = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_registers_per_block CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK device;
  let max_registers_per_block = !@max_registers_per_block in
  let clock_rate = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute clock_rate CU_DEVICE_ATTRIBUTE_CLOCK_RATE device;
  let clock_rate = !@clock_rate in
  let texture_alignment = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute texture_alignment CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT device;
  let texture_alignment = !@texture_alignment in
  let multiprocessor_count = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute multiprocessor_count CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT device;
  let multiprocessor_count = !@multiprocessor_count in
  let kernel_exec_timeout = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute kernel_exec_timeout CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT device;
  let kernel_exec_timeout = 0 <> !@kernel_exec_timeout in
  let integrated = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute integrated CU_DEVICE_ATTRIBUTE_INTEGRATED device;
  let integrated = 0 <> !@integrated in
  let can_map_host_memory = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute can_map_host_memory CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY device;
  let can_map_host_memory = 0 <> !@can_map_host_memory in
  let compute_mode = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute compute_mode CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY device;
  let compute_mode = Cuda.cu_computemode_of_int !@compute_mode in
  let maximum_texture1d_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture1d_width CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH device;
  let maximum_texture1d_width = !@maximum_texture1d_width in
  let maximum_texture2d_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture2d_width CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH device;
  let maximum_texture2d_width = !@maximum_texture2d_width in
  let maximum_texture2d_height = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture2d_height CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT device;
  let maximum_texture2d_height = !@maximum_texture2d_height in
  let maximum_texture3d_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture3d_width CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH device;
  let maximum_texture3d_width = !@maximum_texture3d_width in
  let maximum_texture3d_height = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture3d_height CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT device;
  let maximum_texture3d_height = !@maximum_texture3d_height in
  let maximum_texture3d_depth = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture3d_depth CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH device;
  let maximum_texture3d_depth = !@maximum_texture3d_depth in
  let maximum_texture2d_layered_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture2d_layered_width
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH device;
  let maximum_texture2d_layered_width = !@maximum_texture2d_layered_width in
  let maximum_texture2d_layered_height = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture2d_layered_height
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT device;
  let maximum_texture2d_layered_height = !@maximum_texture2d_layered_height in
  let maximum_texture2d_layered_layers = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture2d_layered_layers
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS device;
  let maximum_texture2d_layered_layers = !@maximum_texture2d_layered_layers in
  let surface_alignment = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute surface_alignment CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT device;
  let surface_alignment = !@surface_alignment in
  let concurrent_kernels = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute concurrent_kernels CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS device;
  let concurrent_kernels = 0 <> !@concurrent_kernels in
  let ecc_enabled = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute ecc_enabled CU_DEVICE_ATTRIBUTE_ECC_ENABLED device;
  let ecc_enabled = 0 <> !@ecc_enabled in
  let pci_bus_id = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute pci_bus_id CU_DEVICE_ATTRIBUTE_PCI_BUS_ID device;
  let pci_bus_id = !@pci_bus_id in
  let pci_device_id = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute pci_device_id CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID device;
  let pci_device_id = !@pci_device_id in
  let tcc_driver = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute tcc_driver CU_DEVICE_ATTRIBUTE_TCC_DRIVER device;
  let tcc_driver = 0 <> !@tcc_driver in
  let memory_clock_rate = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute memory_clock_rate CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE device;
  let memory_clock_rate = !@memory_clock_rate in
  let global_memory_bus_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute global_memory_bus_width CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH device;
  let global_memory_bus_width = !@global_memory_bus_width in
  let l2_cache_size = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute l2_cache_size CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE device;
  let l2_cache_size = !@l2_cache_size in
  let max_threads_per_multiprocessor = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_threads_per_multiprocessor
       CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR device;
  let max_threads_per_multiprocessor = !@max_threads_per_multiprocessor in
  let async_engine_count = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute async_engine_count CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT device;
  let async_engine_count = !@async_engine_count in
  (* let unified_addressing = allocate int 0 in
     check "cu_device_get_attribute"
     @@ Cuda.cu_device_get_attribute unified_addressing CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING device;
     let unified_addressing = 0 <> !@unified_addressing in *)
  let maximum_texture1d_layered_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture1d_layered_width
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH device;
  let maximum_texture1d_layered_width = !@maximum_texture1d_layered_width in
  let maximum_texture1d_layered_layers = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture1d_layered_layers
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS device;
  let maximum_texture1d_layered_layers = !@maximum_texture1d_layered_layers in
  let maximum_texture2d_gather_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture2d_gather_width
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH device;
  let maximum_texture2d_gather_width = !@maximum_texture2d_gather_width in
  let maximum_texture2d_gather_height = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture2d_gather_height
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT device;
  let maximum_texture2d_gather_height = !@maximum_texture2d_gather_height in
  let maximum_texture3d_width_alternate = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture3d_width_alternate
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE device;
  let maximum_texture3d_width_alternate = !@maximum_texture3d_width_alternate in
  let maximum_texture3d_height_alternate = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture3d_height_alternate
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE device;
  let maximum_texture3d_height_alternate = !@maximum_texture3d_height_alternate in
  let maximum_texture3d_depth_alternate = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture3d_depth_alternate
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE device;
  let maximum_texture3d_depth_alternate = !@maximum_texture3d_depth_alternate in
  let pci_domain_id = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute pci_domain_id CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID device;
  let pci_domain_id = !@pci_domain_id in
  let texture_pitch_alignment = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute texture_pitch_alignment CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT device;
  let texture_pitch_alignment = !@texture_pitch_alignment in
  let maximum_texturecubemap_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texturecubemap_width
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH device;
  let maximum_texturecubemap_width = !@maximum_texturecubemap_width in
  let maximum_texturecubemap_layered_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texturecubemap_layered_width
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH device;
  let maximum_texturecubemap_layered_width = !@maximum_texturecubemap_layered_width in
  let maximum_texturecubemap_layered_layers = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texturecubemap_layered_layers
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS device;
  let maximum_texturecubemap_layered_layers = !@maximum_texturecubemap_layered_layers in
  let maximum_surface1d_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surface1d_width CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH device;
  let maximum_surface1d_width = !@maximum_surface1d_width in
  let maximum_surface2d_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surface2d_width CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH device;
  let maximum_surface2d_width = !@maximum_surface2d_width in
  let maximum_surface2d_height = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surface2d_height CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT device;
  let maximum_surface2d_height = !@maximum_surface2d_height in
  let maximum_surface3d_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surface3d_width CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH device;
  let maximum_surface3d_width = !@maximum_surface3d_width in
  let maximum_surface3d_height = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surface3d_height CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT device;
  let maximum_surface3d_height = !@maximum_surface3d_height in
  let maximum_surface3d_depth = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surface3d_depth CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH device;
  let maximum_surface3d_depth = !@maximum_surface3d_depth in
  let maximum_surface1d_layered_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surface1d_layered_width
       CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH device;
  let maximum_surface1d_layered_width = !@maximum_surface1d_layered_width in
  let maximum_surface1d_layered_layers = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surface1d_layered_layers
       CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS device;
  let maximum_surface1d_layered_layers = !@maximum_surface1d_layered_layers in
  let maximum_surface2d_layered_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surface2d_layered_width
       CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH device;
  let maximum_surface2d_layered_width = !@maximum_surface2d_layered_width in
  let maximum_surface2d_layered_height = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surface2d_layered_height
       CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT device;
  let maximum_surface2d_layered_height = !@maximum_surface2d_layered_height in
  let maximum_surface2d_layered_layers = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surface2d_layered_layers
       CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS device;
  let maximum_surface2d_layered_layers = !@maximum_surface2d_layered_layers in
  let maximum_surfacecubemap_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surfacecubemap_width
       CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH device;
  let maximum_surfacecubemap_width = !@maximum_surfacecubemap_width in
  let maximum_surfacecubemap_layered_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surfacecubemap_layered_width
       CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH device;
  let maximum_surfacecubemap_layered_width = !@maximum_surfacecubemap_layered_width in
  let maximum_surfacecubemap_layered_layers = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_surfacecubemap_layered_layers
       CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS device;
  let maximum_surfacecubemap_layered_layers = !@maximum_surfacecubemap_layered_layers in
  let maximum_texture2d_linear_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture2d_linear_width
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH device;
  let maximum_texture2d_linear_width = !@maximum_texture2d_linear_width in
  let maximum_texture2d_linear_height = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture2d_linear_height
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT device;
  let maximum_texture2d_linear_height = !@maximum_texture2d_linear_height in
  let maximum_texture2d_linear_pitch = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture2d_linear_pitch
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH device;
  let maximum_texture2d_linear_pitch = !@maximum_texture2d_linear_pitch in
  let maximum_texture2d_mipmapped_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture2d_mipmapped_width
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH device;
  let maximum_texture2d_mipmapped_width = !@maximum_texture2d_mipmapped_width in
  let maximum_texture2d_mipmapped_height = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture2d_mipmapped_height
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT device;
  let maximum_texture2d_mipmapped_height = !@maximum_texture2d_mipmapped_height in
  let compute_capability_major = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute compute_capability_major CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR device;
  let compute_capability_major = !@compute_capability_major in
  let compute_capability_minor = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute compute_capability_minor CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR device;
  let compute_capability_minor = !@compute_capability_minor in
  let maximum_texture1d_mipmapped_width = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute maximum_texture1d_mipmapped_width
       CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH device;
  let maximum_texture1d_mipmapped_width = !@maximum_texture1d_mipmapped_width in
  let stream_priorities_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute stream_priorities_supported CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED
       device;
  let stream_priorities_supported = 0 <> !@stream_priorities_supported in
  let global_l1_cache_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute global_l1_cache_supported CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED
       device;
  let global_l1_cache_supported = 0 <> !@global_l1_cache_supported in
  let local_l1_cache_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute local_l1_cache_supported CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED device;
  let local_l1_cache_supported = 0 <> !@local_l1_cache_supported in
  let max_shared_memory_per_multiprocessor = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_shared_memory_per_multiprocessor
       CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR device;
  let max_shared_memory_per_multiprocessor = !@max_shared_memory_per_multiprocessor in
  let max_registers_per_multiprocessor = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_registers_per_multiprocessor
       CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR device;
  let max_registers_per_multiprocessor = !@max_registers_per_multiprocessor in
  let managed_memory = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute managed_memory CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY device;
  let managed_memory = 0 <> !@managed_memory in
  let multi_gpu_board = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute multi_gpu_board CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD device;
  let multi_gpu_board = 0 <> !@multi_gpu_board in
  let multi_gpu_board_group_id = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute multi_gpu_board_group_id CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID device;
  let multi_gpu_board_group_id = !@multi_gpu_board_group_id in
  let host_native_atomic_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute host_native_atomic_supported
       CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED device;
  let host_native_atomic_supported = 0 <> !@host_native_atomic_supported in
  let single_to_double_precision_perf_ratio = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute single_to_double_precision_perf_ratio
       CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO device;
  let single_to_double_precision_perf_ratio = !@single_to_double_precision_perf_ratio in
  let pageable_memory_access = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute pageable_memory_access CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS device;
  let pageable_memory_access = 0 <> !@pageable_memory_access in
  let concurrent_managed_access = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute concurrent_managed_access CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
       device;
  let concurrent_managed_access = 0 <> !@concurrent_managed_access in
  let compute_preemption_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute compute_preemption_supported
       CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED device;
  let compute_preemption_supported = 0 <> !@compute_preemption_supported in
  let can_use_host_pointer_for_registered_mem = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute can_use_host_pointer_for_registered_mem
       CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM device;
  let can_use_host_pointer_for_registered_mem = 0 <> !@can_use_host_pointer_for_registered_mem in
  let cooperative_launch = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute cooperative_launch CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH device;
  let cooperative_launch = 0 <> !@cooperative_launch in
  let max_shared_memory_per_block_optin = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_shared_memory_per_block_optin
       CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN device;
  let max_shared_memory_per_block_optin = !@max_shared_memory_per_block_optin in
  let can_flush_remote_writes = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute can_flush_remote_writes CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES device;
  let can_flush_remote_writes = 0 <> !@can_flush_remote_writes in
  let host_register_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute host_register_supported CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED device;
  let host_register_supported = 0 <> !@host_register_supported in
  let pageable_memory_access_uses_host_page_tables = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute pageable_memory_access_uses_host_page_tables
       CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES device;
  let pageable_memory_access_uses_host_page_tables = 0 <> !@pageable_memory_access_uses_host_page_tables in
  let direct_managed_mem_access_from_host = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute direct_managed_mem_access_from_host
       CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST device;
  let direct_managed_mem_access_from_host = 0 <> !@direct_managed_mem_access_from_host in
  let virtual_memory_management_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute virtual_memory_management_supported
       CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED device;
  let virtual_memory_management_supported = 0 <> !@virtual_memory_management_supported in
  let handle_type_posix_file_descriptor_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute handle_type_posix_file_descriptor_supported
       CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED device;
  let handle_type_posix_file_descriptor_supported = 0 <> !@handle_type_posix_file_descriptor_supported in
  let handle_type_win32_handle_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute handle_type_win32_handle_supported
       CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED device;
  let handle_type_win32_handle_supported = 0 <> !@handle_type_win32_handle_supported in
  let handle_type_win32_kmt_handle_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute handle_type_win32_kmt_handle_supported
       CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED device;
  let handle_type_win32_kmt_handle_supported = 0 <> !@handle_type_win32_kmt_handle_supported in
  let max_blocks_per_multiprocessor = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_blocks_per_multiprocessor
       CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR device;
  let max_blocks_per_multiprocessor = !@max_blocks_per_multiprocessor in
  let generic_compression_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute generic_compression_supported
       CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED device;
  let generic_compression_supported = 0 <> !@generic_compression_supported in
  let max_persisting_l2_cache_size = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_persisting_l2_cache_size
       CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE device;
  let max_persisting_l2_cache_size = !@max_persisting_l2_cache_size in
  let max_access_policy_window_size = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute max_access_policy_window_size
       CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE device;
  let max_access_policy_window_size = !@max_access_policy_window_size in
  let gpu_direct_rdma_with_cuda_vmm_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute gpu_direct_rdma_with_cuda_vmm_supported
       CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED device;
  let gpu_direct_rdma_with_cuda_vmm_supported = 0 <> !@gpu_direct_rdma_with_cuda_vmm_supported in
  let reserved_shared_memory_per_block = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute reserved_shared_memory_per_block
       CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK device;
  let reserved_shared_memory_per_block = !@reserved_shared_memory_per_block in
  let sparse_cuda_array_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute sparse_cuda_array_supported CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED
       device;
  let sparse_cuda_array_supported = 0 <> !@sparse_cuda_array_supported in
  let read_only_host_register_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute read_only_host_register_supported
       CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED device;
  let read_only_host_register_supported = 0 <> !@read_only_host_register_supported in
  let timeline_semaphore_interop_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute timeline_semaphore_interop_supported
       CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED device;
  let timeline_semaphore_interop_supported = 0 <> !@timeline_semaphore_interop_supported in
  (* let memory_pools_supported = allocate int 0 in
     check "cu_device_get_attribute"
     @@ Cuda.cu_device_get_attribute memory_pools_supported CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED device;
     let memory_pools_supported = 0 <> !@memory_pools_supported in *)
  let gpu_direct_rdma_supported = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute gpu_direct_rdma_supported CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED
       device;
  let gpu_direct_rdma_supported = 0 <> !@gpu_direct_rdma_supported in
  let gpu_direct_rdma_flush_writes_options = [] in
  let gpu_direct_rdma_writes_ordering = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute gpu_direct_rdma_writes_ordering
       CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING device;
  let gpu_direct_rdma_writes_ordering = 0 <> !@gpu_direct_rdma_writes_ordering in
  let mempool_supported_handle_types = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute mempool_supported_handle_types
       CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES device;
  let mempool_supported_handle_types = 0 <> !@mempool_supported_handle_types in
  (* let cluster_launch = allocate int 0 in
     check "cu_device_get_attribute"
     @@ Cuda.cu_device_get_attribute cluster_launch CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH device;
     let cluster_launch = 0 <> !@cluster_launch in *)
  (* let deferred_mapping_cuda_array_supported = allocate int 0 in
     check "cu_device_get_attribute"
     @@ Cuda.cu_device_get_attribute deferred_mapping_cuda_array_supported CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED device;
     let deferred_mapping_cuda_array_supported = 0 <> !@deferred_mapping_cuda_array_supported in *)
  let can_use_64_bit_stream_mem_ops = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute can_use_64_bit_stream_mem_ops
       CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS device;
  let can_use_64_bit_stream_mem_ops = 0 <> !@can_use_64_bit_stream_mem_ops in
  let can_use_stream_wait_value_nor = allocate int 0 in
  check "cu_device_get_attribute"
  @@ Cuda.cu_device_get_attribute can_use_stream_wait_value_nor
       CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR device;
  let can_use_stream_wait_value_nor = 0 <> !@can_use_stream_wait_value_nor in
  (* let dma_buf_supported = allocate int 0 in
     check "cu_device_get_attribute"
     @@ Cuda.cu_device_get_attribute dma_buf_supported CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED device;
     let dma_buf_supported = 0 <> !@dma_buf_supported in *)
  (* let ipc_event_supported = allocate int 0 in
     check "cu_device_get_attribute"
     @@ Cuda.cu_device_get_attribute ipc_event_supported CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED device;
     let ipc_event_supported = 0 <> !@ipc_event_supported in *)
  (* let mem_sync_domain_count = allocate int 0 in
     check "cu_device_get_attribute"
     @@ Cuda.cu_device_get_attribute mem_sync_domain_count CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT device;
     let mem_sync_domain_count = !@mem_sync_domain_count in *)
  (* let tensor_map_access_supported = allocate int 0 in
     check "cu_device_get_attribute"
     @@ Cuda.cu_device_get_attribute tensor_map_access_supported CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED device;
     let tensor_map_access_supported = 0 <> !@tensor_map_access_supported in *)
  (* let unified_function_pointers = allocate int 0 in
     check "cu_device_get_attribute"
     @@ Cuda.cu_device_get_attribute unified_function_pointers CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS device;
     let unified_function_pointers = 0 <> !@unified_function_pointers in *)
  (* let multicast_supported = allocate int 0 in
     check "cu_device_get_attribute"
     @@ Cuda.cu_device_get_attribute multicast_supported CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED device;
     let multicast_supported = 0 <> !@multicast_supported in *)
  {
    name;
    max_threads_per_block;
    max_block_dim_x;
    max_block_dim_y;
    max_block_dim_z;
    max_grid_dim_x;
    max_grid_dim_y;
    max_grid_dim_z;
    max_shared_memory_per_block;
    total_constant_memory;
    warp_size;
    max_pitch;
    max_registers_per_block;
    clock_rate;
    texture_alignment;
    multiprocessor_count;
    kernel_exec_timeout;
    integrated;
    can_map_host_memory;
    compute_mode;
    maximum_texture1d_width;
    maximum_texture2d_width;
    maximum_texture2d_height;
    maximum_texture3d_width;
    maximum_texture3d_height;
    maximum_texture3d_depth;
    maximum_texture2d_layered_width;
    maximum_texture2d_layered_height;
    maximum_texture2d_layered_layers;
    surface_alignment;
    concurrent_kernels;
    ecc_enabled;
    pci_bus_id;
    pci_device_id;
    tcc_driver;
    memory_clock_rate;
    global_memory_bus_width;
    l2_cache_size;
    max_threads_per_multiprocessor;
    async_engine_count;
    (* unified_addressing; *)
    maximum_texture1d_layered_width;
    maximum_texture1d_layered_layers;
    maximum_texture2d_gather_width;
    maximum_texture2d_gather_height;
    maximum_texture3d_width_alternate;
    maximum_texture3d_height_alternate;
    maximum_texture3d_depth_alternate;
    pci_domain_id;
    texture_pitch_alignment;
    maximum_texturecubemap_width;
    maximum_texturecubemap_layered_width;
    maximum_texturecubemap_layered_layers;
    maximum_surface1d_width;
    maximum_surface2d_width;
    maximum_surface2d_height;
    maximum_surface3d_width;
    maximum_surface3d_height;
    maximum_surface3d_depth;
    maximum_surface1d_layered_width;
    maximum_surface1d_layered_layers;
    maximum_surface2d_layered_width;
    maximum_surface2d_layered_height;
    maximum_surface2d_layered_layers;
    maximum_surfacecubemap_width;
    maximum_surfacecubemap_layered_width;
    maximum_surfacecubemap_layered_layers;
    maximum_texture2d_linear_width;
    maximum_texture2d_linear_height;
    maximum_texture2d_linear_pitch;
    maximum_texture2d_mipmapped_width;
    maximum_texture2d_mipmapped_height;
    compute_capability_major;
    compute_capability_minor;
    maximum_texture1d_mipmapped_width;
    stream_priorities_supported;
    global_l1_cache_supported;
    local_l1_cache_supported;
    max_shared_memory_per_multiprocessor;
    max_registers_per_multiprocessor;
    managed_memory;
    multi_gpu_board;
    multi_gpu_board_group_id;
    host_native_atomic_supported;
    single_to_double_precision_perf_ratio;
    pageable_memory_access;
    concurrent_managed_access;
    compute_preemption_supported;
    can_use_host_pointer_for_registered_mem;
    cooperative_launch;
    max_shared_memory_per_block_optin;
    can_flush_remote_writes;
    host_register_supported;
    pageable_memory_access_uses_host_page_tables;
    direct_managed_mem_access_from_host;
    virtual_memory_management_supported;
    handle_type_posix_file_descriptor_supported;
    handle_type_win32_handle_supported;
    handle_type_win32_kmt_handle_supported;
    max_blocks_per_multiprocessor;
    generic_compression_supported;
    max_persisting_l2_cache_size;
    max_access_policy_window_size;
    gpu_direct_rdma_with_cuda_vmm_supported;
    reserved_shared_memory_per_block;
    sparse_cuda_array_supported;
    read_only_host_register_supported;
    timeline_semaphore_interop_supported;
    (* memory_pools_supported; *)
    gpu_direct_rdma_supported;
    gpu_direct_rdma_flush_writes_options;
    gpu_direct_rdma_writes_ordering;
    mempool_supported_handle_types;
    (* cluster_launch; *)
    (* deferred_mapping_cuda_array_supported; *)
    can_use_64_bit_stream_mem_ops;
    can_use_stream_wait_value_nor;
    (* dma_buf_supported; *)
    (* ipc_event_supported; *)
    (* mem_sync_domain_count; *)
    (* tensor_map_access_supported; *)
    (* unified_function_pointers; *)
    (* multicast_supported; *)
  }

type context = cu_context
type func = cu_function
type stream = cu_stream
