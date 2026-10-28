(** Bindings to the NVIDIA `cuda` library. *)


type result
(** See
    {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9}
     enum CUresult}. *)

val sexp_of_result : result -> Sexplib0.Sexp.t

exception Cuda_error of { status : result; message : string }
(** Error codes returned by CUDA functions are converted to exceptions. The message stores a
    snake-case variant of the offending CUDA function name (see {!Cuda_ffi.Bindings.Functions} for
    the direct funciton bindings). *)

val is_success : result -> bool

val cuda_call_hook : (message:string -> status:result -> unit) option ref
(** The functions called before and after every {!Cuda_ffi.Bindings.Functions} call. [message] is
    the snake-case variant of the corresponding CUDA function name. *)

val init : ?flags:int -> unit -> unit
(** Must be called before any other function. Currently [flags] is unused. See
    {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE_1g0a2f1517e1bd8502c7194c3a8c134bc3}
     cuInit}. *)

(** Managing a CUDA GPU device and its primary context. See:
    {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE}
     Device Management} and
    {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX}
     Primary Context Management}. *)
module Device : sig
  type t
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a}
       CUdevice}. *)

  val sexp_of_t : t -> Sexplib0.Sexp.t

  val get_count : unit -> int
  (** Returns the number of Nvidia devices. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74}
       cuDeviceGetCount}. *)

  val get : ordinal:int -> t
  (** Returns the given device. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb}
       cuDeviceGet}. *)

  val primary_ctx_reset : t -> unit
  (** Destroys all allocations and resets all state on the primary context. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g5d38802e8600340283958a117466ce12}
       cuDevicePrimaryCtxReset}. *)

  val get_free_and_total_mem : unit -> int * int
  (** Gets the free memory on the device of the current context according to the OS, and the total
      memory on the device. See:
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0}
       cuMemGetInfo}. *)

  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g578d7cf687ce20f7e99468e8c14e22de}
       CUdevice_P2PAttribute}. *)
  type p2p_attribute =
    | PERFORMANCE_RANK of int
    | ACCESS_SUPPORTED of bool
    | NATIVE_ATOMIC_SUPPORTED of bool
    | CUDA_ARRAY_ACCESS_SUPPORTED of bool

  val sexp_of_p2p_attribute : p2p_attribute -> Sexplib0.Sexp.t

  val get_p2p_attributes : dst:t -> src:t -> p2p_attribute list
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g4c55c60508f8eba4546b51f2ee545393}
       cuDeviceGetP2PAttribute}. *)

  val can_access_peer : dst:t -> src:t -> bool
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e}
       cuDeviceCanAccessPeer}. *)

  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g637aab2eadb52e1c1c048b8bad9592d1}
       CUcomputemode}. *)
  type computemode =
    | DEFAULT  (** Multiple contexts allowed per device. *)
    | PROHIBITED  (** No contexts can be created on this device at this time. *)
    | EXCLUSIVE_PROCESS
        (** Only one context used by a single process can be present on this device at a time. *)
    | UNCATEGORIZED of int64
        (** Raw driver value not matched by any known [CUcomputemode] constant.
            Value 1 is the legacy [CU_COMPUTEMODE_EXCLUSIVE] removed in CUDA 8.0;
            modern drivers should not return it, but some configurations do. *)

  val sexp_of_computemode : computemode -> Sexplib0.Sexp.t

  (** Convert a raw [CU_DEVICE_ATTRIBUTE_COMPUTE_MODE] integer (as returned by
      [cuDeviceGetAttribute]) to [computemode].  Unrecognised values are wrapped
      in [UNCATEGORIZED] rather than raising. *)
  val computemode_of_int : int -> computemode

  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gf34334d1d6892847a5d05be7ca8db3c6}
       CUflushGPUDirectRDMAWritesOptions}. *)
  type flush_GPU_direct_RDMA_writes_options = HOST | MEMOPS

  val sexp_of_flush_GPU_direct_RDMA_writes_options : flush_GPU_direct_RDMA_writes_options -> Sexplib0.Sexp.t

  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g450a23153d86fce0afe30e25d63caef9}
       CUmemAllocationHandleType}. *)
  type mem_allocation_handle_type = NONE | POSIX_FILE_DESCRIPTOR | WIN32 | WIN32_KMT | FABRIC

  val sexp_of_mem_allocation_handle_type : mem_allocation_handle_type -> Sexplib0.Sexp.t

  type attributes = {
    name : string;
        (** See
            {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f}
             cuDeviceGetName}. *)
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
    compute_mode : computemode;
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
        (** Whether the device supports executing multiple kernels within the same context
            simultaneously. *)
    ecc_enabled : bool;  (** Whether error correction is supported and enabled on the device. *)
    pci_bus_id : int;
    pci_device_id : int;  (** PCI device (also known as slot) identifier of the device. *)
    tcc_driver : bool;
    memory_clock_rate : int;  (** In kilohertz. *)
    global_memory_bus_width : int;  (** In bits. *)
    l2_cache_size : int;  (** In bytes. *)
    max_threads_per_multiprocessor : int;
    async_engine_count : int;
    unified_addressing : bool;
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
        (** Device supports coherently accessing pageable memory without calling cudaHostRegister.
        *)
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
    memory_pools_supported : bool;
    gpu_direct_rdma_supported : bool;
        (** See {{:https://docs.nvidia.com/cuda/gpudirect-rdma/} GPUDirect RDMA}. *)
    gpu_direct_rdma_flush_writes_options : flush_GPU_direct_RDMA_writes_options list;
    gpu_direct_rdma_writes_ordering : bool;
    mempool_supported_handle_types : mem_allocation_handle_type list;
        (** Handle types supported with mempool based IPC. *)
    cluster_launch : bool;
    deferred_mapping_cuda_array_supported : bool;
    can_use_64_bit_stream_mem_ops : bool;
    can_use_stream_wait_value_nor : bool;
    dma_buf_supported : bool;
    ipc_event_supported : bool;
    mem_sync_domain_count : int;  (** Number of memory domains the device supports. *)
    tensor_map_access_supported : bool;
    unified_function_pointers : bool;
    multicast_supported : bool;  (** Device supports switch multicast and reduction operations. *)
  }
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266}
       cuDeviceGetAttribute}. *)

  val sexp_of_attributes : attributes -> Sexplib0.Sexp.t

  val get_attributes : t -> attributes
  (** Populates all the device attributes. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266}
       cuDeviceGetAttribute}. *)
end

(** All CUDA tasks are run under a context, usually under the current context. See:
    {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX} Context
     Management}. *)
module Context : sig
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g9f889e28a45a295b5c8ce13aa05f6cd4}
       enum CUctx_flags}. *)
  type flag =
    | SCHED_AUTO  (** Automatic scheduling. *)
    | SCHED_SPIN  (** Instruct CUDA to actively spin when waiting for results from the GPU. *)
    | SCHED_YIELD  (** Instruct CUDA to yield its thread when waiting for results from the GPU. *)
    | SCHED_BLOCKING_SYNC  (** Set blocking synchronization as default scheduling. *)
    | SCHED_MASK
    | MAP_HOST  (** Deprecated: it is always present regardless of passed config. *)
    | LMEM_RESIZE_TO_MAX  (** Keep local memory allocation after launch. *)
    | COREDUMP_ENABLE  (** Trigger coredumps from exceptions in this context. *)
    | USER_COREDUMP_ENABLE  (** Enable user pipe to trigger coredumps in this context. *)
    | SYNC_MEMOPS  (** Ensure synchronous memory operations on this context will synchronize. *)

  val sexp_of_flag : flag -> Sexplib0.Sexp.t

  type flags = flag list

  val sexp_of_flags : flags -> Sexplib0.Sexp.t

  type t
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9}
       CUcontext}. *)

  val sexp_of_t : t -> Sexplib0.Sexp.t

  val create : flags -> Device.t -> t
  (** NOTE: In most cases it is recommended to use {!get_primary} instead! The context is pushed to
      the CPU-thread-local stack. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf}
       cuCtxCreate}

      The context value is finalized using
      {{:https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDA__CTX_g27a365aebb0eb548166309f58a1e8b8e.html}
       ctxDestroy}. Streams, events, modules and device memory created while this context is
      current retain the context value, so it is not finalized (and the CUDA context not
      destroyed) while any of them is still alive. *)

  val get_flags : unit -> flags
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d}
       cuCtxGetFlags}. *)

  val get_primary : Device.t -> t
  (** The context is {i not} pushed to the stack. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300}
       cuDevicePrimaryCtxRetain}.

      The context is finalized using
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad}
       cuDevicePrimaryCtxRelease}. The underlying CUDA context will be reset once the last reference
      to it is released. *)

  val get_device : unit -> Device.t
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e}
       cuCtxGetDevice}. *)

  val pop_current : unit -> t
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902}
       cuCtxPopCurrent}. *)

  val get_current : unit -> t
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0}
       cuCtxGetCurrent}. *)

  val push_current : t -> unit
  (** Pushes a context on the current CPU thread. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba}
       cuCtxPushCurrent}. *)

  val set_current : t -> unit
  (** If there exists a CUDA context stack on the calling CPU thread, this will replace the top of
      that stack with ctx. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7}
       cuCtxSetCurrent}. *)

  val synchronize : unit -> unit
  (** Blocks for the current context's tasks to complete. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616}
       cuCtxSynchronize}. *)

  val disable_peer_access : t -> unit
  (** Disables peer access between the current context and the given context. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g5b4b6936ea868d4954ce4d841a3b4810}
       cuCtxDisablePeerAccess}. *)

  val enable_peer_access : ?flags:Unsigned.uint -> t -> unit
  (** Flags are unused. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a}
       cuCtxEnablePeerAccess}. *)

  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ge24c2d4214af24139020f1aecaf32665}
       enum CUlimit}. *)
  type limit =
    | STACK_SIZE
    | PRINTF_FIFO_SIZE
    | MALLOC_HEAP_SIZE
    | DEV_RUNTIME_SYNC_DEPTH  (** GPU device runtime launch synchronize depth. *)
    | DEV_RUNTIME_PENDING_LAUNCH_COUNT
    | MAX_L2_FETCH_GRANULARITY  (** Between 0 and 128, in bytes, it is a hint. *)
    | PERSISTING_L2_CACHE_SIZE

  val sexp_of_limit : limit -> Sexplib0.Sexp.t

  val set_limit : limit -> int -> unit
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a}
       cuCtxSetLimit}. *)

  val get_limit : limit -> int
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8}
       cuCtxGetLimit}. *)
end

type bigstring = (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(** This module introduces the type of pointers into on-device global memory, and stream-independent
    memory management functions. All functions from this module run synchronously. See:
    {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM} Memory
     Management}. *)
module Deviceptr : sig
  type t
  (** A pointer to a memory location on a device. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e}
       CUdeviceptr}. *)

  type region = { base: t; offset_bytes: int }
  (** A non-owning borrow of a sub-region of a device allocation. The [base] [Deviceptr.t]
      retains ownership and its finalizer will free the allocation; the [region] is never freed
      independently and carries no finalizer. [offset_bytes] is the byte offset into [base];
      offset 0 addresses the same location as [base] itself. Use {!offset} to construct a
      region and {!region_of} to inject a bare [Deviceptr.t] at offset 0. *)

  val offset : t -> bytes:int -> region
  (** [offset ptr ~bytes:n] returns a region starting [n] bytes into [ptr]'s allocation. [ptr]
      retains ownership; the region is a non-owning borrow. [bytes] is in bytes. Offset 0
      reproduces the same address as [ptr]. *)

  val region_of : t -> region
  (** [region_of ptr] is [offset ptr ~bytes:0]: a region at byte offset 0, i.e. the same
      address as [ptr]. Useful when a consuming site requires a [region]. *)

  val sexp_of_t : t -> Sexplib0.Sexp.t

  val equal : t -> t -> bool
  (** Compares the pointer values for equality. *)

  val hash : t -> int
  (** Converts the pointer to an OCaml int using [Unsigned.UInt64.to_int] (truncating bits as
      needed). *)

  val string_of : t -> string
  (** Hexadecimal representation of the pointer. *)

  val mem_alloc : size_in_bytes:int -> t
  (** The memory is aligned, is not cleared. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467}
       cuMemAlloc}.

      The pointer is finalized using
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a}
       cuMemFree}. This is safe
      {{:https://stackoverflow.com/questions/70767180/cumemallocing-memory-in-one-cuda-context-and-freeing-it-in-another-why-does}
       without needing to set the proper context}. *)

  val mem_free : t -> unit
  (** Double-freeing is prevented by a flag: multiple calls on the same [Deviceptr.t] are safe. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a}
       cuMemFree}. *)

  val memcpy_H_to_D_unsafe : dst:t -> src:unit Ctypes.ptr -> size_in_bytes:int -> unit
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169}
       cuMemcpyHtoD}. *)

  val memcpy_H_to_D :
    ?host_offset:int ->
    ?length:int ->
    ?dst_offset:int ->
    dst:t ->
    src:('a, 'b, 'c) Bigarray.Genarray.t ->
    unit ->
    unit
  (** Copies the bigarray (or its interval) into the device memory. [host_offset] and [length] are
      in numbers of elements. [dst_offset] is a device-side byte offset into [dst] (default 0),
      letting the copy target a sub-region of a larger allocation; when [length] is not given the
      copied size is reduced by [dst_offset] so it does not write past the end of [dst]. See
      {!memcpy_H_to_D_unsafe}. *)

  val alloc_and_memcpy : ('a, 'b, 'c) Bigarray.Genarray.t -> t
  (** Combines {!mem_alloc} and {!memcpy_H_to_D}. *)

  val memcpy_D_to_H_unsafe : dst:unit Ctypes.ptr -> src:t -> size_in_bytes:int -> unit
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893}
       cuMemcpyDtoH}. *)

  val memcpy_D_to_H :
    ?host_offset:int ->
    ?length:int ->
    ?src_offset:int ->
    dst:('a, 'b, 'c) Bigarray.Genarray.t ->
    src:t ->
    unit ->
    unit
  (** Copies from the device memory into the bigarray (or its interval). [host_offset] and [length]
      are in numbers of elements: [length] is the count of elements copied (not an end index),
      filling [dst.(host_offset .. host_offset + length)] and mirroring how {!memcpy_H_to_D} reads
      [src.(host_offset .. host_offset + length)]. [src_offset] is a device-side byte offset into
      [src] (default 0), letting the copy read from a sub-region of a larger allocation; when
      [length] is not given the copied size is reduced by [src_offset] so it does not read past the
      end of [src]. See {!memcpy_D_to_H_unsafe}. *)

  val memcpy_D_to_D :
    ?kind:('a, 'b) Bigarray.kind ->
    ?length:int ->
    ?size_in_bytes:int ->
    ?dst_offset:int ->
    ?src_offset:int ->
    dst:t ->
    src:t ->
    unit ->
    unit
  (** Copies between two memory positions on the same device. The size to copy can optionally be
      provided in numbers of elements via [kind] and [length]. Provide either both [kind] and
      [length], or just [size_in_bytes]. [dst_offset] and [src_offset] are device-side byte offsets
      into [dst] and [src] respectively (default 0), letting either end target a sub-region of a
      larger allocation. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b}
       cuMemcpyDtoD}. *)

  val memcpy_peer :
    ?kind:('a, 'b) Bigarray.kind ->
    ?length:int ->
    ?size_in_bytes:int ->
    ?dst_offset:int ->
    ?src_offset:int ->
    dst:t ->
    dst_ctx:Context.t ->
    src:t ->
    src_ctx:Context.t ->
    unit ->
    unit
  (** Copies between memory positions on two different devices. The size to copy can optionally be
      provided in numbers of elements via [kind] and [length]. Provide either both [kind] and
      [length], or just [size_in_bytes]. [dst_offset] and [src_offset] are device-side byte offsets
      into [dst] and [src] respectively (default 0), letting either end target a sub-region of a
      larger allocation. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1ge1f5c7771544fee150ada8853c7cbf4a}
       cuMemcpyPeer}. *)

  val memset_d8 : ?offset:int -> t -> Unsigned.uchar -> length:int -> unit
  (** [offset] is a device-side byte offset into [t] (default 0), letting the memset target a
      sub-region of a larger allocation. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b}
       cuMemsetD8}. *)

  val memset_d16 : ?offset:int -> t -> Unsigned.ushort -> length:int -> unit
  (** [length] is in number of elements. [offset] is a device-side byte offset into [t] (default
      0). See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c}
       cuMemsetD16}. *)

  val memset_d32 : ?offset:int -> t -> Unsigned.uint32 -> length:int -> unit
  (** [length] is in number of elements. [offset] is a device-side byte offset into [t] (default
      0). See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132}
       cuMemsetD32}. *)
end

(** A CUDA module type represents CUDA code that's ready to execute, i.e. is loaded. See:
    {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE}
     Module Management}. *)
module Module : sig
  (** Compute device classes. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ge443308cb7ed1d52b85b487305779184}
       enum CUjit_target}. *)
  type jit_target =
    | COMPUTE_30
    | COMPUTE_32
    | COMPUTE_35
    | COMPUTE_37
    | COMPUTE_50
    | COMPUTE_52
    | COMPUTE_53
    | COMPUTE_60
    | COMPUTE_61
    | COMPUTE_62
    | COMPUTE_70
    | COMPUTE_72
    | COMPUTE_75
    | COMPUTE_80
    | COMPUTE_86
    | COMPUTE_87
    | COMPUTE_89
    | COMPUTE_90
    | COMPUTE_90A  (** Compute device class 9.0 with accelerated features. *)

  val sexp_of_jit_target : jit_target -> Sexplib0.Sexp.t

  (** Cubin matching fallback strategies. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g4a1a92ea65e18b06907b981848c282f2}
       CUjit_fallback}. *)
  type jit_fallback = PREFER_PTX | PREFER_BINARY

  val sexp_of_jit_fallback : jit_fallback -> Sexplib0.Sexp.t

  (** Caching modes for dlcm. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gce011cfe2d6b1fb734da48a6cf48fd04}
       CUjit_cacheMode}. *)
  type jit_cache_mode =
    | NONE
    | CG  (** Compile with L1 cache disabled. *)
    | CA  (** Compile with L1 cache enabled. *)

  val sexp_of_jit_cache_mode : jit_cache_mode -> Sexplib0.Sexp.t

  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d}
       CUjit_option}. *)
  type jit_option =
    | MAX_REGISTERS of int  (** Max number of registers that a thread may use. *)
    | THREADS_PER_BLOCK of int
        (** Specifies minimum number of threads per block to target compilation for or returns the
            number of threads the compiler actually targeted. Cannot be combined with [TARGET]. *)
    | WALL_TIME of { milliseconds : float }  (**  *)
    | INFO_LOG_BUFFER of bigstring
    | ERROR_LOG_BUFFER of bigstring
    | OPTIMIZATION_LEVEL of int
        (** 0 to 4, with 4 being the default and highest level of optimizations. *)
    | TARGET_FROM_CUCONTEXT
    | TARGET of jit_target
    | FALLBACK_STRATEGY of jit_fallback
    | GENERATE_DEBUG_INFO of bool  (** Helpful for cuda-gdb. *)
    | LOG_VERBOSE of bool
    | GENERATE_LINE_INFO of bool  (** Helpful for cuda-gdb. *)
    | CACHE_MODE of jit_cache_mode
    | POSITION_INDEPENDENT_CODE of bool

  val sexp_of_jit_option : jit_option -> Sexplib0.Sexp.t

  type func
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8}
       CUfunction}. *)

  type t
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef}
       CUmodule}. *)

  val load_data_ex : Nvrtc.compile_to_ptx_result -> jit_option list -> t
  (** Currently, the image passed via this call is the PTX source. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12}
       cuModuleLoadDataEx}.

      The module is finalized using
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b}
       cuModuleUnload}. The finalizer captures the context when [load_data_ex] is called to
      temporarily push it on the stack for unloading. *)

  val get_function : t -> name:string -> func
  (** The returned function retains the module, so the module is not finalized (unloaded) while
      the function is still in use. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf}
       cuModuleGetFunction}. *)

  val get_global : t -> name:string -> Deviceptr.t * Unsigned.size_t
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab}
       cuModuleGetGlobal}. *)

  val max_active_blocks_per_multiprocessor :
    ?dynamic_smem_bytes:int -> func -> block_size:int -> int
  (** The maximum number of blocks of [block_size] threads that can be simultaneously resident on
      one multiprocessor, given the kernel's register and shared memory usage.
      [dynamic_smem_bytes] (default 0) is the per-block dynamic shared memory the launch would
      request, i.e. the [shared_mem_bytes] of {!Stream.launch_kernel}. The result is 0 for a
      configuration that cannot be launched at all -- one asking for more shared memory or more
      threads per block than the device provides -- rather than an error.

      The theoretical occupancy of such a launch -- the fraction of a multiprocessor's warp slots
      that are filled -- is [float (result * warps_per_block) /. float max_warps], where
      [warps_per_block = (block_size + device_props.warp_size - 1) / device_props.warp_size] and
      [max_warps = device_props.max_threads_per_multiprocessor / device_props.warp_size]. The
      hardware allocates whole warps, so a block of 33 threads takes two warp slots rather than 33
      thread slots; the two formulations coincide when [block_size] is a multiple of the warp size.
      [result * device_props.multiprocessor_count] blocks are enough to fill the whole device. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gcc6e1094d05cba2cee17fe33ddd04a98}
       cuOccupancyMaxActiveBlocksPerMultiprocessor}. *)
end

(** CUDA streams are independent FIFO schedules for CUDA tasks, allowing them to potentially run in
    parallel. See:
    {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM}
     Stream Management}. *)
module Stream : sig
  type t
  (** Stores a stream pointer and manages lifetimes of kernel launch arguments. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a}
       CUstream}. *)

  val sexp_of_t : t -> Sexplib0.Sexp.t

  val mem_alloc : t -> size_in_bytes:int -> Deviceptr.t
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f}
       cuMemAllocAsync}.

      The pointer is finalized using
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf}
       cuMemFreeAsync}. *)

  val mem_free : t -> Deviceptr.t -> unit
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf}
       cuMemFreeAsync}. *)

  val memcpy_H_to_D_unsafe :
    dst:Deviceptr.t -> src:unit Ctypes.ptr -> size_in_bytes:int -> t -> unit
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3}
       cuMemcpyHtoDAsync}. *)

  val memcpy_H_to_D :
    ?host_offset:int ->
    ?length:int ->
    ?dst_offset:int ->
    dst:Deviceptr.t ->
    src:('a, 'b, 'c) Bigarray.Genarray.t ->
    t ->
    unit
  (** Copies the bigarray (or its interval) into the device memory asynchronously. [host_offset] and
      [length] are in numbers of elements. [dst_offset] is a device-side byte offset into [dst]
      (default 0), letting the copy target a sub-region of a larger allocation; when [length] is not
      given the copied size is reduced by [dst_offset] so it does not write past the end of [dst].
      See {!memcpy_H_to_D_unsafe}. *)

  (** Parameters to pass to a kernel. *)
  type kernel_param =
    | Tensor of Deviceptr.t
    | Tensor_at of Deviceptr.region
    (** [Tensor_at { base; offset_bytes }] passes [base + offset_bytes] as the [CUdeviceptr]
        kernel argument (tinygrad's [buf.value + off] pattern). The kernel receives an address
        displaced by [offset_bytes] bytes and dereferences from index 0 of that address; the
        kernel signature is unchanged. [check_freed] and lifetime bookkeeping operate on [base]. *)
    | Int of int  (** Passed as C [int]. *)
    | Size_t of Unsigned.size_t
    | Single of float  (** Passed as C [float]. *)
    | Double of float  (** Passed as C [double]. *)

  val sexp_of_kernel_param : kernel_param -> Sexplib0.Sexp.t

  val no_stream : t
  (** The NULL stream which is the main synchronization stream of a device. Manages lifetimes of the
      corresponding kernel launch parameters. *)

  val launch_kernel :
    Module.func ->
    grid_dim_x:int ->
    ?grid_dim_y:int ->
    ?grid_dim_z:int ->
    block_dim_x:int ->
    ?block_dim_y:int ->
    ?block_dim_z:int ->
    shared_mem_bytes:int ->
    t ->
    kernel_param list ->
    unit
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15}
       cuLaunchKernel}. *)

  val memcpy_D_to_H_unsafe :
    dst:unit Ctypes.ptr -> src:Deviceptr.t -> size_in_bytes:int -> t -> unit
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362}
       cuMemcpyDtoHAsync}. *)

  val memcpy_D_to_H :
    ?host_offset:int ->
    ?length:int ->
    ?src_offset:int ->
    dst:('a, 'b, 'c) Bigarray.Genarray.t ->
    src:Deviceptr.t ->
    t ->
    unit
  (** Copies from the device memory into the bigarray (or its interval) asynchronously.
      [host_offset] and [length] are in numbers of elements: [length] is the count of elements
      copied (not an end index), filling [dst.(host_offset .. host_offset + length)] and mirroring
      how {!memcpy_H_to_D} reads [src.(host_offset .. host_offset + length)]. [src_offset] is a
      device-side byte offset into [src] (default 0), letting the copy read from a sub-region of a
      larger allocation; when [length] is not given the copied size is reduced by [src_offset] so it
      does not read past the end of [src]. See {!memcpy_D_to_H_unsafe} and
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362}
       cuMemcpyDtoHAsync}. *)

  val memcpy_D_to_D :
    ?kind:('a, 'b) Bigarray.kind ->
    ?length:int ->
    ?size_in_bytes:int ->
    ?dst_offset:int ->
    ?src_offset:int ->
    dst:Deviceptr.t ->
    src:Deviceptr.t ->
    t ->
    unit
  (** Copies between two memory positions on the same device asynchronously. The size to copy can
      optionally be provided in numbers of elements via [kind] and [length]. Provide either both
      [kind] and [length], or just [size_in_bytes]. [dst_offset] and [src_offset] are device-side
      byte offsets into [dst] and [src] respectively (default 0), letting either end target a
      sub-region of a larger allocation. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8}
       cuMemcpyDtoDAsync}. *)

  val memcpy_peer :
    ?kind:('a, 'b) Bigarray.kind ->
    ?length:int ->
    ?size_in_bytes:int ->
    ?dst_offset:int ->
    ?src_offset:int ->
    dst:Deviceptr.t ->
    dst_ctx:Context.t ->
    src:Deviceptr.t ->
    src_ctx:Context.t ->
    t ->
    unit
  (** Copies between memory positions on two different devices asynchronously. The size to copy can
      optionally be provided in numbers of elements via [kind] and [length]. Provide either both
      [kind] and [length], or just [size_in_bytes]. [dst_offset] and [src_offset] are device-side
      byte offsets into [dst] and [src] respectively (default 0), letting either end target a
      sub-region of a larger allocation. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g82fcecb38018e64b98616a8ac30112f2}
       cuMemcpyPeerAsync}. *)

  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g17c5d5f9b585aa2d6f121847d1a78f4c}
       CUmemAttach_flags}. *)
  type attach_mem =
    | GLOBAL  (** Memory can be accessed by any stream on any device. *)
    | HOST  (** Memory cannot be accessed from devices. *)
    | SINGLE_stream  (** Memory can only be accessed by a single stream. *)

  val sexp_of_attach_mem : attach_mem -> Sexplib0.Sexp.t

  val attach_mem : t -> Deviceptr.t -> int -> attach_mem -> unit
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533}
       cuStreamAttachMemAsync}. *)

  val create : ?non_blocking:bool -> ?lower_priority:int -> unit -> t
  (** Lower [lower_priority] numbers represent higher priorities, the default is [0]. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471}
       cuStreamCreateWithPriority}.

      The stream value is finalized using
      {{:https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDA__STREAM_g244c8833de4596bcd31a06cdf21ee758.html}
       cuStreamDestroy}. This is meant to be safe
      {{:https://stackoverflow.com/questions/64663943/how-to-destroy-a-stream-that-was-created-on-a-specific-device}
       without needing to set the proper context}. *)

  val get_context : t -> Context.t
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1107907025eaa3387fdc590a9379a681}
       cuStreamGetCtx}. *)

  val get_id : t -> Unsigned.uint64
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g5dafd2b6f48caeb13d5110a7f21e60e3}
       cuStreamGetId}. *)

  val is_ready : t -> bool
  (** Returns [false] when the querying status is [CUDA_ERROR_NOT_READY], and [true] if it is
      [CUDA_SUCCESS]. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b}
       cuStreamQuery}. *)

  val synchronize : t -> unit
  (** Waits until a stream's tasks are completed. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad}
       cuStreamSynchronize}. *)

  val memset_d8 : ?offset:int -> Deviceptr.t -> Unsigned.uchar -> length:int -> t -> unit
  (** [offset] is a device-side byte offset into [Deviceptr.t] (default 0), letting the memset
      target a sub-region of a larger allocation. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627}
       cuMemsetD8Async}. *)

  val memset_d16 : ?offset:int -> Deviceptr.t -> Unsigned.ushort -> length:int -> t -> unit
  (** [length] is in number of elements. [offset] is a device-side byte offset into [Deviceptr.t]
      (default 0). See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c}
       cuMemsetD16Async}. *)

  val memset_d32 : ?offset:int -> Deviceptr.t -> Unsigned.uint32 -> length:int -> t -> unit
  (** [length] is in number of elements. [offset] is a device-side byte offset into [Deviceptr.t]
      (default 0). See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5}
       cuMemsetD32Async}. *)

  val total_unreleased_unfinished_delimited_events : t -> int * int * int
  (** Debug information about delimited events carried by the stream: total, unreleased (i.e. not
      destroyed), unfinished. *)

  val get_total_live_streams : unit -> int
  (** The total non-garbage-collected streams across all devices. *)
end

(** CUDA events can be used for synchronization between streams without blocking the CPU, and to
    time the on-device execution. See:
    {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT}
     Event Management}. *)
module Event : sig
  type t
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b}
       CUevent}. *)

  val sexp_of_t : t -> Sexplib0.Sexp.t

  val create : ?blocking_sync:bool -> ?enable_timing:bool -> ?interprocess:bool -> unit -> t
  (** Creates an event {i for the current context}. All of [blocking_sync], [enable_timing] and
      [interprocess] are by default false. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db}
       cuEventCreate} and
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g5ae04079c671c8e659a3a27c7b23f629}
       CUevent_flags}.

      The event value is finalized using
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef}
       cuEventDestroy}. This is safe because the event resources are only released when the event
      completes, so waiting streams are not affected by the finalization. Note: I assume destroying
      an event is safe without setting the proper context. *)

  val elapsed_time : start:t -> end_:t -> float
  (** Returns (an upper bound on) elapsed time in milliseconds with a resolution of around 0.5
      microseconds. Both events must have completed ([query start = true] and [query end_ = true])
      before calling [elapsed_time]. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97}
       cuEventElapsedTime}. *)

  val query : t -> bool
  (** Returns [true] precisely when all work captured by the most recent call to {!record} has been
      completed. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef}
       cuEventQuery}. *)

  val record : ?external_:bool -> t -> Stream.t -> unit
  (** Captures in the event the contents of the stream, i.e. the work scheduled on it. [external_]
      defaults to false (cudajit as of version 0.5 does not expose stream capture). See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b}
       cuEventRecordWithFlags}. *)

  val synchronize : t -> unit
  (** Blocks until the completion of all work captured in the event by the most recent call to
      {!record}. NOTE: if the event was created without [~blocking_sync:true], then the CPU thread
      will busy-wait. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c}
       cuEventSynchronize}. *)

  val wait : ?external_:bool -> Stream.t -> t -> unit
  (** Future work submitted to the stream will wait for the completion of all work captured in the
      event by the most recent call to {!record}. [external_] defaults to false (cudajit as of
      version 0.5 does not expose stream capture). See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f}
       cuStreamWaitEvent}. *)
end

(** This module builds on top of functionality more directly exposed by {!Event}. It optimizes
    resource management for use-cases where events are not reused: there's only one call to
    {!Event.record}, and it's immediately after {!Event.create}. *)
module Delimited_event : sig
  type t
  (** An delimited event encapsulates {!Event.t} and is owned by a stream. It records its owner at
      creation, and gets released (using
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef}
       cuEventDestroy}) when either it or its owner are synchronized (or if neither happens, when it
      is garbage-collected). *)

  val sexp_of_t : t -> Sexplib0.Sexp.t

  val record : ?blocking_sync:bool -> ?interprocess:bool -> ?external_:bool -> Stream.t -> t
  (** Combines {!Event.create} and {!Event.record} to create an event owned by the given stream. *)

  val is_released : t -> bool
  (** Returns true if the delimited event is already released using
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef}
       cuEventDestroy}. The event will be released by {!synchronize} and {!Stream.synchronize}. *)

  val query : t -> bool
  (** See {!Event.query}. [query event] returns [true] when [event] is already released. *)

  val synchronize : t -> unit
  (** See {!Event.synchronize}. [synchronize event] is a no-op if [event] is already released. *)

  val wait : ?external_:bool -> Stream.t -> t -> unit
  (** See {!Event.wait}. [wait stream event] is a no-op if [event] is already released. *)
end

(** CUDA graphs: capture a sequence of stream operations once, then replay it with a single launch
    call. See:
    {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH}
     Graph Management}. *)
module Graph : sig
  type t
  (** A graph under construction (a template of work, not yet executable). The graph value retains
      the OCaml-side resources of the operations captured into it (kernels — and through them
      their modules — kernel parameters, device pointers), so a {!Stream.synchronize} of the
      capture stream cannot free resources a replay still needs. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html} CUgraph}. *)

  type exec
  (** An instantiated, executable graph; retains the captured operations' resources like {!t}. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html} CUgraphExec}. *)

  val sexp_of_t : t -> Sexplib0.Sexp.t
  val sexp_of_exec : exec -> Sexplib0.Sexp.t

  type capture_mode = GLOBAL | THREAD_LOCAL | RELAXED
  (** See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g4687a37c1c157c94d3a2ee16fde49c9c}
       CUstreamCaptureMode}. [GLOBAL] (the CUDA default) invalidates the capture on potentially
      unsafe API calls from any thread; [THREAD_LOCAL] restricts that to the capturing thread;
      [RELAXED] imposes no cross-thread restrictions. *)

  val sexp_of_capture_mode : capture_mode -> Sexplib0.Sexp.t

  val begin_capture : ?mode:capture_mode -> Stream.t -> unit
  (** Starts capturing work submitted to the stream instead of executing it. [mode] defaults to
      [GLOBAL]. Between [begin_capture] and {!end_capture}, operations enqueued on the stream (e.g.
      {!Stream.launch_kernel}) are recorded as graph nodes, not run. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1gac495e0527d1dd6437f95ee482f61865}
       cuStreamBeginCapture}. *)

  val end_capture : Stream.t -> t
  (** Ends the capture begun by {!begin_capture} and returns the captured graph. Raises
      [Cuda_error] if the capture was invalidated. The graph value is finalized using
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g718cfd9681f078693d4be2426fd689c8}
       cuGraphDestroy}; it can also be released eagerly with {!destroy}. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c}
       cuStreamEndCapture}. *)

  val instantiate : t -> exec
  (** Creates an executable graph from a captured graph; the template can be destroyed afterwards.
      The exec value is finalized using
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga32ad4944cc5d408158207c978bc43a7}
       cuGraphExecDestroy}; it can also be released eagerly with {!exec_destroy}, but only once no
      launch of it is pending (synchronize the stream first). See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1}
       cuGraphInstantiate}. *)

  val launch : exec -> Stream.t -> unit
  (** Enqueues the whole captured work sequence on the stream, as one API call. The stream retains
      the exec until it is synchronized (mirroring {!Stream.launch_kernel}'s argument retention),
      so dropping the last reference to a launched exec cannot destroy it while the launch is
      pending. See
      {{:https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471}
       cuGraphLaunch}. *)

  val destroy : t -> unit
  (** Eagerly destroys the graph template. Idempotent; the finalizer covers the non-eager case. *)

  val exec_destroy : exec -> unit
  (** Eagerly destroys the executable graph. Idempotent; the finalizer covers the non-eager case.
  *)
end
