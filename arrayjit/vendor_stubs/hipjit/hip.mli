(** Bindings to the AMD HIP runtime (the `amdhip64` library), mirroring the API of the [Cuda]
    module from the `cudajit` package.

    HIP's module / kernel-launch API deliberately mirrors the CUDA driver API, so most functions
    correspond 1:1. See:
    {{:https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api_reference.html}
     the HIP runtime API reference}.

    Differences from `cudajit` worth knowing:
    - Device pointers are [void*] on the AMD platform; {!Deviceptr.t} still supports byte-offset
      arithmetic.
    - Peer-to-peer copies take device ordinals ({!Device.t}) instead of contexts.
    - Contexts are deprecated in HIP in favor of the current device ({!Device.set_current}); the
      {!Context} module is provided for CUDA-driver-API-style code and works with primary contexts.
    - Events do not support the CUDA "external" record/wait flags. *)

type result
(** See
    {{:https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/global_defines_enums_structs_files/global_enum_and_defines.html}
     enum hipError_t}. *)

val sexp_of_result : result -> Sexplib0.Sexp.t

exception Hip_error of { status : result; message : string }
(** Error codes returned by HIP functions are converted to exceptions. The message stores a
    snake-case variant of the offending HIP function name (see {!Hip_ffi.Bindings.Functions} for
    the direct function bindings). *)

exception Use_after_free of { func : string; arg : string }
(** Raised when a {!Deviceptr.t} is used after {!Deviceptr.mem_free} / {!Stream.mem_free}. *)

val is_success : result -> bool

val error_name : result -> string
(** The name of the error, as reported by [hipGetErrorName]. *)

val error_string : result -> string
(** The description of the error, as reported by [hipGetErrorString]. *)

val hip_call_hook : (message:string -> status:result -> unit) option ref
(** When set, the callback is invoked after every HIP call with the function name (converted to
    snake case) and its status, before errors are raised as exceptions. Useful for logging. *)

val init : ?flags:int -> unit -> unit
(** Must be called before any other function. [flags] must be [0]. See
    {{:https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/initialization_and_version.html}
     hipInit}. *)

val driver_get_version : unit -> int
(** The HIP driver version (on AMD, of the HIP runtime). *)

val runtime_get_version : unit -> int
(** The HIP runtime version. *)

(** Managing a GPU device and its primary context. *)
module Device : sig
  type t

  val sexp_of_t : t -> Sexplib0.Sexp.t

  val get_count : unit -> int
  (** The number of visible HIP devices. *)

  val get : ordinal:int -> t
  (** The device at the given position. *)

  val get_ordinal : t -> int
  (** The inverse of {!get}: on HIP, devices are identified by their ordinal. *)

  val primary_ctx_release : t -> unit
  (** Releases one reference to the device's primary context; when the last reference is released,
      the primary context is destroyed. Note: {!Context.get_primary} already attaches this to the
      OCaml GC. *)

  val primary_ctx_reset : t -> unit
  (** Destroys all allocations and resets all state on the primary context. *)

  val get_free_and_total_mem : unit -> int * int
  (** Gets the free memory of the current device, and its total memory; in bytes. *)

  val total_mem : t -> int
  (** The device's total memory in bytes. *)

  val set_current : t -> unit
  (** Sets the current device for the calling host thread ([hipSetDevice]). This is the idiomatic
      HIP alternative to context management. *)

  val get_current : unit -> t
  (** The current device for the calling host thread. *)

  val can_access_peer : dst:t -> src:t -> bool
  (** Whether [dst] can directly access the memory of [src]. *)

  val enable_peer_access : ?flags:int -> t -> unit
  (** Enables the {e current} device to access the memory of the given peer device. [flags] must
      be [0]. *)

  val disable_peer_access : t -> unit
  (** Disables the current device's access to the memory of the given peer device. *)

  type attributes = {
    name : string;  (** The device's marketing name, e.g. "AMD Radeon(TM) 8060S Graphics". *)
    gcn_arch_name : string;
        (** The GPU architecture name, e.g. "gfx1151"; pass to hiprtc as
            [--offload-arch=<gcn_arch_name>]. *)
    total_global_mem : int;
    shared_mem_per_block : int;
    regs_per_block : int;
    warp_size : int;  (** The wavefront size: 32 on RDNA GPUs, 64 on CDNA/GCN GPUs. *)
    max_threads_per_block : int;
    max_threads_dim : int * int * int;
    max_grid_size : int * int * int;
    clock_rate : int;  (** In kilohertz. *)
    total_const_mem : int;
    compute_capability_major : int;
    compute_capability_minor : int;
    multiprocessor_count : int;  (** The number of compute units. *)
    integrated : bool;  (** APU vs. discrete GPU. *)
    can_map_host_memory : bool;
    compute_mode : int;
    concurrent_kernels : bool;
    pci_bus_id : int;
    pci_device_id : int;
    pci_domain_id : int;
    async_engine_count : int;
    unified_addressing : bool;
    memory_clock_rate : int;  (** In kilohertz. *)
    memory_bus_width : int;  (** In bits. *)
    l2_cache_size : int;
    max_threads_per_multiprocessor : int;
    shared_mem_per_multiprocessor : int;
    managed_memory : bool;
    is_multi_gpu_board : bool;
    host_native_atomic_supported : bool;
    pageable_memory_access : bool;
    concurrent_managed_access : bool;
    cooperative_launch : bool;
    cooperative_multi_device_launch : bool;
    memory_pools_supported : bool;
        (** Whether the device supports [hipMallocAsync] ({!Stream.mem_alloc}). *)
    max_shared_memory_per_multiprocessor : int;
    clock_instruction_rate : int;  (** HIP only: timer frequency of device-side "clock*". *)
    is_large_bar : bool;  (** HIP only. *)
    asic_revision : int;  (** HIP only. *)
  }

  val sexp_of_attributes : attributes -> Sexplib0.Sexp.t

  val get_attributes : t -> attributes
  (** The device's properties, from [hipGetDeviceProperties]. *)
end

(** All of HIP's context management functionality. Note that contexts are deprecated on the AMD
    platform in favor of the current device: this module is provided for compatibility with
    CUDA-driver-API-style code (cudajit); it works with primary contexts. *)
module Context : sig
  type t

  val sexp_of_t : t -> Sexplib0.Sexp.t

  type flag =
    | SCHED_AUTO
    | SCHED_SPIN
    | SCHED_YIELD
    | SCHED_BLOCKING_SYNC
    | SCHED_MASK
    | MAP_HOST  (** Always allowed on ROCm. *)
    | LMEM_RESIZE_TO_MAX  (** Silently ignored on ROCm. *)

  val sexp_of_flag : flag -> Sexplib0.Sexp.t

  type flags = flag list

  val sexp_of_flags : flags -> Sexplib0.Sexp.t

  val create : flags -> Device.t -> t
  (** NOTE: on HIP this is deprecated; prefer {!get_primary}. The context is destroyed when it is
      garbage-collected. *)

  val get_primary : Device.t -> t
  (** The primary context of the given device, retained. The primary context reference is released
      when the result is garbage-collected. Note: on HIP, unlike CUDA, the primary context is
      automatically initialized by runtime-style calls, so this is mostly useful for
      {!set_current} / {!push_current} idioms. *)

  val get_device : unit -> Device.t
  (** The device of the current context. *)

  val pop_current : unit -> t
  (** Detaches the top context from the current thread's stack and returns it. *)

  val get_current : unit -> t
  val push_current : t -> unit
  val set_current : t -> unit

  val synchronize : unit -> unit
  (** Blocks until all the current device's tasks complete ([hipDeviceSynchronize]; HIP deprecates
      [hipCtxSynchronize], for primary-context usage the semantics coincide). *)

  type limit = STACK_SIZE | PRINTF_FIFO_SIZE | MALLOC_HEAP_SIZE

  val sexp_of_limit : limit -> Sexplib0.Sexp.t

  val set_limit : limit -> int -> unit
  (** Note: on HIP, limits are scoped to the current device rather than a context. *)

  val get_limit : limit -> int
end

(** A device pointer with associated use-after-free protection and byte-offset arithmetic. *)
module Deviceptr : sig
  type t

  val sexp_of_t : t -> Sexplib0.Sexp.t

  type region = { base : t; offset_bytes : int }
  (** A sub-region of a device allocation, for use as a kernel argument (see
      {!Stream.kernel_param.Tensor_at}). *)

  val offset : t -> bytes:int -> region
  val region_of : t -> region
  val equal : t -> t -> bool
  val hash : t -> int
  val string_of : t -> string

  val mem_alloc : size_in_bytes:int -> t
  (** The memory is freed when the result is garbage-collected, if it isn't already freed. *)

  val mem_free : t -> unit
  (** Idempotent: does nothing if the memory was already freed. *)

  val alloc_and_memcpy : ('a, 'b, 'c) Bigarray.Genarray.t -> t
  (** Combines {!mem_alloc} and {!memcpy_H_to_D}. *)

  val memcpy_H_to_D :
    ?host_offset:int ->
    ?length:int ->
    ?dst_offset:int ->
    dst:t ->
    src:('a, 'b, 'c) Bigarray.Genarray.t ->
    unit ->
    unit
  (** Copies the bigarray (or its interval) into the device memory. [host_offset] and [length]
      are in numbers of elements; [dst_offset] is in bytes. NOTE: [host_offset] requires
      [length]. *)

  val memcpy_H_to_D_unsafe : dst:t -> src:unit Ctypes.ptr -> size_in_bytes:int -> unit
  (** Raw pointer variant of {!memcpy_H_to_D}. *)

  val memcpy_D_to_H :
    ?host_offset:int ->
    ?length:int ->
    ?src_offset:int ->
    dst:('a, 'b, 'c) Bigarray.Genarray.t ->
    src:t ->
    unit ->
    unit
  (** Copies from the device memory into the bigarray (or its interval). [host_offset] and
      [length] are in numbers of elements; [src_offset] is in bytes. *)

  val memcpy_D_to_H_unsafe : dst:unit Ctypes.ptr -> src:t -> size_in_bytes:int -> unit
  (** Raw pointer variant of {!memcpy_D_to_H}. *)

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
  (** Copies between two device allocations (or their intervals) on the same device. Provide
      either both [kind] and [length], or just [size_in_bytes]; the offsets are in bytes. *)

  val memcpy_peer :
    ?kind:('a, 'b) Bigarray.kind ->
    ?length:int ->
    ?size_in_bytes:int ->
    ?dst_offset:int ->
    ?src_offset:int ->
    dst:t ->
    dst_device:Device.t ->
    src:t ->
    src_device:Device.t ->
    unit ->
    unit
  (** Copies between memory on two different devices. Provide either both [kind] and [length], or
      just [size_in_bytes]; the offsets are in bytes. Note: unlike CUDA, HIP identifies peers by
      device rather than by context. *)

  val memset_d8 : ?offset:int -> t -> Unsigned.uchar -> length:int -> unit
  (** Sets the [length] bytes at (offset bytes from) the device pointer to the given value. *)

  val memset_d16 : ?offset:int -> t -> Unsigned.ushort -> length:int -> unit
  (** Sets [length] 16-bit values; [offset] is in bytes and must be 2-byte aligned. *)

  val memset_d32 : ?offset:int -> t -> int -> length:int -> unit
  (** Sets [length] 32-bit values; [offset] is in bytes and must be 4-byte aligned. *)
end

(** Loading of compiled code objects, and kernel retrieval. *)
module Module : sig
  type func
  (** A kernel extracted from a loaded module. The value retains its module, so the module is not
      unloaded while the kernel is reachable (or queued on a stream). *)

  val sexp_of_func : func -> Sexplib0.Sexp.t

  type t

  val sexp_of_t : t -> Sexplib0.Sexp.t

  (** NOTE: on the AMD platform, [hipModuleLoadDataEx] ignores most JIT options; they are accepted
      for CUDA-driver-API compatibility. *)
  type jit_option =
    | MAX_REGISTERS of int
    | THREADS_PER_BLOCK of int
    | OPTIMIZATION_LEVEL of int
    | GENERATE_DEBUG_INFO of bool
    | LOG_VERBOSE of bool
    | GENERATE_LINE_INFO of bool
    | FAST_COMPILE of bool
    | POSITION_INDEPENDENT_CODE of bool
    | MIN_CTA_PER_SM of int
    | MAX_THREADS_PER_BLOCK of int

  val sexp_of_jit_option : jit_option -> Sexplib0.Sexp.t

  val load_data_ex : Hiprtc.compile_to_code_result -> jit_option list -> t
  (** Loads the compiled code object into the current device. The module is unloaded when the
      result is garbage-collected. *)

  val load_data : Hiprtc.compile_to_code_result -> t
  (** Same as {!load_data_ex} with no options. *)

  val get_function : t -> name:string -> func
  (** Retrieves the kernel with the given name (as compiled, i.e. the [extern "C"] name or the
      mangled name) from the module. *)

  type function_attribute = Hip_ffi.Bindings_types.hip_function_attribute =
    | HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    | HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
    | HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
    | HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
    | HIP_FUNC_ATTRIBUTE_NUM_REGS
    | HIP_FUNC_ATTRIBUTE_PTX_VERSION
    | HIP_FUNC_ATTRIBUTE_BINARY_VERSION
    | HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA
    | HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    | HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    | HIP_FUNC_ATTRIBUTE_UNCATEGORIZED of int64

  val sexp_of_function_attribute : function_attribute -> Sexplib0.Sexp.t

  val get_function_attribute : func -> function_attribute -> int
  (** [hipFuncGetAttribute] on a linked kernel. The attribute of most interest to a code generator
      is [HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES]: the kernel's private (scratch) segment size per
      work-item. The compiler decides it, and the runtime only validates it at dispatch — by which
      point an over-budget kernel aborts the queue rather than failing gracefully. Querying it
      after linking lets a caller decline such a kernel before it is ever launched. *)

  val occupancy_max_active_blocks_per_multiprocessor :
    ?shared_mem_bytes:int -> ?disable_caching_override:bool -> func -> block_size:int -> int
  (** [hipModuleOccupancyMaxActiveBlocksPerMultiprocessor], the counterpart of CUDA's
      [cuOccupancyMaxActiveBlocksPerMultiprocessor]: how many blocks of [block_size] work-items can
      be resident on one compute unit at once, given the kernel's register and shared-memory
      demands. Multiply by the device's [multiprocessor_count] ({!Device.get_attributes}) to size a
      persistent-block grid, or compare against the device's per-CU work-item ceiling for an
      occupancy ratio.

      [shared_mem_bytes] is the dynamic shared memory per block the kernel would be launched with
      (default [0]), same quantity as {!Stream.launch_kernel}'s [~shared_mem_bytes].
      [disable_caching_override] (default [false]) switches to the [...WithFlags] entry point with
      [hipOccupancyDisableCachingOverride]. The AMD runtime accepts the flag but currently ignores
      it (the query returns the same answer as the default), so it is only worth passing for
      source compatibility with CUDA code that sets
      [CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE]. *)

  val occupancy_max_potential_block_size :
    ?shared_mem_bytes:int -> ?block_size_limit:int -> func -> int * int
  (** [hipModuleOccupancyMaxPotentialBlockSize], the counterpart of CUDA's
      [cuOccupancyMaxPotentialBlockSize]: returns [(min_grid_size, block_size)], a block size that
      achieves the maximum potential occupancy for the kernel, and the minimum grid size needed to
      fill the device with blocks of that size.

      [shared_mem_bytes] is the dynamic shared memory per block (default [0]); [block_size_limit]
      caps the block size, with [0] (the default) meaning no limit beyond the kernel's own. *)

  val get_global : t -> name:string -> Deviceptr.t * Unsigned.size_t
  (** Retrieves the device pointer and size of the [__device__] global variable with the given
      name. The result does not own its memory (freeing it is a no-op for the module's lifetime;
      do not call [mem_free] on it). *)
end

(** A stream (queue) of asynchronous operations on a specific device, with kernel-argument
    lifetime bookkeeping and {!Delimited_event} ownership. *)
module Stream : sig
  type t

  val sexp_of_t : t -> Sexplib0.Sexp.t

  val no_stream : t
  (** The NULL stream, which is the main synchronization stream of a device. *)

  val create : ?non_blocking:bool -> ?lower_priority:int -> unit -> t
  (** Creates a stream on the current device. [non_blocking] means the stream does not synchronize
      with the NULL stream. Streams with higher [lower_priority] numbers are lower priority. The
      stream is destroyed (after synchronizing it) when the result is garbage-collected. *)

  val destroy : t -> unit
  (** Synchronizes the stream, releases its bookkeeping, and destroys it. Idempotent: streams are
      also destroyed when garbage-collected, and only the first call takes effect. *)

  val get_device : t -> Device.t
  val get_device_id : t -> int

  val is_ready : t -> bool
  (** Returns [false] when the stream has pending operations, [true] otherwise (releasing kernel
      argument lifetimes in the latter case). *)

  val synchronize : t -> unit
  (** Blocks until the stream's tasks complete; releases kernel-argument lifetimes and the
      stream's {!Delimited_event}s. *)

  val mem_alloc : t -> size_in_bytes:int -> Deviceptr.t
  (** Stream-ordered allocation ([hipMallocAsync]); requires
      {!Device.attributes.memory_pools_supported}. The memory is freed (stream-ordered) when the
      result is garbage-collected, if it isn't already freed. *)

  val mem_free : t -> Deviceptr.t -> unit
  (** Stream-ordered free; idempotent. *)

  val memcpy_H_to_D :
    ?host_offset:int ->
    ?length:int ->
    ?dst_offset:int ->
    dst:Deviceptr.t ->
    src:('a, 'b, 'c) Bigarray.Genarray.t ->
    t ->
    unit
  (** Asynchronous version of {!Deviceptr.memcpy_H_to_D}. NOTE: the caller must keep the bigarray
      and the device pointer alive until the copy completes (e.g. until {!synchronize}). *)

  val memcpy_D_to_H :
    ?host_offset:int ->
    ?length:int ->
    ?src_offset:int ->
    dst:('a, 'b, 'c) Bigarray.Genarray.t ->
    src:Deviceptr.t ->
    t ->
    unit
  (** Asynchronous version of {!Deviceptr.memcpy_D_to_H}; same lifetime caveats as
      {!memcpy_H_to_D}. *)

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
  (** Asynchronous version of {!Deviceptr.memcpy_D_to_D}. *)

  val memcpy_peer :
    ?kind:('a, 'b) Bigarray.kind ->
    ?length:int ->
    ?size_in_bytes:int ->
    ?dst_offset:int ->
    ?src_offset:int ->
    dst:Deviceptr.t ->
    dst_device:Device.t ->
    src:Deviceptr.t ->
    src_device:Device.t ->
    t ->
    unit
  (** Asynchronous version of {!Deviceptr.memcpy_peer}. *)

  val memset_d8 : ?offset:int -> Deviceptr.t -> Unsigned.uchar -> length:int -> t -> unit
  val memset_d16 : ?offset:int -> Deviceptr.t -> Unsigned.ushort -> length:int -> t -> unit
  val memset_d32 : ?offset:int -> Deviceptr.t -> int -> length:int -> t -> unit

  type size_t = Unsigned.size_t

  val sexp_of_size_t : size_t -> Sexplib0.Sexp.t

  (** Kernel launch arguments. *)
  type kernel_param =
    | Tensor of Deviceptr.t  (** A device pointer argument. *)
    | Tensor_at of Deviceptr.region  (** A device pointer argument at a byte offset. *)
    | Int of int  (** A C [int] argument. *)
    | Size_t of size_t  (** A C [size_t] argument. *)
    | Single of float  (** A C [float] argument. *)
    | Double of float  (** A C [double] argument. *)

  val sexp_of_kernel_param : kernel_param -> Sexplib0.Sexp.t

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
  (** Launches the kernel on the stream. [shared_mem_bytes] is the dynamic shared memory size.
      The arguments' lifetimes are retained by the stream until it is synchronized or ready. *)

  val total_unreleased_unfinished_delimited_events : t -> int * int * int
  (** Debugging helper: (total, unreleased, unreleased-and-unfinished) delimited events owned by
      the stream. *)

  val get_total_live_streams : unit -> int
  (** Debugging helper: the number of streams created and not yet destroyed. *)
end

(** Events for synchronizing (within) streams, and for timing. *)
module Event : sig
  type t

  val sexp_of_t : t -> Sexplib0.Sexp.t

  val create : ?blocking_sync:bool -> ?enable_timing:bool -> ?interprocess:bool -> unit -> t
  (** The event is destroyed when the result is garbage-collected (there is deliberately no public
      [destroy]: it would double-destroy with the GC finalizer; see also {!Delimited_event}). *)

  val query : t -> bool

  val record : t -> Stream.t -> unit
  (** Captures in the event the contents of the stream at the time of the call. *)

  val synchronize : t -> unit
  (** Blocks until the completion of all work captured in the event. *)

  val wait : Stream.t -> t -> unit
  (** Makes all future work submitted to the stream wait for the event. *)

  val elapsed_time : start:t -> end_:t -> float
  (** In milliseconds (with a resolution of around 1 microsecond); both events must have
      [enable_timing]. *)
end

(** Events optimized for once-per-event synchronization, that are automatically destroyed after
    the first (successful) synchronization. Delimited events are owned by the stream they are
    recorded on: the stream also releases them on {!Stream.synchronize}. *)
module Delimited_event : sig
  type t

  val sexp_of_t : t -> Sexplib0.Sexp.t

  val record : ?blocking_sync:bool -> ?interprocess:bool -> Stream.t -> t
  (** Records an event capturing the work scheduled on the stream so far; the event is owned by
      the stream. *)

  val query : t -> bool
  (** Returns [true] if the event is released (i.e. already synchronized). *)

  val synchronize : t -> unit
  (** Blocks until the event completes, then destroys (releases) it. No-op if already released. *)

  val wait : Stream.t -> t -> unit
  (** Makes all future work submitted to the stream wait for the event; no-op if the event is
      released. *)

  val is_released : t -> bool
end

(** HIP graphs: capture a sequence of stream operations once, then replay it with a single launch
    call. Mirrors the CUDA graph API (cudajit's [Cuda.Graph]). *)
module Graph : sig
  type t
  (** A captured graph template ([hipGraph_t]), not yet executable. The graph value retains the
      OCaml-side resources of the operations captured into it (kernels — and through them their
      modules — kernel parameters, device pointers), so a {!Stream.synchronize} of the capture
      stream cannot free resources a replay still needs. *)

  type exec
  (** An instantiated, executable graph ([hipGraphExec_t]); retains the captured operations'
      resources like {!t}. *)

  val sexp_of_t : t -> Sexplib0.Sexp.t
  val sexp_of_exec : exec -> Sexplib0.Sexp.t

  type capture_mode = GLOBAL | THREAD_LOCAL | RELAXED
  (** [hipStreamCaptureMode]. [GLOBAL] (the HIP default) invalidates the capture on potentially
      unsafe API calls from any thread; [THREAD_LOCAL] restricts that to the capturing thread;
      [RELAXED] imposes no cross-thread restrictions. *)

  val sexp_of_capture_mode : capture_mode -> Sexplib0.Sexp.t

  val begin_capture : ?mode:capture_mode -> Stream.t -> unit
  (** Starts capturing work submitted to the (non-default) stream instead of executing it. [mode]
      defaults to [GLOBAL]. *)

  val end_capture : Stream.t -> t
  (** Ends the capture begun by {!begin_capture} and returns the captured graph. Raises
      [Hip_error] if the capture was invalidated. The graph value is finalized using
      [hipGraphDestroy]; it can also be released eagerly with {!destroy}. *)

  val instantiate : t -> exec
  (** Creates an executable graph from a captured graph; the template can be destroyed afterwards.
      The exec value is finalized using [hipGraphExecDestroy]; it can also be released eagerly with
      {!exec_destroy}, but only once no launch of it is pending (synchronize the stream first). *)

  val launch : exec -> Stream.t -> unit
  (** Enqueues the whole captured work sequence on the stream, as one API call. The stream retains
      the exec until it is synchronized (mirroring {!Stream.launch_kernel}'s argument retention),
      so dropping the last reference to a launched exec cannot destroy it while the launch is
      pending. *)

  val destroy : t -> unit
  (** Eagerly destroys the graph template. Idempotent; the finalizer covers the non-eager case. *)

  val exec_destroy : exec -> unit
  (** Eagerly destroys the executable graph. Idempotent; the finalizer covers the non-eager case.
  *)
end
