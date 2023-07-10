open Ctypes
open Sexplib0.Sexp_conv

type cu_result =
  | CUDA_SUCCESS
  | CUDA_ERROR_INVALID_VALUE
  | CUDA_ERROR_OUT_OF_MEMORY
  | CUDA_ERROR_NOT_INITIALIZED
  | CUDA_ERROR_DEINITIALIZED
  | CUDA_ERROR_PROFILER_DISABLED
  | CUDA_ERROR_PROFILER_NOT_INITIALIZED
  | CUDA_ERROR_PROFILER_ALREADY_STARTED
  | CUDA_ERROR_PROFILER_ALREADY_STOPPED
  | CUDA_ERROR_STUB_LIBRARY
  (* | CUDA_ERROR_DEVICE_UNAVAILABLE *)
  | CUDA_ERROR_NO_DEVICE
  | CUDA_ERROR_INVALID_DEVICE
  | CUDA_ERROR_DEVICE_NOT_LICENSED
  | CUDA_ERROR_INVALID_IMAGE
  | CUDA_ERROR_INVALID_CONTEXT
  | CUDA_ERROR_CONTEXT_ALREADY_CURRENT
  | CUDA_ERROR_MAP_FAILED
  | CUDA_ERROR_UNMAP_FAILED
  | CUDA_ERROR_ARRAY_IS_MAPPED
  | CUDA_ERROR_ALREADY_MAPPED
  | CUDA_ERROR_NO_BINARY_FOR_GPU
  | CUDA_ERROR_ALREADY_ACQUIRED
  | CUDA_ERROR_NOT_MAPPED
  | CUDA_ERROR_NOT_MAPPED_AS_ARRAY
  | CUDA_ERROR_NOT_MAPPED_AS_POINTER
  | CUDA_ERROR_ECC_UNCORRECTABLE
  | CUDA_ERROR_UNSUPPORTED_LIMIT
  | CUDA_ERROR_CONTEXT_ALREADY_IN_USE
  | CUDA_ERROR_PEER_ACCESS_UNSUPPORTED
  | CUDA_ERROR_INVALID_PTX
  | CUDA_ERROR_INVALID_GRAPHICS_CONTEXT
  | CUDA_ERROR_NVLINK_UNCORRECTABLE
  | CUDA_ERROR_JIT_COMPILER_NOT_FOUND
  | CUDA_ERROR_UNSUPPORTED_PTX_VERSION
  | CUDA_ERROR_JIT_COMPILATION_DISABLED
  | CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY
  (* | CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC *)
  | CUDA_ERROR_INVALID_SOURCE
  | CUDA_ERROR_FILE_NOT_FOUND
  | CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND
  | CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  | CUDA_ERROR_OPERATING_SYSTEM
  | CUDA_ERROR_INVALID_HANDLE
  | CUDA_ERROR_ILLEGAL_STATE
  | CUDA_ERROR_NOT_FOUND
  | CUDA_ERROR_NOT_READY
  | CUDA_ERROR_ILLEGAL_ADDRESS
  | CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
  | CUDA_ERROR_LAUNCH_TIMEOUT
  | CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING
  | CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
  | CUDA_ERROR_PEER_ACCESS_NOT_ENABLED
  | CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
  | CUDA_ERROR_CONTEXT_IS_DESTROYED
  | CUDA_ERROR_ASSERT
  | CUDA_ERROR_TOO_MANY_PEERS
  | CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED
  | CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED
  | CUDA_ERROR_HARDWARE_STACK_ERROR
  | CUDA_ERROR_ILLEGAL_INSTRUCTION
  | CUDA_ERROR_MISALIGNED_ADDRESS
  | CUDA_ERROR_INVALID_ADDRESS_SPACE
  | CUDA_ERROR_INVALID_PC
  | CUDA_ERROR_LAUNCH_FAILED
  | CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE
  | CUDA_ERROR_NOT_PERMITTED
  | CUDA_ERROR_NOT_SUPPORTED
  | CUDA_ERROR_SYSTEM_NOT_READY
  | CUDA_ERROR_SYSTEM_DRIVER_MISMATCH
  | CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE
  | CUDA_ERROR_MPS_CONNECTION_FAILED
  | CUDA_ERROR_MPS_RPC_FAILURE
  | CUDA_ERROR_MPS_SERVER_NOT_READY
  | CUDA_ERROR_MPS_MAX_CLIENTS_REACHED
  | CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED
  (* | CUDA_ERROR_MPS_CLIENT_TERMINATED *)
  (* | CUDA_ERROR_CDP_NOT_SUPPORTED *)
  (* | CUDA_ERROR_CDP_VERSION_MISMATCH *)
  | CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
  | CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
  | CUDA_ERROR_STREAM_CAPTURE_MERGE
  | CUDA_ERROR_STREAM_CAPTURE_UNMATCHED
  | CUDA_ERROR_STREAM_CAPTURE_UNJOINED
  | CUDA_ERROR_STREAM_CAPTURE_ISOLATION
  | CUDA_ERROR_STREAM_CAPTURE_IMPLICIT
  | CUDA_ERROR_CAPTURED_EVENT
  | CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD
  | CUDA_ERROR_TIMEOUT
  | CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE
  | CUDA_ERROR_EXTERNAL_DEVICE
  (* | CUDA_ERROR_INVALID_CLUSTER_SIZE *)
  | CUDA_ERROR_UNKNOWN
  | CUDA_ERROR_UNCATEGORIZED of int64
[@@deriving sexp]

type cu_device = Cu_device of int [@@deriving sexp]

type cu_jit_option =
  | CU_JIT_MAX_REGISTERS
  | CU_JIT_THREADS_PER_BLOCK
  | CU_JIT_WALL_TIME
  | CU_JIT_INFO_LOG_BUFFER
  | CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
  | CU_JIT_ERROR_LOG_BUFFER
  | CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
  | CU_JIT_OPTIMIZATION_LEVEL
  | CU_JIT_TARGET_FROM_CUCONTEXT
  | CU_JIT_TARGET
  | CU_JIT_FALLBACK_STRATEGY
  | CU_JIT_GENERATE_DEBUG_INFO
  | CU_JIT_LOG_VERBOSE
  | CU_JIT_GENERATE_LINE_INFO
  | CU_JIT_CACHE_MODE
  | CU_JIT_NEW_SM3X_OPT
  | CU_JIT_FAST_COMPILE
  | CU_JIT_GLOBAL_SYMBOL_NAMES
  | CU_JIT_GLOBAL_SYMBOL_ADDRESSES
  | CU_JIT_GLOBAL_SYMBOL_COUNT
  | CU_JIT_LTO
  | CU_JIT_FTZ
  | CU_JIT_PREC_DIV
  | CU_JIT_PREC_SQRT
  | CU_JIT_FMA
  (*| CU_JIT_REFERENCED_KERNEL_NAMES
    | CU_JIT_REFERENCED_KERNEL_COUNT
    | CU_JIT_REFERENCED_VARIABLE_NAMES
    | CU_JIT_REFERENCED_VARIABLE_COUNT
    | CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES
    | CU_JIT_POSITION_INDEPENDENT_CODE *)
  | CU_JIT_NUM_OPTIONS
  | CU_JIT_UNCATEGORIZED of int64
[@@deriving sexp]

type cu_context_t
type cu_context = cu_context_t structure ptr

let cu_context : cu_context typ = typedef (ptr @@ structure "CUctx_st") "CUcontext"

type cu_module_t
type cu_module = cu_module_t structure ptr

let cu_module : cu_module typ = typedef (ptr @@ structure "CUmod_st") "CUmodule"

type cu_function_t
type cu_function = cu_function_t structure ptr

let cu_function : cu_function typ = typedef (ptr @@ structure "CUfunc_st") "CUfunction"

(** CUdeviceptr is defined as an unsigned integer type whose size matches the size of a pointer on
    the target platform. *)
let cu_deviceptr_v2 = typedef uint64_t "CUdeviceptr_v2"

let cu_deviceptr = typedef cu_deviceptr_v2 "CUdeviceptr"

type cu_stream_t
type cu_stream = cu_stream_t structure ptr

let cu_stream : cu_stream typ = typedef (ptr @@ structure "CUstream_st") "CUstream"

type cu_jit_target =
  | CU_TARGET_COMPUTE_30
  | CU_TARGET_COMPUTE_32
  | CU_TARGET_COMPUTE_35
  | CU_TARGET_COMPUTE_37
  | CU_TARGET_COMPUTE_50
  | CU_TARGET_COMPUTE_52
  | CU_TARGET_COMPUTE_53
  | CU_TARGET_COMPUTE_60
  | CU_TARGET_COMPUTE_61
  | CU_TARGET_COMPUTE_62
  | CU_TARGET_COMPUTE_70
  | CU_TARGET_COMPUTE_72
  | CU_TARGET_COMPUTE_75
  | CU_TARGET_COMPUTE_80
  | CU_TARGET_COMPUTE_86
  (* | CU_TARGET_COMPUTE_87
     | CU_TARGET_COMPUTE_89
     | CU_TARGET_COMPUTE_90
     | CU_TARGET_COMPUTE_90A *)
  | CU_TARGET_UNCATEGORIZED of int64
[@@deriving sexp]

type cu_jit_fallback = CU_PREFER_PTX | CU_PREFER_BINARY | CU_PREFER_UNCATEGORIZED of int64 [@@deriving sexp]

type cu_jit_cache_mode =
  | CU_JIT_CACHE_OPTION_NONE
  | CU_JIT_CACHE_OPTION_CG
  | CU_JIT_CACHE_OPTION_CA
  | CU_JIT_CACHE_OPTION_UNCATEGORIZED of int64
[@@deriving sexp]

type cu_device_attribute =
  | CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
  | CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
  | CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
  | CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
  | CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
  | CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
  | CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
  | CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
  | CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY
  | CU_DEVICE_ATTRIBUTE_WARP_SIZE
  | CU_DEVICE_ATTRIBUTE_MAX_PITCH
  | CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
  | CU_DEVICE_ATTRIBUTE_CLOCK_RATE
  | CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT
  | CU_DEVICE_ATTRIBUTE_GPU_OVERLAP
  | CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
  | CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT
  | CU_DEVICE_ATTRIBUTE_INTEGRATED
  | CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY
  | CU_DEVICE_ATTRIBUTE_COMPUTE_MODE
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
  | CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT
  | CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS
  | CU_DEVICE_ATTRIBUTE_ECC_ENABLED
  | CU_DEVICE_ATTRIBUTE_PCI_BUS_ID
  | CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID
  | CU_DEVICE_ATTRIBUTE_TCC_DRIVER
  | CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE
  | CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
  | CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
  | CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
  | CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
  (* | CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING *)
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS
  | CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE
  | CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID
  | CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT
  | CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
  | CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
  | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH
  | CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
  | CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR
  | CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY
  | CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD
  | CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID
  | CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO
  | CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS
  | CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
  | CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
  | CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH
  | CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH
  | CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
  | CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES
  | CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
  | CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST
  | CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR
  | CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE
  | CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE
  | CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK
  | CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED
  (* | CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED *)
  | CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED
  | CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS
  | CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING
  | CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES
  (* | CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH *)
  (* | CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED *)
  | CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS
  | CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR
  (* | CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED *)
  (* | CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED *)
  (* | CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT *)
  (* | CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED *)
  (* | CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS *)
  (* | CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED *)
  | CU_DEVICE_ATTRIBUTE_MAX
  | CU_DEVICE_ATTRIBUTE_UNCATEGORIZED of int64
[@@deriving sexp]

type cu_computemode =
  | CU_COMPUTEMODE_DEFAULT
  | CU_COMPUTEMODE_PROHIBITED
  | CU_COMPUTEMODE_EXCLUSIVE_PROCESS
  | CU_COMPUTEMODE_UNCATEGORIZED of int64
[@@deriving sexp]

type cu_flush_GPU_direct_RDMA_writes_options =
  | CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST
  | CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS
  | CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_UNCATEGORIZED of int64
[@@deriving sexp]

type cu_limit =
  | CU_LIMIT_STACK_SIZE
  | CU_LIMIT_PRINTF_FIFO_SIZE
  | CU_LIMIT_MALLOC_HEAP_SIZE
  | CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH
  | CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT
  | CU_LIMIT_MAX_L2_FETCH_GRANULARITY
  | CU_LIMIT_PERSISTING_L2_CACHE_SIZE
  | CU_LIMIT_MAX
  | CU_LIMIT_UNCATEGORIZED of int64
[@@deriving sexp]

module Types (T : Ctypes.TYPE) = struct
  let cu_device_v1 = T.typedef T.int "CUdevice_v1"
  let cu_device_t = T.typedef cu_device_v1 "CUdevice"
  let cu_device = T.view ~read:(fun i -> Cu_device i) ~write:(function Cu_device i -> i) cu_device_t
  let cuda_success = T.constant "CUDA_SUCCESS" T.int64_t
  let cuda_error_invalid_value = T.constant "CUDA_ERROR_INVALID_VALUE" T.int64_t
  let cuda_error_out_of_memory = T.constant "CUDA_ERROR_OUT_OF_MEMORY" T.int64_t
  let cuda_error_not_initialized = T.constant "CUDA_ERROR_NOT_INITIALIZED" T.int64_t
  let cuda_error_deinitialized = T.constant "CUDA_ERROR_DEINITIALIZED" T.int64_t
  let cuda_error_profiler_disabled = T.constant "CUDA_ERROR_PROFILER_DISABLED" T.int64_t
  let cuda_error_profiler_not_initialized = T.constant "CUDA_ERROR_PROFILER_NOT_INITIALIZED" T.int64_t
  let cuda_error_profiler_already_started = T.constant "CUDA_ERROR_PROFILER_ALREADY_STARTED" T.int64_t
  let cuda_error_profiler_already_stopped = T.constant "CUDA_ERROR_PROFILER_ALREADY_STOPPED" T.int64_t
  let cuda_error_stub_library = T.constant "CUDA_ERROR_STUB_LIBRARY" T.int64_t

  (* let cuda_error_device_unavailable = T.constant "CUDA_ERROR_DEVICE_UNAVAILABLE" T.int64_t *)
  let cuda_error_no_device = T.constant "CUDA_ERROR_NO_DEVICE" T.int64_t
  let cuda_error_invalid_device = T.constant "CUDA_ERROR_INVALID_DEVICE" T.int64_t
  let cuda_error_device_not_licensed = T.constant "CUDA_ERROR_DEVICE_NOT_LICENSED" T.int64_t
  let cuda_error_invalid_image = T.constant "CUDA_ERROR_INVALID_IMAGE" T.int64_t
  let cuda_error_invalid_context = T.constant "CUDA_ERROR_INVALID_CONTEXT" T.int64_t
  let cuda_error_context_already_current = T.constant "CUDA_ERROR_CONTEXT_ALREADY_CURRENT" T.int64_t
  let cuda_error_map_failed = T.constant "CUDA_ERROR_MAP_FAILED" T.int64_t
  let cuda_error_unmap_failed = T.constant "CUDA_ERROR_UNMAP_FAILED" T.int64_t
  let cuda_error_array_is_mapped = T.constant "CUDA_ERROR_ARRAY_IS_MAPPED" T.int64_t
  let cuda_error_already_mapped = T.constant "CUDA_ERROR_ALREADY_MAPPED" T.int64_t
  let cuda_error_no_binary_for_gpu = T.constant "CUDA_ERROR_NO_BINARY_FOR_GPU" T.int64_t
  let cuda_error_already_acquired = T.constant "CUDA_ERROR_ALREADY_ACQUIRED" T.int64_t
  let cuda_error_not_mapped = T.constant "CUDA_ERROR_NOT_MAPPED" T.int64_t
  let cuda_error_not_mapped_as_array = T.constant "CUDA_ERROR_NOT_MAPPED_AS_ARRAY" T.int64_t
  let cuda_error_not_mapped_as_pointer = T.constant "CUDA_ERROR_NOT_MAPPED_AS_POINTER" T.int64_t
  let cuda_error_ecc_uncorrectable = T.constant "CUDA_ERROR_ECC_UNCORRECTABLE" T.int64_t
  let cuda_error_unsupported_limit = T.constant "CUDA_ERROR_UNSUPPORTED_LIMIT" T.int64_t
  let cuda_error_context_already_in_use = T.constant "CUDA_ERROR_CONTEXT_ALREADY_IN_USE" T.int64_t
  let cuda_error_peer_access_unsupported = T.constant "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED" T.int64_t
  let cuda_error_invalid_ptx = T.constant "CUDA_ERROR_INVALID_PTX" T.int64_t
  let cuda_error_invalid_graphics_context = T.constant "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT" T.int64_t
  let cuda_error_nvlink_uncorrectable = T.constant "CUDA_ERROR_NVLINK_UNCORRECTABLE" T.int64_t
  let cuda_error_jit_compiler_not_found = T.constant "CUDA_ERROR_JIT_COMPILER_NOT_FOUND" T.int64_t
  let cuda_error_unsupported_ptx_version = T.constant "CUDA_ERROR_UNSUPPORTED_PTX_VERSION" T.int64_t
  let cuda_error_jit_compilation_disabled = T.constant "CUDA_ERROR_JIT_COMPILATION_DISABLED" T.int64_t
  let cuda_error_unsupported_exec_affinity = T.constant "CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY" T.int64_t

  (* let cuda_error_unsupported_devside_sync = T.constant "CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC" T.int64_t *)
  let cuda_error_invalid_source = T.constant "CUDA_ERROR_INVALID_SOURCE" T.int64_t
  let cuda_error_file_not_found = T.constant "CUDA_ERROR_FILE_NOT_FOUND" T.int64_t

  let cuda_error_shared_object_symbol_not_found =
    T.constant "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND" T.int64_t

  let cuda_error_shared_object_init_failed = T.constant "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED" T.int64_t
  let cuda_error_operating_system = T.constant "CUDA_ERROR_OPERATING_SYSTEM" T.int64_t
  let cuda_error_invalid_handle = T.constant "CUDA_ERROR_INVALID_HANDLE" T.int64_t
  let cuda_error_illegal_state = T.constant "CUDA_ERROR_ILLEGAL_STATE" T.int64_t
  let cuda_error_not_found = T.constant "CUDA_ERROR_NOT_FOUND" T.int64_t
  let cuda_error_not_ready = T.constant "CUDA_ERROR_NOT_READY" T.int64_t
  let cuda_error_illegal_address = T.constant "CUDA_ERROR_ILLEGAL_ADDRESS" T.int64_t
  let cuda_error_launch_out_of_resources = T.constant "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES" T.int64_t
  let cuda_error_launch_timeout = T.constant "CUDA_ERROR_LAUNCH_TIMEOUT" T.int64_t

  let cuda_error_launch_incompatible_texturing =
    T.constant "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING" T.int64_t

  let cuda_error_peer_access_already_enabled = T.constant "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED" T.int64_t
  let cuda_error_peer_access_not_enabled = T.constant "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED" T.int64_t
  let cuda_error_primary_context_active = T.constant "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE" T.int64_t
  let cuda_error_context_is_destroyed = T.constant "CUDA_ERROR_CONTEXT_IS_DESTROYED" T.int64_t
  let cuda_error_assert = T.constant "CUDA_ERROR_ASSERT" T.int64_t
  let cuda_error_too_many_peers = T.constant "CUDA_ERROR_TOO_MANY_PEERS" T.int64_t

  let cuda_error_host_memory_already_registered =
    T.constant "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED" T.int64_t

  let cuda_error_host_memory_not_registered = T.constant "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED" T.int64_t
  let cuda_error_hardware_stack_error = T.constant "CUDA_ERROR_HARDWARE_STACK_ERROR" T.int64_t
  let cuda_error_illegal_instruction = T.constant "CUDA_ERROR_ILLEGAL_INSTRUCTION" T.int64_t
  let cuda_error_misaligned_address = T.constant "CUDA_ERROR_MISALIGNED_ADDRESS" T.int64_t
  let cuda_error_invalid_address_space = T.constant "CUDA_ERROR_INVALID_ADDRESS_SPACE" T.int64_t
  let cuda_error_invalid_pc = T.constant "CUDA_ERROR_INVALID_PC" T.int64_t
  let cuda_error_launch_failed = T.constant "CUDA_ERROR_LAUNCH_FAILED" T.int64_t
  let cuda_error_cooperative_launch_too_large = T.constant "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE" T.int64_t
  let cuda_error_not_permitted = T.constant "CUDA_ERROR_NOT_PERMITTED" T.int64_t
  let cuda_error_not_supported = T.constant "CUDA_ERROR_NOT_SUPPORTED" T.int64_t
  let cuda_error_system_not_ready = T.constant "CUDA_ERROR_SYSTEM_NOT_READY" T.int64_t
  let cuda_error_system_driver_mismatch = T.constant "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH" T.int64_t

  let cuda_error_compat_not_supported_on_device =
    T.constant "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE" T.int64_t

  let cuda_error_mps_connection_failed = T.constant "CUDA_ERROR_MPS_CONNECTION_FAILED" T.int64_t
  let cuda_error_mps_rpc_failure = T.constant "CUDA_ERROR_MPS_RPC_FAILURE" T.int64_t
  let cuda_error_mps_server_not_ready = T.constant "CUDA_ERROR_MPS_SERVER_NOT_READY" T.int64_t
  let cuda_error_mps_max_clients_reached = T.constant "CUDA_ERROR_MPS_MAX_CLIENTS_REACHED" T.int64_t
  let cuda_error_mps_max_connections_reached = T.constant "CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED" T.int64_t

  (* let cuda_error_mps_client_terminated = T.constant "CUDA_ERROR_MPS_CLIENT_TERMINATED" T.int64_t *)
  (* let cuda_error_cdp_not_supported = T.constant "CUDA_ERROR_CDP_NOT_SUPPORTED" T.int64_t *)
  (* let cuda_error_cdp_version_mismatch = T.constant "CUDA_ERROR_CDP_VERSION_MISMATCH" T.int64_t *)
  let cuda_error_stream_capture_unsupported = T.constant "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED" T.int64_t
  let cuda_error_stream_capture_invalidated = T.constant "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED" T.int64_t
  let cuda_error_stream_capture_merge = T.constant "CUDA_ERROR_STREAM_CAPTURE_MERGE" T.int64_t
  let cuda_error_stream_capture_unmatched = T.constant "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED" T.int64_t
  let cuda_error_stream_capture_unjoined = T.constant "CUDA_ERROR_STREAM_CAPTURE_UNJOINED" T.int64_t
  let cuda_error_stream_capture_isolation = T.constant "CUDA_ERROR_STREAM_CAPTURE_ISOLATION" T.int64_t
  let cuda_error_stream_capture_implicit = T.constant "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT" T.int64_t
  let cuda_error_captured_event = T.constant "CUDA_ERROR_CAPTURED_EVENT" T.int64_t
  let cuda_error_stream_capture_wrong_thread = T.constant "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD" T.int64_t
  let cuda_error_timeout = T.constant "CUDA_ERROR_TIMEOUT" T.int64_t
  let cuda_error_graph_exec_update_failure = T.constant "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE" T.int64_t
  let cuda_error_external_device = T.constant "CUDA_ERROR_EXTERNAL_DEVICE" T.int64_t

  (* let cuda_error_invalid_cluster_size = T.constant "CUDA_ERROR_INVALID_CLUSTER_SIZE" T.int64_t *)
  let cuda_error_unknown = T.constant "CUDA_ERROR_UNKNOWN" T.int64_t

  let cu_result =
    T.enum ~typedef:true
      ~unexpected:(fun error_code -> CUDA_ERROR_UNCATEGORIZED error_code)
      "CUresult"
      [
        (CUDA_SUCCESS, cuda_success);
        (CUDA_ERROR_INVALID_VALUE, cuda_error_invalid_value);
        (CUDA_ERROR_OUT_OF_MEMORY, cuda_error_out_of_memory);
        (CUDA_ERROR_NOT_INITIALIZED, cuda_error_not_initialized);
        (CUDA_ERROR_DEINITIALIZED, cuda_error_deinitialized);
        (CUDA_ERROR_PROFILER_DISABLED, cuda_error_profiler_disabled);
        (CUDA_ERROR_PROFILER_NOT_INITIALIZED, cuda_error_profiler_not_initialized);
        (CUDA_ERROR_PROFILER_ALREADY_STARTED, cuda_error_profiler_already_started);
        (CUDA_ERROR_PROFILER_ALREADY_STOPPED, cuda_error_profiler_already_stopped);
        (CUDA_ERROR_STUB_LIBRARY, cuda_error_stub_library);
        (* (CUDA_ERROR_DEVICE_UNAVAILABLE, cuda_error_device_unavailable); *)
        (CUDA_ERROR_NO_DEVICE, cuda_error_no_device);
        (CUDA_ERROR_INVALID_DEVICE, cuda_error_invalid_device);
        (CUDA_ERROR_DEVICE_NOT_LICENSED, cuda_error_device_not_licensed);
        (CUDA_ERROR_INVALID_IMAGE, cuda_error_invalid_image);
        (CUDA_ERROR_INVALID_CONTEXT, cuda_error_invalid_context);
        (CUDA_ERROR_CONTEXT_ALREADY_CURRENT, cuda_error_context_already_current);
        (CUDA_ERROR_MAP_FAILED, cuda_error_map_failed);
        (CUDA_ERROR_UNMAP_FAILED, cuda_error_unmap_failed);
        (CUDA_ERROR_ARRAY_IS_MAPPED, cuda_error_array_is_mapped);
        (CUDA_ERROR_ALREADY_MAPPED, cuda_error_already_mapped);
        (CUDA_ERROR_NO_BINARY_FOR_GPU, cuda_error_no_binary_for_gpu);
        (CUDA_ERROR_ALREADY_ACQUIRED, cuda_error_already_acquired);
        (CUDA_ERROR_NOT_MAPPED, cuda_error_not_mapped);
        (CUDA_ERROR_NOT_MAPPED_AS_ARRAY, cuda_error_not_mapped_as_array);
        (CUDA_ERROR_NOT_MAPPED_AS_POINTER, cuda_error_not_mapped_as_pointer);
        (CUDA_ERROR_ECC_UNCORRECTABLE, cuda_error_ecc_uncorrectable);
        (CUDA_ERROR_UNSUPPORTED_LIMIT, cuda_error_unsupported_limit);
        (CUDA_ERROR_CONTEXT_ALREADY_IN_USE, cuda_error_context_already_in_use);
        (CUDA_ERROR_PEER_ACCESS_UNSUPPORTED, cuda_error_peer_access_unsupported);
        (CUDA_ERROR_INVALID_PTX, cuda_error_invalid_ptx);
        (CUDA_ERROR_INVALID_GRAPHICS_CONTEXT, cuda_error_invalid_graphics_context);
        (CUDA_ERROR_NVLINK_UNCORRECTABLE, cuda_error_nvlink_uncorrectable);
        (CUDA_ERROR_JIT_COMPILER_NOT_FOUND, cuda_error_jit_compiler_not_found);
        (CUDA_ERROR_UNSUPPORTED_PTX_VERSION, cuda_error_unsupported_ptx_version);
        (CUDA_ERROR_JIT_COMPILATION_DISABLED, cuda_error_jit_compilation_disabled);
        (CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY, cuda_error_unsupported_exec_affinity);
        (* (CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC, cuda_error_unsupported_devside_sync); *)
        (CUDA_ERROR_INVALID_SOURCE, cuda_error_invalid_source);
        (CUDA_ERROR_FILE_NOT_FOUND, cuda_error_file_not_found);
        (CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND, cuda_error_shared_object_symbol_not_found);
        (CUDA_ERROR_SHARED_OBJECT_INIT_FAILED, cuda_error_shared_object_init_failed);
        (CUDA_ERROR_OPERATING_SYSTEM, cuda_error_operating_system);
        (CUDA_ERROR_INVALID_HANDLE, cuda_error_invalid_handle);
        (CUDA_ERROR_ILLEGAL_STATE, cuda_error_illegal_state);
        (CUDA_ERROR_NOT_FOUND, cuda_error_not_found);
        (CUDA_ERROR_NOT_READY, cuda_error_not_ready);
        (CUDA_ERROR_ILLEGAL_ADDRESS, cuda_error_illegal_address);
        (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, cuda_error_launch_out_of_resources);
        (CUDA_ERROR_LAUNCH_TIMEOUT, cuda_error_launch_timeout);
        (CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, cuda_error_launch_incompatible_texturing);
        (CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED, cuda_error_peer_access_already_enabled);
        (CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, cuda_error_peer_access_not_enabled);
        (CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE, cuda_error_primary_context_active);
        (CUDA_ERROR_CONTEXT_IS_DESTROYED, cuda_error_context_is_destroyed);
        (CUDA_ERROR_ASSERT, cuda_error_assert);
        (CUDA_ERROR_TOO_MANY_PEERS, cuda_error_too_many_peers);
        (CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED, cuda_error_host_memory_already_registered);
        (CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, cuda_error_host_memory_not_registered);
        (CUDA_ERROR_HARDWARE_STACK_ERROR, cuda_error_hardware_stack_error);
        (CUDA_ERROR_ILLEGAL_INSTRUCTION, cuda_error_illegal_instruction);
        (CUDA_ERROR_MISALIGNED_ADDRESS, cuda_error_misaligned_address);
        (CUDA_ERROR_INVALID_ADDRESS_SPACE, cuda_error_invalid_address_space);
        (CUDA_ERROR_INVALID_PC, cuda_error_invalid_pc);
        (CUDA_ERROR_LAUNCH_FAILED, cuda_error_launch_failed);
        (CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE, cuda_error_cooperative_launch_too_large);
        (CUDA_ERROR_NOT_PERMITTED, cuda_error_not_permitted);
        (CUDA_ERROR_NOT_SUPPORTED, cuda_error_not_supported);
        (CUDA_ERROR_SYSTEM_NOT_READY, cuda_error_system_not_ready);
        (CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, cuda_error_system_driver_mismatch);
        (CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE, cuda_error_compat_not_supported_on_device);
        (CUDA_ERROR_MPS_CONNECTION_FAILED, cuda_error_mps_connection_failed);
        (CUDA_ERROR_MPS_RPC_FAILURE, cuda_error_mps_rpc_failure);
        (CUDA_ERROR_MPS_SERVER_NOT_READY, cuda_error_mps_server_not_ready);
        (CUDA_ERROR_MPS_MAX_CLIENTS_REACHED, cuda_error_mps_max_clients_reached);
        (CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED, cuda_error_mps_max_connections_reached);
        (* (CUDA_ERROR_MPS_CLIENT_TERMINATED, cuda_error_mps_client_terminated); *)
        (* (CUDA_ERROR_CDP_NOT_SUPPORTED, cuda_error_cdp_not_supported); *)
        (* (CUDA_ERROR_CDP_VERSION_MISMATCH, cuda_error_cdp_version_mismatch); *)
        (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, cuda_error_stream_capture_unsupported);
        (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED, cuda_error_stream_capture_invalidated);
        (CUDA_ERROR_STREAM_CAPTURE_MERGE, cuda_error_stream_capture_merge);
        (CUDA_ERROR_STREAM_CAPTURE_UNMATCHED, cuda_error_stream_capture_unmatched);
        (CUDA_ERROR_STREAM_CAPTURE_UNJOINED, cuda_error_stream_capture_unjoined);
        (CUDA_ERROR_STREAM_CAPTURE_ISOLATION, cuda_error_stream_capture_isolation);
        (CUDA_ERROR_STREAM_CAPTURE_IMPLICIT, cuda_error_stream_capture_implicit);
        (CUDA_ERROR_CAPTURED_EVENT, cuda_error_captured_event);
        (CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD, cuda_error_stream_capture_wrong_thread);
        (CUDA_ERROR_TIMEOUT, cuda_error_timeout);
        (CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE, cuda_error_graph_exec_update_failure);
        (CUDA_ERROR_EXTERNAL_DEVICE, cuda_error_external_device);
        (* (CUDA_ERROR_INVALID_CLUSTER_SIZE, cuda_error_invalid_cluster_size); *)
        (CUDA_ERROR_UNKNOWN, cuda_error_unknown);
      ]

  let cu_jit_max_registers = T.constant "CU_JIT_MAX_REGISTERS" T.int64_t
  let cu_jit_threads_per_block = T.constant "CU_JIT_THREADS_PER_BLOCK" T.int64_t
  let cu_jit_wall_time = T.constant "CU_JIT_WALL_TIME" T.int64_t
  let cu_jit_info_log_buffer = T.constant "CU_JIT_INFO_LOG_BUFFER" T.int64_t
  let cu_jit_info_log_buffer_size_bytes = T.constant "CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES" T.int64_t
  let cu_jit_error_log_buffer = T.constant "CU_JIT_ERROR_LOG_BUFFER" T.int64_t
  let cu_jit_error_log_buffer_size_bytes = T.constant "CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES" T.int64_t
  let cu_jit_optimization_level = T.constant "CU_JIT_OPTIMIZATION_LEVEL" T.int64_t
  let cu_jit_target_from_cucontext = T.constant "CU_JIT_TARGET_FROM_CUCONTEXT" T.int64_t
  let cu_jit_target = T.constant "CU_JIT_TARGET" T.int64_t
  let cu_jit_fallback_strategy = T.constant "CU_JIT_FALLBACK_STRATEGY" T.int64_t
  let cu_jit_generate_debug_info = T.constant "CU_JIT_GENERATE_DEBUG_INFO" T.int64_t
  let cu_jit_log_verbose = T.constant "CU_JIT_LOG_VERBOSE" T.int64_t
  let cu_jit_generate_line_info = T.constant "CU_JIT_GENERATE_LINE_INFO" T.int64_t
  let cu_jit_cache_mode = T.constant "CU_JIT_CACHE_MODE" T.int64_t
  let cu_jit_new_sm3x_opt = T.constant "CU_JIT_NEW_SM3X_OPT" T.int64_t
  let cu_jit_fast_compile = T.constant "CU_JIT_FAST_COMPILE" T.int64_t
  let cu_jit_global_symbol_names = T.constant "CU_JIT_GLOBAL_SYMBOL_NAMES" T.int64_t
  let cu_jit_global_symbol_addresses = T.constant "CU_JIT_GLOBAL_SYMBOL_ADDRESSES" T.int64_t
  let cu_jit_global_symbol_count = T.constant "CU_JIT_GLOBAL_SYMBOL_COUNT" T.int64_t
  let cu_jit_lto = T.constant "CU_JIT_LTO" T.int64_t
  let cu_jit_ftz = T.constant "CU_JIT_FTZ" T.int64_t
  let cu_jit_prec_div = T.constant "CU_JIT_PREC_DIV" T.int64_t
  let cu_jit_prec_sqrt = T.constant "CU_JIT_PREC_SQRT" T.int64_t
  let cu_jit_fma = T.constant "CU_JIT_FMA" T.int64_t

  (* let cu_jit_referenced_kernel_names = T.constant "CU_JIT_REFERENCED_KERNEL_NAMES" T.int64_t
     let cu_jit_referenced_kernel_count = T.constant "CU_JIT_REFERENCED_KERNEL_COUNT" T.int64_t
     let cu_jit_referenced_variable_names = T.constant "CU_JIT_REFERENCED_VARIABLE_NAMES" T.int64_t
     let cu_jit_referenced_variable_count = T.constant "CU_JIT_REFERENCED_VARIABLE_COUNT" T.int64_t
     let cu_jit_optimize_unused_device_variables = T.constant "CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES" T.int64_t
     let cu_jit_position_independent_code = T.constant "CU_JIT_POSITION_INDEPENDENT_CODE" T.int64_t *)
  let cu_jit_num_options = T.constant "CU_JIT_NUM_OPTIONS" T.int64_t

  let cu_jit_option =
    T.enum ~typedef:true
      ~unexpected:(fun error_code -> CU_JIT_UNCATEGORIZED error_code)
      "CUjit_option"
      [
        (CU_JIT_MAX_REGISTERS, cu_jit_max_registers);
        (CU_JIT_THREADS_PER_BLOCK, cu_jit_threads_per_block);
        (CU_JIT_WALL_TIME, cu_jit_wall_time);
        (CU_JIT_INFO_LOG_BUFFER, cu_jit_info_log_buffer);
        (CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, cu_jit_info_log_buffer_size_bytes);
        (CU_JIT_ERROR_LOG_BUFFER, cu_jit_error_log_buffer);
        (CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, cu_jit_error_log_buffer_size_bytes);
        (CU_JIT_OPTIMIZATION_LEVEL, cu_jit_optimization_level);
        (CU_JIT_TARGET_FROM_CUCONTEXT, cu_jit_target_from_cucontext);
        (CU_JIT_TARGET, cu_jit_target);
        (CU_JIT_FALLBACK_STRATEGY, cu_jit_fallback_strategy);
        (CU_JIT_GENERATE_DEBUG_INFO, cu_jit_generate_debug_info);
        (CU_JIT_LOG_VERBOSE, cu_jit_log_verbose);
        (CU_JIT_GENERATE_LINE_INFO, cu_jit_generate_line_info);
        (CU_JIT_CACHE_MODE, cu_jit_cache_mode);
        (CU_JIT_NEW_SM3X_OPT, cu_jit_new_sm3x_opt);
        (CU_JIT_FAST_COMPILE, cu_jit_fast_compile);
        (CU_JIT_GLOBAL_SYMBOL_NAMES, cu_jit_global_symbol_names);
        (CU_JIT_GLOBAL_SYMBOL_ADDRESSES, cu_jit_global_symbol_addresses);
        (CU_JIT_GLOBAL_SYMBOL_COUNT, cu_jit_global_symbol_count);
        (CU_JIT_LTO, cu_jit_lto);
        (CU_JIT_FTZ, cu_jit_ftz);
        (CU_JIT_PREC_DIV, cu_jit_prec_div);
        (CU_JIT_PREC_SQRT, cu_jit_prec_sqrt);
        (CU_JIT_FMA, cu_jit_fma);
        (* (CU_JIT_REFERENCED_KERNEL_NAMES, cu_jit_referenced_kernel_names);
           (CU_JIT_REFERENCED_KERNEL_COUNT, cu_jit_referenced_kernel_count);
           (CU_JIT_REFERENCED_VARIABLE_NAMES, cu_jit_referenced_variable_names);
           (CU_JIT_REFERENCED_VARIABLE_COUNT, cu_jit_referenced_variable_count);
           (CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES, cu_jit_optimize_unused_device_variables);
           (CU_JIT_POSITION_INDEPENDENT_CODE, cu_jit_position_independent_code); *)
        (CU_JIT_NUM_OPTIONS, cu_jit_num_options);
      ]

  let cu_target_compute_30 = T.constant "CU_TARGET_COMPUTE_30" T.int64_t
  let cu_target_compute_32 = T.constant "CU_TARGET_COMPUTE_32" T.int64_t
  let cu_target_compute_35 = T.constant "CU_TARGET_COMPUTE_35" T.int64_t
  let cu_target_compute_37 = T.constant "CU_TARGET_COMPUTE_37" T.int64_t
  let cu_target_compute_50 = T.constant "CU_TARGET_COMPUTE_50" T.int64_t
  let cu_target_compute_52 = T.constant "CU_TARGET_COMPUTE_52" T.int64_t
  let cu_target_compute_53 = T.constant "CU_TARGET_COMPUTE_53" T.int64_t
  let cu_target_compute_60 = T.constant "CU_TARGET_COMPUTE_60" T.int64_t
  let cu_target_compute_61 = T.constant "CU_TARGET_COMPUTE_61" T.int64_t
  let cu_target_compute_62 = T.constant "CU_TARGET_COMPUTE_62" T.int64_t
  let cu_target_compute_70 = T.constant "CU_TARGET_COMPUTE_70" T.int64_t
  let cu_target_compute_72 = T.constant "CU_TARGET_COMPUTE_72" T.int64_t
  let cu_target_compute_75 = T.constant "CU_TARGET_COMPUTE_75" T.int64_t
  let cu_target_compute_80 = T.constant "CU_TARGET_COMPUTE_80" T.int64_t
  let cu_target_compute_86 = T.constant "CU_TARGET_COMPUTE_86" T.int64_t
  (* let cu_target_compute_87 = T.constant "CU_TARGET_COMPUTE_87" T.int64_t
     let cu_target_compute_89 = T.constant "CU_TARGET_COMPUTE_89" T.int64_t
     let cu_target_compute_90 = T.constant "CU_TARGET_COMPUTE_90" T.int64_t
     let cu_target_compute_90a = T.constant "CU_TARGET_COMPUTE_90A" T.int64_t *)

  let cu_jit_target =
    T.enum ~typedef:true
      ~unexpected:(fun error_code -> CU_TARGET_UNCATEGORIZED error_code)
      "CUjit_target"
      [
        (CU_TARGET_COMPUTE_30, cu_target_compute_30);
        (CU_TARGET_COMPUTE_32, cu_target_compute_32);
        (CU_TARGET_COMPUTE_35, cu_target_compute_35);
        (CU_TARGET_COMPUTE_37, cu_target_compute_37);
        (CU_TARGET_COMPUTE_50, cu_target_compute_50);
        (CU_TARGET_COMPUTE_52, cu_target_compute_52);
        (CU_TARGET_COMPUTE_53, cu_target_compute_53);
        (CU_TARGET_COMPUTE_60, cu_target_compute_60);
        (CU_TARGET_COMPUTE_61, cu_target_compute_61);
        (CU_TARGET_COMPUTE_62, cu_target_compute_62);
        (CU_TARGET_COMPUTE_70, cu_target_compute_70);
        (CU_TARGET_COMPUTE_72, cu_target_compute_72);
        (CU_TARGET_COMPUTE_75, cu_target_compute_75);
        (CU_TARGET_COMPUTE_80, cu_target_compute_80);
        (CU_TARGET_COMPUTE_86, cu_target_compute_86);
        (* (CU_TARGET_COMPUTE_87, cu_target_compute_87);
           (CU_TARGET_COMPUTE_89, cu_target_compute_89);
           (CU_TARGET_COMPUTE_90, cu_target_compute_90);
           (CU_TARGET_COMPUTE_90A, cu_target_compute_90a); *)
      ]

  let cu_prefer_ptx = T.constant "CU_PREFER_PTX" T.int64_t
  let cu_prefer_binary = T.constant "CU_PREFER_BINARY" T.int64_t

  let cu_jit_fallback =
    T.enum ~typedef:true
      ~unexpected:(fun error_code -> CU_PREFER_UNCATEGORIZED error_code)
      "CUjit_fallback"
      [ (CU_PREFER_PTX, cu_prefer_ptx); (CU_PREFER_BINARY, cu_prefer_binary) ]

  let cu_jit_cache_option_none = T.constant "CU_JIT_CACHE_OPTION_NONE" T.int64_t
  let cu_jit_cache_option_cg = T.constant "CU_JIT_CACHE_OPTION_CG" T.int64_t
  let cu_jit_cache_option_ca = T.constant "CU_JIT_CACHE_OPTION_CA" T.int64_t

  let cu_jit_cache_mode =
    T.enum ~typedef:true
      ~unexpected:(fun error_code -> CU_JIT_CACHE_OPTION_UNCATEGORIZED error_code)
      "CUjit_cacheMode"
      [
        (CU_JIT_CACHE_OPTION_NONE, cu_jit_cache_option_none);
        (CU_JIT_CACHE_OPTION_CG, cu_jit_cache_option_cg);
        (CU_JIT_CACHE_OPTION_CA, cu_jit_cache_option_ca);
      ]

  let cu_device_attribute_max_threads_per_block =
    T.constant "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK" T.int64_t

  let cu_device_attribute_max_block_dim_x = T.constant "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X" T.int64_t
  let cu_device_attribute_max_block_dim_y = T.constant "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y" T.int64_t
  let cu_device_attribute_max_block_dim_z = T.constant "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z" T.int64_t
  let cu_device_attribute_max_grid_dim_x = T.constant "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X" T.int64_t
  let cu_device_attribute_max_grid_dim_y = T.constant "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y" T.int64_t
  let cu_device_attribute_max_grid_dim_z = T.constant "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z" T.int64_t

  let cu_device_attribute_max_shared_memory_per_block =
    T.constant "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK" T.int64_t

  let cu_device_attribute_shared_memory_per_block =
    T.constant "CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK" T.int64_t

  let cu_device_attribute_total_constant_memory =
    T.constant "CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY" T.int64_t

  let cu_device_attribute_warp_size = T.constant "CU_DEVICE_ATTRIBUTE_WARP_SIZE" T.int64_t
  let cu_device_attribute_max_pitch = T.constant "CU_DEVICE_ATTRIBUTE_MAX_PITCH" T.int64_t

  let cu_device_attribute_max_registers_per_block =
    T.constant "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK" T.int64_t

  let cu_device_attribute_clock_rate = T.constant "CU_DEVICE_ATTRIBUTE_CLOCK_RATE" T.int64_t
  let cu_device_attribute_texture_alignment = T.constant "CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT" T.int64_t
  let cu_device_attribute_gpu_overlap = T.constant "CU_DEVICE_ATTRIBUTE_GPU_OVERLAP" T.int64_t

  let cu_device_attribute_multiprocessor_count =
    T.constant "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT" T.int64_t

  let cu_device_attribute_kernel_exec_timeout = T.constant "CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT" T.int64_t
  let cu_device_attribute_integrated = T.constant "CU_DEVICE_ATTRIBUTE_INTEGRATED" T.int64_t
  let cu_device_attribute_can_map_host_memory = T.constant "CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY" T.int64_t
  let cu_device_attribute_compute_mode = T.constant "CU_DEVICE_ATTRIBUTE_COMPUTE_MODE" T.int64_t

  let cu_device_attribute_maximum_texture1d_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH" T.int64_t

  let cu_device_attribute_maximum_texture2d_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH" T.int64_t

  let cu_device_attribute_maximum_texture2d_height =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT" T.int64_t

  let cu_device_attribute_maximum_texture3d_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH" T.int64_t

  let cu_device_attribute_maximum_texture3d_height =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT" T.int64_t

  let cu_device_attribute_maximum_texture3d_depth =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH" T.int64_t

  let cu_device_attribute_maximum_texture2d_layered_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH" T.int64_t

  let cu_device_attribute_maximum_texture2d_layered_height =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT" T.int64_t

  let cu_device_attribute_maximum_texture2d_layered_layers =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS" T.int64_t

  let cu_device_attribute_maximum_texture2d_array_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH" T.int64_t

  let cu_device_attribute_maximum_texture2d_array_height =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT" T.int64_t

  let cu_device_attribute_maximum_texture2d_array_numslices =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES" T.int64_t

  let cu_device_attribute_surface_alignment = T.constant "CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT" T.int64_t
  let cu_device_attribute_concurrent_kernels = T.constant "CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS" T.int64_t
  let cu_device_attribute_ecc_enabled = T.constant "CU_DEVICE_ATTRIBUTE_ECC_ENABLED" T.int64_t
  let cu_device_attribute_pci_bus_id = T.constant "CU_DEVICE_ATTRIBUTE_PCI_BUS_ID" T.int64_t
  let cu_device_attribute_pci_device_id = T.constant "CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID" T.int64_t
  let cu_device_attribute_tcc_driver = T.constant "CU_DEVICE_ATTRIBUTE_TCC_DRIVER" T.int64_t
  let cu_device_attribute_memory_clock_rate = T.constant "CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE" T.int64_t

  let cu_device_attribute_global_memory_bus_width =
    T.constant "CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH" T.int64_t

  let cu_device_attribute_l2_cache_size = T.constant "CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE" T.int64_t

  let cu_device_attribute_max_threads_per_multiprocessor =
    T.constant "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR" T.int64_t

  let cu_device_attribute_async_engine_count = T.constant "CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT" T.int64_t
  (* let cu_device_attribute_unified_addressing = T.constant "CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING" T.int64_t *)

  let cu_device_attribute_maximum_texture1d_layered_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH" T.int64_t

  let cu_device_attribute_maximum_texture1d_layered_layers =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS" T.int64_t

  let cu_device_attribute_can_tex2d_gather = T.constant "CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER" T.int64_t

  let cu_device_attribute_maximum_texture2d_gather_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH" T.int64_t

  let cu_device_attribute_maximum_texture2d_gather_height =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT" T.int64_t

  let cu_device_attribute_maximum_texture3d_width_alternate =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE" T.int64_t

  let cu_device_attribute_maximum_texture3d_height_alternate =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE" T.int64_t

  let cu_device_attribute_maximum_texture3d_depth_alternate =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE" T.int64_t

  let cu_device_attribute_pci_domain_id = T.constant "CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID" T.int64_t

  let cu_device_attribute_texture_pitch_alignment =
    T.constant "CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT" T.int64_t

  let cu_device_attribute_maximum_texturecubemap_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH" T.int64_t

  let cu_device_attribute_maximum_texturecubemap_layered_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH" T.int64_t

  let cu_device_attribute_maximum_texturecubemap_layered_layers =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS" T.int64_t

  let cu_device_attribute_maximum_surface1d_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH" T.int64_t

  let cu_device_attribute_maximum_surface2d_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH" T.int64_t

  let cu_device_attribute_maximum_surface2d_height =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT" T.int64_t

  let cu_device_attribute_maximum_surface3d_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH" T.int64_t

  let cu_device_attribute_maximum_surface3d_height =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT" T.int64_t

  let cu_device_attribute_maximum_surface3d_depth =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH" T.int64_t

  let cu_device_attribute_maximum_surface1d_layered_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH" T.int64_t

  let cu_device_attribute_maximum_surface1d_layered_layers =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS" T.int64_t

  let cu_device_attribute_maximum_surface2d_layered_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH" T.int64_t

  let cu_device_attribute_maximum_surface2d_layered_height =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT" T.int64_t

  let cu_device_attribute_maximum_surface2d_layered_layers =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS" T.int64_t

  let cu_device_attribute_maximum_surfacecubemap_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH" T.int64_t

  let cu_device_attribute_maximum_surfacecubemap_layered_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH" T.int64_t

  let cu_device_attribute_maximum_surfacecubemap_layered_layers =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS" T.int64_t

  let cu_device_attribute_maximum_texture1d_linear_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH" T.int64_t

  let cu_device_attribute_maximum_texture2d_linear_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH" T.int64_t

  let cu_device_attribute_maximum_texture2d_linear_height =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT" T.int64_t

  let cu_device_attribute_maximum_texture2d_linear_pitch =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH" T.int64_t

  let cu_device_attribute_maximum_texture2d_mipmapped_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH" T.int64_t

  let cu_device_attribute_maximum_texture2d_mipmapped_height =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT" T.int64_t

  let cu_device_attribute_compute_capability_major =
    T.constant "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR" T.int64_t

  let cu_device_attribute_compute_capability_minor =
    T.constant "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR" T.int64_t

  let cu_device_attribute_maximum_texture1d_mipmapped_width =
    T.constant "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH" T.int64_t

  let cu_device_attribute_stream_priorities_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED" T.int64_t

  let cu_device_attribute_global_l1_cache_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED" T.int64_t

  let cu_device_attribute_local_l1_cache_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED" T.int64_t

  let cu_device_attribute_max_shared_memory_per_multiprocessor =
    T.constant "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR" T.int64_t

  let cu_device_attribute_max_registers_per_multiprocessor =
    T.constant "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR" T.int64_t

  let cu_device_attribute_managed_memory = T.constant "CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY" T.int64_t
  let cu_device_attribute_multi_gpu_board = T.constant "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD" T.int64_t

  let cu_device_attribute_multi_gpu_board_group_id =
    T.constant "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID" T.int64_t

  let cu_device_attribute_host_native_atomic_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED" T.int64_t

  let cu_device_attribute_single_to_double_precision_perf_ratio =
    T.constant "CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO" T.int64_t

  let cu_device_attribute_pageable_memory_access =
    T.constant "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS" T.int64_t

  let cu_device_attribute_concurrent_managed_access =
    T.constant "CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS" T.int64_t

  let cu_device_attribute_compute_preemption_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED" T.int64_t

  let cu_device_attribute_can_use_host_pointer_for_registered_mem =
    T.constant "CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM" T.int64_t

  let cu_device_attribute_cooperative_launch = T.constant "CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH" T.int64_t

  let cu_device_attribute_cooperative_multi_device_launch =
    T.constant "CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH" T.int64_t

  let cu_device_attribute_max_shared_memory_per_block_optin =
    T.constant "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN" T.int64_t

  let cu_device_attribute_can_flush_remote_writes =
    T.constant "CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES" T.int64_t

  let cu_device_attribute_host_register_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED" T.int64_t

  let cu_device_attribute_pageable_memory_access_uses_host_page_tables =
    T.constant "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES" T.int64_t

  let cu_device_attribute_direct_managed_mem_access_from_host =
    T.constant "CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST" T.int64_t

  let cu_device_attribute_virtual_address_management_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED" T.int64_t

  let cu_device_attribute_virtual_memory_management_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED" T.int64_t

  let cu_device_attribute_handle_type_posix_file_descriptor_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED" T.int64_t

  let cu_device_attribute_handle_type_win32_handle_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED" T.int64_t

  let cu_device_attribute_handle_type_win32_kmt_handle_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED" T.int64_t

  let cu_device_attribute_max_blocks_per_multiprocessor =
    T.constant "CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR" T.int64_t

  let cu_device_attribute_generic_compression_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED" T.int64_t

  let cu_device_attribute_max_persisting_l2_cache_size =
    T.constant "CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE" T.int64_t

  let cu_device_attribute_max_access_policy_window_size =
    T.constant "CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE" T.int64_t

  let cu_device_attribute_gpu_direct_rdma_with_cuda_vmm_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED" T.int64_t

  let cu_device_attribute_reserved_shared_memory_per_block =
    T.constant "CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK" T.int64_t

  let cu_device_attribute_sparse_cuda_array_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED" T.int64_t

  let cu_device_attribute_read_only_host_register_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED" T.int64_t

  let cu_device_attribute_timeline_semaphore_interop_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED" T.int64_t

  (* let cu_device_attribute_memory_pools_supported =
     T.constant "CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED" T.int64_t *)

  let cu_device_attribute_gpu_direct_rdma_supported =
    T.constant "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED" T.int64_t

  let cu_device_attribute_gpu_direct_rdma_flush_writes_options =
    T.constant "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS" T.int64_t

  let cu_device_attribute_gpu_direct_rdma_writes_ordering =
    T.constant "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING" T.int64_t

  let cu_device_attribute_mempool_supported_handle_types =
    T.constant "CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES" T.int64_t

  (* let cu_device_attribute_cluster_launch = T.constant "CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH" T.int64_t *)

  (* let cu_device_attribute_deferred_mapping_cuda_array_supported =
     T.constant "CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED" T.int64_t *)

  let cu_device_attribute_can_use_64_bit_stream_mem_ops =
    T.constant "CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS" T.int64_t

  let cu_device_attribute_can_use_stream_wait_value_nor =
    T.constant "CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR" T.int64_t

  (* let cu_device_attribute_dma_buf_supported = T.constant "CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED" T.int64_t *)
  (* let cu_device_attribute_ipc_event_supported = T.constant "CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED" T.int64_t *)

  (* let cu_device_attribute_mem_sync_domain_count =
     T.constant "CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT" T.int64_t *)

  (* let cu_device_attribute_tensor_map_access_supported =
     T.constant "CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED" T.int64_t *)

  (* let cu_device_attribute_unified_function_pointers =
     T.constant "CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS" T.int64_t *)

  (* let cu_device_attribute_multicast_supported = T.constant "CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED" T.int64_t *)
  let cu_device_attribute_max = T.constant "CU_DEVICE_ATTRIBUTE_MAX" T.int64_t

  let cu_device_attribute =
    T.enum ~typedef:true
      ~unexpected:(fun error_code -> CU_DEVICE_ATTRIBUTE_UNCATEGORIZED error_code)
      "CUdevice_attribute"
      [
        (CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cu_device_attribute_max_threads_per_block);
        (CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, cu_device_attribute_max_block_dim_x);
        (CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, cu_device_attribute_max_block_dim_y);
        (CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, cu_device_attribute_max_block_dim_z);
        (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, cu_device_attribute_max_grid_dim_x);
        (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, cu_device_attribute_max_grid_dim_y);
        (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, cu_device_attribute_max_grid_dim_z);
        (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cu_device_attribute_max_shared_memory_per_block);
        (CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, cu_device_attribute_total_constant_memory);
        (CU_DEVICE_ATTRIBUTE_WARP_SIZE, cu_device_attribute_warp_size);
        (CU_DEVICE_ATTRIBUTE_MAX_PITCH, cu_device_attribute_max_pitch);
        (CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, cu_device_attribute_max_registers_per_block);
        (CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cu_device_attribute_clock_rate);
        (CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, cu_device_attribute_texture_alignment);
        (CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, cu_device_attribute_gpu_overlap);
        (CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_device_attribute_multiprocessor_count);
        (CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, cu_device_attribute_kernel_exec_timeout);
        (CU_DEVICE_ATTRIBUTE_INTEGRATED, cu_device_attribute_integrated);
        (CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, cu_device_attribute_can_map_host_memory);
        (CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cu_device_attribute_compute_mode);
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, cu_device_attribute_maximum_texture1d_width);
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, cu_device_attribute_maximum_texture2d_width);
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, cu_device_attribute_maximum_texture2d_height);
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, cu_device_attribute_maximum_texture3d_width);
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, cu_device_attribute_maximum_texture3d_height);
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, cu_device_attribute_maximum_texture3d_depth);
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
          cu_device_attribute_maximum_texture2d_layered_width );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
          cu_device_attribute_maximum_texture2d_layered_height );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
          cu_device_attribute_maximum_texture2d_layered_layers );
        (CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, cu_device_attribute_surface_alignment);
        (CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, cu_device_attribute_concurrent_kernels);
        (CU_DEVICE_ATTRIBUTE_ECC_ENABLED, cu_device_attribute_ecc_enabled);
        (CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, cu_device_attribute_pci_bus_id);
        (CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, cu_device_attribute_pci_device_id);
        (CU_DEVICE_ATTRIBUTE_TCC_DRIVER, cu_device_attribute_tcc_driver);
        (CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, cu_device_attribute_memory_clock_rate);
        (CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, cu_device_attribute_global_memory_bus_width);
        (CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, cu_device_attribute_l2_cache_size);
        ( CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
          cu_device_attribute_max_threads_per_multiprocessor );
        (CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, cu_device_attribute_async_engine_count);
        (* (CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cu_device_attribute_unified_addressing); *)
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
          cu_device_attribute_maximum_texture1d_layered_width );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
          cu_device_attribute_maximum_texture1d_layered_layers );
        (CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER, cu_device_attribute_can_tex2d_gather);
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH,
          cu_device_attribute_maximum_texture2d_gather_width );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT,
          cu_device_attribute_maximum_texture2d_gather_height );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE,
          cu_device_attribute_maximum_texture3d_width_alternate );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE,
          cu_device_attribute_maximum_texture3d_height_alternate );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE,
          cu_device_attribute_maximum_texture3d_depth_alternate );
        (CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, cu_device_attribute_pci_domain_id);
        (CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, cu_device_attribute_texture_pitch_alignment);
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH, cu_device_attribute_maximum_texturecubemap_width);
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH,
          cu_device_attribute_maximum_texturecubemap_layered_width );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS,
          cu_device_attribute_maximum_texturecubemap_layered_layers );
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH, cu_device_attribute_maximum_surface1d_width);
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH, cu_device_attribute_maximum_surface2d_width);
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT, cu_device_attribute_maximum_surface2d_height);
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH, cu_device_attribute_maximum_surface3d_width);
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT, cu_device_attribute_maximum_surface3d_height);
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH, cu_device_attribute_maximum_surface3d_depth);
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH,
          cu_device_attribute_maximum_surface1d_layered_width );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS,
          cu_device_attribute_maximum_surface1d_layered_layers );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH,
          cu_device_attribute_maximum_surface2d_layered_width );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT,
          cu_device_attribute_maximum_surface2d_layered_height );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS,
          cu_device_attribute_maximum_surface2d_layered_layers );
        (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH, cu_device_attribute_maximum_surfacecubemap_width);
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH,
          cu_device_attribute_maximum_surfacecubemap_layered_width );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS,
          cu_device_attribute_maximum_surfacecubemap_layered_layers );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH,
          cu_device_attribute_maximum_texture1d_linear_width );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
          cu_device_attribute_maximum_texture2d_linear_width );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
          cu_device_attribute_maximum_texture2d_linear_height );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH,
          cu_device_attribute_maximum_texture2d_linear_pitch );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH,
          cu_device_attribute_maximum_texture2d_mipmapped_width );
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT,
          cu_device_attribute_maximum_texture2d_mipmapped_height );
        (CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device_attribute_compute_capability_major);
        (CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device_attribute_compute_capability_minor);
        ( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH,
          cu_device_attribute_maximum_texture1d_mipmapped_width );
        (CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, cu_device_attribute_stream_priorities_supported);
        (CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, cu_device_attribute_global_l1_cache_supported);
        (CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED, cu_device_attribute_local_l1_cache_supported);
        ( CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
          cu_device_attribute_max_shared_memory_per_multiprocessor );
        ( CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
          cu_device_attribute_max_registers_per_multiprocessor );
        (CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, cu_device_attribute_managed_memory);
        (CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, cu_device_attribute_multi_gpu_board);
        (CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, cu_device_attribute_multi_gpu_board_group_id);
        (CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, cu_device_attribute_host_native_atomic_supported);
        ( CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO,
          cu_device_attribute_single_to_double_precision_perf_ratio );
        (CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, cu_device_attribute_pageable_memory_access);
        (CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, cu_device_attribute_concurrent_managed_access);
        (CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, cu_device_attribute_compute_preemption_supported);
        ( CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM,
          cu_device_attribute_can_use_host_pointer_for_registered_mem );
        (* (CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1, cu_device_attribute_can_use_stream_mem_ops_v1); *)
        (* ( CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1,
           cu_device_attribute_can_use_64_bit_stream_mem_ops_v1 ); *)
        (* ( CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1,
           cu_device_attribute_can_use_stream_wait_value_nor_v1 ); *)
        (CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, cu_device_attribute_cooperative_launch);
        ( CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH,
          cu_device_attribute_cooperative_multi_device_launch );
        ( CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
          cu_device_attribute_max_shared_memory_per_block_optin );
        (CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, cu_device_attribute_can_flush_remote_writes);
        (CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED, cu_device_attribute_host_register_supported);
        ( CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
          cu_device_attribute_pageable_memory_access_uses_host_page_tables );
        ( CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST,
          cu_device_attribute_direct_managed_mem_access_from_host );
        ( CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
          cu_device_attribute_virtual_memory_management_supported );
        ( CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
          cu_device_attribute_handle_type_posix_file_descriptor_supported );
        ( CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED,
          cu_device_attribute_handle_type_win32_handle_supported );
        ( CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED,
          cu_device_attribute_handle_type_win32_kmt_handle_supported );
        (CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, cu_device_attribute_max_blocks_per_multiprocessor);
        (CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, cu_device_attribute_generic_compression_supported);
        (CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, cu_device_attribute_max_persisting_l2_cache_size);
        (CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE, cu_device_attribute_max_access_policy_window_size);
        ( CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
          cu_device_attribute_gpu_direct_rdma_with_cuda_vmm_supported );
        ( CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK,
          cu_device_attribute_reserved_shared_memory_per_block );
        (CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED, cu_device_attribute_sparse_cuda_array_supported);
        ( CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED,
          cu_device_attribute_read_only_host_register_supported );
        ( CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED,
          cu_device_attribute_timeline_semaphore_interop_supported );
        (* (CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, cu_device_attribute_memory_pools_supported); *)
        (CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, cu_device_attribute_gpu_direct_rdma_supported);
        ( CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS,
          cu_device_attribute_gpu_direct_rdma_flush_writes_options );
        ( CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
          cu_device_attribute_gpu_direct_rdma_writes_ordering );
        ( CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES,
          cu_device_attribute_mempool_supported_handle_types );
        (* (CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH, cu_device_attribute_cluster_launch); *)
        (* ( CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED,
           cu_device_attribute_deferred_mapping_cuda_array_supported ); *)
        (CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, cu_device_attribute_can_use_64_bit_stream_mem_ops);
        (CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR, cu_device_attribute_can_use_stream_wait_value_nor);
        (* (CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cu_device_attribute_dma_buf_supported); *)
        (* (CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED, cu_device_attribute_ipc_event_supported); *)
        (* (CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT, cu_device_attribute_mem_sync_domain_count); *)
        (* (CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED, cu_device_attribute_tensor_map_access_supported); *)
        (* (CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS, cu_device_attribute_unified_function_pointers); *)
        (* (CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cu_device_attribute_multicast_supported); *)
        (CU_DEVICE_ATTRIBUTE_MAX, cu_device_attribute_max);
      ]

  let cu_computemode_default = T.constant "CU_COMPUTEMODE_DEFAULT" T.int64_t
  let cu_computemode_prohibited = T.constant "CU_COMPUTEMODE_PROHIBITED" T.int64_t
  let cu_computemode_exclusive_process = T.constant "CU_COMPUTEMODE_EXCLUSIVE_PROCESS" T.int64_t

  let cu_computemode =
    T.enum ~typedef:true
      ~unexpected:(fun error_code -> CU_COMPUTEMODE_UNCATEGORIZED error_code)
      "CUcomputemode"
      [
        (CU_COMPUTEMODE_DEFAULT, cu_computemode_default);
        (CU_COMPUTEMODE_PROHIBITED, cu_computemode_prohibited);
        (CU_COMPUTEMODE_EXCLUSIVE_PROCESS, cu_computemode_exclusive_process);
      ]

  let cu_flush_gpu_direct_rdma_writes_option_host =
    T.constant "CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST" T.int64_t

  let cu_flush_gpu_direct_rdma_writes_option_memops =
    T.constant "CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS" T.int64_t

  let cu_flush_GPU_direct_RDMA_writes_options =
    T.enum ~typedef:true
      ~unexpected:(fun error_code -> CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_UNCATEGORIZED error_code)
      "CUflushGPUDirectRDMAWritesOptions"
      [
        (CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST, cu_flush_gpu_direct_rdma_writes_option_host);
        (CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS, cu_flush_gpu_direct_rdma_writes_option_memops);
      ]

  let cu_limit_stack_size = T.constant "CU_LIMIT_STACK_SIZE" T.int64_t
  let cu_limit_printf_fifo_size = T.constant "CU_LIMIT_PRINTF_FIFO_SIZE" T.int64_t
  let cu_limit_malloc_heap_size = T.constant "CU_LIMIT_MALLOC_HEAP_SIZE" T.int64_t
  let cu_limit_dev_runtime_sync_depth = T.constant "CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH" T.int64_t

  let cu_limit_dev_runtime_pending_launch_count =
    T.constant "CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT" T.int64_t

  let cu_limit_max_l2_fetch_granularity = T.constant "CU_LIMIT_MAX_L2_FETCH_GRANULARITY" T.int64_t
  let cu_limit_persisting_l2_cache_size = T.constant "CU_LIMIT_PERSISTING_L2_CACHE_SIZE" T.int64_t
  let cu_limit_max = T.constant "CU_LIMIT_MAX" T.int64_t

  let cu_limit =
    T.enum ~typedef:true
      ~unexpected:(fun error_code -> CU_LIMIT_UNCATEGORIZED error_code)
      "CUlimit"
      [
        (CU_LIMIT_STACK_SIZE, cu_limit_stack_size);
        (CU_LIMIT_PRINTF_FIFO_SIZE, cu_limit_printf_fifo_size);
        (CU_LIMIT_MALLOC_HEAP_SIZE, cu_limit_malloc_heap_size);
        (CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH, cu_limit_dev_runtime_sync_depth);
        (CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT, cu_limit_dev_runtime_pending_launch_count);
        (CU_LIMIT_MAX_L2_FETCH_GRANULARITY, cu_limit_max_l2_fetch_granularity);
        (CU_LIMIT_PERSISTING_L2_CACHE_SIZE, cu_limit_persisting_l2_cache_size);
        (CU_LIMIT_MAX, cu_limit_max);
      ]
end
