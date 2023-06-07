open Ctypes

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

type cu_device = Cu_device of int

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

type cu_jit_fallback = CU_PREFER_PTX | CU_PREFER_BINARY | CU_PREFER_UNCATEGORIZED of int64

type cu_jit_cache_mode =
  | CU_JIT_CACHE_OPTION_NONE
  | CU_JIT_CACHE_OPTION_CG
  | CU_JIT_CACHE_OPTION_CA
  | CU_JIT_CACHE_OPTION_UNCATEGORIZED of int64

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
end
