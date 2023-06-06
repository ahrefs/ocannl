(* open Ctypes *)

type cuda_result =
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

module Types (T : Ctypes.TYPE) = struct
 let cuda_result_success = T.constant "CUDA_SUCCESS" T.int64_t
 let cuda_result_error_invalid_value = T.constant "CUDA_ERROR_INVALID_VALUE" T.int64_t
 let cuda_result_error_out_of_memory = T.constant "CUDA_ERROR_OUT_OF_MEMORY" T.int64_t
 let cuda_result_error_not_initialized = T.constant "CUDA_ERROR_NOT_INITIALIZED" T.int64_t
 let cuda_result_error_deinitialized = T.constant "CUDA_ERROR_DEINITIALIZED" T.int64_t
 let cuda_result_error_profiler_disabled = T.constant "CUDA_ERROR_PROFILER_DISABLED" T.int64_t
 let cuda_result_error_profiler_not_initialized = T.constant "CUDA_ERROR_PROFILER_NOT_INITIALIZED" T.int64_t
 let cuda_result_error_profiler_already_started = T.constant "CUDA_ERROR_PROFILER_ALREADY_STARTED" T.int64_t
 let cuda_result_error_profiler_already_stopped = T.constant "CUDA_ERROR_PROFILER_ALREADY_STOPPED" T.int64_t
 let cuda_result_error_stub_library = T.constant "CUDA_ERROR_STUB_LIBRARY" T.int64_t
 (* let cuda_result_error_device_unavailable = T.constant "CUDA_ERROR_DEVICE_UNAVAILABLE" T.int64_t *)
 let cuda_result_error_no_device = T.constant "CUDA_ERROR_NO_DEVICE" T.int64_t
 let cuda_result_error_invalid_device = T.constant "CUDA_ERROR_INVALID_DEVICE" T.int64_t
 let cuda_result_error_device_not_licensed = T.constant "CUDA_ERROR_DEVICE_NOT_LICENSED" T.int64_t
 let cuda_result_error_invalid_image = T.constant "CUDA_ERROR_INVALID_IMAGE" T.int64_t
 let cuda_result_error_invalid_context = T.constant "CUDA_ERROR_INVALID_CONTEXT" T.int64_t
 let cuda_result_error_context_already_current = T.constant "CUDA_ERROR_CONTEXT_ALREADY_CURRENT" T.int64_t
 let cuda_result_error_map_failed = T.constant "CUDA_ERROR_MAP_FAILED" T.int64_t
 let cuda_result_error_unmap_failed = T.constant "CUDA_ERROR_UNMAP_FAILED" T.int64_t
 let cuda_result_error_array_is_mapped = T.constant "CUDA_ERROR_ARRAY_IS_MAPPED" T.int64_t
 let cuda_result_error_already_mapped = T.constant "CUDA_ERROR_ALREADY_MAPPED" T.int64_t
 let cuda_result_error_no_binary_for_gpu = T.constant "CUDA_ERROR_NO_BINARY_FOR_GPU" T.int64_t
 let cuda_result_error_already_acquired = T.constant "CUDA_ERROR_ALREADY_ACQUIRED" T.int64_t
 let cuda_result_error_not_mapped = T.constant "CUDA_ERROR_NOT_MAPPED" T.int64_t
 let cuda_result_error_not_mapped_as_array = T.constant "CUDA_ERROR_NOT_MAPPED_AS_ARRAY" T.int64_t
 let cuda_result_error_not_mapped_as_pointer = T.constant "CUDA_ERROR_NOT_MAPPED_AS_POINTER" T.int64_t
 let cuda_result_error_ecc_uncorrectable = T.constant "CUDA_ERROR_ECC_UNCORRECTABLE" T.int64_t
 let cuda_result_error_unsupported_limit = T.constant "CUDA_ERROR_UNSUPPORTED_LIMIT" T.int64_t
 let cuda_result_error_context_already_in_use = T.constant "CUDA_ERROR_CONTEXT_ALREADY_IN_USE" T.int64_t
 let cuda_result_error_peer_access_unsupported = T.constant "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED" T.int64_t
 let cuda_result_error_invalid_ptx = T.constant "CUDA_ERROR_INVALID_PTX" T.int64_t
 let cuda_result_error_invalid_graphics_context = T.constant "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT" T.int64_t
 let cuda_result_error_nvlink_uncorrectable = T.constant "CUDA_ERROR_NVLINK_UNCORRECTABLE" T.int64_t
 let cuda_result_error_jit_compiler_not_found = T.constant "CUDA_ERROR_JIT_COMPILER_NOT_FOUND" T.int64_t
 let cuda_result_error_unsupported_ptx_version = T.constant "CUDA_ERROR_UNSUPPORTED_PTX_VERSION" T.int64_t
 let cuda_result_error_jit_compilation_disabled = T.constant "CUDA_ERROR_JIT_COMPILATION_DISABLED" T.int64_t
 let cuda_result_error_unsupported_exec_affinity = T.constant "CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY" T.int64_t
 (* let cuda_result_error_unsupported_devside_sync = T.constant "CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC" T.int64_t *)
 let cuda_result_error_invalid_source = T.constant "CUDA_ERROR_INVALID_SOURCE" T.int64_t
 let cuda_result_error_file_not_found = T.constant "CUDA_ERROR_FILE_NOT_FOUND" T.int64_t
 let cuda_result_error_shared_object_symbol_not_found = T.constant "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND" T.int64_t
 let cuda_result_error_shared_object_init_failed = T.constant "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED" T.int64_t
 let cuda_result_error_operating_system = T.constant "CUDA_ERROR_OPERATING_SYSTEM" T.int64_t
 let cuda_result_error_invalid_handle = T.constant "CUDA_ERROR_INVALID_HANDLE" T.int64_t
 let cuda_result_error_illegal_state = T.constant "CUDA_ERROR_ILLEGAL_STATE" T.int64_t
 let cuda_result_error_not_found = T.constant "CUDA_ERROR_NOT_FOUND" T.int64_t
 let cuda_result_error_not_ready = T.constant "CUDA_ERROR_NOT_READY" T.int64_t
 let cuda_result_error_illegal_address = T.constant "CUDA_ERROR_ILLEGAL_ADDRESS" T.int64_t
 let cuda_result_error_launch_out_of_resources = T.constant "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES" T.int64_t
 let cuda_result_error_launch_timeout = T.constant "CUDA_ERROR_LAUNCH_TIMEOUT" T.int64_t
 let cuda_result_error_launch_incompatible_texturing = T.constant "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING" T.int64_t
 let cuda_result_error_peer_access_already_enabled = T.constant "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED" T.int64_t
 let cuda_result_error_peer_access_not_enabled = T.constant "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED" T.int64_t
 let cuda_result_error_primary_context_active = T.constant "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE" T.int64_t
 let cuda_result_error_context_is_destroyed = T.constant "CUDA_ERROR_CONTEXT_IS_DESTROYED" T.int64_t
 let cuda_result_error_assert = T.constant "CUDA_ERROR_ASSERT" T.int64_t
 let cuda_result_error_too_many_peers = T.constant "CUDA_ERROR_TOO_MANY_PEERS" T.int64_t
 let cuda_result_error_host_memory_already_registered = T.constant "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED" T.int64_t
 let cuda_result_error_host_memory_not_registered = T.constant "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED" T.int64_t
 let cuda_result_error_hardware_stack_error = T.constant "CUDA_ERROR_HARDWARE_STACK_ERROR" T.int64_t
 let cuda_result_error_illegal_instruction = T.constant "CUDA_ERROR_ILLEGAL_INSTRUCTION" T.int64_t
 let cuda_result_error_misaligned_address = T.constant "CUDA_ERROR_MISALIGNED_ADDRESS" T.int64_t
 let cuda_result_error_invalid_address_space = T.constant "CUDA_ERROR_INVALID_ADDRESS_SPACE" T.int64_t
 let cuda_result_error_invalid_pc = T.constant "CUDA_ERROR_INVALID_PC" T.int64_t
 let cuda_result_error_launch_failed = T.constant "CUDA_ERROR_LAUNCH_FAILED" T.int64_t
 let cuda_result_error_cooperative_launch_too_large = T.constant "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE" T.int64_t
 let cuda_result_error_not_permitted = T.constant "CUDA_ERROR_NOT_PERMITTED" T.int64_t
 let cuda_result_error_not_supported = T.constant "CUDA_ERROR_NOT_SUPPORTED" T.int64_t
 let cuda_result_error_system_not_ready = T.constant "CUDA_ERROR_SYSTEM_NOT_READY" T.int64_t
 let cuda_result_error_system_driver_mismatch = T.constant "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH" T.int64_t
 let cuda_result_error_compat_not_supported_on_device = T.constant "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE" T.int64_t
 let cuda_result_error_mps_connection_failed = T.constant "CUDA_ERROR_MPS_CONNECTION_FAILED" T.int64_t
 let cuda_result_error_mps_rpc_failure = T.constant "CUDA_ERROR_MPS_RPC_FAILURE" T.int64_t
 let cuda_result_error_mps_server_not_ready = T.constant "CUDA_ERROR_MPS_SERVER_NOT_READY" T.int64_t
 let cuda_result_error_mps_max_clients_reached = T.constant "CUDA_ERROR_MPS_MAX_CLIENTS_REACHED" T.int64_t
 let cuda_result_error_mps_max_connections_reached = T.constant "CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED" T.int64_t
 (* let cuda_result_error_mps_client_terminated = T.constant "CUDA_ERROR_MPS_CLIENT_TERMINATED" T.int64_t *)
 (* let cuda_result_error_cdp_not_supported = T.constant "CUDA_ERROR_CDP_NOT_SUPPORTED" T.int64_t *)
 (* let cuda_result_error_cdp_version_mismatch = T.constant "CUDA_ERROR_CDP_VERSION_MISMATCH" T.int64_t *)
 let cuda_result_error_stream_capture_unsupported = T.constant "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED" T.int64_t
 let cuda_result_error_stream_capture_invalidated = T.constant "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED" T.int64_t
 let cuda_result_error_stream_capture_merge = T.constant "CUDA_ERROR_STREAM_CAPTURE_MERGE" T.int64_t
 let cuda_result_error_stream_capture_unmatched = T.constant "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED" T.int64_t
 let cuda_result_error_stream_capture_unjoined = T.constant "CUDA_ERROR_STREAM_CAPTURE_UNJOINED" T.int64_t
 let cuda_result_error_stream_capture_isolation = T.constant "CUDA_ERROR_STREAM_CAPTURE_ISOLATION" T.int64_t
 let cuda_result_error_stream_capture_implicit = T.constant "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT" T.int64_t
 let cuda_result_error_captured_event = T.constant "CUDA_ERROR_CAPTURED_EVENT" T.int64_t
 let cuda_result_error_stream_capture_wrong_thread = T.constant "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD" T.int64_t
 let cuda_result_error_timeout = T.constant "CUDA_ERROR_TIMEOUT" T.int64_t
 let cuda_result_error_graph_exec_update_failure = T.constant "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE" T.int64_t
 let cuda_result_error_external_device = T.constant "CUDA_ERROR_EXTERNAL_DEVICE" T.int64_t
 (* let cuda_result_error_invalid_cluster_size = T.constant "CUDA_ERROR_INVALID_CLUSTER_SIZE" T.int64_t *)
 let cuda_result_error_unknown = T.constant "CUDA_ERROR_UNKNOWN" T.int64_t

 let cuda_result =
  T.enum ~typedef:true
    ~unexpected:(fun error_code -> CUDA_ERROR_UNCATEGORIZED error_code)
    "CUresult"
    [
      (CUDA_SUCCESS, cuda_result_success);
      (CUDA_ERROR_INVALID_VALUE, cuda_result_error_invalid_value);
      (CUDA_ERROR_OUT_OF_MEMORY, cuda_result_error_out_of_memory);
      (CUDA_ERROR_NOT_INITIALIZED, cuda_result_error_not_initialized);
      (CUDA_ERROR_DEINITIALIZED, cuda_result_error_deinitialized);
      (CUDA_ERROR_PROFILER_DISABLED, cuda_result_error_profiler_disabled);
      (CUDA_ERROR_PROFILER_NOT_INITIALIZED, cuda_result_error_profiler_not_initialized);
      (CUDA_ERROR_PROFILER_ALREADY_STARTED, cuda_result_error_profiler_already_started);
      (CUDA_ERROR_PROFILER_ALREADY_STOPPED, cuda_result_error_profiler_already_stopped);
      (CUDA_ERROR_STUB_LIBRARY, cuda_result_error_stub_library);
      (* (CUDA_ERROR_DEVICE_UNAVAILABLE, cuda_result_error_device_unavailable); *)
      (CUDA_ERROR_NO_DEVICE, cuda_result_error_no_device);
      (CUDA_ERROR_INVALID_DEVICE, cuda_result_error_invalid_device);
      (CUDA_ERROR_DEVICE_NOT_LICENSED, cuda_result_error_device_not_licensed);
      (CUDA_ERROR_INVALID_IMAGE, cuda_result_error_invalid_image);
      (CUDA_ERROR_INVALID_CONTEXT, cuda_result_error_invalid_context);
      (CUDA_ERROR_CONTEXT_ALREADY_CURRENT, cuda_result_error_context_already_current);
      (CUDA_ERROR_MAP_FAILED, cuda_result_error_map_failed);
      (CUDA_ERROR_UNMAP_FAILED, cuda_result_error_unmap_failed);
      (CUDA_ERROR_ARRAY_IS_MAPPED, cuda_result_error_array_is_mapped);
      (CUDA_ERROR_ALREADY_MAPPED, cuda_result_error_already_mapped);
      (CUDA_ERROR_NO_BINARY_FOR_GPU, cuda_result_error_no_binary_for_gpu);
      (CUDA_ERROR_ALREADY_ACQUIRED, cuda_result_error_already_acquired);
      (CUDA_ERROR_NOT_MAPPED, cuda_result_error_not_mapped);
      (CUDA_ERROR_NOT_MAPPED_AS_ARRAY, cuda_result_error_not_mapped_as_array);
      (CUDA_ERROR_NOT_MAPPED_AS_POINTER, cuda_result_error_not_mapped_as_pointer);
      (CUDA_ERROR_ECC_UNCORRECTABLE, cuda_result_error_ecc_uncorrectable);
      (CUDA_ERROR_UNSUPPORTED_LIMIT, cuda_result_error_unsupported_limit);
      (CUDA_ERROR_CONTEXT_ALREADY_IN_USE, cuda_result_error_context_already_in_use);
      (CUDA_ERROR_PEER_ACCESS_UNSUPPORTED, cuda_result_error_peer_access_unsupported);
      (CUDA_ERROR_INVALID_PTX, cuda_result_error_invalid_ptx);
      (CUDA_ERROR_INVALID_GRAPHICS_CONTEXT, cuda_result_error_invalid_graphics_context);
      (CUDA_ERROR_NVLINK_UNCORRECTABLE, cuda_result_error_nvlink_uncorrectable);
      (CUDA_ERROR_JIT_COMPILER_NOT_FOUND, cuda_result_error_jit_compiler_not_found);
      (CUDA_ERROR_UNSUPPORTED_PTX_VERSION, cuda_result_error_unsupported_ptx_version);
      (CUDA_ERROR_JIT_COMPILATION_DISABLED, cuda_result_error_jit_compilation_disabled);
      (CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY, cuda_result_error_unsupported_exec_affinity);
      (* (CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC, cuda_result_error_unsupported_devside_sync); *)
      (CUDA_ERROR_INVALID_SOURCE, cuda_result_error_invalid_source);
      (CUDA_ERROR_FILE_NOT_FOUND, cuda_result_error_file_not_found);
      (CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND, cuda_result_error_shared_object_symbol_not_found);
      (CUDA_ERROR_SHARED_OBJECT_INIT_FAILED, cuda_result_error_shared_object_init_failed);
      (CUDA_ERROR_OPERATING_SYSTEM, cuda_result_error_operating_system);
      (CUDA_ERROR_INVALID_HANDLE, cuda_result_error_invalid_handle);
      (CUDA_ERROR_ILLEGAL_STATE, cuda_result_error_illegal_state);
      (CUDA_ERROR_NOT_FOUND, cuda_result_error_not_found);
      (CUDA_ERROR_NOT_READY, cuda_result_error_not_ready);
      (CUDA_ERROR_ILLEGAL_ADDRESS, cuda_result_error_illegal_address);
      (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, cuda_result_error_launch_out_of_resources);
      (CUDA_ERROR_LAUNCH_TIMEOUT, cuda_result_error_launch_timeout);
      (CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, cuda_result_error_launch_incompatible_texturing);
      (CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED, cuda_result_error_peer_access_already_enabled);
      (CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, cuda_result_error_peer_access_not_enabled);
      (CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE, cuda_result_error_primary_context_active);
      (CUDA_ERROR_CONTEXT_IS_DESTROYED, cuda_result_error_context_is_destroyed);
      (CUDA_ERROR_ASSERT, cuda_result_error_assert);
      (CUDA_ERROR_TOO_MANY_PEERS, cuda_result_error_too_many_peers);
      (CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED, cuda_result_error_host_memory_already_registered);
      (CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, cuda_result_error_host_memory_not_registered);
      (CUDA_ERROR_HARDWARE_STACK_ERROR, cuda_result_error_hardware_stack_error);
      (CUDA_ERROR_ILLEGAL_INSTRUCTION, cuda_result_error_illegal_instruction);
      (CUDA_ERROR_MISALIGNED_ADDRESS, cuda_result_error_misaligned_address);
      (CUDA_ERROR_INVALID_ADDRESS_SPACE, cuda_result_error_invalid_address_space);
      (CUDA_ERROR_INVALID_PC, cuda_result_error_invalid_pc);
      (CUDA_ERROR_LAUNCH_FAILED, cuda_result_error_launch_failed);
      (CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE, cuda_result_error_cooperative_launch_too_large);
      (CUDA_ERROR_NOT_PERMITTED, cuda_result_error_not_permitted);
      (CUDA_ERROR_NOT_SUPPORTED, cuda_result_error_not_supported);
      (CUDA_ERROR_SYSTEM_NOT_READY, cuda_result_error_system_not_ready);
      (CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, cuda_result_error_system_driver_mismatch);
      (CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE, cuda_result_error_compat_not_supported_on_device);
      (CUDA_ERROR_MPS_CONNECTION_FAILED, cuda_result_error_mps_connection_failed);
      (CUDA_ERROR_MPS_RPC_FAILURE, cuda_result_error_mps_rpc_failure);
      (CUDA_ERROR_MPS_SERVER_NOT_READY, cuda_result_error_mps_server_not_ready);
      (CUDA_ERROR_MPS_MAX_CLIENTS_REACHED, cuda_result_error_mps_max_clients_reached);
      (CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED, cuda_result_error_mps_max_connections_reached);
      (* (CUDA_ERROR_MPS_CLIENT_TERMINATED, cuda_result_error_mps_client_terminated); *)
      (* (CUDA_ERROR_CDP_NOT_SUPPORTED, cuda_result_error_cdp_not_supported); *)
      (* (CUDA_ERROR_CDP_VERSION_MISMATCH, cuda_result_error_cdp_version_mismatch); *)
      (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, cuda_result_error_stream_capture_unsupported);
      (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED, cuda_result_error_stream_capture_invalidated);
      (CUDA_ERROR_STREAM_CAPTURE_MERGE, cuda_result_error_stream_capture_merge);
      (CUDA_ERROR_STREAM_CAPTURE_UNMATCHED, cuda_result_error_stream_capture_unmatched);
      (CUDA_ERROR_STREAM_CAPTURE_UNJOINED, cuda_result_error_stream_capture_unjoined);
      (CUDA_ERROR_STREAM_CAPTURE_ISOLATION, cuda_result_error_stream_capture_isolation);
      (CUDA_ERROR_STREAM_CAPTURE_IMPLICIT, cuda_result_error_stream_capture_implicit);
      (CUDA_ERROR_CAPTURED_EVENT, cuda_result_error_captured_event);
      (CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD, cuda_result_error_stream_capture_wrong_thread);
      (CUDA_ERROR_TIMEOUT, cuda_result_error_timeout);
      (CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE, cuda_result_error_graph_exec_update_failure);
      (CUDA_ERROR_EXTERNAL_DEVICE, cuda_result_error_external_device);
      (* (CUDA_ERROR_INVALID_CLUSTER_SIZE, cuda_result_error_invalid_cluster_size); *)
      (CUDA_ERROR_UNKNOWN, cuda_result_error_unknown);
    ]
end