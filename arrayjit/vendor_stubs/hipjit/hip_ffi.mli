(** The one piece of [hipjit.hip_ffi], [hipjit]'s ctypes-generated [Hip_ffi], that its [hip.mli]
    names: the type [Hip.Module.function_attribute] re-exports with a manifest. [hipjit.hip_ffi]
    installs no interface file to copy, so this is the stub's only hand-written interface. It is not
    free to drift: wherever [hipjit.hip_ffi] is installed, [hip_ffi_agreement.ml] (generated in this
    directory from this file) checks that the installed [Hip_ffi] satisfies this signature, on
    [runtest-hip_ffi] and on every build. *)

module Bindings_types : sig
  type hip_function_attribute =
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
end
