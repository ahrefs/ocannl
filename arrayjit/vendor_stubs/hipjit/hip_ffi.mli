(** The one piece of [hipjit]'s ctypes-generated [Hip_ffi] that its [hip.mli] names: the type
    [Hip.Module.function_attribute] re-exports with a manifest. This is the stub's only hand-written
    interface, and it is not free to drift: the compiler checks [hip.mli]'s re-export against it,
    and [hip.mli] is the verbatim copy the agreement rule diffs against the installed one. *)

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
