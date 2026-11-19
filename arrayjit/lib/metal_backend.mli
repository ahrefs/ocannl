module Impl : sig
  include Ir.Backend_impl.Lowered_backend

  val storage_mode_of_pool : device -> pool_id:int -> Metal.Resource.StorageMode.t
  (** Storage mode of the slab backing [pool_id] on [device]. After gh-ocannl-344 every pool is
      [Shared]; this is retained so tests can assert the allocator did not regress to another mode,
      without leaking the backend pointer into a shared type. *)

  val routine_name_probe : unit -> Ir.C_syntax.routine_name_probe
  (** What [tools/vendor_idents.exe] measures this backend's routine-name table through
      (gh-ocannl-1338): the kernel prelude, [newLibraryWithSource:] on the first device under the
      options a link compiles with, and the routine-name mangling. *)
end

val host_capabilities : unit -> (string * string) list
(** The host-side capabilities this backend probes without opening a device (gh-ocannl-1038): the
    Metal math API the running runtime's compile options accept ([metal_math_api], the same query
    every compile makes, named by [Compiler_options.metal_math_api_name]), and the Metal Toolchain's
    [xcrun -sdk macosx metal --version] line or [absent] ([metal_toolchain], gh-ocannl-1828: the
    compiler a test's AIR census re-compiles a run's artifact with, which the backend itself never
    runs). {!Context.host_capabilities} collects them. *)
