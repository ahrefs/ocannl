module Impl : sig
  include Ir.Backend_impl.Lowered_backend

  val storage_mode_of_pool : device -> pool_id:int -> Metal.Resource.StorageMode.t
  (** Storage mode of the slab backing [pool_id] on [device]. After gh-ocannl-344 every pool is
      [Shared]; this is retained so tests can assert the allocator did not regress to another mode,
      without leaking the backend pointer into a shared type. *)
end

val host_capabilities : unit -> (string * string) list
(** The host-side capability this backend probes without opening a device (gh-ocannl-1038): the
    Metal math API the running runtime's compile options accept, the same query every compile makes.
    {!Context.host_capabilities} collects it. *)
