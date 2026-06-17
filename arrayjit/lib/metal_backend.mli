module Fresh () : sig
  include Ir.Backend_impl.Lowered_backend

  val storage_mode_of_pool : device -> pool_id:int -> Metal.Resource.StorageMode.t
  (** Storage mode of the slab backing [pool_id] on [device]. After gh-ocannl-344 every pool is
      [Shared]; this is retained so tests can assert the allocator did not regress to another mode,
      without leaking the backend pointer into a shared type. *)
end
