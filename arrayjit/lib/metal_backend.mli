val storage_mode_for_memory_mode :
  Ir.Tnode.memory_mode option -> Metal.Resource.StorageMode.t
(** Maps a tnode's memory mode to the Metal storage mode used for its buffer. GPU-only modes
    ([Local], [Device_only], [On_device]) map to [Private]; every other mode (and the [None]
    default) maps to [Shared] because the CPU may need to access the buffer. *)

module Fresh () : sig
  include Ir.Backend_impl.Lowered_backend

  val storage_mode_of_pool : device -> pool_id:int -> Metal.Resource.StorageMode.t
  (** Storage mode (private vs. shared) of the slab backing [pool_id] on [device]. Storage mode is a
      per-pool property; this replaces the old concrete [buffer_ptr = Metal.Buffer.t] exposure so
      tests can still assert the allocator selected the right mode without the backend pointer
      leaking into a shared type. *)
end
