val storage_mode_for_memory_mode :
  Ir.Tnode.memory_mode option -> Metal.Resource.StorageMode.t
(** Maps a tnode's memory mode to the Metal storage mode used for its buffer. GPU-only modes
    ([Local], [Device_only], [On_device]) map to [Private]; every other mode (and the [None]
    default) maps to [Shared] because the CPU may need to access the buffer. *)

module Fresh () : Ir.Backend_impl.Lowered_backend
