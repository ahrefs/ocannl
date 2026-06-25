(** {1 Tensor checkpoint persistence: save, load, and restore.}

    Checkpoint files use an S-expression header for metadata followed by contiguous binary payloads
    in native precision format. *)

val save : ctx:Context.t -> appending:bool -> Ocannl_tensor.Tensor.tn_set -> string -> unit
(** [save ~ctx ~appending t_set path] writes tensor data to a checkpoint file.

    When [~appending:false], creates a fresh checkpoint (overwriting any existing file). When
    [~appending:true] and the file exists, replaces tensors with matching IDs and keeps
    non-overlapping entries from the existing file.

    Each tensor's data is retrieved on demand from its device buffer in [ctx] via {!Context.to_host}
    (gh-ocannl-333). Raises if any tnode in [t_set] is not present in [ctx]. *)

val load :
  ctx:Context.t -> ?prefix_namespace:string -> string -> Context.t * Ocannl_tensor.Tensor.tn_set
(** [load ~ctx ?prefix_namespace path] reads tensors from a checkpoint file, creates new tnodes,
    uploads their data into [ctx] via {!Context.from_host}, and returns the updated context together
    with the loaded set (gh-ocannl-333).

    Raises if any loaded tensor ID clashes with an existing tnode in the registry. After loading,
    bumps the session ID floor so that subsequently created tensors get IDs strictly above any
    loaded ID.

    [?prefix_namespace] is reserved for future namespace support (#372). Currently, only [None] or
    [Some ""] are accepted; any non-empty prefix raises an error. *)

val restore : ctx:Context.t -> Ocannl_tensor.Tensor.tn_set -> string -> Context.t
(** [restore ~ctx t_set path] updates existing tensor device buffers from a checkpoint file,
    returning the updated context.

    For each tnode in [t_set], finds its data in the file by ID, reads it into a temporary host
    buffer, and uploads it into the node's device buffer in [ctx] via {!Context.from_host}
    (gh-ocannl-333).

    Raises if:
    - A tensor in [t_set] is missing from the file
    - Precision or dimensions don't match between live tensor and file *)
