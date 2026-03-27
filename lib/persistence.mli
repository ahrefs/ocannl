(** {1 Tensor checkpoint persistence: save, load, and restore.}

    Checkpoint files use an S-expression header for metadata followed by contiguous
    binary payloads in native precision format. *)

val save : appending:bool -> Ocannl_tensor.Tensor.tn_set -> string -> unit
(** [save ~appending t_set path] writes tensor data to a checkpoint file.

    When [~appending:false], creates a fresh checkpoint (overwriting any existing file).
    When [~appending:true] and the file exists, replaces tensors with matching IDs
    and keeps non-overlapping entries from the existing file.

    Raises if any tnode in [t_set] has no hosted array (e.g., virtual or local tnodes).
    Calls [Tnode.do_read] on each tnode to sync device-to-host before saving. *)

val load : ?prefix_namespace:string -> string -> Ocannl_tensor.Tensor.tn_set
(** [load ?prefix_namespace path] reads tensors from a checkpoint file, creates new
    tnodes, registers them, and returns the resulting set.

    Raises if any loaded tensor ID clashes with an existing tnode in the registry.
    After loading, bumps the session ID floor so that subsequently created tensors
    get IDs strictly above any loaded ID.

    [?prefix_namespace] is reserved for future namespace support (#372). Currently,
    only [None] or [Some ""] are accepted; any non-empty prefix raises an error. *)

val restore : Ocannl_tensor.Tensor.tn_set -> string -> unit
(** [restore t_set path] updates existing tensor buffers from a checkpoint file.

    For each tnode in [t_set], finds its data in the file by ID and overwrites
    the hosted buffer. After restoring, clears [prepare_read] (to prevent stale
    device-to-host transfers) and calls [do_write] (to mark all devices as stale
    so the next device access re-transfers from host).

    Raises if:
    - A tensor in [t_set] is missing from the file
    - Precision or dimensions don't match between live tensor and file *)
