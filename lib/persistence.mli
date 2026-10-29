(** {1 Tensor checkpoint persistence: save, load, and restore.}

    A checkpoint file is a 4-byte big-endian header length, an S-expression header of per-tensor
    metadata, then the binary payloads in little-endian native precision format, in the header's
    order.

    Payload offsets are relative to the first byte past the header, and are multiples of the
    header's declared [alignment] (gh-ocannl-467) -- as is that first byte itself, the header being
    space-padded to the boundary. So the offsets are absolute file alignments, which is what lets
    the payloads be mapped rather than copied (see {!load}). *)

val save :
  ctx:Context.t ->
  appending:bool ->
  ?alignment:int ->
  ?metadata:(string * string) list ->
  Ocannl_tensor.Tensor.tn_set ->
  string ->
  unit
(** [save ~ctx ~appending t_set path] writes tensor data to a checkpoint file.

    When [~appending:false], creates a fresh checkpoint (overwriting any existing file). When
    [~appending:true] and the file exists, replaces tensors with matching (namespace, id) pairs and
    keeps non-overlapping entries from the existing file.

    Each tensor's data is retrieved on demand from its device buffer in [ctx] via {!Context.to_host}
    (gh-ocannl-333). Raises if any tnode in [t_set] is not present in [ctx].

    The file is published through {!Utils.Atomic_file}: written to a staging sibling of [path] and
    renamed over it, so a failed save leaves the previous checkpoint intact and leaves no staging
    artifact, and -- because the rename replaces the directory entry rather than the inode --
    mappings taken by an earlier {!load} of the same path keep seeing the data they were mapped
    from. The staging name is unique per writer, so two processes checkpointing the same path do not
    stream into one file. A save also reclaims the staging files of THIS checkpoint left by an
    earlier save that was killed mid-stream — nothing else would, and an abandoned one is the size
    of the model.

    [?alignment] (default 32, GGUF's [general.alignment] default) is the boundary payload offsets
    are rounded up to. It buys SIMD-friendly data pointers; mapping works at any alignment.

    [?metadata] (default none) is a list of key-value pairs stored in the header, published in the
    same atomic write as the tensors and read back by {!metadata} (gh-ocannl-96). An appending save
    replaces the keys it names and keeps the file's others, as it does tensors. Raises
    [Invalid_argument] on a duplicate key. *)

val metadata : string -> (string * string) list
(** [metadata path] reads the key-value pairs a {!save} stored in the checkpoint's header, without
    reading any payload: empty for a checkpoint saved without them. *)

val load :
  ctx:Context.t ->
  ?prefix_namespace:string ->
  ?mmap:bool ->
  string ->
  Context.t * Ocannl_tensor.Tensor.tn_set
(** [load ~ctx ?prefix_namespace path] reads tensors from a checkpoint file, creates new tnodes,
    uploads their data into [ctx] via {!Context.from_host}, and returns the updated context together
    with the loaded set (gh-ocannl-333).

    [?mmap] overrides the [checkpoint_load_mmap] setting for this call: with mapping on (the
    default), a payload whose file bytes are already its host buffer's bytes is wrapped as a
    private, copy-on-write {!Unix.map_file} region instead of being decoded element by element into
    a fresh buffer (gh-ocannl-467). Two kinds of payload keep the decoding path: padded nodes, whose
    payload stores only the logical region and has to be scattered into the padded buffer, and
    payloads whose file offset is not a multiple of their element size, which a checkpoint written
    with a small [?alignment] can produce. The mapping outlives the call, so the values are the same
    either way but the pages are read from the file lazily. Which path each payload took is counted
    in {!Ir.Ndarray.ingestion_counts}, the one place that observes it for every reader.

    A later {!save} over the same path is safe while those mappings are live, on every platform
    (gh-ocannl-588): the rename succeeds and the mappings keep reading the file they were taken
    from, which is why this is no longer conditional on the host being POSIX.

    Raises if any loaded tensor's (namespace, id) pair clashes with an existing tnode in the
    registry. After loading, bumps the session ID floor so that subsequently created tensors get IDs
    strictly above any loaded ID that landed in the ambient namespace.

    [?prefix_namespace] (gh-ocannl-372) rewrites each loaded tensor's namespace to
    [prefix ^ "__" ^ file_namespace], preserving the file's internal namespace structure and the
    session ids (so [Embed_self_id] values are invariant under prefixing). The prefix must match
    [A-Za-z_][A-Za-z0-9_]*; [None] and [Some ""] keep the file namespaces as-is. Pre-namespace
    checkpoints that recorded the namespace as [""] are read as the default namespace [ocannl]. *)

val restore :
  ctx:Context.t ->
  ?mmap:bool ->
  ?check_metadata:((string * string) list -> unit) ->
  Ocannl_tensor.Tensor.tn_set ->
  string ->
  Context.t
(** [restore ~ctx t_set path] updates existing tensor device buffers from a checkpoint file,
    returning the updated context.

    For each tnode in [t_set], finds its data in the file by (namespace, id) — file entries with the
    legacy [""] namespace match the default namespace — reads it into a temporary host buffer (or
    maps it, see {!load}'s [?mmap]), and uploads it into the node's device buffer in [ctx] via
    {!Context.from_host} (gh-ocannl-333).

    [?check_metadata] is called with the header's {!save} metadata before any tensor is read or
    uploaded, from the same open file as the payloads, so a concurrent save over [path] cannot pair
    one checkpoint's metadata with another's tensors; an exception it raises aborts the restore with
    [ctx] untouched. Given, it is called even for an empty [t_set].

    Raises if:
    - A tensor in [t_set] is missing from the file
    - Precision or dimensions don't match between live tensor and file *)
