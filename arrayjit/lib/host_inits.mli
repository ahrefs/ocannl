(** A weakly-owned association from a tensor node to the host buffer holding its initialization data
    (ndarray-backed literals and persistence-loaded nodes). After [gh-ocannl-333] the data is no
    longer stored on {!Tnode.t}; instead it is registered here at construction and uploaded into a
    backend context, on demand, at link time.

    The table is:
    - {b weak in the key}: an entry is reclaimed by the GC once its tensor node becomes unreachable,
      so constructing-but-never-compiling a literal does not leak its buffer;
    - {b read, not consumed}: linking a node into a context {e reads} (does not remove) the entry,
      so the same node can be initialized into multiple independent contexts / devices;
    - {b lazy in the value}: the buffer is shaped only when forced, i.e. after shape inference. *)

val register : Tnode.t -> Ndarray.t Lazy.t -> unit
(** Records the host initialization buffer for [tn]. A later registration for the same node
    overwrites the earlier one (the buffers describe the same node's data). *)

val find : Tnode.t -> Ndarray.t Lazy.t option
(** Returns the host initialization buffer for [tn], if any, without removing it. *)

val mem : Tnode.t -> bool
(** Whether [tn] has registered host initialization data. *)

val single_nan_free : Tnode.t -> bool
(** Whether the registered buffer holds NaN-free f32 data. The lazy classification belongs to that
    registration, so replacing it invalidates the result. Other precisions and absent data return
    false. *)
