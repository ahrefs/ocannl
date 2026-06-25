open Base
module Nd = Ndarray
module Tn = Tnode

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

module Init_table = Stdlib.Ephemeron.K1.Make (struct
  type t = Tn.t

  let equal = Tn.equal
  let hash = Tn.hash
end)

let table : Nd.t Lazy.t Init_table.t = Init_table.create 16

(** Records the host initialization buffer for [tn]. A later registration for the same node
    overwrites the earlier one (the buffers describe the same node's data). *)
let register tn buffer = Init_table.replace table tn buffer

(** Returns the host initialization buffer for [tn], if any, without removing it. *)
let find tn = Init_table.find_opt table tn

(** Whether [tn] has registered host initialization data. *)
let mem tn = Init_table.mem table tn
