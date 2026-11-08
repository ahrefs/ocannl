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

module Init_table = Tn.Weak_table

type entry = { buffer : Nd.t Lazy.t; single_nan_free : bool Lazy.t }

let table : entry Init_table.t = Init_table.create 16

(** Records the host initialization buffer for [tn]. A later registration for the same node
    overwrites the earlier one (the buffers describe the same node's data). *)
let register tn buffer =
  let single_nan_free =
    lazy
      (match Lazy.force buffer with
      | Nd.Single_nd _ as data ->
          Nd.fold_as_float ~init:true ~f:(fun ok _ v -> ok && not (Float.is_nan v)) data
      | _ -> false)
  in
  Init_table.replace table tn { buffer; single_nan_free }

(** Returns the host initialization buffer for [tn], if any, without removing it. *)
let find tn = Option.map (Init_table.find_opt table tn) ~f:(fun entry -> entry.buffer)

(** Whether [tn] has registered host initialization data. *)
let mem tn = Init_table.mem table tn

(** Whether the registered buffer holds f32 values with no NaN. Memoized with the buffer, so a
    replacement registration invalidates the classification and weak ownership is retained. *)
let single_nan_free tn =
  Option.value_map (Init_table.find_opt table tn) ~default:false ~f:(fun entry ->
      Lazy.force entry.single_nan_free)
