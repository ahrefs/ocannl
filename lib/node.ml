(** `Node`: the object type, global state and utils which the `Formula` staged code uses. *)
open Base

type data = Ndarray.t

type t = {
  mutable value: data;
  mutable grad: data;
  label: string;
  id: int;
}

type state = {
  mutable unique_id: int;
  node_store: (int, t) Hashtbl.t;
  params: (string, t) Hashtbl.t;
  (** A subset of nodes that are parameters. Assumes unique param labels. *)
}

let global = {
  unique_id = 0;
  node_store = Hashtbl.create (module Int);
  params = Hashtbl.create (module String);
}
let get uid = Hashtbl.find_exn global.node_store uid

let create ~label =
  let node = {
    value=Ndarray.empty; grad=Ndarray.empty; label;
    id=let uid = global.unique_id in global.unique_id <- global.unique_id + 1; uid
  } in
  assert (phys_equal `Ok @@ Hashtbl.add global.node_store ~key:node.id ~data:node);
  node
