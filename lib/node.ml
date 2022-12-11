
(** Reverse-mode autodiff. *)
(* open Trx *)

open Base

type data = Ndarray.t

(** 
 * Initial scope of the project: we assume there is only one loss function per model, i.e. one
 * root `node`, but that there is parameter tying, i.e. that the graph under the root is
 * an arbitrary DAG.
 *
 * Stages of life:
    1. Constructing models using the engine.
    2. Call `compile` on the model root once all the model with parameter tying is constructed.
    4. Inside the training loop:
      5. Modify in-place the `Ndarray`s of those `param` nodes that are inputs.
      6. Call `forward` on the loss/root.
      7. Log the loss.
      8. Call `backprop` on the loss.
      9. Update the parameters.
*)
(* We should use `genletv` to avoid double-counting in the backprop phase! *)

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
