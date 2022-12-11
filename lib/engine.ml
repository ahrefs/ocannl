open Base

type submodel = {
  forward: (unit -> unit) Codelib.code;
  backprop: (unit -> unit) Codelib.code;
  forward_body: unit Codelib.code;
  backprop_body: unit Codelib.code;
  node_id: int;
  mutable forwarded: bool;
  (** `true` if `forward_body` was already included in a forward pass. *)
  mutable backwarded: bool;
  (** `true` if `backprop_body` was already included in a backward pass. *)
  mutable debug_node: Node.t;
  (** This tracks the computation node as long as the model is not cross-compiled to a different process. *)
}

(** We need the laziness specifically for constructing the backward pass code... *)
type t = submodel Lazy.t

(* Design choice: tensor dims are decided after code is constructed, but before it is compiled.
   I.e. code needs to be recompiled with `Runcode.run` when the dimensions change. *)
(* TODO: maybe propagate a label and use it as a prefix for `genlet`? *)
let add m1 m2 =
  let debug_node = Node.create ~label:"+" in
  let node_id = debug_node.id in
  let n1_id = m1.node_id in
  let n2_id = m2.node_id in
  let n = Codelib.genlet ~name:"addn" (.< Node.get node_id >.) in
  let nv = Codelib.genlet ~name:"addv" (.< .~n.value >.) in
  let n1v = Codelib.genlet ~name:"add1v" (.< (Node.get n1_id).value >.) in
  let n2v = Codelib.genlet ~name:"add2v" (.< (Node.get n2_id).value >.) in
  let op_body = (.< Ndarray.assign_add .~nv .~n1v .~n2v >.) in
  let forward_body =
    if m1.forwarded && m2.forwarded then op_body
    else if m1.forwarded then (.< .~(m2.forward_body); .~op_body >.)
    else if m2.forwarded then (.< .~(m1.forward_body); .~op_body >.)
    else (.< .~(m1.forward_body); .~(m2.forward_body); .~op_body >.) in
  m1.forwarded <- true; m2.forwarded <- true;
  let forward = (.<
    let dims1 = Ndarray.dims .~n1v in
    let dims2 = Ndarray.dims .~n2v in
    assert (Array.equal (=) dims1 dims2);
    .~n.value <- Ndarray.create dims1;
    fun () -> .~forward_body
   >.) in
   let nd = Codelib.genlet ~name:"addd" (.< .~n.grad >.) in
   let n1d = Codelib.genlet ~name:"add1d" (.< (Node.get n1_id).grad >.) in
   let n2d = Codelib.genlet ~name:"add2d" (.< (Node.get n2_id).grad >.) in
   let back_body = (.<
    Ndarray.assign_add .~n1d .~n1d .~nd;
    Ndarray.assign_add .~n2d .~n2d .~nd
   >.) in
   let backprop_body =
     if m1.backwarded && m2.backwarded then back_body
     else if m1.backwarded then (.< .~back_body; .~(m2.backprop_body) >.)
     else if m2.backwarded then (.< .~back_body; .~(m1.backprop_body) >.)
     else (.< .~back_body; .~(m1.backprop_body); .~(m2.backprop_body) >.) in
   m1.backwarded <- true; m2.backwarded <- true;
  let backprop = (.<
   fun () -> ()
  >.) in
  {forward; backprop; forward_body; backprop_body; node_id; forwarded=false; backwarded=false; debug_node}
