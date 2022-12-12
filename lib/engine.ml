open Base

type submodel = {
  toplevel_forward: (unit -> unit) Codelib.code;
  (** Only apply at the root, since otherwise some computation may be elided (incorrect results). *)
  toplevel_backprop: (unit -> unit) Codelib.code;
  (** Only apply at the root! Gradients propagate from the top and are only propagated once. Zeroes
      the gradients before propagating. *)
  forward_body: unit Codelib.code;
  init_values: unit Codelib.code;
  (** Initializes the values. Computed only once per model compilation. *)
  backprop_body: unit Codelib.code;
  zero_grads: unit Codelib.code;
  (** Initializes the backpropagation phase. Computed once per backpropagation. *)
  node_id: int;
  mutable processed: bool;
  (** `true` if `forward_body`/`backprop_body`/`zero_grads` were already included in a parent submodel. *)
  mutable debug_node: Node.t;
  (** This tracks the computation node as long as the model is not cross-compiled to a different process. *)
}

(* Note! The code relies on argument evaluation order. To lift the requirement, we would need: *)
(* type t = submodel Lazy.t *)

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
  (* The code needs to be included in the order it was computed! *)
  let forward_body =
    if m1.processed && m2.processed then op_body
    else if m1.processed then (.< .~(m2.forward_body); .~op_body >.)
    else if m2.processed then (.< .~(m1.forward_body); .~op_body >.)
    else (.< .~(m1.forward_body); .~(m2.forward_body); .~op_body >.) in
  let init_values = (.<
    .~(m1.init_values);
    .~(m2.init_values);
    let dims1 = Ndarray.dims .~n1v in
    let dims2 = Ndarray.dims .~n2v in
    assert (Array.equal (=) dims1 dims2);
    .~n.value <- Ndarray.create dims1;
    fun () -> .~forward_body
  >.) in
  let toplevel_forward = (.< .~init_values; fun () -> .~forward_body >.) in
  let nd = Codelib.genlet ~name:"addd" (.< .~n.grad >.) in
  let n1d = Codelib.genlet ~name:"add1d" (.< (Node.get n1_id).grad >.) in
  let n2d = Codelib.genlet ~name:"add2d" (.< (Node.get n2_id).grad >.) in
  let zero_body = (.< Ndarray.reset_zeros .~nd >.) in
(* The order of zeroing gradients is irrelevant and multiple zeroing is fine, but we avoid it
     and keep the backprop order for readability. *)
  let zero_grads =
    if m1.processed && m2.processed then zero_body
    else if m1.processed then (.< .~zero_body; .~(m2.zero_grads) >.)
    else if m2.processed then (.< .~zero_body; .~(m1.zero_grads) >.)
    else (.< .~zero_body; .~(m2.zero_grads); .~(m1.zero_grads) >.) in
  let back_body = (.<
    Ndarray.assign_add .~n1d .~n1d .~nd;
    Ndarray.assign_add .~n2d .~n2d .~nd
  >.) in
  (* The code needs to be included in the reverse order to which it was computed! *)
  let backprop_body =
     if m1.processed && m2.processed then back_body
     else if m1.processed then (.< .~back_body; .~(m2.backprop_body) >.)
     else if m2.processed then (.< .~back_body; .~(m1.backprop_body) >.)
     else (.< .~back_body; .~(m2.backprop_body); .~(m1.backprop_body) >.) in
  let toplevel_backprop = (.<
    let dims1 = Ndarray.dims .~n1v in
    let dims2 = Ndarray.dims .~n2v in
    assert (Array.equal (=) dims1 dims2);
    .~n.grad <- Ndarray.create dims1;
    fun () ->
      .~(m1.zero_grads);
      .~(m2.zero_grads);
      Ndarray.reset_ones .~nd;
      .~backprop_body
  >.) in
  m1.processed <- true; m2.processed <- true;
  {toplevel_forward; toplevel_backprop; forward_body; backprop_body; init_values; zero_grads;
   node_id; processed=false; debug_node}

let mul m1 m2 =
  let debug_node = Node.create ~label:"*" in
  let node_id = debug_node.id in
  let n1_id = m1.node_id in
  let n2_id = m2.node_id in
  let n = Codelib.genlet ~name:"muln" (.< Node.get node_id >.) in
  let nv = Codelib.genlet ~name:"mulv" (.< .~n.value >.) in
  let n1v = Codelib.genlet ~name:"mul1v" (.< (Node.get n1_id).value >.) in
  let n2v = Codelib.genlet ~name:"mul2v" (.< (Node.get n2_id).value >.) in
  let op_body = (.< Ndarray.assign_mul .~nv .~n1v .~n2v >.) in
  (* The code needs to be included in the order it was computed! *)
  let forward_body =
    if m1.processed && m2.processed then op_body
    else if m1.processed then (.< .~(m2.forward_body); .~op_body >.)
    else if m2.processed then (.< .~(m1.forward_body); .~op_body >.)
    else (.< .~(m1.forward_body); .~(m2.forward_body); .~op_body >.) in
  let init_values = (.<
    .~(m1.init_values);
    .~(m2.init_values);
    let dims1 = Ndarray.dims .~n1v in
    let dims2 = Ndarray.dims .~n2v in
    assert (Array.equal (=) dims1 dims2);
    .~n.value <- Ndarray.create dims1;
  >.) in
  let toplevel_forward = (.< .~init_values; fun () -> .~forward_body >.) in
  let nd = Codelib.genlet ~name:"muld" (.< .~n.grad >.) in
  let n1d = Codelib.genlet ~name:"mul1d" (.< (Node.get n1_id).grad >.) in
  let n2d = Codelib.genlet ~name:"mul2d" (.< (Node.get n2_id).grad >.) in
  let zero_body = (.< Ndarray.reset_zeros .~nd >.) in
  (* The order of zeroing gradients is irrelevant and multiple zeroing is fine, but we avoid it
       and keep the backprop order for readability. *)
  let zero_grads =
    if m1.processed && m2.processed then zero_body
    else if m1.processed then (.< .~zero_body; .~(m2.zero_grads) >.)
    else if m2.processed then (.< .~zero_body; .~(m1.zero_grads) >.)
    else (.< .~zero_body; .~(m2.zero_grads); .~(m1.zero_grads) >.) in
  let back_body = (.<
    Ndarray.assign_add .~n1d .~n1d (Ndarray.mul .~nd .~n2v);
    Ndarray.assign_add .~n2d .~n2d (Ndarray.mul .~nd .~n1v)
  >.) in
  (* The code needs to be included in the reverse order to which it was computed! *)
  let backprop_body =
    if m1.processed && m2.processed then back_body
    else if m1.processed then (.< .~back_body; .~(m2.backprop_body) >.)
    else if m2.processed then (.< .~back_body; .~(m1.backprop_body) >.)
    else (.< .~back_body; .~(m2.backprop_body); .~(m1.backprop_body) >.) in
  let toplevel_backprop = (.<
    let dims1 = Ndarray.dims .~n1v in
    let dims2 = Ndarray.dims .~n2v in
    assert (Array.equal (=) dims1 dims2);
    .~n.grad <- Ndarray.create dims1;
    fun () ->
      .~(m1.zero_grads);
      .~(m2.zero_grads);
      Ndarray.reset_ones .~nd;
      .~backprop_body
  >.) in
  m1.processed <- true; m2.processed <- true;
  {toplevel_forward; toplevel_backprop; forward_body; backprop_body; init_values; zero_grads;
   node_id; processed=false; debug_node}

let relu m =
  let debug_node = Node.create ~label:"*" in
  let node_id = debug_node.id in
  let n1_id = m.node_id in
  let n = Codelib.genlet ~name:"relun" (.< Node.get node_id >.) in
  let nv = Codelib.genlet ~name:"reluv" (.< .~n.value >.) in
  let n1v = Codelib.genlet ~name:"relu1v" (.< (Node.get n1_id).value >.) in
  let op_body = (.< Ndarray.assign_relu .~nv .~n1v >.) in
  (* The code needs to be included in the order it was computed! *)
  let forward_body =
    if m.processed then op_body
    else (.< .~(m.forward_body); .~op_body >.) in
  let init_values = (.<
    .~(m.init_values);
    let dims = Ndarray.dims .~n1v in
    .~n.value <- Ndarray.create dims;
  >.) in
  let toplevel_forward = (.< .~init_values; fun () -> .~forward_body >.) in
  let nd = Codelib.genlet ~name:"relud" (.< .~n.grad >.) in
  let n1d = Codelib.genlet ~name:"relu1d" (.< (Node.get n1_id).grad >.) in
  let zero_body = (.< Ndarray.reset_zeros .~nd >.) in
  (* The order of zeroing gradients is irrelevant and multiple zeroing is fine, but we avoid it
       and keep the backprop order for readability. *)
  let zero_grads =
    if m.processed then zero_body
    else (.< .~zero_body; .~(m.zero_grads) >.) in
  let back_body = (.<
    Ndarray.assign_add .~n1d .~n1d (Ndarray.relu_gate .~nv .~nd)
  >.) in
  (* The code needs to be included in the reverse order to which it was computed! *)
  let backprop_body =
    if m.processed then back_body
    else (.< .~back_body; .~(m.backprop_body) >.) in
  let toplevel_backprop = (.<
    let dims1 = Ndarray.dims .~n1v in
    .~n.grad <- Ndarray.create dims1;
    fun () ->
      .~(m.zero_grads);
      Ndarray.reset_ones .~nd;
      .~backprop_body
  >.) in
  m.processed <- true;
  {toplevel_forward; toplevel_backprop; forward_body; backprop_body; init_values; zero_grads;
   node_id; processed=false; debug_node}

(* FIXME: be careful about where n1v etc. is created vs. where it's used. *)