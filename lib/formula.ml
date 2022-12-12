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
  init_grads: unit Codelib.code;
  (** Initializes the gradient data: typically, simply creates the ndarrays.
      Gradients are zeroed separately. *)
  backprop_body: unit Codelib.code;
  zero_grads: unit Codelib.code;
  (** Initializes the backpropagation phase. Computed once per backpropagation. *)
  node_id: int;
  mutable processed: bool;
  (** [true] if [forward_body]/[backprop_body]/[zero_grads] were already included in a parent submodel. *)
  mutable debug_node: Node.t;
  (** This tracks the computation node as long as the model is not cross-compiled to a different process. *)
}

(* The code relies on argument evaluation order. To lift the requirement, we could use
   [submodel Lazy.t], but that's an unnecessary obfuscation. *)
let l2r_comp_order =
  let l2r_ord = ref None in
  (fun () () ->
    match !l2r_ord with
    | Some b -> b
    | None -> assert false) (l2r_ord := Some false) (l2r_ord := Some true)

(* Design choice: tensor dims are decided after code is constructed, but before it is compiled.
   I.e. code needs to be recompiled with [Runcode.run] when the dimensions change. *)

(* TODO: maybe propagate a label and use it as a prefix for [genlet]? *)

let binop ~label ~name ~op_body ~grad_body m1 m2 =
  let debug_node = Node.create ~label in
  let node_id = debug_node.id in
  let n1_id = m1.node_id in
  let n2_id = m2.node_id in
  let n = Codelib.genlet ~name:(name^"n") (.< Node.get node_id >.) in
  let nv = Codelib.genlet ~name:(name^"v") (.< .~n.value >.) in
  let n1v = Codelib.genlet ~name:(name^"1v") (.< (Node.get n1_id).value >.) in
  let n2v = Codelib.genlet ~name:(name^"2v") (.< (Node.get n2_id).value >.) in
  let op_body = op_body ~nv ~n1v ~n2v in
  (* The code needs to be included in the order it was computed! *)
  let forward_body =
    if m1.processed && m2.processed then op_body
    else if m1.processed then (.< .~(m2.forward_body); .~op_body >.)
    else if m2.processed then (.< .~(m1.forward_body); .~op_body >.)
    else if l2r_comp_order then (.< .~(m1.forward_body); .~(m2.forward_body); .~op_body >.)
    else (.< .~(m2.forward_body); .~(m1.forward_body); .~op_body >.) in
  let init_values_body = (.<
    let dims1 = Ndarray.dims .~n1v in
    let dims2 = Ndarray.dims .~n2v in
    .~n.value <- Ndarray.create dims1;
  >.) in
  (* Not required, but we preserve the order, for readability. *)
  let init_values =
    if m1.processed && m2.processed then init_values_body
    else if m1.processed then (.< .~(m2.init_values); .~init_values_body >.)
    else if m2.processed then (.< .~(m1.init_values); .~init_values_body >.)
    else if l2r_comp_order then (.< .~(m1.init_values); .~(m2.init_values); .~init_values_body >.)
    else (.< .~(m2.init_values); .~(m1.init_values); .~init_values_body >.) in
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
    else if l2r_comp_order then (.< .~zero_body; .~(m2.zero_grads); .~(m1.zero_grads) >.)
    else (.< .~zero_body; .~(m1.zero_grads); .~(m2.zero_grads) >.) in
  (* The code needs to be included in the reverse order to which it was computed! *)
  let grad_body = grad_body ~n1d ~n2d ~nd ~nv ~n1v ~n2v in
  let backprop_body =
     if m1.processed && m2.processed then grad_body
     else if m1.processed then (.< .~grad_body; .~(m2.backprop_body) >.)
     else if m2.processed then (.< .~grad_body; .~(m1.backprop_body) >.)
     else if l2r_comp_order then (.< .~grad_body; .~(m2.backprop_body); .~(m1.backprop_body) >.)
     else (.< .~grad_body; .~(m1.backprop_body); .~(m2.backprop_body) >.) in
  let init_grads_body = (.<
    let dims = Ndarray.dims .~nv in
    .~n.grad <- Ndarray.create dims;
  >.) in
  (* The order is not relevant, we keep the same order as in backprop for readability. *)
  let init_grads =
    if m1.processed && m2.processed then init_grads_body
    else if m1.processed then (.< .~init_grads_body; .~(m2.init_grads) >.)
    else if m2.processed then (.< .~init_grads_body; .~(m1.init_grads) >.)
    else if l2r_comp_order then (.< .~init_grads_body; .~(m2.init_grads); .~(m1.init_grads) >.)
    else (.< .~init_grads_body; .~(m1.init_grads); .~(m2.init_grads) >.) in
  let toplevel_backprop = (.<
    .~init_grads;
    fun () ->
      .~(m1.zero_grads);
      .~(m2.zero_grads);
      Ndarray.reset_ones .~nd;
      .~backprop_body
  >.) in
  m1.processed <- true; m2.processed <- true;
  {toplevel_forward; toplevel_backprop; forward_body; backprop_body;
   init_values; init_grads; zero_grads;
   node_id; processed=false; debug_node}

let unop ~label ~name ~op_body ~grad_body m =
  let debug_node = Node.create ~label in
  let node_id = debug_node.id in
  let n1_id = m.node_id in
  let n = Codelib.genlet ~name:(name^"n") (.< Node.get node_id >.) in
  let nv = Codelib.genlet ~name:(name^"v") (.< .~n.value >.) in
  let n1v = Codelib.genlet ~name:(name^"1v") (.< (Node.get n1_id).value >.) in
  let op_body = op_body ~nv ~n1v in
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
  let grad_body = grad_body ~n1d ~nd ~nv ~n1v in
  (* The code needs to be included in the reverse order to which it was computed! *)
  let backprop_body =
    if m.processed then grad_body
    else (.< .~grad_body; .~(m.backprop_body) >.) in
  let init_grads_body = (.<
    let dims = Ndarray.dims .~nv in
    .~n.grad <- Ndarray.create dims;
  >.) in
  (* The order is not relevant, we keep the same order as in backprop for readability. *)
  let init_grads =
    if m.processed then init_grads_body
    else (.< .~init_grads_body; .~(m.init_grads) >.) in
  let toplevel_backprop = (.<
    .~init_grads;
    fun () ->
      .~(m.zero_grads);
      Ndarray.reset_ones .~nd;
      .~backprop_body
  >.) in
  m.processed <- true;
  {toplevel_forward; toplevel_backprop; forward_body; backprop_body;
  init_values; init_grads; zero_grads;
   node_id; processed=false; debug_node}

(* FIXME: be careful about where n1v etc. is created vs. where it's used. *)

(* ********** User API below ********** *)

(** A parameter or input of the model. The [label] must be unique, i.e. not already present in
    [Node.global.params]. *)
let param ~label ~(init_code:Ndarray.t Codelib.code) : submodel =
  let debug_node = Node.create ~label in
  let node_id = debug_node.id in
  let n = Codelib.genlet ~name:label (.< Node.get node_id >.) in
  let nv = Codelib.genlet ~name:(label ^ "v") (.< .~n.value >.) in
  (* Very unlikely someone will compute just the parameters. *)
  let forward_body = (.< () >.) in
  let init_values = (.<
    Hashtbl.add_exn Node.global.params ~key:label ~data:(.~n);
    .~n.value <- .~init_code;
  >.) in
  let toplevel_forward = (.< .~init_values; fun () -> .~forward_body >.) in
  let nd = Codelib.genlet ~name:"paramd" (.< .~n.grad >.) in
  let zero_grads = (.< Ndarray.reset_zeros .~nd >.) in
  let backprop_body = (.< () >.) in
  (* Very unlikely someone will want dw/dw. *)
  let init_grads = (.<
    let dims = Ndarray.dims .~nv in
    .~n.grad <- Ndarray.create dims;
  >.) in
  let toplevel_backprop = (.<
    .~init_grads;
    fun () ->
      Ndarray.reset_ones .~nd;
      .~backprop_body
  >.) in
  {toplevel_forward; toplevel_backprop; forward_body; backprop_body;
    init_values; init_grads; zero_grads;
    node_id; processed=false; debug_node}

let add =
  let label = "+" in
  let name = "add" in
  let op_body ~nv ~n1v ~n2v = (.< Ndarray.assign_add .~nv .~n1v .~n2v >.) in
  let grad_body ~n1d ~n2d ~nd ~nv:_ ~n1v:_ ~n2v:_ = (.<
    Ndarray.assign_add .~n1d .~n1d .~nd;
    Ndarray.assign_add .~n2d .~n2d .~nd
  >.) in
  binop ~label ~name ~op_body ~grad_body

let mul =
  let label = "*" in
  let name = "mul" in
  let op_body ~nv ~n1v ~n2v = (.< Ndarray.assign_mul .~nv .~n1v .~n2v >.) in
  let grad_body ~n1d ~n2d ~nd ~nv:_ ~n1v ~n2v = (.<
    Ndarray.assign_add .~n1d .~n1d (Ndarray.mul .~nd .~n2v);
    Ndarray.assign_add .~n2d .~n2d (Ndarray.mul .~nd .~n1v)
  >.) in
  binop ~label ~name ~op_body ~grad_body

let relu =
  let label = "relu" in
  let name = "relu" in
  let op_body ~nv ~n1v = (.< Ndarray.assign_relu .~nv .~n1v >.) in
  let grad_body ~n1d ~nd ~nv ~n1v:_ = (.<
    Ndarray.assign_add .~n1d .~n1d (Ndarray.relu_gate .~nv .~nd)
  >.) in
  unop ~label ~name ~op_body ~grad_body

let init_zeroes dims = (.< let p = Ndarray.create dims in Ndarray.reset_zeros p; p >.)
let init_uniform dims = (.< Ndarray.get_uniform ~low:(-1.0) ~high:1.0 dims >.)

(* 
~/ocannl$ dune utop

open Base;;
#load "_build/default/lib/ocannl.cma";;
open Ocannl;;
module F = Formula;;
let x = F.init_zeroes [|3; 3|];;
let w = F.init_uniform [|3; 3|];;
let nn = F.(add (mul (param ~label:"w" ~init_code:w) (param ~label:"x" ~init_code:x)) (param ~label:"b" ~init_code:b));;
let nn_fwd = Codelib.close_code nn.toplevel_forward;;
let nn_bwd = Codelib.close_code nn.toplevel_backprop;;
Codelib.format_code Caml.Format.std_formatter nn_fwd;;
Codelib.format_code Caml.Format.std_formatter nn_bwd;;
*)