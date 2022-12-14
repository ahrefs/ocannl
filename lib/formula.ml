(** The compositional primitives for runtime-compiled code supporting backpropagation. *)

open Base

(** Uses [code option], i.e. [None] instead of [.< () >.], to improve readability of generated code. *)
type t = {
  toplevel_forward: (unit -> unit) Codelib.code;
  (** Only apply at the root, since otherwise some computation may be elided (incorrect results). *)
  toplevel_backprop: (unit -> unit) Codelib.code;
  (** Only apply at the root! Gradients propagate from the top and are only propagated once. Zeroes
      the gradients before propagating. *)
  forward_body: unit Codelib.code option;
  init_values: unit Codelib.code;
  (** Initializes the values. Computed only once per model compilation. *)
  init_grads: unit Codelib.code;
  (** Initializes the gradient data: typically, simply creates the ndarrays.
      Gradients are zeroed separately. *)
  backprop_body: unit Codelib.code option;
  zero_grads: unit Codelib.code;
  (** Initializes the backpropagation phase. Computed once per backpropagation. *)
  node_id: int;
  mutable processed: bool;
  (** [true] if [forward_body]/[backprop_body]/[zero_grads] were already included in a parent `t`. *)
  comp_node: Node.t;
  (** This tracks the computation node as long as the model is not cross-compiled to a different
      process etc. *)
  node: Node.t Codelib.code;
  (** The node storing the computation results. *)
}

(* The code relies on argument evaluation order. To lift the requirement, we could use
   [t Lazy.t], but that's an unnecessary obfuscation. *)
let l2r_comp_order =
  let l2r_ord = ref None in
  (fun () () ->
    match !l2r_ord with
    | Some b -> b
    | None -> assert false) (l2r_ord := Some false) (l2r_ord := Some true)

(* Design choice: tensor shape is decided after code is constructed, but when it is loaded.
   I.e. code needs to be re-executed with [Runcode.run_bytecode] or [Runnative.run_native]
   when the dimensions change. *)

let binop ~op_label ~op_body ~grad_body m1 m2: t =
  let m1_l = m1.comp_node.label in
  let m1_l = if String.length m1_l > 11 then "n"^Int.to_string m1.node_id else m1_l in
  let m2_l = m2.comp_node.label in
  let m2_l = if String.length m2_l > 11 then "n"^Int.to_string m2.node_id else m2_l in
  let label = m1_l ^ op_label ^ m2_l in
  let comp_node = Node.create ~label in
  let node_id = comp_node.id in
  let node = Codelib.genlet ~name:label (.< Node.get node_id >.) in
  let nv = (.< .~node.value >.) in
  let n1v = (.< .~(m1.node).value >.) in
  let n2v = (.< .~(m2.node).value >.) in
  let op_body = op_body ~nv ~n1v ~n2v in
  (* The code needs to be included in the order it was computed! *)
  let forward_body =
    match m1.processed, m1.forward_body, m2.processed, m2.forward_body with
    | true, _, true, _ | true, _, _, None | _, None, true, _ | _, None, _, None -> op_body
    | false, Some m1_body, false, Some m2_body when l2r_comp_order ->
      (.< .~m1_body; .~m2_body; .~op_body >.)
    | false, Some m1_body, false, Some m2_body ->
      (.< .~m2_body; .~m1_body; .~op_body >.) 
    | _, _, false, Some m2_body -> (.< .~m2_body; .~op_body >.)
    | false, Some m1_body, _, _ -> (.< .~m1_body; .~op_body >.)
  in
  let init_values_body = (.<
    .~node.value <- Ndarray.create (Ndarray.shape .~n1v);
  >.) in
  (* Not required, but we preserve the order, for readability. *)
  let init_values =
    if m1.processed && m2.processed then init_values_body
    else if m1.processed then (.< .~(m2.init_values); .~init_values_body >.)
    else if m2.processed then (.< .~(m1.init_values); .~init_values_body >.)
    else if l2r_comp_order then (.< .~(m1.init_values); .~(m2.init_values); .~init_values_body >.)
    else (.< .~(m2.init_values); .~(m1.init_values); .~init_values_body >.) in
  let toplevel_forward = (.< .~init_values; fun () -> .~forward_body >.) in
  let nd = (.< .~node.grad >.) in
  let n1d = (.< .~(m1.node).grad >.) in
  let n2d = (.< .~(m2.node).grad >.) in
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
    match m1.processed, m1.backprop_body, m2.processed, m2.backprop_body with
    | true, _, true, _ | true, _, _, None | _, None, true, _ | _, None, _, None -> grad_body
    | false, Some m1_body, false, Some m2_body when l2r_comp_order ->
      (.< .~grad_body; .~m1_body; .~m2_body >.)
    | false, Some m1_body, false, Some m2_body ->
      (.< .~grad_body; .~m2_body; .~m1_body;  >.) 
    | _, _, false, Some m2_body -> (.< .~grad_body; .~m2_body  >.)
    | false, Some m1_body, _, _ -> (.< .~grad_body; .~m1_body  >.)
    in
  let init_grads_body = (.<
    .~node.grad <- Ndarray.create (Ndarray.shape .~nv);
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
  {toplevel_forward; toplevel_backprop;
   forward_body=Some forward_body; backprop_body=Some backprop_body;
   init_values; init_grads; zero_grads;
   node_id; processed=false; comp_node; node}

let unop ~op_label ~op_body ~grad_body m: t =
  let m_l = m.comp_node.label in
  let m_l = if String.length m_l > 11 then "n"^Int.to_string m.node_id else m_l in
  let label = op_label ^ m_l in
  let comp_node = Node.create ~label in
  let node_id = comp_node.id in
  let node = Codelib.genlet ~name:label (.< Node.get node_id >.) in
  let nv = (.< .~node.value >.) in
  let n1v = (.< .~(m.node).value >.) in
  let op_body = op_body ~nv ~n1v in
  (* The code needs to be included in the order it was computed! *)
  let forward_body =
    match m.processed, m.forward_body with
    | true, _ | _, None -> op_body
    | false, Some m_body -> (.< .~m_body; .~op_body >.) in
  let init_values = (.<
    .~(m.init_values);
    .~node.value <- Ndarray.create (Ndarray.shape .~n1v);
  >.) in
  let toplevel_forward = (.< .~init_values; fun () -> .~forward_body >.) in
  let nd = (.< .~node.grad >.) in
  let n1d = (.< .~(m.node).grad >.) in
  let zero_body = (.< Ndarray.reset_zeros .~nd >.) in
  (* The order of zeroing gradients is irrelevant and multiple zeroing is fine, but we avoid it
       and keep the backprop order for readability. *)
  let zero_grads =
    if m.processed then zero_body
    else (.< .~zero_body; .~(m.zero_grads) >.) in
  let grad_body = grad_body ~n1d ~nd ~nv ~n1v in
  (* The code needs to be included in the reverse order to which it was computed! *)
  let backprop_body =
    match m.processed, m.backprop_body with
    | true, _ | _, None -> grad_body
    | false, Some m_body -> (.< .~grad_body; .~m_body >.) in
  let init_grads_body = (.<
    .~node.grad <- Ndarray.create (Ndarray.shape .~nv);
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
  {toplevel_forward; toplevel_backprop;
   forward_body=Some forward_body; backprop_body=Some backprop_body;
   init_values; init_grads; zero_grads;
   node_id; processed=false; comp_node; node}

(* FIXME: be careful about where n1v etc. is created vs. where it's used. *)

(* ********** User API below ********** *)

(** A terminal: a constant, a parameter, an input of the model. *)
let term ~label ~(init_code:Ndarray.t Codelib.code) : t =
  let comp_node = Node.create ~label in
  let node_id = comp_node.id in
  let node = Codelib.genlet ~name:label (.< Node.get node_id >.) in
  let nv = (.< .~node.value >.) in
  (* Very unlikely someone will compute just the parameters. *)
  let forward_body = None in
  let init_values = (.< .~node.value <- .~init_code >.) in
  let toplevel_forward = (.< .~init_values; fun () -> () >.) in
  let nd = Codelib.genlet ~name:(label^"d") (.< .~node.grad >.) in
  let zero_grads = (.< Ndarray.reset_zeros .~nd >.) in
  let backprop_body = None in
  (* Very unlikely someone will want dw/dw. *)
  let init_grads = (.<
    .~node.grad <- Ndarray.create (Ndarray.shape .~nv);
  >.) in
  let toplevel_backprop = (.<
    .~init_grads;
    fun () -> Ndarray.reset_ones .~nd; ()
  >.) in
  {toplevel_forward; toplevel_backprop; forward_body; backprop_body;
    init_values; init_grads; zero_grads;
    node_id; processed=false; comp_node; node}

let add =
  let op_body ~nv ~n1v ~n2v = (.< Ndarray.assign_add .~nv .~n1v .~n2v >.) in
  let grad_body ~n1d ~n2d ~nd ~nv:_ ~n1v:_ ~n2v:_ = (.<
    Ndarray.assign_add .~n1d .~n1d .~nd;
    Ndarray.assign_add .~n2d .~n2d .~nd
  >.) in
  binop ~op_label:"t" ~op_body ~grad_body

let mul =
  let op_body ~nv ~n1v ~n2v = (.< Ndarray.assign_mul .~nv .~n1v .~n2v >.) in
  let grad_body ~n1d ~n2d ~nd ~nv:_ ~n1v ~n2v = (.<
    Ndarray.assign_add .~n1d .~n1d (Ndarray.mul .~nd .~n2v);
    Ndarray.assign_add .~n2d .~n2d (Ndarray.mul .~nd .~n1v)
  >.) in
  binop ~op_label:"" ~op_body ~grad_body

let relu =
  let op_body ~nv ~n1v = (.< Ndarray.assign_relu .~nv .~n1v >.) in
  let grad_body ~n1d ~nd ~nv ~n1v:_ = (.<
    Ndarray.assign_add .~n1d .~n1d (Ndarray.relu_gate .~nv .~nd)
  >.) in
  unop ~op_label:"r" ~op_body ~grad_body

let init_zeroes shape = (.< let p = Ndarray.create shape in Ndarray.reset_zeros p; p >.)
let init_uniform shape = (.< Ndarray.get_uniform ~low:(-1.0) ~high:1.0 shape >.)

let float_to_label v = "v" ^ (
  Float.to_string v |> String.substr_replace_all ~pattern:"." ~with_:"p"
  |> String.substr_replace_all ~pattern:"-" ~with_:"m")

let number v =
  (* TODO(5): use dimensions inference and broadcasting. *)
  term ~label:(float_to_label v) ~init_code:(.< Ndarray.get_val v [|1|] >.)

module O = struct
  let ( * ) = mul
  let (+) = add
  let (!/) = relu
  let (!~) label shape = term ~label ~init_code:(init_uniform shape)
  let (!.) = number
  let (-) m1 m2 = m1 + !.(-1.) * m2
end

let sprint code =
  let closed, check = Codelib.close_code_delay_check code in
  ignore (Caml.Format.flush_str_formatter());
  Caml.Format.pp_set_margin Caml.Format.str_formatter 160;
  Codelib.format_code Caml.Format.str_formatter closed;
  let s = Caml.Format.flush_str_formatter() in
  let s = String.substr_replace_all s ~pattern:"Base." ~with_:"" in
  let s = String.substr_replace_all s ~pattern:"Ocannl." ~with_:"" in
  let s = String.substr_replace_all s ~pattern:"Ndarray." ~with_:"" in
  let s = String.substr_replace_all s ~pattern:"Node." ~with_:"" in
  s, check

(* TODO: maybe streamline [t] to enable [t_of_sexp]. *)
let sexp_of_t m =
  Sexp.message "Formula" [
    "label", String.sexp_of_t m.comp_node.label; "node_id", Int.sexp_of_t m.node_id;
    "toplevel_forward", String.sexp_of_t @@ fst @@ sprint m.toplevel_forward;
    "toplevel_backprop", String.sexp_of_t @@ fst @@ sprint m.toplevel_backprop;
  ]

include Comparator.Make(struct
    type nonrec t = t
    let compare m1 m2 = Int.compare m1.node_id m2.node_id
    let sexp_of_t = sexp_of_t
end)

module Summable = struct
  type nonrec t = t
  let (+) = add
  let zero = number 0.0
end

(*
let postprocess code =
  let closed, check = Codelib.close_code_delay_check code in
  let ast = Codelib.ast_of_code closed in
  Printast.expression
*)

(* 
~/ocannl$ dune utop

open Base
#load "_build/default/lib/ocannl.cma"
open Ocannl
module F = Formula
let d = [|3; 3|]
let nn = F.O.(!/(!~"w" d * !~"x" d + !~"b" d))
let () = Stdio.print_endline @@ fst @@ F.sprint nn.toplevel_forward
let () = Stdio.print_endline @@ fst @@ F.sprint nn.toplevel_backprop
*)
