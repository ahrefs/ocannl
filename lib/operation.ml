(** Computational primitives for neural networks, integrating [Formula] with [Ndarray]. *)

open Base

(* Since this module is a utility wrapper around [Formula] rather than providing a new type,
   we open [Formula] here. *)
open Formula

(** Whether to inline [Ndarray] operations. *)
let global_inline = ref true

let add_inline =
  let op_body ~nv ~n1v ~n2v = Ndarray.assign_add_code nv n1v n2v in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v:_ ~n2v:_ = .<
    .~(Ndarray.assign_add_code n1g n1g ng);
    .~(Ndarray.assign_add_code n2g n2g ng)
  >. in
  binop ~compose_op:`Pointwise ~op_label:"t" ~op_body ~grad_body

let add_call =
  let op_body ~nv ~n1v ~n2v = .< Ndarray.assign_add .~nv .~n1v .~n2v >. in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v:_ ~n2v:_ = .<
    Ndarray.assign_add .~n1g .~n1g .~ng;
    Ndarray.assign_add .~n2g .~n2g .~ng
  >. in
  binop ~compose_op:`Pointwise ~op_label:"t" ~op_body ~grad_body

let add m1 m2 = if !global_inline then add_inline m1 m2 else add_call m1 m2

let mul_pointwise_inline =
  let op_body ~nv ~n1v ~n2v = Ndarray.assign_mul_code nv n1v n2v in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v ~n2v = .<
    .~(Ndarray.assign_add_code n1g n1g (Ndarray.mul_code .< Ndarray.dims .~n1g >. ng n2v));
    .~(Ndarray.assign_add_code n2g n2g (Ndarray.mul_code .< Ndarray.dims .~n2g >. ng n1v))
  >. in
  binop ~compose_op:`Pointwise ~op_label:"" ~op_body ~grad_body

let mul_pointwise_call =
  let op_body ~nv ~n1v ~n2v = .< Ndarray.assign_mul .~nv .~n1v .~n2v >. in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v ~n2v = .<
    Ndarray.assign_add .~n1g .~n1g (Ndarray.mul (Ndarray.dims .~n1g) .~ng .~n2v);
    Ndarray.assign_add .~n2g .~n2g (Ndarray.mul (Ndarray.dims .~n2g) .~ng .~n1v)
  >. in
  binop ~compose_op:`Pointwise ~op_label:"" ~op_body ~grad_body

let mul_pointwise m1 m2 = if !global_inline then mul_pointwise_inline m1 m2 else mul_pointwise_call m1 m2

let matmul_inline =
  (* FIXME(14): not implemented: either mul_pointwise or matmul need to use a different set of Ndarray
     routines. *)
  let op_body ~nv ~n1v ~n2v = Ndarray.assign_mul_code nv n1v n2v in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v ~n2v = .<
    .~(Ndarray.assign_add_code n1g n1g (Ndarray.mul_code .< Ndarray.dims .~n1g >. ng n2v));
    .~(Ndarray.assign_add_code n2g n2g (Ndarray.mul_code .< Ndarray.dims .~n2g >. ng n1v))
  >. in
  binop ~compose_op:`Compose ~op_label:"" ~op_body ~grad_body

let matmul_call =
  (* FIXME(14): not implemented: either mul_pointwise or matmul need to use a different set of Ndarray
     routines. *)
  let op_body ~nv ~n1v ~n2v = .< Ndarray.assign_mul .~nv .~n1v .~n2v >. in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v ~n2v = .<
    Ndarray.assign_add .~n1g .~n1g (Ndarray.mul (Ndarray.dims .~n1g) .~ng .~n2v);
    Ndarray.assign_add .~n2g .~n2g (Ndarray.mul (Ndarray.dims .~n2g) .~ng .~n1v)
  >. in
  binop ~compose_op:`Compose ~op_label:"" ~op_body ~grad_body

let matmul m1 m2 = if !global_inline then matmul_inline m1 m2 else matmul_call m1 m2

let relu_inline =
  let op_body ~nv ~n1v = Ndarray.assign_relu_code nv n1v in
  let grad_body ~n1g ~ng ~nv ~n1v:_ = .<
    Ndarray.assign_add .~n1g .~n1g (Ndarray.relu_gate (Ndarray.dims .~n1g) .~nv .~ng)
  >. in
  unop ~transpose_op:`Pointwise ~op_label:"r" ~op_body ~grad_body

let relu_call =
  let op_body ~nv ~n1v = .< Ndarray.assign_relu .~nv .~n1v >. in
  let grad_body ~n1g ~ng ~nv ~n1v:_ = .<
    Ndarray.assign_add .~n1g .~n1g (Ndarray.relu_gate (Ndarray.dims .~n1g) .~nv .~ng)
  >. in
  unop ~transpose_op:`Pointwise ~op_label:"r" ~op_body ~grad_body

let relu m = if !global_inline then relu_inline m else relu_call m

let init_zeroes dims_code =
   .< let p = Ndarray.create .~dims_code in Ndarray.reset_zeros p; p >.
let init_uniform dims_code =
   .< Ndarray.get_uniform ~low:(-1.0) ~high:1.0 .~dims_code >.

let float_to_label v = "v" ^ (
  Float.to_string v |> String.substr_replace_all ~pattern:"." ~with_:"p"
  |> String.substr_replace_all ~pattern:"-" ~with_:"m")

let number v =
  (* Note: no axis label so that we do not conflict with user labels. *)
  term ~label:(float_to_label v) (`Constant ([1], ""))
    ~init_code:(fun dims -> .< Ndarray.get_val v .~dims >.)

(** A [stop_gradient] is an identity in the forward pass and a no-op in the backprop pass. *)
let stop_gradient =
  let op_body ~nv ~n1v = .< Ndarray.assign .~nv .~n1v >. in
  let grad_body ~n1g:_ ~ng:_ ~nv:_ ~n1v:_ = .< () >. in
  unop ~transpose_op:`Pointwise ~op_label:"r" ~op_body ~grad_body

let stop_broadcast m =
  let sh = m.shape in
  let init_shape = Shape.{
     batch=Fixed (list_of_dims sh.batch);
     input=Fixed (list_of_dims sh.input); output=Fixed (list_of_dims sh.input);
     axis_labels=sh.axis_labels; deduce_output_from_input=`Not_deduced } in
  let op_body ~nv ~n1v = .< Ndarray.assign .~nv .~n1v >. in
  let grad_body ~n1g:_ ~ng:_ ~nv:_ ~n1v:_ = .< () >. in
  unop ~init_shape ~transpose_op:`Pointwise ~op_label:"r" ~op_body ~grad_body
    
module O = struct
  let ( * ) = matmul
  let ( *. ) = mul_pointwise
  let (+) = add
  let (!/) = relu
  let (!~) label = term ~label ~init_code:init_uniform
  let (!.) = number
  let (-) m1 m2 = m1 + !.(-1.) * m2
end

let get_root id =
  match Map.find !global_roots id with
  | Some r -> r
  | None ->
    let msg = 
      if id >= !first_session_id && id < Node.global.unique_id then
        "get_root: Node "^Int.to_string id^" is a subformula"
      else if id >= Node.global.unique_id then
        "get_root: Node "^Int.to_string id^" has not been created yet"
      else if id < 1 then "get_root: Node IDs start from 1"
      else
        "get_root: Node "^Int.to_string id^" is outside the current session" in
    raise @@ Session_error (msg, None)

let get_node id =
  match Hashtbl.find Node.global.node_store id with
  | Some r -> r
  | None ->
    let msg = 
      if id >= Node.global.unique_id then
        "get_node: Node "^Int.to_string id^" has not been created yet"
      else if id < 1 then "get_root: Node IDs start from 1"
      else
        "get_node: Node "^Int.to_string id^" has been removed or lives on a different machine" in
    raise @@ Session_error (msg, None)

module CLI = struct
  module FO = O
  let init_zeroes = init_zeroes
  let init_uniform = init_uniform
  let term = term
  let stop_broadcast = stop_broadcast
  let stop_gradient = stop_gradient
  let print_global_root = print_global_root
  let print_node = Node.print_node
  let print_formula = print_formula
  let print_global_roots = print_global_roots
  let get_root = get_root
  let get_node = get_node
end

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
module F = Operation
let d = [|3; 3|]
let nn = F.O.(!/(!~"w" d * !~"x" d + !~"b" d))
let () = Stdio.print_endline @@ fst @@ F.sprint nn.toplevel_forward
let () = Stdio.print_endline @@ fst @@ F.sprint nn.toplevel_backprop
*)
