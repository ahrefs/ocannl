(** Computational primitives for neural networks, integrating [Formula] with [Ndarray]. *)

open Base

(* Since this module is a utility wrapper around [Formula] rather than providing a new type,
   we open [Formula] globally. *)
open Formula

(** Whether to inline [Ndarray] operations. *)
let global_inline = ref true

let add_inline =
  let op_body ~nv ~n1v ~n2v indexing =
    Ndarray.(accum_binop_code ~accum:skip_arg_code ~op:add_code ~lhs:nv ~rhs1:n1v ~rhs2:n2v
               indexing.Shape.index_code) in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v:_ ~n2v:_ indexing = .<
    .~(Ndarray.(accum_unop_code ~accum:add_code ~op:Fn.id ~lhs:n1g ~rhs:ng @@
                Shape.backprop1 indexing.Shape.index_code));
    .~(Ndarray.(accum_unop_code ~accum:add_code ~op:Fn.id ~lhs:n2g ~rhs:ng @@
                Shape.backprop2 indexing.Shape.index_code))
  >. in
  binop ~compose_op:`Pointwise ~op_label:"t" ~op_body ~grad_body

let add_call =
  let op_body ~nv ~n1v ~n2v indexing =
    Ndarray.(accum_binop_call ~accum:skip_arg_call ~op:add_call ~lhs:nv ~rhs1:n1v ~rhs2:n2v
                  indexing.Shape.index_call) in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v:_ ~n2v:_ indexing = .<
    .~Ndarray.(accum_unop_call ~accum:add_call ~op:Fn.id ~lhs:n1g ~rhs:ng @@
               Shape.backprop1 indexing.Shape.index_call);
    .~Ndarray.(accum_unop_call ~accum:add_call ~op:Fn.id ~lhs:n2g ~rhs:ng @@
               Shape.backprop2 indexing.Shape.index_call)
  >. in
  binop ~compose_op:`Pointwise ~op_label:"t" ~op_body ~grad_body

let add m1 m2 = if !global_inline then add_inline m1 m2 else add_call m1 m2

let mul_inline ~compose_op =
  let op_body ~nv ~n1v ~n2v indexing =
    Ndarray.(accum_binop_code ~accum:skip_arg_code ~op:mul_code ~lhs:nv ~rhs1:n1v ~rhs2:n2v
               indexing.Shape.index_code) in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v ~n2v indexing = .<
    .~(Ndarray.(accum_binop_code ~accum:add_code ~op:mul_code ~lhs:n1g ~rhs1:ng ~rhs2:n2v @@
                Shape.backprop1 indexing.Shape.index_code));
    .~(Ndarray.(accum_binop_code ~accum:add_code ~op:mul_code ~lhs:n2g ~rhs1:ng ~rhs2:n1v @@
                Shape.backprop2 indexing.Shape.index_code))
  >. in
  binop ~compose_op ~op_label:"" ~op_body ~grad_body

let mul_call ~compose_op =
  let op_body ~nv ~n1v ~n2v indexing =
    Ndarray.(accum_binop_call ~accum:skip_arg_call ~op:mul_call ~lhs:nv ~rhs1:n1v ~rhs2:n2v
               indexing.Shape.index_call) in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v ~n2v indexing = .<
    .~Ndarray.(accum_binop_call ~accum:add_call ~op:mul_call ~lhs:n1g ~rhs1:ng ~rhs2:n2v @@
               Shape.backprop1 indexing.Shape.index_call);
    .~Ndarray.(accum_binop_call ~accum:add_call ~op:mul_call ~lhs:n2g ~rhs1:ng ~rhs2:n1v @@
               Shape.backprop2 indexing.Shape.index_call)
  >. in
  binop ~compose_op ~op_label:"" ~op_body ~grad_body

let pointmul_inline = mul_inline ~compose_op:`Pointwise
let pointmul_call = mul_call ~compose_op:`Pointwise
let pointmul m1 m2 = if !global_inline then pointmul_inline m1 m2 else pointmul_call m1 m2

(* N1: AxB, N2 BxC, N: AxC, A: output of N1, B: input/output of N1/N2, C: input of N2.
   Although the matrix algebra would require that we insert additional transposes in gradient multiplies:
   AxB = AxC * CxB = AxC * (BxC)^T -> N1g += Ng * N2v^T,
   BxC = BxA * AxC = (AxB)^T * AxC -> N2g += N1v^T * Ng,
   in our setup there is no transposing to do, since the projections produce correct indices for their
   corresponding matrices. *)

let matmul_inline = mul_inline ~compose_op:`Compose
let matmul_call = mul_call ~compose_op:`Compose
let matmul m1 m2 = if !global_inline then matmul_inline m1 m2 else matmul_call m1 m2

let relu_inline =
  let op_body ~nv ~n1v indexing =
    Ndarray.(accum_unop_code ~accum:skip_arg_code ~op:relu_code ~lhs:nv ~rhs:n1v
               indexing.Shape.index_code) in
  let grad_body ~n1g ~ng ~nv ~n1v:_ indexing =
    Ndarray.(accum_binop_code ~accum:add_code ~op:relu_gate_code ~lhs:n1g ~rhs1:nv ~rhs2:ng @@
             Shape.backprop_unary indexing.Shape.index_code) in
  unop ~transpose_op:`Pointwise ~op_label:"r" ~op_body ~grad_body

let relu_call =
  let op_body ~nv ~n1v indexing =
    Ndarray.(accum_unop_call ~accum:skip_arg_call ~op:relu_call ~lhs:nv ~rhs:n1v
               indexing.Shape.index_call) in
  let grad_body ~n1g ~ng ~nv ~n1v:_ indexing =
    Ndarray.(accum_binop_call ~accum:add_call ~op:relu_gate_call ~lhs:n1g ~rhs1:nv ~rhs2:ng @@
             Shape.backprop_unary indexing.Shape.index_call) in
  unop ~transpose_op:`Pointwise ~op_label:"r" ~op_body ~grad_body

let relu m = if !global_inline then relu_inline m else relu_call m

let reset_value v ~nv shape =
  Ndarray.(accum_unop_code ~accum:skip_arg_code ~op:(fun _ -> value_code v) ~lhs:nv ~rhs:nv
             (Shape.trivial_projections shape))

let float_to_label v = "v" ^ (
  Float.to_string v |> String.substr_replace_all ~pattern:"." ~with_:"p"
  |> String.substr_replace_all ~pattern:"-" ~with_:"m")

let number v =
  (* Note: no axis label so that we do not conflict with user labels. *)
  term ~label:(float_to_label v) (`Constant ([1], "")) ~op_body:(reset_value v)

let assign ~nv ~n1v indexing =
  Ndarray.(accum_unop_code ~accum:skip_arg_code ~op:(fun v -> v) ~lhs:nv ~rhs:n1v
   indexing.Shape.index_code)

(** A [stop_gradient] is an identity in the forward pass and a no-op in the backprop pass. *)
let stop_gradient =
  let grad_body ~n1g:_ ~ng:_ ~nv:_ ~n1v:_ _indexing = .< () >. in
  unop ~transpose_op:`Pointwise ~op_label:"r" ~op_body:assign ~grad_body

(** A [stop_broadcast] is an identity in both forward and backprop passes, which substitutes-in
    a [Fixed] copy of the shape of the input. *)
let stop_broadcast m =
  let sh = m.shape in
  let init_shape = Shape.{
     batch=Fixed (list_of_dims sh.batch);
     input=Fixed (list_of_dims sh.input); output=Fixed (list_of_dims sh.input);
     axis_labels=sh.axis_labels; deduce_output_from_input=`Not_deduced } in
  let grad_body ~n1g ~ng ~nv:_ ~n1v:_ indexing = assign ~nv:n1g ~n1v:ng indexing in
  unop ~init_shape ~transpose_op:`Pointwise ~op_label:"r" ~op_body:assign ~grad_body
    
module O = struct
  let ( * ) = matmul
  let ( *. ) = pointmul
  let (+) = add
  let (!/) = relu
  (* FIXME: NOT IMPLEMENTED *)
  (* let (!~) label = term ~label ~init_code:init_uniform *)
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

(* *** Printing. *** *)

let sprint_code code =
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

(** We print out up to 5 axes when printing an [Ndarray], as a grid (outer rectangle) of (inner)
    rectangles, possibly repeated. By default, the inner rectangle has an input and an output axis,
    and the batch axes are the vertical ones (the outer rectangle and the repetition). *)
type array_print_style =
[ `Default
(** The default behavior: the inner rectangles comprise both an input and an output axis, if available. *)
| `Prefer_input_axis
(** If there are 2 or more input axes, the inner rectangles comprise the input axes. *)
| `Prefer_output_axis
(** If there are 2 or more output axes, the inner rectangles comprise the output axes. *)
| `Prefer_batch_axis
(** The inner rectangles remain input x output, but the outer rectangles and repetition comprise
    batch axes, if available. *)
| `Exact_layout of string
(** The string should use integer pseudo-labels to represent the priorities of the axes, where the
    priorities correspond to, from highest: horizontal direction, vertical direction of inner rectangle,
    horizontal direction, vertical direction of outer rectangle, repetition. *)
]

let print_formula ~with_grad ~with_code (style: array_print_style) m =
  assert (m.node_id = m.comp_node.id);
  Stdio.print_endline @@ "["^Int.to_string m.node_id^"] "^m.comp_node.label^": "^
                         Shape.to_string_hum m.shape;
  let indices = failwith "INDICES NOT IMPLEMENTED" in
  let order_of_axes =
    match style with
    | `Exact_layout priorities ->
      let p_labels = Shape.axis_labels_of_spec priorities in

       ignore p_labels; []
    | _ -> failwith "STYLE NOT IMPLEMENTED" in
  let labels = failwith "LABEL PASSING NOT IMPLEMENTED" in
  let screen_stop () =
    Stdio.print_endline "Press [Enter] for next screen, [q] [Enter] to quit.";
    String.(Stdio.In_channel.input_line_exn Stdio.stdin = "q")  in
  Ndarray.pp_print Caml.Format.std_formatter ~screen_stop ~indices m.comp_node.value;
  if with_grad then (
    Stdio.print_endline "Gradient:";
    Ndarray.pp_print Caml.Format.std_formatter ~order_of_axes ~labels ~screen_stop ~indices
      m.comp_node.grad);
  if with_code then (
    (match m.forward_body with
     | None -> ()
     | Some fwd_code ->
       Stdio.print_endline "Current forward body:";
       Stdio.print_endline @@ fst @@ sprint_code @@ fwd_code());
    (match m.backprop_body with
     | None -> ()
     | Some bwd_code ->
       Stdio.print_endline "Current backprop body:";
       Stdio.print_endline @@ fst @@ sprint_code @@ bwd_code())
  );
  Stdio.printf "\n%!"

let print_global_root ~with_grad ~with_code (style: array_print_style) root =
  print_formula ~with_grad ~with_code:false style root.formula;
  if with_code then (
    (match root.forward_code with
     | None -> ()
     | Some fwd_code ->
       Stdio.print_endline "Forward:";
       Stdio.print_endline @@ fst @@ sprint_code fwd_code);
    (match root.backprop_code with
     | None -> ()
     | Some bwd_code ->
       Stdio.print_endline "Backprop:";
       Stdio.print_endline @@ fst @@ sprint_code bwd_code)
  );
  Stdio.printf "\n%!"

let print_global_roots ~with_grad ~with_code (style: array_print_style) =
  List.iter (Map.to_alist ~key_order:`Increasing !global_roots) ~f:(fun (node_id, root) ->
      assert (node_id = root.formula.node_id);
      print_global_root ~with_grad ~with_code style root)


module CLI = struct
  module FO = O
  (* FIXME: NOT IMPLEMENTED *)
  (* let init_zeroes = init_zeroes *)
  (* let init_uniform = init_uniform *)
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
