(** Computational primitives for neural networks, integrating [Formula] with [Ndarray]. *)

open Base

(* Since this module is a utility wrapper around [Formula] rather than providing a new type,
   we open [Formula] globally. *)
open Formula

let add =
  let op_body ~nv ~n1v ~n2v projections =
    Ndarray.(accum_binop_code ~accum:skip_arg_code ~op:add_code ~lhs:nv ~rhs1:n1v ~rhs2:n2v projections) in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v:_ ~n2v:_ projections = .<
    .~(Ndarray.(accum_unop_code ~accum:add_code ~op:Fn.id ~lhs:n1g ~rhs:ng @@ Shape.backprop1 projections));
    .~(Ndarray.(accum_unop_code ~accum:add_code ~op:Fn.id ~lhs:n2g ~rhs:ng @@ Shape.backprop2 projections))
  >. in
  binop ~compose_op:`Pointwise ~op_label:"t" ~op_body ~grad_body

let mul ~compose_op =
  let op_body ~nv ~n1v ~n2v projections =
    Ndarray.(accum_binop_code ~accum:skip_arg_code ~op:mul_code ~lhs:nv ~rhs1:n1v ~rhs2:n2v projections) in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v ~n2v projections = .<
    .~(Ndarray.(accum_binop_code ~accum:add_code ~op:mul_code ~lhs:n1g ~rhs1:ng ~rhs2:n2v @@
                Shape.backprop1 projections));
    .~(Ndarray.(accum_binop_code ~accum:add_code ~op:mul_code ~lhs:n2g ~rhs1:ng ~rhs2:n1v @@
                Shape.backprop2 projections))
  >. in
  binop ~compose_op ~op_label:"" ~op_body ~grad_body

let pointmul = mul ~compose_op:`Pointwise

(* N1: AxB, N2 BxC, N: AxC, A: output of N1, B: input/output of N1/N2, C: input of N2.
   Although the matrix algebra would require that we insert additional transposes in gradient multiplies:
   AxB = AxC * CxB = AxC * (BxC)^T -> N1g += Ng * N2v^T,
   BxC = BxA * AxC = (AxB)^T * AxC -> N2g += N1v^T * Ng,
   in our setup there is no transposing to do, since the projections produce correct indices for their
   corresponding matrices. *)

let matmul = mul ~compose_op:`Compose

let relu =
  let op_body ~nv ~n1v projections =
    Ndarray.(accum_unop_code ~accum:skip_arg_code ~op:relu_code ~lhs:nv ~rhs:n1v projections) in
  let grad_body ~n1g ~ng ~nv ~n1v:_ projections =
    Ndarray.(accum_binop_code ~accum:add_code ~op:relu_gate_code ~lhs:n1g ~rhs1:nv ~rhs2:ng @@
             Shape.backprop_unary projections) in
  unop ~transpose_op:`Pointwise ~op_label:"r" ~op_body ~grad_body

let reset_value v ~nv shape =
  Ndarray.(accum_unop_code ~accum:skip_arg_code ~op:(fun _ -> value_code v) ~lhs:nv ~rhs:nv
             (Shape.terminal_projections shape))

let uniform_value ~nv shape =
  Ndarray.(accum_unop_code ~accum:skip_arg_code ~op:(fun _ -> uniform_code ~low:(-1.0) ~high:1.0)
             ~lhs:nv ~rhs:nv @@ Shape.terminal_projections shape)

let float_to_label v = "v" ^ (
  Float.to_string v |> String.substr_replace_all ~pattern:"." ~with_:"p"
  |> String.substr_replace_all ~pattern:"-" ~with_:"m")

let number v =
  (* Note: no axis label so that we do not conflict with user labels. *)
  term ~label:(float_to_label v) (`Constant ([1], "")) ~op_body:(reset_value v)

let assign ~nv ~n1v projections =
  Ndarray.(accum_unop_code ~accum:skip_arg_code ~op:(fun v -> v) ~lhs:nv ~rhs:n1v projections)

(** A [stop_gradient] is an identity in the forward pass and a no-op in the backprop pass. *)
let stop_gradient =
  let grad_body ~n1g:_ ~ng:_ ~nv:_ ~n1v:_ _projections = .< () >. in
  unop ~transpose_op:`Pointwise ~op_label:"r" ~op_body:assign ~grad_body

(** A [stop_broadcast] is an identity in both forward and backprop passes, which substitutes-in
    a [Fixed] copy of the shape of the input. *)
let stop_broadcast m =
  let sh = m.shape in
  let init_shape = Shape.{
     batch=Fixed (list_of_dims sh.batch);
     input=Fixed (list_of_dims sh.input); output=Fixed (list_of_dims sh.input);
     axis_labels=sh.axis_labels; deduce_output_from_input=`Not_deduced } in
  let grad_body ~n1g ~ng ~nv:_ ~n1v:_ projections = assign ~nv:n1g ~n1v:ng projections in
  unop ~init_shape ~transpose_op:`Pointwise ~op_label:"r" ~op_body:assign ~grad_body
    
module O = struct
  let ( * ) = matmul
  let ( *. ) = pointmul
  let (+) = add
  let (!/) = relu
  let (!~) label = term ~label (`Deduced_params `Not_deduced) ~op_body:uniform_value
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
    rectangles, possibly repeated (screens). By default, the inner rectangle has an input and an output
    axis, and the batch axes are the vertical ones (the outer rectangle and the repetition). *)
type array_print_style =
[ `Default
(** Tthe inner rectangles comprise both an input and an output axis, if available; and the screens
    comprise a batch axis, if available and if there is 5 or more axes. *)
| `N5_layout of string
(** The string should provide exclusively non-negative integer pseudo-labels. The numbers [0]-[4] represent
    the priorities of the axes to be printed out, where the priorities correspond to, from highest:
    horizontal directions of inner, outer rectangle, verticals directions of inner, outer rectangle,
    repetition (see also [Ndarray.pp_print]). The numbers [n >= 5] stand for the actual positions [n - 5]
    within the corresponding axes. *)
| `Label_layout of (string * int) list
(** The association from axis labels to integers. The negative numbers [-5] to [-1] represent
    the priorities of the axes to be printed out, where the priorities correspond to, from highest:
    horizontal directions of inner, outer rectangle, verticals directions of inner, outer rectangle,
    repetition (see also [Ndarray.pp_print]). The non-negative numbers stand for the actual positions
    within the corresponding axes. Unspecified axes will be printed at position [0]. *)
]

let print_formula ~with_grad ~with_code (style: array_print_style) m =
  assert (m.node_id = m.comp_node.id);
  let sh = m.shape in
  Stdio.print_endline @@ "["^Int.to_string m.node_id^"] "^m.comp_node.label^": "^
                         Shape.to_string_hum sh;
  let indices =
    match style with
    | `Default ->
      [||]
    | `N5_layout priorities ->
      let p_labels = Shape.axis_labels_of_spec priorities in
      
       ignore p_labels; [||]
    | `Label_layout label_idcs ->
      ignore label_idcs; [||]
     in
  (* let labels = sh.Shape.axis_labels in *)
  let labels = [||] in
  let screen_stop () =
    Stdio.print_endline "Press [Enter] for next screen, [q] [Enter] to quit.";
    String.(Stdio.In_channel.input_line_exn Stdio.stdin = "q")  in
  Ndarray.pp_print Caml.Format.std_formatter ~labels ~screen_stop ~indices m.comp_node.value;
  if with_grad then (
    Stdio.print_endline "Gradient:";
    Ndarray.pp_print Caml.Format.std_formatter ~labels ~screen_stop ~indices
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
  let reset_value = reset_value
  let uniform_value = uniform_value
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
