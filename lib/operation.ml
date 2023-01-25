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

let mul compose_op =
  let op_body ~nv ~n1v ~n2v projections =
    Ndarray.(accum_binop_code ~accum:skip_arg_code ~op:mul_code ~lhs:nv ~rhs1:n1v ~rhs2:n2v projections) in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v ~n2v projections = .<
    .~(Ndarray.(accum_binop_code ~accum:add_code ~op:mul_code ~lhs:n1g ~rhs1:ng ~rhs2:n2v @@
                Shape.backprop1 projections));
    .~(Ndarray.(accum_binop_code ~accum:add_code ~op:mul_code ~lhs:n2g ~rhs1:ng ~rhs2:n1v @@
                Shape.backprop2 projections))
  >. in
  binop ~compose_op ~op_label:"" ~op_body ~grad_body

let pointmul = mul `Pointwise

(* N1: AxB, N2 BxC, N: AxC, A: output of N1, B: input/output of N1/N2, C: input of N2.
   Although the matrix algebra would require that we insert additional transposes in gradient multiplies:
   AxB = AxC * CxB = AxC * (BxC)^T -> N1g += Ng * N2v^T,
   BxC = BxA * AxC = (AxB)^T * AxC -> N2g += N1v^T * Ng,
   in our setup there is no transposing to do, since the projections produce correct indices for their
   corresponding matrices. *)

let matmul = mul `Compose

(** Similar to the explicit mode of [numpy.einsum], the binary variant. Can compute various forms of
    matrix multiplication, inner and outer products, etc.

    Note that ["a,b->c"] from [numpy] is ["a;b=>c"] in OCaNNL, since ["->"] is used to separate the input
    and the output axes. *)
let einsum spec =
  let op_body ~nv ~n1v ~n2v projections =
    Ndarray.(accum_binop_code ~zero_out:true ~accum:add_code ~op:mul_code ~lhs:nv ~rhs1:n1v ~rhs2:n2v
               projections) in
  let grad_body ~n1g ~n2g ~ng ~nv:_ ~n1v ~n2v projections = .<
    .~(Ndarray.(accum_binop_code ~accum:add_code ~op:mul_code ~lhs:n1g ~rhs1:ng ~rhs2:n2v @@
                Shape.backprop1 projections));
    .~(Ndarray.(accum_binop_code ~accum:add_code ~op:mul_code ~lhs:n2g ~rhs1:ng ~rhs2:n1v @@
                Shape.backprop2 projections))
  >. in
  binop ~compose_op:(`Einsum spec) ~op_label:"" ~op_body ~grad_body

(** Similar to the explicit mode of [numpy.einsum], the unary variant. Can permute axes, extract diagonals,
    compute traces etc.

    Note that ["a->c"] from [numpy] is ["a=>c"] in OCaNNL, since ["->"] is used to separate the input
    and the output axes. *)
let einsum1 spec =
  let op_body ~nv ~n1v projections =
    Ndarray.(accum_unop_code ~zero_out:true ~accum:add_code ~op:id_code ~lhs:nv ~rhs:n1v
               projections) in
  let grad_body ~n1g ~ng ~nv:_ ~n1v:_ projections =
    Ndarray.(accum_unop_code ~accum:add_code ~op:id_code ~lhs:n1g ~rhs:ng @@
                Shape.backprop_unary projections) in
  unop ~transpose_op:(`Permute spec) ~op_label:"" ~op_body ~grad_body

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

(** A [stop_broadcast] mutates the partially-inferred shape of a formula in-place, substituting-in
    a [Fixed] marker on the dimensions. This way we avoid introducing a new node. *)
let stop_broadcast m =
  let sh = m.shape in
  sh.batch <- Fixed (Shape.list_of_dims sh.batch);
  sh.batch <- Fixed (Shape.list_of_dims sh.batch);
  sh.batch <- Fixed (Shape.list_of_dims sh.batch);
  m
    
(** [identity] introduces a new node, which is an identity in both the forward and backward pass. *)
let identity m =
  let grad_body ~n1g ~ng ~nv:_ ~n1v:_ projections = assign ~nv:n1g ~n1v:ng projections in
  unop ~init_shape:m.shape ~transpose_op:`Pointwise ~op_label:"r" ~op_body:assign ~grad_body
    
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
      if id >= !first_session_id && id < Ocannl_runtime.Node.global.unique_id then
        "get_root: Node "^Int.to_string id^" is a subformula"
      else if id >= Ocannl_runtime.Node.global.unique_id then
        "get_root: Node "^Int.to_string id^" has not been created yet"
      else if id < 1 then "get_root: Node IDs start from 1"
      else
        "get_root: Node "^Int.to_string id^" is outside the current session" in
    raise @@ Session_error (msg, None)

let get_node id =
  let open Ocannl_runtime.Node in
  match Caml.Hashtbl.find_opt global.node_store id with
  | Some r -> r
  | None ->
    let msg = 
      if id >= global.unique_id then
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
    rectangles, possibly repeated (screens). *)
type array_print_style =
[ `Default
(** The inner rectangles comprise both an input and an output axis, if available. Even if there are only
    1, 2 or 3 input plus output axes, the batch axes are only output as the vertical direction of the
    outer rectangle, and/or the screens. At least one batch axis is output, when available.
    The outer rectangle comprises both an input and an output axis, when both inputs and outputs have
    2 or more axes. If there are no input axes, the last two output axes form the inner rectangles.
    The axes that couldn't be output are printed at position/dimension [0]. *)
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
    within the corresponding axes. Unspecified axes are printed at position [0]. *)
]

let print_formula ~with_grad ~with_code (style: array_print_style) m =
  assert (m.node_id = m.comp_node.id);
  let sh = m.shape in
  Stdio.print_endline @@ "["^Int.to_string m.node_id^"] "^m.comp_node.label^": "^
                         Shape.to_string_hum sh;
  let indices =
    match style with
    | `Default ->
      let axes = Shape.axis_keys_to_idcs sh |> Map.map ~f:(fun _ -> 0) in
      let num_inputs_plus_outputs =
        List.([sh.input; sh.output] |> map ~f:Shape.list_of_dims |> map ~f:length |> reduce_exn ~f:(+)) in
      let axes = Map.change axes {in_axes=Input; from_end=1} ~f:(Option.map ~f:(fun _ -> -1)) in
      let axes =
        if Map.mem axes {in_axes=Input; from_end=1}
        then Map.change axes {in_axes=Output; from_end=1} ~f:(Option.map ~f:(fun _ -> -3))
        else Map.change axes {in_axes=Output; from_end=1} ~f:(Option.map ~f:(fun _ -> -1)) in
      let axes = Map.change axes {in_axes=Input; from_end=2} ~f:(Option.map ~f:(fun _ -> -2)) in
      let axes =
        if Map.mem axes {in_axes=Input; from_end=2}
        then Map.change axes {in_axes=Output; from_end=2} ~f:(Option.map ~f:(fun _ -> -4))
        else if Map.mem axes {in_axes=Input; from_end=1}
        then Map.change axes {in_axes=Output; from_end=2} ~f:(Option.map ~f:(fun _ -> -2))
        else Map.change axes {in_axes=Output; from_end=2} ~f:(Option.map ~f:(fun _ -> -3)) in
      let remaining = Stack.of_list @@ List.filter ~f:(Map.mem axes) @@
        Shape.AxisKey.[{in_axes=Output; from_end=3}; {in_axes=Output; from_end=4};
                       {in_axes=Input; from_end=3}; {in_axes=Input; from_end=4};
                       {in_axes=Output; from_end=5}; {in_axes=Input; from_end=5}] in
      let axes =
        if Stack.is_empty remaining then axes
        else Map.change axes (Stack.pop_exn remaining) ~f:(Option.map ~f:(fun _ -> -4)) in
      let axes =
        if num_inputs_plus_outputs > 3
        then Map.change axes {in_axes=Batch; from_end=1} ~f:(Option.map ~f:(fun _ -> -5))
        else Map.change axes {in_axes=Batch; from_end=1} ~f:(Option.map ~f:(fun _ -> -4)) in
      let axes =
        if num_inputs_plus_outputs > 3
        then axes
        else Map.change axes {in_axes=Batch; from_end=2} ~f:(Option.map ~f:(fun _ -> -5)) in
      let axes =
        if Map.mem axes {in_axes=Batch; from_end=1} || Stack.is_empty remaining
        then axes
        else Map.change axes (Stack.pop_exn remaining) ~f:(Option.map ~f:(fun _ -> -5)) in
      Shape.axis_map_to_dims_index axes
      
    | `N5_layout priorities ->
      let p_labels = Shape.(axis_labels_of_spec priorities).labels |>
                     Map.map ~f:(Fn.compose ((-) 5) Int.of_string) in
      Shape.axis_map_to_dims_index p_labels

    | `Label_layout label_idcs ->
      let inv_labels = Map.to_alist sh.axis_labels |> List.map ~f:(fun (a,b) -> b,a) |>
                       Map.of_alist (module String) in
      let inv_labels = match inv_labels with
        | `Duplicate_key l -> raise @@ Session_error ("`Label_layout found a repeating label: "^l, Some m)
        | `Ok inv_labels -> inv_labels in
      let idcs = List.map label_idcs ~f:(fun (l, i) ->
        match Map.find inv_labels l with Some axis -> axis, i | None ->
          raise @@ Session_error ("`Label_layout label not found in shape: "^l, Some m)) in
      Shape.axis_map_to_dims_index @@ Map.of_alist_exn (module Shape.AxisKey) idcs in
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
  let einsum = einsum
  let einsum1 = einsum
  let reset_value = reset_value
  let uniform_value = uniform_value
  let term = term
  let stop_broadcast = stop_broadcast
  let stop_gradient = stop_gradient
  let print_global_root = print_global_root
  let print_node = Ndarray.print_node
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
