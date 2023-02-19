(** Computational primitives for neural networks, integrating [Formula] with [Code]. *)

open Base

let g n: Code.data = {node_id=n.Ocannl_runtime.Node.id; field=`Grad}
let v n: Code.data = {node_id=n.Ocannl_runtime.Node.id; field=`Value}
let d n field: Code.data = {node_id=n.Ocannl_runtime.Node.id; field}

let add =
  let open Code in
  let op_body ~n ~n1 ~n2 projections =
    Accum_binop {zero_out=false; accum=Skip_arg; op=Add; lhs=v n; rhs1=v n1; rhs2=v n2; projections} in
  let grad_body ~n ~n1 ~n2 ~needs1 ~needs2 projections =
    let grad1 =
      Accum_unop {zero_out=false; accum=Add; op=Identity; lhs=g n1; rhs=g n;
                  projections=(fun () -> Shape.backprop1 @@ projections())} in
    let grad2 =
      Accum_unop {zero_out=false; accum=Add; op=Identity; lhs=g n2; rhs=g n;
                  projections=(fun () -> Shape.backprop2 @@ projections())} in
    if needs1 && needs2 then Seq (grad1, grad2)
    else if needs1 then grad1
    else if needs2 then grad2
    else assert false in
  Formula.binop ~compose_op:`Pointwise ~op_label:"+" ~op_body ~grad_body

let mul compose_op =
  let open Code in
  let op_body ~n ~n1 ~n2 projections =
    Accum_binop {zero_out=false; accum=Skip_arg; op=Mul; lhs=v n; rhs1=v n1; rhs2=v n2; projections} in
  let grad_body ~n ~n1 ~n2 ~needs1 ~needs2 projections =
    let grad1 =
      Accum_binop {zero_out=false; accum=Add; op=Mul; lhs=g n1; rhs1=g n; rhs2=v n2;
                   projections=(fun () -> Shape.backprop1 @@ projections())} in
    let grad2 =
      Accum_binop {zero_out=false; accum=Add; op=Mul; lhs=g n2; rhs1=g n; rhs2=v n1;
                      projections=(fun () -> Shape.backprop2 @@ projections())} in
    if needs1 && needs2 then Seq (grad1, grad2)
    else if needs1 then grad1
    else if needs2 then grad2
    else assert false in
  Formula.binop ~compose_op ~op_label:"*" ~op_body ~grad_body

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
  let open Code in
  let op_body ~n ~n1 ~n2 projections =
    Accum_binop {zero_out=true; accum=Add; op=Mul; lhs=v n; rhs1=v n1; rhs2=v n2;
                 projections} in
  let grad_body ~n ~n1 ~n2 ~needs1 ~needs2 projections =
    let grad1 =
      Accum_binop {zero_out=false; accum=Add; op=Mul; lhs=g n1; rhs1=g n; rhs2=v n2;
                      projections=(fun () -> Shape.backprop1 @@ projections())} in
    let grad2 =
      Accum_binop {zero_out=false; accum=Add; op=Mul; lhs=g n2; rhs1=g n; rhs2=v n1;
                      projections=(fun () -> Shape.backprop2 @@ projections())} in
    if needs1 && needs2 then Seq (grad1, grad2)
    else if needs1 then grad1
    else if needs2 then grad2
    else assert false in
  Formula.binop ~compose_op:(`Einsum spec) ~op_label:";=>" ~op_body ~grad_body

(** Similar to the explicit mode of [numpy.einsum], the unary variant. Can permute axes, extract diagonals,
    compute traces etc.

    Note that ["a->c"] from [numpy] is ["a=>c"] in OCaNNL, since ["->"] is used to separate the input
    and the output axes. *)
let einsum1 spec =
  let open Code in
  let op_body ~n ~n1 projections =
    Accum_unop {zero_out=true; accum=Add; op=Identity; lhs=v n; rhs=v n1; projections} in
  let grad_body ~n ~n1 projections =
    Accum_unop {zero_out=false; accum=Add; op=Identity; lhs=g n1; rhs=g n;
                projections=(fun () -> Shape.backprop_unary @@ projections())} in
  Formula.unop ~transpose_op:(`Permute spec) ~op_label:"=>" ~op_body ~grad_body

let relu =
  let open Code in
  let op_body ~n ~n1 projections =
    Accum_unop {zero_out=false; accum=Skip_arg; op=Relu; lhs=v n; rhs=v n1; projections} in
  let grad_body ~n ~n1 projections =
    Accum_binop {zero_out=false; accum=Add; op=Relu_gate; lhs=g n1; rhs1=v n; rhs2=g n;
                 projections=(fun () -> Shape.backprop_unary @@ projections())} in
  Formula.unop ~transpose_op:`Pointwise ~op_label:"r" ~op_body ~grad_body

let reset_value c ~n field _shape =
  Code.Reset {tensor=d n field; reset_op=`Constant_of_value c}

let reset_range ~n field _shape =
  Code.Reset {tensor=d n field; reset_op=`Range_over_offsets}

let reset_values arr ~n field _shape =
  Code.Reset {tensor=d n field; reset_op=`Fixed_constant arr}

let float_to_label v = Float.to_string_hum ~strip_zero:true v

let number ?(axis_label="") c =
  (* Note: no axis label so that we do not conflict with user labels. *)
  Formula.term ~label:(float_to_label c) (Constant {output_dims=[1]; axis_labels=axis_label})
    ~init_body:(reset_value c)

let range ?(axis_label="") upto =
  Formula.term ~label:("0"^"..."^Int.to_string upto)
   (Constant {output_dims=[upto + 1]; axis_labels=axis_label}) ~init_body:reset_range

let range_of_shape ?(axis_labels="") ?(batch_dims=[]) ?(input_dims=[]) ?(output_dims=[]) () =
  let spec =
    match batch_dims, input_dims with
    | [], [] -> Shape.Constant {output_dims; axis_labels}
    | _, [] -> Data {batch_dims; output_dims; axis_labels}
    | _, _ -> Transform {batch_dims; input_dims; output_dims; axis_labels} in
  let dims = Array.concat_map [|batch_dims; output_dims; input_dims|] ~f:Array.of_list in
  Formula.term ~label:("r"^NodeUI.dims_to_string dims) spec ~init_body:reset_range

let ndarray ?(axis_labels="") ?label ?(batch_dims=[]) ?(input_dims=[]) ?(output_dims=[]) values =
  let spec =
    match label, batch_dims, input_dims with
    | Some _, [], _ -> Shape.Params {input_dims; output_dims; axis_labels}
    | None, [], [] -> Constant {output_dims; axis_labels}
    | None, _, _ -> Transform {batch_dims; input_dims; output_dims; axis_labels}
    | _, _, [] -> Data {batch_dims; output_dims; axis_labels}
    | _, _::_, _::_ ->
      let sh = {Shape.batch=Given batch_dims; input=Given input_dims; output=Given output_dims;
                deduce_output_from_input=`Not_deduced;
                axis_labels=(Shape.axis_labels_of_spec axis_labels).labels} in
      raise @@
      Shape.Shape_error ("Operation.ndarray: cannot provide all of [label], [batch_dims] and [input_dims]",
                         sh, sh) in
  let label =
    match label with
    | Some label -> label
    | None ->
      Caml.Format.pp_set_geometry Caml.Format.str_formatter
        ~max_indent:(!Formula.max_sublabel_length) ~margin:(!Formula.max_sublabel_length*2);
      let (!) = Array.of_list in
      let dims = Array.concat [!batch_dims; !output_dims; !input_dims] in
      let ndarr = Ocannl_runtime.Node.create_ndarray Single dims (`Fixed_constant values) in
      let (!) = List.length in
      NodeUI.pp_tensor_inline ~num_batch_axes: !batch_dims ~num_output_axes: !output_dims
        ~num_input_axes: !input_dims Caml.Format.str_formatter ndarr;
      Caml.Format.flush_str_formatter() in
  let label =
    if String.contains label '\n' then
      "c"^(NodeUI.dims_to_string @@ Array.concat_map [|batch_dims; output_dims; input_dims|] ~f:Array.of_list)
    else label in
  Formula.term ~label spec ~init_body:(reset_values values)

let uniform_value ~n field shape: Code.t =
  Code.(Create {tensor=d n field; dims=(fun () -> Shape.to_dims shape);
                init_op=`Standard_uniform})

let assign ~lhs ~rhs projections =
  let open Code in
  Accum_unop {zero_out=false; accum=Skip_arg; op=Identity; lhs; rhs; projections}

let assign_op field ~n ~n1 projections = assign ~lhs:(field n) ~rhs:(field n1) projections

(** A [stop_gradient] is an identity in the forward pass and a no-op in the backprop pass. *)
let stop_gradient =
  let grad_body ~n:_ ~n1:_ _projections = Code.Noop in
  Formula.unop ~transpose_op:`Pointwise ~op_label:"stop_grad" ~op_body:(assign_op v) ~grad_body

(** A [stop_broadcast] mutates the partially-inferred shape of a formula in-place, substituting-in
    a [Fixed] marker on the dimensions. This way we avoid introducing a new node. *)
let stop_broadcast m =
  let open Formula in
  let sh = m.shape in
  sh.batch <- Fixed (Shape.list_of_dims sh.batch);
  sh.batch <- Fixed (Shape.list_of_dims sh.batch);
  sh.batch <- Fixed (Shape.list_of_dims sh.batch);
  m
    
(** [identity] introduces a new node, which is an identity in both the forward and backward pass. *)
let identity m =
  let grad_body ~n ~n1 projections = assign_op g ~n:n1 ~n1:n projections in
  Formula.(unop ~init_shape:m.shape ~transpose_op:`Pointwise ~op_label:"="
             ~op_body:(assign_op v) ~grad_body)
    
module O = struct
  let ( * ) = matmul
  let ( *. ) = pointmul
  let (+) = add
  let (!/) = relu
  let (!~) label = Formula.term ~label (Deduced_params `Not_deduced) ~init_body:uniform_value
  let (!.) = number
  let (-) m1 m2 = m1 + !.(-1.) * m2
end

let get_root id =
  let open Formula in
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
  match Hashtbl.find global.node_store id with
  | Some r -> r
  | None ->
    let msg = 
      if id >= global.unique_id then
        "get_node: Node "^Int.to_string id^" has not been created yet"
      else if id < 1 then "get_root: Node IDs start from 1"
      else
        "get_node: Node "^Int.to_string id^" has been removed or lives on a different machine" in
    raise @@ Formula.Session_error (msg, None)

(** *** Printing. *** *)

(** We print out up to 5 axes when printing an [Code], as a grid (outer rectangle) of (inner)
    rectangles, possibly repeated (screens). *)
type array_print_style =
[ `Default
(** The inner rectangles comprise both an input and an output axis, if available. Similarly,
    the outer rectangle comprises a second-from-end input axis and a second-from-end output axis,
    if available. At least one batch axis is output, when available.
    The axes that couldn't be output are printed at position/dimension [0]. *)
| `N5_layout of string
(** The string should provide exclusively non-negative integer pseudo-labels. The numbers [0]-[4] represent
    the priorities of the axes to be printed out, where the priorities correspond to, from highest:
    horizontal, vertical direction of the inner rectangle, horizontal, vertical direction of the outer
    rectangle, repetition (see also [NodeUI.pp_print]). The numbers [n >= 5] stand for the actual
    positions [n - 5] within the corresponding axes. *)
| `Label_layout of (string * int) list
(** The association from axis labels to integers. The negative numbers [-5] to [-1] represent
    the priorities of the axes to be printed out, where the priorities correspond to, from highest:
    horizontal, vertical direction of the inner rectangle, horizontal, vertical direction of the outer
    rectangle, repetition (as above). The numbers [n >= 0] stand for the actual positions
    within the corresponding axes. Unspecified axes are printed at position [0]. *)
| `Inline
(** The tensors are printed linearly, in a bracketed manner, optionally prefixed with the labels
    specification. Note that the syntax causes ambiguity for 1-dimensional input axes (underscores are
    used for axes without explicit labels); when there is a 1-dimensional input axis, we output
    the labels specification even if there are no axis labels as a way to display the number of axes.
    The axis nesting is right-to-left (rightmost is innermost).
    The input axes are innermost and the batch axes outermost. The input axes use [,] as a separator
    and [()] as axis delimiters, but the delimiter for the outermost (i.e. leftmost) axis is omitted.
    The output axes use [;] as a separator and [[]] as axis delimiters (obligatory).
    The batch axes use [;] as a separator and [[||]] as axis delimiters (obligatory). *)
]

let print_formula ~with_grad ~with_code (style: array_print_style) m =
  let open Formula in
  let sh = m.shape in
  let prefix =
    "["^Int.to_string m.node_id^"] "^m.comp_node.label^": shape "^
    Shape.to_string_hum ~style:`Axis_number_and_size sh^" " in
  let indices =
    match style with
    | `Default ->
      let axes = Shape.axis_keys_to_idcs sh |> Map.map ~f:(fun _ -> 0) in
      let occupied = Array.create ~len:5 false in
      let set_occu prio = occupied.(prio + 5) <- true; prio in
      let occu prio = occupied.(prio + 5) in
      let num_input_axes = List.length Shape.(list_of_dims @@ dims_of_kind Input sh) in
      let remaining = Stack.of_list @@ List.filter ~f:(Map.mem axes) @@
        Shape.AxisKey.[
          {in_axes=Input; from_end=1}; {in_axes=Output; from_end=1};
          {in_axes=Input; from_end=2}; {in_axes=Output; from_end=2};
          (if num_input_axes > 1 then {in_axes=Batch; from_end=1} else {in_axes=Output; from_end=3});
          {in_axes=Batch; from_end=1}; {in_axes=Batch; from_end=2};
          {in_axes=Input; from_end=3}; {in_axes=Output; from_end=3};
          {in_axes=Input; from_end=4}; {in_axes=Output; from_end=4};
          {in_axes=Input; from_end=5}; {in_axes=Output; from_end=5} ] in
      let rec loop offset axes =
        if Stack.is_empty remaining || offset > 5 then axes
        else if Fn.non occu ~-offset
        then
          loop (offset + 1) @@ Map.change axes (Stack.pop_exn remaining)
            ~f:(Option.map ~f:(fun _ -> set_occu ~-offset))
        else loop (offset + 1) axes in
      let axes = loop 1 axes in
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
      Shape.axis_map_to_dims_index @@ Map.of_alist_exn (module Shape.AxisKey) idcs
    | `Inline -> [||] in
  let needs_spec = Fn.non Map.is_empty sh.axis_labels ||
                   Shape.(List.exists ~f:((=) 1) @@ list_of_dims @@ dims_of_kind Input sh) in
  let labels = Shape.axis_map_to_dims_index ~default:"" sh.axis_labels in
  let labels_spec = if needs_spec then Some (Shape.to_string_hum ~style:`Only_labels sh) else None in
  let num_axes kind = List.length Shape.(list_of_dims @@ dims_of_kind kind sh) in
  let num_batch_axes = num_axes Shape.AxisKey.Batch in
  let num_input_axes = num_axes Shape.AxisKey.Input in
  let num_output_axes = num_axes Shape.AxisKey.Output in
  (match style with
   | `Inline ->
     NodeUI.pp_tensor_inline Caml.Format.std_formatter ~num_batch_axes ~num_input_axes ~num_output_axes
       ?labels_spec m.comp_node.value
   | _ -> NodeUI.pp_tensor Caml.Format.std_formatter ~prefix ~labels ~indices m.comp_node.value);
  if with_grad then (
    match style with
    | `Inline ->
      NodeUI.pp_tensor_inline Caml.Format.std_formatter ~num_batch_axes ~num_input_axes ~num_output_axes
        ?labels_spec m.comp_node.grad
    | _ -> NodeUI.pp_tensor Caml.Format.std_formatter ~prefix:(prefix^" Gradient ") ~labels ~indices
             m.comp_node.grad);
  if with_code then (
    (match m.forward_body with
     | Noop -> ()
     | fwd_code ->
       Stdio.print_endline "Current forward body:";
       Stdio.print_endline @@ Code.sprint_code fwd_code);
    (match m.backprop_body with
     | Noop -> ()
     | bwd_code ->
       Stdio.print_endline "Current backprop body:";
       Stdio.print_endline @@ Code.sprint_code bwd_code)
  );
  Stdio.printf "\n%!"

let print_global_root ~with_grad ~with_code (style: array_print_style) root =
  let open Formula in
  print_formula ~with_grad ~with_code:false style root.formula;
  if with_code then (
    (match root.forward_code with
     | None -> ()
     | Some fwd ->
       Stdio.print_endline "Forward:";
       Stdio.print_endline @@ Code.sprint_program fwd);
    (match root.backprop_code with
     | None -> ()
     | Some bwd ->
       Stdio.print_endline "Backprop:";
       Stdio.print_endline @@ Code.sprint_program bwd)
  );
  Stdio.printf "\n%!"

let print_global_roots ~with_grad ~with_code (style: array_print_style) =
  let open Formula in
  List.iter (Map.to_alist ~key_order:`Increasing !global_roots) ~f:(fun (node_id, root) ->
      assert (node_id = root.formula.node_id);
      print_global_root ~with_grad ~with_code style root)

(** *** Session management. *** *)

let refresh_session ?(with_debug=true) ?(regenerate=false) ?(reinit=false) ?(run=true)
    ?(force_no_init=false) () =
  let open Formula in
  if force_no_init && (regenerate || reinit || run) then
    invalid_arg "refresh_session: set other triggers to false when using force_no_init";
  (* Initialization and the forward processing. *)
  List.iter (Map.to_alist ~key_order:`Increasing !global_roots) ~f:(fun (_node_id, root) ->
    let m = root.formula in
    if regenerate || Option.is_none root.forward_code || Option.is_none root.backprop_code then (
      Sequence.iter root.subtree_shape_updates ~f:Shape.propagate_shapes;
      let forward_prog, backprop_prog = get_toplevel m in
       root.forward_code <- Some forward_prog;
       root.formula.comp_node.forward <- None;
       root.backprop_code <- Some backprop_prog;
       root.formula.comp_node.backprop <- None
    );
    if not force_no_init && 
        (reinit || Option.is_none root.formula.comp_node.forward) then (
      try
        let contents = ExecAsOCaml.load_native ~with_debug (Option.value_exn root.forward_code) in
        match contents, m.comp_node.forward with
        | Some contents, Some forward ->
          m.comp_node.forward <-
            Some (fun () -> try forward() with error ->
                ExecAsOCaml.handle_error "Forward error:" ~formula:m ~contents error)
        | Some contents, None ->
          let msg = "refresh_session: error loading `forward`: routine not set in code:\n"^contents in
          raise @@ Session_error (msg, Some m)
        | _, None ->
          failwith ("refresh_session: error loading `forward`: routine not set"^
                    (if with_debug then "" else " (use `~with_debug:true` for more information)"))
        | _ -> ()
      with Session_error (msg, None) ->
        let msg = "Forward init error: "^msg in
        raise @@ Session_error (msg, Some m);
    );
    if not force_no_init && 
        (reinit || Option.is_none root.formula.comp_node.backprop) then (
      try
        let contents = ExecAsOCaml.load_native ~with_debug (Option.value_exn root.backprop_code) in
        match contents, m.comp_node.backprop with
        | Some contents, Some backprop ->
          m.comp_node.backprop <-
            Some (fun () ->
                try backprop() with error -> ExecAsOCaml.handle_error "Backprop error:" ~formula:m ~contents error)
        | Some contents, None ->
          let msg = "refresh_session: error loading `backprop`: routine not set in code:\n"^contents in
          raise @@ Session_error (msg, Some m)
        | _, None ->
          failwith ("refresh_session: error loading `backprop`: routine not set"^
                    (if with_debug then "" else " (use `~with_debug:true` for more information)"))
        | _ -> ()
      with Session_error (msg, None) ->
        Stdio.print_endline "Forward code (context for backprop init error):";
        Stdio.print_endline @@ Code.sprint_program @@ Option.value_exn root.forward_code;
        let msg = "Backprop init error: "^msg in
        raise @@ Session_error (msg, Some m);
    );
    if run then match root.formula.comp_node.forward with
      | Some forward -> forward()
      | None -> assert false
  );
  (* The backpropagation. *)
  if run then
    List.iter (Map.to_alist ~key_order:`Decreasing !global_roots) ~f:(fun (_node_id, root) ->
      Option.value_exn root.formula.comp_node.backprop ())

(** Discards global roots, rolls back [Node.state.unique_id] to [Formula.first_session_id], discards
    the corresponding elements from [Node.state.node_store]. *)
let drop_session() =
  Formula.global_roots := Map.empty (module Int);
  for i = !Formula.first_session_id to Ocannl_runtime.Node.global.unique_id - 1 do
    Hashtbl.remove Ocannl_runtime.Node.global.node_store i
  done;
  Ocannl_runtime.Node.global.unique_id <- !Formula.first_session_id

(** Discards global roots, advances [Formula.first_session_id] to [Node.state.unique_id]. *)
let close_session() =
  Formula.first_session_id := Ocannl_runtime.Node.global.unique_id;
  Formula.global_roots := Map.empty (module Int)

      
module CLI = struct
  module FO = O
  let einsum = einsum
  let einsum1 = einsum1
  let reset_value = reset_value
  let uniform_value = uniform_value
  let term = Formula.term
  let number = number
  let range = range
  let range_of_shape = range_of_shape
  let ndarray = ndarray
  let stop_broadcast = stop_broadcast
  let stop_gradient = stop_gradient
  let refresh_session = refresh_session
  let drop_session = drop_session
  let close_session = close_session
  let print_global_root = print_global_root
  let print_node = NodeUI.print_node
  let max_sublabel_length = Formula.max_sublabel_length
  let print_formula = print_formula
  let print_global_roots = print_global_roots
  let print_decimals_precision = NodeUI.print_decimals_precision
  let get_root = get_root
  let get_node = get_node
end

module Summable = struct
  type nonrec t = Formula.t
  let (+) = add
  let zero = number 0.0
end
