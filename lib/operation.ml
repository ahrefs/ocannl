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
    if needs1 && needs2 then ParHint (grad1, grad2)
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
    if needs1 && needs2 then ParHint (grad1, grad2)
    else if needs1 then grad1
    else if needs2 then grad2
    else assert false in
  Formula.binop ~compose_op 
    ~op_label:(if Shape.equal_compose_type compose_op `Pointwise then "*." else "*")
    ~op_body ~grad_body

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
    if needs1 && needs2 then ParHint (grad1, grad2)
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

let pointpow p =
  let open Code in
  let op_body ~n ~n1 projections =
    Accum_unop {zero_out=false; accum=Skip_arg; op=ToPowOf p; lhs=v n; rhs=v n1; projections} in
  let grad_body ~n ~n1 projections =
    Accum_binop {zero_out=false; accum=Add; op=Relu_gate; lhs=g n1; rhs1=v n; rhs2=g n;
                 projections=(fun () -> Shape.backprop_unary @@ projections())} in
  let op_label = "**"^Float.(if is_integer p then Int.to_string @@ to_int p else to_string p) in
  Formula.unop ~transpose_op:`Pointwise ~op_label ~op_body ~grad_body

let float_to_label v = Float.to_string_hum ~strip_zero:true v

let number ?(axis_label="") c =
  (* Note: no axis label so that we do not conflict with user labels. *)
  Formula.term ~label:(float_to_label c) (Constant {output_dims=[1]; axis_labels=axis_label})
    ~init_op:(`Constant_of_value c)

let unconstrained_param ?init label =
  (* Note: no axis label so that we do not conflict with user labels. *)
  let init_op = match init with
  | None -> `Standard_uniform
  | Some c -> `Constant_of_value c in
  Formula.term ~label (Deduced_params `Not_constrained) ~init_op

let range ?(axis_label="") upto =
  Formula.term ~label:("0"^"..."^Int.to_string upto)
   (Constant {output_dims=[upto + 1]; axis_labels=axis_label}) ~init_op:`Range_over_offsets

let range_of_shape ?(axis_labels="") ?(batch_dims=[]) ?(input_dims=[]) ?(output_dims=[]) () =
  let spec =
    match batch_dims, input_dims with
    | [], [] -> Shape.Constant {output_dims; axis_labels}
    | _, [] -> Data {batch_dims; output_dims; axis_labels}
    | _, _ -> Transform {batch_dims; input_dims; output_dims; axis_labels} in
  let dims = Array.concat_map [|batch_dims; output_dims; input_dims|] ~f:Array.of_list in
  Formula.term ~label:("r"^NodeUI.dims_to_string dims) spec ~init_op:`Range_over_offsets

let ndarray ?(axis_labels="") ?label ?(batch_dims=[]) ?(input_dims=[]) ?(output_dims=[]) values =
  let spec =
    match label, batch_dims, input_dims with
    | Some _, [], _ -> Shape.Params {input_dims; output_dims; axis_labels}
    | None, [], [] -> Constant {output_dims; axis_labels}
    | None, _, _ -> Transform {batch_dims; input_dims; output_dims; axis_labels}
    | _, _, [] -> Data {batch_dims; output_dims; axis_labels}
    | _, _::_, _::_ ->
      let sh = {Shape.batch=Given batch_dims; input=Given input_dims; output=Given output_dims;
                deduce_output_from_input=`Not_constrained;
                axis_labels=(Shape.axis_labels_of_spec axis_labels).labels; node_id= -1} in
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
  Formula.term ~label spec ~init_op:(`Fixed_constant values)

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
let stop_broadcast m = Shape.set_dims_type m.Formula.shape Shape.fixed

(** [identity] introduces a new node, which is an identity in both the forward and backward pass. *)
let identity m =
  let grad_body ~n ~n1 projections = assign_op g ~n:n1 ~n1:n projections in
  Formula.(unop ~init_shape:m.shape ~transpose_op:`Pointwise ~op_label:"="
             ~op_body:(assign_op v) ~grad_body)

let one_over _m = failwith "Tensor inversion not implemented yet"
let one_over_dot m = pointpow (-1.) m

module O = struct
  let ( * ) = matmul
  let ( *. ) = pointmul
  let (+) = add
  let (!/) = relu
  let (!~) label = Formula.term ~label (Deduced_params `Not_constrained) ~init_op:`Standard_uniform
  let (!.) = number
  let (-) m1 m2 = m1 + !.(-1.) *. m2
  let (~-) m = !.(-1.) *. m
  let (/) m1 m2 = m1 * one_over m2
  let (/.) m1 m2 = m1 *. one_over_dot m2
end
      
module CLI = struct
  module FO = O
  let einsum = einsum
  let einsum1 = einsum1
  let term = Formula.term
  let number = number
  let unconstrained_param = unconstrained_param
  let range = range
  let range_of_shape = range_of_shape
  let ndarray = ndarray
  let stop_broadcast = stop_broadcast
  let stop_gradient = stop_gradient
end

module Summable = struct
  type nonrec t = Formula.t
  let (+) = add
  let zero = number 0.0
end
