(** Computational primitives for neural networks, integrating [Formula] with [Code]. *)

open Base

let g n: Code.data = {node_id=n.Ocannl_runtime.Node.id; field=`Grad}
let v n: Code.data = {node_id=n.Ocannl_runtime.Node.id; field=`Value}
let d n field: Code.data = {node_id=n.Ocannl_runtime.Node.id; field}
let vi node_id: Code.data = {node_id; field=`Value}

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
    (* [Formula] will not invoke [grad_body] if it does not need at least one gradient. *)
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

let float_to_label v = Float.to_string_hum ~strip_zero:true v

let number ~is_form ?(axis_label="") c =
  (* Note: no axis label so that we do not conflict with user labels. *)
  Formula.term ~label:(float_to_label c) ~is_form
    (Constant {output_dims=[1]; axis_labels=axis_label}) ~init_op:(`Constant_of_value c)

let rec pointpow ~is_form p m1: Formula.t =
  let open Code in
  let p_f = number ~is_form p in
  let op_body ~n ~n1 ~n2 projections =
    Accum_binop {zero_out=false; accum=Skip_arg; op=ToPowOf; lhs=v n;
                 rhs1=v n1; rhs2=v n2; projections} in
  (* FIXME: dedicated is_form not needed? Formula should handle it? *)
  let grad_body =
    if not is_form then 
      fun ~n:_ ~n1:_ ~n2:_ ~needs1:_ ~needs2:_ _projections -> Noop
    else if Float.equal p 2.0 then
      let grad_f = pointmul ~is_form:false p_f m1 in
      fun ~n ~n1 ~n2:_ ~needs1 ~needs2 projections ->
        (* [Formula] will not invoke [grad_body] if it does not need at least one gradient. *)
        assert (needs1 && not needs2);
        Accum_binop {zero_out=false; accum=Add; op=Mul; lhs=g n1; rhs1=vi grad_f.node_id; rhs2=g n;
                     projections=(fun () -> Shape.backprop_unary @@ projections())}    
    else
      let grad_powf = pointpow ~is_form:false (p -. 1.) m1 in
      let grad_f = pointmul ~is_form:false p_f grad_powf in
      fun ~n ~n1 ~n2:_ ~needs1 ~needs2 projections ->
        assert (needs1 && not needs2);
        Accum_binop {zero_out=false; accum=Add; op=Mul; lhs=g n1; rhs1=vi grad_f.node_id; rhs2=g n;
                     projections=(fun () -> Shape.backprop_unary @@ projections())} in
  Formula.binop ~compose_op:`Pointwise ~op_label:"**." ~op_body ~grad_body ~is_form m1 p_f

let unconstrained_param ?init label =
  (* Note: no axis label so that we do not conflict with user labels. *)
  let init_op = match init with
  | None -> `Standard_uniform
  | Some c -> `Constant_of_value c in
  Formula.term ~is_form:true ~label (Deduced_params `Not_constrained) ~init_op

let range ~is_form ?(axis_label="") upto =
  Formula.term ~is_form ~label:("0"^"..."^Int.to_string upto)
   (Constant {output_dims=[upto + 1]; axis_labels=axis_label}) ~init_op:`Range_over_offsets

let range_of_shape ~is_form ?(axis_labels="") ?(batch_dims=[]) ?(input_dims=[]) ?(output_dims=[]) () =
  let spec =
    match batch_dims, input_dims with
    | [], [] -> Shape.Constant {output_dims; axis_labels}
    | _, [] -> Data {batch_dims; output_dims; axis_labels}
    | _, _ -> Transform {batch_dims; input_dims; output_dims; axis_labels} in
  let dims = Array.concat_map [|batch_dims; output_dims; input_dims|] ~f:Array.of_list in
  Formula.term ~is_form ~label:("r"^NodeUI.dims_to_string dims) spec ~init_op:`Range_over_offsets

let ndarray ~is_form ?(axis_labels="") ?label ?(batch_dims=[]) ?(input_dims=[]) ?(output_dims=[])
 values =
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
  Formula.term ~is_form ~label spec ~init_op:(`Fixed_constant values)

let given_dims_params ?(axis_labels="") ?(input_dims=[]) ?(output_dims=[]) label values =
  Formula.term ~is_form:true ~label (Params {input_dims; output_dims; axis_labels})
    ~init_op:(`Fixed_constant values)

let assign ~lhs ~rhs projections =
  let open Code in
  Accum_unop {zero_out=false; accum=Skip_arg; op=Identity; lhs; rhs; projections}

let assign_op field ~n ~n1 projections = assign ~lhs:(field n) ~rhs:(field n1) projections

(** A [stop_gradient] is an identity in the forward pass and a no-op in the backprop pass. *)
let stop_gradient =
  let grad_body ~n:_ ~n1:_ _projections = Code.Noop in
  Formula.unop ~transpose_op:`Pointwise ~op_label:"stop_grad" ~op_body:(assign_op v) ~grad_body
    ~is_form:true

(** A [stop_broadcast] mutates the partially-inferred shape of a formula in-place, substituting-in
    a [Fixed] marker on the dimensions. This way we avoid introducing a new node. *)
let stop_broadcast m = Shape.set_dims_type m.Formula.shape Shape.fixed

(** [identity] introduces a new node, which is an identity in both the forward and backward pass. *)
let identity ~is_form m =
  let grad_body ~n ~n1 projections = assign_op g ~n:n1 ~n1:n projections in
  Formula.(unop ~init_shape:m.shape ~transpose_op:`Pointwise ~op_label:"="
             ~op_body:(assign_op v) ~grad_body ~is_form)

module O = struct
  let ( * ) = matmul ~is_form:true
  let ( *. ) = pointmul ~is_form:true
  let (+) = add ~is_form:true
  let ( **. ) base exp = pointpow exp base ~is_form:true
  let (!/) = relu ~is_form:true
  let (!~) label =
   Formula.term ~label ~is_form:true (Deduced_params `Not_constrained) ~init_op:`Standard_uniform
  let (!.) = number ~is_form:true
  let (-) m1 m2 = m1 + !.(-1.) *. m2
  let (~-) m = !.(-1.) *. m
  let (/) m1 m2 = m1 * m2 **. (-1.0)
  let (/.) m1 m2 = m1 *. m2 **. (-1.0)
end
      
module CLI = struct
  module O = O
  let einsum s = einsum s ~is_form:true
  let einsum1 s = einsum1 s ~is_form:true
  let term = Formula.term ~is_form:true
  let number = number ~is_form:true
  let unconstrained_param = unconstrained_param
  let range = range ~is_form:true
  let range_of_shape = range_of_shape ~is_form:true
  let ndarray = ndarray ~is_form:true
  let stop_broadcast = stop_broadcast
  let stop_gradient = stop_gradient
end


module NFO = struct
  let ( * ) = matmul ~is_form:false
  let ( *. ) = pointmul ~is_form:false
  let (+) = add ~is_form:false
  let ( **. ) base exp = pointpow exp base ~is_form:false
  let (!/) = relu ~is_form:false
  let (!~) label =
   Formula.term ~label ~is_form:false (Deduced_params `Not_constrained) ~init_op:`Standard_uniform
  let (!.) = number ~is_form:false
  let (-) m1 m2 = m1 + !.(-1.) *. m2
  let (~-) m = !.(-1.) *. m
  let (/) m1 m2 = m1 * m2 **. (-1.0)
  let (/.) m1 m2 = m1 *. m2 **. (-1.0)
end

module NFCLI = struct
  module O = NFO
  let einsum s = einsum s ~is_form:false
  let einsum1 s = einsum1 s ~is_form:false
  let term = Formula.term ~is_form:false
  let number = number ~is_form:false
  let range = range ~is_form:false
  let range_of_shape = range_of_shape ~is_form:false
  let ndarray = ndarray ~is_form:false
  let stop_broadcast = stop_broadcast
  let stop_gradient = stop_gradient
end

module Summable = struct
  type nonrec t = Formula.t
  let (+) = add ~is_form:true
  let zero = number ~is_form:true 0.0
end
