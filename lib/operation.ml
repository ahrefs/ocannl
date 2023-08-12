(** Computational primitives for neural networks, integrating [Formula] with [Code]. *)

open Base
module CDSL = Code.CDSL

let add =
  let open Code in
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd op_body ~(n : Code.node) ~(n1 : Code.node) ~(n2 : Code.node) ~projections = n =: n1 + n2 in
  let%nn_cd grad_body ~(n : Code.node) ~(n1 : Code.node) ~(n2 : Code.node) ~projections =
    n1.grad =+ n.grad || n2.grad =+ n.grad
  in
  Formula.binop ~compose_op:Pointwise_bin ~op_label:"+" ~op_body ~grad_body

let pointmul =
  let open Code in
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd op_body ~(n : Code.node) ~(n1 : Code.node) ~(n2 : Code.node) ~projections = n =: n1 * n2 in
  let%nn_cd grad_body ~(n : Code.node) ~(n1 : Code.node) ~(n2 : Code.node) ~projections =
    n1.grad =+ n.grad * n2 || n2.grad =+ n1 * n.grad
  in
  Formula.binop ~compose_op:Pointwise_bin ~op_label:"*." ~op_body ~grad_body

(* N1: AxB, N2 BxC, N: AxC, A: output of N1, B: input/output of N1/N2, C: input of N2.
   Although the matrix algebra would require that we insert additional transposes in gradient multiplies:
   AxB = AxC * CxB = AxC * (BxC)^T -> N1g += Ng * N2v^T,
   BxC = BxA * AxC = (AxB)^T * AxC -> N2g += N1v^T * Ng,
   in our setup there is no transposing to do, since the projections produce correct indices for their
   corresponding matrices. *)

let matmul =
  let open Code in
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd op_body ~(n : Code.node) ~(n1 : Code.node) ~(n2 : Code.node) ~projections = n =:+ n1 * n2 in
  let%nn_cd grad_body ~(n : Code.node) ~(n1 : Code.node) ~(n2 : Code.node) ~projections =
    n1.grad =+ n.grad * n2 || n2.grad =+ n1 * n.grad
  in
  Formula.binop ~compose_op:Compose ~op_label:"*" ~op_body ~grad_body

(** Similar to the explicit mode of [numpy.einsum], the binary variant. Can compute various forms of
    matrix multiplication, inner and outer products, etc.

    Note that ["a,b->c"] from [numpy] is ["a;b=>c"] in OCANNL, since ["->"] is used to separate the input
    and the output axes. *)
let einsum ?desc_label spec =
  let open Code in
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd op_body ~(n : Code.node) ~(n1 : Code.node) ~(n2 : Code.node) ~projections = n =:+ n1 * n2 in
  let%nn_cd grad_body ~(n : Code.node) ~(n1 : Code.node) ~(n2 : Code.node) ~projections =
    n1.grad =+ n.grad * n2 || n2.grad =+ n1 * n.grad
  in
  Formula.binop ?desc_label ~compose_op:(Einsum spec) ~op_label:";=>" ~op_body ~grad_body

(** Similar to the explicit mode of [numpy.einsum], the unary variant. Can permute axes, extract diagonals,
    compute traces etc.

    Note that ["a->c"] from [numpy] is ["a=>c"] in OCANNL, since ["->"] is used to separate the input
    and the output axes. *)
let einsum1 ?desc_label spec =
  let open Code in
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd op_body ~(n : Code.node) ~(n1 : Code.node) ~projections = n =:+ n1 in
  let%nn_cd grad_body ~(n : Code.node) ~(n1 : Code.node) ~projections = n1.grad =+ n.grad in
  Formula.unop ?desc_label ~transpose_op:(Permute spec) ~op_label:"=>" ~op_body ~grad_body

let relu =
  let open Code in
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd op_body ~(n : Code.node) ~(n1 : Code.node) ~projections = n =: !/n1 ~projections in
  let%nn_cd grad_body ~(n : Code.node) ~(n1 : Code.node) ~projections = n1.grad =+ n -?/ n.grad in
  Formula.unop ~transpose_op:Pointwise_un ~op_label:"r" ~op_body ~grad_body

let subtensor_label ~over_kind ~from_left ~other_axes_pointwise =
  let kind_spec = match over_kind with Shape.AxisKey.Batch -> "|" | Input -> "/" | Output -> "-" in
  let pointwise_spec = if other_axes_pointwise then "." else "^" in
  if from_left then "@" ^ pointwise_spec ^ kind_spec else "@" ^ kind_spec ^ pointwise_spec

let dynamic_subtensor ?indexed_dims ~over_kind ~from_left ~other_axes_pointwise =
  let open Code in
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd op_body ~(n : Code.node) ~(n1 : Code.node) ~(n2 : Code.node) ~projections = n =: n1 -@> n2 in
  let%nn_cd grad_body ~(n : Code.node) ~(n1 : Code.node) ~(n2 : Code.node) ~projections =
    (* [projections] tracks the dynamic indexing for [n] (and not [n1]) as a slice.
       [-@>] simply means [Arg1]: take the first argument, ignore the second argument. *)
    n1.grad =+ n.grad -@> n2
  in
  let compose_op = Shape.Dynamic_index { over_kind; from_left; other_axes_pointwise; indexed_dims } in
  let op_label = subtensor_label ~over_kind ~from_left ~other_axes_pointwise in
  Formula.binop ~compose_op ~op_label ~op_body ~grad_body

module NFO_without_pow = struct
  let ( * ) = matmul ~is_form:false
  let ( *. ) = pointmul ~is_form:false
  let ( + ) = add ~is_form:false
  let ( !/ ) = relu ~is_form:false
  let ( !. ) = Formula.number ~is_form:false
  let ( !.. ) ?desc_label i = Formula.number ?desc_label ~is_form:false @@ Float.of_int i
  let ( - ) ?desc_label m1 m2 = ( + ) ?desc_label m1 (!.(-1.) *. m2)
  let ( ~- ) ?desc_label m = ( *. ) ?desc_label !.(-1.) m

  let ( @.| ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:true ~other_axes_pointwise:true ~is_form:false

  let ( @./ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:true ~other_axes_pointwise:true ~is_form:false

  let ( @.- ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:true ~other_axes_pointwise:true
      ~is_form:false

  let ( @^| ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:true ~other_axes_pointwise:false
      ~is_form:false

  let ( @^/ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:true ~other_axes_pointwise:false
      ~is_form:false

  let ( @^- ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:true ~other_axes_pointwise:false
      ~is_form:false

  let ( @|. ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:false ~other_axes_pointwise:true
      ~is_form:false

  let ( @/. ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:false ~other_axes_pointwise:true
      ~is_form:false

  let ( @-. ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:false ~other_axes_pointwise:true
      ~is_form:false

  let ( @|^ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:false ~other_axes_pointwise:false
      ~is_form:false

  let ( @/^ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:false ~other_axes_pointwise:false
      ~is_form:false

  let ( @-^ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:false ~other_axes_pointwise:false
      ~is_form:false
end

let rec pointpow ?desc_label ~is_form p m1 : Formula.t =
  let module NFDSL = struct
    module O = NFO_without_pow
  end in
  let open Code in
  let p_f = Formula.number ~is_form p in
  let%nn_cd op_body ~(n : Code.node) ~(n1 : Code.node) ~(n2 : Code.node) ~projections =
    n =: n1 ** n2 ~projections
  in
  let%nn_cd grad_body =
    if not is_form then fun ~n:_ ~n1:_ ~n2:_ ~projections:_ -> Noop
    else if Float.equal p 2.0 then fun ~(n : Code.node) ~(n1 : Code.node) ~n2:_ ~projections ->
      n1.grad =+ p_f *. m1 * n.grad
    else fun ~(n : Code.node) ~(n1 : Code.node) ~n2:_ ~projections ->
      n1.grad =+ p_f *. (m1 **. (p -. 1.)) * n.grad
  in
  Formula.binop ?desc_label ~compose_op:Pointwise_bin ~op_label:"**." ~op_body ~grad_body ~is_form m1 p_f

let range ?desc_label ~is_form ?axis_label upto =
  Formula.term ?desc_label ~is_form ~needs_gradient:false
    ~label:("0" ^ "..." ^ Int.to_string upto)
    ~batch_dims:[] ~input_dims:[]
    ~output_dims:[ Shape.dim (upto + 1) ]
    ?axis_labels:axis_label ~init_op:Range_over_offsets ()

let range_of_shape ?desc_label ~is_form ?(batch_dims = []) ?(input_dims = []) ?(output_dims = []) ?axis_labels
    () =
  let dims = Array.concat_map [| batch_dims; output_dims; input_dims |] ~f:Array.of_list in
  Formula.term ?desc_label ~is_form ~needs_gradient:false ~batch_dims ~input_dims ~output_dims ?axis_labels
    ~label:("r" ^ Shape.dims_to_string dims)
    ~init_op:Range_over_offsets ()

(** In {!Formula.term} the omitted axes are {!Shape.Unknown} -- to be inferred, here they are known and empty.  *)
let data ?desc_label ?axis_labels ?(needs_gradient = false) ~label ?(batch_dims = []) ?(input_dims = [])
    ?(output_dims = []) fetch_op =
  if List.for_all ~f:List.is_empty [ batch_dims; input_dims; output_dims ] then
    invalid_arg "Operation.result: data and the `%nn_dt` syntax do not support shape inference, specify dims";
  Formula.term ?desc_label ~label ~is_form:true ~needs_gradient ~batch_dims ~input_dims ~output_dims
    ?axis_labels ~fetch_op ()

(** Non-form computations that happen at the end (potentially in parallel). *)
let result ?desc_label ?axis_labels ~label ?(batch_dims = []) ?(input_dims = []) ?(output_dims = [])
    postprocess_op =
  if List.for_all ~f:List.is_empty [ batch_dims; input_dims; output_dims ] then
    invalid_arg
      "Operation.result: results and the `%nn_rs` syntax do not support shape inference, specify dims";
  Formula.term ?desc_label ~label ~is_form:false ~needs_gradient:false ~batch_dims ~input_dims ~output_dims
    ?axis_labels ~postprocess_op ()

let assign =
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd assign ~(lhs : Node.tensor_ptr) ~(rhs : Node.tensor_ptr) ~projections = lhs =: rhs ~projections in
  assign

let assign_op field ~(n : Code.node) ~(n1 : Code.node) ~projections =
  assign ~lhs:(field n) ~rhs:(field n1) ~projections

(** A [stop_gradient] is an identity in the forward pass and a no-op in the backprop pass. *)
let stop_gradient =
  let grad_body ~n:_ ~n1:_ ~projections:_ = Code.Noop in
  let op_body = assign_op @@ Code.CDSL.data_of_node Value in
  Formula.unop ~transpose_op:Pointwise_un ~op_label:"stop_grad" ~op_body ~grad_body ~is_form:true

(** A [stop_broadcast] mutates the partially-inferred shape of a formula in-place, substituting-in
    a [Fixed] marker on the dimensions. This way we avoid introducing a new node. *)
let stop_broadcast m = Shape.set_dims_type m.Formula.shape Shape.fixed

(** [identity] introduces a new node, which is an identity in both the forward and backward pass. *)
let identity ?desc_label ~is_form m =
  let grad_body ~(n : Code.node) ~(n1 : Code.node) = assign_op (Code.CDSL.data_of_node Grad) ~n:n1 ~n1:n in
  let op_body = assign_op @@ Code.CDSL.data_of_node Value in
  Formula.(
    unop ?desc_label ~init_shape:m.shape ~transpose_op:Pointwise_un ~op_label:"=" ~op_body ~grad_body ~is_form)

module O = struct
  let ( * ) = matmul ~is_form:true
  let ( *. ) = pointmul ~is_form:true
  let ( + ) = add ~is_form:true
  let ( **. ) ?desc_label base exp = pointpow ?desc_label exp base ~is_form:true
  let ( !/ ) = relu ~is_form:true
  let ( !~ ) ?desc_label label = Formula.params ?desc_label label
  let ( !. ) = Formula.number ~is_form:true
  let ( !.. ) ?desc_label i = Formula.number ?desc_label ~is_form:true @@ Float.of_int i
  let ( - ) ?desc_label m1 m2 = ( + ) ?desc_label m1 (!.(-1.) *. m2)
  let ( ~- ) ?desc_label m = ( *. ) ?desc_label !.(-1.) m
  let ( /. ) ?desc_label m1 m2 = ( *. ) ?desc_label m1 (m2 **. -1.0)

  let ( @.| ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:true ~other_axes_pointwise:true ~is_form:true

  let ( @./ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:true ~other_axes_pointwise:true ~is_form:true

  let ( @.- ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:true ~other_axes_pointwise:true ~is_form:true

  let ( @^| ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:true ~other_axes_pointwise:false ~is_form:true

  let ( @^/ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:true ~other_axes_pointwise:false ~is_form:true

  let ( @^- ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:true ~other_axes_pointwise:false
      ~is_form:true

  let ( @|. ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:false ~other_axes_pointwise:true ~is_form:true

  let ( @/. ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:false ~other_axes_pointwise:true ~is_form:true

  let ( @-. ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:false ~other_axes_pointwise:true
      ~is_form:true

  let ( @|^ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:false ~other_axes_pointwise:false
      ~is_form:true

  let ( @/^ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:false ~other_axes_pointwise:false
      ~is_form:true

  let ( @-^ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:false ~other_axes_pointwise:false
      ~is_form:true
end

module FDSL = struct
  include Formula.FDSL
  module O = O

  let einsum ?desc_label s = einsum ?desc_label s ~is_form:true
  let einsum1 ?desc_label s = einsum1 ?desc_label s ~is_form:true
  let range = range ~is_form:true
  let range_of_shape = range_of_shape ~is_form:true
  let data = data
  let stop_broadcast = stop_broadcast
  let stop_gradient = stop_gradient

  let init_const ~l ?(b = []) ?(i = []) ?(o = []) cs =
    term ~label:l ~needs_gradient:false ~batch_dims:b ~input_dims:i ~output_dims:o ~init_op:(Constant_fill cs)
      ()

  let init_param ~l ?(b = []) ?(i = []) ?(o = []) cs =
    term ~label:l ~needs_gradient:true ~batch_dims:b ~input_dims:i ~output_dims:o ~init_op:(Constant_fill cs)
      ()
end

module NFO = struct
  include NFO_without_pow

  let ( **. ) ?desc_label base exp = pointpow ?desc_label exp base ~is_form:false
  let ( /. ) ?desc_label m1 m2 = ( *. ) ?desc_label m1 (m2 **. -1.0)
end

module NFDSL = struct
  include Formula.NFDSL
  module O = NFO

  let einsum ?desc_label s = einsum ?desc_label s ~is_form:false
  let einsum1 ?desc_label s = einsum1 ?desc_label s ~is_form:false
  let term = Formula.term ~is_form:false
  let result = result
  let range = range ~is_form:false
  let range_of_shape = range_of_shape ~is_form:false
end
