(** Computational primitives for neural networks, integrating [Tensor] with [Code]. *)

open Base
module CDSL = Code.CDSL

let add =
  let open Code in
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd op_body ~(v : Code.node) ~(v1 : Code.node) ~(v2 : Code.node) ~projections = v =: v1 + v2 in
  let%nn_cd grad_body ~(v : Code.node) ~(v1 : Code.node) ~(v2 : Code.node) ~projections =
    g1 =+ g || g2 =+ g
  in
  Tensor.binop ~compose_op:Pointwise_bin ~op_label:"+" ~op_body ~grad_body

let pointmul =
  let open Code in
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd op_body ~(v : Code.node) ~(v1 : Code.node) ~(v2 : Code.node) ~projections = v =: v1 * v2 in
  let%nn_cd grad_body ~(v : Code.node) ~(v1 : Code.node) ~(v2 : Code.node) ~projections =
    g1 =+ g * v2 || g2 =+ v1 * g
  in
  Tensor.binop ~compose_op:Pointwise_bin ~op_label:"*." ~op_body ~grad_body

(* N1: AxB, N2 BxC, v: AxC, A: output of N1, B: input/output of N1/N2, C: input of N2.
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
  let%nn_cd op_body ~(v : Code.node) ~(v1 : Code.node) ~(v2 : Code.node) ~projections = v =:+ v1 * v2 in
  let%nn_cd grad_body ~(v : Code.node) ~(v1 : Code.node) ~(v2 : Code.node) ~projections =
    g1 =+ g * v2 || g2 =+ v1 * g
  in
  Tensor.binop ~compose_op:Compose ~op_label:"*" ~op_body ~grad_body

(** Similar to the explicit mode of [numpy.einsum], the binary variant. Can compute various forms of
    matrix multiplication, inner and outer products, etc.

    Note that ["a,b->c"] from [numpy] is ["a;b=>c"] in OCANNL, since ["->"] is used to separate the input
    and the output axes. *)
let einsum ?desc_label spec =
  let open Code in
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd op_body ~(v : Code.node) ~(v1 : Code.node) ~(v2 : Code.node) ~projections = v =:+ v1 * v2 in
  let%nn_cd grad_body ~(v : Code.node) ~(v1 : Code.node) ~(v2 : Code.node) ~projections =
    g1 =+ g * v2 || g2 =+ v1 * g
  in
  Tensor.binop ?desc_label ~compose_op:(Einsum spec) ~op_label:";=>" ~op_body ~grad_body

(** Similar to the explicit mode of [numpy.einsum], the unary variant. Can permute axes, extract diagonals,
    compute traces etc.

    Note that ["a->c"] from [numpy] is ["a=>c"] in OCANNL, since ["->"] is used to separate the input
    and the output axes. *)
let einsum1 ?desc_label spec =
  let open Code in
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd op_body ~(v : Code.node) ~(v1 : Code.node) ~projections = v =:+ v1 in
  let%nn_cd grad_body ~(v : Code.node) ~(v1 : Code.node) ~projections = g1 =+ g in
  Tensor.unop ?desc_label ~transpose_op:(Permute spec) ~op_label:"=>" ~op_body ~grad_body

let relu =
  let open Code in
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd op_body ~(v : Code.node) ~(v1 : Code.node) ~projections = v =: !/v1 ~projections in
  let%nn_cd grad_body ~(v : Code.node) ~(v1 : Code.node) ~projections = g1 =+ v -?/ g in
  Tensor.unop ~transpose_op:Pointwise_un ~op_label:"r" ~op_body ~grad_body

let subtensor_label ~over_kind ~from_left ~other_axes_pointwise =
  let kind_spec = match over_kind with Shape.AxisKey.Batch -> "|" | Input -> "/" | Output -> "-" in
  let pointwise_spec = if other_axes_pointwise then "." else "^" in
  if from_left then "@" ^ pointwise_spec ^ kind_spec else "@" ^ kind_spec ^ pointwise_spec

let dynamic_subtensor ?indexed_dims ~over_kind ~from_left ~other_axes_pointwise =
  let open Code in
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd op_body ~(v : Code.node) ~(v1 : Code.node) ~(v2 : Code.node) ~projections = v =: v1 -@> v2 in
  let%nn_cd grad_body ~(v : Code.node) ~(v1 : Code.node) ~(v2 : Code.node) ~projections =
    (* [projections] tracks the dynamic indexing for [n] (and not [v1]) as a slice.
       [-@>] simply means [Arg1]: take the first argument, ignore the second argument. *)
    g1 =+ g -@> v2
  in
  let compose_op = Shape.Dynamic_index { over_kind; from_left; other_axes_pointwise; indexed_dims } in
  let op_label = subtensor_label ~over_kind ~from_left ~other_axes_pointwise in
  Tensor.binop ~compose_op ~op_label ~op_body ~grad_body

module NFO_without_pow = struct
  let ( * ) = matmul ~is_diff:false
  let ( *. ) = pointmul ~is_diff:false
  let ( + ) = add ~is_diff:false
  let ( !/ ) = relu ~is_diff:false
  let ( !. ) = Tensor.number ~is_diff:false
  let ( !.. ) ?desc_label i = Tensor.number ?desc_label ~is_diff:false @@ Float.of_int i
  let ( - ) ?desc_label t1 t2 = ( + ) ?desc_label t1 (!.(-1.) *. t2)
  let ( ~- ) ?desc_label t = ( *. ) ?desc_label !.(-1.) t

  let ( @.| ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:true ~other_axes_pointwise:true ~is_diff:false

  let ( @./ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:true ~other_axes_pointwise:true ~is_diff:false

  let ( @.- ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:true ~other_axes_pointwise:true
      ~is_diff:false

  let ( @^| ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:true ~other_axes_pointwise:false
      ~is_diff:false

  let ( @^/ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:true ~other_axes_pointwise:false
      ~is_diff:false

  let ( @^- ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:true ~other_axes_pointwise:false
      ~is_diff:false

  let ( @|. ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:false ~other_axes_pointwise:true
      ~is_diff:false

  let ( @/. ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:false ~other_axes_pointwise:true
      ~is_diff:false

  let ( @-. ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:false ~other_axes_pointwise:true
      ~is_diff:false

  let ( @|^ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:false ~other_axes_pointwise:false
      ~is_diff:false

  let ( @/^ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:false ~other_axes_pointwise:false
      ~is_diff:false

  let ( @-^ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:false ~other_axes_pointwise:false
      ~is_diff:false
end

let rec pointpow ?desc_label ~is_diff p t1 : Tensor.t =
  let module NFDSL = struct
    module O = NFO_without_pow
  end in
  let open Code in
  let p_f = Tensor.number ~is_diff p in
  let%nn_cd op_body ~(v : Code.node) ~(v1 : Code.node) ~(v2 : Code.node) ~projections =
    v =: v1 ** v2 ~projections
  in
  let%nn_cd grad_body =
    if not is_diff then fun ~n:_ ~v1:_ ~v2:_ ~projections:_ -> Noop
    else if Float.equal p 2.0 then fun ~(v : Code.node) ~(v1 : Code.node) ~v2:_ ~projections ->
      g1 =+ p_f *. t1 * g
    else fun ~(v : Code.node) ~(v1 : Code.node) ~v2:_ ~projections ->
      g1 =+ p_f *. (t1 **. (p -. 1.)) * g
  in
  Tensor.binop ?desc_label ~compose_op:Pointwise_bin ~op_label:"**." ~op_body ~grad_body ~is_diff t1 p_f

let range ?desc_label ~is_diff ?axis_label upto =
  Tensor.term ?desc_label ~is_diff ~needs_gradient:false
    ~label:("0" ^ "..." ^ Int.to_string upto)
    ~batch_dims:[] ~input_dims:[]
    ~output_dims:[ Shape.dim (upto + 1) ]
    ?axis_labels:axis_label ~init_op:Range_over_offsets ()

let range_of_shape ?desc_label ~is_diff ?(batch_dims = []) ?(input_dims = []) ?(output_dims = []) ?axis_labels
    () =
  let dims = Array.concat_map [| batch_dims; output_dims; input_dims |] ~f:Array.of_list in
  Tensor.term ?desc_label ~is_diff ~needs_gradient:false ~batch_dims ~input_dims ~output_dims ?axis_labels
    ~label:("r" ^ Shape.dims_to_string dims)
    ~init_op:Range_over_offsets ()

(** In {!Tensor.term} the omitted axes are {!Shape.Unknown} -- to be inferred, here they are known and empty.  *)
let data ?desc_label ?axis_labels ?(needs_gradient = false) ~label ?(batch_dims = []) ?(input_dims = [])
    ?(output_dims = []) fetch_op =
  if List.for_all ~f:List.is_empty [ batch_dims; input_dims; output_dims ] then
    invalid_arg "Operation.result: data and the `%nn_dt` syntax do not support shape inference, specify dims";
  Tensor.term ?desc_label ~label ~is_diff:true ~needs_gradient ~batch_dims ~input_dims ~output_dims
    ?axis_labels ~fetch_op ()

(** No-gradient computations that happen at the end (potentially in parallel). *)
let result ?desc_label ?axis_labels ~label ?(batch_dims = []) ?(input_dims = []) ?(output_dims = [])
    postprocess_op =
  if List.for_all ~f:List.is_empty [ batch_dims; input_dims; output_dims ] then
    invalid_arg
      "Operation.result: results and the `%nn_rs` syntax do not support shape inference, specify dims";
  Tensor.term ?desc_label ~label ~is_diff:false ~needs_gradient:false ~batch_dims ~input_dims ~output_dims
    ?axis_labels ~postprocess_op ()

let assign =
  let module NFDSL = struct
    module O = struct end
  end in
  let%nn_cd assign ~(lhs : Ndarray.ptr) ~(rhs : Ndarray.ptr) ~projections = lhs =: rhs ~projections in
  assign

let assign_op field ~(v : Code.node) ~(v1 : Code.node) ~projections =
  assign ~lhs:(field v) ~rhs:(field v1) ~projections

(** A [stop_gradient] is an identity in the forward pass and a no-op in the backprop pass. *)
let stop_gradient =
  let grad_body ~n:_ ~v1:_ ~projections:_ = Code.Noop in
  let op_body = assign_op @@ Code.CDSL.data_of_node Value in
  Tensor.unop ~transpose_op:Pointwise_un ~op_label:"stop_grad" ~op_body ~grad_body ~is_diff:true

(** A [stop_broadcast] mutates the partially-inferred shape of a tensor in-place, substituting-in
    a [Fixed] marker on the dimensions. This way we avoid introducing a new node. *)
let stop_broadcast t = Shape.set_dims_type t.Tensor.shape Shape.fixed

module O = struct
  let ( * ) = matmul ~is_diff:true
  let ( *. ) = pointmul ~is_diff:true
  let ( + ) = add ~is_diff:true
  let ( **. ) ?desc_label base exp = pointpow ?desc_label exp base ~is_diff:true
  let ( !/ ) = relu ~is_diff:true
  let ( !~ ) ?desc_label label = Tensor.params ?desc_label label
  let ( !. ) = Tensor.number ~is_diff:true
  let ( !.. ) ?desc_label i = Tensor.number ?desc_label ~is_diff:true @@ Float.of_int i
  let ( - ) ?desc_label t1 t2 = ( + ) ?desc_label t1 (!.(-1.) *. t2)
  let ( ~- ) ?desc_label t = ( *. ) ?desc_label !.(-1.) t
  let ( /. ) ?desc_label t1 t2 = ( *. ) ?desc_label t1 (t2 **. -1.0)

  let ( @.| ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:true ~other_axes_pointwise:true ~is_diff:true

  let ( @./ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:true ~other_axes_pointwise:true ~is_diff:true

  let ( @.- ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:true ~other_axes_pointwise:true ~is_diff:true

  let ( @^| ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:true ~other_axes_pointwise:false ~is_diff:true

  let ( @^/ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:true ~other_axes_pointwise:false ~is_diff:true

  let ( @^- ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:true ~other_axes_pointwise:false
      ~is_diff:true

  let ( @|. ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:false ~other_axes_pointwise:true ~is_diff:true

  let ( @/. ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:false ~other_axes_pointwise:true ~is_diff:true

  let ( @-. ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:false ~other_axes_pointwise:true
      ~is_diff:true

  let ( @|^ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Batch ~from_left:false ~other_axes_pointwise:false
      ~is_diff:true

  let ( @/^ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Input ~from_left:false ~other_axes_pointwise:false
      ~is_diff:true

  let ( @-^ ) =
    dynamic_subtensor ~over_kind:Shape.AxisKey.Output ~from_left:false ~other_axes_pointwise:false
      ~is_diff:true
end

module FDSL = struct
  include Tensor.FDSL
  module O = O

  let einsum ?desc_label s = einsum ?desc_label s ~is_diff:true
  let einsum1 ?desc_label s = einsum1 ?desc_label s ~is_diff:true
  let range = range ~is_diff:true
  let range_of_shape = range_of_shape ~is_diff:true
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

  let ( **. ) ?desc_label base exp = pointpow ?desc_label exp base ~is_diff:false
  let ( /. ) ?desc_label t1 t2 = ( *. ) ?desc_label t1 (t2 **. -1.0)
end

module NFDSL = struct
  include Tensor.NFDSL
  module O = NFO

  let einsum ?desc_label s = einsum ?desc_label s ~is_diff:false
  let einsum1 ?desc_label s = einsum1 ?desc_label s ~is_diff:false
  let term = Tensor.term ~is_diff:false
  let result = result
  let range = range ~is_diff:false
  let range_of_shape = range_of_shape ~is_diff:false
end
