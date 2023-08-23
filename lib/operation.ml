(** Computational primitives for neural networks, integrating [Tensor] with [Low_level]. *)

open Base
open Arrayjit
module CDSL = Session.CDSL

module Empty_DSL = struct
  include Tensor.NTDSL
  module O = struct end
end

let add =
  let module NTDSL = Empty_DSL in
  let%nn_cd op_body ~v ~t1 ~t2 ~projections = v =: v1 + v2 in
  let%nn_cd grad_body ~v:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g;
    g2 =+ g
  in
  Tensor.binop ~compose_op:Pointwise_bin ~op_label:"+" ~op_body ~grad_body

let pointmul =
  let module NTDSL = Empty_DSL in
  let%nn_cd op_body ~v ~t1 ~t2 ~projections = v =: v1 * v2 in
  let%nn_cd grad_body ~v:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g * v2;
    g2 =+ v1 * g
  in
  Tensor.binop ~compose_op:Pointwise_bin ~op_label:"*." ~op_body ~grad_body

(* N1: AxB, N2 BxC, v: AxC, A: output of N1, B: input/output of N1/N2, C: input of N2.
   Although the matrix algebra would require that we insert additional transposes in gradient multiplies:
   AxB = AxC * CxB = AxC * (BxC)^T -> N1g += Ng * N2v^T,
   BxC = BxA * AxC = (AxB)^T * AxC -> N2g += N1v^T * Ng,
   in our setup there is no transposing to do, since the projections produce correct indices for their
   corresponding matrices. *)

let matmul =
  let module NTDSL = Empty_DSL in
  let%nn_cd op_body ~v ~t1 ~t2 ~projections = v =:+ v1 * v2 in
  let%nn_cd grad_body ~v:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g * v2;
    g2 =+ v1 * g
  in
  Tensor.binop ~compose_op:Compose ~op_label:"*" ~op_body ~grad_body

(** Similar to the explicit mode of [numpy.einsum], the binary variant. Can compute various forms of
    matrix multiplication, inner and outer products, etc.

    Note that ["a,b->c"] from [numpy] is ["a;b=>c"] in OCANNL, since ["->"] is used to separate the input
    and the output axes. *)
let einsum ?desc_label spec =
  let module NTDSL = Empty_DSL in
  let%nn_cd op_body ~v ~t1 ~t2 ~projections = v =:+ v1 * v2 in
  let%nn_cd grad_body ~v:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g * v2;
    g2 =+ v1 * g
  in
  Tensor.binop ?desc_label ~compose_op:(Einsum spec) ~op_label:";=>" ~op_body ~grad_body

(** Similar to the explicit mode of [numpy.einsum], the unary variant. Can permute axes, extract diagonals,
    compute traces etc.

    Note that ["a->c"] from [numpy] is ["a=>c"] in OCANNL, since ["->"] is used to separate the input
    and the output axes. *)
let einsum1 ?desc_label spec =
  let module NTDSL = Empty_DSL in
  let%nn_cd op_body ~v ~t1 ~projections = v =:+ v1 in
  let%nn_cd grad_body ~v:_ ~g ~t1 ~projections = g1 =+ g in
  Tensor.unop ?desc_label ~transpose_op:(Shape.Permute spec) ~op_label:"=>" ~op_body ~grad_body

let relu =
  let module NTDSL = Empty_DSL in
  let%nn_cd op_body ~v ~t1 ~projections = v =: !/v1 ~projections in
  let%nn_cd grad_body ~v ~g ~t1 ~projections = g1 =+ v -?/ g in
  Tensor.unop ~transpose_op:Pointwise_un ~op_label:"r" ~op_body ~grad_body

module NDO_without_pow = struct
  let ( * ) = matmul ~grad_spec:Prohibit_grad
  let ( *. ) = pointmul ~grad_spec:Prohibit_grad
  let ( + ) = add ~grad_spec:Prohibit_grad
  let ( !/ ) = relu ~grad_spec:Prohibit_grad
  let ( !. ) = Tensor.number ~grad_spec:Prohibit_grad
  let ( !.. ) ?desc_label i = Tensor.number ?desc_label ~grad_spec:Prohibit_grad @@ Float.of_int i
  let ( - ) ?desc_label t1 t2 = ( + ) ?desc_label t1 (!.(-1.) *. t2)
  let ( ~- ) ?desc_label t = ( *. ) ?desc_label !.(-1.) t
end

let rec pointpow ?desc_label ~grad_spec p t1 : Tensor.t =
  let module NTDSL = struct
    include Tensor.NTDSL
    module O = NDO_without_pow
  end in
  let p_t = NTDSL.number p in
  let%nn_cd op_body ~v ~t1 ~t2 ~projections = v =: v1 ** v2 ~projections in
  let%nn_cd grad_body =
    if Tensor.is_prohibit_grad grad_spec then fun ~v:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ -> High_level.Noop
    else if Float.equal p 2.0 then fun ~v:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ p_t *. t1 * g
    else fun ~v:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ p_t *. (t1 **. (p -. 1.)) * g
  in
  Tensor.binop ?desc_label ~compose_op:Pointwise_bin ~op_label:"**." ~op_body ~grad_body ~grad_spec t1 p_t

let range ?desc_label ?(grad_spec = Tensor.Prohibit_grad) ?axis_label upto =
  Tensor.term ?desc_label ~grad_spec
    ~label:("0" ^ "..." ^ Int.to_string upto)
    ~batch_dims:[] ~input_dims:[]
    ~output_dims:[ upto + 1 ]
    ?axis_labels:axis_label ~init_op:Range_over_offsets ()

let range_of_shape ?desc_label ?(grad_spec = Tensor.Prohibit_grad) ?(batch_dims = []) ?(input_dims = [])
    ?(output_dims = []) ?axis_labels () =
  let dims = Array.concat_map [| batch_dims; output_dims; input_dims |] ~f:Array.of_list in
  Tensor.term ?desc_label ~grad_spec ~batch_dims ~input_dims ~output_dims ?axis_labels
    ~label:("r" ^ Indexing.dims_to_string dims)
    ~init_op:Range_over_offsets ()

(** In {!Tensor.term} the omitted axes are {!Shape.Unknown} -- to be inferred, here they are known and empty.  *)
let data ?desc_label ?axis_labels ?(grad_spec = Tensor.Prohibit_grad) ~label ?(batch_dims = [])
    ?(input_dims = []) ?(output_dims = []) fetch_op =
  if List.for_all ~f:List.is_empty [ batch_dims; input_dims; output_dims ] then
    invalid_arg "Operation.data: data ops do not support shape inference, specify dims";
  Tensor.term ?desc_label ~label ~grad_spec ~batch_dims ~input_dims ~output_dims ?axis_labels ~fetch_op ()

(** A [stop_gradient] is an identity in the forward pass and a no-op in the backprop pass. *)
let stop_gradient =
  let module NTDSL = Empty_DSL in
  let grad_body ~v:_ ~g:_ ~t1:_ ~projections:_ = High_level.Noop in
  let%nn_cd op_body ~v ~t1 ~projections = v =: v1 in
  Tensor.unop ~transpose_op:Pointwise_un ~op_label:"stop_grad" ~op_body ~grad_body ~grad_spec:Prohibit_grad

(** A [stop_broadcast] mutates the partially-inferred shape of a tensor in-place, substituting-in
    a [Fixed] marker on the dimensions. This way we avoid introducing a new node. *)
let stop_broadcast t = Shape.set_dims_type t.Tensor.shape Shape.fixed

module O = struct
  let ( * ) = matmul ~grad_spec:If_needed
  let ( *. ) = pointmul ~grad_spec:If_needed
  let ( + ) = add ~grad_spec:If_needed
  let ( **. ) ?desc_label base exp = pointpow ?desc_label exp base ~grad_spec:If_needed
  let ( !/ ) = relu ~grad_spec:If_needed
  let ( !~ ) ?desc_label label = Tensor.params ?desc_label label
  let ( !. ) = Tensor.number ~grad_spec:If_needed
  let ( !.. ) ?desc_label i = Tensor.number ?desc_label ~grad_spec:If_needed @@ Float.of_int i
  let ( - ) ?desc_label t1 t2 = ( + ) ?desc_label t1 (!.(-1.) *. t2)
  let ( ~- ) ?desc_label t = ( *. ) ?desc_label !.(-1.) t
  let ( /. ) ?desc_label t1 t2 = ( *. ) ?desc_label t1 (t2 **. -1.0)
end

module TDSL = struct
  include Tensor.TDSL
  module O = O

  let einsum ?desc_label s = einsum ?desc_label s ~grad_spec:If_needed
  let einsum1 ?desc_label s = einsum1 ?desc_label s ~grad_spec:If_needed
  let range = range ~grad_spec:If_needed
  let range_of_shape = range_of_shape ~grad_spec:If_needed
  let data = data
  let stop_broadcast = stop_broadcast
  let stop_gradient = stop_gradient

  let init_const ~l ?(b = []) ?(i = []) ?(o = []) cs =
    Tensor.term ~label:l ~grad_spec:Prohibit_grad ~batch_dims:b ~input_dims:i ~output_dims:o
      ~init_op:(Constant_fill cs) ()

  (** It's like `Tensor.params` but without shape inference. *)
  let init_param ~l ?(b = []) ?(i = []) ?(o = []) cs =
    Tensor.term ~label:l ~grad_spec:Require_grad ~batch_dims:b ~input_dims:i ~output_dims:o
      ~init_op:(Constant_fill cs) ()
end

module NDO = struct
  include NDO_without_pow

  let ( **. ) ?desc_label base exp = pointpow ?desc_label exp base ~grad_spec:Prohibit_grad
  let ( /. ) ?desc_label t1 t2 = ( *. ) ?desc_label t1 (t2 **. -1.0)
end

module NTDSL = struct
  include Tensor.NTDSL
  module O = NDO

  let einsum ?desc_label s = einsum ?desc_label s ~grad_spec:Prohibit_grad
  let einsum1 ?desc_label s = einsum1 ?desc_label s ~grad_spec:Prohibit_grad
  let term = Tensor.term ~grad_spec:Prohibit_grad
  let range = range ~grad_spec:Prohibit_grad
  let range_of_shape = range_of_shape ~grad_spec:Prohibit_grad

  let counter =
    let module NTDSL = Empty_DSL in
    let%nn_cd op_body ~v ~t1 ~projections = v =+ t1 ~projections in
    let grad_body ~v:_ ~g:_ ~t1:_ ~projections:_ = High_level.Noop in
    Tensor.unop ~op_label:"counter" ~transpose_op:Pointwise_un ~op_body ~grad_body ~grad_spec:Prohibit_grad
end
