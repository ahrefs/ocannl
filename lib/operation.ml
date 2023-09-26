(** Computational primitives for neural networks, integrating [Tensor] with [Assignments]. *)

open Base
module CDSL = Arrayjit.Low_level.CDSL
open Arrayjit

module Initial_NTDSL = struct
  let term = Tensor.term ~grad_spec:Prohibit_grad
  let number = Tensor.number ~grad_spec:Prohibit_grad
  let ndarray = Tensor.ndarray ~grad_spec:Prohibit_grad

  module O = struct end
end

module Initial_TDSL = struct
  let term = Tensor.term ~grad_spec:If_needed
  let number = Tensor.number ~grad_spec:If_needed
  let ndarray = Tensor.ndarray ~grad_spec:If_needed
  let param = Tensor.param

  module O = struct end
end

let add ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 + v2 in
  let%cd grad_asn ~v:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g;
    g2 =+ g
  in
  Tensor.binop ~label:("+" :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn

let sub ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 - v2 in
  let%cd grad_asn ~v:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g;
    g2 =- g
  in
  Tensor.binop ~label:("-" :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn

let mul compose_op =
  let module NTDSL = Initial_NTDSL in
  (* =:+ is needed for matmul and does not hurt for pointmul. *)
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =:+ v1 * v2 in
  let%cd grad_asn ~v:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g * v2;
    g2 =+ v1 * g
  in
  Tensor.binop ~compose_op ~op_asn ~grad_asn

let pointmul ?(label = []) = mul Pointwise_bin ~label:("*." :: label)

(* N1: AxB, N2 BxC, v: AxC, A: output of N1, B: input/output of N1/N2, C: input of N2.
   Although the matrix algebra would require that we insert additional transposes in gradient multiplies:
   AxB = AxC * CxB = AxC * (BxC)^T -> N1g += Ng * N2v^T,
   BxC = BxA * AxC = (AxB)^T * AxC -> N2g += N1v^T * Ng,
   in our setup there is no transposing to do, since the projections produce correct indices for their
   corresponding matrices. *)

let matmul ?(label = []) = mul Compose ~label:("*" :: label)

(** Similar to the explicit mode of [numpy.einsum], the binary variant. Can compute various forms of
    matrix multiplication, inner and outer products, etc.

    Note that ["a,b->c"] from [numpy] is ["a;b=>c"] in OCANNL, since ["->"] is used to separate the input
    and the output axes. *)
let einsum ?(label = []) spec =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =:+ v1 * v2 in
  let%cd grad_asn ~v:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g * v2;
    g2 =+ v1 * g
  in
  Tensor.binop ~label:(";=>" :: label) ~compose_op:(Einsum spec) ~op_asn ~grad_asn

(** Similar to the explicit mode of [numpy.einsum], the unary variant. Can permute axes, extract diagonals,
    compute traces etc.

    Note that ["a->c"] from [numpy] is ["a=>c"] in OCANNL, since ["->"] is used to separate the input
    and the output axes. *)
let einsum1 ?(label = []) spec =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~projections = v =:+ v1 in
  let%cd grad_asn ~v:_ ~g ~t1 ~projections = g1 =+ g in
  Tensor.unop ~label:("=>" :: label) ~transpose_op:(Shape.Permute spec) ~op_asn ~grad_asn

let relu ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~projections = v =: !/v1 ~projections in
  let%cd grad_asn ~v ~g ~t1 ~projections = g1 =+ v -?/ g in
  Tensor.unop ~label:("r" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

module NDO_without_pow = struct
  let ( * ) = matmul ~grad_spec:Prohibit_grad
  let ( *. ) = pointmul ~grad_spec:Prohibit_grad
  let ( + ) = add ~grad_spec:Prohibit_grad
  let ( !/ ) = relu ~grad_spec:Prohibit_grad
  let ( !. ) = Tensor.number ~grad_spec:Prohibit_grad
  let ( !.. ) ?label i = Tensor.number ?label ~grad_spec:Prohibit_grad @@ Float.of_int i
  let ( - ) = sub ~grad_spec:Prohibit_grad
  let ( ~- ) ?label t = ( *. ) ?label !.(-1.) t
end

let rec pointpow ?(label : string list = []) ~grad_spec p t1 : Tensor.t =
  let module NTDSL = struct
    include Initial_NTDSL

    module O = struct
      include NDO_without_pow

      let ( **. ) ?label base exp = pointpow ?label ~grad_spec:Tensor.Prohibit_grad exp base
    end
  end in
  let p_t = NTDSL.number p in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 ** v2 ~projections in
  let%cd grad_asn =
    if Tensor.is_prohibit_grad grad_spec then fun ~v:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ -> Assignments.Noop
    else if Float.equal p 2.0 then fun ~v:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ p_t *. t1 * g
    else if Float.equal p 1.0 then fun ~v:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ g
    else fun ~v:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ p_t *. (t1 **. (p -. 1.)) * g
  in
  Tensor.binop ~label:("**." :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn ~grad_spec t1 p_t

module NDO_without_div = struct
  include NDO_without_pow

  let ( **. ) ?label base exp = pointpow ?label ~grad_spec:Tensor.Prohibit_grad exp base
end

let rec pointdiv ?(label : string list = []) ~grad_spec t1 t2 =
  let module NTDSL = struct
    include Initial_NTDSL

    module O = struct
      include NDO_without_div

      let ( /. ) = pointdiv ~grad_spec:Tensor.Prohibit_grad
    end
  end in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 / v2 in
  (* We cannot use g in a tensor expression since it's an array, so we keep it to the left (RHS1). *)
  let%cd grad_asn ~v:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g / v2;
    g2 =+ g * (-1 *. t1 /. (t2 **. 2))
  in
  Tensor.binop ~label:("/." :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn ~grad_spec t1 t2

let range ?(label = []) ?(grad_spec = Tensor.Prohibit_grad) ?axis_label upto =
  let result =
    Tensor.term
      ~label:(("0" ^ "..." ^ Int.to_string upto) :: label)
      ~grad_spec ~batch_dims:[] ~input_dims:[] ~init_op:Range_over_offsets
  in
  match axis_label with
  | None -> result ~output_dims:[ upto + 1 ] ()
  | Some l -> result ~output_axes:[ (l, upto + 1) ] ()

let range_of_shape ?(label = []) ?(grad_spec = Tensor.Prohibit_grad) ?batch_dims ?input_dims ?output_dims
    ?batch_axes ?input_axes ?output_axes () =
  let f (dims, axes) =
    Array.of_list @@ Option.value ~default:[] @@ Option.first_some dims
    @@ Option.map axes ~f:(List.map ~f:snd)
  in
  let dims =
    Array.concat_map ~f [| (batch_dims, batch_axes); (output_dims, output_axes); (input_dims, input_axes) |]
  in
  let batch_dims = Option.first_some batch_dims @@ Option.some_if (Option.is_none batch_axes) [] in
  let input_dims = Option.first_some input_dims @@ Option.some_if (Option.is_none input_axes) [] in
  let output_dims = Option.first_some output_dims @@ Option.some_if (Option.is_none output_axes) [] in
  Tensor.term
    ~label:(("r" ^ Indexing.dims_to_string dims) :: label)
    ~grad_spec ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes
    ~init_op:Range_over_offsets ()

(** In {!Tensor.term} the omitted axes are {!Shape.Unknown} -- to be inferred, here they are known and empty.  *)
let data ?(label = []) ?(grad_spec = Tensor.Prohibit_grad) ?batch_dims ?input_dims ?output_dims ?batch_axes
    ?input_axes ?output_axes fetch_op =
  let batch_dims = Option.first_some batch_dims @@ Option.some_if (Option.is_none batch_axes) [] in
  let input_dims = Option.first_some input_dims @@ Option.some_if (Option.is_none input_axes) [] in
  let output_dims = Option.first_some output_dims @@ Option.some_if (Option.is_none output_axes) [] in
  if
    List.for_all
      ~f:(Fn.compose List.is_empty @@ Option.value ~default:[])
      [ batch_dims; input_dims; output_dims ]
  then invalid_arg "Operation.data: data ops do not support shape inference, specify dims";
  Tensor.term ~label ~grad_spec ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes
    ~fetch_op ()

(** A [stop_gradient] is an identity in the forward pass and a no-op in the backprop pass. *)
let stop_gradient ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let grad_asn ~v:_ ~g:_ ~t1:_ ~projections:_ = Assignments.Noop in
  let%cd op_asn ~v ~t1 ~projections = v =: v1 in
  Tensor.unop ~label:("stop_grad" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn
    ~grad_spec:Prohibit_grad

let slice ?(label = []) ~grad_spec (batch_idx : Indexing.static_symbol) t1 : Tensor.t =
  let module NTDSL = Initial_NTDSL in
  let op_asn ~v ~t1 ~projections =
    Assignments.Fetch
      {
        array = v;
        fetch_op = Slice { batch_idx; sliced = t1.Tensor.value };
        dims = lazy (Lazy.force projections).Indexing.lhs_dims;
      }
  in
  let%cd grad_asn ~v:_ ~g ~t1 ~projections = g1 =+ g in
  Tensor.unop ~label:("@|" :: label) ~transpose_op:(Batch_slice batch_idx) ~op_asn ~grad_asn ~grad_spec t1

let embed_symbol ?(label = []) static_sym : Tensor.t =
  let module NTDSL = Initial_NTDSL in
  let op_asn ~v ~projections =
    Assignments.Fetch
      {
        array = v;
        fetch_op = Embed_symbol static_sym;
        dims = lazy (Lazy.force projections).Indexing.lhs_dims;
      }
  in
  let grad_asn ~v:_ ~g:_ ~projections:_ = Assignments.Noop in
  Tensor.op ~label:("!@" :: label) ~op_asn ~grad_asn ~grad_spec:Prohibit_grad
    (Shape.make ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] ())
    []

module DO = struct
  let ( * ) = matmul ~grad_spec:If_needed
  let ( *. ) = pointmul ~grad_spec:If_needed
  let ( + ) = add ~grad_spec:If_needed
  let ( **. ) ?label base exp = pointpow ?label exp base ~grad_spec:If_needed
  let ( !/ ) = relu ~grad_spec:If_needed
  let ( !~ ) label = Tensor.param label
  let ( !. ) = Tensor.number ~grad_spec:If_needed
  let ( !.. ) ?label i = Tensor.number ?label ~grad_spec:If_needed @@ Float.of_int i
  let ( !@ ) = embed_symbol
  let ( - ) = sub ~grad_spec:If_needed
  let ( ~- ) ?label t = ( *. ) ?label !.(-1.) t
  let ( /. ) = pointdiv ~grad_spec:If_needed
  let ( @| ) ?label t1 idx = slice ?label ~grad_spec:If_needed idx t1
end

module NDO = struct
  include NDO_without_div

  let ( /. ) = pointdiv ~grad_spec:Prohibit_grad
  let ( @| ) ?label t1 idx = slice ?label ~grad_spec:Prohibit_grad idx t1
end

module TDSL = struct
  include Initial_TDSL
  module O = DO

  let einsum = einsum ~grad_spec:If_needed
  let einsum1 = einsum1 ~grad_spec:If_needed
  let range = range ~grad_spec:If_needed
  let range_of_shape = range_of_shape ~grad_spec:If_needed
  let data = data
  let stop_gradient = stop_gradient

  let init_const ~l ?(b = []) ?(i = []) ?(o = []) values =
    Tensor.term ~label:[l] ~grad_spec:Prohibit_grad ~batch_dims:b ~input_dims:i ~output_dims:o
      ~init_op:(Constant_fill { values; strict = false })
      ()

  (** It's like `Tensor.param` but without shape inference. *)
  let init_param ~l ?(b = []) ?(i = []) ?(o = []) values =
    Tensor.term ~label:[l] ~grad_spec:Require_grad ~batch_dims:b ~input_dims:i ~output_dims:o
      ~init_op:(Constant_fill { values; strict = false })
      ()
end

module NTDSL = struct
  include Initial_NTDSL
  module O = NDO

  let einsum = einsum ~grad_spec:Prohibit_grad
  let einsum1 = einsum1 ~grad_spec:Prohibit_grad
  let term = Tensor.term ~grad_spec:Prohibit_grad
  let range = range ~grad_spec:Prohibit_grad
  let range_of_shape = range_of_shape ~grad_spec:Prohibit_grad

  let counter ?(label = []) =
    let module NTDSL = Initial_NTDSL in
    let%cd op_asn ~v ~t1 ~projections = v =+ t1 ~projections in
    let grad_asn ~v:_ ~g:_ ~t1:_ ~projections:_ = Assignments.Noop in
    Tensor.unop ~label:("counter" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn
      ~grad_spec:Prohibit_grad
end
