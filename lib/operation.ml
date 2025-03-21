(** Computational primitives for neural networks, integrating [Tensor] with [Assignments]. *)

open Base
module Asgns = Arrayjit.Assignments
module Idx = Arrayjit.Indexing
module Tn = Arrayjit.Tnode

let grad t = (Option.value_exn ~here:[%here] ~message:"No-gradient tensor" t.Tensor.diff).grad

module At = struct
  (** Get the value at the given indices. *)
  let ( .@{} ) t = Tn.get_value t.Tensor.value

  let ( .@%{} ) t = Tn.get_value @@ grad t

  (** Set the value at the given indices. *)
  let ( .@{}<- ) t = Tn.set_value t.Tensor.value

  let ( .@%{}<- ) t = Tn.set_value @@ grad t

  (** Get the value at the given index from a single-axis shape tensor. *)
  let ( .@[] ) t indx = Tn.get_value t.Tensor.value [| indx |]

  let ( .@%[] ) t indx = Tn.get_value (grad t) [| indx |]

  (** Set the value at the given index for a single-axis shape tensor. *)
  let ( .@[]<- ) t indx = Tn.set_value (grad t) [| indx |]

  let ( .@%[]<- ) t indx = Tn.set_value (grad t) [| indx |]
end

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
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g;
    g2 =+ g
  in
  Tensor.binop ~label:("+" :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn

let sub ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 - v2 in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g;
    g2 =- g
  in
  Tensor.binop ~label:("-" :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn

let mul compose_op ~op_asn =
  let module NTDSL = Initial_NTDSL in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g * v2;
    g2 =+ v1 * g
  in
  Tensor.binop ~compose_op ~op_asn ~grad_asn

let pointmul ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 * v2 in
  mul Pointwise_bin ~op_asn ~label:("*." :: label)

(* N1: AxB, N2 BxC, v: AxC, A: output of N1, B: input/output of N1/N2, C: input of N2. Although the
   matrix algebra would require that we insert additional transposes in gradient multiplies: AxB =
   AxC * CxB = AxC * (BxC)^T -> N1g += Ng * N2v^T, BxC = BxA * AxC = (AxB)^T * AxC -> N2g += N1v^T *
   Ng, in our setup there is no transposing to do, since the projections produce correct indices for
   their corresponding matrices. *)

let matmul ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =:+ v1 * v2 in
  mul Compose ~op_asn ~label:("*" :: label)

(** Similar to the explicit mode of [numpy.einsum], the binary variant. Can compute various forms of
    matrix multiplication, inner and outer products, etc.

    Note that ["a,b->c"] from [numpy] is ["a;b=>c"] in OCANNL, since ["->"] is used to separate the
    input and the output axes. *)
let einsum ?(label = []) spec =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =:+ v1 * v2 in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g * v2;
    g2 =+ v1 * g
  in
  Tensor.binop ~label:(";=>" :: label) ~compose_op:(Einsum spec) ~op_asn ~grad_asn

(** Like [einsum], but adds instead than multiplying the resulting values. *)
let outer_sum ?(label = []) spec =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =:+ v1 + v2 in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g;
    g2 =+ g
  in
  Tensor.binop ~label:(";=>+" :: label) ~compose_op:(Einsum spec) ~op_asn ~grad_asn

(** Similar to the explicit mode of [numpy.einsum], the unary variant. Can permute axes, extract
    diagonals, compute traces etc.

    Note that ["a->c"] from [numpy] is ["a=>c"] in OCANNL, since ["->"] is used to separate the
    input and the output axes. *)
let einsum1 ?(label = []) spec =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~projections = v =:+ v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ g in
  Tensor.unop ~label:("=>" :: label) ~transpose_op:(Shape.Permute spec) ~op_asn ~grad_asn

module NDO_before_pow = struct
  let ( * ) = matmul ~grad_spec:Prohibit_grad
  let ( *. ) = pointmul ~grad_spec:Prohibit_grad
  let ( + ) = add ~grad_spec:Prohibit_grad
  let ( !. ) = Tensor.number ~grad_spec:Prohibit_grad
  let ( !.. ) ?label i = Tensor.number ?label ~grad_spec:Prohibit_grad @@ Float.of_int i
  let ( - ) = sub ~grad_spec:Prohibit_grad
  let ( ~- ) ?label t = ( *. ) ?label !.(-1.) t
end

let rec pointpow ?(label : string list = []) ~grad_spec p t1 : Tensor.t =
  let module NTDSL = struct
    include Initial_NTDSL

    module O = struct
      include NDO_before_pow

      let ( **. ) ?label base exp = pointpow ?label ~grad_spec:Tensor.Prohibit_grad exp base
    end
  end in
  let p_t = NTDSL.number p in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 ** v2 ~projections in
  let%cd grad_asn =
    if Tensor.is_prohibit_grad grad_spec then fun ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ ->
      Asgns.empty_comp
    else if Float.equal p 2.0 then fun ~t:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ p_t *. t1 * g
    else if Float.equal p 1.0 then fun ~t:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ g
    else fun ~t:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ p_t *. (t1 **. (p -. 1.)) * g
  in
  Tensor.binop ~label:("**." :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn ~grad_spec t1 p_t

module NDO_before_div = struct
  include NDO_before_pow

  let ( **. ) ?label base exp = pointpow ?label ~grad_spec:Tensor.Prohibit_grad exp base
end

module NTDSL_before_div = struct
  include Initial_NTDSL
  module O = NDO_before_div
end

let rec pointdiv ?(label : string list = []) ~grad_spec t1 t2 =
  let module NTDSL = struct
    include Initial_NTDSL

    module O = struct
      include NDO_before_div

      let ( /. ) = pointdiv ~grad_spec:Tensor.Prohibit_grad
    end
  end in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 / v2 in
  (* We cannot use g in a tensor expression since it's an array, so we keep it to the left
     (RHS1). *)
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g / v2;
    g2 =+ g * (-1 *. t1 /. (t2 **. 2))
  in
  Tensor.binop ~label:("/." :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn ~grad_spec t1 t2

let relu ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~projections = v =: relu v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ relu_gate (v1, g) in
  Tensor.unop ~label:("relu" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let sat01 ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~projections = v =: sat01 v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ sat01_gate (v1, g) in
  Tensor.unop ~label:("sat01" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let exp ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~projections = v =: exp v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g * t in
  Tensor.unop ~label:("exp" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let log ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~projections = v =: log v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ g / v1 in
  Tensor.unop ~label:("log" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let log_2 = Float.log 2.0

let exp2 ?(label = []) =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~projections = v =: exp2 v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g * (!.log_2 *. t) in
  Tensor.unop ~label:("exp2" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let log2 ?(label = []) =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~projections = v =: log2 v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ g / (t1 *. !.log_2) in
  Tensor.unop ~label:("log2" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let rec sin ?(label = []) =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~projections = v =: sin v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections =
    g1 =+ g * (cos ?grad_spec:(Some Tensor.Prohibit_grad)) t1
  in
  Tensor.unop ~label:("sin" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

and cos ?(label = []) =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~projections = v =: cos v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections =
    g1 =+ g * (-1 *. (sin ?grad_spec:(Some Tensor.Prohibit_grad)) t1)
  in
  Tensor.unop ~label:("cos" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let sqrt ?(label = []) =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~projections = v =: sqrt v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g / (2 *. t) in
  Tensor.unop ~label:("sqrt" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let recip ?(label = []) =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~projections = v =: recip v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g * (-1 * (t **. 2)) in
  Tensor.unop ~label:("recip" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let recip_sqrt ?(label = []) =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~projections = v =: recip_sqrt v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g * (-0.5 *. (t **. 3)) in
  Tensor.unop ~label:("recip_sqrt" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let tanh ?(label = []) =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~projections = v =: tanh v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g * (1 - (t **. 2)) in
  Tensor.unop ~label:("tanh" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let neg ?(label = []) =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~projections = v =: neg v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ neg g in
  Tensor.unop ~label:("neg" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let not ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~projections = v =: not v1 in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~projections:_ = Asgns.empty_comp in
  Tensor.unop ~label:("not" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let fma ?(label = []) ~grad_spec t1 t2 t3 =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~t3 ~projections = v =: fma v1 v2 v3 in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~t3 ~projections =
    g1 =+ g * v2;
    g2 =+ g * v1;
    g3 =+ g
  in
  Tensor.ternop ~label:("fma" :: label) ~ternary_op:Pointwise_tern ~op_asn ~grad_asn ~grad_spec t1
    t2 t3

let range ?(label = []) ?(grad_spec = Tensor.Prohibit_grad) ?axis_label upto =
  let result =
    Tensor.term
      ~label:(("0" ^ "..." ^ Int.to_string upto) :: label)
      ~grad_spec ~batch_dims:[] ~input_dims:[] ~init_op:Range_over_offsets
  in
  match axis_label with
  | None -> result ~output_dims:[ upto + 1 ] ()
  | Some l -> result ~output_axes:[ (l, upto + 1) ] ()

let range_of_shape ?(label = []) ?(grad_spec = Tensor.Prohibit_grad) ?batch_dims ?input_dims
    ?output_dims ?batch_axes ?input_axes ?output_axes () =
  let f (dims, axes) =
    Array.of_list @@ Option.value ~default:[] @@ Option.first_some dims
    @@ Option.map axes ~f:(List.map ~f:snd)
  in
  let dims =
    Array.concat_map ~f
      [| (batch_dims, batch_axes); (output_dims, output_axes); (input_dims, input_axes) |]
  in
  let batch_dims = Option.first_some batch_dims @@ Option.some_if (Option.is_none batch_axes) [] in
  let input_dims = Option.first_some input_dims @@ Option.some_if (Option.is_none input_axes) [] in
  let output_dims =
    Option.first_some output_dims @@ Option.some_if (Option.is_none output_axes) []
  in
  Tensor.term
    ~label:(("r" ^ Idx.dims_to_string dims) :: label)
    ~grad_spec ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes
    ~init_op:Range_over_offsets ()

(** A [stop_gradient] is an identity in the forward pass and a no-op in the backprop pass. *)
let stop_gradient ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let grad_asn ~t:_ ~g:_ ~t1:_ ~projections:_ = Asgns.empty_comp in
  let%cd op_asn ~v ~t1 ~projections = v =: v1 in
  Tensor.unop ~label:("stop_grad" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn
    ~grad_spec:Prohibit_grad

let slice ?(label = []) ~grad_spec (batch_idx : Idx.static_symbol) t1 : Tensor.t =
  let module NTDSL = Initial_NTDSL in
  let op_asn ~v ~t1 ~projections =
    Asgns.to_comp
    @@ Fetch
         {
           array = v;
           fetch_op = Slice { batch_idx; sliced = t1.Tensor.value };
           dims = lazy (Lazy.force projections).Idx.lhs_dims;
         }
  in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ g in
  Tensor.unop ~label:("@|" :: label) ~transpose_op:(Batch_slice batch_idx) ~op_asn ~grad_asn
    ~grad_spec t1

let embed_symbol ?(label = []) static_sym : Tensor.t =
  let module NTDSL = Initial_NTDSL in
  let op_asn ~v ~projections =
    Asgns.to_comp
    @@ Fetch
         {
           array = v;
           fetch_op = Embed_symbol static_sym;
           dims = lazy (Lazy.force projections).Idx.lhs_dims;
         }
  in
  let grad_asn ~t:_ ~g:_ ~projections:_ = Asgns.empty_comp in
  Tensor.op ~label:("!@" :: label) ~op_asn ~grad_asn ~grad_spec:Prohibit_grad
    (Shape.make ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] ())
    []

module DO = struct
  let ( * ) = matmul ~grad_spec:If_needed
  let ( *. ) = pointmul ~grad_spec:If_needed
  let ( + ) = add ~grad_spec:If_needed
  let ( **. ) ?label base exp = pointpow ?label exp base ~grad_spec:If_needed
  let relu = relu ~grad_spec:If_needed
  let sat01 = sat01 ~grad_spec:If_needed
  let fma = fma ~grad_spec:If_needed
  let ( !. ) = Tensor.number ~grad_spec:If_needed
  let ( !.. ) ?label i = Tensor.number ?label ~grad_spec:If_needed @@ Float.of_int i
  let ( !@ ) = embed_symbol
  let ( - ) = sub ~grad_spec:If_needed
  let ( ~- ) ?label t = ( *. ) ?label !.(-1.) t
  let ( /. ) = pointdiv ~grad_spec:If_needed
  let ( @| ) ?label t1 idx = slice ?label ~grad_spec:If_needed idx t1
  let exp = exp ~grad_spec:If_needed
  let log = log ~grad_spec:If_needed
  let log2 = log2 ~grad_spec:If_needed
  let sin = sin ~grad_spec:If_needed
  let cos = cos ~grad_spec:If_needed
  let neg = neg ~grad_spec:If_needed
  let not = not ~grad_spec:If_needed
  let sqrt = sqrt ~grad_spec:If_needed
  let recip = recip ~grad_spec:If_needed
  let recip_sqrt = recip_sqrt ~grad_spec:If_needed
  let tanh = tanh ~grad_spec:If_needed
end

module NDO = struct
  include NDO_before_div

  let ( /. ) = pointdiv ~grad_spec:Prohibit_grad
  let ( @| ) ?label t1 idx = slice ?label ~grad_spec:Prohibit_grad idx t1
  let relu = relu ~grad_spec:Prohibit_grad
  let sat01 = sat01 ~grad_spec:Prohibit_grad
  let fma = fma ~grad_spec:Prohibit_grad
  let exp = exp ~grad_spec:Prohibit_grad
  let log = log ~grad_spec:Prohibit_grad
  let log2 = log2 ~grad_spec:Prohibit_grad
  let sin = sin ~grad_spec:Prohibit_grad
  let cos = cos ~grad_spec:Prohibit_grad
  let neg = neg ~grad_spec:Prohibit_grad
  let not = not ~grad_spec:Prohibit_grad
  let sqrt = sqrt ~grad_spec:Prohibit_grad
  let recip = recip ~grad_spec:Prohibit_grad
  let recip_sqrt = recip_sqrt ~grad_spec:Prohibit_grad
  let tanh = tanh ~grad_spec:Prohibit_grad
end

module TDSL = struct
  include Initial_TDSL
  module O = DO

  let einsum = einsum ~grad_spec:If_needed
  let outer_sum = outer_sum ~grad_spec:If_needed
  let einsum1 = einsum1 ~grad_spec:If_needed
  let range = range ~grad_spec:If_needed
  let range_of_shape = range_of_shape ~grad_spec:If_needed
  let stop_gradient = stop_gradient

  (** The input [i] dimensions default to empty. The batch dimensions will be inferred if omitted.
      [strict] controls whether [Constant_fill] will try to fit the given values in the tensor and
      contribute to shape inference. If it is not provided explicitly, it will be [true] if [b] is
      omitted, and [false] otherwise. *)
  let init_const ~l ?strict ?b ?(i = []) ~o values =
    let strict =
      match (strict, b) with Some s, _ -> s | None, Some _ -> false | None, None -> true
    in
    Tensor.term ~label:[ l ] ~grad_spec:Prohibit_grad ?batch_dims:b ~input_dims:i ~output_dims:o
      ~init_op:(Constant_fill { values; strict })
      ()

  (** It's like `Tensor.param` but without shape inference. *)
  let init_param ~l ?(b = []) ?(i = []) ?(o = []) values =
    Tensor.term ~label:[ l ] ~grad_spec:Require_grad ~batch_dims:b ~input_dims:i ~output_dims:o
      ~init_op:(Constant_fill { values; strict = false })
      ()
end

module NTDSL = struct
  include Initial_NTDSL
  module O = NDO

  let einsum = einsum ~grad_spec:Prohibit_grad
  let outer_sum = outer_sum ~grad_spec:Prohibit_grad
  let einsum1 = einsum1 ~grad_spec:Prohibit_grad
  let term = Tensor.term ~grad_spec:Prohibit_grad
  let range = range ~grad_spec:Prohibit_grad
  let range_of_shape = range_of_shape ~grad_spec:Prohibit_grad

  let counter ?(label = []) =
    let module NTDSL = Initial_NTDSL in
    let%cd op_asn ~v ~t1 ~projections = v =+ t1 ~projections in
    let grad_asn ~t:_ ~g:_ ~t1:_ ~projections:_ = Asgns.empty_comp in
    Tensor.unop ~label:("counter" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn
      ~grad_spec:Prohibit_grad
end
