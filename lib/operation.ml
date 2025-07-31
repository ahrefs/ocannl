(** Computational primitives for neural networks, integrating [Tensor] with [Assignments]. *)

open Base
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module Tn = Ir.Tnode

let _get_local_debug_runtime = Utils.get_local_debug_runtime
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
  Tensor.unop ~transpose_op:(Shape.Permute spec) ~op_asn ~grad_asn ~label:("=>" :: label)

module NDO_before_pow = struct
  let ( * ) t1 t2 = matmul ~grad_spec:Prohibit_grad t1 t2 ()
  let ( *. ) t1 t2 = pointmul ~grad_spec:Prohibit_grad t1 t2 ()
  let ( + ) t1 t2 = add ~grad_spec:Prohibit_grad t1 t2 ()
  let ( !. ) f = Tensor.number ~grad_spec:Prohibit_grad f
  let ( !.. ) ?label i = Tensor.number ?label ~grad_spec:Prohibit_grad @@ Float.of_int i
  let ( - ) t1 t2 = sub ~grad_spec:Prohibit_grad t1 t2 ()

  let ( ~- ) ?label t =
    pointmul ~grad_spec:Prohibit_grad ?label (Tensor.number ~grad_spec:Prohibit_grad (-1.)) t ()
end

let is_prohibit_grad = function Some Tensor.Prohibit_grad -> true | _ -> false

let rec pointpow ?grad_spec p t1 : Tensor.op_fun =
  let module NTDSL = struct
    include Initial_NTDSL

    module O = struct
      include NDO_before_pow

      let ( **. ) base exp = pointpow ~grad_spec:Tensor.Prohibit_grad exp base ()
    end
  end in
  let p_t = NTDSL.number p in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 ** v2 ~projections in
  fun ?(label = []) ->
    let%cd grad_asn =
      if is_prohibit_grad grad_spec then fun ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ ->
        Asgns.empty_comp
      else if Float.equal p 2.0 then fun ~t:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ p_t *. t1 * g
      else if Float.equal p 1.0 then fun ~t:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ g
      else fun ~t:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ p_t *. (t1 **. (p -. 1.)) * g
    in
    Tensor.binop ~compose_op:Pointwise_bin ~op_asn ~grad_asn t1 p_t ?grad_spec
      ~label:("**." :: label)

module NDO_before_div = struct
  include NDO_before_pow

  let ( **. ) base exp = pointpow ~grad_spec:Tensor.Prohibit_grad exp base ()
end

module NTDSL_before_div = struct
  include Initial_NTDSL
  module O = NDO_before_div
end

let rec pointdiv ?grad_spec t1 t2 : Tensor.op_fun =
  let module NTDSL = struct
    include Initial_NTDSL

    module O = struct
      include NDO_before_div

      let ( /. ) t1 t2 = pointdiv ~grad_spec:Tensor.Prohibit_grad t1 t2 ()
    end
  end in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 / v2 in
  (* We cannot use g in a tensor expression since it's an array, so we keep it to the left
     (RHS1). *)
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g / v2;
    g2 =+ g * (-1 *. t1 /. (t2 **. 2))
  in
  fun ?(label = []) ->
    Tensor.binop ~compose_op:Pointwise_bin ~op_asn ~grad_asn ?grad_spec t1 t2 ~label:("/." :: label)

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

let rec sin ?grad_spec =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~projections = v =: sin v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections =
    g1 =+ g * (cos ?grad_spec:(Some Tensor.Prohibit_grad)) t1 ()
  in
  fun ?(label = []) ->
    Tensor.unop ?grad_spec ~transpose_op:Pointwise_un ~op_asn ~grad_asn ~label:("sin" :: label)

and cos ?grad_spec : Tensor.t -> Tensor.op_fun =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~projections = v =: cos v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections =
    g1 =+ g * (-1 *. (sin ?grad_spec:(Some Tensor.Prohibit_grad)) t1 ())
  in
  fun t ?(label = []) ->
    Tensor.unop ?grad_spec ~transpose_op:Pointwise_un ~op_asn ~grad_asn t ~label:("cos" :: label)

let sqrt ?(label = []) =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~projections = v =: sqrt v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g / (2 *. t) in
  Tensor.unop ~label:("sqrt" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let recip ?(label = []) =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~projections = v =: recip v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g * (-1 *. (t **. 2)) in
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

let uint4x32_to_prec_uniform ?grad_spec =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~projections = v =: uint4x32_to_prec_uniform v1 in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~projections:_ = Asgns.empty_comp in
  fun t1 ->
    Tn.update_prec t1.Tensor.value Ir.Ops.uint4x32;
    Tensor.unop (* A placeholder that will be replaced by the actual precision by Tensor.op. *)
      ~transpose_op:(Uint4x32_to_prec (lazy (assert false)))
      ~op_asn ~grad_asn ?grad_spec t1

let lt ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: (v1 < v2) in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ = Asgns.empty_comp in
  Tensor.binop ~label:("<" :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn

let eq ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: (v1 = v2) in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ = Asgns.empty_comp in
  Tensor.binop ~label:("=" :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn

let ne ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: (v1 <> v2) in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ = Asgns.empty_comp in
  Tensor.binop ~label:("<>" :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn

let threefry4x32 =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 ^^^^ v2 in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ = Asgns.empty_comp in
  fun t1 t2 ->
    Tn.update_prec t1.Tensor.value Ir.Ops.uint4x32;
    Tn.update_prec t2.Tensor.value Ir.Ops.uint4x32;
    fun ?(label = [])
      ?grad_spec
      ?batch_dims
      ?batch_axes
      ?input_dims
      ?output_dims
      ?input_axes
      ?output_axes
      ?deduced
      ()
    ->
      let result =
        Tensor.binop ~compose_op:Pointwise_bin ~op_asn ~grad_asn t1 t2
          ~label:("threefry4x32" :: label) ?grad_spec ?batch_dims ?batch_axes ?input_dims
          ?output_dims ?input_axes ?output_axes ?deduced ()
      in
      (* Set output precision to uint4x32 *)
      Tn.update_prec result.value Ir.Ops.uint4x32;
      result

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

let where ?(label = []) ~grad_spec t1 t2 t3 =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~v ~t1 ~t2 ~t3 ~projections = v =: where v1 v2 v3 in
  (* Just to illustrate that both [0] and [!..0] are handled. *)
  let zero_cst = 0 in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~t3 ~projections =
    g2 =+ where v1 g 0;
    g3 =+ where v1 !..zero_cst g
  in
  Tensor.ternop ~label:("where" :: label) ~ternary_op:Pointwise_tern ~op_asn ~grad_asn ~grad_spec t1
    t2 t3

let range ?(label = []) ?(grad_spec = Tensor.Prohibit_grad) ?axis_label upto =
  let result =
    Tensor.term ~fetch_op:Range_over_offsets ~grad_spec ~batch_dims:[]
      ~label:(("0" ^ "..." ^ Int.to_string upto) :: label)
      ~input_dims:[]
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
  Tensor.term ~fetch_op:Range_over_offsets ~grad_spec ?batch_dims ?batch_axes
    ~label:(("r" ^ Idx.dims_to_string dims) :: label)
    ?input_dims ?output_dims ?input_axes ?output_axes ()

(** A [stop_gradient] is an identity in the forward pass and a no-op in the backprop pass. *)
let stop_gradient ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let grad_asn ~t:_ ~g:_ ~t1:_ ~projections:_ = Asgns.empty_comp in
  let%cd op_asn ~v ~t1 ~projections = v =: v1 in
  Tensor.unop ~label:("stop_grad" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn
    ~grad_spec:Prohibit_grad

let slice (batch_idx : Idx.static_symbol) =
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
  fun ?(label = []) ->
    Tensor.unop ~transpose_op:(Batch_slice batch_idx) ~op_asn ~grad_asn ~label:("@|" :: label)

let embed_symbol ?(label = []) static_sym : Tensor.t =
  Tensor.term ~fetch_op:(Embed_symbol static_sym) ~grad_spec:Prohibit_grad ~label:("!@" :: label)
    ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] ()

let embed_self_id ?(label = []) () : Tensor.t =
  Tensor.term ~fetch_op:Embed_self_id ~grad_spec:Prohibit_grad ~label:("!@self_id" :: label)
    ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] ()

module DO = struct
  let ( * ) ?label t1 t2 = matmul ~grad_spec:If_needed ?label t1 t2 ()
  let ( *. ) ?label t1 t2 = pointmul ~grad_spec:If_needed ?label t1 t2 ()
  let ( + ) ?label t1 t2 = add ~grad_spec:If_needed ?label t1 t2 ()
  let threefry4x32 ?label t1 t2 = threefry4x32 ~grad_spec:If_needed ?label t1 t2 ()

  let uint4x32_to_prec_uniform ?label t1 =
    uint4x32_to_prec_uniform ~grad_spec:If_needed t1 ?label ()

  let ( **. ) ?label base exp = pointpow ?label exp base ~grad_spec:If_needed ()
  let relu ?label t = relu ~grad_spec:If_needed ?label t ()
  let sat01 ?label t = sat01 ~grad_spec:If_needed ?label t ()
  let fma ?label t1 t2 t3 = fma ~grad_spec:If_needed ?label t1 t2 t3 ()
  let ( !. ) f = Tensor.number ~grad_spec:If_needed f
  let ( !.. ) ?label i = Tensor.number ?label ~grad_spec:If_needed @@ Float.of_int i
  let ( !@ ) = embed_symbol
  let ( - ) ?label t1 t2 = sub ~grad_spec:If_needed ?label t1 t2 ()

  let ( ~- ) ?label t =
    pointmul ~grad_spec:If_needed ?label (Tensor.number ~grad_spec:If_needed (-1.)) t ()

  let ( /. ) ?label t1 t2 = pointdiv ~grad_spec:If_needed ?label t1 t2 ()
  let ( @| ) ?label t1 idx = slice ?label ~grad_spec:If_needed idx t1 ()
  let exp ?label t = exp ~grad_spec:If_needed ?label t ()
  let log ?label t = log ~grad_spec:If_needed ?label t ()
  let log2 ?label t = log2 ~grad_spec:If_needed ?label t ()
  let sin ?label t = sin ~grad_spec:If_needed ?label t ()
  let cos ?label t = cos ~grad_spec:If_needed ?label t ()
  let neg ?label t = neg ~grad_spec:If_needed ?label t ()
  let not t = not ~grad_spec:Prohibit_grad t ()
  let sqrt ?label t = sqrt ~grad_spec:If_needed ?label t ()
  let recip ?label t = recip ~grad_spec:If_needed ?label t ()
  let recip_sqrt ?label t = recip_sqrt ~grad_spec:If_needed ?label t ()
  let tanh ?label t = tanh ~grad_spec:If_needed ?label t ()
  let where ?label t1 t2 t3 = where ~grad_spec:If_needed ?label t1 t2 t3 ()
  let ( < ) ?label t1 t2 = lt ~grad_spec:Prohibit_grad ?label t1 t2 ()
  let ( = ) ?label t1 t2 = eq ~grad_spec:Prohibit_grad ?label t1 t2 ()
  let ( <> ) ?label t1 t2 = ne ~grad_spec:Prohibit_grad ?label t1 t2 ()
  let embed_self_id = embed_self_id
  let einsum ?label spec t1 t2 = einsum ?label spec t1 t2 ~grad_spec:If_needed ()
  let einsum1 ?label spec t1 = einsum1 ?label spec t1 ~grad_spec:If_needed ()
  let ndarray = Tensor.ndarray ~grad_spec:If_needed
end

module NDO = struct
  include NDO_before_div

  let ( /. ) ?label t1 t2 = pointdiv ~grad_spec:Prohibit_grad ?label t1 t2 ()
  let ( @| ) ?label t1 idx = slice ?label ~grad_spec:Prohibit_grad idx t1 ()
  let ( !@ ) = embed_symbol
  let relu ?label t = relu ~grad_spec:Prohibit_grad ?label t ()
  let sat01 ?label t = sat01 ~grad_spec:Prohibit_grad ?label t ()
  let fma ?label t1 t2 t3 = fma ~grad_spec:Prohibit_grad ?label t1 t2 t3 ()
  let exp ?label t = exp ~grad_spec:Prohibit_grad ?label t ()
  let log ?label t = log ~grad_spec:Prohibit_grad ?label t ()
  let log2 ?label t = log2 ~grad_spec:Prohibit_grad ?label t ()
  let sin ?label t = sin ~grad_spec:Prohibit_grad ?label t ()
  let cos ?label t = cos ~grad_spec:Prohibit_grad ?label t ()
  let neg ?label t = neg ~grad_spec:Prohibit_grad ?label t ()
  let not ?label t = not ~grad_spec:Prohibit_grad ?label t ()
  let sqrt ?label t = sqrt ~grad_spec:Prohibit_grad ?label t ()
  let threefry4x32 ?label t1 t2 = threefry4x32 ~grad_spec:Prohibit_grad ?label t1 t2 ()

  let uint4x32_to_prec_uniform ?label t1 =
    uint4x32_to_prec_uniform ~grad_spec:Prohibit_grad ?label t1 ()

  let recip ?label t = recip ~grad_spec:Prohibit_grad ?label t ()
  let recip_sqrt ?label t = recip_sqrt ~grad_spec:Prohibit_grad ?label t ()
  let tanh ?label t = tanh ~grad_spec:Prohibit_grad ?label t ()
  let where ?label t1 t2 t3 = where ~grad_spec:Prohibit_grad ?label t1 t2 t3 ()
  let ( < ) ?label t1 t2 = lt ~grad_spec:Prohibit_grad ?label t1 t2 ()
  let ( = ) ?label t1 t2 = eq ~grad_spec:Prohibit_grad ?label t1 t2 ()
  let ( <> ) ?label t1 t2 = ne ~grad_spec:Prohibit_grad ?label t1 t2 ()
  let embed_self_id = embed_self_id
  let einsum ?label spec t1 t2 = einsum spec t1 t2 ~grad_spec:Prohibit_grad ?label ()
  let einsum1 ?label spec t1 = einsum1 spec t1 ~grad_spec:Prohibit_grad ?label ()
  let ndarray = Tensor.ndarray ~grad_spec:Prohibit_grad
end

(** The input [i] dimensions default to empty. The batch and output dimensions will be inferred if
    omitted. Note: the data should have no padding and if padding is inferred, the data will be
    copied; otherwise, the resulting tensor value shares host memory with the ndarray. *)
let reshape ~l ?b ?(i = []) ?o ndarray =
  Tensor.term ~init_data:(Asgns.Reshape ndarray) ?batch_dims:b ~label:[ l ] ~input_dims:i
    ?output_dims:o

(** The dimensions are taken from the provided ndarray, but the split into axis kinds still needs to
    be inferred (or provided). Assumes no padding. See also: {!reshape} and {!TDSL.wrap_param}. *)
let wrap ~l ?b ?(i = []) ?o ndarray =
  Tensor.term ~init_data:(Asgns.Keep_shape_no_padding ndarray) ?batch_dims:b ~label:[ l ]
    ~input_dims:i ?output_dims:o

(** Assumes the ndarray is padded as given. This means the dimensions of the ndarray will differ
    from the dimensions of the tensor by the padding. See also: {!TDSL.wrap}. *)
let wrap_padded ~l ?b ?(i = []) ?o ~padding ~padded_value ndarray =
  Tensor.term
    ~init_data:(Padded { data = ndarray; padding; padded_value })
    ?batch_dims:b ~label:[ l ] ~input_dims:i ?output_dims:o

(** The output dimensions are taken from the provided ndarray, assuming precisely the first axis is
    a batch axis, assumes no input axes and the batch dimensions are inferred. Empty output
    dimensions are allowed and represent scalars. Assumes the data has no padding, and data is
    copied if padding is inferred. See also: {!reshape} and {!wrap}. *)
let rebatch ~l ndarray =
  let output_dims = Ir.Ndarray.dims ndarray |> Array.to_list |> List.tl_exn in
  Tensor.term ~init_data:(Reshape ndarray) ~label:[ l ] ~input_dims:[] ~output_dims

let uniform ?grad_spec =
  uint4x32_to_prec_uniform ?grad_spec
    (threefry4x32 (embed_self_id ())
       (Tensor.term ~fetch_op:Range_over_offsets ~grad_spec:Prohibit_grad
          ~label:[ "range_over_offsets" ] ())
       ())

module TDSL = struct
  module O = DO

  let term = Tensor.term ~grad_spec:If_needed
  let number = Tensor.number ~grad_spec:If_needed
  let ndarray = Tensor.ndarray ~grad_spec:If_needed
  let threefry4x32 = threefry4x32 ~grad_spec:If_needed
  let uint4x32_to_prec_uniform = uint4x32_to_prec_uniform ~grad_spec:If_needed
  let embed_self_id = embed_self_id

  (** The default initialization operation for {!param} calls. *)
  let default_param_init = ref uniform

  let param ?value ?values =
    let t =
      match (value, values) with
      | Some _, Some _ -> invalid_arg "TDSL.param: both value and values are set"
      | Some value, None -> Tensor.param_init [| value |]
      | None, Some values -> Tensor.param_init values
      | None, None -> !default_param_init ~grad_spec:Require_grad ~batch_dims:[] ?batch_axes:None
    in
    Tensor.param ~t

  let einsum = einsum ~grad_spec:If_needed
  let outer_sum = outer_sum ~grad_spec:If_needed
  let einsum1 = einsum1 ~grad_spec:If_needed
  let range = range ~grad_spec:If_needed
  let range_of_shape = range_of_shape ~grad_spec:If_needed
  let stop_gradient = stop_gradient
  let reshape = reshape ~grad_spec:If_needed
  let wrap = wrap ~grad_spec:If_needed
  let wrap_padded = wrap_padded ~grad_spec:If_needed
  let rebatch = rebatch ~grad_spec:If_needed

  (** The input and output dimensions will be inferred if omitted. See {!reshape}. *)
  let reshape_param ~l ?i ?o ndarray =
    let t =
      Tensor.term ~grad_spec:Require_grad ~batch_dims:[] ?batch_axes:None
        ~init_data:(Reshape ndarray) ?fetch_op:None
    in
    Tensor.param ?input_dims:i ?output_dims:o ~t l

  (** See {!wrap}. *)
  let wrap_param ~l ?i ?o ndarray =
    let t =
      Tensor.term ~grad_spec:Require_grad ~batch_dims:[] ?batch_axes:None
        ~init_data:(Keep_shape_no_padding ndarray) ?fetch_op:None
    in
    Tensor.param ?input_dims:i ?output_dims:o ~t l
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
  let reshape = reshape ~grad_spec:Prohibit_grad
  let wrap = wrap ~grad_spec:Prohibit_grad
  let wrap_padded = wrap_padded ~grad_spec:Prohibit_grad
  let rebatch = rebatch ~grad_spec:Prohibit_grad
  let threefry4x32 = threefry4x32
  let uint4x32_to_prec_uniform = uint4x32_to_prec_uniform
  let embed_self_id = embed_self_id

  let counter ?(label = []) =
    let module NTDSL = Initial_NTDSL in
    let%cd op_asn ~v ~t1 ~projections = v =+ t1 ~projections in
    let grad_asn ~t:_ ~g:_ ~t1:_ ~projections:_ = Asgns.empty_comp in
    Tensor.unop ~label:("counter" :: label) ~transpose_op:Pointwise_un ~op_asn ~grad_asn
      ~grad_spec:Prohibit_grad
end
