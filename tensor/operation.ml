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

let add =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =: v1 + v2 in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g;
    g2 =+ g
  in
  Tensor.binop ~op_label:"+" ~compose_op:Pointwise_bin ~op_asn ~grad_asn

let sub =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =: v1 - v2 in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g;
    g2 =- g
  in
  Tensor.binop ~op_label:"-" ~compose_op:Pointwise_bin ~op_asn ~grad_asn

let mul compose_op ~op_asn =
  let module NTDSL = Initial_NTDSL in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g * v2;
    g2 =+ v1 * g
  in
  Tensor.binop ~compose_op ~op_asn ~grad_asn

let pointmul =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =: v1 * v2 in
  mul Pointwise_bin ~op_asn ~op_label:"*."

(* N1: AxB, N2 BxC, v: AxC, A: output of N1, B: input/output of N1/N2, C: input of N2. Although the
   matrix algebra would require that we insert additional transposes in gradient multiplies: AxB =
   AxC * CxB = AxC * (BxC)^T -> N1g += Ng * N2v^T, BxC = BxA * AxC = (AxB)^T * AxC -> N2g += N1v^T *
   Ng, in our setup there is no transposing to do, since the projections produce correct indices for
   their corresponding matrices. *)

let matmul =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =:+ v1 * v2 in
  mul Compose ~op_asn ~op_label:"*"

module NDO_before_pow = struct
  let ( * ) t1 t2 = matmul ~grad_spec:Prohibit_grad t1 t2 ()
  let ( *. ) t1 t2 = pointmul ~grad_spec:Prohibit_grad t1 t2 ()
  let ( + ) t1 t2 = add ~grad_spec:Prohibit_grad t1 t2 ()
  let ( !. ) f = Tensor.number ~grad_spec:Prohibit_grad f
  let ( !.. ) ?label i = Tensor.number ?label ~grad_spec:Prohibit_grad @@ Float.of_int i
  let ( !% ) ?label i = Tensor.bits ?label ~grad_spec:Prohibit_grad i
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
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =: v1 ** v2 ~projections in

  let%cd grad_asn =
    if is_prohibit_grad grad_spec then fun ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ -> Asgns.empty_comp
    else if Float.equal p 2.0 then fun ~t:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ p_t *. t1 * g
    else if Float.equal p 1.0 then fun ~t:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ g
    else fun ~t:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ p_t *. (t1 **. (p -. 1.)) * g
  in
  Tensor.binop ~compose_op:Pointwise_bin ~op_asn ~grad_asn t1 p_t ?grad_spec ~op_label:"**."

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
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =: v1 / v2 in
  (* We cannot use g in a tensor expression since it's an array, so we keep it to the left
     (RHS1). *)
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g / v2;
    g2 =+ g * (-1 *. t1 /. (t2 **. 2))
  in

  Tensor.binop ~compose_op:Pointwise_bin ~op_asn ~grad_asn ?grad_spec t1 t2 ~op_label:"/."

let relu =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~projections = v =: relu v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ relu_gate (v1, g) in
  Tensor.unop ~op_label:"relu" ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let sat01 =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~projections = v =: sat01 v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ sat01_gate (v1, g) in
  Tensor.unop ~op_label:"sat01" ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let exp =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~projections = v =: exp v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g * t in
  Tensor.unop ~op_label:"exp" ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let log =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~projections = v =: log v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ g / v1 in
  Tensor.unop ~op_label:"log" ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let log_2 = Float.log 2.0

let exp2 =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~t ~t1 ~projections = v =: exp2 v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g * (!.log_2 *. t) in
  Tensor.unop ~op_label:"exp2" ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let log2 =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~t ~t1 ~projections = v =: log2 v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ g / (t1 *. !.log_2) in
  Tensor.unop ~op_label:"log2" ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let rec sin ?grad_spec =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~t ~t1 ~projections = v =: sin v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections =
    g1 =+ g * (cos ?grad_spec:(Some Tensor.Prohibit_grad)) t1 ()
  in

  Tensor.unop ?grad_spec ~transpose_op:Pointwise_un ~op_asn ~grad_asn ~op_label:"sin"

and cos ?grad_spec : Tensor.t -> Tensor.op_fun =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~t ~t1 ~projections = v =: cos v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections =
    g1 =+ g * (-1 *. (sin ?grad_spec:(Some Tensor.Prohibit_grad)) t1 ())
  in
  fun t -> Tensor.unop ?grad_spec ~transpose_op:Pointwise_un ~op_asn ~grad_asn t ~op_label:"cos"

let sqrt =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~t ~t1 ~projections = v =: sqrt v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g / (2 *. t) in
  Tensor.unop ~op_label:"sqrt" ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let recip =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~t ~t1 ~projections = v =: recip v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g * (-1 *. (t **. 2)) in
  Tensor.unop ~op_label:"recip" ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let recip_sqrt =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~t ~t1 ~projections = v =: recip_sqrt v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g * (-0.5 *. (t **. 3)) in
  Tensor.unop ~op_label:"recip_sqrt" ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let tanh =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~t ~t1 ~projections = v =: tanh v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections = g1 =+ g * (1 - (t **. 2)) in
  Tensor.unop ~op_label:"tanh" ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let neg =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~t ~t1 ~projections = v =: neg v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ neg g in
  Tensor.unop ~op_label:"neg" ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let not =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~projections = v =: not v1 in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~projections:_ = Asgns.empty_comp in
  Tensor.unop ~op_label:"not" ~transpose_op:Pointwise_un ~op_asn ~grad_asn

let uint4x32_to_prec_uniform ?grad_spec =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~projections = v =: uint4x32_to_prec_uniform v1 in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~projections:_ = Asgns.empty_comp in
  fun t1 ?label ?top_down_prec ->
    (* Ignore what the caller says, since we must learn the precision from the outside. *)
    ignore (top_down_prec : bool option);
    Tn.update_prec t1.Tensor.value Ir.Ops.uint4x32;
    Tensor.unop (* A placeholder that will be replaced by the actual precision by Tensor.op. *)
      ~transpose_op:(Uint4x32_to_prec (lazy (assert false)))
      ~op_asn ~grad_asn ?grad_spec (* Modifying the label would cause identifier pollution. *)
      ?label ~top_down_prec:true t1

let uint4x32_to_prec_uniform1 ?grad_spec =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~projections = v =: uint4x32_to_prec_uniform1 v1 in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~projections:_ = Asgns.empty_comp in
  fun t1 ?label ?top_down_prec ->
    (* Ignore what the caller says, since we must learn the precision from the outside. *)
    ignore (top_down_prec : bool option);
    Tn.update_prec t1.Tensor.value Ir.Ops.uint4x32;
    Tensor.unop (* A placeholder that will be replaced by the actual precision by Tensor.op. *)
      ~transpose_op:Pointwise_un ~op_asn ~grad_asn
      ?grad_spec (* Modifying the label would cause identifier pollution. *)
      ?label ~top_down_prec:true t1

let lt =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =: (v1 < v2) in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ = Asgns.empty_comp in
  Tensor.binop ~op_label:"<" ~compose_op:Pointwise_bin ~op_asn ~grad_asn

let eq =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =: (v1 = v2) in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ = Asgns.empty_comp in
  Tensor.binop ~op_label:"=" ~compose_op:Pointwise_bin ~op_asn ~grad_asn

let ne =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =: (v1 <> v2) in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ = Asgns.empty_comp in
  Tensor.binop ~op_label:"<>" ~compose_op:Pointwise_bin ~op_asn ~grad_asn

let interleave =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~t2 ~projections:_ =
    t =:+ id t1 ~logic:"... | ... -> ..., i => ... | ... -> ..., 2*i";
    t =+ id t2 ~logic:"... | ... -> ..., i => ... | ... -> ..., 2*i + 1"
  in
  let%cd grad_asn ~t ~g:_ ~t1 ~t2 ~projections:_ =
    t1.grad =+ id t.grad ~logic:"... | ... -> ..., 2*i => ... | ... -> ..., i";
    t2.grad =+ id t.grad ~logic:"... | ... -> ..., 2*i + 1 => ... | ... -> ..., i"
  in
  Tensor.binop ~op_label:"interleave" ~compose_op:Defined_by_cd_logic ~op_asn ~grad_asn

let threefry4x32_crypto =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =: v1 ^^^^ v2 in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ = Asgns.empty_comp in
  fun t1 t2 ->
    Tn.update_prec t1.Tensor.value Ir.Ops.uint4x32;
    Tn.update_prec t2.Tensor.value Ir.Ops.uint4x32;
    fun ?grad_spec
      ?label
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
          ~op_label:"threefry4x32_crypto" ?grad_spec ?label ?batch_dims ?batch_axes ?input_dims
          ?output_dims ?input_axes ?output_axes ?deduced ()
      in
      (* Set output precision to uint4x32 *)
      Tn.update_prec result.value Ir.Ops.uint4x32;
      result

let threefry4x32_light =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =: v1 ^^ v2 in
  let%cd grad_asn ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ = Asgns.empty_comp in
  fun t1 t2 ->
    Tn.update_prec t1.Tensor.value Ir.Ops.uint4x32;
    Tn.update_prec t2.Tensor.value Ir.Ops.uint4x32;
    fun ?grad_spec
      ?label
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
          ~op_label:"threefry4x32_light" ?grad_spec ?label ?batch_dims ?batch_axes ?input_dims
          ?output_dims ?input_axes ?output_axes ?deduced ()
      in
      (* Set output precision to uint4x32 *)
      Tn.update_prec result.value Ir.Ops.uint4x32;
      result

let threefry4x32 =
 (* Select based on configuration *)
 fun t1 t2 ->
  let variant = Utils.settings.default_prng_variant in
  if String.equal variant "crypto" then threefry4x32_crypto t1 t2 else threefry4x32_light t1 t2

let fma ~grad_spec t1 t2 t3 =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~t2 ~t3 ~projections = v =: fma v1 v2 v3 in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~t3 ~projections =
    g1 =+ g * v2;
    g2 =+ g * v1;
    g3 =+ g
  in
  Tensor.ternop ~op_label:"fma" ~ternary_op:Pointwise_tern ~op_asn ~grad_asn ~grad_spec t1 t2 t3

let where ~grad_spec t1 t2 t3 =
  let module NTDSL = NTDSL_before_div in
  let%cd op_asn ~t ~t1 ~t2 ~t3 ~projections = v =: where v1 v2 v3 in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~t3 ~projections =
    g2 =+ where v1 g 0;
    g3 =+ where v1 0 g
  in
  Tensor.ternop ~op_label:"where" ~ternary_op:Pointwise_tern ~op_asn ~grad_asn ~grad_spec t1 t2 t3

(** Similar to the explicit mode of [numpy.einsum], the binary variant. Can compute various forms of
    matrix multiplication, inner and outer products, etc.

    Note that ["a,b->c"] from [numpy] is ["a;b=>c"] in OCANNL, since ["->"] is used to separate the
    input and the output axes. *)
let einsum ?(capture_dims = []) spec =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =:+ v1 * v2 in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g * v2;
    g2 =+ v1 * g
  in
  Tensor.binop ~op_label:";=>" ~compose_op:(Einsum (spec, capture_dims)) ~op_asn ~grad_asn

(** Like [einsum], but adds instead than multiplying the resulting values. *)
let outer_sum ?(capture_dims = []) spec =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =:+ v1 + v2 in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g;
    g2 =+ g
  in
  Tensor.binop ~op_label:";=>+" ~compose_op:(Einsum (spec, capture_dims)) ~op_asn ~grad_asn

(** Similar to the explicit mode of [numpy.einsum], the unary variant. Can permute axes, extract
    diagonals, compute traces etc.

    Note that ["a->c"] from [numpy] is ["a=>c"] in OCANNL, since ["->"] is used to separate the
    input and the output axes. *)
let einsum1 ?(capture_dims = []) spec =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~t ~t1 ~projections = v =:+ v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ g in
  Tensor.unop ~transpose_op:(Shape.Permute (spec, capture_dims)) ~op_asn ~grad_asn ~op_label:"=>"

module NDO_before_einmax1 = struct
  let ( + ) ?label t1 t2 = add ?label ~grad_spec:Prohibit_grad t1 t2 ()
  let where ?label t1 t2 t3 = where ?label ~grad_spec:Prohibit_grad t1 t2 t3 ()
  let not ?label t = not ?label ~grad_spec:Prohibit_grad t ()
  let ( < ) ?label t1 t2 = lt ?label ~grad_spec:Prohibit_grad t1 t2 ()
  let ( = ) ?label t1 t2 = eq ?label ~grad_spec:Prohibit_grad t1 t2 ()
end

let einmax1 ?(capture_dims = []) spec =
  let module NTDSL = struct
    include Initial_NTDSL
    module O = NDO_before_einmax1
  end in
  let%cd op_asn ~t ~t1 ~projections = v =:@^ v1 in
  let%cd grad_asn ~t ~g ~t1 ~projections =
    { cond_rhs1 } =: eq (t, t1);
    g1 =+ where cond_rhs1 g 0
  in
  Tensor.unop ~transpose_op:(Shape.Permute (spec, capture_dims)) ~op_asn ~grad_asn ~op_label:"@^=>"

(** This generalizes the tropical matrix multiplication to arbitrary indices combinations.

    LIMITATION: Backpropagation is only correct when the RHS1 (t1) index space includes the RHS2
    (t2) index space. This is the case for convolution-like operations where the kernel indices are
    contracted with strided input indices. For general tropical operations where RHS2 has
    independent indices, the g2 gradient will be incorrect. *)
let tropical ?(capture_dims = []) spec =
  let module NTDSL = struct
    include Initial_NTDSL
    module O = NDO_before_einmax1
  end in
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =:@^ v1 + v2 in
  let%cd grad_asn ~t ~g ~t1 ~t2 ~projections =
    (* Use _rhs1 suffix for both: gives input shape (ih,iw) = (oh,ow) x (wh,ww) outer product. This
       correctly tracks which (input position, kernel position) pair achieved argmax. *)
    { sum_rhs1 } =:@^ add (t1, t2);
    { cond_rhs1 } =: eq (t, sum_rhs1);
    g1 =+ where cond_rhs1 g 0;
    g2 =+ where cond_rhs1 g 0
  in
  Tensor.binop ~compose_op:(Shape.Einsum (spec, capture_dims)) ~op_asn ~grad_asn ~op_label:"@^=>+"

(** A fully-shape-inferred tensor that is initialized with the offset of each cell. *)
let offsets = Tensor.term ~fetch_op:Range_over_offsets ?init_data:None

(** [range] is a 1D tensor of shape [upto], spans [0] inclusive, [upto] exclusive. *)
let range ?(label = []) ?(grad_spec = Tensor.Prohibit_grad) ?axis_label upto =
  let result =
    Tensor.term ~fetch_op:Range_over_offsets ~grad_spec ~batch_dims:[]
      ~label:(("0" ^ "..." ^ Int.to_string (upto - 1)) :: label)
      ~input_dims:[]
  in
  match axis_label with
  | None -> result ~output_dims:[ upto ] ()
  | Some l -> result ~output_axes:[ (l, upto) ] ()

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
let stop_gradient =
  let module NTDSL = Initial_NTDSL in
  let grad_asn ~t:_ ~g:_ ~t1:_ ~projections:_ = Asgns.empty_comp in
  let%cd op_asn ~t ~t1 ~projections = v =: v1 in
  Tensor.unop ~op_label:"stop_grad" ~transpose_op:Pointwise_un ~op_asn ~grad_asn
    ~grad_spec:Prohibit_grad

let slice (batch_idx : Idx.static_symbol) =
  let module NTDSL = Initial_NTDSL in
  let op_asn ~t ~t1 ~projections =
    Asgns.to_comp
    @@ Fetch
         {
           array = t.Tensor.value;
           fetch_op = Slice { batch_idx; sliced = t1.Tensor.value };
           dims = lazy (Lazy.force projections.Tensor.projections).Idx.lhs_dims;
         }
  in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ g in

  Tensor.unop ~transpose_op:(Batch_slice batch_idx) ~op_asn ~grad_asn ~op_label:"@|"

let embed_symbol ?grad_spec ?(label = []) static_sym =
  Tensor.term ~fetch_op:(Embed_symbol static_sym) ?grad_spec ~label:("!@" :: label) ~batch_dims:[]
    ~input_dims:[] ~output_dims:[ 1 ] ()

let embed_self_id ?grad_spec ?(label = []) () =
  Tensor.term ~fetch_op:Embed_self_id ?grad_spec ~label:("!@self_id" :: label) ~batch_dims:[]
    ~input_dims:[] ~output_dims:[ 1 ] ()

let embed_dim ?grad_spec ?(label = []) variable_ref =
  Tensor.term ~fetch_op:(Embed_dim variable_ref.Shape.var_ref) ?grad_spec
    ~label:("!@self_id" :: label) ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] ()

let uniform ?grad_spec () =
  uint4x32_to_prec_uniform ?grad_spec
    (threefry4x32
       (threefry4x32 (embed_self_id ()) (Tensor.get_random_seed ()) ())
       (Tensor.term ~fetch_op:Range_over_offsets ~grad_spec:Prohibit_grad
          ~label:[ "range_over_offsets" ] ())
       ())

(** Generates a single uniform random number using a counter symbol for PRNG state. This is useful
    for sequential sampling in recurrent contexts. *)
let uniform_at ?grad_spec counter =
  uint4x32_to_prec_uniform ?grad_spec
    (threefry4x32
       (threefry4x32 (threefry4x32 (embed_self_id ()) (Tensor.get_random_seed ()) ()) counter ())
       (Tensor.term ~fetch_op:Range_over_offsets ~grad_spec:Prohibit_grad
          ~label:[ "range_over_offsets" ] ())
       ())

(** A wasteful variant of {!uniform} that produces a single value from each 4x32 random bits. The
    bit-spreading in int32_to_uint4x32/uint32_to_uint4x32 ensures good entropy even with the 2-round
    "light" threefry variant. *)
let uniform1 ?grad_spec () =
  uint4x32_to_prec_uniform1 ?grad_spec
    (threefry4x32
       (threefry4x32 (embed_self_id ()) (Tensor.get_random_seed ()) ())
       (Tensor.term ~fetch_op:Range_over_offsets ~grad_spec:Prohibit_grad
          ~label:[ "range_over_offsets" ] ())
       ())

(** A wasteful variant of {!uniform_at} that produces a single value from each 4x32 random bits. The
    bit-spreading in int32_to_uint4x32/uint32_to_uint4x32 ensures good entropy even with the 2-round
    "light" threefry variant. *)
let uniform_at1 ?grad_spec counter =
  uint4x32_to_prec_uniform1 ?grad_spec
    (threefry4x32
       (threefry4x32 (threefry4x32 (embed_self_id ()) (Tensor.get_random_seed ()) ()) counter ())
       (Tensor.term ~fetch_op:Range_over_offsets ~grad_spec:Prohibit_grad
          ~label:[ "range_over_offsets" ] ())
       ())

(** The input [i] dimensions default to empty. The batch and output dimensions will be inferred if
    omitted. Note: the data should have no padding and if padding is inferred, the data will be
    copied; otherwise, the resulting tensor value shares host memory with the ndarray. *)
let reshape ~l ?b ?(i = []) ?o ndarray =
  Tensor.term ~init_data:(Asgns.Reshape ndarray) ?batch_dims:b ~label:[ l ] ~input_dims:i
    ?output_dims:o

(** The dimensions are taken from the provided ndarray, but the split into axis kinds still needs to
    be inferred (or provided). Assumes no padding. Input axes are not inferred (empty if omitted).
    See also: {!reshape} and {!TDSL.wrap_param}. *)
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
  Tensor.term ~init_data:(Reshape ndarray) ~label:[ l ] ~input_dims:[] ?input_axes:None ~output_dims
    ?output_axes:None

(** Creates a tensor by initializing values using a function from indices to values. The dimensions
    are split into axis kinds as specified, there is no shape inference. Recall that input axes are
    rightmost. *)
let init ~l ~prec ?(b = []) ?(i = []) ?(o = []) ~f =
  let all_dims = Array.of_list (b @ o @ i) in
  let ndarray = Ir.Ndarray.init_array ~debug:l prec ~dims:all_dims ~padding:None ~f in
  Tensor.term ~init_data:(Asgns.Keep_shape_no_padding ndarray) ~batch_dims:b ~label:[ l ]
    ~input_dims:i ~output_dims:o

module Make_DSL (Grad_spec : sig
  val grad_spec : Tensor.grad_spec
end) =
struct
  include Grad_spec

  let term = Tensor.term ~grad_spec
  let number = Tensor.number ~grad_spec
  let bits = Tensor.bits ~grad_spec
  let ndarray = Tensor.ndarray ~grad_spec
  let threefry4x32 = threefry4x32 ~grad_spec
  let uint4x32_to_prec_uniform = uint4x32_to_prec_uniform ~grad_spec
  let uint4x32_to_prec_uniform1 = uint4x32_to_prec_uniform1 ~grad_spec
  let embed_self_id = embed_self_id ~grad_spec

  (** The default initialization operation for {!param} calls.

      To avoid user surprises, this defaults to {!uniform1} which does not impose constraints on the
      shape of the tensor, but for efficiency, consider setting this to
      [uniform ~grad_spec:Require_grad] or [normal ~grad_spec:Require_grad] instead. *)
  let default_param_init = ref (uniform1 ~grad_spec:Require_grad)
  (* Useful for debugging: *)
  (* let default_param_init =
    ref (fun () -> Tensor.term ~grad_spec:Require_grad ?init_data:None ~fetch_op:(Constant 0.)) *)

  let param ?value ?values ?param_init =
    let grad_spec =
      if Tensor.is_prohibit_grad Grad_spec.grad_spec then Tensor.Prohibit_grad else Require_grad
    in
    let t =
      match (value, values, param_init) with
      | Some value, None, None -> Tensor.term_init ~grad_spec [| value |]
      | None, Some values, None -> Tensor.term_init ~grad_spec values
      | None, None, Some param_init -> param_init
      | None, None, None ->
          if Tensor.is_prohibit_grad Grad_spec.grad_spec then
            raise
            @@ Utils.User_error
                 "Operation.Make_DSL.param: in non-grad contexts like %%extend_dsls, inline \
                  definitions with initialization require explicit initialization"
          else !default_param_init ()
      | _ -> invalid_arg "TDSL.param: at most one of value, values, and param_init can be set"
    in
    Tensor.param ~t

  let einsum = einsum ~grad_spec
  let outer_sum = outer_sum ~grad_spec
  let einsum1 = einsum1 ~grad_spec
  let einmax1 = einmax1 ~grad_spec
  let tropical = tropical ~grad_spec
  let offsets = offsets ~grad_spec
  let range = range ~grad_spec
  let range_of_shape = range_of_shape ~grad_spec
  let stop_gradient = stop_gradient
  let reshape = reshape ~grad_spec
  let wrap = wrap ~grad_spec
  let wrap_padded = wrap_padded ~grad_spec
  let rebatch = rebatch ~grad_spec
  let init = init ~grad_spec
  let uniform = uniform ~grad_spec

  (** The input and output dimensions will be inferred if omitted. See {!reshape}. *)
  let reshape_param ~l ?i ?o ndarray =
    let t = Tensor.term ~grad_spec:Require_grad ~init_data:(Reshape ndarray) ?fetch_op:None in
    Tensor.param ~t ?input_dims:i ?output_dims:o l

  (** See {!wrap}. *)
  let wrap_param ~l ?i ?o ndarray =
    let t =
      Tensor.term ~grad_spec:Require_grad ~init_data:(Keep_shape_no_padding ndarray) ?fetch_op:None
    in
    Tensor.param ?input_dims:i ?output_dims:o ~t l

  let matmul = matmul ~grad_spec
  let pointmul = pointmul ~grad_spec
  let add = add ~grad_spec
  let pointpow = pointpow ~grad_spec
  let relu = relu ~grad_spec
  let sat01 = sat01 ~grad_spec
  let fma = fma ~grad_spec
  let number_int ?label ?axis_label i = Tensor.number ?label ?axis_label ~grad_spec (Float.of_int i)
  let embed_symbol = embed_symbol ~grad_spec
  let embed_dim = embed_dim ~grad_spec
  let sub = sub ~grad_spec
  let pointdiv = pointdiv ~grad_spec
  let slice = slice ~grad_spec
  let exp = exp ~grad_spec
  let log = log ~grad_spec
  let log2 = log2 ~grad_spec
  let sin = sin ~grad_spec
  let cos = cos ~grad_spec
  let neg = neg ~grad_spec
  let sqrt = sqrt ~grad_spec
  let recip = recip ~grad_spec
  let recip_sqrt = recip_sqrt ~grad_spec
  let tanh = tanh ~grad_spec
  let where = where ~grad_spec
  let not = not ~grad_spec
  let lt = lt ~grad_spec
  let eq = eq ~grad_spec
  let ne = ne ~grad_spec
  let uniform_at = uniform_at ~grad_spec
  let uniform1 = uniform1 ~grad_spec
  let uniform_at1 = uniform_at1 ~grad_spec
  let interleave = interleave ~grad_spec

  module O = struct
    let ( * ) ?label t1 t2 = matmul ?label t1 t2 ()
    let ( *. ) ?label t1 t2 = pointmul ?label t1 t2 ()
    let ( + ) ?label t1 t2 = add ?label t1 t2 ()
    let threefry4x32 ?label t1 t2 = threefry4x32 ?label t1 t2 ()
    let uint4x32_to_prec_uniform ?label t1 = uint4x32_to_prec_uniform ?label t1 ()
    let uint4x32_to_prec_uniform1 ?label t1 = uint4x32_to_prec_uniform1 ?label t1 ()
    let ( **. ) ?label base exp = pointpow ?label exp base ()
    let relu ?label t = relu ?label t ()
    let sat01 ?label t = sat01 ?label t ()
    let fma ?label t1 t2 t3 = fma ?label t1 t2 t3 ()
    let ( !. ) f = number f
    let ( !.. ) ?label i = number ?label @@ Float.of_int i
    let ( !% ) ?label i = bits ?label i
    let ( !@ ) = embed_symbol
    let dim = embed_dim
    let ( - ) ?label t1 t2 = sub ?label t1 t2 ()
    let ( ~- ) ?label t = pointmul ?label (number (-1.)) t ()
    let ( /. ) ?label t1 t2 = pointdiv ?label t1 t2 ()
    let ( @| ) ?label t1 idx = slice ?label idx t1 ()
    let exp ?label t = exp ?label t ()
    let log ?label t = log ?label t ()
    let log2 ?label t = log2 ?label t ()
    let sin ?label t = sin ?label t ()
    let cos ?label t = cos ?label t ()
    let neg ?label t = neg ?label t ()
    let sqrt ?label t = sqrt ?label t ()
    let recip ?label t = recip ?label t ()
    let recip_sqrt ?label t = recip_sqrt ?label t ()
    let tanh ?label t = tanh ?label t ()
    let where ?label t1 t2 t3 = where ?label t1 t2 t3 ()
    let not ?label t = not ?label t ()
    let ( < ) ?label t1 t2 = lt ?label t1 t2 ()
    let ( = ) ?label t1 t2 = eq ?label t1 t2 ()
    let ( <> ) ?label t1 t2 = ne ?label t1 t2 ()
    let embed_self_id = embed_self_id
    let einsum ?label ?capture_dims spec t1 t2 = einsum ?label ?capture_dims spec t1 t2 ()
    let outer_sum ?label ?capture_dims spec t1 t2 = outer_sum ?label ?capture_dims spec t1 t2 ()
    let einsum1 ?label ?capture_dims spec t1 = einsum1 ?label ?capture_dims spec t1 ()
    let einmax1 ?label ?capture_dims spec t1 = einmax1 ?label ?capture_dims spec t1 ()
    let tropical ?label ?capture_dims spec t1 t2 = tropical ?label ?capture_dims spec t1 t2 ()
    let offsets ?label () = offsets ?label ()
    let uniform ?label () = uniform () ?label ()
    let uniform_at ?label counter = uniform_at ?label counter ()
    let uniform1 ?label () = uniform1 () ?label ()
    let uniform_at1 ?label counter = uniform_at1 ?label counter ()
    let interleave ?label t1 t2 = interleave ?label t1 t2 ()
  end
end

module DSL_modules = struct
  module Ir = Ir
  module Row = Row
  module Shape = Shape
  module Tensor = Tensor

  module TDSL = Make_DSL (struct
    let grad_spec = Tensor.If_needed
  end)

  module NTDSL = Make_DSL (struct
    let grad_spec = Tensor.Prohibit_grad
  end)

  module PDSL = Make_DSL (struct
    let grad_spec = Tensor.Require_grad
  end)
end
