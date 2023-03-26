(** Computational primitives for neural networks, integrating [Formula] with [Code]. *)

open Base

module CDSL = Code.CDSL

let add =
  let open Code in
  let module NFDSL = struct module O = struct end end in
  let%nn_cd op_body ~(n:NodeUI.t) ~(n1:NodeUI.t) ~(n2:NodeUI.t) projections =
    n =: n1 + n2 in
  let%nn_cd grad_body ~(n:NodeUI.t) ~(n1:NodeUI.t) ~(n2:NodeUI.t) projections =
    n1.grad =+ n.grad || n2.grad =+ n.grad in
  Formula.binop ~compose_op:Pointwise_bin ~op_label:"+" ~op_body ~grad_body

let mul compose_op =
  let open Code in
  let module NFDSL = struct module O = struct end end in
  let%nn_cd op_body ~(n:NodeUI.t) ~(n1:NodeUI.t) ~(n2:NodeUI.t) projections = 
    n =: n1 * n2 in
  let%nn_cd grad_body ~(n:NodeUI.t) ~(n1:NodeUI.t) ~(n2:NodeUI.t) projections =
    n1.grad =+ n.grad * n2 || n2.grad =+ n1 * n.grad in
  Formula.binop ~compose_op 
    ~op_label:(if Shape.equal_compose_type compose_op Pointwise_bin then "*." else "*")
    ~op_body ~grad_body

let pointmul = mul Pointwise_bin

(* N1: AxB, N2 BxC, N: AxC, A: output of N1, B: input/output of N1/N2, C: input of N2.
   Although the matrix algebra would require that we insert additional transposes in gradient multiplies:
   AxB = AxC * CxB = AxC * (BxC)^T -> N1g += Ng * N2v^T,
   BxC = BxA * AxC = (AxB)^T * AxC -> N2g += N1v^T * Ng,
   in our setup there is no transposing to do, since the projections produce correct indices for their
   corresponding matrices. *)

let matmul = mul Compose

(** Similar to the explicit mode of [numpy.einsum], the binary variant. Can compute various forms of
    matrix multiplication, inner and outer products, etc.

    Note that ["a,b->c"] from [numpy] is ["a;b=>c"] in OCANNL, since ["->"] is used to separate the input
    and the output axes. *)
let einsum spec =
  let open Code in
  let module NFDSL = struct module O = struct end end in
  let%nn_cd op_body ~(n:NodeUI.t) ~(n1:NodeUI.t) ~(n2:NodeUI.t) projections =
    n =+ n1 * n2 in
  let%nn_cd grad_body ~(n:NodeUI.t) ~(n1:NodeUI.t) ~(n2:NodeUI.t) projections =
    n1.grad =+ n.grad * n2 || n2.grad =+ n1 * n.grad in
  Formula.binop ~compose_op:(Einsum spec) ~op_label:";=>" ~op_body ~grad_body

(** Similar to the explicit mode of [numpy.einsum], the unary variant. Can permute axes, extract diagonals,
    compute traces etc.

    Note that ["a->c"] from [numpy] is ["a=>c"] in OCANNL, since ["->"] is used to separate the input
    and the output axes. *)
let einsum1 spec =
  let open Code in
  let module NFDSL = struct module O = struct end end in
  let%nn_cd op_body ~(n:NodeUI.t) ~(n1:NodeUI.t) projections =
    n =+ n1 in
  let%nn_cd grad_body ~(n:NodeUI.t) ~(n1:NodeUI.t) projections =
    n1.grad =+ n.grad in
  Formula.unop ~transpose_op:(Permute spec) ~op_label:"=>" ~op_body ~grad_body

let relu =
  let open Code in
  let module NFDSL = struct module O = struct end end in
  let%nn_cd op_body ~(n:NodeUI.t) ~(n1:NodeUI.t) projections =
    n =: !/ n1 ~projections in
  let%nn_cd grad_body ~(n:NodeUI.t) ~(n1:NodeUI.t) projections =
    n1.grad =+ n -?/ n.grad in
  Formula.unop ~transpose_op:Pointwise_un ~op_label:"r" ~op_body ~grad_body

module NFO_without_pow = struct
  let ( * ) = matmul ~is_form:false
  let ( *. ) = pointmul ~is_form:false
  let (+) = add ~is_form:false
  let (!/) = relu ~is_form:false
  let (!~) label =
   Formula.term ~label ~is_form:false (Deduced_params Not_constrained) (First Standard_uniform)
  let (!.) = Formula.number ~is_form:false
  let (-) m1 m2 = m1 + !.(-1.) *. m2
  let (~-) m = !.(-1.) *. m
end

let rec pointpow ~is_form p m1: Formula.t =
  let module NFDSL = struct module O = NFO_without_pow end in
  let open Code in
  let p_f = Formula.number ~is_form p in
  let%nn_cd op_body ~(n:NodeUI.t) ~(n1:NodeUI.t) ~(n2:NodeUI.t) projections =
    n =: n1 ** n2 ~projections in
  let%nn_cd grad_body =
    if not is_form then 
      fun ~n:_ ~n1:_ ~n2:_ _projections -> Noop
    else if Float.equal p 2.0 then
      fun ~(n:NodeUI.t) ~(n1:NodeUI.t) ~n2:_ projections -> n1.grad =+ p_f *. m1 * n.grad
    else
      fun ~(n:NodeUI.t) ~(n1:NodeUI.t) ~n2:_ projections -> n1.grad =+ (p_f *. m1 **. (p -. 1.)) * n.grad in
  Formula.binop ~compose_op:Pointwise_bin ~op_label:"**." ~op_body ~grad_body ~is_form m1 p_f

let unconstrained_param ?init label =
  (* Note: no axis label so that we do not conflict with user labels. *)
  let init_op = match init with
  | None -> Code.Standard_uniform
  | Some c -> Code.Constant_of_value c in
  Formula.term ~is_form:true ~label (Deduced_params Not_constrained) (First init_op)

let range ~is_form ?(axis_label="") upto =
  Formula.term ~is_form ~label:("0"^"..."^Int.to_string upto)
   (Constant {output_dims=[upto + 1]; axis_labels=axis_label}) (First Range_over_offsets)

let range_of_shape ~is_form ?(axis_labels="") ?(batch_dims=[]) ?(input_dims=[]) ?(output_dims=[]) () =
  let spec =
    match batch_dims, input_dims with
    | [], [] -> Shape.Constant {output_dims; axis_labels}
    | _, [] -> Data {batch_dims; output_dims; axis_labels}
    | _, _ -> Transform {batch_dims; input_dims; output_dims; axis_labels} in
  let dims = Array.concat_map [|batch_dims; output_dims; input_dims|] ~f:Array.of_list in
  Formula.term ~is_form ~label:("r"^NodeUI.dims_to_string dims) spec (First Range_over_offsets)

let data ?(axis_labels="") ~label ~batch_dims ~output_dims reset_op =
  let spec = Shape.Data {batch_dims; output_dims; axis_labels} in
  Formula.term ~label ~needs_gradient:false ~is_form:true spec (Second reset_op)

let assign =
  let module NFDSL = struct module O = struct end end in
  let%nn_cd assign ~(lhs:Code.data) ~(rhs:Code.data) projections =
    lhs =: rhs ~projections in
  assign

let assign_op field ~(n:NodeUI.t) ~(n1:NodeUI.t) projections = assign ~lhs:(field n) ~rhs:(field n1) projections

(** A [stop_gradient] is an identity in the forward pass and a no-op in the backprop pass. *)
let stop_gradient =
  let grad_body ~n:_ ~n1:_ _projections = Code.Noop in
  let op_body = assign_op @@ Code.CDSL.data_of_node `Value in
  Formula.unop ~transpose_op:Pointwise_un ~op_label:"stop_grad" ~op_body ~grad_body
    ~is_form:true

(** A [stop_broadcast] mutates the partially-inferred shape of a formula in-place, substituting-in
    a [Fixed] marker on the dimensions. This way we avoid introducing a new node. *)
let stop_broadcast m = Shape.set_dims_type m.Formula.shape Shape.fixed

(** [identity] introduces a new node, which is an identity in both the forward and backward pass. *)
let identity ~is_form m =
  let grad_body ~(n:NodeUI.t) ~(n1:NodeUI.t) = 
    assign_op (Code.CDSL.data_of_node `Grad) ~n:n1 ~n1:n in
  let op_body = assign_op @@ Code.CDSL.data_of_node `Value in
  Formula.(unop ~init_shape:m.shape ~transpose_op:Pointwise_un ~op_label:"="
             ~op_body ~grad_body ~is_form)

module O = struct
  let ( * ) = matmul ~is_form:true
  let ( *. ) = pointmul ~is_form:true
  let (+) = add ~is_form:true
  let ( **. ) base exp = pointpow exp base ~is_form:true
  let (!/) = relu ~is_form:true
  let (!~) label =
   Formula.term ~label ~is_form:true (Deduced_params Not_constrained) (First Standard_uniform)
  let (!.) = Formula.number ~is_form:true
  let (-) m1 m2 = m1 + !.(-1.) *. m2
  let (~-) m = !.(-1.) *. m
  let (/.) m1 m2 = m1 *. m2 **. (-1.0)
end
      
module FDSL = struct
  include Formula.FDSL
  module O = O
  let einsum s = einsum s ~is_form:true
  let einsum1 s = einsum1 s ~is_form:true
  let unconstrained_param = unconstrained_param
  let range = range ~is_form:true
  let range_of_shape = range_of_shape ~is_form:true
  let data = data
  let stop_broadcast = stop_broadcast
  let stop_gradient = stop_gradient
end


module NFO = struct
  include NFO_without_pow
  let ( **. ) base exp = pointpow exp base ~is_form:false
  let (/.) m1 m2 = m1 *. m2 **. (-1.0)
end

module NFDSL = struct
  include Formula.NFDSL
  module O = NFO
  let einsum s = einsum s ~is_form:false
  let einsum1 s = einsum1 s ~is_form:false
  let term = Formula.term ~is_form:false
  let range = range ~is_form:false
  let range_of_shape = range_of_shape ~is_form:false
end
