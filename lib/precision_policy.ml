open Base
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Tensor = Ocannl_tensor.Tensor

type t = {
  param_prec : Ops.prec option;
  activation_prec : Ops.prec option;
  grad_prec : Ops.prec option;
}

let uniform prec = { param_prec = Some prec; activation_prec = Some prec; grad_prec = Some prec }

let apply ?(except = fun (_ : Tn.t) -> false) policy (root : Tensor.t) =
  let apply_tn ~default prec_opt tn =
    match prec_opt with
    | None -> ()
    | Some prec ->
        if Option.is_none (Tn.get_specified_prec tn) then
          Tn.update_prec ~only_if:Ops.is_float tn (if except tn then default else prec)
  in
  let apply_value = apply_tn ~default:!Tensor.default_value_prec in
  let apply_grad = apply_tn ~default:!Tensor.default_grad_prec in
  let visited = ref (Set.empty (module Tensor)) in
  let first_visit t =
    let fresh = not (Set.mem !visited t) in
    if fresh then visited := Set.add !visited t;
    fresh
  in
  let apply_grad_of (t : Tensor.t) =
    Option.iter t.Tensor.diff ~f:(fun d -> apply_grad policy.grad_prec d.Tensor.grad)
  in
  (* A parameter's children are its initializer expression. Leave that cone to inference's
     promote-only link to the parameter; a node also read by the model is reached there. *)
  let rec walk_model (t : Tensor.t) =
    if first_visit t then (
      let is_param = Set.mem root.Tensor.params t in
      apply_value (if is_param then policy.param_prec else policy.activation_prec) t.Tensor.value;
      apply_grad_of t;
      if not is_param then
        List.iter t.Tensor.children ~f:(fun sub -> walk_model sub.Tensor.subtensor))
  in
  walk_model root
