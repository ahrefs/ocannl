open Base
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Tensor = Ocannl_tensor.Tensor

(** Precision-assignment policy over a model (gh-ocannl-492 task 1): assigns storage precisions
    across a tensor graph instead of the user hand-annotating every tensor. This is the
    storage-precision sibling of the compute-precision policy in {!Ir.Numerics} — the two are the
    "one roof" gh-ocannl-478 asks for.

    Semantics of {!apply}:

    - Only tensor nodes with no user-specified precision are touched ([Tn.get_specified_prec]
      pre-check) — explicit annotations always win, including ndarray-backed data nodes whose
      precision comes from their host array.
    - Only nodes whose (inferred-so-far) precision is a float are re-assigned: integer index tensors
      and uint4x32 RNG-state chains keep their precisions. For not-yet-forced inference chains the
      float check is deferred into the lazy ([Tn.update_prec ~only_if]), so applying a policy forces
      nothing.
    - Assignment classes are structural: [param_prec] for [root.params] (model state, including
      non-trainable buffers), [activation_prec] for every other tensor's value node (op results,
      inputs, constants), [grad_prec] for all gradient nodes. Op-kind selectivity ("softmax stays
      f32") is the [except] predicate's job — or the training recipe's, when gh-ocannl-492 grows
      one; label matching over [Tn.label] is the intended idiom.
    - [except]-ed float nodes are PINNED at the session default precision for their class
      ([Tensor.default_value_prec] / [default_grad_prec]), not merely skipped: precision inference
      propagates top-down as well as bottom-up (see the [top_down_prec] test), so a skipped node
      between policy-assigned neighbors would be inferred into the reduced precision anyway.
    - A [None] field leaves that class alone (inference proceeds as without a policy).

    Call {!apply} after the model (and its gradient graph, if any) is fully constructed and before
    the first compilation: assignments to already-settled nodes fail loudly (the settlement guards
    in [Tn.update_prec]), and op construction after [apply] would infer from pre-policy precisions.

    fp8 note (gh-ocannl-481): the codomain is [Ops.prec], so when a second fp8 format (e4m3) lands
    as a precision, per-class fp8 assignment (e4m3 weights/activations, e5m2 gradients) is
    expressible here without interface changes. *)

type t = {
  param_prec : Ops.prec option;  (** For values of model-state parameters ([root.params]). *)
  activation_prec : Ops.prec option;
      (** For values of every non-param tensor: op results, inputs, unannotated constants. *)
  grad_prec : Ops.prec option;  (** For gradient nodes (of params and intermediates alike). *)
}

(** The same precision for all three classes — the "full bf16 / full f16" styles. Mixed styles (e.g.
    reduced compute with [grad_prec = Some Ops.single]) are record literals away. *)
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
  let rec walk (t : Tensor.t) =
    if not (Set.mem !visited t) then (
      visited := Set.add !visited t;
      let is_param = Set.mem root.Tensor.params t in
      apply_value (if is_param then policy.param_prec else policy.activation_prec) t.Tensor.value;
      Option.iter t.Tensor.diff ~f:(fun d -> apply_grad policy.grad_prec d.Tensor.grad);
      List.iter t.Tensor.children ~f:(fun sub -> walk sub.Tensor.subtensor))
  in
  walk root
