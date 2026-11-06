# Bidirectional precision inference

OCANNL features a rudimentary bidirectional precision inference. It is much less powerful than the constraints-based shape and projections inference. It is somewhat prominent because it contributes the `top_down_prec` flag to the central `Tensor.t` type.

Tensors that choose `top_down_prec=true` "detach" themselves from their defining tensor expression as far as precision goes, and get precision propagated into them from the use sites. By default, terminal tensors are `top_down_prec=true`, and non-terminal tensors are `top_down_prec=false`, except for all the parameter tensors (created via `Tensor.param`), and results of the operation `uint4x32_to_prec_uniform`, that are also `top_down_prec=true`. When a tensor precision is set by the user via `Tnode.update_prec`, this setting takes precedence over any inferences. When a `top_down_prec=true` tensor has its precision set by the user, it contributes this precision in the bottom up inference (together with all `top_down_prec=false` subtensors).

The core algorithm is just a couple dozen lines in the `Tensor.op` function, first the bottom-up pass:

```ocaml
  let default_prec_for default get =
    if top_down_prec then
      (* For top-down precision, don't promote from inputs *)
      lazy default
    else
      (* For bottom-up precision, only promote from non-top-down subtensors *)
      let lazy_v_precs =
        List.filter_map ordered_ts ~f:(fun ti ->
            Option.map (get ti) ~f:(fun v ->
                if ti.top_down_prec then lazy (Tn.get_specified_prec v)
                else lazy (Some (Lazy.force v.prec))))
      in
      lazy
        (List.filter_map lazy_v_precs ~f:Lazy.force
        |> List.reduce ~f:Ir.Ops.promote_prec
        |> Option.value ~default)
  in
```

and later the top-down pass, here from the value node `v`:

```ocaml
  let update_infer_prec tn prec =
    (* Instead of just checking prec, we cross-check with dims (needed for code generation), to
       catch prec forcing bugs. *)
    if not (Lazy.is_val tn.Tn.dims) then Tn.update_infer_prec tn prec
  in
  (* Apply delayed top-down precision updates to parameter subtensors *)
  List.iter top_down_ts ~f:(fun ti -> update_infer_prec ti.value v.Tn.prec);
```

## Initializers follow their parameter

A parameter's initialization expression fills the parameter, so its float tensors are at least as precise as the parameter (gh-ocannl-1967): otherwise a `double` parameter over the `single` default would start from f32-quantized draws, since the bottom-up pass stops at the first operation none of whose operands states a precision (the default initializer's centering `sub`), which settles at `Tensor.default_value_prec`. `Tensor.param` therefore links every float tensor of the expression (stopping at other parameters) to the parameter's precision, whether that one is set by the user, pinned by `Mixed_prec.with_master_weights ~master_prec`, or inferred top-down from a consumer.

The link promotes and never narrows: a `half` parameter still rounds a `single` draw once. Narrowing an initializer to a reduced parameter precision is `Precision_policy.apply`'s assignment.

A parameter's precision can depend on a consumer that reads one of these tensors bottom-up (a tensor shared between an initializer and the model), so the link is `Tnode.follow_prec` rather than `update_infer_prec`: a linked tensor keeps answering bottom-up readers with the precision it had before the link (`Tnode.bottom_up_prec`), which keeps the lazy computation acyclic and independent of forcing order.
