# Bidirectional precision inference

OCANNL features a rudimentary bidirectional precision inference. It is much less powerful than the constraints-based shape and projections inference. It is somewhat prominent because it contributes the `top_down_prec` flag to the central `Tensor.t` type.

Tensors that choose `top_down_prec=true` "detach" themselves from their defining tensor expression as far as precision goes. By default tensors are `top_down_prec=false`, except for all the parameter tensors (created via `Tensor.param`), and results of the operation `uint4x32_to_prec_uniform`. When a tensor precision is set by the user via `Tnode.update_prec`, this setting takes precedence over any inferences. When a `top_down_prec=true` tensor has its precision set by the user, it contributes this precision in the bottom up inference (together with all `top_down_prec=false` subtensors).

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

