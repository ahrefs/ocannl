open Ocannl.Nn_blocks.DSL_modules
let binary_tensor source other =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (NTDSL.add source other)) "w") () in
    w
let binary_literal source =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (NTDSL.add source (TDSL.number ?label:None 0.0)))
         "w") () in
    w
let binary_sub source =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (NTDSL.sub source (TDSL.number ?label:None 2.0)))
         "w") () in
    w
let binary_mul source =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some
                        (NTDSL.pointmul source (TDSL.number ?label:None 2.0)))
         "w") () in
    w
let binary_div source =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some
                        (NTDSL.pointdiv source (TDSL.number ?label:None 2.0)))
         "w") () in
    w
let binary_head x y = let open! TDSL.O in ((add : _) ?label:(Some [])) x y
let binary_both x y =
  let open! TDSL.O in
    let (__fn, (__a, __b)) =
      (((add : _) ?label:(Some [])), ((x, y) : (Tensor.t * Tensor.t))) in
    __fn __a __b
let ternary_head x =
  let open! TDSL.O in
    ((fma : _) ?label:(Some [])) x (TDSL.number ?label:None 2.0)
      (TDSL.number ?label:None 3.0)
let ternary_both x =
  let open! TDSL.O in
    let (__fn, (__a, __b, __c)) =
      ((((fma : _) : _) ?label:(Some [])),
        (((x, (TDSL.number ?label:None 2.0), (TDSL.number ?label:None 3.0)) : 
        (Tensor.t * Tensor.t * Tensor.t)) : _)) in
    __fn __a __b __c
let _ =
  (binary_tensor, binary_literal, binary_sub, binary_mul, binary_div,
    binary_head, binary_both, ternary_head, ternary_both)
