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
let power_int source =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (NTDSL.pointpow (Float.of_int 2) source)) "w") () in
    w
let power_float source =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (NTDSL.pointpow 2.0 source)) "w") () in
    w
let power_expr source e =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (NTDSL.pointpow e source)) "w") () in
    w
let power_effectful source n =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some
                        (let __exponent1 = Float.of_int (!n) in
                         NTDSL.pointpow __exponent1 (incr n; source))) "w")
        () in
    w
module Custom_power =
  struct
    module TDSL =
      struct
        include TDSL
        module O =
          struct
            include TDSL.O
            let ( **. ) base _exponent = NTDSL.relu base
          end
      end
    let explicit_power source e =
      let open! TDSL.O in
        let w =
          (TDSL.param ?more_label:None ?value:None ?values:None
             ?param_init:(Some (TDSL.O.( **. ) source e)) "w") () in
        w
  end
let named_binary source other =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (NTDSL.add source other)) "w") () in
    w
let named_unary source =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (NTDSL.relu source)) "w") () in
    w
let named_ternary source =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some
                        (NTDSL.fma source (TDSL.number ?label:None 2.0)
                           (TDSL.number ?label:None 3.0))) "w") () in
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
    power_int, power_float, power_expr, power_effectful,
    Custom_power.explicit_power, named_binary, named_unary, named_ternary,
    binary_head, binary_both, ternary_head, ternary_both)
