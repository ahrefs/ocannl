open Ocannl.Nn_blocks.DSL_modules

let binary_tensor source other = [%op { w = source + other }]
let binary_literal source = [%op { w = source + 0.0 }]
let binary_sub source = [%op { w = source - 2.0 }]
let binary_mul source = [%op { w = source *. 2.0 }]
let binary_div source = [%op { w = source /. 2.0 }]
let named_binary source other = [%op { w = add source other }]
let named_unary source = [%op { w = relu source }]
let named_ternary source = [%op { w = fma source 2.0 3.0 }]
let binary_head x y = [%op (add : _) (x, y)]
let binary_both x y = [%op (add : _) ((x, y) : Tensor.t * Tensor.t)]
let ternary_head x = [%op (fma : _) (x, 2.0, 3.0)]
let ternary_both x = [%op ((fma : _) : _) (((x, 2.0, 3.0) : Tensor.t * Tensor.t * Tensor.t) : _)]

let _ =
  ( binary_tensor,
    binary_literal,
    binary_sub,
    binary_mul,
    binary_div,
    named_binary,
    named_unary,
    named_ternary,
    binary_head,
    binary_both,
    ternary_head,
    ternary_both )
