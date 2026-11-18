open Ocannl.Nn_blocks.DSL_modules

let binary_tensor source other = [%op { w = source + other }]
let binary_literal source = [%op { w = source + 0.0 }]
let binary_sub source = [%op { w = source - 2.0 }]
let binary_mul source = [%op { w = source *. 2.0 }]
let binary_div source = [%op { w = source /. 2.0 }]
let power_int source = [%op { w = source **. 2 }]
let power_float source = [%op { w = source **. 2.0 }]
let power_expr source e = [%op { w = source **. e }]

let power_effectful source n =
  [%op
    {
      w =
        [%oc
          incr n;
          source]
        **. Float.of_int !n;
    }]

(* A power the user wrote with its qualifier keeps it: the initializer is this module's
   operation. *)
module Custom_power = struct
  module TDSL = struct
    include TDSL

    module O = struct
      include TDSL.O

      let ( **. ) base _exponent = NTDSL.relu base
    end
  end

  let explicit_power source e = [%op { w = TDSL.O.( **. ) source e }]
end

(* The constant-embedding heads take a float or an integer, not a tensor. *)
let number_float x = [%op { w = !.x }]
let number_int n = [%op { w = !..n }]
let number_nested source x = [%op { w = source + !.x }]
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
    power_int,
    power_float,
    power_expr,
    power_effectful,
    Custom_power.explicit_power,
    number_float,
    number_int,
    number_nested,
    named_binary,
    named_unary,
    named_ternary,
    binary_head,
    binary_both,
    ternary_head,
    ternary_both )
