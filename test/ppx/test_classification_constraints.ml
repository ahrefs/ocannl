open Ocannl.Operation.DSL_modules

let%op tail_plain () = fun x -> x + 1.0
let%op tail_typed () = (fun x -> x + 1.0 : Tensor.t -> Tensor.t)
let head_plain x = [%op add x x]
let head_typed x = [%op (add : _) x x]
let pipeline_plain x = [%op x |> add x]
let pipeline_typed x = [%op x |> (add : _) x]
let grouping_plain x = [%op Some (x + 1.0, x + 2.0)]
let grouping_typed x = [%op Some ((x + 1.0, x + 2.0) : Tensor.t * Tensor.t)]
let binary_plain x = [%op add (x, 2.0)]
let binary_typed x = [%op add ((x, 2.0) : Tensor.t * Tensor.t)]
let ternary_plain x = [%op fma (x, 2.0, 3.0)]

let ternary_typed x =
  [%op fma (((x, 2.0, 3.0) : Tensor.t * Tensor.t * Tensor.t) : Tensor.t * Tensor.t * Tensor.t)]

let optout_calls = ref 0

let%op tail_optout () =
  (fun x ->
     Stdlib.incr optout_calls;
     x + 1.0
    : Tensor.t -> Tensor.t)
    [@ocannl.no_scheme]

let%op labelled_binary ~label:__a x = add ((x, x) : Tensor.t * Tensor.t)
let%op labelled_ternary ~label:__b x = fma ((x, x, x) : Tensor.t * Tensor.t * Tensor.t)

let _ =
  ( tail_plain,
    tail_typed,
    head_plain,
    head_typed,
    pipeline_plain,
    pipeline_typed,
    grouping_plain,
    grouping_typed,
    binary_plain,
    binary_typed,
    ternary_plain,
    ternary_typed,
    tail_optout,
    labelled_binary,
    labelled_ternary )
