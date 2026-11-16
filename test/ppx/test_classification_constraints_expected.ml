open Ocannl.Operation.DSL_modules
let tail_plain =
  let open! TDSL.O in
    fun () ->
      let __ocannl_body x =
        (+) ?label:(Some ["tail_plain"]) x (TDSL.number ?label:None 1.0) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"tail_plain" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let tail_typed =
  let open! TDSL.O in
    fun () ->
      (let __ocannl_body x =
         (+) ?label:(Some ["tail_typed"]) x (TDSL.number ?label:None 1.0) in
       let __ocannl_scheme =
         Tensor.Scheme.For_ppx.closure ~name:"tail_typed" ~arity:1
           (function | x::[] -> __ocannl_body x | _ -> assert false) in
       fun x ->
         Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
           (fun () -> __ocannl_body x) : Tensor.t -> Tensor.t)
let head_plain x = let open! TDSL.O in add ?label:(Some []) x x
let head_typed x = let open! TDSL.O in (add ?label:(Some []) : _) x x
let pipeline_plain x =
  let open! TDSL.O in let __x = x in (add ?label:None x) __x
let pipeline_typed x =
  let open! TDSL.O in let __x = x in ((add ?label:None : _) x) __x
let grouping_plain x =
  let open! TDSL.O in
    Some
      (((+) ?label:(Some []) x (TDSL.number ?label:None 1.0)),
        ((+) ?label:(Some []) x (TDSL.number ?label:None 2.0)))
let grouping_typed x =
  let open! TDSL.O in
    Some
      ((((+) ?label:(Some []) x (TDSL.number ?label:None 1.0)),
         ((+) ?label:(Some []) x (TDSL.number ?label:None 2.0))) : (Tensor.t
                                                                    *
                                                                    Tensor.t))
let binary_plain x =
  let open! TDSL.O in (add ?label:(Some [])) x (TDSL.number ?label:None 2.0)
let binary_typed x =
  let open! TDSL.O in
    let (__a, __b) =
      ((x, (TDSL.number ?label:None 2.0)) : (Tensor.t * Tensor.t)) in
    (add ?label:(Some [])) __a __b
let ternary_plain x =
  let open! TDSL.O in
    (fma ?label:(Some [])) x (TDSL.number ?label:None 2.0)
      (TDSL.number ?label:None 3.0)
let ternary_typed x =
  let open! TDSL.O in
    let (__a, __b, __c) =
      (((x, (TDSL.number ?label:None 2.0), (TDSL.number ?label:None 3.0)) : 
      (Tensor.t * Tensor.t * Tensor.t)) : (Tensor.t * Tensor.t * Tensor.t)) in
    (fma ?label:(Some [])) __a __b __c
let _ =
  (tail_plain, tail_typed, head_plain, head_typed, pipeline_plain,
    pipeline_typed, grouping_plain, grouping_typed, binary_plain,
    binary_typed, ternary_plain, ternary_typed)
