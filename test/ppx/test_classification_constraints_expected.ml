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
    let (__fn, (__a, __b)) =
      ((add ?label:(Some [])),
        ((x, (TDSL.number ?label:None 2.0)) : (Tensor.t * Tensor.t))) in
    __fn __a __b
let ternary_plain x =
  let open! TDSL.O in
    (fma ?label:(Some [])) x (TDSL.number ?label:None 2.0)
      (TDSL.number ?label:None 3.0)
let ternary_typed x =
  let open! TDSL.O in
    let (__fn, (__a, __b, __c)) =
      ((fma ?label:(Some [])),
        (((x, (TDSL.number ?label:None 2.0), (TDSL.number ?label:None 3.0)) : 
        (Tensor.t * Tensor.t * Tensor.t)) : (Tensor.t * Tensor.t * Tensor.t))) in
    __fn __a __b __c
let optout_calls = ref 0
let tail_optout =
  let open! TDSL.O in
    fun () ->
      (((fun x ->
           Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
             (fun () -> Stdlib.incr optout_calls);
           (+) ?label:(Some ["tail_optout"]) x (TDSL.number ?label:None 1.0) : 
      Tensor.t -> Tensor.t))[@ocannl.no_scheme ])
let labelled_binary =
  let open! TDSL.O in
    fun ~label:__a x ->
      let (__fn, (__a, __b)) =
        ((add
            ?label:(Some
                      (List.concat
                         [["labelled_binary"];
                         (__a : string list);
                         (x.Tensor.value).Ir.Tnode.label]))),
          ((x, x) : (Tensor.t * Tensor.t))) in
      __fn __a __b
let labelled_ternary =
  let open! TDSL.O in
    fun ~label:__b x ->
      let (__fn, (__a, __b, __c)) =
        ((fma
            ?label:(Some
                      (List.concat
                         [["labelled_ternary"];
                         (__b : string list);
                         (x.Tensor.value).Ir.Tnode.label]))),
          ((x, x, x) : (Tensor.t * Tensor.t * Tensor.t))) in
      __fn __a __b __c
let _ =
  (tail_plain, tail_typed, head_plain, head_typed, pipeline_plain,
    pipeline_typed, grouping_plain, grouping_typed, binary_plain,
    binary_typed, ternary_plain, ternary_typed, tail_optout, labelled_binary,
    labelled_ternary)
