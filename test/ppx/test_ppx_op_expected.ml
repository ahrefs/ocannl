open Base
open Ocannl
open Nn_blocks.DSL_modules
let y0 =
  let hey1 =
    (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
       "hey1") () in
  let open! TDSL.O in
    (+) ?label:(Some ["y0"])
      (( *. ) ?label:None (TDSL.number (Float.of_int 2)) hey1)
      (TDSL.number (Float.of_int 3))
let y1 =
  let hey2 =
    (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
       "hey2") () in
  let open! TDSL.O in
    fun x ->
      (+) ?label:(Some
                    (List.concat [["y1"]; (x.Tensor.value).Ir.Tnode.label]))
        (( * ) ?label:None hey2 (TDSL.number (Float.of_int 2))) x
let y2 =
  let hey3 =
    (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
       "hey3") () in
  let open! TDSL.O in
    fun x1 x2 ->
      (+) ?label:(Some
                    (List.concat
                       [["y2"];
                       (x1.Tensor.value).Ir.Tnode.label;
                       (x2.Tensor.value).Ir.Tnode.label]))
        (( *. ) ?label:None x1 hey3) x2
let a =
  let open! TDSL.O in
    ((TDSL.ndarray
        [|(Float.of_int 1);(Float.of_int 2);(Float.of_int 3);(Float.of_int 4);(
          Float.of_int 5);(Float.of_int 6)|]) ~label:["a"]) ~batch_dims:[]
      ~input_dims:[3] ~output_dims:[2] ()
let b =
  let open! TDSL.O in
    ((TDSL.ndarray
        [|(Float.of_int 7);(Float.of_int 8);(Float.of_int 9);(Float.of_int 10)|])
       ~label:["b"]) ~batch_dims:[2] ~input_dims:[] ~output_dims:[2] ()
let y =
  let hey4 =
    (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
       "hey4") () in
  let open! TDSL.O in
    (+) ?label:(Some ["y"])
      (( * ) ?label:None hey4 (TDSL.number ?label:None ~axis_label:"q" 2.0))
      (TDSL.number ?label:None ~axis_label:"p" 1.0)
let z =
  let hey5 =
    (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
       "hey5") ()
  and hey6 =
    (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
       "hey6") () in
  let open! TDSL.O in
    (+) ?label:(Some ["z"])
      (( * ) ?label:None (TDSL.number ?label:None ~axis_label:"q" 2.0) hey5)
      (( * ) ?label:None hey6 (TDSL.number ?label:None ~axis_label:"p" 1.0))
let stride = 2
and dilation = 3
let z2 =
  let hey7 =
    (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
       "hey7") ()
  and hey8 =
    (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
       "hey8") () in
  let open! TDSL.O in
    einsum ?label:(Some ["z2"])
      (String.concat ~sep:""
         [Int.to_string stride; "*a+"; Int.to_string dilation; "*b,;b=>a,"])
      hey7 hey8
let z3 =
  let s = 2
  and d = 3 in
  let hey10 =
    (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
       "hey10") ()
  and hey9 =
    (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
       "hey9") () in
  let open! TDSL.O in
    einsum ?label:(Some [])
      (String.concat ~sep:""
         ["i, ";
         Int.to_string s;
         "*a+";
         Int.to_string d;
         "*bc; b => i, a, c"]) hey9 hey10
let () = ignore (y0, y1, y2, a, b, y, z, z2, z3)
let mlp_layer =
  let open! TDSL.O in
    fun ~label ~hid_dim () ->
      let b =
        ((TDSL.param ?more_label:(Some label) ?value:None ?values:None
            ?param_init:None "b") ~output_dims:[hid_dim]) ()
      and w =
        (TDSL.param ?more_label:(Some label) ?value:None ?values:None
           ?param_init:None "w") () in
      fun ~x ->
        relu ?label:(Some ["mlp_layer"])
          ((+) ?label:None (( * ) ?label:None w x) b)
let _use_layer =
  let open! TDSL.O in
    let l1 = mlp_layer ~label:["L"] ~hid_dim:3 () in
    let l2 = mlp_layer ~label:["L2"] ~hid_dim:3 () in fun x -> l1 ~x:(l2 ~x)
let _config_layer =
  let open! TDSL.O in
    fun ~label () ->
      let l = mlp_layer ~label:(label @ ["L"]) ~hid_dim:3 () in fun x -> l ~x
let _three_layer_perceptron =
  let open! TDSL.O in
    fun ~label ~dim1 ~dim2 ~dim3 () ->
      let l1 = mlp_layer ~label:(label @ ["L1"]) ~hid_dim:dim1 () in
      let l2 = mlp_layer ~label:(label @ ["L2"]) ~hid_dim:dim2 () in
      let l3 = mlp_layer ~label:(label @ ["L3"]) ~hid_dim:dim3 () in
      fun x -> l3 ~x:(l2 ~x:(l1 ~x))
