open Base
open Ocannl
open Nn_blocks.DSL_modules
let y0 =
  let open! TDSL.O in
    let hey1 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey1") () in
    (+) ?label:(Some ["y0"])
      (( *. ) ?label:None (TDSL.number (Float.of_int 2)) hey1)
      (TDSL.number (Float.of_int 3))
let y1 =
  let open! TDSL.O in
    let hey2 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey2") () in
    fun x ->
      (+) ?label:(Some
                    (List.concat [["y1"]; (x.Tensor.value).Ir.Tnode.label]))
        (( * ) ?label:None hey2 (TDSL.number (Float.of_int 2))) x
let y2 =
  let open! TDSL.O in
    let hey3 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey3") () in
    fun x1 x2 ->
      (+) ?label:(Some
                    (List.concat
                       [["y2"];
                       (x1.Tensor.value).Ir.Tnode.label;
                       (x2.Tensor.value).Ir.Tnode.label]))
        (( *. ) ?label:None x1 hey3) x2
let a =
  let open! TDSL.O in
    (TDSL.ndarray
       [|(Float.of_int 1);(Float.of_int 2);(Float.of_int 3);(Float.of_int 4);(
         Float.of_int 5);(Float.of_int 6)|]) ~label:["a"] ~batch_dims:[]
      ~input_dims:[3] ~output_dims:[2] ()
let b =
  let open! TDSL.O in
    (TDSL.ndarray
       [|(Float.of_int 7);(Float.of_int 8);(Float.of_int 9);(Float.of_int 10)|])
      ~label:["b"] ~batch_dims:[2] ~input_dims:[] ~output_dims:[2] ()
let y =
  let open! TDSL.O in
    let hey4 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey4") () in
    (+) ?label:(Some ["y"])
      (( * ) ?label:None hey4
         (TDSL.ndarray [|2.0|] ~batch_dims:[] ~input_dims:[]
            ~output_axes:[("q", 1)] ()))
      (TDSL.ndarray [|1.0|] ~batch_dims:[] ~input_dims:[]
         ~output_axes:[("p", 1)] ())
let z =
  let open! TDSL.O in
    let hey5 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey5") () in
    let hey6 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey6") () in
    (+) ?label:(Some ["z"])
      (( *. ) ?label:None
         (TDSL.ndarray [|2.0|] ~batch_dims:[] ~input_dims:[]
            ~output_axes:[("q", 1)] ())
         hey5)
      (( *. ) ?label:None hey6
         (TDSL.ndarray [|1.0|] ~batch_dims:[] ~input_dims:[]
            ~output_axes:[("q", 1)] ()))
let stride = 2
and dilation = 3
and use_padding = true
let z2 =
  let open! TDSL.O in
    let hey7 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey7") () in
    let hey8 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey8") () in
    einsum ?label:(Some ["z2"])
      (String.concat ~sep:""
         [Int.to_string stride;
         "*";
         "a";
         if use_padding then "=" else "<";
         "+";
         Int.to_string dilation;
         "*";
         "b";
         "; ";
         "b";
         " => ";
         "a"]) hey7 hey8
let z3 =
  let s = 2
  and d = 3 in
  let open! TDSL.O in
    let hey9 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey9") () in
    let hey10 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey10") () in
    einsum ?label:(Some [])
      (String.concat ~sep:""
         ["i";
         ", ";
         Int.to_string s;
         "*";
         "a";
         if use_padding then "=" else "<";
         "+";
         Int.to_string d;
         "*";
         "bc";
         "; ";
         "b";
         " => ";
         "i";
         ", ";
         "a";
         ", ";
         "c"]) hey9 hey10
let concat_single =
  let open! TDSL.O in
    let hey11 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey11") () in
    concat ?label:(Some ["concat_single"]) "i => i" [|hey11|]
let concat_pair =
  let open! TDSL.O in
    let hey12 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey12") () in
    let hey13 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey13") () in
    let j = Shape.get_variable_ref "j" in
    let i = Shape.get_variable_ref "i" in
    concat ?label:(Some ["concat_pair"]) ~capture_dims:[j; i] "i; j => i ^ j"
      [|hey12;hey13|]
let initialized_no_unit =
  let open! TDSL.O in
    let init_w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (NTDSL.kaiming ~scale_sq:2.0 normal1 ())) "init_w")
        () in
    fun x ->
      relu
        ?label:(Some
                  (List.concat
                     [["initialized_no_unit"];
                     (x.Tensor.value).Ir.Tnode.label]))
        (( * ) ?label:None init_w x)
let initialized_expr =
  let expr_w =
    let open! TDSL.O in
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (NTDSL.kaiming normal1 ())) "expr_w") () in
  let f =
    let open! TDSL.O in
      fun x ->
        relu
          ?label:(Some (List.concat [["f"]; (x.Tensor.value).Ir.Tnode.label]))
          (( * ) ?label:None expr_w x) in
  f
let () = ignore (y0, y1, y2, a, b, y)
let () = ignore (z, z2, z3)
let () = ignore (concat_single, concat_pair)
let () = ignore (initialized_no_unit, initialized_expr)
let mlp_layer =
  let open! TDSL.O in
    fun ~label ~hid_dim () ->
      let w =
        (TDSL.param ?more_label:(Some label) ?value:None ?values:None
           ?param_init:None "w") () in
      let b =
        ((TDSL.param ?more_label:(Some label) ?value:None ?values:None
            ?param_init:None "b") ~output_dims:[hid_dim]) () in
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
let _capture_no_unit =
  let open! TDSL.O in
    fun x ->
      let b = Shape.get_variable_ref "b" in
      einsum1
        ?label:(Some
                  (List.concat
                     [["_capture_no_unit"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[b] "a, b => a" x
let _capture_with_unit =
  let open! TDSL.O in
    fun () ->
      let cw =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "cw") () in
      fun x ->
        let b = Shape.get_variable_ref "b" in
        let y = ( * ) ?label:(Some ["y"]) cw x in
        (/.) ?label:(Some ["_capture_with_unit"])
          (einsum1 ?label:None ~capture_dims:[b] "a, b => a" y) (dim b)
let _capture_siblings =
  let open! TDSL.O in
    let f x =
      let b = Shape.get_variable_ref "b" in
      einsum1 ?label:(Some ["f"]) ~capture_dims:[b] "a, b => a" x in
    let g x =
      let b = Shape.get_variable_ref "b" in
      einsum1 ?label:(Some ["g"]) ~capture_dims:[b] "a, b => b" x in
    fun x -> (+) ?label:(Some ["_capture_siblings"]) (f x) (g x)
let _capture_nested =
  let open! TDSL.O in
    fun x ->
      let b = Shape.get_variable_ref "b" in
      let g y =
        let b = Shape.get_variable_ref "b" in
        einsum1 ?label:(Some ["g"]) ~capture_dims:[b] "a, b => a" y in
      (+) ?label:(Some
                    (List.concat
                       [["_capture_nested"]; (x.Tensor.value).Ir.Tnode.label]))
        (einsum1 ?label:None ~capture_dims:[b] "a, b => a" x) (g x)
let _capture_nested_toplevel =
  let open! TDSL.O in
    let cn =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "cn") () in
    let b = Shape.get_variable_ref "b" in
    let g y =
      let b = Shape.get_variable_ref "b" in
      einsum1 ?label:(Some ["g"]) ~capture_dims:[b] "a, b => a" y in
    g (einsum1 ?label:None ~capture_dims:[b] "a, b, c => a, b" cn)
