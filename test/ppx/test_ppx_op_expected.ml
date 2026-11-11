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
    let __ocannl_body x =
      (+) ?label:(Some
                    (List.concat [["y1"]; (x.Tensor.value).Ir.Tnode.label]))
        (( * ) ?label:None hey2 (TDSL.number (Float.of_int 2))) x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"y1" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let y2 =
  let open! TDSL.O in
    let hey3 =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "hey3") () in
    let __ocannl_body x1 x2 =
      (+) ?label:(Some
                    (List.concat
                       [["y2"];
                       (x1.Tensor.value).Ir.Tnode.label;
                       (x2.Tensor.value).Ir.Tnode.label]))
        (( *. ) ?label:None x1 hey3) x2 in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"y2" ~arity:2
        (function | x1::x2::[] -> __ocannl_body x1 x2 | _ -> assert false) in
    fun x1 x2 ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x1; x2]
        (fun () -> __ocannl_body x1 x2)
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
         "bc";
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
    let i = Shape.get_variable_ref "i" in
    let j = Shape.get_variable_ref "j" in
    concat ?label:(Some ["concat_pair"]) ~capture_dims:[i; j] "i; j => i ^ j"
      [|hey12;hey13|]
let initialized_no_unit =
  let open! TDSL.O in
    let init_w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (NTDSL.kaiming ~scale_sq:2.0 normal1 ())) "init_w")
        () in
    let __ocannl_body x =
      relu
        ?label:(Some
                  (List.concat
                     [["initialized_no_unit"];
                     (x.Tensor.value).Ir.Tnode.label]))
        (( * ) ?label:None init_w x) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"initialized_no_unit" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let initialized_expr =
  let expr_w =
    let open! TDSL.O in
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (NTDSL.kaiming normal1 ())) "expr_w") () in
  let f =
    let open! TDSL.O in
      let __ocannl_body x =
        relu
          ?label:(Some (List.concat [["f"]; (x.Tensor.value).Ir.Tnode.label]))
          (( * ) ?label:None expr_w x) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.deferred ~name:"f" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x) in
  f
let use_kaiming = true
let initialized_compound =
  let open! TDSL.O in
    let cond_w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some
                        (if use_kaiming
                         then NTDSL.kaiming normal1 ()
                         else NTDSL.uniform ())) "cond_w") () in
    let let_w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (let base = normal1 in NTDSL.kaiming base ()))
         "let_w") () in
    let __ocannl_body x =
      relu
        ?label:(Some
                  (List.concat
                     [["initialized_compound"];
                     (x.Tensor.value).Ir.Tnode.label]))
        (( * ) ?label:None cond_w (( * ) ?label:None let_w x)) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"initialized_compound" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let () = ignore (y0, y1, y2, a, b, y)
let () = ignore initialized_compound
let () = ignore (z, z2, z3)
let () = ignore (concat_single, concat_pair)
let () = ignore (initialized_no_unit, initialized_expr)
let mlp_layer =
  let open! TDSL.O in
    fun ~label ~hid_dim () ->
      let __ocannl_result_label = ["mlp_layer"] @ (label : string list) in
      let w =
        (TDSL.param ?more_label:(Some label) ?value:None ?values:None
           ?param_init:None "w") () in
      let b =
        ((TDSL.param ?more_label:(Some label) ?value:None ?values:None
            ?param_init:None "b") ~output_dims:[hid_dim]) () in
      let __ocannl_body ~x =
        relu ?label:(Some __ocannl_result_label)
          ((+) ?label:None (( * ) ?label:None w x) b) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"mlp_layer" ~arity:1
          (function | x::[] -> __ocannl_body ~x | _ -> assert false) in
      fun ~x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body ~x)
let _use_layer =
  let open! TDSL.O in
    let l1 =
      Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
        (fun () -> mlp_layer ~label:["L"] ~hid_dim:3 ()) in
    let l2 =
      Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
        (fun () -> mlp_layer ~label:["L2"] ~hid_dim:3 ()) in
    fun x ->
      Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
        (fun () ->
           l1
             ~x:(Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                   (fun () -> l2 ~x)))
let _config_layer =
  let open! TDSL.O in
    fun ~label () ->
      let __ocannl_result_label = ["_config_layer"] @ (label : string list) in
      let l =
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
          (fun () -> mlp_layer ~label:(label @ ["L"]) ~hid_dim:3 ()) in
      let __ocannl_body x =
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> l ~x) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"_config_layer" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let _three_layer_perceptron =
  let open! TDSL.O in
    fun ~label ~dim1 ~dim2 ~dim3 () ->
      let __ocannl_result_label =
        ["_three_layer_perceptron"] @ (label : string list) in
      let l1 =
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
          (fun () -> mlp_layer ~label:(label @ ["L1"]) ~hid_dim:dim1 ()) in
      let l2 =
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
          (fun () -> mlp_layer ~label:(label @ ["L2"]) ~hid_dim:dim2 ()) in
      let l3 =
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
          (fun () -> mlp_layer ~label:(label @ ["L3"]) ~hid_dim:dim3 ()) in
      let __ocannl_body x =
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
          (fun () ->
             l3
               ~x:(Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                     (fun () ->
                        l2
                          ~x:(Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                                (fun () -> l1 ~x))))) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"_three_layer_perceptron"
          ~arity:1 (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let _capture_no_unit =
  let open! TDSL.O in
    let __ocannl_body x =
      let b = Shape.get_variable_ref "b" in
      einsum1
        ?label:(Some
                  (List.concat
                     [["_capture_no_unit"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[b] "a, b => a" x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_capture_no_unit" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let _capture_with_unit =
  let open! TDSL.O in
    fun () ->
      let cw =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "cw") () in
      let __ocannl_body x =
        let b = Shape.get_variable_ref "b" in
        let y = ( * ) ?label:(Some ["y"]) cw x in
        (/.) ?label:(Some ["_capture_with_unit"])
          (einsum1 ?label:None ~capture_dims:[b] "a, b => a" y)
          (Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> dim b)) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"_capture_with_unit" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let _capture_siblings =
  let open! TDSL.O in
    let f x =
      let b = Shape.get_variable_ref "b" in
      einsum1 ?label:(Some ["f"]) ~capture_dims:[b] "a, b => a" x in
    let g x =
      let b = Shape.get_variable_ref "b" in
      einsum1 ?label:(Some ["g"]) ~capture_dims:[b] "a, b => b" x in
    fun x ->
      (+) ?label:(Some ["_capture_siblings"])
        (Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> f x))
        (Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> g x))
let _capture_nested =
  let open! TDSL.O in
    let __ocannl_body x =
      let b = Shape.get_variable_ref "b" in
      let g y =
        let b = Shape.get_variable_ref "b" in
        einsum1 ?label:(Some ["g"]) ~capture_dims:[b] "a, b => a" y in
      (+) ?label:(Some
                    (List.concat
                       [["_capture_nested"]; (x.Tensor.value).Ir.Tnode.label]))
        (einsum1 ?label:None ~capture_dims:[b] "a, b => a" x)
        (Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> g x)) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_capture_nested" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let _capture_nested_toplevel =
  let open! TDSL.O in
    let cn =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "cn") () in
    let b = Shape.get_variable_ref "b" in
    let g y =
      let b = Shape.get_variable_ref "b" in
      einsum1 ?label:(Some ["g"]) ~capture_dims:[b] "a, b => a" y in
    Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
      (fun () ->
         g (einsum1 ?label:None ~capture_dims:[b] "a, b, c => a, b" cn))
let _scheme_labelled =
  let open! TDSL.O in
    fun () ->
      let __ocannl_body ~gamma ~beta x =
        (+) ?label:(Some ["_scheme_labelled"]) (( *. ) ?label:None gamma x)
          beta in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"_scheme_labelled" ~arity:3
          (function
           | gamma::beta::x::[] -> __ocannl_body ~gamma ~beta x
           | _ -> assert false) in
      fun ~gamma ~beta x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [gamma; beta; x]
          (fun () -> __ocannl_body ~gamma ~beta x)
let _scheme_tail =
  let open! TDSL.O in
    fun () ->
      let l =
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
          (fun () -> mlp_layer ~label:["tail"] ~hid_dim:2 ()) in
      let __ocannl_body x =
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> l ~x) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"_scheme_tail" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let _scheme_step =
  let open! TDSL.O in
    fun () ->
      fun ~train_step x ->
        match train_step with
        | Some () ->
            (+) ?label:(Some ["_scheme_step"]) x
              (TDSL.number ?label:None 1.0)
        | None -> x
let _scheme_mask =
  let open! TDSL.O in
    fun () ->
      fun ?mask x ->
        match mask with
        | None -> x
        | Some m -> ( *. ) ?label:(Some ["_scheme_mask"]) x m
let _scheme_variants =
  let open! TDSL.O in
    fun () ->
      let __ocannl_body ~train_step:(train_step : Tensor.t option)
        ?mask:(mask : Tensor.t option) x =
        let x =
          match train_step with
          | Some s -> (+) ?label:(Some ["x"]) x s
          | None -> x in
        match mask with
        | Some m -> ( *. ) ?label:(Some ["_scheme_variants"]) x m
        | None -> x in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure_variants ~name:"_scheme_variants"
          ~optional:[Some "train_step"; Some "mask"; None]
          (function
           | train_step::mask::(Some x)::[] ->
               __ocannl_body ~train_step ?mask x
           | _ -> assert false) in
      fun ~train_step ?mask x ->
        Tensor.Scheme.For_ppx.apply_variants __ocannl_scheme
          [train_step; mask; Some x]
          (fun () -> __ocannl_body ~train_step ?mask x)
let _scheme_forwarded =
  let open! TDSL.O in
    fun () ->
      let __ocannl_body ~gamma:(gamma : Tensor.t) x =
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
          (fun () -> _scheme_labelled () ~gamma ~beta:gamma x) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"_scheme_forwarded" ~arity:2
          (function
           | gamma::x::[] -> __ocannl_body ~gamma x
           | _ -> assert false) in
      fun ~gamma x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [gamma; x]
          (fun () -> __ocannl_body ~gamma x)
let _scheme_late =
  let open! TDSL.O in
    let __ocannl_body ?mask:(mask : Tensor.t option) x =
      match mask with
      | Some m ->
          ( *.
            ) ?label:(Some
                        (List.concat
                           [["_scheme_late"];
                           (x.Tensor.value).Ir.Tnode.label]))
            x m
      | None -> x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred_variants ~name:"_scheme_late"
        ~optional:[Some "mask"; None]
        (function
         | mask::(Some x)::[] -> __ocannl_body ?mask x
         | _ -> assert false) in
    fun ?mask x ->
      Tensor.Scheme.For_ppx.apply_variants __ocannl_scheme [mask; Some x]
        (fun () -> __ocannl_body ?mask x)
let _late_config =
  let open! TDSL.O in
    fun ~scale:(scale : float) x ->
      ( *.
        ) ?label:(Some
                    (List.concat
                       [["_late_config"]; (x.Tensor.value).Ir.Tnode.label]))
        x (TDSL.O.(!.) scale)
let _scheme_defaulted =
  let open! TDSL.O in
    fun () ->
      fun ?scale:((scale : Tensor.t option)= None) x ->
        match scale with
        | Some s -> ( *. ) ?label:(Some ["_scheme_defaulted"]) x s
        | None -> x
let _scheme_opted_out =
  let open! TDSL.O in
    fun () ->
      let ow =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "ow") () in
      fun x -> ( * ) ?label:(Some ["_scheme_opted_out"]) ow x
let rec _scheme_rec =
  let open! TDSL.O in
    fun () ->
      fun x ->
        if false
        then
          Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
            (fun () -> _scheme_rec () x)
        else
          (+) ?label:(Some ["_scheme_rec"]) x (TDSL.number ?label:None 1.0)
let _no_unit_optional =
  let open! TDSL.O in
    fun ?(stride= 2) ?scale x ->
      let y =
        match scale with
        | None -> x
        | Some s -> ( *. ) ?label:(Some ["y"]) x s in
      Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
        (fun () -> ignore (stride : int));
      (+) ?label:(Some
                    (List.concat
                       [["_no_unit_optional"];
                       (x.Tensor.value).Ir.Tnode.label]))
        y (TDSL.number ?label:None 1.0)
let _late_gamma =
  let open! TDSL.O in
    let __ocannl_body ~gamma x =
      ( *.
        ) ?label:(Some
                    (List.concat
                       [["_late_gamma"];
                       (gamma.Tensor.value).Ir.Tnode.label;
                       (x.Tensor.value).Ir.Tnode.label]))
        gamma x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_late_gamma" ~arity:2
        (function
         | gamma::x::[] -> __ocannl_body ~gamma x
         | _ -> assert false) in
    fun ~gamma x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [gamma; x]
        (fun () -> __ocannl_body ~gamma x)
let _late_beta =
  let open! TDSL.O in
    let __ocannl_body ~beta:(beta : Tensor.t) x =
      Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
        (fun () -> _late_gamma ~gamma:beta x) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_late_beta" ~arity:2
        (function | beta::x::[] -> __ocannl_body ~beta x | _ -> assert false) in
    fun ~beta x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [beta; x]
        (fun () -> __ocannl_body ~beta x)
let _late_scale =
  let open! TDSL.O in
    fun ~scale x ->
      ( *.
        ) ?label:(Some
                    (List.concat
                       [["_late_scale"]; (x.Tensor.value).Ir.Tnode.label]))
        x (TDSL.O.(!.) scale)
let _late_label =
  let open! TDSL.O in
    fun ~label x ->
      Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
        (fun () -> mlp_layer ~label:("late" :: label) ~hid_dim:2 () ~x)
let _late_label_result =
  let open! TDSL.O in
    let llw =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "llw") () in
    fun ~label x ->
      (+) ?label:(Some
                    (List.concat
                       [["_late_label_result"];
                       (label : string list);
                       (x.Tensor.value).Ir.Tnode.label]))
        (( * ) ?label:None llw x) (TDSL.number ?label:None 1.0)
let _late_opt_label =
  let open! TDSL.O in
    fun ?label x ->
      (+) ?label:(Some
                    (List.concat
                       [["_late_opt_label"];
                       (match label with
                        | Some label -> (label : string list)
                        | None -> []);
                       (x.Tensor.value).Ir.Tnode.label]))
        x (TDSL.number ?label:None 1.0)
let _late_label_default =
  let open! TDSL.O in
    fun ?(label= ["dflt"]) ~g:(g : Tensor.t) x ->
      (+) ?label:(Some
                    (List.concat
                       [["_late_label_default"];
                       (label : string list);
                       ((g : Tensor.t).Tensor.value).Ir.Tnode.label;
                       (x.Tensor.value).Ir.Tnode.label]))
        x g
let _late_label_annot =
  let open! TDSL.O in
    fun ~label:(label : string list) x ->
      relu
        ?label:(Some
                  (List.concat
                     [["_late_label_annot"];
                     (label : string list);
                     (x.Tensor.value).Ir.Tnode.label])) x
let _late_label_tensor =
  let open! TDSL.O in
    let __ocannl_body ~label x =
      (+) ?label:(Some
                    (List.concat
                       [["_late_label_tensor"];
                       (label.Tensor.value).Ir.Tnode.label;
                       (x.Tensor.value).Ir.Tnode.label]))
        x label in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_late_label_tensor" ~arity:2
        (function
         | label::x::[] -> __ocannl_body ~label x
         | _ -> assert false) in
    fun ~label x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [label; x]
        (fun () -> __ocannl_body ~label x)
let _late_label_default_tensor =
  let open! TDSL.O in
    fun ?(label= !. 1.0) x ->
      (+) ?label:(Some
                    (List.concat
                       [["_late_label_default_tensor"];
                       (x.Tensor.value).Ir.Tnode.label]))
        x label
let _late_stride =
  let open! TDSL.O in
    let ls_kernel =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "ls_kernel") () in
    fun ~stride x ->
      einsum
        ?label:(Some
                  (List.concat
                     [["_late_stride"]; (x.Tensor.value).Ir.Tnode.label]))
        (String.concat ~sep:""
           [Int.to_string stride;
           "*";
           "o";
           "<";
           "+";
           "k";
           "; ";
           "k";
           " => ";
           "o"]) x ls_kernel
let _unit_stride =
  let open! TDSL.O in
    fun stride () ->
      let us_kernel =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "us_kernel") () in
      let __ocannl_body x =
        einsum ?label:(Some ["_unit_stride"])
          (String.concat ~sep:""
             [Int.to_string stride;
             "*";
             "o";
             "<";
             "+";
             "k";
             "; ";
             "k";
             " => ";
             "o"]) x us_kernel in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"_unit_stride" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let _late_exponent =
  let open! TDSL.O in
    fun ~n x ->
      TDSL.O.( **. )
        ?label:(Some
                  (List.concat
                     [["_late_exponent"]; (x.Tensor.value).Ir.Tnode.label]))
        x n
let _opt_label =
  let open! TDSL.O in
    fun ?label ~hid_dim () ->
      let __ocannl_result_label =
        ["_opt_label"] @
          (match label with
           | Some label -> (label : string list)
           | None -> []) in
      let olw =
        (TDSL.param ?more_label:label ?value:None ?values:None
           ?param_init:None "olw") () in
      let olb =
        ((TDSL.param ?more_label:label ?value:None ?values:None
            ?param_init:None "olb") ~output_dims:[hid_dim]) () in
      let __ocannl_body x =
        (+) ?label:(Some __ocannl_result_label) (( * ) ?label:None olw x) olb in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"_opt_label" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let _opt_label_default =
  let open! TDSL.O in
    fun ?(label= ["dflt"]) () ->
      let __ocannl_result_label =
        ["_opt_label_default"] @ (label : string list) in
      let odw =
        (TDSL.param ?more_label:(Some label) ?value:None ?values:None
           ?param_init:None "odw") () in
      let __ocannl_body x = ( * ) ?label:(Some __ocannl_result_label) odw x in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"_opt_label_default" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let _uncurried_where =
  let open! TDSL.O in
    let __ocannl_body c x y =
      (where
         ?label:(Some
                   (List.concat
                      [["_uncurried_where"];
                      (c.Tensor.value).Ir.Tnode.label;
                      (x.Tensor.value).Ir.Tnode.label;
                      (y.Tensor.value).Ir.Tnode.label]))) c x y in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_uncurried_where" ~arity:3
        (function | c::x::y::[] -> __ocannl_body c x y | _ -> assert false) in
    fun c x y ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [c; x; y]
        (fun () -> __ocannl_body c x y)
let _curried_where =
  let open! TDSL.O in
    let __ocannl_body c x y =
      where
        ?label:(Some
                  (List.concat
                     [["_curried_where"];
                     (c.Tensor.value).Ir.Tnode.label;
                     (x.Tensor.value).Ir.Tnode.label;
                     (y.Tensor.value).Ir.Tnode.label])) c x y in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_curried_where" ~arity:3
        (function | c::x::y::[] -> __ocannl_body c x y | _ -> assert false) in
    fun c x y ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [c; x; y]
        (fun () -> __ocannl_body c x y)
let _uncurried_fma =
  let open! TDSL.O in
    let uf_w =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "uf_w") () in
    let uf_b =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "uf_b") () in
    let __ocannl_body x =
      (fma
         ?label:(Some
                   (List.concat
                      [["_uncurried_fma"]; (x.Tensor.value).Ir.Tnode.label])))
        x uf_w uf_b in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_uncurried_fma" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let _uncurried_plus =
  let open! TDSL.O in
    let up_w =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "up_w") () in
    let __ocannl_body x y =
      ((+) ?label:(Some
                     (List.concat
                        [["_uncurried_plus"];
                        (x.Tensor.value).Ir.Tnode.label;
                        (y.Tensor.value).Ir.Tnode.label]))) x
        (( *. ) ?label:None y up_w) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_uncurried_plus" ~arity:2
        (function | x::y::[] -> __ocannl_body x y | _ -> assert false) in
    fun x y ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x; y]
        (fun () -> __ocannl_body x y)
let _uncurried_nested =
  let open! TDSL.O in
    let __ocannl_body c x y =
      (where
         ?label:(Some
                   (List.concat
                      [["_uncurried_nested"];
                      (c.Tensor.value).Ir.Tnode.label;
                      (x.Tensor.value).Ir.Tnode.label;
                      (y.Tensor.value).Ir.Tnode.label])))
        (((<) ?label:None) x y) (relu ?label:None x)
        ((fma ?label:None) c x y) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_uncurried_nested" ~arity:3
        (function | c::x::y::[] -> __ocannl_body c x y | _ -> assert false) in
    fun c x y ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [c; x; y]
        (fun () -> __ocannl_body c x y)
let _direct_relu =
  let open! TDSL.O in
    let __ocannl_body x =
      relu
        ?label:(Some
                  (List.concat
                     [["_direct_relu"]; (x.Tensor.value).Ir.Tnode.label])) x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_direct_relu" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let _pipe_relu =
  let open! TDSL.O in
    let __ocannl_body x = let __x = x in (relu ?label:None) __x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_pipe_relu" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let _direct_add =
  let open! TDSL.O in
    let __ocannl_body x y =
      (+) ?label:(Some
                    (List.concat
                       [["_direct_add"];
                       (x.Tensor.value).Ir.Tnode.label;
                       (y.Tensor.value).Ir.Tnode.label]))
        y x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_direct_add" ~arity:2
        (function | x::y::[] -> __ocannl_body x y | _ -> assert false) in
    fun x y ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x; y]
        (fun () -> __ocannl_body x y)
let _pipe_add =
  let open! TDSL.O in
    let __ocannl_body x y = let __x = x in ((+) ?label:None y) __x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_pipe_add" ~arity:2
        (function | x::y::[] -> __ocannl_body x y | _ -> assert false) in
    fun x y ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x; y]
        (fun () -> __ocannl_body x y)
let _direct_einsum1 =
  let open! TDSL.O in
    let __ocannl_body x = einsum1 "ab=>a" x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_direct_einsum1" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let _pipe_einsum1 =
  let open! TDSL.O in
    let __ocannl_body x = let __x = x in (einsum1 "ab=>a") __x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_pipe_einsum1" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let _direct_layer =
  let open! TDSL.O in
    let __ocannl_body x =
      Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
        (fun () -> initialized_no_unit x) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_direct_layer" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let _pipe_layer =
  let open! TDSL.O in
    let __ocannl_body x =
      let __x = x in
      Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
        (fun () -> initialized_no_unit __x) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_pipe_layer" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let _direct_partial =
  let open! TDSL.O in
    let __ocannl_body x y =
      Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
        (fun () -> _direct_add y x) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_direct_partial" ~arity:2
        (function | x::y::[] -> __ocannl_body x y | _ -> assert false) in
    fun x y ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x; y]
        (fun () -> __ocannl_body x y)
let _pipe_partial =
  let open! TDSL.O in
    let __ocannl_body x y =
      let __x = x in
      Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
        (fun () ->
           (Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
              (fun () -> _direct_add y)) __x) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_pipe_partial" ~arity:2
        (function | x::y::[] -> __ocannl_body x y | _ -> assert false) in
    fun x y ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x; y]
        (fun () -> __ocannl_body x y)
let _constrained_call =
  let open! TDSL.O in
    let __ocannl_body x =
      (Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
         (fun () -> initialized_no_unit x) : Tensor.t) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_constrained_call" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let _constrained_op =
  let open! TDSL.O in
    let __ocannl_body x =
      ((relu
          ?label:(Some
                    (List.concat
                       [["_constrained_op"]; (x.Tensor.value).Ir.Tnode.label]))
          x : Tensor.t) : Tensor.t) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_constrained_op" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let _constrained_inline =
  let open! TDSL.O in
    fun () ->
      let constrained_w =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "constrained_w") () in
      let __ocannl_body x =
        (((+) ?label:(Some ["_constrained_inline"])) x constrained_w : 
        Tensor.t) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"_constrained_inline" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let _constrained_block =
  let open! TDSL.O in
    let __ocannl_body x =
      stack
        ?label:(Some
                  (List.concat
                     [["_constrained_block"];
                     (x.Tensor.value).Ir.Tnode.label])) `Output
        [|(stack ?label:None `Input [|x;x|] : Tensor.t)|] in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_constrained_block" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let rec _constrained_rec =
  let open! TDSL.O in
    (((fun x ->
         if false
         then
           Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
             (fun () -> _constrained_rec x)
         else
           relu
             ?label:(Some
                       (List.concat
                          [["_constrained_rec"];
                          (x.Tensor.value).Ir.Tnode.label])) x : Tensor.t ->
                                                                   Tensor.t))
    [@ocannl.no_scheme ])
let _constrained_optout =
  let open! TDSL.O in
    (((fun () -> fun x -> relu ?label:(Some ["_constrained_optout"]) x : 
    unit -> Tensor.t -> Tensor.t))[@ocannl.no_scheme ])
let _constrained_lazy =
  let open! TDSL.O in
    let __ocannl_body x =
      lazy
        (relu
           ?label:(Some
                     (List.concat
                        [["_constrained_lazy"];
                        (x.Tensor.value).Ir.Tnode.label])) x : Tensor.t) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_constrained_lazy" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let ( let* ) x f = f x
let _constrained_capture =
  let open! TDSL.O in
    let __ocannl_body x =
      let* y = x
       in
      let b = Shape.get_variable_ref "b" in
      (einsum1
         ?label:(Some
                   (List.concat
                      [["_constrained_capture"];
                      (x.Tensor.value).Ir.Tnode.label])) ~capture_dims:
         [b] "a, b => a" y : Tensor.t) in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"_constrained_capture" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
