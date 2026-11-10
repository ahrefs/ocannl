open Base
open Ocannl
open Nn_blocks.DSL_modules
let late_field =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "w") () in
    let b =
      ((TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
          "b")
         ~output_dims:([%ocaml.error
                         "ppx_ocannl %op: the inline declaration { b } is created once, when the function is defined, where the parameter hid_dim is not bound. Give the function a unit parameter with hid_dim before it -- fun ~hid_dim () ... -- so that the declaration is created at each () application"]))
        () in
    let __ocannl_body hid_dim x =
      (+) ?label:(Some
                    (List.concat
                       [["late_field"];
                       (hid_dim.Tensor.value).Ir.Tnode.label;
                       (x.Tensor.value).Ir.Tnode.label]))
        (( * ) ?label:None w x) b in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"late_field" ~arity:2
        (function
         | hid_dim::x::[] -> __ocannl_body hid_dim x
         | _ -> assert false) in
    fun hid_dim x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [hid_dim; x]
        (fun () -> __ocannl_body hid_dim x)
let late_field_labelled =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "w") () in
    let b =
      ((TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
          "b")
         ~output_dims:([%ocaml.error
                         "ppx_ocannl %op: the inline declaration { b } is created once, when the function is defined, where the parameter hid_dim is not bound. Give the function a unit parameter with hid_dim before it -- fun ~hid_dim () ... -- so that the declaration is created at each () application"]))
        () in
    fun ~hid_dim x ->
      (+) ?label:(Some
                    (List.concat
                       [["late_field_labelled"];
                       (x.Tensor.value).Ir.Tnode.label]))
        (( * ) ?label:None w x) b
let late_init =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some
                        ([%ocaml.error
                           "ppx_ocannl %op: the inline declaration { w } is created once, when the function is defined, where the parameter scale is not bound. Give the function a unit parameter with scale before it -- fun ~scale () ... -- so that the declaration is created at each () application"]))
         "w") () in
    fun ~scale x ->
      ( *
        ) ?label:(Some
                    (List.concat
                       [["late_init"]; (x.Tensor.value).Ir.Tnode.label]))
        w x
let field_after_unit =
  let open! TDSL.O in
    fun () ->
      let w =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "w") () in
      let b =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "b")
           ~output_dims:([%ocaml.error
                           "ppx_ocannl %op: the inline declaration { b } is created where the enclosing () is applied (without a unit parameter: when the function is defined), where the parameter hid_dim is not bound yet. Bind hid_dim before the unit parameter: fun ~hid_dim () ..."]))
          () in
      fun ~hid_dim x ->
        (+) ?label:(Some ["field_after_unit"]) (( * ) ?label:None w x) b
let field_in_nested =
  let open! TDSL.O in
    fun () ->
      let b =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "b")
           ~output_dims:([%ocaml.error
                           "ppx_ocannl %op: the inline declaration { b } is created where the enclosing () is applied (without a unit parameter: when the function is defined), where the parameter n is not bound yet. Bind n before the unit parameter: fun ~n () ..."]))
          () in
      let __ocannl_body x =
        let g n y = (+) ?label:(Some ["g"]) y b in
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
          (fun () -> g (TDSL.number (Float.of_int 3)) x) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"field_in_nested" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let field_before_unit =
  let open! TDSL.O in
    fun ~hid_dim () ->
      let w =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "w") () in
      let b =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "b") ~output_dims:[hid_dim]) () in
      let __ocannl_body x =
        (+) ?label:(Some ["field_before_unit"]) (( * ) ?label:None w x) b in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"field_before_unit" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let field_rebinds =
  let open! TDSL.O in
    let b =
      ((TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
          "b") ~output_dims:(List.map [3] ~f:(fun hid_dim -> hid_dim))) () in
    let __ocannl_body hid_dim x =
      (+) ?label:(Some
                    (List.concat
                       [["field_rebinds"];
                       (hid_dim.Tensor.value).Ir.Tnode.label;
                       (x.Tensor.value).Ir.Tnode.label]))
        x b in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"field_rebinds" ~arity:2
        (function
         | hid_dim::x::[] -> __ocannl_body hid_dim x
         | _ -> assert false) in
    fun hid_dim x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [hid_dim; x]
        (fun () -> __ocannl_body hid_dim x)
let field_let_bound =
  let open! TDSL.O in
    fun ~dims () ->
      let b =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "b")
           ~output_dims:([%ocaml.error
                           "ppx_ocannl %op: the inline declaration { b } is created where the enclosing () is applied (without a unit parameter: when the function is defined), outside the let that binds n. Bind n before the unit parameter -- fun ~n () ... -- or outside %op"]))
          () in
      let __ocannl_body x =
        let n =
          Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
            (fun () -> List.length dims) in
        (+) ?label:(Some ["field_let_bound"]) x b in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"field_let_bound" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let field_let_bound_no_unit =
  let open! TDSL.O in
    let b =
      ((TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
          "b")
         ~output_dims:([%ocaml.error
                         "ppx_ocannl %op: the inline declaration { b } is created where the enclosing () is applied (without a unit parameter: when the function is defined), outside the let that binds n. Bind n before the unit parameter -- fun ~n () ... -- or outside %op"]))
        () in
    let __ocannl_body x =
      let n = 3 in
      (+) ?label:(Some
                    (List.concat
                       [["field_let_bound_no_unit"];
                       (x.Tensor.value).Ir.Tnode.label]))
        x b in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"field_let_bound_no_unit" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let field_let_shadows =
  let open! TDSL.O in
    fun ~n () ->
      let b =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "b")
           ~output_dims:([%ocaml.error
                           "ppx_ocannl %op: the inline declaration { b } is created where the enclosing () is applied (without a unit parameter: when the function is defined), outside the let that binds n. Bind n before the unit parameter -- fun ~n () ... -- or outside %op"]))
          () in
      let __ocannl_body x =
        let n = n + 1 in (+) ?label:(Some ["field_let_shadows"]) x b in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"field_let_shadows" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let init_let_bound =
  let open! TDSL.O in
    fun () ->
      let w =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:(Some
                          ([%ocaml.error
                             "ppx_ocannl %op: the inline declaration { w } is created where the enclosing () is applied (without a unit parameter: when the function is defined), outside the let that binds scale. Bind scale before the unit parameter -- fun ~scale () ... -- or outside %op"]))
           "w") () in
      let __ocannl_body x =
        let scale = 7 in ( * ) ?label:(Some ["init_let_bound"]) w x in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"init_let_bound" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let field_let_tuple =
  let open! TDSL.O in
    fun () ->
      let b =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "b")
           ~output_dims:([%ocaml.error
                           "ppx_ocannl %op: the inline declaration { b } is created where the enclosing () is applied (without a unit parameter: when the function is defined), outside the let that binds n. Bind n before the unit parameter -- fun ~n () ... -- or outside %op"]))
          () in
      let __ocannl_body x =
        let (_, n) as _pair = (2, 3) in
        (+) ?label:(Some ["field_let_tuple"]) x b in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"field_let_tuple" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let field_let_rec =
  let open! TDSL.O in
    fun () ->
      let b =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "b")
           ~output_dims:([%ocaml.error
                           "ppx_ocannl %op: the inline declaration { b } is created where the enclosing () is applied (without a unit parameter: when the function is defined), outside the let that binds n. Bind n before the unit parameter -- fun ~n () ... -- or outside %op"]))
          () in
      let __ocannl_body x =
        let rec n = 3
        and f y = (+) ?label:(Some ["f"]) y b in
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> f x) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"field_let_rec" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let let_above_unit =
  let open! TDSL.O in
    let b =
      ((TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
          "b")
         ~output_dims:([%ocaml.error
                         "ppx_ocannl %op: the inline declaration { b } is created where the enclosing () is applied (without a unit parameter: when the function is defined), outside the let that binds n. Bind n before the unit parameter -- fun ~n () ... -- or outside %op"]))
        () in
    let n = 3 in fun () x -> (+) ?label:(Some ["let_above_unit"]) x b
let field_match_case =
  let open! TDSL.O in
    fun ~dims () ->
      let b =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "b")
           ~output_dims:([%ocaml.error
                           "ppx_ocannl %op: the inline declaration { b } is created where the enclosing () is applied (without a unit parameter: when the function is defined), outside the match case that binds n. Bind n before the unit parameter -- fun ~n () ... -- or outside %op"]))
          () in
      let __ocannl_body x =
        match dims with
        | n::_ -> (+) ?label:(Some ["field_match_case"]) x b
        | [] -> x in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"field_match_case" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let field_function_case =
  let open! TDSL.O in
    fun ~dims () ->
      let b =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "b")
           ~output_dims:([%ocaml.error
                           "ppx_ocannl %op: the inline declaration { b } is created where the enclosing () is applied (without a unit parameter: when the function is defined), outside the function case that binds n. Bind n before the unit parameter -- fun ~n () ... -- or outside %op"]))
          () in
      let __ocannl_body x =
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
          (fun () ->
             List.fold dims ~init:x
               ~f:(fun x -> function | n -> (+) ?label:None x b)) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"field_function_case" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let field_for_loop =
  let open! TDSL.O in
    fun () ->
      let b =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "b")
           ~output_dims:([%ocaml.error
                           "ppx_ocannl %op: the inline declaration { b } is created where the enclosing () is applied (without a unit parameter: when the function is defined), outside the for loop that binds n. Bind n before the unit parameter -- fun ~n () ... -- or outside %op"]))
          () in
      let __ocannl_body x =
        for n = 1 to 2 do
          Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
            (fun () -> ignore ((+) ?label:None x b))
        done;
        x in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"field_for_loop" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let let_unnamed =
  let open! TDSL.O in
    fun ~hid_dim () ->
      let b =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "b") ~output_dims:[hid_dim]) () in
      let __ocannl_body x =
        let n = 2 * hid_dim in
        let y = (+) ?label:(Some ["y"]) x b in
        (+) ?label:(Some ["let_unnamed"]) y (TDSL.O.(!..) n) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"let_unnamed" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let let_in_field =
  let open! TDSL.O in
    fun () ->
      let b =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "b") ~output_dims:(let n = 3 in [n])) () in
      let __ocannl_body x = (+) ?label:(Some ["let_in_field"]) x b in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure ~name:"let_in_field" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
let spec_stride =
  let open! TDSL.O in
    [%ocaml.error
      "ppx_ocannl %op: the unlabelled parameter stride is taken for a tensor, but the einsum spec \"stride*o<+k; k => o\" reads it as an integer. Label it (~stride), or bind it before a unit parameter: fun stride () ..."]
let spec_dilation =
  let open! TDSL.O in
    [%ocaml.error
      "ppx_ocannl %op: the unlabelled parameter dil is taken for a tensor, but the einsum spec \"o<+dil*k; k => o\" reads it as an integer. Label it (~dil), or bind it before a unit parameter: fun dil () ..."]
let embedded_float =
  let open! TDSL.O in
    [%ocaml.error
      "ppx_ocannl %op: the unlabelled parameter scale is taken for a tensor, but it is the float that !. embeds. Label it (~scale), or bind it before a unit parameter: fun scale () ..."]
let embedded_int =
  let open! TDSL.O in
    [%ocaml.error
      "ppx_ocannl %op: the unlabelled parameter count is taken for a tensor, but it is the integer that !.. embeds. Label it (~count), or bind it before a unit parameter: fun count () ..."]
let float_exponent =
  let open! TDSL.O in
    [%ocaml.error
      "ppx_ocannl %op: the unlabelled parameter n is taken for a tensor, but it is the float exponent of **.. Label it (~n), or bind it before a unit parameter: fun n () ..."]
let stride_after_unit =
  let open! TDSL.O in
    fun () ->
      let kernel =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "kernel") () in
      [%ocaml.error
        "ppx_ocannl %op: the unlabelled parameter stride is taken for a tensor, but the einsum spec \"stride*o<+k; k => o\" reads it as an integer. Label it (~stride), or bind it before a unit parameter: fun stride () ..."]
let exponent_rebound =
  let open! TDSL.O in
    let __ocannl_body n x =
      let pow n = TDSL.O.( **. ) ?label:(Some ["pow"]) x n in
      (+) ?label:(Some
                    (List.concat
                       [["exponent_rebound"];
                       (n.Tensor.value).Ir.Tnode.label;
                       (x.Tensor.value).Ir.Tnode.label]))
        (pow ?label:None 2.0) n in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"exponent_rebound" ~arity:2
        (function | n::x::[] -> __ocannl_body n x | _ -> assert false) in
    fun n x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [n; x]
        (fun () -> __ocannl_body n x)
let stride_unschemed =
  let open! TDSL.O in
    fun () ->
      let kernel =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "kernel") () in
      fun stride x ->
        einsum ?label:(Some ["stride_unschemed"])
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
             "o"]) x kernel
