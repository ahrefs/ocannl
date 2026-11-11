open Ocannl
open Nn_blocks.DSL_modules
let test_inline_defs =
  let open! TDSL.O in
    let w =
      [%ocaml.error
        "ppx_ocannl: name clash for inline definition or variable capture 'w' - the name is already defined"] in
    let w =
      [%ocaml.error
        "ppx_ocannl: name clash for inline definition or variable capture 'w' - the name is already defined"] in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "w") () in
    let __ocannl_body x =
      let q = ( * ) ?label:(Some ["q"]) w x in
      let k = ( * ) ?label:(Some ["k"]) w x in
      let v = ( * ) ?label:(Some ["v"]) w x in
      (+) ?label:(Some
                    (List.concat
                       [["test_inline_defs"];
                       (x.Tensor.value).Ir.Tnode.label]))
        (( * ) ?label:None q k) v in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"test_inline_defs" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let test_variable_capture =
  let open! TDSL.O in
    let __ocannl_body x =
      let a = Shape.get_variable_ref "a" in
      let b = Shape.get_variable_ref "b" in
      Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
        (fun () -> Shape.set_equal a b);
      einsum1
        ?label:(Some
                  (List.concat
                     [["test_variable_capture"];
                     (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[[%ocaml.error
                         "ppx_ocannl: repeated variable capture 'a'"];
                      a;
                      b] "ab=>ba" x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"test_variable_capture" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let test_mixed =
  let open! TDSL.O in
    let b =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "b") () in
    let __ocannl_body x =
      let a = Shape.get_variable_ref "a" in
      let b =
        [%ocaml.error
          "ppx_ocannl: name clash for inline definition or variable capture 'b' - the name is already defined"] in
      Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
        (fun () -> Shape.set_equal a b);
      einsum
        ?label:(Some
                  (List.concat
                     [["test_mixed"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[a; b] "ab;bc=>ac" x b in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"test_mixed" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let test_capture_shadows_outer_param =
  let open! TDSL.O in
    fun () ->
      let c =
        [%ocaml.error
          "ppx_ocannl: name clash for inline definition or variable capture 'c' - the name is already defined"] in
      let p = c in
      let __ocannl_body x =
        let c = Shape.get_variable_ref "c" in
        (+) ?label:(Some ["test_capture_shadows_outer_param"]) p
          (einsum1 ?label:None ~capture_dims:[c] "ac=>a" x) in
      let __ocannl_scheme =
        Tensor.Scheme.For_ppx.closure
          ~name:"test_capture_shadows_outer_param" ~arity:1
          (function | x::[] -> __ocannl_body x | _ -> assert false) in
      fun x ->
        Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
          (fun () -> __ocannl_body x)
