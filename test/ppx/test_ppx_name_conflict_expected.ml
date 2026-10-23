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
    fun x ->
      let q = ( * ) ?label:(Some ["q"]) w x in
      let k = ( * ) ?label:(Some ["k"]) w x in
      let v = ( * ) ?label:(Some ["v"]) w x in
      (+) ?label:(Some
                    (List.concat
                       [["test_inline_defs"];
                       (x.Tensor.value).Ir.Tnode.label]))
        (( * ) ?label:None q k) v
let test_variable_capture =
  let open! TDSL.O in
    fun x ->
      let b = Shape.get_variable_ref "b" in
      let a = Shape.get_variable_ref "a" in
      Shape.set_equal a b;
      einsum1
        ?label:(Some
                  (List.concat
                     [["test_variable_capture"];
                     (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[[%ocaml.error
                         "ppx_ocannl %op: repeated variable capture 'a'"];
                      b;
                      a] "a, b => b, a" x
let test_mixed =
  let open! TDSL.O in
    let b =
      (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
         "b") () in
    fun x ->
      let b =
        [%ocaml.error
          "ppx_ocannl: name clash for inline definition or variable capture 'b' - the name is already defined"] in
      let a = Shape.get_variable_ref "a" in
      Shape.set_equal a b;
      einsum
        ?label:(Some
                  (List.concat
                     [["test_mixed"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[b; a] "a, b; b, c => a, c" x b
let test_capture_shadows_outer_param =
  let open! TDSL.O in
    fun () ->
      let c =
        [%ocaml.error
          "ppx_ocannl: name clash for inline definition or variable capture 'c' - the name is already defined"] in
      let p = c in
      fun x ->
        let c = Shape.get_variable_ref "c" in
        (+) ?label:(Some ["test_capture_shadows_outer_param"]) p
          (einsum1 ?label:None ~capture_dims:[c] "a, b => a" x)
