open Ocannl
open Nn_blocks.DSL_modules
let test_inline_defs =
  let w =
    [%ocaml.error
      "ppx_ocannl: name clash for inline definition or variable capture 'w' - the name is already defined"] in
  let w =
    [%ocaml.error
      "ppx_ocannl: name clash for inline definition or variable capture 'w' - the name is already defined"] in
  let w =
    (TDSL.param ?more_label:None ?value:None ?values:None ?param_init:None
       "w") () in
  let open! TDSL.O in
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
  let b = Shape.get_variable_ref "b" in
  let a = Shape.get_variable_ref "a" in
  let open! TDSL.O in
    fun x ->
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
  let b =
    [%ocaml.error
      "ppx_ocannl: name clash for inline definition or variable capture 'b' - the name is already defined"] in
  let b = Shape.get_variable_ref "b" in
  let a = Shape.get_variable_ref "a" in
  let open! TDSL.O in
    fun x ->
      Shape.set_equal a b;
      einsum
        ?label:(Some
                  (List.concat
                     [["test_mixed"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[b; a] "a, b; b, c => a, c" x b
