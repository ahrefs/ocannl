open Ocannl
open Operation.DSL_modules
let test_inline_defs =
  let w =
    [%ocaml.error
      "ppx_ocannl: name clash for inline definition or variable capture 'w' - the name is already defined"] in
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
  let a = Shape.get_variable_ref "a"
  and b = Shape.get_variable_ref "b" in
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
                      a] "ab=>ba" x
let test_mixed =
  let a = Shape.get_variable_ref "a"
  and b =
    [%ocaml.error
      "ppx_ocannl: name clash for inline definition or variable capture 'b' - the name is already defined"] in
  let open! TDSL.O in
    fun x ->
      Shape.set_equal a b;
      einsum
        ?label:(Some
                  (List.concat
                     [["test_mixed"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[b; a] "ab;bc=>ac" x b
