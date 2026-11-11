open Ocannl.Nn_blocks.DSL_modules
let unary =
  let open! TDSL.O in
    let __ocannl_body x =
      let ab =
        [%ocaml.error
          "ppx_ocannl: capture label \"ab\" is absent from einsum spec \"ab => a\"; a multi-letter capture label needs multichar mode: write `ab,` and separate the other axes with commas"] in
      einsum1
        ?label:(Some
                  (List.concat [["unary"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[ab] "a, b => a" x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"unary" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let digit =
  let open! TDSL.O in
    let __ocannl_body x =
      let f0 =
        [%ocaml.error
          "ppx_ocannl: capture label \"f0\" is absent from einsum spec \"f0 => f0\"; a multi-letter capture label needs multichar mode: write `f0,` and separate the other axes with commas"] in
      einsum1
        ?label:(Some
                  (List.concat [["digit"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[f0] "f, 0 => f, 0" x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"digit" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let multichar =
  let open! TDSL.O in
    let __ocannl_body x =
      let ab =
        [%ocaml.error
          "ppx_ocannl: capture label \"ab\" is absent from einsum spec \"a, b => a\""] in
      einsum1
        ?label:(Some
                  (List.concat
                     [["multichar"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[ab] "a, b => a" x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"multichar" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let missing =
  let open! TDSL.O in
    let __ocannl_body x =
      let c =
        [%ocaml.error
          "ppx_ocannl: capture label \"c\" is absent from einsum spec \"ab => a\""] in
      einsum1
        ?label:(Some
                  (List.concat [["missing"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[c] "a, b => a" x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"missing" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let binary =
  let open! TDSL.O in
    let __ocannl_body x y =
      let missing =
        [%ocaml.error
          "ppx_ocannl: capture label \"missing\" is absent from einsum spec \"ab; bc => ac\"; a multi-letter capture label needs multichar mode: write `missing,` and separate the other axes with commas"] in
      einsum
        ?label:(Some
                  (List.concat
                     [["binary"];
                     (x.Tensor.value).Ir.Tnode.label;
                     (y.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[missing] "a, b; b, c => a, c" x y in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"binary" ~arity:2
        (function | x::y::[] -> __ocannl_body x y | _ -> assert false) in
    fun x y ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x; y]
        (fun () -> __ocannl_body x y)
let mixfix =
  let open! TDSL.O in
    let __ocannl_body x y =
      let missing =
        [%ocaml.error
          "ppx_ocannl: capture label \"missing\" is absent from einsum spec \"ab; bc => ac\"; a multi-letter capture label needs multichar mode: write `missing,` and separate the other axes with commas"] in
      einsum
        ?label:(Some
                  (List.concat
                     [["mixfix"];
                     (x.Tensor.value).Ir.Tnode.label;
                     (y.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[missing] "a, b; b, c => a, c" x y in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"mixfix" ~arity:2
        (function | x::y::[] -> __ocannl_body x y | _ -> assert false) in
    fun x y ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x; y]
        (fun () -> __ocannl_body x y)
let concat =
  let open! TDSL.O in
    let __ocannl_body x y =
      let missing =
        [%ocaml.error
          "ppx_ocannl: capture label \"missing\" is absent from einsum spec \"a; b => a^b\""] in
      concat
        ?label:(Some
                  (List.concat
                     [["concat"];
                     (x.Tensor.value).Ir.Tnode.label;
                     (y.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[missing] "a; b => a ^ b" [|x;y|] in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"concat" ~arity:2
        (function | x::y::[] -> __ocannl_body x y | _ -> assert false) in
    fun x y ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x; y]
        (fun () -> __ocannl_body x y)
let cd_unary =
  let open! NTDSL.O in
    fun x ->
      let ab =
        [%ocaml.error
          "ppx_ocannl: capture label \"ab\" is absent from einsum spec \"ab => a\"; a multi-letter capture label needs multichar mode: write `ab,` and separate the other axes with commas"] in
      Tensor.consume_forward_code (einsum1 ~capture_dims:[ab] "a, b => a" x)
let cd_binary =
  let open! NTDSL.O in
    fun x y ->
      let missing =
        [%ocaml.error
          "ppx_ocannl: capture label \"missing\" is absent from einsum spec \"ab; bc => ac\"; a multi-letter capture label needs multichar mode: write `missing,` and separate the other axes with commas"] in
      Tensor.consume_forward_code
        (einsum ~capture_dims:[missing] "a, b; b, c => a, c" x y)
let cd_mixfix =
  let open! NTDSL.O in
    fun x y ->
      let missing =
        [%ocaml.error
          "ppx_ocannl: capture label \"missing\" is absent from einsum spec \"ab; bc => ac\"; a multi-letter capture label needs multichar mode: write `missing,` and separate the other axes with commas"] in
      Tensor.consume_forward_code
        (einsum ~capture_dims:[missing] "a, b; b, c => a, c" x y)
let rows =
  let open! TDSL.O in
    let __ocannl_body x =
      let batch = Shape.get_variable_ref "batch" in
      let input = Shape.get_variable_ref "input" in
      let output = Shape.get_variable_ref "output" in
      einsum1
        ?label:(Some
                  (List.concat [["rows"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[batch; input; output]
        "..batch..|..input..->..output.., a => a" x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"rows" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let result_only =
  let open! TDSL.O in
    let __ocannl_body x =
      let b = Shape.get_variable_ref "b" in
      einsum1
        ?label:(Some
                  (List.concat
                     [["result_only"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[b] "a => b" x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"result_only" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let window =
  let open! TDSL.O in
    fun ~stride ~dilation x ->
      let o = Shape.get_variable_ref "o" in
      let k = Shape.get_variable_ref "k" in
      einsum1
        ?label:(Some
                  (List.concat [["window"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[o; k]
        (String.concat ~sep:""
           [Int.to_string stride;
           "*";
           "o";
           "<";
           "+";
           Int.to_string dilation;
           "*";
           "k";
           " => ";
           "o"]) x
let segments =
  let open! TDSL.O in
    let __ocannl_body x =
      let left = Shape.get_variable_ref "left" in
      let middle = Shape.get_variable_ref "middle" in
      let right = Shape.get_variable_ref "right" in
      einsum1
        ?label:(Some
                  (List.concat
                     [["segments"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[left; middle; right] "left ^ middle ^ right => middle"
        x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"segments" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let multichar_ok =
  let open! TDSL.O in
    let __ocannl_body x =
      let ab = Shape.get_variable_ref "ab" in
      einsum1
        ?label:(Some
                  (List.concat
                     [["multichar_ok"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[ab] "ab, => ab," x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"multichar_ok" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let cd_multichar_ok =
  let open! NTDSL.O in
    fun x ->
      let ab = Shape.get_variable_ref "ab" in
      Tensor.consume_forward_code (einsum1 ~capture_dims:[ab] "ab, => ab," x)
let row_only =
  let open! TDSL.O in
    let __ocannl_body x =
      let axes = Shape.get_variable_ref "axes" in
      einsum1
        ?label:(Some
                  (List.concat
                     [["row_only"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[axes] "..axes,.. => ..axes,.." x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"row_only" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let coefficient =
  let open! TDSL.O in
    let __ocannl_body x =
      let stride =
        [%ocaml.error
          "ppx_ocannl: capture label \"stride\" is absent from einsum spec \"stride*o<+k => o\""] in
      einsum1
        ?label:(Some
                  (List.concat
                     [["coefficient"]; (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[stride]
        (String.concat ~sep:""
           [Int.to_string stride; "*"; "o"; "<"; "+"; "k"; " => "; "o"]) x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"coefficient" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
let dilation_coefficient =
  let open! TDSL.O in
    let __ocannl_body x =
      let dilation =
        [%ocaml.error
          "ppx_ocannl: capture label \"dilation\" is absent from einsum spec \"o<+dilation*k => o\""] in
      einsum1
        ?label:(Some
                  (List.concat
                     [["dilation_coefficient"];
                     (x.Tensor.value).Ir.Tnode.label]))
        ~capture_dims:[dilation]
        (String.concat ~sep:""
           ["o"; "<"; "+"; Int.to_string dilation; "*"; "k"; " => "; "o"]) x in
    let __ocannl_scheme =
      Tensor.Scheme.For_ppx.deferred ~name:"dilation_coefficient" ~arity:1
        (function | x::[] -> __ocannl_body x | _ -> assert false) in
    fun x ->
      Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
        (fun () -> __ocannl_body x)
