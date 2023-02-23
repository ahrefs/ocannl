open Base

open Ppxlib

let rec pat2expr pat =
  let loc = pat.ppat_loc in
  match pat.ppat_desc with
  | Ppat_constraint (pat', typ) ->
    Ast_builder.Default.pexp_constraint ~loc (pat2expr pat') typ
  | Ppat_alias (_, ident)
  | Ppat_var ident ->
    Ast_builder.Default.pexp_ident ~loc {ident with txt = Lident ident.txt}
  | _ ->
     Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
       "ppx_ocannl requires a pattern identifier here: try using an `as` alias."

let rec pat2pat_ref pat =
  let loc = pat.ppat_loc in
  match pat.ppat_desc with
  | Ppat_constraint (pat', _) -> pat2pat_ref pat'
  | Ppat_alias (_, ident)
  | Ppat_var ident -> Ast_builder.Default.ppat_var ~loc {ident with txt = ident.txt ^ "__ref"}
  | _ ->
    Ast_builder.Default.ppat_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl requires a pattern identifier here: try using an `as` alias."

let rec collect_list accu = function
  | [%expr [%e? hd] :: [%e? tl]] -> collect_list (hd::accu) tl
  | [%expr []] -> List.rev accu
  | expr -> List.rev (expr::accu)

let dim_spec_to_string = function
| `Input_dims dim -> "input (tuple) of dim "^Int.to_string dim
| `Output_dims dim -> "output (list) of dim "^Int.to_string dim
| `Batch_dims dim -> "batch (array) of dim "^Int.to_string dim

let ndarray_constant ?axis_labels ?label expr =
  let loc = expr.pexp_loc in
  (* Traverse the backbone of the ndarray to collect the dimensions. *)
  let rec loop_dims accu = function
    | { pexp_desc = Pexp_tuple (exp::_ as exps); _ } -> loop_dims (`Input_dims (List.length exps)::accu) exp
    | { pexp_desc = Pexp_array (exp::_ as exps); _ } -> loop_dims (`Batch_dims (List.length exps)::accu) exp
    | { pexp_desc = Pexp_tuple []; _ } -> `Input_dims 0::accu
    | { pexp_desc = Pexp_array []; _ } -> `Batch_dims 0::accu
    | { pexp_desc = Pexp_construct ({txt=Lident "::"; _}, _); _ } as expr ->
      let exps = collect_list [] expr in
      (match exps with
       | exp::_ -> loop_dims (`Output_dims (List.length exps)::accu) exp
       | [] -> `Output_dims 0::accu)
    | _ -> accu in
  let dims_spec = Array.of_list_rev @@ loop_dims [] expr in
  let open Ast_builder.Default in
  let rec loop_values depth accu expr =
    if depth >= Array.length dims_spec
    then match expr with
      | { pexp_desc = Pexp_constant (Pconst_float _); _ } -> expr::accu
      | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
        [%expr Float.of_int [%e expr]]::accu
      | { pexp_desc = Pexp_tuple _; pexp_loc=loc; _ } ->
        (pexp_extension ~loc
         @@ Location.error_extensionf ~loc
           "OCaNNL: ndarray literal found input axis (tuple), expected number")::accu
      | { pexp_desc = Pexp_array _; pexp_loc=loc; _ } -> 
        (pexp_extension ~loc
         @@ Location.error_extensionf ~loc
           "OCaNNL: ndarray literal found batch axis (array), expected number")::accu
      | { pexp_desc = Pexp_construct ({txt=Lident "::"; _}, _); _ } ->
        (pexp_extension ~loc
         @@ Location.error_extensionf ~loc
           "OCaNNL: ndarray literal found output axis (list), expected number")::accu
      | expr -> expr::accu (* it either computes a number, or becomes a type error *)
    else match expr with
      | { pexp_desc = Pexp_tuple exps; _ } ->
        (match dims_spec.(depth) with
         | `Input_dims dim when dim = List.length exps ->
           List.fold_left exps ~init:accu ~f:(loop_values @@ depth + 1)
         | dim_spec ->
           (pexp_extension ~loc
            @@ Location.error_extensionf ~loc
              "OCaNNL: ndarray literal axis mismatch, got %s, expected %s"
              (dim_spec_to_string @@ `Input_dims (List.length exps)) (dim_spec_to_string dim_spec))
           ::accu)
      | { pexp_desc = Pexp_array exps; _ } ->
        (match dims_spec.(depth) with
         | `Batch_dims dim when dim = List.length exps ->
           List.fold_left exps ~init:accu ~f:(loop_values @@ depth + 1)
         | dim_spec ->
           (pexp_extension ~loc
            @@ Location.error_extensionf ~loc
              "OCaNNL: ndarray literal axis mismatch, got %s, expected %s"
              (dim_spec_to_string @@ `Batch_dims (List.length exps)) (dim_spec_to_string dim_spec))
           ::accu)
      | { pexp_desc = Pexp_construct ({txt=Lident "::"; _}, _); _ } ->
        let exps = collect_list [] expr in
        (match dims_spec.(depth) with
         | `Output_dims dim when dim = List.length exps ->
           List.fold_left exps ~init:accu ~f:(loop_values @@ depth + 1)
         | dim_spec ->
           (pexp_extension ~loc
            @@ Location.error_extensionf ~loc
              "OCaNNL: ndarray literal axis mismatch, got %s, expected %s"
              (dim_spec_to_string @@ `Output_dims (List.length exps)) (dim_spec_to_string dim_spec))
           ::accu)
      | { pexp_loc=loc; _ } ->
        (pexp_extension ~loc
         @@ Location.error_extensionf ~loc
           "OCaNNL: ndarray literal: expected an axis (tuple, list or array)")::accu in
  let result = loop_values 0 [] expr in
  let values = {expr with pexp_desc = Pexp_array (List.rev result)} in
  let batch_dims, output_dims, input_dims =
    Array.fold dims_spec ~init:([], [], []) ~f:(fun (batch_dims, output_dims, input_dims) -> 
      function
      | `Input_dims dim -> batch_dims, output_dims, eint ~loc dim::input_dims
      | `Output_dims dim -> batch_dims, eint ~loc dim::output_dims, input_dims
      | `Batch_dims dim -> eint ~loc dim::batch_dims, output_dims, input_dims) in
  let edims dims = elist ~loc @@ List.rev dims in
  let op =
    match axis_labels, label with
    | None, None -> [%expr Operation.ndarray]
    | Some axis_labels, None -> [%expr Operation.ndarray ?axis_labels:[%e axis_labels]]
    | None, Some label -> [%expr Operation.ndarray ?label:[%e label]]
    | Some axis_labels, Some label ->
      [%expr Operation.ndarray ?axis_labels:[%e axis_labels] ?label:[%e label]] in
  [%expr Network.return_term
      ([%e op] ~batch_dims:[%e edims batch_dims] ~input_dims:[%e edims input_dims]
         ~output_dims:[%e edims output_dims] [%e values])]

let rec translate expr =
  let loc = expr.pexp_loc in
  match expr with
  | { pexp_desc = Pexp_constant (Pconst_float _); _ } ->
    [%expr Network.return_term (Operation.number [%e expr])]

  | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
    [%expr Network.return_term (Operation.number (Float.of_int [%e expr]))]

  | [%expr [%e? { pexp_desc = Pexp_constant (Pconst_string _); _ } as s]
      [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
    [%expr Network.return_term (Operation.number ~axis_label:[%e s] [%e f])]

  | [%expr [%e? { pexp_desc = Pexp_constant (Pconst_string _); _ } as s]
      [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
    [%expr Network.return_term (Operation.number ~axis_label:[%e s] (Float.of_int [%e i]))]

  | { pexp_desc = Pexp_constant (Pconst_string _); _ } ->
    [%expr Network.return_term Operation.O.(!~ [%e expr])]

  | { pexp_desc = Pexp_tuple _; _ } | { pexp_desc = Pexp_array _; _ } 
  | { pexp_desc = Pexp_construct ({txt=Lident "::"; _}, _); _ } -> ndarray_constant expr
    
  | [%expr [%e? expr1] [%e? expr2] [%e? expr3] ] ->
    [%expr Network.apply (Network.apply [%e translate expr1] [%e translate expr2]) [%e translate expr3]]

  | [%expr [%e? expr1] [%e? expr2] ] ->
    [%expr Network.apply [%e translate expr1] [%e translate expr2]]

  | [%expr fun ~config [%p? pat1] [%p? pat2] -> [%e? body] ] ->
    (* TODO(#38): generalize config to any number of labeled arguments with any labels. *)
    let pat1_ref = pat2pat_ref pat1 in
    let pat2_ref = pat2pat_ref pat2 in
    [%expr
      fun ~config ->
        let [%p pat1_ref] = ref [] in
        let [%p pat1] = Network.return (Network.Placeholder [%e pat2expr @@ pat1_ref]) in
        let [%p pat2_ref] = ref [] in
        let [%p pat2] = Network.return (Network.Placeholder [%e pat2expr @@ pat2_ref]) in
        let body = [%e translate body] in
        fun [%p pat1] [%p pat2] ->
          [%e pat2expr pat1_ref] := [%e pat2expr pat1] :: ![%e pat2expr pat1_ref];
          [%e pat2expr pat2_ref] := [%e pat2expr pat2] :: ![%e pat2expr pat2_ref];
          let result__ = Network.unpack body in
          [%e pat2expr pat1_ref] := List.tl_exn ![%e pat2expr pat1_ref];
          [%e pat2expr pat2_ref] := List.tl_exn ![%e pat2expr pat2_ref];
          result__
    ]

  | [%expr fun ~config [%p? pat] -> [%e? body] ] ->
    (* TODO(#38): generalize config to any number of labeled arguments with any labels. *)
    let pat_ref = pat2pat_ref pat in
    [%expr
      fun ~config ->
        let [%p pat_ref] = ref [] in
        let [%p pat] = Network.return (Network.Placeholder [%e pat2expr @@ pat_ref]) in
        let body = [%e translate body] in
        fun [%p pat] ->
          [%e pat2expr pat_ref] := [%e pat2expr pat] :: ![%e pat2expr pat_ref];
          let result__ = Network.unpack body in
          [%e pat2expr pat_ref] := List.tl_exn ![%e pat2expr pat_ref];
          result__
    ]

  | [%expr fun [%p? pat1] [%p? pat2] -> [%e? body] ] ->
    let pat1_ref = pat2pat_ref pat1 in
    let pat2_ref = pat2pat_ref pat2 in
    [%expr
      let [%p pat1_ref] = ref [] in
      let [%p pat1] = Network.return (Network.Placeholder [%e pat2expr @@ pat1_ref]) in
      let [%p pat2_ref] = ref [] in
      let [%p pat2] = Network.return (Network.Placeholder [%e pat2expr @@ pat2_ref]) in
      let body = [%e translate body] in
      fun [%p pat1] [%p pat2] ->
        [%e pat2expr pat1_ref] := [%e pat2expr pat1] :: ![%e pat2expr pat1_ref];
        [%e pat2expr pat2_ref] := [%e pat2expr pat2] :: ![%e pat2expr pat2_ref];
        let result__ = Network.unpack body in
        [%e pat2expr pat1_ref] := List.tl_exn ![%e pat2expr pat1_ref];
        [%e pat2expr pat2_ref] := List.tl_exn ![%e pat2expr pat2_ref];
        result__
    ]

  | [%expr fun [%p? pat] -> [%e? body] ] ->
    let pat_ref = pat2pat_ref pat in
    [%expr
      let [%p pat_ref] = ref [] in
      let [%p pat] = Network.return (Network.Placeholder [%e pat2expr @@ pat_ref]) in
      let body = [%e translate body] in
      fun [%p pat] ->
        [%e pat2expr pat_ref] := [%e pat2expr pat] :: ![%e pat2expr pat_ref];
        let result__ = Network.unpack body in
        [%e pat2expr pat_ref] := List.tl_exn ![%e pat2expr pat_ref];
        result__
  ]

  | [%expr while [%e? test_expr] do [%e? body_expr] done ] ->
    [%expr while [%e test_expr] do [%e translate body_expr] done ]

  | [%expr for [%p? pat] = [%e? init] to [%e? final] do [%e? body] done ] ->
    [%expr for [%p pat] = [%e init] to [%e final] do [%e translate body] done ]

  | [%expr for [%p? pat] = [%e? init] downto [%e? final] do [%e? body] done ] ->
    [%expr for [%p pat] = [%e init] downto [%e final] do [%e translate body] done ]

  | [%expr [%e? expr1] ; [%e? expr2] ] ->
    [%expr [%e translate expr1] ; [%e translate expr2]]

  | [%expr if [%e? expr1] then [%e? expr2] else [%e? expr3]] ->
    [%expr if [%e expr1] then [%e translate expr2] else [%e translate expr3]]

  | [%expr if [%e? expr1] then [%e? expr2]] ->
    [%expr if [%e expr1] then [%e translate expr2]]

  | { pexp_desc = Pexp_match (expr1, cases); _ } ->
    let cases =
       List.map cases
         ~f:(fun ({pc_rhs; _} as c) ->
            {c with pc_rhs=translate pc_rhs}) in
     { expr with pexp_desc = Pexp_match (expr1, cases) }

  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
     let body = translate body in
     let bindings = List.map bindings ~f:(fun binding ->
         {binding with pvb_expr = translate binding.pvb_expr}) in
     {expr with pexp_desc=Pexp_let (recflag, bindings, body)}

  | { pexp_desc = Pexp_open (decl, body); _ } ->
     {expr with pexp_desc=Pexp_open (decl, translate body)}

  | { pexp_desc = Pexp_letmodule (name, module_expr, body); _ } ->
     {expr with pexp_desc=Pexp_letmodule (name, module_expr, translate body)}

  | expr ->
    expr

let expr_expander ~loc ~path:_ payload =
  match payload with
  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
    (* We are at the %ocannl annotation level: do not tranlsate the body. *)
     let bindings = List.map bindings ~f:(fun vb ->
         {vb with pvb_expr=[%expr let open! Network.O in [%e translate vb.pvb_expr]]}) in
     {payload with pexp_desc=Pexp_let (recflag, bindings, body)}
  | expr -> translate expr

let flatten_str ~loc ~path:_ items =
  match items with
  | [x] -> x
  | _ ->
    Ast_helper.Str.include_ {
       pincl_mod = Ast_helper.Mod.structure items
     ; pincl_loc = loc
     ; pincl_attributes = [] }

let translate_str ({pstr_desc; _} as str) =
  match pstr_desc with
  | Pstr_eval (expr, attrs) -> {str with pstr_desc=Pstr_eval (translate expr, attrs)}
  | Pstr_value (recf, vbl) ->
    let f vb =
      let loc = vb.pvb_loc in
      {vb with pvb_expr=[%expr let open! Network.O in [%e translate vb.pvb_expr]]} in
    {str with pstr_desc=Pstr_value (recf, List.map vbl ~f)}
  | _ -> str
     
let str_expander ~loc ~path (payload: structure_item list) =
  flatten_str ~loc ~path @@ List.map payload ~f:translate_str

let rules =
  [Ppxlib.Context_free.Rule.extension  @@
   Extension.declare "ocannl" Extension.Context.expression Ast_pattern.(single_expr_payload __) expr_expander;
   Ppxlib.Context_free.Rule.extension  @@
   Extension.declare "ocannl" Extension.Context.structure_item Ast_pattern.(pstr __) str_expander;
   ]

let () =
  Driver.register_transformation
    ~rules
    "ppx_ocannl"
