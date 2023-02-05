open Base

open Ppxlib

let pat2expr pat =
  let loc = pat.ppat_loc in
  match pat.ppat_desc with
  | Ppat_var ident -> Ast_builder.Default.pexp_ident ~loc {ident with txt = Lident ident.txt}
  | _ ->
     Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
       "OCaNNL currently only supports single identifiers as argument patterns."

let pat2pat_ref pat =
  let loc = pat.ppat_loc in
  match pat.ppat_desc with
  | Ppat_var ident -> Ast_builder.Default.ppat_var ~loc {ident with txt = ident.txt ^ "_ref"}
  | _ ->
     Ast_builder.Default.ppat_extension ~loc @@ Location.error_extensionf ~loc
       "OCaNNL currently only supports single identifiers as argument patterns."

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
    
  | [%expr [%e? expr1] [%e? expr2] [%e? expr3] ] ->
    [%expr Network.apply (Network.apply [%e translate expr1] [%e translate expr2]) [%e translate expr3]]

  | [%expr [%e? expr1] [%e? expr2] ] ->
    [%expr Network.apply [%e translate expr1] [%e translate expr2]]

  | [%expr fun ~config [%p? pat1] [%p? pat2] -> [%e? body] ] ->
    (* TODO(38): generalize config to any number of labeled arguments with any labels. *)
    let pat1_ref = pat2pat_ref pat1 in
    let pat2_ref = pat2pat_ref pat2 in
    [%expr
      fun ~config ->
        let [%p pat1_ref] = ref None in
        let [%p pat1] = Network.return (Network.Placeholder [%e pat2expr @@ pat1_ref]) in
        let [%p pat2_ref] = ref None in
        let [%p pat2] = Network.return (Network.Placeholder [%e pat2expr @@ pat2_ref]) in
        let body = [%e translate body] in
        fun [%p pat1] [%p pat2] ->
          [%p pat1_ref] := [%e pat2expr pat1]; [%p pat2_ref] := [%e pat2expr pat2];
          Network.unpack body
    ]

  | [%expr fun ~config [%p? pat] -> [%e? body] ] ->
    (* TODO(38): generalize config to any number of labeled arguments with any labels. *)
    let pat_ref = pat2pat_ref pat in
    [%expr
      fun ~config ->
        let [%p pat_ref] = ref None in
        let [%p pat] = Network.return (Network.Placeholder [%e pat2expr @@ pat_ref]) in
        let body = [%e translate body] in
        fun [%p pat] ->
          [%p pat_ref] := [%e pat2expr pat];
          Network.unpack body
    ]

  | [%expr fun [%p? pat1] [%p? pat2] -> [%e? body] ] ->
    let pat1_ref = pat2pat_ref pat1 in
    let pat2_ref = pat2pat_ref pat2 in
    [%expr
      let [%p pat1_ref] = ref None in
      let [%p pat1] = Network.return (Network.Placeholder [%e pat2expr @@ pat1_ref]) in
      let [%p pat2_ref] = ref None in
      let [%p pat2] = Network.return (Network.Placeholder [%e pat2expr @@ pat2_ref]) in
      let body = [%e translate body] in
      fun [%p pat1] [%p pat2] ->
        [%p pat1_ref] := [%e pat2expr pat1]; [%p pat2_ref] := [%e pat2expr pat2];
        Network.unpack body
    ]

  | [%expr fun [%p? pat] -> [%e? body] ] ->
    let pat_ref = pat2pat_ref pat in
    [%expr
      let [%p pat_ref] = ref None in
      let [%p pat] = Network.return (Network.Placeholder [%e pat2expr @@ pat_ref]) in
      let body = [%e translate body] in
      fun [%p pat] ->
        [%p pat_ref] := [%e pat2expr pat];
        Network.unpack body
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
         {vb with pvb_expr=[%expr let open Network.O in [%e translate vb.pvb_expr]]}) in
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
      {vb with pvb_expr=[%expr let open Network.O in [%e translate vb.pvb_expr]]} in
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
