open Base

open Ppxlib

open Ppx_nn_shared

let ndarray_op ?axis_labels ?label expr =
  let loc = expr.pexp_loc in
  let values, batch_dims, output_dims, input_dims = ndarray_constant expr in
  let edims dims = Ast_builder.Default.elist ~loc @@ List.rev dims in
  let op =
    match axis_labels, label with
    | None, None -> [%expr Formula.DSL.ndarray]
    | Some axis_labels, None -> [%expr Formula.DSL.ndarray ?axis_labels:[%e axis_labels]]
    | None, Some label -> [%expr Formula.DSL.ndarray ?label:[%e label]]
    | Some axis_labels, Some label ->
      [%expr Formula.DSL.ndarray ?axis_labels:[%e axis_labels] ?label:[%e label]] in
  [%expr
    [%e op] ~batch_dims:[%e edims batch_dims] ~input_dims:[%e edims input_dims]
      ~output_dims:[%e edims output_dims] [%e values]]

let make_vb ?init ~loc ~str_loc ~ident string =
  let pat = Ast_helper.Pat.var ~loc {loc=str_loc; txt=ident} in
  let init = match init with Some c -> [%expr Some [%e c]] | None -> [%expr None] in
  let v = [%expr DSL.unconstrained_param ?init:[%e init] [%e string]] in
  let vb = Ast_helper.Vb.mk ~loc pat v in
  pat, vb

let make_vb_nd ~loc ~str_loc ?axis_labels ~ident ~init_nd string =
  let pat = Ast_helper.Pat.var ~loc {loc=str_loc; txt=ident} in
  let values, batch_dims, output_dims, input_dims = ndarray_constant init_nd in
  let v =
    if not @@ List.is_empty batch_dims then
      Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
        "ppx_ocannl params cannot have batch dims: define a constant or remove the array syntax."
    else
      let edims dims = Ast_builder.Default.elist ~loc @@ List.rev dims in
      let op =
        match axis_labels with
        | None -> [%expr Operation.given_dims_params]
        | Some axis_labels -> [%expr Operation.given_dims_params ?axis_labels:[%e axis_labels]] in
      [%expr [%e op] ~input_dims:[%e edims input_dims]
          ~output_dims:[%e edims output_dims] [%e string] [%e values]] in
  let vb = Ast_helper.Vb.mk ~loc pat v in
  pat, vb

let rec translate expr =
  let loc = expr.pexp_loc in
  match expr with
  | { pexp_desc = Pexp_constant (Pconst_float _); _ } ->
    no_vbs, [%expr Formula.DSL.number [%e expr]]

  | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
    no_vbs, [%expr Formula.DSL.number (Float.of_int [%e expr])]

  | [%expr [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
      [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
    let axis = Ast_helper.Exp.constant ~loc:pexp_loc
        (Pconst_string (String.of_char ch, pexp_loc, None)) in
    no_vbs, [%expr Formula.DSL.number ~axis_label:[%e axis] [%e f]]

  | [%expr [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
      [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
        let axis = Ast_helper.Exp.constant ~loc:pexp_loc
        (Pconst_string (String.of_char ch, pexp_loc, None)) in
    no_vbs, [%expr Formula.DSL.number ~axis_label:[%e axis] (Float.of_int [%e i])]

  | [%expr [%e? { pexp_desc = Pexp_constant (Pconst_string (ident, str_loc, _)); _ } as s]
      [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
    let pat, vb = make_vb ~init:f ~loc ~str_loc ~ident s in
    Map.singleton (module String) ident vb, pat2expr pat

  | [%expr [%e? { pexp_desc = Pexp_constant (Pconst_string (ident, str_loc, _)); _ } as s]
      [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
    let pat, vb = make_vb ~init:[%expr Float.of_int [%e i]] ~loc ~str_loc ~ident s in
    Map.singleton (module String) ident vb, pat2expr pat

  | [%expr [%e? { pexp_desc = Pexp_constant (Pconst_string (ident, str_loc, _)); _ } as s]
      [%e? ({ pexp_desc = Pexp_tuple _; _ } | { pexp_desc = Pexp_array _; _ } 
           | { pexp_desc = Pexp_construct ({txt=Lident "::"; _}, _); _ }) as init_nd]] ->
    let pat, vb = make_vb_nd ~loc ~str_loc ~ident ~init_nd s in
    Map.singleton (module String) ident vb, pat2expr pat

  | { pexp_desc = Pexp_constant (Pconst_string (ident, str_loc, _)); _ } ->
    let pat, vb = make_vb ~loc ~str_loc ~ident expr in
    Map.singleton (module String) ident vb, pat2expr pat

  | { pexp_desc = Pexp_tuple _; _ } | { pexp_desc = Pexp_array _; _ } 
  | { pexp_desc = Pexp_construct ({txt=Lident "::"; _}, _); _ } ->
    no_vbs, ndarray_op expr

  | [%expr [%e? expr1] **.
      [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
    let vbs, e1 = translate expr1 in
    vbs, [%expr DSL.O.( **. ) [%e e1] [%e f]]

  | [%expr [%e? expr1] **.
      [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
    let vbs, e1 = translate expr1 in
    vbs, [%expr DSL.O.( **. ) [%e e1] (Float.of_int [%e i])]

  | [%expr [%e? expr1] [%e? expr2] [%e? expr3] ] ->
    let vbs1, e1 = translate expr1 in
    let vbs2, e2 = translate expr2 in
    let vbs3, e3 = translate expr3 in
    reduce_vbss [vbs1; vbs2; vbs3],
    [%expr [%e e1] [%e e2] [%e e3]]

  | [%expr [%e? expr1] [%e? expr2] ] ->
    let vbs1, e1 = translate expr1 in
    let vbs2, e2 = translate expr2 in
    Map.merge_skewed vbs1 vbs2 ~combine:(fun ~key:_ _v1 v2 -> v2),
    [%expr [%e e1] [%e e2]]

  | [%expr fun ~config [%p? pat1] [%p? pat2] -> [%e? body] ] ->
    (* TODO(#38): generalize config to any number of labeled arguments with any labels. *)
    let vbs, body = translate body in
    no_vbs, [%expr fun ~config -> [%e let_opt ~loc vbs [%expr fun [%p pat1] [%p pat2] -> [%e body]]]]

  | [%expr fun ~config [%p? pat] -> [%e? body] ] ->
    (* TODO(#38): generalize config to any number of labeled arguments with any labels. *)
    let vbs, body = translate body in
    no_vbs, [%expr fun ~config -> [%e let_opt ~loc vbs [%expr fun [%p pat] -> [%e body]]]]

  | [%expr fun [%p? pat1] [%p? pat2] -> [%e? body] ] ->
    let vbs, body = translate body in
    vbs, [%expr fun [%p pat1] [%p pat2] -> [%e body] ]

  | [%expr fun [%p? pat] -> [%e? body] ] ->
    let vbs, body = translate body in
    vbs, [%expr fun [%p pat] -> [%e body] ]

  | [%expr while [%e? test_expr] do [%e? body_expr] done ] ->
    let vbs, body = translate body_expr in
    vbs, [%expr while [%e test_expr] do [%e body] done ]

  | [%expr for [%p? pat] = [%e? init] to [%e? final] do [%e? body_expr] done ] ->
    let vbs, body = translate body_expr in
    vbs, [%expr for [%p pat] = [%e init] to [%e final] do [%e body] done ]

  | [%expr for [%p? pat] = [%e? init] downto [%e? final] do [%e? body_expr] done ] ->
    let vbs, body = translate body_expr in
    vbs, [%expr for [%p pat] = [%e init] downto [%e final] do [%e body] done ]

  | [%expr [%e? expr1] ; [%e? expr2] ] ->
    let vbs1, e1 = translate expr1 in
    let vbs2, e2 = translate expr2 in
    Map.merge_skewed vbs1 vbs2 ~combine:(fun ~key:_ _v1 v2 -> v2), [%expr [%e e1] ; [%e e2]]

  | [%expr if [%e? expr1] then [%e? expr2] else [%e? expr3]] ->
    let vbs2, e2 = translate expr2 in
    let vbs3, e3 = translate expr3 in
    Map.merge_skewed vbs2 vbs3 ~combine:(fun ~key:_ _v1 v2 -> v2), [%expr if [%e expr1] then [%e e2] else [%e e3]]

  | [%expr if [%e? expr1] then [%e? expr2]] ->
    let vbs2, e2 = translate expr2 in
    vbs2, [%expr if [%e expr1] then [%e e2]]

  | { pexp_desc = Pexp_match (expr1, cases); _ } ->
    let vbss, cases =
       List.unzip @@ List.map cases
         ~f:(fun ({pc_rhs; _} as c) ->
            let vbs, pc_rhs = translate pc_rhs in
            vbs, {c with pc_rhs}) in
     reduce_vbss vbss, { expr with pexp_desc = Pexp_match (expr1, cases) }

  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
     let vbss1, bindings = List.unzip @@ List.map bindings
         ~f:(fun binding ->
          let vbs, pvb_expr = translate binding.pvb_expr in
          vbs, {binding with pvb_expr}) in
     let vbs2, body = translate body in
     let all_bindings = (Map.data @@ reduce_vbss vbss1) @ bindings @ Map.data vbs2 in
     no_vbs, {expr with pexp_desc=Pexp_let (recflag, all_bindings, body)}

  | { pexp_desc = Pexp_open (decl, body); _ } ->
    let vbs, body = translate body in
    vbs, {expr with pexp_desc=Pexp_open (decl, body)}

  | { pexp_desc = Pexp_letmodule (name, module_expr, body); _ } ->
    let vbs, body = translate body in
     vbs, {expr with pexp_desc=Pexp_letmodule (name, module_expr, body)}

  | expr ->
    no_vbs, expr

let expr_expander ~loc ~path:_ payload =
  match payload with
  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
    (* We are at the %ocannl annotation level: do not tranlsate the body. *)
     let vbss, bindings = List.unzip @@ List.map bindings
      ~f:(fun vb ->
        let vbs, v = translate vb.pvb_expr in
        vbs, {vb with pvb_expr=[%expr let open! DSL.O in [%e v]]}) in
     let expr = {payload with pexp_desc=Pexp_let (recflag, bindings, body)} in
     let_opt ~loc (reduce_vbss vbss) expr
  | expr ->
    let vbs, expr = translate expr in
    let_opt ~loc vbs expr

let flatten_str ~loc ~path:_ items =
  match items with
  | [x] -> x
  | _ ->
    Ast_helper.Str.include_ {
       pincl_mod = Ast_helper.Mod.structure items
     ; pincl_loc = loc
     ; pincl_attributes = [] }

let translate_str ({pstr_desc; pstr_loc=loc; _} as str) =
  match pstr_desc with
  | Pstr_eval (expr, attrs) ->
    let vbs, expr = translate expr in
    let expr = let_opt ~loc vbs expr in
    {str with pstr_desc=Pstr_eval (expr, attrs)}
  | Pstr_value (recf, bindings) ->
    let f vb =
      let loc = vb.pvb_loc in
      let vbs, v = translate vb.pvb_expr in
      let v = let_opt ~loc vbs v in
      {vb with pvb_expr=[%expr let open! DSL.O in [%e v]]} in
    {str with pstr_desc=Pstr_value (recf, List.map bindings ~f)}
  | _ -> str
     
let str_expander ~loc ~path (payload: structure_item list) =
  flatten_str ~loc ~path @@ List.map payload ~f:translate_str
