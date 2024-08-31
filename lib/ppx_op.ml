open Base
open Ppxlib
open Ppx_arrayjit.Ppx_helper
open Ppx_shared

let ndarray_op ?ident_label ?axis_labels expr =
  let loc = expr.pexp_loc in
  let values, batch_dims, output_dims, input_dims = ndarray_constant expr in
  let edims dims = Ast_builder.Default.elist ~loc dims in
  let op =
    match axis_labels with
    | None -> [%expr TDSL.ndarray]
    | Some axis_labels -> [%expr TDSL.ndarray ~axis_labels:[%e axis_labels]]
  in
  [%expr
    [%e op]
      ~label:[%e opt_pat2string_list ~loc ident_label]
      ~batch_dims:[%e edims batch_dims] ~input_dims:[%e edims input_dims]
      ~output_dims:[%e edims output_dims] [%e values]]

let make_p ~has_config ~loc =
  if has_config then [%expr TDSL.param ~more_label:config.label] else [%expr TDSL.param]

let make_vb ?value ~has_config ~loc ~str_loc ~ident string =
  let pat = Ast_helper.Pat.var ~loc { loc = str_loc; txt = ident } in
  let value = match value with Some c -> [%expr Some [%e c]] | None -> [%expr None] in
  let v = [%expr [%e make_p ~has_config ~loc] ?values:[%e value] [%e string]] in
  let vb = Ast_helper.Vb.mk ~loc pat v in
  (pat, vb)

let make_vb_dims ~has_config ~loc ~str_loc ~ident ~dims ~dims_loc string =
  let pat = Ast_helper.Pat.var ~loc { loc = str_loc; txt = ident } in
  let dims =
    let loc = dims_loc in
    List.fold_right dims ~init:[%expr []] ~f:(fun d ds -> [%expr [%e d] :: [%e ds]])
  in
  let v = [%expr [%e make_p ~has_config ~loc] ~output_dims:[%e dims] [%e string]] in
  let vb = Ast_helper.Vb.mk ~loc pat v in
  (pat, vb)

let make_vb_nd ~has_config ~loc ~str_loc ?axis_labels ~ident ~init_nd string =
  let pat = Ast_helper.Pat.var ~loc { loc = str_loc; txt = ident } in
  let values, batch_dims, output_dims, input_dims = ndarray_constant init_nd in
  let v =
    if not @@ List.is_empty batch_dims then
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "ppx_ocannl param cannot have batch dims: define a constant or remove the array syntax."
    else
      let edims dims = Ast_builder.Default.elist ~loc dims in
      let op =
        match axis_labels with
        | None -> make_p ~has_config ~loc
        | Some axis_labels -> [%expr [%e make_p ~has_config ~loc] ~axis_labels:[%e axis_labels]]
      in
      [%expr
        [%e op] ~input_dims:[%e edims input_dims] ~output_dims:[%e edims output_dims]
          ~values:[%e values] [%e string]]
  in
  let vb = Ast_helper.Vb.mk ~loc pat v in
  (pat, vb)

let rec translate ~has_config ?ident_label expr =
  let loc = expr.pexp_loc in
  let loop = translate ~has_config in
  match expr with
  | { pexp_desc = Pexp_constant (Pconst_float _); _ } ->
      (no_vbs, [%expr TDSL.number ~label:[%e opt_pat2string_list ~loc ident_label] [%e expr]])
  | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
      (no_vbs, [%expr TDSL.number (Float.of_int [%e expr])])
  | [%expr
      [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
        [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
      let axis =
        Ast_helper.Exp.constant ~loc:pexp_loc (Pconst_string (String.of_char ch, pexp_loc, None))
      in
      ( no_vbs,
        [%expr
          TDSL.number ~label:[%e opt_pat2string_list ~loc ident_label] ~axis_label:[%e axis] [%e f]]
      )
  | [%expr
      [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
        [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
      let axis =
        Ast_helper.Exp.constant ~loc:pexp_loc (Pconst_string (String.of_char ch, pexp_loc, None))
      in
      ( no_vbs,
        [%expr
          TDSL.number
            ~label:[%e opt_pat2string_list ~loc ident_label]
            ~axis_label:[%e axis]
            (Float.of_int [%e i])] )
  | [%expr
      [%e? expr1]
      *+ [%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ } as spec] [%e? expr2]]
    when String.contains spec_str '>' ->
      let vbs1, e1 = loop expr1 in
      let vbs2, e2 = loop expr2 in
      ( reduce_vbss [ vbs1; vbs2 ],
        [%expr
          TDSL.einsum ~label:[%e opt_pat2string_list ~loc ident_label] [%e spec] [%e e1] [%e e2]] )
  | [%expr
      [%e? expr1] ++ [%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ } as spec]]
    when String.contains spec_str '>' ->
      let vbs1, e1 = loop expr1 in
      (vbs1, [%expr TDSL.einsum1 ~label:[%e opt_pat2string_list ~loc ident_label] [%e spec] [%e e1]])
  | [%expr
      [%e? { pexp_desc = Pexp_constant (Pconst_string (ident, str_loc, _)); _ } as s]
        [%e?
          ( { pexp_desc = Pexp_constant (Pconst_integer _); pexp_loc = dims_loc; _ }
          | { pexp_desc = Pexp_ident _; pexp_loc = dims_loc; _ }
          | { pexp_desc = Pexp_field _; pexp_loc = dims_loc; _ } ) as d]] ->
      let pat, vb = make_vb_dims ~has_config ~loc ~str_loc ~ident ~dims:[ d ] ~dims_loc s in
      (Map.singleton (module String) ident vb, pat2expr pat)
  | [%expr
      [%e? { pexp_desc = Pexp_constant (Pconst_string (ident, str_loc, _)); _ } as s]
        [%e?
          ( { pexp_desc = Pexp_array _; _ }
          | { pexp_desc = Pexp_construct ({ txt = Lident "::"; _ }, _); _ } ) as init_nd]] ->
      let pat, vb = make_vb_nd ~has_config ~loc ~str_loc ~ident ~init_nd s in
      (Map.singleton (module String) ident vb, pat2expr pat)
  | [%expr
      [%e? { pexp_desc = Pexp_constant (Pconst_string (ident, str_loc, _)); _ } as s]
        [%e? { pexp_desc = Pexp_tuple dims; pexp_loc = dims_loc; _ }]] ->
      let pat, vb = make_vb_dims ~has_config ~loc ~str_loc ~ident ~dims ~dims_loc s in
      (Map.singleton (module String) ident vb, pat2expr pat)
  | { pexp_desc = Pexp_constant (Pconst_string (ident, str_loc, _)); _ } ->
      let pat, vb = make_vb ~has_config ~loc ~str_loc ~ident expr in
      (Map.singleton (module String) ident vb, pat2expr pat)
  | { pexp_desc = Pexp_array _; _ }
  | { pexp_desc = Pexp_construct ({ txt = Lident "::"; _ }, _); _ } ->
      (no_vbs, ndarray_op ?ident_label expr)
  | [%expr [%e? expr1] **. [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
      (* We need to hardcode these two patterns to prevent the numbers from being converted to
         tensors. *)
      let vbs, e1 = loop expr1 in
      ( vbs,
        [%expr
          TDSL.O.( **. )
            ~label:[%e opt_pat2string_list ~loc ident_label]
            [%e e1]
            (Float.of_int [%e i])] )
  | [%expr [%e? expr1] **. [%e? expr2]] ->
      let vbs, e1 = loop expr1 in
      ( vbs,
        [%expr TDSL.O.( **. ) ~label:[%e opt_pat2string_list ~loc ident_label] [%e e1] [%e expr2]]
      )
  | [%expr [%e? expr1] [%e? expr2] [%e? expr3]] ->
      let vbs1, e1 = loop ?ident_label expr1 in
      let vbs2, e2 = loop expr2 in
      let vbs3, e3 = loop expr3 in
      (reduce_vbss [ vbs1; vbs2; vbs3 ], [%expr [%e e1] [%e e2] [%e e3]])
  | [%expr [%e? expr1] [%e? expr2]] ->
      let vbs1, e1 = loop ?ident_label expr1 in
      let vbs2, e2 = loop expr2 in
      (Map.merge_skewed vbs1 vbs2 ~combine:(fun ~key:_ _v1 v2 -> v2), [%expr [%e e1] [%e e2]])
  | [%expr fun ~config -> [%e? body]] ->
      let vbs, body = translate ~has_config:true ?ident_label body in
      (no_vbs, [%expr fun ~config -> [%e let_opt ~loc vbs body]])
  | [%expr fun ~(config : [%typ? config_ty]) -> [%e? body]] ->
      let vbs, body = translate ~has_config:true ?ident_label body in
      (no_vbs, [%expr fun ~(config : [%typ ty]) -> [%e let_opt ~loc vbs body]])
  | [%expr fun [%p? pat] -> [%e? body]] ->
      let vbs, body = loop ?ident_label body in
      (vbs, [%expr fun [%p pat] -> [%e body]])
  | [%expr
      while [%e? test_expr] do
        [%e? body_expr]
      done] ->
      let vbs, body = loop ?ident_label body_expr in
      ( vbs,
        [%expr
          while [%e test_expr] do
            [%e body]
          done] )
  | [%expr
      for [%p? pat] = [%e? init] to [%e? final] do
        [%e? body_expr]
      done] ->
      let vbs, body = loop ?ident_label body_expr in
      ( vbs,
        [%expr
          for [%p pat] = [%e init] to [%e final] do
            [%e body]
          done] )
  | [%expr
      for [%p? pat] = [%e? init] downto [%e? final] do
        [%e? body_expr]
      done] ->
      let vbs, body = loop ?ident_label body_expr in
      ( vbs,
        [%expr
          for [%p pat] = [%e init] downto [%e final] do
            [%e body]
          done] )
  | [%expr
      [%e? expr1];
      [%e? expr2]] ->
      let vbs1, e1 = loop expr1 in
      let vbs2, e2 = loop ?ident_label expr2 in
      ( reduce_vbss [ vbs1; vbs2 ],
        [%expr
          [%e e1];
          [%e e2]] )
  | [%expr if [%e? expr1] then [%e? expr2] else [%e? expr3]] ->
      let vbs2, e2 = loop ?ident_label expr2 in
      let vbs3, e3 = loop ?ident_label expr3 in
      (reduce_vbss [ vbs2; vbs3 ], [%expr if [%e expr1] then [%e e2] else [%e e3]])
  | [%expr if [%e? expr1] then [%e? expr2]] ->
      let vbs2, e2 = loop ?ident_label expr2 in
      (vbs2, [%expr if [%e expr1] then [%e e2]])
  | { pexp_desc = Pexp_match (expr1, cases); _ } ->
      let vbss, cases =
        List.unzip
        @@ List.map cases ~f:(fun ({ pc_rhs; _ } as c) ->
               let vbs, pc_rhs = loop ?ident_label pc_rhs in
               (vbs, { c with pc_rhs }))
      in
      (reduce_vbss vbss, { expr with pexp_desc = Pexp_match (expr1, cases) })
  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
      let vbss1, bindings =
        List.unzip
        @@ List.map bindings ~f:(fun binding ->
               let vbs, pvb_expr = loop ~ident_label:binding.pvb_pat binding.pvb_expr in
               (vbs, { binding with pvb_expr }))
      in
      let vbs2, body = loop ?ident_label body in
      let all_bindings = (Map.data @@ reduce_vbss vbss1) @ bindings @ Map.data vbs2 in
      (no_vbs, { expr with pexp_desc = Pexp_let (recflag, all_bindings, body) })
  | { pexp_desc = Pexp_open (decl, body); _ } ->
      let vbs, body = loop ?ident_label body in
      (vbs, { expr with pexp_desc = Pexp_open (decl, body) })
  | { pexp_desc = Pexp_letmodule (name, module_expr, body); _ } ->
      let vbs, body = loop ?ident_label body in
      (vbs, { expr with pexp_desc = Pexp_letmodule (name, module_expr, body) })
  | { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ } when is_operator op_ident ->
      (no_vbs, [%expr [%e expr] ~label:[%e opt_pat2string_list ~loc ident_label]])
  | expr -> (no_vbs, expr)

let translate ?ident_label expr =
  let vbs, expr = translate ~has_config:false ?ident_label expr in
  let loc = expr.pexp_loc in
  ( vbs,
    match ident_label with
    | Some [%pat? _] ->
        [%expr
          Tensor.with_unchanged_roots ~f:(fun () ->
              let open! TDSL.O in
              [%e expr])]
    | _ ->
        [%expr
          let open! TDSL.O in
          [%e expr]] )

let expr_expander ~loc ~path = expr_expander_with_punning translate ~loc ~path
let str_expander ~loc ~path = str_expander_with_punning translate ~loc ~path
