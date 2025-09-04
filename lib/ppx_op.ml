open Base
open Ppxlib
open Ppx_arrayjit.Ppx_helper
open Ppx_shared

let operators =
  (* TODO: Auto-generate this list from Operation.Make_DSL.O. *)
  Hashtbl.of_alist_exn
    (module String)
    [
      ("*", "matmul");
      ("*.", "pointmul");
      ("+", "add");
      ("threefry4x32", "threefry4x32");
      ("uint4x32_to_prec_uniform", "uint4x32_to_prec_uniform");
      ("uint4x32_to_prec_uniform1", "uint4x32_to_prec_uniform1");
      ("**.", "pointpow");
      ("relu", "relu");
      ("sat01", "sat01");
      ("fma", "fma");
      ("!.", "number");
      ("!..", "number_int");
      ("!%", "bits");
      ("!@", "embed_symbol");
      ("dim", "embed_dim");
      ("-", "sub");
      ("~-", "num_neg");
      ("/.", "pointdiv");
      ("@|", "slice");
      ("exp", "exp");
      ("log", "log");
      ("log2", "log2");
      ("sin", "sin");
      ("cos", "cos");
      ("neg", "neg");
      ("not", "not");
      ("sqrt", "sqrt");
      ("recip", "recip");
      ("recip_sqrt", "recip_sqrt");
      ("tanh", "tanh");
      ("where", "where");
      ("<", "lt");
      ("=", "eq");
      ("<>", "ne");
      ("embed_self_id", "embed_self_id");
      ("einsum", "einsum");
      ("einsum1", "einsum1");
      ("offsets", "offsets");
      ("uniform", "uniform");
      ("uniform_at", "uniform_at");
      ("uniform1", "uniform1");
      ("uniform_at1", "uniform_at1");
    ]

let add_module_qualifier_to_applied_function expr =
  let qualify_if_needed fn =
    match fn.pexp_desc with
    | Pexp_ident { txt = Lident name; loc } when Hashtbl.mem operators name ->
        Ast_builder.Default.pexp_ident ~loc
          { txt = Ldot (Lident "PDSL", Hashtbl.find_exn operators name); loc }
    | _ -> fn
  in
  let rec decompose_app expr acc =
    match expr.pexp_desc with
    | Pexp_apply (fn, args) -> decompose_app fn (args @ acc)
    | _ -> (expr, acc)
  in
  let rec process_expr expr =
    let loc = expr.pexp_loc in
    match expr.pexp_desc with
    | Pexp_apply (_, _) ->
        let fn, args = decompose_app expr [] in
        let qualified_fn = qualify_if_needed fn in
        let processed_args = List.map args ~f:(fun (label, arg) -> (label, process_expr arg)) in
        Ast_builder.Default.pexp_apply ~loc qualified_fn processed_args
    | Pexp_ifthenelse (cond, then_expr, else_expr) ->
        let processed_then = process_expr then_expr in
        let processed_else = Option.map else_expr ~f:process_expr in
        Ast_builder.Default.pexp_ifthenelse ~loc cond processed_then processed_else
    | Pexp_sequence (expr1, expr2) ->
        let processed_expr2 = process_expr expr2 in
        Ast_builder.Default.pexp_sequence ~loc expr1 processed_expr2
    | _ -> expr
  in
  process_expr expr

let make_p ~opt_label ~loc ?value ?values ?param_init ~extra_args name =
  let more_label =
    match opt_label with
    | Some label_pat -> [%expr Some [%e pat2expr label_pat]]
    | None -> [%expr None]
  in
  let value = match value with Some c -> [%expr Some [%e c]] | None -> [%expr None] in
  let values = match values with Some c -> [%expr Some [%e c]] | None -> [%expr None] in
  let param_init =
    match param_init with
    | Some c -> [%expr Some [%e add_module_qualifier_to_applied_function c]]
    | None -> [%expr None]
  in
  let extra_args =
    List.map extra_args ~f:(fun (label, value) ->
        match label.txt with
        | Lident "o" -> (Labelled "output_dims", value)
        | Lident "i" -> (Labelled "input_dims", value)
        | Lident "b" -> (Labelled "batch_dims", value)
        | Lident arg_name -> (Labelled arg_name, value)
        | _ ->
            ( Labelled "syntax_error",
              Ast_builder.Default.pexp_extension ~loc:label.loc
              @@ Location.error_extensionf ~loc:label.loc
                   "inline-definition fields must be simple identifiers" ))
  in
  let name = Ast_helper.Exp.constant ~loc (Pconst_string (name.txt, name.loc, None)) in
  let base_expr =
    [%expr
      TDSL.param ?more_label:[%e more_label] ?value:[%e value] ?values:[%e values]
        ?param_init:[%e param_init] [%e name]]
  in
  let with_extra_args =
    if List.is_empty extra_args then base_expr else Ast_helper.Exp.apply ~loc base_expr extra_args
  in
  [%expr [%e with_extra_args] ()]

let make_vb ~opt_label ?value ?param_init ~extra_args ~loc name =
  let pat = Ast_helper.Pat.var ~loc:name.loc name in
  let v = make_p ~opt_label ~loc ?value ?param_init ~extra_args name in
  let vb = Ast_helper.Vb.mk ~loc pat v in
  (pat, vb)

let make_vb_nd ~opt_label ~init_nd ~extra_args ~loc name =
  let pat = Ast_helper.Pat.var ~loc:name.loc name in
  let values, batch_dims, output_dims, input_dims = ndarray_constant init_nd in
  let v =
    if not @@ List.is_empty batch_dims then
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "ppx_ocannl param cannot have batch dims: define a constant or remove the array syntax."
    else
      let edims dims = Ast_builder.Default.elist ~loc dims in
      let input_dims_expr = edims input_dims in
      let output_dims_expr = edims output_dims in
      let extra_args =
        ({ txt = Lident "input_dims"; loc }, input_dims_expr)
        :: ({ txt = Lident "output_dims"; loc }, output_dims_expr)
        :: extra_args
      in
      make_p ~opt_label ~loc ~values ~extra_args name
  in
  let vb = Ast_helper.Vb.mk ~loc pat v in
  (pat, vb)

let rec translate ~num_configs ~is_toplevel ~opt_label ?label expr =
  let loc = expr.pexp_loc in
  let loop = translate ~num_configs ~is_toplevel:false ~opt_label in
  match expr with
  | { pexp_desc = Pexp_constant (Pconst_float _); _ } ->
      (no_vbs, [%expr TDSL.number ?label:[%e opt_expr ~loc label] [%e expr]])
  | { pexp_desc = Pexp_constant (Pconst_integer (_, Some ('L' | 'l'))); _ } ->
      (no_vbs, [%expr TDSL.bits [%e expr]])
  | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
      (no_vbs, [%expr TDSL.number (Float.of_int [%e expr])])
  | [%expr
      [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
        [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
      let axis =
        Ast_helper.Exp.constant ~loc:pexp_loc (Pconst_string (String.of_char ch, pexp_loc, None))
      in
      (no_vbs, [%expr TDSL.number ?label:[%e opt_expr ~loc label] ~axis_label:[%e axis] [%e f]])
  | [%expr
      [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
        [%e? { pexp_desc = Pexp_constant (Pconst_integer (_, Some ('L' | 'l'))); _ } as i]] ->
      let axis =
        Ast_helper.Exp.constant ~loc:pexp_loc (Pconst_string (String.of_char ch, pexp_loc, None))
      in
      (no_vbs, [%expr TDSL.bits ?label:[%e opt_expr ~loc label] ~axis_label:[%e axis] [%e i]])
  | [%expr
      [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
        [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
      let axis =
        Ast_helper.Exp.constant ~loc:pexp_loc (Pconst_string (String.of_char ch, pexp_loc, None))
      in
      ( no_vbs,
        [%expr
          TDSL.number ?label:[%e opt_expr ~loc label] ~axis_label:[%e axis] (Float.of_int [%e i])]
      )
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
        [%e? expr1]
        ([%e? { pexp_desc = Pexp_ident _; _ } as spec] [%e? expr2])]
    when Hashtbl.mem einsum_binary_ops op_ident ->
      let vbs1, e1 = loop expr1 in
      let vbs2, e2 = loop expr2 in
      ( reduce_vbss [ vbs1; vbs2 ],
        [%expr
          [%e Hashtbl.find_exn einsum_binary_ops op_ident loc]
            ?label:[%e opt_expr ~loc label] [%e spec] [%e e1] [%e e2]] )
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
        [%e? expr1]
        ([%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }] [%e? expr2])]
    when String.contains spec_str '>' && Hashtbl.mem einsum_binary_ops op_ident ->
      let vbs1, e1 = loop expr1 in
      let vbs2, e2 = loop expr2 in
      let spec = substitute_identifiers_in_einsum_spec ~loc spec_str in
      ( reduce_vbss [ vbs1; vbs2 ],
        [%expr
          [%e Hashtbl.find_exn einsum_binary_ops op_ident loc]
            ?label:[%e opt_expr ~loc label] [%e spec] [%e e1] [%e e2]] )
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
        [%e? expr1]
        [%e? { pexp_desc = Pexp_ident _; _ } as spec]]
    when Hashtbl.mem einsum_unary_ops op_ident ->
      let vbs1, e1 = loop expr1 in
      ( vbs1,
        [%expr
          [%e Hashtbl.find_exn einsum_unary_ops op_ident loc]
            ?label:[%e opt_expr ~loc label] [%e spec] [%e e1]] )
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
        [%e? expr1]
        [%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }]]
    when String.contains spec_str '>' && Hashtbl.mem einsum_unary_ops op_ident ->
      let vbs1, e1 = loop expr1 in
      let spec = substitute_identifiers_in_einsum_spec ~loc spec_str in
      ( vbs1,
        [%expr
          [%e Hashtbl.find_exn einsum_unary_ops op_ident loc]
            ?label:[%e opt_expr ~loc label] [%e spec] [%e e1]] )
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
        [%e? expr1]
        ([%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }]
           ([%e? { pexp_desc = Pexp_constant (Pconst_string _); _ } as head] :: [%e? rest])
           [%e? expr2])]
    when String.contains spec_str '>' && Hashtbl.mem einsum_binary_ops op_ident ->
      let capture_vbs, capture_dims_expr = collect_capture_labels ~loc head rest in
      let vbs1, e1 = loop expr1 in
      let vbs2, e2 = loop expr2 in
      let spec = substitute_identifiers_in_einsum_spec ~loc spec_str in
      let combined_vbs = reduce_vbss [ vbs1; vbs2; capture_vbs ] in
      ( combined_vbs,
        [%expr
          [%e Hashtbl.find_exn einsum_binary_ops op_ident loc]
            ?label:[%e opt_expr ~loc label] ~capture_dims:[%e capture_dims_expr] [%e spec] [%e e1]
            [%e e2]] )
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
        [%e? expr1]
        ([%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }]
           ([%e? { pexp_desc = Pexp_constant (Pconst_string _); _ } as head] :: [%e? rest]))]
    when String.contains spec_str '>' && Hashtbl.mem einsum_unary_ops op_ident ->
      let capture_vbs, capture_dims_expr = collect_capture_labels ~loc head rest in
      let vbs1, e1 = loop expr1 in
      let spec = substitute_identifiers_in_einsum_spec ~loc spec_str in
      let combined_vbs = reduce_vbss [ vbs1; capture_vbs ] in
      ( combined_vbs,
        [%expr
          [%e Hashtbl.find_exn einsum_unary_ops op_ident loc]
            ?label:[%e opt_expr ~loc label] ~capture_dims:[%e capture_dims_expr] [%e spec] [%e e1]]
      )
  | { pexp_desc = Pexp_record ([], _); _ } ->
      (* Empty record - not a tensor definition *)
      (no_vbs, expr)
  | {
   pexp_desc =
     Pexp_record
       ( (first_label, ({ pexp_desc = Pexp_constant (Pconst_float _); _ } as value)) :: extra_args,
         None );
   _;
  } -> (
      match first_label.txt with
      | Lident tensor_name ->
          let name = { loc = first_label.loc; txt = tensor_name } in
          let pat, vb = make_vb ~opt_label ~value ~extra_args ~loc name in
          (Map.singleton (module String) tensor_name vb, pat2expr pat)
      | _ ->
          ( no_vbs,
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%op: record field label must be a simple identifier" ))
  | {
   pexp_desc =
     Pexp_record
       ( (first_label, ({ pexp_desc = Pexp_constant (Pconst_integer (_, None)); _ } as int_val))
         :: extra_args,
         None );
   _;
  } -> (
      match first_label.txt with
      | Lident tensor_name ->
          let value = [%expr Float.of_int [%e int_val]] in
          let name = { loc = first_label.loc; txt = tensor_name } in
          let pat, vb = make_vb ~opt_label ~value ~extra_args ~loc name in
          (Map.singleton (module String) tensor_name vb, pat2expr pat)
      | _ ->
          ( no_vbs,
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%op: record field label must be a simple identifier" ))
  | {
   pexp_desc =
     Pexp_record
       ( ( first_label,
           (( { pexp_desc = Pexp_array _; _ }
            | { pexp_desc = Pexp_construct ({ txt = Lident "::"; _ }, _); _ } ) as init_nd) )
         :: extra_args,
         None );
   _;
  } -> (
      (* Record syntax with array/list initialization *)
      match first_label.txt with
      | Lident tensor_name ->
          let name = { loc = first_label.loc; txt = tensor_name } in
          let pat, vb = make_vb_nd ~opt_label ~init_nd ~extra_args ~loc name in
          (* Note: expect a type error if batch_dims exist or extra_args modify the shape *)
          (Map.singleton (module String) tensor_name vb, pat2expr pat)
      | _ ->
          ( no_vbs,
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%op: record field label must be a simple identifier" ))
  | { pexp_desc = Pexp_record ((first_label, first_value) :: extra_args, None); _ } -> (
      (* Record syntax for tensor definitions *)
      match first_label.txt with
      | Lident tensor_name ->
          (* Process the initialization expression *)
          let init_vbs, param_init =
            match first_value with
            | { pexp_desc = Pexp_ident { txt = Lident val_ident; _ }; _ }
              when String.equal val_ident tensor_name ->
                (no_vbs, None)
            | _ ->
                let vbs, e = loop first_value in
                (vbs, Some e)
          in
          let name = { loc = first_label.loc; txt = tensor_name } in
          let pat, vb = make_vb ~opt_label ?param_init ~extra_args ~loc name in
          (* Combine with any bindings from the initialization *)
          let all_vbs = Map.add_exn init_vbs ~key:tensor_name ~data:vb in
          (all_vbs, pat2expr pat)
      | _ ->
          ( no_vbs,
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%op: record field label must be a simple identifier" ))
  | { pexp_desc = Pexp_array _; _ }
  | { pexp_desc = Pexp_construct ({ txt = Lident "::"; _ }, _); _ } ->
      (no_vbs, ndarray_op ?label ~ndarray_fn:[%expr TDSL.ndarray] expr)
  | [%expr !.[%e? expr1]] ->
      (* Hardcoding the patterns for (!.), (!..), and ( **. ) to avoid treating the constants as
         already tensors. *)
      (no_vbs, [%expr TDSL.O.( !. ) [%e expr1]])
  | [%expr !..[%e? expr1]] -> (no_vbs, [%expr TDSL.O.( !.. ) [%e expr1]])
  | [%expr [%e? expr1] **. [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
      let vbs, e1 = loop expr1 in
      (vbs, [%expr TDSL.O.( **. ) ?label:[%e opt_expr ~loc label] [%e e1] (Float.of_int [%e i])])
  | [%expr [%e? expr1] **. [%e? expr2]] ->
      let vbs, e1 = loop expr1 in
      (vbs, [%expr TDSL.O.( **. ) ?label:[%e opt_expr ~loc label] [%e e1] [%e expr2]])
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }] ([%e? expr2], [%e? expr3])]
    when Hashtbl.mem binary_ops op_ident ->
      let e1 = [%expr [%e expr] ?label:[%e opt_expr ~loc label]] in
      let vbs2, e2 = loop expr2 in
      let vbs3, e3 = loop expr3 in
      (reduce_vbss [ vbs2; vbs3 ], [%expr [%e e1] [%e e2] [%e e3]])
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
        ([%e? expr2], [%e? expr3], [%e? expr4])]
    when Hashtbl.mem ternary_ops op_ident ->
      let e1 = [%expr [%e expr] ?label:[%e opt_expr ~loc label]] in
      let vbs2, e2 = loop expr2 in
      let vbs3, e3 = loop expr3 in
      let vbs4, e4 = loop expr4 in
      (reduce_vbss [ vbs2; vbs3; vbs4 ], [%expr [%e e1] [%e e2] [%e e3] [%e e4]])
  | [%expr [%e? expr1] [%e? expr2] [%e? expr3]] ->
      let vbs1, e1 = loop ?label expr1 in
      let vbs2, e2 = loop expr2 in
      let vbs3, e3 = loop expr3 in
      (reduce_vbss [ vbs1; vbs2; vbs3 ], [%expr [%e e1] [%e e2] [%e e3]])
  | [%expr [%e? expr1] [%e? expr2]] ->
      let vbs1, e1 = loop ?label expr1 in
      let vbs2, e2 = loop expr2 in
      (reduce_vbss [ vbs1; vbs2 ], [%expr [%e e1] [%e e2]])
  | { pexp_desc = Pexp_function (args, constr, body); _ } when is_toplevel -> (
      (* Check if there's a unit parameter or a labeled parameter with label "label" *)
      let rec find_unit acc = function
        | [] -> None
        | ({
             pparam_desc =
               Pparam_val
                 (Nolabel, _, { ppat_desc = Ppat_construct ({ txt = Lident "()"; _ }, None); _ });
             _;
           } as unit_param)
          :: rest ->
            Some (List.rev acc, unit_param, rest)
        | hd :: rest -> find_unit (hd :: acc) rest
      in
      let rec find_label_param = function
        | [] -> None
        | { pparam_desc = Pparam_val (Labelled "label", _, pat); _ } :: _ -> Some pat
        | _ :: rest -> find_label_param rest
      in
      match find_unit [] args with
      | Some (before_unit, unit_param, after_unit) ->
          (* With a unit parameter, always bind the collected inline definitions. *)
          let opt_label = find_label_param before_unit in
          let vbs, inner_body =
            let body =
              match (after_unit, body) with
              | [], Pfunction_body body -> body
              | _ -> { expr with pexp_desc = Pexp_function (after_unit, constr, body) }
            in
            translate ~num_configs ~is_toplevel:false ~opt_label ?label body
          in
          let inner_body = let_opt ~loc vbs inner_body in
          ( no_vbs,
            {
              expr with
              pexp_desc =
                Pexp_function (before_unit @ [ unit_param ], constr, Pfunction_body inner_body);
            } )
      | None ->
          (* No unit parameter, everything is "after_unit" *)
          let labels =
            Option.to_list label
            @ List.filter_map args ~f:(function
                | { pparam_desc = Pparam_val (_, _, pat); _ } ->
                    let loc = pat.ppat_loc in
                    Some [%expr [%e pat2expr pat].Tensor.value.Ir.Tnode.label]
                | _ -> None)
          in
          let label_locs = List.map labels ~f:(fun label -> label.pexp_loc) in
          let label_starts = List.map label_locs ~f:(fun l -> l.loc_start) in
          let label_ends = List.map label_locs ~f:(fun l -> l.loc_end) in
          let label_loc =
            if List.is_empty labels then loc
            else
              Location.
                {
                  loc_start = List.reduce_exn label_starts ~f:min_pos;
                  loc_end = List.reduce_exn label_ends ~f:max_pos;
                  loc_ghost = false;
                }
          in
          let label =
            let loc = label_loc in
            [%expr List.concat [%e Ast_builder.Default.elist ~loc labels]]
          in
          let vbs, body =
            match body with
            | Pfunction_body body ->
                let vbs, body = loop ~label body in
                (vbs, Pfunction_body body)
            | Pfunction_cases (cases, loc, attrs) ->
                let vbs, cases =
                  List.unzip
                  @@ List.map cases ~f:(fun ({ pc_rhs; _ } as c) ->
                         let vbs, pc_rhs = loop ~label pc_rhs in
                         (vbs, { c with pc_rhs }))
                in
                ( List.fold vbs
                    ~init:(Map.empty (module String))
                    ~f:(fun acc vbs -> Map.merge_disjoint_exn acc vbs),
                  Pfunction_cases (cases, loc, attrs) )
          in
          (vbs, { expr with pexp_desc = Pexp_function (args, constr, body) }))
  | { pexp_desc = Pexp_function (args, constr, body); _ } ->
      let vbs, body =
        match body with
        | Pfunction_body body ->
            let vbs, body = loop ?label body in
            (vbs, Pfunction_body body)
        | Pfunction_cases (cases, loc, attrs) ->
            let vbs, cases =
              List.unzip
              @@ List.map cases ~f:(fun ({ pc_rhs; _ } as c) ->
                     let vbs, pc_rhs = loop ?label pc_rhs in
                     (vbs, { c with pc_rhs }))
            in
            ( List.fold vbs
                ~init:(Map.empty (module String))
                ~f:(fun acc vbs -> Map.merge_disjoint_exn acc vbs),
              Pfunction_cases (cases, loc, attrs) )
      in
      (vbs, { expr with pexp_desc = Pexp_function (args, constr, body) })
  | [%expr
      while [%e? test_expr] do
        [%e? body_expr]
      done] ->
      let vbs, body = loop ?label body_expr in
      ( vbs,
        [%expr
          while [%e test_expr] do
            [%e body]
          done] )
  | [%expr
      for [%p? pat] = [%e? init] to [%e? final] do
        [%e? body_expr]
      done] ->
      let vbs, body = loop ?label body_expr in
      ( vbs,
        [%expr
          for [%p pat] = [%e init] to [%e final] do
            [%e body]
          done] )
  | [%expr
      for [%p? pat] = [%e? init] downto [%e? final] do
        [%e? body_expr]
      done] ->
      let vbs, body = loop ?label body_expr in
      ( vbs,
        [%expr
          for [%p pat] = [%e init] downto [%e final] do
            [%e body]
          done] )
  | [%expr
      [%e? expr1];
      [%e? expr2]] ->
      let vbs1, e1 = loop expr1 in
      let vbs2, e2 = loop ?label expr2 in
      ( reduce_vbss [ vbs1; vbs2 ],
        [%expr
          [%e e1];
          [%e e2]] )
  | [%expr if [%e? expr1] then [%e? expr2] else [%e? expr3]] ->
      let vbs2, e2 = loop ?label expr2 in
      let vbs3, e3 = loop ?label expr3 in
      (reduce_vbss [ vbs2; vbs3 ], [%expr if [%e expr1] then [%e e2] else [%e e3]])
  | [%expr if [%e? expr1] then [%e? expr2]] ->
      let vbs2, e2 = loop ?label expr2 in
      (vbs2, [%expr if [%e expr1] then [%e e2]])
  | { pexp_desc = Pexp_match (expr1, cases); _ } ->
      let vbss, cases =
        List.unzip
        @@ List.map cases ~f:(fun ({ pc_rhs; _ } as c) ->
               let vbs, pc_rhs = loop ?label pc_rhs in
               (vbs, { c with pc_rhs }))
      in
      (reduce_vbss vbss, { expr with pexp_desc = Pexp_match (expr1, cases) })
  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
      let vbss1, bindings =
        List.unzip
        @@ List.map bindings ~f:(fun binding ->
               let vbs, pvb_expr =
                 loop ~label:[%expr [ [%e pat2string binding.pvb_pat] ]] binding.pvb_expr
               in
               (vbs, { binding with pvb_expr }))
      in
      let vbs2, body = loop ?label body in
      (reduce_vbss (vbss1 @ [ vbs2 ]), { expr with pexp_desc = Pexp_let (recflag, bindings, body) })
  | { pexp_desc = Pexp_open (decl, body); _ } ->
      let vbs, body = loop ?label body in
      (vbs, { expr with pexp_desc = Pexp_open (decl, body) })
  | { pexp_desc = Pexp_letmodule (name, module_expr, body); _ } ->
      let vbs, body = loop ?label body in
      (vbs, { expr with pexp_desc = Pexp_letmodule (name, module_expr, body) })
  | { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }
    when is_primitive_op op_ident || is_operator op_ident ->
      (* FIXME: this heuristic is hacky... *)
      (no_vbs, [%expr [%e expr] ?label:[%e opt_expr ~loc label]])
  | expr -> (no_vbs, expr)

let translate ?ident_label expr =
  let vbs, expr =
    translate ~num_configs:(ref 0) ~is_toplevel:true ~opt_label:None
      ~label:(opt_pat2string_list ~loc:expr.pexp_loc ident_label)
      expr
  in
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
