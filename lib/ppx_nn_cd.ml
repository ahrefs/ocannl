open Base

open Ppxlib

open Ppx_nn_shared

let ndarray_op ?axis_labels ?label expr =
  let loc = expr.pexp_loc in
  let values, batch_dims, output_dims, input_dims = ndarray_constant expr in
  let edims dims = Ast_builder.Default.elist ~loc @@ List.rev dims in
  let op =
    match axis_labels, label with
    | None, None -> [%expr Formula.NFDSL.ndarray]
    | Some axis_labels, None -> [%expr Formula.NFDSL.ndarray ?axis_labels:[%e axis_labels]]
    | None, Some label -> [%expr Formula.NFDSL.ndarray ?label:[%e label]]
    | Some axis_labels, Some label ->
      [%expr Formula.NFDSL.ndarray ?axis_labels:[%e axis_labels] ?label:[%e label]] in
  [%expr
    [%e op] ~batch_dims:[%e edims batch_dims] ~input_dims:[%e edims input_dims]
      ~output_dims:[%e edims output_dims] [%e values]]

type expr_type = Code | Formula | Node | Data | Data_grad | Operator | Unknown [@@deriving equal]

let assignment_op expr =
  let loc = expr.pexp_loc in
  match expr with
  | [%expr (=:)] -> [%expr Code.Skip_arg]
  | [%expr (=+)] -> [%expr Code.Add]
  | [%expr (=*)] -> [%expr Code.Mul]
  | [%expr (=**)] -> [%expr Code.ToPowOf]
  | [%expr (=?/)] -> [%expr Code.Relu_gate]
  | _ ->
    Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl %%nn_cd: expected an assignment operator, one of: %s"
      "=: (Skip_arg), =+ (Add), =* (Mul), =** (ToPowOf), =?/ (Relu_gate)"

let binary_op expr =
  let loc = expr.pexp_loc in
  match expr with
  | [%expr (+)] -> [%expr Code.Add]
  | [%expr ( * )] -> [%expr Code.Mul]
  | [%expr ( ** )] -> [%expr Code.ToPowOf]
  | [%expr (-?/)] -> [%expr Code.Relu_gate]
  | [%expr (-/>)] -> [%expr Code.Skip_arg]
  | _ ->
    Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl %%nn_cd: expected a binary operator, one of: %s"
      "+ (Add), * (Mul), ** (ToPowOf), -?/ (Relu_gate), -/> (Skip_arg)"

let unary_op expr =
  let loc = expr.pexp_loc in
  match expr with
  | [%expr (~=)] -> [%expr Code.Identity]
  | [%expr (!/)] -> [%expr Code.Relu]
  | _ ->
    Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl %%nn_cd: expected a unary operator, one of: = (Identity), !/ (Relu)"

let rec translate expr =
  let loc = expr.pexp_loc in
  match expr with
  | { pexp_desc = Pexp_constant (Pconst_float _); _ } ->
    Formula, [%expr Formula.NFDSL.number [%e expr]]

  | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
    Formula, [%expr Formula.NFDSL.number (Float.of_int [%e expr])]

  | [%expr [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
      [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
    let axis = Ast_helper.Exp.constant ~loc:pexp_loc
        (Pconst_string (String.of_char ch, pexp_loc, None)) in
    Formula, [%expr Formula.NFDSL.number ~axis_label:[%e axis] [%e f]]

  | [%expr [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
      [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
        let axis = Ast_helper.Exp.constant ~loc:pexp_loc
        (Pconst_string (String.of_char ch, pexp_loc, None)) in
    Formula, [%expr Formula.NFDSL.number ~axis_label:[%e axis] (Float.of_int [%e i])]

  | { pexp_desc = Pexp_tuple _; _ } | { pexp_desc = Pexp_array _; _ } 
  | { pexp_desc = Pexp_construct ({txt=Lident "::"; _}, _); _ } ->
    Formula, ndarray_op expr

  | { pexp_desc = Pexp_ident {txt=Lident "n"; _}; _ }
  | { pexp_desc = Pexp_ident {txt=Lident "n1"; _}; _ }
  | { pexp_desc = Pexp_ident {txt=Lident "n2"; _}; _ } ->
    Node, expr

  | [%expr [%e? expr1] **.
      [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
    (* If converting code or a node to a formula was possible we would do it here.
       Since it's not, we let OCaml handle the type errors. Same further below. *)
    let _typ1, expr1 = translate expr1 in
    Formula, [%expr pointpow ~is_form [%e expr1] [%e f]]

  | [%expr [%e? expr1] **.
      [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
    let _typ1, expr1 = translate expr1 in
    Formula, [%expr pointpow ~is_form [%e expr1] (Float.of_int [%e i])]

  | [%expr [%e? expr1].value ] ->
    let typ1, expr1 = translate expr1 in
    let expr1 = match typ1 with
    | Formula -> [%expr DSL.value_of_id [%e expr1].node_id]
    | Node -> [%expr DSL.value_of_node [%e expr1]]
    | _ ->
      Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
        "ppx_ocannl %%nn_cd: the x.value syntax requires x to be a Node or a Formula" in
    Data, expr1

  | [%expr [%e? expr1].grad ] ->
    let typ1, expr1 = translate expr1 in
    let expr1 = match typ1 with
    | Formula -> [%expr DSL.grad_of_id [%e expr1].node_id]
    | Node -> [%expr DSL.grad_of_node [%e expr1]]
    | _ ->
      Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
        "ppx_ocannl %%nn_cd: the x.grad syntax requires x to be a Node or a Formula" in
    Data_grad, expr1

  | [%expr [%e? accu_op] [%e? lhs] ([%e? bin_op] [%e? rhs1] ([%e? rhs2] ~projections:[%e? projections])) ] ->
    let accu_op = assignment_op accu_op in
    let lhs_typ, lhs = translate lhs in
    let lhs = match lhs_typ with
      | Formula -> [%expr DSL.value_of_id [%e lhs].node_id]
      | Node -> [%expr DSL.value_of_node [%e lhs]]
      | _ -> lhs in
    let bin_op = binary_op bin_op in
    let rhs1_typ, rhs1 = translate rhs1 in
    let rhs1 = match rhs1_typ with
      | Formula -> [%expr DSL.value_of_id [%e rhs1].node_id]
      | Node -> [%expr DSL.value_of_node [%e rhs1]]
      | _ -> rhs1 in
    let rhs2_typ, rhs2 = translate rhs2 in
    let rhs2 = match rhs2_typ with
      | Formula -> [%expr DSL.value_of_id [%e rhs2].node_id]
      | Node -> [%expr DSL.value_of_node [%e rhs2]]
      | _ -> rhs2 in
    let guess_zero_out =
      if List.exists ~f:(equal_expr_type Data_grad) [lhs_typ; rhs1_typ; rhs2_typ]
      then [%expr false] else [%expr true] in
    Code, [%expr Code.Accum_binop {
      zero_out=[%e guess_zero_out]; accum=[%e accu_op]; lhs=[%e lhs];
      op=[%e bin_op]; rhs1=[%e rhs1]; rhs2=[%e rhs2]; projections=[%e projections]}
    ]

  | [%expr [%e? accu_op] [%e? lhs] (([%e? un_op] [%e? rhs]) ~projections:[%e? projections]) ]
  | [%expr [%e? accu_op] [%e? lhs] ([%e? un_op] ([%e? rhs] ~projections:[%e? projections])) ] ->
      (* Handle both un_op priority levels -- where application binds tighter and less tight. *)
    let accu_op = assignment_op accu_op in
    let lhs_typ, lhs = translate lhs in
    let lhs = match lhs_typ with
      | Formula -> [%expr DSL.value_of_id [%e lhs].node_id]
      | Node -> [%expr DSL.value_of_node [%e lhs]]
      | _ -> lhs in
    let un_op = unary_op un_op in
    let rhs_typ, rhs = translate rhs in
    let rhs = match rhs_typ with
      | Formula -> [%expr DSL.value_of_id [%e rhs].node_id]
      | Node -> [%expr DSL.value_of_node [%e rhs]]
      | _ -> rhs in
    let guess_zero_out =
      if List.exists ~f:(equal_expr_type Data_grad) [lhs_typ; rhs_typ]
      then [%expr false] else [%expr true] in
    Code, [%expr Code.Accum_unop {
        zero_out=[%e guess_zero_out]; accum=[%e accu_op]; lhs=[%e lhs];
        op=[%e un_op]; rhs=[%e rhs]; projections=[%e projections]}
    ]

  | [%expr [%e? accu_op] [%e? lhs] ([%e? rhs] ~projections:[%e? projections]) ] ->
    let accu_op = assignment_op accu_op in
    let lhs_typ, lhs = translate lhs in
    let lhs = match lhs_typ with
      | Formula -> [%expr DSL.value_of_id [%e lhs].node_id]
      | Node -> [%expr DSL.value_of_node [%e lhs]]
      | _ -> lhs in
    let rhs_typ, rhs = translate rhs in
    let rhs = match rhs_typ with
      | Formula -> [%expr DSL.value_of_id [%e rhs].node_id]
      | Node -> [%expr DSL.value_of_node [%e rhs]]
      | _ -> rhs in
    let guess_zero_out =
      if List.exists ~f:(equal_expr_type Data_grad) [lhs_typ; rhs_typ]
      then [%expr false] else [%expr true] in
    Code, [%expr Code.Accum_unop {
        zero_out=[%e guess_zero_out]; accum=[%e accu_op]; lhs=[%e lhs];
        op=Code.Identity; rhs=[%e rhs]; projections=[%e projections]}
    ]

  | [%expr [%e? expr1] [%e? expr2] [%e? expr3] ] ->
    let typ1, expr1 = translate expr1 in
    let _typ2, expr2 = translate expr2 in
    let _typ3, expr3 = translate expr3 in
    typ1, [%expr [%e expr1] [%e expr2] [%e expr3]]

  | [%expr [%e? expr1] [%e? expr2] ] ->
    let typ1, expr1 = translate expr1 in
    let _typ2, expr2 = translate expr2 in
    typ1, [%expr [%e expr1] [%e expr2]]

  | {pexp_desc=Pexp_fun (arg_label, arg, opt_val, body); _} as expr ->
    let typ, body = translate body in
    typ, {expr with pexp_desc=Pexp_fun (arg_label, arg, opt_val, body)}
 
  | [%expr while [%e? _test_expr] do [%e? _body] done ] ->
    Unknown,
    Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl %%nn_cd: while: low-level code embeddings not supported yet"

  | [%expr for [%p? _pat] = [%e? _init] to [%e? _final] do [%e? _body_expr] done ] ->
    Unknown,
    Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl %%nn_cd: for-to: low-level code embeddings not supported yet"

  | [%expr for [%p? _pat] = [%e? _init] downto [%e? _final] do [%e? _body_expr] done ] ->
    Unknown,
    Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl %%nn_cd: for-downto: low-level code embeddings not supported yet"

  | [%expr [%e? expr1] ; [%e? expr2] ] ->
    let typ1, expr1 = translate expr1 in
    let expr1 = match typ1 with Formula -> [%expr [%e expr1].forward_body] | _ -> expr1 in
    let typ2, expr2 = translate expr2 in
    let expr2 = match typ2 with Formula -> [%expr [%e expr2].forward_body] | _ -> expr2 in
    Code, [%expr Code.Seq ([%e expr1], [%e expr2])]

  | [%expr if [%e? expr1] then [%e? expr2] else [%e? expr3]] ->
    let _typ1, expr1 = translate expr1 in
    let typ2, expr2 = translate expr2 in
    let typ3, expr3 = translate expr3 in
    let typ = if equal_expr_type typ2 Unknown then typ3 else typ2 in
    typ, [%expr if [%e expr1] then [%e expr2] else [%e expr3]]

  | [%expr if [%e? expr1] then [%e? expr2]] ->
    let _typ1, expr1 = translate expr1 in
    let _typ2, expr2 = translate expr2 in
    Code, [%expr if [%e expr1] then [%e expr2] else Code.Noop]

  | { pexp_desc = Pexp_match (expr1, cases); _ } ->
    let typs, cases =
      List.unzip @@ List.map cases ~f:(fun ({pc_rhs; _} as c) ->
        let typ, pc_rhs = translate pc_rhs in typ, {c with pc_rhs}) in
    let typ = Option.value ~default:Unknown @@
      List.find typs ~f:(Fn.non @@ equal_expr_type Unknown) in
    typ, { expr with pexp_desc = Pexp_match (expr1, cases) }

  | { pexp_desc = Pexp_let (_recflag, _bindings, _body); _ } ->
    (* TODO(80): to properly support local bindings, we need to collect the type environment. *)
    Unknown,
    Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl %%nn_cd: for-to: local let-bindings not implemented yet"
     (* let bindings = List.map bindings
         ~f:(fun binding -> {binding with pvb_expr=translate binding.pvb_expr}) in
        {expr with pexp_desc=Pexp_let (recflag, bindings, translate body)} *)

  | { pexp_desc = Pexp_open (decl, body); _ } ->
    let typ, body = translate body in
    typ, {expr with pexp_desc=Pexp_open (decl, body)}

  | { pexp_desc = Pexp_letmodule (name, module_expr, body); _ } ->
    let typ, body = translate body in
    typ, {expr with pexp_desc=Pexp_letmodule (name, module_expr, body)}

  | _ -> Unknown, expr

let expr_expander ~loc ~path:_ payload =
  match payload with
  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
    (* We are at the %ocannl annotation level: do not tranlsate the body. *)
     let bindings = List.map bindings
      ~f:(fun vb -> {
             vb with pvb_expr=[%expr let open! DSL.O in [%e snd @@ translate vb.pvb_expr]]}) in
     {payload with pexp_desc=Pexp_let (recflag, bindings, body)}
  | expr -> snd @@ translate expr

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
  | Pstr_eval (expr, attrs) ->
    {str with pstr_desc=Pstr_eval (snd @@ translate expr, attrs)}
  | Pstr_value (recf, bindings) ->
    let f vb =
      let loc = vb.pvb_loc in
      {vb with pvb_expr=[%expr let open! DSL.O in [%e snd @@ translate vb.pvb_expr]]} in
    {str with pstr_desc=Pstr_value (recf, List.map bindings ~f)}
  | _ -> str
     
let str_expander ~loc ~path (payload: structure_item list) =
  flatten_str ~loc ~path @@ List.map payload ~f:translate_str
