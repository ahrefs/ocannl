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

let rec translate expr =
  let loc = expr.pexp_loc in
  match expr with
  | { pexp_desc = Pexp_constant (Pconst_float _); _ } ->
    [%expr Formula.NFDSL.number [%e expr]]

  | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
    [%expr Formula.NFDSL.number (Float.of_int [%e expr])]

  | [%expr [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
      [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
    let axis = Ast_helper.Exp.constant ~loc:pexp_loc
        (Pconst_string (String.of_char ch, pexp_loc, None)) in
    [%expr Formula.NFDSL.number ~axis_label:[%e axis] [%e f]]

  | [%expr [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
      [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
        let axis = Ast_helper.Exp.constant ~loc:pexp_loc
        (Pconst_string (String.of_char ch, pexp_loc, None)) in
    [%expr Formula.NFDSL.number ~axis_label:[%e axis] (Float.of_int [%e i])]

  | { pexp_desc = Pexp_tuple _; _ } | { pexp_desc = Pexp_array _; _ } 
  | { pexp_desc = Pexp_construct ({txt=Lident "::"; _}, _); _ } ->
    ndarray_op expr

  | [%expr [%e? expr1] **.
      [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
    [%expr pointpow ~is_form [%e translate expr1] [%e f]]

  | [%expr [%e? expr1] **.
      [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
    [%expr pointpow ~is_form [%e translate expr1] (Float.of_int [%e i])]

  | [%expr [%e? expr1] [%e? expr2] [%e? expr3] ] ->
    [%expr [%e translate expr1] [%e translate expr2] [%e translate expr3]]

  | [%expr [%e? expr1] [%e? expr2] ] ->
    [%expr [%e translate expr1] [%e translate expr2]]
  
    | {pexp_desc=Pexp_fun (arg_label, arg, opt_val, body); _} as expr ->
      {expr with pexp_desc=Pexp_fun (arg_label, arg, opt_val, translate body)}
 
  | [%expr while [%e? test_expr] do [%e? body_expr] done ] ->
    [%expr while [%e test_expr] do [%e translate body_expr] done ]

  | [%expr for [%p? pat] = [%e? init] to [%e? final] do [%e? body_expr] done ] ->
    [%expr for [%p pat] = [%e init] to [%e final] do [%e translate body_expr] done ]

  | [%expr for [%p? pat] = [%e? init] downto [%e? final] do [%e? body_expr] done ] ->
    [%expr for [%p pat] = [%e init] downto [%e final] do [%e translate body_expr] done ]

  | [%expr [%e? expr1] ; [%e? expr2] ] ->
    (* FIXME: use Seq *)
    [%expr [%e translate expr1] ; [%e translate expr2]]

  | [%expr if [%e? expr1] then [%e? expr2] else [%e? expr3]] ->
    [%expr if [%e expr1] then [%e translate expr2] else [%e translate expr3]]

  | [%expr if [%e? expr1] then [%e? expr2]] ->
    [%expr if [%e expr1] then [%e translate expr2] else Code.Noop]

  | { pexp_desc = Pexp_match (expr1, cases); _ } ->
    let cases =
      List.map cases ~f:(fun ({pc_rhs; _} as c) -> {c with pc_rhs=translate pc_rhs}) in
     { expr with pexp_desc = Pexp_match (expr1, cases) }

  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
     let bindings = List.map bindings
         ~f:(fun binding -> {binding with pvb_expr=translate binding.pvb_expr}) in
     {expr with pexp_desc=Pexp_let (recflag, bindings, translate body)}

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
     let bindings = List.map bindings
      ~f:(fun vb -> {vb with pvb_expr=[%expr let open! DSL.O in [%e translate vb.pvb_expr]]}) in
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
  | Pstr_eval (expr, attrs) ->
    {str with pstr_desc=Pstr_eval (translate expr, attrs)}
  | Pstr_value (recf, bindings) ->
    let f vb =
      let loc = vb.pvb_loc in
      {vb with pvb_expr=[%expr let open! DSL.O in [%e translate vb.pvb_expr]]} in
    {str with pstr_desc=Pstr_value (recf, List.map bindings ~f)}
  | _ -> str
     
let str_expander ~loc ~path (payload: structure_item list) =
  flatten_str ~loc ~path @@ List.map payload ~f:translate_str
