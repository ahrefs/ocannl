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

type expr_type =
  | Code
  | Formula_nf
  | Formula_or_node_or_data
  | Grad_of_source of expression
  | Unknown

let is_grad = function Grad_of_source _ -> true | _ -> false
let is_unknown = function Unknown -> true | _ -> false

type projections_slot = LHS | RHS1 | RHS2 | Nonslot | Undet [@@deriving equal, sexp]

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

let alphanum_regexp = Str.regexp "^[^a-zA-Z0-9]+$"
let is_operator ident = Str.string_match alphanum_regexp ident 0
let is_assignment ident =
  String.length ident > 1 && Char.equal ident.[0] '=' &&
  not @@ List.mem ["=="; "==="; "=>"; "==>"; "=>>"] ident ~equal:String.equal

let setup_data hs_pat (hs_typ, slot, hs) =
  let loc = hs.pexp_loc in
  match hs_typ with
  | Formula_nf ->
    Some (hs_pat, hs, [%expr [%e pat2expr hs_pat].forward_body]),
    hs_typ, slot, [%expr CDSL.value_of_id [%e pat2expr hs_pat].id]
  | Formula_or_node_or_data -> None, hs_typ, slot, [%expr CDSL.value_of_id [%e hs].id]
  | _ -> None, hs_typ, slot, hs

let with_forward_args setups body =
  let loc = body.pexp_loc in
  let bindings = List.map setups ~f:(fun (pat, v, _) -> Ast_helper.Vb.mk ~loc pat v) in
  let forward_args =
    List.map setups ~f:(fun (_, _, fwd) -> fwd) |>
    List.reduce ~f:(fun code fwd -> [%expr Code.Par ([%e code], [%e fwd])]) in
  Code, Nonslot, (match forward_args with
      | None -> body
      | Some fwd ->
        Ast_helper.Exp.let_ ~loc Nonrecursive bindings [%expr Code.Seq([%e fwd], [%e body])])

let project_xhs debug loc slot = match slot with
  | LHS -> [%expr p.project_lhs]
  | RHS1 -> [%expr p.project_rhs1]
  | RHS2 -> [%expr Option.value_exn p.project_rhs2]
  | Nonslot ->
    Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl %%nn_cd: not a valid accumulation/assignment slot filler at %s" debug
  | Undet ->
    Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl %%nn_cd: insufficient slot filler information at %s %s" debug
      "(incorporate one of: n, n1, n2, m1, m2, lhs, rhs, rhs1, rhs2)"

let rec translate (expr: expression): expr_type * projections_slot * expression =
  let loc = expr.pexp_loc in
  match expr with
  | { pexp_desc = Pexp_constant (Pconst_float _); _ } ->
    Formula_nf, Undet, [%expr Formula.NFDSL.number [%e expr]]

  | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
    Formula_nf, Undet, [%expr Formula.NFDSL.number (Float.of_int [%e expr])]

  | [%expr [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
      [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
    let axis = Ast_helper.Exp.constant ~loc:pexp_loc
        (Pconst_string (String.of_char ch, pexp_loc, None)) in
    Formula_nf, Undet, [%expr Formula.NFDSL.number ~axis_label:[%e axis] [%e f]]

  | [%expr [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
      [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
        let axis = Ast_helper.Exp.constant ~loc:pexp_loc
        (Pconst_string (String.of_char ch, pexp_loc, None)) in
    Formula_nf, Undet, [%expr Formula.NFDSL.number ~axis_label:[%e axis] (Float.of_int [%e i])]

  | { pexp_desc = Pexp_tuple _; _ } | { pexp_desc = Pexp_array _; _ } 
  | { pexp_desc = Pexp_construct ({txt=Lident "::"; _}, _); _ } ->
    Formula_nf, Undet, ndarray_op expr

  | { pexp_desc = Pexp_ident {txt=Lident ("n" | "lhs"); _}; _ } ->
    Formula_or_node_or_data, LHS, expr

  | { pexp_desc = Pexp_ident {txt=Lident ("n1" | "m1" | "rhs1" | "rhs"); _}; _ } ->
    (* [m1], [m2] have their forward code included by [Formula.binop/unop] *)
    Formula_or_node_or_data, RHS1, expr

  | { pexp_desc = Pexp_ident {txt=Lident ("n2" | "m2" | "rhs2"); _}; _ } ->
    Formula_or_node_or_data, RHS2, expr

  | { pexp_desc = Pexp_ident {txt=Lident op_ident; _}; _ } when is_operator op_ident ->
    Formula_nf, Undet, expr

  | [%expr [%e? expr1] || [%e? expr2] ] ->
    (* Check this before the generic application pattern. *)
    let _typ1, _slot1, expr1 = translate expr1 in
    let _typ2, _slot2, expr2 = translate expr2 in
    (* We could warn if typ is not Code and slot is not Nonslot, but that could be annoying. *)
    Code, Nonslot, [%expr Code.ParHint ([%e expr1], [%e expr2])]

  | [%expr [%e? expr1] **. [%e? expr2]] ->
    (* If converting code or a node to a formula was possible we would do it here.
       Since it's not, we let OCaml handle the type errors. Same further below. *)
    let _typ1, slot1, expr1 = translate expr1 in
    Formula_nf, slot1, [%expr pointpow ~is_form:false [%e expr2] [%e expr1]]

  | [%expr [%e? expr1].grad ] ->
    let typ1, slot1, expr1 = translate expr1 in
    let expr1 = match typ1 with
    | Formula_or_node_or_data -> [%expr CDSL.grad_of_id [%e expr1].id]
    | _ ->
      Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
        "ppx_ocannl %%nn_cd: the x.grad syntax requires x to be Node.t, Node.data or Formula.t" in
    Grad_of_source expr1, slot1, expr1

  | [%expr [%e? accu_op] [%e? lhs] ([%e? bin_op] [%e? rhs1] ([%e? rhs2] ~projections:[%e? projections])) ] ->
    let accu_op = assignment_op accu_op in
    let lhs_setup, lhs_typ, _lhs_slot, lhs =
      setup_data [%pat? nonform___lhs] @@ translate lhs in
    let bin_op = binary_op bin_op in
    let rhs1_setup, rhs1_typ, _rhs1_slot, rhs1 =
      setup_data [%pat? nonform___rhs1] @@ translate rhs1 in
    let rhs2_setup, rhs2_typ, _rhs2_slot, rhs2 =
      setup_data [%pat? nonform___rhs2] @@ translate rhs2 in
    let guess_zero_out =
      if List.exists ~f:is_grad [lhs_typ; rhs1_typ; rhs2_typ]
      then [%expr false] else [%expr true] in
    let body = [%expr Code.Accum_binop {
      zero_out=[%e guess_zero_out]; accum=[%e accu_op]; lhs=[%e lhs];
      op=[%e bin_op]; rhs1=[%e rhs1]; rhs2=[%e rhs2]; projections=[%e projections]}
    ] in
    let setups = List.filter_map ~f:Fn.id [lhs_setup; rhs1_setup; rhs2_setup] in
    with_forward_args setups body

  | [%expr [%e? accu_op] [%e? lhs] (([%e? un_op] [%e? rhs]) ~projections:[%e? projections]) ]
  | [%expr [%e? accu_op] [%e? lhs] ([%e? un_op] ([%e? rhs] ~projections:[%e? projections])) ] ->
      (* Handle both un_op priority levels -- where application binds tighter and less tight. *)
    let accu_op = assignment_op accu_op in
    let lhs_setup, lhs_typ, _lhs_slot, lhs = setup_data [%pat? nonform___lhs] @@ translate lhs in
    let un_op = unary_op un_op in
    let rhs_setup, rhs_typ, _rhs_slot, rhs = setup_data [%pat? nonform___rhs] @@ translate rhs in
    let guess_zero_out =
      if List.exists ~f:is_grad [lhs_typ; rhs_typ]
      then [%expr false] else [%expr true] in
    let body = [%expr Code.Accum_unop {
        zero_out=[%e guess_zero_out]; accum=[%e accu_op]; lhs=[%e lhs];
        op=[%e un_op]; rhs=[%e rhs]; projections=[%e projections]}
    ] in
    let setups = List.filter_map ~f:Fn.id [lhs_setup; rhs_setup] in
    with_forward_args setups body

  | [%expr [%e? accu_op] [%e? lhs] ([%e? rhs] ~projections:[%e? projections]) ] ->
    let accu_op = assignment_op accu_op in
    let lhs_setup, lhs_typ, _lhs_slot, lhs = setup_data [%pat? nonform___lhs] @@ translate lhs in
    let rhs_setup, rhs_typ, _rhs_slot, rhs = setup_data [%pat? nonform___rhs] @@ translate rhs in
    let guess_zero_out =
      if List.exists ~f:is_grad [lhs_typ; rhs_typ]
      then [%expr false] else [%expr true] in
    let body = [%expr Code.Accum_unop {
        zero_out=[%e guess_zero_out]; accum=[%e accu_op]; lhs=[%e lhs];
        op=Code.Identity; rhs=[%e rhs]; projections=[%e projections]}
    ] in
    let setups = List.filter_map ~f:Fn.id [lhs_setup; rhs_setup] in
    with_forward_args setups body

  | [%expr [%e? { pexp_desc = Pexp_ident {txt=Lident op_ident; _}; _ } as accu_op]
      [%e? lhs] ([%e? bin_op] [%e? rhs1] ([%e? rhs2])) ] when is_assignment op_ident ->
    let accu_op = assignment_op accu_op in
    let lhs_setup, lhs_typ, lhs_slot, lhs =
      setup_data [%pat? nonform___lhs] @@ translate lhs in
    let bin_op = binary_op bin_op in
    let rhs1_setup, rhs1_typ, rhs1_slot, rhs1 =
      setup_data [%pat? nonform___rhs1] @@ translate rhs1 in
    let rhs2_setup, rhs2_typ, rhs2_slot, rhs2 =
      setup_data [%pat? nonform___rhs2] @@ translate rhs2 in
    let guess_zero_out =
      if List.exists ~f:is_grad [lhs_typ; rhs1_typ; rhs2_typ]
      then [%expr false] else [%expr true] in
    let projections =
      let project_lhs = project_xhs "LHS" lhs.pexp_loc lhs_slot in
      let project_rhs1 = project_xhs "RHS1" rhs1.pexp_loc rhs1_slot in
      let project_rhs2 = project_xhs "RHS2" rhs2.pexp_loc rhs2_slot in
      [%expr fun () -> let p = projections() in Shape.{
          p with project_lhs = [%e project_lhs];
                 project_rhs1 = [%e project_rhs1]; project_rhs2 = Some [%e project_rhs2] }]
    in
    let body = [%expr Code.Accum_binop {
      zero_out=[%e guess_zero_out]; accum=[%e accu_op]; lhs=[%e lhs];
      op=[%e bin_op]; rhs1=[%e rhs1]; rhs2=[%e rhs2]; projections=[%e projections]}
    ] in
    let setups = List.filter_map ~f:Fn.id [lhs_setup; rhs1_setup; rhs2_setup] in
    with_forward_args setups body

  | [%expr [%e? { pexp_desc = Pexp_ident {txt=Lident op_ident; _}; _ } as accu_op]
      [%e? lhs] ([%e? un_op] [%e? rhs]) ] when is_assignment op_ident ->
      (* Handle both un_op priority levels -- where application binds tighter and less tight. *)
    let accu_op = assignment_op accu_op in
    let lhs_setup, lhs_typ, lhs_slot, lhs = setup_data [%pat? nonform___lhs] @@ translate lhs in
    let un_op = unary_op un_op in
    let rhs_setup, rhs_typ, rhs_slot, rhs = setup_data [%pat? nonform___rhs] @@ translate rhs in
    let guess_zero_out =
      if List.exists ~f:is_grad [lhs_typ; rhs_typ]
      then [%expr false] else [%expr true] in
      let project_lhs = project_xhs "LHS" lhs.pexp_loc lhs_slot in
      let project_rhs1 = project_xhs "RHS1" rhs.pexp_loc rhs_slot in
      let projections =
        [%expr fun () -> let p = projections() in Shape.{
          p with project_lhs = [%e project_lhs];
                 project_rhs1 = [%e project_rhs1]; project_rhs2 = None }]
       in
      let body = [%expr Code.Accum_unop {
        zero_out=[%e guess_zero_out]; accum=[%e accu_op]; lhs=[%e lhs];
        op=[%e un_op]; rhs=[%e rhs]; projections=[%e projections]}
    ] in
    let setups = List.filter_map ~f:Fn.id [lhs_setup; rhs_setup] in
    with_forward_args setups body

  | [%expr [%e? { pexp_desc = Pexp_ident {txt=Lident op_ident; _}; _ } as accu_op]
      [%e? lhs] [%e? rhs] ] when is_assignment op_ident ->
      (* Handle both un_op priority levels -- where application binds tighter and less tight. *)
    let accu_op = assignment_op accu_op in
    let lhs_setup, lhs_typ, lhs_slot, lhs = setup_data [%pat? nonform___lhs] @@ translate lhs in
    let rhs_setup, rhs_typ, rhs_slot, rhs = setup_data [%pat? nonform___rhs] @@ translate rhs in
    let guess_zero_out =
      if List.exists ~f:is_grad [lhs_typ; rhs_typ]
      then [%expr false] else [%expr true] in
      let project_lhs = project_xhs "LHS" lhs.pexp_loc lhs_slot in
      let project_rhs1 = project_xhs "RHS1" rhs.pexp_loc rhs_slot in
      let projections =
        [%expr fun () -> let p = projections() in Shape.{
          p with project_lhs = [%e project_lhs];
                 project_rhs1 = [%e project_rhs1]; project_rhs2 = None }]
       in
      let body = [%expr Code.Accum_unop {
        zero_out=[%e guess_zero_out]; accum=[%e accu_op]; lhs=[%e lhs];
        op=Code.Identity; rhs=[%e rhs]; projections=[%e projections]}
    ] in
    let setups = List.filter_map ~f:Fn.id [lhs_setup; rhs_setup] in
    with_forward_args setups body

  | [%expr [%e? expr1] [%e? expr2] [%e? expr3] ] ->
    let typ1, slot1, expr1 = translate expr1 in
    let _typ2, slot2, expr2 = translate expr2 in
    let _typ3, slot3, expr3 = translate expr3 in
    let slot = Option.value ~default:Undet @@
      List.find ~f:(function Undet -> false | _ -> true) [slot1; slot2; slot3] in
    typ1, slot, [%expr [%e expr1] [%e expr2] [%e expr3]]

  | [%expr [%e? expr1] [%e? expr2] ] ->
    let typ1, slot1, expr1 = translate expr1 in
    let _typ2, slot2, expr2 = translate expr2 in
    let slot = Option.value ~default:Undet @@
      List.find ~f:(function Undet -> false | _ -> true) [slot1; slot2] in
    typ1, slot, [%expr [%e expr1] [%e expr2]]

  | {pexp_desc=Pexp_fun (arg_label, arg, opt_val, body); _} as expr ->
    let typ, slot, body = translate body in
    typ, slot, {expr with pexp_desc=Pexp_fun (arg_label, arg, opt_val, body)}
 
  | [%expr while [%e? _test_expr] do [%e? _body] done ] ->
    Code, Nonslot,
    Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl %%nn_cd: while: low-level code embeddings not supported yet"

  | [%expr for [%p? _pat] = [%e? _init] to [%e? _final] do [%e? _body_expr] done ] ->
    Code, Nonslot,
    Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl %%nn_cd: for-to: low-level code embeddings not supported yet"

  | [%expr for [%p? _pat] = [%e? _init] downto [%e? _final] do [%e? _body_expr] done ] ->
    Code, Nonslot,
    Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl %%nn_cd: for-downto: low-level code embeddings not supported yet"

  | [%expr [%e? expr1] ; [%e? expr2] ] ->
    let _typ1, _slot1, expr1 = translate expr1 in
    let _typ2, _slot1, expr2 = translate expr2 in
    Code, Nonslot, [%expr Code.Seq ([%e expr1], [%e expr2])]

  | [%expr if [%e? expr1] then [%e? expr2] else [%e? expr3]] ->
    let typ2, slot2, expr2 = translate expr2 in
    let typ3, slot3, expr3 = translate expr3 in
    let typ = if is_unknown typ2 then typ3 else typ2 in
    let slot = Option.value ~default:Undet @@
      List.find ~f:(function Undet -> false | _ -> true) [slot2; slot3] in
    typ, slot, [%expr if [%e expr1] then [%e expr2] else [%e expr3]]

  | [%expr if [%e? expr1] then [%e? expr2]] ->
    let _typ2, _slot2, expr2 = translate expr2 in
    Code, Nonslot, [%expr if [%e expr1] then [%e expr2] else Code.Noop]

  | { pexp_desc = Pexp_match (expr1, cases); _ } ->
    let typs, slots, cases =
      List.unzip3 @@ List.map cases ~f:(fun ({pc_rhs; _} as c) ->
        let typ, slot, pc_rhs = translate pc_rhs in typ, slot, {c with pc_rhs}) in
    let typ = Option.value ~default:Unknown @@
      List.find typs ~f:(Fn.non is_unknown) in
    let slot = Option.value ~default:Undet @@
      List.find ~f:(function Undet -> false | _ -> true) slots in
    typ, slot, { expr with pexp_desc = Pexp_match (expr1, cases) }

  | { pexp_desc = Pexp_let (_recflag, _bindings, _body); _ } ->
    (* TODO(80): to properly support local bindings, we need to collect the type environment. *)
    Unknown, Undet,
    Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl %%nn_cd: let-in: local let-bindings not implemented yet"
     (* let bindings = List.map bindings
         ~f:(fun binding -> {binding with pvb_expr=translate binding.pvb_expr}) in
        {expr with pexp_desc=Pexp_let (recflag, bindings, translate body)} *)

  | { pexp_desc = Pexp_open (decl, body); _ } ->
    let typ, slot, body = translate body in
    typ, slot, {expr with pexp_desc=Pexp_open (decl, body)}

  | { pexp_desc = Pexp_letmodule (name, module_expr, body); _ } ->
    let typ, slot, body = translate body in
    typ, slot, {expr with pexp_desc=Pexp_letmodule (name, module_expr, body)}

  | _ -> Unknown, Undet, expr

let expr_expander ~loc ~path:_ payload =
  match payload with
  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
    (* We are at the %ocannl annotation level: do not tranlsate the body. *)
     let bindings = List.map bindings
      ~f:(fun vb ->
        let _, _, v = translate vb.pvb_expr in
         { vb with pvb_expr=[%expr let open! NFDSL.O in [%e v]] }) in
     {payload with pexp_desc=Pexp_let (recflag, bindings, body)}
  | expr ->
    let _, _, expr = translate expr in [%expr let open! NFDSL.O in [%e expr]]

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
    let _, _, expr = translate expr in
    let loc = expr.pexp_loc in
    {str with pstr_desc=Pstr_eval ([%expr let open! NFDSL.O in [%e expr]], attrs)}
  | Pstr_value (recf, bindings) ->
    let f vb =
      let loc = vb.pvb_loc in
      let _, _, v = translate vb.pvb_expr in
      {vb with pvb_expr=[%expr let open! NFDSL.O in [%e v]]} in
    {str with pstr_desc=Pstr_value (recf, List.map bindings ~f)}
  | _ -> str
     
let str_expander ~loc ~path (payload: structure_item list) =
  flatten_str ~loc ~path @@ List.map payload ~f:translate_str
