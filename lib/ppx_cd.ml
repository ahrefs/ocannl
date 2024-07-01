open Base
open Ppxlib
open Ppx_arrayjit.Ppx_helper
open Ppx_shared

let ndarray_op ~ident_label ?axis_labels ?label expr =
  let loc = expr.pexp_loc in
  let values, batch_dims, output_dims, input_dims = ndarray_constant expr in
  let edims dims = Ast_builder.Default.elist ~loc dims in
  let op =
    match (axis_labels, label) with
    | None, None -> [%expr NTDSL.ndarray]
    | Some axis_labels, None -> [%expr NTDSL.ndarray ~axis_labels:[%e axis_labels]]
    | None, Some label -> [%expr NTDSL.ndarray ~label:[%e label]]
    | Some axis_labels, Some label ->
        [%expr
          NTDSL.ndarray ~label:[%e opt_pat2string_list ~loc ident_label] ~axis_labels:[%e axis_labels]
            ~label:[%e label]]
  in
  [%expr
    [%e op] ~label:[%e opt_pat2string_list ~loc ident_label] ~batch_dims:[%e edims batch_dims]
      ~input_dims:[%e edims input_dims] ~output_dims:[%e edims output_dims] [%e values]]

type expr_type =
  | Code
  | Array
  | Value_of_tensor of expression
  | Grad_of_tensor of expression
  | Tensor
  | Unknown
  | Merge_value of expression
  | Merge_grad of expression

let is_unknown = function Unknown -> true | _ -> false

type projections_slot = LHS | RHS1 | RHS2 | Nonslot | Undet [@@deriving equal, sexp]

let assignment_op expr =
  (* This should stay in sync with Arrayjit.Ops.assign_op_cd_syntax. *)
  let loc = expr.pexp_loc in
  match expr with
  | [%expr ( =: )] -> (false, [%expr Arrayjit.Ops.Arg2])
  | [%expr ( =+ )] -> (false, [%expr Arrayjit.Ops.Add])
  | [%expr ( =- )] -> (false, [%expr Arrayjit.Ops.Sub])
  | [%expr ( =* )] -> (false, [%expr Arrayjit.Ops.Mul])
  | [%expr ( =/ )] -> (false, [%expr Arrayjit.Ops.Div])
  | [%expr ( =** )] -> (false, [%expr Arrayjit.Ops.ToPowOf])
  | [%expr ( =?/ )] -> (false, [%expr Arrayjit.Ops.Relu_gate])
  | [%expr ( =:+ )] -> (true, [%expr Arrayjit.Ops.Add])
  | [%expr ( =:- )] -> (true, [%expr Arrayjit.Ops.Sub])
  | [%expr ( =:* )] -> (true, [%expr Arrayjit.Ops.Mul])
  | [%expr ( =:/ )] -> (true, [%expr Arrayjit.Ops.Div])
  | [%expr ( =:** )] -> (true, [%expr Arrayjit.Ops.ToPowOf])
  | [%expr ( =:?/ )] -> (true, [%expr Arrayjit.Ops.Relu_gate])
  | _ ->
      ( false,
        Ast_builder.Default.pexp_extension ~loc
        @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: expected an assignment operator, one of: %s %s"
             "=+ (Add), =- (Sub), =* (Mul), =/ (Div), =** (ToPowOf), =?/ (Relu_gate), =: (Arg2), =:+, =:-,"
             " =:*, =:/, =:**, =:?/ (same with initializing the tensor to the neutral value before the start \
              of the calculation)" )

let binary_op expr =
  (* This and is_binary_op should stay in sync with Arrayjit.Ops.binop_cd_syntax. *)
  let loc = expr.pexp_loc in
  match expr with
  | [%expr ( + )] -> ([%expr Shape.Pointwise_bin], [%expr Arrayjit.Ops.Add])
  | [%expr ( - )] -> ([%expr Shape.Pointwise_bin], [%expr Arrayjit.Ops.Sub])
  | [%expr ( * )] ->
      ( Ast_builder.Default.pexp_extension ~loc
        @@ Location.error_extensionf ~loc
             "No default compose type for binary `*`, try e.g. ~logic:\".\" for pointwise, %s"
             "~logic:\"@\" for matrix multiplication",
        [%expr Arrayjit.Ops.Mul] )
  | [%expr ( / )] ->
      ( Ast_builder.Default.pexp_extension ~loc
        @@ Location.error_extensionf ~loc
             "For clarity, no default compose type for binary `/`, use ~logic:\".\" for pointwise division",
        [%expr Arrayjit.Ops.Div] )
  | [%expr ( ** )] -> ([%expr Shape.Pointwise_bin], [%expr Arrayjit.Ops.ToPowOf])
  | [%expr ( -?/ )] -> ([%expr Shape.Pointwise_bin], [%expr Arrayjit.Ops.Relu_gate])
  | [%expr ( -/> )] -> ([%expr Shape.Pointwise_bin], [%expr Arrayjit.Ops.Arg2])
  | [%expr ( -@> )] -> ([%expr Shape.Pointwise_bin], [%expr Arrayjit.Ops.Arg1])
  | _ ->
      ( [%expr Shape.Pointwise_bin],
        Ast_builder.Default.pexp_extension ~loc
        @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: expected a binary operator, one of: %s"
             "+ (Add), - (Sub), * (Mul), / (Div), ** (ToPowOf), -?/ (Relu_gate), -/> (Arg2)" )

let is_binary_op ident = List.mem [ "+"; "-"; "*"; "/"; "**"; "-?/"; "-/>"; "-@>" ] ident ~equal:String.equal

let unary_op expr =
  (* This and is_unary_op should stay in sync with Arrayjit.Ops.unop_cd_syntax. *)
  let loc = expr.pexp_loc in
  match expr with
  | [%expr ( ~= )] -> ([%expr Shape.Pointwise_un], [%expr Arrayjit.Ops.Identity])
  | [%expr ( ?/ )] -> ([%expr Shape.Pointwise_un], [%expr Arrayjit.Ops.Relu])
  | _ ->
      ( [%expr Shape.Pointwise_un],
        Ast_builder.Default.pexp_extension ~loc
        @@ Location.error_extensionf ~loc
             "ppx_ocannl %%cd: expected a unary operator, one of: = (Identity), ?/ (Relu)" )

let is_unary_op ident = List.mem [ "~="; "?/" ] ident ~equal:String.equal

let rec array_of_code c =
  let loc = c.pexp_loc in
  [%expr
    match [%e c] with
    | Arrayjit.Assignments.Accum_binop { lhs; _ } | Accum_unop { lhs; _ } -> lhs
    | Fetch { array; _ } -> array
    | Seq (_, subexpr) | Block_comment (_, subexpr) -> [%e array_of_code [%expr subexpr]]
    | Noop -> Location.error_extensionf ~loc "ppx_ocannl %%cd: Noop code does not refer to any data"]

type binding_setup = { var : pattern; lazy_bind_to : expression; fwd_code_or_noop : expression }

let with_forward_args setups body =
  let loc = body.pexp_loc in
  let bindings =
    List.map setups ~f:(fun { var; lazy_bind_to; _ } ->
        Ast_helper.Vb.mk ~loc var [%expr Lazy.force [%e lazy_bind_to]])
  in
  let forward_args =
    List.map setups ~f:(fun { fwd_code_or_noop; _ } -> fwd_code_or_noop)
    |> List.reduce ~f:(fun code fwd -> [%expr Arrayjit.Assignments.Seq ([%e code], [%e fwd])])
  in
  ( Code,
    Nonslot,
    match forward_args with
    | None -> body
    | Some fwd ->
        [%expr
          (* FIXME: we do not want to force the computation unnecessarily, but we want the bindings? *)
          (*if Arrayjit.Assignments.is_noop [%e body] then Arrayjit.Assignments.Noop else*)
          [%e
            Ast_helper.Exp.let_ ~loc Nonrecursive bindings
              [%expr Arrayjit.Assignments.Seq ([%e fwd], [%e body])]]] )

let project_p_slot debug loc slot =
  match slot with
  | LHS -> [%expr p.project_lhs]
  | RHS1 -> [%expr p.project_rhs.(0)]
  | RHS2 -> [%expr p.project_rhs.(1)]
  | Nonslot ->
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "ppx_ocannl %%cd: not a valid accumulation/assignment slot filler at %s" debug
  | Undet ->
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: insufficient slot filler information at %s %s" debug
           "(incorporate one of: v, v1, v2, g, g1, g2, lhs, rhs, rhs1, rhs2)"

let project_p_dims debug loc slot =
  match slot with
  | LHS -> [%expr p.lhs_dims]
  | RHS1 -> [%expr p.rhs_dims.(0)]
  | RHS2 -> [%expr p.rhs_dims.(1)]
  | Nonslot ->
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "ppx_ocannl %%cd: not a valid accumulation/assignment slot filler at %s" debug
  | Undet ->
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: insufficient slot filler information at %s %s" debug
           "(incorporate one of: v, v1, v2, g, g1, g2, lhs, rhs, rhs1, rhs2)"

type array_setup = {
  slot : projections_slot;
  filler_typ : expr_type;
  binding : binding_setup option;
  array_opt : expression;
  tensor : expression option;
}

let setup_array ~is_lhs filler_pat (filler_typ, slot, filler) =
  let loc = filler.pexp_loc in
  let opt_buffer tn =
    if is_lhs then [%expr Some [%e tn]] else [%expr Some (Arrayjit.Assignments.Node [%e tn])]
  in
  let buffer opt_tn =
    if is_lhs then opt_tn else [%expr Option.map [%e opt_tn] ~f:(fun tn -> Arrayjit.Assignments.Node tn)]
  in
  match filler_typ with
  | Tensor | Unknown ->
      let t = pat2expr filler_pat in
      let fwd_code_or_noop =
        [%expr
          if Tensor.is_fwd_root [%e t] then (
            Tensor.remove_fwd_root [%e t];
            [%e t].Tensor.forward)
          else Arrayjit.Assignments.Noop]
      in
      {
        binding = Some { var = filler_pat; lazy_bind_to = [%expr lazy [%e filler]]; fwd_code_or_noop };
        filler_typ;
        slot;
        array_opt = opt_buffer [%expr [%e t].value];
        tensor = Some t;
      }
  | Code ->
      {
        binding =
          Some
            {
              var = filler_pat;
              lazy_bind_to = [%expr lazy [%e filler]];
              fwd_code_or_noop = pat2expr filler_pat;
            };
        filler_typ;
        slot;
        array_opt = opt_buffer @@ array_of_code filler;
        tensor = None;
      }
  | Array -> { binding = None; filler_typ; slot; array_opt = opt_buffer filler; tensor = None }
  | Value_of_tensor t -> { binding = None; filler_typ; slot; array_opt = opt_buffer filler; tensor = Some t }
  | Grad_of_tensor t -> { binding = None; filler_typ; slot; array_opt = buffer filler; tensor = Some t }
  | (Merge_value _ | Merge_grad _) when is_lhs ->
      {
        binding = None;
        filler_typ;
        slot;
        array_opt =
          Ast_builder.Default.pexp_extension ~loc
          @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: merge buffers cannot be assigned to";
        tensor = None;
      }
  | Merge_value t ->
      {
        binding = None;
        filler_typ;
        slot;
        array_opt = [%expr Some (Merge_buffer [%e filler])];
        tensor = Some t;
      }
  | Merge_grad t ->
      {
        binding = None;
        filler_typ;
        slot;
        array_opt = [%expr Option.map [%e filler] ~f:(fun tn -> Arrayjit.Assignments.Merge_buffer tn)];
        tensor = Some t;
      }

let args_for ~loc = function
  | { filler_typ = Merge_grad _; tensor = Some t; _ } -> (t, [%expr true], [%expr true])
  | { filler_typ = Grad_of_tensor _; tensor = Some t; _ } -> (t, [%expr true], [%expr false])
  | { filler_typ = Merge_value _; tensor = Some t; _ } -> (t, [%expr false], [%expr true])
  | { filler_typ = _; tensor = Some t; _ } -> (t, [%expr false], [%expr false])
  | _ ->
      ( Ast_builder.Default.pexp_extension ~loc
        @@ Location.error_extensionf ~loc
             "ppx_ocannl %%cd: cannot use `~logic` (infer shapes) for arrays, use tensors or `.value` or \
              `.grad` notation",
        [%expr false],
        [%expr false] )

let rec translate ?ident_label ~proj_in_scope (expr : expression) : expr_type * projections_slot * expression
    =
  let loc = expr.pexp_loc in
  match expr with
  | { pexp_desc = Pexp_constant (Pconst_float _); _ } ->
      (Tensor, Undet, [%expr NTDSL.number ~label:[%e opt_pat2string_list ~loc ident_label] [%e expr]])
  | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
      ( Tensor,
        Undet,
        [%expr NTDSL.number ~label:[%e opt_pat2string_list ~loc ident_label] (Float.of_int [%e expr])] )
  | [%expr
      [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
        [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
      let axis = Ast_helper.Exp.constant ~loc:pexp_loc (Pconst_string (String.of_char ch, pexp_loc, None)) in
      ( Tensor,
        Undet,
        [%expr NTDSL.number ~label:[%e opt_pat2string_list ~loc ident_label] ~axis_label:[%e axis] [%e f]] )
  | [%expr
      [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
        [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
      let axis = Ast_helper.Exp.constant ~loc:pexp_loc (Pconst_string (String.of_char ch, pexp_loc, None)) in
      ( Tensor,
        Undet,
        [%expr
          NTDSL.number ~label:[%e opt_pat2string_list ~loc ident_label] ~axis_label:[%e axis]
            (Float.of_int [%e i])] )
      (* | [%expr [%e? { pexp_desc = Pexp_ident { txt = Lident "merge_buffer"; _ }; _ }] [%e? for_tn]] -> let
         typ1, slot1, e1 = translate ~proj_in_scope for_tn in ( typ1, slot1, [%expr ] ) *)
  | { pexp_desc = Pexp_array _; _ } | { pexp_desc = Pexp_construct ({ txt = Lident "::"; _ }, _); _ } ->
      (Tensor, Undet, ndarray_op ~ident_label expr)
  | { pexp_desc = Pexp_ident { txt = Lident ("v" | "lhs"); _ }; _ } -> (Array, LHS, expr)
  | { pexp_desc = Pexp_ident { txt = Lident "g"; _ }; _ } -> (Array, LHS, expr)
  | { pexp_desc = Pexp_ident { txt = Lident "rhs1"; _ }; _ } -> (Array, RHS1, expr)
  | { pexp_desc = Pexp_ident { txt = Lident "t1"; _ }; _ } -> (Tensor, RHS1, expr)
  | { pexp_desc = Pexp_ident { txt = Lident "v1"; _ }; _ } -> (Array, RHS1, [%expr t1.Tensor.value])
  | { pexp_desc = Pexp_ident { txt = Lident "g1"; _ }; _ } ->
      (Grad_of_tensor [%expr t1], RHS1, [%expr Option.map t1.Tensor.diff ~f:(fun d -> d.Tensor.grad)])
  | { pexp_desc = Pexp_ident { txt = Lident "rhs2"; _ }; _ } -> (Array, RHS2, expr)
  | { pexp_desc = Pexp_ident { txt = Lident "t2"; _ }; _ } -> (Tensor, RHS2, expr)
  | { pexp_desc = Pexp_ident { txt = Lident "v2"; _ }; _ } -> (Array, RHS2, [%expr t2.Tensor.value])
  | { pexp_desc = Pexp_ident { txt = Lident "g2"; _ }; _ } ->
      (Grad_of_tensor [%expr t2], RHS2, [%expr Option.map t2.Tensor.diff ~f:(fun d -> d.Tensor.grad)])
  | { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ } when is_operator op_ident ->
      (Tensor, Undet, expr)
  | [%expr [%e? expr1] **. [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
      (* FIXME: `**.` should take a tensor and require that it's a literal. *)
      (* We need to hardcode these two patterns to prevent the numbers from being converted to tensors. *)
      let _typ1, slot1, e1 = translate ~proj_in_scope expr1 in
      ( Tensor,
        slot1,
        [%expr NTDSL.O.( **. ) ~label:[%e opt_pat2string_list ~loc ident_label] [%e e1] (Float.of_int [%e i])]
      )
  | [%expr [%e? expr1] **. [%e? expr2]] ->
      let _typ1, slot1, e1 = translate ~proj_in_scope expr1 in
      ( Tensor,
        slot1,
        [%expr NTDSL.O.( **. ) ~label:[%e opt_pat2string_list ~loc ident_label] [%e e1] [%e expr2]] )
  | [%expr
      [%e? expr1]
      *+ [%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ } as spec] [%e? expr2]]
    when String.contains spec_str '>' ->
      let _typ1, slot1, expr1 = translate ~proj_in_scope expr1 in
      let _typ2, slot2, expr2 = translate ~proj_in_scope expr2 in
      let slot =
        Option.value ~default:Undet @@ List.find ~f:(function Undet -> false | _ -> true) [ slot1; slot2 ]
      in
      ( Tensor,
        slot,
        [%expr NTDSL.einsum ~label:[%e opt_pat2string_list ~loc ident_label] [%e spec] [%e expr1] [%e expr2]]
      )
  | [%expr [%e? expr1] ++ [%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ } as spec]]
    when String.contains spec_str '>' ->
      let _typ1, slot1, expr1 = translate ~proj_in_scope expr1 in
      ( Tensor,
        slot1,
        [%expr NTDSL.einsum1 ~label:[%e opt_pat2string_list ~loc ident_label] [%e spec] [%e expr1]] )
  | [%expr [%e? expr1].grad] -> (
      let typ1, slot1, expr1 = translate ?ident_label ~proj_in_scope expr1 in
      match typ1 with
      | Unknown | Tensor ->
          (Grad_of_tensor expr1, slot1, [%expr Option.map [%e expr1].Tensor.diff ~f:(fun d -> d.Tensor.grad)])
      | Merge_value _ ->
          ( Merge_grad expr1,
            slot1,
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: write .grad.merge instead of .merge.grad" )
      | Code | Array | Value_of_tensor _ | Grad_of_tensor _ | Merge_grad _ ->
          ( Array,
            slot1,
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: only tensors have a gradient" ))
  | [%expr [%e? expr1].value] -> (
      let ((typ1, slot1, expr1) as result) = translate ?ident_label ~proj_in_scope expr1 in
      (* TODO: maybe this is too permissive? E.g. [t1.grad.value] is accepted. *)
      match typ1 with
      | Unknown | Tensor -> (Value_of_tensor expr1, slot1, [%expr [%e expr1].Tensor.value])
      | Code -> (Array, slot1, array_of_code expr1)
      | Array | Value_of_tensor _ | Grad_of_tensor _ | Merge_value _ | Merge_grad _ -> result)
  | [%expr [%e? expr1].merge] -> (
      let typ1, slot1, expr1 = translate ?ident_label ~proj_in_scope expr1 in
      match typ1 with
      | Unknown | Tensor -> (Merge_value expr1, slot1, [%expr [%e expr1].Tensor.value])
      | Value_of_tensor t -> (Merge_value t, slot1, [%expr [%e expr1].Tensor.value])
      | Array | Code ->
          ( Array,
            slot1,
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%cd: only tensor nodes (e.g. `.value` or `.grad`) can be merged" )
      | Grad_of_tensor t -> (Merge_grad t, slot1, expr1)
      | Merge_value _ | Merge_grad _ ->
          ( typ1,
            slot1,
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: repeated .merge not allowed" ))
  | [%expr [%e? accu_op] [%e? lhs] ([%e? bin_op] [%e? rhs1] ([%e? rhs2] ~projections:[%e? projections]))] ->
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l =
        setup_array ~is_lhs:true [%pat? nondiff___lhs] @@ translate ?ident_label ~proj_in_scope:true lhs
      in
      let _, bin_op = binary_op bin_op in
      let setup_r1 = setup_array ~is_lhs:false [%pat? nondiff___rhs1] @@ translate ~proj_in_scope:true rhs1 in
      let setup_r2 = setup_array ~is_lhs:false [%pat? nondiff___rhs2] @@ translate ~proj_in_scope:true rhs2 in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      (* TODO: might be better to treat missing [rhs1, rhs2] as zeros rather than eliding the code. *)
      let body =
        [%expr
          Option.value ~default:Arrayjit.Assignments.Noop
          @@ Option.map3 [%e setup_l.array_opt] [%e setup_r1.array_opt] [%e setup_r2.array_opt]
               ~f:(fun lhs rhs1 rhs2 ->
                 Arrayjit.Assignments.Accum_binop
                   {
                     initialize_neutral = [%e initialize_neutral];
                     accum = [%e accu_op];
                     lhs;
                     op = [%e bin_op];
                     rhs1;
                     rhs2;
                     projections = [%e projections];
                   })]
      in
      let setups = List.filter_map ~f:(fun setup -> setup.binding) [ setup_l; setup_r1; setup_r2 ] in
      with_forward_args setups body
  | [%expr [%e? accu_op] [%e? lhs] (([%e? un_op] [%e? rhs]) ~projections:[%e? projections])]
  | [%expr [%e? accu_op] [%e? lhs] ([%e? un_op] ([%e? rhs] ~projections:[%e? projections]))] ->
      (* Handle both un_op priority levels -- where application binds tighter and less tight. *)
      let initialize_neutral, accu_op = assignment_op accu_op in
      (* FIXME: I think this ignores the slot information here! Just assuming [projections] is as-should-be,
         but that's not consistent with omitting the projections arg (assuming it comes from the context). *)
      let setup_l =
        setup_array ~is_lhs:true [%pat? nondiff___lhs] @@ translate ?ident_label ~proj_in_scope:true lhs
      in
      let _, un_op = unary_op un_op in
      let setup_r = setup_array ~is_lhs:false [%pat? nondiff___rhs] @@ translate ~proj_in_scope:true rhs in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      (* TODO: might be better to treat missing [rhs] as zeros rather than eliding the code. *)
      let body =
        [%expr
          Option.value ~default:Arrayjit.Assignments.Noop
          @@ Option.map2 [%e setup_l.array_opt] [%e setup_r.array_opt] ~f:(fun lhs rhs ->
                 Arrayjit.Assignments.Accum_unop
                   {
                     initialize_neutral = [%e initialize_neutral];
                     accum = [%e accu_op];
                     lhs;
                     op = [%e un_op];
                     rhs;
                     projections = [%e projections];
                   })]
      in
      let setups = List.filter_map ~f:(fun setup -> setup.binding) [ setup_l; setup_r ] in
      with_forward_args setups body
  | [%expr [%e? accu_op] [%e? lhs] ([%e? rhs] ~projections:[%e? projections])] ->
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l =
        setup_array ~is_lhs:true [%pat? nondiff___lhs] @@ translate ?ident_label ~proj_in_scope:true lhs
      in
      let setup_r = setup_array ~is_lhs:false [%pat? nondiff___rhs] @@ translate ~proj_in_scope:true rhs in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let body =
        [%expr
          Option.value ~default:Arrayjit.Assignments.Noop
          @@ Option.map2 [%e setup_l.array_opt] [%e setup_r.array_opt] ~f:(fun lhs rhs ->
                 Arrayjit.Assignments.Accum_unop
                   {
                     initialize_neutral = [%e initialize_neutral];
                     accum = [%e accu_op];
                     lhs;
                     op = Arrayjit.Ops.Identity;
                     rhs;
                     projections = [%e projections];
                   })]
      in
      let setups = List.filter_map ~f:(fun setup -> setup.binding) [ setup_l; setup_r ] in
      with_forward_args setups body
  | [%expr
      [%e? accu_op]
        [%e? lhs]
        ([%e? bin_op]
           [%e? rhs1]
           ([%e? rhs2]
              ~logic:[%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ } as logic]))] ->
      let logic =
        let loc = s_loc in
        if String.equal spec "." then [%expr Shape.Pointwise_bin]
        else if String.equal spec "@" then [%expr Shape.Compose]
        else [%expr Shape.Einsum [%e logic]]
      in
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l =
        setup_array ~is_lhs:true [%pat? nondiff___lhs] @@ translate ?ident_label ~proj_in_scope lhs
      in
      let _, bin_op = binary_op bin_op in
      let setup_r1 = setup_array ~is_lhs:false [%pat? nondiff___rhs1] @@ translate ~proj_in_scope rhs1 in
      let setup_r2 = setup_array ~is_lhs:false [%pat? nondiff___rhs2] @@ translate ~proj_in_scope rhs2 in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let t_expr, lhs_is_grad, _ = args_for ~loc setup_l in
      let t1_expr, rhs1_is_grad, rhs1_is_merge = args_for ~loc setup_r1 in
      let t2_expr, rhs2_is_grad, rhs2_is_merge = args_for ~loc setup_r2 in
      let body =
        [%expr
          Tensor.raw_binop ~initialize_neutral:[%e initialize_neutral] ~accum:[%e accu_op] ~t:[%e t_expr]
            ~lhs_is_grad:[%e lhs_is_grad] ~op:[%e bin_op] ~t1:[%e t1_expr] ~rhs1_is_grad:[%e rhs1_is_grad]
            ~rhs1_is_merge:[%e rhs1_is_merge] ~t2:[%e t2_expr] ~rhs2_is_grad:[%e rhs2_is_grad]
            ~rhs2_is_merge:[%e rhs2_is_merge] ~logic:[%e logic]]
      in
      let setups = List.filter_map ~f:(fun setup -> setup.binding) [ setup_l; setup_r1; setup_r2 ] in
      with_forward_args setups body
  | [%expr
      [%e? accu_op]
        [%e? lhs]
        (([%e? un_op] [%e? rhs])
           ~logic:[%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ } as logic])]
  | [%expr
      [%e? accu_op]
        [%e? lhs]
        ([%e? un_op]
           ([%e? rhs] ~logic:[%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ } as logic]))]
    ->
      (* Handle both un_op priority levels -- where application binds tighter and less tight. *)
      let logic =
        let loc = s_loc in
        if String.equal spec "." then [%expr Shape.Pointwise_un]
        else if String.equal spec "T" then [%expr Shape.Transpose]
        else [%expr Shape.Permute [%e logic]]
      in
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l =
        setup_array ~is_lhs:true [%pat? nondiff___lhs] @@ translate ?ident_label ~proj_in_scope lhs
      in
      let _, un_op = unary_op un_op in
      let setup_r = setup_array ~is_lhs:false [%pat? nondiff___rhs] @@ translate ~proj_in_scope rhs in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let t_expr, lhs_is_grad, _ = args_for ~loc setup_l in
      let t1_expr, rhs_is_grad, rhs_is_merge = args_for ~loc setup_r in
      let body =
        [%expr
          Tensor.raw_unop ~initialize_neutral:[%e initialize_neutral] ~accum:[%e accu_op] ~t:[%e t_expr]
            ~lhs_is_grad:[%e lhs_is_grad] ~op:[%e un_op] ~t1:[%e t1_expr] ~rhs_is_grad:[%e rhs_is_grad]
            ~rhs_is_merge:[%e rhs_is_merge] ~logic:[%e logic]]
      in
      let setups = List.filter_map ~f:(fun setup -> setup.binding) [ setup_l; setup_r ] in
      with_forward_args setups body
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident accu_ident; loc = accu_loc }; _ } as accu_op]
        [%e? lhs]
        ([%e? { pexp_desc = Pexp_ident { txt = Lident binop_ident; loc = op_loc }; _ } as bin_op]
           [%e? rhs1]
           [%e? rhs2])]
    when is_assignment accu_ident && is_binary_op binop_ident && proj_in_scope ->
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l =
        setup_array ~is_lhs:true [%pat? nondiff___lhs] @@ translate ?ident_label ~proj_in_scope lhs
      in
      let _, bin_op = binary_op bin_op in
      let setup_r1 = setup_array ~is_lhs:false [%pat? nondiff___rhs1] @@ translate ~proj_in_scope rhs1 in
      let setup_r2 = setup_array ~is_lhs:false [%pat? nondiff___rhs2] @@ translate ~proj_in_scope rhs2 in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let projections =
        let lhs_dims = project_p_dims "LHS" lhs.pexp_loc setup_l.slot in
        let rhs1_dims = project_p_dims "RHS1" lhs.pexp_loc setup_r1.slot in
        let rhs2_dims = project_p_dims "RHS2" lhs.pexp_loc setup_r2.slot in
        let project_lhs = project_p_slot "LHS" lhs.pexp_loc setup_l.slot in
        let project_rhs1 = project_p_slot "RHS1" rhs1.pexp_loc setup_r1.slot in
        let project_rhs2 = project_p_slot "RHS2" rhs2.pexp_loc setup_r2.slot in
        [%expr
          lazy
            (let p = Lazy.force projections in
             Arrayjit.Indexing.
               {
                 product_space = p.product_space;
                 product_iterators = p.product_iterators;
                 lhs_dims = [%e lhs_dims];
                 rhs_dims = [| [%e rhs1_dims]; [%e rhs2_dims] |];
                 project_lhs = [%e project_lhs];
                 project_rhs = [| [%e project_rhs1]; [%e project_rhs2] |];
                 debug_info =
                   {
                     p.debug_info with
                     trace =
                       ( "ppx_cd " ^ [%e string_expr ~loc:accu_loc accu_ident] ^ " "
                         ^ [%e string_expr ~loc:op_loc binop_ident],
                         Arrayjit.Indexing.unique_debug_id () )
                       :: p.debug_info.trace;
                   };
               })]
      in
      let body =
        [%expr
          Option.value ~default:Arrayjit.Assignments.Noop
          @@ Option.map3 [%e setup_l.array_opt] [%e setup_r1.array_opt] [%e setup_r2.array_opt]
               ~f:(fun lhs rhs1 rhs2 ->
                 Arrayjit.Assignments.Accum_binop
                   {
                     initialize_neutral = [%e initialize_neutral];
                     accum = [%e accu_op];
                     lhs;
                     op = [%e bin_op];
                     rhs1;
                     rhs2;
                     projections = [%e projections];
                   })]
      in
      let setups = List.filter_map ~f:(fun setup -> setup.binding) [ setup_l; setup_r1; setup_r2 ] in
      with_forward_args setups body
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident accu_ident; loc = accu_loc }; _ } as accu_op]
        [%e? lhs]
        ([%e? { pexp_desc = Pexp_ident { txt = Lident unop_ident; loc = op_loc }; _ } as un_op] [%e? rhs])]
    when is_assignment accu_ident && is_unary_op unop_ident && proj_in_scope ->
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l =
        setup_array ~is_lhs:true [%pat? nondiff___lhs] @@ translate ?ident_label ~proj_in_scope lhs
      in
      let _, un_op = unary_op un_op in
      let setup_r1 = setup_array ~is_lhs:false [%pat? nondiff___rhs1] @@ translate ~proj_in_scope rhs in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let projections =
        let lhs_dims = project_p_dims "LHS" lhs.pexp_loc setup_l.slot in
        let rhs1_dims = project_p_dims "RHS1" lhs.pexp_loc setup_r1.slot in
        let project_lhs = project_p_slot "LHS" lhs.pexp_loc setup_l.slot in
        let project_rhs1 = project_p_slot "RHS1" rhs.pexp_loc setup_r1.slot in
        [%expr
          lazy
            (let p = Lazy.force projections in
             Arrayjit.Indexing.
               {
                 product_space = p.product_space;
                 product_iterators = p.product_iterators;
                 lhs_dims = [%e lhs_dims];
                 rhs_dims = [| [%e rhs1_dims] |];
                 project_lhs = [%e project_lhs];
                 project_rhs = [| [%e project_rhs1] |];
                 debug_info =
                   {
                     p.debug_info with
                     trace =
                       ( "ppx_cd " ^ [%e string_expr ~loc:accu_loc accu_ident] ^ " "
                         ^ [%e string_expr ~loc:op_loc unop_ident],
                         Arrayjit.Indexing.unique_debug_id () )
                       :: p.debug_info.trace;
                   };
               })]
      in
      let body =
        [%expr
          Option.value ~default:Arrayjit.Assignments.Noop
          @@ Option.map2 [%e setup_l.array_opt] [%e setup_r1.array_opt] ~f:(fun lhs rhs ->
                 Arrayjit.Assignments.Accum_unop
                   {
                     initialize_neutral = [%e initialize_neutral];
                     accum = [%e accu_op];
                     lhs;
                     op = [%e un_op];
                     rhs;
                     projections = [%e projections];
                   })]
      in
      let setups = List.filter_map ~f:(fun setup -> setup.binding) [ setup_l; setup_r1 ] in
      with_forward_args setups body
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; loc = op_loc }; _ } as accu_op]
        [%e? lhs]
        [%e? rhs]]
    when is_assignment op_ident && proj_in_scope ->
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l =
        setup_array ~is_lhs:true [%pat? nondiff___lhs] @@ translate ?ident_label ~proj_in_scope lhs
      in
      let setup_r1 = setup_array ~is_lhs:false [%pat? nondiff___rhs1] @@ translate ~proj_in_scope rhs in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let projections =
        let lhs_dims = project_p_dims "LHS" lhs.pexp_loc setup_l.slot in
        let rhs1_dims = project_p_dims "RHS1" lhs.pexp_loc setup_r1.slot in
        let project_lhs = project_p_slot "LHS" lhs.pexp_loc setup_l.slot in
        let project_rhs1 = project_p_slot "RHS1" rhs.pexp_loc setup_r1.slot in
        [%expr
          lazy
            (let p = Lazy.force projections in
             Arrayjit.Indexing.
               {
                 product_space = p.product_space;
                 product_iterators = p.product_iterators;
                 lhs_dims = [%e lhs_dims];
                 rhs_dims = [| [%e rhs1_dims] |];
                 project_lhs = [%e project_lhs];
                 project_rhs = [| [%e project_rhs1] |];
                 debug_info =
                   {
                     p.debug_info with
                     trace =
                       ( "ppx_cd " ^ [%e string_expr ~loc:op_loc op_ident],
                         Arrayjit.Indexing.unique_debug_id () )
                       :: p.debug_info.trace;
                   };
               })]
      in
      let body =
        [%expr
          Option.value ~default:Arrayjit.Assignments.Noop
          @@ Option.map2 [%e setup_l.array_opt] [%e setup_r1.array_opt] ~f:(fun lhs rhs ->
                 Arrayjit.Assignments.Accum_unop
                   {
                     initialize_neutral = [%e initialize_neutral];
                     accum = [%e accu_op];
                     lhs;
                     op = Arrayjit.Ops.Identity;
                     rhs;
                     projections = [%e projections];
                   })]
      in
      let setups = List.filter_map ~f:(fun setup -> setup.binding) [ setup_l; setup_r1 ] in
      with_forward_args setups body
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident accu_ident; _ }; _ } as accu_op]
        [%e? lhs]
        ([%e? { pexp_desc = Pexp_ident { txt = Lident binop_ident; _ }; _ } as bin_op] [%e? rhs1] [%e? rhs2])]
    when is_assignment accu_ident && is_binary_op binop_ident ->
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l =
        setup_array ~is_lhs:true [%pat? nondiff___lhs] @@ translate ?ident_label ~proj_in_scope lhs
      in
      let logic, bin_op = binary_op bin_op in
      let setup_r1 = setup_array ~is_lhs:false [%pat? nondiff___rhs1] @@ translate ~proj_in_scope rhs1 in
      let setup_r2 = setup_array ~is_lhs:false [%pat? nondiff___rhs2] @@ translate ~proj_in_scope rhs2 in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let t_expr, lhs_is_grad, _ = args_for ~loc setup_l in
      let t1_expr, rhs1_is_grad, rhs1_is_merge = args_for ~loc setup_r1 in
      let t2_expr, rhs2_is_grad, rhs2_is_merge = args_for ~loc setup_r2 in
      let body =
        [%expr
          Tensor.raw_binop ~initialize_neutral:[%e initialize_neutral] ~accum:[%e accu_op] ~t:[%e t_expr]
            ~lhs_is_grad:[%e lhs_is_grad] ~op:[%e bin_op] ~t1:[%e t1_expr] ~rhs1_is_grad:[%e rhs1_is_grad]
            ~rhs1_is_merge:[%e rhs1_is_merge] ~t2:[%e t2_expr] ~rhs2_is_grad:[%e rhs2_is_grad]
            ~rhs2_is_merge:[%e rhs2_is_merge] ~logic:[%e logic]]
      in
      let setups = List.filter_map ~f:(fun setup -> setup.binding) [ setup_l; setup_r1; setup_r2 ] in
      with_forward_args setups body
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident accu_ident; _ }; _ } as accu_op]
        [%e? lhs]
        ([%e? { pexp_desc = Pexp_ident { txt = Lident unop_ident; _ }; _ } as un_op] [%e? rhs])]
    when is_assignment accu_ident && is_unary_op unop_ident ->
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l =
        setup_array ~is_lhs:true [%pat? nondiff___lhs] @@ translate ?ident_label ~proj_in_scope lhs
      in
      let logic, un_op = unary_op un_op in
      let setup_r = setup_array ~is_lhs:false [%pat? nondiff___rhs] @@ translate ~proj_in_scope rhs in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let t_expr, lhs_is_grad, _ = args_for ~loc setup_l in
      let t1_expr, rhs_is_grad, rhs_is_merge = args_for ~loc setup_r in
      let body =
        [%expr
          Tensor.raw_unop ~initialize_neutral:[%e initialize_neutral] ~accum:[%e accu_op] ~t:[%e t_expr]
            ~lhs_is_grad:[%e lhs_is_grad] ~op:[%e un_op] ~t1:[%e t1_expr] ~rhs_is_grad:[%e rhs_is_grad]
            ~rhs_is_merge:[%e rhs_is_merge] ~logic:[%e logic]]
      in
      let setups = List.filter_map ~f:(fun setup -> setup.binding) [ setup_l; setup_r ] in
      with_forward_args setups body
  | [%expr [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ } as accu_op] [%e? lhs] [%e? rhs]]
    when is_assignment op_ident ->
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l =
        setup_array ~is_lhs:true [%pat? nondiff___lhs] @@ translate ?ident_label ~proj_in_scope lhs
      in
      let setup_r = setup_array ~is_lhs:false [%pat? nondiff___rhs] @@ translate ~proj_in_scope rhs in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let t_expr, lhs_is_grad, _ = args_for ~loc setup_l in
      let t1_expr, rhs_is_grad, rhs_is_merge = args_for ~loc setup_r in
      let body =
        [%expr
          Tensor.raw_unop ~initialize_neutral:[%e initialize_neutral] ~accum:[%e accu_op] ~t:[%e t_expr]
            ~lhs_is_grad:[%e lhs_is_grad] ~op:Arrayjit.Ops.Identity ~t1:[%e t1_expr]
            ~rhs_is_grad:[%e rhs_is_grad] ~rhs_is_merge:[%e rhs_is_merge] ~logic:Shape.Pointwise_un]
      in
      let setups = List.filter_map ~f:(fun setup -> setup.binding) [ setup_l; setup_r ] in
      with_forward_args setups body
  | [%expr [%e? expr1] [%e? expr2] [%e? expr3]] ->
      let typ1, slot1, expr1 = translate ?ident_label ~proj_in_scope expr1 in
      let _typ2, slot2, expr2 = translate ~proj_in_scope expr2 in
      let _typ3, slot3, expr3 = translate ~proj_in_scope expr3 in
      let slot =
        Option.value ~default:Undet
        @@ List.find ~f:(function Undet -> false | _ -> true) [ slot1; slot2; slot3 ]
      in
      (typ1, slot, [%expr [%e expr1] [%e expr2] [%e expr3]])
  | [%expr [%e? expr1] [%e? expr2]] ->
      let typ1, slot1, expr1 = translate ?ident_label ~proj_in_scope expr1 in
      let _typ2, slot2, expr2 = translate ~proj_in_scope expr2 in
      let slot =
        Option.value ~default:Undet @@ List.find ~f:(function Undet -> false | _ -> true) [ slot1; slot2 ]
      in
      (typ1, slot, [%expr [%e expr1] [%e expr2]])
  | { pexp_desc = Pexp_fun ((arg_label : arg_label), arg, opt_val, body); _ } as expr ->
      let proj_in_scope =
        proj_in_scope
        ||
        match arg_label with
        | (Labelled s | Optional s) when String.equal s "projections" -> true
        | _ -> false
      in
      let typ, slot, body = translate ?ident_label ~proj_in_scope body in
      (typ, slot, { expr with pexp_desc = Pexp_fun (arg_label, arg, opt_val, body) })
  | [%expr
      while [%e? _test_expr] do
        [%e? _body]
      done] ->
      ( Code,
        Nonslot,
        Ast_builder.Default.pexp_extension ~loc
        @@ Location.error_extensionf ~loc
             "ppx_ocannl %%cd: while: low-level code embeddings not supported yet" )
  | [%expr
      for [%p? _pat] = [%e? _init] to [%e? _final] do
        [%e? _body_expr]
      done] ->
      ( Code,
        Nonslot,
        Ast_builder.Default.pexp_extension ~loc
        @@ Location.error_extensionf ~loc
             "ppx_ocannl %%cd: for-to: low-level code embeddings not supported yet" )
  | [%expr
      for [%p? _pat] = [%e? _init] downto [%e? _final] do
        [%e? _body_expr]
      done] ->
      ( Code,
        Nonslot,
        Ast_builder.Default.pexp_extension ~loc
        @@ Location.error_extensionf ~loc
             "ppx_ocannl %%cd: for-downto: low-level code embeddings not supported yet" )
  | [%expr
      [%e? expr1];
      [%e? expr2]] ->
      let _typ1, _slot1, expr1 = translate ~proj_in_scope expr1 in
      let _typ2, _slot2, expr2 = translate ?ident_label ~proj_in_scope expr2 in
      (Code, Nonslot, [%expr Arrayjit.Assignments.Seq ([%e expr1], [%e expr2])])
  | [%expr if [%e? expr1] then [%e? expr2] else [%e? expr3]] ->
      let typ2, slot2, expr2 = translate ?ident_label ~proj_in_scope expr2 in
      let typ3, slot3, expr3 = translate ?ident_label ~proj_in_scope expr3 in
      let typ = if is_unknown typ2 then typ3 else typ2 in
      let slot =
        Option.value ~default:Undet @@ List.find ~f:(function Undet -> false | _ -> true) [ slot2; slot3 ]
      in
      (typ, slot, [%expr if [%e expr1] then [%e expr2] else [%e expr3]])
  | [%expr if [%e? expr1] then [%e? expr2]] ->
      let _typ2, _slot2, expr2 = translate ?ident_label ~proj_in_scope expr2 in
      (Code, Nonslot, [%expr if [%e expr1] then [%e expr2] else Arrayjit.Assignments.Noop])
  | { pexp_desc = Pexp_match (expr1, cases); _ } ->
      let typs, slots, cases =
        List.unzip3
        @@ List.map cases ~f:(fun ({ pc_rhs; _ } as c) ->
               let typ, slot, pc_rhs = translate ?ident_label ~proj_in_scope pc_rhs in
               (typ, slot, { c with pc_rhs }))
      in
      let typ = Option.value ~default:Unknown @@ List.find typs ~f:(Fn.non is_unknown) in
      let slot = Option.value ~default:Undet @@ List.find ~f:(function Undet -> false | _ -> true) slots in
      (typ, slot, { expr with pexp_desc = Pexp_match (expr1, cases) })
  | { pexp_desc = Pexp_let (_recflag, _bindings, _body); _ } ->
      (* TODO(80): to properly support local bindings, we need to collect the type environment. *)
      ( Unknown,
        Undet,
        Ast_builder.Default.pexp_extension ~loc
        @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: let-in: local let-bindings not implemented yet" )
  (* let bindings = List.map bindings ~f:(fun binding -> {binding with pvb_expr=translate binding.pvb_expr})
     in {expr with pexp_desc=Pexp_let (recflag, bindings, translate body)} *)
  | { pexp_desc = Pexp_open (decl, body); _ } ->
      let typ, slot, body = translate ?ident_label ~proj_in_scope body in
      (typ, slot, { expr with pexp_desc = Pexp_open (decl, body) })
  | { pexp_desc = Pexp_letmodule (name, module_expr, body); _ } ->
      let typ, slot, body = translate ?ident_label ~proj_in_scope body in
      (typ, slot, { expr with pexp_desc = Pexp_letmodule (name, module_expr, body) })
  | { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ } when is_operator op_ident ->
      (Unknown, Undet, [%expr [%e expr] ~label:[%e opt_pat2string_list ~loc ident_label]])
  | _ -> (Unknown, Undet, expr)

let translate ?ident_label (expr : expression) : expression =
  let _, _, v = translate ?ident_label ~proj_in_scope:false expr in
  match ident_label with
  | Some [%pat? _] ->
      let loc = v.pexp_loc in
      [%expr Tensor.with_unchanged_roots ~f:(fun () -> [%e v])]
  | _ -> v

let expr_expander ~loc ~path:_ payload =
  match payload with
  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
      (* We are at the %cd annotation level: do not tranlsate the body. *)
      let bindings =
        List.map bindings ~f:(fun vb ->
            let v = translate ~ident_label:vb.pvb_pat vb.pvb_expr in
            {
              vb with
              pvb_expr =
                [%expr
                  let open! NTDSL.O in
                  [%e v]];
            })
      in
      { payload with pexp_desc = Pexp_let (recflag, bindings, body) }
  | expr ->
      let expr = translate expr in
      [%expr
        let open! NTDSL.O in
        [%e expr]]

let flatten_str ~loc ~path:_ items =
  match items with
  | [ x ] -> x
  | _ ->
      Ast_helper.Str.include_
        { pincl_mod = Ast_helper.Mod.structure items; pincl_loc = loc; pincl_attributes = [] }

let translate_str ({ pstr_desc; _ } as str) =
  match pstr_desc with
  | Pstr_eval (expr, attrs) ->
      let expr = translate expr in
      let loc = expr.pexp_loc in
      {
        str with
        pstr_desc =
          Pstr_eval
            ( [%expr
                let open! NTDSL.O in
                [%e expr]],
              attrs );
      }
  | Pstr_value (recf, bindings) ->
      let f vb =
        let loc = vb.pvb_loc in
        let v = translate ~ident_label:vb.pvb_pat vb.pvb_expr in
        {
          vb with
          pvb_expr =
            [%expr
              let open! NTDSL.O in
              [%e v]];
        }
      in
      { str with pstr_desc = Pstr_value (recf, List.map bindings ~f) }
  | _ -> str

let str_expander ~loc ~path (payload : structure_item list) =
  flatten_str ~loc ~path @@ List.map payload ~f:translate_str
