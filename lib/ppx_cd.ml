open Base
open Ppxlib
open Ppx_arrayjit.Ppx_helper
open Ppx_shared
module A = Ppxlib_ast.Ast_helper

let ndarray_op ?axis_labels ?label expr =
  let loc = expr.pexp_loc in
  let values, batch_dims, output_dims, input_dims = ndarray_constant expr in
  let edims dims = Ast_builder.Default.elist ~loc dims in
  let op =
    match (axis_labels, label) with
    | None, None -> [%expr NTDSL.ndarray]
    | Some axis_labels, None -> [%expr NTDSL.ndarray ~axis_labels:[%e axis_labels]]
    | None, Some label -> [%expr NTDSL.ndarray ~label:[%e label]]
    | Some axis_labels, Some label ->
        [%expr NTDSL.ndarray ~axis_labels:[%e axis_labels] ~label:[%e label]]
  in
  [%expr
    [%e op] ~batch_dims:[%e edims batch_dims] ~input_dims:[%e edims input_dims]
      ~output_dims:[%e edims output_dims] [%e values]]

type expr_type =
  | Code
  | Array
  | Value_of_tensor of expression
  | Grad_of_tensor of expression
  | Tensor
  | Unknown
  | Merge_value of expression
  | Merge_grad of expression
  | No_grad_tensor_intro of { name : string; name_expr : expression }

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
        @@ Location.error_extensionf ~loc
             "ppx_ocannl %%cd: expected an assignment operator, one of: %s %s"
             "=+ (Add), =- (Sub), =* (Mul), =/ (Div), =** (ToPowOf), =?/ (Relu_gate), =: (Arg2), \
              =:+, =:-,"
             " =:*, =:/, =:**, =:?/ (same with initializing the tensor to the neutral value before \
              the start of the calculation)" )

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
             "For clarity, no default compose type for binary `/`, use ~logic:\".\" for pointwise \
              division",
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

let is_binary_op ident =
  List.mem [ "+"; "-"; "*"; "/"; "**"; "-?/"; "-/>"; "-@>" ] ident ~equal:String.equal

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

type result = {
  vbs : value_binding Map.M(String).t;
      (** [vbs] are the bindings introduced by inline tensor declarations (aka. punning). These
          bindings are discharged with the whole [%cd] extension scope in scope. *)
  typ : expr_type;
  slot : projections_slot;
  expr : expression;
      (** Depending on {!field-typ}, of type:
          - if [Code]: [Assignments.comp];
          - if [Array | Merge_value | Value_of_tensor]: [Tnode.t];
          - if [Merge_grad | Grad_of_tensor]: [Tnode.t option];
          - if [Tensor | Unknown | No_grad_tensor_intro]: [Tensor.t]. *)
  array_opt_of_code : expression option;
      (** Of type: [Tnode.t]. It keeps track of which tensor node to use when [typ] is [Code] but
          the result is used in an array context. *)
}

type array_setup = {
  vb : value_binding option;
      (** This binding is only generated for a tensor expression that is not an identifier, since
          recomputing the expression (when copied) would generate a fresh tensor. It is discharged
          when an assignment is built. *)
  slot : projections_slot;
  filler_typ : expr_type;
  fwd_code_or_noop : expression option;  (** Of type: [Assignments.comp]. *)
  array_opt : expression;  (** Of type: if [slot = LHS] then [Tnode.t] else [Assignments.buffer]. *)
  tensor : expression option;  (** Of type: [Tensor.t]. *)
  pun_hint_tnode : (expression * bool) option;
      (** Of type: [string list]. The tnode, if any, whose label the relevant punned no-gradient
          tensor should incorporate in its label. The bool denotes whether this is a preferred (high
          quality) guess. *)
}

let make_vb ~loc ~name ~name_expr ~hint_label =
  let pat = A.Pat.var ~loc { loc = name_expr.pexp_loc; txt = name } in
  let v =
    match hint_label with
    | None -> [%expr NTDSL.term ~label:[ [%e name_expr] ] ()]
    | Some hint_label -> [%expr NTDSL.term ~label:([%e name_expr] :: [%e hint_label]) ()]
  in
  let vb = A.Vb.mk ~loc pat v in
  vb

let reduce_embs_arr ~loc (rs : array_setup list) =
  List.filter_map rs ~f:(fun hs -> hs.fwd_code_or_noop)
  |> List.reduce ~f:(fun embs comp -> [%expr Base.Set.union [%e embs] [%e comp].embedded_nodes])

(** The expression argument is of type: [Assignments.t]. *)
let assignment ~punned ~lhs ~rhses body =
  let setups = lhs :: rhses in
  let loc = body.pexp_loc in
  let forward_args = List.filter_map setups ~f:(fun { fwd_code_or_noop; _ } -> fwd_code_or_noop) in
  let vbs, body =
    match lhs.filler_typ with
    | No_grad_tensor_intro { name; name_expr } -> (
        let good_hints, bad_hints =
          List.partition_tf ~f:snd @@ List.filter_map rhses ~f:(fun sup -> sup.pun_hint_tnode)
        in
        let hint_data = Option.first_some (List.hd good_hints) (List.hd bad_hints) in
        let hint_label = Option.map ~f:fst hint_data in
        let vbs = Map.singleton (module String) name @@ make_vb ~loc ~name ~name_expr ~hint_label in
        match hint_data with
        | None -> (vbs, body)
        | Some data -> (
            match Hashtbl.add punned ~key:name ~data with
            | `Ok -> (vbs, body)
            | `Duplicate ->
                ( no_vbs,
                  Ast_builder.Default.pexp_extension ~loc
                  @@ Location.error_extensionf ~loc
                       "ppx_ocannl %%cd: duplicate inline declaration of no-gradient tensor %s" name
                )))
    | _ -> (no_vbs, body)
  in
  let body =
    if Option.is_some lhs.vb then
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "ppx_ocannl %%cd: the assigned-to position cannot be an expression building a new tensor"
    else body
  in
  let tensor_vbs = List.filter_map rhses ~f:(fun rhs -> rhs.vb) in
  let body =
    [%expr { asgns = [%e body]; embedded_nodes = Base.Set.empty (module Arrayjit.Tnode) }]
  in
  let comps =
    List.fold (body :: List.rev forward_args) ~init:[%expr []] ~f:(fun xs x ->
        [%expr [%e x] :: [%e xs]])
  in
  let expr = [%expr Arrayjit.Assignments.sequence [%e comps]] in
  let expr =
    if List.is_empty tensor_vbs then expr else A.Exp.let_ ~loc Nonrecursive tensor_vbs expr
  in
  { vbs; typ = Code; slot = Nonslot; expr; array_opt_of_code = Some lhs.array_opt }

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
      @@ Location.error_extensionf ~loc
           "ppx_ocannl %%cd: insufficient slot filler information at %s %s" debug
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
      @@ Location.error_extensionf ~loc
           "ppx_ocannl %%cd: insufficient slot filler information at %s %s" debug
           "(incorporate one of: v, v1, v2, g, g1, g2, lhs, rhs, rhs1, rhs2)"

let guess_pun_hint ~punned ~bad_pun_hints filler_typ filler =
  let loc = filler.pexp_loc in
  let hint = [%expr [%e filler].Arrayjit.Tnode.label] in
  match (filler_typ, filler) with
  | Code, _ -> None
  | _, { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ } when Set.mem bad_pun_hints name ->
      None
  | Array, _ -> Some (hint, false)
  | (Tensor | Unknown), { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }
    when Hashtbl.mem punned name ->
      Hashtbl.find punned name
  | (Tensor | Unknown), { pexp_desc = Pexp_ident _; _ } -> Some (hint, true)
  | (Tensor | Unknown), _ -> Some (hint, false)
  | ( ( Value_of_tensor { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }
      | Grad_of_tensor { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }
      | Merge_value { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }
      | Merge_grad { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ } ),
      _ )
    when Set.mem bad_pun_hints name ->
      None
  | ( ( Value_of_tensor { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }
      | Grad_of_tensor { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }
      | Merge_value { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }
      | Merge_grad { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ } ),
      _ )
    when Hashtbl.mem punned name ->
      Hashtbl.find punned name
  | (Value_of_tensor t | Grad_of_tensor t | Merge_value t | Merge_grad t), _ -> (
      let hint = [%expr [%e t].Tensor.value.Arrayjit.Tnode.label] in
      match t with { pexp_desc = Pexp_ident _; _ } -> Some (hint, true) | _ -> Some (hint, false))
  | No_grad_tensor_intro { name; _ }, _ -> Hashtbl.find punned name

let empty_tns ~loc = [%expr Base.Set.empty (module Arrayjit.Tnode)]

let empty_comp ~loc =
  [%expr { asgns = Arrayjit.Assignments.Noop; embedded_nodes = [%e empty_tns ~loc] }]

let setup_array ~punned ~bad_pun_hints ~is_lhs
    { typ = filler_typ; slot; expr = filler; vbs; array_opt_of_code } =
  assert (Map.is_empty vbs);
  let loc = filler.pexp_loc in
  let opt_buffer tn =
    if is_lhs then [%expr Some [%e tn]] else [%expr Some (Arrayjit.Assignments.Node [%e tn])]
  in
  let buffer opt_tn =
    if is_lhs then opt_tn
    else [%expr Option.map [%e opt_tn] ~f:(fun tn -> Arrayjit.Assignments.Node tn)]
  in
  let pun_hint_tnode = guess_pun_hint ~punned ~bad_pun_hints filler_typ filler in
  let default_setup =
    {
      vb = None;
      fwd_code_or_noop = None;
      filler_typ;
      slot;
      array_opt = opt_buffer [%expr [%e filler].Tensor.value];
      tensor = None;
      pun_hint_tnode;
    }
  in
  match filler_typ with
  | No_grad_tensor_intro _ when not is_lhs ->
      {
        default_setup with
        array_opt =
          Ast_builder.Default.pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "ppx_ocannl %%cd: punning is only allowed in the assigned-to position";
      }
  | (Tensor | Unknown) when match filler with { pexp_desc = Pexp_ident _; _ } -> true | _ -> false
    ->
      let t = filler in
      let fwd_code_or_noop =
        Some
          [%expr
            if Tensor.is_fwd_root [%e t] then (
              Tensor.remove_fwd_root [%e t];
              [%e t].Tensor.forward)
            else [%e empty_comp ~loc]]
      in
      { default_setup with fwd_code_or_noop; tensor = Some t }
  | Value_of_tensor ({ pexp_desc = Pexp_ident _; _ } as t) ->
      let fwd_code_or_noop =
        Some
          [%expr
            if Tensor.is_fwd_root [%e t] then (
              Tensor.remove_fwd_root [%e t];
              [%e t].Tensor.forward)
            else [%e empty_comp ~loc]]
      in
      {
        default_setup with
        fwd_code_or_noop;
        array_opt = opt_buffer [%expr [%e t].Tensor.value];
        tensor = Some t;
      }
  | Value_of_tensor t ->
      {
        default_setup with
        array_opt =
          Ast_builder.Default.pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "ppx_ocannl %%cd: the <tensor>.value notation is only supported when <tensor> is an \
                identifier";
        tensor = Some t;
      }
  | Tensor | Unknown ->
      (* Need to bind the expression computing the tensor so we don't recompute it. *)
      let v =
        match slot with
        | LHS -> [%pat? nondiff__lhs]
        | RHS1 -> [%pat? nondiff__rhs1]
        | RHS2 -> [%pat? nondiff__rhs2]
        | Nonslot | Undet -> [%pat? nondiff__tensor]
      in
      let t = pat2expr v in
      let vb = Some (A.Vb.mk ~loc v filler) in
      let fwd_code_or_noop =
        Some
          [%expr
            if Tensor.is_fwd_root [%e t] then (
              Tensor.remove_fwd_root [%e t];
              [%e t].Tensor.forward)
            else [%e empty_comp ~loc]]
      in
      {
        default_setup with
        vb;
        fwd_code_or_noop;
        array_opt = opt_buffer [%expr [%e t].Tensor.value];
        tensor = Some t;
      }
  | No_grad_tensor_intro _ ->
      (* Inline tensors are guaranteed to be leaf tensors, so they don't have forward code, but they
         are embedded. *)
      let fwd_code_or_noop =
        Some
          [%expr
            Arrayjit.Assignments.
              {
                asgns = Noop;
                embedded_nodes = Base.Set.singleton (module Arrayjit.Tnode) [%e filler].Tensor.value;
              }]
      in
      { default_setup with fwd_code_or_noop; tensor = Some filler }
  | Code when Option.is_none array_opt_of_code ->
      {
        default_setup with
        fwd_code_or_noop = Some filler;
        array_opt =
          Ast_builder.Default.pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "ppx_ocannl %%cd: could not determine a lead array of provided code";
      }
  | Code ->
      {
        default_setup with
        fwd_code_or_noop = Some filler;
        array_opt = buffer (Option.value_exn array_opt_of_code);
      }
  | Array -> { default_setup with array_opt = opt_buffer filler }
  | Grad_of_tensor ({ pexp_desc = Pexp_ident _; _ } as t) ->
      { default_setup with array_opt = buffer filler; tensor = Some t }
  | Grad_of_tensor t ->
      {
        default_setup with
        array_opt =
          Ast_builder.Default.pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "ppx_ocannl %%cd: the <tensor>.grad notation is only supported when <tensor> is an \
                identifier";
        tensor = Some t;
      }
  | (Merge_value _ | Merge_grad _) when is_lhs ->
      {
        default_setup with
        array_opt =
          Ast_builder.Default.pexp_extension ~loc
          @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: merge buffers cannot be assigned to";
      }
  | Merge_value t ->
      { default_setup with array_opt = [%expr Some (Merge_buffer [%e filler])]; tensor = Some t }
  | Merge_grad t ->
      {
        default_setup with
        array_opt =
          [%expr Option.map [%e filler] ~f:(fun tn -> Arrayjit.Assignments.Merge_buffer tn)];
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
             "ppx_ocannl %%cd: cannot use `~logic` (infer shapes) for arrays, use tensors or \
              `.value` or `.grad` notation",
        [%expr false],
        [%expr false] )

let reduce_res_vbs rs = reduce_vbss @@ List.map rs ~f:(fun r -> r.vbs)

let translate (expr : expression) : result =
  let punned = Hashtbl.create (module String) in
  let rec transl ~bad_pun_hints ~proj_in_scope (expr : expression) : result =
    let loc = expr.pexp_loc in
    let default_result =
      { vbs = no_vbs; typ = Tensor; slot = Undet; expr; array_opt_of_code = None }
    in
    let loop = transl ~bad_pun_hints in
    let process_assign_binop ~accu_op ~lhs ~bin_op ~rhs1 ~rhs2 ?projections ~proj_in_scope () =
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l =
        setup_array ~punned ~bad_pun_hints ~is_lhs:true @@ loop ~proj_in_scope:true lhs
      in
      let _, bin_op = binary_op bin_op in
      let setup_r1 = setup_array ~punned ~bad_pun_hints ~is_lhs:false @@ loop ~proj_in_scope rhs1 in
      let setup_r2 = setup_array ~punned ~bad_pun_hints ~is_lhs:false @@ loop ~proj_in_scope rhs2 in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let projections =
        match projections with
        | Some prjs -> prjs
        | None ->
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
                           ( "ppx_cd " ^ [%e expr2string_or_empty accu_op] ^ " "
                             ^ [%e expr2string_or_empty bin_op],
                             Arrayjit.Indexing.unique_debug_id () )
                           :: p.debug_info.trace;
                       };
                   })]
      in
      (* TODO: might be better to treat missing [rhs1, rhs2] as zeros or errors rather than eliding
         the code. *)
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
      assignment ~punned ~lhs:setup_l ~rhses:[ setup_r1; setup_r2 ] body
    in
    let process_assign_unop ~accu_op ~lhs ~un_op ~rhs ?projections ~proj_in_scope () =
      let initialize_neutral, accu_op = assignment_op accu_op in
      (* FIXME: I think this ignores the slot information here! Just assuming [projections] is
         as-should-be, but that's not consistent with omitting the projections arg (assuming it
         comes from the context). *)
      let setup_l = setup_array ~punned ~bad_pun_hints ~is_lhs:true @@ loop ~proj_in_scope lhs in
      let setup_r = setup_array ~punned ~bad_pun_hints ~is_lhs:false @@ loop ~proj_in_scope rhs in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let projections =
        match projections with
        | Some prjs -> prjs
        | None ->
            let lhs_dims = project_p_dims "LHS" lhs.pexp_loc setup_l.slot in
            let rhs1_dims = project_p_dims "RHS1" lhs.pexp_loc setup_r.slot in
            let project_lhs = project_p_slot "LHS" lhs.pexp_loc setup_l.slot in
            let project_rhs1 = project_p_slot "RHS1" rhs.pexp_loc setup_r.slot in
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
                           ( "ppx_cd " ^ [%e expr2string_or_empty accu_op] ^ " "
                             ^ [%e expr2string_or_empty un_op],
                             Arrayjit.Indexing.unique_debug_id () )
                           :: p.debug_info.trace;
                       };
                   })]
      in
      (* TODO: might be better to treat missing [rhs] as zeros or errors rather than eliding the
         code. *)
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
      assignment ~punned ~lhs:setup_l ~rhses:[ setup_r ] body
    in
    let process_raw_binop ~accu_op ~lhs ~bin_op ~rhs1 ~rhs2 ~logic =
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l = setup_array ~punned ~bad_pun_hints ~is_lhs:true @@ loop ~proj_in_scope lhs in
      let setup_r1 = setup_array ~punned ~bad_pun_hints ~is_lhs:false @@ loop ~proj_in_scope rhs1 in
      let setup_r2 = setup_array ~punned ~bad_pun_hints ~is_lhs:false @@ loop ~proj_in_scope rhs2 in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let t_expr, lhs_is_grad, _ = args_for ~loc setup_l in
      let t1_expr, rhs1_is_grad, rhs1_is_merge = args_for ~loc setup_r1 in
      let t2_expr, rhs2_is_grad, rhs2_is_merge = args_for ~loc setup_r2 in
      let body =
        [%expr
          Tensor.raw_binop ~initialize_neutral:[%e initialize_neutral] ~accum:[%e accu_op]
            ~t:[%e t_expr] ~lhs_is_grad:[%e lhs_is_grad] ~op:[%e bin_op] ~t1:[%e t1_expr]
            ~rhs1_is_grad:[%e rhs1_is_grad] ~rhs1_is_merge:[%e rhs1_is_merge] ~t2:[%e t2_expr]
            ~rhs2_is_grad:[%e rhs2_is_grad] ~rhs2_is_merge:[%e rhs2_is_merge] ~logic:[%e logic]]
      in
      assignment ~punned ~lhs:setup_l ~rhses:[ setup_r1; setup_r2 ] body
    in
    let process_raw_unop ~accu_op ~lhs ~un_op ~rhs ~logic =
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l = setup_array ~punned ~bad_pun_hints ~is_lhs:true @@ loop ~proj_in_scope lhs in
      let setup_r = setup_array ~punned ~bad_pun_hints ~is_lhs:false @@ loop ~proj_in_scope rhs in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let t_expr, lhs_is_grad, _ = args_for ~loc setup_l in
      let t1_expr, rhs_is_grad, rhs_is_merge = args_for ~loc setup_r in
      let body =
        [%expr
          Tensor.raw_unop ~initialize_neutral:[%e initialize_neutral] ~accum:[%e accu_op]
            ~t:[%e t_expr] ~lhs_is_grad:[%e lhs_is_grad] ~op:[%e un_op] ~t1:[%e t1_expr]
            ~rhs_is_grad:[%e rhs_is_grad] ~rhs_is_merge:[%e rhs_is_merge] ~logic:[%e logic]]
      in
      assignment ~punned ~lhs:setup_l ~rhses:[ setup_r ] body
    in
    match expr with
    | { pexp_desc = Pexp_constant (Pconst_float _); _ } ->
        { default_result with expr = [%expr NTDSL.number [%e expr]] }
    | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
        { default_result with expr = [%expr NTDSL.number (Float.of_int [%e expr])] }
    | [%expr
        [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
          [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
        let axis =
          Ast_helper.Exp.constant ~loc:pexp_loc (Pconst_string (String.of_char ch, pexp_loc, None))
        in
        { default_result with expr = [%expr NTDSL.number ~axis_label:[%e axis] [%e f]] }
    | [%expr
        [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
          [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
        let axis =
          Ast_helper.Exp.constant ~loc:pexp_loc (Pconst_string (String.of_char ch, pexp_loc, None))
        in
        {
          default_result with
          expr = [%expr NTDSL.number ~axis_label:[%e axis] (Float.of_int [%e i])];
        }
    | { pexp_desc = Pexp_constant (Pconst_string (name, str_loc, _)); _ } ->
        {
          default_result with
          typ = No_grad_tensor_intro { name; name_expr = expr };
          expr = A.Exp.ident ~loc:str_loc { txt = Lident name; loc = str_loc };
        }
    | { pexp_desc = Pexp_array _; _ }
    | { pexp_desc = Pexp_construct ({ txt = Lident "::"; _ }, _); _ } ->
        { default_result with expr = ndarray_op expr }
    | { pexp_desc = Pexp_ident { txt = Lident ("v" | "lhs"); _ }; _ } ->
        { default_result with typ = Array; slot = LHS }
    | { pexp_desc = Pexp_ident { txt = Lident "g"; _ }; _ } ->
        { default_result with typ = Array; slot = LHS }
    | { pexp_desc = Pexp_ident { txt = Lident "rhs1"; _ }; _ } ->
        { default_result with typ = Array; slot = RHS1 }
    | { pexp_desc = Pexp_ident { txt = Lident "t1"; _ }; _ } ->
        { default_result with slot = RHS1; expr }
    | { pexp_desc = Pexp_ident { txt = Lident "v1"; _ }; _ } ->
        { default_result with typ = Array; slot = RHS1; expr = [%expr t1.Tensor.value] }
    | { pexp_desc = Pexp_ident { txt = Lident "g1"; _ }; _ } ->
        {
          default_result with
          typ = Grad_of_tensor [%expr t1];
          slot = RHS1;
          expr = [%expr Option.map t1.Tensor.diff ~f:(fun d -> d.Tensor.grad)];
        }
    | { pexp_desc = Pexp_ident { txt = Lident "rhs2"; _ }; _ } ->
        { default_result with typ = Array; slot = RHS2 }
    | { pexp_desc = Pexp_ident { txt = Lident "t2"; _ }; _ } ->
        { default_result with typ = Tensor; slot = RHS2 }
    | { pexp_desc = Pexp_ident { txt = Lident "v2"; _ }; _ } ->
        { default_result with typ = Array; slot = RHS2; expr = [%expr t2.Tensor.value] }
    | { pexp_desc = Pexp_ident { txt = Lident "g2"; _ }; _ } ->
        {
          default_result with
          typ = Grad_of_tensor [%expr t2];
          slot = RHS2;
          expr = [%expr Option.map t2.Tensor.diff ~f:(fun d -> d.Tensor.grad)];
        }
    | { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ } when is_operator op_ident ->
        default_result
    | [%expr [%e? expr1] **. [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
        (* FIXME: `**.` should take a tensor and require that it's a literal. *)
        (* We need to hardcode these two patterns to prevent the numbers from being converted to tensors. *)
        let res1 = loop ~proj_in_scope expr1 in
        {
          res1 with
          typ = Tensor;
          expr = [%expr NTDSL.O.( **. ) [%e res1.expr] (Float.of_int [%e i])];
        }
    | [%expr [%e? expr1] **. [%e? expr2]] ->
        let res1 = loop ~proj_in_scope expr1 in
        { res1 with typ = Tensor; expr = [%expr NTDSL.O.( **. ) [%e res1.expr] [%e expr2]] }
    | [%expr
        [%e? expr1]
        *+ [%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ } as spec]
             [%e? expr2]]
      when String.contains spec_str '>' ->
        let res1 = loop ~proj_in_scope expr1 in
        let res2 = loop ~proj_in_scope expr2 in
        let slot =
          Option.value ~default:Undet
          @@ List.find ~f:(function Undet -> false | _ -> true) [ res1.slot; res2.slot ]
        in
        {
          vbs = reduce_vbss [ res1.vbs; res2.vbs ];
          typ = Tensor;
          slot;
          expr = [%expr NTDSL.einsum [%e spec] [%e res1.expr] [%e res2.expr]];
          array_opt_of_code = None;
        }
    | [%expr
        [%e? expr1]
        ++ [%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ } as spec]]
      when String.contains spec_str '>' ->
        let res1 = loop ~proj_in_scope expr1 in
        { res1 with typ = Tensor; expr = [%expr NTDSL.einsum1 [%e spec] [%e res1.expr]] }
    | [%expr [%e? expr1].grad] -> (
        let res1 = loop ~proj_in_scope expr1 in
        match res1.typ with
        | Unknown | Tensor | No_grad_tensor_intro _ ->
            {
              res1 with
              typ = Grad_of_tensor expr1;
              expr = [%expr Option.map [%e res1.expr].Tensor.diff ~f:(fun d -> d.Tensor.grad)];
              (* It's never a good idea to embed backprop code outside of a proper backprop pass. *)
            }
        | Merge_value _ ->
            {
              res1 with
              typ = Merge_grad expr1;
              expr =
                Ast_builder.Default.pexp_extension ~loc
                @@ Location.error_extensionf ~loc
                     "ppx_ocannl %%cd: write .grad.merge instead of .merge.grad";
            }
        | Code | Array | Value_of_tensor _ | Grad_of_tensor _ | Merge_grad _ ->
            {
              res1 with
              typ = Array;
              expr =
                Ast_builder.Default.pexp_extension ~loc
                @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: only tensors have a gradient";
            })
    | [%expr [%e? expr1].value] -> (
        let res1 = loop ~proj_in_scope expr1 in
        (* TODO: maybe this is too permissive? E.g. [t1.grad.value] is accepted. *)
        match res1.typ with
        | Unknown | Tensor | No_grad_tensor_intro _ ->
            {
              res1 with
              typ = Value_of_tensor res1.expr;
              expr = [%expr [%e res1.expr].Tensor.value];
            }
        | Code ->
            {
              default_result with
              typ = Array;
              slot = res1.slot;
              expr =
                Ast_builder.Default.pexp_extension ~loc
                @@ Location.error_extensionf ~loc
                     "ppx_ocannl %%cd: <code>.value notation not supported when <code> is not a \
                      tensor";
            }
        | Array | Value_of_tensor _ | Grad_of_tensor _ | Merge_value _ | Merge_grad _ -> res1)
    | [%expr [%e? expr1].merge] -> (
        let res1 = loop ~proj_in_scope expr1 in
        match res1.typ with
        | Unknown | Tensor | No_grad_tensor_intro _ ->
            { res1 with typ = Merge_value res1.expr; expr = [%expr [%e res1.expr].Tensor.value] }
        | Value_of_tensor t ->
            { res1 with typ = Merge_value t; expr = [%expr [%e res1.expr].Tensor.value] }
        | Array | Code ->
            {
              res1 with
              typ = Array;
              expr =
                Ast_builder.Default.pexp_extension ~loc
                @@ Location.error_extensionf ~loc
                     "ppx_ocannl %%cd: only tensor nodes (e.g. `.value` or `.grad`) can be merged";
            }
        | Grad_of_tensor t -> { res1 with vbs = no_vbs; typ = Merge_grad t }
        | Merge_value _ | Merge_grad _ ->
            {
              res1 with
              expr =
                Ast_builder.Default.pexp_extension ~loc
                @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: repeated .merge not allowed";
            })
    | [%expr
        ~~([%e? { pexp_desc = Pexp_constant (Pconst_string _); _ } as comment];
           [%e? expr2])] ->
        let res2 = loop ~proj_in_scope expr2 in
        {
          res2 with
          expr =
            [%expr
              let __comment_block = [%e res2.expr] in
              {
                Arrayjit.Assignments.asgns =
                  Arrayjit.Assignments.Block_comment
                    ([%e comment], __comment_block.Arrayjit.Assignments.asgns);
                embedded_nodes = __comment_block.Arrayjit.Assignments.embedded_nodes;
              }];
        }
    | [%expr
        ~~([%e? { pexp_desc = Pexp_apply (expr, exprs); pexp_loc; _ }];
           [%e? expr2])] ->
        let elements =
          expr :: List.map ~f:snd exprs
          |> List.map ~f:(function
               | { pexp_desc = Pexp_constant (Pconst_string _); _ } as s -> s
               | [%expr [%e? t].value] -> [%expr Arrayjit.Tnode.debug_name [%e t].value]
               | [%expr [%e? t].grad] -> [%expr Arrayjit.Tnode.debug_name [%e t].value ^ ".grad"]
               | t -> [%expr Arrayjit.Tnode.debug_name [%e t].value])
        in
        let res2 = loop ~proj_in_scope expr2 in
        {
          res2 with
          expr =
            [%expr
              let __comment_block = [%e res2.expr] in
              {
                Arrayjit.Assignments.asgns =
                  Arrayjit.Assignments.Block_comment
                    ( String.concat_array ~sep:" " [%e Ast_helper.Exp.array ~loc:pexp_loc elements],
                      __comment_block.Arrayjit.Assignments.asgns );
                embedded_nodes = __comment_block.Arrayjit.Assignments.embedded_nodes;
              }];
        }
    | [%expr
        [%e? accu_op]
          [%e? lhs]
          ([%e? bin_op] [%e? rhs1] ([%e? rhs2] ~projections:[%e? projections]))] ->
        process_assign_binop ~accu_op ~lhs ~bin_op ~rhs1 ~rhs2 ~projections ~proj_in_scope:true ()
    | [%expr [%e? accu_op] [%e? lhs] (([%e? un_op] [%e? rhs]) ~projections:[%e? projections])]
    | [%expr [%e? accu_op] [%e? lhs] ([%e? un_op] ([%e? rhs] ~projections:[%e? projections]))] ->
        let _, un_op = unary_op un_op in
        (* Handle both un_op priority levels -- where application binds tighter and less tight. *)
        process_assign_unop ~accu_op ~lhs ~un_op ~rhs ~projections ~proj_in_scope:true ()
    | [%expr [%e? accu_op] [%e? lhs] ([%e? rhs] ~projections:[%e? projections])] ->
        process_assign_unop ~accu_op ~lhs ~un_op:[%expr Arrayjit.Ops.Identity] ~rhs ~projections
          ~proj_in_scope:true ()
    | [%expr
        [%e? accu_op]
          [%e? lhs]
          ([%e? bin_op]
             [%e? rhs1]
             ([%e? rhs2]
                ~logic:
                  [%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ } as logic]))]
      ->
        let logic =
          let loc = s_loc in
          if String.equal spec "." then [%expr Shape.Pointwise_bin]
          else if String.equal spec "@" then [%expr Shape.Compose]
          else [%expr Shape.Einsum [%e logic]]
        in
        let _, bin_op = binary_op bin_op in
        process_raw_binop ~accu_op ~lhs ~bin_op ~rhs1 ~rhs2 ~logic
    | [%expr
        [%e? accu_op]
          [%e? lhs]
          (([%e? un_op] [%e? rhs])
             ~logic:[%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ } as logic])]
    | [%expr
        [%e? accu_op]
          [%e? lhs]
          ([%e? un_op]
             ([%e? rhs]
                ~logic:
                  [%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ } as logic]))]
      ->
        (* Handle both un_op priority levels -- where application binds tighter and less tight. *)
        let logic =
          let loc = s_loc in
          if String.equal spec "." then [%expr Shape.Pointwise_un]
          else if String.equal spec "T" then [%expr Shape.Transpose]
          else [%expr Shape.Permute [%e logic]]
        in
        let _, un_op = unary_op un_op in
        process_raw_unop ~accu_op ~lhs ~un_op ~rhs ~logic
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_ident; _ }; _ } as accu_op]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident binop_ident; _ }; _ } as bin_op]
             [%e? rhs1]
             [%e? rhs2])]
      when is_assignment accu_ident && is_binary_op binop_ident && proj_in_scope ->
        process_assign_binop ~accu_op ~lhs ~bin_op ~rhs1 ~rhs2 ~proj_in_scope ()
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_ident; _ }; _ } as accu_op]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident unop_ident; _ }; _ } as un_op] [%e? rhs])]
      when is_assignment accu_ident && is_unary_op unop_ident && proj_in_scope ->
        let _, un_op = unary_op un_op in
        process_assign_unop ~accu_op ~lhs ~un_op ~rhs ~proj_in_scope ()
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ } as accu_op]
          [%e? lhs]
          [%e? rhs]]
      when is_assignment op_ident && proj_in_scope ->
        process_assign_unop ~accu_op ~lhs ~un_op:[%expr Arrayjit.Ops.Identity] ~rhs ~proj_in_scope
          ()
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_ident; _ }; _ } as accu_op]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident binop_ident; _ }; _ } as bin_op]
             [%e? rhs1]
             [%e? rhs2])]
      when is_assignment accu_ident && is_binary_op binop_ident ->
        let logic, bin_op = binary_op bin_op in
        process_raw_binop ~accu_op ~lhs ~bin_op ~rhs1 ~rhs2 ~logic
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_ident; _ }; _ } as accu_op]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident unop_ident; _ }; _ } as un_op] [%e? rhs])]
      when is_assignment accu_ident && is_unary_op unop_ident ->
        let logic, un_op = unary_op un_op in
        process_raw_unop ~accu_op ~lhs ~un_op ~rhs ~logic
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ } as accu_op]
          [%e? lhs]
          [%e? rhs]]
      when is_assignment op_ident ->
        process_raw_unop ~accu_op ~lhs ~un_op:[%expr Arrayjit.Ops.Identity] ~rhs
          ~logic:[%expr Shape.Pointwise_un]
    | [%expr [%e? expr1] [%e? expr2] [%e? expr3]] ->
        let res1 = loop ~proj_in_scope expr1 in
        let res2 = loop ~proj_in_scope expr2 in
        let res3 = loop ~proj_in_scope expr3 in
        let slot =
          Option.value ~default:Undet
          @@ List.find
               ~f:(function Undet -> false | _ -> true)
               [ res1.slot; res2.slot; res3.slot ]
        in
        {
          vbs = reduce_vbss [ res1.vbs; res2.vbs; res3.vbs ];
          typ = res1.typ;
          slot;
          expr = [%expr [%e res1.expr] [%e res2.expr] [%e res3.expr]];
          array_opt_of_code = None;
        }
    | [%expr [%e? expr1] [%e? expr2]] ->
        let res1 = loop ~proj_in_scope expr1 in
        let res2 = loop ~proj_in_scope expr2 in
        let slot =
          Option.value ~default:Undet
          @@ List.find ~f:(function Undet -> false | _ -> true) [ res1.slot; res2.slot ]
        in
        {
          vbs = reduce_vbss [ res1.vbs; res2.vbs ];
          typ = res1.typ;
          slot;
          expr = [%expr [%e res1.expr] [%e res2.expr]];
          array_opt_of_code = None;
        }
    | { pexp_desc = Pexp_fun ((arg_label : arg_label), arg, pat, expr1); _ } as expr ->
        let proj_in_scope =
          proj_in_scope
          ||
          match arg_label with
          | (Labelled s | Optional s) when String.equal s "projections" -> true
          | _ -> false
        in
        let bad_pun_hints = Set.union bad_pun_hints @@ collect_pat_idents pat in
        let res1 = transl ~bad_pun_hints ~proj_in_scope expr1 in
        { res1 with expr = { expr with pexp_desc = Pexp_fun (arg_label, arg, pat, res1.expr) } }
    | [%expr
        while [%e? _test_expr] do
          [%e? _body]
        done] ->
        {
          default_result with
          typ = Code;
          slot = Nonslot;
          expr =
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%cd: while: low-level code embeddings not supported yet";
        }
    | [%expr
        for [%p? _pat] = [%e? _init] to [%e? _final] do
          [%e? _body_expr]
        done] ->
        {
          default_result with
          typ = Code;
          slot = Nonslot;
          expr =
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%cd: for-to: low-level code embeddings not supported yet";
        }
    | [%expr
        for [%p? _pat] = [%e? _init] downto [%e? _final] do
          [%e? _body_expr]
        done] ->
        {
          default_result with
          typ = Code;
          slot = Nonslot;
          expr =
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%cd: for-downto: low-level code embeddings not supported yet";
        }
    | [%expr
        [%e? expr1];
        [%e? expr2]] ->
        let res1 = loop ~proj_in_scope expr1 in
        let res2 = loop ~proj_in_scope expr2 in
        {
          vbs = reduce_vbss [ res1.vbs; res2.vbs ];
          typ = Code;
          slot = Nonslot;
          expr = [%expr Arrayjit.Assignments.sequence [ [%e res1.expr]; [%e res2.expr] ]];
          array_opt_of_code = res2.array_opt_of_code;
        }
    | [%expr if [%e? expr1] then [%e? expr2] else [%e? expr3]] ->
        let res2 = loop ~proj_in_scope expr2 in
        let res3 = loop ~proj_in_scope expr3 in
        let typ = if is_unknown res2.typ then res3.typ else res2.typ in
        let slot =
          Option.value ~default:Undet
          @@ List.find ~f:(function Undet -> false | _ -> true) [ res2.slot; res3.slot ]
        in
        {
          vbs = reduce_vbss [ res2.vbs; res3.vbs ];
          typ;
          slot;
          expr = [%expr if [%e expr1] then [%e res2.expr] else [%e res3.expr]];
          array_opt_of_code = None;
        }
    | [%expr if [%e? expr1] then [%e? expr2]] ->
        let res2 = loop ~proj_in_scope expr2 in
        {
          vbs = res2.vbs;
          typ = Code;
          slot = Nonslot;
          expr = [%expr if [%e expr1] then [%e res2.expr] else Arrayjit.Assignments.empty_comp];
          array_opt_of_code = res2.array_opt_of_code;
        }
    | { pexp_desc = Pexp_match (expr1, cases); _ } ->
        let fields, cases =
          List.unzip
          @@ List.map cases ~f:(fun ({ pc_rhs; _ } as c) ->
                 let res = loop ~proj_in_scope pc_rhs in
                 ((res.vbs, res.typ, res.slot), { c with pc_rhs = res.expr }))
        in
        let vbss, typs, slots = List.unzip3 fields in
        let typ = Option.value ~default:Unknown @@ List.find typs ~f:(Fn.non is_unknown) in
        let slot =
          Option.value ~default:Undet @@ List.find ~f:(function Undet -> false | _ -> true) slots
        in
        {
          vbs = reduce_vbss vbss;
          typ;
          slot;
          expr = { expr with pexp_desc = Pexp_match (expr1, cases) };
          array_opt_of_code = None;
        }
    | { pexp_desc = Pexp_let (_recflag, _bindings, _body); _ } ->
        (* TODO(80): to properly support local bindings, we need to collect the type environment. *)
        {
          default_result with
          typ = Unknown;
          expr =
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%cd: let-in: local let-bindings not implemented yet";
        }
    (* let bindings = List.map bindings ~f:(fun binding -> {binding with pvb_expr=loop
       binding.pvb_expr}) in {expr with pexp_desc=Pexp_let (recflag, bindings, loop body)} *)
    | { pexp_desc = Pexp_open (decl, expr1); _ } ->
        let res1 = loop ~proj_in_scope expr1 in
        { res1 with expr = { expr with pexp_desc = Pexp_open (decl, res1.expr) } }
    | { pexp_desc = Pexp_letmodule (name, module_expr, expr1); _ } ->
        let res1 = loop ~proj_in_scope expr1 in
        { res1 with expr = { expr with pexp_desc = Pexp_letmodule (name, module_expr, res1.expr) } }
    | { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ } when is_operator op_ident ->
        { default_result with typ = Unknown; expr = [%expr [%e expr]] }
    | _ -> { default_result with typ = Unknown }
  in
  transl ~proj_in_scope:false ~bad_pun_hints:(Set.empty (module String)) expr

let translate ?ident_label expr =
  let res = translate expr in
  let loc = res.expr.pexp_loc in
  let expr = res.expr in
  ( res.vbs,
    match ident_label with
    | Some [%pat? _] ->
        [%expr
          Tensor.with_unchanged_roots ~f:(fun () ->
              let open! NTDSL.O in
              [%e expr])]
    | _ ->
        [%expr
          let open! NTDSL.O in
          [%e expr]] )

let expr_expander ~loc ~path = expr_expander_with_punning translate ~loc ~path
let str_expander ~loc ~path = str_expander_with_punning translate ~loc ~path
