open Base
open Ppxlib
open Ppx_shared
module A = Ppxlib_ast.Ast_helper

type expr_type =
  | Code of { is_commented : bool }
  | Array
  | Value_of_tensor of expression
  | Grad_of_tensor of expression
  | Tensor
  | Unknown
  | Merge_value of expression
  | Merge_grad of expression
  | No_grad_tensor_intro of {
      name : string;
      name_expr : expression;
      extra_args : (string * expression) list;
    }
  | Function

let is_unknown = function Unknown -> true | _ -> false

type projections_slot = LHS | RHS1 | RHS2 | RHS3 | Scalar | Nonslot | Undet
[@@deriving variants]

let equal_projections_slot a b =
  match (a, b) with
  | LHS, LHS | RHS1, RHS1 | RHS2, RHS2 | RHS3, RHS3 | Scalar, Scalar | Nonslot, Nonslot | Undet, Undet
    ->
      true
  | _ -> false

let slot_to_string = function
  | LHS -> "lhs"
  | RHS1 -> "rhs1"
  | RHS2 -> "rhs2"
  | RHS3 -> "rhs3"
  | Scalar -> "scalar"
  | Nonslot -> "nonslot"
  | Undet -> "?"

(** Generate a slot permutation suffix for projections_debug when slots are permuted.
    Returns empty string for canonical assignment (lhs←lhs, rhs1←rhs1, etc.),
    otherwise returns something like " [lhs←rhs1, rhs1←lhs]". *)
let slot_permutation_suffix ~lhs_slot ~rhs_slots =
  let canonical_rhs = [| RHS1; RHS2; RHS3 |] in
  let is_canonical =
    equal_projections_slot lhs_slot LHS
    && Array.for_alli rhs_slots ~f:(fun i slot ->
           i >= Array.length canonical_rhs || equal_projections_slot slot canonical_rhs.(i))
  in
  if is_canonical then ""
  else
    let parts =
      ("lhs", lhs_slot)
      :: Array.to_list (Array.mapi rhs_slots ~f:(fun i slot -> ("rhs" ^ Int.to_string (i + 1), slot)))
    in
    let mappings =
      List.filter_map parts ~f:(fun (target, source) ->
          let canonical_source =
            if String.equal target "lhs" then LHS
            else if String.equal target "rhs1" then RHS1
            else if String.equal target "rhs2" then RHS2
            else RHS3
          in
          if equal_projections_slot source canonical_source then None
          else Some (target ^ "←" ^ slot_to_string source))
    in
    if List.is_empty mappings then "" else " [" ^ String.concat ~sep:", " mappings ^ "]"

type result = {
  vbs : value_binding list;
      (** [vbs] are the bindings introduced by inline tensor declarations (aka. punning). These
          bindings are discharged with the whole [%cd] extension scope in scope, preserving
          definition order. *)
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

let make_vb ~loc ~name ~name_expr ~hint_label ~extra_args =
  let pat = A.Pat.var ~loc { loc = name_expr.pexp_loc; txt = name } in
  let label_arg =
    match hint_label with
    | None -> (Labelled "label", [%expr [ [%e name_expr] ]])
    | Some hint_label -> (Labelled "label", [%expr [%e name_expr] :: [%e hint_label]])
  in
  let term_args =
    label_arg
    :: (Optional "fetch_op", [%expr None])
    :: List.map extra_args ~f:(fun (label, arg) -> (Labelled label, arg))
  in
  let term_call = A.Exp.apply ~loc [%expr NTDSL.term] term_args in
  let v = [%expr [%e term_call] ()] in
  let vb = A.Vb.mk ~loc pat v in
  vb

(** The expression argument is of type: [Assignments.t]. *)
let assignment ~punned ~lhs ~rhses ?body_for_lhs ?raw_body () =
  let setups = lhs :: rhses in
  let body, is_for_lhs =
    match (body_for_lhs, raw_body) with
    | Some body_for_lhs, None ->
        let loc = body_for_lhs.pexp_loc in
        ([%expr Option.value ~default:Ir.Assignments.Noop [%e body_for_lhs]], true)
    | None, Some raw_body -> (raw_body, false)
    | _ -> assert false
  in
  let loc = body.pexp_loc in
  let forward_args = List.filter_map setups ~f:(fun { fwd_code_or_noop; _ } -> fwd_code_or_noop) in
  let vbs, body =
    match lhs.filler_typ with
    | No_grad_tensor_intro { name; name_expr; extra_args } -> (
        let good_hints, bad_hints =
          List.partition_tf ~f:snd @@ List.filter_map rhses ~f:(fun sup -> sup.pun_hint_tnode)
        in
        let hint_data = Option.first_some (List.hd good_hints) (List.hd bad_hints) in
        let hint_label = Option.map ~f:fst hint_data in
        let vbs = [ make_vb ~loc ~name ~name_expr ~hint_label ~extra_args ] in
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
    (* Note: this is not a binding from an inline declaration, it's a temporary binding. *)
    if Option.is_some lhs.vb then
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "ppx_ocannl %%cd: the assigned-to position cannot be an expression building a new tensor"
    else body
  in
  let tensor_vbs = List.filter_map rhses ~f:(fun rhs -> rhs.vb) in
  let body =
    [%expr { Ir.Assignments.asgns = [%e body]; embedded_nodes = Base.Set.empty (module Ir.Tnode) }]
  in
  let comps =
    List.fold (body :: List.rev forward_args) ~init:[%expr []] ~f:(fun xs x ->
        [%expr [%e x] :: [%e xs]])
  in
  let body = [%expr Ir.Assignments.sequence [%e comps]] in
  let body =
    if List.is_empty tensor_vbs then body else A.Exp.let_ ~loc Nonrecursive tensor_vbs body
  in
  let expr =
    if is_for_lhs then
      [%expr
        Option.value
          ~default:
            Ir.Assignments.{ asgns = Noop; embedded_nodes = Base.Set.empty (module Ir.Tnode) }
        @@ Option.map [%e lhs.array_opt] ~f:(fun lhs -> [%e body])]
    else body
  in
  {
    vbs;
    typ = Code { is_commented = false };
    slot = Nonslot;
    expr;
    array_opt_of_code = Some lhs.array_opt;
  }

let project_p_slot debug loc slot =
  match slot with
  | LHS -> [%expr p.project_lhs]
  | RHS1 -> [%expr p.project_rhs.(0)]
  | RHS2 -> [%expr p.project_rhs.(1)]
  | RHS3 -> [%expr p.project_rhs.(2)]
  | Scalar -> [%expr [| Ir.Indexing.Fixed_idx 0 |]]
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
  | RHS3 -> [%expr p.rhs_dims.(2)]
  | Scalar -> [%expr [| 1 |]]
  | Nonslot ->
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "ppx_ocannl %%cd: not a valid accumulation/assignment slot filler at %s" debug
  | Undet ->
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "ppx_ocannl %%cd: insufficient slot filler information at %s %s" debug
           "(incorporate one of: v, v1, v2, g, g1, g2, lhs, rhs, rhs1, rhs2)"

let guess_pun_hint ~no_filler_label ~punned ~bad_pun_hints filler_typ filler =
  let loc = filler.pexp_loc in
  let hint = [%expr [%e filler].Ir.Tnode.label] in
  match (filler_typ, filler, no_filler_label) with
  | (Code _ | Function), _, _ -> None
  | _, { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }, _ when Set.mem bad_pun_hints name ->
      None
  | Array, _, false -> Some (hint, false)
  | (Tensor | Unknown), { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }, _
    when Hashtbl.mem punned name ->
      Hashtbl.find punned name
  | (Tensor | Unknown), { pexp_desc = Pexp_ident _; _ }, _ ->
      Some ([%expr [%e filler].Tensor.value.Ir.Tnode.label], true)
  | (Tensor | Unknown), _, false ->
      Some ([%expr [%e filler].Tensor.value.Ir.Tnode.label], false)
  | ( ( Value_of_tensor { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }
      | Grad_of_tensor { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }
      | Merge_value { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }
      | Merge_grad { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ } ),
      _,
      _ )
    when Set.mem bad_pun_hints name ->
      None
  | ( ( Value_of_tensor { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }
      | Grad_of_tensor { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }
      | Merge_value { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ }
      | Merge_grad { pexp_desc = Pexp_ident { txt = Lident name; _ }; _ } ),
      _,
      _ )
    when Hashtbl.mem punned name ->
      Hashtbl.find punned name
  | (Value_of_tensor t | Grad_of_tensor t | Merge_value t | Merge_grad t), _, false -> (
      let hint = [%expr [%e t].Tensor.value.Ir.Tnode.label] in
      match t with { pexp_desc = Pexp_ident _; _ } -> Some (hint, true) | _ -> Some (hint, false))
  | No_grad_tensor_intro { name; _ }, _, _ -> Hashtbl.find punned name
  | _, _, true -> None

let empty_tns ~loc = [%expr Base.Set.empty (module Ir.Tnode)]

let empty_comp ~loc =
  [%expr { Ir.Assignments.asgns = Ir.Assignments.Noop; embedded_nodes = [%e empty_tns ~loc] }]

let setup_array ~punned ~bad_pun_hints ~for_slot
    { typ = filler_typ; slot; expr = filler; vbs; array_opt_of_code } =
  let loc = filler.pexp_loc in
  let is_lhs = match for_slot with LHS -> true | _ -> false in
  let opt_buffer tn =
    if is_lhs then [%expr Some [%e tn]] else [%expr Some (Ir.Assignments.Node [%e tn])]
  in
  let buffer opt_tn =
    if is_lhs then opt_tn else [%expr Option.map [%e opt_tn] ~f:(fun tn -> Ir.Assignments.Node tn)]
  in
  let pun_hint_tnode no_filler_label =
    guess_pun_hint ~no_filler_label ~punned ~bad_pun_hints filler_typ filler
  in
  let default_setup no_filler_label =
    {
      vb = None;
      fwd_code_or_noop = None;
      filler_typ;
      slot;
      array_opt = opt_buffer [%expr [%e filler].Tensor.value];
      tensor = None;
      pun_hint_tnode = pun_hint_tnode no_filler_label;
    }
  in
  match (List.is_empty vbs, filler_typ) with
  | (false, _ | _, No_grad_tensor_intro _) when not is_lhs ->
      {
        (default_setup false) with
        array_opt =
          Ast_builder.Default.pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "ppx_ocannl %%cd: inline tensor declarations are not allowed in assignment \
                right-hand side, to prevent over-use in locations with less label information";
      }
  | _, (Tensor | Unknown)
    when match filler with { pexp_desc = Pexp_ident _; _ } -> true | _ -> false ->
      let t = filler in
      let fwd_code_or_noop =
        Some
          [%expr
            if Tensor.is_fwd_root [%e t] then (
              Tensor.remove_fwd_root [%e t];
              [%e t].Tensor.forward)
            else [%e empty_comp ~loc]]
      in
      { (default_setup false) with fwd_code_or_noop; tensor = Some t }
  | _, Value_of_tensor ({ pexp_desc = Pexp_ident _; _ } as t) ->
      let fwd_code_or_noop =
        Some
          [%expr
            if Tensor.is_fwd_root [%e t] then (
              Tensor.remove_fwd_root [%e t];
              [%e t].Tensor.forward)
            else [%e empty_comp ~loc]]
      in
      {
        (default_setup false) with
        fwd_code_or_noop;
        array_opt = opt_buffer [%expr [%e t].Tensor.value];
        tensor = Some t;
      }
  | _, Value_of_tensor t ->
      {
        (default_setup false) with
        array_opt =
          Ast_builder.Default.pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "ppx_ocannl %%cd: the <tensor>.value notation is only supported when <tensor> is an \
                identifier";
        tensor = Some t;
      }
  | _, (Tensor | Unknown) ->
      (* Need to bind the expression computing the tensor so we don't recompute it. *)
      let v =
        (* We must use for_slot rather than slot, because the latter might not be unique. *)
        match for_slot with
        | LHS -> [%pat? nondiff__for_lhs]
        | RHS1 -> [%pat? nondiff__for_rhs1]
        | RHS2 -> [%pat? nondiff__for_rhs2]
        | RHS3 -> [%pat? nondiff__for_rhs3]
        | Scalar | Nonslot | Undet -> [%pat? nondiff__tensor]
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
        (default_setup true) with
        vb;
        fwd_code_or_noop;
        array_opt = opt_buffer [%expr [%e t].Tensor.value];
        tensor = Some t;
      }
  | _, No_grad_tensor_intro _ ->
      (* Inline tensors are not allowed to have forward code, but they are embedded. *)
      let fwd_code_or_noop =
        Some
          [%expr
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = Base.Set.singleton (module Ir.Tnode) [%e filler].Tensor.value;
            }]
      in
      { (default_setup false) with fwd_code_or_noop; tensor = Some filler }
  | _, Function ->
      {
        (default_setup false) with
        fwd_code_or_noop = Some filler;
        array_opt =
          Ast_builder.Default.pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "ppx_ocannl %%cd: a syntactic function in place of an array is not supported";
      }
  | _, Code _ when Option.is_none array_opt_of_code ->
      {
        (default_setup false) with
        fwd_code_or_noop = Some filler;
        array_opt =
          Ast_builder.Default.pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "ppx_ocannl %%cd: could not determine a lead array of provided code";
      }
  | _, Code _ ->
      {
        (default_setup false) with
        fwd_code_or_noop = Some filler;
        array_opt = buffer (Option.value_exn array_opt_of_code);
      }
  | _, Array -> { (default_setup false) with array_opt = opt_buffer filler }
  | _, Grad_of_tensor ({ pexp_desc = Pexp_ident _; _ } as t) ->
      { (default_setup false) with array_opt = buffer filler; tensor = Some t }
  | _, Grad_of_tensor t ->
      {
        (default_setup false) with
        array_opt =
          Ast_builder.Default.pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "ppx_ocannl %%cd: the <tensor>.grad notation is only supported when <tensor> is an \
                identifier";
        tensor = Some t;
      }
  | _, (Merge_value _ | Merge_grad _) when is_lhs ->
      {
        (default_setup false) with
        array_opt =
          Ast_builder.Default.pexp_extension ~loc
          @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: merge buffers cannot be assigned to";
      }
  | _, Merge_value t ->
      {
        (default_setup false) with
        array_opt = [%expr Some (Merge_buffer [%e filler])];
        tensor = Some t;
      }
  | _, Merge_grad t ->
      {
        (default_setup false) with
        array_opt = [%expr Option.map [%e filler] ~f:(fun tn -> Ir.Assignments.Merge_buffer tn)];
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

let compare_slots a b =
  match (a, b) with
  | Nonslot, _ -> 1
  | _, Nonslot -> -1
  | Undet, _ -> 1
  | _, Undet -> -1
  | Scalar, _ -> 1
  | _, Scalar -> -1
  | _ -> 0

(** Helper function to handle cases (for Pexp_match, Pexp_function with cases, etc.) *)
let handle_cases ~bad_pun_hints ~proj_in_scope transl cases =
  let fields, transformed_cases =
    List.unzip
    @@ List.map cases ~f:(fun ({ pc_rhs; _ } as c) ->
        let res = transl ~bad_pun_hints ~proj_in_scope pc_rhs in
        ((res.vbs, res.typ, res.slot), { c with pc_rhs = res.expr }))
  in
  let vbss, typs, slots = List.unzip3 fields in
  (* TODO: make the inference of typ and slot more strict by detecting mismatches. *)
  let typ = Option.value ~default:Unknown @@ List.find typs ~f:(Fn.non is_unknown) in
  let slot = List.hd_exn @@ List.sort slots ~compare:compare_slots in
  let loc = (List.hd_exn cases).pc_lhs.ppat_loc in
  ( transformed_cases,
    {
      vbs = reduce_vbss vbss;
      typ;
      slot;
      expr = [%expr ()];
      (* This will be replaced by the caller *)
      array_opt_of_code = None;
    } )

let translate ?ident_label (expr : expression) : result =
  let punned = Hashtbl.create (module String) in
  let rec transl ~bad_pun_hints ~proj_in_scope (expr : expression) : result =
    let loc = expr.pexp_loc in
    let default_result =
      { vbs = no_vbs; typ = Tensor; slot = Undet; expr; array_opt_of_code = None }
    in
    let loop = transl ~bad_pun_hints in
    let assignment_op accu_op =
      loc
      |> Option.value_or_thunk (Hashtbl.find assignment_ops accu_op) ~default:(fun () _loc ->
          ( false,
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%cd: expected an assignment operator, one of: %s %s"
                 "=+ (Add), =- (Sub), =* (Mul),=/ (Div), =** (ToPowOf), =?/ (Relu_gate), =?^ \
                  (Satur01_gate), =|| (Or),  =&& (And), =@^ (Max), =@- (Min), =^^^^ \
                  (threefry4x32), =: (Arg2), =:+, =:-,"
                 " =:*, =:/, =:**, =:?/, =:?^, =:||, =:&&, =:@^, =:@-, =:^^^^ (same with \
                  initializing the tensor to the neutral value before the start of the \
                  calculation)" ))
    in
    let unary_op un_op =
      loc
      |> Option.value_or_thunk (Hashtbl.find unary_ops un_op) ~default:(fun () loc ->
          ( [%expr Shape.Pointwise_un],
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%cd: expected a unary operator, one of: %s"
                 "id, relu, sat01, exp, log, exp2, log2, sin, cos, sqrt, recip, recip_sqrt, neg, \
                  tanh, uint4x32_to_prec_uniform1" ))
    in
    let vec_unary_op vec_un_op =
      loc
      |> Option.value_or_thunk (Hashtbl.find vec_unary_ops vec_un_op) ~default:(fun () loc ->
          ( [%expr Shape.Uint4x32_to_prec_uniform],
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%cd: expected a vector unary operator, one of: \
                  uint4x32_to_prec_uniform; found: %s"
                 vec_un_op ))
    in
    let binary_op bin_op =
      loc
      |> Option.value_or_thunk (Hashtbl.find binary_ops bin_op) ~default:(fun () _loc ->
          ( [%expr Shape.Pointwise_bin],
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%cd: expected a binary operator, one of: %s"
                 "+ (Add), - (Sub), * (Mul), / (Div), **(ToPowOf), -?/ (Relu_gate), -?^ \
                  (Satur01_gate), -/> (Arg2), <  (Cmplt), = (Cmpeq), <> (Cmpne), || (Or), && \
                  (And), % (Mod), @^(Max), @- (Min), ^^^^ (threefry4x32)" ))
    in
    let ternary_op tern_op =
      loc
      |> Option.value_or_thunk (Hashtbl.find ternary_ops tern_op) ~default:(fun () _loc ->
          ( [%expr Shape.Pointwise_tern],
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%cd: expected a ternary operator, one of: where, fma" ))
    in
    (* TODO: collapse these (code reuse) *)
    let process_assign_ternop ~accu_op ~lhs ~tern_op ~rhs1 ~rhs2 ~rhs3 ?projections ~proj_in_scope
        () =
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l =
        setup_array ~punned ~bad_pun_hints ~for_slot:LHS @@ loop ~proj_in_scope:true lhs
      in
      let _, tern_op = ternary_op tern_op in
      let setup_r1 =
        setup_array ~punned ~bad_pun_hints ~for_slot:RHS1 @@ loop ~proj_in_scope rhs1
      in
      let setup_r2 =
        setup_array ~punned ~bad_pun_hints ~for_slot:RHS2 @@ loop ~proj_in_scope rhs2
      in
      let setup_r3 =
        setup_array ~punned ~bad_pun_hints ~for_slot:RHS3 @@ loop ~proj_in_scope rhs3
      in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let projections_lazy, projections_debug =
        match projections with
        | Some prjs ->
            ([%expr [%e prjs].Tensor.projections], [%expr [%e prjs].Tensor.projections_debug])
        | None ->
            let lhs_dims = project_p_dims "LHS" lhs.pexp_loc setup_l.slot in
            let rhs1_dims = project_p_dims "RHS1" lhs.pexp_loc setup_r1.slot in
            let rhs2_dims = project_p_dims "RHS2" lhs.pexp_loc setup_r2.slot in
            let rhs3_dims = project_p_dims "RHS3" lhs.pexp_loc setup_r3.slot in
            let project_lhs = project_p_slot "LHS" lhs.pexp_loc setup_l.slot in
            let project_rhs1 = project_p_slot "RHS1" rhs1.pexp_loc setup_r1.slot in
            let project_rhs2 = project_p_slot "RHS2" rhs2.pexp_loc setup_r2.slot in
            let project_rhs3 = project_p_slot "RHS3" rhs3.pexp_loc setup_r3.slot in
            let proj_lazy =
              [%expr
                lazy
                  (let p = Lazy.force projections.Tensor.projections in
                   Ir.Indexing.
                     {
                       product_space = p.product_space;
                       product_iterators = p.product_iterators;
                       lhs_dims = [%e lhs_dims];
                       rhs_dims = [| [%e rhs1_dims]; [%e rhs2_dims]; [%e rhs3_dims] |];
                       project_lhs = [%e project_lhs];
                       project_rhs = [| [%e project_rhs1]; [%e project_rhs2]; [%e project_rhs3] |];
                       debug_info =
                         {
                           p.debug_info with
                           trace =
                             ( "ppx_cd " ^ [%e expr2string_or_empty accu_op] ^ " "
                               ^ [%e expr2string_or_empty tern_op],
                               Ir.Indexing.unique_debug_id () )
                             :: p.debug_info.trace;
                         };
                     })]
            in
            let slot_suffix =
              slot_permutation_suffix ~lhs_slot:setup_l.slot
                ~rhs_slots:[| setup_r1.slot; setup_r2.slot; setup_r3.slot |]
            in
            let proj_debug =
              if String.is_empty slot_suffix then [%expr projections.Tensor.projections_debug]
              else
                let suffix_expr = Ast_builder.Default.estring ~loc slot_suffix in
                [%expr projections.Tensor.projections_debug ^ [%e suffix_expr]]
            in
            (proj_lazy, proj_debug)
      in
      (* FIXME: might be better to treat missing [rhs1, rhs2, rhs3] as zeros or errors rather than
         eliding the code, only lhs should decide whether to elide the code. *)
      let body_for_lhs =
        [%expr
          Option.map3 [%e setup_r1.array_opt] [%e setup_r2.array_opt] [%e setup_r3.array_opt]
            ~f:(fun rhs1 rhs2 rhs3 ->
              Ir.Assignments.Accum_op
                {
                  initialize_neutral = [%e initialize_neutral];
                  accum = [%e accu_op];
                  lhs;
                  rhs = Ternop { op = [%e tern_op]; rhs1; rhs2; rhs3 };
                  projections = [%e projections_lazy];
                  projections_debug = [%e projections_debug];
                })]
      in
      assignment ~punned ~lhs:setup_l ~rhses:[ setup_r1; setup_r2; setup_r3 ] ~body_for_lhs ()
    in
    let process_assign_binop ~accu_op ~lhs ~bin_op ~rhs1 ~rhs2 ?projections ~proj_in_scope () =
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l =
        setup_array ~punned ~bad_pun_hints ~for_slot:LHS @@ loop ~proj_in_scope:true lhs
      in
      let _, bin_op = binary_op bin_op in
      let setup_r1 =
        setup_array ~punned ~bad_pun_hints ~for_slot:RHS1 @@ loop ~proj_in_scope rhs1
      in
      let setup_r2 =
        setup_array ~punned ~bad_pun_hints ~for_slot:RHS2 @@ loop ~proj_in_scope rhs2
      in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let projections_lazy, projections_debug =
        match projections with
        | Some prjs ->
            ([%expr [%e prjs].Tensor.projections], [%expr [%e prjs].Tensor.projections_debug])
        | None ->
            let lhs_dims = project_p_dims "LHS" lhs.pexp_loc setup_l.slot in
            let rhs1_dims = project_p_dims "RHS1" lhs.pexp_loc setup_r1.slot in
            let rhs2_dims = project_p_dims "RHS2" lhs.pexp_loc setup_r2.slot in
            let project_lhs = project_p_slot "LHS" lhs.pexp_loc setup_l.slot in
            let project_rhs1 = project_p_slot "RHS1" rhs1.pexp_loc setup_r1.slot in
            let project_rhs2 = project_p_slot "RHS2" rhs2.pexp_loc setup_r2.slot in
            let proj_lazy =
              [%expr
                lazy
                  (let p = Lazy.force projections.Tensor.projections in
                   Ir.Indexing.
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
                               Ir.Indexing.unique_debug_id () )
                             :: p.debug_info.trace;
                         };
                     })]
            in
            let slot_suffix =
              slot_permutation_suffix ~lhs_slot:setup_l.slot
                ~rhs_slots:[| setup_r1.slot; setup_r2.slot |]
            in
            let proj_debug =
              if String.is_empty slot_suffix then [%expr projections.Tensor.projections_debug]
              else
                let suffix_expr = Ast_builder.Default.estring ~loc slot_suffix in
                [%expr projections.Tensor.projections_debug ^ [%e suffix_expr]]
            in
            (proj_lazy, proj_debug)
      in
      (* FIXME: might be better to treat missing [rhs1, rhs2] as zeros or errors rather than eliding
         the code, only lhs should decide whether to elide the code. *)
      let body_for_lhs =
        [%expr
          Option.map2 [%e setup_r1.array_opt] [%e setup_r2.array_opt] ~f:(fun rhs1 rhs2 ->
              Ir.Assignments.Accum_op
                {
                  initialize_neutral = [%e initialize_neutral];
                  accum = [%e accu_op];
                  lhs;
                  rhs = Binop { op = [%e bin_op]; rhs1; rhs2 };
                  projections = [%e projections_lazy];
                  projections_debug = [%e projections_debug];
                })]
      in
      assignment ~punned ~lhs:setup_l ~rhses:[ setup_r1; setup_r2 ] ~body_for_lhs ()
    in
    let process_assign_unop ~accu_op ~lhs ~un_op ~rhs ?projections ~proj_in_scope () =
      let initialize_neutral, accum = assignment_op accu_op in
      let _, op = unary_op un_op in
      (* FIXME: I think this ignores the slot information here! Just assuming [projections] is
         as-should-be, but that's not consistent with omitting the projections arg (assuming it
         comes from the context). *)
      let setup_l = setup_array ~punned ~bad_pun_hints ~for_slot:LHS @@ loop ~proj_in_scope lhs in
      let setup_r = setup_array ~punned ~bad_pun_hints ~for_slot:RHS1 @@ loop ~proj_in_scope rhs in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let projections_lazy, projections_debug =
        match projections with
        | Some prjs ->
            ([%expr [%e prjs].Tensor.projections], [%expr [%e prjs].Tensor.projections_debug])
        | None ->
            let lhs_dims = project_p_dims "LHS" lhs.pexp_loc setup_l.slot in
            let rhs1_dims = project_p_dims "RHS1" lhs.pexp_loc setup_r.slot in
            let project_lhs = project_p_slot "LHS" lhs.pexp_loc setup_l.slot in
            let project_rhs1 = project_p_slot "RHS1" rhs.pexp_loc setup_r.slot in
            let proj_lazy =
              [%expr
                lazy
                  (let p = Lazy.force projections.Tensor.projections in
                   Ir.Indexing.
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
                             ( "ppx_cd " ^ [%e string_expr ~loc accu_op] ^ " "
                               ^ [%e string_expr ~loc un_op],
                               Ir.Indexing.unique_debug_id () )
                             :: p.debug_info.trace;
                         };
                     })]
            in
            let slot_suffix =
              slot_permutation_suffix ~lhs_slot:setup_l.slot ~rhs_slots:[| setup_r.slot |]
            in
            let proj_debug =
              if String.is_empty slot_suffix then [%expr projections.Tensor.projections_debug]
              else
                let suffix_expr = Ast_builder.Default.estring ~loc slot_suffix in
                [%expr projections.Tensor.projections_debug ^ [%e suffix_expr]]
            in
            (proj_lazy, proj_debug)
      in
      (* FIXME: might be better to treat missing [rhs] as zeros or errors rather than eliding the
         code, only lhs should decide whether to elide the code. *)
      let body_for_lhs =
        [%expr
          Option.map [%e setup_r.array_opt] ~f:(fun rhs ->
              Ir.Assignments.Accum_op
                {
                  initialize_neutral = [%e initialize_neutral];
                  accum = [%e accum];
                  lhs;
                  rhs = Unop { op = [%e op]; rhs };
                  projections = [%e projections_lazy];
                  projections_debug = [%e projections_debug];
                })]
      in
      assignment ~punned ~lhs:setup_l ~rhses:[ setup_r ] ~body_for_lhs ()
    in
    let process_vec_unop ~lhs ~vec_un_op ~rhs ?projections ~proj_in_scope () =
      (* Vector unary operations do not have accumulation, they directly set values *)
      let _, op = vec_unary_op vec_un_op in
      let setup_l = setup_array ~punned ~bad_pun_hints ~for_slot:LHS @@ loop ~proj_in_scope lhs in
      let setup_r = setup_array ~punned ~bad_pun_hints ~for_slot:RHS1 @@ loop ~proj_in_scope rhs in
      let projections_lazy, projections_debug =
        match projections with
        | Some prjs ->
            ([%expr [%e prjs].Tensor.projections], [%expr [%e prjs].Tensor.projections_debug])
        | None ->
            let lhs_dims = project_p_dims "LHS" lhs.pexp_loc setup_l.slot in
            let rhs1_dims = project_p_dims "RHS1" lhs.pexp_loc setup_r.slot in
            let project_lhs = project_p_slot "LHS" lhs.pexp_loc setup_l.slot in
            let project_rhs1 = project_p_slot "RHS1" rhs.pexp_loc setup_r.slot in
            let proj_lazy =
              [%expr
                lazy
                  (let p = Lazy.force projections.Tensor.projections in
                   Ir.Indexing.
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
                             ( "ppx_cd vec " ^ [%e string_expr ~loc vec_un_op],
                               Ir.Indexing.unique_debug_id () )
                             :: p.debug_info.trace;
                         };
                     })]
            in
            (proj_lazy, [%expr projections.Tensor.projections_debug])
      in
      let body_for_lhs =
        [%expr
          Option.map [%e setup_r.array_opt] ~f:(fun rhs ->
              Ir.Assignments.Set_vec_unop
                {
                  lhs;
                  op = [%e op];
                  rhs;
                  projections = [%e projections_lazy];
                  projections_debug = [%e projections_debug];
                })]
      in
      assignment ~punned ~lhs:setup_l ~rhses:[ setup_r ] ~body_for_lhs ()
    in
    let process_raw_ternop ~accu_op ~lhs ~tern_op ~rhs1 ~rhs2 ~rhs3 ~logic =
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l = setup_array ~punned ~bad_pun_hints ~for_slot:LHS @@ loop ~proj_in_scope lhs in
      let setup_r1 =
        setup_array ~punned ~bad_pun_hints ~for_slot:RHS1 @@ loop ~proj_in_scope rhs1
      in
      let setup_r2 =
        setup_array ~punned ~bad_pun_hints ~for_slot:RHS2 @@ loop ~proj_in_scope rhs2
      in
      let setup_r3 =
        setup_array ~punned ~bad_pun_hints ~for_slot:RHS3 @@ loop ~proj_in_scope rhs3
      in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let t_expr, lhs_is_grad, _ = args_for ~loc setup_l in
      let t1_expr, rhs1_is_grad, rhs1_is_merge = args_for ~loc setup_r1 in
      let t2_expr, rhs2_is_grad, rhs2_is_merge = args_for ~loc setup_r2 in
      let t3_expr, rhs3_is_grad, rhs3_is_merge = args_for ~loc setup_r3 in
      let raw_body =
        [%expr
          Tensor.raw_ternop ~initialize_neutral:[%e initialize_neutral] ~accum:[%e accu_op]
            ~t:[%e t_expr] ~lhs_is_grad:[%e lhs_is_grad] ~op:[%e tern_op] ~t1:[%e t1_expr]
            ~rhs1_is_grad:[%e rhs1_is_grad] ~rhs1_is_merge:[%e rhs1_is_merge] ~t2:[%e t2_expr]
            ~rhs2_is_grad:[%e rhs2_is_grad] ~rhs2_is_merge:[%e rhs2_is_merge] ~t3:[%e t3_expr]
            ~rhs3_is_grad:[%e rhs3_is_grad] ~rhs3_is_merge:[%e rhs3_is_merge] ~logic:[%e logic]]
      in
      assignment ~punned ~lhs:setup_l ~rhses:[ setup_r1; setup_r2; setup_r3 ] ~raw_body ()
    in
    let process_raw_binop ~accu_op ~lhs ~bin_op ~rhs1 ~rhs2 ~logic =
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l = setup_array ~punned ~bad_pun_hints ~for_slot:LHS @@ loop ~proj_in_scope lhs in
      let setup_r1 =
        setup_array ~punned ~bad_pun_hints ~for_slot:RHS1 @@ loop ~proj_in_scope rhs1
      in
      let setup_r2 =
        setup_array ~punned ~bad_pun_hints ~for_slot:RHS2 @@ loop ~proj_in_scope rhs2
      in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let t_expr, lhs_is_grad, _ = args_for ~loc setup_l in
      let t1_expr, rhs1_is_grad, rhs1_is_merge = args_for ~loc setup_r1 in
      let t2_expr, rhs2_is_grad, rhs2_is_merge = args_for ~loc setup_r2 in
      let raw_body =
        [%expr
          Tensor.raw_binop ~initialize_neutral:[%e initialize_neutral] ~accum:[%e accu_op]
            ~t:[%e t_expr] ~lhs_is_grad:[%e lhs_is_grad] ~op:[%e bin_op] ~t1:[%e t1_expr]
            ~rhs1_is_grad:[%e rhs1_is_grad] ~rhs1_is_merge:[%e rhs1_is_merge] ~t2:[%e t2_expr]
            ~rhs2_is_grad:[%e rhs2_is_grad] ~rhs2_is_merge:[%e rhs2_is_merge] ~logic:[%e logic]]
      in
      assignment ~punned ~lhs:setup_l ~rhses:[ setup_r1; setup_r2 ] ~raw_body ()
    in
    let process_raw_unop ~accu_op ~lhs ~un_op ~rhs ~logic =
      let initialize_neutral, accu_op = assignment_op accu_op in
      let setup_l = setup_array ~punned ~bad_pun_hints ~for_slot:LHS @@ loop ~proj_in_scope lhs in
      let setup_r = setup_array ~punned ~bad_pun_hints ~for_slot:RHS1 @@ loop ~proj_in_scope rhs in
      let initialize_neutral = if initialize_neutral then [%expr true] else [%expr false] in
      let t_expr, lhs_is_grad, _ = args_for ~loc setup_l in
      let t1_expr, rhs_is_grad, rhs_is_merge = args_for ~loc setup_r in
      let raw_body =
        [%expr
          Tensor.raw_unop ~initialize_neutral:[%e initialize_neutral] ~accum:[%e accu_op]
            ~t:[%e t_expr] ~lhs_is_grad:[%e lhs_is_grad] ~op:[%e un_op] ~t1:[%e t1_expr]
            ~rhs_is_grad:[%e rhs_is_grad] ~rhs_is_merge:[%e rhs_is_merge] ~logic:[%e logic]]
      in
      assignment ~punned ~lhs:setup_l ~rhses:[ setup_r ] ~raw_body ()
    in
    match expr with
    | { pexp_desc = Pexp_extension ({ txt = "oc"; _ }, payload); _ } -> (
        (* %oc anti-quotation: preserve the expression without transformation *)
        match payload with
        | PStr [ { pstr_desc = Pstr_eval (expr, _); _ } ] -> { default_result with expr }
        | _ ->
            {
              default_result with
              expr =
                Ast_builder.Default.pexp_extension ~loc
                @@ Location.error_extensionf ~loc "%%oc expects a single expression";
            })
    | { pexp_desc = Pexp_constant (Pconst_float _); _ } ->
        { default_result with expr = [%expr NTDSL.number [%e expr]]; slot = Scalar }
    | { pexp_desc = Pexp_constant (Pconst_integer (_, Some ('L' | 'l'))); _ } ->
        { default_result with expr = [%expr NTDSL.bits [%e expr]]; slot = Scalar }
    | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
        { default_result with expr = [%expr NTDSL.number (Float.of_int [%e expr])]; slot = Scalar }
    | [%expr
        [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
          [%e? { pexp_desc = Pexp_constant (Pconst_float _); _ } as f]] ->
        let axis =
          Ast_helper.Exp.constant ~loc:pexp_loc (Pconst_string (String.of_char ch, pexp_loc, None))
        in
        {
          default_result with
          expr = [%expr NTDSL.number ~axis_label:[%e axis] [%e f]];
          slot = Scalar;
        }
    | [%expr
        [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
          [%e? { pexp_desc = Pexp_constant (Pconst_integer (_, Some ('L' | 'l'))); _ } as i]] ->
        let axis =
          Ast_helper.Exp.constant ~loc:pexp_loc (Pconst_string (String.of_char ch, pexp_loc, None))
        in
        {
          default_result with
          expr = [%expr NTDSL.bits ~axis_label:[%e axis] [%e i]];
          slot = Scalar;
        }
    | [%expr
        [%e? { pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc; _ }]
          [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
        let axis =
          Ast_helper.Exp.constant ~loc:pexp_loc (Pconst_string (String.of_char ch, pexp_loc, None))
        in
        {
          default_result with
          expr = [%expr NTDSL.number ~axis_label:[%e axis] (Float.of_int [%e i])];
          slot = Scalar;
        }
    | { pexp_desc = Pexp_record ((first_label, first_value) :: extra_args, None); _ } -> (
        (* Record syntax for tensor definitions *)
        match (first_label.txt, first_value.pexp_desc) with
        | Lident tensor_name, Pexp_ident { txt = Lident name_expr; _ }
          when String.equal name_expr tensor_name ->
            (* Simple case: just tensor initialization, similar to original string syntax *)
            let name_expr =
              Ast_helper.Exp.constant ~loc:first_label.loc
                (Pconst_string (tensor_name, first_label.loc, None))
            in
            let extra_args =
              List.map extra_args ~f:(function
                | { txt = Lident "o"; _ }, value -> ("output_dims", value)
                | { txt = Lident "i"; _ }, value -> ("input_dims", value)
                | { txt = Lident "b"; _ }, value -> ("batch_dims", value)
                | { txt = Lident label; _ }, value -> (label, value)
                | { loc; _ }, _ ->
                    ( "syntax_error",
                      Ast_builder.Default.pexp_extension ~loc
                      @@ Location.error_extensionf ~loc
                           "inline-definition fields must be function argument labels" ))
            in
            (* NOTE: this binding is not used in assignments therefore is very unlikely to be used.
               But it's needed for code expressions or standalone non-diff tensor expressions. *)
            let vbs =
              [ make_vb ~loc ~name:tensor_name ~name_expr
                  ~hint_label:(Option.map ~f:(fun s -> [%expr [ [%e s] ]]) ident_label)
                  ~extra_args ]
            in
            let slot =
              (* Detect projection slot from tensor name prefix/suffix patterns *)
              let has_prefix p = String.is_prefix tensor_name ~prefix:p in
              let has_suffix s = String.is_suffix tensor_name ~suffix:s in
              if has_prefix "lhs_" || has_suffix "_lhs" then LHS
              else if has_prefix "rhs1_" || has_suffix "_rhs1" then RHS1
              else if has_prefix "rhs2_" || has_suffix "_rhs2" then RHS2
              else if has_prefix "rhs3_" || has_suffix "_rhs3" then RHS3
              else if has_prefix "rhs_" || has_suffix "_rhs" then RHS1
              else Undet
            in
            {
              vbs;
              typ = No_grad_tensor_intro { name = tensor_name; name_expr; extra_args };
              expr =
                A.Exp.ident ~loc:first_label.loc { txt = Lident tensor_name; loc = first_label.loc };
              array_opt_of_code = None;
              slot;
            }
        | Lident _tensor_name, _ ->
            {
              default_result with
              expr =
                Ast_builder.Default.pexp_extension ~loc
                @@ Location.error_extensionf ~loc
                     "ppx_ocannl %%cd: tensors inline-defined in code cannot have initializers, \
                      but you can use an init_data field";
            }
        | _ ->
            {
              default_result with
              expr =
                Ast_builder.Default.pexp_extension ~loc
                @@ Location.error_extensionf ~loc
                     "ppx_ocannl %%cd: for inline-defined tensors, record field label must be a \
                      simple identifier";
            })
    | { pexp_desc = Pexp_array _; _ }
    | { pexp_desc = Pexp_construct ({ txt = Lident "::"; _ }, _); _ } ->
        { default_result with expr = ndarray_op ~ndarray_fn:[%expr NTDSL.ndarray] expr }
    | { pexp_desc = Pexp_ident { txt = Lident "lhs"; _ }; _ } ->
        { default_result with typ = Array; slot = LHS }
    | { pexp_desc = Pexp_ident { txt = Lident "v"; _ }; _ } ->
        { default_result with typ = Array; slot = LHS; expr = [%expr t.Tensor.value] }
    | { pexp_desc = Pexp_ident { txt = Lident "g"; _ }; _ } ->
        { default_result with typ = Array; slot = LHS }
    | { pexp_desc = Pexp_ident { txt = Lident "rhs1"; _ }; _ } ->
        { default_result with typ = Array; slot = RHS1 }
    | { pexp_desc = Pexp_ident { txt = Lident "t"; _ }; _ } -> { default_result with slot = LHS }
    | { pexp_desc = Pexp_ident { txt = Lident "t1"; _ }; _ } -> { default_result with slot = RHS1 }
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
    | { pexp_desc = Pexp_ident { txt = Lident "rhs3"; _ }; _ } ->
        { default_result with typ = Array; slot = RHS3 }
    | { pexp_desc = Pexp_ident { txt = Lident "t3"; _ }; _ } ->
        { default_result with typ = Tensor; slot = RHS3 }
    | { pexp_desc = Pexp_ident { txt = Lident "v3"; _ }; _ } ->
        { default_result with typ = Array; slot = RHS3; expr = [%expr t3.Tensor.value] }
    | { pexp_desc = Pexp_ident { txt = Lident "g3"; _ }; _ } ->
        {
          default_result with
          typ = Grad_of_tensor [%expr t3];
          slot = RHS3;
          expr = [%expr Option.map t3.Tensor.diff ~f:(fun d -> d.Tensor.grad)];
        }
    | { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ } when is_primitive_op op_ident ->
        default_result
    | { pexp_desc = Pexp_ident { txt = Lident ident_name; _ }; _ } ->
        (* Detect projection slot from identifier name prefix/suffix patterns *)
        let has_prefix p = String.is_prefix ident_name ~prefix:p in
        let has_suffix s = String.is_suffix ident_name ~suffix:s in
        let slot =
          if has_prefix "lhs_" || has_suffix "_lhs" then LHS
          else if has_prefix "rhs1_" || has_suffix "_rhs1" then RHS1
          else if has_prefix "rhs2_" || has_suffix "_rhs2" then RHS2
          else if has_prefix "rhs3_" || has_suffix "_rhs3" then RHS3
          else if has_prefix "rhs_" || has_suffix "_rhs" then RHS1
          else Undet
        in
        { default_result with typ = if is_undet slot then Unknown else Tensor; slot }
    | [%expr !.[%e? expr1]] ->
        (* Hardcoding these two patterns (!. and !..) to improve projection derivation expressivity
           and avoid treating the constants as already tensors. *)
        {
          typ = Tensor;
          slot = Scalar;
          expr = [%expr NTDSL.O.( !. ) [%e expr1]];
          array_opt_of_code = None;
          vbs = no_vbs;
        }
    | [%expr !..[%e? expr1]] ->
        {
          typ = Tensor;
          slot = Scalar;
          expr = [%expr NTDSL.O.( !.. ) [%e expr1]];
          array_opt_of_code = None;
          vbs = no_vbs;
        }
    | [%expr [%e? expr1] **. [%e? { pexp_desc = Pexp_constant (Pconst_integer _); _ } as i]] ->
        (* We need to hardcode these two patterns (for **. ) to prevent the numbers from being
           converted to tensors. *)
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
        [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
          [%e? expr1]
          ([%e? { pexp_desc = Pexp_ident _; _ } as spec] [%e? expr2])]
      when Hashtbl.mem einsum_binary_ops op_ident ->
        let res1 = loop ~proj_in_scope expr1 in
        let res2 = loop ~proj_in_scope expr2 in
        let slot = List.hd_exn @@ List.sort [ res1.slot; res2.slot ] ~compare:compare_slots in
        {
          vbs = reduce_vbss [ res1.vbs; res2.vbs ];
          typ = Tensor;
          slot;
          expr =
            [%expr
              [%e Hashtbl.find_exn einsum_binary_ops op_ident loc]
                [%e spec] [%e res1.expr] [%e res2.expr]];
          array_opt_of_code = None;
        }
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
          [%e? expr1]
          ([%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }] [%e? expr2])]
      when String.contains spec_str '>' && Hashtbl.mem einsum_binary_ops op_ident ->
        let res1 = loop ~proj_in_scope expr1 in
        let res2 = loop ~proj_in_scope expr2 in
        let spec = substitute_identifiers_in_einsum_spec ~loc spec_str in
        let slot = List.hd_exn @@ List.sort [ res1.slot; res2.slot ] ~compare:compare_slots in
        {
          vbs = reduce_vbss [ res1.vbs; res2.vbs ];
          typ = Tensor;
          slot;
          expr =
            [%expr
              [%e Hashtbl.find_exn einsum_binary_ops op_ident loc]
                [%e spec] [%e res1.expr] [%e res2.expr]];
          array_opt_of_code = None;
        }
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
          [%e? expr1]
          ([%e? expr2] [%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }])]
      when String.contains spec_str '>' && Hashtbl.mem einsum_binary_ops op_ident ->
        (* Need duplication because of pattern matching restriction. *)
        let res1 = loop ~proj_in_scope expr1 in
        let res2 = loop ~proj_in_scope expr2 in
        let spec = substitute_identifiers_in_einsum_spec ~loc spec_str in
        let slot = List.hd_exn @@ List.sort [ res1.slot; res2.slot ] ~compare:compare_slots in
        {
          vbs = reduce_vbss [ res1.vbs; res2.vbs ];
          typ = Tensor;
          slot;
          expr =
            [%expr
              [%e Hashtbl.find_exn einsum_binary_ops op_ident loc]
                [%e spec] [%e res1.expr] [%e res2.expr]];
          array_opt_of_code = None;
        }
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
          [%e? expr1]
          ([%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }]
             ([%e? { pexp_desc = Pexp_constant (Pconst_string _); _ } as head] :: [%e? rest])
             [%e? expr2])]
      when String.contains spec_str '>' && Hashtbl.mem einsum_binary_ops op_ident ->
        let capture_vbs, capture_dims_expr = collect_capture_labels ~loc head rest in
        let res1 = loop ~proj_in_scope expr1 in
        let res2 = loop ~proj_in_scope expr2 in
        let spec = substitute_identifiers_in_einsum_spec ~loc spec_str in
        let slot = List.hd_exn @@ List.sort [ res1.slot; res2.slot ] ~compare:compare_slots in
        {
          vbs = reduce_vbss [ res1.vbs; res2.vbs; capture_vbs ];
          typ = Tensor;
          slot;
          expr =
            [%expr
              [%e Hashtbl.find_exn einsum_binary_ops op_ident loc]
                ~capture_dims:[%e capture_dims_expr] [%e spec] [%e res1.expr] [%e res2.expr]];
          array_opt_of_code = None;
        }
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
          [%e? expr1]
          ([%e? expr2]
             [%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }]
             ([%e? { pexp_desc = Pexp_constant (Pconst_string _); _ } as head] :: [%e? rest]))]
      when String.contains spec_str '>' && Hashtbl.mem einsum_binary_ops op_ident ->
        let capture_vbs, capture_dims_expr = collect_capture_labels ~loc head rest in
        let res1 = loop ~proj_in_scope expr1 in
        let res2 = loop ~proj_in_scope expr2 in
        let spec = substitute_identifiers_in_einsum_spec ~loc spec_str in
        let slot = List.hd_exn @@ List.sort [ res1.slot; res2.slot ] ~compare:compare_slots in
        {
          vbs = reduce_vbss [ res1.vbs; res2.vbs; capture_vbs ];
          typ = Tensor;
          slot;
          expr =
            [%expr
              [%e Hashtbl.find_exn einsum_binary_ops op_ident loc]
                ~capture_dims:[%e capture_dims_expr] [%e spec] [%e res1.expr] [%e res2.expr]];
          array_opt_of_code = None;
        }
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
          [%e? expr1]
          [%e? { pexp_desc = Pexp_ident _; _ } as spec]]
      when Hashtbl.mem einsum_unary_ops op_ident ->
        let res1 = loop ~proj_in_scope expr1 in
        {
          res1 with
          typ = Tensor;
          expr =
            [%expr [%e Hashtbl.find_exn einsum_unary_ops op_ident loc] [%e spec] [%e res1.expr]];
        }
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
          [%e? expr1]
          [%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }]]
      when String.contains spec_str '>' && Hashtbl.mem einsum_unary_ops op_ident ->
        let res1 = loop ~proj_in_scope expr1 in
        let spec = substitute_identifiers_in_einsum_spec ~loc spec_str in
        {
          res1 with
          typ = Tensor;
          expr =
            [%expr [%e Hashtbl.find_exn einsum_unary_ops op_ident loc] [%e spec] [%e res1.expr]];
        }
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
          [%e? expr1]
          ([%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }]
             ([%e? { pexp_desc = Pexp_constant (Pconst_string _); _ } as head] :: [%e? rest]))]
      when String.contains spec_str '>' && Hashtbl.mem einsum_unary_ops op_ident ->
        let capture_vbs, capture_dims_expr = collect_capture_labels ~loc head rest in
        let res1 = loop ~proj_in_scope expr1 in
        let spec = substitute_identifiers_in_einsum_spec ~loc spec_str in
        {
          vbs = reduce_vbss [ res1.vbs; capture_vbs ];
          typ = Tensor;
          slot = res1.slot;
          expr =
            [%expr
              [%e Hashtbl.find_exn einsum_unary_ops op_ident loc]
                ~capture_dims:[%e capture_dims_expr] [%e spec] [%e res1.expr]];
          array_opt_of_code = None;
        }
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
        | Function | Code _ | Array | Value_of_tensor _ | Grad_of_tensor _ | Merge_grad _ ->
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
        | Function | Code _ ->
            {
              res1 with
              typ = Array;
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
        | Function | Array | Code _ ->
            {
              res1 with
              typ = Array;
              expr =
                Ast_builder.Default.pexp_extension ~loc
                @@ Location.error_extensionf ~loc
                     "ppx_ocannl %%cd: only tensor nodes (e.g. `.value` or `.grad`) can be merged";
            }
        | Grad_of_tensor t -> { res1 with typ = Merge_grad t }
        | Merge_value _ | Merge_grad _ ->
            {
              res1 with
              expr =
                Ast_builder.Default.pexp_extension ~loc
                @@ Location.error_extensionf ~loc "ppx_ocannl %%cd: repeated .merge not allowed";
            })
    | [%expr [%e? expr1].forward] -> (
        let res1 = loop ~proj_in_scope expr1 in
        match res1.typ with
        | Unknown | Tensor | No_grad_tensor_intro _ ->
            {
              res1 with
              typ = Code { is_commented = false };
              expr = [%expr Tensor.consume_forward_code [%e res1.expr]];
            }
        | _ ->
            {
              res1 with
              expr =
                Ast_builder.Default.pexp_extension ~loc
                @@ Location.error_extensionf ~loc
                     "ppx_ocannl %%cd: .forward can only be applied to tensors";
            })
    | [%expr [%e? expr1].backprop] -> (
        let res1 = loop ~proj_in_scope expr1 in
        match res1.typ with
        | Unknown | Tensor | No_grad_tensor_intro _ ->
            {
              res1 with
              typ = Code { is_commented = false };
              expr = [%expr Tensor.consume_backprop_code [%e res1.expr]];
            }
        | _ ->
            {
              res1 with
              expr =
                Ast_builder.Default.pexp_extension ~loc
                @@ Location.error_extensionf ~loc
                     "ppx_ocannl %%cd: .backprop can only be applied to tensors";
            })
    | [%expr [%e? expr1].zero_grads] -> (
        let res1 = loop ~proj_in_scope expr1 in
        match res1.typ with
        | Unknown | Tensor | No_grad_tensor_intro _ ->
            {
              res1 with
              typ = Code { is_commented = false };
              expr =
                [%expr
                  match [%e res1.expr].diff with
                  | None ->
                      raise
                        (Invalid_argument
                           "ppx_ocannl %cd: .zero_grads requires a differentiable tensor")
                  | Some diff -> Ir.Assignments.to_comp diff.zero_grads];
            }
        | _ ->
            {
              res1 with
              expr =
                Ast_builder.Default.pexp_extension ~loc
                @@ Location.error_extensionf ~loc
                     "ppx_ocannl %%cd: .zero_grads can only be applied to tensors";
            })
    | [%expr
        ~~([%e? { pexp_desc = Pexp_constant (Pconst_string _); _ } as comment];
           [%e? expr2])] ->
        let res2 = loop ~proj_in_scope expr2 in
        let block =
          match res2.typ with
          | Code _ -> res2.expr
          | _ ->
              Ast_builder.Default.pexp_extension ~loc
              @@ Location.error_extensionf ~loc
                   "ppx_ocannl %%cd: only code can be commented, e.g. assignments or t.forward, \
                    t.backprop, t.zero_grads"
        in
        {
          res2 with
          expr =
            [%expr
              let __comment_block = [%e block] in
              {
                Ir.Assignments.asgns =
                  Ir.Assignments.Block_comment ([%e comment], __comment_block.Ir.Assignments.asgns);
                embedded_nodes = __comment_block.Ir.Assignments.embedded_nodes;
              }];
        }
    | [%expr
        ~~([%e? { pexp_desc = Pexp_apply (expr, exprs); pexp_loc; _ }];
           [%e? expr2])] ->
        let elements =
          expr :: List.map ~f:snd exprs
          |> List.map ~f:(function
            | { pexp_desc = Pexp_constant (Pconst_string _); _ } as s -> s
            | [%expr [%e? t].value] -> [%expr Ir.Tnode.debug_name [%e t].value]
            | [%expr [%e? t].grad] -> [%expr Ir.Tnode.debug_name [%e t].value ^ ".grad"]
            | t -> [%expr Ir.Tnode.debug_name [%e t].value])
        in
        let res2 = loop ~proj_in_scope expr2 in
        let block =
          match res2.typ with
          | Code _ -> res2.expr
          | _ ->
              Ast_builder.Default.pexp_extension ~loc
              @@ Location.error_extensionf ~loc
                   "ppx_ocannl %%cd: only code can be commented, e.g. assignments or t.forward, \
                    t.backprop, t.zero_grads"
        in
        {
          res2 with
          expr =
            [%expr
              let __comment_block = [%e block] in
              {
                Ir.Assignments.asgns =
                  Ir.Assignments.Block_comment
                    ( String.concat_array ~sep:" " [%e Ast_helper.Exp.array ~loc:pexp_loc elements],
                      __comment_block.Ir.Assignments.asgns );
                embedded_nodes = __comment_block.Ir.Assignments.embedded_nodes;
              }];
        }
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident bin_op; _ }; _ }]
             ([%e? rhs1], [%e? rhs2])
             ~projections:[%e? projections])]
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident bin_op; _ }; _ }]
             [%e? rhs1]
             [%e? rhs2]
             ~projections:[%e? projections])]
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident bin_op; _ }; _ }]
             [%e? rhs1]
             ([%e? rhs2] ~projections:[%e? projections]))] ->
        (* TODO: when clause not needed here and below, it's an error if bin_op is not a primitive
           binary op. But this is error-prone with regard to ordering of the clauses. *)
        process_assign_binop ~accu_op ~lhs ~bin_op ~rhs1 ~rhs2 ~projections ~proj_in_scope:true ()
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident tern_op; _ }; _ }]
             ([%e? rhs1], [%e? rhs2], [%e? rhs3])
             ~projections:[%e? projections])]
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident tern_op; _ }; _ }]
             [%e? rhs1]
             [%e? rhs2]
             [%e? rhs3]
             ~projections:[%e? projections])] ->
        process_assign_ternop ~accu_op ~lhs ~tern_op ~rhs1 ~rhs2 ~rhs3 ~projections
          ~proj_in_scope:true ()
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident un_op; _ }; _ }]
             [%e? rhs]
             ~projections:[%e? projections])]
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          (* FIXME: this was never needed as prefix operators bind tighter? *)
          ([%e? { pexp_desc = Pexp_ident { txt = Lident un_op; _ }; _ }]
             ([%e? rhs] ~projections:[%e? projections]))]
      when Hashtbl.mem unary_ops un_op ->
        process_assign_unop ~accu_op ~lhs ~un_op ~rhs ~projections ~proj_in_scope:true ()
    | [%expr
        [%e? lhs]
        =: [%e? { pexp_desc = Pexp_ident { txt = Lident vec_un_op; _ }; _ }]
             [%e? rhs]
             ~projections:[%e? projections]]
      when Hashtbl.mem vec_unary_ops vec_un_op ->
        process_vec_unop ~lhs ~vec_un_op ~rhs ~projections ~proj_in_scope:true ()
    | [%expr
        [%e? lhs] =: [%e? { pexp_desc = Pexp_ident { txt = Lident vec_un_op; _ }; _ }] [%e? rhs]]
      when Hashtbl.mem vec_unary_ops vec_un_op && proj_in_scope ->
        process_vec_unop ~lhs ~vec_un_op ~rhs ~proj_in_scope ()
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? rhs] ~projections:[%e? projections])] ->
        process_assign_unop ~accu_op ~lhs ~un_op:"id" ~rhs ~projections ~proj_in_scope:true ()
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident bin_op; _ }; _ }]
             ([%e? rhs1], [%e? rhs2])
             ~logic:[%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ } as logic])]
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident bin_op; _ }; _ }]
             [%e? rhs1]
             [%e? rhs2]
             ~logic:[%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ } as logic])]
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident bin_op; _ }; _ }]
             [%e? rhs1]
             ([%e? rhs2]
                ~logic:
                  [%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ } as logic]))]
      ->
        let logic =
          let loc = s_loc in
          if String.equal spec "." then [%expr Shape.Pointwise_bin]
          else if String.equal spec "@" then [%expr Shape.Compose]
          else [%expr Shape.Einsum ([%e logic], [])]
        in
        let _, bin_op = binary_op bin_op in
        process_raw_binop ~accu_op ~lhs ~bin_op ~rhs1 ~rhs2 ~logic
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident tern_op; _ }; _ }]
             ([%e? rhs1], [%e? rhs2], [%e? rhs3])
             ~logic:[%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ }])]
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident tern_op; _ }; _ }]
             [%e? rhs1]
             [%e? rhs2]
             [%e? rhs3]
             ~logic:[%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ }])] ->
        let logic =
          let loc = s_loc in
          if String.equal spec "." then [%expr Shape.Pointwise_bin]
          else if String.equal spec "@" then [%expr Shape.Compose]
          else
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%cd: expected <.> or <@>, found <%s> -- einsum notation for ternary \
                  operators not supported yet, see issue #305"
                 spec
        in
        let _, tern_op = ternary_op tern_op in
        process_raw_ternop ~accu_op ~lhs ~tern_op ~rhs1 ~rhs2 ~rhs3 ~logic
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          (([%e? { pexp_desc = Pexp_ident { txt = Lident unop_ident; _ }; _ }] [%e? rhs])
             ~logic:[%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ } as logic])]
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident unop_ident; _ }; _ }]
             [%e? rhs]
             ~logic:[%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ } as logic])]
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          (* FIXME: this was never needed as prefix operators bind tighter? *)
          ([%e? { pexp_desc = Pexp_ident { txt = Lident unop_ident; _ }; _ }]
             ([%e? rhs]
                ~logic:
                  [%e? { pexp_desc = Pexp_constant (Pconst_string (spec, s_loc, _)); _ } as logic]))]
      when Hashtbl.mem unary_ops unop_ident ->
        (* Handle both un_op priority levels -- where application binds tighter and less tight. *)
        let logic =
          let loc = s_loc in
          if String.equal spec "." then [%expr Shape.Pointwise_un]
          else if String.equal spec "T" then [%expr Shape.Transpose]
          else [%expr Shape.Permute ([%e logic], [])]
        in
        let _, un_op = Hashtbl.find_exn unary_ops unop_ident loc in
        process_raw_unop ~accu_op ~lhs ~un_op ~rhs ~logic
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident bin_op; _ }; _ }] ([%e? rhs1], [%e? rhs2]))]
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident bin_op; _ }; _ }] [%e? rhs1] [%e? rhs2])]
      when is_assignment accu_op && Hashtbl.mem binary_ops bin_op && proj_in_scope ->
        process_assign_binop ~accu_op ~lhs ~bin_op ~rhs1 ~rhs2 ~proj_in_scope ()
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident tern_op; _ }; _ }]
             ([%e? rhs1], [%e? rhs2], [%e? rhs3]))]
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident tern_op; _ }; _ }]
             [%e? rhs1]
             [%e? rhs2]
             [%e? rhs3])]
      when is_assignment accu_op && Hashtbl.mem ternary_ops tern_op && proj_in_scope ->
        process_assign_ternop ~accu_op ~lhs ~tern_op ~rhs1 ~rhs2 ~rhs3 ~proj_in_scope ()
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident un_op; _ }; _ }] [%e? rhs])]
      when is_assignment accu_op && Hashtbl.mem unary_ops un_op && proj_in_scope ->
        process_assign_unop ~accu_op ~lhs ~un_op ~rhs ~proj_in_scope ()
    | [%expr [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }] [%e? lhs] [%e? rhs]]
      when is_assignment accu_op && proj_in_scope ->
        process_assign_unop ~accu_op ~lhs ~un_op:"id" ~rhs ~proj_in_scope ()
    | [%expr
        [%e? lhs] =: [%e? { pexp_desc = Pexp_ident { txt = Lident vec_un_op; _ }; _ }] [%e? rhs]]
      when Hashtbl.mem vec_unary_ops vec_un_op && proj_in_scope ->
        process_vec_unop ~lhs ~vec_un_op ~rhs ~proj_in_scope ()
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident bin_op; _ }; _ }] ([%e? rhs1], [%e? rhs2]))]
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident bin_op; _ }; _ }] [%e? rhs1] [%e? rhs2])]
      when is_assignment accu_op && Hashtbl.mem binary_ops bin_op ->
        let logic, bin_op = binary_op bin_op in
        process_raw_binop ~accu_op ~lhs ~bin_op ~rhs1 ~rhs2 ~logic
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident tern_op; _ }; _ }]
             ([%e? rhs1], [%e? rhs2], [%e? rhs3]))]
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident tern_op; _ }; _ }]
             [%e? rhs1]
             [%e? rhs2]
             [%e? rhs3])]
      when is_assignment accu_op && Hashtbl.mem ternary_ops tern_op ->
        let logic, tern_op = ternary_op tern_op in
        process_raw_ternop ~accu_op ~lhs ~tern_op ~rhs1 ~rhs2 ~rhs3 ~logic
    | [%expr
        [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }]
          [%e? lhs]
          ([%e? { pexp_desc = Pexp_ident { txt = Lident un_op; _ }; _ }] [%e? rhs])]
      when is_assignment accu_op && Hashtbl.mem unary_ops un_op ->
        let logic, un_op = Hashtbl.find_exn unary_ops un_op loc in
        process_raw_unop ~accu_op ~lhs ~un_op ~rhs ~logic
    | [%expr [%e? { pexp_desc = Pexp_ident { txt = Lident accu_op; _ }; _ }] [%e? lhs] [%e? rhs]]
      when is_assignment accu_op ->
        process_raw_unop ~accu_op ~lhs ~un_op:[%expr Ir.Ops.Identity] ~rhs
          ~logic:[%expr Shape.Pointwise_un]
    | [%expr [%e? expr1] [%e? expr2] [%e? expr3]] ->
        let res1 = loop ~proj_in_scope expr1 in
        let res2 = loop ~proj_in_scope expr2 in
        let res3 = loop ~proj_in_scope expr3 in
        let slot =
          List.hd_exn @@ List.sort [ res1.slot; res2.slot; res3.slot ] ~compare:compare_slots
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
        let slot = List.hd_exn @@ List.sort [ res1.slot; res2.slot ] ~compare:compare_slots in
        {
          vbs = reduce_vbss [ res1.vbs; res2.vbs ];
          typ = res1.typ;
          slot;
          expr = [%expr [%e res1.expr] [%e res2.expr]];
          array_opt_of_code = None;
        }
    | { pexp_desc = Pexp_function (args, constr, body); _ } as expr ->
        let proj_in_scope =
          proj_in_scope
          || List.exists args ~f:(function
            | { pparam_desc = Pparam_val ((Labelled s | Optional s), _, _); _ }
              when String.equal s "projections" ->
                true
            | _ -> false)
        in
        let bad_pun_hints =
          Set.union_list (module String)
          @@ bad_pun_hints
             :: List.map args ~f:(fun arg ->
                 match arg.pparam_desc with
                 | Pparam_val (_, _, pat) -> collect_pat_idents pat
                 | _ -> Set.empty (module String))
        in
        let result =
          match body with
          | Pfunction_body body ->
              let res = transl ~bad_pun_hints ~proj_in_scope body in
              {
                res with
                typ = Function;
                expr =
                  { expr with pexp_desc = Pexp_function (args, constr, Pfunction_body res.expr) };
              }
          | Pfunction_cases (cases, loc, attrs) ->
              let transformed_cases, cases_result =
                handle_cases ~bad_pun_hints ~proj_in_scope
                  (fun ~bad_pun_hints ~proj_in_scope -> transl ~bad_pun_hints ~proj_in_scope)
                  cases
              in
              {
                cases_result with
                typ = Function;
                expr =
                  {
                    expr with
                    pexp_desc =
                      Pexp_function (args, constr, Pfunction_cases (transformed_cases, loc, attrs));
                  };
              }
        in
        result
    | [%expr
        while [%e? _test_expr] do
          [%e? _body]
        done] ->
        {
          default_result with
          typ = Code { is_commented = false };
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
          typ = Code { is_commented = false };
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
          typ = Code { is_commented = false };
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
          typ = Code { is_commented = false };
          slot = Nonslot;
          expr = [%expr Ir.Assignments.sequence [ [%e res1.expr]; [%e res2.expr] ]];
          array_opt_of_code = res2.array_opt_of_code;
        }
    | [%expr if [%e? expr1] then [%e? expr2] else [%e? expr3]] ->
        let res2 = loop ~proj_in_scope expr2 in
        let res3 = loop ~proj_in_scope expr3 in
        let typ = if is_unknown res2.typ then res3.typ else res2.typ in
        let slot = List.hd_exn @@ List.sort [ res2.slot; res3.slot ] ~compare:compare_slots in
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
          typ = Code { is_commented = false };
          slot = Nonslot;
          expr = [%expr if [%e expr1] then [%e res2.expr] else Ir.Assignments.empty_comp];
          array_opt_of_code = res2.array_opt_of_code;
        }
    | { pexp_desc = Pexp_match (expr1, cases); _ } ->
        let transformed_cases, cases_result =
          handle_cases ~bad_pun_hints ~proj_in_scope transl cases
        in
        { cases_result with expr = { expr with pexp_desc = Pexp_match (expr1, transformed_cases) } }
    | { pexp_desc = Pexp_let (_recflag, _bindings, _body); _ } ->
        (* TODO(#80): to properly support local bindings, we need to collect the type
           environment. *)
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
    | _ -> { default_result with typ = Unknown }
  in
  let res = transl ~proj_in_scope:false ~bad_pun_hints:(Set.empty (module String)) expr in
  match (res.typ, ident_label) with
  | Code { is_commented = false }, Some string_expr ->
      let loc = res.expr.pexp_loc in
      {
        res with
        expr =
          [%expr
            let uncommented_comp = [%e res.expr] in
            {
              Ir.Assignments.embedded_nodes = uncommented_comp.Ir.Assignments.embedded_nodes;
              asgns =
                Ir.Assignments.Block_comment
                  ([%e string_expr], uncommented_comp.Ir.Assignments.asgns);
            }];
        typ = Code { is_commented = true };
      }
  | _ -> res

let translate ?ident_label expr =
  let ident_label, is_ignore =
    match ident_label with
    | Some [%pat? _] -> (None, true)
    | Some label -> (Some (pat2string label), false)
    | None -> (None, false)
  in
  let res = translate ?ident_label expr in
  let loc = res.expr.pexp_loc in
  let expr = res.expr in
  ( res.vbs,
    if is_ignore then
      [%expr
        Tensor.with_unchanged_roots ~f:(fun () ->
            let open! NTDSL.O in
            [%e expr])]
    else
      [%expr
        let open! NTDSL.O in
        [%e expr]] )

let expr_expander ~loc ~path = expr_expander_with_punning translate ~loc ~path
let str_expander ~loc ~path = str_expander_with_punning translate ~loc ~path
