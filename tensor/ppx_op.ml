open Base
open Ppxlib
open Ppx_shared

let dsl_path dsl_name field = Ldot (Lident dsl_name, field)

let dsl_fn ~loc dsl_name field =
  Ast_builder.Default.pexp_ident ~loc { txt = dsl_path dsl_name field; loc }

let dsl_open_o ~loc dsl_name expr =
  let popen_expr = Ast_helper.Mod.ident ~loc { txt = dsl_path dsl_name "O"; loc } in
  Ast_helper.Exp.open_ ~loc
    (Ast_builder.Default.open_infos ~loc ~expr:popen_expr ~override:Override)
    expr

(* Whether a pattern in [body] binds [name]. *)
let rebinds name body =
  let found = ref false in
  let visitor =
    object
      inherit Ast_traverse.iter as super

      method! pattern p =
        (match p.ppat_desc with
        | (Ppat_var { txt; _ } | Ppat_alias (_, { txt; _ })) when String.equal txt name ->
            found := true
        | _ -> ());
        super#pattern p
    end
  in
  visitor#expression body;
  !found

(* Where a name that is bound too late for the inline declarations was written: as a parameter of
   the function directly under [%op] when it has no [()], or of a function below -- after the [()],
   or nested; or by a binder of the body below the lift point, named by the construct ([let], a
   [match] or [function] case, a [for] loop; gh-ocannl-2058). *)
type late_kind = No_unit | Under_function | Body_binder of string

let params_idents params =
  List.concat_map params ~f:(fun param ->
      match param.pparam_desc with
      | Pparam_val (_, _, pat) -> Set.to_list (collect_pat_idents pat)
      | Pparam_newtype _ -> [])

(* An inline declaration is lifted: bound where the enclosing [()] is applied, or without one where
   the function is defined. A field of it that names a parameter bound later than that used to be
   OCaml's "Unbound value" at the parameter's use, or a silent capture of an outer value of the same
   name (gh-ocannl-1987); and so did a name bound in the body (gh-ocannl-2058). [late]: those names,
   innermost first. *)
let guard_late ~late ~decl (e : expression) =
  let found = ref None in
  let visitor =
    object
      inherit Ast_traverse.iter as super

      method! expression e' =
        (match e'.pexp_desc with
        | Pexp_ident { txt = Lident n; loc } when Option.is_none !found -> (
            match List.Assoc.find late ~equal:String.equal n with
            | Some kind when not (rebinds n e) -> found := Some (n, kind, loc)
            | _ -> ())
        | _ -> ());
        super#expression e'
    end
  in
  visitor#expression e;
  match !found with
  | None -> e
  | Some (n, No_unit, loc) ->
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "ppx_ocannl %%op: the inline declaration { %s } is created once, when the function is \
            defined, where the parameter %s is not bound. Give the function a unit parameter with \
            %s before it -- fun ~%s () ... -- so that the declaration is created at each () \
            application"
           decl n n n
  | Some (n, Under_function, loc) ->
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "ppx_ocannl %%op: the inline declaration { %s } is created where the enclosing () is \
            applied (without a unit parameter: when the function is defined), where the parameter \
            %s is not bound yet. Bind %s before the unit parameter: fun ~%s () ..."
           decl n n n
  | Some (n, Body_binder construct, loc) ->
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "ppx_ocannl %%op: the inline declaration { %s } is created where the enclosing () is \
            applied (without a unit parameter: when the function is defined), outside the %s that \
            binds %s. Bind %s before the unit parameter -- fun ~%s () ... -- or outside %%op"
           decl construct n n n

(* [late] extended with the names [pat] binds, by the body's [construct]. *)
let late_binder ~late construct pat =
  List.map (Set.to_list (collect_pat_idents pat)) ~f:(fun n -> (n, Body_binder construct)) @ late

let make_p ~dsl_name ~opt_label ~late ~loc ?value ?values ?param_init ~extra_args name =
  let guard = guard_late ~late ~decl:name.txt in
  let param_init = Option.map param_init ~f:guard in
  let extra_args = List.map extra_args ~f:(fun (label, value) -> (label, guard value)) in
  let more_label = match opt_label with Some more_label -> more_label | None -> [%expr None] in
  let value = match value with Some c -> [%expr Some [%e c]] | None -> [%expr None] in
  let values = match values with Some c -> [%expr Some [%e c]] | None -> [%expr None] in
  let param_init =
    match param_init with
    | Some c -> [%expr Some [%e add_module_qualifier_to_applied_function ~module_name:"NTDSL" c]]
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
  let name = Ast_helper.Exp.constant ~loc (Ast_helper.Const.string ~loc:name.loc name.txt) in
  let param_op = dsl_fn ~loc dsl_name "param" in
  let base_expr =
    [%expr
      [%e param_op] ?more_label:[%e more_label] ?value:[%e value] ?values:[%e values]
        ?param_init:[%e param_init] [%e name]]
  in
  let with_extra_args =
    if List.is_empty extra_args then base_expr else Ast_helper.Exp.apply ~loc base_expr extra_args
  in
  [%expr [%e with_extra_args] ()]

let make_vb ~dsl_name ~opt_label ~late ?value ?param_init ~extra_args ~loc name =
  let pat = Ast_helper.Pat.var ~loc:name.loc name in
  let v = make_p ~dsl_name ~opt_label ~late ~loc ?value ?param_init ~extra_args name in
  let vb = Ast_helper.Vb.mk ~loc pat v in
  (pat, vb)

let make_vb_nd ~dsl_name ~opt_label ~late ~init_nd ~extra_args ~loc name =
  let pat = Ast_helper.Pat.var ~loc:name.loc name in
  let values, batch_dims, output_dims, input_dims = ndarray_constant init_nd in
  let v =
    if not @@ List.is_empty batch_dims then
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "ppx_ocannl param cannot have batch dims: define a constant or remove the array syntax."
    else
      (* Threads tensor-literal axis labels through to <kind>_axes when present. *)
      let input_name, input_expr = axes_or_dims_named ~loc ~kind:"input" input_dims in
      let output_name, output_expr = axes_or_dims_named ~loc ~kind:"output" output_dims in
      let extra_args =
        ({ txt = Lident input_name; loc }, input_expr)
        :: ({ txt = Lident output_name; loc }, output_expr)
        :: extra_args
      in
      make_p ~dsl_name ~opt_label ~late ~loc ~values ~extra_args name
  in
  let vb = Ast_helper.Vb.mk ~loc pat v in
  (pat, vb)

(* First-leaf heuristic for disambiguating numeric tensor literals (ndarray constants) from block
   tensor literals (stacking): if the first leaf reached by descending through tuples, lists, and
   arrays is a numeric constant, the whole literal is an ndarray constant; otherwise it is a block
   tensor whose components are tensor expressions. *)
let rec is_ndarray_constant_expr expr =
  match expr with
  | [%expr [%e? _] :: [%e? _]] -> (
      match collect_list [] expr with e :: _ -> is_ndarray_constant_expr e | [] -> true)
  | [%expr []] | [%expr [||]] -> true
  | _ -> (
      match expr.pexp_desc with
      | Pexp_constant (Pconst_float _) | Pexp_constant (Pconst_integer _) -> true
      (* Descend through axis-basis annotations, e.g. [((1., 2., 3.) : features)], so numeric
         literals carrying a dimension-basis tag stay on the ndarray path rather than being read as
         a stack. *)
      | Pexp_constraint (e, _) -> is_ndarray_constant_expr e
      | Pexp_tuple (e :: _) | Pexp_array (e :: _) -> is_ndarray_constant_expr e
      | _ -> false)

(* The components of a block tensor, and the operands of a concat, are let-bound to this prefix and
   their position where they can have an effect, so that their tensors are created in source order
   ([in_source_order], gh-ocannl-1869), which lengthens the prefix where a component already
   mentions such a name -- a nested literal's bindings, or a variable of the user's. *)
let component_prefix = "__op_component"

(* Desugars a block tensor literal into a call to the named [stack] operation (which composes
   [einsum1] unsqueeze + [concat] along a fresh leading axis). [axis_kind] selects the axis the new
   dimension is inserted into: list -> output, array -> batch, top-level tuple -> input. *)
let translate_block_tensor ~loc ~loop ~label ~opt_label:_ axis_kind elems =
  match elems with
  | [] ->
      ( no_vbs,
        Ast_builder.Default.pexp_extension ~loc
        @@ Location.error_extensionf ~loc
             "ppx_ocannl %%op: block tensor requires at least one component" )
  | _ ->
      let vbss, translated = List.unzip (List.map elems ~f:loop) in
      let axis_expr =
        match axis_kind with
        | `Output -> [%expr `Output]
        | `Input -> [%expr `Input]
        | `Batch -> [%expr `Batch]
      in
      ( reduce_vbss vbss,
        in_source_order ~loc ~prefix:component_prefix translated ~f:(fun components ->
            [%expr
              stack ?label:[%e opt_expr ~loc label] [%e axis_expr]
                [%e Ast_builder.Default.pexp_array ~loc components]]) )

(* {2 Automatic shape schemes (gh-ocannl-1751)}

   A [%op] function whose returned function takes only tensors gets a shape scheme: the closure
   built at its [()] application traces it (a function without a [()] stage: its first application),
   and each application checks it before the body runs. The ppx has no types, so which parameters
   are tensors is decided syntactically: - an unlabelled variable parameter is a tensor: the
   convention [docs/syntax_extensions.md] states for the parameters to the right of [()] and of a
   function directly under [%op]; - a parameter annotated [(x : Tensor.t)] is a tensor, and one
   annotated [(x : Tensor.t option)] -- labelled, optional ([?(x : Tensor.t option)]) or not -- an
   option-valued tensor (gh-ocannl-1792), whose presence patterns get one scheme each; - an
   unannotated labelled [~x] (no default) is a tensor only with evidence: it occurs directly as an
   operand of a tensor operation below, and its name is never rebound in the body; - other optional
   and defaulted parameters, and any other pattern, are not. A function with any other parameter
   stays unschemed, its expansion unchanged. A wrong classification -- an unlabelled non-tensor
   parameter, a shadowed operator -- is a compile error, which [[@@ocannl.no_scheme]] escapes; an
   unlabelled parameter with [non_tensor_evidence] is refused with the fix named (gh-ocannl-1987).
   The result needs no classification: the trace ignores it. *)

let no_scheme_attr = "ocannl.no_scheme"
let scheme_attr = "ocannl.scheme"
let has_attr_named name attrs = List.exists attrs ~f:(fun a -> String.equal a.attr_name.txt name)

let drop_attr_named name attrs =
  List.filter attrs ~f:(fun a -> not (String.equal a.attr_name.txt name))

let opted_out (e : expression) = has_attr_named no_scheme_attr e.pexp_attributes

let opt_out (e : expression) =
  if opted_out e then e
  else
    let loc = e.pexp_loc in
    {
      e with
      pexp_attributes =
        Ast_helper.Attr.mk ~loc { txt = no_scheme_attr; loc } (PStr []) :: e.pexp_attributes;
    }

(* Operators and functions whose every positional operand is a tensor in [TDSL.O] and [NTDSL.O]. *)
let tensor_operand_ops =
  Hash_set.of_list
    (module String)
    [
      "+";
      "-";
      "*";
      "*.";
      "/.";
      "@^";
      "@-";
      "add";
      "sub";
      "mul";
      "div";
      "relu";
      "sat01";
      "exp";
      "log";
      "exp2";
      "log2";
      "sin";
      "cos";
      "sqrt";
      "recip";
      "recip_sqrt";
      "neg";
      "tanh";
      "where";
      "fma";
      "mul3";
    ]

(* Operators whose first positional operand is a tensor: the einsum and concat operators (their
   second is a spec or an operand applied to one), and the pointwise power. *)
let first_operand_ops =
  Hash_set.of_list
    (module String)
    ("**." :: Einsum_parser.concat_operator_with_generated_specs
    :: (Einsum_parser.binary_operators_with_generated_specs
      @ Einsum_parser.unary_operators_with_generated_specs))

let is_ident name (e : expression) =
  match e.pexp_desc with Pexp_ident { txt = Lident n; _ } -> String.equal n name | _ -> false

(* Whether [name] occurs in [body] directly as a tensor operand (outside [%oc]). *)
let tensor_operand_evidence name body =
  let found = ref false in
  let visitor =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        match e.pexp_desc with
        | Pexp_extension ({ txt = "oc"; _ }, _) -> ()
        | Pexp_apply ({ pexp_desc = Pexp_ident { txt = Lident op; _ }; _ }, args) ->
            let positional =
              List.filter_map args ~f:(function Nolabel, a -> Some a | _ -> None)
            in
            let operands =
              if Hash_set.mem tensor_operand_ops op then positional
              else if Hash_set.mem first_operand_ops op then List.take positional 1
              else []
            in
            let operands =
              List.concat_map operands ~f:(fun a ->
                  match a.pexp_desc with
                  (* A concat's operand tuple. *)
                  | Pexp_tuple elems
                    when String.equal op Einsum_parser.concat_operator_with_generated_specs ->
                      elems
                  | _ -> [ a ])
            in
            if List.exists operands ~f:(is_ident name) then found := true;
            super#expression e
        | _ -> super#expression e
    end
  in
  visitor#expression body;
  !found

(* The integer-to-float conversions, spelled exactly: their argument is an integer. *)
let int_to_float_conversions = [ "Float.of_int"; "float_of_int"; "Int.to_float" ]

(* Why [name] cannot be a tensor, when [body] uses it directly as an integer or a float (outside
   [%oc]): a stride or dilation of a literal einsum spec, of an operator (gh-ocannl-1987) or of a
   named form (gh-ocannl-2060), the operand of a number embedding or the exponent of the pointwise
   power (gh-ocannl-1987), or the argument of an integer-to-float conversion (gh-ocannl-2060). *)
let non_tensor_evidence name body =
  let found = ref None in
  let note why = if Option.is_none !found then found := Some why in
  let positional args = List.filter_map args ~f:(function Nolabel, a -> Some a | _ -> None) in
  let check_spec spec =
    if List.mem (einsum_spec_identifiers spec) name ~equal:String.equal then
      note (Printf.sprintf "the einsum spec %S reads it as an integer" spec)
  in
  let visitor =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        match e.pexp_desc with
        | Pexp_extension ({ txt = "oc"; _ }, _) -> ()
        | Pexp_apply ({ pexp_desc = Pexp_ident { txt = fn; _ }; _ }, args) ->
            (match (fn, positional args) with
            | Lident "!.", a :: _ when is_ident name a -> note "it is the float that !. embeds"
            | Lident "!..", a :: _ when is_ident name a -> note "it is the integer that !.. embeds"
            | Lident "**.", [ _; a ] when is_ident name a -> note "it is the float exponent of **."
            | Lident op, operands
              when List.mem Einsum_parser.operators_with_generated_specs op ~equal:String.equal ->
                List.iter operands ~f:(fun a ->
                    match a.pexp_desc with
                    | Pexp_constant (Pconst_string (spec, _, _))
                    | Pexp_apply ({ pexp_desc = Pexp_constant (Pconst_string (spec, _, _)); _ }, _)
                      ->
                        check_spec spec
                    | _ -> ())
            | _, a :: _
              when is_ident name a
                   && List.mem int_to_float_conversions (Longident.name fn) ~equal:String.equal ->
                note (Printf.sprintf "it is the integer that %s converts" (Longident.name fn))
            | _ -> (
                match named_form_spec e with
                | Some { pexp_desc = Pexp_constant (Pconst_string (spec, _, _)); _ } ->
                    check_spec spec
                | _ -> ()));
            super#expression e
        | _ -> super#expression e
    end
  in
  visitor#expression body;
  if rebinds name body then None else !found

(* The first unlabelled, unannotated parameter with [non_tensor_evidence], as the error naming the
   fix. Such a parameter is read as a tensor -- a label source without [()], a scheme's argument
   after it -- which used to be a type error at its use, naming neither. *)
let unlabelled_non_tensor params body =
  List.find_map params ~f:(fun param ->
      match param.pparam_desc with
      | Pparam_val (Nolabel, None, { ppat_desc = Ppat_var { txt; loc }; _ }) ->
          Option.map (non_tensor_evidence txt body) ~f:(fun why ->
              Ast_builder.Default.pexp_extension ~loc
              @@ Location.error_extensionf ~loc
                   "ppx_ocannl %%op: the unlabelled parameter %s is taken for a tensor, but %s. \
                    Label it (~%s), or bind it before a unit parameter: fun %s () ..."
                   txt why txt txt)
      | _ -> None)

(* A tensor parameter's kind: a tensor, or an option-valued tensor (gh-ocannl-1792), whose presence
   patterns get one scheme each. *)
type param_kind = Required | Option_valued

(* Syntactically [Tensor.t] (any qualification ending in [Tensor.t]). *)
let is_tensor_type (t : core_type) =
  match t.ptyp_desc with
  | Ptyp_constr ({ txt = Ldot ((Lident "Tensor" | Ldot (_, "Tensor")), "t"); _ }, []) -> true
  | _ -> false

(* The kind an annotation [(p : T)] states, if [T] is [Tensor.t] or [Tensor.t option]. *)
let annotated_kind (t : core_type) =
  match t.ptyp_desc with
  | _ when is_tensor_type t -> Some Required
  | Ptyp_constr ({ txt = Lident "option"; _ }, [ arg ]) when is_tensor_type arg ->
      Some Option_valued
  | _ -> None

(* The tensor parameters of a function, or [None] when one parameter is not a tensor. *)
let tensor_params params body =
  Option.all
  @@ List.map params ~f:(fun param ->
      match param.pparam_desc with
      | Pparam_val (Nolabel, None, { ppat_desc = Ppat_var { txt; _ }; _ }) ->
          Some (Nolabel, txt, Required)
      (* An annotation is type-checked evidence: by OCaml at the parameter, and again by the
         expansion's argument list. *)
      | Pparam_val
          ( ((Nolabel | Labelled _) as l),
            None,
            { ppat_desc = Ppat_constraint ({ ppat_desc = Ppat_var { txt; _ }; _ }, t); _ } ) ->
          Option.map (annotated_kind t) ~f:(fun kind -> (l, txt, kind))
      (* An un-defaulted [?p] binds the option itself: only [Tensor.t option] type-checks. *)
      | Pparam_val
          ( (Optional _ as l),
            None,
            { ppat_desc = Ppat_constraint ({ ppat_desc = Ppat_var { txt; _ }; _ }, t); _ } ) -> (
          match annotated_kind t with
          | Some Option_valued -> Some (l, txt, Option_valued)
          | _ -> None)
      | Pparam_val (Labelled l, None, { ppat_desc = Ppat_var { txt; _ }; _ })
        when tensor_operand_evidence txt body && not (rebinds txt body) ->
          Some (Labelled l, txt, Required)
      | _ -> None)

(* Marks a function eligible for a scheme, carrying its name expression. *)
let mark_schemed ~name (e : expression) =
  let loc = e.pexp_loc in
  {
    e with
    pexp_attributes =
      Ast_helper.Attr.mk ~loc { txt = scheme_attr; loc } (PStr [ Ast_helper.Str.eval ~loc name ])
      :: e.pexp_attributes;
  }

let schemed_name (e : expression) =
  List.find_map e.pexp_attributes ~f:(fun a ->
      if String.equal a.attr_name.txt scheme_attr then
        match a.attr_payload with
        | PStr [ { pstr_desc = Pstr_eval (name, _); _ } ] -> Some name
        | _ -> None
      else None)

let eligible_function (e : expression) =
  match e.pexp_desc with
  | Pexp_function ((_ :: _ as params), _, Pfunction_body body) when not (opted_out e) ->
      Option.is_some (tensor_params params body)
  | _ -> false

(* Marks [e]'s tail function -- through [let], [;], [let open] and [let module] -- when eligible. *)
let rec mark_tail ~name (e : expression) =
  match e.pexp_desc with
  | Pexp_function _ -> if eligible_function e then mark_schemed ~name e else e
  | Pexp_let (rf, vbs, body) -> { e with pexp_desc = Pexp_let (rf, vbs, mark_tail ~name body) }
  | Pexp_sequence (e1, e2) -> { e with pexp_desc = Pexp_sequence (e1, mark_tail ~name e2) }
  | Pexp_open (d, body) -> { e with pexp_desc = Pexp_open (d, mark_tail ~name body) }
  | Pexp_letmodule (n, m, body) ->
      { e with pexp_desc = Pexp_letmodule (n, m, mark_tail ~name body) }
  | _ -> e

(* The schemed form of a translated function [fn] with tensor parameters [params]: [let
   __ocannl_body = fn in let __ocannl_scheme = make ~name ~arity trace in fun params -> apply
   __ocannl_scheme [args] (fun () -> __ocannl_body params)]. With an option-valued parameter, the
   [_variants] entry points, over [Tensor.t option] lists: a required tensor [x] is passed as [Some
   x], an option-valued one as it is. [at_unit]: traced at the [()] stage ([closure]), otherwise at
   the first application ([deferred]). *)
let wrap_schemed ~loc ~at_unit ~name ~params (fn : expression) =
  let open Ast_builder.Default in
  let fn = { fn with pexp_attributes = drop_attr_named scheme_attr fn.pexp_attributes } in
  let variants =
    List.exists params ~f:(fun (_, _, kind) -> match kind with Option_valued -> true | _ -> false)
  in
  let vars = List.map params ~f:(fun (_, v, _) -> evar ~loc v) in
  let call =
    pexp_apply ~loc (evar ~loc "__ocannl_body")
      (List.map2_exn params vars ~f:(fun (l, _, _) v -> (l, v)))
  in
  let arg_exprs, arg_pats =
    if not variants then (vars, List.map params ~f:(fun (_, v, _) -> pvar ~loc v))
    else
      List.unzip
      @@ List.map params ~f:(fun (_, v, kind) ->
          match kind with
          | Required -> ([%expr Some [%e evar ~loc v]], [%pat? Some [%p pvar ~loc v]])
          | Option_valued -> (evar ~loc v, pvar ~loc v))
  in
  let tensors = elist ~loc arg_exprs in
  let tracer_pat = plist ~loc arg_pats in
  let make, apply =
    match (at_unit, variants) with
    | true, false -> ([%expr Tensor.Scheme.For_ppx.closure], [%expr Tensor.Scheme.For_ppx.apply])
    | false, false -> ([%expr Tensor.Scheme.For_ppx.deferred], [%expr Tensor.Scheme.For_ppx.apply])
    | true, true ->
        ( [%expr Tensor.Scheme.For_ppx.closure_variants],
          [%expr Tensor.Scheme.For_ppx.apply_variants] )
    | false, true ->
        ( [%expr Tensor.Scheme.For_ppx.deferred_variants],
          [%expr Tensor.Scheme.For_ppx.apply_variants] )
  in
  let shape_arg =
    if not variants then (Labelled "arity", eint ~loc (List.length params))
    else
      ( Labelled "optional",
        elist ~loc
          (List.map params ~f:(fun (l, v, kind) ->
               match kind with
               | Required -> [%expr None]
               | Option_valued ->
                   let n = match l with Nolabel -> v | Labelled n | Optional n -> n in
                   [%expr Some [%e estring ~loc n]])) )
  in
  (* The body's value is returned, for the runtime to keep alive while it reads the tensors the body
     made as the scheme's results (gh-ocannl-1801). *)
  let tracer = [%expr function [%p tracer_pat] -> [%e call] | _ -> assert false] in
  let make_call = pexp_apply ~loc make [ (Labelled "name", name); shape_arg; (Nolabel, tracer) ] in
  let returned =
    pexp_function ~loc
      (List.map params ~f:(fun (l, v, _) ->
           { pparam_desc = Pparam_val (l, None, pvar ~loc v); pparam_loc = loc }))
      None
      (Pfunction_body [%expr [%e apply] __ocannl_scheme [%e tensors] (fun () -> [%e call])])
  in
  [%expr
    let __ocannl_body = [%e fn] in
    let __ocannl_scheme = [%e make_call] in
    [%e returned]]

(* A function application in [%op] code, run with its source site recorded (gh-ocannl-2007): the
   shape-scheme check of an application inside it names the line. The DSL's operators, and the named
   einsum forms that spell them, are not wrapped; any other application may be (or reach) a schemed
   [%op] function, which the ppx cannot tell without types. The site is [__POS__] at the
   application's location, which the compiler fills in: the expansion itself spells no line. *)
let at_site ~loc app = [%expr Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> [%e app])]

(* [at_site]'s inverse, everywhere in an expression whose applications are no scheme's: an
   initializer, where [add_module_qualifier_to_applied_function] reads the applied functions -- of
   the root, and of a conditional's branches, a [let]'s or a sequence's body, and so on. *)
let strip_at_site =
  let mapper =
    object
      inherit Ast_traverse.map as super

      method! expression e =
        match super#expression e with
        | [%expr Tensor.Scheme.For_ppx.at_site [%e? _] (fun () -> [%e? app])] -> app
        | e -> e
    end
  in
  mapper#expression

let default_scheme_name ~loc ?name () =
  match name with
  | Some name -> Ast_builder.Default.estring ~loc name
  | None ->
      Ast_builder.Default.estring ~loc
        (Printf.sprintf "%%op function at %s:%d" loc.loc_start.pos_fname loc.loc_start.pos_lnum)

let concat_fn ~loc = Ast_builder.Default.evar ~loc (snd Einsum_parser.concat_operator_function)

let rec translate ~dsl_name ~num_configs ~is_toplevel ?(in_block = false) ?scheme_name ~opt_label
    ~late ?label expr =
  let loc = expr.pexp_loc in
  let loop = translate ~dsl_name ~num_configs ~is_toplevel:false ~opt_label ~late in
  (* [loop] below a binder of the body: [construct] binds the names of [pat] there. *)
  let loop_under construct pat =
    translate ~dsl_name ~num_configs ~is_toplevel:false ~opt_label
      ~late:(late_binder ~late construct pat)
  in
  (* Block-element loop: like [loop] but flags that the immediate sub-expression sits in
     block-tensor element position, so a bare tuple there denotes an input-axis stack (not OCaml
     tuple grouping). Used only for the operands of a block-tensor stack. *)
  let block_loop =
    translate ~dsl_name ~num_configs ~is_toplevel:false ~in_block:true ~opt_label ~late
  in
  let dsl_fn = dsl_fn ~loc dsl_name in
  match expr with
  (* %oc anti-quotation: preserve the expression without transformation *)
  | [%expr [%oc [%e? oc_expr]]] -> (no_vbs, oc_expr)
  | { pexp_desc = Pexp_extension ({ txt = "oc"; _ }, _); _ } ->
      ( no_vbs,
        Ast_builder.Default.pexp_extension ~loc
        @@ Location.error_extensionf ~loc "%%oc expects a single expression" )
  | { pexp_desc = Pexp_constant (Pconst_float _); _ } ->
      (no_vbs, [%expr [%e dsl_fn "number"] ?label:[%e opt_expr ~loc label] [%e expr]])
  | { pexp_desc = Pexp_constant (Pconst_integer (_, Some ('L' | 'l'))); _ } ->
      (no_vbs, [%expr [%e dsl_fn "bits"] [%e expr]])
  | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
      (no_vbs, [%expr [%e dsl_fn "number"] (Float.of_int [%e expr])])
  (* Axis-labelled constant literals: a type-annotation on a numeric literal sets the output
     dimension basis (semantic tag) of the resulting size-1 axis, e.g. [(2.0 : rgb)], [(1.0 : p)].
     This replaces the old char-literal form (['q' 2.0]), which could not carry multi-character
     tags. Only a bare type constructor name is accepted as the basis. *)
  | {
   pexp_desc =
     Pexp_constraint
       ( ({ pexp_desc = Pexp_constant (Pconst_float _); _ } as f),
         { ptyp_desc = Ptyp_constr ({ txt = Lident basis; _ }, []); ptyp_loc; _ } );
   _;
  } ->
      let axis = Ast_builder.Default.estring ~loc:ptyp_loc basis in
      ( no_vbs,
        [%expr [%e dsl_fn "number"] ?label:[%e opt_expr ~loc label] ~axis_basis:[%e axis] [%e f]] )
  | {
   pexp_desc =
     Pexp_constraint
       ( ({ pexp_desc = Pexp_constant (Pconst_integer (_, Some ('L' | 'l'))); _ } as i),
         { ptyp_desc = Ptyp_constr ({ txt = Lident basis; _ }, []); ptyp_loc; _ } );
   _;
  } ->
      let axis = Ast_builder.Default.estring ~loc:ptyp_loc basis in
      ( no_vbs,
        [%expr [%e dsl_fn "bits"] ?label:[%e opt_expr ~loc label] ~axis_basis:[%e axis] [%e i]] )
  | {
   pexp_desc =
     Pexp_constraint
       ( ({ pexp_desc = Pexp_constant (Pconst_integer _); _ } as i),
         { ptyp_desc = Ptyp_constr ({ txt = Lident basis; _ }, []); ptyp_loc; _ } );
   _;
  } ->
      let axis = Ast_builder.Default.estring ~loc:ptyp_loc basis in
      ( no_vbs,
        [%expr
          [%e dsl_fn "number"] ?label:[%e opt_expr ~loc label] ~axis_basis:[%e axis]
            (Float.of_int [%e i])] )
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
        ([%e? expr2] [%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }])]
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
        [%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }]]
    when String.contains spec_str '>'
         && String.equal op_ident Einsum_parser.concat_operator_with_generated_specs ->
      let vbs1, rhses =
        match expr1.pexp_desc with
        | Pexp_tuple elems ->
            let vbss, elems = List.unzip (List.map elems ~f:loop) in
            (reduce_vbss vbss, elems)
        | _ ->
            let vbs, e = loop expr1 in
            (vbs, [ e ])
      in
      let spec = substitute_identifiers_in_einsum_spec ~loc spec_str in
      ( vbs1,
        in_source_order ~loc ~prefix:component_prefix rhses ~f:(fun rhses ->
            [%expr
              [%e concat_fn ~loc] ?label:[%e opt_expr ~loc label] [%e spec]
                [%e Ast_builder.Default.pexp_array ~loc rhses]]) )
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ }]
        [%e? expr1]
        ([%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }]
           ([%e? { pexp_desc = Pexp_constant (Pconst_string _); _ } as head] :: [%e? rest]))]
    when String.contains spec_str '>'
         && String.equal op_ident Einsum_parser.concat_operator_with_generated_specs ->
      let capture_vbs, capture_dims_expr = collect_capture_labels ~loc head rest in
      let vbs1, rhses =
        match expr1.pexp_desc with
        | Pexp_tuple elems ->
            let vbss, elems = List.unzip (List.map elems ~f:loop) in
            (reduce_vbss vbss, elems)
        | _ ->
            let vbs, e = loop expr1 in
            (vbs, [ e ])
      in
      let spec = substitute_identifiers_in_einsum_spec ~loc spec_str in
      let combined_vbs = reduce_vbss [ vbs1; capture_vbs ] in
      ( combined_vbs,
        in_source_order ~loc ~prefix:component_prefix rhses ~f:(fun rhses ->
            [%expr
              [%e concat_fn ~loc] ?label:[%e opt_expr ~loc label]
                ~capture_dims:[%e capture_dims_expr] [%e spec]
                [%e Ast_builder.Default.pexp_array ~loc rhses]]) )
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
        ([%e? expr2]
           [%e? { pexp_desc = Pexp_constant (Pconst_string (spec_str, _, _)); _ }]
           ([%e? { pexp_desc = Pexp_constant (Pconst_string _); _ } as head] :: [%e? rest]))]
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
          let pat, vb = make_vb ~dsl_name ~opt_label ~late ~value ~extra_args ~loc name in
          ([ vb ], pat2expr pat)
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
          let pat, vb = make_vb ~dsl_name ~opt_label ~late ~value ~extra_args ~loc name in
          ([ vb ], pat2expr pat)
      | _ ->
          ( no_vbs,
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%op: record field label must be a simple identifier" ))
  | {
   pexp_desc =
     Pexp_record
       ( (first_label, (({ pexp_desc = Pexp_array _; _ } | [%expr [%e? _] :: [%e? _]]) as init_nd))
         :: extra_args,
         None );
   _;
  } -> (
      (* Record syntax with array/list initialization *)
      match first_label.txt with
      | Lident tensor_name ->
          let name = { loc = first_label.loc; txt = tensor_name } in
          let pat, vb = make_vb_nd ~dsl_name ~opt_label ~late ~init_nd ~extra_args ~loc name in
          (* Note: expect a type error if batch_dims exist or extra_args modify the shape *)
          ([ vb ], pat2expr pat)
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
            | { pexp_desc = Pexp_extension ({ txt = "oc"; _ }, _); _ } -> (no_vbs, Some first_value)
            | _ ->
                (* An initializer is no schemed application, and [make_p] reads its applied function
                   to qualify it: not wrapped. *)
                let vbs, e = loop first_value in
                (vbs, Some (strip_at_site e))
          in
          let name = { loc = first_label.loc; txt = tensor_name } in
          let pat, vb = make_vb ~dsl_name ~opt_label ~late ?param_init ~extra_args ~loc name in
          (* Combine with any bindings from the initialization *)
          let all_vbs = reduce_vbss [ init_vbs; [ vb ] ] in
          (all_vbs, pat2expr pat)
      | _ ->
          ( no_vbs,
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl %%op: record field label must be a simple identifier" ))
  (* List [a; b]: numeric first leaf -> ndarray constant; otherwise block tensor stacking along a
     new output axis. *)
  | [%expr [%e? _] :: [%e? _]] as list_expr ->
      if is_ndarray_constant_expr list_expr then
        (no_vbs, ndarray_op ?label ~ndarray_fn:(dsl_fn "ndarray") list_expr)
      else
        translate_block_tensor ~loc ~loop:block_loop ~label ~opt_label `Output
          (collect_list [] list_expr)
  (* Array [|a; b|]: numeric first leaf -> ndarray constant; otherwise block tensor stacking along a
     new batch axis. *)
  | { pexp_desc = Pexp_array elems; _ } as arr_expr ->
      if is_ndarray_constant_expr arr_expr then
        (no_vbs, ndarray_op ?label ~ndarray_fn:(dsl_fn "ndarray") arr_expr)
      else translate_block_tensor ~loc ~loop:block_loop ~label ~opt_label `Batch elems
  (* A tensor literal whose outermost axis container carries an axis-label annotation, e.g. [([ 1.;
     2.; 3. ] : rgb)] or [([| … |] : batch)]. *)
  | {
   pexp_desc = Pexp_constraint (({ pexp_desc = Pexp_array _; _ } | [%expr [%e? _] :: [%e? _]]), _);
   _;
  } ->
      (no_vbs, ndarray_op ?label ~ndarray_fn:(dsl_fn "ndarray") expr)
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
  (* 2+ tuple (a, b): numeric first leaf -> ndarray constant; otherwise block tensor stacking along
     a new input axis. Recognized at the top level, or in block-tensor element position (so that a
     tuple nested inside a list/array/tuple introduces an input axis, e.g. [(a, b); (c, d)]). Tuples
     used as operator argument grouping or function arguments are handled by the [Pexp_apply] arm
     below (which keeps their OCaml-tuple meaning), so they are unaffected. *)
  | { pexp_desc = Pexp_tuple elems; _ } when (is_toplevel || in_block) && List.length elems >= 2 ->
      if is_ndarray_constant_expr expr then
        (no_vbs, ndarray_op ?label ~ndarray_fn:(dsl_fn "ndarray") expr)
      else translate_block_tensor ~loc ~loop:block_loop ~label ~opt_label `Input elems
  (* The uncurried forms [op (a, b)] and [op (a, b, c)]: the call head is the operator alone, as in
     the curried form (gh-ocannl-1986). *)
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ } as op]
        ([%e? expr2], [%e? expr3])]
    when Hashtbl.mem binary_ops op_ident ->
      let e1 = [%expr [%e op] ?label:[%e opt_expr ~loc label]] in
      let vbs2, e2 = loop expr2 in
      let vbs3, e3 = loop expr3 in
      (reduce_vbss [ vbs2; vbs3 ], [%expr [%e e1] [%e e2] [%e e3]])
  | [%expr
      [%e? { pexp_desc = Pexp_ident { txt = Lident op_ident; _ }; _ } as op]
        ([%e? expr2], [%e? expr3], [%e? expr4])]
    when Hashtbl.mem ternary_ops op_ident ->
      let e1 = [%expr [%e op] ?label:[%e opt_expr ~loc label]] in
      let vbs2, e2 = loop expr2 in
      let vbs3, e3 = loop expr3 in
      let vbs4, e4 = loop expr4 in
      (reduce_vbss [ vbs2; vbs3; vbs4 ], [%expr [%e e1] [%e e2] [%e e3] [%e e4]])
  | [%expr [%e? arg] |> [%e? fn]] ->
      (* TODO: this is a workaround for optional arguments spoiling type inference. We should
         consider removing ?label arguments from Make_DSL.O -- not sure what benefit we get from
         them. Then consider removing this case, but maybe not... *)
      let vbs1, arg = loop arg in
      let vbs2, fn = loop fn in
      let vbs = reduce_vbss [ vbs1; vbs2 ] in
      ( vbs,
        [%expr
          let __x = [%e arg] in
          [%e at_site ~loc [%expr [%e fn] __x]]] )
  | { pexp_desc = Pexp_apply (fn_expr, args); _ } ->
      (* Smart application handling with unit-parameter heuristic: If there's a unit () argument,
         don't transform args before it *)
      let unit_position =
        List.find_mapi args ~f:(fun i (_, arg_expr) ->
            match arg_expr with [%expr ()] -> Some i | _ -> None)
      in
      let vbs_fn, e_fn = loop ?label fn_expr in
      let vbs_args, processed_args =
        List.unzip
        @@ List.mapi args ~f:(fun i (arg_label, arg_expr) ->
            match unit_position with
            | Some unit_pos when i < unit_pos ->
                (* Before unit: preserve as OCaml expression *)
                (no_vbs, (arg_label, arg_expr))
            | _ -> (
                (* After unit or no unit: transform *)
                match arg_expr.pexp_desc with
                | Pexp_tuple elems ->
                    (* A tuple passed as a function argument is regular OCaml tuple grouping, not an
                       input-axis block-tensor stack (that is only the top-level tuple form). Keep
                       the tuple structure, but still translate each element so %op content inside
                       it (operators, inline params, numeric literals) is expanded rather than left
                       as raw OCaml. *)
                    let vbss, elems = List.unzip (List.map elems ~f:loop) in
                    (reduce_vbss vbss, (arg_label, { arg_expr with pexp_desc = Pexp_tuple elems }))
                | _ ->
                    let vbs, e = loop arg_expr in
                    (vbs, (arg_label, e))))
      in
      let all_vbs = reduce_vbss (vbs_fn :: vbs_args) in
      (* A named einsum form's literal spec, which the translation leaves as written, reads
         identifiers as an operator's does (gh-ocannl-2060). *)
      let app =
        substitute_named_form_spec (Ast_builder.Default.pexp_apply ~loc e_fn processed_args)
      in
      (* A named einsum form is the DSL's operation spelled by name ([einsum1 "ab=>a" x] is [x ++
         "ab=>a"]): no scheme's application, and its own body applies none, so it is not wrapped
         either -- both spellings expand alike. *)
      let operator =
        is_named_spec_function fn_expr
        ||
        match fn_expr.pexp_desc with
        | Pexp_ident { txt = Lident op_ident; _ } ->
            is_primitive_op op_ident || is_operator op_ident
        | _ -> false
      in
      (all_vbs, if operator then app else at_site ~loc app)
  | { pexp_desc = Pexp_function (args, constr, body); _ } when is_toplevel -> (
      (* Check if there's a unit parameter or a labeled parameter with label "label" *)
      let rec find_unit acc = function
        | [] -> None
        | ({ pparam_desc = Pparam_val (Nolabel, _, [%pat? ()]); _ } as unit_param) :: rest ->
            Some (List.rev acc, unit_param, rest)
        | hd :: rest -> find_unit (hd :: acc) rest
      in
      let orig_body =
        match body with
        | Pfunction_body body -> body
        | Pfunction_cases _ -> { expr with pexp_desc = Pexp_function ([], None, body) }
      in
      let labelled_tensor_evidence name =
        tensor_operand_evidence name orig_body && not (rebinds name orig_body)
      in
      (* The parameter labelled [label] is the [string list] that enriches the result's label,
         unless it is a tensor by the evidence above (gh-ocannl-1987): accepted since
         gh-ocannl-1885, it was ignored. Without [()], inline declarations cannot take it: they are
         created when the function is defined. *)
      let is_label_list ?(shadowed = []) pat =
        match pat.ppat_desc with
        | Ppat_var { txt; _ } ->
            List.mem shadowed txt ~equal:String.equal || not (labelled_tensor_evidence txt)
        | Ppat_constraint ({ ppat_desc = Ppat_var _; _ }, t) -> Option.is_none (annotated_kind t)
        | _ -> false
      in
      (* The [string list option] the inline declarations take as [?more_label]: [~label] and a
         defaulted [?(label = ...)] bind the list, an un-defaulted [?label] the option itself
         (gh-ocannl-1885). *)
      let rec find_label_param ?(shadowed = []) ~for_result = function
        | [] -> None
        | { pparam_desc = Pparam_val (Optional "label", None, pat); _ } :: _
          when (not for_result) || is_label_list ~shadowed pat ->
            Some (pat2expr pat)
        | { pparam_desc = Pparam_val ((Labelled "label" | Optional "label"), _, pat); _ } :: _
          when (not for_result) || is_label_list ~shadowed pat ->
            let loc = pat.ppat_loc in
            Some [%expr Some [%e pat2expr pat]]
        | _ :: rest -> find_label_param ~shadowed ~for_result rest
      in
      match find_unit [] args with
      | Some (before_unit, unit_param, after_unit) ->
          (* With a unit parameter, always bind the collected inline definitions. Captures directly
             in this body are bound here too, since the unit function is their innermost function;
             those under the parameters after the unit are already bound per call. *)
          let opt_label = find_label_param ~for_result:false before_unit in
          (* Capture the application label at [()], before a body binder (or a parameter after [()])
             can shadow it. Tensor-valued and ignored [label] parameters are not sources. *)
          let result_vbs, label =
            match
              find_label_param ~shadowed:(params_idents after_unit) ~for_result:true before_unit
            with
            | None -> ([], label)
            | Some more_label ->
                let used = Hash_set.create (module String) in
                let visitor =
                  object
                    inherit Ast_traverse.iter as super

                    method! longident_loc lid =
                      (match lid.txt with Lident name -> Hash_set.add used name | _ -> ());
                      super#longident_loc lid

                    method! pattern p =
                      (match p.ppat_desc with
                      | Ppat_var { txt; _ } | Ppat_alias (_, { txt; _ }) -> Hash_set.add used txt
                      | _ -> ());
                      super#pattern p
                  end
                in
                visitor#expression expr;
                let rec fresh name = if Hash_set.mem used name then fresh (name ^ "_") else name in
                let name = fresh "__ocannl_result_label" in
                let binding_label = Option.value label ~default:[%expr []] in
                let value =
                  [%expr
                    [%e binding_label]
                    @ match [%e more_label] with Some label -> label | None -> []]
                in
                let vb = Ast_helper.Vb.mk ~loc (Ast_builder.Default.pvar ~loc name) value in
                ([ vb ], Some (Ast_builder.Default.evar ~loc name))
          in
          let vbs, inner_body =
            let body =
              match (after_unit, body) with
              | [], Pfunction_body body -> body
              | _ ->
                  {
                    expr with
                    pexp_desc = Pexp_function (after_unit, constr, body);
                    pexp_attributes = drop_attr_named no_scheme_attr expr.pexp_attributes;
                  }
            in
            (* The function the closure returns gets its scheme, traced when the closure is
               built. *)
            let body =
              if opted_out expr || not (String.equal dsl_name "TDSL") then body
              else mark_tail ~name:(default_scheme_name ~loc ?name:scheme_name ()) body
            in
            translate ~dsl_name ~num_configs ~is_toplevel:false ~opt_label ~late ?label body
          in
          let inner_body = let_opt ~loc (result_vbs @ vbs) inner_body in
          ( no_vbs,
            {
              expr with
              pexp_desc =
                Pexp_function (before_unit @ [ unit_param ], constr, Pfunction_body inner_body);
              pexp_attributes = drop_attr_named no_scheme_attr expr.pexp_attributes;
            } )
      | None -> (
          (* No unit parameter, everything is "after_unit": each tensor parameter is a label source.
             An optional one is not (gh-ocannl-1787): [?x] is an option in the body, never a tensor,
             and a defaulted one is configuration ([?(stride = 2)]). A labelled one is by the
             evidence [tensor_params] asks for -- a [Tensor.t] annotation, or a direct operand use
             (gh-ocannl-1885): read as a tensor, [~hid_dim] failed to type-check at the parameter
             ("not a record type"), with nothing naming the missing [()]. *)
          let late = List.map (params_idents args) ~f:(fun n -> (n, No_unit)) @ late in
          let loop = translate ~dsl_name ~num_configs ~is_toplevel:false ~opt_label ~late in
          let labels =
            Option.to_list label
            @ List.filter_map args ~f:(function
              | {
                  pparam_desc =
                    Pparam_val (((Labelled "label" | Optional "label") as l), default, pat);
                  _;
                }
                when is_label_list pat -> (
                  let loc = pat.ppat_loc in
                  (* Without the annotation, which for an un-defaulted [?label] states the
                     option. *)
                  let v =
                    match pat.ppat_desc with
                    | Ppat_constraint (pat, _) -> pat2expr pat
                    | _ -> pat2expr pat
                  in
                  match (l, default) with
                  | Optional _, None ->
                      Some
                        [%expr match [%e v] with Some label -> (label : string list) | None -> []]
                  | _ -> Some [%expr ([%e v] : string list)])
              | { pparam_desc = Pparam_val (Optional _, _, _); _ } -> None
              (* An annotation other than [Tensor.t]: an option-valued tensor, which may be absent,
                 or a non-tensor (gh-ocannl-1792) -- not a label source. *)
              | { pparam_desc = Pparam_val (_, _, { ppat_desc = Ppat_constraint (_, t); _ }); _ }
                when match annotated_kind t with Some Required -> false | _ -> true ->
                  None
              | {
                  pparam_desc = Pparam_val (Labelled _, _, { ppat_desc = Ppat_var { txt; _ }; _ });
                  _;
                }
                when not (labelled_tensor_evidence txt) ->
                  None
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
                let vbs, body = bind_captures ~loc vbs body in
                (vbs, Pfunction_body body)
            | Pfunction_cases (cases, loc, attrs) ->
                let vbs, cases =
                  List.unzip
                  @@ List.map cases ~f:(fun ({ pc_lhs; pc_rhs; _ } as c) ->
                      (* Not [loop_under]: [late] here holds the parameters too. *)
                      let loop =
                        translate ~dsl_name ~num_configs ~is_toplevel:false ~opt_label
                          ~late:(late_binder ~late "function case" pc_lhs)
                      in
                      let vbs, pc_rhs = loop ~label pc_rhs in
                      let vbs, pc_rhs = bind_captures ~loc vbs pc_rhs in
                      (vbs, { c with pc_rhs }))
                in
                (List.concat vbs, Pfunction_cases (cases, loc, attrs))
          in
          let fn = { expr with pexp_desc = Pexp_function (args, constr, body) } in
          match unlabelled_non_tensor args orig_body with
          | Some error -> (no_vbs, error)
          | None ->
              let fn =
                match (expr.pexp_desc, String.equal dsl_name "TDSL" && not (opted_out expr)) with
                | Pexp_function (_ :: _, _, Pfunction_body orig_body), true -> (
                    match tensor_params args orig_body with
                    | Some params ->
                        wrap_schemed ~loc ~at_unit:false
                          ~name:(default_scheme_name ~loc ?name:scheme_name ())
                          ~params
                          {
                            fn with
                            pexp_attributes = drop_attr_named no_scheme_attr fn.pexp_attributes;
                          }
                    | None -> fn)
                | _ ->
                    { fn with pexp_attributes = drop_attr_named no_scheme_attr fn.pexp_attributes }
              in
              (vbs, fn)))
  | { pexp_desc = Pexp_function (args, constr, body); _ } -> (
      let orig_body = body in
      let scheme =
        match (schemed_name expr, body) with
        | Some name, Pfunction_body orig_body ->
            Option.map (tensor_params args orig_body) ~f:(fun params -> (name, params))
        | _ -> None
      in
      let late = List.map (params_idents args) ~f:(fun n -> (n, Under_function)) @ late in
      let loop = translate ~dsl_name ~num_configs ~is_toplevel:false ~opt_label ~late in
      let vbs, body =
        match body with
        | Pfunction_body body ->
            let vbs, body = loop ?label body in
            let vbs, body = bind_captures ~loc vbs body in
            (vbs, Pfunction_body body)
        | Pfunction_cases (cases, loc, attrs) ->
            let vbs, cases =
              List.unzip
              @@ List.map cases ~f:(fun ({ pc_lhs; pc_rhs; _ } as c) ->
                  let loop =
                    translate ~dsl_name ~num_configs ~is_toplevel:false ~opt_label
                      ~late:(late_binder ~late "function case" pc_lhs)
                  in
                  let vbs, pc_rhs = loop ?label pc_rhs in
                  let vbs, pc_rhs = bind_captures ~loc vbs pc_rhs in
                  (vbs, { c with pc_rhs }))
            in
            (List.concat vbs, Pfunction_cases (cases, loc, attrs))
      in
      let fn =
        {
          expr with
          pexp_desc = Pexp_function (args, constr, body);
          pexp_attributes =
            drop_attr_named no_scheme_attr (drop_attr_named scheme_attr expr.pexp_attributes);
        }
      in
      ( vbs,
        match (scheme, orig_body) with
        | Some (name, params), Pfunction_body orig_body -> (
            match unlabelled_non_tensor args orig_body with
            | Some error -> error
            | None -> wrap_schemed ~loc ~at_unit:true ~name ~params fn)
        | _ -> fn ))
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
      let vbs, body = loop_under "for loop" pat ?label body_expr in
      ( vbs,
        [%expr
          for [%p pat] = [%e init] to [%e final] do
            [%e body]
          done] )
  | [%expr
      for [%p? pat] = [%e? init] downto [%e? final] do
        [%e? body_expr]
      done] ->
      let vbs, body = loop_under "for loop" pat ?label body_expr in
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
        @@ List.map cases ~f:(fun ({ pc_lhs; pc_rhs; _ } as c) ->
            let vbs, pc_rhs = loop_under "match case" pc_lhs ?label pc_rhs in
            (vbs, { c with pc_rhs }))
      in
      (reduce_vbss vbss, { expr with pexp_desc = Pexp_match (expr1, cases) })
  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
      (* The names the bindings bind: in scope in the body, and in the right-hand sides of a [let
         rec]. *)
      let bound_late =
        List.fold bindings ~init:late ~f:(fun late vb -> late_binder ~late "let" vb.pvb_pat)
      in
      let loop_bound =
        translate ~dsl_name ~num_configs ~is_toplevel:false ~opt_label ~late:bound_late
      in
      let loop_rhs = match recflag with Recursive -> loop_bound | Nonrecursive -> loop in
      let vbss1, bindings =
        List.unzip
        @@ List.map bindings ~f:(fun binding ->
            let vbs, pvb_expr =
              loop_rhs ~label:[%expr [ [%e pat2string binding.pvb_pat] ]] binding.pvb_expr
            in
            (vbs, { binding with pvb_expr }))
      in
      let vbs2, body = loop_bound ?label body in
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

let translate ?(dsl_name = "TDSL") ?ident_label ?(bind_vbs_under_open = true) expr =
  let scheme_name =
    match ident_label with Some { ppat_desc = Ppat_var { txt; _ }; _ } -> Some txt | _ -> None
  in
  let vbs, expr =
    translate ~dsl_name ~num_configs:(ref 0) ~is_toplevel:true ?scheme_name ~opt_label:None ~late:[]
      ~label:(opt_pat2string_list ~loc:expr.pexp_loc ident_label)
      expr
  in
  let loc = expr.pexp_loc in
  let wrap_in_dsl_open expr =
    match ident_label with
    | Some [%pat? _] ->
        [%expr Tensor.with_unchanged_roots ~f:(fun () -> [%e dsl_open_o ~loc dsl_name expr])]
    | _ -> dsl_open_o ~loc dsl_name expr
  in
  if bind_vbs_under_open then (no_vbs, wrap_in_dsl_open (let_opt ~loc vbs expr))
  else (vbs, wrap_in_dsl_open expr)

let translate_tdsl ?ident_label expr = translate ?ident_label expr

(* [[@@ocannl.no_scheme]] on a binding, and [let rec] (whose right-hand side must stay a syntactic
   function), opt the bound function out of its scheme. *)
let opt_out_binding recflag vb =
  let attr = has_attr_named no_scheme_attr vb.pvb_attributes in
  match recflag with
  | Recursive ->
      {
        vb with
        pvb_expr = opt_out vb.pvb_expr;
        pvb_attributes = drop_attr_named no_scheme_attr vb.pvb_attributes;
      }
  | Nonrecursive when attr ->
      {
        vb with
        pvb_expr = opt_out vb.pvb_expr;
        pvb_attributes = drop_attr_named no_scheme_attr vb.pvb_attributes;
      }
  | Nonrecursive -> vb

let expr_expander ~loc ~path:_ payload =
  match payload with
  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
      (* Bind captures and inline declarations around the whole let so they remain visible to its
         body, but keep their initializer expressions within the DSL operator scope. *)
      let vbss, bindings =
        List.unzip
        @@ List.map bindings ~f:(fun vb ->
            let vb = opt_out_binding recflag vb in
            let vbs, v =
              translate ~bind_vbs_under_open:false ?ident_label:(Some vb.pvb_pat) vb.pvb_expr
            in
            (vbs, { vb with pvb_expr = v }))
      in
      let expr = { payload with pexp_desc = Pexp_let (recflag, bindings, body) } in
      let vbs =
        List.map (reduce_vbss vbss) ~f:(fun vb ->
            { vb with pvb_expr = dsl_open_o ~loc:vb.pvb_loc "TDSL" vb.pvb_expr })
      in
      let_opt ~loc vbs expr
  | expr ->
      let vbs, expr = translate_tdsl ?ident_label:None expr in
      let_opt ~loc vbs expr

let str_expander ~loc ~path payload =
  let payload =
    List.map payload ~f:(fun item ->
        match item.pstr_desc with
        | Pstr_value (recflag, bindings) ->
            {
              item with
              pstr_desc = Pstr_value (recflag, List.map bindings ~f:(opt_out_binding recflag));
            }
        | _ -> item)
  in
  str_expander_with_punning translate_tdsl ~loc ~path payload

(* The [%op] functions a structure's [let%op] items get a scheme for, as the expansion gives them:
   each scheme's name, and whether it is traced at its [()] stage ([true]) or at its first
   application. Lets a test hold a list of definitions to what the expansion schemes, rather than to
   a copy of it. *)
let schemed_definitions (str : structure) =
  let found = ref [] in
  let visitor =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_apply
            ( {
                pexp_desc =
                  Pexp_ident
                    { txt = Ldot (Ldot (Ldot (Lident "Tensor", "Scheme"), "For_ppx"), kind); _ };
                _;
              },
              args )
          when List.mem
                 [ "closure"; "deferred"; "closure_variants"; "deferred_variants" ]
                 kind ~equal:String.equal ->
            List.iter args ~f:(function
              | Labelled "name", { pexp_desc = Pexp_constant (Pconst_string (name, _, _)); _ } ->
                  found := (name, String.is_prefix kind ~prefix:"closure") :: !found
              | _ -> ())
        | _ -> ());
        super#expression e
    end
  in
  List.iter str ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_extension (({ txt = "op"; _ }, PStr payload), _) ->
          visitor#structure_item (str_expander ~loc:item.pstr_loc ~path:"" payload)
      | _ -> ());
  List.rev !found
