open Base

open Ppxlib

module A = Ast_builder.Default

let rec pat2expr pat =
  let loc = pat.ppat_loc in
  match pat.ppat_desc with
  | Ppat_constraint (pat', typ) ->
    Ast_builder.Default.pexp_constraint ~loc (pat2expr pat') typ
  | Ppat_alias (_, ident)
  | Ppat_var ident ->
    Ast_builder.Default.pexp_ident ~loc {ident with txt = Lident ident.txt}
  | _ ->
     Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
       "ppx_minidebug requires a pattern identifier here: try using an `as` alias."

let rec pat2pat_res pat =
  let loc = pat.ppat_loc in
  match pat.ppat_desc with
  | Ppat_constraint (pat', _) -> pat2pat_res pat'
  | Ppat_alias (_, ident)
  | Ppat_var ident -> Ast_builder.Default.ppat_var ~loc {ident with txt = ident.txt ^ "__res"}
  | _ ->
    Ast_builder.Default.ppat_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_minidebug requires a pattern identifier here: try using an `as` alias."

let rec splice_lident ~id_prefix = function
  | Lident id -> Lident (id_prefix ^ id)
  | Ldot (path, id) -> Ldot (path, id_prefix ^ id)
  | Lapply (f, a) -> Lapply (splice_lident ~id_prefix f, a)

let log_preamble ?(message="") ~loc () =
  (* TODO: use Format and Sexp.pp instead of generating strings. *)
  [%expr
    let fname = [%e A.estring ~loc loc.loc_start.pos_fname] in
    Caml.Format.fprintf Debug_runtime.ppf
      "@[\"%s\":%d:%d-%d:%d@ at time UTC@ %s: %s@]" fname
      [%e A.eint ~loc loc.loc_start.pos_lnum]
      [%e A.eint ~loc (loc.loc_start.pos_cnum - loc.loc_start.pos_bol)]
      [%e A.eint ~loc loc.loc_end.pos_lnum]
      [%e A.eint ~loc (loc.loc_end.pos_cnum - loc.loc_end.pos_bol)]
      (Core.Time_ns.to_string_utc @@ Core.Time_ns.now())
      [%e A.estring ~loc message]]

let log_value ~loc ~t_lident_loc ~descr_loc exp =
  let converter = A.pexp_ident ~loc 
      {t_lident_loc with txt = splice_lident ~id_prefix:"sexp_of_" t_lident_loc.txt} in
  [%expr
     [%e log_preamble ~loc:t_lident_loc.loc ()];
     Caml.Format.fprintf Debug_runtime.ppf "%s =@ %a@ "
       [%e A.estring ~loc:descr_loc.loc descr_loc.txt] Sexp.pp_hum ([%e converter] [%e exp])]

exception Not_transforming

let rec collect_fun accu = function
  | [%expr fun [%p? arg] -> [%e? body]] as exp -> collect_fun ((arg, exp.pexp_loc)::accu) body
  | [%expr ([%e? body] :
              [%t? {ptyp_desc=(
                  Ptyp_constr (t_lident_loc, []) | Ptyp_poly (_, {ptyp_desc=Ptyp_constr (t_lident_loc, []); _})
                ); _} as typ ] ) ] ->
    List.rev accu, body, Some t_lident_loc, Some typ
  | body -> List.rev accu, body, None, None

let rec expand_fun body = function
  | [] -> body
  | (arg, loc)::args -> [%expr fun [%p arg] -> [%e expand_fun body args]]

let debug_fun callback bind descr_loc t_lident_loc_opt1 exp =
  let args, body, t_lident_loc_opt2, typ_opt = collect_fun [] exp in
  let loc = exp.pexp_loc in
  let t_lident_loc =
    match t_lident_loc_opt1, t_lident_loc_opt2 with
    | Some t_lident_loc, _ | None, Some t_lident_loc -> t_lident_loc
    | None, None -> raise Not_transforming in
  let arg_logs = List.filter_map args ~f:(function
      | [%pat? ([%p? {ppat_desc=Ppat_var descr_loc; _} as pat ] :
                  [%t? {ptyp_desc=(
                      Ptyp_constr (t_lident_loc, [])
                    | Ptyp_poly (_, {ptyp_desc=Ptyp_constr (t_lident_loc, []); _})
                    ); _} ] ) ], loc ->
        Some (log_value ~loc ~t_lident_loc ~descr_loc (pat2expr pat))
      | _ -> None
  ) in
  let init = log_preamble ~message:descr_loc.txt ~loc () in
  let arg_logs = List.fold arg_logs ~init ~f:(fun e1 e2 -> [%expr [%e e1]; [%e e2]]) in
  let result = pat2pat_res bind in
  let body =
    [%expr
      Caml.Format.pp_open_hvbox Debug_runtime.ppf 2;
      [%e arg_logs];
      let [%p result] = [%e callback body] in
      [%e log_value ~loc ~t_lident_loc ~descr_loc (pat2expr result)];
      Caml.Format.pp_close_box Debug_runtime.ppf ();
      [%e pat2expr result]] in
  let body =
    match typ_opt with None -> body | Some typ -> [%expr ([%e body] : [%t typ])] in
  expand_fun body args

let debug_binding callback vb =
  let pat = vb.pvb_pat in
  let loc = vb.pvb_loc in
  let descr_loc, t_lident_loc_opt =
    match vb.pvb_pat, vb.pvb_expr with
    | [%pat? ([%p? {ppat_desc=Ppat_var descr_loc; _} ] :
                [%t? {ptyp_desc=(
                    Ptyp_constr (t_lident_loc, []) | Ptyp_poly (_, {ptyp_desc=Ptyp_constr (t_lident_loc, []); _})
                ); _} ] ) ], _ ->
      descr_loc, Some t_lident_loc
    | {ppat_desc=Ppat_var descr_loc; _}, 
      [%expr ([%e? _exp] : [%t? {ptyp_desc=(
        Ptyp_constr (t_lident_loc, []) | Ptyp_poly (_, {ptyp_desc=Ptyp_constr (t_lident_loc, []); _})
      ); _} ])] ->
        descr_loc, Some t_lident_loc
    | {ppat_desc=Ppat_var descr_loc; _}, _ ->
        descr_loc, None
    | _ -> (* raise Not_transforming *)  {txt="made_up_d_2"; loc}, None in
  match vb.pvb_expr.pexp_desc, t_lident_loc_opt with
  | Pexp_fun _, _ -> 
    {vb with pvb_expr = debug_fun callback vb.pvb_pat descr_loc t_lident_loc_opt vb.pvb_expr}
  | _, Some t_lident_loc ->
    let result = pat2pat_res pat in
    let exp =
      [%expr
        Caml.Format.pp_open_hvbox Debug_runtime.ppf 2;
        [%e log_preamble ~message:descr_loc.txt ~loc:descr_loc.loc ()];
        let [%p result] = [%e callback vb.pvb_expr] in
        [%e log_value ~loc ~t_lident_loc ~descr_loc (pat2expr result)];
        Caml.Format.pp_close_box Debug_runtime.ppf ();
        [%e pat2expr result]] in
    {vb with pvb_expr = exp}
  | _ -> raise Not_transforming

let traverse =
  object (self)
    inherit Ast_traverse.map (* _with_expansion_context *) as super

    method! expression e =
      let callback e = super#expression e in
      match e with
      | { pexp_desc=Pexp_let (rec_flag, bindings, body); pexp_loc=_; _ } ->
        let bindings = List.map bindings ~f:(fun vb ->
              try debug_binding callback vb
              with Not_transforming -> {vb with pvb_expr = callback vb.pvb_expr}) in
        let body = self#expression body in
        { e with pexp_desc = Pexp_let (rec_flag, bindings, body) }
      | _ -> super#expression e

    method! structure_item si =
      let callback e = super#expression e in
      match si with
      | { pstr_desc=Pstr_value (rec_flag, bindings); pstr_loc=_; _ } ->
        let bindings = List.map bindings ~f:(fun vb ->
            try debug_binding callback vb
            with Not_transforming -> {vb with pvb_expr = callback vb.pvb_expr}) in
        { si with pstr_desc = Pstr_value (rec_flag, bindings) }
      | _ -> super#structure_item si
  end

let debug_this_expander ~ctxt:_ payload =
  let callback e = traverse#expression e in
  match payload with
  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
    (* This is the [let%debug_this ... in] use-case: do not debug the whole body. *)
     let bindings = List.map bindings ~f:(debug_binding callback) in
     {payload with pexp_desc=Pexp_let (recflag, bindings, body)}
  | expr -> expr

let debug_expander ~ctxt:_ payload = traverse#expression payload

let str_expander ~loc ~path:_ payload =
  match List.map payload ~f:(fun si -> traverse#structure_item si) with
  | [item] -> item
  | items ->
    Ast_helper.Str.include_ {
      pincl_mod = Ast_helper.Mod.structure items;
      pincl_loc = loc;
      pincl_attributes = [] }

let rules =
  [Ppxlib.Context_free.Rule.extension  @@
   Extension.V3.declare "minidebug" Extension.Context.expression Ast_pattern.(single_expr_payload __)
     debug_expander;
   Ppxlib.Context_free.Rule.extension  @@
   Extension.V3.declare "debug_this" Extension.Context.expression Ast_pattern.(single_expr_payload __) 
     debug_this_expander;
   Ppxlib.Context_free.Rule.extension  @@
   Extension.declare "minidebug" Extension.Context.structure_item Ast_pattern.(pstr __) str_expander;
     ]

let () =
  Driver.register_transformation
    ~rules
    "ppx_minidebug"
