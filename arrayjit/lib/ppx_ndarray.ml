(* open Base *)
open Ppxlib

(* open Ppx_helper *)

(*
let expr_expander ~loc ~path:_ payload =
  match payload with
  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
      (* We are at the %ocannl annotation level: do not tranlsate the body. *)
      let bindings =
        List.map bindings ~f:(fun vb ->
            let v =
              (if is_cd dt then translate else translate_dt ~is_result:(is_rs dt))
                ~desc_label:vb.pvb_pat vb.pvb_expr
            in
            {
              vb with
              pvb_expr =
                [%expr
                  let open! NFDSL.O in
                  [%e v]];
            })
      in
      { payload with pexp_desc = Pexp_let (recflag, bindings, body) }
  | expr ->
      let expr = (if is_cd dt then translate else translate_dt ~is_result:(is_rs dt)) expr in
      [%expr
        let open! NFDSL.O in
        [%e expr]]

let flatten_str ~loc ~path:_ items =
  match items with
  | [ x ] -> x
  | _ ->
      Ast_helper.Str.include_
        { pincl_mod = Ast_helper.Mod.structure items; pincl_loc = loc; pincl_attributes = [] }

let translate_str ~dt ({ pstr_desc; _ } as str) =
  match pstr_desc with
  | Pstr_eval (expr, attrs) ->
      let expr = (if is_cd dt then translate else translate_dt ~is_result:(is_rs dt)) expr in
      let loc = expr.pexp_loc in
      {
        str with
        pstr_desc =
          Pstr_eval
            ( [%expr
                let open! NFDSL.O in
                [%e expr]],
              attrs );
      }
  | Pstr_value (recf, bindings) ->
      let f vb =
        let loc = vb.pvb_loc in
        let v =
          (if is_cd dt then translate else translate_dt ~is_result:(is_rs dt))
            ~desc_label:vb.pvb_pat vb.pvb_expr
        in
        {
          vb with
          pvb_expr =
            [%expr
              let open! NFDSL.O in
              [%e v]];
        }
      in
      { str with pstr_desc = Pstr_value (recf, List.map bindings ~f) }
  | _ -> str

let str_expander ~loc ~path (payload : structure_item list) =
  flatten_str ~loc ~path @@ List.map payload ~f:translate_str

let rules =
  [
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "nndarray" Extension.Context.expression
         Ast_pattern.(single_expr_payload __)
         expr_expander;
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "nndarray" Extension.Context.structure_item Ast_pattern.(pstr __) str_expander;
  ]
*)
let () = Driver.register_transformation ~rules:[] "ppx_nndarray"
