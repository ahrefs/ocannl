open Base
open Ppxlib

let rules =
  [
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "nn_cd" Extension.Context.expression Ast_pattern.(single_expr_payload __)
    @@ Ppx_nn_cd.expr_expander ~dt:Cd;
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "nn_cd" Extension.Context.structure_item Ast_pattern.(pstr __) Ppx_nn_cd.str_expander;
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "nn_op" Extension.Context.expression
         Ast_pattern.(single_expr_payload __)
         Ppx_nn_op.expr_expander;
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "nn_op" Extension.Context.structure_item Ast_pattern.(pstr __) Ppx_nn_op.str_expander;
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "nn_dt" Extension.Context.expression Ast_pattern.(single_expr_payload __)
    @@ Ppx_nn_cd.expr_expander ~dt:Dt;
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "nn_dt" Extension.Context.structure_item
         Ast_pattern.(pstr __)
         Ppx_nn_cd.str_expander_dt;
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "nn_rs" Extension.Context.expression Ast_pattern.(single_expr_payload __)
    @@ Ppx_nn_cd.expr_expander ~dt:Rs;
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "nn_rs" Extension.Context.structure_item
         Ast_pattern.(pstr __)
         Ppx_nn_cd.str_expander_rs;
  ]

let () = Driver.register_transformation ~rules "ppx_ocannl"
