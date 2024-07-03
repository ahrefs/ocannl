open Base
open Ppxlib

let rules =
  [
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "cd" Extension.Context.expression Ast_pattern.(single_expr_payload __)
    @@ Ppx_cd.expr_expander;
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "cd" Extension.Context.structure_item
         Ast_pattern.(pstr __)
         Ppx_cd.str_expander;
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "op" Extension.Context.expression
         Ast_pattern.(single_expr_payload __)
         Ppx_op.expr_expander;
    Ppxlib.Context_free.Rule.extension
    @@ Extension.declare "op" Extension.Context.structure_item
         Ast_pattern.(pstr __)
         Ppx_op.str_expander;
  ]

let () = Driver.register_transformation ~rules "ppx_ocannl"
