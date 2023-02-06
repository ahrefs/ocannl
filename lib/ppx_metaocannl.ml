open Ppxlib

let meta_bracket ~loc ~path:_ (expr: expression) =
   (* [%expr [%e expr] [@metaocaml.bracket] ] doesn't work *)
   {expr with pexp_attributes=({attr_name={txt="metaocaml.bracket"; loc=loc};
                                attr_loc=loc; attr_payload= PStr []}
                               ::expr.pexp_attributes)}
let meta_escape ~loc ~path:_ expr =
   (* [%expr [%e expr] [@metaocaml.escape] ] *)
   {expr with pexp_attributes=({attr_name={txt="metaocaml.escape"; loc=loc};
                                attr_loc=loc; attr_payload= PStr []}
                               ::expr.pexp_attributes)}

let expr_rule name extender =
  Context_free.Rule.extension  @@
  Extension.declare name Extension.Context.expression Ast_pattern.(single_expr_payload __) extender
let rules = [expr_rule "c" meta_bracket; expr_rule "e" meta_escape]

let () =
  Driver.register_transformation
    ~rules
    "ppx_metaocannl"
