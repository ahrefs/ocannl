(* Negative control: metaquot's expansion names ppxlib.ast's units (Ppxlib_ast.Ast), never ppxlib's
   own, so the declared ppxlib is unused. *)
let one ~loc = [%expr 1]
