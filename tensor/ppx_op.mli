(** The [%op] syntax extension: translates tensor expressions into [Tensor.t]-building code under a
    DSL's [O], lifting inline declarations and giving tensor functions shape schemes.
    [docs/syntax_extensions.md] is the reference for the syntax; {!Ppx_ocannl} registers these. *)

open Ppxlib

val translate :
  ?dsl_name:string ->
  ?ident_label:pattern ->
  ?bind_vbs_under_open:bool ->
  expression ->
  value_binding list * expression
(** Translates under [open dsl_name.O] (default [TDSL]), labelled by [ident_label]; inline
    declarations are bound inside the open, or returned with [~bind_vbs_under_open:false]. *)

val expr_expander : loc:location -> path:string -> expression -> expression
(** The [[%op ...]] expression expander; on a [let], translates the bindings but not the body. *)

val str_expander : loc:location -> path:string -> structure -> structure_item
(** The [[%%op ...]] structure-item expander; [let rec] and [[@@ocannl.no_scheme]] skip schemes. *)

val schemed_definitions : structure -> (string * bool) list
(** The schemes the [%%op] items get: each name, and [true] when traced at its [()] stage. *)
