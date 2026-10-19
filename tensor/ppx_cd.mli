(** The [%cd] syntax extension: translates assignment code over tensors into [Ir.Assignments.comp]
    values. [docs/syntax_extensions.md] is the reference for the syntax; {!Ppx_ocannl} registers
    these expanders. *)

open Ppxlib

val expr_expander : loc:location -> path:string -> expression -> expression
(** The [[%cd ...]] expression expander. *)

val str_expander : loc:location -> path:string -> structure -> structure_item
(** The [[%%cd ...]] structure-item expander. *)
