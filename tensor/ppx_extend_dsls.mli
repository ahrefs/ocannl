(** The [%%extend_dsls] syntax extension: translates value definitions with [%op] once per DSL and
    emits a [DSL_modules] module whose [TDSL] and [NTDSL] (and their [O]) include them. *)

open Ppxlib

val str_expander : loc:location -> path:string -> structure -> structure_item
(** The [[%%extend_dsls ...]] expander; each [O] variant takes [?label] and passes a final [()]. *)
