type context [@@deriving sexp_of]
type code [@@deriving sexp_of]
type code_batch [@@deriving sexp_of]

val initialize : unit -> unit
val is_initialized : unit -> bool
val finalize : context -> unit
val sexp_of_context : context -> Sexplib.Sexp.t
val compile : ?name:string -> Indexing.unit_bindings -> Low_level.optimized -> code
val compile_batch : names:string option array -> Indexing.unit_bindings -> Low_level.optimized option array -> code_batch
val link : context -> code -> context * Indexing.lowered_bindings * (unit -> Tnode.work)

val link_batch :
  context -> code_batch -> context * Indexing.lowered_bindings * (unit -> Tnode.work) option array

val unsafe_cleanup : ?unsafe_shutdown:bool -> unit -> unit

val from_host : context -> Tnode.t -> bool
(** If the array is both hosted and in-context, copies from host to context and returns true. *)

val to_host : context -> Tnode.t -> bool
(** If the array is both hosted and in-context, copies from context to host and returns true. *)

type device

val init : device -> context
val await : device -> unit
val sexp_of_device : device -> Sexplib.Sexp.t
val num_devices : unit -> int
val get_device : ordinal:int -> device
val get_ctx_device : context -> device
val to_ordinal : device -> int
