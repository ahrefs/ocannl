type context [@@deriving sexp_of]
type code [@@deriving sexp_of]
type code_batch [@@deriving sexp_of]

val initialize : unit -> unit
val is_initialized : unit -> bool
val finalize : context -> unit
val sexp_of_context : context -> Sexplib.Sexp.t
val compile : ?name:string -> Indexing.unit_bindings -> Low_level.optimized -> code

val compile_batch :
  names:string option array -> Indexing.unit_bindings -> Low_level.optimized option array -> code_batch

val link : context -> code -> context * Indexing.lowered_bindings * (unit -> Tnode.work)

val link_batch :
  context -> code_batch -> context * Indexing.lowered_bindings * (unit -> Tnode.work) option array

val unsafe_cleanup : ?unsafe_shutdown:bool -> unit -> unit

val from_host : ?rt:(module Minidebug_runtime.Debug_runtime) -> context -> Tnode.t -> unit
(** If the array is both hosted and in-context, copies from host to context. *)

val to_host : ?rt:(module Minidebug_runtime.Debug_runtime) -> context -> Tnode.t -> unit
(** If the array is both hosted and in-context, copies from context to host. *)

val device_to_device :
  ?rt:(module Minidebug_runtime.Debug_runtime) ->
  Tnode.t ->
  into_merge_buffer:bool ->
  dst:context ->
  src:context ->
  unit
(** If the array is in both contexts, copies from [dst] to [src]. *)

val physical_merge_buffers : bool

type device

val init : device -> context
val await : device -> unit
val acknowledge : device -> unit
val is_idle : device -> bool
val is_booked : device -> bool
val sexp_of_device : device -> Sexplib.Sexp.t
val num_devices : unit -> int
val get_device : ordinal:int -> device
val get_ctx_device : context -> device
val to_ordinal : device -> int
