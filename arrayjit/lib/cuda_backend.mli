type context [@@deriving sexp_of]
type code [@@deriving sexp_of]
type code_batch [@@deriving sexp_of]
open Backend_utils.Types

val initialize : config -> unit
val is_initialized : unit -> bool
val finalize : context -> unit
val sexp_of_context : context -> Sexplib.Sexp.t
val compile : ?name:string -> Indexing.unit_bindings -> Low_level.optimized -> code

val compile_batch :
  names:string option array ->
  Indexing.unit_bindings ->
  Low_level.optimized option array ->
  code_batch

val link : context -> code -> context * Indexing.lowered_bindings * Tnode.task

val link_batch :
  context -> code_batch -> context * Indexing.lowered_bindings * Tnode.task option array

val unsafe_cleanup : ?unsafe_shutdown:bool -> unit -> unit

val from_host : ?rt:(module Minidebug_runtime.Debug_runtime) -> context -> Tnode.t -> bool
(** If the array is both hosted and in-context, copies from host to context. *)

val to_host : ?rt:(module Minidebug_runtime.Debug_runtime) -> context -> Tnode.t -> bool
(** If the array is both hosted and in-context, copies from context to host. *)

val device_to_device :
  ?rt:(module Minidebug_runtime.Debug_runtime) ->
  Tnode.t ->
  into_merge_buffer:merge_buffer_use ->
  dst:context ->
  src:context ->
  bool
(** If the array is in both contexts, copies from [dst] to [src]. *)

type buffer_ptr [@@deriving sexp_of]

val to_buffer :
  ?rt:(module Minidebug_runtime.Debug_runtime) -> Tnode.t -> dst:buffer_ptr -> src:context -> unit

val host_to_buffer :
  ?rt:(module Minidebug_runtime.Debug_runtime) -> Ndarray.t -> dst:buffer_ptr -> unit

val buffer_to_host :
  ?rt:(module Minidebug_runtime.Debug_runtime) -> Ndarray.t -> src:buffer_ptr -> unit

val get_buffer : Tnode.t -> context -> buffer_ptr option

type physical_device
type device

val alloc_buffer : ?old_buffer:buffer_ptr * int -> size_in_bytes:int -> unit -> buffer_ptr
val init : device -> context
val await : device -> unit
val is_idle : device -> bool
val sexp_of_device : device -> Sexplib.Sexp.t
val num_physical_devices : unit -> int
val suggested_num_virtual_devices : physical_device -> int
val get_device : ordinal:int -> physical_device
val get_physical_device : device -> physical_device
val new_virtual_device : physical_device -> device
val get_ctx_device : context -> device
val get_name : device -> string
val to_ordinal : physical_device -> int
val to_subordinal : device -> int
