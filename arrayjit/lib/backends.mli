(** {1 The collection of the execution backends} *)

open Base

val sync_suggested_num_streams : int ref

val reinitialize : (module Backend_types.Backend) -> Backend_types.config -> unit
(** Initializes the backend, and if it was already initialized, performs garbage collection. *)

val fresh_backend :
  ?backend_name:string ->
  ?config:Backend_types.config ->
  unit ->
  (module Backend_types.Backend)
(** Reinitializes and returns a backend corresponding to [backend_name], or if omitted, selected via
    the global [backend] setting. See {!reinitialize}. *)
