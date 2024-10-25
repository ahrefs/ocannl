(** {1 The collection of the execution backends} *)

open Base

val sync_suggested_num_streams : int ref

val reinitialize : (module Backend_intf.Backend) -> Backend_intf.config -> unit
(** Initializes the backend, and if it was already initialized, performs garbage collection. *)

val fresh_backend :
  ?backend_name:string -> ?config:Backend_intf.config -> unit -> (module Backend_intf.Backend)
(** Reinitializes and returns a backend corresponding to [backend_name], or if omitted, selected via
    the global [backend] setting. See {!reinitialize}. *)
