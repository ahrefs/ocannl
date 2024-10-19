(** {1 The collection of the execution backends} *)

open Base

val sync_suggested_num_streams : int ref

module Cc_backend : Backend_types.Backend
module Sync_cc_backend : Backend_types.Backend
module Gccjit_backend : Backend_types.Backend
module Sync_gccjit_backend : Backend_types.Backend
module Cuda_backend : Backend_types.Backend

val reinitialize : (module Backend_types.Backend) -> Backend_types.Types.config -> unit
(** Initializes the backend, and if it was already initialized, performs garbage collection. *)

val fresh_backend :
  ?backend_name:string ->
  ?config:Backend_types.Types.config ->
  unit ->
  (module Backend_types.Backend)
(** Reinitializes and returns a backend corresponding to [backend_name], or if omitted, selected via
    the global [backend] setting. See {!reinitialize}. *)
