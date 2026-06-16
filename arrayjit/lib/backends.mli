(** {1 A collection of the execution backends} *)

open Base
module Schedulers = Schedulers

val finalize :
  'dev 'runner 'event 'optimize_ctx.
  (module Ir.Backend_intf.Backend
     with type dev = 'dev
      and type event = 'event
      and type runner = 'runner
      and type optimize_ctx = 'optimize_ctx) ->
  ('dev, 'runner, 'event, 'optimize_ctx) Ir.Backend_intf.context -> unit
(** Frees the pools that are specific to the context -- not contained in the parent context. Note:
    use [finalize] to optimize memory, it is not obligatory because all pools are freed when their
    backend buffers are garbage-collected. *)

val fresh_backend : ?backend_name:string -> unit -> (module Ir.Backend_intf.Backend)
(** Creates a new backend corresponding to [backend_name], or if omitted, selected via the global
    [backend] setting. It should be safe to reinitialize the tensor system before [fresh_backend].
*)
