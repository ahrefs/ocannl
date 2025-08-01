(** {1 A collection of the execution backends} *)

open Base
module Schedulers = Schedulers

val finalize :
  'buffer_ptr 'dev 'runner 'event 'optimize_ctx.
  (module Ir.Backend_intf.Backend
     with type buffer_ptr = 'buffer_ptr
      and type dev = 'dev
      and type event = 'event
      and type runner = 'runner
      and type optimize_ctx = 'optimize_ctx) ->
  ( 'buffer_ptr,
    ('buffer_ptr, 'dev, 'runner, 'event) Ir.Backend_intf.stream,
    'optimize_ctx )
  Ir.Backend_intf.context ->
  unit
(** Frees the arrays that are specific to the context -- not contained in the parent context. Note:
    use [finalize] to optimize memory, it is not obligatory because all arrays are freed when their
    [buffer_ptr]s are garbage-collected.

    Note: this type will get simpler with modular explicits. *)

val fresh_backend :
  ?backend_name:string -> ?config:Ir.Backend_intf.config -> unit -> (module Ir.Backend_intf.Backend)
(** Creates a new backend corresponding to [backend_name], or if omitted, selected via the global
    [backend] setting. It should be safe to reinitialize the tensor system before [fresh_backend].
*)
