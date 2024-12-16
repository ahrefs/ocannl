(** {1 A collection of the execution backends} *)

open Base

val finalize :
  'buffer_ptr 'dev 'runner 'event.
  (module Backend_intf.Backend
     with type buffer_ptr = 'buffer_ptr
      and type dev = 'dev
      and type event = 'event
      and type runner = 'runner) ->
  ('buffer_ptr, ('buffer_ptr, 'dev, 'runner, 'event) Backend_intf.stream) Backend_intf.context ->
  unit
(** Frees the arrays that are specific to the context -- not contained in the parent context. Note:
    use [finalize] to optimize memory, it is not obligatory because all arrays are freed when their
    [buffer_ptr]s are garbage-collected.

    Note: this type will get simpler with modular explicits. *)

val fresh_backend :
  ?backend_name:string -> unit -> (module Backend_intf.Backend)
(** Creates a new backend corresponding to [backend_name], or if omitted, selected via the global
    [backend] setting. *)
