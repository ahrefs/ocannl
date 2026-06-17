(** {1 A collection of the execution backends} *)

open Base
module Schedulers = Schedulers

val plan_pool_segments :
  cap:int ->
  what:string ->
  debug_name:(int -> string) ->
  (int * int) list ->
  (int * int) list * int list
(** gh-ocannl-344 pool-allocator planner. Lays out [(size, alignment)] allocations (in order) into
    pools so no pool's bumped extent exceeds [cap] bytes (the uint32 4 GB per-pool ceiling when
    [large_models = false]). Returns each item's [(segment_index, byte_offset)] and the byte size of
    each segment. Raises {!Ir.Utils.User_error} (naming [what] and [debug_name i]) when a single item
    exceeds [cap]. Exposed for unit testing the segmenting/cap behavior with synthetic sizes. *)

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
