open Base

type point =
  | Delta_pool_allocated
  | Delta_unwind_before_free
  | Link_after_delta
  | Transfer_pool_allocated
  | From_host_before_copy
  | From_host_before_await
  | Init_from_device_before_await
  | Transfer_cleanup_before_await
  | To_host_before_copy
  | Finalize_before_await
  | Finalize_before_free
  | Schedule_cache_before_lock
  | Schedule_cache_before_regime_commit
  | Schedule_cache_before_commit
  | Schedule_cache_before_replay
[@@deriving sexp, equal]

(* One callback per domain, like [Autotune.Seam] (which this module cannot reuse: [Schedule_cache]
   calls {!hit}, and [Autotune] depends on [Schedule_cache]). Sound because every {!hit} runs on the
   domain that called the operation, none inside a task a scheduler enqueues, so on [multidev_cc]
   the worker domains never reach one. *)
let callback : (point -> unit) Stdlib.Domain.DLS.key = Stdlib.Domain.DLS.new_key (fun () _ -> ())
let hit point = Stdlib.Domain.DLS.get callback point

let with_callback cb ~f =
  let prior = Stdlib.Domain.DLS.get callback in
  Stdlib.Domain.DLS.set callback cb;
  Exn.protect ~f ~finally:(fun () -> Stdlib.Domain.DLS.set callback prior)
