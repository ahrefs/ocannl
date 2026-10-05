open Base

type point =
  | Delta_pool_allocated
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

let callback : (point -> unit) ref = ref (fun _ -> ())
let hit point = !callback point

let with_callback cb ~f =
  let prior = !callback in
  callback := cb;
  Exn.protect ~f ~finally:(fun () -> callback := prior)
