(** Test-only fault-injection points for resource-owning seams.

    Production code calls {!hit} at boundaries where an exception must leave resource ownership
    honest. The default callback is a no-op and no configuration selects it. Tests install a
    callback with {!with_callback}; callbacks are process-global, so scenarios must not run in
    parallel. *)

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

val hit : point -> unit

val with_callback : (point -> unit) -> f:(unit -> 'a) -> 'a
(** Installs the callback for [f] and restores the prior callback even when [f] raises. *)
