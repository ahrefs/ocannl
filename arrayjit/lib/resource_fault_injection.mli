(** Test-only fault-injection points for resource-owning seams.

    Production code calls {!hit} at boundaries where an exception must leave resource ownership
    honest. The default callback is a no-op and no configuration selects it. Tests install a
    callback with {!with_callback}.

    The callback is per DOMAIN, like the [Autotune.Seam] hooks: {!hit} calls the value installed in
    the calling domain, a scenario running in another domain neither reaches nor disturbs it, and a
    newly spawned domain starts with the no-op. That is sound because every injection point fires on
    the domain that called the operation (a [Context] transfer, compile or release, a schedule cache
    open, store or lookup) and none inside a task a scheduler enqueues, so the [multidev_cc] worker
    domains never reach one. A new point keeps that property: one placed in an enqueued task (a
    kernel or transfer task, as opposed to the routine-level schedule that only enqueues it) would
    fire on a worker domain and never reach the test's callback. [resource_fault_injection_domains]
    pins both halves. *)

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

val hit : point -> unit

val with_callback : (point -> unit) -> f:(unit -> 'a) -> 'a
(** Installs the callback for [f] and restores the prior callback even when [f] raises. *)
