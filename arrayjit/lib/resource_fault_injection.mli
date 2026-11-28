(** Test-only fault-injection points for resource-owning seams.

    Production code calls {!hit} at boundaries where an exception must leave resource ownership
    honest, passing the concrete {!artifact} identity alongside the {!point} (whose
    {!ownership_phase} is derived exhaustively by {!phase_of_point}). The default callback is a
    no-op and no configuration selects it. Tests install a callback with {!with_event_callback},
    {!with_callback}, or {!with_sticky_async_fault}.

    The callback is per DOMAIN, like the [Autotune.Seam] hooks: {!hit} calls the value installed in
    the calling domain, a scenario running in another domain neither reaches nor disturbs it, and a
    newly spawned domain starts with the no-op. That is sound because every injection point fires on
    the domain that called the operation (a [Context] transfer, compile or release, a schedule cache
    open, store or lookup) and none inside a task a scheduler enqueues, so the [multidev_cc] worker
    domains never reach one. A new point keeps that property: one placed in an enqueued task (a
    kernel or transfer task, as opposed to the routine-level schedule that only enqueues it) would
    fire on a worker domain and never reach the test's callback. [resource_fault_injection_domains]
    pins both halves. *)

type ownership_phase =
  | Before_allocate
  | After_allocate
  | Before_transfer
  | Before_await
  | Before_commit
  | Before_replay
  | Before_free
  | After_free
  | Before_launch
  | Capture_begin
  | Capture_launch
  | Capture_end
  | Capture_instantiate
  | Capture_replay
  | Teardown
[@@deriving sexp, equal]

type artifact =
  | Pool of { device_id : int; pool_id : int; constant : bool option }
  | Merge_slab of { device_id : int; size_in_bytes : int }
  | Routine of { name : string }
  | Routine_segment of { routine : string; index : int; count : int }
  | Context_handle of { device_id : int }
  | Cache_file of { path : string }
  | Graph_capture of { device_id : int; routine : string; key : string option }
  | Scratch_frame of { device_id : int; bytes : int }
[@@deriving sexp, equal]

type point =
  | Delta_pool_before_allocate
      (** Immediately before a delta pool's allocation, under the same refusal handler as the
          allocation itself. The census records the pool after the allocation succeeds. *)
  | Delta_pool_allocated
  | Delta_unwind_before_free
  | Link_after_delta
  | Link_batch_before_segment
  | Transfer_pool_allocated
  | From_host_before_copy
  | From_host_before_await
  | Init_from_device_before_await
  | Transfer_cleanup_before_await
  | To_host_before_copy
  | Merge_slab_before_allocate
  | Finalize_before_await
  | Finalize_before_free
  | Schedule_cache_before_lock
  | Schedule_cache_before_regime_commit
  | Schedule_cache_before_commit
  | Schedule_cache_before_replay
  | Autotune_before_winner_replay
  | Run_before_launch
  | Cuda_scratch_allocated
  | Cuda_scratch_freed
  | Graph_capture_begin
  | Graph_capture_launch
  | Graph_capture_end
  | Graph_capture_instantiate
  | Graph_capture_replay
  | Graph_exec_before_destroy
[@@deriving sexp, equal, enumerate]

type event = { point : point; phase : ownership_phase; artifact : artifact }
[@@deriving sexp, equal]

val phase_of_point : point -> ownership_phase
(** Exhaustive ownership-phase classification for every fault-injection point. *)

val is_async_sync_point : point -> bool
(** [true] for asynchronous synchronization/await boundaries where persistent worker or stream
    errors are sticky by default ([From_host_before_await], [Init_from_device_before_await],
    [Transfer_cleanup_before_await], [Finalize_before_await]). *)

exception Simulated_capture_error of string
(** Test-only exception injected at CUDA/HIP graph-capture points to exercise the same fallback path
    as a non-OOM driver capture error without linking vendor driver bindings into tests. *)

exception Simulated_graph_replay_oom of string
(** Test-only exception injected at {!Graph_capture_replay} to exercise the same exec-drop and plain
    fallback path as an out-of-memory driver launch refusal. *)

val hit : artifact:artifact -> point -> unit

val with_event_callback : (event -> unit) -> f:(unit -> 'a) -> 'a
(** Installs a domain-local callback receiving both the ownership phase and the artifact identity
    for [f], and restores the prior callback even when [f] raises. *)

val with_callback : (point -> unit) -> f:(unit -> 'a) -> 'a
(** Convenience wrapper over {!with_event_callback} that passes [event.point] to the callback. *)

type sticky_fault
(** Domain-local sticky asynchronous fault handle. Once latched by its trigger point, every
    subsequent {!is_async_sync_point} hit raises the same persistent exception until
    {!clear_sticky_fault} is called. *)

val with_sticky_async_fault : trigger:point -> ?exn:exn -> (sticky_fault -> 'a) -> 'a
(** Installs a domain-local sticky asynchronous fault callback for [f]. When [trigger] fires, the
    fault latches and raises [exn] on [trigger] and on every subsequent {!is_async_sync_point} hit
    until {!clear_sticky_fault} is called. Restores the prior callback even when [f] raises. *)

val sticky_fault_hits : sticky_fault -> event list
(** Every {!event} observed while [sticky_fault] was installed, in occurrence order. *)

val sticky_fault_triggered : sticky_fault -> bool
(** Whether [sticky_fault] is currently latched. *)

val clear_sticky_fault : sticky_fault -> unit
(** Clears a latched sticky fault so a subsequent retry can complete cleanly. *)

val pool_cap : default:int -> int
(** The working cap in the calling domain. *)

val with_pool_cap : int -> f:(unit -> 'a) -> 'a
(** Lowers the pool cap to at most the given byte count in the calling domain during [f]. The cap
    must be a positive power of two to preserve the link floor for page-rounded allocations.
    Otherwise raises [Invalid_argument] before entering [f]. An outer cap still applies to an inner
    scope. The prior cap is restored even when [f] raises. *)
