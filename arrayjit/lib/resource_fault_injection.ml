open Base

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

let phase_of_point = function
  | Delta_pool_before_allocate | Merge_slab_before_allocate -> Before_allocate
  | Delta_pool_allocated | Transfer_pool_allocated | Cuda_scratch_allocated -> After_allocate
  | From_host_before_copy | To_host_before_copy -> Before_transfer
  | From_host_before_await | Init_from_device_before_await | Transfer_cleanup_before_await
  | Finalize_before_await ->
      Before_await
  | Schedule_cache_before_lock | Schedule_cache_before_regime_commit | Schedule_cache_before_commit
  | Link_after_delta | Link_batch_before_segment ->
      Before_commit
  | Schedule_cache_before_replay | Autotune_before_winner_replay -> Before_replay
  | Delta_unwind_before_free | Finalize_before_free -> Before_free
  | Cuda_scratch_freed -> After_free
  | Run_before_launch -> Before_launch
  | Graph_capture_begin -> Capture_begin
  | Graph_capture_launch -> Capture_launch
  | Graph_capture_end -> Capture_end
  | Graph_capture_instantiate -> Capture_instantiate
  | Graph_capture_replay -> Capture_replay
  | Graph_exec_before_destroy -> Teardown

let is_async_sync_point = function
  | From_host_before_await | Init_from_device_before_await | Transfer_cleanup_before_await
  | Finalize_before_await ->
      true
  | Delta_pool_before_allocate | Delta_pool_allocated | Delta_unwind_before_free | Link_after_delta
  | Link_batch_before_segment | Transfer_pool_allocated | From_host_before_copy
  | To_host_before_copy | Merge_slab_before_allocate | Finalize_before_free
  | Schedule_cache_before_lock | Schedule_cache_before_regime_commit | Schedule_cache_before_commit
  | Schedule_cache_before_replay | Autotune_before_winner_replay | Run_before_launch
  | Cuda_scratch_allocated | Cuda_scratch_freed | Graph_capture_begin | Graph_capture_launch
  | Graph_capture_end | Graph_capture_instantiate | Graph_capture_replay | Graph_exec_before_destroy
    ->
      false

exception Simulated_capture_error of string
exception Simulated_graph_replay_oom of string

(* One callback per domain, like [Autotune.Seam] (which this module cannot reuse: [Schedule_cache]
   calls {!hit}, and [Autotune] depends on [Schedule_cache]). Sound because every {!hit} runs on the
   domain that called the operation, none inside a task a scheduler enqueues, so on [multidev_cc]
   the worker domains never reach one. *)
let callback : (event -> unit) Stdlib.Domain.DLS.key = Stdlib.Domain.DLS.new_key (fun () _ -> ())

let hit ~artifact point =
  Stdlib.Domain.DLS.get callback { point; phase = phase_of_point point; artifact }

let with_event_callback cb ~f =
  let prior = Stdlib.Domain.DLS.get callback in
  Stdlib.Domain.DLS.set callback cb;
  Exn.protect ~f ~finally:(fun () -> Stdlib.Domain.DLS.set callback prior)

let with_callback cb ~f = with_event_callback (fun ev -> cb ev.point) ~f

type sticky_fault = { mutable armed : bool; mutable triggered : bool; mutable hits : event list }

let sticky_fault_hits fault = List.rev fault.hits
let sticky_fault_triggered fault = fault.triggered

let clear_sticky_fault fault =
  fault.armed <- false;
  fault.triggered <- false

let with_sticky_async_fault ~trigger ?(exn = Failure "injected sticky async failure") f =
  let fault = { armed = true; triggered = false; hits = [] } in
  with_event_callback
    (fun ev ->
      fault.hits <- ev :: fault.hits;
      if fault.armed && equal_point ev.point trigger then (
        fault.triggered <- true;
        raise exn)
      else if fault.triggered && is_async_sync_point ev.point then raise exn)
    ~f:(fun () -> f fault)

let pool_cap_override = Stdlib.Domain.DLS.new_key (fun () -> Int.max_value)
let pool_cap ~default = Int.min default (Stdlib.Domain.DLS.get pool_cap_override)

let with_pool_cap cap ~f =
  if cap <= 0 || not (Int.is_pow2 cap) then
    invalid_arg "Resource_fault_injection.with_pool_cap: expected a positive power of two";
  let prior = Stdlib.Domain.DLS.get pool_cap_override in
  Stdlib.Domain.DLS.set pool_cap_override (Int.min prior cap);
  Exn.protect ~f ~finally:(fun () -> Stdlib.Domain.DLS.set pool_cap_override prior)
