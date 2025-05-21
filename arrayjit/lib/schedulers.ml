open Base
module Tn = Ir.Tnode
open Ir.Backend_intf
open Ir.Backend_impl

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module Multicore (Backend : For_add_scheduler) :
  With_scheduler with type buffer_ptr = Backend.buffer_ptr = struct
  include Backend
  module Domain = Domain [@warning "-3"]
  (* Currently, Backend.config is not used. *)

  type task_list = Ir.Task.t Utils.mutable_list [@@deriving sexp_of]

  module Mut = Stdlib.Mutex
  module Queue = Saturn_lockfree.Single_prod_single_cons_queue

  type task_queue = Ir.Task.t Queue.t

  let sexp_of_task_queue q =
    Sexp.(List [ Atom "task_queue_of_size"; Atom (Int.to_string @@ Queue.size q) ])

  module Device_config = struct
    include (
      Backend : Buffer with type buffer_ptr = Backend.buffer_ptr and type buffer = Backend.buffer)

    type dev = CPU [@@deriving sexp_of]

    type stream_state = {
      mutable keep_spinning : bool;
      mutable stream_error : exn option;
      queue : task_queue;
      mut : (Mut.t[@sexp.opaque]);
      host_wait_for_idle : (Stdlib.Condition.t[@sexp.opaque]);
      dev_wait_for_work : (Stdlib.Condition.t[@sexp.opaque]);
      clock_mut : (Mut.t[@sexp.opaque]);
      clock_tick : (Stdlib.Condition.t[@sexp.opaque]);
      mutable is_ready : bool;
      stream_id : int;
    }
    [@@deriving sexp_of]

    type domain = unit Domain.t

    let sexp_of_domain (d : domain) = Sexp.Atom ("domain-" ^ Int.to_string (Domain.get_id d :> int))

    type runner = { state : stream_state; domain : domain } [@@deriving sexp_of]
    type event = { stream_state : stream_state; is_done : Utils.atomic_bool } [@@deriving sexp_of]

    let name = "multicore_" ^ Backend.name
  end

  module Device_types = Device_types (Device_config)
  include Device (Device_types) (Alloc_buffer_ignore_stream (Device_types) (Backend))
  open Device_config

  let sync { stream_state; is_done } =
    while stream_state.keep_spinning && not (Atomic.get is_done) do
      Mut.lock stream_state.clock_mut;
      if stream_state.keep_spinning && not (Atomic.get is_done) then
        Condition.wait stream_state.clock_tick stream_state.clock_mut;
      Mut.unlock stream_state.clock_mut
    done

  let is_done { stream_state = _; is_done } = Atomic.get is_done
  let get_used_memory _device = get_used_memory ()
  let is_dev_queue_empty state = Queue.size state.queue = 0
  let is_idle stream = is_dev_queue_empty stream.runner.state && stream.runner.state.is_ready

  let%track3_sexp await stream =
    assert (Domain.is_main_domain ());
    let d = stream.runner.state in
    if (not @@ is_idle stream) && d.keep_spinning then (
      Mut.lock d.mut;
      while (not @@ is_idle stream) && d.keep_spinning do
        (* If the stream "is ready", it needs to be woken up first to finish the work. *)
        if d.is_ready then Stdlib.Condition.broadcast d.dev_wait_for_work;
        Stdlib.Condition.wait d.host_wait_for_idle d.mut
      done;
      Mut.unlock d.mut;
      Option.iter d.stream_error ~f:(fun e -> Exn.reraise e @@ get_name stream))

  let%track3_sexp schedule_task stream task =
    assert (Domain.is_main_domain ());
    [%log_result "schedule_task", Ir.Task.describe task, get_name stream];
    let d = stream.runner.state in
    Option.iter d.stream_error ~f:(fun e -> Exn.reraise e @@ get_name stream);
    if not d.keep_spinning then invalid_arg "Multicore_scheduler: stream not available";
    if not @@ Queue.try_push d.queue task then (
      await stream;
      Queue.push_exn d.queue task);
    if d.is_ready then (
      Mut.lock d.mut;
      Stdlib.Condition.broadcast d.dev_wait_for_work;
      Mut.unlock d.mut)

  let global_run_no = ref 0
  let device : device = make_device CPU ~ordinal:0

  let will_wait_for ctx ({ stream_state; is_done = _ } as event) =
    let work () = sync event in
    let task =
      Ir.Task.Task
        {
          context_lifetime = ();
          description =
            [%string "wait on %{get_name ctx.stream} for 0:%{stream_state.stream_id#Int}"];
          work;
        }
    in
    schedule_task ctx.stream task

  let all_work stream =
    assert (Domain.is_main_domain ());
    let stream_state = stream.runner.state in
    let is_done = Atomic.make false in
    let work () =
      assert (Atomic.compare_and_set is_done false true);
      Mut.lock stream_state.clock_mut;
      Stdlib.Condition.broadcast stream_state.clock_tick;
      Mut.unlock stream_state.clock_mut
    in
    schedule_task stream @@ Task { context_lifetime = (); description = "clock tick"; work };
    { stream_state; is_done }

  let%track3_sexp spinup_stream () : stream =
    let create stream_id =
      Int.incr global_run_no;
      let state =
        {
          keep_spinning = true;
          stream_error = None;
          queue = Queue.create ~size_exponent:12;
          mut = Mut.create ();
          is_ready = false;
          host_wait_for_idle = Stdlib.Condition.create ();
          dev_wait_for_work = Stdlib.Condition.create ();
          clock_mut = Mut.create ();
          clock_tick = Stdlib.Condition.create ();
          stream_id;
        }
      in
      let%track3_sexp worker (() : unit) : unit =
        assert (not @@ Domain.is_main_domain ());
        try
          while state.keep_spinning do
            match Queue.pop_opt state.queue with
            | None ->
                Mut.lock state.mut;
                state.is_ready <- true;
                Stdlib.Condition.broadcast state.host_wait_for_idle;
                while is_dev_queue_empty state && state.keep_spinning do
                  Stdlib.Condition.wait state.dev_wait_for_work state.mut
                done;
                state.is_ready <- false;
                Mut.unlock state.mut
            | Some task -> Ir.Task.run task
          done
        with e ->
          state.stream_error <- Some e;
          state.keep_spinning <- false;
          [%log1 "stream", (stream_id : int), "exception", Exn.to_string e];
          (* TODO: we risk raising this error multiple times because await and schedule_task raise
             stream_error. But this is fine if we assume all exceptions are fatal. *)
          raise e
      in
      { state; domain = Domain.spawn worker }
    in
    (* We cannot use make_stream because runner needs stream_id. *)
    Utils.register_new device.streams ~grow_by:8 (fun stream_id ->
        let runner = create stream_id in
        {
          device;
          runner;
          merge_buffer = ref None;
          stream_id;
          allocated_buffer = None;
          updating_for = Hashtbl.create (module Ir.Tnode);
          updating_for_merge_buffer = None;
          reader_streams = Hashtbl.create (module Ir.Tnode);
        })

  let num_devices () = 1
  let suggested_num_streams _device = Domain.recommended_domain_count () - 2

  let static_properties =
    Sexp.List [
      Sexp.Atom "multicore_devices";
      Sexp.List [
        Sexp.Atom "device";
        Sexp.List [
          Sexp.List [Sexp.Atom "device_name"; Sexp.Atom "CPU"];
          Sexp.List [Sexp.Atom "device_ordinal"; [%sexp_of: int] 0];
          Sexp.List [Sexp.Atom "num_domains"; [%sexp_of: int] (Domain.recommended_domain_count ())];
        ]
      ]
    ]

  let%track7_sexp cleanup_stream (stream : stream) : unit =
    (* Allow running in parallel. *)
    (* assert (Domain.is_main_domain ()); *)
    [%log "cleanup_stream: await stream"];
    await stream;
    let r : runner = stream.runner in
    r.state.keep_spinning <- false;
    [%log "cleanup_stream: broadcasting r.state.dev_wait_for_work to wake up the worker"];
    Stdlib.Condition.broadcast r.state.dev_wait_for_work;
    [%log "cleanup_stream: joining the domain"];
    Domain.join r.domain

  let get_device ~ordinal =
    if ordinal <> 0 then
      invalid_arg [%string "Multicore_scheduler.get_device %{ordinal#Int}: only device 0 exists"];
    device

  let%track5_sexp new_stream _device =
    assert (Domain.is_main_domain ());
    let stream = spinup_stream () in
    Stdlib.Gc.finalise cleanup_stream stream;
    stream

  let get_global_debug_info () = Sexp.message "global_debug" []
  let get_debug_info (stream : stream) = sexp_of_runner stream.runner
end

(** For debugging, allow [Sync_scheduler(...).suggested_num_streams] calls to return >1 numbers. *)
let sync_suggested_num_streams = ref 1

(** A minimalisitc wrapper creating backends where all calls run synchronously on the main thread.
    There is only one device, but an arbitrary number of streams. *)
module Sync (Backend : For_add_scheduler) = struct
  include Backend

  module Device_config = struct
    include (
      Backend : Buffer with type buffer_ptr = Backend.buffer_ptr and type buffer = Backend.buffer)

    type dev = CPU [@@deriving sexp_of]
    type runner = unit [@@deriving sexp_of]
    type event = unit [@@deriving sexp_of]

    let name = "sync_" ^ Backend.name
  end

  module Device_types = Device_types (Device_config)
  include Device (Device_types) (Alloc_buffer_ignore_stream (Device_types) (Backend))
  open Device_config

  let sync () = ()
  let is_done () = true
  let will_wait_for _context () = ()

  let alloc_buffer ?old_buffer ~size_in_bytes _stream =
    Backend.alloc_buffer ?old_buffer ~size_in_bytes ()

  let device : device = make_device CPU ~ordinal:0

  let get_device ~ordinal =
    if ordinal <> 0 then
      invalid_arg @@ "Sync_scheduler.get_device: there is only one device, but ordinal="
      ^ Int.to_string ordinal;
    device

  let num_devices () = 1
  let suggested_num_streams _ = !sync_suggested_num_streams
  let get_used_memory _ = Backend.get_used_memory ()
  let new_stream device = make_stream device ()
  let all_work _stream = ()
  let is_idle _stream = true
  let await _stream = ()

  let static_properties =
    Sexp.List [
      Sexp.Atom "sync_devices";
      Sexp.List [
        Sexp.Atom "device";
        Sexp.List [
          Sexp.List [Sexp.Atom "device_name"; Sexp.Atom "CPU"];
          Sexp.List [Sexp.Atom "device_ordinal"; Sexp.Atom "0"];
          Sexp.List [Sexp.Atom "threads"; Sexp.Atom "1"];
        ]
      ]
    ]

  (* let global_run_no = ref 0 *)
  let schedule_task _stream task = Ir.Task.run task
  let get_global_debug_info () = Sexp.message "global_debug" []
  let get_debug_info (stream : stream) = sexp_of_runner stream.runner
end
