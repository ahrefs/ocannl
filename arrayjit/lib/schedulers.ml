open Base
module Debug_runtime = Utils.Debug_runtime
module Tn = Tnode
open Backend_intf
open Backend_impl

let _get_local_debug_runtime = Utils._get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module Multicore (Backend : For_add_scheduler) :
  With_scheduler with type buffer_ptr = Backend.buffer_ptr = struct
  include Backend
  module Domain = Domain [@warning "-3"]

  let global_config = ref Only_devices_parallel

  let initialize config =
    global_config := config;
    initialize config

  let is_initialized = is_initialized

  type task_list = Task.t Utils.mutable_list [@@deriving sexp_of]

  module Mut = Stdlib.Mutex
  module Queue = Saturn_lockfree.Single_prod_single_cons_queue

  type task_queue = Task.t Queue.t

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
      mutable is_ready : bool;
    }
    [@@deriving sexp_of]

    type domain = unit Domain.t

    let sexp_of_domain (d : domain) = Sexp.Atom ("domain-" ^ Int.to_string (Domain.get_id d :> int))

    type runner = { state : stream_state; domain : domain } [@@deriving sexp_of]
    type event = Not_implemented_yet [@@deriving sexp_of]

    let name = "multicore_" ^ Backend.name
  end

  module Device_types = Device_types (Device_config)
  include Device (Device_types) (Alloc_buffer_ignore_stream (Device_types) (Backend))
  open Device_config

  (** TODO: Blocks till the event completes, if it's not done already. *)
  let sync Not_implemented_yet = ()

  (** TODO: Whether the event completed. *)
  let is_done Not_implemented_yet = true

  (** TODO: Schedules waiting for the given event on the context's stream. *)
  let will_wait_for _ctx Not_implemented_yet = ()

  let get_used_memory _device = get_used_memory ()
  let is_dev_queue_empty state = Queue.size state.queue = 0
  let is_idle stream = is_dev_queue_empty stream.runner.state && stream.runner.state.is_ready

  let%track3_l_sexp await stream =
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

  (** TODO: Returns the event indicating if any currently running or scheduled computations on the
      stream have completed. *)
  let all_work _stream = Not_implemented_yet

  let%track3_l_sexp schedule_task stream task =
    assert (Domain.is_main_domain ());
    [%log_result "schedule_task", Task.describe task, get_name stream];
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

  let%track3_l_sexp spinup_stream ~stream_id : stream =
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
      }
    in
    let%track3_l_sexp worker (() : unit) : unit =
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
          | Some task -> Task.run task
        done
      with e ->
        state.stream_error <- Some e;
        state.keep_spinning <- false;
        [%log1 "stream", (stream_id : int), "exception", Exn.to_string e];
        (* TODO: we risk raising this error multiple times because await and schedule_task raise
           stream_error. But this is fine if we assume all exceptions are fatal. *)
        raise e
    in
    make_stream device { state; domain = Domain.spawn worker } ~stream_id

  module Dynarr = Stdlib.Dynarray

  let num_devices () = 1
  let suggested_num_streams _device = Domain.recommended_domain_count () - 1

  let cleanup_stream stream =
    assert (Domain.is_main_domain ());
    await stream;
    let r = stream.runner in
    r.state.keep_spinning <- false;
    Stdlib.Condition.broadcast r.state.dev_wait_for_work;
    Domain.join r.domain

  let get_device ~ordinal =
    if ordinal <> 0 then
      invalid_arg [%string "Multicore_scheduler.get_device %{ordinal#Int}: only device 0 exists"];
    device

  let latest_stream_id = ref (-1)

  let new_stream _device =
    assert (Domain.is_main_domain ());
    Int.incr latest_stream_id;
    let stream = spinup_stream ~stream_id:!latest_stream_id in
    Stdlib.Gc.finalise cleanup_stream stream;
    stream
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
  let latest_stram_id = ref (-1)

  let new_stream device =
    Int.incr latest_stram_id;
    make_stream device () ~stream_id:!latest_stram_id

  let all_work _stream = ()
  let is_idle _stream = true
  let await _stream = ()
  (* let global_run_no = ref 0 *)

  let initialize = Backend.initialize
  let is_initialized = Backend.is_initialized
  let schedule_task _stream task = Task.run task
end