open Base
module Debug_runtime = Utils.Debug_runtime

[%%global_debug_log_level Nothing]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

type 'context routine = {
  context : 'context;
  schedule : Tnode.task;
  bindings : Indexing.lowered_bindings;
  name : string;
}
[@@deriving sexp_of]

type config = [ `Physical_devices_only | `For_parallel_copying | `Most_parallel_devices ]
[@@deriving equal, sexp, variants]

module type No_device_backend = sig
  type code [@@deriving sexp_of]
  type code_batch [@@deriving sexp_of]
  type context [@@deriving sexp_of]
  type nonrec routine = context routine [@@deriving sexp_of]

  val name : string
  val initialize : config -> unit
  val is_initialized : unit -> bool

  val init : label:string -> context
  (** [label] is usually the backend name concatenated with the device number. *)

  val finalize : context -> unit
  (** Finalizes (just) the context. *)

  val compile : ?shared:bool -> ?name:string -> Indexing.unit_bindings -> Assignments.t -> code
  (** If [~shared:true] (default [false]), the backend should prefer to do more compile work in a
      device-agnostic way. If [~shared:false], the backend can opt to postpone compiling altogether until
      [link] is called, to benefit from more optimizations. *)

  val compile_batch :
    ?shared:bool ->
    ?names:string array ->
    ?occupancy:(name:string -> src_n:int -> bool) ->
    Indexing.unit_bindings ->
    Assignments.t array ->
    code_batch
  (** Unlike the [~shared] parameter, [compile_batch] vs. [compile] is mostly about improving the compile time
      and debugging convenience by generating fewer files -- ideally does not affect execution, but there can
      be backend-specific differences. Only array entries for which [occupancy] returns true are included. *)

  val link : context -> code -> routine
  (** Returns the routine for the code's procedure, in a new context derived from the given context. *)

  val link_batch : context -> code_batch -> routine option array
  (** Returns the routines for the procedures included in the code batch. All returned routines share the same
      new context. *)

  val unsafe_cleanup : ?unsafe_shutdown:bool -> unit -> unit
  (** Cleans up all work on a backend. If [~unsafe_shutdown:true], releases resources, potentially making the
      backend unusable. *)

  val from_host : ?rt:(module Minidebug_runtime.Debug_runtime) -> context -> Tnode.t -> unit
  (** If the array is both hosted and in-context, schedules a copy from host to context, otherwise does
      nothing. NOTE: when run for a device, it's the caller's responsibility to synchronize the device before
      the host's data is overwritten. *)

  val to_host : ?rt:(module Minidebug_runtime.Debug_runtime) -> context -> Tnode.t -> unit
  (** If the array is both hosted and in-context, schedules a copy from context to host, otherwise does
      nothing. NOTE: when run for a device, it's the caller's responsibility to synchronize the device before
      the host's data is read. *)

  val device_to_device :
    ?rt:(module Minidebug_runtime.Debug_runtime) ->
    Tnode.t ->
    into_merge_buffer:bool ->
    dst:context ->
    src:context ->
    unit
  (** If the array is present in the [src] context, and in the [dst] context in case
      [~into_merge_buffer:false], schedules a copy of the tensor node from the device of [src] to the device
      of [dst]; otherwise, if [~into_merge_buffer:true], unsets the merge buffer source on [dst]. If
      [~into_merge_buffer:false], copies into the given node on [dst]; if [~into_merge_buffer:true], sets on
      [dst] the merge buffer source to the given node and handles the transfer using it. NOTE: when run for a
      device, it's the caller's responsibility to synchronize the [src] device, if needed, {i before} calling
      [device_to_device], and the [dst] device {i afterward}, according to {!physical_merge_buffers}, before
      any computations on the [src] device overwrite the node. *)

  val physical_merge_buffers : bool
  (** If [physical_merge_buffers = true], the [src] node data is not needed after the task scheduled by
      [device_to_device] finishes. Otherwise, the [src] node data may be needed till after the tasks computing
      with the merge buffer finish. *)
end

module type Backend = sig
  include No_device_backend

  type physical_device
  type device

  val init : device -> context

  val await : device -> unit
  (** Blocks till the device becomes idle, i.e. synchronizes the device. *)

  val is_idle : device -> bool
  (** Whether the device is currently waiting for work. *)

  val sexp_of_device : device -> Sexp.t
  val get_device : ordinal:int -> physical_device
  val num_physical_devices : unit -> int

  val suggested_num_virtual_devices : physical_device -> int
  (** The optimal number of virtual devices for the given physical device to follow the
      {!No_device_backend.config} strategy passed to {!No_device_backend.initialize}. *)

  val new_virtual_device : physical_device -> device
  val get_ctx_device : context -> device
  val get_physical_device : device -> physical_device
  val to_ordinal : physical_device -> int
  val to_subordinal : device -> int
  val get_name : device -> string
end

let forget_printbox (module Runtime : Minidebug_runtime.PrintBox_runtime) =
  (module Runtime : Minidebug_runtime.Debug_runtime)

module Multicore_backend (Backend : No_device_backend) : Backend = struct
  module Domain = Domain [@warning "-3"]

  type task_list = Tnode.task Utils.mutable_list [@@deriving sexp_of]

  type device_state = {
    mutable keep_spinning : bool;
    mutable dev_spinning : bool;
    mutable host_pos : task_list;
    mutable dev_pos : task_list;
    mutable dev_previous_pos : task_list;
    dev_wait : (Utils.waiter[@sexp.opaque]);
  }
  [@@deriving sexp_of]

  type device = {
    state : device_state;
    host_wait_for_idle : (Utils.waiter[@sexp.opaque]);
    ordinal : int;
    domain : (unit Domain.t[@sexp.opaque]);
  }
  [@@deriving sexp_of]

  type physical_device = device [@@deriving sexp_of]

  let is_dev_queue_empty state = Utils.(is_empty @@ tl_exn state.dev_previous_pos)
  let is_idle device = is_dev_queue_empty device.state && device.state.dev_wait.is_waiting ()

  let await device =
    assert (Domain.is_main_domain ());
    let d = device.state in
    let keep_waiting () =
      if d.keep_spinning && d.dev_spinning && not (is_dev_queue_empty d) then (
        ignore (d.dev_wait.release_if_waiting () : bool);
        true)
      else d.keep_spinning && d.dev_spinning && not (d.dev_wait.is_waiting ())
    in
    while not (is_dev_queue_empty d) do
      ignore (device.host_wait_for_idle.await ~keep_waiting () : bool)
    done

  let%track_sexp schedule_task device task =
    assert (Domain.is_main_domain ());
    let d = device.state in
    if not d.keep_spinning then invalid_arg "Multicore_backend: device not available";
    d.host_pos <- Utils.insert ~next:task d.host_pos;
    ignore (d.dev_wait.release_if_waiting () : bool)

  let global_run_no = ref 0

  let spinup_device ~(ordinal : int) : device =
    Int.incr global_run_no;
    let init_pos =
      Utils.Cons { hd = Tnode.{ description = "root of task queue"; work = (fun _rt () -> ()) }; tl = Empty }
    in
    let state =
      {
        keep_spinning = true;
        host_pos = init_pos;
        dev_pos = Empty;
        dev_previous_pos = init_pos;
        dev_spinning = false;
        dev_wait = Utils.waiter ~name:"dev" ();
      }
    in
    let host_wait_for_idle = Utils.waiter ~name:"host" () in
    let keep_waiting () =
      state.keep_spinning && state.dev_spinning && is_dev_queue_empty state
      && not (host_wait_for_idle.is_waiting ())
    in
    let wait_for_dev = state.dev_wait.await ~keep_waiting in
    let run_no = !global_run_no in
    let debug_runtime =
      Utils.get_debug ("dev-multicore-" ^ Int.to_string ordinal ^ "-run-" ^ Int.to_string run_no)
    in
    let%diagn_rt_sexp worker (() : unit) : unit =
      state.dev_spinning <- true;
      try
        while state.keep_spinning do
          match state.dev_pos with
          | Empty ->
              let _host_released : bool = host_wait_for_idle.release_if_waiting () in
              let _could_wait : bool = wait_for_dev () in
              (* not _host_released && not _could_wait: we busy-loop until host processes its release. *)
              (* [%log "WORK WHILE LOOP: EMPTY AFTER WAIT -- dev pos:", (state.dev_pos : task_list)]; *)
              state.dev_pos <- Utils.tl_exn state.dev_previous_pos
          | Cons { hd; tl } ->
              Tnode.run _debug_runtime hd;
              (* [%log "WORK WHILE LOOP: AFTER WORK"]; *)
              state.dev_previous_pos <- state.dev_pos;
              state.dev_pos <- tl
        done;
        state.dev_spinning <- false
      with e ->
        state.dev_spinning <- false;
        ignore (host_wait_for_idle.release_if_waiting () : bool);
        raise e
    in
    { state; host_wait_for_idle; ordinal; domain = Domain.spawn (worker debug_runtime) }

  let%diagn_sexp make_work device task =
    let%diagn_rt_sexp work () = schedule_task device task in
    Tnode.
      { description = "schedules {" ^ task.description ^ "} on device " ^ Int.to_string device.ordinal; work }

  type code = Backend.code [@@deriving sexp_of]
  type code_batch = Backend.code_batch [@@deriving sexp_of]
  type context = { device : device; ctx : Backend.context } [@@deriving sexp_of]
  type nonrec routine = context routine [@@deriving sexp_of]

  let name = "multicore " ^ Backend.name
  let init device = { device; ctx = Backend.init ~label:(name ^ " " ^ Int.to_string device.ordinal) }
  let initialize = Backend.initialize
  let is_initialized = Backend.is_initialized

  let finalize { device; ctx } =
    await device;
    Backend.finalize ctx

  let compile = Backend.compile
  let compile_batch = Backend.compile_batch

  let link { ctx; device } code =
    let task = Backend.link ctx code in
    { task with context = { ctx = task.context; device }; schedule = make_work device task.schedule }

  let link_batch { ctx; device } code_batch =
    Array.map (Backend.link_batch ctx code_batch)
      ~f:
        (Option.map ~f:(fun task ->
             { task with context = { ctx = task.context; device }; schedule = make_work device task.schedule }))

  let%diagn_sexp from_host ?rt context tn =
    if Option.is_some rt then
      raise @@ Utils.User_error "Multicore_backend.from_host: backend cannot be nested in another runtime";
    [%debug_notrace
      let work rt () = Backend.from_host ~rt context.ctx tn in
      schedule_task context.device
        Tnode.
          {
            description =
              "from_host " ^ Tnode.get_debug_name tn ^ " on device " ^ Int.to_string context.device.ordinal;
            work;
          }]

  let%diagn_sexp to_host ?rt context tn =
    if Option.is_some rt then
      raise @@ Utils.User_error "Multicore_backend.to_host: backend cannot be nested in another runtime";
    [%debug_notrace
      let work rt () = Backend.to_host ~rt context.ctx tn in
      schedule_task context.device
        Tnode.
          {
            description =
              "to_host " ^ Tnode.get_debug_name tn ^ " on device " ^ Int.to_string context.device.ordinal;
            work;
          }]

  let%diagn_sexp device_to_device ?rt tn ~into_merge_buffer ~dst ~src =
    if Option.is_some rt then
      raise
      @@ Utils.User_error "Multicore_backend.device_to_device: backend cannot be nested in another runtime";
    [%debug_notrace
      let work rt () = Backend.device_to_device ~rt tn ~into_merge_buffer ~dst:dst.ctx ~src:src.ctx in
      schedule_task dst.device
        Tnode.
          {
            description =
              "device_to_device " ^ Tnode.get_debug_name tn ^ " dst " ^ Int.to_string dst.device.ordinal
              ^ " src " ^ Int.to_string src.device.ordinal;
            work;
          }]

  let physical_merge_buffers = Backend.physical_merge_buffers
  let num_physical_devices () = Domain.recommended_domain_count () - 1
  let suggested_num_virtual_devices _device = 1
  let devices = Array.init (num_physical_devices ()) ~f:(fun ordinal -> spinup_device ~ordinal)

  let%track_sexp unsafe_cleanup ?(unsafe_shutdown = false) () =
    assert (Domain.is_main_domain ());
    let wait_for_finish device =
      await device;
      device.state.keep_spinning <- false;
      ignore (device.state.dev_wait.release_if_waiting () : bool)
    in
    Array.iter devices ~f:wait_for_finish;
    let cleanup ordinal device =
      Domain.join device.domain;
      device.host_wait_for_idle.finalize ();
      device.state.dev_wait.finalize ();
      if not unsafe_shutdown then devices.(ordinal) <- spinup_device ~ordinal
    in
    Array.iteri devices ~f:cleanup;
    Backend.unsafe_cleanup ~unsafe_shutdown ()

  let get_device ~ordinal = devices.(ordinal)
  let new_virtual_device device = device
  let get_physical_device device = device
  let get_ctx_device { device; _ } = device
  let get_name device = Int.to_string device.ordinal
  let to_ordinal { ordinal; _ } = ordinal
  let to_subordinal _ = 0
end

let lower_assignments ?name bindings asgns =
  let name = Option.value_or_thunk name ~default:(fun () -> Assignments.get_name_exn asgns) in
  let unoptim_ll_source = Utils.get_debug_formatter ~fname:(name ^ "-unoptimized.ll") in
  let ll_source = Utils.get_debug_formatter ~fname:(name ^ ".ll") in
  let cd_source = Utils.get_debug_formatter ~fname:(name ^ ".cd") in
  ( name,
    Assignments.lower_proc ~unoptim_ll_source ~ll_source ~cd_source ~name (Indexing.bound_symbols bindings)
      asgns )

let lower_batch_assignments ?names ?occupancy bindings asgns_l =
  let names =
    Option.value_or_thunk names ~default:(fun () ->
        Array.map asgns_l ~f:(fun asgns -> Assignments.get_name_exn asgns))
  in
  let prefix_name = String.(strip ~drop:(equal_char '_') @@ common_prefix @@ Array.to_list names) in
  let unoptim_ll_source = Utils.get_debug_formatter ~fname:(prefix_name ^ "-unoptimized.ll") in
  let ll_source = Utils.get_debug_formatter ~fname:(prefix_name ^ ".ll") in
  let cd_source = Utils.get_debug_formatter ~fname:(prefix_name ^ ".cd") in
  let bound = Indexing.bound_symbols bindings in
  let occupancy = Option.value occupancy ~default:(fun ~name:_ ~src_n:_ -> true) in
  Array.unzip
  @@ Array.mapi names ~f:(fun src_n name ->
         let asgns = asgns_l.(src_n) in
         if occupancy ~name ~src_n then
           ( Some name,
             Some (Assignments.lower_proc ~unoptim_ll_source ~ll_source ~cd_source ~name bound asgns) )
         else (None, None))

module type Simple_backend = sig
  type context [@@deriving sexp_of]
  type procedure [@@deriving sexp_of]
  type ctx_arrays [@@deriving sexp_of]
  type nonrec config = config

  val ctx_arrays : context -> ctx_arrays

  val compile :
    name:string ->
    opt_ctx_arrays:ctx_arrays option ->
    Indexing.unit_bindings ->
    Low_level.optimized ->
    procedure

  val compile_batch :
    names:string option array ->
    opt_ctx_arrays:ctx_arrays option ->
    Indexing.unit_bindings ->
    Low_level.optimized option array ->
    procedure option array

  val link_compiled : context -> procedure -> context * Indexing.lowered_bindings * Tnode.task * string
  val from_host : ?rt:(module Minidebug_runtime.Debug_runtime) -> context -> Tnode.t -> unit
  val to_host : ?rt:(module Minidebug_runtime.Debug_runtime) -> context -> Tnode.t -> unit

  val device_to_device :
    ?rt:(module Minidebug_runtime.Debug_runtime) ->
    Tnode.t ->
    into_merge_buffer:bool ->
    dst:context ->
    src:context ->
    unit

  val physical_merge_buffers : bool
  val name : string
  val initialize : unit -> unit
  val is_initialized : unit -> bool
  val init : label:string -> context
  val finalize : context -> unit
  val unsafe_cleanup : ?unsafe_shutdown:bool -> unit -> unit
end

module Simple_no_device_backend (Backend : Simple_backend with type config := config) : No_device_backend =
struct
  type code =
    | Postponed of { lowered : Low_level.optimized; bindings : Indexing.unit_bindings; name : string }
    | Compiled of Backend.procedure
  [@@deriving sexp_of]

  type code_batch =
    | Postponed of {
        lowereds : Low_level.optimized option array;
        bindings : Indexing.unit_bindings;
        names : string option array;
      }
    | Compiled of Backend.procedure option array
  [@@deriving sexp_of]

  include Backend

  let global_config = ref `Physical_devices_only

  let initialize config =
    global_config := config;
    initialize ()

  type nonrec routine = context routine [@@deriving sexp_of]

  let compile ?(shared = false) ?name bindings asgns : code =
    let name, lowered = lower_assignments ?name bindings asgns in
    if shared then Compiled (Backend.compile ~name ~opt_ctx_arrays:None bindings lowered)
    else Postponed { lowered; bindings; name }

  let compile_batch ?(shared = false) ?names ?occupancy bindings asgns_l : code_batch =
    let names, lowereds = lower_batch_assignments ?names ?occupancy bindings asgns_l in
    if shared then Compiled (compile_batch ~names ~opt_ctx_arrays:None bindings lowereds)
    else Postponed { lowereds; bindings; names }

  let link (old_context : context) (code : code) =
    let context, bindings, schedule, name =
      match code with
      | Postponed { lowered; bindings; name } ->
          let proc = Backend.compile ~name ~opt_ctx_arrays:(Some (ctx_arrays old_context)) bindings lowered in
          link_compiled old_context proc
      | Compiled code -> link_compiled old_context code
    in
    { context; schedule; bindings; name }

  let link_batch (old_context : context) (code_batch : code_batch) : routine option array =
    let procs =
      match code_batch with
      | Postponed { lowereds; bindings; names } ->
          Backend.compile_batch ~names ~opt_ctx_arrays:(Some (ctx_arrays old_context)) bindings lowereds
      | Compiled procs -> procs
    in
    Array.map procs
      ~f:
        (Option.map ~f:(fun proc ->
             let context, bindings, schedule, name = link_compiled old_context proc in
             { context; schedule; bindings; name }))
end

module Gccjit_device : No_device_backend = Simple_no_device_backend ((
  Gccjit_backend : Simple_backend with type context = Gccjit_backend.context))

module Gccjit_backend = Multicore_backend (Gccjit_device)

module Cuda_backend : Backend with type context = Cuda_backend.context = struct
  include (Cuda_backend : module type of Cuda_backend with type config := config)

  type nonrec routine = context routine [@@deriving sexp_of]

  let name = "cuda"

  let compile ?shared:_ ?name bindings asgns : code =
    let name, lowered = lower_assignments ?name bindings asgns in
    compile ~name bindings lowered

  let compile_batch ?shared:_ ?names ?occupancy bindings asgns_l =
    let names, lowereds = lower_batch_assignments ?names ?occupancy bindings asgns_l in
    compile_batch ~names bindings lowereds

  let link context code =
    let context, bindings, schedule = link context code in
    { context; schedule; bindings; name }

  let link_batch context code_batch : routine option array =
    let context, bindings, schedules = link_batch context code_batch in
    Array.map schedules ~f:(Option.map ~f:(fun schedule -> { context; schedule; bindings; name }))
end

let reinitialize (module Backend : Backend) config =
  if not @@ Backend.is_initialized () then Backend.initialize config
  else (
    Core.Gc.full_major ();
    Backend.unsafe_cleanup ())
