open Base
module Debug_runtime = Utils.Debug_runtime
module Tn = Tnode
open Backend_intf
open Backend_impl

let _get_local_debug_runtime = Utils._get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

let check_merge_buffer ~scheduled_node ~code_node =
  let name = function Some tn -> Tnode.debug_name tn | None -> "none" in
  match (scheduled_node, code_node) with
  | _, None -> ()
  | Some actual, Some expected when Tnode.equal actual expected -> ()
  | _ ->
      raise
      @@ Utils.User_error
           ("Merge buffer mismatch, on stream: " ^ name scheduled_node ^ ", expected by code: "
          ^ name code_node)

module Add_buffer_retrieval_and_syncing (Backend : No_buffer_retrieval_or_syncing) = struct
  let work_for context tn =
    let stream = context.stream in
    let default () = Some (Backend.all_work stream) in
    if not @@ Map.mem context.ctx_arrays tn then None
    else
      Hashtbl.update_and_return stream.queried_work_for tn ~f:(function
        | None | Some None -> default ()
        | Some (Some _ as event) -> event)

  let%diagn2_l_sexp from_host (ctx : Backend.context) tn =
    match (tn, Map.find ctx.ctx_arrays tn) with
    | { Tn.array = (lazy (Some hosted)); _ }, Some dst ->
        [%log "copying", Tn.debug_name tn, "to", (dst : Backend.buffer_ptr), "from host"];
        Backend.from_host ~dst_ptr:dst ~dst:ctx hosted;
        true
    | _ -> false

  let%diagn2_l_sexp to_host (ctx : Backend.context) (tn : Tn.t) =
    match (tn, Map.find ctx.ctx_arrays tn) with
    | { Tn.array = (lazy (Some hosted)); _ }, Some src ->
        [%log "copying", Tn.debug_name tn, "at", (src : Backend.buffer_ptr), "to host"];
        Backend.to_host ~src_ptr:src ~src:ctx hosted;
        true
    | _ -> false

  let%diagn2_l_sexp device_to_device (tn : Tn.t) ~into_merge_buffer ~(dst : Backend.context)
      ~(src : Backend.context) =
    let ordinal_of ctx = ctx.stream.device.ordinal in
    let name_of ctx = Backend.(get_name ctx.stream) in
    let same_device = ordinal_of dst = ordinal_of src in
    if same_device && (Tn.known_shared_cross_stream tn || String.equal (name_of src) (name_of dst))
    then false
    else
      match Map.find src.ctx_arrays tn with
      | None -> false
      | Some s_arr -> (
          match into_merge_buffer with
          | No -> (
              match Map.find dst.ctx_arrays tn with
              | None -> false
              | Some d_arr ->
                  Backend.(
                    device_to_device tn ~into_merge_buffer ~dst_ptr:(Some d_arr) ~dst ~src_ptr:s_arr
                      ~src);
                  [%log
                    "copied",
                      Tn.debug_name tn,
                      "from",
                      name_of src,
                      "at",
                      (s_arr : Backend.buffer_ptr),
                      "to",
                      (d_arr : Backend.buffer_ptr)];
                  true)
          | Streaming when same_device ->
              Backend.(
                device_to_device tn ~into_merge_buffer ~dst_ptr:None ~dst ~src_ptr:s_arr ~src);
              [%log "using merge buffer for", Tn.debug_name tn, "from", name_of src];
              true
          | Copy | Streaming ->
              Backend.(
                device_to_device tn ~into_merge_buffer ~dst_ptr:None ~dst ~src_ptr:s_arr ~src);
              [%log "copied into merge buffer", Tn.debug_name tn, "from", name_of src];
              true)
end

module Alloc_buffer_ignore_stream
    (Device_types : Device_types)
    (Backend : Alloc_buffer with type buffer_ptr = Device_types.buffer_ptr and type stream := unit) :
  Alloc_buffer with type buffer_ptr = Backend.buffer_ptr and type stream = Device_types.stream =
struct
  include Device_types

  let alloc_buffer ?old_buffer ~size_in_bytes _stream =
    Backend.alloc_buffer ?old_buffer ~size_in_bytes ()

  let alloc_zero_init_array prec ~dims _stream = Backend.alloc_zero_init_array prec ~dims ()
  let free_buffer = Option.map Backend.free_buffer ~f:(fun memfree _stream ptr -> memfree () ptr)
end

module Multicore_scheduler (Backend : For_add_scheduler) :
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
  let name = "multicore_" ^ name
  let get_name stream = [%string "%{name}:0:%{stream.stream_id#Int}"]

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
module Sync_scheduler (Backend : For_add_scheduler) = struct
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
  let name = "sync_" ^ Backend.name
  let await _stream = ()
  (* let global_run_no = ref 0 *)

  let initialize = Backend.initialize
  let is_initialized = Backend.is_initialized
  let get_name stream = [%string "%{name}:0:%{stream.stream_id#Int}"]
  let schedule_task _stream task = Task.run task
end

let lower_assignments ?name bindings asgns =
  let name = Option.value_or_thunk name ~default:(fun () -> Assignments.get_name_exn asgns) in
  let unoptim_ll_source = Utils.get_debug_formatter ~fname:(name ^ "-unoptimized.ll") in
  let ll_source = Utils.get_debug_formatter ~fname:(name ^ ".ll") in
  let cd_source = Utils.get_debug_formatter ~fname:(name ^ ".cd") in
  ( name,
    Assignments.lower ~unoptim_ll_source ~ll_source ~cd_source ~name
      (Indexing.bound_symbols bindings) asgns )

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
             Some (Assignments.lower ~unoptim_ll_source ~ll_source ~cd_source ~name bound asgns) )
         else (None, None))

let verify_prior_context ~is_in_context ~ctx_arrays ~from_prior_context traced_stores =
  Set.iter from_prior_context ~f:(fun tn ->
      let node = Array.find_map traced_stores ~f:(fun store -> Hashtbl.find store tn) in
      if
        Option.value_map node ~default:false ~f:(fun node ->
            is_in_context node && not (Option.is_some @@ Map.find ctx_arrays tn))
      then raise @@ Utils.User_error ("The linked context lacks node " ^ Tnode.debug_name tn))

let from_prior_context_batch comps =
  Array.filter_map comps ~f:(fun comp ->
      Option.map comp ~f:(fun comp ->
          Set.diff (Assignments.context_nodes comp.Assignments.asgns) comp.embedded_nodes))
  |> Array.fold ~init:(Set.empty (module Tnode)) ~f:Set.union

(** Adds a scheduler and brings a lowered no-device backend on par with lowered device backends. *)
module Add_device
    (Add_scheduler : functor
      (Impl : For_add_scheduler)
      -> With_scheduler with type buffer_ptr = Impl.buffer_ptr)
    (Backend : Lowered_no_device_backend) : Lowered_backend = struct
  include Backend

  type code =
    | Postponed of {
        lowered : Low_level.optimized;
        bindings : Indexing.unit_bindings;
        name : string;
      }
    | Compiled of { lowered : Low_level.optimized; proc : Backend.procedure }
  [@@deriving sexp_of]

  type code_batch =
    | Postponed of {
        lowereds : Low_level.optimized option array;
        bindings : Indexing.unit_bindings;
        names : string option array;
      }
    | Compiled of {
        lowereds : Low_level.optimized option array;
        procs : ctx_arrays option * Backend.procedure option array;
      }
  [@@deriving sexp_of]

  let compile ?(shared = false) ~name bindings lowered : code =
    if shared then
      let proc = compile ~name ~opt_ctx_arrays:None bindings lowered in
      Compiled { lowered; proc }
    else Postponed { lowered; bindings; name }

  let compile_batch ?(shared = false) ~names bindings lowereds : code_batch =
    if shared then
      let procs = compile_batch ~names ~opt_ctx_arrays:None bindings lowereds in
      Compiled { lowereds; procs }
    else Postponed { lowereds; bindings; names }

  include Add_scheduler (Backend)

  let link context (code : code) =
    let runner_label = get_name context.stream in
    let ctx_arrays = context.ctx_arrays in
    let merge_buffer = context.stream.merge_buffer in
    match code with
    | Postponed { lowered; bindings; name } ->
        let proc = Backend.compile ~name ~opt_ctx_arrays:(Some ctx_arrays) bindings lowered in
        link_compiled ~merge_buffer ~runner_label ctx_arrays proc
    | Compiled { proc; _ } -> link_compiled ~merge_buffer ~runner_label ctx_arrays proc

  let link_batch context (code_batch : code_batch) =
    let runner_label = get_name context.stream in
    let ctx_arrays = context.ctx_arrays in
    let merge_buffer = context.stream.merge_buffer in
    (* FIXME: why are we getting and ignoring opt_ctx_arrays here? *)
    let _opt_ctx_arrays, procs =
      match code_batch with
      | Postponed { lowereds; bindings; names } ->
          Backend.compile_batch ~names ~opt_ctx_arrays:(Some ctx_arrays) bindings lowereds
      | Compiled { procs; _ } -> procs
    in
    let (ctx_arrays, bindings), schedules =
      Array.fold_map procs ~init:(ctx_arrays, None) ~f:(fun (ctx_arrays, bindings) -> function
        | Some proc ->
            let ctx_arrays, bindings', schedule =
              link_compiled ~merge_buffer ~runner_label ctx_arrays proc
            in
            Option.iter bindings ~f:(fun bindings -> assert (phys_equal bindings bindings'));
            ((ctx_arrays, Some bindings'), Some (ctx_arrays, schedule))
        | None -> ((ctx_arrays, bindings), None))
    in
    (ctx_arrays, Option.value_exn ~here:[%here] bindings, schedules)

  let from_host ~dst_ptr ~dst hosted =
    let work () = host_to_buffer hosted ~dst:dst_ptr in
    (* TODO: pass description to from_host. *)
    schedule_task dst.stream
      (Task.Task
         { context_lifetime = dst; description = "from_host on " ^ get_name dst.stream; work })

  let to_host ~src_ptr ~src hosted =
    let work () = buffer_to_host hosted ~src:src_ptr in
    (* TODO: pass description to to_host. *)
    schedule_task src.stream
      (Task.Task { context_lifetime = src; description = "to_host on " ^ get_name src.stream; work })

  let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src =
    let dev = dst.stream in
    let size_in_bytes = Tnode.size_in_bytes tn in
    let work =
      (* TODO: log the operation if [Utils.settings.with_log_level > 0]. *)
      match (into_merge_buffer, dst_ptr) with
      | No, None -> invalid_arg "Multicore_scheduler.device_to_device: missing dst_ptr"
      | No, Some dst_ptr -> fun () -> buffer_to_buffer ~dst:dst_ptr ~src:src_ptr ~size_in_bytes
      | Streaming, _ -> fun () -> dev.merge_buffer := Some (src_ptr, tn)
      | Copy, _ ->
          fun () ->
            let size_in_bytes = Tnode.size_in_bytes tn in
            let allocated_capacity =
              match dev.allocated_buffer with None -> 0 | Some buf -> buf.size_in_bytes
            in
            if allocated_capacity < size_in_bytes then
              dev.allocated_buffer <-
                Some (alloc_buffer ?old_buffer:dev.allocated_buffer ~size_in_bytes dst.stream);
            let merge_ptr = (Option.value_exn dev.allocated_buffer).ptr in
            dev.merge_buffer := Some (merge_ptr, tn);
            buffer_to_buffer ~dst:merge_ptr ~src:src_ptr ~size_in_bytes
    in
    let description =
      "device_to_device " ^ Tnode.debug_name tn ^ " dst " ^ get_name dev ^ " src "
      ^ get_name src.stream
    in
    schedule_task dev (Task.Task { context_lifetime = (src, dst); description; work })
end

module Raise_backend (Device : Lowered_backend) : Backend = struct
  include Device
  include Add_buffer_retrieval_and_syncing (Device)

  type nonrec code = {
    from_prior_context : Set.M(Tnode).t;
    name : string;
    lowered : Low_level.optimized;
    code : code;
    expected_merge_node : Tnode.t option;
  }
  [@@deriving sexp_of]

  type nonrec code_batch = {
    from_prior_context : Set.M(Tnode).t;
    lowereds : Low_level.optimized option array;
    code_batch : code_batch;
    names : string option array;
    expected_merge_nodes : Tnode.t option array;
  }
  [@@deriving sexp_of]

  let compile ?shared ?name bindings comp : code =
    let name, lowered = lower_assignments ?name bindings comp.Assignments.asgns in
    let code = compile ?shared ~name bindings lowered in
    let from_prior_context = Set.diff (Assignments.context_nodes comp.asgns) comp.embedded_nodes in
    { from_prior_context; name; lowered; code; expected_merge_node = lowered.Low_level.merge_node }

  let compile_batch ?shared ?names ?occupancy bindings comps =
    let names, lowereds =
      lower_batch_assignments ?names ?occupancy bindings
      @@ Array.map comps ~f:(fun c -> c.Assignments.asgns)
    in
    let code_batch = compile_batch ?shared ~names bindings lowereds in
    let from_prior_context =
      from_prior_context_batch
      @@ Array.mapi lowereds ~f:(fun i -> Option.map ~f:(fun _ -> comps.(i)))
    in
    {
      from_prior_context;
      names;
      lowereds;
      code_batch;
      expected_merge_nodes =
        Array.map lowereds ~f:(fun lowered ->
            Option.(join @@ map lowered ~f:(fun optim -> optim.Low_level.merge_node)));
    }

  let link context (code : code) =
    verify_prior_context ~is_in_context ~ctx_arrays:context.ctx_arrays
      ~from_prior_context:code.from_prior_context [| code.lowered.traced_store |];
    let inputs, outputs = Low_level.input_and_output_nodes code.lowered in
    let ctx_arrays, bindings, schedule = link context code.code in
    let context = make_child ~ctx_arrays context in
    let schedule =
      Task.prepend schedule ~work:(fun () ->
          check_merge_buffer
            ~scheduled_node:(scheduled_merge_node context.stream)
            ~code_node:code.expected_merge_node)
    in
    { context; schedule; bindings; name = code.name; inputs; outputs }

  let link_batch context code_batch =
    verify_prior_context ~is_in_context ~ctx_arrays:context.ctx_arrays
      ~from_prior_context:code_batch.from_prior_context
    @@ Array.filter_map code_batch.lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store));
    let _ctx_arrays, bindings, schedules = link_batch context code_batch.code_batch in
    Array.fold_mapi schedules ~init:context ~f:(fun i context -> function
      | None -> (context, None)
      | Some (ctx_arrays, schedule) ->
          let context = make_child ~ctx_arrays context in
          let expected_merge_node = code_batch.expected_merge_nodes.(i) in
          let inputs, outputs =
            Low_level.input_and_output_nodes @@ Option.value_exn code_batch.lowereds.(i)
          in
          let schedule =
            Task.prepend schedule ~work:(fun () ->
                check_merge_buffer
                  ~scheduled_node:(scheduled_merge_node context.stream)
                  ~code_node:expected_merge_node)
          in
          (context, Some { context; schedule; bindings; name; inputs; outputs }))
end

module Cuda_backend : Backend = Raise_backend ((Cuda_backend : Lowered_backend))

module Make_device_backend_from_lowered
    (Add_scheduler : functor
      (Impl : For_add_scheduler)
      -> With_scheduler with type buffer_ptr = Impl.buffer_ptr)
    (Backend_impl : Lowered_no_device_backend) =
struct
  module Lowered_device = Add_device (Add_scheduler) (Backend_impl)
  module Backend_device = Raise_backend (Lowered_device)
  include Backend_device
end

module Cc_multicore = Make_device_backend_from_lowered (Multicore_scheduler) (Cc_backend)
module Gcc_multicore = Make_device_backend_from_lowered (Multicore_scheduler) (Gcc_backend)
module Cc_sync = Make_device_backend_from_lowered (Sync_scheduler) (Cc_backend)
module Gcc_sync = Make_device_backend_from_lowered (Sync_scheduler) (Gcc_backend)

let reinitialize (module Backend : Backend) config =
  if not @@ Backend.is_initialized () then Backend.initialize config
  else (
    Stdlib.Gc.full_major ();
    Backend.initialize config)

let%track3_sexp finalize (type buffer_ptr dev runner event)
    (module Backend : Backend
      with type buffer_ptr = buffer_ptr
       and type dev = dev
       and type runner = runner
       and type event = event) (ctx : Backend.context) : unit =
  Option.iter Backend.free_buffer ~f:(fun mem_free ->
      if
        Atomic.compare_and_set ctx.finalized false true
        && (not @@ Atomic.get ctx.stream.device.released)
      then (
        Backend.await ctx.stream;
        Map.iteri ctx.ctx_arrays ~f:(fun ~key ~data ->
            if
              (not (Option.exists ctx.parent ~f:(fun pc -> Map.mem pc.ctx_arrays key)))
              && not (Hashtbl.mem ctx.stream.device.cross_stream_candidates key)
            then mem_free ctx.stream data)))

let fresh_backend ?backend_name ?(config = Only_devices_parallel) () =
  let backend =
    match
      Option.value_or_thunk backend_name ~default:(fun () ->
          Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
      |> String.lowercase
    with
    | "cc" -> (module Cc_multicore : Backend)
    | "gccjit" -> (module Gcc_multicore : Backend)
    | "sync_cc" -> (module Cc_sync : Backend)
    | "sync_gccjit" -> (module Gcc_sync : Backend)
    | "cuda" -> (module Cuda_backend : Backend)
    | backend -> invalid_arg [%string "Backends.fresh_backend: unknown backend %{backend}"]
  in
  reinitialize backend config;
  backend
