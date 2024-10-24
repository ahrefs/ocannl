open Base
open Backend_types.Types
module Debug_runtime = Utils.Debug_runtime
module Tn = Tnode

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

module Add_buffer_retrieval_and_syncing (Backend : Backend_types.No_buffer_retrieval_or_syncing) =
struct
  type context = Backend.context
  type event = Backend.event

  let work_for context tn =
    let stream = Backend.get_ctx_stream context in
    let default () = Some (Backend.all_work stream) in
    if not @@ Map.mem (Backend.ctx_arrays context) tn then None
    else
      Hashtbl.update_and_return stream.requested_work_for tn ~f:(function
        | None | Some None -> default ()
        | Some (Some _ as event) -> event)

  let%diagn2_l_sexp from_host (ctx : Backend.context) tn =
    match (tn, Map.find (Backend.ctx_arrays ctx) tn) with
    | { Tn.array = (lazy (Some hosted)); _ }, Some dst ->
        [%log "copying", Tn.debug_name tn, "to", (dst : Backend.buffer_ptr), "from host"];
        Backend.from_host ~dst_ptr:dst ~dst:ctx hosted;
        true
    | _ -> false

  let%diagn2_l_sexp to_host (ctx : Backend.context) (tn : Tn.t) =
    match (tn, Backend.(Map.find @@ ctx_arrays ctx) tn) with
    | { Tn.array = (lazy (Some hosted)); _ }, Some src ->
        [%log "copying", Tn.debug_name tn, "at", (src : Backend.buffer_ptr), "to host"];
        Backend.to_host ~src_ptr:src ~src:ctx hosted;
        true
    | _ -> false

  let%diagn2_l_sexp device_to_device (tn : Tn.t) ~into_merge_buffer ~(dst : Backend.context)
      ~(src : Backend.context) =
    let ordinal_of ctx = Backend.(to_ordinal @@ get_stream_device @@ get_ctx_stream ctx) in
    let name_of ctx = Backend.(get_name @@ get_ctx_stream ctx) in
    let same_device = ordinal_of dst = ordinal_of src in
    if same_device && (Tn.known_shared_cross_stream tn || String.equal (name_of src) (name_of dst))
    then false
    else
      match Backend.(Map.find @@ ctx_arrays src) tn with
      | None -> false
      | Some s_arr -> (
          match into_merge_buffer with
          | No -> (
              match Backend.(Map.find @@ ctx_arrays dst) tn with
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

module Multicore_backend (Backend : Backend_types.No_device_backend) = struct
  include Backend
  module Domain = Domain [@warning "-3"]

  type task_list = Task.t Utils.mutable_list [@@deriving sexp_of]

  module Mut = Stdlib.Mutex
  module Queue = Saturn_lockfree.Single_prod_single_cons_queue

  type task_queue = Task.t Queue.t

  let sexp_of_task_queue q =
    Sexp.(List [ Atom "task_queue_of_size"; Atom (Int.to_string @@ Queue.size q) ])

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

  type runner = unit Domain.t

  let sexp_of_runner (d : runner) = Sexp.Atom ("domain-" ^ Int.to_string (Domain.get_id d :> int))

  type device = CPU [@@deriving sexp_of]
  type event = Not_implemented_yet [@@deriving sexp_of]
  type nonrec stream = (buffer_ptr, event, device, stream_state, runner) stream [@@deriving sexp_of]

  (** TODO: Blocks till the event completes, if it's not done already. *)
  let sync Not_implemented_yet = ()

  (** TODO: Whether the event completed. *)
  let is_done Not_implemented_yet = true

  (** TODO: Schedules waiting for the given event on the context's stream. *)
  let will_wait_for _ctx Not_implemented_yet = ()

  let alloc_buffer ?old_buffer ~size_in_bytes _stream = alloc_buffer ?old_buffer ~size_in_bytes ()
  let get_used_memory _device = get_used_memory ()

  type nonrec code = code [@@deriving sexp_of]
  type nonrec code_batch = code_batch [@@deriving sexp_of]

  let is_dev_queue_empty state = Queue.size state.queue = 0
  let is_idle stream = is_dev_queue_empty stream.state && stream.state.is_ready
  let name = "multicore " ^ name

  let%track3_l_sexp await stream =
    assert (Domain.is_main_domain ());
    let d = stream.state in
    if (not @@ is_idle stream) && d.keep_spinning then (
      Mut.lock d.mut;
      while (not @@ is_idle stream) && d.keep_spinning do
        (* If the stream "is ready", it needs to be woken up first to finish the work. *)
        if d.is_ready then Stdlib.Condition.broadcast d.dev_wait_for_work;
        Stdlib.Condition.wait d.host_wait_for_idle d.mut
      done;
      Mut.unlock d.mut;
      Option.iter d.stream_error ~f:(fun e -> Exn.reraise e @@ name ^ " " ^ stream.unique_name))

  (** TODO: Returns the event indicating if any currently running or scheduled computations on the
      stream have completed. *)
  let all_work _stream = Not_implemented_yet

  let%track3_l_sexp schedule_task stream task =
    assert (Domain.is_main_domain ());
    [%log_result "schedule_task", Task.describe task, stream.unique_name];
    let d = stream.state in
    Option.iter d.stream_error ~f:(fun e -> Exn.reraise e @@ name ^ " " ^ stream.unique_name);
    if not d.keep_spinning then invalid_arg "Multicore_backend: stream not available";
    if not @@ Queue.try_push d.queue task then (
      await stream;
      Queue.push_exn d.queue task);
    if d.is_ready then (
      Mut.lock d.mut;
      Stdlib.Condition.broadcast d.dev_wait_for_work;
      Mut.unlock d.mut)

  let global_run_no = ref 0

  let%track3_l_sexp spinup_stream ~unique_name : stream =
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
        [%log1 unique_name, "exception", Exn.to_string e];
        (* TODO: we risk raising this error multiple times because await and schedule_task raise
           stream_error. But this is fine if we assume all exceptions are fatal. *)
        raise e
    in
    make_stream ~device:CPU ~state ~unique_name ~runner:(Domain.spawn worker)

  type nonrec context = { stream : stream; ctx : context } [@@deriving sexp_of]

  let ctx_arrays context = ctx_arrays context.ctx

  type nonrec routine = context Backend_types.Types.routine [@@deriving sexp_of]
  (** This overrides the routine type from [Backend]. *)

  let init stream = { stream; ctx = init (name ^ " " ^ stream.unique_name) }
  let initialize = initialize
  let is_initialized = is_initialized

  let finalize { stream; ctx } =
    await stream;
    finalize ctx

  let compile = compile
  let compile_batch = compile_batch
  let get_name stream = stream.unique_name

  let link { ctx; stream } code =
    let task = link ~merge_buffer:stream.merge_buffer ctx code in
    {
      task with
      context = { ctx = task.context; stream };
      schedule = Task.enschedule ~schedule_task ~get_stream_name:get_name stream task.schedule;
    }

  let link_batch { ctx; stream } code_batch =
    let ctx, routines = link_batch ~merge_buffer:stream.merge_buffer ctx code_batch in
    ( { ctx; stream },
      Array.map routines
        ~f:
          (Option.map ~f:(fun task ->
               {
                 task with
                 context = { ctx = task.context; stream };
                 schedule =
                   Task.enschedule ~schedule_task ~get_stream_name:get_name stream task.schedule;
               })) )

  module Dynarr = Stdlib.Dynarray

  let num_devices () = 1
  let suggested_num_streams CPU = Domain.recommended_domain_count () - 1
  let used_names = Hash_set.create (module String)

  let cleanup_stream stream =
    assert (Domain.is_main_domain ());
    await stream;
    stream.state.keep_spinning <- false;
    Stdlib.Condition.broadcast stream.state.dev_wait_for_work;
    Hash_set.remove used_names stream.unique_name;
    Domain.join stream.runner

  let get_device ~ordinal =
    if ordinal <> 0 then
      invalid_arg [%string "Multicore_backend.get_device %{ordinal#Int}: only device 0 exists"];
    CPU

  let new_stream CPU =
    assert (Domain.is_main_domain ());
    let rec unique_name suffix =
      let name = "stream " ^ Int.to_string suffix in
      if Hash_set.mem used_names name then unique_name (suffix + 1) else name
    in
    let unique_name = unique_name 0 in
    Hash_set.add used_names unique_name;
    let stream = spinup_stream ~unique_name in
    Stdlib.Gc.finalise cleanup_stream stream;
    stream

  let get_stream_device _stream = CPU
  let get_ctx_stream { stream; _ } = stream
  let to_ordinal _ = 0

  let from_host ~dst_ptr ~dst hosted =
    let work () = host_to_buffer hosted ~dst:dst_ptr in
    (* TODO: pass description to from_host. *)
    schedule_task dst.stream
      (Task.Task
         { context_lifetime = dst; description = "from_host on " ^ dst.stream.unique_name; work })

  let to_host ~src_ptr ~src hosted =
    let work () = buffer_to_host hosted ~src:src_ptr in
    (* TODO: pass description to to_host. *)
    schedule_task src.stream
      (Task.Task
         { context_lifetime = src; description = "to_host on " ^ src.stream.unique_name; work })

  let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src =
    let dev = dst.stream in
    let work =
      (* TODO: log the operation if [Utils.settings.with_log_level > 0]. *)
      match (into_merge_buffer, dst_ptr) with
      | No, None -> invalid_arg "Multicore_backend.device_to_device: missing dst_ptr"
      | No, Some dst_ptr -> fun () -> buffer_to_buffer ~dst:dst_ptr ~src:src_ptr
      | Streaming, _ -> fun () -> dev.merge_buffer := Some (src_ptr, tn)
      | Copy, _ ->
          fun () ->
            let size_in_bytes = Tnode.size_in_bytes tn in
            let allocated_capacity =
              Option.value ~default:0 @@ Option.map dev.allocated_buffer ~f:snd
            in
            if allocated_capacity < size_in_bytes then
              dev.allocated_buffer <-
                Some
                  ( Backend.alloc_buffer ?old_buffer:dev.allocated_buffer ~size_in_bytes (),
                    size_in_bytes );
            let merge_ptr = fst @@ Option.value_exn dev.allocated_buffer in
            dev.merge_buffer := Some (merge_ptr, tn);
            buffer_to_buffer ~dst:merge_ptr ~src:src_ptr
    in
    let description =
      "device_to_device " ^ Tnode.debug_name tn ^ " dst " ^ dev.unique_name ^ " src "
      ^ src.stream.unique_name
    in
    schedule_task dev (Task.Task { context_lifetime = (src, dst); description; work })
end

(** For debugging, allow [Sync_backend(...).suggested_num_streams] calls to return >1 numbers. *)
let sync_suggested_num_streams = ref 1

(** A minimalisitc wrapper creating backends where all calls run synchronously on the main thread.
    There is only one device, but an arbitrary number of streams. *)
module Sync_backend (Backend : Backend_types.No_device_backend) = struct
  include Backend

  type event = unit [@@deriving sexp_of]

  let sync () = ()
  let is_done () = true
  let will_wait_for _context () = ()

  let alloc_buffer ?old_buffer ~size_in_bytes _stream =
    Backend.alloc_buffer ?old_buffer ~size_in_bytes ()

  type device = CPU [@@deriving sexp_of]

  let to_ordinal CPU = 0

  let get_device ~ordinal =
    if ordinal <> 0 then
      invalid_arg @@ "Sync_backend.get_device: there is only one device, but ordinal="
      ^ Int.to_string ordinal;
    CPU

  let num_devices () = 1
  let suggested_num_streams CPU = !sync_suggested_num_streams
  let get_used_memory CPU = Backend.get_used_memory ()
  let next_stream = ref 0

  type stream_state = unit [@@deriving sexp_of]
  type runner = unit [@@deriving sexp_of]
  type nonrec stream = (buffer_ptr, event, device, stream_state, runner) stream [@@deriving sexp_of]

  let new_stream CPU : stream =
    Int.incr next_stream;
    make_stream ~device:CPU ~state:()
      ~unique_name:("stream " ^ Int.to_string (!next_stream - 1))
      ~runner:()

  type code = Backend.code [@@deriving sexp_of]
  type code_batch = Backend.code_batch [@@deriving sexp_of]

  let all_work _stream = ()
  let is_idle _stream = true
  let name = "sync " ^ Backend.name
  let await _stream = ()
  (* let global_run_no = ref 0 *)

  type context = { stream : stream; ctx : Backend.context } [@@deriving sexp_of]

  let get_ctx_stream context = context.stream
  let get_stream_device _stream = CPU
  let ctx_arrays context = ctx_arrays context.ctx

  type nonrec routine = context Backend_types.Types.routine [@@deriving sexp_of]
  (** This overrides the routine type from [Backend]. *)

  let init stream = { stream; ctx = Backend.init name }
  let initialize = Backend.initialize
  let is_initialized = Backend.is_initialized
  let finalize { stream = _; ctx } = Backend.finalize ctx
  let compile = Backend.compile
  let compile_batch = Backend.compile_batch

  let link { ctx; stream } code =
    let task = Backend.link ~merge_buffer:stream.merge_buffer ctx code in
    { task with context = { ctx = task.context; stream } }

  let link_batch { ctx; stream } code_batch =
    let ctx, routines = Backend.link_batch ~merge_buffer:stream.merge_buffer ctx code_batch in
    ( { ctx; stream },
      Array.map routines
        ~f:(Option.map ~f:(fun task -> { task with context = { ctx = task.context; stream } })) )

  let get_name stream = stream.unique_name
  let from_host ~dst_ptr ~dst:_ hosted = host_to_buffer hosted ~dst:dst_ptr
  let to_host ~src_ptr ~src:_ hosted = buffer_to_host hosted ~src:src_ptr

  let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src:_ =
    let dev = dst.stream in
    (* TODO: log the operation if [Utils.settings.with_log_level > 0]. *)
    match (into_merge_buffer, dst_ptr) with
    | No, None -> invalid_arg "Sync_backend.device_to_device: missing dst_ptr"
    | No, Some dst_ptr -> buffer_to_buffer ~dst:dst_ptr ~src:src_ptr
    | Streaming, _ -> dev.merge_buffer := Some (src_ptr, tn)
    | Copy, _ ->
        let size_in_bytes = Tnode.size_in_bytes tn in
        let allocated_capacity =
          Option.value ~default:0 @@ Option.map dev.allocated_buffer ~f:snd
        in
        if allocated_capacity < size_in_bytes then
          dev.allocated_buffer <-
            Some
              ( Backend.alloc_buffer ?old_buffer:dev.allocated_buffer ~size_in_bytes (),
                size_in_bytes );
        let merge_ptr = fst @@ Option.value_exn dev.allocated_buffer in
        dev.merge_buffer := Some (merge_ptr, tn);
        buffer_to_buffer ~dst:merge_ptr ~src:src_ptr
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

let verify_prior_context ~ctx_arrays ~is_in_context ~prior_context ~from_prior_context traced_stores
    =
  let olds = ctx_arrays prior_context in
  Set.iter from_prior_context ~f:(fun tn ->
      let node = Array.find_map traced_stores ~f:(fun store -> Hashtbl.find store tn) in
      if
        Option.value_map node ~default:false ~f:(fun node ->
            is_in_context node && not (Option.is_some @@ Map.find olds tn))
      then raise @@ Utils.User_error ("The linked context lacks node " ^ Tnode.debug_name tn))

let from_prior_context_batch comps =
  Array.filter_map comps ~f:(fun comp ->
      Option.map comp ~f:(fun comp ->
          Set.diff (Assignments.context_nodes comp.Assignments.asgns) comp.embedded_nodes))
  |> Array.fold ~init:(Set.empty (module Tnode)) ~f:Set.union

module Lowered_no_device_backend (Backend : Backend_types.Lowered_no_device_backend) = struct
  include Backend

  type code =
    | Postponed of {
        comp : Assignments.comp;
        lowered : Low_level.optimized;
        bindings : Indexing.unit_bindings;
        name : string;
      }
    | Compiled of {
        from_prior_context : Set.M(Tnode).t;
        lowered : Low_level.optimized;
        proc : Backend.procedure;
      }
  [@@deriving sexp_of]

  type code_batch =
    | Postponed of {
        comps : Assignments.comp option array;
        lowereds : Low_level.optimized option array;
        bindings : Indexing.unit_bindings;
        names : string option array;
      }
    | Compiled of {
        from_prior_context : Set.M(Tnode).t;
        lowereds : Low_level.optimized option array;
        procs : ctx_arrays option * Backend.procedure option array;
      }
  [@@deriving sexp_of]

  let global_config = ref Only_devices_parallel

  let initialize config =
    global_config := config;
    initialize config

  type nonrec routine = context routine [@@deriving sexp_of]

  let expected_merge_node : code -> _ = function
    | Postponed { lowered = Low_level.{ merge_node; _ }; _ }
    | Compiled { lowered = Low_level.{ merge_node; _ }; _ } ->
        merge_node

  let expected_merge_nodes : code_batch -> _ = function
    | Postponed { lowereds; _ } | Compiled { lowereds; _ } ->
        Array.map lowereds ~f:(fun lowered ->
            Option.(join @@ map lowered ~f:(fun optim -> optim.merge_node)))

  let get_lowered : code -> _ = function
    | Postponed { lowered; _ } | Compiled { lowered; _ } -> lowered

  let get_lowereds : code_batch -> _ = function
    | Postponed { lowereds; _ } -> lowereds
    | Compiled { lowereds; _ } -> lowereds

  let compile ?(shared = false) ?name bindings comp : code =
    let name, lowered = lower_assignments ?name bindings comp.Assignments.asgns in

    if shared then
      let proc = compile ~name ~opt_ctx_arrays:None bindings lowered in
      let from_prior_context =
        Set.diff (Assignments.context_nodes comp.asgns) comp.embedded_nodes
      in
      Compiled { from_prior_context; lowered; proc }
    else Postponed { comp; lowered; bindings; name }

  let compile_batch ?(shared = false) ?names ?occupancy bindings comps : code_batch =
    let names, lowereds =
      lower_batch_assignments ?names ?occupancy bindings
      @@ Array.map comps ~f:(fun c -> c.Assignments.asgns)
    in
    if shared then
      let procs = compile_batch ~names ~opt_ctx_arrays:None bindings lowereds in
      let from_prior_context =
        from_prior_context_batch
        @@ Array.mapi lowereds ~f:(fun i -> Option.map ~f:(fun _ -> comps.(i)))
      in
      Compiled { lowereds; procs; from_prior_context }
    else
      Postponed
        {
          comps = Array.mapi lowereds ~f:(fun i -> Option.map ~f:(fun _ -> comps.(i)));
          lowereds;
          bindings;
          names;
        }

  let link ~merge_buffer (prior_context : context) (code : code) =
    let lowered = get_lowered code in
    let verify from_prior_context =
      verify_prior_context ~ctx_arrays ~is_in_context ~prior_context ~from_prior_context
        [| lowered.traced_store |]
    in
    let inputs, outputs = Low_level.input_and_output_nodes lowered in
    let context, bindings, schedule, name =
      match code with
      | Postponed { comp; lowered; bindings; name } ->
          let proc =
            Backend.compile ~name ~opt_ctx_arrays:(Some (ctx_arrays prior_context)) bindings lowered
          in
          let from_prior_context =
            Set.diff (Assignments.context_nodes comp.asgns) comp.embedded_nodes
          in
          verify from_prior_context;
          link_compiled ~merge_buffer prior_context proc
      | Compiled { from_prior_context; proc; _ } ->
          verify from_prior_context;
          link_compiled ~merge_buffer prior_context proc
    in
    let schedule =
      Task.prepend schedule ~work:(fun () ->
          check_merge_buffer ~scheduled_node:(Option.map !merge_buffer ~f:snd)
            ~code_node:(expected_merge_node code))
    in
    { context; schedule; bindings; name; inputs; outputs }

  let link_batch ~merge_buffer (prior_context : context) (code_batch : code_batch) =
    let lowereds = get_lowereds code_batch in
    let verify from_prior_context =
      verify_prior_context ~ctx_arrays ~is_in_context ~prior_context ~from_prior_context
      @@ Array.filter_map lowereds ~f:(Option.map ~f:(fun opt -> opt.Low_level.traced_store))
    in
    let _opt_ctx_arrays, procs =
      match code_batch with
      | Postponed { comps; lowereds; bindings; names } ->
          let procs =
            Backend.compile_batch ~names
              ~opt_ctx_arrays:(Some (ctx_arrays prior_context))
              bindings lowereds
          in
          verify @@ from_prior_context_batch comps;
          procs
      | Compiled { from_prior_context; procs; _ } ->
          verify from_prior_context;
          procs
    in
    let code_nodes = expected_merge_nodes code_batch in
    Array.fold_mapi procs ~init:prior_context ~f:(fun i context -> function
      | Some proc ->
          let context, bindings, schedule, name = link_compiled ~merge_buffer context proc in
          let inputs, outputs = Low_level.input_and_output_nodes @@ Option.value_exn lowereds.(i) in
          let schedule =
            Task.prepend schedule ~work:(fun () ->
                check_merge_buffer ~scheduled_node:(Option.map !merge_buffer ~f:snd)
                  ~code_node:code_nodes.(i))
          in
          (context, Some { context; schedule; bindings; name; inputs; outputs })
      | None -> (context, None))

  let get_used_memory = Ndarray.get_used_memory
end

module Make_no_device_backend (Backend_impl : Backend_types.Lowered_no_device_backend) = struct
  module No_device = Lowered_no_device_backend (Backend_impl)

  module Multicore = struct
    module Device = Multicore_backend (No_device)
    include Device
    include Add_buffer_retrieval_and_syncing (Device)
  end

  module Sync = struct
    module Device = Sync_backend (No_device)
    include Device
    include Add_buffer_retrieval_and_syncing (Device)
  end
end

module Cc = Make_no_device_backend (Cc_backend)
module Gcc = Make_no_device_backend (Gcc_backend)

module Lowered_backend (Device : Backend_types.Lowered_backend) : Backend_types.Backend = struct
  include Device
  include Add_buffer_retrieval_and_syncing (Device)

  type nonrec code = {
    from_prior_context : Set.M(Tnode).t;
    lowered : Low_level.optimized;
    code : code;
    expected_merge_node : Tnode.t option;
  }
  [@@deriving sexp_of]

  type nonrec code_batch = {
    from_prior_context : Set.M(Tnode).t;
    lowereds : Low_level.optimized option array;
    code_batch : code_batch;
    expected_merge_nodes : Tnode.t option array;
  }
  [@@deriving sexp_of]

  type nonrec routine = context routine [@@deriving sexp_of]

  let compile ?shared:_ ?name bindings comp : code =
    let name, lowered = lower_assignments ?name bindings comp.Assignments.asgns in
    let code = compile ~name bindings lowered in
    let from_prior_context = Set.diff (Assignments.context_nodes comp.asgns) comp.embedded_nodes in
    { from_prior_context; lowered; code; expected_merge_node = lowered.Low_level.merge_node }

  let compile_batch ?shared:_ ?names ?occupancy bindings comps =
    let names, lowereds =
      lower_batch_assignments ?names ?occupancy bindings
      @@ Array.map comps ~f:(fun c -> c.Assignments.asgns)
    in
    let code_batch = compile_batch ~names bindings lowereds in
    let from_prior_context =
      from_prior_context_batch
      @@ Array.mapi lowereds ~f:(fun i -> Option.map ~f:(fun _ -> comps.(i)))
    in
    {
      from_prior_context;
      lowereds;
      code_batch;
      expected_merge_nodes =
        Array.map lowereds ~f:(fun lowered ->
            Option.(join @@ map lowered ~f:(fun optim -> optim.Low_level.merge_node)));
    }

  let link context (code : code) =
    verify_prior_context ~ctx_arrays ~is_in_context ~prior_context:context
      ~from_prior_context:code.from_prior_context [| code.lowered.traced_store |];
    let inputs, outputs = Low_level.input_and_output_nodes code.lowered in
    let context, bindings, schedule = link context code.code in
    let schedule =
      Task.prepend schedule ~work:(fun () ->
          check_merge_buffer
            ~scheduled_node:(scheduled_merge_node @@ get_ctx_stream context)
            ~code_node:code.expected_merge_node)
    in
    { context; schedule; bindings; name; inputs; outputs }

  let link_batch context code_batch =
    verify_prior_context ~ctx_arrays ~is_in_context ~prior_context:context
      ~from_prior_context:code_batch.from_prior_context
    @@ Array.filter_map code_batch.lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store));
    let context, bindings, schedules = link_batch context code_batch.code_batch in
    ( context,
      Array.mapi schedules ~f:(fun i ->
          Option.map ~f:(fun schedule ->
              let expected_merge_node = code_batch.expected_merge_nodes.(i) in
              let inputs, outputs =
                Low_level.input_and_output_nodes @@ Option.value_exn code_batch.lowereds.(i)
              in
              let schedule =
                Task.prepend schedule ~work:(fun () ->
                    check_merge_buffer
                      ~scheduled_node:(scheduled_merge_node @@ get_ctx_stream context)
                      ~code_node:expected_merge_node)
              in
              { context; schedule; bindings; name; inputs; outputs })) )
end

module Cuda_backend : Backend_types.Backend = Lowered_backend ((
  Cuda_backend : Backend_types.Lowered_backend))

let reinitialize (module Backend : Backend_types.Backend) config =
  if not @@ Backend.is_initialized () then Backend.initialize config
  else (
    Stdlib.Gc.full_major ();
    Backend.initialize config)

let fresh_backend ?backend_name ?(config = Only_devices_parallel) () =
  let backend =
    match
      Option.value_or_thunk backend_name ~default:(fun () ->
          Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
      |> String.lowercase
    with
    | "cc" -> (module Cc.Multicore : Backend_types.Backend)
    | "gccjit" -> (module Gcc.Multicore : Backend_types.Backend)
    | "sync_cc" -> (module Cc.Sync : Backend_types.Backend)
    | "sync_gccjit" -> (module Gcc.Sync : Backend_types.Backend)
    | "cuda" -> (module Cuda_backend : Backend_types.Backend)
    | backend -> invalid_arg [%string "Backends.fresh_backend: unknown backend %{backend}"]
  in
  reinitialize backend config;
  backend
