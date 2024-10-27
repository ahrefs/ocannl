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

module Multicore_backend (Backend : No_device_backend) = struct
  include Backend
  module Domain = Domain [@warning "-3"]

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

  module Alloc_buffer = struct
    include Backend

    let alloc_buffer ?old_buffer ~size_in_bytes _stream = alloc_buffer ?old_buffer ~size_in_bytes ()
    let alloc_zero_init_array prec ~dims _stream = alloc_zero_init_array prec ~dims ()
  end

  include Device (Device_types (Device_config)) (Alloc_buffer)
  open Device_config

  (** TODO: Blocks till the event completes, if it's not done already. *)
  let sync Not_implemented_yet = ()

  (** TODO: Whether the event completed. *)
  let is_done Not_implemented_yet = true

  (** TODO: Schedules waiting for the given event on the context's stream. *)
  let will_wait_for _ctx Not_implemented_yet = ()

  let get_used_memory _device = get_used_memory ()

  type nonrec code = code [@@deriving sexp_of]
  type nonrec code_batch = code_batch [@@deriving sexp_of]

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
    if not d.keep_spinning then invalid_arg "Multicore_backend: stream not available";
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

  let initialize = initialize
  let is_initialized = is_initialized
  let compile = compile
  let compile_batch = compile_batch

  let link (context : context) code =
    let routine =
      link ~merge_buffer:context.stream.merge_buffer ~runner_label:(get_name context.stream)
        context.ctx_arrays code
    in
    let context = make_child ~ctx_arrays:routine.context context in
    {
      routine with
      context;
      schedule =
        Task.enschedule ~schedule_task ~get_stream_name:get_name context.stream routine.schedule;
    }

  let link_batch (context : context) code_batch =
    let ctx_arrays, routines =
      link_batch ~merge_buffer:context.stream.merge_buffer ~runner_label:(get_name context.stream)
        context.ctx_arrays code_batch
    in
    ( make_child ~ctx_arrays context,
      Array.map routines
        ~f:
          (Option.map ~f:(fun task ->
               {
                 task with
                 context = make_child ~ctx_arrays:task.context context;
                 schedule =
                   Task.enschedule ~schedule_task ~get_stream_name:get_name context.stream
                     task.schedule;
               })) )

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
      invalid_arg [%string "Multicore_backend.get_device %{ordinal#Int}: only device 0 exists"];
    device

  let latest_stream_id = ref (-1)

  let new_stream _device =
    assert (Domain.is_main_domain ());
    Int.incr latest_stream_id;
    let stream = spinup_stream ~stream_id:!latest_stream_id in
    Stdlib.Gc.finalise cleanup_stream stream;
    stream

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
      | No, None -> invalid_arg "Multicore_backend.device_to_device: missing dst_ptr"
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

(** For debugging, allow [Sync_backend(...).suggested_num_streams] calls to return >1 numbers. *)
let sync_suggested_num_streams = ref 1

(** A minimalisitc wrapper creating backends where all calls run synchronously on the main thread.
    There is only one device, but an arbitrary number of streams. *)
module Sync_backend (Backend : No_device_backend) = struct
  include Backend

  module Device_config = struct
    include (
      Backend : Buffer with type buffer_ptr = Backend.buffer_ptr and type buffer = Backend.buffer)

    type dev = CPU [@@deriving sexp_of]
    type runner = unit [@@deriving sexp_of]
    type event = unit [@@deriving sexp_of]

    let name = "sync_" ^ Backend.name
  end

  module Alloc_buffer = struct
    include Backend

    let alloc_buffer ?old_buffer ~size_in_bytes _stream = alloc_buffer ?old_buffer ~size_in_bytes ()
    let alloc_zero_init_array prec ~dims _stream = alloc_zero_init_array prec ~dims ()
  end

  include Device (Device_types (Device_config)) (Alloc_buffer)
  open Device_config

  let sync () = ()
  let is_done () = true
  let will_wait_for _context () = ()

  let alloc_buffer ?old_buffer ~size_in_bytes _stream =
    Backend.alloc_buffer ?old_buffer ~size_in_bytes ()

  let device : device = make_device CPU ~ordinal:0

  let get_device ~ordinal =
    if ordinal <> 0 then
      invalid_arg @@ "Sync_backend.get_device: there is only one device, but ordinal="
      ^ Int.to_string ordinal;
    device

  let num_devices () = 1
  let suggested_num_streams _ = !sync_suggested_num_streams
  let get_used_memory _ = Backend.get_used_memory ()
  let latest_stram_id = ref (-1)

  let new_stream device =
    Int.incr latest_stram_id;
    make_stream device () ~stream_id:!latest_stram_id

  type code = Backend.code [@@deriving sexp_of]
  type code_batch = Backend.code_batch [@@deriving sexp_of]

  let all_work _stream = ()
  let is_idle _stream = true
  let name = "sync_" ^ Backend.name
  let await _stream = ()
  (* let global_run_no = ref 0 *)

  let initialize = Backend.initialize
  let is_initialized = Backend.is_initialized
  let compile = Backend.compile
  let compile_batch = Backend.compile_batch

  let link context code =
    let task =
      Backend.link ~merge_buffer:context.stream.merge_buffer ~runner_label:(get_name context.stream)
        context.ctx_arrays code
    in
    { task with context = make_child ~ctx_arrays:task.context context }

  let link_batch context code_batch =
    let ctx_arrays, routines =
      Backend.link_batch ~merge_buffer:context.stream.merge_buffer
        ~runner_label:(get_name context.stream) context.ctx_arrays code_batch
    in
    ( make_child ~ctx_arrays context,
      Array.map routines
        ~f:
          (Option.map ~f:(fun task ->
               { task with context = make_child ~ctx_arrays:task.context context })) )

  let get_name stream = [%string "%{name}:0:%{stream.stream_id#Int}"]
  let from_host ~dst_ptr ~dst:_ hosted = host_to_buffer hosted ~dst:dst_ptr
  let to_host ~src_ptr ~src:_ hosted = buffer_to_host hosted ~src:src_ptr

  let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src:_ =
    let dev = dst.stream in
    (* TODO: log the operation if [Utils.settings.with_log_level > 0]. *)
    let size_in_bytes = Tnode.size_in_bytes tn in
    match (into_merge_buffer, dst_ptr) with
    | No, None -> invalid_arg "Sync_backend.device_to_device: missing dst_ptr"
    | No, Some dst_ptr -> buffer_to_buffer ~dst:dst_ptr ~src:src_ptr ~size_in_bytes
    | Streaming, _ -> dev.merge_buffer := Some (src_ptr, tn)
    | Copy, _ ->
        let allocated_capacity =
          match dev.allocated_buffer with None -> 0 | Some buf -> buf.size_in_bytes
        in
        if allocated_capacity < size_in_bytes then
          dev.allocated_buffer <-
            Some (Backend.alloc_buffer ?old_buffer:dev.allocated_buffer ~size_in_bytes ());
        let merge_ptr = (Option.value_exn dev.allocated_buffer).ptr in
        dev.merge_buffer := Some (merge_ptr, tn);
        buffer_to_buffer ~dst:merge_ptr ~src:src_ptr ~size_in_bytes
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

module Lowered_no_device_backend (Backend : Lowered_no_device_backend) :
  No_device_backend with type buffer_ptr = Backend.buffer_ptr = struct
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

  let link ~merge_buffer ~runner_label ctx_arrays (code : code) =
    let lowered = get_lowered code in
    let verify from_prior_context =
      verify_prior_context ~is_in_context ~ctx_arrays ~from_prior_context [| lowered.traced_store |]
    in
    let inputs, outputs = Low_level.input_and_output_nodes lowered in
    let ctx_arrays, bindings, schedule, name =
      match code with
      | Postponed { comp; lowered; bindings; name } ->
          let proc = Backend.compile ~name ~opt_ctx_arrays:(Some ctx_arrays) bindings lowered in
          let from_prior_context =
            Set.diff (Assignments.context_nodes comp.asgns) comp.embedded_nodes
          in
          verify from_prior_context;
          link_compiled ~merge_buffer ~runner_label ctx_arrays proc
      | Compiled { from_prior_context; proc; _ } ->
          verify from_prior_context;
          link_compiled ~merge_buffer ~runner_label ctx_arrays proc
    in
    let schedule =
      Task.prepend schedule ~work:(fun () ->
          check_merge_buffer ~scheduled_node:(Option.map !merge_buffer ~f:snd)
            ~code_node:(expected_merge_node code))
    in
    { context = ctx_arrays; schedule; bindings; name; inputs; outputs }

  let link_batch ~merge_buffer ~runner_label ctx_arrays (code_batch : code_batch) =
    let lowereds = get_lowereds code_batch in
    let verify from_prior_context =
      verify_prior_context ~is_in_context ~ctx_arrays ~from_prior_context
      @@ Array.filter_map lowereds ~f:(Option.map ~f:(fun opt -> opt.Low_level.traced_store))
    in
    let _opt_ctx_arrays, procs =
      match code_batch with
      | Postponed { comps; lowereds; bindings; names } ->
          let procs =
            Backend.compile_batch ~names ~opt_ctx_arrays:(Some ctx_arrays) bindings lowereds
          in
          verify @@ from_prior_context_batch comps;
          procs
      | Compiled { from_prior_context; procs; _ } ->
          verify from_prior_context;
          procs
    in
    let code_nodes = expected_merge_nodes code_batch in
    Array.fold_mapi procs ~init:ctx_arrays ~f:(fun i ctx_arrays -> function
      | Some proc ->
          let ctx_arrays, bindings, schedule, name =
            link_compiled ~merge_buffer ~runner_label ctx_arrays proc
          in
          let inputs, outputs = Low_level.input_and_output_nodes @@ Option.value_exn lowereds.(i) in
          let schedule =
            Task.prepend schedule ~work:(fun () ->
                check_merge_buffer ~scheduled_node:(Option.map !merge_buffer ~f:snd)
                  ~code_node:code_nodes.(i))
          in
          (ctx_arrays, Some { context = ctx_arrays; schedule; bindings; name; inputs; outputs })
      | None -> (ctx_arrays, None))

  let get_used_memory = Ndarray.get_used_memory
end

module Make_no_device_backend (Backend_impl : Lowered_no_device_backend) = struct
  module No_device = Lowered_no_device_backend (Backend_impl)

  module Multicore = struct
    module Backend_device = Multicore_backend (No_device)

    (* include Add_buffer_retrieval_and_syncing (Backend_device) *)
    module Syncing = Add_buffer_retrieval_and_syncing (Backend_device)
    include Backend_device
    include Syncing
  end

  module Sync = struct
    module Backend_device = Sync_backend (No_device)
    include Backend_device
    include Add_buffer_retrieval_and_syncing (Backend_device)
  end
end

module Cc = Make_no_device_backend (Cc_backend)
module Gcc = Make_no_device_backend (Gcc_backend)

module Lowered_backend (Device : Lowered_backend) : Backend = struct
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
    verify_prior_context ~is_in_context ~ctx_arrays:context.ctx_arrays
      ~from_prior_context:code.from_prior_context [| code.lowered.traced_store |];
    let inputs, outputs = Low_level.input_and_output_nodes code.lowered in
    let context, bindings, schedule = link context code.code in
    let schedule =
      Task.prepend schedule ~work:(fun () ->
          check_merge_buffer
            ~scheduled_node:(scheduled_merge_node context.stream)
            ~code_node:code.expected_merge_node)
    in
    { context; schedule; bindings; name; inputs; outputs }

  let link_batch context code_batch =
    verify_prior_context ~is_in_context ~ctx_arrays:context.ctx_arrays
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
                      ~scheduled_node:(scheduled_merge_node context.stream)
                      ~code_node:expected_merge_node)
              in
              { context; schedule; bindings; name; inputs; outputs })) )
end

module Cuda_backend : Backend = Lowered_backend ((Cuda_backend : Lowered_backend))

let reinitialize (module Backend : Backend) config =
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
    | "cc" -> (module Cc.Multicore : Backend)
    | "gccjit" -> (module Gcc.Multicore : Backend)
    | "sync_cc" -> (module Cc.Sync : Backend)
    | "sync_gccjit" -> (module Gcc.Sync : Backend)
    | "cuda" -> (module Cuda_backend : Backend)
    | backend -> invalid_arg [%string "Backends.fresh_backend: unknown backend %{backend}"]
  in
  reinitialize backend config;
  backend
