open Base
open Ir
module Tn = Tnode
module Schedulers = Schedulers
open Backend_intf
open Backend_impl

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

let check_merge_buffer stream ~code_node =
  let name = function Some tn -> Tnode.debug_name tn | None -> "none" in
  match (stream.updating_for_merge_buffer, code_node) with
  | _, None -> ()
  | Some (actual, _), Some expected when Tnode.equal actual expected -> ()
  | _ ->
      raise
      @@ Utils.User_error
           ("Merge buffer mismatch, on stream: "
           ^ name (Option.map ~f:fst stream.updating_for_merge_buffer)
           ^ ", expected by code: " ^ name code_node)

module Add_buffer_retrieval_and_syncing (Backend : No_buffer_retrieval_or_syncing) = struct
  let wait_for_all ctx streams tn =
    let s = ctx.stream in
    Hashtbl.update_and_return streams tn
      ~f:
        (Fn.compose (List.filter ~f:(fun (_, e) -> not (Backend.is_done e)))
        @@ Option.value ~default:[])
    |> List.iter ~f:(fun (work_stream, e) ->
           if not (equal_stream work_stream s) then Backend.will_wait_for ctx e)

  let wait_for_ready ~dst ~src tn =
    let s = src.stream in
    let d = dst.stream in
    (* TODO: maybe it's worthwhile to clean up s.updating_for every now and then. *)
    Hashtbl.find s.updating_for tn
    |> Option.iter ~f:(fun upd_e ->
           if not (equal_stream s d || Backend.is_done upd_e) then Backend.will_wait_for dst upd_e)

  let%track2_sexp to_host (ctx : Backend.context) (tn : Tn.t) =
    match (tn, Map.find ctx.ctx_arrays tn) with
    | { Tn.array = Some hosted; _ }, Some src ->
        if Tn.potentially_cross_stream tn then
          wait_for_all ctx ctx.stream.device.shared_writer_streams tn;
        [%log "copying", Tn.debug_name tn, "at", (src : Backend.buffer_ptr), "to host"];
        (* Stdio.printf "copying: %s to_host\n" (Tn.debug_name tn); *)
        Backend.to_host ~src_ptr:src ~src:ctx hosted;
        let s = ctx.stream in
        let e = Backend.all_work s in
        Hashtbl.update s.device.host_writing_streams tn ~f:(fun l ->
            (s, e) :: Option.value ~default:[] l);
        true
    | _ -> false

  let update_writer_event ?e ?from ctx tn =
    let s = ctx.stream in
    let e = Option.value_or_thunk e ~default:(fun () -> Backend.all_work s) in
    let f l = (s, e) :: Option.value ~default:[] l in
    (match (from, tn) with
    | None, _ -> ()
    | Some `Host, Assignments.(Node tn | Merge_buffer tn) ->
        Hashtbl.update s.device.host_reading_streams tn ~f
    | Some (`Src src), (Assignments.Node tn | Assignments.Merge_buffer tn) ->
        Hashtbl.update src.reader_streams tn ~f);
    (* Wait for writing to finish before reading. *)
    (match (from, tn) with
    | _, Assignments.Merge_buffer _ | Some `Host, _ -> ()
    | _, Assignments.Node tn ->
        Tnode.prepare_read
          ~is_done:(fun () -> Backend.is_done e)
          ~sync:(fun () -> Backend.sync e)
          ~transfer:(fun () ->
            assert (to_host ctx tn);
            Backend.await s)
          tn);
    (* To be on the safe side, record events for potentially cross-stream nodes. *)
    match tn with
    | Node tn ->
        if Tn.potentially_cross_stream tn then
          Hashtbl.update s.device.shared_writer_streams tn ~f:(fun l ->
              (s, e) :: Option.value ~default:[] l)
        else Hashtbl.remove s.device.shared_writer_streams tn;
        Hashtbl.update s.updating_for tn ~f:(fun _ -> e)
    | Merge_buffer tn ->
        (* Note: the previous event does not need to be done! *)
        s.updating_for_merge_buffer <- Some (tn, Some e)

  let%track2_sexp from_host (ctx : Backend.context) tn =
    match (tn, Map.find ctx.ctx_arrays tn) with
    | { Tn.array = Some hosted; _ }, Some dst ->
        wait_for_all ctx ctx.stream.reader_streams tn;
        [%log "copying", Tn.debug_name tn, "to", (dst : Backend.buffer_ptr), "from host"];
        (* Stdio.printf "copying: %s from_host\n" (Tn.debug_name tn); *)
        Backend.from_host ~dst_ptr:dst ~dst:ctx hosted;
        update_writer_event ~from:`Host ctx @@ Node tn;
        Hash_set.add tn.host_read_by_devices ctx.stream.device.device_id;
        true
    | _ -> false

  let%diagn2_sexp device_to_device (tn : Tn.t) ~into_merge_buffer ~(dst : Backend.context)
      ~(src : Backend.context) =
    let ordinal_of ctx = ctx.stream.device.ordinal in
    let name_of ctx = Backend.(get_name ctx.stream) in
    let same_device = ordinal_of dst = ordinal_of src in
    if same_device && (Tn.known_shared_cross_streams tn || String.equal (name_of src) (name_of dst))
    then false
    else
      match Map.find src.ctx_arrays tn with
      | None -> false
      | Some s_arr -> (
          wait_for_ready ~dst ~src tn;
          match into_merge_buffer with
          | No -> (
              match Map.find dst.ctx_arrays tn with
              | None -> false
              | Some d_arr ->
                  Backend.(
                    device_to_device tn ~into_merge_buffer ~dst_ptr:(Some d_arr) ~dst ~src_ptr:s_arr
                      ~src);
                  update_writer_event ~from:(`Src src.stream) dst @@ Node tn;
                  [%log "copying", Tn.debug_name tn, "from", name_of src, "to", name_of dst];
                  true)
          | Copy ->
              Backend.(
                device_to_device tn ~into_merge_buffer ~dst_ptr:None ~dst ~src_ptr:s_arr ~src);
              update_writer_event ~from:(`Src src.stream) dst @@ Merge_buffer tn;
              [%log "copy into merge buffer", Tn.debug_name tn, "from", name_of src];
              true
          | Streaming_for task ->
              Backend.(
                device_to_device tn ~into_merge_buffer ~dst_ptr:None ~dst ~src_ptr:s_arr ~src);
              dst.stream.updating_for_merge_buffer <- Some (tn, None);
              let merge_task () = Task.run task in
              merge_task ();
              update_writer_event ~from:(`Src src.stream) dst @@ Merge_buffer tn;
              [%log "streaming into merge buffer", Tn.debug_name tn, "from", name_of src];
              true)

  type r = Backend.context routine [@@deriving sexp_of]

  let%track2_sexp sync_routine (r : r) : r =
    let s = r.context.stream in
    let hosted_inputs = Set.filter r.inputs ~f:(fun tn -> Tn.is_hosted_force tn 47) in
    let pre () =
      assert (Domain.is_main_domain ());
      if Utils.settings.automatic_host_transfers then
        Set.iter hosted_inputs ~f:(fun tn ->
            if not (Hash_set.mem tn.host_read_by_devices s.device.device_id) then
              assert (from_host r.context tn));
      Set.iter r.inputs ~f:(fun tn ->
          if Tn.potentially_cross_stream tn then
            Option.iter (Hashtbl.find s.device.shared_writer_streams tn) ~f:(fun data ->
                let data = List.filter data ~f:(fun (_, e) -> not (Backend.is_done e)) in
                Hashtbl.set s.device.shared_writer_streams ~key:tn ~data;
                List.iter data ~f:(fun (work_stream, e) ->
                    if not (equal_stream work_stream s) then Backend.will_wait_for r.context e))
          else Hashtbl.remove s.device.shared_writer_streams tn)
      (* Since merge buffers are always per-stream, no need to check r.merge_buffer_input. *)
    in
    let post () =
      let e = Backend.all_work s in
      Set.iter r.outputs ~f:(fun tn -> update_writer_event ~e r.context @@ Node tn);
      Set.iter hosted_inputs ~f:(fun tn ->
          Tn.prepare_write
            ~is_done:(fun () -> Backend.is_done e)
            ~sync:(fun () -> Backend.sync e)
            tn)
    in
    { r with schedule = Task.(prepend ~work:pre @@ append ~work:post r.schedule) }

  let sync_device device =
    Utils.weak_iter device.streams ~f:Backend.await;
    Hashtbl.clear device.host_writing_streams;
    Hashtbl.clear device.host_reading_streams;
    Hashtbl.clear device.shared_writer_streams;
    Utils.weak_iter device.streams ~f:(fun s ->
        Hashtbl.clear s.reader_streams;
        s.updating_for_merge_buffer <- None;
        Hashtbl.clear s.updating_for)
end

let%track6_sexp lower_assignments ?name bindings asgns =
  let name : string =
    Option.value_or_thunk name ~default:(fun () -> Assignments.get_name_exn asgns)
  in
  let unoptim_ll_source = Utils.output_to_build_file ~fname:(name ^ "-unoptimized.ll") in
  let ll_source = Utils.output_to_build_file ~fname:(name ^ ".ll") in
  let cd_source = Utils.output_to_build_file ~fname:(name ^ ".cd") in
  ( name,
    Assignments.lower ~unoptim_ll_source ~ll_source ~cd_source ~name
      (Indexing.bound_symbols bindings) asgns )

let lower_batch_assignments ?names ?occupancy bindings asgns_l =
  let names =
    Option.value_or_thunk names ~default:(fun () ->
        Array.map asgns_l ~f:(fun asgns -> Assignments.get_name_exn asgns))
  in
  let prefix_name = String.(strip ~drop:(equal_char '_') @@ common_prefix @@ Array.to_list names) in
  let unoptim_ll_source = Utils.output_to_build_file ~fname:(prefix_name ^ "-unoptimized.ll") in
  let ll_source = Utils.output_to_build_file ~fname:(prefix_name ^ ".ll") in
  let cd_source = Utils.output_to_build_file ~fname:(prefix_name ^ ".cd") in
  let bound = Indexing.bound_symbols bindings in
  let occupancy = Option.value occupancy ~default:(fun ~name:_ ~src_n:_ -> true) in
  Array.unzip
  @@ Array.mapi names ~f:(fun src_n name ->
         let asgns = asgns_l.(src_n) in
         if occupancy ~name ~src_n then
           ( Some name,
             Some (Assignments.lower ~unoptim_ll_source ~ll_source ~cd_source ~name bound asgns) )
         else (None, None))

let%debug3_sexp verify_prior_context ~use_host_memory ~ctx_arrays ~from_prior_context : unit =
  Set.iter from_prior_context ~f:(fun tn ->
      if
        Tn.is_in_context_force ~use_host_memory tn 42
        && not (Option.is_some @@ Map.find ctx_arrays tn)
      then raise @@ Utils.User_error ("The linked context lacks node " ^ Tnode.debug_name tn))

let%debug3_sexp from_prior_context_batch ~use_host_memory (comps : Assignments.comp option array) :
    Tn.t_set =
  Array.filter_map comps ~f:(fun comp ->
      Option.map comp ~f:(fun comp ->
          Set.diff
            (Assignments.context_nodes ~use_host_memory comp.Assignments.asgns)
            comp.embedded_nodes))
  |> Array.fold ~init:(Set.empty (module Tnode)) ~f:Set.union

(** Adds a scheduler and brings a lowered no-device backend on par with lowered device backends. *)
module Add_device
    (Add_scheduler : functor
      (Impl : For_add_scheduler)
      -> With_scheduler with type buffer_ptr = Impl.buffer_ptr)
    (Backend : Lowered_no_device_backend)
    (Config : sig
      val config : config
    end) : Lowered_backend = struct
  include Backend

  type code = { lowered : Low_level.optimized; proc : Backend.procedure } [@@deriving sexp_of]

  type code_batch = {
    lowereds : Low_level.optimized option array;
    procs : Backend.procedure option array;
  }
  [@@deriving sexp_of]

  let compile ~name bindings lowered : code =
    let proc = compile ~name bindings lowered in
    { lowered; proc }

  let compile_batch ~names bindings lowereds : code_batch =
    let procs = compile_batch ~names bindings lowereds in
    { lowereds; procs }

  include Add_scheduler (struct
    include Backend
    include Config
  end)

  let link context (code : code) ctx_arrays =
    let runner_label = get_name context.stream in
    let merge_buffer = context.stream.merge_buffer in
    let bindings, to_schedule = link_compiled ~merge_buffer ~runner_label ctx_arrays code.proc in
    let schedule =
      Task.enschedule ~schedule_task ~get_stream_name:get_name context.stream to_schedule
    in
    (bindings, schedule)

  let link_batch context (code_batch : code_batch) ctx_arrays =
    let runner_label = get_name context.stream in
    let merge_buffer = context.stream.merge_buffer in
    let bindings, schedules =
      Array.fold_mapi code_batch.procs ~init:None ~f:(fun i bindings -> function
        | Some proc ->
            let ctx_arrays = Option.value_exn ctx_arrays.(i) in
            let bindings', to_schedule =
              link_compiled ~merge_buffer ~runner_label ctx_arrays proc
            in
            Option.iter bindings ~f:(fun bindings -> assert (phys_equal bindings bindings'));
            let schedule =
              Task.enschedule ~schedule_task ~get_stream_name:get_name context.stream to_schedule
            in
            (Some bindings', Some schedule)
        | None -> (bindings, None))
    in
    (Option.value_exn ~here:[%here] bindings, schedules)

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
    let s = dst.stream in
    let size_in_bytes = tn.Tnode.size_in_bytes in
    let work =
      (* TODO: log the operation if [Utils.settings.with_log_level > 1]. *)
      match (into_merge_buffer, dst_ptr) with
      | No, None -> invalid_arg "Multicore_scheduler.device_to_device: missing dst_ptr"
      | No, Some dst_ptr -> fun () -> buffer_to_buffer ~dst:dst_ptr ~src:src_ptr ~size_in_bytes
      | Streaming_for _, _ -> fun () -> s.merge_buffer := Some { ptr = src_ptr; size_in_bytes }
      | Copy, _ ->
          fun () ->
            let allocated_capacity =
              match s.allocated_buffer with None -> 0 | Some buf -> buf.size_in_bytes
            in
            if allocated_capacity < size_in_bytes then
              s.allocated_buffer <-
                Some (alloc_buffer ?old_buffer:s.allocated_buffer ~size_in_bytes dst.stream);
            let merge_ptr = (Option.value_exn s.allocated_buffer).ptr in
            s.merge_buffer := s.allocated_buffer;
            buffer_to_buffer ~dst:merge_ptr ~src:src_ptr ~size_in_bytes
    in
    let description =
      "device_to_device " ^ Tnode.debug_name tn ^ " dst " ^ get_name s ^ " src "
      ^ get_name src.stream
    in
    schedule_task s (Task.Task { context_lifetime = (src, dst); description; work })
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

  let%debug3_sexp compile ?name bindings (comp : Assignments.comp) : code =
    let (name : string), (lowered : Low_level.optimized) =
      lower_assignments ?name bindings comp.Assignments.asgns
    in
    let code : Device.code = compile ~name bindings lowered in
    let from_prior_context : Tn.t_set =
      Set.diff (Assignments.context_nodes ~use_host_memory comp.asgns) comp.embedded_nodes
    in
    { from_prior_context; name; lowered; code; expected_merge_node = lowered.Low_level.merge_node }

  let%debug3_sexp compile_batch ?names ?occupancy bindings (comps : Assignments.comp array) :
      code_batch =
    let names, lowereds =
      lower_batch_assignments ?names ?occupancy bindings
      @@ Array.map comps ~f:(fun c -> c.Assignments.asgns)
    in
    let code_batch = compile_batch ~names bindings lowereds in
    let from_prior_context =
      from_prior_context_batch ~use_host_memory
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

  let%track3_sexp alloc_if_needed parent_context ~key ~data:node ctx_arrays =
    if Tnode.is_in_context_force ~use_host_memory key 43 && not (Map.mem ctx_arrays key) then (
      let stream = parent_context.stream in
      [%log Tn.debug_name key];
      [%log (key : Tnode.t)];
      let default () =
        let dst_ptr =
          alloc_zero_init_array (key.prec) ~dims:(key.dims) stream
        in
        (if Utils.settings.automatic_host_transfers && Tn.known_constant key then
           match key.array with
           | Some hosted ->
               Device.from_host ~dst_ptr ~dst:parent_context hosted;
               Hash_set.add key.host_read_by_devices stream.device.device_id
           | _ -> ());
        dst_ptr
      in
      let add_new () = Map.add_exn ctx_arrays ~key ~data:(default ()) in
      let device = stream.device in
      if node.Low_level.read_only then (
        if Tn.known_non_cross_stream key then add_new ()
        else
          let data =
            match use_host_memory with
            | None -> Hashtbl.find_or_add device.cross_stream_candidates key ~default
            | Some get_buffer_ptr ->
                if
                  (not (Hashtbl.mem device.cross_stream_candidates key))
                  && Tn.known_shared_cross_streams key && Tn.is_hosted_force key 44
                then
                  Hashtbl.update_and_return device.cross_stream_candidates key ~f:(fun _ ->
                      get_buffer_ptr ~size_in_bytes:(key.size_in_bytes)
                      @@ Ndarray.get_voidptr_not_managed
                      @@ Option.value_exn ~here:[%here]
                      @@ key.array)
                else Hashtbl.find_or_add device.cross_stream_candidates key ~default
          in
          if Hashtbl.mem device.cross_stream_candidates key then
            Tn.update_memory_sharing key Tn.Shared_cross_streams 39;
          Map.add_exn ctx_arrays ~key ~data)
      else if Tn.known_shared_cross_streams key then (
        if Hashtbl.mem device.owner_stream key then (
          if not (equal_stream stream (Hashtbl.find_exn device.owner_stream key)) then
            raise
            @@ Utils.User_error
                 ("Backends.alloc_if_needed: node " ^ Tn.debug_name key
                ^ " assumed to be cross-stream-shared but then written to on multiple devices"))
        else Hashtbl.add_exn device.owner_stream ~key ~data:stream;
        let data = Hashtbl.find_exn device.cross_stream_candidates key in
        Map.add_exn ctx_arrays ~key ~data)
      else (
        Tn.update_memory_sharing key Tn.Per_stream 41;
        Hashtbl.remove device.cross_stream_candidates key;
        add_new ()))
    else ctx_arrays

  let%debug3_sexp link context (code : code) =
    verify_prior_context ~use_host_memory ~ctx_arrays:context.ctx_arrays
      ~from_prior_context:code.from_prior_context;
    let (inputs, outputs), merge_buffer_input = Low_level.input_and_output_nodes code.lowered in
    let ctx_arrays =
      Hashtbl.fold code.lowered.traced_store ~init:context.ctx_arrays ~f:(alloc_if_needed context)
    in
    let bindings, schedule = link context code.code ctx_arrays in
    let context = make_child ~ctx_arrays context in
    let schedule =
      Task.prepend schedule ~work:(fun () ->
          check_merge_buffer context.stream ~code_node:code.expected_merge_node)
    in
    sync_routine
      { context; schedule; bindings; name = code.name; inputs; merge_buffer_input; outputs }

  let%debug3_sexp link_batch context code_batch =
    verify_prior_context ~use_host_memory ~ctx_arrays:context.ctx_arrays
      ~from_prior_context:code_batch.from_prior_context;
    let ctx_arrays =
      Array.map code_batch.lowereds
        ~f:
          (Option.map ~f:(fun l ->
               Hashtbl.fold l.Low_level.traced_store ~init:context.ctx_arrays
                 ~f:(alloc_if_needed context)))
    in
    let bindings, schedules = link_batch context code_batch.code_batch ctx_arrays in
    Array.fold_mapi schedules ~init:context ~f:(fun i context -> function
      | None -> (context, None)
      | Some schedule ->
          let ctx_arrays = Option.value_exn ctx_arrays.(i) in
          let context = make_child ~ctx_arrays context in
          let expected_merge_node = code_batch.expected_merge_nodes.(i) in
          let (inputs, outputs), merge_buffer_input =
            Low_level.input_and_output_nodes @@ Option.value_exn code_batch.lowereds.(i)
          in
          let schedule =
            Task.prepend schedule ~work:(fun () ->
                check_merge_buffer context.stream ~code_node:expected_merge_node)
          in
          let r =
            sync_routine { context; schedule; bindings; name; inputs; merge_buffer_input; outputs }
          in
          (context, Some r))
end

module Make_device_backend_from_lowered
    (Add_scheduler : functor
      (Impl : For_add_scheduler)
      -> With_scheduler with type buffer_ptr = Impl.buffer_ptr)
    (Backend_impl : Lowered_no_device_backend)
    (Config : sig
      val config : config
    end) =
struct
  module Lowered_device = Add_device (Add_scheduler) (Backend_impl) (Config)
  module Backend_device = Raise_backend (Lowered_device)
  include Backend_device
end

let finalize (type buffer_ptr dev runner event)
    (module Backend : Backend
      with type buffer_ptr = buffer_ptr
       and type dev = dev
       and type runner = runner
       and type event = event) (ctx : Backend.context) : unit =
  Option.iter Backend.free_buffer ~f:(fun mem_free ->
      if Atomic.compare_and_set ctx.finalized false true then (
        Backend.await ctx.stream;
        Map.iteri ctx.ctx_arrays ~f:(fun ~key ~data ->
            if
              (not (Option.exists ctx.parent ~f:(fun pc -> Map.mem pc.ctx_arrays key)))
              && not (Hashtbl.mem ctx.stream.device.cross_stream_candidates key)
            then mem_free ctx.stream data)))

let%track5_sexp fresh_backend ?backend_name ?(config = For_parallel_copying) () =
  Stdlib.Gc.full_major ();
  (* TODO: is running again needed to give time to weak arrays to become empty? *)
  Stdlib.Gc.full_major ();
  (* Note: we invoke functors from within fresh_backend to fully isolate backends from distinct
     calls to fresh_backend. *)
  let module Config = struct
    let config = config
  end in
  match
    Option.value_or_thunk backend_name ~default:(fun () ->
        Utils.get_global_arg ~arg_name:"backend" ~default:"multicore_cc")
    |> String.lowercase
  with
  | "multicore_cc" ->
      (module Make_device_backend_from_lowered (Schedulers.Multicore) (Cc_backend) (Config)
      : Backend)
  | "gccjit" ->
      (module Make_device_backend_from_lowered (Schedulers.Multicore) (Gcc_backend_impl) (Config)
      : Backend)
  | "sync_cc" ->
      (module Make_device_backend_from_lowered (Schedulers.Sync) (Cc_backend) (Config) : Backend)
  | "sync_gccjit" ->
      (module Make_device_backend_from_lowered (Schedulers.Sync) (Gcc_backend_impl) (Config)
      : Backend)
  | "cuda" ->
      (module Raise_backend ((Cuda_backend_impl.Fresh (Config) : Lowered_backend)) : Backend)
  | "metal" ->
      (module Raise_backend ((Metal_backend_impl.Fresh (Config) : Lowered_backend)) : Backend)
  | backend -> invalid_arg [%string "Backends.fresh_backend: unknown backend %{backend}"]
