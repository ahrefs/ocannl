open Base
open Ir
module Tn = Tnode
module Schedulers = Schedulers
open Backend_intf
open Backend_impl

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_BACKENDS=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_BACKENDS"]

(* Dynamic backstop for merge-buffer verification: runs as the first work of a consumer's schedule
   and checks the node most recently scheduled into the stream's merge buffer. The primary check is
   now the static [check_merge_buffer_static] performed at link time (gh-ocannl-288); this remains
   as a defensive backstop for transfers that are scheduled without a downstream link. *)
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

(* Static counterpart of [check_merge_buffer]: verifies at link time -- before any schedule runs --
   that the merge-buffer node statically recorded on the linked [context] (by a [device_to_device]
   transfer routine, see {!Add_buffer_retrieval_and_syncing.device_to_device}) matches the node the
   linked [code] expects. This is the "static verification in the right direction" of
   gh-ocannl-288: the transfer routine's context chains naturally into the consumer's link. *)
let check_merge_buffer_static ~merge_buffer_node ~code_node =
  let name = function Some tn -> Tnode.debug_name tn | None -> "none" in
  match (merge_buffer_node, code_node) with
  | _, None -> ()
  | Some actual, Some expected when Tnode.equal actual expected -> ()
  | _ ->
      raise
      @@ Utils.User_error
           ("Merge buffer mismatch at link time: the linked context provides "
           ^ name merge_buffer_node ^ ", but the linked code expects " ^ name code_node)

module Add_buffer_retrieval_and_syncing (Backend : No_buffer_retrieval_or_syncing) = struct
  let wait_for_ready ~dst ~src tn =
    let s = src.stream in
    let d = dst.stream in
    (* TODO: maybe it's worthwhile to clean up s.updating_for every now and then. *)
    Hashtbl.find s.updating_for tn
    |> Option.iter ~f:(fun upd_e ->
        if not (equal_stream s d || Backend.is_done upd_e) then Backend.will_wait_for dst upd_e)

  let%track3_sexp to_host (ctx : Backend.context) (tn : Tn.t) (hosted : Ndarray.t) =
    match Map.find ctx.ctx_arrays tn with
    | Some src ->
        [%log "copying", Tn.debug_name tn, "at", (src : Backend.buffer_ptr), "to host"];
        (* No cross-stream writer synchronization needed: multi-streaming was removed
           (gh-ocannl-341). Only one stream exists per device, so there are no
           concurrent cross-stream writes to wait for before this device-to-host copy. *)
        Backend.to_host ~src_ptr:src ~src:ctx hosted;
        true
    | None -> false

  let update_writer_event ?e ctx tn =
    let s = ctx.stream in
    let e = Option.value_or_thunk e ~default:(fun () -> Backend.all_work s) in
    match tn with
    | Assignments.Node tn -> Hashtbl.update s.updating_for tn ~f:(fun _ -> e)
    | Assignments.Merge_buffer tn ->
        (* Note: the previous event does not need to be done! *)
        s.updating_for_merge_buffer <- Some (tn, Some e)

  let%track3_sexp from_host (ctx : Backend.context) (tn : Tn.t) (hosted : Ndarray.t) =
    match Map.find ctx.ctx_arrays tn with
    | Some dst ->
        (* No cross-stream reader synchronization needed: multi-streaming was removed
           (gh-ocannl-341). Only one stream exists per device, so there are no concurrent
           cross-stream readers to wait for before this host-to-device upload. *)
        [%log "copying", Tn.debug_name tn, "to", (dst : Backend.buffer_ptr), "from host"];
        Backend.from_host ~dst_ptr:dst ~dst:ctx hosted;
        update_writer_event ctx @@ Node tn;
        true
    | None -> false

  let%track3_sexp init_from_host (ctx : Backend.context) (tn : Tn.t) (hosted : Ndarray.t) =
    match Map.find ctx.ctx_arrays tn with
    | None ->
        let dims = Lazy.force tn.dims in
        (* Use alloc_array since we're immediately copying from host *)
        let dst =
          Backend.alloc_array
            ?mode:(Option.map tn.memory_mode ~f:fst)
            (Lazy.force tn.prec) ~dims ctx.stream
        in
        [%log "copying", Tn.debug_name tn, "to", (dst : Backend.buffer_ptr), "from host"];
        Backend.from_host ~dst_ptr:dst ~dst:ctx hosted;
        update_writer_event ctx @@ Node tn;
        { ctx with ctx_arrays = Map.add_exn ctx.ctx_arrays ~key:tn ~data:dst }
    | Some _ ->
        raise
        @@ Utils.User_error
             ("init_from_host: input context already contains tensor node " ^ Tn.debug_name tn
            ^ ", for stream " ^ Backend.get_name ctx.stream)

  (* [device_to_device] builds a transfer routine instead of scheduling the copy directly. The
     caller schedules it (via [Task.run r.schedule]) or links a consumer against [r.context]. For
     the [Copy] case, [r.context]'s [merge_buffer_node] records the produced node statically, so
     that [link] can verify it against a consumer's [expected_merge_node] at link time -- the
     "static verification in the right direction" of gh-ocannl-288. *)
  let%track3_sexp device_to_device (tn : Tn.t) ~into_merge_buffer ~(dst : Backend.context)
      ~(src : Backend.context) : Backend.context routine option =
    match Map.find src.ctx_arrays tn with
    | None -> None
    | Some s_arr -> (
        match into_merge_buffer with
        | No -> (
            match Map.find dst.ctx_arrays tn with
            | None -> None
            | Some d_arr ->
                if phys_equal s_arr d_arr then None
                else
                  let context = Backend.make_child dst in
                  let description =
                    "device_to_device " ^ Tn.debug_name tn ^ " from " ^ Backend.get_name src.stream
                    ^ " to " ^ Backend.get_name dst.stream
                  in
                  let work () =
                    wait_for_ready ~dst ~src tn;
                    Backend.(
                      device_to_device tn ~into_merge_buffer ~dst_ptr:(Some d_arr) ~dst
                        ~src_ptr:s_arr ~src);
                    update_writer_event dst @@ Node tn;
                    [%log
                      "copying",
                      Tn.debug_name tn,
                      "from",
                      Backend.get_name src.stream,
                      "to",
                      Backend.get_name dst.stream]
                  in
                  let schedule =
                    Task.Task { context_lifetime = (src, dst); description; work }
                  in
                  Some
                    {
                      context;
                      schedule;
                      bindings = [];
                      name = description;
                      inputs = Set.singleton (module Tnode) tn;
                      merge_buffer_input = None;
                      outputs = Set.singleton (module Tnode) tn;
                    })
        | Copy ->
            let context = Backend.make_child dst ~merge_buffer_node:(Some tn) in
            let description =
              "device_to_device " ^ Tn.debug_name tn ^ " into merge buffer from "
              ^ Backend.get_name src.stream
            in
            let work () =
              wait_for_ready ~dst ~src tn;
              Backend.(
                device_to_device tn ~into_merge_buffer ~dst_ptr:None ~dst ~src_ptr:s_arr ~src);
              update_writer_event dst @@ Merge_buffer tn;
              [%log "copy into merge buffer", Tn.debug_name tn, "from", Backend.get_name src.stream]
            in
            let schedule = Task.Task { context_lifetime = (src, dst); description; work } in
            Some
              {
                context;
                schedule;
                bindings = [];
                name = description;
                inputs = Set.singleton (module Tnode) tn;
                merge_buffer_input = None;
                outputs = Set.empty (module Tnode);
              })

  let%track3_sexp init_from_device (tn : Tn.t) ~(dst : Backend.context) ~(src : Backend.context) =
    match Map.find src.ctx_arrays tn with
    | None ->
        raise
        @@ Utils.User_error
             ("init_from_device: tensor node " ^ Tn.debug_name tn ^ " is not in input context "
            ^ Backend.get_name src.stream ^ ", for stream " ^ Backend.get_name dst.stream)
    | Some s_arr -> (
        wait_for_ready ~dst ~src tn;
        match Map.find dst.ctx_arrays tn with
        | Some _ ->
            raise
            @@ Utils.User_error
                 ("init_from_device: tensor node " ^ Tn.debug_name tn
                ^ " already in output context " ^ Backend.get_name dst.stream ^ ", for stream "
                ^ Backend.get_name src.stream)
        | None ->
            let dims = Lazy.force tn.dims in
            (* Use alloc_array since we're immediately copying from another device *)
            let d_arr =
              Backend.alloc_array
                ?mode:(Option.map tn.memory_mode ~f:fst)
                (Lazy.force tn.prec) ~dims dst.stream
            in
            Backend.(
              device_to_device tn ~into_merge_buffer:No ~dst_ptr:(Some d_arr) ~dst ~src_ptr:s_arr
                ~src);
            update_writer_event dst @@ Node tn;
            [%log
              "copying",
              Tn.debug_name tn,
              "from",
              Backend.get_name src.stream,
              "to",
              Backend.get_name dst.stream];
            { dst with ctx_arrays = Map.add_exn dst.ctx_arrays ~key:tn ~data:d_arr })

  type r = Backend.context routine [@@deriving sexp_of]

  let sync_routine (r : r) : r =
    (* Host transfers are no longer automatic (gh-ocannl-333): all CPU-side access goes through
       explicit, on-demand [Context] transfers. [sync_routine] now only records the post-execution
       writer event for the routine's outputs (used for device-side ordering and merge buffers). *)
    let s = r.context.stream in
    let post () =
      let e = Backend.all_work s in
      Set.iter r.outputs ~f:(fun tn -> update_writer_event ~e r.context @@ Node tn)
    in
    { r with schedule = Task.(append ~work:post r.schedule) }

  let sync_device device =
    Utils.weak_iter device.streams ~f:Backend.await;
    Utils.weak_iter device.streams ~f:(fun s ->
        s.updating_for_merge_buffer <- None;
        Hashtbl.clear s.updating_for)
end

let%track6_sexp lower_assignments optim_ctx ?name bindings asgns =
  let name : string =
    Option.value_or_thunk name ~default:(fun () -> Assignments.get_name_exn asgns)
  in
  let unoptim_ll_source = Utils.output_to_build_file ~fname:(name ^ "-unoptimized.ll") in
  let ll_source = Utils.output_to_build_file ~fname:(name ^ ".ll") in
  let cd_source = Utils.output_to_build_file ~fname:(name ^ ".cd") in
  ( name,
    Assignments.lower optim_ctx ~unoptim_ll_source ~ll_source ~cd_source ~name
      (Indexing.bound_symbols bindings) asgns )

let lower_batch_assignments optim_ctx ?names ?occupancy bindings asgns_l =
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
          Some
            (Assignments.lower optim_ctx ~unoptim_ll_source ~ll_source ~cd_source ~name bound asgns)
        )
      else (None, None))

let%debug3_sexp verify_prior_context ~use_host_memory ~ctx_arrays ~from_prior_context : unit =
  Set.iter from_prior_context ~f:(fun tn ->
      if
        Tn.is_in_context_force ~use_host_memory tn 42
        && (not (Option.is_some @@ Map.find ctx_arrays tn))
        (* Nodes with registered host initialization data (ndarray-backed literals, loaded tensors)
           self-initialize in this context at link time from [Host_inits] (gh-ocannl-333), so they
           need not be present in a prior context. *)
        && not (Host_inits.mem tn)
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
      ->
      With_scheduler
        with type buffer_ptr = Impl.buffer_ptr
         and type optimize_ctx = Low_level.optimize_ctx)
    (Backend : Lowered_no_device_backend)
(* : Lowered_backend *) =
struct
  include Backend

  include Add_scheduler (struct
    include Backend
  end)

  type code = { lowered : Low_level.optimized; proc : Backend.procedure } [@@deriving sexp_of]

  type code_batch = {
    lowereds : Low_level.optimized option array;
    procs : Backend.procedure option array;
  }
  [@@deriving sexp_of]

  let compile ~(name : string) bindings lowered : code =
    let proc = compile ~name bindings lowered in
    { lowered; proc }

  let compile_batch ~names bindings lowereds : code_batch =
    let procs = compile_batch ~names bindings lowereds in
    { lowereds; procs }

  let link context (code : code) ctx_arrays : Indexing.lowered_bindings * Task.t =
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
            let ctx_arrays = Option.value_exn ~here:[%here] ctx_arrays.(i) in
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
    let size_in_bytes = Lazy.force tn.Tnode.size_in_bytes in
    let work =
      (* TODO: log the operation if [Utils.settings.with_log_level > 1]. *)
      match (into_merge_buffer, dst_ptr) with
      | No, None -> invalid_arg "Multicore_scheduler.device_to_device: missing dst_ptr"
      | No, Some dst_ptr -> fun () -> buffer_to_buffer ~dst:dst_ptr ~src:src_ptr ~size_in_bytes
      | Copy, _ ->
          fun () ->
            let allocated_capacity =
              match s.allocated_buffer with None -> 0 | Some buf -> buf.size_in_bytes
            in
            if allocated_capacity < size_in_bytes then
              s.allocated_buffer <-
                Some
                  (alloc_buffer ?old_buffer:s.allocated_buffer
                     ?mode:(Option.map tn.Tnode.memory_mode ~f:fst)
                     ~size_in_bytes dst.stream);
            let merge_ptr = (Option.value_exn ~here:[%here] s.allocated_buffer).ptr in
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
  module Device_with_optimize_ctx = struct
    include Device

    type optimize_ctx = Low_level.optimize_ctx [@@deriving sexp_of]
  end

  include Device_with_optimize_ctx
  include Add_buffer_retrieval_and_syncing (Device_with_optimize_ctx)

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

  type nonrec optimize_ctx = Low_level.optimize_ctx

  let empty_optimize_ctx () = { Low_level.computations = Hashtbl.create (module Tnode) }
  let get_optimize_ctx (code : code) = code.lowered.optimize_ctx

  let get_optimize_ctx_batch (code_batch : code_batch) =
    Array.find_map code_batch.lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.optimize_ctx))
    |> Option.value_or_thunk ~default:empty_optimize_ctx

  let%debug3_sexp compile optim_ctx ?name bindings (comp : Assignments.comp) : code =
    let (name : string), (lowered : Low_level.optimized) =
      lower_assignments optim_ctx ?name bindings comp.asgns
    in
    let code : Device.code = compile ~name bindings lowered in
    let from_prior_context : Tn.t_set =
      Set.diff (Assignments.context_nodes ~use_host_memory comp.asgns) comp.embedded_nodes
    in
    { from_prior_context; name; lowered; code; expected_merge_node = lowered.Low_level.merge_node }

  let%debug3_sexp compile_batch optim_ctx ?names ?occupancy bindings
      (comps : Assignments.comp array) : code_batch =
    let names, lowereds =
      lower_batch_assignments optim_ctx ?names ?occupancy bindings
      @@ Array.map comps ~f:(fun c -> c.asgns)
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
      (* Host initialization data for ndarray-backed literals / loaded nodes is registered in the
         weakly-owned [Host_inits] table (gh-ocannl-333). It is read (not consumed) here, so the
         same node can be initialized into multiple independent contexts. *)
      let host_init = Host_inits.find key in
      let default () =
        let dims = Lazy.force key.dims in
        (* Use alloc_array when zero initialization is not needed: - When copying from host
           immediately after allocation - When the node has explicit Zero_out operations in the
           lowered code *)
        let will_copy_from_host = Option.is_some host_init in
        let mode = Option.map key.memory_mode ~f:fst in
        let dst_ptr =
          if will_copy_from_host || node.Low_level.zero_initialized_by_code then
            alloc_array ?mode (Lazy.force key.prec) ~dims stream
          else alloc_zeros ?mode (Lazy.force key.prec) ~dims stream
        in
        Option.iter host_init ~f:(fun nd ->
            Device.from_host ~dst_ptr ~dst:parent_context (Lazy.force nd));
        dst_ptr
      in
      let add_new_exn () =
        try Map.add_exn ctx_arrays ~key ~data:(default ())
        with exn ->
          [%log "Backends.alloc_if_needed: failed to add new node to context", (key : Tnode.t)];
          raise exn
      in
      let add_old_exn data =
        try Map.add_exn ctx_arrays ~key ~data
        with exn ->
          [%log "Backends.alloc_if_needed: failed to add old node to context", (key : Tnode.t)];
          raise exn
      in
      let device = stream.device in
      (* It's the user's responsibility to ensure that constants are initialized on devices, the
         user can choose to run initialization code on multiple streams redundantly, or on the owner
         stream only and then to use init_from_device. Constant / read-only buffers are shared
         across contexts on a device via [constant_buffer_cache]; their host initialization data, if
         any, is copied (not pointer-wrapped) into a device buffer at first use. *)
      if node.Low_level.read_only || Tn.known_constant key then add_old_exn
      @@ Hashtbl.find_or_add device.constant_buffer_cache key ~default
      else add_new_exn ())
    else ctx_arrays

  let%debug3_sexp link context (code : code) =
    verify_prior_context ~use_host_memory ~ctx_arrays:context.ctx_arrays
      ~from_prior_context:code.from_prior_context;
    (* Static merge-buffer verification "in the right direction" (gh-ocannl-288): the linked
       context carries the merge-buffer node of the producing [device_to_device] transfer routine;
       a mismatch with the consuming code raises here, at link time, before any schedule runs. *)
    check_merge_buffer_static ~merge_buffer_node:context.merge_buffer_node
      ~code_node:code.expected_merge_node;
    let (inputs, outputs), merge_buffer_input = Low_level.input_and_output_nodes code.lowered in
    let ctx_arrays =
      Hashtbl.fold code.lowered.traced_store ~init:context.ctx_arrays ~f:(alloc_if_needed context)
    in
    let optimize_ctx = code.lowered.optimize_ctx in
    let bindings, schedule = link context code.code ctx_arrays in
    let context = make_child ~ctx_arrays ~optimize_ctx context in
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
          let optimize_ctx = (Option.value_exn code_batch.lowereds.(i)).Low_level.optimize_ctx in
          let expected_merge_node = code_batch.expected_merge_nodes.(i) in
          (* Static merge-buffer verification at link time (gh-ocannl-288): check the node provided
             by the fold-current context before deriving the consumer's child context. *)
          check_merge_buffer_static ~merge_buffer_node:context.merge_buffer_node
            ~code_node:expected_merge_node;
          let context = make_child ~ctx_arrays ~optimize_ctx context in
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
      ->
      With_scheduler
        with type buffer_ptr = Impl.buffer_ptr
         and type optimize_ctx = Low_level.optimize_ctx)
    (Backend_impl : Lowered_no_device_backend) =
struct
  module Lowered_device = Add_device (Add_scheduler) (Backend_impl)
  module Backend_device = Raise_backend (Lowered_device)
  include Backend_device
end

let finalize (type buffer_ptr dev runner event optimize_ctx)
    (module Backend : Backend
      with type buffer_ptr = buffer_ptr
       and type dev = dev
       and type runner = runner
       and type event = event
       and type optimize_ctx = optimize_ctx) (ctx : Backend.context) : unit =
  Option.iter Backend.free_buffer ~f:(fun mem_free ->
      if Atomic.compare_and_set ctx.finalized false true then (
        Backend.await ctx.stream;
        Map.iteri ctx.ctx_arrays ~f:(fun ~key ~data ->
            if
              (not (Option.exists ctx.parent ~f:(fun pc -> Map.mem pc.ctx_arrays key)))
              && not (Hashtbl.mem ctx.stream.device.constant_buffer_cache key)
            then mem_free ctx.stream data)))

let%track5_sexp fresh_backend ?backend_name () =
  Stdlib.Gc.full_major ();
  (* TODO: is running again needed to give time to weak arrays to become empty? *)
  Stdlib.Gc.full_major ();
  (* Note: we invoke functors from within fresh_backend to fully isolate backends from distinct
     calls to fresh_backend. *)
  match
    Option.value_or_thunk backend_name ~default:(fun () ->
        Utils.get_global_arg ~arg_name:"backend" ~default:"multicore_cc")
    |> String.lowercase
  with
  | "multicore_cc" ->
      (module Make_device_backend_from_lowered (Schedulers.Multicore) (Cc_backend) : Backend)
  | "sync_cc" -> (module Make_device_backend_from_lowered (Schedulers.Sync) (Cc_backend) : Backend)
  | "cuda" -> (module Raise_backend (Cuda_backend_impl.Fresh () : Lowered_backend) : Backend)
  | "metal" -> (module Raise_backend (Metal_backend_impl.Fresh () : Lowered_backend) : Backend)
  | backend -> invalid_arg [%string "Backends.fresh_backend: unknown backend %{backend}"]
