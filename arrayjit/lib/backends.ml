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

(* gh-ocannl-344: pure planner for the pool allocator. Lays out a sequence of [(size, alignment)]
   allocations (in order) into one or more pools so that no pool's bumped extent exceeds [cap] bytes
   -- the per-pool 4 GB ceiling for uint32 offsets when [large_models = false]. Returns, per input
   item, its [(segment_index, byte_offset)], plus the byte size of each segment (pool). Raises
   [Utils.User_error] (naming [what] and [debug_name i]) if a single item exceeds [cap], since no
   pool can hold it without uint64 offsets. Factored out so the segmenting/cap behavior is unit
   testable with synthetic sizes (it does not need real device memory). *)
let plan_pool_segments ~(cap : int) ~(what : string) ~(debug_name : int -> string)
    (items : (int * int) list) : (int * int) list * int list =
  let align_up off a = if a <= 1 then off else (off + a - 1) / a * a in
  let seg = ref 0 and bump = ref 0 in
  let closed = ref [] (* completed segment sizes, reversed *) in
  let assignments =
    List.mapi items ~f:(fun i (size, align) ->
        if size > cap then
          raise
          @@ Utils.User_error
               (Printf.sprintf
                  "%s: tensor node %s needs %d bytes, over the %d-byte per-pool cap; set \
                   large_models=true for uint64 offsets"
                  what (debug_name i) size cap);
        let offset = align_up !bump align in
        if offset + size > cap then (
          (* Close the current pool and open a new one starting this item at offset 0. *)
          closed := !bump :: !closed;
          Int.incr seg;
          bump := size;
          (!seg, 0))
        else (
          bump := offset + size;
          (!seg, offset)))
  in
  let segment_sizes = if List.is_empty items then [] else List.rev (!bump :: !closed) in
  (assignments, segment_sizes)

(* Dynamic backstop for merge-buffer verification: runs as the first work of a consumer's schedule
   and checks the node most recently scheduled into the stream's merge buffer. The primary check is
   now the static [check_merge_buffer_static] performed at link time (gh-ocannl-288); this remains
   as a defensive backstop for transfers that are scheduled without a downstream link. *)
let check_merge_buffer device ~code_node =
  let name = function Some tn -> Tnode.debug_name tn | None -> "none" in
  match (device.updating_for_merge_buffer, code_node) with
  | _, None -> ()
  | Some (actual, _), Some expected when Tnode.equal actual expected -> ()
  | _ ->
      raise
      @@ Utils.User_error
           ("Merge buffer mismatch, on device: "
           ^ name (Option.map ~f:fst device.updating_for_merge_buffer)
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
    let s = src.device in
    let d = dst.device in
    (* TODO: maybe it's worthwhile to clean up s.updating_for every now and then. *)
    Hashtbl.find s.updating_for tn
    |> Option.iter ~f:(fun upd_e ->
        if not (equal_device s d || Backend.is_done upd_e) then Backend.will_wait_for dst upd_e)

  (* Shared allocator seam: mints a deterministic per-device [pool_id] (advancing
     [device.next_pool_id] in the caller's tnode-iteration order), allocates the slab through the
     backend's int-in/int-out API, and returns the [buffer_loc]. Phase-1 policy is one pool per tnode
     at offset 0 -- byte-for-byte equivalent to the old per-tnode allocation. [zero_init] selects the
     old [alloc_zeros] vs [alloc_array] behavior. *)
  let allocate (device : _ Backend_intf.device) (tn : Tn.t) ~zero_init : Backend_intf.buffer_loc =
    let pool_id = device.next_pool_id in
    device.next_pool_id <- pool_id + 1;
    let prec = Lazy.force tn.Tn.prec in
    (* Compute the byte size from dims*prec rather than forcing [tn.size_in_bytes], to keep the
       node's debug printout (and lazy-forcing behavior) byte-for-byte as before. *)
    let size_in_bytes = Array.fold (Lazy.force tn.Tn.dims) ~init:1 ~f:( * ) * Ops.prec_in_bytes prec in
    let mode = Option.map tn.Tn.memory_mode ~f:fst in
    Backend.alloc_pool ?mode device ~pool_id ~size_in_bytes ~alignment:(Ops.prec_in_bytes prec);
    if zero_init then Backend.memset_zero device ~pool_id ~offset:0 ~size_in_bytes;
    { pool_id; offset = 0 }

  let%track3_sexp to_host (ctx : Backend.context) (tn : Tn.t) (hosted : Ndarray.t) =
    match Map.find ctx.ctx_buffers tn with
    | Some loc ->
        [%log "copying", Tn.debug_name tn, "at", (loc : Backend_intf.buffer_loc), "to host"];
        (* No cross-stream writer synchronization needed: multi-streaming was removed
           (gh-ocannl-341). Only one stream exists per device, so there are no
           concurrent cross-stream writes to wait for before this device-to-host copy. *)
        Backend.to_host ~src:ctx ~src_loc:loc hosted;
        true
    | None -> false

  let update_writer_event ?e ctx tn =
    let s = ctx.device in
    let e = Option.value_or_thunk e ~default:(fun () -> Backend.all_work s) in
    match tn with
    | Assignments.Node tn -> Hashtbl.update s.updating_for tn ~f:(fun _ -> e)
    | Assignments.Merge_buffer tn ->
        (* Note: the previous event does not need to be done! *)
        s.updating_for_merge_buffer <- Some (tn, Some e)

  let%track3_sexp from_host (ctx : Backend.context) (tn : Tn.t) (hosted : Ndarray.t) =
    match Map.find ctx.ctx_buffers tn with
    | Some dst ->
        (* No cross-stream reader synchronization needed: multi-streaming was removed
           (gh-ocannl-341). Only one stream exists per device, so there are no concurrent
           cross-stream readers to wait for before this host-to-device upload. *)
        [%log "copying", Tn.debug_name tn, "to", (dst : Backend_intf.buffer_loc), "from host"];
        Backend.from_host ~dst:ctx ~dst_loc:dst hosted;
        update_writer_event ctx @@ Node tn;
        true
    | None -> false

  let%track3_sexp init_from_host (ctx : Backend.context) (tn : Tn.t) (hosted : Ndarray.t) =
    match Map.find ctx.ctx_buffers tn with
    | None ->
        (* No zero-init: we are immediately copying from host. *)
        let dst = allocate ctx.device tn ~zero_init:false in
        [%log "copying", Tn.debug_name tn, "to", (dst : Backend_intf.buffer_loc), "from host"];
        Backend.from_host ~dst:ctx ~dst_loc:dst hosted;
        update_writer_event ctx @@ Node tn;
        { ctx with ctx_buffers = Map.add_exn ctx.ctx_buffers ~key:tn ~data:dst }
    | Some _ ->
        raise
        @@ Utils.User_error
             ("init_from_host: input context already contains tensor node " ^ Tn.debug_name tn
            ^ ", for device " ^ Backend.get_name ctx.device)

  (* [device_to_device] builds a transfer routine instead of scheduling the copy directly. The
     caller schedules it (via [Task.run r.schedule]) or links a consumer against [r.context]. For
     the [Copy] case, [r.context]'s [merge_buffer_node] records the produced node statically, so
     that [link] can verify it against a consumer's [expected_merge_node] at link time -- the
     "static verification in the right direction" of gh-ocannl-288. *)
  let%track3_sexp device_to_device (tn : Tn.t) ~into_merge_buffer ~(dst : Backend.context)
      ~(src : Backend.context) : Backend.context routine option =
    match Map.find src.ctx_buffers tn with
    | None -> None
    | Some s_loc -> (
        match into_merge_buffer with
        | No -> (
            match Map.find dst.ctx_buffers tn with
            | None -> None
            | Some d_loc ->
                (* Same device + same location => physically the same buffer; nothing to copy. *)
                if equal_device src.device dst.device && [%equal: buffer_loc] s_loc d_loc then None
                else
                  let context = Backend.make_child dst in
                  let description =
                    "device_to_device " ^ Tn.debug_name tn ^ " from " ^ Backend.get_name src.device
                    ^ " to " ^ Backend.get_name dst.device
                  in
                  let work () =
                    wait_for_ready ~dst ~src tn;
                    Backend.(
                      device_to_device tn ~into_merge_buffer ~dst_loc:(Some d_loc) ~dst
                        ~src_loc:s_loc ~src);
                    update_writer_event dst @@ Node tn;
                    [%log
                      "copying",
                      Tn.debug_name tn,
                      "from",
                      Backend.get_name src.device,
                      "to",
                      Backend.get_name dst.device]
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
              ^ Backend.get_name src.device
            in
            let work () =
              wait_for_ready ~dst ~src tn;
              Backend.(
                device_to_device tn ~into_merge_buffer ~dst_loc:None ~dst ~src_loc:s_loc ~src);
              update_writer_event dst @@ Merge_buffer tn;
              [%log "copy into merge buffer", Tn.debug_name tn, "from", Backend.get_name src.device]
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
    match Map.find src.ctx_buffers tn with
    | None ->
        raise
        @@ Utils.User_error
             ("init_from_device: tensor node " ^ Tn.debug_name tn ^ " is not in input context "
            ^ Backend.get_name src.device ^ ", for device " ^ Backend.get_name dst.device)
    | Some s_loc -> (
        wait_for_ready ~dst ~src tn;
        match Map.find dst.ctx_buffers tn with
        | Some _ ->
            raise
            @@ Utils.User_error
                 ("init_from_device: tensor node " ^ Tn.debug_name tn
                ^ " already in output context " ^ Backend.get_name dst.device ^ ", for device "
                ^ Backend.get_name src.device)
        | None ->
            (* No zero-init: we are immediately copying from another device. *)
            let d_loc = allocate dst.device tn ~zero_init:false in
            Backend.(
              device_to_device tn ~into_merge_buffer:No ~dst_loc:(Some d_loc) ~dst ~src_loc:s_loc
                ~src);
            update_writer_event dst @@ Node tn;
            [%log
              "copying",
              Tn.debug_name tn,
              "from",
              Backend.get_name src.device,
              "to",
              Backend.get_name dst.device];
            { dst with ctx_buffers = Map.add_exn dst.ctx_buffers ~key:tn ~data:d_loc })

  type r = Backend.context routine [@@deriving sexp_of]

  let sync_routine (r : r) : r =
    (* Host transfers are no longer automatic (gh-ocannl-333): all CPU-side access goes through
       explicit, on-demand [Context] transfers. [sync_routine] now only records the post-execution
       writer event for the routine's outputs (used for device-side ordering and merge buffers). *)
    let s = r.context.device in
    let post () =
      let e = Backend.all_work s in
      Set.iter r.outputs ~f:(fun tn -> update_writer_event ~e r.context @@ Node tn)
    in
    { r with schedule = Task.(append ~work:post r.schedule) }

  let sync_device device =
    Backend.await device;
    device.updating_for_merge_buffer <- None;
    Hashtbl.clear device.updating_for
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

let%debug3_sexp verify_prior_context ~ctx_arrays ~from_prior_context : unit =
  Set.iter from_prior_context ~f:(fun tn ->
      if
        Tn.is_in_context_force tn 42
        && (not (Option.is_some @@ Map.find ctx_arrays tn))
        (* Nodes with registered host initialization data (ndarray-backed literals, loaded tensors)
           self-initialize in this context at link time from [Host_inits] (gh-ocannl-333), so they
           need not be present in a prior context. *)
        && not (Host_inits.mem tn)
      then raise @@ Utils.User_error ("The linked context lacks node " ^ Tnode.debug_name tn))

let%debug3_sexp from_prior_context_batch (comps : Assignments.comp option array) : Tn.t_set =
  Array.filter_map comps ~f:(fun comp ->
      Option.map comp ~f:(fun comp ->
          Set.diff
            (Assignments.context_nodes comp.Assignments.asgns)
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

  let link context (code : code) ctx_buffers : Indexing.lowered_bindings * Task.t =
    let runner_label = get_name context.device in
    let merge_buffer = context.device.merge_buffer in
    (* [resolve] is the device's backend-private [buffer_loc -> base] lookup; [link_compiled] does the
       (eager) [ctx_buffers] and (lazy) merge-buffer resolution with it, backend-side. The generic
       shared layer never sees a raw pointer. *)
    let resolve = resolve_pool context.device in
    let bindings, to_schedule =
      link_compiled ~merge_buffer ~resolve ~runner_label ctx_buffers code.proc
    in
    let schedule =
      Task.enschedule ~schedule_task ~get_stream_name:get_name context.device to_schedule
    in
    (bindings, schedule)

  let link_batch context (code_batch : code_batch) ctx_buffers =
    let runner_label = get_name context.device in
    let merge_buffer = context.device.merge_buffer in
    let resolve = resolve_pool context.device in
    let bindings, schedules =
      Array.fold_mapi code_batch.procs ~init:None ~f:(fun i bindings -> function
        | Some proc ->
            let ctx_buffers = Option.value_exn ~here:[%here] ctx_buffers.(i) in
            let bindings', to_schedule =
              link_compiled ~merge_buffer ~resolve ~runner_label ctx_buffers proc
            in
            Option.iter bindings ~f:(fun bindings -> assert (phys_equal bindings bindings'));
            let schedule =
              Task.enschedule ~schedule_task ~get_stream_name:get_name context.device to_schedule
            in
            (Some bindings', Some schedule)
        | None -> (bindings, None))
    in
    (Option.value_exn ~here:[%here] bindings, schedules)

  (* Transfers take {!Backend_intf.buffer_loc} and resolve to the backend pointer here, against the
     device's private pool table -- the resolution is backend-side, not in the generic shared layer. *)
  let from_host ~dst ~dst_loc hosted =
    let dst_ptr = resolve_pool dst.device dst_loc in
    let work () = host_to_buffer hosted ~dst:dst_ptr in
    schedule_task dst.device
      (Task.Task
         { context_lifetime = dst; description = "from_host on " ^ get_name dst.device; work })

  let to_host ~src ~src_loc hosted =
    let src_ptr = resolve_pool src.device src_loc in
    let work () = buffer_to_host hosted ~src:src_ptr in
    schedule_task src.device
      (Task.Task { context_lifetime = src; description = "to_host on " ^ get_name src.device; work })

  let device_to_device tn ~into_merge_buffer ~dst_loc ~dst ~src_loc ~src =
    let s = dst.device in
    let size_in_bytes = Lazy.force tn.Tnode.size_in_bytes in
    let src_ptr = resolve_pool src.device src_loc in
    let work =
      (* TODO: log the operation if [Utils.settings.with_log_level > 1]. *)
      match (into_merge_buffer, dst_loc) with
      | No, None -> invalid_arg "Multicore_scheduler.device_to_device: missing dst_loc"
      | No, Some dst_loc ->
          let dst_ptr = resolve_pool dst.device dst_loc in
          fun () -> buffer_to_buffer ~dst:dst_ptr ~src:src_ptr ~size_in_bytes
      | Copy, _ ->
          fun () ->
            (* The merge buffer is the device's reserved single-tenant pool; grow it in place when a
               larger node arrives ([alloc_pool] overwrites the reserved pool-id entry). *)
            if s.merge_buffer_capacity < size_in_bytes then (
              alloc_pool
                ?mode:(Option.map tn.Tnode.memory_mode ~f:fst)
                s ~pool_id:merge_buffer_pool_id ~size_in_bytes
                ~alignment:(Ops.prec_in_bytes (Lazy.force tn.Tnode.prec));
              s.merge_buffer_capacity <- size_in_bytes);
            let loc = { pool_id = merge_buffer_pool_id; offset = 0 } in
            s.merge_buffer := Some loc;
            buffer_to_buffer ~dst:(resolve_pool s loc) ~src:src_ptr ~size_in_bytes
    in
    let description =
      "device_to_device " ^ Tnode.debug_name tn ^ " dst " ^ get_name s ^ " src "
      ^ get_name src.device
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
      Set.diff (Assignments.context_nodes comp.asgns) comp.embedded_nodes
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

  let size_in_bytes_of (key : Tn.t) =
    let prec = Lazy.force key.Tn.prec in
    Array.fold (Lazy.force key.Tn.dims) ~init:1 ~f:( * ) * Ops.prec_in_bytes prec

  (* gh-ocannl-344 Phase B/C: allocate a context's delta -- the in-context tnodes not already present
     in [context.ctx_buffers]. Working (non-constant) and constant/read-only nodes are EACH packed
     into pools sized to their group and bump-assigned increasing byte offsets, replacing the
     one-pool-per-tnode policy. Working pools belong to the context (freed at its [finalize]);
     constant pools are deduped per-device via [constant_buffer_cache] and outlive the context (freed
     at device teardown). Enumeration follows [traced_store] order so pool ids and offsets stay
     deterministic across runs. The per-pool 4 GB cap (uint32 offsets unless large_models) is enforced
     by {!Backend_utils.plan_pool_segments}. *)
  let%track3_sexp allocate_delta (context : context) (traced_store : Low_level.traced_store) :
      ctx_buffers =
    let device = context.device in
    let cap = if Utils.settings.large_models then Int.max_value else 0x1_0000_0000 in
    (* Pass 1: partition the delta, preserving [traced_store] iteration order. *)
    let working = ref [] and constants = ref [] in
    Hashtbl.iteri traced_store ~f:(fun ~key ~data:node ->
        if Tnode.is_in_context_force key 43 && not (Map.mem context.ctx_buffers key) then
          if node.Low_level.read_only || Tn.known_constant key then
            constants := (key, node) :: !constants
          else working := (key, node) :: !working);
    let working = List.rev !working and constants = List.rev !constants in
    let ctx_buffers = ref context.ctx_buffers in
    (* Pack a group of (key, node) into one or more pools, segmenting at the cap. [register] decides
       how the resulting [buffer_loc] is recorded (directly into [ctx_buffers] for working nodes, or
       deduped through [constant_buffer_cache] for constants). [base_pool_id] of each segment is a
       freshly minted [next_pool_id]; offsets and pool sizes come from the pure planner. *)
    let pack (group : (Tn.t * Low_level.traced_array) list)
        ~(register : Tn.t -> alloc:(unit -> buffer_loc) -> unit) : unit =
      if not (List.is_empty group) then begin
        let items =
          List.map group ~f:(fun (key, _) ->
              (size_in_bytes_of key, Ops.prec_in_bytes (Lazy.force key.Tn.prec)))
        in
        let assignments, segment_sizes =
          plan_pool_segments ~cap ~what:"Backends.allocate_delta"
            ~debug_name:(fun i -> Tn.debug_name (fst (List.nth_exn group i)))
            items
        in
        (* Mint a pool id per segment up front, sized from the planner. *)
        let seg_pool_ids =
          List.map segment_sizes ~f:(fun size_in_bytes ->
              let pool_id = device.next_pool_id in
              device.next_pool_id <- pool_id + 1;
              (pool_id, size_in_bytes))
          |> Array.of_list
        in
        (* Allocate each segment's slab, padding alignment to the max element precision it holds. *)
        let seg_align = Array.map seg_pool_ids ~f:(fun _ -> ref 1) in
        List.iter2_exn group assignments ~f:(fun (key, _) (seg, _) ->
            let a = Ops.prec_in_bytes (Lazy.force key.Tn.prec) in
            if a > !(seg_align.(seg)) then seg_align.(seg) := a);
        Array.iteri seg_pool_ids ~f:(fun seg (pool_id, size_in_bytes) ->
            alloc_pool device ~pool_id ~size_in_bytes ~alignment:!(seg_align.(seg)));
        (* Place each node at its planned (segment, offset). *)
        List.iter2_exn group assignments ~f:(fun (key, node) (seg, offset) ->
            let pool_id, _ = seg_pool_ids.(seg) in
            let size_in_bytes = size_in_bytes_of key in
            let alloc () : buffer_loc =
              let host_init = Host_inits.find key in
              (* Zero-initialize unless the node will be copied from host immediately, or the lowered
                 code already zero-initializes it. *)
              let zero_init =
                not (Option.is_some host_init || node.Low_level.zero_initialized_by_code)
              in
              if zero_init then memset_zero device ~pool_id ~offset ~size_in_bytes;
              let loc = { pool_id; offset } in
              Option.iter host_init ~f:(fun nd ->
                  Device.from_host ~dst:context ~dst_loc:loc (Lazy.force nd));
              loc
            in
            register key ~alloc)
      end
    in
    (* Pass 2a: working delta -> context-owned pool(s), recorded directly. *)
    pack working ~register:(fun key ~alloc ->
        ctx_buffers := Map.add_exn !ctx_buffers ~key ~data:(alloc ()));
    (* Pass 2b: constants / read-only -> per-device constant pool(s). Constants already allocated on
       this device (a hit in [constant_buffer_cache], possibly from another context tree) resolve
       directly and are excluded from the new slab, so the freshly-minted constant pool holds exactly
       this device's genuinely-new constants -- no wasted holes. The remaining new constants pack
       into one constant pool (or more, past the cap), deduped into the cache. Constant pools outlive
       the context and are skipped by context [finalize] (freed at device teardown). *)
    let new_constants = ref [] in
    List.iter constants ~f:(fun (key, node) ->
        match Hashtbl.find device.constant_buffer_cache key with
        | Some data -> ctx_buffers := Map.add_exn !ctx_buffers ~key ~data
        | None -> new_constants := (key, node) :: !new_constants);
    pack (List.rev !new_constants) ~register:(fun key ~alloc ->
        let data = Hashtbl.find_or_add device.constant_buffer_cache key ~default:alloc in
        ctx_buffers := Map.add_exn !ctx_buffers ~key ~data);
    !ctx_buffers

  let%debug3_sexp link context (code : code) =
    verify_prior_context ~ctx_arrays:context.ctx_buffers
      ~from_prior_context:code.from_prior_context;
    (* Static merge-buffer verification "in the right direction" (gh-ocannl-288): the linked
       context carries the merge-buffer node of the producing [device_to_device] transfer routine;
       a mismatch with the consuming code raises here, at link time, before any schedule runs. *)
    check_merge_buffer_static ~merge_buffer_node:context.merge_buffer_node
      ~code_node:code.expected_merge_node;
    let (inputs, outputs), merge_buffer_input = Low_level.input_and_output_nodes code.lowered in
    let ctx_buffers = allocate_delta context code.lowered.traced_store in
    let optimize_ctx = code.lowered.optimize_ctx in
    let bindings, schedule = link context code.code ctx_buffers in
    let context = make_child ~ctx_buffers ~optimize_ctx context in
    let schedule =
      Task.prepend schedule ~work:(fun () ->
          check_merge_buffer context.device ~code_node:code.expected_merge_node)
    in
    sync_routine
      { context; schedule; bindings; name = code.name; inputs; merge_buffer_input; outputs }

  let%debug3_sexp link_batch context code_batch =
    verify_prior_context ~ctx_arrays:context.ctx_buffers
      ~from_prior_context:code_batch.from_prior_context;
    let ctx_buffers =
      Array.map code_batch.lowereds
        ~f:(Option.map ~f:(fun l -> allocate_delta context l.Low_level.traced_store))
    in
    let bindings, schedules = link_batch context code_batch.code_batch ctx_buffers in
    Array.fold_mapi schedules ~init:context ~f:(fun i context -> function
      | None -> (context, None)
      | Some schedule ->
          let ctx_buffers = Option.value_exn ctx_buffers.(i) in
          let optimize_ctx = (Option.value_exn code_batch.lowereds.(i)).Low_level.optimize_ctx in
          let expected_merge_node = code_batch.expected_merge_nodes.(i) in
          (* Static merge-buffer verification at link time (gh-ocannl-288): check the node provided
             by the fold-current context before deriving the consumer's child context. *)
          check_merge_buffer_static ~merge_buffer_node:context.merge_buffer_node
            ~code_node:expected_merge_node;
          let context = make_child ~ctx_buffers ~optimize_ctx context in
          let (inputs, outputs), merge_buffer_input =
            Low_level.input_and_output_nodes @@ Option.value_exn code_batch.lowereds.(i)
          in
          let schedule =
            Task.prepend schedule ~work:(fun () ->
                check_merge_buffer context.device ~code_node:expected_merge_node)
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

let finalize (type dev runner event optimize_ctx)
    (module Backend : Backend
      with type dev = dev
       and type runner = runner
       and type event = event
       and type optimize_ctx = optimize_ctx) (ctx : Backend.context) : unit =
  Option.iter Backend.free_pool ~f:(fun free_pool ->
      if Atomic.compare_and_set ctx.finalized false true then (
        Backend.await ctx.device;
        Map.iteri ctx.ctx_buffers ~f:(fun ~key ~data:(loc : Ir.Backend_intf.buffer_loc) ->
            if
              (not (Option.exists ctx.parent ~f:(fun pc -> Map.mem pc.ctx_buffers key)))
              && not (Hashtbl.mem ctx.device.constant_buffer_cache key)
            then free_pool ctx.device ~pool_id:loc.pool_id)))

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
