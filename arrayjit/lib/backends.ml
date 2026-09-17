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
  let closed =
    ref []
    (* completed segment sizes, reversed *)
  in
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

(* gh-ocannl-489 liveness-based buffer aliasing: the config gates. Read per call (not cached in a
   lazy) so tests can flip them via the environment between compilation sections of one process. *)
let buffer_aliasing () = Utils.get_global_flag ~default:false ~arg_name:"buffer_aliasing"
let log_buffer_aliasing () = Utils.get_global_flag ~default:false ~arg_name:"log_buffer_aliasing"

(* gh-ocannl-489: liveness-aware companion of [plan_pool_segments] -- lays out a sequence of [(size,
   alignment, precision_class, live_span)] allocations into ONE pool where two allocations may
   occupy overlapping byte ranges iff both have a live span (planner-eligible), the spans are
   disjoint as closed intervals, and the precision classes are equal (sharing bytes across effective
   types would be C strict-aliasing UB in the single-procedure CPU backends). A [None] span means
   always-live: the node conflicts with everything, including other always-live nodes. Placement is
   greedy by decreasing size (ties broken by input order, keeping the layout deterministic): each
   item lands at the lowest suitably-aligned offset that avoids all conflicting placed items.
   Returns per-item byte offsets in input order plus the pool's total size, or [None] when the
   layout exceeds [cap] (the caller falls back to [plan_pool_segments]' bump packing, which segments
   at the cap). Pure, so the coloring is unit testable with synthetic sizes. *)
let plan_arena_offsets ~(cap : int) (items : (int * int * string * (int * int) option) list) :
    (int list * int) option =
  let arr = Array.of_list items in
  let n = Array.length arr in
  let order =
    List.sort (List.init n ~f:Fn.id) ~compare:(fun i j ->
        let si, _, _, _ = arr.(i) and sj, _, _, _ = arr.(j) in
        match compare_int sj si with 0 -> compare_int i j | c -> c)
  in
  let conflicts i j =
    let _, _, ci, spi = arr.(i) and _, _, cj, spj = arr.(j) in
    match (spi, spj) with
    | Some (lo1, hi1), Some (lo2, hi2) when String.equal ci cj -> not (hi1 < lo2 || hi2 < lo1)
    | _ -> true
  in
  let offsets = Array.create ~len:n 0 in
  let placed = ref [] in
  let total = ref 0 in
  List.iter order ~f:(fun i ->
      let size, align, _, _ = arr.(i) in
      let align_up off = if align <= 1 then off else (off + align - 1) / align * align in
      let ranges =
        List.filter_map !placed ~f:(fun j ->
            if conflicts i j then
              let sj, _, _, _ = arr.(j) in
              Some (offsets.(j), offsets.(j) + sj)
            else None)
        |> List.sort ~compare:(fun (a, _) (b, _) -> compare_int a b)
      in
      let off =
        List.fold ranges ~init:(align_up 0) ~f:(fun off (lo, hi) ->
            if off + size <= lo then off else max off (align_up hi))
      in
      offsets.(i) <- off;
      placed := i :: !placed;
      total := max !total (off + size));
  if !total > cap then None else Some (Array.to_list offsets, !total)

let size_in_bytes_of (key : Tn.t) =
  let prec = Lazy.force key.Tn.storage_prec in
  Array.fold (Lazy.force key.Tn.dims) ~init:1 ~f:( * ) * Ops.prec_in_bytes prec

(* gh-ocannl-489 liveness-based buffer aliasing, the per-compile planning half: live spans over the
   FINAL segment sequence (post-schedule, post-fission -- cross-nest merges and kernel cuts change
   which accesses can interleave, so pre-schedule spans would be unsound), filtered down to the
   aliasing-eligible nodes. Eligible = in-context working nodes that the routine writes before any
   read: not read-only, not read-before-write (excludes recurrent nodes and reliance on alloc-time
   zeros), not observable (host reads must stay valid), not host-initialized (live before the
   program), not constants, not the merge node. Callers run this BEFORE backend codegen: the
   candidates registered on [optimize_ctx.alias_candidates] make codegen drop the [restrict]
   qualifier on their kernel parameters (an actually-aliased [restrict] pair is a miscompile).
   Position granularity: statements on backends that synchronize between top-level statements (the C
   backends -- no hardware workgroup binding, parallel dispatches join), segments otherwise (GPU
   kernels lack grid-wide sync between statements). [lowered] is the whole-routine (pre-fission)
   code, whose traced store and placements decide eligibility; [segments] is the final kernel
   sequence in execution order. *)
let plan_alias_spans ~(name : string) ~(limits : hardware_limits) ~(lowered : Low_level.optimized)
    ~(segments : Low_level.optimized list) : (Tn.t, int * int) Base.Hashtbl.t option =
  if not (buffer_aliasing ()) then None
  else
    let stmt_serial = Option.is_none limits.max_threads_per_workgroup in
    match
      Low_level.buffer_access_spans ~stmt_serial
        (List.map segments ~f:(fun seg -> seg.Low_level.llc))
    with
    | None -> None
    | Some spans ->
        let plc = lowered.Low_level.optimize_ctx.placements in
        Hashtbl.filter_keys_inplace spans ~f:(fun tn ->
            match Hashtbl.find lowered.Low_level.traced_store tn with
            | None -> false
            | Some node ->
                Tn.Placements.is_in_context_force plc tn (Site "45:export-span")
                && (not node.Low_level.read_only)
                && (not node.Low_level.read_before_write)
                && (node.Low_level.zeroed_out || node.Low_level.has_assignment)
                (* Written but never consumed in-routine means the write is an EXPORT for later
                   routines (parameter initialization is the ubiquitous case) -- its lifetime
                   extends past this routine, so it must keep dedicated bytes. *)
                && node.Low_level.read_by_other
                && (not (Tn.is_observable tn))
                && (not (Host_inits.mem tn))
                && (not (Tn.Placements.known_constant plc tn))
                && not (Option.exists lowered.Low_level.merge_node ~f:(Tn.equal tn)));
        if Hashtbl.is_empty spans then None
        else (
          Hashtbl.iter_keys spans ~f:(Hash_set.add lowered.Low_level.optimize_ctx.alias_candidates);
          if log_buffer_aliasing () then
            List.iter
              (Hashtbl.to_alist spans
              |> List.sort ~compare:(fun (_, (a, _)) (_, (b, _)) -> compare_int a b))
              ~f:(fun (tn, (lo, hi)) ->
                Stdlib.Printf.eprintf "buffer aliasing: %s: candidate %s live [%d, %d]\n%!" name
                  (Tn.debug_name tn) lo hi);
          Some spans)

(* {2 The shared layout head (gh-ocannl-767)}

   [Raise_backend.compile] + [allocate_delta] on one side and [score_footprint] on the other must
   lay a routine out IDENTICALLY -- the scorer is the allocator's cost model (gh-ocannl-498), so
   scoring one layout and allocating another would let a plan report itself under budget while
   linking asks for a larger pool. The pieces both pipelines run are shared here rather than agreed
   on by comment: zero-sinking, the segment-store fold-back, the working/constants partition in
   canonical order, the per-node plan items, and the per-pool cap. *)

(* The per-pool ceiling: 4 GB for uint32 offsets unless [large_models] lifts it. *)
let pool_cap () = if Utils.settings.large_models then Int.max_value else 0x1_0000_0000

(* gh-ocannl-489 follow-up: with the liveness planner on, sink whole-node initializations toward
   their first use so live spans start there instead of at an up-front zeroing block (which nests
   the backprop gradient chain's intervals and defeats [plan_arena_offsets]). Reordering only --
   values are unchanged; gated to keep the planner-off pipeline byte-identical. Runs before
   scheduling, so segment cuts and cross-nest merges see the sunk order. *)
let maybe_sink_zeros (lowered : Low_level.optimized) : Low_level.optimized =
  if buffer_aliasing () then
    { lowered with Low_level.llc = Low_level.sink_zero_outs lowered.Low_level.llc }
  else lowered

(* Schedule ops applied per segment can CREATE tnodes the pre-fission store has never seen -- a
   hoisted [Stage] registers its packed-constant tile in the segment's filtered store (its placement
   lands in the shared lineage fork, but the allocator enumerates the traced store) -- so fold
   segment-added entries back into [into]. Pre-existing keys are shared mutable records (filtered
   slices alias them), so only genuinely new keys need adding. *)
let fold_segment_stores ~(into : Low_level.traced_store) (segments : Low_level.optimized list) :
    unit =
  List.iter segments ~f:(fun seg ->
      Hashtbl.iteri seg.Low_level.traced_store ~f:(fun ~key ~data ->
          if not (Hashtbl.mem into key) then Hashtbl.add_exn into ~key ~data))

(* Partition a traced store's in-context nodes into the (working, constants) layout groups, each in
   CANONICAL ([Tn.compare], i.e. uid) order -- the one order both the allocator and the scorer lay
   out (gh-ocannl-498): the planners are order-sensitive (the arena's greedy coloring breaks
   equal-size ties by input order, and bump packing's alignment padding and cap segmentation depend
   on the running offset -- sizes 4 then 64 at alignment 32 occupy 96 bytes, reversed 68). Uid order
   is deterministic and shared; pool ids are minted per segment before any placement, so nothing
   else depends on enumeration order. [skip] excludes nodes the caller already holds (the
   allocator's prior context); slice-alias views own no buffer and are excluded automatically
   ([is_in_context_force] returns false for them, gh-ocannl-293 293a) -- their materialized parent
   is laid out like any other node. *)
let partition_layout_groups ~(plc : Tn.Placements.t) ?(skip = fun (_ : Tn.t) -> false)
    (store : Low_level.traced_store) :
    (Tn.t * Low_level.traced_array) list * (Tn.t * Low_level.traced_array) list =
  let working = ref [] and constants = ref [] in
  Hashtbl.iteri store ~f:(fun ~key ~data:node ->
      if Tn.Placements.is_in_context_force plc key (Site "43:layout-group") && not (skip key) then
        if node.Low_level.read_only || Tn.Placements.known_constant plc key then
          constants := (key, node) :: !constants
        else working := (key, node) :: !working);
  let canonical l = List.sort !l ~compare:(fun (a, _) (b, _) -> Tn.compare a b) in
  (canonical working, canonical constants)

(* Within-pool offsets are padded to [Ops.buffer_alignment] (not just the element size) so that
   every node's buffer -- not only each pool's base -- is SIMD-aligned (gh-ocannl-164); ≤31 bytes of
   padding per node. *)
let layout_item (key : Tn.t) : int * int =
  ( size_in_bytes_of key,
    max (Ops.prec_in_bytes (Lazy.force key.Tn.storage_prec)) Ops.buffer_alignment )

let layout_items group = List.map group ~f:(fun (key, _) -> layout_item key)

(* The [plan_arena_offsets] input for a layout group under the given live [spans] (gh-ocannl-489):
   liveness-disjoint same-precision nodes may overlap. *)
let arena_items ~spans group =
  List.map group ~f:(fun ((key : Tn.t), _) ->
      let size, align = layout_item key in
      (size, align, Ops.prec_string (Lazy.force key.Tn.storage_prec), Hashtbl.find spans key))

(* gh-ocannl-498: the byte footprint a routine's placement vector implies, scored with the same
   liveness/arena machinery the allocator uses ([plan_alias_spans] + [plan_arena_offsets]) but
   without a device or a context. This is the cost side [Low_level.flip_candidates] does not carry:
   the recompute-cost bound says what inlining a node COSTS, this says what it SAVES.

   Scored over the routine's whole in-context node set, not a context's allocation delta, so the
   number depends only on the code and the placements -- the precondition for a deterministic budget
   selector ([Memory_budget.fit]) whose choices do not drift with how much of the graph a particular
   context has already allocated. It is therefore a MODEL of the peak, not a prediction of
   [Context.get_used_memory]: the real allocator skips nodes a prior context already holds, and pool
   bases are page-rounded by the driver.

   Every layout-relevant step is the shared helper the allocator pipeline also runs (the layout head
   above), so scorer/allocator agreement is structural rather than by comment. *)
let score_footprint ~(backend_name : string) ~(limits : hardware_limits)
    ~(static_indices : Indexing.static_symbol list) (lowered : Low_level.optimized) :
    Low_level.footprint =
  let lowered = maybe_sink_zeros lowered in
  let segments = Schedule.maybe_default_schedules ~backend_name ~limits ~static_indices lowered in
  let spans = plan_alias_spans ~name:"<footprint>" ~limits ~lowered ~segments in
  (* Score the union store, as the compile's own fold-back does -- but on a copy: scoring must not
     mutate the routine's own traced store. *)
  let store = Hashtbl.copy lowered.Low_level.traced_store in
  fold_segment_stores ~into:store segments;
  let plc = lowered.Low_level.optimize_ctx.placements in
  let working, constants = partition_layout_groups ~plc store in
  let cap = pool_cap () in
  let bump ~what group =
    let _, segment_sizes =
      plan_pool_segments ~cap ~what
        ~debug_name:(fun i -> Tn.debug_name (fst (List.nth_exn group i)))
        (layout_items group)
    in
    List.fold segment_sizes ~init:0 ~f:( + )
  in
  let fp_dedicated = bump ~what:"Backends.score_footprint" working in
  let fp_planned =
    Option.value_map spans ~default:0 ~f:(fun spans ->
        List.count working ~f:(fun (key, _) -> Hashtbl.mem spans key))
  in
  let arena =
    Option.bind spans ~f:(fun spans -> plan_arena_offsets ~cap (arena_items ~spans working))
  in
  let fp_working = match arena with Some (_, total) -> total | None -> fp_dedicated in
  let fp_constants = bump ~what:"Backends.score_footprint" constants in
  {
    fp_total = fp_working + fp_constants;
    fp_working;
    fp_constants;
    fp_dedicated;
    fp_planned;
    fp_nodes = List.length working + List.length constants;
  }

(* gh-ocannl-489: whether [tn]'s buffer shares bytes with another node's in [ctx_buffers] -- i.e.
   the liveness planner ([plan_arena_offsets] via [allocate_delta]) placed it at an overlapping
   range. Exact and layout-derived: bump-packed tenants of one pool have disjoint ranges and never
   trigger this. Used by the read guards: an aliased node's values are not preserved past its last
   in-routine read, so host reads, cross-device reads and later-routine inputs must fail loudly. *)
let buffer_overlaps (ctx_buffers : Backend_intf.ctx_buffers) tn (loc : Backend_intf.buffer_loc) :
    bool =
  let size = size_in_bytes_of tn in
  Map.existsi ctx_buffers ~f:(fun ~key ~data:(l : Backend_intf.buffer_loc) ->
      (not (Tn.equal key tn))
      && l.pool_id = loc.pool_id
      && l.offset < loc.offset + size
      && loc.offset < l.offset + size_in_bytes_of key)

let aliased_read_error ~what tn =
  raise
  @@ Utils.User_error
       (Printf.sprintf
          "%s: tensor node %s was buffer-aliased by the liveness memory planner (config \
           buffer_aliasing): its buffer is shared with other nodes and its values are not \
           preserved past its last read within the routine that computes it. Mark it as observable \
           or materialized-and-read (e.g. via Tnode.set_observable) before compiling that routine, \
           or disable buffer_aliasing."
          what (Tnode.debug_name tn))

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
   linked [code] expects. This is the "static verification in the right direction" of gh-ocannl-288:
   the transfer routine's context chains naturally into the consumer's link. *)
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
     backend's int-in/int-out API, and returns the [buffer_loc]. Phase-1 policy is one pool per
     tnode at offset 0 -- byte-for-byte equivalent to the old per-tnode allocation. [zero_init] asks
     for the slab to be zero-filled after it is minted (see the [memset_zero] below); a node the
     code first-touches ([zero_initialized_by_code]) does not need it. *)
  let allocate (device : _ Backend_intf.device) (tn : Tn.t) ~zero_init : Backend_intf.buffer_loc =
    let pool_id = device.next_pool_id in
    device.next_pool_id <- pool_id + 1;
    let prec = Lazy.force tn.Tn.storage_prec in
    (* Compute the byte size from dims*prec rather than forcing [tn.size_in_bytes], to keep the
       node's debug printout (and lazy-forcing behavior) byte-for-byte as before. *)
    let size_in_bytes =
      Array.fold (Lazy.force tn.Tn.dims) ~init:1 ~f:( * ) * Ops.prec_in_bytes prec
    in
    let mode = Option.map tn.Tn.memory_mode_intent ~f:fst in
    Backend.alloc_pool ?mode device ~pool_id ~size_in_bytes ~alignment:(Ops.prec_in_bytes prec);
    (* gh-ocannl-550: the OTHER shared allocation site — a [from_host] or [copy] whose destination
       node is not in the context yet allocates here, not through [allocate_delta]. Its slabs go
       into the same backend pool tables and are freed by the same context [finalize], so leaving
       them uncounted made the census silently underreport in data-loading and context-copy
       workflows. Not working-vs-constant: this path is a working buffer by construction (a host
       transfer's destination). *)
    Alloc_census.record_pool ~device_id:device.device_id ~pool_id ~constant:false ~size_in_bytes;
    if zero_init then Backend.memset_zero device ~pool_id ~offset:0 ~size_in_bytes;
    { pool_id; offset = 0 }

  let%track3_sexp to_host (ctx : Backend.context) (tn : Tn.t) (hosted : Ndarray.t) =
    match Map.find ctx.ctx_buffers tn with
    | Some loc ->
        if buffer_overlaps ctx.ctx_buffers tn loc then aliased_read_error ~what:"reading to host" tn;
        [%log "copying", Tn.debug_name tn, "at", (loc : Backend_intf.buffer_loc), "to host"];
        (* No cross-stream writer synchronization needed: multi-streaming was removed
           (gh-ocannl-341). Only one stream exists per device, so there are no concurrent
           cross-stream writes to wait for before this device-to-host copy. *)
        Resource_fault_injection.hit To_host_before_copy;
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
        (* A host write to an aliased buffer would be clobbered by the next run of the aliasing
           routine (and would clobber co-tenants meanwhile) -- surely a mistake; fail loudly. *)
        if buffer_overlaps ctx.ctx_buffers tn dst then
          aliased_read_error ~what:"writing from host" tn;
        (* No cross-stream reader synchronization needed: multi-streaming was removed
           (gh-ocannl-341). Only one stream exists per device, so there are no concurrent
           cross-stream readers to wait for before this host-to-device upload. *)
        [%log "copying", Tn.debug_name tn, "to", (dst : Backend_intf.buffer_loc), "from host"];
        Resource_fault_injection.hit From_host_before_copy;
        Backend.from_host ~dst:ctx ~dst_loc:dst hosted;
        update_writer_event ctx @@ Node tn;
        true
    | None -> false

  (* gh-ocannl-550: [allocate] roots a pool in the backend table, and the transfer that follows adds
     its location to the context only on success — so a failing upload leaves a pool no context can
     ever reach, and therefore no [Context.release] can reclaim. Frees the one pool this operation
     minted; unlike [allocate_delta]'s unwind there is no constant-cache involvement here (a
     transfer destination is a working buffer by construction), so this needs nothing beyond the
     free. *)
  let with_transfer_pool device (loc : Backend_intf.buffer_loc) ~f =
    match f () with
    | result -> result
    | exception exn ->
        let backtrace = Stdlib.Printexc.get_raw_backtrace () in
        (* Synchronize best-effort, but do not let a sticky worker/stream error suppress the only
           owner capable of releasing this unreachable fresh pool. Multidev keeps [dev_error] after
           reporting it, so retrying await can deterministically re-raise. *)
        (try
           Resource_fault_injection.hit Transfer_cleanup_before_await;
           Backend.await device
         with _ -> ());
        (try
           Option.iter Backend.free_pool ~f:(fun free -> free device ~pool_id:loc.pool_id);
           Alloc_census.forget_pool ~device_id:device.device_id ~pool_id:loc.pool_id
         with _ -> ());
        Stdlib.Printexc.raise_with_backtrace exn backtrace

  let%track3_sexp init_from_host (ctx : Backend.context) (tn : Tn.t) (hosted : Ndarray.t) =
    match Map.find ctx.ctx_buffers tn with
    | None ->
        (* No zero-init: we are immediately copying from host. *)
        let dst = allocate ctx.device tn ~zero_init:false in
        with_transfer_pool ctx.device dst ~f:(fun () ->
            Resource_fault_injection.hit Transfer_pool_allocated;
            [%log "copying", Tn.debug_name tn, "to", (dst : Backend_intf.buffer_loc), "from host"];
            Resource_fault_injection.hit From_host_before_copy;
            Backend.from_host ~dst:ctx ~dst_loc:dst hosted;
            update_writer_event ctx @@ Node tn;
            (* The upload may be asynchronous. Keep the fresh allocation inside the unwind guard
               until the stream has reported its result: otherwise an error first observed here
               escapes [Context.from_host] before the updated context is returned, and no caller can
               ever release [dst]. The outer Context-level await remains necessary for the
               existing-buffer branch. *)
            Resource_fault_injection.hit From_host_before_await;
            Backend.await ctx.device;
            Backend_intf.evolve_with_buffer ctx tn dst)
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
        if buffer_overlaps src.ctx_buffers tn s_loc then
          aliased_read_error ~what:"device_to_device source" tn;
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
                  let schedule = Task.Task { context_lifetime = (src, dst); description; work } in
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
        (* gh-ocannl-489: same source-read guard as [device_to_device] -- an aliased node's bytes
           are clobbered, so copying them into a fresh context would silently preserve the wrong
           value. *)
        if buffer_overlaps src.ctx_buffers tn s_loc then
          aliased_read_error ~what:"init_from_device source" tn;
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
            with_transfer_pool dst.device d_loc ~f:(fun () ->
                Resource_fault_injection.hit Transfer_pool_allocated;
                Backend.(
                  device_to_device tn ~into_merge_buffer:No ~dst_loc:(Some d_loc) ~dst
                    ~src_loc:s_loc ~src);
                update_writer_event dst @@ Node tn;
                [%log
                  "copying",
                  Tn.debug_name tn,
                  "from",
                  Backend.get_name src.device,
                  "to",
                  Backend.get_name dst.device];
                Backend_intf.evolve_with_buffer dst tn d_loc))

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
  (* Fork the lineage state (computations and placements) so this compile's decisions do not leak
     into the incoming context or into sibling compiles from the same context
     (docs/proposals/context-scoped-memory-modes.md). The forked state travels with the code and
     reaches the child context at link time. *)
  let optim_ctx = Low_level.copy_optimize_ctx optim_ctx in
  let name : string =
    Option.value_or_thunk name ~default:(fun () -> Assignments.get_name_exn asgns)
  in
  let unoptim_ll_source = Utils.output_to_build_file ~fname:(name ^ "-unoptimized.ll") in
  let ll_source = Utils.output_to_build_file ~fname:(name ^ ".ll") in
  let cd_source = Utils.output_to_build_file ~fname:(name ^ ".cd") in
  ( name,
    Assignments.lower optim_ctx ~unoptim_ll_source ~ll_source ~cd_source ~name
      (Indexing.bound_symbols bindings) asgns )

let%debug3_sexp verify_prior_context ~(plc : Tn.Placements.t) ~ctx_arrays ~from_prior_context : unit
    =
  Set.iter from_prior_context ~f:(fun tn ->
      if
        Tn.Placements.is_in_context_force plc tn (Site "42:prior-context-check")
        && (not (Option.is_some @@ Map.find ctx_arrays tn))
        (* Nodes with registered host initialization data (ndarray-backed literals, loaded tensors)
           self-initialize in this context at link time from [Host_inits] (gh-ocannl-333), so they
           need not be present in a prior context. *)
        && not (Host_inits.mem tn)
      then raise @@ Utils.User_error ("The linked context lacks node " ^ Tnode.debug_name tn))

(* Free one pool and drop its census entry -- the pair every cleanup path must keep together, or a
   freed slab stays counted (gh-ocannl-550). *)
let free_and_forget_pool device ~free_pool pool_id =
  free_pool device ~pool_id;
  Alloc_census.forget_pool ~device_id:device.device_id ~pool_id

(* The one pool-freeing fold behind the context [finalize] and [Raise_backend.free_delta]
   (gh-ocannl-767, unifying gh-ocannl-550's cleanup sites): frees the pools reachable through
   [ctx_buffers] that this context/delta owns. Deduped by [pool_id] -- one pool holds several nodes
   (gh-ocannl-344 bump packing / gh-ocannl-489 arenas), so the same id is reached through several
   keys, and a second visit would free an already-freed slab. Skips keys for which [owned_elsewhere]
   holds (the enclosing scope's buffers), and per-device constants -- compared by LOCATION, not key
   presence: a host upload may give a tnode a context-owned working location even when an earlier
   compile cached a CONSTANT location for the same key, and mistaking the working pool for that
   constant leaks it (gh-ocannl-571's transfer negative control). [skip_pool] lets [finalize] honor
   its retry ledger; [before_free]/[after_free] bracket each successful backend free (fault
   injection, ledger recording -- [after_free] runs only when [free_pool] returned, so a raising
   free is not recorded as done). *)
let free_owned_pools ~device ~free_pool ~owned_elsewhere ?(skip_pool = fun _ -> false)
    ?(before_free = fun _ -> ()) ?(after_free = fun _ -> ()) (ctx_buffers : ctx_buffers) : unit =
  Map.fold ctx_buffers
    ~init:(Set.empty (module Int))
    ~f:(fun ~key ~data:(loc : buffer_loc) freed ->
      if
        (not (owned_elsewhere key))
        && (not
              (Option.exists
                 (Hashtbl.find device.constant_buffer_cache key)
                 ~f:(equal_buffer_loc loc)))
        && (not (skip_pool loc.pool_id))
        && not (Set.mem freed loc.pool_id)
      then (
        before_free loc.pool_id;
        free_and_forget_pool device ~free_pool loc.pool_id;
        after_free loc.pool_id;
        Set.add freed loc.pool_id)
      else freed)
  |> (ignore : Set.M(Int).t -> unit)

(** Adds a scheduler and brings a lowered no-device backend on par with lowered device backends. *)
module Add_device
    (Add_scheduler : functor
      (Impl : For_add_scheduler)
      -> With_scheduler with type buffer_ptr = Impl.buffer_ptr)
    (Backend : Lowered_no_device_backend)
(* : Lowered_backend *) =
struct
  include Backend

  include Add_scheduler (struct
    include Backend
  end)

  type code = { lowered : Low_level.optimized; proc : Backend.procedure } [@@deriving sexp_of]

  type code_batch = {
    procs : Backend.procedure array;
    bindings : Indexing.unit_bindings;
        (** Kept for {!link_batch}: the batch's procedures share one set of static-index refs. *)
  }
  [@@deriving sexp_of]

  let compile ~(name : string) bindings lowered : code =
    let proc = compile ~name bindings lowered in
    { lowered; proc }

  let compile_batch ~names bindings lowereds : code_batch =
    let procs = compile_batch ~names bindings lowereds in
    { procs; bindings }

  (* The static-index launch parameters the CALLER writes through, and the snapshot that carries a
     dispatch's values to the kernel's own refs.

     [Indexing.apply] dereferences a launch's static indices inside the kernel task, and an
     asynchronous scheduler ([Schedulers.Multidev]) runs that task on a worker domain some time
     after [Context.run] returned -- by which point the caller's loop ([Train.sequential_loop], or
     any hand-written bind/run/rebind) has rebound them for the next dispatch. Sharing one ref
     between the two would launch the kernel on whichever index the host had reached, which is a
     silent wrong answer -- batches skipped and repeated, and any host schedule keyed on a static
     index (a learning rate on the step counter) read at the wrong point.

     So the caller gets refs of its own, and each dispatch snapshots them on the scheduling thread;
     the task copies the snapshot into the kernel's refs on the worker, in queue order, immediately
     before the launch reads them. [Context.check_launch_bindings] already validates the caller's
     values at dispatch time, so this is the launch honouring the contract that validation states.
     Under [Schedulers.Sync] the task runs inline, which makes it the same read at the same moment,
     one indirection later. *)
  let caller_bindings_and_snapshot (kernel_bindings : Indexing.lowered_bindings) =
    let paired = List.map kernel_bindings ~f:(fun (s, kernel) -> ((s, ref !kernel), kernel)) in
    let snapshot =
      match paired with
      | [] -> None
      | _ ->
          let cells = List.map paired ~f:(fun ((_, caller), kernel) -> (caller, kernel)) in
          Some
            (fun () ->
              let values = List.map cells ~f:(fun (caller, kernel) -> (kernel, !caller)) in
              fun () -> List.iter values ~f:(fun (kernel, value) -> kernel := value))
    in
    (List.map paired ~f:fst, snapshot)

  let link context (code : code) ctx_buffers : Indexing.lowered_bindings * Task.t =
    let runner_label = get_name context.device in
    let merge_buffer = context.device.merge_buffer in
    (* [resolve] is the device's backend-private [buffer_loc -> base] lookup; [link_compiled] does
       the (eager) [ctx_buffers] and (lazy) merge-buffer resolution with it, backend-side. The
       generic shared layer never sees a raw pointer. *)
    let resolve = resolve_pool context.device in
    let kernel_bindings, to_schedule =
      link_compiled ~merge_buffer ~resolve ~runner_label ctx_buffers code.proc
    in
    let bindings, snapshot = caller_bindings_and_snapshot kernel_bindings in
    let schedule =
      Task.enschedule ?snapshot ~schedule_task ~get_stream_name:get_name context.device to_schedule
    in
    (bindings, schedule)

  let link_batch context (code_batch : code_batch) ctx_buffers =
    let runner_label = get_name context.device in
    let merge_buffer = context.device.merge_buffer in
    let resolve = resolve_pool context.device in
    (* One shared bindings assoc for the whole batch (mirroring the CUDA/Metal backends): the
       batch's procedures — in particular the segment kernels of one fissioned routine — must see
       the same static-index refs, or setting a binding through the routine would reach only one of
       them. *)
    let kernel_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols code_batch.bindings) ~f:(fun s -> (s, ref 0))
    in
    (* One caller-side assoc for the whole batch too, and one snapshot closure: the segments of a
       fissioned routine are dispatched by a single [Context.run], so they all carry the same
       values. *)
    let bindings, snapshot = caller_bindings_and_snapshot kernel_bindings in
    let schedules =
      Array.map code_batch.procs ~f:(fun proc ->
          let bindings', to_schedule =
            link_compiled ~lowered_bindings:kernel_bindings ~merge_buffer ~resolve ~runner_label
              ctx_buffers proc
          in
          assert (phys_equal bindings' kernel_bindings);
          Task.enschedule ?snapshot ~schedule_task ~get_stream_name:get_name context.device
            to_schedule)
    in
    (bindings, schedules)

  (* CPU segment tasks are host closures the stream runner executes in order; the generic event
     chain degenerates to no-ops there, so there is nothing cheaper to provide. *)
  let sequence_segments _context ~name:_ ~bindings:_ ~uses_merge_buffer:_ _tasks = None

  (* Transfers take {!Backend_intf.buffer_loc} and resolve to the backend pointer here, against the
     device's private pool table -- the resolution is backend-side, not in the generic shared
     layer. *)
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
      | No, None -> invalid_arg "Add_device.device_to_device: missing dst_loc"
      | No, Some dst_loc ->
          let dst_ptr = resolve_pool dst.device dst_loc in
          fun () -> buffer_to_buffer ~dst:dst_ptr ~src:src_ptr ~size_in_bytes
      | Copy, _ ->
          fun () ->
            (* The merge buffer is the device's reserved single-tenant pool; grow it in place when a
               larger node arrives ([alloc_pool] overwrites the reserved pool-id entry). *)
            if s.merge_buffer_capacity < size_in_bytes then
              alloc_pool
                ?mode:(Option.map tn.Tnode.memory_mode_intent ~f:fst)
                s ~pool_id:merge_buffer_pool_id ~size_in_bytes
                ~alignment:(Ops.prec_in_bytes (Lazy.force tn.Tnode.storage_prec));
            let loc = { pool_id = merge_buffer_pool_id; offset = 0 } in
            buffer_to_buffer ~dst:(resolve_pool s loc) ~src:src_ptr ~size_in_bytes
    in
    let description =
      "device_to_device " ^ Tnode.debug_name tn ^ " dst " ^ get_name s ^ " src "
      ^ get_name src.device
    in
    schedule_task s (Task.Task { context_lifetime = (src, dst); description; work })
end

module Raise_backend (Device : Lowered_backend) : Backend = struct
  include Device
  include Add_buffer_retrieval_and_syncing (Device)

  type fissioned = { batch : Device.code_batch; count : int } [@@deriving sexp_of]

  type nonrec code = {
    from_prior_context : Set.M(Tnode).t;
    name : string;
    lowered : Low_level.optimized;
        (** The whole-routine lowered code, used for allocation and I/O analysis. When the routine
            is fissioned this is the pre-fission form; the per-segment kernels live in [proc]. *)
    proc : (code, fissioned) Either.t;
        (** [First]: a single kernel, as before fission. [Second]: the segment kernels of one
            routine, compiled as a batch and launched back-to-back on the routine's stream. *)
    expected_merge_node : Tnode.t option;
    alias_spans : (Tnode.t, int * int) Base.Hashtbl.t option;
        (** gh-ocannl-489: the liveness planner's per-compile facts -- live spans of the
            aliasing-eligible nodes over the final segment sequence (see
            {!Low_level.buffer_access_spans}). Consumed by [allocate_delta] at link time to lay the
            working pool out with overlapping offsets ([plan_arena_offsets]). [None] when the
            [buffer_aliasing] config is off or the code is opaque to the liveness fold. *)
  }
  [@@deriving sexp_of]

  let empty_optimize_ctx = Low_level.empty_optimize_ctx

  let%debug3_sexp compile optim_ctx ?name ?lowered_transform ?prelowered bindings
      (comp : Assignments.comp) : code =
    let (name : string), (lowered : Low_level.optimized) =
      match prelowered with
      | None -> lower_assignments optim_ctx ?name bindings comp.asgns
      | Some (lowered : Low_level.optimized) ->
          (* gh-ocannl-562 test seam: the caller supplies the lowering. Substituting only the
             codegen input ([lowered_transform]) is not enough to execute hand-built IR — the
             compile's own [code.lowered] is what drives I/O classification, liveness planning and
             the context-buffer delta, so the two would disagree. Here the supplied record IS
             [code.lowered], hence the analysis layer and the kernels see one and the same IR. The
             incoming lineage [optim_ctx] is deliberately unused: the record carries its own
             [optimize_ctx] (the caller's fork), which reaches the child context at link time. *)
          ( Option.value_or_thunk name ~default:(fun () -> Assignments.get_name_exn comp.asgns),
            lowered )
    in
    let lowered = maybe_sink_zeros lowered in
    let limits = Device.hardware_limits () in
    let lowereds =
      Schedule_outcome.tag Schedule_outcome.Transform (fun () ->
          match lowered_transform with
          | Some transform -> (
              (* The transform returns the routine's kernel segments: a singleton for a
                 whole-routine schedule, one element per segment for a fissioning one. *)
              match transform lowered with
              | [] -> invalid_arg "Backend.compile: lowered_transform returned an empty list"
              | segments -> segments)
          | None ->
              (* No explicit schedule: the default annotator parallelizes kernels it can prove safe
                 (docs/proposals/schedule-ir-optops.md §6) -- Grid x Workgroup on GPU backends,
                 pool-rendered Grid on CPU backends; the identity otherwise. Kernel fission may
                 split the routine into several kernels at cross-workgroup dependency edges; they
                 run back-to-back on the routine's stream (see [link]). *)
              Schedule.maybe_default_schedules ~backend_name:Device.name ~limits
                ~static_indices:(Indexing.bound_symbols bindings) lowered)
    in
    (* Per-compile launch-geometry trace (config [schedule_log_launches]): one line per segment with
       its grid/block dims — for diffing what two compiles of nominally identical code actually emit
       (PR #140 round 6). *)
    (if Lazy.force Schedule.log_launches then
       let n_segs = List.length lowereds in
       List.iteri lowereds ~f:(fun i seg ->
           let d = Low_level.launch_dims seg.Low_level.llc in
           Stdlib.Printf.eprintf
             "schedule: %s seg %d/%d grid=[%d;%d;%d] block=[%d;%d;%d] stmts=%d\n%!" name i n_segs
             d.grid.(0) d.grid.(1) d.grid.(2) d.block.(0) d.block.(1) d.block.(2)
             (List.length (Low_level.flat_lines [ seg.Low_level.llc ]))));
    (* gh-ocannl-489 liveness-based buffer aliasing: the per-compile planning half runs BEFORE
       [Device.compile] below, because the candidates it registers on
       [optimize_ctx.alias_candidates] make codegen drop the [restrict] qualifier on their kernel
       parameters (an actually-aliased [restrict] pair is a miscompile). *)
    let alias_spans : (Tn.t, int * int) Base.Hashtbl.t option =
      plan_alias_spans ~name ~limits ~lowered ~segments:lowereds
    in
    let (proc : (Device.code, fissioned) Either.t), (lowered : Low_level.optimized) =
      match lowereds with
      | [] -> assert false
      | [ single ] ->
          Schedule.check_hardware_limits_classified ~name ~limits single;
          let compiled =
            Schedule_outcome.tag Schedule_outcome.Backend_compile (fun () ->
                compile ~name bindings single)
          in
          (Either.First compiled, single)
      | segments ->
          let seg_names = List.mapi segments ~f:(fun i _ -> name ^ "__seg" ^ Int.to_string i) in
          List.iter2_exn seg_names segments ~f:(fun seg_name seg ->
              Schedule.check_hardware_limits_classified ~name:seg_name ~limits seg);
          let batch =
            Schedule_outcome.tag Schedule_outcome.Backend_compile (fun () ->
                compile_batch ~names:(Array.of_list seg_names) bindings (Array.of_list segments))
          in
          (* Keep the whole-routine (pre-fission) lowered code: context allocation and I/O analysis
             need the union footprint, and each segment's [optimized] carries only its filtered
             slice of the traced store -- so fold segment-added entries back in. *)
          fold_segment_stores ~into:lowered.Low_level.traced_store segments;
          (Either.Second { batch; count = List.length segments }, lowered)
    in
    (* Placements of all context nodes are settled by codegen (the [compile] just above), so this
       query resolves against the code's own lineage fork. The raw assignments over-approximate what
       the RESIDUAL schedule needs — a deferral-only routine reads nothing at run time, so linking
       it on a fresh context must not demand its deferred computations' leaves (gh-ocannl-611,
       review round 3). The reconciled traced store is exactly the final schedule's node registry,
       so it is the filter. *)
    let from_prior_context : Tn.t_set =
      let raw =
        Set.diff
          (Assignments.context_nodes ~plc:lowered.Low_level.optimize_ctx.placements comp.asgns)
          comp.embedded_nodes
        |> Set.filter ~f:(Hashtbl.mem lowered.Low_level.traced_store)
      in
      (* Splicing reconciles in the OTHER direction too (round 5): leaves reaching the routine only
         through an inlined cross-routine computation are absent from the raw assignments, yet their
         entry values are required — without them, [verify_prior_context] would accept a context
         where [allocate_delta] zero-fills the spliced inputs and the consumer silently computes
         with zeros. The reconciled interface's inputs are exactly the entry-value-matters nodes.
         Two deliberate bounds on the union: only inputs the raw assignments never MENTION are added
         — a mentioned node's prior-context status is already curated by [context_nodes]' exclusions
         (the random-seed and threefry nodes of init comps are mentioned yet deliberately not
         demanded) — and only for a routine CARRYING an assignments program, since
         [from_prior_context] is an assignments-layer promise: a hand-built [?prelowered] routine
         (empty comp) supplies its inputs through the context API after linking (the ll_test
         seed-then-run pattern). *)
      match comp.asgns with
      | Assignments.Noop -> raw
      | _ ->
          let (inputs, _), _ = Low_level.input_and_output_nodes lowered in
          let reads, writes = Assignments.collect_nodes_guess_output comp.asgns in
          let mentioned = Set.union reads writes in
          (* A RECONCILE-FLIPPED read-before-write input overrides the mention filter (round 6): a
             comp that writes a node AFTER consuming an inherited computation reading it mentions
             the node only as a write, yet the splice needs its entry value. The key is
             [spliced_rbw] — flips made against the FINAL code — not the raw flag: the raw analysis
             also marks every pure input read-before-write, and demanding those broke
             ndarray-literal and seed-node flows across the suite. *)
          let demanded =
            Set.filter inputs ~f:(fun tn ->
                (not (Set.mem mentioned tn)) || Set.mem lowered.Low_level.spliced_rbw tn)
          in
          Set.union raw demanded
    in
    {
      from_prior_context;
      name;
      lowered;
      proc;
      expected_merge_node = lowered.Low_level.merge_node;
      alias_spans;
    }

  (* gh-ocannl-344 Phase B/C: allocate a context's delta -- the in-context tnodes not already
     present in [context.ctx_buffers]. Working (non-constant) and constant/read-only nodes are EACH
     packed into pools sized to their group and bump-assigned increasing byte offsets, replacing the
     one-pool-per-tnode policy. Working pools belong to the context (freed at its [finalize]);
     constant pools are deduped per-device via [constant_buffer_cache] and outlive the context
     (freed at device teardown). Enumeration follows [traced_store] order so pool ids and offsets
     stay deterministic across runs. The per-pool 4 GB cap (uint32 offsets unless large_models) is
     enforced by {!Backend_utils.plan_pool_segments}. *)
  let%track3_sexp allocate_delta (context : context) ~name
      ~(alias_spans : (Tn.t, int * int) Base.Hashtbl.t option) (lowered : Low_level.optimized) :
      ctx_buffers =
    let traced_store = lowered.Low_level.traced_store in
    let device = context.device in
    let cap = pool_cap () in
    (* Pass 1: partition the delta into the canonical layout groups (the shared layout head),
       excluding nodes the prior context already holds. *)
    let working, constants =
      partition_layout_groups ~plc:lowered.Low_level.optimize_ctx.placements
        ~skip:(Map.mem context.ctx_buffers) traced_store
    in
    let ctx_buffers = ref context.ctx_buffers in
    (* Pack a group of (key, node) into one or more pools, segmenting at the cap. [register] decides
       how the resulting [buffer_loc] is recorded (directly into [ctx_buffers] for working nodes, or
       deduped through [constant_buffer_cache] for constants). [base_pool_id] of each segment is a
       freshly minted [next_pool_id]; offsets and pool sizes come from the pure planner. *)
    let place (key, node) ~pool_id ~offset ~register =
      let size_in_bytes = size_in_bytes_of key in
      let alloc () : buffer_loc =
        let host_init = Host_inits.find key in
        (* Zero-initialize unless the node will be copied from host immediately, or the lowered code
           already zero-initializes it. *)
        let zero_init = not (Option.is_some host_init || node.Low_level.zero_initialized_by_code) in
        if zero_init then memset_zero device ~pool_id ~offset ~size_in_bytes;
        let loc = { pool_id; offset } in
        Option.iter host_init ~f:(fun nd ->
            let nd = Lazy.force nd in
            (* Interval analysis, Phase B: [Host_inits] uploads are host writes; propose the scanned
               bounds lazily -- here, where the buffer is forced at link/upload time -- so [Reshape]
               inits wait for shape and padding inference as designed. *)
            Tnode.propose_bounds_from_host key nd;
            Device.from_host ~dst:context ~dst_loc:loc nd);
        loc
      in
      register key ~alloc
    in
    (* gh-ocannl-550: the shared seam is where a pool's byte size is known, so it is where the
       per-class census records it. [~constant] separates the two groups below, whose lifetimes
       differ: a working pool belongs to the context and dies with it, a constant pool is deduped
       per device and outlives it. *)
    (* What THIS call minted, so a failure part way through can give it back (gh-ocannl-550,
       round-four review). [with_delta] in [link] covers failures after the delta is returned; it
       cannot cover a failure inside this function, where the caller never receives [ctx_buffers] and
       the slabs already allocated are rooted with nothing to reach them. Pool ids are fresh from
       [device.next_pool_id], so every id here is unambiguously this call's. *)
    let minted = ref [] in
    let cache_inserts = ref [] in
    let alloc_pool_counted ~constant device ~pool_id ~size_in_bytes ~alignment =
      alloc_pool device ~pool_id ~size_in_bytes ~alignment;
      minted := pool_id :: !minted;
      Alloc_census.record_pool ~device_id:device.device_id ~pool_id ~constant ~size_in_bytes;
      Resource_fault_injection.hit Delta_pool_allocated
    in
    let unwind_partial_delta () =
      (* The uploads scheduled above are asynchronous, so the pools must not be freed under them. *)
      (try Device.await device with _ -> ());
      (* Constant-cache entries this call inserted point INTO the pools about to go, so they have to
         come out first or a later context would resolve a freed slab. Entries that were already
         there belong to earlier compiles and are untouched. *)
      List.iter !cache_inserts ~f:(Hashtbl.remove device.constant_buffer_cache);
      Option.iter free_pool ~f:(fun free_pool ->
          List.dedup_and_sort !minted ~compare:Int.compare
          |> List.iter ~f:(free_and_forget_pool device ~free_pool))
    in
    let pack ?arena ~constant (group : (Tn.t * Low_level.traced_array) list)
        ~(register : Tn.t -> alloc:(unit -> buffer_loc) -> unit) : unit =
      if not (List.is_empty group) then begin
        (* [group] arrives in canonical (uid) order from [partition_layout_groups] -- the order
           [score_footprint] scores, which the order-sensitive planners must see identically
           (gh-ocannl-498; the rationale lives on the shared layout head). Pool ids are minted per
           segment before any placement, and registration is order-independent. *)
        let items = layout_items group in
        (* gh-ocannl-489: with a liveness plan (the working group under [buffer_aliasing]), lay the
           group out as one arena where liveness-disjoint same-precision nodes overlap. Falls back
           to bump packing when the arena would exceed the per-pool cap. *)
        let arena_layout =
          Option.bind arena ~f:(fun spans -> plan_arena_offsets ~cap (arena_items ~spans group))
        in
        match arena_layout with
        | Some (offsets, total) ->
            let pool_id = device.next_pool_id in
            device.next_pool_id <- pool_id + 1;
            let alignment =
              List.fold group ~init:1 ~f:(fun a (key, _) ->
                  max a (Ops.prec_in_bytes (Lazy.force key.Tn.storage_prec)))
            in
            alloc_pool_counted ~constant device ~pool_id ~size_in_bytes:total ~alignment;
            List.iter2_exn group offsets ~f:(fun entry offset ->
                place entry ~pool_id ~offset ~register);
            if log_buffer_aliasing () then
              let _, bump_sizes =
                plan_pool_segments ~cap ~what:"Backends.allocate_delta"
                  ~debug_name:(fun i -> Tn.debug_name (fst (List.nth_exn group i)))
                  items
              in
              let dedicated = List.fold bump_sizes ~init:0 ~f:( + ) in
              let planned =
                Option.value_map arena ~default:0 ~f:(fun spans ->
                    List.count group ~f:(fun (key, _) -> Hashtbl.mem spans key))
              in
              Stdlib.Printf.eprintf
                "buffer aliasing: %s: working pool %d bytes, dedicated packing %d bytes (%.1f%% \
                 saved; %d/%d nodes liveness-planned)\n\
                 %!"
                name total dedicated
                (100. *. Float.of_int (dedicated - total) /. Float.of_int (max 1 dedicated))
                planned (List.length group)
        | None ->
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
            (* Allocate each segment's slab, padding alignment to the max element precision it
               holds. *)
            let seg_align = Array.map seg_pool_ids ~f:(fun _ -> ref 1) in
            List.iter2_exn group assignments ~f:(fun (key, _) (seg, _) ->
                let a = Ops.prec_in_bytes (Lazy.force key.Tn.storage_prec) in
                if a > !(seg_align.(seg)) then seg_align.(seg) := a);
            Array.iteri seg_pool_ids ~f:(fun seg (pool_id, size_in_bytes) ->
                alloc_pool_counted ~constant device ~pool_id ~size_in_bytes
                  ~alignment:!(seg_align.(seg)));
            (* Place each node at its planned (segment, offset). *)
            List.iter2_exn group assignments ~f:(fun entry (seg, offset) ->
                let pool_id, _ = seg_pool_ids.(seg) in
                place entry ~pool_id ~offset ~register)
      end
    in
    (* Pass 2a: working delta -> context-owned pool(s), recorded directly. Only this group is
       liveness-planned (gh-ocannl-489): constants are deduped per-device and outlive the context,
       so their lifetimes are not routine intervals. *)
    let passes () =
      pack ?arena:alias_spans ~constant:false working ~register:(fun key ~alloc ->
          ctx_buffers := Map.add_exn !ctx_buffers ~key ~data:(alloc ()));
      (* Pass 2b: constants / read-only -> per-device constant pool(s). Constants already allocated
         on this device (a hit in [constant_buffer_cache], possibly from another context tree)
         resolve directly and are excluded from the new slab, so the freshly-minted constant pool
         holds exactly this device's genuinely-new constants -- no wasted holes. The remaining new
         constants pack into one constant pool (or more, past the cap), deduped into the cache.
         Constant pools outlive the context and are skipped by context [finalize] (freed at device
         teardown). *)
      let new_constants = ref [] in
      List.iter constants ~f:(fun (key, node) ->
          match Hashtbl.find device.constant_buffer_cache key with
          | Some data -> ctx_buffers := Map.add_exn !ctx_buffers ~key ~data
          | None -> new_constants := (key, node) :: !new_constants);
      pack ~constant:true (List.rev !new_constants) ~register:(fun key ~alloc ->
          let data =
            Hashtbl.find_or_add device.constant_buffer_cache key ~default:(fun () ->
                cache_inserts := key :: !cache_inserts;
                alloc ())
          in
          ctx_buffers := Map.add_exn !ctx_buffers ~key ~data)
    in
    (match passes () with
    | () -> ()
    | exception exn ->
        let backtrace = Stdlib.Printexc.get_raw_backtrace () in
        (* Best-effort: giving the slabs back must not replace the allocation failure the caller has
           to classify (an out-of-memory decline being the whole point). *)
        (try unwind_partial_delta () with _ -> ());
        Stdlib.Printexc.raise_with_backtrace exn backtrace);
    !ctx_buffers

  (* gh-ocannl-550: [allocate_delta] runs BEFORE the backend link, and the pool table roots whatever
     it allocated — so a link that raises (HIP's scratch-budget validator, a driver refusal, an
     aliased-read rejection) used to leave that routine's pools behind with no context through which
     anyone could ever release them. Those are the [Backend_link] declines an autotune search
     absorbs, so they accumulated exactly like the candidates that succeeded.

     Frees the delta of [ctx_buffers] against [context] through [free_owned_pools], i.e. the same
     rule the context [finalize] applies — this stands in for it on the path where no context was
     ever built. *)
  let free_delta context (ctx_buffers : ctx_buffers) =
    (* Sync first, for the same reason [unwind_partial_delta] and the context [finalize] do
       (gh-ocannl-550, round-five review): [allocate_delta] queues [Host_inits] uploads through
       [Device.from_host], so a delta being discarded after a failed link can still have writes in
       flight — and freeing the slab under them is device corruption, on a path that is otherwise a
       contained candidate decline the search carries on from. Best-effort: the device may already
       be refusing work, and that must not replace the link failure the caller has to classify. *)
    (try Device.await context.device with _ -> ());
    Option.iter free_pool ~f:(fun free_pool ->
        free_owned_pools ~device:context.device ~free_pool
          ~owned_elsewhere:(Map.mem context.ctx_buffers) ctx_buffers)

  (* Runs [f] on a freshly allocated delta, freeing that delta if [f] raises. Everything after the
     allocation belongs inside: a failure past [make_child] discards the child too, so its pools are
     just as unreachable as if the child had never existed. *)
  let with_delta context ctx_buffers ~f =
    match
      Resource_fault_injection.hit Link_after_delta;
      f ()
    with
    | result -> result
    | exception exn ->
        let backtrace = Stdlib.Printexc.get_raw_backtrace () in
        (* Best-effort, and it must not replace the link failure the caller has to classify. *)
        (try free_delta context ctx_buffers with _ -> ());
        Stdlib.Printexc.raise_with_backtrace exn backtrace

  let%debug3_sexp link context (code : code) =
    verify_prior_context ~plc:code.lowered.Low_level.optimize_ctx.placements
      ~ctx_arrays:context.ctx_buffers ~from_prior_context:code.from_prior_context;
    (* Static merge-buffer verification "in the right direction" (gh-ocannl-288): the linked context
       carries the merge-buffer node of the producing [device_to_device] transfer routine; a
       mismatch with the consuming code raises here, at link time, before any schedule runs. *)
    check_merge_buffer_static ~merge_buffer_node:context.merge_buffer_node
      ~code_node:code.expected_merge_node;
    let (inputs, outputs), merge_buffer_input = Low_level.input_and_output_nodes code.lowered in
    let ctx_buffers =
      allocate_delta context ~name:code.name ~alias_spans:code.alias_spans code.lowered
    in
    with_delta context ctx_buffers ~f:(fun () ->
        (* gh-ocannl-489: a routine reading a node whose buffer an earlier routine of this lineage
           aliased would read clobbered values -- fail at link time, before any schedule runs.
           Writes (outputs) are allowed: the aliasing routine rewrites everything it reads on each
           run. This code's own aliased nodes are never its inputs (aliasing-eligible nodes are not
           read-before-write), so the check only fires on genuinely cross-routine reads. *)
        Set.iter inputs ~f:(fun tn ->
            Option.iter (Map.find ctx_buffers tn) ~f:(fun loc ->
                if buffer_overlaps ctx_buffers tn loc then
                  aliased_read_error ~what:("linking " ^ code.name ^ ", input") tn));
        let optimize_ctx = code.lowered.Low_level.optimize_ctx in
        let bindings, schedule =
          match code.proc with
          | Either.First single -> link context single ctx_buffers
          | Either.Second { batch; count } ->
              (* Fissioned routine: every segment kernel links against the routine's one ctx_buffers
                 delta and shares the one bindings assoc; the combined task launches the segments in
                 order on the routine's stream, whose FIFO ordering supplies the grid-wide
                 synchronization at each segment boundary (the same contract consecutive routines on
                 one stream already rely on). *)
              let bindings, tasks = link_batch context batch ctx_buffers in
              let tasks = Array.to_list tasks in
              assert (List.length tasks = count);
              (* Device-side ordering at each segment boundary: the cut is where the kernel-internal
                 code lacks grid-wide synchronization, so the stream must provide it. Queue FIFO
                 alone is not enough on Metal — command buffers over untracked resources may overlap
                 in execution (caught by test_random_histograms). Backends that can order the batch
                 device-side more cheaply (one Metal command buffer with a serial compute pass)
                 provide [sequence_segments]; the fallback chains an event per boundary: schedule
                 each next segment to wait for all work enqueued so far. No host blocking. *)
              let schedule =
                match
                  sequence_segments context ~name:code.name ~bindings
                    ~uses_merge_buffer:(Option.is_some code.expected_merge_node)
                    tasks
                with
                | Some fused -> fused
                | None ->
                    Task.Task
                      {
                        context_lifetime = tasks;
                        description = "fissioned segments of " ^ code.name;
                        work =
                          (fun () ->
                            List.iteri tasks ~f:(fun i t ->
                                if i > 0 then will_wait_for context (all_work context.device);
                                Task.run t));
                      }
              in
              (bindings, schedule)
        in
        let context = make_child ~ctx_buffers ~optimize_ctx context in
        let schedule =
          Task.prepend schedule ~work:(fun () ->
              check_merge_buffer context.device ~code_node:code.expected_merge_node)
        in
        sync_routine
          { context; schedule; bindings; name = code.name; inputs; merge_buffer_input; outputs })
end

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

let finalize (type dev runner event)
    (module Backend : Backend with type dev = dev and type runner = runner and type event = event)
    (ctx : Backend.context) : unit =
  (* The flag means "this context's pools have been freed", NOT "a free was attempted" — so cleanup
     that RAISES resets it (gh-ocannl-550). [Backend.await] is the realistic raiser: a device still
     reporting an asynchronous error, or a dead worker domain. Left set, every later release of this
     context would be a silent no-op and its pools would stay rooted for the process — restoring
     exactly the unbounded growth this exists to end, and on the failure paths where it matters
     most, since the tuner catches a failed release and carries on with the next candidate or arm. A
     retry skips the pool ids whose frees already returned successfully; backend frees are
     idempotent too, but relying on that would still call a raw deallocator twice. *)
  let cleanup () =
    Option.iter Backend.free_pool ~f:(fun free_pool ->
        Resource_fault_injection.hit Finalize_before_await;
        Backend.await ctx.device;
        free_owned_pools ~device:ctx.device ~free_pool
          ~owned_elsewhere:(fun key ->
            Option.exists ctx.parent ~f:(fun pc -> Map.mem pc.ctx_buffers key))
          ~skip_pool:(Set.mem ctx.released_pool_ids)
          ~before_free:(fun _ -> Resource_fault_injection.hit Finalize_before_free)
          ~after_free:(fun pool_id ->
            ctx.released_pool_ids <- Set.add ctx.released_pool_ids pool_id)
          ctx.ctx_buffers)
  in
  if Atomic.compare_and_set ctx.finalized false true then
    match cleanup () with
    | () -> Alloc_census.count_context_released ()
    | exception exn ->
        Atomic.set ctx.finalized false;
        raise exn

(* {2 The implemented backends, as singletons}

   One instantiation per backend for the whole process, so backend context types are nameable
   ([Cc_b.context], ...) and two independently-created contexts on the same backend unify -- the
   precondition for [Context.copy] dispatching to backend-specific [device_to_device] via
   {!wrapped_context}. The retired [fresh_backend] applied these functors per call to isolate
   tnode-keyed backend caches between tests (reinitialization reuses tnode ids); tnode identity is
   now the never-reused [Tnode.uid], so stale cache entries cannot alias fresh nodes and the
   isolation is unnecessary. Instantiating a module here must not touch any driver or hardware:
   device backends keep discovery/driver-init lazy, forced at first device use (cudajit is a depopt
   -- the library being installed does not imply a usable driver -- and a CPU-only run must not
   depend on GPU runtimes). On platforms without the corresponding library the dune [select]ed
   [Lowered_backend_missing] stub is instantiated instead, likewise harmless at init and raising on
   use. Either failure mode surfaces at [get_device], where [Context.auto]'s fallback catches it per
   call, as with the retired per-call instantiation. *)

module Cc_b : Backend = Make_device_backend_from_lowered (Schedulers.Sync) (Cc_backend)
module Multidev_cc_b : Backend = Make_device_backend_from_lowered (Schedulers.Multidev) (Cc_backend)
module Cuda_b : Backend = Raise_backend (Cuda_backend_impl.Impl : Lowered_backend)
module Hip_b : Backend = Raise_backend (Hip_backend_impl.Impl : Lowered_backend)
module Metal_b : Backend = Raise_backend (Metal_backend_impl.Impl : Lowered_backend)

type backend = Cc | Multidev_cc | Cuda | Hip | Metal [@@deriving sexp, equal, enumerate]

let get_backend ?backend_name () =
  match
    Option.value_or_thunk backend_name ~default:(fun () ->
        Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
    |> String.lowercase
  with
  (* "sync_cc" and "multicore_cc" are accepted as deprecated aliases of the renamed backends. *)
  | "cc" | "sync_cc" -> Cc
  | "multidev_cc" | "multicore_cc" -> Multidev_cc
  | "cuda" -> Cuda
  | "hip" -> Hip
  | "metal" -> Metal
  | backend -> invalid_arg [%string "Backends.get_backend: unknown backend %{backend}"]

let backend_name = function
  | Cc -> "cc"
  | Multidev_cc -> "multidev_cc"
  | Cuda -> "cuda"
  | Hip -> "hip"
  | Metal -> "metal"

type ('dev, 'runner, 'event) backend_module =
  (module Backend with type dev = 'dev and type runner = 'runner and type event = 'event)
(** A backend singleton's module at known type components -- the package type every generic
    dispatcher below is written against. *)

type wrapped_context =
  | Cc_ctx of Cc_b.context
  | Multidev_cc_ctx of Multidev_cc_b.context
  | Cuda_ctx of Cuda_b.context
  | Hip_ctx of Hip_b.context
  | Metal_ctx of Metal_b.context

type ('dev, 'runner, 'event) backend_impl = {
  bi_backend : backend;
  bi_module : ('dev, 'runner, 'event) backend_module;
  bi_wrap : ('dev, 'runner, 'event) Backend_intf.context -> wrapped_context;
}
(** Everything a {!wrapped_context} constructor statically implies: which backend it is, its
    singleton module at that constructor's type components, and the constructor itself (to rebuild
    the wrapper around a derived context). One record per backend, so the correspondence
    [Cc_ctx <-> Cc <-> Cc_b] is written once instead of once per dispatcher. *)

type packed_impl = Packed_impl : ('dev, 'runner, 'event) backend_impl -> packed_impl

(** A wrapped context with its type components recovered as locally abstract types. *)
type unwrapped =
  | Unwrapped :
      ('dev, 'runner, 'event) backend_impl * ('dev, 'runner, 'event) Backend_intf.context
      -> unwrapped

(** Two wrapped contexts correlated: [Same_backend] recovers the type equality that lets a
    same-backend transfer dispatch to the backend's [device_to_device]. *)
type paired =
  | Same_backend :
      ('dev, 'runner, 'event) backend_impl
      * ('dev, 'runner, 'event) Backend_intf.context
      * ('dev, 'runner, 'event) Backend_intf.context
      -> paired
  | Cross_backend

let cc_impl = { bi_backend = Cc; bi_module = (module Cc_b); bi_wrap = (fun c -> Cc_ctx c) }

let multidev_cc_impl =
  {
    bi_backend = Multidev_cc;
    bi_module = (module Multidev_cc_b);
    bi_wrap = (fun c -> Multidev_cc_ctx c);
  }

let cuda_impl = { bi_backend = Cuda; bi_module = (module Cuda_b); bi_wrap = (fun c -> Cuda_ctx c) }
let hip_impl = { bi_backend = Hip; bi_module = (module Hip_b); bi_wrap = (fun c -> Hip_ctx c) }

let metal_impl =
  { bi_backend = Metal; bi_module = (module Metal_b); bi_wrap = (fun c -> Metal_ctx c) }

(* The matches over the closed disjunctions -- one per question anyone asks of them: which impl a
   backend constructor names, which impl and context a wrapped context carries, and whether two
   wrapped contexts carry the same one. Every dispatcher below goes through these, so a new backend
   adds arms here and nowhere else. *)

let impl_of_backend : backend -> packed_impl = function
  | Cc -> Packed_impl cc_impl
  | Multidev_cc -> Packed_impl multidev_cc_impl
  | Cuda -> Packed_impl cuda_impl
  | Hip -> Packed_impl hip_impl
  | Metal -> Packed_impl metal_impl

let unwrap : wrapped_context -> unwrapped = function
  | Cc_ctx c -> Unwrapped (cc_impl, c)
  | Multidev_cc_ctx c -> Unwrapped (multidev_cc_impl, c)
  | Cuda_ctx c -> Unwrapped (cuda_impl, c)
  | Hip_ctx c -> Unwrapped (hip_impl, c)
  | Metal_ctx c -> Unwrapped (metal_impl, c)

let pair_contexts (src : wrapped_context) (dst : wrapped_context) : paired =
  match (src, dst) with
  | Cc_ctx s, Cc_ctx d -> Same_backend (cc_impl, s, d)
  | Multidev_cc_ctx s, Multidev_cc_ctx d -> Same_backend (multidev_cc_impl, s, d)
  | Cuda_ctx s, Cuda_ctx d -> Same_backend (cuda_impl, s, d)
  | Hip_ctx s, Hip_ctx d -> Same_backend (hip_impl, s, d)
  | Metal_ctx s, Metal_ctx d -> Same_backend (metal_impl, s, d)
  | (Cc_ctx _ | Multidev_cc_ctx _ | Cuda_ctx _ | Hip_ctx _ | Metal_ctx _), _ -> Cross_backend

let backend_module (b : backend) : (module Backend) =
  match impl_of_backend b with
  | Packed_impl i ->
      let (module B) = i.bi_module in
      (module B)

let wrapped_backend w = match unwrap w with Unwrapped (i, _) -> i.bi_backend

let make_context ?(ordinal = 0) backend =
  match impl_of_backend backend with
  | Packed_impl i ->
      let (module B) = i.bi_module in
      i.bi_wrap (B.make_context ~optimize_ctx:(B.empty_optimize_ctx ()) (B.get_device ~ordinal))

type 'a ctx_op = {
  f :
    'dev 'runner 'event.
    ('dev, 'runner, 'event) backend_module ->
    ('dev, 'runner, 'event) Backend_intf.context ->
    ('dev, 'runner, 'event) Backend_intf.context * 'a;
}
(** A context-transforming backend operation, polymorphic over the backend's type components so
    {!with_backend} can rebuild the same {!wrapped_context} constructor around the result. *)

let with_backend (w : wrapped_context) { f } =
  match unwrap w with
  | Unwrapped (i, c) ->
      let c, r = f i.bi_module c in
      (i.bi_wrap c, r)

type 'a ctx_query = {
  q :
    'dev 'runner 'event.
    ('dev, 'runner, 'event) backend_module -> ('dev, 'runner, 'event) Backend_intf.context -> 'a;
}
(** A read-only backend operation; like {!ctx_op} but leaves the context untouched. *)

let query (w : wrapped_context) { q } = match unwrap w with Unwrapped (i, c) -> q i.bi_module c
