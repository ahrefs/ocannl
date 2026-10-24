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
(* The aliasing-eligibility half of [plan_alias_spans] that depends on the node and the
   whole-routine lowering only, never on the kernel sequence: a node failing it gets no live span in
   any schedule, so the arena keeps it apart from every other node. Shared with the device-memory
   floor ([link_floor], gh-ocannl-1554), which counts such nodes in full. *)
let alias_eligible ~(lowered : Low_level.optimized) (tn : Tn.t) : bool =
  let plc = lowered.Low_level.optimize_ctx.placements in
  match Hashtbl.find lowered.Low_level.traced_store tn with
  | None -> false
  | Some node ->
      Tn.Placements.is_in_context_force plc tn (Site "45:export-span")
      && (not node.Low_level.read_only)
      && (not node.Low_level.read_before_write)
      && (node.Low_level.zeroed_out || node.Low_level.has_assignment)
      (* Written but never consumed in-routine means the write is an EXPORT for later routines
         (parameter initialization is the ubiquitous case) -- its lifetime extends past this
         routine, so it must keep dedicated bytes. *)
      && node.Low_level.read_by_other
      && (not (Tn.is_observable tn))
      && (not (Host_inits.mem tn))
      && (not (Tn.Placements.known_constant plc tn))
      && not (Option.exists lowered.Low_level.merge_node ~f:(Tn.equal tn))

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
        Hashtbl.filter_keys_inplace spans ~f:(alias_eligible ~lowered);
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

(* gh-ocannl-1257: on GPU backends, a covering zero whose node is next touched by an accumulation
   that can open from zero itself is dropped, and that accumulation rewritten
   ({!Low_level.elide_covered_zeros}) -- the composed attention backward's [(b, h, s, s)] gradients
   made the zero kernel the step's dominant zeroing cost, all of it bandwidth spent writing zeros
   that the first accumulation then read back. It runs on the final kernel list, after whatever
   transform produced it (the default schedules or a tuned candidate's), so neither the schedule
   transforms nor the search ever see the rewritten accumulations, and a tuned candidate is timed in
   the form it ships in. Value-identical by construction, so no numerics gate: the key is the
   kill-switch, read per call like the aliasing gates. CPU backends keep their zero segments, which
   render as [memset], and so does a hand-built [?prelowered] routine (see [compile]). *)
let elide_covered_zeros () = Utils.get_global_flag ~default:true ~arg_name:"gpu_elide_covered_zeros"

let maybe_elide_zeros ~backend_name (segments : Low_level.optimized list) : Low_level.optimized list
    =
  if not (Schedule.backend_is_gpu backend_name && elide_covered_zeros ()) then segments
  else
    let elided =
      List.filter_map
        (List.zip_exn segments
           (Low_level.elide_covered_zeros (List.map segments ~f:(fun seg -> seg.Low_level.llc))))
        ~f:(fun (seg, llc) ->
          Option.map llc ~f:(fun llc ->
              if phys_equal llc seg.Low_level.llc then seg
              else
                (* The nodes whose only touch was a dropped zero leave the segment's store too
                   (gh-ocannl-1391): every store entry the code no longer touches stays a kernel
                   parameter, and parameter names are disambiguated over the code alone, so a folded
                   zero segment that lost two same-labelled zeros named two parameters alike --
                   harmless inside the routine's batch, whose other kernels still touch them, and a
                   compile error for the segment compiled on its own. *)
                let traced_store =
                  if Low_level.has_staged_compilation llc then seg.Low_level.traced_store
                  else
                    Hashtbl.filteri seg.Low_level.traced_store ~f:(fun ~key ~data:_ ->
                        Low_level.code_touches_tn key llc
                        || not (Low_level.code_touches_tn key seg.Low_level.llc))
                in
                { seg with Low_level.llc; traced_store }))
    in
    if List.is_empty elided then segments else elided

(* gh-ocannl-1006: the kernel segments a compile SHIPPED, recorded where they are decided and read
   back by [Context.compile_outcome], which brackets every compile with {!with_segments_census} and
   stores the result on the routine. The same bracket pattern as [C_syntax.with_census], for the
   same reason: which kernels a routine runs is a property of the compiled routine, and it cannot be
   recovered from outside -- a tuned routine's segments come out of a [lowered_transform] only the
   search saw, so re-lowering its computation reproduces the default pipeline instead. Domain-local,
   like the codegen censuses: backends are process-wide singletons, and two domains compiling at
   once must not read or restore each other's segment lists. *)
let segments_census =
  let key : Low_level.optimized list option ref Stdlib.Domain.DLS.key =
    Stdlib.Domain.DLS.new_key (fun () -> ref None)
  in
  fun () -> Stdlib.Domain.DLS.get key

let record_segments segments = segments_census () := Some segments

let with_segments_census f =
  let cell = segments_census () in
  let saved = !cell in
  cell := None;
  match f () with
  | result ->
      let recorded = !cell in
      cell := saved;
      (result, Option.value recorded ~default:[])
  | exception exn ->
      cell := saved;
      raise exn

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
        if
          (not key.Tn.context_owned)
          && (node.Low_level.read_only || Tn.Placements.known_constant plc key)
        then constants := (key, node) :: !constants
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

(* The arena's precision class: nodes of different classes never share bytes ([plan_arena_offsets]),
   whatever their spans. *)
let alias_class (key : Tn.t) = Ops.prec_string (Lazy.force key.Tn.storage_prec)

(* The [plan_arena_offsets] input for a layout group under the given live [spans] (gh-ocannl-489):
   liveness-disjoint same-precision nodes may overlap. *)
let arena_items ~spans group =
  List.map group ~f:(fun ((key : Tn.t), _) ->
      let size, align = layout_item key in
      (size, align, alias_class key, Hashtbl.find spans key))

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
  let segments =
    maybe_elide_zeros ~backend_name
      (Schedule.maybe_default_schedules ~backend_name ~limits ~static_indices lowered)
  in
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

  (* gh-ocannl-550: [upload_slot] roots a pool in the backend table, and the transfer that follows
     adds its location to the context only on success — so a failing upload leaves a pool no context
     can ever reach, and therefore no [Context.release] can reclaim. [release] gives back what this
     operation took: the one pool it minted, or (gh-ocannl-1125) the bytes it bumped in an upload
     arena whose other tenants stay. Unlike [allocate_delta]'s unwind there is no constant-cache
     involvement here (a transfer destination is a working buffer by construction), so this needs
     nothing beyond that. *)
  let with_transfer_pool device ~release ~f =
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
        (try release () with _ -> ());
        Stdlib.Printexc.raise_with_backtrace exn backtrace

  let free_transfer_pool device pool_id () =
    Option.iter Backend.free_pool ~f:(fun free -> free device ~pool_id);
    Alloc_census.forget_pool ~device_id:device.Backend_intf.device_id ~pool_id

  (* gh-ocannl-1125: absent host-upload and device-copy nodes share bump-packed arenas. Upload
     siblings retain their common pools independently (gh-ocannl-1173). Only a value holding an
     arena's last tenant may extend it, so branches cannot overlap allocations -- holding it at its
     location ([Backend_intf.holds_tenant], gh-ocannl-1328): a sibling holding the same node
     elsewhere is no owner of this pool, and its upload would be freed under it.

     First fit, among arenas of the same residency hint that this value may extend. The first arena
     of a residency hint this value holds a tenant of is sized exactly to its node, so a lone upload
     costs what it always did; each later one targets the capacity that hint's arenas held by this
     value hold so far, rounded up to the node's alignment so that the padding between tenants
     cannot eat the room the doubling promised. Sibling-only arenas do not count (gh-ocannl-1204):
     [k] forks each uploading one node take [k] exact pools, not a doubling series. Capacities
     double: [n] consecutive uploads take O(log n) pools, and the unused room is bounded by the
     capacity already filled. A node over the per-pool cap gets an exact pool of its own, as before.

     Returns the location and the [release] that gives it back if the upload fails: a fresh arena is
     freed and forgotten, a bump into an existing one is rolled back (only if still on top, which it
     is: transfers into a lifecycle are not concurrent) and its pool-mates are left alone. A
     released pool's arena is removed by [finalize]. Released handles must not be reused. *)
  let upload_slot (ctx : Backend.context) (tn : Tn.t) : Backend_intf.buffer_loc * (unit -> unit) =
    let device = ctx.device in
    let arenas = ctx.upload_arenas in
    let size_in_bytes, align = layout_item tn in
    let mode = Option.map tn.Tn.memory_mode_intent ~f:fst in
    let align_up off = (off + align - 1) / align * align in
    let same_mode (a : Backend_intf.upload_arena) =
      [%equal: Tn.memory_mode option] a.arena_mode mode
    in
    let holds_tenant = Backend_intf.holds_tenant ctx.ctx_buffers in
    let fits (a : Backend_intf.upload_arena) =
      let last = List.hd_exn a.tenants in
      same_mode a && align_up last.bump + size_in_bytes <= a.capacity && holds_tenant a last
    in
    match List.find arenas.arenas ~f:fits with
    | Some a ->
        let before = a.tenants in
        let offset = align_up (List.hd_exn before).bump in
        let appended =
          { Backend_intf.tenant = tn; tenant_offset = offset; bump = offset + size_in_bytes }
          :: before
        in
        a.tenants <- appended;
        let release () = if phys_equal a.tenants appended then a.tenants <- before in
        ({ pool_id = a.arena_pool_id; offset }, release)
    | None ->
        let cap = pool_cap () in
        (* gh-ocannl-1204: only the arenas this value holds a tenant of. A sibling's arenas are
           shared lineage state but can never be extended from here, so counting them doubled each
           simultaneously held fork's arena over the last. *)
        let holds (a : Backend_intf.upload_arena) = List.exists a.tenants ~f:(holds_tenant a) in
        let minted =
          List.sum
            (module Int)
            arenas.arenas
            ~f:(fun a -> if same_mode a && holds a then a.capacity else 0)
        in
        let capacity =
          if size_in_bytes >= cap then size_in_bytes
          else min cap (max size_in_bytes (align_up minted))
        in
        let pool_id = device.next_pool_id in
        device.next_pool_id <- pool_id + 1;
        Backend.alloc_pool ?mode device ~pool_id ~size_in_bytes:capacity ~alignment:align;
        (* A transfer destination is a working buffer, just like [allocate_delta]'s working
           slabs. *)
        Alloc_census.record_pool ~device_id:device.device_id ~pool_id ~constant:false
          ~size_in_bytes:capacity;
        let a : Backend_intf.upload_arena =
          {
            arena_pool_id = pool_id;
            arena_mode = mode;
            capacity;
            tenants = [ { tenant = tn; tenant_offset = 0; bump = size_in_bytes } ];
          }
        in
        arenas.arenas <- a :: arenas.arenas;
        let release () =
          arenas.arenas <- List.filter arenas.arenas ~f:(fun b -> not (phys_equal a b));
          free_transfer_pool device pool_id ()
        in
        ({ pool_id; offset = 0 }, release)

  let%track3_sexp init_from_host (ctx : Backend.context) (tn : Tn.t) (hosted : Ndarray.t) =
    match Map.find ctx.ctx_buffers tn with
    | None ->
        (* No zero-init: we are immediately copying from host. *)
        let dst, release = upload_slot ctx tn in
        with_transfer_pool ctx.device ~release ~f:(fun () ->
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
            let d_loc, release = upload_slot dst tn in
            with_transfer_pool dst.device ~release ~f:(fun () ->
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
                (* Keep the destination guarded until asynchronous copy errors have surfaced. A
                   failure must restore an arena bump before any owner is published. *)
                Resource_fault_injection.hit Init_from_device_before_await;
                Backend.await dst.device;
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

let%track6_sexp lower_assignments ~limits optim_ctx ?name bindings asgns =
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
    Assignments.lower
      ~rewrite_target:{ online_softmax_auto_block = limits.online_softmax_auto_block }
      optim_ctx ~unoptim_ll_source ~ll_source ~cd_source ~name (Indexing.bound_symbols bindings)
      asgns )

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

(* Frees every pool of [pool_ids] for a cleanup with no retry owner -- nothing holds the ids
   afterwards, so a pool skipped here is leaked for good. Every pool is attempted, so one free the
   driver refuses strands no other, and the first refusal is re-raised once all were tried
   (gh-ocannl-1493, gh-ocannl-1555). [?before_free] names the fault-injection point hit before each
   free. A context [finalize] is not such a cleanup: it keeps a retry ledger and stops at the first
   refusal on purpose. *)
let free_pools_continuing ?before_free device ~free_pool pool_ids =
  let first_failure =
    List.fold pool_ids ~init:None ~f:(fun failure pool_id ->
        match
          Option.iter before_free ~f:Resource_fault_injection.hit;
          free_and_forget_pool device ~free_pool pool_id
        with
        | () -> failure
        | exception exn -> (
            match failure with
            | Some _ -> failure
            | None -> Some (exn, Stdlib.Printexc.get_raw_backtrace ())))
  in
  Option.iter first_failure ~f:(fun (exn, backtrace) ->
      Stdlib.Printexc.raise_with_backtrace exn backtrace)

(* Uncommitted compile deltas have no upload siblings. Use the same owned-pool classification as
   upload reference registration and context cleanup, freeing each pool once. *)
let free_owned_pools ~device ~free_pool ~owned_elsewhere (ctx_buffers : ctx_buffers) : unit =
  free_pools_continuing device ~free_pool
    (Set.to_list (owned_pool_ids ~device ~owned_elsewhere ctx_buffers))

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
      List.map (Indexing.bound_symbols code_batch.bindings) ~f:(fun s ->
          (s, Indexing.fresh_launch_param ()))
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
      | None ->
          Compile_cost.time_lower (fun () ->
              lower_assignments ~limits:(Device.hardware_limits ()) optim_ctx ?name bindings
                comp.asgns)
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
    let capabilities = Device.codegen_capabilities () in
    let lowereds =
      Compile_cost.time_transform @@ fun () ->
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
    (* A [?prelowered] routine is hand-built IR compiled as written (the gh-ocannl-562 test seam):
       its zeros are what the test is about, so they are not elided. *)
    let lowereds =
      if Option.is_some prelowered then lowereds
      else maybe_elide_zeros ~backend_name:Device.name lowereds
    in
    record_segments lowereds;
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
          Schedule.check_hardware_limits_classified ~name ~limits ~capabilities single;
          let compiled =
            Schedule_outcome.tag Schedule_outcome.Backend_compile (fun () ->
                Compile_cost.time_backend_compile (fun () -> compile ~name bindings single))
          in
          (Either.First compiled, single)
      | segments ->
          let seg_names = List.mapi segments ~f:(fun i _ -> name ^ "__seg" ^ Int.to_string i) in
          List.iter2_exn seg_names segments ~f:(fun seg_name seg ->
              Schedule.check_hardware_limits_classified ~name:seg_name ~limits ~capabilities seg);
          let batch =
            Schedule_outcome.tag Schedule_outcome.Backend_compile (fun () ->
                Compile_cost.time_backend_compile (fun () ->
                    compile_batch ~names:(Array.of_list seg_names) bindings (Array.of_list segments)))
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

  (* gh-ocannl-1493: one layout group of a context's delta, laid out as the allocator lays it out --
     the pools it asks the device for, in allocation order, and where each node lands. The plan is
     pure (no pool id minted, no device touched), so the link ([allocate_delta]) and the link-free
     probe ([link_delta]) run the SAME plan: the probe asks the device for exactly the pools the
     link would, in the same order, sized and aligned alike. *)
  type group_layout = {
    gl_group : (Tn.t * Low_level.traced_array) list;  (** Canonical (uid) order. *)
    gl_constant : bool;
    gl_pools : (int * int) list;  (** [(size_in_bytes, alignment)] per pool, in allocation order. *)
    gl_places : (int * int) list;
        (** Per node of [gl_group]: the index of its pool in [gl_pools], and its byte offset. *)
    gl_arena : bool;
        (** Laid out by the liveness planner (gh-ocannl-489) rather than bump-packed. *)
  }

  type delta_plan = {
    dp_working : group_layout option;
    dp_cached_constants : (Tn.t * buffer_loc) list;
        (** Constants a [constant_buffer_cache] hit already resolves on this device. *)
    dp_new_constants : group_layout option;
  }

  (* Lay a group out into pools. With a liveness plan (the working group under [buffer_aliasing]),
     the group is one arena where liveness-disjoint same-precision nodes overlap; it falls back to
     bump packing when the arena would exceed the per-pool cap. Bump packing segments at the cap,
     each segment's slab aligned to the largest element precision it holds. *)
  let plan_group ?arena ~cap ~constant (group : (Tn.t * Low_level.traced_array) list) :
      group_layout option =
    if List.is_empty group then None
    else
      let arena_layout =
        Option.bind arena ~f:(fun spans -> plan_arena_offsets ~cap (arena_items ~spans group))
      in
      match arena_layout with
      | Some (offsets, total) ->
          let alignment =
            List.fold group ~init:1 ~f:(fun a (key, _) ->
                max a (Ops.prec_in_bytes (Lazy.force key.Tn.storage_prec)))
          in
          Some
            {
              gl_group = group;
              gl_constant = constant;
              gl_pools = [ (total, alignment) ];
              gl_places = List.map offsets ~f:(fun offset -> (0, offset));
              gl_arena = true;
            }
      | None ->
          let assignments, segment_sizes =
            plan_pool_segments ~cap ~what:"Backends.allocate_delta"
              ~debug_name:(fun i -> Tn.debug_name (fst (List.nth_exn group i)))
              (layout_items group)
          in
          let seg_align = Array.of_list_map segment_sizes ~f:(fun _ -> 1) in
          List.iter2_exn group assignments ~f:(fun (key, _) (seg, _) ->
              seg_align.(seg) <-
                max seg_align.(seg) (Ops.prec_in_bytes (Lazy.force key.Tn.storage_prec)));
          Some
            {
              gl_group = group;
              gl_constant = constant;
              gl_pools = List.mapi segment_sizes ~f:(fun i size -> (size, seg_align.(i)));
              gl_places = assignments;
              gl_arena = false;
            }

  (* gh-ocannl-344 Phase B/C: plan a context's delta -- the in-context tnodes not already present in
     [context.ctx_buffers]. Working (non-constant) and constant/read-only nodes are EACH packed into
     pools sized to their group, replacing the one-pool-per-tnode policy. Only the working group is
     liveness-planned (gh-ocannl-489): constants are deduped per-device and outlive the context, so
     their lifetimes are not routine intervals. Constants already allocated on this device (a hit in
     [constant_buffer_cache], possibly from another context tree) resolve directly and are excluded
     from the new slab, so a fresh constant pool holds exactly this device's genuinely-new constants
     -- no wasted holes. Groups arrive in canonical (uid) order from [partition_layout_groups] --
     the order [score_footprint] scores, which the order-sensitive planners must see identically
     (gh-ocannl-498). *)
  let plan_delta (context : context) ~alias_spans (lowered : Low_level.optimized) : delta_plan =
    let cap = pool_cap () in
    let working, constants =
      partition_layout_groups ~plc:lowered.Low_level.optimize_ctx.placements
        ~skip:(Map.mem context.ctx_buffers) lowered.Low_level.traced_store
    in
    let cached, fresh =
      List.partition_map constants ~f:(fun (key, node) ->
          match Hashtbl.find context.device.constant_buffer_cache key with
          | Some data -> First (key, data)
          | None -> Second (key, node))
    in
    {
      dp_working = plan_group ?arena:alias_spans ~cap ~constant:false working;
      dp_cached_constants = cached;
      dp_new_constants = plan_group ~cap ~constant:true fresh;
    }

  (* The pool allocations of one delta, counted at the shared seam: what [allocate_delta] and the
     link-free probe both run, so the device sees the same requests from either. [minted] is what
     this transaction allocated, so a failure part way through can give it back; [held] the bytes
     those pools hold.

     gh-ocannl-1424: a group past the per-pool cap is several pools allocated one after another, so
     a refusal can come part way through, and the driver's free bytes in the refusal are then read
     with this same transaction's earlier pools still allocated. Read bare, that reading looks like
     memory something else left resident -- which is how a 28 GB materialize-all delta on a 12 GiB
     card, refused at its third 4 GiB pool, was reported as 8.6 GB "still resident after arm A". So
     the refusal says so. *)
  type pool_tx = { tx_name : string; minted : int list ref; held : int ref }

  let new_pool_tx ~name = { tx_name = name; minted = ref []; held = ref 0 }

  (* Mints a pool id per planned pool up front, then allocates each; returns the ids in [pools]
     order. [~constant] is the group's class, for the census and the refusal's detail. *)
  let allocate_pools tx device ~constant (pools : (int * int) list) : int array =
    let pool_ids =
      Array.of_list_map pools ~f:(fun _ ->
          let pool_id = device.next_pool_id in
          device.next_pool_id <- pool_id + 1;
          pool_id)
    in
    let count = Array.length pool_ids in
    let group_bytes = List.sum (module Int) pools ~f:fst in
    List.iteri pools ~f:(fun index (size_in_bytes, alignment) ->
        let pool_id = pool_ids.(index) in
        (try alloc_pool device ~pool_id ~size_in_bytes ~alignment
         with Schedule_outcome.Device_memory_exhausted r when count > 1 || !(tx.held) > 0 ->
           let backtrace = Stdlib.Printexc.get_raw_backtrace () in
           Stdlib.Printexc.raise_with_backtrace
             (Schedule_outcome.Device_memory_exhausted
                {
                  r with
                  detail =
                    Printf.sprintf
                      "%s; it was pool %d of %d of %s's %s delta of %d bytes, and this compile's \
                       %d earlier pool(s), %d bytes, were still allocated when the free bytes were \
                       read"
                      r.detail (index + 1) count tx.tx_name
                      (if constant then "constant" else "working")
                      group_bytes (List.length !(tx.minted)) !(tx.held);
                })
             backtrace);
        tx.held := !(tx.held) + size_in_bytes;
        tx.minted := pool_id :: !(tx.minted);
        (* gh-ocannl-550: the shared seam is where a pool's byte size is known, so it is where the
           per-class census records it. [~constant] separates the two groups, whose lifetimes
           differ: a working pool belongs to the context and dies with it, a constant pool is
           deduped per device and outlives it. *)
        Alloc_census.record_pool ~device_id:device.device_id ~pool_id ~constant ~size_in_bytes;
        Resource_fault_injection.hit Delta_pool_allocated);
    pool_ids

  let allocate_group_pools tx device (gl : group_layout) : int array =
    allocate_pools tx device ~constant:gl.gl_constant gl.gl_pools

  (* gh-ocannl-344 Phase B/C: allocate a context's delta ([plan_delta]). Working pools belong to the
     context (freed at its [finalize]); constant pools are deduped per-device via
     [constant_buffer_cache] and outlive the context (freed at device teardown). Pool ids and
     offsets are deterministic across runs: the plan is in canonical order and ids are minted per
     pool before any placement. The per-pool 4 GB cap (uint32 offsets unless large_models) is
     enforced by {!Backend_utils.plan_pool_segments}. *)
  let%track3_sexp allocate_delta (context : context) ~name
      ~(alias_spans : (Tn.t, int * int) Base.Hashtbl.t option) (lowered : Low_level.optimized) :
      ctx_buffers =
    let device = context.device in
    let plan = plan_delta context ~alias_spans lowered in
    let ctx_buffers = ref context.ctx_buffers in
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
    (* What THIS call minted, so a failure part way through can give it back (gh-ocannl-550,
       round-four review). [with_delta] in [link] covers failures after the delta is returned; it
       cannot cover a failure inside this function, where the caller never receives [ctx_buffers]
       and the slabs already allocated are rooted with nothing to reach them. Pool ids are fresh
       from [device.next_pool_id], so every id here is unambiguously this call's. *)
    let tx = new_pool_tx ~name in
    let cache_inserts = ref [] in
    let unwind_partial_delta () =
      (* The uploads scheduled above are asynchronous, so the pools must not be freed under them. *)
      (try Device.await device with _ -> ());
      (* Constant-cache entries this call inserted point INTO the pools about to go, so they have to
         come out first or a later context would resolve a freed slab. Entries that were already
         there belong to earlier compiles and are untouched. *)
      List.iter !cache_inserts ~f:(Hashtbl.remove device.constant_buffer_cache);
      (* gh-ocannl-1555: past a refused free too -- the caller never receives these ids, so a pool
         this skipped would be leaked for good. *)
      Option.iter free_pool ~f:(fun free_pool ->
          free_pools_continuing ~before_free:Delta_unwind_before_free device ~free_pool
            (List.dedup_and_sort !(tx.minted) ~compare:Int.compare))
    in
    let pack (gl : group_layout) ~(register : Tn.t -> alloc:(unit -> buffer_loc) -> unit) : unit =
      let pool_ids = allocate_group_pools tx device gl in
      List.iter2_exn gl.gl_group gl.gl_places ~f:(fun entry (index, offset) ->
          place entry ~pool_id:pool_ids.(index) ~offset ~register);
      if gl.gl_arena && log_buffer_aliasing () then
        let _, bump_sizes =
          plan_pool_segments ~cap:(pool_cap ()) ~what:"Backends.allocate_delta"
            ~debug_name:(fun i -> Tn.debug_name (fst (List.nth_exn gl.gl_group i)))
            (layout_items gl.gl_group)
        in
        let dedicated = List.fold bump_sizes ~init:0 ~f:( + ) in
        let total = List.sum (module Int) gl.gl_pools ~f:fst in
        let planned =
          Option.value_map alias_spans ~default:0 ~f:(fun spans ->
              List.count gl.gl_group ~f:(fun (key, _) -> Hashtbl.mem spans key))
        in
        Stdlib.Printf.eprintf
          "buffer aliasing: %s: working pool %d bytes, dedicated packing %d bytes (%.1f%% saved; \
           %d/%d nodes liveness-planned)\n\
           %!"
          name total dedicated
          (100. *. Float.of_int (dedicated - total) /. Float.of_int (max 1 dedicated))
          planned (List.length gl.gl_group)
    in
    let passes () =
      (* Pass 2a: working delta -> context-owned pool(s), recorded directly. *)
      Option.iter plan.dp_working ~f:(fun gl ->
          pack gl ~register:(fun key ~alloc ->
              ctx_buffers := Map.add_exn !ctx_buffers ~key ~data:(alloc ())));
      (* Pass 2b: constants / read-only -> per-device constant pool(s): the cache hits resolve
         directly, the remaining new constants pack into one constant pool (or more, past the cap),
         deduped into the cache. Constant pools outlive the context and are skipped by context
         [finalize] (freed at device teardown). *)
      List.iter plan.dp_cached_constants ~f:(fun (key, data) ->
          ctx_buffers := Map.add_exn !ctx_buffers ~key ~data);
      Option.iter plan.dp_new_constants ~f:(fun gl ->
          pack gl ~register:(fun key ~alloc ->
              let data =
                Hashtbl.find_or_add device.constant_buffer_cache key ~default:(fun () ->
                    cache_inserts := key :: !cache_inserts;
                    alloc ())
              in
              ctx_buffers := Map.add_exn !ctx_buffers ~key ~data))
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

  (* gh-ocannl-1493: a trial allocation of [groups] -- [(constant, pools)] in allocation order --
     through the seam the link allocates through, freed again before returning, on every path: the
     device answers the question a link would put to it without memset, uploads, constant-cache
     inserts, codegen or link. The pools are given back by [free_pools_continuing]: every one is
     freed even when one free fails, and the first failure is re-raised once all were tried. *)
  let trial_allocate (context : context) ~fn ~name (groups : (bool * (int * int) list) list) =
    match free_pool with
    | None -> invalid_arg (fn ^ ": " ^ Device.name ^ " cannot free a probe's pools")
    | Some free_pool ->
        let device = context.device in
        let tx = new_pool_tx ~name in
        let give_back () = free_pools_continuing device ~free_pool !(tx.minted) in
        (match
           List.iter groups ~f:(fun (constant, pools) ->
               ignore (allocate_pools tx device ~constant pools : int array))
         with
        | () -> ()
        | exception exn ->
            let backtrace = Stdlib.Printexc.get_raw_backtrace () in
            (try give_back () with _ -> ());
            Stdlib.Printexc.raise_with_backtrace exn backtrace);
        give_back ()

  (* gh-ocannl-1493: what linking, on [context], a compile whose [lowered_transform] received
     [lowered] and returned [segments] would allocate -- planned without codegen or link, by the
     post-transform layout steps [compile] runs (the GPU covering-zero elision, the liveness spans)
     and the plan [allocate_delta] runs. Hermetic: it works on copies, since forcing the open
     placements and folding the segment stores back both mutate.

     With [~probe], the planned pools are also allocated, through the seam the link allocates
     through, and freed again before returning, on every path: the device answers the question the
     link would put to it -- [Schedule_outcome.Device_memory_exhausted] where the link would be
     refused -- without memset, uploads, constant-cache inserts, codegen or link. *)
  let link_delta (context : context) ?(name = "link_delta") ?(prelowered = false) ~probe
      (lowered : Low_level.optimized) (segments : Low_level.optimized list) : link_delta =
    let ctx' = Low_level.copy_optimize_ctx lowered.Low_level.optimize_ctx in
    let store' = Hashtbl.copy lowered.Low_level.traced_store in
    let copy (o : Low_level.optimized) =
      {
        o with
        Low_level.traced_store =
          (if phys_equal o.Low_level.traced_store lowered.Low_level.traced_store then store'
           else Hashtbl.copy o.Low_level.traced_store);
        optimize_ctx =
          (if phys_equal o.Low_level.optimize_ctx lowered.Low_level.optimize_ctx then ctx'
           else Low_level.copy_optimize_ctx o.Low_level.optimize_ctx);
      }
    in
    let lowered = copy lowered in
    let segments = List.map segments ~f:copy in
    (* As in [compile]: a [?prelowered] routine's zeros are compiled as written. *)
    let segments =
      if prelowered then segments else maybe_elide_zeros ~backend_name:Device.name segments
    in
    if List.is_empty segments then invalid_arg "Backends.link_delta: no segments";
    let alias_spans =
      plan_alias_spans ~name ~limits:(Device.hardware_limits ()) ~lowered ~segments
    in
    (* As in [compile]: a single kernel allocates from its own segment, a fissioned routine from the
       whole-routine code with the segment-added store entries folded back. *)
    let lowered =
      match segments with
      | [ single ] -> single
      | segments ->
          fold_segment_stores ~into:lowered.Low_level.traced_store segments;
          lowered
    in
    let plan = plan_delta context ~alias_spans lowered in
    let groups = List.filter_opt [ plan.dp_working; plan.dp_new_constants ] in
    let bytes gl = List.sum (module Int) gl.gl_pools ~f:fst in
    let summary =
      {
        ld_working_bytes = Option.value_map plan.dp_working ~default:0 ~f:bytes;
        ld_constant_bytes = Option.value_map plan.dp_new_constants ~default:0 ~f:bytes;
        ld_pools = List.sum (module Int) groups ~f:(fun gl -> List.length gl.gl_pools);
        ld_liveness_planned = Option.is_some alias_spans;
      }
    in
    if probe then
      trial_allocate context ~fn:"Backends.link_delta" ~name
        (List.map groups ~f:(fun gl -> (gl.gl_constant, gl.gl_pools)));
    summary

  (* gh-ocannl-1554: the floor under every schedule of [lowered] -- see [link_floor] in
     [Backend_intf]. Settledness is read before the partition forces the open placements (on the
     copies, as in [link_delta]); a node the base code does not touch is not counted, since a
     single-kernel candidate's store is filtered to its code footprint. *)
  let link_floor (context : context) ?(name = "link_floor") ?(prelowered = false) ~probe
      (lowered : Low_level.optimized) : link_floor =
    let lowered =
      {
        lowered with
        Low_level.traced_store = Hashtbl.copy lowered.Low_level.traced_store;
        optimize_ctx = Low_level.copy_optimize_ctx lowered.Low_level.optimize_ctx;
      }
    in
    (* As in [compile] and [link_delta]: a [?prelowered] routine's zeros are compiled as written. *)
    let lowered =
      if prelowered then lowered
      else
        match maybe_elide_zeros ~backend_name:Device.name [ lowered ] with
        | [ single ] -> single
        | _ -> lowered
    in
    let plc = lowered.Low_level.optimize_ctx.placements in
    let limits = Device.hardware_limits () in
    let off_device =
      Hashtbl.fold lowered.Low_level.traced_store
        ~init:(Set.empty (module Tn))
        ~f:(fun ~key ~data acc ->
          if Schedule.may_keep_off_device ~backend_name:Device.name ~limits plc key data then
            Set.add acc key
          else acc)
    in
    let touched, _ = Schedule.code_footprint lowered.Low_level.llc in
    let working, constants =
      partition_layout_groups ~plc ~skip:(Map.mem context.ctx_buffers)
        lowered.Low_level.traced_store
    in
    let size (key, _) = size_in_bytes_of key in
    let sum = List.sum (module Int) ~f:size in
    let working = List.filter working ~f:(fun (key, _) -> Set.mem touched key) in
    let off, pinned = List.partition_tf working ~f:(fun (key, _) -> Set.mem off_device key) in
    let working_bytes, overlap_bytes =
      if not (buffer_aliasing ()) then (sum pinned, 0)
      else
        let eligible, dedicated =
          List.partition_tf pinned ~f:(fun (key, _) -> alias_eligible ~lowered key)
        in
        let largest =
          Map.of_alist_fold
            (module String)
            (List.map eligible ~f:(fun ((key, _) as item) -> (alias_class key, size item)))
            ~init:0 ~f:Int.max
          |> Map.data
          |> List.sum (module Int) ~f:Fn.id
        in
        (sum dedicated + largest, sum eligible - largest)
    in
    let fresh =
      List.filter constants ~f:(fun (key, _) ->
          Set.mem touched key && not (Hashtbl.mem context.device.constant_buffer_cache key))
    in
    let replaceable, kept =
      List.partition_tf fresh ~f:(fun (key, _) -> Schedule.may_replace_by_packed_tile plc key)
    in
    let constant_bytes = sum kept in
    (* One request per group, cut only at the per-pool cap: with the cap a page multiple, the
       requests' page-rounded total is the floor's rounded up, never above the rounded total of a
       candidate's pools, which hold at least the floor's bytes. *)
    let cap = pool_cap () in
    let alignment items =
      List.fold items ~init:1 ~f:(fun a (key, _) ->
          max a (Ops.prec_in_bytes (Lazy.force key.Tn.storage_prec)))
    in
    let pools ~items bytes =
      let rec cut rest acc =
        if rest <= 0 then List.rev acc
        else
          let piece = min rest cap in
          cut (rest - piece) ((piece, alignment items) :: acc)
      in
      cut bytes []
    in
    let groups =
      [ (false, pools ~items:pinned working_bytes); (true, pools ~items:kept constant_bytes) ]
    in
    if probe then trial_allocate context ~fn:"Backends.link_floor" ~name groups;
    {
      lf_working_bytes = working_bytes;
      lf_constant_bytes = constant_bytes;
      lf_pools = List.sum (module Int) groups ~f:(fun (_, p) -> List.length p);
      lf_off_device_bytes = sum off;
      lf_replaceable_bytes = sum replaceable;
      lf_overlap_bytes = overlap_bytes;
    }

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
        let bindings, schedule, release_hook =
          match code.proc with
          | Either.First single ->
              let bindings, schedule = link context single ctx_buffers in
              (bindings, schedule, None)
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
              let schedule, release_hook =
                match
                  sequence_segments context ~name:code.name ~bindings
                    ~uses_merge_buffer:(Option.is_some code.expected_merge_node)
                    tasks
                with
                | Some (fused, release) -> (fused, Some release)
                | None ->
                    ( Task.Task
                        {
                          context_lifetime = tasks;
                          description = "fissioned segments of " ^ code.name;
                          work =
                            (fun () ->
                              List.iteri tasks ~f:(fun i t ->
                                  if i > 0 then will_wait_for context (all_work context.device);
                                  Task.run t));
                        },
                      None )
              in
              (bindings, schedule, release_hook)
        in
        let context = make_child ~ctx_buffers ~optimize_ctx context in
        (* gh-ocannl-1445: released with the context the routine's buffers live in. *)
        Option.iter release_hook ~f:(fun hook ->
            context.release_hooks := hook :: !(context.release_hooks));
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

let debug_check_upload_arenas =
  lazy (Utils.get_global_flag ~default:false ~arg_name:"debug_check_upload_arenas")

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
        let owners = ctx.upload_arenas.pool_owners in
        let pools =
          owned_pool_ids ~device:ctx.device
            ~owned_elsewhere:(fun key ->
              Option.exists ctx.parent ~f:(fun pc -> Map.mem pc.ctx_buffers key))
            ctx.ctx_buffers
        in
        Stdlib.Mutex.protect ctx.upload_arenas.owners_mutex (fun () ->
            Set.iter pools ~f:(fun pool_id ->
                if not (Set.mem ctx.released_pool_ids pool_id) then (
                  let tips =
                    Option.value (Hashtbl.find owners pool_id) ~default:[ ctx.upload_tip ]
                  in
                  let remaining =
                    List.filter tips ~f:(fun tip -> not (phys_equal tip ctx.upload_tip))
                  in
                  if not (List.is_empty remaining) then (
                    Hashtbl.set owners ~key:pool_id ~data:remaining;
                    List.iter ctx.upload_arenas.arenas ~f:(fun a ->
                        if a.arena_pool_id = pool_id then
                          a.tenants <-
                            List.drop_while a.tenants ~f:(fun t ->
                                not (List.exists remaining ~f:(fun tip -> holds_tenant !tip a t)))))
                  else (
                    Resource_fault_injection.hit Finalize_before_free;
                    free_and_forget_pool ctx.device ~free_pool pool_id;
                    Hashtbl.remove owners pool_id;
                    ctx.upload_arenas.arenas <-
                      List.filter ctx.upload_arenas.arenas ~f:(fun a -> a.arena_pool_id <> pool_id));
                  ctx.released_pool_ids <- Set.add ctx.released_pool_ids pool_id))))
  in
  (* gh-ocannl-1445: the routines' release hooks (cached graph execs), once the pools are freed. Not
     part of [cleanup]: the flag stays committed when a hook fails, since the pools ARE freed and a
     context that reads live again would run kernels over them. What a failed hook leaves is
     destroyed with its routine, by the bindings' own finalizers, or at exit. Every hook is
     attempted, and the first failure is re-raised. *)
  let run_release_hooks () = Backend_impl.attempt_each !(ctx.release_hooks) in
  (* gh-ocannl-1462: the debug self-check of the lineage's arena ownership, after a successful
     cleanup -- a point of committed ownership -- and outside it: the pools ARE freed, so a failed
     check must not reset the flag, and the release hooks still run before it raises. *)
  let arena_violations () =
    if Lazy.force debug_check_upload_arenas then
      Stdlib.Mutex.protect ctx.upload_arenas.owners_mutex (fun () ->
          upload_arena_violations ctx.upload_arenas)
    else []
  in
  if Atomic.compare_and_set ctx.finalized false true then
    match cleanup () with
    | () ->
        Alloc_census.count_context_released ();
        let violations = arena_violations () in
        run_release_hooks ();
        if not (List.is_empty violations) then
          failwith
            ("Backends.finalize (debug_check_upload_arenas): upload-arena ownership invariant \
              violated after a release: "
            ^ String.concat ~sep:"; " violations)
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

(* gh-ocannl-1281: a faulted GPU leaves its driver wedged, and every later process that touches the
   runtime then spins in device initialization forever (gh-ocannl-1211). The guarded class is every
   device backend entry point that ASKS the driver about devices or opens one: discovery
   ([num_devices]), the per-backend properties and limits ([static_properties], [hardware_limits],
   each of which enumerates the devices), a device's identity ([timing_identity]: attributes, and on
   HIP the runtime and compiler versions) and opening a device ([get_device]). Each runs under
   {!Utils.with_device_init_deadline} on every call: [num_devices], [static_properties] and
   [timing_identity] query the driver again each time, and a GPU that faults after its first
   discovery is exactly the wedge this bounds; where a call is a cache read the guard costs a few
   system calls. Outside the class, deliberately: everything that WAITS on device work ([await],
   [sync], the copies and launches), where a long wait is a legitimate long computation and the
   harnesses' per-test caps are the bound. Applied to the three device backends only; the cc
   backends, which [Raise_backend] also builds, touch no driver. A missing backend's stub is guarded
   too, harmlessly: it answers at once. The implementation modules themselves ([Cuda_backend.Impl],
   ...) are below this boundary: code that drives one directly, as the arrayjit backend tests do,
   wraps its own device opening in {!Utils.with_device_init_deadline}; and so does a backend's own
   device discovery reached from outside these entry points, while compiling or linking
   ([min_compute_capability] on CUDA; [all_rdna_wave32], [hiprtc_target_archs] and the scratch
   budget on HIP). A region entered from inside another on the same thread runs under the outer
   one's deadline. *)
module With_device_init_deadline (Device : Lowered_backend) = struct
  include Device

  let guarded f = Utils.with_device_init_deadline ~backend_name:Device.name f
  let num_devices () = guarded Device.num_devices
  let static_properties () = guarded Device.static_properties
  let hardware_limits () = guarded Device.hardware_limits
  let timing_identity device = guarded (fun () -> Device.timing_identity device)
  let get_device ~ordinal = guarded (fun () -> Device.get_device ~ordinal)
end

module Cc_b : Backend = Make_device_backend_from_lowered (Schedulers.Sync) (Cc_backend)
module Multidev_cc_b : Backend = Make_device_backend_from_lowered (Schedulers.Multidev) (Cc_backend)

module Cuda_b : Backend = Raise_backend (With_device_init_deadline (
  Cuda_backend_impl.Impl : Lowered_backend))

module Hip_b : Backend = Raise_backend (With_device_init_deadline (
  Hip_backend_impl.Impl : Lowered_backend))

module Metal_b : Backend = Raise_backend (With_device_init_deadline (
  Metal_backend_impl.Impl : Lowered_backend))

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
