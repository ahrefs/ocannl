open Base
module Tn = Tnode
module Idx = Indexing
module LL = Low_level

type mint_role =
  | Split_outer
  | Split_inner
  | Expand_axis of int
  | Tensorize_lane
  | Partition_seg of int
  | Split_reduce_block
  | Split_reduce_inner
  | Split_reduce_combine of int
  | Fold_mma_lane
  | Fold_mma_block
  | Coalesce_merged
[@@deriving sexp, compare, equal, hash]

type sym_ref = Base of int | Static of int | Minted of int * mint_role
[@@deriving sexp, compare, equal]

module Sym_ref = struct
  module T = struct
    type t = sym_ref [@@deriving sexp, compare]
  end

  include T
  include Comparator.Make (T)
end

type saved_optop =
  | Split of { axis : sym_ref; factor : int; outer : LL.axis_type; inner : LL.axis_type }
  | Swap of { outer : sym_ref; inner : sym_ref }
  | Retype of { axis : sym_ref; ty : LL.axis_type }
  | Unroll of { axis : sym_ref; materialize : bool }
  | Partition of { axis : sym_ref; breakpoints : int list }
  | Pad of { axis : sym_ref; to_multiple_of : int }
  | Coalesce of { outer : sym_ref; inner : sym_ref }
  | Stage of {
      source : int;
      tile_loops : sym_ref list;
      shared : bool;
      cooperative : int option;
      hoisted : bool;
      swizzle : LL.swizzle_kind option; [@sexp.option]
      pad_stride : int option; [@sexp.option]
      pipeline_depth : int option; [@sexp.option]
          (** [None] encodes depth 1, so pre-pipelining cache entries stay readable (the
              [swizzle]/[pad_stride] precedent). *)
      tile_prec : Ops.prec option; [@sexp.option]
    }
  | Privatize of { target : int; over : sym_ref; acc_prec : Ops.prec }
  | Expand_zero of { tn : int }
  | Tensorize of {
      i : sym_ref;
      j : sym_ref;
      k : sym_ref;
      simd_width : int;
      tile : Register_tile.t option; [@sexp.option]
          (** The requested C-tile geometry (gh-ocannl-619); omitted when the renderer chooses, so
              pre-geometry entries stay readable without an [entry_version] bump. *)
    }
  | Fuse_epilogue of { target : int; shared : bool }
  | Split_reduce of { axis : sym_ref; target : int; num_blocks : int }
  | Fold_mma of { query : sym_ref; width : int }
[@@deriving sexp, compare, equal]

type saved_schedule = saved_optop list [@@deriving sexp, compare, equal]

type canonical = {
  digest : string;
  complete : bool;
  base_syms : sym_ref Map.M(Idx.Symbol).t;
      (** Base and Static entries; binder symbols bound by more than one loop are excluded (their
          references would be ambiguous). *)
  tn_refs : int Map.M(Tn).t;
  ref_tns : Tn.t array;
}

let digest c = c.digest
let complete c = c.complete

let tn_of_ref c i =
  if i < 0 || i >= Array.length c.ref_tns then
    invalid_arg
      (Printf.sprintf "Schedule_cache.tn_of_ref: index %d out of range (%d tensor nodes)" i
         (Array.length c.ref_tns))
  else c.ref_tns.(i)

(* The canonical identity of an optimized routine, for schedule replay: the code walk is
   [LL.Canonical_render.emit], with the identity policy below (everything alpha-renamed, so a
   schedule saved in one session replays onto an isomorphic lowering in another) plus this
   consumer's extras — the [base_syms]/[tn_refs] resolution maps and the codegen-companion sections.
   Opposite choices to [Low_level.analysis_digest]'s, which keys by identity; see
   [LL.Canonical_render]. *)
(* The placement class of a node as the digest renders it. *)
let placement_class plc tn =
  match Tn.Placements.get plc tn with
  | None -> ";u"
  | Some (m, _) -> ";" ^ Sexp.to_string (Tn.sexp_of_memory_mode m)

(* The one canonical walk behind both identities this module exports — {!canonicalize} for the
   decided program a schedule applies to, {!canonicalize_source} for the decision problem a
   placement decision answers (gh-ocannl-786). [node_tag] renders what a first-occurrence tensor
   node carries beside its dims and precision; [companions] renders the consumer's sections after
   the code, given the node emitter so they share the numbering. *)
let canonical_of ~static_indices ~node_tag ~companions (llc : LL.t) : canonical =
  let buf = Buffer.create 4096 in
  let add = Buffer.add_string buf in
  let complete = ref true in
  (* The exported resolution: the first binding of each symbol; duplicated binders are dropped. *)
  let first_bind = Hashtbl.create (module Idx.Symbol) in
  let dup_binders = Hash_set.create (module Idx.Symbol) in
  let initial_tokens =
    List.mapi static_indices ~f:(fun k ss ->
        let s = ss.Idx.static_symbol in
        Hashtbl.set first_bind ~key:s ~data:(Static k);
        (s, Printf.sprintf "s%d" k))
  in
  let tn_refs = Hashtbl.create (module Tn) in
  let rev_tns = ref [] in
  let emit_tn tn =
    match Hashtbl.find tn_refs tn with
    | Some i -> add ("t" ^ Int.to_string i)
    | None ->
        let i = Hashtbl.length tn_refs in
        Hashtbl.set tn_refs ~key:tn ~data:i;
        rev_tns := tn :: !rev_tns;
        let dims = Lazy.force tn.Tn.dims in
        (* Hoistability enters the digest alongside dims and precision (gh-ocannl-470, Codex P2 on
           PR #123): a hoisted-[Stage] winner is only valid against constant operands, and a
           non-hoisted winner cached for a same-shape non-constant program must not mask a constant
           program's hoisted candidates — so such programs must not share cache keys. *)
        let hc = if Schedule.hoistable_constant tn then ";const" else "" in
        add
          (Printf.sprintf "t%d=[%s;%s%s%s]" i
             (String.concat_array ~sep:"," (Array.map dims ~f:Int.to_string))
             (Sexp.to_string (Ops.sexp_of_prec (Lazy.force tn.Tn.storage_prec)))
             hc (node_tag tn))
  in
  LL.Canonical_render.emit ~buf
    {
      emit_tn;
      (* A symbol free in the code cannot be resolved to a saved reference. *)
      emit_free_sym =
        (fun _ ->
          complete := false;
          add "?");
      on_bind_loop =
        (fun s ~id ~shadowed ->
          if shadowed then (
            Hash_set.add dup_binders s;
            complete := false)
          else Hashtbl.set first_bind ~key:s ~data:(Base id));
      mark_incomplete = (fun () -> complete := false);
      mma = LL.Canonical_render.Structural_mma;
      initial_tokens;
    }
    llc;
  companions ~add ~emit_tn;
  let base_syms =
    Hashtbl.fold first_bind
      ~init:(Map.empty (module Idx.Symbol))
      ~f:(fun ~key ~data acc ->
        if Hash_set.mem dup_binders key then acc else Map.set acc ~key ~data)
  in
  {
    digest = Stdlib.Digest.to_hex (Stdlib.Digest.string (Buffer.contents buf));
    complete = !complete;
    base_syms;
    tn_refs =
      Hashtbl.fold tn_refs
        ~init:(Map.empty (module Tn))
        ~f:(fun ~key ~data acc -> Map.set acc ~key ~data);
    ref_tns = Array.of_list_rev !rev_tns;
  }

(* Codegen-relevant companions of the decided code, sharing the code walk's node numbering. *)
let companions (opt : LL.optimized) ~add ~emit_tn =
  add "shared:[";
  Set.iter opt.LL.workgroup_shared ~f:(fun tn ->
      emit_tn tn;
      add ",");
  add "];fragments:[";
  Set.iter opt.LL.simdgroup_fragments ~f:(fun tn ->
      emit_tn tn;
      add ",");
  (* Emitted only when non-empty so pre-swizzle canonical digests (and their caches) stay valid. The
     layout kind is part of the entry (gh-ocannl-481 item 3: the two flavors are different physical
     layouts consumed by different renderings, so a cached winner must never alias across them),
     with the original element flavor keeping its bare rendering for the same reason the whole
     section is gated on non-emptiness. *)
  if not (Map.is_empty opt.LL.swizzled) then begin
    add "];swizzled:[";
    Map.iteri opt.LL.swizzled ~f:(fun ~key:tn ~data:kind ->
        emit_tn tn;
        (match kind with LL.Swizzle_elem -> () | LL.Swizzle_b128 -> add ":b128");
        add ",")
  end;
  (* Gated on non-emptiness like [swizzled], and for the same reason. The depth is digest-relevant
     on its own: pipeline depths >= 2 produce identical IR (the prologue/prefetch restructuring does
     not mention the depth) and differ only in the renderer's rotation modulus, so without this
     section a depth-2 winner would alias a depth-3 program (gh-ocannl-487). The rotor needs no
     entry — it is the staging anchor loop, determined by the code itself. *)
  if not (Map.is_empty opt.LL.pipelined) then begin
    add "];pipelined:[";
    Map.iteri opt.LL.pipelined ~f:(fun ~key:tn ~data:{ LL.pt_depth; pt_rotor = _ } ->
        emit_tn tn;
        add (":" ^ Int.to_string pt_depth);
        add ",")
  end;
  add "];merge:";
  (match opt.LL.merge_node with None -> add "-" | Some tn -> emit_tn tn);
  add ";"

let canonicalize ?(static_indices = []) ?(with_placements = true) (opt : LL.optimized) : canonical =
  let plc = opt.LL.optimize_ctx.LL.placements in
  (* The effective placement class enters the digest too (Codex P1 on PR #140): the optimized code
     can be identical while placements differ — [Local] scratch vs an [On_device] buffer — and the
     generated backend code then differs in kind and performance, so such programs must not share
     cache keys. In particular the placement-A/B arms of [Train.tune_placements] would otherwise
     cache-hit each other's entries whenever their code diverges only in placements, skipping the
     second arm's measurement. Placements of nodes reaching the optimized code are settled by the
     end of the pipeline; render an undecided node defensively rather than assert.

     [with_placements = false] gives the structural identity: placement classes can render
     differently across compilation lineages on byte-identical code (decided in one, undecided in
     the other), so per-segment schedule matching in fissioned replays keys on structure only, while
     disk-cache keys and post-schedule dedup keep the placement-aware form. *)
  let node_tag tn = if with_placements then placement_class plc tn else "" in
  canonical_of ~static_indices ~node_tag ~companions:(companions opt) opt.LL.llc

(* The decision problem's identity (gh-ocannl-786): the raw program, and per node what the lineage
   brings to the decision — its effective placement (a prior decision, or the declared intent the
   lookup falls back to) and the inline / footprint preferences the lineage recorded. Nothing this
   specialization decided enters: [lineage] is the placements table of the CONTEXT the lowering was
   decided from ({!Context.placements}), not [opt]'s post-decision table, and the preferences are
   inputs the optimizer reads and never writes. *)
(* What one lineage brings to a node's placement decision: its effective placement, whether that
   placement is a heuristic cap's (flippable back by [Context.decide_inline] -- the same mode imposed
   by legality or intent is not, so two lineages agreeing on the mode can still pose different
   refinement surfaces), and the inline / footprint preferences the lineage recorded. *)
let lineage_tag (plc : Tn.Placements.t) (octx : LL.optimize_ctx) tn =
  (match Tn.Placements.get plc tn with
    | None -> ";u"
    | Some (m, p) ->
        ";"
        ^ Sexp.to_string (Tn.sexp_of_memory_mode m)
        ^ if LL.is_cap_provenance p then "/cap" else "")
  ^ (if Hash_set.mem octx.LL.inline_preferences tn then ";i" else "")
  ^ if Hash_set.mem octx.LL.footprint_preferences tn then ";f" else ""

let canonicalize_source ?(static_indices = []) ?(node_tag = fun _ -> "")
    ~(lineage : Tn.Placements.t) (opt : LL.optimized) : canonical =
  let node_tag tn = lineage_tag lineage opt.LL.optimize_ctx tn ^ node_tag tn in
  canonical_of ~static_indices ~node_tag ~companions:(fun ~add:_ ~emit_tn:_ -> ()) opt.LL.source

(** {2 Registries} *)

type registry = {
  canonical : canonical;
  fwd : sym_ref Map.M(Idx.Symbol).t;
  bwd : Idx.symbol Map.M(Sym_ref).t;
  n_ops : int;
}

let base_registry canonical =
  let bwd =
    Map.fold canonical.base_syms
      ~init:(Map.empty (module Sym_ref))
      ~f:(fun ~key ~data acc -> Map.set acc ~key:data ~data:key)
  in
  { canonical; fwd = canonical.base_syms; bwd; n_ops = 0 }

let resolve r s = Map.find r.fwd s
let resolve_tn r tn = Map.find r.canonical.tn_refs tn

let record r sym sref =
  { r with fwd = Map.set r.fwd ~key:sym ~data:sref; bwd = Map.set r.bwd ~key:sref ~data:sym }

let resolve_exn r s =
  match resolve r s with
  | Some sref -> sref
  | None ->
      invalid_arg
        ("Schedule_cache.to_saved: cannot resolve symbol " ^ Idx.symbol_ident s
       ^ " (not a canonical loop binder, static index, or schedule-minted symbol)")

let resolve_tn_exn r tn =
  match resolve_tn r tn with
  | Some i -> i
  | None ->
      invalid_arg
        ("Schedule_cache.to_saved: cannot resolve tensor node " ^ Tn.debug_name tn
       ^ " (absent from the canonicalized code)")

let unresolve_exn r sref =
  match Map.find r.bwd sref with
  | Some s -> s
  | None ->
      invalid_arg
        (Printf.sprintf "Schedule_cache.of_saved: dangling reference %s"
           (Sexp.to_string (sexp_of_sym_ref sref)))

let to_saved r (sched : Schedule.schedule) : saved_schedule * registry =
  let r, rev_saved =
    List.fold sched ~init:(r, []) ~f:(fun (r, acc) op ->
        let idx = r.n_ops in
        let r, saved =
          match op with
          | Schedule.Split { axis; factor; outer; inner; outer_index; inner_index } ->
              let saved = Split { axis = resolve_exn r axis; factor; outer; inner } in
              let r = record r outer_index (Minted (idx, Split_outer)) in
              let r = record r inner_index (Minted (idx, Split_inner)) in
              (r, saved)
          | Schedule.Swap { outer; inner } ->
              (r, Swap { outer = resolve_exn r outer; inner = resolve_exn r inner })
          | Schedule.Retype { axis; ty } -> (r, Retype { axis = resolve_exn r axis; ty })
          | Schedule.Unroll { axis; materialize } ->
              (r, Unroll { axis = resolve_exn r axis; materialize })
          | Schedule.Partition { axis; breakpoints; segment_indices } ->
              let saved = Partition { axis = resolve_exn r axis; breakpoints } in
              let r =
                List.foldi segment_indices ~init:r ~f:(fun j r s ->
                    record r s (Minted (idx, Partition_seg j)))
              in
              (r, saved)
          | Schedule.Pad { axis; to_multiple_of } ->
              (r, Pad { axis = resolve_exn r axis; to_multiple_of })
          | Schedule.Coalesce { outer; inner; merged } ->
              let saved = Coalesce { outer = resolve_exn r outer; inner = resolve_exn r inner } in
              (record r merged (Minted (idx, Coalesce_merged)), saved)
          | Schedule.Stage
              {
                source;
                tile_loops;
                shared;
                cooperative;
                hoisted;
                swizzle;
                pad_stride;
                pipeline_depth;
                tile_prec;
              } ->
              ( r,
                Stage
                  {
                    source = resolve_tn_exn r source;
                    tile_loops = List.map tile_loops ~f:(resolve_exn r);
                    shared;
                    cooperative;
                    hoisted;
                    swizzle;
                    pad_stride;
                    pipeline_depth = (if pipeline_depth = 1 then None else Some pipeline_depth);
                    tile_prec;
                  } )
          | Schedule.Privatize { target; over; acc_prec } ->
              ( r,
                Privatize { target = resolve_tn_exn r target; over = resolve_exn r over; acc_prec }
              )
          | Schedule.Expand_zero { tn; indices } ->
              let r =
                List.foldi indices ~init:r ~f:(fun j r s ->
                    record r s (Minted (idx, Expand_axis j)))
              in
              (r, Expand_zero { tn = resolve_tn_exn r tn })
          | Schedule.Tensorize { i; j; k; lane; simd_width; tile } ->
              let saved =
                Tensorize
                  {
                    i = resolve_exn r i;
                    j = resolve_exn r j;
                    k = resolve_exn r k;
                    simd_width;
                    tile;
                  }
              in
              (record r lane (Minted (idx, Tensorize_lane)), saved)
          | Schedule.Fuse_epilogue { target; shared } ->
              (r, Fuse_epilogue { target = resolve_tn_exn r target; shared })
          | Schedule.Fold_mma { query; lane; block; width } ->
              let saved = Fold_mma { query = resolve_exn r query; width } in
              let r = record r lane (Minted (idx, Fold_mma_lane)) in
              (record r block (Minted (idx, Fold_mma_block)), saved)
          | Schedule.Split_reduce
              { axis; target; num_blocks; block_index; inner_index; combine_indices } ->
              let saved =
                Split_reduce
                  { axis = resolve_exn r axis; target = resolve_tn_exn r target; num_blocks }
              in
              let r = record r block_index (Minted (idx, Split_reduce_block)) in
              let r = record r inner_index (Minted (idx, Split_reduce_inner)) in
              let r =
                List.foldi combine_indices ~init:r ~f:(fun j r s ->
                    record r s (Minted (idx, Split_reduce_combine j)))
              in
              (r, saved)
        in
        ({ r with n_ops = idx + 1 }, saved :: acc))
  in
  (List.rev rev_saved, r)

let of_saved canonical (saved : saved_schedule) : Schedule.schedule * registry =
  let r, rev_sched =
    List.fold saved
      ~init:(base_registry canonical, [])
      ~f:(fun (r, acc) saved_op ->
        let idx = r.n_ops in
        let r, op =
          match saved_op with
          | Split { axis; factor; outer; inner } ->
              let op, outer_index, inner_index =
                Schedule.split ~axis:(unresolve_exn r axis) ~factor ~outer ~inner
              in
              let r = record r outer_index (Minted (idx, Split_outer)) in
              let r = record r inner_index (Minted (idx, Split_inner)) in
              (r, op)
          | Swap { outer; inner } ->
              (r, Schedule.Swap { outer = unresolve_exn r outer; inner = unresolve_exn r inner })
          | Retype { axis; ty } -> (r, Schedule.Retype { axis = unresolve_exn r axis; ty })
          | Unroll { axis; materialize } ->
              (r, Schedule.Unroll { axis = unresolve_exn r axis; materialize })
          | Partition { axis; breakpoints } ->
              let op, segment_indices =
                Schedule.partition ~axis:(unresolve_exn r axis) ~breakpoints
              in
              let r =
                List.foldi segment_indices ~init:r ~f:(fun j r s ->
                    record r s (Minted (idx, Partition_seg j)))
              in
              (r, op)
          | Pad { axis; to_multiple_of } ->
              (r, Schedule.Pad { axis = unresolve_exn r axis; to_multiple_of })
          | Coalesce { outer; inner } ->
              let op, merged =
                Schedule.coalesce ~outer:(unresolve_exn r outer) ~inner:(unresolve_exn r inner)
              in
              (record r merged (Minted (idx, Coalesce_merged)), op)
          | Stage
              {
                source;
                tile_loops;
                shared;
                cooperative;
                hoisted;
                swizzle;
                pad_stride;
                pipeline_depth;
                tile_prec;
              } ->
              ( r,
                Schedule.Stage
                  {
                    source = tn_of_ref canonical source;
                    tile_loops = List.map tile_loops ~f:(unresolve_exn r);
                    shared;
                    cooperative;
                    hoisted;
                    swizzle;
                    pad_stride;
                    pipeline_depth = Option.value pipeline_depth ~default:1;
                    tile_prec;
                  } )
          | Privatize { target; over; acc_prec } ->
              ( r,
                Schedule.Privatize
                  { target = tn_of_ref canonical target; over = unresolve_exn r over; acc_prec } )
          | Expand_zero { tn } ->
              let op, indices = Schedule.expand_zero ~tn:(tn_of_ref canonical tn) in
              let r =
                List.foldi indices ~init:r ~f:(fun j r s ->
                    record r s (Minted (idx, Expand_axis j)))
              in
              (r, op)
          | Tensorize { i; j; k; simd_width; tile } ->
              let op, lane =
                Schedule.tensorize ?tile ~i:(unresolve_exn r i) ~j:(unresolve_exn r j)
                  ~k:(unresolve_exn r k) ~simd_width ()
              in
              (record r lane (Minted (idx, Tensorize_lane)), op)
          | Fuse_epilogue { target; shared } ->
              (r, Schedule.Fuse_epilogue { target = tn_of_ref canonical target; shared })
          | Fold_mma { query; width } ->
              let op, lane, block = Schedule.fold_mma ~query:(unresolve_exn r query) ~width in
              let r = record r lane (Minted (idx, Fold_mma_lane)) in
              (record r block (Minted (idx, Fold_mma_block)), op)
          | Split_reduce { axis; target; num_blocks } ->
              let op, block_index, inner_index, combine_indices =
                Schedule.split_reduce ~axis:(unresolve_exn r axis)
                  ~target:(tn_of_ref canonical target) ~num_blocks
              in
              let r = record r block_index (Minted (idx, Split_reduce_block)) in
              let r = record r inner_index (Minted (idx, Split_reduce_inner)) in
              let r =
                List.foldi combine_indices ~init:r ~f:(fun j r s ->
                    record r s (Minted (idx, Split_reduce_combine j)))
              in
              (r, op)
        in
        ({ r with n_ops = idx + 1 }, op :: acc))
  in
  (List.rev rev_sched, r)

(** {2 The disk cache} *)

(* gh-ocannl-568: the numerics policy is NOT a property of the code, so it cannot reach the
   canonical digest — it is chosen by the user and consulted at codegen ([Numerics.get] in the
   backends' mma and narrow-arithmetic paths) and by the tile-shape choice of the autotune sketches.
   Two processes differing only in it therefore lower to byte-identical code with an equal digest
   while generating different kernels, and a winner tuned under one policy would replay under the
   other: measured at 5.9x SLOWER than not tuning at all when a default-flags run replayed a
   tf32-tuned tensorized schedule, whose rendering degrades to the scalar fallback under the
   stricter numerics. It also breaks {!Numerics}'s invariant that the policy is identical across
   sibling candidates — a replayed winner is a candidate from another policy regime. So the policy
   enters the disk-cache key (and the entry, below), which is where the hazard lives: within one
   process the policy is fixed, across processes only the cache carries schedules. *)
let numerics_tag () =
  let policy = Numerics.fingerprint (Numerics.get ()) in
  let algebra = Utils.get_global_arg ~default:"all" ~arg_name:"simplify_fp_algebra" in
  (* Schedule.apply simplifies again after transforms such as materializing unroll. Two identical
     base programs may expose different algebra there. Preserve existing all-on cache identities. *)
  let policy =
    if String.equal algebra "all" then policy else policy ^ ";simplify_fp_algebra=" ^ algebra
  in
  String.prefix (Stdlib.Digest.to_hex (Stdlib.Digest.string policy)) 8

(* gh-ocannl-572: the same argument as the numerics tag, for the rest of the settings a backend
   consults when it renders, compiles or dispatches a kernel. They come in two layers: the
   backend-independent ones handled here, and the backend's own, which it reports through
   [hardware_limits.codegen_tag] because only it knows which of its knobs reach its codegen. Neither
   layer is a property of the lowered code, so neither can reach {!digest}. What the numerics policy
   resolves to per backend is neither a knob nor the policy itself; it is derived from the backend's
   [codegen_capabilities] below (gh-ocannl-1117). *)
let codegen_tag ~(limits : Backend_intf.hardware_limits)
    ~(capabilities : Backend_intf.codegen_capabilities) () =
  (* Every key is spelled out at its call site, as an explicit [arg_name] string literal, rather
     than passed through a helper: that literal IS how the consistency tests find a configuration
     read ([Test_utils.Config_key_scan]), so a wrapper taking the name as an argument would hide
     these keys from both the registration check and the classification check -- which a negative
     control on this very function confirmed while addressing Codex's scan-list finding on PR #337.
     (For the same reason, prose here avoids spelling the marker the scanner looks for.) *)
  let gate name value = if value then name else "no-" ^ name in
  let parts =
    [
      (* The whole limits record, not just the backend's [codegen_tag] field (Codex P1 on PR #337):
         [backend] is the backend NAME, so without this two GPUs of one backend -- differing in
         compute capability, mma formats, shared-memory capacity, thread limits -- share every key
         while generating, rendering and timing candidates differently. The record is exactly the
         device description schedule construction already consults, so hashing it keeps the key as
         discriminating as the decisions it stands for, and a device's own [codegen_tag] rides along
         in it. *)
      Sexp.to_string (Backend_intf.sexp_of_hardware_limits limits);
      (* What the numerics policy RESOLVES to on this backend (gh-ocannl-1117): the backend's own
         compute and accumulator resolution functions, tabulated over every precision, and its mma
         arm table (gh-ocannl-1153: CUDA's tf32 arm, which [accum_prec] cannot tell from no arm).
         The numerics tag hashes the configured mode, which is not the same fact — HIP's [Bf16_auto]
         went wide in gh-ocannl-1051 with the mode unchanged, and the component that kept its old
         winners from replaying was added by hand. Derived from the functions codegen calls, the
         identity now moves exactly when some backend's resolution does. *)
      Backend_intf.codegen_capabilities_fingerprint capabilities;
      (if Utils.settings.large_models then "wide-index" else "narrow-index");
      (* The EFFECTIVE predicate, not the raw flag (Codex P1 on PR #337): the gate additionally
         requires [log_level > 1], so hashing the flag alone would give the logged and the unlogged
         regime one key — and hashing [log_level] itself would churn keys on an ordinary verbosity
         bump that changes no kernel. Routine logging rewrites the kernel and disables the
         parallel-grid, vectorized and mma renderings. Its sibling [Utils.with_runtime_debug] lives
         in the CUDA and HIP tags instead: only their compilers read it ([--device-debug] / [-g]),
         so hashing it here would re-tune cc and Metal for nothing (Codex P2). *)
      gate "routine-logs" (Utils.debug_log_from_routines ());
      (* Where routine logs go matters only when there are routine logs: with the gate off, nothing
         generated or timed depends on it, and hashing it would re-tune for nothing (Codex P2 on PR
         #337). *)
      gate "stream-logs"
        (Utils.debug_log_from_routines ()
        && Utils.get_global_flag ~default:false ~arg_name:"debug_log_to_stream_files");
      (* Not which backend runs — how the C-family backends spell their logging expressions
         ([full_printf_support]: [%g] vs scaled integers). Reaches emitted code only through the
         logging statements, hence the same gate as the stream-log routing above. *)
      gate "uniform-logs"
        (Utils.debug_log_from_routines ()
        && Utils.get_global_flag ~default:false ~arg_name:"prefer_backend_uniformity");
      (* An aliasing candidate's kernel parameter drops its [restrict] qualifier, since the
         link-time liveness planner may overlap it with another parameter's bytes (gh-ocannl-489): a
         real change to the emitted C, and to what the C compiler may then assume. Codex P1 on PR
         #337. *)
      gate "buffer-aliasing" (Utils.get_global_flag ~default:false ~arg_name:"buffer_aliasing");
    ]
  in
  String.prefix (Stdlib.Digest.to_hex (Stdlib.Digest.string (String.concat ~sep:"\000" parts))) 8

type trajectory = {
  search_shape : string;
      (** The storing search's candidate-shaping inputs the key does not carry, rendered by
          [Autotune.tune]: every [Search_shaping] configuration key that some source sets
          ([Utils.config_class_fingerprint]) and the arguments overriding the ones it reads. A
          trajectory is an equal-depth record only for a search that times the same candidates in
          the same order, so a replay under another shape reads it as absent. *)
  steps : (int * float) list;
}
[@@deriving sexp]
(** A search's timed record (gh-ocannl-1110), as a cache entry keeps it. *)

type saved_segment = {
  seg_kind : [ `Normal | `Zeros | `Solo ];
  seg_units : int;  (** The segment's length in units ({!Schedule.segmentation}). *)
  seg_digest : string;
      (** The {e pre-schedule} segment's structural canonical digest ([with_placements:false]): what
          a replay checks the segment it cut against before applying [seg_saved]. *)
  seg_saved : saved_schedule;
      (** The segment's schedule, resolved against that canonical form — [`Zeros] expansions and the
          empty schedule of a [`Solo] segment included, so replay derives none of them. *)
}
[@@deriving sexp]
(** One segment of a fissioned winner, in segment order (gh-ocannl-1164). *)

let save_segments ?static_indices segmentation tuples =
  List.map2_exn segmentation tuples ~f:(fun (seg_kind, seg_units) (_, pre, sched, _) ->
      let pre_canon = canonicalize ?static_indices ~with_placements:false pre in
      let seg_saved, registry = to_saved (base_registry pre_canon) sched in
      ({ seg_kind; seg_units; seg_digest = digest pre_canon; seg_saved }, registry))

let segmentation_of segs = List.map segs ~f:(fun s -> (s.seg_kind, s.seg_units))

type entry = {
  version : int;
  backend : string;
  numerics : string;
      (** {!numerics_tag} of the policy the search ran under (gh-ocannl-568). Redundant with the
          key, which carries the same tag — a self-description of the file, and a guard for a
          hand-moved or hand-written entry. *)
  codegen : string option; [@sexp.option]
      (** {!codegen_tag} of the codegen configuration the search ran under (gh-ocannl-572), the same
          self-description as [numerics] and with the same belt-and-braces role. Optional so that
          entries written before this field existed stay readable. *)
  objective : string option; [@sexp.option]
      (** The autotuner's timing objective ([autotune_timing]) the search ran under (gh-ocannl-755),
          the same self-description as [numerics] and [codegen] and with the same belt-and-braces
          role beside the key's own [timing] component. Optional so entries written before this
          field existed stay readable — they are under a different key anyway, so nothing looks them
          up. *)
  source_digest : string;
  saved : saved_schedule;
  segments : saved_segment list option; [@sexp.option]
      (** A fissioned winner: every segment of its fission, in order — the segmentation itself and
          each segment's schedule (gh-ocannl-1164). Replay cuts the routine where these segments say
          ({!Schedule.fission_segmented}'s [replay]) and applies each segment's own schedule, by
          position, after checking the segment's digest, so nothing about the segmentation is
          re-derived under the replaying process's policy and none of that policy's inputs needs to
          be in the key. [None] for whole-routine schedules. With [segments] present, [saved] is
          empty except for a split-reduce winner (gh-ocannl-484 task 3), where it holds the
          whole-routine prelude — resolved against the {e base} canonical form and applied before
          fission, the segments then describing the {e post-prelude} routine. *)
  best_ms : float;
  baseline_ms : float;
  default_ms : float option; [@sexp.option]
      (** The untuned default pipeline's measured time from the search that wrote the entry, for
          diagnostics (gh-ocannl-552). [None] when the default seed was not timed, or for entries
          written before this field existed — optional so such entries stay readable without an
          [entry_version] bump. *)
  mma_best_ms : float option; [@sexp.option]
      (** The best TIMED tensorized candidate of the search that wrote the entry (gh-ocannl-579),
          structural rather than label-keyed, and absent when it timed none — or for entries written
          before this field existed, which therefore replay as "nothing is known". A measurement of
          the PROGRAM, like [best_ms] and [baseline_ms] and under the same key regime, which is what
          makes it replayable: the flip chain's profitability term reads it, so without it a warm
          cache would rank the decision surface differently from the cold run that measured it. *)
  default_fingerprint : string option; [@sexp.option]
      (** {!Schedule.default_schedule_fingerprint} at store time, present iff [default_ms] is: the
          cache key covers only the source digest and the backend, so a config change can redefine
          what "the default pipeline" means without missing the cache. A replaying process compares
          fingerprints and drops a stale [default_ms] (the schedule itself stays valid — only this
          diagnostic is config-relative). *)
  best_steps : trajectory option; [@sexp.option]
      (** The storing search's best-so-far as a step function of its admitted timings
          ([Autotune.report.best_steps], gh-ocannl-1110): a measurement of the program like
          [mma_best_ms], replayed for the same reason — the flip chain abandons a hopeless flip
          against the incumbent's timed record, so without it an incumbent that replayed would leave
          every flip to run its full search. Absent for entries written before the field, which
          replay as "no record". *)
}
[@@deriving sexp]

(* Bumped on a decode-incompatible payload change — and on a SEARCH-MENU change (gh-ocannl-728). A
   stored crown is the best of the menu that searched it; once the menu offers a candidate the search
   never timed, the entry is still a sound schedule but no longer the answer the key asks for, and a
   warm cache would replay it forever. Non-current entries read as misses, so the next search
   re-tunes and overwrites. A bump appends its own line to the history below, which
   [test/operations/cache_version_history] holds to the constant: two parallel bumps then conflict
   in git rather than merging onto one number. A gap is a value claimed by a change that landed
   under a later one; a retired value keeps its line, since entries stored under it may still be on
   disk. *)
(*= entry_version history -- one line per value, oldest first; a bump appends one:
   1: staging#103 -- the first payload: canonical schedule identities and the disk cache
   2: staging#104 -- a fissioned winner's per-segment schedules
   3: gh-ocannl-470 -- the hoisted [Stage] schedule directive
   4: gh-ocannl-568 -- the numerics policy the entry was tuned under
   5: gh-ocannl-728 -- the [bgrid-in] batch flavor of the GPU matmul sketches
   6: retired -- staging#934 (gh-ocannl-1175, reverted): a menu bridging sibling reductions
   7: gh-ocannl-1166 -- the composite playoff, which times candidates the earlier search never did
   8: gh-ocannl-1164 -- a fissioned winner's [segments] carry the segmentation and each schedule
   11: gh-ocannl-1183 -- backprop's contractions become matmul sites through an interchange
   12: gh-ocannl-1175 -- a reduction's zero folds into its own segment's sketches
   13: gh-ocannl-1165 -- the coalesced layout's seeds, a search branch beside the sketch families
*)
let entry_version = 13

let sanitize name =
  String.map name ~f:(fun c ->
      if Char.is_alphanum c || Char.equal c '-' || Char.equal c '_' then c else '_')

(* gh-ocannl-755: the autotuner's timing objective, normalized to the spelling the key carries. Read
   from configuration here the way the numerics and codegen tags read theirs, so a caller that has
   not resolved a mode of its own keys against the one a search in this process would use. A caller
   that HAS resolved one — [Autotune.tune] with an explicit [~timing] — passes it instead: the
   configuration is then not what the times were taken under. A spelling this module does not know
   is passed through rather than rejected; the setting is validated where it is acted on, and a
   cache key's job here is only to keep unlike regimes apart. *)
let objective_tag () =
  sanitize
    (String.lowercase
       (String.strip (Utils.get_global_arg ~arg_name:"autotune_timing" ~default:"queued")))

(* The named components of a cache key, in the order they are concatenated. This list DRIVES
   {!cache_key} (each name dispatches to an arm below, and an unknown name raises), so the
   enumeration cannot go stale against the implementation — which is what makes it usable as the
   thing the digest-completeness registry classifies config keys against (gh-ocannl-572). *)
let key_components = [ "digest"; "backend"; "numerics"; "codegen"; "pool"; "device"; "timing" ]

(* The generation of the CUDA/HIP queued timing policy, which {!cache_key} spells into the [timing]
   component as [queued-v<N>] so an old winner cannot bypass its repaired calibration. The public
   setting and the entry description stay [queued]; only the filename identity changes, and only on
   these backends, whose policy changed (cc and Metal keep the bare [queued]). Generation 1 is that
   bare spelling, from before the policy was versioned. A bump appends its own line below. *)
(*= queued_objective_version history -- one line per value, oldest first; a bump appends one:
   1: gh-ocannl-755 -- the queued objective, spelled bare [queued] on every backend
   2: gh-ocannl-892 -- the ~10 ms window premise restored from depth-200 windows
   3: gh-ocannl-1144 -- cap-directed validations; sub-5 us kernels had stayed at 2--5 ms windows
*)
let queued_objective_version = 3

let cache_key ?objective ~timing_identity ~(limits : Backend_intf.hardware_limits) ~capabilities
    canonical ~backend =
  Option.map timing_identity ~f:(fun identity ->
      let objective = match objective with Some o -> sanitize o | None -> objective_tag () in
      let objective =
        match (String.lowercase backend, objective) with
        | ("cuda" | "hip"), "queued" -> "queued-v" ^ Int.to_string queued_objective_version
        | _ -> objective
      in
      let component = function
        | "digest" -> canonical.digest
        | "backend" -> sanitize backend
        | "numerics" -> "n" ^ numerics_tag ()
        | "codegen" -> "c" ^ codegen_tag ~limits ~capabilities ()
        (* The worker-pool signature (gh-ocannl-530): CPU crowns do not transfer across pools, so a
           pool change re-tunes instead of replaying. [None] (GPU backends) contributes nothing. *)
        | "pool" -> (
            match limits.worker_pool_tag with None -> "" | Some tag -> "p" ^ sanitize tag)
        (* The autotuner's timing objective (gh-ocannl-755). Isolated timing -- one launch plus one
           host sync -- and queued timing crown DIFFERENT candidates, measured, so an entry crowned
           under one is not the answer to a search asking the other, and its stored times are
           readings of a different quantity. Unlike the pool component this never contributes
           nothing: a key that omitted the objective on some path would let the two regimes share a
           file. *)
        | "device" ->
            "d"
            ^ Stdlib.Digest.to_hex
                (Stdlib.Digest.string
                   (Sexp.to_string (Backend_intf.sexp_of_timing_identity identity)))
        | "timing" -> "t" ^ objective
        | other -> invalid_arg ("Schedule_cache.cache_key: unhandled key component " ^ other)
      in
      String.concat ~sep:"-"
        (List.filter_map key_components ~f:(fun name ->
             match component name with "" -> None | part -> Some part)))

let ensure_dir = Utils.Atomic_file.ensure_dir
let cache_file ~dir ~key = Stdlib.Filename.concat dir (sanitize key ^ ".sexp")

(* The key REGIME is deliberately independent of [entry_version]. The latter says whether the
   payload at a key can be decoded; this stamp says whether the directory's filenames were minted by
   the same [key_components] schema. Bump this once when that schema changes — or when an input it
   leaves out changes what a key stands for. Cache-open then discards the superseded generation
   wholesale, with no migration arm for each historical schema (gh-ocannl-835). A bump appends its
   own line to the history below, as [entry_version]'s does. *)
(*= cache_regime_version history -- one line per value, oldest first; a bump appends one:
   1: gh-ocannl-835 -- the regime stamp itself
   2: gh-ocannl-594 -- the [device] component: a concrete timing identity per device
   3: gh-ocannl-1126 -- the [fission] component; empty by default, but default segmentation moved
   4: gh-ocannl-1124 -- lanes for nests with a preamble reduction change the default segmentation
   5: gh-ocannl-1167 -- lane geometry gated per device: HIP segmentation moves under unchanged keys
   6: gh-ocannl-1164 -- [fission] is gone: a fissioned winner persists its own segmentation
*)
let cache_regime_version = 6
let regime_stamp_filename = ".ocannl-schedule-cache-regime"
let regime_lock_filename = ".ocannl-schedule-cache.lock"
let regime_stamp_file dir = Stdlib.Filename.concat dir regime_stamp_filename
let regime_lock_file dir = Stdlib.Filename.concat dir regime_lock_filename

(* POSIX record locks are process-scoped, so two Domains of one process do not serialize each other
   through [lockf]. This mutex supplies that half; the permanent lock file supplies the
   cross-process half and is never unlinked, avoiding the unlink/recreate inode race. *)
let cache_open_mutex = Stdlib.Mutex.create ()

let remove_entry path =
  match Unix.unlink path with
  | () -> true
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> true
  | exception Unix.Unix_error _ -> false

let sweep_superseded_entries dir =
  match Stdlib.Sys.readdir dir with
  | exception Stdlib.Sys_error _ -> false
  | names ->
      Array.fold names ~init:true ~f:(fun removed name ->
          if String.is_suffix name ~suffix:".sexp" then
            remove_entry (Stdlib.Filename.concat dir name) && removed
          else removed)

type regime_stamp = Missing | Version of int | Refuse

let read_regime_stamp dir =
  let path = regime_stamp_file dir in
  if not (Stdlib.Sys.file_exists path) then Missing
  else
    match Int.of_string (String.strip (Stdio.In_channel.read_all path)) with
    | version -> Version version
    | exception _ -> Refuse

let write_regime_stamp dir =
  Utils.Atomic_file.write_all ~path:(regime_stamp_file dir)
    ~data:(Int.to_string cache_regime_version ^ "\n")
    ~before_commit:(fun () -> Resource_fault_injection.hit Schedule_cache_before_regime_commit)
    ()

let open_current_regime dir =
  match read_regime_stamp dir with
  | Version version when version = cache_regime_version -> true
  | Version version when version > cache_regime_version -> false
  | Refuse -> false
  | Missing | Version _ ->
      (* Deletions precede the atomic stamp publication. A crash or refusal before publication
         leaves the old stamp in place, so the next opener retries; a current-regime operation is
         admitted only after every old entry is gone and the new stamp is visible. *)
      if sweep_superseded_entries dir then (
        write_regime_stamp dir;
        true)
      else false

(* Why a cache-open did not admit its operation: the directory is absent (a lookup's ordinary miss
   before the first store, a store's refusal), or the lock, the regime stamp or the sweep of a
   superseded regime refused it. *)
type open_refusal = Missing_dir | Refused_open of string

let unix_refusal error fn arg = Printf.sprintf "%s %s: %s" fn arg (Unix.error_message error)

(* Whether [path] exists. [Sys.file_exists] answers [false] also when the filesystem refused the
   query (an ACL, a transient Windows refusal), which the cache-I/O record must not read as an
   ordinary absence: only [ENOENT]/[ENOTDIR] are. *)
let probe path =
  match Unix.stat path with
  | _ -> Ok true
  | exception Unix.Unix_error ((Unix.ENOENT | Unix.ENOTDIR), _, _) -> Ok false
  | exception Unix.Unix_error (error, fn, arg) -> Error (unix_refusal error fn arg)

let open_cache ~dir f =
  match probe dir with
  | Ok false -> Error Missing_dir
  | Error msg -> Error (Refused_open msg)
  | Ok true ->
      Stdlib.Mutex.lock cache_open_mutex;
      Stdlib.Fun.protect
        ~finally:(fun () -> Stdlib.Mutex.unlock cache_open_mutex)
        (fun () ->
          try
            let fd = Unix.openfile (regime_lock_file dir) [ Unix.O_CREAT; Unix.O_RDWR ] 0o666 in
            Stdlib.Fun.protect
              ~finally:(fun () -> Unix.close fd)
              (fun () ->
                Resource_fault_injection.hit Schedule_cache_before_lock;
                Unix.lockf fd Unix.F_LOCK 0;
                if open_current_regime dir then Ok (f ())
                else
                  Error
                    (Refused_open
                       "regime refused: a newer or malformed stamp, or a superseded entry that \
                        could not be removed"))
          with
          | Unix.Unix_error (error, fn, arg) -> Error (Refused_open (unix_refusal error fn arg))
          | Stdlib.Sys_error msg -> Error (Refused_open msg))

(* {2 The cache-I/O record} (gh-ocannl-1040)

   Every refusal the entry protocol absorbs is invisible by design -- the cache is an optimization
   -- which is right for a tuning run and wrong for a test that claims a store happened: on Windows
   a commit can outlive [Atomic_file]'s bounded retry, or the lock can refuse, and the claim then
   fails for a reason its own predicates never observe. So each store and each lookup notes what it
   came to, to whoever is recording; with nobody recording, nothing is kept. *)
type cache_op = Store | Lookup [@@deriving sexp_of]

type cache_io = { op : cache_op; dir : string; key : string; refusal : string option }
[@@deriving sexp_of]

let io_recorders : cache_io Queue.t list ref = ref []
let io_recorders_mutex = Stdlib.Mutex.create ()

(* Every critical section releases the mutex on any exception: an [Out_of_memory] or [Sys.Break]
   inside one must not leave every later cache operation of the process blocked on it. *)
let with_recorders f = Stdlib.Mutex.protect io_recorders_mutex f
let note_io io = with_recorders (fun () -> List.iter !io_recorders ~f:(fun q -> Queue.enqueue q io))

let recording_cache_io f =
  let q = Queue.create () in
  let detach () =
    with_recorders (fun () ->
        io_recorders := List.filter !io_recorders ~f:(fun q' -> not (phys_equal q q')))
  in
  with_recorders (fun () -> io_recorders := q :: !io_recorders);
  let result = Exn.protect ~f ~finally:detach in
  (result, Queue.to_list q)

(* One entry I/O protocol for every kind of entry the directory holds: schedule winners, abandoned
   searches and placement decisions (gh-ocannl-786) share the lock, the regime stamp and the atomic
   commit, and differ only in payload and version check. *)
let store_sexp ~dir ~key sexp =
  Option.iter key ~f:(fun key ->
      ensure_dir dir;
      let refusal =
        match
          open_cache ~dir (fun () ->
              (* A writer killed between staging and commit leaves its staging file behind; nothing
                 else in the process would ever remove it, and a cache directory is long-lived.
                 Sweep once per process, from the writers rather than on a timer. *)
              Utils.Atomic_file.cleanup_stale_once dir;
              let file = cache_file ~dir ~key in
              (* Uniqueness, failure cleanup and the Windows-safe commit all live in [Atomic_file]:
                 the committed entry is either the old complete file or the new complete file, never
                 an intention. The injection point sits in the staged-but-uncommitted window, which
                 is what makes that guarantee testable. *)
              try
                Utils.Atomic_file.write_all ~path:file ~data:(Sexp.to_string_hum sexp)
                  ~before_commit:(fun () ->
                    Resource_fault_injection.hit Schedule_cache_before_commit)
                  ();
                None
              with Stdlib.Sys_error msg ->
                (* The cache is an optimization, so a filesystem refusal -- a directory that turned
                   unwritable, a Windows peer still holding this entry open past the bounded commit
                   retry -- means the tuning result is not saved, not that the run fails. [publish]
                   has already removed the staging file; an earlier complete entry is still in
                   place. *)
                Some msg)
        with
        | Ok refusal -> refusal
        | Error Missing_dir -> Some (dir ^ ": the cache directory could not be created")
        | Error (Refused_open msg) -> Some msg
      in
      note_io { op = Store; dir; key; refusal })

(* The failures a cache read never absorbs: they are about the process, not the entry, and a miss
   that hides one turns Ctrl-C during a lookup into the start of a search it was meant to stop. *)
let process_level = function
  | Out_of_memory | Stdlib.Sys.Break | Stack_overflow -> true
  | _ -> false

let lookup_sexp ~dir ~key ~of_sexp ~current =
  Option.bind key ~f:(fun key ->
      let opened =
        open_cache ~dir (fun () ->
            Utils.Atomic_file.cleanup_stale_once dir;
            let file = cache_file ~dir ~key in
            match probe file with
            | Ok false -> Ok None
            | Error msg -> Error msg
            | Ok true -> (
                try
                  Resource_fault_injection.hit Schedule_cache_before_replay;
                  let entry = of_sexp (Sexplib.Sexp.load_sexp file) in
                  Ok (if current entry then Some entry else None)
                with
                (* The filesystem refusing an entry the directory listing just showed -- a Windows
                   peer holding it without share-read -- is a refusal; an entry that fails to parse
                   or decode is a miss the lookup decided. Both read as a miss. *)
                | Stdlib.Sys_error msg -> Error msg
                | exn when not (process_level exn) -> Ok None))
      in
      (* A missing directory is the ordinary miss before the first store. *)
      let refusal =
        match opened with
        | Ok (Error msg) | Error (Refused_open msg) -> Some msg
        | Ok (Ok _) | Error Missing_dir -> None
      in
      note_io { op = Lookup; dir; key; refusal };
      match opened with Ok (Ok entry) -> entry | Ok (Error _) | Error _ -> None)

let store ~dir ~key entry = store_sexp ~dir ~key (sexp_of_entry entry)

let lookup ~dir ~key =
  lookup_sexp ~dir ~key ~of_sexp:entry_of_sexp ~current:(fun e -> e.version = entry_version)

(** {2 Abandoned searches} *)

(* This is timed evidence, not a crowned schedule. Keep it beside the winner, under the very same
   key with a namespace prefix: adding this record changes no existing identity. *)
type abandonment_entry = { version : int; source_digest : string; trajectory : trajectory }
[@@deriving sexp]

let abandonment_key key = Option.map key ~f:(fun key -> "abandonment-" ^ key)

let store_abandonment ~dir ~key entry =
  store_sexp ~dir ~key:(abandonment_key key) (sexp_of_abandonment_entry entry)

let lookup_abandonment ~dir ~key =
  lookup_sexp ~dir ~key:(abandonment_key key) ~of_sexp:abandonment_entry_of_sexp ~current:(fun e ->
      e.version = entry_version)

(** {2 The placement-decision store} *)

type placement_flip = { node : int; flip : Low_level.reading } [@@deriving sexp, compare, equal]

type placement_decision = Default | Materialize_all | Refined of placement_flip list
[@@deriving sexp, compare, equal]

type placement_entry = {
  version : int;
  backend : string;
  numerics : string;
  codegen : string;
  objective : string;
  problem_digest : string;
  decision : placement_decision;
  outcome_digest : string;
  shipped_ms : float;
  arm_a_ms : float;
  arm_b_ms : float;
}
[@@deriving sexp]

(* Versions [placement_entry] payloads the way [entry_version] versions schedule entries. *)
(*= placement_entry_version history -- one line per value, oldest first; a bump appends one:
   1: gh-ocannl-786 -- the placement decision and the arm timings that chose it
*)
let placement_entry_version = 1

let placement_key ?objective ~timing_identity ~limits ~capabilities canonical ~backend =
  Option.map (cache_key ?objective ~timing_identity ~limits ~capabilities canonical ~backend)
    ~f:(fun key -> "placements-" ^ key)

let store_placements ~dir ~key entry = store_sexp ~dir ~key (sexp_of_placement_entry entry)

let lookup_placements ~dir ~key =
  lookup_sexp ~dir ~key ~of_sexp:placement_entry_of_sexp ~current:(fun e ->
      e.version = placement_entry_version)

let shipped_label = function Default -> "A" | Materialize_all -> "B" | Refined _ -> "flip"
