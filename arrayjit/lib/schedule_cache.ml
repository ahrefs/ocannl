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
  | Privatize of { target : int; over : sym_ref }
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
let canonicalize ?(static_indices = []) ?(with_placements = true) (opt : LL.optimized) : canonical =
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
  let plc = opt.LL.optimize_ctx.LL.placements in
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
        (* The effective placement class enters the digest too (Codex P1 on PR #140): the optimized
           code can be identical while placements differ — [Local] scratch vs an [On_device] buffer
           — and the generated backend code then differs in kind and performance, so such programs
           must not share cache keys. In particular the placement-A/B arms of
           [Train.tune_placements] would otherwise cache-hit each other's entries whenever their
           code diverges only in placements, skipping the second arm's measurement. Placements of
           nodes reaching the optimized code are settled by the end of the pipeline; render an
           undecided node defensively rather than assert. *)
        let pc =
          (* [with_placements = false] gives the structural identity: placement classes can render
             differently across compilation lineages on byte-identical code (decided in one,
             undecided in the other), so per-segment schedule matching in fissioned replays keys on
             structure only, while disk-cache keys and post-schedule dedup keep the placement-aware
             form. *)
          if not with_placements then ""
          else
            match Tn.Placements.get plc tn with
            | None -> ";u"
            | Some (m, _) -> ";" ^ Sexp.to_string (Tn.sexp_of_memory_mode m)
        in
        add
          (Printf.sprintf "t%d=[%s;%s%s%s]" i
             (String.concat_array ~sep:"," (Array.map dims ~f:Int.to_string))
             (Sexp.to_string (Ops.sexp_of_prec (Lazy.force tn.Tn.storage_prec)))
             hc pc)
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
    opt.LL.llc;
  (* Codegen-relevant companions of the code. *)
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
  add ";";
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
          | Schedule.Privatize { target; over } ->
              (r, Privatize { target = resolve_tn_exn r target; over = resolve_exn r over })
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
          | Privatize { target; over } ->
              ( r,
                Schedule.Privatize
                  { target = tn_of_ref canonical target; over = unresolve_exn r over } )
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
  String.prefix
    (Stdlib.Digest.to_hex (Stdlib.Digest.string (Numerics.fingerprint (Numerics.get ()))))
    8

(* gh-ocannl-572: the same argument as the numerics tag, for the rest of the settings a backend
   consults when it renders, compiles or dispatches a kernel. They come in two layers: the
   backend-independent ones handled here, and the backend's own, which it reports through
   [hardware_limits.codegen_tag] because only it knows which of its knobs reach its codegen. Neither
   layer is a property of the lowered code, so neither can reach {!digest}. *)
let codegen_tag ~(limits : Backend_intf.hardware_limits) () =
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
  segments : (string * saved_schedule) list option; [@sexp.option]
      (** A fissioned winner: per-segment schedules keyed by the {e pre-schedule} segment's
          canonical digest ([saved] is then empty). [None] for whole-routine schedules. *)
  finer_fission : bool option; [@sexp.option]
      (** [Some true]: the [segments] keys address {!Schedule.fission_scheduled}'s [arity_cuts]
          (finer) segmentation (gh-ocannl-574); replay must re-segment under the same mode or the
          keys miss wholesale. Omitted when false, so entries stay byte-stable and pre-gh-574
          entries parse without an [entry_version] bump. *)
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
}
[@@deriving sexp]

let entry_version = 4

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
let key_components = [ "digest"; "backend"; "numerics"; "codegen"; "pool"; "timing" ]

let cache_key ?objective ~(limits : Backend_intf.hardware_limits) canonical ~backend =
  let objective = match objective with Some o -> sanitize o | None -> objective_tag () in
  (* gh-ocannl-892 changes what CUDA/HIP [queued] measures: the old depth-200 winner was ranked on a
     1--2.5 ms contention window, while generation 2 restores the ~10 ms premise. Keep the public
     setting and entry self-description as [queued], but version its filename identity on exactly
     the backends whose policy changed so an old winner cannot bypass the new measurement. *)
  let objective =
    match (String.lowercase backend, objective) with
    | ("cuda" | "hip"), "queued" -> "queued-v2"
    | _ -> objective
  in
  let component = function
    | "digest" -> canonical.digest
    | "backend" -> sanitize backend
    | "numerics" -> "n" ^ numerics_tag ()
    | "codegen" -> "c" ^ codegen_tag ~limits ()
    (* The worker-pool signature (gh-ocannl-530): CPU crowns do not transfer across pools, so a pool
       change re-tunes instead of replaying. [None] (GPU backends) contributes nothing. *)
    | "pool" -> ( match limits.worker_pool_tag with None -> "" | Some tag -> "p" ^ sanitize tag)
    (* The autotuner's timing objective (gh-ocannl-755). Isolated timing -- one launch plus one host
       sync -- and queued timing crown DIFFERENT candidates, measured, so an entry crowned under one
       is not the answer to a search asking the other, and its stored times are readings of a
       different quantity. Unlike the pool component this never contributes nothing: a key that
       omitted the objective on some path would let the two regimes share a file. *)
    | "timing" -> "t" ^ objective
    | other -> invalid_arg ("Schedule_cache.cache_key: unhandled key component " ^ other)
  in
  String.concat ~sep:"-"
    (List.filter_map key_components ~f:(fun name ->
         match component name with "" -> None | part -> Some part))

let ensure_dir = Utils.Atomic_file.ensure_dir
let cache_file ~dir ~key = Stdlib.Filename.concat dir (sanitize key ^ ".sexp")

(* The key REGIME is deliberately independent of [entry_version]. The latter says whether the
   payload at a key can be decoded; this stamp says whether the directory's filenames were minted by
   the same [key_components] schema. Bump this once when that schema changes. Cache-open then
   discards the superseded generation wholesale, with no migration arm for each historical schema
   (gh-ocannl-835). *)
let cache_regime_version = 1
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

let with_cache_open ~dir f =
  if not (Stdlib.Sys.file_exists dir) then None
  else (
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
              if open_current_regime dir then Some (f ()) else None)
        with Unix.Unix_error _ | Stdlib.Sys_error _ -> None))

let store ~dir ~key entry =
  ensure_dir dir;
  ignore
    (with_cache_open ~dir (fun () ->
         (* A writer killed between staging and commit leaves its staging file behind; nothing else
            in the process would ever remove it, and a cache directory is long-lived. Sweep once per
            process, from the writers rather than on a timer. *)
         Utils.Atomic_file.cleanup_stale_once dir;
         let file = cache_file ~dir ~key in
         (* Uniqueness, failure cleanup and the Windows-safe commit all live in [Atomic_file]: the
            committed entry is either the old complete file or the new complete file, never an
            intention. The injection point sits in the staged-but-uncommitted window, which is what
            makes that guarantee testable. *)
         try
           Utils.Atomic_file.write_all ~path:file
             ~data:(Sexp.to_string_hum (sexp_of_entry entry))
             ~before_commit:(fun () -> Resource_fault_injection.hit Schedule_cache_before_commit)
             ()
         with Stdlib.Sys_error _ ->
           (* The cache is an optimization, so a filesystem refusal — a directory that turned
              unwritable, a Windows peer still holding this entry open past the bounded commit retry
              — means the tuning result is not saved, not that the run fails. [publish] has already
              removed the staging file; an earlier complete entry is still in place. *)
           ()))

let lookup ~dir ~key =
  Option.join
    (with_cache_open ~dir (fun () ->
         Utils.Atomic_file.cleanup_stale_once dir;
         let file = cache_file ~dir ~key in
         if not (Stdlib.Sys.file_exists file) then None
         else
           try
             Resource_fault_injection.hit Schedule_cache_before_replay;
             let entry = entry_of_sexp (Sexplib.Sexp.load_sexp file) in
             if entry.version = entry_version then Some entry else None
           with _ -> None))
