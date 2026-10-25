(** {1 The sketch families: site detection and seed construction}

    The structured half of the autotuner's candidate space (gh-ocannl-580), factored out of
    {!Autotune}'s search harness: recognizing a matmul or convolution micro-kernel in a lowering,
    the composed schedule pipelines those sites parameterize ({!sketch_params}), and the refinement
    trees ({!Ir.Schedule_space.tree}) whose leaves {e are} the seed lists. Nothing here times,
    compiles, caches, or classifies failures — the construction depends only on the site types,
    {!Ir.Schedule_space} and a handful of {!Ir.Schedule} helpers, so it is reviewable, and
    extensible, in isolation from the beam search that consumes it.

    {!sketch_seed_params} composes the families (the matmul tree's leaves, else the conv seeds
    crossed with their epilogue-fusion twins) into the seed list the search actually enumerates.

    Its interface bounds the construction contracts consumed by the search harness: [autotune.ml]
    {e includes} this module, so those contracts are in scope unqualified, and [autotune.mli]
    selects the public search and inspection API. *)

open Base
module Sched = Ir.Schedule
module Sspace = Ir.Schedule_space
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Outcome = Ir.Schedule_outcome

(* gh-ocannl-1390: a deliberate sketch-builder precondition refusal -- no site of the family's kind,
   a companion nest the family's geometry cannot cover -- is a statement about the candidate, so it
   is raised typed at its own site, as [Schedule.refuse] is one level down: a search reads it as a
   declined candidate. Any OTHER exception escaping the builder (an array bound, a [List.zip_exn], a
   bare [invalid_arg]) is a builder bug and stays untyped, fatal under
   [strict_failure_classification]. The cause and census key are the ones the builder's former
   blanket [Invalid_argument] conversion produced. Every [invalid_arg]/[failwith] bound before
   [sketch_schedule] in this file is either this or carries an [(* invariant: ... *)] marker, which
   [test/operations/schedule_refusal_scan] enforces. *)
let refuse detail =
  raise
    (Outcome.Cause_at
       (Outcome.Transform, Outcome.Illegal_schedule { check = "Autotune.sketch"; detail }))

(** {2 Matmul detection and sketch schedules}

    Sketch candidates instantiate the composed matmul pipelines pinned by
    test/operations/schedule_register_matmul.ml (GPU register blocktiling: Split + Swap + shared
    Stage + Privatize + materializing Unroll) and schedule_cpu_pack_matmul.ml (CPU operand packing:
    Split + Swap + non-shared Stage + Privatize), parameterized by tile sizes. Detection is
    permissive — a mis-detected site fails its candidate compile (op preconditions,
    [validate_parallel], hardware limits) and is skipped like any other invalid candidate. *)

(* Which pair of a matmul site the coalesced tuner branch merges into one loop (gh-ocannl-1165,
   gh-ocannl-1279): the interior batch loop into the column role, or the innermost two contraction
   loops into one k. See the interface for the full account. *)
type coalesce_layout = Coalesce_columns | Coalesce_contraction [@@deriving sexp]

type sketch_params = {
  sk_gpu : bool;
  sk_mma : bool;
  sk_simd : int;
  sk_bm : int;
  sk_bn : int;
  sk_bk : int;
  sk_tm : int;
  sk_tn : int;
  sk_hoist : bool;
  sk_grid : bool;
  sk_pack_rest : bool;
  sk_conv : bool;
  sk_epilogue : bool;
  sk_batch_grid : bool;
  sk_batch_inner : bool;
  sk_coalesce : coalesce_layout option;
  sk_swizzle : LL.swizzle_kind option;
  sk_depth : int;
  sk_pack_prec : Ir.Ops.prec option;
  sk_tile : Ir.Register_tile.t option;
}
[@@deriving sexp]

(* Resolve the tensor-core input format from storage precision before seeding a typed matmul/conv
   site. Single-precision storage has two possible compute formats: prefer tf32 when the numerics
   policy enables it and the backend advertises that pair, then fall back to genuine f32 (Metal).
   Backends remain the emission source of truth; this only prevents the autotuner from rejecting a
   supported divergent tile up front, or timing a format the capability does not advertise. *)
let mma_input_formats_of_prec (prec : Ir.Ops.prec) : Ir.Backend_intf.mma_input_format list =
  match prec with
  | Ir.Ops.Half_prec _ -> [ Ir.Backend_intf.Mma_f16 ]
  | Ir.Ops.Bfloat16_prec _ -> [ Ir.Backend_intf.Mma_bf16 ]
  | Ir.Ops.Fp8_prec _ -> [ Ir.Backend_intf.Mma_fp8_e5m2 ]
  | Ir.Ops.Single_prec _ ->
      if (Ir.Numerics.get ()).Ir.Numerics.tf32_matmuls then
        [ Ir.Backend_intf.Mma_tf32; Ir.Backend_intf.Mma_f32 ]
      else [ Ir.Backend_intf.Mma_f32 ]
  | _ -> []

(* The accumulator format of a destination's storage precision (gh-ocannl-545). Unlike the
   multiplicands, this admits no policy choice: the accumulator is read back from and written to the
   node, so its format is its storage layout. In particular f32 storage accumulates as [Mma_f32]
   even under the tf32 policy — tf32 truncates the multiplicands, never the accumulator. *)
let mma_acc_format_of_prec (prec : Ir.Ops.prec) : Ir.Backend_intf.mma_input_format option =
  match prec with
  | Ir.Ops.Half_prec _ -> Some Ir.Backend_intf.Mma_f16
  | Ir.Ops.Bfloat16_prec _ -> Some Ir.Backend_intf.Mma_bf16
  | Ir.Ops.Single_prec _ -> Some Ir.Backend_intf.Mma_f32
  | _ -> None

(* The site's resolved format triples, in [mma_input_formats_of_prec]'s preference order. *)
let mma_format_triples ~a_prec ~b_prec ~d_prec =
  match mma_acc_format_of_prec d_prec with
  | None -> []
  | Some d_format ->
      List.concat_map (mma_input_formats_of_prec a_prec) ~f:(fun a_format ->
          List.map (mma_input_formats_of_prec b_prec) ~f:(fun b_format ->
              (a_format, b_format, d_format)))

(* gh-ocannl-680/836: under [Numerics.Fp16_wide] an f16-storage destination may tensorize only in an
   emission scope where the backend's uniform-f16 arm accumulates f32. CUDA sm_80+ supports the
   per-statement inline-PTX m16n8k16 scope and, since gh-ocannl-925, the persistent-fragment scope
   through wmma f32 fragments and a table-addressed converted d boundary; HIP's converted rocWMMA d
   boundary supports both since gh-ocannl-789, and Metal's converted [thread_elements()] boundary
   supports both since gh-ocannl-837. Consulting the scope list here keeps a staged outer-k split
   from acquiring an extra f16 boundary merely because the same intrinsic is wide over its inner
   tile.

   gh-ocannl-838: [Numerics.Bf16_wide] asks the same question of a bf16-storage destination, against
   [mma_bf16_wide_acc_scopes] — HIP's converted rocWMMA boundary serves both scopes, CUDA's
   inline-PTX arm (f32 accumulate in hardware) both since gh-ocannl-1063, and Metal's converted
   [thread_elements()] boundary both since gh-ocannl-923. The witness names whichever policy
   withheld the seed. *)
let wide_acc_withholding (mma : Ir.Backend_intf.mma_capability) ~scope ~d_prec =
  let withheld ~policy scopes =
    if List.mem scopes scope ~equal:Ir.Backend_intf.equal_mma_emission_scope then None
    else Some policy
  in
  match d_prec with
  | Ir.Ops.Half_prec _ when Ir.Numerics.fp16_accum_wide () ->
      withheld ~policy:"Fp16_wide requires a wide uniform-f16"
        mma.Ir.Backend_intf.mma_f16_wide_acc_scopes
  | Ir.Ops.Bfloat16_prec _ when Ir.Numerics.bf16_accum_wide () ->
      withheld ~policy:"Bf16_wide requires a wide uniform-bf16"
        mma.Ir.Backend_intf.mma_bf16_wide_acc_scopes
  | _ -> None

let wide_acc_withholds mma ~scope ~d_prec = Option.is_some (wide_acc_withholding mma ~scope ~d_prec)

let mma_tile_for_precisions (mma : Ir.Backend_intf.mma_capability) ~a_prec ~b_prec ~d_prec =
  List.find_map (mma_format_triples ~a_prec ~b_prec ~d_prec) ~f:(fun key ->
      List.Assoc.find mma.Ir.Backend_intf.mma_format_tiles key
        ~equal:Ir.Backend_intf.equal_mma_format_triple)

let mma_tile_for_precisions_in_scope (mma : Ir.Backend_intf.mma_capability) ~scope ~a_prec ~b_prec
    ~d_prec =
  if wide_acc_withholds mma ~scope ~d_prec then None
  else mma_tile_for_precisions mma ~a_prec ~b_prec ~d_prec

(* The compute precision the CPU register tiling runs a destination of this storage precision at,
   through the same [Numerics.cpu_compute_prec] the emission asks (gh-ocannl-575). *)
let cpu_register_tile_prec ~(limits : Ir.Backend_intf.hardware_limits) prec =
  Ir.Numerics.cpu_compute_prec ~native_fp16_arithmetic:limits.Ir.Backend_intf.native_fp16_arithmetic
    prec

(* gh-ocannl-1115: why the backend and the configuration withhold the tensorized family from a
   matmul site with these storage precisions, before any geometry or site structure is consulted --
   [None] when they do not. The family tree refutes its tensorized branch through this, and the
   tests that claim a tensorized seed's PRESENCE gate on it, so the gate is the seeder's own
   judgment and never a restatement of it (nor the seed list under test). On GPU: routine logging, a
   lane wider than the workgroup, no advertised format tile; on CPU, the register tiling's
   statically decidable rules (gh-ocannl-479) that do not depend on the site's shape: a usable
   vector file, two lanes at the compute precision, uniform vector-capable compute precisions, and
   routine logging. *)
let tensorized_capability_refutation ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits)
    ~a_prec ~b_prec ~d_prec =
  match (is_gpu, limits.Ir.Backend_intf.mma) with
  | true, Some _ when Utils.debug_log_from_routines () ->
      (* Same predicate the GPU [mma_syntax] paths consult: under routine logging the emission skips
         the intrinsic and renders the scalar fallback, so every leaf would be timed (and cached)
         under a tensorized label. *)
      Some
        "routine logging is active (debug_log_from_routines): the mma emission renders the scalar \
         fallback, so every leaf would time under a tensorized label"
  | true, Some { Ir.Backend_intf.mma_simd_width = w; _ }
    when Option.value_map limits.Ir.Backend_intf.max_threads_per_workgroup ~default:false
           ~f:(fun cap -> w > cap) ->
      (* The tensorization lane is a Workgroup axis of extent [w] in every geometry. *)
      Some
        (Printf.sprintf "mma lane width %d exceeds the %d-thread workgroup limit" w
           (Option.value_exn limits.Ir.Backend_intf.max_threads_per_workgroup))
  | true, Some mma -> (
      match mma_tile_for_precisions mma ~a_prec ~b_prec ~d_prec with
      | Some _ -> None
      | None ->
          Some
            (Printf.sprintf
               "backend advertises no mma format tile for operands (%s, %s) with accumulator %s"
               (Ir.Ops.prec_string a_prec) (Ir.Ops.prec_string b_prec) (Ir.Ops.prec_string d_prec)))
  | true, None -> Some "backend advertises no mma capability"
  | _ when is_cpu ->
      let native_fp16 = limits.Ir.Backend_intf.native_fp16_arithmetic in
      let comp_prec = cpu_register_tile_prec ~limits in
      let prec = comp_prec d_prec in
      let uniform_vec_capable =
        (match prec with
          | Ir.Ops.Single_prec _ | Ir.Ops.Double_prec _ -> true
          | Ir.Ops.Half_prec _ -> native_fp16
          | _ -> false)
        && Ir.Ops.equal_prec (comp_prec a_prec) prec
        && Ir.Ops.equal_prec (comp_prec b_prec) prec
        (* The C-tile accumulates at the compute precision, so a divergent accumulator residency
           ([Fp16_wide] + [narrow_compute_f32 = false] on an f16 destination) is an emission decline
           — mirror it here or the candidate is timed under a tensorized label (gh-ocannl-680; Codex
           P1 round 1 on staging PR #477). *)
        && Ir.Ops.equal_prec
             (Ir.Numerics.cpu_accum_prec ~native_fp16_arithmetic:native_fp16 d_prec)
             prec
      in
      let lanes = limits.Ir.Backend_intf.simd_vector.bytes / max 1 (Ir.Ops.prec_in_bytes prec) in
      List.find_map
        [
          ( limits.Ir.Backend_intf.simd_vector.bytes >= 8,
            lazy
              (Printf.sprintf "no usable SIMD vector file (simd_vector.bytes=%d < 8)"
                 limits.Ir.Backend_intf.simd_vector.bytes) );
          ( lanes >= 2,
            lazy
              (Printf.sprintf "fewer than two vector lanes at %s (simd_vector.bytes=%d)"
                 (Ir.Ops.prec_string prec) limits.Ir.Backend_intf.simd_vector.bytes) );
          ( uniform_vec_capable,
            lazy
              "register tiling requires uniform vector-capable compute precisions (f32/f64, or \
               fp16 where arithmetic is native)" );
          (* [Register_tile.check]'s register-file rule, at the seeding (gh-ocannl-1285): a file
             holding no four-row tile declines every geometry, the default included. *)
          ( Ir.Register_tile.rn_cap ~vector_registers:limits.Ir.Backend_intf.simd_vector.registers
            >= 1,
            lazy
              (Printf.sprintf
                 "the vector register file holds no %d-row register tile (simd_vector.registers=%d)"
                 Ir.Register_tile.rm_cap limits.Ir.Backend_intf.simd_vector.registers) );
          ( not (Utils.debug_log_from_routines ()),
            lazy
              "routine logging is active (debug_log_from_routines): [C_syntax.try_register_tile] \
               deterministically declines, so every leaf would time the scalar fallback under a \
               tensorized label" );
        ]
        ~f:(fun (ok, witness) -> if ok then None else Some (Lazy.force witness))
  | _ -> Some "backend kind seeds no tensorized pipeline"

(* The swizzled staged layout, if any, that the backend can read for this site's formats
   (gh-ocannl-481 item 3, D3). [None] leaves the staged seeds untwinned. Operand layout is
   independent of the accumulator lifetime; the geometry that consumes this result applies the
   wide-f16 scope gate separately. *)
let mma_staged_layout_for_precisions (mma : Ir.Backend_intf.mma_capability) ~a_prec ~b_prec ~d_prec
    : LL.swizzle_kind option =
  List.find_map (mma_format_triples ~a_prec ~b_prec ~d_prec) ~f:(fun key ->
      List.Assoc.find mma.Ir.Backend_intf.mma_staged_layouts key
        ~equal:Ir.Backend_intf.equal_mma_format_triple)
  |> Option.map ~f:(function Ir.Backend_intf.Mma_swizzled_b128 -> LL.Swizzle_b128)

type matmul_site = {
  m_i : Idx.symbol;
  m_j : Idx.symbol;
  m_k : Idx.symbol;
      (** The innermost contraction loop — the one a pipeline's k-split divides, whose extent [m_nk]
          the tile's k-extent is judged against. *)
  m_ni : int;
  m_nj : int;
  m_nk : int;
  m_ko : (Idx.symbol * int) list;
      (** Contraction loops enclosing [m_k], in nest order (gh-ocannl-683): a site contracting over
          several axes — attention's out projection [d[b,s,j] += w[j,h,e] * x[b,s,h,e]], whose
          weight carries two input axes — lowers to a reduction nest, of which only the innermost
          loop is [m_k]. They are k-loops lowering has already split: every pipeline treats them as
          k-block loops above the one its own k-split mints ([k_blocks]) — sunk below the output
          roles, staged at, privatized over — so the tiling machinery is shared unchanged with the
          single-axis case, where this is empty and the schedules are byte-identical to before. *)
  m_bo : (Idx.symbol * int) list;
      (** Batch loops enclosing the [m_i] loop, in nest order (gh-ocannl-528): loops beyond the
          [i x j x k] triple that carry their own output axis. They stay [Serial] in the sketch
          pipelines (grid slots are budgeted for the row/column blocks) and join the cross-nest
          alignment chain ([matmul_site_chain]). Empty on plain rank-2 sites. *)
  m_bi : (Idx.symbol * int) list;
      (** Batch loops nested {e between} [m_i] and [m_j] (nest order) — attention's interior head
          axis. The sketch pipelines hoist them above [m_i] with [Swap]s ([batch_hoist_swaps]) so
          the micro-kernel is perfectly nested for [Tensorize] — or, in the GPU pipelines' bgrid-in
          flavor ([sk_batch_inner]), only above the row's in-block remainder ([row_hoists]). Empty
          on plain rank-2 sites. *)
  m_row_axis : int;
      (** The axis of [m_d]'s index map carrying [m_i] (the 2-D tile row). [rank - 2] on plain
          sites; smaller when interior batch axes sit between the roles. [m_j] is always on the
          minor axis. *)
  m_row_stride : int;
      (** [m_a]'s coefficient on [m_i] (gh-ocannl-1148). Greater than [1] only where a strided
          window owns the row: where no row [m_a] reads at unit stride exists (a stride-2 1x1 conv
          at batch 1 reads its input at [(2*oh, 2*ow)] and has no other output axis), or where the
          strided candidate is strictly taller than the unit-stride one (the same conv at batch 2-8,
          gh-ocannl-1397). Such a row can only be tiled through a compacting [Stage] of [m_a]
          ({!Ir.Schedule.stage_compaction_factor}), so the family tree refutes every geometry that
          reads [m_a] in place or packs it at link time. *)
  m_d : Ir.Tnode.t;
  m_a : Ir.Tnode.t;
  m_b : Ir.Tnode.t;
  m_zeroed : bool;  (** A whole-node [Zero_out] of [m_d] is present (needed by [expand_zero]). *)
  m_tb : bool option;
      (** [m_b]'s stored layout: [Some false] = [m_j] on its minor axis ([..., k, ..., j]),
          [Some true] = transposed ([m_k] on the minor axis), [None] = neither cleanly (the
          candidate then fails [Tensorize]'s own role check at compile). Feeds the seeding
          pre-filter (gh-ocannl-479): a rendering that reads B {e in place} inherits this
          orientation — which the register tiling declines when transposed — while a packing [Stage]
          normalizes it. A transposed A never declines (its feeds are scalar splats either way), so
          it is not tracked. *)
  m_fma : bool;
      (** The accumulation is fused in the [Tile_mma] fallback [Tensorize] writes
          ({!Ir.Schedule.fused_accumulation}: an [Ops.FMA], or a floating plain add of the product,
          gh-ocannl-998) — the form the register-tiled rendering requires (its vector twin promises
          bitwise equality only for fused rounding). Candidate schedules rewrite operand reads but
          never the accumulation form, so this is decidable at seeding time. *)
}

let strip_stmts stmts =
  List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true)

(* The perfectly nested serial prefix of a statement: (symbol, extent) per loop, plus the leaf. *)
let rec serial_nest_of (llc : LL.t) : (Idx.symbol * int) list * LL.t =
  match llc with
  | LL.For_loop { index; body; from_ = 0; to_; axis = LL.Serial; _ } -> (
      match strip_stmts (LL.flat_lines [ body ]) with
      | [ single ] ->
          let rest, leaf = serial_nest_of single in
          ((index, to_ + 1) :: rest, leaf)
      | _ -> ([ (index, to_ + 1) ], body))
  | LL.If { body; _ } -> serial_nest_of body
  | _ -> ([], llc)

let rec collect_gets (sc : LL.scalar_t) : (Ir.Tnode.t * Idx.axis_index array) list =
  let arg (s, _prec) = collect_gets s in
  match sc with
  | LL.Get (tn, idcs) -> [ (tn, idcs) ]
  | LL.Ternop (_, a, b, c) -> arg a @ arg b @ arg c
  | LL.Binop (_, a, b) -> arg a @ arg b
  | LL.Unop (_, a) -> arg a
  | LL.Get_dynamic { dyn_value; _ } -> arg dyn_value
  | LL.Local_scope _ | LL.Get_local _ | LL.Get_merge_buffer _ | LL.Constant _ | LL.Constant_bits _
  | LL.Embed_index _ ->
      []

let idx_coeff (idx : Idx.axis_index) sym =
  match idx with
  | Idx.Iterator s when Idx.equal_symbol s sym -> 1
  | Idx.Affine { symbols; _ } ->
      List.sum (module Int) symbols ~f:(fun (c, s) -> if Idx.equal_symbol s sym then c else 0)
  | _ -> 0

(* The unique axis of [idcs] owning [s]: [s] appears in exactly one component, with coefficient 1
   there. Mirrors [Schedule.Tensorize]'s ownership discipline. *)
let unit_axis (idcs : Idx.axis_index array) s : int option =
  let ps =
    Array.filter_mapi idcs ~f:(fun p idx -> Option.some_if (Idx.axis_index_mentions_symbol s idx) p)
  in
  match Array.to_list ps with [ p ] when idx_coeff idcs.(p) s = 1 -> Some p | _ -> None

(* Batched-site classification shared by the relation-based and procedural matchers (gh-ocannl-528).
   Inputs: the perfectly nested serial accumulation statement's loops in nest order (with extents),
   the accumulator's index map [di], and the two operand reads. Roles:

   - The contraction nest is the maximal innermost suffix of loops absent from [di] (a forward
   product lowers its reduction loops after the output loops, so a multi-axis contraction is exactly
   such a suffix; backprop's contractions do not, and reach this matcher through the enabling
   interchange of [detect_matmul_canonical], gh-ocannl-1183): [k] is its innermost loop, the rest
   are [m_ko] (gh-ocannl-683). - Every other loop must own a distinct axis of [di] (unit
   coefficient, sole occurrence). - [j] owns [di]'s minor axis and must be the innermost of the
   write loops (how a forward product lowers them — the sketch pipelines' hoisting normalization
   only handles batch loops above [j]; backprop's other orders arrive through the same interchange,
   gh-ocannl-1325). - Per operand order, [a] must own [k], must not read [j]; [b] must own [j] and
   [k]; [i] is the {e deepest} write loop owned by [a] and absent from [b] — the 2-D tile row; a
   role symbol owns its component alone (a convolution window [ox + kx] is not a tile axis). The
   deepest write loop [a] reads as a single strided term [c*i] is the row's other candidate
   ([m_row_stride = c], gh-ocannl-1148), and the two are chosen between by extent (gh-ocannl-1397):
   the strided one owns the row where no unit-stride candidate exists (a stride-2 1x1 conv at batch
   1 has no other output axis) or where it is strictly taller (the same conv at batch 2-8, whose
   [ow] is taller than its [b]); a tie keeps the unit-stride row, which needs no compacting [Stage].
   Sites with no strided candidate keep their roles and seeds. The exclusions are what keep
   variance-style self-products [d[b,s] += x[b,s,k] * x[b,s,k]] — whose reads mention every loop —
   from masquerading as matmuls: they seeded (and always failed candidate compile) before. -
   Everything else is batch: [m_bo] outside [i], [m_bi] between [i] and [j]; batch symbols and outer
   contraction symbols may appear in the operands freely (their occurrences form the tile block
   base).

   Detection remains permissive about everything else — a mis-detected site fails its candidate
   compile (op preconditions, [validate_parallel], hardware limits) and is skipped. *)
let classify_matmul ~(loops : (Idx.symbol * int) list) ~(d : Ir.Tnode.t)
    ~(di : Idx.axis_index array) ~(o1 : Ir.Tnode.t * Idx.axis_index array)
    ~(o2 : Ir.Tnode.t * Idx.axis_index array) ~(zeroed : bool) ~(fma : bool) : matmul_site option =
  let rank = Array.length di in
  let rev_ks, rev_ws =
    List.split_while (List.rev loops) ~f:(fun (s, _) ->
        not (Array.exists di ~f:(Idx.axis_index_mentions_symbol s)))
  in
  match (rev_ks, rev_ws) with
  | (k, nk) :: rev_ko, (_ :: _ :: _ as rev_ws : (Idx.symbol * int) list) when rank >= 2 -> (
      let ko = List.rev rev_ko in
      let ws = List.rev rev_ws in
      let d_axes = List.map ws ~f:(fun (s, _) -> unit_axis di s) in
      if List.exists d_axes ~f:Option.is_none then None
      else
        let axes = List.zip_exn ws (List.filter_opt d_axes) in
        let distinct =
          let ps = List.map axes ~f:snd in
          List.length (List.dedup_and_sort ps ~compare:Int.compare) = List.length ps
        in
        if not distinct then None
        else
          let (j, nj), pj = List.last_exn axes in
          if pj <> rank - 1 then None
          else
            let front = List.drop_last_exn axes in
            let try_order ((a, ai) : Ir.Tnode.t * Idx.axis_index array)
                ((b, bi) : Ir.Tnode.t * Idx.axis_index array) : matmul_site option =
              (* A tile axis is a plain iterator: a role symbol must be the SOLE symbol of the
                 component it owns. A convolution window [x[..., oy + ky, ox + kx, ic]] mixes an
                 output symbol with a kernel one in a single component; once contraction nests are
                 admitted (gh-ocannl-683) a conv's [(ky, kx, ic)] suffix would otherwise classify as
                 a matmul here — [ic] as [k], the window axes as [i] and a batch loop — and since
                 the matmul family is tried first, the conv family would silently never be seeded
                 for it (schedule_conv_gemm pins the conv seeds). *)
              let plain idx =
                match idx with
                | Idx.Iterator _ | Idx.Affine { symbols = [ _ ]; _ } -> true
                | _ -> false
              in
              let sole_axis idcs s =
                match unit_axis idcs s with Some p when plain idcs.(p) -> Some p | _ -> None
              in
              (* The row's strided fallback (gh-ocannl-1148): [s] is the sole symbol of exactly one
                 component, read there at a coefficient above 1. *)
              let strided_axis idcs s =
                match
                  Array.filter_mapi idcs ~f:(fun p idx ->
                      Option.some_if (Idx.axis_index_mentions_symbol s idx) p)
                with
                | [| p |] when plain idcs.(p) && idx_coeff idcs.(p) s > 1 ->
                    Some (p, idx_coeff idcs.(p) s)
                | _ -> None
              in
              (* The same for the outer contraction loops, wherever an operand mentions one: a
                 conv's kernel-window symbols are exactly the suffix loops that appear mixed into an
                 output axis ([oy + ky]), and with the channel loop innermost the row rule alone
                 would still pick the batch loop as [i]. Strides and offsets stay admissible — these
                 loops are only ever iterated, never tiled. *)
              let ko_plain idcs =
                List.for_all ko ~f:(fun (s, _) ->
                    Array.for_all idcs ~f:(fun idx ->
                        (not (Idx.axis_index_mentions_symbol s idx)) || plain idx))
              in
              if
                Array.exists ai ~f:(Idx.axis_index_mentions_symbol j)
                || Option.is_none (sole_axis ai k)
                || Option.is_none (sole_axis bi j)
                || Option.is_none (sole_axis bi k)
                || (not (ko_plain ai))
                || not (ko_plain bi)
              then None
              else
                let eligible ~stride =
                  List.filter_map front ~f:(fun (((s, _), _) as role) ->
                      if Array.exists bi ~f:(Idx.axis_index_mentions_symbol s) then None
                      else match stride ai s with Some c -> Some (role, c) | None -> None)
                in
                let unit_row =
                  List.last
                    (eligible ~stride:(fun idcs s -> Option.map (sole_axis idcs s) ~f:(fun _ -> 1)))
                and strided_row =
                  List.last
                    (eligible ~stride:(fun idcs s -> Option.map (strided_axis idcs s) ~f:snd))
                in
                (* By extent (gh-ocannl-1397): the strided row wins only where it is strictly
                   taller, so a tie keeps the unit-stride row and its in-place reads. *)
                let row =
                  match (unit_row, strided_row) with
                  | Some (((_, n_unit), _), _), Some (((_, n_strided), _), _)
                    when n_strided > n_unit ->
                      strided_row
                  | Some _, _ -> unit_row
                  | None, _ -> strided_row
                in
                Option.map row ~f:(fun (((i, ni), p_row), row_stride) ->
                    let before_i = ref true in
                    let m_bo = ref [] and m_bi = ref [] in
                    List.iter front ~f:(fun ((s, n), _) ->
                        if Idx.equal_symbol s i then before_i := false
                        else if !before_i then m_bo := (s, n) :: !m_bo
                        else m_bi := (s, n) :: !m_bi);
                    let rank_b = Array.length bi in
                    let m_tb =
                      match (unit_axis bi j, unit_axis bi k) with
                      | Some p, _ when p = rank_b - 1 -> Some false
                      | _, Some p when p = rank_b - 1 -> Some true
                      | _ -> None
                    in
                    {
                      m_i = i;
                      m_j = j;
                      m_k = k;
                      m_ni = ni;
                      m_nj = nj;
                      m_nk = nk;
                      m_ko = ko;
                      m_bo = List.rev !m_bo;
                      m_bi = List.rev !m_bi;
                      m_row_axis = p_row;
                      m_row_stride = row_stride;
                      m_d = d;
                      m_a = a;
                      m_b = b;
                      m_zeroed = zeroed;
                      m_tb;
                      m_fma = fma;
                    })
            in
            match try_order o1 o2 with Some _ as r -> r | None -> try_order o2 o1)
  | _ -> None

let detect_matmul_procedural (llc : LL.t) : matmul_site option =
  let stmts = strip_stmts (LL.flat_lines [ llc ]) in
  let zeroed = List.filter_map stmts ~f:(function LL.Zero_out tn -> Some tn | _ -> None) in
  List.find_map stmts ~f:(fun stmt ->
      match serial_nest_of stmt with
      | (_ :: _ :: _ :: _ as loops), LL.Set { tn = d; idcs = di; llsc; _ } -> (
          let gets = collect_gets llsc in
          let is_d_read (tn, idcs) = phys_equal tn d && Array.equal Idx.equal_axis_index idcs di in
          let d_reads, others = List.partition_tf gets ~f:is_d_read in
          match (d_reads, others) with
          | _ :: _, [ o1; o2 ] ->
              let fma = Option.is_some (Ir.Schedule.fused_accumulation ~d ~di llsc) in
              classify_matmul ~loops ~d ~di ~o1 ~o2
                ~zeroed:(List.exists zeroed ~f:(phys_equal d))
                ~fma
          | _ -> None)
      | _ -> None)

(** {2 Relation-based micro-kernel recognition (gh-494 waypoint-2 remainder)}

    Detection reads off the same extracted artifact the op-legality oracle consumes —
    [LL.affine_accesses]: the rmw markers, index maps, loop boxes and program paths — instead of
    re-walking the code with a procedural structural matcher, so detection and legality share one
    source of access truth. The procedural matchers above are kept for the [legality_crosscheck]
    soak, which raises on any divergence (detection feeds sketch seeding, so changes must be
    behavior-preserving). Known corners where the relations see more than the old walkers (an [If]
    guard whose condition reads a tensor node; an interior statement with no tensor accesses):
    optimized code does not produce them, and the crosscheck guards the claim. *)

module A = Ir.Affine

(* Axis types by loop binder — the one nest discipline the access records do not carry (their
   [a_loops] carry the bounds). Statement-level loops only: an access whose enclosing loops are not
   all found here is inside a [Local_scope] body or [Tile_mma] fallback, which the recognizers
   reject anyway. *)
let rec loop_axis_types acc (llc : LL.t) =
  match llc with
  | LL.Seq (a, b) -> loop_axis_types (loop_axis_types acc a) b
  | LL.For_loop { index; axis; body; _ } -> loop_axis_types ((index, axis) :: acc) body
  | LL.If { body; _ } -> loop_axis_types acc body
  | _ -> acc

let path_head = A.stmt_head

(* Access records per top-level statement, in program order (the extraction fires in program order
   and top-level statement indices are nondecreasing). *)
let accesses_by_statement (accs : Ir.Tnode.t A.access list) =
  List.group accs ~break:(fun a b ->
      not (Option.equal Int.equal (path_head a.A.a_path) (path_head b.A.a_path)))

(* The statement an access path leads to, followed the way [LL.affine_accesses] builds it: a loop
   adds no component, a [Seq] level one [Stmt] per flattened statement, an [If] its [Body]. The head
   [Stmt] is NOT an index into [llc]'s top-level statements when [llc] is a loop whose body is a
   sequence -- the first pass of a split reduction, [for blk { init; for inner { ... } }], is one
   (gh-ocannl-1414). *)
let rec stmt_at (llc : LL.t) (path : A.path_comp list) : LL.t option =
  match (llc, path) with
  | LL.For_loop { body; _ }, _ -> stmt_at body path
  | LL.Seq _, A.Stmt k :: rest ->
      Option.bind (List.nth (LL.flat_lines [ llc ]) k) ~f:(fun s -> stmt_at s rest)
  | LL.If { body; _ }, A.Body :: rest -> stmt_at body rest
  | _, [ (A.Rhs | A.Write) ] -> Some llc
  | _ -> None

(* The accumulation form (fused, or fusable by [Tensorize]: {!Ir.Schedule.fused_accumulation}) is
   scalar structure the access records do not carry; probe the recognized write's own statement
   directly. *)
let fma_form (llc : LL.t) ~stmt_path ~d ~(di : Idx.axis_index array) : bool =
  let is_d_read tn idcs = phys_equal tn d && Array.equal Idx.equal_axis_index idcs di in
  match stmt_at llc stmt_path with
  | Some (LL.Set { tn; idcs; llsc; _ }) ->
      is_d_read tn idcs && Option.is_some (Ir.Schedule.fused_accumulation ~d ~di llsc)
  | _ -> false

(* The perfect all-serial from-0 accumulation statement, as the relations express it: a single
   interpretable write whose enclosing statement's accesses are all the statement's own direct reads
   ([Affine.same_statement]: paths agreeing above the final [Rhs]/[Write] component — a sibling
   statement inside the nest, or a read nested in a [Local_scope] body, breaks the agreement) and
   share its loop box. Returns the write and the non-write accesses split into the write's own
   same-cell reads (the rmw carrier) and the operand reads, in program order. *)
let serial_kernel_of axes (g : Ir.Tnode.t A.access list) =
  let writes = List.filter g ~f:(fun a -> a.A.a_write) in
  match writes with
  | [ w ] when (not w.A.a_whole) && Option.is_none w.A.a_dyn_axis && not w.A.a_vec_last ->
      let serial0 (s, (lo, _)) =
        lo = 0
        &&
        match List.Assoc.find axes s ~equal:Idx.equal_symbol with
        | Some LL.Serial -> true
        | Some _ | None -> false
      in
      let loops_equal =
        List.equal (fun (s1, (l1, h1)) (s2, (l2, h2)) ->
            Idx.equal_symbol s1 s2 && l1 = l2 && h1 = h2)
      in
      if
        List.for_all w.A.a_loops ~f:serial0
        && List.for_all g ~f:(fun a ->
            A.same_statement a.A.a_path w.A.a_path && loops_equal a.A.a_loops w.A.a_loops)
      then
        let same_d a =
          phys_equal a.A.a_tn w.A.a_tn && Array.equal Idx.equal_axis_index a.A.a_map w.A.a_map
        in
        let reads = List.filter g ~f:(fun a -> not a.A.a_write) in
        let d_reads, others = List.partition_tf reads ~f:same_d in
        Some (w, d_reads, others)
      else None
  | _ -> None

let detect_matmul_affine (llc : LL.t) : matmul_site option =
  let accs = LL.affine_accesses llc in
  let axes = loop_axis_types [] llc in
  let zeroed = List.filter accs ~f:(fun a -> a.A.a_write && a.A.a_whole) in
  List.find_map (accesses_by_statement accs) ~f:(fun g ->
      match serial_kernel_of axes g with
      | Some (w, (_ :: _ as _d_reads), [ o1; o2 ]) ->
          let loops = List.map w.A.a_loops ~f:(fun (s, (_, hi)) -> (s, hi + 1)) in
          classify_matmul ~loops ~d:w.A.a_tn ~di:w.A.a_map ~o1:(o1.A.a_tn, o1.A.a_map)
            ~o2:(o2.A.a_tn, o2.A.a_map)
            ~zeroed:(List.exists zeroed ~f:(fun z -> phys_equal z.A.a_tn w.A.a_tn))
            ~fma:(fma_form llc ~stmt_path:w.A.a_path ~d:w.A.a_tn ~di:w.A.a_map)
      | _ -> None)

let matmul_site_equal (x : matmul_site) (y : matmul_site) =
  let batch_equal = List.equal (fun (s1, n1) (s2, n2) -> Idx.equal_symbol s1 s2 && n1 = n2) in
  Idx.equal_symbol x.m_i y.m_i && Idx.equal_symbol x.m_j y.m_j && Idx.equal_symbol x.m_k y.m_k
  && x.m_ni = y.m_ni && x.m_nj = y.m_nj && x.m_nk = y.m_nk && batch_equal x.m_ko y.m_ko
  && batch_equal x.m_bo y.m_bo && batch_equal x.m_bi y.m_bi && x.m_row_axis = y.m_row_axis
  && x.m_row_stride = y.m_row_stride && phys_equal x.m_d y.m_d && phys_equal x.m_a y.m_a
  && phys_equal x.m_b y.m_b && Bool.equal x.m_zeroed y.m_zeroed
  && Option.equal Bool.equal x.m_tb y.m_tb
  && Bool.equal x.m_fma y.m_fma

let detect_matmul (llc : LL.t) : matmul_site option =
  let site = detect_matmul_affine llc in
  (if Lazy.force A.crosscheck_enabled then
     let procedural = detect_matmul_procedural llc in
     match (procedural, site) with
     | None, None -> ()
     | Some p, Some n when matmul_site_equal p n -> ()
     | _ ->
         (* invariant: the two matchers are one relation spelled twice; a divergence is a detection
            bug, never a property of the code being detected. *)
         invalid_arg
           "Autotune.detect_matmul crosscheck: the relation-based and procedural matchers diverge \
            — detection must be behavior-preserving");
  site

let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner })

(* gh-ocannl-485 (PADTO): pad [axis] to the next multiple of [f] when [f] does not divide its
   [extent]. Identity pads are omitted, so divisible sites keep byte-identical schedules (and
   schedule-cache keys). Only sound in pipelines that stage every operand the padded axis reaches
   ([Tensorize] enforces the zero-fringe requirement at apply). *)
let pad_to ~axis ~extent f =
  if f > 0 && extent % f <> 0 then [ Sched.Pad { axis; to_multiple_of = f } ] else []

(* gh-ocannl-485 (PADTO) / gh-ocannl-730: may a pipeline PAD a non-multiple extent to its block
   size, rather than refute the geometry on divisibility?

   Yes exactly when every operand the padded axes reach is read through a zero-fringe staged tile.
   [Sched.Stage]'s per-axis edge guards store 0 into the out-of-range slots of an edge tile, so a
   padded iteration reads exact zeros: a scalar add-reduction gains nothing from it, and the
   tensorized pipelines discharge the reduction mask against the very same tiles while [Tensorize]
   moves the row/column masks onto the fragment transfers. An operand read in place cannot absorb a
   pad — its fringe is whatever the buffer holds — so those geometries keep the full divisibility
   gates.

   What the pad leaves behind in a SCALAR pipeline is an [If] on the accumulation leaf, which
   [Sched.Privatize] classifies rather than rejects: a reduction-axis mask fires within one thread's
   own accumulation, and a row/column mask is literally the target's own index compared against its
   dimension — the same predicate the private tile's transfers already carry.

   [n_staged] of the site's [n_operands] are staged at this geometry. Both GPU matmul pipelines
   stage both operands exactly when their k-block is staged ([sk_bk > 0]): the tensorized family's
   unstaged whole-K form reads the operands in place, and so would a blocktile geometry with no
   k-split. The CPU packed-tensorized pipeline stages per operand, so its grid-outermost shape
   qualifies only when every operand packs. Both the seeding gates and the pipelines' [pad_to]
   triples consult this, so a gate and its pads cannot drift apart; the conv pipelines
   (gh-ocannl-697) stage through the same [Stage] decomposition and judge composition here too. *)
let pad_composition_ok ~n_staged ~n_operands = n_operands > 0 && n_staged = n_operands

(* Blocks of size [b] covering a possibly padded extent [n]. *)
let blocks_of n b = (n + b - 1) / b

(* gh-ocannl-740: [pad_composition_ok] answers whether a geometry MAY pad; this answers whether it
   SHOULD. A padded geometry does [blocks_of n b * b] iterations of an axis whose real extent is
   [n], so its work exceeds the site's by the product of those ratios over the axes it pads — the
   pad-waste ratio. [axes] lists [(name, extent, block)] for each axis the geometry blocks; a block
   of 0 is an unsplit axis (no pad), and a dividing block contributes exactly 1, so a geometry that
   pads nothing has ratio 1. *)
let pad_waste_ratio axes =
  List.fold axes ~init:1. ~f:(fun acc (_, extent, block) ->
      if block <= 0 || extent <= 0 then acc
      else acc *. Float.of_int (blocks_of extent block * block) /. Float.of_int extent)

(* Read per call, like the family's other seeding gates, and through the registered value kind
   ([Utils.Float_at_least]), which owns the clamp: a value below 1 means 1, and a malformed or NaN
   one is refused (gh-ocannl-1332). So the search trajectory identity, which renders the key through
   the same canonicalization, gives every spelling of one behaviour one fingerprint. *)
let pad_waste_max () = Utils.registered_float_at_least ~arg_name:"autotune_pad_waste_max"

(* Every pad-waste witness starts with this, which is how {!lift_pad_waste} recognizes the policy:
   the exclusion sits at a geometry decision, where the CPU packed menu's footprint economy excludes
   too, so the decision alone cannot tell the two apart. *)
let pad_waste_witness_prefix = "pad waste:"

(* The exclusion witness of a geometry blocking [axes] in a menu whose least-padded proposed
   geometry has ratio [menu_min], or [None] when it is within the threshold of that. Only the padded
   axes are named: the witness says where the waste is. *)
let pad_waste_witness ~menu_min axes =
  let ratio = pad_waste_ratio axes in
  let threshold = pad_waste_max () in
  if Float.(ratio <= menu_min) || Float.(ratio < threshold * menu_min) then None
  else
    let padded =
      List.filter_map axes ~f:(fun (name, extent, block) ->
          if block <= 0 || extent <= 0 || extent % block = 0 then None
          else Some (Printf.sprintf "%s=%d to %d" name extent (blocks_of extent block * block)))
    in
    Some
      (Printf.sprintf
         "%s padding %s does %.2fx the site's work, %.2fx the %.2fx of this menu's least-padded \
          geometry, at or above the %gx search-economy threshold (config autotune_pad_waste_max)"
         pad_waste_witness_prefix (String.concat ~sep:", " padded) ratio (ratio /. menu_min)
         menu_min threshold)

(* The pad-waste judgment over one geometry menu (gh-ocannl-740): [menu] pairs each geometry's
   decision with the axes it blocks and its child as the menu's legality gates and policies already
   judged it.

   WHY RELATIVE. Whether a pad is worth a compile depends on what else the menu offers the site, not
   on the waste alone. On a 6-row site every row block pads at least 16/6, and the 16-row tensorized
   geometry at 2.67x was the best sketch the search timed (Metal, M4 Max), so an absolute threshold
   low enough to catch a 64-row block on a 32-row site (2x, one block per batch where a dividing
   32-row sibling gives the same block count at half the work; timed 4.9x and 34x slower than the
   best) would delete it. So the waste is judged against the menu's least-padded PROPOSED geometry —
   the sibling that replaces an excluded one: a geometry doing at least [pad_waste_max ()] times
   that sibling's work is [Excluded] with its ratio as witness. It is search ECONOMY, not legality
   (a padded candidate is exact, gh-ocannl-730): a driver may lift it, the threshold is config
   [autotune_pad_waste_max] ([inf] proposes every pad), and the least-padded geometry is never
   excluded, so no pipeline loses its menu to this judgment.

   Legality beats policy: a [Refuted] child stays refuted and never sets the minimum, nor does an
   [Excluded] one (it is not proposed, so it replaces nothing); the exclusion's payload is the child
   as judged, so lifting the policy re-proposes exactly what the threshold suppressed. *)
let judge_pad_waste (menu : ('d * (string * int * int) list * ('l, 'a) Sspace.child) list) :
    ('d * ('l, 'a) Sspace.child) list =
  let proposed =
    List.filter_map menu ~f:(fun (_, axes, child) ->
        match child with
        | Sspace.Child _ | Sspace.Unknown _ -> Some (pad_waste_ratio axes)
        | Sspace.Excluded _ | Sspace.Refuted _ -> None)
  in
  match List.min_elt proposed ~compare:Float.compare with
  | None -> List.map menu ~f:(fun (d, _, child) -> (d, child))
  | Some menu_min ->
      List.map menu ~f:(fun (d, axes, child) ->
          ( d,
            match child with
            | Sspace.Child _ | Sspace.Unknown _ -> (
                match pad_waste_witness ~menu_min axes with
                | Some w -> Sspace.Excluded (w, lazy child)
                | None -> child)
            | Sspace.Excluded _ | Sspace.Refuted _ -> child ))

let mma_scope_of_reduction_extents extents =
  if List.exists extents ~f:(fun extent -> extent > 1) then Ir.Backend_intf.Mma_fragment_scope
  else Ir.Backend_intf.Mma_per_statement

let matmul_mma_scope (site : matmul_site) ~bk =
  mma_scope_of_reduction_extents
    (List.map site.m_ko ~f:snd @ if bk > 0 then [ blocks_of site.m_nk bk ] else [])

(** {2 Convolution detection and the implicit-GEMM sketch (gh-ocannl-493)}

    A convolution is a matmul over a virtual im2col operand. Conv einsums lower to affine-indexed
    accumulation nests —
    [d[b.., oh.., oc] += a[b.., s*oh + t*kh + off.., ic] * w[perm(oc, kh.., ic)]] — so the
    implicit-GEMM mapping is a re-association of loops that already exist: reorder to
    [outer..; kernel..; row; oc; ic], pack the strided-window [row × ic] slice of [a] (the packing
    [Stage] {e is} im2col — same copy nest, conv index arithmetic) and the [ic × oc] slice of [w]
    (normalizing any stored layout) at the kernel-window anchor, then [Tensorize (row, oc, ic)]
    exactly as for matmuls: the register tiling / tensor cores and the accumulator contraction
    (gh-ocannl-480; resident across the whole kernel-window chain since gh-ocannl-501) apply
    unchanged.

    Unlike the matmul pipelines, the reorder moves the [ic] reduction inside the kernel loops, so
    the per-element reduction order changes: conv sketch candidates match the unscheduled form
    within float-reassociation tolerance (like the GPU fragment paths), while the tensorized
    pipeline stays bitwise against the reorder-only form on the C backends.

    Recognition boundary (gh-ocannl-912): the matcher is rank-generic — every output axis other than
    the row and the out-channel is an outer loop, however many batch and spatial axes there are —
    but it requires plain iterators on the output and the kernel, and lowering indexes an extent-1
    axis at [Fixed_idx 0] with its loop gone. So singleton output axes (batch 1, output extent 1)
    and singleton kernel axes (one input channel, a k-by-1 window) are refused, as is a multi-axis
    channel row (one reduction channel); a 1x1 window lowers to a plain GEMM that [detect_matmul]
    owns. Pinned, with the reasons, by [test/operations/conv_detection_boundary]. *)

type conv_axis = {
  cx_o : Idx.symbol;  (** Output spatial symbol (appears in [d] as a plain iterator). *)
  cx_no : int;
  cx_k : Idx.symbol;  (** Kernel-window symbol (appears in [w], not in [d]). *)
  cx_nk : int;
  cx_stride : int;
  cx_dilation : int;
  cx_offset : int;
      (** Padding offset on the input access. Healthy graphs lower padded convs offset-free: the
          source is physically padded and buffer indices absorb the halo shift, while halo-lost
          operands (layout committed before the padded consumer) are rejected at shape-inference
          time. A nonzero offset can still reach detection from hand-built [Low_level] code. *)
}

type conv_site = {
  c_loops : Idx.symbol list;  (** The accumulation nest's loops, outermost first. *)
  c_outer : (Idx.symbol * int) list;
      (** Loops kept outer, in nest order: batch axes and the non-row output spatial axes. *)
  c_kernel : Idx.symbol list;  (** Kernel-window symbols in nest order (the [k_o] tier). *)
  c_axes : conv_axis list;
  c_row : Idx.symbol;  (** The GEMM row: the conv axis at [d]'s rank-2 position. *)
  c_nrow : int;
  c_oc : Idx.symbol;  (** The GEMM column: [d]'s rank-1 symbol, read by [w] only. *)
  c_noc : int;
  c_red : Idx.symbol;  (** The GEMM reduction: the channel symbol read by both operands. *)
  c_nred : int;
  c_d : Ir.Tnode.t;
  c_d_syms : Idx.symbol list;
      (** [d]'s write index in axis order: plain distinct iterators, a permutation of [c_outer]'s
          symbols, [c_row] (at rank-2) and [c_oc] (last). The axis order of [d]'s expanded zero nest
          ({!Sched.expand_zero}) maps onto the site's chain through it (gh-ocannl-503). *)
  c_a : Ir.Tnode.t;
  c_b : Ir.Tnode.t;
  c_zeroed : bool;
  c_fma : bool;
}

let conv_mma_scope (site : conv_site) =
  mma_scope_of_reduction_extents (List.map site.c_axes ~f:(fun cx -> cx.cx_nk))

let detect_conv_procedural (llc : LL.t) : conv_site option =
  let stmts = strip_stmts (LL.flat_lines [ llc ]) in
  let zeroed = List.filter_map stmts ~f:(function LL.Zero_out tn -> Some tn | _ -> None) in
  List.find_map stmts ~f:(fun stmt ->
      let loops, leaf = serial_nest_of stmt in
      match leaf with
      | LL.Set { tn = d; idcs = di; llsc; _ } when List.length loops >= 4 -> (
          let extent s = List.Assoc.find loops s ~equal:Idx.equal_symbol in
          let gets = collect_gets llsc in
          let is_d_read (tn, idcs) = phys_equal tn d && Array.equal Idx.equal_axis_index idcs di in
          let d_reads, others = List.partition_tf gets ~f:is_d_read in
          match (d_reads, others) with
          | _ :: _, [ o1; o2 ] -> (
              (* [d] written at plain, distinct iterators — its symbols are the GEMM output
                 space. *)
              let d_syms =
                Array.to_list di
                |> List.map ~f:(function Idx.Iterator s -> Some s | _ -> None)
                |> Option.all
              in
              match d_syms with
              | Some d_syms
                when List.length d_syms = Array.length di
                     && (not (List.contains_dup d_syms ~compare:Idx.compare_symbol))
                     && List.length d_syms >= 2 -> (
                  let is_out s = List.mem d_syms s ~equal:Idx.equal_symbol in
                  (* The input operand carries the conv fingerprint: an affine component mixing an
                     output symbol with a kernel symbol. *)
                  let conv_component (idx : Idx.axis_index) =
                    match idx with
                    | Idx.Affine { symbols = [ (c1, s1); (c2, s2) ]; offset } -> (
                        match (is_out s1, is_out s2) with
                        | true, false ->
                            Some
                              {
                                cx_o = s1;
                                cx_no = 0;
                                cx_k = s2;
                                cx_nk = 0;
                                cx_stride = c1;
                                cx_dilation = c2;
                                cx_offset = offset;
                              }
                        | false, true ->
                            Some
                              {
                                cx_o = s2;
                                cx_no = 0;
                                cx_k = s1;
                                cx_stride = c2;
                                cx_dilation = c1;
                                cx_offset = offset;
                                cx_nk = 0;
                              }
                        | _ -> None)
                    | _ -> None
                  in
                  let classify (tn, idcs) =
                    let axes = Array.to_list idcs |> List.filter_map ~f:conv_component in
                    (tn, idcs, axes)
                  in
                  let (a, a_idcs, a_axes), (b, b_idcs, b_axes) =
                    let c1 = classify o1 and c2 = classify o2 in
                    match (c1, c2) with
                    | (_, _, _ :: _), (_, _, []) -> (c1, c2)
                    | (_, _, []), (_, _, _ :: _) -> (c2, c1)
                    | _ -> (c1, c1)
                    (* Both-or-neither convolutional: rejected below (b_axes <> []). *)
                  in
                  let b_plain =
                    Array.to_list b_idcs
                    |> List.map ~f:(function Idx.Iterator s -> Some s | _ -> None)
                    |> Option.all
                  in
                  match b_plain with
                  | Some b_syms when List.is_empty b_axes && not (phys_equal a b) -> (
                      let kernel_syms = List.map a_axes ~f:(fun cx -> cx.cx_k) in
                      let in_b s = List.mem b_syms s ~equal:Idx.equal_symbol in
                      let oc_candidates = List.filter d_syms ~f:in_b in
                      (* Reduction symbols: read by both operands, not output, not kernel. *)
                      let a_plain_syms =
                        Array.to_list a_idcs
                        |> List.filter_map ~f:(function Idx.Iterator s -> Some s | _ -> None)
                      in
                      let red_candidates =
                        List.filter a_plain_syms ~f:(fun s -> in_b s && not (is_out s))
                      in
                      let rank = Array.length di in
                      match (oc_candidates, red_candidates, extent (List.last_exn d_syms)) with
                      | [ oc ], [ red ], Some noc
                        when Idx.equal_symbol oc (List.last_exn d_syms)
                             && (not (List.exists a_plain_syms ~f:(Idx.equal_symbol oc)))
                             && List.for_all kernel_syms ~f:(fun k -> in_b k && not (is_out k))
                             && List.for_all b_syms ~f:(fun s ->
                                 Idx.equal_symbol s oc || Idx.equal_symbol s red
                                 || List.mem kernel_syms s ~equal:Idx.equal_symbol) -> (
                          (* The GEMM row: the conv axis sitting at [d]'s rank-2 position. *)
                          let row_sym =
                            match di.(rank - 2) with Idx.Iterator s -> s | _ -> assert false
                          in
                          match
                            ( List.find a_axes ~f:(fun cx -> Idx.equal_symbol cx.cx_o row_sym),
                              extent row_sym,
                              extent red )
                          with
                          | Some _, Some nrow, Some nred ->
                              let with_extents cx =
                                match (extent cx.cx_o, extent cx.cx_k) with
                                | Some no, Some nk -> Some { cx with cx_no = no; cx_nk = nk }
                                | _ -> None
                              in
                              let axes = Option.all (List.map a_axes ~f:with_extents) in
                              let loop_syms = List.map loops ~f:fst in
                              let m_fma =
                                Option.is_some (Ir.Schedule.fused_accumulation ~d ~di llsc)
                              in
                              let is_kernel s = List.mem kernel_syms s ~equal:Idx.equal_symbol in
                              let outer =
                                List.filter loops ~f:(fun (s, _) ->
                                    is_out s
                                    && (not (Idx.equal_symbol s row_sym))
                                    && not (Idx.equal_symbol s oc))
                              in
                              let kernel_order = List.filter loop_syms ~f:is_kernel in
                              Option.map axes ~f:(fun axes ->
                                  {
                                    c_loops = loop_syms;
                                    c_outer = outer;
                                    c_kernel = kernel_order;
                                    c_axes = axes;
                                    c_row = row_sym;
                                    c_nrow = nrow;
                                    c_oc = oc;
                                    c_noc = noc;
                                    c_red = red;
                                    c_nred = nred;
                                    c_d = d;
                                    c_d_syms = d_syms;
                                    c_a = a;
                                    c_b = b;
                                    c_zeroed = List.exists zeroed ~f:(phys_equal d);
                                    c_fma = m_fma;
                                  })
                          | _ -> None)
                      | _ -> None)
                  | _ -> None)
              | _ -> None)
          | _ -> None)
      | _ -> None)

let detect_conv_affine (llc : LL.t) : conv_site option =
  let accs = LL.affine_accesses llc in
  let axes = loop_axis_types [] llc in
  let zeroed = List.filter accs ~f:(fun a -> a.A.a_write && a.A.a_whole) in
  List.find_map (accesses_by_statement accs) ~f:(fun g ->
      match serial_kernel_of axes g with
      | Some (w, _ :: _, [ ro1; ro2 ]) when List.length w.A.a_loops >= 4 -> (
          let loops = List.map w.A.a_loops ~f:(fun (s, (_, hi)) -> (s, hi + 1)) in
          let extent s = List.Assoc.find loops s ~equal:Idx.equal_symbol in
          let d = w.A.a_tn and di = w.A.a_map in
          let o1 = (ro1.A.a_tn, ro1.A.a_map) and o2 = (ro2.A.a_tn, ro2.A.a_map) in
          (* From here on, the classification is the same role logic as the procedural matcher, fed
             from the extracted maps. *)
          let d_syms =
            Array.to_list di
            |> List.map ~f:(function Idx.Iterator s -> Some s | _ -> None)
            |> Option.all
          in
          match d_syms with
          | Some d_syms
            when List.length d_syms = Array.length di
                 && (not (List.contains_dup d_syms ~compare:Idx.compare_symbol))
                 && List.length d_syms >= 2 -> (
              let is_out s = List.mem d_syms s ~equal:Idx.equal_symbol in
              let conv_component (idx : Idx.axis_index) =
                match idx with
                | Idx.Affine { symbols = [ (c1, s1); (c2, s2) ]; offset } -> (
                    match (is_out s1, is_out s2) with
                    | true, false ->
                        Some
                          {
                            cx_o = s1;
                            cx_no = 0;
                            cx_k = s2;
                            cx_nk = 0;
                            cx_stride = c1;
                            cx_dilation = c2;
                            cx_offset = offset;
                          }
                    | false, true ->
                        Some
                          {
                            cx_o = s2;
                            cx_no = 0;
                            cx_k = s1;
                            cx_stride = c2;
                            cx_dilation = c1;
                            cx_offset = offset;
                            cx_nk = 0;
                          }
                    | _ -> None)
                | _ -> None
              in
              let classify (tn, idcs) =
                let axes = Array.to_list idcs |> List.filter_map ~f:conv_component in
                (tn, idcs, axes)
              in
              let (a, a_idcs, a_axes), (b, b_idcs, b_axes) =
                let c1 = classify o1 and c2 = classify o2 in
                match (c1, c2) with
                | (_, _, _ :: _), (_, _, []) -> (c1, c2)
                | (_, _, []), (_, _, _ :: _) -> (c2, c1)
                | _ -> (c1, c1)
                (* Both-or-neither convolutional: rejected below (b_axes <> []). *)
              in
              let b_plain =
                Array.to_list b_idcs
                |> List.map ~f:(function Idx.Iterator s -> Some s | _ -> None)
                |> Option.all
              in
              match b_plain with
              | Some b_syms when List.is_empty b_axes && not (phys_equal a b) -> (
                  let kernel_syms = List.map a_axes ~f:(fun cx -> cx.cx_k) in
                  let in_b s = List.mem b_syms s ~equal:Idx.equal_symbol in
                  let oc_candidates = List.filter d_syms ~f:in_b in
                  let a_plain_syms =
                    Array.to_list a_idcs
                    |> List.filter_map ~f:(function Idx.Iterator s -> Some s | _ -> None)
                  in
                  let red_candidates =
                    List.filter a_plain_syms ~f:(fun s -> in_b s && not (is_out s))
                  in
                  let rank = Array.length di in
                  match (oc_candidates, red_candidates, extent (List.last_exn d_syms)) with
                  | [ oc ], [ red ], Some noc
                    when Idx.equal_symbol oc (List.last_exn d_syms)
                         && (not (List.exists a_plain_syms ~f:(Idx.equal_symbol oc)))
                         && List.for_all kernel_syms ~f:(fun k -> in_b k && not (is_out k))
                         && List.for_all b_syms ~f:(fun s ->
                             Idx.equal_symbol s oc || Idx.equal_symbol s red
                             || List.mem kernel_syms s ~equal:Idx.equal_symbol) -> (
                      let row_sym =
                        match di.(rank - 2) with Idx.Iterator s -> s | _ -> assert false
                      in
                      match
                        ( List.find a_axes ~f:(fun cx -> Idx.equal_symbol cx.cx_o row_sym),
                          extent row_sym,
                          extent red )
                      with
                      | Some _, Some nrow, Some nred ->
                          let with_extents cx =
                            match (extent cx.cx_o, extent cx.cx_k) with
                            | Some no, Some nk -> Some { cx with cx_no = no; cx_nk = nk }
                            | _ -> None
                          in
                          let caxes = Option.all (List.map a_axes ~f:with_extents) in
                          let loop_syms = List.map loops ~f:fst in
                          let is_kernel s = List.mem kernel_syms s ~equal:Idx.equal_symbol in
                          let outer =
                            List.filter loops ~f:(fun (s, _) ->
                                is_out s
                                && (not (Idx.equal_symbol s row_sym))
                                && not (Idx.equal_symbol s oc))
                          in
                          let kernel_order = List.filter loop_syms ~f:is_kernel in
                          Option.map caxes ~f:(fun caxes ->
                              {
                                c_loops = loop_syms;
                                c_outer = outer;
                                c_kernel = kernel_order;
                                c_axes = caxes;
                                c_row = row_sym;
                                c_nrow = nrow;
                                c_oc = oc;
                                c_noc = noc;
                                c_red = red;
                                c_nred = nred;
                                c_d = d;
                                c_d_syms = d_syms;
                                c_a = a;
                                c_b = b;
                                c_zeroed = List.exists zeroed ~f:(fun z -> phys_equal z.A.a_tn d);
                                c_fma = fma_form llc ~stmt_path:w.A.a_path ~d ~di;
                              })
                      | _ -> None)
                  | _ -> None)
              | _ -> None)
          | _ -> None)
      | _ -> None)

let conv_axis_equal (x : conv_axis) (y : conv_axis) =
  Idx.equal_symbol x.cx_o y.cx_o && x.cx_no = y.cx_no && Idx.equal_symbol x.cx_k y.cx_k
  && x.cx_nk = y.cx_nk && x.cx_stride = y.cx_stride && x.cx_dilation = y.cx_dilation
  && x.cx_offset = y.cx_offset

let conv_site_equal (x : conv_site) (y : conv_site) =
  List.equal Idx.equal_symbol x.c_loops y.c_loops
  && List.equal (fun (s1, n1) (s2, n2) -> Idx.equal_symbol s1 s2 && n1 = n2) x.c_outer y.c_outer
  && List.equal Idx.equal_symbol x.c_kernel y.c_kernel
  && List.equal conv_axis_equal x.c_axes y.c_axes
  && Idx.equal_symbol x.c_row y.c_row && x.c_nrow = y.c_nrow && Idx.equal_symbol x.c_oc y.c_oc
  && x.c_noc = y.c_noc && Idx.equal_symbol x.c_red y.c_red && x.c_nred = y.c_nred
  && phys_equal x.c_d y.c_d
  && List.equal Idx.equal_symbol x.c_d_syms y.c_d_syms
  && phys_equal x.c_a y.c_a && phys_equal x.c_b y.c_b && Bool.equal x.c_zeroed y.c_zeroed
  && Bool.equal x.c_fma y.c_fma

let detect_conv (llc : LL.t) : conv_site option =
  let site = detect_conv_affine llc in
  (if Lazy.force A.crosscheck_enabled then
     let procedural = detect_conv_procedural llc in
     match (procedural, site) with
     | None, None -> ()
     | Some p, Some n when conv_site_equal p n -> ()
     | _ ->
         (* invariant: the two matchers are one relation spelled twice; a divergence is a detection
            bug, never a property of the code being detected. *)
         invalid_arg
           "Autotune.detect_conv crosscheck: the relation-based and procedural matchers diverge — \
            detection must be behavior-preserving");
  site

(** {2 The enabling interchange (gh-ocannl-1183)}

    [classify_matmul] reads the contraction nest off the innermost end of the loop nest, which is
    how the forward pass lowers a product. Backprop does not: a weight gradient lowers as
    [for b, s, o, i: dW[o,i] += dy[b,s,o] * x[b,s,i]] (the contraction loops outermost) and a data
    gradient as [for b, s, o, i: dx[b,s,i] += dy[b,s,o] * w[o,i]] (the contraction loop between
    write loops), so not one backward contraction of a training step was ever a matmul site, and no
    tiled, register-tiled or tensorized sketch was ever seeded for the kernels a large batch is
    dominated by. The answer is the split-reduce precedent's (gh-ocannl-537): an adjacent-[Swap]
    chain that sinks the contraction loops below the write loops, each [Swap] confirmed [Op_legal]
    on the code it is applied to, after which the statement is re-detected on the interchanged code.
    Where the write loop owning the accumulator's minor axis is not the innermost write loop either
    (the tied lm_head's [wte.grad], attention's [v.grad], gh-ocannl-1325), the chain goes on to sink
    that loop below the other write loops. The chain is the prefix of every schedule built from such
    a site, so replaying it from the original code reproduces the code the site was detected on —
    and since [Swap] mints no symbol, the site's symbols name the same loops in both.

    Interchanging only reorders whole cells' sequences: for any one accumulator cell the contraction
    loops still run in their original relative order, so the interchanged code computes the same
    values bitwise. Sites the plain matcher finds, and segments the conv family claims, are left
    exactly as before (no prefix, the same seeds); the interchange only adds sites where neither
    family had one. *)

(* The adjacent-interchange chain taking a statement's loop nest [loops] (outermost first) to its
   write loops followed by its contraction loops, each group in its original relative order: the
   first contraction loop directly enclosing a write loop is swapped below it until none is. *)
let contraction_sink_swaps ~(is_write : Idx.symbol -> bool) (loops : Idx.symbol list) :
    (Idx.symbol * Idx.symbol) list =
  let rec step order swaps =
    let rec find = function
      | x :: (y :: _ as rest) -> if (not (is_write x)) && is_write y then Some (x, y) else find rest
      | _ -> None
    in
    match find order with
    | None -> List.rev swaps
    | Some (x, y) ->
        let order =
          List.concat_map order ~f:(fun s ->
              if Idx.equal_symbol s x then [ y ] else if Idx.equal_symbol s y then [ x ] else [ s ])
        in
        step order ((x, y) :: swaps)
  in
  step loops []

(* The second half of the chain (gh-ocannl-1325): [classify_matmul] also wants the write loop owning
   the accumulator's minor axis innermost among the write loops, and backprop does not always lower
   it there. The tied lm_head's [wte.grad[e,v] += logits.grad[b,s,v] * x[b,s,e]] runs its write
   loops [v, e] (the forward product's order) and attention's [v.grad[b,t,h,d] += softmax[b,s,h,t] *
   y.grad[b,s,h,d]] runs [b, h, d, t], so after the contraction sink neither is a site. Given the
   order the sink leaves ([write loops @ contraction loops]), the adjacent swaps moving that owner
   [minor] below the write loops after it, which keep their relative order. Nothing when [minor] is
   already the innermost write loop, so every chain that was a site before is unchanged. Like the
   sink, it reorders whole cells' sequences only. *)
let minor_owner_sink_swaps ~(is_write : Idx.symbol -> bool) ~(minor : Idx.symbol)
    (sunk : Idx.symbol list) : (Idx.symbol * Idx.symbol) list =
  match
    List.drop_while (List.filter sunk ~f:is_write) ~f:(fun s -> not (Idx.equal_symbol s minor))
  with
  | _ :: below -> List.map below ~f:(fun inner -> (minor, inner))
  | [] -> []

(* The interchange enabling the first statement of [opt] that is an accumulation of a product of two
   reads but whose contraction loops are not its innermost suffix, or whose minor-axis write loop is
   not its innermost write loop: the chain, and the code it produces, once every [Swap] is
   [Op_legal] and the interchanged code is a matmul site. *)
let matmul_interchange (opt : LL.optimized) : (Sched.schedule * LL.optimized * matmul_site) option =
  let candidate stmt =
    match serial_nest_of stmt with
    | (_ :: _ :: _ :: _ as loops), LL.Set { tn = d; idcs = di; llsc; _ } -> (
        let gets = collect_gets llsc in
        let is_d_read (tn, idcs) = phys_equal tn d && Array.equal Idx.equal_axis_index idcs di in
        match List.partition_tf gets ~f:is_d_read with
        | _ :: _, [ _; _ ] ->
            let is_write s = Array.exists di ~f:(Idx.axis_index_mentions_symbol s) in
            let order = List.map loops ~f:fst in
            let sink = contraction_sink_swaps ~is_write order in
            let sunk = List.filter order ~f:is_write @ List.filter order ~f:(Fn.non is_write) in
            let minor =
              match Array.to_list di |> List.last with
              | Some (Idx.Iterator s) when Option.is_some (unit_axis di s) -> Some s
              | _ -> None
            in
            let swaps =
              sink
              @ Option.value_map minor ~default:[] ~f:(fun minor ->
                  minor_owner_sink_swaps ~is_write ~minor sunk)
            in
            if List.is_empty swaps then None
            else Some (List.map swaps ~f:(fun (outer, inner) -> Sched.Swap { outer; inner }))
        | _ -> None)
    | _ -> None
  in
  let legal ops =
    let verdicts = Sched.schedule_legality opt ops in
    List.length verdicts = List.length ops
    && List.for_all verdicts ~f:(fun (_, v) -> Sched.equal_op_verdict v Sched.Op_legal)
  in
  let hermetic (o : LL.optimized) =
    {
      o with
      LL.traced_store = Hashtbl.copy o.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
    }
  in
  List.find_map
    (strip_stmts (LL.flat_lines [ opt.LL.llc ]))
    ~f:(fun stmt ->
      match candidate stmt with
      | Some ops when legal ops -> (
          match Sched.apply_classified ops (hermetic opt) with
          | swapped ->
              Option.map (detect_matmul swapped.LL.llc) ~f:(fun site -> (ops, swapped, site))
          | exception Outcome.Cause_at _ -> None)
      | _ -> None)

(* The matmul site of [opt] as every seeding and schedule-construction entry point sees it: the
   site, the enabling-interchange prefix its schedules carry, and the code it was detected on (the
   interchanged code when the prefix is nonempty). A conv site takes precedence over an interchanged
   one, so the conv family keeps every segment it claimed before. *)
let detect_matmul_canonical (opt : LL.optimized) :
    (matmul_site * Sched.schedule * LL.optimized) option =
  match detect_matmul opt.LL.llc with
  | Some site -> Some (site, [], opt)
  | None ->
      if Option.is_some (detect_conv opt.LL.llc) then None
      else
        Option.map (matmul_interchange opt) ~f:(fun (prefix, swapped, site) ->
            (site, prefix, swapped))

(** {2 The reduction sink (gh-ocannl-1348)}

    Backprop lowers a reduction over the batch into a small output — a bias or layer-norm parameter
    gradient — as [for b, s, o: db[o] += dy[b,s,o]], its reduction loops outermost. The default GPU
    preset maps [o] to hardware below the serial [b] and [s] loops, so every thread reads and writes
    its device-memory cell once per reduction step: code generation localizes an accumulator only
    over a nest of reduction loops BELOW the cell's own loops. The same adjacent-[Swap] chain as the
    enabling interchange above ({!contraction_sink_swaps}) sinks the reduction loops below the write
    loops; the preset then maps the write loops and the localizer keeps the accumulator in a
    register for the whole reduction.

    Values are unchanged bitwise where the accumulator's index map is injective in the write loops
    (each a plain [Iterator] owning its own axis): one cell is then one tuple of write-loop values,
    and its contributions arrive in the reduction loops' original relative order, which the chain
    keeps. A statement whose cell a reduction loop's [Affine] or [Concat] index also moves (a
    convolution's input gradient, an embedding's scattered rows) is left alone, as is one whose
    storage precision is narrower than its accumulator's: localized, it would round once where the
    default rounds after every step. *)

let reduction_sink ~(accum_prec : Ir.Ops.prec -> Ir.Ops.prec) (opt : LL.optimized) : Sched.schedule
    =
  let candidate stmt =
    match serial_nest_of stmt with
    | (_ :: _ :: _ as loops), LL.Set { tn = d; idcs = di; llsc; _ } ->
        let order = List.map loops ~f:fst in
        let is_write s = Array.exists di ~f:(Idx.axis_index_mentions_symbol s) in
        let plain s = Array.count di ~f:(Idx.equal_axis_index (Idx.Iterator s)) = 1 in
        let self_read (tn, idcs) = phys_equal tn d && Array.equal Idx.equal_axis_index idcs di in
        let d_prec = Lazy.force d.Ir.Tnode.storage_prec in
        if
          List.exists (collect_gets llsc) ~f:self_read
          && List.for_all order ~f:(fun s -> (not (is_write s)) || plain s)
          && Ir.Ops.equal_prec (accum_prec d_prec) d_prec
        then contraction_sink_swaps ~is_write order
        else []
    | _ -> []
  in
  (* Statement by statement, each chain confirmed [Op_legal] after the chains kept before it, so a
     statement the oracle cannot clear costs only its own interchange. *)
  let legal ops =
    let verdicts = Sched.schedule_legality opt ops in
    List.length verdicts = List.length ops
    && List.for_all verdicts ~f:(fun (_, v) -> Sched.equal_op_verdict v Sched.Op_legal)
  in
  List.fold
    (strip_stmts (LL.flat_lines [ opt.LL.llc ]))
    ~init:[]
    ~f:(fun kept stmt ->
      match candidate stmt with
      | [] -> kept
      | chain ->
          let ops = kept @ List.map chain ~f:(fun (outer, inner) -> Sched.Swap { outer; inner }) in
          if legal ops then ops else kept)

(* The statically-decidable precondition of {!zero_geometry}, shared with the family tree's
   construction-time verdicts (gh-ocannl-577): a zeroed site whose output lacks a row axis before
   the minor axis fails every pipeline's zero expansion, whatever the tile geometry. *)
let zero_rank_ok (site : matmul_site) =
  let rank = Array.length (Lazy.force site.m_d.Ir.Tnode.dims) in
  rank >= 2 && site.m_row_axis < rank - 1

let zero_expansion_witness (site : matmul_site) : string option =
  if site.m_zeroed && not (zero_rank_ok site) then
    Some "zero expansion needs a row axis before the minor axis (autotune_sketch_output_rank)"
  else None

(* {3 Where a GPU matmul pipeline binds its batch loops (gh-ocannl-643, gh-ocannl-728)}

   [`Serial]: the batch loops run inside each block. [`Grid]: every batch loop is a whole-loop
   [Grid] axis above the row blocks — interior ones ([m_bi]) hoisted above [m_i] — so the batch
   product folds onto [.z]. [`Grid_inner]: interior batch loops stay BELOW the row blocks, hoisted
   only above the row's in-block remainder, so they bind the grid slot beside the column blocks and
   the row blocks fold with the outer batch onto [.z] ([sk_batch_inner]). The site nest, the zeroing
   nest and every companion nest apply the same layout to the same chain positions, which is what
   keeps the positional thread identity across nests. *)
type batch_layout = [ `Serial | `Grid | `Grid_inner ]

let batch_layout_of (p : sketch_params) : batch_layout =
  if not p.sk_batch_grid then `Serial else if p.sk_batch_inner then `Grid_inner else `Grid

let batch_grid_of : batch_layout -> bool = function `Serial -> false | `Grid | `Grid_inner -> true

let batch_inner_of : batch_layout -> bool = function
  | `Grid_inner -> true
  | `Serial | `Grid -> false

(* A pipeline's row-role geometry in two halves, so a [`Grid_inner] layout can hoist the interior
   batch loops between them: the block split whose outer loop is the row's [Grid] axis, the in-block
   remainder loop it leaves ([rp_inner]), and the ops refining that remainder (the blocktile's
   register split; nothing for the tensorized pipeline). A pipeline's [row_parts ?extent] pads the
   row loop to the block first when given the loop's [extent] and the block does not divide it: a
   non-dividing [Split] wraps the remainder loop's body in its guard, which a [Swap] cannot pass,
   while [Pad] guards the leaves and keeps the nest perfect — the site nest's own row loop is padded
   by the pipeline at such geometries anyway ([pad_composition_ok]), so the zeroing and companion
   nests keep its block count and slot extents. *)
type row_parts = { rp_block : Sched.schedule; rp_inner : Idx.symbol; rp_rest : Sched.schedule }

(* Hoist [syms] (perfectly nested below [outer], in nest order) above [outer] with sequential
   adjacent [Swap]s: after each, [outer] is directly above the next one. *)
let hoist_above ~(outer : Idx.symbol) (syms : Idx.symbol list) : Sched.schedule =
  List.map syms ~f:(fun inner -> Sched.Swap { outer; inner })

(* Whether a sketch's fused-epilogue twin places its target in workgroup-shared storage: the
   [shared] knob [family_schedule] passes to [Sched.Fuse_epilogue]. *)
let epilogue_shared (p : sketch_params) = p.sk_gpu && p.sk_mma

(* The covering per-cell zero nest of [target] that a fused-epilogue twin must give the site's
   geometry itself (gh-ocannl-1453). [Sched.Fuse_epilogue] with [shared] — the GPU tensorized
   twins', [epilogue_shared] — places a target not already on device in workgroup-shared storage,
   one copy per workgroup that all its threads write. [companion_geometry] covers only nests writing
   a MATERIALIZED node (routine-local scratch is otherwise per-thread), so where the target is not
   materialized — an intermediate whose only consumer is the fused tail — the per-cell zero a GPU
   fission segment folds in beside the site (gh-ocannl-1175) stayed bare: every thread of the
   workgroup stored every cell, which gh-ocannl-959's binding legality refuses at render time. The
   whole-routine twins never hit this, because the whole-node [Zero_out] they expand always takes
   the site's geometry ([zero_geometry], [conv_zero_geometry]); this gives the folded zero the same.

   [Some (nest_order, axis_order)]: the zero nest's loop symbols in nest order and in [target]'s
   axis order. [None] when there is nothing to annotate: no shared placement, a materialized target
   (whose zero [companion_geometry] covers), or no folded zero of [target] in the lowering. A folded
   zero that is not a perfect nest storing [target] at its own loop symbols, one per axis, has no
   positional mapping onto the site's chain, and declines the candidate. *)
let shared_folded_zero ~(p : sketch_params) ~(target : Ir.Tnode.t) (opt : LL.optimized) :
    (Idx.symbol list * Idx.symbol list) option =
  if
    (not (p.sk_epilogue && epilogue_shared p))
    || Ir.Tnode.Placements.is_materialized_peek opt.LL.optimize_ctx.LL.placements target
  then None
  else
    let real llc =
      List.filter (LL.flat_lines [ llc ]) ~f:(function
        | LL.Noop | LL.Comment _ -> false
        | _ -> true)
    in
    match
      List.filter (real opt.LL.llc) ~f:(function
        | LL.Zero_out _ -> false
        | stmt -> Option.exists (LL.zero_initializer_target stmt) ~f:(Ir.Tnode.equal target))
    with
    | [] -> None
    | nest :: rest -> (
        let rec perfect loops (llc : LL.t) =
          match (llc, real llc) with
          | LL.For_loop { index; from_ = 0; body; _ }, _ -> perfect (index :: loops) body
          | _, [ LL.Set { tn; idcs; _ } ] when Ir.Tnode.equal tn target -> (
              let loops = List.rev loops in
              let axis_order =
                Array.to_list idcs
                |> List.map ~f:(function Idx.Iterator s -> Some s | _ -> None)
                |> Option.all
              in
              match axis_order with
              | Some axis_order
                when List.is_empty rest
                     && List.length axis_order = List.length loops
                     && List.for_all loops ~f:(List.mem axis_order ~equal:Idx.equal_symbol) ->
                  Some (loops, axis_order)
              | _ -> None)
          | _ -> None
        in
        match perfect [] nest with
        | Some _ as found -> found
        | None ->
            refuse
              (Printf.sprintf
                 "Autotune sketch: the fused twin places %s in workgroup-shared storage, but its \
                  folded zero is not a single perfect nest storing it at its own loop symbols, so \
                  the site's geometry cannot be mapped onto it (gh-ocannl-1453)"
                 (Ir.Tnode.debug_name target)))

(* Zero-geometry ops shared by the sketch pipelines: expand the whole-node [Zero_out] of the output
   and give the resulting nest a compatible parallel geometry, via [mk_zops] on its two fresh loop
   symbols and — under a [`Grid_inner] layout only — the interior batch zero loops [mk_zops] must
   hoist under the row's block split ([~interior]; empty otherwise). When the site is NOT zeroed — a
   fission exposes a covering zero as an ordinary per-cell companion, and other whole-node zeros
   land in their own [`Zeros] segment — there is nothing to expand here. Companion geometry maps the
   per-cell zero with the site; [Privatize] can forward it into the accumulator tile using the
   shared zero-seed proof. Otherwise the tile/fragment loads the initialized target.

   [folded] (gh-ocannl-1453) is that per-cell zero where the caller, not companion geometry, owns
   its annotation ({!shared_folded_zero}): it takes exactly the ops the expanded whole-node zero
   would, minus the expansion. Its loops must nest in [m_d]'s axis order, as [Expand_zero] nests
   them. *)
let zero_geometry ?(layout : batch_layout = `Serial) ?folded (site : matmul_site)
    ~(mk_zops : zi:Idx.symbol -> zj:Idx.symbol -> interior:Idx.symbol list -> Sched.schedule) :
    Sched.schedule =
  if (not site.m_zeroed) && Option.is_none folded then []
  else (
    if not (zero_rank_ok site) then
      (* This is a known limitation of the generated sketch, not an arbitrary exception from a user
         transform. Preserve that distinction at the narrow site so strict candidate failure
         classification records a decline and continues trying the remaining seeds. *)
      raise
        (Outcome.Cause_at
           ( Outcome.Transform,
             Outcome.Unsupported
               {
                 feature = "autotune_sketch_output_rank";
                 detail = "Autotune sketch: zero expansion needs a row axis before the minor axis";
               } ));
    let ez, zsyms =
      match folded with
      | Some (nest_order, axis_order) when not site.m_zeroed ->
          if not (List.equal Idx.equal_symbol nest_order axis_order) then
            refuse
              "Autotune sketch: the folded zero's loops do not nest in the output's axis order \
               (gh-ocannl-1453)";
          ([], axis_order)
      | _ ->
          let ez, zsyms = Sched.expand_zero ~tn:site.m_d in
          ([ ez ], zsyms)
    in
    (* Batched outputs (gh-ocannl-528): the row/column zero loops get the accumulation's geometry;
       the batch-axis zero loops stay [Serial], like the batch loops of the accumulation nest —
       except under the [sk_batch_grid] twins (gh-ocannl-643), where they mirror the accumulation
       nest's batch geometry: interior-batch zero loops (node axes between the row axis and the
       minor axis) hoist above the row loop with the same sequential adjacent [Swap]s as
       [batch_hoist_swaps], and every batch zero loop retypes to [Grid] — the zero nest's loop order
       and per-position geometry then match the accumulation nest's by construction, which is what
       keeps a hardware thread zeroing exactly the cells it accumulates. The row loop precedes the
       column loop in the zero nest ([m_row_axis < rank - 1]), matching the accumulation's
       positional hardware-slot order. Under [`Grid_inner] (gh-ocannl-728) the interior-batch zero
       loops are instead handed to [mk_zops], which hoists them only above the row's in-block
       remainder, exactly where the accumulation nest's interior batch loops sit. *)
    let zi = List.nth_exn zsyms site.m_row_axis and zj = List.last_exn zsyms in
    let rank = List.length zsyms in
    let interior = List.filteri zsyms ~f:(fun ax _ -> ax > site.m_row_axis && ax < rank - 1) in
    let batch_ops =
      match layout with
      | `Serial -> []
      | (`Grid | `Grid_inner) as layout ->
          List.concat_mapi zsyms ~f:(fun ax zs ->
              if ax = site.m_row_axis || ax = rank - 1 then []
              else
                (match layout with
                  | `Grid when ax > site.m_row_axis -> [ Sched.Swap { outer = zi; inner = zs } ]
                  | _ -> [])
                @ [ Sched.Retype { axis = zs; ty = LL.Grid } ])
    in
    let interior = match layout with `Grid_inner -> interior | `Serial | `Grid -> [] in
    ez @ batch_ops @ mk_zops ~zi ~zj ~interior)

(* The [For_loop] symbols of a statement, at statement positions ({!LL.write_demand}): loops beneath
   a [Scan_loop] included, the scan's own binder not -- it is no schedulable loop. *)
let nest_loop_syms (llc : LL.t) =
  List.filter_map (LL.write_demand llc).binders ~f:(fun b ->
      Option.map b.LL.Access_fold.axis ~f:(fun _ -> b.LL.Access_fold.index))

(* The would-be epilogue tail's loop symbols: the first real statement after the last statement
   writing [target] — the nest [Sched.Fuse_epilogue] consumes (its perfect-Serial-nest and
   sole-consumer vetting happens in the op itself). Used by the fused twins to leave that nest
   unannotated (fuse-before-annotate, gh-ocannl-501): [sketch_schedule] appends the fusion op last,
   by which point an annotated tail nest — [Fuse_epilogue] requires a perfect Serial tail — would be
   rejected. The relocated tail write lands under the accumulation nest's own geometry instead, so
   coverage is preserved without the dropped annotation. *)
let epilogue_tail_loop_syms ~(target : Ir.Tnode.t) (opt : LL.optimized) : Idx.symbol list =
  let stmts =
    List.filter (LL.flat_lines [ opt.LL.llc ]) ~f:(function
      | LL.Noop | LL.Comment _ -> false
      | _ -> true)
  in
  let writes_target s = List.mem (LL.write_demand s).written target ~equal:Ir.Tnode.equal in
  match List.filter_mapi stmts ~f:(fun i s -> Option.some_if (writes_target s) i) |> List.last with
  | None -> []
  | Some r -> ( match List.nth stmts (r + 1) with Some tail -> nest_loop_syms tail | None -> [])

(* Companion geometry for the GPU matmul sketches (gh-ocannl-521).

   A GPU sketch builds hardware geometry for the accumulation nest alone. Launch dimensions are
   global to the kernel, so every OTHER materialized-writing nest in the same routine — the
   bias/relu tail of a classifier head, the elementwise companions an aligned-merged fission segment
   carries — must be nested under loops covering the same active slots, or
   [Low_level.validate_parallel] rejects it and the whole candidate fails to compile. The GPU seeds
   used to leave those nests bare and depend on [Sched.Fuse_epilogue] absorbing the companion into
   the accumulation nest; when the fusion declines (a guarded reduction output, a whole-K [Tile_mma]
   accumulator), the seed had no surviving form at all — the cascade that left every GPU backend
   with tensorized candidates seeded in bulk and none ever timed.

   The precedent is [conv_aligned_grid], which reuses the default CPU preset's aligned cross-nest
   analysis rather than re-proving alignment. The same analysis extends to workgroup geometry: it is
   {!Sched.aligned_chains} that decides WHICH loops may be annotated and that chain position [k]
   means the same thread coordinate in every linked nest; only the geometry per position is the
   sketch's own, which is what a preset cannot supply for a tensorized nest (two [Grid] slots plus a
   [Workgroup] lane). Emitting a positionally identical geometry on each companion nest therefore
   preserves the alignment the analysis proved, and covers the same slots the site's nest binds.

   [site_syms] is the accumulation nest's chain, [annotate chain] the ops for one companion nest's
   whole chain (the chain is passed entire, in nest order with extents, so the caller can also emit
   the positional loop-reorder [Swap]s that mirror the site pipeline's interior-batch hoisting —
   gh-ocannl-643; positional identity across nests is preserved because the permutation and the
   per-position geometry are functions of the shared chain-position roles alone), [skip] the loop
   symbols of a nest to leave alone (the fused twins' epilogue tail, which the fusion relocates
   under the accumulation nest's geometry — annotating it would make [Fuse_epilogue] reject the
   candidate for the wrong reason), and [expanded_zeros] the nodes whose whole-node [Zero_out] the
   caller expands with the same geometry.

   [None] when the analysis bails, when the site's own chain was trimmed below [site_syms] (the
   nests could not be aligned at this arity — a companion annotated anyway would read cells another
   thread wrote, with no intra-kernel synchronization to order them), or when a companion's chain
   does not match the site's in arity and extents. A [None] must fail the candidate rather than fall
   back to a bare companion: on GPU there is no all-serial fallback.

   The query runs at the site's own arity ([max_chain = length site_syms], gh-ocannl-569): the
   analysis' default cap of 2 is the two-loop presets' Grid+Workgroup shape, and under it a batched
   (rank-3+) site could never match its full chain — every seed for gpt2's FFN-class kernels
   declined here, serializing the minor output axis. A companion that genuinely cannot follow the
   full arity (a reduction over the site's minor axis, e.g. the lm_head's max-logits row) still
   trims the component's common prefix below [site_syms] and correctly declines.

   Ordering, shared with the zeroing geometry this reuses: a tensorized nest's workgroup slot is the
   [Tensorize] lane, whose per-lane element ownership is architecture-opaque, so the accumulator
   fragment loads cells other lanes wrote in a companion (the folded zero of gh-ocannl-1175), and a
   later companion reads cells the fragment store wrote. The GPU renderers order both: every
   accumulator load of [d] from memory (the per-call intrinsic, the resident-fragment scope opening,
   the lane-0 fallback) opens with a workgroup barrier, and every store-back closes with one --
   [threadgroup_barrier(mem_threadgroup | mem_device)] on Metal, [__syncthreads()] on CUDA and HIP.
   The barrier spans the whole workgroup, so the ordering holds for any lane width the renderer
   emits; [qkv_zero_init] executes every seed of a folded projection against a materialized run. *)
(* The materialized nodes a statement writes, by debug name: what a companion witness names. *)
let materialized_writes ~plc (stmt : LL.t) : string =
  (LL.write_demand stmt).written
  |> List.filter ~f:(Ir.Tnode.Placements.is_materialized_peek plc)
  |> List.map ~f:Ir.Tnode.debug_name
  |> List.dedup_and_sort ~compare:String.compare
  |> String.concat ~sep:","

(* The companion nests a parallel site geometry must cover: the top-level statements writing a
   MATERIALIZED node — [validate_parallel]'s rule is about shared memory, and routine-local scratch
   is per-thread by construction — other than the site's own nest (it mentions the site's loops),
   the nests [skip] names (a fused flavor's epilogue tail) and the [expanded_zeros] the pipeline
   expands itself. Settled at construction: the placements are read through the pure
   [is_materialized_peek], the forcing query's own answer. The walk enters [Scan_loop] bodies, as
   [validate_parallel]'s does (gh-ocannl-1134): a scan's row writes (the online softmax's max and
   normalizer) are a companion's writes like any other. *)
let companion_demand ~(site_syms : (Idx.symbol * int) list) ~(skip : Idx.symbol list)
    ~(expanded_zeros : Ir.Tnode.t list) (opt : LL.optimized) : LL.t list =
  let plc = opt.LL.optimize_ctx.LL.placements in
  let mentions syms stmt =
    List.exists (nest_loop_syms stmt) ~f:(fun s -> List.mem syms s ~equal:Idx.equal_symbol)
  in
  let site_sym_list = List.map site_syms ~f:fst in
  List.filter (LL.flat_lines [ opt.LL.llc ]) ~f:(fun stmt ->
      match stmt with
      | LL.Noop | LL.Comment _ -> false
      | LL.Zero_out tn when List.exists expanded_zeros ~f:(Ir.Tnode.equal tn) -> false
      | _ ->
          (not (String.is_empty (materialized_writes ~plc stmt)))
          && (not (mentions site_sym_list stmt))
          && not (mentions skip stmt))

let companion_geometry ~(site_syms : (Idx.symbol * int) list) ~(skip : Idx.symbol list)
    ~(expanded_zeros : Ir.Tnode.t list) ~(annotate : (Idx.symbol * int) list -> Sched.schedule)
    (opt : LL.optimized) : (Sched.schedule, string) Result.t =
  let plc = opt.LL.optimize_ctx.LL.placements in
  (* Restricting the demand to materialized writes also keeps the query out of the way of the
     pipelines it never needed to constrain: a site with nothing to cover neither consults the
     analysis nor can be failed by it. An aligned scan nest takes the site's geometry on its
     enclosing loops and runs serially per thread; a scan over the site's own column axis trims
     below the site's arity and is refuted. *)
  let needs = companion_demand ~site_syms ~skip ~expanded_zeros opt in
  if List.is_empty needs then Ok []
  else
    let shape cs = String.concat ~sep:"x" (List.map cs ~f:(fun (_, e) -> Int.to_string e)) in
    let same_shape cs =
      List.length cs = List.length site_syms
      && List.for_all2_exn cs site_syms ~f:(fun (_, e) (_, e') -> e = e')
    in
    let written stmt = materialized_writes ~plc stmt in
    match Sched.aligned_chains ~max_chain:(List.length site_syms) ~expanded_zeros opt with
    | None ->
        Error
          (Printf.sprintf
             "the cross-nest race analysis bails on this routine, so the %d companion nest(s) (%s) \
              cannot be given aligned geometry"
             (List.length needs)
             (String.concat ~sep:"; " (List.map needs ~f:written)))
    | Some chains ->
        (* The site's own nest must keep the analysis' full chain: a trimmed one means the nests
           could not be aligned at this arity, and a companion annotated anyway would read cells
           another thread wrote, with no intra-kernel synchronization to order them. *)
        let own_ok =
          List.exists chains ~f:(fun (_, cs) ->
              List.equal (fun (a, _) (b, _) -> Idx.equal_symbol a b) cs site_syms && same_shape cs)
        in
        (* Loop symbols are unique per loop construct, so a nest is identified by its chain's
           outermost symbol occurring among the statement's loops. *)
        let chain_of stmt =
          let syms = nest_loop_syms stmt in
          List.find_map chains ~f:(fun (_, cs) ->
              match cs with
              | (s, _) :: _ when List.mem syms s ~equal:Idx.equal_symbol -> Some cs
              | _ -> None)
        in
        if not own_ok then
          Error
            (Printf.sprintf
               "the accumulation nest's aligned chain was trimmed below its %s geometry, so its \
                companions cannot share it"
               (shape site_syms))
        else
          List.fold_until needs ~init:[]
            ~f:(fun acc stmt ->
              match chain_of stmt with
              | Some cs when same_shape cs -> Continue (acc @ annotate cs)
              | Some cs ->
                  Stop
                    (Error
                       (Printf.sprintf
                          "companion nest writing %s has aligned chain %s, the accumulation nest %s"
                          (written stmt) (shape cs) (shape site_syms)))
              | None ->
                  Stop
                    (Error
                       (Printf.sprintf "companion nest writing %s has no aligned parallel chain"
                          (written stmt))))
            ~finish:(fun acc -> Ok acc)

(* A companion nest that cannot take the accumulation nest's aligned geometry is a limitation of the
   generated sketch, not an arbitrary exception from a user transform — the same distinction
   [zero_geometry] draws for non-rank-2 outputs. Raise it at the narrow site as a typed
   [Unsupported] cause so strict candidate failure classification records a decline and keeps trying
   the remaining seeds; a plain [invalid_arg] here aborts the whole search under the default
   [strict_failure_classification=true]. *)
let companion_coverage_unsupported ~tensorized why =
  raise
    (Outcome.Cause_at
       ( Outcome.Transform,
         Outcome.Unsupported
           {
             feature = "autotune_sketch_companion_coverage";
             detail =
               Printf.sprintf "Autotune sketch: %sGPU matmul companion coverage (gh-521): %s"
                 (if tensorized then "tensorized " else "")
                 why;
           } ))

(* The chain the GPU matmul sketches annotate, as [companion_geometry] wants it: the accumulation
   nest's own outer loops in nest order — batch loops included (gh-ocannl-528) — which is exactly
   what {!Sched.aligned_chains} reports for that nest when the site is parallelizable at full
   arity. *)
let matmul_site_chain (site : matmul_site) =
  site.m_bo @ ((site.m_i, site.m_ni) :: site.m_bi) @ [ (site.m_j, site.m_nj) ]

(* The statically-decidable component of the GPU sketches' companion-coverage rule, decided once at
   tree-construction time (gh-ocannl-577). [companion_geometry]'s Ok/Error verdict never depends on
   the geometry [annotate] emits — only the lowering, the site's chain, the fused flavor's [skip]
   and the zeroing expansion select it — so one query with a trivial annotator settles buildability
   for every tile completion of the flavor: curated menus, twins and the whole tile lattice alike.
   [fused] selects the [Fuse_epilogue] twins' flavor, which skips the epilogue tail nest — coverage
   can pass fused where it fails unfused (the tail was the failing companion, or its exclusion
   empties the demand before the alignment analysis is consulted), the pre-gh-521 "only the fused
   twin survives" regime, so the two flavors are judged separately — though not independently: the
   fused demand is a subset of the unfused one and the alignment analysis itself does not depend on
   [skip], so an unfused-Ok verdict implies the fused one ([matmul_family_tree] shares it, paying
   the analysis twice only where the unfused flavor is refuted). The raise sites in the schedule
   builders stay as the safety net for parameters replayed against a different lowering (fission
   recombination). *)
let matmul_coverage_witness ~(opt : LL.optimized) ~(fused : bool) (site : matmul_site) :
    string option =
  match
    companion_geometry ~site_syms:(matmul_site_chain site)
      ~skip:(if fused then epilogue_tail_loop_syms ~target:site.m_d opt else [])
      ~expanded_zeros:(if site.m_zeroed then [ site.m_d ] else [])
      ~annotate:(fun _ -> [])
      opt
  with
  | Ok _ -> None
  | Error why -> Some (Printf.sprintf "GPU matmul companion coverage (gh-521): %s" why)

(* The CPU analog of the rule above, for the CPU tensorized flavors that bind a hardware dimension
   (gh-ocannl-1561): the whole-triple form's [Workgroup] lane — the zeroing's column loop and the
   [Tensorize] lane at the full column extent [m_nj], so active at every seeded extent — plus its
   [Grid] row split under [sk_bm > 0], and the packed Grid shapes' row-block loop (at least two
   blocks, so active). The builders annotate only the accumulation nest and the output zero they
   expand; they give companions no geometry, so where a nest writing a materialized node remains
   beside them, [validate_parallel] rejects the candidate at every compile (every hardware index
   would execute the companion). The packed Serial and Hoisted shapes bind nothing (all-serial
   loops, a unit lane) and stay unaffected. The demand is the GPU rule's, with the fused flavor's
   epilogue tail skipped (the fusion relocates it into the accumulation nest) and a whole-node zero
   of the output excluded; its emptiness, not an alignment, decides, since no CPU builder gives a
   companion the site's geometry. Before this, a whole-routine search over a matmul with any
   materialized companion -- footprint_diagonal's product read through its diagonal, in the
   materialize-all arm -- offered [W_sketch[mma-cpu 0x0x0]] (the whole-triple form: [0] is the
   unsplit row, not a size) and paid a failed compile for it. The [F_sketch] granularity, where
   fission leaves the matmul alone in its segment, keeps the form. [None] when nothing would be left
   uncovered, else the companions' materialized writes. *)
let cpu_mma_companion_writes ~(opt : LL.optimized) ~(fused : bool) (site : matmul_site) :
    string option =
  match
    companion_demand ~site_syms:(matmul_site_chain site)
      ~skip:(if fused then epilogue_tail_loop_syms ~target:site.m_d opt else [])
      ~expanded_zeros:(if site.m_zeroed then [ site.m_d ] else [])
      opt
  with
  | [] -> None
  | needs ->
      let plc = opt.LL.optimize_ctx.LL.placements in
      (* Node names deduplicated: a companion reduction's zero and its accumulation write one
         node. *)
      Some
        (Printf.sprintf "%d companion nest(s) writing materialized %s" (List.length needs)
           (String.concat ~sep:"; "
              (List.dedup_and_sort ~compare:String.compare
                 (List.map needs ~f:(materialized_writes ~plc)))))

(* The refutation of a CPU tensorized flavor whose [what] binds a hardware dimension, given
   {!cpu_mma_companion_writes}' verdict. *)
let cpu_mma_companion_witness ~what companions =
  Option.map companions ~f:(fun c ->
      Printf.sprintf
        "CPU matmul companion coverage (gh-ocannl-1561): %s binds a hardware dimension, and %s \
         outside it, which validate_parallel rejects"
        what c)

(* Chain-position roles matching [matmul_site_chain]: row/column positions get the pipeline's
   geometry; batch positions stay [Serial] by default (serial loops above hardware loops are legal:
   hardware loops bind, not iterate) and become whole-loop [Grid] axes under the [sk_batch_grid]
   twins (gh-ocannl-643) — the row/column blocks keep grid slots 0 and 1, and the batch grid axes
   land on slots [>= 2], which fold onto the hardware [.z] dimension (see [Low_level]'s
   hardware-axis section comment). *)
let matmul_chain_roles (site : matmul_site) : [ `Batch | `Row | `Col ] list =
  List.map site.m_bo ~f:(fun _ -> `Batch)
  @ (`Row :: List.map site.m_bi ~f:(fun _ -> `Batch))
  @ [ `Col ]

(* One companion nest's schedule under the shared chain-position roles: the interior batch loops
   (the [`Batch] positions after the [`Row] one) are hoisted above the nest's own row loop with the
   same sequential adjacent [Swap]s as [batch_hoist_swaps] applies to the site nest, then each chain
   position gets its role's annotation. Emitted per companion because the swaps name the companion's
   own symbols; positional thread identity across nests is preserved because the permutation and the
   per-position geometry are functions of the role list alone (gh-ocannl-643). With batch positions
   unannotated the hoists are omitted: they would be dead reordering. Under [`Grid_inner]
   (gh-ocannl-728) the interior batch loops hoist only above the row's in-block remainder, between
   the two halves of the row's geometry ([row_parts]), as the site nest's do. *)
let companion_role_ops ~(roles : [ `Batch | `Row | `Col ] array)
    ~(annotate_role : [ `Batch | `Row | `Col ] -> Idx.symbol -> Sched.schedule)
    ~(row_parts : ?extent:int -> Idx.symbol -> row_parts) ~(layout : batch_layout)
    (cs : (Idx.symbol * int) list) : Sched.schedule =
  let row_pos =
    Option.map ~f:fst (Array.findi roles ~f:(fun _ r -> match r with `Row -> true | _ -> false))
  in
  let interior =
    List.filter_mapi cs ~f:(fun pos (s, _) ->
        match (roles.(pos), row_pos) with `Batch, Some r when pos > r -> Some s | _ -> None)
  in
  let annotations () =
    List.concat (List.mapi cs ~f:(fun pos (s, _) -> annotate_role roles.(pos) s))
  in
  match (layout, row_pos) with
  | `Serial, _ | _, None -> annotations ()
  | `Grid, Some r -> hoist_above ~outer:(fst (List.nth_exn cs r)) interior @ annotations ()
  | `Grid_inner, Some _ ->
      List.concat
        (List.mapi cs ~f:(fun pos (s, extent) ->
             match roles.(pos) with
             | `Row ->
                 let rp = row_parts ?extent:(Some extent) s in
                 rp.rp_block @ hoist_above ~outer:rp.rp_inner interior @ rp.rp_rest
             | role -> annotate_role role s))

(* The batch loops of a site, in [matmul_site_chain] order (outer batch loops, then the interior
   ones the pipelines hoist above the row loop — the final nest order). *)
let matmul_batch_loops (site : matmul_site) : (Idx.symbol * int) list = site.m_bo @ site.m_bi

(* {3 The launch geometry a seed will have (gh-ocannl-709)}

   A GPU candidate's grid and workgroup extents are decided by the parameters, so the seeder can
   predict them and consult the SAME cap reading the pre-driver gate uses
   ([Schedule.launch_geometry_excess], whose rows are the device's per-dimension limits). What the
   seeder adds is only the prediction; before gh-ocannl-709 it also carried its own copy of one cap
   (the [.z] fold's), which is how the other four dimensions could only be learned one wasted
   compile at a time. *)

(* CUDA/HIP cap [gridDim.y] and [gridDim.z] at 65535 ([gridDim.x] alone is 2^31-scale; Metal is
   larger still, and advertises no cap). The authoritative per-backend cap is
   [hardware_limits.max_grid_yz]; this constant is the conservative fallback where a backend
   advertises none, so seeding stays deterministic across machines instead of proposing candidates
   that only the 16-bit backends would refuse. It saturates the record ([seeding_limits]) rather
   than a single dimension: the two grid dimensions share the one field, so a fallback applied to
   one and not the other would be the very asymmetry this section removes. *)
let max_grid_fold_extent = 65535

(* The limits a SEED is judged against: the backend's own, with an unadvertised grid cap saturated
   to the conservative fallback. The gate deliberately does not do this — there, an unadvertised cap
   is genuinely no cap, and a hand-built schedule on Metal may fold as wide as it likes. *)
let seeding_limits (limits : Ir.Backend_intf.hardware_limits) : Ir.Backend_intf.hardware_limits =
  {
    limits with
    Ir.Backend_intf.max_grid_yz =
      Some (Option.value limits.Ir.Backend_intf.max_grid_yz ~default:max_grid_fold_extent);
  }

(* The launch geometry of a nest whose hardware-annotated loops have these extents in NEST order
   (outermost first): the slot rule's one owner, [Sched.launch_geometry_of_nests] (shared with the
   default annotators' lane plans, gh-ocannl-1133), for a single nest. A fold whose product
   overflows predicts the largest [.z] extent there is, which every capped device refuses. *)
let predicted_launch_geometry ~(grid : int list) ~(block : int list) : Sched.launch_geometry =
  match Sched.launch_geometry_of_nests [ (grid, block) ] with
  | Some geometry -> geometry
  | None ->
      {
        (Option.value_exn (Sched.launch_geometry_of_nests [ ([], block) ])) with
        Sched.lg_grid_y = Some (Option.value (List.nth (List.rev grid) 1) ~default:1);
        lg_grid_z = Some Int.max_value;
      }

(* Why a device refuses this predicted geometry, phrased as the seed's refutation witness: the same
   sentence [Schedule.check_hardware_limits_classified] would put in its decline, so a refutation
   log and a decline log read alike. *)
let launch_geometry_refutation ~(limits : Ir.Backend_intf.hardware_limits)
    (geom : Sched.launch_geometry) : string option =
  Option.map
    (Sched.launch_geometry_excess ~limits:(seeding_limits limits) geom)
    ~f:(fun x -> "the candidate " ^ x.Sched.lx_phrase)

(* Whether the [sk_batch_grid] twins are worth a decision level for this site: there are batch loops
   to spread, and their product is more than one block. A STRUCTURAL question only — whether the
   product fits the device's [.z] dimension is the launch predicate's business, and asking it here
   too is what gh-ocannl-709 found: the one dimension of five that seeding pre-filtered, in its own
   hand-written encoding of a cap the gate already held. The level now appears on every batched
   site, and an over-cap fold is refuted leaf by leaf with the gate's own sentence — the same
   treatment as the other four dimensions, and a reason where there used to be an absence. *)
let batch_grid_twin_ok (site : matmul_site) : bool =
  let product = List.fold (matmul_batch_loops site) ~init:1 ~f:(fun acc (_, n) -> acc * n) in
  (not (List.is_empty (matmul_batch_loops site))) && product >= 2

(* The launch geometry a GPU matmul sketch will have, from the parameters alone. Grid loops in nest
   order: the batch loops the [sk_batch_grid] twins retype (outermost), then the row-block loop,
   then the column-block loop — so the row blocks bind [.y] and the batch product folds onto [.z];
   under [sk_batch_inner] the interior batch loops sit between the row and column blocks instead,
   binding [.y] while the row blocks fold with the outer batch onto [.z]. Workgroup loops: the
   register splits [i_w] then [j_w] for the blocktile pipeline (so [bn/tn] binds [.x] and [bm/tm]
   binds [.y]), the tensorization lane alone for the mma pipeline, whose column block IS the lane
   width. Parameters that name no block geometry — every CPU pipeline — predict nothing: the C
   backends render annotated loops serially and have no launch to bound. *)
let matmul_launch_geometry (site : matmul_site) (p : sketch_params) : Sched.launch_geometry =
  if not (p.sk_gpu && p.sk_bm > 0 && p.sk_bn > 0) then Sched.unknown_launch_geometry
  else
    let rows = blocks_of site.m_ni p.sk_bm and cols = blocks_of site.m_nj p.sk_bn in
    let grid =
      match batch_layout_of p with
      | `Serial -> [ rows; cols ]
      | `Grid -> List.map (matmul_batch_loops site) ~f:snd @ [ rows; cols ]
      | `Grid_inner -> List.map site.m_bo ~f:snd @ (rows :: List.map site.m_bi ~f:snd) @ [ cols ]
    in
    let block =
      if p.sk_mma then [ p.sk_simd ]
      else if p.sk_tm > 0 && p.sk_tn > 0 then [ p.sk_bm / p.sk_tm; p.sk_bn / p.sk_tn ]
      else []
    in
    predicted_launch_geometry ~grid ~block

(* The launch geometry a GPU convolution sketch will have. The outer output loops are [Grid] in nest
   order; the blocked flavor appends its row-block loop, making those blocks the innermost grid
   coordinate ([.x]) and folding any excess outer coordinates onto [.z]. The tensorization lane is
   the pipeline's only [Workgroup] loop. As for the matmul prediction above, CPU parameters name no
   device launch and therefore predict nothing. *)
let conv_launch_geometry (site : conv_site) (p : sketch_params) : Sched.launch_geometry =
  if not p.sk_gpu then Sched.unknown_launch_geometry
  else
    let grid =
      List.map site.c_outer ~f:snd @ if p.sk_bm > 0 then [ blocks_of site.c_nrow p.sk_bm ] else []
    in
    predicted_launch_geometry ~grid ~block:[ p.sk_simd ]

(* The site nest's own batch geometry under [sk_batch_grid]: whole-loop [Grid] retypes of the batch
   loops (the hoists below have placed them). *)
let site_batch_ops ~(layout : batch_layout) (site : matmul_site) : Sched.schedule =
  match layout with
  | `Serial -> []
  | `Grid | `Grid_inner ->
      List.map (matmul_batch_loops site) ~f:(fun (g, _) -> Sched.Retype { axis = g; ty = LL.Grid })

(* Hoist interior batch loops above the [m_i] loop (gh-ocannl-528), making the [i x j x k]
   micro-kernel perfectly nested for the splits, sinks and [Tensorize] below. Sequential adjacent
   [Swap]s: after each, [m_i] is directly above the next interior batch loop. *)
let batch_hoist_swaps (site : matmul_site) : Sched.schedule =
  hoist_above ~outer:site.m_i (List.map site.m_bi ~f:fst)

(* The GPU pipelines' interior-batch hoists, split around the row's block split [sp_i] whose
   in-block remainder is [i_i]: [(before, after)]. Every layout but [`Grid_inner] hoists above [m_i]
   before any split; [`Grid_inner] (gh-ocannl-728) hoists above [i_i] right after [sp_i], leaving
   the interior batch loops between the row blocks and the column blocks. *)
let row_hoists ~(layout : batch_layout) (site : matmul_site) ~(i_i : Idx.symbol) :
    Sched.schedule * Sched.schedule =
  match layout with
  | `Grid_inner -> ([], hoist_above ~outer:i_i (List.map site.m_bi ~f:fst))
  | `Serial | `Grid -> (batch_hoist_swaps site, [])

(* The k-block loops of a pipeline, in nest order (gh-ocannl-683): the site's outer contraction
   loops followed by the loop the pipeline's own k-split minted ([k_o], or nothing for the unsplit
   whole-[m_k] forms). A multi-axis contraction is a k-loop lowering has already split, so wherever
   a pipeline names "the k-block loop" — the loops the output roles sink below, the anchor the
   staged tiles reload at, the loop the accumulator is privatized over (the OUTERMOST block loop, so
   the private tile stays resident across the whole reduction) — it names this list. Empty [m_ko]
   makes it exactly [k_o], so single-axis sites keep byte-identical schedules. *)
let k_blocks (site : matmul_site) (k_o : Idx.symbol list) : Idx.symbol list =
  List.map site.m_ko ~f:fst @ k_o

(* How a refutation names the extent a tile's k-extent is judged against (gh-ocannl-683). Only the
   INNERMOST contraction loop's extent [m_nk] takes part in the divisibility gates — the outer
   contraction loops are k-block loops the pipeline inherits already split ([k_blocks]) — so on a
   multi-axis site a bare "k=32" reads as the site's whole contraction and misleads anyone reading
   refutation logs or {!Ir.Schedule_space.refutations}: attention's out projection has [m_nk = 32]
   over a total K of 256. Single-axis sites render "k=%d" exactly as before, which the sketch
   goldens quote. *)
let k_extent_label (site : matmul_site) : string =
  if List.is_empty site.m_ko then Printf.sprintf "k=%d" site.m_nk
  else
    Printf.sprintf "innermost contraction extent k=%d (of K=%d over %d loops)" site.m_nk
      (List.fold site.m_ko ~init:site.m_nk ~f:(fun acc (_, n) -> acc * n))
      (1 + List.length site.m_ko)

(* The register-blocktiled GPU matmul (schedule_register_matmul.ml): each output dimension split
   twice (block tile -> Grid, register tile -> Workgroup), register loops sunk innermost, operands
   staged through workgroup-shared tiles at the k-block loop, output privatized, register loops
   materially unrolled. The zeroing nest gets the same geometry (barriers need slot-uniform
   workgroup extents), and companion nests the matching per-position split pair
   ([companion_geometry], gh-ocannl-521). *)
let gpu_sketch_schedule ~accum_prec ~(opt : LL.optimized) (site : matmul_site)
    ({ sk_bm = bm; sk_bn = bn; sk_bk = bk; sk_tm = tm; sk_tn = tn; sk_epilogue; sk_batch_grid; _ }
     as p) : Sched.schedule =
  let layout = batch_layout_of p in
  (* One geometry description drives the accumulation nest, the expanded zeroing nest and the
     companion nests: per row/column chain position, the block split (Grid) and the register split
     (Workgroup), which is what makes their slots and workgroup extents agree by construction; batch
     positions stay [Serial] (gh-ocannl-528), or become whole-loop [Grid] axes under the
     [sk_batch_grid] twins (gh-ocannl-643), placed by the [batch_layout]. *)
  let split_pair ~blk ~reg sym =
    let sp, _, inner = Sched.split ~axis:sym ~factor:blk ~outer:LL.Grid ~inner:LL.Serial in
    let sp2, _, _ = Sched.split ~axis:inner ~factor:reg ~outer:LL.Workgroup ~inner:LL.Serial in
    { rp_block = [ sp ]; rp_inner = inner; rp_rest = [ sp2 ] }
  in
  let row_parts ?extent sym =
    let rp = split_pair ~blk:bm ~reg:tm sym in
    match extent with
    | None -> rp
    | Some extent -> { rp with rp_block = pad_to ~axis:sym ~extent bm @ rp.rp_block }
  in
  let annotate_role role sym =
    match role with
    | `Batch -> if sk_batch_grid then [ Sched.Retype { axis = sym; ty = LL.Grid } ] else []
    | (`Row | `Col) as rc ->
        let blk, reg = match rc with `Row -> (bm, tm) | `Col -> (bn, tn) in
        let rp = split_pair ~blk ~reg sym in
        rp.rp_block @ rp.rp_rest
  in
  let roles = Array.of_list (matmul_chain_roles site) in
  let annotate = companion_role_ops ~roles ~annotate_role ~row_parts ~layout in
  let cops =
    match
      companion_geometry ~site_syms:(matmul_site_chain site)
        ~skip:(if sk_epilogue then epilogue_tail_loop_syms ~target:site.m_d opt else [])
        ~expanded_zeros:(if site.m_zeroed then [ site.m_d ] else [])
        ~annotate opt
    with
    | Ok ops -> ops
    | Error why -> companion_coverage_unsupported ~tensorized:false why
  in
  let zops =
    zero_geometry ~layout site ~mk_zops:(fun ~zi ~zj ~interior ->
        let rp = row_parts ?extent:(if List.is_empty interior then None else Some site.m_ni) zi in
        rp.rp_block @ hoist_above ~outer:rp.rp_inner interior @ rp.rp_rest @ annotate_role `Col zj)
  in
  let zops = cops @ zops @ site_batch_ops ~layout site in
  let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let hoists_before, hoists_after = row_hoists ~layout site ~i_i in
  let sp_i2, i_w, i_t = Sched.split ~axis:i_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_j, j_o, j_i = Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
  let sp_j2, j_w, j_t = Sched.split ~axis:j_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let kb = k_blocks site [ k_o ] in
  let swaps = sink i_t ([ j_o; j_w; j_t ] @ kb @ [ k_i ]) @ sink j_t (kb @ [ k_i ]) in
  (* Pad composition (gh-ocannl-485, gh-ocannl-730): this pipeline stages BOTH operands through
     zero-fringe workgroup tiles at every geometry, so non-multiple extents pad to the block sizes
     instead of refuting — the same argument the tensorized family has used since gh-ocannl-485,
     with the leaf guards discharged by [Privatize]'s mask classification rather than by
     [Tensorize]. Identity pads are omitted, so a dividing site keeps a byte-identical schedule. *)
  let pads =
    if not (pad_composition_ok ~n_staged:(if bk > 0 then 2 else 0) ~n_operands:2) then []
    else
      pad_to ~axis:site.m_i ~extent:site.m_ni bm
      @ pad_to ~axis:site.m_j ~extent:site.m_nj bn
      @ pad_to ~axis:site.m_k ~extent:site.m_nk bk
  in
  hoists_before @ pads @ zops @ [ sp_i ] @ hoists_after @ [ sp_i2; sp_j; sp_j2; sp_k ] @ swaps
  @ [
      Sched.Stage
        {
          source = site.m_a;
          tile_loops = [ i_w; i_t; k_i ];
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        };
      Sched.Stage
        {
          source = site.m_b;
          tile_loops = [ k_i; j_w; j_t ];
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        };
      Sched.privatize ~accum_prec ~target:site.m_d ~over:(List.hd_exn kb);
      Sched.Unroll { axis = i_t; materialize = true };
      Sched.Unroll { axis = j_t; materialize = true };
    ]

(* A constant operand eligible for hoisted (out-of-routine) packing (gh-ocannl-470). The same
   predicate enters the canonical digest ([Schedule_cache.canonicalize]), so a cached winner for a
   same-shape program of different operand constancy never replays here — hoisted candidates are
   always measured for constant sites. *)
let hoistable = Sched.hoistable_constant

(* Whether the matmul pipelines pack operand [tn] of [site] at link time (gh-ocannl-1148): a
   hoistable constant, unless it is the A of a strided-row site, whose tile must be compacted --
   which hoisted staging refuses (v1) -- so it packs in kernel instead. Every pipeline and every
   family-tree verdict asks this, never [hoistable] directly, so a shape the tree proposes stages
   its operands exactly as the builder will. *)
let site_hoistable (site : matmul_site) tn =
  hoistable tn && (site.m_row_stride = 1 || not (phys_equal tn site.m_a))

(* Why a strided-row site whose only constant operand is its A packs nothing at link time
   (gh-ocannl-1148). *)
let strided_row_link_time_witness (site : matmul_site) =
  Printf.sprintf
    "the only host-init-backed constant operand is A, whose stride-%d row needs a compacting \
     Stage, which link-time packing refuses (v1)"
    site.m_row_stride

(* The family tree's witness for a strided-row site's geometries that would read A in place
   (gh-ocannl-1148). *)
let strided_row_in_place_witness (site : matmul_site) ~what =
  Printf.sprintf
    "the row is a stride-%d window of A: %s reads A in place, and Tensorize needs the row at unit \
     coefficient, which only a compacting Stage of A gives"
    site.m_row_stride what

(* The CPU operand-packing matmul (schedule_cpu_pack_matmul.ml): all-serial tiling with the tile
   loops sunk to [i_o j_o k_o k_i i_i j_i], operands packed into contiguous stack scratch, output
   privatized across the k-block loop. With [sk_hoist], constant operands are instead packed once at
   link time into the per-device constant pool. *)
let cpu_sketch_schedule ~accum_prec (site : matmul_site)
    { sk_bm = bm; sk_bn = bn; sk_bk = bk; sk_hoist; _ } : Sched.schedule =
  let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
  let sp_j, j_o, j_i = Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Serial ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let kb = k_blocks site [ k_o ] in
  batch_hoist_swaps site @ [ sp_i; sp_j; sp_k ]
  @ sink i_i ([ j_o; j_i ] @ kb @ [ k_i ])
  @ sink j_i (kb @ [ k_i; i_i ])
  @ [
      Sched.Stage
        {
          source = site.m_a;
          tile_loops = [ i_i; k_i ];
          shared = false;
          cooperative = None;
          hoisted = sk_hoist && site_hoistable site site.m_a;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        };
      Sched.Stage
        {
          source = site.m_b;
          tile_loops = [ k_i; j_i ];
          shared = false;
          cooperative = None;
          hoisted = sk_hoist && site_hoistable site site.m_b;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        };
      Sched.privatize ~accum_prec ~target:site.m_d ~over:(List.hd_exn kb);
    ]

(* Tensorized (tile-MMA) GPU matmul (docs/proposals/tensorize-mma.md; the pinned pipelines of
   schedule_mma_matmul.ml): Split the output dims into Grid blocks, then [Tensorize] the inner
   micro-kernel into a [Tile_mma] block statement. Stage-only composition — [Privatize] must NOT
   join it: it would relocate the accumulator into thread-local scratch, which the MMA loads cannot
   address ([mma_syntax] declines thread-space operands, silently costing the whole tensorization),
   and [Tile_mma]'s block semantics already keep the accumulator fragments register-resident across
   the reduction. With [sk_bk = 0] the single block statement spans the full reduction, streaming
   operand tiles from device memory and amortizing [d] traffic entirely; with [sk_bk > 0] both
   operands are staged through cooperative shared tiles at the k-block loop (lane-aware Stage),
   costing one [d] fragment load/store per k-block. The zeroing nest mirrors the accumulation's grid
   geometry, with an inner Workgroup loop of extent [sk_simd] covering the lane slot
   (barrier-strength uniformity: every workgroup extent must equal the lane width once a [Tile_mma]
   is present) — the seeds constrain [sk_bn = sk_simd] so the zeroing's grid blocks align with
   [j]'s. Companion nests (a bias/relu tail; the elementwise statements an aligned-merged fission
   segment carries) get the same geometry, which is what lets an UNFUSED tensorized candidate
   compile at all (gh-ocannl-521): before, only the [Fuse_epilogue] twin could survive a companion,
   and when the fusion declined the seed had no surviving form. *)
let gpu_mma_sketch_schedule ~(opt : LL.optimized) (site : matmul_site)
    ({
       sk_bm = bm;
       sk_bn = bn;
       sk_bk = bk;
       sk_simd = w;
       sk_epilogue;
       sk_swizzle;
       sk_depth;
       sk_batch_grid;
       _;
     } as p) : Sched.schedule =
  let layout = batch_layout_of p in
  (* The column role splits at the lane width, not at [bn]: the inner loop IS the workgroup slot the
     [Tile_mma]'s lane occupies, and a barrier-carrying kernel requires equal extents at a slot. The
     seeds constrain [sk_bn = sk_simd], so this is also the accumulation nest's column block. Batch
     positions stay [Serial] (gh-ocannl-528), or become whole-loop [Grid] axes under the
     [sk_batch_grid] twins (gh-ocannl-643), placed by the [batch_layout]. *)
  let row_parts ?extent sym =
    let sp, _, inner = Sched.split ~axis:sym ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let pad = match extent with None -> [] | Some extent -> pad_to ~axis:sym ~extent bm in
    { rp_block = pad @ [ sp ]; rp_inner = inner; rp_rest = [] }
  in
  let annotate_role role sym =
    match role with
    | `Batch -> if sk_batch_grid then [ Sched.Retype { axis = sym; ty = LL.Grid } ] else []
    | `Row -> (row_parts sym).rp_block
    | `Col ->
        let sp, _, _ = Sched.split ~axis:sym ~factor:w ~outer:LL.Grid ~inner:LL.Workgroup in
        [ sp ]
  in
  let roles = Array.of_list (matmul_chain_roles site) in
  let annotate = companion_role_ops ~roles ~annotate_role ~row_parts ~layout in
  let cops =
    match
      companion_geometry ~site_syms:(matmul_site_chain site)
        ~skip:(if sk_epilogue then epilogue_tail_loop_syms ~target:site.m_d opt else [])
        ~expanded_zeros:(if site.m_zeroed then [ site.m_d ] else [])
        ~annotate opt
    with
    | Ok ops -> ops
    | Error why -> companion_coverage_unsupported ~tensorized:true why
  in
  let zops =
    let folded = shared_folded_zero ~p ~target:site.m_d opt in
    zero_geometry ~layout ?folded site ~mk_zops:(fun ~zi ~zj ~interior ->
        let rp = row_parts ?extent:(if List.is_empty interior then None else Some site.m_ni) zi in
        rp.rp_block @ hoist_above ~outer:rp.rp_inner interior @ rp.rp_rest @ annotate_role `Col zj)
  in
  let zops = cops @ zops @ site_batch_ops ~layout site in
  let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let hoists_before, hoists_after = row_hoists ~layout site ~i_i in
  let sp_j, j_o, j_i = Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
  if bk = 0 then
    (* Unsplit: the block statement spans [m_k]; a site's outer contraction loops stay above it. *)
    let kb = k_blocks site [] in
    let tz, _lane = Sched.tensorize ~i:i_i ~j:j_i ~k:site.m_k ~simd_width:w () in
    hoists_before @ zops @ [ sp_i ] @ hoists_after @ [ sp_j ] @ sink i_i [ j_o ] @ sink j_i kb
    @ sink i_i kb @ [ tz ]
  else
    let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let kb = k_blocks site [ k_o ] in
    let tz, _lane = Sched.tensorize ~i:i_i ~j:j_i ~k:k_i ~simd_width:w () in
    (* Pad-composition seeding (gh-ocannl-485): with both operands staged through zero-fringe
       cooperative tiles, non-multiple extents pad to the block sizes — the guards land on the leaf
       accumulation, [Tensorize] moves the row/column masks to the fragment transfers and discharges
       the reduction mask against the staged tiles. *)
    let pads =
      if not (pad_composition_ok ~n_staged:(if bk > 0 then 2 else 0) ~n_operands:2) then []
      else
        pad_to ~axis:site.m_i ~extent:site.m_ni bm
        @ pad_to ~axis:site.m_j ~extent:site.m_nj bn
        @ pad_to ~axis:site.m_k ~extent:site.m_nk bk
    in
    hoists_before @ pads @ zops @ [ sp_i ] @ hoists_after @ [ sp_j; sp_k ] @ sink i_i [ j_o ]
    @ sink j_i kb @ sink i_i kb
    @ [
        (* The swizzled twin (gh-ocannl-481 item 3, D3) marks BOTH operand tiles: the tile sizes and
           the whole rest of the pipeline are identical to its plain sibling, so a timing difference
           between the two is the layout's, and nothing else's. *)
        Sched.Stage
          {
            source = site.m_a;
            tile_loops = [ i_i; k_i ];
            shared = true;
            cooperative = Some w;
            hoisted = false;
            swizzle = sk_swizzle;
            pad_stride = None;
            pipeline_depth = sk_depth;
            tile_prec = None;
          };
        Sched.Stage
          {
            source = site.m_b;
            tile_loops = [ k_i; j_i ];
            shared = true;
            cooperative = Some w;
            hoisted = false;
            swizzle = sk_swizzle;
            pad_stride = None;
            pipeline_depth = sk_depth;
            tile_prec = None;
          };
        tz;
      ]

(* Whole-triple tensorized CPU matmul (gh-ocannl-469; bin/schedule_bench.ml's [tensorize] variant):
   one [Tile_mma] statement the C backends render tinyBLAS-style — the C-tile in an RM×RN grid of
   vector registers held across the k-loop, edges as narrower tiles. The zeroing's column loop
   becomes the Workgroup axis with the lane width matching its extent (coverage rule; the lane loop
   renders serially on the C backends). With [sk_bm > 0] the row loops split into pool-parallel Grid
   blocks; [sk_bm = 0] keeps the single-statement form. *)
let cpu_mma_sketch_schedule (site : matmul_site) { sk_bm = bm; sk_tile = tile; _ } : Sched.schedule
    =
  let zops =
    zero_geometry site ~mk_zops:(fun ~zi ~zj ~interior:_ ->
        let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
        if bm = 0 then [ rz ]
        else
          let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
          [ sp_zi; rz ])
  in
  let kb = k_blocks site [] in
  if bm = 0 then
    let tz, _lane =
      Sched.tensorize ?tile ~i:site.m_i ~j:site.m_j ~k:site.m_k ~simd_width:site.m_nj ()
    in
    batch_hoist_swaps site @ zops @ sink site.m_j kb @ sink site.m_i kb @ [ tz ]
  else
    let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let tz, _lane = Sched.tensorize ?tile ~i:i_i ~j:site.m_j ~k:site.m_k ~simd_width:site.m_nj () in
    batch_hoist_swaps site @ zops @ [ sp_i ] @ sink site.m_j kb @ sink i_i kb @ [ tz ]

(* Cache-blocked, operand-packed tensorized CPU matmul: [Tile_mma] composed with the S4 packing
   pipeline (the remaining piece of gh-ocannl-469). GEBP loop structure, all-Serial: [j_o? { k_o {
   pack B~[bk x bn]; i_o { pack A~[bm x bk]; Tile_mma(bm, bn, bk) } } }] — the packing [Stage]s land
   at their own anchors (B~ at [k_o], once per (j_o, k_o) block; A~ at [i_o]) and the register-tiled
   micro-kernel streams the contiguous, cache-resident tiles ([lda = bk], [ldb = bn]). [tile_loops]
   are passed in micro-kernel order ([k_i; j_i] for B), so a transposed source packs into the
   normalized layout and [Tensorize] sees [ta = tb = false]. [sk_bn = 0] leaves [j] unsplit (one B~
   row panel of [bk x nj] per k-block). The lane width is 1: the C backends render the lane loop
   serially, and a unit lane keeps the kernel's parallel geometry trivial. Hoisted packing (constant
   operands, gh-ocannl-470) is proposed per operand like the scalar S4 pipeline.

   With [sk_grid], the row-block loop [i_o] is [Grid]-typed and pool-parallelizes; the whole-node
   [Zero_out] of the output — no longer legal beside a hardware-annotated loop ([validate_parallel])
   — expands into a nest whose row loop Grid-splits with the same [bm] geometry ([zero_geometry];
   the unit-lane Workgroup axis has extent 1, stays inactive, and needs no coverage from the zeroing
   nest). Four shapes (see [sk_grid]):

   - [sk_hoist]: hoisted-only packing — hoistable operands are packed at link time into the constant
   pool, the rest are read in place, so the kernel body touches only materialized buffers; the Grid
   loop stays outermost (one dispatch spanning the whole GEBP triple). The typical inference GEMM:
   activations (in place) x constant weights (hoisted-packed panel). - [sk_hoist] with
   [sk_pack_rest]: the mixed grid-outermost shape (gh-ocannl-473) — same loop structure, but
   non-hoistable operands get a non-hoisted in-kernel Stage; their tiles land inside the Grid body
   and rely on the renderer's per-chunk privatization (an in-place read forfeits the pack entirely;
   an A~ tile is [bm x bk], comfortably per-chunk). - [sk_pack_rest] alone: grid-outermost in-kernel
   packing (gh-ocannl-475) — both operands pack inside the Grid body, each chunk re-packing its own
   B~ panel; one dispatch, tiles under the renderer's per-chunk cap. - Otherwise, in-kernel packing:
   [i_o] sinks under [j_o]/[k_o] exactly as in the serial shape, so the B~ panel packs outside the
   Grid body (read-only inside, shared across the row-block chunks) while the per-row-block A~ tile
   is privatized to per-chunk block-scope storage by the renderer ([C_syntax.parallel_grid_safe]'s
   privatization rule). *)
let cpu_mma_pack_sketch_schedule (site : matmul_site)
    {
      sk_bm = bm;
      sk_bn = bn;
      sk_bk = bk;
      sk_hoist;
      sk_grid;
      sk_pack_rest;
      sk_pack_prec;
      sk_tile = tile;
      _;
    } : Sched.schedule =
  let outer_i = if sk_grid then LL.Grid else LL.Serial in
  let grid_outermost = sk_grid && (sk_hoist || sk_pack_rest) in
  let sp_i, i_o, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:outer_i ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let splits, j_col, j_swaps =
    if bn = 0 then ([ sp_i; sp_k ], site.m_j, [])
    else
      let sp_j, j_o, j_i =
        Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Serial ~inner:LL.Serial
      in
      ([ sp_i; sp_j; sp_k ], j_i, sink i_i [ j_o ] @ if grid_outermost then [] else sink i_o [ j_o ])
  in
  let stage ~hoisted source tile_loops =
    (* [sk_pack_prec] is the site's compute precision; mint the packed tile there where the source
       stores narrower (gh-ocannl-575) — the widening rides the packing copy and the register-tiled
       micro-kernel reads uniform compute-precision panels. Normalized to [None] when the source
       already stores at it, keeping the [Stage] canonical. *)
    let tile_prec =
      Option.bind sk_pack_prec ~f:(fun p ->
          if Ir.Ops.equal_prec p (Lazy.force source.Ir.Tnode.storage_prec) then None else Some p)
    in
    Sched.Stage
      {
        source;
        tile_loops;
        shared = false;
        cooperative = None;
        hoisted;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec;
      }
  in
  let stages =
    if grid_outermost then
      List.filter_map
        [ (site.m_b, [ k_i; j_col ]); (site.m_a, [ i_i; k_i ]) ]
        ~f:(fun (src, tls) ->
          if site_hoistable site src then Some (stage ~hoisted:true src tls)
          else if sk_pack_rest then Some (stage ~hoisted:false src tls)
          else None)
    else
      [
        stage ~hoisted:(sk_hoist && site_hoistable site site.m_b) site.m_b [ k_i; j_col ];
        stage ~hoisted:(sk_hoist && site_hoistable site site.m_a) site.m_a [ i_i; k_i ];
      ]
  in
  let zops =
    if not sk_grid then []
    else
      zero_geometry site ~mk_zops:(fun ~zi ~zj:_ ~interior:_ ->
          let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
          [ sp_zi ])
  in
  (* Pad-composition seeding (gh-ocannl-485), only when both operands are staged (hoisted packing
     zero-fills its pad slots, so it qualifies): non-multiple extents pad to the block sizes and
     [Tensorize] masks the fragment transfers. An in-place operand read cannot absorb a pad, so the
     grid-outermost hoisted-only shape pads only if every operand packs. *)
  let both_staged = pad_composition_ok ~n_staged:(List.length stages) ~n_operands:2 in
  let pads =
    if not both_staged then []
    else
      pad_to ~axis:site.m_i ~extent:site.m_ni bm
      @ (if bn = 0 then [] else pad_to ~axis:site.m_j ~extent:site.m_nj bn)
      @ pad_to ~axis:site.m_k ~extent:site.m_nk bk
  in
  let tz, _lane = Sched.tensorize ?tile ~i:i_i ~j:j_col ~k:k_i ~simd_width:1 () in
  let kb = k_blocks site [ k_o ] in
  batch_hoist_swaps site @ pads @ zops @ splits @ j_swaps @ sink j_col kb @ sink i_i kb
  @ (if grid_outermost then [] else sink i_o kb)
  @ stages @ [ tz ]

(* Adjacent-transposition reorder of a perfect serial nest: [Swap]s that turn [current] (nest order,
   outermost first) into [target]. Selection sort — for each target position, bubble the wanted loop
   outward one level at a time (each [Swap] exchanges a directly-nested pair). *)
let reorder_swaps ~current ~target : Sched.schedule =
  let cur = Array.of_list current in
  let swaps = ref [] in
  let not_a_permutation () =
    (* invariant: every caller derives [current] and [target] from one detected site's loops, so a
       loop of [target] missing from [current], or wanted twice, is a builder bug. *)
    invalid_arg "Autotune.reorder_swaps: not a permutation"
  in
  List.iteri target ~f:(fun p want ->
      match Array.findi cur ~f:(fun _ s -> Idx.equal_symbol s want) with
      | (None | Some (0, _)) when p > 0 -> not_a_permutation ()
      | None -> not_a_permutation ()
      | Some (q, _) ->
          if q < p then not_a_permutation ();
          for r = q downto p + 1 do
            swaps := Sched.Swap { outer = cur.(r - 1); inner = cur.(r) } :: !swaps;
            let tmp = cur.(r - 1) in
            cur.(r - 1) <- cur.(r);
            cur.(r) <- tmp
          done);
  List.rev !swaps

(* The segment's real top-level statements as the conv seeding counts them: glue excluded, and the
   conv site's own covering zero initializer excluded (the pipeline's zero geometry handles it);
   every other statement is a companion nest. *)
let conv_real_stmts (site : conv_site) (opt : LL.optimized) : LL.t list =
  List.filter (LL.flat_lines [ opt.LL.llc ]) ~f:(function
    | LL.Noop | LL.Comment _ -> false
    | stmt -> not (Option.exists (LL.zero_initializer_target stmt) ~f:(Ir.Tnode.equal site.c_d)))

(* The conv output's epilogue tail (see [epilogue_tail_loop_syms]): the fused twins on
   aligned-merged segments omit the preset's [Retype] on that nest (fuse-before-annotate,
   gh-ocannl-501). *)
let conv_tail_loop_syms (site : conv_site) (opt : LL.optimized) : Idx.symbol list =
  epilogue_tail_loop_syms ~target:site.c_d opt

(* The chain the GPU conv pipeline annotates, as [companion_geometry] wants it: the outer output
   loops, the GEMM row, the out-channel column — the conv analog of [matmul_site_chain]. *)
let conv_site_chain (site : conv_site) =
  site.c_outer @ [ (site.c_row, site.c_nrow); (site.c_oc, site.c_noc) ]

(* The GPU conv pipeline's companion query, one spelling for the builder and the seeding witness:
   the fused flavor skips the epilogue tail nest [Fuse_epilogue] consumes, and a zeroed site's
   whole-node [Zero_out] is the pipeline's own ([conv_zero_geometry]). *)
let conv_companion_geometry ~(opt : LL.optimized) ~(fused : bool) ~annotate (site : conv_site) =
  companion_geometry ~site_syms:(conv_site_chain site)
    ~skip:(if fused then conv_tail_loop_syms site opt else [])
    ~expanded_zeros:(if site.c_zeroed then [ site.c_d ] else [])
    ~annotate opt

(* The statically-decidable companion-coverage verdict of the GPU conv seeds (gh-ocannl-1406), the
   conv analog of [matmul_coverage_witness]: [companion_geometry]'s Ok/Error never depends on the
   geometry [annotate] emits, so one query with a trivial annotator settles buildability for every
   flavor of the leg. It replaces a [real_stmts <= 2] count cap, which refused aligned-merged
   segments (conv + bias/relu + pooling, 3-6 statements) whose companions the pipeline already
   annotates, and admitted segments whose single companion it cannot. Flavor-indexed: the conv
   family tree (gh-ocannl-697) judges each fusion flavor by its own verdict, before any geometry —
   skipping the epilogue tail can empty the demand, so a fused flavor can build where the unfused
   one is refuted. The fused demand is a subset of the unfused one, so an unfused [None] implies the
   fused one. The builder's raise stays as the safety net for parameters replayed against another
   lowering. *)
let conv_coverage_witness ~(opt : LL.optimized) ~(fused : bool) (site : conv_site) : string option =
  match conv_companion_geometry ~opt ~fused ~annotate:(fun _ -> []) site with
  | Ok _ -> None
  | Error why -> Some (Printf.sprintf "GPU conv companion coverage (gh-521): %s" why)

(* The CPU conv pipeline's flavors that Grid-type a loop of the conv nest itself — the Grid flavor
   and the row panels, wherever the segment holds at most one statement besides the conv (more, and
   they adopt the preset's aligned whole-segment geometry, [conv_aligned_grid]) — leave every other
   nest outside the only annotated loop. [validate_parallel] rejects such a nest that writes a
   materialized node (every pool chunk would execute it), so the flavor is refuted where one
   remains: the companion demand, with the fused flavor's tail skipped (the fusion relocates it
   under the Grid loop) and a whole-node zero of the output excluded (the pipeline expands and
   annotates it). Before gh-ocannl-697 the unfused Grid flavor of a conv+tail segment was proposed
   and failed validation at every compile, its fused twin the only survivor. *)
let conv_direct_grid_witness ~(opt : LL.optimized) ~(fused : bool) (site : conv_site) :
    string option =
  match
    companion_demand ~site_syms:(conv_site_chain site)
      ~skip:(if fused then conv_tail_loop_syms site opt else [])
      ~expanded_zeros:(if site.c_zeroed then [ site.c_d ] else [])
      opt
  with
  | [] -> None
  | needs ->
      let plc = opt.LL.optimize_ctx.LL.placements in
      Some
        (Printf.sprintf
           "the conv nest's own Grid loop leaves %d companion nest(s) writing materialized %s \
            outside it, which validate_parallel rejects"
           (List.length needs)
           (String.concat ~sep:"; " (List.map needs ~f:(materialized_writes ~plc))))

(* Whole-segment Grid alignment for the conv pipeline on merged segments (gh-ocannl-493): the
   pipeline's own [Retype] covers only the conv nest, so on an aligned-merged segment (lenet's
   conv+bias/relu+pooling) the companions' materialized writes would fail [validate_parallel]. Reuse
   the default CPU preset's aligned cross-nest analysis instead of re-proving alignment: a non-empty
   [Sched.default_cpu] schedule Grid-retypes the outermost qualifying loop of {e every}
   materialized-writing nest, with the equal-extent common-prefix trims applied — exactly the
   whole-segment geometry the fissioned default runs. Accept it as the conv sketch's grid ops when
   it covers the conv nest at its outermost loop (which must be an outer output loop of extent >= 2,
   so the pipeline's reorder keeps it outermost and pool chunking has work to split). *)
let conv_aligned_grid (site : conv_site) (opt : LL.optimized) : Sched.schedule option =
  match (site.c_outer, site.c_loops) with
  | (outermost, n) :: _, first :: _ when n >= 2 && Idx.equal_symbol outermost first -> (
      match Sched.default_cpu ~min_parallel:1 opt with
      | [] -> None
      | sched ->
          if
            List.exists sched ~f:(function
              | Sched.Retype { axis; ty = LL.Grid } -> Idx.equal_symbol axis outermost
              | _ -> false)
          then Some sched
          else None)
  | _ -> None

(* The current nest order after a [Split] of the GEMM row into [row_o { row_i }] (in place, as
   [rewrite_loop] produces it), for feeding [reorder_swaps]. A dividing block factor keeps the split
   guard-free (see [Schedule.apply_op]'s [Split]), so the nest stays a perfect serial nest and the
   subsequent [Swap]s are well-formed. *)
let conv_split_row_current (site : conv_site) ~row_o ~row_i : Idx.symbol list =
  List.concat_map site.c_loops ~f:(fun s ->
      if Idx.equal_symbol s site.c_row then [ row_o; row_i ] else [ s ])

(* The implicit-GEMM conv pipeline (gh-ocannl-493), CPU route: reorder the accumulation nest to
   [outer..; kernel..; row; oc; ic], pack the input's [row × ic] strided-window slice and the
   kernel's [ic × oc] slice (both anchor under the innermost kernel-window loop; the packing IS
   im2col, one window slice at a time, and normalizes the kernel's stored layout), then [Tensorize
   (row, oc, ic)] — the register-tiled [Tile_mma] micro-kernel, with the accumulator contracted to a
   fragment resident across the whole kernel-window chain (gh-ocannl-480, gh-ocannl-501: one
   fragment init/store per output tile). With [sk_grid], the outermost output loop is [Grid]-typed
   and pool-parallelizes; a whole-node [Zero_out] of the output then expands with the matching
   geometry. On a segment with more than one companion statement the grid ops come from
   [conv_aligned_grid] instead, so every companion nest is annotated with the aligned whole-segment
   geometry (such segments carry no [Zero_out] — the preset's analysis bails on those, and the seeds
   gate accordingly).

   With [sk_bm > 0] (gh-ocannl-500) the GEMM row is split into panels of [sk_bm] rows before the
   reorder — cache-blocked GEBP-style panels, the conv analog of [cpu_mma_sketch_schedule]'s
   row-block split — and the in-panel [row_i × oc] micro-kernel is tensorized (the register tiling
   covers its own sub-tile edges). [sk_bm] must divide [c_nrow] so the split stays guard-free (a
   remainder guard would break the reorder's perfect nesting). The panel loop's parallelism source
   depends on the segment: on a conv-alone segment the panel loop is [Grid]-typed directly (one pool
   chunk per row-block); on an aligned-merged segment (conv + materialized companions, e.g. lenet's
   conv+bias/relu+pooling) the whole-segment [Grid] geometry comes from [conv_aligned_grid] as for
   the unblocked flavor and the panel loop stays [Serial] — pure cache blocking within each pool
   chunk. Both cases exclude whole-node [Zero_out] (the seeds gate accordingly). A covering per-cell
   zero companion follows the segment's aligned Grid geometry. *)
let cpu_conv_sketch_schedule ~(opt : LL.optimized) (site : conv_site)
    { sk_grid; sk_bm; sk_epilogue; sk_pack_prec; sk_tile = tile; _ } : Sched.schedule =
  let stage source tile_loops =
    (* Mint the im2col/panel tiles at the site's compute precision where the source stores narrower
       (gh-ocannl-575); see [cpu_mma_pack_sketch_schedule]. *)
    let tile_prec =
      Option.bind sk_pack_prec ~f:(fun p ->
          if Ir.Ops.equal_prec p (Lazy.force source.Ir.Tnode.storage_prec) then None else Some p)
    in
    Sched.Stage
      {
        source;
        tile_loops;
        shared = false;
        cooperative = None;
        hoisted = false;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec;
      }
  in
  (* Fuse-before-annotate (gh-ocannl-501): the fused twin of an aligned-merged seed omits the preset
     [Retype] on the tail nest [Fuse_epilogue] consumes — see [conv_tail_loop_syms]. *)
  let drop_tail_retypes sched =
    if not sk_epilogue then sched
    else
      let tail_syms = conv_tail_loop_syms site opt in
      List.filter sched ~f:(function
        | Sched.Retype { axis; _ } -> not (List.mem tail_syms axis ~equal:Idx.equal_symbol)
        | _ -> true)
  in
  if sk_bm > 0 then
    (* A non-dividing row block pads the row to the block size (gh-ocannl-485): the pad's
       leaf-statement guards keep the nest perfectly nested for the reorder's [Swap]s, and both
       operands pack through zero-fringe tiles, so [Tensorize] masks the fragment transfers. *)
    let row_pads = pad_to ~axis:site.c_row ~extent:site.c_nrow sk_bm in
    (* On a merged segment the aligned whole-segment [Grid] annotation parallelizes; the panel loop
       is a serial cache block. On a conv-alone segment the panel loop is the parallel [Grid]. *)
    let grid_ops, panel_axis =
      if List.length (conv_real_stmts site opt) > 2 then
        match conv_aligned_grid site opt with
        | Some sched -> (drop_tail_retypes sched, LL.Serial)
        | None -> refuse "Autotune conv sketch: companion nests do not align for Grid"
      else ([], LL.Grid)
    in
    let sp_row, row_o, row_i =
      Sched.split ~axis:site.c_row ~factor:sk_bm ~outer:panel_axis ~inner:LL.Serial
    in
    let current = conv_split_row_current site ~row_o ~row_i in
    let target =
      List.map site.c_outer ~f:fst @ [ row_o ] @ site.c_kernel @ [ row_i; site.c_oc; site.c_red ]
    in
    let tz, _lane = Sched.tensorize ?tile ~i:row_i ~j:site.c_oc ~k:site.c_red ~simd_width:1 () in
    row_pads @ grid_ops
    @ (sp_row :: reorder_swaps ~current ~target)
    @ [ stage site.c_a [ row_i; site.c_red ]; stage site.c_b [ site.c_red; site.c_oc ]; tz ]
  else
    let loop_syms =
      List.map site.c_outer ~f:fst @ site.c_kernel @ [ site.c_row; site.c_oc; site.c_red ]
    in
    let tz, _lane =
      Sched.tensorize ?tile ~i:site.c_row ~j:site.c_oc ~k:site.c_red ~simd_width:1 ()
    in
    let zops, grid_ops =
      if not sk_grid then ([], [])
      else if List.length (conv_real_stmts site opt) > 2 then
        match conv_aligned_grid site opt with
        | Some sched -> ([], drop_tail_retypes sched)
        | None -> refuse "Autotune conv sketch: companion nests do not align for Grid"
      else
        match site.c_outer with
        | [] -> refuse "Autotune conv sketch: no outer loop to Grid-parallelize"
        | (outermost, _) :: _ ->
            let zops =
              if not site.c_zeroed then []
              else
                let ez, zsyms = Sched.expand_zero ~tn:site.c_d in
                match zsyms with
                | z0 :: _ -> [ ez; Sched.Retype { axis = z0; ty = LL.Grid } ]
                | [] -> [ ez ]
            in
            (zops, [ Sched.Retype { axis = outermost; ty = LL.Grid } ])
    in
    zops @ grid_ops
    @ reorder_swaps ~current:site.c_loops ~target:loop_syms
    @ [ stage site.c_a [ site.c_row; site.c_red ]; stage site.c_b [ site.c_red; site.c_oc ]; tz ]

(* The conv-rank generalization of [zero_geometry] (gh-ocannl-503): on a whole-routine lowering
   ([c_zeroed]) the conv output's whole-node [Zero_out] shares the annotated kernel, where
   [validate_parallel] rejects a bare whole-node write. Expand it into a per-cell nest, reorder that
   nest into the site's chain order (the expansion's loops follow [d]'s axis order, [c_d_syms]; the
   site's chain follows its nest order, which positional thread identity across nests is read in),
   and give it the geometry [annotate] gives every companion: Grid on the outer output loops, the
   row-block split, and the column distributed over the [Tensorize] lane width as a [Workgroup] loop
   — barrier-strength uniformity against the lane loops of [Tensorize] and the cooperative [Stage]s.
   That is exactly the geometry a GPU fission segment's folded per-cell zero companion takes
   (gh-ocannl-1175); the renderers' workgroup barrier before every accumulator load of [d] orders
   the zero against the fragment load, as for the companions ([companion_geometry]).

   [folded] (gh-ocannl-1453) is that folded per-cell zero where the fused twin places a
   non-materialized [d] in workgroup-shared storage and companion geometry therefore leaves it to
   this pipeline ({!shared_folded_zero}): it gets the same reorder and geometry, minus the
   expansion. Empty when the site is not zeroed and nothing is folded. *)
let conv_zero_geometry ?folded (site : conv_site) ~(site_syms : (Idx.symbol * int) list)
    ~(annotate : (Idx.symbol * int) list -> Sched.schedule) : Sched.schedule =
  let place ~ez ~current zsyms =
    let by_axis = List.zip_exn site.c_d_syms zsyms in
    let zchain =
      List.map site_syms ~f:(fun (s, n) ->
          (List.Assoc.find_exn by_axis s ~equal:Idx.equal_symbol, n))
    in
    ez @ reorder_swaps ~current ~target:(List.map zchain ~f:fst) @ annotate zchain
  in
  if site.c_zeroed then
    let ez, zsyms = Sched.expand_zero ~tn:site.c_d in
    place ~ez:[ ez ] ~current:zsyms zsyms
  else
    match folded with
    | None -> []
    | Some (nest_order, axis_order) -> place ~ez:[] ~current:nest_order axis_order

(* The GPU staged leg of the implicit-GEMM conv pipeline (gh-ocannl-493): the same loop
   re-association as the CPU route, with the outer output loops [Grid]-typed (one threadgroup per
   outer coordinate, the kernel-window loops serial inside it) and both slices staged through
   cooperative workgroup-shared tiles at the kernel-window anchor (lane-aware [Stage], the lane
   width matching [Tensorize]'s — barrier-strength uniformity). Reusing [Tensorize] inherits the
   accumulator contraction (gh-ocannl-480) unchanged: the [row × oc] fragment stays resident across
   the whole kernel-window chain (gh-ocannl-501), on Metal in simdgroup registers. GPU fission
   expands eligible covering zeros into per-cell companions, which take the same Grid and lane
   geometry; a whole-routine lowering's whole-node [Zero_out] is expanded here into the same nest
   with the same geometry ([conv_zero_geometry], gh-ocannl-503).

   With [sk_bm > 0] (gh-ocannl-500) the GEMM row is additionally split into [Grid] blocks of [sk_bm]
   rows: one threadgroup per (outer.., row-block) coordinate instead of one per outer coordinate, so
   small-spatial sites fill the device better. [sk_bm] must divide [c_nrow] (a remainder guard would
   push the cooperative-load barriers under divergent control flow, rejected by
   [validate_parallel]). Only the row is blocked — a 2-D conv already binds two outer [Grid] loops
   (batch, the non-row output spatial axis), so a second [Grid] block on [oc] would exceed the
   three-slot budget; [oc] stays the tensorized column extent. Every materialized companion,
   including the covering zero, follows the site's outer Grid coordinates, row-block split and
   tensorization lane through [companion_geometry]. *)
let gpu_conv_sketch_schedule ~(opt : LL.optimized) (site : conv_site)
    ({ sk_simd = w; sk_bm; sk_bn; sk_bk; sk_tm; sk_depth; sk_epilogue; _ } as p) : Sched.schedule =
  let stage source tile_loops =
    Sched.Stage
      {
        source;
        tile_loops;
        shared = true;
        cooperative = Some w;
        hoisted = false;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = sk_depth;
        tile_prec = None;
      }
  in
  (* The expanded zero and other companions use the site's output chain. Tensorize adds one
     workgroup lane but no column Grid slot: distribute companion columns over that same lane width,
     keeping any column blocks serial. Outer output coordinates and row blocks have the same Grid
     roles in every nest. *)
  let site_syms = conv_site_chain site in
  let annotate chain =
    List.concat_mapi chain ~f:(fun pos (sym, extent) ->
        if pos < List.length site.c_outer then [ Sched.Retype { axis = sym; ty = LL.Grid } ]
        else if pos = List.length site.c_outer then
          if sk_bm = 0 then []
          else
            let sp, _, _ = Sched.split ~axis:sym ~factor:sk_bm ~outer:LL.Grid ~inner:LL.Serial in
            pad_to ~axis:sym ~extent sk_bm @ [ sp ]
        else
          let sp, _, _ = Sched.split ~axis:sym ~factor:w ~outer:LL.Serial ~inner:LL.Workgroup in
          [ sp ])
  in
  let companions =
    match conv_companion_geometry ~opt ~fused:sk_epilogue ~annotate site with
    | Ok ops -> ops
    | Error why -> companion_coverage_unsupported ~tensorized:true why
  in
  let companions =
    let folded = shared_folded_zero ~p ~target:site.c_d opt in
    conv_zero_geometry ?folded site ~site_syms ~annotate @ companions
  in
  let outer_grid =
    List.map site.c_outer ~f:(fun (s, _) -> Sched.Retype { axis = s; ty = LL.Grid })
  in
  (* Pad-composition seeding (gh-ocannl-485): the conv seeds carry the intrinsic-tile pad multiples
     for the column ([sk_bn]) and reduction ([sk_bk]) extents, and — in the unblocked flavor — the
     row ([sk_tm]); a non-dividing row block pads to the block size. Both operand slices stage
     through zero-fringe cooperative tiles, so [Tensorize] masks the fragment transfers and
     discharges the reduction mask. *)
  let col_red_pads =
    (if sk_bn > 0 then pad_to ~axis:site.c_oc ~extent:site.c_noc sk_bn else [])
    @ if sk_bk > 0 then pad_to ~axis:site.c_red ~extent:site.c_nred sk_bk else []
  in
  if sk_bm > 0 then
    let pads = pad_to ~axis:site.c_row ~extent:site.c_nrow sk_bm @ col_red_pads in
    let sp_row, row_o, row_i =
      Sched.split ~axis:site.c_row ~factor:sk_bm ~outer:LL.Grid ~inner:LL.Serial
    in
    let current = conv_split_row_current site ~row_o ~row_i in
    let target =
      List.map site.c_outer ~f:fst @ [ row_o ] @ site.c_kernel @ [ row_i; site.c_oc; site.c_red ]
    in
    let tz, _lane = Sched.tensorize ~i:row_i ~j:site.c_oc ~k:site.c_red ~simd_width:w () in
    companions @ pads @ (outer_grid @ [ sp_row ]) @ reorder_swaps ~current ~target
    @ [ stage site.c_a [ row_i; site.c_red ]; stage site.c_b [ site.c_red; site.c_oc ]; tz ]
  else
    let pads =
      (if sk_tm > 0 then pad_to ~axis:site.c_row ~extent:site.c_nrow sk_tm else []) @ col_red_pads
    in
    let loop_syms =
      List.map site.c_outer ~f:fst @ site.c_kernel @ [ site.c_row; site.c_oc; site.c_red ]
    in
    let tz, _lane = Sched.tensorize ~i:site.c_row ~j:site.c_oc ~k:site.c_red ~simd_width:w () in
    companions @ pads @ outer_grid
    @ reorder_swaps ~current:site.c_loops ~target:loop_syms
    @ [ stage site.c_a [ site.c_row; site.c_red ]; stage site.c_b [ site.c_red; site.c_oc ]; tz ]

(* Building a sketch is a narrow phase seam of its own (gh-ocannl-536). An applicability
   precondition -- no matmul site, a companion nest whose geometry the family cannot cover -- is the
   same verdict as a [Schedule.apply] precondition: this candidate is not applicable, and the search
   is better off recording a decline. Escaping untyped they were unclassified and therefore FATAL
   under strict classification, so a single inapplicable GPU sketch family ended the whole search
   (reproducible on Metal with test/operations/autotune_fission_sketch before gh-ocannl-536). Each
   is raised typed at its own site ([refuse], [companion_coverage_unsupported], [zero_geometry]'s
   [Unsupported]) since gh-ocannl-1390, rather than by converting every [Invalid_argument] escaping
   the builder: an arbitrary exception escaping a builder stays fatal, as one escaping a transform
   does. *)
let family_schedule ~accum_prec ~p (opt : LL.optimized) : Sched.schedule =
  let sched, d =
    if p.sk_conv then
      match detect_conv opt.LL.llc with
      | None -> refuse "Autotune sketch: no convolution site detected"
      | Some site ->
          ( (if p.sk_gpu then gpu_conv_sketch_schedule ~opt site p
             else cpu_conv_sketch_schedule ~opt site p),
            site.c_d )
    else
      match detect_matmul_canonical opt with
      | None -> refuse "Autotune sketch: no matmul micro-kernel detected"
      | Some (site, prefix, opt) ->
          (* [opt] is now the code the site was detected on; [prefix] (the enabling interchange,
             empty for a site the plain matcher finds) takes the caller's code there. *)
          let sched =
            if p.sk_mma then
              if p.sk_gpu then gpu_mma_sketch_schedule ~opt site p
              else if p.sk_bk > 0 then cpu_mma_pack_sketch_schedule site p
              else cpu_mma_sketch_schedule site p
            else if p.sk_gpu then gpu_sketch_schedule ~accum_prec ~opt site p
            else cpu_sketch_schedule ~accum_prec site p
          in
          (prefix @ sched, site.m_d)
  in
  if p.sk_epilogue then
    (* [shared] is the fragment-site knob: only the GPU MMA sketches store through the contracted
       fragment; the block-tiling pipeline stores through [Privatize], where [Fuse_epilogue] rejects
       [shared] outright and the twin would fail for the wrong reason. *)
    sched @ [ Sched.Fuse_epilogue { target = d; shared = epilogue_shared p } ]
  else sched

(* {2 The coalesced layouts (gh-ocannl-1165, gh-ocannl-1279)}

   The q/k/v projections' site [d[b,s,h,e] += w[h,e,k] * x[b,s,k]] carries the heads as an interior
   batch loop ([m_bi]) right above the column role [m_j = e], so its column tile is at most one head
   wide: #915's bgrid-in flavor recovered the launch order of the heads-merged layout, not its tile
   width. Every access reads [h; e] as adjacent plain iterators over unpadded axes, so the pair is
   one loop to the compiler ([Sched.Coalesce]): the merged column role ranges over all the heads'
   columns, and a 64-wide tile spans two heads of useful columns. That is [Coalesce_columns].

   Attention's out projection [d[b,s,j] += w[j,h,e] * x[b,s,h,e]] has the same pair on the other
   side of the product: a contraction nest ([m_ko = [h]], [m_k = e]), whose k-tile the innermost
   extent caps at one head. Both operands read [(h, e)] as adjacent plain iterators and the
   accumulator reads neither, so [Coalesce (last m_ko, m_k)] makes the contraction one k loop of
   extent [h * e]. That is [Coalesce_contraction]. It leaves every output coordinate alone, so the
   zeroing nest is not coalesced: [d]'s cells are the same cells.

   A coalesced site is a different lowering, so it is prepared once, hermetically, and the family is
   enumerated over the prepared code: [coalesced] returns the structural prefix and the code it
   produces. The prefix is a function of the site alone. For the columns, [Coalesce (last m_bi,
   m_j)], preceded, where [d]'s whole-node [Zero_out] shares the routine, by its expansion with the
   zeroing nest's own last two loops coalesced alike, and where a covering per-cell zero nest of [d]
   shares it (the folded zero of a GPU fission segment, gh-ocannl-1175), by that nest's coalescing:
   either way the zeroing takes the merged chain's geometry as an ordinary companion
   ([companion_geometry]), where an uncoalesced one would keep the old chain. For the contraction,
   the one [Coalesce], prepared by [coalesce_reduction_pair] — the preparation the split-reduce
   sites of a two-loop reduction band share (gh-ocannl-1414), so a k-split of the merged loop
   composes with this prefix. Whether a prefix applies is the op's own verdict, probed: a per-head
   operand, a padded inner axis, a weight storing the pair apart ([w[h, j, e]]) or any other access
   shape declines there, and no coalesced seed exists. The prepared code is then RE-DETECTED, and a
   contraction prefix is kept only where the re-detected site is the expected one: the merged loop
   as [m_k], of extent the pair's product, with the pair removed from [m_ko] and the output roles
   unchanged. *)

(* gh-ocannl-1414: the reduction pair [(outer, inner)] of [opt] merged into one loop, as the merged
   loop's fresh symbol and the coalesced code (on a hermetic copy), or [None] where [Sched.Coalesce]
   declines -- the pair is not a perfect nest of two [Serial] loops, or some access does not read it
   as two adjacent plain iterators over unpadded axes. The op's own recognizer decides, as for the
   split. Shared by the split-reduce sites ({!Autotune.split_reduce_sites}) and the contraction
   layout ([coalesced], gh-ocannl-1279), so the two prepare one pair identically. *)
let coalesce_reduction_pair ?(static_indices = []) (opt : LL.optimized) ~outer ~inner :
    (Idx.symbol * LL.optimized) option =
  let op, merged = Sched.coalesce ~outer ~inner in
  match Sched.op_legality opt op with
  | Sched.Op_illegal _ | Sched.Op_unknown _ -> None
  | Sched.Op_legal -> (
      let hermetic =
        {
          opt with
          LL.traced_store = Hashtbl.copy opt.LL.traced_store;
          LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
        }
      in
      match Sched.apply_classified ~static_indices [ op ] hermetic with
      | coalesced -> Some (merged, coalesced)
      | exception Outcome.Cause_at _ -> None)

let coalesced ~(layout : coalesce_layout) (opt : LL.optimized) :
    (Sched.schedule * LL.optimized) option =
  match detect_matmul opt.LL.llc with
  | None -> None
  | Some site -> (
      match layout with
      | Coalesce_contraction -> (
          match List.last site.m_ko with
          | None -> None
          | Some (h, nh) -> (
              match coalesce_reduction_pair opt ~outer:h ~inner:site.m_k with
              | None -> None
              | Some (merged, prepared) -> (
                  let batch_equal =
                    List.equal (fun (s1, n1) (s2, n2) -> Idx.equal_symbol s1 s2 && n1 = n2)
                  in
                  match detect_matmul prepared.LL.llc with
                  | Some re
                    when Idx.equal_symbol re.m_k merged
                         && re.m_nk = nh * site.m_nk
                         && batch_equal re.m_ko (List.drop_last_exn site.m_ko)
                         && Idx.equal_symbol re.m_i site.m_i && Idx.equal_symbol re.m_j site.m_j
                         && batch_equal re.m_bo site.m_bo && batch_equal re.m_bi site.m_bi
                         && phys_equal re.m_d site.m_d ->
                      Some ([ Sched.Coalesce { outer = h; inner = site.m_k; merged } ], prepared)
                  | _ -> None)))
      | Coalesce_columns -> (
          match List.last site.m_bi with
          | None -> None
          | Some (h, _) -> (
              (* A per-cell zeroing nest's own last two loops (rank >= 3 here: an interior batch
                 axis sits between the row and column axes). *)
              let coalesce_last_two zsyms =
                match List.rev zsyms with
                | zj :: zh :: _ -> [ fst (Sched.coalesce ~outer:zh ~inner:zj) ]
                | _ -> []
              in
              let zero_ops =
                if site.m_zeroed then
                  let ez, zsyms = Sched.expand_zero ~tn:site.m_d in
                  ez :: coalesce_last_two zsyms
                else
                  (* The covering per-cell zero nest a GPU fission segment carries beside its site
                     (gh-ocannl-1175's folded zero): coalesced alike, or it would keep the
                     uncoalesced chain and refute the branch on companion coverage. *)
                  List.concat_map (LL.flat_lines [ opt.LL.llc ]) ~f:(fun stmt ->
                      match (LL.zero_initializer_target stmt, stmt) with
                      | Some tn, LL.For_loop _ when Ir.Tnode.equal tn site.m_d ->
                          coalesce_last_two (List.map (fst (serial_nest_of stmt)) ~f:fst)
                      | _ -> [])
              in
              let prefix = zero_ops @ [ fst (Sched.coalesce ~outer:h ~inner:site.m_j) ] in
              let hermetic =
                {
                  opt with
                  LL.traced_store = Hashtbl.copy opt.LL.traced_store;
                  LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
                }
              in
              match Sched.apply_classified prefix hermetic with
              | exception Outcome.Cause_at _ -> None
              | prepared -> Some (prefix, prepared))))

let coalesce_prefix ~layout (opt : LL.optimized) : Sched.schedule option =
  Option.map (coalesced ~layout opt) ~f:fst

let coalesce_layout_name = function
  | Coalesce_columns -> "interior batch loop coalesces into the column role (gh-ocannl-1165)"
  | Coalesce_contraction -> "two innermost contraction loops coalesce into one k (gh-ocannl-1279)"

(* gh-ocannl-1390: the builder's fault-injection seam, the sketch-level twin of
   [Schedule.with_transform_fault]: a test makes the builder raise what a builder bug raises, since
   a genuine one has no stable trigger. Per domain; the default is a no-op. *)
let builder_fault : (sketch_params -> unit) Stdlib.Domain.DLS.key =
  Stdlib.Domain.DLS.new_key (fun () _ -> ())

let with_sketch_builder_fault fault body =
  let previous = Stdlib.Domain.DLS.get builder_fault in
  Stdlib.Domain.DLS.set builder_fault fault;
  Exn.protect ~f:body ~finally:(fun () -> Stdlib.Domain.DLS.set builder_fault previous)

let sketch_schedule ~accum_prec ~p (opt : LL.optimized) : Sched.schedule =
  Stdlib.Domain.DLS.get builder_fault p;
  match p.sk_coalesce with
  | None -> family_schedule ~accum_prec ~p opt
  | Some layout -> (
      match coalesced ~layout opt with
      | None -> refuse ("Autotune sketch: no matmul site whose " ^ coalesce_layout_name layout)
      | Some (prefix, prepared) -> prefix @ family_schedule ~accum_prec ~p prepared)

(** {2 The matmul family tree's decisions}

    What a commitment on the matmul family tree {e is} (gh-ocannl-591). Every level's choices are
    values of {!Family_decision.t}, so a consumer that reads a decision back off a path — the
    certain-traffic floor {!sketch_path_traffic_floor}, the lattice lift {!lift_geometry_lattice},
    the tests, the enablement ranking to come — matches on data. The display strings are
    {!Family_decision.to_label}, a rendering of the datum and nothing else: rewording one moves what
    a log or a golden prints and cannot change what any consumer computes, which is the property the
    strings-plus-[Scanf] protocol did not have (a reworded geometry label silently made every scan
    arm fall through, zeroing the floor's increment on every path — a sound bound, so nothing
    raised).

    The level name ({!Family_decision.level}) is likewise derived from the decision, so a node's
    level and its children's identities cannot drift apart, and renaming a level is a rendering
    change too. *)

module Family_decision = struct
  type geometry = { g_bm : int; g_bn : int; g_bk : int; g_tm : int; g_tn : int }
  (** A tile geometry as committed by a [geometry] level. Which fields are meaningful is the
      constructor's business ({!geometry_choice}), so the zero-encodings that the shared shape
      carries are read only where they mean something:

      - [g_bm], [g_bk]: the row and depth blocks, always meaningful.
      - [g_bn]: the column block; [0] in {!Cpu_packed} encodes the unsplit full column extent, and
        in {!Gpu_mma} it is pinned at the mma lane width.
      - [g_tm], [g_tn]: the per-thread tile of {!Gpu_blocktile}; [0] elsewhere.

      [g_bk = 0] in {!Gpu_mma} is the unstaged full-K block — the one distinction the traffic floor
      turns on, since an unstaged geometry reads its operands in place and stages nothing. *)

  (** How a [geometry] level was committed. The five forms are the five curated menus (plus the
      lattice branch), and they are distinct constructors rather than one shape because what a
      completion below them stages differs: the GPU pipelines stage both operand tiles in workgroup
      memory, the CPU packed shapes pack panels whose traffic depends on the packing shape above,
      and the CPU blocktile stages nothing. *)
  type geometry_choice =
    | Gpu_blocktile of geometry
        (** The GPU scalar blocktile menu: both operand tiles are staged in kernel. *)
    | Gpu_mma of geometry
        (** The GPU tensorized menu: [g_bk > 0] stages both operand tiles in kernel, [g_bk = 0] is
            the unstaged full-K block. *)
    | Cpu_blocktile of int  (** The CPU blocktile menu's single block size (bm = bn = bk). *)
    | Cpu_packed of geometry
        (** The CPU packed-composition menu; what it costs depends on the {!Packing_shape} above it.
        *)
    | Lattice
        (** The staged tile-size lattice beyond the curated menu (gh-ocannl-514 phase 5), excluded
            by default and lifted by {!lift_geometry_lattice}. Its own axes commit as
            {!Lattice_box}. *)

  (** One committed decision of the matmul family tree. Each constructor belongs to exactly one
      level ({!level}) and carries the whole identity of the commitment — no consumer needs the
      level name, or the label, to know what was decided. *)
  type t =
    | Family of [ `Matmul | `Conv ]
        (** The combined view's root ({!sketch_family_tree}, gh-ocannl-1488): which family the
            site's own shape admits, the other refuted with the witness of why. A per-family tree
            ({!matmul_family_tree}, {!conv_family_tree}) carries no such level. *)
    | Fusion of [ `Unfused | `Fused ]
        (** A family's root: the epilogue-fusion flavor (gh-ocannl-613). *)
    | Pipeline of [ `Blocktile | `Tensorized ]  (** Which composed pipeline. *)
    | Batch of [ `Serial | `Grid | `Grid_inner ]
        (** The batch-geometry twin (gh-ocannl-643), GPU only; [`Grid_inner] binds the interior
            batch loops inside the row blocks (gh-ocannl-728, [sk_batch_inner]). *)
    | Packing of [ `In_kernel | `Hoisted ]
        (** The CPU blocktile pipeline's link-time packing twin (gh-ocannl-470). *)
    | Geometry of geometry_choice  (** The tile geometry, per the pipeline's own menu. *)
    | Lattice_box of { lb_axis : [ `Bm | `Bk ]; lb_lo : int; lb_hi : int }
        (** One binary interval refinement of a lattice axis: the value range still open below this
            commitment, [lb_lo = lb_hi] at a singleton. Boxes are priced at [lb_lo], their most
            favorable corner. *)
    | Twin of [ `Plain | `Swizzled | `Depth of int ]
        (** The per-staged-geometry twins: the swizzled staged layout, the pipelined depths. *)
    | Tensorized_form of [ `Whole_triple | `Packed ]  (** The CPU tensorized composition. *)
    | Row_block of int
        (** The CPU whole-triple row block; [0] is the unsplit form, [> 0] a pool-rendered Grid
            split. *)
    | Packing_shape of
        [ `Serial | `Hoisted | `Hoisted_grid | `Hoisted_grid_pack_rest | `Grid_pack_rest | `Grid ]
        (** Which CPU packed composition: where the panels are packed (in kernel, at link time, per
            Grid chunk) — what makes a packed geometry's traffic additional or merely relocated. *)
    | Register_tile of Ir.Register_tile.t option
        (** The CPU tensorized micro-kernel's C-tile geometry (gh-ocannl-619): [None] is the
            renderer's own ranking-model choice, [Some] a schedule-carried alternative the renderer
            honours exactly. The level appears only where {!Ir.Register_tile.alternatives} offers
            one, so a site with a single affordable width keeps its pre-level leaf list. *)
    | Conv_leg of [ `Cpu | `Gpu ]
        (** The conv family's route (gh-ocannl-697): the CPU register-tiled implicit-GEMM pipeline
            or the GPU staged one; the leg the backend kind does not take is refuted. *)
    | Conv_shape of [ `Serial | `Grid | `Row_block of int ]
        (** The CPU conv pipeline's flavor: serial, Grid-parallel, or the row split into panels of
            that many rows (gh-ocannl-500). *)
    | Conv_staging of [ `Plain | `Pipelined ]
        (** The GPU conv pipeline's staging: single-copy tiles, or rotating copies whose depth the
            [twin] level below each geometry commits (gh-ocannl-487). Above the geometry, so the
            leaves keep the flat list's order (every plain geometry, then the depth twins). *)
    | Conv_rows of int
        (** The GPU conv geometry: the GEMM row blocked into Grid blocks of that many rows, [0] the
            whole (intrinsic-tile padded) extent. Column and reduction stay whole-extent, padded to
            the intrinsic tile — a site-wide property, not a decision. *)

  type path = (string * t) list
  (** The path a consumer reads: {!Ir.Schedule_space}'s [(level, decision)] vector at this label
      type. The level string is display; the decision is the identity. *)

  (* The decisions are pure data — variants over ints — and the [autotune] library carries no ppx
     deriving, so structural [Poly] equality IS the intended equality here (no floats, no functions,
     no abstract payloads). *)

  (** Two decisions are the same commitment. *)
  let equal (a : t) (b : t) = Poly.equal a b

  (** Total order on decisions, for keying and sorting. *)
  let compare (a : t) (b : t) = Poly.compare a b

  (** The level a decision belongs to — the name {!Ir.Schedule_space.Choice} carries. Derived, so
      the tree cannot mint a node whose level disagrees with its children's decisions. *)
  let level = function
    | Family _ -> "family"
    | Fusion _ -> "fusion"
    | Pipeline _ -> "pipeline"
    | Batch _ -> "batch"
    | Packing _ -> "packing"
    | Geometry _ -> "geometry"
    | Lattice_box { lb_axis = `Bm; _ } -> "bm"
    | Lattice_box { lb_axis = `Bk; _ } -> "bk"
    | Twin _ -> "twin"
    | Tensorized_form _ -> "tensorized-form"
    | Row_block _ -> "row-block"
    | Packing_shape _ -> "packing-shape"
    | Register_tile _ -> "register-tile"
    | Conv_leg _ -> "conv-leg"
    | Conv_shape _ -> "conv-shape"
    | Conv_staging _ -> "conv-staging"
    | Conv_rows _ -> "conv-rows"

  (** The display rendering — for logs, decline reports and goldens. Nothing reads it back. *)
  let to_label =
    let geom { g_bm; g_bn; g_bk; g_tm; g_tn } =
      Printf.sprintf "bm%d bn%d bk%d tm%d tn%d" g_bm g_bn g_bk g_tm g_tn
    in
    let geom3 { g_bm; g_bn; g_bk; _ } = Printf.sprintf "bm%d bn%d bk%d" g_bm g_bn g_bk in
    function
    | Family `Matmul -> "matmul"
    | Family `Conv -> "conv"
    | Fusion `Unfused -> "unfused"
    | Fusion `Fused -> "fused"
    | Pipeline `Blocktile -> "blocktile"
    | Pipeline `Tensorized -> "tensorized"
    | Batch `Serial -> "batch-serial"
    | Batch `Grid -> "batch-grid"
    | Batch `Grid_inner -> "batch-grid-inner"
    | Packing `In_kernel -> "in-kernel"
    | Packing `Hoisted -> "hoisted"
    | Geometry (Gpu_blocktile g) -> geom g
    | Geometry (Gpu_mma g) -> geom3 g
    | Geometry (Cpu_blocktile b) -> Printf.sprintf "b%d" b
    | Geometry (Cpu_packed g) -> geom3 g
    | Geometry Lattice -> "lattice"
    | Lattice_box { lb_axis; lb_lo; lb_hi } ->
        let axis = match lb_axis with `Bm -> "bm" | `Bk -> "bk" in
        if lb_lo = lb_hi then Printf.sprintf "%s=%d" axis lb_lo
        else Printf.sprintf "%s %d..%d" axis lb_lo lb_hi
    | Twin `Plain -> "plain"
    | Twin `Swizzled -> "swizzled"
    | Twin (`Depth d) -> Printf.sprintf "depth%d" d
    | Tensorized_form `Whole_triple -> "whole-triple"
    | Tensorized_form `Packed -> "packed"
    | Row_block bm -> Printf.sprintf "bm%d" bm
    | Packing_shape `Serial -> "serial"
    | Packing_shape `Hoisted -> "hoisted"
    | Packing_shape `Hoisted_grid -> "hoisted-grid"
    | Packing_shape `Hoisted_grid_pack_rest -> "hoisted-grid-pack-rest"
    | Packing_shape `Grid_pack_rest -> "grid-pack-rest"
    | Packing_shape `Grid -> "grid"
    | Register_tile None -> "auto"
    | Register_tile (Some t) -> Ir.Register_tile.to_string t
    | Conv_leg `Cpu -> "cpu"
    | Conv_leg `Gpu -> "gpu"
    | Conv_shape `Serial -> "serial"
    | Conv_shape `Grid -> "grid"
    | Conv_shape (`Row_block bm) -> Printf.sprintf "bm%d" bm
    | Conv_staging `Plain -> "plain"
    | Conv_staging `Pipelined -> "pipelined"
    | Conv_rows 0 -> "rows-whole"
    | Conv_rows bm -> Printf.sprintf "rows-bm%d" bm

  (** A decision path as ["level=label > …"], for logs and reports. *)
  let render_path (path : path) = Sspace.render_path ~label:to_label path
end

type family_tree = (Family_decision.t, sketch_params) Sspace.tree
(** The matmul family's trees and children at the decision label type. *)

type family_child = (Family_decision.t, sketch_params) Sspace.child

(* Every [Choice] node of the family tree. The level name is derived from the children's decisions —
   all children of one node commit the same level — so a node's level and its children's identities
   cannot drift apart (gh-ocannl-591). *)
let decided_choice (children : (Family_decision.t * family_child) list) : family_tree =
  match children with
  | [] -> invalid_arg "Sketch_families.decided_choice: a decision level with no children"
  | (d, _) :: _ -> Sspace.Choice { level = Family_decision.level d; children }

(* The matmul family as a refinement tree (gh-ocannl-514 phase 1): the hand-written seed
   enumeration factored into decision levels — pipeline, then per-pipeline shape/geometry levels,
   twins as their own level — with the seed list recovered as the tree's {!Sspace.leaves}, in the
   exact order the flat enumeration produced (enumeration order reaches candidate timing order and
   dedup keep-first, so the factoring must preserve it; levels therefore appear in emission order,
   e.g. the packing shape ABOVE its geometries). Children domains depend on earlier commitments —
   a packing shape constrains which geometries remain, a twin exists only for staged geometries —
   which is what makes this a tree of staged choices rather than a product of independent domains.
   Subtrees are lazy so a future fathom (phase 4) can prune a choice without forcing what is below
   it; a choice whose every child was filtered out is an infeasible node with no leaves. Pinned by
   test/operations/sketch_family_tree.ml against the pre-factoring golden. *)
(* CPU Grid shapes render on the pool only when the configuration allows it: an explicit
   [cc_parallel_grid=none] or [cc_parallel_chunks=1] makes [C_syntax.collect_parallel_grid]
   deterministically collect nothing, so the candidate runs serially under a Grid label. Only the
   explicit settings are mirrored here — [auto] resolves through the backend's compiler probe and
   pool sizing, which stay render-settled (the same seeding-vs-builder boundary as companion
   coverage). *)
let cpu_grid_rendering_disabled =
  lazy
    (let mode =
       String.lowercase
         (String.strip (Utils.get_global_arg ~arg_name:"cc_parallel_grid" ~default:"auto"))
     in
     if String.equal mode "none" then
       Some
         "cc_parallel_grid=none: Grid loops render serially, the shape would time under a Grid \
          label"
     else
       match
         Int.of_string
           (String.strip (Utils.get_global_arg ~arg_name:"cc_parallel_chunks" ~default:"0"))
       with
       | 1 ->
           Some
             "cc_parallel_chunks=1: a single chunk renders serially, the shape would time under a \
              Grid label"
       | _ -> None
       | exception _ -> None)

(* gh-ocannl-514 phase 5: the witness marking the tile-size lattice branches beyond the curated
   geometry menus. [leaves] (the tuner's seed lists) never enumerate an [Excluded] branch, so the
   lattice exists in the space without changing what is timed; [lift_geometry_lattice] (config
   [model_default_geometry_lattice]) is the driver's lift. *)
let geometry_lattice_witness =
  "beyond the curated menu (search economy): the staged tile-size lattice; config \
   model_default_geometry_lattice lifts it for the model-argmin search"

(* Binary interval refinement over a sorted axis menu (gh-ocannl-514 phase 5): interior choices
   split the value range in half — boxes labelled "<axis> <lo>..<hi>", singletons "<axis>=<v>" — and
   [box_verdict lo hi] judges a box by its extreme corners before it is built (a cap monotone in the
   axis refutes the whole box from its most favorable corner: the issue's "a tile-size interval
   whose minimum footprint exceeds shared memory refutes the whole box"). Subtrees stay lazy, so a
   refuted (or, during search, fathomed) half is never expanded — the property that makes searching
   the full lattice logarithmic-effective rather than enumerative. *)
let interval_axis ~(axis : [ `Bm | `Bk ]) ~(values : int array)
    ~(box_verdict : int -> int -> string option) ~(singleton : int -> family_child) : family_tree =
  let label lo hi =
    Family_decision.Lattice_box { lb_axis = axis; lb_lo = values.(lo); lb_hi = values.(hi) }
  in
  let rec child lo hi =
    match box_verdict values.(lo) values.(hi) with
    | Some w -> Sspace.Refuted w
    | None when lo = hi -> singleton values.(lo)
    | None -> Sspace.Child (lazy (split lo hi))
  and split lo hi =
    let mid = (lo + hi) / 2 in
    decided_choice [ (label lo mid, child lo mid); (label (mid + 1) hi, child (mid + 1) hi) ]
  in
  match Array.length values with
  | 0 -> invalid_arg "Autotune.interval_axis: empty axis"
  | 1 -> decided_choice [ (label 0 0, child 0 0) ]
  | n -> split 0 (n - 1)

(* One epilogue-fusion flavor of the matmul family: the pipeline level and everything below it.
   [fused] selects the leaves' [sk_epilogue] and the flavor's own companion-coverage verdict
   ([coverage_witness], owned by {!matmul_family_tree} so that the flavors can share it), and the
   CPU tensorized flavors' ([cpu_companions], {!cpu_mma_companion_writes}). *)
let matmul_flavor_tree ?(coalesced : coalesce_layout option)
    ?(capabilities = Ir.Backend_intf.no_codegen_capabilities) ~is_gpu ~is_cpu
    ~(limits : Ir.Backend_intf.hardware_limits) ~(coverage_witness : string option Lazy.t)
    ~(cpu_companions : string option Lazy.t) ~(fused : bool) site : family_tree =
  (* Builder preconditions statically decidable at tree construction (gh-ocannl-577): rules the
     schedule builders settle identically for every tile completion refute here, with their
     witnesses, instead of failing candidate by candidate at build — the same lift phase 2 gave the
     hardware floors and configuration disables. The guard sits above the geometry menus and the
     tile lattice, so a refuted family never expands a lattice box (the gh-514 phase-6 finding: the
     lifted lattice priced spaces whose every member died at candidate build). The zero-expansion
     rule binds every pipeline; companion coverage only the GPU ones — but under [is_gpu] both
     pipeline branches are GPU pipelines, so one guard serves both preconditions. *)
  let precondition_witness =
    match zero_expansion_witness site with
    | Some _ as w -> w
    | None -> if is_gpu then Lazy.force coverage_witness else None
  in
  let precondition_guard child =
    match precondition_witness with Some w -> Sspace.Refuted w | None -> child
  in
  let divides c n = c <= n && n % c = 0 in
  (* Every leaf is judged against the device's launch caps before it becomes a candidate
     (gh-ocannl-709), through the same {!Schedule.launch_geometry_excess} the pre-driver gate
     consults — so the refutation carries the gate's own sentence, and a search records WHY a
     geometry is unreachable on this device instead of paying a compile to be told. One hook rather
     than a predicate per pipeline: a pipeline added to this family is filtered by construction, as
     long as [matmul_launch_geometry] can predict its geometry. *)
  let leaf p =
    match launch_geometry_refutation ~limits (matmul_launch_geometry site p) with
    | Some witness -> Sspace.Refuted witness
    | None -> Sspace.Child (lazy (Sspace.Leaf p))
  in
  let choice children = decided_choice children in
  (* [subt] defers subtree construction into the child's lazy: verdicts are decided at parent
     construction (fathoming needs them without expansion), subtrees are not. *)
  let subt t = Sspace.Child (lazy (t ())) in
  (* The first failing conjunct is the witness: a refutation names the one constraint whose
     violation already refutes every completion, not the whole gate. *)
  let refute_unless (conds : (bool * string) list) (ok : unit -> family_child) : family_child =
    match List.find conds ~f:(fun (c, _) -> not c) with
    | Some (_, witness) -> Sspace.Refuted witness
    | None -> ok ()
  in
  let ndiv what c ~into n =
    (divides c n, Printf.sprintf "%s=%d does not divide %s=%d" what c into n)
  in
  (* The k-extent gate names what it actually compares against ([k_extent_label]). *)
  let ndiv_k what c =
    (divides c site.m_nk, Printf.sprintf "%s=%d does not divide %s" what c (k_extent_label site))
  in
  (* The axes a site geometry blocks, for its pad-waste judgment ([pad_waste_witness]); a 0 block is
     an unsplit axis. *)
  let site_pad_axes ~bm ~bn ~bk =
    [ ("m", site.m_ni, bm); ("n", site.m_nj, bn); ("k", site.m_nk, bk) ]
  in
  let base_params =
    {
      sk_gpu = false;
      sk_mma = false;
      sk_simd = 0;
      sk_bm = 0;
      sk_bn = 0;
      sk_bk = 0;
      sk_tm = 0;
      sk_tn = 0;
      sk_hoist = false;
      sk_grid = false;
      sk_pack_rest = false;
      sk_conv = false;
      sk_epilogue = fused;
      sk_swizzle = None;
      sk_depth = 1;
      sk_batch_grid = false;
      sk_batch_inner = false;
      sk_coalesce = None;
      sk_pack_prec = None;
      sk_tile = None;
    }
  in
  (* Both GPU pipeline branches are parameterized by the batch-geometry flavor (gh-ocannl-643):
     [~batch] threads into their leaves' [sk_batch_grid] and [sk_batch_inner]. The CPU branches
     ignore it — they are only ever built with [`Serial] (see [with_batch_twins] at the pipeline
     level). *)
  (* The GPU scalar blocktile geometries [(bm, bn, bk, tm, tn)]. The family's own menu keeps its
     k-tile at 8 or 16: on an uncoalesced multi-axis contraction the tile is judged against the
     innermost extent alone, one head. The contraction-coalesced site (gh-ocannl-1279) contracts
     over one merged k, so its menu is a bounded K menu instead: depths 256 down to 32 — 256 being
     gpt2_mini's whole [h * e], one k-block per output tile — over three block shapes, deepest
     first: a 16x16 tile of 2x2 register tiles and a 32x32 tile of 4x4 ones (64 threads each), and
     a 64x64 tile of 4x4 ones (256 threads), the block the uncoalesced family's CUDA winner uses
     (gh-ocannl-1459). The usual judgments below bound it: the staged tiles' workgroup-memory floor
     (in f32, 16x16x256 needs 32 KiB, exactly Metal's threadgroup limit, 32x32x256 twice that; the
     64x64 block needs 32 KiB at depth 64, so CUDA's 48 KiB admits it at 64 and 32, HIP's 64 KiB at
     128 too, Metal's 32 KiB at 64 exactly), the thread cap, pad composition and pad waste, and the
     launch caps at the leaf. The shallow 8/16 depths are not repeated here: over a merged k they
     are the family's own tiles of the uncoalesced site, already seeded beside this branch. *)
  let coalesced_k_depths = [ 256; 128; 64; 32 ] in
  let blocktile_geometries () =
    match coalesced with
    | Some Coalesce_contraction ->
        List.concat_map coalesced_k_depths ~f:(fun bk ->
            [ (16, 16, bk, 2, 2); (32, 32, bk, 4, 4); (64, 64, bk, 4, 4) ])
    | Some Coalesce_columns | None ->
        [
          (64, 64, 8, 4, 4);
          (32, 32, 8, 4, 4);
          (16, 16, 8, 4, 4);
          (32, 32, 16, 2, 2);
          (16, 16, 8, 2, 2);
        ]
  in
  (* A K-menu depth whose half already covers the merged extent in one block only adds zero fringe
     (K = 120 tiled 256 deep is more than half padding; 128 holds it): the shallower depth is in the
     menu. A depth just above K stays — K = 60 tiled 64 deep is one block with a 4-slot pad. The
     family's own menu never meets this: its depths are at most 16, and smaller extents pad there by
     design (gh-ocannl-730). *)
  let k_menu_gate bk =
    match coalesced with
    | Some Coalesce_contraction ->
        [
          ( bk / 2 < site.m_nk,
            Printf.sprintf
              "bk=%d: half that depth already covers the coalesced contraction extent %s in one \
               block (a deeper tile only pads)"
              bk (k_extent_label site) );
        ]
    | Some Coalesce_columns | None -> []
  in
  let blocktile_child ~(batch : batch_layout) =
    if is_gpu then
      let a_prec = Lazy.force site.m_a.Ir.Tnode.storage_prec in
      let b_prec = Lazy.force site.m_b.Ir.Tnode.storage_prec in
      subt (fun () ->
          (* The menu's pad waste is judged relative to its least-padded proposed geometry
             (gh-ocannl-740, [judge_pad_waste]); a dividing geometry has ratio 1. *)
          choice
          @@ judge_pad_waste
               (List.map (blocktile_geometries ()) ~f:(fun (bm, bn, bk, tm, tn) ->
                    ( Family_decision.Geometry
                        (Gpu_blocktile { g_bm = bm; g_bn = bn; g_bk = bk; g_tm = tm; g_tn = tn }),
                      site_pad_axes ~bm ~bn ~bk,
                      refute_unless
                        ((* Pad composition (gh-ocannl-730): [gpu_sketch_schedule] stages both
                            operands through zero-fringe workgroup tiles at every geometry, so the
                            block extents pad rather than gate — the tensorized family's
                            gh-ocannl-485 argument, measured on this pipeline. The REGISTER split is
                            not padded: [tm]/[tn] still divide their block tiles. *)
                         (if pad_composition_ok ~n_staged:(if bk > 0 then 2 else 0) ~n_operands:2
                          then []
                          else
                            [
                              ndiv "bm" bm ~into:"m" site.m_ni;
                              ndiv "bn" bn ~into:"n" site.m_nj;
                              ndiv_k "bk" bk;
                            ])
                        @ [ ndiv "tm" tm ~into:"bm" bm; ndiv "tn" tn ~into:"bn" bn ]
                        @ k_menu_gate bk
                        (* The launch size is statically known — two Workgroup dimensions of [bm/tm]
                           and [bn/tn] threads — so a known thread cap refutes pre-compile what
                           [Schedule.check_hardware_limits_classified] would reject per candidate;
                           likewise the two [shared] operand stages' workgroup-memory floor. *)
                        @ (match limits.Ir.Backend_intf.max_threads_per_workgroup with
                          | Some cap when tm > 0 && tn > 0 && bm / tm * (bn / tn) > cap ->
                              [
                                ( false,
                                  Printf.sprintf
                                    "block tile launches %d threads per workgroup (bm/tm * bn/tn), \
                                     exceeding the %d-thread limit"
                                    (bm / tm * (bn / tn))
                                    cap );
                              ]
                          | _ -> [])
                        @
                        (* Charged as the gate charges them, each tile rounded to the backend's
                           allocation granule (gh-ocannl-1330); no tile-MMA scope here. *)
                        let staged =
                          Sched.kernel_workgroup_bytes ~capabilities
                            ~arrays:
                              [
                                bm * bk * Ir.Ops.prec_in_bytes a_prec;
                                bk * bn * Ir.Ops.prec_in_bytes b_prec;
                              ]
                            ~mma_triples:[]
                        in
                        match limits.Ir.Backend_intf.max_workgroup_memory_bytes with
                        | Some cap when staged > cap ->
                            [
                              ( false,
                                Printf.sprintf
                                  "staged operand tiles need %d bytes of workgroup memory, \
                                   exceeding the %d-byte limit"
                                  staged cap );
                            ]
                        | _ -> [])
                        (fun () ->
                          leaf
                            {
                              base_params with
                              sk_gpu = true;
                              sk_bm = bm;
                              sk_bn = bn;
                              sk_bk = bk;
                              sk_tm = tm;
                              sk_tn = tn;
                              sk_batch_grid = batch_grid_of batch;
                              sk_batch_inner = batch_inner_of batch;
                            }) ))))
    else if is_cpu then
      (* Hoisted vs in-kernel packing stays a measured choice (gh-ocannl-470): when a constant
         operand can be packed at link time, propose each tiling in both flavors. The packing level
         sits ABOVE its geometries: the flat enumeration emitted all in-kernel tilings before the
         hoisted twins.

         The divisibility gate is kept by MEASUREMENT, not by necessity (gh-ocannl-741). Both
         operands are staged through zero-fringe tiles ([shared = false] changes only where the tile
         lives), so [pad_composition_ok] holds and a [pad_to] triple would be sound: padded
         candidates match the serial reference bitwise and cost no more than their dividing control.
         But on cc the family loses to the CPU tensorized family even where it divides (attention's
         out projection at head_dim 16: 18.0 ms vs 13.5 ms on an M-series Mac, 21.2 ms vs 17.3 ms on
         AVX2), and so at the sites padding would add too (head_dim 12, padded: 16.1 vs 13.1 ms and
         20.8 vs 16.9 ms), so the padded leaves would only lengthen the search. Revisit if this
         family starts winning where it divides. *)
      let geoms hoist =
        choice
          (List.map [ 16; 8 ] ~f:(fun b ->
               ( Family_decision.Geometry (Cpu_blocktile b),
                 refute_unless
                   [ ndiv "b" b ~into:"m" site.m_ni; ndiv "b" b ~into:"n" site.m_nj; ndiv_k "b" b ]
                   (fun () ->
                     leaf { base_params with sk_bm = b; sk_bn = b; sk_bk = b; sk_hoist = hoist }) )))
      in
      subt (fun () ->
          choice
            [
              (Family_decision.Packing `In_kernel, subt (fun () -> geoms false));
              ( Family_decision.Packing `Hoisted,
                if site_hoistable site site.m_a || site_hoistable site site.m_b then
                  subt (fun () -> geoms true)
                else
                  Sspace.Refuted
                    (if hoistable site.m_a then strided_row_link_time_witness site
                     else
                       "hoisted packing needs a host-init-backed constant operand; neither operand \
                        is one") );
            ])
    else Sspace.Refuted "backend kind seeds no scalar blocktile pipeline"
  in
  let mma_child ~(batch : batch_layout) =
    let a_prec = Lazy.force site.m_a.Ir.Tnode.storage_prec in
    let b_prec = Lazy.force site.m_b.Ir.Tnode.storage_prec in
    let d_prec = Lazy.force site.m_d.Ir.Tnode.storage_prec in
    match
      ( tensorized_capability_refutation ~is_gpu ~is_cpu ~limits ~a_prec ~b_prec ~d_prec,
        is_gpu,
        limits.Ir.Backend_intf.mma )
    with
    | Some witness, _, _ -> Sspace.Refuted witness
    | None, true, Some ({ Ir.Backend_intf.mma_simd_width = w; _ } as mma) -> (
        match mma_tile_for_precisions mma ~a_prec ~b_prec ~d_prec with
        | None ->
            (* [tensorized_capability_refutation] refuted this triple above. *)
            assert false
        | Some (tm_t, tn_t, tk_t) ->
            (* [bn = w] keeps the zeroing's column grid blocks aligned with [j]'s (see
               [gpu_mma_sketch_schedule]); [bk = 0] = unstaged full-K block. Staged seeds
               ([bk > 0]) no longer require the extents to be block multiples: the builder pads the
               non-multiple axes and [Tensorize] masks the edges (gh-ocannl-485) — block sizes must
               still be intrinsic-tile multiples. Unstaged seeds read the operands in place, so a
               pad cannot be absorbed and the full divisibility gates remain. *)
            (* The swizzled layout the emission can read for these formats, if any, and the tile
               extents it needs (gh-ocannl-481 item 3, D3): [Swizzle_b128] permutes whole 16-byte
               units, so each staged tile's minor extent — [bk] elements of A, [bn] of B — must span
               a power-of-two count > 1 of them. Judged here rather than left to raise inside
               [Schedule.apply]: an inapplicable twin is refuted with its witness, not merely
               failed. *)
            let staged_layout = mma_staged_layout_for_precisions mma ~a_prec ~b_prec ~d_prec in
            let scope_of_bk bk = matmul_mma_scope site ~bk in
            let scope_name = function
              | Ir.Backend_intf.Mma_per_statement -> "per-statement"
              | Ir.Backend_intf.Mma_fragment_scope -> "persistent-fragment"
            in
            let wide_scope_ok scope = not (wide_acc_withholds mma ~scope ~d_prec) in
            let wide_scope_witness scope =
              Printf.sprintf
                "%s accumulator in the %s emission scope, which the backend does not advertise"
                (Option.value ~default:"The wide policy requires a wide"
                   (wide_acc_withholding mma ~scope ~d_prec))
                (scope_name scope)
            in
            let b128_units_ok prec extent =
              let bytes = extent * Ir.Ops.prec_in_bytes prec in
              bytes % 16 = 0
              &&
              let units = bytes / 16 in
              units > 1 && units land (units - 1) = 0
            in
            let nmul what c ~of_ n =
              ( c % n = 0,
                Printf.sprintf "%s=%d is not a multiple of the intrinsic tile %s=%d" what c of_ n )
            in
            (* A sound workgroup-memory floor for staged geometries: any completion allocates at
               least the cooperative operand tiles ([bm x bk] of A, [bk x bn] of B), [depth]-fold
               under software pipelining — other shared allocations only add. Exceeding the
               advertised limit refutes every completion below the child pre-compile, where
               [Schedule.check_hardware_limits_classified] would otherwise reject it one candidate
               compile at a time. [None]/unknown limit refutes nothing. *)
            let staged_tiles_exceed ?(masked_fragment = true) ~bm ~bn ~bk ~depth () =
              match limits.Ir.Backend_intf.max_workgroup_memory_bytes with
              | Some cap when bk > 0 ->
                  (* The gate's own charge ([Sched.kernel_workgroup_bytes]): each operand tile one
                     array of [depth] copies, rounded to the backend's allocation granule, plus the
                     per-kernel scratch of the site's triple, which every completion's [Tile_mma]
                     carries — Metal's coordinate table where its boundary converts (gh-ocannl-1330,
                     gh-ocannl-1255). *)
                  (* A geometry padding the row or column role pad-masks the accumulator, and on
                     a staged pipeline [Tensorize] places that masked fragment, a [bm x bn] tile at
                     [d]'s storage precision, in workgroup-shared memory beside the operand tiles
                     (gh-ocannl-1459: the coalesced K menu's deep tiles reach the limit with it,
                     32x32x128 f32 on Metal's 32 KiB). A dividing geometry has no mask, so no
                     fragment is charged: the floor stays a floor. A box of row blocks spanning
                     dividing and padding members is judged without it ([~masked_fragment:false]):
                     the charge is not monotone in [bm], so its corner would not bound the box. *)
                  let masked_fragment =
                    if (not masked_fragment) || (divides bm site.m_ni && divides bn site.m_nj) then
                      []
                    else [ bm * bn * Ir.Ops.prec_in_bytes d_prec ]
                  in
                  let charge ~mma_triples =
                    Sched.kernel_workgroup_bytes ~capabilities
                      ~arrays:
                        ([
                           depth * bm * bk * Ir.Ops.prec_in_bytes a_prec;
                           depth * bk * bn * Ir.Ops.prec_in_bytes b_prec;
                         ]
                        @ masked_fragment)
                      ~mma_triples
                  in
                  let bytes = charge ~mma_triples:[ (d_prec, a_prec, b_prec) ] in
                  let scratch = bytes - charge ~mma_triples:[] in
                  if bytes > cap then
                    Some
                      (Printf.sprintf
                         "staged operand tiles need %d bytes of workgroup memory%s%s%s, exceeding \
                          the %d-byte limit"
                         bytes
                         (if depth > 1 then Printf.sprintf " at pipeline depth %d" depth else "")
                         (if scratch > 0 then
                            Printf.sprintf " (%d of them tile-MMA scratch)" scratch
                          else "")
                         (match masked_fragment with
                         | [ f ] ->
                             Printf.sprintf " (with the %d-byte pad-masked accumulator fragment)" f
                         | _ -> "")
                         cap)
                  else None
              | _ -> None
            in
            (* gh-ocannl-514 phase 5: the staged tile-size lattice beyond the curated menu — every
               intrinsic-tile multiple of bm crossed with every staged (> 0) multiple of bk, bn
               pinned at the lane width like the curated staged seeds. Boxes are judged at their
               most favorable corner (smallest tiles) against the workgroup-memory floor; the
               certain staging traffic of a box's completions makes the search bound non-uniform
               across the family ([sketch_path_traffic_floor]). Excluded so the tuner's seed list
               ([leaves]) stays the curated menu; the model-argmin driver lifts it under config
               [model_default_geometry_lattice]. Curated staged pairs reappear as lattice singletons
               — the search may re-score them, it never re-times anything. *)
            (* The curated geometries [(bm, bn, bk)]; [bk = 0] is the unstaged full-K block. On the
               contraction-coalesced site (gh-ocannl-1459) the menu is a staged K menu like the
               scalar one ([blocktile_geometries]): depths 256 down to 32 over the two row blocks,
               deepest first, bounded by the same judgments (the staged-tile floor, [k_menu_gate],
               pad composition and waste). No unstaged geometry: it would read the operands in place
               through their flattened [Sub_axis] address, which [Tensorize] declines; staging
               normalizes them into dense cooperative tiles first, and the accumulator is the site's
               own, uncoalesced. *)
            let mma_geometries () =
              match coalesced with
              | Some Coalesce_contraction ->
                  List.concat_map coalesced_k_depths ~f:(fun bk -> [ (16, w, bk); (32, w, bk) ])
              | Some Coalesce_columns | None ->
                  [ (16, w, 0); (32, w, 0); (16, w, 32); (32, w, 32); (32, w, 16) ]
            in
            let lattice_child =
              if site.m_ni / tm_t = 0 || site.m_nk / tk_t = 0 then
                Sspace.Refuted
                  (Printf.sprintf
                     "no staged lattice: m=%d or %s is below one intrinsic tile (%dx%d)" site.m_ni
                     (k_extent_label site) tm_t tk_t)
              else if w % tn_t <> 0 then
                (* The lattice pins bn at the lane width like the curated staged seeds, so the
                   intrinsic column tile must divide it — the curated menu checks this per entry
                   ([nmul "bn"]); here it refutes the whole branch. *)
                Sspace.Refuted
                  (Printf.sprintf
                     "lattice pins bn at the lane width %d, which is not a multiple of the \
                      intrinsic tile n=%d"
                     w tn_t)
              else
                Sspace.Excluded
                  ( geometry_lattice_witness,
                    lazy
                      (Sspace.Child
                         (lazy
                           (* The axis menus materialize only here, inside the exclusion's lazy
                              payload: their lengths are m and k over the intrinsic tile, so an
                              un-lifted tree — every ordinary autotuning run — must not pay for them
                              (Codex P2 on PR #327). *)
                           (let mults t ~upto = List.init (upto / t) ~f:(fun i -> (i + 1) * t) in
                            let bms = Array.of_list (mults tm_t ~upto:site.m_ni) in
                            let bks = Array.of_list (mults tk_t ~upto:site.m_nk) in
                            interval_axis ~axis:`Bm ~values:bms
                              ~box_verdict:(fun bm_lo bm_hi ->
                                staged_tiles_exceed ~masked_fragment:(bm_lo = bm_hi) ~bm:bm_lo ~bn:w
                                  ~bk:bks.(0) ~depth:1 ())
                              ~singleton:(fun bm ->
                                Sspace.Child
                                  (lazy
                                    (interval_axis ~axis:`Bk ~values:bks
                                       ~box_verdict:(fun bk_lo _bk_hi ->
                                         staged_tiles_exceed ~bm ~bn:w ~bk:bk_lo ~depth:1 ())
                                       ~singleton:(fun bk ->
                                         if not (wide_scope_ok (scope_of_bk bk)) then
                                           Sspace.Refuted (wide_scope_witness (scope_of_bk bk))
                                         else
                                           leaf
                                             {
                                               base_params with
                                               sk_gpu = true;
                                               sk_mma = true;
                                               sk_simd = w;
                                               sk_bm = bm;
                                               sk_bn = w;
                                               sk_bk = bk;
                                               sk_batch_grid = batch_grid_of batch;
                                               sk_batch_inner = batch_inner_of batch;
                                             }))))))) )
            in
            subt (fun () ->
                choice
                  ((* The curated menu's pad waste is judged relative to its least-padded proposed
                      geometry (gh-ocannl-740, [judge_pad_waste]), above the twins: a geometry not
                      worth one compile is not worth its twins'. An unstaged geometry divides by the
                      gates below, so its ratio is 1. The lattice is already excluded wholesale. *)
                   judge_pad_waste
                     (List.map (mma_geometries ()) ~f:(fun (bm, bn, bk) ->
                          ( Family_decision.Geometry
                              (Gpu_mma { g_bm = bm; g_bn = bn; g_bk = bk; g_tm = 0; g_tn = 0 }),
                            site_pad_axes ~bm ~bn ~bk,
                            refute_unless
                              ([
                                 nmul "bm" bm ~of_:"m" tm_t;
                                 nmul "bn" bn ~of_:"n" tn_t;
                                 ( wide_scope_ok (scope_of_bk bk),
                                   wide_scope_witness (scope_of_bk bk) );
                               ]
                              @ (match staged_tiles_exceed ~bm ~bn ~bk ~depth:1 () with
                                | Some w -> [ (false, w) ]
                                | None -> [])
                              @
                              if bk = 0 then
                                [
                                  ( site.m_row_stride = 1,
                                    strided_row_in_place_witness site
                                      ~what:"an unstaged full-K block" );
                                  ndiv "bm" bm ~into:"m" site.m_ni;
                                  ndiv "bn" bn ~into:"n" site.m_nj;
                                  ( site.m_nk % tk_t = 0,
                                    Printf.sprintf
                                      "unstaged full-K block: %s is not a multiple of the \
                                       intrinsic tile k=%d"
                                      (k_extent_label site) tk_t );
                                ]
                              else nmul "bk" bk ~of_:"k" tk_t :: k_menu_gate bk)
                              (fun () ->
                                let base =
                                  {
                                    base_params with
                                    sk_gpu = true;
                                    sk_mma = true;
                                    sk_simd = w;
                                    sk_bm = bm;
                                    sk_bn = bn;
                                    sk_bk = bk;
                                    sk_batch_grid = batch_grid_of batch;
                                    sk_batch_inner = batch_inner_of batch;
                                  }
                                in
                                (* The twins level (per staged geometry): the swizzled layout and
                                   the pipelined depths, each measured against the shared plain
                                   sibling — see the field docs on [sk_swizzle]/[sk_depth]. Unstaged
                                   geometries have no cooperative copy, so the twin choices do not
                                   arise at all; ineligible staged twins are refuted (emission
                                   constraint) or excluded (measured-cost policy) with their
                                   witnesses (gh-ocannl-481 item 3 D3; Codex P2 on PRs #303 and
                                   #317). *)
                                let swizzle_twins =
                                  match staged_layout with
                                  | None -> []
                                  | Some LL.Swizzle_elem -> []
                                  | Some LL.Swizzle_b128 when bk = 0 -> []
                                  | Some LL.Swizzle_b128 ->
                                      [
                                        ( Family_decision.Twin `Swizzled,
                                          if b128_units_ok a_prec bk && b128_units_ok b_prec bn then
                                            leaf { base with sk_swizzle = Some LL.Swizzle_b128 }
                                          else
                                            Sspace.Refuted
                                              "Swizzle_b128 permutes whole 16-byte units: each \
                                               staged tile's minor extent must span a power-of-two \
                                               count > 1 of them" );
                                      ]
                                in
                                let depth_twins =
                                  if List.is_empty mma.Ir.Backend_intf.mma_pipeline_depths || bk = 0
                                  then []
                                  else
                                    List.map mma.Ir.Backend_intf.mma_pipeline_depths ~f:(fun d ->
                                        ( Family_decision.Twin (`Depth d),
                                          if d < 1 || d > 2 then
                                            (* The capability list is advisory; the implemented
                                               range is [Schedule.apply_stage]'s — the wait-all
                                               emission has single-step lookahead, deeper pipelines
                                               need commit_group/wait_group N. *)
                                            Sspace.Refuted
                                              (Printf.sprintf
                                                 "pipeline depth %d is outside the implemented \
                                                  range 1..2 (Schedule.apply_stage)"
                                                 d)
                                          else
                                            match staged_tiles_exceed ~bm ~bn ~bk ~depth:d () with
                                            | Some w ->
                                                (* Legality beats policy: the multiplied footprint
                                                   refutes before the measured-cost exclusions
                                                   apply. *)
                                                Sspace.Refuted w
                                            | None ->
                                                if
                                                  not
                                                    (divides bm site.m_ni && divides bn site.m_nj
                                                   && divides bk site.m_nk)
                                                then
                                                  Sspace.Excluded
                                                    ( "pad-masked site: Tensorize stays on the \
                                                       barrier-bracketed arm, so the twin could \
                                                       only pay the doubled shared-memory \
                                                       footprint (Codex P2 on PR #303)",
                                                      lazy (leaf { base with sk_depth = d }) )
                                                else if
                                                  Ir.Ops.prec_in_bytes a_prec < 4
                                                  || Ir.Ops.prec_in_bytes b_prec < 4
                                                then
                                                  Sspace.Excluded
                                                    ( "below the async arms' 4-byte element floor: \
                                                       only the synchronous form would render — \
                                                       the occupancy cost phase 1 measured, with \
                                                       no overlap to buy back (Codex P2 on PR \
                                                       #317)",
                                                      lazy (leaf { base with sk_depth = d }) )
                                                else leaf { base with sk_depth = d } ))
                                in
                                subt (fun () ->
                                    choice
                                      (((Family_decision.Twin `Plain, leaf base) :: swizzle_twins)
                                      @ depth_twins))) )))
                  @ [ (Family_decision.Geometry Lattice, lattice_child) ])))
    | None, _, _ when is_cpu ->
        (* The register-tiled [Tile_mma] rendering needs no MMA units (cc's [limits.mma] is a token
           1x1x1 capability). Statement rules the renderer checks per emission
           ([C_syntax.try_register_tile]) that are already decidable here judge the branch
           (gh-ocannl-479): a candidate that statically must render the scalar fallback refutes the
           family's tensorized promise — it would otherwise be timed, and possibly crowned and
           cached, under a tensorized label, making "the tensorized candidate lost"
           indistinguishable from "it never ran tensorized". Statically decidable: operand {e
           compute}-precision uniformity and vector-capability (gh-ocannl-575: resolved through the
           same [Numerics.cpu_compute_prec] the emission asks, so narrow storage computing in f32
           seeds, and pure-fp16 seeds exactly where the probe reports native arithmetic), the fused
           accumulation form, the micro-kernel column extent vs. the vector lane count, and
           transposed-B storage for renderings that read B {e in place} (a packing Stage normalizes
           the layout, so the packed flavors are exempt). What is only knowable at emission (address
           spaces, footprint interactions with other locals) is covered by the decline diagnostics
           and the [C_syntax.mma_census]. *)
        (* The capability-level rules (vector file, lanes, compute-precision uniformity, routine
           logging) were judged by [tensorized_capability_refutation] above; the site's own form and
           shape are judged here. *)
        let prec = cpu_register_tile_prec ~limits d_prec in
        (* [lanes] is the widest the register file offers, which is what the decline messages quote;
           whether a given extent can be tiled is [lanes_fit], which lets the renderer's per-extent
           halving ({!Ir.Backend_intf.simd_lanes_for}) through instead of gating every extent on the
           widest vector. *)
        let lanes = limits.Ir.Backend_intf.simd_vector.bytes / max 1 (Ir.Ops.prec_in_bytes prec) in
        let lanes_fit extent =
          Option.is_some
            (Ir.Backend_intf.simd_lanes_for ~vector_bytes:limits.Ir.Backend_intf.simd_vector.bytes
               ~elt_bytes:(max 1 (Ir.Ops.prec_in_bytes prec))
               ~extent)
        in
        let tb_in_place = Option.value site.m_tb ~default:false in
        (* The register-tile level (gh-ocannl-619): every CPU tensorized leaf is twinned with the
           geometries {!Ir.Register_tile.alternatives} proposes for ITS micro-kernel extents — the
           whole-triple's rows are [m] or the row block, the packed composition's are the packed
           tile — "auto" first, so a site with no alternative keeps its pre-level leaf. The
           renderer's default is not re-seeded under a label: [auto] IS that geometry, and a
           duplicate would time it twice. Under [autotune_register_tile_rm_twin] the level also
           carries {!Ir.Register_tile.rm_twin}, the two-row twin gh-ocannl-947 gates behind a
           measured win: off by default, since each twin is one more candidate per leaf. *)
        let rm_twin_seeded = Utils.registered_flag ~arg_name:"autotune_register_tile_rm_twin" in
        let leaf (p : sketch_params) =
          let m = if p.sk_bm = 0 then site.m_ni else p.sk_bm in
          let n = if p.sk_bn = 0 then site.m_nj else p.sk_bn in
          let file = limits.Ir.Backend_intf.simd_vector in
          let elt_bytes = max 1 (Ir.Ops.prec_in_bytes prec) in
          match
            Ir.Register_tile.alternatives ~file ~elt_bytes ~m ~n
            @
            if rm_twin_seeded then Option.to_list (Ir.Register_tile.rm_twin ~file ~elt_bytes ~m ~n)
            else []
          with
          | [] -> leaf p
          | alts ->
              subt (fun () ->
                  choice
                    ((Family_decision.Register_tile None, leaf p)
                    :: List.map alts ~f:(fun t ->
                        (Family_decision.Register_tile (Some t), leaf { p with sk_tile = Some t }))
                    ))
        in
        refute_unless
          [ (site.m_fma, "register tiling requires the fused accumulation form") ]
          (fun () ->
            let whole () =
              (* Whole-triple [Tile_mma] reads both operands in place over the full column extent:
                 the stored B orientation and [n = m_nj] reach the renderer as-is. *)
              choice
                (List.map [ 0; 64; 16 ] ~f:(fun bm ->
                     ( Family_decision.Row_block bm,
                       refute_unless
                         ([
                            ( bm = 0 || divides bm site.m_ni,
                              Printf.sprintf "bm=%d does not divide m=%d" bm site.m_ni );
                          ]
                         (* [bm > 0] splits the rows into pool-rendered Grid blocks — which needs at
                            least two of them, or the Grid loop renders serially under a Grid label,
                            like the packed shapes' guard. *)
                         @ (if bm > 0 && blocks_of site.m_ni bm < 2 then
                              [
                                ( false,
                                  Printf.sprintf
                                    "bm=%d gives %d row block(s); a Grid split needs at least 2" bm
                                    (blocks_of site.m_ni bm) );
                              ]
                            else [])
                         @
                         match (bm > 0, Lazy.force cpu_grid_rendering_disabled) with
                         | true, Some w -> [ (false, w) ]
                         | _ -> [])
                         (fun () -> leaf { base_params with sk_mma = true; sk_bm = bm }) )))
            in
            let whole_child =
              refute_unless
                ([
                   ( site.m_row_stride = 1,
                     strided_row_in_place_witness site ~what:"the whole-triple form" );
                   ( not tb_in_place,
                     "stored B is transposed: whole-triple reads B in place, which the register \
                      tiling statically declines" );
                   ( lanes_fit site.m_nj,
                     Printf.sprintf "column extent n=%d is below one vector of lanes (%d)" site.m_nj
                       lanes );
                 ]
                @
                (* Every [bm] of the form keeps the lane, so the verdict is the form's. *)
                match
                  cpu_mma_companion_witness ~what:"the whole-triple form's Workgroup lane"
                    (Lazy.force cpu_companions)
                with
                | Some w -> [ (false, w) ]
                | None -> [])
                (fun () ->
                  match site.m_tb with
                  | None ->
                      (* Per [m_tb]'s own contract, [None] means no role symbol occupies B's minor
                         axis — in-place reads inherit that layout and Tensorize's role check
                         rejects every one of them at candidate compile, deterministically. The
                         packed forms stay available: a packing Stage normalizes the layout. *)
                      Sspace.Refuted
                        "stored B has no role symbol on its minor axis: whole-triple reads B in \
                         place and cannot satisfy Tensorize's role check"
                  | Some _ -> subt (fun () -> whole ()))
            in
            (* Cache-blocked packed composition ([cpu_mma_pack_sketch_schedule]; [bk > 0] selects
               it): [bn = 0] = unsplit column panel. The packed tiles are function-scope stack
               arrays, so their combined footprint is capped — which is also roughly the L2
               residency the blocking aims for. Non-multiple extents no longer gate the packed
               composition (gh-ocannl-485): both operands pack through zero-fringe tiles, so the
               builder pads the axes to the block sizes and [Tensorize] masks the edges. Shapes that
               read an operand in place cannot absorb a pad, so each geometry carries whether the
               extents divide outright ([full_div]) and those shapes refute non-dividing
               geometries. *)
            let packed () =
              (* Packed tiles are minted at the COMPUTE precision (gh-ocannl-575: the widening folds
                 into the packing copy), so the footprint caps are judged in its element size — an
                 f32 panel over fp16 storage is twice the storage bytes, and a pure-fp16 panel on a
                 native target is half the f32 one. *)
              let prec_bytes = Ir.Ops.prec_in_bytes prec in
              let pack_prec =
                (* Per-operand storage precs may differ (only the compute precs are uniform); the
                   schedule builder normalizes back to [None] per [Stage] whose source already
                   stores at the compute precision. *)
                if
                  Ir.Ops.equal_prec prec (Lazy.force site.m_a.Ir.Tnode.storage_prec)
                  && Ir.Ops.equal_prec prec (Lazy.force site.m_b.Ir.Tnode.storage_prec)
                then None
                else Some prec
              in
              let tile_bytes_cap = 256 * 1024 in
              let menu =
                List.map
                  [ (64, 0, 64); (64, 0, 256); (128, 128, 128); (64, 128, 256); (16, 0, 16) ]
                  ~f:(fun (bm, bn, bk) ->
                    let bn_eff = if bn = 0 then site.m_nj else bn in
                    let tiles_bytes = ((bm * bk) + (bk * bn_eff)) * prec_bytes in
                    let verdict =
                      (* The packed micro-kernel's column extent is the B~ panel width — a legality
                         floor. The footprint threshold is search economy (roughly the L2 residency
                         the blocking aims for — an oversized tile still compiles, and a hoisted
                         panel is not even a stack array), so it excludes rather than refutes: a
                         driver may lift it. *)
                      if not (lanes_fit bn_eff) then
                        Some
                          (`Refute
                             (Printf.sprintf "B~ panel width %d is below one vector of lanes (%d)"
                                bn_eff lanes))
                      else if tiles_bytes > tile_bytes_cap then
                        Some
                          (`Exclude
                             (Printf.sprintf
                                "packed tiles (%d bytes) exceed the %d-byte stack/cache-economy \
                                 threshold (heuristic, not a compiler limit)"
                                tiles_bytes tile_bytes_cap))
                      else None
                    in
                    let full_div =
                      divides bm site.m_ni
                      && (bn = 0 || divides bn site.m_nj)
                      && divides bk site.m_nk
                    in
                    ( Family_decision.Geometry
                        (Cpu_packed { g_bm = bm; g_bn = bn; g_bk = bk; g_tm = 0; g_tn = 0 }),
                      {
                        base_params with
                        sk_mma = true;
                        sk_bm = bm;
                        sk_bn = bn;
                        sk_bk = bk;
                        sk_pack_prec = pack_prec;
                      },
                      verdict,
                      full_div,
                      tiles_bytes ))
              in
              (* What refutes every Grid shape whatever its geometry: a configuration that renders
                 Grid loops serially, or a companion the row-block loop would leave uncovered
                 ({!cpu_mma_companion_writes}). *)
              let grid_refutation =
                lazy
                  (match Lazy.force cpu_grid_rendering_disabled with
                  | Some _ as w -> w
                  | None ->
                      cpu_mma_companion_witness ~what:"the packed Grid shape's row-block loop"
                        (Lazy.force cpu_companions))
              in
              let grid_ok p = blocks_of site.m_ni p.sk_bm >= 2 in
              let too_few_blocks p =
                Printf.sprintf "bm=%d gives %d row block(s); a Grid split needs at least 2" p.sk_bm
                  (blocks_of site.m_ni p.sk_bm)
              in
              (* Per-chunk privatized-tile floor for the Grid shapes ([C_syntax]'s
                 [per_chunk_private_bytes_cap], config [cc_grid_private_bytes_cap]): a known
                 in-kernel tile exceeding the cap makes [parallel_grid_safe] decline the Grid
                 rendering — the candidate would run serially under a Grid label. Other per-chunk
                 locals can still trip the cap at render; passing here is necessary, not sufficient
                 (the census and decline diagnostics cover the rest). *)
              let chunk_cap =
                match
                  Int.of_string
                    (String.strip
                       (Utils.get_global_arg ~arg_name:"cc_grid_private_bytes_cap" ~default:"262144"))
                with
                | c when c > 0 -> c
                | _ -> 256 * 1024
                | exception _ -> 256 * 1024
              in
              let over_chunk_cap ~what bytes =
                if bytes > chunk_cap then
                  Some
                    (Printf.sprintf
                       "%s (%d bytes) exceeds the per-chunk privatization cap (%d, config \
                        cc_grid_private_bytes_cap)"
                       what bytes chunk_cap)
                else None
              in
              (* One packing shape's geometries: the shared menu judged per shape. The shape level
                 sits ABOVE the geometries, matching the flat enumeration's variant-major emission
                 order. *)
              let geoms ~f =
                (* The shape's own judgment comes first (legality beats policy): a shape that reads
                   an operand in place refutes a non-dividing geometry outright, and the ones that
                   stage both operands pad it. Then the pad waste, judged per shape relative to its
                   least-padded proposed geometry (gh-ocannl-740, [judge_pad_waste]); a geometry the
                   footprint threshold excludes is not proposed, so it neither sets that minimum nor
                   is judged against it. *)
                choice
                @@ judge_pad_waste
                     (List.map menu ~f:(fun (label, p, verdict, full_div, tiles_bytes) ->
                          ( label,
                            site_pad_axes ~bm:p.sk_bm ~bn:p.sk_bn ~bk:p.sk_bk,
                            match verdict with
                            | Some (`Refute w) -> Sspace.Refuted w
                            | Some (`Exclude w) ->
                                (* The payload re-judges the geometry under the shape with only the
                                   economy threshold lifted — it may still refute (block counts,
                                   per-chunk caps). *)
                                Sspace.Excluded (w, lazy (f p full_div tiles_bytes))
                            | None -> f p full_div tiles_bytes )))
              in
              let any_hoistable = site_hoistable site site.m_a || site_hoistable site site.m_b in
              let no_constant =
                if hoistable site.m_a then strided_row_link_time_witness site
                else "no host-init-backed constant operand to pack at link time"
              in
              (* See the flat enumeration's rationale, now attached to the shapes it judges: hoisted
                 packing (gh-ocannl-470) needs a constant operand; the hoisted-only Grid shape reads
                 non-hoistable operands in place (no pad absorption, and a transposed non-hoistable
                 B statically declines the register tiling); the mixed grid-outermost shape
                 (gh-ocannl-473) exists exactly when one operand is hoistable and the other is not;
                 grid-outermost per-chunk re-packing (gh-ocannl-475) is proposed only where no
                 hoistable operand leaves a one-dispatch alternative, and its tiles must fit the
                 renderer's per-chunk privatization cap. Grid shapes need at least two row blocks
                 (c_syntax.ml [collect_parallel_grid]). *)
              choice
                [
                  ( Family_decision.Packing_shape `Serial,
                    subt (fun () -> geoms ~f:(fun p _ _ -> leaf p)) );
                  ( Family_decision.Packing_shape `Hoisted,
                    if any_hoistable then
                      subt (fun () -> geoms ~f:(fun p _ _ -> leaf { p with sk_hoist = true }))
                    else Sspace.Refuted no_constant );
                  ( Family_decision.Packing_shape `Hoisted_grid,
                    if not any_hoistable then Sspace.Refuted no_constant
                    else if Option.is_some (Lazy.force grid_refutation) then
                      Sspace.Refuted (Option.value_exn (Lazy.force grid_refutation))
                    else if site.m_row_stride > 1 then
                      (* A strided-row site's A is never [site_hoistable], so this shape omits its
                         stage. *)
                      Sspace.Refuted
                        (strided_row_in_place_witness site ~what:"the hoisted-grid shape")
                    else if
                      (* A non-hoistable B is read in place by this shape (its stage is omitted), so
                         only the clean untransposed orientation survives: transposed B statically
                         declines the register tiling, and [m_tb = None] means no role symbol
                         occupies B's minor axis — Tensorize's role check rejects every in-place
                         read deterministically. *)
                      (not (site_hoistable site site.m_b))
                      && not (Option.value_map site.m_tb ~default:false ~f:not)
                    then
                      Sspace.Refuted
                        (match site.m_tb with
                        | Some true ->
                            "non-hoistable transposed B would be read in place, which the register \
                             tiling statically declines"
                        | _ ->
                            "stored B has no role symbol on its minor axis: the hoisted-grid shape \
                             reads non-hoistable B in place and cannot satisfy Tensorize's role \
                             check")
                    else
                      subt (fun () ->
                          geoms ~f:(fun p full_div _ ->
                              if
                                not
                                  (full_div
                                  || (site_hoistable site site.m_a && site_hoistable site site.m_b)
                                  )
                              then
                                Sspace.Refuted
                                  "extents do not divide the blocks: the non-hoistable operand is \
                                   read in place and cannot absorb the zero-fringe pad"
                              else if not (grid_ok p) then Sspace.Refuted (too_few_blocks p)
                              else leaf { p with sk_hoist = true; sk_grid = true })) );
                  ( Family_decision.Packing_shape `Hoisted_grid_pack_rest,
                    if not any_hoistable then Sspace.Refuted no_constant
                    else if Option.is_some (Lazy.force grid_refutation) then
                      Sspace.Refuted (Option.value_exn (Lazy.force grid_refutation))
                    else if site_hoistable site site.m_a && site_hoistable site site.m_b then
                      Sspace.Excluded
                        ( "both operands are hoistable: nothing is left to pack in-kernel, the \
                           shape degenerates to hoisted-grid",
                          lazy
                            (subt (fun () ->
                                 geoms ~f:(fun p _ _ ->
                                     if not (grid_ok p) then Sspace.Refuted (too_few_blocks p)
                                     else
                                       leaf
                                         {
                                           p with
                                           sk_hoist = true;
                                           sk_grid = true;
                                           sk_pack_rest = true;
                                         }))) )
                    else
                      subt (fun () ->
                          geoms ~f:(fun p _ _ ->
                              let bn_eff = if p.sk_bn = 0 then site.m_nj else p.sk_bn in
                              (* The non-hoistable operand's in-kernel packing Stage lands inside
                                 the Grid body and privatizes per chunk. *)
                              let rest_tile =
                                if site_hoistable site site.m_a then p.sk_bk * bn_eff * prec_bytes
                                else p.sk_bm * p.sk_bk * prec_bytes
                              in
                              if not (grid_ok p) then Sspace.Refuted (too_few_blocks p)
                              else
                                match over_chunk_cap ~what:"per-chunk packed tile" rest_tile with
                                | Some w -> Sspace.Refuted w
                                | None ->
                                    leaf
                                      {
                                        p with
                                        sk_hoist = true;
                                        sk_grid = true;
                                        sk_pack_rest = true;
                                      })) );
                  ( Family_decision.Packing_shape `Grid_pack_rest,
                    (* The builder hoists every hoistable source in the grid-outermost form
                       regardless of [sk_hoist], so only the non-hoistable tiles privatize per chunk
                       — the cap judges exactly those (on a no-hoistable site this is both tiles; on
                       a lifted one-hoistable site, just the in-kernel one). *)
                    let privatized_bytes p =
                      let bn_eff = if p.sk_bn = 0 then site.m_nj else p.sk_bn in
                      (if site_hoistable site site.m_a then 0 else p.sk_bm * p.sk_bk * prec_bytes)
                      + if site_hoistable site site.m_b then 0 else p.sk_bk * bn_eff * prec_bytes
                    in
                    let judged () =
                      geoms ~f:(fun p _ _ ->
                          if not (grid_ok p) then Sspace.Refuted (too_few_blocks p)
                          else
                            match
                              over_chunk_cap ~what:"per-chunk packed tiles" (privatized_bytes p)
                            with
                            | Some w -> Sspace.Refuted w
                            | None -> leaf { p with sk_grid = true; sk_pack_rest = true })
                    in
                    if Option.is_some (Lazy.force grid_refutation) then
                      Sspace.Refuted (Option.value_exn (Lazy.force grid_refutation))
                    else if any_hoistable then
                      Sspace.Excluded
                        ( "a hoistable operand exists: the hoisted shapes cover the one-dispatch \
                           role without per-chunk re-packing",
                          lazy (subt judged) )
                    else subt judged );
                  ( Family_decision.Packing_shape `Grid,
                    match Lazy.force grid_refutation with
                    | Some w -> Sspace.Refuted w
                    | None ->
                        subt (fun () ->
                            geoms ~f:(fun p _ _ ->
                                (* The per-row-block A~ tile privatizes per chunk; the read-only B~
                                   panel is shared and does not count against the cap. *)
                                let a_tile = p.sk_bm * p.sk_bk * prec_bytes in
                                if not (grid_ok p) then Sspace.Refuted (too_few_blocks p)
                                else
                                  match
                                    over_chunk_cap ~what:"per-chunk privatized A~ tile" a_tile
                                  with
                                  | Some w -> Sspace.Refuted w
                                  | None -> leaf { p with sk_grid = true })) );
                ]
            in
            subt (fun () ->
                choice
                  [
                    (Family_decision.Tensorized_form `Whole_triple, whole_child);
                    (Family_decision.Tensorized_form `Packed, subt (fun () -> packed ()));
                  ]))
    | None, _, _ ->
        (* [tensorized_capability_refutation] refutes every other backend kind above. *)
        assert false
  in
  (* The batch-geometry level (gh-ocannl-643), per GPU pipeline, ABOVE the geometry menus:
     "batch-serial" first, so an unbatched (or CPU) site's leaf list is byte-identical to the
     pre-level enumeration — the level only appears where there are batch loops to spread, and the
     grid twins follow all serial-batch geometries of their pipeline, like the other
     propose-both-measure levels. Whether the device can LAUNCH a twin's fold is not asked here: it
     is one dimension of the launch predicate every leaf passes through (gh-ocannl-709), so an
     over-cap fold refutes with a reason instead of vanishing with the level. The coverage witness
     is flavor-independent here ([companion_geometry]'s verdict never depends on the emitted
     geometry), so both flavors share the one precondition guard. *)
  let with_batch_twins mk =
    if is_gpu && batch_grid_twin_ok site then
      subt (fun () ->
          choice
            ([
               (Family_decision.Batch `Serial, mk ~batch:`Serial);
               (Family_decision.Batch `Grid, mk ~batch:`Grid);
             ]
            @
            (* gh-ocannl-728: the interior-batch launch order, only where there are interior batch
               loops to place — elsewhere it is the [`Grid] twin's schedule. *)
            if List.is_empty site.m_bi then []
            else [ (Family_decision.Batch `Grid_inner, mk ~batch:`Grid_inner) ]))
    else mk ~batch:`Serial
  in
  choice
    [
      (Family_decision.Pipeline `Blocktile, precondition_guard (with_batch_twins blocktile_child));
      (Family_decision.Pipeline `Tensorized, precondition_guard (with_batch_twins mma_child));
    ]

(* The matmul family: the epilogue-fusion level (gh-ocannl-613) above one {!matmul_flavor_tree} per
   flavor. Unfused first, so the leaves keep the seeds-then-twins order the flat enumeration
   established (candidate timing order, dedup keep-first), and the search's threshold tightens over
   the unfused leaves before the twins compete. The fused flavor is a construction-time verdict of
   the root: refuted with the fusion recognizer's own reason when the base code carries no fusable
   tail ([Sched.fuse_epilogue_witness] — the check runs on the base code, where the plain
   accumulation-nest site applies), so a site that mints no twins says why. Each flavor's pipelines
   are then judged under the flavor's own preconditions — the fused coverage verdict is
   flavor-indexed (gh-ocannl-577) but implied by the unfused one ([matmul_coverage_witness]), so it
   is derived from it and the alignment analysis runs twice only where the unfused flavor is
   refuted. Both flavors' subtrees stay lazy like every other level: nothing below the root is built
   before a consumer descends into it. *)
let matmul_family_tree ?coalesced ?capabilities ~is_gpu ~is_cpu
    ~(limits : Ir.Backend_intf.hardware_limits) ~(opt : LL.optimized) site : family_tree =
  let unfused_coverage = lazy (matmul_coverage_witness ~opt ~fused:false site) in
  let fused_coverage =
    lazy
      (match Lazy.force unfused_coverage with
      | None -> None
      | Some _ -> matmul_coverage_witness ~opt ~fused:true site)
  in
  let flavor ~fused ~coverage_witness =
    Sspace.Child
      (lazy
        (matmul_flavor_tree ?coalesced ?capabilities ~is_gpu ~is_cpu ~limits ~coverage_witness
           ~cpu_companions:(lazy (cpu_mma_companion_writes ~opt ~fused site))
           ~fused site))
  in
  decided_choice
    [
      (Family_decision.Fusion `Unfused, flavor ~fused:false ~coverage_witness:unfused_coverage);
      ( Family_decision.Fusion `Fused,
        match Sched.fuse_epilogue_witness ~target:site.m_d opt with
        | Some w -> Sspace.Refuted w
        | None -> flavor ~fused:true ~coverage_witness:fused_coverage );
    ]

let matmul_seed_params ?coalesced ?capabilities ~is_gpu ~is_cpu
    ~(limits : Ir.Backend_intf.hardware_limits) ~opt site : sketch_params list =
  Sspace.leaves (matmul_family_tree ?coalesced ?capabilities ~is_gpu ~is_cpu ~limits ~opt site)

(** {2 The convolution family tree (gh-ocannl-697)}

    The conv family as a refinement tree beside {!matmul_family_tree}, on the same decision type and
    under the same root: [fusion = unfused | fused], the fused child refuted with the fusion
    recognizer's reason where the site's output feeds no fusable tail. Each flavor is its own
    subtree, judged under its own preconditions, so a fused twin is no longer a flag flipped over
    the unfused seed list: it has a verdict of its own, and a refused fused flavor leaves its
    witness. Below the root, one level per decision the flat enumeration made implicitly:

    - [conv-leg = cpu | gpu]: the implicit-GEMM pipeline's two routes (gh-ocannl-493), the one that
      does not apply to the backend kind refuted. CPU first, the order the flat list had.
    - CPU: [conv-shape = serial | grid | bm8 | bm16 | bm32] — the serial pipeline, its Grid-parallel
      flavor, and the cache-blocked row panels (gh-ocannl-500).
    - GPU: [conv-staging = plain | pipelined] above [conv-rows = whole | bm8 | bm16 | bm32], the
      pipelined geometries then committing their [twin = depth<d>] (gh-ocannl-487). Staging sits
      ABOVE the geometry so the leaves keep the flat list's order for any advertised depth list:
      every plain geometry, then each geometry's depth twins in turn.

    Every leaf is judged against the device's launch caps through {!launch_geometry_refutation}
    (gh-ocannl-739), so an over-cap geometry is refuted with the gate's own sentence instead of
    being filtered out of the list. *)

let conv_offset_witness =
  "a conv axis carries a nonzero padding offset, at which the packing anchor would mispack"

(* The CPU conv leg's packing precision (gh-ocannl-575): the site's compute precision, resolved
   through the same [Numerics.cpu_compute_prec] the emission asks, carried as [sk_pack_prec] when
   either operand stores narrower — [cpu_conv_sketch_schedule]'s [Stage]s then mint that operand's
   tiles at it. Shared by the tree (which seeds it) and the traffic floor (which prices the tiles at
   it, gh-ocannl-1489), so the two cannot disagree on a staged tile's width. *)
let cpu_conv_pack_prec ~(limits : Ir.Backend_intf.hardware_limits) (site : conv_site) =
  let native_fp16 = limits.Ir.Backend_intf.native_fp16_arithmetic in
  let comp_prec p = Ir.Numerics.cpu_compute_prec ~native_fp16_arithmetic:native_fp16 p in
  let cprec = comp_prec (Lazy.force site.c_d.Ir.Tnode.storage_prec) in
  let a_prec = Lazy.force site.c_a.Ir.Tnode.storage_prec
  and b_prec = Lazy.force site.c_b.Ir.Tnode.storage_prec in
  if Ir.Ops.equal_prec cprec a_prec && Ir.Ops.equal_prec cprec b_prec then None else Some cprec

(* The bytes a staged tile of [source] holds per cell under [pack_prec] (the CPU leg's
   [sk_pack_prec]; [None] on the GPU leg, whose [Stage] declares no [tile_prec]): [Stage] mints the
   tile at [pack_prec] exactly where the source stores at another precision, else at the
   source's. *)
let staged_tile_elt_bytes ~pack_prec (source : Ir.Tnode.t) =
  let source_prec = Lazy.force source.Ir.Tnode.storage_prec in
  match pack_prec with
  | Some p when not (Ir.Ops.equal_prec p source_prec) -> Ir.Ops.prec_in_bytes p
  | _ -> Ir.Ops.prec_in_bytes source_prec

(* The conv family's verdicts on one site, decided once and shared by both fusion flavors: every
   gate below is flavor-independent, so each flavor's subtree reads the same judgments (forced on
   first use) and only its leaves' [sk_epilogue] differs. *)
let conv_family_tree ?(capabilities = Ir.Backend_intf.no_codegen_capabilities) ~is_gpu ~is_cpu
    ~(limits : Ir.Backend_intf.hardware_limits) ~(opt : LL.optimized) (site : conv_site) :
    family_tree =
  let prec = Lazy.force site.c_d.Ir.Tnode.storage_prec in
  let a_prec = Lazy.force site.c_a.Ir.Tnode.storage_prec
  and b_prec = Lazy.force site.c_b.Ir.Tnode.storage_prec in
  (* Strided rows are seeded (gh-ocannl-502): the row's tile part in the input access is the single
     term [stride*row] — the kernel-window symbol lands in the outer part, at the staging anchor —
     so the compacting [Stage] packs the window densely, which satisfies [Tensorize]'s
     unit-coefficient index discipline. Every axis must still be offset-free — which padded convs
     from the tensor front end are: the halo is part of the physically padded buffer, buffer indices
     absorb the shift, and [Stage]'s edge guards compare against the padded [Tn.dims] (pinned by the
     cvp leg of schedule_conv_gemm). The gate is defense-in-depth against hand-built [Low_level]
     sites with genuine offsets, where the packing anchor would mispack (Codex P1 on PR #168).
     Candidates are timed, not value-checked, so unsound seeds must not be proposed at all; both
     legs pack through the same [Stage] decomposition. *)
  let offset_free = List.for_all site.c_axes ~f:(fun cx -> cx.cx_offset = 0) in
  let n_real = List.length (conv_real_stmts site opt) in
  let aligned = lazy (Option.is_some (conv_aligned_grid site opt)) in
  (* The flavor-indexed builder preconditions (gh-ocannl-697): each fusion flavor judged by its own
     companion demand. The fused demand is a subset of the unfused one (it skips the tail), so an
     unfused pass implies the fused one and the fused query runs only where the unfused one fails,
     as in [matmul_family_tree]. *)
  let flavor_indexed judge =
    let unfused = lazy (judge ~fused:false) in
    let fused = lazy (match Lazy.force unfused with None -> None | Some _ -> judge ~fused:true) in
    fun ~fused:f -> Lazy.force (if f then fused else unfused)
  in
  let gpu_coverage = flavor_indexed (fun ~fused -> conv_coverage_witness ~opt ~fused site) in
  let direct_grid = flavor_indexed (fun ~fused -> conv_direct_grid_witness ~opt ~fused site) in
  let leaf p =
    match launch_geometry_refutation ~limits (conv_launch_geometry site p) with
    | Some witness -> Sspace.Refuted witness
    | None -> Sspace.Child (lazy (Sspace.Leaf p))
  in
  let subt t = Sspace.Child (lazy (t ())) in
  (* A guard names every failing conjunct, as the census has always reported them. *)
  let failing conds =
    List.filter_map conds ~f:(fun (holds, why) -> if holds then None else Some (Lazy.force why))
  in
  let guard conds ok =
    match failing conds with [] -> ok () | ws -> Sspace.Refuted (String.concat ~sep:"; " ws)
  in
  let base =
    {
      sk_gpu = false;
      sk_mma = true;
      sk_simd = 0;
      sk_bm = 0;
      sk_bn = 0;
      sk_bk = 0;
      sk_tm = 0;
      sk_tn = 0;
      sk_hoist = false;
      sk_grid = false;
      sk_pack_rest = false;
      sk_conv = true;
      sk_epilogue = false;
      sk_swizzle = None;
      sk_depth = 1;
      sk_batch_grid = false;
      sk_batch_inner = false;
      sk_coalesce = None;
      sk_pack_prec = None;
      sk_tile = None;
    }
  in
  (* The CPU leg: the register-tiled implicit-GEMM pipeline ([cpu_conv_sketch_schedule]). *)
  let cpu_leg ~fused =
    if not is_cpu then Sspace.Refuted "not a CPU target"
    else
      (* Compute-precision uniformity and vector-capability, resolved through the same
         [Numerics.cpu_compute_prec] the emission asks (gh-ocannl-575) — see the matmul family's CPU
         branch for the rationale. *)
      let native_fp16 = limits.Ir.Backend_intf.native_fp16_arithmetic in
      let comp_prec p = Ir.Numerics.cpu_compute_prec ~native_fp16_arithmetic:native_fp16 p in
      let cprec = comp_prec prec in
      let uniform_vec_capable =
        (match cprec with
          | Ir.Ops.Single_prec _ | Ir.Ops.Double_prec _ -> true
          | Ir.Ops.Half_prec _ -> native_fp16
          | _ -> false)
        && Ir.Ops.equal_prec (comp_prec a_prec) cprec
        && Ir.Ops.equal_prec (comp_prec b_prec) cprec
        (* The C-tile accumulates at the compute precision, so a divergent accumulator residency
           ([Fp16_wide] + [narrow_compute_f32 = false] on an f16 destination) is an emission decline
           — mirror it here or the candidate is timed under a tensorized label (gh-ocannl-680; Codex
           P1 round 1 on staging PR #477). *)
        && Ir.Ops.equal_prec
             (Ir.Numerics.cpu_accum_prec ~native_fp16_arithmetic:native_fp16 prec)
             cprec
      in
      (* The renderer fills the widest vector the out-channel extent allows, halving where it must
         ({!Ir.Backend_intf.simd_lanes_for}); seeding asks the same question, or it would withhold
         candidates the renderer would in fact tile. *)
      let lanes =
        Ir.Backend_intf.simd_lanes_for ~vector_bytes:limits.Ir.Backend_intf.simd_vector.bytes
          ~elt_bytes:(max 1 (Ir.Ops.prec_in_bytes cprec))
          ~extent:site.c_noc
      in
      guard
        [
          ( limits.Ir.Backend_intf.simd_vector.bytes >= 8,
            lazy
              (Printf.sprintf "the %d-byte vector file is under 8 bytes"
                 limits.Ir.Backend_intf.simd_vector.bytes) );
          ( Option.is_some lanes,
            lazy
              (Printf.sprintf "no SIMD lane count fits the out-channel extent %d at %s" site.c_noc
                 (Ir.Ops.prec_string cprec)) );
          ( uniform_vec_capable,
            lazy
              "the operands' and the accumulator's compute precisions are not one vector-capable \
               precision" );
          (site.c_fma, lazy "the accumulation is not a fused multiply-add");
          (offset_free, lazy conv_offset_witness);
        ]
        (fun () ->
          let base =
            { base with sk_epilogue = fused; sk_pack_prec = cpu_conv_pack_prec ~limits site }
          in
          (* The register-tile level (gh-ocannl-619 for the matmul family; gh-ocannl-697 here):
             every CPU conv leaf is twinned with the geometries {!Ir.Register_tile.alternatives}
             proposes for ITS micro-kernel extents — the GEMM row ([c_nrow], or the panel's [sk_bm]
             rows) by the out-channel column — at the register tiling's compute precision, "auto"
             (the renderer's own choice) first, so a site with no alternative keeps its pre-level
             leaf. Under [autotune_register_tile_rm_twin] the level also carries the two-row twin,
             as in the matmul family. *)
          let leaf =
            let rm_twin_seeded = Utils.registered_flag ~arg_name:"autotune_register_tile_rm_twin" in
            let file = limits.Ir.Backend_intf.simd_vector in
            let elt_bytes = max 1 (Ir.Ops.prec_in_bytes (cpu_register_tile_prec ~limits prec)) in
            fun (p : sketch_params) ->
              let m = if p.sk_bm = 0 then site.c_nrow else p.sk_bm and n = site.c_noc in
              match
                Ir.Register_tile.alternatives ~file ~elt_bytes ~m ~n
                @
                if rm_twin_seeded then
                  Option.to_list (Ir.Register_tile.rm_twin ~file ~elt_bytes ~m ~n)
                else []
              with
              | [] -> leaf p
              | alts ->
                  subt (fun () ->
                      decided_choice
                        ((Family_decision.Register_tile None, leaf p)
                        :: List.map alts ~f:(fun t ->
                            ( Family_decision.Register_tile (Some t),
                              leaf { p with sk_tile = Some t } ))))
          in
          let outer_ok = match site.c_outer with (_, n) :: _ -> n >= 2 | [] -> false in
          let no_alignment =
            Printf.sprintf
              "%d real statements and the default preset's aligned analysis gives no whole-segment \
               Grid"
              n_real
          in
          (* Grid flavors need every materialized write in the routine covered by the Grid axis
             ([validate_parallel]), and the conv pipeline's own [Retype] only annotates the conv
             nest: proposed when the conv statement is alone — or has exactly one companion
             statement, the would-be epilogue tail, whose fused twin relocates the tail write under
             the Grid loop. Segments with more companions (an aligned-merged segment, e.g. lenet's
             conv+bias/relu+pooling) are proposed when the default preset's aligned cross-nest
             analysis Grid-annotates the whole segment ([conv_aligned_grid]) — the pipeline then
             adopts that whole-segment geometry, and its fused twin omits the preset [Retype] on the
             tail nest the fusion consumes (fuse-before-annotate, gh-ocannl-501). *)
          (* At most one companion statement: the builder Grid-types the conv nest's own loop, so
             the flavor's companion demand must be empty ([conv_direct_grid_witness]). *)
          let direct p =
            match direct_grid ~fused with Some w -> Sspace.Refuted w | None -> leaf p
          in
          let grid =
            if not outer_ok then Sspace.Refuted "no outer output loop of extent at least 2"
            else if n_real <= 2 then direct { base with sk_grid = true }
            else if Lazy.force aligned then leaf { base with sk_grid = true }
            else Sspace.Refuted no_alignment
          in
          (* Cache-blocked row-panel flavors (gh-ocannl-500): the GEMM row split into panels of
             [sk_bm] rows ([cpu_conv_sketch_schedule]'s [sk_bm] leg), at least two (possibly padded,
             gh-ocannl-485) panels. A conv-alone segment [Grid]-parallelizes the panel loop, an
             aligned-merged segment adopts the whole-segment [Grid] geometry and blocks the row
             serially for cache residency. The whole-routine zeroed graph keeps the serial and Grid
             flavors: its [Zero_out] is in the routine, so the aligned analysis bails. *)
          (* The builder Grid-types the panel loop itself at two or fewer real statements, so
             there the flavor's own companion demand decides ([direct]): the fused flavor, whose
             tail the fusion relocates into the conv nest, passes where the unfused one is refuted
             (Codex P2, round 1 on staging#1310 -- the pre-tree gate asked for the aligned
             whole-segment geometry there, which the builder does not use, and so withheld the
             fused panels of an unaligned conv+tail segment). With more statements the builder
             adopts that aligned geometry, so it must exist. *)
          let row_block bm =
            if site.c_zeroed then
              Sspace.Refuted "the conv output's whole-node Zero_out is in this lowering (c_zeroed)"
            else if blocks_of site.c_nrow bm < 2 then
              Sspace.Refuted
                (Printf.sprintf "a %d-row panel gives fewer than two panels of the %d rows" bm
                   site.c_nrow)
            else if n_real <= 2 then direct { base with sk_bm = bm }
            else if Lazy.force aligned then leaf { base with sk_bm = bm }
            else Sspace.Refuted no_alignment
          in
          (* The shape menu's pad waste (gh-ocannl-1393, [judge_pad_waste]): a row panel that does
             not divide the row pads it, relative to the unsplit shapes' ratio of 1. It never fires
             at the default threshold — two or more panels keep the waste under 2x — but a lowered
             [autotune_pad_waste_max] judges the panels like any geometry menu. *)
          let row_axis bm = [ ("row", site.c_nrow, bm) ] in
          subt (fun () ->
              decided_choice
                (judge_pad_waste
                   ([
                      (Family_decision.Conv_shape `Serial, row_axis 0, leaf base);
                      (Family_decision.Conv_shape `Grid, row_axis 0, grid);
                    ]
                   @ List.map [ 8; 16; 32 ] ~f:(fun bm ->
                       (Family_decision.Conv_shape (`Row_block bm), row_axis bm, row_block bm))))))
  in
  (* The GPU leg: the staged implicit-GEMM pipeline ([gpu_conv_sketch_schedule]). *)
  let gpu_leg ~fused =
    match (is_gpu, limits.Ir.Backend_intf.mma) with
    | false, _ -> Sspace.Refuted "not a GPU target"
    | true, None -> Sspace.Refuted "the backend advertises no mma capability"
    | true, Some ({ Ir.Backend_intf.mma_simd_width = w; _ } as mma) -> (
        let scope = conv_mma_scope site in
        match mma_tile_for_precisions_in_scope mma ~scope ~a_prec ~b_prec ~d_prec:prec with
        | None ->
            Sspace.Refuted
              (match wide_acc_withholding mma ~scope ~d_prec:prec with
              | Some policy ->
                  policy
                  ^ " accumulator in the conv's emission scope, which the backend does not \
                     advertise"
              | None ->
                  Printf.sprintf
                    "the backend advertises no mma tile for operands (%s, %s) with accumulator %s"
                    (Ir.Ops.prec_string a_prec) (Ir.Ops.prec_string b_prec)
                    (Ir.Ops.prec_string prec))
        | Some (tm_t, tn_t, tk_t) ->
            (* The site's covering zero is the pipeline's to annotate, whichever form it arrives in:
               a GPU fission segment's folded per-cell companion, or — on a whole-routine lowering
               ([c_zeroed]) — the whole-node [Zero_out], which [conv_zero_geometry] expands into
               that same nest (gh-ocannl-503). The coverage query ([conv_companion_geometry])
               excludes it either way.

               Companion coverage (gh-ocannl-1406): the builder's own [companion_geometry] verdict,
               not a statement count, judged per fusion flavor ([gpu_coverage], gh-ocannl-697): an
               unfused flavor whose tail cannot follow the accumulation nest's chain is refuted
               here, while its fused flavor, whose demand skips that tail, keeps its geometries. *)
            guard
              [ (offset_free, lazy conv_offset_witness) ]
              (fun () ->
                match gpu_coverage ~fused with
                | Some w -> Sspace.Refuted w
                | None ->
                    (* Pad-composition seeding (gh-ocannl-485): non-multiple column, reduction and
                       row extents pad to the intrinsic tile (the seed carries the multiples in
                       [sk_bn]/[sk_bk]/[sk_tm], 0 = already a multiple) and [Tensorize] masks the
                       edges. The intrinsic-tile divisibility of the row is a PER-BLOCK property
                       (gh-ocannl-500): a blocked flavor's micro-kernel row is [sk_bm], a multiple
                       of the intrinsic row tile, while column and reduction stay whole-extent
                       tensorized (blocking them would exceed the three-[Grid]-slot budget on 2-D
                       convs). *)
                    let noc_p = blocks_of site.c_noc tn_t * tn_t in
                    let nred_p = blocks_of site.c_nred tk_t * tk_t in
                    let rows_p = blocks_of site.c_nrow tm_t * tm_t in
                    let pad_n = if site.c_noc % tn_t = 0 then 0 else tn_t in
                    let pad_k = if site.c_nred % tk_t = 0 then 0 else tk_t in
                    let pad_m = if site.c_nrow % tm_t = 0 then 0 else tm_t in
                    (* The gate's own charge ([Sched.kernel_workgroup_bytes]): each tile one array
                       of [copies] rotating copies, rounded to the backend's allocation granule,
                       plus the per-kernel tile-MMA scratch of the site's triple, which every
                       completion's [Tile_mma] carries (gh-ocannl-1330, gh-ocannl-1255). Each tile
                       at its own operand's storage precision, the precision
                       [gpu_conv_sketch_schedule]'s [Stage] (no [tile_prec]) declares it at: a
                       mixed-precision site's narrow operand tiles are not priced at its wide
                       accumulator (gh-ocannl-1452). A geometry whose accumulator is masked
                       ([acc_masked]) also places its accumulator fragment, [rows x noc_p] at the
                       output's storage precision, in workgroup-shared memory
                       ([Sched.contract_tensorized_accumulator]'s [shared_frag]: a staged pipeline
                       under pad masks); otherwise it stays in simdgroup fragments. A reduction pad
                       alone masks nothing: [Tensorize] discharges its guard, the staged operands'
                       padded slots being exact zeros. *)
                    let memory_witness ~copies ~acc_masked rows =
                      match limits.Ir.Backend_intf.max_workgroup_memory_bytes with
                      | None -> None
                      | Some cap ->
                          let bytes =
                            Sched.kernel_workgroup_bytes ~capabilities
                              ~arrays:
                                ([
                                   copies * rows * nred_p * Ir.Ops.prec_in_bytes a_prec;
                                   copies * nred_p * noc_p * Ir.Ops.prec_in_bytes b_prec;
                                 ]
                                @
                                if acc_masked then [ rows * noc_p * Ir.Ops.prec_in_bytes prec ]
                                else [])
                              ~mma_triples:[ (prec, a_prec, b_prec) ]
                          in
                          if bytes <= cap then None
                          else
                            Some
                              (Printf.sprintf
                                 "staged tiles need %d bytes of workgroup memory%s, exceeding the \
                                  %d-byte limit"
                                 bytes
                                 (if copies > 1 then Printf.sprintf " at pipeline depth %d" copies
                                  else "")
                                 cap)
                    in
                    (* A geometry whose [Tensorize] masks the accumulator: a column pad, or a row
                       pad or row block that does not divide the row extent. *)
                    let acc_masked p =
                      p.sk_bn > 0 || if p.sk_bm = 0 then p.sk_tm > 0 else site.c_nrow % p.sk_bm <> 0
                    in
                    (* A geometry with any pad: the accumulator masks, or a reduction pad. *)
                    let masked p = p.sk_bk > 0 || acc_masked p in
                    (* Depth twins ride the async arms' element floor (Codex P2 on PR #317, as in
                       the matmul sketch): staged tiles of sub-4-byte elements render the portable
                       synchronous form only. *)
                    let async_wide =
                      Ir.Ops.prec_in_bytes a_prec >= 4 && Ir.Ops.prec_in_bytes b_prec >= 4
                    in
                    let gpu_base =
                      {
                        base with
                        sk_gpu = true;
                        sk_simd = w;
                        sk_epilogue = fused;
                        sk_bn = pad_n;
                        sk_bk = pad_k;
                      }
                    in
                    (* One geometry of the menu: its single-copy judgment, then — under the
                       pipelined staging — its depth twins, each judged at its own footprint.
                       Legality beats policy: the multiplied footprint refutes before the
                       measured-cost exclusions apply. *)
                    let geometry ~pipelined bm =
                      let p0, rows =
                        if bm = 0 then ({ gpu_base with sk_tm = pad_m }, rows_p)
                        else ({ gpu_base with sk_bm = bm }, bm)
                      in
                      let single =
                        if bm > 0 && bm % tm_t <> 0 then
                          Some
                            (Printf.sprintf "row block not a multiple of the %d-row mma tile" tm_t)
                        else if bm > 0 && blocks_of site.c_nrow bm < 2 then
                          Some
                            (Printf.sprintf "row block gives fewer than two blocks of the %d rows"
                               site.c_nrow)
                        else memory_witness ~copies:1 ~acc_masked:(acc_masked p0) rows
                      in
                      match single with
                      | Some w -> Sspace.Refuted w
                      | None when not pipelined -> leaf p0
                      | None ->
                          subt (fun () ->
                              decided_choice
                                (List.map mma.Ir.Backend_intf.mma_pipeline_depths ~f:(fun d ->
                                     ( Family_decision.Twin (`Depth d),
                                       let twin = leaf { p0 with sk_depth = d } in
                                       if d < 1 || d > 2 then
                                         (* The capability list is advisory; the implemented range
                                            is [Schedule.apply_stage]'s. *)
                                         Sspace.Refuted
                                           (Printf.sprintf
                                              "pipeline depth %d is outside the implemented range \
                                               1..2 (Schedule.apply_stage)"
                                              d)
                                       else
                                         match
                                           ( memory_witness ~copies:d ~acc_masked:(acc_masked p0)
                                               rows,
                                             twin )
                                         with
                                         | Some w, _ -> Sspace.Refuted w
                                         | None, (Sspace.Refuted _ as refuted) -> refuted
                                         | None, _ ->
                                             if masked p0 then
                                               Sspace.Excluded
                                                 ( "pad-masked flavor: Tensorize stays on the \
                                                    barrier-bracketed per-call arm, whose leading \
                                                    bracket sits between the prefetch and the \
                                                    compute, so the twin could only pay the \
                                                    doubled footprint (Codex P2 on PR #303)",
                                                   lazy twin )
                                             else if not async_wide then
                                               Sspace.Excluded
                                                 ( "below the async arms' 4-byte element floor: \
                                                    only the synchronous copy would render, so the \
                                                    twin could only pay the doubled footprint \
                                                    (Codex P2 on PR #317)",
                                                   lazy twin )
                                             else twin ))))
                    in
                    (* The geometry menu's pad waste (gh-ocannl-1393, [judge_pad_waste]): the row
                       padded to the row block (the intrinsic row tile for the whole extent), the
                       out-channel and reduction extents to the intrinsic tile. Those two are
                       site-wide — every geometry pads them alike — so they cancel in the relative
                       judgment: a small-channel site is never excluded for its channel pads, since
                       the menu has no less-padded sibling to replace it with, and the least-padded
                       geometry is never excluded. What the menu judges is the row: never at the
                       default threshold (two or more row blocks keep their waste under 2x), but
                       under a lowered [autotune_pad_waste_max]. *)
                    let menu ~pipelined () =
                      decided_choice
                        (judge_pad_waste
                           (List.map [ 0; 8; 16; 32 ] ~f:(fun bm ->
                                ( Family_decision.Conv_rows bm,
                                  [
                                    ("row", site.c_nrow, if bm = 0 then tm_t else bm);
                                    ("oc", site.c_noc, tn_t);
                                    ("red", site.c_nred, tk_t);
                                  ],
                                  geometry ~pipelined bm ))))
                    in
                    subt (fun () ->
                        decided_choice
                          [
                            (Family_decision.Conv_staging `Plain, subt (menu ~pipelined:false));
                            ( Family_decision.Conv_staging `Pipelined,
                              if List.is_empty mma.Ir.Backend_intf.mma_pipeline_depths then
                                Sspace.Refuted "the backend advertises no pipelined staging depth"
                              else subt (menu ~pipelined:true) );
                          ])))
  in
  let flavor ~fused =
    Sspace.Child
      (lazy
        (decided_choice
           [
             (Family_decision.Conv_leg `Cpu, cpu_leg ~fused);
             (Family_decision.Conv_leg `Gpu, gpu_leg ~fused);
           ]))
  in
  (* Unfused first, so the leaves keep the seeds-then-twins order the flat enumeration established
     (candidate timing order, dedup keep-first). The fused flavor is a construction-time verdict of
     the root, as in {!matmul_family_tree}: refuted with the fusion recognizer's own reason when the
     base code carries no fusable tail. The whole-window accumulator contraction
     ([contract_tensorized_accumulator], gh-ocannl-501) lands the fragment store-back after the full
     kernel window, so [Fuse_epilogue]'s exactly-once check passes by construction on every conv
     flavor, multi-window (2-D) convs included. *)
  decided_choice
    [
      (Family_decision.Fusion `Unfused, flavor ~fused:false);
      ( Family_decision.Fusion `Fused,
        match Sched.fuse_epilogue_witness ~target:site.c_d opt with
        | Some w -> Sspace.Refuted w
        | None -> flavor ~fused:true );
    ]

let conv_sketch_tree ?capabilities ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits)
    (opt : LL.optimized) : family_tree option =
  Option.map (detect_conv opt.LL.llc) ~f:(fun site ->
      conv_family_tree ?capabilities ~is_gpu ~is_cpu ~limits ~opt site)

(** {2 The composed seed list} *)

(* What one lowering's family seeding found (gh-ocannl-1407): the family that claimed it, with the
   tree its seeds are the leaves of. *)
type family_seeding =
  | Seeding_matmul of { site : matmul_site; tree : family_tree }
  | Seeding_conv of { site : conv_site; real_stmts : int; tree : family_tree }
  | Seeding_none

(* The families composed into the seed list the search enumerates: the matmul family when a matmul
   site is detected, else the convolution family. Both are refinement trees rooted at the
   epilogue-fusion level (gh-ocannl-613, gh-ocannl-697), so the seeds are the claiming tree's
   leaves, fused twins included, each flavor under its own preconditions. Returned beside the seeds,
   so the seeding census reports the verdicts the seeds came from. *)
let claimed_family ?capabilities ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits)
    (opt : LL.optimized) : family_seeding =
  match detect_matmul_canonical opt with
  | Some (site, _, opt) ->
      Seeding_matmul
        { site; tree = matmul_family_tree ?capabilities ~is_gpu ~is_cpu ~limits ~opt site }
  | None -> (
      match detect_conv opt.LL.llc with
      | None -> Seeding_none
      | Some site ->
          Seeding_conv
            {
              site;
              real_stmts = List.length (conv_real_stmts site opt);
              tree = conv_family_tree ?capabilities ~is_gpu ~is_cpu ~limits ~opt site;
            })

let family_seeding ?capabilities ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits)
    (opt : LL.optimized) : family_seeding * sketch_params list =
  let family = claimed_family ?capabilities ~is_gpu ~is_cpu ~limits opt in
  ( family,
    match family with
    | Seeding_matmul { tree; _ } | Seeding_conv { tree; _ } -> Sspace.leaves tree
    | Seeding_none -> [] )

let no_matmul_site_witness =
  "no matmul site: no accumulation nest classifies as a contraction, directly or through the \
   enabling interchange"

let no_conv_site_witness =
  "no conv site: no accumulation nest carries the implicit-GEMM convolution fingerprint"

let matmul_precedence_witness =
  "a conv site, but the matmul family claims the lowering first: a lowering seeds one family"

(* The combined view (gh-ocannl-1488): a [family = matmul | conv] root above the claiming family's
   own tree, so the family choice is a decision with witnesses like every level below it. Each
   family is judged on the site's own shape — its own detector — and a refuted family says why; the
   claiming family's child is the tree {!family_seeding} built, so the leaves (the seed list) are
   unchanged. Both detectors can only agree on a site the matmul classifier admits and the conv one
   does too, which [classify_matmul]'s rules close (a 1x1 window is a plain GEMM, and the conv
   detector refuses singleton kernel axes); should both claim one, matmul keeps its precedence and
   the conv child is refuted with that witness rather than seeding a second family. [None] when
   neither family has a site, as before. The search driver's entry into the family space
   ({!Autotune.model_family_search}), unforced below the claiming family's root. *)
let sketch_family_tree ?capabilities ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits)
    (opt : LL.optimized) : family_tree option =
  let root ~matmul ~conv =
    Some
      (decided_choice
         [ (Family_decision.Family `Matmul, matmul); (Family_decision.Family `Conv, conv) ])
  in
  match claimed_family ?capabilities ~is_gpu ~is_cpu ~limits opt with
  | Seeding_matmul { tree; _ } ->
      root
        ~matmul:(Sspace.Child (Lazy.from_val tree))
        ~conv:
          (Sspace.Refuted
             (match detect_conv opt.LL.llc with
             | None -> no_conv_site_witness
             | Some _ -> matmul_precedence_witness))
  | Seeding_conv { tree; _ } ->
      root ~matmul:(Sspace.Refuted no_matmul_site_witness) ~conv:(Sspace.Child (Lazy.from_val tree))
  | Seeding_none -> None

let sketch_seed_params ?capabilities ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits)
    (opt : LL.optimized) : sketch_params list =
  snd (family_seeding ?capabilities ~is_gpu ~is_cpu ~limits opt)

(* The coalesced branch's seeds (gh-ocannl-1165, gh-ocannl-1279): per layout, the matmul family
   enumerated over the prepared coalesced code ([coalesced]), so every refutation — launch caps,
   companion coverage of the merged chain, divisibility, workgroup memory — is judged on the site
   the schedule will actually build, and each leaf is stamped with its layout. A separate branch
   beside {!sketch_seed_params}, not a level of its tree: the tree is over one lowering, this branch
   over others. Columns first, then the contraction, whose leaves come from its own K menu
   ([matmul_flavor_tree ~coalesced]); a site has at most one pair of each. GPU leaves only.

   What each layout admits follows from WHAT it flattens. The columns layout flattens the
   accumulator: [Tensorize] reads the output fragment axis by axis and [Fuse_epilogue] relocates
   its tail per axis of the store-back site, and both decline a [Sub_axis] there, so it keeps the
   unfused scalar blocktile leaves (the residual it targets is the scalar family's tile extent
   anyway: #915 closed the tensorized gap on CUDA tf32). The contraction layout flattens only the
   operands, and every pipeline it seeds stages them first, so the micro-kernel reads dense tiles
   and the accumulator is the site's own (gh-ocannl-1459): it admits the scalar leaves, the STAGED
   tensorized ones (its mma menu has no unstaged geometry, [matmul_flavor_tree]), and both their
   fused twins. The CPU pipelines' hoisted packing is not reached: its link-time packing program
   addresses each source axis by its own dim, and refuses a flattened source. The probe's minted
   symbols are discarded: the seeds carry none, and a seeding pass must not shift the names every
   later compile mints. *)
(* Each layout's verdict, for the seeding census (gh-ocannl-1407): its seeds, and [cv_refusal]
   exactly when it has none. Empty off GPU. *)
type coalesced_verdict = {
  cv_layout : coalesce_layout;
  cv_seeds : sketch_params list;
  cv_refusal : string option;
}

let coalesced_seeding ?capabilities ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits)
    (opt : LL.optimized) : coalesced_verdict list =
  if not is_gpu then []
  else
    Idx.discarding_symbols (fun () ->
        List.map [ Coalesce_columns; Coalesce_contraction ] ~f:(fun layout ->
            let refused why = { cv_layout = layout; cv_seeds = []; cv_refusal = Some why } in
            let admitted, admits =
              match layout with
              | Coalesce_columns ->
                  ( (fun p -> p.sk_gpu && (not p.sk_mma) && not p.sk_epilogue),
                    "unfused GPU scalar-blocktile leaf" )
              | Coalesce_contraction ->
                  (* The menu proposes no unstaged tensorized geometry; the filter restates the
                     boundary so a menu change cannot cross it silently. *)
                  ((fun p -> p.sk_gpu && ((not p.sk_mma) || p.sk_bk > 0)), "staged GPU leaf")
            in
            match coalesced ~layout opt with
            | None -> refused "no such loop pair at a matmul site, or the Coalesce op declines it"
            | Some (_, prepared) -> (
                match detect_matmul prepared.LL.llc with
                | None -> refused "the coalesced code re-detects no matmul site"
                | Some site -> (
                    let seeds =
                      matmul_seed_params ~coalesced:layout ?capabilities ~is_gpu ~is_cpu ~limits
                        ~opt:prepared site
                      |> List.filter_map ~f:(fun p ->
                          if admitted p then Some { p with sk_coalesce = Some layout } else None)
                    in
                    match seeds with
                    | [] -> refused ("the coalesced site's family has no " ^ admits)
                    | _ -> { cv_layout = layout; cv_seeds = seeds; cv_refusal = None }))))

let coalesced_seed_params ?capabilities ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits)
    (opt : LL.optimized) : sketch_params list =
  List.concat_map (coalesced_seeding ?capabilities ~is_gpu ~is_cpu ~limits opt) ~f:(fun v ->
      v.cv_seeds)

(** {2 The seeding census (gh-ocannl-1407)}

    Seeding refusals are decided before any candidate exists, so neither the decline report nor
    [schedule_log_declines] (render and compile declines) can see them. The census reads them off
    the judgment the seeds came from: per lowering, the statements and whole-node zeros it holds,
    every top-level conv nest, the family that claimed it, and that family's refusing gates. *)

type seeding_census = {
  sc_stmts : int;
  sc_whole_zeros : int;
  sc_conv_nests : (Ir.Tnode.t * bool) list;
  sc_family : family_seeding;
  sc_seeds : sketch_params list;
  sc_coalesced : coalesced_verdict list option;
}

let seeding_census ?(coalesced = false) ?capabilities ~is_gpu ~is_cpu
    ~(limits : Ir.Backend_intf.hardware_limits) (opt : LL.optimized) : seeding_census =
  (* A dry run: the symbols any probe mints are rewound, so taking a census never shifts the names
     later compiles mint. *)
  Idx.discarding_symbols (fun () ->
      let sc_family, sc_seeds = family_seeding ?capabilities ~is_gpu ~is_cpu ~limits opt in
      let stmts = strip_stmts (LL.flat_lines [ opt.LL.llc ]) in
      (* A whole-node [Zero_out] or its expanded covering nest, as the seeding's companion
         accounting reads it. *)
      let zeroed = List.filter_map stmts ~f:LL.zero_initializer_target in
      let sc_conv_nests =
        List.filter_map stmts ~f:(fun stmt ->
            Option.map (detect_conv stmt) ~f:(fun site ->
                (site.c_d, List.exists zeroed ~f:(Ir.Tnode.equal site.c_d))))
      in
      {
        sc_stmts = List.length stmts;
        sc_whole_zeros = List.length zeroed;
        sc_conv_nests;
        sc_family;
        sc_seeds;
        sc_coalesced =
          (if coalesced then Some (coalesced_seeding ?capabilities ~is_gpu ~is_cpu ~limits opt)
           else None);
      })

let seeding_census_prefix = "autotune-seeding:"

let seeding_census_lines ~where (c : seeding_census) : string list =
  let line fmt = Printf.ksprintf (fun s -> seeding_census_prefix ^ " " ^ s) fmt in
  let detail fmt = Printf.ksprintf (fun s -> seeding_census_prefix ^ "   " ^ s) fmt in
  let verdicts tree =
    Printf.sprintf " refuted=%d excluded=%d"
      (List.length (Sspace.refutations tree))
      (List.length (Sspace.exclusions tree))
  in
  let family, site, verdicts =
    match c.sc_family with
    | Seeding_none -> ("none", "", "")
    | Seeding_matmul { site; tree } ->
        ("matmul", " site=" ^ Ir.Tnode.debug_name site.m_d, verdicts tree)
    | Seeding_conv { site; tree; _ } ->
        ("conv", " site=" ^ Ir.Tnode.debug_name site.c_d, verdicts tree)
  in
  let head =
    line "where=%S stmts=%d whole_zeros=%d family=%s%s seeds=%d%s%s" where c.sc_stmts
      c.sc_whole_zeros family site (List.length c.sc_seeds) verdicts
      (match c.sc_coalesced with
      | None -> ""
      | Some vs ->
          Printf.sprintf " coalesced_seeds=%d"
            (List.sum (module Int) vs ~f:(fun v -> List.length v.cv_seeds)))
  in
  (* One line per distinct witness, in traversal order, with its multiplicity and the first refuted
     commitment's path — the same rendering for both families, whose refusals are both their trees'
     refutations (gh-ocannl-697). *)
  let refutation_lines tree =
    let counts = Hashtbl.create (module String) in
    let firsts =
      List.filter (Sspace.refutations tree) ~f:(fun (_, witness) ->
          let seen = Hashtbl.mem counts witness in
          Hashtbl.incr counts witness;
          not seen)
    in
    List.map firsts ~f:(fun (path, witness) ->
        detail "refuted x%d at %s: %s" (Hashtbl.find_exn counts witness)
          (Sspace.render_path ~label:Family_decision.to_label path)
          witness)
  in
  let family_details =
    match c.sc_family with
    | Seeding_none -> []
    | Seeding_matmul { tree; _ } -> refutation_lines tree
    | Seeding_conv { site; real_stmts; tree } ->
        detail "conv site %s: zeroed=%b real_stmts=%d" (Ir.Tnode.debug_name site.c_d) site.c_zeroed
          real_stmts
        :: refutation_lines tree
  in
  let nests =
    match c.sc_conv_nests with
    | [] -> []
    | nests ->
        [
          detail "conv nests ([Z] = zeroed in this lowering): %s"
            (String.concat ~sep:" "
               (List.map nests ~f:(fun (tn, z) -> Ir.Tnode.debug_name tn ^ if z then "[Z]" else "")));
        ]
  in
  let coalesced =
    List.map (Option.value c.sc_coalesced ~default:[]) ~f:(fun v ->
        let layout =
          match v.cv_layout with
          | Coalesce_columns -> "columns"
          | Coalesce_contraction -> "contraction"
        in
        match v.cv_refusal with
        | Some why -> detail "coalesced %s: refused: %s" layout why
        | None ->
            (* The leaves' distinct geometries, in seeding order (gh-ocannl-1459): the batch,
               epilogue and twin flavors of one geometry share its entry, so the line says which
               tiles the layout seeds without restating every flavor. *)
            let geometry (q : sketch_params) =
              if q.sk_mma then Printf.sprintf "mma %dx%dx%d" q.sk_bm q.sk_bn q.sk_bk
              else Printf.sprintf "%dx%dx%d/%dx%d" q.sk_bm q.sk_bn q.sk_bk q.sk_tm q.sk_tn
            in
            let geometries =
              List.fold v.cv_seeds ~init:[] ~f:(fun acc q ->
                  let g = geometry q in
                  if List.mem acc g ~equal:String.equal then acc else g :: acc)
              |> List.rev
            in
            detail "coalesced %s: %d seeds: %s" layout (List.length v.cv_seeds)
              (String.concat ~sep:" " geometries))
  in
  (head :: family_details) @ coalesced @ nests

(* The exported tree view of the matmul family alone (site detection included);
   {!sketch_family_tree} is the view of whichever family claims the lowering. *)
let matmul_sketch_tree ?capabilities ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits)
    (opt : LL.optimized) : family_tree option =
  Option.map (detect_matmul_canonical opt) ~f:(fun (site, _, opt) ->
      matmul_family_tree ?capabilities ~is_gpu ~is_cpu ~limits ~opt site)

(* gh-ocannl-514 phase 5: lift every tile-lattice exclusion in the family tree, preserving the
   laziness of everything else — a lifted branch remains subject to legality (box refutations), and
   the recursion continues below the lift so nested exclusions of other policies stay excluded. Used
   by [model_default] under config [model_default_geometry_lattice]. *)
let lift_geometry_lattice (tree : family_tree) : family_tree =
  let rec tree_f = function
    | Sspace.Leaf _ as l -> l
    | Sspace.Choice { level; children } ->
        Sspace.Choice
          { level; children = List.map children ~f:(fun (lbl, c) -> (lbl, child_f lbl c)) }
  and child_f lbl = function
    (* The lattice branch is identified by its DECISION (gh-ocannl-591), not by its exclusion
       witness: the witness is prose, and prose a reword can silently desync. *)
    | Sspace.Excluded _ as c
      when Family_decision.equal lbl (Family_decision.Geometry Family_decision.Lattice) ->
        child_f lbl (Sspace.lift_excluded c)
    | Sspace.Child sub -> Sspace.Child (lazy (tree_f (Lazy.force sub)))
    | Sspace.Unknown (w, sub) -> Sspace.Unknown (w, lazy (tree_f (Lazy.force sub)))
    | (Sspace.Excluded _ | Sspace.Refuted _) as c -> c
  in
  tree_f tree

(* gh-ocannl-740: lift every pad-waste exclusion in the family tree, as [lift_geometry_lattice]
   lifts the lattice: laziness preserved, a lifted branch still subject to legality, other policies'
   exclusions kept. A test or driver asking what the menus may propose, rather than what the search
   economy proposes, reads this tree's leaves. *)
let lift_pad_waste (tree : family_tree) : family_tree =
  let rec tree_f = function
    | Sspace.Leaf _ as l -> l
    | Sspace.Choice { level; children } ->
        Sspace.Choice { level; children = List.map children ~f:(fun (lbl, c) -> (lbl, child_f c)) }
  and child_f = function
    | Sspace.Excluded (w, _) as c when String.is_prefix w ~prefix:pad_waste_witness_prefix ->
        child_f (Sspace.lift_excluded c)
    | Sspace.Child sub -> Sspace.Child (lazy (tree_f (Lazy.force sub)))
    | Sspace.Unknown (w, sub) -> Sspace.Unknown (w, lazy (tree_f (Lazy.force sub)))
    | (Sspace.Excluded _ | Sspace.Refuted _) as c -> c
  in
  tree_f tree

(* gh-ocannl-1489: the conv family's certain-traffic increment. Both legs end in the same staging
   ([cpu_conv_sketch_schedule], [gpu_conv_sketch_schedule]): [Stage c_a [row; red]] and [Stage c_b
   [red; oc]], then [Tensorize]. Each [Stage] mints a FRESH tile node — absent from the base
   program, so whatever [Cost_model.analyze] charges it is additional to the base nodes the
   schedule-invariant floor already counts — sized by the tile part's range over the tile loops (a
   strided row is compacted by [Schedule.stage_compaction_factor], so the stride does not enlarge
   it). The load nest writes every slot (edge guards store 0 into the fringe), and the micro-kernel
   reads every slot through [Tile_mma]'s fallback, which [Low_level.affine_accesses] walks.
   [analyze] charges a node [min (node bytes, sum of its per-access images)], guarded accesses
   counted taken — an upper bound on its distinct cells — so each tile node is charged at least one
   whole tile written and one read: [2 * (M*K*wA + K*N*wB)].

   - CPU leg: M is the GEMM row ([Conv_shape `Serial | `Grid]) or the panel's [bm] rows ([`Row_block
   bm], the row padded to a multiple of it); K and N the reduction and out-channel extents,
   unpadded; the widths those of the tiles' own precision ([cpu_conv_pack_prec]: the packing
   precision where an operand stores narrower, [Stage] admitting exact widenings only). - GPU leg: K
   and N padded to the selected format's intrinsic tile (the tree's own resolver,
   [mma_tile_for_precisions_in_scope] at [conv_mma_scope]); M the row block, or the row padded to
   the intrinsic row tile at [Conv_rows 0]; the operands' storage widths ([tile_prec = None]). No
   format tile refutes the leg, and the increment is then vacuous: zero.

   Zero until the row geometry commits — every prefix above it, whatever the leg, prices nothing —
   so the increment is monotone along every path without restating the menus' feasibility here.
   Below it the fusion flavor, the staging, the register tile and the pipelined depth change
   nothing: a rotating tile keeps its logical dimensions ([Schedule.apply_stage] records the depth
   in [opt.pipelined]), and its extra copies are not counted. Pinned by test/operations/
   conv_family_tree against every leaf's scheduled [analyze] (the soundness oracle). *)
let conv_path_traffic_floor ~(limits : Ir.Backend_intf.hardware_limits) (site : conv_site) :
    Family_decision.path -> int =
  let tiles ~pack_prec ~m ~k ~n =
    let wa = staged_tile_elt_bytes ~pack_prec site.c_a
    and wb = staged_tile_elt_bytes ~pack_prec site.c_b in
    2 * ((m * k * wa) + (k * n * wb))
  in
  let cpu =
    lazy
      (let pack_prec = cpu_conv_pack_prec ~limits site in
       fun m -> tiles ~pack_prec ~m ~k:site.c_nred ~n:site.c_noc)
  in
  let gpu =
    lazy
      (Option.bind limits.Ir.Backend_intf.mma ~f:(fun mma ->
           let d_prec = Lazy.force site.c_d.Ir.Tnode.storage_prec in
           let a_prec = Lazy.force site.c_a.Ir.Tnode.storage_prec
           and b_prec = Lazy.force site.c_b.Ir.Tnode.storage_prec in
           mma_tile_for_precisions_in_scope mma ~scope:(conv_mma_scope site) ~a_prec ~b_prec ~d_prec)
      |> Option.map ~f:(fun (tm_t, tn_t, tk_t) ->
          let pad n t = blocks_of n t * t in
          let k = pad site.c_nred tk_t and n = pad site.c_noc tn_t in
          fun bm ->
            let m = if bm = 0 then pad site.c_nrow tm_t else bm in
            tiles ~pack_prec:None ~m ~k ~n))
  in
  fun path ->
    List.fold path ~init:0 ~f:(fun acc (_, d) ->
        match d with
        | Family_decision.Conv_shape (`Serial | `Grid) -> Lazy.force cpu site.c_nrow
        | Family_decision.Conv_shape (`Row_block bm) -> Lazy.force cpu bm
        | Family_decision.Conv_rows bm ->
            Option.value_map (Lazy.force gpu) ~default:0 ~f:(fun f -> f bm)
        | _ -> acc)

(* gh-ocannl-514 phase 5: the certain-traffic increment of a family-tree decision path — bytes that
   {e every} completion below the path moves beyond the schedule-invariant
   [Cost_model.completion_floor], mirroring [Cost_model.analyze]'s counting downward so the composed
   bound stays below every leaf's model score. The committed staging decisions are READ OFF THE
   PATH'S DATA (gh-ocannl-591): each entry is a [Family_decision.t], so a geometry commitment is the
   geometry, a lattice box is its own interval, and the display labels this used to re-parse are
   nothing but a rendering — rewording one can no longer make the arms fall through and zero the
   increment. A box contributes its most favorable (smallest-tiles) corner, so the increment is
   monotone in refinement like the floor it extends. Staged operand tiles are distinct nodes whose
   distinct-cell footprints [analyze] charges on every staged leaf (reads and writes for the
   in-kernel GPU stages; reads only for the CPU packed panels, whose hoisted flavors write at link
   time); everything not certain contributes zero. The decision also says which pipeline minted it
   (a [Gpu_mma] geometry is a GPU tensorized commitment by construction), so the floor no longer
   needs the caller's backend kind alongside the path. *)
let sketch_path_traffic_floor ~(limits : Ir.Backend_intf.hardware_limits) (opt : LL.optimized) :
    Family_decision.path -> int =
  match detect_matmul_canonical opt with
  | None -> (
      (* [claimed_family]'s order: the conv family claims a lowering only where no matmul site does,
         so the conv branch prices exactly the paths of the tree that lowering seeds. *)
      match detect_conv opt.LL.llc with
      | None -> fun _path -> 0
      | Some site -> conv_path_traffic_floor ~limits site)
  | Some (site, _, _) -> (
      let a_prec = Lazy.force site.m_a.Ir.Tnode.storage_prec in
      let b_prec = Lazy.force site.m_b.Ir.Tnode.storage_prec in
      let pa = Ir.Ops.prec_in_bytes a_prec and pb = Ir.Ops.prec_in_bytes b_prec in
      let tile_bytes ~bm ~bn ~bk = (bm * bk * pa) + (bk * bn * pb) in
      let w =
        Option.value_map limits.Ir.Backend_intf.mma ~default:0 ~f:(fun m ->
            m.Ir.Backend_intf.mma_simd_width)
      in
      (* The same per-format tile selection [matmul_family_tree] builds the lattice from — the
         canonical [mma_tile] can be coarser than the selected format's (CUDA's TF32 16x16x8 against
         16x16x16), and an open-axis corner priced at the coarser minimum would overstate the floor
         over the finer completions (Codex P1 on PR #327). No format tile means the tensorized
         branch is refuted and no staged completion exists; the increment is then vacuous, priced at
         the canonical tile. *)
      let tile_min =
        Option.value_map limits.Ir.Backend_intf.mma ~default:(1, 1, 1) ~f:(fun m ->
            let d_prec = Lazy.force site.m_d.Ir.Tnode.storage_prec in
            match mma_tile_for_precisions m ~a_prec ~b_prec ~d_prec with
            | Some tile -> tile
            | None -> m.Ir.Backend_intf.mma_tile)
      in
      fun path ->
        let decisions = List.map path ~f:snd in
        let committed =
          List.find_map decisions ~f:(function Family_decision.Geometry g -> Some g | _ -> None)
        in
        match committed with
        | Some (Family_decision.Gpu_blocktile { g_bm = bm; g_bn = bn; g_bk = bk; _ }) ->
            (* The scalar blocktile pipeline stages both operand tiles in kernel: written and
               read. *)
            2 * tile_bytes ~bm ~bn ~bk
        | Some (Family_decision.Gpu_mma { g_bm = bm; g_bn = bn; g_bk = bk; _ }) ->
            (* Staged mma geometries (bk > 0) stage both operand tiles in kernel; unstaged read in
               place. *)
            if bk > 0 then 2 * tile_bytes ~bm ~bn ~bk else 0
        | Some (Family_decision.Cpu_packed { g_bm = bm; g_bn = bn; g_bk = bk; _ }) ->
            (* CPU packed shapes: only in-kernel packing of both panels ([Packing_shape `Serial])
               certainly ADDS traffic — the packing nest reads the original operands (already in the
               base floor) and writes the panels, which the micro-kernel then reads. A hoisted panel
               is packed at link time and REPLACES the original operand's reads, so its bytes are
               not additional (with both operands hoisted the addition can be exactly zero — Codex
               P1 on PR #327); the hoisted-only Grid and mixed grid-outermost shapes likewise
               replace or split. Zero for all of those. bn = 0 encodes the full column extent. *)
            let serial_packing =
              List.exists decisions ~f:(function
                | Family_decision.Packing_shape `Serial -> true
                | _ -> false)
            in
            if serial_packing then
              let bn_eff = if bn = 0 then site.m_nj else bn in
              tile_bytes ~bm ~bn:bn_eff ~bk
            else 0
        | Some (Family_decision.Cpu_blocktile _) ->
            (* The CPU blocktile pipeline stages nothing: its tiles are registers, and its hoisted
               flavor packs at link time. *)
            0
        | Some Family_decision.Lattice ->
            (* Lattice boxes: bn is pinned at the lane width, bm/bk at their box minima (the most
               favorable corner), the intrinsic tile when the axis is not yet committed. All lattice
               leaves are staged, so entering the lattice already floors at the intrinsic corner. *)
            let tm_t, _, tk_t = tile_min in
            let corner axis default =
              List.fold decisions ~init:default ~f:(fun acc -> function
                | Family_decision.Lattice_box { lb_axis; lb_lo; _ } when Poly.equal lb_axis axis ->
                    lb_lo
                | _ -> acc)
            in
            2 * tile_bytes ~bm:(corner `Bm tm_t) ~bn:w ~bk:(corner `Bk tk_t)
        | None -> 0)
