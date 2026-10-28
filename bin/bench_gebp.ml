(* The packed GEBP schedules [narrow_gebp_bench] and [schedule_bench]'s CPU branch time, as a
   library so that the parity test in test/operations runs THIS builder rather than a copy of it
   (gh-ocannl-627), and so that the two benches build their packed variants through one owner of the
   pad-before-split logic (gh-ocannl-1562).

   The composition is the autotuner's [Sketch_families.cpu_mma_pack_sketch_schedule] with an unsplit
   column panel: split i by [bm] and k by [bk], sink to [k_o { i_o { i_i { j { k_i }}}}] (the
   [Grid_outermost] layout keeps [i_o] outermost), pack B~ [bk x n] over [k_i; j] and A~ [bm x bk]
   over [i_i; k_i], and tensorize the [i_i x j x k_i] micro-kernel. An extent its block does not
   divide is PADDED first, exactly as the tuner's builder does (gh-ocannl-485): [Sched.Pad] guards
   each leaf statement, so the [Split] that follows divides cleanly and the nest stays perfectly
   nested. Split alone would not do for the rows: a non-dividing [Split] of i wraps [i_i]'s body in
   its remainder guard, and the [Swap] that sinks the k-block loop below [i_i] refuses a loop whose
   body is a guard rather than the next loop. (A k remainder alone would compose — its guard lands
   inside [k_i], below every [Swap], where [Tensorize] reads it as a mask — but k pads too, as in
   the tuner's builder, so the bench times the composition the tuner seeds.) Both operands are
   staged whenever anything pads, so every padded slot a micro-kernel reads is a zero-fringe tile
   slot (an in-kernel tile's edge guard, or a hoisted panel's zero-filled pad slots) and [Tensorize]
   absorbs the guards as masks. A layout that reads A in place ([~stage_a:false], schedule_bench's
   [pm_hoist]) has no tile to carry the fringe, so the builder refuses to pad it. The column axis is
   never split, so it is never padded: its tail is the register-tiled micro-kernel's own. *)

open Base
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Idx = Ir.Indexing

type site = {
  i : Idx.symbol;
  j : Idx.symbol;
  k : Idx.symbol;
  ni : int;  (** The row extent, read off the [i] loop. *)
  nj : int;  (** The column extent, read off the [j] loop. *)
  nk : int;  (** The reduction extent, read off the [k] loop. *)
}

let padded ~extent ~block = (extent + block - 1) / block * block

(* The first 3-deep perfect nest of the lowered code, outermost first, with each loop's extent. An
   extent-1 loop is simplified away before the transform runs, so a degenerate extent leaves no such
   nest; that is reported, not raised as [Not_found_s]. *)
let find_site (opt : LL.optimized) : site =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) =
    match llc with
    | LL.For_loop { index; from_; to_; body; _ } ->
        (index, to_ - from_ + 1)
        :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  let paths = List.map (LL.flat_lines [ opt.LL.llc ]) ~f:path in
  match List.find paths ~f:(fun p -> List.length p = 3) with
  | Some [ (i, ni); (j, nj); (k, nk) ] -> { i; j; k; ni; nj; nk }
  | _ ->
      failwith
        (Printf.sprintf
           "Bench_gebp: no 3-deep i/j/k loop nest to schedule (deepest nest found: %d loops) — the \
            packed variants need a non-degenerate extent"
           (List.fold paths ~init:0 ~f:(fun acc p -> Int.max acc (List.length p))))

(* One pad the composition schedules: the row axis [`I] or the reduction axis [`K], its extent, the
   block it pads to, and the padded extent. *)
type pad = { axis : [ `I | `K ]; extent : int; block : int; padded : int }

let axis_name = function `I -> "i" | `K -> "k"

(* The pads the composition schedules at this blocking; empty when both blocks divide. The ONE
   descriptor: [packed_schedule] derives its [Sched.Pad]s from it and the bench's report renders it,
   so what a run says it padded is what it scheduled. *)
let pads ~bm ~bk ~ni ~nk =
  List.filter_map
    [ (`I, ni, bm); (`K, nk, bk) ]
    ~f:(fun (axis, extent, block) ->
      let padded = padded ~extent ~block in
      if padded = extent then None else Some { axis; extent; block; padded })

(* The report line's rendering of [pads], e.g. ["i 76 to 128, k 76 to 128"]; shared by the benches
   so what each says it padded is spelled one way. *)
let describe_pads pads =
  String.concat ~sep:", "
    (List.map pads ~f:(fun { axis; extent; padded; _ } ->
         Printf.sprintf "%s %d to %d" (axis_name axis) extent padded))

(* How much more multiply-add work the padded schedule does than the logical [ni x nj x nk]: the
   product of [padded / extent] over the padded axes. The masked rows and the zero k slots are
   computed and discarded, so a timing at a padded blocking measures this much arithmetic. *)
let work_ratio pads =
  List.fold pads ~init:1. ~f:(fun acc { extent; padded; _ } ->
      acc *. Float.of_int padded /. Float.of_int extent)

(* Where the row-block loop runs and what encloses what. - [Serial]: [k_o { i_o { ... } }], all
   Serial — [narrow_gebp_bench]'s and [schedule_bench]'s [packmma]. The output's whole-node zeroing
   stays legal. - [Grid_below_k]: [k_o { Grid i_o { ... } }] — [schedule_bench]'s [packmma_par]: the
   B~ panel packed at [k_o] sits outside the parallel construct, which the pool re-enters once per k
   block. - [Grid_outermost]: [Grid i_o { k_o { ... } }] — one dispatch spanning the whole triple
   (gh-ocannl-473 / gh-ocannl-475): [narrow_gebp_bench]'s [packmma_par] and [schedule_bench]'s
   [pm_bpk], [pm_mixed] and [pm_hoist]; anything packed at [k_o] lands inside the Grid body, so an
   in-kernel B~ is either re-packed per row block or packed whole ahead of the Grid ([b_pack]). Both
   Grid layouts zero the output in its own Grid-split nest (a whole-node zeroing is not legal beside
   a hardware-annotated loop), so each thread zeroes the rows it then accumulates. *)
type layout = Serial | Grid_below_k | Grid_outermost

(* Why a layout that reads A in place ([~stage_a:false]) cannot be built at this site, if it cannot:
   a padded slot of i or k would be a read past A's extent with no staged tile's zero fringe under
   it for [Tensorize] to mask, and [Tensorize] refuses exactly that ("pad guard on <i> requires
   operand <a> to be a zero-fringe staged tile"; [gebp_padded_parity] pins that the refusal is the
   compiler's). [None] when nothing pads. The ONE predicate: the builder raises on it and a bench
   gates on it, so the two cannot disagree about what pads. *)
let in_place_a_refusal ~bm ~bk ~ni ~nk =
  match pads ~bm ~bk ~ni ~nk with
  | [] -> None
  | pads ->
      Some
        (Printf.sprintf
           "A is read in place, so the %s cannot be padded — a padded slot needs a staged tile's \
            zero fringe to read"
           (String.concat ~sep:" and "
              (List.map pads ~f:(fun { axis; extent; block; _ } ->
                   Printf.sprintf "%s extent %d (block %d)" (axis_name axis) extent block))))

(* How a [Grid_outermost] layout packs an in-kernel B~ (gh-ocannl-1631). [Serial] and [Grid_below_k]
   pack one [bk x n] panel per k block at [k_o], outside the row-block loop: once per k block,
   shared by every row block. Under [Grid_outermost] [k_o] sits inside the Grid body, so:

   - [Per_row_block]: the panel anchored at [k_o] is re-packed by every row block, [ceil(ni / bm)]
   times per k block, into per-chunk privatized storage — concurrently, so with no more row blocks
   than workers the redundant copies cost the critical path nothing. - [Once]: the stage tiles over
   [k_o] as well, which leaves it no outer-part loop to anchor at, so its copy nest goes to the
   routine root, ahead of the Grid: the whole [K x n] B~ (k padded) is packed once — each k block's
   panel once — serially, and read by every row block.

   [grid_b_pack] picks [Once] exactly when the per-row-block layout's per-chunk tiles (the B~ panel,
   the A~ tile, and the masked accumulator fragment when rows pad) outgrow
   [cc_grid_private_bytes_cap]: past it the renderer cannot privatize them and declines the whole
   Grid to serial ([narrow_gebp_bench]'s default n = 512 at 64 x 256: [packmma_par] ran at 2.08 ms
   against serial [packmma]'s 1.78, and at 0.91 packing once, on rog). Below it the serial pre-pass
   is pure critical-path cost: gh-ocannl-1564's cells at n = 48 to 128 ran 8-24% slower packing
   once. The whole B~ is routine-scope scratch at any size: past the cc stack budget the renderer
   declares it on the heap (gh-ocannl-1669). A hoisted B~ is packed once at link time either way and
   keeps its per-k-block panel. *)
type b_pack = Per_row_block | Once

let grid_b_pack ~elem_bytes ~stage_a ~bm ~bk ~ni ~nj =
  let per_chunk =
    elem_bytes * ((bk * nj) + (if stage_a then bm * bk else 0) + if ni % bm <> 0 then bm * nj else 0)
  in
  if per_chunk > Lazy.force Ir.C_syntax.per_chunk_private_bytes_cap then Once else Per_row_block

(* The k-block split (gh-ocannl-1631): with k padded, every element the packing nests copy is
   guarded by [bk*k_o + k_i < nk] (a [Where] storing the zero fringe), in every k block, although
   only the last block's guard can fail. Partitioning [k_o] at its last full block — the [Partition]
   [Sched.Pad]'s contract names for exactly this — leaves the full blocks a segment whose range
   decides every such guard, which [Sched.apply]'s trailing simplify folds away, and keeps the
   guards in the edge block alone: the k analog of [Tensorize]'s full-row-block split
   (gh-ocannl-1563). The micro-kernel itself carries no k guard either way (its reduction masks are
   discharged against the zero-filled operands), so values are unchanged. Empty when k divides or
   has no full block. *)
let k_block_split ~nk ~bk ~k_o =
  let full = nk / bk in
  if nk % bk = 0 || full = 0 then [] else [ fst (Sched.partition ~axis:k_o ~breakpoints:[ full ]) ]

let packed_schedule ~layout ?(hoist_b = false) ?(stage_a = true) ?b_pack ~tile_prec ?tile ~bm ~bk
    ~(a : Ir.Tnode.t) ~(b : Ir.Tnode.t) ~(c : Ir.Tnode.t) (opt : LL.optimized) : Sched.schedule =
  let { i; j; k; ni; nj; nk } = find_site opt in
  (if not stage_a then
     match in_place_a_refusal ~bm ~bk ~ni ~nk with
     | Some why -> invalid_arg ("Bench_gebp.packed_schedule: " ^ why)
     | None -> ());
  let pads = pads ~bm ~bk ~ni ~nk in
  let pad_ops =
    List.map pads ~f:(fun { axis; block; _ } ->
        Sched.Pad { axis = (match axis with `I -> i | `K -> k); to_multiple_of = block })
  in
  let b_pack_resolved =
    match b_pack with
    | Some p -> p
    | None ->
        let prec = Option.value tile_prec ~default:(Lazy.force b.Ir.Tnode.storage_prec) in
        grid_b_pack ~elem_bytes:(Ir.Ops.prec_in_bytes prec) ~stage_a ~bm ~bk ~ni ~nj
  in
  let grid = match layout with Serial -> false | Grid_below_k | Grid_outermost -> true in
  let outer_i = if grid then LL.Grid else LL.Serial in
  let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:outer_i ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
  let stage ?(hoisted = false) source tile_loops =
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
  let zops =
    (* The zeroing nest is split by [bm] WITHOUT a pad: nothing swaps or tensorizes it, so [Split]'s
       remainder guard is harmless there, and it has the padded accumulation's block count, [ceil(ni
       / bm)], either way. *)
    if not grid then []
    else
      let ez, zsyms = Sched.expand_zero ~tn:c in
      let zi = List.hd_exn zsyms in
      let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
      [ ez; sp_zi ]
  in
  let tz, _lane = Sched.tensorize ?tile ~i:i_i ~j ~k:k_i ~simd_width:1 () in
  pad_ops @ zops @ [ sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ]
  @ (match layout with Serial | Grid_below_k -> sink i_o [ k_o ] | Grid_outermost -> [])
  @ [
      stage ~hoisted:hoist_b b
        (match (layout, hoist_b, b_pack_resolved) with
        | Grid_outermost, false, Once -> [ k_o; k_i; j ]
        | _ -> [ k_i; j ]);
    ]
  @ (if stage_a then [ stage a [ i_i; k_i ] ] else [])
  @ [ tz ] @ k_block_split ~nk ~bk ~k_o

(* The operand values the bench mints, at index pair [(row, column)] of each operand: drawn through
   [Bench_checksum.mix] so that no shift of any index is a symmetry, and exactly representable at
   every storage precision. [narrow_gebp_bench]'s comment on its [ma]/[mb] says why each half of
   that matters. Shared so the parity test's producers are the bench's. *)
let a_value row col = Float.of_int (1 + (Bench_checksum.mix ~salt:0x5A17 row col % 3)) *. 0.25
let b_value row col = Float.of_int ((Bench_checksum.mix ~salt:0x3C6E row col % 5) - 2) *. 0.5
