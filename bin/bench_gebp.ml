(* The packed GEBP schedule [narrow_gebp_bench] times, as a library so that the parity test in
   test/operations runs THIS builder rather than a copy of it (gh-ocannl-627).

   The composition is the autotuner's [Sketch_families.cpu_mma_pack_sketch_schedule] with an unsplit
   column panel: split i by [bm] and k by [bk], sink to [k_o { i_o { i_i { j { k_i }}}}] (the Grid
   flavor keeps [i_o] outermost), pack B~ [bk x n] over [k_i; j] and A~ [bm x bk] over [i_i; k_i],
   and tensorize the [i_i x j x k_i] micro-kernel. An extent its block does not divide is PADDED
   first, exactly as the tuner's builder does (gh-ocannl-485): [Sched.Pad] guards each leaf
   statement, so the [Split] that follows divides cleanly and the nest stays perfectly nested. Split
   alone would not do for the rows: a non-dividing [Split] of i wraps [i_i]'s body in its remainder
   guard, and the [Swap] that sinks the k-block loop below [i_i] refuses a loop whose body is a
   guard rather than the next loop. (A k remainder alone would compose — its guard lands inside
   [k_i], below every [Swap], where [Tensorize] reads it as a mask — but k pads too, as in the
   tuner's builder, so the bench times the composition the tuner seeds.) Both operands are staged at
   every geometry, so every padded slot a micro-kernel reads is a zero-fringe tile slot and
   [Tensorize] absorbs the guards as masks. The column axis is never split, so it is never padded:
   its tail is the register-tiled micro-kernel's own. *)

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

(* How much more multiply-add work the padded schedule does than the logical [ni x nj x nk]: the
   product of [padded / extent] over the padded axes. The masked rows and the zero k slots are
   computed and discarded, so a timing at a padded blocking measures this much arithmetic. *)
let work_ratio pads =
  List.fold pads ~init:1. ~f:(fun acc { extent; padded; _ } ->
      acc *. Float.of_int padded /. Float.of_int extent)

let packed_schedule ~grid ~tile_prec ?tile ~bm ~bk ~(a : Ir.Tnode.t) ~(b : Ir.Tnode.t)
    ~(c : Ir.Tnode.t) (opt : LL.optimized) : Sched.schedule =
  let { i; j; k; ni; nk; _ } = find_site opt in
  let pad_ops =
    List.map (pads ~bm ~bk ~ni ~nk) ~f:(fun { axis; block; _ } ->
        Sched.Pad { axis = (match axis with `I -> i | `K -> k); to_multiple_of = block })
  in
  let outer_i = if grid then LL.Grid else LL.Serial in
  let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:outer_i ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
  let stage source tile_loops =
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
  let zops =
    (* The Grid flavor zeroes the output in its own Grid-split nest, so each thread zeroes the rows
       it then accumulates. That nest is split by [bm] WITHOUT a pad: nothing swaps or tensorizes
       it, so [Split]'s remainder guard is harmless there, and it has the padded accumulation's
       block count, [ceil(ni / bm)], either way. *)
    if not grid then []
    else
      let ez, zsyms = Sched.expand_zero ~tn:c in
      let zi = List.hd_exn zsyms in
      let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
      [ ez; sp_zi ]
  in
  let tz, _lane = Sched.tensorize ?tile ~i:i_i ~j ~k:k_i ~simd_width:1 () in
  pad_ops @ zops @ [ sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ]
  @ (if grid then [] else sink i_o [ k_o ])
  @ [ stage b [ k_i; j ]; stage a [ i_i; k_i ] ]
  @ [ tz ]

(* The operand values the bench mints, at index pair [(row, column)] of each operand: drawn through
   [Bench_checksum.mix] so that no shift of any index is a symmetry, and exactly representable at
   every storage precision. [narrow_gebp_bench]'s comment on its [ma]/[mb] says why each half of
   that matters. Shared so the parity test's producers are the bench's. *)
let a_value row col = Float.of_int (1 + (Bench_checksum.mix ~salt:0x5A17 row col % 3)) *. 0.25
let b_value row col = Float.of_int ((Bench_checksum.mix ~salt:0x3C6E row col % 5) - 2) *. 0.5
