(* gh-ocannl-1282: a loop symbol two sibling loops bind with different bounds must not borrow one
   loop's bounds for the other's occurrences.

   [Affine.pair_conflict] once took ONE range per symbol for both sides of the pair, and
   [Schedule.pair_aligned_query] handed it the left nest's bounds first. With the same symbol [s]
   bound [0..1] by the writer's inner loop and [0..3] by the reader's,

   {v for p = 0..1: for s = 0..1: a[p; s+2] := 1 + p for q = 0..1: for s = 0..3: b[q; s] := a[1-q;
   s] v}

   the reader's [s] was restricted to [0..1], its columns looked disjoint from the writer's [2..3],
   and [aligned_chains] accepted the cross-thread dependence (thread q reads row [1-q], which the
   OTHER thread writes) as aligned. Each side now ranges its own symbols ([Left_v]/[Right_v]), and a
   nest's environment hulls a symbol several of its loops bind.

   Every case asserts on [aligned_chains] itself, so a claim fails when the answer moves either way:
   the declined cases can only pass by declining, and the positive control — the same nests with the
   reader on its own thread's row — keeps them from passing because the analysis declines
   everything. The fresh-symbol controls state the answer the shared symbol must not change.

   Since gh-ocannl-1402, [Low_level.optimize] refuses a routine whose loops share a binder (the
   loop-binder lint, pinned by [loop_binder_lint]), so the shared shape can no longer arrive through
   it. It can still arrive past it -- a schedule rewrite or [?prelowered] IR is not linted -- and
   [aligned_chains] must stay sound there. Every case therefore optimizes the fresh-symbol program
   and, for the shared variants, renames the second binder onto the first in the OPTIMIZED code
   ([share]): the same program the query used to be handed, built without lying to the lint. *)

open Base
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Sched = Ir.Schedule
open Ll_test
open Verdict.Claims

let node = node_factory ~first_id:982_000_000 ~dims:[| 2; 4 |] ()

(* The writer nest: [for p < 2: for s < 2: a[p; s+2] := 1 + p]. *)
let writer a ~s =
  let p = sym () in
  loop_n p 2 (loop_n s 2 (set a [| iter p; aff [ (1, s) ] 2 |] (tick p)))

(* The reader nest: [for q < 2: for s < 4: b[q; s] := a[row; s]], [row] built from [q]. *)
let reader a b ~s ~row =
  let q = sym () in
  loop_n q 2 (loop_n s 4 (set b [| iter q; iter s |] (get a [| row q; iter s |])))

let other_row q = aff [ (-1, q) ] 1
let own_row q = iter q

(* [share ~from ~into llc] renames every binder and occurrence of [from] to [into]: the
   post-optimize route to a routine whose loops share a binder. Only the constructs these nests
   optimize to are walked; anything else fails the case rather than pass through half-renamed. *)
let share ~from ~into (llc : LL.t) : LL.t =
  let sy s = if Idx.equal_symbol s from then into else s in
  let idx : Idx.axis_index -> Idx.axis_index = function
    | Iterator s -> Iterator (sy s)
    | Affine { symbols; offset } ->
        Idx.affine ~symbols:(List.map symbols ~f:(fun (c, s) -> (c, sy s))) ~offset
    | (Fixed_idx _ | Sub_axis) as i -> i
    | Concat _ -> failwith "share: a Concat index"
  in
  let rec scalar (sc : LL.scalar_t) : LL.scalar_t =
    match sc with
    | Get (tn, idcs) -> Get (tn, Array.map idcs ~f:idx)
    | Embed_index i -> Embed_index (idx i)
    | Binop (op, (a, pa), (b, pb)) -> Binop (op, (scalar a, pa), (scalar b, pb))
    | Unop (op, (a, pa)) -> Unop (op, (scalar a, pa))
    | (Constant _ | Constant_bits _) as c -> c
    | _ -> failwith "share: an unexpected scalar construct"
  in
  let rec proc (c : LL.t) : LL.t =
    match c with
    | Seq (a, b) -> Seq (proc a, proc b)
    | For_loop r -> For_loop { r with index = sy r.index; body = proc r.body }
    | Set r -> Set { r with idcs = Array.map r.idcs ~f:idx; llsc = scalar r.llsc }
    | (Noop | Comment _ | Zero_out _) as c -> c
    | _ -> failwith "share: an unexpected statement construct"
  in
  proc llc

(* Optimize the fresh-symbol program, then make [s'] share [s]'s binder unless [fresh]. *)
let chains ~name ~fresh ~s ~s' stmts ~a ~b =
  let o = optimize ~materialized:[ a; b ] ~name (LL.unflat_lines stmts) in
  let o = if fresh then o else { o with LL.llc = share ~from:s' ~into:s o.LL.llc } in
  let binds_s = function LL.For_loop { index; _ } -> Idx.equal_symbol index s | _ -> false in
  p
    (name ^ ": the queried routine binds s " ^ if fresh then "once" else "in two loops")
    (count_stmt ~f:binds_s o.LL.llc = if fresh then 1 else 2);
  Sched.aligned_chains o

let cross_nest ~name ~order ~fresh ~row =
  let a = node "a" and b = node "b" in
  let s = sym () and s' = sym () in
  let w = writer a ~s and r = reader a b ~s:s' ~row in
  chains ~name ~fresh ~s ~s' ~a ~b
    (match order with `Writer_first -> [ w; r ] | `Reader_first -> [ r; w ])

(* The same pair inside ONE nest: the parallel loop [p] encloses both sibling inner loops, so the
   per-nest hazard analysis meets both bindings of [s] in one loop environment. *)
let within_nest ~name ~fresh ~row =
  let a = node "a" and b = node "b" in
  let t = sym () and s = sym () and s' = sym () in
  let body =
    LL.unflat_lines
      [
        loop_n s 2 (set a [| iter t; aff [ (1, s) ] 2 |] (tick t));
        loop_n s' 4 (set b [| iter t; iter s' |] (get a [| row t; iter s' |]));
      ]
  in
  chains ~name ~fresh ~s ~s' ~a ~b [ loop_n t 2 body ]

let declined name verdict = p (name ^ ": aligned_chains declines") (Option.is_none verdict)
let aligned name verdict = p (name ^ ": aligned_chains accepts") (Option.is_some verdict)

let () =
  Stdio.printf "=== cross-nest: two nests bind the same inner symbol ===\n";
  declined "shared s, writer nest first"
    (cross_nest ~name:"ac_shared_wf" ~order:`Writer_first ~fresh:false ~row:other_row);
  declined "shared s, reader nest first"
    (cross_nest ~name:"ac_shared_rf" ~order:`Reader_first ~fresh:false ~row:other_row);
  declined "control: fresh reader symbol, writer nest first"
    (cross_nest ~name:"ac_fresh_wf" ~order:`Writer_first ~fresh:true ~row:other_row);
  declined "control: fresh reader symbol, reader nest first"
    (cross_nest ~name:"ac_fresh_rf" ~order:`Reader_first ~fresh:true ~row:other_row);
  aligned "control: shared s, reader on its own row"
    (cross_nest ~name:"ac_own_row" ~order:`Writer_first ~fresh:false ~row:own_row);
  Stdio.printf "\n=== within one nest: sibling loops bind the same symbol ===\n";
  declined "shared s, sibling loops under p"
    (within_nest ~name:"ac_nest_shared" ~fresh:false ~row:other_row);
  declined "control: fresh second sibling symbol"
    (within_nest ~name:"ac_nest_fresh" ~fresh:true ~row:other_row);
  aligned "control: shared s, reader on its own row"
    (within_nest ~name:"ac_nest_own_row" ~fresh:false ~row:own_row)
