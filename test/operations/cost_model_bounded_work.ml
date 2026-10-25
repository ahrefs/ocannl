(* gh-ocannl-1458: [Cost_model.bounded_work], the CPU default preset's per-dispatch work estimate.
   Pure, over hand-built IR: what it counts (stores, arithmetic, reads; loop extents multiply), that
   it saturates at its cap instead of overflowing, and that it refuses -- [None], never a large
   number -- the constructs it does not model, since unknown work must not pass a profitability
   gate. *)
open Base
module Cost_model = Ir.Cost_model
module LL = Ir.Low_level
open Ll_builders
open Verdict.Claims

let node = node_factory ~first_id:1458500 ~dims:[| 100 |] ()
let work ?(cap = 1_000_000) llc = Cost_model.bounded_work ~cap llc

let () =
  let x = node "x" and y = node "y" in
  let i = sym () in
  (* One store and one read per cell: a copy has no FLOPs but has work. *)
  let copy = loop_n i 100 (set_at y (iter i) (get x [| iter i |])) in
  p "a 100-cell copy costs 200 (a store and a read per cell)"
    (Option.equal Int.equal (work copy) (Some 200));
  let j = sym () and k = sym () in
  let cheap = loop_n j 100 (set_at y (iter j) (add (embed j) (c 1.))) in
  let poly e = List.fold (List.init 8 ~f:Fn.id) ~init:e ~f:(fun e _ -> mul e (c 0.5)) in
  let costly = loop_n k 100 (set_at y (iter k) (add (poly (embed k)) (c 1.))) in
  p "equal stores over equal extents: the cheap expression costs 200, the costly one 1000"
    (Option.equal Int.equal (work cheap) (Some 200)
    && Option.equal Int.equal (work costly) (Some 1000))

let () =
  let y = node "y_sat" in
  let i = sym () and j = sym () in
  let huge = loop_n i 1_000_000 (loop_n j 1_000_000 (set_at y (fixed 0) (c 1.))) in
  p "a 10^12-store nest saturates at the cap"
    (Option.equal Int.equal (work ~cap:16384 huge) (Some 16384));
  let i2 = sym () and j2 = sym () and k2 = sym () in
  let wider =
    loop_n i2 (1 lsl 40) (loop_n j2 (1 lsl 40) (loop_n k2 (1 lsl 40) (set_at y (fixed 0) (c 1.))))
  in
  p "at an Int.max_value cap the product saturates instead of wrapping"
    (Option.equal Int.equal (work ~cap:Int.max_value wider) (Some Int.max_value))

let () =
  let y = node "y_edge" in
  let i = sym () and j = sym () in
  p "an empty loop costs nothing"
    (Option.equal Int.equal (work (loop ~from_:5 ~upto:4 i (set_at y (iter i) (c 1.)))) (Some 0));
  p "a whole-node zero costs its cells" (Option.equal Int.equal (work (zero y)) (Some 100));
  let guarded = loop_n j 10 (if_ (lt (embed j) (ic 5)) (set_at y (iter j) (c 1.))) in
  p "a guarded body is charged guards-taken (the comparison plus the store per iteration)"
    (Option.equal Int.equal (work guarded) (Some 20))

let () =
  let y = node "y_unknown" and st = node "st" ~dims:[| 1 |] in
  virtualize st;
  let i = sym () and s = sym () in
  p "a barrier is unknown work"
    (Option.is_none (work (loop_n i 10 (seq LL.Workgroup_barrier (set_at y (iter i) (c 1.))))));
  let cr = carry ~init:(c 0.) st in
  let scan_loop =
    scan ~upto:9 s ~carried:[ cr ]
      (seq (set_next cr (add (prev cr) (c 1.))) (set_at y (iter s) (next cr)))
  in
  p "a scan loop is unknown work" (Option.is_none (work scan_loop))
