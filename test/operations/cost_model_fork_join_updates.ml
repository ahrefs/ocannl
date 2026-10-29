(* gh-ocannl-1458: [Cost_model.fork_join_updates], the one work count of a CPU fork/join -- the
   default CPU preset's per-dispatch gate and the renderer's launch floor both read it. Pure, over
   hand-built IR: what it counts (stores and register updates; loop extents multiply; scope bodies),
   what it deliberately does not (the arithmetic inside a store's expression), the [Tile_mma]
   charge, and that it saturates at its cap instead of overflowing. *)
open Base
module Cost_model = Ir.Cost_model
module LL = Ir.Low_level
open Ll_builders
open Verdict.Claims

let node = node_factory ~first_id:1458500 ~dims:[| 100 |] ()
let work ?(cap = 1_000_000) llc = Cost_model.fork_join_updates ~cap llc

let () =
  let x = node "x" and y = node "y" in
  let i = sym () and j = sym () and k = sym () and l = sym () in
  let copy = loop_n i 100 (set_at y (iter i) (get x [| iter i |])) in
  p "a 100-cell copy costs 100 (one store per cell)" (work copy = 100);
  let poly e = List.fold (List.init 8 ~f:Fn.id) ~init:e ~f:(fun e _ -> mul e (c 0.5)) in
  let costly = loop_n j 100 (set_at y (iter j) (add (poly (embed j)) (c 1.))) in
  p "a store's expression is not charged: the 9-op expression costs what the copy does"
    (work costly = work copy);
  let reduction =
    loop_n k 100 (loop_n l 16 (set_at y (iter k) (add (get y [| iter k |]) (get x [| iter l |]))))
  in
  p "a reduction body is charged per inner iteration: 100 x 16" (work reduction = 1600)

let () =
  let y = node "y_sat" in
  let i = sym () and j = sym () in
  let huge = loop_n i 1_000_000 (loop_n j 1_000_000 (set_at y (fixed 0) (c 1.))) in
  p "a 10^12-store nest saturates at the cap" (work ~cap:16384 huge = 16384);
  let i2 = sym () and j2 = sym () and k2 = sym () in
  let wider =
    loop_n i2 (1 lsl 40) (loop_n j2 (1 lsl 40) (loop_n k2 (1 lsl 40) (set_at y (fixed 0) (c 1.))))
  in
  p "at an Int.max_value cap the product saturates instead of wrapping"
    (work ~cap:Int.max_value wider = Int.max_value)

let () =
  let y = node "y_edge" in
  let i = sym () and j = sym () in
  p "an empty loop costs nothing" (work (loop ~from_:5 ~upto:4 i (set_at y (iter i) (c 1.))) = 0);
  p "a whole-node zero costs its cells" (work (zero y) = 100);
  let guarded = loop_n j 10 (if_idx (lt (embed j) (ic 5)) (set_at y (iter j) (c 1.))) in
  p "a guarded body is charged guards-taken (a store per iteration)" (work guarded = 10)

let () =
  let y = node "y_unknown" and st = node "st" ~dims:[| 1 |] in
  virtualize st;
  let i = sym () and s = sym () in
  p "a barrier reads as the cap"
    (work ~cap:777 (loop_n i 10 (seq LL.Workgroup_barrier (set_at y (iter i) (c 1.)))) = 777);
  let cr = carry ~init:(c 0.) st in
  let scan_loop =
    scan ~upto:9 s ~carried:[ cr ]
      (seq (set_next cr (add (prev cr) (c 1.))) (set_at y (iter s) (next cr)))
  in
  p "a scan loop reads as the cap" (work ~cap:777 scan_loop = 777)
