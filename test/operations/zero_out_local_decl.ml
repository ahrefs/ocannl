(* Regression test for gh-ocannl-420: a legitimately-needed [Zero_out] on a *local* (non-virtual,
   non-materialized) tensor node must not be double-zeroed. The node's C declaration already gets [=
   {0}], so the explicit zeroing loop is redundant and elided.

   [sparse = input ++ "i=>i0"] is a non-surjective einsum (only column 0 is written), so it
   genuinely needs zero-initialization for the other columns. It is declared non-virtual, which
   finalization resolves to [Local] -- exactly the case Step 2 of the fix targets. (Consuming it
   twice used to keep it from being virtualized; since gh-ocannl-1795 a read off column 0 is served
   by its [Zero_out], the written column's component omitted as disjoint, so the placement is stated
   rather than provoked.) The generated code should show [sparse[...] = {0}] in the declaration but
   no [sparse[...] = 0;] zeroing loop in the main logic. *)

module Train = Ocannl.Train
open Ocannl.Nn_blocks.DSL_modules

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let input = TDSL.range 4 in
  let%op sparse = input ++ "i=>i0" in
  Ir.Tnode.update_memory_mode sparse.value Ir.Tnode.Never_virtual (Ir.Tnode.Site "99:test-setup");
  let%op out = (sparse ++ "i2=>i") *. (sparse ++ "i2=>i") in
  Ocannl.Train.set_materialized out.value;
  let ctx = Ocannl.Train.forward_once ctx out in
  Train.printf_tree ctx out

(* gh-ocannl-1625: the leg above cannot discriminate by value -- [sparse] is Local scratch, which no
   host write can prefill, and its [= {0}] declaration is exactly what the elision relies on. This
   executed counterpart pins the elision's LIMITS on hand-built IR, whose statement order is the
   test's own: only the first-touch, function-scope [Zero_out] of a Local node may be skipped, while
   one that follows a write, or sits inside a loop, must still run. The observed outputs are
   materialized and seeded with {!Ll_test.sentinel}, which no statement writes; every expected cell
   is an exact small integer. *)
module L = Ll_test
open Verdict.Claims

let zold_node = L.node_factory ~first_id:1_625_000 ~dims:[| 1 |] ()

(* Compiles [llc] AS WRITTEN (the identity transform: no schedule may move a zero), with [locals]
   declared Local and every seeded or read node materialized. *)
let run_as_written ~name ~locals ~seed ~read llc =
  Base.List.iter locals ~f:(fun tn ->
      Ir.Tnode.update_memory_mode tn Ir.Tnode.Local (Ir.Tnode.Site "99:test-setup"));
  Base.List.iter (Base.List.map seed ~f:fst) ~f:L.materialize;
  let o = L.optimize ~materialized:(Base.List.map seed ~f:fst) ~name llc in
  L.execute ~name o ~seed ~read

let () =
  print_newline ();
  (* A re-zero after a write: [acc] is zeroed at first touch (the elidable one), written, observed,
     then zeroed AGAIN -- that second [Zero_out] is a real one. *)
  let acc = zold_node ~dims:[| 2 |] "zold_acc"
  and o1 = zold_node "zold_o1"
  and o2 = zold_node ~dims:[| 2 |] "zold_o2" in
  let i = L.sym () in
  let llc =
    Ir.Low_level.unflat_lines
      [
        L.zero acc;
        L.set acc [| L.fixed 0 |] (Ir.Low_level.Constant 7.);
        L.set o1 [| L.fixed 0 |] (L.get acc [| L.fixed 0 |]);
        L.zero acc;
        L.loop_n i 2 (L.set o2 [| L.iter i |] (L.get acc [| L.iter i |]));
      ]
  in
  let got =
    run_as_written ~name:"zold_rezero" ~locals:[ acc ]
      ~seed:[ (o1, L.blank 1); (o2, L.blank 2) ]
      ~read:[ o1; o2 ] llc
  in
  p_all2 "re-zero after a write: the write is observed, then the second zero clears every cell"
    (Array.of_list got)
    [| [| 7. |]; [| 0.; 0. |] |]
    ~f:(L.close ~tol:0.);
  (* A zero inside a loop: [accb] is re-zeroed every iteration before accumulating [x.(t)], so [o3 =
     x], not its prefix sums. *)
  let accb = zold_node ~dims:[| 2 |] "zold_accb"
  and x = zold_node ~dims:[| 3 |] "zold_x"
  and o3 = zold_node ~dims:[| 3 |] "zold_o3" in
  let t = L.sym () in
  let it = [| L.iter t |] in
  let llc =
    L.loop_n t 3
      (Ir.Low_level.unflat_lines
         [
           L.zero accb;
           L.set accb
             [| L.fixed 0 |]
             (Ir.Low_level.Binop
                ( Ir.Ops.Add,
                  (L.get accb [| L.fixed 0 |], Ir.Ops.single),
                  (L.get x it, Ir.Ops.single) ));
           L.set o3 it (L.get accb [| L.fixed 0 |]);
         ])
  in
  let got =
    run_as_written ~name:"zold_loop" ~locals:[ accb ]
      ~seed:[ (x, [| 1.; 2.; 3. |]); (o3, L.blank 3) ]
      ~read:[ o3 ] llc
  in
  p_all2 "zero inside a loop: each iteration's accumulator opens from zero, not the running sum"
    (Array.of_list got)
    [| [| 1.; 2.; 3. |] |]
    ~f:(L.close ~tol:0.)
