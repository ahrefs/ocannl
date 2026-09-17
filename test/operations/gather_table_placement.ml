(* gh-ocannl-734: what [Low_level.optimize] does with the table of a dynamic gather ([Get_dynamic])
   that reaches it BEFORE cleanup.

   The ordinary pipeline never builds one: [Assignments] lowering emits no [Get_dynamic], and the
   one the pipeline does produce comes from [rewrite_one_hot_reductions], which runs after both
   virtualization arms. Hand-built IR reaches it through [Ll_test], a supported input class for the
   analysis probes -- and used to die inside cleanup with

   Tnode.update_memory_mode: update 152:cleanup-dropped-set -> 17:surviving-read for <table> is
   already virtual

   because [virtual_llc]'s [Get_dynamic] arm asserted, in a comment, that a local table materializes
   without ever deciding it. The table stayed a virtualization candidate, cleanup's [Set] arm
   committed it [Virtual 152] and dropped its store as dead, and cleanup's own [Get_dynamic] arm
   then asked for [Never_virtual 17] over the corpse.

   Two legs of contract below: an UNDECIDED table is materialized by the gather (legs 1-2, executed,
   because a placement that is right structurally and wrong in the buffer is exactly what a
   collision-free optimize could still be hiding), and a table the author DECLARED virtual -- the
   one reading that cannot be satisfied either way -- is refused with a message naming the node, the
   two readings, and what to do instead (leg 3). *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
open Verdict.Claims

(* [table[dyn]] with the runtime row read out of [idx]: a one-axis gather, so the sole index slot is
   the dynamic one and [Ll_test.gather] plants the placeholder there. *)
let gather ~table ~idx : LL.scalar_t =
  Ll_test.gather ~tn:table
    ~idcs:[| Ll_test.fixed 0 |]
    ~dyn_axis:0
    ~dyn_value:(Ll_test.get idx [| Ll_test.fixed 0 |], Ops.single)

let () =
  let node = Ll_test.node_factory ~first_id:9900 ~dims:[| 4 |] () in
  (* The table's four cells identify their own row (1, 2, 3, 4), so a gather landing on the wrong
     row, or a dropped setter leaving the zero-init behind, changes the value read back. *)
  let filled i = Float.of_int (i + 1) in
  let row = 2 in
  let setter table k =
    Ll_test.loop_n k 4 (Ll_test.set_at table (Ll_test.iter k) (Ll_test.tick k))
  in

  (* === leg 1: the gather is the table's only reader === *)
  let tbl = node "gtp_tbl" in
  let idx = node ~dims:[| 1 |] "gtp_idx" in
  let out = node ~dims:[| 1 |] "gtp_out" in
  Ll_test.materialize idx;
  Ll_test.materialize out;
  let k = Ll_test.sym () in
  let prog =
    Ll_test.seq (setter tbl k) (Ll_test.set_at out (Ll_test.fixed 0) (gather ~table:tbl ~idx))
  in
  let o = Ll_test.optimize ~materialized:[ idx; out ] ~name:"gtp_only" prog in
  p "a gather table left undecided is materialized by the gather" (Ll_test.known_non_virtual o tbl);
  let got =
    Ll_test.execute ~name:"gtp_only" o
      ~seed:[ (idx, [| Float.of_int row |]); (out, [| Ll_test.sentinel |]) ]
      ~read:[ out ]
  in
  p "the gather reads the table row the runtime index names" (Ll_test.same got [ [| filled row |] ]);

  (* === leg 2: the same node read BOTH ways -- the shape the issue reported ===

     The plain [Get] is walked first and mints a [Local_scope] over the table (it is still an
     inlining candidate at that point); the gather then materializes the table, and cleanup retracts
     the scope it minted back to a plain [Get] of the now-materialized buffer. Both readings have to
     land on the buffer the surviving setter wrote, which the sum pins: 2 (row 1, read plainly) + 3
     (row 2, gathered) and no other pair of the table's rows. *)
  let tbl2 = node "gtp_tbl2" in
  let idx2 = node ~dims:[| 1 |] "gtp_idx2" in
  let out2 = node ~dims:[| 1 |] "gtp_out2" in
  Ll_test.materialize idx2;
  Ll_test.materialize out2;
  let k2 = Ll_test.sym () in
  let prog2 =
    Ll_test.seq (setter tbl2 k2)
      (Ll_test.set_at out2 (Ll_test.fixed 0)
         (Ll_test.add (Ll_test.get tbl2 [| Ll_test.fixed 1 |]) (gather ~table:tbl2 ~idx:idx2)))
  in
  let o2 = Ll_test.optimize ~materialized:[ idx2; out2 ] ~name:"gtp_both" prog2 in
  p "a table read both plainly and as a gather table is materialized"
    (Ll_test.known_non_virtual o2 tbl2);
  let got2 =
    Ll_test.execute ~name:"gtp_both" o2
      ~seed:[ (idx2, [| Float.of_int row |]); (out2, [| Ll_test.sentinel |]) ]
      ~read:[ out2 ]
  in
  p "the plain and the gathered reading agree on the materialized table"
    (Ll_test.same got2 [ [| filled 1 +. filled row |] ]);

  (* === leg 3: a table the author declared virtual === *)
  let tbl3 = node "gtp_tbl3" in
  let idx3 = node ~dims:[| 1 |] "gtp_idx3" in
  let out3 = node ~dims:[| 1 |] "gtp_out3" in
  Ll_test.virtualize tbl3;
  Ll_test.materialize idx3;
  Ll_test.materialize out3;
  let k3 = Ll_test.sym () in
  let prog3 =
    Ll_test.seq (setter tbl3 k3)
      (Ll_test.set_at out3 (Ll_test.fixed 0) (gather ~table:tbl3 ~idx:idx3))
  in
  let rejection =
    match Ll_test.optimize ~materialized:[ idx3; out3 ] ~name:"gtp_virtual" prog3 with
    | (_ : LL.optimized) -> None
    | exception Utils.User_error msg -> Some msg
  in
  p "gathering from a node declared virtual is refused" (Option.is_some rejection);
  let says substring =
    Option.value_map rejection ~default:false ~f:(String.is_substring ~substring)
  in
  (* The three things the bare provenance collision did not say. *)
  p "the refusal names the node" (says (Tn.debug_name tbl3));
  p "the refusal names the gather reading" (says "Get_dynamic");
  p "the refusal names the virtualization decision" (says "is virtual in this routine");
  p "the refusal says what to do instead" (says "set_materialized");
  (* And it is not the old message, which named a provenance transition instead. *)
  p "the refusal is not the raw provenance collision" (not (says "update_memory_mode"));
  Stdlib.prerr_endline
    ("gh-734 rejection (not part of the golden): " ^ Option.value rejection ~default:"<none>")
