(* gh-ocannl-1165: the [Coalesce] op.

   The q/k/v projections' site [d[b,s,h,e] += w[h,e,k] * x[b,s,k]] carries its heads as an interior
   batch loop right above the column role, so no column tile is wider than one head. Every access
   reads [h; e] as adjacent plain iterators over unpadded axes, so the pair is ONE loop to the
   compiler: [Sched.Coalesce] re-indexes it as [Sub_axis; Iterator f].

   On hand-built nests (every backend, executed): the positive control applies, is [Op_legal], and
   computes the uncoalesced nest's values bitwise; a per-head operand (one symbol of the pair read
   alone) and a padded inner axis (dim larger than the loop extent: its stride is not the inner
   extent, so the composed index is not the address) decline, each for its own reason. *)

open Base
module LL = Ir.Low_level
module Sched = Ir.Schedule
module L = Ll_test
open Verdict.Claims

let raises_with ~substring f =
  match f () with
  | _ -> false
  | exception Invalid_argument msg -> String.is_substring msg ~substring

(* {1 The op on hand-built nests} *)

let () =
  let bb = 2 and ss = 3 and hh = 3 and ee = 4 and kk = 5 in
  let mk = L.node_factory ~first_id:116500 ~dims:[||] () in
  let w = mk ~dims:[| hh; ee; kk |] "co_w" and x = mk ~dims:[| bb; ss; kk |] "co_x" in
  let g = mk ~dims:[| hh |] "co_g" in
  List.iter [ w; x; g ] ~f:L.materialize;
  (* [d[b,s,h,e] := 0; for k: d[b,s,h,e] += w[h,e,k] * x[b,s,k] * extra] over the nest b, s, h, e,
     with the pair (h, e) returned for the op to name. *)
  let nest ~dst ~extra =
    let b = L.sym () and s = L.sym () and h = L.sym () and e = L.sym () and k = L.sym () in
    let at = [| L.iter b; L.iter s; L.iter h; L.iter e |] in
    let prod =
      L.mul
        (L.get w [| L.iter h; L.iter e; L.iter k |])
        (L.get x [| L.iter b; L.iter s; L.iter k |])
    in
    let rhs = match extra with None -> prod | Some f -> L.mul prod (f ~h) in
    let llc =
      L.loop_n b bb
        (L.loop_n s ss
           (L.loop_n h hh
              (L.loop_n e ee
                 (L.seq
                    (L.set dst at (L.c 0.))
                    (L.loop_n k kk (L.set dst at (L.add (L.get dst at) rhs)))))))
    in
    (llc, h, e)
  in
  let seed =
    [
      ( w,
        Array.init (hh * ee * kk)
          ~f:(L.cycle_flat ~dims:[| hh; ee; kk |] ~modulus:11 ~offset:(-5.) ~stride:0.5) );
      ( x,
        Array.init (bb * ss * kk)
          ~f:(L.cycle_flat ~dims:[| bb; ss; kk |] ~modulus:13 ~offset:(-6.) ~stride:0.25) );
      (g, Array.init hh ~f:(fun i -> Float.of_int (i + 1)));
    ]
  in
  (* Positive control: applies, is [Op_legal], merges the pair into one loop of extent [hh * ee]
     whose accesses read [Sub_axis; Iterator f], and computes what the uncoalesced nest computes. *)
  let d = mk ~dims:[| bb; ss; hh; ee |] "co_d" in
  L.materialize d;
  let llc, h, e = nest ~dst:d ~extra:None in
  let o = L.optimize ~name:"co_plain" llc in
  let op, merged = Sched.coalesce ~outer:h ~inner:e in
  p "control: Coalesce of the head pair is Op_legal"
    (Sched.equal_op_verdict (Sched.op_legality o op) Sched.Op_legal);
  let oc = Sched.apply [ op ] o in
  let merged_extent = ref None and flattened_writes = ref 0 in
  L.walk oc.LL.llc ~on_stmt:(function
    | LL.For_loop { index; to_; _ } when Ir.Indexing.equal_symbol index merged ->
        merged_extent := Some (to_ + 1)
    | LL.Set { tn; idcs; _ }
      when Ir.Tnode.equal tn d
           && Array.equal Ir.Indexing.equal_axis_index (Array.sub idcs ~pos:2 ~len:2)
                [| Ir.Indexing.Sub_axis; Ir.Indexing.Iterator merged |] ->
        Int.incr flattened_writes
    | _ -> ());
  p "control: the merged loop spans every head's columns"
    (Poly.equal !merged_extent (Some (hh * ee)));
  p "control: the writes read the merged index flattened over the pair" (!flattened_writes > 0);
  let run name o = List.hd_exn (L.execute ~name o ~seed:(List.take seed 2) ~read:[ d ]) in
  let want = run "co_plain_run" o in
  let got = run "co_merged_run" oc in
  p "control: the coalesced nest computes the uncoalesced one's values bitwise"
    (Array.exists want ~f:(fun v -> Float.(v <> 0.)) && Array.equal Float.equal want got);
  (* A per-head operand reads [h] alone: the composed index cannot stand for it. *)
  let dg = mk ~dims:[| bb; ss; hh; ee |] "co_dg" in
  L.materialize dg;
  let llc, h, e = nest ~dst:dg ~extra:(Some (fun ~h -> L.get g [| L.iter h |])) in
  let o = L.optimize ~name:"co_per_head" llc in
  let op = fst (Sched.coalesce ~outer:h ~inner:e) in
  p "per-head operand: Coalesce declines because a read does not hold the pair"
    (raises_with ~substring:"does not read" (fun () -> Sched.apply [ op ] o));
  p "per-head operand: the oracle proves it illegal"
    (match Sched.op_legality o op with Sched.Op_illegal _ -> true | _ -> false);
  (* A padded inner axis: the node's dim exceeds the loop's extent, so the inner stride is not the
     inner extent and [ee*h + e] is not the address. *)
  let dp = mk ~dims:[| bb; ss; hh; ee + 2 |] "co_dp" in
  L.materialize dp;
  let llc, h, e = nest ~dst:dp ~extra:None in
  let o = L.optimize ~name:"co_padded" llc in
  let op = fst (Sched.coalesce ~outer:h ~inner:e) in
  p "padded inner axis: Coalesce declines on the dims"
    (raises_with ~substring:"padded axis" (fun () -> Sched.apply [ op ] o));
  p "padded inner axis: the oracle proves it illegal"
    (match Sched.op_legality o op with Sched.Op_illegal _ -> true | _ -> false)
