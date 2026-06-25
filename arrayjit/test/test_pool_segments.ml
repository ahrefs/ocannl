(* Regression test for gh-ocannl-344 AC 6: the pool allocator's per-pool 4 GB (uint32-offset) cap
   and its segmenting/error behavior, exercised on the pure planner [Backends.plan_pool_segments]
   with synthetic byte sizes (no real device memory needed -- a real >4 GB tensor is impractical to
   back).

   The invariants pinned: (1) items whose bumped extent stays under [cap] share one pool with
   increasing, alignment-respecting offsets; (2) an item that would push the running extent past
   [cap] opens a NEW pool at offset 0; (3) a single item larger than [cap] raises (no pool can hold
   it without uint64 offsets). If the allocator dropped the cap split, "splits-at-cap" would print a
   single segment; if it dropped the over-cap error, the last line would print [false]. *)

open Base
module B = Context.Backends_deprecated

let show label (assignments, sizes) =
  Stdio.printf "%s: assignments=[%s] sizes=[%s]\n" label
    (String.concat ~sep:"; " (List.map assignments ~f:(fun (s, o) -> Printf.sprintf "(%d,%d)" s o)))
    (String.concat ~sep:"; " (List.map sizes ~f:Int.to_string))

let () =
  let dn _ = "node" in
  let plan items = B.plan_pool_segments ~cap:1000 ~what:"test" ~debug_name:dn items in
  (* Three items fit under the cap; the 4- and 8-aligned items pad the running offset. *)
  show "fits-one-pool" (plan [ (100, 1); (100, 4); (50, 8) ]);
  (* 600 + 600 > 1000, so the second item opens a second pool. *)
  show "splits-at-cap" (plan [ (600, 1); (600, 1) ]);
  (* Exactly cap fills pool 0; the next byte overflows into pool 1. *)
  show "exact-then-split" (plan [ (1000, 1); (1, 1) ]);
  (* Alignment forces padding: a 5-byte item then an 8-aligned item lands at offset 8. *)
  show "alignment-padding" (plan [ (5, 1); (8, 8) ]);
  show "empty" (plan []);
  (* A single item larger than the cap cannot fit any pool -> must raise. *)
  let raised =
    try
      ignore (plan [ (2000, 1) ]);
      false
    with _ -> true
  in
  Stdio.printf "over-cap single item raises = %b\n" raised
