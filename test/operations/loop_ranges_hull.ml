(* gh-ocannl-1305: [Low_level.loop_ranges] holds ONE range per loop symbol, the hull of every loop
   binding it, so no consumer can read a first-binding-only range for a reused symbol.

   Loop symbols are not unique within a routine once scheduling has run: the binder lint
   ([Low_level.validate_loop_binders], gh-ocannl-1402) holds at [optimize]'s entry only, and
   schedule rewrites emit sibling nests sharing a symbol by design. Before this, [loop_bounds]
   returned one entry per binding and every caller turned it into a range with [List.Assoc.find],
   which keeps the first; the within-nest form of gh-ocannl-1282 got past the per-nest hazard check
   that way. The shapes below are raw IR (the binder lint would refuse them at [optimize]), which is
   exactly the post-schedule form the consumers see. The last leg shows why the hull matters to a
   consumer: an access-validity query ranged by the first binding alone admits cells the second
   sibling reads out of bounds. *)

open Base
open Ll_builders
open Verdict.Claims
module R = Ir.Affine.Loop_ranges

let mk = node_factory ~first_id:13050 ~dims:[| 4 |] ()

let range_is r s (lo, hi) =
  match R.find r s with Some (lo', hi') -> lo = lo' && hi = hi' | None -> false

let bindings_of r s = List.count (R.to_alist r) ~f:(fun (s', _) -> Idx.equal_symbol s s')

let () =
  let dst = mk "lrh_dst" and src = mk "lrh_src" in
  (* Two sibling nests binding [s], over [0..1] and then [5..7]: the shape a schedule rewrite's
     sibling copies take. Neither binding is a sub-range of the other, so a hull is the only answer
     that reports both. *)
  let shared = sym () in
  let body = set_at dst (iter shared) (get src [| iter shared |]) in
  let siblings = seq (loop ~upto:1 shared body) (loop ~from_:5 ~upto:7 shared body) in
  let r = LL.loop_ranges siblings in
  p "sibling nests: the range of the shared symbol is the hull of both bindings"
    (range_is r shared (0, 7));
  p "sibling nests: the shared symbol has exactly one entry" (bindings_of r shared = 1);
  (* The same within one nest: two inner loops of an outer loop rebind [s] (the within-nest form of
     gh-ocannl-1282), and the outer loop keeps its own range. *)
  let o = sym () and s = sym () in
  let inner upto from_ = loop ~from_ ~upto s (set dst [| iter s |] (get src [| iter o |])) in
  let nest = loop ~upto:3 o (seq (inner 1 0) (inner 9 2)) in
  let r = LL.loop_ranges nest in
  p "inner siblings: the range of the shared symbol is the hull of both bindings"
    (range_is r s (0, 9));
  p "inner siblings: the enclosing loop keeps its own range" (range_is r o (0, 3));
  p "inner siblings: one entry per symbol"
    (bindings_of r s = 1 && bindings_of r o = 1 && List.length (R.to_alist r) = 2);
  (* The constructors hull regardless of binding order, and a symbol no loop binds stays absent: the
     peel's static-symbol classification reads that absence. *)
  let a = sym () and unbound = sym () in
  p "of_bindings hulls in either order"
    (range_is (R.of_bindings [ (a, (5, 7)); (a, (0, 1)) ]) a (0, 7)
    && range_is (R.of_bindings [ (a, (0, 1)); (a, (5, 7)) ]) a (0, 7));
  p "bind widens an existing range"
    (range_is (R.bind (R.of_bindings [ (a, (2, 3)) ]) a (0, 1)) a (0, 3));
  p "a symbol no loop binds has no range"
    (Option.is_none (R.find r unbound) && not (R.mem r unbound));
  (* A consumer: is [dst[s]] within [dst]'s 4 cells over every iteration of the sibling nests? The
     second sibling's [s = 5..7] reads past the end, so the query must say no; ranged by the first
     binding ([0..1]) alone it would say yes. *)
  let range = R.find (LL.loop_ranges siblings) in
  let coords = Ir.Affine.view ~range ~dims:[| 4 |] [| iter shared |] in
  p "an access-validity query ranged by loop_ranges sees the second sibling's out-of-bounds cells"
    (not (Ir.Affine.within_box ~range coords));
  p
    "the same query over the first sibling's range alone admits the access (the hull is what \
     decides)"
    (Ir.Affine.within_box
       ~range:(fun s' -> if Idx.equal_symbol shared s' then Some (0, 1) else None)
       coords);
  (* gh-ocannl-1628: [Affine.Loop_chain] enforces binder uniqueness per enclosing chain at
     construction, while sibling loops remain separate single-binder chains. *)
  let module C = Ir.Affine.Loop_chain in
  let lc_s = sym () and lc_t = sym () in
  let dup_expected = { C.symbol = lc_s; outer = (0, 3); inner = (1, 2) } in
  p "Loop_chain.of_bindings refuses a duplicate binder with a typed error"
    (match C.of_bindings [ (lc_s, (0, 3)); (lc_t, (0, 1)); (lc_s, (1, 2)) ] with
    | Error dup -> C.equal_duplicate dup dup_expected
    | Ok _ -> false);
  p "Loop_chain.enter refuses a duplicate binder with a typed error"
    (match C.enter (C.of_bindings_exn [ (lc_s, (0, 3)) ]) lc_s (1, 2) with
    | Error dup -> C.equal_duplicate dup dup_expected
    | Ok _ -> false);
  p "Loop_chain.map_symbols refuses a rename collision with a typed error"
    (let chain = C.of_bindings_exn [ (lc_s, (0, 3)); (lc_t, (1, 2)) ] in
     match C.map_symbols chain ~f:(fun _ -> lc_s) with
     | Error dup -> C.equal_duplicate dup dup_expected
     | Ok _ -> false);
  let nested_dup = loop ~upto:3 lc_s (loop ~from_:1 ~upto:2 lc_s (set_at dst (iter lc_s) (c 1.))) in
  p "affine_relations raises Duplicate_binder on nested loops rebinding a symbol"
    (match LL.affine_relations nested_dup with
    | _ -> false
    | exception C.Duplicate_binder dup -> C.equal_duplicate dup dup_expected);
  let sibling_shared =
    seq
      (loop ~upto:3 lc_s (set_at dst (iter lc_s) (c 1.)))
      (loop ~from_:1 ~upto:2 lc_s (set_at src (iter lc_s) (c 2.)))
  in
  let eq_bounds (lo1, hi1) (lo2, hi2) = lo1 = lo2 && hi1 = hi2 in
  p "sibling loops sharing a symbol still extract as separate single-binder chains"
    (match LL.affine_accesses sibling_shared with
    | [ a1; a2 ] ->
        C.length a1.a_loops = 1
        && C.length a2.a_loops = 1
        && Option.equal eq_bounds (C.find a1.a_loops lc_s) (Some (0, 3))
        && Option.equal eq_bounds (C.find a2.a_loops lc_s) (Some (1, 2))
    | _ -> false)
