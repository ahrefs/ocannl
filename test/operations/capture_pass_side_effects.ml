(* gh-ocannl-1551, item 2: what a capture walk (phase 1 of a shared [For_loop]'s candidate in
   [Low_level.virtual_llc]) does to code its template drops, executed against the reading where it
   cannot matter.

   Duplicate store: a node stored per statement INSIDE a capture walk -- possible only when every
   enclosing loop its index omits has width 1, or the store is refused as [147], and when its
   right-hand side reads no loop-varying operand, or it is refused as [9] -- was stored once per
   phase-1 walk and again by the emit walk, and [check_and_store_virtual] prepends components: a
   non-idempotent accumulation replayed once per walk. [zero y; for i<1 { w[i] := x[i] + 1; for j<4
   { y[j] := y[j] + k[0] } }]: [y] must hold exactly one accumulating component and execute to
   [k[0]], with one and three candidates sharing the width-1 loop, and with a later candidate
   reading [y] in the same body (the capture walk needs [y]'s computation while it walks, so the
   walk's own store of it cannot simply be skipped).

   Dead sibling arm: the capture walk holds the CANDIDATE's write-index loops opaque for every
   setter, so it walked a materialized sibling's arm that the emit walk decides dead -- inlining
   there committed a local producer it could not serve [Never_virtual] (placement), or registered an
   inherited template's reads as spliced (interface), for a read that never executes. The sibling
   [o2[i] := where (i < 4) x[i] r[i]] shares the loop with the candidate [w].

   Each case also executes against the same program with the node in question materialized. *)

open Base
open Ll_test
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode

let dim = 4
let mk = node_factory ~first_id:15510 ~dims:[| dim |] ()
let x_vals = Array.init dim ~f:(fun i -> 40. +. Float.of_int i)
let k_val = 7.

let inputs (o : LL.optimized) =
  let (ins, _), _ = LL.input_and_output_nodes o in
  ins

(* The [Set]s of [tn] across every component the lineage stored for it. *)
let stored_sets (ctx : LL.optimize_ctx) tn =
  Option.value_map (Hashtbl.find ctx.LL.computations tn) ~default:0 ~f:(fun comps ->
      List.sum
        (module Int)
        comps
        ~f:(fun (_, body) ->
          count_stmt ~f:(function LL.Set { tn = t; _ } -> Tn.equal t tn | _ -> false) body))

(* [zero y; for i<1 { w_1[i] := x[i] + 1; ...; w_n[i] := x[i] + n; for j<4 { y[j] += k[0] }; reader
   }; out_y[j] := y[j]; out_k[0] := w_k[0]], where the optional [reader] is a later candidate [r[i]
   := y[i] + 1] of the same loop. *)
let case_duplicate ~label ~tag ~candidates ~with_reader =
  let build ~materialize_y =
    let x = mk ~dims:[| 1 |] (tag ^ "x") and k0 = mk ~dims:[| 1 |] (tag ^ "k") in
    let y = mk (tag ^ "y") in
    let out_y = mk (tag ^ "outy") in
    let ws = List.init candidates ~f:(fun k -> mk ~dims:[| 1 |] (tag ^ "w" ^ Int.to_string k)) in
    let outs =
      List.init candidates ~f:(fun k -> mk ~dims:[| 1 |] (tag ^ "outw" ^ Int.to_string k))
    in
    let r = mk ~dims:[| 1 |] (tag ^ "r") and out_r = mk ~dims:[| 1 |] (tag ^ "outr") in
    List.iter ([ x; k0; out_y; out_r ] @ outs) ~f:materialize;
    if materialize_y then materialize y;
    let i = sym () and j = sym () and s2 = sym () in
    let setters =
      List.mapi ws ~f:(fun k w ->
          set w [| iter i |] (add (get x [| iter i |]) (c (Float.of_int (k + 1)))))
    in
    let acc = loop_n j dim (set y [| iter j |] (add (get y [| iter j |]) (get k0 [| fixed 0 |]))) in
    let reader =
      if with_reader then [ set r [| iter i |] (add (get y [| iter i |]) (c 1.)) ] else []
    in
    let reads =
      List.map2_exn ws outs ~f:(fun w out -> set out [| fixed 0 |] (get w [| fixed 0 |]))
      @ if with_reader then [ set out_r [| fixed 0 |] (get r [| fixed 0 |]) ] else []
    in
    let llc =
      List.reduce_exn ~f:seq
        ([
           zero y;
           loop ~upto:0 i (List.reduce_exn ~f:seq (setters @ [ acc ] @ reader));
           loop_n s2 dim (set out_y [| iter s2 |] (get y [| iter s2 |]));
         ]
        @ reads)
    in
    let seed =
      [ (x, [| 5. |]); (k0, [| k_val |]); (out_y, blank dim); (out_r, blank 1) ]
      @ List.map outs ~f:(fun o -> (o, blank 1))
    in
    let read = (out_y :: outs) @ if with_reader then [ out_r ] else [] in
    (llc, y, seed, read)
  in
  let ctx = LL.empty_optimize_ctx () in
  let llc, y, seed, read = build ~materialize_y:false in
  match
    try Ok (optimize_in ctx ~name:("cpse_" ^ tag) llc) with Utils.User_error msg -> Error msg
  with
  | Error msg -> p (label ^ ": the routine compiles (refused: " ^ msg ^ ")") false
  | Ok o ->
      p (label ^ ": y is deferred") (known_virtual o y);
      p (label ^ ": y holds exactly one accumulating component") (stored_sets ctx y = 1);
      let got = execute ~name:("cpse_" ^ tag) o ~seed ~read in
      p_all
        (label ^ ": y executes to k, accumulated once")
        (Array.to_list (List.hd_exn got))
        ~f:(Float.equal k_val);
      let llc_m, _, seed_m, read_m = build ~materialize_y:true in
      let got_m =
        execute
          ~name:("cpse_" ^ tag ^ "_m")
          (optimize ~name:("cpse_" ^ tag ^ "_m") llc_m)
          ~seed:seed_m ~read:read_m
      in
      p_all2
        (label ^ ": the deferred reading agrees with the materialized one")
        (Array.concat got) (Array.concat got_m) ~f:Float.equal

(* A local producer written cell by cell, [r[0] := ell[0] + 100; r[1] := ell[1] + 100] -- covered,
   so the raw analysis demands nothing of it, but its fixed-cell components cannot be served at a
   loop-varying read ([13:call-site-index-mismatch]) -- read only in the sibling's dead arm. *)
let case_dead_arm_placement () =
  let label = "dead sibling arm, local producer" and tag = "dp" in
  let n = 2 in
  let mk2 = node_factory ~first_id:15600 ~dims:[| n |] () in
  let x_vals = Array.sub x_vals ~pos:0 ~len:n in
  let build ~sfx ~materialize_r =
    let ell = mk2 (tag ^ "ell" ^ sfx) and r = mk2 (tag ^ "r" ^ sfx) and x = mk2 (tag ^ "x" ^ sfx) in
    let w = mk2 (tag ^ "w" ^ sfx) and o2 = mk2 (tag ^ "o2" ^ sfx) in
    let out = mk2 (tag ^ "out" ^ sfx) in
    List.iter [ ell; x; o2; out ] ~f:materialize;
    if materialize_r then materialize r;
    let s = sym () and s2 = sym () in
    let llc =
      List.reduce_exn ~f:seq
        [
          set r [| fixed 0 |] (add (get ell [| fixed 0 |]) (c 100.));
          set r [| fixed 1 |] (add (get ell [| fixed 1 |]) (c 100.));
          loop_n s n
            (seq
               (set w [| iter s |] (add (get x [| iter s |]) (c 1.)))
               (set o2
                  [| iter s |]
                  (where_ (lt (embed s) (ic n)) (get x [| iter s |]) (get r [| iter s |]))));
          loop_n s2 n (set out [| iter s2 |] (get w [| iter s2 |]));
        ]
    in
    let seed =
      [
        (ell, Array.init n ~f:(fun i -> 9000. +. Float.of_int i));
        (x, x_vals);
        (o2, blank n);
        (out, blank n);
      ]
    in
    (llc, r, w, seed, [ o2; out ])
  in
  let llc, r, w, seed, read = build ~sfx:"" ~materialize_r:false in
  let o = optimize ~name:("cpse_" ^ tag) llc in
  p (label ^ ": the candidate w stays virtual") (known_virtual o w);
  p
    (label ^ ": r, read only where the sibling's arm is dead, is not materialized for it")
    (not (known_non_virtual o r));
  let got = execute ~name:("cpse_" ^ tag) o ~seed ~read in
  p_all2 (label ^ ": the sibling executes to its live arm") (List.hd_exn got) x_vals ~f:Float.equal;
  let llc_m, _, _, seed_m, read_m = build ~sfx:"m" ~materialize_r:true in
  let got_m =
    execute
      ~name:("cpse_" ^ tag ^ "_m")
      (optimize ~name:("cpse_" ^ tag ^ "_m") llc_m)
      ~seed:seed_m ~read:read_m
  in
  p_all2
    (label ^ ": the reading agrees with the materialized one")
    (Array.concat got) (Array.concat got_m) ~f:Float.equal

(* Routine A defers [v := extra + 100]; routine B writes [extra] under a taken [If], reads it raw,
   and reads [v] only in the sibling's dead arm. [extra] must keep its raw guards-taken verdict: the
   strict splice verdict, which a guarded write does not satisfy, applies only to nodes read inside
   inlined bodies, and no inlined body survives here. *)
let case_dead_arm_splice () =
  let label = "dead sibling arm, inherited template" and tag = "ds" in
  let build ~v_materialized =
    let gflag = mk ~dims:[| 1 |] (tag ^ "gflag") and extra = mk (tag ^ "extra") in
    let v = mk (tag ^ "v") and x = mk (tag ^ "x") and w = mk (tag ^ "w") in
    let o2 = mk (tag ^ "o2") and ew_out = mk (tag ^ "ewout") and out = mk (tag ^ "out") in
    List.iter [ gflag; extra; x; o2; ew_out; out ] ~f:materialize;
    if v_materialized then materialize v;
    let llc_a =
      let s = sym () in
      loop_n s dim (set v [| iter s |] (add (get extra [| iter s |]) (c 100.)))
    in
    let llc_b =
      let sg = sym () and sh = sym () and s = sym () and s2 = sym () in
      List.reduce_exn ~f:seq
        [
          if_ (get gflag [| fixed 0 |]) (loop_n sg dim (set extra [| iter sg |] (ramp 3. sg)));
          loop_n sh dim (set ew_out [| iter sh |] (get extra [| iter sh |]));
          loop_n s dim
            (seq
               (set w [| iter s |] (add (get x [| iter s |]) (c 1.)))
               (set o2
                  [| iter s |]
                  (where_ (lt (embed s) (ic dim)) (get x [| iter s |]) (get v [| iter s |]))));
          loop_n s2 dim (set out [| iter s2 |] (get w [| iter s2 |]));
        ]
    in
    let seed =
      [
        (gflag, [| 1. |]);
        (extra, blank dim);
        (x, x_vals);
        (o2, blank dim);
        (ew_out, blank dim);
        (out, blank dim);
      ]
    in
    (llc_a, llc_b, v, extra, w, seed, [ ew_out; o2; out ])
  in
  let llc_a, llc_b, v, extra, w, seed, read = build ~v_materialized:false in
  let ctx = LL.empty_optimize_ctx () in
  let o_a = optimize_in ctx ~name:("cpse_" ^ tag ^ "_a") llc_a in
  p (label ^ ": routine A defers v") (known_virtual o_a v);
  let o_b = optimize_in ctx ~name:("cpse_" ^ tag ^ "_b") llc_b in
  p (label ^ ": the candidate w stays virtual") (known_virtual o_b w);
  p
    (label ^ ": extra keeps its raw guards-taken verdict (not spliced, not an input)")
    ((not (read_before_write o_b extra))
    && (not (Set.mem o_b.LL.spliced_rbw extra))
    && not (Set.mem (inputs o_b) extra));
  let got = execute ~name:("cpse_" ^ tag ^ "_b") o_b ~seed ~read in
  p_all2
    (label ^ ": the sibling executes to its live arm")
    (List.nth_exn got 1) x_vals ~f:Float.equal;
  let _, llc_bm, v_m, _, _, seed_m, read_m = build ~v_materialized:true in
  let got_m =
    execute
      ~name:("cpse_" ^ tag ^ "_m")
      (optimize ~name:("cpse_" ^ tag ^ "_m") llc_bm)
      ~seed:((v_m, Array.init dim ~f:(fun i -> 9000. +. Float.of_int i)) :: seed_m)
      ~read:read_m
  in
  p_all2
    (label ^ ": the reading agrees with the materialized one")
    (Array.concat got) (Array.concat got_m) ~f:Float.equal

let () =
  case_duplicate ~label:"duplicate store, one candidate" ~tag:"d1" ~candidates:1 ~with_reader:false;
  case_duplicate ~label:"duplicate store, three candidates" ~tag:"d3" ~candidates:3
    ~with_reader:false;
  case_duplicate ~label:"duplicate store, later reader" ~tag:"dr" ~candidates:1 ~with_reader:true;
  case_dead_arm_placement ();
  case_dead_arm_splice ()
