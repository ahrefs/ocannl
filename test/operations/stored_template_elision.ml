(* gh-ocannl-625: the virtualizer stores a deferred computation with its UNEVALUATED operands elided
   ([check_and_store_virtual] applying [Low_level.elide_unevaluated_operands]) -- the arm of a
   [Where] its condition never selects, the operand behind a gate that is decided shut -- so every
   reader of the lineage's computations table sees only reads that execute. Before, the entry-time
   scans of a later routine walked a template as captured, and such a read still counted: a
   merge-buffer read tainted the template (rejecting cross-routine consumption of code that never
   reads the buffer), and an ordinary read registered its node as spliced (re-judging the consumer's
   own raw reads of it under the strict splice verdicts). Both failed safe, loudly or with a
   stricter interface; this pins the exact reading.

   Each case defers [v] in routine A and consumes it in routine B on the same lineage, then runs the
   two routines with [v] declared materialized, which is the reading the deferral must agree with.
   Leaves are discriminating ([10 + i], [100 i + k], ...), outputs are seeded with the sentinel.

   Merge: [v := where 1 (ell + 100) merge(msrc)]. Consuming [v] in B is legal.

   Gate: [v := ell + relu_gate (-1) gx], whose gate is shut, so [gx] is no input of B. (The
   materialized reading still renders the read: the routine's simplify does not fold gates.)

   Provenance: [v := where 0 extra ell], and B writes [extra] under an [If], then reads it
   unguarded. The raw guards-taken verdict counts the guarded write as coverage; the strict splice
   verdict, which [extra] gets once it is marked spliced, does not (a guarded write covers only the
   reads it dominates) -- so [extra] must stay off B's interface.

   Kept loop: [v[i] := sum_k where (k < 2) a[i,k] merge(msrc)]. The reduction loop survives
   instantiation (freshened, same extent), so its extent decides the guard at store time. The
   write-index symbol [i] could not: [instantiate_computations] substitutes the read site's index
   for it, which need not lie in the producer's extent.

   Live code: everything that evaluates stays as captured -- a fixed embedded index is not folded to
   a constant, on a [uint4x32] node or a float one. A template can replay beneath a threefry
   operand, which the consumer's simplify never enters, and there the routine simplify's folds
   change values (an index and a constant convert to [uint4x32] through different builtins).

   Consumer (gh-ocannl-1469): the same reading on the CONSUMER's side. B's own [Where] with a
   decided condition around a merge-tainted inherited read compiles and computes the live arm --
   decided by a constant, by B's own loop extent, or (for a virtual consumer stored as a template)
   by a reduction loop the template keeps -- while a template's write-index loop decides nothing,
   and a live or undecided arm is still refused. *)

open Base
open Ll_test
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode

let dim = 4
let kdim = 2
let mk = node_factory ~first_id:6250 ~dims:[| dim |] ()

let inputs (o : LL.optimized) =
  let (ins, _), _ = LL.input_and_output_nodes o in
  ins

(* How many scalars satisfying [f] the computations the lineage stored for [self] carry; [None] when
   nothing is stored for it. *)
let template_count (ctx : LL.optimize_ctx) self ~f =
  Option.map (Hashtbl.find ctx.LL.computations self) ~f:(fun comps ->
      List.sum (module Int) comps ~f:(fun (_, body) -> count_scalar ~f body))

let merge_read_of src = function LL.Get_merge_buffer (tn, _) -> Tn.equal tn src | _ -> false
let read_of tn = function LL.Get (t, _) -> Tn.equal t tn | _ -> false

(* Routine B on the deferring lineage, or the message it was refused with. *)
let consume ctx ~name llc =
  try Ok (optimize_in ctx ~name llc) with Utils.User_error msg -> Error msg

(* The materialized reading: A and B in a fresh lineage, [v] declared materialized, A run on the
   seeds, then B on the context A left. *)
let materialized_reading ~name ~v ~llc_a ~llc_b ~seed_a ~seed_b ~read =
  materialize v;
  let ctx = LL.empty_optimize_ctx () in
  let o_a = optimize_in ctx ~name:(name ^ "_ma") llc_a in
  let o_b = optimize_in ctx ~name:(name ^ "_mb") llc_b in
  let after_a = run ~name:(name ^ "_ma") o_a ~seed:seed_a in
  execute ~ctx:after_a ~name:(name ^ "_mb") o_b ~seed:seed_b ~read

let case_merge () =
  let build () =
    let msrc = mk "msrc" and ell = mk "ell" and v = mk "v" and out = mk "out" in
    List.iter [ msrc; ell; out ] ~f:materialize;
    let llc_a =
      let s = sym () in
      loop_n s dim
        (set v
           [| iter s |]
           (where_ (ic 1)
              (add (get ell [| iter s |]) (c 100.))
              (LL.Get_merge_buffer (msrc, [| iter s |]))))
    in
    let llc_b =
      let s = sym () in
      loop_n s dim (set out [| iter s |] (get v [| iter s |]))
    in
    (msrc, ell, v, out, llc_a, llc_b)
  in
  let ell_vals = Array.init dim ~f:(fun i -> 10. +. Float.of_int i) in
  let expected = Array.map ell_vals ~f:(fun x -> x +. 100.) in
  let msrc, ell, v, out, llc_a, llc_b = build () in
  let ctx = LL.empty_optimize_ctx () in
  let o_a = optimize_in ctx ~name:"sts_merge_a" llc_a in
  p "merge: routine A defers v" (known_virtual o_a v);
  p "merge: the stored template no longer reads the merge buffer"
    (Option.equal Int.equal (template_count ctx v ~f:(merge_read_of msrc)) (Some 0));
  match consume ctx ~name:"sts_merge_b" llc_b with
  | Error msg -> p ("merge: consuming v in a later routine is legal (refused: " ^ msg ^ ")") false
  | Ok o_b ->
      p "merge: consuming v in a later routine is legal" true;
      p "merge: routine B declares no merge buffer" (Option.is_none o_b.LL.merge_node);
      let got =
        execute ~name:"sts_merge_b" o_b ~seed:[ (ell, ell_vals); (out, blank dim) ] ~read:[ out ]
      in
      p_all2 "merge: executed values are the selected arm's" (List.hd_exn got) expected
        ~f:Float.equal;
      let _, ell_m, v_m, out_m, llc_am, llc_bm = build () in
      let got_m =
        materialized_reading ~name:"sts_merge" ~v:v_m ~llc_a:llc_am ~llc_b:llc_bm
          ~seed_a:[ (ell_m, ell_vals) ]
          ~seed_b:[ (out_m, blank dim) ]
          ~read:[ out_m ]
      in
      p_all2 "merge: the deferred reading agrees with the materialized one" (List.hd_exn got)
        (List.hd_exn got_m) ~f:Float.equal

let case_gate () =
  let build () =
    let gx = mk "gx" and ell = mk "gell" and v = mk "gv" and out = mk "gout" in
    List.iter [ gx; ell; out ] ~f:materialize;
    let llc_a =
      let s = sym () in
      loop_n s dim
        (set v
           [| iter s |]
           (add (get ell [| iter s |]) (binop Ir.Ops.Relu_gate (c (-1.)) (get gx [| iter s |]))))
    in
    let llc_b =
      let s = sym () in
      loop_n s dim (set out [| iter s |] (get v [| iter s |]))
    in
    (gx, ell, v, out, llc_a, llc_b)
  in
  let ell_vals = Array.init dim ~f:(fun i -> 20. +. Float.of_int i) in
  let gx_vals = Array.init dim ~f:(fun i -> 50. +. Float.of_int i) in
  let gx, ell, v, out, llc_a, llc_b = build () in
  let ctx = LL.empty_optimize_ctx () in
  let o_a = optimize_in ctx ~name:"sts_gate_a" llc_a in
  p "gate: routine A defers v" (known_virtual o_a v);
  p "gate: the stored template no longer reads behind the shut gate"
    (Option.equal Int.equal (template_count ctx v ~f:(read_of gx)) (Some 0));
  let o_b = optimize_in ctx ~name:"sts_gate_b" llc_b in
  p "gate: the gated leaf is not an input of routine B" (not (Set.mem (inputs o_b) gx));
  let got =
    execute ~name:"sts_gate_b" o_b ~seed:[ (ell, ell_vals); (out, blank dim) ] ~read:[ out ]
  in
  p_all2 "gate: executed values are the live operand's" (List.hd_exn got) ell_vals ~f:Float.equal;
  let gx_m, ell_m, v_m, out_m, llc_am, llc_bm = build () in
  let got_m =
    materialized_reading ~name:"sts_gate" ~v:v_m ~llc_a:llc_am ~llc_b:llc_bm
      ~seed_a:[ (ell_m, ell_vals); (gx_m, gx_vals) ]
      ~seed_b:[ (out_m, blank dim) ]
      ~read:[ out_m ]
  in
  p_all2 "gate: the deferred reading agrees with the materialized one" (List.hd_exn got)
    (List.hd_exn got_m) ~f:Float.equal

(* Review round 1: a gate is decided at every precision it may be evaluated at -- a homogeneous
   operator evaluates its operands at its result precision, through the backend's [compute_prec].
   [2049] rounds to [2048] in half but not in single, so a half-valued [(2049 = 2048) && gx] is open
   where half arithmetic is native and shut where it is widened; its second operand must stay
   (deciding at the gate's own single annotation would elide it). *)
let case_precision_dependent_gate () =
  let gx = mk "pgx" in
  materialize gx;
  let v = (node_factory ~prec:Ir.Ops.half ~first_id:6400 ~dims:[| dim |] ()) "pgv" in
  let ctx = LL.empty_optimize_ctx () in
  let llc =
    let s = sym () in
    loop_n s dim
      (set v
         [| iter s |]
         (binop Ir.Ops.And (binop Ir.Ops.Cmpeq (c 2049.) (c 2048.)) (get gx [| iter s |])))
  in
  let o = optimize_in ctx ~name:"sts_pgate_a" llc in
  p "precision-dependent gate: routine A defers v" (known_virtual o v);
  p "precision-dependent gate: the operand behind it is kept"
    (Option.equal Int.equal (template_count ctx v ~f:(read_of gx)) (Some 1))

let case_provenance () =
  let build () =
    let gflag = mk ~dims:[| 1 |] "gflag" and extra = mk "extra" and ell = mk "pell" in
    let v = mk "pv" and ew_out = mk "ewout" and out = mk "pout" in
    List.iter [ gflag; extra; ell; ew_out; out ] ~f:materialize;
    let llc_a =
      let s = sym () in
      loop_n s dim
        (set v [| iter s |] (where_ (ic 0) (get extra [| iter s |]) (get ell [| iter s |])))
    in
    let llc_b =
      let sg = sym () and sh = sym () and s = sym () in
      List.reduce_exn ~f:seq
        [
          if_ (get gflag [| fixed 0 |]) (loop_n sg dim (set extra [| iter sg |] (ramp 3. sg)));
          loop_n sh dim (set ew_out [| iter sh |] (get extra [| iter sh |]));
          loop_n s dim (set out [| iter s |] (get v [| iter s |]));
        ]
    in
    (gflag, extra, ell, v, ew_out, out, llc_a, llc_b)
  in
  let ell_vals = Array.init dim ~f:(fun i -> 33. +. Float.of_int i) in
  let gflag, extra, ell, v, ew_out, out, llc_a, llc_b = build () in
  let ctx = LL.empty_optimize_ctx () in
  let o_a = optimize_in ctx ~name:"sts_prov_a" llc_a in
  p "provenance: routine A defers v" (known_virtual o_a v);
  p "provenance: the stored template no longer reads the dead arm"
    (Option.equal Int.equal (template_count ctx v ~f:(read_of extra)) (Some 0));
  let o_b = optimize_in ctx ~name:"sts_prov_b" llc_b in
  p "provenance: extra keeps its raw guards-taken verdict (not spliced, not an input)"
    ((not (read_before_write o_b extra))
    && (not (Set.mem o_b.LL.spliced_rbw extra))
    && not (Set.mem (inputs o_b) extra));
  p "provenance: the live arm's leaf is a spliced input" (Set.mem (inputs o_b) ell);
  let seed_b ~gflag ~extra ~ell ~ew_out ~out =
    [
      (gflag, [| 1. |]); (ell, ell_vals); (extra, blank dim); (ew_out, blank dim); (out, blank dim);
    ]
  in
  let got =
    execute ~name:"sts_prov_b" o_b
      ~seed:(seed_b ~gflag ~extra ~ell ~ew_out ~out)
      ~read:[ ew_out; out ]
  in
  p_all2 "provenance: the read after the taken guard sees the write" (List.nth_exn got 0)
    (Array.init dim ~f:(fun i -> 3. +. Float.of_int i))
    ~f:Float.equal;
  p_all2 "provenance: executed values are the selected arm's" (List.nth_exn got 1) ell_vals
    ~f:Float.equal;
  let gflag_m, extra_m, ell_m, v_m, ew_out_m, out_m, llc_am, llc_bm = build () in
  let got_m =
    materialized_reading ~name:"sts_prov" ~v:v_m ~llc_a:llc_am ~llc_b:llc_bm
      ~seed_a:[ (ell_m, ell_vals) ]
      ~seed_b:(seed_b ~gflag:gflag_m ~extra:extra_m ~ell:ell_m ~ew_out:ew_out_m ~out:out_m)
      ~read:[ ew_out_m; out_m ]
  in
  p_all2 "provenance: the deferred reading agrees with the materialized one" (Array.concat got)
    (Array.concat got_m) ~f:Float.equal

let case_kept_loop () =
  let build () =
    let msrc = mk "kmsrc" and a = mk ~dims:[| dim; kdim |] "ka" and v = mk "kv" in
    let out = mk "kout" in
    List.iter [ msrc; a; out ] ~f:materialize;
    let llc_a =
      let s = sym () and k = sym () in
      seq (zero v)
        (loop_n s dim
           (loop_n k kdim
              (set v
                 [| iter s |]
                 (add
                    (get v [| iter s |])
                    (where_
                       (lt (embed k) (ic kdim))
                       (get a [| iter s; iter k |])
                       (LL.Get_merge_buffer (msrc, [| iter s |])))))))
    in
    let llc_b =
      let s = sym () in
      loop_n s dim (set out [| iter s |] (get v [| iter s |]))
    in
    (msrc, a, v, out, llc_a, llc_b)
  in
  let a_vals =
    Array.init (dim * kdim) ~f:(fun j ->
        (100. *. Float.of_int (j / kdim)) +. Float.of_int (j % kdim) +. 1.)
  in
  let expected =
    Array.init dim ~f:(fun i ->
        Array.fold (Array.sub a_vals ~pos:(i * kdim) ~len:kdim) ~init:0. ~f:( +. ))
  in
  let msrc, a, v, out, llc_a, llc_b = build () in
  let ctx = LL.empty_optimize_ctx () in
  let o_a = optimize_in ctx ~name:"sts_kept_a" llc_a in
  p "kept loop: routine A defers v" (known_virtual o_a v);
  p "kept loop: the reduction extent folded the guard at store time"
    (Option.equal Int.equal (template_count ctx v ~f:(merge_read_of msrc)) (Some 0));
  match consume ctx ~name:"sts_kept_b" llc_b with
  | Error msg ->
      p ("kept loop: consuming v in a later routine is legal (refused: " ^ msg ^ ")") false
  | Ok o_b ->
      p "kept loop: consuming v in a later routine is legal" true;
      let got =
        execute ~name:"sts_kept_b" o_b ~seed:[ (a, a_vals); (out, blank dim) ] ~read:[ out ]
      in
      p_all2 "kept loop: executed values are the reduction's" (List.hd_exn got) expected
        ~f:Float.equal;
      let _, a_m, v_m, out_m, llc_am, llc_bm = build () in
      let got_m =
        materialized_reading ~name:"sts_kept" ~v:v_m ~llc_a:llc_am ~llc_b:llc_bm
          ~seed_a:[ (a_m, a_vals) ]
          ~seed_b:[ (out_m, blank dim) ]
          ~read:[ out_m ]
      in
      p_all2 "kept loop: the deferred reading agrees with the materialized one" (List.hd_exn got)
        (List.hd_exn got_m) ~f:Float.equal

let case_live_code () =
  let fixed_embed = function LL.Embed_index (Ir.Indexing.Fixed_idx _) -> true | _ -> false in
  let stored_embeds prec label =
    let v = (node_factory ~prec ~first_id:6350 ~dims:[| 1 |] ()) label in
    let ctx = LL.empty_optimize_ctx () in
    let o =
      optimize_in ctx ~name:("sts_live_" ^ label) (set v [| fixed 0 |] (embed_idx (fixed 0)))
    in
    p ("live code: routine A defers " ^ label) (known_virtual o v);
    template_count ctx v ~f:fixed_embed
  in
  p "live code: a uint4x32 template keeps its embedded index"
    (Option.equal Int.equal (stored_embeds Ir.Ops.uint4x32 "pctr") (Some 1));
  p "live code: a float template keeps its embedded index"
    (Option.equal Int.equal (stored_embeds Ir.Ops.single "fctr") (Some 1))

(* gh-ocannl-1469, the consumer's half. Routine A defers [v := ell + merge(msrc)], whose template
   genuinely reads the merge buffer, so consuming [v] in a later routine is refused wherever the
   read executes. Routine B wraps the read in a [Where] of its OWN, and when the condition is
   decided [virtual_llc] must not descend into the dead arm: B compiles, declares no merge buffer,
   and computes the live arm. The materialized reading runs B with [v] materialized and holding junk
   the live arm never reads. *)
let x_vals = Array.init dim ~f:(fun i -> 40. +. Float.of_int i)

(* A fresh lineage whose routine A deferred the merge-reading [v]. *)
let tainted_lineage ~label ~tag =
  let msrc = mk (tag ^ "msrc") and ell = mk (tag ^ "ell") and v = mk (tag ^ "v") in
  List.iter [ msrc; ell ] ~f:materialize;
  let ctx = LL.empty_optimize_ctx () in
  let llc_a =
    let s = sym () in
    loop_n s dim
      (set v [| iter s |] (add (get ell [| iter s |]) (LL.Get_merge_buffer (msrc, [| iter s |]))))
  in
  let o_a = optimize_in ctx ~name:("sts_" ^ tag ^ "_a") llc_a in
  p
    (label ^ ": routine A defers v, its template reading the merge buffer")
    (known_virtual o_a v
    && Option.equal Int.equal (template_count ctx v ~f:(merge_read_of msrc)) (Some 1));
  (ctx, v)

let consumer_b ~shape ~x ~v ~out =
  let s = sym () in
  loop_n s dim (set out [| iter s |] (shape s ~x:(get x [| iter s |]) ~v:(get v [| iter s |])))

let case_consumer_decided ~label ~tag shape =
  let ctx, v = tainted_lineage ~label ~tag in
  let x = mk (tag ^ "x") and out = mk (tag ^ "out") in
  List.iter [ x; out ] ~f:materialize;
  match consume ctx ~name:("sts_" ^ tag ^ "_b") (consumer_b ~shape ~x ~v ~out) with
  | Error msg -> p (label ^ ": routine B compiles (refused: " ^ msg ^ ")") false
  | Ok o_b ->
      p (label ^ ": routine B compiles") true;
      p (label ^ ": routine B declares no merge buffer") (Option.is_none o_b.LL.merge_node);
      let got =
        execute
          ~name:("sts_" ^ tag ^ "_b")
          o_b
          ~seed:[ (x, x_vals); (out, blank dim) ]
          ~read:[ out ]
      in
      p_all2
        (label ^ ": executed values are the live arm's")
        (List.hd_exn got) x_vals ~f:Float.equal;
      let v_m = mk (tag ^ "vm") and x_m = mk (tag ^ "xm") and out_m = mk (tag ^ "outm") in
      List.iter [ v_m; x_m; out_m ] ~f:materialize;
      let o_m = optimize ~name:("sts_" ^ tag ^ "_m") (consumer_b ~shape ~x:x_m ~v:v_m ~out:out_m) in
      let got_m =
        execute
          ~name:("sts_" ^ tag ^ "_m")
          o_m
          ~seed:
            [
              (v_m, Array.init dim ~f:(fun i -> 9000. +. Float.of_int i));
              (x_m, x_vals);
              (out_m, blank dim);
            ]
          ~read:[ out_m ]
      in
      p_all2
        (label ^ ": the deferred reading agrees with the materialized one")
        (List.hd_exn got) (List.hd_exn got_m) ~f:Float.equal

(* The rejection still fires wherever the merge-tainted read may execute. *)
let case_consumer_refused ~label ~tag shape =
  let ctx, v = tainted_lineage ~label ~tag in
  let x = mk (tag ^ "x") and out = mk (tag ^ "out") in
  List.iter [ x; out ] ~f:materialize;
  match consume ctx ~name:("sts_" ^ tag ^ "_b") (consumer_b ~shape ~x ~v ~out) with
  | Ok _ -> p (label ^ ": routine B is refused (it compiled)") false
  | Error msg ->
      p
        (label ^ ": routine B is refused for the cross-routine merge splice")
        (String.is_substring msg ~substring:"reads a merge buffer")

(* A virtual consumer is stored as a template, and the walk producing it decides with the extent of
   a reduction loop the template keeps: [w[i] := sum_k where (k < 2) x2[i,k] v[i]]. *)
let case_consumer_kept_loop () =
  let label = "consumer kept loop" and tag = "ck" in
  let ctx, v = tainted_lineage ~label ~tag in
  let build () =
    let x2 = mk ~dims:[| dim; kdim |] (tag ^ "x2") and w = mk (tag ^ "w") in
    let out = mk (tag ^ "out") in
    List.iter [ x2; out ] ~f:materialize;
    let llc ~v =
      let s = sym () and k = sym () and s2 = sym () in
      List.reduce_exn ~f:seq
        [
          zero w;
          loop_n s dim
            (loop_n k kdim
               (set w
                  [| iter s |]
                  (add
                     (get w [| iter s |])
                     (where_
                        (lt (embed k) (ic kdim))
                        (get x2 [| iter s; iter k |])
                        (get v [| iter s |])))));
          loop_n s2 dim (set out [| iter s2 |] (get w [| iter s2 |]));
        ]
    in
    (x2, w, out, llc)
  in
  let x2_vals =
    Array.init (dim * kdim) ~f:(fun j ->
        (100. *. Float.of_int (j / kdim)) +. Float.of_int (j % kdim) +. 1.)
  in
  let expected =
    Array.init dim ~f:(fun i ->
        Array.fold (Array.sub x2_vals ~pos:(i * kdim) ~len:kdim) ~init:0. ~f:( +. ))
  in
  let x2, w, out, llc = build () in
  match consume ctx ~name:"sts_ck_b" (llc ~v) with
  | Error msg -> p (label ^ ": routine B compiles (refused: " ^ msg ^ ")") false
  | Ok o_b ->
      p (label ^ ": routine B compiles, deferring w") (known_virtual o_b w);
      p (label ^ ": routine B declares no merge buffer") (Option.is_none o_b.LL.merge_node);
      let got =
        execute ~name:"sts_ck_b" o_b ~seed:[ (x2, x2_vals); (out, blank dim) ] ~read:[ out ]
      in
      p_all2
        (label ^ ": executed values are the reduction's")
        (List.hd_exn got) expected ~f:Float.equal;
      let x2_m, _, out_m, llc_m = build () in
      let v_m = mk (tag ^ "vm") in
      materialize v_m;
      let o_m = optimize ~name:"sts_ck_m" (llc_m ~v:v_m) in
      let got_m =
        execute ~name:"sts_ck_m" o_m
          ~seed:
            [
              (v_m, Array.init dim ~f:(fun i -> 9000. +. Float.of_int i));
              (x2_m, x2_vals);
              (out_m, blank dim);
            ]
          ~read:[ out_m ]
      in
      p_all2
        (label ^ ": the deferred reading agrees with the materialized one")
        (List.hd_exn got) (List.hd_exn got_m) ~f:Float.equal

(* The walk that produces a template decides with no loop of the template's write index: the
   template replays with the read site's index in its place, which need not lie in that loop's
   extent. [w[i] := where (i < 4) x[i] y[i]] keeps its read of [y]. *)
let case_consumer_write_index_opaque () =
  let x = mk "wox" and y = mk "woy" and w = mk "wow" in
  List.iter [ x; y ] ~f:materialize;
  let ctx = LL.empty_optimize_ctx () in
  let llc =
    let s = sym () in
    loop_n s dim
      (set w
         [| iter s |]
         (where_ (lt (embed s) (ic dim)) (get x [| iter s |]) (get y [| iter s |])))
  in
  let o = optimize_in ctx ~name:"sts_wo" llc in
  p "write index: w is deferred" (known_virtual o w);
  p "write index: its loop decides nothing in the stored template"
    (Option.equal Int.equal (template_count ctx w ~f:(read_of y)) (Some 1))

let () =
  case_merge ();
  case_provenance ();
  case_gate ();
  case_precision_dependent_gate ();
  case_kept_loop ();
  case_live_code ();
  case_consumer_decided ~label:"consumer where 1 x v" ~tag:"c1" (fun _ ~x ~v -> where_ (ic 1) x v);
  case_consumer_decided ~label:"consumer where 0 v x" ~tag:"c0" (fun _ ~x ~v -> where_ (ic 0) v x);
  case_consumer_decided ~label:"consumer where (i < 4) x v" ~tag:"ce" (fun s ~x ~v ->
      where_ (lt (embed s) (ic dim)) x v);
  case_consumer_kept_loop ();
  case_consumer_write_index_opaque ();
  case_consumer_refused ~label:"consumer where 0 x v (v live)" ~tag:"cl" (fun _ ~x ~v ->
      where_ (ic 0) x v);
  case_consumer_refused ~label:"consumer where (i < 2) x v (undecided)" ~tag:"cu" (fun s ~x ~v ->
      where_ (lt (embed s) (ic 2)) x v)
