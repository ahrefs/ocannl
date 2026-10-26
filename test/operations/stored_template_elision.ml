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
   a constant, on a [uint4x32] node or a float one. Folds are the consumer's simplify's to make
   where the template replays (threefry operands included, gh-ocannl-1498), at the slot it replays
   into: an index and a constant convert to [uint4x32] through different builtins.

   Consumer (gh-ocannl-1469): the same reading on the CONSUMER's side. B's own [Where] with a
   decided condition around a merge-tainted inherited read compiles and computes the live arm --
   decided by a constant, by B's own loop extent, or (for a virtual consumer stored as a template)
   by a reduction loop the template keeps -- while a template's write-index loop decides nothing,
   and a live or undecided arm is still refused.

   Capture refusal (gh-ocannl-1514): a VIRTUAL consumer whose dead arm only its own write-index loop
   decides. The walk capturing its template cannot decide the arm, so it leaves the merge-tainted
   read un-inlined and refuses that store ([154:capture-reads-merge]): the consumer materializes,
   and the emit walk, holding the loop's extent, skips the arm. Only the store whose own code holds
   the read is refused -- not a sibling's in a shared loop, not an earlier or later candidate of the
   same loop -- and a consumer that cannot materialize (declared virtual) or whose arm is live where
   it runs is still refused with the merge-splice message.

   The same capture refusal for the two other inherited reads that refuse a routine where they
   execute (gh-ocannl-1551): a node an earlier routine placed as routine-local scratch
   ([155:capture-reads-local]; [Local] is a backend-finalization decision, so routine A is linked),
   and a computation that cannot be instantiated at the read's indices, a fixed-cell producer read
   at a loop-varying index ([156:capture-read-uninlinable]). Each runs the whole capture family, and
   each refusal keeps its own message where the read is live or the consumer is declared virtual.

   Routine-local scratch on the routine's own terms (gh-ocannl-1620): B's reads of an inherited
   [Local] node are judged on the code its emit walk evaluates. A read only in a dead arm (by a
   constant or by B's loop extent) decides nothing; one B's own earlier writes cover executes on B's
   scratch, after a dead read too -- but a virtual consumer holding it materializes ([155], captured
   at its loop or stored per statement), since a template would replay it where the scratch is not;
   a live, undecided, partial, guarded, read-before-write or in-place read is refused with the
   routine-local message. Lineages with and without a stored template of [v]. *)

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

(* gh-ocannl-1514: the capture refusal. Each case builds routine B twice from [build ~sfx ~v]: on
   the tainted lineage, and in a fresh one where [v] is a materialized node holding junk the live
   arm never reads -- the reading the refusal must agree with. [build] returns B's code, its seeds,
   the nodes read back with their expected values, and the placement claims about B. *)
let junk = Array.init dim ~f:(fun i -> 9000. +. Float.of_int i)
let x1_vals = Array.map x_vals ~f:(fun x -> x +. 1.)

(* A lineage whose routine A left [v] such that a later routine's read of it refuses that routine
   where the read executes: [lineage ~label ~tag] builds it, a capture meeting the read refuses its
   store with [refusal], and a live read is refused with a message containing [message]. *)
type lineage = {
  lineage : label:string -> tag:string -> LL.optimize_ctx * Tn.t;
  prelude : Tn.t -> LL.t option;
      (** What routine B does to [v] ahead of the consumer, if anything. *)
  refusal : Tn.provenance;
  message : string;
}

let with_prelude lin v llc =
  Option.value_map (lin.prelude v) ~default:llc ~f:(fun pre -> seq pre llc)

let merge_lineage =
  {
    lineage = tainted_lineage;
    prelude = (fun _ -> None);
    refusal = Tn.Site "154:capture-reads-merge";
    message = "reads a merge buffer";
  }

(* gh-ocannl-1551: routine A places [v] as routine-local scratch AFTER storing its computation: [v]
   is written cell by cell, each component stored, and A's own loop-varying read cannot instantiate
   a fixed-cell component ([13:call-site-index-mismatch]), which commits [v] [Never_virtual] there;
   backend finalization resolves it to [Local], so routine A is linked.

   gh-ocannl-1620: with [~stored:false], routine A writes [v] under a guard instead, a computation
   the virtualizer refuses to store ([142:guarded-computation]): the same [Local] scratch with no
   computation in the table.

   In the capture family routine B only READS [v]: [decide_placements] judges the reads the emitted
   code evaluates (gh-ocannl-1620), so a read in an arm only the emit walk decides dead reaches the
   capture walk, whose hit is the refusal under test. *)
let local_scratch ?(stored = true) ~label ~tag () =
  let a = mk (tag ^ "a") and outa = mk (tag ^ "outa") and v = mk (tag ^ "v") in
  List.iter [ a; outa ] ~f:materialize;
  let ctx = LL.empty_optimize_ctx () in
  let llc_a =
    let s = sym () and s2 = sym () in
    if stored then
      List.reduce_exn ~f:seq
        (List.init dim ~f:(fun k -> set v [| fixed k |] (add (get a [| fixed k |]) (c 1000.)))
        @ [ loop_n s dim (set outa [| iter s |] (get v [| iter s |])) ])
    else
      seq
        (loop_n s dim
           (if_ (lt (embed s) (ic dim)) (set v [| iter s |] (add (get a [| iter s |]) (c 1000.)))))
        (loop_n s2 dim (set outa [| iter s2 |] (get v [| iter s2 |])))
  in
  let name = "sts_" ^ tag ^ "_a" in
  let o_a = optimize_in ctx ~name llc_a in
  let linked = link_finalized ~ctx:(Context.auto ()) ~placements:[ v ] ~name o_a in
  let _ctx = run_linked linked ~seed:[ (a, Array.init dim ~f:(fun i -> 60. +. Float.of_int i)) ] in
  p
    (label ^ ": routine A placed v as routine-local scratch, "
    ^ if stored then "its computation in the table" else "with no computation in the table")
    (known_local o_a v && Bool.equal stored (Hashtbl.mem ctx.LL.computations v));
  (ctx, v)

let local_lineage =
  {
    lineage = (fun ~label ~tag -> local_scratch ~label ~tag ());
    prelude = (fun _ -> None);
    refusal = Tn.Site "155:capture-reads-local";
    message = "routine-local scratch";
  }

(* gh-ocannl-1551: routine A defers a fixed-cell producer [v[0] := ell[0] + 100], which cannot be
   instantiated at a loop-varying read ([13:call-site-index-mismatch]); the deferring routine
   dropped its setter, so no materialized read can serve it either. *)
let uninlinable_lineage =
  let lineage ~label ~tag =
    let ell = mk (tag ^ "ell") and v = mk (tag ^ "v") in
    materialize ell;
    let ctx = LL.empty_optimize_ctx () in
    let o_a =
      optimize_in ctx
        ~name:("sts_" ^ tag ^ "_a")
        (set v [| fixed 0 |] (add (get ell [| fixed 0 |]) (c 100.)))
    in
    p (label ^ ": routine A defers the fixed-cell v") (known_virtual o_a v);
    (ctx, v)
  in
  {
    lineage;
    prelude = (fun _ -> None);
    refusal = Tn.Site "156:capture-read-uninlinable";
    message = "could not be inlined at a read site";
  }

let refused_by capture o tn =
  Option.equal Tn.equal_provenance (rejection_code o tn) (Some capture.refusal)

let case_capture ~lineage:lin ~label ~tag build =
  let ctx, v = lin.lineage ~label ~tag in
  let llc, seed, reads, placement_claims = build ~refused:(refused_by lin) ~sfx:"" ~v in
  let llc = with_prelude lin v llc in
  match consume ctx ~name:("sts_" ^ tag ^ "_b") llc with
  | Error msg -> p (label ^ ": routine B compiles (refused: " ^ msg ^ ")") false
  | Ok o_b ->
      p (label ^ ": routine B compiles") true;
      p
        (label ^ ": routine B declares no merge buffer and does not read v")
        (Option.is_none o_b.LL.merge_node && not (Set.mem (inputs o_b) v));
      List.iter placement_claims ~f:(fun (claim, holds) -> p (label ^ ": " ^ claim) (holds o_b));
      let got = execute ~name:("sts_" ^ tag ^ "_b") o_b ~seed ~read:(List.map reads ~f:fst) in
      List.iter2_exn got reads ~f:(fun got (tn, expected) ->
          p_all2
            (label ^ ": " ^ Tn.debug_name tn ^ " executes to the live arm's values")
            got expected ~f:Float.equal);
      let v_m = mk (tag ^ "vm") in
      materialize v_m;
      let llc_m, seed_m, reads_m, _ = build ~refused:(refused_by lin) ~sfx:"m" ~v:v_m in
      let llc_m = with_prelude lin v_m llc_m in
      let o_m = optimize ~name:("sts_" ^ tag ^ "_m") llc_m in
      let got_m =
        execute
          ~name:("sts_" ^ tag ^ "_m")
          o_m ~seed:((v_m, junk) :: seed_m) ~read:(List.map reads_m ~f:fst)
      in
      p_all2
        (label ^ ": the deferred reading agrees with the materialized one")
        (Array.concat got) (Array.concat got_m) ~f:Float.equal

(* [w[i] := shape i x[i] v[i]] for [i < 4], then [out[i] := w[i]]; [w] is a virtual candidate. *)
let own_loop_consumer ~tag shape ~refused ~sfx ~v =
  let x = mk (tag ^ "x" ^ sfx) and w = mk (tag ^ "w" ^ sfx) and out = mk (tag ^ "out" ^ sfx) in
  List.iter [ x; out ] ~f:materialize;
  let s = sym () and s2 = sym () in
  let llc =
    seq
      (loop_n s dim (set w [| iter s |] (shape s ~x:(get x [| iter s |]) ~v:(get v [| iter s |]))))
      (loop_n s2 dim (set out [| iter s2 |] (get w [| iter s2 |])))
  in
  ( llc,
    [ (x, x_vals); (out, blank dim) ],
    [ (out, x_vals) ],
    [ ("w is refused by its capture and materializes", fun o -> refused o w) ] )

(* A materialized sibling [o2] of the shared loop holds the read: [w]'s template drops it at
   consumption, so [w] stays virtual. *)
let sibling_consumer ~tag ~refused:_ ~sfx ~v =
  let x = mk (tag ^ "x" ^ sfx) and w = mk (tag ^ "w" ^ sfx) and o2 = mk (tag ^ "o2" ^ sfx) in
  let out = mk (tag ^ "out" ^ sfx) in
  List.iter [ x; o2; out ] ~f:materialize;
  let s = sym () and s2 = sym () in
  let llc =
    seq
      (loop_n s dim
         (seq
            (set w [| iter s |] (add (get x [| iter s |]) (c 1.)))
            (set o2
               [| iter s |]
               (where_ (lt (embed s) (ic dim)) (get x [| iter s |]) (get v [| iter s |])))))
      (loop_n s2 dim (set out [| iter s2 |] (get w [| iter s2 |])))
  in
  ( llc,
    [ (x, x_vals); (o2, blank dim); (out, blank dim) ],
    [ (o2, x_vals); (out, x1_vals) ],
    [ ("w, whose own code does not read v, stays virtual", fun o -> known_virtual o w) ] )

(* Phase-1 ordering around a refused candidate: an earlier stored provider [p] and a later candidate
   [r] reading the refused [w], all three captured at one loop. *)
let ordered_consumers ~tag ~refused ~sfx ~v =
  let x = mk (tag ^ "x" ^ sfx) and pp = mk (tag ^ "p" ^ sfx) and w = mk (tag ^ "w" ^ sfx) in
  let r = mk (tag ^ "r" ^ sfx) and out = mk (tag ^ "out" ^ sfx) in
  List.iter [ x; out ] ~f:materialize;
  let s = sym () and s2 = sym () in
  let llc =
    seq
      (loop_n s dim
         (List.reduce_exn ~f:seq
            [
              set pp [| iter s |] (mul (get x [| iter s |]) (c 2.));
              set w
                [| iter s |]
                (where_ (lt (embed s) (ic dim)) (get pp [| iter s |]) (get v [| iter s |]));
              set r [| iter s |] (add (get w [| iter s |]) (c 1.));
            ]))
      (loop_n s2 dim (set out [| iter s2 |] (get r [| iter s2 |])))
  in
  ( llc,
    [ (x, x_vals); (out, blank dim) ],
    [ (out, Array.map x_vals ~f:(fun x -> (2. *. x) +. 1.)) ],
    [
      ("the earlier provider p stays virtual", fun o -> known_virtual o pp);
      ("w is refused by its capture and materializes", fun o -> refused o w);
      ("the later reader r stays virtual", fun o -> known_virtual o r);
    ] )

(* A candidate with an earlier stored component: [zero w] is stored on its own before the loop's
   capture refuses [w]; the materialized [w] keeps the whole computation. *)
let zeroed_consumer ~tag ~refused ~sfx ~v =
  let x2 = mk ~dims:[| dim; kdim |] (tag ^ "x2" ^ sfx) and w = mk (tag ^ "w" ^ sfx) in
  let out = mk (tag ^ "out" ^ sfx) in
  List.iter [ x2; out ] ~f:materialize;
  let s = sym () and k = sym () and s2 = sym () in
  let llc =
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
                      (lt (embed s) (ic dim))
                      (get x2 [| iter s; iter k |])
                      (get v [| iter s |])))));
        loop_n s2 dim (set out [| iter s2 |] (get w [| iter s2 |]));
      ]
  in
  let x2_vals =
    Array.init (dim * kdim) ~f:(fun j ->
        (100. *. Float.of_int (j / kdim)) +. Float.of_int (j % kdim) +. 1.)
  in
  let expected =
    Array.init dim ~f:(fun i ->
        Array.fold (Array.sub x2_vals ~pos:(i * kdim) ~len:kdim) ~init:0. ~f:( +. ))
  in
  ( llc,
    [ (x2, x2_vals); (out, blank dim) ],
    [ (out, expected) ],
    [ ("w is refused by its capture and materializes", fun o -> refused o w) ] )

(* Still refused: a consumer that cannot materialize (declared virtual), and one whose arm is live
   where it runs (the emit walk raises). *)
let case_capture_refused ~lineage:lin ~label ~tag ~declare_virtual shape =
  let ctx, v = lin.lineage ~label ~tag in
  let x = mk (tag ^ "x") and w = mk (tag ^ "w") and out = mk (tag ^ "out") in
  List.iter [ x; out ] ~f:materialize;
  if declare_virtual then virtualize w;
  let s = sym () and s2 = sym () in
  let llc =
    seq
      (loop_n s dim (set w [| iter s |] (shape s ~x:(get x [| iter s |]) ~v:(get v [| iter s |]))))
      (loop_n s2 dim (set out [| iter s2 |] (get w [| iter s2 |])))
  in
  match consume ctx ~name:("sts_" ^ tag ^ "_b") (with_prelude lin v llc) with
  | Ok _ -> p (label ^ ": routine B is refused (it compiled)") false
  | Error msg ->
      p
        (label ^ ": routine B is refused with the read's own message")
        (String.is_substring msg ~substring:lin.message)

(* The capture family over one lineage; [kind] prefixes labels and tags. *)
let capture_family ~kind ~lineage =
  let label l = if String.is_empty kind then l else kind ^ ": " ^ l and tag t = kind ^ t in
  case_capture ~lineage
    ~label:(label "virtual consumer where (i < 4) x v")
    ~tag:(tag "vc4")
    (own_loop_consumer ~tag:(tag "vc4") (fun s ~x ~v -> where_ (lt (embed s) (ic dim)) x v));
  case_capture ~lineage
    ~label:(label "virtual consumer where (i < 0) v x")
    ~tag:(tag "vc0")
    (own_loop_consumer ~tag:(tag "vc0") (fun s ~x ~v -> where_ (lt (embed s) (ic 0)) v x));
  case_capture ~lineage ~label:(label "sibling holds the read") ~tag:(tag "cs")
    (sibling_consumer ~tag:(tag "cs"));
  case_capture ~lineage ~label:(label "ordered candidates") ~tag:(tag "co")
    (ordered_consumers ~tag:(tag "co"));
  case_capture ~lineage ~label:(label "earlier zero component") ~tag:(tag "cz")
    (zeroed_consumer ~tag:(tag "cz"));
  case_capture_refused ~lineage
    ~label:(label "virtual consumer where (i < 2) x v (live where it runs)")
    ~tag:(tag "vcu") ~declare_virtual:false (fun s ~x ~v -> where_ (lt (embed s) (ic 2)) x v);
  case_capture_refused ~lineage ~label:(label "declared-virtual consumer where (i < 4) x v")
    ~tag:(tag "vcd") ~declare_virtual:true (fun s ~x ~v -> where_ (lt (embed s) (ic dim)) x v)

(* gh-ocannl-1620: routine B's own reads of the inherited scratch [v], judged on the code B's
   emitted walk evaluates. An outcome distinguishes an actionable refusal from a crash. *)
type local_outcome = Compiled of LL.optimized | Refused of string | Crashed of string

let consume_local ctx ~name llc =
  try Compiled (optimize_in ctx ~name llc) with
  | Utils.User_error msg -> Refused msg
  | exn -> Crashed (Exn.to_string exn)

let local_vals = Array.init dim ~f:(fun i -> 500. +. Float.of_int i)

(* [build ~sfx ~v] returns B's code, its seeds, the nodes read back with their expected values, and
   placement claims; the materialized reading builds it again over a materialized [v] holding junk,
   which no evaluated read may observe. *)
let case_local_accepted ?stored ~label ~tag build =
  let ctx, v = local_scratch ?stored ~label ~tag () in
  let llc, seed, reads, claims = build ~sfx:"" ~v in
  match consume_local ctx ~name:("sts_" ^ tag ^ "_b") llc with
  | Refused msg | Crashed msg -> p (label ^ ": routine B compiles (refused: " ^ msg ^ ")") false
  | Compiled o_b ->
      p
        (label ^ ": routine B compiles, v stays routine-local scratch and is no input of B")
        (known_local o_b v && not (Set.mem (inputs o_b) v));
      List.iter claims ~f:(fun (claim, holds) -> p (label ^ ": " ^ claim) (holds o_b));
      let got = execute ~name:("sts_" ^ tag ^ "_b") o_b ~seed ~read:(List.map reads ~f:fst) in
      List.iter2_exn got reads ~f:(fun got (tn, expected) ->
          p_all2
            (label ^ ": " ^ Tn.debug_name tn ^ " executes to its expected values")
            got expected ~f:Float.equal);
      let v_m = mk (tag ^ "vm") in
      materialize v_m;
      let llc_m, seed_m, reads_m, _ = build ~sfx:"m" ~v:v_m in
      let o_m = optimize ~name:("sts_" ^ tag ^ "_m") llc_m in
      let got_m =
        execute
          ~name:("sts_" ^ tag ^ "_m")
          o_m ~seed:((v_m, junk) :: seed_m) ~read:(List.map reads_m ~f:fst)
      in
      p_all2
        (label ^ ": the reading agrees with the materialized one")
        (Array.concat got) (Array.concat got_m) ~f:Float.equal

let case_local_refused ?stored ~label ~tag llc_of =
  let ctx, v = local_scratch ?stored ~label ~tag () in
  match consume_local ctx ~name:("sts_" ^ tag ^ "_b") (llc_of ~v) with
  | Compiled _ -> p (label ^ ": routine B is refused (it compiled)") false
  | Crashed msg ->
      p (label ^ ": routine B is refused with an actionable message (crashed: " ^ msg ^ ")") false
  | Refused msg ->
      p
        (label ^ ": routine B is refused, naming v routine-local scratch")
        (String.is_substring msg ~substring:"routine-local scratch")

(* Item 1: [out[i] := shape i x[i] v[i]], B only reading [v]. *)
let dead_arm_reader ~tag shape ~sfx ~v =
  let x = mk (tag ^ "x" ^ sfx) and out = mk (tag ^ "out" ^ sfx) in
  List.iter [ x; out ] ~f:materialize;
  let s = sym () in
  ( loop_n s dim (set out [| iter s |] (shape s ~x:(get x [| iter s |]) ~v:(get v [| iter s |]))),
    [ (x, x_vals); (out, blank dim) ],
    [ (out, x_vals) ],
    [] )

let fill_v v =
  let s = sym () in
  loop_n s dim (set v [| iter s |] (ramp 500. s))

(* Item 2: B writes every cell of [v], then [out[i] := v[i] + x[i]] reads it. *)
let covered_reader ~tag ~sfx ~v =
  let x = mk (tag ^ "x" ^ sfx) and out = mk (tag ^ "out" ^ sfx) in
  List.iter [ x; out ] ~f:materialize;
  let s = sym () in
  ( seq (fill_v v)
      (loop_n s dim (set out [| iter s |] (add (get v [| iter s |]) (get x [| iter s |])))),
    [ (x, x_vals); (out, blank dim) ],
    [ (out, Array.map2_exn local_vals x_vals ~f:( +. )) ],
    [] )

(* A read in a dead arm ahead of B's write, which no coverage verdict may count. *)
let dead_then_covered_reader ~tag ~sfx ~v =
  let x = mk (tag ^ "x" ^ sfx) and o2 = mk (tag ^ "o2" ^ sfx) and out = mk (tag ^ "out" ^ sfx) in
  List.iter [ x; o2; out ] ~f:materialize;
  let s = sym () and s2 = sym () in
  ( List.reduce_exn ~f:seq
      [
        loop_n s dim (set o2 [| iter s |] (where_ (ic 1) (get x [| iter s |]) (get v [| iter s |])));
        fill_v v;
        loop_n s2 dim (set out [| iter s2 |] (add (get v [| iter s2 |]) (get x [| iter s2 |])));
      ],
    [ (x, x_vals); (o2, blank dim); (out, blank dim) ],
    [ (o2, x_vals); (out, Array.map2_exn local_vals x_vals ~f:( +. )) ],
    [] )

let refused_local o tn =
  Option.equal Tn.equal_provenance (rejection_code o tn) (Some (Tn.Site "155:capture-reads-local"))

(* A virtual candidate [w] whose own code reads the covered [v], captured at its loop: the read
   executes here, but [w]'s template would carry it to every routine that replays it. *)
let covered_capture_reader ?(declare_virtual = false) ~tag ~sfx ~v () =
  let x = mk (tag ^ "x" ^ sfx) and w = mk (tag ^ "w" ^ sfx) and out = mk (tag ^ "out" ^ sfx) in
  List.iter [ x; out ] ~f:materialize;
  if declare_virtual then virtualize w;
  let s = sym () and s2 = sym () in
  ( List.reduce_exn ~f:seq
      [
        fill_v v;
        loop_n s dim (set w [| iter s |] (add (get v [| iter s |]) (get x [| iter s |])));
        loop_n s2 dim (set out [| iter s2 |] (get w [| iter s2 |]));
      ],
    [ (x, x_vals); (out, blank dim) ],
    [ (out, Array.map2_exn local_vals x_vals ~f:( +. )) ],
    [ ("w is refused by its capture and materializes", fun o -> refused_local o w) ] )

(* The same through the per-statement store, outside any loop capture. *)
let covered_statement_reader ~tag ~sfx ~v =
  let w = mk (tag ^ "w" ^ sfx) and out = mk (tag ^ "out" ^ sfx) in
  materialize out;
  ( List.reduce_exn ~f:seq
      [
        fill_v v;
        set w [| fixed 0 |] (add (get v [| fixed 1 |]) (c 0.5));
        set out [| fixed 0 |] (get w [| fixed 0 |]);
      ],
    [ (out, blank dim) ],
    [ (out, Array.init dim ~f:(fun i -> if i = 0 then 501.5 else sentinel)) ],
    [ ("w is refused by its per-statement store and materializes", fun o -> refused_local o w) ] )

let case_local_reads () =
  let dead_const _ ~x ~v = where_ (ic 1) x v
  and dead_extent s ~x ~v = where_ (lt (embed s) (ic dim)) x v in
  case_local_accepted ~label:"local, read-only: dead by a constant" ~tag:"lr1"
    (dead_arm_reader ~tag:"lr1" dead_const);
  case_local_accepted ~stored:false ~label:"local, read-only, no template: dead by a constant"
    ~tag:"lr1n"
    (dead_arm_reader ~tag:"lr1n" dead_const);
  case_local_accepted ~label:"local, read-only: dead by B's loop extent" ~tag:"lr2"
    (dead_arm_reader ~tag:"lr2" dead_extent);
  let reads shape ~v =
    let llc, _, _, _ = dead_arm_reader ~tag:"lrx" shape ~sfx:"" ~v in
    llc
  in
  case_local_refused ~label:"local, read-only: live arm" ~tag:"lr3"
    (reads (fun _ ~x ~v -> where_ (ic 0) x v));
  case_local_refused ~stored:false ~label:"local, read-only, no template: live arm" ~tag:"lr3n"
    (reads (fun _ ~x ~v -> where_ (ic 0) x v));
  case_local_refused ~label:"local, read-only: undecided arm" ~tag:"lr4"
    (reads (fun s ~x ~v -> where_ (lt (embed s) (ic 2)) x v));
  case_local_accepted ~label:"local, covered: a materialized consumer" ~tag:"lc1"
    (covered_reader ~tag:"lc1");
  case_local_accepted ~stored:false ~label:"local, covered, no template: a materialized consumer"
    ~tag:"lc1n" (covered_reader ~tag:"lc1n");
  case_local_accepted ~label:"local, covered after a dead read" ~tag:"lc2"
    (dead_then_covered_reader ~tag:"lc2");
  case_local_accepted ~label:"local, covered: a captured virtual consumer" ~tag:"lc3"
    (fun ~sfx ~v -> covered_capture_reader ~tag:"lc3" ~sfx ~v ());
  case_local_accepted ~stored:false
    ~label:"local, covered, no template: a captured virtual consumer" ~tag:"lc3n" (fun ~sfx ~v ->
      covered_capture_reader ~tag:"lc3n" ~sfx ~v ());
  case_local_accepted ~label:"local, covered: a per-statement virtual consumer" ~tag:"lc4"
    (covered_statement_reader ~tag:"lc4");
  (* A packed-uniform producer is captured RAW, with no walk to meet its read (local review). *)
  (let label = "local, covered: a raw-captured packed-uniform producer" and tag = "lc6" in
   let ctx, v = local_scratch ~label ~tag () in
   let w = mk (tag ^ "w") and i = sym () in
   let llc =
     seq (fill_v v)
       (loop_n i 1
          (set_from_vec ~length:4 ~vec_unop:Ir.Ops.Uint4x32_to_prec_uniform w
             [| aff [ (4, i) ] 0 |]
             (get v [| iter i |], Ir.Ops.single)))
   in
   match consume_local ctx ~name:("sts_" ^ tag ^ "_b") llc with
   | Refused msg | Crashed msg -> p (label ^ ": routine B compiles (refused: " ^ msg ^ ")") false
   | Compiled o_b ->
       p (label ^ ": the producer is refused by its capture and materializes") (refused_local o_b w);
       p
         (label ^ ": no template of it reads v")
         (Option.value (template_count ctx w ~f:(read_of v)) ~default:0 = 0));
  case_local_refused ~label:"local, covered: a declared-virtual consumer" ~tag:"lc5" (fun ~v ->
      let llc, _, _, _ = covered_capture_reader ~declare_virtual:true ~tag:"lc5" ~sfx:"" ~v () in
      llc);
  let read_v ~tag ~v =
    let out = mk (tag ^ "out") in
    materialize out;
    let s = sym () in
    loop_n s dim (set out [| iter s |] (get v [| iter s |]))
  in
  case_local_refused ~label:"local: a partial write" ~tag:"lu1" (fun ~v ->
      seq (set v [| fixed 0 |] (c 7.)) (read_v ~tag:"lu1" ~v));
  case_local_refused ~stored:false ~label:"local, no template: a partial write" ~tag:"lu1n"
    (fun ~v -> seq (set v [| fixed 0 |] (c 7.)) (read_v ~tag:"lu1n" ~v));
  case_local_refused ~label:"local: a partial write, read at another cell" ~tag:"lu1c" (fun ~v ->
      let out = mk "lu1cout" in
      materialize out;
      seq (set v [| fixed 0 |] (c 7.)) (set out [| fixed 0 |] (get v [| fixed 1 |])));
  case_local_refused ~label:"local: read before write" ~tag:"lu2" (fun ~v ->
      seq (read_v ~tag:"lu2" ~v) (fill_v v));
  case_local_refused ~label:"local: an accumulation with no initialization" ~tag:"lu3" (fun ~v ->
      let x = mk "lu3x" in
      materialize x;
      let s = sym () in
      seq
        (loop_n s dim (set v [| iter s |] (add (get v [| iter s |]) (get x [| iter s |]))))
        (read_v ~tag:"lu3" ~v));
  case_local_refused ~label:"local: a guarded write, then an unguarded read" ~tag:"lu4" (fun ~v ->
      let s = sym () in
      seq
        (loop_n s dim (if_ (lt (embed s) (ic 2)) (set v [| iter s |] (ramp 500. s))))
        (read_v ~tag:"lu4" ~v))

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
      where_ (lt (embed s) (ic 2)) x v);
  case_local_reads ();
  capture_family ~kind:"" ~lineage:merge_lineage;
  capture_family ~kind:"local" ~lineage:local_lineage;
  capture_family ~kind:"uninlinable" ~lineage:uninlinable_lineage
